//! Parquet scan pushdown planning for lazy dataframe plans.

const std = @import("std");
const names_mod = @import("dataframe_names.zig");
const profile_pushdown_mod = @import("dataframe_lazy_pushdown_profile.zig");
const options_mod = @import("dataframe_options.zig");

const DeviceColumnCompareOp = options_mod.DeviceColumnCompareOp;
const DeviceParquetRangeFilter = options_mod.DeviceParquetRangeFilter;
const DeviceScalar = options_mod.DeviceScalar;
const ParquetRangePredicate = options_mod.ParquetRangePredicate;
const Range = options_mod.Range;
const appendOwnedNameUnique = names_mod.appendOwnedNameUnique;
const appendBorrowedNameUnique = names_mod.appendBorrowedNameUnique;
const nameInBorrowedList = names_mod.nameInBorrowedList;
const freeOwnedNameItems = names_mod.freeOwnedNameItems;
const freeNameList = names_mod.freeNameList;

pub const LazyScanPushdown = struct {
    allocator: std.mem.Allocator,
    projection: ?[][]const u8 = null,
    range_predicate: ?DeviceParquetRangeFilter = null,

    pub fn deinit(self: *LazyScanPushdown) void {
        if (self.projection) |names| freeNameList(self.allocator, names);
        if (self.range_predicate) |predicate| self.allocator.free(predicate.column);
        self.* = undefined;
    }
};

pub fn planLazyScanPushdown(allocator: std.mem.Allocator, ops: anytype) std.mem.Allocator.Error!LazyScanPushdown {
    var required_names: std.ArrayList([]const u8) = .empty;
    errdefer required_names.deinit(allocator);
    errdefer freeOwnedNameItems(allocator, required_names.items);
    var derived_names: std.ArrayList([]const u8) = .empty;
    defer derived_names.deinit(allocator);

    var saw_select = false;
    var projection_blocked = false;
    var range_predicate: ?DeviceParquetRangeFilter = null;
    errdefer if (range_predicate) |predicate| allocator.free(predicate.column);

    op_loop: for (ops) |op| {
        switch (op) {
            .select => |names| {
                saw_select = true;
                for (names) |name| {
                    if (!nameInBorrowedList(name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, name);
                    }
                }
            },
            .with_column_binary => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                try appendOwnedNameUnique(allocator, &required_names, expr.lhs_name);
                try appendOwnedNameUnique(allocator, &required_names, expr.rhs_name);
            },
            .with_column_scalar => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
            },
            .with_column_compare => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                try appendOwnedNameUnique(allocator, &required_names, expr.lhs_name);
                try appendOwnedNameUnique(allocator, &required_names, expr.rhs_name);
            },
            .with_column_compare_scalar => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
            },
            .group_by_count => |group| {
                if (!nameInBorrowedList(group.key_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, group.key_name);
                }
                saw_select = true;
                break :op_loop;
            },
            .group_by_value => |group| {
                if (!nameInBorrowedList(group.key_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, group.key_name);
                }
                if (!nameInBorrowedList(group.value_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, group.value_name);
                }
                saw_select = true;
                break :op_loop;
            },
            .group_by_stats => |group| {
                if (!nameInBorrowedList(group.key_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, group.key_name);
                }
                if (!nameInBorrowedList(group.value_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, group.value_name);
                }
                saw_select = true;
                break :op_loop;
            },
            .group_by_stats_on => |group| {
                for (group.key_names) |key_name| {
                    if (!nameInBorrowedList(key_name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, key_name);
                    }
                }
                if (!nameInBorrowedList(group.value_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, group.value_name);
                }
                saw_select = true;
                break :op_loop;
            },
            .group_by_profile => |group| {
                if (!nameInBorrowedList(group.key_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, group.key_name);
                }
                if (!nameInBorrowedList(group.value_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, group.value_name);
                }
                saw_select = true;
                break :op_loop;
            },
            .group_by_profile_on => |group| {
                for (group.key_names) |key_name| {
                    if (!nameInBorrowedList(key_name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, key_name);
                    }
                }
                if (!nameInBorrowedList(group.value_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, group.value_name);
                }
                saw_select = true;
                break :op_loop;
            },
            .join_on => |join| {
                // A join changes the output schema by adding right-side payload
                // columns.  Without source schema metadata at this planning
                // layer, a later select cannot be safely split into left-source
                // columns vs. right payload columns.  Keep row-group predicate
                // pruning, but conservatively disable Parquet projection
                // pushdown for the left source rather than risk dropping a left
                // payload or requesting a right column from the source scan.
                projection_blocked = true;
                for (join.left_key_names) |key_name| {
                    if (!nameInBorrowedList(key_name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, key_name);
                    }
                }
                break :op_loop;
            },
            .asof_join => |join| {
                projection_blocked = true;
                if (!nameInBorrowedList(join.left_key_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, join.left_key_name);
                }
                break :op_loop;
            },
            .concat_rows => {
                break :op_loop;
            },
            .distinct_rows => {
                projection_blocked = true;
            },
            .distinct_on => |names| {
                for (names) |name| {
                    if (!nameInBorrowedList(name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, name);
                    }
                }
            },
            .filter_column => |name| {
                if (!nameInBorrowedList(name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, name);
                }
            },
            .filter_scalar => |filter_op| {
                const filter_depends_on_source = !nameInBorrowedList(filter_op.name, derived_names.items);
                if (filter_depends_on_source) try appendOwnedNameUnique(allocator, &required_names, filter_op.name);
                if (filter_depends_on_source and range_predicate == null) {
                    if (parquetRangePredicateFromScalar(filter_op.scalar, filter_op.op)) |predicate| {
                        range_predicate = .{
                            .column = try allocator.dupe(u8, filter_op.name),
                            .predicate = predicate,
                        };
                    }
                }
            },
            .sort_by => |sort| {
                if (!nameInBorrowedList(sort.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, sort.name);
            },
            .top_k => |top| {
                if (!nameInBorrowedList(top.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, top.name);
            },
            .rank_profile_by,
            .rolling_profile,
            .rolling_moment_profile,
            .rolling_range_profile,
            .rolling_normalize_profile,
            .expanding_normalize_profile,
            .rolling_quantile_profile,
            .expanding_quantile_profile,
            .rolling_bool_profile,
            .rolling_drawdown_profile,
            .rolling_robust_profile,
            .rolling_rank_profile,
            .lag_profile,
            .lead_profile,
            .clip_profile,
            .rolling_clip_profile,
            .expanding_clip_profile,
            .threshold_profile,
            .rolling_threshold_profile,
            .expanding_threshold_profile,
            .expanding_profile,
            .expanding_bool_profile,
            .expanding_rank_profile,
            .expanding_robust_profile,
            .expanding_moment_profile,
            .standardize_profile,
            .robust_profile,
            .drawdown_profile,
            .extrema_profile,
            .trend_profile,
            .rolling_trend_profile,
            .expanding_trend_profile,
            .change_point_profile,
            .rolling_change_point_profile,
            .expanding_change_point_profile,
            .sign_profile,
            .rolling_sign_profile,
            .expanding_sign_profile,
            .crossover_profile,
            .rolling_crossover_profile,
            .expanding_crossover_profile,
            .bucket_profile,
            .ema_profile,
            .linear_fit_profile,
            .error_profile,
            .rolling_error_profile,
            .expanding_error_profile,
            .classification_profile,
            .rolling_classification_profile,
            .expanding_classification_profile,
            .bool_transition_profile,
            .rolling_bool_transition_profile,
            .expanding_bool_transition_profile,
            .rolling_correlation_profile,
            .expanding_correlation_profile,
            .expanding_linear_fit_profile,
            .rolling_linear_fit_profile,
            .validity_profile,
            .rolling_validity_profile,
            .expanding_validity_profile,
            => {
                projection_blocked = true;
                try profile_pushdown_mod.addDependencies(allocator, &required_names, derived_names.items, op);
                break :op_loop;
            },
            .filter_mask, .slice_rows, .take_rows, .head, .tail => {},
        }
    }

    const projection = if (saw_select and !projection_blocked) blk: {
        const owned = try required_names.toOwnedSlice(allocator);
        required_names = .empty;
        break :blk owned;
    } else null;
    if (projection == null) freeOwnedNameItems(allocator, required_names.items);
    required_names.deinit(allocator);

    const out = LazyScanPushdown{
        .allocator = allocator,
        .projection = projection,
        .range_predicate = range_predicate,
    };
    range_predicate = null;
    return out;
}

fn parquetRangePredicateFromScalar(scalar: DeviceScalar, op: DeviceColumnCompareOp) ?ParquetRangePredicate {
    return switch (scalar) {
        .bool => |value| blk: {
            const exact = switch (op) {
                .eq => value,
                .ne => !value,
                .gt, .ge, .lt, .le => break :blk null,
            };
            break :blk .{ .bool = .{ .min = exact, .max = exact } };
        },
        .i8 => |value| if (rangeFromScalarPredicate(i8, value, op)) |range| .{ .i8 = range } else null,
        .i16 => |value| if (rangeFromScalarPredicate(i16, value, op)) |range| .{ .i16 = range } else null,
        .i32 => |value| if (rangeFromScalarPredicate(i32, value, op)) |range| .{ .i32 = range } else null,
        .i64 => |value| if (rangeFromScalarPredicate(i64, value, op)) |range| .{ .i64 = range } else null,
        .u8 => |value| if (rangeFromScalarPredicate(u8, value, op)) |range| .{ .u8 = range } else null,
        .u16 => |value| if (rangeFromScalarPredicate(u16, value, op)) |range| .{ .u16 = range } else null,
        .u32 => |value| if (rangeFromScalarPredicate(u32, value, op)) |range| .{ .u32 = range } else null,
        .u64 => |value| if (rangeFromScalarPredicate(u64, value, op)) |range| .{ .u64 = range } else null,
        .usize => |value| if (rangeFromScalarPredicate(usize, value, op)) |range| .{ .usize = range } else null,
        .isize => |value| if (rangeFromScalarPredicate(isize, value, op)) |range| .{ .isize = range } else null,
        .f16 => |value| if (rangeFromScalarPredicate(f16, value, op)) |range| .{ .f16 = range } else null,
        .f32 => |value| if (rangeFromScalarPredicate(f32, value, op)) |range| .{ .f32 = range } else null,
        .f64 => |value| if (rangeFromScalarPredicate(f64, value, op)) |range| .{ .f64 = range } else null,
        .bf16, .c64, .c128 => null,
    };
}

fn rangeFromScalarPredicate(comptime T: type, value: T, op: DeviceColumnCompareOp) ?Range(T) {
    if (comptime @typeInfo(T) == .float) {
        if (std.math.isNan(value)) return null;
    }
    return switch (op) {
        .eq => .{ .min = value, .max = value },
        .gt, .ge => .{ .min = value },
        .lt, .le => .{ .max = value },
        .ne => null,
    };
}
