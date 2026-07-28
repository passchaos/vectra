const std = @import("std");
const names_mod = @import("dataframe_names.zig");
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

pub fn formatLazyScanPushdown(writer: *std.Io.Writer, pushdown: anytype) std.Io.Writer.Error!void {
    var printed = false;
    if (pushdown.range_predicate) |predicate| {
        try writer.print("range={s}", .{predicate.column});
        printed = true;
    }
    if (pushdown.projection) |names| {
        if (printed) try writer.print(", ", .{});
        try writer.print("projection=[", .{});
        for (names, 0..) |name, i| {
            if (i != 0) try writer.print(",", .{});
            try writer.print("{s}", .{name});
        }
        try writer.print("]", .{});
        printed = true;
    }
    if (!printed) try writer.print("none", .{});
}

pub fn formatLazyOp(writer: *std.Io.Writer, op: anytype) std.Io.Writer.Error!void {
    switch (op) {
        .select => |names| {
            try writer.print("select[", .{});
            for (names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("]", .{});
        },
        .filter_mask => |mask| try writer.print("filter_mask(dtype={s}, rows={d})", .{ mask.dtype().name(), mask.len() }),
        .filter_scalar => |filter_op| try writer.print("filter_scalar({s}, op={s}, dtype={s})", .{ filter_op.name, @tagName(filter_op.op), @tagName(filter_op.scalar) }),
        .with_column_binary => |expr| try writer.print("with_column_binary({s}={s} {s} {s})", .{ expr.name, expr.lhs_name, @tagName(expr.op), expr.rhs_name }),
        .with_column_scalar => |expr| try writer.print("with_column_scalar({s}={s} {s} scalar:{s})", .{ expr.name, expr.input_name, @tagName(expr.op), @tagName(expr.scalar) }),
        .with_column_compare => |expr| try writer.print("with_column_compare({s}={s} {s} {s})", .{ expr.name, expr.lhs_name, @tagName(expr.op), expr.rhs_name }),
        .with_column_compare_scalar => |expr| try writer.print("with_column_compare_scalar({s}={s} {s} scalar:{s})", .{ expr.name, expr.input_name, @tagName(expr.op), @tagName(expr.scalar) }),
        .group_by_count => |group| try writer.print("group_by_count({s} -> {s})", .{ group.key_name, group.output_name }),
        .group_by_value => |group| try writer.print("group_by_{s}({s}, value={s} -> {s})", .{ @tagName(group.aggregation), group.key_name, group.value_name, group.output_name }),
        .group_by_stats => |group| try writer.print("group_by_stats({s}, value={s}, prefix={s})", .{ group.key_name, group.value_name, group.output_prefix }),
        .group_by_stats_on => |group| {
            try writer.print("group_by_stats_on([", .{});
            for (group.key_names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("], value={s}, prefix={s})", .{ group.value_name, group.output_prefix });
        },
        .group_by_profile => |group| try writer.print("group_by_profile({s}, value={s}, prefix={s})", .{ group.key_name, group.value_name, group.output_prefix }),
        .group_by_profile_on => |group| {
            try writer.print("group_by_profile_on([", .{});
            for (group.key_names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("], value={s}, prefix={s})", .{ group.value_name, group.output_prefix });
        },
        .join_on => |join| {
            try writer.print("{s}_join_on(left=[", .{@tagName(join.kind)});
            for (join.left_key_names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("], right=[", .{});
            for (join.right_key_names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("])", .{});
        },
        .asof_join => |join| try writer.print("asof_join({s}->{s}, strategy={s})", .{ join.left_key_name, join.right_key_name, @tagName(join.options.strategy) }),
        .concat_rows => |right| try writer.print("concat_rows(rows={d}, cols={d})", .{ right.height(), right.width() }),
        .distinct_rows => try writer.print("distinct_rows", .{}),
        .distinct_on => |names| {
            try writer.print("distinct_on([", .{});
            for (names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("])", .{});
        },
        .sort_by => |sort| try writer.print("sort_by({s}, desc={})", .{ sort.name, sort.options.descending }),
        .top_k => |top| try writer.print("top_k({s}, k={d}, desc={})", .{ top.name, top.k, top.options.descending }),
        .rank_profile_by => |rank| try writer.print("rank_profile_by({s}, prefix={s}, desc={})", .{ rank.name, rank.output_prefix, rank.options.descending }),
        .rolling_profile => |rolling| try writer.print("rolling_profile({s}, prefix={s}, window={d})", .{ rolling.name, rolling.output_prefix, rolling.options.window }),
        .rolling_moment_profile => |rolling| try writer.print("rolling_moment_profile({s}, prefix={s}, window={d})", .{ rolling.name, rolling.output_prefix, rolling.options.window }),
        .rolling_range_profile => |rolling| try writer.print("rolling_range_profile({s}, prefix={s}, window={d})", .{ rolling.name, rolling.output_prefix, rolling.options.window }),
        .rolling_normalize_profile => |rolling| try writer.print("rolling_normalize_profile({s}, prefix={s}, window={d})", .{ rolling.name, rolling.output_prefix, rolling.options.window }),
        .expanding_normalize_profile => |expanding| try writer.print("expanding_normalize_profile({s}, prefix={s}, min_periods={d})", .{ expanding.name, expanding.output_prefix, expanding.options.min_periods }),
        .rolling_quantile_profile => |rolling| try writer.print("rolling_quantile_profile({s}, prefix={s}, window={d})", .{ rolling.name, rolling.output_prefix, rolling.options.window }),
        .expanding_quantile_profile => |expanding| try writer.print("expanding_quantile_profile({s}, prefix={s}, min_periods={d})", .{ expanding.name, expanding.output_prefix, expanding.options.min_periods }),
        .rolling_bool_profile => |rolling| try writer.print("rolling_bool_profile({s}, prefix={s}, window={d})", .{ rolling.name, rolling.output_prefix, rolling.options.window }),
        .rolling_drawdown_profile => |rolling| try writer.print("rolling_drawdown_profile({s}, prefix={s}, window={d})", .{ rolling.name, rolling.output_prefix, rolling.options.window }),
        .rolling_robust_profile => |rolling| try writer.print("rolling_robust_profile({s}, prefix={s}, window={d})", .{ rolling.name, rolling.output_prefix, rolling.options.window }),
        .rolling_rank_profile => |rolling| try writer.print("rolling_rank_profile({s}, prefix={s}, window={d})", .{ rolling.name, rolling.output_prefix, rolling.options.window }),
        .lag_profile => |lag| try writer.print("lag_profile({s}, prefix={s}, periods={d})", .{ lag.name, lag.output_prefix, lag.options.periods }),
        .lead_profile => |lead| try writer.print("lead_profile({s}, prefix={s}, periods={d})", .{ lead.name, lead.output_prefix, lead.options.periods }),
        .clip_profile => |clip| try writer.print("clip_profile({s}, prefix={s}, [{d},{d}])", .{ clip.name, clip.output_prefix, clip.options.lower, clip.options.upper }),
        .rolling_clip_profile => |clip| try writer.print("rolling_clip_profile({s}, prefix={s}, [{d},{d}], window={d})", .{ clip.name, clip.output_prefix, clip.clip_options.lower, clip.clip_options.upper, clip.options.window }),
        .expanding_clip_profile => |clip| try writer.print("expanding_clip_profile({s}, prefix={s}, [{d},{d}], min_periods={d})", .{ clip.name, clip.output_prefix, clip.clip_options.lower, clip.clip_options.upper, clip.options.min_periods }),
        .threshold_profile => |threshold| try writer.print("threshold_profile({s}, prefix={s}, threshold={d})", .{ threshold.name, threshold.output_prefix, threshold.options.threshold }),
        .rolling_threshold_profile => |threshold| try writer.print("rolling_threshold_profile({s}, prefix={s}, threshold={d}, window={d})", .{ threshold.name, threshold.output_prefix, threshold.threshold, threshold.options.window }),
        .expanding_threshold_profile => |threshold| try writer.print("expanding_threshold_profile({s}, prefix={s}, threshold={d}, min_periods={d})", .{ threshold.name, threshold.output_prefix, threshold.threshold, threshold.options.min_periods }),
        .expanding_profile => |expanding| try writer.print("expanding_profile({s}, prefix={s}, min_periods={d})", .{ expanding.name, expanding.output_prefix, expanding.options.min_periods }),
        .expanding_bool_profile => |expanding| try writer.print("expanding_bool_profile({s}, prefix={s}, min_periods={d})", .{ expanding.name, expanding.output_prefix, expanding.options.min_periods }),
        .expanding_rank_profile => |expanding| try writer.print("expanding_rank_profile({s}, prefix={s}, min_periods={d})", .{ expanding.name, expanding.output_prefix, expanding.options.min_periods }),
        .expanding_robust_profile => |expanding| try writer.print("expanding_robust_profile({s}, prefix={s}, min_periods={d})", .{ expanding.name, expanding.output_prefix, expanding.options.min_periods }),
        .expanding_moment_profile => |expanding| try writer.print("expanding_moment_profile({s}, prefix={s}, min_periods={d})", .{ expanding.name, expanding.output_prefix, expanding.options.min_periods }),
        .standardize_profile => |standardize| try writer.print("standardize_profile({s}, prefix={s}, min_periods={d})", .{ standardize.name, standardize.output_prefix, standardize.options.min_periods }),
        .robust_profile => |robust| try writer.print("robust_profile({s}, prefix={s}, min_periods={d})", .{ robust.name, robust.output_prefix, robust.options.min_periods }),
        .drawdown_profile => |drawdown| try writer.print("drawdown_profile({s}, prefix={s}, min_periods={d})", .{ drawdown.name, drawdown.output_prefix, drawdown.options.min_periods }),
        .extrema_profile => |extrema| try writer.print("extrema_profile({s}, prefix={s}, min_periods={d})", .{ extrema.name, extrema.output_prefix, extrema.options.min_periods }),
        .trend_profile => |trend| try writer.print("trend_profile({s}, prefix={s}, periods={d})", .{ trend.name, trend.output_prefix, trend.options.periods }),
        .rolling_trend_profile => |trend| try writer.print("rolling_trend_profile({s}, prefix={s}, periods={d}, window={d})", .{ trend.name, trend.output_prefix, trend.trend_options.periods, trend.options.window }),
        .expanding_trend_profile => |trend| try writer.print("expanding_trend_profile({s}, prefix={s}, periods={d}, min_periods={d})", .{ trend.name, trend.output_prefix, trend.trend_options.periods, trend.options.min_periods }),
        .change_point_profile => |change| try writer.print("change_point_profile({s}, prefix={s}, threshold={d}, periods={d})", .{ change.name, change.output_prefix, change.threshold, change.options.periods }),
        .rolling_change_point_profile => |change| try writer.print("rolling_change_point_profile({s}, prefix={s}, threshold={d}, periods={d}, window={d})", .{ change.name, change.output_prefix, change.threshold, change.change_options.periods, change.options.window }),
        .expanding_change_point_profile => |change| try writer.print("expanding_change_point_profile({s}, prefix={s}, threshold={d}, periods={d}, min_periods={d})", .{ change.name, change.output_prefix, change.threshold, change.change_options.periods, change.options.min_periods }),
        .sign_profile => |sign| try writer.print("sign_profile({s}, prefix={s}, periods={d})", .{ sign.name, sign.output_prefix, sign.options.periods }),
        .rolling_sign_profile => |sign| try writer.print("rolling_sign_profile({s}, prefix={s}, periods={d}, window={d})", .{ sign.name, sign.output_prefix, sign.sign_options.periods, sign.options.window }),
        .expanding_sign_profile => |sign| try writer.print("expanding_sign_profile({s}, prefix={s}, periods={d}, min_periods={d})", .{ sign.name, sign.output_prefix, sign.sign_options.periods, sign.options.min_periods }),
        .crossover_profile => |cross| try writer.print("crossover_profile({s},{s}, prefix={s}, periods={d})", .{ cross.lhs_name, cross.rhs_name, cross.output_prefix, cross.options.periods }),
        .rolling_crossover_profile => |cross| try writer.print("rolling_crossover_profile({s},{s}, prefix={s}, periods={d}, window={d})", .{ cross.lhs_name, cross.rhs_name, cross.output_prefix, cross.cross_options.periods, cross.options.window }),
        .expanding_crossover_profile => |cross| try writer.print("expanding_crossover_profile({s},{s}, prefix={s}, periods={d}, min_periods={d})", .{ cross.lhs_name, cross.rhs_name, cross.output_prefix, cross.cross_options.periods, cross.options.min_periods }),
        .bucket_profile => |bucket| try writer.print("bucket_profile({s}, prefix={s}, buckets={d})", .{ bucket.name, bucket.output_prefix, bucket.options.buckets }),
        .ema_profile => |ema| try writer.print("ema_profile({s}, prefix={s}, alpha={d})", .{ ema.name, ema.output_prefix, ema.options.alpha }),
        .linear_fit_profile => |fit| try writer.print("linear_fit_profile({s}->{s}, prefix={s})", .{ fit.x_name, fit.y_name, fit.output_prefix }),
        .error_profile => |err| try writer.print("error_profile(actual={s}, predicted={s}, prefix={s})", .{ err.actual_name, err.predicted_name, err.output_prefix }),
        .rolling_error_profile => |err| try writer.print("rolling_error_profile(actual={s}, predicted={s}, prefix={s}, window={d})", .{ err.actual_name, err.predicted_name, err.output_prefix, err.options.window }),
        .expanding_error_profile => |err| try writer.print("expanding_error_profile(actual={s}, predicted={s}, prefix={s}, min_periods={d})", .{ err.actual_name, err.predicted_name, err.output_prefix, err.options.min_periods }),
        .classification_profile => |class| try writer.print("classification_profile(actual={s}, predicted={s}, prefix={s})", .{ class.actual_name, class.predicted_name, class.output_prefix }),
        .rolling_classification_profile => |class| try writer.print("rolling_classification_profile(actual={s}, predicted={s}, prefix={s}, window={d})", .{ class.actual_name, class.predicted_name, class.output_prefix, class.options.window }),
        .expanding_classification_profile => |class| try writer.print("expanding_classification_profile(actual={s}, predicted={s}, prefix={s}, min_periods={d})", .{ class.actual_name, class.predicted_name, class.output_prefix, class.options.min_periods }),
        .bool_transition_profile => |transition| try writer.print("bool_transition_profile({s}, prefix={s}, periods={d})", .{ transition.name, transition.output_prefix, transition.options.periods }),
        .rolling_bool_transition_profile => |transition| try writer.print("rolling_bool_transition_profile({s}, prefix={s}, periods={d}, window={d})", .{ transition.name, transition.output_prefix, transition.transition_options.periods, transition.options.window }),
        .expanding_bool_transition_profile => |transition| try writer.print("expanding_bool_transition_profile({s}, prefix={s}, periods={d}, min_periods={d})", .{ transition.name, transition.output_prefix, transition.transition_options.periods, transition.options.min_periods }),
        .rolling_correlation_profile => |corr| try writer.print("rolling_correlation_profile({s},{s}, prefix={s}, window={d})", .{ corr.x_name, corr.y_name, corr.output_prefix, corr.options.window }),
        .expanding_correlation_profile => |corr| try writer.print("expanding_correlation_profile({s},{s}, prefix={s}, min_periods={d})", .{ corr.x_name, corr.y_name, corr.output_prefix, corr.options.min_periods }),
        .expanding_linear_fit_profile => |fit| try writer.print("expanding_linear_fit_profile({s}->{s}, prefix={s}, min_periods={d})", .{ fit.x_name, fit.y_name, fit.output_prefix, fit.options.min_periods }),
        .rolling_linear_fit_profile => |fit| try writer.print("rolling_linear_fit_profile({s}->{s}, prefix={s}, window={d})", .{ fit.x_name, fit.y_name, fit.output_prefix, fit.options.window }),
        .validity_profile => |validity| try writer.print("validity_profile({s}, prefix={s})", .{ validity.name, validity.output_prefix }),
        .rolling_validity_profile => |validity| try writer.print("rolling_validity_profile({s}, prefix={s}, window={d})", .{ validity.name, validity.output_prefix, validity.options.window }),
        .expanding_validity_profile => |validity| try writer.print("expanding_validity_profile({s}, prefix={s}, min_periods={d})", .{ validity.name, validity.output_prefix, validity.options.min_periods }),
        .head => |n| try writer.print("head({d})", .{n}),
        .tail => |n| try writer.print("tail({d})", .{n}),
    }
}

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
            .rank_profile_by => |rank| {
                // A rank profile appends derived rank/window columns while
                // preserving the rest of the input table.  Without source schema
                // metadata here, a later select cannot be split safely into
                // source columns vs. rank-derived columns, so keep scalar
                // predicate pruning but avoid Parquet projection pushdown.
                projection_blocked = true;
                if (!nameInBorrowedList(rank.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, rank.name);
                break :op_loop;
            },
            .rolling_profile => |rolling| {
                // Rolling profiles append several derived columns and preserve
                // the existing table, so projection pushdown needs schema
                // awareness to avoid dropping later-selected source columns.
                projection_blocked = true;
                if (!nameInBorrowedList(rolling.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, rolling.name);
                break :op_loop;
            },
            .rolling_moment_profile => |rolling| {
                // Rolling moment profiles append higher-order distribution
                // diagnostics while preserving source columns. Keep predicates
                // but block projection until generated-field schema is explicit.
                projection_blocked = true;
                if (!nameInBorrowedList(rolling.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, rolling.name);
                break :op_loop;
            },
            .rolling_range_profile => |rolling| {
                // Rolling range profiles append low/high/range/position fields
                // and preserve the input table, so projection pushdown needs
                // schema awareness before it can safely pass this operation.
                projection_blocked = true;
                if (!nameInBorrowedList(rolling.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, rolling.name);
                break :op_loop;
            },
            .rolling_normalize_profile => |rolling| {
                // Rolling normalize profiles append window-local scaling fields
                // and preserve the input table, so projection pushdown needs
                // derived-field schema awareness before it can pass through.
                projection_blocked = true;
                if (!nameInBorrowedList(rolling.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, rolling.name);
                break :op_loop;
            },
            .expanding_normalize_profile => |expanding| {
                projection_blocked = true;
                if (!nameInBorrowedList(expanding.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, expanding.name);
                break :op_loop;
            },
            .rolling_quantile_profile => |rolling| {
                // Rolling quantile profiles append window distribution fields and
                // preserve the input table, so projection pushdown needs
                // generated-field schema awareness before it can pass through.
                projection_blocked = true;
                if (!nameInBorrowedList(rolling.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, rolling.name);
                break :op_loop;
            },
            .expanding_quantile_profile => |expanding| {
                projection_blocked = true;
                if (!nameInBorrowedList(expanding.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, expanding.name);
                break :op_loop;
            },
            .rolling_bool_profile => |rolling| {
                // Rolling bool profiles append count/rate/predicate diagnostics
                // while preserving source columns. Keep scan predicates, but do
                // not push projection through generated bool-window fields until
                // the lazy planner tracks source-vs-derived schema explicitly.
                projection_blocked = true;
                if (!nameInBorrowedList(rolling.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, rolling.name);
                break :op_loop;
            },
            .rolling_drawdown_profile => |rolling| {
                // Rolling drawdown profiles append window-local risk diagnostics
                // and preserve the source schema. Keep scan predicates but block
                // projection pushdown until generated fields are tracked.
                projection_blocked = true;
                if (!nameInBorrowedList(rolling.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, rolling.name);
                break :op_loop;
            },
            .rolling_robust_profile => |rolling| {
                projection_blocked = true;
                if (!nameInBorrowedList(rolling.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, rolling.name);
                break :op_loop;
            },
            .rolling_rank_profile => |rolling| {
                projection_blocked = true;
                if (!nameInBorrowedList(rolling.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, rolling.name);
                break :op_loop;
            },
            .lag_profile => |lag| {
                // Lag profiles append multiple derived columns and preserve the
                // input schema.  Like rank/rolling profiles, keep scan predicate
                // pruning but avoid unsafe projection pushdown until the planner
                // has schema-level knowledge of derived vs. source fields.
                projection_blocked = true;
                if (!nameInBorrowedList(lag.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, lag.name);
                break :op_loop;
            },
            .lead_profile => |lead| {
                // Lead profiles append forward-looking derived columns and
                // preserve the input schema. Keep scan predicates, but block
                // projection pushdown across generated lead fields.
                projection_blocked = true;
                if (!nameInBorrowedList(lead.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, lead.name);
                break :op_loop;
            },
            .clip_profile => |clip| {
                // Clip profiles append cleaning diagnostics and preserve the
                // source column, so projection pushdown must wait for generated
                // field schema awareness.
                projection_blocked = true;
                if (!nameInBorrowedList(clip.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, clip.name);
                break :op_loop;
            },
            .rolling_clip_profile => |clip| {
                // Rolling clip profiles append window-level clipping summaries
                // and keep source columns. Predicate pruning is still safe, but
                // projection pushdown needs generated-field schema awareness.
                projection_blocked = true;
                if (!nameInBorrowedList(clip.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, clip.name);
                break :op_loop;
            },
            .expanding_clip_profile => |clip| {
                // Expanding clip profiles append cumulative clipping summaries
                // while preserving the input table. Keep scan predicates, but
                // block projection until generated fields are tracked.
                projection_blocked = true;
                if (!nameInBorrowedList(clip.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, clip.name);
                break :op_loop;
            },
            .threshold_profile => |threshold| {
                projection_blocked = true;
                if (!nameInBorrowedList(threshold.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, threshold.name);
                break :op_loop;
            },
            .rolling_threshold_profile => |threshold| {
                // Rolling threshold profiles append window-level distance/rate
                // diagnostics while preserving the input table. Keep scalar
                // predicate pruning, but block projection pushdown until the
                // planner tracks generated profile fields separately.
                projection_blocked = true;
                if (!nameInBorrowedList(threshold.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, threshold.name);
                break :op_loop;
            },
            .expanding_threshold_profile => |threshold| {
                // Expanding threshold profiles append cumulative distance/rate
                // diagnostics and preserve source fields. Keep scan predicates
                // but block projection pushdown across generated columns until
                // lazy schema tracking can split source vs. derived names.
                projection_blocked = true;
                if (!nameInBorrowedList(threshold.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, threshold.name);
                break :op_loop;
            },
            .expanding_profile => |expanding| {
                // Expanding profiles append cumulative derived columns while
                // preserving source columns.  Keep the source dependency for
                // scans, but do not push projection through this schema-changing
                // operation until planner schema metadata is richer.
                projection_blocked = true;
                if (!nameInBorrowedList(expanding.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, expanding.name);
                break :op_loop;
            },
            .expanding_bool_profile => |expanding| {
                projection_blocked = true;
                if (!nameInBorrowedList(expanding.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, expanding.name);
                break :op_loop;
            },
            .expanding_rank_profile => |expanding| {
                projection_blocked = true;
                if (!nameInBorrowedList(expanding.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, expanding.name);
                break :op_loop;
            },
            .expanding_robust_profile => |expanding| {
                projection_blocked = true;
                if (!nameInBorrowedList(expanding.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, expanding.name);
                break :op_loop;
            },
            .expanding_moment_profile => |expanding| {
                // Expanding moment profiles append higher-order cumulative
                // distribution diagnostics while preserving source columns.
                projection_blocked = true;
                if (!nameInBorrowedList(expanding.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, expanding.name);
                break :op_loop;
            },
            .standardize_profile => |standardize| {
                // Standardization adds derived scale columns while retaining the
                // input schema. It depends on the whole source column, so keep
                // predicate pruning but avoid unsafe projection pushdown until
                // derived-field schema metadata is available.
                projection_blocked = true;
                if (!nameInBorrowedList(standardize.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, standardize.name);
                break :op_loop;
            },
            .robust_profile => |robust| {
                // Robust profiles append median/MAD/IQR-derived columns while
                // preserving the input table. Keep predicate pruning but avoid
                // projection pushdown until derived-field schema tracking can
                // distinguish source and generated columns.
                projection_blocked = true;
                if (!nameInBorrowedList(robust.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, robust.name);
                break :op_loop;
            },
            .drawdown_profile => |drawdown| {
                // Drawdown profiles append sequence-derived columns while
                // preserving source fields. Keep scan predicates, but avoid
                // projection pushdown until source-vs-derived schema tracking is
                // rich enough to safely split later selects.
                projection_blocked = true;
                if (!nameInBorrowedList(drawdown.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, drawdown.name);
                break :op_loop;
            },
            .extrema_profile => |extrema| {
                projection_blocked = true;
                if (!nameInBorrowedList(extrema.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, extrema.name);
                break :op_loop;
            },
            .trend_profile => |trend| {
                // Trend profiles are row-order dependent and append several
                // derived columns. Preserve scan predicates, but block Parquet
                // projection pushdown until the planner can reason about
                // generated fields separately from source fields.
                projection_blocked = true;
                if (!nameInBorrowedList(trend.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, trend.name);
                break :op_loop;
            },
            .rolling_trend_profile => |trend| {
                projection_blocked = true;
                if (!nameInBorrowedList(trend.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, trend.name);
                break :op_loop;
            },
            .expanding_trend_profile => |trend| {
                projection_blocked = true;
                if (!nameInBorrowedList(trend.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, trend.name);
                break :op_loop;
            },
            .change_point_profile => |change| {
                // Change-point profiles are row-order dependent and append
                // derived jump diagnostics. Keep predicate pruning but block
                // projection across generated fields until schema tracking is
                // explicit.
                projection_blocked = true;
                if (!nameInBorrowedList(change.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, change.name);
                break :op_loop;
            },
            .rolling_change_point_profile => |change| {
                // Rolling change-point profiles append aggregate jump
                // diagnostics over order-dependent deltas. Keep predicate
                // pruning but block projection across generated fields.
                projection_blocked = true;
                if (!nameInBorrowedList(change.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, change.name);
                break :op_loop;
            },
            .expanding_change_point_profile => |change| {
                projection_blocked = true;
                if (!nameInBorrowedList(change.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, change.name);
                break :op_loop;
            },
            .sign_profile => |sign| {
                projection_blocked = true;
                if (!nameInBorrowedList(sign.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, sign.name);
                break :op_loop;
            },
            .rolling_sign_profile => |sign| {
                projection_blocked = true;
                if (!nameInBorrowedList(sign.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, sign.name);
                break :op_loop;
            },
            .expanding_sign_profile => |sign| {
                projection_blocked = true;
                if (!nameInBorrowedList(sign.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, sign.name);
                break :op_loop;
            },
            .crossover_profile => |cross| {
                // Crossover profiles depend on two source columns and append
                // several signal columns. Keep scan predicates but block
                // projection pushdown until derived-field schema tracking can
                // safely split source and generated columns.
                projection_blocked = true;
                if (!nameInBorrowedList(cross.lhs_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, cross.lhs_name);
                if (!nameInBorrowedList(cross.rhs_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, cross.rhs_name);
                break :op_loop;
            },
            .rolling_crossover_profile => |cross| {
                // Rolling crossover profiles append window-level signal
                // summaries derived from two source columns. Predicate pruning
                // remains safe, but projection pushdown must wait until the
                // lazy planner tracks generated crossover fields explicitly.
                projection_blocked = true;
                if (!nameInBorrowedList(cross.lhs_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, cross.lhs_name);
                if (!nameInBorrowedList(cross.rhs_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, cross.rhs_name);
                break :op_loop;
            },
            .expanding_crossover_profile => |cross| {
                projection_blocked = true;
                if (!nameInBorrowedList(cross.lhs_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, cross.lhs_name);
                if (!nameInBorrowedList(cross.rhs_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, cross.rhs_name);
                break :op_loop;
            },
            .bucket_profile => |bucket| {
                // Bucket profiles depend on the whole source distribution and
                // append several derived fields, so keep predicates but block
                // projection pushdown until generated-field schema metadata is
                // tracked explicitly.
                projection_blocked = true;
                if (!nameInBorrowedList(bucket.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, bucket.name);
                break :op_loop;
            },
            .ema_profile => |ema| {
                // EMA profiles are order-dependent and append derived columns,
                // so keep predicate pruning but block projection pushdown until
                // generated-field schema metadata is explicit.
                projection_blocked = true;
                if (!nameInBorrowedList(ema.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, ema.name);
                break :op_loop;
            },
            .linear_fit_profile => |fit| {
                // Linear-fit profiles depend on two source columns and append
                // model diagnostics. Keep predicate pruning but block projection
                // pushdown until generated-field schema metadata is explicit.
                projection_blocked = true;
                if (!nameInBorrowedList(fit.x_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, fit.x_name);
                if (!nameInBorrowedList(fit.y_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, fit.y_name);
                break :op_loop;
            },
            .error_profile => |err| {
                projection_blocked = true;
                if (!nameInBorrowedList(err.actual_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, err.actual_name);
                if (!nameInBorrowedList(err.predicted_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, err.predicted_name);
                break :op_loop;
            },
            .rolling_error_profile => |err| {
                projection_blocked = true;
                if (!nameInBorrowedList(err.actual_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, err.actual_name);
                if (!nameInBorrowedList(err.predicted_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, err.predicted_name);
                break :op_loop;
            },
            .expanding_error_profile => |err| {
                projection_blocked = true;
                if (!nameInBorrowedList(err.actual_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, err.actual_name);
                if (!nameInBorrowedList(err.predicted_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, err.predicted_name);
                break :op_loop;
            },
            .classification_profile => |class| {
                projection_blocked = true;
                if (!nameInBorrowedList(class.actual_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, class.actual_name);
                if (!nameInBorrowedList(class.predicted_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, class.predicted_name);
                break :op_loop;
            },
            .rolling_classification_profile => |class| {
                projection_blocked = true;
                if (!nameInBorrowedList(class.actual_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, class.actual_name);
                if (!nameInBorrowedList(class.predicted_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, class.predicted_name);
                break :op_loop;
            },
            .expanding_classification_profile => |class| {
                projection_blocked = true;
                if (!nameInBorrowedList(class.actual_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, class.actual_name);
                if (!nameInBorrowedList(class.predicted_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, class.predicted_name);
                break :op_loop;
            },
            .bool_transition_profile => |transition| {
                projection_blocked = true;
                if (!nameInBorrowedList(transition.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, transition.name);
                break :op_loop;
            },
            .rolling_bool_transition_profile => |transition| {
                // Rolling transition summaries append window-level event-rate
                // columns and preserve the input table. Keep predicates, but
                // block projection pushdown until generated fields are tracked.
                projection_blocked = true;
                if (!nameInBorrowedList(transition.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, transition.name);
                break :op_loop;
            },
            .expanding_bool_transition_profile => |transition| {
                projection_blocked = true;
                if (!nameInBorrowedList(transition.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, transition.name);
                break :op_loop;
            },
            .rolling_correlation_profile => |corr| {
                // Rolling correlation profiles depend on two source columns and
                // append several window diagnostics. Keep predicate pruning but
                // block projection pushdown until generated-field schema
                // metadata is explicit.
                projection_blocked = true;
                if (!nameInBorrowedList(corr.x_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, corr.x_name);
                if (!nameInBorrowedList(corr.y_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, corr.y_name);
                break :op_loop;
            },
            .expanding_correlation_profile => |corr| {
                projection_blocked = true;
                if (!nameInBorrowedList(corr.x_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, corr.x_name);
                if (!nameInBorrowedList(corr.y_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, corr.y_name);
                break :op_loop;
            },
            .expanding_linear_fit_profile => |fit| {
                projection_blocked = true;
                if (!nameInBorrowedList(fit.x_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, fit.x_name);
                if (!nameInBorrowedList(fit.y_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, fit.y_name);
                break :op_loop;
            },
            .rolling_linear_fit_profile => |fit| {
                // Rolling linear-fit profiles append window-local model fields
                // from two source columns. Preserve scan predicates but do not
                // push projection across generated regression diagnostics until
                // the planner can distinguish source and derived schemas.
                projection_blocked = true;
                if (!nameInBorrowedList(fit.x_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, fit.x_name);
                if (!nameInBorrowedList(fit.y_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, fit.y_name);
                break :op_loop;
            },
            .validity_profile => |validity| {
                // Validity profiles are schema-changing data-quality diagnostics
                // over one source column. Keep source dependency for scans and
                // avoid projection pushdown across generated validity fields.
                projection_blocked = true;
                if (!nameInBorrowedList(validity.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, validity.name);
                break :op_loop;
            },
            .rolling_validity_profile => |validity| {
                // Rolling validity profiles append data-quality window metrics
                // and preserve source fields. Keep scan predicates but avoid
                // unsafe projection across generated validity diagnostics.
                projection_blocked = true;
                if (!nameInBorrowedList(validity.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, validity.name);
                break :op_loop;
            },
            .expanding_validity_profile => |validity| {
                // Expanding validity profiles append cumulative quality metrics
                // and preserve source fields, so projection pushdown must wait
                // for derived-field schema tracking.
                projection_blocked = true;
                if (!nameInBorrowedList(validity.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, validity.name);
                break :op_loop;
            },
            .filter_mask, .head, .tail => {},
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
