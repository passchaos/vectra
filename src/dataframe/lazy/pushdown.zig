//! Parquet scan pushdown planning for lazy dataframe plans.

const std = @import("std");
const array_mod = @import("../../array.zig");
const names_mod = @import("../../dataframe_names.zig");
const profile_pushdown_mod = @import("pushdown_profile.zig");
const null_pushdown_mod = @import("pushdown/null.zig");
const range_pushdown_mod = @import("pushdown/range.zig");
const requirements_mod = @import("pushdown/requirements.zig");
const options_mod = @import("../../dataframe_options.zig");

const DeviceParquetNullFilter = options_mod.DeviceParquetNullFilter;
const DeviceParquetRangeFilter = options_mod.DeviceParquetRangeFilter;
const appendBorrowedNameUnique = names_mod.appendBorrowedNameUnique;
const nameInBorrowedList = names_mod.nameInBorrowedList;
const freeOwnedNameItems = names_mod.freeOwnedNameItems;
const freeNameList = names_mod.freeNameList;
const clearNullPredicate = null_pushdown_mod.clearNullPredicate;
const setNullPredicate = null_pushdown_mod.setNullPredicate;
const mergeRangePredicate = range_pushdown_mod.mergeRangePredicate;
const parquetRangePredicateFromBounds = range_pushdown_mod.parquetRangePredicateFromBounds;
const parquetRangePredicateFromDroppedScalar = range_pushdown_mod.parquetRangePredicateFromDroppedScalar;
const parquetRangePredicateFromSingletonColumn = range_pushdown_mod.parquetRangePredicateFromSingletonColumn;
const parquetRangePredicateFromScalar = range_pushdown_mod.parquetRangePredicateFromScalar;

const addBinaryColumnOutputRequirements = requirements_mod.addBinaryColumnOutputRequirements;
const addGroupedValueOutputRequirements = requirements_mod.addGroupedValueOutputRequirements;
const addGroupedWeightedPairOutputRequirements = requirements_mod.addGroupedWeightedPairOutputRequirements;
const addGroupedWeightedValueOutputRequirements = requirements_mod.addGroupedWeightedValueOutputRequirements;
const addListColumnOutputRequirements = requirements_mod.addListColumnOutputRequirements;
const addRowMultiOutputRequirements = requirements_mod.addRowMultiOutputRequirements;
const addRowSingleOutputRequirements = requirements_mod.addRowSingleOutputRequirements;
const addRowWeightedPairColumnOutputRequirements = requirements_mod.addRowWeightedPairColumnOutputRequirements;
const addSourceNameRequirement = requirements_mod.addSourceNameRequirement;
const addSourceNameRequirements = requirements_mod.addSourceNameRequirements;
const addTernaryColumnOutputRequirements = requirements_mod.addTernaryColumnOutputRequirements;
const addUnaryColumnOutputRequirements = requirements_mod.addUnaryColumnOutputRequirements;
const addWeightedPairRowSingleOutputRequirements = requirements_mod.addWeightedPairRowSingleOutputRequirements;
const addWeightedRowMultiOutputRequirements = requirements_mod.addWeightedRowMultiOutputRequirements;
const addWeightedRowSingleOutputRequirements = requirements_mod.addWeightedRowSingleOutputRequirements;
const markDerivedName = requirements_mod.markDerivedName;

pub const LazyScanPushdown = struct {
    allocator: std.mem.Allocator,
    projection: ?[][]const u8 = null,
    range_predicate: ?DeviceParquetRangeFilter = null,
    null_predicate: ?DeviceParquetNullFilter = null,

    pub fn hasProjection(self: LazyScanPushdown) bool {
        return self.projection != null;
    }

    pub fn projectionColumnCount(self: LazyScanPushdown) usize {
        return if (self.projection) |names| names.len else 0;
    }

    pub fn projectionNames(self: LazyScanPushdown) []const []const u8 {
        return if (self.projection) |names| names else &.{};
    }

    pub fn projectionNameAt(self: LazyScanPushdown, index: usize) ?[]const u8 {
        const names = self.projection orelse return null;
        if (index >= names.len) return null;
        return names[index];
    }

    pub fn projectionIndex(self: LazyScanPushdown, name: []const u8) ?usize {
        const names = self.projection orelse return null;
        for (names, 0..) |candidate, index| {
            if (std.mem.eql(u8, candidate, name)) return index;
        }
        return null;
    }

    pub fn projectionContains(self: LazyScanPushdown, name: []const u8) bool {
        return self.projectionIndex(name) != null;
    }

    pub fn projectionNamesUnique(self: LazyScanPushdown) bool {
        const names = self.projection orelse return true;
        for (names, 0..) |name, index| {
            if (nameInBorrowedList(name, names[0..index])) return false;
        }
        return true;
    }

    pub fn hasDuplicateProjectionNames(self: LazyScanPushdown) bool {
        return !self.projectionNamesUnique();
    }

    pub fn duplicateProjectionNameCount(self: LazyScanPushdown) usize {
        const names = self.projection orelse return 0;
        var count: usize = 0;
        for (names, 0..) |name, index| {
            if (nameInBorrowedList(name, names[0..index])) count += 1;
        }
        return count;
    }

    pub fn hasAllProjectionNames(self: LazyScanPushdown, names: []const []const u8) bool {
        for (names) |name| {
            if (!self.projectionContains(name)) return false;
        }
        return true;
    }

    pub fn hasAnyProjectionName(self: LazyScanPushdown, names: []const []const u8) bool {
        for (names) |name| {
            if (self.projectionContains(name)) return true;
        }
        return false;
    }

    pub fn projectsColumn(self: LazyScanPushdown, name: []const u8) bool {
        return !self.hasProjection() or self.projectionContains(name);
    }

    pub fn hasRangePredicate(self: LazyScanPushdown) bool {
        return self.range_predicate != null;
    }

    pub fn rangePredicateColumn(self: LazyScanPushdown) ?[]const u8 {
        return if (self.range_predicate) |predicate| predicate.column else null;
    }

    pub fn rangePredicateDType(self: LazyScanPushdown) ?array_mod.DType {
        const predicate = self.range_predicate orelse return null;
        return switch (predicate.predicate) {
            .f64 => .f64,
            .f32 => .f32,
            .i64 => .i64,
            .i32 => .i32,
            .bool => .bool,
        };
    }

    pub fn hasRangePredicateFor(self: LazyScanPushdown, column: []const u8) bool {
        const active_column = self.rangePredicateColumn() orelse return false;
        return std.mem.eql(u8, active_column, column);
    }

    pub fn hasNullPredicate(self: LazyScanPushdown) bool {
        return self.null_predicate != null;
    }

    pub fn nullPredicateColumn(self: LazyScanPushdown) ?[]const u8 {
        return if (self.null_predicate) |predicate| predicate.column else null;
    }

    pub fn nullPredicateWantNulls(self: LazyScanPushdown) ?bool {
        return if (self.null_predicate) |predicate| predicate.want_nulls else null;
    }

    pub fn hasNullPredicateFor(self: LazyScanPushdown, column: []const u8) bool {
        const active_column = self.nullPredicateColumn() orelse return false;
        return std.mem.eql(u8, active_column, column);
    }

    pub fn hasPredicate(self: LazyScanPushdown) bool {
        return self.hasRangePredicate() or self.hasNullPredicate();
    }

    pub fn hasPushdown(self: LazyScanPushdown) bool {
        return self.hasProjection() or self.hasPredicate();
    }

    pub fn isEmpty(self: LazyScanPushdown) bool {
        return !self.hasPushdown();
    }

    pub fn isNonEmpty(self: LazyScanPushdown) bool {
        return self.hasPushdown();
    }

    pub fn projectionMetadataNbytes(self: LazyScanPushdown) usize {
        const names = self.projection orelse return 0;
        var total = names.len * @sizeOf([]const u8);
        for (names) |name| total += name.len;
        return total;
    }

    pub fn rangePredicateMetadataNbytes(self: LazyScanPushdown) usize {
        return if (self.range_predicate) |predicate| predicate.column.len else 0;
    }

    pub fn nullPredicateMetadataNbytes(self: LazyScanPushdown) usize {
        return if (self.null_predicate) |predicate| predicate.column.len else 0;
    }

    pub fn predicateMetadataNbytes(self: LazyScanPushdown) usize {
        return self.rangePredicateMetadataNbytes() + self.nullPredicateMetadataNbytes();
    }

    pub fn pushdownMetadataNbytes(self: LazyScanPushdown) usize {
        return self.projectionMetadataNbytes() + self.predicateMetadataNbytes();
    }

    pub fn deinit(self: *LazyScanPushdown) void {
        if (self.projection) |names| freeNameList(self.allocator, names);
        if (self.range_predicate) |predicate| self.allocator.free(predicate.column);
        if (self.null_predicate) |predicate| self.allocator.free(predicate.column);
        self.* = undefined;
    }
};

pub fn planLazyScanPushdown(allocator: std.mem.Allocator, ops: anytype) std.mem.Allocator.Error!LazyScanPushdown {
    var required_names: std.ArrayList([]const u8) = .empty;
    errdefer required_names.deinit(allocator);
    errdefer freeOwnedNameItems(allocator, required_names.items);
    var derived_names: std.ArrayList([]const u8) = .empty;
    defer derived_names.deinit(allocator);
    var literal_scalars = std.StringHashMap(options_mod.DeviceScalar).init(allocator);
    defer literal_scalars.deinit();

    var saw_select = false;
    var projection_blocked = false;
    var range_predicate: ?DeviceParquetRangeFilter = null;
    errdefer if (range_predicate) |predicate| allocator.free(predicate.column);
    var null_predicate: ?DeviceParquetNullFilter = null;
    errdefer if (null_predicate) |predicate| allocator.free(predicate.column);

    op_loop: for (ops) |op| {
        switch (op) {
            .select => |names| {
                saw_select = true;
                try addSourceNameRequirements(allocator, &required_names, derived_names.items, names);
            },
            .select_column_indices,
            .select_column_range,
            .select_last_columns,
            .drop_column_indices,
            .drop_column_range,
            .drop_last_columns,
            .reverse_columns,
            .sort_columns_by_name,
            .select_name_prefix,
            .select_name_suffix,
            .select_name_contains,
            .select_name_glob,
            .drop_name_prefix,
            .drop_name_suffix,
            .drop_name_contains,
            .drop_name_glob,
            .select_dtypes,
            .select_dtype_class,
            .drop_dtypes,
            .drop_dtype_class,
            .select_nullable_columns,
            .select_non_nullable_columns,
            .select_columns_with_nulls,
            .select_columns_without_nulls,
            .select_columns_with_nans,
            .select_columns_without_nans,
            .select_columns_with_infs,
            .select_columns_without_infs,
            .select_columns_with_positive_infs,
            .select_columns_without_positive_infs,
            .select_columns_with_negative_infs,
            .select_columns_without_negative_infs,
            .select_columns_with_zeros,
            .select_columns_without_zeros,
            .select_columns_with_positive_zeros,
            .select_columns_without_positive_zeros,
            .select_columns_with_negative_zeros,
            .select_columns_without_negative_zeros,
            .select_columns_with_non_zeros,
            .select_columns_without_non_zeros,
            .select_columns_with_positives,
            .select_columns_without_positives,
            .select_columns_with_signbits,
            .select_columns_without_signbits,
            .select_columns_with_negatives,
            .select_columns_without_negatives,
            .select_columns_with_finites,
            .select_columns_without_finites,
            .select_columns_with_normals,
            .select_columns_without_normals,
            .select_columns_with_subnormals,
            .select_columns_without_subnormals,
            .select_columns_with_non_finites,
            .select_columns_without_non_finites,
            .drop_nullable_columns,
            .drop_non_nullable_columns,
            .drop_columns_with_nulls,
            .drop_columns_without_nulls,
            .drop_columns_with_nans,
            .drop_columns_without_nans,
            .drop_columns_with_infs,
            .drop_columns_without_infs,
            .drop_columns_with_positive_infs,
            .drop_columns_without_positive_infs,
            .drop_columns_with_negative_infs,
            .drop_columns_without_negative_infs,
            .drop_columns_with_zeros,
            .drop_columns_without_zeros,
            .drop_columns_with_positive_zeros,
            .drop_columns_without_positive_zeros,
            .drop_columns_with_negative_zeros,
            .drop_columns_without_negative_zeros,
            .drop_columns_with_non_zeros,
            .drop_columns_without_non_zeros,
            .drop_columns_with_positives,
            .drop_columns_without_positives,
            .drop_columns_with_signbits,
            .drop_columns_without_signbits,
            .drop_columns_with_negatives,
            .drop_columns_without_negatives,
            .drop_columns_with_finites,
            .drop_columns_without_finites,
            .drop_columns_with_normals,
            .drop_columns_without_normals,
            .drop_columns_with_subnormals,
            .drop_columns_without_subnormals,
            .drop_columns_with_non_finites,
            .drop_columns_without_non_finites,
            => {
                // Schema-derived selectors require the full source schema
                // before they can be expanded into exact column names.  This
                // lightweight scan planner only sees pending operations, so
                // keep them as collect-time operations instead of guessing a
                // Parquet projection from incomplete information.
                projection_blocked = true;
                break :op_loop;
            },
            .rename_column,
            .rename_columns,
            .add_column_name_prefix,
            .add_column_name_suffix,
            .strip_column_name_prefix,
            .strip_column_name_suffix,
            .replace_column_name_prefix,
            .replace_column_name_suffix,
            .move_column,
            .move_column_before,
            .move_column_after,
            .copy_column,
            .copy_column_at,
            .copy_column_before,
            .copy_column_after,
            .drop_columns,
            => {
                // Schema rewrites change the names visible to all following
                // operations.  Without a full alias map in this conservative
                // scan planner, stop projection/range inference here rather
                // than pushing a post-rename/drop column name into the source
                // scan.  The eager collect path still applies the operation in
                // order after materializing the scan.
                projection_blocked = true;
                break :op_loop;
            },
            .drop_nulls => |names| {
                try addSourceNameRequirements(allocator, &required_names, derived_names.items, names);
                if (range_predicate == null and names.len == 1 and !nameInBorrowedList(names[0], derived_names.items)) {
                    try setNullPredicate(allocator, &null_predicate, names[0], false);
                }
            },
            .drop_all_nulls => |names| {
                try addSourceNameRequirements(allocator, &required_names, derived_names.items, names);
                if (range_predicate == null and names.len == 1 and !nameInBorrowedList(names[0], derived_names.items)) {
                    try setNullPredicate(allocator, &null_predicate, names[0], false);
                }
            },
            .filter_all_nulls => |names| {
                try addSourceNameRequirements(allocator, &required_names, derived_names.items, names);
                if (range_predicate == null and names.len == 1 and !nameInBorrowedList(names[0], derived_names.items)) {
                    try setNullPredicate(allocator, &null_predicate, names[0], true);
                }
            },
            .filter_nulls_column => |name| {
                if (!nameInBorrowedList(name, derived_names.items)) {
                    try addSourceNameRequirement(allocator, &required_names, derived_names.items, name);
                    if (range_predicate == null) try setNullPredicate(allocator, &null_predicate, name, true);
                }
            },
            .drop_nans => |names| {
                if (names.len == 0) {
                    // Empty dropNaNs input means "consider every visible
                    // column".  This planner does not carry a full scan schema,
                    // so materialize the source rather than incorrectly
                    // treating the filter as dependency-free.
                    projection_blocked = true;
                    break :op_loop;
                }
                try addSourceNameRequirements(allocator, &required_names, derived_names.items, names);
            },
            .filter_nans_column => |name| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, name);
            },
            .drop_infs => |names| {
                if (names.len == 0) {
                    // Empty dropInfs input means "consider every visible
                    // column".  This planner does not carry a full scan schema,
                    // so materialize the source rather than incorrectly
                    // treating the filter as dependency-free.
                    projection_blocked = true;
                    break :op_loop;
                }
                try addSourceNameRequirements(allocator, &required_names, derived_names.items, names);
            },
            .filter_infs_column => |name| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, name);
            },
            .drop_positive_infs => |names| {
                if (names.len == 0) {
                    projection_blocked = true;
                    break :op_loop;
                }
                try addSourceNameRequirements(allocator, &required_names, derived_names.items, names);
            },
            .filter_positive_infs_column => |name| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, name);
            },
            .drop_negative_infs => |names| {
                if (names.len == 0) {
                    projection_blocked = true;
                    break :op_loop;
                }
                try addSourceNameRequirements(allocator, &required_names, derived_names.items, names);
            },
            .filter_negative_infs_column => |name| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, name);
            },
            .drop_zeros => |names| {
                if (names.len == 0) {
                    projection_blocked = true;
                    break :op_loop;
                }
                try addSourceNameRequirements(allocator, &required_names, derived_names.items, names);
            },
            .filter_zeros_column => |name| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, name);
            },
            .drop_positive_zeros => |names| {
                if (names.len == 0) {
                    projection_blocked = true;
                    break :op_loop;
                }
                try addSourceNameRequirements(allocator, &required_names, derived_names.items, names);
            },
            .filter_positive_zeros_column => |name| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, name);
            },
            .drop_negative_zeros => |names| {
                if (names.len == 0) {
                    projection_blocked = true;
                    break :op_loop;
                }
                try addSourceNameRequirements(allocator, &required_names, derived_names.items, names);
            },
            .filter_negative_zeros_column => |name| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, name);
            },
            .drop_non_zeros => |names| {
                if (names.len == 0) {
                    projection_blocked = true;
                    break :op_loop;
                }
                try addSourceNameRequirements(allocator, &required_names, derived_names.items, names);
            },
            .filter_non_zeros_column => |name| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, name);
            },
            .drop_positives => |names| {
                if (names.len == 0) {
                    projection_blocked = true;
                    break :op_loop;
                }
                try addSourceNameRequirements(allocator, &required_names, derived_names.items, names);
            },
            .filter_positives_column => |name| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, name);
            },
            .drop_signbits => |names| {
                if (names.len == 0) {
                    projection_blocked = true;
                    break :op_loop;
                }
                try addSourceNameRequirements(allocator, &required_names, derived_names.items, names);
            },
            .filter_signbits_column => |name| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, name);
            },
            .drop_negatives => |names| {
                if (names.len == 0) {
                    projection_blocked = true;
                    break :op_loop;
                }
                try addSourceNameRequirements(allocator, &required_names, derived_names.items, names);
            },
            .filter_negatives_column => |name| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, name);
            },
            .drop_finites => |names| {
                if (names.len == 0) {
                    projection_blocked = true;
                    break :op_loop;
                }
                try addSourceNameRequirements(allocator, &required_names, derived_names.items, names);
            },
            .filter_finites_column => |name| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, name);
            },
            .drop_normals => |names| {
                if (names.len == 0) {
                    projection_blocked = true;
                    break :op_loop;
                }
                try addSourceNameRequirements(allocator, &required_names, derived_names.items, names);
            },
            .filter_normals_column => |name| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, name);
            },
            .drop_subnormals => |names| {
                if (names.len == 0) {
                    projection_blocked = true;
                    break :op_loop;
                }
                try addSourceNameRequirements(allocator, &required_names, derived_names.items, names);
            },
            .filter_subnormals_column => |name| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, name);
            },
            .drop_non_finites => |names| {
                if (names.len == 0) {
                    // Empty dropNonFinites input means "consider every visible
                    // column".  This planner does not carry a full scan schema,
                    // so materialize the source rather than incorrectly
                    // treating the filter as dependency-free.
                    projection_blocked = true;
                    break :op_loop;
                }
                try addSourceNameRequirements(allocator, &required_names, derived_names.items, names);
            },
            .filter_non_finites_column => |name| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, name);
            },
            .with_row_index => |row_index| {
                try markDerivedName(allocator, &derived_names, &literal_scalars, row_index.name);
            },
            .with_column_abs,
            .with_column_neg,
            .with_column_square,
            .with_column_reciprocal,
            .with_column_sign,
            .with_column_sqrt,
            .with_column_rsqrt,
            .with_column_cbrt,
            .with_column_floor,
            .with_column_ceil,
            .with_column_round,
            .with_column_trunc,
            .with_column_deg2rad,
            .with_column_rad2deg,
            .with_column_expit,
            .with_column_logit,
            .with_column_softplus,
            .with_column_logsigmoid,
            .with_column_relu,
            .with_column_relu6,
            .with_column_tanhshrink,
            .with_column_softsign,
            .with_column_hardsigmoid,
            .with_column_hardswish,
            .with_column_silu,
            .with_column_swish,
            .with_column_mish,
            .with_column_gelu,
            .with_column_selu,
            .with_column_exp,
            .with_column_exp2,
            .with_column_expm1,
            .with_column_sin,
            .with_column_cos,
            .with_column_tan,
            .with_column_asin,
            .with_column_acos,
            .with_column_atan,
            .with_column_sinh,
            .with_column_cosh,
            .with_column_tanh,
            .with_column_asinh,
            .with_column_acosh,
            .with_column_atanh,
            .with_column_log,
            .with_column_log1p,
            .with_column_lgamma,
            .with_column_sinc,
            .with_column_log2,
            .with_column_log10,
            => |expr| {
                try addUnaryColumnOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, expr.name, expr.input_name);
            },
            inline .with_column_leaky_relu,
            .with_column_pow_scalar,
            .with_column_floor_div_scalar,
            .with_column_mod_scalar,
            .with_column_remainder_scalar,
            .with_column_log_add_exp_scalar,
            .with_column_log_add_exp2_scalar,
            .with_column_xlogy_scalar,
            .with_column_fmax_scalar,
            .with_column_fmin_scalar,
            .with_column_hypot_scalar,
            .with_column_atan2_scalar,
            .with_column_next_after_scalar,
            .with_column_copysign_scalar,
            .with_column_heaviside_scalar,
            .with_column_ldexp_scalar,
            .with_column_threshold,
            .with_column_hardtanh,
            .with_column_between,
            .with_column_maximum_scalar,
            .with_column_minimum_scalar,
            .with_column_clip_min,
            .with_column_clip_max,
            .with_column_hardshrink,
            .with_column_softshrink,
            .with_column_elu,
            .with_column_celu,
            .with_column_scalar,
            => |expr| {
                try addUnaryColumnOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, expr.name, expr.input_name);
            },
            .with_column_binary => |expr| {
                try addBinaryColumnOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, expr.name, expr.lhs_name, expr.rhs_name);
            },
            .with_column_lerp_scalar => |expr| {
                try addBinaryColumnOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, expr.name, expr.lhs_name, expr.rhs_name);
            },
            .with_column_addcmul_scalar => |expr| {
                try addTernaryColumnOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, expr.name, expr.base_name, expr.lhs_name, expr.rhs_name);
            },
            .with_column_addcdiv_scalar => |expr| {
                try addTernaryColumnOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, expr.name, expr.base_name, expr.lhs_name, expr.rhs_name);
            },
            .with_column_clip_array => |expr| {
                try addTernaryColumnOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, expr.name, expr.input_name, expr.lhs_name, expr.rhs_name);
            },
            .with_column_where => |expr| {
                try addTernaryColumnOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, expr.name, expr.input_name, expr.lhs_name, expr.rhs_name);
            },
            .with_column_where_scalar => |expr| {
                try addBinaryColumnOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, expr.name, expr.input_name, expr.mask_name);
            },
            .with_column_isin => |expr| {
                try addBinaryColumnOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, expr.name, expr.input_name, expr.test_name);
            },
            .with_column_isin_values => |expr| {
                try addUnaryColumnOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, expr.name, expr.input_name);
            },
            .with_column_masked_put_scalar => |expr| {
                try addBinaryColumnOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, expr.name, expr.input_name, expr.mask_name);
            },
            .with_column_put_flat_scalar => |expr| {
                try addUnaryColumnOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, expr.name, expr.input_name);
            },
            .with_column_put_flat => |expr| {
                try addBinaryColumnOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, expr.name, expr.input_name, expr.value_name);
            },
            .with_column_put_flat_scalar_mode => |expr| {
                try addUnaryColumnOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, expr.name, expr.input_name);
            },
            .with_column_put_flat_scalar_signed => |expr| {
                try addUnaryColumnOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, expr.name, expr.input_name);
            },
            .with_column_isclose_scalar => |expr| {
                try addUnaryColumnOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, expr.name, expr.input_name);
            },
            .with_column_logical => |expr| {
                try addBinaryColumnOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, expr.name, expr.lhs_name, expr.rhs_name);
            },
            .with_column_logical_scalar => |expr| {
                try addUnaryColumnOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, expr.name, expr.input_name);
            },
            .with_column_literal => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                try literal_scalars.put(expr.name, expr.scalar);
            },
            inline .with_column_literal_at, .with_column_literal_before, .with_column_literal_after => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                // Positioned insertion/replacement is a schema rewrite.  Keep
                // execution order intact rather than inferring projection
                // through the changed column order.
                projection_blocked = true;
                break :op_loop;
            },
            .cast_column => |cast| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, cast.name);
            },
            .fill_null_column => |fill| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, fill.name);
            },
            .fill_nan_column => |fill| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, fill.name);
            },
            .fill_inf_column => |fill| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, fill.name);
            },
            .fill_positive_inf_column => |fill| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, fill.name);
            },
            .fill_negative_inf_column => |fill| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, fill.name);
            },
            .fill_zero_column => |fill| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, fill.name);
            },
            .fill_positive_zero_column => |fill| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, fill.name);
            },
            .fill_negative_zero_column => |fill| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, fill.name);
            },
            .fill_non_zero_column => |fill| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, fill.name);
            },
            .fill_positive_column => |fill| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, fill.name);
            },
            .fill_signbit_column => |fill| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, fill.name);
            },
            .fill_negative_column => |fill| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, fill.name);
            },
            .fill_finite_column => |fill| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, fill.name);
            },
            .fill_normal_column => |fill| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, fill.name);
            },
            .fill_subnormal_column => |fill| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, fill.name);
            },
            .fill_non_finite_column => |fill| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, fill.name);
            },
            .fill_null_forward_column, .fill_null_backward_column => |name| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, name);
            },
            .null_if_column => |fill| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, fill.name);
            },
            .null_if_values_column => |null_if| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, null_if.name);
            },
            .null_if_nan_column, .null_if_inf_column, .null_if_positive_inf_column, .null_if_negative_inf_column, .null_if_zero_column, .null_if_positive_zero_column, .null_if_negative_zero_column, .null_if_non_zero_column, .null_if_positive_column, .null_if_signbit_column, .null_if_negative_column, .null_if_finite_column, .null_if_normal_column, .null_if_subnormal_column, .null_if_non_finite_column => |name| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, name);
            },
            .coalesce_columns => |coalesce| {
                try addBinaryColumnOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, coalesce.output_name, coalesce.primary_name, coalesce.fallback_name);
            },
            .coalesce_columns_many => |coalesce| {
                try addListColumnOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, coalesce.output_name, coalesce.names);
            },
            .is_null_column, .is_valid_column, .is_nan_column, .is_zero_column, .is_positive_zero_column, .is_negative_zero_column, .is_non_zero_column, .is_positive_column, .is_signbit_column, .is_negative_column, .is_finite_column, .is_normal_column, .is_subnormal_column, .is_non_finite_column, .is_inf_column, .is_positive_inf_column, .is_negative_inf_column => |predicate| {
                try addUnaryColumnOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, predicate.output_name, predicate.name);
            },
            .row_null_count, .row_valid_count, .row_any_null, .row_all_null, .row_any_valid, .row_all_valid, .row_null_ratio, .row_valid_ratio, .row_first_valid_index, .row_last_valid_index, .row_first_null_index, .row_last_null_index, .row_argmin, .row_argmax, .row_median, .row_iqr, .row_interdecile_range, .row_midhinge, .row_trimean, .row_bowley_skewness, .row_quartile_coeff_dispersion, .row_kelley_skewness, .row_mad, .row_mode, .row_entropy, .row_gini_impurity, .row_perplexity, .row_inverse_simpson, .row_simpson_concentration, .row_evenness, .row_mode_count, .row_mode_ratio, .row_mode_margin, .row_mode_margin_ratio, .row_count_distinct, .row_n_unique, .row_is_duplicated, .row_is_unique, .row_sum, .row_mean, .row_logsumexp, .row_logmeanexp, .row_softmax_entropy, .row_softmax_perplexity, .row_softmax_confidence, .row_softmax_margin, .row_softmax_evenness, .row_softmax_concentration, .row_softmax_normalized_hhi, .row_softmax_gini_impurity, .row_softmax_inverse_simpson, .row_softmax_simpson_evenness, .row_logit_margin, .row_geometric_mean, .row_magnitude_geometric_mean, .row_harmonic_mean, .row_skewness, .row_magnitude_skewness, .row_kurtosis, .row_magnitude_kurtosis, .row_prod, .row_min, .row_max, .row_ptp, .row_magnitude_ptp, .row_midrange, .row_magnitude_midrange, .row_range_coeff, .row_magnitude_range_coeff, .row_mean_abs, .row_hhi, .row_magnitude_normalized_hhi, .row_magnitude_sparsity, .row_magnitude_inverse_simpson, .row_magnitude_simpson_evenness, .row_magnitude_dominance, .row_magnitude_dominance_margin, .row_magnitude_entropy, .row_magnitude_perplexity, .row_magnitude_evenness, .row_mean_abs_dev, .row_gini_mean_diff, .row_gini_coefficient, .row_mean_abs_dev_ratio, .row_rms, .row_l1_norm, .row_l2_norm, .row_true_count, .row_false_count, .row_any_true, .row_all_true, .row_any_false, .row_all_false, .row_first_true_index, .row_last_true_index, .row_first_false_index, .row_last_false_index, .row_true_ratio, .row_false_ratio, .row_any_zero, .row_all_zero, .row_any_non_zero, .row_all_non_zero, .row_any_positive_zero, .row_all_positive_zero, .row_any_negative_zero, .row_all_negative_zero, .row_any_positive, .row_all_positive, .row_any_signbit, .row_all_signbit, .row_any_negative, .row_all_negative, .row_any_nan, .row_all_nan, .row_any_inf, .row_all_inf, .row_any_positive_inf, .row_all_positive_inf, .row_any_negative_inf, .row_all_negative_inf, .row_any_finite, .row_all_finite, .row_any_normal, .row_all_normal, .row_any_subnormal, .row_all_subnormal, .row_any_non_finite, .row_all_non_finite, .row_nan_count, .row_nan_ratio, .row_inf_count, .row_inf_ratio, .row_positive_inf_count, .row_negative_inf_count, .row_positive_inf_ratio, .row_negative_inf_ratio, .row_zero_count, .row_zero_ratio, .row_positive_zero_count, .row_negative_zero_count, .row_positive_zero_ratio, .row_negative_zero_ratio, .row_non_zero_count, .row_non_zero_ratio, .row_first_nan_index, .row_last_nan_index, .row_first_inf_index, .row_last_inf_index, .row_first_positive_inf_index, .row_last_positive_inf_index, .row_first_negative_inf_index, .row_last_negative_inf_index, .row_first_finite_index, .row_last_finite_index, .row_first_normal_index, .row_last_normal_index, .row_first_subnormal_index, .row_last_subnormal_index, .row_first_non_finite_index, .row_last_non_finite_index, .row_first_positive_zero_index, .row_last_positive_zero_index, .row_first_negative_zero_index, .row_last_negative_zero_index, .row_first_signbit_index, .row_last_signbit_index, .row_first_zero_index, .row_last_zero_index, .row_first_non_zero_index, .row_last_non_zero_index, .row_first_positive_index, .row_last_positive_index, .row_first_negative_index, .row_last_negative_index, .row_positive_count, .row_positive_ratio, .row_signbit_count, .row_signbit_ratio, .row_negative_count, .row_negative_ratio, .row_finite_count, .row_finite_ratio, .row_normal_count, .row_normal_ratio, .row_subnormal_count, .row_subnormal_ratio, .row_non_finite_count, .row_non_finite_ratio => |row_count| {
                if (!(try addRowSingleOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, &projection_blocked, row_count.names, row_count.output_name))) break :op_loop;
            },
            .row_cumulative_argmin, .row_cumulative_argmax, .row_cumulative_mode, .row_cumulative_mode_count, .row_cumulative_mode_ratio, .row_cumulative_mode_margin, .row_cumulative_mode_margin_ratio, .row_cumulative_distinct_count, .row_cumulative_n_unique, .row_cumulative_first_true_index, .row_cumulative_last_true_index, .row_cumulative_first_false_index, .row_cumulative_last_false_index, .row_cumulative_first_valid_index, .row_cumulative_last_valid_index, .row_cumulative_first_null_index, .row_cumulative_last_null_index, .row_cumulative_null_count, .row_cumulative_valid_count, .row_cumulative_any_null, .row_cumulative_all_null, .row_cumulative_any_valid, .row_cumulative_all_valid, .row_cumulative_null_ratio, .row_cumulative_valid_ratio, .row_cumulative_true_count, .row_cumulative_false_count, .row_cumulative_true_ratio, .row_cumulative_false_ratio, .row_cumulative_positive_zero_count, .row_cumulative_negative_zero_count, .row_cumulative_signbit_count, .row_cumulative_positive_zero_ratio, .row_cumulative_negative_zero_ratio, .row_cumulative_signbit_ratio, .row_cumulative_nan_count, .row_cumulative_inf_count, .row_cumulative_positive_inf_count, .row_cumulative_negative_inf_count, .row_cumulative_finite_count, .row_cumulative_normal_count, .row_cumulative_subnormal_count, .row_cumulative_non_finite_count, .row_cumulative_nan_ratio, .row_cumulative_inf_ratio, .row_cumulative_positive_inf_ratio, .row_cumulative_negative_inf_ratio, .row_cumulative_finite_ratio, .row_cumulative_normal_ratio, .row_cumulative_subnormal_ratio, .row_cumulative_non_finite_ratio, .row_cumulative_any_zero, .row_cumulative_all_zero, .row_cumulative_any_non_zero, .row_cumulative_all_non_zero, .row_cumulative_any_positive_zero, .row_cumulative_all_positive_zero, .row_cumulative_any_negative_zero, .row_cumulative_all_negative_zero, .row_cumulative_any_positive, .row_cumulative_all_positive, .row_cumulative_any_signbit, .row_cumulative_all_signbit, .row_cumulative_any_negative, .row_cumulative_all_negative, .row_cumulative_any_nan, .row_cumulative_all_nan, .row_cumulative_any_inf, .row_cumulative_all_inf, .row_cumulative_any_positive_inf, .row_cumulative_all_positive_inf, .row_cumulative_any_negative_inf, .row_cumulative_all_negative_inf, .row_cumulative_any_finite, .row_cumulative_all_finite, .row_cumulative_any_normal, .row_cumulative_all_normal, .row_cumulative_any_subnormal, .row_cumulative_all_subnormal, .row_cumulative_any_non_finite, .row_cumulative_all_non_finite, .row_cumulative_first_nan_index, .row_cumulative_last_nan_index, .row_cumulative_first_inf_index, .row_cumulative_last_inf_index, .row_cumulative_first_positive_inf_index, .row_cumulative_last_positive_inf_index, .row_cumulative_first_negative_inf_index, .row_cumulative_last_negative_inf_index, .row_cumulative_first_finite_index, .row_cumulative_last_finite_index, .row_cumulative_first_normal_index, .row_cumulative_last_normal_index, .row_cumulative_first_subnormal_index, .row_cumulative_last_subnormal_index, .row_cumulative_first_non_finite_index, .row_cumulative_last_non_finite_index, .row_cumulative_zero_count, .row_cumulative_first_zero_index, .row_cumulative_last_zero_index, .row_cumulative_first_positive_zero_index, .row_cumulative_last_positive_zero_index, .row_cumulative_first_negative_zero_index, .row_cumulative_last_negative_zero_index, .row_cumulative_non_zero_count, .row_cumulative_first_non_zero_index, .row_cumulative_last_non_zero_index, .row_cumulative_first_positive_index, .row_cumulative_last_positive_index, .row_cumulative_first_signbit_index, .row_cumulative_last_signbit_index, .row_cumulative_first_negative_index, .row_cumulative_last_negative_index, .row_cumulative_positive_count, .row_cumulative_negative_count, .row_cumulative_zero_ratio, .row_cumulative_non_zero_ratio, .row_cumulative_positive_ratio, .row_cumulative_negative_ratio, .row_cumulative_any_true, .row_cumulative_all_true, .row_cumulative_any_false, .row_cumulative_all_false, .row_centered, .row_zscore, .row_robust_zscore, .row_average_rank, .row_ordinal_rank, .row_dense_rank, .row_competition_rank, .row_percent_rank, .row_cume_dist, .row_cumulative_sum, .row_cumulative_mean, .row_cumulative_logsumexp, .row_cumulative_logmeanexp, .row_cumulative_geometric_mean, .row_cumulative_harmonic_mean, .row_cumulative_skewness, .row_cumulative_kurtosis, .row_cumulative_rms, .row_cumulative_mean_abs, .row_cumulative_mean_square, .row_cumulative_max_abs, .row_cumulative_min_abs, .row_cumulative_l1_norm, .row_cumulative_l2_norm, .row_cumulative_product, .row_cumulative_max, .row_cumulative_min, .row_cumulative_range, .row_iqr_outlier, .row_tukey_winsorize, .row_max_indicator, .row_min_indicator, .row_minmax_scale, .row_l2_normalize, .row_l1_normalize, .row_sum_normalize, .row_mean_normalize, .row_max_abs_normalize, .row_softmax, .row_log_softmax, .row_softmin, .row_log_softmin => |row_outputs| {
                if (!(try addRowMultiOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, &projection_blocked, row_outputs.names, row_outputs.output_names))) break :op_loop;
            },
            .row_cumulative_variance, .row_cumulative_stddev, .row_cumulative_sem, .row_cumulative_cv, .row_cumulative_fano => |row_outputs| {
                if (!(try addRowMultiOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, &projection_blocked, row_outputs.names, row_outputs.output_names))) break :op_loop;
            },
            .row_variance, .row_magnitude_variance, .row_stddev, .row_magnitude_stddev, .row_sem, .row_magnitude_sem, .row_cv, .row_magnitude_cv, .row_magnitude_fano, .row_fano => |row_dispersion| {
                if (!(try addRowSingleOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, &projection_blocked, row_dispersion.names, row_dispersion.output_name))) break :op_loop;
            },
            .row_quantile => |row_quantile| {
                if (!(try addRowSingleOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, &projection_blocked, row_quantile.names, row_quantile.output_name))) break :op_loop;
            },
            .row_quantile_range => |row_quantile_range| {
                if (!(try addRowSingleOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, &projection_blocked, row_quantile_range.names, row_quantile_range.output_name))) break :op_loop;
            },
            .row_trimmed_mean => |row_trimmed_mean| {
                if (!(try addRowSingleOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, &projection_blocked, row_trimmed_mean.names, row_trimmed_mean.output_name))) break :op_loop;
            },
            .row_winsorized_mean => |row_winsorized_mean| {
                if (!(try addRowSingleOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, &projection_blocked, row_winsorized_mean.names, row_winsorized_mean.output_name))) break :op_loop;
            },
            .row_cumulative_weighted_sum, .row_cumulative_weighted_mean, .row_cumulative_weighted_mean_square, .row_cumulative_weighted_rms, .row_cumulative_weighted_mean_abs, .row_cumulative_weighted_l1_norm, .row_cumulative_weighted_l2_norm, .row_cumulative_weighted_min, .row_cumulative_weighted_max, .row_cumulative_weighted_max_abs, .row_cumulative_weighted_min_abs, .row_cumulative_weighted_range, .row_cumulative_weighted_midrange, .row_cumulative_weighted_range_coeff, .row_cumulative_weighted_product, .row_cumulative_weighted_geometric_mean, .row_cumulative_weighted_harmonic_mean, .row_cumulative_weighted_logsumexp, .row_cumulative_weighted_logmeanexp, .row_cumulative_weighted_skewness, .row_cumulative_weighted_kurtosis, .row_cumulative_weighted_median, .row_cumulative_weighted_iqr, .row_cumulative_weighted_mad, .row_cumulative_weighted_interdecile_range, .row_cumulative_weighted_midhinge, .row_cumulative_weighted_trimean, .row_cumulative_weighted_bowley_skewness, .row_cumulative_weighted_quartile_coeff_dispersion, .row_cumulative_weighted_kelley_skewness, .row_cumulative_weighted_mode, .row_cumulative_weighted_mode_weight, .row_cumulative_weighted_mode_ratio, .row_cumulative_weighted_mode_margin, .row_cumulative_weighted_mode_margin_ratio, .row_cumulative_weighted_entropy, .row_cumulative_weighted_gini_impurity, .row_cumulative_weighted_perplexity, .row_cumulative_weighted_inverse_simpson, .row_cumulative_weighted_simpson_concentration, .row_cumulative_weighted_evenness, .row_cumulative_weighted_mean_abs_dev, .row_cumulative_weighted_mean_abs_dev_ratio, .row_cumulative_weighted_gini_mean_diff, .row_cumulative_weighted_gini_coefficient, .row_cumulative_weighted_weight_sum, .row_cumulative_weighted_positive_count, .row_cumulative_weighted_effective_n => |row_weighted_outputs| {
                if (!(try addWeightedRowMultiOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, &projection_blocked, row_weighted_outputs.value_names, row_weighted_outputs.weight_names, row_weighted_outputs.output_names))) break :op_loop;
            },
            .row_cumulative_weighted_quantile, .row_cumulative_weighted_trimmed_mean, .row_cumulative_weighted_winsorized_mean => |row_weighted_outputs| {
                if (!(try addWeightedRowMultiOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, &projection_blocked, row_weighted_outputs.value_names, row_weighted_outputs.weight_names, row_weighted_outputs.output_names))) break :op_loop;
            },
            .row_cumulative_weighted_variance, .row_cumulative_weighted_stddev, .row_cumulative_weighted_sem, .row_cumulative_weighted_cv, .row_cumulative_weighted_fano => |row_weighted_outputs| {
                if (!(try addWeightedRowMultiOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, &projection_blocked, row_weighted_outputs.value_names, row_weighted_outputs.weight_names, row_weighted_outputs.output_names))) break :op_loop;
            },
            .row_pair_count, .row_weighted_mean, .row_weighted_sum, .row_weighted_weight_sum, .row_weighted_positive_count, .row_weighted_effective_n, .row_weighted_mean_square, .row_weighted_rms, .row_weighted_mean_abs, .row_weighted_l1_norm, .row_weighted_l2_norm, .row_weighted_min, .row_weighted_max, .row_weighted_max_abs, .row_weighted_min_abs, .row_weighted_range, .row_weighted_midrange, .row_weighted_range_coeff, .row_weighted_product, .row_weighted_geometric_mean, .row_weighted_harmonic_mean, .row_weighted_logsumexp, .row_weighted_logmeanexp, .row_weighted_median, .row_weighted_iqr, .row_weighted_mad, .row_weighted_interdecile_range, .row_weighted_midhinge, .row_weighted_trimean, .row_weighted_bowley_skewness, .row_weighted_quartile_coeff_dispersion, .row_weighted_kelley_skewness, .row_weighted_mode, .row_weighted_mode_weight, .row_weighted_mode_ratio, .row_weighted_mode_margin, .row_weighted_mode_margin_ratio, .row_weighted_entropy, .row_weighted_gini_impurity, .row_weighted_perplexity, .row_weighted_inverse_simpson, .row_weighted_simpson_concentration, .row_weighted_evenness, .row_weighted_mean_abs_dev, .row_weighted_mean_abs_dev_ratio, .row_weighted_gini_mean_diff, .row_weighted_gini_coefficient, .row_weighted_skewness, .row_weighted_kurtosis, .row_dot, .row_cosine_similarity, .row_squared_euclidean_distance, .row_euclidean_distance, .row_manhattan_distance, .row_chebyshev_distance, .row_canberra_distance, .row_bray_curtis_distance, .row_mean_error, .row_mae, .row_mse, .row_rmse, .row_mape, .row_smape, .row_covariance, .row_correlation, .row_beta => |row_weighted| {
                if (!(try addWeightedRowSingleOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, &projection_blocked, row_weighted.value_names, row_weighted.weight_names, row_weighted.output_name))) break :op_loop;
            },
            .row_weighted_variance, .row_weighted_stddev, .row_weighted_sem, .row_weighted_cv, .row_weighted_fano => |row_weighted| {
                if (!(try addWeightedRowSingleOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, &projection_blocked, row_weighted.value_names, row_weighted.weight_names, row_weighted.output_name))) break :op_loop;
            },
            .row_weighted_quantile, .row_weighted_trimmed_mean, .row_weighted_winsorized_mean => |row_weighted| {
                if (!(try addWeightedRowSingleOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, &projection_blocked, row_weighted.value_names, row_weighted.weight_names, row_weighted.output_name))) break :op_loop;
            },
            .row_weighted_pair_weight_sum, .row_weighted_pair_positive_count, .row_weighted_pair_effective_n, .row_weighted_dot, .row_weighted_cosine_similarity, .row_weighted_squared_euclidean_distance, .row_weighted_euclidean_distance, .row_weighted_manhattan_distance, .row_weighted_chebyshev_distance, .row_weighted_canberra_distance, .row_weighted_bray_curtis_distance, .row_weighted_mean_error, .row_weighted_mae, .row_weighted_mse, .row_weighted_rmse, .row_weighted_mape, .row_weighted_smape, .row_weighted_covariance, .row_weighted_correlation, .row_weighted_beta => |row_weighted| {
                if (!(try addWeightedPairRowSingleOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, &projection_blocked, row_weighted.lhs_names, row_weighted.rhs_names, row_weighted.weight_names, row_weighted.output_name))) break :op_loop;
            },
            .row_cumulative_weighted_pair_weight_sum, .row_cumulative_weighted_pair_positive_count, .row_cumulative_weighted_pair_effective_n, .row_cumulative_weighted_dot, .row_cumulative_weighted_cosine_similarity, .row_cumulative_weighted_squared_euclidean_distance, .row_cumulative_weighted_euclidean_distance, .row_cumulative_weighted_manhattan_distance, .row_cumulative_weighted_chebyshev_distance, .row_cumulative_weighted_canberra_distance, .row_cumulative_weighted_bray_curtis_distance, .row_cumulative_weighted_mean_error, .row_cumulative_weighted_mae, .row_cumulative_weighted_mse, .row_cumulative_weighted_rmse, .row_cumulative_weighted_mape, .row_cumulative_weighted_smape => |row_weighted| try addRowWeightedPairColumnOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, &projection_blocked, row_weighted),
            .row_cumulative_weighted_covariance => |row_weighted| try addRowWeightedPairColumnOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, &projection_blocked, row_weighted),
            .with_column_compare => |expr| {
                try addBinaryColumnOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, expr.name, expr.lhs_name, expr.rhs_name);
            },
            .with_column_compare_scalar => |expr| {
                try addUnaryColumnOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, expr.name, expr.input_name);
            },
            .group_id => |row_count| {
                if (!(try addRowSingleOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, &projection_blocked, row_count.names, row_count.output_name))) break :op_loop;
            },
            .group_first_row_index => |row_count| {
                if (!(try addRowSingleOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, &projection_blocked, row_count.names, row_count.output_name))) break :op_loop;
            },
            .group_last_row_index => |row_count| {
                if (!(try addRowSingleOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, &projection_blocked, row_count.names, row_count.output_name))) break :op_loop;
            },
            .group_is_first_row => |row_count| {
                if (!(try addRowSingleOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, &projection_blocked, row_count.names, row_count.output_name))) break :op_loop;
            },
            .group_is_last_row => |row_count| {
                if (!(try addRowSingleOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, &projection_blocked, row_count.names, row_count.output_name))) break :op_loop;
            },
            .group_is_singleton => |row_count| {
                if (!(try addRowSingleOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, &projection_blocked, row_count.names, row_count.output_name))) break :op_loop;
            },
            .group_is_duplicated => |row_count| {
                if (!(try addRowSingleOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, &projection_blocked, row_count.names, row_count.output_name))) break :op_loop;
            },
            .group_cume_dist => |row_count| {
                if (!(try addRowSingleOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, &projection_blocked, row_count.names, row_count.output_name))) break :op_loop;
            },
            .group_percent_rank => |row_count| {
                if (!(try addRowSingleOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, &projection_blocked, row_count.names, row_count.output_name))) break :op_loop;
            },
            .group_reverse_cume_dist => |row_count| {
                if (!(try addRowSingleOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, &projection_blocked, row_count.names, row_count.output_name))) break :op_loop;
            },
            .group_reverse_percent_rank => |row_count| {
                if (!(try addRowSingleOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, &projection_blocked, row_count.names, row_count.output_name))) break :op_loop;
            },
            .group_lag, .group_lead, .group_first_row_value, .group_last_row_value, .group_nth_row_value, .group_first_valid_value, .group_last_valid_value, .group_nth_valid_value, .group_fill_null_forward, .group_fill_null_backward, .group_cumulative_valid_count, .group_cumulative_null_count, .group_cumulative_valid_ratio, .group_cumulative_null_ratio, .group_cumulative_first_valid_index, .group_cumulative_last_valid_index, .group_cumulative_first_null_index, .group_cumulative_last_null_index, .group_cumulative_nan_count, .group_cumulative_nan_ratio, .group_cumulative_inf_count, .group_cumulative_inf_ratio, .group_cumulative_positive_inf_count, .group_cumulative_positive_inf_ratio, .group_cumulative_negative_inf_count, .group_cumulative_negative_inf_ratio, .group_cumulative_finite_count, .group_cumulative_finite_ratio, .group_cumulative_normal_count, .group_cumulative_normal_ratio, .group_cumulative_subnormal_count, .group_cumulative_subnormal_ratio, .group_cumulative_non_finite_count, .group_cumulative_non_finite_ratio, .group_cumulative_zero_count, .group_cumulative_zero_ratio, .group_cumulative_positive_zero_count, .group_cumulative_positive_zero_ratio, .group_cumulative_negative_zero_count, .group_cumulative_negative_zero_ratio, .group_cumulative_non_zero_count, .group_cumulative_non_zero_ratio, .group_cumulative_positive_count, .group_cumulative_positive_ratio, .group_cumulative_signbit_count, .group_cumulative_signbit_ratio, .group_cumulative_negative_count, .group_cumulative_negative_ratio, .group_cumulative_first_nan_index, .group_cumulative_last_nan_index, .group_cumulative_first_inf_index, .group_cumulative_last_inf_index, .group_cumulative_first_positive_inf_index, .group_cumulative_last_positive_inf_index, .group_cumulative_first_negative_inf_index, .group_cumulative_last_negative_inf_index, .group_cumulative_first_finite_index, .group_cumulative_last_finite_index, .group_cumulative_first_normal_index, .group_cumulative_last_normal_index, .group_cumulative_first_subnormal_index, .group_cumulative_last_subnormal_index, .group_cumulative_first_non_finite_index, .group_cumulative_last_non_finite_index, .group_cumulative_first_zero_index, .group_cumulative_last_zero_index, .group_cumulative_first_positive_zero_index, .group_cumulative_last_positive_zero_index, .group_cumulative_first_negative_zero_index, .group_cumulative_last_negative_zero_index, .group_cumulative_first_non_zero_index, .group_cumulative_last_non_zero_index, .group_cumulative_first_positive_index, .group_cumulative_last_positive_index, .group_cumulative_first_signbit_index, .group_cumulative_last_signbit_index, .group_cumulative_first_negative_index, .group_cumulative_last_negative_index, .group_cumulative_distinct_count, .group_cumulative_n_unique, .group_cumulative_mode, .group_cumulative_mode_count, .group_cumulative_mode_ratio, .group_cumulative_mode_margin, .group_cumulative_mode_margin_ratio, .group_cumulative_entropy, .group_cumulative_gini_impurity, .group_cumulative_perplexity, .group_cumulative_inverse_simpson, .group_cumulative_simpson_concentration, .group_cumulative_evenness, .group_cumulative_mean_abs_dev, .group_cumulative_mean_abs_dev_ratio, .group_cumulative_gini_mean_diff, .group_cumulative_gini_coefficient, .group_cumulative_median, .group_cumulative_iqr, .group_cumulative_mad, .group_cumulative_interdecile_range, .group_cumulative_midhinge, .group_cumulative_trimean, .group_cumulative_bowley_skewness, .group_cumulative_quartile_coeff_dispersion, .group_cumulative_kelley_skewness, .group_cumulative_any, .group_cumulative_all, .group_cumulative_true_count, .group_cumulative_false_count, .group_cumulative_true_ratio, .group_cumulative_false_ratio, .group_cumulative_first_true_index, .group_cumulative_last_true_index, .group_cumulative_first_false_index, .group_cumulative_last_false_index, .group_cumulative_sum, .group_cumulative_mean, .group_cumulative_product, .group_cumulative_min, .group_cumulative_max, .group_cumulative_variance, .group_cumulative_stddev, .group_cumulative_sem, .group_cumulative_cv, .group_cumulative_fano, .group_cumulative_skewness, .group_cumulative_kurtosis, .group_cumulative_mean_abs, .group_cumulative_mean_square, .group_cumulative_rms, .group_cumulative_max_abs, .group_cumulative_min_abs, .group_cumulative_l1_norm, .group_cumulative_l2_norm, .group_cumulative_range, .group_cumulative_midrange, .group_cumulative_range_coeff, .group_cumulative_logsumexp, .group_cumulative_logmeanexp, .group_cumulative_geometric_mean, .group_cumulative_harmonic_mean, .group_cumulative_argmin, .group_cumulative_argmax => |shift| {
                if (!(try addGroupedValueOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, &projection_blocked, shift.names, shift.value_name, shift.output_name))) break :op_loop;
            },
            .group_cumulative_quantile, .group_cumulative_trimmed_mean, .group_cumulative_winsorized_mean => |shift| {
                if (!(try addGroupedValueOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, &projection_blocked, shift.names, shift.value_name, shift.output_name))) break :op_loop;
            },
            .group_cumulative_weighted_sum, .group_cumulative_weighted_product, .group_cumulative_weighted_weight_sum, .group_cumulative_weighted_positive_count, .group_cumulative_weighted_effective_n, .group_cumulative_weighted_mean, .group_cumulative_weighted_mean_square, .group_cumulative_weighted_rms, .group_cumulative_weighted_min, .group_cumulative_weighted_max, .group_cumulative_weighted_median, .group_cumulative_weighted_iqr, .group_cumulative_weighted_mad, .group_cumulative_weighted_interdecile_range, .group_cumulative_weighted_midhinge, .group_cumulative_weighted_trimean, .group_cumulative_weighted_bowley_skewness, .group_cumulative_weighted_quartile_coeff_dispersion, .group_cumulative_weighted_kelley_skewness, .group_cumulative_weighted_mode, .group_cumulative_weighted_mode_weight, .group_cumulative_weighted_mode_ratio, .group_cumulative_weighted_mode_margin, .group_cumulative_weighted_mode_margin_ratio, .group_cumulative_weighted_entropy, .group_cumulative_weighted_gini_impurity, .group_cumulative_weighted_perplexity, .group_cumulative_weighted_inverse_simpson, .group_cumulative_weighted_simpson_concentration, .group_cumulative_weighted_evenness, .group_cumulative_weighted_mean_abs_dev, .group_cumulative_weighted_mean_abs_dev_ratio, .group_cumulative_weighted_gini_mean_diff, .group_cumulative_weighted_gini_coefficient, .group_cumulative_weighted_mean_abs, .group_cumulative_weighted_l1_norm, .group_cumulative_weighted_l2_norm, .group_cumulative_weighted_max_abs, .group_cumulative_weighted_min_abs, .group_cumulative_weighted_geometric_mean, .group_cumulative_weighted_harmonic_mean, .group_cumulative_weighted_logsumexp, .group_cumulative_weighted_logmeanexp, .group_cumulative_weighted_range, .group_cumulative_weighted_midrange, .group_cumulative_weighted_range_coeff, .group_cumulative_weighted_variance, .group_cumulative_weighted_stddev, .group_cumulative_weighted_sem, .group_cumulative_weighted_cv, .group_cumulative_weighted_fano, .group_cumulative_weighted_skewness, .group_cumulative_weighted_kurtosis => |shift| {
                if (!(try addGroupedWeightedValueOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, &projection_blocked, shift.names, shift.value_name, shift.weight_name, shift.output_name))) break :op_loop;
            },
            .group_cumulative_weighted_quantile, .group_cumulative_weighted_trimmed_mean, .group_cumulative_weighted_winsorized_mean => |shift| {
                if (!(try addGroupedWeightedValueOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, &projection_blocked, shift.names, shift.value_name, shift.weight_name, shift.output_name))) break :op_loop;
            },
            .group_cumulative_weighted_dot, .group_cumulative_weighted_cosine_similarity, .group_cumulative_weighted_squared_euclidean_distance, .group_cumulative_weighted_euclidean_distance, .group_cumulative_weighted_manhattan_distance, .group_cumulative_weighted_chebyshev_distance, .group_cumulative_weighted_canberra_distance, .group_cumulative_weighted_bray_curtis_distance, .group_cumulative_weighted_mean_error, .group_cumulative_weighted_mae, .group_cumulative_weighted_mse, .group_cumulative_weighted_rmse, .group_cumulative_weighted_mape, .group_cumulative_weighted_smape, .group_cumulative_weighted_covariance, .group_cumulative_weighted_correlation, .group_cumulative_weighted_beta => |shift| {
                if (!(try addGroupedWeightedPairOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, &projection_blocked, shift.names, shift.lhs_name, shift.rhs_name, shift.weight_name, shift.output_name))) break :op_loop;
            },
            .group_row_number => |row_count| {
                if (!(try addRowSingleOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, &projection_blocked, row_count.names, row_count.output_name))) break :op_loop;
            },
            .group_size => |row_count| {
                if (!(try addRowSingleOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, &projection_blocked, row_count.names, row_count.output_name))) break :op_loop;
            },
            .group_reverse_row_number => |row_count| {
                if (!(try addRowSingleOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, &projection_blocked, row_count.names, row_count.output_name))) break :op_loop;
            },
            .group_by_count => |group| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, group.key_name);
                saw_select = true;
                break :op_loop;
            },
            .group_by_count_on => |group| {
                try addSourceNameRequirements(allocator, &required_names, derived_names.items, group.key_names);
                saw_select = true;
                break :op_loop;
            },
            .group_by_rows => |group| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, group.key_name);
                projection_blocked = true;
                saw_select = true;
                break :op_loop;
            },
            .group_by_rows_on => |group| {
                try addSourceNameRequirements(allocator, &required_names, derived_names.items, group.key_names);
                projection_blocked = true;
                saw_select = true;
                break :op_loop;
            },
            .group_by_sorted_rows => |group| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, group.key_name);
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, group.sort_name);
                projection_blocked = true;
                saw_select = true;
                break :op_loop;
            },
            .group_by_sorted_rows_on => |group| {
                try addSourceNameRequirements(allocator, &required_names, derived_names.items, group.key_names);
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, group.sort_name);
                projection_blocked = true;
                saw_select = true;
                break :op_loop;
            },
            .group_by_sorted_rows_columns => |group| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, group.key_name);
                try addSourceNameRequirements(allocator, &required_names, derived_names.items, group.sort_names);
                projection_blocked = true;
                saw_select = true;
                break :op_loop;
            },
            .group_by_sorted_rows_columns_on => |group| {
                try addSourceNameRequirements(allocator, &required_names, derived_names.items, group.key_names);
                try addSourceNameRequirements(allocator, &required_names, derived_names.items, group.sort_names);
                projection_blocked = true;
                saw_select = true;
                break :op_loop;
            },
            .group_by_value => |group| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, group.key_name);
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, group.value_name);
                saw_select = true;
                break :op_loop;
            },
            .group_by_value_on => |group| {
                try addSourceNameRequirements(allocator, &required_names, derived_names.items, group.key_names);
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, group.value_name);
                saw_select = true;
                break :op_loop;
            },
            .group_by_weighted => |group| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, group.key_name);
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, group.value_name);
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, group.weight_name);
                saw_select = true;
                break :op_loop;
            },
            .group_by_weighted_on => |group| {
                try addSourceNameRequirements(allocator, &required_names, derived_names.items, group.key_names);
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, group.value_name);
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, group.weight_name);
                saw_select = true;
                break :op_loop;
            },
            .group_by_pair => |group| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, group.key_name);
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, group.lhs_name);
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, group.rhs_name);
                saw_select = true;
                break :op_loop;
            },
            .group_by_pair_on => |group| {
                try addSourceNameRequirements(allocator, &required_names, derived_names.items, group.key_names);
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, group.lhs_name);
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, group.rhs_name);
                saw_select = true;
                break :op_loop;
            },
            .group_by_weighted_pair => |group| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, group.key_name);
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, group.lhs_name);
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, group.rhs_name);
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, group.weight_name);
                saw_select = true;
                break :op_loop;
            },
            .group_by_weighted_pair_on => |group| {
                try addSourceNameRequirements(allocator, &required_names, derived_names.items, group.key_names);
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, group.lhs_name);
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, group.rhs_name);
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, group.weight_name);
                saw_select = true;
                break :op_loop;
            },
            .group_by_stats => |group| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, group.key_name);
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, group.value_name);
                saw_select = true;
                break :op_loop;
            },
            .group_by_stats_on => |group| {
                try addSourceNameRequirements(allocator, &required_names, derived_names.items, group.key_names);
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, group.value_name);
                saw_select = true;
                break :op_loop;
            },
            .group_by_profile => |group| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, group.key_name);
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, group.value_name);
                saw_select = true;
                break :op_loop;
            },
            .group_by_profile_on => |group| {
                try addSourceNameRequirements(allocator, &required_names, derived_names.items, group.key_names);
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, group.value_name);
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
                try addSourceNameRequirements(allocator, &required_names, derived_names.items, join.left_key_names);
                break :op_loop;
            },
            .asof_join => |join| {
                projection_blocked = true;
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, join.left_key_name);
                break :op_loop;
            },
            .concat_rows => {
                break :op_loop;
            },
            .concat_columns => {
                projection_blocked = true;
                break :op_loop;
            },
            .distinct_rows => {
                projection_blocked = true;
            },
            .distinct_rows_last => {
                projection_blocked = true;
            },
            .distinct_rows_none => {
                projection_blocked = true;
            },
            .distinct_on => |names| {
                try addSourceNameRequirements(allocator, &required_names, derived_names.items, names);
            },
            .distinct_on_last => |names| {
                try addSourceNameRequirements(allocator, &required_names, derived_names.items, names);
            },
            .distinct_on_none => |names| {
                try addSourceNameRequirements(allocator, &required_names, derived_names.items, names);
            },
            .filter_column => |name| {
                if (!nameInBorrowedList(name, derived_names.items)) {
                    try addSourceNameRequirement(allocator, &required_names, derived_names.items, name);
                    try mergeRangePredicate(allocator, &range_predicate, name, .{ .bool = .{ .min = true, .max = true } });
                    clearNullPredicate(allocator, &null_predicate);
                }
            },
            .filter_between_column => |range| {
                const filter_depends_on_source = !nameInBorrowedList(range.name, derived_names.items);
                if (filter_depends_on_source) {
                    try addSourceNameRequirement(allocator, &required_names, derived_names.items, range.name);
                }
                if (range.keep_inside and filter_depends_on_source) {
                    if (parquetRangePredicateFromBounds(range.lower, range.upper, range.lower_inclusive, range.upper_inclusive)) |predicate| {
                        try mergeRangePredicate(allocator, &range_predicate, range.name, predicate);
                    }
                }
            },
            .filter_isin_column => |membership| {
                if (!nameInBorrowedList(membership.input_name, derived_names.items)) {
                    try addSourceNameRequirement(allocator, &required_names, derived_names.items, membership.input_name);
                    if (!membership.invert) {
                        if (literal_scalars.get(membership.test_name)) |scalar| {
                            if (parquetRangePredicateFromScalar(scalar, .eq)) |predicate| {
                                try mergeRangePredicate(allocator, &range_predicate, membership.input_name, predicate);
                                clearNullPredicate(allocator, &null_predicate);
                            }
                        }
                    }
                }
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, membership.test_name);
            },
            .filter_isin_values => |membership| {
                if (!nameInBorrowedList(membership.input_name, derived_names.items)) {
                    try addSourceNameRequirement(allocator, &required_names, derived_names.items, membership.input_name);
                    if (!membership.invert) {
                        if (parquetRangePredicateFromSingletonColumn(membership.values)) |predicate| {
                            try mergeRangePredicate(allocator, &range_predicate, membership.input_name, predicate);
                            clearNullPredicate(allocator, &null_predicate);
                        }
                    }
                }
            },
            .drop_rows_by_mask_column => |name| {
                if (!nameInBorrowedList(name, derived_names.items)) {
                    try addSourceNameRequirement(allocator, &required_names, derived_names.items, name);
                    try mergeRangePredicate(allocator, &range_predicate, name, .{ .bool = .{ .min = false, .max = false } });
                    clearNullPredicate(allocator, &null_predicate);
                }
            },
            .where_indices_column => |predicate| {
                try addUnaryColumnOutputRequirements(allocator, &required_names, &derived_names, &literal_scalars, predicate.output_name, predicate.name);
            },
            .filter_scalar => |filter_op| {
                const filter_depends_on_source = !nameInBorrowedList(filter_op.name, derived_names.items);
                if (filter_depends_on_source) try addSourceNameRequirement(allocator, &required_names, derived_names.items, filter_op.name);
                if (filter_depends_on_source) {
                    const maybe_predicate = if (filter_op.keep_matches)
                        parquetRangePredicateFromScalar(filter_op.scalar, filter_op.op)
                    else
                        parquetRangePredicateFromDroppedScalar(filter_op.scalar, filter_op.op);
                    if (maybe_predicate) |predicate| {
                        try mergeRangePredicate(allocator, &range_predicate, filter_op.name, predicate);
                        clearNullPredicate(allocator, &null_predicate);
                    }
                }
            },
            .sort_by => |sort| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, sort.name);
            },
            .sort_by_columns => |sort| {
                try addSourceNameRequirements(allocator, &required_names, derived_names.items, sort.names);
            },
            .top_k => |top| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, top.name);
            },
            .top_k_columns => |top| {
                try addSourceNameRequirements(allocator, &required_names, derived_names.items, top.names);
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
            .filter_mask, .slice_rows, .slice_rows_signed, .drop_rows, .drop_rows_mode, .drop_rows_signed, .drop_rows_signed_mode, .drop_row_range, .drop_last_rows, .slice_rows_step, .slice_rows_signed_step, .stride_rows, .take_rows, .take_rows_optional, .take_rows_mode, .take_rows_signed, .take_rows_signed_mode, .repeat_rows, .tile_rows, .sample_rows, .sample_rows_fraction, .sample_rows_with_replacement, .sample_rows_fraction_with_replacement, .roll_rows, .shift_rows, .reverse_rows, .head, .tail => {},
            .take_rows_by_column => |name| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, name);
            },
            .take_rows_by_column_mode => |take_mode| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, take_mode.name);
            },
            .drop_rows_by_column => |name| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, name);
            },
            .drop_rows_by_column_mode => |take_mode| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, take_mode.name);
            },
            .repeat_rows_by => |count_name| {
                try addSourceNameRequirement(allocator, &required_names, derived_names.items, count_name);
            },
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
        .null_predicate = null_predicate,
    };
    range_predicate = null;
    null_predicate = null;
    return out;
}
