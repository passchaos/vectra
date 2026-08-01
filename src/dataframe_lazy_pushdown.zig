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
            .drop_name_prefix,
            .drop_name_suffix,
            .drop_name_contains,
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
                for (names) |name| {
                    if (!nameInBorrowedList(name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, name);
                    }
                }
            },
            .filter_nulls_column => |name| {
                if (!nameInBorrowedList(name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, name);
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
                for (names) |name| {
                    if (!nameInBorrowedList(name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, name);
                    }
                }
            },
            .filter_nans_column => |name| {
                if (!nameInBorrowedList(name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, name);
                }
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
                for (names) |name| {
                    if (!nameInBorrowedList(name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, name);
                    }
                }
            },
            .filter_infs_column => |name| {
                if (!nameInBorrowedList(name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, name);
                }
            },
            .drop_positive_infs => |names| {
                if (names.len == 0) {
                    projection_blocked = true;
                    break :op_loop;
                }
                for (names) |name| {
                    if (!nameInBorrowedList(name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, name);
                    }
                }
            },
            .filter_positive_infs_column => |name| {
                if (!nameInBorrowedList(name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, name);
                }
            },
            .drop_negative_infs => |names| {
                if (names.len == 0) {
                    projection_blocked = true;
                    break :op_loop;
                }
                for (names) |name| {
                    if (!nameInBorrowedList(name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, name);
                    }
                }
            },
            .filter_negative_infs_column => |name| {
                if (!nameInBorrowedList(name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, name);
                }
            },
            .drop_zeros => |names| {
                if (names.len == 0) {
                    projection_blocked = true;
                    break :op_loop;
                }
                for (names) |name| {
                    if (!nameInBorrowedList(name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, name);
                    }
                }
            },
            .filter_zeros_column => |name| {
                if (!nameInBorrowedList(name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, name);
                }
            },
            .drop_positive_zeros => |names| {
                if (names.len == 0) {
                    projection_blocked = true;
                    break :op_loop;
                }
                for (names) |name| {
                    if (!nameInBorrowedList(name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, name);
                    }
                }
            },
            .filter_positive_zeros_column => |name| {
                if (!nameInBorrowedList(name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, name);
                }
            },
            .drop_negative_zeros => |names| {
                if (names.len == 0) {
                    projection_blocked = true;
                    break :op_loop;
                }
                for (names) |name| {
                    if (!nameInBorrowedList(name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, name);
                    }
                }
            },
            .filter_negative_zeros_column => |name| {
                if (!nameInBorrowedList(name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, name);
                }
            },
            .drop_non_zeros => |names| {
                if (names.len == 0) {
                    projection_blocked = true;
                    break :op_loop;
                }
                for (names) |name| {
                    if (!nameInBorrowedList(name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, name);
                    }
                }
            },
            .filter_non_zeros_column => |name| {
                if (!nameInBorrowedList(name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, name);
                }
            },
            .drop_positives => |names| {
                if (names.len == 0) {
                    projection_blocked = true;
                    break :op_loop;
                }
                for (names) |name| {
                    if (!nameInBorrowedList(name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, name);
                    }
                }
            },
            .filter_positives_column => |name| {
                if (!nameInBorrowedList(name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, name);
                }
            },
            .drop_signbits => |names| {
                if (names.len == 0) {
                    projection_blocked = true;
                    break :op_loop;
                }
                for (names) |name| {
                    if (!nameInBorrowedList(name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, name);
                    }
                }
            },
            .filter_signbits_column => |name| {
                if (!nameInBorrowedList(name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, name);
                }
            },
            .drop_negatives => |names| {
                if (names.len == 0) {
                    projection_blocked = true;
                    break :op_loop;
                }
                for (names) |name| {
                    if (!nameInBorrowedList(name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, name);
                    }
                }
            },
            .filter_negatives_column => |name| {
                if (!nameInBorrowedList(name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, name);
                }
            },
            .drop_finites => |names| {
                if (names.len == 0) {
                    projection_blocked = true;
                    break :op_loop;
                }
                for (names) |name| {
                    if (!nameInBorrowedList(name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, name);
                    }
                }
            },
            .filter_finites_column => |name| {
                if (!nameInBorrowedList(name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, name);
                }
            },
            .drop_normals => |names| {
                if (names.len == 0) {
                    projection_blocked = true;
                    break :op_loop;
                }
                for (names) |name| {
                    if (!nameInBorrowedList(name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, name);
                    }
                }
            },
            .filter_normals_column => |name| {
                if (!nameInBorrowedList(name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, name);
                }
            },
            .drop_subnormals => |names| {
                if (names.len == 0) {
                    projection_blocked = true;
                    break :op_loop;
                }
                for (names) |name| {
                    if (!nameInBorrowedList(name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, name);
                    }
                }
            },
            .filter_subnormals_column => |name| {
                if (!nameInBorrowedList(name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, name);
                }
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
                for (names) |name| {
                    if (!nameInBorrowedList(name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, name);
                    }
                }
            },
            .filter_non_finites_column => |name| {
                if (!nameInBorrowedList(name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, name);
                }
            },
            .with_row_index => |row_index| {
                try appendBorrowedNameUnique(allocator, &derived_names, row_index.name);
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
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                if (!nameInBorrowedList(expr.input_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
                }
            },
            .with_column_leaky_relu => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                if (!nameInBorrowedList(expr.input_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
                }
            },
            .with_column_pow_scalar => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                if (!nameInBorrowedList(expr.input_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
                }
            },
            .with_column_floor_div_scalar => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                if (!nameInBorrowedList(expr.input_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
                }
            },
            .with_column_mod_scalar => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                if (!nameInBorrowedList(expr.input_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
                }
            },
            .with_column_remainder_scalar => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                if (!nameInBorrowedList(expr.input_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
                }
            },
            .with_column_log_add_exp_scalar => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                if (!nameInBorrowedList(expr.input_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
                }
            },
            .with_column_log_add_exp2_scalar => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                if (!nameInBorrowedList(expr.input_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
                }
            },
            .with_column_xlogy_scalar => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                if (!nameInBorrowedList(expr.input_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
                }
            },
            .with_column_fmax_scalar => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                if (!nameInBorrowedList(expr.input_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
                }
            },
            .with_column_fmin_scalar => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                if (!nameInBorrowedList(expr.input_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
                }
            },
            .with_column_hypot_scalar => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                if (!nameInBorrowedList(expr.input_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
                }
            },
            .with_column_atan2_scalar => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                if (!nameInBorrowedList(expr.input_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
                }
            },
            .with_column_next_after_scalar => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                if (!nameInBorrowedList(expr.input_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
                }
            },
            .with_column_copysign_scalar => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                if (!nameInBorrowedList(expr.input_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
                }
            },
            .with_column_heaviside_scalar => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                if (!nameInBorrowedList(expr.input_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
                }
            },
            .with_column_ldexp_scalar => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                if (!nameInBorrowedList(expr.input_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
                }
            },
            .with_column_threshold => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                if (!nameInBorrowedList(expr.input_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
                }
            },
            .with_column_hardtanh => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                if (!nameInBorrowedList(expr.input_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
                }
            },
            .with_column_maximum_scalar => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                if (!nameInBorrowedList(expr.input_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
                }
            },
            .with_column_minimum_scalar => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                if (!nameInBorrowedList(expr.input_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
                }
            },
            .with_column_clip_min => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                if (!nameInBorrowedList(expr.input_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
                }
            },
            .with_column_clip_max => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                if (!nameInBorrowedList(expr.input_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
                }
            },
            .with_column_hardshrink => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                if (!nameInBorrowedList(expr.input_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
                }
            },
            .with_column_softshrink => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                if (!nameInBorrowedList(expr.input_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
                }
            },
            .with_column_elu => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                if (!nameInBorrowedList(expr.input_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
                }
            },
            .with_column_celu => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                if (!nameInBorrowedList(expr.input_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
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
            .with_column_lerp_scalar => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                try appendOwnedNameUnique(allocator, &required_names, expr.lhs_name);
                try appendOwnedNameUnique(allocator, &required_names, expr.rhs_name);
            },
            .with_column_addcmul_scalar => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                if (!nameInBorrowedList(expr.base_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.base_name);
                }
                if (!nameInBorrowedList(expr.lhs_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.lhs_name);
                }
                if (!nameInBorrowedList(expr.rhs_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.rhs_name);
                }
            },
            .with_column_addcdiv_scalar => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                if (!nameInBorrowedList(expr.base_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.base_name);
                }
                if (!nameInBorrowedList(expr.lhs_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.lhs_name);
                }
                if (!nameInBorrowedList(expr.rhs_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.rhs_name);
                }
            },
            .with_column_clip_array => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                if (!nameInBorrowedList(expr.input_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
                }
                if (!nameInBorrowedList(expr.lhs_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.lhs_name);
                }
                if (!nameInBorrowedList(expr.rhs_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.rhs_name);
                }
            },
            .with_column_where => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                if (!nameInBorrowedList(expr.input_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
                }
                if (!nameInBorrowedList(expr.lhs_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.lhs_name);
                }
                if (!nameInBorrowedList(expr.rhs_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.rhs_name);
                }
            },
            .with_column_where_scalar => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                if (!nameInBorrowedList(expr.input_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
                }
                if (!nameInBorrowedList(expr.mask_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.mask_name);
                }
            },
            .with_column_isin => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                if (!nameInBorrowedList(expr.input_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
                }
                if (!nameInBorrowedList(expr.test_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.test_name);
                }
            },
            .with_column_masked_put_scalar => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                if (!nameInBorrowedList(expr.input_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
                }
                if (!nameInBorrowedList(expr.mask_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.mask_name);
                }
            },
            .with_column_put_flat_scalar => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                if (!nameInBorrowedList(expr.input_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
                }
            },
            .with_column_put_flat => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                if (!nameInBorrowedList(expr.input_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
                }
                if (!nameInBorrowedList(expr.value_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.value_name);
                }
            },
            .with_column_put_flat_scalar_mode => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                if (!nameInBorrowedList(expr.input_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
                }
            },
            .with_column_put_flat_scalar_signed => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                if (!nameInBorrowedList(expr.input_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
                }
            },
            .with_column_isclose_scalar => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
            },
            .with_column_logical => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                if (!nameInBorrowedList(expr.lhs_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.lhs_name);
                }
                if (!nameInBorrowedList(expr.rhs_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, expr.rhs_name);
                }
            },
            .with_column_logical_scalar => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
            },
            .with_column_literal => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
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
                if (!nameInBorrowedList(cast.name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, cast.name);
                }
            },
            .fill_null_column => |fill| {
                if (!nameInBorrowedList(fill.name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, fill.name);
                }
            },
            .fill_nan_column => |fill| {
                if (!nameInBorrowedList(fill.name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, fill.name);
                }
            },
            .fill_inf_column => |fill| {
                if (!nameInBorrowedList(fill.name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, fill.name);
                }
            },
            .fill_positive_inf_column => |fill| {
                if (!nameInBorrowedList(fill.name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, fill.name);
                }
            },
            .fill_negative_inf_column => |fill| {
                if (!nameInBorrowedList(fill.name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, fill.name);
                }
            },
            .fill_zero_column => |fill| {
                if (!nameInBorrowedList(fill.name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, fill.name);
                }
            },
            .fill_positive_zero_column => |fill| {
                if (!nameInBorrowedList(fill.name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, fill.name);
                }
            },
            .fill_negative_zero_column => |fill| {
                if (!nameInBorrowedList(fill.name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, fill.name);
                }
            },
            .fill_non_zero_column => |fill| {
                if (!nameInBorrowedList(fill.name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, fill.name);
                }
            },
            .fill_positive_column => |fill| {
                if (!nameInBorrowedList(fill.name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, fill.name);
                }
            },
            .fill_signbit_column => |fill| {
                if (!nameInBorrowedList(fill.name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, fill.name);
                }
            },
            .fill_negative_column => |fill| {
                if (!nameInBorrowedList(fill.name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, fill.name);
                }
            },
            .fill_finite_column => |fill| {
                if (!nameInBorrowedList(fill.name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, fill.name);
                }
            },
            .fill_normal_column => |fill| {
                if (!nameInBorrowedList(fill.name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, fill.name);
                }
            },
            .fill_subnormal_column => |fill| {
                if (!nameInBorrowedList(fill.name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, fill.name);
                }
            },
            .fill_non_finite_column => |fill| {
                if (!nameInBorrowedList(fill.name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, fill.name);
                }
            },
            .coalesce_columns => |coalesce| {
                try appendBorrowedNameUnique(allocator, &derived_names, coalesce.output_name);
                if (!nameInBorrowedList(coalesce.primary_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, coalesce.primary_name);
                }
                if (!nameInBorrowedList(coalesce.fallback_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, coalesce.fallback_name);
                }
            },
            .is_null_column, .is_valid_column, .is_nan_column, .is_zero_column, .is_positive_zero_column, .is_negative_zero_column, .is_non_zero_column, .is_positive_column, .is_signbit_column, .is_negative_column, .is_finite_column, .is_normal_column, .is_subnormal_column, .is_non_finite_column, .is_inf_column, .is_positive_inf_column, .is_negative_inf_column => |predicate| {
                try appendBorrowedNameUnique(allocator, &derived_names, predicate.output_name);
                if (!nameInBorrowedList(predicate.name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, predicate.name);
                }
            },
            .row_null_count, .row_valid_count, .row_null_ratio, .row_valid_ratio, .row_first_valid_index, .row_last_valid_index, .row_first_null_index, .row_last_null_index, .row_argmin, .row_argmax, .row_median, .row_iqr, .row_interdecile_range, .row_midhinge, .row_trimean, .row_bowley_skewness, .row_quartile_coeff_dispersion, .row_kelley_skewness, .row_mad, .row_mode, .row_entropy, .row_gini_impurity, .row_perplexity, .row_inverse_simpson, .row_simpson_concentration, .row_evenness, .row_mode_count, .row_mode_ratio, .row_mode_margin, .row_mode_margin_ratio, .row_count_distinct, .row_n_unique, .row_sum, .row_mean, .row_logsumexp, .row_logmeanexp, .row_softmax_entropy, .row_softmax_perplexity, .row_softmax_confidence, .row_softmax_margin, .row_softmax_evenness, .row_softmax_concentration, .row_softmax_normalized_hhi, .row_softmax_gini_impurity, .row_softmax_inverse_simpson, .row_softmax_simpson_evenness, .row_logit_margin, .row_geometric_mean, .row_magnitude_geometric_mean, .row_harmonic_mean, .row_skewness, .row_magnitude_skewness, .row_kurtosis, .row_magnitude_kurtosis, .row_prod, .row_min, .row_max, .row_ptp, .row_magnitude_ptp, .row_midrange, .row_magnitude_midrange, .row_range_coeff, .row_magnitude_range_coeff, .row_mean_abs, .row_hhi, .row_magnitude_normalized_hhi, .row_magnitude_sparsity, .row_magnitude_inverse_simpson, .row_magnitude_simpson_evenness, .row_magnitude_dominance, .row_magnitude_dominance_margin, .row_magnitude_entropy, .row_magnitude_perplexity, .row_magnitude_evenness, .row_mean_abs_dev, .row_gini_mean_diff, .row_gini_coefficient, .row_mean_abs_dev_ratio, .row_rms, .row_l1_norm, .row_l2_norm, .row_true_count, .row_false_count, .row_any_true, .row_all_true, .row_any_false, .row_all_false, .row_first_true_index, .row_last_true_index, .row_first_false_index, .row_last_false_index, .row_true_ratio, .row_false_ratio, .row_nan_count, .row_nan_ratio, .row_inf_count, .row_inf_ratio, .row_positive_inf_count, .row_negative_inf_count, .row_positive_inf_ratio, .row_negative_inf_ratio, .row_zero_count, .row_zero_ratio, .row_positive_zero_count, .row_negative_zero_count, .row_positive_zero_ratio, .row_negative_zero_ratio, .row_non_zero_count, .row_non_zero_ratio, .row_positive_count, .row_positive_ratio, .row_signbit_count, .row_signbit_ratio, .row_negative_count, .row_negative_ratio, .row_finite_count, .row_finite_ratio, .row_normal_count, .row_normal_ratio, .row_subnormal_count, .row_subnormal_ratio, .row_non_finite_count, .row_non_finite_ratio => |row_count| {
                try appendBorrowedNameUnique(allocator, &derived_names, row_count.output_name);
                if (row_count.names.len == 0) {
                    // Empty row-count input means "all columns visible at this
                    // point".  The lightweight pushdown planner has no complete
                    // source schema or alias map, so it must materialize the
                    // scan rather than project an empty dependency set.
                    projection_blocked = true;
                    break :op_loop;
                }
                for (row_count.names) |name| {
                    if (!nameInBorrowedList(name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, name);
                    }
                }
            },
            .row_cumulative_argmin, .row_cumulative_argmax, .row_cumulative_mode, .row_cumulative_mode_count, .row_cumulative_mode_ratio, .row_cumulative_mode_margin, .row_cumulative_mode_margin_ratio, .row_cumulative_distinct_count, .row_cumulative_n_unique, .row_cumulative_first_true_index, .row_cumulative_last_true_index, .row_cumulative_first_false_index, .row_cumulative_last_false_index, .row_cumulative_first_valid_index, .row_cumulative_last_valid_index, .row_cumulative_first_null_index, .row_cumulative_last_null_index, .row_cumulative_null_count, .row_cumulative_valid_count, .row_cumulative_null_ratio, .row_cumulative_valid_ratio, .row_cumulative_true_count, .row_cumulative_false_count, .row_cumulative_true_ratio, .row_cumulative_false_ratio, .row_cumulative_positive_zero_count, .row_cumulative_negative_zero_count, .row_cumulative_signbit_count, .row_cumulative_positive_zero_ratio, .row_cumulative_negative_zero_ratio, .row_cumulative_signbit_ratio, .row_cumulative_nan_count, .row_cumulative_inf_count, .row_cumulative_positive_inf_count, .row_cumulative_negative_inf_count, .row_cumulative_finite_count, .row_cumulative_normal_count, .row_cumulative_subnormal_count, .row_cumulative_non_finite_count, .row_cumulative_nan_ratio, .row_cumulative_inf_ratio, .row_cumulative_positive_inf_ratio, .row_cumulative_negative_inf_ratio, .row_cumulative_finite_ratio, .row_cumulative_normal_ratio, .row_cumulative_subnormal_ratio, .row_cumulative_non_finite_ratio, .row_cumulative_zero_count, .row_cumulative_non_zero_count, .row_cumulative_positive_count, .row_cumulative_negative_count, .row_cumulative_zero_ratio, .row_cumulative_non_zero_ratio, .row_cumulative_positive_ratio, .row_cumulative_negative_ratio, .row_cumulative_any_true, .row_cumulative_all_true, .row_cumulative_any_false, .row_cumulative_all_false, .row_centered, .row_zscore, .row_robust_zscore, .row_average_rank, .row_ordinal_rank, .row_dense_rank, .row_competition_rank, .row_percent_rank, .row_cume_dist, .row_cumulative_sum, .row_cumulative_mean, .row_cumulative_logsumexp, .row_cumulative_logmeanexp, .row_cumulative_geometric_mean, .row_cumulative_harmonic_mean, .row_cumulative_skewness, .row_cumulative_kurtosis, .row_cumulative_rms, .row_cumulative_mean_abs, .row_cumulative_mean_square, .row_cumulative_max_abs, .row_cumulative_min_abs, .row_cumulative_l1_norm, .row_cumulative_l2_norm, .row_cumulative_product, .row_cumulative_max, .row_cumulative_min, .row_cumulative_range, .row_iqr_outlier, .row_tukey_winsorize, .row_max_indicator, .row_min_indicator, .row_minmax_scale, .row_l2_normalize, .row_l1_normalize, .row_sum_normalize, .row_mean_normalize, .row_max_abs_normalize, .row_softmax, .row_log_softmax, .row_softmin, .row_log_softmin => |row_outputs| {
                for (row_outputs.output_names) |output_name| {
                    try appendBorrowedNameUnique(allocator, &derived_names, output_name);
                }
                if (row_outputs.names.len == 0) {
                    projection_blocked = true;
                    break :op_loop;
                }
                for (row_outputs.names) |name| {
                    if (!nameInBorrowedList(name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, name);
                    }
                }
            },
            .row_cumulative_variance, .row_cumulative_stddev, .row_cumulative_sem, .row_cumulative_cv, .row_cumulative_fano => |row_outputs| {
                for (row_outputs.output_names) |output_name| {
                    try appendBorrowedNameUnique(allocator, &derived_names, output_name);
                }
                if (row_outputs.names.len == 0) {
                    projection_blocked = true;
                    break :op_loop;
                }
                for (row_outputs.names) |name| {
                    if (!nameInBorrowedList(name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, name);
                    }
                }
            },
            .row_variance, .row_magnitude_variance, .row_stddev, .row_magnitude_stddev, .row_sem, .row_magnitude_sem, .row_cv, .row_magnitude_cv, .row_magnitude_fano, .row_fano => |row_dispersion| {
                try appendBorrowedNameUnique(allocator, &derived_names, row_dispersion.output_name);
                if (row_dispersion.names.len == 0) {
                    projection_blocked = true;
                    break :op_loop;
                }
                for (row_dispersion.names) |name| {
                    if (!nameInBorrowedList(name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, name);
                    }
                }
            },
            .row_quantile => |row_quantile| {
                try appendBorrowedNameUnique(allocator, &derived_names, row_quantile.output_name);
                if (row_quantile.names.len == 0) {
                    projection_blocked = true;
                    break :op_loop;
                }
                for (row_quantile.names) |name| {
                    if (!nameInBorrowedList(name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, name);
                    }
                }
            },
            .row_quantile_range => |row_quantile_range| {
                try appendBorrowedNameUnique(allocator, &derived_names, row_quantile_range.output_name);
                if (row_quantile_range.names.len == 0) {
                    projection_blocked = true;
                    break :op_loop;
                }
                for (row_quantile_range.names) |name| {
                    if (!nameInBorrowedList(name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, name);
                    }
                }
            },
            .row_trimmed_mean => |row_trimmed_mean| {
                try appendBorrowedNameUnique(allocator, &derived_names, row_trimmed_mean.output_name);
                if (row_trimmed_mean.names.len == 0) {
                    projection_blocked = true;
                    break :op_loop;
                }
                for (row_trimmed_mean.names) |name| {
                    if (!nameInBorrowedList(name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, name);
                    }
                }
            },
            .row_winsorized_mean => |row_winsorized_mean| {
                try appendBorrowedNameUnique(allocator, &derived_names, row_winsorized_mean.output_name);
                if (row_winsorized_mean.names.len == 0) {
                    projection_blocked = true;
                    break :op_loop;
                }
                for (row_winsorized_mean.names) |name| {
                    if (!nameInBorrowedList(name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, name);
                    }
                }
            },
            .row_pair_count, .row_weighted_mean, .row_weighted_median, .row_weighted_iqr, .row_weighted_mad, .row_weighted_mode, .row_weighted_mode_weight, .row_weighted_mode_ratio, .row_weighted_mode_margin, .row_weighted_mode_margin_ratio, .row_weighted_entropy, .row_weighted_gini_impurity, .row_weighted_perplexity, .row_weighted_inverse_simpson, .row_weighted_simpson_concentration, .row_weighted_evenness, .row_dot, .row_cosine_similarity, .row_squared_euclidean_distance, .row_euclidean_distance, .row_manhattan_distance, .row_chebyshev_distance, .row_canberra_distance, .row_bray_curtis_distance, .row_mean_error, .row_mae, .row_mse, .row_rmse, .row_mape, .row_smape, .row_covariance, .row_correlation, .row_beta => |row_weighted| {
                try appendBorrowedNameUnique(allocator, &derived_names, row_weighted.output_name);
                if (row_weighted.value_names.len == 0 or row_weighted.weight_names.len == 0) {
                    projection_blocked = true;
                    break :op_loop;
                }
                for (row_weighted.value_names) |name| {
                    if (!nameInBorrowedList(name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, name);
                    }
                }
                for (row_weighted.weight_names) |name| {
                    if (!nameInBorrowedList(name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, name);
                    }
                }
            },
            .row_weighted_variance, .row_weighted_stddev => |row_weighted| {
                try appendBorrowedNameUnique(allocator, &derived_names, row_weighted.output_name);
                if (row_weighted.value_names.len == 0 or row_weighted.weight_names.len == 0) {
                    projection_blocked = true;
                    break :op_loop;
                }
                for (row_weighted.value_names) |name| {
                    if (!nameInBorrowedList(name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, name);
                    }
                }
                for (row_weighted.weight_names) |name| {
                    if (!nameInBorrowedList(name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, name);
                    }
                }
            },
            .row_weighted_quantile => |row_weighted| {
                try appendBorrowedNameUnique(allocator, &derived_names, row_weighted.output_name);
                if (row_weighted.value_names.len == 0 or row_weighted.weight_names.len == 0) {
                    projection_blocked = true;
                    break :op_loop;
                }
                for (row_weighted.value_names) |name| {
                    if (!nameInBorrowedList(name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, name);
                    }
                }
                for (row_weighted.weight_names) |name| {
                    if (!nameInBorrowedList(name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, name);
                    }
                }
            },
            .row_weighted_covariance, .row_weighted_correlation, .row_weighted_beta => |row_weighted| {
                try appendBorrowedNameUnique(allocator, &derived_names, row_weighted.output_name);
                if (row_weighted.lhs_names.len == 0 or row_weighted.lhs_names.len != row_weighted.rhs_names.len or row_weighted.lhs_names.len != row_weighted.weight_names.len) {
                    projection_blocked = true;
                    break :op_loop;
                }
                for (row_weighted.lhs_names) |name| {
                    if (!nameInBorrowedList(name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, name);
                    }
                }
                for (row_weighted.rhs_names) |name| {
                    if (!nameInBorrowedList(name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, name);
                    }
                }
                for (row_weighted.weight_names) |name| {
                    if (!nameInBorrowedList(name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, name);
                    }
                }
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
            .drop_rows_by_mask_column => |name| {
                if (!nameInBorrowedList(name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, name);
                }
            },
            .where_indices_column => |predicate| {
                try appendBorrowedNameUnique(allocator, &derived_names, predicate.output_name);
                if (!nameInBorrowedList(predicate.name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, predicate.name);
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
            .filter_mask, .slice_rows, .slice_rows_signed, .drop_rows, .drop_rows_mode, .drop_rows_signed, .drop_rows_signed_mode, .drop_row_range, .drop_last_rows, .slice_rows_step, .slice_rows_signed_step, .stride_rows, .take_rows, .take_rows_optional, .take_rows_mode, .take_rows_signed, .take_rows_signed_mode, .repeat_rows, .tile_rows, .sample_rows, .sample_rows_with_replacement, .roll_rows, .shift_rows, .reverse_rows, .head, .tail => {},
            .take_rows_by_column => |name| {
                if (!nameInBorrowedList(name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, name);
                }
            },
            .take_rows_by_column_mode => |take_mode| {
                if (!nameInBorrowedList(take_mode.name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, take_mode.name);
                }
            },
            .drop_rows_by_column => |name| {
                if (!nameInBorrowedList(name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, name);
                }
            },
            .drop_rows_by_column_mode => |take_mode| {
                if (!nameInBorrowedList(take_mode.name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, take_mode.name);
                }
            },
            .repeat_rows_by => |count_name| {
                if (!nameInBorrowedList(count_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, count_name);
                }
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
