//! Deinitialization helper for DeviceLazyOp payloads.

const std = @import("std");
const names_mod = @import("dataframe_names.zig");
const profile_deinit_mod = @import("dataframe_lazy_op_deinit_profile.zig");

const freeNameList = names_mod.freeNameList;
const freeNameOutput = profile_deinit_mod.freeNameOutput;
const freePairOutput = profile_deinit_mod.freePairOutput;

pub fn deinit(comptime Self: type, self: *Self, allocator: std.mem.Allocator) void {
    switch (self.*) {
        .select => |names| {
            for (names) |name| allocator.free(name);
            allocator.free(names);
        },
        .select_column_indices => |indices| allocator.free(indices),
        .select_column_range => {},
        .select_last_columns => {},
        .drop_column_indices => |indices| allocator.free(indices),
        .drop_column_range => {},
        .drop_last_columns => {},
        .reverse_columns => {},
        .sort_columns_by_name => {},
        .select_name_prefix => |pattern| allocator.free(pattern.pattern),
        .select_name_suffix => |pattern| allocator.free(pattern.pattern),
        .select_name_contains => |pattern| allocator.free(pattern.pattern),
        .drop_name_prefix => |pattern| allocator.free(pattern.pattern),
        .drop_name_suffix => |pattern| allocator.free(pattern.pattern),
        .drop_name_contains => |pattern| allocator.free(pattern.pattern),
        .select_dtypes => |dtypes| allocator.free(dtypes),
        .select_dtype_class,
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
        => {},
        .drop_dtypes => |dtypes| allocator.free(dtypes),
        .drop_dtype_class,
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
        => {},
        .with_row_index => |row_index| allocator.free(row_index.name),
        .rename_column => |rename| {
            allocator.free(rename.old_name);
            allocator.free(rename.new_name);
        },
        .rename_columns => |rename| {
            freeNameList(allocator, rename.old_names);
            freeNameList(allocator, rename.new_names);
        },
        .add_column_name_prefix => |pattern| allocator.free(pattern.pattern),
        .add_column_name_suffix => |pattern| allocator.free(pattern.pattern),
        .move_column => |move| allocator.free(move.name),
        .move_column_before => |move| {
            allocator.free(move.name);
            allocator.free(move.anchor_name);
        },
        .move_column_after => |move| {
            allocator.free(move.name);
            allocator.free(move.anchor_name);
        },
        .copy_column => |copy| {
            allocator.free(copy.source_name);
            allocator.free(copy.new_name);
        },
        .copy_column_at => |copy| {
            allocator.free(copy.source_name);
            allocator.free(copy.new_name);
        },
        .copy_column_before => |copy| {
            allocator.free(copy.source_name);
            allocator.free(copy.new_name);
            allocator.free(copy.anchor_name);
        },
        .copy_column_after => |copy| {
            allocator.free(copy.source_name);
            allocator.free(copy.new_name);
            allocator.free(copy.anchor_name);
        },
        .drop_columns => |names| freeNameList(allocator, names),
        .drop_nulls, .drop_nans, .drop_infs, .drop_positive_infs, .drop_negative_infs, .drop_zeros, .drop_positive_zeros, .drop_negative_zeros, .drop_non_zeros, .drop_positives, .drop_signbits, .drop_negatives, .drop_finites, .drop_normals, .drop_subnormals, .drop_non_finites => |names| freeNameList(allocator, names),
        .filter_nulls_column, .filter_nans_column, .filter_infs_column, .filter_positive_infs_column, .filter_negative_infs_column, .filter_zeros_column, .filter_positive_zeros_column, .filter_negative_zeros_column, .filter_non_zeros_column, .filter_positives_column, .filter_signbits_column, .filter_negatives_column, .filter_finites_column, .filter_normals_column, .filter_subnormals_column, .filter_non_finites_column => |name| allocator.free(name),
        .with_column_abs => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_neg => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_square => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_reciprocal => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_sqrt => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_rsqrt => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_cbrt => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_floor => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_ceil => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_round => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_trunc => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_exp => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_exp2 => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_expm1 => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_sin => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_cos => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_tan => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_asin => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_acos => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_atan => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_sinh => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_cosh => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_tanh => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_asinh => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_acosh => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_atanh => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_log => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_log1p => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_lgamma => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_sinc => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_log2 => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_log10 => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_binary => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.lhs_name);
            allocator.free(expr.rhs_name);
        },
        .with_column_scalar => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_literal => |expr| allocator.free(expr.name),
        .with_column_literal_at => |expr| allocator.free(expr.name),
        .with_column_literal_before => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.anchor_name);
        },
        .with_column_literal_after => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.anchor_name);
        },
        .cast_column => |cast| allocator.free(cast.name),
        .fill_null_column, .fill_nan_column, .fill_inf_column, .fill_positive_inf_column, .fill_negative_inf_column, .fill_zero_column, .fill_positive_zero_column, .fill_negative_zero_column, .fill_non_zero_column, .fill_positive_column, .fill_signbit_column, .fill_negative_column, .fill_finite_column, .fill_normal_column, .fill_subnormal_column, .fill_non_finite_column => |fill| allocator.free(fill.name),
        .coalesce_columns => |coalesce| {
            allocator.free(coalesce.primary_name);
            allocator.free(coalesce.fallback_name);
            allocator.free(coalesce.output_name);
        },
        .is_null_column, .is_valid_column, .is_nan_column, .is_zero_column, .is_positive_zero_column, .is_negative_zero_column, .is_non_zero_column, .is_positive_column, .is_signbit_column, .is_negative_column, .is_finite_column, .is_normal_column, .is_subnormal_column, .is_non_finite_column, .is_inf_column, .is_positive_inf_column, .is_negative_inf_column => |predicate| {
            allocator.free(predicate.name);
            allocator.free(predicate.output_name);
        },
        .row_null_count, .row_valid_count, .row_nan_count, .row_inf_count, .row_positive_inf_count, .row_negative_inf_count, .row_zero_count, .row_positive_zero_count, .row_negative_zero_count, .row_non_zero_count, .row_positive_count, .row_signbit_count, .row_negative_count, .row_finite_count, .row_normal_count, .row_subnormal_count, .row_non_finite_count => |row_count| {
            freeNameList(allocator, row_count.names);
            allocator.free(row_count.output_name);
        },
        .with_column_compare => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.lhs_name);
            allocator.free(expr.rhs_name);
        },
        .with_column_compare_scalar => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .filter_mask => |*mask| mask.deinit(),
        .filter_column => |name| allocator.free(name),
        .filter_scalar => |filter_op| allocator.free(filter_op.name),
        .group_by_count => |group| {
            allocator.free(group.key_name);
            allocator.free(group.output_name);
        },
        .group_by_value => |group| {
            allocator.free(group.key_name);
            allocator.free(group.value_name);
            allocator.free(group.output_name);
        },
        .group_by_stats => |group| {
            allocator.free(group.key_name);
            allocator.free(group.value_name);
            allocator.free(group.output_prefix);
        },
        .group_by_stats_on => |group| {
            freeNameList(allocator, group.key_names);
            allocator.free(group.value_name);
            allocator.free(group.output_prefix);
        },
        .group_by_profile => |group| {
            allocator.free(group.key_name);
            allocator.free(group.value_name);
            allocator.free(group.output_prefix);
        },
        .group_by_profile_on => |group| {
            freeNameList(allocator, group.key_names);
            allocator.free(group.value_name);
            allocator.free(group.output_prefix);
        },
        .join_on => |*join| {
            join.right.deinit();
            freeNameList(allocator, join.left_key_names);
            freeNameList(allocator, join.right_key_names);
            allocator.free(join.options.right_suffix);
        },
        .asof_join => |*join| {
            join.right.deinit();
            allocator.free(join.left_key_name);
            allocator.free(join.right_key_name);
            allocator.free(join.options.right_suffix);
        },
        .concat_rows => |*right| right.deinit(),
        .distinct_on => |names| freeNameList(allocator, names),
        .sort_by => |sort| allocator.free(sort.name),
        .top_k => |top| allocator.free(top.name),
        .rank_profile_by => |payload| freeNameOutput(allocator, payload),
        .rolling_profile => |payload| freeNameOutput(allocator, payload),
        .rolling_moment_profile => |payload| freeNameOutput(allocator, payload),
        .rolling_range_profile => |payload| freeNameOutput(allocator, payload),
        .rolling_normalize_profile => |payload| freeNameOutput(allocator, payload),
        .expanding_normalize_profile => |payload| freeNameOutput(allocator, payload),
        .rolling_quantile_profile => |payload| freeNameOutput(allocator, payload),
        .expanding_quantile_profile => |payload| freeNameOutput(allocator, payload),
        .rolling_bool_profile => |payload| freeNameOutput(allocator, payload),
        .rolling_drawdown_profile => |payload| freeNameOutput(allocator, payload),
        .rolling_robust_profile => |payload| freeNameOutput(allocator, payload),
        .rolling_rank_profile => |payload| freeNameOutput(allocator, payload),
        .lag_profile => |payload| freeNameOutput(allocator, payload),
        .lead_profile => |payload| freeNameOutput(allocator, payload),
        .clip_profile => |payload| freeNameOutput(allocator, payload),
        .rolling_clip_profile => |payload| freeNameOutput(allocator, payload),
        .expanding_clip_profile => |payload| freeNameOutput(allocator, payload),
        .threshold_profile => |payload| freeNameOutput(allocator, payload),
        .rolling_threshold_profile => |payload| freeNameOutput(allocator, payload),
        .expanding_threshold_profile => |payload| freeNameOutput(allocator, payload),
        .expanding_profile => |payload| freeNameOutput(allocator, payload),
        .expanding_bool_profile => |payload| freeNameOutput(allocator, payload),
        .expanding_rank_profile => |payload| freeNameOutput(allocator, payload),
        .expanding_robust_profile => |payload| freeNameOutput(allocator, payload),
        .expanding_moment_profile => |payload| freeNameOutput(allocator, payload),
        .standardize_profile => |payload| freeNameOutput(allocator, payload),
        .robust_profile => |payload| freeNameOutput(allocator, payload),
        .drawdown_profile => |payload| freeNameOutput(allocator, payload),
        .extrema_profile => |payload| freeNameOutput(allocator, payload),
        .trend_profile => |payload| freeNameOutput(allocator, payload),
        .rolling_trend_profile => |payload| freeNameOutput(allocator, payload),
        .expanding_trend_profile => |payload| freeNameOutput(allocator, payload),
        .change_point_profile => |payload| freeNameOutput(allocator, payload),
        .rolling_change_point_profile => |payload| freeNameOutput(allocator, payload),
        .expanding_change_point_profile => |payload| freeNameOutput(allocator, payload),
        .sign_profile => |payload| freeNameOutput(allocator, payload),
        .rolling_sign_profile => |payload| freeNameOutput(allocator, payload),
        .expanding_sign_profile => |payload| freeNameOutput(allocator, payload),
        .crossover_profile => |payload| freePairOutput(allocator, payload, "lhs_name", "rhs_name"),
        .rolling_crossover_profile => |payload| freePairOutput(allocator, payload, "lhs_name", "rhs_name"),
        .expanding_crossover_profile => |payload| freePairOutput(allocator, payload, "lhs_name", "rhs_name"),
        .bucket_profile => |payload| freeNameOutput(allocator, payload),
        .ema_profile => |payload| freeNameOutput(allocator, payload),
        .linear_fit_profile => |payload| freePairOutput(allocator, payload, "x_name", "y_name"),
        .error_profile => |payload| freePairOutput(allocator, payload, "actual_name", "predicted_name"),
        .rolling_error_profile => |payload| freePairOutput(allocator, payload, "actual_name", "predicted_name"),
        .expanding_error_profile => |payload| freePairOutput(allocator, payload, "actual_name", "predicted_name"),
        .classification_profile => |payload| freePairOutput(allocator, payload, "actual_name", "predicted_name"),
        .rolling_classification_profile => |payload| freePairOutput(allocator, payload, "actual_name", "predicted_name"),
        .expanding_classification_profile => |payload| freePairOutput(allocator, payload, "actual_name", "predicted_name"),
        .bool_transition_profile => |payload| freeNameOutput(allocator, payload),
        .rolling_bool_transition_profile => |payload| freeNameOutput(allocator, payload),
        .expanding_bool_transition_profile => |payload| freeNameOutput(allocator, payload),
        .rolling_correlation_profile => |payload| freePairOutput(allocator, payload, "x_name", "y_name"),
        .expanding_correlation_profile => |payload| freePairOutput(allocator, payload, "x_name", "y_name"),
        .expanding_linear_fit_profile => |payload| freePairOutput(allocator, payload, "x_name", "y_name"),
        .rolling_linear_fit_profile => |payload| freePairOutput(allocator, payload, "x_name", "y_name"),
        .validity_profile => |payload| freeNameOutput(allocator, payload),
        .rolling_validity_profile => |payload| freeNameOutput(allocator, payload),
        .expanding_validity_profile => |payload| freeNameOutput(allocator, payload),
        .drop_rows, .take_rows => |row_indices| allocator.free(row_indices),
        .distinct_rows, .slice_rows, .drop_row_range, .drop_last_rows, .slice_rows_step, .stride_rows, .sample_rows, .sample_rows_with_replacement, .reverse_rows, .head, .tail => {},
    }
    self.* = undefined;
}
