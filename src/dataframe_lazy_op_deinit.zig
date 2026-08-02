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
        .select_name_glob => |pattern| allocator.free(pattern.pattern),
        .drop_name_prefix => |pattern| allocator.free(pattern.pattern),
        .drop_name_suffix => |pattern| allocator.free(pattern.pattern),
        .drop_name_contains => |pattern| allocator.free(pattern.pattern),
        .drop_name_glob => |pattern| allocator.free(pattern.pattern),
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
        .strip_column_name_prefix => |pattern| allocator.free(pattern.pattern),
        .strip_column_name_suffix => |pattern| allocator.free(pattern.pattern),
        .replace_column_name_prefix => |replace| {
            allocator.free(replace.old_pattern);
            allocator.free(replace.new_pattern);
        },
        .replace_column_name_suffix => |replace| {
            allocator.free(replace.old_pattern);
            allocator.free(replace.new_pattern);
        },
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
        .drop_nulls, .drop_all_nulls, .filter_all_nulls, .drop_nans, .drop_infs, .drop_positive_infs, .drop_negative_infs, .drop_zeros, .drop_positive_zeros, .drop_negative_zeros, .drop_non_zeros, .drop_positives, .drop_signbits, .drop_negatives, .drop_finites, .drop_normals, .drop_subnormals, .drop_non_finites => |names| freeNameList(allocator, names),
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
        .with_column_sign => |expr| {
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
        .with_column_deg2rad => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_rad2deg => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_expit => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_logit => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_softplus => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_logsigmoid => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_relu => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_leaky_relu => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_relu6 => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_pow_scalar => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_floor_div_scalar => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_mod_scalar => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_remainder_scalar => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_log_add_exp_scalar => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_log_add_exp2_scalar => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_xlogy_scalar => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_fmax_scalar => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_fmin_scalar => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_hypot_scalar => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_atan2_scalar => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_next_after_scalar => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_copysign_scalar => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_heaviside_scalar => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_ldexp_scalar => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_threshold => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_hardtanh => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_between => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_maximum_scalar => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_minimum_scalar => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_clip_min => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_clip_max => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_hardshrink => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_softshrink => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_tanhshrink => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_elu => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_celu => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_softsign => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_hardsigmoid => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_hardswish => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_silu => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_swish => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_mish => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_gelu => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_selu => |expr| {
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
        .with_column_lerp_scalar => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.lhs_name);
            allocator.free(expr.rhs_name);
        },
        .with_column_addcmul_scalar => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.base_name);
            allocator.free(expr.lhs_name);
            allocator.free(expr.rhs_name);
        },
        .with_column_addcdiv_scalar => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.base_name);
            allocator.free(expr.lhs_name);
            allocator.free(expr.rhs_name);
        },
        .with_column_clip_array => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
            allocator.free(expr.lhs_name);
            allocator.free(expr.rhs_name);
        },
        .with_column_where => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
            allocator.free(expr.lhs_name);
            allocator.free(expr.rhs_name);
        },
        .with_column_where_scalar => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
            allocator.free(expr.mask_name);
        },
        .with_column_isin => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
            allocator.free(expr.test_name);
        },
        .with_column_isin_values => |*expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
            expr.values.deinit();
        },
        .with_column_masked_put_scalar => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
            allocator.free(expr.mask_name);
        },
        .with_column_put_flat_scalar => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
            allocator.free(expr.row_indices);
        },
        .with_column_put_flat => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
            allocator.free(expr.row_indices);
            allocator.free(expr.value_name);
        },
        .with_column_put_flat_scalar_mode => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
            allocator.free(expr.row_indices);
        },
        .with_column_put_flat_scalar_signed => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
            allocator.free(expr.row_indices);
        },
        .with_column_isclose_scalar => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_logical => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.lhs_name);
            allocator.free(expr.rhs_name);
        },
        .with_column_logical_scalar => |expr| {
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
        .fill_null_column, .fill_nan_column, .fill_inf_column, .fill_positive_inf_column, .fill_negative_inf_column, .fill_zero_column, .fill_positive_zero_column, .fill_negative_zero_column, .fill_non_zero_column, .fill_positive_column, .fill_signbit_column, .fill_negative_column, .fill_finite_column, .fill_normal_column, .fill_subnormal_column, .fill_non_finite_column, .null_if_column => |fill| allocator.free(fill.name),
        .fill_null_forward_column, .fill_null_backward_column => |name| allocator.free(name),
        .null_if_values_column => |*null_if| {
            allocator.free(null_if.name);
            null_if.values.deinit();
        },
        .null_if_nan_column, .null_if_inf_column, .null_if_positive_inf_column, .null_if_negative_inf_column, .null_if_zero_column, .null_if_positive_zero_column, .null_if_negative_zero_column, .null_if_non_zero_column, .null_if_positive_column, .null_if_signbit_column, .null_if_negative_column, .null_if_finite_column, .null_if_normal_column, .null_if_subnormal_column, .null_if_non_finite_column => |name| allocator.free(name),
        .coalesce_columns => |coalesce| {
            allocator.free(coalesce.primary_name);
            allocator.free(coalesce.fallback_name);
            allocator.free(coalesce.output_name);
        },
        .coalesce_columns_many => |coalesce| {
            freeNameList(allocator, coalesce.names);
            allocator.free(coalesce.output_name);
        },
        .is_null_column, .is_valid_column, .is_nan_column, .is_zero_column, .is_positive_zero_column, .is_negative_zero_column, .is_non_zero_column, .is_positive_column, .is_signbit_column, .is_negative_column, .is_finite_column, .is_normal_column, .is_subnormal_column, .is_non_finite_column, .is_inf_column, .is_positive_inf_column, .is_negative_inf_column => |predicate| {
            allocator.free(predicate.name);
            allocator.free(predicate.output_name);
        },
        .row_null_count, .row_valid_count, .row_any_null, .row_all_null, .row_any_valid, .row_all_valid, .row_null_ratio, .row_valid_ratio, .row_first_valid_index, .row_last_valid_index, .row_first_null_index, .row_last_null_index, .row_argmin, .row_argmax, .row_median, .row_iqr, .row_interdecile_range, .row_midhinge, .row_trimean, .row_bowley_skewness, .row_quartile_coeff_dispersion, .row_kelley_skewness, .row_mad, .row_mode, .row_entropy, .row_gini_impurity, .row_perplexity, .row_inverse_simpson, .row_simpson_concentration, .row_evenness, .row_mode_count, .row_mode_ratio, .row_mode_margin, .row_mode_margin_ratio, .row_count_distinct, .row_n_unique, .row_is_duplicated, .row_is_unique, .row_sum, .row_mean, .row_logsumexp, .row_logmeanexp, .row_softmax_entropy, .row_softmax_perplexity, .row_softmax_confidence, .row_softmax_margin, .row_softmax_evenness, .row_softmax_concentration, .row_softmax_normalized_hhi, .row_softmax_gini_impurity, .row_softmax_inverse_simpson, .row_softmax_simpson_evenness, .row_logit_margin, .row_geometric_mean, .row_magnitude_geometric_mean, .row_harmonic_mean, .row_skewness, .row_magnitude_skewness, .row_kurtosis, .row_magnitude_kurtosis, .row_prod, .row_min, .row_max, .row_ptp, .row_magnitude_ptp, .row_midrange, .row_magnitude_midrange, .row_range_coeff, .row_magnitude_range_coeff, .row_mean_abs, .row_hhi, .row_magnitude_normalized_hhi, .row_magnitude_sparsity, .row_magnitude_inverse_simpson, .row_magnitude_simpson_evenness, .row_magnitude_dominance, .row_magnitude_dominance_margin, .row_magnitude_entropy, .row_magnitude_perplexity, .row_magnitude_evenness, .row_mean_abs_dev, .row_gini_mean_diff, .row_gini_coefficient, .row_mean_abs_dev_ratio, .row_rms, .row_l1_norm, .row_l2_norm, .row_true_count, .row_false_count, .row_any_true, .row_all_true, .row_any_false, .row_all_false, .row_first_true_index, .row_last_true_index, .row_first_false_index, .row_last_false_index, .row_true_ratio, .row_false_ratio, .row_any_zero, .row_all_zero, .row_any_non_zero, .row_all_non_zero, .row_any_positive_zero, .row_all_positive_zero, .row_any_negative_zero, .row_all_negative_zero, .row_any_positive, .row_all_positive, .row_any_signbit, .row_all_signbit, .row_any_negative, .row_all_negative, .row_any_nan, .row_all_nan, .row_any_inf, .row_all_inf, .row_any_positive_inf, .row_all_positive_inf, .row_any_negative_inf, .row_all_negative_inf, .row_any_finite, .row_all_finite, .row_any_normal, .row_all_normal, .row_any_subnormal, .row_all_subnormal, .row_any_non_finite, .row_all_non_finite, .row_nan_count, .row_nan_ratio, .row_inf_count, .row_inf_ratio, .row_positive_inf_count, .row_negative_inf_count, .row_positive_inf_ratio, .row_negative_inf_ratio, .row_zero_count, .row_zero_ratio, .row_positive_zero_count, .row_negative_zero_count, .row_positive_zero_ratio, .row_negative_zero_ratio, .row_non_zero_count, .row_non_zero_ratio, .row_first_nan_index, .row_last_nan_index, .row_first_inf_index, .row_last_inf_index, .row_first_positive_inf_index, .row_last_positive_inf_index, .row_first_negative_inf_index, .row_last_negative_inf_index, .row_first_finite_index, .row_last_finite_index, .row_first_normal_index, .row_last_normal_index, .row_first_subnormal_index, .row_last_subnormal_index, .row_first_non_finite_index, .row_last_non_finite_index, .row_first_positive_zero_index, .row_last_positive_zero_index, .row_first_negative_zero_index, .row_last_negative_zero_index, .row_first_signbit_index, .row_last_signbit_index, .row_first_zero_index, .row_last_zero_index, .row_first_non_zero_index, .row_last_non_zero_index, .row_first_positive_index, .row_last_positive_index, .row_first_negative_index, .row_last_negative_index, .row_positive_count, .row_positive_ratio, .row_signbit_count, .row_signbit_ratio, .row_negative_count, .row_negative_ratio, .row_finite_count, .row_finite_ratio, .row_normal_count, .row_normal_ratio, .row_subnormal_count, .row_subnormal_ratio, .row_non_finite_count, .row_non_finite_ratio => |row_count| {
            freeNameList(allocator, row_count.names);
            allocator.free(row_count.output_name);
        },
        .row_cumulative_argmin, .row_cumulative_argmax, .row_cumulative_mode, .row_cumulative_mode_count, .row_cumulative_mode_ratio, .row_cumulative_mode_margin, .row_cumulative_mode_margin_ratio, .row_cumulative_distinct_count, .row_cumulative_n_unique, .row_cumulative_first_true_index, .row_cumulative_last_true_index, .row_cumulative_first_false_index, .row_cumulative_last_false_index, .row_cumulative_first_valid_index, .row_cumulative_last_valid_index, .row_cumulative_first_null_index, .row_cumulative_last_null_index, .row_cumulative_null_count, .row_cumulative_valid_count, .row_cumulative_any_null, .row_cumulative_all_null, .row_cumulative_any_valid, .row_cumulative_all_valid, .row_cumulative_null_ratio, .row_cumulative_valid_ratio, .row_cumulative_true_count, .row_cumulative_false_count, .row_cumulative_true_ratio, .row_cumulative_false_ratio, .row_cumulative_positive_zero_count, .row_cumulative_negative_zero_count, .row_cumulative_signbit_count, .row_cumulative_positive_zero_ratio, .row_cumulative_negative_zero_ratio, .row_cumulative_signbit_ratio, .row_cumulative_nan_count, .row_cumulative_inf_count, .row_cumulative_positive_inf_count, .row_cumulative_negative_inf_count, .row_cumulative_finite_count, .row_cumulative_normal_count, .row_cumulative_subnormal_count, .row_cumulative_non_finite_count, .row_cumulative_nan_ratio, .row_cumulative_inf_ratio, .row_cumulative_positive_inf_ratio, .row_cumulative_negative_inf_ratio, .row_cumulative_finite_ratio, .row_cumulative_normal_ratio, .row_cumulative_subnormal_ratio, .row_cumulative_non_finite_ratio, .row_cumulative_any_zero, .row_cumulative_all_zero, .row_cumulative_any_non_zero, .row_cumulative_all_non_zero, .row_cumulative_any_positive_zero, .row_cumulative_all_positive_zero, .row_cumulative_any_negative_zero, .row_cumulative_all_negative_zero, .row_cumulative_any_positive, .row_cumulative_all_positive, .row_cumulative_any_signbit, .row_cumulative_all_signbit, .row_cumulative_any_negative, .row_cumulative_all_negative, .row_cumulative_any_nan, .row_cumulative_all_nan, .row_cumulative_any_inf, .row_cumulative_all_inf, .row_cumulative_any_positive_inf, .row_cumulative_all_positive_inf, .row_cumulative_any_negative_inf, .row_cumulative_all_negative_inf, .row_cumulative_any_finite, .row_cumulative_all_finite, .row_cumulative_any_normal, .row_cumulative_all_normal, .row_cumulative_any_subnormal, .row_cumulative_all_subnormal, .row_cumulative_any_non_finite, .row_cumulative_all_non_finite, .row_cumulative_first_nan_index, .row_cumulative_last_nan_index, .row_cumulative_first_inf_index, .row_cumulative_last_inf_index, .row_cumulative_first_positive_inf_index, .row_cumulative_last_positive_inf_index, .row_cumulative_first_negative_inf_index, .row_cumulative_last_negative_inf_index, .row_cumulative_first_finite_index, .row_cumulative_last_finite_index, .row_cumulative_first_normal_index, .row_cumulative_last_normal_index, .row_cumulative_first_subnormal_index, .row_cumulative_last_subnormal_index, .row_cumulative_first_non_finite_index, .row_cumulative_last_non_finite_index, .row_cumulative_zero_count, .row_cumulative_first_zero_index, .row_cumulative_last_zero_index, .row_cumulative_first_positive_zero_index, .row_cumulative_last_positive_zero_index, .row_cumulative_first_negative_zero_index, .row_cumulative_last_negative_zero_index, .row_cumulative_non_zero_count, .row_cumulative_first_non_zero_index, .row_cumulative_last_non_zero_index, .row_cumulative_first_positive_index, .row_cumulative_last_positive_index, .row_cumulative_first_signbit_index, .row_cumulative_last_signbit_index, .row_cumulative_first_negative_index, .row_cumulative_last_negative_index, .row_cumulative_positive_count, .row_cumulative_negative_count, .row_cumulative_zero_ratio, .row_cumulative_non_zero_ratio, .row_cumulative_positive_ratio, .row_cumulative_negative_ratio, .row_cumulative_any_true, .row_cumulative_all_true, .row_cumulative_any_false, .row_cumulative_all_false, .row_centered, .row_zscore, .row_robust_zscore, .row_average_rank, .row_ordinal_rank, .row_dense_rank, .row_competition_rank, .row_percent_rank, .row_cume_dist, .row_cumulative_sum, .row_cumulative_mean, .row_cumulative_logsumexp, .row_cumulative_logmeanexp, .row_cumulative_geometric_mean, .row_cumulative_harmonic_mean, .row_cumulative_skewness, .row_cumulative_kurtosis, .row_cumulative_rms, .row_cumulative_mean_abs, .row_cumulative_mean_square, .row_cumulative_max_abs, .row_cumulative_min_abs, .row_cumulative_l1_norm, .row_cumulative_l2_norm, .row_cumulative_product, .row_cumulative_max, .row_cumulative_min, .row_cumulative_range, .row_iqr_outlier, .row_tukey_winsorize, .row_max_indicator, .row_min_indicator, .row_minmax_scale, .row_l2_normalize, .row_l1_normalize, .row_sum_normalize, .row_mean_normalize, .row_max_abs_normalize, .row_softmax, .row_log_softmax, .row_softmin, .row_log_softmin => |row_outputs| {
            freeNameList(allocator, row_outputs.names);
            freeNameList(allocator, row_outputs.output_names);
        },
        .row_cumulative_variance, .row_cumulative_stddev, .row_cumulative_sem, .row_cumulative_cv, .row_cumulative_fano => |row_outputs| {
            freeNameList(allocator, row_outputs.names);
            freeNameList(allocator, row_outputs.output_names);
        },
        .row_variance, .row_magnitude_variance, .row_stddev, .row_magnitude_stddev, .row_sem, .row_magnitude_sem, .row_cv, .row_magnitude_cv, .row_magnitude_fano, .row_fano => |row_dispersion| {
            freeNameList(allocator, row_dispersion.names);
            allocator.free(row_dispersion.output_name);
        },
        .row_quantile => |row_quantile| {
            freeNameList(allocator, row_quantile.names);
            allocator.free(row_quantile.output_name);
        },
        .row_quantile_range => |row_quantile_range| {
            freeNameList(allocator, row_quantile_range.names);
            allocator.free(row_quantile_range.output_name);
        },
        .row_trimmed_mean => |row_trimmed_mean| {
            freeNameList(allocator, row_trimmed_mean.names);
            allocator.free(row_trimmed_mean.output_name);
        },
        .row_winsorized_mean => |row_winsorized_mean| {
            freeNameList(allocator, row_winsorized_mean.names);
            allocator.free(row_winsorized_mean.output_name);
        },
        .row_pair_count, .row_weighted_mean, .row_weighted_sum, .row_weighted_weight_sum, .row_weighted_positive_count, .row_weighted_effective_n, .row_weighted_mean_square, .row_weighted_rms, .row_weighted_mean_abs, .row_weighted_l1_norm, .row_weighted_l2_norm, .row_weighted_min, .row_weighted_max, .row_weighted_max_abs, .row_weighted_min_abs, .row_weighted_range, .row_weighted_midrange, .row_weighted_range_coeff, .row_weighted_product, .row_weighted_geometric_mean, .row_weighted_harmonic_mean, .row_weighted_logsumexp, .row_weighted_logmeanexp, .row_weighted_median, .row_weighted_iqr, .row_weighted_mad, .row_weighted_interdecile_range, .row_weighted_midhinge, .row_weighted_trimean, .row_weighted_bowley_skewness, .row_weighted_quartile_coeff_dispersion, .row_weighted_kelley_skewness, .row_weighted_mode, .row_weighted_mode_weight, .row_weighted_mode_ratio, .row_weighted_mode_margin, .row_weighted_mode_margin_ratio, .row_weighted_entropy, .row_weighted_gini_impurity, .row_weighted_perplexity, .row_weighted_inverse_simpson, .row_weighted_simpson_concentration, .row_weighted_evenness, .row_weighted_mean_abs_dev, .row_weighted_mean_abs_dev_ratio, .row_weighted_gini_mean_diff, .row_weighted_gini_coefficient, .row_weighted_skewness, .row_weighted_kurtosis, .row_dot, .row_cosine_similarity, .row_squared_euclidean_distance, .row_euclidean_distance, .row_manhattan_distance, .row_chebyshev_distance, .row_canberra_distance, .row_bray_curtis_distance, .row_mean_error, .row_mae, .row_mse, .row_rmse, .row_mape, .row_smape, .row_covariance, .row_correlation, .row_beta => |row_weighted| {
            freeNameList(allocator, row_weighted.value_names);
            freeNameList(allocator, row_weighted.weight_names);
            allocator.free(row_weighted.output_name);
        },
        .row_weighted_variance, .row_weighted_stddev, .row_weighted_sem, .row_weighted_cv, .row_weighted_fano => |row_weighted| {
            freeNameList(allocator, row_weighted.value_names);
            freeNameList(allocator, row_weighted.weight_names);
            allocator.free(row_weighted.output_name);
        },
        .row_weighted_quantile, .row_weighted_trimmed_mean, .row_weighted_winsorized_mean => |row_weighted| {
            freeNameList(allocator, row_weighted.value_names);
            freeNameList(allocator, row_weighted.weight_names);
            allocator.free(row_weighted.output_name);
        },
        .row_weighted_pair_weight_sum, .row_weighted_pair_positive_count, .row_weighted_pair_effective_n, .row_weighted_dot, .row_weighted_cosine_similarity, .row_weighted_squared_euclidean_distance, .row_weighted_euclidean_distance, .row_weighted_manhattan_distance, .row_weighted_chebyshev_distance, .row_weighted_canberra_distance, .row_weighted_bray_curtis_distance, .row_weighted_mean_error, .row_weighted_mae, .row_weighted_mse, .row_weighted_rmse, .row_weighted_mape, .row_weighted_smape, .row_weighted_covariance, .row_weighted_correlation, .row_weighted_beta => |row_weighted| {
            freeNameList(allocator, row_weighted.lhs_names);
            freeNameList(allocator, row_weighted.rhs_names);
            freeNameList(allocator, row_weighted.weight_names);
            allocator.free(row_weighted.output_name);
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
        .filter_between_column => |range| allocator.free(range.name),
        .filter_isin_column => |membership| {
            allocator.free(membership.input_name);
            allocator.free(membership.test_name);
        },
        .filter_isin_values => |*membership| {
            allocator.free(membership.input_name);
            membership.values.deinit();
        },
        .drop_rows_by_mask_column => |name| allocator.free(name),
        .where_indices_column => |predicate| {
            allocator.free(predicate.name);
            allocator.free(predicate.output_name);
        },
        .filter_scalar => |filter_op| allocator.free(filter_op.name),
        .group_id => |row_count| {
            freeNameList(allocator, row_count.names);
            allocator.free(row_count.output_name);
        },
        .group_first_row_index => |row_count| {
            freeNameList(allocator, row_count.names);
            allocator.free(row_count.output_name);
        },
        .group_last_row_index => |row_count| {
            freeNameList(allocator, row_count.names);
            allocator.free(row_count.output_name);
        },
        .group_is_first_row => |row_count| {
            freeNameList(allocator, row_count.names);
            allocator.free(row_count.output_name);
        },
        .group_is_last_row => |row_count| {
            freeNameList(allocator, row_count.names);
            allocator.free(row_count.output_name);
        },
        .group_is_singleton => |row_count| {
            freeNameList(allocator, row_count.names);
            allocator.free(row_count.output_name);
        },
        .group_is_duplicated => |row_count| {
            freeNameList(allocator, row_count.names);
            allocator.free(row_count.output_name);
        },
        .group_cume_dist => |row_count| {
            freeNameList(allocator, row_count.names);
            allocator.free(row_count.output_name);
        },
        .group_percent_rank => |row_count| {
            freeNameList(allocator, row_count.names);
            allocator.free(row_count.output_name);
        },
        .group_reverse_cume_dist => |row_count| {
            freeNameList(allocator, row_count.names);
            allocator.free(row_count.output_name);
        },
        .group_reverse_percent_rank => |row_count| {
            freeNameList(allocator, row_count.names);
            allocator.free(row_count.output_name);
        },
        .group_lag, .group_lead, .group_first_row_value, .group_last_row_value, .group_nth_row_value, .group_first_valid_value, .group_last_valid_value, .group_nth_valid_value, .group_fill_null_forward, .group_fill_null_backward, .group_cumulative_valid_count, .group_cumulative_null_count, .group_cumulative_valid_ratio, .group_cumulative_null_ratio, .group_cumulative_first_valid_index, .group_cumulative_last_valid_index, .group_cumulative_first_null_index, .group_cumulative_last_null_index, .group_cumulative_nan_count, .group_cumulative_nan_ratio, .group_cumulative_inf_count, .group_cumulative_inf_ratio, .group_cumulative_positive_inf_count, .group_cumulative_positive_inf_ratio, .group_cumulative_negative_inf_count, .group_cumulative_negative_inf_ratio, .group_cumulative_finite_count, .group_cumulative_finite_ratio, .group_cumulative_normal_count, .group_cumulative_normal_ratio, .group_cumulative_subnormal_count, .group_cumulative_subnormal_ratio, .group_cumulative_non_finite_count, .group_cumulative_non_finite_ratio, .group_cumulative_zero_count, .group_cumulative_zero_ratio, .group_cumulative_positive_zero_count, .group_cumulative_positive_zero_ratio, .group_cumulative_negative_zero_count, .group_cumulative_negative_zero_ratio, .group_cumulative_non_zero_count, .group_cumulative_non_zero_ratio, .group_cumulative_positive_count, .group_cumulative_positive_ratio, .group_cumulative_signbit_count, .group_cumulative_signbit_ratio, .group_cumulative_negative_count, .group_cumulative_negative_ratio, .group_cumulative_first_nan_index, .group_cumulative_last_nan_index, .group_cumulative_first_inf_index, .group_cumulative_last_inf_index, .group_cumulative_first_positive_inf_index, .group_cumulative_last_positive_inf_index, .group_cumulative_first_negative_inf_index, .group_cumulative_last_negative_inf_index, .group_cumulative_first_finite_index, .group_cumulative_last_finite_index, .group_cumulative_first_normal_index, .group_cumulative_last_normal_index, .group_cumulative_first_subnormal_index, .group_cumulative_last_subnormal_index, .group_cumulative_first_non_finite_index, .group_cumulative_last_non_finite_index, .group_cumulative_first_zero_index, .group_cumulative_last_zero_index, .group_cumulative_first_positive_zero_index, .group_cumulative_last_positive_zero_index, .group_cumulative_first_negative_zero_index, .group_cumulative_last_negative_zero_index, .group_cumulative_first_non_zero_index, .group_cumulative_last_non_zero_index, .group_cumulative_first_positive_index, .group_cumulative_last_positive_index, .group_cumulative_first_signbit_index, .group_cumulative_last_signbit_index, .group_cumulative_first_negative_index, .group_cumulative_last_negative_index, .group_cumulative_distinct_count, .group_cumulative_n_unique, .group_cumulative_mode, .group_cumulative_mode_count, .group_cumulative_mode_ratio, .group_cumulative_mode_margin, .group_cumulative_mode_margin_ratio, .group_cumulative_entropy, .group_cumulative_gini_impurity, .group_cumulative_perplexity, .group_cumulative_inverse_simpson, .group_cumulative_simpson_concentration, .group_cumulative_evenness, .group_cumulative_mean_abs_dev, .group_cumulative_mean_abs_dev_ratio, .group_cumulative_gini_mean_diff, .group_cumulative_gini_coefficient, .group_cumulative_median, .group_cumulative_iqr, .group_cumulative_mad, .group_cumulative_interdecile_range, .group_cumulative_midhinge, .group_cumulative_trimean, .group_cumulative_bowley_skewness, .group_cumulative_quartile_coeff_dispersion, .group_cumulative_kelley_skewness, .group_cumulative_any, .group_cumulative_all, .group_cumulative_true_count, .group_cumulative_false_count, .group_cumulative_true_ratio, .group_cumulative_false_ratio, .group_cumulative_first_true_index, .group_cumulative_last_true_index, .group_cumulative_first_false_index, .group_cumulative_last_false_index, .group_cumulative_sum, .group_cumulative_mean, .group_cumulative_product, .group_cumulative_min, .group_cumulative_max, .group_cumulative_variance, .group_cumulative_stddev, .group_cumulative_sem, .group_cumulative_cv, .group_cumulative_fano, .group_cumulative_skewness, .group_cumulative_kurtosis, .group_cumulative_mean_abs, .group_cumulative_mean_square, .group_cumulative_rms, .group_cumulative_max_abs, .group_cumulative_min_abs, .group_cumulative_l1_norm, .group_cumulative_l2_norm, .group_cumulative_range, .group_cumulative_midrange, .group_cumulative_range_coeff, .group_cumulative_logsumexp, .group_cumulative_logmeanexp, .group_cumulative_geometric_mean, .group_cumulative_harmonic_mean, .group_cumulative_argmin, .group_cumulative_argmax => |shift| {
            freeNameList(allocator, shift.names);
            allocator.free(shift.value_name);
            allocator.free(shift.output_name);
        },
        .group_cumulative_quantile, .group_cumulative_trimmed_mean, .group_cumulative_winsorized_mean => |shift| {
            freeNameList(allocator, shift.names);
            allocator.free(shift.value_name);
            allocator.free(shift.output_name);
        },
        .group_cumulative_weighted_sum, .group_cumulative_weighted_product, .group_cumulative_weighted_weight_sum, .group_cumulative_weighted_positive_count, .group_cumulative_weighted_effective_n, .group_cumulative_weighted_mean, .group_cumulative_weighted_mean_square, .group_cumulative_weighted_rms, .group_cumulative_weighted_min, .group_cumulative_weighted_max, .group_cumulative_weighted_median, .group_cumulative_weighted_iqr, .group_cumulative_weighted_mad, .group_cumulative_weighted_interdecile_range, .group_cumulative_weighted_midhinge, .group_cumulative_weighted_trimean, .group_cumulative_weighted_bowley_skewness, .group_cumulative_weighted_quartile_coeff_dispersion, .group_cumulative_weighted_kelley_skewness, .group_cumulative_weighted_mode, .group_cumulative_weighted_mode_weight, .group_cumulative_weighted_mode_ratio, .group_cumulative_weighted_mode_margin, .group_cumulative_weighted_mode_margin_ratio, .group_cumulative_weighted_entropy, .group_cumulative_weighted_gini_impurity, .group_cumulative_weighted_perplexity, .group_cumulative_weighted_inverse_simpson, .group_cumulative_weighted_simpson_concentration, .group_cumulative_weighted_evenness, .group_cumulative_weighted_mean_abs_dev, .group_cumulative_weighted_mean_abs_dev_ratio, .group_cumulative_weighted_gini_mean_diff, .group_cumulative_weighted_gini_coefficient, .group_cumulative_weighted_mean_abs, .group_cumulative_weighted_l1_norm, .group_cumulative_weighted_l2_norm, .group_cumulative_weighted_max_abs, .group_cumulative_weighted_min_abs, .group_cumulative_weighted_geometric_mean, .group_cumulative_weighted_harmonic_mean, .group_cumulative_weighted_logsumexp, .group_cumulative_weighted_logmeanexp, .group_cumulative_weighted_range, .group_cumulative_weighted_midrange, .group_cumulative_weighted_range_coeff, .group_cumulative_weighted_variance, .group_cumulative_weighted_stddev, .group_cumulative_weighted_sem, .group_cumulative_weighted_cv, .group_cumulative_weighted_fano, .group_cumulative_weighted_skewness, .group_cumulative_weighted_kurtosis => |shift| {
            freeNameList(allocator, shift.names);
            allocator.free(shift.value_name);
            allocator.free(shift.weight_name);
            allocator.free(shift.output_name);
        },
        .group_cumulative_weighted_quantile, .group_cumulative_weighted_trimmed_mean, .group_cumulative_weighted_winsorized_mean => |shift| {
            freeNameList(allocator, shift.names);
            allocator.free(shift.value_name);
            allocator.free(shift.weight_name);
            allocator.free(shift.output_name);
        },
        .group_cumulative_weighted_dot, .group_cumulative_weighted_cosine_similarity, .group_cumulative_weighted_squared_euclidean_distance, .group_cumulative_weighted_euclidean_distance, .group_cumulative_weighted_manhattan_distance, .group_cumulative_weighted_chebyshev_distance, .group_cumulative_weighted_canberra_distance, .group_cumulative_weighted_bray_curtis_distance, .group_cumulative_weighted_mean_error, .group_cumulative_weighted_mae, .group_cumulative_weighted_mse, .group_cumulative_weighted_rmse, .group_cumulative_weighted_mape, .group_cumulative_weighted_smape, .group_cumulative_weighted_covariance, .group_cumulative_weighted_correlation, .group_cumulative_weighted_beta => |shift| {
            freeNameList(allocator, shift.names);
            allocator.free(shift.lhs_name);
            allocator.free(shift.rhs_name);
            allocator.free(shift.weight_name);
            allocator.free(shift.output_name);
        },
        .group_row_number => |row_count| {
            freeNameList(allocator, row_count.names);
            allocator.free(row_count.output_name);
        },
        .group_size => |row_count| {
            freeNameList(allocator, row_count.names);
            allocator.free(row_count.output_name);
        },
        .group_reverse_row_number => |row_count| {
            freeNameList(allocator, row_count.names);
            allocator.free(row_count.output_name);
        },
        .group_by_count => |group| {
            allocator.free(group.key_name);
            allocator.free(group.output_name);
        },
        .group_by_count_on => |group| {
            freeNameList(allocator, group.key_names);
            allocator.free(group.output_name);
        },
        .group_by_rows => |group| allocator.free(group.key_name),
        .group_by_rows_on => |group| freeNameList(allocator, group.key_names),
        .group_by_sorted_rows => |group| {
            allocator.free(group.key_name);
            allocator.free(group.sort_name);
        },
        .group_by_sorted_rows_on => |group| {
            freeNameList(allocator, group.key_names);
            allocator.free(group.sort_name);
        },
        .group_by_sorted_rows_columns => |group| {
            allocator.free(group.key_name);
            freeNameList(allocator, group.sort_names);
            allocator.free(group.options);
        },
        .group_by_sorted_rows_columns_on => |group| {
            freeNameList(allocator, group.key_names);
            freeNameList(allocator, group.sort_names);
            allocator.free(group.options);
        },
        .group_by_value => |group| {
            allocator.free(group.key_name);
            allocator.free(group.value_name);
            allocator.free(group.output_name);
        },
        .group_by_value_on => |group| {
            freeNameList(allocator, group.key_names);
            allocator.free(group.value_name);
            allocator.free(group.output_name);
        },
        .group_by_weighted => |group| {
            allocator.free(group.key_name);
            allocator.free(group.value_name);
            allocator.free(group.weight_name);
            allocator.free(group.output_name);
        },
        .group_by_weighted_on => |group| {
            freeNameList(allocator, group.key_names);
            allocator.free(group.value_name);
            allocator.free(group.weight_name);
            allocator.free(group.output_name);
        },
        .group_by_pair => |group| {
            allocator.free(group.key_name);
            allocator.free(group.lhs_name);
            allocator.free(group.rhs_name);
            allocator.free(group.output_name);
        },
        .group_by_pair_on => |group| {
            freeNameList(allocator, group.key_names);
            allocator.free(group.lhs_name);
            allocator.free(group.rhs_name);
            allocator.free(group.output_name);
        },
        .group_by_weighted_pair => |group| {
            allocator.free(group.key_name);
            allocator.free(group.lhs_name);
            allocator.free(group.rhs_name);
            allocator.free(group.weight_name);
            allocator.free(group.output_name);
        },
        .group_by_weighted_pair_on => |group| {
            freeNameList(allocator, group.key_names);
            allocator.free(group.lhs_name);
            allocator.free(group.rhs_name);
            allocator.free(group.weight_name);
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
        .concat_rows, .concat_columns => |*right| right.deinit(),
        .distinct_on, .distinct_on_last, .distinct_on_none => |names| freeNameList(allocator, names),
        .sort_by => |sort| allocator.free(sort.name),
        .sort_by_columns => |sort| {
            freeNameList(allocator, sort.names);
            allocator.free(sort.options);
        },
        .top_k => |top| allocator.free(top.name),
        .top_k_columns => |top| {
            freeNameList(allocator, top.names);
            allocator.free(top.options);
        },
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
        .drop_rows_mode => |drop_mode| allocator.free(drop_mode.row_indices),
        .drop_rows_signed => |row_indices| allocator.free(row_indices),
        .drop_rows_signed_mode => |drop_mode| allocator.free(drop_mode.row_indices),
        .take_rows_optional => |row_indices| allocator.free(row_indices),
        .take_rows_mode => |take_mode| allocator.free(take_mode.row_indices),
        .take_rows_signed => |row_indices| allocator.free(row_indices),
        .take_rows_signed_mode => |take_mode| allocator.free(take_mode.row_indices),
        .take_rows_by_column => |name| allocator.free(name),
        .take_rows_by_column_mode => |take_mode| allocator.free(take_mode.name),
        .drop_rows_by_column => |name| allocator.free(name),
        .drop_rows_by_column_mode => |take_mode| allocator.free(take_mode.name),
        .repeat_rows_by => |count_name| allocator.free(count_name),
        .distinct_rows, .distinct_rows_last, .distinct_rows_none, .slice_rows, .slice_rows_signed, .drop_row_range, .drop_last_rows, .slice_rows_step, .slice_rows_signed_step, .stride_rows, .repeat_rows, .tile_rows, .sample_rows, .sample_rows_fraction, .sample_rows_with_replacement, .sample_rows_fraction_with_replacement, .roll_rows, .shift_rows, .reverse_rows, .head, .tail => {},
    }
    self.* = undefined;
}
