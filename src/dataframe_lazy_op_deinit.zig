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
        .row_null_count, .row_valid_count, .row_null_ratio, .row_valid_ratio, .row_first_valid_index, .row_last_valid_index, .row_first_null_index, .row_last_null_index, .row_argmin, .row_argmax, .row_median, .row_iqr, .row_interdecile_range, .row_midhinge, .row_trimean, .row_bowley_skewness, .row_quartile_coeff_dispersion, .row_kelley_skewness, .row_mad, .row_mode, .row_entropy, .row_gini_impurity, .row_perplexity, .row_inverse_simpson, .row_simpson_concentration, .row_evenness, .row_mode_count, .row_mode_ratio, .row_mode_margin, .row_mode_margin_ratio, .row_count_distinct, .row_n_unique, .row_sum, .row_mean, .row_logsumexp, .row_logmeanexp, .row_softmax_entropy, .row_softmax_perplexity, .row_softmax_confidence, .row_softmax_margin, .row_softmax_evenness, .row_softmax_concentration, .row_softmax_normalized_hhi, .row_softmax_gini_impurity, .row_softmax_inverse_simpson, .row_softmax_simpson_evenness, .row_logit_margin, .row_geometric_mean, .row_magnitude_geometric_mean, .row_harmonic_mean, .row_skewness, .row_magnitude_skewness, .row_kurtosis, .row_magnitude_kurtosis, .row_prod, .row_min, .row_max, .row_ptp, .row_magnitude_ptp, .row_midrange, .row_magnitude_midrange, .row_range_coeff, .row_magnitude_range_coeff, .row_mean_abs, .row_hhi, .row_magnitude_normalized_hhi, .row_magnitude_sparsity, .row_magnitude_inverse_simpson, .row_magnitude_simpson_evenness, .row_magnitude_dominance, .row_magnitude_dominance_margin, .row_magnitude_entropy, .row_magnitude_perplexity, .row_magnitude_evenness, .row_mean_abs_dev, .row_gini_mean_diff, .row_gini_coefficient, .row_mean_abs_dev_ratio, .row_rms, .row_l1_norm, .row_l2_norm, .row_true_count, .row_false_count, .row_any_true, .row_all_true, .row_any_false, .row_all_false, .row_first_true_index, .row_last_true_index, .row_first_false_index, .row_last_false_index, .row_true_ratio, .row_false_ratio, .row_nan_count, .row_nan_ratio, .row_inf_count, .row_inf_ratio, .row_positive_inf_count, .row_negative_inf_count, .row_positive_inf_ratio, .row_negative_inf_ratio, .row_zero_count, .row_zero_ratio, .row_positive_zero_count, .row_negative_zero_count, .row_positive_zero_ratio, .row_negative_zero_ratio, .row_non_zero_count, .row_non_zero_ratio, .row_positive_count, .row_positive_ratio, .row_signbit_count, .row_signbit_ratio, .row_negative_count, .row_negative_ratio, .row_finite_count, .row_finite_ratio, .row_normal_count, .row_normal_ratio, .row_subnormal_count, .row_subnormal_ratio, .row_non_finite_count, .row_non_finite_ratio => |row_count| {
            freeNameList(allocator, row_count.names);
            allocator.free(row_count.output_name);
        },
        .row_centered, .row_zscore, .row_minmax_scale, .row_l2_normalize, .row_l1_normalize, .row_sum_normalize, .row_max_abs_normalize, .row_softmax, .row_log_softmax, .row_softmin, .row_log_softmin => |row_outputs| {
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
        .row_pair_count, .row_weighted_mean, .row_weighted_median, .row_weighted_iqr, .row_weighted_mad, .row_weighted_mode, .row_weighted_mode_weight, .row_weighted_mode_ratio, .row_weighted_mode_margin, .row_weighted_mode_margin_ratio, .row_weighted_entropy, .row_weighted_gini_impurity, .row_weighted_perplexity, .row_weighted_inverse_simpson, .row_weighted_simpson_concentration, .row_weighted_evenness, .row_dot, .row_cosine_similarity, .row_squared_euclidean_distance, .row_euclidean_distance, .row_manhattan_distance, .row_chebyshev_distance, .row_canberra_distance, .row_bray_curtis_distance, .row_mean_error, .row_mae, .row_mse, .row_rmse, .row_mape, .row_smape, .row_covariance, .row_correlation, .row_beta => |row_weighted| {
            freeNameList(allocator, row_weighted.value_names);
            freeNameList(allocator, row_weighted.weight_names);
            allocator.free(row_weighted.output_name);
        },
        .row_weighted_variance, .row_weighted_stddev => |row_weighted| {
            freeNameList(allocator, row_weighted.value_names);
            freeNameList(allocator, row_weighted.weight_names);
            allocator.free(row_weighted.output_name);
        },
        .row_weighted_quantile => |row_weighted| {
            freeNameList(allocator, row_weighted.value_names);
            freeNameList(allocator, row_weighted.weight_names);
            allocator.free(row_weighted.output_name);
        },
        .row_weighted_covariance, .row_weighted_correlation, .row_weighted_beta => |row_weighted| {
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
        .drop_rows_by_mask_column => |name| allocator.free(name),
        .where_indices_column => |predicate| {
            allocator.free(predicate.name);
            allocator.free(predicate.output_name);
        },
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
        .distinct_rows, .slice_rows, .slice_rows_signed, .drop_row_range, .drop_last_rows, .slice_rows_step, .slice_rows_signed_step, .stride_rows, .repeat_rows, .tile_rows, .sample_rows, .sample_rows_with_replacement, .roll_rows, .shift_rows, .reverse_rows, .head, .tail => {},
    }
    self.* = undefined;
}
