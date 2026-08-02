//! Payload type definitions for `DeviceLazyOp`.
//!
//! The lazy operation union remains in `dataframe_lazy_op.zig`, but the payload
//! structs live here so adding a new lazy operation does not make the union file
//! absorb every field-level detail.

const options_mod = @import("dataframe_options.zig");
const profile_payloads = @import("dataframe_lazy_op_profile_payloads.zig");
const array_mod = @import("array.zig");

const DeviceColumnBinaryOp = options_mod.DeviceColumnBinaryOp;
const DeviceColumnCompareOp = options_mod.DeviceColumnCompareOp;
const DeviceColumnLogicalOp = options_mod.DeviceColumnLogicalOp;
const DeviceDTypeClass = options_mod.DeviceDTypeClass;
const DeviceScalar = options_mod.DeviceScalar;
const DeviceSortOptions = options_mod.DeviceSortOptions;
const DeviceJoinOptions = options_mod.DeviceJoinOptions;
const DeviceAsofOptions = options_mod.DeviceAsofOptions;

pub const DeviceLazyGroupByAggregation = enum {
    sum,
    prod,
    min,
    max,
    mean,
    first,
    last,
    first_row,
    last_row,
    nth,
    nth_row,
    nth_index,
    nth_row_index,
    n_unique,
    mode,
    mode_count,
    mode_ratio,
    mode_margin,
    mode_margin_ratio,
    entropy,
    gini_impurity,
    perplexity,
    inverse_simpson,
    simpson_concentration,
    evenness,
    gini_mean_diff,
    gini_coefficient,
    mean_abs_dev,
    mean_abs_dev_ratio,
    median,
    quantile,
    iqr,
    mad,
    trimmed_mean,
    winsorized_mean,
    interdecile_range,
    midhinge,
    trimean,
    bowley_skewness,
    quartile_coeff_dispersion,
    kelley_skewness,
    variance,
    stddev,
    sem,
    cv,
    fano,
    skewness,
    kurtosis,
    magnitude_variance,
    magnitude_stddev,
    magnitude_sem,
    magnitude_cv,
    magnitude_fano,
    magnitude_skewness,
    magnitude_kurtosis,
    mean_abs,
    mean_square,
    rms,
    l1_norm,
    l2_norm,
    max_abs,
    min_abs,
    hhi,
    magnitude_normalized_hhi,
    magnitude_sparsity,
    magnitude_inverse_simpson,
    magnitude_simpson_evenness,
    magnitude_dominance,
    magnitude_dominance_margin,
    magnitude_entropy,
    magnitude_perplexity,
    magnitude_evenness,
    geometric_mean,
    harmonic_mean,
    logsumexp,
    logmeanexp,
    ptp,
    midrange,
    range_coeff,
    any,
    all,
    true_count,
    false_count,
    true_ratio,
    false_ratio,
    first_true_index,
    last_true_index,
    first_false_index,
    last_false_index,
    any_valid,
    all_valid,
    any_null,
    all_null,
    valid_count,
    null_count,
    valid_ratio,
    null_ratio,
    first_valid_index,
    last_valid_index,
    first_null_index,
    last_null_index,
    nan_count,
    nan_ratio,
    inf_count,
    inf_ratio,
    positive_inf_count,
    positive_inf_ratio,
    negative_inf_count,
    negative_inf_ratio,
    first_nan_index,
    last_nan_index,
    first_inf_index,
    last_inf_index,
    first_positive_inf_index,
    last_positive_inf_index,
    first_negative_inf_index,
    last_negative_inf_index,
    finite_count,
    finite_ratio,
    first_finite_index,
    last_finite_index,
    normal_count,
    normal_ratio,
    first_normal_index,
    last_normal_index,
    subnormal_count,
    subnormal_ratio,
    first_subnormal_index,
    last_subnormal_index,
    non_finite_count,
    non_finite_ratio,
    first_non_finite_index,
    last_non_finite_index,
    zero_count,
    zero_ratio,
    first_zero_index,
    last_zero_index,
    positive_zero_count,
    positive_zero_ratio,
    negative_zero_count,
    negative_zero_ratio,
    first_positive_zero_index,
    last_positive_zero_index,
    first_negative_zero_index,
    last_negative_zero_index,
    non_zero_count,
    non_zero_ratio,
    first_non_zero_index,
    last_non_zero_index,
    positive_count,
    positive_ratio,
    first_positive_index,
    last_positive_index,
    signbit_count,
    signbit_ratio,
    first_signbit_index,
    last_signbit_index,
    negative_count,
    negative_ratio,
    first_negative_index,
    last_negative_index,
    argmin,
    argmax,
};

pub const DeviceLazyWeightedGroupByAggregation = enum {
    weighted_sum,
    weighted_product,
    weighted_weight_sum,
    weighted_positive_count,
    weighted_effective_n,
    weighted_mean,
    weighted_mean_square,
    weighted_rms,
    weighted_min,
    weighted_max,
    weighted_mean_abs,
    weighted_l1_norm,
    weighted_l2_norm,
    weighted_max_abs,
    weighted_min_abs,
    weighted_geometric_mean,
    weighted_harmonic_mean,
    weighted_logsumexp,
    weighted_logmeanexp,
    weighted_range,
    weighted_midrange,
    weighted_range_coeff,
    weighted_variance,
    weighted_stddev,
    weighted_sem,
    weighted_cv,
    weighted_fano,
    weighted_skewness,
    weighted_kurtosis,
    weighted_quantile,
    weighted_median,
    weighted_iqr,
    weighted_mad,
    weighted_mode,
    weighted_mode_weight,
    weighted_mode_ratio,
    weighted_mode_margin,
    weighted_mode_margin_ratio,
    weighted_entropy,
    weighted_gini_impurity,
    weighted_perplexity,
    weighted_inverse_simpson,
    weighted_simpson_concentration,
    weighted_evenness,
};

pub const DeviceLazyPairGroupByAggregation = enum {
    dot,
    cosine_similarity,
    squared_euclidean_distance,
    euclidean_distance,
    manhattan_distance,
    chebyshev_distance,
    canberra_distance,
    bray_curtis_distance,
    mean_error,
    mae,
    mse,
    rmse,
    mape,
    smape,
    pair_count,
    covariance,
    correlation,
    beta,
};

pub const DeviceLazyWeightedPairGroupByAggregation = enum {
    weighted_dot,
    weighted_cosine_similarity,
    weighted_squared_euclidean_distance,
    weighted_euclidean_distance,
    weighted_manhattan_distance,
    weighted_chebyshev_distance,
    weighted_canberra_distance,
    weighted_bray_curtis_distance,
    weighted_mean_error,
    weighted_mae,
    weighted_mse,
    weighted_rmse,
    weighted_mape,
    weighted_smape,
    weighted_covariance,
    weighted_correlation,
    weighted_beta,
};

pub const DeviceLazyJoinKind = enum {
    inner,
    left,
    full,
    semi,
    anti,
};

pub fn DeviceLazyPayloads(comptime DeviceDataFrame: type, comptime DeviceColumn: type) type {
    return struct {
        pub const Select = [][]const u8;
        pub const ColumnIndices = []usize;
        pub const ColumnRange = struct {
            start: usize,
            stop: usize,
        };
        pub const SortColumnsByName = struct {
            descending: bool,
        };
        pub const NamePattern = struct {
            pattern: []const u8,
        };
        pub const DTypes = []array_mod.DType;
        pub const DTypeClass = DeviceDTypeClass;
        pub const RowIndex = struct {
            name: []const u8,
            offset: usize,
        };
        pub const RenameColumn = struct {
            old_name: []const u8,
            new_name: []const u8,
        };
        pub const RenameColumns = struct {
            old_names: [][]const u8,
            new_names: [][]const u8,
        };
        pub const RenameNamePattern = struct {
            pattern: []const u8,
        };
        pub const RenameNameReplacement = struct {
            old_pattern: []const u8,
            new_pattern: []const u8,
        };
        pub const MoveColumn = struct {
            name: []const u8,
            target_index: usize,
        };
        pub const MoveColumnRelative = struct {
            name: []const u8,
            anchor_name: []const u8,
        };
        pub const CopyColumn = struct {
            source_name: []const u8,
            new_name: []const u8,
        };
        pub const CopyColumnAt = struct {
            source_name: []const u8,
            new_name: []const u8,
            target_index: usize,
        };
        pub const CopyColumnRelative = struct {
            source_name: []const u8,
            new_name: []const u8,
            anchor_name: []const u8,
        };
        pub const DropColumns = [][]const u8;
        pub const DropNulls = [][]const u8;
        pub const WithColumnUnary = struct {
            name: []const u8,
            input_name: []const u8,
        };
        pub const WithColumnBinary = struct {
            name: []const u8,
            lhs_name: []const u8,
            rhs_name: []const u8,
            op: DeviceColumnBinaryOp,
        };
        pub const WithColumnScalar = struct {
            name: []const u8,
            input_name: []const u8,
            op: DeviceColumnBinaryOp,
            scalar: DeviceScalar,
        };
        pub const WithColumnParamUnary = struct {
            name: []const u8,
            input_name: []const u8,
            scalar: DeviceScalar,
        };
        pub const WithColumnParamUnary2 = struct {
            name: []const u8,
            input_name: []const u8,
            lhs_scalar: DeviceScalar,
            rhs_scalar: DeviceScalar,
        };
        pub const WithColumnBetween = struct {
            name: []const u8,
            input_name: []const u8,
            lower: DeviceScalar,
            upper: DeviceScalar,
            lower_inclusive: bool,
            upper_inclusive: bool,
        };
        pub const RangeFilterColumn = struct {
            name: []const u8,
            lower: DeviceScalar,
            upper: DeviceScalar,
            lower_inclusive: bool,
            upper_inclusive: bool,
            keep_inside: bool,
        };
        pub const MembershipFilterColumn = struct {
            input_name: []const u8,
            test_name: []const u8,
            invert: bool,
        };
        pub const WithColumnBinaryParam = struct {
            name: []const u8,
            lhs_name: []const u8,
            rhs_name: []const u8,
            scalar: DeviceScalar,
        };
        pub const WithColumnTernaryParam = struct {
            name: []const u8,
            base_name: []const u8,
            lhs_name: []const u8,
            rhs_name: []const u8,
            scalar: DeviceScalar,
        };
        pub const WithColumnTernary = struct {
            name: []const u8,
            input_name: []const u8,
            lhs_name: []const u8,
            rhs_name: []const u8,
        };
        pub const WithColumnWhereScalar = struct {
            name: []const u8,
            input_name: []const u8,
            mask_name: []const u8,
            scalar: DeviceScalar,
        };
        pub const WithColumnIsIn = struct {
            name: []const u8,
            input_name: []const u8,
            test_name: []const u8,
            invert: bool,
        };
        pub const WithColumnIsInValues = struct {
            name: []const u8,
            input_name: []const u8,
            values: DeviceColumn,
            invert: bool,
        };
        pub const FilterIsInValues = struct {
            input_name: []const u8,
            values: DeviceColumn,
            invert: bool,
        };
        pub const WithColumnPutFlatScalar = struct {
            name: []const u8,
            input_name: []const u8,
            row_indices: []const usize,
            scalar: DeviceScalar,
        };
        pub const WithColumnPutFlat = struct {
            name: []const u8,
            input_name: []const u8,
            row_indices: []const usize,
            value_name: []const u8,
        };
        pub const WithColumnPutFlatScalarMode = struct {
            name: []const u8,
            input_name: []const u8,
            row_indices: []const usize,
            scalar: DeviceScalar,
            mode: array_mod.IndexMode,
        };
        pub const WithColumnPutFlatScalarSigned = struct {
            name: []const u8,
            input_name: []const u8,
            row_indices: []const isize,
            scalar: DeviceScalar,
        };
        pub const WithColumnIsCloseScalar = struct {
            name: []const u8,
            input_name: []const u8,
            scalar: DeviceScalar,
            rtol: DeviceScalar,
            atol: DeviceScalar,
            equal_nan: bool,
        };
        pub const WithColumnLogicalScalar = struct {
            name: []const u8,
            input_name: []const u8,
            op: DeviceColumnLogicalOp,
            scalar: bool,
        };
        pub const WithColumnLogical = struct {
            name: []const u8,
            lhs_name: []const u8,
            rhs_name: []const u8,
            op: DeviceColumnLogicalOp,
        };
        pub const WithColumnLdexpScalar = struct {
            name: []const u8,
            input_name: []const u8,
            exponent: i32,
        };
        pub const WithColumnLiteral = struct {
            name: []const u8,
            scalar: DeviceScalar,
        };
        pub const WithColumnLiteralAt = struct {
            name: []const u8,
            scalar: DeviceScalar,
            target_index: usize,
        };
        pub const WithColumnLiteralRelative = struct {
            name: []const u8,
            scalar: DeviceScalar,
            anchor_name: []const u8,
        };
        pub const CastColumn = struct {
            name: []const u8,
            dtype: array_mod.DType,
        };
        pub const FillNullColumn = struct {
            name: []const u8,
            scalar: DeviceScalar,
        };
        pub const CoalesceColumns = struct {
            primary_name: []const u8,
            fallback_name: []const u8,
            output_name: []const u8,
        };
        pub const NullPredicateColumn = struct {
            name: []const u8,
            output_name: []const u8,
        };
        pub const RowValidityCount = struct {
            names: [][]const u8,
            output_name: []const u8,
        };
        pub const GroupShift = struct {
            names: [][]const u8,
            value_name: []const u8,
            output_name: []const u8,
            offset: usize,
        };
        pub const GroupShiftQuantile = struct {
            names: [][]const u8,
            value_name: []const u8,
            output_name: []const u8,
            quantile: f64,
        };
        pub const GroupWeightedShift = struct {
            names: [][]const u8,
            value_name: []const u8,
            weight_name: []const u8,
            output_name: []const u8,
        };
        pub const GroupWeightedShiftQuantile = struct {
            names: [][]const u8,
            value_name: []const u8,
            weight_name: []const u8,
            output_name: []const u8,
            quantile: f64,
        };
        pub const GroupWeightedPairShift = struct {
            names: [][]const u8,
            lhs_name: []const u8,
            rhs_name: []const u8,
            weight_name: []const u8,
            output_name: []const u8,
            correction: f64 = 0.0,
        };
        pub const RowColumnOutputs = struct {
            names: [][]const u8,
            output_names: [][]const u8,
        };
        pub const RowColumnOutputsDispersion = struct {
            names: [][]const u8,
            output_names: [][]const u8,
            correction: f64,
        };
        pub const RowWeightedMean = struct {
            value_names: [][]const u8,
            weight_names: [][]const u8,
            output_name: []const u8,
        };
        pub const RowWeightedDispersion = struct {
            value_names: [][]const u8,
            weight_names: [][]const u8,
            output_name: []const u8,
            correction: f64,
        };
        pub const RowWeightedPair = struct {
            lhs_names: [][]const u8,
            rhs_names: [][]const u8,
            weight_names: [][]const u8,
            output_name: []const u8,
            correction: f64,
        };
        pub const RowWeightedQuantile = struct {
            value_names: [][]const u8,
            weight_names: [][]const u8,
            output_name: []const u8,
            q: f64,
        };
        pub const RowQuantile = struct {
            names: [][]const u8,
            output_name: []const u8,
            q: f64,
        };
        pub const RowQuantileRange = struct {
            names: [][]const u8,
            output_name: []const u8,
            low_q: f64,
            high_q: f64,
        };
        pub const RowTrimmedMean = struct {
            names: [][]const u8,
            output_name: []const u8,
            trim_fraction: f64,
        };
        pub const RowWinsorizedMean = struct {
            names: [][]const u8,
            output_name: []const u8,
            winsor_fraction: f64,
        };
        pub const RowNumericDispersion = struct {
            names: [][]const u8,
            output_name: []const u8,
            correction: f64,
        };
        pub const WithColumnCompare = struct {
            name: []const u8,
            lhs_name: []const u8,
            rhs_name: []const u8,
            op: DeviceColumnCompareOp,
        };
        pub const WithColumnCompareScalar = struct {
            name: []const u8,
            input_name: []const u8,
            op: DeviceColumnCompareOp,
            scalar: DeviceScalar,
        };
        pub const FilterMask = DeviceColumn;
        pub const FilterColumn = []const u8;
        pub const FilterScalar = struct {
            name: []const u8,
            op: DeviceColumnCompareOp,
            scalar: DeviceScalar,
            keep_matches: bool = true,
        };
        pub const NullIfValues = struct {
            name: []const u8,
            values: DeviceColumn,
        };
        pub const GroupByCount = struct {
            key_name: []const u8,
            output_name: []const u8,
        };
        pub const GroupByCountOn = struct {
            key_names: [][]const u8,
            output_name: []const u8,
        };
        pub const GroupByRows = struct {
            key_name: []const u8,
            start: usize = 0,
            signed_start: isize = 0,
            use_signed_start: bool = false,
            step: usize = 1,
            n: usize,
            keep_tail: bool,
        };
        pub const GroupByRowsOn = struct {
            key_names: [][]const u8,
            start: usize = 0,
            signed_start: isize = 0,
            use_signed_start: bool = false,
            step: usize = 1,
            n: usize,
            keep_tail: bool,
        };
        pub const GroupBySortedRows = struct {
            key_name: []const u8,
            sort_name: []const u8,
            n: usize,
            options: DeviceSortOptions,
            keep_bottom: bool,
        };
        pub const GroupBySortedRowsOn = struct {
            key_names: [][]const u8,
            sort_name: []const u8,
            n: usize,
            options: DeviceSortOptions,
            keep_bottom: bool,
        };
        pub const GroupBySortedRowsColumns = struct {
            key_name: []const u8,
            sort_names: [][]const u8,
            n: usize,
            options: []DeviceSortOptions,
            keep_bottom: bool,
        };
        pub const GroupBySortedRowsColumnsOn = struct {
            key_names: [][]const u8,
            sort_names: [][]const u8,
            n: usize,
            options: []DeviceSortOptions,
            keep_bottom: bool,
        };
        pub const GroupByValue = struct {
            key_name: []const u8,
            value_name: []const u8,
            output_name: []const u8,
            aggregation: DeviceLazyGroupByAggregation,
            quantile: f64 = 0.5,
            index: usize = 0,
        };
        pub const GroupByValueOn = struct {
            key_names: [][]const u8,
            value_name: []const u8,
            output_name: []const u8,
            aggregation: DeviceLazyGroupByAggregation,
            quantile: f64 = 0.5,
            index: usize = 0,
        };
        pub const GroupByWeighted = struct {
            key_name: []const u8,
            value_name: []const u8,
            weight_name: []const u8,
            output_name: []const u8,
            aggregation: DeviceLazyWeightedGroupByAggregation,
            quantile: f64 = 0.5,
        };
        pub const GroupByWeightedOn = struct {
            key_names: [][]const u8,
            value_name: []const u8,
            weight_name: []const u8,
            output_name: []const u8,
            aggregation: DeviceLazyWeightedGroupByAggregation,
            quantile: f64 = 0.5,
        };
        pub const GroupByPair = struct {
            key_name: []const u8,
            lhs_name: []const u8,
            rhs_name: []const u8,
            output_name: []const u8,
            aggregation: DeviceLazyPairGroupByAggregation,
        };
        pub const GroupByPairOn = struct {
            key_names: [][]const u8,
            lhs_name: []const u8,
            rhs_name: []const u8,
            output_name: []const u8,
            aggregation: DeviceLazyPairGroupByAggregation,
        };
        pub const GroupByWeightedPair = struct {
            key_name: []const u8,
            lhs_name: []const u8,
            rhs_name: []const u8,
            weight_name: []const u8,
            output_name: []const u8,
            aggregation: DeviceLazyWeightedPairGroupByAggregation,
            correction: f64 = 0.0,
        };
        pub const GroupByWeightedPairOn = struct {
            key_names: [][]const u8,
            lhs_name: []const u8,
            rhs_name: []const u8,
            weight_name: []const u8,
            output_name: []const u8,
            aggregation: DeviceLazyWeightedPairGroupByAggregation,
            correction: f64 = 0.0,
        };
        pub const GroupByOutput = struct {
            key_name: []const u8,
            value_name: []const u8,
            output_prefix: []const u8,
        };
        pub const GroupByOutputOn = struct {
            key_names: [][]const u8,
            value_name: []const u8,
            output_prefix: []const u8,
        };
        pub const JoinOn = struct {
            kind: DeviceLazyJoinKind,
            right: DeviceDataFrame,
            left_key_names: [][]const u8,
            right_key_names: [][]const u8,
            options: DeviceJoinOptions,
        };
        pub const AsofJoin = struct {
            right: DeviceDataFrame,
            left_key_name: []const u8,
            right_key_name: []const u8,
            options: DeviceAsofOptions,
        };
        pub const ConcatRows = DeviceDataFrame;
        pub const DistinctOn = [][]const u8;
        pub const SortBy = struct {
            name: []const u8,
            options: DeviceSortOptions,
        };
        pub const SortByColumns = struct {
            names: [][]const u8,
            options: []DeviceSortOptions,
        };
        pub const TopK = struct {
            name: []const u8,
            options: DeviceSortOptions,
            k: usize,
        };
        pub const TopKColumns = struct {
            names: [][]const u8,
            options: []DeviceSortOptions,
            k: usize,
        };
        pub const RowSlice = struct {
            start: usize,
            stop: usize,
        };
        pub const RowSliceSigned = struct {
            start: isize,
            length: usize,
        };
        pub const RowStride = struct {
            start: usize,
            step: usize,
        };
        pub const RowSliceStep = struct {
            start: usize,
            stop: usize,
            step: usize,
        };
        pub const RowSliceSignedStep = struct {
            start: isize,
            stop: isize,
            step: usize,
        };
        pub const RowSample = struct {
            count: usize,
            seed: u64,
        };
        pub const RowSampleFraction = struct {
            fraction: f64,
            seed: u64,
        };
        pub const RowTake = []usize;
        pub const RowTakeOptional = []?usize;
        pub const RowTakeMode = struct {
            row_indices: []usize,
            mode: array_mod.IndexMode,
        };
        pub const RowTakeSigned = []isize;
        pub const RowTakeSignedMode = struct {
            row_indices: []isize,
            mode: array_mod.IndexMode,
        };
        pub const RowTakeByColumnMode = struct {
            name: []const u8,
            mode: array_mod.IndexMode,
        };
        pub const RankProfileBy = profile_payloads.RankProfileBy;
        pub const RollingProfile = profile_payloads.RollingProfile;
        pub const ExpandingProfile = profile_payloads.ExpandingProfile;
        pub const RollingRobustProfile = profile_payloads.RollingRobustProfile;
        pub const RollingRankProfile = profile_payloads.RollingRankProfile;
        pub const LagProfile = profile_payloads.LagProfile;
        pub const ClipProfile = profile_payloads.ClipProfile;
        pub const RollingClipProfile = profile_payloads.RollingClipProfile;
        pub const ExpandingClipProfile = profile_payloads.ExpandingClipProfile;
        pub const ThresholdProfile = profile_payloads.ThresholdProfile;
        pub const RollingThresholdProfile = profile_payloads.RollingThresholdProfile;
        pub const ExpandingThresholdProfile = profile_payloads.ExpandingThresholdProfile;
        pub const ExpandingRankProfile = profile_payloads.ExpandingRankProfile;
        pub const ExpandingRobustProfile = profile_payloads.ExpandingRobustProfile;
        pub const StandardizeProfile = profile_payloads.StandardizeProfile;
        pub const RobustProfile = profile_payloads.RobustProfile;
        pub const DrawdownProfile = profile_payloads.DrawdownProfile;
        pub const ExtremaProfile = profile_payloads.ExtremaProfile;
        pub const TrendProfile = profile_payloads.TrendProfile;
        pub const RollingTrendProfile = profile_payloads.RollingTrendProfile;
        pub const ExpandingTrendProfile = profile_payloads.ExpandingTrendProfile;
        pub const ChangePointProfile = profile_payloads.ChangePointProfile;
        pub const RollingChangePointProfile = profile_payloads.RollingChangePointProfile;
        pub const ExpandingChangePointProfile = profile_payloads.ExpandingChangePointProfile;
        pub const RollingSignProfile = profile_payloads.RollingSignProfile;
        pub const ExpandingSignProfile = profile_payloads.ExpandingSignProfile;
        pub const CrossoverProfile = profile_payloads.CrossoverProfile;
        pub const RollingCrossoverProfile = profile_payloads.RollingCrossoverProfile;
        pub const ExpandingCrossoverProfile = profile_payloads.ExpandingCrossoverProfile;
        pub const BucketProfile = profile_payloads.BucketProfile;
        pub const EmaProfile = profile_payloads.EmaProfile;
        pub const LinearFitProfile = profile_payloads.LinearFitProfile;
        pub const PairOutput = profile_payloads.ActualPredictedOutput;
        pub const RollingPairOutput = profile_payloads.RollingPairOutput;
        pub const ExpandingPairOutput = profile_payloads.ExpandingPairOutput;
        pub const BoolTransitionProfile = profile_payloads.TrendProfile;
        pub const RollingBoolTransitionProfile = profile_payloads.RollingBoolTransitionProfile;
        pub const ExpandingBoolTransitionProfile = profile_payloads.ExpandingBoolTransitionProfile;
        pub const RollingCorrelationProfile = profile_payloads.RollingCorrelationProfile;
        pub const ExpandingXYProfile = profile_payloads.ExpandingXYProfile;
        pub const RollingLinearFitProfile = profile_payloads.RollingLinearFitProfile;
        pub const ValidityProfile = profile_payloads.NameOutput;
        pub const RollingValidityProfile = profile_payloads.RollingProfile;
        pub const ExpandingValidityProfile = profile_payloads.ExpandingProfile;
        pub const HeadTail = usize;
    };
}
