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
    n_unique,
    mode,
    median,
    quantile,
    iqr,
    mad,
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
    mean_abs,
    mean_square,
    rms,
    l1_norm,
    l2_norm,
    max_abs,
    min_abs,
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
    valid_count,
    null_count,
    valid_ratio,
    null_ratio,
    argmin,
    argmax,
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
        pub const GroupByValue = struct {
            key_name: []const u8,
            value_name: []const u8,
            output_name: []const u8,
            aggregation: DeviceLazyGroupByAggregation,
            quantile: f64 = 0.5,
        };
        pub const GroupByValueOn = struct {
            key_names: [][]const u8,
            value_name: []const u8,
            output_name: []const u8,
            aggregation: DeviceLazyGroupByAggregation,
            quantile: f64 = 0.5,
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
