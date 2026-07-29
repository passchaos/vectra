//! Profile-related payload type definitions for `DeviceLazyOp`.

const options_mod = @import("dataframe_options.zig");

const DeviceSortOptions = options_mod.DeviceSortOptions;
const DeviceRollingOptions = options_mod.DeviceRollingOptions;
const DeviceLagOptions = options_mod.DeviceLagOptions;
const DeviceExpandingOptions = options_mod.DeviceExpandingOptions;
const DeviceExpandingRankOptions = options_mod.DeviceExpandingRankOptions;
const DeviceStandardizeOptions = options_mod.DeviceStandardizeOptions;
const DeviceRobustOptions = options_mod.DeviceRobustOptions;
const DeviceDrawdownOptions = options_mod.DeviceDrawdownOptions;
const DeviceExtremaOptions = options_mod.DeviceExtremaOptions;
const DeviceTrendOptions = options_mod.DeviceTrendOptions;
const DeviceCrossoverOptions = options_mod.DeviceCrossoverOptions;
const DeviceBucketOptions = options_mod.DeviceBucketOptions;
const DeviceEmaOptions = options_mod.DeviceEmaOptions;
const DeviceLinearFitOptions = options_mod.DeviceLinearFitOptions;
const DeviceClipOptions = options_mod.DeviceClipOptions;
const DeviceThresholdOptions = options_mod.DeviceThresholdOptions;
const DeviceRollingCorrelationOptions = options_mod.DeviceRollingCorrelationOptions;
const DeviceRollingRankOptions = options_mod.DeviceRollingRankOptions;
const DeviceRollingRobustOptions = options_mod.DeviceRollingRobustOptions;

pub const NameOutput = struct {
    name: []const u8,
    output_prefix: []const u8,
};

pub fn NameOutputOptions(comptime Options: type) type {
    return struct {
        name: []const u8,
        output_prefix: []const u8,
        options: Options,
    };
}

pub const ActualPredictedOutput = struct {
    actual_name: []const u8,
    predicted_name: []const u8,
    output_prefix: []const u8,
};

pub const RankProfileBy = NameOutputOptions(DeviceSortOptions);
pub const RollingProfile = NameOutputOptions(DeviceRollingOptions);
pub const ExpandingProfile = NameOutputOptions(DeviceExpandingOptions);
pub const RollingRobustProfile = NameOutputOptions(DeviceRollingRobustOptions);
pub const RollingRankProfile = NameOutputOptions(DeviceRollingRankOptions);
pub const LagProfile = NameOutputOptions(DeviceLagOptions);
pub const ClipProfile = NameOutputOptions(DeviceClipOptions);
pub const ThresholdProfile = NameOutputOptions(DeviceThresholdOptions);
pub const ExpandingRankProfile = NameOutputOptions(DeviceExpandingRankOptions);
pub const ExpandingRobustProfile = NameOutputOptions(DeviceRobustOptions);
pub const StandardizeProfile = NameOutputOptions(DeviceStandardizeOptions);
pub const RobustProfile = NameOutputOptions(DeviceRobustOptions);
pub const DrawdownProfile = NameOutputOptions(DeviceDrawdownOptions);
pub const ExtremaProfile = NameOutputOptions(DeviceExtremaOptions);
pub const TrendProfile = NameOutputOptions(DeviceTrendOptions);
pub const BucketProfile = NameOutputOptions(DeviceBucketOptions);
pub const EmaProfile = NameOutputOptions(DeviceEmaOptions);
pub const BoolTransitionProfile = NameOutputOptions(DeviceTrendOptions);
pub const ValidityProfile = NameOutput;
pub const RollingValidityProfile = NameOutputOptions(DeviceRollingOptions);
pub const ExpandingValidityProfile = NameOutputOptions(DeviceExpandingOptions);

pub const RollingClipProfile = struct {
    name: []const u8,
    output_prefix: []const u8,
    clip_options: DeviceClipOptions,
    options: DeviceRollingOptions,
};

pub const ExpandingClipProfile = struct {
    name: []const u8,
    output_prefix: []const u8,
    clip_options: DeviceClipOptions,
    options: DeviceExpandingOptions,
};

pub const RollingThresholdProfile = struct {
    name: []const u8,
    output_prefix: []const u8,
    threshold: f64,
    options: DeviceRollingOptions,
};

pub const ExpandingThresholdProfile = struct {
    name: []const u8,
    output_prefix: []const u8,
    threshold: f64,
    options: DeviceExpandingOptions,
};

pub const RollingTrendProfile = struct {
    name: []const u8,
    output_prefix: []const u8,
    trend_options: DeviceTrendOptions,
    options: DeviceRollingOptions,
};

pub const ExpandingTrendProfile = struct {
    name: []const u8,
    output_prefix: []const u8,
    trend_options: DeviceTrendOptions,
    options: DeviceExpandingOptions,
};

pub const ChangePointProfile = struct {
    name: []const u8,
    output_prefix: []const u8,
    threshold: f64,
    options: DeviceTrendOptions,
};

pub const RollingChangePointProfile = struct {
    name: []const u8,
    output_prefix: []const u8,
    threshold: f64,
    change_options: DeviceTrendOptions,
    options: DeviceRollingOptions,
};

pub const ExpandingChangePointProfile = struct {
    name: []const u8,
    output_prefix: []const u8,
    threshold: f64,
    change_options: DeviceTrendOptions,
    options: DeviceExpandingOptions,
};

pub const RollingSignProfile = struct {
    name: []const u8,
    output_prefix: []const u8,
    sign_options: DeviceTrendOptions,
    options: DeviceRollingOptions,
};

pub const ExpandingSignProfile = struct {
    name: []const u8,
    output_prefix: []const u8,
    sign_options: DeviceTrendOptions,
    options: DeviceExpandingOptions,
};

pub const CrossoverProfile = struct {
    lhs_name: []const u8,
    rhs_name: []const u8,
    output_prefix: []const u8,
    options: DeviceCrossoverOptions,
};

pub const RollingCrossoverProfile = struct {
    lhs_name: []const u8,
    rhs_name: []const u8,
    output_prefix: []const u8,
    cross_options: DeviceCrossoverOptions,
    options: DeviceRollingOptions,
};

pub const ExpandingCrossoverProfile = struct {
    lhs_name: []const u8,
    rhs_name: []const u8,
    output_prefix: []const u8,
    cross_options: DeviceCrossoverOptions,
    options: DeviceExpandingOptions,
};

pub const LinearFitProfile = struct {
    x_name: []const u8,
    y_name: []const u8,
    output_prefix: []const u8,
    options: DeviceLinearFitOptions,
};

pub const PairOutput = ActualPredictedOutput;

pub const RollingPairOutput = struct {
    actual_name: []const u8,
    predicted_name: []const u8,
    output_prefix: []const u8,
    options: DeviceRollingOptions,
};

pub const ExpandingPairOutput = struct {
    actual_name: []const u8,
    predicted_name: []const u8,
    output_prefix: []const u8,
    options: DeviceExpandingOptions,
};

pub const RollingBoolTransitionProfile = struct {
    name: []const u8,
    output_prefix: []const u8,
    transition_options: DeviceTrendOptions,
    options: DeviceRollingOptions,
};

pub const ExpandingBoolTransitionProfile = struct {
    name: []const u8,
    output_prefix: []const u8,
    transition_options: DeviceTrendOptions,
    options: DeviceExpandingOptions,
};

pub const RollingCorrelationProfile = struct {
    x_name: []const u8,
    y_name: []const u8,
    output_prefix: []const u8,
    options: DeviceRollingCorrelationOptions,
};

pub const ExpandingXYProfile = struct {
    x_name: []const u8,
    y_name: []const u8,
    output_prefix: []const u8,
    options: DeviceExpandingOptions,
};

pub const RollingLinearFitProfile = struct {
    x_name: []const u8,
    y_name: []const u8,
    output_prefix: []const u8,
    options: DeviceRollingCorrelationOptions,
};
