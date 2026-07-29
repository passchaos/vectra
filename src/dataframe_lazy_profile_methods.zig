//! Public lazy-frame profile method wrappers.
//!
//! These wrappers all lower to small lazy operation builders. Keeping them out
//! of `dataframe_lazy_frame.zig` prevents the generic lazy frame facade from
//! growing with every eager profile family while preserving method-call syntax
//! through aliases inside `DeviceLazyFrame`.

const std = @import("std");
const array_mod = @import("array.zig");
const lazy_profile_mod = @import("dataframe_lazy_profile_plan.zig");
const lazy_sort_mod = @import("dataframe_lazy_sort_plan.zig");
const options_mod = @import("dataframe_options.zig");
const series_mod = @import("series.zig");

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
const DeviceDataError = series_mod.DataError || array_mod.ArrayError;

pub fn rollingProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputOptions(self, "rolling_profile", name, output_prefix, options_value);
}

pub fn rollingMomentProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputOptions(self, "rolling_moment_profile", name, output_prefix, options_value);
}

pub fn rollingRangeProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputOptions(self, "rolling_range_profile", name, output_prefix, options_value);
}

pub fn rollingNormalizeProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputOptions(self, "rolling_normalize_profile", name, output_prefix, options_value);
}

pub fn expandingNormalizeProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceExpandingOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputOptions(self, "expanding_normalize_profile", name, output_prefix, options_value);
}

pub fn rollingQuantileProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputOptions(self, "rolling_quantile_profile", name, output_prefix, options_value);
}

pub fn expandingQuantileProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceExpandingOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputOptions(self, "expanding_quantile_profile", name, output_prefix, options_value);
}

pub fn rollingBoolProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputOptions(self, "rolling_bool_profile", name, output_prefix, options_value);
}

pub fn rollingDrawdownProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputOptions(self, "rolling_drawdown_profile", name, output_prefix, options_value);
}

pub fn rollingRobustProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingRobustOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputOptions(self, "rolling_robust_profile", name, output_prefix, options_value);
}

pub fn rollingRankProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingRankOptions) DeviceDataError!void {
    return lazy_sort_mod.rollingRankProfile(self, name, output_prefix, options_value);
}

pub fn lagProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceLagOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputOptions(self, "lag_profile", name, output_prefix, options_value);
}

pub fn leadProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceLagOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputOptions(self, "lead_profile", name, output_prefix, options_value);
}

pub fn clipProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceClipOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputOptions(self, "clip_profile", name, output_prefix, options_value);
}

pub fn rollingClipProfile(self: anytype, name: []const u8, output_prefix: []const u8, clip_options: DeviceClipOptions, options_value: DeviceRollingOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputExtraOptions(self, "rolling_clip_profile", name, output_prefix, "clip_options", clip_options, options_value);
}

pub fn expandingClipProfile(self: anytype, name: []const u8, output_prefix: []const u8, clip_options: DeviceClipOptions, options_value: DeviceExpandingOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputExtraOptions(self, "expanding_clip_profile", name, output_prefix, "clip_options", clip_options, options_value);
}

pub fn thresholdProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceThresholdOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputOptions(self, "threshold_profile", name, output_prefix, options_value);
}

pub fn rollingThresholdProfile(self: anytype, name: []const u8, output_prefix: []const u8, threshold: f64, options_value: DeviceRollingOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputThresholdOptions(self, "rolling_threshold_profile", name, output_prefix, threshold, options_value);
}

pub fn expandingThresholdProfile(self: anytype, name: []const u8, output_prefix: []const u8, threshold: f64, options_value: DeviceExpandingOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputThresholdOptions(self, "expanding_threshold_profile", name, output_prefix, threshold, options_value);
}

pub fn expandingProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceExpandingOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputOptions(self, "expanding_profile", name, output_prefix, options_value);
}

pub fn expandingBoolProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceExpandingOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputOptions(self, "expanding_bool_profile", name, output_prefix, options_value);
}

pub fn expandingRankProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceExpandingRankOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputOptions(self, "expanding_rank_profile", name, output_prefix, options_value);
}

pub fn expandingRobustProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceRobustOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputOptions(self, "expanding_robust_profile", name, output_prefix, options_value);
}

pub fn expandingMomentProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceExpandingOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputOptions(self, "expanding_moment_profile", name, output_prefix, options_value);
}

pub fn standardizeProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceStandardizeOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputOptions(self, "standardize_profile", name, output_prefix, options_value);
}

pub fn robustProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceRobustOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputOptions(self, "robust_profile", name, output_prefix, options_value);
}

pub fn drawdownProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceDrawdownOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputOptions(self, "drawdown_profile", name, output_prefix, options_value);
}

pub fn extremaProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceExtremaOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputOptions(self, "extrema_profile", name, output_prefix, options_value);
}

pub fn trendProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceTrendOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputOptions(self, "trend_profile", name, output_prefix, options_value);
}

pub fn rollingTrendProfile(self: anytype, name: []const u8, output_prefix: []const u8, trend_options: DeviceTrendOptions, options_value: DeviceRollingOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputExtraOptions(self, "rolling_trend_profile", name, output_prefix, "trend_options", trend_options, options_value);
}

pub fn expandingTrendProfile(self: anytype, name: []const u8, output_prefix: []const u8, trend_options: DeviceTrendOptions, options_value: DeviceExpandingOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputExtraOptions(self, "expanding_trend_profile", name, output_prefix, "trend_options", trend_options, options_value);
}

pub fn changePointProfile(self: anytype, name: []const u8, output_prefix: []const u8, threshold: f64, options_value: DeviceTrendOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputThresholdOptions(self, "change_point_profile", name, output_prefix, threshold, options_value);
}

pub fn rollingChangePointProfile(self: anytype, name: []const u8, output_prefix: []const u8, threshold: f64, change_options: DeviceTrendOptions, options_value: DeviceRollingOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputThresholdExtraOptions(self, "rolling_change_point_profile", name, output_prefix, threshold, "change_options", change_options, options_value);
}

pub fn expandingChangePointProfile(self: anytype, name: []const u8, output_prefix: []const u8, threshold: f64, change_options: DeviceTrendOptions, options_value: DeviceExpandingOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputThresholdExtraOptions(self, "expanding_change_point_profile", name, output_prefix, threshold, "change_options", change_options, options_value);
}

pub fn signProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceTrendOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputOptions(self, "sign_profile", name, output_prefix, options_value);
}

pub fn rollingSignProfile(self: anytype, name: []const u8, output_prefix: []const u8, sign_options: DeviceTrendOptions, options_value: DeviceRollingOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputExtraOptions(self, "rolling_sign_profile", name, output_prefix, "sign_options", sign_options, options_value);
}

pub fn expandingSignProfile(self: anytype, name: []const u8, output_prefix: []const u8, sign_options: DeviceTrendOptions, options_value: DeviceExpandingOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputExtraOptions(self, "expanding_sign_profile", name, output_prefix, "sign_options", sign_options, options_value);
}

pub fn crossoverProfile(
    self: anytype,
    lhs_name: []const u8,
    rhs_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceCrossoverOptions,
) DeviceDataError!void {
    return lazy_profile_mod.appendPairOutputOptions(self, "crossover_profile", "lhs_name", lhs_name, "rhs_name", rhs_name, output_prefix, options_value);
}

pub fn rollingCrossoverProfile(
    self: anytype,
    lhs_name: []const u8,
    rhs_name: []const u8,
    output_prefix: []const u8,
    cross_options: DeviceCrossoverOptions,
    options_value: DeviceRollingOptions,
) DeviceDataError!void {
    return lazy_profile_mod.appendPairOutputExtraOptions(self, "rolling_crossover_profile", "lhs_name", lhs_name, "rhs_name", rhs_name, output_prefix, "cross_options", cross_options, options_value);
}

pub fn expandingCrossoverProfile(
    self: anytype,
    lhs_name: []const u8,
    rhs_name: []const u8,
    output_prefix: []const u8,
    cross_options: DeviceCrossoverOptions,
    options_value: DeviceExpandingOptions,
) DeviceDataError!void {
    return lazy_profile_mod.appendPairOutputExtraOptions(self, "expanding_crossover_profile", "lhs_name", lhs_name, "rhs_name", rhs_name, output_prefix, "cross_options", cross_options, options_value);
}

pub fn bucketProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceBucketOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputOptions(self, "bucket_profile", name, output_prefix, options_value);
}

pub fn emaProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceEmaOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputOptions(self, "ema_profile", name, output_prefix, options_value);
}

pub fn linearFitProfile(
    self: anytype,
    x_name: []const u8,
    y_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceLinearFitOptions,
) DeviceDataError!void {
    return lazy_profile_mod.appendPairOutputOptions(self, "linear_fit_profile", "x_name", x_name, "y_name", y_name, output_prefix, options_value);
}

pub fn errorProfile(
    self: anytype,
    actual_name: []const u8,
    predicted_name: []const u8,
    output_prefix: []const u8,
) DeviceDataError!void {
    return lazy_profile_mod.appendPairOutput(self, "error_profile", "actual_name", actual_name, "predicted_name", predicted_name, output_prefix);
}

pub fn rollingErrorProfile(
    self: anytype,
    actual_name: []const u8,
    predicted_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceRollingOptions,
) DeviceDataError!void {
    return lazy_profile_mod.appendPairOutputOptions(self, "rolling_error_profile", "actual_name", actual_name, "predicted_name", predicted_name, output_prefix, options_value);
}

pub fn expandingErrorProfile(
    self: anytype,
    actual_name: []const u8,
    predicted_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceExpandingOptions,
) DeviceDataError!void {
    return lazy_profile_mod.appendPairOutputOptions(self, "expanding_error_profile", "actual_name", actual_name, "predicted_name", predicted_name, output_prefix, options_value);
}

pub fn classificationProfile(
    self: anytype,
    actual_name: []const u8,
    predicted_name: []const u8,
    output_prefix: []const u8,
) DeviceDataError!void {
    return lazy_profile_mod.appendPairOutput(self, "classification_profile", "actual_name", actual_name, "predicted_name", predicted_name, output_prefix);
}

pub fn rollingClassificationProfile(
    self: anytype,
    actual_name: []const u8,
    predicted_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceRollingOptions,
) DeviceDataError!void {
    return lazy_profile_mod.appendPairOutputOptions(self, "rolling_classification_profile", "actual_name", actual_name, "predicted_name", predicted_name, output_prefix, options_value);
}

pub fn expandingClassificationProfile(
    self: anytype,
    actual_name: []const u8,
    predicted_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceExpandingOptions,
) DeviceDataError!void {
    return lazy_profile_mod.appendPairOutputOptions(self, "expanding_classification_profile", "actual_name", actual_name, "predicted_name", predicted_name, output_prefix, options_value);
}

pub fn boolTransitionProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceTrendOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputOptions(self, "bool_transition_profile", name, output_prefix, options_value);
}

pub fn rollingBoolTransitionProfile(self: anytype, name: []const u8, output_prefix: []const u8, transition_options: DeviceTrendOptions, options_value: DeviceRollingOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputExtraOptions(self, "rolling_bool_transition_profile", name, output_prefix, "transition_options", transition_options, options_value);
}

pub fn expandingBoolTransitionProfile(self: anytype, name: []const u8, output_prefix: []const u8, transition_options: DeviceTrendOptions, options_value: DeviceExpandingOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputExtraOptions(self, "expanding_bool_transition_profile", name, output_prefix, "transition_options", transition_options, options_value);
}

pub fn rollingCorrelationProfile(
    self: anytype,
    x_name: []const u8,
    y_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceRollingCorrelationOptions,
) DeviceDataError!void {
    return lazy_profile_mod.appendPairOutputOptions(self, "rolling_correlation_profile", "x_name", x_name, "y_name", y_name, output_prefix, options_value);
}

pub fn expandingCorrelationProfile(
    self: anytype,
    x_name: []const u8,
    y_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceExpandingOptions,
) DeviceDataError!void {
    return lazy_profile_mod.appendPairOutputOptions(self, "expanding_correlation_profile", "x_name", x_name, "y_name", y_name, output_prefix, options_value);
}

pub fn expandingLinearFitProfile(
    self: anytype,
    x_name: []const u8,
    y_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceExpandingOptions,
) DeviceDataError!void {
    return lazy_profile_mod.appendPairOutputOptions(self, "expanding_linear_fit_profile", "x_name", x_name, "y_name", y_name, output_prefix, options_value);
}

pub fn rollingLinearFitProfile(
    self: anytype,
    x_name: []const u8,
    y_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceRollingCorrelationOptions,
) DeviceDataError!void {
    return lazy_profile_mod.appendPairOutputOptions(self, "rolling_linear_fit_profile", "x_name", x_name, "y_name", y_name, output_prefix, options_value);
}

pub fn validityProfile(self: anytype, name: []const u8, output_prefix: []const u8) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutput(self, "validity_profile", name, output_prefix);
}

pub fn rollingValidityProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputOptions(self, "rolling_validity_profile", name, output_prefix, options_value);
}

pub fn expandingValidityProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceExpandingOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputOptions(self, "expanding_validity_profile", name, output_prefix, options_value);
}
