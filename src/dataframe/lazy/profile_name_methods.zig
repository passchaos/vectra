//! Lazy name-output/trend profile method wrappers.

const std = @import("std");
const array_mod = @import("../../array.zig");
const lazy_profile_mod = @import("profile_plan.zig");
const trend_methods_mod = @import("profile_trend_methods.zig");
const lazy_sort_mod = @import("sort_plan.zig");
const options_mod = @import("../../dataframe_options.zig");
const series_mod = @import("../../series.zig");

const DeviceRollingOptions = options_mod.DeviceRollingOptions;
const DeviceLagOptions = options_mod.DeviceLagOptions;
const DeviceExpandingOptions = options_mod.DeviceExpandingOptions;
const DeviceExpandingRankOptions = options_mod.DeviceExpandingRankOptions;
const DeviceStandardizeOptions = options_mod.DeviceStandardizeOptions;
const DeviceRobustOptions = options_mod.DeviceRobustOptions;
const DeviceDrawdownOptions = options_mod.DeviceDrawdownOptions;
const DeviceExtremaOptions = options_mod.DeviceExtremaOptions;
const DeviceClipOptions = options_mod.DeviceClipOptions;
const DeviceThresholdOptions = options_mod.DeviceThresholdOptions;
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

pub const trendProfile = trend_methods_mod.trendProfile;
pub const rollingTrendProfile = trend_methods_mod.rollingTrendProfile;
pub const expandingTrendProfile = trend_methods_mod.expandingTrendProfile;
pub const changePointProfile = trend_methods_mod.changePointProfile;
pub const rollingChangePointProfile = trend_methods_mod.rollingChangePointProfile;
pub const expandingChangePointProfile = trend_methods_mod.expandingChangePointProfile;
pub const signProfile = trend_methods_mod.signProfile;
pub const rollingSignProfile = trend_methods_mod.rollingSignProfile;
pub const expandingSignProfile = trend_methods_mod.expandingSignProfile;
