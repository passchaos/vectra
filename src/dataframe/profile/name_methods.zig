//! Eager name-output/trend profile method wrappers.

const std = @import("std");
const array_mod = @import("../../array.zig");
const threshold_mod = @import("../../dataframe_threshold.zig");
const bool_profile_mod = @import("../bool_profile.zig");
const clip_mod = @import("../../dataframe_clip.zig");
const risk_mod = @import("../risk.zig");
const standardize_mod = @import("../../dataframe_standardize.zig");
const robust_mod = @import("../../dataframe_robust.zig");
const trend_methods_mod = @import("trend_methods.zig");
const shift_mod = @import("../../dataframe_shift.zig");
const quantile_mod = @import("../quantile.zig");
const rank_mod = @import("../../dataframe_rank.zig");
const stats_profile_mod = @import("../../dataframe_stats_profile.zig");
const moment_mod = @import("../../dataframe_moment.zig");
const normalize_mod = @import("../../dataframe_normalize.zig");
const range_mod = @import("../../dataframe_range.zig");
const options_mod = @import("../../dataframe_options.zig");
const series_mod = @import("../../series.zig");

const DeviceDataError = series_mod.DataError || array_mod.ArrayError;
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

fn FrameType(comptime Frame: type) type {
    return switch (@typeInfo(Frame)) {
        .pointer => |ptr| ptr.child,
        else => Frame,
    };
}

fn frameValue(self: anytype) FrameType(@TypeOf(self)) {
    return switch (@typeInfo(@TypeOf(self))) {
        .pointer => self.*,
        else => self,
    };
}

pub fn rollingProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return stats_profile_mod.rollingProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, options_value);
}

pub fn rollingMomentProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return moment_mod.rollingMomentProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, options_value);
}

pub fn rollingRangeProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return range_mod.rollingRangeProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, options_value);
}

pub fn rollingNormalizeProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return normalize_mod.rollingNormalizeProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, options_value);
}

pub fn expandingNormalizeProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceExpandingOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return normalize_mod.expandingNormalizeProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, options_value);
}

pub fn rollingQuantileProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return quantile_mod.rollingQuantileProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, options_value);
}

pub fn expandingQuantileProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceExpandingOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return quantile_mod.expandingQuantileProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, options_value);
}

pub fn rollingBoolProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return bool_profile_mod.rollingBoolProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, options_value);
}

pub fn rollingDrawdownProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return risk_mod.rollingDrawdownProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, options_value);
}

pub fn rollingRobustProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingRobustOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return robust_mod.rollingRobustProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, options_value);
}

pub fn rollingRankProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingRankOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return rank_mod.rollingRankProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, options_value);
}

pub fn lagProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceLagOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return shift_mod.lagProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, options_value);
}

pub fn leadProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceLagOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return shift_mod.leadProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, options_value);
}

pub fn clipProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceClipOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return clip_mod.clipProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, options_value);
}

pub fn rollingClipProfile(self: anytype, name: []const u8, output_prefix: []const u8, clip_options: DeviceClipOptions, options_value: DeviceRollingOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return clip_mod.rollingClipProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, clip_options, options_value);
}

pub fn expandingClipProfile(self: anytype, name: []const u8, output_prefix: []const u8, clip_options: DeviceClipOptions, options_value: DeviceExpandingOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return clip_mod.expandingClipProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, clip_options, options_value);
}

pub fn thresholdProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceThresholdOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return threshold_mod.thresholdProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, options_value);
}

pub fn rollingThresholdProfile(self: anytype, name: []const u8, output_prefix: []const u8, threshold: f64, options_value: DeviceRollingOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return threshold_mod.rollingThresholdProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, threshold, options_value);
}

pub fn expandingThresholdProfile(self: anytype, name: []const u8, output_prefix: []const u8, threshold: f64, options_value: DeviceExpandingOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return threshold_mod.expandingThresholdProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, threshold, options_value);
}

pub fn expandingProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceExpandingOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return stats_profile_mod.expandingProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, options_value);
}

pub fn expandingBoolProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceExpandingOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return bool_profile_mod.expandingBoolProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, options_value);
}

pub fn expandingRankProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceExpandingRankOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return rank_mod.expandingRankProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, options_value);
}

pub fn expandingRobustProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceRobustOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return robust_mod.expandingRobustProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, options_value);
}

pub fn expandingMomentProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceExpandingOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return moment_mod.expandingMomentProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, options_value);
}

pub fn standardizeProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceStandardizeOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return standardize_mod.standardizeProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, options_value);
}

pub fn robustProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceRobustOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return robust_mod.robustProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, options_value);
}

pub fn drawdownProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceDrawdownOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return risk_mod.drawdownProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, options_value);
}

pub fn extremaProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceExtremaOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return risk_mod.extremaProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, options_value);
}

pub const trendProfile = trend_methods_mod.trendProfile;
pub const changePointProfile = trend_methods_mod.changePointProfile;
pub const rollingChangePointProfile = trend_methods_mod.rollingChangePointProfile;
pub const expandingChangePointProfile = trend_methods_mod.expandingChangePointProfile;
pub const rollingTrendProfile = trend_methods_mod.rollingTrendProfile;
pub const expandingTrendProfile = trend_methods_mod.expandingTrendProfile;
pub const signProfile = trend_methods_mod.signProfile;
pub const rollingSignProfile = trend_methods_mod.rollingSignProfile;
pub const expandingSignProfile = trend_methods_mod.expandingSignProfile;
