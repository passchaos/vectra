//! Lazy trend/change/sign profile method wrappers.

const std = @import("std");
const array_mod = @import("../../array.zig");
const lazy_profile_mod = @import("profile_plan.zig");
const options_mod = @import("../../dataframe_options.zig");
const series_mod = @import("../../series.zig");

const DeviceRollingOptions = options_mod.DeviceRollingOptions;
const DeviceExpandingOptions = options_mod.DeviceExpandingOptions;
const DeviceTrendOptions = options_mod.DeviceTrendOptions;
const DeviceDataError = series_mod.DataError || array_mod.ArrayError;

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
