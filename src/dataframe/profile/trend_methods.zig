//! Eager trend/change/sign profile method wrappers.

const std = @import("std");
const array_mod = @import("../../array.zig");
const trend_mod = @import("../../dataframe_trend.zig");
const change_mod = @import("../../dataframe_change.zig");
const sign_mod = @import("../../dataframe_sign.zig");
const options_mod = @import("../../dataframe_options.zig");
const series_mod = @import("../../series.zig");

const DeviceDataError = series_mod.DataError || array_mod.ArrayError;
const DeviceRollingOptions = options_mod.DeviceRollingOptions;
const DeviceExpandingOptions = options_mod.DeviceExpandingOptions;
const DeviceTrendOptions = options_mod.DeviceTrendOptions;

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

pub fn trendProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceTrendOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return trend_mod.trendProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, options_value);
}

pub fn changePointProfile(self: anytype, name: []const u8, output_prefix: []const u8, threshold: f64, options_value: DeviceTrendOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return change_mod.changePointProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, threshold, options_value);
}

pub fn rollingChangePointProfile(self: anytype, name: []const u8, output_prefix: []const u8, threshold: f64, change_options: DeviceTrendOptions, options_value: DeviceRollingOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return change_mod.rollingChangePointProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, threshold, change_options, options_value);
}

pub fn expandingChangePointProfile(self: anytype, name: []const u8, output_prefix: []const u8, threshold: f64, change_options: DeviceTrendOptions, options_value: DeviceExpandingOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return change_mod.expandingChangePointProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, threshold, change_options, options_value);
}

pub fn rollingTrendProfile(self: anytype, name: []const u8, output_prefix: []const u8, trend_options: DeviceTrendOptions, options_value: DeviceRollingOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return trend_mod.rollingTrendProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, trend_options, options_value);
}

pub fn expandingTrendProfile(self: anytype, name: []const u8, output_prefix: []const u8, trend_options: DeviceTrendOptions, options_value: DeviceExpandingOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return trend_mod.expandingTrendProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, trend_options, options_value);
}

pub fn signProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceTrendOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return sign_mod.signProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, options_value);
}

pub fn rollingSignProfile(self: anytype, name: []const u8, output_prefix: []const u8, sign_options: DeviceTrendOptions, options_value: DeviceRollingOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return sign_mod.rollingSignProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, sign_options, options_value);
}

pub fn expandingSignProfile(self: anytype, name: []const u8, output_prefix: []const u8, sign_options: DeviceTrendOptions, options_value: DeviceExpandingOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return sign_mod.expandingSignProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, sign_options, options_value);
}
