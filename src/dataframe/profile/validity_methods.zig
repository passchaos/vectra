//! Eager validity profile method wrappers.

const std = @import("std");
const array_mod = @import("../../array.zig");
const validity_mod = @import("../validity.zig");
const options_mod = @import("../../dataframe_options.zig");
const series_mod = @import("../../series.zig");

const DeviceDataError = series_mod.DataError || array_mod.ArrayError;
const DeviceRollingOptions = options_mod.DeviceRollingOptions;
const DeviceExpandingOptions = options_mod.DeviceExpandingOptions;

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

pub fn validityProfile(self: anytype, name: []const u8, output_prefix: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return validity_mod.validityProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix);
}

pub fn rollingValidityProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return validity_mod.rollingValidityProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, options_value);
}

pub fn expandingValidityProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceExpandingOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return validity_mod.expandingValidityProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, options_value);
}
