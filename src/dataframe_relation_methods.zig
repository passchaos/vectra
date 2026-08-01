//! Eager DeviceDataFrame group-by and join method wrappers.
//!
//! These wrappers dispatch to grouped aggregation and join implementation modules.
//! They live outside `dataframe.zig` so the public facade stays compact while
//! preserving method-call syntax through aliases on `DeviceDataFrame`.

const std = @import("std");
const array_mod = @import("array.zig");
const group_profile_mod = @import("dataframe_group_profile.zig");
const group_multi_mod = @import("dataframe_group_multi.zig");
const join_mod = @import("dataframe_join.zig");
const options_mod = @import("dataframe_options.zig");
const series_mod = @import("series.zig");

const DeviceDataError = series_mod.DataError || array_mod.ArrayError;
const DeviceJoinOptions = options_mod.DeviceJoinOptions;
const DeviceAsofOptions = options_mod.DeviceAsofOptions;

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

pub fn groupByCount(self: anytype, key_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_profile_mod.groupByCount(FrameType(@TypeOf(self)), frameValue(self), key_name, output_name);
}

pub fn groupByCountOn(self: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, output_name);
}

pub fn valueCounts(self: anytype, key_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return valueCountsAs(self, key_name, "count");
}

pub fn valueCountsAs(self: anytype, key_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return groupByCount(self, key_name, output_name);
}

pub fn valueCountsOn(self: anytype, key_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return valueCountsOnAs(self, key_names, "count");
}

pub fn valueCountsOnAs(self: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return groupByCountOn(self, key_names, output_name);
}

pub fn valueCountsSorted(self: anytype, key_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return valueCountsSortedAs(self, key_name, "count");
}

pub fn valueCountsSortedAs(self: anytype, key_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var counts = try valueCountsAs(self, key_name, output_name);
    defer counts.deinit();
    return counts.sortBy(output_name, .{ .descending = true });
}

pub fn valueCountsOnSorted(self: anytype, key_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return valueCountsOnSortedAs(self, key_names, "count");
}

pub fn valueCountsOnSortedAs(self: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var counts = try valueCountsOnAs(self, key_names, output_name);
    defer counts.deinit();
    return counts.sortBy(output_name, .{ .descending = true });
}

pub fn valueCountsSortedOn(self: anytype, key_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return valueCountsOnSorted(self, key_names);
}

pub fn valueCountsSortedOnAs(self: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return valueCountsOnSortedAs(self, key_names, output_name);
}

pub fn groupBySum(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_profile_mod.groupByNumeric(FrameType(@TypeOf(self)), .sum, frameValue(self), key_name, value_name, output_name);
}

pub fn groupBySumOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByNumericOn(FrameType(@TypeOf(self)), .sum, frameValue(self), key_names, value_name, output_name);
}

pub fn groupByMin(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_profile_mod.groupByNumeric(FrameType(@TypeOf(self)), .min, frameValue(self), key_name, value_name, output_name);
}

pub fn groupByMinOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByNumericOn(FrameType(@TypeOf(self)), .min, frameValue(self), key_names, value_name, output_name);
}

pub fn groupByMax(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_profile_mod.groupByNumeric(FrameType(@TypeOf(self)), .max, frameValue(self), key_name, value_name, output_name);
}

pub fn groupByMaxOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByNumericOn(FrameType(@TypeOf(self)), .max, frameValue(self), key_names, value_name, output_name);
}

pub fn groupByMean(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_profile_mod.groupByMean(FrameType(@TypeOf(self)), frameValue(self), key_name, value_name, output_name);
}

pub fn groupByMeanOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByMeanOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByFirst(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByFirstOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByFirstOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByFirstOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByLast(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByLastOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByLastOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByLastOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByStats(self: anytype, key_name: []const u8, value_name: []const u8, output_prefix: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_profile_mod.groupByStats(FrameType(@TypeOf(self)), frameValue(self), key_name, value_name, output_prefix);
}

pub fn groupByStatsOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_prefix: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByStatsOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_prefix);
}

pub fn groupByProfile(self: anytype, key_name: []const u8, value_name: []const u8, output_prefix: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_profile_mod.groupByProfile(FrameType(@TypeOf(self)), frameValue(self), key_name, value_name, output_prefix);
}

pub fn groupByProfileOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_prefix: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByProfileOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_prefix);
}

pub fn innerJoin(
    self: anytype,
    right: FrameType(@TypeOf(self)),
    left_key_name: []const u8,
    right_key_name: []const u8,
    options_value: DeviceJoinOptions,
) DeviceDataError!FrameType(@TypeOf(self)) {
    return join_mod.innerJoin(FrameType(@TypeOf(self)), frameValue(self), right, left_key_name, right_key_name, options_value);
}

pub fn innerJoinOn(
    self: anytype,
    right: FrameType(@TypeOf(self)),
    left_key_names: []const []const u8,
    right_key_names: []const []const u8,
    options_value: DeviceJoinOptions,
) DeviceDataError!FrameType(@TypeOf(self)) {
    return join_mod.innerJoinOn(FrameType(@TypeOf(self)), frameValue(self), right, left_key_names, right_key_names, options_value);
}

pub fn leftJoin(
    self: anytype,
    right: FrameType(@TypeOf(self)),
    left_key_name: []const u8,
    right_key_name: []const u8,
    options_value: DeviceJoinOptions,
) DeviceDataError!FrameType(@TypeOf(self)) {
    return join_mod.leftJoin(FrameType(@TypeOf(self)), frameValue(self), right, left_key_name, right_key_name, options_value);
}

pub fn leftJoinOn(
    self: anytype,
    right: FrameType(@TypeOf(self)),
    left_key_names: []const []const u8,
    right_key_names: []const []const u8,
    options_value: DeviceJoinOptions,
) DeviceDataError!FrameType(@TypeOf(self)) {
    return join_mod.leftJoinOn(FrameType(@TypeOf(self)), frameValue(self), right, left_key_names, right_key_names, options_value);
}

pub fn fullJoin(
    self: anytype,
    right: FrameType(@TypeOf(self)),
    left_key_name: []const u8,
    right_key_name: []const u8,
    options_value: DeviceJoinOptions,
) DeviceDataError!FrameType(@TypeOf(self)) {
    return join_mod.fullJoin(FrameType(@TypeOf(self)), frameValue(self), right, left_key_name, right_key_name, options_value);
}

pub fn fullJoinOn(
    self: anytype,
    right: FrameType(@TypeOf(self)),
    left_key_names: []const []const u8,
    right_key_names: []const []const u8,
    options_value: DeviceJoinOptions,
) DeviceDataError!FrameType(@TypeOf(self)) {
    return join_mod.fullJoinOn(FrameType(@TypeOf(self)), frameValue(self), right, left_key_names, right_key_names, options_value);
}

pub fn semiJoin(
    self: anytype,
    right: FrameType(@TypeOf(self)),
    left_key_name: []const u8,
    right_key_name: []const u8,
) DeviceDataError!FrameType(@TypeOf(self)) {
    return join_mod.semiJoin(FrameType(@TypeOf(self)), frameValue(self), right, left_key_name, right_key_name);
}

pub fn semiJoinOn(
    self: anytype,
    right: FrameType(@TypeOf(self)),
    left_key_names: []const []const u8,
    right_key_names: []const []const u8,
) DeviceDataError!FrameType(@TypeOf(self)) {
    return join_mod.semiJoinOn(FrameType(@TypeOf(self)), frameValue(self), right, left_key_names, right_key_names);
}

pub fn antiJoin(
    self: anytype,
    right: FrameType(@TypeOf(self)),
    left_key_name: []const u8,
    right_key_name: []const u8,
) DeviceDataError!FrameType(@TypeOf(self)) {
    return join_mod.antiJoin(FrameType(@TypeOf(self)), frameValue(self), right, left_key_name, right_key_name);
}

pub fn antiJoinOn(
    self: anytype,
    right: FrameType(@TypeOf(self)),
    left_key_names: []const []const u8,
    right_key_names: []const []const u8,
) DeviceDataError!FrameType(@TypeOf(self)) {
    return join_mod.antiJoinOn(FrameType(@TypeOf(self)), frameValue(self), right, left_key_names, right_key_names);
}

pub fn asofJoin(
    self: anytype,
    right: FrameType(@TypeOf(self)),
    left_key_name: []const u8,
    right_key_name: []const u8,
    options_value: DeviceAsofOptions,
) DeviceDataError!FrameType(@TypeOf(self)) {
    return join_mod.asofJoin(FrameType(@TypeOf(self)), frameValue(self), right, left_key_name, right_key_name, options_value);
}
