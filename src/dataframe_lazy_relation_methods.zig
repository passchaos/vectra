//! Lazy group-by, join, concat, and distinct method wrappers.

const std = @import("std");
const array_mod = @import("array.zig");
const lazy_group_mod = @import("dataframe_lazy_group_plan.zig");
const lazy_join_mod = @import("dataframe_lazy_join_plan.zig");
const lazy_op_mod = @import("dataframe_lazy_op.zig");
const options_mod = @import("dataframe_options.zig");
const series_mod = @import("series.zig");

const DeviceLazyGroupByAggregation = lazy_op_mod.DeviceLazyGroupByAggregation;
const DeviceLazyJoinKind = lazy_op_mod.DeviceLazyJoinKind;
const DeviceJoinOptions = options_mod.DeviceJoinOptions;
const DeviceAsofOptions = options_mod.DeviceAsofOptions;
const DeviceDataError = series_mod.DataError || array_mod.ArrayError;

pub fn groupByCount(self: anytype, key_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByCount(self, key_name, output_name);
}

pub fn groupByValue(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8, aggregation: DeviceLazyGroupByAggregation) DeviceDataError!void {
    return lazy_group_mod.groupByValue(self, key_name, value_name, output_name, aggregation);
}

pub fn groupBySum(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .sum);
}

pub fn groupByMin(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .min);
}

pub fn groupByMax(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .max);
}

pub fn groupByMean(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .mean);
}

pub fn groupByStats(self: anytype, key_name: []const u8, value_name: []const u8, output_prefix: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByStats(self, key_name, value_name, output_prefix);
}

pub fn groupByStatsOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_prefix: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByStatsOn(self, key_names, value_name, output_prefix);
}

pub fn groupByProfile(self: anytype, key_name: []const u8, value_name: []const u8, output_prefix: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByProfile(self, key_name, value_name, output_prefix);
}

pub fn groupByProfileOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_prefix: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByProfileOn(self, key_names, value_name, output_prefix);
}

pub fn joinOn(
    self: anytype,
    right: anytype,
    left_key_names: []const []const u8,
    right_key_names: []const []const u8,
    kind: DeviceLazyJoinKind,
    options_value: DeviceJoinOptions,
) DeviceDataError!void {
    return lazy_join_mod.joinOn(self, right, left_key_names, right_key_names, kind, options_value);
}

pub fn innerJoinOn(self: anytype, right: anytype, left_key_names: []const []const u8, right_key_names: []const []const u8, options_value: DeviceJoinOptions) DeviceDataError!void {
    return self.joinOn(right, left_key_names, right_key_names, .inner, options_value);
}

pub fn leftJoinOn(self: anytype, right: anytype, left_key_names: []const []const u8, right_key_names: []const []const u8, options_value: DeviceJoinOptions) DeviceDataError!void {
    return self.joinOn(right, left_key_names, right_key_names, .left, options_value);
}

pub fn fullJoinOn(self: anytype, right: anytype, left_key_names: []const []const u8, right_key_names: []const []const u8, options_value: DeviceJoinOptions) DeviceDataError!void {
    return self.joinOn(right, left_key_names, right_key_names, .full, options_value);
}

pub fn semiJoinOn(self: anytype, right: anytype, left_key_names: []const []const u8, right_key_names: []const []const u8) DeviceDataError!void {
    return self.joinOn(right, left_key_names, right_key_names, .semi, .{});
}

pub fn antiJoinOn(self: anytype, right: anytype, left_key_names: []const []const u8, right_key_names: []const []const u8) DeviceDataError!void {
    return self.joinOn(right, left_key_names, right_key_names, .anti, .{});
}

pub fn asofJoin(
    self: anytype,
    right: anytype,
    left_key_name: []const u8,
    right_key_name: []const u8,
    options_value: DeviceAsofOptions,
) DeviceDataError!void {
    return lazy_join_mod.asofJoin(self, right, left_key_name, right_key_name, options_value);
}

pub fn concatRows(self: anytype, right: anytype) DeviceDataError!void {
    return lazy_join_mod.concatRows(self, right);
}

pub fn appendRows(self: anytype, right: anytype) DeviceDataError!void {
    return self.concatRows(right);
}

pub fn vstack(self: anytype, right: anytype) DeviceDataError!void {
    return self.concatRows(right);
}

pub fn distinctRows(self: anytype) DeviceDataError!void {
    try self.ops.append(self.allocator, .{ .distinct_rows = {} });
}

pub fn distinctRowsLast(self: anytype) DeviceDataError!void {
    try self.ops.append(self.allocator, .{ .distinct_rows_last = {} });
}

pub fn distinctOn(self: anytype, key_names: []const []const u8) DeviceDataError!void {
    return lazy_join_mod.distinctOn(self, key_names);
}

pub fn distinctOnLast(self: anytype, key_names: []const []const u8) DeviceDataError!void {
    return lazy_join_mod.distinctOnLast(self, key_names);
}

pub fn dropDuplicates(self: anytype) DeviceDataError!void {
    return self.distinctRows();
}

pub fn dropDuplicatesOn(self: anytype, key_names: []const []const u8) DeviceDataError!void {
    return self.distinctOn(key_names);
}

pub fn dropDuplicatesLast(self: anytype) DeviceDataError!void {
    return self.distinctRowsLast();
}

pub fn dropDuplicatesOnLast(self: anytype, key_names: []const []const u8) DeviceDataError!void {
    return self.distinctOnLast(key_names);
}

pub fn uniqueRows(self: anytype) DeviceDataError!void {
    return self.distinctRows();
}
