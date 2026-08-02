//! Lazy group-by operation builders for DeviceLazyFrame.

const std = @import("std");
const array_mod = @import("array.zig");
const lazy_op_mod = @import("dataframe_lazy_op.zig");
const names_mod = @import("dataframe_names.zig");
const series_mod = @import("series.zig");

const DeviceLazyGroupByAggregation = lazy_op_mod.DeviceLazyGroupByAggregation;
const DeviceLazyWeightedGroupByAggregation = lazy_op_mod.DeviceLazyWeightedGroupByAggregation;
const DeviceLazyPairGroupByAggregation = lazy_op_mod.DeviceLazyPairGroupByAggregation;
const DeviceLazyWeightedPairGroupByAggregation = lazy_op_mod.DeviceLazyWeightedPairGroupByAggregation;
const DeviceDataError = series_mod.DataError || array_mod.ArrayError;
const cloneNameList = names_mod.cloneNameList;
const freeNameList = names_mod.freeNameList;

pub fn groupByCount(frame: anytype, key_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_key = try frame.allocator.dupe(u8, key_name);
    errdefer frame.allocator.free(owned_key);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .group_by_count = .{
        .key_name = owned_key,
        .output_name = owned_output,
    } });
}

pub fn groupByCountOn(frame: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .group_by_count_on = .{
        .key_names = owned_keys,
        .output_name = owned_output,
    } });
}

pub fn groupByValue(frame: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8, aggregation: DeviceLazyGroupByAggregation) DeviceDataError!void {
    return groupByValueQuantile(frame, key_name, value_name, output_name, aggregation, defaultAggregationParameter(aggregation));
}

fn defaultAggregationParameter(aggregation: DeviceLazyGroupByAggregation) f64 {
    return switch (aggregation) {
        .trimmed_mean, .winsorized_mean => 0.0,
        else => 0.5,
    };
}

pub fn groupByValueQuantile(frame: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8, aggregation: DeviceLazyGroupByAggregation, quantile: f64) DeviceDataError!void {
    const owned_key = try frame.allocator.dupe(u8, key_name);
    errdefer frame.allocator.free(owned_key);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .group_by_value = .{
        .key_name = owned_key,
        .value_name = owned_value,
        .output_name = owned_output,
        .aggregation = aggregation,
        .quantile = quantile,
    } });
}

pub fn groupByValueOn(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, aggregation: DeviceLazyGroupByAggregation) DeviceDataError!void {
    return groupByValueOnQuantile(frame, key_names, value_name, output_name, aggregation, defaultAggregationParameter(aggregation));
}

pub fn groupByValueOnQuantile(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, aggregation: DeviceLazyGroupByAggregation, quantile: f64) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .group_by_value_on = .{
        .key_names = owned_keys,
        .value_name = owned_value,
        .output_name = owned_output,
        .aggregation = aggregation,
        .quantile = quantile,
    } });
}

pub fn groupByWeighted(frame: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8, aggregation: DeviceLazyWeightedGroupByAggregation) DeviceDataError!void {
    return groupByWeightedQuantile(frame, key_name, value_name, weight_name, output_name, aggregation, 0.5);
}

pub fn groupByWeightedQuantile(frame: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8, aggregation: DeviceLazyWeightedGroupByAggregation, quantile: f64) DeviceDataError!void {
    const owned_key = try frame.allocator.dupe(u8, key_name);
    errdefer frame.allocator.free(owned_key);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_weight = try frame.allocator.dupe(u8, weight_name);
    errdefer frame.allocator.free(owned_weight);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .group_by_weighted = .{
        .key_name = owned_key,
        .value_name = owned_value,
        .weight_name = owned_weight,
        .output_name = owned_output,
        .aggregation = aggregation,
        .quantile = quantile,
    } });
}

pub fn groupByWeightedOn(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8, aggregation: DeviceLazyWeightedGroupByAggregation) DeviceDataError!void {
    return groupByWeightedOnQuantile(frame, key_names, value_name, weight_name, output_name, aggregation, 0.5);
}

pub fn groupByWeightedOnQuantile(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8, aggregation: DeviceLazyWeightedGroupByAggregation, quantile: f64) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_weight = try frame.allocator.dupe(u8, weight_name);
    errdefer frame.allocator.free(owned_weight);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .group_by_weighted_on = .{
        .key_names = owned_keys,
        .value_name = owned_value,
        .weight_name = owned_weight,
        .output_name = owned_output,
        .aggregation = aggregation,
        .quantile = quantile,
    } });
}

pub fn groupByPair(frame: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8, aggregation: DeviceLazyPairGroupByAggregation) DeviceDataError!void {
    const owned_key = try frame.allocator.dupe(u8, key_name);
    errdefer frame.allocator.free(owned_key);
    const owned_lhs = try frame.allocator.dupe(u8, lhs_name);
    errdefer frame.allocator.free(owned_lhs);
    const owned_rhs = try frame.allocator.dupe(u8, rhs_name);
    errdefer frame.allocator.free(owned_rhs);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .group_by_pair = .{
        .key_name = owned_key,
        .lhs_name = owned_lhs,
        .rhs_name = owned_rhs,
        .output_name = owned_output,
        .aggregation = aggregation,
    } });
}

pub fn groupByPairOn(frame: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8, aggregation: DeviceLazyPairGroupByAggregation) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_lhs = try frame.allocator.dupe(u8, lhs_name);
    errdefer frame.allocator.free(owned_lhs);
    const owned_rhs = try frame.allocator.dupe(u8, rhs_name);
    errdefer frame.allocator.free(owned_rhs);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .group_by_pair_on = .{
        .key_names = owned_keys,
        .lhs_name = owned_lhs,
        .rhs_name = owned_rhs,
        .output_name = owned_output,
        .aggregation = aggregation,
    } });
}

pub fn groupByWeightedPair(frame: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8, aggregation: DeviceLazyWeightedPairGroupByAggregation, correction: f64) DeviceDataError!void {
    const owned_key = try frame.allocator.dupe(u8, key_name);
    errdefer frame.allocator.free(owned_key);
    const owned_lhs = try frame.allocator.dupe(u8, lhs_name);
    errdefer frame.allocator.free(owned_lhs);
    const owned_rhs = try frame.allocator.dupe(u8, rhs_name);
    errdefer frame.allocator.free(owned_rhs);
    const owned_weight = try frame.allocator.dupe(u8, weight_name);
    errdefer frame.allocator.free(owned_weight);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .group_by_weighted_pair = .{
        .key_name = owned_key,
        .lhs_name = owned_lhs,
        .rhs_name = owned_rhs,
        .weight_name = owned_weight,
        .output_name = owned_output,
        .aggregation = aggregation,
        .correction = correction,
    } });
}

pub fn groupByWeightedPairOn(frame: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8, aggregation: DeviceLazyWeightedPairGroupByAggregation, correction: f64) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_lhs = try frame.allocator.dupe(u8, lhs_name);
    errdefer frame.allocator.free(owned_lhs);
    const owned_rhs = try frame.allocator.dupe(u8, rhs_name);
    errdefer frame.allocator.free(owned_rhs);
    const owned_weight = try frame.allocator.dupe(u8, weight_name);
    errdefer frame.allocator.free(owned_weight);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .group_by_weighted_pair_on = .{
        .key_names = owned_keys,
        .lhs_name = owned_lhs,
        .rhs_name = owned_rhs,
        .weight_name = owned_weight,
        .output_name = owned_output,
        .aggregation = aggregation,
        .correction = correction,
    } });
}

pub fn groupByStats(frame: anytype, key_name: []const u8, value_name: []const u8, output_prefix: []const u8) DeviceDataError!void {
    const owned_key = try frame.allocator.dupe(u8, key_name);
    errdefer frame.allocator.free(owned_key);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_prefix = try frame.allocator.dupe(u8, output_prefix);
    errdefer frame.allocator.free(owned_prefix);
    try frame.ops.append(frame.allocator, .{ .group_by_stats = .{
        .key_name = owned_key,
        .value_name = owned_value,
        .output_prefix = owned_prefix,
    } });
}

pub fn groupByStatsOn(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_prefix: []const u8) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_prefix = try frame.allocator.dupe(u8, output_prefix);
    errdefer frame.allocator.free(owned_prefix);
    try frame.ops.append(frame.allocator, .{ .group_by_stats_on = .{
        .key_names = owned_keys,
        .value_name = owned_value,
        .output_prefix = owned_prefix,
    } });
}

pub fn groupByProfile(frame: anytype, key_name: []const u8, value_name: []const u8, output_prefix: []const u8) DeviceDataError!void {
    const owned_key = try frame.allocator.dupe(u8, key_name);
    errdefer frame.allocator.free(owned_key);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_prefix = try frame.allocator.dupe(u8, output_prefix);
    errdefer frame.allocator.free(owned_prefix);
    try frame.ops.append(frame.allocator, .{ .group_by_profile = .{
        .key_name = owned_key,
        .value_name = owned_value,
        .output_prefix = owned_prefix,
    } });
}

pub fn groupByProfileOn(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_prefix: []const u8) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_prefix = try frame.allocator.dupe(u8, output_prefix);
    errdefer frame.allocator.free(owned_prefix);
    try frame.ops.append(frame.allocator, .{ .group_by_profile_on = .{
        .key_names = owned_keys,
        .value_name = owned_value,
        .output_prefix = owned_prefix,
    } });
}
