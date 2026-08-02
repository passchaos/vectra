//! Lazy group-by operation builders for DeviceLazyFrame.

const std = @import("std");
const array_mod = @import("array.zig");
const lazy_op_mod = @import("dataframe_lazy_op.zig");
const names_mod = @import("dataframe_names.zig");
const options_mod = @import("dataframe_options.zig");
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

pub fn groupByRows(frame: anytype, key_name: []const u8, n: usize, keep_tail: bool) DeviceDataError!void {
    const owned_key = try frame.allocator.dupe(u8, key_name);
    errdefer frame.allocator.free(owned_key);
    try frame.ops.append(frame.allocator, .{ .group_by_rows = .{
        .key_name = owned_key,
        .start = 0,
        .n = n,
        .keep_tail = keep_tail,
    } });
}

pub fn groupByRowsOn(frame: anytype, key_names: []const []const u8, n: usize, keep_tail: bool) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    try frame.ops.append(frame.allocator, .{ .group_by_rows_on = .{
        .key_names = owned_keys,
        .start = 0,
        .n = n,
        .keep_tail = keep_tail,
    } });
}

pub fn groupBySliceRows(frame: anytype, key_name: []const u8, start: usize, length: usize) DeviceDataError!void {
    const owned_key = try frame.allocator.dupe(u8, key_name);
    errdefer frame.allocator.free(owned_key);
    try frame.ops.append(frame.allocator, .{ .group_by_rows = .{
        .key_name = owned_key,
        .start = start,
        .n = length,
        .keep_tail = false,
    } });
}

pub fn groupBySliceRowsOn(frame: anytype, key_names: []const []const u8, start: usize, length: usize) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    try frame.ops.append(frame.allocator, .{ .group_by_rows_on = .{
        .key_names = owned_keys,
        .start = start,
        .n = length,
        .keep_tail = false,
    } });
}

pub fn groupBySortedRows(frame: anytype, key_name: []const u8, sort_name: []const u8, n: usize, options: options_mod.DeviceSortOptions, keep_bottom: bool) DeviceDataError!void {
    const owned_key = try frame.allocator.dupe(u8, key_name);
    errdefer frame.allocator.free(owned_key);
    const owned_sort = try frame.allocator.dupe(u8, sort_name);
    errdefer frame.allocator.free(owned_sort);
    try frame.ops.append(frame.allocator, .{ .group_by_sorted_rows = .{
        .key_name = owned_key,
        .sort_name = owned_sort,
        .n = n,
        .options = options,
        .keep_bottom = keep_bottom,
    } });
}

pub fn groupBySortedRowsOn(frame: anytype, key_names: []const []const u8, sort_name: []const u8, n: usize, options: options_mod.DeviceSortOptions, keep_bottom: bool) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_sort = try frame.allocator.dupe(u8, sort_name);
    errdefer frame.allocator.free(owned_sort);
    try frame.ops.append(frame.allocator, .{ .group_by_sorted_rows_on = .{
        .key_names = owned_keys,
        .sort_name = owned_sort,
        .n = n,
        .options = options,
        .keep_bottom = keep_bottom,
    } });
}

pub fn groupBySortedRowsByColumns(frame: anytype, key_name: []const u8, sort_names: []const []const u8, n: usize, options: []const options_mod.DeviceSortOptions, keep_bottom: bool) DeviceDataError!void {
    const owned_key = try frame.allocator.dupe(u8, key_name);
    errdefer frame.allocator.free(owned_key);
    const owned_sorts = try cloneNameList(frame.allocator, sort_names);
    errdefer freeNameList(frame.allocator, owned_sorts);
    const owned_options = try frame.allocator.dupe(options_mod.DeviceSortOptions, options);
    errdefer frame.allocator.free(owned_options);
    try frame.ops.append(frame.allocator, .{ .group_by_sorted_rows_columns = .{
        .key_name = owned_key,
        .sort_names = owned_sorts,
        .n = n,
        .options = owned_options,
        .keep_bottom = keep_bottom,
    } });
}

pub fn groupBySortedRowsByColumnsOn(frame: anytype, key_names: []const []const u8, sort_names: []const []const u8, n: usize, options: []const options_mod.DeviceSortOptions, keep_bottom: bool) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_sorts = try cloneNameList(frame.allocator, sort_names);
    errdefer freeNameList(frame.allocator, owned_sorts);
    const owned_options = try frame.allocator.dupe(options_mod.DeviceSortOptions, options);
    errdefer frame.allocator.free(owned_options);
    try frame.ops.append(frame.allocator, .{ .group_by_sorted_rows_columns_on = .{
        .key_names = owned_keys,
        .sort_names = owned_sorts,
        .n = n,
        .options = owned_options,
        .keep_bottom = keep_bottom,
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
    return groupByValueQuantileIndex(frame, key_name, value_name, output_name, aggregation, quantile, 0);
}

pub fn groupByValueIndex(frame: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8, aggregation: DeviceLazyGroupByAggregation, index: usize) DeviceDataError!void {
    return groupByValueQuantileIndex(frame, key_name, value_name, output_name, aggregation, defaultAggregationParameter(aggregation), index);
}

fn groupByValueQuantileIndex(frame: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8, aggregation: DeviceLazyGroupByAggregation, quantile: f64, index: usize) DeviceDataError!void {
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
        .index = index,
    } });
}

pub fn groupByValueOn(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, aggregation: DeviceLazyGroupByAggregation) DeviceDataError!void {
    return groupByValueOnQuantile(frame, key_names, value_name, output_name, aggregation, defaultAggregationParameter(aggregation));
}

pub fn groupByValueOnQuantile(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, aggregation: DeviceLazyGroupByAggregation, quantile: f64) DeviceDataError!void {
    return groupByValueOnQuantileIndex(frame, key_names, value_name, output_name, aggregation, quantile, 0);
}

pub fn groupByValueOnIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, aggregation: DeviceLazyGroupByAggregation, index: usize) DeviceDataError!void {
    return groupByValueOnQuantileIndex(frame, key_names, value_name, output_name, aggregation, defaultAggregationParameter(aggregation), index);
}

fn groupByValueOnQuantileIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, aggregation: DeviceLazyGroupByAggregation, quantile: f64, index: usize) DeviceDataError!void {
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
        .index = index,
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
