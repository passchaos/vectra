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
        .signed_start = 0,
        .use_signed_start = false,
        .step = 1,
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
        .signed_start = 0,
        .use_signed_start = false,
        .step = 1,
        .n = n,
        .keep_tail = keep_tail,
    } });
}

pub fn groupBySliceRows(frame: anytype, key_name: []const u8, start: usize, length: usize) DeviceDataError!void {
    return groupBySliceRowsStep(frame, key_name, start, length, 1);
}

pub fn groupBySliceRowsStep(frame: anytype, key_name: []const u8, start: usize, length: usize, step: usize) DeviceDataError!void {
    const owned_key = try frame.allocator.dupe(u8, key_name);
    errdefer frame.allocator.free(owned_key);
    try frame.ops.append(frame.allocator, .{ .group_by_rows = .{
        .key_name = owned_key,
        .start = start,
        .signed_start = 0,
        .use_signed_start = false,
        .step = step,
        .n = length,
        .keep_tail = false,
    } });
}

pub fn groupBySliceRowsOn(frame: anytype, key_names: []const []const u8, start: usize, length: usize) DeviceDataError!void {
    return groupBySliceRowsStepOn(frame, key_names, start, length, 1);
}

pub fn groupBySliceRowsStepOn(frame: anytype, key_names: []const []const u8, start: usize, length: usize, step: usize) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    try frame.ops.append(frame.allocator, .{ .group_by_rows_on = .{
        .key_names = owned_keys,
        .start = start,
        .signed_start = 0,
        .use_signed_start = false,
        .step = step,
        .n = length,
        .keep_tail = false,
    } });
}

pub fn groupBySliceRowsSigned(frame: anytype, key_name: []const u8, start: isize, length: usize) DeviceDataError!void {
    return groupBySliceRowsSignedStep(frame, key_name, start, length, 1);
}

pub fn groupBySliceRowsSignedStep(frame: anytype, key_name: []const u8, start: isize, length: usize, step: usize) DeviceDataError!void {
    const owned_key = try frame.allocator.dupe(u8, key_name);
    errdefer frame.allocator.free(owned_key);
    try frame.ops.append(frame.allocator, .{ .group_by_rows = .{
        .key_name = owned_key,
        .start = 0,
        .signed_start = start,
        .use_signed_start = true,
        .step = step,
        .n = length,
        .keep_tail = false,
    } });
}

pub fn groupBySliceRowsSignedOn(frame: anytype, key_names: []const []const u8, start: isize, length: usize) DeviceDataError!void {
    return groupBySliceRowsSignedStepOn(frame, key_names, start, length, 1);
}

pub fn groupBySliceRowsSignedStepOn(frame: anytype, key_names: []const []const u8, start: isize, length: usize, step: usize) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    try frame.ops.append(frame.allocator, .{ .group_by_rows_on = .{
        .key_names = owned_keys,
        .start = 0,
        .signed_start = start,
        .use_signed_start = true,
        .step = step,
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

pub fn withGroupId(frame: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .group_id = .{
        .names = owned_keys,
        .output_name = owned_output,
    } });
}

pub fn withGroupFirstRowIndex(frame: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .group_first_row_index = .{
        .names = owned_keys,
        .output_name = owned_output,
    } });
}

pub fn withGroupLastRowIndex(frame: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .group_last_row_index = .{
        .names = owned_keys,
        .output_name = owned_output,
    } });
}

pub fn withGroupIsFirstRow(frame: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .group_is_first_row = .{
        .names = owned_keys,
        .output_name = owned_output,
    } });
}

pub fn withGroupIsLastRow(frame: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .group_is_last_row = .{
        .names = owned_keys,
        .output_name = owned_output,
    } });
}

pub fn withGroupIsSingleton(frame: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .group_is_singleton = .{
        .names = owned_keys,
        .output_name = owned_output,
    } });
}

pub fn withGroupIsDuplicated(frame: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .group_is_duplicated = .{
        .names = owned_keys,
        .output_name = owned_output,
    } });
}

pub fn withGroupCumeDist(frame: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .group_cume_dist = .{
        .names = owned_keys,
        .output_name = owned_output,
    } });
}

pub fn withGroupPercentRank(frame: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .group_percent_rank = .{
        .names = owned_keys,
        .output_name = owned_output,
    } });
}

pub fn withGroupReverseCumeDist(frame: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .group_reverse_cume_dist = .{
        .names = owned_keys,
        .output_name = owned_output,
    } });
}

pub fn withGroupReversePercentRank(frame: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .group_reverse_percent_rank = .{
        .names = owned_keys,
        .output_name = owned_output,
    } });
}

fn withGroupShift(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, offset: usize, comptime lead: bool) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, if (lead) .{ .group_lead = .{
        .names = owned_keys,
        .value_name = owned_value,
        .output_name = owned_output,
        .offset = offset,
    } } else .{ .group_lag = .{
        .names = owned_keys,
        .value_name = owned_value,
        .output_name = owned_output,
        .offset = offset,
    } });
}

pub fn withGroupLag(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, offset: usize) DeviceDataError!void {
    return withGroupShift(frame, key_names, value_name, output_name, offset, false);
}

pub fn withGroupLead(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, offset: usize) DeviceDataError!void {
    return withGroupShift(frame, key_names, value_name, output_name, offset, true);
}

fn withGroupBoundaryValue(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, comptime keep_last: bool) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, if (keep_last) .{ .group_last_row_value = .{
        .names = owned_keys,
        .value_name = owned_value,
        .output_name = owned_output,
        .offset = 0,
    } } else .{ .group_first_row_value = .{
        .names = owned_keys,
        .value_name = owned_value,
        .output_name = owned_output,
        .offset = 0,
    } });
}

pub fn withGroupFirstRowValue(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupBoundaryValue(frame, key_names, value_name, output_name, false);
}

pub fn withGroupLastRowValue(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupBoundaryValue(frame, key_names, value_name, output_name, true);
}

pub fn withGroupNthRowValue(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, n: usize) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .group_nth_row_value = .{
        .names = owned_keys,
        .value_name = owned_value,
        .output_name = owned_output,
        .offset = n,
    } });
}

fn withGroupValidValue(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, comptime keep_last: bool) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, if (keep_last) .{ .group_last_valid_value = .{
        .names = owned_keys,
        .value_name = owned_value,
        .output_name = owned_output,
        .offset = 0,
    } } else .{ .group_first_valid_value = .{
        .names = owned_keys,
        .value_name = owned_value,
        .output_name = owned_output,
        .offset = 0,
    } });
}

pub fn withGroupFirstValidValue(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupValidValue(frame, key_names, value_name, output_name, false);
}

pub fn withGroupLastValidValue(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupValidValue(frame, key_names, value_name, output_name, true);
}

pub fn withGroupNthValidValue(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, n: usize) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .group_nth_valid_value = .{
        .names = owned_keys,
        .value_name = owned_value,
        .output_name = owned_output,
        .offset = n,
    } });
}

fn withGroupFillNull(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, comptime backward: bool) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, if (backward) .{ .group_fill_null_backward = .{
        .names = owned_keys,
        .value_name = owned_value,
        .output_name = owned_output,
        .offset = 0,
    } } else .{ .group_fill_null_forward = .{
        .names = owned_keys,
        .value_name = owned_value,
        .output_name = owned_output,
        .offset = 0,
    } });
}

pub fn withGroupFillNullForward(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupFillNull(frame, key_names, value_name, output_name, false);
}

pub fn withGroupFillNullBackward(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupFillNull(frame, key_names, value_name, output_name, true);
}

fn withGroupCumulativeValidityCount(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, comptime count_nulls: bool) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, if (count_nulls) .{ .group_cumulative_null_count = .{
        .names = owned_keys,
        .value_name = owned_value,
        .output_name = owned_output,
        .offset = 0,
    } } else .{ .group_cumulative_valid_count = .{
        .names = owned_keys,
        .value_name = owned_value,
        .output_name = owned_output,
        .offset = 0,
    } });
}

pub fn withGroupCumulativeValidCount(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeValidityCount(frame, key_names, value_name, output_name, false);
}

pub fn withGroupCumulativeNullCount(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeValidityCount(frame, key_names, value_name, output_name, true);
}

fn withGroupCumulativeValidityRatio(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, comptime null_ratio: bool) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, if (null_ratio) .{ .group_cumulative_null_ratio = .{
        .names = owned_keys,
        .value_name = owned_value,
        .output_name = owned_output,
        .offset = 0,
    } } else .{ .group_cumulative_valid_ratio = .{
        .names = owned_keys,
        .value_name = owned_value,
        .output_name = owned_output,
        .offset = 0,
    } });
}

pub fn withGroupCumulativeValidRatio(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeValidityRatio(frame, key_names, value_name, output_name, false);
}

pub fn withGroupCumulativeNullRatio(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeValidityRatio(frame, key_names, value_name, output_name, true);
}

fn withGroupCumulativeValidityIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, comptime op: enum { first_valid, last_valid, first_null, last_null }) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, switch (op) {
        .first_valid => .{ .group_cumulative_first_valid_index = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .last_valid => .{ .group_cumulative_last_valid_index = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .first_null => .{ .group_cumulative_first_null_index = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .last_null => .{ .group_cumulative_last_null_index = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
    });
}

pub fn withGroupCumulativeFirstValidIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeValidityIndex(frame, key_names, value_name, output_name, .first_valid);
}

pub fn withGroupCumulativeLastValidIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeValidityIndex(frame, key_names, value_name, output_name, .last_valid);
}

pub fn withGroupCumulativeFirstNullIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeValidityIndex(frame, key_names, value_name, output_name, .first_null);
}

pub fn withGroupCumulativeLastNullIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeValidityIndex(frame, key_names, value_name, output_name, .last_null);
}

fn withGroupCumulativeQuality(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, comptime op: enum { nan_count, nan_ratio, inf_count, inf_ratio, positive_inf_count, positive_inf_ratio, negative_inf_count, negative_inf_ratio, finite_count, finite_ratio, normal_count, normal_ratio, subnormal_count, subnormal_ratio, non_finite_count, non_finite_ratio, zero_count, zero_ratio, positive_zero_count, positive_zero_ratio, negative_zero_count, negative_zero_ratio, non_zero_count, non_zero_ratio, positive_count, positive_ratio, signbit_count, signbit_ratio, negative_count, negative_ratio }) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, switch (op) {
        .nan_count => .{ .group_cumulative_nan_count = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .nan_ratio => .{ .group_cumulative_nan_ratio = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .inf_count => .{ .group_cumulative_inf_count = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .inf_ratio => .{ .group_cumulative_inf_ratio = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .positive_inf_count => .{ .group_cumulative_positive_inf_count = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .positive_inf_ratio => .{ .group_cumulative_positive_inf_ratio = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .negative_inf_count => .{ .group_cumulative_negative_inf_count = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .negative_inf_ratio => .{ .group_cumulative_negative_inf_ratio = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .finite_count => .{ .group_cumulative_finite_count = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .finite_ratio => .{ .group_cumulative_finite_ratio = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .normal_count => .{ .group_cumulative_normal_count = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .normal_ratio => .{ .group_cumulative_normal_ratio = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .subnormal_count => .{ .group_cumulative_subnormal_count = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .subnormal_ratio => .{ .group_cumulative_subnormal_ratio = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .non_finite_count => .{ .group_cumulative_non_finite_count = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .non_finite_ratio => .{ .group_cumulative_non_finite_ratio = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .zero_count => .{ .group_cumulative_zero_count = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .zero_ratio => .{ .group_cumulative_zero_ratio = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .positive_zero_count => .{ .group_cumulative_positive_zero_count = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .positive_zero_ratio => .{ .group_cumulative_positive_zero_ratio = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .negative_zero_count => .{ .group_cumulative_negative_zero_count = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .negative_zero_ratio => .{ .group_cumulative_negative_zero_ratio = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .non_zero_count => .{ .group_cumulative_non_zero_count = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .non_zero_ratio => .{ .group_cumulative_non_zero_ratio = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .positive_count => .{ .group_cumulative_positive_count = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .positive_ratio => .{ .group_cumulative_positive_ratio = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .signbit_count => .{ .group_cumulative_signbit_count = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .signbit_ratio => .{ .group_cumulative_signbit_ratio = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .negative_count => .{ .group_cumulative_negative_count = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .negative_ratio => .{ .group_cumulative_negative_ratio = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
    });
}

pub fn withGroupCumulativeNaNCount(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQuality(frame, key_names, value_name, output_name, .nan_count);
}

pub fn withGroupCumulativeNaNRatio(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQuality(frame, key_names, value_name, output_name, .nan_ratio);
}

pub fn withGroupCumulativeInfCount(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQuality(frame, key_names, value_name, output_name, .inf_count);
}

pub fn withGroupCumulativeInfRatio(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQuality(frame, key_names, value_name, output_name, .inf_ratio);
}

pub fn withGroupCumulativePositiveInfCount(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQuality(frame, key_names, value_name, output_name, .positive_inf_count);
}

pub fn withGroupCumulativePositiveInfRatio(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQuality(frame, key_names, value_name, output_name, .positive_inf_ratio);
}

pub fn withGroupCumulativeNegativeInfCount(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQuality(frame, key_names, value_name, output_name, .negative_inf_count);
}

pub fn withGroupCumulativeNegativeInfRatio(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQuality(frame, key_names, value_name, output_name, .negative_inf_ratio);
}

pub fn withGroupCumulativeFiniteCount(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQuality(frame, key_names, value_name, output_name, .finite_count);
}

pub fn withGroupCumulativeFiniteRatio(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQuality(frame, key_names, value_name, output_name, .finite_ratio);
}

pub fn withGroupCumulativeNormalCount(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQuality(frame, key_names, value_name, output_name, .normal_count);
}

pub fn withGroupCumulativeNormalRatio(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQuality(frame, key_names, value_name, output_name, .normal_ratio);
}

pub fn withGroupCumulativeSubnormalCount(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQuality(frame, key_names, value_name, output_name, .subnormal_count);
}

pub fn withGroupCumulativeSubnormalRatio(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQuality(frame, key_names, value_name, output_name, .subnormal_ratio);
}

pub fn withGroupCumulativeNonFiniteCount(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQuality(frame, key_names, value_name, output_name, .non_finite_count);
}

pub fn withGroupCumulativeNonFiniteRatio(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQuality(frame, key_names, value_name, output_name, .non_finite_ratio);
}

pub fn withGroupCumulativeZeroCount(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQuality(frame, key_names, value_name, output_name, .zero_count);
}

pub fn withGroupCumulativeZeroRatio(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQuality(frame, key_names, value_name, output_name, .zero_ratio);
}

pub fn withGroupCumulativePositiveZeroCount(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQuality(frame, key_names, value_name, output_name, .positive_zero_count);
}

pub fn withGroupCumulativePositiveZeroRatio(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQuality(frame, key_names, value_name, output_name, .positive_zero_ratio);
}

pub fn withGroupCumulativeNegativeZeroCount(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQuality(frame, key_names, value_name, output_name, .negative_zero_count);
}

pub fn withGroupCumulativeNegativeZeroRatio(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQuality(frame, key_names, value_name, output_name, .negative_zero_ratio);
}

pub fn withGroupCumulativeNonZeroCount(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQuality(frame, key_names, value_name, output_name, .non_zero_count);
}

pub fn withGroupCumulativeNonZeroRatio(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQuality(frame, key_names, value_name, output_name, .non_zero_ratio);
}

pub fn withGroupCumulativePositiveCount(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQuality(frame, key_names, value_name, output_name, .positive_count);
}

pub fn withGroupCumulativePositiveRatio(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQuality(frame, key_names, value_name, output_name, .positive_ratio);
}

pub fn withGroupCumulativeSignBitCount(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQuality(frame, key_names, value_name, output_name, .signbit_count);
}

pub fn withGroupCumulativeSignBitRatio(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQuality(frame, key_names, value_name, output_name, .signbit_ratio);
}

pub fn withGroupCumulativeNegativeCount(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQuality(frame, key_names, value_name, output_name, .negative_count);
}

pub fn withGroupCumulativeNegativeRatio(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQuality(frame, key_names, value_name, output_name, .negative_ratio);
}

fn withGroupCumulativeQualityIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, comptime op: enum { first_nan, last_nan, first_inf, last_inf, first_positive_inf, last_positive_inf, first_negative_inf, last_negative_inf, first_finite, last_finite, first_normal, last_normal, first_subnormal, last_subnormal, first_non_finite, last_non_finite, first_zero, last_zero, first_positive_zero, last_positive_zero, first_negative_zero, last_negative_zero, first_non_zero, last_non_zero, first_positive, last_positive, first_signbit, last_signbit, first_negative, last_negative }) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, switch (op) {
        .first_nan => .{ .group_cumulative_first_nan_index = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .last_nan => .{ .group_cumulative_last_nan_index = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .first_inf => .{ .group_cumulative_first_inf_index = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .last_inf => .{ .group_cumulative_last_inf_index = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .first_positive_inf => .{ .group_cumulative_first_positive_inf_index = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .last_positive_inf => .{ .group_cumulative_last_positive_inf_index = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .first_negative_inf => .{ .group_cumulative_first_negative_inf_index = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .last_negative_inf => .{ .group_cumulative_last_negative_inf_index = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .first_finite => .{ .group_cumulative_first_finite_index = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .last_finite => .{ .group_cumulative_last_finite_index = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .first_normal => .{ .group_cumulative_first_normal_index = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .last_normal => .{ .group_cumulative_last_normal_index = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .first_subnormal => .{ .group_cumulative_first_subnormal_index = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .last_subnormal => .{ .group_cumulative_last_subnormal_index = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .first_non_finite => .{ .group_cumulative_first_non_finite_index = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .last_non_finite => .{ .group_cumulative_last_non_finite_index = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .first_zero => .{ .group_cumulative_first_zero_index = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .last_zero => .{ .group_cumulative_last_zero_index = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .first_positive_zero => .{ .group_cumulative_first_positive_zero_index = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .last_positive_zero => .{ .group_cumulative_last_positive_zero_index = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .first_negative_zero => .{ .group_cumulative_first_negative_zero_index = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .last_negative_zero => .{ .group_cumulative_last_negative_zero_index = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .first_non_zero => .{ .group_cumulative_first_non_zero_index = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .last_non_zero => .{ .group_cumulative_last_non_zero_index = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .first_positive => .{ .group_cumulative_first_positive_index = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .last_positive => .{ .group_cumulative_last_positive_index = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .first_signbit => .{ .group_cumulative_first_signbit_index = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .last_signbit => .{ .group_cumulative_last_signbit_index = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .first_negative => .{ .group_cumulative_first_negative_index = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .last_negative => .{ .group_cumulative_last_negative_index = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
    });
}

pub fn withGroupCumulativeFirstNaNIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQualityIndex(frame, key_names, value_name, output_name, .first_nan);
}

pub fn withGroupCumulativeLastNaNIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQualityIndex(frame, key_names, value_name, output_name, .last_nan);
}

pub fn withGroupCumulativeFirstInfIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQualityIndex(frame, key_names, value_name, output_name, .first_inf);
}

pub fn withGroupCumulativeLastInfIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQualityIndex(frame, key_names, value_name, output_name, .last_inf);
}

pub fn withGroupCumulativeFirstPositiveInfIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQualityIndex(frame, key_names, value_name, output_name, .first_positive_inf);
}

pub fn withGroupCumulativeLastPositiveInfIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQualityIndex(frame, key_names, value_name, output_name, .last_positive_inf);
}

pub fn withGroupCumulativeFirstNegativeInfIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQualityIndex(frame, key_names, value_name, output_name, .first_negative_inf);
}

pub fn withGroupCumulativeLastNegativeInfIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQualityIndex(frame, key_names, value_name, output_name, .last_negative_inf);
}

pub fn withGroupCumulativeFirstFiniteIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQualityIndex(frame, key_names, value_name, output_name, .first_finite);
}

pub fn withGroupCumulativeLastFiniteIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQualityIndex(frame, key_names, value_name, output_name, .last_finite);
}

pub fn withGroupCumulativeFirstNormalIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQualityIndex(frame, key_names, value_name, output_name, .first_normal);
}

pub fn withGroupCumulativeLastNormalIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQualityIndex(frame, key_names, value_name, output_name, .last_normal);
}

pub fn withGroupCumulativeFirstSubnormalIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQualityIndex(frame, key_names, value_name, output_name, .first_subnormal);
}

pub fn withGroupCumulativeLastSubnormalIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQualityIndex(frame, key_names, value_name, output_name, .last_subnormal);
}

pub fn withGroupCumulativeFirstNonFiniteIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQualityIndex(frame, key_names, value_name, output_name, .first_non_finite);
}

pub fn withGroupCumulativeLastNonFiniteIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQualityIndex(frame, key_names, value_name, output_name, .last_non_finite);
}

pub fn withGroupCumulativeFirstZeroIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQualityIndex(frame, key_names, value_name, output_name, .first_zero);
}

pub fn withGroupCumulativeLastZeroIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQualityIndex(frame, key_names, value_name, output_name, .last_zero);
}

pub fn withGroupCumulativeFirstPositiveZeroIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQualityIndex(frame, key_names, value_name, output_name, .first_positive_zero);
}

pub fn withGroupCumulativeLastPositiveZeroIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQualityIndex(frame, key_names, value_name, output_name, .last_positive_zero);
}

pub fn withGroupCumulativeFirstNegativeZeroIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQualityIndex(frame, key_names, value_name, output_name, .first_negative_zero);
}

pub fn withGroupCumulativeLastNegativeZeroIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQualityIndex(frame, key_names, value_name, output_name, .last_negative_zero);
}

pub fn withGroupCumulativeFirstNonZeroIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQualityIndex(frame, key_names, value_name, output_name, .first_non_zero);
}

pub fn withGroupCumulativeLastNonZeroIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQualityIndex(frame, key_names, value_name, output_name, .last_non_zero);
}

pub fn withGroupCumulativeFirstPositiveIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQualityIndex(frame, key_names, value_name, output_name, .first_positive);
}

pub fn withGroupCumulativeLastPositiveIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQualityIndex(frame, key_names, value_name, output_name, .last_positive);
}

pub fn withGroupCumulativeFirstSignBitIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQualityIndex(frame, key_names, value_name, output_name, .first_signbit);
}

pub fn withGroupCumulativeLastSignBitIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQualityIndex(frame, key_names, value_name, output_name, .last_signbit);
}

pub fn withGroupCumulativeFirstNegativeIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQualityIndex(frame, key_names, value_name, output_name, .first_negative);
}

pub fn withGroupCumulativeLastNegativeIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeQualityIndex(frame, key_names, value_name, output_name, .last_negative);
}

fn withGroupCumulativeDistinctCountCore(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, comptime n_unique: bool) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, if (n_unique) .{ .group_cumulative_n_unique = .{
        .names = owned_keys,
        .value_name = owned_value,
        .output_name = owned_output,
        .offset = 0,
    } } else .{ .group_cumulative_distinct_count = .{
        .names = owned_keys,
        .value_name = owned_value,
        .output_name = owned_output,
        .offset = 0,
    } });
}

pub fn withGroupCumulativeDistinctCount(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeDistinctCountCore(frame, key_names, value_name, output_name, false);
}

pub fn withGroupCumulativeNUnique(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeDistinctCountCore(frame, key_names, value_name, output_name, true);
}

fn withGroupCumulativeModeOp(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, comptime op: enum { value, count, ratio, margin, margin_ratio }) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, switch (op) {
        .value => .{ .group_cumulative_mode = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .count => .{ .group_cumulative_mode_count = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .ratio => .{ .group_cumulative_mode_ratio = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .margin => .{ .group_cumulative_mode_margin = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .margin_ratio => .{ .group_cumulative_mode_margin_ratio = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
    });
}

pub fn withGroupCumulativeMode(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeModeOp(frame, key_names, value_name, output_name, .value);
}

pub fn withGroupCumulativeModeCount(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeModeOp(frame, key_names, value_name, output_name, .count);
}

pub fn withGroupCumulativeModeRatio(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeModeOp(frame, key_names, value_name, output_name, .ratio);
}

pub fn withGroupCumulativeModeMargin(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeModeOp(frame, key_names, value_name, output_name, .margin);
}

pub fn withGroupCumulativeModeMarginRatio(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeModeOp(frame, key_names, value_name, output_name, .margin_ratio);
}

fn withGroupCumulativeDistribution(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, comptime op: enum { entropy, gini_impurity, perplexity, inverse_simpson, simpson_concentration, evenness }) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, switch (op) {
        .entropy => .{ .group_cumulative_entropy = .{ .names = owned_keys, .value_name = owned_value, .output_name = owned_output, .offset = 0 } },
        .gini_impurity => .{ .group_cumulative_gini_impurity = .{ .names = owned_keys, .value_name = owned_value, .output_name = owned_output, .offset = 0 } },
        .perplexity => .{ .group_cumulative_perplexity = .{ .names = owned_keys, .value_name = owned_value, .output_name = owned_output, .offset = 0 } },
        .inverse_simpson => .{ .group_cumulative_inverse_simpson = .{ .names = owned_keys, .value_name = owned_value, .output_name = owned_output, .offset = 0 } },
        .simpson_concentration => .{ .group_cumulative_simpson_concentration = .{ .names = owned_keys, .value_name = owned_value, .output_name = owned_output, .offset = 0 } },
        .evenness => .{ .group_cumulative_evenness = .{ .names = owned_keys, .value_name = owned_value, .output_name = owned_output, .offset = 0 } },
    });
}

pub fn withGroupCumulativeEntropy(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeDistribution(frame, key_names, value_name, output_name, .entropy);
}

pub fn withGroupCumulativeGiniImpurity(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeDistribution(frame, key_names, value_name, output_name, .gini_impurity);
}

pub fn withGroupCumulativePerplexity(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeDistribution(frame, key_names, value_name, output_name, .perplexity);
}

pub fn withGroupCumulativeInverseSimpson(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeDistribution(frame, key_names, value_name, output_name, .inverse_simpson);
}

pub fn withGroupCumulativeSimpsonConcentration(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeDistribution(frame, key_names, value_name, output_name, .simpson_concentration);
}

pub fn withGroupCumulativeEvenness(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeDistribution(frame, key_names, value_name, output_name, .evenness);
}

fn withGroupCumulativeInequality(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, comptime op: enum { mean_abs_dev, mean_abs_dev_ratio, gini_mean_diff, gini_coefficient }) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, switch (op) {
        .mean_abs_dev => .{ .group_cumulative_mean_abs_dev = .{ .names = owned_keys, .value_name = owned_value, .output_name = owned_output, .offset = 0 } },
        .mean_abs_dev_ratio => .{ .group_cumulative_mean_abs_dev_ratio = .{ .names = owned_keys, .value_name = owned_value, .output_name = owned_output, .offset = 0 } },
        .gini_mean_diff => .{ .group_cumulative_gini_mean_diff = .{ .names = owned_keys, .value_name = owned_value, .output_name = owned_output, .offset = 0 } },
        .gini_coefficient => .{ .group_cumulative_gini_coefficient = .{ .names = owned_keys, .value_name = owned_value, .output_name = owned_output, .offset = 0 } },
    });
}

pub fn withGroupCumulativeMeanAbsDev(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeInequality(frame, key_names, value_name, output_name, .mean_abs_dev);
}

pub fn withGroupCumulativeMeanAbsDevRatio(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeInequality(frame, key_names, value_name, output_name, .mean_abs_dev_ratio);
}

pub fn withGroupCumulativeGiniMeanDiff(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeInequality(frame, key_names, value_name, output_name, .gini_mean_diff);
}

pub fn withGroupCumulativeGiniCoefficient(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeInequality(frame, key_names, value_name, output_name, .gini_coefficient);
}

pub fn withGroupCumulativeMedian(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .group_cumulative_median = .{
        .names = owned_keys,
        .value_name = owned_value,
        .output_name = owned_output,
        .offset = 0,
    } });
}

pub fn withGroupCumulativeQuantile(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, q: f64) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .group_cumulative_quantile = .{
        .names = owned_keys,
        .value_name = owned_value,
        .output_name = owned_output,
        .quantile = q,
    } });
}

pub fn withGroupCumulativeIqr(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .group_cumulative_iqr = .{ .names = owned_keys, .value_name = owned_value, .output_name = owned_output, .offset = 0 } });
}

pub fn withGroupCumulativeMad(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .group_cumulative_mad = .{ .names = owned_keys, .value_name = owned_value, .output_name = owned_output, .offset = 0 } });
}

pub fn withGroupCumulativeTrimmedMean(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, trim_fraction: f64) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .group_cumulative_trimmed_mean = .{
        .names = owned_keys,
        .value_name = owned_value,
        .output_name = owned_output,
        .quantile = trim_fraction,
    } });
}

pub fn withGroupCumulativeWinsorizedMean(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, winsor_fraction: f64) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .group_cumulative_winsorized_mean = .{
        .names = owned_keys,
        .value_name = owned_value,
        .output_name = owned_output,
        .quantile = winsor_fraction,
    } });
}

pub fn withGroupCumulativeInterdecileRange(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .group_cumulative_interdecile_range = .{ .names = owned_keys, .value_name = owned_value, .output_name = owned_output, .offset = 0 } });
}

pub fn withGroupCumulativeMidhinge(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .group_cumulative_midhinge = .{ .names = owned_keys, .value_name = owned_value, .output_name = owned_output, .offset = 0 } });
}

pub fn withGroupCumulativeTrimean(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .group_cumulative_trimean = .{ .names = owned_keys, .value_name = owned_value, .output_name = owned_output, .offset = 0 } });
}

pub fn withGroupCumulativeBowleySkewness(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .group_cumulative_bowley_skewness = .{ .names = owned_keys, .value_name = owned_value, .output_name = owned_output, .offset = 0 } });
}

pub fn withGroupCumulativeQuartileCoeffDispersion(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .group_cumulative_quartile_coeff_dispersion = .{ .names = owned_keys, .value_name = owned_value, .output_name = owned_output, .offset = 0 } });
}

pub fn withGroupCumulativeKelleySkewness(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .group_cumulative_kelley_skewness = .{ .names = owned_keys, .value_name = owned_value, .output_name = owned_output, .offset = 0 } });
}

fn withGroupCumulativeBool(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, comptime op: enum { any, all, true_count, false_count, true_ratio, false_ratio }) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, switch (op) {
        .any => .{ .group_cumulative_any = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .all => .{ .group_cumulative_all = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .true_count => .{ .group_cumulative_true_count = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .false_count => .{ .group_cumulative_false_count = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .true_ratio => .{ .group_cumulative_true_ratio = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .false_ratio => .{ .group_cumulative_false_ratio = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
    });
}

pub fn withGroupCumulativeAny(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeBool(frame, key_names, value_name, output_name, .any);
}

pub fn withGroupCumulativeAll(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeBool(frame, key_names, value_name, output_name, .all);
}

pub fn withGroupCumulativeTrueCount(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeBool(frame, key_names, value_name, output_name, .true_count);
}

pub fn withGroupCumulativeFalseCount(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeBool(frame, key_names, value_name, output_name, .false_count);
}

pub fn withGroupCumulativeTrueRatio(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeBool(frame, key_names, value_name, output_name, .true_ratio);
}

pub fn withGroupCumulativeFalseRatio(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeBool(frame, key_names, value_name, output_name, .false_ratio);
}

fn withGroupCumulativeBoolIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, comptime op: enum { first_true, last_true, first_false, last_false }) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, switch (op) {
        .first_true => .{ .group_cumulative_first_true_index = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .last_true => .{ .group_cumulative_last_true_index = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .first_false => .{ .group_cumulative_first_false_index = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .last_false => .{ .group_cumulative_last_false_index = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
    });
}

pub fn withGroupCumulativeFirstTrueIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeBoolIndex(frame, key_names, value_name, output_name, .first_true);
}

pub fn withGroupCumulativeLastTrueIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeBoolIndex(frame, key_names, value_name, output_name, .last_true);
}

pub fn withGroupCumulativeFirstFalseIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeBoolIndex(frame, key_names, value_name, output_name, .first_false);
}

pub fn withGroupCumulativeLastFalseIndex(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeBoolIndex(frame, key_names, value_name, output_name, .last_false);
}

fn withGroupCumulativeNumeric(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, comptime op: enum { sum, mean, product, min, max, variance, stddev, sem, cv, fano, skewness, kurtosis, mean_abs, mean_square, rms, max_abs, min_abs, l1_norm, l2_norm, range, midrange, range_coeff, logsumexp, logmeanexp, geometric_mean, harmonic_mean }) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, switch (op) {
        .sum => .{ .group_cumulative_sum = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .mean => .{ .group_cumulative_mean = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .product => .{ .group_cumulative_product = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .min => .{ .group_cumulative_min = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .max => .{ .group_cumulative_max = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .variance => .{ .group_cumulative_variance = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .stddev => .{ .group_cumulative_stddev = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .sem => .{ .group_cumulative_sem = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .cv => .{ .group_cumulative_cv = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .fano => .{ .group_cumulative_fano = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .skewness => .{ .group_cumulative_skewness = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .kurtosis => .{ .group_cumulative_kurtosis = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .mean_abs => .{ .group_cumulative_mean_abs = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .mean_square => .{ .group_cumulative_mean_square = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .rms => .{ .group_cumulative_rms = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .max_abs => .{ .group_cumulative_max_abs = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .min_abs => .{ .group_cumulative_min_abs = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .l1_norm => .{ .group_cumulative_l1_norm = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .l2_norm => .{ .group_cumulative_l2_norm = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .range => .{ .group_cumulative_range = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .midrange => .{ .group_cumulative_midrange = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .range_coeff => .{ .group_cumulative_range_coeff = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .logsumexp => .{ .group_cumulative_logsumexp = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .logmeanexp => .{ .group_cumulative_logmeanexp = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .geometric_mean => .{ .group_cumulative_geometric_mean = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
        .harmonic_mean => .{ .group_cumulative_harmonic_mean = .{
            .names = owned_keys,
            .value_name = owned_value,
            .output_name = owned_output,
            .offset = 0,
        } },
    });
}

pub fn withGroupCumulativeSum(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeNumeric(frame, key_names, value_name, output_name, .sum);
}

pub fn withGroupCumulativeMean(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeNumeric(frame, key_names, value_name, output_name, .mean);
}

fn withGroupCumulativeWeightedMoment(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8, comptime op: enum { sum, product, weight_sum, positive_count, effective_n, mean, mean_square, rms, min, max, mean_abs, l1_norm, l2_norm, max_abs, min_abs, geometric_mean, harmonic_mean, logsumexp, logmeanexp, range, midrange, range_coeff, variance, stddev, sem, cv, fano, skewness, kurtosis }) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_weight = try frame.allocator.dupe(u8, weight_name);
    errdefer frame.allocator.free(owned_weight);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, switch (op) {
        .sum => .{ .group_cumulative_weighted_sum = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .product => .{ .group_cumulative_weighted_product = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .weight_sum => .{ .group_cumulative_weighted_weight_sum = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .positive_count => .{ .group_cumulative_weighted_positive_count = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .effective_n => .{ .group_cumulative_weighted_effective_n = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .mean => .{ .group_cumulative_weighted_mean = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .mean_square => .{ .group_cumulative_weighted_mean_square = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .rms => .{ .group_cumulative_weighted_rms = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .min => .{ .group_cumulative_weighted_min = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .max => .{ .group_cumulative_weighted_max = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .mean_abs => .{ .group_cumulative_weighted_mean_abs = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .l1_norm => .{ .group_cumulative_weighted_l1_norm = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .l2_norm => .{ .group_cumulative_weighted_l2_norm = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .max_abs => .{ .group_cumulative_weighted_max_abs = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .min_abs => .{ .group_cumulative_weighted_min_abs = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .geometric_mean => .{ .group_cumulative_weighted_geometric_mean = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .harmonic_mean => .{ .group_cumulative_weighted_harmonic_mean = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .logsumexp => .{ .group_cumulative_weighted_logsumexp = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .logmeanexp => .{ .group_cumulative_weighted_logmeanexp = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .range => .{ .group_cumulative_weighted_range = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .midrange => .{ .group_cumulative_weighted_midrange = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .range_coeff => .{ .group_cumulative_weighted_range_coeff = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .variance => .{ .group_cumulative_weighted_variance = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .stddev => .{ .group_cumulative_weighted_stddev = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .sem => .{ .group_cumulative_weighted_sem = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .cv => .{ .group_cumulative_weighted_cv = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .fano => .{ .group_cumulative_weighted_fano = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .skewness => .{ .group_cumulative_weighted_skewness = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .kurtosis => .{ .group_cumulative_weighted_kurtosis = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
    });
}

pub fn withGroupCumulativeWeightedMean(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedMoment(frame, key_names, value_name, weight_name, output_name, .mean);
}

pub fn withGroupCumulativeWeightedSum(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedMoment(frame, key_names, value_name, weight_name, output_name, .sum);
}

pub fn withGroupCumulativeWeightedProduct(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedMoment(frame, key_names, value_name, weight_name, output_name, .product);
}

pub fn withGroupCumulativeWeightedWeightSum(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedMoment(frame, key_names, value_name, weight_name, output_name, .weight_sum);
}

pub fn withGroupCumulativeWeightedPositiveCount(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedMoment(frame, key_names, value_name, weight_name, output_name, .positive_count);
}

pub fn withGroupCumulativeWeightedEffectiveN(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedMoment(frame, key_names, value_name, weight_name, output_name, .effective_n);
}

pub fn withGroupCumulativeWeightedMeanSquare(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedMoment(frame, key_names, value_name, weight_name, output_name, .mean_square);
}

pub fn withGroupCumulativeWeightedRms(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedMoment(frame, key_names, value_name, weight_name, output_name, .rms);
}

pub fn withGroupCumulativeWeightedMin(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedMoment(frame, key_names, value_name, weight_name, output_name, .min);
}

pub fn withGroupCumulativeWeightedMax(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedMoment(frame, key_names, value_name, weight_name, output_name, .max);
}

pub fn withGroupCumulativeWeightedMeanAbs(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedMoment(frame, key_names, value_name, weight_name, output_name, .mean_abs);
}
pub fn withGroupCumulativeWeightedL1Norm(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedMoment(frame, key_names, value_name, weight_name, output_name, .l1_norm);
}
pub fn withGroupCumulativeWeightedL2Norm(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedMoment(frame, key_names, value_name, weight_name, output_name, .l2_norm);
}
pub fn withGroupCumulativeWeightedMaxAbs(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedMoment(frame, key_names, value_name, weight_name, output_name, .max_abs);
}
pub fn withGroupCumulativeWeightedMinAbs(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedMoment(frame, key_names, value_name, weight_name, output_name, .min_abs);
}

pub fn withGroupCumulativeWeightedGeometricMean(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedMoment(frame, key_names, value_name, weight_name, output_name, .geometric_mean);
}

pub fn withGroupCumulativeWeightedHarmonicMean(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedMoment(frame, key_names, value_name, weight_name, output_name, .harmonic_mean);
}

pub fn withGroupCumulativeWeightedLogSumExp(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedMoment(frame, key_names, value_name, weight_name, output_name, .logsumexp);
}

pub fn withGroupCumulativeWeightedLogMeanExp(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedMoment(frame, key_names, value_name, weight_name, output_name, .logmeanexp);
}

pub fn withGroupCumulativeWeightedRange(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedMoment(frame, key_names, value_name, weight_name, output_name, .range);
}

pub fn withGroupCumulativeWeightedMidrange(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedMoment(frame, key_names, value_name, weight_name, output_name, .midrange);
}

pub fn withGroupCumulativeWeightedRangeCoeff(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedMoment(frame, key_names, value_name, weight_name, output_name, .range_coeff);
}

fn withGroupCumulativeWeightedQuantileCore(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8, q: f64, comptime op: enum { median, quantile, iqr, mad, trimmed_mean, winsorized_mean }) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_weight = try frame.allocator.dupe(u8, weight_name);
    errdefer frame.allocator.free(owned_weight);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, switch (op) {
        .median => .{ .group_cumulative_weighted_median = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .quantile => .{ .group_cumulative_weighted_quantile = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output, .quantile = q } },
        .iqr => .{ .group_cumulative_weighted_iqr = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .mad => .{ .group_cumulative_weighted_mad = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .trimmed_mean => .{ .group_cumulative_weighted_trimmed_mean = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output, .quantile = q } },
        .winsorized_mean => .{ .group_cumulative_weighted_winsorized_mean = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output, .quantile = q } },
    });
}

pub fn withGroupCumulativeWeightedMedian(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedQuantileCore(frame, key_names, value_name, weight_name, output_name, 0.5, .median);
}

pub fn withGroupCumulativeWeightedQuantile(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8, q: f64) DeviceDataError!void {
    return withGroupCumulativeWeightedQuantileCore(frame, key_names, value_name, weight_name, output_name, q, .quantile);
}

pub fn withGroupCumulativeWeightedIqr(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedQuantileCore(frame, key_names, value_name, weight_name, output_name, 0.5, .iqr);
}

pub fn withGroupCumulativeWeightedMad(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedQuantileCore(frame, key_names, value_name, weight_name, output_name, 0.5, .mad);
}

pub fn withGroupCumulativeWeightedTrimmedMean(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8, trim_fraction: f64) DeviceDataError!void {
    return withGroupCumulativeWeightedQuantileCore(frame, key_names, value_name, weight_name, output_name, trim_fraction, .trimmed_mean);
}

pub fn withGroupCumulativeWeightedWinsorizedMean(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8, winsor_fraction: f64) DeviceDataError!void {
    return withGroupCumulativeWeightedQuantileCore(frame, key_names, value_name, weight_name, output_name, winsor_fraction, .winsorized_mean);
}

fn withGroupCumulativeWeightedModeCore(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8, comptime op: enum { mode, weight, ratio, margin, margin_ratio }) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_weight = try frame.allocator.dupe(u8, weight_name);
    errdefer frame.allocator.free(owned_weight);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, switch (op) {
        .mode => .{ .group_cumulative_weighted_mode = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .weight => .{ .group_cumulative_weighted_mode_weight = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .ratio => .{ .group_cumulative_weighted_mode_ratio = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .margin => .{ .group_cumulative_weighted_mode_margin = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .margin_ratio => .{ .group_cumulative_weighted_mode_margin_ratio = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
    });
}

pub fn withGroupCumulativeWeightedMode(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedModeCore(frame, key_names, value_name, weight_name, output_name, .mode);
}

pub fn withGroupCumulativeWeightedModeWeight(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedModeCore(frame, key_names, value_name, weight_name, output_name, .weight);
}

pub fn withGroupCumulativeWeightedModeRatio(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedModeCore(frame, key_names, value_name, weight_name, output_name, .ratio);
}

pub fn withGroupCumulativeWeightedModeMargin(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedModeCore(frame, key_names, value_name, weight_name, output_name, .margin);
}

pub fn withGroupCumulativeWeightedModeMarginRatio(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedModeCore(frame, key_names, value_name, weight_name, output_name, .margin_ratio);
}

fn withGroupCumulativeWeightedDistributionCore(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8, comptime op: enum { entropy, gini_impurity, perplexity, inverse_simpson, simpson_concentration, evenness }) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_weight = try frame.allocator.dupe(u8, weight_name);
    errdefer frame.allocator.free(owned_weight);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, switch (op) {
        .entropy => .{ .group_cumulative_weighted_entropy = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .gini_impurity => .{ .group_cumulative_weighted_gini_impurity = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .perplexity => .{ .group_cumulative_weighted_perplexity = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .inverse_simpson => .{ .group_cumulative_weighted_inverse_simpson = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .simpson_concentration => .{ .group_cumulative_weighted_simpson_concentration = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .evenness => .{ .group_cumulative_weighted_evenness = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
    });
}

pub fn withGroupCumulativeWeightedEntropy(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedDistributionCore(frame, key_names, value_name, weight_name, output_name, .entropy);
}

pub fn withGroupCumulativeWeightedGiniImpurity(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedDistributionCore(frame, key_names, value_name, weight_name, output_name, .gini_impurity);
}

pub fn withGroupCumulativeWeightedPerplexity(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedDistributionCore(frame, key_names, value_name, weight_name, output_name, .perplexity);
}

pub fn withGroupCumulativeWeightedInverseSimpson(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedDistributionCore(frame, key_names, value_name, weight_name, output_name, .inverse_simpson);
}

pub fn withGroupCumulativeWeightedSimpsonConcentration(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedDistributionCore(frame, key_names, value_name, weight_name, output_name, .simpson_concentration);
}

pub fn withGroupCumulativeWeightedEvenness(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedDistributionCore(frame, key_names, value_name, weight_name, output_name, .evenness);
}

fn withGroupCumulativeWeightedInequality(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8, comptime op: enum { mean_abs_dev, mean_abs_dev_ratio, gini_mean_diff, gini_coefficient }) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_weight = try frame.allocator.dupe(u8, weight_name);
    errdefer frame.allocator.free(owned_weight);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, switch (op) {
        .mean_abs_dev => .{ .group_cumulative_weighted_mean_abs_dev = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .mean_abs_dev_ratio => .{ .group_cumulative_weighted_mean_abs_dev_ratio = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .gini_mean_diff => .{ .group_cumulative_weighted_gini_mean_diff = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
        .gini_coefficient => .{ .group_cumulative_weighted_gini_coefficient = .{ .names = owned_keys, .value_name = owned_value, .weight_name = owned_weight, .output_name = owned_output } },
    });
}

pub fn withGroupCumulativeWeightedMeanAbsDev(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedInequality(frame, key_names, value_name, weight_name, output_name, .mean_abs_dev);
}

pub fn withGroupCumulativeWeightedMeanAbsDevRatio(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedInequality(frame, key_names, value_name, weight_name, output_name, .mean_abs_dev_ratio);
}

pub fn withGroupCumulativeWeightedGiniMeanDiff(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedInequality(frame, key_names, value_name, weight_name, output_name, .gini_mean_diff);
}

pub fn withGroupCumulativeWeightedGiniCoefficient(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedInequality(frame, key_names, value_name, weight_name, output_name, .gini_coefficient);
}

fn withGroupCumulativeWeightedPairMoment(frame: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8, correction: f64, comptime op: enum { dot, cosine_similarity, squared_euclidean_distance, euclidean_distance, manhattan_distance, chebyshev_distance, canberra_distance, bray_curtis_distance, mean_error, mae, mse, rmse, mape, smape, covariance, correlation, beta }) DeviceDataError!void {
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
    try frame.ops.append(frame.allocator, switch (op) {
        .dot => .{ .group_cumulative_weighted_dot = .{ .names = owned_keys, .lhs_name = owned_lhs, .rhs_name = owned_rhs, .weight_name = owned_weight, .output_name = owned_output, .correction = correction } },
        .cosine_similarity => .{ .group_cumulative_weighted_cosine_similarity = .{ .names = owned_keys, .lhs_name = owned_lhs, .rhs_name = owned_rhs, .weight_name = owned_weight, .output_name = owned_output, .correction = correction } },
        .squared_euclidean_distance => .{ .group_cumulative_weighted_squared_euclidean_distance = .{ .names = owned_keys, .lhs_name = owned_lhs, .rhs_name = owned_rhs, .weight_name = owned_weight, .output_name = owned_output, .correction = correction } },
        .euclidean_distance => .{ .group_cumulative_weighted_euclidean_distance = .{ .names = owned_keys, .lhs_name = owned_lhs, .rhs_name = owned_rhs, .weight_name = owned_weight, .output_name = owned_output, .correction = correction } },
        .manhattan_distance => .{ .group_cumulative_weighted_manhattan_distance = .{ .names = owned_keys, .lhs_name = owned_lhs, .rhs_name = owned_rhs, .weight_name = owned_weight, .output_name = owned_output, .correction = correction } },
        .chebyshev_distance => .{ .group_cumulative_weighted_chebyshev_distance = .{ .names = owned_keys, .lhs_name = owned_lhs, .rhs_name = owned_rhs, .weight_name = owned_weight, .output_name = owned_output, .correction = correction } },
        .canberra_distance => .{ .group_cumulative_weighted_canberra_distance = .{ .names = owned_keys, .lhs_name = owned_lhs, .rhs_name = owned_rhs, .weight_name = owned_weight, .output_name = owned_output, .correction = correction } },
        .bray_curtis_distance => .{ .group_cumulative_weighted_bray_curtis_distance = .{ .names = owned_keys, .lhs_name = owned_lhs, .rhs_name = owned_rhs, .weight_name = owned_weight, .output_name = owned_output, .correction = correction } },
        .mean_error => .{ .group_cumulative_weighted_mean_error = .{ .names = owned_keys, .lhs_name = owned_lhs, .rhs_name = owned_rhs, .weight_name = owned_weight, .output_name = owned_output, .correction = correction } },
        .mae => .{ .group_cumulative_weighted_mae = .{ .names = owned_keys, .lhs_name = owned_lhs, .rhs_name = owned_rhs, .weight_name = owned_weight, .output_name = owned_output, .correction = correction } },
        .mse => .{ .group_cumulative_weighted_mse = .{ .names = owned_keys, .lhs_name = owned_lhs, .rhs_name = owned_rhs, .weight_name = owned_weight, .output_name = owned_output, .correction = correction } },
        .rmse => .{ .group_cumulative_weighted_rmse = .{ .names = owned_keys, .lhs_name = owned_lhs, .rhs_name = owned_rhs, .weight_name = owned_weight, .output_name = owned_output, .correction = correction } },
        .mape => .{ .group_cumulative_weighted_mape = .{ .names = owned_keys, .lhs_name = owned_lhs, .rhs_name = owned_rhs, .weight_name = owned_weight, .output_name = owned_output, .correction = correction } },
        .smape => .{ .group_cumulative_weighted_smape = .{ .names = owned_keys, .lhs_name = owned_lhs, .rhs_name = owned_rhs, .weight_name = owned_weight, .output_name = owned_output, .correction = correction } },
        .covariance => .{ .group_cumulative_weighted_covariance = .{ .names = owned_keys, .lhs_name = owned_lhs, .rhs_name = owned_rhs, .weight_name = owned_weight, .output_name = owned_output, .correction = correction } },
        .correlation => .{ .group_cumulative_weighted_correlation = .{ .names = owned_keys, .lhs_name = owned_lhs, .rhs_name = owned_rhs, .weight_name = owned_weight, .output_name = owned_output, .correction = correction } },
        .beta => .{ .group_cumulative_weighted_beta = .{ .names = owned_keys, .lhs_name = owned_lhs, .rhs_name = owned_rhs, .weight_name = owned_weight, .output_name = owned_output, .correction = correction } },
    });
}

pub fn withGroupCumulativeWeightedCovariance(frame: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withGroupCumulativeWeightedPairMoment(frame, key_names, lhs_name, rhs_name, weight_name, output_name, correction, .covariance);
}

pub fn withGroupCumulativeWeightedCorrelation(frame: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withGroupCumulativeWeightedPairMoment(frame, key_names, lhs_name, rhs_name, weight_name, output_name, correction, .correlation);
}

pub fn withGroupCumulativeWeightedBeta(frame: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withGroupCumulativeWeightedPairMoment(frame, key_names, lhs_name, rhs_name, weight_name, output_name, correction, .beta);
}

pub fn withGroupCumulativeWeightedDot(frame: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedPairMoment(frame, key_names, lhs_name, rhs_name, weight_name, output_name, 0.0, .dot);
}

pub fn withGroupCumulativeWeightedCosineSimilarity(frame: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedPairMoment(frame, key_names, lhs_name, rhs_name, weight_name, output_name, 0.0, .cosine_similarity);
}

pub fn withGroupCumulativeWeightedSquaredEuclideanDistance(frame: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedPairMoment(frame, key_names, lhs_name, rhs_name, weight_name, output_name, 0.0, .squared_euclidean_distance);
}

pub fn withGroupCumulativeWeightedEuclideanDistance(frame: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedPairMoment(frame, key_names, lhs_name, rhs_name, weight_name, output_name, 0.0, .euclidean_distance);
}

pub fn withGroupCumulativeWeightedManhattanDistance(frame: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedPairMoment(frame, key_names, lhs_name, rhs_name, weight_name, output_name, 0.0, .manhattan_distance);
}

pub fn withGroupCumulativeWeightedChebyshevDistance(frame: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedPairMoment(frame, key_names, lhs_name, rhs_name, weight_name, output_name, 0.0, .chebyshev_distance);
}

pub fn withGroupCumulativeWeightedCanberraDistance(frame: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedPairMoment(frame, key_names, lhs_name, rhs_name, weight_name, output_name, 0.0, .canberra_distance);
}

pub fn withGroupCumulativeWeightedBrayCurtisDistance(frame: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedPairMoment(frame, key_names, lhs_name, rhs_name, weight_name, output_name, 0.0, .bray_curtis_distance);
}

pub fn withGroupCumulativeWeightedMeanError(frame: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedPairMoment(frame, key_names, lhs_name, rhs_name, weight_name, output_name, 0.0, .mean_error);
}

pub fn withGroupCumulativeWeightedMae(frame: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedPairMoment(frame, key_names, lhs_name, rhs_name, weight_name, output_name, 0.0, .mae);
}

pub fn withGroupCumulativeWeightedMse(frame: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedPairMoment(frame, key_names, lhs_name, rhs_name, weight_name, output_name, 0.0, .mse);
}

pub fn withGroupCumulativeWeightedRmse(frame: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedPairMoment(frame, key_names, lhs_name, rhs_name, weight_name, output_name, 0.0, .rmse);
}

pub fn withGroupCumulativeWeightedMape(frame: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedPairMoment(frame, key_names, lhs_name, rhs_name, weight_name, output_name, 0.0, .mape);
}

pub fn withGroupCumulativeWeightedSmape(frame: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedPairMoment(frame, key_names, lhs_name, rhs_name, weight_name, output_name, 0.0, .smape);
}

pub fn withGroupCumulativeWeightedVariance(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedMoment(frame, key_names, value_name, weight_name, output_name, .variance);
}

pub fn withGroupCumulativeWeightedStddev(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedMoment(frame, key_names, value_name, weight_name, output_name, .stddev);
}

pub fn withGroupCumulativeWeightedSem(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedMoment(frame, key_names, value_name, weight_name, output_name, .sem);
}

pub fn withGroupCumulativeWeightedCv(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedMoment(frame, key_names, value_name, weight_name, output_name, .cv);
}

pub fn withGroupCumulativeWeightedFano(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedMoment(frame, key_names, value_name, weight_name, output_name, .fano);
}

pub fn withGroupCumulativeWeightedSkewness(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedMoment(frame, key_names, value_name, weight_name, output_name, .skewness);
}

pub fn withGroupCumulativeWeightedKurtosis(frame: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeWeightedMoment(frame, key_names, value_name, weight_name, output_name, .kurtosis);
}

pub const withGroupCumulativeWeightedMeanSquared = withGroupCumulativeWeightedMeanSquare;
pub const withGroupCumulativeWeightedMeanSq = withGroupCumulativeWeightedMeanSquare;
pub const withGroupCumulativeWeightedRMS = withGroupCumulativeWeightedRms;
pub const withGroupCumWeightedMeanSquare = withGroupCumulativeWeightedMeanSquare;
pub const withGroupCumWeightedMeanSquared = withGroupCumulativeWeightedMeanSquare;
pub const withGroupCumWeightedMeanSq = withGroupCumulativeWeightedMeanSquare;
pub const withGroupCumWeightedRms = withGroupCumulativeWeightedRms;
pub const withGroupCumWeightedRMS = withGroupCumulativeWeightedRms;
pub const withGroupCumulativeWeightedVar = withGroupCumulativeWeightedVariance;
pub const withGroupCumulativeWeightedStd = withGroupCumulativeWeightedStddev;
pub const withGroupCumulativeWeightedSEM = withGroupCumulativeWeightedSem;
pub const withGroupCumulativeWeightedCV = withGroupCumulativeWeightedCv;
pub const withGroupCumWeightedSem = withGroupCumulativeWeightedSem;
pub const withGroupCumWeightedSEM = withGroupCumulativeWeightedSem;
pub const withGroupCumWeightedCv = withGroupCumulativeWeightedCv;
pub const withGroupCumWeightedCV = withGroupCumulativeWeightedCv;
pub const withGroupCumWeightedFano = withGroupCumulativeWeightedFano;
pub const withGroupCumulativeWeightedSkew = withGroupCumulativeWeightedSkewness;
pub const withGroupCumWeightedSkewness = withGroupCumulativeWeightedSkewness;
pub const withGroupCumWeightedSkew = withGroupCumulativeWeightedSkewness;
pub const withGroupCumulativeWeightedKurt = withGroupCumulativeWeightedKurtosis;
pub const withGroupCumWeightedKurtosis = withGroupCumulativeWeightedKurtosis;
pub const withGroupCumWeightedKurt = withGroupCumulativeWeightedKurtosis;
pub const withGroupCumWeightedMedian = withGroupCumulativeWeightedMedian;
pub const withGroupCumWeightedQuantile = withGroupCumulativeWeightedQuantile;
pub const withGroupCumulativeWeightedIQR = withGroupCumulativeWeightedIqr;
pub const withGroupCumulativeWeightedMAD = withGroupCumulativeWeightedMad;
pub const withGroupCumulativeWeightedMedianAbsDev = withGroupCumulativeWeightedMad;
pub const withGroupCumWeightedIqr = withGroupCumulativeWeightedIqr;
pub const withGroupCumWeightedIQR = withGroupCumulativeWeightedIqr;
pub const withGroupCumWeightedMad = withGroupCumulativeWeightedMad;
pub const withGroupCumWeightedMAD = withGroupCumulativeWeightedMad;
pub const withGroupCumWeightedMedianAbsDev = withGroupCumulativeWeightedMad;
pub const withGroupCumWeightedMode = withGroupCumulativeWeightedMode;
pub const withGroupCumWeightedModeWeight = withGroupCumulativeWeightedModeWeight;
pub const withGroupCumWeightedModeRatio = withGroupCumulativeWeightedModeRatio;
pub const withGroupCumWeightedModeMargin = withGroupCumulativeWeightedModeMargin;
pub const withGroupCumWeightedModeMarginRatio = withGroupCumulativeWeightedModeMarginRatio;
pub const withGroupCumulativeWeightedGini = withGroupCumulativeWeightedGiniImpurity;
pub const withGroupCumulativeWeightedConcentration = withGroupCumulativeWeightedSimpsonConcentration;
pub const withGroupCumulativeWeightedMeanAbsoluteDeviation = withGroupCumulativeWeightedMeanAbsDev;
pub const withGroupCumulativeWeightedGiniCoeff = withGroupCumulativeWeightedGiniCoefficient;
pub const withGroupCumWeightedMeanAbsDev = withGroupCumulativeWeightedMeanAbsDev;
pub const withGroupCumWeightedMeanAbsDevRatio = withGroupCumulativeWeightedMeanAbsDevRatio;
pub const withGroupCumWeightedMeanAbsoluteDeviation = withGroupCumulativeWeightedMeanAbsDev;
pub const withGroupCumWeightedGiniMeanDiff = withGroupCumulativeWeightedGiniMeanDiff;
pub const withGroupCumWeightedGiniCoefficient = withGroupCumulativeWeightedGiniCoefficient;
pub const withGroupCumWeightedGiniCoeff = withGroupCumulativeWeightedGiniCoefficient;
pub const withGroupCumWeightedEntropy = withGroupCumulativeWeightedEntropy;
pub const withGroupCumWeightedGiniImpurity = withGroupCumulativeWeightedGiniImpurity;
pub const withGroupCumWeightedGini = withGroupCumulativeWeightedGiniImpurity;
pub const withGroupCumWeightedPerplexity = withGroupCumulativeWeightedPerplexity;
pub const withGroupCumWeightedInverseSimpson = withGroupCumulativeWeightedInverseSimpson;
pub const withGroupCumWeightedSimpsonConcentration = withGroupCumulativeWeightedSimpsonConcentration;
pub const withGroupCumWeightedConcentration = withGroupCumulativeWeightedSimpsonConcentration;
pub const withGroupCumWeightedEvenness = withGroupCumulativeWeightedEvenness;
pub const withGroupCumulativeWeightedCosine = withGroupCumulativeWeightedCosineSimilarity;
pub const withGroupCumWeightedDot = withGroupCumulativeWeightedDot;
pub const withGroupCumWeightedCosineSimilarity = withGroupCumulativeWeightedCosineSimilarity;
pub const withGroupCumWeightedCosine = withGroupCumulativeWeightedCosineSimilarity;
pub const withGroupCumWeightedSquaredEuclideanDistance = withGroupCumulativeWeightedSquaredEuclideanDistance;
pub const withGroupCumWeightedEuclideanDistance = withGroupCumulativeWeightedEuclideanDistance;
pub const withGroupCumWeightedManhattanDistance = withGroupCumulativeWeightedManhattanDistance;
pub const withGroupCumWeightedChebyshevDistance = withGroupCumulativeWeightedChebyshevDistance;
pub const withGroupCumWeightedCanberraDistance = withGroupCumulativeWeightedCanberraDistance;
pub const withGroupCumWeightedBrayCurtisDistance = withGroupCumulativeWeightedBrayCurtisDistance;
pub const withGroupCumWeightedMeanError = withGroupCumulativeWeightedMeanError;
pub const withGroupCumWeightedBias = withGroupCumulativeWeightedMeanError;
pub const withGroupCumWeightedMae = withGroupCumulativeWeightedMae;
pub const withGroupCumWeightedMse = withGroupCumulativeWeightedMse;
pub const withGroupCumWeightedRmse = withGroupCumulativeWeightedRmse;
pub const withGroupCumWeightedMape = withGroupCumulativeWeightedMape;
pub const withGroupCumWeightedSmape = withGroupCumulativeWeightedSmape;
pub const withGroupCumulativeWeightedCov = withGroupCumulativeWeightedCovariance;
pub const withGroupCumulativeWeightedCorr = withGroupCumulativeWeightedCorrelation;
pub const withGroupCumWeightedCovariance = withGroupCumulativeWeightedCovariance;
pub const withGroupCumWeightedCov = withGroupCumulativeWeightedCovariance;
pub const withGroupCumWeightedCorrelation = withGroupCumulativeWeightedCorrelation;
pub const withGroupCumWeightedCorr = withGroupCumulativeWeightedCorrelation;
pub const withGroupCumWeightedBeta = withGroupCumulativeWeightedBeta;

pub fn withGroupCumulativeProduct(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeNumeric(frame, key_names, value_name, output_name, .product);
}

pub fn withGroupCumulativeMin(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeNumeric(frame, key_names, value_name, output_name, .min);
}

pub fn withGroupCumulativeMax(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeNumeric(frame, key_names, value_name, output_name, .max);
}

pub fn withGroupCumulativeVariance(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeNumeric(frame, key_names, value_name, output_name, .variance);
}

pub fn withGroupCumulativeStddev(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeNumeric(frame, key_names, value_name, output_name, .stddev);
}

pub fn withGroupCumulativeSem(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeNumeric(frame, key_names, value_name, output_name, .sem);
}

pub fn withGroupCumulativeCv(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeNumeric(frame, key_names, value_name, output_name, .cv);
}

pub fn withGroupCumulativeFano(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeNumeric(frame, key_names, value_name, output_name, .fano);
}

pub fn withGroupCumulativeSkewness(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeNumeric(frame, key_names, value_name, output_name, .skewness);
}

pub fn withGroupCumulativeKurtosis(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeNumeric(frame, key_names, value_name, output_name, .kurtosis);
}

pub fn withGroupCumulativeMeanAbs(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeNumeric(frame, key_names, value_name, output_name, .mean_abs);
}

pub fn withGroupCumulativeMeanSquare(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeNumeric(frame, key_names, value_name, output_name, .mean_square);
}

pub fn withGroupCumulativeRms(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeNumeric(frame, key_names, value_name, output_name, .rms);
}

pub fn withGroupCumulativeMaxAbs(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeNumeric(frame, key_names, value_name, output_name, .max_abs);
}

pub fn withGroupCumulativeMinAbs(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeNumeric(frame, key_names, value_name, output_name, .min_abs);
}

pub fn withGroupCumulativeL1Norm(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeNumeric(frame, key_names, value_name, output_name, .l1_norm);
}

pub fn withGroupCumulativeL2Norm(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeNumeric(frame, key_names, value_name, output_name, .l2_norm);
}

pub fn withGroupCumulativeRange(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeNumeric(frame, key_names, value_name, output_name, .range);
}

pub fn withGroupCumulativeMidrange(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeNumeric(frame, key_names, value_name, output_name, .midrange);
}

pub fn withGroupCumulativeRangeCoeff(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeNumeric(frame, key_names, value_name, output_name, .range_coeff);
}

pub fn withGroupCumulativeLogSumExp(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeNumeric(frame, key_names, value_name, output_name, .logsumexp);
}

pub fn withGroupCumulativeLogMeanExp(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeNumeric(frame, key_names, value_name, output_name, .logmeanexp);
}

pub fn withGroupCumulativeGeometricMean(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeNumeric(frame, key_names, value_name, output_name, .geometric_mean);
}

pub fn withGroupCumulativeHarmonicMean(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeNumeric(frame, key_names, value_name, output_name, .harmonic_mean);
}

fn withGroupCumulativeArg(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, comptime argmax: bool) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_value = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_value);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, if (argmax) .{ .group_cumulative_argmax = .{
        .names = owned_keys,
        .value_name = owned_value,
        .output_name = owned_output,
        .offset = 0,
    } } else .{ .group_cumulative_argmin = .{
        .names = owned_keys,
        .value_name = owned_value,
        .output_name = owned_output,
        .offset = 0,
    } });
}

pub fn withGroupCumulativeArgMin(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeArg(frame, key_names, value_name, output_name, false);
}

pub fn withGroupCumulativeArgMax(frame: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return withGroupCumulativeArg(frame, key_names, value_name, output_name, true);
}

pub fn withGroupRowNumber(frame: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .group_row_number = .{
        .names = owned_keys,
        .output_name = owned_output,
    } });
}

pub fn withGroupSize(frame: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .group_size = .{
        .names = owned_keys,
        .output_name = owned_output,
    } });
}

pub fn withGroupReverseRowNumber(frame: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_keys = try cloneNameList(frame.allocator, key_names);
    errdefer freeNameList(frame.allocator, owned_keys);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .group_reverse_row_number = .{
        .names = owned_keys,
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
