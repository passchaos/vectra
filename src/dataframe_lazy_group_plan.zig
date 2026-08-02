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
