//! Multi-key group-by materializers for device dataframes. Keeping these
//! dispatch helpers here keeps the public dataframe facade smaller while the
//! generic `DeviceDataFrame` parameter preserves the original API shape.

const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const dataframe_device_column_mod = @import("dataframe_device_column.zig");
const group_profile_mod = @import("dataframe_group_profile.zig");
const keys_mod = @import("dataframe_keys.zig");
const names_mod = @import("dataframe_names.zig");
const numeric_mod = @import("dataframe_numeric.zig");
const options_mod = @import("dataframe_options.zig");
const validity_mod = @import("dataframe_validity.zig");

const DeviceColumn = dataframe_device_column_mod.DeviceColumn;
const DeviceTypedColumn = dataframe_device_column_mod.DeviceTypedColumn;
const DeviceGroupByAggregation = options_mod.DeviceGroupByAggregation;
const MomentProfile = group_profile_mod.MomentProfile;
const compareSortValues = numeric_mod.compareSortValues;
const castToF64 = numeric_mod.castToF64;
const groupKeyEqual = numeric_mod.groupKeyEqual;
const rowHasValidKeys = keys_mod.rowHasValidKeys;
const columnRowValid = keys_mod.columnRowValid;
const findMultiKeyGroupIndex = keys_mod.findMultiKeyGroupIndex;
const validityValues = validity_mod.validityValues;

const GroupByOnError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    TypeUnsupported,
    UnsupportedType,
    InvalidCsv,
    EmptyDataFrame,
    InvalidDevice,
    IndexOutOfBounds,
};

const GroupByMomentAggregation = enum {
    variance,
    stddev,
    sem,
    cv,
    skewness,
    kurtosis,
};

const GroupByRealAggregation = enum {
    mean_abs,
    rms,
    l1_norm,
    l2_norm,
    geometric_mean,
    harmonic_mean,
    ptp,
    midrange,
    range_coeff,
};

const GroupByBoolAggregation = enum {
    any,
    all,
    true_count,
    false_count,
    true_ratio,
    false_ratio,
};

const GroupByValidityAggregation = enum {
    valid_count,
    null_count,
    valid_ratio,
    null_ratio,
};

const GroupByArgAggregation = enum {
    argmin,
    argmax,
};

pub fn groupByStatsOnDispatchValue(
    comptime DeviceDataFrame: type,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_prefix: []const u8,
    value: DeviceColumn,
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    return switch (value) {
        .i8 => |typed| groupByStatsOnTyped(DeviceDataFrame, i8, allocator, frame, key_names, output_prefix, typed, device_value),
        .i16 => |typed| groupByStatsOnTyped(DeviceDataFrame, i16, allocator, frame, key_names, output_prefix, typed, device_value),
        .i32 => |typed| groupByStatsOnTyped(DeviceDataFrame, i32, allocator, frame, key_names, output_prefix, typed, device_value),
        .i64 => |typed| groupByStatsOnTyped(DeviceDataFrame, i64, allocator, frame, key_names, output_prefix, typed, device_value),
        .u8 => |typed| groupByStatsOnTyped(DeviceDataFrame, u8, allocator, frame, key_names, output_prefix, typed, device_value),
        .u16 => |typed| groupByStatsOnTyped(DeviceDataFrame, u16, allocator, frame, key_names, output_prefix, typed, device_value),
        .u32 => |typed| groupByStatsOnTyped(DeviceDataFrame, u32, allocator, frame, key_names, output_prefix, typed, device_value),
        .u64 => |typed| groupByStatsOnTyped(DeviceDataFrame, u64, allocator, frame, key_names, output_prefix, typed, device_value),
        .usize => |typed| groupByStatsOnTyped(DeviceDataFrame, usize, allocator, frame, key_names, output_prefix, typed, device_value),
        .isize => |typed| groupByStatsOnTyped(DeviceDataFrame, isize, allocator, frame, key_names, output_prefix, typed, device_value),
        .f16 => |typed| groupByStatsOnTyped(DeviceDataFrame, f16, allocator, frame, key_names, output_prefix, typed, device_value),
        .f32 => |typed| groupByStatsOnTyped(DeviceDataFrame, f32, allocator, frame, key_names, output_prefix, typed, device_value),
        .f64 => |typed| groupByStatsOnTyped(DeviceDataFrame, f64, allocator, frame, key_names, output_prefix, typed, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn groupByStatsOnTyped(
    comptime DeviceDataFrame: type,
    comptime V: type,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_prefix: []const u8,
    value: DeviceTypedColumn(V),
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    if (frame.rows != value.len()) return error.LengthMismatch;
    const values = try value.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_value_validity = try validityValues(value, allocator);
    defer if (maybe_value_validity) |validity| allocator.free(validity);

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(allocator);
    var counts: std.ArrayList(i64) = .empty;
    defer counts.deinit(allocator);
    var sums: std.ArrayList(V) = .empty;
    defer sums.deinit(allocator);
    var mins: std.ArrayList(V) = .empty;
    defer mins.deinit(allocator);
    var maxes: std.ArrayList(V) = .empty;
    defer maxes.deinit(allocator);
    var mean_sums: std.ArrayList(f64) = .empty;
    defer mean_sums.deinit(allocator);

    for (values, 0..) |value_item, row| {
        if (maybe_value_validity) |validity| {
            if (!validity[row]) continue;
        }
        if (!try rowHasValidKeys(allocator, frame, key_names, row)) continue;
        const maybe_group_index = try findMultiKeyGroupIndex(allocator, frame, key_names, representative_rows.items, row);
        if (maybe_group_index == null) {
            try representative_rows.append(allocator, row);
            try counts.append(allocator, 1);
            try sums.append(allocator, value_item);
            try mins.append(allocator, value_item);
            try maxes.append(allocator, value_item);
            try mean_sums.append(allocator, castToF64(V, value_item));
            continue;
        }
        const group_index = maybe_group_index.?;
        counts.items[group_index] += 1;
        sums.items[group_index] += value_item;
        if (compareSortValues(V, value_item, mins.items[group_index]) < 0) mins.items[group_index] = value_item;
        if (compareSortValues(V, value_item, maxes.items[group_index]) > 0) maxes.items[group_index] = value_item;
        mean_sums.items[group_index] += castToF64(V, value_item);
    }

    const means = try allocator.alloc(f64, counts.items.len);
    defer allocator.free(means);
    for (mean_sums.items, counts.items, means) |sum_value, count, *slot| {
        slot.* = sum_value / @as(f64, @floatFromInt(count));
    }

    const output_names = try names_mod.statsOutputNames(allocator, "", output_prefix);
    defer names_mod.freeStatsOutputNames(allocator, output_names);
    const total_cols = key_names.len + 5;
    var names = try allocator.alloc([]const u8, total_cols);
    defer allocator.free(names);
    var columns = try allocator.alloc(DeviceColumn, total_cols);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        allocator.free(columns);
    }

    for (key_names) |key_name| {
        names[initialized] = key_name;
        columns[initialized] = try (try frame.column(key_name)).take(representative_rows.items);
        initialized += 1;
    }
    names[initialized] = output_names[1];
    columns[initialized] = try DeviceColumn.fromSlice(i64, allocator, counts.items, device_value);
    initialized += 1;
    names[initialized] = output_names[2];
    columns[initialized] = try DeviceColumn.fromSlice(V, allocator, sums.items, device_value);
    initialized += 1;
    names[initialized] = output_names[3];
    columns[initialized] = try DeviceColumn.fromSlice(V, allocator, mins.items, device_value);
    initialized += 1;
    names[initialized] = output_names[4];
    columns[initialized] = try DeviceColumn.fromSlice(V, allocator, maxes.items, device_value);
    initialized += 1;
    names[initialized] = output_names[5];
    columns[initialized] = try DeviceColumn.fromSlice(f64, allocator, means, device_value);
    initialized += 1;
    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, allocator, names, columns, representative_rows.items.len, device_value);
}

pub fn groupByCountOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(frame.allocator);
    var counts: std.ArrayList(i64) = .empty;
    defer counts.deinit(frame.allocator);

    for (0..frame.rows) |row| {
        if (!try rowHasValidKeys(frame.allocator, frame, key_names, row)) continue;
        const group_index = (try findMultiKeyGroupIndex(frame.allocator, frame, key_names, representative_rows.items, row)) orelse blk: {
            try representative_rows.append(frame.allocator, row);
            try counts.append(frame.allocator, 0);
            break :blk representative_rows.items.len - 1;
        };
        counts.items[group_index] += 1;
    }

    const count_column = try DeviceColumn.fromSlice(i64, frame.allocator, counts.items, frame.device);
    return initMultiKeyAggregatedDataFrame(DeviceDataFrame, frame, key_names, representative_rows.items, output_name, count_column);
}

pub fn groupByNumericOn(
    comptime DeviceDataFrame: type,
    op: DeviceGroupByAggregation,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);
    const value = try frame.column(value_name);
    return groupByNumericOnDispatchValue(DeviceDataFrame, op, frame.allocator, frame, key_names, output_name, value.*, frame.device);
}

pub fn groupByMeanOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);
    const value = try frame.column(value_name);
    return groupByMeanOnDispatchValue(DeviceDataFrame, frame.allocator, frame, key_names, output_name, value.*, frame.device);
}

fn groupByTakeOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
    comptime keep_last: bool,
) GroupByOnError!DeviceDataFrame {
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);
    const value = try frame.column(value_name);

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(frame.allocator);
    var value_rows: std.ArrayList(usize) = .empty;
    defer value_rows.deinit(frame.allocator);

    for (0..frame.rows) |row| {
        if (!try rowHasValidKeys(frame.allocator, frame, key_names, row)) continue;
        if (!try columnRowValid(frame.allocator, value.*, row)) continue;
        const maybe_group_index = try findMultiKeyGroupIndex(frame.allocator, frame, key_names, representative_rows.items, row);
        if (maybe_group_index) |group_index| {
            if (keep_last) value_rows.items[group_index] = row;
        } else {
            try representative_rows.append(frame.allocator, row);
            try value_rows.append(frame.allocator, row);
        }
    }

    const value_column = try value.take(value_rows.items);
    return initMultiKeyAggregatedDataFrame(DeviceDataFrame, frame, key_names, representative_rows.items, output_name, value_column);
}

pub fn groupByFirstOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByTakeOn(DeviceDataFrame, frame, key_names, value_name, output_name, false);
}

pub fn groupByLastOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByTakeOn(DeviceDataFrame, frame, key_names, value_name, output_name, true);
}

pub fn groupByNUniqueOnDispatchValue(
    comptime DeviceDataFrame: type,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
    value: DeviceColumn,
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    return switch (value) {
        .bool => |typed| groupByNUniqueOnTyped(DeviceDataFrame, bool, allocator, frame, key_names, output_name, typed, device_value),
        .i8 => |typed| groupByNUniqueOnTyped(DeviceDataFrame, i8, allocator, frame, key_names, output_name, typed, device_value),
        .i16 => |typed| groupByNUniqueOnTyped(DeviceDataFrame, i16, allocator, frame, key_names, output_name, typed, device_value),
        .i32 => |typed| groupByNUniqueOnTyped(DeviceDataFrame, i32, allocator, frame, key_names, output_name, typed, device_value),
        .i64 => |typed| groupByNUniqueOnTyped(DeviceDataFrame, i64, allocator, frame, key_names, output_name, typed, device_value),
        .u8 => |typed| groupByNUniqueOnTyped(DeviceDataFrame, u8, allocator, frame, key_names, output_name, typed, device_value),
        .u16 => |typed| groupByNUniqueOnTyped(DeviceDataFrame, u16, allocator, frame, key_names, output_name, typed, device_value),
        .u32 => |typed| groupByNUniqueOnTyped(DeviceDataFrame, u32, allocator, frame, key_names, output_name, typed, device_value),
        .u64 => |typed| groupByNUniqueOnTyped(DeviceDataFrame, u64, allocator, frame, key_names, output_name, typed, device_value),
        .usize => |typed| groupByNUniqueOnTyped(DeviceDataFrame, usize, allocator, frame, key_names, output_name, typed, device_value),
        .isize => |typed| groupByNUniqueOnTyped(DeviceDataFrame, isize, allocator, frame, key_names, output_name, typed, device_value),
        .f16 => |typed| groupByNUniqueOnTyped(DeviceDataFrame, f16, allocator, frame, key_names, output_name, typed, device_value),
        .f32 => |typed| groupByNUniqueOnTyped(DeviceDataFrame, f32, allocator, frame, key_names, output_name, typed, device_value),
        .f64 => |typed| groupByNUniqueOnTyped(DeviceDataFrame, f64, allocator, frame, key_names, output_name, typed, device_value),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn groupByNUniqueOnTyped(
    comptime DeviceDataFrame: type,
    comptime V: type,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
    value: DeviceTypedColumn(V),
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    if (frame.rows != value.len()) return error.LengthMismatch;
    const values = try value.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_value_validity = try validityValues(value, allocator);
    defer if (maybe_value_validity) |validity| allocator.free(validity);

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(allocator);
    var distinct_value_rows: std.ArrayList(std.ArrayList(usize)) = .empty;
    defer {
        for (distinct_value_rows.items) |*rows| rows.deinit(allocator);
        distinct_value_rows.deinit(allocator);
    }

    for (values, 0..) |value_item, row| {
        if (maybe_value_validity) |validity| {
            if (!validity[row]) continue;
        }
        if (!try rowHasValidKeys(allocator, frame, key_names, row)) continue;
        const group_index = (try findMultiKeyGroupIndex(allocator, frame, key_names, representative_rows.items, row)) orelse blk: {
            try representative_rows.append(allocator, row);
            try distinct_value_rows.append(allocator, .empty);
            break :blk representative_rows.items.len - 1;
        };

        var seen = false;
        for (distinct_value_rows.items[group_index].items) |previous_row| {
            if (groupKeyEqual(V, values[previous_row], value_item)) {
                seen = true;
                break;
            }
        }
        if (!seen) try distinct_value_rows.items[group_index].append(allocator, row);
    }

    const counts = try allocator.alloc(i64, distinct_value_rows.items.len);
    defer allocator.free(counts);
    for (distinct_value_rows.items, counts) |rows, *slot| {
        slot.* = @intCast(rows.items.len);
    }

    const count_column = try DeviceColumn.fromSlice(i64, allocator, counts, device_value);
    return initMultiKeyAggregatedDataFrame(DeviceDataFrame, frame, key_names, representative_rows.items, output_name, count_column);
}

pub fn groupByNUniqueOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);
    const value = try frame.column(value_name);
    return groupByNUniqueOnDispatchValue(DeviceDataFrame, frame.allocator, frame, key_names, output_name, value.*, frame.device);
}

pub fn groupByModeOnDispatchValue(
    comptime DeviceDataFrame: type,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
    value: DeviceColumn,
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    return switch (value) {
        .bool => |typed| groupByModeOnTyped(DeviceDataFrame, bool, allocator, frame, key_names, output_name, typed, device_value),
        .i8 => |typed| groupByModeOnTyped(DeviceDataFrame, i8, allocator, frame, key_names, output_name, typed, device_value),
        .i16 => |typed| groupByModeOnTyped(DeviceDataFrame, i16, allocator, frame, key_names, output_name, typed, device_value),
        .i32 => |typed| groupByModeOnTyped(DeviceDataFrame, i32, allocator, frame, key_names, output_name, typed, device_value),
        .i64 => |typed| groupByModeOnTyped(DeviceDataFrame, i64, allocator, frame, key_names, output_name, typed, device_value),
        .u8 => |typed| groupByModeOnTyped(DeviceDataFrame, u8, allocator, frame, key_names, output_name, typed, device_value),
        .u16 => |typed| groupByModeOnTyped(DeviceDataFrame, u16, allocator, frame, key_names, output_name, typed, device_value),
        .u32 => |typed| groupByModeOnTyped(DeviceDataFrame, u32, allocator, frame, key_names, output_name, typed, device_value),
        .u64 => |typed| groupByModeOnTyped(DeviceDataFrame, u64, allocator, frame, key_names, output_name, typed, device_value),
        .usize => |typed| groupByModeOnTyped(DeviceDataFrame, usize, allocator, frame, key_names, output_name, typed, device_value),
        .isize => |typed| groupByModeOnTyped(DeviceDataFrame, isize, allocator, frame, key_names, output_name, typed, device_value),
        .f16 => |typed| groupByModeOnTyped(DeviceDataFrame, f16, allocator, frame, key_names, output_name, typed, device_value),
        .f32 => |typed| groupByModeOnTyped(DeviceDataFrame, f32, allocator, frame, key_names, output_name, typed, device_value),
        .f64 => |typed| groupByModeOnTyped(DeviceDataFrame, f64, allocator, frame, key_names, output_name, typed, device_value),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn groupByModeOnTyped(
    comptime DeviceDataFrame: type,
    comptime V: type,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
    value: DeviceTypedColumn(V),
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    if (frame.rows != value.len()) return error.LengthMismatch;
    const values = try value.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_value_validity = try validityValues(value, allocator);
    defer if (maybe_value_validity) |validity| allocator.free(validity);

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(allocator);
    var group_value_rows: std.ArrayList(std.ArrayList(usize)) = .empty;
    defer {
        for (group_value_rows.items) |*rows| rows.deinit(allocator);
        group_value_rows.deinit(allocator);
    }

    for (values, 0..) |_, row| {
        if (maybe_value_validity) |validity| {
            if (!validity[row]) continue;
        }
        if (!try rowHasValidKeys(allocator, frame, key_names, row)) continue;
        const group_index = (try findMultiKeyGroupIndex(allocator, frame, key_names, representative_rows.items, row)) orelse blk: {
            try representative_rows.append(allocator, row);
            try group_value_rows.append(allocator, .empty);
            break :blk representative_rows.items.len - 1;
        };
        try group_value_rows.items[group_index].append(allocator, row);
    }

    const mode_rows = try allocator.alloc(usize, group_value_rows.items.len);
    defer allocator.free(mode_rows);
    for (group_value_rows.items, mode_rows) |rows, *slot| {
        var best_row: usize = rows.items[0];
        var best_count: usize = 0;
        for (rows.items, 0..) |candidate_row, candidate_index| {
            var seen = false;
            for (rows.items[0..candidate_index]) |previous_row| {
                if (groupKeyEqual(V, values[previous_row], values[candidate_row])) {
                    seen = true;
                    break;
                }
            }
            if (seen) continue;

            var count: usize = 0;
            for (rows.items[candidate_index..]) |match_row| {
                if (groupKeyEqual(V, values[candidate_row], values[match_row])) count += 1;
            }
            if (count > best_count) {
                best_row = candidate_row;
                best_count = count;
            }
        }
        slot.* = best_row;
    }

    const mode_values = try allocator.alloc(V, mode_rows.len);
    defer allocator.free(mode_values);
    for (mode_rows, mode_values) |row, *slot| slot.* = values[row];

    const mode_column = try DeviceColumn.fromSlice(V, allocator, mode_values, device_value);
    return initMultiKeyAggregatedDataFrame(DeviceDataFrame, frame, key_names, representative_rows.items, output_name, mode_column);
}

pub fn groupByModeOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);
    const value = try frame.column(value_name);
    return groupByModeOnDispatchValue(DeviceDataFrame, frame.allocator, frame, key_names, output_name, value.*, frame.device);
}

pub fn groupByMomentOnDispatchValue(
    comptime DeviceDataFrame: type,
    aggregation: GroupByMomentAggregation,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
    value: DeviceColumn,
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    return switch (value) {
        .i8 => |typed| groupByMomentOnTyped(DeviceDataFrame, i8, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .i16 => |typed| groupByMomentOnTyped(DeviceDataFrame, i16, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .i32 => |typed| groupByMomentOnTyped(DeviceDataFrame, i32, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .i64 => |typed| groupByMomentOnTyped(DeviceDataFrame, i64, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u8 => |typed| groupByMomentOnTyped(DeviceDataFrame, u8, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u16 => |typed| groupByMomentOnTyped(DeviceDataFrame, u16, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u32 => |typed| groupByMomentOnTyped(DeviceDataFrame, u32, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u64 => |typed| groupByMomentOnTyped(DeviceDataFrame, u64, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .usize => |typed| groupByMomentOnTyped(DeviceDataFrame, usize, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .isize => |typed| groupByMomentOnTyped(DeviceDataFrame, isize, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .f16 => |typed| groupByMomentOnTyped(DeviceDataFrame, f16, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .f32 => |typed| groupByMomentOnTyped(DeviceDataFrame, f32, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .f64 => |typed| groupByMomentOnTyped(DeviceDataFrame, f64, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn groupByMomentOnTyped(
    comptime DeviceDataFrame: type,
    comptime V: type,
    aggregation: GroupByMomentAggregation,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
    value: DeviceTypedColumn(V),
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    if (frame.rows != value.len()) return error.LengthMismatch;
    const values = try value.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_value_validity = try validityValues(value, allocator);
    defer if (maybe_value_validity) |validity| allocator.free(validity);

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(allocator);
    var profiles: std.ArrayList(MomentProfile) = .empty;
    defer profiles.deinit(allocator);

    for (values, 0..) |value_item, row| {
        if (maybe_value_validity) |validity| {
            if (!validity[row]) continue;
        }
        if (!try rowHasValidKeys(allocator, frame, key_names, row)) continue;
        const group_index = (try findMultiKeyGroupIndex(allocator, frame, key_names, representative_rows.items, row)) orelse blk: {
            try representative_rows.append(allocator, row);
            try profiles.append(allocator, .{});
            break :blk representative_rows.items.len - 1;
        };
        profiles.items[group_index].update(castToF64(V, value_item));
    }

    const values_out = try allocator.alloc(f64, profiles.items.len);
    defer allocator.free(values_out);
    for (profiles.items, values_out) |profile, *slot| {
        slot.* = switch (aggregation) {
            .variance => profile.variance(),
            .stddev => profile.stddev(),
            .sem => profile.sem(),
            .cv => profile.cv(),
            .skewness => profile.skewness(),
            .kurtosis => profile.kurtosis(),
        };
    }

    const output_column = try DeviceColumn.fromSlice(f64, allocator, values_out, device_value);
    return initMultiKeyAggregatedDataFrame(DeviceDataFrame, frame, key_names, representative_rows.items, output_name, output_column);
}

fn groupByMomentOn(
    comptime DeviceDataFrame: type,
    aggregation: GroupByMomentAggregation,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);
    const value = try frame.column(value_name);
    return groupByMomentOnDispatchValue(DeviceDataFrame, aggregation, frame.allocator, frame, key_names, output_name, value.*, frame.device);
}

pub fn groupByVarianceOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByMomentOn(DeviceDataFrame, .variance, frame, key_names, value_name, output_name);
}

pub fn groupByStddevOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByMomentOn(DeviceDataFrame, .stddev, frame, key_names, value_name, output_name);
}

pub fn groupBySemOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByMomentOn(DeviceDataFrame, .sem, frame, key_names, value_name, output_name);
}

pub fn groupByCvOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByMomentOn(DeviceDataFrame, .cv, frame, key_names, value_name, output_name);
}

pub fn groupBySkewnessOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByMomentOn(DeviceDataFrame, .skewness, frame, key_names, value_name, output_name);
}

pub fn groupByKurtosisOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByMomentOn(DeviceDataFrame, .kurtosis, frame, key_names, value_name, output_name);
}

pub fn groupByRealOnDispatchValue(
    comptime DeviceDataFrame: type,
    aggregation: GroupByRealAggregation,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
    value: DeviceColumn,
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    return switch (value) {
        .i8 => |typed| groupByRealOnTyped(DeviceDataFrame, i8, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .i16 => |typed| groupByRealOnTyped(DeviceDataFrame, i16, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .i32 => |typed| groupByRealOnTyped(DeviceDataFrame, i32, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .i64 => |typed| groupByRealOnTyped(DeviceDataFrame, i64, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u8 => |typed| groupByRealOnTyped(DeviceDataFrame, u8, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u16 => |typed| groupByRealOnTyped(DeviceDataFrame, u16, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u32 => |typed| groupByRealOnTyped(DeviceDataFrame, u32, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u64 => |typed| groupByRealOnTyped(DeviceDataFrame, u64, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .usize => |typed| groupByRealOnTyped(DeviceDataFrame, usize, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .isize => |typed| groupByRealOnTyped(DeviceDataFrame, isize, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .f16 => |typed| groupByRealOnTyped(DeviceDataFrame, f16, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .f32 => |typed| groupByRealOnTyped(DeviceDataFrame, f32, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .f64 => |typed| groupByRealOnTyped(DeviceDataFrame, f64, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn groupByRealOnTyped(
    comptime DeviceDataFrame: type,
    comptime V: type,
    aggregation: GroupByRealAggregation,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
    value: DeviceTypedColumn(V),
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    if (frame.rows != value.len()) return error.LengthMismatch;
    const values = try value.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_value_validity = try validityValues(value, allocator);
    defer if (maybe_value_validity) |validity| allocator.free(validity);

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(allocator);
    var totals: std.ArrayList(f64) = .empty;
    defer totals.deinit(allocator);
    var counts: std.ArrayList(i64) = .empty;
    defer counts.deinit(allocator);
    var zero_seen: std.ArrayList(bool) = .empty;
    defer zero_seen.deinit(allocator);
    var aux_values: std.ArrayList(f64) = .empty;
    defer aux_values.deinit(allocator);

    for (values, 0..) |value_item, row| {
        if (maybe_value_validity) |validity| {
            if (!validity[row]) continue;
        }
        if (!try rowHasValidKeys(allocator, frame, key_names, row)) continue;
        const group_index = (try findMultiKeyGroupIndex(allocator, frame, key_names, representative_rows.items, row)) orelse blk: {
            try representative_rows.append(allocator, row);
            try totals.append(allocator, 0.0);
            try counts.append(allocator, 0);
            try zero_seen.append(allocator, false);
            try aux_values.append(allocator, 0.0);
            break :blk representative_rows.items.len - 1;
        };
        const value_f64 = castToF64(V, value_item);
        switch (aggregation) {
            .mean_abs, .l1_norm => totals.items[group_index] += @abs(value_f64),
            .rms, .l2_norm => totals.items[group_index] += value_f64 * value_f64,
            .geometric_mean => {
                if (value_f64 < 0.0) {
                    totals.items[group_index] = std.math.nan(f64);
                } else if (value_f64 == 0.0 and !std.math.isNan(totals.items[group_index])) {
                    zero_seen.items[group_index] = true;
                    totals.items[group_index] = 0.0;
                } else if (!zero_seen.items[group_index] and !std.math.isNan(totals.items[group_index])) {
                    totals.items[group_index] += std.math.log(f64, std.math.e, value_f64);
                }
            },
            .harmonic_mean => {
                if (value_f64 == 0.0 and !std.math.isNan(totals.items[group_index])) {
                    totals.items[group_index] = std.math.inf(f64);
                } else if (!std.math.isInf(totals.items[group_index])) {
                    totals.items[group_index] += 1.0 / value_f64;
                }
            },
            .ptp, .midrange, .range_coeff => {
                if (counts.items[group_index] == 0) {
                    totals.items[group_index] = value_f64;
                    aux_values.items[group_index] = value_f64;
                } else if (std.math.isNan(value_f64)) {
                    totals.items[group_index] = value_f64;
                    aux_values.items[group_index] = value_f64;
                } else if (!std.math.isNan(totals.items[group_index])) {
                    if (value_f64 < totals.items[group_index]) totals.items[group_index] = value_f64;
                    if (value_f64 > aux_values.items[group_index]) aux_values.items[group_index] = value_f64;
                }
            },
        }
        counts.items[group_index] += 1;
    }

    const out = try allocator.alloc(f64, totals.items.len);
    defer allocator.free(out);
    for (totals.items, counts.items, zero_seen.items, aux_values.items, out) |total, count, has_zero, aux_value, *slot| {
        slot.* = switch (aggregation) {
            .mean_abs => total / @as(f64, @floatFromInt(count)),
            .rms => std.math.sqrt(total / @as(f64, @floatFromInt(count))),
            .l1_norm => total,
            .l2_norm => std.math.sqrt(total),
            .geometric_mean => if (std.math.isNan(total)) std.math.nan(f64) else if (has_zero) 0.0 else std.math.exp(total / @as(f64, @floatFromInt(count))),
            .harmonic_mean => if (std.math.isInf(total)) 0.0 else @as(f64, @floatFromInt(count)) / total,
            .ptp => aux_value - total,
            .midrange => (total + aux_value) / 2.0,
            .range_coeff => blk: {
                const denominator = aux_value + total;
                break :blk if (denominator == 0.0) std.math.nan(f64) else (aux_value - total) / denominator;
            },
        };
    }

    const output_column = try DeviceColumn.fromSlice(f64, allocator, out, device_value);
    return initMultiKeyAggregatedDataFrame(DeviceDataFrame, frame, key_names, representative_rows.items, output_name, output_column);
}

fn groupByRealOn(
    comptime DeviceDataFrame: type,
    aggregation: GroupByRealAggregation,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);
    const value = try frame.column(value_name);
    return groupByRealOnDispatchValue(DeviceDataFrame, aggregation, frame.allocator, frame, key_names, output_name, value.*, frame.device);
}

pub fn groupByMeanAbsOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRealOn(DeviceDataFrame, .mean_abs, frame, key_names, value_name, output_name);
}

pub fn groupByRmsOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRealOn(DeviceDataFrame, .rms, frame, key_names, value_name, output_name);
}

pub fn groupByL1NormOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRealOn(DeviceDataFrame, .l1_norm, frame, key_names, value_name, output_name);
}

pub fn groupByL2NormOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRealOn(DeviceDataFrame, .l2_norm, frame, key_names, value_name, output_name);
}

pub fn groupByGeometricMeanOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRealOn(DeviceDataFrame, .geometric_mean, frame, key_names, value_name, output_name);
}

pub fn groupByHarmonicMeanOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRealOn(DeviceDataFrame, .harmonic_mean, frame, key_names, value_name, output_name);
}

pub fn groupByPtpOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRealOn(DeviceDataFrame, .ptp, frame, key_names, value_name, output_name);
}

pub fn groupByMidrangeOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRealOn(DeviceDataFrame, .midrange, frame, key_names, value_name, output_name);
}

pub fn groupByRangeCoeffOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRealOn(DeviceDataFrame, .range_coeff, frame, key_names, value_name, output_name);
}

fn groupByQuantileLess(_: void, lhs: f64, rhs: f64) bool {
    const lhs_nan = std.math.isNan(lhs);
    const rhs_nan = std.math.isNan(rhs);
    if (lhs_nan != rhs_nan) return !lhs_nan;
    if (lhs_nan and rhs_nan) return false;
    return lhs < rhs;
}

fn quantileFromSorted(sorted_values: []const f64, q: f64) f64 {
    const max_index = sorted_values.len - 1;
    const position = q * @as(f64, @floatFromInt(max_index));
    const lower_float = @floor(position);
    const lower: usize = @intFromFloat(lower_float);
    const upper = @min(lower + 1, max_index);
    const weight = position - lower_float;
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight;
}

pub fn groupByQuantileOnDispatchValue(
    comptime DeviceDataFrame: type,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
    value: DeviceColumn,
    device_value: array_mod.Device,
    q: f64,
) GroupByOnError!DeviceDataFrame {
    if (std.math.isNan(q) or q < 0.0 or q > 1.0) return error.InvalidShape;
    return switch (value) {
        .i8 => |typed| groupByQuantileOnTyped(DeviceDataFrame, i8, allocator, frame, key_names, output_name, typed, device_value, q),
        .i16 => |typed| groupByQuantileOnTyped(DeviceDataFrame, i16, allocator, frame, key_names, output_name, typed, device_value, q),
        .i32 => |typed| groupByQuantileOnTyped(DeviceDataFrame, i32, allocator, frame, key_names, output_name, typed, device_value, q),
        .i64 => |typed| groupByQuantileOnTyped(DeviceDataFrame, i64, allocator, frame, key_names, output_name, typed, device_value, q),
        .u8 => |typed| groupByQuantileOnTyped(DeviceDataFrame, u8, allocator, frame, key_names, output_name, typed, device_value, q),
        .u16 => |typed| groupByQuantileOnTyped(DeviceDataFrame, u16, allocator, frame, key_names, output_name, typed, device_value, q),
        .u32 => |typed| groupByQuantileOnTyped(DeviceDataFrame, u32, allocator, frame, key_names, output_name, typed, device_value, q),
        .u64 => |typed| groupByQuantileOnTyped(DeviceDataFrame, u64, allocator, frame, key_names, output_name, typed, device_value, q),
        .usize => |typed| groupByQuantileOnTyped(DeviceDataFrame, usize, allocator, frame, key_names, output_name, typed, device_value, q),
        .isize => |typed| groupByQuantileOnTyped(DeviceDataFrame, isize, allocator, frame, key_names, output_name, typed, device_value, q),
        .f16 => |typed| groupByQuantileOnTyped(DeviceDataFrame, f16, allocator, frame, key_names, output_name, typed, device_value, q),
        .f32 => |typed| groupByQuantileOnTyped(DeviceDataFrame, f32, allocator, frame, key_names, output_name, typed, device_value, q),
        .f64 => |typed| groupByQuantileOnTyped(DeviceDataFrame, f64, allocator, frame, key_names, output_name, typed, device_value, q),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn groupByQuantileOnTyped(
    comptime DeviceDataFrame: type,
    comptime V: type,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
    value: DeviceTypedColumn(V),
    device_value: array_mod.Device,
    q: f64,
) GroupByOnError!DeviceDataFrame {
    if (frame.rows != value.len()) return error.LengthMismatch;
    const values = try value.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_value_validity = try validityValues(value, allocator);
    defer if (maybe_value_validity) |validity| allocator.free(validity);

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(allocator);
    var group_values: std.ArrayList(std.ArrayList(f64)) = .empty;
    defer {
        for (group_values.items) |*rows| rows.deinit(allocator);
        group_values.deinit(allocator);
    }

    for (values, 0..) |value_item, row| {
        if (maybe_value_validity) |validity| {
            if (!validity[row]) continue;
        }
        if (!try rowHasValidKeys(allocator, frame, key_names, row)) continue;
        const group_index = (try findMultiKeyGroupIndex(allocator, frame, key_names, representative_rows.items, row)) orelse blk: {
            try representative_rows.append(allocator, row);
            try group_values.append(allocator, .empty);
            break :blk representative_rows.items.len - 1;
        };
        try group_values.items[group_index].append(allocator, castToF64(V, value_item));
    }

    const quantiles = try allocator.alloc(f64, group_values.items.len);
    defer allocator.free(quantiles);
    for (group_values.items, quantiles) |values_for_group, *slot| {
        std.sort.insertion(f64, values_for_group.items, {}, groupByQuantileLess);
        slot.* = quantileFromSorted(values_for_group.items, q);
    }

    const quantile_column = try DeviceColumn.fromSlice(f64, allocator, quantiles, device_value);
    return initMultiKeyAggregatedDataFrame(DeviceDataFrame, frame, key_names, representative_rows.items, output_name, quantile_column);
}

pub fn groupByQuantileOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
    q: f64,
) GroupByOnError!DeviceDataFrame {
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);
    const value = try frame.column(value_name);
    return groupByQuantileOnDispatchValue(DeviceDataFrame, frame.allocator, frame, key_names, output_name, value.*, frame.device, q);
}

pub fn groupByMedianOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByQuantileOn(DeviceDataFrame, frame, key_names, value_name, output_name, 0.5);
}

fn groupByBoolOn(
    comptime DeviceDataFrame: type,
    aggregation: GroupByBoolAggregation,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);
    const value = try frame.column(value_name);
    if (value.* != .bool) return error.TypeUnsupported;

    const values = try value.bool.values.toOwnedSlice(frame.allocator);
    defer frame.allocator.free(values);
    const maybe_value_validity = try validityValues(value.bool, frame.allocator);
    defer if (maybe_value_validity) |validity| frame.allocator.free(validity);

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(frame.allocator);
    var true_counts: std.ArrayList(i64) = .empty;
    defer true_counts.deinit(frame.allocator);
    var false_counts: std.ArrayList(i64) = .empty;
    defer false_counts.deinit(frame.allocator);

    for (values, 0..) |value_item, row| {
        if (maybe_value_validity) |validity| {
            if (!validity[row]) continue;
        }
        if (!try rowHasValidKeys(frame.allocator, frame, key_names, row)) continue;
        const group_index = (try findMultiKeyGroupIndex(frame.allocator, frame, key_names, representative_rows.items, row)) orelse blk: {
            try representative_rows.append(frame.allocator, row);
            try true_counts.append(frame.allocator, 0);
            try false_counts.append(frame.allocator, 0);
            break :blk representative_rows.items.len - 1;
        };
        if (value_item) {
            true_counts.items[group_index] += 1;
        } else {
            false_counts.items[group_index] += 1;
        }
    }

    const output_column: DeviceColumn = switch (aggregation) {
        .any, .all => blk: {
            const outputs = try frame.allocator.alloc(bool, true_counts.items.len);
            defer frame.allocator.free(outputs);
            for (true_counts.items, false_counts.items, outputs) |true_count, false_count, *slot| {
                slot.* = switch (aggregation) {
                    .any => true_count != 0,
                    .all => false_count == 0,
                    else => unreachable,
                };
            }
            break :blk try DeviceColumn.fromSlice(bool, frame.allocator, outputs, frame.device);
        },
        .true_count => try DeviceColumn.fromSlice(i64, frame.allocator, true_counts.items, frame.device),
        .false_count => try DeviceColumn.fromSlice(i64, frame.allocator, false_counts.items, frame.device),
        .true_ratio, .false_ratio => blk: {
            const ratios = try frame.allocator.alloc(f64, true_counts.items.len);
            defer frame.allocator.free(ratios);
            for (true_counts.items, false_counts.items, ratios) |true_count, false_count, *slot| {
                const valid_count = true_count + false_count;
                if (valid_count == 0) {
                    slot.* = std.math.nan(f64);
                    continue;
                }
                const numerator = switch (aggregation) {
                    .true_ratio => true_count,
                    .false_ratio => false_count,
                    else => unreachable,
                };
                slot.* = @as(f64, @floatFromInt(numerator)) / @as(f64, @floatFromInt(valid_count));
            }
            break :blk try DeviceColumn.fromSlice(f64, frame.allocator, ratios, frame.device);
        },
    };
    return initMultiKeyAggregatedDataFrame(DeviceDataFrame, frame, key_names, representative_rows.items, output_name, output_column);
}

pub fn groupByAnyOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByBoolOn(DeviceDataFrame, .any, frame, key_names, value_name, output_name);
}

pub fn groupByAllOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByBoolOn(DeviceDataFrame, .all, frame, key_names, value_name, output_name);
}

pub fn groupByTrueCountOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByBoolOn(DeviceDataFrame, .true_count, frame, key_names, value_name, output_name);
}

pub fn groupByFalseCountOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByBoolOn(DeviceDataFrame, .false_count, frame, key_names, value_name, output_name);
}

pub fn groupByTrueRatioOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByBoolOn(DeviceDataFrame, .true_ratio, frame, key_names, value_name, output_name);
}

pub fn groupByFalseRatioOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByBoolOn(DeviceDataFrame, .false_ratio, frame, key_names, value_name, output_name);
}

fn groupByValidityCountOn(
    comptime DeviceDataFrame: type,
    aggregation: GroupByValidityAggregation,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);
    const value = try frame.column(value_name);

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(frame.allocator);
    var valid_counts: std.ArrayList(i64) = .empty;
    defer valid_counts.deinit(frame.allocator);
    var null_counts: std.ArrayList(i64) = .empty;
    defer null_counts.deinit(frame.allocator);

    for (0..frame.rows) |row| {
        if (!try rowHasValidKeys(frame.allocator, frame, key_names, row)) continue;
        const value_valid = try columnRowValid(frame.allocator, value.*, row);
        const group_index = (try findMultiKeyGroupIndex(frame.allocator, frame, key_names, representative_rows.items, row)) orelse blk: {
            try representative_rows.append(frame.allocator, row);
            try valid_counts.append(frame.allocator, 0);
            try null_counts.append(frame.allocator, 0);
            break :blk representative_rows.items.len - 1;
        };
        if (value_valid) {
            valid_counts.items[group_index] += 1;
        } else {
            null_counts.items[group_index] += 1;
        }
    }

    const output_column: DeviceColumn = switch (aggregation) {
        .valid_count => try DeviceColumn.fromSlice(i64, frame.allocator, valid_counts.items, frame.device),
        .null_count => try DeviceColumn.fromSlice(i64, frame.allocator, null_counts.items, frame.device),
        .valid_ratio, .null_ratio => blk: {
            const ratios = try frame.allocator.alloc(f64, valid_counts.items.len);
            defer frame.allocator.free(ratios);
            for (valid_counts.items, null_counts.items, ratios) |valid_count, null_count, *slot| {
                const total_count = valid_count + null_count;
                if (total_count == 0) {
                    slot.* = std.math.nan(f64);
                    continue;
                }
                const numerator = switch (aggregation) {
                    .valid_ratio => valid_count,
                    .null_ratio => null_count,
                    else => unreachable,
                };
                slot.* = @as(f64, @floatFromInt(numerator)) / @as(f64, @floatFromInt(total_count));
            }
            break :blk try DeviceColumn.fromSlice(f64, frame.allocator, ratios, frame.device);
        },
    };
    return initMultiKeyAggregatedDataFrame(DeviceDataFrame, frame, key_names, representative_rows.items, output_name, output_column);
}

pub fn groupByValidCountOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByValidityCountOn(DeviceDataFrame, .valid_count, frame, key_names, value_name, output_name);
}

pub fn groupByNullCountOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByValidityCountOn(DeviceDataFrame, .null_count, frame, key_names, value_name, output_name);
}

pub fn groupByValidRatioOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByValidityCountOn(DeviceDataFrame, .valid_ratio, frame, key_names, value_name, output_name);
}

pub fn groupByNullRatioOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByValidityCountOn(DeviceDataFrame, .null_ratio, frame, key_names, value_name, output_name);
}

pub fn groupByArgOnDispatchValue(
    comptime DeviceDataFrame: type,
    aggregation: GroupByArgAggregation,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
    value: DeviceColumn,
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    return switch (value) {
        .i8 => |typed| groupByArgOnTyped(DeviceDataFrame, i8, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .i16 => |typed| groupByArgOnTyped(DeviceDataFrame, i16, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .i32 => |typed| groupByArgOnTyped(DeviceDataFrame, i32, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .i64 => |typed| groupByArgOnTyped(DeviceDataFrame, i64, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u8 => |typed| groupByArgOnTyped(DeviceDataFrame, u8, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u16 => |typed| groupByArgOnTyped(DeviceDataFrame, u16, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u32 => |typed| groupByArgOnTyped(DeviceDataFrame, u32, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u64 => |typed| groupByArgOnTyped(DeviceDataFrame, u64, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .usize => |typed| groupByArgOnTyped(DeviceDataFrame, usize, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .isize => |typed| groupByArgOnTyped(DeviceDataFrame, isize, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .f16 => |typed| groupByArgOnTyped(DeviceDataFrame, f16, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .f32 => |typed| groupByArgOnTyped(DeviceDataFrame, f32, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .f64 => |typed| groupByArgOnTyped(DeviceDataFrame, f64, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn groupByArgOnTyped(
    comptime DeviceDataFrame: type,
    comptime V: type,
    aggregation: GroupByArgAggregation,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
    value: DeviceTypedColumn(V),
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    if (frame.rows != value.len()) return error.LengthMismatch;
    const values = try value.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_value_validity = try validityValues(value, allocator);
    defer if (maybe_value_validity) |validity| allocator.free(validity);

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(allocator);
    var best_rows: std.ArrayList(usize) = .empty;
    defer best_rows.deinit(allocator);

    for (values, 0..) |value_item, row| {
        if (maybe_value_validity) |validity| {
            if (!validity[row]) continue;
        }
        if (!try rowHasValidKeys(allocator, frame, key_names, row)) continue;
        const group_index = (try findMultiKeyGroupIndex(allocator, frame, key_names, representative_rows.items, row)) orelse blk: {
            try representative_rows.append(allocator, row);
            try best_rows.append(allocator, row);
            break :blk representative_rows.items.len - 1;
        };
        const best_row = best_rows.items[group_index];
        const better = switch (aggregation) {
            .argmin => compareSortValues(V, value_item, values[best_row]) < 0,
            .argmax => compareSortValues(V, value_item, values[best_row]) > 0,
        };
        if (better) best_rows.items[group_index] = row;
    }

    const out = try allocator.alloc(i64, best_rows.items.len);
    defer allocator.free(out);
    for (best_rows.items, out) |row, *slot| slot.* = @intCast(row);

    const output_column = try DeviceColumn.fromSlice(i64, allocator, out, device_value);
    return initMultiKeyAggregatedDataFrame(DeviceDataFrame, frame, key_names, representative_rows.items, output_name, output_column);
}

fn groupByArgOn(
    comptime DeviceDataFrame: type,
    aggregation: GroupByArgAggregation,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);
    const value = try frame.column(value_name);
    return groupByArgOnDispatchValue(DeviceDataFrame, aggregation, frame.allocator, frame, key_names, output_name, value.*, frame.device);
}

pub fn groupByArgMinOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByArgOn(DeviceDataFrame, .argmin, frame, key_names, value_name, output_name);
}

pub fn groupByArgMaxOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByArgOn(DeviceDataFrame, .argmax, frame, key_names, value_name, output_name);
}

fn initMultiKeyAggregatedDataFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    representative_rows: []const usize,
    output_name: []const u8,
    value_column: DeviceColumn,
) GroupByOnError!DeviceDataFrame {
    var owned_value = value_column;
    var value_moved = false;
    errdefer if (!value_moved) owned_value.deinit();
    if (owned_value.len() != representative_rows.len) return error.LengthMismatch;
    if (!owned_value.device().sameDevice(frame.device)) return error.InvalidDevice;

    const total_cols = key_names.len + 1;
    var names = try frame.allocator.alloc([]const u8, total_cols);
    defer frame.allocator.free(names);
    var columns = try frame.allocator.alloc(DeviceColumn, total_cols);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }

    for (key_names) |key_name| {
        names[initialized] = key_name;
        columns[initialized] = try (try frame.column(key_name)).take(representative_rows);
        initialized += 1;
    }
    names[initialized] = output_name;
    columns[initialized] = owned_value;
    value_moved = true;
    initialized += 1;
    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, names, columns, representative_rows.len, frame.device);
}

pub fn groupByNumericOnDispatchValue(
    comptime DeviceDataFrame: type,
    op: DeviceGroupByAggregation,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
    value: DeviceColumn,
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    return switch (value) {
        .i8 => |typed| groupByNumericOnTyped(DeviceDataFrame, op, i8, allocator, frame, key_names, output_name, typed, device_value),
        .i16 => |typed| groupByNumericOnTyped(DeviceDataFrame, op, i16, allocator, frame, key_names, output_name, typed, device_value),
        .i32 => |typed| groupByNumericOnTyped(DeviceDataFrame, op, i32, allocator, frame, key_names, output_name, typed, device_value),
        .i64 => |typed| groupByNumericOnTyped(DeviceDataFrame, op, i64, allocator, frame, key_names, output_name, typed, device_value),
        .u8 => |typed| groupByNumericOnTyped(DeviceDataFrame, op, u8, allocator, frame, key_names, output_name, typed, device_value),
        .u16 => |typed| groupByNumericOnTyped(DeviceDataFrame, op, u16, allocator, frame, key_names, output_name, typed, device_value),
        .u32 => |typed| groupByNumericOnTyped(DeviceDataFrame, op, u32, allocator, frame, key_names, output_name, typed, device_value),
        .u64 => |typed| groupByNumericOnTyped(DeviceDataFrame, op, u64, allocator, frame, key_names, output_name, typed, device_value),
        .usize => |typed| groupByNumericOnTyped(DeviceDataFrame, op, usize, allocator, frame, key_names, output_name, typed, device_value),
        .isize => |typed| groupByNumericOnTyped(DeviceDataFrame, op, isize, allocator, frame, key_names, output_name, typed, device_value),
        .f16 => |typed| groupByNumericOnTyped(DeviceDataFrame, op, f16, allocator, frame, key_names, output_name, typed, device_value),
        .f32 => |typed| groupByNumericOnTyped(DeviceDataFrame, op, f32, allocator, frame, key_names, output_name, typed, device_value),
        .f64 => |typed| groupByNumericOnTyped(DeviceDataFrame, op, f64, allocator, frame, key_names, output_name, typed, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn groupByNumericOnTyped(
    comptime DeviceDataFrame: type,
    op: DeviceGroupByAggregation,
    comptime V: type,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
    value: DeviceTypedColumn(V),
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    if (frame.rows != value.len()) return error.LengthMismatch;
    const values = try value.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_value_validity = try validityValues(value, allocator);
    defer if (maybe_value_validity) |validity| allocator.free(validity);

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(allocator);
    var aggregates: std.ArrayList(V) = .empty;
    defer aggregates.deinit(allocator);

    for (values, 0..) |value_item, row| {
        if (maybe_value_validity) |validity| {
            if (!validity[row]) continue;
        }
        if (!try rowHasValidKeys(allocator, frame, key_names, row)) continue;
        const maybe_group_index = try findMultiKeyGroupIndex(allocator, frame, key_names, representative_rows.items, row);
        if (maybe_group_index == null) {
            try representative_rows.append(allocator, row);
            try aggregates.append(allocator, value_item);
            continue;
        }
        const group_index = maybe_group_index.?;
        switch (op) {
            .sum => aggregates.items[group_index] += value_item,
            .prod => aggregates.items[group_index] *= value_item,
            .min => {
                if (compareSortValues(V, value_item, aggregates.items[group_index]) < 0) aggregates.items[group_index] = value_item;
            },
            .max => {
                if (compareSortValues(V, value_item, aggregates.items[group_index]) > 0) aggregates.items[group_index] = value_item;
            },
        }
    }

    const aggregate_column = try DeviceColumn.fromSlice(V, allocator, aggregates.items, device_value);
    return initMultiKeyAggregatedDataFrame(DeviceDataFrame, frame, key_names, representative_rows.items, output_name, aggregate_column);
}

pub fn groupByMeanOnDispatchValue(
    comptime DeviceDataFrame: type,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
    value: DeviceColumn,
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    return switch (value) {
        .i8 => |typed| groupByMeanOnTyped(DeviceDataFrame, i8, allocator, frame, key_names, output_name, typed, device_value),
        .i16 => |typed| groupByMeanOnTyped(DeviceDataFrame, i16, allocator, frame, key_names, output_name, typed, device_value),
        .i32 => |typed| groupByMeanOnTyped(DeviceDataFrame, i32, allocator, frame, key_names, output_name, typed, device_value),
        .i64 => |typed| groupByMeanOnTyped(DeviceDataFrame, i64, allocator, frame, key_names, output_name, typed, device_value),
        .u8 => |typed| groupByMeanOnTyped(DeviceDataFrame, u8, allocator, frame, key_names, output_name, typed, device_value),
        .u16 => |typed| groupByMeanOnTyped(DeviceDataFrame, u16, allocator, frame, key_names, output_name, typed, device_value),
        .u32 => |typed| groupByMeanOnTyped(DeviceDataFrame, u32, allocator, frame, key_names, output_name, typed, device_value),
        .u64 => |typed| groupByMeanOnTyped(DeviceDataFrame, u64, allocator, frame, key_names, output_name, typed, device_value),
        .usize => |typed| groupByMeanOnTyped(DeviceDataFrame, usize, allocator, frame, key_names, output_name, typed, device_value),
        .isize => |typed| groupByMeanOnTyped(DeviceDataFrame, isize, allocator, frame, key_names, output_name, typed, device_value),
        .f16 => |typed| groupByMeanOnTyped(DeviceDataFrame, f16, allocator, frame, key_names, output_name, typed, device_value),
        .f32 => |typed| groupByMeanOnTyped(DeviceDataFrame, f32, allocator, frame, key_names, output_name, typed, device_value),
        .f64 => |typed| groupByMeanOnTyped(DeviceDataFrame, f64, allocator, frame, key_names, output_name, typed, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn groupByMeanOnTyped(
    comptime DeviceDataFrame: type,
    comptime V: type,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
    value: DeviceTypedColumn(V),
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    if (frame.rows != value.len()) return error.LengthMismatch;
    const values = try value.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_value_validity = try validityValues(value, allocator);
    defer if (maybe_value_validity) |validity| allocator.free(validity);

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(allocator);
    var sums: std.ArrayList(f64) = .empty;
    defer sums.deinit(allocator);
    var counts: std.ArrayList(i64) = .empty;
    defer counts.deinit(allocator);

    for (values, 0..) |value_item, row| {
        if (maybe_value_validity) |validity| {
            if (!validity[row]) continue;
        }
        if (!try rowHasValidKeys(allocator, frame, key_names, row)) continue;
        const group_index = (try findMultiKeyGroupIndex(allocator, frame, key_names, representative_rows.items, row)) orelse blk: {
            try representative_rows.append(allocator, row);
            try sums.append(allocator, 0);
            try counts.append(allocator, 0);
            break :blk representative_rows.items.len - 1;
        };
        sums.items[group_index] += castToF64(V, value_item);
        counts.items[group_index] += 1;
    }

    const means = try allocator.alloc(f64, sums.items.len);
    defer allocator.free(means);
    for (sums.items, counts.items, means) |sum_value, count, *slot| {
        slot.* = sum_value / @as(f64, @floatFromInt(count));
    }

    const mean_column = try DeviceColumn.fromSlice(f64, allocator, means, device_value);
    return initMultiKeyAggregatedDataFrame(DeviceDataFrame, frame, key_names, representative_rows.items, output_name, mean_column);
}

pub fn groupByProfileOnDispatchValue(
    comptime DeviceDataFrame: type,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_prefix: []const u8,
    value: DeviceColumn,
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    return switch (value) {
        .i8 => |typed| groupByProfileOnTyped(DeviceDataFrame, i8, allocator, frame, key_names, output_prefix, typed, device_value),
        .i16 => |typed| groupByProfileOnTyped(DeviceDataFrame, i16, allocator, frame, key_names, output_prefix, typed, device_value),
        .i32 => |typed| groupByProfileOnTyped(DeviceDataFrame, i32, allocator, frame, key_names, output_prefix, typed, device_value),
        .i64 => |typed| groupByProfileOnTyped(DeviceDataFrame, i64, allocator, frame, key_names, output_prefix, typed, device_value),
        .u8 => |typed| groupByProfileOnTyped(DeviceDataFrame, u8, allocator, frame, key_names, output_prefix, typed, device_value),
        .u16 => |typed| groupByProfileOnTyped(DeviceDataFrame, u16, allocator, frame, key_names, output_prefix, typed, device_value),
        .u32 => |typed| groupByProfileOnTyped(DeviceDataFrame, u32, allocator, frame, key_names, output_prefix, typed, device_value),
        .u64 => |typed| groupByProfileOnTyped(DeviceDataFrame, u64, allocator, frame, key_names, output_prefix, typed, device_value),
        .usize => |typed| groupByProfileOnTyped(DeviceDataFrame, usize, allocator, frame, key_names, output_prefix, typed, device_value),
        .isize => |typed| groupByProfileOnTyped(DeviceDataFrame, isize, allocator, frame, key_names, output_prefix, typed, device_value),
        .f16 => |typed| groupByProfileOnTyped(DeviceDataFrame, f16, allocator, frame, key_names, output_prefix, typed, device_value),
        .f32 => |typed| groupByProfileOnTyped(DeviceDataFrame, f32, allocator, frame, key_names, output_prefix, typed, device_value),
        .f64 => |typed| groupByProfileOnTyped(DeviceDataFrame, f64, allocator, frame, key_names, output_prefix, typed, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn groupByProfileOnTyped(
    comptime DeviceDataFrame: type,
    comptime V: type,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_prefix: []const u8,
    value: DeviceTypedColumn(V),
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    if (frame.rows != value.len()) return error.LengthMismatch;
    const values = try value.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_value_validity = try validityValues(value, allocator);
    defer if (maybe_value_validity) |validity| allocator.free(validity);

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(allocator);
    var profiles: std.ArrayList(MomentProfile) = .empty;
    defer profiles.deinit(allocator);

    for (values, 0..) |value_item, row| {
        if (maybe_value_validity) |validity| {
            if (!validity[row]) continue;
        }
        if (!try rowHasValidKeys(allocator, frame, key_names, row)) continue;
        const maybe_group_index = try findMultiKeyGroupIndex(allocator, frame, key_names, representative_rows.items, row);
        const group_index = maybe_group_index orelse blk: {
            try representative_rows.append(allocator, row);
            try profiles.append(allocator, .{});
            break :blk representative_rows.items.len - 1;
        };
        profiles.items[group_index].update(castToF64(V, value_item));
    }

    var metrics = try group_profile_mod.materializeMetrics(allocator, profiles.items);
    defer metrics.deinit();
    var key_columns = try allocator.alloc(DeviceColumn, key_names.len);
    var initialized: usize = 0;
    defer {
        for (key_columns[0..initialized]) |*col| col.deinit();
        allocator.free(key_columns);
    }
    for (key_names, key_columns) |key_name, *slot| {
        slot.* = try (try frame.column(key_name)).take(representative_rows.items);
        initialized += 1;
    }

    return group_profile_mod.initProfileDataFrame(DeviceDataFrame, allocator, key_names, output_prefix, key_columns, metrics, device_value);
}

pub fn groupByStatsOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_prefix: []const u8,
) GroupByOnError!DeviceDataFrame {
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);
    const value = try frame.column(value_name);
    return groupByStatsOnDispatchValue(DeviceDataFrame, frame.allocator, frame, key_names, output_prefix, value.*, frame.device);
}

pub fn groupByProfileOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_prefix: []const u8,
) GroupByOnError!DeviceDataFrame {
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);
    const value = try frame.column(value_name);
    return groupByProfileOnDispatchValue(DeviceDataFrame, frame.allocator, frame, key_names, output_prefix, value.*, frame.device);
}
