const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const dataframe_device_column_mod = @import("dataframe_device_column.zig");
const group_basic_mod = @import("dataframe_group_basic.zig");
const metrics_mod = @import("dataframe_group_metrics.zig");
const numeric_mod = @import("dataframe_numeric.zig");
const names_mod = @import("dataframe_names.zig");
const options_mod = @import("dataframe_options.zig");
const validity_mod = @import("dataframe_validity.zig");

const DeviceColumn = dataframe_device_column_mod.DeviceColumn;
const DeviceTypedColumn = dataframe_device_column_mod.DeviceTypedColumn;
const DeviceGroupByAggregation = options_mod.DeviceGroupByAggregation;
const findGroupIndex = numeric_mod.findGroupIndex;
const compareSortValues = numeric_mod.compareSortValues;
const castToF64 = numeric_mod.castToF64;
const validityValues = validity_mod.validityValues;

const GroupByMethodError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    TypeUnsupported,
    UnsupportedType,
    InvalidCsv,
    EmptyDataFrame,
    InvalidDevice,
};

pub const MomentProfile = metrics_mod.MomentProfile;
pub const MetricSlices = metrics_mod.MetricSlices;
pub const materializeMetrics = metrics_mod.materializeMetrics;
pub const initProfileDataFrame = metrics_mod.initProfileDataFrame;

pub const groupByCountTyped = group_basic_mod.groupByCountTyped;
pub const initAggregatedDataFrame = group_basic_mod.initAggregatedDataFrame;
pub const groupByNumericDispatchKey = group_basic_mod.groupByNumericDispatchKey;
pub const groupByMeanDispatchKey = group_basic_mod.groupByMeanDispatchKey;

pub fn groupByStatsDispatchKey(
    comptime DeviceDataFrame: type,
    allocator: std.mem.Allocator,
    key_name: []const u8,
    output_prefix: []const u8,
    key: DeviceColumn,
    value: DeviceColumn,
    device_value: array_mod.Device,
) (array_mod.ArrayError || std.mem.Allocator.Error || error{ LengthMismatch, TypeMismatch, TypeUnsupported, InvalidDevice })!DeviceDataFrame {
    return switch (key) {
        .bool => |typed| groupByStatsDispatchValue(DeviceDataFrame, bool, allocator, key_name, output_prefix, typed, value, device_value),
        .i8 => |typed| groupByStatsDispatchValue(DeviceDataFrame, i8, allocator, key_name, output_prefix, typed, value, device_value),
        .i16 => |typed| groupByStatsDispatchValue(DeviceDataFrame, i16, allocator, key_name, output_prefix, typed, value, device_value),
        .i32 => |typed| groupByStatsDispatchValue(DeviceDataFrame, i32, allocator, key_name, output_prefix, typed, value, device_value),
        .i64 => |typed| groupByStatsDispatchValue(DeviceDataFrame, i64, allocator, key_name, output_prefix, typed, value, device_value),
        .u8 => |typed| groupByStatsDispatchValue(DeviceDataFrame, u8, allocator, key_name, output_prefix, typed, value, device_value),
        .u16 => |typed| groupByStatsDispatchValue(DeviceDataFrame, u16, allocator, key_name, output_prefix, typed, value, device_value),
        .u32 => |typed| groupByStatsDispatchValue(DeviceDataFrame, u32, allocator, key_name, output_prefix, typed, value, device_value),
        .u64 => |typed| groupByStatsDispatchValue(DeviceDataFrame, u64, allocator, key_name, output_prefix, typed, value, device_value),
        .usize => |typed| groupByStatsDispatchValue(DeviceDataFrame, usize, allocator, key_name, output_prefix, typed, value, device_value),
        .isize => |typed| groupByStatsDispatchValue(DeviceDataFrame, isize, allocator, key_name, output_prefix, typed, value, device_value),
        .f16 => |typed| groupByStatsDispatchValue(DeviceDataFrame, f16, allocator, key_name, output_prefix, typed, value, device_value),
        .f32 => |typed| groupByStatsDispatchValue(DeviceDataFrame, f32, allocator, key_name, output_prefix, typed, value, device_value),
        .f64 => |typed| groupByStatsDispatchValue(DeviceDataFrame, f64, allocator, key_name, output_prefix, typed, value, device_value),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn groupByStatsDispatchValue(
    comptime DeviceDataFrame: type,
    comptime K: type,
    allocator: std.mem.Allocator,
    key_name: []const u8,
    output_prefix: []const u8,
    key: DeviceTypedColumn(K),
    value: DeviceColumn,
    device_value: array_mod.Device,
) (array_mod.ArrayError || std.mem.Allocator.Error || error{ LengthMismatch, TypeMismatch, TypeUnsupported, InvalidDevice })!DeviceDataFrame {
    return switch (value) {
        .i8 => |typed| groupByStatsTyped(DeviceDataFrame, K, i8, allocator, key_name, output_prefix, key, typed, device_value),
        .i16 => |typed| groupByStatsTyped(DeviceDataFrame, K, i16, allocator, key_name, output_prefix, key, typed, device_value),
        .i32 => |typed| groupByStatsTyped(DeviceDataFrame, K, i32, allocator, key_name, output_prefix, key, typed, device_value),
        .i64 => |typed| groupByStatsTyped(DeviceDataFrame, K, i64, allocator, key_name, output_prefix, key, typed, device_value),
        .u8 => |typed| groupByStatsTyped(DeviceDataFrame, K, u8, allocator, key_name, output_prefix, key, typed, device_value),
        .u16 => |typed| groupByStatsTyped(DeviceDataFrame, K, u16, allocator, key_name, output_prefix, key, typed, device_value),
        .u32 => |typed| groupByStatsTyped(DeviceDataFrame, K, u32, allocator, key_name, output_prefix, key, typed, device_value),
        .u64 => |typed| groupByStatsTyped(DeviceDataFrame, K, u64, allocator, key_name, output_prefix, key, typed, device_value),
        .usize => |typed| groupByStatsTyped(DeviceDataFrame, K, usize, allocator, key_name, output_prefix, key, typed, device_value),
        .isize => |typed| groupByStatsTyped(DeviceDataFrame, K, isize, allocator, key_name, output_prefix, key, typed, device_value),
        .f16 => |typed| groupByStatsTyped(DeviceDataFrame, K, f16, allocator, key_name, output_prefix, key, typed, device_value),
        .f32 => |typed| groupByStatsTyped(DeviceDataFrame, K, f32, allocator, key_name, output_prefix, key, typed, device_value),
        .f64 => |typed| groupByStatsTyped(DeviceDataFrame, K, f64, allocator, key_name, output_prefix, key, typed, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn groupByStatsTyped(
    comptime DeviceDataFrame: type,
    comptime K: type,
    comptime V: type,
    allocator: std.mem.Allocator,
    key_name: []const u8,
    output_prefix: []const u8,
    key: DeviceTypedColumn(K),
    value: DeviceTypedColumn(V),
    device_value: array_mod.Device,
) (array_mod.ArrayError || std.mem.Allocator.Error || error{ LengthMismatch, TypeMismatch, TypeUnsupported, InvalidDevice })!DeviceDataFrame {
    if (key.len() != value.len()) return error.LengthMismatch;
    if (!key.device().sameDevice(value.device())) return error.InvalidDevice;

    const keys = try key.values.toOwnedSlice(allocator);
    defer allocator.free(keys);
    const values = try value.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_key_validity = try validityValues(key, allocator);
    defer if (maybe_key_validity) |validity| allocator.free(validity);
    const maybe_value_validity = try validityValues(value, allocator);
    defer if (maybe_value_validity) |validity| allocator.free(validity);

    var unique_keys: std.ArrayList(K) = .empty;
    defer unique_keys.deinit(allocator);
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

    for (keys, values, 0..) |key_value, value_item, row| {
        if (maybe_key_validity) |validity| {
            if (!validity[row]) continue;
        }
        if (maybe_value_validity) |validity| {
            if (!validity[row]) continue;
        }
        const maybe_group_index = findGroupIndex(K, unique_keys.items, key_value);
        if (maybe_group_index == null) {
            try unique_keys.append(allocator, key_value);
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

    var key_col = try DeviceColumn.fromSlice(K, allocator, unique_keys.items, device_value);
    errdefer key_col.deinit();
    var count_col = try DeviceColumn.fromSlice(i64, allocator, counts.items, device_value);
    errdefer count_col.deinit();
    var sum_col = try DeviceColumn.fromSlice(V, allocator, sums.items, device_value);
    errdefer sum_col.deinit();
    var min_col = try DeviceColumn.fromSlice(V, allocator, mins.items, device_value);
    errdefer min_col.deinit();
    var max_col = try DeviceColumn.fromSlice(V, allocator, maxes.items, device_value);
    errdefer max_col.deinit();
    var mean_col = try DeviceColumn.fromSlice(f64, allocator, means, device_value);
    errdefer mean_col.deinit();

    const names = try names_mod.statsOutputNames(allocator, key_name, output_prefix);
    defer names_mod.freeStatsOutputNames(allocator, names);
    const columns = try allocator.alloc(DeviceColumn, 6);
    errdefer allocator.free(columns);
    columns[0] = key_col;
    columns[1] = count_col;
    columns[2] = sum_col;
    columns[3] = min_col;
    columns[4] = max_col;
    columns[5] = mean_col;
    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, allocator, names, columns, unique_keys.items.len, device_value);
}

pub fn groupByProfileDispatchKey(
    comptime DeviceDataFrame: type,
    allocator: std.mem.Allocator,
    key_name: []const u8,
    output_prefix: []const u8,
    key: DeviceColumn,
    value: DeviceColumn,
    device_value: array_mod.Device,
) (array_mod.ArrayError || std.mem.Allocator.Error || error{ LengthMismatch, TypeMismatch, TypeUnsupported, InvalidDevice })!DeviceDataFrame {
    return switch (key) {
        .bool => |typed| groupByProfileDispatchValue(DeviceDataFrame, bool, allocator, key_name, output_prefix, typed, value, device_value),
        .i8 => |typed| groupByProfileDispatchValue(DeviceDataFrame, i8, allocator, key_name, output_prefix, typed, value, device_value),
        .i16 => |typed| groupByProfileDispatchValue(DeviceDataFrame, i16, allocator, key_name, output_prefix, typed, value, device_value),
        .i32 => |typed| groupByProfileDispatchValue(DeviceDataFrame, i32, allocator, key_name, output_prefix, typed, value, device_value),
        .i64 => |typed| groupByProfileDispatchValue(DeviceDataFrame, i64, allocator, key_name, output_prefix, typed, value, device_value),
        .u8 => |typed| groupByProfileDispatchValue(DeviceDataFrame, u8, allocator, key_name, output_prefix, typed, value, device_value),
        .u16 => |typed| groupByProfileDispatchValue(DeviceDataFrame, u16, allocator, key_name, output_prefix, typed, value, device_value),
        .u32 => |typed| groupByProfileDispatchValue(DeviceDataFrame, u32, allocator, key_name, output_prefix, typed, value, device_value),
        .u64 => |typed| groupByProfileDispatchValue(DeviceDataFrame, u64, allocator, key_name, output_prefix, typed, value, device_value),
        .usize => |typed| groupByProfileDispatchValue(DeviceDataFrame, usize, allocator, key_name, output_prefix, typed, value, device_value),
        .isize => |typed| groupByProfileDispatchValue(DeviceDataFrame, isize, allocator, key_name, output_prefix, typed, value, device_value),
        .f16 => |typed| groupByProfileDispatchValue(DeviceDataFrame, f16, allocator, key_name, output_prefix, typed, value, device_value),
        .f32 => |typed| groupByProfileDispatchValue(DeviceDataFrame, f32, allocator, key_name, output_prefix, typed, value, device_value),
        .f64 => |typed| groupByProfileDispatchValue(DeviceDataFrame, f64, allocator, key_name, output_prefix, typed, value, device_value),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn groupByProfileDispatchValue(
    comptime DeviceDataFrame: type,
    comptime K: type,
    allocator: std.mem.Allocator,
    key_name: []const u8,
    output_prefix: []const u8,
    key: DeviceTypedColumn(K),
    value: DeviceColumn,
    device_value: array_mod.Device,
) (array_mod.ArrayError || std.mem.Allocator.Error || error{ LengthMismatch, TypeMismatch, TypeUnsupported, InvalidDevice })!DeviceDataFrame {
    return switch (value) {
        .i8 => |typed| groupByProfileTyped(DeviceDataFrame, K, i8, allocator, key_name, output_prefix, key, typed, device_value),
        .i16 => |typed| groupByProfileTyped(DeviceDataFrame, K, i16, allocator, key_name, output_prefix, key, typed, device_value),
        .i32 => |typed| groupByProfileTyped(DeviceDataFrame, K, i32, allocator, key_name, output_prefix, key, typed, device_value),
        .i64 => |typed| groupByProfileTyped(DeviceDataFrame, K, i64, allocator, key_name, output_prefix, key, typed, device_value),
        .u8 => |typed| groupByProfileTyped(DeviceDataFrame, K, u8, allocator, key_name, output_prefix, key, typed, device_value),
        .u16 => |typed| groupByProfileTyped(DeviceDataFrame, K, u16, allocator, key_name, output_prefix, key, typed, device_value),
        .u32 => |typed| groupByProfileTyped(DeviceDataFrame, K, u32, allocator, key_name, output_prefix, key, typed, device_value),
        .u64 => |typed| groupByProfileTyped(DeviceDataFrame, K, u64, allocator, key_name, output_prefix, key, typed, device_value),
        .usize => |typed| groupByProfileTyped(DeviceDataFrame, K, usize, allocator, key_name, output_prefix, key, typed, device_value),
        .isize => |typed| groupByProfileTyped(DeviceDataFrame, K, isize, allocator, key_name, output_prefix, key, typed, device_value),
        .f16 => |typed| groupByProfileTyped(DeviceDataFrame, K, f16, allocator, key_name, output_prefix, key, typed, device_value),
        .f32 => |typed| groupByProfileTyped(DeviceDataFrame, K, f32, allocator, key_name, output_prefix, key, typed, device_value),
        .f64 => |typed| groupByProfileTyped(DeviceDataFrame, K, f64, allocator, key_name, output_prefix, key, typed, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn groupByProfileTyped(
    comptime DeviceDataFrame: type,
    comptime K: type,
    comptime V: type,
    allocator: std.mem.Allocator,
    key_name: []const u8,
    output_prefix: []const u8,
    key: DeviceTypedColumn(K),
    value: DeviceTypedColumn(V),
    device_value: array_mod.Device,
) (array_mod.ArrayError || std.mem.Allocator.Error || error{ LengthMismatch, TypeMismatch, TypeUnsupported, InvalidDevice })!DeviceDataFrame {
    if (key.len() != value.len()) return error.LengthMismatch;
    if (!key.device().sameDevice(value.device())) return error.InvalidDevice;

    const keys = try key.values.toOwnedSlice(allocator);
    defer allocator.free(keys);
    const values = try value.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_key_validity = try validityValues(key, allocator);
    defer if (maybe_key_validity) |validity| allocator.free(validity);
    const maybe_value_validity = try validityValues(value, allocator);
    defer if (maybe_value_validity) |validity| allocator.free(validity);

    var unique_keys: std.ArrayList(K) = .empty;
    defer unique_keys.deinit(allocator);
    var profiles: std.ArrayList(MomentProfile) = .empty;
    defer profiles.deinit(allocator);

    // Keep all moment-derived metrics in one pass over each group.  Besides
    // being cheaper than issuing many independent group-bys, this preserves one
    // API seam for a future Axiom grouped-moment kernel on CPU/CUDA/MPS.
    for (keys, values, 0..) |key_value, value_item, row| {
        if (maybe_key_validity) |validity| {
            if (!validity[row]) continue;
        }
        if (maybe_value_validity) |validity| {
            if (!validity[row]) continue;
        }
        const group_index = findGroupIndex(K, unique_keys.items, key_value) orelse blk: {
            try unique_keys.append(allocator, key_value);
            try profiles.append(allocator, .{});
            break :blk unique_keys.items.len - 1;
        };
        profiles.items[group_index].update(castToF64(V, value_item));
    }

    var metrics = try materializeMetrics(allocator, profiles.items);
    defer metrics.deinit();
    var key_col = try DeviceColumn.fromSlice(K, allocator, unique_keys.items, device_value);
    defer key_col.deinit();
    return initProfileDataFrame(DeviceDataFrame, allocator, &.{key_name}, output_prefix, &.{key_col}, metrics, device_value);
}

pub fn groupByCount(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_name: []const u8,
    output_name: []const u8,
) GroupByMethodError!DeviceDataFrame {
    const key = try frame.column(key_name);
    return switch (key.*) {
        .bool => |typed| groupByCountTyped(DeviceDataFrame, bool, frame.allocator, key_name, output_name, typed, frame.device),
        .i8 => |typed| groupByCountTyped(DeviceDataFrame, i8, frame.allocator, key_name, output_name, typed, frame.device),
        .i16 => |typed| groupByCountTyped(DeviceDataFrame, i16, frame.allocator, key_name, output_name, typed, frame.device),
        .i32 => |typed| groupByCountTyped(DeviceDataFrame, i32, frame.allocator, key_name, output_name, typed, frame.device),
        .i64 => |typed| groupByCountTyped(DeviceDataFrame, i64, frame.allocator, key_name, output_name, typed, frame.device),
        .u8 => |typed| groupByCountTyped(DeviceDataFrame, u8, frame.allocator, key_name, output_name, typed, frame.device),
        .u16 => |typed| groupByCountTyped(DeviceDataFrame, u16, frame.allocator, key_name, output_name, typed, frame.device),
        .u32 => |typed| groupByCountTyped(DeviceDataFrame, u32, frame.allocator, key_name, output_name, typed, frame.device),
        .u64 => |typed| groupByCountTyped(DeviceDataFrame, u64, frame.allocator, key_name, output_name, typed, frame.device),
        .usize => |typed| groupByCountTyped(DeviceDataFrame, usize, frame.allocator, key_name, output_name, typed, frame.device),
        .isize => |typed| groupByCountTyped(DeviceDataFrame, isize, frame.allocator, key_name, output_name, typed, frame.device),
        .f16 => |typed| groupByCountTyped(DeviceDataFrame, f16, frame.allocator, key_name, output_name, typed, frame.device),
        .f32 => |typed| groupByCountTyped(DeviceDataFrame, f32, frame.allocator, key_name, output_name, typed, frame.device),
        .f64 => |typed| groupByCountTyped(DeviceDataFrame, f64, frame.allocator, key_name, output_name, typed, frame.device),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

pub fn groupByNumeric(
    comptime DeviceDataFrame: type,
    op: DeviceGroupByAggregation,
    frame: DeviceDataFrame,
    key_name: []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByMethodError!DeviceDataFrame {
    const key = try frame.column(key_name);
    const value = try frame.column(value_name);
    return groupByNumericDispatchKey(DeviceDataFrame, op, frame.allocator, key_name, output_name, key.*, value.*, frame.device);
}

pub fn groupByMean(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_name: []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByMethodError!DeviceDataFrame {
    const key = try frame.column(key_name);
    const value = try frame.column(value_name);
    return groupByMeanDispatchKey(DeviceDataFrame, frame.allocator, key_name, output_name, key.*, value.*, frame.device);
}

pub fn groupByStats(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_name: []const u8,
    value_name: []const u8,
    output_prefix: []const u8,
) GroupByMethodError!DeviceDataFrame {
    const key = try frame.column(key_name);
    const value = try frame.column(value_name);
    return groupByStatsDispatchKey(DeviceDataFrame, frame.allocator, key_name, output_prefix, key.*, value.*, frame.device);
}

pub fn groupByProfile(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_name: []const u8,
    value_name: []const u8,
    output_prefix: []const u8,
) GroupByMethodError!DeviceDataFrame {
    const key = try frame.column(key_name);
    const value = try frame.column(value_name);
    return groupByProfileDispatchKey(DeviceDataFrame, frame.allocator, key_name, output_prefix, key.*, value.*, frame.device);
}
