const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const names_mod = @import("dataframe_names.zig");
const dataframe_device_column_mod = @import("dataframe_device_column.zig");
const rank_columns_mod = @import("dataframe_rank_columns.zig");
const rank_metrics_mod = @import("dataframe_rank_metrics.zig");
const options_mod = @import("dataframe_options.zig");

const DeviceColumn = dataframe_device_column_mod.DeviceColumn;
const DeviceSortOptions = options_mod.DeviceSortOptions;
const DeviceRollingRankOptions = options_mod.DeviceRollingRankOptions;
const DeviceExpandingRankOptions = options_mod.DeviceExpandingRankOptions;
const freeOwnedNameItems = names_mod.freeOwnedNameItems;

const RankFrameError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    InvalidCsv,
    EmptyDataFrame,
    UnsupportedType,
    InvalidDevice,
};

pub const RankMetrics = rank_metrics_mod.RankMetrics;
pub const RankProfileColumnCount = rank_metrics_mod.RankProfileColumnCount;
pub const rankProfileOutputNames = rank_metrics_mod.rankProfileOutputNames;
pub const rankProfile = rank_metrics_mod.rankProfile;
pub const RankWindowMetrics = rank_metrics_mod.RankWindowMetrics;
pub const RollingRankProfileColumnCount = rank_metrics_mod.RollingRankProfileColumnCount;
pub const rollingRankProfileOutputNames = rank_metrics_mod.rollingRankProfileOutputNames;
pub const ExpandingRankProfileColumnCount = rank_metrics_mod.ExpandingRankProfileColumnCount;
pub const expandingRankProfileOutputNames = rank_metrics_mod.expandingRankProfileOutputNames;
pub const rollingRankProfile = rank_metrics_mod.rollingRankProfile;
pub const expandingRankProfile = rank_metrics_mod.expandingRankProfile;

pub const rankProfileColumnsByKey = rank_columns_mod.rankProfileColumnsByKey;
pub const rollingRankProfileColumnsByValue = rank_columns_mod.rollingRankProfileColumnsByValue;
pub const expandingRankProfileColumnsByValue = rank_columns_mod.expandingRankProfileColumnsByValue;

pub fn argsortBy(frame: anytype, name: []const u8, options_value: DeviceSortOptions) RankFrameError![]usize {
    const sort_key = try frame.column(name);
    return sort_key.argsort(frame.allocator, options_value);
}

pub fn isSortedBy(frame: anytype, name: []const u8, options_value: DeviceSortOptions) RankFrameError!bool {
    const order = try argsortBy(frame, name, options_value);
    defer frame.allocator.free(order);
    for (order, 0..) |row, expected| {
        if (row != expected) return false;
    }
    return true;
}

pub fn sortBy(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    options_value: DeviceSortOptions,
) RankFrameError!DeviceDataFrame {
    const order = try argsortBy(frame, name, options_value);
    defer frame.allocator.free(order);
    return frame.take(order);
}

pub fn argsortByColumns(frame: anytype, names: []const []const u8, options_values: []const DeviceSortOptions) RankFrameError![]usize {
    if (names.len != options_values.len) return error.LengthMismatch;
    var order = try frame.allocator.alloc(usize, frame.rows);
    errdefer frame.allocator.free(order);
    for (order, 0..) |*slot, i| slot.* = i;

    // Stable sorts from the least-significant key to the most-significant key
    // produce lexicographic ordering while reusing the existing per-column
    // null-placement and dtype comparison rules.
    var key_index = names.len;
    while (key_index > 0) {
        key_index -= 1;
        var reordered = try frame.take(order);
        defer reordered.deinit();
        const key_order = try argsortBy(reordered, names[key_index], options_values[key_index]);
        defer frame.allocator.free(key_order);
        const next_order = try frame.allocator.alloc(usize, frame.rows);
        for (key_order, next_order) |local_index, *slot| slot.* = order[local_index];
        frame.allocator.free(order);
        order = next_order;
    }
    return order;
}

pub fn isSortedByColumns(frame: anytype, names: []const []const u8, options_values: []const DeviceSortOptions) RankFrameError!bool {
    const order = try argsortByColumns(frame, names, options_values);
    defer frame.allocator.free(order);
    for (order, 0..) |row, expected| {
        if (row != expected) return false;
    }
    return true;
}

pub fn sortByColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    names: []const []const u8,
    options_values: []const DeviceSortOptions,
) RankFrameError!DeviceDataFrame {
    const order = try argsortByColumns(frame, names, options_values);
    defer frame.allocator.free(order);
    return frame.take(order);
}

pub fn topKByColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    names: []const []const u8,
    k: usize,
    options_values: []const DeviceSortOptions,
) RankFrameError!DeviceDataFrame {
    var sorted = try sortByColumns(DeviceDataFrame, frame, names, options_values);
    defer sorted.deinit();
    return sorted.head(k);
}

pub fn topKBy(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    k: usize,
    options_value: DeviceSortOptions,
) RankFrameError!DeviceDataFrame {
    var sorted = try sortBy(DeviceDataFrame, frame, name, options_value);
    defer sorted.deinit();
    return sorted.head(k);
}

pub fn rankProfileBy(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceSortOptions,
) RankFrameError!DeviceDataFrame {
    const rank_key = try frame.column(name);
    var rank_columns = try rankProfileColumnsByKey(frame.allocator, rank_key.*, options_value, frame.device, frame.rows);
    var rank_columns_transferred: usize = 0;
    errdefer {
        for (rank_columns[rank_columns_transferred..]) |*col| col.deinit();
    }

    const source_names = try frame.allocator.alloc([]const u8, frame.columns.len + rank_columns.len);
    defer frame.allocator.free(source_names);
    for (frame.names, 0..) |source_name, i| source_names[i] = source_name;

    var rank_names = try rankProfileOutputNames(frame.allocator, output_prefix);
    defer freeOwnedNameItems(frame.allocator, rank_names[0..]);
    for (rank_names, 0..) |rank_name, i| source_names[frame.columns.len + i] = rank_name;

    const DeviceColumnType = std.meta.Elem(@TypeOf(frame.columns));
    var columns = try frame.allocator.alloc(DeviceColumnType, frame.columns.len + rank_columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }
    for (frame.columns, 0..) |col, i| {
        columns[i] = try col.clone();
        initialized += 1;
    }
    for (&rank_columns) |*rank_col| {
        columns[initialized] = rank_col.*;
        initialized += 1;
        rank_columns_transferred += 1;
    }

    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, source_names, columns, frame.rows, frame.device);
}

fn appendRankColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    source_names: []const []const u8,
    rank_columns: anytype,
) RankFrameError!DeviceDataFrame {
    const DeviceColumnType = std.meta.Elem(@TypeOf(frame.columns));
    var columns = try frame.allocator.alloc(DeviceColumnType, frame.columns.len + rank_columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }
    for (frame.columns, 0..) |col, i| {
        columns[i] = try col.clone();
        initialized += 1;
    }
    for (&rank_columns) |*rank_col| {
        columns[initialized] = rank_col.*;
        initialized += 1;
    }
    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, source_names, columns, frame.rows, frame.device);
}

fn rankFrameFromColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    output_prefix: []const u8,
    rank_columns_value: anytype,
    comptime namesFn: anytype,
) RankFrameError!DeviceDataFrame {
    var rank_columns = rank_columns_value;
    var rank_columns_transferred: usize = 0;
    errdefer {
        for (rank_columns[rank_columns_transferred..]) |*col| col.deinit();
    }

    const source_names = try frame.allocator.alloc([]const u8, frame.columns.len + rank_columns.len);
    defer frame.allocator.free(source_names);
    for (frame.names, 0..) |source_name, i| source_names[i] = source_name;

    var rank_names = try namesFn(frame.allocator, output_prefix);
    defer freeOwnedNameItems(frame.allocator, rank_names[0..]);
    for (rank_names, 0..) |rank_name, i| source_names[frame.columns.len + i] = rank_name;

    const out = try appendRankColumns(DeviceDataFrame, frame, source_names, rank_columns);
    rank_columns_transferred = rank_columns.len;
    return out;
}

pub fn rollingRankProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceRollingRankOptions,
) RankFrameError!DeviceDataFrame {
    const rolling_value = try frame.column(name);
    const rolling_columns = try rollingRankProfileColumnsByValue(frame.allocator, rolling_value.*, options_value, frame.device, frame.rows);
    return rankFrameFromColumns(DeviceDataFrame, frame, output_prefix, rolling_columns, rollingRankProfileOutputNames);
}

pub fn expandingRankProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceExpandingRankOptions,
) RankFrameError!DeviceDataFrame {
    const expanding_value = try frame.column(name);
    const expanding_columns = try expandingRankProfileColumnsByValue(frame.allocator, expanding_value.*, options_value, frame.device, frame.rows);
    return rankFrameFromColumns(DeviceDataFrame, frame, output_prefix, expanding_columns, expandingRankProfileOutputNames);
}
