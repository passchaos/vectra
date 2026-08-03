const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const names_mod = @import("dataframe_names.zig");
const dataframe_device_column_mod = @import("dataframe/device_column.zig");
const metrics_mod = @import("dataframe_crossover_metrics.zig");
const columns_mod = @import("dataframe_crossover_columns.zig");
const numeric_mod = @import("dataframe_numeric.zig");
const options_mod = @import("dataframe_options.zig");
const validity_mod = @import("dataframe_validity.zig");

const DeviceColumn = dataframe_device_column_mod.DeviceColumn;
const DeviceCrossoverOptions = options_mod.DeviceCrossoverOptions;
const DeviceRollingOptions = options_mod.DeviceRollingOptions;
const DeviceExpandingOptions = options_mod.DeviceExpandingOptions;

pub const CrossoverMetrics = metrics_mod.CrossoverMetrics;
pub const CrossoverSummaryMetrics = metrics_mod.CrossoverSummaryMetrics;
pub const CrossoverProfileColumnCount = metrics_mod.CrossoverProfileColumnCount;
pub const RollingCrossoverProfileColumnCount = metrics_mod.RollingCrossoverProfileColumnCount;
pub const ExpandingCrossoverProfileColumnCount = metrics_mod.ExpandingCrossoverProfileColumnCount;
pub const crossoverProfileOutputNames = metrics_mod.crossoverProfileOutputNames;
pub const rollingCrossoverProfileOutputNames = metrics_mod.rollingCrossoverProfileOutputNames;
pub const expandingCrossoverProfileOutputNames = metrics_mod.expandingCrossoverProfileOutputNames;
pub const crossoverProfile = metrics_mod.crossoverProfile;
pub const rollingCrossoverProfile = metrics_mod.rollingCrossoverProfile;
pub const expandingCrossoverProfile = metrics_mod.expandingCrossoverProfile;

pub const crossoverProfileColumnsByValue = columns_mod.crossoverProfileColumnsByValue;
pub const rollingCrossoverProfileColumnsByValue = columns_mod.rollingCrossoverProfileColumnsByValue;
pub const expandingCrossoverProfileColumnsByValue = columns_mod.expandingCrossoverProfileColumnsByValue;

const CrossoverFrameError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    TypeUnsupported,
    UnsupportedType,
    InvalidCsv,
    EmptyDataFrame,
    InvalidDevice,
};

fn appendCrossoverColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    source_names: []const []const u8,
    cross_columns: anytype,
) CrossoverFrameError!DeviceDataFrame {
    const DeviceColumnType = std.meta.Elem(@TypeOf(frame.columns));
    var columns = try frame.allocator.alloc(DeviceColumnType, frame.columns.len + cross_columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }
    for (frame.columns, 0..) |col, i| {
        columns[i] = try col.clone();
        initialized += 1;
    }
    for (&cross_columns) |*cross_col| {
        columns[initialized] = cross_col.*;
        initialized += 1;
    }
    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, source_names, columns, frame.rows, frame.device);
}

fn crossoverFrameFromColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    output_prefix: []const u8,
    cross_columns_value: anytype,
    comptime namesFn: anytype,
) CrossoverFrameError!DeviceDataFrame {
    var cross_columns = cross_columns_value;
    var cross_columns_transferred: usize = 0;
    errdefer {
        for (cross_columns[cross_columns_transferred..]) |*col| col.deinit();
    }

    const source_names = try frame.allocator.alloc([]const u8, frame.columns.len + cross_columns.len);
    defer frame.allocator.free(source_names);
    for (frame.names, 0..) |source_name, i| source_names[i] = source_name;

    var cross_names = try namesFn(frame.allocator, output_prefix);
    defer names_mod.freeOwnedNameItems(frame.allocator, cross_names[0..]);
    for (cross_names, 0..) |cross_name, i| source_names[frame.columns.len + i] = cross_name;

    const out = try appendCrossoverColumns(DeviceDataFrame, frame, source_names, cross_columns);
    cross_columns_transferred = cross_columns.len;
    return out;
}

pub fn crossoverProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    lhs_name: []const u8,
    rhs_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceCrossoverOptions,
) CrossoverFrameError!DeviceDataFrame {
    const lhs = try frame.column(lhs_name);
    const rhs = try frame.column(rhs_name);
    if (lhs.dtype() != rhs.dtype()) return error.TypeMismatch;
    const cross_columns = try crossoverProfileColumnsByValue(frame.allocator, lhs.*, rhs.*, options_value, frame.device, frame.rows);
    return crossoverFrameFromColumns(DeviceDataFrame, frame, output_prefix, cross_columns, crossoverProfileOutputNames);
}

pub fn rollingCrossoverProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    lhs_name: []const u8,
    rhs_name: []const u8,
    output_prefix: []const u8,
    cross_options: DeviceCrossoverOptions,
    options_value: DeviceRollingOptions,
) CrossoverFrameError!DeviceDataFrame {
    const lhs = try frame.column(lhs_name);
    const rhs = try frame.column(rhs_name);
    if (lhs.dtype() != rhs.dtype()) return error.TypeMismatch;
    const cross_columns = try rollingCrossoverProfileColumnsByValue(frame.allocator, lhs.*, rhs.*, cross_options, options_value, frame.device, frame.rows);
    return crossoverFrameFromColumns(DeviceDataFrame, frame, output_prefix, cross_columns, rollingCrossoverProfileOutputNames);
}

pub fn expandingCrossoverProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    lhs_name: []const u8,
    rhs_name: []const u8,
    output_prefix: []const u8,
    cross_options: DeviceCrossoverOptions,
    options_value: DeviceExpandingOptions,
) CrossoverFrameError!DeviceDataFrame {
    const lhs = try frame.column(lhs_name);
    const rhs = try frame.column(rhs_name);
    if (lhs.dtype() != rhs.dtype()) return error.TypeMismatch;
    const cross_columns = try expandingCrossoverProfileColumnsByValue(frame.allocator, lhs.*, rhs.*, cross_options, options_value, frame.device, frame.rows);
    return crossoverFrameFromColumns(DeviceDataFrame, frame, output_prefix, cross_columns, expandingCrossoverProfileOutputNames);
}
