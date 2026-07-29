const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const names_mod = @import("dataframe_names.zig");
const dataframe_device_column_mod = @import("dataframe_device_column.zig");
const columns_mod = @import("dataframe_linear_fit_columns.zig");
const metrics_mod = @import("dataframe_linear_fit_metrics.zig");
const options_mod = @import("dataframe_options.zig");

const DeviceColumn = dataframe_device_column_mod.DeviceColumn;
const DeviceLinearFitOptions = options_mod.DeviceLinearFitOptions;
const DeviceExpandingOptions = options_mod.DeviceExpandingOptions;
const DeviceRollingCorrelationOptions = options_mod.DeviceRollingCorrelationOptions;

pub const LinearFitMetrics = metrics_mod.LinearFitMetrics;
pub const WindowLinearFitMetrics = metrics_mod.WindowLinearFitMetrics;
pub const LinearFitProfileColumnCount = metrics_mod.LinearFitProfileColumnCount;
pub const ExpandingLinearFitProfileColumnCount = metrics_mod.ExpandingLinearFitProfileColumnCount;
pub const RollingLinearFitProfileColumnCount = metrics_mod.RollingLinearFitProfileColumnCount;
pub const linearFitProfileOutputNames = metrics_mod.linearFitProfileOutputNames;
pub const expandingLinearFitProfileOutputNames = metrics_mod.expandingLinearFitProfileOutputNames;
pub const rollingLinearFitProfileOutputNames = metrics_mod.rollingLinearFitProfileOutputNames;
pub const linearFitProfile = metrics_mod.linearFitProfile;
pub const expandingLinearFitProfile = metrics_mod.expandingLinearFitProfile;
pub const rollingLinearFitProfile = metrics_mod.rollingLinearFitProfile;

pub const linearFitProfileColumnsByValue = columns_mod.linearFitProfileColumnsByValue;
pub const expandingLinearFitProfileColumnsByValue = columns_mod.expandingLinearFitProfileColumnsByValue;
pub const rollingLinearFitProfileColumnsByValue = columns_mod.rollingLinearFitProfileColumnsByValue;

const LinearFitFrameError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    TypeUnsupported,
    UnsupportedType,
    InvalidCsv,
    EmptyDataFrame,
    InvalidDevice,
};

fn appendFitColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    source_names: []const []const u8,
    fit_columns: anytype,
) LinearFitFrameError!DeviceDataFrame {
    const DeviceColumnType = std.meta.Elem(@TypeOf(frame.columns));
    var columns = try frame.allocator.alloc(DeviceColumnType, frame.columns.len + fit_columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }
    for (frame.columns, 0..) |col, i| {
        columns[i] = try col.clone();
        initialized += 1;
    }
    for (&fit_columns) |*fit_col| {
        columns[initialized] = fit_col.*;
        initialized += 1;
    }
    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, source_names, columns, frame.rows, frame.device);
}

fn fitFrameFromColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    output_prefix: []const u8,
    fit_columns_value: anytype,
    comptime namesFn: anytype,
) LinearFitFrameError!DeviceDataFrame {
    var fit_columns = fit_columns_value;
    var fit_columns_transferred: usize = 0;
    errdefer {
        for (fit_columns[fit_columns_transferred..]) |*col| col.deinit();
    }

    const source_names = try frame.allocator.alloc([]const u8, frame.columns.len + fit_columns.len);
    defer frame.allocator.free(source_names);
    for (frame.names, 0..) |source_name, i| source_names[i] = source_name;

    var fit_names = try namesFn(frame.allocator, output_prefix);
    defer names_mod.freeOwnedNameItems(frame.allocator, fit_names[0..]);
    for (fit_names, 0..) |fit_name, i| source_names[frame.columns.len + i] = fit_name;

    const out = try appendFitColumns(DeviceDataFrame, frame, source_names, fit_columns);
    fit_columns_transferred = fit_columns.len;
    return out;
}

fn validateFitInputs(frame: anytype, x_name: []const u8, y_name: []const u8) LinearFitFrameError!struct { x: @TypeOf(frame.column(x_name) catch unreachable), y: @TypeOf(frame.column(y_name) catch unreachable) } {
    const x = try frame.column(x_name);
    const y = try frame.column(y_name);
    if (x.dtype() != y.dtype()) return error.TypeMismatch;
    return .{ .x = x, .y = y };
}

pub fn linearFitProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    x_name: []const u8,
    y_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceLinearFitOptions,
) LinearFitFrameError!DeviceDataFrame {
    const inputs = try validateFitInputs(frame, x_name, y_name);
    const fit_columns = try linearFitProfileColumnsByValue(frame.allocator, inputs.x.*, inputs.y.*, options_value, frame.device, frame.rows);
    return fitFrameFromColumns(DeviceDataFrame, frame, output_prefix, fit_columns, linearFitProfileOutputNames);
}

pub fn expandingLinearFitProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    x_name: []const u8,
    y_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceExpandingOptions,
) LinearFitFrameError!DeviceDataFrame {
    const inputs = try validateFitInputs(frame, x_name, y_name);
    const fit_columns = try expandingLinearFitProfileColumnsByValue(frame.allocator, inputs.x.*, inputs.y.*, options_value, frame.device, frame.rows);
    return fitFrameFromColumns(DeviceDataFrame, frame, output_prefix, fit_columns, expandingLinearFitProfileOutputNames);
}

pub fn rollingLinearFitProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    x_name: []const u8,
    y_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceRollingCorrelationOptions,
) LinearFitFrameError!DeviceDataFrame {
    const inputs = try validateFitInputs(frame, x_name, y_name);
    const fit_columns = try rollingLinearFitProfileColumnsByValue(frame.allocator, inputs.x.*, inputs.y.*, options_value, frame.device, frame.rows);
    return fitFrameFromColumns(DeviceDataFrame, frame, output_prefix, fit_columns, rollingLinearFitProfileOutputNames);
}
