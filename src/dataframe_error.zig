const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const names_mod = @import("dataframe_names.zig");
const dataframe_device_column_mod = @import("dataframe/device_column.zig");
const error_columns_mod = @import("dataframe_error_columns.zig");
const error_metrics_mod = @import("dataframe_error_metrics.zig");
const options_mod = @import("dataframe_options.zig");

const DeviceColumn = dataframe_device_column_mod.DeviceColumn;
const DeviceRollingOptions = options_mod.DeviceRollingOptions;
const DeviceExpandingOptions = options_mod.DeviceExpandingOptions;

pub const ErrorMetrics = error_metrics_mod.ErrorMetrics;
pub const ErrorSummaryMetrics = error_metrics_mod.ErrorSummaryMetrics;
pub const ErrorProfileColumnCount = error_metrics_mod.ErrorProfileColumnCount;
pub const RollingErrorProfileColumnCount = error_metrics_mod.RollingErrorProfileColumnCount;
pub const ExpandingErrorProfileColumnCount = error_metrics_mod.ExpandingErrorProfileColumnCount;
pub const errorProfileOutputNames = error_metrics_mod.errorProfileOutputNames;
pub const rollingErrorProfileOutputNames = error_metrics_mod.rollingErrorProfileOutputNames;
pub const expandingErrorProfileOutputNames = error_metrics_mod.expandingErrorProfileOutputNames;
pub const errorProfile = error_metrics_mod.errorProfile;
pub const rollingErrorProfile = error_metrics_mod.rollingErrorProfile;
pub const expandingErrorProfile = error_metrics_mod.expandingErrorProfile;

pub const errorProfileColumnsByValue = error_columns_mod.errorProfileColumnsByValue;
pub const rollingErrorProfileColumnsByValue = error_columns_mod.rollingErrorProfileColumnsByValue;
pub const expandingErrorProfileColumnsByValue = error_columns_mod.expandingErrorProfileColumnsByValue;

const ErrorFrameError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    TypeUnsupported,
    UnsupportedType,
    InvalidCsv,
    EmptyDataFrame,
    InvalidDevice,
};

fn appendErrorColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    source_names: []const []const u8,
    error_columns: anytype,
) ErrorFrameError!DeviceDataFrame {
    const DeviceColumnType = std.meta.Elem(@TypeOf(frame.columns));
    var columns = try frame.allocator.alloc(DeviceColumnType, frame.columns.len + error_columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }
    for (frame.columns, 0..) |col, i| {
        columns[i] = try col.clone();
        initialized += 1;
    }
    for (&error_columns) |*error_col| {
        columns[initialized] = error_col.*;
        initialized += 1;
    }
    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, source_names, columns, frame.rows, frame.device);
}

fn errorFrameFromColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    output_prefix: []const u8,
    error_columns_value: anytype,
    comptime namesFn: anytype,
) ErrorFrameError!DeviceDataFrame {
    var error_columns = error_columns_value;
    var error_columns_transferred: usize = 0;
    errdefer {
        for (error_columns[error_columns_transferred..]) |*col| col.deinit();
    }

    const source_names = try frame.allocator.alloc([]const u8, frame.columns.len + error_columns.len);
    defer frame.allocator.free(source_names);
    for (frame.names, 0..) |source_name, i| source_names[i] = source_name;

    var error_names = try namesFn(frame.allocator, output_prefix);
    defer names_mod.freeOwnedNameItems(frame.allocator, error_names[0..]);
    for (error_names, 0..) |error_name, i| source_names[frame.columns.len + i] = error_name;

    const out = try appendErrorColumns(DeviceDataFrame, frame, source_names, error_columns);
    error_columns_transferred = error_columns.len;
    return out;
}

fn validateErrorInputs(frame: anytype, actual_name: []const u8, predicted_name: []const u8) ErrorFrameError!struct { actual: @TypeOf(frame.column(actual_name) catch unreachable), predicted: @TypeOf(frame.column(predicted_name) catch unreachable) } {
    const actual = try frame.column(actual_name);
    const predicted = try frame.column(predicted_name);
    if (actual.dtype() != predicted.dtype()) return error.TypeMismatch;
    return .{ .actual = actual, .predicted = predicted };
}

pub fn errorProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    actual_name: []const u8,
    predicted_name: []const u8,
    output_prefix: []const u8,
) ErrorFrameError!DeviceDataFrame {
    const inputs = try validateErrorInputs(frame, actual_name, predicted_name);
    const error_columns = try errorProfileColumnsByValue(frame.allocator, inputs.actual.*, inputs.predicted.*, frame.device, frame.rows);
    return errorFrameFromColumns(DeviceDataFrame, frame, output_prefix, error_columns, errorProfileOutputNames);
}

pub fn rollingErrorProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    actual_name: []const u8,
    predicted_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceRollingOptions,
) ErrorFrameError!DeviceDataFrame {
    const inputs = try validateErrorInputs(frame, actual_name, predicted_name);
    const error_columns = try rollingErrorProfileColumnsByValue(frame.allocator, inputs.actual.*, inputs.predicted.*, options_value, frame.device, frame.rows);
    return errorFrameFromColumns(DeviceDataFrame, frame, output_prefix, error_columns, rollingErrorProfileOutputNames);
}

pub fn expandingErrorProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    actual_name: []const u8,
    predicted_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceExpandingOptions,
) ErrorFrameError!DeviceDataFrame {
    const inputs = try validateErrorInputs(frame, actual_name, predicted_name);
    const error_columns = try expandingErrorProfileColumnsByValue(frame.allocator, inputs.actual.*, inputs.predicted.*, options_value, frame.device, frame.rows);
    return errorFrameFromColumns(DeviceDataFrame, frame, output_prefix, error_columns, expandingErrorProfileOutputNames);
}
