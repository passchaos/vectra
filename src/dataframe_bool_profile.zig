const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const names_mod = @import("dataframe_names.zig");
const dataframe_device_column_mod = @import("dataframe/device_column.zig");
const metrics_mod = @import("dataframe_bool_profile_metrics.zig");
const options_mod = @import("dataframe_options.zig");
const validity_mod = @import("dataframe_validity.zig");

const DeviceColumn = dataframe_device_column_mod.DeviceColumn;
const DeviceTypedColumn = dataframe_device_column_mod.DeviceTypedColumn;
const DeviceRollingOptions = options_mod.DeviceRollingOptions;
const DeviceExpandingOptions = options_mod.DeviceExpandingOptions;
const validityValues = validity_mod.validityValues;

pub const BoolProfileMetrics = metrics_mod.BoolProfileMetrics;
pub const RollingBoolProfileColumnCount = metrics_mod.RollingBoolProfileColumnCount;
pub const ExpandingBoolProfileColumnCount = metrics_mod.ExpandingBoolProfileColumnCount;
pub const rollingBoolProfileOutputNames = metrics_mod.rollingBoolProfileOutputNames;
pub const expandingBoolProfileOutputNames = metrics_mod.expandingBoolProfileOutputNames;
pub const rollingBoolProfile = metrics_mod.rollingBoolProfile;
pub const expandingBoolProfile = metrics_mod.expandingBoolProfile;

pub fn rollingBoolProfileColumns(
    allocator: std.mem.Allocator,
    source: DeviceTypedColumn(bool),
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![RollingBoolProfileColumnCount]DeviceColumn {
    const min_periods = options_value.min_periods orelse options_value.window;
    if (source.len() != rows) return error.LengthMismatch;

    const values = try source.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(source, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    var metrics = try rollingBoolProfile(allocator, values, maybe_validity, options_value.window, min_periods);
    defer metrics.deinit();

    var columns: [RollingBoolProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.true_counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSlice(i64, allocator, metrics.false_counts, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.true_rates, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(bool, allocator, metrics.any_values, metrics.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(bool, allocator, metrics.all_values, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn expandingBoolProfileColumns(
    allocator: std.mem.Allocator,
    source: DeviceTypedColumn(bool),
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingBoolProfileColumnCount]DeviceColumn {
    if (source.len() != rows) return error.LengthMismatch;

    const values = try source.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(source, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    var metrics = try expandingBoolProfile(allocator, values, maybe_validity, options_value.min_periods);
    defer metrics.deinit();

    var columns: [ExpandingBoolProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.true_counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSlice(i64, allocator, metrics.false_counts, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.true_rates, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(bool, allocator, metrics.any_values, metrics.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(bool, allocator, metrics.all_values, metrics.validity, device_value);
    initialized += 1;
    return columns;
}

const BoolFrameError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    TypeUnsupported,
    UnsupportedType,
    InvalidCsv,
    EmptyDataFrame,
    InvalidDevice,
};

fn appendBoolColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    source_names: []const []const u8,
    bool_columns: anytype,
) BoolFrameError!DeviceDataFrame {
    const DeviceColumnType = std.meta.Elem(@TypeOf(frame.columns));
    var columns = try frame.allocator.alloc(DeviceColumnType, frame.columns.len + bool_columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }
    for (frame.columns, 0..) |col, i| {
        columns[i] = try col.clone();
        initialized += 1;
    }
    for (&bool_columns) |*bool_col| {
        columns[initialized] = bool_col.*;
        initialized += 1;
    }
    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, source_names, columns, frame.rows, frame.device);
}

fn boolFrameFromColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    output_prefix: []const u8,
    bool_columns_value: anytype,
    comptime namesFn: anytype,
) BoolFrameError!DeviceDataFrame {
    var bool_columns = bool_columns_value;
    var bool_columns_transferred: usize = 0;
    errdefer {
        for (bool_columns[bool_columns_transferred..]) |*col| col.deinit();
    }

    const source_names = try frame.allocator.alloc([]const u8, frame.columns.len + bool_columns.len);
    defer frame.allocator.free(source_names);
    for (frame.names, 0..) |source_name, i| source_names[i] = source_name;

    var bool_names = try namesFn(frame.allocator, output_prefix);
    defer names_mod.freeOwnedNameItems(frame.allocator, bool_names[0..]);
    for (bool_names, 0..) |bool_name, i| source_names[frame.columns.len + i] = bool_name;

    const out = try appendBoolColumns(DeviceDataFrame, frame, source_names, bool_columns);
    bool_columns_transferred = bool_columns.len;
    return out;
}

fn boolSource(frame: anytype, name: []const u8) BoolFrameError!@TypeOf((frame.column(name) catch unreachable).bool) {
    const source = try frame.column(name);
    if (source.dtype() != .bool) return error.TypeMismatch;
    return source.bool;
}

pub fn rollingBoolProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceRollingOptions,
) BoolFrameError!DeviceDataFrame {
    const source = try boolSource(frame, name);
    const bool_columns = try rollingBoolProfileColumns(frame.allocator, source, options_value, frame.device, frame.rows);
    return boolFrameFromColumns(DeviceDataFrame, frame, output_prefix, bool_columns, rollingBoolProfileOutputNames);
}

pub fn expandingBoolProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceExpandingOptions,
) BoolFrameError!DeviceDataFrame {
    const source = try boolSource(frame, name);
    const bool_columns = try expandingBoolProfileColumns(frame.allocator, source, options_value, frame.device, frame.rows);
    return boolFrameFromColumns(DeviceDataFrame, frame, output_prefix, bool_columns, expandingBoolProfileOutputNames);
}
