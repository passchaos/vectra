const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const names_mod = @import("dataframe_names.zig");
const dataframe_device_column_mod = @import("dataframe/device_column.zig");
const quantile_metrics_mod = @import("dataframe_quantile_metrics.zig");
const numeric_mod = @import("dataframe_numeric.zig");
const options_mod = @import("dataframe_options.zig");
const validity_mod = @import("dataframe_validity.zig");

const DeviceColumn = dataframe_device_column_mod.DeviceColumn;
const DeviceTypedColumn = dataframe_device_column_mod.DeviceTypedColumn;
const DeviceRollingOptions = options_mod.DeviceRollingOptions;
const DeviceExpandingOptions = options_mod.DeviceExpandingOptions;
const castToF64 = numeric_mod.castToF64;
const validityValues = validity_mod.validityValues;

pub const QuantileMetrics = quantile_metrics_mod.QuantileMetrics;
pub const RollingQuantileProfileColumnCount = quantile_metrics_mod.RollingQuantileProfileColumnCount;
pub const ExpandingQuantileProfileColumnCount = quantile_metrics_mod.ExpandingQuantileProfileColumnCount;
pub const rollingQuantileProfileOutputNames = quantile_metrics_mod.rollingQuantileProfileOutputNames;
pub const expandingQuantileProfileOutputNames = quantile_metrics_mod.expandingQuantileProfileOutputNames;
pub const rollingQuantileProfile = quantile_metrics_mod.rollingQuantileProfile;
pub const expandingQuantileProfile = quantile_metrics_mod.expandingQuantileProfile;

pub fn rollingQuantileProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![RollingQuantileProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| rollingQuantileProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| rollingQuantileProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| rollingQuantileProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| rollingQuantileProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| rollingQuantileProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| rollingQuantileProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| rollingQuantileProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| rollingQuantileProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| rollingQuantileProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| rollingQuantileProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| rollingQuantileProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| rollingQuantileProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| rollingQuantileProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn rollingQuantileProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![RollingQuantileProfileColumnCount]DeviceColumn {
    const min_periods = options_value.min_periods orelse options_value.window;
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);
    var metrics = try rollingQuantileProfile(allocator, values, maybe_validity, options_value.window, min_periods);
    defer metrics.deinit();

    var columns: [RollingQuantileProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.q1, metrics.validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.medians, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.q3, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.iqrs, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn expandingQuantileProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingQuantileProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| expandingQuantileProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| expandingQuantileProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| expandingQuantileProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| expandingQuantileProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| expandingQuantileProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| expandingQuantileProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| expandingQuantileProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| expandingQuantileProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| expandingQuantileProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| expandingQuantileProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| expandingQuantileProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| expandingQuantileProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| expandingQuantileProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn expandingQuantileProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingQuantileProfileColumnCount]DeviceColumn {
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);
    var metrics = try expandingQuantileProfile(allocator, values, maybe_validity, options_value.min_periods);
    defer metrics.deinit();

    var columns: [ExpandingQuantileProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.q1, metrics.validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.medians, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.q3, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.iqrs, metrics.validity, device_value);
    initialized += 1;
    return columns;
}

const QuantileFrameError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    TypeUnsupported,
    UnsupportedType,
    InvalidCsv,
    EmptyDataFrame,
    InvalidDevice,
};

fn appendQuantileColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    source_names: []const []const u8,
    quantile_columns: anytype,
) QuantileFrameError!DeviceDataFrame {
    const DeviceColumnType = std.meta.Elem(@TypeOf(frame.columns));
    var columns = try frame.allocator.alloc(DeviceColumnType, frame.columns.len + quantile_columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }
    for (frame.columns, 0..) |col, i| {
        columns[i] = try col.clone();
        initialized += 1;
    }
    for (&quantile_columns) |*quantile_col| {
        columns[initialized] = quantile_col.*;
        initialized += 1;
    }
    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, source_names, columns, frame.rows, frame.device);
}

fn quantileFrameFromColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    output_prefix: []const u8,
    quantile_columns_value: anytype,
    comptime namesFn: anytype,
) QuantileFrameError!DeviceDataFrame {
    var quantile_columns = quantile_columns_value;
    var quantile_columns_transferred: usize = 0;
    errdefer {
        for (quantile_columns[quantile_columns_transferred..]) |*col| col.deinit();
    }

    const source_names = try frame.allocator.alloc([]const u8, frame.columns.len + quantile_columns.len);
    defer frame.allocator.free(source_names);
    for (frame.names, 0..) |source_name, i| source_names[i] = source_name;

    var quantile_names = try namesFn(frame.allocator, output_prefix);
    defer names_mod.freeOwnedNameItems(frame.allocator, quantile_names[0..]);
    for (quantile_names, 0..) |quantile_name, i| source_names[frame.columns.len + i] = quantile_name;

    const out = try appendQuantileColumns(DeviceDataFrame, frame, source_names, quantile_columns);
    quantile_columns_transferred = quantile_columns.len;
    return out;
}

pub fn rollingQuantileProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceRollingOptions,
) QuantileFrameError!DeviceDataFrame {
    const rolling_value = try frame.column(name);
    const rolling_columns = try rollingQuantileProfileColumnsByValue(frame.allocator, rolling_value.*, options_value, frame.device, frame.rows);
    return quantileFrameFromColumns(DeviceDataFrame, frame, output_prefix, rolling_columns, rollingQuantileProfileOutputNames);
}

pub fn expandingQuantileProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceExpandingOptions,
) QuantileFrameError!DeviceDataFrame {
    const expanding_value = try frame.column(name);
    const expanding_columns = try expandingQuantileProfileColumnsByValue(frame.allocator, expanding_value.*, options_value, frame.device, frame.rows);
    return quantileFrameFromColumns(DeviceDataFrame, frame, output_prefix, expanding_columns, expandingQuantileProfileOutputNames);
}
