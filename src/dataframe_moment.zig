const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const names_mod = @import("dataframe_names.zig");
const dataframe_device_column_mod = @import("dataframe_device_column.zig");
const moment_metrics_mod = @import("dataframe_moment_metrics.zig");
const numeric_mod = @import("dataframe_numeric.zig");
const options_mod = @import("dataframe_options.zig");
const validity_mod = @import("dataframe_validity.zig");

const DeviceColumn = dataframe_device_column_mod.DeviceColumn;
const DeviceTypedColumn = dataframe_device_column_mod.DeviceTypedColumn;
const DeviceRollingOptions = options_mod.DeviceRollingOptions;
const DeviceExpandingOptions = options_mod.DeviceExpandingOptions;
const castToF64 = numeric_mod.castToF64;
const validityValues = validity_mod.validityValues;

pub const MomentMetrics = moment_metrics_mod.MomentMetrics;
pub const RollingMomentProfileColumnCount = moment_metrics_mod.RollingMomentProfileColumnCount;
pub const ExpandingMomentProfileColumnCount = moment_metrics_mod.ExpandingMomentProfileColumnCount;
pub const rollingMomentProfileOutputNames = moment_metrics_mod.rollingMomentProfileOutputNames;
pub const expandingMomentProfileOutputNames = moment_metrics_mod.expandingMomentProfileOutputNames;
pub const rollingMomentProfile = moment_metrics_mod.rollingMomentProfile;
pub const expandingMomentProfile = moment_metrics_mod.expandingMomentProfile;

pub fn rollingMomentProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![RollingMomentProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| rollingMomentProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| rollingMomentProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| rollingMomentProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| rollingMomentProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| rollingMomentProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| rollingMomentProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| rollingMomentProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| rollingMomentProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| rollingMomentProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| rollingMomentProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| rollingMomentProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| rollingMomentProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| rollingMomentProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn rollingMomentProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![RollingMomentProfileColumnCount]DeviceColumn {
    const min_periods = options_value.min_periods orelse options_value.window;
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);
    var metrics = try rollingMomentProfile(allocator, values, maybe_validity, options_value.window, min_periods);
    defer metrics.deinit();

    var columns: [RollingMomentProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.m3_values, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.m4_values, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.skewnesses, metrics.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.kurtoses, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn expandingMomentProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingMomentProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| expandingMomentProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| expandingMomentProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| expandingMomentProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| expandingMomentProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| expandingMomentProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| expandingMomentProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| expandingMomentProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| expandingMomentProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| expandingMomentProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| expandingMomentProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| expandingMomentProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| expandingMomentProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| expandingMomentProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn expandingMomentProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingMomentProfileColumnCount]DeviceColumn {
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);
    var metrics = try expandingMomentProfile(allocator, values, maybe_validity, options_value.min_periods);
    defer metrics.deinit();

    var columns: [ExpandingMomentProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.m3_values, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.m4_values, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.skewnesses, metrics.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.kurtoses, metrics.validity, device_value);
    initialized += 1;
    return columns;
}

const MomentFrameError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    TypeUnsupported,
    UnsupportedType,
    InvalidCsv,
    EmptyDataFrame,
    InvalidDevice,
};

fn appendMomentColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    source_names: []const []const u8,
    moment_columns: anytype,
) MomentFrameError!DeviceDataFrame {
    const DeviceColumnType = std.meta.Elem(@TypeOf(frame.columns));
    var columns = try frame.allocator.alloc(DeviceColumnType, frame.columns.len + moment_columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }
    for (frame.columns, 0..) |col, i| {
        columns[i] = try col.clone();
        initialized += 1;
    }
    for (&moment_columns) |*moment_col| {
        columns[initialized] = moment_col.*;
        initialized += 1;
    }
    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, source_names, columns, frame.rows, frame.device);
}

fn momentFrameFromColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    output_prefix: []const u8,
    moment_columns_value: anytype,
    comptime namesFn: anytype,
) MomentFrameError!DeviceDataFrame {
    var moment_columns = moment_columns_value;
    var moment_columns_transferred: usize = 0;
    errdefer {
        for (moment_columns[moment_columns_transferred..]) |*col| col.deinit();
    }

    const source_names = try frame.allocator.alloc([]const u8, frame.columns.len + moment_columns.len);
    defer frame.allocator.free(source_names);
    for (frame.names, 0..) |source_name, i| source_names[i] = source_name;

    var moment_names = try namesFn(frame.allocator, output_prefix);
    defer names_mod.freeOwnedNameItems(frame.allocator, moment_names[0..]);
    for (moment_names, 0..) |moment_name, i| source_names[frame.columns.len + i] = moment_name;

    const out = try appendMomentColumns(DeviceDataFrame, frame, source_names, moment_columns);
    moment_columns_transferred = moment_columns.len;
    return out;
}

pub fn rollingMomentProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceRollingOptions,
) MomentFrameError!DeviceDataFrame {
    const rolling_value = try frame.column(name);
    const rolling_columns = try rollingMomentProfileColumnsByValue(frame.allocator, rolling_value.*, options_value, frame.device, frame.rows);
    return momentFrameFromColumns(DeviceDataFrame, frame, output_prefix, rolling_columns, rollingMomentProfileOutputNames);
}

pub fn expandingMomentProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceExpandingOptions,
) MomentFrameError!DeviceDataFrame {
    const expanding_value = try frame.column(name);
    const expanding_columns = try expandingMomentProfileColumnsByValue(frame.allocator, expanding_value.*, options_value, frame.device, frame.rows);
    return momentFrameFromColumns(DeviceDataFrame, frame, output_prefix, expanding_columns, expandingMomentProfileOutputNames);
}
