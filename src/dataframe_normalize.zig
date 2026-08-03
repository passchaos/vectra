const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const names_mod = @import("dataframe_names.zig");
const dataframe_device_column_mod = @import("dataframe/device_column.zig");
const normalize_metrics_mod = @import("dataframe_normalize_metrics.zig");
const numeric_mod = @import("dataframe_numeric.zig");
const options_mod = @import("dataframe_options.zig");
const validity_mod = @import("dataframe/validity.zig");

const DeviceColumn = dataframe_device_column_mod.DeviceColumn;
const DeviceTypedColumn = dataframe_device_column_mod.DeviceTypedColumn;
const DeviceRollingOptions = options_mod.DeviceRollingOptions;
const DeviceExpandingOptions = options_mod.DeviceExpandingOptions;
const castToF64 = numeric_mod.castToF64;
const validityValues = validity_mod.validityValues;

pub const NormalizeMetrics = normalize_metrics_mod.NormalizeMetrics;
pub const RollingNormalizeProfileColumnCount = normalize_metrics_mod.RollingNormalizeProfileColumnCount;
pub const ExpandingNormalizeProfileColumnCount = normalize_metrics_mod.ExpandingNormalizeProfileColumnCount;
pub const rollingNormalizeProfileOutputNames = normalize_metrics_mod.rollingNormalizeProfileOutputNames;
pub const expandingNormalizeProfileOutputNames = normalize_metrics_mod.expandingNormalizeProfileOutputNames;
pub const rollingNormalizeProfile = normalize_metrics_mod.rollingNormalizeProfile;
pub const expandingNormalizeProfile = normalize_metrics_mod.expandingNormalizeProfile;

pub fn rollingNormalizeProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![RollingNormalizeProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| rollingNormalizeProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| rollingNormalizeProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| rollingNormalizeProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| rollingNormalizeProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| rollingNormalizeProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| rollingNormalizeProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| rollingNormalizeProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| rollingNormalizeProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| rollingNormalizeProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| rollingNormalizeProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| rollingNormalizeProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| rollingNormalizeProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| rollingNormalizeProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn rollingNormalizeProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![RollingNormalizeProfileColumnCount]DeviceColumn {
    const min_periods = options_value.min_periods orelse options_value.window;
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);
    var metrics = try rollingNormalizeProfile(allocator, values, maybe_validity, options_value.window, min_periods);
    defer metrics.deinit();

    var columns: [RollingNormalizeProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.centered, metrics.validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.zscores, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.minmax, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn expandingNormalizeProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingNormalizeProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| expandingNormalizeProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| expandingNormalizeProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| expandingNormalizeProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| expandingNormalizeProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| expandingNormalizeProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| expandingNormalizeProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| expandingNormalizeProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| expandingNormalizeProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| expandingNormalizeProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| expandingNormalizeProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| expandingNormalizeProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| expandingNormalizeProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| expandingNormalizeProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn expandingNormalizeProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingNormalizeProfileColumnCount]DeviceColumn {
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);
    var metrics = try expandingNormalizeProfile(allocator, values, maybe_validity, options_value.min_periods);
    defer metrics.deinit();

    var columns: [ExpandingNormalizeProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.centered, metrics.validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.zscores, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.minmax, metrics.validity, device_value);
    initialized += 1;
    return columns;
}

const NormalizeFrameError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    TypeUnsupported,
    UnsupportedType,
    InvalidCsv,
    EmptyDataFrame,
    InvalidDevice,
};

fn appendNormalizeColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    source_names: []const []const u8,
    normalize_columns: anytype,
) NormalizeFrameError!DeviceDataFrame {
    const DeviceColumnType = std.meta.Elem(@TypeOf(frame.columns));
    var columns = try frame.allocator.alloc(DeviceColumnType, frame.columns.len + normalize_columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }
    for (frame.columns, 0..) |col, i| {
        columns[i] = try col.clone();
        initialized += 1;
    }
    for (&normalize_columns) |*normalize_col| {
        columns[initialized] = normalize_col.*;
        initialized += 1;
    }
    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, source_names, columns, frame.rows, frame.device);
}

fn normalizeFrameFromColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    output_prefix: []const u8,
    normalize_columns_value: anytype,
    comptime namesFn: anytype,
) NormalizeFrameError!DeviceDataFrame {
    var normalize_columns = normalize_columns_value;
    var normalize_columns_transferred: usize = 0;
    errdefer {
        for (normalize_columns[normalize_columns_transferred..]) |*col| col.deinit();
    }

    const source_names = try frame.allocator.alloc([]const u8, frame.columns.len + normalize_columns.len);
    defer frame.allocator.free(source_names);
    for (frame.names, 0..) |source_name, i| source_names[i] = source_name;

    var normalize_names = try namesFn(frame.allocator, output_prefix);
    defer names_mod.freeOwnedNameItems(frame.allocator, normalize_names[0..]);
    for (normalize_names, 0..) |normalize_name, i| source_names[frame.columns.len + i] = normalize_name;

    const out = try appendNormalizeColumns(DeviceDataFrame, frame, source_names, normalize_columns);
    normalize_columns_transferred = normalize_columns.len;
    return out;
}

pub fn rollingNormalizeProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceRollingOptions,
) NormalizeFrameError!DeviceDataFrame {
    const rolling_value = try frame.column(name);
    const rolling_columns = try rollingNormalizeProfileColumnsByValue(frame.allocator, rolling_value.*, options_value, frame.device, frame.rows);
    return normalizeFrameFromColumns(DeviceDataFrame, frame, output_prefix, rolling_columns, rollingNormalizeProfileOutputNames);
}

pub fn expandingNormalizeProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceExpandingOptions,
) NormalizeFrameError!DeviceDataFrame {
    const expanding_value = try frame.column(name);
    const expanding_columns = try expandingNormalizeProfileColumnsByValue(frame.allocator, expanding_value.*, options_value, frame.device, frame.rows);
    return normalizeFrameFromColumns(DeviceDataFrame, frame, output_prefix, expanding_columns, expandingNormalizeProfileOutputNames);
}
