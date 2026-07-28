const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const names_mod = @import("dataframe_names.zig");
const dataframe_device_column_mod = @import("dataframe_device_column.zig");
const sign_metrics_mod = @import("dataframe_sign_metrics.zig");
const numeric_mod = @import("dataframe_numeric.zig");
const options_mod = @import("dataframe_options.zig");
const validity_mod = @import("dataframe_validity.zig");

const DeviceColumn = dataframe_device_column_mod.DeviceColumn;
const DeviceTypedColumn = dataframe_device_column_mod.DeviceTypedColumn;
const DeviceTrendOptions = options_mod.DeviceTrendOptions;
const DeviceRollingOptions = options_mod.DeviceRollingOptions;
const DeviceExpandingOptions = options_mod.DeviceExpandingOptions;
const castToF64 = numeric_mod.castToF64;
const validityValues = validity_mod.validityValues;

pub const SignMetrics = sign_metrics_mod.SignMetrics;
pub const SignSummaryMetrics = sign_metrics_mod.SignSummaryMetrics;
pub const SignProfileColumnCount = sign_metrics_mod.SignProfileColumnCount;
pub const RollingSignProfileColumnCount = sign_metrics_mod.RollingSignProfileColumnCount;
pub const ExpandingSignProfileColumnCount = sign_metrics_mod.ExpandingSignProfileColumnCount;
pub const signProfileOutputNames = sign_metrics_mod.signProfileOutputNames;
pub const rollingSignProfileOutputNames = sign_metrics_mod.rollingSignProfileOutputNames;
pub const expandingSignProfileOutputNames = sign_metrics_mod.expandingSignProfileOutputNames;
pub const signProfile = sign_metrics_mod.signProfile;
pub const rollingSignProfile = sign_metrics_mod.rollingSignProfile;
pub const expandingSignProfile = sign_metrics_mod.expandingSignProfile;

pub fn signProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceTrendOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![SignProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| signProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| signProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| signProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| signProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| signProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| signProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| signProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| signProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| signProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| signProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| signProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| signProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| signProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn signProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceTrendOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![SignProfileColumnCount]DeviceColumn {
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);
    var metrics = try signProfile(allocator, values, maybe_validity, options_value.periods);
    defer metrics.deinit();

    var columns: [SignProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(i64, allocator, metrics.signs, metrics.sign_validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(bool, allocator, metrics.flips, metrics.flip_validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(i64, allocator, metrics.positive_streak, metrics.sign_validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(i64, allocator, metrics.negative_streak, metrics.sign_validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(i64, allocator, metrics.zero_streak, metrics.sign_validity, device_value);
    initialized += 1;
    return columns;
}
pub fn rollingSignProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    sign_options: DeviceTrendOptions,
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![RollingSignProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| rollingSignProfileColumnsTyped(i8, allocator, typed, sign_options, options_value, device_value),
        .i16 => |typed| rollingSignProfileColumnsTyped(i16, allocator, typed, sign_options, options_value, device_value),
        .i32 => |typed| rollingSignProfileColumnsTyped(i32, allocator, typed, sign_options, options_value, device_value),
        .i64 => |typed| rollingSignProfileColumnsTyped(i64, allocator, typed, sign_options, options_value, device_value),
        .u8 => |typed| rollingSignProfileColumnsTyped(u8, allocator, typed, sign_options, options_value, device_value),
        .u16 => |typed| rollingSignProfileColumnsTyped(u16, allocator, typed, sign_options, options_value, device_value),
        .u32 => |typed| rollingSignProfileColumnsTyped(u32, allocator, typed, sign_options, options_value, device_value),
        .u64 => |typed| rollingSignProfileColumnsTyped(u64, allocator, typed, sign_options, options_value, device_value),
        .usize => |typed| rollingSignProfileColumnsTyped(usize, allocator, typed, sign_options, options_value, device_value),
        .isize => |typed| rollingSignProfileColumnsTyped(isize, allocator, typed, sign_options, options_value, device_value),
        .f16 => |typed| rollingSignProfileColumnsTyped(f16, allocator, typed, sign_options, options_value, device_value),
        .f32 => |typed| rollingSignProfileColumnsTyped(f32, allocator, typed, sign_options, options_value, device_value),
        .f64 => |typed| rollingSignProfileColumnsTyped(f64, allocator, typed, sign_options, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn rollingSignProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    sign_options: DeviceTrendOptions,
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![RollingSignProfileColumnCount]DeviceColumn {
    const min_periods = options_value.min_periods orelse options_value.window;
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);
    var metrics = try rollingSignProfile(allocator, values, maybe_validity, sign_options.periods, options_value.window, min_periods);
    defer metrics.deinit();

    var columns: [RollingSignProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.positive_rates, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.negative_rates, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.zero_rates, metrics.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.flip_rates, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn expandingSignProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    sign_options: DeviceTrendOptions,
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingSignProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| expandingSignProfileColumnsTyped(i8, allocator, typed, sign_options, options_value, device_value),
        .i16 => |typed| expandingSignProfileColumnsTyped(i16, allocator, typed, sign_options, options_value, device_value),
        .i32 => |typed| expandingSignProfileColumnsTyped(i32, allocator, typed, sign_options, options_value, device_value),
        .i64 => |typed| expandingSignProfileColumnsTyped(i64, allocator, typed, sign_options, options_value, device_value),
        .u8 => |typed| expandingSignProfileColumnsTyped(u8, allocator, typed, sign_options, options_value, device_value),
        .u16 => |typed| expandingSignProfileColumnsTyped(u16, allocator, typed, sign_options, options_value, device_value),
        .u32 => |typed| expandingSignProfileColumnsTyped(u32, allocator, typed, sign_options, options_value, device_value),
        .u64 => |typed| expandingSignProfileColumnsTyped(u64, allocator, typed, sign_options, options_value, device_value),
        .usize => |typed| expandingSignProfileColumnsTyped(usize, allocator, typed, sign_options, options_value, device_value),
        .isize => |typed| expandingSignProfileColumnsTyped(isize, allocator, typed, sign_options, options_value, device_value),
        .f16 => |typed| expandingSignProfileColumnsTyped(f16, allocator, typed, sign_options, options_value, device_value),
        .f32 => |typed| expandingSignProfileColumnsTyped(f32, allocator, typed, sign_options, options_value, device_value),
        .f64 => |typed| expandingSignProfileColumnsTyped(f64, allocator, typed, sign_options, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn expandingSignProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    sign_options: DeviceTrendOptions,
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingSignProfileColumnCount]DeviceColumn {
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);
    var metrics = try expandingSignProfile(allocator, values, maybe_validity, sign_options.periods, options_value.min_periods);
    defer metrics.deinit();

    var columns: [ExpandingSignProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.positive_rates, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.negative_rates, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.zero_rates, metrics.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.flip_rates, metrics.validity, device_value);
    initialized += 1;
    return columns;
}

const SignFrameError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    TypeUnsupported,
    UnsupportedType,
    InvalidCsv,
    EmptyDataFrame,
    InvalidDevice,
};

fn appendSignColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    source_names: []const []const u8,
    sign_columns: anytype,
) SignFrameError!DeviceDataFrame {
    const DeviceColumnType = std.meta.Elem(@TypeOf(frame.columns));
    var columns = try frame.allocator.alloc(DeviceColumnType, frame.columns.len + sign_columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }
    for (frame.columns, 0..) |col, i| {
        columns[i] = try col.clone();
        initialized += 1;
    }
    for (&sign_columns) |*sign_col| {
        columns[initialized] = sign_col.*;
        initialized += 1;
    }
    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, source_names, columns, frame.rows, frame.device);
}

fn signFrameFromColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    output_prefix: []const u8,
    sign_columns_value: anytype,
    comptime namesFn: anytype,
) SignFrameError!DeviceDataFrame {
    var sign_columns = sign_columns_value;
    var sign_columns_transferred: usize = 0;
    errdefer {
        for (sign_columns[sign_columns_transferred..]) |*col| col.deinit();
    }

    const source_names = try frame.allocator.alloc([]const u8, frame.columns.len + sign_columns.len);
    defer frame.allocator.free(source_names);
    for (frame.names, 0..) |source_name, i| source_names[i] = source_name;

    var sign_names = try namesFn(frame.allocator, output_prefix);
    defer names_mod.freeOwnedNameItems(frame.allocator, sign_names[0..]);
    for (sign_names, 0..) |sign_name, i| source_names[frame.columns.len + i] = sign_name;

    const out = try appendSignColumns(DeviceDataFrame, frame, source_names, sign_columns);
    sign_columns_transferred = sign_columns.len;
    return out;
}

pub fn signProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceTrendOptions,
) SignFrameError!DeviceDataFrame {
    const sign_value = try frame.column(name);
    const sign_columns = try signProfileColumnsByValue(frame.allocator, sign_value.*, options_value, frame.device, frame.rows);
    return signFrameFromColumns(DeviceDataFrame, frame, output_prefix, sign_columns, signProfileOutputNames);
}

pub fn rollingSignProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    sign_options: DeviceTrendOptions,
    options_value: DeviceRollingOptions,
) SignFrameError!DeviceDataFrame {
    const sign_value = try frame.column(name);
    const sign_columns = try rollingSignProfileColumnsByValue(frame.allocator, sign_value.*, sign_options, options_value, frame.device, frame.rows);
    return signFrameFromColumns(DeviceDataFrame, frame, output_prefix, sign_columns, rollingSignProfileOutputNames);
}

pub fn expandingSignProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    sign_options: DeviceTrendOptions,
    options_value: DeviceExpandingOptions,
) SignFrameError!DeviceDataFrame {
    const sign_value = try frame.column(name);
    const sign_columns = try expandingSignProfileColumnsByValue(frame.allocator, sign_value.*, sign_options, options_value, frame.device, frame.rows);
    return signFrameFromColumns(DeviceDataFrame, frame, output_prefix, sign_columns, expandingSignProfileOutputNames);
}
