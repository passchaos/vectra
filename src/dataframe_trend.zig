const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const names_mod = @import("dataframe_names.zig");
const dataframe_device_column_mod = @import("dataframe_device_column.zig");
const trend_metrics_mod = @import("dataframe_trend_metrics.zig");
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

pub const TrendMetrics = trend_metrics_mod.TrendMetrics;
pub const TrendSummaryMetrics = trend_metrics_mod.TrendSummaryMetrics;
pub const TrendProfileColumnCount = trend_metrics_mod.TrendProfileColumnCount;
pub const RollingTrendProfileColumnCount = trend_metrics_mod.RollingTrendProfileColumnCount;
pub const ExpandingTrendProfileColumnCount = trend_metrics_mod.ExpandingTrendProfileColumnCount;
pub const trendProfileOutputNames = trend_metrics_mod.trendProfileOutputNames;
pub const rollingTrendProfileOutputNames = trend_metrics_mod.rollingTrendProfileOutputNames;
pub const expandingTrendProfileOutputNames = trend_metrics_mod.expandingTrendProfileOutputNames;
pub const trendProfile = trend_metrics_mod.trendProfile;
pub const rollingTrendProfile = trend_metrics_mod.rollingTrendProfile;
pub const expandingTrendProfile = trend_metrics_mod.expandingTrendProfile;

pub fn trendProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceTrendOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![TrendProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| trendProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| trendProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| trendProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| trendProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| trendProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| trendProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| trendProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| trendProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| trendProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| trendProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| trendProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| trendProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| trendProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn trendProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceTrendOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![TrendProfileColumnCount]DeviceColumn {
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);
    var metrics = try trendProfile(allocator, values, maybe_validity, options_value.periods);
    defer metrics.deinit();

    var columns: [TrendProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(i64, allocator, metrics.trends, metrics.validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(i64, allocator, metrics.up_streak, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(i64, allocator, metrics.down_streak, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(i64, allocator, metrics.flat_streak, metrics.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(bool, allocator, metrics.reversal, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn rollingTrendProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    trend_options: DeviceTrendOptions,
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![RollingTrendProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| rollingTrendProfileColumnsTyped(i8, allocator, typed, trend_options, options_value, device_value),
        .i16 => |typed| rollingTrendProfileColumnsTyped(i16, allocator, typed, trend_options, options_value, device_value),
        .i32 => |typed| rollingTrendProfileColumnsTyped(i32, allocator, typed, trend_options, options_value, device_value),
        .i64 => |typed| rollingTrendProfileColumnsTyped(i64, allocator, typed, trend_options, options_value, device_value),
        .u8 => |typed| rollingTrendProfileColumnsTyped(u8, allocator, typed, trend_options, options_value, device_value),
        .u16 => |typed| rollingTrendProfileColumnsTyped(u16, allocator, typed, trend_options, options_value, device_value),
        .u32 => |typed| rollingTrendProfileColumnsTyped(u32, allocator, typed, trend_options, options_value, device_value),
        .u64 => |typed| rollingTrendProfileColumnsTyped(u64, allocator, typed, trend_options, options_value, device_value),
        .usize => |typed| rollingTrendProfileColumnsTyped(usize, allocator, typed, trend_options, options_value, device_value),
        .isize => |typed| rollingTrendProfileColumnsTyped(isize, allocator, typed, trend_options, options_value, device_value),
        .f16 => |typed| rollingTrendProfileColumnsTyped(f16, allocator, typed, trend_options, options_value, device_value),
        .f32 => |typed| rollingTrendProfileColumnsTyped(f32, allocator, typed, trend_options, options_value, device_value),
        .f64 => |typed| rollingTrendProfileColumnsTyped(f64, allocator, typed, trend_options, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn rollingTrendProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    trend_options: DeviceTrendOptions,
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![RollingTrendProfileColumnCount]DeviceColumn {
    const min_periods = options_value.min_periods orelse options_value.window;
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);
    var metrics = try rollingTrendProfile(allocator, values, maybe_validity, trend_options.periods, options_value.window, min_periods);
    defer metrics.deinit();

    var columns: [RollingTrendProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.up_rates, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.down_rates, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.flat_rates, metrics.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.reversal_rates, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn expandingTrendProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    trend_options: DeviceTrendOptions,
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingTrendProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| expandingTrendProfileColumnsTyped(i8, allocator, typed, trend_options, options_value, device_value),
        .i16 => |typed| expandingTrendProfileColumnsTyped(i16, allocator, typed, trend_options, options_value, device_value),
        .i32 => |typed| expandingTrendProfileColumnsTyped(i32, allocator, typed, trend_options, options_value, device_value),
        .i64 => |typed| expandingTrendProfileColumnsTyped(i64, allocator, typed, trend_options, options_value, device_value),
        .u8 => |typed| expandingTrendProfileColumnsTyped(u8, allocator, typed, trend_options, options_value, device_value),
        .u16 => |typed| expandingTrendProfileColumnsTyped(u16, allocator, typed, trend_options, options_value, device_value),
        .u32 => |typed| expandingTrendProfileColumnsTyped(u32, allocator, typed, trend_options, options_value, device_value),
        .u64 => |typed| expandingTrendProfileColumnsTyped(u64, allocator, typed, trend_options, options_value, device_value),
        .usize => |typed| expandingTrendProfileColumnsTyped(usize, allocator, typed, trend_options, options_value, device_value),
        .isize => |typed| expandingTrendProfileColumnsTyped(isize, allocator, typed, trend_options, options_value, device_value),
        .f16 => |typed| expandingTrendProfileColumnsTyped(f16, allocator, typed, trend_options, options_value, device_value),
        .f32 => |typed| expandingTrendProfileColumnsTyped(f32, allocator, typed, trend_options, options_value, device_value),
        .f64 => |typed| expandingTrendProfileColumnsTyped(f64, allocator, typed, trend_options, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn expandingTrendProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    trend_options: DeviceTrendOptions,
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingTrendProfileColumnCount]DeviceColumn {
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);
    var metrics = try expandingTrendProfile(allocator, values, maybe_validity, trend_options.periods, options_value.min_periods);
    defer metrics.deinit();

    var columns: [ExpandingTrendProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.up_rates, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.down_rates, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.flat_rates, metrics.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.reversal_rates, metrics.validity, device_value);
    initialized += 1;
    return columns;
}

const TrendFrameError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    TypeUnsupported,
    UnsupportedType,
    InvalidCsv,
    EmptyDataFrame,
    InvalidDevice,
};

fn appendTrendColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    source_names: []const []const u8,
    trend_columns: anytype,
) TrendFrameError!DeviceDataFrame {
    const DeviceColumnType = std.meta.Elem(@TypeOf(frame.columns));
    var columns = try frame.allocator.alloc(DeviceColumnType, frame.columns.len + trend_columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }
    for (frame.columns, 0..) |col, i| {
        columns[i] = try col.clone();
        initialized += 1;
    }
    for (&trend_columns) |*trend_col| {
        columns[initialized] = trend_col.*;
        initialized += 1;
    }
    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, source_names, columns, frame.rows, frame.device);
}

fn trendProfileFrameFromColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    output_prefix: []const u8,
    trend_columns_value: anytype,
    comptime namesFn: anytype,
) TrendFrameError!DeviceDataFrame {
    var trend_columns = trend_columns_value;
    var trend_columns_transferred: usize = 0;
    errdefer {
        for (trend_columns[trend_columns_transferred..]) |*col| col.deinit();
    }

    const source_names = try frame.allocator.alloc([]const u8, frame.columns.len + trend_columns.len);
    defer frame.allocator.free(source_names);
    for (frame.names, 0..) |source_name, i| source_names[i] = source_name;

    var trend_names = try namesFn(frame.allocator, output_prefix);
    defer names_mod.freeOwnedNameItems(frame.allocator, trend_names[0..]);
    for (trend_names, 0..) |trend_name, i| source_names[frame.columns.len + i] = trend_name;

    const out = try appendTrendColumns(DeviceDataFrame, frame, source_names, trend_columns);
    trend_columns_transferred = trend_columns.len;
    return out;
}

pub fn trendProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceTrendOptions,
) TrendFrameError!DeviceDataFrame {
    const trend_value = try frame.column(name);
    const trend_columns = try trendProfileColumnsByValue(frame.allocator, trend_value.*, options_value, frame.device, frame.rows);
    return trendProfileFrameFromColumns(DeviceDataFrame, frame, output_prefix, trend_columns, trendProfileOutputNames);
}

pub fn rollingTrendProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    trend_options: DeviceTrendOptions,
    options_value: DeviceRollingOptions,
) TrendFrameError!DeviceDataFrame {
    const trend_value = try frame.column(name);
    const trend_columns = try rollingTrendProfileColumnsByValue(frame.allocator, trend_value.*, trend_options, options_value, frame.device, frame.rows);
    return trendProfileFrameFromColumns(DeviceDataFrame, frame, output_prefix, trend_columns, rollingTrendProfileOutputNames);
}

pub fn expandingTrendProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    trend_options: DeviceTrendOptions,
    options_value: DeviceExpandingOptions,
) TrendFrameError!DeviceDataFrame {
    const trend_value = try frame.column(name);
    const trend_columns = try expandingTrendProfileColumnsByValue(frame.allocator, trend_value.*, trend_options, options_value, frame.device, frame.rows);
    return trendProfileFrameFromColumns(DeviceDataFrame, frame, output_prefix, trend_columns, expandingTrendProfileOutputNames);
}
