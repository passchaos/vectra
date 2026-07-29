const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const names_mod = @import("dataframe_names.zig");
const dataframe_device_column_mod = @import("dataframe_device_column.zig");
const stats_metrics_mod = @import("dataframe_stats_metrics.zig");
const numeric_mod = @import("dataframe_numeric.zig");
const options_mod = @import("dataframe_options.zig");
const validity_mod = @import("dataframe_validity.zig");

const DeviceColumn = dataframe_device_column_mod.DeviceColumn;
const DeviceTypedColumn = dataframe_device_column_mod.DeviceTypedColumn;
const DeviceRollingOptions = options_mod.DeviceRollingOptions;
const DeviceExpandingOptions = options_mod.DeviceExpandingOptions;
const castToF64 = numeric_mod.castToF64;
const validityValues = validity_mod.validityValues;

pub const RollingStatsMetrics = stats_metrics_mod.RollingStatsMetrics;
pub const ExpandingStatsMetrics = stats_metrics_mod.ExpandingStatsMetrics;
pub const RollingProfileColumnCount = stats_metrics_mod.RollingProfileColumnCount;
pub const ExpandingProfileColumnCount = stats_metrics_mod.ExpandingProfileColumnCount;
pub const rollingProfileOutputNames = stats_metrics_mod.rollingProfileOutputNames;
pub const expandingProfileOutputNames = stats_metrics_mod.expandingProfileOutputNames;
pub const rollingProfile = stats_metrics_mod.rollingProfile;
pub const expandingProfile = stats_metrics_mod.expandingProfile;

pub fn rollingProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![RollingProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| rollingProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| rollingProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| rollingProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| rollingProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| rollingProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| rollingProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| rollingProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| rollingProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| rollingProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| rollingProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| rollingProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| rollingProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| rollingProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn rollingProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![RollingProfileColumnCount]DeviceColumn {
    const min_periods = options_value.min_periods orelse options_value.window;
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);
    var metrics = try rollingProfile(allocator, values, maybe_validity, options_value.window, min_periods);
    defer metrics.deinit();

    var columns: [RollingProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.sums, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.means, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.variances, metrics.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.stddevs, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn expandingProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| expandingProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| expandingProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| expandingProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| expandingProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| expandingProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| expandingProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| expandingProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| expandingProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| expandingProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| expandingProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| expandingProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| expandingProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| expandingProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn expandingProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingProfileColumnCount]DeviceColumn {
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);
    var metrics = try expandingProfile(allocator, values, maybe_validity, options_value.min_periods);
    defer metrics.deinit();

    var columns: [ExpandingProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.sums, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.means, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.mins, metrics.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.maxes, metrics.validity, device_value);
    initialized += 1;
    return columns;
}

const StatsFrameError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    TypeUnsupported,
    UnsupportedType,
    InvalidCsv,
    EmptyDataFrame,
    InvalidDevice,
};

fn appendStatsColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    source_names: []const []const u8,
    stats_columns: anytype,
) StatsFrameError!DeviceDataFrame {
    const DeviceColumnType = std.meta.Elem(@TypeOf(frame.columns));
    var columns = try frame.allocator.alloc(DeviceColumnType, frame.columns.len + stats_columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }
    for (frame.columns, 0..) |col, i| {
        columns[i] = try col.clone();
        initialized += 1;
    }
    for (&stats_columns) |*stats_col| {
        columns[initialized] = stats_col.*;
        initialized += 1;
    }
    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, source_names, columns, frame.rows, frame.device);
}

fn statsFrameFromColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    output_prefix: []const u8,
    stats_columns_value: anytype,
    comptime namesFn: anytype,
) StatsFrameError!DeviceDataFrame {
    var stats_columns = stats_columns_value;
    var stats_columns_transferred: usize = 0;
    errdefer {
        for (stats_columns[stats_columns_transferred..]) |*col| col.deinit();
    }

    const source_names = try frame.allocator.alloc([]const u8, frame.columns.len + stats_columns.len);
    defer frame.allocator.free(source_names);
    for (frame.names, 0..) |source_name, i| source_names[i] = source_name;

    var stats_names = try namesFn(frame.allocator, output_prefix);
    defer names_mod.freeOwnedNameItems(frame.allocator, stats_names[0..]);
    for (stats_names, 0..) |stats_name, i| source_names[frame.columns.len + i] = stats_name;

    const out = try appendStatsColumns(DeviceDataFrame, frame, source_names, stats_columns);
    stats_columns_transferred = stats_columns.len;
    return out;
}

pub fn rollingProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceRollingOptions,
) StatsFrameError!DeviceDataFrame {
    const rolling_value = try frame.column(name);
    const rolling_columns = try rollingProfileColumnsByValue(frame.allocator, rolling_value.*, options_value, frame.device, frame.rows);
    return statsFrameFromColumns(DeviceDataFrame, frame, output_prefix, rolling_columns, rollingProfileOutputNames);
}

pub fn expandingProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceExpandingOptions,
) StatsFrameError!DeviceDataFrame {
    const expanding_value = try frame.column(name);
    const expanding_columns = try expandingProfileColumnsByValue(frame.allocator, expanding_value.*, options_value, frame.device, frame.rows);
    return statsFrameFromColumns(DeviceDataFrame, frame, output_prefix, expanding_columns, expandingProfileOutputNames);
}
