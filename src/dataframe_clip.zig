const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const names_mod = @import("dataframe_names.zig");
const dataframe_device_column_mod = @import("dataframe_device_column.zig");
const clip_metrics_mod = @import("dataframe_clip_metrics.zig");
const numeric_mod = @import("dataframe_numeric.zig");
const options_mod = @import("dataframe_options.zig");
const validity_mod = @import("dataframe_validity.zig");

const DeviceColumn = dataframe_device_column_mod.DeviceColumn;
const DeviceTypedColumn = dataframe_device_column_mod.DeviceTypedColumn;
const DeviceClipOptions = options_mod.DeviceClipOptions;
const DeviceRollingOptions = options_mod.DeviceRollingOptions;
const DeviceExpandingOptions = options_mod.DeviceExpandingOptions;
const castToF64 = numeric_mod.castToF64;
const validityValues = validity_mod.validityValues;

pub const ClipMetrics = clip_metrics_mod.ClipMetrics;
pub const ClipSummaryMetrics = clip_metrics_mod.ClipSummaryMetrics;
pub const ClipProfileColumnCount = clip_metrics_mod.ClipProfileColumnCount;
pub const RollingClipProfileColumnCount = clip_metrics_mod.RollingClipProfileColumnCount;
pub const ExpandingClipProfileColumnCount = clip_metrics_mod.ExpandingClipProfileColumnCount;
pub const clipProfileOutputNames = clip_metrics_mod.clipProfileOutputNames;
pub const rollingClipProfileOutputNames = clip_metrics_mod.rollingClipProfileOutputNames;
pub const expandingClipProfileOutputNames = clip_metrics_mod.expandingClipProfileOutputNames;
pub const clipProfile = clip_metrics_mod.clipProfile;
pub const rollingClipProfile = clip_metrics_mod.rollingClipProfile;
pub const expandingClipProfile = clip_metrics_mod.expandingClipProfile;

pub fn clipProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceClipOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![ClipProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| clipProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| clipProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| clipProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| clipProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| clipProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| clipProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| clipProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| clipProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| clipProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| clipProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| clipProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| clipProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| clipProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn clipProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceClipOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![ClipProfileColumnCount]DeviceColumn {
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);

    var metrics = try clipProfile(allocator, values, maybe_validity, options_value.lower, options_value.upper);
    defer metrics.deinit();

    var columns: [ClipProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.clipped, metrics.validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(bool, allocator, metrics.below, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(bool, allocator, metrics.above, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(bool, allocator, metrics.in_range, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn rollingClipProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    clip_options: DeviceClipOptions,
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![RollingClipProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| rollingClipProfileColumnsTyped(i8, allocator, typed, clip_options, options_value, device_value),
        .i16 => |typed| rollingClipProfileColumnsTyped(i16, allocator, typed, clip_options, options_value, device_value),
        .i32 => |typed| rollingClipProfileColumnsTyped(i32, allocator, typed, clip_options, options_value, device_value),
        .i64 => |typed| rollingClipProfileColumnsTyped(i64, allocator, typed, clip_options, options_value, device_value),
        .u8 => |typed| rollingClipProfileColumnsTyped(u8, allocator, typed, clip_options, options_value, device_value),
        .u16 => |typed| rollingClipProfileColumnsTyped(u16, allocator, typed, clip_options, options_value, device_value),
        .u32 => |typed| rollingClipProfileColumnsTyped(u32, allocator, typed, clip_options, options_value, device_value),
        .u64 => |typed| rollingClipProfileColumnsTyped(u64, allocator, typed, clip_options, options_value, device_value),
        .usize => |typed| rollingClipProfileColumnsTyped(usize, allocator, typed, clip_options, options_value, device_value),
        .isize => |typed| rollingClipProfileColumnsTyped(isize, allocator, typed, clip_options, options_value, device_value),
        .f16 => |typed| rollingClipProfileColumnsTyped(f16, allocator, typed, clip_options, options_value, device_value),
        .f32 => |typed| rollingClipProfileColumnsTyped(f32, allocator, typed, clip_options, options_value, device_value),
        .f64 => |typed| rollingClipProfileColumnsTyped(f64, allocator, typed, clip_options, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn rollingClipProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    clip_options: DeviceClipOptions,
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![RollingClipProfileColumnCount]DeviceColumn {
    const min_periods = options_value.min_periods orelse options_value.window;
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);

    var metrics = try rollingClipProfile(allocator, values, maybe_validity, clip_options.lower, clip_options.upper, options_value.window, min_periods);
    defer metrics.deinit();

    var columns: [RollingClipProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.mean_clipped, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.clipped_rates, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.below_rates, metrics.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.above_rates, metrics.validity, device_value);
    initialized += 1;
    columns[5] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.in_range_rates, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn expandingClipProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    clip_options: DeviceClipOptions,
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingClipProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| expandingClipProfileColumnsTyped(i8, allocator, typed, clip_options, options_value, device_value),
        .i16 => |typed| expandingClipProfileColumnsTyped(i16, allocator, typed, clip_options, options_value, device_value),
        .i32 => |typed| expandingClipProfileColumnsTyped(i32, allocator, typed, clip_options, options_value, device_value),
        .i64 => |typed| expandingClipProfileColumnsTyped(i64, allocator, typed, clip_options, options_value, device_value),
        .u8 => |typed| expandingClipProfileColumnsTyped(u8, allocator, typed, clip_options, options_value, device_value),
        .u16 => |typed| expandingClipProfileColumnsTyped(u16, allocator, typed, clip_options, options_value, device_value),
        .u32 => |typed| expandingClipProfileColumnsTyped(u32, allocator, typed, clip_options, options_value, device_value),
        .u64 => |typed| expandingClipProfileColumnsTyped(u64, allocator, typed, clip_options, options_value, device_value),
        .usize => |typed| expandingClipProfileColumnsTyped(usize, allocator, typed, clip_options, options_value, device_value),
        .isize => |typed| expandingClipProfileColumnsTyped(isize, allocator, typed, clip_options, options_value, device_value),
        .f16 => |typed| expandingClipProfileColumnsTyped(f16, allocator, typed, clip_options, options_value, device_value),
        .f32 => |typed| expandingClipProfileColumnsTyped(f32, allocator, typed, clip_options, options_value, device_value),
        .f64 => |typed| expandingClipProfileColumnsTyped(f64, allocator, typed, clip_options, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn expandingClipProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    clip_options: DeviceClipOptions,
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingClipProfileColumnCount]DeviceColumn {
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);

    var metrics = try expandingClipProfile(allocator, values, maybe_validity, clip_options.lower, clip_options.upper, options_value.min_periods);
    defer metrics.deinit();

    var columns: [ExpandingClipProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.mean_clipped, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.clipped_rates, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.below_rates, metrics.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.above_rates, metrics.validity, device_value);
    initialized += 1;
    columns[5] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.in_range_rates, metrics.validity, device_value);
    initialized += 1;
    return columns;
}

const ClipFrameError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    TypeUnsupported,
    UnsupportedType,
    InvalidCsv,
    EmptyDataFrame,
    InvalidDevice,
};

fn appendClipColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    source_names: []const []const u8,
    clip_columns: anytype,
) ClipFrameError!DeviceDataFrame {
    const DeviceColumnType = std.meta.Elem(@TypeOf(frame.columns));
    var columns = try frame.allocator.alloc(DeviceColumnType, frame.columns.len + clip_columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }
    for (frame.columns, 0..) |col, i| {
        columns[i] = try col.clone();
        initialized += 1;
    }
    for (&clip_columns) |*clip_col| {
        columns[initialized] = clip_col.*;
        initialized += 1;
    }
    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, source_names, columns, frame.rows, frame.device);
}

fn clipFrameFromColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    output_prefix: []const u8,
    clip_columns_value: anytype,
    comptime namesFn: anytype,
) ClipFrameError!DeviceDataFrame {
    var clip_columns = clip_columns_value;
    var clip_columns_transferred: usize = 0;
    errdefer {
        for (clip_columns[clip_columns_transferred..]) |*col| col.deinit();
    }

    const source_names = try frame.allocator.alloc([]const u8, frame.columns.len + clip_columns.len);
    defer frame.allocator.free(source_names);
    for (frame.names, 0..) |source_name, i| source_names[i] = source_name;

    var clip_names = try namesFn(frame.allocator, output_prefix);
    defer names_mod.freeOwnedNameItems(frame.allocator, clip_names[0..]);
    for (clip_names, 0..) |clip_name, i| source_names[frame.columns.len + i] = clip_name;

    const out = try appendClipColumns(DeviceDataFrame, frame, source_names, clip_columns);
    clip_columns_transferred = clip_columns.len;
    return out;
}

pub fn clipProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceClipOptions,
) ClipFrameError!DeviceDataFrame {
    const clip_value = try frame.column(name);
    const clip_columns = try clipProfileColumnsByValue(frame.allocator, clip_value.*, options_value, frame.device, frame.rows);
    return clipFrameFromColumns(DeviceDataFrame, frame, output_prefix, clip_columns, clipProfileOutputNames);
}

pub fn rollingClipProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    clip_options: DeviceClipOptions,
    options_value: DeviceRollingOptions,
) ClipFrameError!DeviceDataFrame {
    const clip_value = try frame.column(name);
    const clip_columns = try rollingClipProfileColumnsByValue(frame.allocator, clip_value.*, clip_options, options_value, frame.device, frame.rows);
    return clipFrameFromColumns(DeviceDataFrame, frame, output_prefix, clip_columns, rollingClipProfileOutputNames);
}

pub fn expandingClipProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    clip_options: DeviceClipOptions,
    options_value: DeviceExpandingOptions,
) ClipFrameError!DeviceDataFrame {
    const clip_value = try frame.column(name);
    const clip_columns = try expandingClipProfileColumnsByValue(frame.allocator, clip_value.*, clip_options, options_value, frame.device, frame.rows);
    return clipFrameFromColumns(DeviceDataFrame, frame, output_prefix, clip_columns, expandingClipProfileOutputNames);
}
