const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const names_mod = @import("dataframe_names.zig");
const dataframe_device_column_mod = @import("dataframe/device_column.zig");
const change_metrics_mod = @import("dataframe_change_metrics.zig");
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

pub const ChangePointMetrics = change_metrics_mod.ChangePointMetrics;
pub const ChangeSummaryMetrics = change_metrics_mod.ChangeSummaryMetrics;
pub const ChangePointProfileColumnCount = change_metrics_mod.ChangePointProfileColumnCount;
pub const RollingChangePointProfileColumnCount = change_metrics_mod.RollingChangePointProfileColumnCount;
pub const ExpandingChangePointProfileColumnCount = change_metrics_mod.ExpandingChangePointProfileColumnCount;
pub const changePointProfileOutputNames = change_metrics_mod.changePointProfileOutputNames;
pub const rollingChangePointProfileOutputNames = change_metrics_mod.rollingChangePointProfileOutputNames;
pub const expandingChangePointProfileOutputNames = change_metrics_mod.expandingChangePointProfileOutputNames;
pub const changePointProfile = change_metrics_mod.changePointProfile;
pub const rollingChangePointProfile = change_metrics_mod.rollingChangePointProfile;
pub const expandingChangePointProfile = change_metrics_mod.expandingChangePointProfile;

pub fn changePointProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    threshold: f64,
    options_value: DeviceTrendOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![ChangePointProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| changePointProfileColumnsTyped(i8, allocator, typed, threshold, options_value, device_value),
        .i16 => |typed| changePointProfileColumnsTyped(i16, allocator, typed, threshold, options_value, device_value),
        .i32 => |typed| changePointProfileColumnsTyped(i32, allocator, typed, threshold, options_value, device_value),
        .i64 => |typed| changePointProfileColumnsTyped(i64, allocator, typed, threshold, options_value, device_value),
        .u8 => |typed| changePointProfileColumnsTyped(u8, allocator, typed, threshold, options_value, device_value),
        .u16 => |typed| changePointProfileColumnsTyped(u16, allocator, typed, threshold, options_value, device_value),
        .u32 => |typed| changePointProfileColumnsTyped(u32, allocator, typed, threshold, options_value, device_value),
        .u64 => |typed| changePointProfileColumnsTyped(u64, allocator, typed, threshold, options_value, device_value),
        .usize => |typed| changePointProfileColumnsTyped(usize, allocator, typed, threshold, options_value, device_value),
        .isize => |typed| changePointProfileColumnsTyped(isize, allocator, typed, threshold, options_value, device_value),
        .f16 => |typed| changePointProfileColumnsTyped(f16, allocator, typed, threshold, options_value, device_value),
        .f32 => |typed| changePointProfileColumnsTyped(f32, allocator, typed, threshold, options_value, device_value),
        .f64 => |typed| changePointProfileColumnsTyped(f64, allocator, typed, threshold, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn changePointProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    threshold: f64,
    options_value: DeviceTrendOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![ChangePointProfileColumnCount]DeviceColumn {
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);
    var metrics = try changePointProfile(allocator, values, maybe_validity, threshold, options_value.periods);
    defer metrics.deinit();

    var columns: [ChangePointProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.deltas, metrics.validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.abs_deltas, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.pct_changes, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(bool, allocator, metrics.change_points, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn rollingChangePointProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    threshold: f64,
    change_options: DeviceTrendOptions,
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![RollingChangePointProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| rollingChangePointProfileColumnsTyped(i8, allocator, typed, threshold, change_options, options_value, device_value),
        .i16 => |typed| rollingChangePointProfileColumnsTyped(i16, allocator, typed, threshold, change_options, options_value, device_value),
        .i32 => |typed| rollingChangePointProfileColumnsTyped(i32, allocator, typed, threshold, change_options, options_value, device_value),
        .i64 => |typed| rollingChangePointProfileColumnsTyped(i64, allocator, typed, threshold, change_options, options_value, device_value),
        .u8 => |typed| rollingChangePointProfileColumnsTyped(u8, allocator, typed, threshold, change_options, options_value, device_value),
        .u16 => |typed| rollingChangePointProfileColumnsTyped(u16, allocator, typed, threshold, change_options, options_value, device_value),
        .u32 => |typed| rollingChangePointProfileColumnsTyped(u32, allocator, typed, threshold, change_options, options_value, device_value),
        .u64 => |typed| rollingChangePointProfileColumnsTyped(u64, allocator, typed, threshold, change_options, options_value, device_value),
        .usize => |typed| rollingChangePointProfileColumnsTyped(usize, allocator, typed, threshold, change_options, options_value, device_value),
        .isize => |typed| rollingChangePointProfileColumnsTyped(isize, allocator, typed, threshold, change_options, options_value, device_value),
        .f16 => |typed| rollingChangePointProfileColumnsTyped(f16, allocator, typed, threshold, change_options, options_value, device_value),
        .f32 => |typed| rollingChangePointProfileColumnsTyped(f32, allocator, typed, threshold, change_options, options_value, device_value),
        .f64 => |typed| rollingChangePointProfileColumnsTyped(f64, allocator, typed, threshold, change_options, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn rollingChangePointProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    threshold: f64,
    change_options: DeviceTrendOptions,
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![RollingChangePointProfileColumnCount]DeviceColumn {
    const min_periods = options_value.min_periods orelse options_value.window;
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);
    var metrics = try rollingChangePointProfile(allocator, values, maybe_validity, threshold, change_options.periods, options_value.window, min_periods);
    defer metrics.deinit();

    var columns: [RollingChangePointProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSlice(i64, allocator, metrics.change_counts, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.change_rates, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.mean_abs_delta, metrics.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.max_abs_delta, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn expandingChangePointProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    threshold: f64,
    change_options: DeviceTrendOptions,
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingChangePointProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| expandingChangePointProfileColumnsTyped(i8, allocator, typed, threshold, change_options, options_value, device_value),
        .i16 => |typed| expandingChangePointProfileColumnsTyped(i16, allocator, typed, threshold, change_options, options_value, device_value),
        .i32 => |typed| expandingChangePointProfileColumnsTyped(i32, allocator, typed, threshold, change_options, options_value, device_value),
        .i64 => |typed| expandingChangePointProfileColumnsTyped(i64, allocator, typed, threshold, change_options, options_value, device_value),
        .u8 => |typed| expandingChangePointProfileColumnsTyped(u8, allocator, typed, threshold, change_options, options_value, device_value),
        .u16 => |typed| expandingChangePointProfileColumnsTyped(u16, allocator, typed, threshold, change_options, options_value, device_value),
        .u32 => |typed| expandingChangePointProfileColumnsTyped(u32, allocator, typed, threshold, change_options, options_value, device_value),
        .u64 => |typed| expandingChangePointProfileColumnsTyped(u64, allocator, typed, threshold, change_options, options_value, device_value),
        .usize => |typed| expandingChangePointProfileColumnsTyped(usize, allocator, typed, threshold, change_options, options_value, device_value),
        .isize => |typed| expandingChangePointProfileColumnsTyped(isize, allocator, typed, threshold, change_options, options_value, device_value),
        .f16 => |typed| expandingChangePointProfileColumnsTyped(f16, allocator, typed, threshold, change_options, options_value, device_value),
        .f32 => |typed| expandingChangePointProfileColumnsTyped(f32, allocator, typed, threshold, change_options, options_value, device_value),
        .f64 => |typed| expandingChangePointProfileColumnsTyped(f64, allocator, typed, threshold, change_options, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn expandingChangePointProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    threshold: f64,
    change_options: DeviceTrendOptions,
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingChangePointProfileColumnCount]DeviceColumn {
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);
    var metrics = try expandingChangePointProfile(allocator, values, maybe_validity, threshold, change_options.periods, options_value.min_periods);
    defer metrics.deinit();

    var columns: [ExpandingChangePointProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSlice(i64, allocator, metrics.change_counts, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.change_rates, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.mean_abs_delta, metrics.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.max_abs_delta, metrics.validity, device_value);
    initialized += 1;
    return columns;
}

const ChangeFrameError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    TypeUnsupported,
    UnsupportedType,
    InvalidCsv,
    EmptyDataFrame,
    InvalidDevice,
};

fn appendChangeColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    source_names: []const []const u8,
    change_columns: anytype,
) ChangeFrameError!DeviceDataFrame {
    const DeviceColumnType = std.meta.Elem(@TypeOf(frame.columns));
    var columns = try frame.allocator.alloc(DeviceColumnType, frame.columns.len + change_columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }
    for (frame.columns, 0..) |col, i| {
        columns[i] = try col.clone();
        initialized += 1;
    }
    for (&change_columns) |*change_col| {
        columns[initialized] = change_col.*;
        initialized += 1;
    }
    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, source_names, columns, frame.rows, frame.device);
}

fn changePointFrameFromColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    output_prefix: []const u8,
    change_columns_value: anytype,
    comptime namesFn: anytype,
) ChangeFrameError!DeviceDataFrame {
    var change_columns = change_columns_value;
    var change_columns_transferred: usize = 0;
    errdefer {
        for (change_columns[change_columns_transferred..]) |*col| col.deinit();
    }

    const source_names = try frame.allocator.alloc([]const u8, frame.columns.len + change_columns.len);
    defer frame.allocator.free(source_names);
    for (frame.names, 0..) |source_name, i| source_names[i] = source_name;

    var change_names = try namesFn(frame.allocator, output_prefix);
    defer names_mod.freeOwnedNameItems(frame.allocator, change_names[0..]);
    for (change_names, 0..) |change_name, i| source_names[frame.columns.len + i] = change_name;

    const out = try appendChangeColumns(DeviceDataFrame, frame, source_names, change_columns);
    change_columns_transferred = change_columns.len;
    return out;
}

pub fn changePointProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    threshold: f64,
    options_value: DeviceTrendOptions,
) ChangeFrameError!DeviceDataFrame {
    const change_value = try frame.column(name);
    const change_columns = try changePointProfileColumnsByValue(frame.allocator, change_value.*, threshold, options_value, frame.device, frame.rows);
    return changePointFrameFromColumns(DeviceDataFrame, frame, output_prefix, change_columns, changePointProfileOutputNames);
}

pub fn rollingChangePointProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    threshold: f64,
    change_options: DeviceTrendOptions,
    options_value: DeviceRollingOptions,
) ChangeFrameError!DeviceDataFrame {
    const change_value = try frame.column(name);
    const change_columns = try rollingChangePointProfileColumnsByValue(frame.allocator, change_value.*, threshold, change_options, options_value, frame.device, frame.rows);
    return changePointFrameFromColumns(DeviceDataFrame, frame, output_prefix, change_columns, rollingChangePointProfileOutputNames);
}

pub fn expandingChangePointProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    threshold: f64,
    change_options: DeviceTrendOptions,
    options_value: DeviceExpandingOptions,
) ChangeFrameError!DeviceDataFrame {
    const change_value = try frame.column(name);
    const change_columns = try expandingChangePointProfileColumnsByValue(frame.allocator, change_value.*, threshold, change_options, options_value, frame.device, frame.rows);
    return changePointFrameFromColumns(DeviceDataFrame, frame, output_prefix, change_columns, expandingChangePointProfileOutputNames);
}
