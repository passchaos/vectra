const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const names_mod = @import("dataframe_names.zig");
const dataframe_device_column_mod = @import("dataframe_device_column.zig");
const robust_metrics_mod = @import("dataframe_robust_metrics.zig");
const numeric_mod = @import("dataframe_numeric.zig");
const options_mod = @import("dataframe_options.zig");
const validity_mod = @import("dataframe_validity.zig");

const DeviceColumn = dataframe_device_column_mod.DeviceColumn;
const DeviceTypedColumn = dataframe_device_column_mod.DeviceTypedColumn;
const DeviceRollingRobustOptions = options_mod.DeviceRollingRobustOptions;
const DeviceRobustOptions = options_mod.DeviceRobustOptions;
const castToF64 = numeric_mod.castToF64;
const validityValues = validity_mod.validityValues;

const RobustFrameError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    TypeUnsupported,
    UnsupportedType,
    InvalidCsv,
    EmptyDataFrame,
    InvalidDevice,
};

pub const RobustMetrics = robust_metrics_mod.RobustMetrics;
pub const RobustProfileColumnCount = robust_metrics_mod.RobustProfileColumnCount;
pub const RollingRobustProfileColumnCount = robust_metrics_mod.RollingRobustProfileColumnCount;
pub const ExpandingRobustProfileColumnCount = robust_metrics_mod.ExpandingRobustProfileColumnCount;
pub const robustProfileOutputNames = robust_metrics_mod.robustProfileOutputNames;
pub const rollingRobustProfileOutputNames = robust_metrics_mod.rollingRobustProfileOutputNames;
pub const expandingRobustProfileOutputNames = robust_metrics_mod.expandingRobustProfileOutputNames;
pub const robustProfile = robust_metrics_mod.robustProfile;
pub const rollingRobustProfile = robust_metrics_mod.rollingRobustProfile;
pub const expandingRobustProfile = robust_metrics_mod.expandingRobustProfile;

pub fn rollingRobustProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceRollingRobustOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![RollingRobustProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| rollingRobustProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| rollingRobustProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| rollingRobustProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| rollingRobustProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| rollingRobustProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| rollingRobustProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| rollingRobustProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| rollingRobustProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| rollingRobustProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| rollingRobustProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| rollingRobustProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| rollingRobustProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| rollingRobustProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn rollingRobustProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceRollingRobustOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![RollingRobustProfileColumnCount]DeviceColumn {
    const min_periods = options_value.min_periods orelse options_value.window;
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);
    var metrics = try rollingRobustProfile(allocator, values, maybe_validity, options_value.window, min_periods, options_value.iqr_multiplier);
    defer metrics.deinit();

    var columns: [RollingRobustProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.centered, metrics.validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.mad_zscore, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(bool, allocator, metrics.outlier, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.winsorized, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn expandingRobustProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceRobustOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingRobustProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| expandingRobustProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| expandingRobustProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| expandingRobustProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| expandingRobustProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| expandingRobustProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| expandingRobustProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| expandingRobustProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| expandingRobustProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| expandingRobustProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| expandingRobustProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| expandingRobustProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| expandingRobustProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| expandingRobustProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn expandingRobustProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceRobustOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingRobustProfileColumnCount]DeviceColumn {
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);
    var metrics = try expandingRobustProfile(allocator, values, maybe_validity, options_value.min_periods, options_value.iqr_multiplier);
    defer metrics.deinit();

    var columns: [ExpandingRobustProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.centered, metrics.validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.mad_zscore, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(bool, allocator, metrics.outlier, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.winsorized, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn robustProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceRobustOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![RobustProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| robustProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| robustProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| robustProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| robustProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| robustProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| robustProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| robustProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| robustProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| robustProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| robustProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| robustProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| robustProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| robustProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn robustProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceRobustOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![RobustProfileColumnCount]DeviceColumn {
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);
    var metrics = try robustProfile(allocator, values, maybe_validity, options_value.min_periods, options_value.iqr_multiplier);
    defer metrics.deinit();

    var columns: [RobustProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.centered, metrics.validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.mad_zscore, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(bool, allocator, metrics.outlier, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.winsorized, metrics.validity, device_value);
    initialized += 1;
    return columns;
}

pub fn robustProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceRobustOptions,
) RobustFrameError!DeviceDataFrame {
    const robust_value = try frame.column(name);
    var robust_columns = try robustProfileColumnsByValue(frame.allocator, robust_value.*, options_value, frame.device, frame.rows);
    var robust_columns_transferred: usize = 0;
    errdefer {
        for (robust_columns[robust_columns_transferred..]) |*col| col.deinit();
    }

    const source_names = try frame.allocator.alloc([]const u8, frame.columns.len + robust_columns.len);
    defer frame.allocator.free(source_names);
    for (frame.names, 0..) |source_name, i| source_names[i] = source_name;

    var robust_names = try robustProfileOutputNames(frame.allocator, output_prefix);
    defer names_mod.freeOwnedNameItems(frame.allocator, robust_names[0..]);
    for (robust_names, 0..) |robust_name, i| source_names[frame.columns.len + i] = robust_name;

    const DeviceColumnType = std.meta.Elem(@TypeOf(frame.columns));
    var columns = try frame.allocator.alloc(DeviceColumnType, frame.columns.len + robust_columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }
    for (frame.columns, 0..) |col, i| {
        columns[i] = try col.clone();
        initialized += 1;
    }
    for (&robust_columns) |*robust_col| {
        columns[initialized] = robust_col.*;
        initialized += 1;
        robust_columns_transferred += 1;
    }

    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, source_names, columns, frame.rows, frame.device);
}

fn robustFrameFromColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    output_prefix: []const u8,
    robust_columns_value: anytype,
    comptime namesFn: anytype,
) RobustFrameError!DeviceDataFrame {
    var robust_columns = robust_columns_value;
    var robust_columns_transferred: usize = 0;
    errdefer {
        for (robust_columns[robust_columns_transferred..]) |*col| col.deinit();
    }

    const source_names = try frame.allocator.alloc([]const u8, frame.columns.len + robust_columns.len);
    defer frame.allocator.free(source_names);
    for (frame.names, 0..) |source_name, i| source_names[i] = source_name;

    var robust_names = try namesFn(frame.allocator, output_prefix);
    defer names_mod.freeOwnedNameItems(frame.allocator, robust_names[0..]);
    for (robust_names, 0..) |robust_name, i| source_names[frame.columns.len + i] = robust_name;

    const DeviceColumnType = std.meta.Elem(@TypeOf(frame.columns));
    var columns = try frame.allocator.alloc(DeviceColumnType, frame.columns.len + robust_columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }
    for (frame.columns, 0..) |col, i| {
        columns[i] = try col.clone();
        initialized += 1;
    }
    for (&robust_columns) |*robust_col| {
        columns[initialized] = robust_col.*;
        initialized += 1;
        robust_columns_transferred += 1;
    }

    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, source_names, columns, frame.rows, frame.device);
}

pub fn rollingRobustProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceRollingRobustOptions,
) RobustFrameError!DeviceDataFrame {
    const rolling_value = try frame.column(name);
    const rolling_columns = try rollingRobustProfileColumnsByValue(frame.allocator, rolling_value.*, options_value, frame.device, frame.rows);
    return robustFrameFromColumns(DeviceDataFrame, frame, output_prefix, rolling_columns, rollingRobustProfileOutputNames);
}

pub fn expandingRobustProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceRobustOptions,
) RobustFrameError!DeviceDataFrame {
    const expanding_value = try frame.column(name);
    const expanding_columns = try expandingRobustProfileColumnsByValue(frame.allocator, expanding_value.*, options_value, frame.device, frame.rows);
    return robustFrameFromColumns(DeviceDataFrame, frame, output_prefix, expanding_columns, expandingRobustProfileOutputNames);
}
