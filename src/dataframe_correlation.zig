const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const names_mod = @import("dataframe_names.zig");
const dataframe_device_column_mod = @import("dataframe/device_column.zig");
const correlation_metrics_mod = @import("dataframe_correlation_metrics.zig");
const numeric_mod = @import("dataframe_numeric.zig");
const options_mod = @import("dataframe_options.zig");
const validity_mod = @import("dataframe_validity.zig");

const DeviceColumn = dataframe_device_column_mod.DeviceColumn;
const DeviceTypedColumn = dataframe_device_column_mod.DeviceTypedColumn;
const DeviceRollingCorrelationOptions = options_mod.DeviceRollingCorrelationOptions;
const DeviceExpandingOptions = options_mod.DeviceExpandingOptions;
const castToF64 = numeric_mod.castToF64;
const validityValues = validity_mod.validityValues;

pub const CorrelationMetrics = correlation_metrics_mod.CorrelationMetrics;
pub const RollingCorrelationProfileColumnCount = correlation_metrics_mod.RollingCorrelationProfileColumnCount;
pub const ExpandingCorrelationProfileColumnCount = correlation_metrics_mod.ExpandingCorrelationProfileColumnCount;
pub const rollingCorrelationProfileOutputNames = correlation_metrics_mod.rollingCorrelationProfileOutputNames;
pub const expandingCorrelationProfileOutputNames = correlation_metrics_mod.expandingCorrelationProfileOutputNames;
pub const rollingCorrelationProfile = correlation_metrics_mod.rollingCorrelationProfile;
pub const expandingCorrelationProfile = correlation_metrics_mod.expandingCorrelationProfile;

pub fn rollingCorrelationProfileColumnsByValue(
    allocator: std.mem.Allocator,
    x: DeviceColumn,
    y: DeviceColumn,
    options_value: DeviceRollingCorrelationOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, InvalidDevice })![RollingCorrelationProfileColumnCount]DeviceColumn {
    if (x.len() != rows or y.len() != rows) return error.LengthMismatch;
    if (x.dtype() != y.dtype()) return error.TypeMismatch;
    return switch (x) {
        .i8 => |typed| rollingCorrelationProfileColumnsTyped(i8, allocator, typed, y.i8, options_value, device_value),
        .i16 => |typed| rollingCorrelationProfileColumnsTyped(i16, allocator, typed, y.i16, options_value, device_value),
        .i32 => |typed| rollingCorrelationProfileColumnsTyped(i32, allocator, typed, y.i32, options_value, device_value),
        .i64 => |typed| rollingCorrelationProfileColumnsTyped(i64, allocator, typed, y.i64, options_value, device_value),
        .u8 => |typed| rollingCorrelationProfileColumnsTyped(u8, allocator, typed, y.u8, options_value, device_value),
        .u16 => |typed| rollingCorrelationProfileColumnsTyped(u16, allocator, typed, y.u16, options_value, device_value),
        .u32 => |typed| rollingCorrelationProfileColumnsTyped(u32, allocator, typed, y.u32, options_value, device_value),
        .u64 => |typed| rollingCorrelationProfileColumnsTyped(u64, allocator, typed, y.u64, options_value, device_value),
        .usize => |typed| rollingCorrelationProfileColumnsTyped(usize, allocator, typed, y.usize, options_value, device_value),
        .isize => |typed| rollingCorrelationProfileColumnsTyped(isize, allocator, typed, y.isize, options_value, device_value),
        .f16 => |typed| rollingCorrelationProfileColumnsTyped(f16, allocator, typed, y.f16, options_value, device_value),
        .f32 => |typed| rollingCorrelationProfileColumnsTyped(f32, allocator, typed, y.f32, options_value, device_value),
        .f64 => |typed| rollingCorrelationProfileColumnsTyped(f64, allocator, typed, y.f64, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn rollingCorrelationProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    x_column: DeviceTypedColumn(T),
    y_column: DeviceTypedColumn(T),
    options_value: DeviceRollingCorrelationOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, InvalidDevice })![RollingCorrelationProfileColumnCount]DeviceColumn {
    const min_periods = options_value.min_periods orelse options_value.window;
    if (x_column.len() != y_column.len()) return error.LengthMismatch;
    if (!x_column.device().sameDevice(y_column.device())) return error.InvalidDevice;

    const xs_typed = try x_column.values.toOwnedSlice(allocator);
    defer allocator.free(xs_typed);
    const ys_typed = try y_column.values.toOwnedSlice(allocator);
    defer allocator.free(ys_typed);
    const maybe_x_validity = try validityValues(x_column, allocator);
    defer if (maybe_x_validity) |validity| allocator.free(validity);
    const maybe_y_validity = try validityValues(y_column, allocator);
    defer if (maybe_y_validity) |validity| allocator.free(validity);

    const rows = xs_typed.len;
    const xs = try allocator.alloc(f64, rows);
    defer allocator.free(xs);
    const ys = try allocator.alloc(f64, rows);
    defer allocator.free(ys);
    for (xs_typed, ys_typed, 0..) |x_value, y_value, row| {
        xs[row] = castToF64(T, x_value);
        ys[row] = castToF64(T, y_value);
    }

    var metrics = try rollingCorrelationProfile(
        allocator,
        xs,
        ys,
        maybe_x_validity,
        maybe_y_validity,
        options_value.window,
        min_periods,
    );
    defer metrics.deinit();

    var columns: [RollingCorrelationProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.pair_counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.covariances, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.correlations, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.betas, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn expandingCorrelationProfileColumnsByValue(
    allocator: std.mem.Allocator,
    x: DeviceColumn,
    y: DeviceColumn,
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, InvalidDevice })![ExpandingCorrelationProfileColumnCount]DeviceColumn {
    if (x.len() != rows or y.len() != rows) return error.LengthMismatch;
    if (x.dtype() != y.dtype()) return error.TypeMismatch;
    return switch (x) {
        .i8 => |typed| expandingCorrelationProfileColumnsTyped(i8, allocator, typed, y.i8, options_value, device_value),
        .i16 => |typed| expandingCorrelationProfileColumnsTyped(i16, allocator, typed, y.i16, options_value, device_value),
        .i32 => |typed| expandingCorrelationProfileColumnsTyped(i32, allocator, typed, y.i32, options_value, device_value),
        .i64 => |typed| expandingCorrelationProfileColumnsTyped(i64, allocator, typed, y.i64, options_value, device_value),
        .u8 => |typed| expandingCorrelationProfileColumnsTyped(u8, allocator, typed, y.u8, options_value, device_value),
        .u16 => |typed| expandingCorrelationProfileColumnsTyped(u16, allocator, typed, y.u16, options_value, device_value),
        .u32 => |typed| expandingCorrelationProfileColumnsTyped(u32, allocator, typed, y.u32, options_value, device_value),
        .u64 => |typed| expandingCorrelationProfileColumnsTyped(u64, allocator, typed, y.u64, options_value, device_value),
        .usize => |typed| expandingCorrelationProfileColumnsTyped(usize, allocator, typed, y.usize, options_value, device_value),
        .isize => |typed| expandingCorrelationProfileColumnsTyped(isize, allocator, typed, y.isize, options_value, device_value),
        .f16 => |typed| expandingCorrelationProfileColumnsTyped(f16, allocator, typed, y.f16, options_value, device_value),
        .f32 => |typed| expandingCorrelationProfileColumnsTyped(f32, allocator, typed, y.f32, options_value, device_value),
        .f64 => |typed| expandingCorrelationProfileColumnsTyped(f64, allocator, typed, y.f64, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn expandingCorrelationProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    x_column: DeviceTypedColumn(T),
    y_column: DeviceTypedColumn(T),
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, InvalidDevice })![ExpandingCorrelationProfileColumnCount]DeviceColumn {
    if (x_column.len() != y_column.len()) return error.LengthMismatch;
    if (!x_column.device().sameDevice(y_column.device())) return error.InvalidDevice;

    const xs_typed = try x_column.values.toOwnedSlice(allocator);
    defer allocator.free(xs_typed);
    const ys_typed = try y_column.values.toOwnedSlice(allocator);
    defer allocator.free(ys_typed);
    const maybe_x_validity = try validityValues(x_column, allocator);
    defer if (maybe_x_validity) |validity| allocator.free(validity);
    const maybe_y_validity = try validityValues(y_column, allocator);
    defer if (maybe_y_validity) |validity| allocator.free(validity);

    const rows = xs_typed.len;
    const xs = try allocator.alloc(f64, rows);
    defer allocator.free(xs);
    const ys = try allocator.alloc(f64, rows);
    defer allocator.free(ys);
    for (xs_typed, ys_typed, 0..) |x_value, y_value, row| {
        xs[row] = castToF64(T, x_value);
        ys[row] = castToF64(T, y_value);
    }

    var metrics = try expandingCorrelationProfile(
        allocator,
        xs,
        ys,
        maybe_x_validity,
        maybe_y_validity,
        options_value.min_periods,
    );
    defer metrics.deinit();

    var columns: [ExpandingCorrelationProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.pair_counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.covariances, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.correlations, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.betas, metrics.validity, device_value);
    initialized += 1;
    return columns;
}

const CorrelationFrameError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    TypeUnsupported,
    UnsupportedType,
    InvalidCsv,
    EmptyDataFrame,
    InvalidDevice,
};

fn appendCorrelationColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    source_names: []const []const u8,
    corr_columns: anytype,
) CorrelationFrameError!DeviceDataFrame {
    const DeviceColumnType = std.meta.Elem(@TypeOf(frame.columns));
    var columns = try frame.allocator.alloc(DeviceColumnType, frame.columns.len + corr_columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }
    for (frame.columns, 0..) |col, i| {
        columns[i] = try col.clone();
        initialized += 1;
    }
    for (&corr_columns) |*corr_col| {
        columns[initialized] = corr_col.*;
        initialized += 1;
    }
    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, source_names, columns, frame.rows, frame.device);
}

fn correlationFrameFromColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    output_prefix: []const u8,
    corr_columns_value: anytype,
    comptime namesFn: anytype,
) CorrelationFrameError!DeviceDataFrame {
    var corr_columns = corr_columns_value;
    var corr_columns_transferred: usize = 0;
    errdefer {
        for (corr_columns[corr_columns_transferred..]) |*col| col.deinit();
    }

    const source_names = try frame.allocator.alloc([]const u8, frame.columns.len + corr_columns.len);
    defer frame.allocator.free(source_names);
    for (frame.names, 0..) |source_name, i| source_names[i] = source_name;

    var corr_names = try namesFn(frame.allocator, output_prefix);
    defer names_mod.freeOwnedNameItems(frame.allocator, corr_names[0..]);
    for (corr_names, 0..) |corr_name, i| source_names[frame.columns.len + i] = corr_name;

    const out = try appendCorrelationColumns(DeviceDataFrame, frame, source_names, corr_columns);
    corr_columns_transferred = corr_columns.len;
    return out;
}

fn validateCorrelationInputs(frame: anytype, x_name: []const u8, y_name: []const u8) CorrelationFrameError!struct { x: @TypeOf(frame.column(x_name) catch unreachable), y: @TypeOf(frame.column(y_name) catch unreachable) } {
    const x = try frame.column(x_name);
    const y = try frame.column(y_name);
    if (x.dtype() != y.dtype()) return error.TypeMismatch;
    return .{ .x = x, .y = y };
}

pub fn rollingCorrelationProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    x_name: []const u8,
    y_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceRollingCorrelationOptions,
) CorrelationFrameError!DeviceDataFrame {
    const inputs = try validateCorrelationInputs(frame, x_name, y_name);
    const corr_columns = try rollingCorrelationProfileColumnsByValue(frame.allocator, inputs.x.*, inputs.y.*, options_value, frame.device, frame.rows);
    return correlationFrameFromColumns(DeviceDataFrame, frame, output_prefix, corr_columns, rollingCorrelationProfileOutputNames);
}

pub fn expandingCorrelationProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    x_name: []const u8,
    y_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceExpandingOptions,
) CorrelationFrameError!DeviceDataFrame {
    const inputs = try validateCorrelationInputs(frame, x_name, y_name);
    const corr_columns = try expandingCorrelationProfileColumnsByValue(frame.allocator, inputs.x.*, inputs.y.*, options_value, frame.device, frame.rows);
    return correlationFrameFromColumns(DeviceDataFrame, frame, output_prefix, corr_columns, expandingCorrelationProfileOutputNames);
}
