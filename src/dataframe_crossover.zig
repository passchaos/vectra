const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const names_mod = @import("dataframe_names.zig");
const dataframe_device_column_mod = @import("dataframe_device_column.zig");
const metrics_mod = @import("dataframe_crossover_metrics.zig");
const numeric_mod = @import("dataframe_numeric.zig");
const options_mod = @import("dataframe_options.zig");
const validity_mod = @import("dataframe_validity.zig");

const DeviceColumn = dataframe_device_column_mod.DeviceColumn;
const DeviceTypedColumn = dataframe_device_column_mod.DeviceTypedColumn;
const DeviceCrossoverOptions = options_mod.DeviceCrossoverOptions;
const DeviceRollingOptions = options_mod.DeviceRollingOptions;
const DeviceExpandingOptions = options_mod.DeviceExpandingOptions;
const castToF64 = numeric_mod.castToF64;
const validityValues = validity_mod.validityValues;

pub const CrossoverMetrics = metrics_mod.CrossoverMetrics;
pub const CrossoverSummaryMetrics = metrics_mod.CrossoverSummaryMetrics;
pub const CrossoverProfileColumnCount = metrics_mod.CrossoverProfileColumnCount;
pub const RollingCrossoverProfileColumnCount = metrics_mod.RollingCrossoverProfileColumnCount;
pub const ExpandingCrossoverProfileColumnCount = metrics_mod.ExpandingCrossoverProfileColumnCount;
pub const crossoverProfileOutputNames = metrics_mod.crossoverProfileOutputNames;
pub const rollingCrossoverProfileOutputNames = metrics_mod.rollingCrossoverProfileOutputNames;
pub const expandingCrossoverProfileOutputNames = metrics_mod.expandingCrossoverProfileOutputNames;
pub const crossoverProfile = metrics_mod.crossoverProfile;
pub const rollingCrossoverProfile = metrics_mod.rollingCrossoverProfile;
pub const expandingCrossoverProfile = metrics_mod.expandingCrossoverProfile;

pub fn crossoverProfileColumnsByValue(
    allocator: std.mem.Allocator,
    lhs: DeviceColumn,
    rhs: DeviceColumn,
    options_value: DeviceCrossoverOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, InvalidDevice })![CrossoverProfileColumnCount]DeviceColumn {
    if (lhs.len() != rows or rhs.len() != rows) return error.LengthMismatch;
    if (lhs.dtype() != rhs.dtype()) return error.TypeMismatch;
    return switch (lhs) {
        .i8 => |typed| crossoverProfileColumnsTyped(i8, allocator, typed, rhs.i8, options_value, device_value),
        .i16 => |typed| crossoverProfileColumnsTyped(i16, allocator, typed, rhs.i16, options_value, device_value),
        .i32 => |typed| crossoverProfileColumnsTyped(i32, allocator, typed, rhs.i32, options_value, device_value),
        .i64 => |typed| crossoverProfileColumnsTyped(i64, allocator, typed, rhs.i64, options_value, device_value),
        .u8 => |typed| crossoverProfileColumnsTyped(u8, allocator, typed, rhs.u8, options_value, device_value),
        .u16 => |typed| crossoverProfileColumnsTyped(u16, allocator, typed, rhs.u16, options_value, device_value),
        .u32 => |typed| crossoverProfileColumnsTyped(u32, allocator, typed, rhs.u32, options_value, device_value),
        .u64 => |typed| crossoverProfileColumnsTyped(u64, allocator, typed, rhs.u64, options_value, device_value),
        .usize => |typed| crossoverProfileColumnsTyped(usize, allocator, typed, rhs.usize, options_value, device_value),
        .isize => |typed| crossoverProfileColumnsTyped(isize, allocator, typed, rhs.isize, options_value, device_value),
        .f16 => |typed| crossoverProfileColumnsTyped(f16, allocator, typed, rhs.f16, options_value, device_value),
        .f32 => |typed| crossoverProfileColumnsTyped(f32, allocator, typed, rhs.f32, options_value, device_value),
        .f64 => |typed| crossoverProfileColumnsTyped(f64, allocator, typed, rhs.f64, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn crossoverProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    lhs: DeviceTypedColumn(T),
    rhs: DeviceTypedColumn(T),
    options_value: DeviceCrossoverOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, InvalidDevice })![CrossoverProfileColumnCount]DeviceColumn {
    if (lhs.len() != rhs.len()) return error.LengthMismatch;
    if (!lhs.device().sameDevice(rhs.device())) return error.InvalidDevice;

    const lhs_values_typed = try lhs.values.toOwnedSlice(allocator);
    defer allocator.free(lhs_values_typed);
    const rhs_values_typed = try rhs.values.toOwnedSlice(allocator);
    defer allocator.free(rhs_values_typed);
    const maybe_lhs_validity = try validityValues(lhs, allocator);
    defer if (maybe_lhs_validity) |validity| allocator.free(validity);
    const maybe_rhs_validity = try validityValues(rhs, allocator);
    defer if (maybe_rhs_validity) |validity| allocator.free(validity);

    const rows = lhs_values_typed.len;
    const lhs_values = try allocator.alloc(f64, rows);
    defer allocator.free(lhs_values);
    const rhs_values = try allocator.alloc(f64, rows);
    defer allocator.free(rhs_values);
    for (lhs_values_typed, rhs_values_typed, 0..) |lhs_value, rhs_value, row| {
        lhs_values[row] = castToF64(T, lhs_value);
        rhs_values[row] = castToF64(T, rhs_value);
    }

    var metrics = try crossoverProfile(allocator, lhs_values, rhs_values, maybe_lhs_validity, maybe_rhs_validity, options_value.periods);
    defer metrics.deinit();

    var columns: [CrossoverProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.spreads, metrics.spread_validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.ratios, metrics.spread_validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(bool, allocator, metrics.cross_above, metrics.cross_validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(bool, allocator, metrics.cross_below, metrics.cross_validity, device_value);
    initialized += 1;
    return columns;
}
pub fn rollingCrossoverProfileColumnsByValue(
    allocator: std.mem.Allocator,
    lhs: DeviceColumn,
    rhs: DeviceColumn,
    cross_options: DeviceCrossoverOptions,
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, InvalidDevice })![RollingCrossoverProfileColumnCount]DeviceColumn {
    if (lhs.len() != rows or rhs.len() != rows) return error.LengthMismatch;
    if (lhs.dtype() != rhs.dtype()) return error.TypeMismatch;
    return switch (lhs) {
        .i8 => |typed| rollingCrossoverProfileColumnsTyped(i8, allocator, typed, rhs.i8, cross_options, options_value, device_value),
        .i16 => |typed| rollingCrossoverProfileColumnsTyped(i16, allocator, typed, rhs.i16, cross_options, options_value, device_value),
        .i32 => |typed| rollingCrossoverProfileColumnsTyped(i32, allocator, typed, rhs.i32, cross_options, options_value, device_value),
        .i64 => |typed| rollingCrossoverProfileColumnsTyped(i64, allocator, typed, rhs.i64, cross_options, options_value, device_value),
        .u8 => |typed| rollingCrossoverProfileColumnsTyped(u8, allocator, typed, rhs.u8, cross_options, options_value, device_value),
        .u16 => |typed| rollingCrossoverProfileColumnsTyped(u16, allocator, typed, rhs.u16, cross_options, options_value, device_value),
        .u32 => |typed| rollingCrossoverProfileColumnsTyped(u32, allocator, typed, rhs.u32, cross_options, options_value, device_value),
        .u64 => |typed| rollingCrossoverProfileColumnsTyped(u64, allocator, typed, rhs.u64, cross_options, options_value, device_value),
        .usize => |typed| rollingCrossoverProfileColumnsTyped(usize, allocator, typed, rhs.usize, cross_options, options_value, device_value),
        .isize => |typed| rollingCrossoverProfileColumnsTyped(isize, allocator, typed, rhs.isize, cross_options, options_value, device_value),
        .f16 => |typed| rollingCrossoverProfileColumnsTyped(f16, allocator, typed, rhs.f16, cross_options, options_value, device_value),
        .f32 => |typed| rollingCrossoverProfileColumnsTyped(f32, allocator, typed, rhs.f32, cross_options, options_value, device_value),
        .f64 => |typed| rollingCrossoverProfileColumnsTyped(f64, allocator, typed, rhs.f64, cross_options, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn rollingCrossoverProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    lhs: DeviceTypedColumn(T),
    rhs: DeviceTypedColumn(T),
    cross_options: DeviceCrossoverOptions,
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, InvalidDevice })![RollingCrossoverProfileColumnCount]DeviceColumn {
    const min_periods = options_value.min_periods orelse options_value.window;
    if (lhs.len() != rhs.len()) return error.LengthMismatch;
    if (!lhs.device().sameDevice(rhs.device())) return error.InvalidDevice;

    const lhs_values_typed = try lhs.values.toOwnedSlice(allocator);
    defer allocator.free(lhs_values_typed);
    const rhs_values_typed = try rhs.values.toOwnedSlice(allocator);
    defer allocator.free(rhs_values_typed);
    const maybe_lhs_validity = try validityValues(lhs, allocator);
    defer if (maybe_lhs_validity) |validity| allocator.free(validity);
    const maybe_rhs_validity = try validityValues(rhs, allocator);
    defer if (maybe_rhs_validity) |validity| allocator.free(validity);

    const rows = lhs_values_typed.len;
    const lhs_values = try allocator.alloc(f64, rows);
    defer allocator.free(lhs_values);
    const rhs_values = try allocator.alloc(f64, rows);
    defer allocator.free(rhs_values);
    for (lhs_values_typed, rhs_values_typed, 0..) |lhs_value, rhs_value, row| {
        lhs_values[row] = castToF64(T, lhs_value);
        rhs_values[row] = castToF64(T, rhs_value);
    }

    var metrics = try rollingCrossoverProfile(
        allocator,
        lhs_values,
        rhs_values,
        maybe_lhs_validity,
        maybe_rhs_validity,
        cross_options.periods,
        options_value.window,
        min_periods,
    );
    defer metrics.deinit();

    var columns: [RollingCrossoverProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSlice(i64, allocator, metrics.cross_above_counts, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSlice(i64, allocator, metrics.cross_below_counts, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.cross_above_rates, metrics.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.cross_below_rates, metrics.validity, device_value);
    initialized += 1;
    columns[5] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.mean_abs_spreads, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn expandingCrossoverProfileColumnsByValue(
    allocator: std.mem.Allocator,
    lhs: DeviceColumn,
    rhs: DeviceColumn,
    cross_options: DeviceCrossoverOptions,
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, InvalidDevice })![ExpandingCrossoverProfileColumnCount]DeviceColumn {
    if (lhs.len() != rows or rhs.len() != rows) return error.LengthMismatch;
    if (lhs.dtype() != rhs.dtype()) return error.TypeMismatch;
    return switch (lhs) {
        .i8 => |typed| expandingCrossoverProfileColumnsTyped(i8, allocator, typed, rhs.i8, cross_options, options_value, device_value),
        .i16 => |typed| expandingCrossoverProfileColumnsTyped(i16, allocator, typed, rhs.i16, cross_options, options_value, device_value),
        .i32 => |typed| expandingCrossoverProfileColumnsTyped(i32, allocator, typed, rhs.i32, cross_options, options_value, device_value),
        .i64 => |typed| expandingCrossoverProfileColumnsTyped(i64, allocator, typed, rhs.i64, cross_options, options_value, device_value),
        .u8 => |typed| expandingCrossoverProfileColumnsTyped(u8, allocator, typed, rhs.u8, cross_options, options_value, device_value),
        .u16 => |typed| expandingCrossoverProfileColumnsTyped(u16, allocator, typed, rhs.u16, cross_options, options_value, device_value),
        .u32 => |typed| expandingCrossoverProfileColumnsTyped(u32, allocator, typed, rhs.u32, cross_options, options_value, device_value),
        .u64 => |typed| expandingCrossoverProfileColumnsTyped(u64, allocator, typed, rhs.u64, cross_options, options_value, device_value),
        .usize => |typed| expandingCrossoverProfileColumnsTyped(usize, allocator, typed, rhs.usize, cross_options, options_value, device_value),
        .isize => |typed| expandingCrossoverProfileColumnsTyped(isize, allocator, typed, rhs.isize, cross_options, options_value, device_value),
        .f16 => |typed| expandingCrossoverProfileColumnsTyped(f16, allocator, typed, rhs.f16, cross_options, options_value, device_value),
        .f32 => |typed| expandingCrossoverProfileColumnsTyped(f32, allocator, typed, rhs.f32, cross_options, options_value, device_value),
        .f64 => |typed| expandingCrossoverProfileColumnsTyped(f64, allocator, typed, rhs.f64, cross_options, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn expandingCrossoverProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    lhs: DeviceTypedColumn(T),
    rhs: DeviceTypedColumn(T),
    cross_options: DeviceCrossoverOptions,
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, InvalidDevice })![ExpandingCrossoverProfileColumnCount]DeviceColumn {
    if (lhs.len() != rhs.len()) return error.LengthMismatch;
    if (!lhs.device().sameDevice(rhs.device())) return error.InvalidDevice;

    const lhs_values_typed = try lhs.values.toOwnedSlice(allocator);
    defer allocator.free(lhs_values_typed);
    const rhs_values_typed = try rhs.values.toOwnedSlice(allocator);
    defer allocator.free(rhs_values_typed);
    const maybe_lhs_validity = try validityValues(lhs, allocator);
    defer if (maybe_lhs_validity) |validity| allocator.free(validity);
    const maybe_rhs_validity = try validityValues(rhs, allocator);
    defer if (maybe_rhs_validity) |validity| allocator.free(validity);

    const rows = lhs_values_typed.len;
    const lhs_values = try allocator.alloc(f64, rows);
    defer allocator.free(lhs_values);
    const rhs_values = try allocator.alloc(f64, rows);
    defer allocator.free(rhs_values);
    for (lhs_values_typed, rhs_values_typed, 0..) |lhs_value, rhs_value, row| {
        lhs_values[row] = castToF64(T, lhs_value);
        rhs_values[row] = castToF64(T, rhs_value);
    }

    var metrics = try expandingCrossoverProfile(
        allocator,
        lhs_values,
        rhs_values,
        maybe_lhs_validity,
        maybe_rhs_validity,
        cross_options.periods,
        options_value.min_periods,
    );
    defer metrics.deinit();

    var columns: [ExpandingCrossoverProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSlice(i64, allocator, metrics.cross_above_counts, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSlice(i64, allocator, metrics.cross_below_counts, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.cross_above_rates, metrics.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.cross_below_rates, metrics.validity, device_value);
    initialized += 1;
    columns[5] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.mean_abs_spreads, metrics.validity, device_value);
    initialized += 1;
    return columns;
}

const CrossoverFrameError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    TypeUnsupported,
    UnsupportedType,
    InvalidCsv,
    EmptyDataFrame,
    InvalidDevice,
};

fn appendCrossoverColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    source_names: []const []const u8,
    cross_columns: anytype,
) CrossoverFrameError!DeviceDataFrame {
    const DeviceColumnType = std.meta.Elem(@TypeOf(frame.columns));
    var columns = try frame.allocator.alloc(DeviceColumnType, frame.columns.len + cross_columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }
    for (frame.columns, 0..) |col, i| {
        columns[i] = try col.clone();
        initialized += 1;
    }
    for (&cross_columns) |*cross_col| {
        columns[initialized] = cross_col.*;
        initialized += 1;
    }
    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, source_names, columns, frame.rows, frame.device);
}

fn crossoverFrameFromColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    output_prefix: []const u8,
    cross_columns_value: anytype,
    comptime namesFn: anytype,
) CrossoverFrameError!DeviceDataFrame {
    var cross_columns = cross_columns_value;
    var cross_columns_transferred: usize = 0;
    errdefer {
        for (cross_columns[cross_columns_transferred..]) |*col| col.deinit();
    }

    const source_names = try frame.allocator.alloc([]const u8, frame.columns.len + cross_columns.len);
    defer frame.allocator.free(source_names);
    for (frame.names, 0..) |source_name, i| source_names[i] = source_name;

    var cross_names = try namesFn(frame.allocator, output_prefix);
    defer names_mod.freeOwnedNameItems(frame.allocator, cross_names[0..]);
    for (cross_names, 0..) |cross_name, i| source_names[frame.columns.len + i] = cross_name;

    const out = try appendCrossoverColumns(DeviceDataFrame, frame, source_names, cross_columns);
    cross_columns_transferred = cross_columns.len;
    return out;
}

pub fn crossoverProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    lhs_name: []const u8,
    rhs_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceCrossoverOptions,
) CrossoverFrameError!DeviceDataFrame {
    const lhs = try frame.column(lhs_name);
    const rhs = try frame.column(rhs_name);
    if (lhs.dtype() != rhs.dtype()) return error.TypeMismatch;
    const cross_columns = try crossoverProfileColumnsByValue(frame.allocator, lhs.*, rhs.*, options_value, frame.device, frame.rows);
    return crossoverFrameFromColumns(DeviceDataFrame, frame, output_prefix, cross_columns, crossoverProfileOutputNames);
}

pub fn rollingCrossoverProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    lhs_name: []const u8,
    rhs_name: []const u8,
    output_prefix: []const u8,
    cross_options: DeviceCrossoverOptions,
    options_value: DeviceRollingOptions,
) CrossoverFrameError!DeviceDataFrame {
    const lhs = try frame.column(lhs_name);
    const rhs = try frame.column(rhs_name);
    if (lhs.dtype() != rhs.dtype()) return error.TypeMismatch;
    const cross_columns = try rollingCrossoverProfileColumnsByValue(frame.allocator, lhs.*, rhs.*, cross_options, options_value, frame.device, frame.rows);
    return crossoverFrameFromColumns(DeviceDataFrame, frame, output_prefix, cross_columns, rollingCrossoverProfileOutputNames);
}

pub fn expandingCrossoverProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    lhs_name: []const u8,
    rhs_name: []const u8,
    output_prefix: []const u8,
    cross_options: DeviceCrossoverOptions,
    options_value: DeviceExpandingOptions,
) CrossoverFrameError!DeviceDataFrame {
    const lhs = try frame.column(lhs_name);
    const rhs = try frame.column(rhs_name);
    if (lhs.dtype() != rhs.dtype()) return error.TypeMismatch;
    const cross_columns = try expandingCrossoverProfileColumnsByValue(frame.allocator, lhs.*, rhs.*, cross_options, options_value, frame.device, frame.rows);
    return crossoverFrameFromColumns(DeviceDataFrame, frame, output_prefix, cross_columns, expandingCrossoverProfileOutputNames);
}
