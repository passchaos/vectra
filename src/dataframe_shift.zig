const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const names_mod = @import("dataframe_names.zig");
const dataframe_device_column_mod = @import("dataframe/device_column.zig");
const metrics_mod = @import("dataframe_shift_metrics.zig");
const numeric_mod = @import("dataframe_numeric.zig");
const options_mod = @import("dataframe_options.zig");
const validity_mod = @import("dataframe_validity.zig");

const DeviceColumn = dataframe_device_column_mod.DeviceColumn;
const DeviceTypedColumn = dataframe_device_column_mod.DeviceTypedColumn;
const DeviceLagOptions = options_mod.DeviceLagOptions;
const castToF64 = numeric_mod.castToF64;
const validityValues = validity_mod.validityValues;

pub const ShiftMetrics = metrics_mod.ShiftMetrics;
pub const LagProfileColumnCount = metrics_mod.LagProfileColumnCount;
pub const LeadProfileColumnCount = metrics_mod.LeadProfileColumnCount;
pub const lagProfileOutputNames = metrics_mod.lagProfileOutputNames;
pub const leadProfileOutputNames = metrics_mod.leadProfileOutputNames;
pub const lagProfile = metrics_mod.lagProfile;
pub const leadProfile = metrics_mod.leadProfile;

pub fn lagProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceLagOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![LagProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| lagProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| lagProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| lagProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| lagProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| lagProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| lagProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| lagProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| lagProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| lagProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| lagProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| lagProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| lagProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| lagProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn lagProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceLagOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![LagProfileColumnCount]DeviceColumn {
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);
    var metrics = try lagProfile(allocator, values, maybe_validity, options_value.periods);
    defer metrics.deinit();

    var columns: [LagProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.shifted, metrics.shift_validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.diff, metrics.change_validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.pct_change, metrics.change_validity, device_value);
    initialized += 1;
    return columns;
}
pub fn leadProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceLagOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![LeadProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| leadProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| leadProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| leadProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| leadProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| leadProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| leadProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| leadProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| leadProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| leadProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| leadProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| leadProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| leadProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| leadProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn leadProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceLagOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![LeadProfileColumnCount]DeviceColumn {
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);
    var metrics = try leadProfile(allocator, values, maybe_validity, options_value.periods);
    defer metrics.deinit();

    var columns: [LeadProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.shifted, metrics.shift_validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.diff, metrics.change_validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.pct_change, metrics.change_validity, device_value);
    initialized += 1;
    return columns;
}

const ShiftFrameError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    TypeUnsupported,
    UnsupportedType,
    InvalidCsv,
    EmptyDataFrame,
    InvalidDevice,
};

fn appendShiftColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    source_names: []const []const u8,
    shift_columns: anytype,
) ShiftFrameError!DeviceDataFrame {
    const DeviceColumnType = std.meta.Elem(@TypeOf(frame.columns));
    var columns = try frame.allocator.alloc(DeviceColumnType, frame.columns.len + shift_columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }
    for (frame.columns, 0..) |col, i| {
        columns[i] = try col.clone();
        initialized += 1;
    }
    for (&shift_columns) |*shift_col| {
        columns[initialized] = shift_col.*;
        initialized += 1;
    }
    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, source_names, columns, frame.rows, frame.device);
}

fn shiftFrameFromColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    output_prefix: []const u8,
    shift_columns_value: anytype,
    comptime namesFn: anytype,
) ShiftFrameError!DeviceDataFrame {
    var shift_columns = shift_columns_value;
    var shift_columns_transferred: usize = 0;
    errdefer {
        for (shift_columns[shift_columns_transferred..]) |*col| col.deinit();
    }

    const source_names = try frame.allocator.alloc([]const u8, frame.columns.len + shift_columns.len);
    defer frame.allocator.free(source_names);
    for (frame.names, 0..) |source_name, i| source_names[i] = source_name;

    var shift_names = try namesFn(frame.allocator, output_prefix);
    defer names_mod.freeOwnedNameItems(frame.allocator, shift_names[0..]);
    for (shift_names, 0..) |shift_name, i| source_names[frame.columns.len + i] = shift_name;

    const out = try appendShiftColumns(DeviceDataFrame, frame, source_names, shift_columns);
    shift_columns_transferred = shift_columns.len;
    return out;
}

pub fn lagProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceLagOptions,
) ShiftFrameError!DeviceDataFrame {
    const lag_value = try frame.column(name);
    const lag_columns = try lagProfileColumnsByValue(frame.allocator, lag_value.*, options_value, frame.device, frame.rows);
    return shiftFrameFromColumns(DeviceDataFrame, frame, output_prefix, lag_columns, lagProfileOutputNames);
}

pub fn leadProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceLagOptions,
) ShiftFrameError!DeviceDataFrame {
    const lead_value = try frame.column(name);
    const lead_columns = try leadProfileColumnsByValue(frame.allocator, lead_value.*, options_value, frame.device, frame.rows);
    return shiftFrameFromColumns(DeviceDataFrame, frame, output_prefix, lead_columns, leadProfileOutputNames);
}
