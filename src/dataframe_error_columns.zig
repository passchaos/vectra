//! Column materializers for prediction-error dataframe profiles.

const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_device_column_mod = @import("dataframe/device_column.zig");
const error_metrics_mod = @import("dataframe_error_metrics.zig");
const numeric_mod = @import("dataframe_numeric.zig");
const options_mod = @import("dataframe_options.zig");
const validity_mod = @import("dataframe_validity_core.zig");

const DeviceColumn = dataframe_device_column_mod.DeviceColumn;
const DeviceTypedColumn = dataframe_device_column_mod.DeviceTypedColumn;
const DeviceRollingOptions = options_mod.DeviceRollingOptions;
const DeviceExpandingOptions = options_mod.DeviceExpandingOptions;
const castToF64 = numeric_mod.castToF64;
const validityValues = validity_mod.validityValues;

pub const ErrorProfileColumnCount = error_metrics_mod.ErrorProfileColumnCount;
pub const RollingErrorProfileColumnCount = error_metrics_mod.RollingErrorProfileColumnCount;
pub const ExpandingErrorProfileColumnCount = error_metrics_mod.ExpandingErrorProfileColumnCount;
pub const errorProfile = error_metrics_mod.errorProfile;
pub const rollingErrorProfile = error_metrics_mod.rollingErrorProfile;
pub const expandingErrorProfile = error_metrics_mod.expandingErrorProfile;

pub fn errorProfileColumnsByValue(
    allocator: std.mem.Allocator,
    actual: DeviceColumn,
    predicted: DeviceColumn,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, InvalidDevice })![ErrorProfileColumnCount]DeviceColumn {
    if (actual.len() != rows or predicted.len() != rows) return error.LengthMismatch;
    if (actual.dtype() != predicted.dtype()) return error.TypeMismatch;
    return switch (actual) {
        .i8 => |typed| errorProfileColumnsTyped(i8, allocator, typed, predicted.i8, device_value),
        .i16 => |typed| errorProfileColumnsTyped(i16, allocator, typed, predicted.i16, device_value),
        .i32 => |typed| errorProfileColumnsTyped(i32, allocator, typed, predicted.i32, device_value),
        .i64 => |typed| errorProfileColumnsTyped(i64, allocator, typed, predicted.i64, device_value),
        .u8 => |typed| errorProfileColumnsTyped(u8, allocator, typed, predicted.u8, device_value),
        .u16 => |typed| errorProfileColumnsTyped(u16, allocator, typed, predicted.u16, device_value),
        .u32 => |typed| errorProfileColumnsTyped(u32, allocator, typed, predicted.u32, device_value),
        .u64 => |typed| errorProfileColumnsTyped(u64, allocator, typed, predicted.u64, device_value),
        .usize => |typed| errorProfileColumnsTyped(usize, allocator, typed, predicted.usize, device_value),
        .isize => |typed| errorProfileColumnsTyped(isize, allocator, typed, predicted.isize, device_value),
        .f16 => |typed| errorProfileColumnsTyped(f16, allocator, typed, predicted.f16, device_value),
        .f32 => |typed| errorProfileColumnsTyped(f32, allocator, typed, predicted.f32, device_value),
        .f64 => |typed| errorProfileColumnsTyped(f64, allocator, typed, predicted.f64, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn errorProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    actual_column: DeviceTypedColumn(T),
    predicted_column: DeviceTypedColumn(T),
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, InvalidDevice })![ErrorProfileColumnCount]DeviceColumn {
    if (actual_column.len() != predicted_column.len()) return error.LengthMismatch;
    if (!actual_column.device().sameDevice(predicted_column.device())) return error.InvalidDevice;

    const actual_values_typed = try actual_column.values.toOwnedSlice(allocator);
    defer allocator.free(actual_values_typed);
    const predicted_values_typed = try predicted_column.values.toOwnedSlice(allocator);
    defer allocator.free(predicted_values_typed);
    const maybe_actual_validity = try validityValues(actual_column, allocator);
    defer if (maybe_actual_validity) |validity| allocator.free(validity);
    const maybe_predicted_validity = try validityValues(predicted_column, allocator);
    defer if (maybe_predicted_validity) |validity| allocator.free(validity);

    const rows = actual_values_typed.len;
    const actual_values = try allocator.alloc(f64, rows);
    defer allocator.free(actual_values);
    const predicted_values = try allocator.alloc(f64, rows);
    defer allocator.free(predicted_values);
    for (actual_values_typed, predicted_values_typed, 0..) |actual_value, predicted_value, row| {
        actual_values[row] = castToF64(T, actual_value);
        predicted_values[row] = castToF64(T, predicted_value);
    }

    var metrics = try errorProfile(allocator, actual_values, predicted_values, maybe_actual_validity, maybe_predicted_validity);
    defer metrics.deinit();

    var columns: [ErrorProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.errors, metrics.validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.abs_errors, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.squared_errors, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.ape, metrics.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.smape, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn rollingErrorProfileColumnsByValue(
    allocator: std.mem.Allocator,
    actual: DeviceColumn,
    predicted: DeviceColumn,
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, InvalidDevice })![RollingErrorProfileColumnCount]DeviceColumn {
    if (actual.len() != rows or predicted.len() != rows) return error.LengthMismatch;
    if (actual.dtype() != predicted.dtype()) return error.TypeMismatch;
    return switch (actual) {
        .i8 => |typed| rollingErrorProfileColumnsTyped(i8, allocator, typed, predicted.i8, options_value, device_value),
        .i16 => |typed| rollingErrorProfileColumnsTyped(i16, allocator, typed, predicted.i16, options_value, device_value),
        .i32 => |typed| rollingErrorProfileColumnsTyped(i32, allocator, typed, predicted.i32, options_value, device_value),
        .i64 => |typed| rollingErrorProfileColumnsTyped(i64, allocator, typed, predicted.i64, options_value, device_value),
        .u8 => |typed| rollingErrorProfileColumnsTyped(u8, allocator, typed, predicted.u8, options_value, device_value),
        .u16 => |typed| rollingErrorProfileColumnsTyped(u16, allocator, typed, predicted.u16, options_value, device_value),
        .u32 => |typed| rollingErrorProfileColumnsTyped(u32, allocator, typed, predicted.u32, options_value, device_value),
        .u64 => |typed| rollingErrorProfileColumnsTyped(u64, allocator, typed, predicted.u64, options_value, device_value),
        .usize => |typed| rollingErrorProfileColumnsTyped(usize, allocator, typed, predicted.usize, options_value, device_value),
        .isize => |typed| rollingErrorProfileColumnsTyped(isize, allocator, typed, predicted.isize, options_value, device_value),
        .f16 => |typed| rollingErrorProfileColumnsTyped(f16, allocator, typed, predicted.f16, options_value, device_value),
        .f32 => |typed| rollingErrorProfileColumnsTyped(f32, allocator, typed, predicted.f32, options_value, device_value),
        .f64 => |typed| rollingErrorProfileColumnsTyped(f64, allocator, typed, predicted.f64, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn rollingErrorProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    actual_column: DeviceTypedColumn(T),
    predicted_column: DeviceTypedColumn(T),
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, InvalidDevice })![RollingErrorProfileColumnCount]DeviceColumn {
    const min_periods = options_value.min_periods orelse options_value.window;
    if (actual_column.len() != predicted_column.len()) return error.LengthMismatch;
    if (!actual_column.device().sameDevice(predicted_column.device())) return error.InvalidDevice;

    const actual_values_typed = try actual_column.values.toOwnedSlice(allocator);
    defer allocator.free(actual_values_typed);
    const predicted_values_typed = try predicted_column.values.toOwnedSlice(allocator);
    defer allocator.free(predicted_values_typed);
    const maybe_actual_validity = try validityValues(actual_column, allocator);
    defer if (maybe_actual_validity) |validity| allocator.free(validity);
    const maybe_predicted_validity = try validityValues(predicted_column, allocator);
    defer if (maybe_predicted_validity) |validity| allocator.free(validity);

    const rows = actual_values_typed.len;
    const actual_values = try allocator.alloc(f64, rows);
    defer allocator.free(actual_values);
    const predicted_values = try allocator.alloc(f64, rows);
    defer allocator.free(predicted_values);
    for (actual_values_typed, predicted_values_typed, 0..) |actual_value, predicted_value, row| {
        actual_values[row] = castToF64(T, actual_value);
        predicted_values[row] = castToF64(T, predicted_value);
    }

    var metrics = try rollingErrorProfile(
        allocator,
        actual_values,
        predicted_values,
        maybe_actual_validity,
        maybe_predicted_validity,
        options_value.window,
        min_periods,
    );
    defer metrics.deinit();

    var columns: [RollingErrorProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.mae, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.rmse, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.mape, metrics.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.smape, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn expandingErrorProfileColumnsByValue(
    allocator: std.mem.Allocator,
    actual: DeviceColumn,
    predicted: DeviceColumn,
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, InvalidDevice })![ExpandingErrorProfileColumnCount]DeviceColumn {
    if (actual.len() != rows or predicted.len() != rows) return error.LengthMismatch;
    if (actual.dtype() != predicted.dtype()) return error.TypeMismatch;
    return switch (actual) {
        .i8 => |typed| expandingErrorProfileColumnsTyped(i8, allocator, typed, predicted.i8, options_value, device_value),
        .i16 => |typed| expandingErrorProfileColumnsTyped(i16, allocator, typed, predicted.i16, options_value, device_value),
        .i32 => |typed| expandingErrorProfileColumnsTyped(i32, allocator, typed, predicted.i32, options_value, device_value),
        .i64 => |typed| expandingErrorProfileColumnsTyped(i64, allocator, typed, predicted.i64, options_value, device_value),
        .u8 => |typed| expandingErrorProfileColumnsTyped(u8, allocator, typed, predicted.u8, options_value, device_value),
        .u16 => |typed| expandingErrorProfileColumnsTyped(u16, allocator, typed, predicted.u16, options_value, device_value),
        .u32 => |typed| expandingErrorProfileColumnsTyped(u32, allocator, typed, predicted.u32, options_value, device_value),
        .u64 => |typed| expandingErrorProfileColumnsTyped(u64, allocator, typed, predicted.u64, options_value, device_value),
        .usize => |typed| expandingErrorProfileColumnsTyped(usize, allocator, typed, predicted.usize, options_value, device_value),
        .isize => |typed| expandingErrorProfileColumnsTyped(isize, allocator, typed, predicted.isize, options_value, device_value),
        .f16 => |typed| expandingErrorProfileColumnsTyped(f16, allocator, typed, predicted.f16, options_value, device_value),
        .f32 => |typed| expandingErrorProfileColumnsTyped(f32, allocator, typed, predicted.f32, options_value, device_value),
        .f64 => |typed| expandingErrorProfileColumnsTyped(f64, allocator, typed, predicted.f64, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn expandingErrorProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    actual_column: DeviceTypedColumn(T),
    predicted_column: DeviceTypedColumn(T),
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, InvalidDevice })![ExpandingErrorProfileColumnCount]DeviceColumn {
    if (actual_column.len() != predicted_column.len()) return error.LengthMismatch;
    if (!actual_column.device().sameDevice(predicted_column.device())) return error.InvalidDevice;

    const actual_values_typed = try actual_column.values.toOwnedSlice(allocator);
    defer allocator.free(actual_values_typed);
    const predicted_values_typed = try predicted_column.values.toOwnedSlice(allocator);
    defer allocator.free(predicted_values_typed);
    const maybe_actual_validity = try validityValues(actual_column, allocator);
    defer if (maybe_actual_validity) |validity| allocator.free(validity);
    const maybe_predicted_validity = try validityValues(predicted_column, allocator);
    defer if (maybe_predicted_validity) |validity| allocator.free(validity);

    const rows = actual_values_typed.len;
    const actual_values = try allocator.alloc(f64, rows);
    defer allocator.free(actual_values);
    const predicted_values = try allocator.alloc(f64, rows);
    defer allocator.free(predicted_values);
    for (actual_values_typed, predicted_values_typed, 0..) |actual_value, predicted_value, row| {
        actual_values[row] = castToF64(T, actual_value);
        predicted_values[row] = castToF64(T, predicted_value);
    }

    var metrics = try expandingErrorProfile(
        allocator,
        actual_values,
        predicted_values,
        maybe_actual_validity,
        maybe_predicted_validity,
        options_value.min_periods,
    );
    defer metrics.deinit();

    var columns: [ExpandingErrorProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.mae, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.rmse, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.mape, metrics.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.smape, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
