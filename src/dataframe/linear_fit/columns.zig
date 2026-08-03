//! Column materializers for linear-fit dataframe profiles.

const std = @import("std");
const array_mod = @import("../../array.zig");
const dataframe_device_column_mod = @import("../device_column.zig");
const metrics_mod = @import("metrics.zig");
const numeric_mod = @import("../../dataframe_numeric.zig");
const options_mod = @import("../../dataframe_options.zig");
const validity_mod = @import("../validity.zig");

const DeviceColumn = dataframe_device_column_mod.DeviceColumn;
const DeviceTypedColumn = dataframe_device_column_mod.DeviceTypedColumn;
const DeviceLinearFitOptions = options_mod.DeviceLinearFitOptions;
const DeviceExpandingOptions = options_mod.DeviceExpandingOptions;
const DeviceRollingCorrelationOptions = options_mod.DeviceRollingCorrelationOptions;
const castToF64 = numeric_mod.castToF64;
const validityValues = validity_mod.validityValues;

pub const LinearFitProfileColumnCount = metrics_mod.LinearFitProfileColumnCount;
pub const ExpandingLinearFitProfileColumnCount = metrics_mod.ExpandingLinearFitProfileColumnCount;
pub const RollingLinearFitProfileColumnCount = metrics_mod.RollingLinearFitProfileColumnCount;
pub const linearFitProfile = metrics_mod.linearFitProfile;
pub const expandingLinearFitProfile = metrics_mod.expandingLinearFitProfile;
pub const rollingLinearFitProfile = metrics_mod.rollingLinearFitProfile;

pub fn linearFitProfileColumnsByValue(
    allocator: std.mem.Allocator,
    x: DeviceColumn,
    y: DeviceColumn,
    options_value: DeviceLinearFitOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, InvalidDevice })![LinearFitProfileColumnCount]DeviceColumn {
    if (x.len() != rows or y.len() != rows) return error.LengthMismatch;
    if (x.dtype() != y.dtype()) return error.TypeMismatch;
    return switch (x) {
        .i8 => |typed| linearFitProfileColumnsTyped(i8, allocator, typed, y.i8, options_value, device_value),
        .i16 => |typed| linearFitProfileColumnsTyped(i16, allocator, typed, y.i16, options_value, device_value),
        .i32 => |typed| linearFitProfileColumnsTyped(i32, allocator, typed, y.i32, options_value, device_value),
        .i64 => |typed| linearFitProfileColumnsTyped(i64, allocator, typed, y.i64, options_value, device_value),
        .u8 => |typed| linearFitProfileColumnsTyped(u8, allocator, typed, y.u8, options_value, device_value),
        .u16 => |typed| linearFitProfileColumnsTyped(u16, allocator, typed, y.u16, options_value, device_value),
        .u32 => |typed| linearFitProfileColumnsTyped(u32, allocator, typed, y.u32, options_value, device_value),
        .u64 => |typed| linearFitProfileColumnsTyped(u64, allocator, typed, y.u64, options_value, device_value),
        .usize => |typed| linearFitProfileColumnsTyped(usize, allocator, typed, y.usize, options_value, device_value),
        .isize => |typed| linearFitProfileColumnsTyped(isize, allocator, typed, y.isize, options_value, device_value),
        .f16 => |typed| linearFitProfileColumnsTyped(f16, allocator, typed, y.f16, options_value, device_value),
        .f32 => |typed| linearFitProfileColumnsTyped(f32, allocator, typed, y.f32, options_value, device_value),
        .f64 => |typed| linearFitProfileColumnsTyped(f64, allocator, typed, y.f64, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn linearFitProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    x_column: DeviceTypedColumn(T),
    y_column: DeviceTypedColumn(T),
    options_value: DeviceLinearFitOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, InvalidDevice })![LinearFitProfileColumnCount]DeviceColumn {
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

    var metrics = try linearFitProfile(allocator, xs, ys, maybe_x_validity, maybe_y_validity, options_value.min_periods);
    defer metrics.deinit();

    var columns: [LinearFitProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.fitted, metrics.validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.residuals, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.residual_z, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.slopes, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn expandingLinearFitProfileColumnsByValue(
    allocator: std.mem.Allocator,
    x: DeviceColumn,
    y: DeviceColumn,
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, InvalidDevice })![ExpandingLinearFitProfileColumnCount]DeviceColumn {
    if (x.len() != rows or y.len() != rows) return error.LengthMismatch;
    if (x.dtype() != y.dtype()) return error.TypeMismatch;
    return switch (x) {
        .i8 => |typed| expandingLinearFitProfileColumnsTyped(i8, allocator, typed, y.i8, options_value, device_value),
        .i16 => |typed| expandingLinearFitProfileColumnsTyped(i16, allocator, typed, y.i16, options_value, device_value),
        .i32 => |typed| expandingLinearFitProfileColumnsTyped(i32, allocator, typed, y.i32, options_value, device_value),
        .i64 => |typed| expandingLinearFitProfileColumnsTyped(i64, allocator, typed, y.i64, options_value, device_value),
        .u8 => |typed| expandingLinearFitProfileColumnsTyped(u8, allocator, typed, y.u8, options_value, device_value),
        .u16 => |typed| expandingLinearFitProfileColumnsTyped(u16, allocator, typed, y.u16, options_value, device_value),
        .u32 => |typed| expandingLinearFitProfileColumnsTyped(u32, allocator, typed, y.u32, options_value, device_value),
        .u64 => |typed| expandingLinearFitProfileColumnsTyped(u64, allocator, typed, y.u64, options_value, device_value),
        .usize => |typed| expandingLinearFitProfileColumnsTyped(usize, allocator, typed, y.usize, options_value, device_value),
        .isize => |typed| expandingLinearFitProfileColumnsTyped(isize, allocator, typed, y.isize, options_value, device_value),
        .f16 => |typed| expandingLinearFitProfileColumnsTyped(f16, allocator, typed, y.f16, options_value, device_value),
        .f32 => |typed| expandingLinearFitProfileColumnsTyped(f32, allocator, typed, y.f32, options_value, device_value),
        .f64 => |typed| expandingLinearFitProfileColumnsTyped(f64, allocator, typed, y.f64, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn expandingLinearFitProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    x_column: DeviceTypedColumn(T),
    y_column: DeviceTypedColumn(T),
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, InvalidDevice })![ExpandingLinearFitProfileColumnCount]DeviceColumn {
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

    var metrics = try expandingLinearFitProfile(allocator, xs, ys, maybe_x_validity, maybe_y_validity, options_value.min_periods);
    defer metrics.deinit();

    var columns: [ExpandingLinearFitProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.pair_counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.slopes, metrics.fit_validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.intercepts, metrics.fit_validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.fitted, metrics.row_validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.residuals, metrics.row_validity, device_value);
    initialized += 1;
    columns[5] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.residual_z, metrics.row_validity, device_value);
    initialized += 1;
    return columns;
}
pub fn rollingLinearFitProfileColumnsByValue(
    allocator: std.mem.Allocator,
    x: DeviceColumn,
    y: DeviceColumn,
    options_value: DeviceRollingCorrelationOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, InvalidDevice })![RollingLinearFitProfileColumnCount]DeviceColumn {
    if (x.len() != rows or y.len() != rows) return error.LengthMismatch;
    if (x.dtype() != y.dtype()) return error.TypeMismatch;
    return switch (x) {
        .i8 => |typed| rollingLinearFitProfileColumnsTyped(i8, allocator, typed, y.i8, options_value, device_value),
        .i16 => |typed| rollingLinearFitProfileColumnsTyped(i16, allocator, typed, y.i16, options_value, device_value),
        .i32 => |typed| rollingLinearFitProfileColumnsTyped(i32, allocator, typed, y.i32, options_value, device_value),
        .i64 => |typed| rollingLinearFitProfileColumnsTyped(i64, allocator, typed, y.i64, options_value, device_value),
        .u8 => |typed| rollingLinearFitProfileColumnsTyped(u8, allocator, typed, y.u8, options_value, device_value),
        .u16 => |typed| rollingLinearFitProfileColumnsTyped(u16, allocator, typed, y.u16, options_value, device_value),
        .u32 => |typed| rollingLinearFitProfileColumnsTyped(u32, allocator, typed, y.u32, options_value, device_value),
        .u64 => |typed| rollingLinearFitProfileColumnsTyped(u64, allocator, typed, y.u64, options_value, device_value),
        .usize => |typed| rollingLinearFitProfileColumnsTyped(usize, allocator, typed, y.usize, options_value, device_value),
        .isize => |typed| rollingLinearFitProfileColumnsTyped(isize, allocator, typed, y.isize, options_value, device_value),
        .f16 => |typed| rollingLinearFitProfileColumnsTyped(f16, allocator, typed, y.f16, options_value, device_value),
        .f32 => |typed| rollingLinearFitProfileColumnsTyped(f32, allocator, typed, y.f32, options_value, device_value),
        .f64 => |typed| rollingLinearFitProfileColumnsTyped(f64, allocator, typed, y.f64, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn rollingLinearFitProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    x_column: DeviceTypedColumn(T),
    y_column: DeviceTypedColumn(T),
    options_value: DeviceRollingCorrelationOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, InvalidDevice })![RollingLinearFitProfileColumnCount]DeviceColumn {
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

    var metrics = try rollingLinearFitProfile(
        allocator,
        xs,
        ys,
        maybe_x_validity,
        maybe_y_validity,
        options_value.window,
        min_periods,
    );
    defer metrics.deinit();

    var columns: [RollingLinearFitProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.pair_counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.slopes, metrics.fit_validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.intercepts, metrics.fit_validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.fitted, metrics.row_validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.residuals, metrics.row_validity, device_value);
    initialized += 1;
    columns[5] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.residual_z, metrics.row_validity, device_value);
    initialized += 1;
    return columns;
}
