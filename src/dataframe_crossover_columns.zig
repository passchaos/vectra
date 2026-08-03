//! Crossover profile column materializers.

const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_device_column_mod = @import("dataframe/device_column.zig");
const metrics_mod = @import("dataframe_crossover_metrics.zig");
const numeric_mod = @import("dataframe_numeric.zig");
const options_mod = @import("dataframe_options.zig");
const validity_mod = @import("dataframe/validity.zig");

const DeviceColumn = dataframe_device_column_mod.DeviceColumn;
const DeviceTypedColumn = dataframe_device_column_mod.DeviceTypedColumn;
const DeviceCrossoverOptions = options_mod.DeviceCrossoverOptions;
const DeviceRollingOptions = options_mod.DeviceRollingOptions;
const DeviceExpandingOptions = options_mod.DeviceExpandingOptions;
const castToF64 = numeric_mod.castToF64;
const validityValues = validity_mod.validityValues;
const CrossoverProfileColumnCount = metrics_mod.CrossoverProfileColumnCount;
const RollingCrossoverProfileColumnCount = metrics_mod.RollingCrossoverProfileColumnCount;
const ExpandingCrossoverProfileColumnCount = metrics_mod.ExpandingCrossoverProfileColumnCount;
const crossoverProfile = metrics_mod.crossoverProfile;
const rollingCrossoverProfile = metrics_mod.rollingCrossoverProfile;
const expandingCrossoverProfile = metrics_mod.expandingCrossoverProfile;

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
