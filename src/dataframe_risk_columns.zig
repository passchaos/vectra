//! DeviceColumn builders for risk profile metrics.

const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_device_column_mod = @import("dataframe_device_column.zig");
const risk_metrics_mod = @import("dataframe_risk_metrics.zig");
const numeric_mod = @import("dataframe_numeric.zig");
const options_mod = @import("dataframe_options.zig");
const validity_mod = @import("dataframe_validity.zig");

const DeviceColumn = dataframe_device_column_mod.DeviceColumn;
const DeviceTypedColumn = dataframe_device_column_mod.DeviceTypedColumn;
const DeviceRollingOptions = options_mod.DeviceRollingOptions;
const DeviceDrawdownOptions = options_mod.DeviceDrawdownOptions;
const DeviceExtremaOptions = options_mod.DeviceExtremaOptions;
const castToF64 = numeric_mod.castToF64;
const validityValues = validity_mod.validityValues;

pub const DrawdownMetrics = risk_metrics_mod.DrawdownMetrics;
pub const RollingDrawdownMetrics = risk_metrics_mod.RollingDrawdownMetrics;
pub const ExtremaMetrics = risk_metrics_mod.ExtremaMetrics;
pub const DrawdownProfileColumnCount = risk_metrics_mod.DrawdownProfileColumnCount;
pub const RollingDrawdownProfileColumnCount = risk_metrics_mod.RollingDrawdownProfileColumnCount;
pub const ExtremaProfileColumnCount = risk_metrics_mod.ExtremaProfileColumnCount;
pub const drawdownProfileOutputNames = risk_metrics_mod.drawdownProfileOutputNames;
pub const rollingDrawdownProfileOutputNames = risk_metrics_mod.rollingDrawdownProfileOutputNames;
pub const extremaProfileOutputNames = risk_metrics_mod.extremaProfileOutputNames;
pub const drawdownProfile = risk_metrics_mod.drawdownProfile;
pub const rollingDrawdownProfile = risk_metrics_mod.rollingDrawdownProfile;
pub const extremaProfile = risk_metrics_mod.extremaProfile;

pub fn rollingDrawdownProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![RollingDrawdownProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| rollingDrawdownProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| rollingDrawdownProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| rollingDrawdownProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| rollingDrawdownProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| rollingDrawdownProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| rollingDrawdownProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| rollingDrawdownProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| rollingDrawdownProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| rollingDrawdownProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| rollingDrawdownProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| rollingDrawdownProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| rollingDrawdownProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| rollingDrawdownProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn rollingDrawdownProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![RollingDrawdownProfileColumnCount]DeviceColumn {
    const min_periods = options_value.min_periods orelse options_value.window;
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);

    var metrics = try rollingDrawdownProfile(allocator, values, maybe_validity, options_value.window, min_periods);
    defer metrics.deinit();

    var columns: [RollingDrawdownProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.peaks, metrics.validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.drawdowns, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.drawdown_pcts, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(i64, allocator, metrics.peak_ages, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn drawdownProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceDrawdownOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![DrawdownProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| drawdownProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| drawdownProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| drawdownProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| drawdownProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| drawdownProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| drawdownProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| drawdownProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| drawdownProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| drawdownProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| drawdownProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| drawdownProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| drawdownProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| drawdownProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn drawdownProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceDrawdownOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![DrawdownProfileColumnCount]DeviceColumn {
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);

    var metrics = try drawdownProfile(allocator, values, maybe_validity, options_value.min_periods);
    defer metrics.deinit();

    var columns: [DrawdownProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.running_peak, metrics.validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.drawdown, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.drawdown_pct, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn extremaProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceExtremaOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![ExtremaProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| extremaProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| extremaProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| extremaProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| extremaProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| extremaProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| extremaProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| extremaProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| extremaProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| extremaProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| extremaProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| extremaProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| extremaProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| extremaProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn extremaProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceExtremaOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![ExtremaProfileColumnCount]DeviceColumn {
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);

    var metrics = try extremaProfile(allocator, values, maybe_validity, options_value.min_periods);
    defer metrics.deinit();

    var columns: [ExtremaProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.running_low, metrics.validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.running_high, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(bool, allocator, metrics.new_low, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(bool, allocator, metrics.new_high, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
