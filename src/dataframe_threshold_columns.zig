//! DeviceColumn builders for threshold profile metrics.

const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_device_column_mod = @import("dataframe/device_column.zig");
const threshold_metrics_mod = @import("dataframe_threshold_metrics.zig");
const numeric_mod = @import("dataframe_numeric.zig");
const options_mod = @import("dataframe_options.zig");
const validity_mod = @import("dataframe_validity.zig");

const DeviceColumn = dataframe_device_column_mod.DeviceColumn;
const DeviceTypedColumn = dataframe_device_column_mod.DeviceTypedColumn;
const DeviceThresholdOptions = options_mod.DeviceThresholdOptions;
const DeviceRollingOptions = options_mod.DeviceRollingOptions;
const DeviceExpandingOptions = options_mod.DeviceExpandingOptions;
const castToF64 = numeric_mod.castToF64;
const validityValues = validity_mod.validityValues;

pub const ThresholdMetrics = threshold_metrics_mod.ThresholdMetrics;
pub const ThresholdSummaryMetrics = threshold_metrics_mod.ThresholdSummaryMetrics;
pub const ThresholdProfileColumnCount = threshold_metrics_mod.ThresholdProfileColumnCount;
pub const RollingThresholdProfileColumnCount = threshold_metrics_mod.RollingThresholdProfileColumnCount;
pub const ExpandingThresholdProfileColumnCount = threshold_metrics_mod.ExpandingThresholdProfileColumnCount;
pub const thresholdProfileOutputNames = threshold_metrics_mod.thresholdProfileOutputNames;
pub const rollingThresholdProfileOutputNames = threshold_metrics_mod.rollingThresholdProfileOutputNames;
pub const expandingThresholdProfileOutputNames = threshold_metrics_mod.expandingThresholdProfileOutputNames;
pub const thresholdProfile = threshold_metrics_mod.thresholdProfile;
pub const rollingThresholdProfile = threshold_metrics_mod.rollingThresholdProfile;
pub const expandingThresholdProfile = threshold_metrics_mod.expandingThresholdProfile;

pub fn thresholdProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceThresholdOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![ThresholdProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| thresholdProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| thresholdProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| thresholdProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| thresholdProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| thresholdProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| thresholdProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| thresholdProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| thresholdProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| thresholdProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| thresholdProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| thresholdProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| thresholdProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| thresholdProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn thresholdProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceThresholdOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![ThresholdProfileColumnCount]DeviceColumn {
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);

    var metrics = try thresholdProfile(allocator, values, maybe_validity, options_value.threshold);
    defer metrics.deinit();

    var columns: [ThresholdProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.distances, metrics.validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.abs_distances, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(bool, allocator, metrics.above, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(bool, allocator, metrics.below, metrics.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(bool, allocator, metrics.at, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn rollingThresholdProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    threshold: f64,
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![RollingThresholdProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| rollingThresholdProfileColumnsTyped(i8, allocator, typed, threshold, options_value, device_value),
        .i16 => |typed| rollingThresholdProfileColumnsTyped(i16, allocator, typed, threshold, options_value, device_value),
        .i32 => |typed| rollingThresholdProfileColumnsTyped(i32, allocator, typed, threshold, options_value, device_value),
        .i64 => |typed| rollingThresholdProfileColumnsTyped(i64, allocator, typed, threshold, options_value, device_value),
        .u8 => |typed| rollingThresholdProfileColumnsTyped(u8, allocator, typed, threshold, options_value, device_value),
        .u16 => |typed| rollingThresholdProfileColumnsTyped(u16, allocator, typed, threshold, options_value, device_value),
        .u32 => |typed| rollingThresholdProfileColumnsTyped(u32, allocator, typed, threshold, options_value, device_value),
        .u64 => |typed| rollingThresholdProfileColumnsTyped(u64, allocator, typed, threshold, options_value, device_value),
        .usize => |typed| rollingThresholdProfileColumnsTyped(usize, allocator, typed, threshold, options_value, device_value),
        .isize => |typed| rollingThresholdProfileColumnsTyped(isize, allocator, typed, threshold, options_value, device_value),
        .f16 => |typed| rollingThresholdProfileColumnsTyped(f16, allocator, typed, threshold, options_value, device_value),
        .f32 => |typed| rollingThresholdProfileColumnsTyped(f32, allocator, typed, threshold, options_value, device_value),
        .f64 => |typed| rollingThresholdProfileColumnsTyped(f64, allocator, typed, threshold, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn rollingThresholdProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    threshold: f64,
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![RollingThresholdProfileColumnCount]DeviceColumn {
    const min_periods = options_value.min_periods orelse options_value.window;
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);

    var metrics = try rollingThresholdProfile(allocator, values, maybe_validity, threshold, options_value.window, min_periods);
    defer metrics.deinit();

    var columns: [RollingThresholdProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.mean_distances, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.mean_abs_distances, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.above_rates, metrics.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.below_rates, metrics.validity, device_value);
    initialized += 1;
    columns[5] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.at_rates, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn expandingThresholdProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    threshold: f64,
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingThresholdProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| expandingThresholdProfileColumnsTyped(i8, allocator, typed, threshold, options_value, device_value),
        .i16 => |typed| expandingThresholdProfileColumnsTyped(i16, allocator, typed, threshold, options_value, device_value),
        .i32 => |typed| expandingThresholdProfileColumnsTyped(i32, allocator, typed, threshold, options_value, device_value),
        .i64 => |typed| expandingThresholdProfileColumnsTyped(i64, allocator, typed, threshold, options_value, device_value),
        .u8 => |typed| expandingThresholdProfileColumnsTyped(u8, allocator, typed, threshold, options_value, device_value),
        .u16 => |typed| expandingThresholdProfileColumnsTyped(u16, allocator, typed, threshold, options_value, device_value),
        .u32 => |typed| expandingThresholdProfileColumnsTyped(u32, allocator, typed, threshold, options_value, device_value),
        .u64 => |typed| expandingThresholdProfileColumnsTyped(u64, allocator, typed, threshold, options_value, device_value),
        .usize => |typed| expandingThresholdProfileColumnsTyped(usize, allocator, typed, threshold, options_value, device_value),
        .isize => |typed| expandingThresholdProfileColumnsTyped(isize, allocator, typed, threshold, options_value, device_value),
        .f16 => |typed| expandingThresholdProfileColumnsTyped(f16, allocator, typed, threshold, options_value, device_value),
        .f32 => |typed| expandingThresholdProfileColumnsTyped(f32, allocator, typed, threshold, options_value, device_value),
        .f64 => |typed| expandingThresholdProfileColumnsTyped(f64, allocator, typed, threshold, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn expandingThresholdProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    threshold: f64,
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingThresholdProfileColumnCount]DeviceColumn {
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);

    var metrics = try expandingThresholdProfile(allocator, values, maybe_validity, threshold, options_value.min_periods);
    defer metrics.deinit();

    var columns: [ExpandingThresholdProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.mean_distances, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.mean_abs_distances, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.above_rates, metrics.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.below_rates, metrics.validity, device_value);
    initialized += 1;
    columns[5] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.at_rates, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
