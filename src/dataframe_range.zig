const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_device_column_mod = @import("dataframe_device_column.zig");
const numeric_mod = @import("dataframe_numeric.zig");
const options_mod = @import("dataframe_options.zig");
const validity_mod = @import("dataframe_validity.zig");

const DeviceColumn = dataframe_device_column_mod.DeviceColumn;
const DeviceTypedColumn = dataframe_device_column_mod.DeviceTypedColumn;
const DeviceRollingOptions = options_mod.DeviceRollingOptions;
const castToF64 = numeric_mod.castToF64;
const validityValues = validity_mod.validityValues;

pub const RangeMetrics = struct {
    allocator: std.mem.Allocator,
    lows: []f64,
    highs: []f64,
    ranges: []f64,
    positions: []f64,
    validity: []bool,

    pub fn deinit(self: *RangeMetrics) void {
        self.allocator.free(self.lows);
        self.allocator.free(self.highs);
        self.allocator.free(self.ranges);
        self.allocator.free(self.positions);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub const RollingRangeProfileColumnCount = 4;

pub fn rollingRangeProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![RollingRangeProfileColumnCount][]const u8 {
    var names: [RollingRangeProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "rolling_low", "rolling_high", "rolling_range", "rolling_position" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn validate(values: []const f64, maybe_validity: ?[]const bool) error{LengthMismatch}!void {
    if (maybe_validity) |validity| {
        if (validity.len != values.len) return error.LengthMismatch;
    }
}

fn rowValid(maybe_validity: ?[]const bool, row: usize) bool {
    return if (maybe_validity) |mask| mask[row] else true;
}

pub fn rollingRangeProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    window: usize,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!RangeMetrics {
    if (window == 0) return error.InvalidShape;
    if (min_periods == 0 or min_periods > window) return error.InvalidShape;
    try validate(values, maybe_validity);

    const lows = try allocator.alloc(f64, values.len);
    errdefer allocator.free(lows);
    const highs = try allocator.alloc(f64, values.len);
    errdefer allocator.free(highs);
    const ranges = try allocator.alloc(f64, values.len);
    errdefer allocator.free(ranges);
    const positions = try allocator.alloc(f64, values.len);
    errdefer allocator.free(positions);
    const validity = try allocator.alloc(bool, values.len);
    errdefer allocator.free(validity);

    // Recompute each trailing window in host memory, preserving a single future
    // lowering seam for device rolling min/max kernels.
    for (values, 0..) |value, row| {
        const start = if (row + 1 > window) row + 1 - window else 0;
        var count: usize = 0;
        var low: f64 = 0;
        var high: f64 = 0;
        for (start..row + 1) |window_row| {
            if (!rowValid(maybe_validity, window_row)) continue;
            const x = values[window_row];
            if (count == 0) {
                low = x;
                high = x;
            } else {
                if (x < low) low = x;
                if (x > high) high = x;
            }
            count += 1;
        }

        const has_enough = rowValid(maybe_validity, row) and count >= min_periods;
        validity[row] = has_enough;
        if (has_enough) {
            const range = high - low;
            lows[row] = low;
            highs[row] = high;
            ranges[row] = range;
            positions[row] = if (range == 0) std.math.nan(f64) else (value - low) / range;
        } else {
            lows[row] = 0;
            highs[row] = 0;
            ranges[row] = 0;
            positions[row] = 0;
        }
    }

    return .{ .allocator = allocator, .lows = lows, .highs = highs, .ranges = ranges, .positions = positions, .validity = validity };
}

pub fn rollingRangeProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![RollingRangeProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| rollingRangeProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| rollingRangeProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| rollingRangeProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| rollingRangeProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| rollingRangeProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| rollingRangeProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| rollingRangeProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| rollingRangeProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| rollingRangeProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| rollingRangeProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| rollingRangeProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| rollingRangeProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| rollingRangeProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn rollingRangeProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![RollingRangeProfileColumnCount]DeviceColumn {
    const min_periods = options_value.min_periods orelse options_value.window;
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);

    var metrics = try rollingRangeProfile(allocator, values, maybe_validity, options_value.window, min_periods);
    defer metrics.deinit();

    var columns: [RollingRangeProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.lows, metrics.validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.highs, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.ranges, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.positions, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
