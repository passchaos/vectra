const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const names_mod = @import("dataframe_names.zig");
const dataframe_device_column_mod = @import("dataframe_device_column.zig");
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

pub const CrossoverMetrics = struct {
    allocator: std.mem.Allocator,
    spreads: []f64,
    ratios: []f64,
    cross_above: []bool,
    cross_below: []bool,
    spread_validity: []bool,
    cross_validity: []bool,

    pub fn deinit(self: *CrossoverMetrics) void {
        self.allocator.free(self.spreads);
        self.allocator.free(self.ratios);
        self.allocator.free(self.cross_above);
        self.allocator.free(self.cross_below);
        self.allocator.free(self.spread_validity);
        self.allocator.free(self.cross_validity);
        self.* = undefined;
    }
};

pub const CrossoverSummaryMetrics = struct {
    allocator: std.mem.Allocator,
    counts: []i64,
    cross_above_counts: []i64,
    cross_below_counts: []i64,
    cross_above_rates: []f64,
    cross_below_rates: []f64,
    mean_abs_spreads: []f64,
    validity: []bool,

    pub fn deinit(self: *CrossoverSummaryMetrics) void {
        self.allocator.free(self.counts);
        self.allocator.free(self.cross_above_counts);
        self.allocator.free(self.cross_below_counts);
        self.allocator.free(self.cross_above_rates);
        self.allocator.free(self.cross_below_rates);
        self.allocator.free(self.mean_abs_spreads);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub const CrossoverProfileColumnCount = 4;

pub fn crossoverProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![CrossoverProfileColumnCount][]const u8 {
    var names: [CrossoverProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "spread", "ratio", "cross_above", "cross_below" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const RollingCrossoverProfileColumnCount = 6;

pub fn rollingCrossoverProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![RollingCrossoverProfileColumnCount][]const u8 {
    var names: [RollingCrossoverProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "rolling_cross_count", "rolling_cross_above_count", "rolling_cross_below_count", "rolling_cross_above_rate", "rolling_cross_below_rate", "rolling_mean_abs_spread" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const ExpandingCrossoverProfileColumnCount = 6;

pub fn expandingCrossoverProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![ExpandingCrossoverProfileColumnCount][]const u8 {
    var names: [ExpandingCrossoverProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "expanding_cross_count", "expanding_cross_above_count", "expanding_cross_below_count", "expanding_cross_above_rate", "expanding_cross_below_rate", "expanding_mean_abs_spread" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn validatePairLengths(lhs: []const f64, rhs: []const f64, maybe_lhs_validity: ?[]const bool, maybe_rhs_validity: ?[]const bool) error{LengthMismatch}!void {
    if (lhs.len != rhs.len) return error.LengthMismatch;
    if (maybe_lhs_validity) |validity| {
        if (validity.len != lhs.len) return error.LengthMismatch;
    }
    if (maybe_rhs_validity) |validity| {
        if (validity.len != rhs.len) return error.LengthMismatch;
    }
}

fn rowValid(maybe_lhs_validity: ?[]const bool, maybe_rhs_validity: ?[]const bool, row: usize) bool {
    return (if (maybe_lhs_validity) |mask| mask[row] else true) and (if (maybe_rhs_validity) |mask| mask[row] else true);
}

fn allocSummary(allocator: std.mem.Allocator, rows: usize) std.mem.Allocator.Error!CrossoverSummaryMetrics {
    const counts = try allocator.alloc(i64, rows);
    errdefer allocator.free(counts);
    const cross_above_counts = try allocator.alloc(i64, rows);
    errdefer allocator.free(cross_above_counts);
    const cross_below_counts = try allocator.alloc(i64, rows);
    errdefer allocator.free(cross_below_counts);
    const cross_above_rates = try allocator.alloc(f64, rows);
    errdefer allocator.free(cross_above_rates);
    const cross_below_rates = try allocator.alloc(f64, rows);
    errdefer allocator.free(cross_below_rates);
    const mean_abs_spreads = try allocator.alloc(f64, rows);
    errdefer allocator.free(mean_abs_spreads);
    const validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(validity);
    return .{
        .allocator = allocator,
        .counts = counts,
        .cross_above_counts = cross_above_counts,
        .cross_below_counts = cross_below_counts,
        .cross_above_rates = cross_above_rates,
        .cross_below_rates = cross_below_rates,
        .mean_abs_spreads = mean_abs_spreads,
        .validity = validity,
    };
}

fn fillSpreads(lhs: []const f64, rhs: []const f64, maybe_lhs_validity: ?[]const bool, maybe_rhs_validity: ?[]const bool, spreads: []f64, spread_validity: []bool) void {
    for (lhs, rhs, 0..) |left, right, row| {
        const valid = rowValid(maybe_lhs_validity, maybe_rhs_validity, row);
        spread_validity[row] = valid;
        spreads[row] = if (valid) left - right else 0;
    }
}

fn isCrossAbove(previous_spread: f64, current_spread: f64) bool {
    return previous_spread <= 0 and current_spread > 0;
}

fn isCrossBelow(previous_spread: f64, current_spread: f64) bool {
    return previous_spread >= 0 and current_spread < 0;
}

fn writeSummary(row: usize, count: usize, above_count: usize, below_count: usize, sum_abs_spread: f64, min_periods: usize, out: CrossoverSummaryMetrics) void {
    out.counts[row] = @intCast(count);
    out.cross_above_counts[row] = @intCast(above_count);
    out.cross_below_counts[row] = @intCast(below_count);
    const has_enough = count >= min_periods;
    out.validity[row] = has_enough;
    if (has_enough) {
        // Rates use the same valid-spread denominator as the mean spread metric
        // so nullable gaps affect all summary fields consistently.
        const n: f64 = @floatFromInt(count);
        out.cross_above_rates[row] = @as(f64, @floatFromInt(above_count)) / n;
        out.cross_below_rates[row] = @as(f64, @floatFromInt(below_count)) / n;
        out.mean_abs_spreads[row] = sum_abs_spread / n;
    } else {
        out.cross_above_rates[row] = 0;
        out.cross_below_rates[row] = 0;
        out.mean_abs_spreads[row] = 0;
    }
}

pub fn crossoverProfile(
    allocator: std.mem.Allocator,
    lhs: []const f64,
    rhs: []const f64,
    maybe_lhs_validity: ?[]const bool,
    maybe_rhs_validity: ?[]const bool,
    periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!CrossoverMetrics {
    if (periods == 0) return error.InvalidShape;
    try validatePairLengths(lhs, rhs, maybe_lhs_validity, maybe_rhs_validity);

    const rows = lhs.len;
    const spreads = try allocator.alloc(f64, rows);
    errdefer allocator.free(spreads);
    const ratios = try allocator.alloc(f64, rows);
    errdefer allocator.free(ratios);
    const cross_above = try allocator.alloc(bool, rows);
    errdefer allocator.free(cross_above);
    const cross_below = try allocator.alloc(bool, rows);
    errdefer allocator.free(cross_below);
    const spread_validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(spread_validity);
    const cross_validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(cross_validity);

    fillSpreads(lhs, rhs, maybe_lhs_validity, maybe_rhs_validity, spreads, spread_validity);
    for (lhs, rhs, 0..) |left, right, row| {
        const current_valid = spread_validity[row];
        ratios[row] = if (current_valid) if (right == 0) std.math.nan(f64) else left / right else 0;
        cross_above[row] = false;
        cross_below[row] = false;
        if (row < periods) {
            cross_validity[row] = false;
            continue;
        }

        const previous_row = row - periods;
        const event_valid = current_valid and spread_validity[previous_row];
        cross_validity[row] = event_valid;
        if (event_valid) {
            cross_above[row] = isCrossAbove(spreads[previous_row], spreads[row]);
            cross_below[row] = isCrossBelow(spreads[previous_row], spreads[row]);
        }
    }

    return .{
        .allocator = allocator,
        .spreads = spreads,
        .ratios = ratios,
        .cross_above = cross_above,
        .cross_below = cross_below,
        .spread_validity = spread_validity,
        .cross_validity = cross_validity,
    };
}

pub fn rollingCrossoverProfile(
    allocator: std.mem.Allocator,
    lhs: []const f64,
    rhs: []const f64,
    maybe_lhs_validity: ?[]const bool,
    maybe_rhs_validity: ?[]const bool,
    periods: usize,
    window: usize,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!CrossoverSummaryMetrics {
    if (periods == 0 or window == 0) return error.InvalidShape;
    if (min_periods == 0 or min_periods > window) return error.InvalidShape;
    try validatePairLengths(lhs, rhs, maybe_lhs_validity, maybe_rhs_validity);

    const rows = lhs.len;
    const spreads = try allocator.alloc(f64, rows);
    defer allocator.free(spreads);
    const spread_validity = try allocator.alloc(bool, rows);
    defer allocator.free(spread_validity);
    fillSpreads(lhs, rhs, maybe_lhs_validity, maybe_rhs_validity, spreads, spread_validity);

    const cross_above = try allocator.alloc(bool, rows);
    defer allocator.free(cross_above);
    const cross_below = try allocator.alloc(bool, rows);
    defer allocator.free(cross_below);
    const cross_validity = try allocator.alloc(bool, rows);
    defer allocator.free(cross_validity);
    for (0..rows) |row| {
        cross_above[row] = false;
        cross_below[row] = false;
        if (row < periods) {
            cross_validity[row] = false;
            continue;
        }
        const previous_row = row - periods;
        const event_valid = spread_validity[row] and spread_validity[previous_row];
        cross_validity[row] = event_valid;
        if (event_valid) {
            cross_above[row] = isCrossAbove(spreads[previous_row], spreads[row]);
            cross_below[row] = isCrossBelow(spreads[previous_row], spreads[row]);
        }
    }

    var out = try allocSummary(allocator, rows);
    errdefer out.deinit();
    for (0..rows) |row| {
        const start = if (row + 1 > window) row + 1 - window else 0;
        var count: usize = 0;
        var above_count: usize = 0;
        var below_count: usize = 0;
        var sum_abs_spread: f64 = 0;
        for (start..row + 1) |window_row| {
            if (!spread_validity[window_row]) continue;
            count += 1;
            sum_abs_spread += @abs(spreads[window_row]);
            if (cross_validity[window_row] and cross_above[window_row]) above_count += 1;
            if (cross_validity[window_row] and cross_below[window_row]) below_count += 1;
        }
        writeSummary(row, count, above_count, below_count, sum_abs_spread, min_periods, out);
    }

    return out;
}

pub fn expandingCrossoverProfile(
    allocator: std.mem.Allocator,
    lhs: []const f64,
    rhs: []const f64,
    maybe_lhs_validity: ?[]const bool,
    maybe_rhs_validity: ?[]const bool,
    periods: usize,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!CrossoverSummaryMetrics {
    if (periods == 0 or min_periods == 0) return error.InvalidShape;
    try validatePairLengths(lhs, rhs, maybe_lhs_validity, maybe_rhs_validity);

    const rows = lhs.len;
    const spreads = try allocator.alloc(f64, rows);
    defer allocator.free(spreads);
    const spread_validity = try allocator.alloc(bool, rows);
    defer allocator.free(spread_validity);
    fillSpreads(lhs, rhs, maybe_lhs_validity, maybe_rhs_validity, spreads, spread_validity);

    var out = try allocSummary(allocator, rows);
    errdefer out.deinit();

    var count: usize = 0;
    var above_count: usize = 0;
    var below_count: usize = 0;
    var sum_abs_spread: f64 = 0;
    for (0..rows) |row| {
        if (spread_validity[row]) {
            count += 1;
            sum_abs_spread += @abs(spreads[row]);
        }

        if (spread_validity[row] and row >= periods) {
            const previous_row = row - periods;
            if (spread_validity[previous_row]) {
                if (isCrossAbove(spreads[previous_row], spreads[row])) above_count += 1;
                if (isCrossBelow(spreads[previous_row], spreads[row])) below_count += 1;
            }
        }

        writeSummary(row, count, above_count, below_count, sum_abs_spread, min_periods, out);
    }

    return out;
}

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
