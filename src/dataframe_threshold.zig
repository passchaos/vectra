const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_device_column_mod = @import("dataframe_device_column.zig");
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

pub const ThresholdMetrics = struct {
    allocator: std.mem.Allocator,
    distances: []f64,
    abs_distances: []f64,
    above: []bool,
    below: []bool,
    at: []bool,
    validity: []bool,

    pub fn deinit(self: *ThresholdMetrics) void {
        self.allocator.free(self.distances);
        self.allocator.free(self.abs_distances);
        self.allocator.free(self.above);
        self.allocator.free(self.below);
        self.allocator.free(self.at);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub const ThresholdSummaryMetrics = struct {
    allocator: std.mem.Allocator,
    counts: []i64,
    mean_distances: []f64,
    mean_abs_distances: []f64,
    above_rates: []f64,
    below_rates: []f64,
    at_rates: []f64,
    validity: []bool,

    pub fn deinit(self: *ThresholdSummaryMetrics) void {
        self.allocator.free(self.counts);
        self.allocator.free(self.mean_distances);
        self.allocator.free(self.mean_abs_distances);
        self.allocator.free(self.above_rates);
        self.allocator.free(self.below_rates);
        self.allocator.free(self.at_rates);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub const ThresholdProfileColumnCount = 5;

pub fn thresholdProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![ThresholdProfileColumnCount][]const u8 {
    var names: [ThresholdProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "distance", "abs_distance", "above", "below", "at" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const RollingThresholdProfileColumnCount = 6;

pub fn rollingThresholdProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![RollingThresholdProfileColumnCount][]const u8 {
    var names: [RollingThresholdProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{
        "rolling_threshold_count",
        "rolling_mean_distance",
        "rolling_mean_abs_distance",
        "rolling_above_rate",
        "rolling_below_rate",
        "rolling_at_rate",
    };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const ExpandingThresholdProfileColumnCount = 6;

pub fn expandingThresholdProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![ExpandingThresholdProfileColumnCount][]const u8 {
    var names: [ExpandingThresholdProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{
        "expanding_threshold_count",
        "expanding_mean_distance",
        "expanding_mean_abs_distance",
        "expanding_above_rate",
        "expanding_below_rate",
        "expanding_at_rate",
    };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn validateLengths(values: []const f64, maybe_validity: ?[]const bool) error{LengthMismatch}!void {
    if (maybe_validity) |validity| {
        if (validity.len != values.len) return error.LengthMismatch;
    }
}

fn rowValid(maybe_validity: ?[]const bool, row: usize) bool {
    return if (maybe_validity) |mask| mask[row] else true;
}

fn allocSummary(allocator: std.mem.Allocator, rows: usize) std.mem.Allocator.Error!ThresholdSummaryMetrics {
    const counts = try allocator.alloc(i64, rows);
    errdefer allocator.free(counts);
    const mean_distances = try allocator.alloc(f64, rows);
    errdefer allocator.free(mean_distances);
    const mean_abs_distances = try allocator.alloc(f64, rows);
    errdefer allocator.free(mean_abs_distances);
    const above_rates = try allocator.alloc(f64, rows);
    errdefer allocator.free(above_rates);
    const below_rates = try allocator.alloc(f64, rows);
    errdefer allocator.free(below_rates);
    const at_rates = try allocator.alloc(f64, rows);
    errdefer allocator.free(at_rates);
    const validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(validity);
    return .{
        .allocator = allocator,
        .counts = counts,
        .mean_distances = mean_distances,
        .mean_abs_distances = mean_abs_distances,
        .above_rates = above_rates,
        .below_rates = below_rates,
        .at_rates = at_rates,
        .validity = validity,
    };
}

fn addDistance(distance: f64, count: *usize, distance_sum: *f64, abs_distance_sum: *f64, above_count: *usize, below_count: *usize, at_count: *usize) void {
    distance_sum.* += distance;
    abs_distance_sum.* += @abs(distance);
    if (distance > 0) {
        above_count.* += 1;
    } else if (distance < 0) {
        below_count.* += 1;
    } else {
        at_count.* += 1;
    }
    count.* += 1;
}

fn removeDistance(distance: f64, count: *usize, distance_sum: *f64, abs_distance_sum: *f64, above_count: *usize, below_count: *usize, at_count: *usize) void {
    distance_sum.* -= distance;
    abs_distance_sum.* -= @abs(distance);
    if (distance > 0) {
        above_count.* -= 1;
    } else if (distance < 0) {
        below_count.* -= 1;
    } else {
        at_count.* -= 1;
    }
    count.* -= 1;
}

fn writeSummary(row: usize, min_periods: usize, count: usize, distance_sum: f64, abs_distance_sum: f64, above_count: usize, below_count: usize, at_count: usize, out: ThresholdSummaryMetrics) void {
    out.counts[row] = @intCast(count);
    const has_enough = count >= min_periods;
    out.validity[row] = has_enough;
    if (has_enough) {
        const n: f64 = @floatFromInt(count);
        out.mean_distances[row] = distance_sum / n;
        out.mean_abs_distances[row] = abs_distance_sum / n;
        out.above_rates[row] = @as(f64, @floatFromInt(above_count)) / n;
        out.below_rates[row] = @as(f64, @floatFromInt(below_count)) / n;
        out.at_rates[row] = @as(f64, @floatFromInt(at_count)) / n;
    } else {
        out.mean_distances[row] = 0;
        out.mean_abs_distances[row] = 0;
        out.above_rates[row] = 0;
        out.below_rates[row] = 0;
        out.at_rates[row] = 0;
    }
}

pub fn thresholdProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    threshold: f64,
) (std.mem.Allocator.Error || error{LengthMismatch})!ThresholdMetrics {
    try validateLengths(values, maybe_validity);

    const rows = values.len;
    const distances = try allocator.alloc(f64, rows);
    errdefer allocator.free(distances);
    const abs_distances = try allocator.alloc(f64, rows);
    errdefer allocator.free(abs_distances);
    const above = try allocator.alloc(bool, rows);
    errdefer allocator.free(above);
    const below = try allocator.alloc(bool, rows);
    errdefer allocator.free(below);
    const at = try allocator.alloc(bool, rows);
    errdefer allocator.free(at);
    const validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(validity);

    for (values, 0..) |value, row| {
        const valid = rowValid(maybe_validity, row);
        validity[row] = valid;
        if (valid) {
            const distance = value - threshold;
            distances[row] = distance;
            abs_distances[row] = @abs(distance);
            above[row] = distance > 0;
            below[row] = distance < 0;
            at[row] = distance == 0;
        } else {
            distances[row] = 0;
            abs_distances[row] = 0;
            above[row] = false;
            below[row] = false;
            at[row] = false;
        }
    }

    return .{
        .allocator = allocator,
        .distances = distances,
        .abs_distances = abs_distances,
        .above = above,
        .below = below,
        .at = at,
        .validity = validity,
    };
}

pub fn rollingThresholdProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    threshold: f64,
    window: usize,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!ThresholdSummaryMetrics {
    if (window == 0) return error.InvalidShape;
    if (min_periods == 0 or min_periods > window) return error.InvalidShape;
    try validateLengths(values, maybe_validity);

    var out = try allocSummary(allocator, values.len);
    errdefer out.deinit();

    var count: usize = 0;
    var distance_sum: f64 = 0;
    var abs_distance_sum: f64 = 0;
    var above_count: usize = 0;
    var below_count: usize = 0;
    var at_count: usize = 0;

    for (values, 0..) |value, row| {
        if (rowValid(maybe_validity, row)) {
            addDistance(value - threshold, &count, &distance_sum, &abs_distance_sum, &above_count, &below_count, &at_count);
        }

        if (row >= window) {
            const evict_row = row - window;
            if (rowValid(maybe_validity, evict_row)) {
                removeDistance(values[evict_row] - threshold, &count, &distance_sum, &abs_distance_sum, &above_count, &below_count, &at_count);
            }
        }

        writeSummary(row, min_periods, count, distance_sum, abs_distance_sum, above_count, below_count, at_count, out);
    }

    return out;
}

pub fn expandingThresholdProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    threshold: f64,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!ThresholdSummaryMetrics {
    if (min_periods == 0) return error.InvalidShape;
    try validateLengths(values, maybe_validity);

    var out = try allocSummary(allocator, values.len);
    errdefer out.deinit();

    var count: usize = 0;
    var distance_sum: f64 = 0;
    var abs_distance_sum: f64 = 0;
    var above_count: usize = 0;
    var below_count: usize = 0;
    var at_count: usize = 0;

    for (values, 0..) |value, row| {
        if (rowValid(maybe_validity, row)) {
            addDistance(value - threshold, &count, &distance_sum, &abs_distance_sum, &above_count, &below_count, &at_count);
        }

        writeSummary(row, min_periods, count, distance_sum, abs_distance_sum, above_count, below_count, at_count, out);
    }

    return out;
}

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
