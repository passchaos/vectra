//! Rolling and expanding threshold summary metric kernels.

const std = @import("std");
const base_mod = @import("dataframe_threshold_metrics.zig");

const validateLengths = base_mod.validateLengths;
const rowValid = base_mod.rowValid;

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
