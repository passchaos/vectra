//! Rolling and expanding crossover summary metric kernels.

const std = @import("std");
const base_mod = @import("metrics.zig");

const validatePairLengths = base_mod.validatePairLengths;
const fillSpreads = base_mod.fillSpreads;
const isCrossAbove = base_mod.isCrossAbove;
const isCrossBelow = base_mod.isCrossBelow;

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
