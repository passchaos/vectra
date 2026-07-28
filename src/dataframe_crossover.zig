const std = @import("std");

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
