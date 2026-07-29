//! Pairwise crossover metric kernels and output-name helpers.

const std = @import("std");
const summary_metrics_mod = @import("dataframe_crossover_summary_metrics.zig");

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

pub const CrossoverSummaryMetrics = summary_metrics_mod.CrossoverSummaryMetrics;
pub const RollingCrossoverProfileColumnCount = summary_metrics_mod.RollingCrossoverProfileColumnCount;
pub const rollingCrossoverProfileOutputNames = summary_metrics_mod.rollingCrossoverProfileOutputNames;
pub const ExpandingCrossoverProfileColumnCount = summary_metrics_mod.ExpandingCrossoverProfileColumnCount;
pub const expandingCrossoverProfileOutputNames = summary_metrics_mod.expandingCrossoverProfileOutputNames;
pub fn validatePairLengths(lhs: []const f64, rhs: []const f64, maybe_lhs_validity: ?[]const bool, maybe_rhs_validity: ?[]const bool) error{LengthMismatch}!void {
    if (lhs.len != rhs.len) return error.LengthMismatch;
    if (maybe_lhs_validity) |validity| {
        if (validity.len != lhs.len) return error.LengthMismatch;
    }
    if (maybe_rhs_validity) |validity| {
        if (validity.len != rhs.len) return error.LengthMismatch;
    }
}

pub fn rowValid(maybe_lhs_validity: ?[]const bool, maybe_rhs_validity: ?[]const bool, row: usize) bool {
    return (if (maybe_lhs_validity) |mask| mask[row] else true) and (if (maybe_rhs_validity) |mask| mask[row] else true);
}

pub fn fillSpreads(lhs: []const f64, rhs: []const f64, maybe_lhs_validity: ?[]const bool, maybe_rhs_validity: ?[]const bool, spreads: []f64, spread_validity: []bool) void {
    for (lhs, rhs, 0..) |left, right, row| {
        const valid = rowValid(maybe_lhs_validity, maybe_rhs_validity, row);
        spread_validity[row] = valid;
        spreads[row] = if (valid) left - right else 0;
    }
}

pub fn isCrossAbove(previous_spread: f64, current_spread: f64) bool {
    return previous_spread <= 0 and current_spread > 0;
}

pub fn isCrossBelow(previous_spread: f64, current_spread: f64) bool {
    return previous_spread >= 0 and current_spread < 0;
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

pub const rollingCrossoverProfile = summary_metrics_mod.rollingCrossoverProfile;
pub const expandingCrossoverProfile = summary_metrics_mod.expandingCrossoverProfile;
