//! Trend metric kernels and output-name helpers.

const std = @import("std");
const summary_metrics_mod = @import("dataframe_trend_summary_metrics.zig");

pub const TrendMetrics = struct {
    allocator: std.mem.Allocator,
    trends: []i64,
    up_streak: []i64,
    down_streak: []i64,
    flat_streak: []i64,
    reversal: []bool,
    validity: []bool,

    pub fn deinit(self: *TrendMetrics) void {
        self.allocator.free(self.trends);
        self.allocator.free(self.up_streak);
        self.allocator.free(self.down_streak);
        self.allocator.free(self.flat_streak);
        self.allocator.free(self.reversal);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub const TrendProfileColumnCount = 5;

pub fn trendProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![TrendProfileColumnCount][]const u8 {
    var names: [TrendProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "trend", "up_streak", "down_streak", "flat_streak", "reversal" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const TrendSummaryMetrics = summary_metrics_mod.TrendSummaryMetrics;
pub const RollingTrendProfileColumnCount = summary_metrics_mod.RollingTrendProfileColumnCount;
pub const rollingTrendProfileOutputNames = summary_metrics_mod.rollingTrendProfileOutputNames;
pub const ExpandingTrendProfileColumnCount = summary_metrics_mod.ExpandingTrendProfileColumnCount;
pub const expandingTrendProfileOutputNames = summary_metrics_mod.expandingTrendProfileOutputNames;
pub fn validate(values: []const f64, maybe_validity: ?[]const bool, periods: usize) error{ InvalidShape, LengthMismatch }!void {
    if (periods == 0) return error.InvalidShape;
    if (maybe_validity) |validity| {
        if (validity.len != values.len) return error.LengthMismatch;
    }
}

pub fn rowValid(maybe_validity: ?[]const bool, row: usize) bool {
    return if (maybe_validity) |mask| mask[row] else true;
}

pub fn trendValue(current: f64, previous: f64) i64 {
    return if (current > previous) 1 else if (current < previous) -1 else 0;
}

pub fn computeTrendEvents(allocator: std.mem.Allocator, values: []const f64, maybe_validity: ?[]const bool, periods: usize) !struct { trends: []i64, reversals: []bool, validity: []bool } {
    const trends = try allocator.alloc(i64, values.len);
    errdefer allocator.free(trends);
    const reversals = try allocator.alloc(bool, values.len);
    errdefer allocator.free(reversals);
    const validity = try allocator.alloc(bool, values.len);
    errdefer allocator.free(validity);

    var previous_nonzero_trend: i64 = 0;
    for (values, 0..) |value, row| {
        if (row < periods) {
            trends[row] = 0;
            reversals[row] = false;
            validity[row] = false;
            previous_nonzero_trend = 0;
            continue;
        }

        const previous_row = row - periods;
        const valid = rowValid(maybe_validity, row) and rowValid(maybe_validity, previous_row);
        validity[row] = valid;
        if (!valid) {
            trends[row] = 0;
            reversals[row] = false;
            previous_nonzero_trend = 0;
            continue;
        }

        const trend = trendValue(value, values[previous_row]);
        trends[row] = trend;
        reversals[row] = trend != 0 and previous_nonzero_trend != 0 and trend != previous_nonzero_trend;
        if (trend != 0) previous_nonzero_trend = trend;
    }

    return .{ .trends = trends, .reversals = reversals, .validity = validity };
}

pub fn trendProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!TrendMetrics {
    try validate(values, maybe_validity, periods);

    const events = try computeTrendEvents(allocator, values, maybe_validity, periods);
    errdefer allocator.free(events.trends);
    errdefer allocator.free(events.reversals);
    errdefer allocator.free(events.validity);

    const up_streak = try allocator.alloc(i64, values.len);
    errdefer allocator.free(up_streak);
    const down_streak = try allocator.alloc(i64, values.len);
    errdefer allocator.free(down_streak);
    const flat_streak = try allocator.alloc(i64, values.len);
    errdefer allocator.free(flat_streak);

    var current_up: i64 = 0;
    var current_down: i64 = 0;
    var current_flat: i64 = 0;
    for (0..values.len) |row| {
        if (!events.validity[row]) {
            up_streak[row] = 0;
            down_streak[row] = 0;
            flat_streak[row] = 0;
            current_up = 0;
            current_down = 0;
            current_flat = 0;
            continue;
        }

        switch (events.trends[row]) {
            1 => {
                current_up += 1;
                current_down = 0;
                current_flat = 0;
            },
            -1 => {
                current_down += 1;
                current_up = 0;
                current_flat = 0;
            },
            else => {
                current_flat += 1;
                current_up = 0;
                current_down = 0;
            },
        }
        up_streak[row] = current_up;
        down_streak[row] = current_down;
        flat_streak[row] = current_flat;
    }

    return .{
        .allocator = allocator,
        .trends = events.trends,
        .up_streak = up_streak,
        .down_streak = down_streak,
        .flat_streak = flat_streak,
        .reversal = events.reversals,
        .validity = events.validity,
    };
}

pub const rollingTrendProfile = summary_metrics_mod.rollingTrendProfile;
pub const expandingTrendProfile = summary_metrics_mod.expandingTrendProfile;
