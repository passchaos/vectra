//! Rolling and expanding trend summary metric kernels.

const std = @import("std");
const base_mod = @import("dataframe_trend_metrics.zig");

const validate = base_mod.validate;
const computeTrendEvents = base_mod.computeTrendEvents;

pub const TrendSummaryMetrics = struct {
    allocator: std.mem.Allocator,
    counts: []i64,
    up_rates: []f64,
    down_rates: []f64,
    flat_rates: []f64,
    reversal_rates: []f64,
    validity: []bool,

    pub fn deinit(self: *TrendSummaryMetrics) void {
        self.allocator.free(self.counts);
        self.allocator.free(self.up_rates);
        self.allocator.free(self.down_rates);
        self.allocator.free(self.flat_rates);
        self.allocator.free(self.reversal_rates);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub const RollingTrendProfileColumnCount = 5;

pub fn rollingTrendProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![RollingTrendProfileColumnCount][]const u8 {
    var names: [RollingTrendProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "rolling_trend_count", "rolling_up_rate", "rolling_down_rate", "rolling_flat_rate", "rolling_reversal_rate" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const ExpandingTrendProfileColumnCount = 5;

pub fn expandingTrendProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![ExpandingTrendProfileColumnCount][]const u8 {
    var names: [ExpandingTrendProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "expanding_trend_count", "expanding_up_rate", "expanding_down_rate", "expanding_flat_rate", "expanding_reversal_rate" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn allocSummary(allocator: std.mem.Allocator, rows: usize) std.mem.Allocator.Error!TrendSummaryMetrics {
    const counts = try allocator.alloc(i64, rows);
    errdefer allocator.free(counts);
    const up_rates = try allocator.alloc(f64, rows);
    errdefer allocator.free(up_rates);
    const down_rates = try allocator.alloc(f64, rows);
    errdefer allocator.free(down_rates);
    const flat_rates = try allocator.alloc(f64, rows);
    errdefer allocator.free(flat_rates);
    const reversal_rates = try allocator.alloc(f64, rows);
    errdefer allocator.free(reversal_rates);
    const validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(validity);
    return .{
        .allocator = allocator,
        .counts = counts,
        .up_rates = up_rates,
        .down_rates = down_rates,
        .flat_rates = flat_rates,
        .reversal_rates = reversal_rates,
        .validity = validity,
    };
}

fn writeSummary(row: usize, count: usize, up_count: usize, down_count: usize, flat_count: usize, reversal_count: usize, min_periods: usize, out: TrendSummaryMetrics) void {
    out.counts[row] = @intCast(count);
    const has_enough = count >= min_periods;
    out.validity[row] = has_enough;
    if (has_enough) {
        const n: f64 = @floatFromInt(count);
        out.up_rates[row] = @as(f64, @floatFromInt(up_count)) / n;
        out.down_rates[row] = @as(f64, @floatFromInt(down_count)) / n;
        out.flat_rates[row] = @as(f64, @floatFromInt(flat_count)) / n;
        out.reversal_rates[row] = @as(f64, @floatFromInt(reversal_count)) / n;
    } else {
        out.up_rates[row] = 0;
        out.down_rates[row] = 0;
        out.flat_rates[row] = 0;
        out.reversal_rates[row] = 0;
    }
}

pub fn rollingTrendProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    periods: usize,
    window: usize,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!TrendSummaryMetrics {
    if (window == 0) return error.InvalidShape;
    if (min_periods == 0 or min_periods > window) return error.InvalidShape;
    try validate(values, maybe_validity, periods);

    const events = try computeTrendEvents(allocator, values, maybe_validity, periods);
    defer allocator.free(events.trends);
    defer allocator.free(events.reversals);
    defer allocator.free(events.validity);

    var out = try allocSummary(allocator, values.len);
    errdefer out.deinit();
    for (0..values.len) |row| {
        const start = if (row + 1 > window) row + 1 - window else 0;
        var count: usize = 0;
        var up_count: usize = 0;
        var down_count: usize = 0;
        var flat_count: usize = 0;
        var reversal_count: usize = 0;
        for (start..row + 1) |window_row| {
            if (!events.validity[window_row]) continue;
            switch (events.trends[window_row]) {
                1 => up_count += 1,
                -1 => down_count += 1,
                else => flat_count += 1,
            }
            if (events.reversals[window_row]) reversal_count += 1;
            count += 1;
        }
        writeSummary(row, count, up_count, down_count, flat_count, reversal_count, min_periods, out);
    }
    return out;
}

pub fn expandingTrendProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    periods: usize,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!TrendSummaryMetrics {
    if (min_periods == 0) return error.InvalidShape;
    try validate(values, maybe_validity, periods);

    const events = try computeTrendEvents(allocator, values, maybe_validity, periods);
    defer allocator.free(events.trends);
    defer allocator.free(events.reversals);
    defer allocator.free(events.validity);

    var out = try allocSummary(allocator, values.len);
    errdefer out.deinit();
    var count: usize = 0;
    var up_count: usize = 0;
    var down_count: usize = 0;
    var flat_count: usize = 0;
    var reversal_count: usize = 0;
    for (0..values.len) |row| {
        if (events.validity[row]) {
            switch (events.trends[row]) {
                1 => up_count += 1,
                -1 => down_count += 1,
                else => flat_count += 1,
            }
            if (events.reversals[row]) reversal_count += 1;
            count += 1;
        }
        writeSummary(row, count, up_count, down_count, flat_count, reversal_count, min_periods, out);
    }
    return out;
}
