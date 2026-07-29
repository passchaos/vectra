//! Rolling and expanding change-point summary metric kernels.

const std = @import("std");
const base_mod = @import("dataframe_change_metrics.zig");

const validate = base_mod.validate;
const validPair = base_mod.validPair;
const absDelta = base_mod.absDelta;

pub const ChangeSummaryMetrics = struct {
    allocator: std.mem.Allocator,
    counts: []i64,
    change_counts: []i64,
    change_rates: []f64,
    mean_abs_delta: []f64,
    max_abs_delta: []f64,
    validity: []bool,

    pub fn deinit(self: *ChangeSummaryMetrics) void {
        self.allocator.free(self.counts);
        self.allocator.free(self.change_counts);
        self.allocator.free(self.change_rates);
        self.allocator.free(self.mean_abs_delta);
        self.allocator.free(self.max_abs_delta);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub const RollingChangePointProfileColumnCount = 5;

pub fn rollingChangePointProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![RollingChangePointProfileColumnCount][]const u8 {
    var names: [RollingChangePointProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "rolling_change_count", "rolling_change_point_count", "rolling_change_rate", "rolling_mean_abs_delta", "rolling_max_abs_delta" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const ExpandingChangePointProfileColumnCount = 5;

pub fn expandingChangePointProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![ExpandingChangePointProfileColumnCount][]const u8 {
    var names: [ExpandingChangePointProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "expanding_change_count", "expanding_change_point_count", "expanding_change_rate", "expanding_mean_abs_delta", "expanding_max_abs_delta" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn allocSummary(allocator: std.mem.Allocator, rows: usize) std.mem.Allocator.Error!ChangeSummaryMetrics {
    const counts = try allocator.alloc(i64, rows);
    errdefer allocator.free(counts);
    const change_counts = try allocator.alloc(i64, rows);
    errdefer allocator.free(change_counts);
    const change_rates = try allocator.alloc(f64, rows);
    errdefer allocator.free(change_rates);
    const mean_abs_delta = try allocator.alloc(f64, rows);
    errdefer allocator.free(mean_abs_delta);
    const max_abs_delta = try allocator.alloc(f64, rows);
    errdefer allocator.free(max_abs_delta);
    const validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(validity);
    return .{
        .allocator = allocator,
        .counts = counts,
        .change_counts = change_counts,
        .change_rates = change_rates,
        .mean_abs_delta = mean_abs_delta,
        .max_abs_delta = max_abs_delta,
        .validity = validity,
    };
}

fn writeSummary(row: usize, min_periods: usize, count: usize, change_count: usize, sum_abs: f64, max_abs: f64, out: ChangeSummaryMetrics) void {
    out.counts[row] = @intCast(count);
    out.change_counts[row] = @intCast(change_count);
    const has_enough = count >= min_periods;
    out.validity[row] = has_enough;
    if (has_enough) {
        const n: f64 = @floatFromInt(count);
        out.change_rates[row] = @as(f64, @floatFromInt(change_count)) / n;
        out.mean_abs_delta[row] = sum_abs / n;
        out.max_abs_delta[row] = max_abs;
    } else {
        out.change_rates[row] = 0;
        out.mean_abs_delta[row] = 0;
        out.max_abs_delta[row] = 0;
    }
}

pub fn rollingChangePointProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    threshold: f64,
    periods: usize,
    window: usize,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!ChangeSummaryMetrics {
    if (window == 0) return error.InvalidShape;
    if (min_periods == 0 or min_periods > window) return error.InvalidShape;
    try validate(values, maybe_validity, threshold, periods);

    const rows = values.len;
    const per_row_validity = try allocator.alloc(bool, rows);
    defer allocator.free(per_row_validity);
    const per_row_abs_delta = try allocator.alloc(f64, rows);
    defer allocator.free(per_row_abs_delta);
    const per_row_change = try allocator.alloc(bool, rows);
    defer allocator.free(per_row_change);

    for (0..rows) |row| {
        if (row < periods) {
            per_row_validity[row] = false;
            per_row_abs_delta[row] = 0;
            per_row_change[row] = false;
            continue;
        }

        const previous_row = row - periods;
        const valid = validPair(maybe_validity, row, previous_row);
        per_row_validity[row] = valid;
        if (!valid) {
            per_row_abs_delta[row] = 0;
            per_row_change[row] = false;
            continue;
        }

        const delta = absDelta(values, row, previous_row);
        per_row_abs_delta[row] = delta;
        per_row_change[row] = delta >= threshold;
    }

    var out = try allocSummary(allocator, rows);
    errdefer out.deinit();
    for (0..rows) |row| {
        const start = if (row + 1 > window) row + 1 - window else 0;
        var count: usize = 0;
        var change_count: usize = 0;
        var sum_abs: f64 = 0;
        var max_abs: f64 = 0;
        for (start..row + 1) |window_row| {
            if (!per_row_validity[window_row]) continue;
            const delta = per_row_abs_delta[window_row];
            if (count == 0 or delta > max_abs) max_abs = delta;
            sum_abs += delta;
            if (per_row_change[window_row]) change_count += 1;
            count += 1;
        }
        writeSummary(row, min_periods, count, change_count, sum_abs, max_abs, out);
    }

    return out;
}

pub fn expandingChangePointProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    threshold: f64,
    periods: usize,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!ChangeSummaryMetrics {
    if (min_periods == 0) return error.InvalidShape;
    try validate(values, maybe_validity, threshold, periods);

    var out = try allocSummary(allocator, values.len);
    errdefer out.deinit();

    var count: usize = 0;
    var change_count: usize = 0;
    var sum_abs: f64 = 0;
    var max_abs: f64 = 0;
    for (0..values.len) |row| {
        if (row >= periods) {
            const previous_row = row - periods;
            if (validPair(maybe_validity, row, previous_row)) {
                const delta = absDelta(values, row, previous_row);
                if (count == 0 or delta > max_abs) max_abs = delta;
                sum_abs += delta;
                if (delta >= threshold) change_count += 1;
                count += 1;
            }
        }
        writeSummary(row, min_periods, count, change_count, sum_abs, max_abs, out);
    }

    return out;
}
