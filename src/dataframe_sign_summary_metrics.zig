//! Rolling and expanding sign summary metric kernels.

const std = @import("std");
const base_mod = @import("dataframe_sign_metrics.zig");

const validate = base_mod.validate;
const rowValid = base_mod.rowValid;
const signOf = base_mod.signOf;
const computeSignEvents = base_mod.computeSignEvents;

pub const SignSummaryMetrics = struct {
    allocator: std.mem.Allocator,
    counts: []i64,
    positive_rates: []f64,
    negative_rates: []f64,
    zero_rates: []f64,
    flip_rates: []f64,
    validity: []bool,

    pub fn deinit(self: *SignSummaryMetrics) void {
        self.allocator.free(self.counts);
        self.allocator.free(self.positive_rates);
        self.allocator.free(self.negative_rates);
        self.allocator.free(self.zero_rates);
        self.allocator.free(self.flip_rates);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub const RollingSignProfileColumnCount = 5;

pub fn rollingSignProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![RollingSignProfileColumnCount][]const u8 {
    var names: [RollingSignProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "rolling_sign_count", "rolling_positive_rate", "rolling_negative_rate", "rolling_zero_rate", "rolling_sign_flip_rate" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const ExpandingSignProfileColumnCount = 5;

pub fn expandingSignProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![ExpandingSignProfileColumnCount][]const u8 {
    var names: [ExpandingSignProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "expanding_sign_count", "expanding_positive_rate", "expanding_negative_rate", "expanding_zero_rate", "expanding_sign_flip_rate" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn allocSummary(allocator: std.mem.Allocator, rows: usize) std.mem.Allocator.Error!SignSummaryMetrics {
    const counts = try allocator.alloc(i64, rows);
    errdefer allocator.free(counts);
    const positive_rates = try allocator.alloc(f64, rows);
    errdefer allocator.free(positive_rates);
    const negative_rates = try allocator.alloc(f64, rows);
    errdefer allocator.free(negative_rates);
    const zero_rates = try allocator.alloc(f64, rows);
    errdefer allocator.free(zero_rates);
    const flip_rates = try allocator.alloc(f64, rows);
    errdefer allocator.free(flip_rates);
    const validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(validity);
    return .{
        .allocator = allocator,
        .counts = counts,
        .positive_rates = positive_rates,
        .negative_rates = negative_rates,
        .zero_rates = zero_rates,
        .flip_rates = flip_rates,
        .validity = validity,
    };
}

fn writeSummary(row: usize, min_periods: usize, count: usize, positive_count: usize, negative_count: usize, zero_count: usize, flip_count: usize, out: SignSummaryMetrics) void {
    out.counts[row] = @intCast(count);
    const has_enough = count >= min_periods;
    out.validity[row] = has_enough;
    if (has_enough) {
        const n: f64 = @floatFromInt(count);
        out.positive_rates[row] = @as(f64, @floatFromInt(positive_count)) / n;
        out.negative_rates[row] = @as(f64, @floatFromInt(negative_count)) / n;
        out.zero_rates[row] = @as(f64, @floatFromInt(zero_count)) / n;
        out.flip_rates[row] = @as(f64, @floatFromInt(flip_count)) / n;
    } else {
        out.positive_rates[row] = 0;
        out.negative_rates[row] = 0;
        out.zero_rates[row] = 0;
        out.flip_rates[row] = 0;
    }
}

pub fn rollingSignProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    periods: usize,
    window: usize,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!SignSummaryMetrics {
    if (window == 0) return error.InvalidShape;
    if (min_periods == 0 or min_periods > window) return error.InvalidShape;
    try validate(values, maybe_validity, periods);

    const events = try computeSignEvents(allocator, values, maybe_validity, periods);
    defer allocator.free(events.signs);
    defer allocator.free(events.flips);
    defer allocator.free(events.sign_validity);
    defer allocator.free(events.flip_validity);

    var out = try allocSummary(allocator, values.len);
    errdefer out.deinit();
    for (0..values.len) |row| {
        const start = if (row + 1 > window) row + 1 - window else 0;
        var count: usize = 0;
        var positive_count: usize = 0;
        var negative_count: usize = 0;
        var zero_count: usize = 0;
        var flip_count: usize = 0;
        for (start..row + 1) |window_row| {
            if (!events.sign_validity[window_row]) continue;
            switch (events.signs[window_row]) {
                1 => positive_count += 1,
                -1 => negative_count += 1,
                else => zero_count += 1,
            }
            if (events.flip_validity[window_row] and events.flips[window_row]) flip_count += 1;
            count += 1;
        }
        writeSummary(row, min_periods, count, positive_count, negative_count, zero_count, flip_count, out);
    }
    return out;
}

pub fn expandingSignProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    periods: usize,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!SignSummaryMetrics {
    if (min_periods == 0) return error.InvalidShape;
    try validate(values, maybe_validity, periods);

    var out = try allocSummary(allocator, values.len);
    errdefer out.deinit();

    var count: usize = 0;
    var positive_count: usize = 0;
    var negative_count: usize = 0;
    var zero_count: usize = 0;
    var flip_count: usize = 0;

    for (values, 0..) |value, row| {
        if (rowValid(maybe_validity, row)) {
            const sign = signOf(value);
            switch (sign) {
                1 => positive_count += 1,
                -1 => negative_count += 1,
                else => zero_count += 1,
            }
            if (row >= periods) {
                const previous_row = row - periods;
                if (rowValid(maybe_validity, previous_row) and sign != signOf(values[previous_row])) flip_count += 1;
            }
            count += 1;
        }
        writeSummary(row, min_periods, count, positive_count, negative_count, zero_count, flip_count, out);
    }

    return out;
}
