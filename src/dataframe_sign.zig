const std = @import("std");

pub const SignMetrics = struct {
    allocator: std.mem.Allocator,
    signs: []i64,
    flips: []bool,
    positive_streak: []i64,
    negative_streak: []i64,
    zero_streak: []i64,
    sign_validity: []bool,
    flip_validity: []bool,

    pub fn deinit(self: *SignMetrics) void {
        self.allocator.free(self.signs);
        self.allocator.free(self.flips);
        self.allocator.free(self.positive_streak);
        self.allocator.free(self.negative_streak);
        self.allocator.free(self.zero_streak);
        self.allocator.free(self.sign_validity);
        self.allocator.free(self.flip_validity);
        self.* = undefined;
    }
};

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

fn validate(values: []const f64, maybe_validity: ?[]const bool, periods: usize) error{ InvalidShape, LengthMismatch }!void {
    if (periods == 0) return error.InvalidShape;
    if (maybe_validity) |validity| {
        if (validity.len != values.len) return error.LengthMismatch;
    }
}

fn rowValid(maybe_validity: ?[]const bool, row: usize) bool {
    return if (maybe_validity) |mask| mask[row] else true;
}

fn signOf(value: f64) i64 {
    return if (value > 0) 1 else if (value < 0) -1 else 0;
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

fn computeSignEvents(allocator: std.mem.Allocator, values: []const f64, maybe_validity: ?[]const bool, periods: usize) !struct { signs: []i64, flips: []bool, sign_validity: []bool, flip_validity: []bool } {
    const signs = try allocator.alloc(i64, values.len);
    errdefer allocator.free(signs);
    const flips = try allocator.alloc(bool, values.len);
    errdefer allocator.free(flips);
    const sign_validity = try allocator.alloc(bool, values.len);
    errdefer allocator.free(sign_validity);
    const flip_validity = try allocator.alloc(bool, values.len);
    errdefer allocator.free(flip_validity);

    for (values, 0..) |value, row| {
        const valid = rowValid(maybe_validity, row);
        sign_validity[row] = valid;
        if (!valid) {
            signs[row] = 0;
            flips[row] = false;
            flip_validity[row] = false;
            continue;
        }

        const sign = signOf(value);
        signs[row] = sign;
        if (row < periods) {
            flips[row] = false;
            flip_validity[row] = false;
        } else {
            const previous_row = row - periods;
            const previous_valid = rowValid(maybe_validity, previous_row);
            flip_validity[row] = previous_valid;
            flips[row] = if (previous_valid) sign != signOf(values[previous_row]) else false;
        }
    }

    return .{ .signs = signs, .flips = flips, .sign_validity = sign_validity, .flip_validity = flip_validity };
}

pub fn signProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!SignMetrics {
    try validate(values, maybe_validity, periods);

    const events = try computeSignEvents(allocator, values, maybe_validity, periods);
    errdefer allocator.free(events.signs);
    errdefer allocator.free(events.flips);
    errdefer allocator.free(events.sign_validity);
    errdefer allocator.free(events.flip_validity);

    const positive_streak = try allocator.alloc(i64, values.len);
    errdefer allocator.free(positive_streak);
    const negative_streak = try allocator.alloc(i64, values.len);
    errdefer allocator.free(negative_streak);
    const zero_streak = try allocator.alloc(i64, values.len);
    errdefer allocator.free(zero_streak);

    var pos: i64 = 0;
    var neg: i64 = 0;
    var zero: i64 = 0;
    for (0..values.len) |row| {
        if (!events.sign_validity[row]) {
            positive_streak[row] = 0;
            negative_streak[row] = 0;
            zero_streak[row] = 0;
            pos = 0;
            neg = 0;
            zero = 0;
            continue;
        }

        switch (events.signs[row]) {
            1 => {
                pos += 1;
                neg = 0;
                zero = 0;
            },
            -1 => {
                neg += 1;
                pos = 0;
                zero = 0;
            },
            else => {
                zero += 1;
                pos = 0;
                neg = 0;
            },
        }
        positive_streak[row] = pos;
        negative_streak[row] = neg;
        zero_streak[row] = zero;
    }

    return .{
        .allocator = allocator,
        .signs = events.signs,
        .flips = events.flips,
        .positive_streak = positive_streak,
        .negative_streak = negative_streak,
        .zero_streak = zero_streak,
        .sign_validity = events.sign_validity,
        .flip_validity = events.flip_validity,
    };
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
