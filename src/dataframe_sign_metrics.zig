//! Sign/change-direction metric kernels and output-name helpers.

const std = @import("std");
const summary_metrics_mod = @import("dataframe_sign_summary_metrics.zig");

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

pub const SignProfileColumnCount = 5;

pub fn signProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![SignProfileColumnCount][]const u8 {
    var names: [SignProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "sign", "sign_flip", "positive_streak", "negative_streak", "zero_streak" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const SignSummaryMetrics = summary_metrics_mod.SignSummaryMetrics;
pub const RollingSignProfileColumnCount = summary_metrics_mod.RollingSignProfileColumnCount;
pub const rollingSignProfileOutputNames = summary_metrics_mod.rollingSignProfileOutputNames;
pub const ExpandingSignProfileColumnCount = summary_metrics_mod.ExpandingSignProfileColumnCount;
pub const expandingSignProfileOutputNames = summary_metrics_mod.expandingSignProfileOutputNames;
pub fn validate(values: []const f64, maybe_validity: ?[]const bool, periods: usize) error{ InvalidShape, LengthMismatch }!void {
    if (periods == 0) return error.InvalidShape;
    if (maybe_validity) |validity| {
        if (validity.len != values.len) return error.LengthMismatch;
    }
}

pub fn rowValid(maybe_validity: ?[]const bool, row: usize) bool {
    return if (maybe_validity) |mask| mask[row] else true;
}

pub fn signOf(value: f64) i64 {
    return if (value > 0) 1 else if (value < 0) -1 else 0;
}

pub fn computeSignEvents(allocator: std.mem.Allocator, values: []const f64, maybe_validity: ?[]const bool, periods: usize) !struct { signs: []i64, flips: []bool, sign_validity: []bool, flip_validity: []bool } {
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

pub const rollingSignProfile = summary_metrics_mod.rollingSignProfile;
pub const expandingSignProfile = summary_metrics_mod.expandingSignProfile;
