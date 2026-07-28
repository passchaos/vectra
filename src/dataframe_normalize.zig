const std = @import("std");

pub const NormalizeMetrics = struct {
    allocator: std.mem.Allocator,
    centered: []f64,
    zscores: []f64,
    minmax: []f64,
    validity: []bool,

    pub fn deinit(self: *NormalizeMetrics) void {
        self.allocator.free(self.centered);
        self.allocator.free(self.zscores);
        self.allocator.free(self.minmax);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

fn validate(values: []const f64, maybe_validity: ?[]const bool) error{LengthMismatch}!void {
    if (maybe_validity) |validity| {
        if (validity.len != values.len) return error.LengthMismatch;
    }
}

fn rowValid(maybe_validity: ?[]const bool, row: usize) bool {
    return if (maybe_validity) |mask| mask[row] else true;
}

fn allocMetrics(allocator: std.mem.Allocator, rows: usize) std.mem.Allocator.Error!NormalizeMetrics {
    const centered = try allocator.alloc(f64, rows);
    errdefer allocator.free(centered);
    const zscores = try allocator.alloc(f64, rows);
    errdefer allocator.free(zscores);
    const minmax = try allocator.alloc(f64, rows);
    errdefer allocator.free(minmax);
    const validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(validity);
    return .{ .allocator = allocator, .centered = centered, .zscores = zscores, .minmax = minmax, .validity = validity };
}

fn writeInvalid(out: NormalizeMetrics, row: usize) void {
    out.centered[row] = 0;
    out.zscores[row] = 0;
    out.minmax[row] = 0;
    out.validity[row] = false;
}

fn writeNormalized(out: NormalizeMetrics, row: usize, x: f64, count: usize, sum: f64, sum_sq: f64, low: f64, high: f64, min_periods: usize, current_valid: bool) void {
    const has_enough = current_valid and count >= min_periods;
    out.validity[row] = has_enough;
    if (!has_enough) {
        out.centered[row] = 0;
        out.zscores[row] = 0;
        out.minmax[row] = 0;
        return;
    }

    const n: f64 = @floatFromInt(count);
    const mean = sum / n;
    const raw_variance = sum_sq / n - mean * mean;
    const variance = if (raw_variance < 0) 0 else raw_variance;
    const stddev = std.math.sqrt(variance);
    const range = high - low;
    const delta = x - mean;
    out.centered[row] = delta;
    out.zscores[row] = if (stddev == 0) std.math.nan(f64) else delta / stddev;
    out.minmax[row] = if (range == 0) std.math.nan(f64) else (x - low) / range;
}

pub fn rollingNormalizeProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    window: usize,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!NormalizeMetrics {
    if (window == 0) return error.InvalidShape;
    if (min_periods == 0 or min_periods > window) return error.InvalidShape;
    try validate(values, maybe_validity);

    var out = try allocMetrics(allocator, values.len);
    errdefer out.deinit();

    for (values, 0..) |value, row| {
        const start = if (row + 1 > window) row + 1 - window else 0;
        var count: usize = 0;
        var sum: f64 = 0;
        var sum_sq: f64 = 0;
        var low: f64 = 0;
        var high: f64 = 0;
        for (start..row + 1) |window_row| {
            if (!rowValid(maybe_validity, window_row)) continue;
            const x = values[window_row];
            if (count == 0) {
                low = x;
                high = x;
            } else {
                if (x < low) low = x;
                if (x > high) high = x;
            }
            sum += x;
            sum_sq += x * x;
            count += 1;
        }
        writeNormalized(out, row, value, count, sum, sum_sq, low, high, min_periods, rowValid(maybe_validity, row));
    }

    return out;
}

pub fn expandingNormalizeProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!NormalizeMetrics {
    if (min_periods == 0) return error.InvalidShape;
    try validate(values, maybe_validity);

    var out = try allocMetrics(allocator, values.len);
    errdefer out.deinit();

    var count: usize = 0;
    var sum: f64 = 0;
    var sum_sq: f64 = 0;
    var low: f64 = 0;
    var high: f64 = 0;
    for (values, 0..) |value, row| {
        const valid = rowValid(maybe_validity, row);
        if (valid) {
            if (count == 0) {
                low = value;
                high = value;
            } else {
                if (value < low) low = value;
                if (value > high) high = value;
            }
            sum += value;
            sum_sq += value * value;
            count += 1;
        }
        writeNormalized(out, row, value, count, sum, sum_sq, low, high, min_periods, valid);
    }

    return out;
}
