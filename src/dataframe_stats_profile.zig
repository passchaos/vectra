const std = @import("std");

pub const RollingStatsMetrics = struct {
    allocator: std.mem.Allocator,
    counts: []i64,
    sums: []f64,
    means: []f64,
    variances: []f64,
    stddevs: []f64,
    validity: []bool,

    pub fn deinit(self: *RollingStatsMetrics) void {
        self.allocator.free(self.counts);
        self.allocator.free(self.sums);
        self.allocator.free(self.means);
        self.allocator.free(self.variances);
        self.allocator.free(self.stddevs);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub const ExpandingStatsMetrics = struct {
    allocator: std.mem.Allocator,
    counts: []i64,
    sums: []f64,
    means: []f64,
    mins: []f64,
    maxes: []f64,
    validity: []bool,

    pub fn deinit(self: *ExpandingStatsMetrics) void {
        self.allocator.free(self.counts);
        self.allocator.free(self.sums);
        self.allocator.free(self.means);
        self.allocator.free(self.mins);
        self.allocator.free(self.maxes);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub const RollingProfileColumnCount = 5;

pub fn rollingProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![RollingProfileColumnCount][]const u8 {
    var names: [RollingProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "rolling_count", "rolling_sum", "rolling_mean", "rolling_variance", "rolling_stddev" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const ExpandingProfileColumnCount = 5;

pub fn expandingProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![ExpandingProfileColumnCount][]const u8 {
    var names: [ExpandingProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "expanding_count", "expanding_sum", "expanding_mean", "expanding_min", "expanding_max" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn validate(values: []const f64, maybe_validity: ?[]const bool) error{LengthMismatch}!void {
    if (maybe_validity) |validity| {
        if (validity.len != values.len) return error.LengthMismatch;
    }
}

fn rowValid(maybe_validity: ?[]const bool, row: usize) bool {
    return if (maybe_validity) |mask| mask[row] else true;
}

pub fn rollingProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    window: usize,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!RollingStatsMetrics {
    if (window == 0) return error.InvalidShape;
    if (min_periods == 0 or min_periods > window) return error.InvalidShape;
    try validate(values, maybe_validity);

    const counts = try allocator.alloc(i64, values.len);
    errdefer allocator.free(counts);
    const sums = try allocator.alloc(f64, values.len);
    errdefer allocator.free(sums);
    const means = try allocator.alloc(f64, values.len);
    errdefer allocator.free(means);
    const variances = try allocator.alloc(f64, values.len);
    errdefer allocator.free(variances);
    const stddevs = try allocator.alloc(f64, values.len);
    errdefer allocator.free(stddevs);
    const validity = try allocator.alloc(bool, values.len);
    errdefer allocator.free(validity);

    var running_sum: f64 = 0;
    var running_sum_sq: f64 = 0;
    var running_count: usize = 0;
    for (values, 0..) |value, row| {
        if (rowValid(maybe_validity, row)) {
            running_sum += value;
            running_sum_sq += value * value;
            running_count += 1;
        }
        if (row >= window) {
            const evict_row = row - window;
            if (rowValid(maybe_validity, evict_row)) {
                const x = values[evict_row];
                running_sum -= x;
                running_sum_sq -= x * x;
                running_count -= 1;
            }
        }

        counts[row] = @intCast(running_count);
        const has_enough = running_count >= min_periods;
        validity[row] = has_enough;
        if (has_enough) {
            const n: f64 = @floatFromInt(running_count);
            const mean = running_sum / n;
            const raw_variance = running_sum_sq / n - mean * mean;
            const variance = if (raw_variance < 0) 0 else raw_variance;
            sums[row] = running_sum;
            means[row] = mean;
            variances[row] = variance;
            stddevs[row] = std.math.sqrt(variance);
        } else {
            sums[row] = 0;
            means[row] = 0;
            variances[row] = 0;
            stddevs[row] = 0;
        }
    }

    return .{ .allocator = allocator, .counts = counts, .sums = sums, .means = means, .variances = variances, .stddevs = stddevs, .validity = validity };
}

pub fn expandingProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!ExpandingStatsMetrics {
    if (min_periods == 0) return error.InvalidShape;
    try validate(values, maybe_validity);

    const counts = try allocator.alloc(i64, values.len);
    errdefer allocator.free(counts);
    const sums = try allocator.alloc(f64, values.len);
    errdefer allocator.free(sums);
    const means = try allocator.alloc(f64, values.len);
    errdefer allocator.free(means);
    const mins = try allocator.alloc(f64, values.len);
    errdefer allocator.free(mins);
    const maxes = try allocator.alloc(f64, values.len);
    errdefer allocator.free(maxes);
    const validity = try allocator.alloc(bool, values.len);
    errdefer allocator.free(validity);

    var running_count: usize = 0;
    var running_sum: f64 = 0;
    var running_min: f64 = 0;
    var running_max: f64 = 0;
    for (values, 0..) |value, row| {
        if (rowValid(maybe_validity, row)) {
            if (running_count == 0) {
                running_min = value;
                running_max = value;
            } else {
                if (value < running_min) running_min = value;
                if (value > running_max) running_max = value;
            }
            running_sum += value;
            running_count += 1;
        }

        counts[row] = @intCast(running_count);
        const has_enough = running_count >= min_periods;
        validity[row] = has_enough;
        if (has_enough) {
            sums[row] = running_sum;
            means[row] = running_sum / @as(f64, @floatFromInt(running_count));
            mins[row] = running_min;
            maxes[row] = running_max;
        } else {
            sums[row] = 0;
            means[row] = 0;
            mins[row] = 0;
            maxes[row] = 0;
        }
    }

    return .{ .allocator = allocator, .counts = counts, .sums = sums, .means = means, .mins = mins, .maxes = maxes, .validity = validity };
}
