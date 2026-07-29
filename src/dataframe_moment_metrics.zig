//! Higher-order moment metric kernels and output-name helpers.

const std = @import("std");

pub const MomentMetrics = struct {
    allocator: std.mem.Allocator,
    counts: []i64,
    m3_values: []f64,
    m4_values: []f64,
    skewnesses: []f64,
    kurtoses: []f64,
    validity: []bool,

    pub fn deinit(self: *MomentMetrics) void {
        self.allocator.free(self.counts);
        self.allocator.free(self.m3_values);
        self.allocator.free(self.m4_values);
        self.allocator.free(self.skewnesses);
        self.allocator.free(self.kurtoses);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub const RollingMomentProfileColumnCount = 5;

pub fn rollingMomentProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![RollingMomentProfileColumnCount][]const u8 {
    var names: [RollingMomentProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "rolling_moment_count", "rolling_m3", "rolling_m4", "rolling_skewness", "rolling_kurtosis" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const ExpandingMomentProfileColumnCount = 5;

pub fn expandingMomentProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![ExpandingMomentProfileColumnCount][]const u8 {
    var names: [ExpandingMomentProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "expanding_moment_count", "expanding_m3", "expanding_m4", "expanding_skewness", "expanding_kurtosis" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

const OnlineMoment = struct {
    count: i64 = 0,
    mean: f64 = 0,
    m2: f64 = 0,
    m3: f64 = 0,
    m4: f64 = 0,

    fn update(self: *OnlineMoment, value: f64) void {
        const n1: f64 = @floatFromInt(self.count);
        self.count += 1;
        const n: f64 = @floatFromInt(self.count);
        const delta = value - self.mean;
        const delta_n = delta / n;
        const delta_n2 = delta_n * delta_n;
        const term1 = delta * delta_n * n1;
        self.mean += delta_n;
        self.m4 += term1 * delta_n2 * (n * n - 3.0 * n + 3.0) + 6.0 * delta_n2 * self.m2 - 4.0 * delta_n * self.m3;
        self.m3 += term1 * delta_n * (n - 2.0) - 3.0 * delta_n * self.m2;
        self.m2 += term1;
    }

    fn skewness(self: OnlineMoment) f64 {
        if (self.count < 2 or self.m2 == 0) return std.math.nan(f64);
        const n: f64 = @floatFromInt(self.count);
        return std.math.sqrt(n) * self.m3 / std.math.pow(f64, self.m2, 1.5);
    }

    fn kurtosis(self: OnlineMoment) f64 {
        if (self.count < 2 or self.m2 == 0) return std.math.nan(f64);
        const n: f64 = @floatFromInt(self.count);
        return n * self.m4 / (self.m2 * self.m2) - 3.0;
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

fn allocMetrics(allocator: std.mem.Allocator, rows: usize) std.mem.Allocator.Error!MomentMetrics {
    const counts = try allocator.alloc(i64, rows);
    errdefer allocator.free(counts);
    const m3_values = try allocator.alloc(f64, rows);
    errdefer allocator.free(m3_values);
    const m4_values = try allocator.alloc(f64, rows);
    errdefer allocator.free(m4_values);
    const skewnesses = try allocator.alloc(f64, rows);
    errdefer allocator.free(skewnesses);
    const kurtoses = try allocator.alloc(f64, rows);
    errdefer allocator.free(kurtoses);
    const validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(validity);
    return .{ .allocator = allocator, .counts = counts, .m3_values = m3_values, .m4_values = m4_values, .skewnesses = skewnesses, .kurtoses = kurtoses, .validity = validity };
}

fn writeInvalid(out: MomentMetrics, row: usize, count: usize) void {
    out.counts[row] = @intCast(count);
    out.validity[row] = false;
    out.m3_values[row] = 0;
    out.m4_values[row] = 0;
    out.skewnesses[row] = 0;
    out.kurtoses[row] = 0;
}

fn writeCentralMoments(out: MomentMetrics, row: usize, count: usize, sum2: f64, sum3: f64, sum4: f64, min_periods: usize) void {
    out.counts[row] = @intCast(count);
    const has_enough = count >= min_periods;
    out.validity[row] = has_enough;
    if (!has_enough) {
        out.m3_values[row] = 0;
        out.m4_values[row] = 0;
        out.skewnesses[row] = 0;
        out.kurtoses[row] = 0;
        return;
    }
    const n: f64 = @floatFromInt(count);
    const variance = sum2 / n;
    const m3 = sum3 / n;
    const m4 = sum4 / n;
    out.m3_values[row] = m3;
    out.m4_values[row] = m4;
    if (count < 2 or variance == 0) {
        out.skewnesses[row] = std.math.nan(f64);
        out.kurtoses[row] = std.math.nan(f64);
    } else {
        out.skewnesses[row] = m3 / std.math.pow(f64, variance, 1.5);
        out.kurtoses[row] = m4 / (variance * variance) - 3.0;
    }
}

pub fn rollingMomentProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    window: usize,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!MomentMetrics {
    if (window == 0) return error.InvalidShape;
    if (min_periods == 0 or min_periods > window) return error.InvalidShape;
    try validate(values, maybe_validity);

    var out = try allocMetrics(allocator, values.len);
    errdefer out.deinit();

    for (0..values.len) |row| {
        const start = if (row + 1 > window) row + 1 - window else 0;
        var count: usize = 0;
        var sum: f64 = 0;
        for (start..row + 1) |window_row| {
            if (!rowValid(maybe_validity, window_row)) continue;
            sum += values[window_row];
            count += 1;
        }
        if (count < min_periods) {
            writeInvalid(out, row, count);
            continue;
        }

        const mean = sum / @as(f64, @floatFromInt(count));
        var sum2: f64 = 0;
        var sum3: f64 = 0;
        var sum4: f64 = 0;
        for (start..row + 1) |window_row| {
            if (!rowValid(maybe_validity, window_row)) continue;
            const centered = values[window_row] - mean;
            const centered2 = centered * centered;
            sum2 += centered2;
            sum3 += centered2 * centered;
            sum4 += centered2 * centered2;
        }
        writeCentralMoments(out, row, count, sum2, sum3, sum4, min_periods);
    }

    return out;
}

pub fn expandingMomentProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!MomentMetrics {
    if (min_periods == 0) return error.InvalidShape;
    try validate(values, maybe_validity);

    var out = try allocMetrics(allocator, values.len);
    errdefer out.deinit();

    var profile: OnlineMoment = .{};
    for (values, 0..) |value, row| {
        if (rowValid(maybe_validity, row)) profile.update(value);

        out.counts[row] = profile.count;
        const has_enough = profile.count >= @as(i64, @intCast(min_periods));
        out.validity[row] = has_enough;
        if (has_enough) {
            const n: f64 = @floatFromInt(profile.count);
            out.m3_values[row] = profile.m3 / n;
            out.m4_values[row] = profile.m4 / n;
            out.skewnesses[row] = profile.skewness();
            out.kurtoses[row] = profile.kurtosis();
        } else {
            out.m3_values[row] = 0;
            out.m4_values[row] = 0;
            out.skewnesses[row] = 0;
            out.kurtoses[row] = 0;
        }
    }

    return out;
}
