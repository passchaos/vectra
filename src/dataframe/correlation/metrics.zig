//! Correlation metric kernels and output-name helpers.

const std = @import("std");

pub const CorrelationMetrics = struct {
    allocator: std.mem.Allocator,
    pair_counts: []i64,
    covariances: []f64,
    correlations: []f64,
    betas: []f64,
    validity: []bool,

    pub fn deinit(self: *CorrelationMetrics) void {
        self.allocator.free(self.pair_counts);
        self.allocator.free(self.covariances);
        self.allocator.free(self.correlations);
        self.allocator.free(self.betas);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub const RollingCorrelationProfileColumnCount = 4;

pub fn rollingCorrelationProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![RollingCorrelationProfileColumnCount][]const u8 {
    var names: [RollingCorrelationProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "rolling_pair_count", "rolling_covariance", "rolling_correlation", "rolling_beta" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const ExpandingCorrelationProfileColumnCount = 4;

pub fn expandingCorrelationProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![ExpandingCorrelationProfileColumnCount][]const u8 {
    var names: [ExpandingCorrelationProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "expanding_pair_count", "expanding_covariance", "expanding_correlation", "expanding_beta" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn validatePairLengths(
    xs: []const f64,
    ys: []const f64,
    maybe_x_validity: ?[]const bool,
    maybe_y_validity: ?[]const bool,
) error{LengthMismatch}!void {
    if (xs.len != ys.len) return error.LengthMismatch;
    if (maybe_x_validity) |validity| {
        if (validity.len != xs.len) return error.LengthMismatch;
    }
    if (maybe_y_validity) |validity| {
        if (validity.len != ys.len) return error.LengthMismatch;
    }
}

fn rowValid(maybe_x_validity: ?[]const bool, maybe_y_validity: ?[]const bool, row: usize) bool {
    return (if (maybe_x_validity) |mask| mask[row] else true) and (if (maybe_y_validity) |mask| mask[row] else true);
}

fn allocMetrics(allocator: std.mem.Allocator, rows: usize) std.mem.Allocator.Error!CorrelationMetrics {
    const pair_counts = try allocator.alloc(i64, rows);
    errdefer allocator.free(pair_counts);
    const covariances = try allocator.alloc(f64, rows);
    errdefer allocator.free(covariances);
    const correlations = try allocator.alloc(f64, rows);
    errdefer allocator.free(correlations);
    const betas = try allocator.alloc(f64, rows);
    errdefer allocator.free(betas);
    const validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(validity);
    return .{
        .allocator = allocator,
        .pair_counts = pair_counts,
        .covariances = covariances,
        .correlations = correlations,
        .betas = betas,
        .validity = validity,
    };
}

fn writeStats(
    row: usize,
    min_periods: usize,
    count: usize,
    sum_x: f64,
    sum_y: f64,
    sum_xx: f64,
    sum_yy: f64,
    sum_xy: f64,
    out: CorrelationMetrics,
) void {
    out.pair_counts[row] = @intCast(count);
    const has_enough = count >= min_periods;
    out.validity[row] = has_enough;
    if (has_enough) {
        const n: f64 = @floatFromInt(count);
        const mean_x = sum_x / n;
        const mean_y = sum_y / n;
        const cov = sum_xy / n - mean_x * mean_y;
        const var_x_raw = sum_xx / n - mean_x * mean_x;
        const var_y_raw = sum_yy / n - mean_y * mean_y;
        const var_x = if (var_x_raw < 0) 0 else var_x_raw;
        const var_y = if (var_y_raw < 0) 0 else var_y_raw;
        out.covariances[row] = cov;
        out.correlations[row] = if (var_x == 0 or var_y == 0) std.math.nan(f64) else cov / std.math.sqrt(var_x * var_y);
        out.betas[row] = if (var_x == 0) std.math.nan(f64) else cov / var_x;
    } else {
        out.covariances[row] = 0;
        out.correlations[row] = 0;
        out.betas[row] = 0;
    }
}

pub fn rollingCorrelationProfile(
    allocator: std.mem.Allocator,
    xs: []const f64,
    ys: []const f64,
    maybe_x_validity: ?[]const bool,
    maybe_y_validity: ?[]const bool,
    window: usize,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!CorrelationMetrics {
    if (window == 0) return error.InvalidShape;
    if (min_periods == 0 or min_periods > window) return error.InvalidShape;
    try validatePairLengths(xs, ys, maybe_x_validity, maybe_y_validity);

    var out = try allocMetrics(allocator, xs.len);
    errdefer out.deinit();

    // Recompute each trailing window in host memory, mirroring the dataframe
    // rolling profile APIs while retaining a stable seam for future device-side
    // rolling covariance/correlation kernels.
    for (0..xs.len) |row| {
        const start = if (row + 1 > window) row + 1 - window else 0;
        var count: usize = 0;
        var sum_x: f64 = 0;
        var sum_y: f64 = 0;
        var sum_xx: f64 = 0;
        var sum_yy: f64 = 0;
        var sum_xy: f64 = 0;
        for (start..row + 1) |window_row| {
            if (!rowValid(maybe_x_validity, maybe_y_validity, window_row)) continue;
            const x = xs[window_row];
            const y = ys[window_row];
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_yy += y * y;
            sum_xy += x * y;
            count += 1;
        }
        writeStats(row, min_periods, count, sum_x, sum_y, sum_xx, sum_yy, sum_xy, out);
    }

    return out;
}

pub fn expandingCorrelationProfile(
    allocator: std.mem.Allocator,
    xs: []const f64,
    ys: []const f64,
    maybe_x_validity: ?[]const bool,
    maybe_y_validity: ?[]const bool,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!CorrelationMetrics {
    if (min_periods == 0) return error.InvalidShape;
    try validatePairLengths(xs, ys, maybe_x_validity, maybe_y_validity);

    var out = try allocMetrics(allocator, xs.len);
    errdefer out.deinit();

    var count: usize = 0;
    var sum_x: f64 = 0;
    var sum_y: f64 = 0;
    var sum_xx: f64 = 0;
    var sum_yy: f64 = 0;
    var sum_xy: f64 = 0;
    for (xs, ys, 0..) |x, y, row| {
        if (rowValid(maybe_x_validity, maybe_y_validity, row)) {
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_yy += y * y;
            sum_xy += x * y;
            count += 1;
        }
        writeStats(row, min_periods, count, sum_x, sum_y, sum_xx, sum_yy, sum_xy, out);
    }

    return out;
}
