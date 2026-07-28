//! Linear regression metric kernels and output-name helpers.

const std = @import("std");

pub const LinearFitMetrics = struct {
    allocator: std.mem.Allocator,
    fitted: []f64,
    residuals: []f64,
    residual_z: []f64,
    slopes: []f64,
    validity: []bool,

    pub fn deinit(self: *LinearFitMetrics) void {
        self.allocator.free(self.fitted);
        self.allocator.free(self.residuals);
        self.allocator.free(self.residual_z);
        self.allocator.free(self.slopes);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

const FitParams = struct {
    has_fit: bool,
    slope: f64,
    intercept: f64,
};

pub const WindowLinearFitMetrics = struct {
    allocator: std.mem.Allocator,
    pair_counts: []i64,
    slopes: []f64,
    intercepts: []f64,
    fitted: []f64,
    residuals: []f64,
    residual_z: []f64,
    fit_validity: []bool,
    row_validity: []bool,

    pub fn deinit(self: *WindowLinearFitMetrics) void {
        self.allocator.free(self.pair_counts);
        self.allocator.free(self.slopes);
        self.allocator.free(self.intercepts);
        self.allocator.free(self.fitted);
        self.allocator.free(self.residuals);
        self.allocator.free(self.residual_z);
        self.allocator.free(self.fit_validity);
        self.allocator.free(self.row_validity);
        self.* = undefined;
    }
};

pub const LinearFitProfileColumnCount = 4;

pub fn linearFitProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![LinearFitProfileColumnCount][]const u8 {
    var names: [LinearFitProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "fitted", "residual", "residual_zscore", "slope" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const ExpandingLinearFitProfileColumnCount = 6;

pub fn expandingLinearFitProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![ExpandingLinearFitProfileColumnCount][]const u8 {
    var names: [ExpandingLinearFitProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "expanding_pair_count", "expanding_slope", "expanding_intercept", "expanding_fitted", "expanding_residual", "expanding_residual_zscore" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const RollingLinearFitProfileColumnCount = 6;

pub fn rollingLinearFitProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![RollingLinearFitProfileColumnCount][]const u8 {
    var names: [RollingLinearFitProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "rolling_pair_count", "rolling_slope", "rolling_intercept", "rolling_fitted", "rolling_residual", "rolling_residual_zscore" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn validatePairLengths(xs: []const f64, ys: []const f64, maybe_x_validity: ?[]const bool, maybe_y_validity: ?[]const bool) error{LengthMismatch}!void {
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

fn fitFromSums(count: usize, sum_x: f64, sum_y: f64, sum_xx: f64, sum_xy: f64, min_periods: usize) FitParams {
    const has_fit = count >= min_periods;
    if (!has_fit) return .{ .has_fit = false, .slope = std.math.nan(f64), .intercept = std.math.nan(f64) };
    const n: f64 = @floatFromInt(count);
    const denom = n * sum_xx - sum_x * sum_x;
    const slope = if (denom == 0) std.math.nan(f64) else (n * sum_xy - sum_x * sum_y) / denom;
    const intercept = if (std.math.isNan(slope)) std.math.nan(f64) else (sum_y - slope * sum_x) / n;
    return .{ .has_fit = true, .slope = slope, .intercept = intercept };
}

fn residualStd(xs: []const f64, ys: []const f64, maybe_x_validity: ?[]const bool, maybe_y_validity: ?[]const bool, start: usize, end: usize, count: usize, slope: f64, intercept: f64) f64 {
    if (count == 0 or std.math.isNan(slope)) return std.math.nan(f64);
    var residual_sum_sq: f64 = 0;
    for (start..end) |row| {
        if (!rowValid(maybe_x_validity, maybe_y_validity, row)) continue;
        const fit = intercept + slope * xs[row];
        const residual = ys[row] - fit;
        residual_sum_sq += residual * residual;
    }
    return std.math.sqrt(residual_sum_sq / @as(f64, @floatFromInt(count)));
}

fn allocWindowMetrics(allocator: std.mem.Allocator, rows: usize) std.mem.Allocator.Error!WindowLinearFitMetrics {
    const pair_counts = try allocator.alloc(i64, rows);
    errdefer allocator.free(pair_counts);
    const slopes = try allocator.alloc(f64, rows);
    errdefer allocator.free(slopes);
    const intercepts = try allocator.alloc(f64, rows);
    errdefer allocator.free(intercepts);
    const fitted = try allocator.alloc(f64, rows);
    errdefer allocator.free(fitted);
    const residuals = try allocator.alloc(f64, rows);
    errdefer allocator.free(residuals);
    const residual_z = try allocator.alloc(f64, rows);
    errdefer allocator.free(residual_z);
    const fit_validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(fit_validity);
    const row_validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(row_validity);
    return .{
        .allocator = allocator,
        .pair_counts = pair_counts,
        .slopes = slopes,
        .intercepts = intercepts,
        .fitted = fitted,
        .residuals = residuals,
        .residual_z = residual_z,
        .fit_validity = fit_validity,
        .row_validity = row_validity,
    };
}

fn writeWindowRow(row: usize, x: f64, y: f64, count: usize, fit: FitParams, stddev: f64, current_valid: bool, out: WindowLinearFitMetrics) void {
    out.pair_counts[row] = @intCast(count);
    out.fit_validity[row] = fit.has_fit;
    if (!fit.has_fit) {
        out.slopes[row] = 0;
        out.intercepts[row] = 0;
        out.fitted[row] = 0;
        out.residuals[row] = 0;
        out.residual_z[row] = 0;
        out.row_validity[row] = false;
        return;
    }

    out.slopes[row] = fit.slope;
    out.intercepts[row] = fit.intercept;
    out.row_validity[row] = current_valid;
    if (current_valid) {
        const fitted = fit.intercept + fit.slope * x;
        const residual = y - fitted;
        out.fitted[row] = fitted;
        out.residuals[row] = residual;
        out.residual_z[row] = if (stddev == 0 or std.math.isNan(stddev)) std.math.nan(f64) else residual / stddev;
    } else {
        out.fitted[row] = 0;
        out.residuals[row] = 0;
        out.residual_z[row] = 0;
    }
}

pub fn linearFitProfile(
    allocator: std.mem.Allocator,
    xs: []const f64,
    ys: []const f64,
    maybe_x_validity: ?[]const bool,
    maybe_y_validity: ?[]const bool,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!LinearFitMetrics {
    if (min_periods == 0) return error.InvalidShape;
    try validatePairLengths(xs, ys, maybe_x_validity, maybe_y_validity);

    var count: usize = 0;
    var sum_x: f64 = 0;
    var sum_y: f64 = 0;
    var sum_xx: f64 = 0;
    var sum_xy: f64 = 0;
    for (xs, ys, 0..) |x, y, row| {
        if (!rowValid(maybe_x_validity, maybe_y_validity, row)) continue;
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
        count += 1;
    }

    const rows = xs.len;
    const fitted = try allocator.alloc(f64, rows);
    errdefer allocator.free(fitted);
    const residuals = try allocator.alloc(f64, rows);
    errdefer allocator.free(residuals);
    const residual_z = try allocator.alloc(f64, rows);
    errdefer allocator.free(residual_z);
    const slopes = try allocator.alloc(f64, rows);
    errdefer allocator.free(slopes);
    const validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(validity);

    const fit = fitFromSums(count, sum_x, sum_y, sum_xx, sum_xy, min_periods);
    const stddev = residualStd(xs, ys, maybe_x_validity, maybe_y_validity, 0, rows, count, fit.slope, fit.intercept);

    for (xs, ys, 0..) |x, y, row| {
        const valid = rowValid(maybe_x_validity, maybe_y_validity, row) and fit.has_fit;
        validity[row] = valid;
        if (valid) {
            const fitted_value = fit.intercept + fit.slope * x;
            const residual = y - fitted_value;
            fitted[row] = fitted_value;
            residuals[row] = residual;
            residual_z[row] = if (stddev == 0 or std.math.isNan(stddev)) std.math.nan(f64) else residual / stddev;
            slopes[row] = fit.slope;
        } else {
            fitted[row] = 0;
            residuals[row] = 0;
            residual_z[row] = 0;
            slopes[row] = 0;
        }
    }

    return .{
        .allocator = allocator,
        .fitted = fitted,
        .residuals = residuals,
        .residual_z = residual_z,
        .slopes = slopes,
        .validity = validity,
    };
}

pub fn expandingLinearFitProfile(
    allocator: std.mem.Allocator,
    xs: []const f64,
    ys: []const f64,
    maybe_x_validity: ?[]const bool,
    maybe_y_validity: ?[]const bool,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!WindowLinearFitMetrics {
    if (min_periods == 0) return error.InvalidShape;
    try validatePairLengths(xs, ys, maybe_x_validity, maybe_y_validity);

    var out = try allocWindowMetrics(allocator, xs.len);
    errdefer out.deinit();

    var count: usize = 0;
    var sum_x: f64 = 0;
    var sum_y: f64 = 0;
    var sum_xx: f64 = 0;
    var sum_xy: f64 = 0;
    for (xs, ys, 0..) |x, y, row| {
        if (rowValid(maybe_x_validity, maybe_y_validity, row)) {
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
            count += 1;
        }

        const fit = fitFromSums(count, sum_x, sum_y, sum_xx, sum_xy, min_periods);
        const stddev = residualStd(xs, ys, maybe_x_validity, maybe_y_validity, 0, row + 1, count, fit.slope, fit.intercept);
        writeWindowRow(row, x, y, count, fit, stddev, rowValid(maybe_x_validity, maybe_y_validity, row), out);
    }

    return out;
}

pub fn rollingLinearFitProfile(
    allocator: std.mem.Allocator,
    xs: []const f64,
    ys: []const f64,
    maybe_x_validity: ?[]const bool,
    maybe_y_validity: ?[]const bool,
    window: usize,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!WindowLinearFitMetrics {
    if (window == 0) return error.InvalidShape;
    if (min_periods == 0 or min_periods > window) return error.InvalidShape;
    try validatePairLengths(xs, ys, maybe_x_validity, maybe_y_validity);

    var out = try allocWindowMetrics(allocator, xs.len);
    errdefer out.deinit();

    // Each row receives the ordinary least-squares line fitted over its trailing
    // valid-pair window. Recomputing windows on the host matches the current
    // rolling correlation implementation and keeps a public seam for future
    // device-side rolling regression kernels.
    for (xs, ys, 0..) |x_current, y_current, row| {
        const start = if (row + 1 > window) row + 1 - window else 0;
        var count: usize = 0;
        var sum_x: f64 = 0;
        var sum_y: f64 = 0;
        var sum_xx: f64 = 0;
        var sum_xy: f64 = 0;
        for (start..row + 1) |window_row| {
            if (!rowValid(maybe_x_validity, maybe_y_validity, window_row)) continue;
            const x = xs[window_row];
            const y = ys[window_row];
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
            count += 1;
        }

        const fit = fitFromSums(count, sum_x, sum_y, sum_xx, sum_xy, min_periods);
        const stddev = residualStd(xs, ys, maybe_x_validity, maybe_y_validity, start, row + 1, count, fit.slope, fit.intercept);
        writeWindowRow(row, x_current, y_current, count, fit, stddev, rowValid(maybe_x_validity, maybe_y_validity, row), out);
    }

    return out;
}
