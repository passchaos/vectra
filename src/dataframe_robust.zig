const std = @import("std");

const mad_normal = 0.6744897501960817;

pub const RobustMetrics = struct {
    allocator: std.mem.Allocator,
    centered: []f64,
    mad_zscore: []f64,
    outlier: []bool,
    winsorized: []f64,
    validity: []bool,

    pub fn deinit(self: *RobustMetrics) void {
        self.allocator.free(self.centered);
        self.allocator.free(self.mad_zscore);
        self.allocator.free(self.outlier);
        self.allocator.free(self.winsorized);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub const RobustProfileColumnCount = 4;

pub fn robustProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![RobustProfileColumnCount][]const u8 {
    var names: [RobustProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "median_centered", "mad_zscore", "iqr_outlier", "winsorized" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const RollingRobustProfileColumnCount = 4;

pub fn rollingRobustProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![RollingRobustProfileColumnCount][]const u8 {
    var names: [RollingRobustProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "rolling_median_centered", "rolling_mad_zscore", "rolling_iqr_outlier", "rolling_winsorized" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const ExpandingRobustProfileColumnCount = 4;

pub fn expandingRobustProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![ExpandingRobustProfileColumnCount][]const u8 {
    var names: [ExpandingRobustProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "expanding_median_centered", "expanding_mad_zscore", "expanding_iqr_outlier", "expanding_winsorized" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn validate(values: []const f64, maybe_validity: ?[]const bool, min_periods: usize) error{ InvalidShape, LengthMismatch }!void {
    if (min_periods == 0) return error.InvalidShape;
    if (maybe_validity) |validity| {
        if (validity.len != values.len) return error.LengthMismatch;
    }
}

fn rowValid(maybe_validity: ?[]const bool, row: usize) bool {
    return if (maybe_validity) |mask| mask[row] else true;
}

fn compareFloat(lhs: f64, rhs: f64) bool {
    const lhs_nan = std.math.isNan(lhs);
    const rhs_nan = std.math.isNan(rhs);
    if (lhs_nan != rhs_nan) return !lhs_nan;
    if (lhs_nan and rhs_nan) return false;
    return lhs < rhs;
}

fn quantileSorted(values: []const f64, probability: f64) f64 {
    if (values.len == 0) return std.math.nan(f64);
    if (values.len == 1) return values[0];
    const position = probability * @as(f64, @floatFromInt(values.len - 1));
    const lower: usize = @intFromFloat(@floor(position));
    const upper: usize = if (lower + 1 < values.len and position > @as(f64, @floatFromInt(lower))) lower + 1 else lower;
    const fraction = position - @as(f64, @floatFromInt(lower));
    return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

fn sort(values: []f64) void {
    std.sort.insertion(f64, values, {}, struct {
        fn lessThan(_: void, lhs: f64, rhs: f64) bool {
            return compareFloat(lhs, rhs);
        }
    }.lessThan);
}

fn allocMetrics(allocator: std.mem.Allocator, rows: usize) std.mem.Allocator.Error!RobustMetrics {
    const centered = try allocator.alloc(f64, rows);
    errdefer allocator.free(centered);
    const mad_zscore = try allocator.alloc(f64, rows);
    errdefer allocator.free(mad_zscore);
    const outlier = try allocator.alloc(bool, rows);
    errdefer allocator.free(outlier);
    const winsorized = try allocator.alloc(f64, rows);
    errdefer allocator.free(winsorized);
    const validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(validity);
    return .{
        .allocator = allocator,
        .centered = centered,
        .mad_zscore = mad_zscore,
        .outlier = outlier,
        .winsorized = winsorized,
        .validity = validity,
    };
}

fn fillInvalid(out: RobustMetrics, row: usize) void {
    out.centered[row] = 0;
    out.mad_zscore[row] = 0;
    out.outlier[row] = false;
    out.winsorized[row] = 0;
    out.validity[row] = false;
}

fn robustStats(sorted_values: []const f64, deviations: []f64, iqr_multiplier: f64) struct { median: f64, mad: f64, lower_fence: f64, upper_fence: f64 } {
    const median = quantileSorted(sorted_values, 0.5);
    const q1 = quantileSorted(sorted_values, 0.25);
    const q3 = quantileSorted(sorted_values, 0.75);
    const iqr = q3 - q1;
    const lower_fence = q1 - iqr_multiplier * iqr;
    const upper_fence = q3 + iqr_multiplier * iqr;
    for (sorted_values, deviations[0..sorted_values.len]) |value, *slot| slot.* = @abs(value - median);
    sort(deviations[0..sorted_values.len]);
    const mad = quantileSorted(deviations[0..sorted_values.len], 0.5);
    return .{ .median = median, .mad = mad, .lower_fence = lower_fence, .upper_fence = upper_fence };
}

fn writeRobustRow(out: RobustMetrics, row: usize, value: f64, stats: anytype) void {
    const centered = value - stats.median;
    out.centered[row] = centered;
    out.mad_zscore[row] = if (stats.mad == 0) std.math.nan(f64) else mad_normal * centered / stats.mad;
    out.outlier[row] = value < stats.lower_fence or value > stats.upper_fence;
    out.winsorized[row] = @min(@max(value, stats.lower_fence), stats.upper_fence);
    out.validity[row] = true;
}

pub fn robustProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    min_periods: usize,
    iqr_multiplier: f64,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!RobustMetrics {
    try validate(values, maybe_validity, min_periods);

    var valid_count: usize = 0;
    for (0..values.len) |row| {
        if (rowValid(maybe_validity, row)) valid_count += 1;
    }

    const valid_values = try allocator.alloc(f64, valid_count);
    defer allocator.free(valid_values);
    var write: usize = 0;
    for (values, 0..) |value, row| {
        if (!rowValid(maybe_validity, row)) continue;
        valid_values[write] = value;
        write += 1;
    }
    sort(valid_values);

    var out = try allocMetrics(allocator, values.len);
    errdefer out.deinit();

    const has_enough = valid_count >= min_periods;
    if (!has_enough or valid_count == 0) {
        for (0..values.len) |row| fillInvalid(out, row);
        return out;
    }

    const deviations = try allocator.alloc(f64, valid_count);
    defer allocator.free(deviations);
    const stats = robustStats(valid_values, deviations, iqr_multiplier);

    for (values, 0..) |value, row| {
        if (!rowValid(maybe_validity, row)) {
            fillInvalid(out, row);
            continue;
        }
        writeRobustRow(out, row, value, stats);
    }
    return out;
}

pub fn rollingRobustProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    window: usize,
    min_periods: usize,
    iqr_multiplier: f64,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!RobustMetrics {
    if (window == 0) return error.InvalidShape;
    if (min_periods == 0 or min_periods > window) return error.InvalidShape;
    if (maybe_validity) |validity| if (validity.len != values.len) return error.LengthMismatch;

    var out = try allocMetrics(allocator, values.len);
    errdefer out.deinit();
    const scratch = try allocator.alloc(f64, window);
    defer allocator.free(scratch);
    const deviations = try allocator.alloc(f64, window);
    defer allocator.free(deviations);

    for (values, 0..) |value, row| {
        const start = if (row + 1 > window) row + 1 - window else 0;
        var count: usize = 0;
        for (start..row + 1) |window_row| {
            if (!rowValid(maybe_validity, window_row)) continue;
            scratch[count] = values[window_row];
            count += 1;
        }

        if (!rowValid(maybe_validity, row) or count < min_periods) {
            fillInvalid(out, row);
            continue;
        }

        const window_values = scratch[0..count];
        sort(window_values);
        const stats = robustStats(window_values, deviations, iqr_multiplier);
        writeRobustRow(out, row, value, stats);
    }

    return out;
}

pub fn expandingRobustProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    min_periods: usize,
    iqr_multiplier: f64,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!RobustMetrics {
    try validate(values, maybe_validity, min_periods);

    var out = try allocMetrics(allocator, values.len);
    errdefer out.deinit();
    const scratch = try allocator.alloc(f64, values.len);
    defer allocator.free(scratch);
    const deviations = try allocator.alloc(f64, values.len);
    defer allocator.free(deviations);

    var valid_count: usize = 0;
    for (values, 0..) |value, row| {
        if (rowValid(maybe_validity, row)) {
            scratch[valid_count] = value;
            valid_count += 1;
        }

        if (!rowValid(maybe_validity, row) or valid_count < min_periods) {
            fillInvalid(out, row);
            continue;
        }

        const prefix_values = scratch[0..valid_count];
        sort(prefix_values);
        const stats = robustStats(prefix_values, deviations, iqr_multiplier);
        writeRobustRow(out, row, value, stats);
    }

    return out;
}
