const std = @import("std");

pub const StandardizeMetrics = struct {
    allocator: std.mem.Allocator,
    centered: []f64,
    zscores: []f64,
    minmax: []f64,
    validity: []bool,

    pub fn deinit(self: *StandardizeMetrics) void {
        self.allocator.free(self.centered);
        self.allocator.free(self.zscores);
        self.allocator.free(self.minmax);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub const StandardizeProfileColumnCount = 3;

pub fn standardizeProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![StandardizeProfileColumnCount][]const u8 {
    var names: [StandardizeProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "centered", "zscore", "minmax" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn validateLength(values: []const f64, maybe_validity: ?[]const bool) error{LengthMismatch}!void {
    if (maybe_validity) |validity| {
        if (validity.len != values.len) return error.LengthMismatch;
    }
}

fn rowValid(maybe_validity: ?[]const bool, row: usize) bool {
    return if (maybe_validity) |mask| mask[row] else true;
}

pub fn standardizeProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!StandardizeMetrics {
    if (min_periods == 0) return error.InvalidShape;
    try validateLength(values, maybe_validity);

    var count: usize = 0;
    var sum: f64 = 0;
    var sum_sq: f64 = 0;
    var min_value: f64 = 0;
    var max_value: f64 = 0;
    for (values, 0..) |value, row| {
        if (!rowValid(maybe_validity, row)) continue;
        if (count == 0) {
            min_value = value;
            max_value = value;
        } else {
            if (value < min_value) min_value = value;
            if (value > max_value) max_value = value;
        }
        sum += value;
        sum_sq += value * value;
        count += 1;
    }

    const rows = values.len;
    const centered = try allocator.alloc(f64, rows);
    errdefer allocator.free(centered);
    const zscores = try allocator.alloc(f64, rows);
    errdefer allocator.free(zscores);
    const minmax = try allocator.alloc(f64, rows);
    errdefer allocator.free(minmax);
    const validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(validity);

    const has_enough = count >= min_periods;
    const mean = if (count == 0) 0 else sum / @as(f64, @floatFromInt(count));
    const raw_variance = if (count == 0) 0 else sum_sq / @as(f64, @floatFromInt(count)) - mean * mean;
    const variance = if (raw_variance < 0) 0 else raw_variance;
    const stddev = std.math.sqrt(variance);
    const range = max_value - min_value;

    // Generate common whole-column scaling features in a single pass over the
    // materialized values. This mirrors feature-engineering pipelines that ask
    // for centered, z-score, and min-max forms together.
    for (values, 0..) |value, row| {
        const valid = rowValid(maybe_validity, row) and has_enough;
        validity[row] = valid;
        if (valid) {
            const delta = value - mean;
            centered[row] = delta;
            zscores[row] = if (stddev == 0) std.math.nan(f64) else delta / stddev;
            minmax[row] = if (range == 0) std.math.nan(f64) else (value - min_value) / range;
        } else {
            centered[row] = 0;
            zscores[row] = 0;
            minmax[row] = 0;
        }
    }

    return .{
        .allocator = allocator,
        .centered = centered,
        .zscores = zscores,
        .minmax = minmax,
        .validity = validity,
    };
}
