//! Threshold distance metric kernels and output-name helpers.

const std = @import("std");
const summary_metrics_mod = @import("dataframe_threshold_summary_metrics.zig");

pub const ThresholdMetrics = struct {
    allocator: std.mem.Allocator,
    distances: []f64,
    abs_distances: []f64,
    above: []bool,
    below: []bool,
    at: []bool,
    validity: []bool,

    pub fn deinit(self: *ThresholdMetrics) void {
        self.allocator.free(self.distances);
        self.allocator.free(self.abs_distances);
        self.allocator.free(self.above);
        self.allocator.free(self.below);
        self.allocator.free(self.at);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub const ThresholdProfileColumnCount = 5;

pub fn thresholdProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![ThresholdProfileColumnCount][]const u8 {
    var names: [ThresholdProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "distance", "abs_distance", "above", "below", "at" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const ThresholdSummaryMetrics = summary_metrics_mod.ThresholdSummaryMetrics;
pub const RollingThresholdProfileColumnCount = summary_metrics_mod.RollingThresholdProfileColumnCount;
pub const rollingThresholdProfileOutputNames = summary_metrics_mod.rollingThresholdProfileOutputNames;
pub const ExpandingThresholdProfileColumnCount = summary_metrics_mod.ExpandingThresholdProfileColumnCount;
pub const expandingThresholdProfileOutputNames = summary_metrics_mod.expandingThresholdProfileOutputNames;
pub fn validateLengths(values: []const f64, maybe_validity: ?[]const bool) error{LengthMismatch}!void {
    if (maybe_validity) |validity| {
        if (validity.len != values.len) return error.LengthMismatch;
    }
}

pub fn rowValid(maybe_validity: ?[]const bool, row: usize) bool {
    return if (maybe_validity) |mask| mask[row] else true;
}

pub fn thresholdProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    threshold: f64,
) (std.mem.Allocator.Error || error{LengthMismatch})!ThresholdMetrics {
    try validateLengths(values, maybe_validity);

    const rows = values.len;
    const distances = try allocator.alloc(f64, rows);
    errdefer allocator.free(distances);
    const abs_distances = try allocator.alloc(f64, rows);
    errdefer allocator.free(abs_distances);
    const above = try allocator.alloc(bool, rows);
    errdefer allocator.free(above);
    const below = try allocator.alloc(bool, rows);
    errdefer allocator.free(below);
    const at = try allocator.alloc(bool, rows);
    errdefer allocator.free(at);
    const validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(validity);

    for (values, 0..) |value, row| {
        const valid = rowValid(maybe_validity, row);
        validity[row] = valid;
        if (valid) {
            const distance = value - threshold;
            distances[row] = distance;
            abs_distances[row] = @abs(distance);
            above[row] = distance > 0;
            below[row] = distance < 0;
            at[row] = distance == 0;
        } else {
            distances[row] = 0;
            abs_distances[row] = 0;
            above[row] = false;
            below[row] = false;
            at[row] = false;
        }
    }

    return .{
        .allocator = allocator,
        .distances = distances,
        .abs_distances = abs_distances,
        .above = above,
        .below = below,
        .at = at,
        .validity = validity,
    };
}

pub const rollingThresholdProfile = summary_metrics_mod.rollingThresholdProfile;
pub const expandingThresholdProfile = summary_metrics_mod.expandingThresholdProfile;
