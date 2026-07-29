//! Clip metric kernels and output-name helpers.

const std = @import("std");
const summary_metrics_mod = @import("dataframe_clip_summary_metrics.zig");

pub const ClipMetrics = struct {
    allocator: std.mem.Allocator,
    clipped: []f64,
    below: []bool,
    above: []bool,
    in_range: []bool,
    validity: []bool,

    pub fn deinit(self: *ClipMetrics) void {
        self.allocator.free(self.clipped);
        self.allocator.free(self.below);
        self.allocator.free(self.above);
        self.allocator.free(self.in_range);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub const ClipProfileColumnCount = 4;

pub fn clipProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![ClipProfileColumnCount][]const u8 {
    var names: [ClipProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "clipped", "below", "above", "in_range" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const ClipSummaryMetrics = summary_metrics_mod.ClipSummaryMetrics;
pub const RollingClipProfileColumnCount = summary_metrics_mod.RollingClipProfileColumnCount;
pub const rollingClipProfileOutputNames = summary_metrics_mod.rollingClipProfileOutputNames;
pub const ExpandingClipProfileColumnCount = summary_metrics_mod.ExpandingClipProfileColumnCount;
pub const expandingClipProfileOutputNames = summary_metrics_mod.expandingClipProfileOutputNames;
pub fn validate(values: []const f64, maybe_validity: ?[]const bool, lower: f64, upper: f64) error{ InvalidShape, LengthMismatch }!void {
    if (lower > upper) return error.InvalidShape;
    if (maybe_validity) |validity| {
        if (validity.len != values.len) return error.LengthMismatch;
    }
}

pub fn rowValid(maybe_validity: ?[]const bool, row: usize) bool {
    return if (maybe_validity) |mask| mask[row] else true;
}

pub fn clippedValue(value: f64, lower: f64, upper: f64) f64 {
    return @min(@max(value, lower), upper);
}

pub fn classify(value: f64, lower: f64, upper: f64) struct { clipped: f64, below: bool, above: bool, in_range: bool } {
    const below = value < lower;
    const above = value > upper;
    return .{
        .clipped = clippedValue(value, lower, upper),
        .below = below,
        .above = above,
        .in_range = !below and !above,
    };
}

pub fn clipProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    lower: f64,
    upper: f64,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!ClipMetrics {
    try validate(values, maybe_validity, lower, upper);

    const rows = values.len;
    const clipped = try allocator.alloc(f64, rows);
    errdefer allocator.free(clipped);
    const below = try allocator.alloc(bool, rows);
    errdefer allocator.free(below);
    const above = try allocator.alloc(bool, rows);
    errdefer allocator.free(above);
    const in_range = try allocator.alloc(bool, rows);
    errdefer allocator.free(in_range);
    const validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(validity);

    for (values, 0..) |value, row| {
        const valid = rowValid(maybe_validity, row);
        validity[row] = valid;
        if (valid) {
            const c = classify(value, lower, upper);
            clipped[row] = c.clipped;
            below[row] = c.below;
            above[row] = c.above;
            in_range[row] = c.in_range;
        } else {
            clipped[row] = 0;
            below[row] = false;
            above[row] = false;
            in_range[row] = false;
        }
    }

    return .{
        .allocator = allocator,
        .clipped = clipped,
        .below = below,
        .above = above,
        .in_range = in_range,
        .validity = validity,
    };
}

pub const rollingClipProfile = summary_metrics_mod.rollingClipProfile;
pub const expandingClipProfile = summary_metrics_mod.expandingClipProfile;
