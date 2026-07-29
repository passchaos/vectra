//! Change-point metric kernels and output-name helpers.

const std = @import("std");
const summary_metrics_mod = @import("dataframe_change_summary_metrics.zig");

pub const ChangePointMetrics = struct {
    allocator: std.mem.Allocator,
    deltas: []f64,
    abs_deltas: []f64,
    pct_changes: []f64,
    change_points: []bool,
    validity: []bool,

    pub fn deinit(self: *ChangePointMetrics) void {
        self.allocator.free(self.deltas);
        self.allocator.free(self.abs_deltas);
        self.allocator.free(self.pct_changes);
        self.allocator.free(self.change_points);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub const ChangePointProfileColumnCount = 4;

pub fn changePointProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![ChangePointProfileColumnCount][]const u8 {
    var names: [ChangePointProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "change_delta", "change_abs_delta", "change_pct", "change_point" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const ChangeSummaryMetrics = summary_metrics_mod.ChangeSummaryMetrics;
pub const RollingChangePointProfileColumnCount = summary_metrics_mod.RollingChangePointProfileColumnCount;
pub const rollingChangePointProfileOutputNames = summary_metrics_mod.rollingChangePointProfileOutputNames;
pub const ExpandingChangePointProfileColumnCount = summary_metrics_mod.ExpandingChangePointProfileColumnCount;
pub const expandingChangePointProfileOutputNames = summary_metrics_mod.expandingChangePointProfileOutputNames;
pub fn validate(values: []const f64, maybe_validity: ?[]const bool, threshold: f64, periods: usize) error{ InvalidShape, LengthMismatch }!void {
    if (periods == 0) return error.InvalidShape;
    if (threshold < 0) return error.InvalidShape;
    if (maybe_validity) |validity| {
        if (validity.len != values.len) return error.LengthMismatch;
    }
}

pub fn rowValid(maybe_validity: ?[]const bool, row: usize) bool {
    return if (maybe_validity) |mask| mask[row] else true;
}

pub fn validPair(maybe_validity: ?[]const bool, row: usize, previous_row: usize) bool {
    return rowValid(maybe_validity, row) and rowValid(maybe_validity, previous_row);
}

pub fn absDelta(values: []const f64, row: usize, previous_row: usize) f64 {
    return @abs(values[row] - values[previous_row]);
}

pub fn changePointProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    threshold: f64,
    periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!ChangePointMetrics {
    try validate(values, maybe_validity, threshold, periods);

    const rows = values.len;
    const deltas = try allocator.alloc(f64, rows);
    errdefer allocator.free(deltas);
    const abs_deltas = try allocator.alloc(f64, rows);
    errdefer allocator.free(abs_deltas);
    const pct_changes = try allocator.alloc(f64, rows);
    errdefer allocator.free(pct_changes);
    const change_points = try allocator.alloc(bool, rows);
    errdefer allocator.free(change_points);
    const validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(validity);

    for (values, 0..) |value, row| {
        if (row < periods) {
            deltas[row] = 0;
            abs_deltas[row] = 0;
            pct_changes[row] = 0;
            change_points[row] = false;
            validity[row] = false;
            continue;
        }

        const previous_row = row - periods;
        const valid = validPair(maybe_validity, row, previous_row);
        validity[row] = valid;
        if (!valid) {
            deltas[row] = 0;
            abs_deltas[row] = 0;
            pct_changes[row] = 0;
            change_points[row] = false;
            continue;
        }

        const previous = values[previous_row];
        const delta = value - previous;
        const abs_delta = @abs(delta);
        deltas[row] = delta;
        abs_deltas[row] = abs_delta;
        pct_changes[row] = if (previous == 0) std.math.nan(f64) else delta / previous;
        change_points[row] = abs_delta >= threshold;
    }

    return .{
        .allocator = allocator,
        .deltas = deltas,
        .abs_deltas = abs_deltas,
        .pct_changes = pct_changes,
        .change_points = change_points,
        .validity = validity,
    };
}

pub const rollingChangePointProfile = summary_metrics_mod.rollingChangePointProfile;
pub const expandingChangePointProfile = summary_metrics_mod.expandingChangePointProfile;
