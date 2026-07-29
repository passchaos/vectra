//! Rolling drawdown metric kernels.

const std = @import("std");
const base_mod = @import("dataframe_risk_metrics.zig");

const validateLength = base_mod.validateLength;
const rowValid = base_mod.rowValid;

pub const RollingDrawdownMetrics = struct {
    allocator: std.mem.Allocator,
    peaks: []f64,
    drawdowns: []f64,
    drawdown_pcts: []f64,
    peak_ages: []i64,
    validity: []bool,

    pub fn deinit(self: *RollingDrawdownMetrics) void {
        self.allocator.free(self.peaks);
        self.allocator.free(self.drawdowns);
        self.allocator.free(self.drawdown_pcts);
        self.allocator.free(self.peak_ages);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub const RollingDrawdownProfileColumnCount = 4;

pub fn rollingDrawdownProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![RollingDrawdownProfileColumnCount][]const u8 {
    var names: [RollingDrawdownProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "rolling_peak", "rolling_drawdown", "rolling_drawdown_pct", "rolling_peak_age" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub fn rollingDrawdownProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    window: usize,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!RollingDrawdownMetrics {
    if (window == 0) return error.InvalidShape;
    if (min_periods == 0 or min_periods > window) return error.InvalidShape;
    try validateLength(values, maybe_validity);

    const rows = values.len;
    const peaks = try allocator.alloc(f64, rows);
    errdefer allocator.free(peaks);
    const drawdowns = try allocator.alloc(f64, rows);
    errdefer allocator.free(drawdowns);
    const drawdown_pcts = try allocator.alloc(f64, rows);
    errdefer allocator.free(drawdown_pcts);
    const peak_ages = try allocator.alloc(i64, rows);
    errdefer allocator.free(peak_ages);
    const validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(validity);

    // Equal highs choose the most recent row so peak_age answers "rows since the
    // latest peak that still defines this window".
    for (values, 0..) |current, row| {
        const start = if (row + 1 > window) row + 1 - window else 0;
        var count: usize = 0;
        var peak: f64 = 0;
        var peak_row: usize = start;
        for (start..row + 1) |window_row| {
            if (!rowValid(maybe_validity, window_row)) continue;
            const x = values[window_row];
            if (count == 0 or x >= peak) {
                peak = x;
                peak_row = window_row;
            }
            count += 1;
        }

        const has_enough = rowValid(maybe_validity, row) and count >= min_periods;
        validity[row] = has_enough;
        if (has_enough) {
            const dd = current - peak;
            peaks[row] = peak;
            drawdowns[row] = dd;
            drawdown_pcts[row] = if (peak == 0) std.math.nan(f64) else dd / peak;
            peak_ages[row] = @intCast(row - peak_row);
        } else {
            peaks[row] = 0;
            drawdowns[row] = 0;
            drawdown_pcts[row] = 0;
            peak_ages[row] = 0;
        }
    }

    return .{
        .allocator = allocator,
        .peaks = peaks,
        .drawdowns = drawdowns,
        .drawdown_pcts = drawdown_pcts,
        .peak_ages = peak_ages,
        .validity = validity,
    };
}
