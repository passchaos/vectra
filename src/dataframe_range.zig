const std = @import("std");

pub const RangeMetrics = struct {
    allocator: std.mem.Allocator,
    lows: []f64,
    highs: []f64,
    ranges: []f64,
    positions: []f64,
    validity: []bool,

    pub fn deinit(self: *RangeMetrics) void {
        self.allocator.free(self.lows);
        self.allocator.free(self.highs);
        self.allocator.free(self.ranges);
        self.allocator.free(self.positions);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub const RollingRangeProfileColumnCount = 4;

pub fn rollingRangeProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![RollingRangeProfileColumnCount][]const u8 {
    var names: [RollingRangeProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "rolling_low", "rolling_high", "rolling_range", "rolling_position" };
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

pub fn rollingRangeProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    window: usize,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!RangeMetrics {
    if (window == 0) return error.InvalidShape;
    if (min_periods == 0 or min_periods > window) return error.InvalidShape;
    try validate(values, maybe_validity);

    const lows = try allocator.alloc(f64, values.len);
    errdefer allocator.free(lows);
    const highs = try allocator.alloc(f64, values.len);
    errdefer allocator.free(highs);
    const ranges = try allocator.alloc(f64, values.len);
    errdefer allocator.free(ranges);
    const positions = try allocator.alloc(f64, values.len);
    errdefer allocator.free(positions);
    const validity = try allocator.alloc(bool, values.len);
    errdefer allocator.free(validity);

    // Recompute each trailing window in host memory, preserving a single future
    // lowering seam for device rolling min/max kernels.
    for (values, 0..) |value, row| {
        const start = if (row + 1 > window) row + 1 - window else 0;
        var count: usize = 0;
        var low: f64 = 0;
        var high: f64 = 0;
        for (start..row + 1) |window_row| {
            if (!rowValid(maybe_validity, window_row)) continue;
            const x = values[window_row];
            if (count == 0) {
                low = x;
                high = x;
            } else {
                if (x < low) low = x;
                if (x > high) high = x;
            }
            count += 1;
        }

        const has_enough = rowValid(maybe_validity, row) and count >= min_periods;
        validity[row] = has_enough;
        if (has_enough) {
            const range = high - low;
            lows[row] = low;
            highs[row] = high;
            ranges[row] = range;
            positions[row] = if (range == 0) std.math.nan(f64) else (value - low) / range;
        } else {
            lows[row] = 0;
            highs[row] = 0;
            ranges[row] = 0;
            positions[row] = 0;
        }
    }

    return .{ .allocator = allocator, .lows = lows, .highs = highs, .ranges = ranges, .positions = positions, .validity = validity };
}
