const std = @import("std");

pub const ShiftMetrics = struct {
    allocator: std.mem.Allocator,
    shifted: []f64,
    diff: []f64,
    pct_change: []f64,
    shift_validity: []bool,
    change_validity: []bool,

    pub fn deinit(self: *ShiftMetrics) void {
        self.allocator.free(self.shifted);
        self.allocator.free(self.diff);
        self.allocator.free(self.pct_change);
        self.allocator.free(self.shift_validity);
        self.allocator.free(self.change_validity);
        self.* = undefined;
    }
};

fn validate(values: []const f64, maybe_validity: ?[]const bool, periods: usize) error{ InvalidShape, LengthMismatch }!void {
    if (periods == 0) return error.InvalidShape;
    if (maybe_validity) |validity| {
        if (validity.len != values.len) return error.LengthMismatch;
    }
}

fn rowValid(maybe_validity: ?[]const bool, row: usize) bool {
    return if (maybe_validity) |mask| mask[row] else true;
}

fn allocMetrics(allocator: std.mem.Allocator, rows: usize) std.mem.Allocator.Error!ShiftMetrics {
    const shifted = try allocator.alloc(f64, rows);
    errdefer allocator.free(shifted);
    const diff = try allocator.alloc(f64, rows);
    errdefer allocator.free(diff);
    const pct_change = try allocator.alloc(f64, rows);
    errdefer allocator.free(pct_change);
    const shift_validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(shift_validity);
    const change_validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(change_validity);
    return .{
        .allocator = allocator,
        .shifted = shifted,
        .diff = diff,
        .pct_change = pct_change,
        .shift_validity = shift_validity,
        .change_validity = change_validity,
    };
}

pub fn lagProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!ShiftMetrics {
    try validate(values, maybe_validity, periods);
    var out = try allocMetrics(allocator, values.len);
    errdefer out.deinit();

    for (values, 0..) |value, row| {
        if (row < periods) {
            out.shifted[row] = 0;
            out.diff[row] = 0;
            out.pct_change[row] = 0;
            out.shift_validity[row] = false;
            out.change_validity[row] = false;
            continue;
        }

        const lag_row = row - periods;
        const lag_valid = rowValid(maybe_validity, lag_row);
        const previous = values[lag_row];
        out.shifted[row] = previous;
        out.shift_validity[row] = lag_valid;

        const can_change = rowValid(maybe_validity, row) and lag_valid;
        out.change_validity[row] = can_change;
        if (can_change) {
            const diff = value - previous;
            out.diff[row] = diff;
            out.pct_change[row] = if (previous == 0) std.math.nan(f64) else diff / previous;
        } else {
            out.diff[row] = 0;
            out.pct_change[row] = 0;
        }
    }

    return out;
}

pub fn leadProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!ShiftMetrics {
    try validate(values, maybe_validity, periods);
    var out = try allocMetrics(allocator, values.len);
    errdefer out.deinit();

    for (values, 0..) |value, row| {
        const lead_row = row + periods;
        if (lead_row >= values.len) {
            out.shifted[row] = 0;
            out.diff[row] = 0;
            out.pct_change[row] = 0;
            out.shift_validity[row] = false;
            out.change_validity[row] = false;
            continue;
        }

        const lead_valid = rowValid(maybe_validity, lead_row);
        const future = values[lead_row];
        out.shifted[row] = future;
        out.shift_validity[row] = lead_valid;

        const can_change = rowValid(maybe_validity, row) and lead_valid;
        out.change_validity[row] = can_change;
        if (can_change) {
            const diff = future - value;
            out.diff[row] = diff;
            out.pct_change[row] = if (value == 0) std.math.nan(f64) else diff / value;
        } else {
            out.diff[row] = 0;
            out.pct_change[row] = 0;
        }
    }

    return out;
}
