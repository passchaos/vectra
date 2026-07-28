const std = @import("std");

pub const QuantileMetrics = struct {
    allocator: std.mem.Allocator,
    q1: []f64,
    medians: []f64,
    q3: []f64,
    iqrs: []f64,
    validity: []bool,

    pub fn deinit(self: *QuantileMetrics) void {
        self.allocator.free(self.q1);
        self.allocator.free(self.medians);
        self.allocator.free(self.q3);
        self.allocator.free(self.iqrs);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

fn validate(values: []const f64, maybe_validity: ?[]const bool) error{LengthMismatch}!void {
    if (maybe_validity) |validity| {
        if (validity.len != values.len) return error.LengthMismatch;
    }
}

fn rowValid(maybe_validity: ?[]const bool, row: usize) bool {
    return if (maybe_validity) |mask| mask[row] else true;
}

fn lessFloat(lhs: f64, rhs: f64) bool {
    const lhs_nan = std.math.isNan(lhs);
    const rhs_nan = std.math.isNan(rhs);
    if (lhs_nan != rhs_nan) return !lhs_nan;
    if (lhs_nan and rhs_nan) return false;
    return lhs < rhs;
}

fn sort(values: []f64) void {
    std.sort.insertion(f64, values, {}, struct {
        fn lessThan(_: void, lhs: f64, rhs: f64) bool {
            return lessFloat(lhs, rhs);
        }
    }.lessThan);
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

fn allocMetrics(allocator: std.mem.Allocator, rows: usize) std.mem.Allocator.Error!QuantileMetrics {
    const q1 = try allocator.alloc(f64, rows);
    errdefer allocator.free(q1);
    const medians = try allocator.alloc(f64, rows);
    errdefer allocator.free(medians);
    const q3 = try allocator.alloc(f64, rows);
    errdefer allocator.free(q3);
    const iqrs = try allocator.alloc(f64, rows);
    errdefer allocator.free(iqrs);
    const validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(validity);
    return .{ .allocator = allocator, .q1 = q1, .medians = medians, .q3 = q3, .iqrs = iqrs, .validity = validity };
}

fn writeQuantiles(row: usize, values: []f64, out: QuantileMetrics) void {
    sort(values);
    const q1 = quantileSorted(values, 0.25);
    const median = quantileSorted(values, 0.5);
    const q3 = quantileSorted(values, 0.75);
    out.q1[row] = q1;
    out.medians[row] = median;
    out.q3[row] = q3;
    out.iqrs[row] = q3 - q1;
    out.validity[row] = true;
}

fn writeInvalid(row: usize, out: QuantileMetrics) void {
    out.q1[row] = 0;
    out.medians[row] = 0;
    out.q3[row] = 0;
    out.iqrs[row] = 0;
    out.validity[row] = false;
}

pub fn rollingQuantileProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    window: usize,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!QuantileMetrics {
    if (window == 0) return error.InvalidShape;
    if (min_periods == 0 or min_periods > window) return error.InvalidShape;
    try validate(values, maybe_validity);

    var out = try allocMetrics(allocator, values.len);
    errdefer out.deinit();
    const scratch = try allocator.alloc(f64, window);
    defer allocator.free(scratch);

    for (0..values.len) |row| {
        const start = if (row + 1 > window) row + 1 - window else 0;
        var count: usize = 0;
        for (start..row + 1) |window_row| {
            if (!rowValid(maybe_validity, window_row)) continue;
            scratch[count] = values[window_row];
            count += 1;
        }

        if (!rowValid(maybe_validity, row) or count < min_periods) {
            writeInvalid(row, out);
            continue;
        }
        writeQuantiles(row, scratch[0..count], out);
    }

    return out;
}

pub fn expandingQuantileProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!QuantileMetrics {
    if (min_periods == 0) return error.InvalidShape;
    try validate(values, maybe_validity);

    var out = try allocMetrics(allocator, values.len);
    errdefer out.deinit();
    const scratch = try allocator.alloc(f64, values.len);
    defer allocator.free(scratch);

    var valid_count: usize = 0;
    for (values, 0..) |value, row| {
        if (rowValid(maybe_validity, row)) {
            scratch[valid_count] = value;
            valid_count += 1;
        }

        if (valid_count < min_periods) {
            writeInvalid(row, out);
            continue;
        }
        writeQuantiles(row, scratch[0..valid_count], out);
    }

    return out;
}
