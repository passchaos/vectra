const std = @import("std");

pub const BoolProfileMetrics = struct {
    allocator: std.mem.Allocator,
    true_counts: []i64,
    false_counts: []i64,
    true_rates: []f64,
    any_values: []bool,
    all_values: []bool,
    validity: []bool,

    pub fn deinit(self: *BoolProfileMetrics) void {
        self.allocator.free(self.true_counts);
        self.allocator.free(self.false_counts);
        self.allocator.free(self.true_rates);
        self.allocator.free(self.any_values);
        self.allocator.free(self.all_values);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

fn validateLength(values: []const bool, maybe_validity: ?[]const bool) error{LengthMismatch}!void {
    if (maybe_validity) |validity| {
        if (validity.len != values.len) return error.LengthMismatch;
    }
}

fn rowValid(maybe_validity: ?[]const bool, row: usize) bool {
    return if (maybe_validity) |mask| mask[row] else true;
}

fn allocMetrics(allocator: std.mem.Allocator, rows: usize) std.mem.Allocator.Error!BoolProfileMetrics {
    const true_counts = try allocator.alloc(i64, rows);
    errdefer allocator.free(true_counts);
    const false_counts = try allocator.alloc(i64, rows);
    errdefer allocator.free(false_counts);
    const true_rates = try allocator.alloc(f64, rows);
    errdefer allocator.free(true_rates);
    const any_values = try allocator.alloc(bool, rows);
    errdefer allocator.free(any_values);
    const all_values = try allocator.alloc(bool, rows);
    errdefer allocator.free(all_values);
    const validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(validity);
    return .{
        .allocator = allocator,
        .true_counts = true_counts,
        .false_counts = false_counts,
        .true_rates = true_rates,
        .any_values = any_values,
        .all_values = all_values,
        .validity = validity,
    };
}

fn writeRow(row: usize, min_periods: usize, current_valid: bool, true_count: usize, false_count: usize, out: BoolProfileMetrics) void {
    const valid_count = true_count + false_count;
    out.true_counts[row] = @intCast(true_count);
    out.false_counts[row] = @intCast(false_count);
    const has_enough = current_valid and valid_count >= min_periods;
    out.validity[row] = has_enough;
    if (has_enough) {
        out.true_rates[row] = @as(f64, @floatFromInt(true_count)) / @as(f64, @floatFromInt(valid_count));
        out.any_values[row] = true_count != 0;
        out.all_values[row] = false_count == 0;
    } else {
        out.true_rates[row] = 0;
        out.any_values[row] = false;
        out.all_values[row] = false;
    }
}

pub fn rollingBoolProfile(
    allocator: std.mem.Allocator,
    values: []const bool,
    maybe_validity: ?[]const bool,
    window: usize,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!BoolProfileMetrics {
    if (window == 0) return error.InvalidShape;
    if (min_periods == 0 or min_periods > window) return error.InvalidShape;
    try validateLength(values, maybe_validity);

    var out = try allocMetrics(allocator, values.len);
    errdefer out.deinit();

    var running_true: usize = 0;
    var running_false: usize = 0;
    for (values, 0..) |value, row| {
        if (rowValid(maybe_validity, row)) {
            if (value) {
                running_true += 1;
            } else {
                running_false += 1;
            }
        }

        if (row >= window) {
            const evict_row = row - window;
            if (rowValid(maybe_validity, evict_row)) {
                if (values[evict_row]) {
                    running_true -= 1;
                } else {
                    running_false -= 1;
                }
            }
        }

        // Counts are always materialized as window diagnostics. Predicate/rate
        // outputs are nullable because they describe the current row's trailing
        // context: a null current row or too few valid observations leaves the
        // derived state unknown while preserving audit counts.
        writeRow(row, min_periods, rowValid(maybe_validity, row), running_true, running_false, out);
    }

    return out;
}

pub fn expandingBoolProfile(
    allocator: std.mem.Allocator,
    values: []const bool,
    maybe_validity: ?[]const bool,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!BoolProfileMetrics {
    if (min_periods == 0) return error.InvalidShape;
    try validateLength(values, maybe_validity);

    var out = try allocMetrics(allocator, values.len);
    errdefer out.deinit();

    var running_true: usize = 0;
    var running_false: usize = 0;
    for (values, 0..) |value, row| {
        if (rowValid(maybe_validity, row)) {
            if (value) {
                running_true += 1;
            } else {
                running_false += 1;
            }
        }

        writeRow(row, min_periods, rowValid(maybe_validity, row), running_true, running_false, out);
    }

    return out;
}
