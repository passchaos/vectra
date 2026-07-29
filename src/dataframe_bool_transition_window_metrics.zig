//! Rolling and expanding boolean transition metric kernels.

const std = @import("std");

/// Host-side rolling summaries for nullable boolean transition events.
///
/// `dataframe.zig` owns the public dataframe/device abstractions; this module is
/// intentionally slice-oriented so the large dataframe file only has to
/// materialize columns and can keep specialized window logic out-of-line.
pub const RollingBoolTransitionMetrics = struct {
    allocator: std.mem.Allocator,
    counts: []i64,
    rising_counts: []i64,
    falling_counts: []i64,
    toggle_counts: []i64,
    rising_rates: []f64,
    falling_rates: []f64,
    toggle_rates: []f64,
    metric_validity: []bool,

    pub fn deinit(self: *RollingBoolTransitionMetrics) void {
        self.allocator.free(self.counts);
        self.allocator.free(self.rising_counts);
        self.allocator.free(self.falling_counts);
        self.allocator.free(self.toggle_counts);
        self.allocator.free(self.rising_rates);
        self.allocator.free(self.falling_rates);
        self.allocator.free(self.toggle_rates);
        self.allocator.free(self.metric_validity);
        self.* = undefined;
    }
};

pub const RollingBoolTransitionProfileColumnCount = 7;

pub fn rollingBoolTransitionProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![RollingBoolTransitionProfileColumnCount][]const u8 {
    var names: [RollingBoolTransitionProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "rolling_transition_count", "rolling_rising_count", "rolling_falling_count", "rolling_toggle_count", "rolling_rising_rate", "rolling_falling_rate", "rolling_toggle_rate" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub fn rollingBoolTransitionProfile(
    allocator: std.mem.Allocator,
    values: []const bool,
    maybe_validity: ?[]const bool,
    periods: usize,
    window: usize,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!RollingBoolTransitionMetrics {
    if (periods == 0 or window == 0) return error.InvalidShape;
    if (min_periods == 0 or min_periods > window) return error.InvalidShape;
    if (maybe_validity) |validity| {
        if (validity.len != values.len) return error.LengthMismatch;
    }

    const rows = values.len;
    const event_validity = try allocator.alloc(bool, rows);
    defer allocator.free(event_validity);
    const rising_events = try allocator.alloc(bool, rows);
    defer allocator.free(rising_events);
    const falling_events = try allocator.alloc(bool, rows);
    defer allocator.free(falling_events);
    const toggle_events = try allocator.alloc(bool, rows);
    defer allocator.free(toggle_events);

    for (values, 0..) |value, row| {
        rising_events[row] = false;
        falling_events[row] = false;
        toggle_events[row] = false;
        if (row < periods) {
            event_validity[row] = false;
            continue;
        }

        const previous_row = row - periods;
        const current_valid = if (maybe_validity) |mask| mask[row] else true;
        const previous_valid = if (maybe_validity) |mask| mask[previous_row] else true;
        const valid = current_valid and previous_valid;
        event_validity[row] = valid;
        if (valid) {
            rising_events[row] = !values[previous_row] and value;
            falling_events[row] = values[previous_row] and !value;
            toggle_events[row] = values[previous_row] != value;
        }
    }

    const counts = try allocator.alloc(i64, rows);
    errdefer allocator.free(counts);
    const rising_counts = try allocator.alloc(i64, rows);
    errdefer allocator.free(rising_counts);
    const falling_counts = try allocator.alloc(i64, rows);
    errdefer allocator.free(falling_counts);
    const toggle_counts = try allocator.alloc(i64, rows);
    errdefer allocator.free(toggle_counts);
    const rising_rates = try allocator.alloc(f64, rows);
    errdefer allocator.free(rising_rates);
    const falling_rates = try allocator.alloc(f64, rows);
    errdefer allocator.free(falling_rates);
    const toggle_rates = try allocator.alloc(f64, rows);
    errdefer allocator.free(toggle_rates);
    const metric_validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(metric_validity);

    for (0..rows) |row| {
        const start = if (row + 1 > window) row + 1 - window else 0;
        var count: usize = 0;
        var rising_count: usize = 0;
        var falling_count: usize = 0;
        var toggle_count: usize = 0;
        for (start..row + 1) |window_row| {
            if (!event_validity[window_row]) continue;
            count += 1;
            if (rising_events[window_row]) rising_count += 1;
            if (falling_events[window_row]) falling_count += 1;
            if (toggle_events[window_row]) toggle_count += 1;
        }

        counts[row] = @intCast(count);
        rising_counts[row] = @intCast(rising_count);
        falling_counts[row] = @intCast(falling_count);
        toggle_counts[row] = @intCast(toggle_count);
        const has_enough = count >= min_periods;
        metric_validity[row] = has_enough;
        if (has_enough) {
            // The denominator is the number of valid transition observations,
            // not the number of valid source values. This keeps period-based
            // transitions from diluting early windows where there is not yet a
            // prior row to compare against.
            const n: f64 = @floatFromInt(count);
            rising_rates[row] = @as(f64, @floatFromInt(rising_count)) / n;
            falling_rates[row] = @as(f64, @floatFromInt(falling_count)) / n;
            toggle_rates[row] = @as(f64, @floatFromInt(toggle_count)) / n;
        } else {
            rising_rates[row] = 0;
            falling_rates[row] = 0;
            toggle_rates[row] = 0;
        }
    }

    return .{
        .allocator = allocator,
        .counts = counts,
        .rising_counts = rising_counts,
        .falling_counts = falling_counts,
        .toggle_counts = toggle_counts,
        .rising_rates = rising_rates,
        .falling_rates = falling_rates,
        .toggle_rates = toggle_rates,
        .metric_validity = metric_validity,
    };
}

pub const ExpandingBoolTransitionMetrics = RollingBoolTransitionMetrics;
pub const ExpandingBoolTransitionProfileColumnCount = 7;

pub fn expandingBoolTransitionProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![ExpandingBoolTransitionProfileColumnCount][]const u8 {
    var names: [ExpandingBoolTransitionProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "expanding_transition_count", "expanding_rising_count", "expanding_falling_count", "expanding_toggle_count", "expanding_rising_rate", "expanding_falling_rate", "expanding_toggle_rate" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub fn expandingBoolTransitionProfile(
    allocator: std.mem.Allocator,
    values: []const bool,
    maybe_validity: ?[]const bool,
    periods: usize,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!ExpandingBoolTransitionMetrics {
    if (periods == 0 or min_periods == 0) return error.InvalidShape;
    if (maybe_validity) |validity| {
        if (validity.len != values.len) return error.LengthMismatch;
    }

    const rows = values.len;
    const counts = try allocator.alloc(i64, rows);
    errdefer allocator.free(counts);
    const rising_counts = try allocator.alloc(i64, rows);
    errdefer allocator.free(rising_counts);
    const falling_counts = try allocator.alloc(i64, rows);
    errdefer allocator.free(falling_counts);
    const toggle_counts = try allocator.alloc(i64, rows);
    errdefer allocator.free(toggle_counts);
    const rising_rates = try allocator.alloc(f64, rows);
    errdefer allocator.free(rising_rates);
    const falling_rates = try allocator.alloc(f64, rows);
    errdefer allocator.free(falling_rates);
    const toggle_rates = try allocator.alloc(f64, rows);
    errdefer allocator.free(toggle_rates);
    const metric_validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(metric_validity);

    var count: usize = 0;
    var rising_count: usize = 0;
    var falling_count: usize = 0;
    var toggle_count: usize = 0;
    for (values, 0..) |value, row| {
        if (row >= periods) {
            const previous_row = row - periods;
            const current_valid = if (maybe_validity) |mask| mask[row] else true;
            const previous_valid = if (maybe_validity) |mask| mask[previous_row] else true;
            if (current_valid and previous_valid) {
                count += 1;
                const previous = values[previous_row];
                if (!previous and value) rising_count += 1;
                if (previous and !value) falling_count += 1;
                if (previous != value) toggle_count += 1;
            }
        }

        counts[row] = @intCast(count);
        rising_counts[row] = @intCast(rising_count);
        falling_counts[row] = @intCast(falling_count);
        toggle_counts[row] = @intCast(toggle_count);
        const has_enough = count >= min_periods;
        metric_validity[row] = has_enough;
        if (has_enough) {
            const n: f64 = @floatFromInt(count);
            rising_rates[row] = @as(f64, @floatFromInt(rising_count)) / n;
            falling_rates[row] = @as(f64, @floatFromInt(falling_count)) / n;
            toggle_rates[row] = @as(f64, @floatFromInt(toggle_count)) / n;
        } else {
            rising_rates[row] = 0;
            falling_rates[row] = 0;
            toggle_rates[row] = 0;
        }
    }

    return .{
        .allocator = allocator,
        .counts = counts,
        .rising_counts = rising_counts,
        .falling_counts = falling_counts,
        .toggle_counts = toggle_counts,
        .rising_rates = rising_rates,
        .falling_rates = falling_rates,
        .toggle_rates = toggle_rates,
        .metric_validity = metric_validity,
    };
}
