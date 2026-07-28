//! Boolean transition metric kernels and output-name helpers.

const std = @import("std");

/// Row-level boolean transition events and streaks.
pub const BoolTransitionProfile = struct {
    allocator: std.mem.Allocator,
    rising: []bool,
    falling: []bool,
    toggled: []bool,
    true_streak: []i64,
    false_streak: []i64,
    transition_validity: []bool,
    streak_validity: []bool,

    pub fn deinit(self: *BoolTransitionProfile) void {
        self.allocator.free(self.rising);
        self.allocator.free(self.falling);
        self.allocator.free(self.toggled);
        self.allocator.free(self.true_streak);
        self.allocator.free(self.false_streak);
        self.allocator.free(self.transition_validity);
        self.allocator.free(self.streak_validity);
        self.* = undefined;
    }
};

pub const BoolTransitionProfileColumnCount = 5;

pub fn boolTransitionProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![BoolTransitionProfileColumnCount][]const u8 {
    var names: [BoolTransitionProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "rising", "falling", "toggled", "true_streak", "false_streak" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub fn boolTransitionProfile(
    allocator: std.mem.Allocator,
    values: []const bool,
    maybe_validity: ?[]const bool,
    periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!BoolTransitionProfile {
    if (periods == 0) return error.InvalidShape;
    if (maybe_validity) |validity| {
        if (validity.len != values.len) return error.LengthMismatch;
    }

    const rows = values.len;
    const rising = try allocator.alloc(bool, rows);
    errdefer allocator.free(rising);
    const falling = try allocator.alloc(bool, rows);
    errdefer allocator.free(falling);
    const toggled = try allocator.alloc(bool, rows);
    errdefer allocator.free(toggled);
    const true_streak = try allocator.alloc(i64, rows);
    errdefer allocator.free(true_streak);
    const false_streak = try allocator.alloc(i64, rows);
    errdefer allocator.free(false_streak);
    const transition_validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(transition_validity);
    const streak_validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(streak_validity);

    var current_true_streak: i64 = 0;
    var current_false_streak: i64 = 0;
    for (values, 0..) |value, row| {
        const valid = if (maybe_validity) |mask| mask[row] else true;
        streak_validity[row] = valid;
        if (valid) {
            if (value) {
                current_true_streak += 1;
                current_false_streak = 0;
            } else {
                current_false_streak += 1;
                current_true_streak = 0;
            }
            true_streak[row] = current_true_streak;
            false_streak[row] = current_false_streak;
        } else {
            current_true_streak = 0;
            current_false_streak = 0;
            true_streak[row] = 0;
            false_streak[row] = 0;
        }

        if (row < periods) {
            rising[row] = false;
            falling[row] = false;
            toggled[row] = false;
            transition_validity[row] = false;
            continue;
        }
        const previous_row = row - periods;
        const previous_valid = if (maybe_validity) |mask| mask[previous_row] else true;
        const can_transition = valid and previous_valid;
        transition_validity[row] = can_transition;
        if (can_transition) {
            rising[row] = !values[previous_row] and value;
            falling[row] = values[previous_row] and !value;
            toggled[row] = values[previous_row] != value;
        } else {
            rising[row] = false;
            falling[row] = false;
            toggled[row] = false;
        }
    }

    return .{
        .allocator = allocator,
        .rising = rising,
        .falling = falling,
        .toggled = toggled,
        .true_streak = true_streak,
        .false_streak = false_streak,
        .transition_validity = transition_validity,
        .streak_validity = streak_validity,
    };
}

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
