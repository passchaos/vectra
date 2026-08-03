//! Boolean transition metric kernels and output-name helpers.

const std = @import("std");
const window_metrics_mod = @import("window_metrics.zig");

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

pub const RollingBoolTransitionMetrics = window_metrics_mod.RollingBoolTransitionMetrics;
pub const RollingBoolTransitionProfileColumnCount = window_metrics_mod.RollingBoolTransitionProfileColumnCount;
pub const rollingBoolTransitionProfileOutputNames = window_metrics_mod.rollingBoolTransitionProfileOutputNames;
pub const rollingBoolTransitionProfile = window_metrics_mod.rollingBoolTransitionProfile;
pub const ExpandingBoolTransitionMetrics = window_metrics_mod.ExpandingBoolTransitionMetrics;
pub const ExpandingBoolTransitionProfileColumnCount = window_metrics_mod.ExpandingBoolTransitionProfileColumnCount;
pub const expandingBoolTransitionProfileOutputNames = window_metrics_mod.expandingBoolTransitionProfileOutputNames;
pub const expandingBoolTransitionProfile = window_metrics_mod.expandingBoolTransitionProfile;
