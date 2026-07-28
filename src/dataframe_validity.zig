const std = @import("std");
const array_mod = @import("array.zig");

pub const ValidityMetrics = struct {
    allocator: std.mem.Allocator,
    is_null: []bool,
    is_valid: []bool,
    valid_streak: []i64,
    null_streak: []i64,

    pub fn deinit(self: *ValidityMetrics) void {
        self.allocator.free(self.is_null);
        self.allocator.free(self.is_valid);
        self.allocator.free(self.valid_streak);
        self.allocator.free(self.null_streak);
        self.* = undefined;
    }
};

pub const ValiditySummaryMetrics = struct {
    allocator: std.mem.Allocator,
    total_counts: []i64,
    valid_counts: []i64,
    null_counts: []i64,
    valid_rates: []f64,
    null_rates: []f64,
    validity: []bool,

    pub fn deinit(self: *ValiditySummaryMetrics) void {
        self.allocator.free(self.total_counts);
        self.allocator.free(self.valid_counts);
        self.allocator.free(self.null_counts);
        self.allocator.free(self.valid_rates);
        self.allocator.free(self.null_rates);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub fn countNulls(validity_values: []const bool) usize {
    var nulls: usize = 0;
    for (validity_values) |valid| {
        if (!valid) nulls += 1;
    }
    return nulls;
}

pub fn countNullsInArray(mask: array_mod.Array(bool)) array_mod.ArrayError!usize {
    const values = try mask.toOwnedSlice(mask.allocator);
    defer mask.allocator.free(values);
    return countNulls(values);
}

pub const ValidityProfileColumnCount = 4;

pub fn validityProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![ValidityProfileColumnCount][]const u8 {
    var names: [ValidityProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "is_null", "is_valid", "valid_streak", "null_streak" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const RollingValidityProfileColumnCount = 5;

pub fn rollingValidityProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![RollingValidityProfileColumnCount][]const u8 {
    var names: [RollingValidityProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "rolling_validity_count", "rolling_valid_count", "rolling_null_count", "rolling_valid_rate", "rolling_null_rate" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const ExpandingValidityProfileColumnCount = 5;

pub fn expandingValidityProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![ExpandingValidityProfileColumnCount][]const u8 {
    var names: [ExpandingValidityProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "expanding_validity_count", "expanding_valid_count", "expanding_null_count", "expanding_valid_rate", "expanding_null_rate" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn rowValid(maybe_validity: ?[]const bool, row: usize) bool {
    return if (maybe_validity) |validity| validity[row] else true;
}

fn validateLength(rows: usize, maybe_validity: ?[]const bool) error{LengthMismatch}!void {
    if (maybe_validity) |validity| {
        if (validity.len != rows) return error.LengthMismatch;
    }
}

fn allocSummary(allocator: std.mem.Allocator, rows: usize) std.mem.Allocator.Error!ValiditySummaryMetrics {
    const total_counts = try allocator.alloc(i64, rows);
    errdefer allocator.free(total_counts);
    const valid_counts = try allocator.alloc(i64, rows);
    errdefer allocator.free(valid_counts);
    const null_counts = try allocator.alloc(i64, rows);
    errdefer allocator.free(null_counts);
    const valid_rates = try allocator.alloc(f64, rows);
    errdefer allocator.free(valid_rates);
    const null_rates = try allocator.alloc(f64, rows);
    errdefer allocator.free(null_rates);
    const validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(validity);
    return .{
        .allocator = allocator,
        .total_counts = total_counts,
        .valid_counts = valid_counts,
        .null_counts = null_counts,
        .valid_rates = valid_rates,
        .null_rates = null_rates,
        .validity = validity,
    };
}

fn writeSummary(row: usize, min_periods: usize, valid_count: usize, null_count: usize, out: ValiditySummaryMetrics) void {
    const total_count = valid_count + null_count;
    out.total_counts[row] = @intCast(total_count);
    out.valid_counts[row] = @intCast(valid_count);
    out.null_counts[row] = @intCast(null_count);
    const has_enough = total_count >= min_periods;
    out.validity[row] = has_enough;
    if (has_enough) {
        const n: f64 = @floatFromInt(total_count);
        out.valid_rates[row] = @as(f64, @floatFromInt(valid_count)) / n;
        out.null_rates[row] = @as(f64, @floatFromInt(null_count)) / n;
    } else {
        out.valid_rates[row] = 0;
        out.null_rates[row] = 0;
    }
}

pub fn validityProfile(allocator: std.mem.Allocator, rows: usize, maybe_validity: ?[]const bool) (std.mem.Allocator.Error || error{LengthMismatch})!ValidityMetrics {
    try validateLength(rows, maybe_validity);

    const is_null = try allocator.alloc(bool, rows);
    errdefer allocator.free(is_null);
    const is_valid = try allocator.alloc(bool, rows);
    errdefer allocator.free(is_valid);
    const valid_streak = try allocator.alloc(i64, rows);
    errdefer allocator.free(valid_streak);
    const null_streak = try allocator.alloc(i64, rows);
    errdefer allocator.free(null_streak);

    var current_valid_streak: i64 = 0;
    var current_null_streak: i64 = 0;
    for (0..rows) |row| {
        const valid = rowValid(maybe_validity, row);
        is_valid[row] = valid;
        is_null[row] = !valid;
        if (valid) {
            current_valid_streak += 1;
            current_null_streak = 0;
        } else {
            current_null_streak += 1;
            current_valid_streak = 0;
        }
        valid_streak[row] = current_valid_streak;
        null_streak[row] = current_null_streak;
    }

    return .{
        .allocator = allocator,
        .is_null = is_null,
        .is_valid = is_valid,
        .valid_streak = valid_streak,
        .null_streak = null_streak,
    };
}

pub fn rollingValidityProfile(
    allocator: std.mem.Allocator,
    rows: usize,
    maybe_validity: ?[]const bool,
    window: usize,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!ValiditySummaryMetrics {
    if (window == 0) return error.InvalidShape;
    if (min_periods == 0 or min_periods > window) return error.InvalidShape;
    try validateLength(rows, maybe_validity);

    var out = try allocSummary(allocator, rows);
    errdefer out.deinit();

    var running_valid: usize = 0;
    var running_null: usize = 0;
    for (0..rows) |row| {
        if (rowValid(maybe_validity, row)) {
            running_valid += 1;
        } else {
            running_null += 1;
        }

        if (row >= window) {
            const evict_row = row - window;
            if (rowValid(maybe_validity, evict_row)) {
                running_valid -= 1;
            } else {
                running_null -= 1;
            }
        }

        writeSummary(row, min_periods, running_valid, running_null, out);
    }

    return out;
}

pub fn expandingValidityProfile(
    allocator: std.mem.Allocator,
    rows: usize,
    maybe_validity: ?[]const bool,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!ValiditySummaryMetrics {
    if (min_periods == 0) return error.InvalidShape;
    try validateLength(rows, maybe_validity);

    var out = try allocSummary(allocator, rows);
    errdefer out.deinit();

    var running_valid: usize = 0;
    var running_null: usize = 0;
    for (0..rows) |row| {
        if (rowValid(maybe_validity, row)) {
            running_valid += 1;
        } else {
            running_null += 1;
        }

        writeSummary(row, min_periods, running_valid, running_null, out);
    }

    return out;
}
