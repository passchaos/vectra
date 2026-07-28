//! Rank metric kernels and rank profile output-name helpers.

const std = @import("std");

pub const RankMetrics = struct {
    allocator: std.mem.Allocator,
    ordinal: []i64,
    competition: []i64,
    dense: []i64,
    percent: []f64,
    cume: []f64,

    pub fn deinit(self: *RankMetrics) void {
        self.allocator.free(self.ordinal);
        self.allocator.free(self.competition);
        self.allocator.free(self.dense);
        self.allocator.free(self.percent);
        self.allocator.free(self.cume);
        self.* = undefined;
    }
};

pub const RankProfileColumnCount = 5;

pub fn rankProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![RankProfileColumnCount][]const u8 {
    var names: [RankProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{
        "ordinal_rank",
        "competition_rank",
        "dense_rank",
        "percent_rank",
        "cume_dist",
    };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub fn rankProfile(
    allocator: std.mem.Allocator,
    rows: usize,
    order: []const usize,
    keysTie: *const fn (usize, usize) bool,
) std.mem.Allocator.Error!RankMetrics {
    const ordinal = try allocator.alloc(i64, rows);
    errdefer allocator.free(ordinal);
    const competition = try allocator.alloc(i64, rows);
    errdefer allocator.free(competition);
    const dense = try allocator.alloc(i64, rows);
    errdefer allocator.free(dense);
    const percent = try allocator.alloc(f64, rows);
    errdefer allocator.free(percent);
    const cume = try allocator.alloc(f64, rows);
    errdefer allocator.free(cume);

    var group_start: usize = 0;
    var dense_rank: i64 = 0;
    while (group_start < rows) {
        var group_end = group_start + 1;
        while (group_end < rows and keysTie(order[group_start], order[group_end])) {
            group_end += 1;
        }

        dense_rank += 1;
        const competition_rank: i64 = @intCast(group_start + 1);
        const percent_rank: f64 = if (rows <= 1) 0 else @as(f64, @floatFromInt(group_start)) / @as(f64, @floatFromInt(rows - 1));
        const cume_dist: f64 = if (rows == 0) std.math.nan(f64) else @as(f64, @floatFromInt(group_end)) / @as(f64, @floatFromInt(rows));

        for (order[group_start..group_end], group_start..) |row, sorted_position| {
            ordinal[row] = @intCast(sorted_position + 1);
            competition[row] = competition_rank;
            dense[row] = dense_rank;
            percent[row] = percent_rank;
            cume[row] = cume_dist;
        }
        group_start = group_end;
    }

    return .{
        .allocator = allocator,
        .ordinal = ordinal,
        .competition = competition,
        .dense = dense,
        .percent = percent,
        .cume = cume,
    };
}

pub const RankWindowMetrics = struct {
    allocator: std.mem.Allocator,
    counts: []i64,
    ranks: []i64,
    percent_ranks: []f64,
    cume_dist: []f64,
    validity: []bool,

    pub fn deinit(self: *RankWindowMetrics) void {
        self.allocator.free(self.counts);
        self.allocator.free(self.ranks);
        self.allocator.free(self.percent_ranks);
        self.allocator.free(self.cume_dist);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub const RollingRankProfileColumnCount = 4;

pub fn rollingRankProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![RollingRankProfileColumnCount][]const u8 {
    var names: [RollingRankProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "rolling_rank_count", "rolling_rank", "rolling_percent_rank", "rolling_cume_dist" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const ExpandingRankProfileColumnCount = 4;

pub fn expandingRankProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![ExpandingRankProfileColumnCount][]const u8 {
    var names: [ExpandingRankProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "expanding_rank_count", "expanding_rank", "expanding_percent_rank", "expanding_cume_dist" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn validate(rows: usize, maybe_validity: ?[]const bool) error{LengthMismatch}!void {
    if (maybe_validity) |validity| {
        if (validity.len != rows) return error.LengthMismatch;
    }
}

fn rowValid(maybe_validity: ?[]const bool, row: usize) bool {
    return if (maybe_validity) |mask| mask[row] else true;
}

fn allocMetrics(allocator: std.mem.Allocator, rows: usize) std.mem.Allocator.Error!RankWindowMetrics {
    const counts = try allocator.alloc(i64, rows);
    errdefer allocator.free(counts);
    const ranks = try allocator.alloc(i64, rows);
    errdefer allocator.free(ranks);
    const percent_ranks = try allocator.alloc(f64, rows);
    errdefer allocator.free(percent_ranks);
    const cume_dist = try allocator.alloc(f64, rows);
    errdefer allocator.free(cume_dist);
    const validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(validity);
    return .{ .allocator = allocator, .counts = counts, .ranks = ranks, .percent_ranks = percent_ranks, .cume_dist = cume_dist, .validity = validity };
}

fn writeRow(row: usize, count: usize, before_count: usize, equal_count: usize, min_periods: usize, current_valid: bool, out: RankWindowMetrics) void {
    out.counts[row] = @intCast(count);
    const has_enough = current_valid and count >= min_periods;
    out.validity[row] = has_enough;
    if (has_enough) {
        out.ranks[row] = @intCast(before_count + 1);
        out.percent_ranks[row] = if (count <= 1) 0 else @as(f64, @floatFromInt(before_count)) / @as(f64, @floatFromInt(count - 1));
        out.cume_dist[row] = @as(f64, @floatFromInt(before_count + equal_count)) / @as(f64, @floatFromInt(count));
    } else {
        out.ranks[row] = 0;
        out.percent_ranks[row] = 0;
        out.cume_dist[row] = 0;
    }
}

pub fn rollingRankProfile(
    allocator: std.mem.Allocator,
    rows: usize,
    maybe_validity: ?[]const bool,
    window: usize,
    min_periods: usize,
    descending: bool,
    compare: *const fn (usize, usize) i8,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!RankWindowMetrics {
    if (window == 0) return error.InvalidShape;
    if (min_periods == 0 or min_periods > window) return error.InvalidShape;
    try validate(rows, maybe_validity);

    var out = try allocMetrics(allocator, rows);
    errdefer out.deinit();

    for (0..rows) |row| {
        const start = if (row + 1 > window) row + 1 - window else 0;
        var count: usize = 0;
        var before_count: usize = 0;
        var equal_count: usize = 0;
        const current_valid = rowValid(maybe_validity, row);
        if (current_valid) {
            for (start..row + 1) |window_row| {
                if (!rowValid(maybe_validity, window_row)) continue;
                count += 1;
                const cmp = compare(window_row, row);
                const before = if (descending) cmp > 0 else cmp < 0;
                if (before) before_count += 1;
                if (cmp == 0) equal_count += 1;
            }
        } else {
            for (start..row + 1) |window_row| {
                if (rowValid(maybe_validity, window_row)) count += 1;
            }
        }
        writeRow(row, count, before_count, equal_count, min_periods, current_valid, out);
    }

    return out;
}

pub fn expandingRankProfile(
    allocator: std.mem.Allocator,
    rows: usize,
    maybe_validity: ?[]const bool,
    min_periods: usize,
    descending: bool,
    compare: *const fn (usize, usize) i8,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!RankWindowMetrics {
    if (min_periods == 0) return error.InvalidShape;
    try validate(rows, maybe_validity);

    var out = try allocMetrics(allocator, rows);
    errdefer out.deinit();

    var valid_count: usize = 0;
    for (0..rows) |row| {
        const current_valid = rowValid(maybe_validity, row);
        if (current_valid) valid_count += 1;

        var before_count: usize = 0;
        var equal_count: usize = 0;
        if (current_valid) {
            for (0..row + 1) |prefix_row| {
                if (!rowValid(maybe_validity, prefix_row)) continue;
                const cmp = compare(prefix_row, row);
                const before = if (descending) cmp > 0 else cmp < 0;
                if (before) before_count += 1;
                if (cmp == 0) equal_count += 1;
            }
        }

        writeRow(row, valid_count, before_count, equal_count, min_periods, current_valid, out);
    }

    return out;
}
