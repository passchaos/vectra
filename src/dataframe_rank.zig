const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const names_mod = @import("dataframe_names.zig");
const dataframe_device_column_mod = @import("dataframe_device_column.zig");
const numeric_mod = @import("dataframe_numeric.zig");
const options_mod = @import("dataframe_options.zig");
const validity_mod = @import("dataframe_validity.zig");

const DeviceColumn = dataframe_device_column_mod.DeviceColumn;
const DeviceTypedColumn = dataframe_device_column_mod.DeviceTypedColumn;
const DeviceSortOptions = options_mod.DeviceSortOptions;
const DeviceRollingRankOptions = options_mod.DeviceRollingRankOptions;
const DeviceExpandingRankOptions = options_mod.DeviceExpandingRankOptions;
const argsortTypedColumn = dataframe_device_column_mod.argsortTypedColumn;
const compareSortValues = numeric_mod.compareSortValues;
const validityValues = validity_mod.validityValues;
const freeOwnedNameItems = names_mod.freeOwnedNameItems;

const RankFrameError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    InvalidCsv,
    EmptyDataFrame,
    UnsupportedType,
    InvalidDevice,
};

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

pub fn rankProfileColumnsByKey(
    allocator: std.mem.Allocator,
    key: DeviceColumn,
    options_value: DeviceSortOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![RankProfileColumnCount]DeviceColumn {
    if (key.len() != rows) return error.LengthMismatch;
    return switch (key) {
        .bool => |typed| rankProfileColumnsTyped(bool, allocator, typed, options_value, device_value),
        .i8 => |typed| rankProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| rankProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| rankProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| rankProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| rankProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| rankProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| rankProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| rankProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| rankProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| rankProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| rankProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| rankProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| rankProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn rankProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceSortOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![RankProfileColumnCount]DeviceColumn {
    const rows = column.len();
    const values = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);
    const order = try argsortTypedColumn(T, column, allocator, options_value);
    defer allocator.free(order);

    const TieCtx = struct {
        values: []const T,
        validity: ?[]const bool,
        fn keysTie(ctx: @This(), lhs: usize, rhs: usize) bool {
            const lhs_valid = if (ctx.validity) |validity| validity[lhs] else true;
            const rhs_valid = if (ctx.validity) |validity| validity[rhs] else true;
            if (lhs_valid != rhs_valid) return false;
            if (!lhs_valid) return true;
            return compareSortValues(T, ctx.values[lhs], ctx.values[rhs]) == 0;
        }
    };
    const tie_ctx = TieCtx{ .values = values, .validity = maybe_validity };
    const tie = struct {
        var context: TieCtx = undefined;
        fn call(lhs: usize, rhs: usize) bool {
            return context.keysTie(lhs, rhs);
        }
    };
    tie.context = tie_ctx;

    var metrics = try rankProfile(allocator, rows, order, tie.call);
    defer metrics.deinit();

    var columns: [RankProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.ordinal, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSlice(i64, allocator, metrics.competition, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSlice(i64, allocator, metrics.dense, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSlice(f64, allocator, metrics.percent, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSlice(f64, allocator, metrics.cume, device_value);
    initialized += 1;
    return columns;
}
pub fn rollingRankProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceRollingRankOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![RollingRankProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .bool => |typed| rollingRankProfileColumnsTyped(bool, allocator, typed, options_value, device_value),
        .i8 => |typed| rollingRankProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| rollingRankProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| rollingRankProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| rollingRankProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| rollingRankProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| rollingRankProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| rollingRankProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| rollingRankProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| rollingRankProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| rollingRankProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| rollingRankProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| rollingRankProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| rollingRankProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn rollingRankProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceRollingRankOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![RollingRankProfileColumnCount]DeviceColumn {
    const min_periods = options_value.min_periods orelse options_value.window;
    const values = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);
    const rows = values.len;

    const CmpCtx = struct {
        values: []const T,
        fn compare(ctx: @This(), lhs: usize, rhs: usize) i8 {
            return compareSortValues(T, ctx.values[lhs], ctx.values[rhs]);
        }
    };
    const cmp_ctx = CmpCtx{ .values = values };
    const cmp = struct {
        var context: CmpCtx = undefined;
        fn call(lhs: usize, rhs: usize) i8 {
            return context.compare(lhs, rhs);
        }
    };
    cmp.context = cmp_ctx;
    var metrics = try rollingRankProfile(allocator, rows, maybe_validity, options_value.window, min_periods, options_value.descending, cmp.call);
    defer metrics.deinit();

    var columns: [RollingRankProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(i64, allocator, metrics.ranks, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.percent_ranks, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.cume_dist, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn expandingRankProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceExpandingRankOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingRankProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .bool => |typed| expandingRankProfileColumnsTyped(bool, allocator, typed, options_value, device_value),
        .i8 => |typed| expandingRankProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| expandingRankProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| expandingRankProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| expandingRankProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| expandingRankProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| expandingRankProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| expandingRankProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| expandingRankProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| expandingRankProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| expandingRankProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| expandingRankProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| expandingRankProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| expandingRankProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn expandingRankProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceExpandingRankOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingRankProfileColumnCount]DeviceColumn {
    const values = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);
    const rows = values.len;

    const CmpCtx = struct {
        values: []const T,
        fn compare(ctx: @This(), lhs: usize, rhs: usize) i8 {
            return compareSortValues(T, ctx.values[lhs], ctx.values[rhs]);
        }
    };
    const cmp_ctx = CmpCtx{ .values = values };
    const cmp = struct {
        var context: CmpCtx = undefined;
        fn call(lhs: usize, rhs: usize) i8 {
            return context.compare(lhs, rhs);
        }
    };
    cmp.context = cmp_ctx;
    var metrics = try expandingRankProfile(allocator, rows, maybe_validity, options_value.min_periods, options_value.descending, cmp.call);
    defer metrics.deinit();

    var columns: [ExpandingRankProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(i64, allocator, metrics.ranks, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.percent_ranks, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.cume_dist, metrics.validity, device_value);
    initialized += 1;
    return columns;
}

pub fn argsortBy(frame: anytype, name: []const u8, options_value: DeviceSortOptions) RankFrameError![]usize {
    const sort_key = try frame.column(name);
    return sort_key.argsort(frame.allocator, options_value);
}

pub fn sortBy(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    options_value: DeviceSortOptions,
) RankFrameError!DeviceDataFrame {
    const order = try argsortBy(frame, name, options_value);
    defer frame.allocator.free(order);
    return frame.take(order);
}

pub fn topKBy(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    k: usize,
    options_value: DeviceSortOptions,
) RankFrameError!DeviceDataFrame {
    var sorted = try sortBy(DeviceDataFrame, frame, name, options_value);
    defer sorted.deinit();
    return sorted.head(k);
}

pub fn rankProfileBy(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceSortOptions,
) RankFrameError!DeviceDataFrame {
    const rank_key = try frame.column(name);
    var rank_columns = try rankProfileColumnsByKey(frame.allocator, rank_key.*, options_value, frame.device, frame.rows);
    var rank_columns_transferred: usize = 0;
    errdefer {
        for (rank_columns[rank_columns_transferred..]) |*col| col.deinit();
    }

    const source_names = try frame.allocator.alloc([]const u8, frame.columns.len + rank_columns.len);
    defer frame.allocator.free(source_names);
    for (frame.names, 0..) |source_name, i| source_names[i] = source_name;

    var rank_names = try rankProfileOutputNames(frame.allocator, output_prefix);
    defer freeOwnedNameItems(frame.allocator, rank_names[0..]);
    for (rank_names, 0..) |rank_name, i| source_names[frame.columns.len + i] = rank_name;

    const DeviceColumnType = std.meta.Elem(@TypeOf(frame.columns));
    var columns = try frame.allocator.alloc(DeviceColumnType, frame.columns.len + rank_columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }
    for (frame.columns, 0..) |col, i| {
        columns[i] = try col.clone();
        initialized += 1;
    }
    for (&rank_columns) |*rank_col| {
        columns[initialized] = rank_col.*;
        initialized += 1;
        rank_columns_transferred += 1;
    }

    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, source_names, columns, frame.rows, frame.device);
}

fn appendRankColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    source_names: []const []const u8,
    rank_columns: anytype,
) RankFrameError!DeviceDataFrame {
    const DeviceColumnType = std.meta.Elem(@TypeOf(frame.columns));
    var columns = try frame.allocator.alloc(DeviceColumnType, frame.columns.len + rank_columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }
    for (frame.columns, 0..) |col, i| {
        columns[i] = try col.clone();
        initialized += 1;
    }
    for (&rank_columns) |*rank_col| {
        columns[initialized] = rank_col.*;
        initialized += 1;
    }
    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, source_names, columns, frame.rows, frame.device);
}

fn rankFrameFromColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    output_prefix: []const u8,
    rank_columns_value: anytype,
    comptime namesFn: anytype,
) RankFrameError!DeviceDataFrame {
    var rank_columns = rank_columns_value;
    var rank_columns_transferred: usize = 0;
    errdefer {
        for (rank_columns[rank_columns_transferred..]) |*col| col.deinit();
    }

    const source_names = try frame.allocator.alloc([]const u8, frame.columns.len + rank_columns.len);
    defer frame.allocator.free(source_names);
    for (frame.names, 0..) |source_name, i| source_names[i] = source_name;

    var rank_names = try namesFn(frame.allocator, output_prefix);
    defer freeOwnedNameItems(frame.allocator, rank_names[0..]);
    for (rank_names, 0..) |rank_name, i| source_names[frame.columns.len + i] = rank_name;

    const out = try appendRankColumns(DeviceDataFrame, frame, source_names, rank_columns);
    rank_columns_transferred = rank_columns.len;
    return out;
}

pub fn rollingRankProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceRollingRankOptions,
) RankFrameError!DeviceDataFrame {
    const rolling_value = try frame.column(name);
    const rolling_columns = try rollingRankProfileColumnsByValue(frame.allocator, rolling_value.*, options_value, frame.device, frame.rows);
    return rankFrameFromColumns(DeviceDataFrame, frame, output_prefix, rolling_columns, rollingRankProfileOutputNames);
}

pub fn expandingRankProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceExpandingRankOptions,
) RankFrameError!DeviceDataFrame {
    const expanding_value = try frame.column(name);
    const expanding_columns = try expandingRankProfileColumnsByValue(frame.allocator, expanding_value.*, options_value, frame.device, frame.rows);
    return rankFrameFromColumns(DeviceDataFrame, frame, output_prefix, expanding_columns, expandingRankProfileOutputNames);
}
