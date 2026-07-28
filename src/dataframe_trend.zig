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
const DeviceTrendOptions = options_mod.DeviceTrendOptions;
const DeviceRollingOptions = options_mod.DeviceRollingOptions;
const DeviceExpandingOptions = options_mod.DeviceExpandingOptions;
const castToF64 = numeric_mod.castToF64;
const validityValues = validity_mod.validityValues;

pub const TrendMetrics = struct {
    allocator: std.mem.Allocator,
    trends: []i64,
    up_streak: []i64,
    down_streak: []i64,
    flat_streak: []i64,
    reversal: []bool,
    validity: []bool,

    pub fn deinit(self: *TrendMetrics) void {
        self.allocator.free(self.trends);
        self.allocator.free(self.up_streak);
        self.allocator.free(self.down_streak);
        self.allocator.free(self.flat_streak);
        self.allocator.free(self.reversal);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub const TrendSummaryMetrics = struct {
    allocator: std.mem.Allocator,
    counts: []i64,
    up_rates: []f64,
    down_rates: []f64,
    flat_rates: []f64,
    reversal_rates: []f64,
    validity: []bool,

    pub fn deinit(self: *TrendSummaryMetrics) void {
        self.allocator.free(self.counts);
        self.allocator.free(self.up_rates);
        self.allocator.free(self.down_rates);
        self.allocator.free(self.flat_rates);
        self.allocator.free(self.reversal_rates);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub const TrendProfileColumnCount = 5;

pub fn trendProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![TrendProfileColumnCount][]const u8 {
    var names: [TrendProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "trend", "up_streak", "down_streak", "flat_streak", "reversal" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const RollingTrendProfileColumnCount = 5;

pub fn rollingTrendProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![RollingTrendProfileColumnCount][]const u8 {
    var names: [RollingTrendProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "rolling_trend_count", "rolling_up_rate", "rolling_down_rate", "rolling_flat_rate", "rolling_reversal_rate" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const ExpandingTrendProfileColumnCount = 5;

pub fn expandingTrendProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![ExpandingTrendProfileColumnCount][]const u8 {
    var names: [ExpandingTrendProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "expanding_trend_count", "expanding_up_rate", "expanding_down_rate", "expanding_flat_rate", "expanding_reversal_rate" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn validate(values: []const f64, maybe_validity: ?[]const bool, periods: usize) error{ InvalidShape, LengthMismatch }!void {
    if (periods == 0) return error.InvalidShape;
    if (maybe_validity) |validity| {
        if (validity.len != values.len) return error.LengthMismatch;
    }
}

fn rowValid(maybe_validity: ?[]const bool, row: usize) bool {
    return if (maybe_validity) |mask| mask[row] else true;
}

fn trendValue(current: f64, previous: f64) i64 {
    return if (current > previous) 1 else if (current < previous) -1 else 0;
}

fn allocSummary(allocator: std.mem.Allocator, rows: usize) std.mem.Allocator.Error!TrendSummaryMetrics {
    const counts = try allocator.alloc(i64, rows);
    errdefer allocator.free(counts);
    const up_rates = try allocator.alloc(f64, rows);
    errdefer allocator.free(up_rates);
    const down_rates = try allocator.alloc(f64, rows);
    errdefer allocator.free(down_rates);
    const flat_rates = try allocator.alloc(f64, rows);
    errdefer allocator.free(flat_rates);
    const reversal_rates = try allocator.alloc(f64, rows);
    errdefer allocator.free(reversal_rates);
    const validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(validity);
    return .{
        .allocator = allocator,
        .counts = counts,
        .up_rates = up_rates,
        .down_rates = down_rates,
        .flat_rates = flat_rates,
        .reversal_rates = reversal_rates,
        .validity = validity,
    };
}

fn computeTrendEvents(allocator: std.mem.Allocator, values: []const f64, maybe_validity: ?[]const bool, periods: usize) !struct { trends: []i64, reversals: []bool, validity: []bool } {
    const trends = try allocator.alloc(i64, values.len);
    errdefer allocator.free(trends);
    const reversals = try allocator.alloc(bool, values.len);
    errdefer allocator.free(reversals);
    const validity = try allocator.alloc(bool, values.len);
    errdefer allocator.free(validity);

    var previous_nonzero_trend: i64 = 0;
    for (values, 0..) |value, row| {
        if (row < periods) {
            trends[row] = 0;
            reversals[row] = false;
            validity[row] = false;
            previous_nonzero_trend = 0;
            continue;
        }

        const previous_row = row - periods;
        const valid = rowValid(maybe_validity, row) and rowValid(maybe_validity, previous_row);
        validity[row] = valid;
        if (!valid) {
            trends[row] = 0;
            reversals[row] = false;
            previous_nonzero_trend = 0;
            continue;
        }

        const trend = trendValue(value, values[previous_row]);
        trends[row] = trend;
        reversals[row] = trend != 0 and previous_nonzero_trend != 0 and trend != previous_nonzero_trend;
        if (trend != 0) previous_nonzero_trend = trend;
    }

    return .{ .trends = trends, .reversals = reversals, .validity = validity };
}

fn writeSummary(row: usize, count: usize, up_count: usize, down_count: usize, flat_count: usize, reversal_count: usize, min_periods: usize, out: TrendSummaryMetrics) void {
    out.counts[row] = @intCast(count);
    const has_enough = count >= min_periods;
    out.validity[row] = has_enough;
    if (has_enough) {
        const n: f64 = @floatFromInt(count);
        out.up_rates[row] = @as(f64, @floatFromInt(up_count)) / n;
        out.down_rates[row] = @as(f64, @floatFromInt(down_count)) / n;
        out.flat_rates[row] = @as(f64, @floatFromInt(flat_count)) / n;
        out.reversal_rates[row] = @as(f64, @floatFromInt(reversal_count)) / n;
    } else {
        out.up_rates[row] = 0;
        out.down_rates[row] = 0;
        out.flat_rates[row] = 0;
        out.reversal_rates[row] = 0;
    }
}

pub fn trendProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!TrendMetrics {
    try validate(values, maybe_validity, periods);

    const events = try computeTrendEvents(allocator, values, maybe_validity, periods);
    errdefer allocator.free(events.trends);
    errdefer allocator.free(events.reversals);
    errdefer allocator.free(events.validity);

    const up_streak = try allocator.alloc(i64, values.len);
    errdefer allocator.free(up_streak);
    const down_streak = try allocator.alloc(i64, values.len);
    errdefer allocator.free(down_streak);
    const flat_streak = try allocator.alloc(i64, values.len);
    errdefer allocator.free(flat_streak);

    var current_up: i64 = 0;
    var current_down: i64 = 0;
    var current_flat: i64 = 0;
    for (0..values.len) |row| {
        if (!events.validity[row]) {
            up_streak[row] = 0;
            down_streak[row] = 0;
            flat_streak[row] = 0;
            current_up = 0;
            current_down = 0;
            current_flat = 0;
            continue;
        }

        switch (events.trends[row]) {
            1 => {
                current_up += 1;
                current_down = 0;
                current_flat = 0;
            },
            -1 => {
                current_down += 1;
                current_up = 0;
                current_flat = 0;
            },
            else => {
                current_flat += 1;
                current_up = 0;
                current_down = 0;
            },
        }
        up_streak[row] = current_up;
        down_streak[row] = current_down;
        flat_streak[row] = current_flat;
    }

    return .{
        .allocator = allocator,
        .trends = events.trends,
        .up_streak = up_streak,
        .down_streak = down_streak,
        .flat_streak = flat_streak,
        .reversal = events.reversals,
        .validity = events.validity,
    };
}

pub fn rollingTrendProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    periods: usize,
    window: usize,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!TrendSummaryMetrics {
    if (window == 0) return error.InvalidShape;
    if (min_periods == 0 or min_periods > window) return error.InvalidShape;
    try validate(values, maybe_validity, periods);

    const events = try computeTrendEvents(allocator, values, maybe_validity, periods);
    defer allocator.free(events.trends);
    defer allocator.free(events.reversals);
    defer allocator.free(events.validity);

    var out = try allocSummary(allocator, values.len);
    errdefer out.deinit();
    for (0..values.len) |row| {
        const start = if (row + 1 > window) row + 1 - window else 0;
        var count: usize = 0;
        var up_count: usize = 0;
        var down_count: usize = 0;
        var flat_count: usize = 0;
        var reversal_count: usize = 0;
        for (start..row + 1) |window_row| {
            if (!events.validity[window_row]) continue;
            switch (events.trends[window_row]) {
                1 => up_count += 1,
                -1 => down_count += 1,
                else => flat_count += 1,
            }
            if (events.reversals[window_row]) reversal_count += 1;
            count += 1;
        }
        writeSummary(row, count, up_count, down_count, flat_count, reversal_count, min_periods, out);
    }
    return out;
}

pub fn expandingTrendProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    periods: usize,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!TrendSummaryMetrics {
    if (min_periods == 0) return error.InvalidShape;
    try validate(values, maybe_validity, periods);

    const events = try computeTrendEvents(allocator, values, maybe_validity, periods);
    defer allocator.free(events.trends);
    defer allocator.free(events.reversals);
    defer allocator.free(events.validity);

    var out = try allocSummary(allocator, values.len);
    errdefer out.deinit();
    var count: usize = 0;
    var up_count: usize = 0;
    var down_count: usize = 0;
    var flat_count: usize = 0;
    var reversal_count: usize = 0;
    for (0..values.len) |row| {
        if (events.validity[row]) {
            switch (events.trends[row]) {
                1 => up_count += 1,
                -1 => down_count += 1,
                else => flat_count += 1,
            }
            if (events.reversals[row]) reversal_count += 1;
            count += 1;
        }
        writeSummary(row, count, up_count, down_count, flat_count, reversal_count, min_periods, out);
    }
    return out;
}

pub fn trendProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceTrendOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![TrendProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| trendProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| trendProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| trendProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| trendProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| trendProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| trendProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| trendProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| trendProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| trendProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| trendProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| trendProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| trendProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| trendProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn trendProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceTrendOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![TrendProfileColumnCount]DeviceColumn {
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);
    var metrics = try trendProfile(allocator, values, maybe_validity, options_value.periods);
    defer metrics.deinit();

    var columns: [TrendProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(i64, allocator, metrics.trends, metrics.validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(i64, allocator, metrics.up_streak, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(i64, allocator, metrics.down_streak, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(i64, allocator, metrics.flat_streak, metrics.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(bool, allocator, metrics.reversal, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn rollingTrendProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    trend_options: DeviceTrendOptions,
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![RollingTrendProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| rollingTrendProfileColumnsTyped(i8, allocator, typed, trend_options, options_value, device_value),
        .i16 => |typed| rollingTrendProfileColumnsTyped(i16, allocator, typed, trend_options, options_value, device_value),
        .i32 => |typed| rollingTrendProfileColumnsTyped(i32, allocator, typed, trend_options, options_value, device_value),
        .i64 => |typed| rollingTrendProfileColumnsTyped(i64, allocator, typed, trend_options, options_value, device_value),
        .u8 => |typed| rollingTrendProfileColumnsTyped(u8, allocator, typed, trend_options, options_value, device_value),
        .u16 => |typed| rollingTrendProfileColumnsTyped(u16, allocator, typed, trend_options, options_value, device_value),
        .u32 => |typed| rollingTrendProfileColumnsTyped(u32, allocator, typed, trend_options, options_value, device_value),
        .u64 => |typed| rollingTrendProfileColumnsTyped(u64, allocator, typed, trend_options, options_value, device_value),
        .usize => |typed| rollingTrendProfileColumnsTyped(usize, allocator, typed, trend_options, options_value, device_value),
        .isize => |typed| rollingTrendProfileColumnsTyped(isize, allocator, typed, trend_options, options_value, device_value),
        .f16 => |typed| rollingTrendProfileColumnsTyped(f16, allocator, typed, trend_options, options_value, device_value),
        .f32 => |typed| rollingTrendProfileColumnsTyped(f32, allocator, typed, trend_options, options_value, device_value),
        .f64 => |typed| rollingTrendProfileColumnsTyped(f64, allocator, typed, trend_options, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn rollingTrendProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    trend_options: DeviceTrendOptions,
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![RollingTrendProfileColumnCount]DeviceColumn {
    const min_periods = options_value.min_periods orelse options_value.window;
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);
    var metrics = try rollingTrendProfile(allocator, values, maybe_validity, trend_options.periods, options_value.window, min_periods);
    defer metrics.deinit();

    var columns: [RollingTrendProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.up_rates, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.down_rates, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.flat_rates, metrics.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.reversal_rates, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn expandingTrendProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    trend_options: DeviceTrendOptions,
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingTrendProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| expandingTrendProfileColumnsTyped(i8, allocator, typed, trend_options, options_value, device_value),
        .i16 => |typed| expandingTrendProfileColumnsTyped(i16, allocator, typed, trend_options, options_value, device_value),
        .i32 => |typed| expandingTrendProfileColumnsTyped(i32, allocator, typed, trend_options, options_value, device_value),
        .i64 => |typed| expandingTrendProfileColumnsTyped(i64, allocator, typed, trend_options, options_value, device_value),
        .u8 => |typed| expandingTrendProfileColumnsTyped(u8, allocator, typed, trend_options, options_value, device_value),
        .u16 => |typed| expandingTrendProfileColumnsTyped(u16, allocator, typed, trend_options, options_value, device_value),
        .u32 => |typed| expandingTrendProfileColumnsTyped(u32, allocator, typed, trend_options, options_value, device_value),
        .u64 => |typed| expandingTrendProfileColumnsTyped(u64, allocator, typed, trend_options, options_value, device_value),
        .usize => |typed| expandingTrendProfileColumnsTyped(usize, allocator, typed, trend_options, options_value, device_value),
        .isize => |typed| expandingTrendProfileColumnsTyped(isize, allocator, typed, trend_options, options_value, device_value),
        .f16 => |typed| expandingTrendProfileColumnsTyped(f16, allocator, typed, trend_options, options_value, device_value),
        .f32 => |typed| expandingTrendProfileColumnsTyped(f32, allocator, typed, trend_options, options_value, device_value),
        .f64 => |typed| expandingTrendProfileColumnsTyped(f64, allocator, typed, trend_options, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn expandingTrendProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    trend_options: DeviceTrendOptions,
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingTrendProfileColumnCount]DeviceColumn {
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);
    var metrics = try expandingTrendProfile(allocator, values, maybe_validity, trend_options.periods, options_value.min_periods);
    defer metrics.deinit();

    var columns: [ExpandingTrendProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.up_rates, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.down_rates, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.flat_rates, metrics.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.reversal_rates, metrics.validity, device_value);
    initialized += 1;
    return columns;
}

const TrendFrameError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    TypeUnsupported,
    UnsupportedType,
    InvalidCsv,
    EmptyDataFrame,
    InvalidDevice,
};

fn appendTrendColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    source_names: []const []const u8,
    trend_columns: anytype,
) TrendFrameError!DeviceDataFrame {
    const DeviceColumnType = std.meta.Elem(@TypeOf(frame.columns));
    var columns = try frame.allocator.alloc(DeviceColumnType, frame.columns.len + trend_columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }
    for (frame.columns, 0..) |col, i| {
        columns[i] = try col.clone();
        initialized += 1;
    }
    for (&trend_columns) |*trend_col| {
        columns[initialized] = trend_col.*;
        initialized += 1;
    }
    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, source_names, columns, frame.rows, frame.device);
}

fn trendProfileFrameFromColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    output_prefix: []const u8,
    trend_columns_value: anytype,
    comptime namesFn: anytype,
) TrendFrameError!DeviceDataFrame {
    var trend_columns = trend_columns_value;
    var trend_columns_transferred: usize = 0;
    errdefer {
        for (trend_columns[trend_columns_transferred..]) |*col| col.deinit();
    }

    const source_names = try frame.allocator.alloc([]const u8, frame.columns.len + trend_columns.len);
    defer frame.allocator.free(source_names);
    for (frame.names, 0..) |source_name, i| source_names[i] = source_name;

    var trend_names = try namesFn(frame.allocator, output_prefix);
    defer names_mod.freeOwnedNameItems(frame.allocator, trend_names[0..]);
    for (trend_names, 0..) |trend_name, i| source_names[frame.columns.len + i] = trend_name;

    const out = try appendTrendColumns(DeviceDataFrame, frame, source_names, trend_columns);
    trend_columns_transferred = trend_columns.len;
    return out;
}

pub fn trendProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceTrendOptions,
) TrendFrameError!DeviceDataFrame {
    const trend_value = try frame.column(name);
    const trend_columns = try trendProfileColumnsByValue(frame.allocator, trend_value.*, options_value, frame.device, frame.rows);
    return trendProfileFrameFromColumns(DeviceDataFrame, frame, output_prefix, trend_columns, trendProfileOutputNames);
}

pub fn rollingTrendProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    trend_options: DeviceTrendOptions,
    options_value: DeviceRollingOptions,
) TrendFrameError!DeviceDataFrame {
    const trend_value = try frame.column(name);
    const trend_columns = try rollingTrendProfileColumnsByValue(frame.allocator, trend_value.*, trend_options, options_value, frame.device, frame.rows);
    return trendProfileFrameFromColumns(DeviceDataFrame, frame, output_prefix, trend_columns, rollingTrendProfileOutputNames);
}

pub fn expandingTrendProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    trend_options: DeviceTrendOptions,
    options_value: DeviceExpandingOptions,
) TrendFrameError!DeviceDataFrame {
    const trend_value = try frame.column(name);
    const trend_columns = try expandingTrendProfileColumnsByValue(frame.allocator, trend_value.*, trend_options, options_value, frame.device, frame.rows);
    return trendProfileFrameFromColumns(DeviceDataFrame, frame, output_prefix, trend_columns, expandingTrendProfileOutputNames);
}
