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
const DeviceRollingOptions = options_mod.DeviceRollingOptions;
const DeviceExpandingOptions = options_mod.DeviceExpandingOptions;
const castToF64 = numeric_mod.castToF64;
const validityValues = validity_mod.validityValues;

pub const RollingStatsMetrics = struct {
    allocator: std.mem.Allocator,
    counts: []i64,
    sums: []f64,
    means: []f64,
    variances: []f64,
    stddevs: []f64,
    validity: []bool,

    pub fn deinit(self: *RollingStatsMetrics) void {
        self.allocator.free(self.counts);
        self.allocator.free(self.sums);
        self.allocator.free(self.means);
        self.allocator.free(self.variances);
        self.allocator.free(self.stddevs);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub const ExpandingStatsMetrics = struct {
    allocator: std.mem.Allocator,
    counts: []i64,
    sums: []f64,
    means: []f64,
    mins: []f64,
    maxes: []f64,
    validity: []bool,

    pub fn deinit(self: *ExpandingStatsMetrics) void {
        self.allocator.free(self.counts);
        self.allocator.free(self.sums);
        self.allocator.free(self.means);
        self.allocator.free(self.mins);
        self.allocator.free(self.maxes);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub const RollingProfileColumnCount = 5;

pub fn rollingProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![RollingProfileColumnCount][]const u8 {
    var names: [RollingProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "rolling_count", "rolling_sum", "rolling_mean", "rolling_variance", "rolling_stddev" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const ExpandingProfileColumnCount = 5;

pub fn expandingProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![ExpandingProfileColumnCount][]const u8 {
    var names: [ExpandingProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "expanding_count", "expanding_sum", "expanding_mean", "expanding_min", "expanding_max" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn validate(values: []const f64, maybe_validity: ?[]const bool) error{LengthMismatch}!void {
    if (maybe_validity) |validity| {
        if (validity.len != values.len) return error.LengthMismatch;
    }
}

fn rowValid(maybe_validity: ?[]const bool, row: usize) bool {
    return if (maybe_validity) |mask| mask[row] else true;
}

pub fn rollingProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    window: usize,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!RollingStatsMetrics {
    if (window == 0) return error.InvalidShape;
    if (min_periods == 0 or min_periods > window) return error.InvalidShape;
    try validate(values, maybe_validity);

    const counts = try allocator.alloc(i64, values.len);
    errdefer allocator.free(counts);
    const sums = try allocator.alloc(f64, values.len);
    errdefer allocator.free(sums);
    const means = try allocator.alloc(f64, values.len);
    errdefer allocator.free(means);
    const variances = try allocator.alloc(f64, values.len);
    errdefer allocator.free(variances);
    const stddevs = try allocator.alloc(f64, values.len);
    errdefer allocator.free(stddevs);
    const validity = try allocator.alloc(bool, values.len);
    errdefer allocator.free(validity);

    var running_sum: f64 = 0;
    var running_sum_sq: f64 = 0;
    var running_count: usize = 0;
    for (values, 0..) |value, row| {
        if (rowValid(maybe_validity, row)) {
            running_sum += value;
            running_sum_sq += value * value;
            running_count += 1;
        }
        if (row >= window) {
            const evict_row = row - window;
            if (rowValid(maybe_validity, evict_row)) {
                const x = values[evict_row];
                running_sum -= x;
                running_sum_sq -= x * x;
                running_count -= 1;
            }
        }

        counts[row] = @intCast(running_count);
        const has_enough = running_count >= min_periods;
        validity[row] = has_enough;
        if (has_enough) {
            const n: f64 = @floatFromInt(running_count);
            const mean = running_sum / n;
            const raw_variance = running_sum_sq / n - mean * mean;
            const variance = if (raw_variance < 0) 0 else raw_variance;
            sums[row] = running_sum;
            means[row] = mean;
            variances[row] = variance;
            stddevs[row] = std.math.sqrt(variance);
        } else {
            sums[row] = 0;
            means[row] = 0;
            variances[row] = 0;
            stddevs[row] = 0;
        }
    }

    return .{ .allocator = allocator, .counts = counts, .sums = sums, .means = means, .variances = variances, .stddevs = stddevs, .validity = validity };
}

pub fn expandingProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!ExpandingStatsMetrics {
    if (min_periods == 0) return error.InvalidShape;
    try validate(values, maybe_validity);

    const counts = try allocator.alloc(i64, values.len);
    errdefer allocator.free(counts);
    const sums = try allocator.alloc(f64, values.len);
    errdefer allocator.free(sums);
    const means = try allocator.alloc(f64, values.len);
    errdefer allocator.free(means);
    const mins = try allocator.alloc(f64, values.len);
    errdefer allocator.free(mins);
    const maxes = try allocator.alloc(f64, values.len);
    errdefer allocator.free(maxes);
    const validity = try allocator.alloc(bool, values.len);
    errdefer allocator.free(validity);

    var running_count: usize = 0;
    var running_sum: f64 = 0;
    var running_min: f64 = 0;
    var running_max: f64 = 0;
    for (values, 0..) |value, row| {
        if (rowValid(maybe_validity, row)) {
            if (running_count == 0) {
                running_min = value;
                running_max = value;
            } else {
                if (value < running_min) running_min = value;
                if (value > running_max) running_max = value;
            }
            running_sum += value;
            running_count += 1;
        }

        counts[row] = @intCast(running_count);
        const has_enough = running_count >= min_periods;
        validity[row] = has_enough;
        if (has_enough) {
            sums[row] = running_sum;
            means[row] = running_sum / @as(f64, @floatFromInt(running_count));
            mins[row] = running_min;
            maxes[row] = running_max;
        } else {
            sums[row] = 0;
            means[row] = 0;
            mins[row] = 0;
            maxes[row] = 0;
        }
    }

    return .{ .allocator = allocator, .counts = counts, .sums = sums, .means = means, .mins = mins, .maxes = maxes, .validity = validity };
}

pub fn rollingProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![RollingProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| rollingProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| rollingProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| rollingProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| rollingProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| rollingProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| rollingProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| rollingProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| rollingProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| rollingProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| rollingProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| rollingProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| rollingProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| rollingProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn rollingProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![RollingProfileColumnCount]DeviceColumn {
    const min_periods = options_value.min_periods orelse options_value.window;
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);
    var metrics = try rollingProfile(allocator, values, maybe_validity, options_value.window, min_periods);
    defer metrics.deinit();

    var columns: [RollingProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.sums, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.means, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.variances, metrics.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.stddevs, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn expandingProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| expandingProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| expandingProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| expandingProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| expandingProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| expandingProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| expandingProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| expandingProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| expandingProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| expandingProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| expandingProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| expandingProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| expandingProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| expandingProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn expandingProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingProfileColumnCount]DeviceColumn {
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);
    var metrics = try expandingProfile(allocator, values, maybe_validity, options_value.min_periods);
    defer metrics.deinit();

    var columns: [ExpandingProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.sums, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.means, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.mins, metrics.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.maxes, metrics.validity, device_value);
    initialized += 1;
    return columns;
}

const StatsFrameError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    TypeUnsupported,
    UnsupportedType,
    InvalidCsv,
    EmptyDataFrame,
    InvalidDevice,
};

fn appendStatsColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    source_names: []const []const u8,
    stats_columns: anytype,
) StatsFrameError!DeviceDataFrame {
    const DeviceColumnType = std.meta.Elem(@TypeOf(frame.columns));
    var columns = try frame.allocator.alloc(DeviceColumnType, frame.columns.len + stats_columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }
    for (frame.columns, 0..) |col, i| {
        columns[i] = try col.clone();
        initialized += 1;
    }
    for (&stats_columns) |*stats_col| {
        columns[initialized] = stats_col.*;
        initialized += 1;
    }
    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, source_names, columns, frame.rows, frame.device);
}

fn statsFrameFromColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    output_prefix: []const u8,
    stats_columns_value: anytype,
    comptime namesFn: anytype,
) StatsFrameError!DeviceDataFrame {
    var stats_columns = stats_columns_value;
    var stats_columns_transferred: usize = 0;
    errdefer {
        for (stats_columns[stats_columns_transferred..]) |*col| col.deinit();
    }

    const source_names = try frame.allocator.alloc([]const u8, frame.columns.len + stats_columns.len);
    defer frame.allocator.free(source_names);
    for (frame.names, 0..) |source_name, i| source_names[i] = source_name;

    var stats_names = try namesFn(frame.allocator, output_prefix);
    defer names_mod.freeOwnedNameItems(frame.allocator, stats_names[0..]);
    for (stats_names, 0..) |stats_name, i| source_names[frame.columns.len + i] = stats_name;

    const out = try appendStatsColumns(DeviceDataFrame, frame, source_names, stats_columns);
    stats_columns_transferred = stats_columns.len;
    return out;
}

pub fn rollingProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceRollingOptions,
) StatsFrameError!DeviceDataFrame {
    const rolling_value = try frame.column(name);
    const rolling_columns = try rollingProfileColumnsByValue(frame.allocator, rolling_value.*, options_value, frame.device, frame.rows);
    return statsFrameFromColumns(DeviceDataFrame, frame, output_prefix, rolling_columns, rollingProfileOutputNames);
}

pub fn expandingProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceExpandingOptions,
) StatsFrameError!DeviceDataFrame {
    const expanding_value = try frame.column(name);
    const expanding_columns = try expandingProfileColumnsByValue(frame.allocator, expanding_value.*, options_value, frame.device, frame.rows);
    return statsFrameFromColumns(DeviceDataFrame, frame, output_prefix, expanding_columns, expandingProfileOutputNames);
}
