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

pub const NormalizeMetrics = struct {
    allocator: std.mem.Allocator,
    centered: []f64,
    zscores: []f64,
    minmax: []f64,
    validity: []bool,

    pub fn deinit(self: *NormalizeMetrics) void {
        self.allocator.free(self.centered);
        self.allocator.free(self.zscores);
        self.allocator.free(self.minmax);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub const RollingNormalizeProfileColumnCount = 3;

pub fn rollingNormalizeProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![RollingNormalizeProfileColumnCount][]const u8 {
    var names: [RollingNormalizeProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "rolling_centered", "rolling_zscore", "rolling_minmax" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const ExpandingNormalizeProfileColumnCount = 3;

pub fn expandingNormalizeProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![ExpandingNormalizeProfileColumnCount][]const u8 {
    var names: [ExpandingNormalizeProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "expanding_centered", "expanding_zscore", "expanding_minmax" };
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

fn allocMetrics(allocator: std.mem.Allocator, rows: usize) std.mem.Allocator.Error!NormalizeMetrics {
    const centered = try allocator.alloc(f64, rows);
    errdefer allocator.free(centered);
    const zscores = try allocator.alloc(f64, rows);
    errdefer allocator.free(zscores);
    const minmax = try allocator.alloc(f64, rows);
    errdefer allocator.free(minmax);
    const validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(validity);
    return .{ .allocator = allocator, .centered = centered, .zscores = zscores, .minmax = minmax, .validity = validity };
}

fn writeInvalid(out: NormalizeMetrics, row: usize) void {
    out.centered[row] = 0;
    out.zscores[row] = 0;
    out.minmax[row] = 0;
    out.validity[row] = false;
}

fn writeNormalized(out: NormalizeMetrics, row: usize, x: f64, count: usize, sum: f64, sum_sq: f64, low: f64, high: f64, min_periods: usize, current_valid: bool) void {
    const has_enough = current_valid and count >= min_periods;
    out.validity[row] = has_enough;
    if (!has_enough) {
        out.centered[row] = 0;
        out.zscores[row] = 0;
        out.minmax[row] = 0;
        return;
    }

    const n: f64 = @floatFromInt(count);
    const mean = sum / n;
    const raw_variance = sum_sq / n - mean * mean;
    const variance = if (raw_variance < 0) 0 else raw_variance;
    const stddev = std.math.sqrt(variance);
    const range = high - low;
    const delta = x - mean;
    out.centered[row] = delta;
    out.zscores[row] = if (stddev == 0) std.math.nan(f64) else delta / stddev;
    out.minmax[row] = if (range == 0) std.math.nan(f64) else (x - low) / range;
}

pub fn rollingNormalizeProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    window: usize,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!NormalizeMetrics {
    if (window == 0) return error.InvalidShape;
    if (min_periods == 0 or min_periods > window) return error.InvalidShape;
    try validate(values, maybe_validity);

    var out = try allocMetrics(allocator, values.len);
    errdefer out.deinit();

    for (values, 0..) |value, row| {
        const start = if (row + 1 > window) row + 1 - window else 0;
        var count: usize = 0;
        var sum: f64 = 0;
        var sum_sq: f64 = 0;
        var low: f64 = 0;
        var high: f64 = 0;
        for (start..row + 1) |window_row| {
            if (!rowValid(maybe_validity, window_row)) continue;
            const x = values[window_row];
            if (count == 0) {
                low = x;
                high = x;
            } else {
                if (x < low) low = x;
                if (x > high) high = x;
            }
            sum += x;
            sum_sq += x * x;
            count += 1;
        }
        writeNormalized(out, row, value, count, sum, sum_sq, low, high, min_periods, rowValid(maybe_validity, row));
    }

    return out;
}

pub fn expandingNormalizeProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!NormalizeMetrics {
    if (min_periods == 0) return error.InvalidShape;
    try validate(values, maybe_validity);

    var out = try allocMetrics(allocator, values.len);
    errdefer out.deinit();

    var count: usize = 0;
    var sum: f64 = 0;
    var sum_sq: f64 = 0;
    var low: f64 = 0;
    var high: f64 = 0;
    for (values, 0..) |value, row| {
        const valid = rowValid(maybe_validity, row);
        if (valid) {
            if (count == 0) {
                low = value;
                high = value;
            } else {
                if (value < low) low = value;
                if (value > high) high = value;
            }
            sum += value;
            sum_sq += value * value;
            count += 1;
        }
        writeNormalized(out, row, value, count, sum, sum_sq, low, high, min_periods, valid);
    }

    return out;
}

pub fn rollingNormalizeProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![RollingNormalizeProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| rollingNormalizeProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| rollingNormalizeProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| rollingNormalizeProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| rollingNormalizeProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| rollingNormalizeProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| rollingNormalizeProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| rollingNormalizeProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| rollingNormalizeProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| rollingNormalizeProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| rollingNormalizeProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| rollingNormalizeProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| rollingNormalizeProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| rollingNormalizeProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn rollingNormalizeProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![RollingNormalizeProfileColumnCount]DeviceColumn {
    const min_periods = options_value.min_periods orelse options_value.window;
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);
    var metrics = try rollingNormalizeProfile(allocator, values, maybe_validity, options_value.window, min_periods);
    defer metrics.deinit();

    var columns: [RollingNormalizeProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.centered, metrics.validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.zscores, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.minmax, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn expandingNormalizeProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingNormalizeProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| expandingNormalizeProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| expandingNormalizeProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| expandingNormalizeProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| expandingNormalizeProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| expandingNormalizeProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| expandingNormalizeProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| expandingNormalizeProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| expandingNormalizeProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| expandingNormalizeProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| expandingNormalizeProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| expandingNormalizeProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| expandingNormalizeProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| expandingNormalizeProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn expandingNormalizeProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingNormalizeProfileColumnCount]DeviceColumn {
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);
    var metrics = try expandingNormalizeProfile(allocator, values, maybe_validity, options_value.min_periods);
    defer metrics.deinit();

    var columns: [ExpandingNormalizeProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.centered, metrics.validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.zscores, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.minmax, metrics.validity, device_value);
    initialized += 1;
    return columns;
}

const NormalizeFrameError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    TypeUnsupported,
    UnsupportedType,
    InvalidCsv,
    EmptyDataFrame,
    InvalidDevice,
};

fn appendNormalizeColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    source_names: []const []const u8,
    normalize_columns: anytype,
) NormalizeFrameError!DeviceDataFrame {
    const DeviceColumnType = std.meta.Elem(@TypeOf(frame.columns));
    var columns = try frame.allocator.alloc(DeviceColumnType, frame.columns.len + normalize_columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }
    for (frame.columns, 0..) |col, i| {
        columns[i] = try col.clone();
        initialized += 1;
    }
    for (&normalize_columns) |*normalize_col| {
        columns[initialized] = normalize_col.*;
        initialized += 1;
    }
    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, source_names, columns, frame.rows, frame.device);
}

fn normalizeFrameFromColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    output_prefix: []const u8,
    normalize_columns_value: anytype,
    comptime namesFn: anytype,
) NormalizeFrameError!DeviceDataFrame {
    var normalize_columns = normalize_columns_value;
    var normalize_columns_transferred: usize = 0;
    errdefer {
        for (normalize_columns[normalize_columns_transferred..]) |*col| col.deinit();
    }

    const source_names = try frame.allocator.alloc([]const u8, frame.columns.len + normalize_columns.len);
    defer frame.allocator.free(source_names);
    for (frame.names, 0..) |source_name, i| source_names[i] = source_name;

    var normalize_names = try namesFn(frame.allocator, output_prefix);
    defer names_mod.freeOwnedNameItems(frame.allocator, normalize_names[0..]);
    for (normalize_names, 0..) |normalize_name, i| source_names[frame.columns.len + i] = normalize_name;

    const out = try appendNormalizeColumns(DeviceDataFrame, frame, source_names, normalize_columns);
    normalize_columns_transferred = normalize_columns.len;
    return out;
}

pub fn rollingNormalizeProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceRollingOptions,
) NormalizeFrameError!DeviceDataFrame {
    const rolling_value = try frame.column(name);
    const rolling_columns = try rollingNormalizeProfileColumnsByValue(frame.allocator, rolling_value.*, options_value, frame.device, frame.rows);
    return normalizeFrameFromColumns(DeviceDataFrame, frame, output_prefix, rolling_columns, rollingNormalizeProfileOutputNames);
}

pub fn expandingNormalizeProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceExpandingOptions,
) NormalizeFrameError!DeviceDataFrame {
    const expanding_value = try frame.column(name);
    const expanding_columns = try expandingNormalizeProfileColumnsByValue(frame.allocator, expanding_value.*, options_value, frame.device, frame.rows);
    return normalizeFrameFromColumns(DeviceDataFrame, frame, output_prefix, expanding_columns, expandingNormalizeProfileOutputNames);
}
