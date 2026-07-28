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

pub const QuantileMetrics = struct {
    allocator: std.mem.Allocator,
    q1: []f64,
    medians: []f64,
    q3: []f64,
    iqrs: []f64,
    validity: []bool,

    pub fn deinit(self: *QuantileMetrics) void {
        self.allocator.free(self.q1);
        self.allocator.free(self.medians);
        self.allocator.free(self.q3);
        self.allocator.free(self.iqrs);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub const RollingQuantileProfileColumnCount = 4;

pub fn rollingQuantileProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![RollingQuantileProfileColumnCount][]const u8 {
    var names: [RollingQuantileProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "rolling_q1", "rolling_median", "rolling_q3", "rolling_iqr" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const ExpandingQuantileProfileColumnCount = 4;

pub fn expandingQuantileProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![ExpandingQuantileProfileColumnCount][]const u8 {
    var names: [ExpandingQuantileProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "expanding_q1", "expanding_median", "expanding_q3", "expanding_iqr" };
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

fn lessFloat(lhs: f64, rhs: f64) bool {
    const lhs_nan = std.math.isNan(lhs);
    const rhs_nan = std.math.isNan(rhs);
    if (lhs_nan != rhs_nan) return !lhs_nan;
    if (lhs_nan and rhs_nan) return false;
    return lhs < rhs;
}

fn sort(values: []f64) void {
    std.sort.insertion(f64, values, {}, struct {
        fn lessThan(_: void, lhs: f64, rhs: f64) bool {
            return lessFloat(lhs, rhs);
        }
    }.lessThan);
}

fn quantileSorted(values: []const f64, probability: f64) f64 {
    if (values.len == 0) return std.math.nan(f64);
    if (values.len == 1) return values[0];
    const position = probability * @as(f64, @floatFromInt(values.len - 1));
    const lower: usize = @intFromFloat(@floor(position));
    const upper: usize = if (lower + 1 < values.len and position > @as(f64, @floatFromInt(lower))) lower + 1 else lower;
    const fraction = position - @as(f64, @floatFromInt(lower));
    return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

fn allocMetrics(allocator: std.mem.Allocator, rows: usize) std.mem.Allocator.Error!QuantileMetrics {
    const q1 = try allocator.alloc(f64, rows);
    errdefer allocator.free(q1);
    const medians = try allocator.alloc(f64, rows);
    errdefer allocator.free(medians);
    const q3 = try allocator.alloc(f64, rows);
    errdefer allocator.free(q3);
    const iqrs = try allocator.alloc(f64, rows);
    errdefer allocator.free(iqrs);
    const validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(validity);
    return .{ .allocator = allocator, .q1 = q1, .medians = medians, .q3 = q3, .iqrs = iqrs, .validity = validity };
}

fn writeQuantiles(row: usize, values: []f64, out: QuantileMetrics) void {
    sort(values);
    const q1 = quantileSorted(values, 0.25);
    const median = quantileSorted(values, 0.5);
    const q3 = quantileSorted(values, 0.75);
    out.q1[row] = q1;
    out.medians[row] = median;
    out.q3[row] = q3;
    out.iqrs[row] = q3 - q1;
    out.validity[row] = true;
}

fn writeInvalid(row: usize, out: QuantileMetrics) void {
    out.q1[row] = 0;
    out.medians[row] = 0;
    out.q3[row] = 0;
    out.iqrs[row] = 0;
    out.validity[row] = false;
}

pub fn rollingQuantileProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    window: usize,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!QuantileMetrics {
    if (window == 0) return error.InvalidShape;
    if (min_periods == 0 or min_periods > window) return error.InvalidShape;
    try validate(values, maybe_validity);

    var out = try allocMetrics(allocator, values.len);
    errdefer out.deinit();
    const scratch = try allocator.alloc(f64, window);
    defer allocator.free(scratch);

    for (0..values.len) |row| {
        const start = if (row + 1 > window) row + 1 - window else 0;
        var count: usize = 0;
        for (start..row + 1) |window_row| {
            if (!rowValid(maybe_validity, window_row)) continue;
            scratch[count] = values[window_row];
            count += 1;
        }

        if (!rowValid(maybe_validity, row) or count < min_periods) {
            writeInvalid(row, out);
            continue;
        }
        writeQuantiles(row, scratch[0..count], out);
    }

    return out;
}

pub fn expandingQuantileProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!QuantileMetrics {
    if (min_periods == 0) return error.InvalidShape;
    try validate(values, maybe_validity);

    var out = try allocMetrics(allocator, values.len);
    errdefer out.deinit();
    const scratch = try allocator.alloc(f64, values.len);
    defer allocator.free(scratch);

    var valid_count: usize = 0;
    for (values, 0..) |value, row| {
        if (rowValid(maybe_validity, row)) {
            scratch[valid_count] = value;
            valid_count += 1;
        }

        if (valid_count < min_periods) {
            writeInvalid(row, out);
            continue;
        }
        writeQuantiles(row, scratch[0..valid_count], out);
    }

    return out;
}

pub fn rollingQuantileProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![RollingQuantileProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| rollingQuantileProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| rollingQuantileProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| rollingQuantileProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| rollingQuantileProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| rollingQuantileProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| rollingQuantileProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| rollingQuantileProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| rollingQuantileProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| rollingQuantileProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| rollingQuantileProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| rollingQuantileProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| rollingQuantileProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| rollingQuantileProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn rollingQuantileProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![RollingQuantileProfileColumnCount]DeviceColumn {
    const min_periods = options_value.min_periods orelse options_value.window;
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);
    var metrics = try rollingQuantileProfile(allocator, values, maybe_validity, options_value.window, min_periods);
    defer metrics.deinit();

    var columns: [RollingQuantileProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.q1, metrics.validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.medians, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.q3, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.iqrs, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn expandingQuantileProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingQuantileProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| expandingQuantileProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| expandingQuantileProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| expandingQuantileProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| expandingQuantileProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| expandingQuantileProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| expandingQuantileProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| expandingQuantileProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| expandingQuantileProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| expandingQuantileProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| expandingQuantileProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| expandingQuantileProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| expandingQuantileProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| expandingQuantileProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn expandingQuantileProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingQuantileProfileColumnCount]DeviceColumn {
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);
    var metrics = try expandingQuantileProfile(allocator, values, maybe_validity, options_value.min_periods);
    defer metrics.deinit();

    var columns: [ExpandingQuantileProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.q1, metrics.validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.medians, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.q3, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.iqrs, metrics.validity, device_value);
    initialized += 1;
    return columns;
}

const QuantileFrameError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    TypeUnsupported,
    UnsupportedType,
    InvalidCsv,
    EmptyDataFrame,
    InvalidDevice,
};

fn appendQuantileColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    source_names: []const []const u8,
    quantile_columns: anytype,
) QuantileFrameError!DeviceDataFrame {
    const DeviceColumnType = std.meta.Elem(@TypeOf(frame.columns));
    var columns = try frame.allocator.alloc(DeviceColumnType, frame.columns.len + quantile_columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }
    for (frame.columns, 0..) |col, i| {
        columns[i] = try col.clone();
        initialized += 1;
    }
    for (&quantile_columns) |*quantile_col| {
        columns[initialized] = quantile_col.*;
        initialized += 1;
    }
    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, source_names, columns, frame.rows, frame.device);
}

fn quantileFrameFromColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    output_prefix: []const u8,
    quantile_columns_value: anytype,
    comptime namesFn: anytype,
) QuantileFrameError!DeviceDataFrame {
    var quantile_columns = quantile_columns_value;
    var quantile_columns_transferred: usize = 0;
    errdefer {
        for (quantile_columns[quantile_columns_transferred..]) |*col| col.deinit();
    }

    const source_names = try frame.allocator.alloc([]const u8, frame.columns.len + quantile_columns.len);
    defer frame.allocator.free(source_names);
    for (frame.names, 0..) |source_name, i| source_names[i] = source_name;

    var quantile_names = try namesFn(frame.allocator, output_prefix);
    defer names_mod.freeOwnedNameItems(frame.allocator, quantile_names[0..]);
    for (quantile_names, 0..) |quantile_name, i| source_names[frame.columns.len + i] = quantile_name;

    const out = try appendQuantileColumns(DeviceDataFrame, frame, source_names, quantile_columns);
    quantile_columns_transferred = quantile_columns.len;
    return out;
}

pub fn rollingQuantileProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceRollingOptions,
) QuantileFrameError!DeviceDataFrame {
    const rolling_value = try frame.column(name);
    const rolling_columns = try rollingQuantileProfileColumnsByValue(frame.allocator, rolling_value.*, options_value, frame.device, frame.rows);
    return quantileFrameFromColumns(DeviceDataFrame, frame, output_prefix, rolling_columns, rollingQuantileProfileOutputNames);
}

pub fn expandingQuantileProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceExpandingOptions,
) QuantileFrameError!DeviceDataFrame {
    const expanding_value = try frame.column(name);
    const expanding_columns = try expandingQuantileProfileColumnsByValue(frame.allocator, expanding_value.*, options_value, frame.device, frame.rows);
    return quantileFrameFromColumns(DeviceDataFrame, frame, output_prefix, expanding_columns, expandingQuantileProfileOutputNames);
}
