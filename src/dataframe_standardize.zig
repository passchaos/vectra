const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const names_mod = @import("dataframe_names.zig");
const dataframe_device_column_mod = @import("dataframe/device_column.zig");
const numeric_mod = @import("dataframe_numeric.zig");
const options_mod = @import("dataframe_options.zig");
const validity_mod = @import("dataframe/validity.zig");

const DeviceColumn = dataframe_device_column_mod.DeviceColumn;
const DeviceTypedColumn = dataframe_device_column_mod.DeviceTypedColumn;
const DeviceStandardizeOptions = options_mod.DeviceStandardizeOptions;
const castToF64 = numeric_mod.castToF64;
const validityValues = validity_mod.validityValues;

const StandardizeFrameError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    TypeUnsupported,
    UnsupportedType,
    InvalidCsv,
    EmptyDataFrame,
    InvalidDevice,
};

pub const StandardizeMetrics = struct {
    allocator: std.mem.Allocator,
    centered: []f64,
    zscores: []f64,
    minmax: []f64,
    validity: []bool,

    pub fn deinit(self: *StandardizeMetrics) void {
        self.allocator.free(self.centered);
        self.allocator.free(self.zscores);
        self.allocator.free(self.minmax);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub const StandardizeProfileColumnCount = 3;

pub fn standardizeProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![StandardizeProfileColumnCount][]const u8 {
    var names: [StandardizeProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "centered", "zscore", "minmax" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn validateLength(values: []const f64, maybe_validity: ?[]const bool) error{LengthMismatch}!void {
    if (maybe_validity) |validity| {
        if (validity.len != values.len) return error.LengthMismatch;
    }
}

fn rowValid(maybe_validity: ?[]const bool, row: usize) bool {
    return if (maybe_validity) |mask| mask[row] else true;
}

pub fn standardizeProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!StandardizeMetrics {
    if (min_periods == 0) return error.InvalidShape;
    try validateLength(values, maybe_validity);

    var count: usize = 0;
    var sum: f64 = 0;
    var sum_sq: f64 = 0;
    var min_value: f64 = 0;
    var max_value: f64 = 0;
    for (values, 0..) |value, row| {
        if (!rowValid(maybe_validity, row)) continue;
        if (count == 0) {
            min_value = value;
            max_value = value;
        } else {
            if (value < min_value) min_value = value;
            if (value > max_value) max_value = value;
        }
        sum += value;
        sum_sq += value * value;
        count += 1;
    }

    const rows = values.len;
    const centered = try allocator.alloc(f64, rows);
    errdefer allocator.free(centered);
    const zscores = try allocator.alloc(f64, rows);
    errdefer allocator.free(zscores);
    const minmax = try allocator.alloc(f64, rows);
    errdefer allocator.free(minmax);
    const validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(validity);

    const has_enough = count >= min_periods;
    const mean = if (count == 0) 0 else sum / @as(f64, @floatFromInt(count));
    const raw_variance = if (count == 0) 0 else sum_sq / @as(f64, @floatFromInt(count)) - mean * mean;
    const variance = if (raw_variance < 0) 0 else raw_variance;
    const stddev = std.math.sqrt(variance);
    const range = max_value - min_value;

    // Generate common whole-column scaling features in a single pass over the
    // materialized values. This mirrors feature-engineering pipelines that ask
    // for centered, z-score, and min-max forms together.
    for (values, 0..) |value, row| {
        const valid = rowValid(maybe_validity, row) and has_enough;
        validity[row] = valid;
        if (valid) {
            const delta = value - mean;
            centered[row] = delta;
            zscores[row] = if (stddev == 0) std.math.nan(f64) else delta / stddev;
            minmax[row] = if (range == 0) std.math.nan(f64) else (value - min_value) / range;
        } else {
            centered[row] = 0;
            zscores[row] = 0;
            minmax[row] = 0;
        }
    }

    return .{
        .allocator = allocator,
        .centered = centered,
        .zscores = zscores,
        .minmax = minmax,
        .validity = validity,
    };
}

pub fn standardizeProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceStandardizeOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![StandardizeProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| standardizeProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| standardizeProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| standardizeProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| standardizeProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| standardizeProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| standardizeProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| standardizeProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| standardizeProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| standardizeProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| standardizeProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| standardizeProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| standardizeProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| standardizeProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn standardizeProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceStandardizeOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![StandardizeProfileColumnCount]DeviceColumn {
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);

    var metrics = try standardizeProfile(allocator, values, maybe_validity, options_value.min_periods);
    defer metrics.deinit();

    var columns: [StandardizeProfileColumnCount]DeviceColumn = undefined;
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

pub fn standardizeProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceStandardizeOptions,
) StandardizeFrameError!DeviceDataFrame {
    const standardize_value = try frame.column(name);
    var standardize_columns = try standardizeProfileColumnsByValue(frame.allocator, standardize_value.*, options_value, frame.device, frame.rows);
    var standardize_columns_transferred: usize = 0;
    errdefer {
        for (standardize_columns[standardize_columns_transferred..]) |*col| col.deinit();
    }

    const source_names = try frame.allocator.alloc([]const u8, frame.columns.len + standardize_columns.len);
    defer frame.allocator.free(source_names);
    for (frame.names, 0..) |source_name, i| source_names[i] = source_name;

    var standardize_names = try standardizeProfileOutputNames(frame.allocator, output_prefix);
    defer names_mod.freeOwnedNameItems(frame.allocator, standardize_names[0..]);
    for (standardize_names, 0..) |standardize_name, i| source_names[frame.columns.len + i] = standardize_name;

    const DeviceColumnType = std.meta.Elem(@TypeOf(frame.columns));
    var columns = try frame.allocator.alloc(DeviceColumnType, frame.columns.len + standardize_columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }
    for (frame.columns, 0..) |col, i| {
        columns[i] = try col.clone();
        initialized += 1;
    }
    for (&standardize_columns) |*standardize_col| {
        columns[initialized] = standardize_col.*;
        initialized += 1;
        standardize_columns_transferred += 1;
    }

    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, source_names, columns, frame.rows, frame.device);
}
