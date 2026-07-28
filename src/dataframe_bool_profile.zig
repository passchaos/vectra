const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const names_mod = @import("dataframe_names.zig");
const dataframe_device_column_mod = @import("dataframe_device_column.zig");
const options_mod = @import("dataframe_options.zig");
const validity_mod = @import("dataframe_validity.zig");

const DeviceColumn = dataframe_device_column_mod.DeviceColumn;
const DeviceTypedColumn = dataframe_device_column_mod.DeviceTypedColumn;
const DeviceRollingOptions = options_mod.DeviceRollingOptions;
const DeviceExpandingOptions = options_mod.DeviceExpandingOptions;
const validityValues = validity_mod.validityValues;

pub const BoolProfileMetrics = struct {
    allocator: std.mem.Allocator,
    true_counts: []i64,
    false_counts: []i64,
    true_rates: []f64,
    any_values: []bool,
    all_values: []bool,
    validity: []bool,

    pub fn deinit(self: *BoolProfileMetrics) void {
        self.allocator.free(self.true_counts);
        self.allocator.free(self.false_counts);
        self.allocator.free(self.true_rates);
        self.allocator.free(self.any_values);
        self.allocator.free(self.all_values);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub const RollingBoolProfileColumnCount = 5;

pub fn rollingBoolProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![RollingBoolProfileColumnCount][]const u8 {
    var names: [RollingBoolProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "rolling_true_count", "rolling_false_count", "rolling_true_rate", "rolling_any", "rolling_all" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const ExpandingBoolProfileColumnCount = 5;

pub fn expandingBoolProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![ExpandingBoolProfileColumnCount][]const u8 {
    var names: [ExpandingBoolProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "expanding_true_count", "expanding_false_count", "expanding_true_rate", "expanding_any", "expanding_all" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn validateLength(values: []const bool, maybe_validity: ?[]const bool) error{LengthMismatch}!void {
    if (maybe_validity) |validity| {
        if (validity.len != values.len) return error.LengthMismatch;
    }
}

fn rowValid(maybe_validity: ?[]const bool, row: usize) bool {
    return if (maybe_validity) |mask| mask[row] else true;
}

fn allocMetrics(allocator: std.mem.Allocator, rows: usize) std.mem.Allocator.Error!BoolProfileMetrics {
    const true_counts = try allocator.alloc(i64, rows);
    errdefer allocator.free(true_counts);
    const false_counts = try allocator.alloc(i64, rows);
    errdefer allocator.free(false_counts);
    const true_rates = try allocator.alloc(f64, rows);
    errdefer allocator.free(true_rates);
    const any_values = try allocator.alloc(bool, rows);
    errdefer allocator.free(any_values);
    const all_values = try allocator.alloc(bool, rows);
    errdefer allocator.free(all_values);
    const validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(validity);
    return .{
        .allocator = allocator,
        .true_counts = true_counts,
        .false_counts = false_counts,
        .true_rates = true_rates,
        .any_values = any_values,
        .all_values = all_values,
        .validity = validity,
    };
}

fn writeRow(row: usize, min_periods: usize, current_valid: bool, true_count: usize, false_count: usize, out: BoolProfileMetrics) void {
    const valid_count = true_count + false_count;
    out.true_counts[row] = @intCast(true_count);
    out.false_counts[row] = @intCast(false_count);
    const has_enough = current_valid and valid_count >= min_periods;
    out.validity[row] = has_enough;
    if (has_enough) {
        out.true_rates[row] = @as(f64, @floatFromInt(true_count)) / @as(f64, @floatFromInt(valid_count));
        out.any_values[row] = true_count != 0;
        out.all_values[row] = false_count == 0;
    } else {
        out.true_rates[row] = 0;
        out.any_values[row] = false;
        out.all_values[row] = false;
    }
}

pub fn rollingBoolProfile(
    allocator: std.mem.Allocator,
    values: []const bool,
    maybe_validity: ?[]const bool,
    window: usize,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!BoolProfileMetrics {
    if (window == 0) return error.InvalidShape;
    if (min_periods == 0 or min_periods > window) return error.InvalidShape;
    try validateLength(values, maybe_validity);

    var out = try allocMetrics(allocator, values.len);
    errdefer out.deinit();

    var running_true: usize = 0;
    var running_false: usize = 0;
    for (values, 0..) |value, row| {
        if (rowValid(maybe_validity, row)) {
            if (value) {
                running_true += 1;
            } else {
                running_false += 1;
            }
        }

        if (row >= window) {
            const evict_row = row - window;
            if (rowValid(maybe_validity, evict_row)) {
                if (values[evict_row]) {
                    running_true -= 1;
                } else {
                    running_false -= 1;
                }
            }
        }

        // Counts are always materialized as window diagnostics. Predicate/rate
        // outputs are nullable because they describe the current row's trailing
        // context: a null current row or too few valid observations leaves the
        // derived state unknown while preserving audit counts.
        writeRow(row, min_periods, rowValid(maybe_validity, row), running_true, running_false, out);
    }

    return out;
}

pub fn expandingBoolProfile(
    allocator: std.mem.Allocator,
    values: []const bool,
    maybe_validity: ?[]const bool,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!BoolProfileMetrics {
    if (min_periods == 0) return error.InvalidShape;
    try validateLength(values, maybe_validity);

    var out = try allocMetrics(allocator, values.len);
    errdefer out.deinit();

    var running_true: usize = 0;
    var running_false: usize = 0;
    for (values, 0..) |value, row| {
        if (rowValid(maybe_validity, row)) {
            if (value) {
                running_true += 1;
            } else {
                running_false += 1;
            }
        }

        writeRow(row, min_periods, rowValid(maybe_validity, row), running_true, running_false, out);
    }

    return out;
}

pub fn rollingBoolProfileColumns(
    allocator: std.mem.Allocator,
    source: DeviceTypedColumn(bool),
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![RollingBoolProfileColumnCount]DeviceColumn {
    const min_periods = options_value.min_periods orelse options_value.window;
    if (source.len() != rows) return error.LengthMismatch;

    const values = try source.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(source, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    var metrics = try rollingBoolProfile(allocator, values, maybe_validity, options_value.window, min_periods);
    defer metrics.deinit();

    var columns: [RollingBoolProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.true_counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSlice(i64, allocator, metrics.false_counts, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.true_rates, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(bool, allocator, metrics.any_values, metrics.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(bool, allocator, metrics.all_values, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn expandingBoolProfileColumns(
    allocator: std.mem.Allocator,
    source: DeviceTypedColumn(bool),
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingBoolProfileColumnCount]DeviceColumn {
    if (source.len() != rows) return error.LengthMismatch;

    const values = try source.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(source, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    var metrics = try expandingBoolProfile(allocator, values, maybe_validity, options_value.min_periods);
    defer metrics.deinit();

    var columns: [ExpandingBoolProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.true_counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSlice(i64, allocator, metrics.false_counts, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.true_rates, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(bool, allocator, metrics.any_values, metrics.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(bool, allocator, metrics.all_values, metrics.validity, device_value);
    initialized += 1;
    return columns;
}

const BoolFrameError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    TypeUnsupported,
    UnsupportedType,
    InvalidCsv,
    EmptyDataFrame,
    InvalidDevice,
};

fn appendBoolColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    source_names: []const []const u8,
    bool_columns: anytype,
) BoolFrameError!DeviceDataFrame {
    const DeviceColumnType = std.meta.Elem(@TypeOf(frame.columns));
    var columns = try frame.allocator.alloc(DeviceColumnType, frame.columns.len + bool_columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }
    for (frame.columns, 0..) |col, i| {
        columns[i] = try col.clone();
        initialized += 1;
    }
    for (&bool_columns) |*bool_col| {
        columns[initialized] = bool_col.*;
        initialized += 1;
    }
    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, source_names, columns, frame.rows, frame.device);
}

fn boolFrameFromColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    output_prefix: []const u8,
    bool_columns_value: anytype,
    comptime namesFn: anytype,
) BoolFrameError!DeviceDataFrame {
    var bool_columns = bool_columns_value;
    var bool_columns_transferred: usize = 0;
    errdefer {
        for (bool_columns[bool_columns_transferred..]) |*col| col.deinit();
    }

    const source_names = try frame.allocator.alloc([]const u8, frame.columns.len + bool_columns.len);
    defer frame.allocator.free(source_names);
    for (frame.names, 0..) |source_name, i| source_names[i] = source_name;

    var bool_names = try namesFn(frame.allocator, output_prefix);
    defer names_mod.freeOwnedNameItems(frame.allocator, bool_names[0..]);
    for (bool_names, 0..) |bool_name, i| source_names[frame.columns.len + i] = bool_name;

    const out = try appendBoolColumns(DeviceDataFrame, frame, source_names, bool_columns);
    bool_columns_transferred = bool_columns.len;
    return out;
}

fn boolSource(frame: anytype, name: []const u8) BoolFrameError!@TypeOf((frame.column(name) catch unreachable).bool) {
    const source = try frame.column(name);
    if (source.dtype() != .bool) return error.TypeMismatch;
    return source.bool;
}

pub fn rollingBoolProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceRollingOptions,
) BoolFrameError!DeviceDataFrame {
    const source = try boolSource(frame, name);
    const bool_columns = try rollingBoolProfileColumns(frame.allocator, source, options_value, frame.device, frame.rows);
    return boolFrameFromColumns(DeviceDataFrame, frame, output_prefix, bool_columns, rollingBoolProfileOutputNames);
}

pub fn expandingBoolProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceExpandingOptions,
) BoolFrameError!DeviceDataFrame {
    const source = try boolSource(frame, name);
    const bool_columns = try expandingBoolProfileColumns(frame.allocator, source, options_value, frame.device, frame.rows);
    return boolFrameFromColumns(DeviceDataFrame, frame, output_prefix, bool_columns, expandingBoolProfileOutputNames);
}
