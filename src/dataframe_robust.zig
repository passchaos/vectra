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
const DeviceRollingRobustOptions = options_mod.DeviceRollingRobustOptions;
const DeviceRobustOptions = options_mod.DeviceRobustOptions;
const castToF64 = numeric_mod.castToF64;
const validityValues = validity_mod.validityValues;

const RobustFrameError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    TypeUnsupported,
    UnsupportedType,
    InvalidCsv,
    EmptyDataFrame,
    InvalidDevice,
};

const mad_normal = 0.6744897501960817;

pub const RobustMetrics = struct {
    allocator: std.mem.Allocator,
    centered: []f64,
    mad_zscore: []f64,
    outlier: []bool,
    winsorized: []f64,
    validity: []bool,

    pub fn deinit(self: *RobustMetrics) void {
        self.allocator.free(self.centered);
        self.allocator.free(self.mad_zscore);
        self.allocator.free(self.outlier);
        self.allocator.free(self.winsorized);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub const RobustProfileColumnCount = 4;

pub fn robustProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![RobustProfileColumnCount][]const u8 {
    var names: [RobustProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "median_centered", "mad_zscore", "iqr_outlier", "winsorized" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const RollingRobustProfileColumnCount = 4;

pub fn rollingRobustProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![RollingRobustProfileColumnCount][]const u8 {
    var names: [RollingRobustProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "rolling_median_centered", "rolling_mad_zscore", "rolling_iqr_outlier", "rolling_winsorized" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const ExpandingRobustProfileColumnCount = 4;

pub fn expandingRobustProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![ExpandingRobustProfileColumnCount][]const u8 {
    var names: [ExpandingRobustProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "expanding_median_centered", "expanding_mad_zscore", "expanding_iqr_outlier", "expanding_winsorized" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn validate(values: []const f64, maybe_validity: ?[]const bool, min_periods: usize) error{ InvalidShape, LengthMismatch }!void {
    if (min_periods == 0) return error.InvalidShape;
    if (maybe_validity) |validity| {
        if (validity.len != values.len) return error.LengthMismatch;
    }
}

fn rowValid(maybe_validity: ?[]const bool, row: usize) bool {
    return if (maybe_validity) |mask| mask[row] else true;
}

fn compareFloat(lhs: f64, rhs: f64) bool {
    const lhs_nan = std.math.isNan(lhs);
    const rhs_nan = std.math.isNan(rhs);
    if (lhs_nan != rhs_nan) return !lhs_nan;
    if (lhs_nan and rhs_nan) return false;
    return lhs < rhs;
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

fn sort(values: []f64) void {
    std.sort.insertion(f64, values, {}, struct {
        fn lessThan(_: void, lhs: f64, rhs: f64) bool {
            return compareFloat(lhs, rhs);
        }
    }.lessThan);
}

fn allocMetrics(allocator: std.mem.Allocator, rows: usize) std.mem.Allocator.Error!RobustMetrics {
    const centered = try allocator.alloc(f64, rows);
    errdefer allocator.free(centered);
    const mad_zscore = try allocator.alloc(f64, rows);
    errdefer allocator.free(mad_zscore);
    const outlier = try allocator.alloc(bool, rows);
    errdefer allocator.free(outlier);
    const winsorized = try allocator.alloc(f64, rows);
    errdefer allocator.free(winsorized);
    const validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(validity);
    return .{
        .allocator = allocator,
        .centered = centered,
        .mad_zscore = mad_zscore,
        .outlier = outlier,
        .winsorized = winsorized,
        .validity = validity,
    };
}

fn fillInvalid(out: RobustMetrics, row: usize) void {
    out.centered[row] = 0;
    out.mad_zscore[row] = 0;
    out.outlier[row] = false;
    out.winsorized[row] = 0;
    out.validity[row] = false;
}

fn robustStats(sorted_values: []const f64, deviations: []f64, iqr_multiplier: f64) struct { median: f64, mad: f64, lower_fence: f64, upper_fence: f64 } {
    const median = quantileSorted(sorted_values, 0.5);
    const q1 = quantileSorted(sorted_values, 0.25);
    const q3 = quantileSorted(sorted_values, 0.75);
    const iqr = q3 - q1;
    const lower_fence = q1 - iqr_multiplier * iqr;
    const upper_fence = q3 + iqr_multiplier * iqr;
    for (sorted_values, deviations[0..sorted_values.len]) |value, *slot| slot.* = @abs(value - median);
    sort(deviations[0..sorted_values.len]);
    const mad = quantileSorted(deviations[0..sorted_values.len], 0.5);
    return .{ .median = median, .mad = mad, .lower_fence = lower_fence, .upper_fence = upper_fence };
}

fn writeRobustRow(out: RobustMetrics, row: usize, value: f64, stats: anytype) void {
    const centered = value - stats.median;
    out.centered[row] = centered;
    out.mad_zscore[row] = if (stats.mad == 0) std.math.nan(f64) else mad_normal * centered / stats.mad;
    out.outlier[row] = value < stats.lower_fence or value > stats.upper_fence;
    out.winsorized[row] = @min(@max(value, stats.lower_fence), stats.upper_fence);
    out.validity[row] = true;
}

pub fn robustProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    min_periods: usize,
    iqr_multiplier: f64,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!RobustMetrics {
    try validate(values, maybe_validity, min_periods);

    var valid_count: usize = 0;
    for (0..values.len) |row| {
        if (rowValid(maybe_validity, row)) valid_count += 1;
    }

    const valid_values = try allocator.alloc(f64, valid_count);
    defer allocator.free(valid_values);
    var write: usize = 0;
    for (values, 0..) |value, row| {
        if (!rowValid(maybe_validity, row)) continue;
        valid_values[write] = value;
        write += 1;
    }
    sort(valid_values);

    var out = try allocMetrics(allocator, values.len);
    errdefer out.deinit();

    const has_enough = valid_count >= min_periods;
    if (!has_enough or valid_count == 0) {
        for (0..values.len) |row| fillInvalid(out, row);
        return out;
    }

    const deviations = try allocator.alloc(f64, valid_count);
    defer allocator.free(deviations);
    const stats = robustStats(valid_values, deviations, iqr_multiplier);

    for (values, 0..) |value, row| {
        if (!rowValid(maybe_validity, row)) {
            fillInvalid(out, row);
            continue;
        }
        writeRobustRow(out, row, value, stats);
    }
    return out;
}

pub fn rollingRobustProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    window: usize,
    min_periods: usize,
    iqr_multiplier: f64,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!RobustMetrics {
    if (window == 0) return error.InvalidShape;
    if (min_periods == 0 or min_periods > window) return error.InvalidShape;
    if (maybe_validity) |validity| if (validity.len != values.len) return error.LengthMismatch;

    var out = try allocMetrics(allocator, values.len);
    errdefer out.deinit();
    const scratch = try allocator.alloc(f64, window);
    defer allocator.free(scratch);
    const deviations = try allocator.alloc(f64, window);
    defer allocator.free(deviations);

    for (values, 0..) |value, row| {
        const start = if (row + 1 > window) row + 1 - window else 0;
        var count: usize = 0;
        for (start..row + 1) |window_row| {
            if (!rowValid(maybe_validity, window_row)) continue;
            scratch[count] = values[window_row];
            count += 1;
        }

        if (!rowValid(maybe_validity, row) or count < min_periods) {
            fillInvalid(out, row);
            continue;
        }

        const window_values = scratch[0..count];
        sort(window_values);
        const stats = robustStats(window_values, deviations, iqr_multiplier);
        writeRobustRow(out, row, value, stats);
    }

    return out;
}

pub fn expandingRobustProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    min_periods: usize,
    iqr_multiplier: f64,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!RobustMetrics {
    try validate(values, maybe_validity, min_periods);

    var out = try allocMetrics(allocator, values.len);
    errdefer out.deinit();
    const scratch = try allocator.alloc(f64, values.len);
    defer allocator.free(scratch);
    const deviations = try allocator.alloc(f64, values.len);
    defer allocator.free(deviations);

    var valid_count: usize = 0;
    for (values, 0..) |value, row| {
        if (rowValid(maybe_validity, row)) {
            scratch[valid_count] = value;
            valid_count += 1;
        }

        if (!rowValid(maybe_validity, row) or valid_count < min_periods) {
            fillInvalid(out, row);
            continue;
        }

        const prefix_values = scratch[0..valid_count];
        sort(prefix_values);
        const stats = robustStats(prefix_values, deviations, iqr_multiplier);
        writeRobustRow(out, row, value, stats);
    }

    return out;
}

pub fn rollingRobustProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceRollingRobustOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![RollingRobustProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| rollingRobustProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| rollingRobustProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| rollingRobustProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| rollingRobustProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| rollingRobustProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| rollingRobustProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| rollingRobustProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| rollingRobustProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| rollingRobustProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| rollingRobustProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| rollingRobustProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| rollingRobustProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| rollingRobustProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn rollingRobustProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceRollingRobustOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![RollingRobustProfileColumnCount]DeviceColumn {
    const min_periods = options_value.min_periods orelse options_value.window;
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);
    var metrics = try rollingRobustProfile(allocator, values, maybe_validity, options_value.window, min_periods, options_value.iqr_multiplier);
    defer metrics.deinit();

    var columns: [RollingRobustProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.centered, metrics.validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.mad_zscore, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(bool, allocator, metrics.outlier, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.winsorized, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn expandingRobustProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceRobustOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingRobustProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| expandingRobustProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| expandingRobustProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| expandingRobustProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| expandingRobustProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| expandingRobustProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| expandingRobustProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| expandingRobustProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| expandingRobustProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| expandingRobustProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| expandingRobustProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| expandingRobustProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| expandingRobustProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| expandingRobustProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn expandingRobustProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceRobustOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingRobustProfileColumnCount]DeviceColumn {
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);
    var metrics = try expandingRobustProfile(allocator, values, maybe_validity, options_value.min_periods, options_value.iqr_multiplier);
    defer metrics.deinit();

    var columns: [ExpandingRobustProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.centered, metrics.validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.mad_zscore, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(bool, allocator, metrics.outlier, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.winsorized, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn robustProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceRobustOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![RobustProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| robustProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| robustProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| robustProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| robustProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| robustProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| robustProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| robustProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| robustProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| robustProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| robustProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| robustProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| robustProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| robustProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn robustProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceRobustOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![RobustProfileColumnCount]DeviceColumn {
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);
    var metrics = try robustProfile(allocator, values, maybe_validity, options_value.min_periods, options_value.iqr_multiplier);
    defer metrics.deinit();

    var columns: [RobustProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.centered, metrics.validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.mad_zscore, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(bool, allocator, metrics.outlier, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.winsorized, metrics.validity, device_value);
    initialized += 1;
    return columns;
}

pub fn robustProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceRobustOptions,
) RobustFrameError!DeviceDataFrame {
    const robust_value = try frame.column(name);
    var robust_columns = try robustProfileColumnsByValue(frame.allocator, robust_value.*, options_value, frame.device, frame.rows);
    var robust_columns_transferred: usize = 0;
    errdefer {
        for (robust_columns[robust_columns_transferred..]) |*col| col.deinit();
    }

    const source_names = try frame.allocator.alloc([]const u8, frame.columns.len + robust_columns.len);
    defer frame.allocator.free(source_names);
    for (frame.names, 0..) |source_name, i| source_names[i] = source_name;

    var robust_names = try robustProfileOutputNames(frame.allocator, output_prefix);
    defer names_mod.freeOwnedNameItems(frame.allocator, robust_names[0..]);
    for (robust_names, 0..) |robust_name, i| source_names[frame.columns.len + i] = robust_name;

    const DeviceColumnType = std.meta.Elem(@TypeOf(frame.columns));
    var columns = try frame.allocator.alloc(DeviceColumnType, frame.columns.len + robust_columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }
    for (frame.columns, 0..) |col, i| {
        columns[i] = try col.clone();
        initialized += 1;
    }
    for (&robust_columns) |*robust_col| {
        columns[initialized] = robust_col.*;
        initialized += 1;
        robust_columns_transferred += 1;
    }

    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, source_names, columns, frame.rows, frame.device);
}
