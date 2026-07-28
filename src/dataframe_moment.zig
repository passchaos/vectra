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

pub const MomentMetrics = struct {
    allocator: std.mem.Allocator,
    counts: []i64,
    m3_values: []f64,
    m4_values: []f64,
    skewnesses: []f64,
    kurtoses: []f64,
    validity: []bool,

    pub fn deinit(self: *MomentMetrics) void {
        self.allocator.free(self.counts);
        self.allocator.free(self.m3_values);
        self.allocator.free(self.m4_values);
        self.allocator.free(self.skewnesses);
        self.allocator.free(self.kurtoses);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub const RollingMomentProfileColumnCount = 5;

pub fn rollingMomentProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![RollingMomentProfileColumnCount][]const u8 {
    var names: [RollingMomentProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "rolling_moment_count", "rolling_m3", "rolling_m4", "rolling_skewness", "rolling_kurtosis" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const ExpandingMomentProfileColumnCount = 5;

pub fn expandingMomentProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![ExpandingMomentProfileColumnCount][]const u8 {
    var names: [ExpandingMomentProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "expanding_moment_count", "expanding_m3", "expanding_m4", "expanding_skewness", "expanding_kurtosis" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

const OnlineMoment = struct {
    count: i64 = 0,
    mean: f64 = 0,
    m2: f64 = 0,
    m3: f64 = 0,
    m4: f64 = 0,

    fn update(self: *OnlineMoment, value: f64) void {
        const n1: f64 = @floatFromInt(self.count);
        self.count += 1;
        const n: f64 = @floatFromInt(self.count);
        const delta = value - self.mean;
        const delta_n = delta / n;
        const delta_n2 = delta_n * delta_n;
        const term1 = delta * delta_n * n1;
        self.mean += delta_n;
        self.m4 += term1 * delta_n2 * (n * n - 3.0 * n + 3.0) + 6.0 * delta_n2 * self.m2 - 4.0 * delta_n * self.m3;
        self.m3 += term1 * delta_n * (n - 2.0) - 3.0 * delta_n * self.m2;
        self.m2 += term1;
    }

    fn skewness(self: OnlineMoment) f64 {
        if (self.count < 2 or self.m2 == 0) return std.math.nan(f64);
        const n: f64 = @floatFromInt(self.count);
        return std.math.sqrt(n) * self.m3 / std.math.pow(f64, self.m2, 1.5);
    }

    fn kurtosis(self: OnlineMoment) f64 {
        if (self.count < 2 or self.m2 == 0) return std.math.nan(f64);
        const n: f64 = @floatFromInt(self.count);
        return n * self.m4 / (self.m2 * self.m2) - 3.0;
    }
};

fn validate(values: []const f64, maybe_validity: ?[]const bool) error{LengthMismatch}!void {
    if (maybe_validity) |validity| {
        if (validity.len != values.len) return error.LengthMismatch;
    }
}

fn rowValid(maybe_validity: ?[]const bool, row: usize) bool {
    return if (maybe_validity) |mask| mask[row] else true;
}

fn allocMetrics(allocator: std.mem.Allocator, rows: usize) std.mem.Allocator.Error!MomentMetrics {
    const counts = try allocator.alloc(i64, rows);
    errdefer allocator.free(counts);
    const m3_values = try allocator.alloc(f64, rows);
    errdefer allocator.free(m3_values);
    const m4_values = try allocator.alloc(f64, rows);
    errdefer allocator.free(m4_values);
    const skewnesses = try allocator.alloc(f64, rows);
    errdefer allocator.free(skewnesses);
    const kurtoses = try allocator.alloc(f64, rows);
    errdefer allocator.free(kurtoses);
    const validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(validity);
    return .{ .allocator = allocator, .counts = counts, .m3_values = m3_values, .m4_values = m4_values, .skewnesses = skewnesses, .kurtoses = kurtoses, .validity = validity };
}

fn writeInvalid(out: MomentMetrics, row: usize, count: usize) void {
    out.counts[row] = @intCast(count);
    out.validity[row] = false;
    out.m3_values[row] = 0;
    out.m4_values[row] = 0;
    out.skewnesses[row] = 0;
    out.kurtoses[row] = 0;
}

fn writeCentralMoments(out: MomentMetrics, row: usize, count: usize, sum2: f64, sum3: f64, sum4: f64, min_periods: usize) void {
    out.counts[row] = @intCast(count);
    const has_enough = count >= min_periods;
    out.validity[row] = has_enough;
    if (!has_enough) {
        out.m3_values[row] = 0;
        out.m4_values[row] = 0;
        out.skewnesses[row] = 0;
        out.kurtoses[row] = 0;
        return;
    }
    const n: f64 = @floatFromInt(count);
    const variance = sum2 / n;
    const m3 = sum3 / n;
    const m4 = sum4 / n;
    out.m3_values[row] = m3;
    out.m4_values[row] = m4;
    if (count < 2 or variance == 0) {
        out.skewnesses[row] = std.math.nan(f64);
        out.kurtoses[row] = std.math.nan(f64);
    } else {
        out.skewnesses[row] = m3 / std.math.pow(f64, variance, 1.5);
        out.kurtoses[row] = m4 / (variance * variance) - 3.0;
    }
}

pub fn rollingMomentProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    window: usize,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!MomentMetrics {
    if (window == 0) return error.InvalidShape;
    if (min_periods == 0 or min_periods > window) return error.InvalidShape;
    try validate(values, maybe_validity);

    var out = try allocMetrics(allocator, values.len);
    errdefer out.deinit();

    for (0..values.len) |row| {
        const start = if (row + 1 > window) row + 1 - window else 0;
        var count: usize = 0;
        var sum: f64 = 0;
        for (start..row + 1) |window_row| {
            if (!rowValid(maybe_validity, window_row)) continue;
            sum += values[window_row];
            count += 1;
        }
        if (count < min_periods) {
            writeInvalid(out, row, count);
            continue;
        }

        const mean = sum / @as(f64, @floatFromInt(count));
        var sum2: f64 = 0;
        var sum3: f64 = 0;
        var sum4: f64 = 0;
        for (start..row + 1) |window_row| {
            if (!rowValid(maybe_validity, window_row)) continue;
            const centered = values[window_row] - mean;
            const centered2 = centered * centered;
            sum2 += centered2;
            sum3 += centered2 * centered;
            sum4 += centered2 * centered2;
        }
        writeCentralMoments(out, row, count, sum2, sum3, sum4, min_periods);
    }

    return out;
}

pub fn expandingMomentProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!MomentMetrics {
    if (min_periods == 0) return error.InvalidShape;
    try validate(values, maybe_validity);

    var out = try allocMetrics(allocator, values.len);
    errdefer out.deinit();

    var profile: OnlineMoment = .{};
    for (values, 0..) |value, row| {
        if (rowValid(maybe_validity, row)) profile.update(value);

        out.counts[row] = profile.count;
        const has_enough = profile.count >= @as(i64, @intCast(min_periods));
        out.validity[row] = has_enough;
        if (has_enough) {
            const n: f64 = @floatFromInt(profile.count);
            out.m3_values[row] = profile.m3 / n;
            out.m4_values[row] = profile.m4 / n;
            out.skewnesses[row] = profile.skewness();
            out.kurtoses[row] = profile.kurtosis();
        } else {
            out.m3_values[row] = 0;
            out.m4_values[row] = 0;
            out.skewnesses[row] = 0;
            out.kurtoses[row] = 0;
        }
    }

    return out;
}

pub fn rollingMomentProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![RollingMomentProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| rollingMomentProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| rollingMomentProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| rollingMomentProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| rollingMomentProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| rollingMomentProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| rollingMomentProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| rollingMomentProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| rollingMomentProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| rollingMomentProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| rollingMomentProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| rollingMomentProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| rollingMomentProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| rollingMomentProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn rollingMomentProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![RollingMomentProfileColumnCount]DeviceColumn {
    const min_periods = options_value.min_periods orelse options_value.window;
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);
    var metrics = try rollingMomentProfile(allocator, values, maybe_validity, options_value.window, min_periods);
    defer metrics.deinit();

    var columns: [RollingMomentProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.m3_values, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.m4_values, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.skewnesses, metrics.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.kurtoses, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn expandingMomentProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingMomentProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| expandingMomentProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| expandingMomentProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| expandingMomentProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| expandingMomentProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| expandingMomentProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| expandingMomentProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| expandingMomentProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| expandingMomentProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| expandingMomentProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| expandingMomentProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| expandingMomentProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| expandingMomentProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| expandingMomentProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn expandingMomentProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingMomentProfileColumnCount]DeviceColumn {
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);
    var metrics = try expandingMomentProfile(allocator, values, maybe_validity, options_value.min_periods);
    defer metrics.deinit();

    var columns: [ExpandingMomentProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.m3_values, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.m4_values, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.skewnesses, metrics.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.kurtoses, metrics.validity, device_value);
    initialized += 1;
    return columns;
}

const MomentFrameError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    TypeUnsupported,
    UnsupportedType,
    InvalidCsv,
    EmptyDataFrame,
    InvalidDevice,
};

fn appendMomentColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    source_names: []const []const u8,
    moment_columns: anytype,
) MomentFrameError!DeviceDataFrame {
    const DeviceColumnType = std.meta.Elem(@TypeOf(frame.columns));
    var columns = try frame.allocator.alloc(DeviceColumnType, frame.columns.len + moment_columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }
    for (frame.columns, 0..) |col, i| {
        columns[i] = try col.clone();
        initialized += 1;
    }
    for (&moment_columns) |*moment_col| {
        columns[initialized] = moment_col.*;
        initialized += 1;
    }
    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, source_names, columns, frame.rows, frame.device);
}

fn momentFrameFromColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    output_prefix: []const u8,
    moment_columns_value: anytype,
    comptime namesFn: anytype,
) MomentFrameError!DeviceDataFrame {
    var moment_columns = moment_columns_value;
    var moment_columns_transferred: usize = 0;
    errdefer {
        for (moment_columns[moment_columns_transferred..]) |*col| col.deinit();
    }

    const source_names = try frame.allocator.alloc([]const u8, frame.columns.len + moment_columns.len);
    defer frame.allocator.free(source_names);
    for (frame.names, 0..) |source_name, i| source_names[i] = source_name;

    var moment_names = try namesFn(frame.allocator, output_prefix);
    defer names_mod.freeOwnedNameItems(frame.allocator, moment_names[0..]);
    for (moment_names, 0..) |moment_name, i| source_names[frame.columns.len + i] = moment_name;

    const out = try appendMomentColumns(DeviceDataFrame, frame, source_names, moment_columns);
    moment_columns_transferred = moment_columns.len;
    return out;
}

pub fn rollingMomentProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceRollingOptions,
) MomentFrameError!DeviceDataFrame {
    const rolling_value = try frame.column(name);
    const rolling_columns = try rollingMomentProfileColumnsByValue(frame.allocator, rolling_value.*, options_value, frame.device, frame.rows);
    return momentFrameFromColumns(DeviceDataFrame, frame, output_prefix, rolling_columns, rollingMomentProfileOutputNames);
}

pub fn expandingMomentProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceExpandingOptions,
) MomentFrameError!DeviceDataFrame {
    const expanding_value = try frame.column(name);
    const expanding_columns = try expandingMomentProfileColumnsByValue(frame.allocator, expanding_value.*, options_value, frame.device, frame.rows);
    return momentFrameFromColumns(DeviceDataFrame, frame, output_prefix, expanding_columns, expandingMomentProfileOutputNames);
}
