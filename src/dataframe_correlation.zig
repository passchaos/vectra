const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_device_column_mod = @import("dataframe_device_column.zig");
const numeric_mod = @import("dataframe_numeric.zig");
const options_mod = @import("dataframe_options.zig");
const validity_mod = @import("dataframe_validity.zig");

const DeviceColumn = dataframe_device_column_mod.DeviceColumn;
const DeviceTypedColumn = dataframe_device_column_mod.DeviceTypedColumn;
const DeviceRollingCorrelationOptions = options_mod.DeviceRollingCorrelationOptions;
const DeviceExpandingOptions = options_mod.DeviceExpandingOptions;
const castToF64 = numeric_mod.castToF64;
const validityValues = validity_mod.validityValues;

pub const CorrelationMetrics = struct {
    allocator: std.mem.Allocator,
    pair_counts: []i64,
    covariances: []f64,
    correlations: []f64,
    betas: []f64,
    validity: []bool,

    pub fn deinit(self: *CorrelationMetrics) void {
        self.allocator.free(self.pair_counts);
        self.allocator.free(self.covariances);
        self.allocator.free(self.correlations);
        self.allocator.free(self.betas);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub const RollingCorrelationProfileColumnCount = 4;

pub fn rollingCorrelationProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![RollingCorrelationProfileColumnCount][]const u8 {
    var names: [RollingCorrelationProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "rolling_pair_count", "rolling_covariance", "rolling_correlation", "rolling_beta" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const ExpandingCorrelationProfileColumnCount = 4;

pub fn expandingCorrelationProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![ExpandingCorrelationProfileColumnCount][]const u8 {
    var names: [ExpandingCorrelationProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "expanding_pair_count", "expanding_covariance", "expanding_correlation", "expanding_beta" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn validatePairLengths(
    xs: []const f64,
    ys: []const f64,
    maybe_x_validity: ?[]const bool,
    maybe_y_validity: ?[]const bool,
) error{LengthMismatch}!void {
    if (xs.len != ys.len) return error.LengthMismatch;
    if (maybe_x_validity) |validity| {
        if (validity.len != xs.len) return error.LengthMismatch;
    }
    if (maybe_y_validity) |validity| {
        if (validity.len != ys.len) return error.LengthMismatch;
    }
}

fn rowValid(maybe_x_validity: ?[]const bool, maybe_y_validity: ?[]const bool, row: usize) bool {
    return (if (maybe_x_validity) |mask| mask[row] else true) and (if (maybe_y_validity) |mask| mask[row] else true);
}

fn allocMetrics(allocator: std.mem.Allocator, rows: usize) std.mem.Allocator.Error!CorrelationMetrics {
    const pair_counts = try allocator.alloc(i64, rows);
    errdefer allocator.free(pair_counts);
    const covariances = try allocator.alloc(f64, rows);
    errdefer allocator.free(covariances);
    const correlations = try allocator.alloc(f64, rows);
    errdefer allocator.free(correlations);
    const betas = try allocator.alloc(f64, rows);
    errdefer allocator.free(betas);
    const validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(validity);
    return .{
        .allocator = allocator,
        .pair_counts = pair_counts,
        .covariances = covariances,
        .correlations = correlations,
        .betas = betas,
        .validity = validity,
    };
}

fn writeStats(
    row: usize,
    min_periods: usize,
    count: usize,
    sum_x: f64,
    sum_y: f64,
    sum_xx: f64,
    sum_yy: f64,
    sum_xy: f64,
    out: CorrelationMetrics,
) void {
    out.pair_counts[row] = @intCast(count);
    const has_enough = count >= min_periods;
    out.validity[row] = has_enough;
    if (has_enough) {
        const n: f64 = @floatFromInt(count);
        const mean_x = sum_x / n;
        const mean_y = sum_y / n;
        const cov = sum_xy / n - mean_x * mean_y;
        const var_x_raw = sum_xx / n - mean_x * mean_x;
        const var_y_raw = sum_yy / n - mean_y * mean_y;
        const var_x = if (var_x_raw < 0) 0 else var_x_raw;
        const var_y = if (var_y_raw < 0) 0 else var_y_raw;
        out.covariances[row] = cov;
        out.correlations[row] = if (var_x == 0 or var_y == 0) std.math.nan(f64) else cov / std.math.sqrt(var_x * var_y);
        out.betas[row] = if (var_x == 0) std.math.nan(f64) else cov / var_x;
    } else {
        out.covariances[row] = 0;
        out.correlations[row] = 0;
        out.betas[row] = 0;
    }
}

pub fn rollingCorrelationProfile(
    allocator: std.mem.Allocator,
    xs: []const f64,
    ys: []const f64,
    maybe_x_validity: ?[]const bool,
    maybe_y_validity: ?[]const bool,
    window: usize,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!CorrelationMetrics {
    if (window == 0) return error.InvalidShape;
    if (min_periods == 0 or min_periods > window) return error.InvalidShape;
    try validatePairLengths(xs, ys, maybe_x_validity, maybe_y_validity);

    var out = try allocMetrics(allocator, xs.len);
    errdefer out.deinit();

    // Recompute each trailing window in host memory, mirroring the dataframe
    // rolling profile APIs while retaining a stable seam for future device-side
    // rolling covariance/correlation kernels.
    for (0..xs.len) |row| {
        const start = if (row + 1 > window) row + 1 - window else 0;
        var count: usize = 0;
        var sum_x: f64 = 0;
        var sum_y: f64 = 0;
        var sum_xx: f64 = 0;
        var sum_yy: f64 = 0;
        var sum_xy: f64 = 0;
        for (start..row + 1) |window_row| {
            if (!rowValid(maybe_x_validity, maybe_y_validity, window_row)) continue;
            const x = xs[window_row];
            const y = ys[window_row];
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_yy += y * y;
            sum_xy += x * y;
            count += 1;
        }
        writeStats(row, min_periods, count, sum_x, sum_y, sum_xx, sum_yy, sum_xy, out);
    }

    return out;
}

pub fn expandingCorrelationProfile(
    allocator: std.mem.Allocator,
    xs: []const f64,
    ys: []const f64,
    maybe_x_validity: ?[]const bool,
    maybe_y_validity: ?[]const bool,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!CorrelationMetrics {
    if (min_periods == 0) return error.InvalidShape;
    try validatePairLengths(xs, ys, maybe_x_validity, maybe_y_validity);

    var out = try allocMetrics(allocator, xs.len);
    errdefer out.deinit();

    var count: usize = 0;
    var sum_x: f64 = 0;
    var sum_y: f64 = 0;
    var sum_xx: f64 = 0;
    var sum_yy: f64 = 0;
    var sum_xy: f64 = 0;
    for (xs, ys, 0..) |x, y, row| {
        if (rowValid(maybe_x_validity, maybe_y_validity, row)) {
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_yy += y * y;
            sum_xy += x * y;
            count += 1;
        }
        writeStats(row, min_periods, count, sum_x, sum_y, sum_xx, sum_yy, sum_xy, out);
    }

    return out;
}

pub fn rollingCorrelationProfileColumnsByValue(
    allocator: std.mem.Allocator,
    x: DeviceColumn,
    y: DeviceColumn,
    options_value: DeviceRollingCorrelationOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, InvalidDevice })![RollingCorrelationProfileColumnCount]DeviceColumn {
    if (x.len() != rows or y.len() != rows) return error.LengthMismatch;
    if (x.dtype() != y.dtype()) return error.TypeMismatch;
    return switch (x) {
        .i8 => |typed| rollingCorrelationProfileColumnsTyped(i8, allocator, typed, y.i8, options_value, device_value),
        .i16 => |typed| rollingCorrelationProfileColumnsTyped(i16, allocator, typed, y.i16, options_value, device_value),
        .i32 => |typed| rollingCorrelationProfileColumnsTyped(i32, allocator, typed, y.i32, options_value, device_value),
        .i64 => |typed| rollingCorrelationProfileColumnsTyped(i64, allocator, typed, y.i64, options_value, device_value),
        .u8 => |typed| rollingCorrelationProfileColumnsTyped(u8, allocator, typed, y.u8, options_value, device_value),
        .u16 => |typed| rollingCorrelationProfileColumnsTyped(u16, allocator, typed, y.u16, options_value, device_value),
        .u32 => |typed| rollingCorrelationProfileColumnsTyped(u32, allocator, typed, y.u32, options_value, device_value),
        .u64 => |typed| rollingCorrelationProfileColumnsTyped(u64, allocator, typed, y.u64, options_value, device_value),
        .usize => |typed| rollingCorrelationProfileColumnsTyped(usize, allocator, typed, y.usize, options_value, device_value),
        .isize => |typed| rollingCorrelationProfileColumnsTyped(isize, allocator, typed, y.isize, options_value, device_value),
        .f16 => |typed| rollingCorrelationProfileColumnsTyped(f16, allocator, typed, y.f16, options_value, device_value),
        .f32 => |typed| rollingCorrelationProfileColumnsTyped(f32, allocator, typed, y.f32, options_value, device_value),
        .f64 => |typed| rollingCorrelationProfileColumnsTyped(f64, allocator, typed, y.f64, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn rollingCorrelationProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    x_column: DeviceTypedColumn(T),
    y_column: DeviceTypedColumn(T),
    options_value: DeviceRollingCorrelationOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, InvalidDevice })![RollingCorrelationProfileColumnCount]DeviceColumn {
    const min_periods = options_value.min_periods orelse options_value.window;
    if (x_column.len() != y_column.len()) return error.LengthMismatch;
    if (!x_column.device().sameDevice(y_column.device())) return error.InvalidDevice;

    const xs_typed = try x_column.values.toOwnedSlice(allocator);
    defer allocator.free(xs_typed);
    const ys_typed = try y_column.values.toOwnedSlice(allocator);
    defer allocator.free(ys_typed);
    const maybe_x_validity = try validityValues(x_column, allocator);
    defer if (maybe_x_validity) |validity| allocator.free(validity);
    const maybe_y_validity = try validityValues(y_column, allocator);
    defer if (maybe_y_validity) |validity| allocator.free(validity);

    const rows = xs_typed.len;
    const xs = try allocator.alloc(f64, rows);
    defer allocator.free(xs);
    const ys = try allocator.alloc(f64, rows);
    defer allocator.free(ys);
    for (xs_typed, ys_typed, 0..) |x_value, y_value, row| {
        xs[row] = castToF64(T, x_value);
        ys[row] = castToF64(T, y_value);
    }

    var metrics = try rollingCorrelationProfile(
        allocator,
        xs,
        ys,
        maybe_x_validity,
        maybe_y_validity,
        options_value.window,
        min_periods,
    );
    defer metrics.deinit();

    var columns: [RollingCorrelationProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.pair_counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.covariances, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.correlations, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.betas, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn expandingCorrelationProfileColumnsByValue(
    allocator: std.mem.Allocator,
    x: DeviceColumn,
    y: DeviceColumn,
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, InvalidDevice })![ExpandingCorrelationProfileColumnCount]DeviceColumn {
    if (x.len() != rows or y.len() != rows) return error.LengthMismatch;
    if (x.dtype() != y.dtype()) return error.TypeMismatch;
    return switch (x) {
        .i8 => |typed| expandingCorrelationProfileColumnsTyped(i8, allocator, typed, y.i8, options_value, device_value),
        .i16 => |typed| expandingCorrelationProfileColumnsTyped(i16, allocator, typed, y.i16, options_value, device_value),
        .i32 => |typed| expandingCorrelationProfileColumnsTyped(i32, allocator, typed, y.i32, options_value, device_value),
        .i64 => |typed| expandingCorrelationProfileColumnsTyped(i64, allocator, typed, y.i64, options_value, device_value),
        .u8 => |typed| expandingCorrelationProfileColumnsTyped(u8, allocator, typed, y.u8, options_value, device_value),
        .u16 => |typed| expandingCorrelationProfileColumnsTyped(u16, allocator, typed, y.u16, options_value, device_value),
        .u32 => |typed| expandingCorrelationProfileColumnsTyped(u32, allocator, typed, y.u32, options_value, device_value),
        .u64 => |typed| expandingCorrelationProfileColumnsTyped(u64, allocator, typed, y.u64, options_value, device_value),
        .usize => |typed| expandingCorrelationProfileColumnsTyped(usize, allocator, typed, y.usize, options_value, device_value),
        .isize => |typed| expandingCorrelationProfileColumnsTyped(isize, allocator, typed, y.isize, options_value, device_value),
        .f16 => |typed| expandingCorrelationProfileColumnsTyped(f16, allocator, typed, y.f16, options_value, device_value),
        .f32 => |typed| expandingCorrelationProfileColumnsTyped(f32, allocator, typed, y.f32, options_value, device_value),
        .f64 => |typed| expandingCorrelationProfileColumnsTyped(f64, allocator, typed, y.f64, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn expandingCorrelationProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    x_column: DeviceTypedColumn(T),
    y_column: DeviceTypedColumn(T),
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, InvalidDevice })![ExpandingCorrelationProfileColumnCount]DeviceColumn {
    if (x_column.len() != y_column.len()) return error.LengthMismatch;
    if (!x_column.device().sameDevice(y_column.device())) return error.InvalidDevice;

    const xs_typed = try x_column.values.toOwnedSlice(allocator);
    defer allocator.free(xs_typed);
    const ys_typed = try y_column.values.toOwnedSlice(allocator);
    defer allocator.free(ys_typed);
    const maybe_x_validity = try validityValues(x_column, allocator);
    defer if (maybe_x_validity) |validity| allocator.free(validity);
    const maybe_y_validity = try validityValues(y_column, allocator);
    defer if (maybe_y_validity) |validity| allocator.free(validity);

    const rows = xs_typed.len;
    const xs = try allocator.alloc(f64, rows);
    defer allocator.free(xs);
    const ys = try allocator.alloc(f64, rows);
    defer allocator.free(ys);
    for (xs_typed, ys_typed, 0..) |x_value, y_value, row| {
        xs[row] = castToF64(T, x_value);
        ys[row] = castToF64(T, y_value);
    }

    var metrics = try expandingCorrelationProfile(
        allocator,
        xs,
        ys,
        maybe_x_validity,
        maybe_y_validity,
        options_value.min_periods,
    );
    defer metrics.deinit();

    var columns: [ExpandingCorrelationProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.pair_counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.covariances, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.correlations, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.betas, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
