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
const DeviceLinearFitOptions = options_mod.DeviceLinearFitOptions;
const DeviceExpandingOptions = options_mod.DeviceExpandingOptions;
const DeviceRollingCorrelationOptions = options_mod.DeviceRollingCorrelationOptions;
const castToF64 = numeric_mod.castToF64;
const validityValues = validity_mod.validityValues;

pub const LinearFitMetrics = struct {
    allocator: std.mem.Allocator,
    fitted: []f64,
    residuals: []f64,
    residual_z: []f64,
    slopes: []f64,
    validity: []bool,

    pub fn deinit(self: *LinearFitMetrics) void {
        self.allocator.free(self.fitted);
        self.allocator.free(self.residuals);
        self.allocator.free(self.residual_z);
        self.allocator.free(self.slopes);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

const FitParams = struct {
    has_fit: bool,
    slope: f64,
    intercept: f64,
};

pub const WindowLinearFitMetrics = struct {
    allocator: std.mem.Allocator,
    pair_counts: []i64,
    slopes: []f64,
    intercepts: []f64,
    fitted: []f64,
    residuals: []f64,
    residual_z: []f64,
    fit_validity: []bool,
    row_validity: []bool,

    pub fn deinit(self: *WindowLinearFitMetrics) void {
        self.allocator.free(self.pair_counts);
        self.allocator.free(self.slopes);
        self.allocator.free(self.intercepts);
        self.allocator.free(self.fitted);
        self.allocator.free(self.residuals);
        self.allocator.free(self.residual_z);
        self.allocator.free(self.fit_validity);
        self.allocator.free(self.row_validity);
        self.* = undefined;
    }
};

pub const LinearFitProfileColumnCount = 4;

pub fn linearFitProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![LinearFitProfileColumnCount][]const u8 {
    var names: [LinearFitProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "fitted", "residual", "residual_zscore", "slope" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const ExpandingLinearFitProfileColumnCount = 6;

pub fn expandingLinearFitProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![ExpandingLinearFitProfileColumnCount][]const u8 {
    var names: [ExpandingLinearFitProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "expanding_pair_count", "expanding_slope", "expanding_intercept", "expanding_fitted", "expanding_residual", "expanding_residual_zscore" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const RollingLinearFitProfileColumnCount = 6;

pub fn rollingLinearFitProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![RollingLinearFitProfileColumnCount][]const u8 {
    var names: [RollingLinearFitProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "rolling_pair_count", "rolling_slope", "rolling_intercept", "rolling_fitted", "rolling_residual", "rolling_residual_zscore" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn validatePairLengths(xs: []const f64, ys: []const f64, maybe_x_validity: ?[]const bool, maybe_y_validity: ?[]const bool) error{LengthMismatch}!void {
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

fn fitFromSums(count: usize, sum_x: f64, sum_y: f64, sum_xx: f64, sum_xy: f64, min_periods: usize) FitParams {
    const has_fit = count >= min_periods;
    if (!has_fit) return .{ .has_fit = false, .slope = std.math.nan(f64), .intercept = std.math.nan(f64) };
    const n: f64 = @floatFromInt(count);
    const denom = n * sum_xx - sum_x * sum_x;
    const slope = if (denom == 0) std.math.nan(f64) else (n * sum_xy - sum_x * sum_y) / denom;
    const intercept = if (std.math.isNan(slope)) std.math.nan(f64) else (sum_y - slope * sum_x) / n;
    return .{ .has_fit = true, .slope = slope, .intercept = intercept };
}

fn residualStd(xs: []const f64, ys: []const f64, maybe_x_validity: ?[]const bool, maybe_y_validity: ?[]const bool, start: usize, end: usize, count: usize, slope: f64, intercept: f64) f64 {
    if (count == 0 or std.math.isNan(slope)) return std.math.nan(f64);
    var residual_sum_sq: f64 = 0;
    for (start..end) |row| {
        if (!rowValid(maybe_x_validity, maybe_y_validity, row)) continue;
        const fit = intercept + slope * xs[row];
        const residual = ys[row] - fit;
        residual_sum_sq += residual * residual;
    }
    return std.math.sqrt(residual_sum_sq / @as(f64, @floatFromInt(count)));
}

fn allocWindowMetrics(allocator: std.mem.Allocator, rows: usize) std.mem.Allocator.Error!WindowLinearFitMetrics {
    const pair_counts = try allocator.alloc(i64, rows);
    errdefer allocator.free(pair_counts);
    const slopes = try allocator.alloc(f64, rows);
    errdefer allocator.free(slopes);
    const intercepts = try allocator.alloc(f64, rows);
    errdefer allocator.free(intercepts);
    const fitted = try allocator.alloc(f64, rows);
    errdefer allocator.free(fitted);
    const residuals = try allocator.alloc(f64, rows);
    errdefer allocator.free(residuals);
    const residual_z = try allocator.alloc(f64, rows);
    errdefer allocator.free(residual_z);
    const fit_validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(fit_validity);
    const row_validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(row_validity);
    return .{
        .allocator = allocator,
        .pair_counts = pair_counts,
        .slopes = slopes,
        .intercepts = intercepts,
        .fitted = fitted,
        .residuals = residuals,
        .residual_z = residual_z,
        .fit_validity = fit_validity,
        .row_validity = row_validity,
    };
}

fn writeWindowRow(row: usize, x: f64, y: f64, count: usize, fit: FitParams, stddev: f64, current_valid: bool, out: WindowLinearFitMetrics) void {
    out.pair_counts[row] = @intCast(count);
    out.fit_validity[row] = fit.has_fit;
    if (!fit.has_fit) {
        out.slopes[row] = 0;
        out.intercepts[row] = 0;
        out.fitted[row] = 0;
        out.residuals[row] = 0;
        out.residual_z[row] = 0;
        out.row_validity[row] = false;
        return;
    }

    out.slopes[row] = fit.slope;
    out.intercepts[row] = fit.intercept;
    out.row_validity[row] = current_valid;
    if (current_valid) {
        const fitted = fit.intercept + fit.slope * x;
        const residual = y - fitted;
        out.fitted[row] = fitted;
        out.residuals[row] = residual;
        out.residual_z[row] = if (stddev == 0 or std.math.isNan(stddev)) std.math.nan(f64) else residual / stddev;
    } else {
        out.fitted[row] = 0;
        out.residuals[row] = 0;
        out.residual_z[row] = 0;
    }
}

pub fn linearFitProfile(
    allocator: std.mem.Allocator,
    xs: []const f64,
    ys: []const f64,
    maybe_x_validity: ?[]const bool,
    maybe_y_validity: ?[]const bool,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!LinearFitMetrics {
    if (min_periods == 0) return error.InvalidShape;
    try validatePairLengths(xs, ys, maybe_x_validity, maybe_y_validity);

    var count: usize = 0;
    var sum_x: f64 = 0;
    var sum_y: f64 = 0;
    var sum_xx: f64 = 0;
    var sum_xy: f64 = 0;
    for (xs, ys, 0..) |x, y, row| {
        if (!rowValid(maybe_x_validity, maybe_y_validity, row)) continue;
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
        count += 1;
    }

    const rows = xs.len;
    const fitted = try allocator.alloc(f64, rows);
    errdefer allocator.free(fitted);
    const residuals = try allocator.alloc(f64, rows);
    errdefer allocator.free(residuals);
    const residual_z = try allocator.alloc(f64, rows);
    errdefer allocator.free(residual_z);
    const slopes = try allocator.alloc(f64, rows);
    errdefer allocator.free(slopes);
    const validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(validity);

    const fit = fitFromSums(count, sum_x, sum_y, sum_xx, sum_xy, min_periods);
    const stddev = residualStd(xs, ys, maybe_x_validity, maybe_y_validity, 0, rows, count, fit.slope, fit.intercept);

    for (xs, ys, 0..) |x, y, row| {
        const valid = rowValid(maybe_x_validity, maybe_y_validity, row) and fit.has_fit;
        validity[row] = valid;
        if (valid) {
            const fitted_value = fit.intercept + fit.slope * x;
            const residual = y - fitted_value;
            fitted[row] = fitted_value;
            residuals[row] = residual;
            residual_z[row] = if (stddev == 0 or std.math.isNan(stddev)) std.math.nan(f64) else residual / stddev;
            slopes[row] = fit.slope;
        } else {
            fitted[row] = 0;
            residuals[row] = 0;
            residual_z[row] = 0;
            slopes[row] = 0;
        }
    }

    return .{
        .allocator = allocator,
        .fitted = fitted,
        .residuals = residuals,
        .residual_z = residual_z,
        .slopes = slopes,
        .validity = validity,
    };
}

pub fn expandingLinearFitProfile(
    allocator: std.mem.Allocator,
    xs: []const f64,
    ys: []const f64,
    maybe_x_validity: ?[]const bool,
    maybe_y_validity: ?[]const bool,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!WindowLinearFitMetrics {
    if (min_periods == 0) return error.InvalidShape;
    try validatePairLengths(xs, ys, maybe_x_validity, maybe_y_validity);

    var out = try allocWindowMetrics(allocator, xs.len);
    errdefer out.deinit();

    var count: usize = 0;
    var sum_x: f64 = 0;
    var sum_y: f64 = 0;
    var sum_xx: f64 = 0;
    var sum_xy: f64 = 0;
    for (xs, ys, 0..) |x, y, row| {
        if (rowValid(maybe_x_validity, maybe_y_validity, row)) {
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
            count += 1;
        }

        const fit = fitFromSums(count, sum_x, sum_y, sum_xx, sum_xy, min_periods);
        const stddev = residualStd(xs, ys, maybe_x_validity, maybe_y_validity, 0, row + 1, count, fit.slope, fit.intercept);
        writeWindowRow(row, x, y, count, fit, stddev, rowValid(maybe_x_validity, maybe_y_validity, row), out);
    }

    return out;
}

pub fn rollingLinearFitProfile(
    allocator: std.mem.Allocator,
    xs: []const f64,
    ys: []const f64,
    maybe_x_validity: ?[]const bool,
    maybe_y_validity: ?[]const bool,
    window: usize,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!WindowLinearFitMetrics {
    if (window == 0) return error.InvalidShape;
    if (min_periods == 0 or min_periods > window) return error.InvalidShape;
    try validatePairLengths(xs, ys, maybe_x_validity, maybe_y_validity);

    var out = try allocWindowMetrics(allocator, xs.len);
    errdefer out.deinit();

    // Each row receives the ordinary least-squares line fitted over its trailing
    // valid-pair window. Recomputing windows on the host matches the current
    // rolling correlation implementation and keeps a public seam for future
    // device-side rolling regression kernels.
    for (xs, ys, 0..) |x_current, y_current, row| {
        const start = if (row + 1 > window) row + 1 - window else 0;
        var count: usize = 0;
        var sum_x: f64 = 0;
        var sum_y: f64 = 0;
        var sum_xx: f64 = 0;
        var sum_xy: f64 = 0;
        for (start..row + 1) |window_row| {
            if (!rowValid(maybe_x_validity, maybe_y_validity, window_row)) continue;
            const x = xs[window_row];
            const y = ys[window_row];
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
            count += 1;
        }

        const fit = fitFromSums(count, sum_x, sum_y, sum_xx, sum_xy, min_periods);
        const stddev = residualStd(xs, ys, maybe_x_validity, maybe_y_validity, start, row + 1, count, fit.slope, fit.intercept);
        writeWindowRow(row, x_current, y_current, count, fit, stddev, rowValid(maybe_x_validity, maybe_y_validity, row), out);
    }

    return out;
}

pub fn linearFitProfileColumnsByValue(
    allocator: std.mem.Allocator,
    x: DeviceColumn,
    y: DeviceColumn,
    options_value: DeviceLinearFitOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, InvalidDevice })![LinearFitProfileColumnCount]DeviceColumn {
    if (x.len() != rows or y.len() != rows) return error.LengthMismatch;
    if (x.dtype() != y.dtype()) return error.TypeMismatch;
    return switch (x) {
        .i8 => |typed| linearFitProfileColumnsTyped(i8, allocator, typed, y.i8, options_value, device_value),
        .i16 => |typed| linearFitProfileColumnsTyped(i16, allocator, typed, y.i16, options_value, device_value),
        .i32 => |typed| linearFitProfileColumnsTyped(i32, allocator, typed, y.i32, options_value, device_value),
        .i64 => |typed| linearFitProfileColumnsTyped(i64, allocator, typed, y.i64, options_value, device_value),
        .u8 => |typed| linearFitProfileColumnsTyped(u8, allocator, typed, y.u8, options_value, device_value),
        .u16 => |typed| linearFitProfileColumnsTyped(u16, allocator, typed, y.u16, options_value, device_value),
        .u32 => |typed| linearFitProfileColumnsTyped(u32, allocator, typed, y.u32, options_value, device_value),
        .u64 => |typed| linearFitProfileColumnsTyped(u64, allocator, typed, y.u64, options_value, device_value),
        .usize => |typed| linearFitProfileColumnsTyped(usize, allocator, typed, y.usize, options_value, device_value),
        .isize => |typed| linearFitProfileColumnsTyped(isize, allocator, typed, y.isize, options_value, device_value),
        .f16 => |typed| linearFitProfileColumnsTyped(f16, allocator, typed, y.f16, options_value, device_value),
        .f32 => |typed| linearFitProfileColumnsTyped(f32, allocator, typed, y.f32, options_value, device_value),
        .f64 => |typed| linearFitProfileColumnsTyped(f64, allocator, typed, y.f64, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn linearFitProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    x_column: DeviceTypedColumn(T),
    y_column: DeviceTypedColumn(T),
    options_value: DeviceLinearFitOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, InvalidDevice })![LinearFitProfileColumnCount]DeviceColumn {
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

    var metrics = try linearFitProfile(allocator, xs, ys, maybe_x_validity, maybe_y_validity, options_value.min_periods);
    defer metrics.deinit();

    var columns: [LinearFitProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.fitted, metrics.validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.residuals, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.residual_z, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.slopes, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn expandingLinearFitProfileColumnsByValue(
    allocator: std.mem.Allocator,
    x: DeviceColumn,
    y: DeviceColumn,
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, InvalidDevice })![ExpandingLinearFitProfileColumnCount]DeviceColumn {
    if (x.len() != rows or y.len() != rows) return error.LengthMismatch;
    if (x.dtype() != y.dtype()) return error.TypeMismatch;
    return switch (x) {
        .i8 => |typed| expandingLinearFitProfileColumnsTyped(i8, allocator, typed, y.i8, options_value, device_value),
        .i16 => |typed| expandingLinearFitProfileColumnsTyped(i16, allocator, typed, y.i16, options_value, device_value),
        .i32 => |typed| expandingLinearFitProfileColumnsTyped(i32, allocator, typed, y.i32, options_value, device_value),
        .i64 => |typed| expandingLinearFitProfileColumnsTyped(i64, allocator, typed, y.i64, options_value, device_value),
        .u8 => |typed| expandingLinearFitProfileColumnsTyped(u8, allocator, typed, y.u8, options_value, device_value),
        .u16 => |typed| expandingLinearFitProfileColumnsTyped(u16, allocator, typed, y.u16, options_value, device_value),
        .u32 => |typed| expandingLinearFitProfileColumnsTyped(u32, allocator, typed, y.u32, options_value, device_value),
        .u64 => |typed| expandingLinearFitProfileColumnsTyped(u64, allocator, typed, y.u64, options_value, device_value),
        .usize => |typed| expandingLinearFitProfileColumnsTyped(usize, allocator, typed, y.usize, options_value, device_value),
        .isize => |typed| expandingLinearFitProfileColumnsTyped(isize, allocator, typed, y.isize, options_value, device_value),
        .f16 => |typed| expandingLinearFitProfileColumnsTyped(f16, allocator, typed, y.f16, options_value, device_value),
        .f32 => |typed| expandingLinearFitProfileColumnsTyped(f32, allocator, typed, y.f32, options_value, device_value),
        .f64 => |typed| expandingLinearFitProfileColumnsTyped(f64, allocator, typed, y.f64, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn expandingLinearFitProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    x_column: DeviceTypedColumn(T),
    y_column: DeviceTypedColumn(T),
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, InvalidDevice })![ExpandingLinearFitProfileColumnCount]DeviceColumn {
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

    var metrics = try expandingLinearFitProfile(allocator, xs, ys, maybe_x_validity, maybe_y_validity, options_value.min_periods);
    defer metrics.deinit();

    var columns: [ExpandingLinearFitProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.pair_counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.slopes, metrics.fit_validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.intercepts, metrics.fit_validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.fitted, metrics.row_validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.residuals, metrics.row_validity, device_value);
    initialized += 1;
    columns[5] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.residual_z, metrics.row_validity, device_value);
    initialized += 1;
    return columns;
}
pub fn rollingLinearFitProfileColumnsByValue(
    allocator: std.mem.Allocator,
    x: DeviceColumn,
    y: DeviceColumn,
    options_value: DeviceRollingCorrelationOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, InvalidDevice })![RollingLinearFitProfileColumnCount]DeviceColumn {
    if (x.len() != rows or y.len() != rows) return error.LengthMismatch;
    if (x.dtype() != y.dtype()) return error.TypeMismatch;
    return switch (x) {
        .i8 => |typed| rollingLinearFitProfileColumnsTyped(i8, allocator, typed, y.i8, options_value, device_value),
        .i16 => |typed| rollingLinearFitProfileColumnsTyped(i16, allocator, typed, y.i16, options_value, device_value),
        .i32 => |typed| rollingLinearFitProfileColumnsTyped(i32, allocator, typed, y.i32, options_value, device_value),
        .i64 => |typed| rollingLinearFitProfileColumnsTyped(i64, allocator, typed, y.i64, options_value, device_value),
        .u8 => |typed| rollingLinearFitProfileColumnsTyped(u8, allocator, typed, y.u8, options_value, device_value),
        .u16 => |typed| rollingLinearFitProfileColumnsTyped(u16, allocator, typed, y.u16, options_value, device_value),
        .u32 => |typed| rollingLinearFitProfileColumnsTyped(u32, allocator, typed, y.u32, options_value, device_value),
        .u64 => |typed| rollingLinearFitProfileColumnsTyped(u64, allocator, typed, y.u64, options_value, device_value),
        .usize => |typed| rollingLinearFitProfileColumnsTyped(usize, allocator, typed, y.usize, options_value, device_value),
        .isize => |typed| rollingLinearFitProfileColumnsTyped(isize, allocator, typed, y.isize, options_value, device_value),
        .f16 => |typed| rollingLinearFitProfileColumnsTyped(f16, allocator, typed, y.f16, options_value, device_value),
        .f32 => |typed| rollingLinearFitProfileColumnsTyped(f32, allocator, typed, y.f32, options_value, device_value),
        .f64 => |typed| rollingLinearFitProfileColumnsTyped(f64, allocator, typed, y.f64, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn rollingLinearFitProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    x_column: DeviceTypedColumn(T),
    y_column: DeviceTypedColumn(T),
    options_value: DeviceRollingCorrelationOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, InvalidDevice })![RollingLinearFitProfileColumnCount]DeviceColumn {
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

    var metrics = try rollingLinearFitProfile(
        allocator,
        xs,
        ys,
        maybe_x_validity,
        maybe_y_validity,
        options_value.window,
        min_periods,
    );
    defer metrics.deinit();

    var columns: [RollingLinearFitProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.pair_counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.slopes, metrics.fit_validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.intercepts, metrics.fit_validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.fitted, metrics.row_validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.residuals, metrics.row_validity, device_value);
    initialized += 1;
    columns[5] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.residual_z, metrics.row_validity, device_value);
    initialized += 1;
    return columns;
}

const LinearFitFrameError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    TypeUnsupported,
    UnsupportedType,
    InvalidCsv,
    EmptyDataFrame,
    InvalidDevice,
};

fn appendFitColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    source_names: []const []const u8,
    fit_columns: anytype,
) LinearFitFrameError!DeviceDataFrame {
    const DeviceColumnType = std.meta.Elem(@TypeOf(frame.columns));
    var columns = try frame.allocator.alloc(DeviceColumnType, frame.columns.len + fit_columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }
    for (frame.columns, 0..) |col, i| {
        columns[i] = try col.clone();
        initialized += 1;
    }
    for (&fit_columns) |*fit_col| {
        columns[initialized] = fit_col.*;
        initialized += 1;
    }
    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, source_names, columns, frame.rows, frame.device);
}

fn fitFrameFromColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    output_prefix: []const u8,
    fit_columns_value: anytype,
    comptime namesFn: anytype,
) LinearFitFrameError!DeviceDataFrame {
    var fit_columns = fit_columns_value;
    var fit_columns_transferred: usize = 0;
    errdefer {
        for (fit_columns[fit_columns_transferred..]) |*col| col.deinit();
    }

    const source_names = try frame.allocator.alloc([]const u8, frame.columns.len + fit_columns.len);
    defer frame.allocator.free(source_names);
    for (frame.names, 0..) |source_name, i| source_names[i] = source_name;

    var fit_names = try namesFn(frame.allocator, output_prefix);
    defer names_mod.freeOwnedNameItems(frame.allocator, fit_names[0..]);
    for (fit_names, 0..) |fit_name, i| source_names[frame.columns.len + i] = fit_name;

    const out = try appendFitColumns(DeviceDataFrame, frame, source_names, fit_columns);
    fit_columns_transferred = fit_columns.len;
    return out;
}

fn validateFitInputs(frame: anytype, x_name: []const u8, y_name: []const u8) LinearFitFrameError!struct { x: @TypeOf(frame.column(x_name) catch unreachable), y: @TypeOf(frame.column(y_name) catch unreachable) } {
    const x = try frame.column(x_name);
    const y = try frame.column(y_name);
    if (x.dtype() != y.dtype()) return error.TypeMismatch;
    return .{ .x = x, .y = y };
}

pub fn linearFitProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    x_name: []const u8,
    y_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceLinearFitOptions,
) LinearFitFrameError!DeviceDataFrame {
    const inputs = try validateFitInputs(frame, x_name, y_name);
    const fit_columns = try linearFitProfileColumnsByValue(frame.allocator, inputs.x.*, inputs.y.*, options_value, frame.device, frame.rows);
    return fitFrameFromColumns(DeviceDataFrame, frame, output_prefix, fit_columns, linearFitProfileOutputNames);
}

pub fn expandingLinearFitProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    x_name: []const u8,
    y_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceExpandingOptions,
) LinearFitFrameError!DeviceDataFrame {
    const inputs = try validateFitInputs(frame, x_name, y_name);
    const fit_columns = try expandingLinearFitProfileColumnsByValue(frame.allocator, inputs.x.*, inputs.y.*, options_value, frame.device, frame.rows);
    return fitFrameFromColumns(DeviceDataFrame, frame, output_prefix, fit_columns, expandingLinearFitProfileOutputNames);
}

pub fn rollingLinearFitProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    x_name: []const u8,
    y_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceRollingCorrelationOptions,
) LinearFitFrameError!DeviceDataFrame {
    const inputs = try validateFitInputs(frame, x_name, y_name);
    const fit_columns = try rollingLinearFitProfileColumnsByValue(frame.allocator, inputs.x.*, inputs.y.*, options_value, frame.device, frame.rows);
    return fitFrameFromColumns(DeviceDataFrame, frame, output_prefix, fit_columns, rollingLinearFitProfileOutputNames);
}
