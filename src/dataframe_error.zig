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

pub const ErrorMetrics = struct {
    allocator: std.mem.Allocator,
    errors: []f64,
    abs_errors: []f64,
    squared_errors: []f64,
    ape: []f64,
    smape: []f64,
    validity: []bool,

    pub fn deinit(self: *ErrorMetrics) void {
        self.allocator.free(self.errors);
        self.allocator.free(self.abs_errors);
        self.allocator.free(self.squared_errors);
        self.allocator.free(self.ape);
        self.allocator.free(self.smape);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub const ErrorSummaryMetrics = struct {
    allocator: std.mem.Allocator,
    counts: []i64,
    mae: []f64,
    rmse: []f64,
    mape: []f64,
    smape: []f64,
    validity: []bool,

    pub fn deinit(self: *ErrorSummaryMetrics) void {
        self.allocator.free(self.counts);
        self.allocator.free(self.mae);
        self.allocator.free(self.rmse);
        self.allocator.free(self.mape);
        self.allocator.free(self.smape);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub const ErrorProfileColumnCount = 5;

pub fn errorProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![ErrorProfileColumnCount][]const u8 {
    var names: [ErrorProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "error", "abs_error", "squared_error", "ape", "smape" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const RollingErrorProfileColumnCount = 5;

pub fn rollingErrorProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![RollingErrorProfileColumnCount][]const u8 {
    var names: [RollingErrorProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "rolling_error_count", "rolling_mae", "rolling_rmse", "rolling_mape", "rolling_smape" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const ExpandingErrorProfileColumnCount = 5;

pub fn expandingErrorProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![ExpandingErrorProfileColumnCount][]const u8 {
    var names: [ExpandingErrorProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "expanding_error_count", "expanding_mae", "expanding_rmse", "expanding_mape", "expanding_smape" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn validatePairLengths(
    actual: []const f64,
    predicted: []const f64,
    maybe_actual_validity: ?[]const bool,
    maybe_predicted_validity: ?[]const bool,
) error{LengthMismatch}!void {
    if (actual.len != predicted.len) return error.LengthMismatch;
    if (maybe_actual_validity) |validity| {
        if (validity.len != actual.len) return error.LengthMismatch;
    }
    if (maybe_predicted_validity) |validity| {
        if (validity.len != predicted.len) return error.LengthMismatch;
    }
}

fn rowValid(maybe_actual_validity: ?[]const bool, maybe_predicted_validity: ?[]const bool, row: usize) bool {
    return (if (maybe_actual_validity) |mask| mask[row] else true) and (if (maybe_predicted_validity) |mask| mask[row] else true);
}

fn errorTerms(actual: f64, predicted: f64) struct { err: f64, abs_err: f64, squared: f64, ape: f64, smape: f64 } {
    const err = actual - predicted;
    const abs_err = @abs(err);
    const denom = @abs(actual) + @abs(predicted);
    return .{
        .err = err,
        .abs_err = abs_err,
        .squared = err * err,
        .ape = if (actual == 0) std.math.nan(f64) else abs_err / @abs(actual),
        .smape = if (denom == 0) std.math.nan(f64) else 2.0 * abs_err / denom,
    };
}

fn allocSummary(allocator: std.mem.Allocator, rows: usize) std.mem.Allocator.Error!ErrorSummaryMetrics {
    const counts = try allocator.alloc(i64, rows);
    errdefer allocator.free(counts);
    const mae = try allocator.alloc(f64, rows);
    errdefer allocator.free(mae);
    const rmse = try allocator.alloc(f64, rows);
    errdefer allocator.free(rmse);
    const mape = try allocator.alloc(f64, rows);
    errdefer allocator.free(mape);
    const smape = try allocator.alloc(f64, rows);
    errdefer allocator.free(smape);
    const validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(validity);
    return .{
        .allocator = allocator,
        .counts = counts,
        .mae = mae,
        .rmse = rmse,
        .mape = mape,
        .smape = smape,
        .validity = validity,
    };
}

fn writeSummaryRow(row: usize, min_periods: usize, count: usize, abs_sum: f64, sq_sum: f64, ape_sum: f64, smape_sum: f64, out: ErrorSummaryMetrics) void {
    out.counts[row] = @intCast(count);
    const has_enough = count >= min_periods;
    out.validity[row] = has_enough;
    if (has_enough) {
        const n: f64 = @floatFromInt(count);
        out.mae[row] = abs_sum / n;
        out.rmse[row] = std.math.sqrt(sq_sum / n);
        out.mape[row] = ape_sum / n;
        out.smape[row] = smape_sum / n;
    } else {
        out.mae[row] = 0;
        out.rmse[row] = 0;
        out.mape[row] = 0;
        out.smape[row] = 0;
    }
}

pub fn errorProfile(
    allocator: std.mem.Allocator,
    actual: []const f64,
    predicted: []const f64,
    maybe_actual_validity: ?[]const bool,
    maybe_predicted_validity: ?[]const bool,
) (std.mem.Allocator.Error || error{LengthMismatch})!ErrorMetrics {
    try validatePairLengths(actual, predicted, maybe_actual_validity, maybe_predicted_validity);

    const rows = actual.len;
    const errors = try allocator.alloc(f64, rows);
    errdefer allocator.free(errors);
    const abs_errors = try allocator.alloc(f64, rows);
    errdefer allocator.free(abs_errors);
    const squared_errors = try allocator.alloc(f64, rows);
    errdefer allocator.free(squared_errors);
    const ape = try allocator.alloc(f64, rows);
    errdefer allocator.free(ape);
    const smape = try allocator.alloc(f64, rows);
    errdefer allocator.free(smape);
    const validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(validity);

    for (actual, predicted, 0..) |actual_value, predicted_value, row| {
        const valid = rowValid(maybe_actual_validity, maybe_predicted_validity, row);
        validity[row] = valid;
        if (valid) {
            const terms = errorTerms(actual_value, predicted_value);
            errors[row] = terms.err;
            abs_errors[row] = terms.abs_err;
            squared_errors[row] = terms.squared;
            ape[row] = terms.ape;
            smape[row] = terms.smape;
        } else {
            errors[row] = 0;
            abs_errors[row] = 0;
            squared_errors[row] = 0;
            ape[row] = 0;
            smape[row] = 0;
        }
    }

    return .{
        .allocator = allocator,
        .errors = errors,
        .abs_errors = abs_errors,
        .squared_errors = squared_errors,
        .ape = ape,
        .smape = smape,
        .validity = validity,
    };
}

pub fn rollingErrorProfile(
    allocator: std.mem.Allocator,
    actual: []const f64,
    predicted: []const f64,
    maybe_actual_validity: ?[]const bool,
    maybe_predicted_validity: ?[]const bool,
    window: usize,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!ErrorSummaryMetrics {
    if (window == 0) return error.InvalidShape;
    if (min_periods == 0 or min_periods > window) return error.InvalidShape;
    try validatePairLengths(actual, predicted, maybe_actual_validity, maybe_predicted_validity);

    var out = try allocSummary(allocator, actual.len);
    errdefer out.deinit();

    for (0..actual.len) |row| {
        const start = if (row + 1 > window) row + 1 - window else 0;
        var count: usize = 0;
        var abs_sum: f64 = 0;
        var sq_sum: f64 = 0;
        var ape_sum: f64 = 0;
        var smape_sum: f64 = 0;
        for (start..row + 1) |window_row| {
            if (!rowValid(maybe_actual_validity, maybe_predicted_validity, window_row)) continue;
            const terms = errorTerms(actual[window_row], predicted[window_row]);
            abs_sum += terms.abs_err;
            sq_sum += terms.squared;
            ape_sum += terms.ape;
            smape_sum += terms.smape;
            count += 1;
        }
        writeSummaryRow(row, min_periods, count, abs_sum, sq_sum, ape_sum, smape_sum, out);
    }

    return out;
}

pub fn expandingErrorProfile(
    allocator: std.mem.Allocator,
    actual: []const f64,
    predicted: []const f64,
    maybe_actual_validity: ?[]const bool,
    maybe_predicted_validity: ?[]const bool,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!ErrorSummaryMetrics {
    if (min_periods == 0) return error.InvalidShape;
    try validatePairLengths(actual, predicted, maybe_actual_validity, maybe_predicted_validity);

    var out = try allocSummary(allocator, actual.len);
    errdefer out.deinit();

    var count: usize = 0;
    var abs_sum: f64 = 0;
    var sq_sum: f64 = 0;
    var ape_sum: f64 = 0;
    var smape_sum: f64 = 0;
    for (actual, predicted, 0..) |actual_value, predicted_value, row| {
        if (rowValid(maybe_actual_validity, maybe_predicted_validity, row)) {
            const terms = errorTerms(actual_value, predicted_value);
            abs_sum += terms.abs_err;
            sq_sum += terms.squared;
            ape_sum += terms.ape;
            smape_sum += terms.smape;
            count += 1;
        }
        writeSummaryRow(row, min_periods, count, abs_sum, sq_sum, ape_sum, smape_sum, out);
    }

    return out;
}

pub fn errorProfileColumnsByValue(
    allocator: std.mem.Allocator,
    actual: DeviceColumn,
    predicted: DeviceColumn,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, InvalidDevice })![ErrorProfileColumnCount]DeviceColumn {
    if (actual.len() != rows or predicted.len() != rows) return error.LengthMismatch;
    if (actual.dtype() != predicted.dtype()) return error.TypeMismatch;
    return switch (actual) {
        .i8 => |typed| errorProfileColumnsTyped(i8, allocator, typed, predicted.i8, device_value),
        .i16 => |typed| errorProfileColumnsTyped(i16, allocator, typed, predicted.i16, device_value),
        .i32 => |typed| errorProfileColumnsTyped(i32, allocator, typed, predicted.i32, device_value),
        .i64 => |typed| errorProfileColumnsTyped(i64, allocator, typed, predicted.i64, device_value),
        .u8 => |typed| errorProfileColumnsTyped(u8, allocator, typed, predicted.u8, device_value),
        .u16 => |typed| errorProfileColumnsTyped(u16, allocator, typed, predicted.u16, device_value),
        .u32 => |typed| errorProfileColumnsTyped(u32, allocator, typed, predicted.u32, device_value),
        .u64 => |typed| errorProfileColumnsTyped(u64, allocator, typed, predicted.u64, device_value),
        .usize => |typed| errorProfileColumnsTyped(usize, allocator, typed, predicted.usize, device_value),
        .isize => |typed| errorProfileColumnsTyped(isize, allocator, typed, predicted.isize, device_value),
        .f16 => |typed| errorProfileColumnsTyped(f16, allocator, typed, predicted.f16, device_value),
        .f32 => |typed| errorProfileColumnsTyped(f32, allocator, typed, predicted.f32, device_value),
        .f64 => |typed| errorProfileColumnsTyped(f64, allocator, typed, predicted.f64, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn errorProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    actual_column: DeviceTypedColumn(T),
    predicted_column: DeviceTypedColumn(T),
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, InvalidDevice })![ErrorProfileColumnCount]DeviceColumn {
    if (actual_column.len() != predicted_column.len()) return error.LengthMismatch;
    if (!actual_column.device().sameDevice(predicted_column.device())) return error.InvalidDevice;

    const actual_values_typed = try actual_column.values.toOwnedSlice(allocator);
    defer allocator.free(actual_values_typed);
    const predicted_values_typed = try predicted_column.values.toOwnedSlice(allocator);
    defer allocator.free(predicted_values_typed);
    const maybe_actual_validity = try validityValues(actual_column, allocator);
    defer if (maybe_actual_validity) |validity| allocator.free(validity);
    const maybe_predicted_validity = try validityValues(predicted_column, allocator);
    defer if (maybe_predicted_validity) |validity| allocator.free(validity);

    const rows = actual_values_typed.len;
    const actual_values = try allocator.alloc(f64, rows);
    defer allocator.free(actual_values);
    const predicted_values = try allocator.alloc(f64, rows);
    defer allocator.free(predicted_values);
    for (actual_values_typed, predicted_values_typed, 0..) |actual_value, predicted_value, row| {
        actual_values[row] = castToF64(T, actual_value);
        predicted_values[row] = castToF64(T, predicted_value);
    }

    var metrics = try errorProfile(allocator, actual_values, predicted_values, maybe_actual_validity, maybe_predicted_validity);
    defer metrics.deinit();

    var columns: [ErrorProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.errors, metrics.validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.abs_errors, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.squared_errors, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.ape, metrics.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.smape, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn rollingErrorProfileColumnsByValue(
    allocator: std.mem.Allocator,
    actual: DeviceColumn,
    predicted: DeviceColumn,
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, InvalidDevice })![RollingErrorProfileColumnCount]DeviceColumn {
    if (actual.len() != rows or predicted.len() != rows) return error.LengthMismatch;
    if (actual.dtype() != predicted.dtype()) return error.TypeMismatch;
    return switch (actual) {
        .i8 => |typed| rollingErrorProfileColumnsTyped(i8, allocator, typed, predicted.i8, options_value, device_value),
        .i16 => |typed| rollingErrorProfileColumnsTyped(i16, allocator, typed, predicted.i16, options_value, device_value),
        .i32 => |typed| rollingErrorProfileColumnsTyped(i32, allocator, typed, predicted.i32, options_value, device_value),
        .i64 => |typed| rollingErrorProfileColumnsTyped(i64, allocator, typed, predicted.i64, options_value, device_value),
        .u8 => |typed| rollingErrorProfileColumnsTyped(u8, allocator, typed, predicted.u8, options_value, device_value),
        .u16 => |typed| rollingErrorProfileColumnsTyped(u16, allocator, typed, predicted.u16, options_value, device_value),
        .u32 => |typed| rollingErrorProfileColumnsTyped(u32, allocator, typed, predicted.u32, options_value, device_value),
        .u64 => |typed| rollingErrorProfileColumnsTyped(u64, allocator, typed, predicted.u64, options_value, device_value),
        .usize => |typed| rollingErrorProfileColumnsTyped(usize, allocator, typed, predicted.usize, options_value, device_value),
        .isize => |typed| rollingErrorProfileColumnsTyped(isize, allocator, typed, predicted.isize, options_value, device_value),
        .f16 => |typed| rollingErrorProfileColumnsTyped(f16, allocator, typed, predicted.f16, options_value, device_value),
        .f32 => |typed| rollingErrorProfileColumnsTyped(f32, allocator, typed, predicted.f32, options_value, device_value),
        .f64 => |typed| rollingErrorProfileColumnsTyped(f64, allocator, typed, predicted.f64, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn rollingErrorProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    actual_column: DeviceTypedColumn(T),
    predicted_column: DeviceTypedColumn(T),
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, InvalidDevice })![RollingErrorProfileColumnCount]DeviceColumn {
    const min_periods = options_value.min_periods orelse options_value.window;
    if (actual_column.len() != predicted_column.len()) return error.LengthMismatch;
    if (!actual_column.device().sameDevice(predicted_column.device())) return error.InvalidDevice;

    const actual_values_typed = try actual_column.values.toOwnedSlice(allocator);
    defer allocator.free(actual_values_typed);
    const predicted_values_typed = try predicted_column.values.toOwnedSlice(allocator);
    defer allocator.free(predicted_values_typed);
    const maybe_actual_validity = try validityValues(actual_column, allocator);
    defer if (maybe_actual_validity) |validity| allocator.free(validity);
    const maybe_predicted_validity = try validityValues(predicted_column, allocator);
    defer if (maybe_predicted_validity) |validity| allocator.free(validity);

    const rows = actual_values_typed.len;
    const actual_values = try allocator.alloc(f64, rows);
    defer allocator.free(actual_values);
    const predicted_values = try allocator.alloc(f64, rows);
    defer allocator.free(predicted_values);
    for (actual_values_typed, predicted_values_typed, 0..) |actual_value, predicted_value, row| {
        actual_values[row] = castToF64(T, actual_value);
        predicted_values[row] = castToF64(T, predicted_value);
    }

    var metrics = try rollingErrorProfile(
        allocator,
        actual_values,
        predicted_values,
        maybe_actual_validity,
        maybe_predicted_validity,
        options_value.window,
        min_periods,
    );
    defer metrics.deinit();

    var columns: [RollingErrorProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.mae, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.rmse, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.mape, metrics.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.smape, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn expandingErrorProfileColumnsByValue(
    allocator: std.mem.Allocator,
    actual: DeviceColumn,
    predicted: DeviceColumn,
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, InvalidDevice })![ExpandingErrorProfileColumnCount]DeviceColumn {
    if (actual.len() != rows or predicted.len() != rows) return error.LengthMismatch;
    if (actual.dtype() != predicted.dtype()) return error.TypeMismatch;
    return switch (actual) {
        .i8 => |typed| expandingErrorProfileColumnsTyped(i8, allocator, typed, predicted.i8, options_value, device_value),
        .i16 => |typed| expandingErrorProfileColumnsTyped(i16, allocator, typed, predicted.i16, options_value, device_value),
        .i32 => |typed| expandingErrorProfileColumnsTyped(i32, allocator, typed, predicted.i32, options_value, device_value),
        .i64 => |typed| expandingErrorProfileColumnsTyped(i64, allocator, typed, predicted.i64, options_value, device_value),
        .u8 => |typed| expandingErrorProfileColumnsTyped(u8, allocator, typed, predicted.u8, options_value, device_value),
        .u16 => |typed| expandingErrorProfileColumnsTyped(u16, allocator, typed, predicted.u16, options_value, device_value),
        .u32 => |typed| expandingErrorProfileColumnsTyped(u32, allocator, typed, predicted.u32, options_value, device_value),
        .u64 => |typed| expandingErrorProfileColumnsTyped(u64, allocator, typed, predicted.u64, options_value, device_value),
        .usize => |typed| expandingErrorProfileColumnsTyped(usize, allocator, typed, predicted.usize, options_value, device_value),
        .isize => |typed| expandingErrorProfileColumnsTyped(isize, allocator, typed, predicted.isize, options_value, device_value),
        .f16 => |typed| expandingErrorProfileColumnsTyped(f16, allocator, typed, predicted.f16, options_value, device_value),
        .f32 => |typed| expandingErrorProfileColumnsTyped(f32, allocator, typed, predicted.f32, options_value, device_value),
        .f64 => |typed| expandingErrorProfileColumnsTyped(f64, allocator, typed, predicted.f64, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn expandingErrorProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    actual_column: DeviceTypedColumn(T),
    predicted_column: DeviceTypedColumn(T),
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, InvalidDevice })![ExpandingErrorProfileColumnCount]DeviceColumn {
    if (actual_column.len() != predicted_column.len()) return error.LengthMismatch;
    if (!actual_column.device().sameDevice(predicted_column.device())) return error.InvalidDevice;

    const actual_values_typed = try actual_column.values.toOwnedSlice(allocator);
    defer allocator.free(actual_values_typed);
    const predicted_values_typed = try predicted_column.values.toOwnedSlice(allocator);
    defer allocator.free(predicted_values_typed);
    const maybe_actual_validity = try validityValues(actual_column, allocator);
    defer if (maybe_actual_validity) |validity| allocator.free(validity);
    const maybe_predicted_validity = try validityValues(predicted_column, allocator);
    defer if (maybe_predicted_validity) |validity| allocator.free(validity);

    const rows = actual_values_typed.len;
    const actual_values = try allocator.alloc(f64, rows);
    defer allocator.free(actual_values);
    const predicted_values = try allocator.alloc(f64, rows);
    defer allocator.free(predicted_values);
    for (actual_values_typed, predicted_values_typed, 0..) |actual_value, predicted_value, row| {
        actual_values[row] = castToF64(T, actual_value);
        predicted_values[row] = castToF64(T, predicted_value);
    }

    var metrics = try expandingErrorProfile(
        allocator,
        actual_values,
        predicted_values,
        maybe_actual_validity,
        maybe_predicted_validity,
        options_value.min_periods,
    );
    defer metrics.deinit();

    var columns: [ExpandingErrorProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.mae, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.rmse, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.mape, metrics.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.smape, metrics.validity, device_value);
    initialized += 1;
    return columns;
}

const ErrorFrameError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    TypeUnsupported,
    UnsupportedType,
    InvalidCsv,
    EmptyDataFrame,
    InvalidDevice,
};

fn appendErrorColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    source_names: []const []const u8,
    error_columns: anytype,
) ErrorFrameError!DeviceDataFrame {
    const DeviceColumnType = std.meta.Elem(@TypeOf(frame.columns));
    var columns = try frame.allocator.alloc(DeviceColumnType, frame.columns.len + error_columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }
    for (frame.columns, 0..) |col, i| {
        columns[i] = try col.clone();
        initialized += 1;
    }
    for (&error_columns) |*error_col| {
        columns[initialized] = error_col.*;
        initialized += 1;
    }
    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, source_names, columns, frame.rows, frame.device);
}

fn errorFrameFromColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    output_prefix: []const u8,
    error_columns_value: anytype,
    comptime namesFn: anytype,
) ErrorFrameError!DeviceDataFrame {
    var error_columns = error_columns_value;
    var error_columns_transferred: usize = 0;
    errdefer {
        for (error_columns[error_columns_transferred..]) |*col| col.deinit();
    }

    const source_names = try frame.allocator.alloc([]const u8, frame.columns.len + error_columns.len);
    defer frame.allocator.free(source_names);
    for (frame.names, 0..) |source_name, i| source_names[i] = source_name;

    var error_names = try namesFn(frame.allocator, output_prefix);
    defer names_mod.freeOwnedNameItems(frame.allocator, error_names[0..]);
    for (error_names, 0..) |error_name, i| source_names[frame.columns.len + i] = error_name;

    const out = try appendErrorColumns(DeviceDataFrame, frame, source_names, error_columns);
    error_columns_transferred = error_columns.len;
    return out;
}

fn validateErrorInputs(frame: anytype, actual_name: []const u8, predicted_name: []const u8) ErrorFrameError!struct { actual: @TypeOf(frame.column(actual_name) catch unreachable), predicted: @TypeOf(frame.column(predicted_name) catch unreachable) } {
    const actual = try frame.column(actual_name);
    const predicted = try frame.column(predicted_name);
    if (actual.dtype() != predicted.dtype()) return error.TypeMismatch;
    return .{ .actual = actual, .predicted = predicted };
}

pub fn errorProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    actual_name: []const u8,
    predicted_name: []const u8,
    output_prefix: []const u8,
) ErrorFrameError!DeviceDataFrame {
    const inputs = try validateErrorInputs(frame, actual_name, predicted_name);
    const error_columns = try errorProfileColumnsByValue(frame.allocator, inputs.actual.*, inputs.predicted.*, frame.device, frame.rows);
    return errorFrameFromColumns(DeviceDataFrame, frame, output_prefix, error_columns, errorProfileOutputNames);
}

pub fn rollingErrorProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    actual_name: []const u8,
    predicted_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceRollingOptions,
) ErrorFrameError!DeviceDataFrame {
    const inputs = try validateErrorInputs(frame, actual_name, predicted_name);
    const error_columns = try rollingErrorProfileColumnsByValue(frame.allocator, inputs.actual.*, inputs.predicted.*, options_value, frame.device, frame.rows);
    return errorFrameFromColumns(DeviceDataFrame, frame, output_prefix, error_columns, rollingErrorProfileOutputNames);
}

pub fn expandingErrorProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    actual_name: []const u8,
    predicted_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceExpandingOptions,
) ErrorFrameError!DeviceDataFrame {
    const inputs = try validateErrorInputs(frame, actual_name, predicted_name);
    const error_columns = try expandingErrorProfileColumnsByValue(frame.allocator, inputs.actual.*, inputs.predicted.*, options_value, frame.device, frame.rows);
    return errorFrameFromColumns(DeviceDataFrame, frame, output_prefix, error_columns, expandingErrorProfileOutputNames);
}
