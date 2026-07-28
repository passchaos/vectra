const std = @import("std");

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
