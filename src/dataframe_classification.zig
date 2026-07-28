const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_device_column_mod = @import("dataframe_device_column.zig");
const options_mod = @import("dataframe_options.zig");
const validity_mod = @import("dataframe_validity.zig");

const DeviceColumn = dataframe_device_column_mod.DeviceColumn;
const DeviceTypedColumn = dataframe_device_column_mod.DeviceTypedColumn;
const DeviceRollingOptions = options_mod.DeviceRollingOptions;
const DeviceExpandingOptions = options_mod.DeviceExpandingOptions;
const validityValues = validity_mod.validityValues;

pub const ClassificationProfile = struct {
    allocator: std.mem.Allocator,
    tp: []bool,
    fp: []bool,
    tn: []bool,
    fn_values: []bool,
    correct: []bool,
    validity: []bool,

    pub fn deinit(self: *ClassificationProfile) void {
        self.allocator.free(self.tp);
        self.allocator.free(self.fp);
        self.allocator.free(self.tn);
        self.allocator.free(self.fn_values);
        self.allocator.free(self.correct);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub const ClassificationSummaryProfile = struct {
    allocator: std.mem.Allocator,
    counts: []i64,
    tp_counts: []i64,
    fp_counts: []i64,
    tn_counts: []i64,
    fn_counts: []i64,
    accuracies: []f64,
    precisions: []f64,
    recalls: []f64,
    metric_validity: []bool,

    pub fn deinit(self: *ClassificationSummaryProfile) void {
        self.allocator.free(self.counts);
        self.allocator.free(self.tp_counts);
        self.allocator.free(self.fp_counts);
        self.allocator.free(self.tn_counts);
        self.allocator.free(self.fn_counts);
        self.allocator.free(self.accuracies);
        self.allocator.free(self.precisions);
        self.allocator.free(self.recalls);
        self.allocator.free(self.metric_validity);
        self.* = undefined;
    }
};

pub const ClassificationProfileColumnCount = 5;

pub fn classificationProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![ClassificationProfileColumnCount][]const u8 {
    var names: [ClassificationProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "tp", "fp", "tn", "fn", "correct" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const RollingClassificationProfileColumnCount = 8;

pub fn rollingClassificationProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![RollingClassificationProfileColumnCount][]const u8 {
    var names: [RollingClassificationProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "rolling_class_count", "rolling_tp_count", "rolling_fp_count", "rolling_tn_count", "rolling_fn_count", "rolling_accuracy", "rolling_precision", "rolling_recall" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const ExpandingClassificationProfileColumnCount = 8;

pub fn expandingClassificationProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![ExpandingClassificationProfileColumnCount][]const u8 {
    var names: [ExpandingClassificationProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "expanding_class_count", "expanding_tp_count", "expanding_fp_count", "expanding_tn_count", "expanding_fn_count", "expanding_accuracy", "expanding_precision", "expanding_recall" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn validatePairLengths(
    actual: []const bool,
    predicted: []const bool,
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

fn addConfusion(actual: bool, predicted: bool, tp: *usize, fp: *usize, tn: *usize, fn_count: *usize) void {
    if (actual and predicted) {
        tp.* += 1;
    } else if (!actual and predicted) {
        fp.* += 1;
    } else if (!actual and !predicted) {
        tn.* += 1;
    } else {
        fn_count.* += 1;
    }
}

fn removeConfusion(actual: bool, predicted: bool, tp: *usize, fp: *usize, tn: *usize, fn_count: *usize) void {
    if (actual and predicted) {
        tp.* -= 1;
    } else if (!actual and predicted) {
        fp.* -= 1;
    } else if (!actual and !predicted) {
        tn.* -= 1;
    } else {
        fn_count.* -= 1;
    }
}

fn writeSummaryRow(
    row: usize,
    min_periods: usize,
    tp: usize,
    fp: usize,
    tn: usize,
    fn_count: usize,
    out: ClassificationSummaryProfile,
) void {
    const count = tp + fp + tn + fn_count;
    out.counts[row] = @intCast(count);
    out.tp_counts[row] = @intCast(tp);
    out.fp_counts[row] = @intCast(fp);
    out.tn_counts[row] = @intCast(tn);
    out.fn_counts[row] = @intCast(fn_count);
    const has_enough = count >= min_periods;
    out.metric_validity[row] = has_enough;
    if (has_enough) {
        const n: f64 = @floatFromInt(count);
        const predicted_positive = tp + fp;
        const actual_positive = tp + fn_count;
        out.accuracies[row] = @as(f64, @floatFromInt(tp + tn)) / n;
        out.precisions[row] = if (predicted_positive == 0) std.math.nan(f64) else @as(f64, @floatFromInt(tp)) / @as(f64, @floatFromInt(predicted_positive));
        out.recalls[row] = if (actual_positive == 0) std.math.nan(f64) else @as(f64, @floatFromInt(tp)) / @as(f64, @floatFromInt(actual_positive));
    } else {
        out.accuracies[row] = 0;
        out.precisions[row] = 0;
        out.recalls[row] = 0;
    }
}

fn allocSummary(allocator: std.mem.Allocator, rows: usize) std.mem.Allocator.Error!ClassificationSummaryProfile {
    const counts = try allocator.alloc(i64, rows);
    errdefer allocator.free(counts);
    const tp_counts = try allocator.alloc(i64, rows);
    errdefer allocator.free(tp_counts);
    const fp_counts = try allocator.alloc(i64, rows);
    errdefer allocator.free(fp_counts);
    const tn_counts = try allocator.alloc(i64, rows);
    errdefer allocator.free(tn_counts);
    const fn_counts = try allocator.alloc(i64, rows);
    errdefer allocator.free(fn_counts);
    const accuracies = try allocator.alloc(f64, rows);
    errdefer allocator.free(accuracies);
    const precisions = try allocator.alloc(f64, rows);
    errdefer allocator.free(precisions);
    const recalls = try allocator.alloc(f64, rows);
    errdefer allocator.free(recalls);
    const metric_validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(metric_validity);

    return .{
        .allocator = allocator,
        .counts = counts,
        .tp_counts = tp_counts,
        .fp_counts = fp_counts,
        .tn_counts = tn_counts,
        .fn_counts = fn_counts,
        .accuracies = accuracies,
        .precisions = precisions,
        .recalls = recalls,
        .metric_validity = metric_validity,
    };
}

pub fn classificationProfile(
    allocator: std.mem.Allocator,
    actual: []const bool,
    predicted: []const bool,
    maybe_actual_validity: ?[]const bool,
    maybe_predicted_validity: ?[]const bool,
) (std.mem.Allocator.Error || error{LengthMismatch})!ClassificationProfile {
    try validatePairLengths(actual, predicted, maybe_actual_validity, maybe_predicted_validity);

    const rows = actual.len;
    const tp = try allocator.alloc(bool, rows);
    errdefer allocator.free(tp);
    const fp = try allocator.alloc(bool, rows);
    errdefer allocator.free(fp);
    const tn = try allocator.alloc(bool, rows);
    errdefer allocator.free(tn);
    const fn_values = try allocator.alloc(bool, rows);
    errdefer allocator.free(fn_values);
    const correct = try allocator.alloc(bool, rows);
    errdefer allocator.free(correct);
    const validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(validity);

    for (actual, predicted, 0..) |actual_value, predicted_value, row| {
        const valid = rowValid(maybe_actual_validity, maybe_predicted_validity, row);
        validity[row] = valid;
        if (valid) {
            tp[row] = actual_value and predicted_value;
            fp[row] = !actual_value and predicted_value;
            tn[row] = !actual_value and !predicted_value;
            fn_values[row] = actual_value and !predicted_value;
            correct[row] = actual_value == predicted_value;
        } else {
            tp[row] = false;
            fp[row] = false;
            tn[row] = false;
            fn_values[row] = false;
            correct[row] = false;
        }
    }

    return .{
        .allocator = allocator,
        .tp = tp,
        .fp = fp,
        .tn = tn,
        .fn_values = fn_values,
        .correct = correct,
        .validity = validity,
    };
}

pub fn rollingClassificationProfile(
    allocator: std.mem.Allocator,
    actual: []const bool,
    predicted: []const bool,
    maybe_actual_validity: ?[]const bool,
    maybe_predicted_validity: ?[]const bool,
    window: usize,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!ClassificationSummaryProfile {
    if (window == 0) return error.InvalidShape;
    if (min_periods == 0 or min_periods > window) return error.InvalidShape;
    try validatePairLengths(actual, predicted, maybe_actual_validity, maybe_predicted_validity);

    var out = try allocSummary(allocator, actual.len);
    errdefer out.deinit();

    var running_tp: usize = 0;
    var running_fp: usize = 0;
    var running_tn: usize = 0;
    var running_fn: usize = 0;
    for (actual, predicted, 0..) |actual_value, predicted_value, row| {
        if (rowValid(maybe_actual_validity, maybe_predicted_validity, row)) {
            addConfusion(actual_value, predicted_value, &running_tp, &running_fp, &running_tn, &running_fn);
        }

        if (row >= window) {
            const evict_row = row - window;
            if (rowValid(maybe_actual_validity, maybe_predicted_validity, evict_row)) {
                removeConfusion(actual[evict_row], predicted[evict_row], &running_tp, &running_fp, &running_tn, &running_fn);
            }
        }

        writeSummaryRow(row, min_periods, running_tp, running_fp, running_tn, running_fn, out);
    }

    return out;
}

pub fn expandingClassificationProfile(
    allocator: std.mem.Allocator,
    actual: []const bool,
    predicted: []const bool,
    maybe_actual_validity: ?[]const bool,
    maybe_predicted_validity: ?[]const bool,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!ClassificationSummaryProfile {
    if (min_periods == 0) return error.InvalidShape;
    try validatePairLengths(actual, predicted, maybe_actual_validity, maybe_predicted_validity);

    var out = try allocSummary(allocator, actual.len);
    errdefer out.deinit();

    var tp_count: usize = 0;
    var fp_count: usize = 0;
    var tn_count: usize = 0;
    var fn_count: usize = 0;
    for (actual, predicted, 0..) |actual_value, predicted_value, row| {
        if (rowValid(maybe_actual_validity, maybe_predicted_validity, row)) {
            addConfusion(actual_value, predicted_value, &tp_count, &fp_count, &tn_count, &fn_count);
        }

        writeSummaryRow(row, min_periods, tp_count, fp_count, tn_count, fn_count, out);
    }

    return out;
}

pub fn classificationProfileColumns(
    allocator: std.mem.Allocator,
    actual: DeviceTypedColumn(bool),
    predicted: DeviceTypedColumn(bool),
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{ LengthMismatch, InvalidDevice })![ClassificationProfileColumnCount]DeviceColumn {
    if (actual.len() != rows or predicted.len() != rows) return error.LengthMismatch;
    if (!actual.device().sameDevice(predicted.device())) return error.InvalidDevice;

    const actual_values = try actual.values.toOwnedSlice(allocator);
    defer allocator.free(actual_values);
    const predicted_values = try predicted.values.toOwnedSlice(allocator);
    defer allocator.free(predicted_values);
    const maybe_actual_validity = try validityValues(actual, allocator);
    defer if (maybe_actual_validity) |validity| allocator.free(validity);
    const maybe_predicted_validity = try validityValues(predicted, allocator);
    defer if (maybe_predicted_validity) |validity| allocator.free(validity);

    var profile = try classificationProfile(allocator, actual_values, predicted_values, maybe_actual_validity, maybe_predicted_validity);
    defer profile.deinit();

    var columns: [ClassificationProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(bool, allocator, profile.tp, profile.validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(bool, allocator, profile.fp, profile.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(bool, allocator, profile.tn, profile.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(bool, allocator, profile.fn_values, profile.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(bool, allocator, profile.correct, profile.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn rollingClassificationProfileColumns(
    allocator: std.mem.Allocator,
    actual: DeviceTypedColumn(bool),
    predicted: DeviceTypedColumn(bool),
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{ LengthMismatch, InvalidDevice })![RollingClassificationProfileColumnCount]DeviceColumn {
    const min_periods = options_value.min_periods orelse options_value.window;
    if (actual.len() != rows or predicted.len() != rows) return error.LengthMismatch;
    if (!actual.device().sameDevice(predicted.device())) return error.InvalidDevice;

    const actual_values = try actual.values.toOwnedSlice(allocator);
    defer allocator.free(actual_values);
    const predicted_values = try predicted.values.toOwnedSlice(allocator);
    defer allocator.free(predicted_values);
    const maybe_actual_validity = try validityValues(actual, allocator);
    defer if (maybe_actual_validity) |validity| allocator.free(validity);
    const maybe_predicted_validity = try validityValues(predicted, allocator);
    defer if (maybe_predicted_validity) |validity| allocator.free(validity);

    var profile = try rollingClassificationProfile(
        allocator,
        actual_values,
        predicted_values,
        maybe_actual_validity,
        maybe_predicted_validity,
        options_value.window,
        min_periods,
    );
    defer profile.deinit();

    var columns: [RollingClassificationProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, profile.counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSlice(i64, allocator, profile.tp_counts, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSlice(i64, allocator, profile.fp_counts, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSlice(i64, allocator, profile.tn_counts, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSlice(i64, allocator, profile.fn_counts, device_value);
    initialized += 1;
    columns[5] = try DeviceColumn.fromSliceWithValidity(f64, allocator, profile.accuracies, profile.metric_validity, device_value);
    initialized += 1;
    columns[6] = try DeviceColumn.fromSliceWithValidity(f64, allocator, profile.precisions, profile.metric_validity, device_value);
    initialized += 1;
    columns[7] = try DeviceColumn.fromSliceWithValidity(f64, allocator, profile.recalls, profile.metric_validity, device_value);
    initialized += 1;
    return columns;
}
pub fn expandingClassificationProfileColumns(
    allocator: std.mem.Allocator,
    actual: DeviceTypedColumn(bool),
    predicted: DeviceTypedColumn(bool),
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{ LengthMismatch, InvalidDevice })![ExpandingClassificationProfileColumnCount]DeviceColumn {
    if (actual.len() != rows or predicted.len() != rows) return error.LengthMismatch;
    if (!actual.device().sameDevice(predicted.device())) return error.InvalidDevice;

    const actual_values = try actual.values.toOwnedSlice(allocator);
    defer allocator.free(actual_values);
    const predicted_values = try predicted.values.toOwnedSlice(allocator);
    defer allocator.free(predicted_values);
    const maybe_actual_validity = try validityValues(actual, allocator);
    defer if (maybe_actual_validity) |validity| allocator.free(validity);
    const maybe_predicted_validity = try validityValues(predicted, allocator);
    defer if (maybe_predicted_validity) |validity| allocator.free(validity);

    var profile = try expandingClassificationProfile(
        allocator,
        actual_values,
        predicted_values,
        maybe_actual_validity,
        maybe_predicted_validity,
        options_value.min_periods,
    );
    defer profile.deinit();

    var columns: [ExpandingClassificationProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, profile.counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSlice(i64, allocator, profile.tp_counts, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSlice(i64, allocator, profile.fp_counts, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSlice(i64, allocator, profile.tn_counts, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSlice(i64, allocator, profile.fn_counts, device_value);
    initialized += 1;
    columns[5] = try DeviceColumn.fromSliceWithValidity(f64, allocator, profile.accuracies, profile.metric_validity, device_value);
    initialized += 1;
    columns[6] = try DeviceColumn.fromSliceWithValidity(f64, allocator, profile.precisions, profile.metric_validity, device_value);
    initialized += 1;
    columns[7] = try DeviceColumn.fromSliceWithValidity(f64, allocator, profile.recalls, profile.metric_validity, device_value);
    initialized += 1;
    return columns;
}
