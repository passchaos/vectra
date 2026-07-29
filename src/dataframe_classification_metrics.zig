//! Classification metric kernels and output-name helpers.

const std = @import("std");
const summary_metrics_mod = @import("dataframe_classification_summary_metrics.zig");

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

pub const ClassificationSummaryProfile = summary_metrics_mod.ClassificationSummaryProfile;
pub const RollingClassificationProfileColumnCount = summary_metrics_mod.RollingClassificationProfileColumnCount;
pub const rollingClassificationProfileOutputNames = summary_metrics_mod.rollingClassificationProfileOutputNames;
pub const ExpandingClassificationProfileColumnCount = summary_metrics_mod.ExpandingClassificationProfileColumnCount;
pub const expandingClassificationProfileOutputNames = summary_metrics_mod.expandingClassificationProfileOutputNames;
pub fn validatePairLengths(
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

pub fn rowValid(maybe_actual_validity: ?[]const bool, maybe_predicted_validity: ?[]const bool, row: usize) bool {
    return (if (maybe_actual_validity) |mask| mask[row] else true) and (if (maybe_predicted_validity) |mask| mask[row] else true);
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

pub const rollingClassificationProfile = summary_metrics_mod.rollingClassificationProfile;
pub const expandingClassificationProfile = summary_metrics_mod.expandingClassificationProfile;
