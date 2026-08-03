//! Eager pair/advanced DeviceDataFrame profile method wrappers.

const std = @import("std");
const array_mod = @import("../../array.zig");
const bool_transition_mod = @import("../bool_transition.zig");
const classification_mod = @import("../classification.zig");
const error_mod = @import("../error.zig");
const correlation_mod = @import("../correlation.zig");
const linear_fit_mod = @import("../../dataframe_linear_fit.zig");
const crossover_mod = @import("../crossover.zig");
const bucket_mod = @import("../../dataframe_bucket.zig");
const ema_mod = @import("../../dataframe_ema.zig");
const options_mod = @import("../../dataframe_options.zig");
const series_mod = @import("../../series.zig");

const DeviceDataError = series_mod.DataError || array_mod.ArrayError;
const DeviceRollingOptions = options_mod.DeviceRollingOptions;
const DeviceExpandingOptions = options_mod.DeviceExpandingOptions;
const DeviceTrendOptions = options_mod.DeviceTrendOptions;
const DeviceCrossoverOptions = options_mod.DeviceCrossoverOptions;
const DeviceBucketOptions = options_mod.DeviceBucketOptions;
const DeviceEmaOptions = options_mod.DeviceEmaOptions;
const DeviceLinearFitOptions = options_mod.DeviceLinearFitOptions;
const DeviceRollingCorrelationOptions = options_mod.DeviceRollingCorrelationOptions;

fn FrameType(comptime Frame: type) type {
    return switch (@typeInfo(Frame)) {
        .pointer => |ptr| ptr.child,
        else => Frame,
    };
}

fn frameValue(self: anytype) FrameType(@TypeOf(self)) {
    return switch (@typeInfo(@TypeOf(self))) {
        .pointer => self.*,
        else => self,
    };
}

pub fn crossoverProfile(
    self: anytype,
    lhs_name: []const u8,
    rhs_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceCrossoverOptions,
) DeviceDataError!FrameType(@TypeOf(self)) {
    return crossover_mod.crossoverProfileFrame(FrameType(@TypeOf(self)), frameValue(self), lhs_name, rhs_name, output_prefix, options_value);
}

pub fn rollingCrossoverProfile(
    self: anytype,
    lhs_name: []const u8,
    rhs_name: []const u8,
    output_prefix: []const u8,
    cross_options: DeviceCrossoverOptions,
    options_value: DeviceRollingOptions,
) DeviceDataError!FrameType(@TypeOf(self)) {
    return crossover_mod.rollingCrossoverProfileFrame(FrameType(@TypeOf(self)), frameValue(self), lhs_name, rhs_name, output_prefix, cross_options, options_value);
}

pub fn expandingCrossoverProfile(
    self: anytype,
    lhs_name: []const u8,
    rhs_name: []const u8,
    output_prefix: []const u8,
    cross_options: DeviceCrossoverOptions,
    options_value: DeviceExpandingOptions,
) DeviceDataError!FrameType(@TypeOf(self)) {
    return crossover_mod.expandingCrossoverProfileFrame(FrameType(@TypeOf(self)), frameValue(self), lhs_name, rhs_name, output_prefix, cross_options, options_value);
}

pub fn bucketProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceBucketOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return bucket_mod.bucketProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, options_value);
}

pub fn emaProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceEmaOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return ema_mod.emaProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, options_value);
}

pub fn linearFitProfile(
    self: anytype,
    x_name: []const u8,
    y_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceLinearFitOptions,
) DeviceDataError!FrameType(@TypeOf(self)) {
    return linear_fit_mod.linearFitProfileFrame(FrameType(@TypeOf(self)), frameValue(self), x_name, y_name, output_prefix, options_value);
}

pub fn errorProfile(
    self: anytype,
    actual_name: []const u8,
    predicted_name: []const u8,
    output_prefix: []const u8,
) DeviceDataError!FrameType(@TypeOf(self)) {
    return error_mod.errorProfileFrame(FrameType(@TypeOf(self)), frameValue(self), actual_name, predicted_name, output_prefix);
}

pub fn rollingErrorProfile(
    self: anytype,
    actual_name: []const u8,
    predicted_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceRollingOptions,
) DeviceDataError!FrameType(@TypeOf(self)) {
    return error_mod.rollingErrorProfileFrame(FrameType(@TypeOf(self)), frameValue(self), actual_name, predicted_name, output_prefix, options_value);
}

pub fn expandingErrorProfile(
    self: anytype,
    actual_name: []const u8,
    predicted_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceExpandingOptions,
) DeviceDataError!FrameType(@TypeOf(self)) {
    return error_mod.expandingErrorProfileFrame(FrameType(@TypeOf(self)), frameValue(self), actual_name, predicted_name, output_prefix, options_value);
}

pub fn classificationProfile(
    self: anytype,
    actual_name: []const u8,
    predicted_name: []const u8,
    output_prefix: []const u8,
) DeviceDataError!FrameType(@TypeOf(self)) {
    return classification_mod.classificationProfileFrame(FrameType(@TypeOf(self)), frameValue(self), actual_name, predicted_name, output_prefix);
}

pub fn rollingClassificationProfile(
    self: anytype,
    actual_name: []const u8,
    predicted_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceRollingOptions,
) DeviceDataError!FrameType(@TypeOf(self)) {
    return classification_mod.rollingClassificationProfileFrame(FrameType(@TypeOf(self)), frameValue(self), actual_name, predicted_name, output_prefix, options_value);
}

pub fn expandingClassificationProfile(
    self: anytype,
    actual_name: []const u8,
    predicted_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceExpandingOptions,
) DeviceDataError!FrameType(@TypeOf(self)) {
    return classification_mod.expandingClassificationProfileFrame(FrameType(@TypeOf(self)), frameValue(self), actual_name, predicted_name, output_prefix, options_value);
}

pub fn boolTransitionProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceTrendOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return bool_transition_mod.boolTransitionProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, options_value);
}

pub fn rollingBoolTransitionProfile(self: anytype, name: []const u8, output_prefix: []const u8, transition_options: DeviceTrendOptions, options_value: DeviceRollingOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return bool_transition_mod.rollingBoolTransitionProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, transition_options, options_value);
}

pub fn expandingBoolTransitionProfile(self: anytype, name: []const u8, output_prefix: []const u8, transition_options: DeviceTrendOptions, options_value: DeviceExpandingOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return bool_transition_mod.expandingBoolTransitionProfileFrame(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, transition_options, options_value);
}

pub fn rollingCorrelationProfile(
    self: anytype,
    x_name: []const u8,
    y_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceRollingCorrelationOptions,
) DeviceDataError!FrameType(@TypeOf(self)) {
    return correlation_mod.rollingCorrelationProfileFrame(FrameType(@TypeOf(self)), frameValue(self), x_name, y_name, output_prefix, options_value);
}

pub fn expandingCorrelationProfile(
    self: anytype,
    x_name: []const u8,
    y_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceExpandingOptions,
) DeviceDataError!FrameType(@TypeOf(self)) {
    return correlation_mod.expandingCorrelationProfileFrame(FrameType(@TypeOf(self)), frameValue(self), x_name, y_name, output_prefix, options_value);
}

pub fn expandingLinearFitProfile(
    self: anytype,
    x_name: []const u8,
    y_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceExpandingOptions,
) DeviceDataError!FrameType(@TypeOf(self)) {
    return linear_fit_mod.expandingLinearFitProfileFrame(FrameType(@TypeOf(self)), frameValue(self), x_name, y_name, output_prefix, options_value);
}

pub fn rollingLinearFitProfile(
    self: anytype,
    x_name: []const u8,
    y_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceRollingCorrelationOptions,
) DeviceDataError!FrameType(@TypeOf(self)) {
    return linear_fit_mod.rollingLinearFitProfileFrame(FrameType(@TypeOf(self)), frameValue(self), x_name, y_name, output_prefix, options_value);
}
