//! Lazy pair/advanced profile method wrappers.

const std = @import("std");
const array_mod = @import("../../array.zig");
const lazy_profile_mod = @import("profile_plan.zig");
const options_mod = @import("../../dataframe_options.zig");
const series_mod = @import("../../series.zig");

const DeviceRollingOptions = options_mod.DeviceRollingOptions;
const DeviceExpandingOptions = options_mod.DeviceExpandingOptions;
const DeviceTrendOptions = options_mod.DeviceTrendOptions;
const DeviceCrossoverOptions = options_mod.DeviceCrossoverOptions;
const DeviceBucketOptions = options_mod.DeviceBucketOptions;
const DeviceEmaOptions = options_mod.DeviceEmaOptions;
const DeviceLinearFitOptions = options_mod.DeviceLinearFitOptions;
const DeviceRollingCorrelationOptions = options_mod.DeviceRollingCorrelationOptions;
const DeviceDataError = series_mod.DataError || array_mod.ArrayError;

pub fn crossoverProfile(
    self: anytype,
    lhs_name: []const u8,
    rhs_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceCrossoverOptions,
) DeviceDataError!void {
    return lazy_profile_mod.appendPairOutputOptions(self, "crossover_profile", "lhs_name", lhs_name, "rhs_name", rhs_name, output_prefix, options_value);
}

pub fn rollingCrossoverProfile(
    self: anytype,
    lhs_name: []const u8,
    rhs_name: []const u8,
    output_prefix: []const u8,
    cross_options: DeviceCrossoverOptions,
    options_value: DeviceRollingOptions,
) DeviceDataError!void {
    return lazy_profile_mod.appendPairOutputExtraOptions(self, "rolling_crossover_profile", "lhs_name", lhs_name, "rhs_name", rhs_name, output_prefix, "cross_options", cross_options, options_value);
}

pub fn expandingCrossoverProfile(
    self: anytype,
    lhs_name: []const u8,
    rhs_name: []const u8,
    output_prefix: []const u8,
    cross_options: DeviceCrossoverOptions,
    options_value: DeviceExpandingOptions,
) DeviceDataError!void {
    return lazy_profile_mod.appendPairOutputExtraOptions(self, "expanding_crossover_profile", "lhs_name", lhs_name, "rhs_name", rhs_name, output_prefix, "cross_options", cross_options, options_value);
}

pub fn bucketProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceBucketOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputOptions(self, "bucket_profile", name, output_prefix, options_value);
}

pub fn emaProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceEmaOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputOptions(self, "ema_profile", name, output_prefix, options_value);
}

pub fn linearFitProfile(
    self: anytype,
    x_name: []const u8,
    y_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceLinearFitOptions,
) DeviceDataError!void {
    return lazy_profile_mod.appendPairOutputOptions(self, "linear_fit_profile", "x_name", x_name, "y_name", y_name, output_prefix, options_value);
}

pub fn errorProfile(
    self: anytype,
    actual_name: []const u8,
    predicted_name: []const u8,
    output_prefix: []const u8,
) DeviceDataError!void {
    return lazy_profile_mod.appendPairOutput(self, "error_profile", "actual_name", actual_name, "predicted_name", predicted_name, output_prefix);
}

pub fn rollingErrorProfile(
    self: anytype,
    actual_name: []const u8,
    predicted_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceRollingOptions,
) DeviceDataError!void {
    return lazy_profile_mod.appendPairOutputOptions(self, "rolling_error_profile", "actual_name", actual_name, "predicted_name", predicted_name, output_prefix, options_value);
}

pub fn expandingErrorProfile(
    self: anytype,
    actual_name: []const u8,
    predicted_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceExpandingOptions,
) DeviceDataError!void {
    return lazy_profile_mod.appendPairOutputOptions(self, "expanding_error_profile", "actual_name", actual_name, "predicted_name", predicted_name, output_prefix, options_value);
}

pub fn classificationProfile(
    self: anytype,
    actual_name: []const u8,
    predicted_name: []const u8,
    output_prefix: []const u8,
) DeviceDataError!void {
    return lazy_profile_mod.appendPairOutput(self, "classification_profile", "actual_name", actual_name, "predicted_name", predicted_name, output_prefix);
}

pub fn rollingClassificationProfile(
    self: anytype,
    actual_name: []const u8,
    predicted_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceRollingOptions,
) DeviceDataError!void {
    return lazy_profile_mod.appendPairOutputOptions(self, "rolling_classification_profile", "actual_name", actual_name, "predicted_name", predicted_name, output_prefix, options_value);
}

pub fn expandingClassificationProfile(
    self: anytype,
    actual_name: []const u8,
    predicted_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceExpandingOptions,
) DeviceDataError!void {
    return lazy_profile_mod.appendPairOutputOptions(self, "expanding_classification_profile", "actual_name", actual_name, "predicted_name", predicted_name, output_prefix, options_value);
}

pub fn boolTransitionProfile(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceTrendOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputOptions(self, "bool_transition_profile", name, output_prefix, options_value);
}

pub fn rollingBoolTransitionProfile(self: anytype, name: []const u8, output_prefix: []const u8, transition_options: DeviceTrendOptions, options_value: DeviceRollingOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputExtraOptions(self, "rolling_bool_transition_profile", name, output_prefix, "transition_options", transition_options, options_value);
}

pub fn expandingBoolTransitionProfile(self: anytype, name: []const u8, output_prefix: []const u8, transition_options: DeviceTrendOptions, options_value: DeviceExpandingOptions) DeviceDataError!void {
    return lazy_profile_mod.appendNameOutputExtraOptions(self, "expanding_bool_transition_profile", name, output_prefix, "transition_options", transition_options, options_value);
}

pub fn rollingCorrelationProfile(
    self: anytype,
    x_name: []const u8,
    y_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceRollingCorrelationOptions,
) DeviceDataError!void {
    return lazy_profile_mod.appendPairOutputOptions(self, "rolling_correlation_profile", "x_name", x_name, "y_name", y_name, output_prefix, options_value);
}

pub fn expandingCorrelationProfile(
    self: anytype,
    x_name: []const u8,
    y_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceExpandingOptions,
) DeviceDataError!void {
    return lazy_profile_mod.appendPairOutputOptions(self, "expanding_correlation_profile", "x_name", x_name, "y_name", y_name, output_prefix, options_value);
}

pub fn expandingLinearFitProfile(
    self: anytype,
    x_name: []const u8,
    y_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceExpandingOptions,
) DeviceDataError!void {
    return lazy_profile_mod.appendPairOutputOptions(self, "expanding_linear_fit_profile", "x_name", x_name, "y_name", y_name, output_prefix, options_value);
}

pub fn rollingLinearFitProfile(
    self: anytype,
    x_name: []const u8,
    y_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceRollingCorrelationOptions,
) DeviceDataError!void {
    return lazy_profile_mod.appendPairOutputOptions(self, "rolling_linear_fit_profile", "x_name", x_name, "y_name", y_name, output_prefix, options_value);
}
