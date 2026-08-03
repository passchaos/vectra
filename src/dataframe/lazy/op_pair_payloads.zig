//! Pair/advanced profile payload type definitions for `DeviceLazyOp`.

const options_mod = @import("../../dataframe_options.zig");

const DeviceRollingOptions = options_mod.DeviceRollingOptions;
const DeviceExpandingOptions = options_mod.DeviceExpandingOptions;
const DeviceTrendOptions = options_mod.DeviceTrendOptions;
const DeviceCrossoverOptions = options_mod.DeviceCrossoverOptions;
const DeviceLinearFitOptions = options_mod.DeviceLinearFitOptions;
const DeviceRollingCorrelationOptions = options_mod.DeviceRollingCorrelationOptions;

pub const ActualPredictedOutput = struct {
    actual_name: []const u8,
    predicted_name: []const u8,
    output_prefix: []const u8,
};

pub const CrossoverProfile = struct {
    lhs_name: []const u8,
    rhs_name: []const u8,
    output_prefix: []const u8,
    options: DeviceCrossoverOptions,
};

pub const RollingCrossoverProfile = struct {
    lhs_name: []const u8,
    rhs_name: []const u8,
    output_prefix: []const u8,
    cross_options: DeviceCrossoverOptions,
    options: DeviceRollingOptions,
};

pub const ExpandingCrossoverProfile = struct {
    lhs_name: []const u8,
    rhs_name: []const u8,
    output_prefix: []const u8,
    cross_options: DeviceCrossoverOptions,
    options: DeviceExpandingOptions,
};

pub const LinearFitProfile = struct {
    x_name: []const u8,
    y_name: []const u8,
    output_prefix: []const u8,
    options: DeviceLinearFitOptions,
};

pub const PairOutput = ActualPredictedOutput;

pub const RollingPairOutput = struct {
    actual_name: []const u8,
    predicted_name: []const u8,
    output_prefix: []const u8,
    options: DeviceRollingOptions,
};

pub const ExpandingPairOutput = struct {
    actual_name: []const u8,
    predicted_name: []const u8,
    output_prefix: []const u8,
    options: DeviceExpandingOptions,
};

pub const RollingBoolTransitionProfile = struct {
    name: []const u8,
    output_prefix: []const u8,
    transition_options: DeviceTrendOptions,
    options: DeviceRollingOptions,
};

pub const ExpandingBoolTransitionProfile = struct {
    name: []const u8,
    output_prefix: []const u8,
    transition_options: DeviceTrendOptions,
    options: DeviceExpandingOptions,
};

pub const RollingCorrelationProfile = struct {
    x_name: []const u8,
    y_name: []const u8,
    output_prefix: []const u8,
    options: DeviceRollingCorrelationOptions,
};

pub const ExpandingXYProfile = struct {
    x_name: []const u8,
    y_name: []const u8,
    output_prefix: []const u8,
    options: DeviceExpandingOptions,
};

pub const RollingLinearFitProfile = struct {
    x_name: []const u8,
    y_name: []const u8,
    output_prefix: []const u8,
    options: DeviceRollingCorrelationOptions,
};
