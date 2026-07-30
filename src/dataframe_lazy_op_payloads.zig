//! Payload type definitions for `DeviceLazyOp`.
//!
//! The lazy operation union remains in `dataframe_lazy_op.zig`, but the payload
//! structs live here so adding a new lazy operation does not make the union file
//! absorb every field-level detail.

const options_mod = @import("dataframe_options.zig");
const profile_payloads = @import("dataframe_lazy_op_profile_payloads.zig");
const array_mod = @import("array.zig");

const DeviceColumnBinaryOp = options_mod.DeviceColumnBinaryOp;
const DeviceColumnCompareOp = options_mod.DeviceColumnCompareOp;
const DeviceDTypeClass = options_mod.DeviceDTypeClass;
const DeviceScalar = options_mod.DeviceScalar;
const DeviceSortOptions = options_mod.DeviceSortOptions;
const DeviceJoinOptions = options_mod.DeviceJoinOptions;
const DeviceAsofOptions = options_mod.DeviceAsofOptions;

pub const DeviceLazyGroupByAggregation = enum {
    sum,
    min,
    max,
    mean,
};

pub const DeviceLazyJoinKind = enum {
    inner,
    left,
    full,
    semi,
    anti,
};

pub fn DeviceLazyPayloads(comptime DeviceDataFrame: type, comptime DeviceColumn: type) type {
    return struct {
        pub const Select = [][]const u8;
        pub const NamePattern = struct {
            pattern: []const u8,
        };
        pub const DTypes = []array_mod.DType;
        pub const DTypeClass = DeviceDTypeClass;
        pub const RowIndex = struct {
            name: []const u8,
            offset: usize,
        };
        pub const RenameColumn = struct {
            old_name: []const u8,
            new_name: []const u8,
        };
        pub const MoveColumn = struct {
            name: []const u8,
            target_index: usize,
        };
        pub const MoveColumnRelative = struct {
            name: []const u8,
            anchor_name: []const u8,
        };
        pub const DropColumns = [][]const u8;
        pub const DropNulls = [][]const u8;
        pub const WithColumnBinary = struct {
            name: []const u8,
            lhs_name: []const u8,
            rhs_name: []const u8,
            op: DeviceColumnBinaryOp,
        };
        pub const WithColumnScalar = struct {
            name: []const u8,
            input_name: []const u8,
            op: DeviceColumnBinaryOp,
            scalar: DeviceScalar,
        };
        pub const WithColumnLiteral = struct {
            name: []const u8,
            scalar: DeviceScalar,
        };
        pub const CastColumn = struct {
            name: []const u8,
            dtype: array_mod.DType,
        };
        pub const FillNullColumn = struct {
            name: []const u8,
            scalar: DeviceScalar,
        };
        pub const WithColumnCompare = struct {
            name: []const u8,
            lhs_name: []const u8,
            rhs_name: []const u8,
            op: DeviceColumnCompareOp,
        };
        pub const WithColumnCompareScalar = struct {
            name: []const u8,
            input_name: []const u8,
            op: DeviceColumnCompareOp,
            scalar: DeviceScalar,
        };
        pub const FilterMask = DeviceColumn;
        pub const FilterColumn = []const u8;
        pub const FilterScalar = struct {
            name: []const u8,
            op: DeviceColumnCompareOp,
            scalar: DeviceScalar,
        };
        pub const GroupByCount = struct {
            key_name: []const u8,
            output_name: []const u8,
        };
        pub const GroupByValue = struct {
            key_name: []const u8,
            value_name: []const u8,
            output_name: []const u8,
            aggregation: DeviceLazyGroupByAggregation,
        };
        pub const GroupByOutput = struct {
            key_name: []const u8,
            value_name: []const u8,
            output_prefix: []const u8,
        };
        pub const GroupByOutputOn = struct {
            key_names: [][]const u8,
            value_name: []const u8,
            output_prefix: []const u8,
        };
        pub const JoinOn = struct {
            kind: DeviceLazyJoinKind,
            right: DeviceDataFrame,
            left_key_names: [][]const u8,
            right_key_names: [][]const u8,
            options: DeviceJoinOptions,
        };
        pub const AsofJoin = struct {
            right: DeviceDataFrame,
            left_key_name: []const u8,
            right_key_name: []const u8,
            options: DeviceAsofOptions,
        };
        pub const ConcatRows = DeviceDataFrame;
        pub const DistinctOn = [][]const u8;
        pub const SortBy = struct {
            name: []const u8,
            options: DeviceSortOptions,
        };
        pub const TopK = struct {
            name: []const u8,
            options: DeviceSortOptions,
            k: usize,
        };
        pub const RowSlice = struct {
            start: usize,
            stop: usize,
        };
        pub const RowTake = []usize;
        pub const RankProfileBy = profile_payloads.RankProfileBy;
        pub const RollingProfile = profile_payloads.RollingProfile;
        pub const ExpandingProfile = profile_payloads.ExpandingProfile;
        pub const RollingRobustProfile = profile_payloads.RollingRobustProfile;
        pub const RollingRankProfile = profile_payloads.RollingRankProfile;
        pub const LagProfile = profile_payloads.LagProfile;
        pub const ClipProfile = profile_payloads.ClipProfile;
        pub const RollingClipProfile = profile_payloads.RollingClipProfile;
        pub const ExpandingClipProfile = profile_payloads.ExpandingClipProfile;
        pub const ThresholdProfile = profile_payloads.ThresholdProfile;
        pub const RollingThresholdProfile = profile_payloads.RollingThresholdProfile;
        pub const ExpandingThresholdProfile = profile_payloads.ExpandingThresholdProfile;
        pub const ExpandingRankProfile = profile_payloads.ExpandingRankProfile;
        pub const ExpandingRobustProfile = profile_payloads.ExpandingRobustProfile;
        pub const StandardizeProfile = profile_payloads.StandardizeProfile;
        pub const RobustProfile = profile_payloads.RobustProfile;
        pub const DrawdownProfile = profile_payloads.DrawdownProfile;
        pub const ExtremaProfile = profile_payloads.ExtremaProfile;
        pub const TrendProfile = profile_payloads.TrendProfile;
        pub const RollingTrendProfile = profile_payloads.RollingTrendProfile;
        pub const ExpandingTrendProfile = profile_payloads.ExpandingTrendProfile;
        pub const ChangePointProfile = profile_payloads.ChangePointProfile;
        pub const RollingChangePointProfile = profile_payloads.RollingChangePointProfile;
        pub const ExpandingChangePointProfile = profile_payloads.ExpandingChangePointProfile;
        pub const RollingSignProfile = profile_payloads.RollingSignProfile;
        pub const ExpandingSignProfile = profile_payloads.ExpandingSignProfile;
        pub const CrossoverProfile = profile_payloads.CrossoverProfile;
        pub const RollingCrossoverProfile = profile_payloads.RollingCrossoverProfile;
        pub const ExpandingCrossoverProfile = profile_payloads.ExpandingCrossoverProfile;
        pub const BucketProfile = profile_payloads.BucketProfile;
        pub const EmaProfile = profile_payloads.EmaProfile;
        pub const LinearFitProfile = profile_payloads.LinearFitProfile;
        pub const PairOutput = profile_payloads.ActualPredictedOutput;
        pub const RollingPairOutput = profile_payloads.RollingPairOutput;
        pub const ExpandingPairOutput = profile_payloads.ExpandingPairOutput;
        pub const BoolTransitionProfile = profile_payloads.TrendProfile;
        pub const RollingBoolTransitionProfile = profile_payloads.RollingBoolTransitionProfile;
        pub const ExpandingBoolTransitionProfile = profile_payloads.ExpandingBoolTransitionProfile;
        pub const RollingCorrelationProfile = profile_payloads.RollingCorrelationProfile;
        pub const ExpandingXYProfile = profile_payloads.ExpandingXYProfile;
        pub const RollingLinearFitProfile = profile_payloads.RollingLinearFitProfile;
        pub const ValidityProfile = profile_payloads.NameOutput;
        pub const RollingValidityProfile = profile_payloads.RollingProfile;
        pub const ExpandingValidityProfile = profile_payloads.ExpandingProfile;
        pub const HeadTail = usize;
    };
}
