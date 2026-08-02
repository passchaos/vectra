//! Eager DeviceDataFrame group-by and join method wrappers.
//!
//! These wrappers dispatch to grouped aggregation and join implementation modules.
//! They live outside `dataframe.zig` so the public facade stays compact while
//! preserving method-call syntax through aliases on `DeviceDataFrame`.

const std = @import("std");
const array_mod = @import("array.zig");
const group_profile_mod = @import("dataframe_group_profile.zig");
const group_multi_mod = @import("dataframe_group_multi.zig");
const join_mod = @import("dataframe_join.zig");
const options_mod = @import("dataframe_options.zig");
const series_mod = @import("series.zig");

const DeviceDataError = series_mod.DataError || array_mod.ArrayError;
const DeviceJoinOptions = options_mod.DeviceJoinOptions;
const DeviceAsofOptions = options_mod.DeviceAsofOptions;

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

pub fn groupByCount(self: anytype, key_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_profile_mod.groupByCount(FrameType(@TypeOf(self)), frameValue(self), key_name, output_name);
}

pub fn groupByCountOn(self: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, output_name);
}

pub fn groupByHeadRows(self: anytype, key_name: []const u8, n: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByHeadRowsOn(self, key_names[0..], n);
}

pub fn groupByHeadRowsOn(self: anytype, key_names: []const []const u8, n: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByHeadRowsOn(FrameType(@TypeOf(self)), frameValue(self), key_names, n);
}

pub fn groupByTailRows(self: anytype, key_name: []const u8, n: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByTailRowsOn(self, key_names[0..], n);
}

pub fn groupByTailRowsOn(self: anytype, key_names: []const []const u8, n: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByTailRowsOn(FrameType(@TypeOf(self)), frameValue(self), key_names, n);
}

pub fn groupBySliceRows(self: anytype, key_name: []const u8, start: usize, length: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupBySliceRowsOn(self, key_names[0..], start, length);
}

pub fn groupBySliceRowsOn(self: anytype, key_names: []const []const u8, start: usize, length: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupBySliceRowsOn(FrameType(@TypeOf(self)), frameValue(self), key_names, start, length);
}

pub fn groupBySliceRowsStep(self: anytype, key_name: []const u8, start: usize, length: usize, step: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupBySliceRowsStepOn(self, key_names[0..], start, length, step);
}

pub fn groupBySliceRowsStepOn(self: anytype, key_names: []const []const u8, start: usize, length: usize, step: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupBySliceRowsStepOn(FrameType(@TypeOf(self)), frameValue(self), key_names, start, length, step);
}

pub fn groupBySliceRowsSigned(self: anytype, key_name: []const u8, start: isize, length: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupBySliceRowsSignedOn(self, key_names[0..], start, length);
}

pub fn groupBySliceRowsSignedOn(self: anytype, key_names: []const []const u8, start: isize, length: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupBySliceRowsSignedOn(FrameType(@TypeOf(self)), frameValue(self), key_names, start, length);
}

pub fn groupBySliceRowsSignedStep(self: anytype, key_name: []const u8, start: isize, length: usize, step: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupBySliceRowsSignedStepOn(self, key_names[0..], start, length, step);
}

pub fn groupBySliceRowsSignedStepOn(self: anytype, key_names: []const []const u8, start: isize, length: usize, step: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupBySliceRowsSignedStepOn(FrameType(@TypeOf(self)), frameValue(self), key_names, start, length, step);
}

pub fn groupByTopRows(self: anytype, key_name: []const u8, sort_name: []const u8, n: usize, options_value: options_mod.DeviceSortOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByTopRowsOn(self, key_names[0..], sort_name, n, options_value);
}

pub fn groupByTopRowsOn(self: anytype, key_names: []const []const u8, sort_name: []const u8, n: usize, options_value: options_mod.DeviceSortOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByTopRowsOn(FrameType(@TypeOf(self)), frameValue(self), key_names, sort_name, n, options_value);
}

pub fn groupByBottomRows(self: anytype, key_name: []const u8, sort_name: []const u8, n: usize, options_value: options_mod.DeviceSortOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByBottomRowsOn(self, key_names[0..], sort_name, n, options_value);
}

pub fn groupByBottomRowsOn(self: anytype, key_names: []const []const u8, sort_name: []const u8, n: usize, options_value: options_mod.DeviceSortOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByBottomRowsOn(FrameType(@TypeOf(self)), frameValue(self), key_names, sort_name, n, options_value);
}

pub fn groupByTopRowsByColumns(self: anytype, key_name: []const u8, sort_names: []const []const u8, n: usize, options_values: []const options_mod.DeviceSortOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByTopRowsByColumnsOn(self, key_names[0..], sort_names, n, options_values);
}

pub fn groupByTopRowsByColumnsOn(self: anytype, key_names: []const []const u8, sort_names: []const []const u8, n: usize, options_values: []const options_mod.DeviceSortOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByTopRowsByColumnsOn(FrameType(@TypeOf(self)), frameValue(self), key_names, sort_names, n, options_values);
}

pub fn groupByBottomRowsByColumns(self: anytype, key_name: []const u8, sort_names: []const []const u8, n: usize, options_values: []const options_mod.DeviceSortOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByBottomRowsByColumnsOn(self, key_names[0..], sort_names, n, options_values);
}

pub fn groupByBottomRowsByColumnsOn(self: anytype, key_names: []const []const u8, sort_names: []const []const u8, n: usize, options_values: []const options_mod.DeviceSortOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByBottomRowsByColumnsOn(FrameType(@TypeOf(self)), frameValue(self), key_names, sort_names, n, options_values);
}

pub fn withGroupId(self: anytype, key_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupIdOn(self, key_names[0..], output_name);
}

pub fn withGroupIdOn(self: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupIdOn(FrameType(@TypeOf(self)), frameValue(self), key_names, output_name);
}

pub const withGroupIndex = withGroupId;
pub const withGroupIndexOn = withGroupIdOn;

pub fn withGroupFirstRowIndex(self: anytype, key_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupFirstRowIndexOn(self, key_names[0..], output_name);
}

pub fn withGroupFirstRowIndexOn(self: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupFirstRowIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, output_name);
}

pub fn withGroupLastRowIndex(self: anytype, key_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupLastRowIndexOn(self, key_names[0..], output_name);
}

pub fn withGroupLastRowIndexOn(self: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupLastRowIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, output_name);
}

pub fn withGroupIsFirstRow(self: anytype, key_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupIsFirstRowOn(self, key_names[0..], output_name);
}

pub fn withGroupIsFirstRowOn(self: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupIsFirstRowOn(FrameType(@TypeOf(self)), frameValue(self), key_names, output_name);
}

pub fn withGroupIsLastRow(self: anytype, key_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupIsLastRowOn(self, key_names[0..], output_name);
}

pub fn withGroupIsLastRowOn(self: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupIsLastRowOn(FrameType(@TypeOf(self)), frameValue(self), key_names, output_name);
}

pub fn withGroupIsSingleton(self: anytype, key_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupIsSingletonOn(self, key_names[0..], output_name);
}

pub fn withGroupIsSingletonOn(self: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupIsSingletonOn(FrameType(@TypeOf(self)), frameValue(self), key_names, output_name);
}

pub fn withGroupIsDuplicated(self: anytype, key_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupIsDuplicatedOn(self, key_names[0..], output_name);
}

pub fn withGroupIsDuplicatedOn(self: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupIsDuplicatedOn(FrameType(@TypeOf(self)), frameValue(self), key_names, output_name);
}

pub fn withGroupCumeDist(self: anytype, key_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumeDistOn(self, key_names[0..], output_name);
}

pub fn withGroupCumeDistOn(self: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumeDistOn(FrameType(@TypeOf(self)), frameValue(self), key_names, output_name);
}

pub const withGroupCumulativeDistribution = withGroupCumeDist;
pub const withGroupCumulativeDistributionOn = withGroupCumeDistOn;

pub fn withGroupPercentRank(self: anytype, key_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupPercentRankOn(self, key_names[0..], output_name);
}

pub fn withGroupPercentRankOn(self: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupPercentRankOn(FrameType(@TypeOf(self)), frameValue(self), key_names, output_name);
}

pub const withGroupPercentileRank = withGroupPercentRank;
pub const withGroupPercentileRankOn = withGroupPercentRankOn;

pub fn withGroupReverseCumeDist(self: anytype, key_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupReverseCumeDistOn(self, key_names[0..], output_name);
}

pub fn withGroupReverseCumeDistOn(self: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupReverseCumeDistOn(FrameType(@TypeOf(self)), frameValue(self), key_names, output_name);
}

pub const withGroupReverseCumulativeDistribution = withGroupReverseCumeDist;
pub const withGroupReverseCumulativeDistributionOn = withGroupReverseCumeDistOn;

pub fn withGroupReversePercentRank(self: anytype, key_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupReversePercentRankOn(self, key_names[0..], output_name);
}

pub fn withGroupReversePercentRankOn(self: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupReversePercentRankOn(FrameType(@TypeOf(self)), frameValue(self), key_names, output_name);
}

pub const withGroupReversePercentileRank = withGroupReversePercentRank;
pub const withGroupReversePercentileRankOn = withGroupReversePercentRankOn;

pub fn withGroupLag(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8, offset: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupLagOn(self, key_names[0..], value_name, output_name, offset);
}

pub fn withGroupLagOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, offset: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupLagOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name, offset);
}

pub fn withGroupLead(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8, offset: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupLeadOn(self, key_names[0..], value_name, output_name, offset);
}

pub fn withGroupLeadOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, offset: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupLeadOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name, offset);
}

pub fn withGroupFirstRowValue(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupFirstRowValueOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupFirstRowValueOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupFirstRowValueOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupLastRowValue(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupLastRowValueOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupLastRowValueOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupLastRowValueOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupNthRowValue(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8, n: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupNthRowValueOn(self, key_names[0..], value_name, output_name, n);
}

pub fn withGroupNthRowValueOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, n: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupNthRowValueOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name, n);
}

pub const withGroupNthValue = withGroupNthRowValue;
pub const withGroupNthValueOn = withGroupNthRowValueOn;

pub fn withGroupFirstValidValue(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupFirstValidValueOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupFirstValidValueOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupFirstValidValueOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupLastValidValue(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupLastValidValueOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupLastValidValueOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupLastValidValueOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupNthValidValue(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8, n: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupNthValidValueOn(self, key_names[0..], value_name, output_name, n);
}

pub fn withGroupNthValidValueOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, n: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupNthValidValueOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name, n);
}

pub fn withGroupFillNullForward(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupFillNullForwardOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupFillNullForwardOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupFillNullForwardOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupFillNullBackward(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupFillNullBackwardOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupFillNullBackwardOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupFillNullBackwardOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeValidCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeValidCountOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeValidCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeValidCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeNullCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeNullCountOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeNullCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeNullCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const withGroupCumValidCount = withGroupCumulativeValidCount;
pub const withGroupCumValidCountOn = withGroupCumulativeValidCountOn;
pub const withGroupCumNullCount = withGroupCumulativeNullCount;
pub const withGroupCumNullCountOn = withGroupCumulativeNullCountOn;

pub fn withGroupCumulativeValidRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeValidRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeValidRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeValidRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeNullRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeNullRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeNullRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeNullRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const withGroupCumValidRatio = withGroupCumulativeValidRatio;
pub const withGroupCumValidRatioOn = withGroupCumulativeValidRatioOn;
pub const withGroupCumNullRatio = withGroupCumulativeNullRatio;
pub const withGroupCumNullRatioOn = withGroupCumulativeNullRatioOn;

pub fn withGroupCumulativeFirstValidIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFirstValidIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFirstValidIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeFirstValidIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeLastValidIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeLastValidIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeLastValidIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeLastValidIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeFirstNullIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFirstNullIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFirstNullIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeFirstNullIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeLastNullIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeLastNullIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeLastNullIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeLastNullIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const withGroupCumFirstValidIndex = withGroupCumulativeFirstValidIndex;
pub const withGroupCumFirstValidIndexOn = withGroupCumulativeFirstValidIndexOn;
pub const withGroupCumLastValidIndex = withGroupCumulativeLastValidIndex;
pub const withGroupCumLastValidIndexOn = withGroupCumulativeLastValidIndexOn;
pub const withGroupCumFirstNullIndex = withGroupCumulativeFirstNullIndex;
pub const withGroupCumFirstNullIndexOn = withGroupCumulativeFirstNullIndexOn;
pub const withGroupCumLastNullIndex = withGroupCumulativeLastNullIndex;
pub const withGroupCumLastNullIndexOn = withGroupCumulativeLastNullIndexOn;

pub fn withGroupCumulativeNaNCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeNaNCountOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeNaNCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeNaNCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeNaNRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeNaNRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeNaNRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeNaNRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeInfCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeInfCountOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeInfCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeInfCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeInfRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeInfRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeInfRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeInfRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativePositiveInfCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativePositiveInfCountOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativePositiveInfCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativePositiveInfCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativePositiveInfRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativePositiveInfRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativePositiveInfRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativePositiveInfRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeNegativeInfCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeNegativeInfCountOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeNegativeInfCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeNegativeInfCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeNegativeInfRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeNegativeInfRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeNegativeInfRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeNegativeInfRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeFiniteCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFiniteCountOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFiniteCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeFiniteCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeFiniteRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFiniteRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFiniteRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeFiniteRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeNormalCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeNormalCountOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeNormalCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeNormalCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeNormalRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeNormalRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeNormalRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeNormalRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeSubnormalCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeSubnormalCountOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeSubnormalCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeSubnormalCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeSubnormalRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeSubnormalRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeSubnormalRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeSubnormalRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeNonFiniteCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeNonFiniteCountOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeNonFiniteCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeNonFiniteCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeNonFiniteRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeNonFiniteRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeNonFiniteRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeNonFiniteRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeZeroCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeZeroCountOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeZeroCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeZeroCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeZeroRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeZeroRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeZeroRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeZeroRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativePositiveZeroCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativePositiveZeroCountOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativePositiveZeroCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativePositiveZeroCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativePositiveZeroRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativePositiveZeroRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativePositiveZeroRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativePositiveZeroRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeNegativeZeroCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeNegativeZeroCountOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeNegativeZeroCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeNegativeZeroCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeNegativeZeroRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeNegativeZeroRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeNegativeZeroRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeNegativeZeroRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeNonZeroCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeNonZeroCountOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeNonZeroCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeNonZeroCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeNonZeroRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeNonZeroRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeNonZeroRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeNonZeroRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativePositiveCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativePositiveCountOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativePositiveCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativePositiveCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativePositiveRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativePositiveRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativePositiveRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativePositiveRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeSignBitCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeSignBitCountOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeSignBitCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeSignBitCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeSignBitRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeSignBitRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeSignBitRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeSignBitRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeNegativeCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeNegativeCountOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeNegativeCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeNegativeCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeNegativeRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeNegativeRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeNegativeRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeNegativeRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const withGroupCumNaNCount = withGroupCumulativeNaNCount;

pub const withGroupCumNaNCountOn = withGroupCumulativeNaNCountOn;

pub const withGroupCumNaNRatio = withGroupCumulativeNaNRatio;

pub const withGroupCumNaNRatioOn = withGroupCumulativeNaNRatioOn;

pub const withGroupCumulativeNanCount = withGroupCumulativeNaNCount;

pub const withGroupCumulativeNanCountOn = withGroupCumulativeNaNCountOn;

pub const withGroupCumNanCount = withGroupCumulativeNaNCount;

pub const withGroupCumNanCountOn = withGroupCumulativeNaNCountOn;

pub const withGroupCumulativeNanRatio = withGroupCumulativeNaNRatio;

pub const withGroupCumulativeNanRatioOn = withGroupCumulativeNaNRatioOn;

pub const withGroupCumNanRatio = withGroupCumulativeNaNRatio;

pub const withGroupCumNanRatioOn = withGroupCumulativeNaNRatioOn;

pub const withGroupCumInfCount = withGroupCumulativeInfCount;

pub const withGroupCumInfCountOn = withGroupCumulativeInfCountOn;

pub const withGroupCumInfRatio = withGroupCumulativeInfRatio;

pub const withGroupCumInfRatioOn = withGroupCumulativeInfRatioOn;

pub const withGroupCumPositiveInfCount = withGroupCumulativePositiveInfCount;

pub const withGroupCumPositiveInfCountOn = withGroupCumulativePositiveInfCountOn;

pub const withGroupCumPositiveInfRatio = withGroupCumulativePositiveInfRatio;

pub const withGroupCumPositiveInfRatioOn = withGroupCumulativePositiveInfRatioOn;

pub const withGroupCumNegativeInfCount = withGroupCumulativeNegativeInfCount;

pub const withGroupCumNegativeInfCountOn = withGroupCumulativeNegativeInfCountOn;

pub const withGroupCumNegativeInfRatio = withGroupCumulativeNegativeInfRatio;

pub const withGroupCumNegativeInfRatioOn = withGroupCumulativeNegativeInfRatioOn;

pub const withGroupCumFiniteCount = withGroupCumulativeFiniteCount;

pub const withGroupCumFiniteCountOn = withGroupCumulativeFiniteCountOn;

pub const withGroupCumFiniteRatio = withGroupCumulativeFiniteRatio;

pub const withGroupCumFiniteRatioOn = withGroupCumulativeFiniteRatioOn;

pub const withGroupCumNormalCount = withGroupCumulativeNormalCount;

pub const withGroupCumNormalCountOn = withGroupCumulativeNormalCountOn;

pub const withGroupCumNormalRatio = withGroupCumulativeNormalRatio;

pub const withGroupCumNormalRatioOn = withGroupCumulativeNormalRatioOn;

pub const withGroupCumSubnormalCount = withGroupCumulativeSubnormalCount;

pub const withGroupCumSubnormalCountOn = withGroupCumulativeSubnormalCountOn;

pub const withGroupCumSubnormalRatio = withGroupCumulativeSubnormalRatio;

pub const withGroupCumSubnormalRatioOn = withGroupCumulativeSubnormalRatioOn;

pub const withGroupCumNonFiniteCount = withGroupCumulativeNonFiniteCount;

pub const withGroupCumNonFiniteCountOn = withGroupCumulativeNonFiniteCountOn;

pub const withGroupCumNonFiniteRatio = withGroupCumulativeNonFiniteRatio;

pub const withGroupCumNonFiniteRatioOn = withGroupCumulativeNonFiniteRatioOn;

pub const withGroupCumZeroCount = withGroupCumulativeZeroCount;

pub const withGroupCumZeroCountOn = withGroupCumulativeZeroCountOn;

pub const withGroupCumZeroRatio = withGroupCumulativeZeroRatio;

pub const withGroupCumZeroRatioOn = withGroupCumulativeZeroRatioOn;

pub const withGroupCumPositiveZeroCount = withGroupCumulativePositiveZeroCount;

pub const withGroupCumPositiveZeroCountOn = withGroupCumulativePositiveZeroCountOn;

pub const withGroupCumPositiveZeroRatio = withGroupCumulativePositiveZeroRatio;

pub const withGroupCumPositiveZeroRatioOn = withGroupCumulativePositiveZeroRatioOn;

pub const withGroupCumNegativeZeroCount = withGroupCumulativeNegativeZeroCount;

pub const withGroupCumNegativeZeroCountOn = withGroupCumulativeNegativeZeroCountOn;

pub const withGroupCumNegativeZeroRatio = withGroupCumulativeNegativeZeroRatio;

pub const withGroupCumNegativeZeroRatioOn = withGroupCumulativeNegativeZeroRatioOn;

pub const withGroupCumNonZeroCount = withGroupCumulativeNonZeroCount;

pub const withGroupCumNonZeroCountOn = withGroupCumulativeNonZeroCountOn;

pub const withGroupCumNonZeroRatio = withGroupCumulativeNonZeroRatio;

pub const withGroupCumNonZeroRatioOn = withGroupCumulativeNonZeroRatioOn;

pub const withGroupCumPositiveCount = withGroupCumulativePositiveCount;

pub const withGroupCumPositiveCountOn = withGroupCumulativePositiveCountOn;

pub const withGroupCumPositiveRatio = withGroupCumulativePositiveRatio;

pub const withGroupCumPositiveRatioOn = withGroupCumulativePositiveRatioOn;

pub const withGroupCumSignBitCount = withGroupCumulativeSignBitCount;

pub const withGroupCumSignBitCountOn = withGroupCumulativeSignBitCountOn;

pub const withGroupCumSignBitRatio = withGroupCumulativeSignBitRatio;

pub const withGroupCumSignBitRatioOn = withGroupCumulativeSignBitRatioOn;

pub const withGroupCumNegativeCount = withGroupCumulativeNegativeCount;

pub const withGroupCumNegativeCountOn = withGroupCumulativeNegativeCountOn;

pub const withGroupCumNegativeRatio = withGroupCumulativeNegativeRatio;

pub const withGroupCumNegativeRatioOn = withGroupCumulativeNegativeRatioOn;

pub fn withGroupCumulativeFirstNaNIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFirstNaNIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFirstNaNIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeFirstNaNIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeLastNaNIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeLastNaNIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeLastNaNIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeLastNaNIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeFirstInfIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFirstInfIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFirstInfIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeFirstInfIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeLastInfIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeLastInfIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeLastInfIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeLastInfIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeFirstPositiveInfIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFirstPositiveInfIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFirstPositiveInfIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeFirstPositiveInfIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeLastPositiveInfIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeLastPositiveInfIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeLastPositiveInfIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeLastPositiveInfIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeFirstNegativeInfIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFirstNegativeInfIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFirstNegativeInfIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeFirstNegativeInfIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeLastNegativeInfIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeLastNegativeInfIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeLastNegativeInfIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeLastNegativeInfIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeFirstFiniteIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFirstFiniteIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFirstFiniteIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeFirstFiniteIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeLastFiniteIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeLastFiniteIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeLastFiniteIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeLastFiniteIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeFirstNormalIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFirstNormalIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFirstNormalIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeFirstNormalIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeLastNormalIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeLastNormalIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeLastNormalIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeLastNormalIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeFirstSubnormalIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFirstSubnormalIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFirstSubnormalIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeFirstSubnormalIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeLastSubnormalIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeLastSubnormalIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeLastSubnormalIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeLastSubnormalIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeFirstNonFiniteIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFirstNonFiniteIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFirstNonFiniteIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeFirstNonFiniteIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeLastNonFiniteIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeLastNonFiniteIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeLastNonFiniteIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeLastNonFiniteIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeFirstZeroIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFirstZeroIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFirstZeroIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeFirstZeroIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeLastZeroIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeLastZeroIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeLastZeroIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeLastZeroIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeFirstPositiveZeroIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFirstPositiveZeroIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFirstPositiveZeroIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeFirstPositiveZeroIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeLastPositiveZeroIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeLastPositiveZeroIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeLastPositiveZeroIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeLastPositiveZeroIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeFirstNegativeZeroIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFirstNegativeZeroIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFirstNegativeZeroIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeFirstNegativeZeroIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeLastNegativeZeroIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeLastNegativeZeroIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeLastNegativeZeroIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeLastNegativeZeroIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeFirstNonZeroIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFirstNonZeroIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFirstNonZeroIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeFirstNonZeroIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeLastNonZeroIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeLastNonZeroIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeLastNonZeroIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeLastNonZeroIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeFirstPositiveIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFirstPositiveIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFirstPositiveIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeFirstPositiveIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeLastPositiveIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeLastPositiveIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeLastPositiveIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeLastPositiveIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeFirstSignBitIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFirstSignBitIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFirstSignBitIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeFirstSignBitIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeLastSignBitIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeLastSignBitIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeLastSignBitIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeLastSignBitIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeFirstNegativeIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFirstNegativeIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFirstNegativeIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeFirstNegativeIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeLastNegativeIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeLastNegativeIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeLastNegativeIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeLastNegativeIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const withGroupCumFirstNaNIndex = withGroupCumulativeFirstNaNIndex;
pub const withGroupCumFirstNaNIndexOn = withGroupCumulativeFirstNaNIndexOn;
pub const withGroupCumLastNaNIndex = withGroupCumulativeLastNaNIndex;
pub const withGroupCumLastNaNIndexOn = withGroupCumulativeLastNaNIndexOn;
pub const withGroupCumulativeFirstNanIndex = withGroupCumulativeFirstNaNIndex;
pub const withGroupCumulativeFirstNanIndexOn = withGroupCumulativeFirstNaNIndexOn;
pub const withGroupCumulativeLastNanIndex = withGroupCumulativeLastNaNIndex;
pub const withGroupCumulativeLastNanIndexOn = withGroupCumulativeLastNaNIndexOn;
pub const withGroupCumFirstNanIndex = withGroupCumulativeFirstNaNIndex;
pub const withGroupCumFirstNanIndexOn = withGroupCumulativeFirstNaNIndexOn;
pub const withGroupCumLastNanIndex = withGroupCumulativeLastNaNIndex;
pub const withGroupCumLastNanIndexOn = withGroupCumulativeLastNaNIndexOn;
pub const withGroupCumFirstInfIndex = withGroupCumulativeFirstInfIndex;
pub const withGroupCumFirstInfIndexOn = withGroupCumulativeFirstInfIndexOn;
pub const withGroupCumLastInfIndex = withGroupCumulativeLastInfIndex;
pub const withGroupCumLastInfIndexOn = withGroupCumulativeLastInfIndexOn;
pub const withGroupCumFirstPositiveInfIndex = withGroupCumulativeFirstPositiveInfIndex;
pub const withGroupCumFirstPositiveInfIndexOn = withGroupCumulativeFirstPositiveInfIndexOn;
pub const withGroupCumLastPositiveInfIndex = withGroupCumulativeLastPositiveInfIndex;
pub const withGroupCumLastPositiveInfIndexOn = withGroupCumulativeLastPositiveInfIndexOn;
pub const withGroupCumFirstNegativeInfIndex = withGroupCumulativeFirstNegativeInfIndex;
pub const withGroupCumFirstNegativeInfIndexOn = withGroupCumulativeFirstNegativeInfIndexOn;
pub const withGroupCumLastNegativeInfIndex = withGroupCumulativeLastNegativeInfIndex;
pub const withGroupCumLastNegativeInfIndexOn = withGroupCumulativeLastNegativeInfIndexOn;
pub const withGroupCumFirstFiniteIndex = withGroupCumulativeFirstFiniteIndex;
pub const withGroupCumFirstFiniteIndexOn = withGroupCumulativeFirstFiniteIndexOn;
pub const withGroupCumLastFiniteIndex = withGroupCumulativeLastFiniteIndex;
pub const withGroupCumLastFiniteIndexOn = withGroupCumulativeLastFiniteIndexOn;
pub const withGroupCumFirstNormalIndex = withGroupCumulativeFirstNormalIndex;
pub const withGroupCumFirstNormalIndexOn = withGroupCumulativeFirstNormalIndexOn;
pub const withGroupCumLastNormalIndex = withGroupCumulativeLastNormalIndex;
pub const withGroupCumLastNormalIndexOn = withGroupCumulativeLastNormalIndexOn;
pub const withGroupCumFirstSubnormalIndex = withGroupCumulativeFirstSubnormalIndex;
pub const withGroupCumFirstSubnormalIndexOn = withGroupCumulativeFirstSubnormalIndexOn;
pub const withGroupCumLastSubnormalIndex = withGroupCumulativeLastSubnormalIndex;
pub const withGroupCumLastSubnormalIndexOn = withGroupCumulativeLastSubnormalIndexOn;
pub const withGroupCumFirstNonFiniteIndex = withGroupCumulativeFirstNonFiniteIndex;
pub const withGroupCumFirstNonFiniteIndexOn = withGroupCumulativeFirstNonFiniteIndexOn;
pub const withGroupCumLastNonFiniteIndex = withGroupCumulativeLastNonFiniteIndex;
pub const withGroupCumLastNonFiniteIndexOn = withGroupCumulativeLastNonFiniteIndexOn;
pub const withGroupCumFirstZeroIndex = withGroupCumulativeFirstZeroIndex;
pub const withGroupCumFirstZeroIndexOn = withGroupCumulativeFirstZeroIndexOn;
pub const withGroupCumLastZeroIndex = withGroupCumulativeLastZeroIndex;
pub const withGroupCumLastZeroIndexOn = withGroupCumulativeLastZeroIndexOn;
pub const withGroupCumFirstPositiveZeroIndex = withGroupCumulativeFirstPositiveZeroIndex;
pub const withGroupCumFirstPositiveZeroIndexOn = withGroupCumulativeFirstPositiveZeroIndexOn;
pub const withGroupCumLastPositiveZeroIndex = withGroupCumulativeLastPositiveZeroIndex;
pub const withGroupCumLastPositiveZeroIndexOn = withGroupCumulativeLastPositiveZeroIndexOn;
pub const withGroupCumFirstNegativeZeroIndex = withGroupCumulativeFirstNegativeZeroIndex;
pub const withGroupCumFirstNegativeZeroIndexOn = withGroupCumulativeFirstNegativeZeroIndexOn;
pub const withGroupCumLastNegativeZeroIndex = withGroupCumulativeLastNegativeZeroIndex;
pub const withGroupCumLastNegativeZeroIndexOn = withGroupCumulativeLastNegativeZeroIndexOn;
pub const withGroupCumFirstNonZeroIndex = withGroupCumulativeFirstNonZeroIndex;
pub const withGroupCumFirstNonZeroIndexOn = withGroupCumulativeFirstNonZeroIndexOn;
pub const withGroupCumLastNonZeroIndex = withGroupCumulativeLastNonZeroIndex;
pub const withGroupCumLastNonZeroIndexOn = withGroupCumulativeLastNonZeroIndexOn;
pub const withGroupCumFirstPositiveIndex = withGroupCumulativeFirstPositiveIndex;
pub const withGroupCumFirstPositiveIndexOn = withGroupCumulativeFirstPositiveIndexOn;
pub const withGroupCumLastPositiveIndex = withGroupCumulativeLastPositiveIndex;
pub const withGroupCumLastPositiveIndexOn = withGroupCumulativeLastPositiveIndexOn;
pub const withGroupCumFirstSignBitIndex = withGroupCumulativeFirstSignBitIndex;
pub const withGroupCumFirstSignBitIndexOn = withGroupCumulativeFirstSignBitIndexOn;
pub const withGroupCumLastSignBitIndex = withGroupCumulativeLastSignBitIndex;
pub const withGroupCumLastSignBitIndexOn = withGroupCumulativeLastSignBitIndexOn;
pub const withGroupCumFirstNegativeIndex = withGroupCumulativeFirstNegativeIndex;
pub const withGroupCumFirstNegativeIndexOn = withGroupCumulativeFirstNegativeIndexOn;
pub const withGroupCumLastNegativeIndex = withGroupCumulativeLastNegativeIndex;
pub const withGroupCumLastNegativeIndexOn = withGroupCumulativeLastNegativeIndexOn;

pub fn withGroupCumulativeDistinctCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeDistinctCountOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeDistinctCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeDistinctCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeNUnique(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeNUniqueOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeNUniqueOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeNUniqueOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const withGroupCumulativeCountDistinct = withGroupCumulativeDistinctCount;
pub const withGroupCumulativeCountDistinctOn = withGroupCumulativeDistinctCountOn;
pub const withGroupCumulativeNunique = withGroupCumulativeNUnique;
pub const withGroupCumulativeNuniqueOn = withGroupCumulativeNUniqueOn;
pub const withGroupCumDistinctCount = withGroupCumulativeDistinctCount;
pub const withGroupCumDistinctCountOn = withGroupCumulativeDistinctCountOn;
pub const withGroupCumCountDistinct = withGroupCumulativeDistinctCount;
pub const withGroupCumCountDistinctOn = withGroupCumulativeDistinctCountOn;
pub const withGroupCumNUnique = withGroupCumulativeNUnique;
pub const withGroupCumNUniqueOn = withGroupCumulativeNUniqueOn;
pub const withGroupCumNunique = withGroupCumulativeNUnique;
pub const withGroupCumNuniqueOn = withGroupCumulativeNUniqueOn;

pub fn withGroupCumulativeMode(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeModeOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeModeOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeModeOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeModeCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeModeCountOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeModeCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeModeCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeModeRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeModeRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeModeRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeModeRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeModeMargin(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeModeMarginOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeModeMarginOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeModeMarginOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeModeMarginRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeModeMarginRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeModeMarginRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeModeMarginRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const withGroupCumMode = withGroupCumulativeMode;
pub const withGroupCumModeOn = withGroupCumulativeModeOn;
pub const withGroupCumModeCount = withGroupCumulativeModeCount;
pub const withGroupCumModeCountOn = withGroupCumulativeModeCountOn;
pub const withGroupCumModeRatio = withGroupCumulativeModeRatio;
pub const withGroupCumModeRatioOn = withGroupCumulativeModeRatioOn;
pub const withGroupCumModeMargin = withGroupCumulativeModeMargin;
pub const withGroupCumModeMarginOn = withGroupCumulativeModeMarginOn;
pub const withGroupCumModeMarginRatio = withGroupCumulativeModeMarginRatio;
pub const withGroupCumModeMarginRatioOn = withGroupCumulativeModeMarginRatioOn;

pub fn withGroupCumulativeAny(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeAnyOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeAnyOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeAnyOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeAll(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeAllOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeAllOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeAllOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeTrueCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeTrueCountOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeTrueCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeTrueCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeFalseCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFalseCountOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFalseCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeFalseCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeTrueRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeTrueRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeTrueRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeTrueRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeFalseRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFalseRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFalseRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeFalseRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const withGroupCumAny = withGroupCumulativeAny;
pub const withGroupCumAnyOn = withGroupCumulativeAnyOn;
pub const withGroupCumAll = withGroupCumulativeAll;
pub const withGroupCumAllOn = withGroupCumulativeAllOn;
pub const withGroupCumTrueCount = withGroupCumulativeTrueCount;
pub const withGroupCumTrueCountOn = withGroupCumulativeTrueCountOn;
pub const withGroupCumFalseCount = withGroupCumulativeFalseCount;
pub const withGroupCumFalseCountOn = withGroupCumulativeFalseCountOn;
pub const withGroupCumTrueRatio = withGroupCumulativeTrueRatio;
pub const withGroupCumTrueRatioOn = withGroupCumulativeTrueRatioOn;
pub const withGroupCumFalseRatio = withGroupCumulativeFalseRatio;
pub const withGroupCumFalseRatioOn = withGroupCumulativeFalseRatioOn;

pub fn withGroupCumulativeFirstTrueIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFirstTrueIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFirstTrueIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeFirstTrueIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeLastTrueIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeLastTrueIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeLastTrueIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeLastTrueIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeFirstFalseIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFirstFalseIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFirstFalseIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeFirstFalseIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeLastFalseIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeLastFalseIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeLastFalseIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeLastFalseIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const withGroupCumFirstTrueIndex = withGroupCumulativeFirstTrueIndex;
pub const withGroupCumFirstTrueIndexOn = withGroupCumulativeFirstTrueIndexOn;
pub const withGroupCumLastTrueIndex = withGroupCumulativeLastTrueIndex;
pub const withGroupCumLastTrueIndexOn = withGroupCumulativeLastTrueIndexOn;
pub const withGroupCumFirstFalseIndex = withGroupCumulativeFirstFalseIndex;
pub const withGroupCumFirstFalseIndexOn = withGroupCumulativeFirstFalseIndexOn;
pub const withGroupCumLastFalseIndex = withGroupCumulativeLastFalseIndex;
pub const withGroupCumLastFalseIndexOn = withGroupCumulativeLastFalseIndexOn;

pub fn withGroupCumulativeSum(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeSumOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeSumOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeSumOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const withGroupCumSum = withGroupCumulativeSum;
pub const withGroupCumSumOn = withGroupCumulativeSumOn;

pub fn withGroupCumulativeMean(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeMeanOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeMeanOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeMeanOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const withGroupCumMean = withGroupCumulativeMean;
pub const withGroupCumMeanOn = withGroupCumulativeMeanOn;

pub fn withGroupCumulativeProduct(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeProductOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeProductOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeProductOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const withGroupCumProduct = withGroupCumulativeProduct;
pub const withGroupCumProductOn = withGroupCumulativeProductOn;
pub const withGroupCumProd = withGroupCumulativeProduct;
pub const withGroupCumProdOn = withGroupCumulativeProductOn;

pub fn withGroupCumulativeMin(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeMinOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeMinOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeMinOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeMax(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeMaxOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeMaxOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeMaxOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeVariance(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeVarianceOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeVarianceOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeVarianceOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeStddev(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeStddevOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeStddevOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeStddevOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeSem(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeSemOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeSemOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeSemOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeCv(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeCvOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeCvOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeCvOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeFano(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFanoOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFanoOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeFanoOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeSkewness(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeSkewnessOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeSkewnessOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeSkewnessOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeKurtosis(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeKurtosisOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeKurtosisOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeKurtosisOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeMeanAbs(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeMeanAbsOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeMeanAbsOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeMeanAbsOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeMeanSquare(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeMeanSquareOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeMeanSquareOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeMeanSquareOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeRms(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeRmsOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeRmsOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeRmsOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeMaxAbs(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeMaxAbsOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeMaxAbsOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeMaxAbsOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeMinAbs(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeMinAbsOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeMinAbsOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeMinAbsOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeL1Norm(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeL1NormOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeL1NormOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeL1NormOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeL2Norm(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeL2NormOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeL2NormOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeL2NormOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeRange(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeRangeOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeRangeOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeRangeOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeMidrange(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeMidrangeOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeMidrangeOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeMidrangeOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeRangeCoeff(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeRangeCoeffOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeRangeCoeffOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeRangeCoeffOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeLogSumExp(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeLogSumExpOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeLogSumExpOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeLogSumExpOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeLogMeanExp(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeLogMeanExpOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeLogMeanExpOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeLogMeanExpOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeGeometricMean(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeGeometricMeanOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeGeometricMeanOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeGeometricMeanOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeHarmonicMean(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeHarmonicMeanOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeHarmonicMeanOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeHarmonicMeanOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeArgMin(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeArgMinOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeArgMinOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeArgMinOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn withGroupCumulativeArgMax(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeArgMaxOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeArgMaxOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupCumulativeArgMaxOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const withGroupCumMin = withGroupCumulativeMin;
pub const withGroupCumMinOn = withGroupCumulativeMinOn;
pub const withGroupCumMax = withGroupCumulativeMax;
pub const withGroupCumMaxOn = withGroupCumulativeMaxOn;
pub const withGroupCumulativeVar = withGroupCumulativeVariance;
pub const withGroupCumulativeVarOn = withGroupCumulativeVarianceOn;
pub const withGroupCumVariance = withGroupCumulativeVariance;
pub const withGroupCumVarianceOn = withGroupCumulativeVarianceOn;
pub const withGroupCumVar = withGroupCumulativeVariance;
pub const withGroupCumVarOn = withGroupCumulativeVarianceOn;
pub const withGroupCumulativeStd = withGroupCumulativeStddev;
pub const withGroupCumulativeStdOn = withGroupCumulativeStddevOn;
pub const withGroupCumStddev = withGroupCumulativeStddev;
pub const withGroupCumStddevOn = withGroupCumulativeStddevOn;
pub const withGroupCumStd = withGroupCumulativeStddev;
pub const withGroupCumStdOn = withGroupCumulativeStddevOn;
pub const withGroupCumulativeSEM = withGroupCumulativeSem;
pub const withGroupCumulativeSEMOn = withGroupCumulativeSemOn;
pub const withGroupCumSem = withGroupCumulativeSem;
pub const withGroupCumSemOn = withGroupCumulativeSemOn;
pub const withGroupCumulativeCV = withGroupCumulativeCv;
pub const withGroupCumulativeCVOn = withGroupCumulativeCvOn;
pub const withGroupCumCv = withGroupCumulativeCv;
pub const withGroupCumCvOn = withGroupCumulativeCvOn;
pub const withGroupCumFano = withGroupCumulativeFano;
pub const withGroupCumFanoOn = withGroupCumulativeFanoOn;
pub const withGroupCumulativeIndexOfDispersion = withGroupCumulativeFano;
pub const withGroupCumulativeIndexOfDispersionOn = withGroupCumulativeFanoOn;
pub const withGroupCumIndexOfDispersion = withGroupCumulativeFano;
pub const withGroupCumIndexOfDispersionOn = withGroupCumulativeFanoOn;
pub const withGroupCumulativeSkew = withGroupCumulativeSkewness;
pub const withGroupCumulativeSkewOn = withGroupCumulativeSkewnessOn;
pub const withGroupCumSkewness = withGroupCumulativeSkewness;
pub const withGroupCumSkewnessOn = withGroupCumulativeSkewnessOn;
pub const withGroupCumSkew = withGroupCumulativeSkewness;
pub const withGroupCumSkewOn = withGroupCumulativeSkewnessOn;
pub const withGroupCumulativeKurt = withGroupCumulativeKurtosis;
pub const withGroupCumulativeKurtOn = withGroupCumulativeKurtosisOn;
pub const withGroupCumKurtosis = withGroupCumulativeKurtosis;
pub const withGroupCumKurtosisOn = withGroupCumulativeKurtosisOn;
pub const withGroupCumKurt = withGroupCumulativeKurtosis;
pub const withGroupCumKurtOn = withGroupCumulativeKurtosisOn;
pub const withGroupCumulativeMeanAbsolute = withGroupCumulativeMeanAbs;
pub const withGroupCumulativeMeanAbsoluteOn = withGroupCumulativeMeanAbsOn;
pub const withGroupCumMeanAbs = withGroupCumulativeMeanAbs;
pub const withGroupCumMeanAbsOn = withGroupCumulativeMeanAbsOn;
pub const withGroupCumMeanAbsolute = withGroupCumulativeMeanAbs;
pub const withGroupCumMeanAbsoluteOn = withGroupCumulativeMeanAbsOn;
pub const withGroupCumulativeMeanSquared = withGroupCumulativeMeanSquare;
pub const withGroupCumulativeMeanSquaredOn = withGroupCumulativeMeanSquareOn;
pub const withGroupCumulativeMeanSq = withGroupCumulativeMeanSquare;
pub const withGroupCumulativeMeanSqOn = withGroupCumulativeMeanSquareOn;
pub const withGroupCumMeanSquare = withGroupCumulativeMeanSquare;
pub const withGroupCumMeanSquareOn = withGroupCumulativeMeanSquareOn;
pub const withGroupCumMeanSquared = withGroupCumulativeMeanSquare;
pub const withGroupCumMeanSquaredOn = withGroupCumulativeMeanSquareOn;
pub const withGroupCumMeanSq = withGroupCumulativeMeanSquare;
pub const withGroupCumMeanSqOn = withGroupCumulativeMeanSquareOn;
pub const withGroupCumulativeRMS = withGroupCumulativeRms;
pub const withGroupCumulativeRMSOn = withGroupCumulativeRmsOn;
pub const withGroupCumRms = withGroupCumulativeRms;
pub const withGroupCumRmsOn = withGroupCumulativeRmsOn;
pub const withGroupCumRMS = withGroupCumulativeRms;
pub const withGroupCumRMSOn = withGroupCumulativeRmsOn;
pub const withGroupCumulativeMaxAbsolute = withGroupCumulativeMaxAbs;
pub const withGroupCumulativeMaxAbsoluteOn = withGroupCumulativeMaxAbsOn;
pub const withGroupCumMaxAbs = withGroupCumulativeMaxAbs;
pub const withGroupCumMaxAbsOn = withGroupCumulativeMaxAbsOn;
pub const withGroupCumMaxAbsolute = withGroupCumulativeMaxAbs;
pub const withGroupCumMaxAbsoluteOn = withGroupCumulativeMaxAbsOn;
pub const withGroupCumulativeLInfNorm = withGroupCumulativeMaxAbs;
pub const withGroupCumulativeLInfNormOn = withGroupCumulativeMaxAbsOn;
pub const withGroupCumulativeLinfNorm = withGroupCumulativeMaxAbs;
pub const withGroupCumulativeLinfNormOn = withGroupCumulativeMaxAbsOn;
pub const withGroupCumLInfNorm = withGroupCumulativeMaxAbs;
pub const withGroupCumLInfNormOn = withGroupCumulativeMaxAbsOn;
pub const withGroupCumLinfNorm = withGroupCumulativeMaxAbs;
pub const withGroupCumLinfNormOn = withGroupCumulativeMaxAbsOn;
pub const withGroupCumulativeMinAbsolute = withGroupCumulativeMinAbs;
pub const withGroupCumulativeMinAbsoluteOn = withGroupCumulativeMinAbsOn;
pub const withGroupCumMinAbs = withGroupCumulativeMinAbs;
pub const withGroupCumMinAbsOn = withGroupCumulativeMinAbsOn;
pub const withGroupCumMinAbsolute = withGroupCumulativeMinAbs;
pub const withGroupCumMinAbsoluteOn = withGroupCumulativeMinAbsOn;
pub const withGroupCumL1Norm = withGroupCumulativeL1Norm;
pub const withGroupCumL1NormOn = withGroupCumulativeL1NormOn;
pub const withGroupCumL2Norm = withGroupCumulativeL2Norm;
pub const withGroupCumL2NormOn = withGroupCumulativeL2NormOn;
pub const withGroupCumulativePtp = withGroupCumulativeRange;
pub const withGroupCumulativePtpOn = withGroupCumulativeRangeOn;
pub const withGroupCumulativePTP = withGroupCumulativeRange;
pub const withGroupCumulativePTPOn = withGroupCumulativeRangeOn;
pub const withGroupCumulativePeakToPeak = withGroupCumulativeRange;
pub const withGroupCumulativePeakToPeakOn = withGroupCumulativeRangeOn;
pub const withGroupCumRange = withGroupCumulativeRange;
pub const withGroupCumRangeOn = withGroupCumulativeRangeOn;
pub const withGroupCumPtp = withGroupCumulativeRange;
pub const withGroupCumPtpOn = withGroupCumulativeRangeOn;
pub const withGroupCumPTP = withGroupCumulativeRange;
pub const withGroupCumPTPOn = withGroupCumulativeRangeOn;
pub const withGroupCumPeakToPeak = withGroupCumulativeRange;
pub const withGroupCumPeakToPeakOn = withGroupCumulativeRangeOn;
pub const withGroupCumMidrange = withGroupCumulativeMidrange;
pub const withGroupCumMidrangeOn = withGroupCumulativeMidrangeOn;
pub const withGroupCumulativeRangeCoefficient = withGroupCumulativeRangeCoeff;
pub const withGroupCumulativeRangeCoefficientOn = withGroupCumulativeRangeCoeffOn;
pub const withGroupCumRangeCoeff = withGroupCumulativeRangeCoeff;
pub const withGroupCumRangeCoeffOn = withGroupCumulativeRangeCoeffOn;
pub const withGroupCumRangeCoefficient = withGroupCumulativeRangeCoeff;
pub const withGroupCumRangeCoefficientOn = withGroupCumulativeRangeCoeffOn;
pub const withGroupCumulativeLogsumexp = withGroupCumulativeLogSumExp;
pub const withGroupCumulativeLogsumexpOn = withGroupCumulativeLogSumExpOn;
pub const withGroupCumLogSumExp = withGroupCumulativeLogSumExp;
pub const withGroupCumLogSumExpOn = withGroupCumulativeLogSumExpOn;
pub const withGroupCumLogsumexp = withGroupCumulativeLogSumExp;
pub const withGroupCumLogsumexpOn = withGroupCumulativeLogSumExpOn;
pub const withGroupCumulativeLogmeanexp = withGroupCumulativeLogMeanExp;
pub const withGroupCumulativeLogmeanexpOn = withGroupCumulativeLogMeanExpOn;
pub const withGroupCumLogMeanExp = withGroupCumulativeLogMeanExp;
pub const withGroupCumLogMeanExpOn = withGroupCumulativeLogMeanExpOn;
pub const withGroupCumLogmeanexp = withGroupCumulativeLogMeanExp;
pub const withGroupCumLogmeanexpOn = withGroupCumulativeLogMeanExpOn;
pub const withGroupCumulativeGeoMean = withGroupCumulativeGeometricMean;
pub const withGroupCumulativeGeoMeanOn = withGroupCumulativeGeometricMeanOn;
pub const withGroupCumGeometricMean = withGroupCumulativeGeometricMean;
pub const withGroupCumGeometricMeanOn = withGroupCumulativeGeometricMeanOn;
pub const withGroupCumGeoMean = withGroupCumulativeGeometricMean;
pub const withGroupCumGeoMeanOn = withGroupCumulativeGeometricMeanOn;
pub const withGroupCumulativeHarmMean = withGroupCumulativeHarmonicMean;
pub const withGroupCumulativeHarmMeanOn = withGroupCumulativeHarmonicMeanOn;
pub const withGroupCumHarmonicMean = withGroupCumulativeHarmonicMean;
pub const withGroupCumHarmonicMeanOn = withGroupCumulativeHarmonicMeanOn;
pub const withGroupCumHarmMean = withGroupCumulativeHarmonicMean;
pub const withGroupCumHarmMeanOn = withGroupCumulativeHarmonicMeanOn;
pub const withGroupCumArgMin = withGroupCumulativeArgMin;
pub const withGroupCumArgMinOn = withGroupCumulativeArgMinOn;
pub const withGroupCumulativeArgmin = withGroupCumulativeArgMin;
pub const withGroupCumulativeArgminOn = withGroupCumulativeArgMinOn;
pub const withGroupCumArgmin = withGroupCumulativeArgMin;
pub const withGroupCumArgminOn = withGroupCumulativeArgMinOn;
pub const withGroupCumArgMax = withGroupCumulativeArgMax;
pub const withGroupCumArgMaxOn = withGroupCumulativeArgMaxOn;
pub const withGroupCumulativeArgmax = withGroupCumulativeArgMax;
pub const withGroupCumulativeArgmaxOn = withGroupCumulativeArgMaxOn;
pub const withGroupCumArgmax = withGroupCumulativeArgMax;
pub const withGroupCumArgmaxOn = withGroupCumulativeArgMaxOn;

pub fn withGroupRowNumber(self: anytype, key_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupRowNumberOn(self, key_names[0..], output_name);
}

pub fn withGroupRowNumberOn(self: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupRowNumberOn(FrameType(@TypeOf(self)), frameValue(self), key_names, output_name);
}

pub const withGroupCumCount = withGroupRowNumber;
pub const withGroupCumCountOn = withGroupRowNumberOn;

pub fn withGroupSize(self: anytype, key_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupSizeOn(self, key_names[0..], output_name);
}

pub fn withGroupSizeOn(self: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupSizeOn(FrameType(@TypeOf(self)), frameValue(self), key_names, output_name);
}

pub const withGroupCount = withGroupSize;
pub const withGroupCountOn = withGroupSizeOn;

pub fn withGroupReverseRowNumber(self: anytype, key_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return withGroupReverseRowNumberOn(self, key_names[0..], output_name);
}

pub fn withGroupReverseRowNumberOn(self: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.withGroupReverseRowNumberOn(FrameType(@TypeOf(self)), frameValue(self), key_names, output_name);
}

pub const withGroupReverseCumCount = withGroupReverseRowNumber;
pub const withGroupReverseCumCountOn = withGroupReverseRowNumberOn;

pub fn valueCounts(self: anytype, key_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return valueCountsAs(self, key_name, "count");
}

pub fn valueCountsAs(self: anytype, key_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return groupByCount(self, key_name, output_name);
}

pub fn valueCountsOn(self: anytype, key_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return valueCountsOnAs(self, key_names, "count");
}

pub fn valueCountsOnAs(self: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return groupByCountOn(self, key_names, output_name);
}

pub fn valueCountsSorted(self: anytype, key_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return valueCountsSortedAs(self, key_name, "count");
}

pub fn valueCountsSortedAs(self: anytype, key_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var counts = try valueCountsAs(self, key_name, output_name);
    defer counts.deinit();
    return counts.sortBy(output_name, .{ .descending = true });
}

pub fn valueCountsOnSorted(self: anytype, key_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return valueCountsOnSortedAs(self, key_names, "count");
}

pub fn valueCountsOnSortedAs(self: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var counts = try valueCountsOnAs(self, key_names, output_name);
    defer counts.deinit();
    return counts.sortBy(output_name, .{ .descending = true });
}

pub fn valueCountsSortedOn(self: anytype, key_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return valueCountsOnSorted(self, key_names);
}

pub fn valueCountsSortedOnAs(self: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return valueCountsOnSortedAs(self, key_names, output_name);
}

pub fn groupBySum(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_profile_mod.groupByNumeric(FrameType(@TypeOf(self)), .sum, frameValue(self), key_name, value_name, output_name);
}

pub fn groupBySumOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByNumericOn(FrameType(@TypeOf(self)), .sum, frameValue(self), key_names, value_name, output_name);
}

pub fn groupByProd(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_profile_mod.groupByNumeric(FrameType(@TypeOf(self)), .prod, frameValue(self), key_name, value_name, output_name);
}

pub fn groupByProduct(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return groupByProd(self, key_name, value_name, output_name);
}

pub fn groupByProdOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByNumericOn(FrameType(@TypeOf(self)), .prod, frameValue(self), key_names, value_name, output_name);
}

pub fn groupByProductOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return groupByProdOn(self, key_names, value_name, output_name);
}

pub fn groupByMin(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_profile_mod.groupByNumeric(FrameType(@TypeOf(self)), .min, frameValue(self), key_name, value_name, output_name);
}

pub fn groupByMinOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByNumericOn(FrameType(@TypeOf(self)), .min, frameValue(self), key_names, value_name, output_name);
}

pub fn groupByMax(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_profile_mod.groupByNumeric(FrameType(@TypeOf(self)), .max, frameValue(self), key_name, value_name, output_name);
}

pub fn groupByMaxOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByNumericOn(FrameType(@TypeOf(self)), .max, frameValue(self), key_names, value_name, output_name);
}

pub fn groupByMean(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_profile_mod.groupByMean(FrameType(@TypeOf(self)), frameValue(self), key_name, value_name, output_name);
}

pub fn groupByMeanOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByMeanOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByFirst(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByFirstOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByFirstOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByFirstOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByLast(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByLastOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByLastOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByLastOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByFirstRow(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByFirstRowOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByFirstRowOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByFirstRowOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByLastRow(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByLastRowOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByLastRowOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByLastRowOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByNth(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8, n: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByNthOn(self, key_names[0..], value_name, output_name, n);
}

pub fn groupByNthOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, n: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByNthOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name, n);
}

pub fn groupByNthRow(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8, n: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByNthRowOn(self, key_names[0..], value_name, output_name, n);
}

pub fn groupByNthRowOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, n: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByNthRowOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name, n);
}

pub fn groupByNthIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8, n: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByNthIndexOn(self, key_names[0..], value_name, output_name, n);
}

pub fn groupByNthIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, n: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByNthIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name, n);
}

pub fn groupByNthRowIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8, n: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByNthRowIndexOn(self, key_names[0..], value_name, output_name, n);
}

pub fn groupByNthRowIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, n: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByNthRowIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name, n);
}

pub fn groupByNUnique(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByNUniqueOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByNUniqueOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByNUniqueOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const groupByNunique = groupByNUnique;
pub const groupByNuniqueOn = groupByNUniqueOn;

pub fn groupByMode(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByModeOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByModeOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByModeOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByModeCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByModeCountOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByModeCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByModeCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByModeRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByModeRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByModeRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByModeRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByModeMargin(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByModeMarginOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByModeMarginOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByModeMarginOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByModeMarginRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByModeMarginRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByModeMarginRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByModeMarginRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByEntropy(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByEntropyOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByEntropyOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByEntropyOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByGiniImpurity(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByGiniImpurityOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByGiniImpurityOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByGiniImpurityOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const groupByGini = groupByGiniImpurity;
pub const groupByGiniOn = groupByGiniImpurityOn;

pub fn groupByPerplexity(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByPerplexityOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByPerplexityOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByPerplexityOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByInverseSimpson(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByInverseSimpsonOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByInverseSimpsonOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByInverseSimpsonOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupBySimpsonConcentration(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupBySimpsonConcentrationOn(self, key_names[0..], value_name, output_name);
}

pub fn groupBySimpsonConcentrationOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupBySimpsonConcentrationOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const groupByConcentration = groupBySimpsonConcentration;
pub const groupByConcentrationOn = groupBySimpsonConcentrationOn;

pub fn groupByEvenness(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByEvennessOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByEvennessOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByEvennessOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByGiniMeanDiff(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByGiniMeanDiffOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByGiniMeanDiffOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByGiniMeanDiffOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByGiniCoefficient(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByGiniCoefficientOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByGiniCoefficientOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByGiniCoefficientOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const groupByGiniCoeff = groupByGiniCoefficient;
pub const groupByGiniCoeffOn = groupByGiniCoefficientOn;

pub fn groupByWeightedMean(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByWeightedMeanOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn groupByWeightedMeanOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByWeightedMeanOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, weight_name, output_name);
}

pub fn groupByWeightedVariance(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByWeightedVarianceOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn groupByWeightedVarianceOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByWeightedVarianceOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, weight_name, output_name);
}

pub const groupByWeightedVar = groupByWeightedVariance;
pub const groupByWeightedVarOn = groupByWeightedVarianceOn;

pub fn groupByWeightedStddev(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByWeightedStddevOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn groupByWeightedStddevOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByWeightedStddevOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, weight_name, output_name);
}

pub const groupByWeightedStd = groupByWeightedStddev;
pub const groupByWeightedStdOn = groupByWeightedStddevOn;

pub fn groupByWeightedQuantile(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8, q: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByWeightedQuantileOn(self, key_names[0..], value_name, weight_name, output_name, q);
}

pub fn groupByWeightedQuantileOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8, q: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByWeightedQuantileOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, weight_name, output_name, q);
}

pub fn groupByWeightedMedian(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByWeightedMedianOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn groupByWeightedMedianOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByWeightedMedianOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, weight_name, output_name);
}

pub fn groupByWeightedIqr(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByWeightedIqrOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn groupByWeightedIqrOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByWeightedIqrOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, weight_name, output_name);
}

pub const groupByWeightedIQR = groupByWeightedIqr;
pub const groupByWeightedIQROn = groupByWeightedIqrOn;

pub fn groupByWeightedMad(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByWeightedMadOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn groupByWeightedMadOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByWeightedMadOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, weight_name, output_name);
}

pub const groupByWeightedMAD = groupByWeightedMad;
pub const groupByWeightedMADOn = groupByWeightedMadOn;

pub fn groupByWeightedMode(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByWeightedModeOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn groupByWeightedModeOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByWeightedModeOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, weight_name, output_name);
}

pub fn groupByWeightedModeWeight(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByWeightedModeWeightOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn groupByWeightedModeWeightOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByWeightedModeWeightOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, weight_name, output_name);
}

pub fn groupByWeightedModeRatio(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByWeightedModeRatioOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn groupByWeightedModeRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByWeightedModeRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, weight_name, output_name);
}

pub fn groupByWeightedModeMargin(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByWeightedModeMarginOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn groupByWeightedModeMarginOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByWeightedModeMarginOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, weight_name, output_name);
}

pub fn groupByWeightedModeMarginRatio(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByWeightedModeMarginRatioOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn groupByWeightedModeMarginRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByWeightedModeMarginRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, weight_name, output_name);
}

pub fn groupByWeightedEntropy(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByWeightedEntropyOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn groupByWeightedEntropyOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByWeightedEntropyOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, weight_name, output_name);
}

pub fn groupByWeightedGiniImpurity(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByWeightedGiniImpurityOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn groupByWeightedGiniImpurityOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByWeightedGiniImpurityOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, weight_name, output_name);
}

pub const groupByWeightedGini = groupByWeightedGiniImpurity;
pub const groupByWeightedGiniOn = groupByWeightedGiniImpurityOn;

pub fn groupByWeightedPerplexity(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByWeightedPerplexityOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn groupByWeightedPerplexityOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByWeightedPerplexityOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, weight_name, output_name);
}

pub fn groupByWeightedInverseSimpson(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByWeightedInverseSimpsonOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn groupByWeightedInverseSimpsonOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByWeightedInverseSimpsonOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, weight_name, output_name);
}

pub fn groupByWeightedSimpsonConcentration(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByWeightedSimpsonConcentrationOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn groupByWeightedSimpsonConcentrationOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByWeightedSimpsonConcentrationOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, weight_name, output_name);
}

pub const groupByWeightedConcentration = groupByWeightedSimpsonConcentration;
pub const groupByWeightedConcentrationOn = groupByWeightedSimpsonConcentrationOn;

pub fn groupByWeightedEvenness(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByWeightedEvennessOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn groupByWeightedEvennessOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByWeightedEvennessOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, weight_name, output_name);
}

pub fn groupByPairCount(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByPairCountOn(self, key_names[0..], lhs_name, rhs_name, output_name);
}

pub fn groupByPairCountOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByPairCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, lhs_name, rhs_name, output_name);
}

pub fn groupByCovariance(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByCovarianceOn(self, key_names[0..], lhs_name, rhs_name, output_name);
}

pub fn groupByCovarianceOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByCovarianceOn(FrameType(@TypeOf(self)), frameValue(self), key_names, lhs_name, rhs_name, output_name);
}

pub const groupByCov = groupByCovariance;
pub const groupByCovOn = groupByCovarianceOn;

pub fn groupByCorrelation(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByCorrelationOn(self, key_names[0..], lhs_name, rhs_name, output_name);
}

pub fn groupByCorrelationOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByCorrelationOn(FrameType(@TypeOf(self)), frameValue(self), key_names, lhs_name, rhs_name, output_name);
}

pub const groupByCorr = groupByCorrelation;
pub const groupByCorrOn = groupByCorrelationOn;

pub fn groupByBeta(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByBetaOn(self, key_names[0..], lhs_name, rhs_name, output_name);
}

pub fn groupByBetaOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByBetaOn(FrameType(@TypeOf(self)), frameValue(self), key_names, lhs_name, rhs_name, output_name);
}

pub fn groupByDot(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByDotOn(self, key_names[0..], lhs_name, rhs_name, output_name);
}

pub fn groupByDotOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByDotOn(FrameType(@TypeOf(self)), frameValue(self), key_names, lhs_name, rhs_name, output_name);
}

pub fn groupByCosineSimilarity(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByCosineSimilarityOn(self, key_names[0..], lhs_name, rhs_name, output_name);
}

pub fn groupByCosineSimilarityOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByCosineSimilarityOn(FrameType(@TypeOf(self)), frameValue(self), key_names, lhs_name, rhs_name, output_name);
}

pub const groupByCosine = groupByCosineSimilarity;
pub const groupByCosineOn = groupByCosineSimilarityOn;

pub fn groupBySquaredEuclideanDistance(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupBySquaredEuclideanDistanceOn(self, key_names[0..], lhs_name, rhs_name, output_name);
}

pub fn groupBySquaredEuclideanDistanceOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupBySquaredEuclideanDistanceOn(FrameType(@TypeOf(self)), frameValue(self), key_names, lhs_name, rhs_name, output_name);
}

pub fn groupByEuclideanDistance(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByEuclideanDistanceOn(self, key_names[0..], lhs_name, rhs_name, output_name);
}

pub fn groupByEuclideanDistanceOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByEuclideanDistanceOn(FrameType(@TypeOf(self)), frameValue(self), key_names, lhs_name, rhs_name, output_name);
}

pub fn groupByManhattanDistance(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByManhattanDistanceOn(self, key_names[0..], lhs_name, rhs_name, output_name);
}

pub fn groupByManhattanDistanceOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByManhattanDistanceOn(FrameType(@TypeOf(self)), frameValue(self), key_names, lhs_name, rhs_name, output_name);
}

pub fn groupByChebyshevDistance(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByChebyshevDistanceOn(self, key_names[0..], lhs_name, rhs_name, output_name);
}

pub fn groupByChebyshevDistanceOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByChebyshevDistanceOn(FrameType(@TypeOf(self)), frameValue(self), key_names, lhs_name, rhs_name, output_name);
}

pub fn groupByCanberraDistance(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByCanberraDistanceOn(self, key_names[0..], lhs_name, rhs_name, output_name);
}

pub fn groupByCanberraDistanceOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByCanberraDistanceOn(FrameType(@TypeOf(self)), frameValue(self), key_names, lhs_name, rhs_name, output_name);
}

pub fn groupByBrayCurtisDistance(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByBrayCurtisDistanceOn(self, key_names[0..], lhs_name, rhs_name, output_name);
}

pub fn groupByBrayCurtisDistanceOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByBrayCurtisDistanceOn(FrameType(@TypeOf(self)), frameValue(self), key_names, lhs_name, rhs_name, output_name);
}

pub fn groupByMeanError(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByMeanErrorOn(self, key_names[0..], lhs_name, rhs_name, output_name);
}

pub fn groupByMeanErrorOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByMeanErrorOn(FrameType(@TypeOf(self)), frameValue(self), key_names, lhs_name, rhs_name, output_name);
}

pub const groupByBias = groupByMeanError;
pub const groupByBiasOn = groupByMeanErrorOn;

pub fn groupByMae(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByMaeOn(self, key_names[0..], lhs_name, rhs_name, output_name);
}

pub fn groupByMaeOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByMaeOn(FrameType(@TypeOf(self)), frameValue(self), key_names, lhs_name, rhs_name, output_name);
}

pub fn groupByMse(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByMseOn(self, key_names[0..], lhs_name, rhs_name, output_name);
}

pub fn groupByMseOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByMseOn(FrameType(@TypeOf(self)), frameValue(self), key_names, lhs_name, rhs_name, output_name);
}

pub fn groupByRmse(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByRmseOn(self, key_names[0..], lhs_name, rhs_name, output_name);
}

pub fn groupByRmseOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByRmseOn(FrameType(@TypeOf(self)), frameValue(self), key_names, lhs_name, rhs_name, output_name);
}

pub fn groupByMape(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByMapeOn(self, key_names[0..], lhs_name, rhs_name, output_name);
}

pub fn groupByMapeOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByMapeOn(FrameType(@TypeOf(self)), frameValue(self), key_names, lhs_name, rhs_name, output_name);
}

pub fn groupBySmape(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupBySmapeOn(self, key_names[0..], lhs_name, rhs_name, output_name);
}

pub fn groupBySmapeOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupBySmapeOn(FrameType(@TypeOf(self)), frameValue(self), key_names, lhs_name, rhs_name, output_name);
}

pub fn groupByWeightedDot(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByWeightedDotOn(self, key_names[0..], lhs_name, rhs_name, weight_name, output_name);
}

pub fn groupByWeightedDotOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByWeightedDotOn(FrameType(@TypeOf(self)), frameValue(self), key_names, lhs_name, rhs_name, weight_name, output_name);
}

pub fn groupByWeightedCosineSimilarity(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByWeightedCosineSimilarityOn(self, key_names[0..], lhs_name, rhs_name, weight_name, output_name);
}

pub fn groupByWeightedCosineSimilarityOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByWeightedCosineSimilarityOn(FrameType(@TypeOf(self)), frameValue(self), key_names, lhs_name, rhs_name, weight_name, output_name);
}

pub const groupByWeightedCosine = groupByWeightedCosineSimilarity;
pub const groupByWeightedCosineOn = groupByWeightedCosineSimilarityOn;

pub fn groupByWeightedSquaredEuclideanDistance(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByWeightedSquaredEuclideanDistanceOn(self, key_names[0..], lhs_name, rhs_name, weight_name, output_name);
}

pub fn groupByWeightedSquaredEuclideanDistanceOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByWeightedSquaredEuclideanDistanceOn(FrameType(@TypeOf(self)), frameValue(self), key_names, lhs_name, rhs_name, weight_name, output_name);
}

pub fn groupByWeightedEuclideanDistance(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByWeightedEuclideanDistanceOn(self, key_names[0..], lhs_name, rhs_name, weight_name, output_name);
}

pub fn groupByWeightedEuclideanDistanceOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByWeightedEuclideanDistanceOn(FrameType(@TypeOf(self)), frameValue(self), key_names, lhs_name, rhs_name, weight_name, output_name);
}

pub fn groupByWeightedManhattanDistance(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByWeightedManhattanDistanceOn(self, key_names[0..], lhs_name, rhs_name, weight_name, output_name);
}

pub fn groupByWeightedManhattanDistanceOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByWeightedManhattanDistanceOn(FrameType(@TypeOf(self)), frameValue(self), key_names, lhs_name, rhs_name, weight_name, output_name);
}

pub fn groupByWeightedChebyshevDistance(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByWeightedChebyshevDistanceOn(self, key_names[0..], lhs_name, rhs_name, weight_name, output_name);
}

pub fn groupByWeightedChebyshevDistanceOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByWeightedChebyshevDistanceOn(FrameType(@TypeOf(self)), frameValue(self), key_names, lhs_name, rhs_name, weight_name, output_name);
}

pub fn groupByWeightedCanberraDistance(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByWeightedCanberraDistanceOn(self, key_names[0..], lhs_name, rhs_name, weight_name, output_name);
}

pub fn groupByWeightedCanberraDistanceOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByWeightedCanberraDistanceOn(FrameType(@TypeOf(self)), frameValue(self), key_names, lhs_name, rhs_name, weight_name, output_name);
}

pub fn groupByWeightedBrayCurtisDistance(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByWeightedBrayCurtisDistanceOn(self, key_names[0..], lhs_name, rhs_name, weight_name, output_name);
}

pub fn groupByWeightedBrayCurtisDistanceOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByWeightedBrayCurtisDistanceOn(FrameType(@TypeOf(self)), frameValue(self), key_names, lhs_name, rhs_name, weight_name, output_name);
}

pub fn groupByWeightedMeanError(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByWeightedMeanErrorOn(self, key_names[0..], lhs_name, rhs_name, weight_name, output_name);
}

pub fn groupByWeightedMeanErrorOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByWeightedMeanErrorOn(FrameType(@TypeOf(self)), frameValue(self), key_names, lhs_name, rhs_name, weight_name, output_name);
}

pub const groupByWeightedBias = groupByWeightedMeanError;
pub const groupByWeightedBiasOn = groupByWeightedMeanErrorOn;

pub fn groupByWeightedMae(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByWeightedMaeOn(self, key_names[0..], lhs_name, rhs_name, weight_name, output_name);
}

pub fn groupByWeightedMaeOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByWeightedMaeOn(FrameType(@TypeOf(self)), frameValue(self), key_names, lhs_name, rhs_name, weight_name, output_name);
}

pub fn groupByWeightedMse(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByWeightedMseOn(self, key_names[0..], lhs_name, rhs_name, weight_name, output_name);
}

pub fn groupByWeightedMseOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByWeightedMseOn(FrameType(@TypeOf(self)), frameValue(self), key_names, lhs_name, rhs_name, weight_name, output_name);
}

pub fn groupByWeightedRmse(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByWeightedRmseOn(self, key_names[0..], lhs_name, rhs_name, weight_name, output_name);
}

pub fn groupByWeightedRmseOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByWeightedRmseOn(FrameType(@TypeOf(self)), frameValue(self), key_names, lhs_name, rhs_name, weight_name, output_name);
}

pub fn groupByWeightedMape(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByWeightedMapeOn(self, key_names[0..], lhs_name, rhs_name, weight_name, output_name);
}

pub fn groupByWeightedMapeOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByWeightedMapeOn(FrameType(@TypeOf(self)), frameValue(self), key_names, lhs_name, rhs_name, weight_name, output_name);
}

pub fn groupByWeightedSmape(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByWeightedSmapeOn(self, key_names[0..], lhs_name, rhs_name, weight_name, output_name);
}

pub fn groupByWeightedSmapeOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByWeightedSmapeOn(FrameType(@TypeOf(self)), frameValue(self), key_names, lhs_name, rhs_name, weight_name, output_name);
}

pub fn groupByWeightedCovariance(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByWeightedCovarianceOn(self, key_names[0..], lhs_name, rhs_name, weight_name, output_name, correction);
}

pub fn groupByWeightedCovarianceOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByWeightedCovarianceOn(FrameType(@TypeOf(self)), frameValue(self), key_names, lhs_name, rhs_name, weight_name, output_name, correction);
}

pub const groupByWeightedCov = groupByWeightedCovariance;
pub const groupByWeightedCovOn = groupByWeightedCovarianceOn;

pub fn groupByWeightedCorrelation(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByWeightedCorrelationOn(self, key_names[0..], lhs_name, rhs_name, weight_name, output_name, correction);
}

pub fn groupByWeightedCorrelationOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByWeightedCorrelationOn(FrameType(@TypeOf(self)), frameValue(self), key_names, lhs_name, rhs_name, weight_name, output_name, correction);
}

pub const groupByWeightedCorr = groupByWeightedCorrelation;
pub const groupByWeightedCorrOn = groupByWeightedCorrelationOn;

pub fn groupByWeightedBeta(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByWeightedBetaOn(self, key_names[0..], lhs_name, rhs_name, weight_name, output_name, correction);
}

pub fn groupByWeightedBetaOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByWeightedBetaOn(FrameType(@TypeOf(self)), frameValue(self), key_names, lhs_name, rhs_name, weight_name, output_name, correction);
}

pub fn groupByMeanAbsDev(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByMeanAbsDevOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByMeanAbsDevOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByMeanAbsDevOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByMeanAbsDevRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByMeanAbsDevRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByMeanAbsDevRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByMeanAbsDevRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByMedian(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByMedianOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByMedianOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByMedianOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByQuantile(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8, q: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByQuantileOn(self, key_names[0..], value_name, output_name, q);
}

pub fn groupByQuantileOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, q: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByQuantileOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name, q);
}

pub fn groupByIqr(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByIqrOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByIqrOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByIqrOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const groupByIQR = groupByIqr;
pub const groupByIQROn = groupByIqrOn;

pub fn groupByMad(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByMadOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByMadOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByMadOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const groupByMAD = groupByMad;
pub const groupByMADOn = groupByMadOn;
pub const groupByMedianAbsDev = groupByMad;
pub const groupByMedianAbsDevOn = groupByMadOn;

pub fn groupByTrimmedMean(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8, trim_fraction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByTrimmedMeanOn(self, key_names[0..], value_name, output_name, trim_fraction);
}

pub fn groupByTrimmedMeanOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, trim_fraction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByTrimmedMeanOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name, trim_fraction);
}

pub fn groupByWinsorizedMean(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8, winsor_fraction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByWinsorizedMeanOn(self, key_names[0..], value_name, output_name, winsor_fraction);
}

pub fn groupByWinsorizedMeanOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, winsor_fraction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByWinsorizedMeanOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name, winsor_fraction);
}

pub fn groupByInterdecileRange(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByInterdecileRangeOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByInterdecileRangeOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByInterdecileRangeOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const groupByIdr = groupByInterdecileRange;
pub const groupByIdrOn = groupByInterdecileRangeOn;
pub const groupByIDR = groupByInterdecileRange;
pub const groupByIDROn = groupByInterdecileRangeOn;

pub fn groupByMidhinge(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByMidhingeOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByMidhingeOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByMidhingeOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByTrimean(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByTrimeanOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByTrimeanOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByTrimeanOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByBowleySkewness(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByBowleySkewnessOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByBowleySkewnessOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByBowleySkewnessOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const groupByBowleySkew = groupByBowleySkewness;
pub const groupByBowleySkewOn = groupByBowleySkewnessOn;

pub fn groupByQuartileCoeffDispersion(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByQuartileCoeffDispersionOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByQuartileCoeffDispersionOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByQuartileCoeffDispersionOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const groupByQcd = groupByQuartileCoeffDispersion;
pub const groupByQcdOn = groupByQuartileCoeffDispersionOn;

pub fn groupByKelleySkewness(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByKelleySkewnessOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByKelleySkewnessOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByKelleySkewnessOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const groupByKelleySkew = groupByKelleySkewness;
pub const groupByKelleySkewOn = groupByKelleySkewnessOn;

pub fn groupByVariance(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByVarianceOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByVarianceOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByVarianceOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByStddev(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByStddevOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByStddevOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByStddevOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const groupByStd = groupByStddev;
pub const groupByStdOn = groupByStddevOn;

pub fn groupBySem(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupBySemOn(self, key_names[0..], value_name, output_name);
}

pub fn groupBySemOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupBySemOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const groupBySEM = groupBySem;
pub const groupBySEMOn = groupBySemOn;

pub fn groupByCv(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByCvOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByCvOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByCvOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const groupByCV = groupByCv;
pub const groupByCVOn = groupByCvOn;

pub fn groupByFano(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByFanoOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByFanoOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByFanoOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const groupByIndexOfDispersion = groupByFano;
pub const groupByIndexOfDispersionOn = groupByFanoOn;

pub fn groupBySkewness(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupBySkewnessOn(self, key_names[0..], value_name, output_name);
}

pub fn groupBySkewnessOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupBySkewnessOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByKurtosis(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByKurtosisOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByKurtosisOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByKurtosisOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const groupBySkew = groupBySkewness;
pub const groupBySkewOn = groupBySkewnessOn;
pub const groupByKurt = groupByKurtosis;
pub const groupByKurtOn = groupByKurtosisOn;

pub fn groupByMagnitudeVariance(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByMagnitudeVarianceOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByMagnitudeVarianceOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByMagnitudeVarianceOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const groupByAbsVariance = groupByMagnitudeVariance;
pub const groupByAbsVarianceOn = groupByMagnitudeVarianceOn;
pub const groupByMagnitudeVar = groupByMagnitudeVariance;
pub const groupByMagnitudeVarOn = groupByMagnitudeVarianceOn;
pub const groupByAbsVar = groupByMagnitudeVariance;
pub const groupByAbsVarOn = groupByMagnitudeVarianceOn;

pub fn groupByMagnitudeStddev(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByMagnitudeStddevOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByMagnitudeStddevOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByMagnitudeStddevOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const groupByAbsStddev = groupByMagnitudeStddev;
pub const groupByAbsStddevOn = groupByMagnitudeStddevOn;
pub const groupByMagnitudeStd = groupByMagnitudeStddev;
pub const groupByMagnitudeStdOn = groupByMagnitudeStddevOn;
pub const groupByAbsStd = groupByMagnitudeStddev;
pub const groupByAbsStdOn = groupByMagnitudeStddevOn;

pub fn groupByMagnitudeSem(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByMagnitudeSemOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByMagnitudeSemOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByMagnitudeSemOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const groupByAbsSem = groupByMagnitudeSem;
pub const groupByAbsSemOn = groupByMagnitudeSemOn;

pub fn groupByMagnitudeCv(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByMagnitudeCvOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByMagnitudeCvOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByMagnitudeCvOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const groupByAbsCv = groupByMagnitudeCv;
pub const groupByAbsCvOn = groupByMagnitudeCvOn;
pub const groupByAbsCV = groupByMagnitudeCv;
pub const groupByAbsCVOn = groupByMagnitudeCvOn;

pub fn groupByMagnitudeFano(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByMagnitudeFanoOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByMagnitudeFanoOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByMagnitudeFanoOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const groupByAbsFano = groupByMagnitudeFano;
pub const groupByAbsFanoOn = groupByMagnitudeFanoOn;
pub const groupByMagnitudeIndexOfDispersion = groupByMagnitudeFano;
pub const groupByMagnitudeIndexOfDispersionOn = groupByMagnitudeFanoOn;
pub const groupByAbsIndexOfDispersion = groupByMagnitudeFano;
pub const groupByAbsIndexOfDispersionOn = groupByMagnitudeFanoOn;

pub fn groupByMagnitudeSkewness(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByMagnitudeSkewnessOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByMagnitudeSkewnessOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByMagnitudeSkewnessOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const groupByAbsSkewness = groupByMagnitudeSkewness;
pub const groupByAbsSkewnessOn = groupByMagnitudeSkewnessOn;
pub const groupByMagnitudeSkew = groupByMagnitudeSkewness;
pub const groupByMagnitudeSkewOn = groupByMagnitudeSkewnessOn;
pub const groupByAbsSkew = groupByMagnitudeSkewness;
pub const groupByAbsSkewOn = groupByMagnitudeSkewnessOn;

pub fn groupByMagnitudeKurtosis(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByMagnitudeKurtosisOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByMagnitudeKurtosisOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByMagnitudeKurtosisOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const groupByAbsKurtosis = groupByMagnitudeKurtosis;
pub const groupByAbsKurtosisOn = groupByMagnitudeKurtosisOn;
pub const groupByMagnitudeKurt = groupByMagnitudeKurtosis;
pub const groupByMagnitudeKurtOn = groupByMagnitudeKurtosisOn;
pub const groupByAbsKurt = groupByMagnitudeKurtosis;
pub const groupByAbsKurtOn = groupByMagnitudeKurtosisOn;

pub fn groupByMeanAbs(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByMeanAbsOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByMeanAbsOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByMeanAbsOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByMeanSquare(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByMeanSquareOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByMeanSquareOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByMeanSquareOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const groupByMeanSq = groupByMeanSquare;
pub const groupByMeanSqOn = groupByMeanSquareOn;

pub fn groupByRms(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByRmsOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByRmsOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByRmsOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const groupByRMS = groupByRms;
pub const groupByRMSOn = groupByRmsOn;

pub fn groupByL1Norm(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByL1NormOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByL1NormOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByL1NormOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByL2Norm(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByL2NormOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByL2NormOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByL2NormOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByMaxAbs(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByMaxAbsOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByMaxAbsOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByMaxAbsOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByMinAbs(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByMinAbsOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByMinAbsOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByMinAbsOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByHhi(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByHhiOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByHhiOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByHhiOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const groupByHerfindahl = groupByHhi;
pub const groupByHerfindahlOn = groupByHhiOn;
pub const groupByHerfindahlHirschman = groupByHhi;
pub const groupByHerfindahlHirschmanOn = groupByHhiOn;

pub fn groupByMagnitudeNormalizedHhi(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByMagnitudeNormalizedHhiOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByMagnitudeNormalizedHhiOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByMagnitudeNormalizedHhiOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const groupByAbsNormalizedHhi = groupByMagnitudeNormalizedHhi;
pub const groupByAbsNormalizedHhiOn = groupByMagnitudeNormalizedHhiOn;

pub fn groupByMagnitudeSparsity(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByMagnitudeSparsityOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByMagnitudeSparsityOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByMagnitudeSparsityOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const groupByAbsSparsity = groupByMagnitudeSparsity;
pub const groupByAbsSparsityOn = groupByMagnitudeSparsityOn;

pub fn groupByMagnitudeInverseSimpson(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByMagnitudeInverseSimpsonOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByMagnitudeInverseSimpsonOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByMagnitudeInverseSimpsonOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const groupByAbsInverseSimpson = groupByMagnitudeInverseSimpson;
pub const groupByAbsInverseSimpsonOn = groupByMagnitudeInverseSimpsonOn;

pub fn groupByMagnitudeSimpsonEvenness(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByMagnitudeSimpsonEvennessOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByMagnitudeSimpsonEvennessOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByMagnitudeSimpsonEvennessOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const groupByAbsSimpsonEvenness = groupByMagnitudeSimpsonEvenness;
pub const groupByAbsSimpsonEvennessOn = groupByMagnitudeSimpsonEvennessOn;

pub fn groupByMagnitudeDominance(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByMagnitudeDominanceOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByMagnitudeDominanceOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByMagnitudeDominanceOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const groupByAbsDominance = groupByMagnitudeDominance;
pub const groupByAbsDominanceOn = groupByMagnitudeDominanceOn;

pub fn groupByMagnitudeDominanceMargin(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByMagnitudeDominanceMarginOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByMagnitudeDominanceMarginOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByMagnitudeDominanceMarginOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const groupByAbsDominanceMargin = groupByMagnitudeDominanceMargin;
pub const groupByAbsDominanceMarginOn = groupByMagnitudeDominanceMarginOn;

pub fn groupByMagnitudeEntropy(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByMagnitudeEntropyOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByMagnitudeEntropyOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByMagnitudeEntropyOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const groupByAbsEntropy = groupByMagnitudeEntropy;
pub const groupByAbsEntropyOn = groupByMagnitudeEntropyOn;

pub fn groupByMagnitudePerplexity(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByMagnitudePerplexityOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByMagnitudePerplexityOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByMagnitudePerplexityOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const groupByAbsPerplexity = groupByMagnitudePerplexity;
pub const groupByAbsPerplexityOn = groupByMagnitudePerplexityOn;

pub fn groupByMagnitudeEvenness(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByMagnitudeEvennessOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByMagnitudeEvennessOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByMagnitudeEvennessOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const groupByAbsEvenness = groupByMagnitudeEvenness;
pub const groupByAbsEvennessOn = groupByMagnitudeEvennessOn;

pub fn groupByGeometricMean(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByGeometricMeanOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByGeometricMeanOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByGeometricMeanOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const groupByGeoMean = groupByGeometricMean;
pub const groupByGeoMeanOn = groupByGeometricMeanOn;

pub fn groupByHarmonicMean(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByHarmonicMeanOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByHarmonicMeanOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByHarmonicMeanOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByLogSumExp(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByLogSumExpOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByLogSumExpOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByLogSumExpOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const groupByLogsumexp = groupByLogSumExp;
pub const groupByLogsumexpOn = groupByLogSumExpOn;

pub fn groupByLogMeanExp(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByLogMeanExpOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByLogMeanExpOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByLogMeanExpOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const groupByLogmeanexp = groupByLogMeanExp;
pub const groupByLogmeanexpOn = groupByLogMeanExpOn;

pub fn groupByPtp(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByPtpOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByPtpOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByPtpOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const groupByPTP = groupByPtp;
pub const groupByPTPOn = groupByPtpOn;
pub const groupByPeakToPeak = groupByPtp;
pub const groupByPeakToPeakOn = groupByPtpOn;

pub fn groupByMidrange(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByMidrangeOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByMidrangeOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByMidrangeOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByRangeCoeff(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByRangeCoeffOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByRangeCoeffOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByRangeCoeffOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const groupByRangeCoefficient = groupByRangeCoeff;
pub const groupByRangeCoefficientOn = groupByRangeCoeffOn;

pub fn groupByAny(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByAnyOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByAnyOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByAnyOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByAll(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByAllOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByAllOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByAllOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByTrueCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByTrueCountOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByTrueCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByTrueCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByFalseCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByFalseCountOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByFalseCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByFalseCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByTrueRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByTrueRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByTrueRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByTrueRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByFalseRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByFalseRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByFalseRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByFalseRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByFirstTrueIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByFirstTrueIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByFirstTrueIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByFirstTrueIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByLastTrueIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByLastTrueIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByLastTrueIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByLastTrueIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByFirstFalseIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByFirstFalseIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByFirstFalseIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByFirstFalseIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByLastFalseIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByLastFalseIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByLastFalseIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByLastFalseIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByAnyValid(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByAnyValidOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByAnyValidOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByAnyValidOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByAllValid(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByAllValidOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByAllValidOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByAllValidOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByAnyNull(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByAnyNullOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByAnyNullOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByAnyNullOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByAllNull(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByAllNullOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByAllNullOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByAllNullOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByValidCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByValidCountOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByValidCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByValidCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByNullCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByNullCountOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByNullCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByNullCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByValidRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByValidRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByValidRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByValidRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByNullRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByNullRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByNullRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByNullRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByFirstValidIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByFirstValidIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByFirstValidIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByFirstValidIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByLastValidIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByLastValidIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByLastValidIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByLastValidIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByFirstNullIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByFirstNullIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByFirstNullIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByFirstNullIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByLastNullIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByLastNullIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByLastNullIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByLastNullIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByNaNCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByNaNCountOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByNaNCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByNaNCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByNaNRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByNaNRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByNaNRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByNaNRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByInfCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByInfCountOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByInfCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByInfCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByInfRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByInfRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByInfRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByInfRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByPositiveInfCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByPositiveInfCountOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByPositiveInfCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByPositiveInfCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByPositiveInfRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByPositiveInfRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByPositiveInfRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByPositiveInfRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByNegativeInfCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByNegativeInfCountOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByNegativeInfCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByNegativeInfCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByNegativeInfRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByNegativeInfRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByNegativeInfRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByNegativeInfRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByFirstNaNIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByFirstNaNIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByFirstNaNIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByFirstNaNIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByLastNaNIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByLastNaNIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByLastNaNIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByLastNaNIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByFirstInfIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByFirstInfIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByFirstInfIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByFirstInfIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByLastInfIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByLastInfIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByLastInfIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByLastInfIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByFirstPositiveInfIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByFirstPositiveInfIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByFirstPositiveInfIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByFirstPositiveInfIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByLastPositiveInfIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByLastPositiveInfIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByLastPositiveInfIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByLastPositiveInfIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByFirstNegativeInfIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByFirstNegativeInfIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByFirstNegativeInfIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByFirstNegativeInfIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByLastNegativeInfIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByLastNegativeInfIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByLastNegativeInfIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByLastNegativeInfIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByFiniteCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByFiniteCountOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByFiniteCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByFiniteCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByFiniteRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByFiniteRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByFiniteRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByFiniteRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByFirstFiniteIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByFirstFiniteIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByFirstFiniteIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByFirstFiniteIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByLastFiniteIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByLastFiniteIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByLastFiniteIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByLastFiniteIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByNormalCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByNormalCountOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByNormalCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByNormalCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByNormalRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByNormalRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByNormalRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByNormalRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByFirstNormalIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByFirstNormalIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByFirstNormalIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByFirstNormalIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByLastNormalIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByLastNormalIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByLastNormalIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByLastNormalIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupBySubnormalCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupBySubnormalCountOn(self, key_names[0..], value_name, output_name);
}

pub fn groupBySubnormalCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupBySubnormalCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupBySubnormalRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupBySubnormalRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn groupBySubnormalRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupBySubnormalRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByFirstSubnormalIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByFirstSubnormalIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByFirstSubnormalIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByFirstSubnormalIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByLastSubnormalIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByLastSubnormalIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByLastSubnormalIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByLastSubnormalIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByNonFiniteCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByNonFiniteCountOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByNonFiniteCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByNonFiniteCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByNonFiniteRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByNonFiniteRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByNonFiniteRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByNonFiniteRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByFirstNonFiniteIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByFirstNonFiniteIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByFirstNonFiniteIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByFirstNonFiniteIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByLastNonFiniteIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByLastNonFiniteIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByLastNonFiniteIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByLastNonFiniteIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByZeroCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByZeroCountOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByZeroCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByZeroCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByZeroRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByZeroRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByZeroRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByZeroRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByFirstZeroIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByFirstZeroIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByFirstZeroIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByFirstZeroIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByLastZeroIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByLastZeroIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByLastZeroIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByLastZeroIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByPositiveZeroCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByPositiveZeroCountOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByPositiveZeroCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByPositiveZeroCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByPositiveZeroRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByPositiveZeroRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByPositiveZeroRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByPositiveZeroRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByNegativeZeroCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByNegativeZeroCountOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByNegativeZeroCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByNegativeZeroCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByNegativeZeroRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByNegativeZeroRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByNegativeZeroRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByNegativeZeroRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByFirstPositiveZeroIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByFirstPositiveZeroIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByFirstPositiveZeroIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByFirstPositiveZeroIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByLastPositiveZeroIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByLastPositiveZeroIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByLastPositiveZeroIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByLastPositiveZeroIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByFirstNegativeZeroIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByFirstNegativeZeroIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByFirstNegativeZeroIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByFirstNegativeZeroIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByLastNegativeZeroIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByLastNegativeZeroIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByLastNegativeZeroIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByLastNegativeZeroIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByNonZeroCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByNonZeroCountOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByNonZeroCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByNonZeroCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByNonZeroRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByNonZeroRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByNonZeroRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByNonZeroRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByFirstNonZeroIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByFirstNonZeroIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByFirstNonZeroIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByFirstNonZeroIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByLastNonZeroIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByLastNonZeroIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByLastNonZeroIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByLastNonZeroIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByPositiveCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByPositiveCountOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByPositiveCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByPositiveCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByPositiveRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByPositiveRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByPositiveRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByPositiveRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByFirstPositiveIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByFirstPositiveIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByFirstPositiveIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByFirstPositiveIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByLastPositiveIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByLastPositiveIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByLastPositiveIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByLastPositiveIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupBySignBitCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupBySignBitCountOn(self, key_names[0..], value_name, output_name);
}

pub fn groupBySignBitCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupBySignBitCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupBySignBitRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupBySignBitRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn groupBySignBitRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupBySignBitRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByFirstSignBitIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByFirstSignBitIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByFirstSignBitIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByFirstSignBitIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByLastSignBitIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByLastSignBitIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByLastSignBitIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByLastSignBitIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByNegativeCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByNegativeCountOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByNegativeCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByNegativeCountOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByNegativeRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByNegativeRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByNegativeRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByNegativeRatioOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByFirstNegativeIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByFirstNegativeIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByFirstNegativeIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByFirstNegativeIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByLastNegativeIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByLastNegativeIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByLastNegativeIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByLastNegativeIndexOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByArgMin(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByArgMinOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByArgMinOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByArgMinOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub fn groupByArgMax(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    const key_names = [_][]const u8{key_name};
    return groupByArgMaxOn(self, key_names[0..], value_name, output_name);
}

pub fn groupByArgMaxOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByArgMaxOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_name);
}

pub const groupByArgmin = groupByArgMin;
pub const groupByArgminOn = groupByArgMinOn;
pub const groupByArgmax = groupByArgMax;
pub const groupByArgmaxOn = groupByArgMaxOn;

pub fn groupByStats(self: anytype, key_name: []const u8, value_name: []const u8, output_prefix: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_profile_mod.groupByStats(FrameType(@TypeOf(self)), frameValue(self), key_name, value_name, output_prefix);
}

pub fn groupByStatsOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_prefix: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByStatsOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_prefix);
}

pub fn groupByProfile(self: anytype, key_name: []const u8, value_name: []const u8, output_prefix: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_profile_mod.groupByProfile(FrameType(@TypeOf(self)), frameValue(self), key_name, value_name, output_prefix);
}

pub fn groupByProfileOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_prefix: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return group_multi_mod.groupByProfileOn(FrameType(@TypeOf(self)), frameValue(self), key_names, value_name, output_prefix);
}

pub fn innerJoin(
    self: anytype,
    right: FrameType(@TypeOf(self)),
    left_key_name: []const u8,
    right_key_name: []const u8,
    options_value: DeviceJoinOptions,
) DeviceDataError!FrameType(@TypeOf(self)) {
    return join_mod.innerJoin(FrameType(@TypeOf(self)), frameValue(self), right, left_key_name, right_key_name, options_value);
}

pub fn innerJoinOn(
    self: anytype,
    right: FrameType(@TypeOf(self)),
    left_key_names: []const []const u8,
    right_key_names: []const []const u8,
    options_value: DeviceJoinOptions,
) DeviceDataError!FrameType(@TypeOf(self)) {
    return join_mod.innerJoinOn(FrameType(@TypeOf(self)), frameValue(self), right, left_key_names, right_key_names, options_value);
}

pub fn leftJoin(
    self: anytype,
    right: FrameType(@TypeOf(self)),
    left_key_name: []const u8,
    right_key_name: []const u8,
    options_value: DeviceJoinOptions,
) DeviceDataError!FrameType(@TypeOf(self)) {
    return join_mod.leftJoin(FrameType(@TypeOf(self)), frameValue(self), right, left_key_name, right_key_name, options_value);
}

pub fn leftJoinOn(
    self: anytype,
    right: FrameType(@TypeOf(self)),
    left_key_names: []const []const u8,
    right_key_names: []const []const u8,
    options_value: DeviceJoinOptions,
) DeviceDataError!FrameType(@TypeOf(self)) {
    return join_mod.leftJoinOn(FrameType(@TypeOf(self)), frameValue(self), right, left_key_names, right_key_names, options_value);
}

pub fn fullJoin(
    self: anytype,
    right: FrameType(@TypeOf(self)),
    left_key_name: []const u8,
    right_key_name: []const u8,
    options_value: DeviceJoinOptions,
) DeviceDataError!FrameType(@TypeOf(self)) {
    return join_mod.fullJoin(FrameType(@TypeOf(self)), frameValue(self), right, left_key_name, right_key_name, options_value);
}

pub fn fullJoinOn(
    self: anytype,
    right: FrameType(@TypeOf(self)),
    left_key_names: []const []const u8,
    right_key_names: []const []const u8,
    options_value: DeviceJoinOptions,
) DeviceDataError!FrameType(@TypeOf(self)) {
    return join_mod.fullJoinOn(FrameType(@TypeOf(self)), frameValue(self), right, left_key_names, right_key_names, options_value);
}

pub fn semiJoin(
    self: anytype,
    right: FrameType(@TypeOf(self)),
    left_key_name: []const u8,
    right_key_name: []const u8,
) DeviceDataError!FrameType(@TypeOf(self)) {
    return join_mod.semiJoin(FrameType(@TypeOf(self)), frameValue(self), right, left_key_name, right_key_name);
}

pub fn semiJoinOn(
    self: anytype,
    right: FrameType(@TypeOf(self)),
    left_key_names: []const []const u8,
    right_key_names: []const []const u8,
) DeviceDataError!FrameType(@TypeOf(self)) {
    return join_mod.semiJoinOn(FrameType(@TypeOf(self)), frameValue(self), right, left_key_names, right_key_names);
}

pub fn antiJoin(
    self: anytype,
    right: FrameType(@TypeOf(self)),
    left_key_name: []const u8,
    right_key_name: []const u8,
) DeviceDataError!FrameType(@TypeOf(self)) {
    return join_mod.antiJoin(FrameType(@TypeOf(self)), frameValue(self), right, left_key_name, right_key_name);
}

pub fn antiJoinOn(
    self: anytype,
    right: FrameType(@TypeOf(self)),
    left_key_names: []const []const u8,
    right_key_names: []const []const u8,
) DeviceDataError!FrameType(@TypeOf(self)) {
    return join_mod.antiJoinOn(FrameType(@TypeOf(self)), frameValue(self), right, left_key_names, right_key_names);
}

pub fn asofJoin(
    self: anytype,
    right: FrameType(@TypeOf(self)),
    left_key_name: []const u8,
    right_key_name: []const u8,
    options_value: DeviceAsofOptions,
) DeviceDataError!FrameType(@TypeOf(self)) {
    return join_mod.asofJoin(FrameType(@TypeOf(self)), frameValue(self), right, left_key_name, right_key_name, options_value);
}
