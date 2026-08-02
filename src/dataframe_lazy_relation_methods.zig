//! Lazy group-by, join, concat, and distinct method wrappers.

const std = @import("std");
const array_mod = @import("array.zig");
const lazy_group_mod = @import("dataframe_lazy_group_plan.zig");
const lazy_join_mod = @import("dataframe_lazy_join_plan.zig");
const lazy_op_mod = @import("dataframe_lazy_op.zig");
const options_mod = @import("dataframe_options.zig");
const series_mod = @import("series.zig");

const DeviceLazyGroupByAggregation = lazy_op_mod.DeviceLazyGroupByAggregation;
const DeviceLazyWeightedGroupByAggregation = lazy_op_mod.DeviceLazyWeightedGroupByAggregation;
const DeviceLazyJoinKind = lazy_op_mod.DeviceLazyJoinKind;
const DeviceJoinOptions = options_mod.DeviceJoinOptions;
const DeviceAsofOptions = options_mod.DeviceAsofOptions;
const DeviceDataError = series_mod.DataError || array_mod.ArrayError;

pub fn groupByCount(self: anytype, key_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByCount(self, key_name, output_name);
}

pub fn groupByCountOn(self: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByCountOn(self, key_names, output_name);
}

pub fn groupByHeadRows(self: anytype, key_name: []const u8, n: usize) DeviceDataError!void {
    return lazy_group_mod.groupByRows(self, key_name, n, false);
}

pub fn groupByHeadRowsOn(self: anytype, key_names: []const []const u8, n: usize) DeviceDataError!void {
    return lazy_group_mod.groupByRowsOn(self, key_names, n, false);
}

pub fn groupByTailRows(self: anytype, key_name: []const u8, n: usize) DeviceDataError!void {
    return lazy_group_mod.groupByRows(self, key_name, n, true);
}

pub fn groupByTailRowsOn(self: anytype, key_names: []const []const u8, n: usize) DeviceDataError!void {
    return lazy_group_mod.groupByRowsOn(self, key_names, n, true);
}

pub fn groupBySliceRows(self: anytype, key_name: []const u8, start: usize, length: usize) DeviceDataError!void {
    return lazy_group_mod.groupBySliceRows(self, key_name, start, length);
}

pub fn groupBySliceRowsOn(self: anytype, key_names: []const []const u8, start: usize, length: usize) DeviceDataError!void {
    return lazy_group_mod.groupBySliceRowsOn(self, key_names, start, length);
}

pub fn groupBySliceRowsStep(self: anytype, key_name: []const u8, start: usize, length: usize, step: usize) DeviceDataError!void {
    return lazy_group_mod.groupBySliceRowsStep(self, key_name, start, length, step);
}

pub fn groupBySliceRowsStepOn(self: anytype, key_names: []const []const u8, start: usize, length: usize, step: usize) DeviceDataError!void {
    return lazy_group_mod.groupBySliceRowsStepOn(self, key_names, start, length, step);
}

pub fn groupBySliceRowsSigned(self: anytype, key_name: []const u8, start: isize, length: usize) DeviceDataError!void {
    return lazy_group_mod.groupBySliceRowsSigned(self, key_name, start, length);
}

pub fn groupBySliceRowsSignedOn(self: anytype, key_names: []const []const u8, start: isize, length: usize) DeviceDataError!void {
    return lazy_group_mod.groupBySliceRowsSignedOn(self, key_names, start, length);
}

pub fn groupBySliceRowsSignedStep(self: anytype, key_name: []const u8, start: isize, length: usize, step: usize) DeviceDataError!void {
    return lazy_group_mod.groupBySliceRowsSignedStep(self, key_name, start, length, step);
}

pub fn groupBySliceRowsSignedStepOn(self: anytype, key_names: []const []const u8, start: isize, length: usize, step: usize) DeviceDataError!void {
    return lazy_group_mod.groupBySliceRowsSignedStepOn(self, key_names, start, length, step);
}

pub fn groupByTopRows(self: anytype, key_name: []const u8, sort_name: []const u8, n: usize, options_value: options_mod.DeviceSortOptions) DeviceDataError!void {
    return lazy_group_mod.groupBySortedRows(self, key_name, sort_name, n, options_value, false);
}

pub fn groupByTopRowsOn(self: anytype, key_names: []const []const u8, sort_name: []const u8, n: usize, options_value: options_mod.DeviceSortOptions) DeviceDataError!void {
    return lazy_group_mod.groupBySortedRowsOn(self, key_names, sort_name, n, options_value, false);
}

pub fn groupByBottomRows(self: anytype, key_name: []const u8, sort_name: []const u8, n: usize, options_value: options_mod.DeviceSortOptions) DeviceDataError!void {
    return lazy_group_mod.groupBySortedRows(self, key_name, sort_name, n, options_value, true);
}

pub fn groupByBottomRowsOn(self: anytype, key_names: []const []const u8, sort_name: []const u8, n: usize, options_value: options_mod.DeviceSortOptions) DeviceDataError!void {
    return lazy_group_mod.groupBySortedRowsOn(self, key_names, sort_name, n, options_value, true);
}

pub fn groupByTopRowsByColumns(self: anytype, key_name: []const u8, sort_names: []const []const u8, n: usize, options_values: []const options_mod.DeviceSortOptions) DeviceDataError!void {
    return lazy_group_mod.groupBySortedRowsByColumns(self, key_name, sort_names, n, options_values, false);
}

pub fn groupByTopRowsByColumnsOn(self: anytype, key_names: []const []const u8, sort_names: []const []const u8, n: usize, options_values: []const options_mod.DeviceSortOptions) DeviceDataError!void {
    return lazy_group_mod.groupBySortedRowsByColumnsOn(self, key_names, sort_names, n, options_values, false);
}

pub fn groupByBottomRowsByColumns(self: anytype, key_name: []const u8, sort_names: []const []const u8, n: usize, options_values: []const options_mod.DeviceSortOptions) DeviceDataError!void {
    return lazy_group_mod.groupBySortedRowsByColumns(self, key_name, sort_names, n, options_values, true);
}

pub fn groupByBottomRowsByColumnsOn(self: anytype, key_names: []const []const u8, sort_names: []const []const u8, n: usize, options_values: []const options_mod.DeviceSortOptions) DeviceDataError!void {
    return lazy_group_mod.groupBySortedRowsByColumnsOn(self, key_names, sort_names, n, options_values, true);
}

pub fn withGroupId(self: anytype, key_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupIdOn(self, key_names[0..], output_name);
}

pub fn withGroupIdOn(self: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupId(self, key_names, output_name);
}

pub const withGroupIndex = withGroupId;
pub const withGroupIndexOn = withGroupIdOn;

pub fn withGroupFirstRowIndex(self: anytype, key_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupFirstRowIndexOn(self, key_names[0..], output_name);
}

pub fn withGroupFirstRowIndexOn(self: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupFirstRowIndex(self, key_names, output_name);
}

pub fn withGroupLastRowIndex(self: anytype, key_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupLastRowIndexOn(self, key_names[0..], output_name);
}

pub fn withGroupLastRowIndexOn(self: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupLastRowIndex(self, key_names, output_name);
}

pub fn withGroupIsFirstRow(self: anytype, key_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupIsFirstRowOn(self, key_names[0..], output_name);
}

pub fn withGroupIsFirstRowOn(self: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupIsFirstRow(self, key_names, output_name);
}

pub fn withGroupIsLastRow(self: anytype, key_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupIsLastRowOn(self, key_names[0..], output_name);
}

pub fn withGroupIsLastRowOn(self: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupIsLastRow(self, key_names, output_name);
}

pub fn withGroupIsSingleton(self: anytype, key_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupIsSingletonOn(self, key_names[0..], output_name);
}

pub fn withGroupIsSingletonOn(self: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupIsSingleton(self, key_names, output_name);
}

pub fn withGroupIsDuplicated(self: anytype, key_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupIsDuplicatedOn(self, key_names[0..], output_name);
}

pub fn withGroupIsDuplicatedOn(self: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupIsDuplicated(self, key_names, output_name);
}

pub fn withGroupCumeDist(self: anytype, key_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumeDistOn(self, key_names[0..], output_name);
}

pub fn withGroupCumeDistOn(self: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumeDist(self, key_names, output_name);
}

pub const withGroupCumulativeDistribution = withGroupCumeDist;
pub const withGroupCumulativeDistributionOn = withGroupCumeDistOn;

pub fn withGroupPercentRank(self: anytype, key_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupPercentRankOn(self, key_names[0..], output_name);
}

pub fn withGroupPercentRankOn(self: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupPercentRank(self, key_names, output_name);
}

pub const withGroupPercentileRank = withGroupPercentRank;
pub const withGroupPercentileRankOn = withGroupPercentRankOn;

pub fn withGroupReverseCumeDist(self: anytype, key_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupReverseCumeDistOn(self, key_names[0..], output_name);
}

pub fn withGroupReverseCumeDistOn(self: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupReverseCumeDist(self, key_names, output_name);
}

pub const withGroupReverseCumulativeDistribution = withGroupReverseCumeDist;
pub const withGroupReverseCumulativeDistributionOn = withGroupReverseCumeDistOn;

pub fn withGroupReversePercentRank(self: anytype, key_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupReversePercentRankOn(self, key_names[0..], output_name);
}

pub fn withGroupReversePercentRankOn(self: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupReversePercentRank(self, key_names, output_name);
}

pub const withGroupReversePercentileRank = withGroupReversePercentRank;
pub const withGroupReversePercentileRankOn = withGroupReversePercentRankOn;

pub fn withGroupLag(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8, offset: usize) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupLagOn(self, key_names[0..], value_name, output_name, offset);
}

pub fn withGroupLagOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, offset: usize) DeviceDataError!void {
    return lazy_group_mod.withGroupLag(self, key_names, value_name, output_name, offset);
}

pub fn withGroupLead(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8, offset: usize) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupLeadOn(self, key_names[0..], value_name, output_name, offset);
}

pub fn withGroupLeadOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, offset: usize) DeviceDataError!void {
    return lazy_group_mod.withGroupLead(self, key_names, value_name, output_name, offset);
}

pub fn withGroupFirstRowValue(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupFirstRowValueOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupFirstRowValueOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupFirstRowValue(self, key_names, value_name, output_name);
}

pub fn withGroupLastRowValue(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupLastRowValueOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupLastRowValueOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupLastRowValue(self, key_names, value_name, output_name);
}

pub fn withGroupNthRowValue(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8, n: usize) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupNthRowValueOn(self, key_names[0..], value_name, output_name, n);
}

pub fn withGroupNthRowValueOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, n: usize) DeviceDataError!void {
    return lazy_group_mod.withGroupNthRowValue(self, key_names, value_name, output_name, n);
}

pub const withGroupNthValue = withGroupNthRowValue;
pub const withGroupNthValueOn = withGroupNthRowValueOn;

pub fn withGroupFirstValidValue(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupFirstValidValueOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupFirstValidValueOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupFirstValidValue(self, key_names, value_name, output_name);
}

pub fn withGroupLastValidValue(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupLastValidValueOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupLastValidValueOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupLastValidValue(self, key_names, value_name, output_name);
}

pub fn withGroupNthValidValue(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8, n: usize) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupNthValidValueOn(self, key_names[0..], value_name, output_name, n);
}

pub fn withGroupNthValidValueOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, n: usize) DeviceDataError!void {
    return lazy_group_mod.withGroupNthValidValue(self, key_names, value_name, output_name, n);
}

pub fn withGroupFillNullForward(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupFillNullForwardOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupFillNullForwardOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupFillNullForward(self, key_names, value_name, output_name);
}

pub fn withGroupFillNullBackward(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupFillNullBackwardOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupFillNullBackwardOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupFillNullBackward(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeValidCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeValidCountOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeValidCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeValidCount(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeNullCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeNullCountOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeNullCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeNullCount(self, key_names, value_name, output_name);
}

pub const withGroupCumValidCount = withGroupCumulativeValidCount;
pub const withGroupCumValidCountOn = withGroupCumulativeValidCountOn;
pub const withGroupCumNullCount = withGroupCumulativeNullCount;
pub const withGroupCumNullCountOn = withGroupCumulativeNullCountOn;

pub fn withGroupCumulativeValidRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeValidRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeValidRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeValidRatio(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeNullRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeNullRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeNullRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeNullRatio(self, key_names, value_name, output_name);
}

pub const withGroupCumValidRatio = withGroupCumulativeValidRatio;
pub const withGroupCumValidRatioOn = withGroupCumulativeValidRatioOn;
pub const withGroupCumNullRatio = withGroupCumulativeNullRatio;
pub const withGroupCumNullRatioOn = withGroupCumulativeNullRatioOn;

pub fn withGroupCumulativeFirstValidIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFirstValidIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFirstValidIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeFirstValidIndex(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeLastValidIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeLastValidIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeLastValidIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeLastValidIndex(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeFirstNullIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFirstNullIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFirstNullIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeFirstNullIndex(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeLastNullIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeLastNullIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeLastNullIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeLastNullIndex(self, key_names, value_name, output_name);
}

pub const withGroupCumFirstValidIndex = withGroupCumulativeFirstValidIndex;
pub const withGroupCumFirstValidIndexOn = withGroupCumulativeFirstValidIndexOn;
pub const withGroupCumLastValidIndex = withGroupCumulativeLastValidIndex;
pub const withGroupCumLastValidIndexOn = withGroupCumulativeLastValidIndexOn;
pub const withGroupCumFirstNullIndex = withGroupCumulativeFirstNullIndex;
pub const withGroupCumFirstNullIndexOn = withGroupCumulativeFirstNullIndexOn;
pub const withGroupCumLastNullIndex = withGroupCumulativeLastNullIndex;
pub const withGroupCumLastNullIndexOn = withGroupCumulativeLastNullIndexOn;

pub fn withGroupCumulativeNaNCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeNaNCountOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeNaNCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeNaNCount(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeNaNRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeNaNRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeNaNRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeNaNRatio(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeInfCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeInfCountOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeInfCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeInfCount(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeInfRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeInfRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeInfRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeInfRatio(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativePositiveInfCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativePositiveInfCountOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativePositiveInfCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativePositiveInfCount(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativePositiveInfRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativePositiveInfRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativePositiveInfRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativePositiveInfRatio(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeNegativeInfCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeNegativeInfCountOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeNegativeInfCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeNegativeInfCount(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeNegativeInfRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeNegativeInfRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeNegativeInfRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeNegativeInfRatio(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeFiniteCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFiniteCountOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFiniteCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeFiniteCount(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeFiniteRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFiniteRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFiniteRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeFiniteRatio(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeNormalCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeNormalCountOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeNormalCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeNormalCount(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeNormalRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeNormalRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeNormalRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeNormalRatio(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeSubnormalCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeSubnormalCountOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeSubnormalCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeSubnormalCount(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeSubnormalRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeSubnormalRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeSubnormalRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeSubnormalRatio(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeNonFiniteCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeNonFiniteCountOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeNonFiniteCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeNonFiniteCount(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeNonFiniteRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeNonFiniteRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeNonFiniteRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeNonFiniteRatio(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeZeroCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeZeroCountOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeZeroCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeZeroCount(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeZeroRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeZeroRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeZeroRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeZeroRatio(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativePositiveZeroCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativePositiveZeroCountOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativePositiveZeroCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativePositiveZeroCount(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativePositiveZeroRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativePositiveZeroRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativePositiveZeroRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativePositiveZeroRatio(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeNegativeZeroCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeNegativeZeroCountOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeNegativeZeroCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeNegativeZeroCount(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeNegativeZeroRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeNegativeZeroRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeNegativeZeroRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeNegativeZeroRatio(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeNonZeroCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeNonZeroCountOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeNonZeroCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeNonZeroCount(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeNonZeroRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeNonZeroRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeNonZeroRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeNonZeroRatio(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativePositiveCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativePositiveCountOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativePositiveCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativePositiveCount(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativePositiveRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativePositiveRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativePositiveRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativePositiveRatio(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeSignBitCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeSignBitCountOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeSignBitCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeSignBitCount(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeSignBitRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeSignBitRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeSignBitRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeSignBitRatio(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeNegativeCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeNegativeCountOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeNegativeCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeNegativeCount(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeNegativeRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeNegativeRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeNegativeRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeNegativeRatio(self, key_names, value_name, output_name);
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

pub fn withGroupCumulativeFirstNaNIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFirstNaNIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFirstNaNIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeFirstNaNIndex(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeLastNaNIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeLastNaNIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeLastNaNIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeLastNaNIndex(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeFirstInfIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFirstInfIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFirstInfIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeFirstInfIndex(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeLastInfIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeLastInfIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeLastInfIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeLastInfIndex(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeFirstPositiveInfIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFirstPositiveInfIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFirstPositiveInfIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeFirstPositiveInfIndex(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeLastPositiveInfIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeLastPositiveInfIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeLastPositiveInfIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeLastPositiveInfIndex(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeFirstNegativeInfIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFirstNegativeInfIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFirstNegativeInfIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeFirstNegativeInfIndex(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeLastNegativeInfIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeLastNegativeInfIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeLastNegativeInfIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeLastNegativeInfIndex(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeFirstFiniteIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFirstFiniteIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFirstFiniteIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeFirstFiniteIndex(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeLastFiniteIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeLastFiniteIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeLastFiniteIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeLastFiniteIndex(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeFirstNormalIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFirstNormalIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFirstNormalIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeFirstNormalIndex(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeLastNormalIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeLastNormalIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeLastNormalIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeLastNormalIndex(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeFirstSubnormalIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFirstSubnormalIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFirstSubnormalIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeFirstSubnormalIndex(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeLastSubnormalIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeLastSubnormalIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeLastSubnormalIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeLastSubnormalIndex(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeFirstNonFiniteIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFirstNonFiniteIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFirstNonFiniteIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeFirstNonFiniteIndex(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeLastNonFiniteIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeLastNonFiniteIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeLastNonFiniteIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeLastNonFiniteIndex(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeFirstZeroIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFirstZeroIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFirstZeroIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeFirstZeroIndex(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeLastZeroIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeLastZeroIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeLastZeroIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeLastZeroIndex(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeFirstPositiveZeroIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFirstPositiveZeroIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFirstPositiveZeroIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeFirstPositiveZeroIndex(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeLastPositiveZeroIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeLastPositiveZeroIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeLastPositiveZeroIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeLastPositiveZeroIndex(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeFirstNegativeZeroIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFirstNegativeZeroIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFirstNegativeZeroIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeFirstNegativeZeroIndex(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeLastNegativeZeroIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeLastNegativeZeroIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeLastNegativeZeroIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeLastNegativeZeroIndex(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeFirstNonZeroIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFirstNonZeroIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFirstNonZeroIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeFirstNonZeroIndex(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeLastNonZeroIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeLastNonZeroIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeLastNonZeroIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeLastNonZeroIndex(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeFirstPositiveIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFirstPositiveIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFirstPositiveIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeFirstPositiveIndex(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeLastPositiveIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeLastPositiveIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeLastPositiveIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeLastPositiveIndex(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeFirstSignBitIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFirstSignBitIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFirstSignBitIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeFirstSignBitIndex(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeLastSignBitIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeLastSignBitIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeLastSignBitIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeLastSignBitIndex(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeFirstNegativeIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFirstNegativeIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFirstNegativeIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeFirstNegativeIndex(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeLastNegativeIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeLastNegativeIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeLastNegativeIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeLastNegativeIndex(self, key_names, value_name, output_name);
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

pub fn withGroupCumulativeDistinctCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeDistinctCountOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeDistinctCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeDistinctCount(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeNUnique(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeNUniqueOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeNUniqueOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeNUnique(self, key_names, value_name, output_name);
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

pub fn withGroupCumulativeMode(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeModeOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeModeOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeMode(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeModeCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeModeCountOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeModeCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeModeCount(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeModeRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeModeRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeModeRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeModeRatio(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeModeMargin(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeModeMarginOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeModeMarginOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeModeMargin(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeModeMarginRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeModeMarginRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeModeMarginRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeModeMarginRatio(self, key_names, value_name, output_name);
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

pub fn withGroupCumulativeEntropy(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeEntropyOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeEntropyOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeEntropy(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeGiniImpurity(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeGiniImpurityOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeGiniImpurityOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeGiniImpurity(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativePerplexity(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativePerplexityOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativePerplexityOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativePerplexity(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeInverseSimpson(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeInverseSimpsonOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeInverseSimpsonOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeInverseSimpson(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeSimpsonConcentration(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeSimpsonConcentrationOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeSimpsonConcentrationOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeSimpsonConcentration(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeEvenness(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeEvennessOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeEvennessOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeEvenness(self, key_names, value_name, output_name);
}

pub const withGroupCumulativeGini = withGroupCumulativeGiniImpurity;
pub const withGroupCumulativeGiniOn = withGroupCumulativeGiniImpurityOn;
pub const withGroupCumulativeConcentration = withGroupCumulativeSimpsonConcentration;
pub const withGroupCumulativeConcentrationOn = withGroupCumulativeSimpsonConcentrationOn;
pub const withGroupCumEntropy = withGroupCumulativeEntropy;
pub const withGroupCumEntropyOn = withGroupCumulativeEntropyOn;
pub const withGroupCumGiniImpurity = withGroupCumulativeGiniImpurity;
pub const withGroupCumGiniImpurityOn = withGroupCumulativeGiniImpurityOn;
pub const withGroupCumPerplexity = withGroupCumulativePerplexity;
pub const withGroupCumPerplexityOn = withGroupCumulativePerplexityOn;
pub const withGroupCumInverseSimpson = withGroupCumulativeInverseSimpson;
pub const withGroupCumInverseSimpsonOn = withGroupCumulativeInverseSimpsonOn;
pub const withGroupCumSimpsonConcentration = withGroupCumulativeSimpsonConcentration;
pub const withGroupCumSimpsonConcentrationOn = withGroupCumulativeSimpsonConcentrationOn;
pub const withGroupCumEvenness = withGroupCumulativeEvenness;
pub const withGroupCumEvennessOn = withGroupCumulativeEvennessOn;
pub const withGroupCumGini = withGroupCumulativeGiniImpurity;
pub const withGroupCumGiniOn = withGroupCumulativeGiniImpurityOn;
pub const withGroupCumConcentration = withGroupCumulativeSimpsonConcentration;
pub const withGroupCumConcentrationOn = withGroupCumulativeSimpsonConcentrationOn;

pub fn withGroupCumulativeMeanAbsDev(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeMeanAbsDevOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeMeanAbsDevOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeMeanAbsDev(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeMeanAbsDevRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeMeanAbsDevRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeMeanAbsDevRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeMeanAbsDevRatio(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeGiniMeanDiff(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeGiniMeanDiffOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeGiniMeanDiffOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeGiniMeanDiff(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeGiniCoefficient(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeGiniCoefficientOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeGiniCoefficientOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeGiniCoefficient(self, key_names, value_name, output_name);
}

pub const withGroupCumulativeMeanAbsoluteDeviation = withGroupCumulativeMeanAbsDev;
pub const withGroupCumulativeMeanAbsoluteDeviationOn = withGroupCumulativeMeanAbsDevOn;
pub const withGroupCumulativeGiniCoeff = withGroupCumulativeGiniCoefficient;
pub const withGroupCumulativeGiniCoeffOn = withGroupCumulativeGiniCoefficientOn;
pub const withGroupCumMeanAbsDev = withGroupCumulativeMeanAbsDev;
pub const withGroupCumMeanAbsDevOn = withGroupCumulativeMeanAbsDevOn;
pub const withGroupCumMeanAbsDevRatio = withGroupCumulativeMeanAbsDevRatio;
pub const withGroupCumMeanAbsDevRatioOn = withGroupCumulativeMeanAbsDevRatioOn;
pub const withGroupCumGiniMeanDiff = withGroupCumulativeGiniMeanDiff;
pub const withGroupCumGiniMeanDiffOn = withGroupCumulativeGiniMeanDiffOn;
pub const withGroupCumGiniCoefficient = withGroupCumulativeGiniCoefficient;
pub const withGroupCumGiniCoefficientOn = withGroupCumulativeGiniCoefficientOn;
pub const withGroupCumMeanAbsoluteDeviation = withGroupCumulativeMeanAbsDev;
pub const withGroupCumMeanAbsoluteDeviationOn = withGroupCumulativeMeanAbsDevOn;
pub const withGroupCumGiniCoeff = withGroupCumulativeGiniCoefficient;
pub const withGroupCumGiniCoeffOn = withGroupCumulativeGiniCoefficientOn;

pub fn withGroupCumulativeMedian(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeMedianOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeMedianOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeMedian(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeQuantile(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8, q: f64) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeQuantileOn(self, key_names[0..], value_name, output_name, q);
}

pub fn withGroupCumulativeQuantileOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, q: f64) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeQuantile(self, key_names, value_name, output_name, q);
}

pub const withGroupCumQuantile = withGroupCumulativeQuantile;
pub const withGroupCumQuantileOn = withGroupCumulativeQuantileOn;
pub const withGroupCumMedian = withGroupCumulativeMedian;
pub const withGroupCumMedianOn = withGroupCumulativeMedianOn;

pub fn withGroupCumulativeIqr(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeIqrOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeIqrOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeIqr(self, key_names, value_name, output_name);
}

pub const withGroupCumulativeIQR = withGroupCumulativeIqr;
pub const withGroupCumulativeIQROn = withGroupCumulativeIqrOn;
pub const withGroupCumIqr = withGroupCumulativeIqr;
pub const withGroupCumIqrOn = withGroupCumulativeIqrOn;
pub const withGroupCumIQR = withGroupCumulativeIqr;
pub const withGroupCumIQROn = withGroupCumulativeIqrOn;

pub fn withGroupCumulativeMad(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeMadOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeMadOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeMad(self, key_names, value_name, output_name);
}

pub const withGroupCumulativeMAD = withGroupCumulativeMad;
pub const withGroupCumulativeMADOn = withGroupCumulativeMadOn;
pub const withGroupCumulativeMedianAbsDev = withGroupCumulativeMad;
pub const withGroupCumulativeMedianAbsDevOn = withGroupCumulativeMadOn;
pub const withGroupCumMad = withGroupCumulativeMad;
pub const withGroupCumMadOn = withGroupCumulativeMadOn;
pub const withGroupCumMAD = withGroupCumulativeMad;
pub const withGroupCumMADOn = withGroupCumulativeMadOn;
pub const withGroupCumMedianAbsDev = withGroupCumulativeMad;
pub const withGroupCumMedianAbsDevOn = withGroupCumulativeMadOn;

pub fn withGroupCumulativeTrimmedMean(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8, trim_fraction: f64) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeTrimmedMeanOn(self, key_names[0..], value_name, output_name, trim_fraction);
}

pub fn withGroupCumulativeTrimmedMeanOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, trim_fraction: f64) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeTrimmedMean(self, key_names, value_name, output_name, trim_fraction);
}

pub const withGroupCumTrimmedMean = withGroupCumulativeTrimmedMean;
pub const withGroupCumTrimmedMeanOn = withGroupCumulativeTrimmedMeanOn;

pub fn withGroupCumulativeWinsorizedMean(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8, winsor_fraction: f64) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWinsorizedMeanOn(self, key_names[0..], value_name, output_name, winsor_fraction);
}

pub fn withGroupCumulativeWinsorizedMeanOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, winsor_fraction: f64) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWinsorizedMean(self, key_names, value_name, output_name, winsor_fraction);
}

pub const withGroupCumWinsorizedMean = withGroupCumulativeWinsorizedMean;
pub const withGroupCumWinsorizedMeanOn = withGroupCumulativeWinsorizedMeanOn;

pub fn withGroupCumulativeInterdecileRange(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeInterdecileRangeOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeInterdecileRangeOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeInterdecileRange(self, key_names, value_name, output_name);
}

pub const withGroupCumulativeIdr = withGroupCumulativeInterdecileRange;
pub const withGroupCumulativeIdrOn = withGroupCumulativeInterdecileRangeOn;
pub const withGroupCumulativeIDR = withGroupCumulativeInterdecileRange;
pub const withGroupCumulativeIDROn = withGroupCumulativeInterdecileRangeOn;
pub const withGroupCumIdr = withGroupCumulativeInterdecileRange;
pub const withGroupCumIdrOn = withGroupCumulativeInterdecileRangeOn;
pub const withGroupCumIDR = withGroupCumulativeInterdecileRange;
pub const withGroupCumIDROn = withGroupCumulativeInterdecileRangeOn;

pub fn withGroupCumulativeMidhinge(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeMidhingeOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeMidhingeOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeMidhinge(self, key_names, value_name, output_name);
}

pub const withGroupCumMidhinge = withGroupCumulativeMidhinge;
pub const withGroupCumMidhingeOn = withGroupCumulativeMidhingeOn;

pub fn withGroupCumulativeTrimean(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeTrimeanOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeTrimeanOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeTrimean(self, key_names, value_name, output_name);
}

pub const withGroupCumTrimean = withGroupCumulativeTrimean;
pub const withGroupCumTrimeanOn = withGroupCumulativeTrimeanOn;

pub fn withGroupCumulativeBowleySkewness(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeBowleySkewnessOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeBowleySkewnessOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeBowleySkewness(self, key_names, value_name, output_name);
}

pub const withGroupCumulativeBowleySkew = withGroupCumulativeBowleySkewness;
pub const withGroupCumulativeBowleySkewOn = withGroupCumulativeBowleySkewnessOn;
pub const withGroupCumBowleySkewness = withGroupCumulativeBowleySkewness;
pub const withGroupCumBowleySkewnessOn = withGroupCumulativeBowleySkewnessOn;
pub const withGroupCumBowleySkew = withGroupCumulativeBowleySkewness;
pub const withGroupCumBowleySkewOn = withGroupCumulativeBowleySkewnessOn;

pub fn withGroupCumulativeQuartileCoeffDispersion(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeQuartileCoeffDispersionOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeQuartileCoeffDispersionOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeQuartileCoeffDispersion(self, key_names, value_name, output_name);
}

pub const withGroupCumulativeQcd = withGroupCumulativeQuartileCoeffDispersion;
pub const withGroupCumulativeQcdOn = withGroupCumulativeQuartileCoeffDispersionOn;
pub const withGroupCumulativeQCD = withGroupCumulativeQuartileCoeffDispersion;
pub const withGroupCumulativeQCDOn = withGroupCumulativeQuartileCoeffDispersionOn;
pub const withGroupCumQuartileCoeffDispersion = withGroupCumulativeQuartileCoeffDispersion;
pub const withGroupCumQuartileCoeffDispersionOn = withGroupCumulativeQuartileCoeffDispersionOn;
pub const withGroupCumQcd = withGroupCumulativeQuartileCoeffDispersion;
pub const withGroupCumQcdOn = withGroupCumulativeQuartileCoeffDispersionOn;
pub const withGroupCumQCD = withGroupCumulativeQuartileCoeffDispersion;
pub const withGroupCumQCDOn = withGroupCumulativeQuartileCoeffDispersionOn;

pub fn withGroupCumulativeKelleySkewness(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeKelleySkewnessOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeKelleySkewnessOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeKelleySkewness(self, key_names, value_name, output_name);
}

pub const withGroupCumulativeKelleySkew = withGroupCumulativeKelleySkewness;
pub const withGroupCumulativeKelleySkewOn = withGroupCumulativeKelleySkewnessOn;
pub const withGroupCumKelleySkewness = withGroupCumulativeKelleySkewness;
pub const withGroupCumKelleySkewnessOn = withGroupCumulativeKelleySkewnessOn;
pub const withGroupCumKelleySkew = withGroupCumulativeKelleySkewness;
pub const withGroupCumKelleySkewOn = withGroupCumulativeKelleySkewnessOn;

pub fn withGroupCumulativeAny(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeAnyOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeAnyOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeAny(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeAll(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeAllOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeAllOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeAll(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeTrueCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeTrueCountOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeTrueCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeTrueCount(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeFalseCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFalseCountOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFalseCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeFalseCount(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeTrueRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeTrueRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeTrueRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeTrueRatio(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeFalseRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFalseRatioOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFalseRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeFalseRatio(self, key_names, value_name, output_name);
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

pub fn withGroupCumulativeFirstTrueIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFirstTrueIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFirstTrueIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeFirstTrueIndex(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeLastTrueIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeLastTrueIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeLastTrueIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeLastTrueIndex(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeFirstFalseIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFirstFalseIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFirstFalseIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeFirstFalseIndex(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeLastFalseIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeLastFalseIndexOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeLastFalseIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeLastFalseIndex(self, key_names, value_name, output_name);
}

pub const withGroupCumFirstTrueIndex = withGroupCumulativeFirstTrueIndex;
pub const withGroupCumFirstTrueIndexOn = withGroupCumulativeFirstTrueIndexOn;
pub const withGroupCumLastTrueIndex = withGroupCumulativeLastTrueIndex;
pub const withGroupCumLastTrueIndexOn = withGroupCumulativeLastTrueIndexOn;
pub const withGroupCumFirstFalseIndex = withGroupCumulativeFirstFalseIndex;
pub const withGroupCumFirstFalseIndexOn = withGroupCumulativeFirstFalseIndexOn;
pub const withGroupCumLastFalseIndex = withGroupCumulativeLastFalseIndex;
pub const withGroupCumLastFalseIndexOn = withGroupCumulativeLastFalseIndexOn;

pub fn withGroupCumulativeSum(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeSumOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeSumOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeSum(self, key_names, value_name, output_name);
}

pub const withGroupCumSum = withGroupCumulativeSum;
pub const withGroupCumSumOn = withGroupCumulativeSumOn;

pub fn withGroupCumulativeMean(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeMeanOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeMeanOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeMean(self, key_names, value_name, output_name);
}

pub const withGroupCumMean = withGroupCumulativeMean;
pub const withGroupCumMeanOn = withGroupCumulativeMeanOn;

pub fn withGroupCumulativeWeightedMean(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedMeanOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedMeanOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedMean(self, key_names, value_name, weight_name, output_name);
}

pub const withGroupCumWeightedMean = withGroupCumulativeWeightedMean;
pub const withGroupCumWeightedMeanOn = withGroupCumulativeWeightedMeanOn;

pub fn withGroupCumulativeWeightedMedian(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedMedianOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedMedianOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedMedian(self, key_names, value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedQuantile(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8, q: f64) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedQuantileOn(self, key_names[0..], value_name, weight_name, output_name, q);
}

pub fn withGroupCumulativeWeightedQuantileOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8, q: f64) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedQuantile(self, key_names, value_name, weight_name, output_name, q);
}

pub fn withGroupCumulativeWeightedIqr(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedIqrOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedIqrOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedIqr(self, key_names, value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedMad(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedMadOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedMadOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedMad(self, key_names, value_name, weight_name, output_name);
}

pub const withGroupCumWeightedMedian = withGroupCumulativeWeightedMedian;
pub const withGroupCumWeightedMedianOn = withGroupCumulativeWeightedMedianOn;
pub const withGroupCumWeightedQuantile = withGroupCumulativeWeightedQuantile;
pub const withGroupCumWeightedQuantileOn = withGroupCumulativeWeightedQuantileOn;
pub const withGroupCumulativeWeightedIQR = withGroupCumulativeWeightedIqr;
pub const withGroupCumulativeWeightedIQROn = withGroupCumulativeWeightedIqrOn;
pub const withGroupCumulativeWeightedMAD = withGroupCumulativeWeightedMad;
pub const withGroupCumulativeWeightedMADOn = withGroupCumulativeWeightedMadOn;
pub const withGroupCumulativeWeightedMedianAbsDev = withGroupCumulativeWeightedMad;
pub const withGroupCumulativeWeightedMedianAbsDevOn = withGroupCumulativeWeightedMadOn;
pub const withGroupCumWeightedIqr = withGroupCumulativeWeightedIqr;
pub const withGroupCumWeightedIqrOn = withGroupCumulativeWeightedIqrOn;
pub const withGroupCumWeightedIQR = withGroupCumulativeWeightedIqr;
pub const withGroupCumWeightedIQROn = withGroupCumulativeWeightedIqrOn;
pub const withGroupCumWeightedMad = withGroupCumulativeWeightedMad;
pub const withGroupCumWeightedMadOn = withGroupCumulativeWeightedMadOn;
pub const withGroupCumWeightedMAD = withGroupCumulativeWeightedMad;
pub const withGroupCumWeightedMADOn = withGroupCumulativeWeightedMadOn;
pub const withGroupCumWeightedMedianAbsDev = withGroupCumulativeWeightedMad;
pub const withGroupCumWeightedMedianAbsDevOn = withGroupCumulativeWeightedMadOn;

pub fn withGroupCumulativeWeightedMode(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedModeOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedModeOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedMode(self, key_names, value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedModeWeight(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedModeWeightOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedModeWeightOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedModeWeight(self, key_names, value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedModeRatio(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedModeRatioOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedModeRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedModeRatio(self, key_names, value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedModeMargin(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedModeMarginOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedModeMarginOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedModeMargin(self, key_names, value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedModeMarginRatio(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedModeMarginRatioOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedModeMarginRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedModeMarginRatio(self, key_names, value_name, weight_name, output_name);
}

pub const withGroupCumWeightedMode = withGroupCumulativeWeightedMode;
pub const withGroupCumWeightedModeOn = withGroupCumulativeWeightedModeOn;
pub const withGroupCumWeightedModeWeight = withGroupCumulativeWeightedModeWeight;
pub const withGroupCumWeightedModeWeightOn = withGroupCumulativeWeightedModeWeightOn;
pub const withGroupCumWeightedModeRatio = withGroupCumulativeWeightedModeRatio;
pub const withGroupCumWeightedModeRatioOn = withGroupCumulativeWeightedModeRatioOn;
pub const withGroupCumWeightedModeMargin = withGroupCumulativeWeightedModeMargin;
pub const withGroupCumWeightedModeMarginOn = withGroupCumulativeWeightedModeMarginOn;
pub const withGroupCumWeightedModeMarginRatio = withGroupCumulativeWeightedModeMarginRatio;
pub const withGroupCumWeightedModeMarginRatioOn = withGroupCumulativeWeightedModeMarginRatioOn;

pub fn withGroupCumulativeWeightedEntropy(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedEntropyOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedEntropyOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedEntropy(self, key_names, value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedGiniImpurity(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedGiniImpurityOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedGiniImpurityOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedGiniImpurity(self, key_names, value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedPerplexity(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedPerplexityOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedPerplexityOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedPerplexity(self, key_names, value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedInverseSimpson(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedInverseSimpsonOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedInverseSimpsonOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedInverseSimpson(self, key_names, value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedSimpsonConcentration(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedSimpsonConcentrationOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedSimpsonConcentrationOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedSimpsonConcentration(self, key_names, value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedEvenness(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedEvennessOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedEvennessOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedEvenness(self, key_names, value_name, weight_name, output_name);
}

pub const withGroupCumulativeWeightedGini = withGroupCumulativeWeightedGiniImpurity;
pub const withGroupCumulativeWeightedGiniOn = withGroupCumulativeWeightedGiniImpurityOn;
pub const withGroupCumulativeWeightedConcentration = withGroupCumulativeWeightedSimpsonConcentration;
pub const withGroupCumulativeWeightedConcentrationOn = withGroupCumulativeWeightedSimpsonConcentrationOn;
pub const withGroupCumWeightedEntropy = withGroupCumulativeWeightedEntropy;
pub const withGroupCumWeightedEntropyOn = withGroupCumulativeWeightedEntropyOn;
pub const withGroupCumWeightedGiniImpurity = withGroupCumulativeWeightedGiniImpurity;
pub const withGroupCumWeightedGiniImpurityOn = withGroupCumulativeWeightedGiniImpurityOn;
pub const withGroupCumWeightedGini = withGroupCumulativeWeightedGiniImpurity;
pub const withGroupCumWeightedGiniOn = withGroupCumulativeWeightedGiniImpurityOn;
pub const withGroupCumWeightedPerplexity = withGroupCumulativeWeightedPerplexity;
pub const withGroupCumWeightedPerplexityOn = withGroupCumulativeWeightedPerplexityOn;
pub const withGroupCumWeightedInverseSimpson = withGroupCumulativeWeightedInverseSimpson;
pub const withGroupCumWeightedInverseSimpsonOn = withGroupCumulativeWeightedInverseSimpsonOn;
pub const withGroupCumWeightedSimpsonConcentration = withGroupCumulativeWeightedSimpsonConcentration;
pub const withGroupCumWeightedSimpsonConcentrationOn = withGroupCumulativeWeightedSimpsonConcentrationOn;
pub const withGroupCumWeightedConcentration = withGroupCumulativeWeightedSimpsonConcentration;
pub const withGroupCumWeightedConcentrationOn = withGroupCumulativeWeightedSimpsonConcentrationOn;
pub const withGroupCumWeightedEvenness = withGroupCumulativeWeightedEvenness;
pub const withGroupCumWeightedEvennessOn = withGroupCumulativeWeightedEvennessOn;

pub fn withGroupCumulativeWeightedDot(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedDotOn(self, key_names[0..], lhs_name, rhs_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedDotOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedDot(self, key_names, lhs_name, rhs_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedCosineSimilarity(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedCosineSimilarityOn(self, key_names[0..], lhs_name, rhs_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedCosineSimilarityOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedCosineSimilarity(self, key_names, lhs_name, rhs_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedSquaredEuclideanDistance(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedSquaredEuclideanDistanceOn(self, key_names[0..], lhs_name, rhs_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedSquaredEuclideanDistanceOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedSquaredEuclideanDistance(self, key_names, lhs_name, rhs_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedEuclideanDistance(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedEuclideanDistanceOn(self, key_names[0..], lhs_name, rhs_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedEuclideanDistanceOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedEuclideanDistance(self, key_names, lhs_name, rhs_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedManhattanDistance(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedManhattanDistanceOn(self, key_names[0..], lhs_name, rhs_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedManhattanDistanceOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedManhattanDistance(self, key_names, lhs_name, rhs_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedChebyshevDistance(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedChebyshevDistanceOn(self, key_names[0..], lhs_name, rhs_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedChebyshevDistanceOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedChebyshevDistance(self, key_names, lhs_name, rhs_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedCanberraDistance(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedCanberraDistanceOn(self, key_names[0..], lhs_name, rhs_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedCanberraDistanceOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedCanberraDistance(self, key_names, lhs_name, rhs_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedBrayCurtisDistance(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedBrayCurtisDistanceOn(self, key_names[0..], lhs_name, rhs_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedBrayCurtisDistanceOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedBrayCurtisDistance(self, key_names, lhs_name, rhs_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedMeanError(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedMeanErrorOn(self, key_names[0..], lhs_name, rhs_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedMeanErrorOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedMeanError(self, key_names, lhs_name, rhs_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedMae(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedMaeOn(self, key_names[0..], lhs_name, rhs_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedMaeOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedMae(self, key_names, lhs_name, rhs_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedMse(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedMseOn(self, key_names[0..], lhs_name, rhs_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedMseOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedMse(self, key_names, lhs_name, rhs_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedRmse(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedRmseOn(self, key_names[0..], lhs_name, rhs_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedRmseOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedRmse(self, key_names, lhs_name, rhs_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedMape(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedMapeOn(self, key_names[0..], lhs_name, rhs_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedMapeOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedMape(self, key_names, lhs_name, rhs_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedSmape(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedSmapeOn(self, key_names[0..], lhs_name, rhs_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedSmapeOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedSmape(self, key_names, lhs_name, rhs_name, weight_name, output_name);
}

pub const withGroupCumulativeWeightedCosine = withGroupCumulativeWeightedCosineSimilarity;
pub const withGroupCumulativeWeightedCosineOn = withGroupCumulativeWeightedCosineSimilarityOn;
pub const withGroupCumulativeWeightedBias = withGroupCumulativeWeightedMeanError;
pub const withGroupCumulativeWeightedBiasOn = withGroupCumulativeWeightedMeanErrorOn;
pub const withGroupCumWeightedDot = withGroupCumulativeWeightedDot;
pub const withGroupCumWeightedDotOn = withGroupCumulativeWeightedDotOn;
pub const withGroupCumWeightedCosineSimilarity = withGroupCumulativeWeightedCosineSimilarity;
pub const withGroupCumWeightedCosineSimilarityOn = withGroupCumulativeWeightedCosineSimilarityOn;
pub const withGroupCumWeightedSquaredEuclideanDistance = withGroupCumulativeWeightedSquaredEuclideanDistance;
pub const withGroupCumWeightedSquaredEuclideanDistanceOn = withGroupCumulativeWeightedSquaredEuclideanDistanceOn;
pub const withGroupCumWeightedEuclideanDistance = withGroupCumulativeWeightedEuclideanDistance;
pub const withGroupCumWeightedEuclideanDistanceOn = withGroupCumulativeWeightedEuclideanDistanceOn;
pub const withGroupCumWeightedManhattanDistance = withGroupCumulativeWeightedManhattanDistance;
pub const withGroupCumWeightedManhattanDistanceOn = withGroupCumulativeWeightedManhattanDistanceOn;
pub const withGroupCumWeightedChebyshevDistance = withGroupCumulativeWeightedChebyshevDistance;
pub const withGroupCumWeightedChebyshevDistanceOn = withGroupCumulativeWeightedChebyshevDistanceOn;
pub const withGroupCumWeightedCanberraDistance = withGroupCumulativeWeightedCanberraDistance;
pub const withGroupCumWeightedCanberraDistanceOn = withGroupCumulativeWeightedCanberraDistanceOn;
pub const withGroupCumWeightedBrayCurtisDistance = withGroupCumulativeWeightedBrayCurtisDistance;
pub const withGroupCumWeightedBrayCurtisDistanceOn = withGroupCumulativeWeightedBrayCurtisDistanceOn;
pub const withGroupCumWeightedMeanError = withGroupCumulativeWeightedMeanError;
pub const withGroupCumWeightedMeanErrorOn = withGroupCumulativeWeightedMeanErrorOn;
pub const withGroupCumWeightedMae = withGroupCumulativeWeightedMae;
pub const withGroupCumWeightedMaeOn = withGroupCumulativeWeightedMaeOn;
pub const withGroupCumWeightedMse = withGroupCumulativeWeightedMse;
pub const withGroupCumWeightedMseOn = withGroupCumulativeWeightedMseOn;
pub const withGroupCumWeightedRmse = withGroupCumulativeWeightedRmse;
pub const withGroupCumWeightedRmseOn = withGroupCumulativeWeightedRmseOn;
pub const withGroupCumWeightedMape = withGroupCumulativeWeightedMape;
pub const withGroupCumWeightedMapeOn = withGroupCumulativeWeightedMapeOn;
pub const withGroupCumWeightedSmape = withGroupCumulativeWeightedSmape;
pub const withGroupCumWeightedSmapeOn = withGroupCumulativeWeightedSmapeOn;
pub const withGroupCumWeightedCosine = withGroupCumulativeWeightedCosineSimilarity;
pub const withGroupCumWeightedCosineOn = withGroupCumulativeWeightedCosineSimilarityOn;
pub const withGroupCumWeightedBias = withGroupCumulativeWeightedMeanError;
pub const withGroupCumWeightedBiasOn = withGroupCumulativeWeightedMeanErrorOn;

pub fn withGroupCumulativeWeightedCovariance(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedCovarianceOn(self, key_names[0..], lhs_name, rhs_name, weight_name, output_name, correction);
}

pub fn withGroupCumulativeWeightedCovarianceOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedCovariance(self, key_names, lhs_name, rhs_name, weight_name, output_name, correction);
}

pub const withGroupCumulativeWeightedCov = withGroupCumulativeWeightedCovariance;
pub const withGroupCumulativeWeightedCovOn = withGroupCumulativeWeightedCovarianceOn;

pub fn withGroupCumulativeWeightedCorrelation(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedCorrelationOn(self, key_names[0..], lhs_name, rhs_name, weight_name, output_name, correction);
}

pub fn withGroupCumulativeWeightedCorrelationOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedCorrelation(self, key_names, lhs_name, rhs_name, weight_name, output_name, correction);
}

pub const withGroupCumulativeWeightedCorr = withGroupCumulativeWeightedCorrelation;
pub const withGroupCumulativeWeightedCorrOn = withGroupCumulativeWeightedCorrelationOn;

pub fn withGroupCumulativeWeightedBeta(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedBetaOn(self, key_names[0..], lhs_name, rhs_name, weight_name, output_name, correction);
}

pub fn withGroupCumulativeWeightedBetaOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedBeta(self, key_names, lhs_name, rhs_name, weight_name, output_name, correction);
}

pub const withGroupCumWeightedCovariance = withGroupCumulativeWeightedCovariance;
pub const withGroupCumWeightedCovarianceOn = withGroupCumulativeWeightedCovarianceOn;
pub const withGroupCumWeightedCov = withGroupCumulativeWeightedCovariance;
pub const withGroupCumWeightedCovOn = withGroupCumulativeWeightedCovarianceOn;
pub const withGroupCumWeightedCorrelation = withGroupCumulativeWeightedCorrelation;
pub const withGroupCumWeightedCorrelationOn = withGroupCumulativeWeightedCorrelationOn;
pub const withGroupCumWeightedCorr = withGroupCumulativeWeightedCorrelation;
pub const withGroupCumWeightedCorrOn = withGroupCumulativeWeightedCorrelationOn;
pub const withGroupCumWeightedBeta = withGroupCumulativeWeightedBeta;
pub const withGroupCumWeightedBetaOn = withGroupCumulativeWeightedBetaOn;

pub fn withGroupCumulativeWeightedSem(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedSemOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedSemOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedSem(self, key_names, value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedCv(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedCvOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedCvOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedCv(self, key_names, value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedFano(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedFanoOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedFanoOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedFano(self, key_names, value_name, weight_name, output_name);
}

pub const withGroupCumulativeWeightedSEM = withGroupCumulativeWeightedSem;
pub const withGroupCumulativeWeightedSEMOn = withGroupCumulativeWeightedSemOn;
pub const withGroupCumulativeWeightedCV = withGroupCumulativeWeightedCv;
pub const withGroupCumulativeWeightedCVOn = withGroupCumulativeWeightedCvOn;
pub const withGroupCumWeightedSem = withGroupCumulativeWeightedSem;
pub const withGroupCumWeightedSemOn = withGroupCumulativeWeightedSemOn;
pub const withGroupCumWeightedSEM = withGroupCumulativeWeightedSem;
pub const withGroupCumWeightedSEMOn = withGroupCumulativeWeightedSemOn;
pub const withGroupCumWeightedCv = withGroupCumulativeWeightedCv;
pub const withGroupCumWeightedCvOn = withGroupCumulativeWeightedCvOn;
pub const withGroupCumWeightedCV = withGroupCumulativeWeightedCv;
pub const withGroupCumWeightedCVOn = withGroupCumulativeWeightedCvOn;
pub const withGroupCumWeightedFano = withGroupCumulativeWeightedFano;
pub const withGroupCumWeightedFanoOn = withGroupCumulativeWeightedFanoOn;

pub fn withGroupCumulativeWeightedMeanSquare(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedMeanSquareOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedMeanSquareOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedMeanSquare(self, key_names, value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedRms(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedRmsOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedRmsOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedRms(self, key_names, value_name, weight_name, output_name);
}

pub const withGroupCumulativeWeightedMeanSquared = withGroupCumulativeWeightedMeanSquare;
pub const withGroupCumulativeWeightedMeanSquaredOn = withGroupCumulativeWeightedMeanSquareOn;
pub const withGroupCumulativeWeightedMeanSq = withGroupCumulativeWeightedMeanSquare;
pub const withGroupCumulativeWeightedMeanSqOn = withGroupCumulativeWeightedMeanSquareOn;
pub const withGroupCumulativeWeightedRMS = withGroupCumulativeWeightedRms;
pub const withGroupCumulativeWeightedRMSOn = withGroupCumulativeWeightedRmsOn;
pub const withGroupCumWeightedMeanSquare = withGroupCumulativeWeightedMeanSquare;
pub const withGroupCumWeightedMeanSquareOn = withGroupCumulativeWeightedMeanSquareOn;
pub const withGroupCumWeightedMeanSquared = withGroupCumulativeWeightedMeanSquare;
pub const withGroupCumWeightedMeanSquaredOn = withGroupCumulativeWeightedMeanSquareOn;
pub const withGroupCumWeightedMeanSq = withGroupCumulativeWeightedMeanSquare;
pub const withGroupCumWeightedMeanSqOn = withGroupCumulativeWeightedMeanSquareOn;
pub const withGroupCumWeightedRms = withGroupCumulativeWeightedRms;
pub const withGroupCumWeightedRmsOn = withGroupCumulativeWeightedRmsOn;
pub const withGroupCumWeightedRMS = withGroupCumulativeWeightedRms;
pub const withGroupCumWeightedRMSOn = withGroupCumulativeWeightedRmsOn;

pub fn withGroupCumulativeWeightedMeanAbs(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedMeanAbsOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedMeanAbsOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedMeanAbs(self, key_names, value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedL1Norm(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedL1NormOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedL1NormOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedL1Norm(self, key_names, value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedL2Norm(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedL2NormOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedL2NormOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedL2Norm(self, key_names, value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedMaxAbs(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedMaxAbsOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedMaxAbsOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedMaxAbs(self, key_names, value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedMinAbs(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedMinAbsOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedMinAbsOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedMinAbs(self, key_names, value_name, weight_name, output_name);
}

pub const withGroupCumulativeWeightedL1 = withGroupCumulativeWeightedL1Norm;
pub const withGroupCumulativeWeightedL1On = withGroupCumulativeWeightedL1NormOn;
pub const withGroupCumulativeWeightedL2 = withGroupCumulativeWeightedL2Norm;
pub const withGroupCumulativeWeightedL2On = withGroupCumulativeWeightedL2NormOn;
pub const withGroupCumulativeWeightedMaxAbsolute = withGroupCumulativeWeightedMaxAbs;
pub const withGroupCumulativeWeightedMaxAbsoluteOn = withGroupCumulativeWeightedMaxAbsOn;
pub const withGroupCumulativeWeightedMinAbsolute = withGroupCumulativeWeightedMinAbs;
pub const withGroupCumulativeWeightedMinAbsoluteOn = withGroupCumulativeWeightedMinAbsOn;
pub const withGroupCumWeightedMeanAbs = withGroupCumulativeWeightedMeanAbs;
pub const withGroupCumWeightedMeanAbsOn = withGroupCumulativeWeightedMeanAbsOn;
pub const withGroupCumWeightedL1Norm = withGroupCumulativeWeightedL1Norm;
pub const withGroupCumWeightedL1NormOn = withGroupCumulativeWeightedL1NormOn;
pub const withGroupCumWeightedL1 = withGroupCumulativeWeightedL1Norm;
pub const withGroupCumWeightedL1On = withGroupCumulativeWeightedL1NormOn;
pub const withGroupCumWeightedL2Norm = withGroupCumulativeWeightedL2Norm;
pub const withGroupCumWeightedL2NormOn = withGroupCumulativeWeightedL2NormOn;
pub const withGroupCumWeightedL2 = withGroupCumulativeWeightedL2Norm;
pub const withGroupCumWeightedL2On = withGroupCumulativeWeightedL2NormOn;
pub const withGroupCumWeightedMaxAbs = withGroupCumulativeWeightedMaxAbs;
pub const withGroupCumWeightedMaxAbsOn = withGroupCumulativeWeightedMaxAbsOn;
pub const withGroupCumWeightedMinAbs = withGroupCumulativeWeightedMinAbs;
pub const withGroupCumWeightedMinAbsOn = withGroupCumulativeWeightedMinAbsOn;

pub fn withGroupCumulativeWeightedGeometricMean(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedGeometricMeanOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedGeometricMeanOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedGeometricMean(self, key_names, value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedHarmonicMean(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedHarmonicMeanOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedHarmonicMeanOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedHarmonicMean(self, key_names, value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedLogSumExp(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedLogSumExpOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedLogSumExpOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedLogSumExp(self, key_names, value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedLogMeanExp(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedLogMeanExpOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedLogMeanExpOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedLogMeanExp(self, key_names, value_name, weight_name, output_name);
}

pub const withGroupCumulativeWeightedGeoMean = withGroupCumulativeWeightedGeometricMean;
pub const withGroupCumulativeWeightedGeoMeanOn = withGroupCumulativeWeightedGeometricMeanOn;
pub const withGroupCumulativeWeightedHarmMean = withGroupCumulativeWeightedHarmonicMean;
pub const withGroupCumulativeWeightedHarmMeanOn = withGroupCumulativeWeightedHarmonicMeanOn;
pub const withGroupCumulativeWeightedLogsumexp = withGroupCumulativeWeightedLogSumExp;
pub const withGroupCumulativeWeightedLogsumexpOn = withGroupCumulativeWeightedLogSumExpOn;
pub const withGroupCumulativeWeightedLogmeanexp = withGroupCumulativeWeightedLogMeanExp;
pub const withGroupCumulativeWeightedLogmeanexpOn = withGroupCumulativeWeightedLogMeanExpOn;
pub const withGroupCumWeightedGeometricMean = withGroupCumulativeWeightedGeometricMean;
pub const withGroupCumWeightedGeometricMeanOn = withGroupCumulativeWeightedGeometricMeanOn;
pub const withGroupCumWeightedGeoMean = withGroupCumulativeWeightedGeometricMean;
pub const withGroupCumWeightedGeoMeanOn = withGroupCumulativeWeightedGeometricMeanOn;
pub const withGroupCumWeightedHarmonicMean = withGroupCumulativeWeightedHarmonicMean;
pub const withGroupCumWeightedHarmonicMeanOn = withGroupCumulativeWeightedHarmonicMeanOn;
pub const withGroupCumWeightedHarmMean = withGroupCumulativeWeightedHarmonicMean;
pub const withGroupCumWeightedHarmMeanOn = withGroupCumulativeWeightedHarmonicMeanOn;
pub const withGroupCumWeightedLogSumExp = withGroupCumulativeWeightedLogSumExp;
pub const withGroupCumWeightedLogSumExpOn = withGroupCumulativeWeightedLogSumExpOn;
pub const withGroupCumWeightedLogsumexp = withGroupCumulativeWeightedLogSumExp;
pub const withGroupCumWeightedLogsumexpOn = withGroupCumulativeWeightedLogSumExpOn;
pub const withGroupCumWeightedLogMeanExp = withGroupCumulativeWeightedLogMeanExp;
pub const withGroupCumWeightedLogMeanExpOn = withGroupCumulativeWeightedLogMeanExpOn;
pub const withGroupCumWeightedLogmeanexp = withGroupCumulativeWeightedLogMeanExp;
pub const withGroupCumWeightedLogmeanexpOn = withGroupCumulativeWeightedLogMeanExpOn;

pub fn withGroupCumulativeWeightedVariance(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedVarianceOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedVarianceOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedVariance(self, key_names, value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedStddev(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeWeightedStddevOn(self, key_names[0..], value_name, weight_name, output_name);
}

pub fn withGroupCumulativeWeightedStddevOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeWeightedStddev(self, key_names, value_name, weight_name, output_name);
}

pub const withGroupCumulativeWeightedVar = withGroupCumulativeWeightedVariance;
pub const withGroupCumulativeWeightedVarOn = withGroupCumulativeWeightedVarianceOn;
pub const withGroupCumulativeWeightedStd = withGroupCumulativeWeightedStddev;
pub const withGroupCumulativeWeightedStdOn = withGroupCumulativeWeightedStddevOn;
pub const withGroupCumWeightedVariance = withGroupCumulativeWeightedVariance;
pub const withGroupCumWeightedVarianceOn = withGroupCumulativeWeightedVarianceOn;
pub const withGroupCumWeightedVar = withGroupCumulativeWeightedVariance;
pub const withGroupCumWeightedVarOn = withGroupCumulativeWeightedVarianceOn;
pub const withGroupCumWeightedStddev = withGroupCumulativeWeightedStddev;
pub const withGroupCumWeightedStddevOn = withGroupCumulativeWeightedStddevOn;
pub const withGroupCumWeightedStd = withGroupCumulativeWeightedStddev;
pub const withGroupCumWeightedStdOn = withGroupCumulativeWeightedStddevOn;

pub fn withGroupCumulativeProduct(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeProductOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeProductOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeProduct(self, key_names, value_name, output_name);
}

pub const withGroupCumProduct = withGroupCumulativeProduct;
pub const withGroupCumProductOn = withGroupCumulativeProductOn;
pub const withGroupCumProd = withGroupCumulativeProduct;
pub const withGroupCumProdOn = withGroupCumulativeProductOn;

pub fn withGroupCumulativeMin(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeMinOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeMinOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeMin(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeMax(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeMaxOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeMaxOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeMax(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeVariance(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeVarianceOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeVarianceOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeVariance(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeStddev(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeStddevOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeStddevOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeStddev(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeSem(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeSemOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeSemOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeSem(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeCv(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeCvOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeCvOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeCv(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeFano(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeFanoOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeFanoOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeFano(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeSkewness(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeSkewnessOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeSkewnessOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeSkewness(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeKurtosis(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeKurtosisOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeKurtosisOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeKurtosis(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeMeanAbs(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeMeanAbsOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeMeanAbsOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeMeanAbs(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeMeanSquare(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeMeanSquareOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeMeanSquareOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeMeanSquare(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeRms(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeRmsOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeRmsOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeRms(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeMaxAbs(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeMaxAbsOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeMaxAbsOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeMaxAbs(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeMinAbs(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeMinAbsOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeMinAbsOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeMinAbs(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeL1Norm(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeL1NormOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeL1NormOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeL1Norm(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeL2Norm(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeL2NormOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeL2NormOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeL2Norm(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeRange(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeRangeOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeRangeOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeRange(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeMidrange(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeMidrangeOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeMidrangeOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeMidrange(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeRangeCoeff(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeRangeCoeffOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeRangeCoeffOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeRangeCoeff(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeLogSumExp(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeLogSumExpOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeLogSumExpOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeLogSumExp(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeLogMeanExp(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeLogMeanExpOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeLogMeanExpOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeLogMeanExp(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeGeometricMean(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeGeometricMeanOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeGeometricMeanOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeGeometricMean(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeHarmonicMean(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeHarmonicMeanOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeHarmonicMeanOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeHarmonicMean(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeArgMin(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeArgMinOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeArgMinOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeArgMin(self, key_names, value_name, output_name);
}

pub fn withGroupCumulativeArgMax(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupCumulativeArgMaxOn(self, key_names[0..], value_name, output_name);
}

pub fn withGroupCumulativeArgMaxOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupCumulativeArgMax(self, key_names, value_name, output_name);
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

pub fn withGroupRowNumber(self: anytype, key_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupRowNumberOn(self, key_names[0..], output_name);
}

pub fn withGroupRowNumberOn(self: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupRowNumber(self, key_names, output_name);
}

pub const withGroupCumCount = withGroupRowNumber;
pub const withGroupCumCountOn = withGroupRowNumberOn;

pub fn withGroupSize(self: anytype, key_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupSizeOn(self, key_names[0..], output_name);
}

pub fn withGroupSizeOn(self: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupSize(self, key_names, output_name);
}

pub const withGroupCount = withGroupSize;
pub const withGroupCountOn = withGroupSizeOn;

pub fn withGroupReverseRowNumber(self: anytype, key_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const key_names = [_][]const u8{key_name};
    return withGroupReverseRowNumberOn(self, key_names[0..], output_name);
}

pub fn withGroupReverseRowNumberOn(self: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.withGroupReverseRowNumber(self, key_names, output_name);
}

pub const withGroupReverseCumCount = withGroupReverseRowNumber;
pub const withGroupReverseCumCountOn = withGroupReverseRowNumberOn;

pub fn valueCounts(self: anytype, key_name: []const u8) DeviceDataError!void {
    return valueCountsAs(self, key_name, "count");
}

pub fn valueCountsAs(self: anytype, key_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByCount(key_name, output_name);
}

pub fn valueCountsOn(self: anytype, key_names: []const []const u8) DeviceDataError!void {
    return valueCountsOnAs(self, key_names, "count");
}

pub fn valueCountsOnAs(self: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByCountOn(key_names, output_name);
}

pub fn valueCountsSorted(self: anytype, key_name: []const u8) DeviceDataError!void {
    return valueCountsSortedAs(self, key_name, "count");
}

pub fn valueCountsSortedAs(self: anytype, key_name: []const u8, output_name: []const u8) DeviceDataError!void {
    try self.valueCountsAs(key_name, output_name);
    try self.sortBy(output_name, .{ .descending = true });
}

pub fn valueCountsOnSorted(self: anytype, key_names: []const []const u8) DeviceDataError!void {
    return valueCountsOnSortedAs(self, key_names, "count");
}

pub fn valueCountsOnSortedAs(self: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    try self.valueCountsOnAs(key_names, output_name);
    try self.sortBy(output_name, .{ .descending = true });
}

pub fn valueCountsSortedOn(self: anytype, key_names: []const []const u8) DeviceDataError!void {
    return valueCountsOnSorted(self, key_names);
}

pub fn valueCountsSortedOnAs(self: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return valueCountsOnSortedAs(self, key_names, output_name);
}

pub fn groupByValue(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8, aggregation: DeviceLazyGroupByAggregation) DeviceDataError!void {
    return lazy_group_mod.groupByValue(self, key_name, value_name, output_name, aggregation);
}

pub fn groupByValueOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, aggregation: DeviceLazyGroupByAggregation) DeviceDataError!void {
    return lazy_group_mod.groupByValueOn(self, key_names, value_name, output_name, aggregation);
}

pub fn groupByWeighted(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8, aggregation: DeviceLazyWeightedGroupByAggregation) DeviceDataError!void {
    return lazy_group_mod.groupByWeighted(self, key_name, value_name, weight_name, output_name, aggregation);
}

pub fn groupByWeightedOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8, aggregation: DeviceLazyWeightedGroupByAggregation) DeviceDataError!void {
    return lazy_group_mod.groupByWeightedOn(self, key_names, value_name, weight_name, output_name, aggregation);
}

pub fn groupBySum(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .sum);
}

pub fn groupBySumOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .sum);
}

pub fn groupByProd(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .prod);
}

pub fn groupByProduct(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByProd(key_name, value_name, output_name);
}

pub fn groupByProdOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .prod);
}

pub fn groupByProductOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByProdOn(key_names, value_name, output_name);
}

pub fn groupByMin(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .min);
}

pub fn groupByMinOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .min);
}

pub fn groupByMax(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .max);
}

pub fn groupByMaxOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .max);
}

pub fn groupByMean(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .mean);
}

pub fn groupByMeanOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .mean);
}

pub fn groupByFirst(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .first);
}

pub fn groupByFirstOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .first);
}

pub fn groupByLast(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .last);
}

pub fn groupByLastOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .last);
}

pub fn groupByFirstRow(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .first_row);
}

pub fn groupByFirstRowOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .first_row);
}

pub fn groupByLastRow(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .last_row);
}

pub fn groupByLastRowOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .last_row);
}

pub fn groupByNth(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8, n: usize) DeviceDataError!void {
    return lazy_group_mod.groupByValueIndex(self, key_name, value_name, output_name, .nth, n);
}

pub fn groupByNthOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, n: usize) DeviceDataError!void {
    return lazy_group_mod.groupByValueOnIndex(self, key_names, value_name, output_name, .nth, n);
}

pub fn groupByNthRow(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8, n: usize) DeviceDataError!void {
    return lazy_group_mod.groupByValueIndex(self, key_name, value_name, output_name, .nth_row, n);
}

pub fn groupByNthRowOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, n: usize) DeviceDataError!void {
    return lazy_group_mod.groupByValueOnIndex(self, key_names, value_name, output_name, .nth_row, n);
}

pub fn groupByNthIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8, n: usize) DeviceDataError!void {
    return lazy_group_mod.groupByValueIndex(self, key_name, value_name, output_name, .nth_index, n);
}

pub fn groupByNthIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, n: usize) DeviceDataError!void {
    return lazy_group_mod.groupByValueOnIndex(self, key_names, value_name, output_name, .nth_index, n);
}

pub fn groupByNthRowIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8, n: usize) DeviceDataError!void {
    return lazy_group_mod.groupByValueIndex(self, key_name, value_name, output_name, .nth_row_index, n);
}

pub fn groupByNthRowIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, n: usize) DeviceDataError!void {
    return lazy_group_mod.groupByValueOnIndex(self, key_names, value_name, output_name, .nth_row_index, n);
}

pub fn groupByNUnique(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .n_unique);
}

pub fn groupByNUniqueOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .n_unique);
}

pub const groupByNunique = groupByNUnique;
pub const groupByNuniqueOn = groupByNUniqueOn;

pub fn groupByMode(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .mode);
}

pub fn groupByModeOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .mode);
}

pub fn groupByModeCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .mode_count);
}

pub fn groupByModeCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .mode_count);
}

pub fn groupByModeRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .mode_ratio);
}

pub fn groupByModeRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .mode_ratio);
}

pub fn groupByModeMargin(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .mode_margin);
}

pub fn groupByModeMarginOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .mode_margin);
}

pub fn groupByModeMarginRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .mode_margin_ratio);
}

pub fn groupByModeMarginRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .mode_margin_ratio);
}

pub fn groupByEntropy(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .entropy);
}

pub fn groupByEntropyOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .entropy);
}

pub fn groupByGiniImpurity(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .gini_impurity);
}

pub fn groupByGiniImpurityOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .gini_impurity);
}

pub const groupByGini = groupByGiniImpurity;
pub const groupByGiniOn = groupByGiniImpurityOn;

pub fn groupByPerplexity(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .perplexity);
}

pub fn groupByPerplexityOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .perplexity);
}

pub fn groupByInverseSimpson(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .inverse_simpson);
}

pub fn groupByInverseSimpsonOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .inverse_simpson);
}

pub fn groupBySimpsonConcentration(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .simpson_concentration);
}

pub fn groupBySimpsonConcentrationOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .simpson_concentration);
}

pub const groupByConcentration = groupBySimpsonConcentration;
pub const groupByConcentrationOn = groupBySimpsonConcentrationOn;

pub fn groupByEvenness(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .evenness);
}

pub fn groupByEvennessOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .evenness);
}

pub fn groupByGiniMeanDiff(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .gini_mean_diff);
}

pub fn groupByGiniMeanDiffOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .gini_mean_diff);
}

pub fn groupByGiniCoefficient(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .gini_coefficient);
}

pub fn groupByGiniCoefficientOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .gini_coefficient);
}

pub const groupByGiniCoeff = groupByGiniCoefficient;
pub const groupByGiniCoeffOn = groupByGiniCoefficientOn;

pub fn groupByWeightedMean(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeighted(key_name, value_name, weight_name, output_name, .weighted_mean);
}

pub fn groupByWeightedMeanOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeightedOn(key_names, value_name, weight_name, output_name, .weighted_mean);
}

pub fn groupByWeightedMeanSquare(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeighted(key_name, value_name, weight_name, output_name, .weighted_mean_square);
}

pub fn groupByWeightedMeanSquareOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeightedOn(key_names, value_name, weight_name, output_name, .weighted_mean_square);
}

pub fn groupByWeightedRms(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeighted(key_name, value_name, weight_name, output_name, .weighted_rms);
}

pub fn groupByWeightedRmsOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeightedOn(key_names, value_name, weight_name, output_name, .weighted_rms);
}

pub const groupByWeightedMeanSquared = groupByWeightedMeanSquare;
pub const groupByWeightedMeanSquaredOn = groupByWeightedMeanSquareOn;
pub const groupByWeightedMeanSq = groupByWeightedMeanSquare;
pub const groupByWeightedMeanSqOn = groupByWeightedMeanSquareOn;
pub const groupByWeightedRMS = groupByWeightedRms;
pub const groupByWeightedRMSOn = groupByWeightedRmsOn;

pub fn groupByWeightedMeanAbs(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeighted(key_name, value_name, weight_name, output_name, .weighted_mean_abs);
}

pub fn groupByWeightedMeanAbsOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeightedOn(key_names, value_name, weight_name, output_name, .weighted_mean_abs);
}

pub fn groupByWeightedL1Norm(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeighted(key_name, value_name, weight_name, output_name, .weighted_l1_norm);
}

pub fn groupByWeightedL1NormOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeightedOn(key_names, value_name, weight_name, output_name, .weighted_l1_norm);
}

pub fn groupByWeightedL2Norm(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeighted(key_name, value_name, weight_name, output_name, .weighted_l2_norm);
}

pub fn groupByWeightedL2NormOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeightedOn(key_names, value_name, weight_name, output_name, .weighted_l2_norm);
}

pub fn groupByWeightedMaxAbs(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeighted(key_name, value_name, weight_name, output_name, .weighted_max_abs);
}

pub fn groupByWeightedMaxAbsOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeightedOn(key_names, value_name, weight_name, output_name, .weighted_max_abs);
}

pub fn groupByWeightedMinAbs(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeighted(key_name, value_name, weight_name, output_name, .weighted_min_abs);
}

pub fn groupByWeightedMinAbsOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeightedOn(key_names, value_name, weight_name, output_name, .weighted_min_abs);
}

pub const groupByWeightedL1 = groupByWeightedL1Norm;
pub const groupByWeightedL1On = groupByWeightedL1NormOn;
pub const groupByWeightedL2 = groupByWeightedL2Norm;
pub const groupByWeightedL2On = groupByWeightedL2NormOn;
pub const groupByWeightedMaxAbsolute = groupByWeightedMaxAbs;
pub const groupByWeightedMaxAbsoluteOn = groupByWeightedMaxAbsOn;
pub const groupByWeightedMinAbsolute = groupByWeightedMinAbs;
pub const groupByWeightedMinAbsoluteOn = groupByWeightedMinAbsOn;

pub fn groupByWeightedGeometricMean(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeighted(key_name, value_name, weight_name, output_name, .weighted_geometric_mean);
}

pub fn groupByWeightedGeometricMeanOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeightedOn(key_names, value_name, weight_name, output_name, .weighted_geometric_mean);
}

pub fn groupByWeightedHarmonicMean(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeighted(key_name, value_name, weight_name, output_name, .weighted_harmonic_mean);
}

pub fn groupByWeightedHarmonicMeanOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeightedOn(key_names, value_name, weight_name, output_name, .weighted_harmonic_mean);
}

pub const groupByWeightedGeoMean = groupByWeightedGeometricMean;
pub const groupByWeightedGeoMeanOn = groupByWeightedGeometricMeanOn;
pub const groupByWeightedHarmMean = groupByWeightedHarmonicMean;
pub const groupByWeightedHarmMeanOn = groupByWeightedHarmonicMeanOn;

pub fn groupByWeightedLogSumExp(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeighted(key_name, value_name, weight_name, output_name, .weighted_logsumexp);
}

pub fn groupByWeightedLogSumExpOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeightedOn(key_names, value_name, weight_name, output_name, .weighted_logsumexp);
}

pub fn groupByWeightedLogMeanExp(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeighted(key_name, value_name, weight_name, output_name, .weighted_logmeanexp);
}

pub fn groupByWeightedLogMeanExpOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeightedOn(key_names, value_name, weight_name, output_name, .weighted_logmeanexp);
}

pub const groupByWeightedLogsumexp = groupByWeightedLogSumExp;
pub const groupByWeightedLogsumexpOn = groupByWeightedLogSumExpOn;
pub const groupByWeightedLogmeanexp = groupByWeightedLogMeanExp;
pub const groupByWeightedLogmeanexpOn = groupByWeightedLogMeanExpOn;

pub fn groupByWeightedVariance(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeighted(key_name, value_name, weight_name, output_name, .weighted_variance);
}

pub fn groupByWeightedVarianceOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeightedOn(key_names, value_name, weight_name, output_name, .weighted_variance);
}

pub const groupByWeightedVar = groupByWeightedVariance;
pub const groupByWeightedVarOn = groupByWeightedVarianceOn;

pub fn groupByWeightedStddev(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeighted(key_name, value_name, weight_name, output_name, .weighted_stddev);
}

pub fn groupByWeightedStddevOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeightedOn(key_names, value_name, weight_name, output_name, .weighted_stddev);
}

pub const groupByWeightedStd = groupByWeightedStddev;
pub const groupByWeightedStdOn = groupByWeightedStddevOn;

pub fn groupByWeightedSem(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeighted(key_name, value_name, weight_name, output_name, .weighted_sem);
}

pub fn groupByWeightedSemOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeightedOn(key_names, value_name, weight_name, output_name, .weighted_sem);
}

pub fn groupByWeightedCv(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeighted(key_name, value_name, weight_name, output_name, .weighted_cv);
}

pub fn groupByWeightedCvOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeightedOn(key_names, value_name, weight_name, output_name, .weighted_cv);
}

pub fn groupByWeightedFano(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeighted(key_name, value_name, weight_name, output_name, .weighted_fano);
}

pub fn groupByWeightedFanoOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeightedOn(key_names, value_name, weight_name, output_name, .weighted_fano);
}

pub const groupByWeightedSEM = groupByWeightedSem;
pub const groupByWeightedSEMOn = groupByWeightedSemOn;
pub const groupByWeightedCV = groupByWeightedCv;
pub const groupByWeightedCVOn = groupByWeightedCvOn;

pub fn groupByWeightedQuantile(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8, q: f64) DeviceDataError!void {
    return lazy_group_mod.groupByWeightedQuantile(self, key_name, value_name, weight_name, output_name, .weighted_quantile, q);
}

pub fn groupByWeightedQuantileOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8, q: f64) DeviceDataError!void {
    return lazy_group_mod.groupByWeightedOnQuantile(self, key_names, value_name, weight_name, output_name, .weighted_quantile, q);
}

pub fn groupByWeightedMedian(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeighted(key_name, value_name, weight_name, output_name, .weighted_median);
}

pub fn groupByWeightedMedianOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeightedOn(key_names, value_name, weight_name, output_name, .weighted_median);
}

pub fn groupByWeightedIqr(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeighted(key_name, value_name, weight_name, output_name, .weighted_iqr);
}

pub fn groupByWeightedIqrOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeightedOn(key_names, value_name, weight_name, output_name, .weighted_iqr);
}

pub const groupByWeightedIQR = groupByWeightedIqr;
pub const groupByWeightedIQROn = groupByWeightedIqrOn;

pub fn groupByWeightedMad(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeighted(key_name, value_name, weight_name, output_name, .weighted_mad);
}

pub fn groupByWeightedMadOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeightedOn(key_names, value_name, weight_name, output_name, .weighted_mad);
}

pub const groupByWeightedMAD = groupByWeightedMad;
pub const groupByWeightedMADOn = groupByWeightedMadOn;

pub fn groupByWeightedMode(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeighted(key_name, value_name, weight_name, output_name, .weighted_mode);
}

pub fn groupByWeightedModeOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeightedOn(key_names, value_name, weight_name, output_name, .weighted_mode);
}

pub fn groupByWeightedModeWeight(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeighted(key_name, value_name, weight_name, output_name, .weighted_mode_weight);
}

pub fn groupByWeightedModeWeightOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeightedOn(key_names, value_name, weight_name, output_name, .weighted_mode_weight);
}

pub fn groupByWeightedModeRatio(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeighted(key_name, value_name, weight_name, output_name, .weighted_mode_ratio);
}

pub fn groupByWeightedModeRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeightedOn(key_names, value_name, weight_name, output_name, .weighted_mode_ratio);
}

pub fn groupByWeightedModeMargin(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeighted(key_name, value_name, weight_name, output_name, .weighted_mode_margin);
}

pub fn groupByWeightedModeMarginOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeightedOn(key_names, value_name, weight_name, output_name, .weighted_mode_margin);
}

pub fn groupByWeightedModeMarginRatio(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeighted(key_name, value_name, weight_name, output_name, .weighted_mode_margin_ratio);
}

pub fn groupByWeightedModeMarginRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeightedOn(key_names, value_name, weight_name, output_name, .weighted_mode_margin_ratio);
}

pub fn groupByWeightedEntropy(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeighted(key_name, value_name, weight_name, output_name, .weighted_entropy);
}

pub fn groupByWeightedEntropyOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeightedOn(key_names, value_name, weight_name, output_name, .weighted_entropy);
}

pub fn groupByWeightedGiniImpurity(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeighted(key_name, value_name, weight_name, output_name, .weighted_gini_impurity);
}

pub fn groupByWeightedGiniImpurityOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeightedOn(key_names, value_name, weight_name, output_name, .weighted_gini_impurity);
}

pub const groupByWeightedGini = groupByWeightedGiniImpurity;
pub const groupByWeightedGiniOn = groupByWeightedGiniImpurityOn;

pub fn groupByWeightedPerplexity(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeighted(key_name, value_name, weight_name, output_name, .weighted_perplexity);
}

pub fn groupByWeightedPerplexityOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeightedOn(key_names, value_name, weight_name, output_name, .weighted_perplexity);
}

pub fn groupByWeightedInverseSimpson(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeighted(key_name, value_name, weight_name, output_name, .weighted_inverse_simpson);
}

pub fn groupByWeightedInverseSimpsonOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeightedOn(key_names, value_name, weight_name, output_name, .weighted_inverse_simpson);
}

pub fn groupByWeightedSimpsonConcentration(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeighted(key_name, value_name, weight_name, output_name, .weighted_simpson_concentration);
}

pub fn groupByWeightedSimpsonConcentrationOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeightedOn(key_names, value_name, weight_name, output_name, .weighted_simpson_concentration);
}

pub const groupByWeightedConcentration = groupByWeightedSimpsonConcentration;
pub const groupByWeightedConcentrationOn = groupByWeightedSimpsonConcentrationOn;

pub fn groupByWeightedEvenness(self: anytype, key_name: []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeighted(key_name, value_name, weight_name, output_name, .weighted_evenness);
}

pub fn groupByWeightedEvennessOn(self: anytype, key_names: []const []const u8, value_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByWeightedOn(key_names, value_name, weight_name, output_name, .weighted_evenness);
}

pub fn groupByPairCount(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByPair(self, key_name, lhs_name, rhs_name, output_name, .pair_count);
}

pub fn groupByPairCountOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByPairOn(self, key_names, lhs_name, rhs_name, output_name, .pair_count);
}

pub fn groupByCovariance(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByPair(self, key_name, lhs_name, rhs_name, output_name, .covariance);
}

pub fn groupByCovarianceOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByPairOn(self, key_names, lhs_name, rhs_name, output_name, .covariance);
}

pub const groupByCov = groupByCovariance;
pub const groupByCovOn = groupByCovarianceOn;

pub fn groupByCorrelation(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByPair(self, key_name, lhs_name, rhs_name, output_name, .correlation);
}

pub fn groupByCorrelationOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByPairOn(self, key_names, lhs_name, rhs_name, output_name, .correlation);
}

pub const groupByCorr = groupByCorrelation;
pub const groupByCorrOn = groupByCorrelationOn;

pub fn groupByBeta(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByPair(self, key_name, lhs_name, rhs_name, output_name, .beta);
}

pub fn groupByBetaOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByPairOn(self, key_names, lhs_name, rhs_name, output_name, .beta);
}

pub fn groupByDot(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByPair(self, key_name, lhs_name, rhs_name, output_name, .dot);
}

pub fn groupByDotOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByPairOn(self, key_names, lhs_name, rhs_name, output_name, .dot);
}

pub fn groupByCosineSimilarity(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByPair(self, key_name, lhs_name, rhs_name, output_name, .cosine_similarity);
}

pub fn groupByCosineSimilarityOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByPairOn(self, key_names, lhs_name, rhs_name, output_name, .cosine_similarity);
}

pub const groupByCosine = groupByCosineSimilarity;
pub const groupByCosineOn = groupByCosineSimilarityOn;

pub fn groupBySquaredEuclideanDistance(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByPair(self, key_name, lhs_name, rhs_name, output_name, .squared_euclidean_distance);
}

pub fn groupBySquaredEuclideanDistanceOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByPairOn(self, key_names, lhs_name, rhs_name, output_name, .squared_euclidean_distance);
}

pub fn groupByEuclideanDistance(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByPair(self, key_name, lhs_name, rhs_name, output_name, .euclidean_distance);
}

pub fn groupByEuclideanDistanceOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByPairOn(self, key_names, lhs_name, rhs_name, output_name, .euclidean_distance);
}

pub fn groupByManhattanDistance(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByPair(self, key_name, lhs_name, rhs_name, output_name, .manhattan_distance);
}

pub fn groupByManhattanDistanceOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByPairOn(self, key_names, lhs_name, rhs_name, output_name, .manhattan_distance);
}

pub fn groupByChebyshevDistance(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByPair(self, key_name, lhs_name, rhs_name, output_name, .chebyshev_distance);
}

pub fn groupByChebyshevDistanceOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByPairOn(self, key_names, lhs_name, rhs_name, output_name, .chebyshev_distance);
}

pub fn groupByCanberraDistance(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByPair(self, key_name, lhs_name, rhs_name, output_name, .canberra_distance);
}

pub fn groupByCanberraDistanceOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByPairOn(self, key_names, lhs_name, rhs_name, output_name, .canberra_distance);
}

pub fn groupByBrayCurtisDistance(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByPair(self, key_name, lhs_name, rhs_name, output_name, .bray_curtis_distance);
}

pub fn groupByBrayCurtisDistanceOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByPairOn(self, key_names, lhs_name, rhs_name, output_name, .bray_curtis_distance);
}

pub fn groupByMeanError(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByPair(self, key_name, lhs_name, rhs_name, output_name, .mean_error);
}

pub fn groupByMeanErrorOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByPairOn(self, key_names, lhs_name, rhs_name, output_name, .mean_error);
}

pub const groupByBias = groupByMeanError;
pub const groupByBiasOn = groupByMeanErrorOn;

pub fn groupByMae(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByPair(self, key_name, lhs_name, rhs_name, output_name, .mae);
}

pub fn groupByMaeOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByPairOn(self, key_names, lhs_name, rhs_name, output_name, .mae);
}

pub fn groupByMse(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByPair(self, key_name, lhs_name, rhs_name, output_name, .mse);
}

pub fn groupByMseOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByPairOn(self, key_names, lhs_name, rhs_name, output_name, .mse);
}

pub fn groupByRmse(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByPair(self, key_name, lhs_name, rhs_name, output_name, .rmse);
}

pub fn groupByRmseOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByPairOn(self, key_names, lhs_name, rhs_name, output_name, .rmse);
}

pub fn groupByMape(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByPair(self, key_name, lhs_name, rhs_name, output_name, .mape);
}

pub fn groupByMapeOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByPairOn(self, key_names, lhs_name, rhs_name, output_name, .mape);
}

pub fn groupBySmape(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByPair(self, key_name, lhs_name, rhs_name, output_name, .smape);
}

pub fn groupBySmapeOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByPairOn(self, key_names, lhs_name, rhs_name, output_name, .smape);
}

pub fn groupByWeightedDot(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByWeightedPair(self, key_name, lhs_name, rhs_name, weight_name, output_name, .weighted_dot, 0.0);
}

pub fn groupByWeightedDotOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByWeightedPairOn(self, key_names, lhs_name, rhs_name, weight_name, output_name, .weighted_dot, 0.0);
}

pub fn groupByWeightedCosineSimilarity(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByWeightedPair(self, key_name, lhs_name, rhs_name, weight_name, output_name, .weighted_cosine_similarity, 0.0);
}

pub fn groupByWeightedCosineSimilarityOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByWeightedPairOn(self, key_names, lhs_name, rhs_name, weight_name, output_name, .weighted_cosine_similarity, 0.0);
}

pub const groupByWeightedCosine = groupByWeightedCosineSimilarity;
pub const groupByWeightedCosineOn = groupByWeightedCosineSimilarityOn;

pub fn groupByWeightedSquaredEuclideanDistance(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByWeightedPair(self, key_name, lhs_name, rhs_name, weight_name, output_name, .weighted_squared_euclidean_distance, 0.0);
}

pub fn groupByWeightedSquaredEuclideanDistanceOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByWeightedPairOn(self, key_names, lhs_name, rhs_name, weight_name, output_name, .weighted_squared_euclidean_distance, 0.0);
}

pub fn groupByWeightedEuclideanDistance(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByWeightedPair(self, key_name, lhs_name, rhs_name, weight_name, output_name, .weighted_euclidean_distance, 0.0);
}

pub fn groupByWeightedEuclideanDistanceOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByWeightedPairOn(self, key_names, lhs_name, rhs_name, weight_name, output_name, .weighted_euclidean_distance, 0.0);
}

pub fn groupByWeightedManhattanDistance(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByWeightedPair(self, key_name, lhs_name, rhs_name, weight_name, output_name, .weighted_manhattan_distance, 0.0);
}

pub fn groupByWeightedManhattanDistanceOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByWeightedPairOn(self, key_names, lhs_name, rhs_name, weight_name, output_name, .weighted_manhattan_distance, 0.0);
}

pub fn groupByWeightedChebyshevDistance(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByWeightedPair(self, key_name, lhs_name, rhs_name, weight_name, output_name, .weighted_chebyshev_distance, 0.0);
}

pub fn groupByWeightedChebyshevDistanceOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByWeightedPairOn(self, key_names, lhs_name, rhs_name, weight_name, output_name, .weighted_chebyshev_distance, 0.0);
}

pub fn groupByWeightedCanberraDistance(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByWeightedPair(self, key_name, lhs_name, rhs_name, weight_name, output_name, .weighted_canberra_distance, 0.0);
}

pub fn groupByWeightedCanberraDistanceOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByWeightedPairOn(self, key_names, lhs_name, rhs_name, weight_name, output_name, .weighted_canberra_distance, 0.0);
}

pub fn groupByWeightedBrayCurtisDistance(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByWeightedPair(self, key_name, lhs_name, rhs_name, weight_name, output_name, .weighted_bray_curtis_distance, 0.0);
}

pub fn groupByWeightedBrayCurtisDistanceOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByWeightedPairOn(self, key_names, lhs_name, rhs_name, weight_name, output_name, .weighted_bray_curtis_distance, 0.0);
}

pub fn groupByWeightedMeanError(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByWeightedPair(self, key_name, lhs_name, rhs_name, weight_name, output_name, .weighted_mean_error, 0.0);
}

pub fn groupByWeightedMeanErrorOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByWeightedPairOn(self, key_names, lhs_name, rhs_name, weight_name, output_name, .weighted_mean_error, 0.0);
}

pub const groupByWeightedBias = groupByWeightedMeanError;
pub const groupByWeightedBiasOn = groupByWeightedMeanErrorOn;

pub fn groupByWeightedMae(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByWeightedPair(self, key_name, lhs_name, rhs_name, weight_name, output_name, .weighted_mae, 0.0);
}

pub fn groupByWeightedMaeOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByWeightedPairOn(self, key_names, lhs_name, rhs_name, weight_name, output_name, .weighted_mae, 0.0);
}

pub fn groupByWeightedMse(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByWeightedPair(self, key_name, lhs_name, rhs_name, weight_name, output_name, .weighted_mse, 0.0);
}

pub fn groupByWeightedMseOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByWeightedPairOn(self, key_names, lhs_name, rhs_name, weight_name, output_name, .weighted_mse, 0.0);
}

pub fn groupByWeightedRmse(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByWeightedPair(self, key_name, lhs_name, rhs_name, weight_name, output_name, .weighted_rmse, 0.0);
}

pub fn groupByWeightedRmseOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByWeightedPairOn(self, key_names, lhs_name, rhs_name, weight_name, output_name, .weighted_rmse, 0.0);
}

pub fn groupByWeightedMape(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByWeightedPair(self, key_name, lhs_name, rhs_name, weight_name, output_name, .weighted_mape, 0.0);
}

pub fn groupByWeightedMapeOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByWeightedPairOn(self, key_names, lhs_name, rhs_name, weight_name, output_name, .weighted_mape, 0.0);
}

pub fn groupByWeightedSmape(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByWeightedPair(self, key_name, lhs_name, rhs_name, weight_name, output_name, .weighted_smape, 0.0);
}

pub fn groupByWeightedSmapeOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByWeightedPairOn(self, key_names, lhs_name, rhs_name, weight_name, output_name, .weighted_smape, 0.0);
}

pub fn groupByWeightedCovariance(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return lazy_group_mod.groupByWeightedPair(self, key_name, lhs_name, rhs_name, weight_name, output_name, .weighted_covariance, correction);
}

pub fn groupByWeightedCovarianceOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return lazy_group_mod.groupByWeightedPairOn(self, key_names, lhs_name, rhs_name, weight_name, output_name, .weighted_covariance, correction);
}

pub const groupByWeightedCov = groupByWeightedCovariance;
pub const groupByWeightedCovOn = groupByWeightedCovarianceOn;

pub fn groupByWeightedCorrelation(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return lazy_group_mod.groupByWeightedPair(self, key_name, lhs_name, rhs_name, weight_name, output_name, .weighted_correlation, correction);
}

pub fn groupByWeightedCorrelationOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return lazy_group_mod.groupByWeightedPairOn(self, key_names, lhs_name, rhs_name, weight_name, output_name, .weighted_correlation, correction);
}

pub const groupByWeightedCorr = groupByWeightedCorrelation;
pub const groupByWeightedCorrOn = groupByWeightedCorrelationOn;

pub fn groupByWeightedBeta(self: anytype, key_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return lazy_group_mod.groupByWeightedPair(self, key_name, lhs_name, rhs_name, weight_name, output_name, .weighted_beta, correction);
}

pub fn groupByWeightedBetaOn(self: anytype, key_names: []const []const u8, lhs_name: []const u8, rhs_name: []const u8, weight_name: []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return lazy_group_mod.groupByWeightedPairOn(self, key_names, lhs_name, rhs_name, weight_name, output_name, .weighted_beta, correction);
}

pub fn groupByMeanAbsDev(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .mean_abs_dev);
}

pub fn groupByMeanAbsDevOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .mean_abs_dev);
}

pub fn groupByMeanAbsDevRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .mean_abs_dev_ratio);
}

pub fn groupByMeanAbsDevRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .mean_abs_dev_ratio);
}

pub fn groupByMedian(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .median);
}

pub fn groupByMedianOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .median);
}

pub fn groupByQuantile(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8, q: f64) DeviceDataError!void {
    return lazy_group_mod.groupByValueQuantile(self, key_name, value_name, output_name, .quantile, q);
}

pub fn groupByQuantileOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, q: f64) DeviceDataError!void {
    return lazy_group_mod.groupByValueOnQuantile(self, key_names, value_name, output_name, .quantile, q);
}

pub fn groupByIqr(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .iqr);
}

pub fn groupByIqrOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .iqr);
}

pub const groupByIQR = groupByIqr;
pub const groupByIQROn = groupByIqrOn;

pub fn groupByMad(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .mad);
}

pub fn groupByMadOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .mad);
}

pub const groupByMAD = groupByMad;
pub const groupByMADOn = groupByMadOn;
pub const groupByMedianAbsDev = groupByMad;
pub const groupByMedianAbsDevOn = groupByMadOn;

pub fn groupByTrimmedMean(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8, trim_fraction: f64) DeviceDataError!void {
    return lazy_group_mod.groupByValueQuantile(self, key_name, value_name, output_name, .trimmed_mean, trim_fraction);
}

pub fn groupByTrimmedMeanOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, trim_fraction: f64) DeviceDataError!void {
    return lazy_group_mod.groupByValueOnQuantile(self, key_names, value_name, output_name, .trimmed_mean, trim_fraction);
}

pub fn groupByWinsorizedMean(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8, winsor_fraction: f64) DeviceDataError!void {
    return lazy_group_mod.groupByValueQuantile(self, key_name, value_name, output_name, .winsorized_mean, winsor_fraction);
}

pub fn groupByWinsorizedMeanOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8, winsor_fraction: f64) DeviceDataError!void {
    return lazy_group_mod.groupByValueOnQuantile(self, key_names, value_name, output_name, .winsorized_mean, winsor_fraction);
}

pub fn groupByInterdecileRange(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .interdecile_range);
}

pub fn groupByInterdecileRangeOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .interdecile_range);
}

pub const groupByIdr = groupByInterdecileRange;
pub const groupByIdrOn = groupByInterdecileRangeOn;
pub const groupByIDR = groupByInterdecileRange;
pub const groupByIDROn = groupByInterdecileRangeOn;

pub fn groupByMidhinge(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .midhinge);
}

pub fn groupByMidhingeOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .midhinge);
}

pub fn groupByTrimean(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .trimean);
}

pub fn groupByTrimeanOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .trimean);
}

pub fn groupByBowleySkewness(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .bowley_skewness);
}

pub fn groupByBowleySkewnessOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .bowley_skewness);
}

pub const groupByBowleySkew = groupByBowleySkewness;
pub const groupByBowleySkewOn = groupByBowleySkewnessOn;

pub fn groupByQuartileCoeffDispersion(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .quartile_coeff_dispersion);
}

pub fn groupByQuartileCoeffDispersionOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .quartile_coeff_dispersion);
}

pub const groupByQcd = groupByQuartileCoeffDispersion;
pub const groupByQcdOn = groupByQuartileCoeffDispersionOn;

pub fn groupByKelleySkewness(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .kelley_skewness);
}

pub fn groupByKelleySkewnessOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .kelley_skewness);
}

pub const groupByKelleySkew = groupByKelleySkewness;
pub const groupByKelleySkewOn = groupByKelleySkewnessOn;

pub fn groupByVariance(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .variance);
}

pub fn groupByVarianceOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .variance);
}

pub fn groupByStddev(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .stddev);
}

pub fn groupByStddevOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .stddev);
}

pub const groupByStd = groupByStddev;
pub const groupByStdOn = groupByStddevOn;

pub fn groupBySem(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .sem);
}

pub fn groupBySemOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .sem);
}

pub const groupBySEM = groupBySem;
pub const groupBySEMOn = groupBySemOn;

pub fn groupByCv(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .cv);
}

pub fn groupByCvOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .cv);
}

pub const groupByCV = groupByCv;
pub const groupByCVOn = groupByCvOn;

pub fn groupByFano(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .fano);
}

pub fn groupByFanoOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .fano);
}

pub const groupByIndexOfDispersion = groupByFano;
pub const groupByIndexOfDispersionOn = groupByFanoOn;

pub fn groupBySkewness(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .skewness);
}

pub fn groupBySkewnessOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .skewness);
}

pub fn groupByKurtosis(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .kurtosis);
}

pub fn groupByKurtosisOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .kurtosis);
}

pub const groupBySkew = groupBySkewness;
pub const groupBySkewOn = groupBySkewnessOn;
pub const groupByKurt = groupByKurtosis;
pub const groupByKurtOn = groupByKurtosisOn;

pub fn groupByMagnitudeVariance(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .magnitude_variance);
}

pub fn groupByMagnitudeVarianceOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .magnitude_variance);
}

pub const groupByAbsVariance = groupByMagnitudeVariance;
pub const groupByAbsVarianceOn = groupByMagnitudeVarianceOn;
pub const groupByMagnitudeVar = groupByMagnitudeVariance;
pub const groupByMagnitudeVarOn = groupByMagnitudeVarianceOn;
pub const groupByAbsVar = groupByMagnitudeVariance;
pub const groupByAbsVarOn = groupByMagnitudeVarianceOn;

pub fn groupByMagnitudeStddev(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .magnitude_stddev);
}

pub fn groupByMagnitudeStddevOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .magnitude_stddev);
}

pub const groupByAbsStddev = groupByMagnitudeStddev;
pub const groupByAbsStddevOn = groupByMagnitudeStddevOn;
pub const groupByMagnitudeStd = groupByMagnitudeStddev;
pub const groupByMagnitudeStdOn = groupByMagnitudeStddevOn;
pub const groupByAbsStd = groupByMagnitudeStddev;
pub const groupByAbsStdOn = groupByMagnitudeStddevOn;

pub fn groupByMagnitudeSem(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .magnitude_sem);
}

pub fn groupByMagnitudeSemOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .magnitude_sem);
}

pub const groupByAbsSem = groupByMagnitudeSem;
pub const groupByAbsSemOn = groupByMagnitudeSemOn;

pub fn groupByMagnitudeCv(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .magnitude_cv);
}

pub fn groupByMagnitudeCvOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .magnitude_cv);
}

pub const groupByAbsCv = groupByMagnitudeCv;
pub const groupByAbsCvOn = groupByMagnitudeCvOn;
pub const groupByAbsCV = groupByMagnitudeCv;
pub const groupByAbsCVOn = groupByMagnitudeCvOn;

pub fn groupByMagnitudeFano(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .magnitude_fano);
}

pub fn groupByMagnitudeFanoOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .magnitude_fano);
}

pub const groupByAbsFano = groupByMagnitudeFano;
pub const groupByAbsFanoOn = groupByMagnitudeFanoOn;
pub const groupByMagnitudeIndexOfDispersion = groupByMagnitudeFano;
pub const groupByMagnitudeIndexOfDispersionOn = groupByMagnitudeFanoOn;
pub const groupByAbsIndexOfDispersion = groupByMagnitudeFano;
pub const groupByAbsIndexOfDispersionOn = groupByMagnitudeFanoOn;

pub fn groupByMagnitudeSkewness(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .magnitude_skewness);
}

pub fn groupByMagnitudeSkewnessOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .magnitude_skewness);
}

pub const groupByAbsSkewness = groupByMagnitudeSkewness;
pub const groupByAbsSkewnessOn = groupByMagnitudeSkewnessOn;
pub const groupByMagnitudeSkew = groupByMagnitudeSkewness;
pub const groupByMagnitudeSkewOn = groupByMagnitudeSkewnessOn;
pub const groupByAbsSkew = groupByMagnitudeSkewness;
pub const groupByAbsSkewOn = groupByMagnitudeSkewnessOn;

pub fn groupByMagnitudeKurtosis(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .magnitude_kurtosis);
}

pub fn groupByMagnitudeKurtosisOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .magnitude_kurtosis);
}

pub const groupByAbsKurtosis = groupByMagnitudeKurtosis;
pub const groupByAbsKurtosisOn = groupByMagnitudeKurtosisOn;
pub const groupByMagnitudeKurt = groupByMagnitudeKurtosis;
pub const groupByMagnitudeKurtOn = groupByMagnitudeKurtosisOn;
pub const groupByAbsKurt = groupByMagnitudeKurtosis;
pub const groupByAbsKurtOn = groupByMagnitudeKurtosisOn;

pub fn groupByMeanAbs(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .mean_abs);
}

pub fn groupByMeanAbsOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .mean_abs);
}

pub fn groupByMeanSquare(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .mean_square);
}

pub fn groupByMeanSquareOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .mean_square);
}

pub const groupByMeanSq = groupByMeanSquare;
pub const groupByMeanSqOn = groupByMeanSquareOn;

pub fn groupByRms(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .rms);
}

pub fn groupByRmsOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .rms);
}

pub const groupByRMS = groupByRms;
pub const groupByRMSOn = groupByRmsOn;

pub fn groupByL1Norm(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .l1_norm);
}

pub fn groupByL1NormOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .l1_norm);
}

pub fn groupByL2Norm(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .l2_norm);
}

pub fn groupByL2NormOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .l2_norm);
}

pub fn groupByMaxAbs(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .max_abs);
}

pub fn groupByMaxAbsOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .max_abs);
}

pub fn groupByMinAbs(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .min_abs);
}

pub fn groupByMinAbsOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .min_abs);
}

pub fn groupByHhi(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .hhi);
}

pub fn groupByHhiOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .hhi);
}

pub const groupByHerfindahl = groupByHhi;
pub const groupByHerfindahlOn = groupByHhiOn;
pub const groupByHerfindahlHirschman = groupByHhi;
pub const groupByHerfindahlHirschmanOn = groupByHhiOn;

pub fn groupByMagnitudeNormalizedHhi(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .magnitude_normalized_hhi);
}

pub fn groupByMagnitudeNormalizedHhiOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .magnitude_normalized_hhi);
}

pub const groupByAbsNormalizedHhi = groupByMagnitudeNormalizedHhi;
pub const groupByAbsNormalizedHhiOn = groupByMagnitudeNormalizedHhiOn;

pub fn groupByMagnitudeSparsity(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .magnitude_sparsity);
}

pub fn groupByMagnitudeSparsityOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .magnitude_sparsity);
}

pub const groupByAbsSparsity = groupByMagnitudeSparsity;
pub const groupByAbsSparsityOn = groupByMagnitudeSparsityOn;

pub fn groupByMagnitudeInverseSimpson(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .magnitude_inverse_simpson);
}

pub fn groupByMagnitudeInverseSimpsonOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .magnitude_inverse_simpson);
}

pub const groupByAbsInverseSimpson = groupByMagnitudeInverseSimpson;
pub const groupByAbsInverseSimpsonOn = groupByMagnitudeInverseSimpsonOn;

pub fn groupByMagnitudeSimpsonEvenness(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .magnitude_simpson_evenness);
}

pub fn groupByMagnitudeSimpsonEvennessOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .magnitude_simpson_evenness);
}

pub const groupByAbsSimpsonEvenness = groupByMagnitudeSimpsonEvenness;
pub const groupByAbsSimpsonEvennessOn = groupByMagnitudeSimpsonEvennessOn;

pub fn groupByMagnitudeDominance(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .magnitude_dominance);
}

pub fn groupByMagnitudeDominanceOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .magnitude_dominance);
}

pub const groupByAbsDominance = groupByMagnitudeDominance;
pub const groupByAbsDominanceOn = groupByMagnitudeDominanceOn;

pub fn groupByMagnitudeDominanceMargin(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .magnitude_dominance_margin);
}

pub fn groupByMagnitudeDominanceMarginOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .magnitude_dominance_margin);
}

pub const groupByAbsDominanceMargin = groupByMagnitudeDominanceMargin;
pub const groupByAbsDominanceMarginOn = groupByMagnitudeDominanceMarginOn;

pub fn groupByMagnitudeEntropy(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .magnitude_entropy);
}

pub fn groupByMagnitudeEntropyOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .magnitude_entropy);
}

pub const groupByAbsEntropy = groupByMagnitudeEntropy;
pub const groupByAbsEntropyOn = groupByMagnitudeEntropyOn;

pub fn groupByMagnitudePerplexity(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .magnitude_perplexity);
}

pub fn groupByMagnitudePerplexityOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .magnitude_perplexity);
}

pub const groupByAbsPerplexity = groupByMagnitudePerplexity;
pub const groupByAbsPerplexityOn = groupByMagnitudePerplexityOn;

pub fn groupByMagnitudeEvenness(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .magnitude_evenness);
}

pub fn groupByMagnitudeEvennessOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .magnitude_evenness);
}

pub const groupByAbsEvenness = groupByMagnitudeEvenness;
pub const groupByAbsEvennessOn = groupByMagnitudeEvennessOn;

pub fn groupByGeometricMean(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .geometric_mean);
}

pub fn groupByGeometricMeanOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .geometric_mean);
}

pub const groupByGeoMean = groupByGeometricMean;
pub const groupByGeoMeanOn = groupByGeometricMeanOn;

pub fn groupByHarmonicMean(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .harmonic_mean);
}

pub fn groupByHarmonicMeanOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .harmonic_mean);
}

pub fn groupByLogSumExp(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .logsumexp);
}

pub fn groupByLogSumExpOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .logsumexp);
}

pub const groupByLogsumexp = groupByLogSumExp;
pub const groupByLogsumexpOn = groupByLogSumExpOn;

pub fn groupByLogMeanExp(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .logmeanexp);
}

pub fn groupByLogMeanExpOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .logmeanexp);
}

pub const groupByLogmeanexp = groupByLogMeanExp;
pub const groupByLogmeanexpOn = groupByLogMeanExpOn;

pub fn groupByPtp(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .ptp);
}

pub fn groupByPtpOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .ptp);
}

pub const groupByPTP = groupByPtp;
pub const groupByPTPOn = groupByPtpOn;
pub const groupByPeakToPeak = groupByPtp;
pub const groupByPeakToPeakOn = groupByPtpOn;

pub fn groupByMidrange(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .midrange);
}

pub fn groupByMidrangeOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .midrange);
}

pub fn groupByRangeCoeff(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .range_coeff);
}

pub fn groupByRangeCoeffOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .range_coeff);
}

pub const groupByRangeCoefficient = groupByRangeCoeff;
pub const groupByRangeCoefficientOn = groupByRangeCoeffOn;

pub fn groupByAny(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .any);
}

pub fn groupByAnyOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .any);
}

pub fn groupByAll(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .all);
}

pub fn groupByAllOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .all);
}

pub fn groupByTrueCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .true_count);
}

pub fn groupByTrueCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .true_count);
}

pub fn groupByFalseCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .false_count);
}

pub fn groupByFalseCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .false_count);
}

pub fn groupByTrueRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .true_ratio);
}

pub fn groupByTrueRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .true_ratio);
}

pub fn groupByFalseRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .false_ratio);
}

pub fn groupByFalseRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .false_ratio);
}

pub fn groupByFirstTrueIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .first_true_index);
}

pub fn groupByFirstTrueIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .first_true_index);
}

pub fn groupByLastTrueIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .last_true_index);
}

pub fn groupByLastTrueIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .last_true_index);
}

pub fn groupByFirstFalseIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .first_false_index);
}

pub fn groupByFirstFalseIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .first_false_index);
}

pub fn groupByLastFalseIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .last_false_index);
}

pub fn groupByLastFalseIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .last_false_index);
}

pub fn groupByAnyValid(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .any_valid);
}

pub fn groupByAnyValidOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .any_valid);
}

pub fn groupByAllValid(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .all_valid);
}

pub fn groupByAllValidOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .all_valid);
}

pub fn groupByAnyNull(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .any_null);
}

pub fn groupByAnyNullOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .any_null);
}

pub fn groupByAllNull(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .all_null);
}

pub fn groupByAllNullOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .all_null);
}

pub fn groupByValidCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .valid_count);
}

pub fn groupByValidCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .valid_count);
}

pub fn groupByNullCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .null_count);
}

pub fn groupByNullCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .null_count);
}

pub fn groupByValidRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .valid_ratio);
}

pub fn groupByValidRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .valid_ratio);
}

pub fn groupByNullRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .null_ratio);
}

pub fn groupByNullRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .null_ratio);
}

pub fn groupByFirstValidIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .first_valid_index);
}

pub fn groupByFirstValidIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .first_valid_index);
}

pub fn groupByLastValidIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .last_valid_index);
}

pub fn groupByLastValidIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .last_valid_index);
}

pub fn groupByFirstNullIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .first_null_index);
}

pub fn groupByFirstNullIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .first_null_index);
}

pub fn groupByLastNullIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .last_null_index);
}

pub fn groupByLastNullIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .last_null_index);
}

pub fn groupByNaNCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .nan_count);
}

pub fn groupByNaNCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .nan_count);
}

pub fn groupByNaNRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .nan_ratio);
}

pub fn groupByNaNRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .nan_ratio);
}

pub fn groupByInfCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .inf_count);
}

pub fn groupByInfCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .inf_count);
}

pub fn groupByInfRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .inf_ratio);
}

pub fn groupByInfRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .inf_ratio);
}

pub fn groupByPositiveInfCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .positive_inf_count);
}

pub fn groupByPositiveInfCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .positive_inf_count);
}

pub fn groupByPositiveInfRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .positive_inf_ratio);
}

pub fn groupByPositiveInfRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .positive_inf_ratio);
}

pub fn groupByNegativeInfCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .negative_inf_count);
}

pub fn groupByNegativeInfCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .negative_inf_count);
}

pub fn groupByNegativeInfRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .negative_inf_ratio);
}

pub fn groupByNegativeInfRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .negative_inf_ratio);
}

pub fn groupByFirstNaNIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .first_nan_index);
}

pub fn groupByFirstNaNIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .first_nan_index);
}

pub fn groupByLastNaNIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .last_nan_index);
}

pub fn groupByLastNaNIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .last_nan_index);
}

pub fn groupByFirstInfIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .first_inf_index);
}

pub fn groupByFirstInfIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .first_inf_index);
}

pub fn groupByLastInfIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .last_inf_index);
}

pub fn groupByLastInfIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .last_inf_index);
}

pub fn groupByFirstPositiveInfIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .first_positive_inf_index);
}

pub fn groupByFirstPositiveInfIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .first_positive_inf_index);
}

pub fn groupByLastPositiveInfIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .last_positive_inf_index);
}

pub fn groupByLastPositiveInfIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .last_positive_inf_index);
}

pub fn groupByFirstNegativeInfIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .first_negative_inf_index);
}

pub fn groupByFirstNegativeInfIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .first_negative_inf_index);
}

pub fn groupByLastNegativeInfIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .last_negative_inf_index);
}

pub fn groupByLastNegativeInfIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .last_negative_inf_index);
}

pub fn groupByFiniteCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .finite_count);
}

pub fn groupByFiniteCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .finite_count);
}

pub fn groupByFiniteRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .finite_ratio);
}

pub fn groupByFiniteRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .finite_ratio);
}

pub fn groupByFirstFiniteIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .first_finite_index);
}

pub fn groupByFirstFiniteIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .first_finite_index);
}

pub fn groupByLastFiniteIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .last_finite_index);
}

pub fn groupByLastFiniteIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .last_finite_index);
}

pub fn groupByNormalCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .normal_count);
}

pub fn groupByNormalCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .normal_count);
}

pub fn groupByNormalRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .normal_ratio);
}

pub fn groupByNormalRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .normal_ratio);
}

pub fn groupByFirstNormalIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .first_normal_index);
}

pub fn groupByFirstNormalIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .first_normal_index);
}

pub fn groupByLastNormalIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .last_normal_index);
}

pub fn groupByLastNormalIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .last_normal_index);
}

pub fn groupBySubnormalCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .subnormal_count);
}

pub fn groupBySubnormalCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .subnormal_count);
}

pub fn groupBySubnormalRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .subnormal_ratio);
}

pub fn groupBySubnormalRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .subnormal_ratio);
}

pub fn groupByFirstSubnormalIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .first_subnormal_index);
}

pub fn groupByFirstSubnormalIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .first_subnormal_index);
}

pub fn groupByLastSubnormalIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .last_subnormal_index);
}

pub fn groupByLastSubnormalIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .last_subnormal_index);
}

pub fn groupByNonFiniteCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .non_finite_count);
}

pub fn groupByNonFiniteCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .non_finite_count);
}

pub fn groupByNonFiniteRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .non_finite_ratio);
}

pub fn groupByNonFiniteRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .non_finite_ratio);
}

pub fn groupByFirstNonFiniteIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .first_non_finite_index);
}

pub fn groupByFirstNonFiniteIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .first_non_finite_index);
}

pub fn groupByLastNonFiniteIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .last_non_finite_index);
}

pub fn groupByLastNonFiniteIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .last_non_finite_index);
}

pub fn groupByZeroCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .zero_count);
}

pub fn groupByZeroCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .zero_count);
}

pub fn groupByZeroRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .zero_ratio);
}

pub fn groupByZeroRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .zero_ratio);
}

pub fn groupByFirstZeroIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .first_zero_index);
}

pub fn groupByFirstZeroIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .first_zero_index);
}

pub fn groupByLastZeroIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .last_zero_index);
}

pub fn groupByLastZeroIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .last_zero_index);
}

pub fn groupByPositiveZeroCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .positive_zero_count);
}

pub fn groupByPositiveZeroCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .positive_zero_count);
}

pub fn groupByPositiveZeroRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .positive_zero_ratio);
}

pub fn groupByPositiveZeroRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .positive_zero_ratio);
}

pub fn groupByNegativeZeroCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .negative_zero_count);
}

pub fn groupByNegativeZeroCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .negative_zero_count);
}

pub fn groupByNegativeZeroRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .negative_zero_ratio);
}

pub fn groupByNegativeZeroRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .negative_zero_ratio);
}

pub fn groupByFirstPositiveZeroIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .first_positive_zero_index);
}

pub fn groupByFirstPositiveZeroIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .first_positive_zero_index);
}

pub fn groupByLastPositiveZeroIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .last_positive_zero_index);
}

pub fn groupByLastPositiveZeroIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .last_positive_zero_index);
}

pub fn groupByFirstNegativeZeroIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .first_negative_zero_index);
}

pub fn groupByFirstNegativeZeroIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .first_negative_zero_index);
}

pub fn groupByLastNegativeZeroIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .last_negative_zero_index);
}

pub fn groupByLastNegativeZeroIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .last_negative_zero_index);
}

pub fn groupByNonZeroCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .non_zero_count);
}

pub fn groupByNonZeroCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .non_zero_count);
}

pub fn groupByNonZeroRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .non_zero_ratio);
}

pub fn groupByNonZeroRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .non_zero_ratio);
}

pub fn groupByFirstNonZeroIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .first_non_zero_index);
}

pub fn groupByFirstNonZeroIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .first_non_zero_index);
}

pub fn groupByLastNonZeroIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .last_non_zero_index);
}

pub fn groupByLastNonZeroIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .last_non_zero_index);
}

pub fn groupByPositiveCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .positive_count);
}

pub fn groupByPositiveCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .positive_count);
}

pub fn groupByPositiveRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .positive_ratio);
}

pub fn groupByPositiveRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .positive_ratio);
}

pub fn groupByFirstPositiveIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .first_positive_index);
}

pub fn groupByFirstPositiveIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .first_positive_index);
}

pub fn groupByLastPositiveIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .last_positive_index);
}

pub fn groupByLastPositiveIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .last_positive_index);
}

pub fn groupBySignBitCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .signbit_count);
}

pub fn groupBySignBitCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .signbit_count);
}

pub fn groupBySignBitRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .signbit_ratio);
}

pub fn groupBySignBitRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .signbit_ratio);
}

pub fn groupByFirstSignBitIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .first_signbit_index);
}

pub fn groupByFirstSignBitIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .first_signbit_index);
}

pub fn groupByLastSignBitIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .last_signbit_index);
}

pub fn groupByLastSignBitIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .last_signbit_index);
}

pub fn groupByNegativeCount(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .negative_count);
}

pub fn groupByNegativeCountOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .negative_count);
}

pub fn groupByNegativeRatio(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .negative_ratio);
}

pub fn groupByNegativeRatioOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .negative_ratio);
}

pub fn groupByFirstNegativeIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .first_negative_index);
}

pub fn groupByFirstNegativeIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .first_negative_index);
}

pub fn groupByLastNegativeIndex(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .last_negative_index);
}

pub fn groupByLastNegativeIndexOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .last_negative_index);
}

pub fn groupByArgMin(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .argmin);
}

pub fn groupByArgMinOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .argmin);
}

pub fn groupByArgMax(self: anytype, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValue(key_name, value_name, output_name, .argmax);
}

pub fn groupByArgMaxOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
    return self.groupByValueOn(key_names, value_name, output_name, .argmax);
}

pub const groupByArgmin = groupByArgMin;
pub const groupByArgminOn = groupByArgMinOn;
pub const groupByArgmax = groupByArgMax;
pub const groupByArgmaxOn = groupByArgMaxOn;

pub fn groupByStats(self: anytype, key_name: []const u8, value_name: []const u8, output_prefix: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByStats(self, key_name, value_name, output_prefix);
}

pub fn groupByStatsOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_prefix: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByStatsOn(self, key_names, value_name, output_prefix);
}

pub fn groupByProfile(self: anytype, key_name: []const u8, value_name: []const u8, output_prefix: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByProfile(self, key_name, value_name, output_prefix);
}

pub fn groupByProfileOn(self: anytype, key_names: []const []const u8, value_name: []const u8, output_prefix: []const u8) DeviceDataError!void {
    return lazy_group_mod.groupByProfileOn(self, key_names, value_name, output_prefix);
}

pub fn joinOn(
    self: anytype,
    right: anytype,
    left_key_names: []const []const u8,
    right_key_names: []const []const u8,
    kind: DeviceLazyJoinKind,
    options_value: DeviceJoinOptions,
) DeviceDataError!void {
    return lazy_join_mod.joinOn(self, right, left_key_names, right_key_names, kind, options_value);
}

pub fn innerJoinOn(self: anytype, right: anytype, left_key_names: []const []const u8, right_key_names: []const []const u8, options_value: DeviceJoinOptions) DeviceDataError!void {
    return self.joinOn(right, left_key_names, right_key_names, .inner, options_value);
}

pub fn leftJoinOn(self: anytype, right: anytype, left_key_names: []const []const u8, right_key_names: []const []const u8, options_value: DeviceJoinOptions) DeviceDataError!void {
    return self.joinOn(right, left_key_names, right_key_names, .left, options_value);
}

pub fn fullJoinOn(self: anytype, right: anytype, left_key_names: []const []const u8, right_key_names: []const []const u8, options_value: DeviceJoinOptions) DeviceDataError!void {
    return self.joinOn(right, left_key_names, right_key_names, .full, options_value);
}

pub fn semiJoinOn(self: anytype, right: anytype, left_key_names: []const []const u8, right_key_names: []const []const u8) DeviceDataError!void {
    return self.joinOn(right, left_key_names, right_key_names, .semi, .{});
}

pub fn antiJoinOn(self: anytype, right: anytype, left_key_names: []const []const u8, right_key_names: []const []const u8) DeviceDataError!void {
    return self.joinOn(right, left_key_names, right_key_names, .anti, .{});
}

pub fn asofJoin(
    self: anytype,
    right: anytype,
    left_key_name: []const u8,
    right_key_name: []const u8,
    options_value: DeviceAsofOptions,
) DeviceDataError!void {
    return lazy_join_mod.asofJoin(self, right, left_key_name, right_key_name, options_value);
}

pub fn concatRows(self: anytype, right: anytype) DeviceDataError!void {
    return lazy_join_mod.concatRows(self, right);
}

pub fn appendRows(self: anytype, right: anytype) DeviceDataError!void {
    return self.concatRows(right);
}

pub fn vstack(self: anytype, right: anytype) DeviceDataError!void {
    return self.concatRows(right);
}

pub fn concatColumns(self: anytype, right: anytype) DeviceDataError!void {
    return lazy_join_mod.concatColumns(self, right);
}

pub fn appendColumns(self: anytype, right: anytype) DeviceDataError!void {
    return self.concatColumns(right);
}

pub fn hstack(self: anytype, right: anytype) DeviceDataError!void {
    return self.concatColumns(right);
}

pub fn distinctRows(self: anytype) DeviceDataError!void {
    try self.ops.append(self.allocator, .{ .distinct_rows = {} });
}

pub fn distinctRowsLast(self: anytype) DeviceDataError!void {
    try self.ops.append(self.allocator, .{ .distinct_rows_last = {} });
}

pub fn distinctRowsNone(self: anytype) DeviceDataError!void {
    try self.ops.append(self.allocator, .{ .distinct_rows_none = {} });
}

pub fn distinctOn(self: anytype, key_names: []const []const u8) DeviceDataError!void {
    return lazy_join_mod.distinctOn(self, key_names);
}

pub fn distinctOnLast(self: anytype, key_names: []const []const u8) DeviceDataError!void {
    return lazy_join_mod.distinctOnLast(self, key_names);
}

pub fn distinctOnNone(self: anytype, key_names: []const []const u8) DeviceDataError!void {
    return lazy_join_mod.distinctOnNone(self, key_names);
}

pub fn dropDuplicates(self: anytype) DeviceDataError!void {
    return self.distinctRows();
}

pub fn dropDuplicatesOn(self: anytype, key_names: []const []const u8) DeviceDataError!void {
    return self.distinctOn(key_names);
}

pub fn dropDuplicatesLast(self: anytype) DeviceDataError!void {
    return self.distinctRowsLast();
}

pub fn dropDuplicatesOnLast(self: anytype, key_names: []const []const u8) DeviceDataError!void {
    return self.distinctOnLast(key_names);
}

pub fn dropDuplicatesNone(self: anytype) DeviceDataError!void {
    return self.distinctRowsNone();
}

pub fn dropDuplicatesOnNone(self: anytype, key_names: []const []const u8) DeviceDataError!void {
    return self.distinctOnNone(key_names);
}

pub fn uniqueRows(self: anytype) DeviceDataError!void {
    return self.distinctRows();
}
