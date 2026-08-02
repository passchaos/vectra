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
