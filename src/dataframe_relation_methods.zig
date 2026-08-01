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
