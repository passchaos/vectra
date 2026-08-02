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
