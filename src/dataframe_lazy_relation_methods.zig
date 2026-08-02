//! Lazy group-by, join, concat, and distinct method wrappers.

const std = @import("std");
const array_mod = @import("array.zig");
const lazy_group_mod = @import("dataframe_lazy_group_plan.zig");
const lazy_join_mod = @import("dataframe_lazy_join_plan.zig");
const lazy_op_mod = @import("dataframe_lazy_op.zig");
const options_mod = @import("dataframe_options.zig");
const series_mod = @import("series.zig");

const DeviceLazyGroupByAggregation = lazy_op_mod.DeviceLazyGroupByAggregation;
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
