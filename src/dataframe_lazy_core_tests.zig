const std = @import("std");
const boltha = @import("boltha");
const vectra = @import("vectra");

const DataFrame = vectra.DataFrame;
const DeviceColumn = vectra.DeviceColumn;
const DeviceDataFrame = vectra.DeviceDataFrame;
const DeviceLazyFrame = vectra.DeviceLazyFrame;
const DeviceDType = vectra.DeviceDType;
const DeviceValidityEncoding = vectra.DeviceValidityEncoding;
const DeviceParquetScan = vectra.DeviceParquetScan;

fn lazyCollectTable(gpa: std.mem.Allocator) !DeviceDataFrame {
    var sales = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 3.0, 5.0, 7.0 }, .cpu);
    defer sales.deinit();
    var units = try DeviceColumn.fromSlice(i64, gpa, &.{ 1, 2, 3, 4 }, .cpu);
    defer units.deinit();
    var active = try DeviceColumn.fromSlice(bool, gpa, &.{ true, false, true, true }, .cpu);
    defer active.deinit();

    return DeviceDataFrame.init(gpa, &.{
        .{ .name = "sales", .data = sales },
        .{ .name = "units", .data = units },
        .{ .name = "active", .data = active },
    });
}

test "device lazy frame collects plan operations" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();
    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.withColumnScalar("sales_x2", "sales", f64, 2.0, .mul);
    try plan.withColumnCompareScalar("big_sale", "sales_x2", f64, 10.0, .gt);
    try plan.filterColumnScalar("sales", f64, 2.5, .gt);
    try plan.sortBy("sales", .{ .descending = true });
    try plan.select(&.{ "sales", "units", "sales_x2", "big_sale", "active" });
    try plan.select(&.{ "sales", "units", "sales_x2", "big_sale" });
    try plan.head(3);
    try plan.head(2);

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "raw_ops=8") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "optimized_ops=6") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_scalar(sales_x2") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_compare_scalar(big_sale") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "filter_scalar(sales") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 2), result.height());
    try std.testing.expectEqual(@as(usize, 4), result.width());
    const result_sales = try (try result.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales);
    const result_units = try (try result.column("units")).i64.toOwnedSlice(gpa);
    defer gpa.free(result_units);
    const result_sales_x2 = try (try result.column("sales_x2")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_x2);
    const result_big_sale = try (try result.column("big_sale")).bool.toOwnedSlice(gpa);
    defer gpa.free(result_big_sale);
    try std.testing.expectEqualSlices(f64, &.{ 7.0, 5.0 }, result_sales);
    try std.testing.expectEqualSlices(i64, &.{ 4, 3 }, result_units);
    try std.testing.expectEqualSlices(f64, &.{ 14.0, 10.0 }, result_sales_x2);
    try std.testing.expectEqualSlices(bool, &.{ true, false }, result_big_sale);
}

test "device lazy frame collects topk operations" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();
    var topk_plan = try DeviceLazyFrame.init(gpa, table);
    defer topk_plan.deinit();
    try topk_plan.sortBy("sales", .{ .descending = true });
    try topk_plan.head(2);
    const topk_explain = try topk_plan.explain(gpa);
    defer gpa.free(topk_explain);
    try std.testing.expect(std.mem.indexOf(u8, topk_explain, "top_k(sales, k=2") != null);
    var topk = try topk_plan.collect();
    defer topk.deinit();
    const topk_sales = try (try topk.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(topk_sales);
    try std.testing.expectEqualSlices(f64, &.{ 7.0, 5.0 }, topk_sales);
}

test "device lazy frame collects rank operations" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();
    var rank_plan = try DeviceLazyFrame.init(gpa, table);
    defer rank_plan.deinit();
    try rank_plan.rankProfileBy("sales", "sales_rank", .{ .descending = true });
    try rank_plan.select(&.{ "sales", "sales_rank_ordinal_rank", "sales_rank_percent_rank", "sales_rank_cume_dist" });
    const rank_explain = try rank_plan.explain(gpa);
    defer gpa.free(rank_explain);
    try std.testing.expect(std.mem.indexOf(u8, rank_explain, "rank_profile_by(sales") != null);
    var ranked = try rank_plan.collect();
    defer ranked.deinit();
    try std.testing.expectEqual(@as(usize, 4), ranked.height());
    try std.testing.expectEqual(@as(usize, 4), ranked.width());
    const ranked_ordinal = try (try ranked.column("sales_rank_ordinal_rank")).i64.toOwnedSlice(gpa);
    defer gpa.free(ranked_ordinal);
    const ranked_percent = try (try ranked.column("sales_rank_percent_rank")).f64.toOwnedSlice(gpa);
    defer gpa.free(ranked_percent);
    const ranked_cume = try (try ranked.column("sales_rank_cume_dist")).f64.toOwnedSlice(gpa);
    defer gpa.free(ranked_cume);
    try std.testing.expectEqualSlices(i64, &.{ 4, 3, 2, 1 }, ranked_ordinal);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), ranked_percent[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), ranked_percent[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), ranked_percent[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ranked_percent[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), ranked_cume[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), ranked_cume[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), ranked_cume[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), ranked_cume[3], 1e-12);
}

test "device lazy frame collects rolling rank operations" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();
    var rolling_rank_plan = try DeviceLazyFrame.init(gpa, table);
    defer rolling_rank_plan.deinit();
    try rolling_rank_plan.rollingRankProfile("sales", "sales_roll", .{ .window = 2, .min_periods = 2, .descending = true });
    try rolling_rank_plan.select(&.{ "sales", "sales_roll_rolling_rank_count", "sales_roll_rolling_rank", "sales_roll_rolling_percent_rank", "sales_roll_rolling_cume_dist" });
    const rolling_rank_explain = try rolling_rank_plan.explain(gpa);
    defer gpa.free(rolling_rank_explain);
    try std.testing.expect(std.mem.indexOf(u8, rolling_rank_explain, "rolling_rank_profile(sales") != null);
    var rolling_ranked = try rolling_rank_plan.collect();
    defer rolling_ranked.deinit();
    try std.testing.expectEqual(@as(usize, 4), rolling_ranked.height());
    try std.testing.expectEqual(@as(usize, 5), rolling_ranked.width());
    const lazy_rolling_rank_count = try (try rolling_ranked.column("sales_roll_rolling_rank_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_rank_count);
    const lazy_rolling_rank = try (try rolling_ranked.column("sales_roll_rolling_rank")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_rank);
    const lazy_rolling_percent_rank = try (try rolling_ranked.column("sales_roll_rolling_percent_rank")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_percent_rank);
    const lazy_rolling_cume_dist = try (try rolling_ranked.column("sales_roll_rolling_cume_dist")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_cume_dist);
    const lazy_rolling_rank_validity = try (try rolling_ranked.column("sales_roll_rolling_rank")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_rank_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 2, 2 }, lazy_rolling_rank_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_rolling_rank_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 1, 1 }, lazy_rolling_rank);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_rolling_percent_rank[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_rolling_percent_rank[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_rolling_percent_rank[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_rolling_cume_dist[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_rolling_cume_dist[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_rolling_cume_dist[3], 1e-12);
}

test "device lazy frame collects expanding rank operations" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();
    var expanding_rank_plan = try DeviceLazyFrame.init(gpa, table);
    defer expanding_rank_plan.deinit();
    try expanding_rank_plan.expandingRankProfile("sales", "sales_expand", .{ .min_periods = 2, .descending = true });
    try expanding_rank_plan.select(&.{ "sales", "sales_expand_expanding_rank_count", "sales_expand_expanding_rank", "sales_expand_expanding_percent_rank", "sales_expand_expanding_cume_dist" });
    const expanding_rank_explain = try expanding_rank_plan.explain(gpa);
    defer gpa.free(expanding_rank_explain);
    try std.testing.expect(std.mem.indexOf(u8, expanding_rank_explain, "expanding_rank_profile(sales") != null);
    var expanding_ranked = try expanding_rank_plan.collect();
    defer expanding_ranked.deinit();
    try std.testing.expectEqual(@as(usize, 4), expanding_ranked.height());
    try std.testing.expectEqual(@as(usize, 5), expanding_ranked.width());
    const lazy_expanding_rank_count = try (try expanding_ranked.column("sales_expand_expanding_rank_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_rank_count);
    const lazy_expanding_rank = try (try expanding_ranked.column("sales_expand_expanding_rank")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_rank);
    const lazy_expanding_percent_rank = try (try expanding_ranked.column("sales_expand_expanding_percent_rank")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_percent_rank);
    const lazy_expanding_cume_dist = try (try expanding_ranked.column("sales_expand_expanding_cume_dist")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_cume_dist);
    const lazy_expanding_rank_validity = try (try expanding_ranked.column("sales_expand_expanding_rank")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_rank_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4 }, lazy_expanding_rank_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_expanding_rank_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 1, 1 }, lazy_expanding_rank);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_expanding_percent_rank[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_expanding_percent_rank[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_expanding_percent_rank[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_expanding_cume_dist[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), lazy_expanding_cume_dist[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), lazy_expanding_cume_dist[3], 1e-12);
}

test "device lazy frame collects rolling robust operations" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();
    var rolling_robust_plan = try DeviceLazyFrame.init(gpa, table);
    defer rolling_robust_plan.deinit();
    try rolling_robust_plan.rollingRobustProfile("sales", "sales", .{ .window = 2, .min_periods = 2 });
    try rolling_robust_plan.select(&.{ "sales", "sales_rolling_median_centered", "sales_rolling_mad_zscore", "sales_rolling_iqr_outlier", "sales_rolling_winsorized" });
    const rolling_robust_explain = try rolling_robust_plan.explain(gpa);
    defer gpa.free(rolling_robust_explain);
    try std.testing.expect(std.mem.indexOf(u8, rolling_robust_explain, "rolling_robust_profile(sales") != null);
    var lazy_rolling_robust = try rolling_robust_plan.collect();
    defer lazy_rolling_robust.deinit();
    try std.testing.expectEqual(@as(usize, 4), lazy_rolling_robust.height());
    try std.testing.expectEqual(@as(usize, 5), lazy_rolling_robust.width());
    const lazy_rolling_median_centered = try (try lazy_rolling_robust.column("sales_rolling_median_centered")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_median_centered);
    const lazy_rolling_mad_zscore = try (try lazy_rolling_robust.column("sales_rolling_mad_zscore")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_mad_zscore);
    const lazy_rolling_iqr_outlier = try (try lazy_rolling_robust.column("sales_rolling_iqr_outlier")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_iqr_outlier);
    const lazy_rolling_winsorized = try (try lazy_rolling_robust.column("sales_rolling_winsorized")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_winsorized);
    const lazy_rolling_robust_validity = try (try lazy_rolling_robust.column("sales_rolling_median_centered")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_robust_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_rolling_robust_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_rolling_median_centered[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_median_centered[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_median_centered[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.6744897501960817), lazy_rolling_mad_zscore[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.6744897501960817), lazy_rolling_mad_zscore[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.6744897501960817), lazy_rolling_mad_zscore[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false }, lazy_rolling_iqr_outlier);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), lazy_rolling_winsorized[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), lazy_rolling_winsorized[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 7.0), lazy_rolling_winsorized[3], 1e-12);
}

test "device lazy frame collects rolling operations" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();
    var rolling_plan = try DeviceLazyFrame.init(gpa, table);
    defer rolling_plan.deinit();
    try rolling_plan.rollingProfile("sales", "sales", .{ .window = 2, .min_periods = 1 });
    try rolling_plan.select(&.{ "sales", "sales_rolling_mean", "sales_rolling_stddev" });
    const rolling_explain = try rolling_plan.explain(gpa);
    defer gpa.free(rolling_explain);
    try std.testing.expect(std.mem.indexOf(u8, rolling_explain, "rolling_profile(sales") != null);
    var rolling = try rolling_plan.collect();
    defer rolling.deinit();
    try std.testing.expectEqual(@as(usize, 4), rolling.height());
    try std.testing.expectEqual(@as(usize, 3), rolling.width());
    const rolling_mean = try (try rolling.column("sales_rolling_mean")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_mean);
    const rolling_stddev = try (try rolling.column("sales_rolling_stddev")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_stddev);
    const rolling_validity = try (try rolling.column("sales_rolling_mean")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(rolling_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, rolling_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), rolling_mean[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), rolling_mean[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), rolling_mean[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 6.0), rolling_mean[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), rolling_stddev[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_stddev[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_stddev[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_stddev[3], 1e-12);
}

test "device lazy frame collects rolling moment operations" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();
    var rolling_moment_plan = try DeviceLazyFrame.init(gpa, table);
    defer rolling_moment_plan.deinit();
    try rolling_moment_plan.rollingMomentProfile("sales", "sales", .{ .window = 2, .min_periods = 2 });
    try rolling_moment_plan.select(&.{ "sales", "sales_rolling_moment_count", "sales_rolling_m3", "sales_rolling_m4", "sales_rolling_skewness", "sales_rolling_kurtosis" });
    const rolling_moment_explain = try rolling_moment_plan.explain(gpa);
    defer gpa.free(rolling_moment_explain);
    try std.testing.expect(std.mem.indexOf(u8, rolling_moment_explain, "rolling_moment_profile(sales") != null);
    var rolling_moments = try rolling_moment_plan.collect();
    defer rolling_moments.deinit();
    try std.testing.expectEqual(@as(usize, 4), rolling_moments.height());
    try std.testing.expectEqual(@as(usize, 6), rolling_moments.width());
    const lazy_moment_count = try (try rolling_moments.column("sales_rolling_moment_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_moment_count);
    const lazy_m3 = try (try rolling_moments.column("sales_rolling_m3")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_m3);
    const lazy_m4 = try (try rolling_moments.column("sales_rolling_m4")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_m4);
    const lazy_skewness = try (try rolling_moments.column("sales_rolling_skewness")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_skewness);
    const lazy_kurtosis = try (try rolling_moments.column("sales_rolling_kurtosis")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_kurtosis);
    const lazy_moment_validity = try (try rolling_moments.column("sales_rolling_skewness")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_moment_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 2, 2 }, lazy_moment_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_moment_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_m3[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_m3[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_m3[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0625), lazy_m4[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_m4[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_m4[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_skewness[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_skewness[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_skewness[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -2.0), lazy_kurtosis[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -2.0), lazy_kurtosis[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -2.0), lazy_kurtosis[3], 1e-12);
}

test "device lazy frame collects ema operations" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();
    var ema_plan = try DeviceLazyFrame.init(gpa, table);
    defer ema_plan.deinit();
    try ema_plan.emaProfile("sales", "sales", .{ .alpha = 0.5, .min_periods = 1 });
    try ema_plan.select(&.{ "sales", "sales_ema", "sales_ema_residual", "sales_ema_ratio" });
    const ema_explain = try ema_plan.explain(gpa);
    defer gpa.free(ema_explain);
    try std.testing.expect(std.mem.indexOf(u8, ema_explain, "ema_profile(sales") != null);
    var ema = try ema_plan.collect();
    defer ema.deinit();
    try std.testing.expectEqual(@as(usize, 4), ema.height());
    try std.testing.expectEqual(@as(usize, 4), ema.width());
    const lazy_ema = try (try ema.column("sales_ema")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ema);
    const lazy_ema_residual = try (try ema.column("sales_ema_residual")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ema_residual);
    const lazy_ema_ratio = try (try ema.column("sales_ema_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ema_ratio);
    const lazy_ema_validity = try (try ema.column("sales_ema")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_ema_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, lazy_ema_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_ema[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), lazy_ema[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.75), lazy_ema[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.375), lazy_ema[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_ema_residual[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_ema_residual[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.25), lazy_ema_residual[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.625), lazy_ema_residual[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_ema_ratio[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.2), lazy_ema_ratio[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 3.75), lazy_ema_ratio[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 7.0 / 5.375), lazy_ema_ratio[3], 1e-12);
}

test "device lazy frame collects rolling range operations" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();
    var rolling_range_plan = try DeviceLazyFrame.init(gpa, table);
    defer rolling_range_plan.deinit();
    try rolling_range_plan.rollingRangeProfile("sales", "sales", .{ .window = 2, .min_periods = 1 });
    try rolling_range_plan.select(&.{ "sales", "sales_rolling_low", "sales_rolling_high", "sales_rolling_position" });
    const rolling_range_explain = try rolling_range_plan.explain(gpa);
    defer gpa.free(rolling_range_explain);
    try std.testing.expect(std.mem.indexOf(u8, rolling_range_explain, "rolling_range_profile(sales") != null);
    var rolling_range = try rolling_range_plan.collect();
    defer rolling_range.deinit();
    try std.testing.expectEqual(@as(usize, 4), rolling_range.height());
    try std.testing.expectEqual(@as(usize, 4), rolling_range.width());
    const lazy_rolling_low = try (try rolling_range.column("sales_rolling_low")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_low);
    const lazy_rolling_high = try (try rolling_range.column("sales_rolling_high")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_high);
    const lazy_rolling_position = try (try rolling_range.column("sales_rolling_position")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_position);
    const lazy_rolling_range_validity = try (try rolling_range.column("sales_rolling_position")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_range_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, lazy_rolling_range_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_rolling_low[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_rolling_low[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), lazy_rolling_low[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), lazy_rolling_low[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_rolling_high[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), lazy_rolling_high[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), lazy_rolling_high[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 7.0), lazy_rolling_high[3], 1e-12);
    try std.testing.expect(std.math.isNan(lazy_rolling_position[0]));
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_position[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_position[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_position[3], 1e-12);
}

test "device lazy frame collects rolling norm operations" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();
    var rolling_norm_plan = try DeviceLazyFrame.init(gpa, table);
    defer rolling_norm_plan.deinit();
    try rolling_norm_plan.rollingNormalizeProfile("sales", "sales", .{ .window = 2, .min_periods = 1 });
    try rolling_norm_plan.select(&.{ "sales", "sales_rolling_centered", "sales_rolling_zscore", "sales_rolling_minmax" });
    const rolling_norm_explain = try rolling_norm_plan.explain(gpa);
    defer gpa.free(rolling_norm_explain);
    try std.testing.expect(std.mem.indexOf(u8, rolling_norm_explain, "rolling_normalize_profile(sales") != null);
    var rolling_norm = try rolling_norm_plan.collect();
    defer rolling_norm.deinit();
    try std.testing.expectEqual(@as(usize, 4), rolling_norm.height());
    try std.testing.expectEqual(@as(usize, 4), rolling_norm.width());
    const lazy_rolling_centered = try (try rolling_norm.column("sales_rolling_centered")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_centered);
    const lazy_rolling_zscore = try (try rolling_norm.column("sales_rolling_zscore")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_zscore);
    const lazy_rolling_minmax = try (try rolling_norm.column("sales_rolling_minmax")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_minmax);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_rolling_centered[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_rolling_centered[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_centered[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_centered[3], 1e-12);
    try std.testing.expect(std.math.isNan(lazy_rolling_zscore[0]));
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_zscore[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_zscore[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_zscore[3], 1e-12);
    try std.testing.expect(std.math.isNan(lazy_rolling_minmax[0]));
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_minmax[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_minmax[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_minmax[3], 1e-12);
}

test "device lazy frame collects rolling quantile operations" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();
    var rolling_quantile_plan = try DeviceLazyFrame.init(gpa, table);
    defer rolling_quantile_plan.deinit();
    try rolling_quantile_plan.rollingQuantileProfile("sales", "sales", .{ .window = 2, .min_periods = 1 });
    try rolling_quantile_plan.select(&.{ "sales", "sales_rolling_q1", "sales_rolling_median", "sales_rolling_q3", "sales_rolling_iqr" });
    const rolling_quantile_explain = try rolling_quantile_plan.explain(gpa);
    defer gpa.free(rolling_quantile_explain);
    try std.testing.expect(std.mem.indexOf(u8, rolling_quantile_explain, "rolling_quantile_profile(sales") != null);
    var rolling_quantile = try rolling_quantile_plan.collect();
    defer rolling_quantile.deinit();
    try std.testing.expectEqual(@as(usize, 4), rolling_quantile.height());
    try std.testing.expectEqual(@as(usize, 5), rolling_quantile.width());
    const lazy_rolling_q1 = try (try rolling_quantile.column("sales_rolling_q1")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_q1);
    const lazy_rolling_median = try (try rolling_quantile.column("sales_rolling_median")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_median);
    const lazy_rolling_q3 = try (try rolling_quantile.column("sales_rolling_q3")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_q3);
    const lazy_rolling_iqr = try (try rolling_quantile.column("sales_rolling_iqr")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_iqr);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_rolling_q1[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.25), lazy_rolling_q1[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.5), lazy_rolling_q1[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.5), lazy_rolling_q1[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_rolling_median[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), lazy_rolling_median[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), lazy_rolling_median[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 6.0), lazy_rolling_median[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_rolling_q3[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.75), lazy_rolling_q3[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.5), lazy_rolling_q3[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 6.5), lazy_rolling_q3[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_rolling_iqr[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_rolling_iqr[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_iqr[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_iqr[3], 1e-12);
}

test "device lazy frame collects expanding quantile operations" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();
    var expanding_quantile_plan = try DeviceLazyFrame.init(gpa, table);
    defer expanding_quantile_plan.deinit();
    try expanding_quantile_plan.expandingQuantileProfile("sales", "sales", .{ .min_periods = 2 });
    try expanding_quantile_plan.select(&.{ "sales", "sales_expanding_q1", "sales_expanding_median", "sales_expanding_q3", "sales_expanding_iqr" });
    const expanding_quantile_explain = try expanding_quantile_plan.explain(gpa);
    defer gpa.free(expanding_quantile_explain);
    try std.testing.expect(std.mem.indexOf(u8, expanding_quantile_explain, "expanding_quantile_profile(sales") != null);
    var expanding_quantile = try expanding_quantile_plan.collect();
    defer expanding_quantile.deinit();
    try std.testing.expectEqual(@as(usize, 4), expanding_quantile.height());
    try std.testing.expectEqual(@as(usize, 5), expanding_quantile.width());
    const lazy_expanding_q1 = try (try expanding_quantile.column("sales_expanding_q1")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_q1);
    const lazy_expanding_median = try (try expanding_quantile.column("sales_expanding_median")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_median);
    const lazy_expanding_q3 = try (try expanding_quantile.column("sales_expanding_q3")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_q3);
    const lazy_expanding_iqr = try (try expanding_quantile.column("sales_expanding_iqr")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_iqr);
    const lazy_expanding_quantile_validity = try (try expanding_quantile.column("sales_expanding_median")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_quantile_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_expanding_quantile_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 2.25), lazy_expanding_q1[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), lazy_expanding_q1[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.75), lazy_expanding_q1[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), lazy_expanding_median[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), lazy_expanding_median[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), lazy_expanding_median[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.75), lazy_expanding_q3[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), lazy_expanding_q3[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.5), lazy_expanding_q3[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_expanding_iqr[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), lazy_expanding_iqr[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.75), lazy_expanding_iqr[3], 1e-12);
}

test "device lazy frame collects rolling bool operations" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();
    var rolling_bool_plan = try DeviceLazyFrame.init(gpa, table);
    defer rolling_bool_plan.deinit();
    try rolling_bool_plan.rollingBoolProfile("active", "active", .{ .window = 2, .min_periods = 1 });
    try rolling_bool_plan.select(&.{ "active", "active_rolling_true_count", "active_rolling_false_count", "active_rolling_true_rate", "active_rolling_any", "active_rolling_all" });
    const rolling_bool_explain = try rolling_bool_plan.explain(gpa);
    defer gpa.free(rolling_bool_explain);
    try std.testing.expect(std.mem.indexOf(u8, rolling_bool_explain, "rolling_bool_profile(active") != null);
    var rolling_bool = try rolling_bool_plan.collect();
    defer rolling_bool.deinit();
    try std.testing.expectEqual(@as(usize, 4), rolling_bool.height());
    try std.testing.expectEqual(@as(usize, 6), rolling_bool.width());
    const lazy_true_count = try (try rolling_bool.column("active_rolling_true_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_true_count);
    const lazy_false_count = try (try rolling_bool.column("active_rolling_false_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_false_count);
    const lazy_true_rate = try (try rolling_bool.column("active_rolling_true_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_true_rate);
    const lazy_rolling_any = try (try rolling_bool.column("active_rolling_any")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_any);
    const lazy_rolling_all = try (try rolling_bool.column("active_rolling_all")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_all);
    const lazy_bool_validity = try (try rolling_bool.column("active_rolling_true_rate")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_bool_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 1, 1, 2 }, lazy_true_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 1, 0 }, lazy_false_count);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, lazy_bool_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_true_rate[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_true_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_true_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_true_rate[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, lazy_rolling_any);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, lazy_rolling_all);
}

test "device lazy frame collects expanding bool operations" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();
    var expanding_bool_plan = try DeviceLazyFrame.init(gpa, table);
    defer expanding_bool_plan.deinit();
    try expanding_bool_plan.expandingBoolProfile("active", "active", .{ .min_periods = 2 });
    try expanding_bool_plan.select(&.{ "active", "active_expanding_true_count", "active_expanding_false_count", "active_expanding_true_rate", "active_expanding_any", "active_expanding_all" });
    const expanding_bool_explain = try expanding_bool_plan.explain(gpa);
    defer gpa.free(expanding_bool_explain);
    try std.testing.expect(std.mem.indexOf(u8, expanding_bool_explain, "expanding_bool_profile(active") != null);
    var lazy_expanding_bool = try expanding_bool_plan.collect();
    defer lazy_expanding_bool.deinit();
    try std.testing.expectEqual(@as(usize, 4), lazy_expanding_bool.height());
    try std.testing.expectEqual(@as(usize, 6), lazy_expanding_bool.width());
    const lazy_expanding_true_count = try (try lazy_expanding_bool.column("active_expanding_true_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_true_count);
    const lazy_expanding_false_count = try (try lazy_expanding_bool.column("active_expanding_false_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_false_count);
    const lazy_expanding_true_rate = try (try lazy_expanding_bool.column("active_expanding_true_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_true_rate);
    const lazy_expanding_any = try (try lazy_expanding_bool.column("active_expanding_any")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_any);
    const lazy_expanding_all = try (try lazy_expanding_bool.column("active_expanding_all")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_all);
    const lazy_expanding_bool_validity = try (try lazy_expanding_bool.column("active_expanding_true_rate")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_bool_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 1, 2, 3 }, lazy_expanding_true_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 1, 1 }, lazy_expanding_false_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_expanding_bool_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_expanding_true_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), lazy_expanding_true_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), lazy_expanding_true_rate[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_expanding_any);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false }, lazy_expanding_all);
}
