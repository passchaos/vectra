const std = @import("std");
const vectra = @import("vectra");

const DeviceLazyFrame = vectra.DeviceLazyFrame;
const helpers = @import("dataframe_lazy_test_helpers.zig");
const lazyCollectTable = helpers.lazyCollectTable;
const lazyQualityTable = helpers.lazyQualityTable;

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

test "device lazy frame filters by named boolean columns" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();
    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.withColumnCompareScalar("big_sale", "sales", f64, 4.0, .gt);
    try plan.filterColumn("big_sale");
    try plan.select(&.{ "sales", "big_sale" });

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "filter_column(big_sale)") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 2), result.height());
    try std.testing.expectEqual(@as(usize, 2), result.width());
    const result_sales = try (try result.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales);
    const result_big_sale = try (try result.column("big_sale")).bool.toOwnedSlice(gpa);
    defer gpa.free(result_big_sale);
    try std.testing.expectEqualSlices(f64, &.{ 5.0, 7.0 }, result_sales);
    try std.testing.expectEqualSlices(bool, &.{ true, true }, result_big_sale);

    var source_bool_plan = try DeviceLazyFrame.init(gpa, table);
    defer source_bool_plan.deinit();
    try source_bool_plan.filterColumn("active");
    try source_bool_plan.select(&.{ "sales", "active" });
    var active_result = try source_bool_plan.collect();
    defer active_result.deinit();
    const active_sales = try (try active_result.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(active_sales);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 5.0, 7.0 }, active_sales);
}

test "device lazy frame selects columns by dtype" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();
    var numeric_plan = try DeviceLazyFrame.init(gpa, table);
    defer numeric_plan.deinit();
    try numeric_plan.selectNumeric();

    const numeric_explain = try numeric_plan.explain(gpa);
    defer gpa.free(numeric_explain);
    try std.testing.expect(std.mem.indexOf(u8, numeric_explain, "select_dtype_class(numeric)") != null);

    var numeric = try numeric_plan.collect();
    defer numeric.deinit();
    try std.testing.expectEqual(@as(usize, 2), numeric.width());
    try std.testing.expectEqual(@as(?usize, 0), numeric.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), numeric.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, null), numeric.columnIndex("active"));

    var exact_plan = try DeviceLazyFrame.init(gpa, table);
    defer exact_plan.deinit();
    try exact_plan.selectByDTypes(&.{ .bool, .f64 });
    const exact_explain = try exact_plan.explain(gpa);
    defer gpa.free(exact_explain);
    try std.testing.expect(std.mem.indexOf(u8, exact_explain, "select_dtypes[bool,f64]") != null);

    var exact = try exact_plan.collect();
    defer exact.deinit();
    try std.testing.expectEqual(@as(usize, 2), exact.width());
    try std.testing.expectEqual(@as(?usize, 0), exact.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), exact.columnIndex("active"));

    var empty_plan = try DeviceLazyFrame.init(gpa, table);
    defer empty_plan.deinit();
    try empty_plan.selectFloat();
    try empty_plan.selectInteger();
    var empty = try empty_plan.collect();
    defer empty.deinit();
    try std.testing.expectEqual(@as(usize, 0), empty.width());
    try std.testing.expectEqual(table.height(), empty.height());
}

test "device lazy frame casts columns" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();
    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.castColumn("units", .f64);
    try plan.castColumn("active", .i8);
    try plan.select(&.{ "units", "active" });

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "cast_column(units->f64)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "cast_column(active->i8)") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 2), result.width());
    try std.testing.expectEqual(vectra.DeviceDType.f64, try result.columnDType("units"));
    try std.testing.expectEqual(vectra.DeviceDType.i8, try result.columnDType("active"));
    const units = try (try result.column("units")).f64.toOwnedSlice(gpa);
    defer gpa.free(units);
    const active = try (try result.column("active")).i8.toOwnedSlice(gpa);
    defer gpa.free(active);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 2.0, 3.0, 4.0 }, units);
    try std.testing.expectEqualSlices(i8, &.{ 1, 0, 1, 1 }, active);

    var invalid_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_plan.deinit();
    try invalid_plan.castColumn("missing", .f64);
    try std.testing.expectError(error.ColumnNotFound, invalid_plan.collect());
}

test "device lazy frame fills nullable columns" {
    const gpa = std.testing.allocator;
    var table = try lazyQualityTable(gpa);
    defer table.deinit();
    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.fillNullColumn("quality", f64, -1.0);

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "fill_null_column(quality=scalar:f64)") != null);

    var filled = try plan.collect();
    defer filled.deinit();
    try std.testing.expectEqual(@as(usize, 0), (try filled.column("quality")).nullCount());
    const quality = try (try filled.column("quality")).f64.toOwnedSlice(gpa);
    defer gpa.free(quality);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, -1.0, 3.0, 4.0 }, quality);

    var invalid_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_plan.deinit();
    try invalid_plan.fillNullColumn("quality", i64, 0);
    try std.testing.expectError(error.TypeUnsupported, invalid_plan.collect());
}

test "device lazy frame drops null rows" {
    const gpa = std.testing.allocator;
    var table = try lazyQualityTable(gpa);
    defer table.deinit();
    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.dropNullsColumn("quality");

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "drop_nulls[quality]") != null);

    var dropped = try plan.collect();
    defer dropped.deinit();
    try std.testing.expectEqual(@as(usize, 3), dropped.height());
    const quality = try (try dropped.column("quality")).f64.toOwnedSlice(gpa);
    defer gpa.free(quality);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 3.0, 4.0 }, quality);
    try std.testing.expectEqual(@as(usize, 0), (try dropped.column("quality")).nullCount());

    var invalid_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_plan.deinit();
    try invalid_plan.dropNullsColumn("missing");
    try std.testing.expectError(error.ColumnNotFound, invalid_plan.collect());
}

test "device lazy frame renames and drops columns" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();
    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.withColumnLiteral("segment", i32, 42);
    try plan.withColumnLiteral("always_true", bool, true);
    try plan.withRowIndex("row_nr", 100);
    try plan.renameColumn("sales", "revenue");
    try plan.dropColumn("active");
    try plan.select(&.{ "row_nr", "segment", "always_true", "revenue", "units" });

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_literal(segment=scalar:i32)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_literal(always_true=scalar:bool)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_row_index(row_nr, offset=100)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "rename_column(sales->revenue)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "drop_columns[active]") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 4), result.height());
    try std.testing.expectEqual(@as(usize, 5), result.width());
    try std.testing.expectEqual(@as(?usize, 0), result.columnIndex("row_nr"));
    try std.testing.expectEqual(@as(?usize, 1), result.columnIndex("segment"));
    try std.testing.expectEqual(@as(?usize, 2), result.columnIndex("always_true"));
    try std.testing.expectEqual(@as(?usize, 3), result.columnIndex("revenue"));
    try std.testing.expectEqual(@as(?usize, null), result.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, null), result.columnIndex("active"));
    const row_nr = try (try result.column("row_nr")).usize.toOwnedSlice(gpa);
    defer gpa.free(row_nr);
    const segment = try (try result.column("segment")).i32.toOwnedSlice(gpa);
    defer gpa.free(segment);
    const always_true = try (try result.column("always_true")).bool.toOwnedSlice(gpa);
    defer gpa.free(always_true);
    const revenue = try (try result.column("revenue")).f64.toOwnedSlice(gpa);
    defer gpa.free(revenue);
    const units = try (try result.column("units")).i64.toOwnedSlice(gpa);
    defer gpa.free(units);
    try std.testing.expectEqualSlices(usize, &.{ 100, 101, 102, 103 }, row_nr);
    try std.testing.expectEqualSlices(i32, &.{ 42, 42, 42, 42 }, segment);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, always_true);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 3.0, 5.0, 7.0 }, revenue);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4 }, units);

    var drop_many_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_many_plan.deinit();
    try drop_many_plan.dropColumns(&.{ "units", "active" });
    var drop_many = try drop_many_plan.collect();
    defer drop_many.deinit();
    try std.testing.expectEqual(@as(usize, 1), drop_many.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_many.columnIndex("sales"));

    var invalid_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_plan.deinit();
    try invalid_plan.renameColumn("sales", "units");
    try std.testing.expectError(error.InvalidShape, invalid_plan.collect());

    var invalid_index_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_index_plan.deinit();
    try invalid_index_plan.withRowIndex("sales", 0);
    try std.testing.expectError(error.InvalidShape, invalid_index_plan.collect());

    var replace_literal_plan = try DeviceLazyFrame.init(gpa, table);
    defer replace_literal_plan.deinit();
    try replace_literal_plan.withColumnLiteral("sales", f64, 1.0);
    try replace_literal_plan.select(&.{"sales"});
    var replaced_literal = try replace_literal_plan.collect();
    defer replaced_literal.deinit();
    const replaced_sales = try (try replaced_literal.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(replaced_sales);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 1.0, 1.0, 1.0 }, replaced_sales);
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

test "device lazy frame collects row slice operations" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();
    var slice_plan = try DeviceLazyFrame.init(gpa, table);
    defer slice_plan.deinit();
    try slice_plan.sliceRows(1, 3);
    try slice_plan.select(&.{"sales"});

    const slice_explain = try slice_plan.explain(gpa);
    defer gpa.free(slice_explain);
    try std.testing.expect(std.mem.indexOf(u8, slice_explain, "slice_rows(1..3)") != null);

    var sliced = try slice_plan.collect();
    defer sliced.deinit();
    try std.testing.expectEqual(@as(usize, 2), sliced.height());
    try std.testing.expectEqual(@as(usize, 1), sliced.width());
    const sliced_sales = try (try sliced.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(sliced_sales);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 5.0 }, sliced_sales);

    var len_plan = try DeviceLazyFrame.init(gpa, table);
    defer len_plan.deinit();
    try len_plan.slice(2, 8);
    var len_sliced = try len_plan.collect();
    defer len_sliced.deinit();
    const len_sliced_sales = try (try len_sliced.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(len_sliced_sales);
    try std.testing.expectEqualSlices(f64, &.{ 5.0, 7.0 }, len_sliced_sales);

    var take_plan = try DeviceLazyFrame.init(gpa, table);
    defer take_plan.deinit();
    try take_plan.take(&.{ 3, 1, 1 });
    try take_plan.select(&.{ "sales", "units" });
    const take_explain = try take_plan.explain(gpa);
    defer gpa.free(take_explain);
    try std.testing.expect(std.mem.indexOf(u8, take_explain, "take_rows([3,1,1])") != null);
    var taken = try take_plan.collect();
    defer taken.deinit();
    try std.testing.expectEqual(@as(usize, 3), taken.height());
    try std.testing.expectEqual(@as(usize, 2), taken.width());
    const taken_sales = try (try taken.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(taken_sales);
    const taken_units = try (try taken.column("units")).i64.toOwnedSlice(gpa);
    defer gpa.free(taken_units);
    try std.testing.expectEqualSlices(f64, &.{ 7.0, 3.0, 3.0 }, taken_sales);
    try std.testing.expectEqualSlices(i64, &.{ 4, 2, 2 }, taken_units);

    var invalid_take_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_take_plan.deinit();
    try invalid_take_plan.take(&.{4});
    try std.testing.expectError(error.IndexOutOfBounds, invalid_take_plan.collect());
}

test "device lazy frame reverses rows" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();
    var reverse_plan = try DeviceLazyFrame.init(gpa, table);
    defer reverse_plan.deinit();
    try reverse_plan.reverseRows();
    try reverse_plan.select(&.{ "sales", "units", "active" });

    const reverse_explain = try reverse_plan.explain(gpa);
    defer gpa.free(reverse_explain);
    try std.testing.expect(std.mem.indexOf(u8, reverse_explain, "reverse_rows") != null);

    var reversed = try reverse_plan.collect();
    defer reversed.deinit();
    try std.testing.expectEqual(@as(usize, 4), reversed.height());
    try std.testing.expectEqual(@as(usize, 3), reversed.width());
    const sales = try (try reversed.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(sales);
    const units = try (try reversed.column("units")).i64.toOwnedSlice(gpa);
    defer gpa.free(units);
    const active = try (try reversed.column("active")).bool.toOwnedSlice(gpa);
    defer gpa.free(active);
    try std.testing.expectEqualSlices(f64, &.{ 7.0, 5.0, 3.0, 2.0 }, sales);
    try std.testing.expectEqualSlices(i64, &.{ 4, 3, 2, 1 }, units);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, active);
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
