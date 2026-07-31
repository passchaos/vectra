const std = @import("std");
const vectra = @import("vectra");

const DeviceColumn = vectra.DeviceColumn;
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

test "device lazy frame selects and drops columns by dtype" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();

    var positional_plan = try DeviceLazyFrame.init(gpa, table);
    defer positional_plan.deinit();
    try positional_plan.selectByColumnIndices(&.{ 2, 0 });
    const positional_explain = try positional_plan.explain(gpa);
    defer gpa.free(positional_explain);
    try std.testing.expect(std.mem.indexOf(u8, positional_explain, "select_column_indices([2,0])") != null);
    var positional = try positional_plan.collect();
    defer positional.deinit();
    try std.testing.expectEqual(@as(usize, 2), positional.width());
    try std.testing.expectEqual(@as(?usize, 0), positional.columnIndex("active"));
    try std.testing.expectEqual(@as(?usize, 1), positional.columnIndex("sales"));

    var range_plan = try DeviceLazyFrame.init(gpa, table);
    defer range_plan.deinit();
    try range_plan.selectColumnRange(1, 3);
    const range_explain = try range_plan.explain(gpa);
    defer gpa.free(range_explain);
    try std.testing.expect(std.mem.indexOf(u8, range_explain, "select_column_range(1..3)") != null);
    var range = try range_plan.collect();
    defer range.deinit();
    try std.testing.expectEqual(@as(usize, 2), range.width());
    try std.testing.expectEqual(@as(?usize, 0), range.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, 1), range.columnIndex("active"));

    var first_plan = try DeviceLazyFrame.init(gpa, table);
    defer first_plan.deinit();
    try first_plan.selectFirstColumns(2);
    const first_explain = try first_plan.explain(gpa);
    defer gpa.free(first_explain);
    try std.testing.expect(std.mem.indexOf(u8, first_explain, "select_column_range(0..2)") != null);
    var first = try first_plan.collect();
    defer first.deinit();
    try std.testing.expectEqual(@as(?usize, 0), first.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), first.columnIndex("units"));

    var last_plan = try DeviceLazyFrame.init(gpa, table);
    defer last_plan.deinit();
    try last_plan.selectLastColumns(2);
    const last_explain = try last_plan.explain(gpa);
    defer gpa.free(last_explain);
    try std.testing.expect(std.mem.indexOf(u8, last_explain, "select_last_columns(2)") != null);
    var last = try last_plan.collect();
    defer last.deinit();
    try std.testing.expectEqual(@as(?usize, 0), last.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, 1), last.columnIndex("active"));

    var drop_positional_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_positional_plan.deinit();
    try drop_positional_plan.dropByColumnIndices(&.{1});
    const drop_positional_explain = try drop_positional_plan.explain(gpa);
    defer gpa.free(drop_positional_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_positional_explain, "drop_column_indices([1])") != null);
    var drop_positional = try drop_positional_plan.collect();
    defer drop_positional.deinit();
    try std.testing.expectEqual(@as(usize, 2), drop_positional.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_positional.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), drop_positional.columnIndex("active"));

    var drop_range_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_range_plan.deinit();
    try drop_range_plan.dropColumnRange(1, 3);
    const drop_range_explain = try drop_range_plan.explain(gpa);
    defer gpa.free(drop_range_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_range_explain, "drop_column_range(1..3)") != null);
    var drop_range = try drop_range_plan.collect();
    defer drop_range.deinit();
    try std.testing.expectEqual(@as(usize, 1), drop_range.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_range.columnIndex("sales"));

    var drop_first_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_first_plan.deinit();
    try drop_first_plan.dropFirstColumns(1);
    const drop_first_explain = try drop_first_plan.explain(gpa);
    defer gpa.free(drop_first_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_first_explain, "drop_column_range(0..1)") != null);
    var drop_first = try drop_first_plan.collect();
    defer drop_first.deinit();
    try std.testing.expectEqual(@as(?usize, 0), drop_first.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, 1), drop_first.columnIndex("active"));

    var drop_last_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_last_plan.deinit();
    try drop_last_plan.dropLastColumns(1);
    const drop_last_explain = try drop_last_plan.explain(gpa);
    defer gpa.free(drop_last_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_last_explain, "drop_last_columns(1)") != null);
    var drop_last = try drop_last_plan.collect();
    defer drop_last.deinit();
    try std.testing.expectEqual(@as(?usize, 0), drop_last.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), drop_last.columnIndex("units"));

    var reverse_columns_plan = try DeviceLazyFrame.init(gpa, table);
    defer reverse_columns_plan.deinit();
    try reverse_columns_plan.reverseColumns();
    const reverse_columns_explain = try reverse_columns_plan.explain(gpa);
    defer gpa.free(reverse_columns_explain);
    try std.testing.expect(std.mem.indexOf(u8, reverse_columns_explain, "reverse_columns") != null);
    var reversed_columns = try reverse_columns_plan.collect();
    defer reversed_columns.deinit();
    try std.testing.expectEqual(@as(?usize, 0), reversed_columns.columnIndex("active"));
    try std.testing.expectEqual(@as(?usize, 1), reversed_columns.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, 2), reversed_columns.columnIndex("sales"));

    var sort_columns_plan = try DeviceLazyFrame.init(gpa, table);
    defer sort_columns_plan.deinit();
    try sort_columns_plan.sortColumnsByName(false);
    const sort_columns_explain = try sort_columns_plan.explain(gpa);
    defer gpa.free(sort_columns_explain);
    try std.testing.expect(std.mem.indexOf(u8, sort_columns_explain, "sort_columns_by_name(desc=false)") != null);
    var sorted_columns = try sort_columns_plan.collect();
    defer sorted_columns.deinit();
    try std.testing.expectEqual(@as(?usize, 0), sorted_columns.columnIndex("active"));
    try std.testing.expectEqual(@as(?usize, 1), sorted_columns.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 2), sorted_columns.columnIndex("units"));

    var sort_columns_desc_plan = try DeviceLazyFrame.init(gpa, table);
    defer sort_columns_desc_plan.deinit();
    try sort_columns_desc_plan.sortColumnsByName(true);
    var sorted_columns_desc = try sort_columns_desc_plan.collect();
    defer sorted_columns_desc.deinit();
    try std.testing.expectEqual(@as(?usize, 0), sorted_columns_desc.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, 1), sorted_columns_desc.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 2), sorted_columns_desc.columnIndex("active"));

    var invalid_positional_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_positional_plan.deinit();
    try invalid_positional_plan.selectByColumnIndices(&.{3});
    try std.testing.expectError(error.IndexOutOfBounds, invalid_positional_plan.collect());

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

    var drop_numeric_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_numeric_plan.deinit();
    try drop_numeric_plan.dropNumeric();
    const drop_numeric_explain = try drop_numeric_plan.explain(gpa);
    defer gpa.free(drop_numeric_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_numeric_explain, "drop_dtype_class(numeric)") != null);

    var non_numeric = try drop_numeric_plan.collect();
    defer non_numeric.deinit();
    try std.testing.expectEqual(@as(usize, 1), non_numeric.width());
    try std.testing.expectEqual(@as(?usize, 0), non_numeric.columnIndex("active"));
    try std.testing.expectEqual(@as(?usize, null), non_numeric.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, null), non_numeric.columnIndex("units"));

    var drop_exact_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_exact_plan.deinit();
    try drop_exact_plan.dropByDTypes(&.{ .bool, .f64 });
    const drop_exact_explain = try drop_exact_plan.explain(gpa);
    defer gpa.free(drop_exact_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_exact_explain, "drop_dtypes[bool,f64]") != null);

    var drop_exact = try drop_exact_plan.collect();
    defer drop_exact.deinit();
    try std.testing.expectEqual(@as(usize, 1), drop_exact.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_exact.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, null), drop_exact.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, null), drop_exact.columnIndex("active"));

    var drop_all_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_all_plan.deinit();
    try drop_all_plan.dropByDTypes(&.{ .f64, .i64, .bool });
    var drop_all = try drop_all_plan.collect();
    defer drop_all.deinit();
    try std.testing.expectEqual(@as(usize, 0), drop_all.width());
    try std.testing.expectEqual(table.height(), drop_all.height());
}

test "device lazy frame selects and drops columns by nullability" {
    const gpa = std.testing.allocator;

    var sales = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 3.0, 5.0 }, .cpu);
    defer sales.deinit();
    var audited_units = try DeviceColumn.fromSliceWithValidity(i64, gpa, &.{ 1, 2, 3 }, &.{ true, true, true }, .cpu);
    defer audited_units.deinit();
    var quality = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 0.8, 0.0, 0.9 }, &.{ true, false, true }, .cpu);
    defer quality.deinit();
    var active = try DeviceColumn.fromSlice(bool, gpa, &.{ true, false, true }, .cpu);
    defer active.deinit();

    var table = try vectra.DeviceDataFrame.init(gpa, &.{
        .{ .name = "sales", .data = sales },
        .{ .name = "audited_units", .data = audited_units },
        .{ .name = "quality", .data = quality },
        .{ .name = "active", .data = active },
    });
    defer table.deinit();

    var nullable_plan = try DeviceLazyFrame.init(gpa, table);
    defer nullable_plan.deinit();
    try nullable_plan.selectNullableColumns();
    const nullable_explain = try nullable_plan.explain(gpa);
    defer gpa.free(nullable_explain);
    try std.testing.expect(std.mem.indexOf(u8, nullable_explain, "select_nullable_columns") != null);
    var nullable = try nullable_plan.collect();
    defer nullable.deinit();
    try std.testing.expectEqual(@as(usize, 2), nullable.width());
    try std.testing.expectEqual(@as(?usize, 0), nullable.columnIndex("audited_units"));
    try std.testing.expectEqual(@as(?usize, 1), nullable.columnIndex("quality"));

    var non_nullable_plan = try DeviceLazyFrame.init(gpa, table);
    defer non_nullable_plan.deinit();
    try non_nullable_plan.selectNonNullableColumns();
    const non_nullable_explain = try non_nullable_plan.explain(gpa);
    defer gpa.free(non_nullable_explain);
    try std.testing.expect(std.mem.indexOf(u8, non_nullable_explain, "select_non_nullable_columns") != null);
    var non_nullable = try non_nullable_plan.collect();
    defer non_nullable.deinit();
    try std.testing.expectEqual(@as(usize, 2), non_nullable.width());
    try std.testing.expectEqual(@as(?usize, 0), non_nullable.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), non_nullable.columnIndex("active"));

    var with_nulls_plan = try DeviceLazyFrame.init(gpa, table);
    defer with_nulls_plan.deinit();
    try with_nulls_plan.selectColumnsWithNulls();
    const with_nulls_explain = try with_nulls_plan.explain(gpa);
    defer gpa.free(with_nulls_explain);
    try std.testing.expect(std.mem.indexOf(u8, with_nulls_explain, "select_columns_with_nulls") != null);
    var with_nulls = try with_nulls_plan.collect();
    defer with_nulls.deinit();
    try std.testing.expectEqual(@as(usize, 1), with_nulls.width());
    try std.testing.expectEqual(@as(?usize, 0), with_nulls.columnIndex("quality"));
    const quality_values = try (try with_nulls.column("quality")).f64.toOwnedSlice(gpa);
    defer gpa.free(quality_values);
    try std.testing.expectEqualSlices(f64, &.{ 0.8, 0.0, 0.9 }, quality_values);

    var without_nulls_plan = try DeviceLazyFrame.init(gpa, table);
    defer without_nulls_plan.deinit();
    try without_nulls_plan.selectColumnsWithoutNulls();
    const without_nulls_explain = try without_nulls_plan.explain(gpa);
    defer gpa.free(without_nulls_explain);
    try std.testing.expect(std.mem.indexOf(u8, without_nulls_explain, "select_columns_without_nulls") != null);
    var without_nulls = try without_nulls_plan.collect();
    defer without_nulls.deinit();
    try std.testing.expectEqual(@as(usize, 3), without_nulls.width());
    try std.testing.expectEqual(@as(?usize, 0), without_nulls.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), without_nulls.columnIndex("audited_units"));
    try std.testing.expectEqual(@as(?usize, 2), without_nulls.columnIndex("active"));

    var drop_nullable_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_nullable_plan.deinit();
    try drop_nullable_plan.dropNullableColumns();
    const drop_nullable_explain = try drop_nullable_plan.explain(gpa);
    defer gpa.free(drop_nullable_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_nullable_explain, "drop_nullable_columns") != null);
    var drop_nullable = try drop_nullable_plan.collect();
    defer drop_nullable.deinit();
    try std.testing.expectEqual(@as(usize, 2), drop_nullable.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_nullable.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), drop_nullable.columnIndex("active"));

    var drop_non_nullable_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_non_nullable_plan.deinit();
    try drop_non_nullable_plan.dropNonNullableColumns();
    const drop_non_nullable_explain = try drop_non_nullable_plan.explain(gpa);
    defer gpa.free(drop_non_nullable_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_non_nullable_explain, "drop_non_nullable_columns") != null);
    var drop_non_nullable = try drop_non_nullable_plan.collect();
    defer drop_non_nullable.deinit();
    try std.testing.expectEqual(@as(usize, 2), drop_non_nullable.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_non_nullable.columnIndex("audited_units"));
    try std.testing.expectEqual(@as(?usize, 1), drop_non_nullable.columnIndex("quality"));

    var drop_with_nulls_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_with_nulls_plan.deinit();
    try drop_with_nulls_plan.dropColumnsWithNulls();
    const drop_with_nulls_explain = try drop_with_nulls_plan.explain(gpa);
    defer gpa.free(drop_with_nulls_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_with_nulls_explain, "drop_columns_with_nulls") != null);
    var drop_with_nulls = try drop_with_nulls_plan.collect();
    defer drop_with_nulls.deinit();
    try std.testing.expectEqual(@as(usize, 3), drop_with_nulls.width());
    try std.testing.expectEqual(@as(?usize, null), drop_with_nulls.columnIndex("quality"));

    var drop_without_nulls_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_without_nulls_plan.deinit();
    try drop_without_nulls_plan.dropColumnsWithoutNulls();
    const drop_without_nulls_explain = try drop_without_nulls_plan.explain(gpa);
    defer gpa.free(drop_without_nulls_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_without_nulls_explain, "drop_columns_without_nulls") != null);
    var drop_without_nulls = try drop_without_nulls_plan.collect();
    defer drop_without_nulls.deinit();
    try std.testing.expectEqual(@as(usize, 1), drop_without_nulls.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_without_nulls.columnIndex("quality"));
}

test "device lazy frame selects and drops columns by name pattern" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();

    var prefix_plan = try DeviceLazyFrame.init(gpa, table);
    defer prefix_plan.deinit();
    try prefix_plan.withColumnScalar("sales_x2", "sales", f64, 2.0, .mul);
    try prefix_plan.selectByNamePrefix("sales");

    const prefix_explain = try prefix_plan.explain(gpa);
    defer gpa.free(prefix_explain);
    try std.testing.expect(std.mem.indexOf(u8, prefix_explain, "select_name_prefix(sales)") != null);

    var prefixed = try prefix_plan.collect();
    defer prefixed.deinit();
    try std.testing.expectEqual(@as(usize, 2), prefixed.width());
    try std.testing.expectEqual(@as(?usize, 0), prefixed.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), prefixed.columnIndex("sales_x2"));
    const sales_x2 = try (try prefixed.column("sales_x2")).f64.toOwnedSlice(gpa);
    defer gpa.free(sales_x2);
    try std.testing.expectEqualSlices(f64, &.{ 4.0, 6.0, 10.0, 14.0 }, sales_x2);

    var suffix_plan = try DeviceLazyFrame.init(gpa, table);
    defer suffix_plan.deinit();
    try suffix_plan.selectByNameSuffix("s");

    const suffix_explain = try suffix_plan.explain(gpa);
    defer gpa.free(suffix_explain);
    try std.testing.expect(std.mem.indexOf(u8, suffix_explain, "select_name_suffix(s)") != null);

    var suffixed = try suffix_plan.collect();
    defer suffixed.deinit();
    try std.testing.expectEqual(@as(usize, 2), suffixed.width());
    try std.testing.expectEqual(@as(?usize, 0), suffixed.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), suffixed.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, null), suffixed.columnIndex("active"));

    var contains_plan = try DeviceLazyFrame.init(gpa, table);
    defer contains_plan.deinit();
    try contains_plan.selectByNameContains("ct");

    const contains_explain = try contains_plan.explain(gpa);
    defer gpa.free(contains_explain);
    try std.testing.expect(std.mem.indexOf(u8, contains_explain, "select_name_contains(ct)") != null);

    var contained = try contains_plan.collect();
    defer contained.deinit();
    try std.testing.expectEqual(@as(usize, 1), contained.width());
    try std.testing.expectEqual(@as(?usize, 0), contained.columnIndex("active"));
    const active = try (try contained.column("active")).bool.toOwnedSlice(gpa);
    defer gpa.free(active);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, true }, active);

    var empty_plan = try DeviceLazyFrame.init(gpa, table);
    defer empty_plan.deinit();
    try empty_plan.selectByNameContains("missing");
    var empty = try empty_plan.collect();
    defer empty.deinit();
    try std.testing.expectEqual(@as(usize, 0), empty.width());
    try std.testing.expectEqual(table.height(), empty.height());

    var drop_prefix_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_prefix_plan.deinit();
    try drop_prefix_plan.withColumnScalar("sales_x2", "sales", f64, 2.0, .mul);
    try drop_prefix_plan.dropByNamePrefix("sales");

    const drop_prefix_explain = try drop_prefix_plan.explain(gpa);
    defer gpa.free(drop_prefix_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_prefix_explain, "drop_name_prefix(sales)") != null);

    var drop_prefixed = try drop_prefix_plan.collect();
    defer drop_prefixed.deinit();
    try std.testing.expectEqual(@as(usize, 2), drop_prefixed.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_prefixed.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, 1), drop_prefixed.columnIndex("active"));
    try std.testing.expectEqual(@as(?usize, null), drop_prefixed.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, null), drop_prefixed.columnIndex("sales_x2"));

    var drop_suffix_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_suffix_plan.deinit();
    try drop_suffix_plan.dropByNameSuffix("s");

    const drop_suffix_explain = try drop_suffix_plan.explain(gpa);
    defer gpa.free(drop_suffix_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_suffix_explain, "drop_name_suffix(s)") != null);

    var drop_suffixed = try drop_suffix_plan.collect();
    defer drop_suffixed.deinit();
    try std.testing.expectEqual(@as(usize, 1), drop_suffixed.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_suffixed.columnIndex("active"));
    try std.testing.expectEqual(@as(?usize, null), drop_suffixed.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, null), drop_suffixed.columnIndex("units"));

    var drop_contains_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_contains_plan.deinit();
    try drop_contains_plan.dropByNameContains("ct");

    const drop_contains_explain = try drop_contains_plan.explain(gpa);
    defer gpa.free(drop_contains_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_contains_explain, "drop_name_contains(ct)") != null);

    var drop_contained = try drop_contains_plan.collect();
    defer drop_contained.deinit();
    try std.testing.expectEqual(@as(usize, 2), drop_contained.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_contained.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), drop_contained.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, null), drop_contained.columnIndex("active"));

    var drop_empty_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_empty_plan.deinit();
    try drop_empty_plan.dropByNamePrefix("");
    var drop_empty = try drop_empty_plan.collect();
    defer drop_empty.deinit();
    try std.testing.expectEqual(@as(usize, 0), drop_empty.width());
    try std.testing.expectEqual(table.height(), drop_empty.height());
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

test "device lazy frame coalesces nullable columns" {
    const gpa = std.testing.allocator;
    var table = try lazyQualityTable(gpa);
    defer table.deinit();
    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.withColumnLiteral("fallback_quality", f64, 9.0);
    try plan.coalesceColumns("quality", "fallback_quality", "quality_coalesced");

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_literal(fallback_quality=scalar:f64)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "coalesce_columns(quality,fallback_quality->quality_coalesced)") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 0), (try result.column("quality_coalesced")).nullCount());
    const values = try (try result.column("quality_coalesced")).f64.toOwnedSlice(gpa);
    defer gpa.free(values);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 9.0, 3.0, 4.0 }, values);

    var mismatch_plan = try DeviceLazyFrame.init(gpa, table);
    defer mismatch_plan.deinit();
    try mismatch_plan.withColumnLiteral("fallback_i64", i64, 9);
    try mismatch_plan.coalesceColumns("quality", "fallback_i64", "bad");
    try std.testing.expectError(error.TypeMismatch, mismatch_plan.collect());
}

test "device lazy frame derives null predicate columns" {
    const gpa = std.testing.allocator;
    var table = try lazyQualityTable(gpa);
    defer table.deinit();
    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.isNullColumn("quality", "quality_is_null");
    try plan.isValidColumn("quality", "quality_is_valid");

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "is_null_column(quality->quality_is_null)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "is_valid_column(quality->quality_is_valid)") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 3), result.width());
    const is_null = try (try result.column("quality_is_null")).bool.toOwnedSlice(gpa);
    defer gpa.free(is_null);
    const is_valid = try (try result.column("quality_is_valid")).bool.toOwnedSlice(gpa);
    defer gpa.free(is_valid);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false }, is_null);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, true }, is_valid);

    var invalid_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_plan.deinit();
    try invalid_plan.isNullColumn("missing", "missing_is_null");
    try std.testing.expectError(error.ColumnNotFound, invalid_plan.collect());
}

test "device lazy frame derives row null and valid count columns" {
    const gpa = std.testing.allocator;

    var sales = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 3.0, 5.0, 7.0 }, .cpu);
    defer sales.deinit();
    var quality = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 0.8, 0.0, 0.9, 1.0 }, &.{ true, false, true, true }, .cpu);
    defer quality.deinit();
    var flag = try DeviceColumn.fromSliceWithValidity(bool, gpa, &.{ true, false, true, false }, &.{ true, true, false, false }, .cpu);
    defer flag.deinit();

    var table = try vectra.DeviceDataFrame.init(gpa, &.{
        .{ .name = "sales", .data = sales },
        .{ .name = "quality", .data = quality },
        .{ .name = "flag", .data = flag },
    });
    defer table.deinit();

    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.withRowValidCount(&.{}, "row_valids_all");
    try plan.withRowNullCount(&.{ "quality", "flag" }, "row_nulls");
    try plan.select(&.{ "row_nulls", "row_valids_all" });

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_null_count([quality,flag]->row_nulls)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_valid_count([]->row_valids_all)") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 2), result.width());
    const row_nulls = try (try result.column("row_nulls")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_nulls);
    const row_valids_all = try (try result.column("row_valids_all")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_valids_all);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 1, 1 }, row_nulls);
    try std.testing.expectEqualSlices(i64, &.{ 3, 2, 2, 2 }, row_valids_all);

    var invalid_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_plan.deinit();
    try invalid_plan.withRowNullCount(&.{ "quality", "missing" }, "bad_count");
    try std.testing.expectError(error.ColumnNotFound, invalid_plan.collect());
}

test "device lazy frame derives NaN and finite predicate columns" {
    const gpa = std.testing.allocator;

    var metric = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, std.math.nan(f64), std.math.inf(f64), 7.0 }, &.{ true, true, true, false }, .cpu);
    defer metric.deinit();
    var id = try DeviceColumn.fromSlice(i64, gpa, &.{ 10, 20, 30, 40 }, .cpu);
    defer id.deinit();

    var table = try vectra.DeviceDataFrame.init(gpa, &.{
        .{ .name = "metric", .data = metric },
        .{ .name = "id", .data = id },
    });
    defer table.deinit();

    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.isNanColumn("metric", "metric_is_nan");
    try plan.isFiniteColumn("metric", "metric_is_finite");
    try plan.isInfColumn("metric", "metric_is_inf");
    try plan.isFiniteColumn("id", "id_is_finite");
    try plan.select(&.{ "metric_is_nan", "metric_is_finite", "metric_is_inf", "id_is_finite" });

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "is_nan_column(metric->metric_is_nan)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "is_finite_column(metric->metric_is_finite)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "is_inf_column(metric->metric_is_inf)") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 4), result.width());
    const metric_is_nan = try (try result.column("metric_is_nan")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_is_nan);
    const metric_is_finite = try (try result.column("metric_is_finite")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_is_finite);
    const metric_is_inf = try (try result.column("metric_is_inf")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_is_inf);
    const id_is_finite = try (try result.column("id_is_finite")).bool.toOwnedSlice(gpa);
    defer gpa.free(id_is_finite);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false }, metric_is_nan);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false }, metric_is_finite);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false }, metric_is_inf);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, id_is_finite);

    var fill_nan_plan = try DeviceLazyFrame.init(gpa, table);
    defer fill_nan_plan.deinit();
    try fill_nan_plan.fillNaNColumn("metric", f64, -1.0);
    const fill_nan_explain = try fill_nan_plan.explain(gpa);
    defer gpa.free(fill_nan_explain);
    try std.testing.expect(std.mem.indexOf(u8, fill_nan_explain, "fill_nan_column(metric=scalar:f64)") != null);
    var filled_nan = try fill_nan_plan.collect();
    defer filled_nan.deinit();
    const filled_metric = try (try filled_nan.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(filled_metric);
    const filled_metric_validity = try (try filled_nan.column("metric")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(filled_metric_validity);
    try std.testing.expectEqual(@as(f64, 1.0), filled_metric[0]);
    try std.testing.expectEqual(@as(f64, -1.0), filled_metric[1]);
    try std.testing.expect(std.math.isInf(filled_metric[2]));
    try std.testing.expectEqual(@as(f64, 7.0), filled_metric[3]);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false }, filled_metric_validity);

    var fill_nan_mismatch_plan = try DeviceLazyFrame.init(gpa, table);
    defer fill_nan_mismatch_plan.deinit();
    try fill_nan_mismatch_plan.fillNaNColumn("metric", i64, 0);
    try std.testing.expectError(error.TypeUnsupported, fill_nan_mismatch_plan.collect());

    var fill_inf_plan = try DeviceLazyFrame.init(gpa, table);
    defer fill_inf_plan.deinit();
    try fill_inf_plan.fillInfColumn("metric", f64, -9.0);
    const fill_inf_explain = try fill_inf_plan.explain(gpa);
    defer gpa.free(fill_inf_explain);
    try std.testing.expect(std.mem.indexOf(u8, fill_inf_explain, "fill_inf_column(metric=scalar:f64)") != null);
    var filled_inf = try fill_inf_plan.collect();
    defer filled_inf.deinit();
    const filled_inf_metric = try (try filled_inf.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(filled_inf_metric);
    const filled_inf_validity = try (try filled_inf.column("metric")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(filled_inf_validity);
    try std.testing.expectEqual(@as(f64, 1.0), filled_inf_metric[0]);
    try std.testing.expect(std.math.isNan(filled_inf_metric[1]));
    try std.testing.expectEqual(@as(f64, -9.0), filled_inf_metric[2]);
    try std.testing.expectEqual(@as(f64, 7.0), filled_inf_metric[3]);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false }, filled_inf_validity);

    var fill_inf_mismatch_plan = try DeviceLazyFrame.init(gpa, table);
    defer fill_inf_mismatch_plan.deinit();
    try fill_inf_mismatch_plan.fillInfColumn("metric", i64, 0);
    try std.testing.expectError(error.TypeUnsupported, fill_inf_mismatch_plan.collect());

    var fill_non_finite_plan = try DeviceLazyFrame.init(gpa, table);
    defer fill_non_finite_plan.deinit();
    try fill_non_finite_plan.fillNonFiniteColumn("metric", f64, -5.0);
    const fill_non_finite_explain = try fill_non_finite_plan.explain(gpa);
    defer gpa.free(fill_non_finite_explain);
    try std.testing.expect(std.mem.indexOf(u8, fill_non_finite_explain, "fill_non_finite_column(metric=scalar:f64)") != null);
    var filled_non_finite = try fill_non_finite_plan.collect();
    defer filled_non_finite.deinit();
    const filled_non_finite_metric = try (try filled_non_finite.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(filled_non_finite_metric);
    const filled_non_finite_validity = try (try filled_non_finite.column("metric")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(filled_non_finite_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, -5.0, -5.0, 7.0 }, filled_non_finite_metric);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false }, filled_non_finite_validity);

    var fill_non_finite_mismatch_plan = try DeviceLazyFrame.init(gpa, table);
    defer fill_non_finite_mismatch_plan.deinit();
    try fill_non_finite_mismatch_plan.fillNonFiniteColumn("metric", i64, 0);
    try std.testing.expectError(error.TypeUnsupported, fill_non_finite_mismatch_plan.collect());

    var select_nan_columns_plan = try DeviceLazyFrame.init(gpa, table);
    defer select_nan_columns_plan.deinit();
    try select_nan_columns_plan.selectColumnsWithNaNs();
    const select_nan_columns_explain = try select_nan_columns_plan.explain(gpa);
    defer gpa.free(select_nan_columns_explain);
    try std.testing.expect(std.mem.indexOf(u8, select_nan_columns_explain, "select_columns_with_nans") != null);
    var nan_columns = try select_nan_columns_plan.collect();
    defer nan_columns.deinit();
    try std.testing.expectEqual(@as(usize, 1), nan_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), nan_columns.columnIndex("metric"));

    var select_non_nan_columns_plan = try DeviceLazyFrame.init(gpa, table);
    defer select_non_nan_columns_plan.deinit();
    try select_non_nan_columns_plan.selectColumnsWithoutNaNs();
    const select_non_nan_columns_explain = try select_non_nan_columns_plan.explain(gpa);
    defer gpa.free(select_non_nan_columns_explain);
    try std.testing.expect(std.mem.indexOf(u8, select_non_nan_columns_explain, "select_columns_without_nans") != null);
    var non_nan_columns = try select_non_nan_columns_plan.collect();
    defer non_nan_columns.deinit();
    try std.testing.expectEqual(@as(usize, 1), non_nan_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), non_nan_columns.columnIndex("id"));

    var drop_nan_columns_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_nan_columns_plan.deinit();
    try drop_nan_columns_plan.dropColumnsWithNaNs();
    const drop_nan_columns_explain = try drop_nan_columns_plan.explain(gpa);
    defer gpa.free(drop_nan_columns_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_nan_columns_explain, "drop_columns_with_nans") != null);
    var drop_nan_columns = try drop_nan_columns_plan.collect();
    defer drop_nan_columns.deinit();
    try std.testing.expectEqual(@as(usize, 1), drop_nan_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_nan_columns.columnIndex("id"));

    var drop_non_nan_columns_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_non_nan_columns_plan.deinit();
    try drop_non_nan_columns_plan.dropColumnsWithoutNaNs();
    const drop_non_nan_columns_explain = try drop_non_nan_columns_plan.explain(gpa);
    defer gpa.free(drop_non_nan_columns_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_non_nan_columns_explain, "drop_columns_without_nans") != null);
    var drop_non_nan_columns = try drop_non_nan_columns_plan.collect();
    defer drop_non_nan_columns.deinit();
    try std.testing.expectEqual(@as(usize, 1), drop_non_nan_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_non_nan_columns.columnIndex("metric"));

    var select_inf_columns_plan = try DeviceLazyFrame.init(gpa, table);
    defer select_inf_columns_plan.deinit();
    try select_inf_columns_plan.selectColumnsWithInfs();
    const select_inf_columns_explain = try select_inf_columns_plan.explain(gpa);
    defer gpa.free(select_inf_columns_explain);
    try std.testing.expect(std.mem.indexOf(u8, select_inf_columns_explain, "select_columns_with_infs") != null);
    var inf_columns = try select_inf_columns_plan.collect();
    defer inf_columns.deinit();
    try std.testing.expectEqual(@as(usize, 1), inf_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), inf_columns.columnIndex("metric"));

    var select_non_inf_columns_plan = try DeviceLazyFrame.init(gpa, table);
    defer select_non_inf_columns_plan.deinit();
    try select_non_inf_columns_plan.selectColumnsWithoutInfs();
    const select_non_inf_columns_explain = try select_non_inf_columns_plan.explain(gpa);
    defer gpa.free(select_non_inf_columns_explain);
    try std.testing.expect(std.mem.indexOf(u8, select_non_inf_columns_explain, "select_columns_without_infs") != null);
    var non_inf_columns = try select_non_inf_columns_plan.collect();
    defer non_inf_columns.deinit();
    try std.testing.expectEqual(@as(usize, 1), non_inf_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), non_inf_columns.columnIndex("id"));

    var drop_inf_columns_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_inf_columns_plan.deinit();
    try drop_inf_columns_plan.dropColumnsWithInfs();
    const drop_inf_columns_explain = try drop_inf_columns_plan.explain(gpa);
    defer gpa.free(drop_inf_columns_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_inf_columns_explain, "drop_columns_with_infs") != null);
    var drop_inf_columns = try drop_inf_columns_plan.collect();
    defer drop_inf_columns.deinit();
    try std.testing.expectEqual(@as(usize, 1), drop_inf_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_inf_columns.columnIndex("id"));

    var drop_non_inf_columns_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_non_inf_columns_plan.deinit();
    try drop_non_inf_columns_plan.dropColumnsWithoutInfs();
    const drop_non_inf_columns_explain = try drop_non_inf_columns_plan.explain(gpa);
    defer gpa.free(drop_non_inf_columns_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_non_inf_columns_explain, "drop_columns_without_infs") != null);
    var drop_non_inf_columns = try drop_non_inf_columns_plan.collect();
    defer drop_non_inf_columns.deinit();
    try std.testing.expectEqual(@as(usize, 1), drop_non_inf_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_non_inf_columns.columnIndex("metric"));

    var select_non_finite_columns_plan = try DeviceLazyFrame.init(gpa, table);
    defer select_non_finite_columns_plan.deinit();
    try select_non_finite_columns_plan.selectColumnsWithNonFinites();
    const select_non_finite_columns_explain = try select_non_finite_columns_plan.explain(gpa);
    defer gpa.free(select_non_finite_columns_explain);
    try std.testing.expect(std.mem.indexOf(u8, select_non_finite_columns_explain, "select_columns_with_non_finites") != null);
    var non_finite_columns = try select_non_finite_columns_plan.collect();
    defer non_finite_columns.deinit();
    try std.testing.expectEqual(@as(usize, 1), non_finite_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), non_finite_columns.columnIndex("metric"));

    var select_finite_columns_plan = try DeviceLazyFrame.init(gpa, table);
    defer select_finite_columns_plan.deinit();
    try select_finite_columns_plan.selectColumnsWithoutNonFinites();
    const select_finite_columns_explain = try select_finite_columns_plan.explain(gpa);
    defer gpa.free(select_finite_columns_explain);
    try std.testing.expect(std.mem.indexOf(u8, select_finite_columns_explain, "select_columns_without_non_finites") != null);
    var finite_columns = try select_finite_columns_plan.collect();
    defer finite_columns.deinit();
    try std.testing.expectEqual(@as(usize, 1), finite_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), finite_columns.columnIndex("id"));

    var drop_non_finite_columns_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_non_finite_columns_plan.deinit();
    try drop_non_finite_columns_plan.dropColumnsWithNonFinites();
    const drop_non_finite_columns_explain = try drop_non_finite_columns_plan.explain(gpa);
    defer gpa.free(drop_non_finite_columns_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_non_finite_columns_explain, "drop_columns_with_non_finites") != null);
    var drop_non_finite_columns = try drop_non_finite_columns_plan.collect();
    defer drop_non_finite_columns.deinit();
    try std.testing.expectEqual(@as(usize, 1), drop_non_finite_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_non_finite_columns.columnIndex("id"));

    var drop_finite_columns_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_finite_columns_plan.deinit();
    try drop_finite_columns_plan.dropColumnsWithoutNonFinites();
    const drop_finite_columns_explain = try drop_finite_columns_plan.explain(gpa);
    defer gpa.free(drop_finite_columns_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_finite_columns_explain, "drop_columns_without_non_finites") != null);
    var drop_finite_columns = try drop_finite_columns_plan.collect();
    defer drop_finite_columns.deinit();
    try std.testing.expectEqual(@as(usize, 1), drop_finite_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_finite_columns.columnIndex("metric"));

    var drop_nan_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_nan_plan.deinit();
    try drop_nan_plan.dropNaNsColumn("metric");
    const drop_nan_explain = try drop_nan_plan.explain(gpa);
    defer gpa.free(drop_nan_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_nan_explain, "drop_nans[metric]") != null);
    var dropped_nan_rows = try drop_nan_plan.collect();
    defer dropped_nan_rows.deinit();
    try std.testing.expectEqual(@as(usize, 3), dropped_nan_rows.height());
    const dropped_nan_metric = try (try dropped_nan_rows.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(dropped_nan_metric);
    try std.testing.expect(!std.math.isNan(dropped_nan_metric[0]));
    try std.testing.expect(std.math.isInf(dropped_nan_metric[1]));
    try std.testing.expectEqual(@as(f64, 7.0), dropped_nan_metric[2]);

    var filter_nan_plan = try DeviceLazyFrame.init(gpa, table);
    defer filter_nan_plan.deinit();
    try filter_nan_plan.filterNaNsColumn("metric");
    const filter_nan_explain = try filter_nan_plan.explain(gpa);
    defer gpa.free(filter_nan_explain);
    try std.testing.expect(std.mem.indexOf(u8, filter_nan_explain, "filter_nans_column(metric)") != null);
    var filtered_nan_rows = try filter_nan_plan.collect();
    defer filtered_nan_rows.deinit();
    try std.testing.expectEqual(@as(usize, 1), filtered_nan_rows.height());
    const filtered_nan_metric = try (try filtered_nan_rows.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(filtered_nan_metric);
    try std.testing.expect(std.math.isNan(filtered_nan_metric[0]));

    var drop_inf_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_inf_plan.deinit();
    try drop_inf_plan.dropInfsColumn("metric");
    const drop_inf_explain = try drop_inf_plan.explain(gpa);
    defer gpa.free(drop_inf_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_inf_explain, "drop_infs[metric]") != null);
    var dropped_inf_rows = try drop_inf_plan.collect();
    defer dropped_inf_rows.deinit();
    try std.testing.expectEqual(@as(usize, 3), dropped_inf_rows.height());
    const dropped_inf_metric = try (try dropped_inf_rows.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(dropped_inf_metric);
    try std.testing.expectEqual(@as(f64, 1.0), dropped_inf_metric[0]);
    try std.testing.expect(std.math.isNan(dropped_inf_metric[1]));
    try std.testing.expectEqual(@as(f64, 7.0), dropped_inf_metric[2]);

    var filter_inf_plan = try DeviceLazyFrame.init(gpa, table);
    defer filter_inf_plan.deinit();
    try filter_inf_plan.filterInfsColumn("metric");
    const filter_inf_explain = try filter_inf_plan.explain(gpa);
    defer gpa.free(filter_inf_explain);
    try std.testing.expect(std.mem.indexOf(u8, filter_inf_explain, "filter_infs_column(metric)") != null);
    var filtered_inf_rows = try filter_inf_plan.collect();
    defer filtered_inf_rows.deinit();
    try std.testing.expectEqual(@as(usize, 1), filtered_inf_rows.height());
    const filtered_inf_metric = try (try filtered_inf_rows.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(filtered_inf_metric);
    try std.testing.expect(std.math.isInf(filtered_inf_metric[0]));

    var drop_non_finite_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_non_finite_plan.deinit();
    try drop_non_finite_plan.dropNonFinitesColumn("metric");
    const drop_non_finite_explain = try drop_non_finite_plan.explain(gpa);
    defer gpa.free(drop_non_finite_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_non_finite_explain, "drop_non_finites[metric]") != null);
    var dropped_non_finite_rows = try drop_non_finite_plan.collect();
    defer dropped_non_finite_rows.deinit();
    try std.testing.expectEqual(@as(usize, 2), dropped_non_finite_rows.height());
    const dropped_non_finite_metric = try (try dropped_non_finite_rows.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(dropped_non_finite_metric);
    try std.testing.expectEqual(@as(f64, 1.0), dropped_non_finite_metric[0]);
    try std.testing.expectEqual(@as(f64, 7.0), dropped_non_finite_metric[1]);

    var filter_non_finite_plan = try DeviceLazyFrame.init(gpa, table);
    defer filter_non_finite_plan.deinit();
    try filter_non_finite_plan.filterNonFinitesColumn("metric");
    const filter_non_finite_explain = try filter_non_finite_plan.explain(gpa);
    defer gpa.free(filter_non_finite_explain);
    try std.testing.expect(std.mem.indexOf(u8, filter_non_finite_explain, "filter_non_finites_column(metric)") != null);
    var filtered_non_finite_rows = try filter_non_finite_plan.collect();
    defer filtered_non_finite_rows.deinit();
    try std.testing.expectEqual(@as(usize, 2), filtered_non_finite_rows.height());
    const filtered_non_finite_metric = try (try filtered_non_finite_rows.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(filtered_non_finite_metric);
    try std.testing.expect(std.math.isNan(filtered_non_finite_metric[0]));
    try std.testing.expect(std.math.isInf(filtered_non_finite_metric[1]));

    var row_special_plan = try DeviceLazyFrame.init(gpa, table);
    defer row_special_plan.deinit();
    try row_special_plan.withRowNaNCount(&.{ "metric", "id" }, "row_nan_count");
    try row_special_plan.withRowInfCount(&.{}, "row_inf_count");
    try row_special_plan.withRowFiniteCount(&.{ "metric", "id" }, "row_finite_count");
    try row_special_plan.withRowNonFiniteCount(&.{}, "row_non_finite_count");
    try row_special_plan.select(&.{ "row_nan_count", "row_inf_count", "row_finite_count", "row_non_finite_count" });
    const row_special_explain = try row_special_plan.explain(gpa);
    defer gpa.free(row_special_explain);
    try std.testing.expect(std.mem.indexOf(u8, row_special_explain, "row_nan_count([metric,id]->row_nan_count)") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_special_explain, "row_inf_count([]->row_inf_count)") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_special_explain, "row_finite_count([metric,id]->row_finite_count)") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_special_explain, "row_non_finite_count([]->row_non_finite_count)") != null);
    var row_special = try row_special_plan.collect();
    defer row_special.deinit();
    const row_nan_count = try (try row_special.column("row_nan_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_nan_count);
    const row_inf_count = try (try row_special.column("row_inf_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_inf_count);
    const row_finite_count = try (try row_special.column("row_finite_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_finite_count);
    const row_non_finite_count = try (try row_special.column("row_non_finite_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_non_finite_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 0 }, row_nan_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 0 }, row_inf_count);
    try std.testing.expectEqualSlices(i64, &.{ 2, 1, 1, 1 }, row_finite_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 1, 0 }, row_non_finite_count);

    var invalid_row_count_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_row_count_plan.deinit();
    try invalid_row_count_plan.withRowInfCount(&.{"missing"}, "bad_count");
    try std.testing.expectError(error.ColumnNotFound, invalid_row_count_plan.collect());

    var invalid_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_plan.deinit();
    try invalid_plan.isFiniteColumn("missing", "missing_is_finite");
    try std.testing.expectError(error.ColumnNotFound, invalid_plan.collect());
}

test "device lazy frame derives signed Inf predicate columns" {
    const gpa = std.testing.allocator;

    var metric = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, std.math.inf(f64), -std.math.inf(f64), std.math.nan(f64), 9.0 }, &.{ true, true, true, true, false }, .cpu);
    defer metric.deinit();
    var id = try DeviceColumn.fromSlice(i64, gpa, &.{ 10, 20, 30, 40, 50 }, .cpu);
    defer id.deinit();

    var table = try vectra.DeviceDataFrame.init(gpa, &.{
        .{ .name = "metric", .data = metric },
        .{ .name = "id", .data = id },
    });
    defer table.deinit();

    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.isPositiveInfColumn("metric", "metric_is_pos_inf");
    try plan.isNegativeInfColumn("metric", "metric_is_neg_inf");
    try plan.isPositiveInfColumn("id", "id_is_pos_inf");
    try plan.select(&.{ "metric_is_pos_inf", "metric_is_neg_inf", "id_is_pos_inf" });

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "is_positive_inf_column(metric->metric_is_pos_inf)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "is_negative_inf_column(metric->metric_is_neg_inf)") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 3), result.width());
    const metric_is_pos_inf = try (try result.column("metric_is_pos_inf")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_is_pos_inf);
    const metric_is_neg_inf = try (try result.column("metric_is_neg_inf")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_is_neg_inf);
    const id_is_pos_inf = try (try result.column("id_is_pos_inf")).bool.toOwnedSlice(gpa);
    defer gpa.free(id_is_pos_inf);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false, false }, metric_is_pos_inf);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, false }, metric_is_neg_inf);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false, false }, id_is_pos_inf);

    var invalid_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_plan.deinit();
    try invalid_plan.isPositiveInfColumn("missing", "missing_is_pos_inf");
    try std.testing.expectError(error.ColumnNotFound, invalid_plan.collect());
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

test "device lazy frame filters null rows" {
    const gpa = std.testing.allocator;
    var table = try lazyQualityTable(gpa);
    defer table.deinit();
    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.filterNullsColumn("quality");

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "filter_nulls_column(quality)") != null);

    var filtered = try plan.collect();
    defer filtered.deinit();
    try std.testing.expectEqual(@as(usize, 1), filtered.height());
    const quality = try (try filtered.column("quality")).f64.toOwnedSlice(gpa);
    defer gpa.free(quality);
    const validity = try (try filtered.column("quality")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(validity);
    try std.testing.expectEqualSlices(f64, &.{2.0}, quality);
    try std.testing.expectEqualSlices(bool, &.{false}, validity);

    var invalid_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_plan.deinit();
    try invalid_plan.filterNullsColumn("missing");
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

    var rename_many_plan = try DeviceLazyFrame.init(gpa, table);
    defer rename_many_plan.deinit();
    try rename_many_plan.renameColumns(&.{ "sales", "units" }, &.{ "revenue", "quantity" });
    const rename_many_explained = try rename_many_plan.explain(gpa);
    defer gpa.free(rename_many_explained);
    try std.testing.expect(std.mem.indexOf(u8, rename_many_explained, "rename_columns[sales->revenue,units->quantity]") != null);
    var renamed_many = try rename_many_plan.collect();
    defer renamed_many.deinit();
    try std.testing.expectEqual(@as(?usize, 0), renamed_many.columnIndex("revenue"));
    try std.testing.expectEqual(@as(?usize, 1), renamed_many.columnIndex("quantity"));
    try std.testing.expectEqual(@as(?usize, 2), renamed_many.columnIndex("active"));

    var prefix_plan = try DeviceLazyFrame.init(gpa, table);
    defer prefix_plan.deinit();
    try prefix_plan.addColumnNamePrefix("src_");
    const prefix_explained = try prefix_plan.explain(gpa);
    defer gpa.free(prefix_explained);
    try std.testing.expect(std.mem.indexOf(u8, prefix_explained, "add_column_name_prefix(src_)") != null);
    var prefixed = try prefix_plan.collect();
    defer prefixed.deinit();
    try std.testing.expectEqual(@as(?usize, 0), prefixed.columnIndex("src_sales"));
    try std.testing.expectEqual(@as(?usize, 1), prefixed.columnIndex("src_units"));
    try std.testing.expectEqual(@as(?usize, 2), prefixed.columnIndex("src_active"));

    var suffix_plan = try DeviceLazyFrame.init(gpa, table);
    defer suffix_plan.deinit();
    try suffix_plan.addColumnNameSuffix("_raw");
    const suffix_explained = try suffix_plan.explain(gpa);
    defer gpa.free(suffix_explained);
    try std.testing.expect(std.mem.indexOf(u8, suffix_explained, "add_column_name_suffix(_raw)") != null);
    var suffixed = try suffix_plan.collect();
    defer suffixed.deinit();
    try std.testing.expectEqual(@as(?usize, 0), suffixed.columnIndex("sales_raw"));
    try std.testing.expectEqual(@as(?usize, 1), suffixed.columnIndex("units_raw"));
    try std.testing.expectEqual(@as(?usize, 2), suffixed.columnIndex("active_raw"));

    var invalid_many_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_many_plan.deinit();
    try invalid_many_plan.renameColumns(&.{ "sales", "units" }, &.{ "revenue", "revenue" });
    try std.testing.expectError(error.InvalidShape, invalid_many_plan.collect());

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

test "device lazy frame places literal columns" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();

    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.withColumnLiteralAt("segment", i32, 42, 0);
    try plan.withColumnLiteralBefore("rank", i16, 5, "units");
    try plan.withColumnLiteralAfter("score", f32, 1.5, "active");

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_literal_at(segment=scalar:i32, index=0)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_literal_before(rank=scalar:i16 before units)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_literal_after(score=scalar:f32 after active)") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 6), result.width());
    try std.testing.expectEqual(@as(?usize, 0), result.columnIndex("segment"));
    try std.testing.expectEqual(@as(?usize, 1), result.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 2), result.columnIndex("rank"));
    try std.testing.expectEqual(@as(?usize, 3), result.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, 4), result.columnIndex("active"));
    try std.testing.expectEqual(@as(?usize, 5), result.columnIndex("score"));
    const segment = try (try result.column("segment")).i32.toOwnedSlice(gpa);
    defer gpa.free(segment);
    const score = try (try result.column("score")).f32.toOwnedSlice(gpa);
    defer gpa.free(score);
    try std.testing.expectEqualSlices(i32, &.{ 42, 42, 42, 42 }, segment);
    try std.testing.expectEqualSlices(f32, &.{ 1.5, 1.5, 1.5, 1.5 }, score);

    var replace_plan = try DeviceLazyFrame.init(gpa, table);
    defer replace_plan.deinit();
    try replace_plan.withColumnLiteralAt("sales", f64, 9.0, 2);
    var replaced = try replace_plan.collect();
    defer replaced.deinit();
    try std.testing.expectEqual(@as(usize, 3), replaced.width());
    try std.testing.expectEqual(@as(?usize, 0), replaced.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, 1), replaced.columnIndex("active"));
    try std.testing.expectEqual(@as(?usize, 2), replaced.columnIndex("sales"));
    const replaced_sales = try (try replaced.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(replaced_sales);
    try std.testing.expectEqualSlices(f64, &.{ 9.0, 9.0, 9.0, 9.0 }, replaced_sales);

    var missing_anchor_plan = try DeviceLazyFrame.init(gpa, table);
    defer missing_anchor_plan.deinit();
    try missing_anchor_plan.withColumnLiteralBefore("bad", i8, 1, "missing");
    try std.testing.expectError(error.ColumnNotFound, missing_anchor_plan.collect());

    var bounds_plan = try DeviceLazyFrame.init(gpa, table);
    defer bounds_plan.deinit();
    try bounds_plan.withColumnLiteralAt("bad", i8, 1, table.width() + 1);
    try std.testing.expectError(error.IndexOutOfBounds, bounds_plan.collect());
}

test "device lazy frame moves columns" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();

    var move_plan = try DeviceLazyFrame.init(gpa, table);
    defer move_plan.deinit();
    try move_plan.moveColumn("active", 0);

    const move_explain = try move_plan.explain(gpa);
    defer gpa.free(move_explain);
    try std.testing.expect(std.mem.indexOf(u8, move_explain, "move_column(active -> index=0)") != null);

    var moved = try move_plan.collect();
    defer moved.deinit();
    try std.testing.expectEqual(@as(usize, 3), moved.width());
    try std.testing.expectEqual(@as(?usize, 0), moved.columnIndex("active"));
    try std.testing.expectEqual(@as(?usize, 1), moved.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 2), moved.columnIndex("units"));
    const moved_active = try (try moved.column("active")).bool.toOwnedSlice(gpa);
    defer gpa.free(moved_active);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, true }, moved_active);

    var before_plan = try DeviceLazyFrame.init(gpa, table);
    defer before_plan.deinit();
    try before_plan.moveColumnBefore("units", "sales");
    const before_explain = try before_plan.explain(gpa);
    defer gpa.free(before_explain);
    try std.testing.expect(std.mem.indexOf(u8, before_explain, "move_column_before(units before sales)") != null);

    var before = try before_plan.collect();
    defer before.deinit();
    try std.testing.expectEqual(@as(?usize, 0), before.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, 1), before.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 2), before.columnIndex("active"));

    var after_plan = try DeviceLazyFrame.init(gpa, table);
    defer after_plan.deinit();
    try after_plan.moveColumnAfter("sales", "active");
    const after_explain = try after_plan.explain(gpa);
    defer gpa.free(after_explain);
    try std.testing.expect(std.mem.indexOf(u8, after_explain, "move_column_after(sales after active)") != null);

    var after = try after_plan.collect();
    defer after.deinit();
    try std.testing.expectEqual(@as(?usize, 0), after.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, 1), after.columnIndex("active"));
    try std.testing.expectEqual(@as(?usize, 2), after.columnIndex("sales"));
    const after_sales = try (try after.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(after_sales);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 3.0, 5.0, 7.0 }, after_sales);

    var missing_plan = try DeviceLazyFrame.init(gpa, table);
    defer missing_plan.deinit();
    try missing_plan.moveColumn("missing", 0);
    try std.testing.expectError(error.ColumnNotFound, missing_plan.collect());

    var bounds_plan = try DeviceLazyFrame.init(gpa, table);
    defer bounds_plan.deinit();
    try bounds_plan.moveColumn("sales", table.width());
    try std.testing.expectError(error.IndexOutOfBounds, bounds_plan.collect());
}

test "device lazy frame copies columns" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();

    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.copyColumn("sales", "sales_copy");
    try plan.copyColumnBefore("active", "active_copy", "units");
    try plan.copyColumnAfter("units", "units_after", "active");

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "copy_column(sales->sales_copy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "copy_column_before(active->active_copy before units)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "copy_column_after(units->units_after after active)") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 6), result.width());
    try std.testing.expectEqual(@as(?usize, 0), result.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), result.columnIndex("active_copy"));
    try std.testing.expectEqual(@as(?usize, 2), result.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, 3), result.columnIndex("active"));
    try std.testing.expectEqual(@as(?usize, 4), result.columnIndex("units_after"));
    try std.testing.expectEqual(@as(?usize, 5), result.columnIndex("sales_copy"));
    const copied_sales = try (try result.column("sales_copy")).f64.toOwnedSlice(gpa);
    defer gpa.free(copied_sales);
    const copied_active = try (try result.column("active_copy")).bool.toOwnedSlice(gpa);
    defer gpa.free(copied_active);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 3.0, 5.0, 7.0 }, copied_sales);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, true }, copied_active);

    var at_plan = try DeviceLazyFrame.init(gpa, table);
    defer at_plan.deinit();
    try at_plan.copyColumnAt("units", "units_first", 0);
    const at_explained = try at_plan.explain(gpa);
    defer gpa.free(at_explained);
    try std.testing.expect(std.mem.indexOf(u8, at_explained, "copy_column_at(units->units_first, index=0)") != null);

    var at_result = try at_plan.collect();
    defer at_result.deinit();
    try std.testing.expectEqual(@as(?usize, 0), at_result.columnIndex("units_first"));
    try std.testing.expectEqual(@as(?usize, 1), at_result.columnIndex("sales"));
    const units_first = try (try at_result.column("units_first")).i64.toOwnedSlice(gpa);
    defer gpa.free(units_first);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4 }, units_first);

    var missing_plan = try DeviceLazyFrame.init(gpa, table);
    defer missing_plan.deinit();
    try missing_plan.copyColumn("missing", "copy");
    try std.testing.expectError(error.ColumnNotFound, missing_plan.collect());

    var bounds_plan = try DeviceLazyFrame.init(gpa, table);
    defer bounds_plan.deinit();
    try bounds_plan.copyColumnAt("sales", "copy", table.width() + 1);
    try std.testing.expectError(error.IndexOutOfBounds, bounds_plan.collect());
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

    var drop_rows_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_rows_plan.deinit();
    try drop_rows_plan.dropRows(&.{ 1, 1 });
    try drop_rows_plan.select(&.{ "sales", "units" });
    const drop_rows_explain = try drop_rows_plan.explain(gpa);
    defer gpa.free(drop_rows_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_rows_explain, "drop_rows([1,1])") != null);
    var rows_dropped = try drop_rows_plan.collect();
    defer rows_dropped.deinit();
    try std.testing.expectEqual(@as(usize, 3), rows_dropped.height());
    try std.testing.expectEqual(@as(usize, 2), rows_dropped.width());
    const rows_dropped_sales = try (try rows_dropped.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(rows_dropped_sales);
    const rows_dropped_units = try (try rows_dropped.column("units")).i64.toOwnedSlice(gpa);
    defer gpa.free(rows_dropped_units);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 5.0, 7.0 }, rows_dropped_sales);
    try std.testing.expectEqualSlices(i64, &.{ 1, 3, 4 }, rows_dropped_units);

    var invalid_drop_rows_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_drop_rows_plan.deinit();
    try invalid_drop_rows_plan.dropRows(&.{table.height()});
    try std.testing.expectError(error.IndexOutOfBounds, invalid_drop_rows_plan.collect());

    var drop_range_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_range_plan.deinit();
    try drop_range_plan.dropRowRange(1, 3);
    try drop_range_plan.select(&.{ "sales", "units" });
    const drop_range_explain = try drop_range_plan.explain(gpa);
    defer gpa.free(drop_range_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_range_explain, "drop_row_range(1..3)") != null);
    var range_dropped = try drop_range_plan.collect();
    defer range_dropped.deinit();
    const range_dropped_sales = try (try range_dropped.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(range_dropped_sales);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 7.0 }, range_dropped_sales);

    var drop_first_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_first_plan.deinit();
    try drop_first_plan.dropFirstRows(2);
    try drop_first_plan.select(&.{"sales"});
    const drop_first_explain = try drop_first_plan.explain(gpa);
    defer gpa.free(drop_first_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_first_explain, "drop_row_range(0..2)") != null);
    var first_dropped = try drop_first_plan.collect();
    defer first_dropped.deinit();
    const first_dropped_sales = try (try first_dropped.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(first_dropped_sales);
    try std.testing.expectEqualSlices(f64, &.{ 5.0, 7.0 }, first_dropped_sales);

    var drop_last_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_last_plan.deinit();
    try drop_last_plan.dropLastRows(1);
    try drop_last_plan.select(&.{"sales"});
    const drop_last_explain = try drop_last_plan.explain(gpa);
    defer gpa.free(drop_last_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_last_explain, "drop_last_rows(1)") != null);
    var last_dropped = try drop_last_plan.collect();
    defer last_dropped.deinit();
    const last_dropped_sales = try (try last_dropped.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(last_dropped_sales);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 3.0, 5.0 }, last_dropped_sales);

    var step_plan = try DeviceLazyFrame.init(gpa, table);
    defer step_plan.deinit();
    try step_plan.sliceRowsStep(0, table.height(), 2);
    try step_plan.select(&.{ "sales", "units" });
    const step_explain = try step_plan.explain(gpa);
    defer gpa.free(step_explain);
    try std.testing.expect(std.mem.indexOf(u8, step_explain, "slice_rows_step(0..4, step=2)") != null);
    var stepped = try step_plan.collect();
    defer stepped.deinit();
    try std.testing.expectEqual(@as(usize, 2), stepped.height());
    try std.testing.expectEqual(@as(usize, 2), stepped.width());
    const stepped_sales = try (try stepped.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(stepped_sales);
    const stepped_units = try (try stepped.column("units")).i64.toOwnedSlice(gpa);
    defer gpa.free(stepped_units);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 5.0 }, stepped_sales);
    try std.testing.expectEqualSlices(i64, &.{ 1, 3 }, stepped_units);

    var invalid_step_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_step_plan.deinit();
    try invalid_step_plan.sliceRowsStep(0, table.height(), 0);
    try std.testing.expectError(error.InvalidShape, invalid_step_plan.collect());

    var stride_plan = try DeviceLazyFrame.init(gpa, table);
    defer stride_plan.deinit();
    try stride_plan.strideRows(0, 2);
    try stride_plan.select(&.{ "sales", "units" });
    const stride_explain = try stride_plan.explain(gpa);
    defer gpa.free(stride_explain);
    try std.testing.expect(std.mem.indexOf(u8, stride_explain, "stride_rows(start=0, step=2)") != null);
    var strided = try stride_plan.collect();
    defer strided.deinit();
    try std.testing.expectEqual(@as(usize, 2), strided.height());
    try std.testing.expectEqual(@as(usize, 2), strided.width());
    const strided_sales = try (try strided.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(strided_sales);
    const strided_units = try (try strided.column("units")).i64.toOwnedSlice(gpa);
    defer gpa.free(strided_units);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 5.0 }, strided_sales);
    try std.testing.expectEqualSlices(i64, &.{ 1, 3 }, strided_units);

    var empty_stride_plan = try DeviceLazyFrame.init(gpa, table);
    defer empty_stride_plan.deinit();
    try empty_stride_plan.strideRows(table.height(), 1);
    var empty_stride = try empty_stride_plan.collect();
    defer empty_stride.deinit();
    try std.testing.expectEqual(@as(usize, 0), empty_stride.height());
    try std.testing.expectEqual(table.width(), empty_stride.width());

    var invalid_stride_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_stride_plan.deinit();
    try invalid_stride_plan.strideRows(0, 0);
    try std.testing.expectError(error.InvalidShape, invalid_stride_plan.collect());

    var sample_plan = try DeviceLazyFrame.init(gpa, table);
    defer sample_plan.deinit();
    try sample_plan.sampleRows(2, 1234);
    try sample_plan.select(&.{ "sales", "units" });
    const sample_explain = try sample_plan.explain(gpa);
    defer gpa.free(sample_explain);
    try std.testing.expect(std.mem.indexOf(u8, sample_explain, "sample_rows(count=2, seed=1234)") != null);
    var sampled = try sample_plan.collect();
    defer sampled.deinit();
    try std.testing.expectEqual(@as(usize, 2), sampled.height());
    try std.testing.expectEqual(@as(usize, 2), sampled.width());

    var sample_again_plan = try DeviceLazyFrame.init(gpa, table);
    defer sample_again_plan.deinit();
    try sample_again_plan.sampleRows(2, 1234);
    try sample_again_plan.select(&.{ "sales", "units" });
    var sampled_again = try sample_again_plan.collect();
    defer sampled_again.deinit();
    const sampled_sales = try (try sampled.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(sampled_sales);
    const sampled_again_sales = try (try sampled_again.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(sampled_again_sales);
    try std.testing.expectEqualSlices(f64, sampled_sales, sampled_again_sales);

    var invalid_sample_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_sample_plan.deinit();
    try invalid_sample_plan.sampleRows(table.height() + 1, 1234);
    try std.testing.expectError(error.InvalidShape, invalid_sample_plan.collect());

    var replacement_plan = try DeviceLazyFrame.init(gpa, table);
    defer replacement_plan.deinit();
    try replacement_plan.sampleRowsWithReplacement(table.height() + 2, 4321);
    try replacement_plan.select(&.{ "sales", "units" });
    const replacement_explain = try replacement_plan.explain(gpa);
    defer gpa.free(replacement_explain);
    try std.testing.expect(std.mem.indexOf(u8, replacement_explain, "sample_rows_with_replacement(count=6, seed=4321)") != null);
    var sampled_replacement = try replacement_plan.collect();
    defer sampled_replacement.deinit();
    try std.testing.expectEqual(@as(usize, 6), sampled_replacement.height());
    try std.testing.expectEqual(@as(usize, 2), sampled_replacement.width());

    var replacement_again_plan = try DeviceLazyFrame.init(gpa, table);
    defer replacement_again_plan.deinit();
    try replacement_again_plan.sampleRowsWithReplacement(table.height() + 2, 4321);
    try replacement_again_plan.select(&.{ "sales", "units" });
    var sampled_replacement_again = try replacement_again_plan.collect();
    defer sampled_replacement_again.deinit();
    const replacement_sales = try (try sampled_replacement.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(replacement_sales);
    const replacement_again_sales = try (try sampled_replacement_again.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(replacement_again_sales);
    try std.testing.expectEqualSlices(f64, replacement_sales, replacement_again_sales);

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
