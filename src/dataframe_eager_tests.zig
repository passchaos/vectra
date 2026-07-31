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

test "dataframe select filter groupby and csv" {
    const gpa = std.testing.allocator;
    var df = try DataFrame.init(gpa, &.{
        .{ .name = "city", .data = .{ .string = &.{ "hz", "bj", "hz" } } },
        .{ .name = "sales", .data = .{ .f64 = &.{ 2.0, 3.0, 5.0 } } },
        .{ .name = "units", .data = .{ .i64 = &.{ 1, 2, 3 } } },
    });
    defer df.deinit();
    try std.testing.expectEqual(@as(usize, 3), df.height());
    var filtered = try df.filter(&.{ true, false, true });
    defer filtered.deinit();
    try std.testing.expectEqual(@as(usize, 2), filtered.height());
    var grouped = try df.groupBySum("city", "sales");
    defer grouped.deinit();
    try std.testing.expectEqual(@as(usize, 2), grouped.height());
    var desc = try df.describe();
    defer desc.deinit();
    try std.testing.expectEqual(@as(usize, 4), desc.height());
    const csv = try df.writeCsv(gpa);
    defer gpa.free(csv);
    var parsed = try DataFrame.readCsv(gpa, csv, true);
    defer parsed.deinit();
    try std.testing.expectEqual(df.height(), parsed.height());
}

test "device dataframe owns fixed-width columns on a shared device" {
    const gpa = std.testing.allocator;

    var sales = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 3.0, 5.0 }, .cpu);
    defer sales.deinit();
    var units = try DeviceColumn.fromSliceWithValidity(i64, gpa, &.{ 1, 2, 3 }, &.{ true, false, true }, .cpu);
    defer units.deinit();
    var active = try DeviceColumn.fromSlice(bool, gpa, &.{ true, false, true }, .cpu);
    defer active.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "sales", .data = sales },
        .{ .name = "units", .data = units },
        .{ .name = "active", .data = active },
    });
    defer table.deinit();

    try std.testing.expectEqual(@as(usize, 3), table.height());
    try std.testing.expectEqual(@as(usize, 3), table.width());
    try std.testing.expect(table.device.isCpu());
    try std.testing.expectEqual(DeviceDType.i64, try table.columnDType("units"));

    const units_col = try table.column("units");
    try std.testing.expect(units_col.nullable());
    try std.testing.expect(units_col.hasNulls());
    try std.testing.expectEqual(@as(usize, 1), units_col.nullCount());

    var view = try table.view();
    defer view.deinit();
    try std.testing.expectEqual(@as(usize, 3), view.height());
    try std.testing.expectEqual(DeviceDType.f64, view.columns[0].dtype);
    try std.testing.expectEqual(DeviceValidityEncoding.bool_mask, view.columns[1].validity_encoding);
    try std.testing.expect(view.columns[0].data_ptr != 0);

    var selected = try table.select(&.{"sales"});
    defer selected.deinit();
    try std.testing.expectEqual(@as(usize, 1), selected.width());
    try std.testing.expectEqual(DeviceDType.f64, try selected.columnDType("sales"));

    var positional_selected = try table.selectByColumnIndices(&.{ 2, 0 });
    defer positional_selected.deinit();
    try std.testing.expectEqual(@as(usize, 2), positional_selected.width());
    try std.testing.expectEqual(@as(?usize, 0), positional_selected.columnIndex("active"));
    try std.testing.expectEqual(@as(?usize, 1), positional_selected.columnIndex("sales"));

    var range_selected = try table.selectColumnRange(1, 3);
    defer range_selected.deinit();
    try std.testing.expectEqual(@as(usize, 2), range_selected.width());
    try std.testing.expectEqual(@as(?usize, 0), range_selected.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, 1), range_selected.columnIndex("active"));

    var first_two = try table.selectFirstColumns(2);
    defer first_two.deinit();
    try std.testing.expectEqual(@as(usize, 2), first_two.width());
    try std.testing.expectEqual(@as(?usize, 0), first_two.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), first_two.columnIndex("units"));

    var last_two = try table.selectLastColumns(2);
    defer last_two.deinit();
    try std.testing.expectEqual(@as(usize, 2), last_two.width());
    try std.testing.expectEqual(@as(?usize, 0), last_two.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, 1), last_two.columnIndex("active"));

    var positional_dropped = try table.dropByColumnIndices(&.{1});
    defer positional_dropped.deinit();
    try std.testing.expectEqual(@as(usize, 2), positional_dropped.width());
    try std.testing.expectEqual(@as(?usize, 0), positional_dropped.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), positional_dropped.columnIndex("active"));

    var range_dropped = try table.dropColumnRange(1, 3);
    defer range_dropped.deinit();
    try std.testing.expectEqual(@as(usize, 1), range_dropped.width());
    try std.testing.expectEqual(@as(?usize, 0), range_dropped.columnIndex("sales"));

    var drop_first = try table.dropFirstColumns(1);
    defer drop_first.deinit();
    try std.testing.expectEqual(@as(usize, 2), drop_first.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_first.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, 1), drop_first.columnIndex("active"));

    var drop_last = try table.dropLastColumns(1);
    defer drop_last.deinit();
    try std.testing.expectEqual(@as(usize, 2), drop_last.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_last.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), drop_last.columnIndex("units"));

    var reversed_columns = try table.reverseColumns();
    defer reversed_columns.deinit();
    try std.testing.expectEqual(@as(?usize, 0), reversed_columns.columnIndex("active"));
    try std.testing.expectEqual(@as(?usize, 1), reversed_columns.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, 2), reversed_columns.columnIndex("sales"));
    const reversed_columns_units_validity = try (try reversed_columns.column("units")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(reversed_columns_units_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, reversed_columns_units_validity);

    var columns_sorted = try table.sortColumnsByName(false);
    defer columns_sorted.deinit();
    try std.testing.expectEqual(@as(?usize, 0), columns_sorted.columnIndex("active"));
    try std.testing.expectEqual(@as(?usize, 1), columns_sorted.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 2), columns_sorted.columnIndex("units"));

    var columns_sorted_desc = try table.sortColumnsByName(true);
    defer columns_sorted_desc.deinit();
    try std.testing.expectEqual(@as(?usize, 0), columns_sorted_desc.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, 1), columns_sorted_desc.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 2), columns_sorted_desc.columnIndex("active"));
    try std.testing.expectError(error.IndexOutOfBounds, table.selectByColumnIndices(&.{3}));
    try std.testing.expectError(error.IndexOutOfBounds, table.dropByColumnIndices(&.{3}));

    var numeric_selected = try table.selectNumeric();
    defer numeric_selected.deinit();
    try std.testing.expectEqual(@as(usize, 2), numeric_selected.width());
    try std.testing.expectEqual(@as(?usize, 0), numeric_selected.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), numeric_selected.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, null), numeric_selected.columnIndex("active"));

    var float_selected = try table.selectFloat();
    defer float_selected.deinit();
    try std.testing.expectEqual(@as(usize, 1), float_selected.width());
    try std.testing.expectEqual(DeviceDType.f64, try float_selected.columnDType("sales"));

    var bool_selected = try table.selectBool();
    defer bool_selected.deinit();
    try std.testing.expectEqual(@as(usize, 1), bool_selected.width());
    try std.testing.expectEqual(DeviceDType.bool, try bool_selected.columnDType("active"));

    var exact_selected = try table.selectByDTypes(&.{ .i64, .bool });
    defer exact_selected.deinit();
    try std.testing.expectEqual(@as(usize, 2), exact_selected.width());
    try std.testing.expectEqual(@as(?usize, 0), exact_selected.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, 1), exact_selected.columnIndex("active"));

    var empty_dtype_selected = try table.selectByDTypes(&.{.c64});
    defer empty_dtype_selected.deinit();
    try std.testing.expectEqual(@as(usize, 0), empty_dtype_selected.width());
    try std.testing.expectEqual(table.height(), empty_dtype_selected.height());

    var numeric_dropped = try table.dropNumeric();
    defer numeric_dropped.deinit();
    try std.testing.expectEqual(@as(usize, 1), numeric_dropped.width());
    try std.testing.expectEqual(@as(?usize, 0), numeric_dropped.columnIndex("active"));
    try std.testing.expectEqual(@as(?usize, null), numeric_dropped.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, null), numeric_dropped.columnIndex("units"));

    var float_dropped = try table.dropFloat();
    defer float_dropped.deinit();
    try std.testing.expectEqual(@as(usize, 2), float_dropped.width());
    try std.testing.expectEqual(@as(?usize, 0), float_dropped.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, 1), float_dropped.columnIndex("active"));
    try std.testing.expectEqual(@as(?usize, null), float_dropped.columnIndex("sales"));

    var exact_dropped = try table.dropByDTypes(&.{ .i64, .bool });
    defer exact_dropped.deinit();
    try std.testing.expectEqual(@as(usize, 1), exact_dropped.width());
    try std.testing.expectEqual(DeviceDType.f64, try exact_dropped.columnDType("sales"));

    var no_dtype_dropped = try table.dropByDTypes(&.{.c64});
    defer no_dtype_dropped.deinit();
    try std.testing.expectEqual(table.width(), no_dtype_dropped.width());
    try std.testing.expectEqual(@as(?usize, 0), no_dtype_dropped.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 2), no_dtype_dropped.columnIndex("active"));

    var all_dropped = try table.dropByDTypes(&.{ .f64, .i64, .bool });
    defer all_dropped.deinit();
    try std.testing.expectEqual(@as(usize, 0), all_dropped.width());
    try std.testing.expectEqual(table.height(), all_dropped.height());

    var literalized = try table.withColumnLiteral("region_id", i32, 7);
    defer literalized.deinit();
    try std.testing.expectEqual(@as(usize, 4), literalized.width());
    try std.testing.expectEqual(DeviceDType.i32, try literalized.columnDType("region_id"));
    const region_id = try (try literalized.column("region_id")).i32.toOwnedSlice(gpa);
    defer gpa.free(region_id);
    try std.testing.expectEqualSlices(i32, &.{ 7, 7, 7 }, region_id);

    var literal_bool = try table.withColumnLiteral("literal_active", bool, true);
    defer literal_bool.deinit();
    const literal_active = try (try literal_bool.column("literal_active")).bool.toOwnedSlice(gpa);
    defer gpa.free(literal_active);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true }, literal_active);

    var replaced_sales = try table.withColumnLiteral("sales", f64, 1.0);
    defer replaced_sales.deinit();
    try std.testing.expectEqual(@as(usize, 3), replaced_sales.width());
    const replaced_sales_values = try (try replaced_sales.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(replaced_sales_values);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 1.0, 1.0 }, replaced_sales_values);

    var discount_col = try DeviceColumn.fromSlice(f64, gpa, &.{ 0.1, 0.2, 0.3 }, .cpu);
    defer discount_col.deinit();
    var inserted_discount = try table.withColumnAt("discount", discount_col, 1);
    defer inserted_discount.deinit();
    try std.testing.expectEqual(@as(usize, 4), inserted_discount.width());
    try std.testing.expectEqual(@as(?usize, 0), inserted_discount.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), inserted_discount.columnIndex("discount"));
    try std.testing.expectEqual(@as(?usize, 2), inserted_discount.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, 3), inserted_discount.columnIndex("active"));

    var segment_first = try table.withColumnLiteralAt("segment", i32, 42, 0);
    defer segment_first.deinit();
    try std.testing.expectEqual(@as(?usize, 0), segment_first.columnIndex("segment"));
    try std.testing.expectEqual(@as(?usize, 1), segment_first.columnIndex("sales"));
    const segment_values = try (try segment_first.column("segment")).i32.toOwnedSlice(gpa);
    defer gpa.free(segment_values);
    try std.testing.expectEqualSlices(i32, &.{ 42, 42, 42 }, segment_values);

    var rank_before_units = try table.withColumnLiteralBefore("rank", i16, 5, "units");
    defer rank_before_units.deinit();
    try std.testing.expectEqual(@as(?usize, 0), rank_before_units.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), rank_before_units.columnIndex("rank"));
    try std.testing.expectEqual(@as(?usize, 2), rank_before_units.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, 3), rank_before_units.columnIndex("active"));

    var score_after_units = try table.withColumnLiteralAfter("score", f32, 1.5, "units");
    defer score_after_units.deinit();
    try std.testing.expectEqual(@as(?usize, 0), score_after_units.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), score_after_units.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, 2), score_after_units.columnIndex("score"));
    try std.testing.expectEqual(@as(?usize, 3), score_after_units.columnIndex("active"));

    var repositioned_sales = try table.withColumnLiteralAt("sales", f64, 9.0, 2);
    defer repositioned_sales.deinit();
    try std.testing.expectEqual(@as(usize, 3), repositioned_sales.width());
    try std.testing.expectEqual(@as(?usize, 0), repositioned_sales.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, 1), repositioned_sales.columnIndex("active"));
    try std.testing.expectEqual(@as(?usize, 2), repositioned_sales.columnIndex("sales"));
    const repositioned_sales_values = try (try repositioned_sales.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(repositioned_sales_values);
    try std.testing.expectEqualSlices(f64, &.{ 9.0, 9.0, 9.0 }, repositioned_sales_values);
    try std.testing.expectError(error.IndexOutOfBounds, table.withColumnLiteralAt("bad", i8, 1, table.width() + 1));
    try std.testing.expectError(error.ColumnNotFound, table.withColumnLiteralBefore("bad", i8, 1, "missing"));

    var copied_sales = try table.copyColumn("sales", "sales_copy");
    defer copied_sales.deinit();
    try std.testing.expectEqual(@as(usize, 4), copied_sales.width());
    try std.testing.expectEqual(@as(?usize, 3), copied_sales.columnIndex("sales_copy"));
    const copied_sales_values = try (try copied_sales.column("sales_copy")).f64.toOwnedSlice(gpa);
    defer gpa.free(copied_sales_values);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 3.0, 5.0 }, copied_sales_values);

    var copied_units_first = try table.copyColumnAt("units", "units_copy", 0);
    defer copied_units_first.deinit();
    try std.testing.expectEqual(@as(?usize, 0), copied_units_first.columnIndex("units_copy"));
    try std.testing.expectEqual(@as(?usize, 1), copied_units_first.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 2), copied_units_first.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, 3), copied_units_first.columnIndex("active"));
    const copied_units = try (try copied_units_first.column("units_copy")).i64.toOwnedSlice(gpa);
    defer gpa.free(copied_units);
    const copied_units_validity = try (try copied_units_first.column("units_copy")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(copied_units_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3 }, copied_units);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, copied_units_validity);

    var copied_active_before_units = try table.copyColumnBefore("active", "active_copy", "units");
    defer copied_active_before_units.deinit();
    try std.testing.expectEqual(@as(?usize, 0), copied_active_before_units.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), copied_active_before_units.columnIndex("active_copy"));
    try std.testing.expectEqual(@as(?usize, 2), copied_active_before_units.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, 3), copied_active_before_units.columnIndex("active"));

    var copied_sales_after_active = try table.copyColumnAfter("sales", "sales_after", "active");
    defer copied_sales_after_active.deinit();
    try std.testing.expectEqual(@as(?usize, 3), copied_sales_after_active.columnIndex("sales_after"));
    try std.testing.expectError(error.ColumnNotFound, table.copyColumn("missing", "copy"));
    try std.testing.expectError(error.ColumnNotFound, table.copyColumnBefore("sales", "copy", "missing"));
    try std.testing.expectError(error.IndexOutOfBounds, table.copyColumnAt("sales", "copy", table.width() + 1));

    var cast_units = try table.castColumn("units", .f64);
    defer cast_units.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try cast_units.columnDType("units"));
    const cast_units_values = try (try cast_units.column("units")).f64.toOwnedSlice(gpa);
    defer gpa.free(cast_units_values);
    const cast_units_validity = try (try cast_units.column("units")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(cast_units_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 2.0, 3.0 }, cast_units_values);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, cast_units_validity);

    var filled_units = try table.fillNullColumn("units", i64, 99);
    defer filled_units.deinit();
    try std.testing.expectEqual(DeviceDType.i64, try filled_units.columnDType("units"));
    try std.testing.expectEqual(@as(usize, 0), (try filled_units.column("units")).nullCount());
    const filled_units_values = try (try filled_units.column("units")).i64.toOwnedSlice(gpa);
    defer gpa.free(filled_units_values);
    try std.testing.expectEqualSlices(i64, &.{ 1, 99, 3 }, filled_units_values);
    try std.testing.expectError(error.TypeUnsupported, table.fillNullColumn("units", f64, 0.0));
    try std.testing.expectError(error.ColumnNotFound, table.fillNullColumn("missing", i64, 0));

    var fallback_units_col = try DeviceColumn.fromSliceWithValidity(i64, gpa, &.{ 10, 20, 30 }, &.{ true, true, false }, .cpu);
    defer fallback_units_col.deinit();
    var fallback_table = try table.withColumn("fallback_units", fallback_units_col);
    defer fallback_table.deinit();
    var coalesced_units = try fallback_table.coalesceColumns("units", "fallback_units", "units_coalesced");
    defer coalesced_units.deinit();
    try std.testing.expectEqual(DeviceDType.i64, try coalesced_units.columnDType("units_coalesced"));
    try std.testing.expectEqual(@as(usize, 0), (try coalesced_units.column("units_coalesced")).nullCount());
    const coalesced_values = try (try coalesced_units.column("units_coalesced")).i64.toOwnedSlice(gpa);
    defer gpa.free(coalesced_values);
    try std.testing.expectEqualSlices(i64, &.{ 1, 20, 3 }, coalesced_values);
    try std.testing.expectError(error.TypeMismatch, fallback_table.coalesceColumns("units", "sales", "bad"));
    try std.testing.expectError(error.ColumnNotFound, fallback_table.coalesceColumns("missing", "fallback_units", "bad"));

    var null_flags = try table.isNullColumn("units", "units_is_null");
    defer null_flags.deinit();
    try std.testing.expectEqual(DeviceDType.bool, try null_flags.columnDType("units_is_null"));
    const units_is_null = try (try null_flags.column("units_is_null")).bool.toOwnedSlice(gpa);
    defer gpa.free(units_is_null);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false }, units_is_null);

    var valid_flags = try table.isValidColumn("units", "units_is_valid");
    defer valid_flags.deinit();
    const units_is_valid = try (try valid_flags.column("units_is_valid")).bool.toOwnedSlice(gpa);
    defer gpa.free(units_is_valid);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, units_is_valid);

    var nonnull_flags = try table.isNullColumn("sales", "sales_is_null");
    defer nonnull_flags.deinit();
    const sales_is_null = try (try nonnull_flags.column("sales_is_null")).bool.toOwnedSlice(gpa);
    defer gpa.free(sales_is_null);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false }, sales_is_null);
    try std.testing.expectError(error.ColumnNotFound, table.isNullColumn("missing", "missing_is_null"));

    var row_null_counts = try table.withRowNullCount(&.{}, "row_null_count");
    defer row_null_counts.deinit();
    try std.testing.expectEqual(DeviceDType.i64, try row_null_counts.columnDType("row_null_count"));
    const row_null_count = try (try row_null_counts.column("row_null_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_null_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0 }, row_null_count);

    var row_valid_counts = try table.withRowValidCount(&.{ "sales", "units", "active" }, "row_valid_count");
    defer row_valid_counts.deinit();
    const row_valid_count = try (try row_valid_counts.column("row_valid_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_valid_count);
    try std.testing.expectEqualSlices(i64, &.{ 3, 2, 3 }, row_valid_count);
    try std.testing.expectError(error.ColumnNotFound, table.withRowNullCount(&.{"missing"}, "bad_count"));

    var dropped_nulls = try table.dropNullsColumn("units");
    defer dropped_nulls.deinit();
    try std.testing.expectEqual(@as(usize, 2), dropped_nulls.height());
    const dropped_nulls_units = try (try dropped_nulls.column("units")).i64.toOwnedSlice(gpa);
    defer gpa.free(dropped_nulls_units);
    try std.testing.expectEqualSlices(i64, &.{ 1, 3 }, dropped_nulls_units);
    const dropped_nulls_sales = try (try dropped_nulls.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(dropped_nulls_sales);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 5.0 }, dropped_nulls_sales);
    try std.testing.expectError(error.ColumnNotFound, table.dropNullsColumn("missing"));

    var only_nulls = try table.filterNullsColumn("units");
    defer only_nulls.deinit();
    try std.testing.expectEqual(@as(usize, 1), only_nulls.height());
    const only_nulls_units = try (try only_nulls.column("units")).i64.toOwnedSlice(gpa);
    defer gpa.free(only_nulls_units);
    const only_nulls_validity = try (try only_nulls.column("units")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(only_nulls_validity);
    try std.testing.expectEqualSlices(i64, &.{2}, only_nulls_units);
    try std.testing.expectEqualSlices(bool, &.{false}, only_nulls_validity);
    var no_sales_nulls = try table.filterNullsColumn("sales");
    defer no_sales_nulls.deinit();
    try std.testing.expectEqual(@as(usize, 0), no_sales_nulls.height());
    try std.testing.expectEqual(table.width(), no_sales_nulls.width());

    var reversed = try table.reverseRows();
    defer reversed.deinit();
    try std.testing.expectEqual(table.height(), reversed.height());
    const reversed_sales = try (try reversed.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(reversed_sales);
    const reversed_units = try (try reversed.column("units")).i64.toOwnedSlice(gpa);
    defer gpa.free(reversed_units);
    const reversed_units_validity = try (try reversed.column("units")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(reversed_units_validity);
    try std.testing.expectEqualSlices(f64, &.{ 5.0, 3.0, 2.0 }, reversed_sales);
    try std.testing.expectEqualSlices(i64, &.{ 3, 2, 1 }, reversed_units);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, reversed_units_validity);

    var cast_active = try table.castColumn("active", .i8);
    defer cast_active.deinit();
    try std.testing.expectEqual(DeviceDType.i8, try cast_active.columnDType("active"));
    const cast_active_values = try (try cast_active.column("active")).i8.toOwnedSlice(gpa);
    defer gpa.free(cast_active_values);
    try std.testing.expectEqualSlices(i8, &.{ 1, 0, 1 }, cast_active_values);
    try std.testing.expectError(error.ColumnNotFound, table.castColumn("missing", .f64));

    var indexed = try table.withRowIndex("row_nr", 10);
    defer indexed.deinit();
    try std.testing.expectEqual(@as(usize, 4), indexed.width());
    try std.testing.expectEqual(DeviceDType.usize, try indexed.columnDType("row_nr"));
    const row_nr = try (try indexed.column("row_nr")).usize.toOwnedSlice(gpa);
    defer gpa.free(row_nr);
    try std.testing.expectEqualSlices(usize, &.{ 10, 11, 12 }, row_nr);
    try std.testing.expectError(error.InvalidShape, table.withRowIndex("sales", 0));

    var renamed = try table.renameColumn("sales", "revenue");
    defer renamed.deinit();
    try std.testing.expectEqual(@as(?usize, 0), renamed.columnIndex("revenue"));
    try std.testing.expectEqual(@as(?usize, null), renamed.columnIndex("sales"));
    const revenue_values = try (try renamed.column("revenue")).f64.toOwnedSlice(gpa);
    defer gpa.free(revenue_values);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 3.0, 5.0 }, revenue_values);
    try std.testing.expectError(error.InvalidShape, table.renameColumn("sales", "units"));
    try std.testing.expectError(error.ColumnNotFound, table.renameColumn("missing", "new_name"));

    var renamed_many = try table.renameColumns(&.{ "sales", "units" }, &.{ "revenue", "quantity" });
    defer renamed_many.deinit();
    try std.testing.expectEqual(@as(?usize, 0), renamed_many.columnIndex("revenue"));
    try std.testing.expectEqual(@as(?usize, 1), renamed_many.columnIndex("quantity"));
    try std.testing.expectEqual(@as(?usize, 2), renamed_many.columnIndex("active"));
    try std.testing.expectEqual(@as(?usize, null), renamed_many.columnIndex("sales"));
    const quantity_values = try (try renamed_many.column("quantity")).i64.toOwnedSlice(gpa);
    defer gpa.free(quantity_values);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3 }, quantity_values);

    var prefixed_names = try table.addColumnNamePrefix("src_");
    defer prefixed_names.deinit();
    try std.testing.expectEqual(@as(?usize, 0), prefixed_names.columnIndex("src_sales"));
    try std.testing.expectEqual(@as(?usize, 1), prefixed_names.columnIndex("src_units"));
    try std.testing.expectEqual(@as(?usize, 2), prefixed_names.columnIndex("src_active"));

    var suffixed_names = try table.addColumnNameSuffix("_raw");
    defer suffixed_names.deinit();
    try std.testing.expectEqual(@as(?usize, 0), suffixed_names.columnIndex("sales_raw"));
    try std.testing.expectEqual(@as(?usize, 1), suffixed_names.columnIndex("units_raw"));
    try std.testing.expectEqual(@as(?usize, 2), suffixed_names.columnIndex("active_raw"));
    try std.testing.expectError(error.LengthMismatch, table.renameColumns(&.{"sales"}, &.{ "revenue", "extra" }));
    try std.testing.expectError(error.InvalidShape, table.renameColumns(&.{"sales"}, &.{"units"}));
    try std.testing.expectError(error.ColumnNotFound, table.renameColumns(&.{"missing"}, &.{"new_name"}));

    var moved_front = try table.moveColumn("active", 0);
    defer moved_front.deinit();
    try std.testing.expectEqual(@as(?usize, 0), moved_front.columnIndex("active"));
    try std.testing.expectEqual(@as(?usize, 1), moved_front.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 2), moved_front.columnIndex("units"));
    const moved_front_active = try (try moved_front.column("active")).bool.toOwnedSlice(gpa);
    defer gpa.free(moved_front_active);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, moved_front_active);

    var moved_before = try table.moveColumnBefore("units", "sales");
    defer moved_before.deinit();
    try std.testing.expectEqual(@as(?usize, 0), moved_before.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, 1), moved_before.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 2), moved_before.columnIndex("active"));

    var moved_after = try table.moveColumnAfter("sales", "active");
    defer moved_after.deinit();
    try std.testing.expectEqual(@as(?usize, 0), moved_after.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, 1), moved_after.columnIndex("active"));
    try std.testing.expectEqual(@as(?usize, 2), moved_after.columnIndex("sales"));
    const moved_after_sales = try (try moved_after.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(moved_after_sales);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 3.0, 5.0 }, moved_after_sales);
    try std.testing.expectError(error.ColumnNotFound, table.moveColumn("missing", 0));
    try std.testing.expectError(error.ColumnNotFound, table.moveColumnBefore("sales", "missing"));
    try std.testing.expectError(error.IndexOutOfBounds, table.moveColumn("sales", table.width()));

    var dropped = try table.dropColumn("active");
    defer dropped.deinit();
    try std.testing.expectEqual(@as(usize, 2), dropped.width());
    try std.testing.expectEqual(@as(?usize, null), dropped.columnIndex("active"));
    try std.testing.expectEqual(DeviceDType.f64, try dropped.columnDType("sales"));

    var dropped_many = try table.dropColumns(&.{ "units", "active" });
    defer dropped_many.deinit();
    try std.testing.expectEqual(@as(usize, 1), dropped_many.width());
    try std.testing.expectEqual(DeviceDType.f64, try dropped_many.columnDType("sales"));
    try std.testing.expectError(error.ColumnNotFound, table.dropColumn("missing"));

    var head = try table.head(2);
    defer head.deinit();
    try std.testing.expectEqual(@as(usize, 2), head.height());
    const head_units = try head.column("units");
    try std.testing.expectEqual(@as(usize, 1), head_units.nullCount());

    var rows_dropped = try table.dropRows(&.{ 1, 1 });
    defer rows_dropped.deinit();
    try std.testing.expectEqual(@as(usize, 2), rows_dropped.height());
    try std.testing.expectEqual(table.width(), rows_dropped.width());
    const rows_dropped_sales = try (try rows_dropped.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(rows_dropped_sales);
    const rows_dropped_units_validity = try (try rows_dropped.column("units")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(rows_dropped_units_validity);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 5.0 }, rows_dropped_sales);
    try std.testing.expectEqualSlices(bool, &.{ true, true }, rows_dropped_units_validity);

    var row_range_dropped = try table.dropRowRange(0, 2);
    defer row_range_dropped.deinit();
    try std.testing.expectEqual(@as(usize, 1), row_range_dropped.height());
    const row_range_dropped_sales = try (try row_range_dropped.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_range_dropped_sales);
    try std.testing.expectEqualSlices(f64, &.{5.0}, row_range_dropped_sales);

    var first_row_dropped = try table.dropFirstRows(1);
    defer first_row_dropped.deinit();
    try std.testing.expectEqual(@as(usize, 2), first_row_dropped.height());
    const first_row_dropped_units_validity = try (try first_row_dropped.column("units")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(first_row_dropped_units_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true }, first_row_dropped_units_validity);

    var last_row_dropped = try table.dropLastRows(1);
    defer last_row_dropped.deinit();
    try std.testing.expectEqual(@as(usize, 2), last_row_dropped.height());
    const last_row_dropped_sales = try (try last_row_dropped.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(last_row_dropped_sales);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 3.0 }, last_row_dropped_sales);
    try std.testing.expectError(error.IndexOutOfBounds, table.dropRows(&.{table.height()}));

    var stepped_slice = try table.sliceRowsStep(0, table.height(), 2);
    defer stepped_slice.deinit();
    try std.testing.expectEqual(@as(usize, 2), stepped_slice.height());
    const stepped_slice_sales = try (try stepped_slice.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(stepped_slice_sales);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 5.0 }, stepped_slice_sales);

    var stepped_inner = try table.sliceRowsStep(1, table.height(), 2);
    defer stepped_inner.deinit();
    try std.testing.expectEqual(@as(usize, 1), stepped_inner.height());
    const stepped_inner_units = try (try stepped_inner.column("units")).i64.toOwnedSlice(gpa);
    defer gpa.free(stepped_inner_units);
    const stepped_inner_validity = try (try stepped_inner.column("units")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(stepped_inner_validity);
    try std.testing.expectEqualSlices(i64, &.{2}, stepped_inner_units);
    try std.testing.expectEqualSlices(bool, &.{false}, stepped_inner_validity);

    var stepped_len = try table.sliceStep(0, table.height(), 2);
    defer stepped_len.deinit();
    try std.testing.expectEqual(@as(usize, 2), stepped_len.height());
    try std.testing.expectError(error.InvalidShape, table.sliceRowsStep(0, table.height(), 0));

    var sampled = try table.sampleRows(2, 1234);
    defer sampled.deinit();
    try std.testing.expectEqual(@as(usize, 2), sampled.height());
    try std.testing.expectEqual(table.width(), sampled.width());
    const sampled_sales = try (try sampled.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(sampled_sales);
    var sampled_again = try table.sampleRows(2, 1234);
    defer sampled_again.deinit();
    const sampled_again_sales = try (try sampled_again.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(sampled_again_sales);
    try std.testing.expectEqualSlices(f64, sampled_sales, sampled_again_sales);
    try std.testing.expectError(error.InvalidShape, table.sampleRows(table.height() + 1, 1234));

    var sampled_replacement = try table.sampleRowsWithReplacement(table.height() + 2, 4321);
    defer sampled_replacement.deinit();
    try std.testing.expectEqual(table.height() + 2, sampled_replacement.height());
    try std.testing.expectEqual(table.width(), sampled_replacement.width());
    const sampled_replacement_sales = try (try sampled_replacement.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(sampled_replacement_sales);
    var sampled_replacement_again = try table.sampleRowsWithReplacement(table.height() + 2, 4321);
    defer sampled_replacement_again.deinit();
    const sampled_replacement_again_sales = try (try sampled_replacement_again.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(sampled_replacement_again_sales);
    try std.testing.expectEqualSlices(f64, sampled_replacement_sales, sampled_replacement_again_sales);

    var strided = try table.strideRows(0, 2);
    defer strided.deinit();
    try std.testing.expectEqual(@as(usize, 2), strided.height());
    const strided_sales = try (try strided.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(strided_sales);
    const strided_units = try (try strided.column("units")).i64.toOwnedSlice(gpa);
    defer gpa.free(strided_units);
    const strided_units_validity = try (try strided.column("units")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(strided_units_validity);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 5.0 }, strided_sales);
    try std.testing.expectEqualSlices(i64, &.{ 1, 3 }, strided_units);
    try std.testing.expectEqualSlices(bool, &.{ true, true }, strided_units_validity);

    var empty_stride = try table.strideRows(table.height(), 1);
    defer empty_stride.deinit();
    try std.testing.expectEqual(@as(usize, 0), empty_stride.height());
    try std.testing.expectEqual(table.width(), empty_stride.width());
    try std.testing.expectError(error.InvalidShape, table.strideRows(0, 0));

    var filtered = try table.filter(&.{ true, false, true });
    defer filtered.deinit();
    try std.testing.expectEqual(@as(usize, 2), filtered.height());
    const filtered_units = try filtered.column("units");
    try std.testing.expectEqual(@as(usize, 0), filtered_units.nullCount());
}

test "device dataframe selects and drops columns by nullability" {
    const gpa = std.testing.allocator;

    var sales = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 3.0, 5.0 }, .cpu);
    defer sales.deinit();
    var audited_units = try DeviceColumn.fromSliceWithValidity(i64, gpa, &.{ 1, 2, 3 }, &.{ true, true, true }, .cpu);
    defer audited_units.deinit();
    var quality = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 0.8, 0.0, 0.9 }, &.{ true, false, true }, .cpu);
    defer quality.deinit();
    var active = try DeviceColumn.fromSlice(bool, gpa, &.{ true, false, true }, .cpu);
    defer active.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "sales", .data = sales },
        .{ .name = "audited_units", .data = audited_units },
        .{ .name = "quality", .data = quality },
        .{ .name = "active", .data = active },
    });
    defer table.deinit();

    var nullable = try table.selectNullableColumns();
    defer nullable.deinit();
    try std.testing.expectEqual(@as(usize, 2), nullable.width());
    try std.testing.expectEqual(@as(?usize, 0), nullable.columnIndex("audited_units"));
    try std.testing.expectEqual(@as(?usize, 1), nullable.columnIndex("quality"));
    try std.testing.expect((try nullable.column("audited_units")).nullable());
    try std.testing.expectEqual(@as(usize, 0), (try nullable.column("audited_units")).nullCount());

    var non_nullable = try table.selectNonNullableColumns();
    defer non_nullable.deinit();
    try std.testing.expectEqual(@as(usize, 2), non_nullable.width());
    try std.testing.expectEqual(@as(?usize, 0), non_nullable.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), non_nullable.columnIndex("active"));

    var with_nulls = try table.selectColumnsWithNulls();
    defer with_nulls.deinit();
    try std.testing.expectEqual(@as(usize, 1), with_nulls.width());
    try std.testing.expectEqual(@as(?usize, 0), with_nulls.columnIndex("quality"));
    const quality_values = try (try with_nulls.column("quality")).f64.toOwnedSlice(gpa);
    defer gpa.free(quality_values);
    try std.testing.expectEqualSlices(f64, &.{ 0.8, 0.0, 0.9 }, quality_values);

    var without_nulls = try table.selectColumnsWithoutNulls();
    defer without_nulls.deinit();
    try std.testing.expectEqual(@as(usize, 3), without_nulls.width());
    try std.testing.expectEqual(@as(?usize, 0), without_nulls.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), without_nulls.columnIndex("audited_units"));
    try std.testing.expectEqual(@as(?usize, 2), without_nulls.columnIndex("active"));

    var drop_nullable = try table.dropNullableColumns();
    defer drop_nullable.deinit();
    try std.testing.expectEqual(@as(usize, 2), drop_nullable.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_nullable.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), drop_nullable.columnIndex("active"));

    var drop_non_nullable = try table.dropNonNullableColumns();
    defer drop_non_nullable.deinit();
    try std.testing.expectEqual(@as(usize, 2), drop_non_nullable.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_non_nullable.columnIndex("audited_units"));
    try std.testing.expectEqual(@as(?usize, 1), drop_non_nullable.columnIndex("quality"));

    var drop_with_nulls = try table.dropColumnsWithNulls();
    defer drop_with_nulls.deinit();
    try std.testing.expectEqual(@as(usize, 3), drop_with_nulls.width());
    try std.testing.expectEqual(@as(?usize, null), drop_with_nulls.columnIndex("quality"));

    var drop_without_nulls = try table.dropColumnsWithoutNulls();
    defer drop_without_nulls.deinit();
    try std.testing.expectEqual(@as(usize, 1), drop_without_nulls.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_without_nulls.columnIndex("quality"));
}

test "device dataframe derives NaN and finite predicate columns" {
    const gpa = std.testing.allocator;

    var metric = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, std.math.nan(f64), std.math.inf(f64), 7.0 }, &.{ true, true, true, false }, .cpu);
    defer metric.deinit();
    var id = try DeviceColumn.fromSlice(i64, gpa, &.{ 10, 20, 30, 40 }, .cpu);
    defer id.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "metric", .data = metric },
        .{ .name = "id", .data = id },
    });
    defer table.deinit();

    var nan_flags = try table.isNanColumn("metric", "metric_is_nan");
    defer nan_flags.deinit();
    try std.testing.expectEqual(DeviceDType.bool, try nan_flags.columnDType("metric_is_nan"));
    const metric_is_nan = try (try nan_flags.column("metric_is_nan")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_is_nan);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false }, metric_is_nan);

    var finite_flags = try table.isFiniteColumn("metric", "metric_is_finite");
    defer finite_flags.deinit();
    const metric_is_finite = try (try finite_flags.column("metric_is_finite")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_is_finite);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false }, metric_is_finite);

    var inf_flags = try table.isInfColumn("metric", "metric_is_inf");
    defer inf_flags.deinit();
    const metric_is_inf = try (try inf_flags.column("metric_is_inf")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_is_inf);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false }, metric_is_inf);

    var filled_nan = try table.fillNaNColumn("metric", f64, -1.0);
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
    try std.testing.expectError(error.TypeUnsupported, table.fillNaNColumn("metric", i64, 0));
    try std.testing.expectError(error.ColumnNotFound, table.fillNaNColumn("missing", f64, 0.0));

    var filled_inf = try table.fillInfColumn("metric", f64, -9.0);
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
    try std.testing.expectError(error.TypeUnsupported, table.fillInfColumn("metric", i64, 0));
    try std.testing.expectError(error.ColumnNotFound, table.fillInfColumn("missing", f64, 0.0));

    var filled_non_finite = try table.fillNonFiniteColumn("metric", f64, -5.0);
    defer filled_non_finite.deinit();
    const filled_non_finite_metric = try (try filled_non_finite.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(filled_non_finite_metric);
    const filled_non_finite_validity = try (try filled_non_finite.column("metric")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(filled_non_finite_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, -5.0, -5.0, 7.0 }, filled_non_finite_metric);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false }, filled_non_finite_validity);
    try std.testing.expectError(error.TypeUnsupported, table.fillNonFiniteColumn("metric", i64, 0));
    try std.testing.expectError(error.ColumnNotFound, table.fillNonFiniteColumn("missing", f64, 0.0));

    var integer_finite_flags = try table.isFiniteColumn("id", "id_is_finite");
    defer integer_finite_flags.deinit();
    const id_is_finite = try (try integer_finite_flags.column("id_is_finite")).bool.toOwnedSlice(gpa);
    defer gpa.free(id_is_finite);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, id_is_finite);
    try std.testing.expectError(error.ColumnNotFound, table.isNanColumn("missing", "missing_is_nan"));

    var columns_with_nans = try table.selectColumnsWithNaNs();
    defer columns_with_nans.deinit();
    try std.testing.expectEqual(@as(usize, 1), columns_with_nans.width());
    try std.testing.expectEqual(@as(?usize, 0), columns_with_nans.columnIndex("metric"));

    var columns_without_nans = try table.selectColumnsWithoutNaNs();
    defer columns_without_nans.deinit();
    try std.testing.expectEqual(@as(usize, 1), columns_without_nans.width());
    try std.testing.expectEqual(@as(?usize, 0), columns_without_nans.columnIndex("id"));

    var drop_nan_columns = try table.dropColumnsWithNaNs();
    defer drop_nan_columns.deinit();
    try std.testing.expectEqual(@as(usize, 1), drop_nan_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_nan_columns.columnIndex("id"));

    var drop_non_nan_columns = try table.dropColumnsWithoutNaNs();
    defer drop_non_nan_columns.deinit();
    try std.testing.expectEqual(@as(usize, 1), drop_non_nan_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_non_nan_columns.columnIndex("metric"));

    var columns_with_infs = try table.selectColumnsWithInfs();
    defer columns_with_infs.deinit();
    try std.testing.expectEqual(@as(usize, 1), columns_with_infs.width());
    try std.testing.expectEqual(@as(?usize, 0), columns_with_infs.columnIndex("metric"));

    var columns_without_infs = try table.selectColumnsWithoutInfs();
    defer columns_without_infs.deinit();
    try std.testing.expectEqual(@as(usize, 1), columns_without_infs.width());
    try std.testing.expectEqual(@as(?usize, 0), columns_without_infs.columnIndex("id"));

    var drop_inf_columns = try table.dropColumnsWithInfs();
    defer drop_inf_columns.deinit();
    try std.testing.expectEqual(@as(usize, 1), drop_inf_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_inf_columns.columnIndex("id"));

    var drop_non_inf_columns = try table.dropColumnsWithoutInfs();
    defer drop_non_inf_columns.deinit();
    try std.testing.expectEqual(@as(usize, 1), drop_non_inf_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_non_inf_columns.columnIndex("metric"));

    var columns_with_non_finites = try table.selectColumnsWithNonFinites();
    defer columns_with_non_finites.deinit();
    try std.testing.expectEqual(@as(usize, 1), columns_with_non_finites.width());
    try std.testing.expectEqual(@as(?usize, 0), columns_with_non_finites.columnIndex("metric"));

    var columns_without_non_finites = try table.selectColumnsWithoutNonFinites();
    defer columns_without_non_finites.deinit();
    try std.testing.expectEqual(@as(usize, 1), columns_without_non_finites.width());
    try std.testing.expectEqual(@as(?usize, 0), columns_without_non_finites.columnIndex("id"));

    var drop_non_finite_columns = try table.dropColumnsWithNonFinites();
    defer drop_non_finite_columns.deinit();
    try std.testing.expectEqual(@as(usize, 1), drop_non_finite_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_non_finite_columns.columnIndex("id"));

    var drop_finite_columns = try table.dropColumnsWithoutNonFinites();
    defer drop_finite_columns.deinit();
    try std.testing.expectEqual(@as(usize, 1), drop_finite_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_finite_columns.columnIndex("metric"));

    var dropped_nan_rows = try table.dropNaNsColumn("metric");
    defer dropped_nan_rows.deinit();
    try std.testing.expectEqual(@as(usize, 3), dropped_nan_rows.height());
    const dropped_nan_metric = try (try dropped_nan_rows.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(dropped_nan_metric);
    try std.testing.expect(!std.math.isNan(dropped_nan_metric[0]));
    try std.testing.expect(std.math.isInf(dropped_nan_metric[1]));
    try std.testing.expectEqual(@as(f64, 7.0), dropped_nan_metric[2]);

    var filtered_nan_rows = try table.filterNaNsColumn("metric");
    defer filtered_nan_rows.deinit();
    try std.testing.expectEqual(@as(usize, 1), filtered_nan_rows.height());
    const filtered_nan_metric = try (try filtered_nan_rows.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(filtered_nan_metric);
    try std.testing.expect(std.math.isNan(filtered_nan_metric[0]));
    try std.testing.expectError(error.ColumnNotFound, table.dropNaNsColumn("missing"));

    var dropped_inf_rows = try table.dropInfsColumn("metric");
    defer dropped_inf_rows.deinit();
    try std.testing.expectEqual(@as(usize, 3), dropped_inf_rows.height());
    const dropped_inf_metric = try (try dropped_inf_rows.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(dropped_inf_metric);
    try std.testing.expectEqual(@as(f64, 1.0), dropped_inf_metric[0]);
    try std.testing.expect(std.math.isNan(dropped_inf_metric[1]));
    try std.testing.expectEqual(@as(f64, 7.0), dropped_inf_metric[2]);

    var filtered_inf_rows = try table.filterInfsColumn("metric");
    defer filtered_inf_rows.deinit();
    try std.testing.expectEqual(@as(usize, 1), filtered_inf_rows.height());
    const filtered_inf_metric = try (try filtered_inf_rows.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(filtered_inf_metric);
    try std.testing.expect(std.math.isInf(filtered_inf_metric[0]));
    try std.testing.expectError(error.ColumnNotFound, table.dropInfsColumn("missing"));

    var dropped_non_finite_rows = try table.dropNonFinitesColumn("metric");
    defer dropped_non_finite_rows.deinit();
    try std.testing.expectEqual(@as(usize, 2), dropped_non_finite_rows.height());
    const dropped_non_finite_metric = try (try dropped_non_finite_rows.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(dropped_non_finite_metric);
    try std.testing.expectEqual(@as(f64, 1.0), dropped_non_finite_metric[0]);
    try std.testing.expectEqual(@as(f64, 7.0), dropped_non_finite_metric[1]);

    var filtered_non_finite_rows = try table.filterNonFinitesColumn("metric");
    defer filtered_non_finite_rows.deinit();
    try std.testing.expectEqual(@as(usize, 2), filtered_non_finite_rows.height());
    const filtered_non_finite_metric = try (try filtered_non_finite_rows.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(filtered_non_finite_metric);
    try std.testing.expect(std.math.isNan(filtered_non_finite_metric[0]));
    try std.testing.expect(std.math.isInf(filtered_non_finite_metric[1]));
    try std.testing.expectError(error.ColumnNotFound, table.dropNonFinitesColumn("missing"));

    var row_nan_counts = try table.withRowNaNCount(&.{ "metric", "id" }, "row_nan_count");
    defer row_nan_counts.deinit();
    const row_nan_count = try (try row_nan_counts.column("row_nan_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_nan_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 0 }, row_nan_count);

    var row_inf_counts = try table.withRowInfCount(&.{}, "row_inf_count");
    defer row_inf_counts.deinit();
    const row_inf_count = try (try row_inf_counts.column("row_inf_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_inf_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 0 }, row_inf_count);

    var row_finite_counts = try table.withRowFiniteCount(&.{ "metric", "id" }, "row_finite_count");
    defer row_finite_counts.deinit();
    const row_finite_count = try (try row_finite_counts.column("row_finite_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_finite_count);
    try std.testing.expectEqualSlices(i64, &.{ 2, 1, 1, 1 }, row_finite_count);

    var row_non_finite_counts = try table.withRowNonFiniteCount(&.{}, "row_non_finite_count");
    defer row_non_finite_counts.deinit();
    const row_non_finite_count = try (try row_non_finite_counts.column("row_non_finite_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_non_finite_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 1, 0 }, row_non_finite_count);
    try std.testing.expectError(error.ColumnNotFound, table.withRowNaNCount(&.{"missing"}, "bad_count"));
}

test "device dataframe derives signed Inf predicate columns" {
    const gpa = std.testing.allocator;
    const BF16 = vectra.BFloat16;
    const C64 = vectra.Complex64;

    var metric = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, std.math.inf(f64), -std.math.inf(f64), std.math.nan(f64), 9.0 }, &.{ true, true, true, true, false }, .cpu);
    defer metric.deinit();
    var bf16_metric = try DeviceColumn.fromSlice(BF16, gpa, &.{
        BF16.fromF32(1.0),
        BF16.fromF32(std.math.inf(f32)),
        BF16.fromF32(-std.math.inf(f32)),
        BF16.fromF32(3.0),
        BF16.fromF32(-4.0),
    }, .cpu);
    defer bf16_metric.deinit();
    var complex_metric = try DeviceColumn.fromSlice(C64, gpa, &.{
        C64.init(1.0, 0.0),
        C64.init(std.math.inf(f32), 2.0),
        C64.init(3.0, -std.math.inf(f32)),
        C64.init(std.math.inf(f32), -std.math.inf(f32)),
        C64.init(5.0, 6.0),
    }, .cpu);
    defer complex_metric.deinit();
    var id = try DeviceColumn.fromSlice(i64, gpa, &.{ 10, 20, 30, 40, 50 }, .cpu);
    defer id.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "metric", .data = metric },
        .{ .name = "bf16_metric", .data = bf16_metric },
        .{ .name = "complex_metric", .data = complex_metric },
        .{ .name = "id", .data = id },
    });
    defer table.deinit();

    var metric_positive_flags = try table.isPositiveInfColumn("metric", "metric_is_pos_inf");
    defer metric_positive_flags.deinit();
    try std.testing.expectEqual(DeviceDType.bool, try metric_positive_flags.columnDType("metric_is_pos_inf"));
    const metric_is_pos_inf = try (try metric_positive_flags.column("metric_is_pos_inf")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_is_pos_inf);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false, false }, metric_is_pos_inf);

    var metric_negative_flags = try table.isNegativeInfColumn("metric", "metric_is_neg_inf");
    defer metric_negative_flags.deinit();
    const metric_is_neg_inf = try (try metric_negative_flags.column("metric_is_neg_inf")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_is_neg_inf);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, false }, metric_is_neg_inf);

    var bf16_positive_flags = try table.isPositiveInfColumn("bf16_metric", "bf16_is_pos_inf");
    defer bf16_positive_flags.deinit();
    const bf16_is_pos_inf = try (try bf16_positive_flags.column("bf16_is_pos_inf")).bool.toOwnedSlice(gpa);
    defer gpa.free(bf16_is_pos_inf);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false, false }, bf16_is_pos_inf);

    var bf16_negative_flags = try table.isNegativeInfColumn("bf16_metric", "bf16_is_neg_inf");
    defer bf16_negative_flags.deinit();
    const bf16_is_neg_inf = try (try bf16_negative_flags.column("bf16_is_neg_inf")).bool.toOwnedSlice(gpa);
    defer gpa.free(bf16_is_neg_inf);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, false }, bf16_is_neg_inf);

    var complex_positive_flags = try table.isPositiveInfColumn("complex_metric", "complex_is_pos_inf");
    defer complex_positive_flags.deinit();
    const complex_is_pos_inf = try (try complex_positive_flags.column("complex_is_pos_inf")).bool.toOwnedSlice(gpa);
    defer gpa.free(complex_is_pos_inf);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true, false }, complex_is_pos_inf);

    var complex_negative_flags = try table.isNegativeInfColumn("complex_metric", "complex_is_neg_inf");
    defer complex_negative_flags.deinit();
    const complex_is_neg_inf = try (try complex_negative_flags.column("complex_is_neg_inf")).bool.toOwnedSlice(gpa);
    defer gpa.free(complex_is_neg_inf);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, true, false }, complex_is_neg_inf);

    var integer_positive_flags = try table.isPositiveInfColumn("id", "id_is_pos_inf");
    defer integer_positive_flags.deinit();
    const id_is_pos_inf = try (try integer_positive_flags.column("id_is_pos_inf")).bool.toOwnedSlice(gpa);
    defer gpa.free(id_is_pos_inf);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false, false }, id_is_pos_inf);

    try std.testing.expectError(error.ColumnNotFound, table.isPositiveInfColumn("missing", "missing_is_pos_inf"));
    try std.testing.expectError(error.ColumnNotFound, table.isNegativeInfColumn("missing", "missing_is_neg_inf"));
}

test "device dataframe derives normal predicate columns" {
    const gpa = std.testing.allocator;

    var metric = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, 0.0, std.math.floatTrueMin(f64), std.math.inf(f64), -2.0 }, &.{ true, true, true, true, false }, .cpu);
    defer metric.deinit();
    var id = try DeviceColumn.fromSlice(i64, gpa, &.{ 10, 20, 30, 40, 50 }, .cpu);
    defer id.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "metric", .data = metric },
        .{ .name = "id", .data = id },
    });
    defer table.deinit();

    var metric_flags = try table.isNormalColumn("metric", "metric_is_normal");
    defer metric_flags.deinit();
    try std.testing.expectEqual(DeviceDType.bool, try metric_flags.columnDType("metric_is_normal"));
    const metric_is_normal = try (try metric_flags.column("metric_is_normal")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_is_normal);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false, false }, metric_is_normal);

    var integer_flags = try table.isNormalColumn("id", "id_is_normal");
    defer integer_flags.deinit();
    const id_is_normal = try (try integer_flags.column("id_is_normal")).bool.toOwnedSlice(gpa);
    defer gpa.free(id_is_normal);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false, false }, id_is_normal);

    var row_normal_counts = try table.withRowNormalCount(&.{ "metric", "id" }, "row_normal_count");
    defer row_normal_counts.deinit();
    const row_normal_count = try (try row_normal_counts.column("row_normal_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_normal_count);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 0, 0, 0 }, row_normal_count);
    try std.testing.expectError(error.ColumnNotFound, table.isNormalColumn("missing", "missing_is_normal"));
    try std.testing.expectError(error.ColumnNotFound, table.withRowNormalCount(&.{"missing"}, "bad_count"));
}

test "device dataframe selects normal columns" {
    const gpa = std.testing.allocator;

    var normal_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ 1.0, 2.0, 3.0 }, .cpu);
    defer normal_metric.deinit();
    var zero_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ 0.0, -0.0, 0.0 }, .cpu);
    defer zero_metric.deinit();
    var mixed_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ std.math.floatTrueMin(f64), -4.0, std.math.nan(f64) }, .cpu);
    defer mixed_metric.deinit();
    var special_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ std.math.inf(f64), std.math.nan(f64), 0.0 }, .cpu);
    defer special_metric.deinit();
    var id = try DeviceColumn.fromSlice(i64, gpa, &.{ 10, 20, 30 }, .cpu);
    defer id.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "normal_metric", .data = normal_metric },
        .{ .name = "zero_metric", .data = zero_metric },
        .{ .name = "mixed_metric", .data = mixed_metric },
        .{ .name = "special_metric", .data = special_metric },
        .{ .name = "id", .data = id },
    });
    defer table.deinit();

    var with_normals = try table.selectColumnsWithNormals();
    defer with_normals.deinit();
    try std.testing.expectEqual(@as(usize, 2), with_normals.width());
    try std.testing.expectEqual(@as(?usize, 0), with_normals.columnIndex("normal_metric"));
    try std.testing.expectEqual(@as(?usize, 1), with_normals.columnIndex("mixed_metric"));

    var without_normals = try table.selectColumnsWithoutNormals();
    defer without_normals.deinit();
    try std.testing.expectEqual(@as(usize, 3), without_normals.width());
    try std.testing.expectEqual(@as(?usize, 0), without_normals.columnIndex("zero_metric"));
    try std.testing.expectEqual(@as(?usize, 1), without_normals.columnIndex("special_metric"));
    try std.testing.expectEqual(@as(?usize, 2), without_normals.columnIndex("id"));

    var drop_with_normals = try table.dropColumnsWithNormals();
    defer drop_with_normals.deinit();
    try std.testing.expectEqual(@as(usize, 3), drop_with_normals.width());
    try std.testing.expectEqual(@as(?usize, null), drop_with_normals.columnIndex("normal_metric"));
    try std.testing.expectEqual(@as(?usize, null), drop_with_normals.columnIndex("mixed_metric"));

    var drop_without_normals = try table.dropColumnsWithoutNormals();
    defer drop_without_normals.deinit();
    try std.testing.expectEqual(@as(usize, 2), drop_without_normals.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_without_normals.columnIndex("normal_metric"));
    try std.testing.expectEqual(@as(?usize, 1), drop_without_normals.columnIndex("mixed_metric"));
}

test "device dataframe selects signed Inf columns" {
    const gpa = std.testing.allocator;

    var pos_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ 1.0, std.math.inf(f64), 2.0 }, .cpu);
    defer pos_metric.deinit();
    var neg_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ 3.0, -std.math.inf(f64), 4.0 }, .cpu);
    defer neg_metric.deinit();
    var both_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ std.math.inf(f64), -std.math.inf(f64), 5.0 }, .cpu);
    defer both_metric.deinit();
    var finite_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ 6.0, 7.0, 8.0 }, .cpu);
    defer finite_metric.deinit();
    var id = try DeviceColumn.fromSlice(i64, gpa, &.{ 10, 20, 30 }, .cpu);
    defer id.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "pos_metric", .data = pos_metric },
        .{ .name = "neg_metric", .data = neg_metric },
        .{ .name = "both_metric", .data = both_metric },
        .{ .name = "finite_metric", .data = finite_metric },
        .{ .name = "id", .data = id },
    });
    defer table.deinit();

    var with_positive = try table.selectColumnsWithPositiveInfs();
    defer with_positive.deinit();
    try std.testing.expectEqual(@as(usize, 2), with_positive.width());
    try std.testing.expectEqual(@as(?usize, 0), with_positive.columnIndex("pos_metric"));
    try std.testing.expectEqual(@as(?usize, 1), with_positive.columnIndex("both_metric"));

    var without_positive = try table.selectColumnsWithoutPositiveInfs();
    defer without_positive.deinit();
    try std.testing.expectEqual(@as(usize, 3), without_positive.width());
    try std.testing.expectEqual(@as(?usize, 0), without_positive.columnIndex("neg_metric"));
    try std.testing.expectEqual(@as(?usize, 1), without_positive.columnIndex("finite_metric"));
    try std.testing.expectEqual(@as(?usize, 2), without_positive.columnIndex("id"));

    var drop_with_positive = try table.dropColumnsWithPositiveInfs();
    defer drop_with_positive.deinit();
    try std.testing.expectEqual(@as(usize, 3), drop_with_positive.width());
    try std.testing.expectEqual(@as(?usize, null), drop_with_positive.columnIndex("pos_metric"));
    try std.testing.expectEqual(@as(?usize, null), drop_with_positive.columnIndex("both_metric"));

    var drop_without_positive = try table.dropColumnsWithoutPositiveInfs();
    defer drop_without_positive.deinit();
    try std.testing.expectEqual(@as(usize, 2), drop_without_positive.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_without_positive.columnIndex("pos_metric"));
    try std.testing.expectEqual(@as(?usize, 1), drop_without_positive.columnIndex("both_metric"));

    var with_negative = try table.selectColumnsWithNegativeInfs();
    defer with_negative.deinit();
    try std.testing.expectEqual(@as(usize, 2), with_negative.width());
    try std.testing.expectEqual(@as(?usize, 0), with_negative.columnIndex("neg_metric"));
    try std.testing.expectEqual(@as(?usize, 1), with_negative.columnIndex("both_metric"));

    var without_negative = try table.selectColumnsWithoutNegativeInfs();
    defer without_negative.deinit();
    try std.testing.expectEqual(@as(usize, 3), without_negative.width());
    try std.testing.expectEqual(@as(?usize, 0), without_negative.columnIndex("pos_metric"));
    try std.testing.expectEqual(@as(?usize, 1), without_negative.columnIndex("finite_metric"));
    try std.testing.expectEqual(@as(?usize, 2), without_negative.columnIndex("id"));

    var drop_with_negative = try table.dropColumnsWithNegativeInfs();
    defer drop_with_negative.deinit();
    try std.testing.expectEqual(@as(usize, 3), drop_with_negative.width());
    try std.testing.expectEqual(@as(?usize, null), drop_with_negative.columnIndex("neg_metric"));
    try std.testing.expectEqual(@as(?usize, null), drop_with_negative.columnIndex("both_metric"));

    var drop_without_negative = try table.dropColumnsWithoutNegativeInfs();
    defer drop_without_negative.deinit();
    try std.testing.expectEqual(@as(usize, 2), drop_without_negative.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_without_negative.columnIndex("neg_metric"));
    try std.testing.expectEqual(@as(?usize, 1), drop_without_negative.columnIndex("both_metric"));
}

test "device dataframe fills signed Inf values" {
    const gpa = std.testing.allocator;

    var metric = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, std.math.inf(f64), -std.math.inf(f64), std.math.nan(f64), 9.0 }, &.{ true, true, true, true, false }, .cpu);
    defer metric.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "metric", .data = metric },
    });
    defer table.deinit();

    var filled_positive = try table.fillPositiveInfColumn("metric", f64, 100.0);
    defer filled_positive.deinit();
    const positive_values = try (try filled_positive.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(positive_values);
    const positive_validity = try (try filled_positive.column("metric")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(positive_validity);
    try std.testing.expectEqual(@as(f64, 1.0), positive_values[0]);
    try std.testing.expectEqual(@as(f64, 100.0), positive_values[1]);
    try std.testing.expect(std.math.isNegativeInf(positive_values[2]));
    try std.testing.expect(std.math.isNan(positive_values[3]));
    try std.testing.expectEqual(@as(f64, 9.0), positive_values[4]);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, false }, positive_validity);

    var filled_negative = try table.fillNegativeInfColumn("metric", f64, -100.0);
    defer filled_negative.deinit();
    const negative_values = try (try filled_negative.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(negative_values);
    const negative_validity = try (try filled_negative.column("metric")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(negative_validity);
    try std.testing.expectEqual(@as(f64, 1.0), negative_values[0]);
    try std.testing.expect(std.math.isPositiveInf(negative_values[1]));
    try std.testing.expectEqual(@as(f64, -100.0), negative_values[2]);
    try std.testing.expect(std.math.isNan(negative_values[3]));
    try std.testing.expectEqual(@as(f64, 9.0), negative_values[4]);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, false }, negative_validity);

    try std.testing.expectError(error.TypeUnsupported, table.fillPositiveInfColumn("metric", i64, 0));
    try std.testing.expectError(error.TypeUnsupported, table.fillNegativeInfColumn("metric", i64, 0));
    try std.testing.expectError(error.ColumnNotFound, table.fillPositiveInfColumn("missing", f64, 0.0));
    try std.testing.expectError(error.ColumnNotFound, table.fillNegativeInfColumn("missing", f64, 0.0));
}

test "device dataframe filters signed Inf rows" {
    const gpa = std.testing.allocator;

    var metric = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, std.math.inf(f64), -std.math.inf(f64), std.math.nan(f64), 9.0 }, &.{ true, true, true, true, false }, .cpu);
    defer metric.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "metric", .data = metric },
    });
    defer table.deinit();

    var dropped_positive = try table.dropPositiveInfsColumn("metric");
    defer dropped_positive.deinit();
    try std.testing.expectEqual(@as(usize, 4), dropped_positive.height());
    const dropped_positive_values = try (try dropped_positive.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(dropped_positive_values);
    const dropped_positive_validity = try (try dropped_positive.column("metric")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(dropped_positive_validity);
    try std.testing.expectEqual(@as(f64, 1.0), dropped_positive_values[0]);
    try std.testing.expect(std.math.isNegativeInf(dropped_positive_values[1]));
    try std.testing.expect(std.math.isNan(dropped_positive_values[2]));
    try std.testing.expectEqual(@as(f64, 9.0), dropped_positive_values[3]);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false }, dropped_positive_validity);

    var filtered_positive = try table.filterPositiveInfsColumn("metric");
    defer filtered_positive.deinit();
    try std.testing.expectEqual(@as(usize, 1), filtered_positive.height());
    const filtered_positive_values = try (try filtered_positive.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(filtered_positive_values);
    try std.testing.expect(std.math.isPositiveInf(filtered_positive_values[0]));

    var dropped_negative = try table.dropNegativeInfsColumn("metric");
    defer dropped_negative.deinit();
    try std.testing.expectEqual(@as(usize, 4), dropped_negative.height());
    const dropped_negative_values = try (try dropped_negative.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(dropped_negative_values);
    const dropped_negative_validity = try (try dropped_negative.column("metric")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(dropped_negative_validity);
    try std.testing.expectEqual(@as(f64, 1.0), dropped_negative_values[0]);
    try std.testing.expect(std.math.isPositiveInf(dropped_negative_values[1]));
    try std.testing.expect(std.math.isNan(dropped_negative_values[2]));
    try std.testing.expectEqual(@as(f64, 9.0), dropped_negative_values[3]);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false }, dropped_negative_validity);

    var filtered_negative = try table.filterNegativeInfsColumn("metric");
    defer filtered_negative.deinit();
    try std.testing.expectEqual(@as(usize, 1), filtered_negative.height());
    const filtered_negative_values = try (try filtered_negative.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(filtered_negative_values);
    try std.testing.expect(std.math.isNegativeInf(filtered_negative_values[0]));

    var row_positive_counts = try table.withRowPositiveInfCount(&.{}, "row_positive_inf_count");
    defer row_positive_counts.deinit();
    const row_positive_inf_count = try (try row_positive_counts.column("row_positive_inf_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_positive_inf_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 0, 0 }, row_positive_inf_count);

    var row_negative_counts = try table.withRowNegativeInfCount(&.{"metric"}, "row_negative_inf_count");
    defer row_negative_counts.deinit();
    const row_negative_inf_count = try (try row_negative_counts.column("row_negative_inf_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_negative_inf_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 0, 0 }, row_negative_inf_count);

    try std.testing.expectError(error.ColumnNotFound, table.dropPositiveInfsColumn("missing"));
    try std.testing.expectError(error.ColumnNotFound, table.filterNegativeInfsColumn("missing"));
    try std.testing.expectError(error.ColumnNotFound, table.withRowPositiveInfCount(&.{"missing"}, "bad_count"));
}

test "device dataframe selects and drops columns by name pattern" {
    const gpa = std.testing.allocator;

    var sales_q1 = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 3.0, 5.0 }, .cpu);
    defer sales_q1.deinit();
    var sales_q2 = try DeviceColumn.fromSlice(f64, gpa, &.{ 7.0, 11.0, 13.0 }, .cpu);
    defer sales_q2.deinit();
    var cost_q2 = try DeviceColumn.fromSlice(f64, gpa, &.{ 1.0, 4.0, 9.0 }, .cpu);
    defer cost_q2.deinit();
    var active_flag = try DeviceColumn.fromSlice(bool, gpa, &.{ true, false, true }, .cpu);
    defer active_flag.deinit();
    var region_code = try DeviceColumn.fromSlice(i64, gpa, &.{ 10, 20, 30 }, .cpu);
    defer region_code.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "sales_q1", .data = sales_q1 },
        .{ .name = "sales_q2", .data = sales_q2 },
        .{ .name = "cost_q2", .data = cost_q2 },
        .{ .name = "active_flag", .data = active_flag },
        .{ .name = "region_code", .data = region_code },
    });
    defer table.deinit();

    var prefixed = try table.selectByNamePrefix("sales_");
    defer prefixed.deinit();
    try std.testing.expectEqual(table.height(), prefixed.height());
    try std.testing.expectEqual(@as(usize, 2), prefixed.width());
    try std.testing.expectEqual(@as(?usize, 0), prefixed.columnIndex("sales_q1"));
    try std.testing.expectEqual(@as(?usize, 1), prefixed.columnIndex("sales_q2"));
    try std.testing.expectEqual(@as(?usize, null), prefixed.columnIndex("cost_q2"));

    var suffixed = try table.selectByNameSuffix("_q2");
    defer suffixed.deinit();
    try std.testing.expectEqual(@as(usize, 2), suffixed.width());
    try std.testing.expectEqual(@as(?usize, 0), suffixed.columnIndex("sales_q2"));
    try std.testing.expectEqual(@as(?usize, 1), suffixed.columnIndex("cost_q2"));
    const suffix_cost = try (try suffixed.column("cost_q2")).f64.toOwnedSlice(gpa);
    defer gpa.free(suffix_cost);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 4.0, 9.0 }, suffix_cost);

    var contained = try table.selectByNameContains("code");
    defer contained.deinit();
    try std.testing.expectEqual(@as(usize, 1), contained.width());
    try std.testing.expectEqual(DeviceDType.i64, try contained.columnDType("region_code"));
    const codes = try (try contained.column("region_code")).i64.toOwnedSlice(gpa);
    defer gpa.free(codes);
    try std.testing.expectEqualSlices(i64, &.{ 10, 20, 30 }, codes);

    var no_matches = try table.selectByNamePrefix("missing_");
    defer no_matches.deinit();
    try std.testing.expectEqual(@as(usize, 0), no_matches.width());
    try std.testing.expectEqual(table.height(), no_matches.height());

    var drop_prefixed = try table.dropByNamePrefix("sales_");
    defer drop_prefixed.deinit();
    try std.testing.expectEqual(@as(usize, 3), drop_prefixed.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_prefixed.columnIndex("cost_q2"));
    try std.testing.expectEqual(@as(?usize, 1), drop_prefixed.columnIndex("active_flag"));
    try std.testing.expectEqual(@as(?usize, 2), drop_prefixed.columnIndex("region_code"));
    try std.testing.expectEqual(@as(?usize, null), drop_prefixed.columnIndex("sales_q1"));

    var drop_suffixed = try table.dropByNameSuffix("_q2");
    defer drop_suffixed.deinit();
    try std.testing.expectEqual(@as(usize, 3), drop_suffixed.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_suffixed.columnIndex("sales_q1"));
    try std.testing.expectEqual(@as(?usize, 1), drop_suffixed.columnIndex("active_flag"));
    try std.testing.expectEqual(@as(?usize, 2), drop_suffixed.columnIndex("region_code"));
    try std.testing.expectEqual(@as(?usize, null), drop_suffixed.columnIndex("cost_q2"));

    var drop_contained = try table.dropByNameContains("flag");
    defer drop_contained.deinit();
    try std.testing.expectEqual(@as(usize, 4), drop_contained.width());
    try std.testing.expectEqual(@as(?usize, null), drop_contained.columnIndex("active_flag"));
    const drop_contained_codes = try (try drop_contained.column("region_code")).i64.toOwnedSlice(gpa);
    defer gpa.free(drop_contained_codes);
    try std.testing.expectEqualSlices(i64, &.{ 10, 20, 30 }, drop_contained_codes);

    var drop_no_matches = try table.dropByNameContains("missing");
    defer drop_no_matches.deinit();
    try std.testing.expectEqual(table.width(), drop_no_matches.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_no_matches.columnIndex("sales_q1"));
    try std.testing.expectEqual(@as(?usize, 4), drop_no_matches.columnIndex("region_code"));

    var drop_all = try table.dropByNamePrefix("");
    defer drop_all.deinit();
    try std.testing.expectEqual(@as(usize, 0), drop_all.width());
    try std.testing.expectEqual(table.height(), drop_all.height());
}

test "device dataframe round-trips legacy dataframe fixed-width columns" {
    const gpa = std.testing.allocator;
    var legacy = try DataFrame.init(gpa, &.{
        .{ .name = "sales", .data = .{ .f64 = &.{ 2.0, 3.0, 5.0 } } },
        .{ .name = "units", .data = .{ .i64 = &.{ 1, 2, 3 } } },
        .{ .name = "active", .data = .{ .bool = &.{ true, false, true } } },
    });
    defer legacy.deinit();

    var device_table = try DeviceDataFrame.fromDataFrame(gpa, legacy, .cpu);
    defer device_table.deinit();
    try std.testing.expectEqual(@as(usize, 3), device_table.height());
    try std.testing.expectEqual(DeviceDType.f64, try device_table.columnDType("sales"));

    var roundtrip = try device_table.toDataFrame();
    defer roundtrip.deinit();
    try std.testing.expectEqual(legacy.height(), roundtrip.height());
    try std.testing.expectEqualSlices(f64, legacy.columns[0].f64, roundtrip.columns[0].f64);
    try std.testing.expectEqualSlices(i64, legacy.columns[1].i64, roundtrip.columns[1].i64);
    try std.testing.expectEqualSlices(bool, legacy.columns[2].bool, roundtrip.columns[2].bool);
}

test "device dataframe exports boltha arrow record batch" {
    const gpa = std.testing.allocator;

    var sales = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 3.0, 5.0 }, .cpu);
    defer sales.deinit();
    var units = try DeviceColumn.fromSliceWithValidity(i64, gpa, &.{ 1, 2, 3 }, &.{ true, false, true }, .cpu);
    defer units.deinit();
    var active = try DeviceColumn.fromSlice(bool, gpa, &.{ true, false, true }, .cpu);
    defer active.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "sales", .data = sales },
        .{ .name = "units", .data = units },
        .{ .name = "active", .data = active },
    });
    defer table.deinit();

    var schema = try table.toArrowSchema(gpa);
    defer schema.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 3), schema.fieldCount());
    try std.testing.expectEqual(@as(?usize, 0), schema.fieldIndexByName("sales"));
    try std.testing.expect(schema.fields[0].data_type.eql(.{ .floating_point = .double }));
    try std.testing.expect(schema.fields[1].nullable);
    try std.testing.expect(schema.fields[1].data_type.eql(.{ .int = .{ .bit_width = 64, .signed = true } }));
    try std.testing.expect(schema.fields[2].data_type.eql(.bool));

    var batch = try table.toArrowRecordBatch(gpa);
    defer batch.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 3), batch.row_count);
    try std.testing.expectEqual(@as(usize, 3), batch.columnCount());
    try std.testing.expectEqual(@as(?f64, 2.0), batch.columns[0].float64.value(0));
    try std.testing.expectEqual(@as(?i64, 1), batch.columns[1].int64.value(0));
    try std.testing.expectEqual(@as(?i64, null), batch.columns[1].int64.value(1));
    try std.testing.expectEqual(@as(?bool, true), batch.columns[2].boolean.value(0));
    try std.testing.expectEqual(@as(usize, 1), batch.columns[1].nullCount());

    var arrow_table = try table.toArrowTable(gpa);
    defer arrow_table.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 1), arrow_table.batchCount());
    try std.testing.expectEqual(@as(usize, 3), arrow_table.row_count);
    try std.testing.expectEqual(@as(?usize, 1), arrow_table.columnIndexByName("units"));
}

test "device dataframe eager column expressions and boolean mask filtering" {
    const gpa = std.testing.allocator;

    var sales = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 3.0, 5.0 }, .cpu);
    defer sales.deinit();
    var cost = try DeviceColumn.fromSlice(f64, gpa, &.{ 1.0, 1.5, 2.0 }, .cpu);
    defer cost.deinit();
    var units = try DeviceColumn.fromSliceWithValidity(i64, gpa, &.{ 1, 2, 3 }, &.{ true, false, true }, .cpu);
    defer units.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "sales", .data = sales },
        .{ .name = "cost", .data = cost },
        .{ .name = "units", .data = units },
    });
    defer table.deinit();

    var margin = try table.subColumns("sales", "cost");
    defer margin.deinit();
    const margin_values = try margin.f64.toOwnedSlice(gpa);
    defer gpa.free(margin_values);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 1.5, 3.0 }, margin_values);

    var doubled = try table.binaryColumnScalar("sales", f64, 2.0, .mul);
    defer doubled.deinit();
    const doubled_values = try doubled.f64.toOwnedSlice(gpa);
    defer gpa.free(doubled_values);
    try std.testing.expectEqualSlices(f64, &.{ 4.0, 6.0, 10.0 }, doubled_values);

    var mask = try table.compareColumnScalar("sales", f64, 2.5, .gt);
    defer mask.deinit();
    try std.testing.expectEqual(DeviceDType.bool, mask.dtype());
    const mask_values = try mask.bool.toOwnedSlice(gpa);
    defer gpa.free(mask_values);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true }, mask_values);

    var filtered = try table.filterColumnMask(mask);
    defer filtered.deinit();
    try std.testing.expectEqual(@as(usize, 2), filtered.height());
    const filtered_sales = try filtered.column("sales");
    const filtered_sales_values = try filtered_sales.f64.toOwnedSlice(gpa);
    defer gpa.free(filtered_sales_values);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 5.0 }, filtered_sales_values);

    var units_mask = try table.compareColumnScalar("units", i64, 1, .gt);
    defer units_mask.deinit();
    try std.testing.expectEqual(@as(usize, 1), units_mask.bool.null_count);
    var nullable_mask_filtered = try table.filterColumnMask(units_mask);
    defer nullable_mask_filtered.deinit();
    try std.testing.expectEqual(@as(usize, 1), nullable_mask_filtered.height());
    const nullable_mask_sales = try (try nullable_mask_filtered.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(nullable_mask_sales);
    try std.testing.expectEqualSlices(f64, &.{5.0}, nullable_mask_sales);

    var mask_table = try table.withColumn("units_gt_one", units_mask);
    defer mask_table.deinit();
    var named_mask_filtered = try mask_table.filterColumn("units_gt_one");
    defer named_mask_filtered.deinit();
    try std.testing.expectEqual(@as(usize, 1), named_mask_filtered.height());
    const named_mask_sales = try (try named_mask_filtered.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(named_mask_sales);
    try std.testing.expectEqualSlices(f64, &.{5.0}, named_mask_sales);
    try std.testing.expectError(error.TypeMismatch, mask_table.filterColumn("sales"));
}
