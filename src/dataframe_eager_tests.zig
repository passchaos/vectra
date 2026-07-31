const std = @import("std");
const boltha = @import("boltha");
const vectra = @import("vectra");

const DataFrame = vectra.DataFrame;
const DeviceColumn = vectra.DeviceColumn;
const DeviceDataFrame = vectra.DeviceDataFrame;
const DeviceLazyFrame = vectra.DeviceLazyFrame;
const DeviceDType = vectra.DeviceDType;
const DeviceScalar = vectra.DeviceScalar;
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

    var row_null_ratios = try table.withRowNullRatio(&.{ "sales", "units", "active" }, "row_null_ratio");
    defer row_null_ratios.deinit();
    const row_null_ratio = try (try row_null_ratios.column("row_null_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_null_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 1.0 / 3.0, 0.0 }, row_null_ratio);

    var row_valid_ratios = try table.withRowValidRatio(&.{ "sales", "units", "active" }, "row_valid_ratio");
    defer row_valid_ratios.deinit();
    const row_valid_ratio = try (try row_valid_ratios.column("row_valid_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_valid_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 2.0 / 3.0, 1.0 }, row_valid_ratio);

    var validity_a = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, 2.0, 3.0, 4.0 }, &.{ true, false, false, true }, .cpu);
    defer validity_a.deinit();
    var validity_b = try DeviceColumn.fromSliceWithValidity(i64, gpa, &.{ 10, 20, 30, 40 }, &.{ false, true, false, true }, .cpu);
    defer validity_b.deinit();
    var validity_c = try DeviceColumn.fromSliceWithValidity(bool, gpa, &.{ true, false, true, false }, &.{ false, false, true, true }, .cpu);
    defer validity_c.deinit();
    var weight_a = try DeviceColumn.fromSlice(f64, gpa, &.{ 1.0, 2.0, 3.0, 4.0 }, .cpu);
    defer weight_a.deinit();
    var weight_b = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 1.0, 5.0, 1.0 }, .cpu);
    defer weight_b.deinit();
    var validity_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "a", .data = validity_a },
        .{ .name = "b", .data = validity_b },
        .{ .name = "c", .data = validity_c },
        .{ .name = "wa", .data = weight_a },
        .{ .name = "wb", .data = weight_b },
    });
    defer validity_table.deinit();

    var row_pair_count_table = try validity_table.withRowPairCount(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_pair_count");
    defer row_pair_count_table.deinit();
    try std.testing.expectEqual(DeviceDType.i64, try row_pair_count_table.columnDType("row_pair_count"));
    const row_pair_count = try (try row_pair_count_table.column("row_pair_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_pair_count);
    try std.testing.expectEqualSlices(i64, &.{ 1, 1, 0, 2 }, row_pair_count);

    var row_argmin_table = try validity_table.withRowArgMin(&.{ "a", "b" }, "row_argmin");
    defer row_argmin_table.deinit();
    const row_argmin_column = try row_argmin_table.column("row_argmin");
    try std.testing.expect(row_argmin_column.i64.nullable());
    const row_argmin = try row_argmin_column.i64.toOwnedSlice(gpa);
    defer gpa.free(row_argmin);
    const row_argmin_validity = try row_argmin_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_argmin_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 0 }, row_argmin);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_argmin_validity);

    var row_argmax_table = try validity_table.withRowArgMax(&.{ "a", "b" }, "row_argmax");
    defer row_argmax_table.deinit();
    const row_argmax_column = try row_argmax_table.column("row_argmax");
    try std.testing.expect(row_argmax_column.i64.nullable());
    const row_argmax = try row_argmax_column.i64.toOwnedSlice(gpa);
    defer gpa.free(row_argmax);
    const row_argmax_validity = try row_argmax_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_argmax_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 1 }, row_argmax);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_argmax_validity);

    var row_quantile_table = try validity_table.withRowQuantile(&.{ "a", "b" }, "row_quantile", 0.25);
    defer row_quantile_table.deinit();
    const row_quantile_column = try row_quantile_table.column("row_quantile");
    try std.testing.expect(row_quantile_column.f64.nullable());
    const row_quantile = try row_quantile_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_quantile);
    const row_quantile_validity = try row_quantile_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_quantile_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 13.0 }, row_quantile);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_quantile_validity);

    var row_median_table = try validity_table.withRowMedian(&.{ "a", "b" }, "row_median");
    defer row_median_table.deinit();
    const row_median_column = try row_median_table.column("row_median");
    try std.testing.expect(row_median_column.f64.nullable());
    const row_median = try row_median_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_median);
    const row_median_validity = try row_median_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_median_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 22.0 }, row_median);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_median_validity);

    var row_iqr_table = try validity_table.withRowIqr(&.{ "a", "b" }, "row_iqr");
    defer row_iqr_table.deinit();
    const row_iqr_column = try row_iqr_table.column("row_iqr");
    try std.testing.expect(row_iqr_column.f64.nullable());
    const row_iqr = try row_iqr_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_iqr);
    const row_iqr_validity = try row_iqr_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_iqr_validity);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.0, 18.0 }, row_iqr);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_iqr_validity);

    var row_mad_table = try validity_table.withRowMad(&.{ "a", "b" }, "row_mad");
    defer row_mad_table.deinit();
    const row_mad_column = try row_mad_table.column("row_mad");
    try std.testing.expect(row_mad_column.f64.nullable());
    const row_mad = try row_mad_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_mad);
    const row_mad_validity = try row_mad_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_mad_validity);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.0, 18.0 }, row_mad);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_mad_validity);

    var row_mode_table = try validity_table.withRowMode(&.{ "a", "b" }, "row_mode");
    defer row_mode_table.deinit();
    const row_mode_column = try row_mode_table.column("row_mode");
    try std.testing.expect(row_mode_column.f64.nullable());
    const row_mode = try row_mode_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_mode);
    const row_mode_validity = try row_mode_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_mode_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 4.0 }, row_mode);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_mode_validity);

    var row_entropy_table = try validity_table.withRowEntropy(&.{ "a", "b", "wa" }, "row_entropy");
    defer row_entropy_table.deinit();
    const row_entropy_column = try row_entropy_table.column("row_entropy");
    try std.testing.expect(row_entropy_column.f64.nullable());
    const row_entropy = try row_entropy_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_entropy);
    const row_entropy_validity = try row_entropy_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_entropy_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_entropy[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log(f64, std.math.e, @as(f64, 2.0)), row_entropy[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_entropy[2], 1e-12);
    try std.testing.expectApproxEqAbs(-(@as(f64, 2.0 / 3.0) * std.math.log(f64, std.math.e, @as(f64, 2.0 / 3.0)) + @as(f64, 1.0 / 3.0) * std.math.log(f64, std.math.e, @as(f64, 1.0 / 3.0))), row_entropy[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, row_entropy_validity);

    var row_gini_table = try validity_table.withRowGiniImpurity(&.{ "a", "b", "wa" }, "row_gini");
    defer row_gini_table.deinit();
    const row_gini_column = try row_gini_table.column("row_gini");
    try std.testing.expect(row_gini_column.f64.nullable());
    const row_gini = try row_gini_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_gini);
    const row_gini_validity = try row_gini_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_gini_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_gini[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), row_gini[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_gini[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0 / 9.0), row_gini[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, row_gini_validity);

    var row_perplexity_table = try validity_table.withRowPerplexity(&.{ "a", "b", "wa" }, "row_perplexity");
    defer row_perplexity_table.deinit();
    const row_perplexity_column = try row_perplexity_table.column("row_perplexity");
    try std.testing.expect(row_perplexity_column.f64.nullable());
    const row_perplexity = try row_perplexity_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_perplexity);
    const row_perplexity_validity = try row_perplexity_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_perplexity_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_perplexity[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), row_perplexity[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_perplexity[2], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.exp(-(@as(f64, 2.0 / 3.0) * std.math.log(f64, std.math.e, @as(f64, 2.0 / 3.0)) + @as(f64, 1.0 / 3.0) * std.math.log(f64, std.math.e, @as(f64, 1.0 / 3.0)))), row_perplexity[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, row_perplexity_validity);

    var row_inverse_simpson_table = try validity_table.withRowInverseSimpson(&.{ "a", "b", "wa" }, "row_inverse_simpson");
    defer row_inverse_simpson_table.deinit();
    const row_inverse_simpson_column = try row_inverse_simpson_table.column("row_inverse_simpson");
    try std.testing.expect(row_inverse_simpson_column.f64.nullable());
    const row_inverse_simpson = try row_inverse_simpson_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_inverse_simpson);
    const row_inverse_simpson_validity = try row_inverse_simpson_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_inverse_simpson_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_inverse_simpson[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), row_inverse_simpson[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_inverse_simpson[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.8), row_inverse_simpson[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, row_inverse_simpson_validity);

    var row_concentration_table = try validity_table.withRowSimpsonConcentration(&.{ "a", "b", "wa" }, "row_concentration");
    defer row_concentration_table.deinit();
    const row_concentration = try (try row_concentration_table.column("row_concentration")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_concentration);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_concentration[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), row_concentration[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_concentration[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 9.0), row_concentration[3], 1e-12);

    var row_evenness_table = try validity_table.withRowEvenness(&.{ "a", "b", "wa" }, "row_evenness");
    defer row_evenness_table.deinit();
    const row_evenness = try (try row_evenness_table.column("row_evenness")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_evenness);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_evenness[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_evenness[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_evenness[2], 1e-12);
    try std.testing.expectApproxEqAbs(-(@as(f64, 2.0 / 3.0) * std.math.log(f64, std.math.e, @as(f64, 2.0 / 3.0)) + @as(f64, 1.0 / 3.0) * std.math.log(f64, std.math.e, @as(f64, 1.0 / 3.0))) / std.math.log(f64, std.math.e, @as(f64, 2.0)), row_evenness[3], 1e-12);

    var row_mode_count_table = try validity_table.withRowModeCount(&.{ "a", "b", "wa" }, "row_mode_count");
    defer row_mode_count_table.deinit();
    const row_mode_count_column = try row_mode_count_table.column("row_mode_count");
    try std.testing.expect(row_mode_count_column.i64.nullable());
    const row_mode_count = try row_mode_count_column.i64.toOwnedSlice(gpa);
    defer gpa.free(row_mode_count);
    const row_mode_count_validity = try row_mode_count_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_mode_count_validity);
    try std.testing.expectEqualSlices(i64, &.{ 2, 1, 1, 2 }, row_mode_count);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, row_mode_count_validity);

    var row_mode_ratio_table = try validity_table.withRowModeRatio(&.{ "a", "b", "wa" }, "row_mode_ratio");
    defer row_mode_ratio_table.deinit();
    const row_mode_ratio_column = try row_mode_ratio_table.column("row_mode_ratio");
    try std.testing.expect(row_mode_ratio_column.f64.nullable());
    const row_mode_ratio = try row_mode_ratio_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_mode_ratio);
    const row_mode_ratio_validity = try row_mode_ratio_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_mode_ratio_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_mode_ratio[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), row_mode_ratio[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_mode_ratio[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), row_mode_ratio[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, row_mode_ratio_validity);

    var row_weighted_mean_table = try validity_table.withRowWeightedMean(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_mean");
    defer row_weighted_mean_table.deinit();
    const row_weighted_mean_column = try row_weighted_mean_table.column("row_weighted_mean");
    try std.testing.expect(row_weighted_mean_column.f64.nullable());
    const row_weighted_mean = try row_weighted_mean_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_mean);
    const row_weighted_mean_validity = try row_weighted_mean_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_mean_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_weighted_mean[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), row_weighted_mean[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_weighted_mean[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 56.0 / 5.0), row_weighted_mean[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_mean_validity);

    var row_weighted_quantile_table = try validity_table.withRowWeightedQuantile(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_quantile", 0.9);
    defer row_weighted_quantile_table.deinit();
    const row_weighted_quantile_column = try row_weighted_quantile_table.column("row_weighted_quantile");
    try std.testing.expect(row_weighted_quantile_column.f64.nullable());
    const row_weighted_quantile = try row_weighted_quantile_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_quantile);
    const row_weighted_quantile_validity = try row_weighted_quantile_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_quantile_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 40.0 }, row_weighted_quantile);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_quantile_validity);

    var row_weighted_median_table = try validity_table.withRowWeightedMedian(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_median");
    defer row_weighted_median_table.deinit();
    const row_weighted_median_column = try row_weighted_median_table.column("row_weighted_median");
    try std.testing.expect(row_weighted_median_column.f64.nullable());
    const row_weighted_median = try row_weighted_median_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_median);
    const row_weighted_median_validity = try row_weighted_median_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_median_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 4.0 }, row_weighted_median);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_median_validity);

    var row_weighted_iqr_table = try validity_table.withRowWeightedIqr(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_iqr");
    defer row_weighted_iqr_table.deinit();
    const row_weighted_iqr_column = try row_weighted_iqr_table.column("row_weighted_iqr");
    try std.testing.expect(row_weighted_iqr_column.f64.nullable());
    const row_weighted_iqr = try row_weighted_iqr_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_iqr);
    const row_weighted_iqr_validity = try row_weighted_iqr_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_iqr_validity);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.0, 0.0 }, row_weighted_iqr);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_iqr_validity);

    var row_weighted_mad_table = try validity_table.withRowWeightedMad(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_mad");
    defer row_weighted_mad_table.deinit();
    const row_weighted_mad_column = try row_weighted_mad_table.column("row_weighted_mad");
    try std.testing.expect(row_weighted_mad_column.f64.nullable());
    const row_weighted_mad = try row_weighted_mad_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_mad);
    const row_weighted_mad_validity = try row_weighted_mad_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_mad_validity);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.0, 0.0 }, row_weighted_mad);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_mad_validity);

    var row_weighted_mode_table = try validity_table.withRowWeightedMode(&.{ "a", "b", "wa" }, &.{ "wb", "wa", "wb" }, "row_weighted_mode");
    defer row_weighted_mode_table.deinit();
    const row_weighted_mode_column = try row_weighted_mode_table.column("row_weighted_mode");
    try std.testing.expect(row_weighted_mode_column.f64.nullable());
    const row_weighted_mode = try row_weighted_mode_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_mode);
    const row_weighted_mode_validity = try row_weighted_mode_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_mode_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 3.0, 40.0 }, row_weighted_mode);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, row_weighted_mode_validity);

    var row_weighted_mode_weight_table = try validity_table.withRowWeightedModeWeight(&.{ "a", "b", "wa" }, &.{ "wb", "wa", "wb" }, "row_weighted_mode_weight");
    defer row_weighted_mode_weight_table.deinit();
    const row_weighted_mode_weight = try (try row_weighted_mode_weight_table.column("row_weighted_mode_weight")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_mode_weight);
    try std.testing.expectEqualSlices(f64, &.{ 4.0, 2.0, 5.0, 4.0 }, row_weighted_mode_weight);

    var row_weighted_mode_ratio_table = try validity_table.withRowWeightedModeRatio(&.{ "a", "b", "wa" }, &.{ "wb", "wa", "wb" }, "row_weighted_mode_ratio");
    defer row_weighted_mode_ratio_table.deinit();
    const row_weighted_mode_ratio = try (try row_weighted_mode_ratio_table.column("row_weighted_mode_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_mode_ratio);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_weighted_mode_ratio[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), row_weighted_mode_ratio[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_weighted_mode_ratio[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), row_weighted_mode_ratio[3], 1e-12);

    var row_weighted_entropy_table = try validity_table.withRowWeightedEntropy(&.{ "a", "b", "wa" }, &.{ "wb", "wa", "wb" }, "row_weighted_entropy");
    defer row_weighted_entropy_table.deinit();
    const row_weighted_entropy_column = try row_weighted_entropy_table.column("row_weighted_entropy");
    try std.testing.expect(row_weighted_entropy_column.f64.nullable());
    const row_weighted_entropy = try row_weighted_entropy_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_entropy);
    const row_weighted_entropy_validity = try row_weighted_entropy_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_entropy_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_weighted_entropy[0], 1e-12);
    try std.testing.expectApproxEqAbs(-(@as(f64, 2.0 / 3.0) * std.math.log(f64, std.math.e, @as(f64, 2.0 / 3.0)) + @as(f64, 1.0 / 3.0) * std.math.log(f64, std.math.e, @as(f64, 1.0 / 3.0))), row_weighted_entropy[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_weighted_entropy[2], 1e-12);
    try std.testing.expectApproxEqAbs(-(@as(f64, 2.0 / 3.0) * std.math.log(f64, std.math.e, @as(f64, 2.0 / 3.0)) + @as(f64, 1.0 / 3.0) * std.math.log(f64, std.math.e, @as(f64, 1.0 / 3.0))), row_weighted_entropy[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, row_weighted_entropy_validity);

    var row_weighted_gini_table = try validity_table.withRowWeightedGiniImpurity(&.{ "a", "b", "wa" }, &.{ "wb", "wa", "wb" }, "row_weighted_gini");
    defer row_weighted_gini_table.deinit();
    const row_weighted_gini_column = try row_weighted_gini_table.column("row_weighted_gini");
    try std.testing.expect(row_weighted_gini_column.f64.nullable());
    const row_weighted_gini = try row_weighted_gini_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_gini);
    const row_weighted_gini_validity = try row_weighted_gini_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_gini_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_weighted_gini[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0 / 9.0), row_weighted_gini[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_weighted_gini[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0 / 9.0), row_weighted_gini[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, row_weighted_gini_validity);

    var row_weighted_perplexity_table = try validity_table.withRowWeightedPerplexity(&.{ "a", "b", "wa" }, &.{ "wb", "wa", "wb" }, "row_weighted_perplexity");
    defer row_weighted_perplexity_table.deinit();
    const row_weighted_perplexity = try (try row_weighted_perplexity_table.column("row_weighted_perplexity")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_perplexity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_weighted_perplexity[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.exp(-(@as(f64, 2.0 / 3.0) * std.math.log(f64, std.math.e, @as(f64, 2.0 / 3.0)) + @as(f64, 1.0 / 3.0) * std.math.log(f64, std.math.e, @as(f64, 1.0 / 3.0)))), row_weighted_perplexity[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_weighted_perplexity[2], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.exp(-(@as(f64, 2.0 / 3.0) * std.math.log(f64, std.math.e, @as(f64, 2.0 / 3.0)) + @as(f64, 1.0 / 3.0) * std.math.log(f64, std.math.e, @as(f64, 1.0 / 3.0)))), row_weighted_perplexity[3], 1e-12);

    var row_weighted_inverse_table = try validity_table.withRowWeightedInverseSimpson(&.{ "a", "b", "wa" }, &.{ "wb", "wa", "wb" }, "row_weighted_inverse");
    defer row_weighted_inverse_table.deinit();
    const row_weighted_inverse = try (try row_weighted_inverse_table.column("row_weighted_inverse")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_inverse);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_weighted_inverse[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.8), row_weighted_inverse[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_weighted_inverse[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.8), row_weighted_inverse[3], 1e-12);

    var row_weighted_concentration_table = try validity_table.withRowWeightedSimpsonConcentration(&.{ "a", "b", "wa" }, &.{ "wb", "wa", "wb" }, "row_weighted_concentration");
    defer row_weighted_concentration_table.deinit();
    const row_weighted_concentration = try (try row_weighted_concentration_table.column("row_weighted_concentration")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_concentration);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_weighted_concentration[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 9.0), row_weighted_concentration[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_weighted_concentration[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 9.0), row_weighted_concentration[3], 1e-12);

    var row_weighted_evenness_table = try validity_table.withRowWeightedEvenness(&.{ "a", "b", "wa" }, &.{ "wb", "wa", "wb" }, "row_weighted_evenness");
    defer row_weighted_evenness_table.deinit();
    const row_weighted_evenness = try (try row_weighted_evenness_table.column("row_weighted_evenness")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_evenness);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_weighted_evenness[0], 1e-12);
    try std.testing.expectApproxEqAbs(-(@as(f64, 2.0 / 3.0) * std.math.log(f64, std.math.e, @as(f64, 2.0 / 3.0)) + @as(f64, 1.0 / 3.0) * std.math.log(f64, std.math.e, @as(f64, 1.0 / 3.0))) / std.math.log(f64, std.math.e, @as(f64, 2.0)), row_weighted_evenness[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_weighted_evenness[2], 1e-12);
    try std.testing.expectApproxEqAbs(-(@as(f64, 2.0 / 3.0) * std.math.log(f64, std.math.e, @as(f64, 2.0 / 3.0)) + @as(f64, 1.0 / 3.0) * std.math.log(f64, std.math.e, @as(f64, 1.0 / 3.0))) / std.math.log(f64, std.math.e, @as(f64, 2.0)), row_weighted_evenness[3], 1e-12);

    var row_weighted_variance_table = try validity_table.withRowWeightedVariance(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_variance", 0.0);
    defer row_weighted_variance_table.deinit();
    const row_weighted_variance_column = try row_weighted_variance_table.column("row_weighted_variance");
    try std.testing.expect(row_weighted_variance_column.f64.nullable());
    const row_weighted_variance = try row_weighted_variance_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_variance);
    const row_weighted_variance_validity = try row_weighted_variance_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_variance_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_weighted_variance[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_weighted_variance[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_weighted_variance[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 207.36), row_weighted_variance[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_variance_validity);

    var row_weighted_stddev_table = try validity_table.withRowWeightedStddev(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_stddev", 0.0);
    defer row_weighted_stddev_table.deinit();
    const row_weighted_stddev_column = try row_weighted_stddev_table.column("row_weighted_stddev");
    try std.testing.expect(row_weighted_stddev_column.f64.nullable());
    const row_weighted_stddev = try row_weighted_stddev_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_stddev);
    const row_weighted_stddev_validity = try row_weighted_stddev_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_stddev_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_weighted_stddev[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_weighted_stddev[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_weighted_stddev[2], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 207.36)), row_weighted_stddev[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_stddev_validity);

    var row_weighted_covariance_table = try validity_table.withRowWeightedCovariance(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_covariance", 0.0);
    defer row_weighted_covariance_table.deinit();
    const row_weighted_covariance_column = try row_weighted_covariance_table.column("row_weighted_covariance");
    try std.testing.expect(row_weighted_covariance_column.f64.nullable());
    const row_weighted_covariance = try row_weighted_covariance_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_covariance);
    const row_weighted_covariance_validity = try row_weighted_covariance_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_covariance_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_weighted_covariance[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_weighted_covariance[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_weighted_covariance[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -17.28), row_weighted_covariance[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_covariance_validity);

    var row_weighted_correlation_table = try validity_table.withRowWeightedCorrelation(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_correlation", 0.0);
    defer row_weighted_correlation_table.deinit();
    const row_weighted_correlation_column = try row_weighted_correlation_table.column("row_weighted_correlation");
    try std.testing.expect(row_weighted_correlation_column.f64.nullable());
    const row_weighted_correlation = try row_weighted_correlation_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_correlation);
    const row_weighted_correlation_validity = try row_weighted_correlation_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_correlation_validity);
    try std.testing.expect(std.math.isNan(row_weighted_correlation[0]));
    try std.testing.expect(std.math.isNan(row_weighted_correlation[1]));
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_weighted_correlation[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -1.0), row_weighted_correlation[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_correlation_validity);

    var row_weighted_beta_table = try validity_table.withRowWeightedBeta(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_beta", 0.0);
    defer row_weighted_beta_table.deinit();
    const row_weighted_beta_column = try row_weighted_beta_table.column("row_weighted_beta");
    try std.testing.expect(row_weighted_beta_column.f64.nullable());
    const row_weighted_beta = try row_weighted_beta_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_beta);
    const row_weighted_beta_validity = try row_weighted_beta_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_beta_validity);
    try std.testing.expect(std.math.isNan(row_weighted_beta[0]));
    try std.testing.expect(std.math.isNan(row_weighted_beta[1]));
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_weighted_beta[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -1.0 / 12.0), row_weighted_beta[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_beta_validity);

    var row_dot_table = try validity_table.withRowDot(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_dot");
    defer row_dot_table.deinit();
    const row_dot_column = try row_dot_table.column("row_dot");
    try std.testing.expect(row_dot_column.f64.nullable());
    const row_dot = try row_dot_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_dot);
    const row_dot_validity = try row_dot_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_dot_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 56.0 }, row_dot);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_dot_validity);

    var row_cosine_table = try validity_table.withRowCosineSimilarity(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_cosine");
    defer row_cosine_table.deinit();
    const row_cosine_column = try row_cosine_table.column("row_cosine");
    try std.testing.expect(row_cosine_column.f64.nullable());
    const row_cosine = try row_cosine_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_cosine);
    const row_cosine_validity = try row_cosine_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_cosine_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_cosine[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_cosine[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_cosine[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 56.0) / (std.math.sqrt(@as(f64, 1616.0)) * std.math.sqrt(@as(f64, 17.0))), row_cosine[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_cosine_validity);

    var row_sqdist_table = try validity_table.withRowSquaredEuclideanDistance(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_sqdist");
    defer row_sqdist_table.deinit();
    const row_sqdist = try (try row_sqdist_table.column("row_sqdist")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_sqdist);
    const row_sqdist_validity = try (try row_sqdist_table.column("row_sqdist")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_sqdist_validity);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 361.0, 0.0, 1521.0 }, row_sqdist);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_sqdist_validity);

    var row_euclidean_table = try validity_table.withRowEuclideanDistance(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_euclidean");
    defer row_euclidean_table.deinit();
    const row_euclidean = try (try row_euclidean_table.column("row_euclidean")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_euclidean);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_euclidean[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 19.0), row_euclidean[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_euclidean[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 39.0), row_euclidean[3], 1e-12);

    var row_manhattan_table = try validity_table.withRowManhattanDistance(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_manhattan");
    defer row_manhattan_table.deinit();
    const row_manhattan = try (try row_manhattan_table.column("row_manhattan")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_manhattan);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 19.0, 0.0, 39.0 }, row_manhattan);

    var row_chebyshev_table = try validity_table.withRowChebyshevDistance(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_chebyshev");
    defer row_chebyshev_table.deinit();
    const row_chebyshev = try (try row_chebyshev_table.column("row_chebyshev")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_chebyshev);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 19.0, 0.0, 39.0 }, row_chebyshev);

    var row_canberra_table = try validity_table.withRowCanberraDistance(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_canberra");
    defer row_canberra_table.deinit();
    const row_canberra = try (try row_canberra_table.column("row_canberra")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_canberra);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_canberra[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 19.0 / 21.0), row_canberra[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_canberra[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0 + 39.0 / 41.0), row_canberra[3], 1e-12);

    var row_bray_table = try validity_table.withRowBrayCurtisDistance(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_bray");
    defer row_bray_table.deinit();
    const row_bray = try (try row_bray_table.column("row_bray")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_bray);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_bray[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 19.0 / 21.0), row_bray[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_bray[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 39.0 / 49.0), row_bray[3], 1e-12);

    var row_mean_error_table = try validity_table.withRowMeanError(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_mean_error");
    defer row_mean_error_table.deinit();
    const row_mean_error_column = try row_mean_error_table.column("row_mean_error");
    try std.testing.expect(row_mean_error_column.f64.nullable());
    const row_mean_error = try row_mean_error_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_mean_error);
    const row_mean_error_validity = try row_mean_error_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_mean_error_validity);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 19.0, 0.0, 19.5 }, row_mean_error);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_mean_error_validity);

    var row_mae_table = try validity_table.withRowMae(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_mae");
    defer row_mae_table.deinit();
    const row_mae = try (try row_mae_table.column("row_mae")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_mae);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 19.0, 0.0, 19.5 }, row_mae);

    var row_mse_table = try validity_table.withRowMse(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_mse");
    defer row_mse_table.deinit();
    const row_mse = try (try row_mse_table.column("row_mse")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_mse);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 361.0, 0.0, 760.5 }, row_mse);

    var row_rmse_table = try validity_table.withRowRmse(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_rmse");
    defer row_rmse_table.deinit();
    const row_rmse = try (try row_rmse_table.column("row_rmse")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_rmse);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_rmse[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 19.0), row_rmse[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_rmse[2], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 760.5)), row_rmse[3], 1e-12);

    var row_mape_table = try validity_table.withRowMape(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_mape");
    defer row_mape_table.deinit();
    const row_mape_column = try row_mape_table.column("row_mape");
    try std.testing.expect(row_mape_column.f64.nullable());
    const row_mape = try row_mape_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_mape);
    const row_mape_validity = try row_mape_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_mape_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_mape[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 19.0 / 20.0), row_mape[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_mape[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 39.0 / 80.0), row_mape[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_mape_validity);

    var row_smape_table = try validity_table.withRowSmape(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_smape");
    defer row_smape_table.deinit();
    const row_smape_column = try row_smape_table.column("row_smape");
    try std.testing.expect(row_smape_column.f64.nullable());
    const row_smape = try row_smape_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_smape);
    const row_smape_validity = try row_smape_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_smape_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_smape[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 38.0 / 21.0), row_smape[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_smape[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 39.0 / 41.0), row_smape[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_smape_validity);
    var row_covariance_table = try validity_table.withRowCovariance(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_covariance");
    defer row_covariance_table.deinit();
    const row_covariance_column = try row_covariance_table.column("row_covariance");
    try std.testing.expect(row_covariance_column.f64.nullable());
    const row_covariance = try row_covariance_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_covariance);
    const row_covariance_validity = try row_covariance_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_covariance_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_covariance[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_covariance[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_covariance[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -27.0), row_covariance[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_covariance_validity);

    var row_correlation_table = try validity_table.withRowCorrelation(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_correlation");
    defer row_correlation_table.deinit();
    const row_correlation_column = try row_correlation_table.column("row_correlation");
    try std.testing.expect(row_correlation_column.f64.nullable());
    const row_correlation = try row_correlation_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_correlation);
    const row_correlation_validity = try row_correlation_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_correlation_validity);
    try std.testing.expect(std.math.isNan(row_correlation[0]));
    try std.testing.expect(std.math.isNan(row_correlation[1]));
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_correlation[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -1.0), row_correlation[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_correlation_validity);

    var row_beta_table = try validity_table.withRowBeta(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_beta");
    defer row_beta_table.deinit();
    const row_beta_column = try row_beta_table.column("row_beta");
    try std.testing.expect(row_beta_column.f64.nullable());
    const row_beta = try row_beta_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_beta);
    const row_beta_validity = try row_beta_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_beta_validity);
    try std.testing.expect(std.math.isNan(row_beta[0]));
    try std.testing.expect(std.math.isNan(row_beta[1]));
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_beta[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -1.0 / 12.0), row_beta[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_beta_validity);
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowPairCount(&.{"a"}, &.{ "wa", "wb" }, "bad_row_pair_count"));
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowWeightedMean(&.{"a"}, &.{ "wa", "wb" }, "bad_row_weighted_mean"));
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowWeightedQuantile(&.{"a"}, &.{ "wa", "wb" }, "bad_row_weighted_quantile", 0.5));
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowWeightedMode(&.{"a"}, &.{ "wa", "wb" }, "bad_row_weighted_mode"));
    try std.testing.expectError(error.InvalidShape, validity_table.withRowWeightedQuantile(&.{ "a", "b" }, &.{ "wa", "wb" }, "bad_row_weighted_quantile", 1.5));
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowWeightedVariance(&.{"a"}, &.{ "wa", "wb" }, "bad_row_weighted_variance", 0.0));
    try std.testing.expectError(error.InvalidShape, validity_table.withRowWeightedVariance(&.{ "a", "b" }, &.{ "wa", "wb" }, "bad_row_weighted_variance", -1.0));
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowWeightedCovariance(&.{"a"}, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "bad_row_weighted_covariance", 0.0));
    try std.testing.expectError(error.InvalidShape, validity_table.withRowWeightedCovariance(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "bad_row_weighted_covariance", -1.0));
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowDot(&.{"a"}, &.{ "wa", "wb" }, "bad_row_dot"));
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowCovariance(&.{"a"}, &.{ "wa", "wb" }, "bad_row_covariance"));

    var row_distinct_table = try validity_table.withRowCountDistinct(&.{ "a", "b" }, "row_distinct");
    defer row_distinct_table.deinit();
    const row_distinct = try (try row_distinct_table.column("row_distinct")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_distinct);
    try std.testing.expectEqualSlices(i64, &.{ 1, 1, 0, 2 }, row_distinct);

    var row_unique_table = try validity_table.withRowNUnique(&.{ "a", "b" }, "row_unique");
    defer row_unique_table.deinit();
    const row_unique = try (try row_unique_table.column("row_unique")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_unique);
    try std.testing.expectEqualSlices(i64, &.{ 1, 1, 0, 2 }, row_unique);
    try std.testing.expectError(error.InvalidShape, validity_table.withRowQuantile(&.{ "a", "b" }, "bad_row_quantile", 1.5));

    var row_sum_table = try validity_table.withRowSum(&.{ "a", "b" }, "row_sum");
    defer row_sum_table.deinit();
    const row_sum_column = try row_sum_table.column("row_sum");
    try std.testing.expectEqual(DeviceDType.f64, row_sum_column.dtype());
    try std.testing.expect(row_sum_column.f64.nullable());
    const row_sum = try row_sum_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_sum);
    const row_sum_validity = try row_sum_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_sum_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 44.0 }, row_sum);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_sum_validity);

    var row_mean_table = try validity_table.withRowMean(&.{ "a", "b" }, "row_mean");
    defer row_mean_table.deinit();
    const row_mean_column = try row_mean_table.column("row_mean");
    const row_mean = try row_mean_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_mean);
    const row_mean_validity = try row_mean_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_mean_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 22.0 }, row_mean);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_mean_validity);

    var row_geo_table = try validity_table.withRowGeometricMean(&.{ "a", "b" }, "row_geo");
    defer row_geo_table.deinit();
    const row_geo_column = try row_geo_table.column("row_geo");
    try std.testing.expect(row_geo_column.f64.nullable());
    const row_geo = try row_geo_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_geo);
    const row_geo_validity = try row_geo_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_geo_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_geo[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), row_geo[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_geo[2], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 160.0)), row_geo[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_geo_validity);

    var row_harm_table = try validity_table.withRowHarmonicMean(&.{ "a", "b" }, "row_harm");
    defer row_harm_table.deinit();
    const row_harm_column = try row_harm_table.column("row_harm");
    try std.testing.expect(row_harm_column.f64.nullable());
    const row_harm = try row_harm_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_harm);
    const row_harm_validity = try row_harm_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_harm_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_harm[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), row_harm[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_harm[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 80.0 / 11.0), row_harm[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_harm_validity);

    var row_skew_table = try validity_table.withRowSkewness(&.{ "a", "b" }, "row_skew");
    defer row_skew_table.deinit();
    const row_skew_column = try row_skew_table.column("row_skew");
    try std.testing.expect(row_skew_column.f64.nullable());
    const row_skew = try row_skew_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_skew);
    const row_skew_validity = try row_skew_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_skew_validity);
    try std.testing.expect(std.math.isNan(row_skew[0]));
    try std.testing.expect(std.math.isNan(row_skew[1]));
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_skew[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_skew[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_skew_validity);

    var row_kurt_table = try validity_table.withRowKurtosis(&.{ "a", "b" }, "row_kurt");
    defer row_kurt_table.deinit();
    const row_kurt_column = try row_kurt_table.column("row_kurt");
    try std.testing.expect(row_kurt_column.f64.nullable());
    const row_kurt = try row_kurt_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_kurt);
    const row_kurt_validity = try row_kurt_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_kurt_validity);
    try std.testing.expect(std.math.isNan(row_kurt[0]));
    try std.testing.expect(std.math.isNan(row_kurt[1]));
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_kurt[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -2.0), row_kurt[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_kurt_validity);

    var row_prod_table = try validity_table.withRowProd(&.{ "a", "b" }, "row_prod");
    defer row_prod_table.deinit();
    const row_prod_column = try row_prod_table.column("row_prod");
    const row_prod = try row_prod_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_prod);
    const row_prod_validity = try row_prod_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_prod_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 160.0 }, row_prod);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_prod_validity);

    var row_min_table = try validity_table.withRowMin(&.{ "a", "b" }, "row_min");
    defer row_min_table.deinit();
    const row_min_column = try row_min_table.column("row_min");
    const row_min = try row_min_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_min);
    const row_min_validity = try row_min_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_min_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 4.0 }, row_min);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_min_validity);

    var row_max_table = try validity_table.withRowMax(&.{ "a", "b" }, "row_max");
    defer row_max_table.deinit();
    const row_max_column = try row_max_table.column("row_max");
    const row_max = try row_max_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_max);
    const row_max_validity = try row_max_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_max_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 40.0 }, row_max);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_max_validity);

    var row_ptp_table = try validity_table.withRowPtp(&.{ "a", "b" }, "row_ptp");
    defer row_ptp_table.deinit();
    const row_ptp_column = try row_ptp_table.column("row_ptp");
    const row_ptp = try row_ptp_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_ptp);
    const row_ptp_validity = try row_ptp_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_ptp_validity);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.0, 36.0 }, row_ptp);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_ptp_validity);

    var row_mean_abs_table = try validity_table.withRowMeanAbs(&.{ "a", "b" }, "row_mean_abs");
    defer row_mean_abs_table.deinit();
    const row_mean_abs_column = try row_mean_abs_table.column("row_mean_abs");
    try std.testing.expect(row_mean_abs_column.f64.nullable());
    const row_mean_abs = try row_mean_abs_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_mean_abs);
    const row_mean_abs_validity = try row_mean_abs_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_mean_abs_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 22.0 }, row_mean_abs);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_mean_abs_validity);

    var row_rms_table = try validity_table.withRowRms(&.{ "a", "b" }, "row_rms");
    defer row_rms_table.deinit();
    const row_rms_column = try row_rms_table.column("row_rms");
    try std.testing.expect(row_rms_column.f64.nullable());
    const row_rms = try row_rms_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_rms);
    const row_rms_validity = try row_rms_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_rms_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_rms[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), row_rms[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_rms[2], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 808.0)), row_rms[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_rms_validity);

    var row_l1_table = try validity_table.withRowL1Norm(&.{ "a", "b" }, "row_l1");
    defer row_l1_table.deinit();
    const row_l1_column = try row_l1_table.column("row_l1");
    const row_l1 = try row_l1_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_l1);
    const row_l1_validity = try row_l1_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_l1_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 44.0 }, row_l1);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_l1_validity);

    var row_l2_table = try validity_table.withRowL2Norm(&.{ "a", "b" }, "row_l2");
    defer row_l2_table.deinit();
    const row_l2_column = try row_l2_table.column("row_l2");
    const row_l2 = try row_l2_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_l2);
    const row_l2_validity = try row_l2_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_l2_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_l2[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), row_l2[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_l2[2], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 1616.0)), row_l2[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_l2_validity);

    var row_variance_table = try validity_table.withRowVariance(&.{ "a", "b" }, "row_variance", 0.0);
    defer row_variance_table.deinit();
    const row_variance_column = try row_variance_table.column("row_variance");
    try std.testing.expect(row_variance_column.f64.nullable());
    const row_variance = try row_variance_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_variance);
    const row_variance_validity = try row_variance_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_variance_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_variance[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_variance[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_variance[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 324.0), row_variance[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_variance_validity);

    var row_stddev_table = try validity_table.withRowStddev(&.{ "a", "b" }, "row_stddev", 1.0);
    defer row_stddev_table.deinit();
    const row_stddev_column = try row_stddev_table.column("row_stddev");
    try std.testing.expect(row_stddev_column.f64.nullable());
    const row_stddev = try row_stddev_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_stddev);
    const row_stddev_validity = try row_stddev_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_stddev_validity);
    try std.testing.expect(std.math.isNan(row_stddev[0]));
    try std.testing.expect(std.math.isNan(row_stddev[1]));
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_stddev[2], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 648.0)), row_stddev[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_stddev_validity);

    var row_sem_table = try validity_table.withRowSem(&.{ "a", "b" }, "row_sem", 1.0);
    defer row_sem_table.deinit();
    const row_sem_column = try row_sem_table.column("row_sem");
    try std.testing.expect(row_sem_column.f64.nullable());
    const row_sem = try row_sem_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_sem);
    const row_sem_validity = try row_sem_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_sem_validity);
    try std.testing.expect(std.math.isNan(row_sem[0]));
    try std.testing.expect(std.math.isNan(row_sem[1]));
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_sem[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 18.0), row_sem[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_sem_validity);

    var row_cv_table = try validity_table.withRowCv(&.{ "a", "b" }, "row_cv", 0.0);
    defer row_cv_table.deinit();
    const row_cv_column = try row_cv_table.column("row_cv");
    try std.testing.expect(row_cv_column.f64.nullable());
    const row_cv = try row_cv_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_cv);
    const row_cv_validity = try row_cv_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_cv_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_cv[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_cv[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_cv[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 18.0 / 22.0), row_cv[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_cv_validity);
    try std.testing.expectError(error.InvalidShape, validity_table.withRowVariance(&.{ "a", "b" }, "bad_row_variance", -1.0));
    try std.testing.expectError(error.TypeMismatch, validity_table.withRowSum(&.{"c"}, "bad_row_sum"));

    var row_first_valid_table = try validity_table.withRowFirstValidIndex(&.{ "a", "b", "c" }, "first_valid");
    defer row_first_valid_table.deinit();
    const row_first_valid_column = try row_first_valid_table.column("first_valid");
    try std.testing.expect(row_first_valid_column.i64.nullable());
    const row_first_valid = try row_first_valid_column.i64.toOwnedSlice(gpa);
    defer gpa.free(row_first_valid);
    const row_first_valid_validity = try row_first_valid_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_first_valid_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 2, 0 }, row_first_valid);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, row_first_valid_validity);

    var row_last_valid_table = try validity_table.withRowLastValidIndex(&.{ "a", "b", "c" }, "last_valid");
    defer row_last_valid_table.deinit();
    const row_last_valid_column = try row_last_valid_table.column("last_valid");
    const row_last_valid = try row_last_valid_column.i64.toOwnedSlice(gpa);
    defer gpa.free(row_last_valid);
    const row_last_valid_validity = try row_last_valid_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_last_valid_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 2, 2 }, row_last_valid);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, row_last_valid_validity);

    var row_first_null_table = try validity_table.withRowFirstNullIndex(&.{ "a", "b", "c" }, "first_null");
    defer row_first_null_table.deinit();
    const row_first_null_column = try row_first_null_table.column("first_null");
    try std.testing.expect(row_first_null_column.i64.nullable());
    const row_first_null = try row_first_null_column.i64.toOwnedSlice(gpa);
    defer gpa.free(row_first_null);
    const row_first_null_validity = try row_first_null_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_first_null_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 0, 0 }, row_first_null);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false }, row_first_null_validity);

    var row_last_null_table = try validity_table.withRowLastNullIndex(&.{ "a", "b", "c" }, "last_null");
    defer row_last_null_table.deinit();
    const row_last_null_column = try row_last_null_table.column("last_null");
    const row_last_null = try row_last_null_column.i64.toOwnedSlice(gpa);
    defer gpa.free(row_last_null);
    const row_last_null_validity = try row_last_null_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_last_null_validity);
    try std.testing.expectEqualSlices(i64, &.{ 2, 2, 1, 0 }, row_last_null);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false }, row_last_null_validity);

    var row_true_counts = try table.withRowTrueCount(&.{"active"}, "row_true_count");
    defer row_true_counts.deinit();
    const row_true_count = try (try row_true_counts.column("row_true_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_true_count);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 1 }, row_true_count);

    var row_false_counts = try table.withRowFalseCount(&.{"active"}, "row_false_count");
    defer row_false_counts.deinit();
    const row_false_count = try (try row_false_counts.column("row_false_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_false_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0 }, row_false_count);

    var row_true_ratios = try table.withRowTrueRatio(&.{"active"}, "row_true_ratio");
    defer row_true_ratios.deinit();
    const row_true_ratio_column = try row_true_ratios.column("row_true_ratio");
    try std.testing.expect(row_true_ratio_column.f64.nullable());
    const row_true_ratio = try row_true_ratio_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_true_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 0.0, 1.0 }, row_true_ratio);

    var row_false_ratios = try table.withRowFalseRatio(&.{"active"}, "row_false_ratio");
    defer row_false_ratios.deinit();
    const row_false_ratio_column = try row_false_ratios.column("row_false_ratio");
    try std.testing.expect(row_false_ratio_column.f64.nullable());
    const row_false_ratio = try row_false_ratio_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_false_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 1.0, 0.0 }, row_false_ratio);

    var row_any_true_table = try table.withRowAnyTrue(&.{"active"}, "row_any_true");
    defer row_any_true_table.deinit();
    const row_any_true = try (try row_any_true_table.column("row_any_true")).bool.toOwnedSlice(gpa);
    defer gpa.free(row_any_true);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, row_any_true);

    var row_all_true_table = try table.withRowAllTrue(&.{"active"}, "row_all_true");
    defer row_all_true_table.deinit();
    const row_all_true = try (try row_all_true_table.column("row_all_true")).bool.toOwnedSlice(gpa);
    defer gpa.free(row_all_true);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, row_all_true);

    var row_any_false_table = try table.withRowAnyFalse(&.{"active"}, "row_any_false");
    defer row_any_false_table.deinit();
    const row_any_false = try (try row_any_false_table.column("row_any_false")).bool.toOwnedSlice(gpa);
    defer gpa.free(row_any_false);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false }, row_any_false);

    var row_all_false_table = try table.withRowAllFalse(&.{"active"}, "row_all_false");
    defer row_all_false_table.deinit();
    const row_all_false = try (try row_all_false_table.column("row_all_false")).bool.toOwnedSlice(gpa);
    defer gpa.free(row_all_false);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false }, row_all_false);
    try std.testing.expectError(error.ColumnNotFound, table.withRowNullCount(&.{"missing"}, "bad_count"));
    try std.testing.expectError(error.TypeMismatch, table.withRowTrueCount(&.{"sales"}, "bad_bool_count"));
    try std.testing.expectError(error.TypeMismatch, table.withRowTrueRatio(&.{"sales"}, "bad_bool_ratio"));

    var signal_a = try DeviceColumn.fromSliceWithValidity(bool, gpa, &.{ false, true, false, true }, &.{ true, true, true, false }, .cpu);
    defer signal_a.deinit();
    var signal_b = try DeviceColumn.fromSliceWithValidity(bool, gpa, &.{ true, false, false, false }, &.{ true, false, true, true }, .cpu);
    defer signal_b.deinit();
    var signal_c = try DeviceColumn.fromSliceWithValidity(bool, gpa, &.{ false, true, false, true }, &.{ false, true, true, true }, .cpu);
    defer signal_c.deinit();
    var signal_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ 1.0, 2.0, 3.0, 4.0 }, .cpu);
    defer signal_metric.deinit();
    var signal_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "a", .data = signal_a },
        .{ .name = "b", .data = signal_b },
        .{ .name = "c", .data = signal_c },
        .{ .name = "metric", .data = signal_metric },
    });
    defer signal_table.deinit();

    var row_first_true_table = try signal_table.withRowFirstTrueIndex(&.{ "a", "b", "c" }, "first_true");
    defer row_first_true_table.deinit();
    const row_first_true_column = try row_first_true_table.column("first_true");
    try std.testing.expect(row_first_true_column.i64.nullable());
    const row_first_true = try row_first_true_column.i64.toOwnedSlice(gpa);
    defer gpa.free(row_first_true);
    const row_first_true_validity = try row_first_true_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_first_true_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 0, 2 }, row_first_true);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_first_true_validity);

    var row_last_true_table = try signal_table.withRowLastTrueIndex(&.{ "a", "b", "c" }, "last_true");
    defer row_last_true_table.deinit();
    const row_last_true_column = try row_last_true_table.column("last_true");
    const row_last_true = try row_last_true_column.i64.toOwnedSlice(gpa);
    defer gpa.free(row_last_true);
    const row_last_true_validity = try row_last_true_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_last_true_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 0, 2 }, row_last_true);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_last_true_validity);

    var row_first_false_table = try signal_table.withRowFirstFalseIndex(&.{ "a", "b", "c" }, "first_false");
    defer row_first_false_table.deinit();
    const row_first_false_column = try row_first_false_table.column("first_false");
    const row_first_false = try row_first_false_column.i64.toOwnedSlice(gpa);
    defer gpa.free(row_first_false);
    const row_first_false_validity = try row_first_false_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_first_false_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 1 }, row_first_false);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, true }, row_first_false_validity);

    var row_last_false_table = try signal_table.withRowLastFalseIndex(&.{ "a", "b", "c" }, "last_false");
    defer row_last_false_table.deinit();
    const row_last_false_column = try row_last_false_table.column("last_false");
    const row_last_false = try row_last_false_column.i64.toOwnedSlice(gpa);
    defer gpa.free(row_last_false);
    const row_last_false_validity = try row_last_false_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_last_false_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 2, 1 }, row_last_false);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, true }, row_last_false_validity);
    try std.testing.expectError(error.TypeMismatch, signal_table.withRowFirstTrueIndex(&.{"metric"}, "bad_bool_index"));

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

    var rolled = try table.rollRows(1);
    defer rolled.deinit();
    const rolled_sales = try (try rolled.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolled_sales);
    const rolled_units = try (try rolled.column("units")).i64.toOwnedSlice(gpa);
    defer gpa.free(rolled_units);
    const rolled_units_validity = try (try rolled.column("units")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(rolled_units_validity);
    try std.testing.expectEqualSlices(f64, &.{ 5.0, 2.0, 3.0 }, rolled_sales);
    try std.testing.expectEqualSlices(i64, &.{ 3, 1, 2 }, rolled_units);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false }, rolled_units_validity);

    var rolled_negative = try table.rollRows(-1);
    defer rolled_negative.deinit();
    const rolled_negative_sales = try (try rolled_negative.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolled_negative_sales);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 5.0, 2.0 }, rolled_negative_sales);

    var shifted = try table.shiftRows(1);
    defer shifted.deinit();
    const shifted_sales = try (try shifted.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(shifted_sales);
    const shifted_sales_validity = try (try shifted.column("sales")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(shifted_sales_validity);
    const shifted_units = try (try shifted.column("units")).i64.toOwnedSlice(gpa);
    defer gpa.free(shifted_units);
    const shifted_units_validity = try (try shifted.column("units")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(shifted_units_validity);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 2.0, 3.0 }, shifted_sales);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true }, shifted_sales_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 2 }, shifted_units);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false }, shifted_units_validity);

    var shifted_negative = try table.shiftRows(-1);
    defer shifted_negative.deinit();
    const shifted_negative_sales = try (try shifted_negative.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(shifted_negative_sales);
    const shifted_negative_sales_validity = try (try shifted_negative.column("sales")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(shifted_negative_sales_validity);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 5.0, 0.0 }, shifted_negative_sales);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false }, shifted_negative_sales_validity);

    var shifted_all = try table.shiftRows(10);
    defer shifted_all.deinit();
    const shifted_all_sales_validity = try (try shifted_all.column("sales")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(shifted_all_sales_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false }, shifted_all_sales_validity);

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

    var rows_dropped_wrap = try table.dropRowsMode(&.{table.height() + 1}, .wrap);
    defer rows_dropped_wrap.deinit();
    const rows_dropped_wrap_sales = try (try rows_dropped_wrap.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(rows_dropped_wrap_sales);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 5.0 }, rows_dropped_wrap_sales);
    try std.testing.expectError(error.IndexOutOfBounds, table.dropRowsMode(&.{table.height()}, .raise));

    var rows_dropped_signed = try table.dropRowsSigned(&.{-1});
    defer rows_dropped_signed.deinit();
    const rows_dropped_signed_sales = try (try rows_dropped_signed.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(rows_dropped_signed_sales);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 3.0 }, rows_dropped_signed_sales);

    var rows_dropped_signed_clip = try table.dropRowsSignedMode(&.{ -9, 9 }, .clip);
    defer rows_dropped_signed_clip.deinit();
    const rows_dropped_signed_clip_sales = try (try rows_dropped_signed_clip.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(rows_dropped_signed_clip_sales);
    try std.testing.expectEqualSlices(f64, &.{3.0}, rows_dropped_signed_clip_sales);

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

    var taken_signed = try table.takeSigned(&.{ -1, 0 });
    defer taken_signed.deinit();
    const taken_signed_sales = try (try taken_signed.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(taken_signed_sales);
    const taken_signed_units_validity = try (try taken_signed.column("units")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(taken_signed_units_validity);
    try std.testing.expectEqualSlices(f64, &.{ 5.0, 2.0 }, taken_signed_sales);
    try std.testing.expectEqualSlices(bool, &.{ true, true }, taken_signed_units_validity);
    try std.testing.expectError(error.IndexOutOfBounds, table.takeSigned(&.{-4}));

    var taken_wrap = try table.takeMode(&.{ table.height() + 1, 0 }, .wrap);
    defer taken_wrap.deinit();
    const taken_wrap_sales = try (try taken_wrap.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(taken_wrap_sales);
    const taken_wrap_units_validity = try (try taken_wrap.column("units")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(taken_wrap_units_validity);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 2.0 }, taken_wrap_sales);
    try std.testing.expectEqualSlices(bool, &.{ false, true }, taken_wrap_units_validity);
    try std.testing.expectError(error.IndexOutOfBounds, table.takeMode(&.{table.height()}, .raise));

    var taken_signed_clip = try table.takeSignedMode(&.{ -9, 9 }, .clip);
    defer taken_signed_clip.deinit();
    const taken_signed_clip_sales = try (try taken_signed_clip.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(taken_signed_clip_sales);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 5.0 }, taken_signed_clip_sales);

    var taken_optional = try table.takeOptional(&.{ 2, null, 1 });
    defer taken_optional.deinit();
    const taken_optional_sales = try (try taken_optional.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(taken_optional_sales);
    const taken_optional_sales_validity = try (try taken_optional.column("sales")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(taken_optional_sales_validity);
    const taken_optional_units = try (try taken_optional.column("units")).i64.toOwnedSlice(gpa);
    defer gpa.free(taken_optional_units);
    const taken_optional_units_validity = try (try taken_optional.column("units")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(taken_optional_units_validity);
    try std.testing.expectEqualSlices(f64, &.{ 5.0, 0.0, 3.0 }, taken_optional_sales);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, taken_optional_sales_validity);
    try std.testing.expectEqualSlices(i64, &.{ 3, 0, 2 }, taken_optional_units);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false }, taken_optional_units_validity);
    try std.testing.expectError(error.IndexOutOfBounds, table.takeOptional(&.{table.height()}));

    var row_pick = try DeviceColumn.fromSliceWithValidity(isize, gpa, &.{ 2, 0, -1 }, &.{ true, false, true }, .cpu);
    defer row_pick.deinit();
    var take_by_source = try DeviceDataFrame.init(gpa, &.{ .{ .name = "sales", .data = sales }, .{ .name = "units", .data = units }, .{ .name = "row_pick", .data = row_pick } });
    defer take_by_source.deinit();
    var taken_by_column = try take_by_source.takeByColumn("row_pick");
    defer taken_by_column.deinit();
    const taken_by_sales = try (try taken_by_column.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(taken_by_sales);
    const taken_by_sales_validity = try (try taken_by_column.column("sales")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(taken_by_sales_validity);
    const taken_by_units = try (try taken_by_column.column("units")).i64.toOwnedSlice(gpa);
    defer gpa.free(taken_by_units);
    const taken_by_units_validity = try (try taken_by_column.column("units")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(taken_by_units_validity);
    try std.testing.expectEqualSlices(f64, &.{ 5.0, 0.0, 5.0 }, taken_by_sales);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, taken_by_sales_validity);
    try std.testing.expectEqualSlices(i64, &.{ 3, 0, 3 }, taken_by_units);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, taken_by_units_validity);

    var row_pick_wrap = try DeviceColumn.fromSlice(usize, gpa, &.{ table.height() + 1, 0, 2 }, .cpu);
    defer row_pick_wrap.deinit();
    var take_by_wrap_source = try DeviceDataFrame.init(gpa, &.{ .{ .name = "sales", .data = sales }, .{ .name = "row_pick", .data = row_pick_wrap } });
    defer take_by_wrap_source.deinit();
    var taken_by_wrap = try take_by_wrap_source.takeByColumnMode("row_pick", .wrap);
    defer taken_by_wrap.deinit();
    const taken_by_wrap_sales = try (try taken_by_wrap.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(taken_by_wrap_sales);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 2.0, 5.0 }, taken_by_wrap_sales);
    try std.testing.expectError(error.TypeMismatch, table.takeByColumn("sales"));

    var row_pick_bad = try DeviceColumn.fromSlice(usize, gpa, &.{ table.height(), 0, 1 }, .cpu);
    defer row_pick_bad.deinit();
    var take_by_bad_source = try DeviceDataFrame.init(gpa, &.{ .{ .name = "sales", .data = sales }, .{ .name = "row_pick", .data = row_pick_bad } });
    defer take_by_bad_source.deinit();
    try std.testing.expectError(error.IndexOutOfBounds, take_by_bad_source.takeByColumn("row_pick"));

    var drop_pick = try DeviceColumn.fromSliceWithValidity(isize, gpa, &.{ 1, -1, 0 }, &.{ true, false, true }, .cpu);
    defer drop_pick.deinit();
    var drop_by_source = try DeviceDataFrame.init(gpa, &.{ .{ .name = "sales", .data = sales }, .{ .name = "units", .data = units }, .{ .name = "drop_pick", .data = drop_pick } });
    defer drop_by_source.deinit();
    var dropped_by_column = try drop_by_source.dropRowsByColumn("drop_pick");
    defer dropped_by_column.deinit();
    const dropped_by_sales = try (try dropped_by_column.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(dropped_by_sales);
    const dropped_by_units_validity = try (try dropped_by_column.column("units")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(dropped_by_units_validity);
    try std.testing.expectEqualSlices(f64, &.{5.0}, dropped_by_sales);
    try std.testing.expectEqualSlices(bool, &.{true}, dropped_by_units_validity);

    var drop_pick_wrap = try DeviceColumn.fromSlice(usize, gpa, &.{ table.height() + 1, table.height() + 1, table.height() + 1 }, .cpu);
    defer drop_pick_wrap.deinit();
    var drop_by_wrap_source = try DeviceDataFrame.init(gpa, &.{ .{ .name = "sales", .data = sales }, .{ .name = "drop_pick", .data = drop_pick_wrap } });
    defer drop_by_wrap_source.deinit();
    var dropped_by_wrap = try drop_by_wrap_source.dropRowsByColumnMode("drop_pick", .wrap);
    defer dropped_by_wrap.deinit();
    const dropped_by_wrap_sales = try (try dropped_by_wrap.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(dropped_by_wrap_sales);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 5.0 }, dropped_by_wrap_sales);
    try std.testing.expectError(error.TypeMismatch, table.dropRowsByColumn("sales"));
    try std.testing.expectError(error.IndexOutOfBounds, take_by_bad_source.dropRowsByColumn("row_pick"));

    var repeated_rows = try table.repeatRows(2);
    defer repeated_rows.deinit();
    try std.testing.expectEqual(@as(usize, 6), repeated_rows.height());
    const repeated_sales = try (try repeated_rows.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(repeated_sales);
    const repeated_units_validity = try (try repeated_rows.column("units")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(repeated_units_validity);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 2.0, 3.0, 3.0, 5.0, 5.0 }, repeated_sales);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false, true, true }, repeated_units_validity);

    var repeated_zero = try table.repeatRows(0);
    defer repeated_zero.deinit();
    try std.testing.expectEqual(@as(usize, 0), repeated_zero.height());
    try std.testing.expectEqual(table.width(), repeated_zero.width());

    var tiled_rows = try table.tileRows(2);
    defer tiled_rows.deinit();
    try std.testing.expectEqual(@as(usize, 6), tiled_rows.height());
    const tiled_sales = try (try tiled_rows.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(tiled_sales);
    const tiled_units_validity = try (try tiled_rows.column("units")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(tiled_units_validity);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 3.0, 5.0, 2.0, 3.0, 5.0 }, tiled_sales);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, true, false, true }, tiled_units_validity);

    var tiled_zero = try table.tileRows(0);
    defer tiled_zero.deinit();
    try std.testing.expectEqual(@as(usize, 0), tiled_zero.height());
    try std.testing.expectEqual(table.width(), tiled_zero.width());

    var repeat_counts = try DeviceColumn.fromSlice(usize, gpa, &.{ 1, 0, 2 }, .cpu);
    defer repeat_counts.deinit();
    var repeat_count_table = try DeviceDataFrame.init(gpa, &.{ .{ .name = "sales", .data = sales }, .{ .name = "units", .data = units }, .{ .name = "repeat_count", .data = repeat_counts } });
    defer repeat_count_table.deinit();
    var repeated_by = try repeat_count_table.repeatRowsByColumn("repeat_count");
    defer repeated_by.deinit();
    try std.testing.expectEqual(@as(usize, 3), repeated_by.height());
    const repeated_by_sales = try (try repeated_by.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(repeated_by_sales);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 5.0, 5.0 }, repeated_by_sales);
    try std.testing.expectError(error.TypeMismatch, table.repeatRowsByColumn("sales"));

    var negative_counts = try DeviceColumn.fromSlice(i64, gpa, &.{ 1, -1, 1 }, .cpu);
    defer negative_counts.deinit();
    var negative_count_table = try DeviceDataFrame.init(gpa, &.{ .{ .name = "sales", .data = sales }, .{ .name = "repeat_count", .data = negative_counts } });
    defer negative_count_table.deinit();
    try std.testing.expectError(error.InvalidShape, negative_count_table.repeatRowsByColumn("repeat_count"));

    var stepped_slice = try table.sliceRowsStep(0, table.height(), 2);
    defer stepped_slice.deinit();
    try std.testing.expectEqual(@as(usize, 2), stepped_slice.height());
    const stepped_slice_sales = try (try stepped_slice.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(stepped_slice_sales);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 5.0 }, stepped_slice_sales);

    var signed_slice = try table.sliceRowsSigned(-2, 2);
    defer signed_slice.deinit();
    const signed_slice_sales = try (try signed_slice.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(signed_slice_sales);
    const signed_slice_units_validity = try (try signed_slice.column("units")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(signed_slice_units_validity);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 5.0 }, signed_slice_sales);
    try std.testing.expectEqualSlices(bool, &.{ false, true }, signed_slice_units_validity);
    try std.testing.expectError(error.IndexOutOfBounds, table.sliceRowsSigned(-1, 2));

    var signed_stepped_slice = try table.sliceRowsSignedStep(-3, 3, 2);
    defer signed_stepped_slice.deinit();
    const signed_stepped_sales = try (try signed_stepped_slice.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(signed_stepped_sales);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 5.0 }, signed_stepped_sales);
    try std.testing.expectError(error.InvalidShape, table.sliceRowsSignedStep(-3, 3, 0));

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

test "device dataframe derives zero predicate columns" {
    const gpa = std.testing.allocator;

    var metric = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 0.0, -0.0, 3.0, std.math.nan(f64), std.math.inf(f64), -2.0 }, &.{ true, true, true, true, true, false }, .cpu);
    defer metric.deinit();
    var id = try DeviceColumn.fromSlice(i64, gpa, &.{ 0, 5, 0, -7, 9, 0 }, .cpu);
    defer id.deinit();
    var flag = try DeviceColumn.fromSlice(bool, gpa, &.{ false, true, false, true, true, false }, .cpu);
    defer flag.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "metric", .data = metric },
        .{ .name = "id", .data = id },
        .{ .name = "flag", .data = flag },
    });
    defer table.deinit();

    var zero_flags = try table.isZeroColumn("metric", "metric_is_zero");
    defer zero_flags.deinit();
    try std.testing.expectEqual(DeviceDType.bool, try zero_flags.columnDType("metric_is_zero"));
    const metric_is_zero = try (try zero_flags.column("metric_is_zero")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_is_zero);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false, false, false }, metric_is_zero);

    var non_zero_flags = try table.isNonZeroColumn("metric", "metric_is_non_zero");
    defer non_zero_flags.deinit();
    const metric_is_non_zero = try (try non_zero_flags.column("metric_is_non_zero")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_is_non_zero);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, true, true, false }, metric_is_non_zero);

    var id_zero_flags = try table.isZeroColumn("id", "id_is_zero");
    defer id_zero_flags.deinit();
    const id_is_zero = try (try id_zero_flags.column("id_is_zero")).bool.toOwnedSlice(gpa);
    defer gpa.free(id_is_zero);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, false, false, true }, id_is_zero);

    var flag_non_zero_flags = try table.isNonZeroColumn("flag", "flag_is_non_zero");
    defer flag_non_zero_flags.deinit();
    const flag_is_non_zero = try (try flag_non_zero_flags.column("flag_is_non_zero")).bool.toOwnedSlice(gpa);
    defer gpa.free(flag_is_non_zero);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true, true, false }, flag_is_non_zero);

    var row_zero_counts = try table.withRowZeroCount(&.{ "metric", "id", "flag" }, "row_zero_count");
    defer row_zero_counts.deinit();
    const row_zero_count = try (try row_zero_counts.column("row_zero_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_zero_count);
    try std.testing.expectEqualSlices(i64, &.{ 3, 1, 2, 0, 0, 2 }, row_zero_count);

    var row_non_zero_counts = try table.withRowNonZeroCount(&.{ "metric", "id", "flag" }, "row_non_zero_count");
    defer row_non_zero_counts.deinit();
    const row_non_zero_count = try (try row_non_zero_counts.column("row_non_zero_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_non_zero_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 2, 1, 3, 3, 0 }, row_non_zero_count);

    var row_zero_ratios = try table.withRowZeroRatio(&.{ "metric", "id", "flag" }, "row_zero_ratio");
    defer row_zero_ratios.deinit();
    const row_zero_ratio = try (try row_zero_ratios.column("row_zero_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_zero_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 1.0 / 3.0, 2.0 / 3.0, 0.0, 0.0, 1.0 }, row_zero_ratio);

    var row_non_zero_ratios = try table.withRowNonZeroRatio(&.{ "metric", "id", "flag" }, "row_non_zero_ratio");
    defer row_non_zero_ratios.deinit();
    const row_non_zero_ratio = try (try row_non_zero_ratios.column("row_non_zero_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_non_zero_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 2.0 / 3.0, 1.0 / 3.0, 1.0, 1.0, 0.0 }, row_non_zero_ratio);

    var metric_zero_ratios = try table.withRowZeroRatio(&.{"metric"}, "metric_zero_ratio");
    defer metric_zero_ratios.deinit();
    const metric_zero_ratio_column = try metric_zero_ratios.column("metric_zero_ratio");
    try std.testing.expect(metric_zero_ratio_column.f64.nullable());
    const metric_zero_ratio = try metric_zero_ratio_column.f64.toOwnedSlice(gpa);
    defer gpa.free(metric_zero_ratio);
    const metric_zero_ratio_validity = try metric_zero_ratio_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(metric_zero_ratio_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 1.0, 0.0, 0.0, 0.0, 0.0 }, metric_zero_ratio);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, true, false }, metric_zero_ratio_validity);

    var dropped_zero_rows = try table.dropZerosColumn("metric");
    defer dropped_zero_rows.deinit();
    try std.testing.expectEqual(@as(usize, 4), dropped_zero_rows.height());
    const dropped_zero_metric = try (try dropped_zero_rows.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(dropped_zero_metric);
    const dropped_zero_validity = try (try dropped_zero_rows.column("metric")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(dropped_zero_validity);
    try std.testing.expectEqual(@as(f64, 3.0), dropped_zero_metric[0]);
    try std.testing.expect(std.math.isNan(dropped_zero_metric[1]));
    try std.testing.expect(std.math.isPositiveInf(dropped_zero_metric[2]));
    try std.testing.expectEqual(@as(f64, -2.0), dropped_zero_metric[3]);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false }, dropped_zero_validity);

    var filtered_zero_rows = try table.filterZerosColumn("metric");
    defer filtered_zero_rows.deinit();
    try std.testing.expectEqual(@as(usize, 2), filtered_zero_rows.height());
    const filtered_zero_metric = try (try filtered_zero_rows.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(filtered_zero_metric);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, -0.0 }, filtered_zero_metric);

    var filtered_non_zero_rows = try table.filterNonZerosColumn("metric");
    defer filtered_non_zero_rows.deinit();
    try std.testing.expectEqual(@as(usize, 3), filtered_non_zero_rows.height());
    const filtered_non_zero_metric = try (try filtered_non_zero_rows.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(filtered_non_zero_metric);
    try std.testing.expectEqual(@as(f64, 3.0), filtered_non_zero_metric[0]);
    try std.testing.expect(std.math.isNan(filtered_non_zero_metric[1]));
    try std.testing.expect(std.math.isPositiveInf(filtered_non_zero_metric[2]));

    try std.testing.expectError(error.ColumnNotFound, table.isZeroColumn("missing", "missing_is_zero"));
    try std.testing.expectError(error.ColumnNotFound, table.isNonZeroColumn("missing", "missing_is_non_zero"));
    try std.testing.expectError(error.ColumnNotFound, table.withRowZeroCount(&.{"missing"}, "bad_zero_count"));
    try std.testing.expectError(error.ColumnNotFound, table.filterZerosColumn("missing"));
    try std.testing.expectError(error.ColumnNotFound, table.dropNonZerosColumn("missing"));
}

test "device dataframe derives sign predicate columns" {
    const gpa = std.testing.allocator;

    var metric = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ -2.0, -0.0, 0.0, 3.0, std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64), 9.0 }, &.{ true, true, true, true, true, true, true, false }, .cpu);
    defer metric.deinit();
    var id = try DeviceColumn.fromSlice(i64, gpa, &.{ -3, 0, 4, -5, 6, 0, -7, 8 }, .cpu);
    defer id.deinit();
    var unsigned = try DeviceColumn.fromSlice(u64, gpa, &.{ 0, 2, 0, 5, 0, 9, 11, 0 }, .cpu);
    defer unsigned.deinit();
    var flag = try DeviceColumn.fromSlice(bool, gpa, &.{ false, true, false, true, true, false, true, false }, .cpu);
    defer flag.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "metric", .data = metric },
        .{ .name = "id", .data = id },
        .{ .name = "unsigned", .data = unsigned },
        .{ .name = "flag", .data = flag },
    });
    defer table.deinit();

    var positive_flags = try table.isPositiveColumn("metric", "metric_is_positive");
    defer positive_flags.deinit();
    try std.testing.expectEqual(DeviceDType.bool, try positive_flags.columnDType("metric_is_positive"));
    const metric_is_positive = try (try positive_flags.column("metric_is_positive")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_is_positive);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, true, false, true, false, false }, metric_is_positive);

    var negative_flags = try table.isNegativeColumn("metric", "metric_is_negative");
    defer negative_flags.deinit();
    const metric_is_negative = try (try negative_flags.column("metric_is_negative")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_is_negative);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false, false, false, true, false }, metric_is_negative);

    var signbit_flags = try table.isSignBitColumn("metric", "metric_signbit");
    defer signbit_flags.deinit();
    try std.testing.expectEqual(DeviceDType.bool, try signbit_flags.columnDType("metric_signbit"));
    const metric_signbit = try (try signbit_flags.column("metric_signbit")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_signbit);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false, false, false, true, false }, metric_signbit);

    var id_signbit_flags = try table.isSignBitColumn("id", "id_signbit");
    defer id_signbit_flags.deinit();
    const id_signbit = try (try id_signbit_flags.column("id_signbit")).bool.toOwnedSlice(gpa);
    defer gpa.free(id_signbit);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true, false, false, true, false }, id_signbit);

    var positive_zero_flags = try table.isPositiveZeroColumn("metric", "metric_is_positive_zero");
    defer positive_zero_flags.deinit();
    try std.testing.expectEqual(DeviceDType.bool, try positive_zero_flags.columnDType("metric_is_positive_zero"));
    const metric_is_positive_zero = try (try positive_zero_flags.column("metric_is_positive_zero")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_is_positive_zero);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, false, false, false, false }, metric_is_positive_zero);

    var negative_zero_flags = try table.isNegativeZeroColumn("metric", "metric_is_negative_zero");
    defer negative_zero_flags.deinit();
    const metric_is_negative_zero = try (try negative_zero_flags.column("metric_is_negative_zero")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_is_negative_zero);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false, false, false, false, false }, metric_is_negative_zero);

    var id_positive_flags = try table.isPositiveColumn("id", "id_is_positive");
    defer id_positive_flags.deinit();
    const id_is_positive = try (try id_positive_flags.column("id_is_positive")).bool.toOwnedSlice(gpa);
    defer gpa.free(id_is_positive);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, true, false, false, true }, id_is_positive);

    var unsigned_negative_flags = try table.isNegativeColumn("unsigned", "unsigned_is_negative");
    defer unsigned_negative_flags.deinit();
    const unsigned_is_negative = try (try unsigned_negative_flags.column("unsigned_is_negative")).bool.toOwnedSlice(gpa);
    defer gpa.free(unsigned_is_negative);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false, false, false, false, false }, unsigned_is_negative);

    var bool_positive_flags = try table.isPositiveColumn("flag", "flag_is_positive");
    defer bool_positive_flags.deinit();
    const flag_is_positive = try (try bool_positive_flags.column("flag_is_positive")).bool.toOwnedSlice(gpa);
    defer gpa.free(flag_is_positive);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false, false, false, false, false }, flag_is_positive);

    var row_positive_zero_counts = try table.withRowPositiveZeroCount(&.{ "metric", "id", "unsigned", "flag" }, "row_positive_zero_count");
    defer row_positive_zero_counts.deinit();
    const row_positive_zero_count = try (try row_positive_zero_counts.column("row_positive_zero_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_positive_zero_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 0, 0, 0, 0, 0 }, row_positive_zero_count);

    var row_negative_zero_counts = try table.withRowNegativeZeroCount(&.{ "metric", "id", "unsigned", "flag" }, "row_negative_zero_count");
    defer row_negative_zero_counts.deinit();
    const row_negative_zero_count = try (try row_negative_zero_counts.column("row_negative_zero_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_negative_zero_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 0, 0, 0, 0, 0 }, row_negative_zero_count);

    var row_positive_zero_ratios = try table.withRowPositiveZeroRatio(&.{ "metric", "id", "unsigned", "flag" }, "row_positive_zero_ratio");
    defer row_positive_zero_ratios.deinit();
    const row_positive_zero_ratio = try (try row_positive_zero_ratios.column("row_positive_zero_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_positive_zero_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0 }, row_positive_zero_ratio);

    var row_negative_zero_ratios = try table.withRowNegativeZeroRatio(&.{ "metric", "id", "unsigned", "flag" }, "row_negative_zero_ratio");
    defer row_negative_zero_ratios.deinit();
    const row_negative_zero_ratio = try (try row_negative_zero_ratios.column("row_negative_zero_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_negative_zero_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, row_negative_zero_ratio);

    var row_positive_counts = try table.withRowPositiveCount(&.{ "metric", "id", "unsigned", "flag" }, "row_positive_count");
    defer row_positive_counts.deinit();
    const row_positive_count = try (try row_positive_counts.column("row_positive_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_positive_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 1, 2, 1, 2, 1, 1 }, row_positive_count);

    var row_signbit_counts = try table.withRowSignBitCount(&.{ "metric", "id", "unsigned", "flag" }, "row_signbit_count");
    defer row_signbit_counts.deinit();
    const row_signbit_count = try (try row_signbit_counts.column("row_signbit_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_signbit_count);
    try std.testing.expectEqualSlices(i64, &.{ 2, 1, 0, 1, 0, 0, 2, 0 }, row_signbit_count);

    var row_negative_counts = try table.withRowNegativeCount(&.{ "metric", "id", "unsigned", "flag" }, "row_negative_count");
    defer row_negative_counts.deinit();
    const row_negative_count = try (try row_negative_counts.column("row_negative_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_negative_count);
    try std.testing.expectEqualSlices(i64, &.{ 2, 0, 0, 1, 0, 0, 2, 0 }, row_negative_count);

    var row_positive_ratios = try table.withRowPositiveRatio(&.{ "metric", "id", "unsigned", "flag" }, "row_positive_ratio");
    defer row_positive_ratios.deinit();
    const row_positive_ratio = try (try row_positive_ratios.column("row_positive_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_positive_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.25, 0.25, 0.5, 0.25, 0.5, 0.25, 1.0 / 3.0 }, row_positive_ratio);

    var row_signbit_ratios = try table.withRowSignBitRatio(&.{ "metric", "id", "unsigned", "flag" }, "row_signbit_ratio");
    defer row_signbit_ratios.deinit();
    const row_signbit_ratio = try (try row_signbit_ratios.column("row_signbit_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_signbit_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 0.5, 0.25, 0.0, 0.25, 0.0, 0.0, 0.5, 0.0 }, row_signbit_ratio);

    var row_negative_ratios = try table.withRowNegativeRatio(&.{ "metric", "id", "unsigned", "flag" }, "row_negative_ratio");
    defer row_negative_ratios.deinit();
    const row_negative_ratio = try (try row_negative_ratios.column("row_negative_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_negative_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 0.5, 0.0, 0.0, 0.25, 0.0, 0.0, 0.5, 0.0 }, row_negative_ratio);

    var dropped_positive_rows = try table.dropPositivesColumn("metric");
    defer dropped_positive_rows.deinit();
    try std.testing.expectEqual(@as(usize, 6), dropped_positive_rows.height());
    const dropped_positive_metric = try (try dropped_positive_rows.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(dropped_positive_metric);
    const dropped_positive_validity = try (try dropped_positive_rows.column("metric")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(dropped_positive_validity);
    try std.testing.expectEqual(@as(f64, -2.0), dropped_positive_metric[0]);
    try std.testing.expectEqual(@as(f64, -0.0), dropped_positive_metric[1]);
    try std.testing.expectEqual(@as(f64, 0.0), dropped_positive_metric[2]);
    try std.testing.expect(std.math.isNan(dropped_positive_metric[3]));
    try std.testing.expect(std.math.isNegativeInf(dropped_positive_metric[4]));
    try std.testing.expectEqual(@as(f64, 9.0), dropped_positive_metric[5]);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, true, false }, dropped_positive_validity);

    var filtered_positive_rows = try table.filterPositivesColumn("metric");
    defer filtered_positive_rows.deinit();
    try std.testing.expectEqual(@as(usize, 2), filtered_positive_rows.height());
    const filtered_positive_metric = try (try filtered_positive_rows.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(filtered_positive_metric);
    try std.testing.expectEqual(@as(f64, 3.0), filtered_positive_metric[0]);
    try std.testing.expect(std.math.isPositiveInf(filtered_positive_metric[1]));

    var filtered_signbit_rows = try table.filterSignBitsColumn("metric");
    defer filtered_signbit_rows.deinit();
    try std.testing.expectEqual(@as(usize, 3), filtered_signbit_rows.height());
    const filtered_signbit_metric = try (try filtered_signbit_rows.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(filtered_signbit_metric);
    try std.testing.expectEqual(@as(f64, -2.0), filtered_signbit_metric[0]);
    try std.testing.expectEqual(@as(f64, -0.0), filtered_signbit_metric[1]);
    try std.testing.expect(std.math.isNegativeInf(filtered_signbit_metric[2]));

    var filtered_positive_zero_rows = try table.filterPositiveZerosColumn("metric");
    defer filtered_positive_zero_rows.deinit();
    try std.testing.expectEqual(@as(usize, 1), filtered_positive_zero_rows.height());
    const filtered_positive_zero_metric = try (try filtered_positive_zero_rows.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(filtered_positive_zero_metric);
    try std.testing.expectEqual(@as(f64, 0.0), filtered_positive_zero_metric[0]);

    var dropped_negative_zero_rows = try table.dropNegativeZerosColumn("metric");
    defer dropped_negative_zero_rows.deinit();
    try std.testing.expectEqual(@as(usize, 7), dropped_negative_zero_rows.height());
    const dropped_negative_zero_metric = try (try dropped_negative_zero_rows.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(dropped_negative_zero_metric);
    try std.testing.expectEqual(@as(f64, -2.0), dropped_negative_zero_metric[0]);
    try std.testing.expectEqual(@as(f64, 0.0), dropped_negative_zero_metric[1]);
    try std.testing.expectEqual(@as(f64, 3.0), dropped_negative_zero_metric[2]);
    try std.testing.expect(std.math.isNan(dropped_negative_zero_metric[3]));
    try std.testing.expect(std.math.isPositiveInf(dropped_negative_zero_metric[4]));
    try std.testing.expect(std.math.isNegativeInf(dropped_negative_zero_metric[5]));
    try std.testing.expectEqual(@as(f64, 9.0), dropped_negative_zero_metric[6]);

    var filtered_negative_rows = try table.filterNegativesColumn("id");
    defer filtered_negative_rows.deinit();
    try std.testing.expectEqual(@as(usize, 3), filtered_negative_rows.height());
    const filtered_negative_id = try (try filtered_negative_rows.column("id")).i64.toOwnedSlice(gpa);
    defer gpa.free(filtered_negative_id);
    try std.testing.expectEqualSlices(i64, &.{ -3, -5, -7 }, filtered_negative_id);

    try std.testing.expectError(error.ColumnNotFound, table.isPositiveColumn("missing", "missing_is_positive"));
    try std.testing.expectError(error.ColumnNotFound, table.isNegativeColumn("missing", "missing_is_negative"));
    try std.testing.expectError(error.ColumnNotFound, table.isSignBitColumn("missing", "missing_signbit"));
    try std.testing.expectError(error.ColumnNotFound, table.isPositiveZeroColumn("missing", "missing_is_positive_zero"));
    try std.testing.expectError(error.ColumnNotFound, table.isNegativeZeroColumn("missing", "missing_is_negative_zero"));
    try std.testing.expectError(error.ColumnNotFound, table.withRowPositiveZeroCount(&.{"missing"}, "bad_positive_zero_count"));
    try std.testing.expectError(error.ColumnNotFound, table.withRowPositiveCount(&.{"missing"}, "bad_positive_count"));
    try std.testing.expectError(error.ColumnNotFound, table.withRowSignBitCount(&.{"missing"}, "bad_signbit_count"));
    try std.testing.expectError(error.ColumnNotFound, table.filterPositiveZerosColumn("missing"));
    try std.testing.expectError(error.ColumnNotFound, table.dropNegativeZerosColumn("missing"));
    try std.testing.expectError(error.ColumnNotFound, table.filterPositivesColumn("missing"));
    try std.testing.expectError(error.ColumnNotFound, table.filterSignBitsColumn("missing"));
    try std.testing.expectError(error.ColumnNotFound, table.dropSignBitsColumn("missing"));
    try std.testing.expectError(error.ColumnNotFound, table.dropNegativesColumn("missing"));
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

    var non_finite_flags = try table.isNonFiniteColumn("metric", "metric_is_non_finite");
    defer non_finite_flags.deinit();
    const metric_is_non_finite = try (try non_finite_flags.column("metric_is_non_finite")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_is_non_finite);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, false }, metric_is_non_finite);

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

    var integer_non_finite_flags = try table.isNonFiniteColumn("id", "id_is_non_finite");
    defer integer_non_finite_flags.deinit();
    const id_is_non_finite = try (try integer_non_finite_flags.column("id_is_non_finite")).bool.toOwnedSlice(gpa);
    defer gpa.free(id_is_non_finite);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false }, id_is_non_finite);
    try std.testing.expectError(error.ColumnNotFound, table.isNanColumn("missing", "missing_is_nan"));
    try std.testing.expectError(error.ColumnNotFound, table.isNonFiniteColumn("missing", "missing_is_non_finite"));

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

    var dropped_finite_rows = try table.dropFinitesColumn("metric");
    defer dropped_finite_rows.deinit();
    try std.testing.expectEqual(@as(usize, 3), dropped_finite_rows.height());
    const dropped_finite_metric = try (try dropped_finite_rows.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(dropped_finite_metric);
    const dropped_finite_validity = try (try dropped_finite_rows.column("metric")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(dropped_finite_validity);
    try std.testing.expect(std.math.isNan(dropped_finite_metric[0]));
    try std.testing.expect(std.math.isInf(dropped_finite_metric[1]));
    try std.testing.expectEqual(@as(f64, 7.0), dropped_finite_metric[2]);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false }, dropped_finite_validity);

    var filtered_finite_rows = try table.filterFinitesColumn("metric");
    defer filtered_finite_rows.deinit();
    try std.testing.expectEqual(@as(usize, 1), filtered_finite_rows.height());
    const filtered_finite_metric = try (try filtered_finite_rows.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(filtered_finite_metric);
    try std.testing.expectEqual(@as(f64, 1.0), filtered_finite_metric[0]);

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
    try std.testing.expectError(error.ColumnNotFound, table.dropFinitesColumn("missing"));
    try std.testing.expectError(error.ColumnNotFound, table.filterFinitesColumn("missing"));
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

    var row_nan_ratios = try table.withRowNaNRatio(&.{ "metric", "id" }, "row_nan_ratio");
    defer row_nan_ratios.deinit();
    const row_nan_ratio = try (try row_nan_ratios.column("row_nan_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_nan_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.5, 0.0, 0.0 }, row_nan_ratio);

    var row_inf_ratios = try table.withRowInfRatio(&.{}, "row_inf_ratio");
    defer row_inf_ratios.deinit();
    const row_inf_ratio = try (try row_inf_ratios.column("row_inf_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_inf_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.5, 0.0 }, row_inf_ratio);

    var row_finite_ratios = try table.withRowFiniteRatio(&.{ "metric", "id" }, "row_finite_ratio");
    defer row_finite_ratios.deinit();
    const row_finite_ratio = try (try row_finite_ratios.column("row_finite_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_finite_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 0.5, 0.5, 1.0 }, row_finite_ratio);

    var row_non_finite_ratios = try table.withRowNonFiniteRatio(&.{}, "row_non_finite_ratio");
    defer row_non_finite_ratios.deinit();
    const row_non_finite_ratio = try (try row_non_finite_ratios.column("row_non_finite_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_non_finite_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.5, 0.5, 0.0 }, row_non_finite_ratio);

    var metric_nan_ratios = try table.withRowNanRatio(&.{"metric"}, "metric_nan_ratio");
    defer metric_nan_ratios.deinit();
    const metric_nan_ratio_column = try metric_nan_ratios.column("metric_nan_ratio");
    try std.testing.expect(metric_nan_ratio_column.f64.nullable());
    const metric_nan_ratio = try metric_nan_ratio_column.f64.toOwnedSlice(gpa);
    defer gpa.free(metric_nan_ratio);
    const metric_nan_ratio_validity = try metric_nan_ratio_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(metric_nan_ratio_validity);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 1.0, 0.0, 0.0 }, metric_nan_ratio);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false }, metric_nan_ratio_validity);
    try std.testing.expectError(error.ColumnNotFound, table.withRowNaNCount(&.{"missing"}, "bad_count"));
    try std.testing.expectError(error.ColumnNotFound, table.withRowNaNRatio(&.{"missing"}, "bad_ratio"));
}

test "device dataframe selects zero columns" {
    const gpa = std.testing.allocator;

    var zero_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ 0.0, -0.0, 0.0 }, .cpu);
    defer zero_metric.deinit();
    var mixed_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ 0.0, 4.0, std.math.nan(f64) }, .cpu);
    defer mixed_metric.deinit();
    var non_zero_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ 1.0, std.math.nan(f64), std.math.inf(f64) }, .cpu);
    defer non_zero_metric.deinit();
    var null_metric = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 0.0, 0.0, 0.0 }, &.{ false, false, false }, .cpu);
    defer null_metric.deinit();
    var id = try DeviceColumn.fromSlice(i64, gpa, &.{ 0, 5, 0 }, .cpu);
    defer id.deinit();
    var flag = try DeviceColumn.fromSlice(bool, gpa, &.{ false, true, false }, .cpu);
    defer flag.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "zero_metric", .data = zero_metric },
        .{ .name = "mixed_metric", .data = mixed_metric },
        .{ .name = "non_zero_metric", .data = non_zero_metric },
        .{ .name = "null_metric", .data = null_metric },
        .{ .name = "id", .data = id },
        .{ .name = "flag", .data = flag },
    });
    defer table.deinit();

    var with_zeros = try table.selectColumnsWithZeros();
    defer with_zeros.deinit();
    try std.testing.expectEqual(@as(usize, 4), with_zeros.width());
    try std.testing.expectEqual(@as(?usize, 0), with_zeros.columnIndex("zero_metric"));
    try std.testing.expectEqual(@as(?usize, 1), with_zeros.columnIndex("mixed_metric"));
    try std.testing.expectEqual(@as(?usize, 2), with_zeros.columnIndex("id"));
    try std.testing.expectEqual(@as(?usize, 3), with_zeros.columnIndex("flag"));

    var without_zeros = try table.selectColumnsWithoutZeros();
    defer without_zeros.deinit();
    try std.testing.expectEqual(@as(usize, 2), without_zeros.width());
    try std.testing.expectEqual(@as(?usize, 0), without_zeros.columnIndex("non_zero_metric"));
    try std.testing.expectEqual(@as(?usize, 1), without_zeros.columnIndex("null_metric"));

    var with_non_zeros = try table.selectColumnsWithNonZeros();
    defer with_non_zeros.deinit();
    try std.testing.expectEqual(@as(usize, 4), with_non_zeros.width());
    try std.testing.expectEqual(@as(?usize, 0), with_non_zeros.columnIndex("mixed_metric"));
    try std.testing.expectEqual(@as(?usize, 1), with_non_zeros.columnIndex("non_zero_metric"));
    try std.testing.expectEqual(@as(?usize, 2), with_non_zeros.columnIndex("id"));
    try std.testing.expectEqual(@as(?usize, 3), with_non_zeros.columnIndex("flag"));

    var drop_without_non_zeros = try table.dropColumnsWithoutNonZeros();
    defer drop_without_non_zeros.deinit();
    try std.testing.expectEqual(@as(usize, 4), drop_without_non_zeros.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_without_non_zeros.columnIndex("mixed_metric"));
    try std.testing.expectEqual(@as(?usize, 1), drop_without_non_zeros.columnIndex("non_zero_metric"));
    try std.testing.expectEqual(@as(?usize, 2), drop_without_non_zeros.columnIndex("id"));
    try std.testing.expectEqual(@as(?usize, 3), drop_without_non_zeros.columnIndex("flag"));

    var drop_with_zeros = try table.dropColumnsWithZeros();
    defer drop_with_zeros.deinit();
    try std.testing.expectEqual(@as(usize, 2), drop_with_zeros.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_with_zeros.columnIndex("non_zero_metric"));
    try std.testing.expectEqual(@as(?usize, 1), drop_with_zeros.columnIndex("null_metric"));

    var with_positive_zeros = try table.selectColumnsWithPositiveZeros();
    defer with_positive_zeros.deinit();
    try std.testing.expectEqual(@as(usize, 2), with_positive_zeros.width());
    try std.testing.expectEqual(@as(?usize, 0), with_positive_zeros.columnIndex("zero_metric"));
    try std.testing.expectEqual(@as(?usize, 1), with_positive_zeros.columnIndex("mixed_metric"));

    var with_negative_zeros = try table.selectColumnsWithNegativeZeros();
    defer with_negative_zeros.deinit();
    try std.testing.expectEqual(@as(usize, 1), with_negative_zeros.width());
    try std.testing.expectEqual(@as(?usize, 0), with_negative_zeros.columnIndex("zero_metric"));

    var drop_without_negative_zeros = try table.dropColumnsWithoutNegativeZeros();
    defer drop_without_negative_zeros.deinit();
    try std.testing.expectEqual(@as(usize, 1), drop_without_negative_zeros.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_without_negative_zeros.columnIndex("zero_metric"));
}

test "device dataframe selects sign columns" {
    const gpa = std.testing.allocator;

    var positive_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ 1.0, std.math.inf(f64), 3.0 }, .cpu);
    defer positive_metric.deinit();
    var negative_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ -1.0, -std.math.inf(f64), -3.0 }, .cpu);
    defer negative_metric.deinit();
    var mixed_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ -1.0, 0.0, 2.0 }, .cpu);
    defer mixed_metric.deinit();
    var zero_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ 0.0, -0.0, 0.0 }, .cpu);
    defer zero_metric.deinit();
    var null_metric = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ -1.0, 2.0, -3.0 }, &.{ false, false, false }, .cpu);
    defer null_metric.deinit();
    var id = try DeviceColumn.fromSlice(i64, gpa, &.{ -1, 0, 3 }, .cpu);
    defer id.deinit();
    var unsigned = try DeviceColumn.fromSlice(u64, gpa, &.{ 0, 4, 0 }, .cpu);
    defer unsigned.deinit();
    var flag = try DeviceColumn.fromSlice(bool, gpa, &.{ true, false, true }, .cpu);
    defer flag.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "positive_metric", .data = positive_metric },
        .{ .name = "negative_metric", .data = negative_metric },
        .{ .name = "mixed_metric", .data = mixed_metric },
        .{ .name = "zero_metric", .data = zero_metric },
        .{ .name = "null_metric", .data = null_metric },
        .{ .name = "id", .data = id },
        .{ .name = "unsigned", .data = unsigned },
        .{ .name = "flag", .data = flag },
    });
    defer table.deinit();

    var with_positives = try table.selectColumnsWithPositives();
    defer with_positives.deinit();
    try std.testing.expectEqual(@as(usize, 4), with_positives.width());
    try std.testing.expectEqual(@as(?usize, 0), with_positives.columnIndex("positive_metric"));
    try std.testing.expectEqual(@as(?usize, 1), with_positives.columnIndex("mixed_metric"));
    try std.testing.expectEqual(@as(?usize, 2), with_positives.columnIndex("id"));
    try std.testing.expectEqual(@as(?usize, 3), with_positives.columnIndex("unsigned"));

    var without_positives = try table.selectColumnsWithoutPositives();
    defer without_positives.deinit();
    try std.testing.expectEqual(@as(usize, 4), without_positives.width());
    try std.testing.expectEqual(@as(?usize, 0), without_positives.columnIndex("negative_metric"));
    try std.testing.expectEqual(@as(?usize, 1), without_positives.columnIndex("zero_metric"));
    try std.testing.expectEqual(@as(?usize, 2), without_positives.columnIndex("null_metric"));
    try std.testing.expectEqual(@as(?usize, 3), without_positives.columnIndex("flag"));

    var with_signbits = try table.selectColumnsWithSignBits();
    defer with_signbits.deinit();
    try std.testing.expectEqual(@as(usize, 4), with_signbits.width());
    try std.testing.expectEqual(@as(?usize, 0), with_signbits.columnIndex("negative_metric"));
    try std.testing.expectEqual(@as(?usize, 1), with_signbits.columnIndex("mixed_metric"));
    try std.testing.expectEqual(@as(?usize, 2), with_signbits.columnIndex("zero_metric"));
    try std.testing.expectEqual(@as(?usize, 3), with_signbits.columnIndex("id"));

    var with_negatives = try table.selectColumnsWithNegatives();
    defer with_negatives.deinit();
    try std.testing.expectEqual(@as(usize, 3), with_negatives.width());
    try std.testing.expectEqual(@as(?usize, 0), with_negatives.columnIndex("negative_metric"));
    try std.testing.expectEqual(@as(?usize, 1), with_negatives.columnIndex("mixed_metric"));
    try std.testing.expectEqual(@as(?usize, 2), with_negatives.columnIndex("id"));

    var without_negatives = try table.selectColumnsWithoutNegatives();
    defer without_negatives.deinit();
    try std.testing.expectEqual(@as(usize, 5), without_negatives.width());
    try std.testing.expectEqual(@as(?usize, 0), without_negatives.columnIndex("positive_metric"));
    try std.testing.expectEqual(@as(?usize, 1), without_negatives.columnIndex("zero_metric"));
    try std.testing.expectEqual(@as(?usize, 2), without_negatives.columnIndex("null_metric"));
    try std.testing.expectEqual(@as(?usize, 3), without_negatives.columnIndex("unsigned"));
    try std.testing.expectEqual(@as(?usize, 4), without_negatives.columnIndex("flag"));

    var drop_with_positives = try table.dropColumnsWithPositives();
    defer drop_with_positives.deinit();
    try std.testing.expectEqual(@as(usize, 4), drop_with_positives.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_with_positives.columnIndex("negative_metric"));
    try std.testing.expectEqual(@as(?usize, 1), drop_with_positives.columnIndex("zero_metric"));
    try std.testing.expectEqual(@as(?usize, 2), drop_with_positives.columnIndex("null_metric"));
    try std.testing.expectEqual(@as(?usize, 3), drop_with_positives.columnIndex("flag"));

    var drop_without_signbits = try table.dropColumnsWithoutSignBits();
    defer drop_without_signbits.deinit();
    try std.testing.expectEqual(@as(usize, 4), drop_without_signbits.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_without_signbits.columnIndex("negative_metric"));
    try std.testing.expectEqual(@as(?usize, 1), drop_without_signbits.columnIndex("mixed_metric"));
    try std.testing.expectEqual(@as(?usize, 2), drop_without_signbits.columnIndex("zero_metric"));
    try std.testing.expectEqual(@as(?usize, 3), drop_without_signbits.columnIndex("id"));

    var drop_without_negatives = try table.dropColumnsWithoutNegatives();
    defer drop_without_negatives.deinit();
    try std.testing.expectEqual(@as(usize, 3), drop_without_negatives.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_without_negatives.columnIndex("negative_metric"));
    try std.testing.expectEqual(@as(?usize, 1), drop_without_negatives.columnIndex("mixed_metric"));
    try std.testing.expectEqual(@as(?usize, 2), drop_without_negatives.columnIndex("id"));
}

test "device dataframe selects finite columns" {
    const gpa = std.testing.allocator;

    var finite_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ 1.0, 2.0, 3.0 }, .cpu);
    defer finite_metric.deinit();
    var mixed_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ std.math.nan(f64), 4.0, std.math.inf(f64) }, .cpu);
    defer mixed_metric.deinit();
    var non_finite_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64) }, .cpu);
    defer non_finite_metric.deinit();
    var null_metric = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 8.0, 9.0, 10.0 }, &.{ false, false, false }, .cpu);
    defer null_metric.deinit();
    var id = try DeviceColumn.fromSlice(i64, gpa, &.{ 10, 20, 30 }, .cpu);
    defer id.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "finite_metric", .data = finite_metric },
        .{ .name = "mixed_metric", .data = mixed_metric },
        .{ .name = "non_finite_metric", .data = non_finite_metric },
        .{ .name = "null_metric", .data = null_metric },
        .{ .name = "id", .data = id },
    });
    defer table.deinit();

    var with_finites = try table.selectColumnsWithFinites();
    defer with_finites.deinit();
    try std.testing.expectEqual(@as(usize, 3), with_finites.width());
    try std.testing.expectEqual(@as(?usize, 0), with_finites.columnIndex("finite_metric"));
    try std.testing.expectEqual(@as(?usize, 1), with_finites.columnIndex("mixed_metric"));
    try std.testing.expectEqual(@as(?usize, 2), with_finites.columnIndex("id"));

    var without_finites = try table.selectColumnsWithoutFinites();
    defer without_finites.deinit();
    try std.testing.expectEqual(@as(usize, 2), without_finites.width());
    try std.testing.expectEqual(@as(?usize, 0), without_finites.columnIndex("non_finite_metric"));
    try std.testing.expectEqual(@as(?usize, 1), without_finites.columnIndex("null_metric"));

    var drop_with_finites = try table.dropColumnsWithFinites();
    defer drop_with_finites.deinit();
    try std.testing.expectEqual(@as(usize, 2), drop_with_finites.width());
    try std.testing.expectEqual(@as(?usize, null), drop_with_finites.columnIndex("finite_metric"));
    try std.testing.expectEqual(@as(?usize, null), drop_with_finites.columnIndex("mixed_metric"));
    try std.testing.expectEqual(@as(?usize, null), drop_with_finites.columnIndex("id"));

    var drop_without_finites = try table.dropColumnsWithoutFinites();
    defer drop_without_finites.deinit();
    try std.testing.expectEqual(@as(usize, 3), drop_without_finites.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_without_finites.columnIndex("finite_metric"));
    try std.testing.expectEqual(@as(?usize, 1), drop_without_finites.columnIndex("mixed_metric"));
    try std.testing.expectEqual(@as(?usize, 2), drop_without_finites.columnIndex("id"));
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

    var metric_subnormal_flags = try table.isSubnormalColumn("metric", "metric_is_subnormal");
    defer metric_subnormal_flags.deinit();
    try std.testing.expectEqual(DeviceDType.bool, try metric_subnormal_flags.columnDType("metric_is_subnormal"));
    const metric_is_subnormal = try (try metric_subnormal_flags.column("metric_is_subnormal")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_is_subnormal);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, false }, metric_is_subnormal);

    var integer_flags = try table.isNormalColumn("id", "id_is_normal");
    defer integer_flags.deinit();
    const id_is_normal = try (try integer_flags.column("id_is_normal")).bool.toOwnedSlice(gpa);
    defer gpa.free(id_is_normal);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false, false }, id_is_normal);

    var integer_subnormal_flags = try table.isSubnormalColumn("id", "id_is_subnormal");
    defer integer_subnormal_flags.deinit();
    const id_is_subnormal = try (try integer_subnormal_flags.column("id_is_subnormal")).bool.toOwnedSlice(gpa);
    defer gpa.free(id_is_subnormal);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false, false }, id_is_subnormal);

    var row_normal_counts = try table.withRowNormalCount(&.{ "metric", "id" }, "row_normal_count");
    defer row_normal_counts.deinit();
    const row_normal_count = try (try row_normal_counts.column("row_normal_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_normal_count);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 0, 0, 0 }, row_normal_count);

    var row_subnormal_counts = try table.withRowSubnormalCount(&.{ "metric", "id" }, "row_subnormal_count");
    defer row_subnormal_counts.deinit();
    const row_subnormal_count = try (try row_subnormal_counts.column("row_subnormal_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_subnormal_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 0, 0 }, row_subnormal_count);

    var row_normal_ratios = try table.withRowNormalRatio(&.{ "metric", "id" }, "row_normal_ratio");
    defer row_normal_ratios.deinit();
    const row_normal_ratio = try (try row_normal_ratios.column("row_normal_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_normal_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 0.5, 0.0, 0.0, 0.0, 0.0 }, row_normal_ratio);

    var row_subnormal_ratios = try table.withRowSubnormalRatio(&.{ "metric", "id" }, "row_subnormal_ratio");
    defer row_subnormal_ratios.deinit();
    const row_subnormal_ratio = try (try row_subnormal_ratios.column("row_subnormal_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_subnormal_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.5, 0.0, 0.0 }, row_subnormal_ratio);

    var dropped_normal_rows = try table.dropNormalsColumn("metric");
    defer dropped_normal_rows.deinit();
    try std.testing.expectEqual(@as(usize, 4), dropped_normal_rows.height());
    const dropped_normal_metric = try (try dropped_normal_rows.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(dropped_normal_metric);
    const dropped_normal_validity = try (try dropped_normal_rows.column("metric")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(dropped_normal_validity);
    try std.testing.expectEqual(@as(f64, 0.0), dropped_normal_metric[0]);
    try std.testing.expectEqual(@as(f64, std.math.floatTrueMin(f64)), dropped_normal_metric[1]);
    try std.testing.expect(std.math.isPositiveInf(dropped_normal_metric[2]));
    try std.testing.expectEqual(@as(f64, -2.0), dropped_normal_metric[3]);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false }, dropped_normal_validity);

    var filtered_normal_rows = try table.filterNormalsColumn("metric");
    defer filtered_normal_rows.deinit();
    try std.testing.expectEqual(@as(usize, 1), filtered_normal_rows.height());
    const filtered_normal_metric = try (try filtered_normal_rows.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(filtered_normal_metric);
    try std.testing.expectEqual(@as(f64, 1.0), filtered_normal_metric[0]);

    var dropped_subnormal_rows = try table.dropSubnormalsColumn("metric");
    defer dropped_subnormal_rows.deinit();
    try std.testing.expectEqual(@as(usize, 4), dropped_subnormal_rows.height());
    const dropped_subnormal_metric = try (try dropped_subnormal_rows.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(dropped_subnormal_metric);
    const dropped_subnormal_validity = try (try dropped_subnormal_rows.column("metric")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(dropped_subnormal_validity);
    try std.testing.expectEqual(@as(f64, 1.0), dropped_subnormal_metric[0]);
    try std.testing.expectEqual(@as(f64, 0.0), dropped_subnormal_metric[1]);
    try std.testing.expect(std.math.isPositiveInf(dropped_subnormal_metric[2]));
    try std.testing.expectEqual(@as(f64, -2.0), dropped_subnormal_metric[3]);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false }, dropped_subnormal_validity);

    var filtered_subnormal_rows = try table.filterSubnormalsColumn("metric");
    defer filtered_subnormal_rows.deinit();
    try std.testing.expectEqual(@as(usize, 1), filtered_subnormal_rows.height());
    const filtered_subnormal_metric = try (try filtered_subnormal_rows.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(filtered_subnormal_metric);
    try std.testing.expectEqual(@as(f64, std.math.floatTrueMin(f64)), filtered_subnormal_metric[0]);

    try std.testing.expectError(error.ColumnNotFound, table.isNormalColumn("missing", "missing_is_normal"));
    try std.testing.expectError(error.ColumnNotFound, table.isSubnormalColumn("missing", "missing_is_subnormal"));
    try std.testing.expectError(error.ColumnNotFound, table.withRowNormalCount(&.{"missing"}, "bad_count"));
    try std.testing.expectError(error.ColumnNotFound, table.withRowSubnormalCount(&.{"missing"}, "bad_subnormal_count"));
    try std.testing.expectError(error.ColumnNotFound, table.dropNormalsColumn("missing"));
    try std.testing.expectError(error.ColumnNotFound, table.filterNormalsColumn("missing"));
    try std.testing.expectError(error.ColumnNotFound, table.dropSubnormalsColumn("missing"));
    try std.testing.expectError(error.ColumnNotFound, table.filterSubnormalsColumn("missing"));
}

test "device dataframe fills zero values" {
    const gpa = std.testing.allocator;

    var metric = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 0.0, -0.0, 3.0, std.math.nan(f64), std.math.inf(f64), -2.0 }, &.{ true, true, true, true, true, false }, .cpu);
    defer metric.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "metric", .data = metric },
    });
    defer table.deinit();

    var filled_zero = try table.fillZeroColumn("metric", f64, 42.0);
    defer filled_zero.deinit();
    const zero_values = try (try filled_zero.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(zero_values);
    const zero_validity = try (try filled_zero.column("metric")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(zero_validity);
    try std.testing.expectEqual(@as(f64, 42.0), zero_values[0]);
    try std.testing.expectEqual(@as(f64, 42.0), zero_values[1]);
    try std.testing.expectEqual(@as(f64, 3.0), zero_values[2]);
    try std.testing.expect(std.math.isNan(zero_values[3]));
    try std.testing.expect(std.math.isPositiveInf(zero_values[4]));
    try std.testing.expectEqual(@as(f64, -2.0), zero_values[5]);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, true, false }, zero_validity);

    var filled_non_zero = try table.fillNonZeroColumn("metric", f64, -7.0);
    defer filled_non_zero.deinit();
    const non_zero_values = try (try filled_non_zero.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(non_zero_values);
    try std.testing.expectEqual(@as(f64, 0.0), non_zero_values[0]);
    try std.testing.expectEqual(@as(f64, -0.0), non_zero_values[1]);
    try std.testing.expectEqual(@as(f64, -7.0), non_zero_values[2]);
    try std.testing.expectEqual(@as(f64, -7.0), non_zero_values[3]);
    try std.testing.expectEqual(@as(f64, -7.0), non_zero_values[4]);
    try std.testing.expectEqual(@as(f64, -2.0), non_zero_values[5]);

    try std.testing.expectError(error.TypeUnsupported, table.fillZeroColumn("metric", i64, 0));
    try std.testing.expectError(error.ColumnNotFound, table.fillZeroColumn("missing", f64, 0.0));
}

test "device dataframe fills sign values" {
    const gpa = std.testing.allocator;

    var metric = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ -2.0, -0.0, 0.0, 3.0, std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64), 9.0 }, &.{ true, true, true, true, true, true, true, false }, .cpu);
    defer metric.deinit();
    var id = try DeviceColumn.fromSlice(i64, gpa, &.{ -3, 0, 4, -5, 6, -7, 8, 0 }, .cpu);
    defer id.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "metric", .data = metric },
        .{ .name = "id", .data = id },
    });
    defer table.deinit();

    var filled_positive = try table.fillPositiveColumn("metric", f64, 42.0);
    defer filled_positive.deinit();
    const positive_values = try (try filled_positive.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(positive_values);
    const positive_validity = try (try filled_positive.column("metric")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(positive_validity);
    try std.testing.expectEqual(@as(f64, -2.0), positive_values[0]);
    try std.testing.expectEqual(@as(f64, -0.0), positive_values[1]);
    try std.testing.expectEqual(@as(f64, 0.0), positive_values[2]);
    try std.testing.expectEqual(@as(f64, 42.0), positive_values[3]);
    try std.testing.expect(std.math.isNan(positive_values[4]));
    try std.testing.expectEqual(@as(f64, 42.0), positive_values[5]);
    try std.testing.expect(std.math.isNegativeInf(positive_values[6]));
    try std.testing.expectEqual(@as(f64, 9.0), positive_values[7]);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, true, true, true, false }, positive_validity);

    var filled_signbit = try table.fillSignBitColumn("metric", f64, -42.0);
    defer filled_signbit.deinit();
    const signbit_values = try (try filled_signbit.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(signbit_values);
    try std.testing.expectEqual(@as(f64, -42.0), signbit_values[0]);
    try std.testing.expectEqual(@as(f64, -42.0), signbit_values[1]);
    try std.testing.expectEqual(@as(f64, 0.0), signbit_values[2]);
    try std.testing.expectEqual(@as(f64, 3.0), signbit_values[3]);
    try std.testing.expect(std.math.isNan(signbit_values[4]));
    try std.testing.expect(std.math.isPositiveInf(signbit_values[5]));
    try std.testing.expectEqual(@as(f64, -42.0), signbit_values[6]);
    try std.testing.expectEqual(@as(f64, 9.0), signbit_values[7]);

    var filled_negative = try table.fillNegativeColumn("metric", f64, 7.0);
    defer filled_negative.deinit();
    const negative_values = try (try filled_negative.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(negative_values);
    try std.testing.expectEqual(@as(f64, 7.0), negative_values[0]);
    try std.testing.expectEqual(@as(f64, -0.0), negative_values[1]);
    try std.testing.expectEqual(@as(f64, 0.0), negative_values[2]);
    try std.testing.expectEqual(@as(f64, 3.0), negative_values[3]);
    try std.testing.expect(std.math.isNan(negative_values[4]));
    try std.testing.expect(std.math.isPositiveInf(negative_values[5]));
    try std.testing.expectEqual(@as(f64, 7.0), negative_values[6]);
    try std.testing.expectEqual(@as(f64, 9.0), negative_values[7]);

    var filled_negative_id = try table.fillNegativeColumn("id", i64, 99);
    defer filled_negative_id.deinit();
    const id_values = try (try filled_negative_id.column("id")).i64.toOwnedSlice(gpa);
    defer gpa.free(id_values);
    try std.testing.expectEqualSlices(i64, &.{ 99, 0, 4, 99, 6, 99, 8, 0 }, id_values);

    var filled_positive_zero = try table.fillPositiveZeroColumn("metric", f64, 11.0);
    defer filled_positive_zero.deinit();
    const positive_zero_values = try (try filled_positive_zero.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(positive_zero_values);
    try std.testing.expectEqual(@as(f64, -2.0), positive_zero_values[0]);
    try std.testing.expectEqual(@as(f64, -0.0), positive_zero_values[1]);
    try std.testing.expectEqual(@as(f64, 11.0), positive_zero_values[2]);
    try std.testing.expectEqual(@as(f64, 3.0), positive_zero_values[3]);
    try std.testing.expect(std.math.isNan(positive_zero_values[4]));
    try std.testing.expect(std.math.isPositiveInf(positive_zero_values[5]));
    try std.testing.expect(std.math.isNegativeInf(positive_zero_values[6]));
    try std.testing.expectEqual(@as(f64, 9.0), positive_zero_values[7]);

    var filled_negative_zero = try table.fillNegativeZeroColumn("metric", f64, -11.0);
    defer filled_negative_zero.deinit();
    const negative_zero_values = try (try filled_negative_zero.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(negative_zero_values);
    try std.testing.expectEqual(@as(f64, -2.0), negative_zero_values[0]);
    try std.testing.expectEqual(@as(f64, -11.0), negative_zero_values[1]);
    try std.testing.expectEqual(@as(f64, 0.0), negative_zero_values[2]);
    try std.testing.expectEqual(@as(f64, 3.0), negative_zero_values[3]);
    try std.testing.expect(std.math.isNan(negative_zero_values[4]));
    try std.testing.expect(std.math.isPositiveInf(negative_zero_values[5]));
    try std.testing.expect(std.math.isNegativeInf(negative_zero_values[6]));
    try std.testing.expectEqual(@as(f64, 9.0), negative_zero_values[7]);

    try std.testing.expectError(error.TypeUnsupported, table.fillPositiveZeroColumn("metric", i64, 0));
    try std.testing.expectError(error.ColumnNotFound, table.fillNegativeZeroColumn("missing", f64, 0.0));
    try std.testing.expectError(error.TypeUnsupported, table.fillSignBitColumn("metric", i64, 0));
    try std.testing.expectError(error.ColumnNotFound, table.fillSignBitColumn("missing", f64, 0.0));
    try std.testing.expectError(error.TypeUnsupported, table.fillPositiveColumn("metric", i64, 0));
    try std.testing.expectError(error.ColumnNotFound, table.fillNegativeColumn("missing", f64, 0.0));
}

test "device dataframe fills finite values" {
    const gpa = std.testing.allocator;

    var metric = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, std.math.floatTrueMin(f64), 0.0, std.math.nan(f64), std.math.inf(f64), -2.0 }, &.{ true, true, true, true, true, false }, .cpu);
    defer metric.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "metric", .data = metric },
    });
    defer table.deinit();

    var filled_finite = try table.fillFiniteColumn("metric", f64, 42.0);
    defer filled_finite.deinit();
    const filled_values = try (try filled_finite.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(filled_values);
    const filled_validity = try (try filled_finite.column("metric")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(filled_validity);
    try std.testing.expectEqual(@as(f64, 42.0), filled_values[0]);
    try std.testing.expectEqual(@as(f64, 42.0), filled_values[1]);
    try std.testing.expectEqual(@as(f64, 42.0), filled_values[2]);
    try std.testing.expect(std.math.isNan(filled_values[3]));
    try std.testing.expect(std.math.isPositiveInf(filled_values[4]));
    try std.testing.expectEqual(@as(f64, -2.0), filled_values[5]);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, true, false }, filled_validity);

    try std.testing.expectError(error.TypeUnsupported, table.fillFiniteColumn("metric", i64, 0));
    try std.testing.expectError(error.ColumnNotFound, table.fillFiniteColumn("missing", f64, 0.0));
}

test "device dataframe fills normal values" {
    const gpa = std.testing.allocator;

    var metric = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, std.math.floatTrueMin(f64), 0.0, std.math.nan(f64), -2.0 }, &.{ true, true, true, true, false }, .cpu);
    defer metric.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "metric", .data = metric },
    });
    defer table.deinit();

    var filled_normal = try table.fillNormalColumn("metric", f64, 42.0);
    defer filled_normal.deinit();
    const filled_values = try (try filled_normal.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(filled_values);
    const filled_validity = try (try filled_normal.column("metric")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(filled_validity);
    try std.testing.expectEqual(@as(f64, 42.0), filled_values[0]);
    try std.testing.expectEqual(@as(f64, std.math.floatTrueMin(f64)), filled_values[1]);
    try std.testing.expectEqual(@as(f64, 0.0), filled_values[2]);
    try std.testing.expect(std.math.isNan(filled_values[3]));
    try std.testing.expectEqual(@as(f64, -2.0), filled_values[4]);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, false }, filled_validity);

    try std.testing.expectError(error.TypeUnsupported, table.fillNormalColumn("metric", i64, 0));
    try std.testing.expectError(error.ColumnNotFound, table.fillNormalColumn("missing", f64, 0.0));
}

test "device dataframe fills subnormal values" {
    const gpa = std.testing.allocator;

    var metric = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, std.math.floatTrueMin(f64), 0.0, std.math.nan(f64), -2.0 }, &.{ true, true, true, true, false }, .cpu);
    defer metric.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "metric", .data = metric },
    });
    defer table.deinit();

    var filled_subnormal = try table.fillSubnormalColumn("metric", f64, 42.0);
    defer filled_subnormal.deinit();
    const filled_values = try (try filled_subnormal.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(filled_values);
    const filled_validity = try (try filled_subnormal.column("metric")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(filled_validity);
    try std.testing.expectEqual(@as(f64, 1.0), filled_values[0]);
    try std.testing.expectEqual(@as(f64, 42.0), filled_values[1]);
    try std.testing.expectEqual(@as(f64, 0.0), filled_values[2]);
    try std.testing.expect(std.math.isNan(filled_values[3]));
    try std.testing.expectEqual(@as(f64, -2.0), filled_values[4]);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, false }, filled_validity);

    try std.testing.expectError(error.TypeUnsupported, table.fillSubnormalColumn("metric", i64, 0));
    try std.testing.expectError(error.ColumnNotFound, table.fillSubnormalColumn("missing", f64, 0.0));
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

    var with_subnormals = try table.selectColumnsWithSubnormals();
    defer with_subnormals.deinit();
    try std.testing.expectEqual(@as(usize, 1), with_subnormals.width());
    try std.testing.expectEqual(@as(?usize, 0), with_subnormals.columnIndex("mixed_metric"));

    var without_subnormals = try table.selectColumnsWithoutSubnormals();
    defer without_subnormals.deinit();
    try std.testing.expectEqual(@as(usize, 4), without_subnormals.width());
    try std.testing.expectEqual(@as(?usize, 0), without_subnormals.columnIndex("normal_metric"));
    try std.testing.expectEqual(@as(?usize, 1), without_subnormals.columnIndex("zero_metric"));
    try std.testing.expectEqual(@as(?usize, 2), without_subnormals.columnIndex("special_metric"));
    try std.testing.expectEqual(@as(?usize, 3), without_subnormals.columnIndex("id"));

    var drop_with_subnormals = try table.dropColumnsWithSubnormals();
    defer drop_with_subnormals.deinit();
    try std.testing.expectEqual(@as(usize, 4), drop_with_subnormals.width());
    try std.testing.expectEqual(@as(?usize, null), drop_with_subnormals.columnIndex("mixed_metric"));

    var drop_without_subnormals = try table.dropColumnsWithoutSubnormals();
    defer drop_without_subnormals.deinit();
    try std.testing.expectEqual(@as(usize, 1), drop_without_subnormals.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_without_subnormals.columnIndex("mixed_metric"));
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

    var row_positive_ratios = try table.withRowPositiveInfRatio(&.{"metric"}, "row_positive_inf_ratio");
    defer row_positive_ratios.deinit();
    const row_positive_inf_ratio = try (try row_positive_ratios.column("row_positive_inf_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_positive_inf_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 1.0, 0.0, 0.0, 0.0 }, row_positive_inf_ratio);

    var row_negative_ratios = try table.withRowNegativeInfRatio(&.{"metric"}, "row_negative_inf_ratio");
    defer row_negative_ratios.deinit();
    const row_negative_inf_ratio_column = try row_negative_ratios.column("row_negative_inf_ratio");
    try std.testing.expect(row_negative_inf_ratio_column.f64.nullable());
    const row_negative_inf_ratio = try row_negative_inf_ratio_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_negative_inf_ratio);
    const row_negative_inf_ratio_validity = try row_negative_inf_ratio_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_negative_inf_ratio_validity);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 1.0, 0.0, 0.0 }, row_negative_inf_ratio);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, false }, row_negative_inf_ratio_validity);

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

    var sales_cost_midpoint_table = try table.withColumnLerpScalar("sales_cost_midpoint", "sales", "cost", f64, 0.5);
    defer sales_cost_midpoint_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try sales_cost_midpoint_table.columnDType("sales_cost_midpoint"));
    const sales_cost_midpoint = try (try sales_cost_midpoint_table.column("sales_cost_midpoint")).f64.toOwnedSlice(gpa);
    defer gpa.free(sales_cost_midpoint);
    try std.testing.expectEqualSlices(f64, &.{ 1.5, 2.25, 3.5 }, sales_cost_midpoint);
    try std.testing.expectError(error.TypeUnsupported, table.withColumnLerpScalar("bad_lerp", "sales", "units", f64, 0.5));
    try std.testing.expectError(error.ColumnNotFound, table.withColumnLerpScalar("missing_lerp", "sales", "missing", f64, 0.5));

    var sales_addcmul_table = try table.withColumnAddcmulScalar("sales_addcmul", "sales", "cost", "cost", f64, 2.0);
    defer sales_addcmul_table.deinit();
    const sales_addcmul = try (try sales_addcmul_table.column("sales_addcmul")).f64.toOwnedSlice(gpa);
    defer gpa.free(sales_addcmul);
    try std.testing.expectEqualSlices(f64, &.{ 4.0, 7.5, 13.0 }, sales_addcmul);

    var sales_addcdiv_table = try table.withColumnAddcdivScalar("sales_addcdiv", "sales", "sales", "cost", f64, 0.5);
    defer sales_addcdiv_table.deinit();
    const sales_addcdiv = try (try sales_addcdiv_table.column("sales_addcdiv")).f64.toOwnedSlice(gpa);
    defer gpa.free(sales_addcdiv);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 4.0, 6.25 }, sales_addcdiv);
    try std.testing.expectError(error.TypeUnsupported, table.withColumnAddcdivScalar("bad_addcdiv", "units", "units", "units", i64, 1));
    try std.testing.expectError(error.ColumnNotFound, table.withColumnAddcmulScalar("missing_addcmul", "sales", "missing", "cost", f64, 1.0));

    var sales_clipped_table = try sales_addcdiv_table.withColumnClipArray("sales_clipped", "sales", "cost", "sales_addcdiv");
    defer sales_clipped_table.deinit();
    const sales_clipped = try (try sales_clipped_table.column("sales_clipped")).f64.toOwnedSlice(gpa);
    defer gpa.free(sales_clipped);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 3.0, 5.0 }, sales_clipped);
    try std.testing.expectError(error.TypeUnsupported, table.withColumnClipArray("bad_clip_array", "sales", "units", "cost"));
    try std.testing.expectError(error.ColumnNotFound, table.withColumnClipArray("missing_clip_array", "sales", "cost", "missing"));

    var doubled = try table.binaryColumnScalar("sales", f64, 2.0, .mul);
    defer doubled.deinit();
    const doubled_values = try doubled.f64.toOwnedSlice(gpa);
    defer gpa.free(doubled_values);
    try std.testing.expectEqualSlices(f64, &.{ 4.0, 6.0, 10.0 }, doubled_values);

    var sales_close_table = try table.withColumnIscloseScalar("sales_close_3", "sales", f64, 3.1, 0.0, 0.2);
    defer sales_close_table.deinit();
    try std.testing.expectEqual(DeviceDType.bool, try sales_close_table.columnDType("sales_close_3"));
    const sales_close = try (try sales_close_table.column("sales_close_3")).bool.toOwnedSlice(gpa);
    defer gpa.free(sales_close);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false }, sales_close);
    try std.testing.expectError(error.TypeUnsupported, table.withColumnIscloseScalar("bad_isclose", "units", i64, 2, 0, 1));
    try std.testing.expectError(error.ColumnNotFound, table.withColumnIscloseScalar("missing_isclose", "missing", f64, 3.1, 0.0, 0.2));

    var nullable_sales = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, 2.05, 3.0 }, &.{ true, false, true }, .cpu);
    defer nullable_sales.deinit();
    var nullable_sales_table = try DeviceDataFrame.init(gpa, &.{.{ .name = "metric", .data = nullable_sales }});
    defer nullable_sales_table.deinit();
    var all_null_metric = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 4.0, 5.0 }, &.{ false, false }, .cpu);
    defer all_null_metric.deinit();
    var all_null_metric_table = try DeviceDataFrame.init(gpa, &.{.{ .name = "metric", .data = all_null_metric }});
    defer all_null_metric_table.deinit();
    var repeated_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ 1.0, 1.0, std.math.nan(f64), std.math.nan(f64) }, .cpu);
    defer repeated_metric.deinit();
    var repeated_metric_table = try DeviceDataFrame.init(gpa, &.{.{ .name = "metric", .data = repeated_metric }});
    defer repeated_metric_table.deinit();
    var modal_units = try DeviceColumn.fromSliceWithValidity(i64, gpa, &.{ 2, 3, 3, 2, 9 }, &.{ true, true, true, true, false }, .cpu);
    defer modal_units.deinit();
    var modal_units_table = try DeviceDataFrame.init(gpa, &.{.{ .name = "units", .data = modal_units }});
    defer modal_units_table.deinit();
    var nullable_close_table = try nullable_sales_table.withColumnIscloseWithDeviceScalars("metric_close", "metric", .{ .f64 = 2.0 }, .{ .f64 = 0.0 }, .{ .f64 = 0.1 });
    defer nullable_close_table.deinit();
    const nullable_close_column = try nullable_close_table.column("metric_close");
    try std.testing.expect(nullable_close_column.bool.nullable());
    try std.testing.expectEqual(@as(usize, 1), nullable_close_column.bool.null_count);
    const nullable_close = try nullable_close_column.bool.toOwnedSlice(gpa);
    defer gpa.free(nullable_close);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false }, nullable_close);

    var nan_close_column = try DeviceColumn.fromSlice(f64, gpa, &.{ std.math.nan(f64), 2.0, 2.2 }, .cpu);
    defer nan_close_column.deinit();
    var nan_close_source = try DeviceDataFrame.init(gpa, &.{.{ .name = "metric", .data = nan_close_column }});
    defer nan_close_source.deinit();
    var anomaly_metric = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ std.math.inf(f64), -std.math.inf(f64), std.math.nan(f64), 4.0 }, &.{ true, false, true, true }, .cpu);
    defer anomaly_metric.deinit();
    var anomaly_metric_table = try DeviceDataFrame.init(gpa, &.{.{ .name = "metric", .data = anomaly_metric }});
    defer anomaly_metric_table.deinit();
    var signed_inf_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ std.math.inf(f64), -std.math.inf(f64), 1.0 }, .cpu);
    defer signed_inf_metric.deinit();
    var signed_inf_table = try DeviceDataFrame.init(gpa, &.{.{ .name = "metric", .data = signed_inf_metric }});
    defer signed_inf_table.deinit();
    var signed_zero_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ 0.0, -0.0, 1.0 }, .cpu);
    defer signed_zero_metric.deinit();
    var signed_zero_table = try DeviceDataFrame.init(gpa, &.{.{ .name = "metric", .data = signed_zero_metric }});
    defer signed_zero_table.deinit();
    var sign_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ -2.0, 0.0, 3.0, -0.0 }, .cpu);
    defer sign_metric.deinit();
    var sign_metric_table = try DeviceDataFrame.init(gpa, &.{.{ .name = "metric", .data = sign_metric }});
    defer sign_metric_table.deinit();
    var ieee_class_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ std.math.floatMin(f64), std.math.floatMin(f64) / 2.0, 0.0, std.math.inf(f64) }, .cpu);
    defer ieee_class_metric.deinit();
    var ieee_class_table = try DeviceDataFrame.init(gpa, &.{.{ .name = "metric", .data = ieee_class_metric }});
    defer ieee_class_table.deinit();
    var nan_close_table = try nan_close_source.withColumnIscloseScalarEqualNan("metric_nan_close", "metric", f64, std.math.nan(f64), 0.0, 0.0, true);
    defer nan_close_table.deinit();
    const nan_close = try (try nan_close_table.column("metric_nan_close")).bool.toOwnedSlice(gpa);
    defer gpa.free(nan_close);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false }, nan_close);

    try std.testing.expect(try table.allcloseColumnScalar("sales", f64, 3.3, 0.0, 2.0));
    try std.testing.expect(!try table.allcloseColumnScalar("sales", f64, 3.0, 0.0, 0.5));
    try std.testing.expect(try table.allcloseColumnWithDeviceScalars("cost", .{ .f64 = 1.5 }, .{ .f64 = 0.0 }, .{ .f64 = 0.5 }));
    try std.testing.expect(!try nullable_sales_table.allcloseColumnScalar("metric", f64, 2.0, 0.0, 10.0));
    try std.testing.expect(try nan_close_source.allcloseColumnScalarEqualNan("metric", f64, std.math.nan(f64), 0.0, 0.0, true) == false);
    try std.testing.expectError(error.TypeUnsupported, table.allcloseColumnScalar("units", i64, 2, 0, 1));
    try std.testing.expectError(error.ColumnNotFound, table.allcloseColumnScalar("missing", f64, 1.0, 0.0, 0.0));
    try std.testing.expectEqual(@as(usize, 3), try table.countNonzeroColumn("sales"));
    try std.testing.expectEqual(@as(usize, 2), try table.countNonzeroColumn("units"));
    try std.testing.expectError(error.ColumnNotFound, table.countNonzeroColumn("missing"));
    try std.testing.expectEqual(@as(usize, 0), try table.zeroCountColumn("sales"));
    try std.testing.expectEqual(@as(usize, 0), try table.countZeroColumn("units"));
    try std.testing.expectEqual(DeviceScalar{ .f64 = 0.0 }, try table.zeroRatioColumn("sales"));
    try std.testing.expectEqual(DeviceScalar{ .f64 = 1.0 }, try table.nonzeroRatioColumn("sales"));
    try std.testing.expectEqual(@as(?usize, null), try table.firstZeroIndexColumn("sales"));
    try std.testing.expectEqual(@as(?usize, null), try table.lastZeroIndexColumn("sales"));
    try std.testing.expectEqual(@as(?usize, 0), try table.firstNonzeroIndexColumn("sales"));
    try std.testing.expectEqual(@as(?usize, 2), try table.lastNonzeroIndexColumn("sales"));
    try std.testing.expectEqual(@as(?usize, 0), try signed_zero_table.firstZeroIndexColumn("metric"));
    try std.testing.expectEqual(@as(?usize, 1), try signed_zero_table.lastZeroIndexColumn("metric"));
    try std.testing.expectEqual(@as(?usize, 0), try signed_zero_table.firstPositiveZeroIndexColumn("metric"));
    try std.testing.expectEqual(@as(?usize, 0), try signed_zero_table.lastPositiveZeroIndexColumn("metric"));
    try std.testing.expectEqual(@as(?usize, 1), try signed_zero_table.firstNegativeZeroIndexColumn("metric"));
    try std.testing.expectEqual(@as(?usize, 1), try signed_zero_table.lastNegativeZeroIndexColumn("metric"));
    try std.testing.expectEqual(@as(?usize, null), try table.firstPositiveZeroIndexColumn("sales"));
    try std.testing.expectEqual(@as(?usize, null), try all_null_metric_table.lastNegativeZeroIndexColumn("metric"));
    try std.testing.expectEqual(@as(?usize, 2), try signed_zero_table.firstNonzeroIndexColumn("metric"));
    try std.testing.expectEqual(@as(?usize, 2), try signed_zero_table.lastNonzeroIndexColumn("metric"));
    try std.testing.expectEqual(@as(usize, 0), try table.positiveZeroCountColumn("sales"));
    try std.testing.expectEqual(@as(usize, 0), try table.negativeZeroCountColumn("sales"));
    try std.testing.expectEqual(@as(usize, 1), try signed_zero_table.positiveZeroCountColumn("metric"));
    try std.testing.expectEqual(@as(usize, 1), try signed_zero_table.negativeZeroCountColumn("metric"));
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), (try signed_zero_table.positiveZeroRatioColumn("metric")).f64, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), (try signed_zero_table.negativeZeroRatioColumn("metric")).f64, 1e-12);
    try std.testing.expectEqual(@as(usize, 1), try sign_metric_table.positiveCountColumn("metric"));
    try std.testing.expectEqual(@as(usize, 1), try sign_metric_table.negativeCountColumn("metric"));
    try std.testing.expectEqual(@as(usize, 2), try sign_metric_table.signBitCountColumn("metric"));
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), (try sign_metric_table.positiveRatioColumn("metric")).f64, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), (try sign_metric_table.negativeRatioColumn("metric")).f64, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), (try sign_metric_table.signBitRatioColumn("metric")).f64, 1e-12);
    try std.testing.expectEqual(@as(?usize, 2), try sign_metric_table.firstPositiveIndexColumn("metric"));
    try std.testing.expectEqual(@as(?usize, 2), try sign_metric_table.lastPositiveIndexColumn("metric"));
    try std.testing.expectEqual(@as(?usize, 0), try sign_metric_table.firstNegativeIndexColumn("metric"));
    try std.testing.expectEqual(@as(?usize, 0), try sign_metric_table.lastNegativeIndexColumn("metric"));
    try std.testing.expectEqual(@as(?usize, 0), try sign_metric_table.firstSignBitIndexColumn("metric"));
    try std.testing.expectEqual(@as(?usize, 3), try sign_metric_table.lastSignBitIndexColumn("metric"));
    try std.testing.expectEqual(@as(?usize, null), try all_null_metric_table.firstPositiveIndexColumn("metric"));
    try std.testing.expectEqual(@as(?usize, null), try all_null_metric_table.firstSignBitIndexColumn("metric"));
    try std.testing.expectEqual(@as(usize, 1), try ieee_class_table.normalCountColumn("metric"));
    try std.testing.expectEqual(@as(usize, 1), try ieee_class_table.subnormalCountColumn("metric"));
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), (try ieee_class_table.normalRatioColumn("metric")).f64, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), (try ieee_class_table.subnormalRatioColumn("metric")).f64, 1e-12);
    try std.testing.expectEqual(@as(?usize, 0), try ieee_class_table.firstNormalIndexColumn("metric"));
    try std.testing.expectEqual(@as(?usize, 0), try ieee_class_table.lastNormalIndexColumn("metric"));
    try std.testing.expectEqual(@as(?usize, 1), try ieee_class_table.firstSubnormalIndexColumn("metric"));
    try std.testing.expectEqual(@as(?usize, 1), try ieee_class_table.lastSubnormalIndexColumn("metric"));
    try std.testing.expectEqual(@as(?usize, null), try all_null_metric_table.firstNormalIndexColumn("metric"));
    try std.testing.expectEqual(@as(?usize, null), try table.lastSubnormalIndexColumn("sales"));
    try std.testing.expectEqual(@as(usize, 0), try all_null_metric_table.zeroCountColumn("metric"));
    try std.testing.expect(std.math.isNan((try all_null_metric_table.zeroRatioColumn("metric")).f64));
    try std.testing.expectEqual(@as(?usize, null), try all_null_metric_table.firstZeroIndexColumn("metric"));
    try std.testing.expectEqual(@as(?usize, null), try all_null_metric_table.firstNonzeroIndexColumn("metric"));
    try std.testing.expectError(error.ColumnNotFound, table.zeroCountColumn("missing"));
    try std.testing.expectEqual(@as(usize, 0), try table.nanCountColumn("sales"));
    try std.testing.expectEqual(@as(usize, 0), try table.infCountColumn("sales"));
    try std.testing.expectEqual(@as(usize, 3), try table.finiteCountColumn("sales"));
    try std.testing.expectEqual(@as(usize, 0), try table.nonFiniteCountColumn("sales"));
    try std.testing.expectEqual(@as(usize, 2), try table.finiteCountColumn("units"));
    try std.testing.expectEqual(@as(usize, 1), try nan_close_source.nanCountColumn("metric"));
    try std.testing.expectEqual(@as(usize, 1), try anomaly_metric_table.nanCountColumn("metric"));
    try std.testing.expectEqual(@as(usize, 1), try anomaly_metric_table.infCountColumn("metric"));
    try std.testing.expectEqual(@as(usize, 1), try anomaly_metric_table.positiveInfCountColumn("metric"));
    try std.testing.expectEqual(@as(usize, 0), try anomaly_metric_table.negativeInfCountColumn("metric"));
    try std.testing.expectEqual(@as(usize, 1), try anomaly_metric_table.finiteCountColumn("metric"));
    try std.testing.expectEqual(@as(usize, 2), try anomaly_metric_table.nonFiniteCountColumn("metric"));
    try std.testing.expectEqual(@as(usize, 1), try signed_inf_table.positiveInfCountColumn("metric"));
    try std.testing.expectEqual(@as(usize, 1), try signed_inf_table.negativeInfCountColumn("metric"));
    try std.testing.expectEqual(DeviceScalar{ .f64 = 1.0 }, try table.finiteRatioColumn("units"));
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), (try anomaly_metric_table.nanRatioColumn("metric")).f64, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), (try anomaly_metric_table.infRatioColumn("metric")).f64, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), (try signed_inf_table.positiveInfRatioColumn("metric")).f64, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), (try signed_inf_table.negativeInfRatioColumn("metric")).f64, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), (try anomaly_metric_table.finiteRatioColumn("metric")).f64, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), (try anomaly_metric_table.nonFiniteRatioColumn("metric")).f64, 1e-12);
    try std.testing.expect(std.math.isNan((try all_null_metric_table.nanRatioColumn("metric")).f64));
    try std.testing.expectEqual(@as(?usize, 2), try anomaly_metric_table.firstNanIndexColumn("metric"));
    try std.testing.expectEqual(@as(?usize, 2), try anomaly_metric_table.lastNanIndexColumn("metric"));
    try std.testing.expectEqual(@as(?usize, 0), try anomaly_metric_table.firstInfIndexColumn("metric"));
    try std.testing.expectEqual(@as(?usize, 0), try anomaly_metric_table.lastInfIndexColumn("metric"));
    try std.testing.expectEqual(@as(?usize, 0), try signed_inf_table.firstPositiveInfIndexColumn("metric"));
    try std.testing.expectEqual(@as(?usize, 0), try signed_inf_table.lastPositiveInfIndexColumn("metric"));
    try std.testing.expectEqual(@as(?usize, 1), try signed_inf_table.firstNegativeInfIndexColumn("metric"));
    try std.testing.expectEqual(@as(?usize, 1), try signed_inf_table.lastNegativeInfIndexColumn("metric"));
    try std.testing.expectEqual(@as(?usize, null), try anomaly_metric_table.firstNegativeInfIndexColumn("metric"));
    try std.testing.expectEqual(@as(?usize, null), try all_null_metric_table.lastPositiveInfIndexColumn("metric"));
    try std.testing.expectEqual(@as(?usize, 3), try anomaly_metric_table.firstFiniteIndexColumn("metric"));
    try std.testing.expectEqual(@as(?usize, 3), try anomaly_metric_table.lastFiniteIndexColumn("metric"));
    try std.testing.expectEqual(@as(?usize, 0), try anomaly_metric_table.firstNonFiniteIndexColumn("metric"));
    try std.testing.expectEqual(@as(?usize, 2), try anomaly_metric_table.lastNonFiniteIndexColumn("metric"));
    try std.testing.expectEqual(@as(?usize, null), try all_null_metric_table.firstNanIndexColumn("metric"));
    try std.testing.expectEqual(@as(?usize, null), try all_null_metric_table.firstFiniteIndexColumn("metric"));
    try std.testing.expectError(error.ColumnNotFound, table.nanCountColumn("missing"));
    try std.testing.expectEqual(@as(usize, 0), try table.nullCountColumn("sales"));
    try std.testing.expectEqual(@as(usize, 3), try table.validCountColumn("sales"));
    try std.testing.expectEqual(@as(?usize, 0), try table.firstValidIndexColumn("sales"));
    try std.testing.expectEqual(@as(?usize, 2), try table.lastValidIndexColumn("sales"));
    try std.testing.expectEqual(@as(?usize, null), try table.firstNullIndexColumn("sales"));
    try std.testing.expectEqual(@as(?usize, null), try table.lastNullIndexColumn("sales"));
    try std.testing.expectEqual(@as(usize, 1), try table.nullCountColumn("units"));
    try std.testing.expectEqual(@as(usize, 2), try table.validCountColumn("units"));
    try std.testing.expectEqual(@as(?usize, 0), try table.firstValidIndexColumn("units"));
    try std.testing.expectEqual(@as(?usize, 2), try table.lastValidIndexColumn("units"));
    try std.testing.expectEqual(@as(?usize, 1), try table.firstNullIndexColumn("units"));
    try std.testing.expectEqual(@as(?usize, 1), try table.lastNullIndexColumn("units"));
    const units_null_ratio = try table.nullRatioColumn("units");
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), units_null_ratio.f64, 1e-12);
    const units_valid_ratio = try table.validRatioColumn("units");
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), units_valid_ratio.f64, 1e-12);
    try std.testing.expectEqual(@as(usize, 1), try nullable_sales_table.nullCountColumn("metric"));
    try std.testing.expectEqual(@as(usize, 2), try nullable_sales_table.validCountColumn("metric"));
    try std.testing.expectEqual(@as(?usize, null), try all_null_metric_table.firstValidIndexColumn("metric"));
    try std.testing.expectEqual(@as(?usize, null), try all_null_metric_table.lastValidIndexColumn("metric"));
    try std.testing.expectEqual(@as(?usize, 0), try all_null_metric_table.firstNullIndexColumn("metric"));
    try std.testing.expectEqual(@as(?usize, 1), try all_null_metric_table.lastNullIndexColumn("metric"));
    try std.testing.expectEqual(DeviceScalar{ .f64 = 1.0 }, try all_null_metric_table.nullRatioColumn("metric"));
    try std.testing.expectError(error.ColumnNotFound, table.nullCountColumn("missing"));
    try std.testing.expectEqual(@as(usize, 3), try table.nUniqueColumn("sales"));
    try std.testing.expectEqual(@as(usize, 2), try table.nUniqueColumn("units"));
    try std.testing.expectEqual(@as(usize, 2), try nullable_sales_table.nUniqueColumn("metric"));
    try std.testing.expectEqual(@as(usize, 2), try nullable_sales_table.countDistinctColumn("metric"));
    try std.testing.expectEqual(@as(usize, 0), try all_null_metric_table.nUniqueColumn("metric"));
    try std.testing.expectEqual(@as(usize, 2), try repeated_metric_table.nUniqueColumn("metric"));
    try std.testing.expectEqual(@as(usize, 2), try repeated_metric_table.countDistinctColumn("metric"));
    try std.testing.expectError(error.ColumnNotFound, table.nUniqueColumn("missing"));
    try std.testing.expectEqual(DeviceScalar{ .f64 = 1.0 }, try repeated_metric_table.modeColumn("metric"));
    try std.testing.expectEqual(DeviceScalar{ .i64 = 2 }, try modal_units_table.modeColumn("units"));
    try std.testing.expectError(error.EmptyArray, all_null_metric_table.modeColumn("metric"));
    try std.testing.expectError(error.ColumnNotFound, table.modeColumn("missing"));
    try std.testing.expectEqual(DeviceScalar{ .f64 = 10.0 }, try table.sumColumn("sales"));
    try std.testing.expectEqual(DeviceScalar{ .i64 = 4 }, try table.sumColumn("units"));
    try std.testing.expectEqual(DeviceScalar{ .f64 = 30.0 }, try table.prodColumn("sales"));
    try std.testing.expectEqual(DeviceScalar{ .i64 = 3 }, try table.prodColumn("units"));
    try std.testing.expectEqual(DeviceScalar{ .f64 = 3.0 }, try nullable_sales_table.prodColumn("metric"));
    try std.testing.expectEqual(DeviceScalar{ .f64 = 1.0 }, try all_null_metric_table.prodColumn("metric"));
    try std.testing.expectError(error.ColumnNotFound, table.prodColumn("missing"));
    const sales_mean = try table.meanColumn("sales");
    try std.testing.expectApproxEqAbs(@as(f64, 10.0 / 3.0), sales_mean.f64, 1e-12);
    try std.testing.expectEqual(DeviceScalar{ .f64 = 2.0 }, try nullable_sales_table.meanColumn("metric"));
    try std.testing.expectError(error.TypeUnsupported, table.meanColumn("units"));
    try std.testing.expectEqual(DeviceScalar{ .f64 = 3.0 }, try table.medianColumn("sales"));
    try std.testing.expectEqual(DeviceScalar{ .f64 = 2.0 }, try table.medianColumn("units"));
    try std.testing.expectEqual(DeviceScalar{ .f64 = 2.5 }, try table.quantileColumn("sales", 0.25));
    try std.testing.expectEqual(DeviceScalar{ .f64 = 2.0 }, try nullable_sales_table.medianColumn("metric"));
    try std.testing.expectEqual(DeviceScalar{ .f64 = 2.5 }, try nullable_sales_table.quantileColumn("metric", 0.75));
    try std.testing.expectError(error.EmptyArray, all_null_metric_table.medianColumn("metric"));
    try std.testing.expectError(error.InvalidShape, table.quantileColumn("sales", 1.5));
    try std.testing.expectError(error.ColumnNotFound, table.medianColumn("missing"));
    const sales_variance = try table.varianceColumn("sales", 0.0);
    try std.testing.expectApproxEqAbs(@as(f64, 14.0 / 9.0), sales_variance.f64, 1e-12);
    const sales_stddev = try table.stddevColumn("sales", 0.0);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 14.0 / 9.0)), sales_stddev.f64, 1e-12);
    try std.testing.expectEqual(DeviceScalar{ .f64 = 1.0 }, try table.varColumn("units", 0.0));
    try std.testing.expectEqual(DeviceScalar{ .f64 = 1.0 }, try table.stdColumn("units", 0.0));
    try std.testing.expectEqual(DeviceScalar{ .f64 = 1.0 }, try nullable_sales_table.varianceColumn("metric", 0.0));
    try std.testing.expectEqual(DeviceScalar{ .f64 = 2.0 }, try nullable_sales_table.varianceColumn("metric", 1.0));
    const sales_sem = try table.semColumn("sales", 0.0);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 14.0 / 9.0)) / std.math.sqrt(@as(f64, 3.0)), sales_sem.f64, 1e-12);
    const sales_cv = try table.cvColumn("sales", 0.0);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 14.0 / 9.0)) / @as(f64, 10.0 / 3.0), sales_cv.f64, 1e-12);
    const sales_skewness = try table.skewnessColumn("sales");
    try std.testing.expectApproxEqAbs(@as(f64, std.math.sqrt(3.0) * (20.0 / 9.0) / std.math.pow(f64, 14.0 / 3.0, 1.5)), sales_skewness.f64, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -1.5), (try table.kurtosisColumn("sales")).f64, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), (try nullable_sales_table.skewColumn("metric")).f64, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -2.0), (try nullable_sales_table.kurtColumn("metric")).f64, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0 / 3.0), (try table.meanAbsColumn("sales")).f64, 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 38.0 / 3.0)), (try table.rmsColumn("sales")).f64, 1e-12);
    try std.testing.expectEqual(DeviceScalar{ .f64 = 2.0 }, try nullable_sales_table.meanAbsColumn("metric"));
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 5.0)), (try nullable_sales_table.rmsColumn("metric")).f64, 1e-12);
    try std.testing.expectEqual(DeviceScalar{ .f64 = 10.0 }, try table.l1NormColumn("sales"));
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 38.0)), (try table.l2NormColumn("sales")).f64, 1e-12);
    try std.testing.expectEqual(DeviceScalar{ .f64 = 4.0 }, try nullable_sales_table.l1NormColumn("metric"));
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 10.0)), (try nullable_sales_table.l2NormColumn("metric")).f64, 1e-12);
    try std.testing.expectApproxEqAbs(std.math.pow(f64, 30.0, 1.0 / 3.0), (try table.geometricMeanColumn("sales")).f64, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 90.0 / 31.0), (try table.harmonicMeanColumn("sales")).f64, 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 3.0)), (try nullable_sales_table.geoMeanColumn("metric")).f64, 1e-12);
    try std.testing.expectEqual(DeviceScalar{ .f64 = 1.5 }, try nullable_sales_table.harmMeanColumn("metric"));
    try std.testing.expectEqual(DeviceScalar{ .f64 = 1.0 }, try table.madColumn("sales"));
    try std.testing.expectEqual(DeviceScalar{ .f64 = 1.5 }, try table.iqrColumn("sales"));
    try std.testing.expectEqual(DeviceScalar{ .f64 = 1.0 }, try nullable_sales_table.medianAbsDevColumn("metric"));
    try std.testing.expectEqual(DeviceScalar{ .f64 = 1.0 }, try nullable_sales_table.iqrColumn("metric"));
    try std.testing.expectError(error.EmptyArray, all_null_metric_table.varianceColumn("metric", 0.0));
    try std.testing.expectError(error.EmptyArray, all_null_metric_table.skewnessColumn("metric"));
    try std.testing.expectError(error.EmptyArray, all_null_metric_table.kurtosisColumn("metric"));
    try std.testing.expectError(error.EmptyArray, all_null_metric_table.meanAbsColumn("metric"));
    try std.testing.expectError(error.EmptyArray, all_null_metric_table.rmsColumn("metric"));
    try std.testing.expectError(error.EmptyArray, all_null_metric_table.l1NormColumn("metric"));
    try std.testing.expectError(error.EmptyArray, all_null_metric_table.l2NormColumn("metric"));
    try std.testing.expectError(error.EmptyArray, all_null_metric_table.geometricMeanColumn("metric"));
    try std.testing.expectError(error.EmptyArray, all_null_metric_table.harmonicMeanColumn("metric"));
    try std.testing.expectError(error.EmptyArray, all_null_metric_table.madColumn("metric"));
    try std.testing.expectError(error.EmptyArray, all_null_metric_table.iqrColumn("metric"));
    try std.testing.expectError(error.InvalidShape, table.varianceColumn("sales", -1.0));
    try std.testing.expectError(error.InvalidShape, table.semColumn("sales", -1.0));
    try std.testing.expectError(error.InvalidShape, table.cvColumn("sales", -1.0));
    try std.testing.expectError(error.ColumnNotFound, table.stddevColumn("missing", 0.0));
    try std.testing.expectEqual(DeviceScalar{ .f64 = 2.0 }, try table.minColumn("sales"));
    try std.testing.expectEqual(DeviceScalar{ .f64 = 5.0 }, try table.maxColumn("sales"));
    try std.testing.expectEqual(DeviceScalar{ .i64 = 1 }, try table.minColumn("units"));
    try std.testing.expectEqual(DeviceScalar{ .i64 = 3 }, try table.maxColumn("units"));
    try std.testing.expectEqual(DeviceScalar{ .f64 = 3.0 }, try table.ptpColumn("sales"));
    try std.testing.expectEqual(DeviceScalar{ .i64 = 2 }, try table.ptpColumn("units"));
    try std.testing.expectEqual(DeviceScalar{ .f64 = 2.0 }, try nullable_sales_table.ptpColumn("metric"));
    try std.testing.expectError(error.EmptyArray, all_null_metric_table.ptpColumn("metric"));
    try std.testing.expectError(error.ColumnNotFound, table.ptpColumn("missing"));
    try std.testing.expectEqual(@as(usize, 0), try table.argminColumn("sales"));
    try std.testing.expectEqual(@as(usize, 2), try table.argmaxColumn("sales"));
    try std.testing.expectEqual(@as(usize, 0), try table.argminColumn("units"));
    try std.testing.expectEqual(@as(usize, 2), try table.argmaxColumn("units"));
    try std.testing.expectEqual(@as(usize, 0), try nullable_sales_table.argminColumn("metric"));
    try std.testing.expectEqual(@as(usize, 2), try nullable_sales_table.argmaxColumn("metric"));
    try std.testing.expectError(error.EmptyArray, all_null_metric_table.argminColumn("metric"));
    try std.testing.expectError(error.EmptyArray, all_null_metric_table.argmaxColumn("metric"));
    try std.testing.expectError(error.ColumnNotFound, table.argminColumn("missing"));
    try std.testing.expectError(error.ColumnNotFound, table.argmaxColumn("missing"));

    var cost_delta = try table.withColumnAbs("cost_abs", "cost");
    defer cost_delta.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try cost_delta.columnDType("cost_abs"));
    const cost_abs = try (try cost_delta.column("cost_abs")).f64.toOwnedSlice(gpa);
    defer gpa.free(cost_abs);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 1.5, 2.0 }, cost_abs);
    try std.testing.expectError(error.ColumnNotFound, table.withColumnAbs("bad_abs", "missing"));

    var rounding_active = try DeviceColumn.fromSlice(bool, gpa, &.{ true, false, true }, .cpu);
    defer rounding_active.deinit();
    var rounding_type_table = try DeviceDataFrame.init(gpa, &.{.{ .name = "active", .data = rounding_active }});
    defer rounding_type_table.deinit();
    try std.testing.expectEqual(@as(usize, 1), try rounding_type_table.zeroCountColumn("active"));
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), (try rounding_type_table.zeroRatioColumn("active")).f64, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), (try rounding_type_table.nonZeroRatioColumn("active")).f64, 1e-12);
    try std.testing.expectEqual(@as(?usize, 1), try rounding_type_table.firstZeroIndexColumn("active"));
    try std.testing.expectEqual(@as(?usize, 1), try rounding_type_table.lastZeroIndexColumn("active"));
    try std.testing.expectEqual(@as(?usize, 0), try rounding_type_table.firstNonzeroIndexColumn("active"));
    try std.testing.expectEqual(@as(?usize, 2), try rounding_type_table.lastNonzeroIndexColumn("active"));
    try std.testing.expectEqual(@as(usize, 2), try rounding_type_table.nUniqueColumn("active"));
    try std.testing.expectEqual(@as(usize, 2), try rounding_type_table.countDistinctColumn("active"));
    try std.testing.expectEqual(DeviceScalar{ .bool = true }, try rounding_type_table.modeColumn("active"));
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.sumColumn("active"));
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.prodColumn("active"));
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.medianColumn("active"));
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.quantileColumn("active", 0.5));
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.varianceColumn("active", 0.0));
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.stddevColumn("active", 0.0));
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.semColumn("active", 0.0));
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.cvColumn("active", 0.0));
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.skewnessColumn("active"));
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.kurtosisColumn("active"));
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.meanAbsColumn("active"));
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.rmsColumn("active"));
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.l1NormColumn("active"));
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.l2NormColumn("active"));
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.geometricMeanColumn("active"));
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.harmonicMeanColumn("active"));
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.madColumn("active"));
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.iqrColumn("active"));
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.minColumn("active"));
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.maxColumn("active"));
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.ptpColumn("active"));
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.argminColumn("active"));
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.argmaxColumn("active"));
    try std.testing.expect(try rounding_type_table.anyColumn("active"));
    try std.testing.expect(!try rounding_type_table.allColumn("active"));
    try std.testing.expectEqual(@as(usize, 2), try rounding_type_table.countTrueColumn("active"));
    try std.testing.expectEqual(@as(usize, 1), try rounding_type_table.countFalseColumn("active"));
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), (try rounding_type_table.trueRatioColumn("active")).f64, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), (try rounding_type_table.falseRatioColumn("active")).f64, 1e-12);
    try std.testing.expectEqual(@as(?usize, 0), try rounding_type_table.firstTrueIndexColumn("active"));
    try std.testing.expectEqual(@as(?usize, 2), try rounding_type_table.lastTrueIndexColumn("active"));
    try std.testing.expectEqual(@as(?usize, 1), try rounding_type_table.firstFalseIndexColumn("active"));
    try std.testing.expectEqual(@as(?usize, 1), try rounding_type_table.lastFalseIndexColumn("active"));
    try std.testing.expectError(error.TypeUnsupported, table.anyColumn("sales"));
    try std.testing.expectError(error.TypeUnsupported, table.trueRatioColumn("sales"));
    try std.testing.expectError(error.TypeUnsupported, table.firstTrueIndexColumn("sales"));

    var nullable_bool = try DeviceColumn.fromSliceWithValidity(bool, gpa, &.{ false, true, false }, &.{ true, false, true }, .cpu);
    defer nullable_bool.deinit();
    var nullable_bool_table = try DeviceDataFrame.init(gpa, &.{.{ .name = "flag", .data = nullable_bool }});
    defer nullable_bool_table.deinit();
    try std.testing.expect(!try nullable_bool_table.anyColumn("flag"));
    try std.testing.expect(!try nullable_bool_table.allColumn("flag"));
    try std.testing.expectEqual(@as(usize, 0), try nullable_bool_table.countTrueColumn("flag"));
    try std.testing.expectEqual(@as(usize, 2), try nullable_bool_table.countFalseColumn("flag"));
    try std.testing.expectEqual(DeviceScalar{ .f64 = 0.0 }, try nullable_bool_table.trueRatioColumn("flag"));
    try std.testing.expectEqual(DeviceScalar{ .f64 = 1.0 }, try nullable_bool_table.falseRatioColumn("flag"));
    try std.testing.expectEqual(@as(?usize, null), try nullable_bool_table.firstTrueIndexColumn("flag"));
    try std.testing.expectEqual(@as(?usize, null), try nullable_bool_table.lastTrueIndexColumn("flag"));
    try std.testing.expectEqual(@as(?usize, 0), try nullable_bool_table.firstFalseIndexColumn("flag"));
    try std.testing.expectEqual(@as(?usize, 2), try nullable_bool_table.lastFalseIndexColumn("flag"));

    var nullable_any_false_table = try nullable_bool_table.withRowAnyFalse(&.{"flag"}, "row_any_false");
    defer nullable_any_false_table.deinit();
    const nullable_any_false = try (try nullable_any_false_table.column("row_any_false")).bool.toOwnedSlice(gpa);
    defer gpa.free(nullable_any_false);
    const nullable_any_false_validity = try (try nullable_any_false_table.column("row_any_false")).bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(nullable_any_false_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, nullable_any_false);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, nullable_any_false_validity);

    var all_null_bool = try DeviceColumn.fromSliceWithValidity(bool, gpa, &.{ true, false }, &.{ false, false }, .cpu);
    defer all_null_bool.deinit();
    var all_null_bool_table = try DeviceDataFrame.init(gpa, &.{.{ .name = "flag", .data = all_null_bool }});
    defer all_null_bool_table.deinit();
    try std.testing.expect(std.math.isNan((try all_null_bool_table.trueRatioColumn("flag")).f64));
    try std.testing.expect(std.math.isNan((try all_null_bool_table.falseRatioColumn("flag")).f64));

    var where_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ -1.0, 2.0, 5.0 }, .cpu);
    defer where_metric.deinit();
    var where_mask = try DeviceColumn.fromSlice(bool, gpa, &.{ true, false, true }, .cpu);
    defer where_mask.deinit();
    var where_fallback = try DeviceColumn.fromSlice(f64, gpa, &.{ 10.0, 20.0, 30.0 }, .cpu);
    defer where_fallback.deinit();
    var where_needles = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 5.0, 8.0 }, .cpu);
    defer where_needles.deinit();
    var where_table = try DeviceDataFrame.init(gpa, &.{ .{ .name = "metric", .data = where_metric }, .{ .name = "mask", .data = where_mask }, .{ .name = "fallback", .data = where_fallback }, .{ .name = "needles", .data = where_needles } });
    defer where_table.deinit();

    var metric_isin_table = try where_table.withColumnIsIn("metric_isin", "metric", "needles");
    defer metric_isin_table.deinit();
    const metric_isin = try (try metric_isin_table.column("metric_isin")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_isin);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true }, metric_isin);

    var metric_isin_inverted_table = try where_table.withColumnIsInInverted("metric_isin_inverted", "metric", "needles");
    defer metric_isin_inverted_table.deinit();
    const metric_isin_inverted = try (try metric_isin_inverted_table.column("metric_isin_inverted")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_isin_inverted);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false }, metric_isin_inverted);
    try std.testing.expectError(error.TypeUnsupported, where_table.withColumnIsIn("bad_isin", "metric", "mask"));

    var where_scalar_table = try where_table.withColumnWhereScalar("metric_where", "metric", "mask", f64, 0.0);
    defer where_scalar_table.deinit();
    const metric_where = try (try where_scalar_table.column("metric_where")).f64.toOwnedSlice(gpa);
    defer gpa.free(metric_where);
    try std.testing.expectEqualSlices(f64, &.{ -1.0, 0.0, 5.0 }, metric_where);
    try std.testing.expectError(error.TypeUnsupported, table.withColumnWhereScalar("bad_where", "sales", "cost", f64, 0.0));
    try std.testing.expectError(error.ColumnNotFound, where_table.withColumnWhereScalar("missing_where", "metric", "missing", f64, 0.0));

    var where_column_table = try where_table.withColumnWhere("metric_where_column", "metric", "mask", "fallback");
    defer where_column_table.deinit();
    const metric_where_column = try (try where_column_table.column("metric_where_column")).f64.toOwnedSlice(gpa);
    defer gpa.free(metric_where_column);
    try std.testing.expectEqualSlices(f64, &.{ -1.0, 20.0, 5.0 }, metric_where_column);
    try std.testing.expectError(error.TypeUnsupported, where_table.withColumnWhere("bad_where_column", "metric", "mask", "mask"));
    try std.testing.expectError(error.ColumnNotFound, where_table.withColumnWhere("missing_where_column", "metric", "mask", "missing"));

    var masked_put_table = try where_table.withColumnMaskedPutScalar("metric_masked", "metric", "mask", f64, 9.0);
    defer masked_put_table.deinit();
    const metric_masked = try (try masked_put_table.column("metric_masked")).f64.toOwnedSlice(gpa);
    defer gpa.free(metric_masked);
    try std.testing.expectEqualSlices(f64, &.{ 9.0, 2.0, 9.0 }, metric_masked);
    var put_mask_table = try where_table.withColumnPutMaskScalar("metric_put_mask", "metric", "mask", f64, -3.0);
    defer put_mask_table.deinit();
    const metric_put_mask = try (try put_mask_table.column("metric_put_mask")).f64.toOwnedSlice(gpa);
    defer gpa.free(metric_put_mask);
    try std.testing.expectEqualSlices(f64, &.{ -3.0, 2.0, -3.0 }, metric_put_mask);
    try std.testing.expectError(error.TypeUnsupported, table.withColumnMaskedPutScalar("bad_masked", "sales", "cost", f64, 9.0));
    try std.testing.expectError(error.ColumnNotFound, where_table.withColumnMaskedPutScalar("missing_masked", "metric", "missing", f64, 9.0));

    var unit_replacements = try DeviceColumn.fromSliceWithValidity(i64, gpa, &.{ 4, 5, 6 }, &.{ true, true, false }, .cpu);
    defer unit_replacements.deinit();
    var put_values_source = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "units", .data = units },
        .{ .name = "unit_replacements", .data = unit_replacements },
    });
    defer put_values_source.deinit();
    var units_put_values_table = try put_values_source.withColumnPutFlat("units_put_values", "units", &.{ 2, 0, 2 }, "unit_replacements");
    defer units_put_values_table.deinit();
    const units_put_values = try (try units_put_values_table.column("units_put_values")).i64.toOwnedSlice(gpa);
    defer gpa.free(units_put_values);
    const units_put_values_validity = try (try units_put_values_table.column("units_put_values")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(units_put_values_validity);
    try std.testing.expectEqualSlices(i64, &.{ 5, 2, 6 }, units_put_values);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false }, units_put_values_validity);
    try std.testing.expectError(error.TypeUnsupported, table.withColumnPutFlat("bad_put_values", "sales", &.{ 0, 1, 2 }, "units"));
    try std.testing.expectError(error.ShapeMismatch, put_values_source.withColumnPutFlat("bad_put_values_shape", "units", &.{ 0, 1 }, "unit_replacements"));

    var units_put_flat_table = try table.withColumnPutFlatScalar("units_put", "units", &.{1}, i64, 9);
    defer units_put_flat_table.deinit();
    const units_put = try (try units_put_flat_table.column("units_put")).i64.toOwnedSlice(gpa);
    defer gpa.free(units_put);
    const units_put_validity = try (try units_put_flat_table.column("units_put")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(units_put_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 9, 3 }, units_put);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true }, units_put_validity);

    var units_index_put_table = try table.withColumnIndexPutScalar("units_index_put", "units", &.{ 0, 2 }, i64, -1);
    defer units_index_put_table.deinit();
    const units_index_put = try (try units_index_put_table.column("units_index_put")).i64.toOwnedSlice(gpa);
    defer gpa.free(units_index_put);
    try std.testing.expectEqualSlices(i64, &.{ -1, 2, -1 }, units_index_put);
    try std.testing.expectError(error.IndexOutOfBounds, table.withColumnPutFlatScalar("bad_put_flat", "sales", &.{table.height()}, f64, 0.0));

    var units_put_signed_table = try table.withColumnPutFlatScalarSigned("units_put_signed", "units", &.{-1}, i64, 7);
    defer units_put_signed_table.deinit();
    const units_put_signed = try (try units_put_signed_table.column("units_put_signed")).i64.toOwnedSlice(gpa);
    defer gpa.free(units_put_signed);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 7 }, units_put_signed);
    try std.testing.expectError(error.IndexOutOfBounds, table.withColumnPutFlatScalarSigned("bad_put_signed", "sales", &.{-4}, f64, 0.0));

    var units_put_wrap_table = try table.withColumnPutFlatScalarMode("units_put_wrap", "units", &.{table.height() + 1}, i64, 8, .wrap);
    defer units_put_wrap_table.deinit();
    const units_put_wrap = try (try units_put_wrap_table.column("units_put_wrap")).i64.toOwnedSlice(gpa);
    defer gpa.free(units_put_wrap);
    try std.testing.expectEqualSlices(i64, &.{ 1, 8, 3 }, units_put_wrap);

    var units_put_clip_table = try table.withColumnPutFlatScalarMode("units_put_clip", "units", &.{table.height() + 10}, i64, 6, .clip);
    defer units_put_clip_table.deinit();
    const units_put_clip = try (try units_put_clip_table.column("units_put_clip")).i64.toOwnedSlice(gpa);
    defer gpa.free(units_put_clip);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 6 }, units_put_clip);

    var active_and_table = try rounding_type_table.withColumnLogicalAndScalar("active_and", "active", false);
    defer active_and_table.deinit();
    const active_and = try (try active_and_table.column("active_and")).bool.toOwnedSlice(gpa);
    defer gpa.free(active_and);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false }, active_and);

    var active_or_table = try rounding_type_table.withColumnLogicalOrScalar("active_or", "active", false);
    defer active_or_table.deinit();
    const active_or = try (try active_or_table.column("active_or")).bool.toOwnedSlice(gpa);
    defer gpa.free(active_or);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, active_or);

    var active_xor_table = try rounding_type_table.withColumnLogicalXorScalar("active_xor", "active", true);
    defer active_xor_table.deinit();
    const active_xor = try (try active_xor_table.column("active_xor")).bool.toOwnedSlice(gpa);
    defer gpa.free(active_xor);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false }, active_xor);
    try std.testing.expectError(error.TypeUnsupported, table.withColumnLogicalAndScalar("bad_logical", "sales", true));
    try std.testing.expectError(error.ColumnNotFound, rounding_type_table.withColumnLogicalXorScalar("missing_logical", "missing", true));

    var bool_rhs = try DeviceColumn.fromSlice(bool, gpa, &.{ false, false, true }, .cpu);
    defer bool_rhs.deinit();
    var bool_pair_table = try DeviceDataFrame.init(gpa, &.{ .{ .name = "lhs", .data = rounding_active }, .{ .name = "rhs", .data = bool_rhs } });
    defer bool_pair_table.deinit();
    var logical_pair_table = try bool_pair_table.withColumnLogicalOr("lhs_or_rhs", "lhs", "rhs");
    defer logical_pair_table.deinit();
    const lhs_or_rhs = try (try logical_pair_table.column("lhs_or_rhs")).bool.toOwnedSlice(gpa);
    defer gpa.free(lhs_or_rhs);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, lhs_or_rhs);
    var logical_xor_pair_table = try bool_pair_table.withColumnLogicalXor("lhs_xor_rhs", "lhs", "rhs");
    defer logical_xor_pair_table.deinit();
    const lhs_xor_rhs = try (try logical_xor_pair_table.column("lhs_xor_rhs")).bool.toOwnedSlice(gpa);
    defer gpa.free(lhs_xor_rhs);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false }, lhs_xor_rhs);
    try std.testing.expectError(error.TypeUnsupported, table.withColumnLogicalAnd("bad_logical_pair", "sales", "cost"));
    try std.testing.expectError(error.ColumnNotFound, bool_pair_table.withColumnLogicalAnd("missing_logical_pair", "lhs", "missing"));

    var neg_sales_table = try table.withColumnNeg("sales_neg", "sales");
    defer neg_sales_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try neg_sales_table.columnDType("sales_neg"));
    const sales_neg = try (try neg_sales_table.column("sales_neg")).f64.toOwnedSlice(gpa);
    defer gpa.free(sales_neg);
    try std.testing.expectEqualSlices(f64, &.{ -2.0, -3.0, -5.0 }, sales_neg);
    try std.testing.expectError(error.ColumnNotFound, table.withColumnNeg("bad_neg", "missing"));

    var sign_sales_table = try neg_sales_table.withColumnSign("sales_neg_sign", "sales_neg");
    defer sign_sales_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try sign_sales_table.columnDType("sales_neg_sign"));
    const sales_neg_sign = try (try sign_sales_table.column("sales_neg_sign")).f64.toOwnedSlice(gpa);
    defer gpa.free(sales_neg_sign);
    try std.testing.expectEqualSlices(f64, &.{ -1.0, -1.0, -1.0 }, sales_neg_sign);
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.withColumnSign("bad_sign", "active"));
    try std.testing.expectError(error.ColumnNotFound, table.withColumnSign("missing_sign", "missing"));

    var sign_units_table = try table.withColumnSign("units_sign", "units");
    defer sign_units_table.deinit();
    try std.testing.expectEqual(DeviceDType.i64, try sign_units_table.columnDType("units_sign"));
    const units_sign = try (try sign_units_table.column("units_sign")).i64.toOwnedSlice(gpa);
    defer gpa.free(units_sign);
    try std.testing.expectEqualSlices(i64, &.{ 1, 1, 1 }, units_sign);

    var square_sales_table = try table.withColumnSquare("sales_square", "sales");
    defer square_sales_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try square_sales_table.columnDType("sales_square"));
    const sales_square = try (try square_sales_table.column("sales_square")).f64.toOwnedSlice(gpa);
    defer gpa.free(sales_square);
    try std.testing.expectEqualSlices(f64, &.{ 4.0, 9.0, 25.0 }, sales_square);
    try std.testing.expectError(error.ColumnNotFound, table.withColumnSquare("bad_square", "missing"));

    var reciprocal_sales_table = try table.withColumnReciprocal("sales_recip", "sales");
    defer reciprocal_sales_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try reciprocal_sales_table.columnDType("sales_recip"));
    const sales_recip = try (try reciprocal_sales_table.column("sales_recip")).f64.toOwnedSlice(gpa);
    defer gpa.free(sales_recip);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), sales_recip[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), sales_recip[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.2), sales_recip[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, table.withColumnReciprocal("bad_recip", "units"));
    try std.testing.expectError(error.ColumnNotFound, table.withColumnReciprocal("missing_recip", "missing"));

    var sqrt_sales_table = try table.withColumnSqrt("sales_sqrt", "sales");
    defer sqrt_sales_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try sqrt_sales_table.columnDType("sales_sqrt"));
    const sales_sqrt = try (try sqrt_sales_table.column("sales_sqrt")).f64.toOwnedSlice(gpa);
    defer gpa.free(sales_sqrt);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 2.0)), sales_sqrt[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 3.0)), sales_sqrt[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 5.0)), sales_sqrt[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, table.withColumnSqrt("bad_sqrt", "units"));
    try std.testing.expectError(error.ColumnNotFound, table.withColumnSqrt("missing_sqrt", "missing"));

    var rsqrt_sales_table = try table.withColumnRsqrt("sales_rsqrt", "sales");
    defer rsqrt_sales_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try rsqrt_sales_table.columnDType("sales_rsqrt"));
    const sales_rsqrt = try (try rsqrt_sales_table.column("sales_rsqrt")).f64.toOwnedSlice(gpa);
    defer gpa.free(sales_rsqrt);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0) / std.math.sqrt(@as(f64, 2.0)), sales_rsqrt[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0) / std.math.sqrt(@as(f64, 3.0)), sales_rsqrt[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0) / std.math.sqrt(@as(f64, 5.0)), sales_rsqrt[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, table.withColumnRsqrt("bad_rsqrt", "units"));
    try std.testing.expectError(error.ColumnNotFound, table.withColumnRsqrt("missing_rsqrt", "missing"));

    var cbrt_sales_table = try table.withColumnCbrt("sales_cbrt", "sales");
    defer cbrt_sales_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try cbrt_sales_table.columnDType("sales_cbrt"));
    const sales_cbrt = try (try cbrt_sales_table.column("sales_cbrt")).f64.toOwnedSlice(gpa);
    defer gpa.free(sales_cbrt);
    try std.testing.expectApproxEqAbs(std.math.cbrt(@as(f64, 2.0)), sales_cbrt[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.cbrt(@as(f64, 3.0)), sales_cbrt[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.cbrt(@as(f64, 5.0)), sales_cbrt[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, table.withColumnCbrt("bad_cbrt", "units"));
    try std.testing.expectError(error.ColumnNotFound, table.withColumnCbrt("missing_cbrt", "missing"));

    var ratio = try DeviceColumn.fromSlice(f64, gpa, &.{ -0.5, 0.0, 0.5 }, .cpu);
    defer ratio.deinit();
    var inverse_units = try DeviceColumn.fromSlice(i64, gpa, &.{ 1, 2, 3 }, .cpu);
    defer inverse_units.deinit();
    var inverse_trig_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "ratio", .data = ratio },
        .{ .name = "units", .data = inverse_units },
    });
    defer inverse_trig_table.deinit();

    var floor_cost_table = try table.withColumnFloor("cost_floor", "cost");
    defer floor_cost_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try floor_cost_table.columnDType("cost_floor"));
    const cost_floor = try (try floor_cost_table.column("cost_floor")).f64.toOwnedSlice(gpa);
    defer gpa.free(cost_floor);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 1.0, 2.0 }, cost_floor);
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.withColumnFloor("bad_floor", "active"));
    try std.testing.expectError(error.ColumnNotFound, table.withColumnFloor("missing_floor", "missing"));

    var ceil_cost_table = try table.withColumnCeil("cost_ceil", "cost");
    defer ceil_cost_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try ceil_cost_table.columnDType("cost_ceil"));
    const cost_ceil = try (try ceil_cost_table.column("cost_ceil")).f64.toOwnedSlice(gpa);
    defer gpa.free(cost_ceil);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 2.0, 2.0 }, cost_ceil);
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.withColumnCeil("bad_ceil", "active"));
    try std.testing.expectError(error.ColumnNotFound, table.withColumnCeil("missing_ceil", "missing"));

    var round_cost_table = try table.withColumnRound("cost_round", "cost");
    defer round_cost_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try round_cost_table.columnDType("cost_round"));
    const cost_round = try (try round_cost_table.column("cost_round")).f64.toOwnedSlice(gpa);
    defer gpa.free(cost_round);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 2.0, 2.0 }, cost_round);
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.withColumnRound("bad_round", "active"));
    try std.testing.expectError(error.ColumnNotFound, table.withColumnRound("missing_round", "missing"));

    var trunc_cost_table = try table.withColumnTrunc("cost_trunc", "cost");
    defer trunc_cost_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try trunc_cost_table.columnDType("cost_trunc"));
    const cost_trunc = try (try trunc_cost_table.column("cost_trunc")).f64.toOwnedSlice(gpa);
    defer gpa.free(cost_trunc);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 1.0, 2.0 }, cost_trunc);
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.withColumnTrunc("bad_trunc", "active"));
    try std.testing.expectError(error.ColumnNotFound, table.withColumnTrunc("missing_trunc", "missing"));

    var floor_units_table = try table.withColumnFloor("units_floor", "units");
    defer floor_units_table.deinit();
    try std.testing.expectEqual(DeviceDType.i64, try floor_units_table.columnDType("units_floor"));
    const units_floor = try (try floor_units_table.column("units_floor")).i64.toOwnedSlice(gpa);
    defer gpa.free(units_floor);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3 }, units_floor);

    var deg2rad_cost_table = try table.withColumnDeg2rad("cost_rad", "cost");
    defer deg2rad_cost_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try deg2rad_cost_table.columnDType("cost_rad"));
    const cost_rad = try (try deg2rad_cost_table.column("cost_rad")).f64.toOwnedSlice(gpa);
    defer gpa.free(cost_rad);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0) * std.math.pi / @as(f64, 180.0), cost_rad[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5) * std.math.pi / @as(f64, 180.0), cost_rad[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0) * std.math.pi / @as(f64, 180.0), cost_rad[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, table.withColumnDeg2rad("bad_deg2rad", "units"));
    try std.testing.expectError(error.ColumnNotFound, table.withColumnDeg2rad("missing_deg2rad", "missing"));

    var rad2deg_cost_table = try deg2rad_cost_table.withColumnRad2deg("cost_deg", "cost_rad");
    defer rad2deg_cost_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try rad2deg_cost_table.columnDType("cost_deg"));
    const cost_deg = try (try rad2deg_cost_table.column("cost_deg")).f64.toOwnedSlice(gpa);
    defer gpa.free(cost_deg);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), cost_deg[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), cost_deg[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), cost_deg[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, table.withColumnRad2deg("bad_rad2deg", "units"));
    try std.testing.expectError(error.ColumnNotFound, table.withColumnRad2deg("missing_rad2deg", "missing"));

    var expit_ratio_table = try inverse_trig_table.withColumnExpit("ratio_expit", "ratio");
    defer expit_ratio_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try expit_ratio_table.columnDType("ratio_expit"));
    const ratio_expit = try (try expit_ratio_table.column("ratio_expit")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_expit);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0) / (@as(f64, 1.0) + std.math.exp(@as(f64, 0.5))), ratio_expit[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), ratio_expit[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0) / (@as(f64, 1.0) + std.math.exp(@as(f64, -0.5))), ratio_expit[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, inverse_trig_table.withColumnExpit("bad_expit", "units"));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnExpit("missing_expit", "missing"));

    var logit_ratio_table = try inverse_trig_table.withColumnLogit("ratio_logit", "ratio");
    defer logit_ratio_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try logit_ratio_table.columnDType("ratio_logit"));
    const ratio_logit = try (try logit_ratio_table.column("ratio_logit")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_logit);
    try std.testing.expect(std.math.isNan(ratio_logit[0]));
    try std.testing.expect(std.math.isNegativeInf(ratio_logit[1]));
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ratio_logit[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, inverse_trig_table.withColumnLogit("bad_logit", "units"));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnLogit("missing_logit", "missing"));

    var softplus_ratio_table = try inverse_trig_table.withColumnSoftplus("ratio_softplus", "ratio");
    defer softplus_ratio_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try softplus_ratio_table.columnDType("ratio_softplus"));
    const ratio_softplus = try (try softplus_ratio_table.column("ratio_softplus")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_softplus);
    try std.testing.expectApproxEqAbs(@max(@as(f64, -0.5), @as(f64, 0.0)) + std.math.log1p(std.math.exp(-@abs(@as(f64, -0.5)))), ratio_softplus[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log1p(@as(f64, 1.0)), ratio_softplus[1], 1e-12);
    try std.testing.expectApproxEqAbs(@max(@as(f64, 0.5), @as(f64, 0.0)) + std.math.log1p(std.math.exp(-@abs(@as(f64, 0.5)))), ratio_softplus[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, inverse_trig_table.withColumnSoftplus("bad_softplus", "units"));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnSoftplus("missing_softplus", "missing"));

    var logsigmoid_ratio_table = try inverse_trig_table.withColumnLogsigmoid("ratio_logsigmoid", "ratio");
    defer logsigmoid_ratio_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try logsigmoid_ratio_table.columnDType("ratio_logsigmoid"));
    const ratio_logsigmoid = try (try logsigmoid_ratio_table.column("ratio_logsigmoid")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_logsigmoid);
    try std.testing.expectApproxEqAbs(-(@max(@as(f64, 0.5), @as(f64, 0.0)) + std.math.log1p(std.math.exp(-@abs(@as(f64, -0.5))))), ratio_logsigmoid[0], 1e-12);
    try std.testing.expectApproxEqAbs(-std.math.log1p(@as(f64, 1.0)), ratio_logsigmoid[1], 1e-12);
    try std.testing.expectApproxEqAbs(-(@max(@as(f64, -0.5), @as(f64, 0.0)) + std.math.log1p(std.math.exp(-@abs(@as(f64, 0.5))))), ratio_logsigmoid[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, inverse_trig_table.withColumnLogsigmoid("bad_logsigmoid", "units"));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnLogsigmoid("missing_logsigmoid", "missing"));

    var relu_ratio_table = try inverse_trig_table.withColumnRelu("ratio_relu", "ratio");
    defer relu_ratio_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try relu_ratio_table.columnDType("ratio_relu"));
    const ratio_relu = try (try relu_ratio_table.column("ratio_relu")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_relu);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.5 }, ratio_relu);
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.withColumnRelu("bad_relu", "active"));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnRelu("missing_relu", "missing"));

    var leaky_relu_ratio_table = try inverse_trig_table.withColumnLeakyRelu("ratio_leaky_relu", "ratio", f64, 0.1);
    defer leaky_relu_ratio_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try leaky_relu_ratio_table.columnDType("ratio_leaky_relu"));
    const ratio_leaky_relu = try (try leaky_relu_ratio_table.column("ratio_leaky_relu")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_leaky_relu);
    try std.testing.expectApproxEqAbs(@as(f64, -0.05), ratio_leaky_relu[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ratio_leaky_relu[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), ratio_leaky_relu[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.withColumnLeakyRelu("bad_leaky_relu", "active", f64, 0.1));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnLeakyRelu("missing_leaky_relu", "missing", f64, 0.1));

    var nullable_ratio = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ -2.0, 3.0, -4.0 }, &.{ true, false, true }, .cpu);
    defer nullable_ratio.deinit();
    var nullable_ratio_table = try DeviceDataFrame.init(gpa, &.{.{ .name = "ratio", .data = nullable_ratio }});
    defer nullable_ratio_table.deinit();
    var nullable_leaky_relu_table = try nullable_ratio_table.withColumnLeakyRelu("ratio_leaky_relu", "ratio", f64, 0.25);
    defer nullable_leaky_relu_table.deinit();
    const nullable_leaky_relu_column = try nullable_leaky_relu_table.column("ratio_leaky_relu");
    try std.testing.expect(nullable_leaky_relu_column.f64.nullable());
    try std.testing.expectEqual(@as(usize, 1), nullable_leaky_relu_column.f64.null_count);
    const nullable_leaky_relu = try nullable_leaky_relu_column.f64.toOwnedSlice(gpa);
    defer gpa.free(nullable_leaky_relu);
    try std.testing.expectApproxEqAbs(@as(f64, -0.5), nullable_leaky_relu[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), nullable_leaky_relu[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -1.0), nullable_leaky_relu[2], 1e-12);

    var leaky_relu_units_table = try table.withColumnLeakyRelu("units_leaky_relu", "units", i64, 2);
    defer leaky_relu_units_table.deinit();
    try std.testing.expectEqual(DeviceDType.i64, try leaky_relu_units_table.columnDType("units_leaky_relu"));
    const units_leaky_relu = try (try leaky_relu_units_table.column("units_leaky_relu")).i64.toOwnedSlice(gpa);
    defer gpa.free(units_leaky_relu);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3 }, units_leaky_relu);

    var signed_units = try DeviceColumn.fromSlice(i64, gpa, &.{ -2, 3, -4 }, .cpu);
    defer signed_units.deinit();
    var signed_units_table = try DeviceDataFrame.init(gpa, &.{.{ .name = "signed_units", .data = signed_units }});
    defer signed_units_table.deinit();
    var signed_units_leaky_relu_table = try signed_units_table.withColumnLeakyReluWithDeviceScalar("signed_units_leaky_relu", "signed_units", .{ .f64 = 2.0 });
    defer signed_units_leaky_relu_table.deinit();
    const signed_units_leaky_relu = try (try signed_units_leaky_relu_table.column("signed_units_leaky_relu")).i64.toOwnedSlice(gpa);
    defer gpa.free(signed_units_leaky_relu);
    try std.testing.expectEqualSlices(i64, &.{ -4, 3, -8 }, signed_units_leaky_relu);
    try std.testing.expectError(error.TypeUnsupported, signed_units_table.withColumnLeakyReluWithDeviceScalar("bad_fractional_slope", "signed_units", .{ .f64 = 0.5 }));

    var relu6_cost_table = try table.withColumnRelu6("cost_relu6", "cost");
    defer relu6_cost_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try relu6_cost_table.columnDType("cost_relu6"));
    const cost_relu6 = try (try relu6_cost_table.column("cost_relu6")).f64.toOwnedSlice(gpa);
    defer gpa.free(cost_relu6);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 1.5, 2.0 }, cost_relu6);
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.withColumnRelu6("bad_relu6", "active"));
    try std.testing.expectError(error.ColumnNotFound, table.withColumnRelu6("missing_relu6", "missing"));

    var pow_ratio_table = try inverse_trig_table.withColumnPowScalar("ratio_pow", "ratio", f64, 2.0);
    defer pow_ratio_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try pow_ratio_table.columnDType("ratio_pow"));
    const ratio_pow = try (try pow_ratio_table.column("ratio_pow")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_pow);
    try std.testing.expectEqualSlices(f64, &.{ 0.25, 0.0, 0.25 }, ratio_pow);
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.withColumnPowScalar("bad_pow", "active", f64, 2.0));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnPowScalar("missing_pow", "missing", f64, 2.0));

    var pow_units_table = try table.withColumnPowWithDeviceScalar("units_pow", "units", .{ .f64 = 2.0 });
    defer pow_units_table.deinit();
    try std.testing.expectEqual(DeviceDType.i64, try pow_units_table.columnDType("units_pow"));
    const units_pow = try (try pow_units_table.column("units_pow")).i64.toOwnedSlice(gpa);
    defer gpa.free(units_pow);
    try std.testing.expectEqualSlices(i64, &.{ 1, 4, 9 }, units_pow);
    try std.testing.expectError(error.TypeUnsupported, table.withColumnPowWithDeviceScalar("bad_fractional_pow", "units", .{ .f64 = 2.5 }));

    var floor_div_units_table = try signed_units_table.withColumnFloorDivWithDeviceScalar("signed_units_floor_div", "signed_units", .{ .f64 = 2.0 });
    defer floor_div_units_table.deinit();
    try std.testing.expectEqual(DeviceDType.i64, try floor_div_units_table.columnDType("signed_units_floor_div"));
    const signed_units_floor_div = try (try floor_div_units_table.column("signed_units_floor_div")).i64.toOwnedSlice(gpa);
    defer gpa.free(signed_units_floor_div);
    try std.testing.expectEqualSlices(i64, &.{ -1, 1, -2 }, signed_units_floor_div);
    try std.testing.expectError(error.TypeUnsupported, signed_units_table.withColumnFloorDivWithDeviceScalar("bad_fractional_floor_div", "signed_units", .{ .f64 = 2.5 }));

    var mod_units_table = try signed_units_table.withColumnModScalar("signed_units_mod", "signed_units", i64, 3);
    defer mod_units_table.deinit();
    try std.testing.expectEqual(DeviceDType.i64, try mod_units_table.columnDType("signed_units_mod"));
    const signed_units_mod = try (try mod_units_table.column("signed_units_mod")).i64.toOwnedSlice(gpa);
    defer gpa.free(signed_units_mod);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 2 }, signed_units_mod);

    var remainder_units_table = try signed_units_table.withColumnRemainderScalar("signed_units_remainder", "signed_units", i64, 3);
    defer remainder_units_table.deinit();
    try std.testing.expectEqual(DeviceDType.i64, try remainder_units_table.columnDType("signed_units_remainder"));
    const signed_units_remainder = try (try remainder_units_table.column("signed_units_remainder")).i64.toOwnedSlice(gpa);
    defer gpa.free(signed_units_remainder);
    try std.testing.expectEqualSlices(i64, signed_units_mod, signed_units_remainder);
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.withColumnModScalar("bad_mod", "active", i64, 3));
    try std.testing.expectError(error.ColumnNotFound, signed_units_table.withColumnRemainderScalar("missing_remainder", "missing", i64, 3));

    var ratio_mod_table = try inverse_trig_table.withColumnModScalar("ratio_mod", "ratio", f64, 0.4);
    defer ratio_mod_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try ratio_mod_table.columnDType("ratio_mod"));
    const ratio_mod = try (try ratio_mod_table.column("ratio_mod")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_mod);
    try std.testing.expectApproxEqAbs(@mod(@as(f64, -0.5), @as(f64, 0.4)), ratio_mod[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ratio_mod[1], 1e-12);
    try std.testing.expectApproxEqAbs(@mod(@as(f64, 0.5), @as(f64, 0.4)), ratio_mod[2], 1e-12);

    var logaddexp_ratio_table = try inverse_trig_table.withColumnLogAddExpScalar("ratio_logaddexp", "ratio", f64, 0.0);
    defer logaddexp_ratio_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try logaddexp_ratio_table.columnDType("ratio_logaddexp"));
    const ratio_logaddexp = try (try logaddexp_ratio_table.column("ratio_logaddexp")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_logaddexp);
    try std.testing.expectApproxEqAbs(@max(@as(f64, -0.5), @as(f64, 0.0)) + std.math.log1p(std.math.exp(-@abs(@as(f64, -0.5)))), ratio_logaddexp[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.ln2, ratio_logaddexp[1], 1e-12);
    try std.testing.expectApproxEqAbs(@max(@as(f64, 0.5), @as(f64, 0.0)) + std.math.log1p(std.math.exp(-@abs(@as(f64, 0.5)))), ratio_logaddexp[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, inverse_trig_table.withColumnLogAddExpScalar("bad_logaddexp", "units", f64, 0.0));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnLogAddExpScalar("missing_logaddexp", "missing", f64, 0.0));

    var logaddexp2_ratio_table = try inverse_trig_table.withColumnLogAddExp2Scalar("ratio_logaddexp2", "ratio", f64, 0.0);
    defer logaddexp2_ratio_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try logaddexp2_ratio_table.columnDType("ratio_logaddexp2"));
    const ratio_logaddexp2 = try (try logaddexp2_ratio_table.column("ratio_logaddexp2")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_logaddexp2);
    try std.testing.expectApproxEqAbs(@max(@as(f64, -0.5), @as(f64, 0.0)) + std.math.log2(@as(f64, 1.0) + std.math.pow(f64, 2.0, -@abs(@as(f64, -0.5)))), ratio_logaddexp2[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), ratio_logaddexp2[1], 1e-12);
    try std.testing.expectApproxEqAbs(@max(@as(f64, 0.5), @as(f64, 0.0)) + std.math.log2(@as(f64, 1.0) + std.math.pow(f64, 2.0, -@abs(@as(f64, 0.5)))), ratio_logaddexp2[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, inverse_trig_table.withColumnLogAddExp2Scalar("bad_logaddexp2", "units", f64, 0.0));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnLogAddExp2Scalar("missing_logaddexp2", "missing", f64, 0.0));

    var xlogy_ratio_table = try inverse_trig_table.withColumnXlogyScalar("ratio_xlogy", "ratio", f64, std.math.e);
    defer xlogy_ratio_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try xlogy_ratio_table.columnDType("ratio_xlogy"));
    const ratio_xlogy = try (try xlogy_ratio_table.column("ratio_xlogy")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_xlogy);
    try std.testing.expectApproxEqAbs(@as(f64, -0.5), ratio_xlogy[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ratio_xlogy[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), ratio_xlogy[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, inverse_trig_table.withColumnXlogyScalar("bad_xlogy", "units", f64, std.math.e));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnXlogyScalar("missing_xlogy", "missing", f64, std.math.e));

    var fmax_ratio_table = try inverse_trig_table.withColumnFmaxScalar("ratio_fmax", "ratio", f64, 0.25);
    defer fmax_ratio_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try fmax_ratio_table.columnDType("ratio_fmax"));
    const ratio_fmax = try (try fmax_ratio_table.column("ratio_fmax")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_fmax);
    try std.testing.expectEqualSlices(f64, &.{ 0.25, 0.25, 0.5 }, ratio_fmax);
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.withColumnFmaxScalar("bad_fmax", "active", f64, 0.25));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnFmaxScalar("missing_fmax", "missing", f64, 0.25));

    var fmin_ratio_table = try inverse_trig_table.withColumnFminScalar("ratio_fmin", "ratio", f64, 0.25);
    defer fmin_ratio_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try fmin_ratio_table.columnDType("ratio_fmin"));
    const ratio_fmin = try (try fmin_ratio_table.column("ratio_fmin")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_fmin);
    try std.testing.expectEqualSlices(f64, &.{ -0.5, 0.0, 0.25 }, ratio_fmin);
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.withColumnFminScalar("bad_fmin", "active", f64, 0.25));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnFminScalar("missing_fmin", "missing", f64, 0.25));

    var hypot_ratio_table = try inverse_trig_table.withColumnHypotWithDeviceScalar("ratio_hypot", "ratio", .{ .f32 = 0.5 });
    defer hypot_ratio_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try hypot_ratio_table.columnDType("ratio_hypot"));
    const ratio_hypot = try (try hypot_ratio_table.column("ratio_hypot")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_hypot);
    try std.testing.expectApproxEqAbs(std.math.hypot(@as(f64, -0.5), @as(f64, 0.5)), ratio_hypot[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), ratio_hypot[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.hypot(@as(f64, 0.5), @as(f64, 0.5)), ratio_hypot[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, inverse_trig_table.withColumnHypotScalar("bad_hypot", "units", f64, 0.5));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnHypotScalar("missing_hypot", "missing", f64, 0.5));

    var atan2_ratio_table = try inverse_trig_table.withColumnAtan2Scalar("ratio_atan2", "ratio", f64, 0.5);
    defer atan2_ratio_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try atan2_ratio_table.columnDType("ratio_atan2"));
    const ratio_atan2 = try (try atan2_ratio_table.column("ratio_atan2")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_atan2);
    try std.testing.expectApproxEqAbs(std.math.atan2(@as(f64, -0.5), @as(f64, 0.5)), ratio_atan2[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ratio_atan2[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.atan2(@as(f64, 0.5), @as(f64, 0.5)), ratio_atan2[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, inverse_trig_table.withColumnAtan2WithDeviceScalar("bad_atan2", "units", .{ .f64 = 0.5 }));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnAtan2Scalar("missing_atan2", "missing", f64, 0.5));

    var next_after_ratio_table = try inverse_trig_table.withColumnNextAfterScalar("ratio_next_after", "ratio", f64, 1.0);
    defer next_after_ratio_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try next_after_ratio_table.columnDType("ratio_next_after"));
    const ratio_next_after = try (try next_after_ratio_table.column("ratio_next_after")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_next_after);
    try std.testing.expectEqual(std.math.nextAfter(f64, @as(f64, -0.5), @as(f64, 1.0)), ratio_next_after[0]);
    try std.testing.expectEqual(std.math.nextAfter(f64, @as(f64, 0.0), @as(f64, 1.0)), ratio_next_after[1]);
    try std.testing.expectEqual(std.math.nextAfter(f64, @as(f64, 0.5), @as(f64, 1.0)), ratio_next_after[2]);
    try std.testing.expectError(error.TypeUnsupported, inverse_trig_table.withColumnNextAfterScalar("bad_next_after", "units", f64, 1.0));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnNextAfterScalar("missing_next_after", "missing", f64, 1.0));

    var copysign_ratio_table = try inverse_trig_table.withColumnCopysignWithDeviceScalar("ratio_copysign", "ratio", .{ .f64 = -1.0 });
    defer copysign_ratio_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try copysign_ratio_table.columnDType("ratio_copysign"));
    const ratio_copysign = try (try copysign_ratio_table.column("ratio_copysign")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_copysign);
    try std.testing.expectApproxEqAbs(@as(f64, -0.5), ratio_copysign[0], 1e-12);
    try std.testing.expectEqual(@as(f64, -0.0), ratio_copysign[1]);
    try std.testing.expect(std.math.signbit(ratio_copysign[1]));
    try std.testing.expectApproxEqAbs(@as(f64, -0.5), ratio_copysign[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, inverse_trig_table.withColumnCopysignScalar("bad_copysign", "units", f64, -1.0));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnCopysignScalar("missing_copysign", "missing", f64, -1.0));

    var heaviside_ratio_table = try inverse_trig_table.withColumnHeavisideScalar("ratio_heaviside", "ratio", f64, 0.25);
    defer heaviside_ratio_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try heaviside_ratio_table.columnDType("ratio_heaviside"));
    const ratio_heaviside = try (try heaviside_ratio_table.column("ratio_heaviside")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_heaviside);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.25, 1.0 }, ratio_heaviside);
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.withColumnHeavisideScalar("bad_heaviside", "active", f64, 0.25));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnHeavisideScalar("missing_heaviside", "missing", f64, 0.25));

    var heaviside_units_table = try inverse_trig_table.withColumnHeavisideWithDeviceScalar("units_heaviside", "units", .{ .i64 = 9 });
    defer heaviside_units_table.deinit();
    try std.testing.expectEqual(DeviceDType.i64, try heaviside_units_table.columnDType("units_heaviside"));
    const units_heaviside = try (try heaviside_units_table.column("units_heaviside")).i64.toOwnedSlice(gpa);
    defer gpa.free(units_heaviside);
    try std.testing.expectEqualSlices(i64, &.{ 1, 1, 1 }, units_heaviside);
    try std.testing.expectError(error.TypeUnsupported, inverse_trig_table.withColumnHeavisideWithDeviceScalar("bad_fractional_heaviside", "units", .{ .f64 = 0.5 }));

    var ldexp_ratio_table = try inverse_trig_table.withColumnLdexpScalar("ratio_ldexp", "ratio", 2);
    defer ldexp_ratio_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try ldexp_ratio_table.columnDType("ratio_ldexp"));
    const ratio_ldexp = try (try ldexp_ratio_table.column("ratio_ldexp")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_ldexp);
    try std.testing.expectEqualSlices(f64, &.{ -2.0, 0.0, 2.0 }, ratio_ldexp);
    try std.testing.expectError(error.TypeUnsupported, inverse_trig_table.withColumnLdexpScalar("bad_ldexp", "units", 2));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnLdexpScalar("missing_ldexp", "missing", 2));

    var nan_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ std.math.nan(f64), -1.0, 2.0 }, .cpu);
    defer nan_metric.deinit();
    var nan_table = try DeviceDataFrame.init(gpa, &.{.{ .name = "metric", .data = nan_metric }});
    defer nan_table.deinit();
    var fmax_nan_table = try nan_table.withColumnFmaxScalar("metric_fmax", "metric", f64, 0.5);
    defer fmax_nan_table.deinit();
    const metric_fmax = try (try fmax_nan_table.column("metric_fmax")).f64.toOwnedSlice(gpa);
    defer gpa.free(metric_fmax);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), metric_fmax[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), metric_fmax[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), metric_fmax[2], 1e-12);

    var threshold_ratio_table = try inverse_trig_table.withColumnThreshold("ratio_threshold", "ratio", f64, -0.25, 1.0);
    defer threshold_ratio_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try threshold_ratio_table.columnDType("ratio_threshold"));
    const ratio_threshold = try (try threshold_ratio_table.column("ratio_threshold")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_threshold);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 0.0, 0.5 }, ratio_threshold);
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.withColumnThreshold("bad_threshold", "active", f64, -0.25, 1.0));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnThreshold("missing_threshold", "missing", f64, -0.25, 1.0));

    var threshold_units_table = try table.withColumnThreshold("units_threshold", "units", i64, 2, 0);
    defer threshold_units_table.deinit();
    try std.testing.expectEqual(DeviceDType.i64, try threshold_units_table.columnDType("units_threshold"));
    const units_threshold = try (try threshold_units_table.column("units_threshold")).i64.toOwnedSlice(gpa);
    defer gpa.free(units_threshold);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 3 }, units_threshold);

    var hardtanh_ratio_table = try inverse_trig_table.withColumnHardtanh("ratio_hardtanh", "ratio", f64, -0.25, 0.25);
    defer hardtanh_ratio_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try hardtanh_ratio_table.columnDType("ratio_hardtanh"));
    const ratio_hardtanh = try (try hardtanh_ratio_table.column("ratio_hardtanh")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_hardtanh);
    try std.testing.expectEqualSlices(f64, &.{ -0.25, 0.0, 0.25 }, ratio_hardtanh);
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.withColumnHardtanh("bad_hardtanh", "active", f64, -0.25, 0.25));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnHardtanh("missing_hardtanh", "missing", f64, -0.25, 0.25));

    var hardtanh_units_table = try table.withColumnHardtanh("units_hardtanh", "units", i64, 2, 3);
    defer hardtanh_units_table.deinit();
    try std.testing.expectEqual(DeviceDType.i64, try hardtanh_units_table.columnDType("units_hardtanh"));
    const units_hardtanh = try (try hardtanh_units_table.column("units_hardtanh")).i64.toOwnedSlice(gpa);
    defer gpa.free(units_hardtanh);
    try std.testing.expectEqualSlices(i64, &.{ 2, 2, 3 }, units_hardtanh);

    var maximum_ratio_table = try inverse_trig_table.withColumnMaximumScalar("ratio_max", "ratio", f64, 0.25);
    defer maximum_ratio_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try maximum_ratio_table.columnDType("ratio_max"));
    const ratio_max = try (try maximum_ratio_table.column("ratio_max")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_max);
    try std.testing.expectEqualSlices(f64, &.{ 0.25, 0.25, 0.5 }, ratio_max);
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.withColumnMaximumScalar("bad_max", "active", f64, 0.25));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnMaximumScalar("missing_max", "missing", f64, 0.25));

    var minimum_ratio_table = try inverse_trig_table.withColumnMinimumScalar("ratio_min", "ratio", f64, 0.25);
    defer minimum_ratio_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try minimum_ratio_table.columnDType("ratio_min"));
    const ratio_min = try (try minimum_ratio_table.column("ratio_min")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_min);
    try std.testing.expectEqualSlices(f64, &.{ -0.5, 0.0, 0.25 }, ratio_min);
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.withColumnMinimumScalar("bad_min", "active", f64, 0.25));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnMinimumScalar("missing_min", "missing", f64, 0.25));

    var clip_min_ratio_table = try inverse_trig_table.withColumnClipMin("ratio_clip_min", "ratio", f64, -0.25);
    defer clip_min_ratio_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try clip_min_ratio_table.columnDType("ratio_clip_min"));
    const ratio_clip_min = try (try clip_min_ratio_table.column("ratio_clip_min")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_clip_min);
    try std.testing.expectEqualSlices(f64, &.{ -0.25, 0.0, 0.5 }, ratio_clip_min);
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.withColumnClipMin("bad_clip_min", "active", f64, -0.25));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnClipMin("missing_clip_min", "missing", f64, -0.25));

    var clip_max_ratio_table = try inverse_trig_table.withColumnClipMax("ratio_clip_max", "ratio", f64, 0.25);
    defer clip_max_ratio_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try clip_max_ratio_table.columnDType("ratio_clip_max"));
    const ratio_clip_max = try (try clip_max_ratio_table.column("ratio_clip_max")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_clip_max);
    try std.testing.expectEqualSlices(f64, &.{ -0.5, 0.0, 0.25 }, ratio_clip_max);
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.withColumnClipMax("bad_clip_max", "active", f64, 0.25));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnClipMax("missing_clip_max", "missing", f64, 0.25));

    var maximum_units_table = try table.withColumnMaximumWithDeviceScalar("units_max", "units", .{ .f64 = 2.0 });
    defer maximum_units_table.deinit();
    try std.testing.expectEqual(DeviceDType.i64, try maximum_units_table.columnDType("units_max"));
    const units_max = try (try maximum_units_table.column("units_max")).i64.toOwnedSlice(gpa);
    defer gpa.free(units_max);
    try std.testing.expectEqualSlices(i64, &.{ 2, 2, 3 }, units_max);
    try std.testing.expectError(error.TypeUnsupported, table.withColumnClipMinWithDeviceScalar("bad_fractional_clip_min", "units", .{ .f64 = 2.5 }));

    var hardshrink_ratio_table = try inverse_trig_table.withColumnHardshrink("ratio_hardshrink", "ratio", f64, 0.25);
    defer hardshrink_ratio_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try hardshrink_ratio_table.columnDType("ratio_hardshrink"));
    const ratio_hardshrink = try (try hardshrink_ratio_table.column("ratio_hardshrink")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_hardshrink);
    try std.testing.expectEqualSlices(f64, &.{ -0.5, 0.0, 0.5 }, ratio_hardshrink);
    try std.testing.expectError(error.TypeUnsupported, inverse_trig_table.withColumnHardshrink("bad_hardshrink", "units", f64, 0.25));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnHardshrink("missing_hardshrink", "missing", f64, 0.25));

    var softshrink_ratio_table = try inverse_trig_table.withColumnSoftshrink("ratio_softshrink", "ratio", f64, 0.25);
    defer softshrink_ratio_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try softshrink_ratio_table.columnDType("ratio_softshrink"));
    const ratio_softshrink = try (try softshrink_ratio_table.column("ratio_softshrink")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_softshrink);
    try std.testing.expectEqualSlices(f64, &.{ -0.25, 0.0, 0.25 }, ratio_softshrink);
    try std.testing.expectError(error.TypeUnsupported, inverse_trig_table.withColumnSoftshrink("bad_softshrink", "units", f64, 0.25));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnSoftshrink("missing_softshrink", "missing", f64, 0.25));

    var tanhshrink_ratio_table = try inverse_trig_table.withColumnTanhshrink("ratio_tanhshrink", "ratio");
    defer tanhshrink_ratio_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try tanhshrink_ratio_table.columnDType("ratio_tanhshrink"));
    const ratio_tanhshrink = try (try tanhshrink_ratio_table.column("ratio_tanhshrink")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_tanhshrink);
    try std.testing.expectApproxEqAbs(@as(f64, -0.5) - std.math.tanh(@as(f64, -0.5)), ratio_tanhshrink[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ratio_tanhshrink[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5) - std.math.tanh(@as(f64, 0.5)), ratio_tanhshrink[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, inverse_trig_table.withColumnTanhshrink("bad_tanhshrink", "units"));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnTanhshrink("missing_tanhshrink", "missing"));

    var elu_ratio_table = try inverse_trig_table.withColumnElu("ratio_elu", "ratio", f64, 0.5);
    defer elu_ratio_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try elu_ratio_table.columnDType("ratio_elu"));
    const ratio_elu = try (try elu_ratio_table.column("ratio_elu")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_elu);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5) * std.math.expm1(@as(f64, -0.5)), ratio_elu[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ratio_elu[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), ratio_elu[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, inverse_trig_table.withColumnElu("bad_elu", "units", f64, 0.5));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnElu("missing_elu", "missing", f64, 0.5));

    var celu_ratio_table = try inverse_trig_table.withColumnCelu("ratio_celu", "ratio", f64, 2.0);
    defer celu_ratio_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try celu_ratio_table.columnDType("ratio_celu"));
    const ratio_celu = try (try celu_ratio_table.column("ratio_celu")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_celu);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0) * std.math.expm1(@as(f64, -0.25)), ratio_celu[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ratio_celu[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), ratio_celu[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, inverse_trig_table.withColumnCelu("bad_celu", "units", f64, 2.0));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnCelu("missing_celu", "missing", f64, 2.0));

    var softsign_ratio_table = try inverse_trig_table.withColumnSoftsign("ratio_softsign", "ratio");
    defer softsign_ratio_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try softsign_ratio_table.columnDType("ratio_softsign"));
    const ratio_softsign = try (try softsign_ratio_table.column("ratio_softsign")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_softsign);
    try std.testing.expectApproxEqAbs(@as(f64, -0.5) / @as(f64, 1.5), ratio_softsign[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ratio_softsign[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5) / @as(f64, 1.5), ratio_softsign[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, inverse_trig_table.withColumnSoftsign("bad_softsign", "units"));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnSoftsign("missing_softsign", "missing"));

    var hardsigmoid_ratio_table = try inverse_trig_table.withColumnHardsigmoid("ratio_hardsigmoid", "ratio");
    defer hardsigmoid_ratio_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try hardsigmoid_ratio_table.columnDType("ratio_hardsigmoid"));
    const ratio_hardsigmoid = try (try hardsigmoid_ratio_table.column("ratio_hardsigmoid")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_hardsigmoid);
    try std.testing.expectApproxEqAbs((@as(f64, -0.5) + @as(f64, 3.0)) / @as(f64, 6.0), ratio_hardsigmoid[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), ratio_hardsigmoid[1], 1e-12);
    try std.testing.expectApproxEqAbs((@as(f64, 0.5) + @as(f64, 3.0)) / @as(f64, 6.0), ratio_hardsigmoid[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, inverse_trig_table.withColumnHardsigmoid("bad_hardsigmoid", "units"));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnHardsigmoid("missing_hardsigmoid", "missing"));

    var hardswish_ratio_table = try inverse_trig_table.withColumnHardswish("ratio_hardswish", "ratio");
    defer hardswish_ratio_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try hardswish_ratio_table.columnDType("ratio_hardswish"));
    const ratio_hardswish = try (try hardswish_ratio_table.column("ratio_hardswish")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_hardswish);
    try std.testing.expectApproxEqAbs(@as(f64, -0.5) * ((@as(f64, -0.5) + @as(f64, 3.0)) / @as(f64, 6.0)), ratio_hardswish[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ratio_hardswish[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5) * ((@as(f64, 0.5) + @as(f64, 3.0)) / @as(f64, 6.0)), ratio_hardswish[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, inverse_trig_table.withColumnHardswish("bad_hardswish", "units"));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnHardswish("missing_hardswish", "missing"));

    var silu_ratio_table = try inverse_trig_table.withColumnSilu("ratio_silu", "ratio");
    defer silu_ratio_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try silu_ratio_table.columnDType("ratio_silu"));
    const ratio_silu = try (try silu_ratio_table.column("ratio_silu")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_silu);
    try std.testing.expectApproxEqAbs(@as(f64, -0.5) / (@as(f64, 1.0) + std.math.exp(@as(f64, 0.5))), ratio_silu[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ratio_silu[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5) / (@as(f64, 1.0) + std.math.exp(@as(f64, -0.5))), ratio_silu[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, inverse_trig_table.withColumnSilu("bad_silu", "units"));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnSilu("missing_silu", "missing"));

    var swish_ratio_table = try inverse_trig_table.withColumnSwish("ratio_swish", "ratio");
    defer swish_ratio_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try swish_ratio_table.columnDType("ratio_swish"));
    const ratio_swish = try (try swish_ratio_table.column("ratio_swish")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_swish);
    try std.testing.expectApproxEqAbs(ratio_silu[0], ratio_swish[0], 1e-12);
    try std.testing.expectApproxEqAbs(ratio_silu[1], ratio_swish[1], 1e-12);
    try std.testing.expectApproxEqAbs(ratio_silu[2], ratio_swish[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, inverse_trig_table.withColumnSwish("bad_swish", "units"));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnSwish("missing_swish", "missing"));

    var mish_ratio_table = try inverse_trig_table.withColumnMish("ratio_mish", "ratio");
    defer mish_ratio_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try mish_ratio_table.columnDType("ratio_mish"));
    const ratio_mish = try (try mish_ratio_table.column("ratio_mish")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_mish);
    try std.testing.expectApproxEqAbs(@as(f64, -0.5) * std.math.tanh(@max(@as(f64, -0.5), @as(f64, 0.0)) + std.math.log1p(std.math.exp(-@abs(@as(f64, -0.5))))), ratio_mish[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ratio_mish[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5) * std.math.tanh(@max(@as(f64, 0.5), @as(f64, 0.0)) + std.math.log1p(std.math.exp(-@abs(@as(f64, 0.5))))), ratio_mish[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, inverse_trig_table.withColumnMish("bad_mish", "units"));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnMish("missing_mish", "missing"));

    var gelu_ratio_table = try inverse_trig_table.withColumnGelu("ratio_gelu", "ratio");
    defer gelu_ratio_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try gelu_ratio_table.columnDType("ratio_gelu"));
    const ratio_gelu = try (try gelu_ratio_table.column("ratio_gelu")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_gelu);
    try std.testing.expectApproxEqAbs(@as(f64, -0.5) * @as(f64, 0.5) * (@as(f64, 1.0) + std.math.tanh(@sqrt(@as(f64, 2.0) / std.math.pi) * (@as(f64, -0.5) + @as(f64, 0.044715) * @as(f64, -0.125)))), ratio_gelu[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ratio_gelu[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5) * @as(f64, 0.5) * (@as(f64, 1.0) + std.math.tanh(@sqrt(@as(f64, 2.0) / std.math.pi) * (@as(f64, 0.5) + @as(f64, 0.044715) * @as(f64, 0.125)))), ratio_gelu[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, inverse_trig_table.withColumnGelu("bad_gelu", "units"));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnGelu("missing_gelu", "missing"));

    var selu_ratio_table = try inverse_trig_table.withColumnSelu("ratio_selu", "ratio");
    defer selu_ratio_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try selu_ratio_table.columnDType("ratio_selu"));
    const ratio_selu = try (try selu_ratio_table.column("ratio_selu")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_selu);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0507009873554805) * @as(f64, 1.6732632423543772) * std.math.expm1(@as(f64, -0.5)), ratio_selu[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ratio_selu[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0507009873554805) * @as(f64, 0.5), ratio_selu[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, inverse_trig_table.withColumnSelu("bad_selu", "units"));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnSelu("missing_selu", "missing"));

    var exp_cost_table = try table.withColumnExp("cost_exp", "cost");
    defer exp_cost_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try exp_cost_table.columnDType("cost_exp"));
    const cost_exp = try (try exp_cost_table.column("cost_exp")).f64.toOwnedSlice(gpa);
    defer gpa.free(cost_exp);
    try std.testing.expectApproxEqAbs(std.math.exp(@as(f64, 1.0)), cost_exp[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.exp(@as(f64, 1.5)), cost_exp[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.exp(@as(f64, 2.0)), cost_exp[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, table.withColumnExp("bad_exp", "units"));
    try std.testing.expectError(error.ColumnNotFound, table.withColumnExp("missing_exp", "missing"));

    var exp2_cost_table = try table.withColumnExp2("cost_exp2", "cost");
    defer exp2_cost_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try exp2_cost_table.columnDType("cost_exp2"));
    const cost_exp2 = try (try exp2_cost_table.column("cost_exp2")).f64.toOwnedSlice(gpa);
    defer gpa.free(cost_exp2);
    try std.testing.expectApproxEqAbs(std.math.exp2(@as(f64, 1.0)), cost_exp2[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.exp2(@as(f64, 1.5)), cost_exp2[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.exp2(@as(f64, 2.0)), cost_exp2[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, table.withColumnExp2("bad_exp2", "units"));
    try std.testing.expectError(error.ColumnNotFound, table.withColumnExp2("missing_exp2", "missing"));

    var expm1_cost_table = try table.withColumnExpm1("cost_expm1", "cost");
    defer expm1_cost_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try expm1_cost_table.columnDType("cost_expm1"));
    const cost_expm1 = try (try expm1_cost_table.column("cost_expm1")).f64.toOwnedSlice(gpa);
    defer gpa.free(cost_expm1);
    try std.testing.expectApproxEqAbs(std.math.expm1(@as(f64, 1.0)), cost_expm1[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.expm1(@as(f64, 1.5)), cost_expm1[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.expm1(@as(f64, 2.0)), cost_expm1[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, table.withColumnExpm1("bad_expm1", "units"));
    try std.testing.expectError(error.ColumnNotFound, table.withColumnExpm1("missing_expm1", "missing"));

    var sin_cost_table = try table.withColumnSin("cost_sin", "cost");
    defer sin_cost_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try sin_cost_table.columnDType("cost_sin"));
    const cost_sin = try (try sin_cost_table.column("cost_sin")).f64.toOwnedSlice(gpa);
    defer gpa.free(cost_sin);
    try std.testing.expectApproxEqAbs(std.math.sin(@as(f64, 1.0)), cost_sin[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sin(@as(f64, 1.5)), cost_sin[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sin(@as(f64, 2.0)), cost_sin[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, table.withColumnSin("bad_sin", "units"));
    try std.testing.expectError(error.ColumnNotFound, table.withColumnSin("missing_sin", "missing"));

    var cos_cost_table = try table.withColumnCos("cost_cos", "cost");
    defer cos_cost_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try cos_cost_table.columnDType("cost_cos"));
    const cost_cos = try (try cos_cost_table.column("cost_cos")).f64.toOwnedSlice(gpa);
    defer gpa.free(cost_cos);
    try std.testing.expectApproxEqAbs(std.math.cos(@as(f64, 1.0)), cost_cos[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.cos(@as(f64, 1.5)), cost_cos[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.cos(@as(f64, 2.0)), cost_cos[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, table.withColumnCos("bad_cos", "units"));
    try std.testing.expectError(error.ColumnNotFound, table.withColumnCos("missing_cos", "missing"));

    var tan_cost_table = try table.withColumnTan("cost_tan", "cost");
    defer tan_cost_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try tan_cost_table.columnDType("cost_tan"));
    const cost_tan = try (try tan_cost_table.column("cost_tan")).f64.toOwnedSlice(gpa);
    defer gpa.free(cost_tan);
    try std.testing.expectApproxEqAbs(std.math.tan(@as(f64, 1.0)), cost_tan[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.tan(@as(f64, 1.5)), cost_tan[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.tan(@as(f64, 2.0)), cost_tan[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, table.withColumnTan("bad_tan", "units"));
    try std.testing.expectError(error.ColumnNotFound, table.withColumnTan("missing_tan", "missing"));

    var asin_ratio_table = try inverse_trig_table.withColumnAsin("ratio_asin", "ratio");
    defer asin_ratio_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try asin_ratio_table.columnDType("ratio_asin"));
    const ratio_asin = try (try asin_ratio_table.column("ratio_asin")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_asin);
    try std.testing.expectApproxEqAbs(std.math.asin(@as(f64, -0.5)), ratio_asin[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.asin(@as(f64, 0.0)), ratio_asin[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.asin(@as(f64, 0.5)), ratio_asin[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, inverse_trig_table.withColumnAsin("bad_asin", "units"));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnAsin("missing_asin", "missing"));

    var acos_ratio_table = try inverse_trig_table.withColumnAcos("ratio_acos", "ratio");
    defer acos_ratio_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try acos_ratio_table.columnDType("ratio_acos"));
    const ratio_acos = try (try acos_ratio_table.column("ratio_acos")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_acos);
    try std.testing.expectApproxEqAbs(std.math.acos(@as(f64, -0.5)), ratio_acos[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.acos(@as(f64, 0.0)), ratio_acos[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.acos(@as(f64, 0.5)), ratio_acos[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, inverse_trig_table.withColumnAcos("bad_acos", "units"));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnAcos("missing_acos", "missing"));

    var atan_ratio_table = try inverse_trig_table.withColumnAtan("ratio_atan", "ratio");
    defer atan_ratio_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try atan_ratio_table.columnDType("ratio_atan"));
    const ratio_atan = try (try atan_ratio_table.column("ratio_atan")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_atan);
    try std.testing.expectApproxEqAbs(std.math.atan(@as(f64, -0.5)), ratio_atan[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.atan(@as(f64, 0.0)), ratio_atan[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.atan(@as(f64, 0.5)), ratio_atan[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, inverse_trig_table.withColumnAtan("bad_atan", "units"));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnAtan("missing_atan", "missing"));

    var sinh_cost_table = try table.withColumnSinh("cost_sinh", "cost");
    defer sinh_cost_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try sinh_cost_table.columnDType("cost_sinh"));
    const cost_sinh = try (try sinh_cost_table.column("cost_sinh")).f64.toOwnedSlice(gpa);
    defer gpa.free(cost_sinh);
    try std.testing.expectApproxEqAbs(std.math.sinh(@as(f64, 1.0)), cost_sinh[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sinh(@as(f64, 1.5)), cost_sinh[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sinh(@as(f64, 2.0)), cost_sinh[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, table.withColumnSinh("bad_sinh", "units"));
    try std.testing.expectError(error.ColumnNotFound, table.withColumnSinh("missing_sinh", "missing"));

    var cosh_cost_table = try table.withColumnCosh("cost_cosh", "cost");
    defer cosh_cost_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try cosh_cost_table.columnDType("cost_cosh"));
    const cost_cosh = try (try cosh_cost_table.column("cost_cosh")).f64.toOwnedSlice(gpa);
    defer gpa.free(cost_cosh);
    try std.testing.expectApproxEqAbs(std.math.cosh(@as(f64, 1.0)), cost_cosh[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.cosh(@as(f64, 1.5)), cost_cosh[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.cosh(@as(f64, 2.0)), cost_cosh[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, table.withColumnCosh("bad_cosh", "units"));
    try std.testing.expectError(error.ColumnNotFound, table.withColumnCosh("missing_cosh", "missing"));

    var tanh_cost_table = try table.withColumnTanh("cost_tanh", "cost");
    defer tanh_cost_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try tanh_cost_table.columnDType("cost_tanh"));
    const cost_tanh = try (try tanh_cost_table.column("cost_tanh")).f64.toOwnedSlice(gpa);
    defer gpa.free(cost_tanh);
    try std.testing.expectApproxEqAbs(std.math.tanh(@as(f64, 1.0)), cost_tanh[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.tanh(@as(f64, 1.5)), cost_tanh[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.tanh(@as(f64, 2.0)), cost_tanh[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, table.withColumnTanh("bad_tanh", "units"));
    try std.testing.expectError(error.ColumnNotFound, table.withColumnTanh("missing_tanh", "missing"));

    var asinh_ratio_table = try inverse_trig_table.withColumnAsinh("ratio_asinh", "ratio");
    defer asinh_ratio_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try asinh_ratio_table.columnDType("ratio_asinh"));
    const ratio_asinh = try (try asinh_ratio_table.column("ratio_asinh")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_asinh);
    try std.testing.expectApproxEqAbs(std.math.asinh(@as(f64, -0.5)), ratio_asinh[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.asinh(@as(f64, 0.0)), ratio_asinh[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.asinh(@as(f64, 0.5)), ratio_asinh[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, inverse_trig_table.withColumnAsinh("bad_asinh", "units"));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnAsinh("missing_asinh", "missing"));

    var acosh_cost_table = try table.withColumnAcosh("cost_acosh", "cost");
    defer acosh_cost_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try acosh_cost_table.columnDType("cost_acosh"));
    const cost_acosh = try (try acosh_cost_table.column("cost_acosh")).f64.toOwnedSlice(gpa);
    defer gpa.free(cost_acosh);
    try std.testing.expectApproxEqAbs(std.math.acosh(@as(f64, 1.0)), cost_acosh[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.acosh(@as(f64, 1.5)), cost_acosh[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.acosh(@as(f64, 2.0)), cost_acosh[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, table.withColumnAcosh("bad_acosh", "units"));
    try std.testing.expectError(error.ColumnNotFound, table.withColumnAcosh("missing_acosh", "missing"));

    var atanh_ratio_table = try inverse_trig_table.withColumnAtanh("ratio_atanh", "ratio");
    defer atanh_ratio_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try atanh_ratio_table.columnDType("ratio_atanh"));
    const ratio_atanh = try (try atanh_ratio_table.column("ratio_atanh")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio_atanh);
    try std.testing.expectApproxEqAbs(std.math.atanh(@as(f64, -0.5)), ratio_atanh[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.atanh(@as(f64, 0.0)), ratio_atanh[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.atanh(@as(f64, 0.5)), ratio_atanh[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, inverse_trig_table.withColumnAtanh("bad_atanh", "units"));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnAtanh("missing_atanh", "missing"));

    var log_sales_table = try table.withColumnLog("sales_log", "sales");
    defer log_sales_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try log_sales_table.columnDType("sales_log"));
    const sales_log = try (try log_sales_table.column("sales_log")).f64.toOwnedSlice(gpa);
    defer gpa.free(sales_log);
    try std.testing.expectApproxEqAbs(std.math.log(f64, std.math.e, @as(f64, 2.0)), sales_log[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log(f64, std.math.e, @as(f64, 3.0)), sales_log[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log(f64, std.math.e, @as(f64, 5.0)), sales_log[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, table.withColumnLog("bad_log", "units"));
    try std.testing.expectError(error.ColumnNotFound, table.withColumnLog("missing_log", "missing"));

    var log1p_sales_table = try table.withColumnLog1p("sales_log1p", "sales");
    defer log1p_sales_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try log1p_sales_table.columnDType("sales_log1p"));
    const sales_log1p = try (try log1p_sales_table.column("sales_log1p")).f64.toOwnedSlice(gpa);
    defer gpa.free(sales_log1p);
    try std.testing.expectApproxEqAbs(std.math.log1p(@as(f64, 2.0)), sales_log1p[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log1p(@as(f64, 3.0)), sales_log1p[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log1p(@as(f64, 5.0)), sales_log1p[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, table.withColumnLog1p("bad_log1p", "units"));
    try std.testing.expectError(error.ColumnNotFound, table.withColumnLog1p("missing_log1p", "missing"));

    var lgamma_sales_table = try table.withColumnLgamma("sales_lgamma", "sales");
    defer lgamma_sales_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try lgamma_sales_table.columnDType("sales_lgamma"));
    const sales_lgamma = try (try lgamma_sales_table.column("sales_lgamma")).f64.toOwnedSlice(gpa);
    defer gpa.free(sales_lgamma);
    try std.testing.expectApproxEqAbs(std.math.lgamma(f64, @as(f64, 2.0)), sales_lgamma[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.lgamma(f64, @as(f64, 3.0)), sales_lgamma[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.lgamma(f64, @as(f64, 5.0)), sales_lgamma[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, table.withColumnLgamma("bad_lgamma", "units"));
    try std.testing.expectError(error.ColumnNotFound, table.withColumnLgamma("missing_lgamma", "missing"));

    var sinc_cost_table = try table.withColumnSinc("cost_sinc", "cost");
    defer sinc_cost_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try sinc_cost_table.columnDType("cost_sinc"));
    const cost_sinc = try (try sinc_cost_table.column("cost_sinc")).f64.toOwnedSlice(gpa);
    defer gpa.free(cost_sinc);
    try std.testing.expectApproxEqAbs(std.math.sin(std.math.pi * @as(f64, 1.0)) / (std.math.pi * @as(f64, 1.0)), cost_sinc[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sin(std.math.pi * @as(f64, 1.5)) / (std.math.pi * @as(f64, 1.5)), cost_sinc[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sin(std.math.pi * @as(f64, 2.0)) / (std.math.pi * @as(f64, 2.0)), cost_sinc[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, table.withColumnSinc("bad_sinc", "units"));
    try std.testing.expectError(error.ColumnNotFound, table.withColumnSinc("missing_sinc", "missing"));

    var log2_sales_table = try table.withColumnLog2("sales_log2", "sales");
    defer log2_sales_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try log2_sales_table.columnDType("sales_log2"));
    const sales_log2 = try (try log2_sales_table.column("sales_log2")).f64.toOwnedSlice(gpa);
    defer gpa.free(sales_log2);
    try std.testing.expectApproxEqAbs(std.math.log2(@as(f64, 2.0)), sales_log2[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log2(@as(f64, 3.0)), sales_log2[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log2(@as(f64, 5.0)), sales_log2[2], 1e-12);

    var log10_sales_table = try table.withColumnLog10("sales_log10", "sales");
    defer log10_sales_table.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try log10_sales_table.columnDType("sales_log10"));
    const sales_log10 = try (try log10_sales_table.column("sales_log10")).f64.toOwnedSlice(gpa);
    defer gpa.free(sales_log10);
    try std.testing.expectApproxEqAbs(std.math.log10(@as(f64, 2.0)), sales_log10[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log10(@as(f64, 3.0)), sales_log10[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log10(@as(f64, 5.0)), sales_log10[2], 1e-12);
    try std.testing.expectError(error.TypeUnsupported, table.withColumnLog2("bad_log2", "units"));
    try std.testing.expectError(error.TypeUnsupported, table.withColumnLog10("bad_log10", "units"));
    try std.testing.expectError(error.ColumnNotFound, table.withColumnLog2("missing_log2", "missing"));

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

    var where_indices = try mask_table.whereIndicesColumn("units_gt_one", "row_index");
    defer where_indices.deinit();
    try std.testing.expectEqual(@as(usize, 1), where_indices.width());
    const where_index_values = try (try where_indices.column("row_index")).usize.toOwnedSlice(gpa);
    defer gpa.free(where_index_values);
    try std.testing.expectEqualSlices(usize, &.{2}, where_index_values);
    try std.testing.expectError(error.TypeMismatch, mask_table.whereIndicesColumn("sales", "bad_rows"));

    var named_mask_dropped = try mask_table.dropRowsByColumnMask("units_gt_one");
    defer named_mask_dropped.deinit();
    try std.testing.expectEqual(@as(usize, 2), named_mask_dropped.height());
    const named_mask_dropped_sales = try (try named_mask_dropped.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(named_mask_dropped_sales);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 3.0 }, named_mask_dropped_sales);
    try std.testing.expectError(error.TypeMismatch, mask_table.dropRowsByColumnMask("sales"));
    try std.testing.expectError(error.TypeMismatch, mask_table.filterColumn("sales"));
}
