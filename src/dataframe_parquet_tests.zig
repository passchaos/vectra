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

test "device dataframe round-trips through boltha parquet" {
    const gpa = std.testing.allocator;

    var id = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 2, 3 }, .cpu);
    defer id.deinit();
    var sales = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 2.0, 3.0, 5.0 }, &.{ true, false, true }, .cpu);
    defer sales.deinit();
    var active = try DeviceColumn.fromSlice(bool, gpa, &.{ true, false, true }, .cpu);
    defer active.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "id", .data = id },
        .{ .name = "sales", .data = sales },
        .{ .name = "active", .data = active },
    });
    defer table.deinit();

    const bytes = try table.toParquetBytes(gpa);
    defer gpa.free(bytes);
    try std.testing.expect(bytes.len > 0);

    var restored = try DeviceDataFrame.fromParquetBytes(gpa, bytes, .cpu);
    defer restored.deinit();
    try std.testing.expectEqual(table.height(), restored.height());
    try std.testing.expectEqual(table.width(), restored.width());
    try std.testing.expectEqual(DeviceDType.i32, try restored.columnDType("id"));
    try std.testing.expectEqual(DeviceDType.f64, try restored.columnDType("sales"));
    try std.testing.expectEqual(DeviceDType.bool, try restored.columnDType("active"));

    const ids = try (try restored.column("id")).i32.toOwnedSlice(gpa);
    defer gpa.free(ids);
    const sales_values = try (try restored.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(sales_values);
    const sales_validity = try (try restored.column("sales")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(sales_validity);
    const active_values = try (try restored.column("active")).bool.toOwnedSlice(gpa);
    defer gpa.free(active_values);

    try std.testing.expectEqualSlices(i32, &.{ 1, 2, 3 }, ids);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 0.0, 5.0 }, sales_values);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, sales_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, active_values);
}

test "device dataframe reads boltha parquet with range pruning" {
    const gpa = std.testing.allocator;

    var id_field = try boltha.arrow.Field.init(gpa, "id", .{ .int = .{ .bit_width = 32, .signed = true } }, false);
    defer id_field.deinit(gpa);
    var sales_field = try boltha.arrow.Field.init(gpa, "sales", .{ .floating_point = .double }, false);
    defer sales_field.deinit(gpa);
    const schema = try boltha.arrow.Schema.init(gpa, &.{ id_field, sales_field });

    const batches = try gpa.alloc(boltha.arrow.RecordBatch, 2);
    const schema0 = try boltha.arrow.Schema.init(gpa, &.{ id_field, sales_field });
    const cols0 = try gpa.alloc(boltha.arrow.AnyArray, 2);
    cols0[0] = .{ .int32 = try boltha.arrow.PrimitiveArray(i32).fromSlice(gpa, &.{ 1, 2 }) };
    cols0[1] = .{ .float64 = try boltha.arrow.PrimitiveArray(f64).fromSlice(gpa, &.{ 10.0, 20.0 }) };
    batches[0] = try boltha.arrow.RecordBatch.initOwned(schema0, cols0);
    const schema1 = try boltha.arrow.Schema.init(gpa, &.{ id_field, sales_field });
    const cols1 = try gpa.alloc(boltha.arrow.AnyArray, 2);
    cols1[0] = .{ .int32 = try boltha.arrow.PrimitiveArray(i32).fromSlice(gpa, &.{ 100, 101 }) };
    cols1[1] = .{ .float64 = try boltha.arrow.PrimitiveArray(f64).fromSlice(gpa, &.{ 1000.0, 1010.0 }) };
    batches[1] = try boltha.arrow.RecordBatch.initOwned(schema1, cols1);

    var arrow_table = try boltha.arrow.Table.initOwned(schema, batches);
    defer arrow_table.deinit(gpa);
    var parquet_bytes: std.ArrayList(u8) = .empty;
    defer parquet_bytes.deinit(gpa);
    try boltha.parquet.writeTable(gpa, &parquet_bytes, arrow_table);

    var pruned = try DeviceDataFrame.fromParquetBytesPruned(
        gpa,
        parquet_bytes.items,
        "id",
        .{ .i32 = .{ .min = 100, .max = 101 } },
        .cpu,
    );
    defer pruned.deinit();
    try std.testing.expectEqual(@as(usize, 2), pruned.height());
    const ids = try (try pruned.column("id")).i32.toOwnedSlice(gpa);
    defer gpa.free(ids);
    const sales_values = try (try pruned.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(sales_values);
    try std.testing.expectEqualSlices(i32, &.{ 100, 101 }, ids);
    try std.testing.expectEqualSlices(f64, &.{ 1000.0, 1010.0 }, sales_values);

    var empty = try DeviceDataFrame.fromParquetBytesPruned(
        gpa,
        parquet_bytes.items,
        "id",
        .{ .i32 = .{ .min = 10_000 } },
        .cpu,
    );
    defer empty.deinit();
    try std.testing.expectEqual(@as(usize, 0), empty.height());
    try std.testing.expectEqual(@as(usize, 2), empty.width());
}

test "device parquet scan pushes range predicate and projection into collect" {
    const gpa = std.testing.allocator;

    var id = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 2, 3 }, .cpu);
    defer id.deinit();
    var sales = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 3.0, 5.0 }, .cpu);
    defer sales.deinit();
    var active = try DeviceColumn.fromSlice(bool, gpa, &.{ true, false, true }, .cpu);
    defer active.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "id", .data = id },
        .{ .name = "sales", .data = sales },
        .{ .name = "active", .data = active },
    });
    defer table.deinit();

    const bytes = try table.toParquetBytes(gpa);
    defer gpa.free(bytes);

    var scan = try DeviceParquetScan.init(gpa, bytes, .cpu);
    defer scan.deinit();
    try scan.whereRange("id", .{ .i32 = .{ .min = 2, .max = 3 } });
    try scan.select(&.{ "id", "sales" });

    const explain = try scan.explain(gpa);
    defer gpa.free(explain);
    try std.testing.expect(std.mem.indexOf(u8, explain, "range=id") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "projection=[id,sales]") != null);

    var result = try scan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 3), result.height());
    try std.testing.expectEqual(@as(usize, 2), result.width());
    try std.testing.expectEqual(DeviceDType.i32, try result.columnDType("id"));
    try std.testing.expectEqual(DeviceDType.f64, try result.columnDType("sales"));
    try std.testing.expectEqual(@as(?usize, null), result.columnIndex("active"));
}

test "device lazy frame pushes scalar filters and projection into parquet scan source" {
    const gpa = std.testing.allocator;

    var id = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 2, 3 }, .cpu);
    defer id.deinit();
    var sales = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 3.0, 5.0 }, .cpu);
    defer sales.deinit();
    var active = try DeviceColumn.fromSlice(bool, gpa, &.{ true, false, true }, .cpu);
    defer active.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "id", .data = id },
        .{ .name = "sales", .data = sales },
        .{ .name = "active", .data = active },
    });
    defer table.deinit();

    const bytes = try table.toParquetBytes(gpa);
    defer gpa.free(bytes);

    var lazy_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer lazy_scan.deinit();
    try lazy_scan.withColumnScalar("sales_x2", "sales", f64, 2.0, .mul);
    try lazy_scan.filterColumnScalar("sales", f64, 2.5, .gt);
    try lazy_scan.select(&.{ "sales_x2", "id" });

    const explain = try lazy_scan.explain(gpa);
    defer gpa.free(explain);
    try std.testing.expect(std.mem.indexOf(u8, explain, "source=parquet_scan") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_scalar(sales_x2") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "scan_pushdown: range=sales, projection=[sales,id]") != null);

    var result = try lazy_scan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 2), result.height());
    try std.testing.expectEqual(@as(usize, 2), result.width());
    try std.testing.expectEqual(@as(?usize, null), result.columnIndex("active"));
    try std.testing.expectEqual(@as(?usize, null), result.columnIndex("sales"));
    const result_sales_x2 = try (try result.column("sales_x2")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_x2);
    try std.testing.expectEqualSlices(f64, &.{ 6.0, 10.0 }, result_sales_x2);
}

test "device lazy frame pushes null predicate dependencies into parquet scan source" {
    const gpa = std.testing.allocator;

    var id = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 2, 3 }, .cpu);
    defer id.deinit();
    var sales = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 2.0, 3.0, 5.0 }, &.{ true, false, true }, .cpu);
    defer sales.deinit();
    var active = try DeviceColumn.fromSlice(bool, gpa, &.{ true, false, true }, .cpu);
    defer active.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "id", .data = id },
        .{ .name = "sales", .data = sales },
        .{ .name = "active", .data = active },
    });
    defer table.deinit();

    const bytes = try table.toParquetBytes(gpa);
    defer gpa.free(bytes);

    var lazy_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer lazy_scan.deinit();
    try lazy_scan.isNullColumn("sales", "sales_is_null");
    try lazy_scan.select(&.{"sales_is_null"});

    const explain = try lazy_scan.explain(gpa);
    defer gpa.free(explain);
    try std.testing.expect(std.mem.indexOf(u8, explain, "scan_pushdown: projection=[sales]") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "is_null_column(sales->sales_is_null)") != null);

    var result = try lazy_scan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 1), result.width());
    try std.testing.expectEqual(@as(?usize, 0), result.columnIndex("sales_is_null"));
    try std.testing.expectEqual(@as(?usize, null), result.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, null), result.columnIndex("id"));
    const is_null = try (try result.column("sales_is_null")).bool.toOwnedSlice(gpa);
    defer gpa.free(is_null);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false }, is_null);

    var row_count_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer row_count_scan.deinit();
    try row_count_scan.withRowNullCount(&.{ "sales", "active" }, "row_nulls");
    try row_count_scan.select(&.{"row_nulls"});

    const row_count_explain = try row_count_scan.explain(gpa);
    defer gpa.free(row_count_explain);
    try std.testing.expect(std.mem.indexOf(u8, row_count_explain, "scan_pushdown: projection=[sales,active]") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_count_explain, "row_null_count([sales,active]->row_nulls)") != null);

    var row_count_result = try row_count_scan.collect();
    defer row_count_result.deinit();
    const row_nulls = try (try row_count_result.column("row_nulls")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_nulls);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0 }, row_nulls);

    var all_count_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer all_count_scan.deinit();
    try all_count_scan.withRowValidCount(&.{}, "row_valids_all");
    try all_count_scan.select(&.{"row_valids_all"});

    const all_count_explain = try all_count_scan.explain(gpa);
    defer gpa.free(all_count_explain);
    try std.testing.expect(std.mem.indexOf(u8, all_count_explain, "scan_pushdown: none") != null);
    try std.testing.expect(std.mem.indexOf(u8, all_count_explain, "row_valid_count([]->row_valids_all)") != null);

    var all_count_result = try all_count_scan.collect();
    defer all_count_result.deinit();
    const row_valids_all = try (try all_count_result.column("row_valids_all")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_valids_all);
    try std.testing.expectEqualSlices(i64, &.{ 3, 2, 3 }, row_valids_all);

    var finite_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer finite_scan.deinit();
    try finite_scan.isFiniteColumn("sales", "sales_is_finite");
    try finite_scan.select(&.{"sales_is_finite"});

    const finite_explain = try finite_scan.explain(gpa);
    defer gpa.free(finite_explain);
    try std.testing.expect(std.mem.indexOf(u8, finite_explain, "scan_pushdown: projection=[sales]") != null);
    try std.testing.expect(std.mem.indexOf(u8, finite_explain, "is_finite_column(sales->sales_is_finite)") != null);

    var finite_result = try finite_scan.collect();
    defer finite_result.deinit();
    const sales_is_finite = try (try finite_result.column("sales_is_finite")).bool.toOwnedSlice(gpa);
    defer gpa.free(sales_is_finite);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, sales_is_finite);

    var inf_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer inf_scan.deinit();
    try inf_scan.isInfColumn("sales", "sales_is_inf");
    try inf_scan.select(&.{"sales_is_inf"});

    const inf_explain = try inf_scan.explain(gpa);
    defer gpa.free(inf_explain);
    try std.testing.expect(std.mem.indexOf(u8, inf_explain, "scan_pushdown: projection=[sales]") != null);
    try std.testing.expect(std.mem.indexOf(u8, inf_explain, "is_inf_column(sales->sales_is_inf)") != null);

    var fill_nan_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer fill_nan_scan.deinit();
    try fill_nan_scan.fillNaNColumn("sales", f64, -1.0);
    try fill_nan_scan.select(&.{"sales"});

    const fill_nan_explain = try fill_nan_scan.explain(gpa);
    defer gpa.free(fill_nan_explain);
    try std.testing.expect(std.mem.indexOf(u8, fill_nan_explain, "scan_pushdown: projection=[sales]") != null);
    try std.testing.expect(std.mem.indexOf(u8, fill_nan_explain, "fill_nan_column(sales=scalar:f64)") != null);

    var fill_inf_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer fill_inf_scan.deinit();
    try fill_inf_scan.fillInfColumn("sales", f64, -9.0);
    try fill_inf_scan.select(&.{"sales"});

    const fill_inf_explain = try fill_inf_scan.explain(gpa);
    defer gpa.free(fill_inf_explain);
    try std.testing.expect(std.mem.indexOf(u8, fill_inf_explain, "scan_pushdown: projection=[sales]") != null);
    try std.testing.expect(std.mem.indexOf(u8, fill_inf_explain, "fill_inf_column(sales=scalar:f64)") != null);

    var fill_non_finite_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer fill_non_finite_scan.deinit();
    try fill_non_finite_scan.fillNonFiniteColumn("sales", f64, -5.0);
    try fill_non_finite_scan.select(&.{"sales"});

    const fill_non_finite_explain = try fill_non_finite_scan.explain(gpa);
    defer gpa.free(fill_non_finite_explain);
    try std.testing.expect(std.mem.indexOf(u8, fill_non_finite_explain, "scan_pushdown: projection=[sales]") != null);
    try std.testing.expect(std.mem.indexOf(u8, fill_non_finite_explain, "fill_non_finite_column(sales=scalar:f64)") != null);

    var drop_nan_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer drop_nan_scan.deinit();
    try drop_nan_scan.dropNaNsColumn("sales");
    try drop_nan_scan.select(&.{"id"});

    const drop_nan_explain = try drop_nan_scan.explain(gpa);
    defer gpa.free(drop_nan_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_nan_explain, "scan_pushdown: projection=[sales,id]") != null);
    try std.testing.expect(std.mem.indexOf(u8, drop_nan_explain, "drop_nans[sales]") != null);

    var filter_nan_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer filter_nan_scan.deinit();
    try filter_nan_scan.filterNaNsColumn("sales");

    const filter_nan_explain = try filter_nan_scan.explain(gpa);
    defer gpa.free(filter_nan_explain);
    try std.testing.expect(std.mem.indexOf(u8, filter_nan_explain, "scan_pushdown: none") != null);
    try std.testing.expect(std.mem.indexOf(u8, filter_nan_explain, "filter_nans_column(sales)") != null);

    var drop_inf_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer drop_inf_scan.deinit();
    try drop_inf_scan.dropInfsColumn("sales");
    try drop_inf_scan.select(&.{"id"});

    const drop_inf_explain = try drop_inf_scan.explain(gpa);
    defer gpa.free(drop_inf_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_inf_explain, "scan_pushdown: projection=[sales,id]") != null);
    try std.testing.expect(std.mem.indexOf(u8, drop_inf_explain, "drop_infs[sales]") != null);

    var filter_inf_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer filter_inf_scan.deinit();
    try filter_inf_scan.filterInfsColumn("sales");

    const filter_inf_explain = try filter_inf_scan.explain(gpa);
    defer gpa.free(filter_inf_explain);
    try std.testing.expect(std.mem.indexOf(u8, filter_inf_explain, "scan_pushdown: none") != null);
    try std.testing.expect(std.mem.indexOf(u8, filter_inf_explain, "filter_infs_column(sales)") != null);

    var drop_non_finite_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer drop_non_finite_scan.deinit();
    try drop_non_finite_scan.dropNonFinitesColumn("sales");
    try drop_non_finite_scan.select(&.{"id"});

    const drop_non_finite_explain = try drop_non_finite_scan.explain(gpa);
    defer gpa.free(drop_non_finite_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_non_finite_explain, "scan_pushdown: projection=[sales,id]") != null);
    try std.testing.expect(std.mem.indexOf(u8, drop_non_finite_explain, "drop_non_finites[sales]") != null);

    var filter_non_finite_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer filter_non_finite_scan.deinit();
    try filter_non_finite_scan.filterNonFinitesColumn("sales");

    const filter_non_finite_explain = try filter_non_finite_scan.explain(gpa);
    defer gpa.free(filter_non_finite_explain);
    try std.testing.expect(std.mem.indexOf(u8, filter_non_finite_explain, "scan_pushdown: none") != null);
    try std.testing.expect(std.mem.indexOf(u8, filter_non_finite_explain, "filter_non_finites_column(sales)") != null);

    var row_nan_count_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer row_nan_count_scan.deinit();
    try row_nan_count_scan.withRowNaNCount(&.{ "sales", "active" }, "row_nan_count");
    try row_nan_count_scan.select(&.{"row_nan_count"});

    const row_nan_count_explain = try row_nan_count_scan.explain(gpa);
    defer gpa.free(row_nan_count_explain);
    try std.testing.expect(std.mem.indexOf(u8, row_nan_count_explain, "scan_pushdown: projection=[sales,active]") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_nan_count_explain, "row_nan_count([sales,active]->row_nan_count)") != null);

    var row_inf_count_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer row_inf_count_scan.deinit();
    try row_inf_count_scan.withRowInfCount(&.{}, "row_inf_count");
    try row_inf_count_scan.select(&.{"row_inf_count"});

    const row_inf_count_explain = try row_inf_count_scan.explain(gpa);
    defer gpa.free(row_inf_count_explain);
    try std.testing.expect(std.mem.indexOf(u8, row_inf_count_explain, "scan_pushdown: none") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_inf_count_explain, "row_inf_count([]->row_inf_count)") != null);

    var row_finite_count_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer row_finite_count_scan.deinit();
    try row_finite_count_scan.withRowFiniteCount(&.{ "sales", "active" }, "row_finite_count");
    try row_finite_count_scan.select(&.{"row_finite_count"});

    const row_finite_count_explain = try row_finite_count_scan.explain(gpa);
    defer gpa.free(row_finite_count_explain);
    try std.testing.expect(std.mem.indexOf(u8, row_finite_count_explain, "scan_pushdown: projection=[sales,active]") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_finite_count_explain, "row_finite_count([sales,active]->row_finite_count)") != null);

    var row_non_finite_count_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer row_non_finite_count_scan.deinit();
    try row_non_finite_count_scan.withRowNonFiniteCount(&.{}, "row_non_finite_count");
    try row_non_finite_count_scan.select(&.{"row_non_finite_count"});

    const row_non_finite_count_explain = try row_non_finite_count_scan.explain(gpa);
    defer gpa.free(row_non_finite_count_explain);
    try std.testing.expect(std.mem.indexOf(u8, row_non_finite_count_explain, "scan_pushdown: none") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_non_finite_count_explain, "row_non_finite_count([]->row_non_finite_count)") != null);
}

test "device lazy frame pushes coalesce dependencies into parquet scan source" {
    const gpa = std.testing.allocator;

    var id = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 2, 3 }, .cpu);
    defer id.deinit();
    var sales = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 2.0, 3.0, 5.0 }, &.{ true, false, true }, .cpu);
    defer sales.deinit();
    var fallback = try DeviceColumn.fromSlice(f64, gpa, &.{ 8.0, 9.0, 10.0 }, .cpu);
    defer fallback.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "id", .data = id },
        .{ .name = "sales", .data = sales },
        .{ .name = "fallback", .data = fallback },
    });
    defer table.deinit();

    const bytes = try table.toParquetBytes(gpa);
    defer gpa.free(bytes);

    var lazy_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer lazy_scan.deinit();
    try lazy_scan.coalesceColumns("sales", "fallback", "sales_filled");
    try lazy_scan.select(&.{"sales_filled"});

    const explain = try lazy_scan.explain(gpa);
    defer gpa.free(explain);
    try std.testing.expect(std.mem.indexOf(u8, explain, "scan_pushdown: projection=[sales,fallback]") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "coalesce_columns(sales,fallback->sales_filled)") != null);

    var result = try lazy_scan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 1), result.width());
    const filled = try (try result.column("sales_filled")).f64.toOwnedSlice(gpa);
    defer gpa.free(filled);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 9.0, 5.0 }, filled);
}

test "device lazy frame keeps schema-derived and schema-rewrite ops out of parquet projection pushdown" {
    const gpa = std.testing.allocator;

    var id = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 2, 3 }, .cpu);
    defer id.deinit();
    var sales = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 3.0, 5.0 }, .cpu);
    defer sales.deinit();
    var active = try DeviceColumn.fromSlice(bool, gpa, &.{ true, false, true }, .cpu);
    defer active.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "id", .data = id },
        .{ .name = "sales", .data = sales },
        .{ .name = "active", .data = active },
    });
    defer table.deinit();

    const bytes = try table.toParquetBytes(gpa);
    defer gpa.free(bytes);

    var lazy_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer lazy_scan.deinit();
    try lazy_scan.selectByNameSuffix("es");

    const explain = try lazy_scan.explain(gpa);
    defer gpa.free(explain);
    try std.testing.expect(std.mem.indexOf(u8, explain, "scan_pushdown: none") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "select_name_suffix(es)") != null);

    var result = try lazy_scan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 3), result.height());
    try std.testing.expectEqual(@as(usize, 1), result.width());
    try std.testing.expectEqual(@as(?usize, 0), result.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, null), result.columnIndex("id"));
    try std.testing.expectEqual(@as(?usize, null), result.columnIndex("active"));
    const result_sales = try (try result.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 3.0, 5.0 }, result_sales);

    var lazy_drop_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer lazy_drop_scan.deinit();
    try lazy_drop_scan.dropByNameSuffix("ive");

    const drop_explain = try lazy_drop_scan.explain(gpa);
    defer gpa.free(drop_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_explain, "scan_pushdown: none") != null);
    try std.testing.expect(std.mem.indexOf(u8, drop_explain, "drop_name_suffix(ive)") != null);

    var dropped = try lazy_drop_scan.collect();
    defer dropped.deinit();
    try std.testing.expectEqual(@as(usize, 2), dropped.width());
    try std.testing.expectEqual(@as(?usize, 0), dropped.columnIndex("id"));
    try std.testing.expectEqual(@as(?usize, 1), dropped.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, null), dropped.columnIndex("active"));

    var lazy_drop_dtype_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer lazy_drop_dtype_scan.deinit();
    try lazy_drop_dtype_scan.dropFloat();

    const drop_dtype_explain = try lazy_drop_dtype_scan.explain(gpa);
    defer gpa.free(drop_dtype_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_dtype_explain, "scan_pushdown: none") != null);
    try std.testing.expect(std.mem.indexOf(u8, drop_dtype_explain, "drop_dtype_class(float)") != null);

    var dtype_dropped = try lazy_drop_dtype_scan.collect();
    defer dtype_dropped.deinit();
    try std.testing.expectEqual(@as(usize, 2), dtype_dropped.width());
    try std.testing.expectEqual(@as(?usize, 0), dtype_dropped.columnIndex("id"));
    try std.testing.expectEqual(@as(?usize, 1), dtype_dropped.columnIndex("active"));
    try std.testing.expectEqual(@as(?usize, null), dtype_dropped.columnIndex("sales"));

    var lazy_nullability_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer lazy_nullability_scan.deinit();
    try lazy_nullability_scan.selectColumnsWithNulls();

    const nullability_explain = try lazy_nullability_scan.explain(gpa);
    defer gpa.free(nullability_explain);
    try std.testing.expect(std.mem.indexOf(u8, nullability_explain, "scan_pushdown: none") != null);
    try std.testing.expect(std.mem.indexOf(u8, nullability_explain, "select_columns_with_nulls") != null);

    var lazy_reverse_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer lazy_reverse_scan.deinit();
    try lazy_reverse_scan.reverseColumns();

    const reverse_explain = try lazy_reverse_scan.explain(gpa);
    defer gpa.free(reverse_explain);
    try std.testing.expect(std.mem.indexOf(u8, reverse_explain, "scan_pushdown: none") != null);
    try std.testing.expect(std.mem.indexOf(u8, reverse_explain, "reverse_columns") != null);

    var lazy_sort_columns_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer lazy_sort_columns_scan.deinit();
    try lazy_sort_columns_scan.sortColumnsByName(false);

    const sort_columns_explain = try lazy_sort_columns_scan.explain(gpa);
    defer gpa.free(sort_columns_explain);
    try std.testing.expect(std.mem.indexOf(u8, sort_columns_explain, "scan_pushdown: none") != null);
    try std.testing.expect(std.mem.indexOf(u8, sort_columns_explain, "sort_columns_by_name(desc=false)") != null);

    var lazy_move_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer lazy_move_scan.deinit();
    try lazy_move_scan.moveColumn("active", 0);

    const move_explain = try lazy_move_scan.explain(gpa);
    defer gpa.free(move_explain);
    try std.testing.expect(std.mem.indexOf(u8, move_explain, "scan_pushdown: none") != null);
    try std.testing.expect(std.mem.indexOf(u8, move_explain, "move_column(active -> index=0)") != null);

    var moved = try lazy_move_scan.collect();
    defer moved.deinit();
    try std.testing.expectEqual(@as(usize, 3), moved.width());
    try std.testing.expectEqual(@as(?usize, 0), moved.columnIndex("active"));
    try std.testing.expectEqual(@as(?usize, 1), moved.columnIndex("id"));
    try std.testing.expectEqual(@as(?usize, 2), moved.columnIndex("sales"));

    var lazy_placed_literal_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer lazy_placed_literal_scan.deinit();
    try lazy_placed_literal_scan.withColumnLiteralAt("segment", i32, 7, 0);

    const placed_literal_explain = try lazy_placed_literal_scan.explain(gpa);
    defer gpa.free(placed_literal_explain);
    try std.testing.expect(std.mem.indexOf(u8, placed_literal_explain, "scan_pushdown: none") != null);
    try std.testing.expect(std.mem.indexOf(u8, placed_literal_explain, "with_column_literal_at(segment=scalar:i32, index=0)") != null);

    var placed_literal = try lazy_placed_literal_scan.collect();
    defer placed_literal.deinit();
    try std.testing.expectEqual(@as(usize, 4), placed_literal.width());
    try std.testing.expectEqual(@as(?usize, 0), placed_literal.columnIndex("segment"));
    try std.testing.expectEqual(@as(?usize, 1), placed_literal.columnIndex("id"));
    try std.testing.expectEqual(@as(?usize, 2), placed_literal.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 3), placed_literal.columnIndex("active"));

    var lazy_copy_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer lazy_copy_scan.deinit();
    try lazy_copy_scan.copyColumnAt("sales", "sales_copy", 0);

    const copy_explain = try lazy_copy_scan.explain(gpa);
    defer gpa.free(copy_explain);
    try std.testing.expect(std.mem.indexOf(u8, copy_explain, "scan_pushdown: none") != null);
    try std.testing.expect(std.mem.indexOf(u8, copy_explain, "copy_column_at(sales->sales_copy, index=0)") != null);

    var copied = try lazy_copy_scan.collect();
    defer copied.deinit();
    try std.testing.expectEqual(@as(usize, 4), copied.width());
    try std.testing.expectEqual(@as(?usize, 0), copied.columnIndex("sales_copy"));
    try std.testing.expectEqual(@as(?usize, 1), copied.columnIndex("id"));
    try std.testing.expectEqual(@as(?usize, 2), copied.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 3), copied.columnIndex("active"));
    const copied_sales = try (try copied.column("sales_copy")).f64.toOwnedSlice(gpa);
    defer gpa.free(copied_sales);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 3.0, 5.0 }, copied_sales);

    var lazy_rename_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer lazy_rename_scan.deinit();
    try lazy_rename_scan.renameColumns(&.{ "id", "sales" }, &.{ "row_id", "revenue" });

    const rename_explain = try lazy_rename_scan.explain(gpa);
    defer gpa.free(rename_explain);
    try std.testing.expect(std.mem.indexOf(u8, rename_explain, "scan_pushdown: none") != null);
    try std.testing.expect(std.mem.indexOf(u8, rename_explain, "rename_columns[id->row_id,sales->revenue]") != null);

    var renamed = try lazy_rename_scan.collect();
    defer renamed.deinit();
    try std.testing.expectEqual(@as(?usize, 0), renamed.columnIndex("row_id"));
    try std.testing.expectEqual(@as(?usize, 1), renamed.columnIndex("revenue"));
    try std.testing.expectEqual(@as(?usize, 2), renamed.columnIndex("active"));
}

test "device lazy frame derives row index after parquet projection" {
    const gpa = std.testing.allocator;

    var id = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 2, 3 }, .cpu);
    defer id.deinit();
    var sales = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 3.0, 5.0 }, .cpu);
    defer sales.deinit();
    var active = try DeviceColumn.fromSlice(bool, gpa, &.{ true, false, true }, .cpu);
    defer active.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "id", .data = id },
        .{ .name = "sales", .data = sales },
        .{ .name = "active", .data = active },
    });
    defer table.deinit();

    const bytes = try table.toParquetBytes(gpa);
    defer gpa.free(bytes);

    var lazy_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer lazy_scan.deinit();
    try lazy_scan.withRowIndex("row_nr", 5);
    try lazy_scan.select(&.{ "row_nr", "id" });

    const explain = try lazy_scan.explain(gpa);
    defer gpa.free(explain);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_row_index(row_nr, offset=5)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "scan_pushdown: projection=[id]") != null);

    var result = try lazy_scan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 3), result.height());
    try std.testing.expectEqual(@as(usize, 2), result.width());
    try std.testing.expectEqual(@as(?usize, null), result.columnIndex("active"));
    try std.testing.expectEqual(@as(?usize, null), result.columnIndex("sales"));
    const row_nr = try (try result.column("row_nr")).usize.toOwnedSlice(gpa);
    defer gpa.free(row_nr);
    const ids = try (try result.column("id")).i32.toOwnedSlice(gpa);
    defer gpa.free(ids);
    try std.testing.expectEqualSlices(usize, &.{ 5, 6, 7 }, row_nr);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2, 3 }, ids);
}
