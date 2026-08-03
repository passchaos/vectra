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

test "device dataframe round-trips Vectra extension dtypes through boltha parquet" {
    const gpa = std.testing.allocator;
    const BF16 = vectra.BFloat16;
    const C64 = vectra.Complex64;
    const C128 = vectra.Complex128;

    var quality = try DeviceColumn.fromSliceWithValidity(
        BF16,
        gpa,
        &.{ BF16.fromF32(1.5), BF16.fromF32(-2.25), BF16.fromF32(4.0) },
        &.{ true, false, true },
        .cpu,
    );
    defer quality.deinit();
    var z32 = try DeviceColumn.fromSliceWithValidity(
        C64,
        gpa,
        &.{ C64.init(1.0, -2.0), C64.init(9.0, 9.0), C64.init(3.5, 4.5) },
        &.{ true, false, true },
        .cpu,
    );
    defer z32.deinit();
    var z64 = try DeviceColumn.fromSlice(C128, gpa, &.{ C128.init(1.25, -0.5), C128.init(-2.0, 8.0), C128.init(0.0, 3.0) }, .cpu);
    defer z64.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "quality", .data = quality },
        .{ .name = "z32", .data = z32 },
        .{ .name = "z64", .data = z64 },
    });
    defer table.deinit();

    const bytes = try table.toParquetBytes(gpa);
    defer gpa.free(bytes);

    var restored = try DeviceDataFrame.fromParquetBytes(gpa, bytes, .cpu);
    defer restored.deinit();
    try std.testing.expectEqual(DeviceDType.bf16, try restored.columnDType("quality"));
    try std.testing.expectEqual(DeviceDType.c64, try restored.columnDType("z32"));
    try std.testing.expectEqual(DeviceDType.c128, try restored.columnDType("z64"));

    const restored_quality = try (try restored.column("quality")).bf16.toOwnedSlice(gpa);
    defer gpa.free(restored_quality);
    const restored_quality_validity = try (try restored.column("quality")).bf16.validity.?.toOwnedSlice(gpa);
    defer gpa.free(restored_quality_validity);
    try std.testing.expectEqual(BF16.fromF32(1.5).bits, restored_quality[0].bits);
    try std.testing.expectEqual(BF16.fromF32(0.0).bits, restored_quality[1].bits);
    try std.testing.expectEqual(BF16.fromF32(4.0).bits, restored_quality[2].bits);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, restored_quality_validity);

    const restored_z32 = try (try restored.column("z32")).c64.toOwnedSlice(gpa);
    defer gpa.free(restored_z32);
    const restored_z32_validity = try (try restored.column("z32")).c64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(restored_z32_validity);
    try std.testing.expectEqual(@as(f32, 1.0), restored_z32[0].re);
    try std.testing.expectEqual(@as(f32, -2.0), restored_z32[0].im);
    try std.testing.expectEqual(@as(f32, 0.0), restored_z32[1].re);
    try std.testing.expectEqual(@as(f32, 0.0), restored_z32[1].im);
    try std.testing.expectEqual(@as(f32, 3.5), restored_z32[2].re);
    try std.testing.expectEqual(@as(f32, 4.5), restored_z32[2].im);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, restored_z32_validity);

    const restored_z64 = try (try restored.column("z64")).c128.toOwnedSlice(gpa);
    defer gpa.free(restored_z64);
    try std.testing.expectEqual(@as(f64, 1.25), restored_z64[0].re);
    try std.testing.expectEqual(@as(f64, -0.5), restored_z64[0].im);
    try std.testing.expectEqual(@as(f64, -2.0), restored_z64[1].re);
    try std.testing.expectEqual(@as(f64, 8.0), restored_z64[1].im);
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
    var sales = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 2.0, 0.0, 5.0 }, &.{ true, false, true }, .cpu);
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
    try std.testing.expect(std.mem.indexOf(u8, explain, "pushdown=range=id, projection=[id,sales], bounds=i32[min=2,max=3]") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "range=id") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "projection=[id,sales]") != null);

    var result = try scan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 3), result.height());
    try std.testing.expectEqual(@as(usize, 2), result.width());
    try std.testing.expectEqual(DeviceDType.i32, try result.columnDType("id"));
    try std.testing.expectEqual(DeviceDType.f64, try result.columnDType("sales"));
    try std.testing.expectEqual(@as(?usize, null), result.columnIndex("active"));

    var replacement_scan = try DeviceParquetScan.init(gpa, bytes, .cpu);
    defer replacement_scan.deinit();
    try replacement_scan.whereNull("sales", true);
    try replacement_scan.whereRange("id", .{ .i32 = .{ .min = 2 } });
    const range_replacement_explain = try replacement_scan.explain(gpa);
    defer gpa.free(range_replacement_explain);
    try std.testing.expect(std.mem.indexOf(u8, range_replacement_explain, "pushdown=range=id, bounds=i32[min=2,max=null]") != null);
    try std.testing.expect(std.mem.indexOf(u8, range_replacement_explain, "range=id") != null);
    try std.testing.expect(std.mem.indexOf(u8, range_replacement_explain, "null=sales") == null);

    try replacement_scan.whereNull("sales", false);
    const null_replacement_explain = try replacement_scan.explain(gpa);
    defer gpa.free(null_replacement_explain);
    try std.testing.expect(std.mem.indexOf(u8, null_replacement_explain, "pushdown=null=sales:non_null") != null);
    try std.testing.expect(std.mem.indexOf(u8, null_replacement_explain, "null=sales:non_null") != null);
    try std.testing.expect(std.mem.indexOf(u8, null_replacement_explain, "range=id") == null);

    var non_null_result = try replacement_scan.collect();
    defer non_null_result.deinit();
    const non_null_ids = try (try non_null_result.column("id")).i32.toOwnedSlice(gpa);
    defer gpa.free(non_null_ids);
    // Direct Boltha table pruning is row-group level. This one-row-group file
    // contains at least one non-null sales value, so the scan keeps the full
    // group. Lazy collect applies row-level filters after scan pushdown.
    try std.testing.expectEqualSlices(i32, &.{ 1, 2, 3 }, non_null_ids);

    var null_scan = try DeviceParquetScan.init(gpa, bytes, .cpu);
    defer null_scan.deinit();
    try null_scan.whereNull("sales", true);
    try null_scan.select(&.{"id"});
    const null_explain = try null_scan.explain(gpa);
    defer gpa.free(null_explain);
    try std.testing.expect(std.mem.indexOf(u8, null_explain, "pushdown=null=sales:only, projection=[id]") != null);
    try std.testing.expect(std.mem.indexOf(u8, null_explain, "null=sales:only") != null);
    var null_result = try null_scan.collect();
    defer null_result.deinit();
    const null_ids = try (try null_result.column("id")).i32.toOwnedSlice(gpa);
    defer gpa.free(null_ids);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2, 3 }, null_ids);
}

test "device lazy parquet scan pushes between filters as range predicates" {
    const gpa = std.testing.allocator;

    var id = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 2, 3, 4 }, .cpu);
    defer id.deinit();
    var sales = try DeviceColumn.fromSlice(f64, gpa, &.{ 1.0, 3.0, 5.0, 8.0 }, .cpu);
    defer sales.deinit();
    var active = try DeviceColumn.fromSlice(bool, gpa, &.{ true, false, true, false }, .cpu);
    defer active.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "id", .data = id },
        .{ .name = "sales", .data = sales },
        .{ .name = "active", .data = active },
    });
    defer table.deinit();

    const bytes = try table.toParquetBytes(gpa);
    defer gpa.free(bytes);

    var between_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer between_scan.deinit();
    try between_scan.filterBetweenColumn("sales", f64, 3.0, 5.0);
    try between_scan.select(&.{"id"});

    const between_explain = try between_scan.explain(gpa);
    defer gpa.free(between_explain);
    try std.testing.expect(std.mem.indexOf(u8, between_explain, "scan_pushdown: range=sales, projection=[sales,id]") != null);
    try std.testing.expect(std.mem.indexOf(u8, between_explain, "bounds=f64[min=3,max=5]") != null);
    try std.testing.expect(std.mem.indexOf(u8, between_explain, "filter_between_column(sales, lower:f64, upper:f64") != null);

    var between = try between_scan.collect();
    defer between.deinit();
    try std.testing.expectEqual(@as(usize, 2), between.height());
    try std.testing.expectEqual(@as(usize, 1), between.width());
    try std.testing.expectEqual(@as(?usize, null), between.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, null), between.columnIndex("active"));
    const between_ids = try (try between.column("id")).i32.toOwnedSlice(gpa);
    defer gpa.free(between_ids);
    try std.testing.expectEqualSlices(i32, &.{ 2, 3 }, between_ids);

    var exclusive_id_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer exclusive_id_scan.deinit();
    try exclusive_id_scan.filterBetweenColumnClosed("id", i32, 1, 4, false, false);
    try exclusive_id_scan.select(&.{"sales"});

    const exclusive_id_explain = try exclusive_id_scan.explain(gpa);
    defer gpa.free(exclusive_id_explain);
    try std.testing.expect(std.mem.indexOf(u8, exclusive_id_explain, "scan_pushdown: range=id, projection=[id,sales]") != null);
    try std.testing.expect(std.mem.indexOf(u8, exclusive_id_explain, "bounds=i32[min=2,max=3]") != null);

    var exclusive_id = try exclusive_id_scan.collect();
    defer exclusive_id.deinit();
    try std.testing.expectEqual(@as(usize, 2), exclusive_id.height());
    try std.testing.expectEqual(@as(?usize, null), exclusive_id.columnIndex("id"));
    const exclusive_sales = try (try exclusive_id.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(exclusive_sales);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 5.0 }, exclusive_sales);

    var scalar_gt_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer scalar_gt_scan.deinit();
    try scalar_gt_scan.filterColumnScalar("id", i32, 1, .gt);
    try scalar_gt_scan.select(&.{"sales"});

    const scalar_gt_explain = try scalar_gt_scan.explain(gpa);
    defer gpa.free(scalar_gt_explain);
    try std.testing.expect(std.mem.indexOf(u8, scalar_gt_explain, "scan_pushdown: range=id, projection=[id,sales]") != null);
    try std.testing.expect(std.mem.indexOf(u8, scalar_gt_explain, "bounds=i32[min=2,max=null]") != null);

    var scalar_lt_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer scalar_lt_scan.deinit();
    try scalar_lt_scan.filterColumnScalar("id", i32, 4, .lt);
    try scalar_lt_scan.select(&.{"sales"});

    const scalar_lt_explain = try scalar_lt_scan.explain(gpa);
    defer gpa.free(scalar_lt_explain);
    try std.testing.expect(std.mem.indexOf(u8, scalar_lt_explain, "scan_pushdown: range=id, projection=[id,sales]") != null);
    try std.testing.expect(std.mem.indexOf(u8, scalar_lt_explain, "bounds=i32[min=null,max=3]") != null);

    var drop_lt_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer drop_lt_scan.deinit();
    try drop_lt_scan.dropColumnScalar("id", i32, 3, .lt);
    try drop_lt_scan.select(&.{"sales"});

    const drop_lt_explain = try drop_lt_scan.explain(gpa);
    defer gpa.free(drop_lt_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_lt_explain, "scan_pushdown: range=id, projection=[id,sales]") != null);
    try std.testing.expect(std.mem.indexOf(u8, drop_lt_explain, "bounds=i32[min=3,max=null]") != null);
    var drop_lt = try drop_lt_scan.collect();
    defer drop_lt.deinit();
    const drop_lt_sales = try (try drop_lt.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(drop_lt_sales);
    try std.testing.expectEqualSlices(f64, &.{ 5.0, 8.0 }, drop_lt_sales);

    var drop_ge_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer drop_ge_scan.deinit();
    try drop_ge_scan.dropColumnScalar("id", i32, 3, .ge);
    try drop_ge_scan.select(&.{"sales"});

    const drop_ge_explain = try drop_ge_scan.explain(gpa);
    defer gpa.free(drop_ge_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_ge_explain, "bounds=i32[min=null,max=2]") != null);

    var scalar_intersection_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer scalar_intersection_scan.deinit();
    try scalar_intersection_scan.filterColumnScalar("id", i32, 1, .gt);
    try scalar_intersection_scan.filterColumnScalar("id", i32, 4, .lt);
    try scalar_intersection_scan.select(&.{"sales"});

    const scalar_intersection_explain = try scalar_intersection_scan.explain(gpa);
    defer gpa.free(scalar_intersection_explain);
    try std.testing.expect(std.mem.indexOf(u8, scalar_intersection_explain, "scan_pushdown: range=id, projection=[id,sales]") != null);
    try std.testing.expect(std.mem.indexOf(u8, scalar_intersection_explain, "bounds=i32[min=2,max=3]") != null);
    var scalar_intersection = try scalar_intersection_scan.collect();
    defer scalar_intersection.deinit();
    const scalar_intersection_sales = try (try scalar_intersection.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(scalar_intersection_sales);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 5.0 }, scalar_intersection_sales);

    var mixed_intersection_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer mixed_intersection_scan.deinit();
    try mixed_intersection_scan.filterColumnScalar("id", i32, 1, .gt);
    try mixed_intersection_scan.filterBetweenColumn("id", i32, 1, 3);
    try mixed_intersection_scan.select(&.{"sales"});

    const mixed_intersection_explain = try mixed_intersection_scan.explain(gpa);
    defer gpa.free(mixed_intersection_explain);
    try std.testing.expect(std.mem.indexOf(u8, mixed_intersection_explain, "bounds=i32[min=2,max=3]") != null);

    var outside_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer outside_scan.deinit();
    try outside_scan.filterOutsideColumn("sales", f64, 3.0, 5.0);
    try outside_scan.select(&.{"id"});

    const outside_explain = try outside_scan.explain(gpa);
    defer gpa.free(outside_explain);
    // Outside filters are a disjoint range union, so the planner only pushes
    // the column projection and lets the eager dataframe filter enforce rows.
    try std.testing.expect(std.mem.indexOf(u8, outside_explain, "scan_pushdown: projection=[sales,id]") != null);

    var outside = try outside_scan.collect();
    defer outside.deinit();
    try std.testing.expectEqual(@as(usize, 2), outside.height());
    const outside_ids = try (try outside.column("id")).i32.toOwnedSlice(gpa);
    defer gpa.free(outside_ids);
    try std.testing.expectEqualSlices(i32, &.{ 1, 4 }, outside_ids);
}

test "device lazy parquet scan pushes source boolean masks as range predicates" {
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

    var filter_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer filter_scan.deinit();
    try filter_scan.filterColumn("active");
    try filter_scan.select(&.{"id"});

    const filter_explain = try filter_scan.explain(gpa);
    defer gpa.free(filter_explain);
    try std.testing.expect(std.mem.indexOf(u8, filter_explain, "scan_pushdown: range=active, projection=[active,id]") != null);
    try std.testing.expect(std.mem.indexOf(u8, filter_explain, "bounds=bool[min=true,max=true]") != null);
    var filtered = try filter_scan.collect();
    defer filtered.deinit();
    const filtered_ids = try (try filtered.column("id")).i32.toOwnedSlice(gpa);
    defer gpa.free(filtered_ids);
    try std.testing.expectEqualSlices(i32, &.{ 1, 3 }, filtered_ids);

    var drop_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer drop_scan.deinit();
    try drop_scan.dropRowsByColumnMask("active");
    try drop_scan.select(&.{"id"});

    const drop_explain = try drop_scan.explain(gpa);
    defer gpa.free(drop_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_explain, "scan_pushdown: range=active, projection=[active,id]") != null);
    try std.testing.expect(std.mem.indexOf(u8, drop_explain, "bounds=bool[min=false,max=false]") != null);
    var dropped = try drop_scan.collect();
    defer dropped.deinit();
    const dropped_ids = try (try dropped.column("id")).i32.toOwnedSlice(gpa);
    defer gpa.free(dropped_ids);
    try std.testing.expectEqualSlices(i32, &.{2}, dropped_ids);
}

test "device lazy parquet scan pushes singleton isin values as range predicates" {
    const gpa = std.testing.allocator;

    var id = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 2, 3 }, .cpu);
    defer id.deinit();
    var sales = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 3.0, 5.0 }, .cpu);
    defer sales.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "id", .data = id },
        .{ .name = "sales", .data = sales },
    });
    defer table.deinit();

    const bytes = try table.toParquetBytes(gpa);
    defer gpa.free(bytes);

    var singleton_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer singleton_scan.deinit();
    try singleton_scan.filterIsInValues("sales", f64, &.{3.0});
    try singleton_scan.select(&.{"id"});

    const singleton_explain = try singleton_scan.explain(gpa);
    defer gpa.free(singleton_explain);
    try std.testing.expect(std.mem.indexOf(u8, singleton_explain, "scan_pushdown: range=sales, projection=[sales,id]") != null);
    try std.testing.expect(std.mem.indexOf(u8, singleton_explain, "bounds=f64[min=3,max=3]") != null);
    try std.testing.expect(std.mem.indexOf(u8, singleton_explain, "filter_isin_values(sales, values_dtype=f64, values_len=1, invert=false)") != null);

    var singleton = try singleton_scan.collect();
    defer singleton.deinit();
    const singleton_ids = try (try singleton.column("id")).i32.toOwnedSlice(gpa);
    defer gpa.free(singleton_ids);
    try std.testing.expectEqualSlices(i32, &.{2}, singleton_ids);

    var multi_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer multi_scan.deinit();
    try multi_scan.filterIsInValues("sales", f64, &.{ 3.0, 5.0 });
    try multi_scan.select(&.{"id"});
    const multi_explain = try multi_scan.explain(gpa);
    defer gpa.free(multi_explain);
    try std.testing.expect(std.mem.indexOf(u8, multi_explain, "scan_pushdown: projection=[sales,id]") != null);
    try std.testing.expect(std.mem.indexOf(u8, multi_explain, "range=sales") == null);

    var null_candidate = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{3.0}, &.{false}, .cpu);
    defer null_candidate.deinit();
    var null_candidate_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer null_candidate_scan.deinit();
    try null_candidate_scan.filterIsInValuesColumn("sales", null_candidate);
    try null_candidate_scan.select(&.{"id"});

    const null_candidate_explain = try null_candidate_scan.explain(gpa);
    defer gpa.free(null_candidate_explain);
    try std.testing.expect(std.mem.indexOf(u8, null_candidate_explain, "scan_pushdown: projection=[sales,id]") != null);
    try std.testing.expect(std.mem.indexOf(u8, null_candidate_explain, "range=sales") == null);
}

test "device lazy parquet scan pushes literal isin columns as range predicates" {
    const gpa = std.testing.allocator;

    var id = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 2, 3 }, .cpu);
    defer id.deinit();
    var sales = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 3.0, 5.0 }, .cpu);
    defer sales.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "id", .data = id },
        .{ .name = "sales", .data = sales },
    });
    defer table.deinit();

    const bytes = try table.toParquetBytes(gpa);
    defer gpa.free(bytes);

    var literal_isin_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer literal_isin_scan.deinit();
    try literal_isin_scan.withColumnLiteral("needle", f64, 3.0);
    try literal_isin_scan.filterIsInColumn("sales", "needle");
    try literal_isin_scan.select(&.{"id"});

    const literal_isin_explain = try literal_isin_scan.explain(gpa);
    defer gpa.free(literal_isin_explain);
    try std.testing.expect(std.mem.indexOf(u8, literal_isin_explain, "scan_pushdown: range=sales, projection=[sales,id]") != null);
    try std.testing.expect(std.mem.indexOf(u8, literal_isin_explain, "bounds=f64[min=3,max=3]") != null);
    try std.testing.expect(std.mem.indexOf(u8, literal_isin_explain, "filter_isin_column(sales, test:needle, invert=false)") != null);

    var literal_isin = try literal_isin_scan.collect();
    defer literal_isin.deinit();
    const literal_isin_ids = try (try literal_isin.column("id")).i32.toOwnedSlice(gpa);
    defer gpa.free(literal_isin_ids);
    try std.testing.expectEqualSlices(i32, &.{2}, literal_isin_ids);

    var literal_notin_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer literal_notin_scan.deinit();
    try literal_notin_scan.withColumnLiteral("needle", f64, 3.0);
    try literal_notin_scan.filterNotInColumn("sales", "needle");
    try literal_notin_scan.select(&.{"id"});
    const literal_notin_explain = try literal_notin_scan.explain(gpa);
    defer gpa.free(literal_notin_explain);
    try std.testing.expect(std.mem.indexOf(u8, literal_notin_explain, "scan_pushdown: projection=[sales,id]") != null);
    try std.testing.expect(std.mem.indexOf(u8, literal_notin_explain, "range=sales") == null);

    var positioned_literal_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer positioned_literal_scan.deinit();
    try positioned_literal_scan.withColumnLiteralAt("needle", f64, 3.0, 0);
    try positioned_literal_scan.filterIsInColumn("sales", "needle");
    try positioned_literal_scan.select(&.{"id"});
    const positioned_literal_explain = try positioned_literal_scan.explain(gpa);
    defer gpa.free(positioned_literal_explain);
    try std.testing.expect(std.mem.indexOf(u8, positioned_literal_explain, "scan_pushdown: none") != null);
    try std.testing.expect(std.mem.indexOf(u8, positioned_literal_explain, "with_column_literal_at(needle=scalar:f64, index=0)") != null);

    var overwritten_literal_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer overwritten_literal_scan.deinit();
    try overwritten_literal_scan.withColumnLiteral("needle", f64, 3.0);
    try overwritten_literal_scan.withColumnScalar("needle", "sales", f64, 1.0, .add);
    try overwritten_literal_scan.filterIsInColumn("sales", "needle");
    try overwritten_literal_scan.select(&.{"id"});
    const overwritten_literal_explain = try overwritten_literal_scan.explain(gpa);
    defer gpa.free(overwritten_literal_explain);
    try std.testing.expect(std.mem.indexOf(u8, overwritten_literal_explain, "scan_pushdown: projection=[sales,id]") != null);
    try std.testing.expect(std.mem.indexOf(u8, overwritten_literal_explain, "range=sales") == null);

    var unary_overwrite_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer unary_overwrite_scan.deinit();
    try unary_overwrite_scan.withColumnLiteral("needle", f64, 3.0);
    try unary_overwrite_scan.withColumnAbs("needle", "sales");
    try unary_overwrite_scan.filterIsInColumn("sales", "needle");
    try unary_overwrite_scan.select(&.{"id"});
    const unary_overwrite_explain = try unary_overwrite_scan.explain(gpa);
    defer gpa.free(unary_overwrite_explain);
    try std.testing.expect(std.mem.indexOf(u8, unary_overwrite_explain, "scan_pushdown: projection=[sales,id]") != null);
    try std.testing.expect(std.mem.indexOf(u8, unary_overwrite_explain, "range=sales") == null);

    var binary_overwrite_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer binary_overwrite_scan.deinit();
    try binary_overwrite_scan.withColumnLiteral("needle", f64, 3.0);
    try binary_overwrite_scan.withColumnBinary("needle", "sales", "sales", .add);
    try binary_overwrite_scan.filterIsInColumn("sales", "needle");
    try binary_overwrite_scan.select(&.{"id"});
    const binary_overwrite_explain = try binary_overwrite_scan.explain(gpa);
    defer gpa.free(binary_overwrite_explain);
    try std.testing.expect(std.mem.indexOf(u8, binary_overwrite_explain, "scan_pushdown: projection=[sales,id]") != null);
    try std.testing.expect(std.mem.indexOf(u8, binary_overwrite_explain, "range=sales") == null);
}

test "device lazy frame pushes scalar filters and projection into parquet scan source" {
    const gpa = std.testing.allocator;

    var id = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 2, 3 }, .cpu);
    defer id.deinit();
    var sales = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 3.0, 5.0 }, .cpu);
    defer sales.deinit();
    var ratio = try DeviceColumn.fromSlice(f64, gpa, &.{ -0.25, 0.25, 0.5 }, .cpu);
    defer ratio.deinit();
    var active = try DeviceColumn.fromSlice(bool, gpa, &.{ true, false, true }, .cpu);
    defer active.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "id", .data = id },
        .{ .name = "sales", .data = sales },
        .{ .name = "ratio", .data = ratio },
        .{ .name = "active", .data = active },
    });
    defer table.deinit();

    const bytes = try table.toParquetBytes(gpa);
    defer gpa.free(bytes);

    var lazy_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer lazy_scan.deinit();
    try lazy_scan.withColumnScalar("sales_x2", "sales", f64, 2.0, .mul);
    try lazy_scan.withColumnAbs("sales_abs", "sales");
    try lazy_scan.withColumnNeg("sales_neg", "sales");
    try lazy_scan.withColumnSign("sales_neg_sign", "sales_neg");
    try lazy_scan.withColumnSquare("sales_square", "sales");
    try lazy_scan.withColumnReciprocal("sales_recip", "sales");
    try lazy_scan.withColumnSqrt("sales_sqrt", "sales");
    try lazy_scan.withColumnRsqrt("sales_rsqrt", "sales");
    try lazy_scan.withColumnCbrt("sales_cbrt", "sales");
    try lazy_scan.withColumnFloor("sales_recip_floor", "sales_recip");
    try lazy_scan.withColumnCeil("sales_recip_ceil", "sales_recip");
    try lazy_scan.withColumnRound("sales_recip_round", "sales_recip");
    try lazy_scan.withColumnTrunc("sales_recip_trunc", "sales_recip");
    try lazy_scan.withColumnDeg2rad("sales_deg2rad", "sales");
    try lazy_scan.withColumnRad2deg("sales_roundtrip_deg", "sales_deg2rad");
    try lazy_scan.withColumnExpit("ratio_expit", "ratio");
    try lazy_scan.withColumnLogit("ratio_logit", "ratio");
    try lazy_scan.withColumnSoftplus("ratio_softplus", "ratio");
    try lazy_scan.withColumnLogsigmoid("ratio_logsigmoid", "ratio");
    try lazy_scan.withColumnRelu("sales_neg_relu", "sales_neg");
    try lazy_scan.withColumnLeakyRelu("sales_neg_leaky_relu", "sales_neg", f64, 0.1);
    try lazy_scan.withColumnRelu6("sales_relu6", "sales");
    try lazy_scan.withColumnPowScalar("sales_pow2", "sales", f64, 2.0);
    try lazy_scan.withColumnFloorDivScalar("sales_floor_div2", "sales", f64, 2.0);
    try lazy_scan.withColumnModScalar("sales_mod2", "sales", f64, 2.0);
    try lazy_scan.withColumnRemainderScalar("sales_remainder2", "sales", f64, 2.0);
    try lazy_scan.withColumnLogAddExpScalar("sales_logaddexp0", "sales", f64, 0.0);
    try lazy_scan.withColumnLogAddExp2Scalar("sales_logaddexp2_0", "sales", f64, 0.0);
    try lazy_scan.withColumnXlogyScalar("sales_xlogy_e", "sales", f64, std.math.e);
    try lazy_scan.withColumnFmaxScalar("sales_fmax4", "sales", f64, 4.0);
    try lazy_scan.withColumnFminScalar("sales_fmin4", "sales", f64, 4.0);
    try lazy_scan.withColumnHypotScalar("sales_hypot4", "sales", f64, 4.0);
    try lazy_scan.withColumnAtan2Scalar("sales_atan2_4", "sales", f64, 4.0);
    try lazy_scan.withColumnNextAfterScalar("sales_next_after6", "sales", f64, 6.0);
    try lazy_scan.withColumnCopysignScalar("sales_copysign_neg", "sales", f64, -1.0);
    try lazy_scan.withColumnHeavisideScalar("sales_neg_heaviside", "sales_neg", f64, 0.25);
    try lazy_scan.withColumnLdexpScalar("sales_ldexp1", "sales", 1);
    try lazy_scan.withColumnLerpScalar("sales_lerp_ratio", "sales", "ratio", f64, 0.25);
    try lazy_scan.withColumnAddcmulScalar("sales_addcmul", "sales", "ratio", "ratio", f64, 2.0);
    try lazy_scan.withColumnAddcdivScalar("sales_addcdiv", "sales", "sales", "sales_x2", f64, 0.5);
    try lazy_scan.withColumnClipArray("sales_clipped", "sales", "ratio", "sales_addcdiv");
    try lazy_scan.withColumnIscloseScalar("sales_close3", "sales", f64, 3.0, 0.0, 0.1);
    try lazy_scan.withColumnLogicalOrScalar("active_or_false", "active", false);
    try lazy_scan.withColumnWhereScalar("sales_when_active", "sales", "active", f64, -1.0);
    try lazy_scan.withColumnWhere("sales_where_active", "sales", "active", "sales_neg");
    try lazy_scan.withColumnMaskedPutScalar("sales_masked_active", "sales", "active", f64, -2.0);
    try lazy_scan.withColumnLogicalXor("active_xor_copy", "active", "active_or_false");
    try lazy_scan.withColumnThreshold("sales_neg_threshold", "sales_neg", f64, -4.0, 0.0);
    try lazy_scan.withColumnHardtanh("sales_neg_hardtanh", "sales_neg", f64, -4.0, -1.0);
    try lazy_scan.withColumnMaximumScalar("sales_neg_max", "sales_neg", f64, -4.0);
    try lazy_scan.withColumnMinimumScalar("sales_neg_min", "sales_neg", f64, -4.0);
    try lazy_scan.withColumnClipMin("sales_neg_clip_min", "sales_neg", f64, -4.0);
    try lazy_scan.withColumnClipMax("sales_neg_clip_max", "sales_neg", f64, -4.0);
    try lazy_scan.withColumnHardshrink("sales_neg_hardshrink", "sales_neg", f64, 4.0);
    try lazy_scan.withColumnSoftshrink("sales_neg_softshrink", "sales_neg", f64, 4.0);
    try lazy_scan.withColumnTanhshrink("sales_neg_tanhshrink", "sales_neg");
    try lazy_scan.withColumnElu("sales_neg_elu", "sales_neg", f64, 0.5);
    try lazy_scan.withColumnCelu("sales_neg_celu", "sales_neg", f64, 2.0);
    try lazy_scan.withColumnSoftsign("sales_neg_softsign", "sales_neg");
    try lazy_scan.withColumnHardsigmoid("sales_neg_hardsigmoid", "sales_neg");
    try lazy_scan.withColumnHardswish("sales_neg_hardswish", "sales_neg");
    try lazy_scan.withColumnSilu("sales_neg_silu", "sales_neg");
    try lazy_scan.withColumnSwish("sales_neg_swish", "sales_neg");
    try lazy_scan.withColumnMish("sales_neg_mish", "sales_neg");
    try lazy_scan.withColumnGelu("sales_neg_gelu", "sales_neg");
    try lazy_scan.withColumnSelu("sales_neg_selu", "sales_neg");
    try lazy_scan.withColumnExp("sales_exp", "sales");
    try lazy_scan.withColumnExp2("sales_exp2", "sales");
    try lazy_scan.withColumnExpm1("sales_expm1", "sales");
    try lazy_scan.withColumnSin("sales_sin", "sales");
    try lazy_scan.withColumnCos("sales_cos", "sales");
    try lazy_scan.withColumnTan("sales_tan", "sales");
    try lazy_scan.withColumnAsin("ratio_asin", "ratio");
    try lazy_scan.withColumnAcos("ratio_acos", "ratio");
    try lazy_scan.withColumnAtan("ratio_atan", "ratio");
    try lazy_scan.withColumnSinh("sales_sinh", "sales");
    try lazy_scan.withColumnCosh("sales_cosh", "sales");
    try lazy_scan.withColumnTanh("sales_tanh", "sales");
    try lazy_scan.withColumnAsinh("sales_asinh", "sales");
    try lazy_scan.withColumnAcosh("sales_acosh", "sales");
    try lazy_scan.withColumnAtanh("ratio_atanh", "ratio");
    try lazy_scan.withColumnLog("sales_log", "sales");
    try lazy_scan.withColumnLog1p("sales_log1p", "sales");
    try lazy_scan.withColumnLgamma("sales_lgamma", "sales");
    try lazy_scan.withColumnSinc("sales_sinc", "sales");
    try lazy_scan.withColumnLog2("sales_log2", "sales");
    try lazy_scan.withColumnLog10("sales_log10", "sales");
    try lazy_scan.filterColumnScalar("sales", f64, 2.5, .gt);
    try lazy_scan.select(&.{ "sales_x2", "sales_abs", "sales_neg", "sales_neg_sign", "sales_square", "sales_recip", "sales_sqrt", "sales_rsqrt", "sales_cbrt", "sales_recip_floor", "sales_recip_ceil", "sales_recip_round", "sales_recip_trunc", "sales_deg2rad", "sales_roundtrip_deg", "ratio_expit", "ratio_logit", "ratio_softplus", "ratio_logsigmoid", "sales_neg_relu", "sales_neg_leaky_relu", "sales_relu6", "sales_pow2", "sales_floor_div2", "sales_mod2", "sales_remainder2", "sales_logaddexp0", "sales_logaddexp2_0", "sales_xlogy_e", "sales_fmax4", "sales_fmin4", "sales_hypot4", "sales_atan2_4", "sales_next_after6", "sales_copysign_neg", "sales_neg_heaviside", "sales_ldexp1", "sales_lerp_ratio", "sales_addcmul", "sales_addcdiv", "sales_clipped", "sales_close3", "active_or_false", "sales_when_active", "sales_where_active", "sales_masked_active", "active_xor_copy", "sales_neg_threshold", "sales_neg_hardtanh", "sales_neg_max", "sales_neg_min", "sales_neg_clip_min", "sales_neg_clip_max", "sales_neg_hardshrink", "sales_neg_softshrink", "sales_neg_tanhshrink", "sales_neg_elu", "sales_neg_celu", "sales_neg_softsign", "sales_neg_hardsigmoid", "sales_neg_hardswish", "sales_neg_silu", "sales_neg_swish", "sales_neg_mish", "sales_neg_gelu", "sales_neg_selu", "sales_exp", "sales_exp2", "sales_expm1", "sales_sin", "sales_cos", "sales_tan", "ratio_asin", "ratio_acos", "ratio_atan", "sales_sinh", "sales_cosh", "sales_tanh", "sales_asinh", "sales_acosh", "ratio_atanh", "sales_log", "sales_log1p", "sales_lgamma", "sales_sinc", "sales_log2", "sales_log10", "id" });

    const explain = try lazy_scan.explain(gpa);
    defer gpa.free(explain);
    try std.testing.expect(std.mem.indexOf(u8, explain, "source=parquet_scan") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_scalar(sales_x2") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_abs(sales_abs=abs(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_neg(sales_neg=neg(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_sign(sales_neg_sign=sign(sales_neg))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_square(sales_square=square(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_reciprocal(sales_recip=reciprocal(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_sqrt(sales_sqrt=sqrt(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_rsqrt(sales_rsqrt=rsqrt(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_cbrt(sales_cbrt=cbrt(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_floor(sales_recip_floor=floor(sales_recip))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_ceil(sales_recip_ceil=ceil(sales_recip))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_round(sales_recip_round=round(sales_recip))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_trunc(sales_recip_trunc=trunc(sales_recip))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_deg2rad(sales_deg2rad=deg2rad(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_rad2deg(sales_roundtrip_deg=rad2deg(sales_deg2rad))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_expit(ratio_expit=expit(ratio))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_logit(ratio_logit=logit(ratio))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_softplus(ratio_softplus=softplus(ratio))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_logsigmoid(ratio_logsigmoid=logsigmoid(ratio))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_relu(sales_neg_relu=relu(sales_neg))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_leaky_relu(sales_neg_leaky_relu=leaky_relu(sales_neg, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_relu6(sales_relu6=relu6(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_pow_scalar(sales_pow2=pow(sales, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_floor_div_scalar(sales_floor_div2=floor_div(sales, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_mod_scalar(sales_mod2=mod(sales, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_remainder_scalar(sales_remainder2=remainder(sales, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_log_add_exp_scalar(sales_logaddexp0=log_add_exp(sales, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_log_add_exp2_scalar(sales_logaddexp2_0=log_add_exp2(sales, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_xlogy_scalar(sales_xlogy_e=xlogy(sales, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_fmax_scalar(sales_fmax4=fmax(sales, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_fmin_scalar(sales_fmin4=fmin(sales, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_hypot_scalar(sales_hypot4=hypot(sales, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_atan2_scalar(sales_atan2_4=atan2(sales, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_next_after_scalar(sales_next_after6=next_after(sales, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_copysign_scalar(sales_copysign_neg=copysign(sales, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_heaviside_scalar(sales_neg_heaviside=heaviside(sales_neg, value_at_zero:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_ldexp_scalar(sales_ldexp1=ldexp(sales, exponent:1))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_lerp_scalar(sales_lerp_ratio=lerp(sales, ratio, weight:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_addcmul_scalar(sales_addcmul=addcmul(sales, ratio, ratio, value:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_addcdiv_scalar(sales_addcdiv=addcdiv(sales, sales, sales_x2, value:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_clip_array(sales_clipped=clip_array(sales, min:ratio, max:sales_addcdiv))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_isclose_scalar(sales_close3=isclose(sales, scalar:f64, rtol:f64, atol:f64, equal_nan=false))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_logical_scalar(active_or_false=logical_or(active, scalar:false))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_where_scalar(sales_when_active=where(sales, mask:active, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_where(sales_where_active=where(sales, mask:active, other:sales_neg))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_masked_put_scalar(sales_masked_active=masked_put(sales, mask:active, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_logical(active_xor_copy=logical_xor(active, active_or_false))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_threshold(sales_neg_threshold=threshold(sales_neg, threshold:f64, replacement:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_hardtanh(sales_neg_hardtanh=hardtanh(sales_neg, min:f64, max:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_maximum_scalar(sales_neg_max=maximum(sales_neg, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_minimum_scalar(sales_neg_min=minimum(sales_neg, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_clip_min(sales_neg_clip_min=clip_min(sales_neg, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_clip_max(sales_neg_clip_max=clip_max(sales_neg, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_hardshrink(sales_neg_hardshrink=hardshrink(sales_neg, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_softshrink(sales_neg_softshrink=softshrink(sales_neg, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_tanhshrink(sales_neg_tanhshrink=tanhshrink(sales_neg))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_elu(sales_neg_elu=elu(sales_neg, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_celu(sales_neg_celu=celu(sales_neg, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_softsign(sales_neg_softsign=softsign(sales_neg))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_hardsigmoid(sales_neg_hardsigmoid=hardsigmoid(sales_neg))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_hardswish(sales_neg_hardswish=hardswish(sales_neg))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_silu(sales_neg_silu=silu(sales_neg))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_swish(sales_neg_swish=swish(sales_neg))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_mish(sales_neg_mish=mish(sales_neg))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_gelu(sales_neg_gelu=gelu(sales_neg))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_selu(sales_neg_selu=selu(sales_neg))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_exp(sales_exp=exp(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_exp2(sales_exp2=exp2(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_expm1(sales_expm1=expm1(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_sin(sales_sin=sin(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_cos(sales_cos=cos(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_tan(sales_tan=tan(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_asin(ratio_asin=asin(ratio))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_acos(ratio_acos=acos(ratio))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_atan(ratio_atan=atan(ratio))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_sinh(sales_sinh=sinh(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_cosh(sales_cosh=cosh(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_tanh(sales_tanh=tanh(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_asinh(sales_asinh=asinh(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_acosh(sales_acosh=acosh(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_atanh(ratio_atanh=atanh(ratio))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_log(sales_log=log(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_log1p(sales_log1p=log1p(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_lgamma(sales_lgamma=lgamma(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_sinc(sales_sinc=sinc(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_log2(sales_log2=log2(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_log10(sales_log10=log10(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "scan_pushdown: range=sales, projection=[sales,ratio,active,id]") != null);

    var result = try lazy_scan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 2), result.height());
    try std.testing.expectEqual(@as(usize, 88), result.width());
    try std.testing.expectEqual(@as(?usize, null), result.columnIndex("active"));
    try std.testing.expectEqual(@as(?usize, null), result.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, null), result.columnIndex("ratio"));
    const result_sales_x2 = try (try result.column("sales_x2")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_x2);
    const result_sales_abs = try (try result.column("sales_abs")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_abs);
    const result_sales_neg = try (try result.column("sales_neg")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg);
    const result_sales_neg_sign = try (try result.column("sales_neg_sign")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_sign);
    const result_sales_square = try (try result.column("sales_square")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_square);
    const result_sales_recip = try (try result.column("sales_recip")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_recip);
    const result_sales_sqrt = try (try result.column("sales_sqrt")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_sqrt);
    const result_sales_rsqrt = try (try result.column("sales_rsqrt")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_rsqrt);
    const result_sales_cbrt = try (try result.column("sales_cbrt")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_cbrt);
    const result_sales_recip_floor = try (try result.column("sales_recip_floor")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_recip_floor);
    const result_sales_recip_ceil = try (try result.column("sales_recip_ceil")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_recip_ceil);
    const result_sales_recip_round = try (try result.column("sales_recip_round")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_recip_round);
    const result_sales_recip_trunc = try (try result.column("sales_recip_trunc")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_recip_trunc);
    const result_sales_deg2rad = try (try result.column("sales_deg2rad")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_deg2rad);
    const result_sales_roundtrip_deg = try (try result.column("sales_roundtrip_deg")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_roundtrip_deg);
    const result_ratio_expit = try (try result.column("ratio_expit")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_ratio_expit);
    const result_ratio_logit = try (try result.column("ratio_logit")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_ratio_logit);
    const result_ratio_softplus = try (try result.column("ratio_softplus")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_ratio_softplus);
    const result_ratio_logsigmoid = try (try result.column("ratio_logsigmoid")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_ratio_logsigmoid);
    const result_sales_neg_relu = try (try result.column("sales_neg_relu")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_relu);
    const result_sales_neg_leaky_relu = try (try result.column("sales_neg_leaky_relu")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_leaky_relu);
    const result_sales_relu6 = try (try result.column("sales_relu6")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_relu6);
    const result_sales_pow2 = try (try result.column("sales_pow2")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_pow2);
    const result_sales_floor_div2 = try (try result.column("sales_floor_div2")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_floor_div2);
    const result_sales_mod2 = try (try result.column("sales_mod2")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_mod2);
    const result_sales_remainder2 = try (try result.column("sales_remainder2")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_remainder2);
    const result_sales_logaddexp0 = try (try result.column("sales_logaddexp0")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_logaddexp0);
    const result_sales_logaddexp2_0 = try (try result.column("sales_logaddexp2_0")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_logaddexp2_0);
    const result_sales_xlogy_e = try (try result.column("sales_xlogy_e")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_xlogy_e);
    const result_sales_fmax4 = try (try result.column("sales_fmax4")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_fmax4);
    const result_sales_fmin4 = try (try result.column("sales_fmin4")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_fmin4);
    const result_sales_hypot4 = try (try result.column("sales_hypot4")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_hypot4);
    const result_sales_atan2_4 = try (try result.column("sales_atan2_4")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_atan2_4);
    const result_sales_next_after6 = try (try result.column("sales_next_after6")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_next_after6);
    const result_sales_copysign_neg = try (try result.column("sales_copysign_neg")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_copysign_neg);
    const result_sales_neg_heaviside = try (try result.column("sales_neg_heaviside")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_heaviside);
    const result_sales_ldexp1 = try (try result.column("sales_ldexp1")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_ldexp1);
    const result_sales_lerp_ratio = try (try result.column("sales_lerp_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_lerp_ratio);
    const result_sales_addcmul = try (try result.column("sales_addcmul")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_addcmul);
    const result_sales_addcdiv = try (try result.column("sales_addcdiv")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_addcdiv);
    const result_sales_clipped = try (try result.column("sales_clipped")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_clipped);
    const result_sales_close3 = try (try result.column("sales_close3")).bool.toOwnedSlice(gpa);
    defer gpa.free(result_sales_close3);
    const result_active_or_false = try (try result.column("active_or_false")).bool.toOwnedSlice(gpa);
    defer gpa.free(result_active_or_false);
    const result_sales_when_active = try (try result.column("sales_when_active")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_when_active);
    const result_sales_where_active = try (try result.column("sales_where_active")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_where_active);
    const result_sales_masked_active = try (try result.column("sales_masked_active")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_masked_active);
    const result_active_xor_copy = try (try result.column("active_xor_copy")).bool.toOwnedSlice(gpa);
    defer gpa.free(result_active_xor_copy);
    const result_sales_neg_threshold = try (try result.column("sales_neg_threshold")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_threshold);
    const result_sales_neg_hardtanh = try (try result.column("sales_neg_hardtanh")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_hardtanh);
    const result_sales_neg_max = try (try result.column("sales_neg_max")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_max);
    const result_sales_neg_min = try (try result.column("sales_neg_min")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_min);
    const result_sales_neg_clip_min = try (try result.column("sales_neg_clip_min")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_clip_min);
    const result_sales_neg_clip_max = try (try result.column("sales_neg_clip_max")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_clip_max);
    const result_sales_neg_hardshrink = try (try result.column("sales_neg_hardshrink")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_hardshrink);
    const result_sales_neg_softshrink = try (try result.column("sales_neg_softshrink")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_softshrink);
    const result_sales_neg_tanhshrink = try (try result.column("sales_neg_tanhshrink")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_tanhshrink);
    const result_sales_neg_elu = try (try result.column("sales_neg_elu")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_elu);
    const result_sales_neg_celu = try (try result.column("sales_neg_celu")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_celu);
    const result_sales_neg_softsign = try (try result.column("sales_neg_softsign")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_softsign);
    const result_sales_neg_hardsigmoid = try (try result.column("sales_neg_hardsigmoid")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_hardsigmoid);
    const result_sales_neg_hardswish = try (try result.column("sales_neg_hardswish")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_hardswish);
    const result_sales_neg_silu = try (try result.column("sales_neg_silu")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_silu);
    const result_sales_neg_swish = try (try result.column("sales_neg_swish")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_swish);
    const result_sales_neg_mish = try (try result.column("sales_neg_mish")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_mish);
    const result_sales_neg_gelu = try (try result.column("sales_neg_gelu")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_gelu);
    const result_sales_neg_selu = try (try result.column("sales_neg_selu")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_selu);
    const result_sales_exp = try (try result.column("sales_exp")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_exp);
    const result_sales_exp2 = try (try result.column("sales_exp2")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_exp2);
    const result_sales_expm1 = try (try result.column("sales_expm1")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_expm1);
    const result_sales_sin = try (try result.column("sales_sin")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_sin);
    const result_sales_cos = try (try result.column("sales_cos")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_cos);
    const result_sales_tan = try (try result.column("sales_tan")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_tan);
    const result_ratio_asin = try (try result.column("ratio_asin")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_ratio_asin);
    const result_ratio_acos = try (try result.column("ratio_acos")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_ratio_acos);
    const result_ratio_atan = try (try result.column("ratio_atan")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_ratio_atan);
    const result_sales_sinh = try (try result.column("sales_sinh")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_sinh);
    const result_sales_cosh = try (try result.column("sales_cosh")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_cosh);
    const result_sales_tanh = try (try result.column("sales_tanh")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_tanh);
    const result_sales_asinh = try (try result.column("sales_asinh")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_asinh);
    const result_sales_acosh = try (try result.column("sales_acosh")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_acosh);
    const result_ratio_atanh = try (try result.column("ratio_atanh")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_ratio_atanh);
    const result_sales_log = try (try result.column("sales_log")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_log);
    const result_sales_log1p = try (try result.column("sales_log1p")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_log1p);
    const result_sales_lgamma = try (try result.column("sales_lgamma")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_lgamma);
    const result_sales_sinc = try (try result.column("sales_sinc")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_sinc);
    const result_sales_log2 = try (try result.column("sales_log2")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_log2);
    const result_sales_log10 = try (try result.column("sales_log10")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_log10);
    try std.testing.expectEqualSlices(f64, &.{ 6.0, 10.0 }, result_sales_x2);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 5.0 }, result_sales_abs);
    try std.testing.expectEqualSlices(f64, &.{ -3.0, -5.0 }, result_sales_neg);
    try std.testing.expectEqualSlices(f64, &.{ -1.0, -1.0 }, result_sales_neg_sign);
    try std.testing.expectEqualSlices(f64, &.{ 9.0, 25.0 }, result_sales_square);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), result_sales_recip[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.2), result_sales_recip[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 3.0)), result_sales_sqrt[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 5.0)), result_sales_sqrt[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0) / std.math.sqrt(@as(f64, 3.0)), result_sales_rsqrt[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0) / std.math.sqrt(@as(f64, 5.0)), result_sales_rsqrt[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.cbrt(@as(f64, 3.0)), result_sales_cbrt[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.cbrt(@as(f64, 5.0)), result_sales_cbrt[1], 1e-12);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0 }, result_sales_recip_floor);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 1.0 }, result_sales_recip_ceil);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0 }, result_sales_recip_round);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0 }, result_sales_recip_trunc);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0) * std.math.pi / @as(f64, 180.0), result_sales_deg2rad[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0) * std.math.pi / @as(f64, 180.0), result_sales_deg2rad[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), result_sales_roundtrip_deg[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), result_sales_roundtrip_deg[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0) / (@as(f64, 1.0) + std.math.exp(-@as(f64, 0.25))), result_ratio_expit[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0) / (@as(f64, 1.0) + std.math.exp(-@as(f64, 0.5))), result_ratio_expit[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log(f64, std.math.e, @as(f64, 0.25) / @as(f64, 0.75)), result_ratio_logit[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log(f64, std.math.e, @as(f64, 0.5) / @as(f64, 0.5)), result_ratio_logit[1], 1e-12);
    try std.testing.expectApproxEqAbs(@max(@as(f64, 0.25), @as(f64, 0.0)) + std.math.log1p(std.math.exp(-@abs(@as(f64, 0.25)))), result_ratio_softplus[0], 1e-12);
    try std.testing.expectApproxEqAbs(@max(@as(f64, 0.5), @as(f64, 0.0)) + std.math.log1p(std.math.exp(-@abs(@as(f64, 0.5)))), result_ratio_softplus[1], 1e-12);
    try std.testing.expectApproxEqAbs(-(@max(-@as(f64, 0.25), @as(f64, 0.0)) + std.math.log1p(std.math.exp(-@abs(@as(f64, 0.25))))), result_ratio_logsigmoid[0], 1e-12);
    try std.testing.expectApproxEqAbs(-(@max(-@as(f64, 0.5), @as(f64, 0.0)) + std.math.log1p(std.math.exp(-@abs(@as(f64, 0.5))))), result_ratio_logsigmoid[1], 1e-12);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0 }, result_sales_neg_relu);
    try std.testing.expectApproxEqAbs(@as(f64, -0.3), result_sales_neg_leaky_relu[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.5), result_sales_neg_leaky_relu[1], 1e-12);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 5.0 }, result_sales_relu6);
    try std.testing.expectEqualSlices(f64, &.{ 9.0, 25.0 }, result_sales_pow2);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 2.0 }, result_sales_floor_div2);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 1.0 }, result_sales_mod2);
    try std.testing.expectEqualSlices(f64, result_sales_mod2, result_sales_remainder2);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0) + std.math.log1p(std.math.exp(@as(f64, -3.0))), result_sales_logaddexp0[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0) + std.math.log1p(std.math.exp(@as(f64, -5.0))), result_sales_logaddexp0[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0) + std.math.log2(@as(f64, 1.0) + std.math.pow(f64, 2.0, -@as(f64, 3.0))), result_sales_logaddexp2_0[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0) + std.math.log2(@as(f64, 1.0) + std.math.pow(f64, 2.0, -@as(f64, 5.0))), result_sales_logaddexp2_0[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), result_sales_xlogy_e[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), result_sales_xlogy_e[1], 1e-12);
    try std.testing.expectEqualSlices(f64, &.{ 4.0, 5.0 }, result_sales_fmax4);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 4.0 }, result_sales_fmin4);
    try std.testing.expectApproxEqAbs(std.math.hypot(@as(f64, 3.0), @as(f64, 4.0)), result_sales_hypot4[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.hypot(@as(f64, 5.0), @as(f64, 4.0)), result_sales_hypot4[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.atan2(@as(f64, 3.0), @as(f64, 4.0)), result_sales_atan2_4[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.atan2(@as(f64, 5.0), @as(f64, 4.0)), result_sales_atan2_4[1], 1e-12);
    try std.testing.expectEqual(std.math.nextAfter(f64, @as(f64, 3.0), @as(f64, 6.0)), result_sales_next_after6[0]);
    try std.testing.expectEqual(std.math.nextAfter(f64, @as(f64, 5.0), @as(f64, 6.0)), result_sales_next_after6[1]);
    try std.testing.expectEqualSlices(f64, &.{ -3.0, -5.0 }, result_sales_copysign_neg);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0 }, result_sales_neg_heaviside);
    try std.testing.expectEqualSlices(f64, &.{ 6.0, 10.0 }, result_sales_ldexp1);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0) + (@as(f64, 0.25) - @as(f64, 3.0)) * @as(f64, 0.25), result_sales_lerp_ratio[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0) + (@as(f64, 0.5) - @as(f64, 5.0)) * @as(f64, 0.25), result_sales_lerp_ratio[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0) + @as(f64, 2.0) * @as(f64, 0.25) * @as(f64, 0.25), result_sales_addcmul[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0) + @as(f64, 2.0) * @as(f64, 0.5) * @as(f64, 0.5), result_sales_addcmul[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0) + @as(f64, 0.5) * @as(f64, 3.0) / @as(f64, 6.0), result_sales_addcdiv[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0) + @as(f64, 0.5) * @as(f64, 5.0) / @as(f64, 10.0), result_sales_addcdiv[1], 1e-12);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 5.0 }, result_sales_clipped);
    try std.testing.expectEqualSlices(bool, &.{ true, false }, result_sales_close3);
    try std.testing.expectEqualSlices(bool, &.{ false, true }, result_active_or_false);
    try std.testing.expectEqualSlices(f64, &.{ -1.0, 5.0 }, result_sales_when_active);
    try std.testing.expectEqualSlices(f64, &.{ -3.0, 5.0 }, result_sales_where_active);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, -2.0 }, result_sales_masked_active);
    try std.testing.expectEqualSlices(bool, &.{ false, false }, result_active_xor_copy);
    try std.testing.expectEqualSlices(f64, &.{ -3.0, 0.0 }, result_sales_neg_threshold);
    try std.testing.expectEqualSlices(f64, &.{ -3.0, -4.0 }, result_sales_neg_hardtanh);
    try std.testing.expectEqualSlices(f64, &.{ -3.0, -4.0 }, result_sales_neg_max);
    try std.testing.expectEqualSlices(f64, &.{ -4.0, -5.0 }, result_sales_neg_min);
    try std.testing.expectEqualSlices(f64, &.{ -3.0, -4.0 }, result_sales_neg_clip_min);
    try std.testing.expectEqualSlices(f64, &.{ -4.0, -5.0 }, result_sales_neg_clip_max);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, -5.0 }, result_sales_neg_hardshrink);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, -1.0 }, result_sales_neg_softshrink);
    try std.testing.expectApproxEqAbs(@as(f64, -3.0) - std.math.tanh(@as(f64, -3.0)), result_sales_neg_tanhshrink[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -5.0) - std.math.tanh(@as(f64, -5.0)), result_sales_neg_tanhshrink[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5) * std.math.expm1(@as(f64, -3.0)), result_sales_neg_elu[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5) * std.math.expm1(@as(f64, -5.0)), result_sales_neg_elu[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0) * std.math.expm1(@as(f64, -3.0) / @as(f64, 2.0)), result_sales_neg_celu[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0) * std.math.expm1(@as(f64, -5.0) / @as(f64, 2.0)), result_sales_neg_celu[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -3.0) / @as(f64, 4.0), result_sales_neg_softsign[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -5.0) / @as(f64, 6.0), result_sales_neg_softsign[1], 1e-12);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0 }, result_sales_neg_hardsigmoid);
    try std.testing.expectEqualSlices(f64, &.{ -0.0, -0.0 }, result_sales_neg_hardswish);
    try std.testing.expectApproxEqAbs(@as(f64, -3.0) / (@as(f64, 1.0) + std.math.exp(@as(f64, 3.0))), result_sales_neg_silu[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -5.0) / (@as(f64, 1.0) + std.math.exp(@as(f64, 5.0))), result_sales_neg_silu[1], 1e-12);
    try std.testing.expectApproxEqAbs(result_sales_neg_silu[0], result_sales_neg_swish[0], 1e-12);
    try std.testing.expectApproxEqAbs(result_sales_neg_silu[1], result_sales_neg_swish[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -3.0) * std.math.tanh(@max(@as(f64, -3.0), @as(f64, 0.0)) + std.math.log1p(std.math.exp(-@abs(@as(f64, -3.0))))), result_sales_neg_mish[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -5.0) * std.math.tanh(@max(@as(f64, -5.0), @as(f64, 0.0)) + std.math.log1p(std.math.exp(-@abs(@as(f64, -5.0))))), result_sales_neg_mish[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -3.0) * @as(f64, 0.5) * (@as(f64, 1.0) + std.math.tanh(@sqrt(@as(f64, 2.0) / std.math.pi) * (@as(f64, -3.0) + @as(f64, 0.044715) * @as(f64, -27.0)))), result_sales_neg_gelu[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -5.0) * @as(f64, 0.5) * (@as(f64, 1.0) + std.math.tanh(@sqrt(@as(f64, 2.0) / std.math.pi) * (@as(f64, -5.0) + @as(f64, 0.044715) * @as(f64, -125.0)))), result_sales_neg_gelu[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0507009873554805) * @as(f64, 1.6732632423543772) * std.math.expm1(@as(f64, -3.0)), result_sales_neg_selu[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0507009873554805) * @as(f64, 1.6732632423543772) * std.math.expm1(@as(f64, -5.0)), result_sales_neg_selu[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.exp(@as(f64, 3.0)), result_sales_exp[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.exp(@as(f64, 5.0)), result_sales_exp[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.exp2(@as(f64, 3.0)), result_sales_exp2[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.exp2(@as(f64, 5.0)), result_sales_exp2[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.expm1(@as(f64, 3.0)), result_sales_expm1[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.expm1(@as(f64, 5.0)), result_sales_expm1[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sin(@as(f64, 3.0)), result_sales_sin[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sin(@as(f64, 5.0)), result_sales_sin[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.cos(@as(f64, 3.0)), result_sales_cos[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.cos(@as(f64, 5.0)), result_sales_cos[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.tan(@as(f64, 3.0)), result_sales_tan[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.tan(@as(f64, 5.0)), result_sales_tan[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.asin(@as(f64, 0.25)), result_ratio_asin[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.asin(@as(f64, 0.5)), result_ratio_asin[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.acos(@as(f64, 0.25)), result_ratio_acos[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.acos(@as(f64, 0.5)), result_ratio_acos[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.atan(@as(f64, 0.25)), result_ratio_atan[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.atan(@as(f64, 0.5)), result_ratio_atan[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sinh(@as(f64, 3.0)), result_sales_sinh[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sinh(@as(f64, 5.0)), result_sales_sinh[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.cosh(@as(f64, 3.0)), result_sales_cosh[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.cosh(@as(f64, 5.0)), result_sales_cosh[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.tanh(@as(f64, 3.0)), result_sales_tanh[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.tanh(@as(f64, 5.0)), result_sales_tanh[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.asinh(@as(f64, 3.0)), result_sales_asinh[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.asinh(@as(f64, 5.0)), result_sales_asinh[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.acosh(@as(f64, 3.0)), result_sales_acosh[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.acosh(@as(f64, 5.0)), result_sales_acosh[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.atanh(@as(f64, 0.25)), result_ratio_atanh[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.atanh(@as(f64, 0.5)), result_ratio_atanh[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log(f64, std.math.e, @as(f64, 3.0)), result_sales_log[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log(f64, std.math.e, @as(f64, 5.0)), result_sales_log[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log1p(@as(f64, 3.0)), result_sales_log1p[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log1p(@as(f64, 5.0)), result_sales_log1p[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.lgamma(f64, @as(f64, 3.0)), result_sales_lgamma[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.lgamma(f64, @as(f64, 5.0)), result_sales_lgamma[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sin(std.math.pi * @as(f64, 3.0)) / (std.math.pi * @as(f64, 3.0)), result_sales_sinc[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sin(std.math.pi * @as(f64, 5.0)) / (std.math.pi * @as(f64, 5.0)), result_sales_sinc[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log2(@as(f64, 3.0)), result_sales_log2[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log2(@as(f64, 5.0)), result_sales_log2[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log10(@as(f64, 3.0)), result_sales_log10[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log10(@as(f64, 5.0)), result_sales_log10[1], 1e-12);
}

test "device lazy frame pushes null predicate dependencies into parquet scan source" {
    const gpa = std.testing.allocator;

    var id = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 2, 3 }, .cpu);
    defer id.deinit();
    var sales = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ std.math.inf(f64), -std.math.inf(f64), 5.0 }, &.{ true, false, true }, .cpu);
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

    var zero_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer zero_scan.deinit();
    try zero_scan.isZeroColumn("active", "active_is_zero");
    try zero_scan.isPositiveZeroColumn("sales", "sales_is_positive_zero");
    try zero_scan.isNegativeZeroColumn("sales", "sales_is_negative_zero");
    try zero_scan.isNonZeroColumn("active", "active_is_non_zero");
    try zero_scan.select(&.{ "active_is_zero", "sales_is_positive_zero", "sales_is_negative_zero", "active_is_non_zero" });

    const zero_explain = try zero_scan.explain(gpa);
    defer gpa.free(zero_explain);
    try std.testing.expect(std.mem.indexOf(u8, zero_explain, "scan_pushdown: projection=[active,sales]") != null);
    try std.testing.expect(std.mem.indexOf(u8, zero_explain, "is_zero_column(active->active_is_zero)") != null);
    try std.testing.expect(std.mem.indexOf(u8, zero_explain, "is_positive_zero_column(sales->sales_is_positive_zero)") != null);
    try std.testing.expect(std.mem.indexOf(u8, zero_explain, "is_negative_zero_column(sales->sales_is_negative_zero)") != null);
    try std.testing.expect(std.mem.indexOf(u8, zero_explain, "is_non_zero_column(active->active_is_non_zero)") != null);

    var zero_result = try zero_scan.collect();
    defer zero_result.deinit();
    const active_is_zero = try (try zero_result.column("active_is_zero")).bool.toOwnedSlice(gpa);
    defer gpa.free(active_is_zero);
    const sales_is_positive_zero = try (try zero_result.column("sales_is_positive_zero")).bool.toOwnedSlice(gpa);
    defer gpa.free(sales_is_positive_zero);
    const sales_is_negative_zero = try (try zero_result.column("sales_is_negative_zero")).bool.toOwnedSlice(gpa);
    defer gpa.free(sales_is_negative_zero);
    const active_is_non_zero = try (try zero_result.column("active_is_non_zero")).bool.toOwnedSlice(gpa);
    defer gpa.free(active_is_non_zero);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false }, active_is_zero);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false }, sales_is_positive_zero);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false }, sales_is_negative_zero);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, active_is_non_zero);

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

    var sign_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer sign_scan.deinit();
    try sign_scan.isPositiveColumn("sales", "sales_is_positive");
    try sign_scan.isNegativeColumn("sales", "sales_is_negative");
    try sign_scan.isSignBitColumn("sales", "sales_signbit");
    try sign_scan.select(&.{ "sales_is_positive", "sales_is_negative", "sales_signbit" });

    const sign_explain = try sign_scan.explain(gpa);
    defer gpa.free(sign_explain);
    try std.testing.expect(std.mem.indexOf(u8, sign_explain, "scan_pushdown: projection=[sales]") != null);
    try std.testing.expect(std.mem.indexOf(u8, sign_explain, "is_positive_column(sales->sales_is_positive)") != null);
    try std.testing.expect(std.mem.indexOf(u8, sign_explain, "is_negative_column(sales->sales_is_negative)") != null);
    try std.testing.expect(std.mem.indexOf(u8, sign_explain, "is_signbit_column(sales->sales_signbit)") != null);

    var sign_result = try sign_scan.collect();
    defer sign_result.deinit();
    const sales_is_positive = try (try sign_result.column("sales_is_positive")).bool.toOwnedSlice(gpa);
    defer gpa.free(sales_is_positive);
    const sales_is_negative = try (try sign_result.column("sales_is_negative")).bool.toOwnedSlice(gpa);
    defer gpa.free(sales_is_negative);
    const sales_signbit = try (try sign_result.column("sales_signbit")).bool.toOwnedSlice(gpa);
    defer gpa.free(sales_signbit);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, sales_is_positive);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false }, sales_is_negative);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false }, sales_signbit);

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
    try std.testing.expectEqualSlices(bool, &.{ false, false, true }, sales_is_finite);

    var non_finite_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer non_finite_scan.deinit();
    try non_finite_scan.isNonFiniteColumn("sales", "sales_is_non_finite");
    try non_finite_scan.select(&.{"sales_is_non_finite"});

    const non_finite_explain = try non_finite_scan.explain(gpa);
    defer gpa.free(non_finite_explain);
    try std.testing.expect(std.mem.indexOf(u8, non_finite_explain, "scan_pushdown: projection=[sales]") != null);
    try std.testing.expect(std.mem.indexOf(u8, non_finite_explain, "is_non_finite_column(sales->sales_is_non_finite)") != null);

    var non_finite_result = try non_finite_scan.collect();
    defer non_finite_result.deinit();
    const sales_is_non_finite = try (try non_finite_result.column("sales_is_non_finite")).bool.toOwnedSlice(gpa);
    defer gpa.free(sales_is_non_finite);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false }, sales_is_non_finite);

    var normal_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer normal_scan.deinit();
    try normal_scan.isNormalColumn("sales", "sales_is_normal");
    try normal_scan.select(&.{"sales_is_normal"});

    const normal_explain = try normal_scan.explain(gpa);
    defer gpa.free(normal_explain);
    try std.testing.expect(std.mem.indexOf(u8, normal_explain, "scan_pushdown: projection=[sales]") != null);
    try std.testing.expect(std.mem.indexOf(u8, normal_explain, "is_normal_column(sales->sales_is_normal)") != null);

    var normal_result = try normal_scan.collect();
    defer normal_result.deinit();
    const sales_is_normal = try (try normal_result.column("sales_is_normal")).bool.toOwnedSlice(gpa);
    defer gpa.free(sales_is_normal);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true }, sales_is_normal);

    var subnormal_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer subnormal_scan.deinit();
    try subnormal_scan.isSubnormalColumn("sales", "sales_is_subnormal");
    try subnormal_scan.select(&.{"sales_is_subnormal"});

    const subnormal_explain = try subnormal_scan.explain(gpa);
    defer gpa.free(subnormal_explain);
    try std.testing.expect(std.mem.indexOf(u8, subnormal_explain, "scan_pushdown: projection=[sales]") != null);
    try std.testing.expect(std.mem.indexOf(u8, subnormal_explain, "is_subnormal_column(sales->sales_is_subnormal)") != null);

    var subnormal_result = try subnormal_scan.collect();
    defer subnormal_result.deinit();
    const sales_is_subnormal = try (try subnormal_result.column("sales_is_subnormal")).bool.toOwnedSlice(gpa);
    defer gpa.free(sales_is_subnormal);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false }, sales_is_subnormal);

    var inf_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer inf_scan.deinit();
    try inf_scan.isInfColumn("sales", "sales_is_inf");
    try inf_scan.select(&.{"sales_is_inf"});

    const inf_explain = try inf_scan.explain(gpa);
    defer gpa.free(inf_explain);
    try std.testing.expect(std.mem.indexOf(u8, inf_explain, "scan_pushdown: projection=[sales]") != null);
    try std.testing.expect(std.mem.indexOf(u8, inf_explain, "is_inf_column(sales->sales_is_inf)") != null);

    var inf_result = try inf_scan.collect();
    defer inf_result.deinit();
    const sales_is_inf = try (try inf_result.column("sales_is_inf")).bool.toOwnedSlice(gpa);
    defer gpa.free(sales_is_inf);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false }, sales_is_inf);

    var positive_inf_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer positive_inf_scan.deinit();
    try positive_inf_scan.isPositiveInfColumn("sales", "sales_is_pos_inf");
    try positive_inf_scan.select(&.{"sales_is_pos_inf"});

    const positive_inf_explain = try positive_inf_scan.explain(gpa);
    defer gpa.free(positive_inf_explain);
    try std.testing.expect(std.mem.indexOf(u8, positive_inf_explain, "scan_pushdown: projection=[sales]") != null);
    try std.testing.expect(std.mem.indexOf(u8, positive_inf_explain, "is_positive_inf_column(sales->sales_is_pos_inf)") != null);

    var positive_inf_result = try positive_inf_scan.collect();
    defer positive_inf_result.deinit();
    const sales_is_pos_inf = try (try positive_inf_result.column("sales_is_pos_inf")).bool.toOwnedSlice(gpa);
    defer gpa.free(sales_is_pos_inf);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false }, sales_is_pos_inf);

    var negative_inf_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer negative_inf_scan.deinit();
    try negative_inf_scan.isNegativeInfColumn("sales", "sales_is_neg_inf");
    try negative_inf_scan.select(&.{"sales_is_neg_inf"});

    const negative_inf_explain = try negative_inf_scan.explain(gpa);
    defer gpa.free(negative_inf_explain);
    try std.testing.expect(std.mem.indexOf(u8, negative_inf_explain, "scan_pushdown: projection=[sales]") != null);
    try std.testing.expect(std.mem.indexOf(u8, negative_inf_explain, "is_negative_inf_column(sales->sales_is_neg_inf)") != null);

    var negative_inf_result = try negative_inf_scan.collect();
    defer negative_inf_result.deinit();
    const sales_is_neg_inf = try (try negative_inf_result.column("sales_is_neg_inf")).bool.toOwnedSlice(gpa);
    defer gpa.free(sales_is_neg_inf);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false }, sales_is_neg_inf);

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

    var fill_positive_inf_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer fill_positive_inf_scan.deinit();
    try fill_positive_inf_scan.fillPositiveInfColumn("sales", f64, 100.0);
    try fill_positive_inf_scan.select(&.{"sales"});

    const fill_positive_inf_explain = try fill_positive_inf_scan.explain(gpa);
    defer gpa.free(fill_positive_inf_explain);
    try std.testing.expect(std.mem.indexOf(u8, fill_positive_inf_explain, "scan_pushdown: projection=[sales]") != null);
    try std.testing.expect(std.mem.indexOf(u8, fill_positive_inf_explain, "fill_positive_inf_column(sales=scalar:f64)") != null);

    var fill_negative_inf_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer fill_negative_inf_scan.deinit();
    try fill_negative_inf_scan.fillNegativeInfColumn("sales", f64, -100.0);
    try fill_negative_inf_scan.select(&.{"sales"});

    const fill_negative_inf_explain = try fill_negative_inf_scan.explain(gpa);
    defer gpa.free(fill_negative_inf_explain);
    try std.testing.expect(std.mem.indexOf(u8, fill_negative_inf_explain, "scan_pushdown: projection=[sales]") != null);
    try std.testing.expect(std.mem.indexOf(u8, fill_negative_inf_explain, "fill_negative_inf_column(sales=scalar:f64)") != null);

    var fill_zero_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer fill_zero_scan.deinit();
    try fill_zero_scan.fillZeroColumn("sales", f64, 42.0);
    try fill_zero_scan.select(&.{"sales"});

    const fill_zero_explain = try fill_zero_scan.explain(gpa);
    defer gpa.free(fill_zero_explain);
    try std.testing.expect(std.mem.indexOf(u8, fill_zero_explain, "scan_pushdown: projection=[sales]") != null);
    try std.testing.expect(std.mem.indexOf(u8, fill_zero_explain, "fill_zero_column(sales=scalar:f64)") != null);

    var fill_positive_zero_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer fill_positive_zero_scan.deinit();
    try fill_positive_zero_scan.fillPositiveZeroColumn("sales", f64, 42.0);
    try fill_positive_zero_scan.select(&.{"sales"});

    const fill_positive_zero_explain = try fill_positive_zero_scan.explain(gpa);
    defer gpa.free(fill_positive_zero_explain);
    try std.testing.expect(std.mem.indexOf(u8, fill_positive_zero_explain, "scan_pushdown: projection=[sales]") != null);
    try std.testing.expect(std.mem.indexOf(u8, fill_positive_zero_explain, "fill_positive_zero_column(sales=scalar:f64)") != null);

    var fill_signbit_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer fill_signbit_scan.deinit();
    try fill_signbit_scan.fillSignBitColumn("sales", f64, -42.0);
    try fill_signbit_scan.select(&.{"sales"});

    const fill_signbit_explain = try fill_signbit_scan.explain(gpa);
    defer gpa.free(fill_signbit_explain);
    try std.testing.expect(std.mem.indexOf(u8, fill_signbit_explain, "scan_pushdown: projection=[sales]") != null);
    try std.testing.expect(std.mem.indexOf(u8, fill_signbit_explain, "fill_signbit_column(sales=scalar:f64)") != null);

    var fill_positive_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer fill_positive_scan.deinit();
    try fill_positive_scan.fillPositiveColumn("sales", f64, 42.0);
    try fill_positive_scan.select(&.{"sales"});

    const fill_positive_explain = try fill_positive_scan.explain(gpa);
    defer gpa.free(fill_positive_explain);
    try std.testing.expect(std.mem.indexOf(u8, fill_positive_explain, "scan_pushdown: projection=[sales]") != null);
    try std.testing.expect(std.mem.indexOf(u8, fill_positive_explain, "fill_positive_column(sales=scalar:f64)") != null);

    var fill_finite_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer fill_finite_scan.deinit();
    try fill_finite_scan.fillFiniteColumn("sales", f64, 42.0);
    try fill_finite_scan.select(&.{"sales"});

    const fill_finite_explain = try fill_finite_scan.explain(gpa);
    defer gpa.free(fill_finite_explain);
    try std.testing.expect(std.mem.indexOf(u8, fill_finite_explain, "scan_pushdown: projection=[sales]") != null);
    try std.testing.expect(std.mem.indexOf(u8, fill_finite_explain, "fill_finite_column(sales=scalar:f64)") != null);

    var fill_normal_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer fill_normal_scan.deinit();
    try fill_normal_scan.fillNormalColumn("sales", f64, 42.0);
    try fill_normal_scan.select(&.{"sales"});

    const fill_normal_explain = try fill_normal_scan.explain(gpa);
    defer gpa.free(fill_normal_explain);
    try std.testing.expect(std.mem.indexOf(u8, fill_normal_explain, "scan_pushdown: projection=[sales]") != null);
    try std.testing.expect(std.mem.indexOf(u8, fill_normal_explain, "fill_normal_column(sales=scalar:f64)") != null);

    var fill_subnormal_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer fill_subnormal_scan.deinit();
    try fill_subnormal_scan.fillSubnormalColumn("sales", f64, 42.0);
    try fill_subnormal_scan.select(&.{"sales"});

    const fill_subnormal_explain = try fill_subnormal_scan.explain(gpa);
    defer gpa.free(fill_subnormal_explain);
    try std.testing.expect(std.mem.indexOf(u8, fill_subnormal_explain, "scan_pushdown: projection=[sales]") != null);
    try std.testing.expect(std.mem.indexOf(u8, fill_subnormal_explain, "fill_subnormal_column(sales=scalar:f64)") != null);

    var fill_non_finite_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer fill_non_finite_scan.deinit();
    try fill_non_finite_scan.fillNonFiniteColumn("sales", f64, -5.0);
    try fill_non_finite_scan.select(&.{"sales"});

    const fill_non_finite_explain = try fill_non_finite_scan.explain(gpa);
    defer gpa.free(fill_non_finite_explain);
    try std.testing.expect(std.mem.indexOf(u8, fill_non_finite_explain, "scan_pushdown: projection=[sales]") != null);
    try std.testing.expect(std.mem.indexOf(u8, fill_non_finite_explain, "fill_non_finite_column(sales=scalar:f64)") != null);

    var drop_nulls_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer drop_nulls_scan.deinit();
    try drop_nulls_scan.dropNullsColumn("sales");
    try drop_nulls_scan.select(&.{"id"});

    const drop_nulls_explain = try drop_nulls_scan.explain(gpa);
    defer gpa.free(drop_nulls_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_nulls_explain, "scan_pushdown: null=sales:non_null, projection=[sales,id]") != null);
    try std.testing.expect(std.mem.indexOf(u8, drop_nulls_explain, "drop_nulls[sales]") != null);

    var drop_nulls = try drop_nulls_scan.collect();
    defer drop_nulls.deinit();
    const drop_null_ids = try (try drop_nulls.column("id")).i32.toOwnedSlice(gpa);
    defer gpa.free(drop_null_ids);
    try std.testing.expectEqualSlices(i32, &.{ 1, 3 }, drop_null_ids);

    var filter_nulls_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer filter_nulls_scan.deinit();
    try filter_nulls_scan.filterNullsColumn("sales");
    try filter_nulls_scan.select(&.{"id"});

    const filter_nulls_explain = try filter_nulls_scan.explain(gpa);
    defer gpa.free(filter_nulls_explain);
    try std.testing.expect(std.mem.indexOf(u8, filter_nulls_explain, "scan_pushdown: null=sales:only, projection=[sales,id]") != null);
    try std.testing.expect(std.mem.indexOf(u8, filter_nulls_explain, "filter_nulls_column(sales)") != null);

    var filter_nulls = try filter_nulls_scan.collect();
    defer filter_nulls.deinit();
    const filter_null_ids = try (try filter_nulls.column("id")).i32.toOwnedSlice(gpa);
    defer gpa.free(filter_null_ids);
    try std.testing.expectEqualSlices(i32, &.{2}, filter_null_ids);

    var drop_all_nulls_single_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer drop_all_nulls_single_scan.deinit();
    try drop_all_nulls_single_scan.dropAllNulls(&.{"sales"});
    try drop_all_nulls_single_scan.select(&.{"id"});

    const drop_all_nulls_single_explain = try drop_all_nulls_single_scan.explain(gpa);
    defer gpa.free(drop_all_nulls_single_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_all_nulls_single_explain, "scan_pushdown: null=sales:non_null, projection=[sales,id]") != null);
    try std.testing.expect(std.mem.indexOf(u8, drop_all_nulls_single_explain, "drop_all_nulls[sales]") != null);

    var filter_all_nulls_single_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer filter_all_nulls_single_scan.deinit();
    try filter_all_nulls_single_scan.filterAllNulls(&.{"sales"});
    try filter_all_nulls_single_scan.select(&.{"id"});

    const filter_all_nulls_single_explain = try filter_all_nulls_single_scan.explain(gpa);
    defer gpa.free(filter_all_nulls_single_explain);
    try std.testing.expect(std.mem.indexOf(u8, filter_all_nulls_single_explain, "scan_pushdown: null=sales:only, projection=[sales,id]") != null);
    try std.testing.expect(std.mem.indexOf(u8, filter_all_nulls_single_explain, "filter_all_nulls[sales]") != null);

    var range_then_null_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer range_then_null_scan.deinit();
    try range_then_null_scan.filterColumnScalar("id", i32, 1, .gt);
    try range_then_null_scan.dropNullsColumn("sales");
    try range_then_null_scan.select(&.{"id"});

    const range_then_null_explain = try range_then_null_scan.explain(gpa);
    defer gpa.free(range_then_null_explain);
    try std.testing.expect(std.mem.indexOf(u8, range_then_null_explain, "scan_pushdown: range=id, projection=[id,sales]") != null);
    try std.testing.expect(std.mem.indexOf(u8, range_then_null_explain, "null=sales") == null);

    var null_then_range_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer null_then_range_scan.deinit();
    try null_then_range_scan.dropNullsColumn("sales");
    try null_then_range_scan.filterColumnScalar("id", i32, 1, .gt);
    try null_then_range_scan.select(&.{"id"});

    const null_then_range_explain = try null_then_range_scan.explain(gpa);
    defer gpa.free(null_then_range_explain);
    try std.testing.expect(std.mem.indexOf(u8, null_then_range_explain, "scan_pushdown: range=id, projection=[sales,id]") != null);
    try std.testing.expect(std.mem.indexOf(u8, null_then_range_explain, "null=sales") == null);

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

    var drop_positive_inf_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer drop_positive_inf_scan.deinit();
    try drop_positive_inf_scan.dropPositiveInfsColumn("sales");
    try drop_positive_inf_scan.select(&.{"id"});

    const drop_positive_inf_explain = try drop_positive_inf_scan.explain(gpa);
    defer gpa.free(drop_positive_inf_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_positive_inf_explain, "scan_pushdown: projection=[sales,id]") != null);
    try std.testing.expect(std.mem.indexOf(u8, drop_positive_inf_explain, "drop_positive_infs[sales]") != null);

    var filter_positive_inf_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer filter_positive_inf_scan.deinit();
    try filter_positive_inf_scan.filterPositiveInfsColumn("sales");

    const filter_positive_inf_explain = try filter_positive_inf_scan.explain(gpa);
    defer gpa.free(filter_positive_inf_explain);
    try std.testing.expect(std.mem.indexOf(u8, filter_positive_inf_explain, "scan_pushdown: none") != null);
    try std.testing.expect(std.mem.indexOf(u8, filter_positive_inf_explain, "filter_positive_infs_column(sales)") != null);

    var drop_negative_inf_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer drop_negative_inf_scan.deinit();
    try drop_negative_inf_scan.dropNegativeInfsColumn("sales");
    try drop_negative_inf_scan.select(&.{"id"});

    const drop_negative_inf_explain = try drop_negative_inf_scan.explain(gpa);
    defer gpa.free(drop_negative_inf_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_negative_inf_explain, "scan_pushdown: projection=[sales,id]") != null);
    try std.testing.expect(std.mem.indexOf(u8, drop_negative_inf_explain, "drop_negative_infs[sales]") != null);

    var filter_negative_inf_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer filter_negative_inf_scan.deinit();
    try filter_negative_inf_scan.filterNegativeInfsColumn("sales");

    const filter_negative_inf_explain = try filter_negative_inf_scan.explain(gpa);
    defer gpa.free(filter_negative_inf_explain);
    try std.testing.expect(std.mem.indexOf(u8, filter_negative_inf_explain, "scan_pushdown: none") != null);
    try std.testing.expect(std.mem.indexOf(u8, filter_negative_inf_explain, "filter_negative_infs_column(sales)") != null);

    var drop_positive_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer drop_positive_scan.deinit();
    try drop_positive_scan.dropPositivesColumn("sales");
    try drop_positive_scan.select(&.{"id"});

    const drop_positive_explain = try drop_positive_scan.explain(gpa);
    defer gpa.free(drop_positive_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_positive_explain, "scan_pushdown: projection=[sales,id]") != null);
    try std.testing.expect(std.mem.indexOf(u8, drop_positive_explain, "drop_positives[sales]") != null);

    var filter_positive_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer filter_positive_scan.deinit();
    try filter_positive_scan.filterPositivesColumn("sales");

    const filter_positive_explain = try filter_positive_scan.explain(gpa);
    defer gpa.free(filter_positive_explain);
    try std.testing.expect(std.mem.indexOf(u8, filter_positive_explain, "scan_pushdown: none") != null);
    try std.testing.expect(std.mem.indexOf(u8, filter_positive_explain, "filter_positives_column(sales)") != null);

    var filter_signbit_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer filter_signbit_scan.deinit();
    try filter_signbit_scan.filterSignBitsColumn("sales");

    const filter_signbit_explain = try filter_signbit_scan.explain(gpa);
    defer gpa.free(filter_signbit_explain);
    try std.testing.expect(std.mem.indexOf(u8, filter_signbit_explain, "scan_pushdown: none") != null);
    try std.testing.expect(std.mem.indexOf(u8, filter_signbit_explain, "filter_signbits_column(sales)") != null);

    var filter_negative_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer filter_negative_scan.deinit();
    try filter_negative_scan.filterNegativesColumn("sales");

    const filter_negative_explain = try filter_negative_scan.explain(gpa);
    defer gpa.free(filter_negative_explain);
    try std.testing.expect(std.mem.indexOf(u8, filter_negative_explain, "scan_pushdown: none") != null);
    try std.testing.expect(std.mem.indexOf(u8, filter_negative_explain, "filter_negatives_column(sales)") != null);

    var drop_zero_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer drop_zero_scan.deinit();
    try drop_zero_scan.dropZerosColumn("active");
    try drop_zero_scan.select(&.{"id"});

    const drop_zero_explain = try drop_zero_scan.explain(gpa);
    defer gpa.free(drop_zero_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_zero_explain, "scan_pushdown: projection=[active,id]") != null);
    try std.testing.expect(std.mem.indexOf(u8, drop_zero_explain, "drop_zeros[active]") != null);

    var filter_zero_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer filter_zero_scan.deinit();
    try filter_zero_scan.filterZerosColumn("active");

    const filter_zero_explain = try filter_zero_scan.explain(gpa);
    defer gpa.free(filter_zero_explain);
    try std.testing.expect(std.mem.indexOf(u8, filter_zero_explain, "scan_pushdown: none") != null);
    try std.testing.expect(std.mem.indexOf(u8, filter_zero_explain, "filter_zeros_column(active)") != null);

    var drop_positive_zero_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer drop_positive_zero_scan.deinit();
    try drop_positive_zero_scan.dropPositiveZerosColumn("sales");
    try drop_positive_zero_scan.select(&.{"id"});

    const drop_positive_zero_explain = try drop_positive_zero_scan.explain(gpa);
    defer gpa.free(drop_positive_zero_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_positive_zero_explain, "scan_pushdown: projection=[sales,id]") != null);
    try std.testing.expect(std.mem.indexOf(u8, drop_positive_zero_explain, "drop_positive_zeros[sales]") != null);

    var filter_negative_zero_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer filter_negative_zero_scan.deinit();
    try filter_negative_zero_scan.filterNegativeZerosColumn("sales");

    const filter_negative_zero_explain = try filter_negative_zero_scan.explain(gpa);
    defer gpa.free(filter_negative_zero_explain);
    try std.testing.expect(std.mem.indexOf(u8, filter_negative_zero_explain, "scan_pushdown: none") != null);
    try std.testing.expect(std.mem.indexOf(u8, filter_negative_zero_explain, "filter_negative_zeros_column(sales)") != null);

    var drop_finite_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer drop_finite_scan.deinit();
    try drop_finite_scan.dropFinitesColumn("sales");
    try drop_finite_scan.select(&.{"id"});

    const drop_finite_explain = try drop_finite_scan.explain(gpa);
    defer gpa.free(drop_finite_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_finite_explain, "scan_pushdown: projection=[sales,id]") != null);
    try std.testing.expect(std.mem.indexOf(u8, drop_finite_explain, "drop_finites[sales]") != null);

    var filter_finite_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer filter_finite_scan.deinit();
    try filter_finite_scan.filterFinitesColumn("sales");

    const filter_finite_explain = try filter_finite_scan.explain(gpa);
    defer gpa.free(filter_finite_explain);
    try std.testing.expect(std.mem.indexOf(u8, filter_finite_explain, "scan_pushdown: none") != null);
    try std.testing.expect(std.mem.indexOf(u8, filter_finite_explain, "filter_finites_column(sales)") != null);

    var drop_normal_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer drop_normal_scan.deinit();
    try drop_normal_scan.dropNormalsColumn("sales");
    try drop_normal_scan.select(&.{"id"});

    const drop_normal_explain = try drop_normal_scan.explain(gpa);
    defer gpa.free(drop_normal_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_normal_explain, "scan_pushdown: projection=[sales,id]") != null);
    try std.testing.expect(std.mem.indexOf(u8, drop_normal_explain, "drop_normals[sales]") != null);

    var filter_normal_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer filter_normal_scan.deinit();
    try filter_normal_scan.filterNormalsColumn("sales");

    const filter_normal_explain = try filter_normal_scan.explain(gpa);
    defer gpa.free(filter_normal_explain);
    try std.testing.expect(std.mem.indexOf(u8, filter_normal_explain, "scan_pushdown: none") != null);
    try std.testing.expect(std.mem.indexOf(u8, filter_normal_explain, "filter_normals_column(sales)") != null);

    var drop_subnormal_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer drop_subnormal_scan.deinit();
    try drop_subnormal_scan.dropSubnormalsColumn("sales");
    try drop_subnormal_scan.select(&.{"id"});

    const drop_subnormal_explain = try drop_subnormal_scan.explain(gpa);
    defer gpa.free(drop_subnormal_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_subnormal_explain, "scan_pushdown: projection=[sales,id]") != null);
    try std.testing.expect(std.mem.indexOf(u8, drop_subnormal_explain, "drop_subnormals[sales]") != null);

    var filter_subnormal_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer filter_subnormal_scan.deinit();
    try filter_subnormal_scan.filterSubnormalsColumn("sales");

    const filter_subnormal_explain = try filter_subnormal_scan.explain(gpa);
    defer gpa.free(filter_subnormal_explain);
    try std.testing.expect(std.mem.indexOf(u8, filter_subnormal_explain, "scan_pushdown: none") != null);
    try std.testing.expect(std.mem.indexOf(u8, filter_subnormal_explain, "filter_subnormals_column(sales)") != null);

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

    var row_positive_inf_count_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer row_positive_inf_count_scan.deinit();
    try row_positive_inf_count_scan.withRowPositiveInfCount(&.{ "sales", "active" }, "row_positive_inf_count");
    try row_positive_inf_count_scan.select(&.{"row_positive_inf_count"});

    const row_positive_inf_count_explain = try row_positive_inf_count_scan.explain(gpa);
    defer gpa.free(row_positive_inf_count_explain);
    try std.testing.expect(std.mem.indexOf(u8, row_positive_inf_count_explain, "scan_pushdown: projection=[sales,active]") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_positive_inf_count_explain, "row_positive_inf_count([sales,active]->row_positive_inf_count)") != null);

    var row_negative_inf_count_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer row_negative_inf_count_scan.deinit();
    try row_negative_inf_count_scan.withRowNegativeInfCount(&.{ "sales", "active" }, "row_negative_inf_count");
    try row_negative_inf_count_scan.select(&.{"row_negative_inf_count"});

    const row_negative_inf_count_explain = try row_negative_inf_count_scan.explain(gpa);
    defer gpa.free(row_negative_inf_count_explain);
    try std.testing.expect(std.mem.indexOf(u8, row_negative_inf_count_explain, "scan_pushdown: projection=[sales,active]") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_negative_inf_count_explain, "row_negative_inf_count([sales,active]->row_negative_inf_count)") != null);

    var row_positive_count_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer row_positive_count_scan.deinit();
    try row_positive_count_scan.withRowPositiveCount(&.{ "sales", "active" }, "row_positive_count");
    try row_positive_count_scan.select(&.{"row_positive_count"});

    const row_positive_count_explain = try row_positive_count_scan.explain(gpa);
    defer gpa.free(row_positive_count_explain);
    try std.testing.expect(std.mem.indexOf(u8, row_positive_count_explain, "scan_pushdown: projection=[sales,active]") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_positive_count_explain, "row_positive_count([sales,active]->row_positive_count)") != null);
    var row_positive_count_result = try row_positive_count_scan.collect();
    defer row_positive_count_result.deinit();
    const row_positive_count = try (try row_positive_count_result.column("row_positive_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_positive_count);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 1 }, row_positive_count);

    var row_signbit_count_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer row_signbit_count_scan.deinit();
    try row_signbit_count_scan.withRowSignBitCount(&.{ "sales", "active" }, "row_signbit_count");
    try row_signbit_count_scan.select(&.{"row_signbit_count"});

    const row_signbit_count_explain = try row_signbit_count_scan.explain(gpa);
    defer gpa.free(row_signbit_count_explain);
    try std.testing.expect(std.mem.indexOf(u8, row_signbit_count_explain, "scan_pushdown: projection=[sales,active]") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_signbit_count_explain, "row_signbit_count([sales,active]->row_signbit_count)") != null);
    var row_signbit_count_result = try row_signbit_count_scan.collect();
    defer row_signbit_count_result.deinit();
    const row_signbit_count = try (try row_signbit_count_result.column("row_signbit_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_signbit_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0 }, row_signbit_count);

    var row_zero_count_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer row_zero_count_scan.deinit();
    try row_zero_count_scan.withRowZeroCount(&.{ "sales", "active" }, "row_zero_count");
    try row_zero_count_scan.select(&.{"row_zero_count"});

    const row_zero_count_explain = try row_zero_count_scan.explain(gpa);
    defer gpa.free(row_zero_count_explain);
    try std.testing.expect(std.mem.indexOf(u8, row_zero_count_explain, "scan_pushdown: projection=[sales,active]") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_zero_count_explain, "row_zero_count([sales,active]->row_zero_count)") != null);
    var row_zero_count_result = try row_zero_count_scan.collect();
    defer row_zero_count_result.deinit();
    const row_zero_count = try (try row_zero_count_result.column("row_zero_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_zero_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0 }, row_zero_count);

    var row_positive_zero_count_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer row_positive_zero_count_scan.deinit();
    try row_positive_zero_count_scan.withRowPositiveZeroCount(&.{ "sales", "active" }, "row_positive_zero_count");
    try row_positive_zero_count_scan.select(&.{"row_positive_zero_count"});

    const row_positive_zero_count_explain = try row_positive_zero_count_scan.explain(gpa);
    defer gpa.free(row_positive_zero_count_explain);
    try std.testing.expect(std.mem.indexOf(u8, row_positive_zero_count_explain, "scan_pushdown: projection=[sales,active]") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_positive_zero_count_explain, "row_positive_zero_count([sales,active]->row_positive_zero_count)") != null);
    var row_positive_zero_count_result = try row_positive_zero_count_scan.collect();
    defer row_positive_zero_count_result.deinit();
    const row_positive_zero_count = try (try row_positive_zero_count_result.column("row_positive_zero_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_positive_zero_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0 }, row_positive_zero_count);

    var row_finite_count_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer row_finite_count_scan.deinit();
    try row_finite_count_scan.withRowFiniteCount(&.{ "sales", "active" }, "row_finite_count");
    try row_finite_count_scan.select(&.{"row_finite_count"});

    const row_finite_count_explain = try row_finite_count_scan.explain(gpa);
    defer gpa.free(row_finite_count_explain);
    try std.testing.expect(std.mem.indexOf(u8, row_finite_count_explain, "scan_pushdown: projection=[sales,active]") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_finite_count_explain, "row_finite_count([sales,active]->row_finite_count)") != null);

    var row_normal_count_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer row_normal_count_scan.deinit();
    try row_normal_count_scan.withRowNormalCount(&.{ "sales", "active" }, "row_normal_count");
    try row_normal_count_scan.select(&.{"row_normal_count"});

    const row_normal_count_explain = try row_normal_count_scan.explain(gpa);
    defer gpa.free(row_normal_count_explain);
    try std.testing.expect(std.mem.indexOf(u8, row_normal_count_explain, "scan_pushdown: projection=[sales,active]") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_normal_count_explain, "row_normal_count([sales,active]->row_normal_count)") != null);
    var row_normal_count_result = try row_normal_count_scan.collect();
    defer row_normal_count_result.deinit();
    const row_normal_count = try (try row_normal_count_result.column("row_normal_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_normal_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1 }, row_normal_count);

    var row_subnormal_count_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer row_subnormal_count_scan.deinit();
    try row_subnormal_count_scan.withRowSubnormalCount(&.{ "sales", "active" }, "row_subnormal_count");
    try row_subnormal_count_scan.select(&.{"row_subnormal_count"});

    const row_subnormal_count_explain = try row_subnormal_count_scan.explain(gpa);
    defer gpa.free(row_subnormal_count_explain);
    try std.testing.expect(std.mem.indexOf(u8, row_subnormal_count_explain, "scan_pushdown: projection=[sales,active]") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_subnormal_count_explain, "row_subnormal_count([sales,active]->row_subnormal_count)") != null);
    var row_subnormal_count_result = try row_subnormal_count_scan.collect();
    defer row_subnormal_count_result.deinit();
    const row_subnormal_count = try (try row_subnormal_count_result.column("row_subnormal_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_subnormal_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0 }, row_subnormal_count);

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

test "device lazy frame pushes put-flat value dependencies into parquet scan source" {
    const gpa = std.testing.allocator;

    var id = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 2, 3 }, .cpu);
    defer id.deinit();
    var sales = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 3.0, 5.0 }, .cpu);
    defer sales.deinit();
    var replacements = try DeviceColumn.fromSlice(f64, gpa, &.{ -2.0, -3.0, -5.0 }, .cpu);
    defer replacements.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "id", .data = id },
        .{ .name = "sales", .data = sales },
        .{ .name = "replacements", .data = replacements },
    });
    defer table.deinit();

    const bytes = try table.toParquetBytes(gpa);
    defer gpa.free(bytes);

    var lazy_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer lazy_scan.deinit();
    try lazy_scan.withColumnPutFlat("sales_put_values", "sales", &.{ 2, 0, 1 }, "replacements");
    try lazy_scan.select(&.{"sales_put_values"});

    const explain = try lazy_scan.explain(gpa);
    defer gpa.free(explain);
    try std.testing.expect(std.mem.indexOf(u8, explain, "scan_pushdown: projection=[sales,replacements]") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_put_flat(sales_put_values=put_flat(sales, indices=[2,0,1], values:replacements))") != null);

    var result = try lazy_scan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 1), result.width());
    const values = try (try result.column("sales_put_values")).f64.toOwnedSlice(gpa);
    defer gpa.free(values);
    try std.testing.expectEqualSlices(f64, &.{ -3.0, -5.0, -2.0 }, values);
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

    var group_rows_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer group_rows_scan.deinit();
    try group_rows_scan.groupByTopRows("active", "sales", 1, .{ .descending = true });
    const group_rows_explain = try group_rows_scan.explain(gpa);
    defer gpa.free(group_rows_explain);
    try std.testing.expect(std.mem.indexOf(u8, group_rows_explain, "scan_pushdown: none") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_rows_explain, "group_by_top_rows(active, sort=sales, n=1, desc=true)") != null);
    var group_rows = try group_rows_scan.collect();
    defer group_rows.deinit();
    try std.testing.expectEqual(@as(usize, 3), group_rows.width());
    try std.testing.expectEqual(@as(?usize, 0), group_rows.columnIndex("id"));
    try std.testing.expectEqual(@as(?usize, 1), group_rows.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 2), group_rows.columnIndex("active"));

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

    var lazy_nan_columns_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer lazy_nan_columns_scan.deinit();
    try lazy_nan_columns_scan.selectColumnsWithNaNs();

    const nan_columns_explain = try lazy_nan_columns_scan.explain(gpa);
    defer gpa.free(nan_columns_explain);
    try std.testing.expect(std.mem.indexOf(u8, nan_columns_explain, "scan_pushdown: none") != null);
    try std.testing.expect(std.mem.indexOf(u8, nan_columns_explain, "select_columns_with_nans") != null);

    var lazy_inf_columns_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer lazy_inf_columns_scan.deinit();
    try lazy_inf_columns_scan.selectColumnsWithInfs();

    const inf_columns_explain = try lazy_inf_columns_scan.explain(gpa);
    defer gpa.free(inf_columns_explain);
    try std.testing.expect(std.mem.indexOf(u8, inf_columns_explain, "scan_pushdown: none") != null);
    try std.testing.expect(std.mem.indexOf(u8, inf_columns_explain, "select_columns_with_infs") != null);

    var lazy_zero_columns_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer lazy_zero_columns_scan.deinit();
    try lazy_zero_columns_scan.selectColumnsWithZeros();

    const zero_columns_explain = try lazy_zero_columns_scan.explain(gpa);
    defer gpa.free(zero_columns_explain);
    try std.testing.expect(std.mem.indexOf(u8, zero_columns_explain, "scan_pushdown: none") != null);
    try std.testing.expect(std.mem.indexOf(u8, zero_columns_explain, "select_columns_with_zeros") != null);

    var lazy_positive_zero_columns_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer lazy_positive_zero_columns_scan.deinit();
    try lazy_positive_zero_columns_scan.selectColumnsWithPositiveZeros();

    const positive_zero_columns_explain = try lazy_positive_zero_columns_scan.explain(gpa);
    defer gpa.free(positive_zero_columns_explain);
    try std.testing.expect(std.mem.indexOf(u8, positive_zero_columns_explain, "scan_pushdown: none") != null);
    try std.testing.expect(std.mem.indexOf(u8, positive_zero_columns_explain, "select_columns_with_positive_zeros") != null);

    var lazy_positive_columns_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer lazy_positive_columns_scan.deinit();
    try lazy_positive_columns_scan.selectColumnsWithPositives();

    const positive_columns_explain = try lazy_positive_columns_scan.explain(gpa);
    defer gpa.free(positive_columns_explain);
    try std.testing.expect(std.mem.indexOf(u8, positive_columns_explain, "scan_pushdown: none") != null);
    try std.testing.expect(std.mem.indexOf(u8, positive_columns_explain, "select_columns_with_positives") != null);
    var positive_columns = try lazy_positive_columns_scan.collect();
    defer positive_columns.deinit();
    try std.testing.expectEqual(@as(usize, 2), positive_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), positive_columns.columnIndex("id"));
    try std.testing.expectEqual(@as(?usize, 1), positive_columns.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, null), positive_columns.columnIndex("active"));

    var lazy_signbit_columns_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer lazy_signbit_columns_scan.deinit();
    try lazy_signbit_columns_scan.selectColumnsWithSignBits();

    const signbit_columns_explain = try lazy_signbit_columns_scan.explain(gpa);
    defer gpa.free(signbit_columns_explain);
    try std.testing.expect(std.mem.indexOf(u8, signbit_columns_explain, "scan_pushdown: none") != null);
    try std.testing.expect(std.mem.indexOf(u8, signbit_columns_explain, "select_columns_with_signbits") != null);

    var lazy_finite_columns_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer lazy_finite_columns_scan.deinit();
    try lazy_finite_columns_scan.selectColumnsWithFinites();

    const finite_columns_explain = try lazy_finite_columns_scan.explain(gpa);
    defer gpa.free(finite_columns_explain);
    try std.testing.expect(std.mem.indexOf(u8, finite_columns_explain, "scan_pushdown: none") != null);
    try std.testing.expect(std.mem.indexOf(u8, finite_columns_explain, "select_columns_with_finites") != null);

    var lazy_normal_columns_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer lazy_normal_columns_scan.deinit();
    try lazy_normal_columns_scan.selectColumnsWithNormals();

    const normal_columns_explain = try lazy_normal_columns_scan.explain(gpa);
    defer gpa.free(normal_columns_explain);
    try std.testing.expect(std.mem.indexOf(u8, normal_columns_explain, "scan_pushdown: none") != null);
    try std.testing.expect(std.mem.indexOf(u8, normal_columns_explain, "select_columns_with_normals") != null);

    var lazy_subnormal_columns_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer lazy_subnormal_columns_scan.deinit();
    try lazy_subnormal_columns_scan.selectColumnsWithSubnormals();

    const subnormal_columns_explain = try lazy_subnormal_columns_scan.explain(gpa);
    defer gpa.free(subnormal_columns_explain);
    try std.testing.expect(std.mem.indexOf(u8, subnormal_columns_explain, "scan_pushdown: none") != null);
    try std.testing.expect(std.mem.indexOf(u8, subnormal_columns_explain, "select_columns_with_subnormals") != null);

    var lazy_non_finite_columns_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer lazy_non_finite_columns_scan.deinit();
    try lazy_non_finite_columns_scan.selectColumnsWithNonFinites();

    const non_finite_columns_explain = try lazy_non_finite_columns_scan.explain(gpa);
    defer gpa.free(non_finite_columns_explain);
    try std.testing.expect(std.mem.indexOf(u8, non_finite_columns_explain, "scan_pushdown: none") != null);
    try std.testing.expect(std.mem.indexOf(u8, non_finite_columns_explain, "select_columns_with_non_finites") != null);

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
