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
    try lazy_scan.withColumnAbs("sales_abs", "sales");
    try lazy_scan.withColumnNeg("sales_neg", "sales");
    try lazy_scan.withColumnSquare("sales_square", "sales");
    try lazy_scan.withColumnReciprocal("sales_recip", "sales");
    try lazy_scan.filterColumnScalar("sales", f64, 2.5, .gt);
    try lazy_scan.select(&.{ "sales_x2", "sales_abs", "sales_neg", "sales_square", "sales_recip", "id" });

    const explain = try lazy_scan.explain(gpa);
    defer gpa.free(explain);
    try std.testing.expect(std.mem.indexOf(u8, explain, "source=parquet_scan") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_scalar(sales_x2") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_abs(sales_abs=abs(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_neg(sales_neg=neg(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_square(sales_square=square(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_reciprocal(sales_recip=reciprocal(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "scan_pushdown: range=sales, projection=[sales,id]") != null);

    var result = try lazy_scan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 2), result.height());
    try std.testing.expectEqual(@as(usize, 6), result.width());
    try std.testing.expectEqual(@as(?usize, null), result.columnIndex("active"));
    try std.testing.expectEqual(@as(?usize, null), result.columnIndex("sales"));
    const result_sales_x2 = try (try result.column("sales_x2")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_x2);
    const result_sales_abs = try (try result.column("sales_abs")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_abs);
    const result_sales_neg = try (try result.column("sales_neg")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg);
    const result_sales_square = try (try result.column("sales_square")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_square);
    const result_sales_recip = try (try result.column("sales_recip")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_recip);
    try std.testing.expectEqualSlices(f64, &.{ 6.0, 10.0 }, result_sales_x2);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 5.0 }, result_sales_abs);
    try std.testing.expectEqualSlices(f64, &.{ -3.0, -5.0 }, result_sales_neg);
    try std.testing.expectEqualSlices(f64, &.{ 9.0, 25.0 }, result_sales_square);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), result_sales_recip[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.2), result_sales_recip[1], 1e-12);
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
