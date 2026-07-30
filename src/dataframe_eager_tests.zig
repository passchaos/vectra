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

    var cast_units = try table.castColumn("units", .f64);
    defer cast_units.deinit();
    try std.testing.expectEqual(DeviceDType.f64, try cast_units.columnDType("units"));
    const cast_units_values = try (try cast_units.column("units")).f64.toOwnedSlice(gpa);
    defer gpa.free(cast_units_values);
    const cast_units_validity = try (try cast_units.column("units")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(cast_units_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 2.0, 3.0 }, cast_units_values);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, cast_units_validity);

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

    var filtered = try table.filter(&.{ true, false, true });
    defer filtered.deinit();
    try std.testing.expectEqual(@as(usize, 2), filtered.height());
    const filtered_units = try filtered.column("units");
    try std.testing.expectEqual(@as(usize, 0), filtered_units.nullCount());
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
