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

fn expectApproxOrNan(expected: f64, actual: f64) !void {
    if (std.math.isNan(expected)) {
        try std.testing.expect(std.math.isNan(actual));
    } else {
        try std.testing.expectApproxEqAbs(expected, actual, 1e-12);
    }
}

fn expectF64SliceApproxOrNan(expected: []const f64, actual: []const f64) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |expected_item, actual_item| {
        try expectApproxOrNan(expected_item, actual_item);
    }
}

fn expectF64ColumnApproxOrNanWithValidity(frame: anytype, allocator: std.mem.Allocator, name: []const u8, expected_values: []const f64, expected_validity: []const bool) !void {
    const column = try frame.column(name);
    const values = try column.f64.toOwnedSlice(allocator);
    defer allocator.free(values);
    try expectF64SliceApproxOrNan(expected_values, values);
    if (column.f64.validity) |mask| {
        const validity = try mask.toOwnedSlice(allocator);
        defer allocator.free(validity);
        try std.testing.expectEqualSlices(bool, expected_validity, validity);
    } else {
        for (expected_validity) |valid| try std.testing.expect(valid);
    }
}

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

    var lazy_table = try DeviceLazyFrame.init(gpa, table);
    defer lazy_table.deinit();

    try std.testing.expectEqual(@as(usize, 3), table.height());
    try std.testing.expectEqual(@as(usize, 3), table.width());
    try std.testing.expectEqual(table.height(), table.rowCount());
    try std.testing.expectEqual(table.height(), table.nRows());
    try std.testing.expectEqual(table.width(), table.columnCount());
    try std.testing.expectEqual(table.width(), table.cols());
    try std.testing.expectEqual(table.width(), table.nCols());
    const names = table.columnNames();
    try std.testing.expectEqual(@as(usize, 3), names.len);
    try std.testing.expect(std.mem.eql(u8, "sales", names[0]));
    try std.testing.expect(std.mem.eql(u8, "units", names[1]));
    try std.testing.expect(std.mem.eql(u8, "active", names[2]));
    const column_labels = table.columnLabels();
    try std.testing.expectEqual(@as(usize, 3), column_labels.len);
    try std.testing.expect(std.mem.eql(u8, "sales", column_labels[0]));
    try std.testing.expect(std.mem.eql(u8, "units", column_labels[1]));
    try std.testing.expect(std.mem.eql(u8, "active", column_labels[2]));
    try std.testing.expect(table.columnNamesUnique());
    try std.testing.expect(!table.hasDuplicateColumnNames());
    try std.testing.expectEqual(@as(usize, 0), table.duplicateColumnNameCount());
    var duplicate_name_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "dup", .data = sales },
        .{ .name = "dup", .data = units },
    });
    defer duplicate_name_table.deinit();
    try std.testing.expect(!duplicate_name_table.columnNamesUnique());
    try std.testing.expect(duplicate_name_table.hasDuplicateColumnNames());
    try std.testing.expectEqual(@as(usize, 1), duplicate_name_table.duplicateColumnNameCount());
    try std.testing.expect(lazy_table.columnNamesUnique());
    try std.testing.expect(!lazy_table.hasDuplicateColumnNames());
    try std.testing.expectEqual(@as(usize, 0), lazy_table.duplicateColumnNameCount());
    const lazy_table_labels = try lazy_table.columnLabels(gpa);
    defer {
        for (lazy_table_labels) |label| gpa.free(label);
        gpa.free(lazy_table_labels);
    }
    try std.testing.expectEqualStrings("sales", lazy_table_labels[0]);
    try std.testing.expectEqualStrings("units", lazy_table_labels[1]);
    try std.testing.expectEqual(@as(?usize, 1), try lazy_table.columnIndex("units"));
    try std.testing.expect((try lazy_table.columnIndex("missing")) == null);
    try std.testing.expect(std.mem.eql(u8, "sales", try table.columnNameAt(0)));
    try std.testing.expectEqual(DeviceDType.f64, try table.columnDTypeAt(0));
    try std.testing.expectEqual(DeviceDType.bool, (try table.columnAt(2)).dtype());
    const sales_view = try table.columnView("sales");
    try std.testing.expectEqual(DeviceDType.f64, sales_view.dtype);
    try std.testing.expectEqual(@as(usize, 3), sales_view.rows);
    const units_view = try table.columnViewAt(1);
    try std.testing.expect(units_view.nullable());
    try std.testing.expect(units_view.hasNulls());
    try std.testing.expectEqual(DeviceValidityEncoding.bool_mask, units_view.validity_encoding);
    try std.testing.expectError(error.IndexOutOfBounds, table.columnNameAt(3));
    try std.testing.expectError(error.IndexOutOfBounds, table.columnDTypeAt(3));
    try std.testing.expectError(error.IndexOutOfBounds, table.columnAt(3));
    try std.testing.expectError(error.IndexOutOfBounds, table.columnViewAt(3));
    const dtypes = try table.columnDTypes(gpa);
    defer gpa.free(dtypes);
    try std.testing.expectEqualSlices(DeviceDType, &.{ .f64, .i64, .bool }, dtypes);
    const dtype_names = try table.columnDTypeNames(gpa);
    defer gpa.free(dtype_names);
    try std.testing.expect(std.mem.eql(u8, "f64", dtype_names[0]));
    try std.testing.expect(std.mem.eql(u8, "i64", dtype_names[1]));
    try std.testing.expect(std.mem.eql(u8, "bool", dtype_names[2]));
    const dtype_name_alias = try table.dtypeNames(gpa);
    defer gpa.free(dtype_name_alias);
    try std.testing.expect(std.mem.eql(u8, "f64", dtype_name_alias[0]));
    try std.testing.expect(std.mem.eql(u8, "i64", dtype_name_alias[1]));
    try std.testing.expect(std.mem.eql(u8, "bool", dtype_name_alias[2]));
    const dtype_byte_sizes = try table.columnDTypeByteSizes(gpa);
    defer gpa.free(dtype_byte_sizes);
    try std.testing.expectEqualSlices(usize, &.{ @sizeOf(f64), @sizeOf(i64), @sizeOf(bool) }, dtype_byte_sizes);
    const dtype_bit_sizes = try table.columnDTypeBitSizes(gpa);
    defer gpa.free(dtype_bit_sizes);
    try std.testing.expectEqualSlices(usize, &.{ @sizeOf(f64) * 8, @sizeOf(i64) * 8, @sizeOf(bool) * 8 }, dtype_bit_sizes);
    const numeric_mask = try table.columnIsNumericMask(gpa);
    defer gpa.free(numeric_mask);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false }, numeric_mask);
    const real_mask = try table.columnIsRealMask(gpa);
    defer gpa.free(real_mask);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false }, real_mask);
    const float_mask = try table.columnIsFloatMask(gpa);
    defer gpa.free(float_mask);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false }, float_mask);
    const integer_mask = try table.columnIsIntegerMask(gpa);
    defer gpa.free(integer_mask);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false }, integer_mask);
    const signed_integer_mask = try table.columnIsSignedIntegerMask(gpa);
    defer gpa.free(signed_integer_mask);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false }, signed_integer_mask);
    const unsigned_integer_mask = try table.columnIsUnsignedIntegerMask(gpa);
    defer gpa.free(unsigned_integer_mask);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false }, unsigned_integer_mask);
    const bool_mask = try table.columnIsBoolMask(gpa);
    defer gpa.free(bool_mask);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true }, bool_mask);
    const complex_mask = try table.columnIsComplexMask(gpa);
    defer gpa.free(complex_mask);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false }, complex_mask);
    const dtype_class_mask = try table.columnDTypeClassMask(gpa, .numeric);
    defer gpa.free(dtype_class_mask);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false }, dtype_class_mask);
    try std.testing.expectEqual(@as(usize, 2), table.columnDTypeClassCount(.numeric));
    try std.testing.expectEqual(@as(usize, 2), table.numericColumnCount());
    try std.testing.expectEqual(@as(usize, 2), table.realColumnCount());
    try std.testing.expectEqual(@as(usize, 1), table.floatColumnCount());
    try std.testing.expectEqual(@as(usize, 1), table.integerColumnCount());
    try std.testing.expectEqual(@as(usize, 1), table.signedIntegerColumnCount());
    try std.testing.expectEqual(@as(usize, 0), table.unsignedIntegerColumnCount());
    try std.testing.expectEqual(@as(usize, 1), table.boolColumnCount());
    try std.testing.expectEqual(@as(usize, 0), table.complexColumnCount());

    const lazy_table_dtype_byte_sizes = try lazy_table.columnDTypeByteSizes(gpa);
    defer gpa.free(lazy_table_dtype_byte_sizes);
    try std.testing.expectEqualSlices(usize, dtype_byte_sizes, lazy_table_dtype_byte_sizes);
    const lazy_table_dtype_bit_sizes = try lazy_table.columnDTypeBitSizes(gpa);
    defer gpa.free(lazy_table_dtype_bit_sizes);
    try std.testing.expectEqualSlices(usize, dtype_bit_sizes, lazy_table_dtype_bit_sizes);
    const lazy_table_numeric_mask = try lazy_table.columnDTypeClassMask(gpa, .numeric);
    defer gpa.free(lazy_table_numeric_mask);
    try std.testing.expectEqualSlices(bool, dtype_class_mask, lazy_table_numeric_mask);
    const lazy_table_float_mask = try lazy_table.columnIsFloatMask(gpa);
    defer gpa.free(lazy_table_float_mask);
    try std.testing.expectEqualSlices(bool, float_mask, lazy_table_float_mask);
    const lazy_table_signed_mask = try lazy_table.columnIsSignedIntegerMask(gpa);
    defer gpa.free(lazy_table_signed_mask);
    try std.testing.expectEqualSlices(bool, signed_integer_mask, lazy_table_signed_mask);
    const lazy_table_bool_mask = try lazy_table.columnIsBoolMask(gpa);
    defer gpa.free(lazy_table_bool_mask);
    try std.testing.expectEqualSlices(bool, bool_mask, lazy_table_bool_mask);
    const lazy_table_complex_mask = try lazy_table.columnIsComplexMask(gpa);
    defer gpa.free(lazy_table_complex_mask);
    try std.testing.expectEqualSlices(bool, complex_mask, lazy_table_complex_mask);
    try std.testing.expectEqual(table.columnDTypeClassCount(.numeric), try lazy_table.columnDTypeClassCount(.numeric));
    try std.testing.expectEqual(table.numericColumnCount(), try lazy_table.numericColumnCount());
    try std.testing.expectEqual(table.realColumnCount(), try lazy_table.realColumnCount());
    try std.testing.expectEqual(table.floatColumnCount(), try lazy_table.floatColumnCount());
    try std.testing.expectEqual(table.integerColumnCount(), try lazy_table.integerColumnCount());
    try std.testing.expectEqual(table.signedIntegerColumnCount(), try lazy_table.signedIntegerColumnCount());
    try std.testing.expectEqual(table.unsignedIntegerColumnCount(), try lazy_table.unsignedIntegerColumnCount());
    try std.testing.expectEqual(table.boolColumnCount(), try lazy_table.boolColumnCount());
    try std.testing.expectEqual(table.complexColumnCount(), try lazy_table.complexColumnCount());

    const null_counts = try table.columnNullCounts(gpa);
    defer gpa.free(null_counts);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 0 }, null_counts);
    const valid_counts = try table.columnValidCounts(gpa);
    defer gpa.free(valid_counts);
    try std.testing.expectEqualSlices(usize, &.{ 3, 2, 3 }, valid_counts);
    const lazy_table_null_counts = try lazy_table.columnNullCounts(gpa);
    defer gpa.free(lazy_table_null_counts);
    try std.testing.expectEqualSlices(usize, null_counts, lazy_table_null_counts);
    const lazy_table_valid_counts = try lazy_table.columnValidCounts(gpa);
    defer gpa.free(lazy_table_valid_counts);
    try std.testing.expectEqualSlices(usize, valid_counts, lazy_table_valid_counts);
    const lazy_table_projected_null_counts = try lazy_table.columnNullCountsProjection(gpa, &.{ "units", "active" });
    defer gpa.free(lazy_table_projected_null_counts);
    try std.testing.expectEqualSlices(usize, &.{ 1, 0 }, lazy_table_projected_null_counts);
    const lazy_table_projected_valid_counts = try lazy_table.columnValidCountsProjection(gpa, &.{ "units", "active" });
    defer gpa.free(lazy_table_projected_valid_counts);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, lazy_table_projected_valid_counts);
    try std.testing.expectEqual(@as(usize, 9), table.cellCount());
    try std.testing.expectEqual(table.height(), try lazy_table.height());
    try std.testing.expectEqual(table.width(), try lazy_table.cols());
    try std.testing.expect(lazy_table.shapeEquals(3, 3));
    try std.testing.expect(lazy_table.hasShape(3, 3));
    try std.testing.expect(!lazy_table.shapeEquals(3, 2));
    try std.testing.expect(lazy_table.sameHeight(&lazy_table));
    try std.testing.expect(lazy_table.sameWidth(&lazy_table));
    try std.testing.expect(lazy_table.sameShape(&lazy_table));
    try std.testing.expect(!lazy_table.isEmpty());
    try std.testing.expect(lazy_table.isNonEmpty());
    try std.testing.expectEqual(@as(usize, 1), table.nullCount());
    try std.testing.expectEqual(@as(usize, 8), table.validCount());
    try std.testing.expectEqual(table.nullCount(), try lazy_table.nullCount());
    try std.testing.expectEqual(table.validCount(), try lazy_table.validCount());
    try std.testing.expectEqual(@as(usize, 1), try lazy_table.nullCountProjection(&.{ "units", "active" }));
    try std.testing.expectEqual(@as(usize, 5), try lazy_table.validCountProjection(&.{ "units", "active" }));
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 9.0), table.nullRatio(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 8.0 / 9.0), table.validRatio(), 1e-12);
    try std.testing.expectApproxEqAbs(table.nullRatio(), try lazy_table.nullRatio(), 1e-12);
    try std.testing.expectApproxEqAbs(table.validRatio(), try lazy_table.validRatio(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 6.0), try lazy_table.nullRatioProjection(&.{ "units", "active" }), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 6.0), try lazy_table.validRatioProjection(&.{ "units", "active" }), 1e-12);
    const null_ratios = try table.columnNullRatios(gpa);
    defer gpa.free(null_ratios);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), null_ratios[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), null_ratios[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), null_ratios[2], 1e-12);
    const valid_ratios = try table.columnValidRatios(gpa);
    defer gpa.free(valid_ratios);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), valid_ratios[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), valid_ratios[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), valid_ratios[2], 1e-12);
    const lazy_table_null_ratios = try lazy_table.columnNullRatios(gpa);
    defer gpa.free(lazy_table_null_ratios);
    try std.testing.expectApproxEqAbs(null_ratios[1], lazy_table_null_ratios[1], 1e-12);
    const lazy_table_valid_ratios = try lazy_table.columnValidRatios(gpa);
    defer gpa.free(lazy_table_valid_ratios);
    try std.testing.expectApproxEqAbs(valid_ratios[1], lazy_table_valid_ratios[1], 1e-12);
    const lazy_table_projected_null_ratios = try lazy_table.columnNullRatiosProjection(gpa, &.{ "units", "active" });
    defer gpa.free(lazy_table_projected_null_ratios);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), lazy_table_projected_null_ratios[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_table_projected_null_ratios[1], 1e-12);
    const lazy_table_projected_valid_ratios = try lazy_table.columnValidRatiosProjection(gpa, &.{ "units", "active" });
    defer gpa.free(lazy_table_projected_valid_ratios);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), lazy_table_projected_valid_ratios[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_table_projected_valid_ratios[1], 1e-12);
    try std.testing.expectError(error.ColumnNotFound, lazy_table.columnNullCountsProjection(gpa, &.{"missing"}));
    try std.testing.expectError(error.ColumnNotFound, lazy_table.columnValidCountsProjection(gpa, &.{"missing"}));
    try std.testing.expectError(error.ColumnNotFound, lazy_table.columnNullRatiosProjection(gpa, &.{"missing"}));
    try std.testing.expectError(error.ColumnNotFound, lazy_table.columnValidRatiosProjection(gpa, &.{"missing"}));
    const distinct_counts = try table.columnDistinctCounts(gpa);
    defer gpa.free(distinct_counts);
    try std.testing.expectEqualSlices(usize, &.{ 3, 2, 2 }, distinct_counts);
    const n_unique_counts = try table.columnNUniqueCounts(gpa);
    defer gpa.free(n_unique_counts);
    try std.testing.expectEqualSlices(usize, &.{ 3, 2, 2 }, n_unique_counts);
    const n_unique_alias = try table.columnNUnique(gpa);
    defer gpa.free(n_unique_alias);
    try std.testing.expectEqualSlices(usize, &.{ 3, 2, 2 }, n_unique_alias);
    const duplicate_counts = try table.columnDuplicateCounts(gpa);
    defer gpa.free(duplicate_counts);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 1 }, duplicate_counts);
    const repeated_counts = try table.columnRepeatedCounts(gpa);
    defer gpa.free(repeated_counts);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 1 }, repeated_counts);
    const distinct_ratios = try table.columnDistinctRatios(gpa);
    defer gpa.free(distinct_ratios);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), distinct_ratios[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), distinct_ratios[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), distinct_ratios[2], 1e-12);
    const n_unique_ratios = try table.columnNUniqueRatios(gpa);
    defer gpa.free(n_unique_ratios);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), n_unique_ratios[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), n_unique_ratios[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), n_unique_ratios[2], 1e-12);
    const duplicate_ratios = try table.columnDuplicateRatios(gpa);
    defer gpa.free(duplicate_ratios);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), duplicate_ratios[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), duplicate_ratios[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), duplicate_ratios[2], 1e-12);
    const unique_mask = try table.columnIsUniqueMask(gpa);
    defer gpa.free(unique_mask);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false }, unique_mask);
    const duplicate_mask = try table.columnHasDuplicatesMask(gpa);
    defer gpa.free(duplicate_mask);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true }, duplicate_mask);
    const duplicate_alias = try table.columnHasDuplicateValues(gpa);
    defer gpa.free(duplicate_alias);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true }, duplicate_alias);
    const nullable_mask = try table.columnNullableMask(gpa);
    defer gpa.free(nullable_mask);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false }, nullable_mask);
    try std.testing.expectEqual(@as(usize, 1), table.nullableColumnCount());
    try std.testing.expectEqual(@as(usize, 2), table.nonNullableColumnCount());
    try std.testing.expectEqual(@as(?bool, false), try lazy_table.columnNullableAt(0));
    try std.testing.expectEqual(@as(?bool, true), try lazy_table.columnNullableAt(1));
    try std.testing.expect((try lazy_table.columnNullableAt(99)) == null);
    try std.testing.expect(try lazy_table.columnNullable("units"));
    try std.testing.expectError(error.ColumnNotFound, lazy_table.columnNullable("missing"));
    const lazy_table_nullable_mask = try lazy_table.columnNullableMask(gpa);
    defer gpa.free(lazy_table_nullable_mask);
    try std.testing.expectEqualSlices(bool, nullable_mask, lazy_table_nullable_mask);
    try std.testing.expectEqual(table.nullableColumnCount(), try lazy_table.nullableColumnCount());
    try std.testing.expectEqual(table.nonNullableColumnCount(), try lazy_table.nonNullableColumnCount());
    try std.testing.expect(lazy_table.hasNullableColumns());
    try std.testing.expect(!lazy_table.allColumnsNullable());
    const has_nulls_mask = try table.columnHasNullsMask(gpa);
    defer gpa.free(has_nulls_mask);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false }, has_nulls_mask);
    const lazy_has_nulls_mask = try lazy_table.columnHasNullsMask(gpa);
    defer gpa.free(lazy_has_nulls_mask);
    try std.testing.expectEqualSlices(bool, has_nulls_mask, lazy_has_nulls_mask);
    const lazy_projected_has_nulls_mask = try lazy_table.columnHasNullsMaskProjection(gpa, &.{ "units", "active" });
    defer gpa.free(lazy_projected_has_nulls_mask);
    try std.testing.expectEqualSlices(bool, &.{ true, false }, lazy_projected_has_nulls_mask);
    try std.testing.expectEqual(@as(usize, 1), table.columnsWithNullsCount());
    try std.testing.expectEqual(@as(usize, 2), table.columnsWithoutNullsCount());
    try std.testing.expectEqual(table.columnsWithNullsCount(), try lazy_table.columnsWithNullsCount());
    try std.testing.expectEqual(table.columnsWithoutNullsCount(), try lazy_table.columnsWithoutNullsCount());
    try std.testing.expectEqual(@as(usize, 1), try lazy_table.columnsWithNullsCountProjection(&.{ "units", "active" }));
    try std.testing.expectEqual(@as(usize, 1), try lazy_table.columnsWithoutNullsCountProjection(&.{ "units", "active" }));
    const data_nbytes = try table.columnDataNbytes(gpa);
    defer gpa.free(data_nbytes);
    try std.testing.expectEqualSlices(usize, &.{ 3 * @sizeOf(f64), 3 * @sizeOf(i64), 3 * @sizeOf(bool) }, data_nbytes);
    const data_memory_usage = try table.columnDataMemoryUsage(gpa);
    defer gpa.free(data_memory_usage);
    try std.testing.expectEqualSlices(usize, data_nbytes, data_memory_usage);
    const lazy_data_nbytes = try lazy_table.columnDataNbytes(gpa);
    defer gpa.free(lazy_data_nbytes);
    try std.testing.expectEqualSlices(usize, data_nbytes, lazy_data_nbytes);
    const lazy_projected_data_nbytes = try lazy_table.columnDataNbytesProjection(gpa, &.{ "units", "active" });
    defer gpa.free(lazy_projected_data_nbytes);
    try std.testing.expectEqualSlices(usize, &.{ 3 * @sizeOf(i64), 3 * @sizeOf(bool) }, lazy_projected_data_nbytes);
    const validity_nbytes = try table.columnValidityNbytes(gpa);
    defer gpa.free(validity_nbytes);
    try std.testing.expectEqualSlices(usize, &.{ 0, 3 * @sizeOf(bool), 0 }, validity_nbytes);
    const validity_memory_usage = try table.columnValidityMemoryUsage(gpa);
    defer gpa.free(validity_memory_usage);
    try std.testing.expectEqualSlices(usize, validity_nbytes, validity_memory_usage);
    const lazy_validity_nbytes = try lazy_table.columnValidityNbytes(gpa);
    defer gpa.free(lazy_validity_nbytes);
    try std.testing.expectEqualSlices(usize, validity_nbytes, lazy_validity_nbytes);
    const lazy_projected_validity_nbytes = try lazy_table.columnValidityNbytesProjection(gpa, &.{ "units", "active" });
    defer gpa.free(lazy_projected_validity_nbytes);
    try std.testing.expectEqualSlices(usize, &.{ 3 * @sizeOf(bool), 0 }, lazy_projected_validity_nbytes);
    const total_nbytes = try table.columnTotalNbytes(gpa);
    defer gpa.free(total_nbytes);
    try std.testing.expectEqualSlices(usize, &.{ 3 * @sizeOf(f64), 3 * @sizeOf(i64) + 3 * @sizeOf(bool), 3 * @sizeOf(bool) }, total_nbytes);
    const column_memory_usage = try table.columnMemoryUsage(gpa);
    defer gpa.free(column_memory_usage);
    try std.testing.expectEqualSlices(usize, total_nbytes, column_memory_usage);
    const lazy_total_nbytes = try lazy_table.columnTotalNbytes(gpa);
    defer gpa.free(lazy_total_nbytes);
    try std.testing.expectEqualSlices(usize, total_nbytes, lazy_total_nbytes);
    const lazy_projected_total_nbytes = try lazy_table.columnTotalNbytesProjection(gpa, &.{ "units", "active" });
    defer gpa.free(lazy_projected_total_nbytes);
    try std.testing.expectEqualSlices(usize, &.{ 3 * @sizeOf(i64) + 3 * @sizeOf(bool), 3 * @sizeOf(bool) }, lazy_projected_total_nbytes);
    const lazy_projected_memory = try lazy_table.columnMemoryUsageProjection(gpa, &.{ "units", "active" });
    defer gpa.free(lazy_projected_memory);
    try std.testing.expectEqualSlices(usize, lazy_projected_total_nbytes, lazy_projected_memory);
    try std.testing.expectEqual(@as(usize, 3 * @sizeOf(f64) + 3 * @sizeOf(i64) + 3 * @sizeOf(bool)), table.dataNbytes());
    try std.testing.expectEqual(table.dataNbytes(), table.dataMemoryUsage());
    try std.testing.expectEqual(@as(usize, 3 * @sizeOf(bool)), table.validityNbytes());
    try std.testing.expectEqual(table.validityNbytes(), table.validityMemoryUsage());
    try std.testing.expectEqual(table.dataNbytes() + table.validityNbytes(), table.totalNbytes());
    try std.testing.expectEqual(table.totalNbytes(), table.memoryUsage());
    try std.testing.expectEqual(table.totalNbytes(), table.estimatedSize());
    try std.testing.expectEqual(table.totalNbytes(), lazy_table.sourceNbytes());
    try std.testing.expectEqual(table.totalNbytes(), lazy_table.sourceByteCount());
    try std.testing.expectEqual(table.totalNbytes(), lazy_table.nbytes());
    try std.testing.expectEqual(table.totalNbytes(), lazy_table.byteCount());
    try std.testing.expect(lazy_table.hasBytes());
    try std.testing.expectEqual(table.dataNbytes(), try lazy_table.dataNbytes());
    try std.testing.expectEqual(table.validityNbytes(), try lazy_table.validityNbytes());
    try std.testing.expectEqual(table.totalNbytes(), try lazy_table.totalNbytes());
    try std.testing.expectEqual(@as(usize, 3 * @sizeOf(i64)), try lazy_table.dataNbytesProjection(&.{"units"}));
    try std.testing.expectEqual(@as(usize, 3 * @sizeOf(bool)), try lazy_table.validityNbytesProjection(&.{"units"}));
    try std.testing.expectEqual(@as(usize, 3 * @sizeOf(i64) + 3 * @sizeOf(bool)), try lazy_table.totalNbytesProjection(&.{"units"}));
    try std.testing.expectError(error.ColumnNotFound, lazy_table.columnDataNbytesProjection(gpa, &.{"missing"}));
    try std.testing.expectEqual(table.totalNbytes(), lazy_table.ownedNbytes());
    try std.testing.expectEqual(lazy_table.ownedNbytes(), lazy_table.memoryUsage());
    try std.testing.expectEqual(lazy_table.ownedNbytes(), lazy_table.estimatedSize());
    const schema = try table.schemaSummary(gpa);
    defer gpa.free(schema);
    try std.testing.expectEqual(@as(usize, 3), schema.len);
    const lazy_table_schema = try lazy_table.schemaSummary(gpa);
    defer gpa.free(lazy_table_schema);
    try std.testing.expectEqual(@as(usize, 3), lazy_table_schema.len);
    try std.testing.expect(lazy_table.schemaEqualsSchemas(schema));
    try std.testing.expect(lazy_table.sameSchemaSchemas(schema));
    try std.testing.expect(lazy_table.schemaCompatibleSchemas(schema));
    try std.testing.expect(lazy_table.schemaEquals(&lazy_table));
    try std.testing.expect(lazy_table.sameSchema(&lazy_table));
    try std.testing.expect(lazy_table.schemaCompatible(&lazy_table));
    const lazy_table_schema_at = (try lazy_table.columnSchemaAt(1)).?;
    try std.testing.expect(lazy_table_schema_at.schemaEquals(schema[1]));
    try std.testing.expect((try lazy_table.columnSchemaAt(99)) == null);
    const lazy_table_units_schema = try lazy_table.columnSchema("units");
    try std.testing.expect(lazy_table_units_schema.schemaEquals(schema[1]));
    try std.testing.expectError(error.ColumnNotFound, lazy_table.columnSchema("missing"));
    const lazy_table_schema_alias = try lazy_table.schema(gpa);
    defer gpa.free(lazy_table_schema_alias);
    try std.testing.expectEqual(@as(usize, 3), lazy_table_schema_alias.len);
    try std.testing.expect(lazy_table_schema_alias[1].schemaEquals(schema[1]));
    const lazy_table_column_schemas = try lazy_table.columnSchemas(gpa);
    defer gpa.free(lazy_table_column_schemas);
    try std.testing.expectEqual(@as(usize, 3), lazy_table_column_schemas.len);
    try std.testing.expect(lazy_table_column_schemas[1].schemaEquals(schema[1]));
    try std.testing.expect(std.mem.eql(u8, "units", schema[1].name));
    try std.testing.expectEqual(DeviceDType.i64, schema[1].dtype);
    try std.testing.expect(schema[1].nullable);
    try std.testing.expect(schema[1].nullableColumn());
    try std.testing.expect(schema[1].hasNulls());
    try std.testing.expect(!schema[1].allValid());
    try std.testing.expectEqual(@as(usize, 1), schema[1].null_count);
    try std.testing.expectEqual(@as(usize, 2), schema[1].valid_count);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), schema[1].nullRatio(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), schema[1].validRatio(), 1e-12);
    try std.testing.expectEqual(3 * @sizeOf(i64), schema[1].data_nbytes);
    try std.testing.expectEqual(3 * @sizeOf(bool), schema[1].validity_nbytes);
    try std.testing.expectEqual(schema[1].data_nbytes, schema[1].dataMemoryUsage());
    try std.testing.expectEqual(schema[1].validity_nbytes, schema[1].validityMemoryUsage());
    try std.testing.expectEqual(schema[1].total_nbytes, schema[1].memoryUsage());
    try std.testing.expectEqual(schema[1].total_nbytes, schema[1].estimatedSize());
    try std.testing.expect(schema[1].device.isCpu());
    try std.testing.expect(schema[1].isCpu());
    try std.testing.expect(schema[1].isHostBacked());
    try std.testing.expect(!schema[1].isCuda());
    try std.testing.expect(!schema[1].isCudaBacked());
    try std.testing.expect(!schema[1].isMps());
    try std.testing.expect(!schema[1].isMpsBacked());
    try std.testing.expect(!schema[1].isAcceleratorBacked());
    try std.testing.expect(!schema[1].isRemoteBacked());
    const schema_alias = try table.schema(gpa);
    defer gpa.free(schema_alias);
    try std.testing.expectEqual(@as(usize, 3), schema_alias.len);
    try std.testing.expectEqual(DeviceDType.bool, schema_alias[2].dtype);
    const units_schema = try table.columnSchema("units");
    try std.testing.expect(std.mem.eql(u8, "units", units_schema.name));
    try std.testing.expectEqual(DeviceDType.i64, units_schema.dtype);
    try std.testing.expectEqual(@as(usize, 3), units_schema.len());
    try std.testing.expect(std.mem.eql(u8, "i64", units_schema.dtypeName()));
    try std.testing.expectEqual(@as(usize, @sizeOf(i64)), units_schema.dtypeByteSize());
    try std.testing.expectEqual(DeviceDType.i64.bitSize(), units_schema.dtypeBitSize());
    try std.testing.expect(units_schema.isNumeric());
    try std.testing.expect(units_schema.isReal());
    try std.testing.expect(units_schema.isInteger());
    try std.testing.expect(units_schema.isSignedInteger());
    try std.testing.expect(!units_schema.isUnsignedInteger());
    try std.testing.expect(!units_schema.isBool());
    try std.testing.expectEqual(@as(usize, 1), units_schema.null_count);
    try std.testing.expectEqual(@as(usize, 1), units_schema.nullCount());
    try std.testing.expectEqual(@as(usize, 2), units_schema.validCount());
    try std.testing.expectEqual(units_schema.data_nbytes, units_schema.dataNbytes());
    try std.testing.expect(units_schema.dataPtr() != 0);
    try std.testing.expectEqual(units_schema.validity_nbytes, units_schema.validityNbytes());
    try std.testing.expect(units_schema.hasValidity());
    try std.testing.expect(units_schema.validityPtr() != null);
    try std.testing.expectEqual(units_schema.total_nbytes, units_schema.totalNbytes());
    try std.testing.expect(units_schema.isCpu());
    try std.testing.expect(units_schema.isHostBacked());
    try std.testing.expect(!units_schema.isDeviceBacked());
    try std.testing.expect(units_schema.isDeviceAvailable());
    try std.testing.expect(std.mem.eql(u8, "cpu", units_schema.deviceBackendName()));
    try std.testing.expectEqual(vectra.Device.cpu.backend, units_schema.deviceBackend());
    try std.testing.expect(units_schema.deviceValue().sameDevice(.cpu));
    try std.testing.expectEqual(@as(usize, 0), units_schema.deviceIndex());
    try std.testing.expect(units_schema.anyNull());
    try std.testing.expect(!units_schema.allNull());
    try std.testing.expect(units_schema.anyValid());
    try std.testing.expect(!units_schema.allValid());
    try std.testing.expect(units_schema.schemaEquals(try table.columnSchemaAt(1)));
    try std.testing.expect(units_schema.sameSchema(try table.columnSchema("units")));
    try std.testing.expect(units_schema.schemaCompatible(try table.columnSchemaAt(1)));
    try std.testing.expect(units_schema.sameStorage(try table.columnSchemaAt(1)));
    const active_schema = try table.columnSchemaAt(2);
    try std.testing.expect(std.mem.eql(u8, "active", active_schema.name));
    try std.testing.expectEqual(DeviceDType.bool, active_schema.dtype);
    try std.testing.expect(active_schema.isBool());
    try std.testing.expect(units_schema.sameDevice(active_schema));
    try std.testing.expect(units_schema.sameLength(active_schema));
    try std.testing.expect(units_schema.lengthEquals(3));
    try std.testing.expect(!units_schema.sameDType(active_schema));
    try std.testing.expect(!units_schema.sameNullability(active_schema));
    try std.testing.expect(!units_schema.sameStorage(active_schema));
    try std.testing.expect(!units_schema.schemaEquals(active_schema));
    try std.testing.expectError(error.ColumnNotFound, table.columnSchema("missing"));
    try std.testing.expectError(error.IndexOutOfBounds, table.columnSchemaAt(3));
    try std.testing.expect(table.isNonEmpty());
    try std.testing.expect(!table.isEmpty());
    try std.testing.expect(table.hasRows());
    try std.testing.expect(table.hasColumns());
    try std.testing.expect(table.shapeEquals(3, 3));
    try std.testing.expect(table.hasShape(3, 3));
    try std.testing.expect(!table.shapeEquals(3, 2));
    try std.testing.expect(table.hasColumn("sales"));
    try std.testing.expect(!table.hasColumn("missing"));
    try std.testing.expect(table.hasAllColumns(&.{ "sales", "units" }));
    try std.testing.expect(!table.hasAllColumns(&.{ "sales", "missing" }));
    try std.testing.expect(table.hasAnyColumn(&.{ "missing", "active" }));
    try std.testing.expect(!table.hasAnyColumn(&.{ "missing", "absent" }));
    var table_clone = try table.clone();
    defer table_clone.deinit();
    try std.testing.expect(table.sameStorage(table));
    try std.testing.expect(!table.sameStorage(table_clone));
    try std.testing.expect(table.sameShape(table_clone));
    try std.testing.expect(table.sameHeight(table_clone));
    try std.testing.expect(table.sameWidth(table_clone));
    try std.testing.expect(try table.equals(table_clone));
    try std.testing.expect(try table.frameEquals(table_clone));
    try std.testing.expect(try table.allClose(table_clone, 0.0, 0.0));
    try std.testing.expect(try table.frameAllClose(table_clone, 0.0, 0.0));
    try std.testing.expect(table.schemaEquals(table_clone));
    try std.testing.expect(table.sameSchema(table_clone));
    try std.testing.expect(table.schemaCompatible(table_clone));
    var close_sales = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.01, 2.99, 5.02 }, .cpu);
    defer close_sales.deinit();
    var close_units = try DeviceColumn.fromSliceWithValidity(i64, gpa, &.{ 1, 2, 3 }, &.{ true, false, true }, .cpu);
    defer close_units.deinit();
    var close_active = try DeviceColumn.fromSlice(bool, gpa, &.{ true, false, true }, .cpu);
    defer close_active.deinit();
    var close_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "sales", .data = close_sales },
        .{ .name = "units", .data = close_units },
        .{ .name = "active", .data = close_active },
    });
    defer close_table.deinit();
    try std.testing.expect(!try table.equals(close_table));
    try std.testing.expect(try table.allClose(close_table, 0.0, 0.05));
    try std.testing.expect(!try table.allClose(close_table, 0.0, 0.001));
    try std.testing.expectError(error.InvalidShape, table.allClose(close_table, -1.0, 0.0));
    var reordered = try table.select(&.{ "sales", "active", "units" });
    defer reordered.deinit();
    try std.testing.expect(table.sameShape(reordered));
    try std.testing.expect(!try table.equals(reordered));
    try std.testing.expect(!table.schemaEquals(reordered));
    var changed = try table.withColumnLiteral("active", bool, false);
    defer changed.deinit();
    try std.testing.expect(!try table.equals(changed));
    try std.testing.expect(table.schemaEquals(changed));
    var different_nulls = try table.fillNullColumn("units", i64, 0);
    defer different_nulls.deinit();
    try std.testing.expect(!try table.equals(different_nulls));
    try std.testing.expect(!table.schemaEquals(different_nulls));
    try std.testing.expect(table.device.isCpu());
    try std.testing.expect(table.isCpu());
    try std.testing.expect(table.isHostBacked());
    try std.testing.expect(!table.isCuda());
    try std.testing.expect(!table.isCudaBacked());
    try std.testing.expect(!table.isMps());
    try std.testing.expect(!table.isMpsBacked());
    try std.testing.expect(!table.isAcceleratorBacked());
    try std.testing.expect(!table.isRemoteBacked());
    try std.testing.expect(!table.isDeviceBacked());
    try std.testing.expect(table.isDeviceAvailable());
    try std.testing.expect(std.mem.eql(u8, "cpu", table.deviceBackendName()));
    try std.testing.expectEqual(vectra.Device.cpu.backend, table.deviceBackend());
    try std.testing.expect(table.deviceValue().sameDevice(.cpu));
    try std.testing.expectEqual(@as(usize, 0), table.deviceIndex());
    try std.testing.expect(table.sameDevice(table));
    try std.testing.expect(lazy_table.sameDevice(&lazy_table));
    try std.testing.expect(lazy_table.sameStorage(&lazy_table));
    try std.testing.expect(lazy_table.sharesStorage(&lazy_table));
    try std.testing.expect(lazy_table.sameSource(&lazy_table));
    try std.testing.expect(lazy_table.sharesSource(&lazy_table));
    try std.testing.expectEqual(DeviceDType.i64, try table.columnDType("units"));

    const units_col = try table.column("units");
    try std.testing.expect(units_col.nullable());
    try std.testing.expect(units_col.hasNulls());
    try std.testing.expectEqual(@as(usize, 1), units_col.nullCount());
    try std.testing.expectEqual(@as(usize, 3), units_col.len());
    try std.testing.expectEqual(units_col.len(), units_col.rowCount());
    try std.testing.expectEqual(units_col.len(), units_col.height());
    try std.testing.expectEqual(units_col.len(), units_col.nRows());
    try std.testing.expectEqual(@as(usize, 3), units_col.shape().rows);
    try std.testing.expect(units_col.shapeEquals(3));
    try std.testing.expect(units_col.hasShape(3));
    try std.testing.expect(!units_col.isEmpty());
    try std.testing.expect(units_col.isNonEmpty());
    try std.testing.expect(units_col.hasRows());
    try std.testing.expectEqual(units_col.len(), units_col.cellCount());
    try std.testing.expectEqual(DeviceDType.i64, units_col.dtype());
    try std.testing.expect(std.mem.eql(u8, "i64", units_col.dtypeName()));
    try std.testing.expectEqual(DeviceDType.i64.bitSize(), units_col.dtypeBitSize());
    try std.testing.expect(units_col.isNumeric());
    try std.testing.expect(units_col.isReal());
    try std.testing.expect(!units_col.isFloat());
    try std.testing.expect(units_col.isInteger());
    try std.testing.expect(units_col.isSignedInteger());
    try std.testing.expect(!units_col.isUnsignedInteger());
    try std.testing.expect(!units_col.isBool());
    try std.testing.expect(!units_col.isComplex());
    try std.testing.expect(units_col.isCpu());
    try std.testing.expect(units_col.isHostBacked());
    try std.testing.expect(!units_col.isCudaBacked());
    try std.testing.expect(!units_col.isMpsBacked());
    try std.testing.expect(!units_col.isAcceleratorBacked());
    try std.testing.expect(!units_col.isRemoteBacked());
    try std.testing.expect(!units_col.isDeviceBacked());
    try std.testing.expect(units_col.isDeviceAvailable());
    try std.testing.expect(std.mem.eql(u8, "cpu", units_col.deviceBackendName()));
    try std.testing.expectEqual(vectra.Device.cpu.backend, units_col.deviceBackend());
    try std.testing.expect(units_col.deviceValue().sameDevice(.cpu));
    try std.testing.expectEqual(@as(usize, 0), units_col.deviceIndex());
    try std.testing.expectEqual(units_col.dataNbytes(), units_col.dataMemoryUsage());
    try std.testing.expect(units_col.dataPtr() != 0);
    try std.testing.expectEqual(units_col.validityNbytes(), units_col.validityMemoryUsage());
    try std.testing.expect(units_col.hasValidity());
    try std.testing.expect(units_col.validityPtr() != null);
    try std.testing.expectEqual(DeviceValidityEncoding.bool_mask, units_col.validityEncoding());
    try std.testing.expectEqual(units_col.totalNbytes(), units_col.memoryUsage());
    try std.testing.expectEqual(units_col.totalNbytes(), units_col.estimatedSize());
    try std.testing.expect(units_col.schema("units").schemaEquals(units_schema));

    var view = try table.view();
    defer view.deinit();
    try std.testing.expectEqual(@as(usize, 3), view.height());
    try std.testing.expectEqual(view.height(), view.rowCount());
    try std.testing.expectEqual(view.height(), view.nRows());
    try std.testing.expectEqual(@as(usize, 3), view.width());
    try std.testing.expectEqual(view.width(), view.columnCount());
    try std.testing.expectEqual(view.width(), view.cols());
    try std.testing.expectEqual(view.width(), view.nCols());
    try std.testing.expect(view.columnNamesUnique());
    try std.testing.expect(!view.hasDuplicateColumnNames());
    try std.testing.expectEqual(@as(usize, 0), view.duplicateColumnNameCount());
    try std.testing.expect(!view.isEmpty());
    try std.testing.expect(view.isNonEmpty());
    try std.testing.expect(view.hasRows());
    try std.testing.expect(view.hasColumns());
    try std.testing.expect(view.isCpu());
    try std.testing.expect(view.isHostBacked());
    try std.testing.expect(!view.isCudaBacked());
    try std.testing.expect(!view.isMpsBacked());
    try std.testing.expect(!view.isAcceleratorBacked());
    try std.testing.expect(!view.isRemoteBacked());
    try std.testing.expect(!view.isDeviceBacked());
    try std.testing.expect(view.isDeviceAvailable());
    try std.testing.expect(std.mem.eql(u8, "cpu", view.deviceBackendName()));
    try std.testing.expectEqual(vectra.Device.cpu.backend, view.deviceBackend());
    try std.testing.expect(view.deviceValue().sameDevice(.cpu));
    try std.testing.expectEqual(@as(usize, 0), view.deviceIndex());
    try std.testing.expect(view.sameDevice(view));
    try std.testing.expect(view.sameShape(view));
    try std.testing.expect(view.shapeEquals(3, 3));
    try std.testing.expect(view.hasShape(3, 3));
    const view_dtypes = try view.columnDTypes(gpa);
    defer gpa.free(view_dtypes);
    try std.testing.expectEqualSlices(DeviceDType, &.{ .f64, .i64, .bool }, view_dtypes);
    const view_dtypes_alias = try view.dtypes(gpa);
    defer gpa.free(view_dtypes_alias);
    try std.testing.expectEqualSlices(DeviceDType, view_dtypes, view_dtypes_alias);
    const view_dtype_names = try view.dtypeNames(gpa);
    defer gpa.free(view_dtype_names);
    try std.testing.expect(std.mem.eql(u8, "f64", view_dtype_names[0]));
    try std.testing.expect(std.mem.eql(u8, "i64", view_dtype_names[1]));
    try std.testing.expect(std.mem.eql(u8, "bool", view_dtype_names[2]));
    const view_dtype_byte_sizes = try view.columnDTypeByteSizes(gpa);
    defer gpa.free(view_dtype_byte_sizes);
    try std.testing.expectEqualSlices(usize, &.{ @sizeOf(f64), @sizeOf(i64), @sizeOf(bool) }, view_dtype_byte_sizes);
    const view_dtype_bit_sizes = try view.columnDTypeBitSizes(gpa);
    defer gpa.free(view_dtype_bit_sizes);
    try std.testing.expectEqualSlices(usize, &.{ DeviceDType.f64.bitSize(), DeviceDType.i64.bitSize(), DeviceDType.bool.bitSize() }, view_dtype_bit_sizes);
    try std.testing.expectEqual(@as(usize, 2), view.numericColumnCount());
    try std.testing.expectEqual(@as(usize, 2), view.realColumnCount());
    try std.testing.expectEqual(@as(usize, 1), view.floatColumnCount());
    try std.testing.expectEqual(@as(usize, 1), view.integerColumnCount());
    try std.testing.expectEqual(@as(usize, 1), view.signedIntegerColumnCount());
    try std.testing.expectEqual(@as(usize, 0), view.unsignedIntegerColumnCount());
    try std.testing.expectEqual(@as(usize, 1), view.boolColumnCount());
    try std.testing.expectEqual(@as(usize, 0), view.complexColumnCount());
    const view_numeric_mask = try view.columnIsNumericMask(gpa);
    defer gpa.free(view_numeric_mask);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false }, view_numeric_mask);
    const view_bool_mask = try view.columnIsBoolMask(gpa);
    defer gpa.free(view_bool_mask);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true }, view_bool_mask);
    try std.testing.expect(view.hasColumn("sales"));
    try std.testing.expect(view.hasAllColumns(&.{ "sales", "units" }));
    try std.testing.expect(view.hasAnyColumn(&.{ "missing", "units" }));
    try std.testing.expect(!view.hasColumn("missing"));
    try std.testing.expectEqual(@as(?usize, 1), view.columnIndex("units"));
    try std.testing.expectEqual(DeviceDType.i64, try view.columnDType("units"));
    try std.testing.expectEqual(DeviceDType.bool, try view.columnDTypeAt(2));
    try std.testing.expect(std.mem.eql(u8, "active", try view.columnNameAt(2)));
    const view_names = view.columnLabels();
    try std.testing.expect(std.mem.eql(u8, "sales", view_names[0]));
    const sales_column_view = try view.column("sales");
    try std.testing.expectEqual(DeviceDType.f64, sales_column_view.dtype);
    try std.testing.expectEqual(@as(usize, 3), sales_column_view.len());
    try std.testing.expectEqual(sales_column_view.len(), sales_column_view.rowCount());
    try std.testing.expectEqual(sales_column_view.len(), sales_column_view.height());
    try std.testing.expectEqual(sales_column_view.len(), sales_column_view.nRows());
    try std.testing.expectEqual(@as(usize, 3), sales_column_view.shape().rows);
    try std.testing.expect(sales_column_view.shapeEquals(3));
    try std.testing.expect(sales_column_view.hasShape(3));
    try std.testing.expect(!sales_column_view.isEmpty());
    try std.testing.expect(sales_column_view.isNonEmpty());
    try std.testing.expect(sales_column_view.hasRows());
    try std.testing.expectEqual(sales_column_view.len(), sales_column_view.cellCount());
    try std.testing.expect(std.mem.eql(u8, "f64", sales_column_view.dtypeName()));
    try std.testing.expectEqual(@as(usize, @sizeOf(f64)), sales_column_view.dtypeByteSize());
    try std.testing.expectEqual(@as(usize, @bitSizeOf(f64)), sales_column_view.dtypeBitSize());
    try std.testing.expect(sales_column_view.isNumeric());
    try std.testing.expect(sales_column_view.isReal());
    try std.testing.expect(sales_column_view.isFloat());
    try std.testing.expect(!sales_column_view.isInteger());
    try std.testing.expect(!sales_column_view.isBool());
    try std.testing.expect(!sales_column_view.isComplex());
    try std.testing.expectEqual(@as(usize, 3 * @sizeOf(f64)), sales_column_view.dataNbytes());
    try std.testing.expectEqual(sales_column_view.dataNbytes(), sales_column_view.dataMemoryUsage());
    try std.testing.expectEqual(@as(usize, 0), sales_column_view.validityNbytes());
    try std.testing.expectEqual(sales_column_view.totalNbytes(), sales_column_view.memoryUsage());
    try std.testing.expectEqual(sales_column_view.totalNbytes(), sales_column_view.estimatedSize());
    try std.testing.expectEqual(@as(usize, 0), sales_column_view.nullCount());
    try std.testing.expectEqual(@as(usize, 3), sales_column_view.validCount());
    try std.testing.expect(!sales_column_view.anyNull());
    try std.testing.expect(!sales_column_view.allNull());
    try std.testing.expect(sales_column_view.anyValid());
    try std.testing.expect(sales_column_view.allValid());
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), sales_column_view.nullRatio(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), sales_column_view.validRatio(), 1e-12);
    try std.testing.expect(sales_column_view.isCpu());
    try std.testing.expect(sales_column_view.isHostBacked());
    try std.testing.expect(!sales_column_view.isCudaBacked());
    try std.testing.expect(!sales_column_view.isMpsBacked());
    try std.testing.expect(!sales_column_view.isAcceleratorBacked());
    try std.testing.expect(!sales_column_view.isRemoteBacked());
    try std.testing.expect(!sales_column_view.isDeviceBacked());
    try std.testing.expect(sales_column_view.isDeviceAvailable());
    try std.testing.expect(std.mem.eql(u8, "cpu", sales_column_view.deviceBackendName()));
    try std.testing.expectEqual(vectra.Device.cpu.backend, sales_column_view.deviceBackend());
    try std.testing.expect(sales_column_view.deviceValue().sameDevice(.cpu));
    try std.testing.expectEqual(@as(usize, 0), sales_column_view.deviceIndex());
    try std.testing.expect(sales_column_view.dataPtr() != 0);
    try std.testing.expectEqual(sales_column_view.data_ptr, sales_column_view.dataPtr());
    try std.testing.expect(!sales_column_view.hasValidity());
    try std.testing.expect(sales_column_view.validityPtr() == null);
    try std.testing.expect(sales_column_view.schema("sales").schemaEquals(try view.columnSchema("sales")));
    const units_column_view = try view.columnViewAt(1);
    try std.testing.expect(sales_column_view.sameDevice(units_column_view));
    try std.testing.expect(sales_column_view.sameLength(units_column_view));
    try std.testing.expect(sales_column_view.sameShape(units_column_view));
    try std.testing.expect(sales_column_view.lengthEquals(3));
    try std.testing.expect(!sales_column_view.sameDType(units_column_view));
    try std.testing.expect(!sales_column_view.sameNullability(units_column_view));
    try std.testing.expect(!sales_column_view.schemaEquals(units_column_view));
    try std.testing.expect(sales_column_view.sameSchema(sales_column_view));
    try std.testing.expect(sales_column_view.schemaCompatible(sales_column_view));
    try std.testing.expect(sales_column_view.sameStorage(sales_column_view));
    try std.testing.expect(!sales_column_view.sameStorage(units_column_view));
    const sales_col = try table.column("sales");
    try std.testing.expect(sales_col.sameDevice(units_col.*));
    try std.testing.expect(sales_col.sameLength(units_col.*));
    try std.testing.expect(sales_col.sameShape(units_col.*));
    try std.testing.expect(sales_col.lengthEquals(3));
    try std.testing.expect(!sales_col.sameDType(units_col.*));
    try std.testing.expect(!sales_col.sameNullability(units_col.*));
    try std.testing.expect(!sales_col.schemaEquals(units_col.*));
    try std.testing.expect(sales_col.schemaEquals(sales_col.*));
    try std.testing.expect(sales_col.sameSchema(sales_col.*));
    try std.testing.expect(sales_col.schemaCompatible(sales_col.*));
    try std.testing.expect(sales_col.sameStorage(sales_col.*));
    try std.testing.expect(!sales_col.sameStorage(units_col.*));
    try std.testing.expectEqual(DeviceDType.i64, units_column_view.dtype);
    try std.testing.expectEqual(DeviceValidityEncoding.bool_mask, units_column_view.validity_encoding);
    try std.testing.expectEqual(@as(usize, 1), units_column_view.nullCount());
    try std.testing.expectEqual(@as(usize, 2), units_column_view.validCount());
    try std.testing.expect(units_column_view.anyNull());
    try std.testing.expect(!units_column_view.allNull());
    try std.testing.expect(units_column_view.anyValid());
    try std.testing.expect(!units_column_view.allValid());
    try std.testing.expect(units_column_view.hasValidity());
    try std.testing.expectEqual(units_column_view.validity_ptr, units_column_view.validityPtr());
    try std.testing.expectEqual(DeviceValidityEncoding.bool_mask, units_column_view.validityEncoding());
    try std.testing.expectEqual(units_column_view.totalNbytes(), units_column_view.dataNbytes() + units_column_view.validityNbytes());
    const active_column_view = try view.columnViewAt(2);
    try std.testing.expect(active_column_view.isBool());
    try std.testing.expect(!active_column_view.isNumeric());
    const view_null_counts = try view.columnNullCounts(gpa);
    defer gpa.free(view_null_counts);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 0 }, view_null_counts);
    const view_valid_counts = try view.columnValidCounts(gpa);
    defer gpa.free(view_valid_counts);
    try std.testing.expectEqualSlices(usize, &.{ 3, 2, 3 }, view_valid_counts);
    try std.testing.expectEqual(@as(usize, 1), view.nullCount());
    try std.testing.expectEqual(@as(usize, 8), view.validCount());
    try std.testing.expectEqual(@as(usize, 9), view.cellCount());
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 9.0), view.nullRatio(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 8.0 / 9.0), view.validRatio(), 1e-12);
    const view_null_ratios = try view.columnNullRatios(gpa);
    defer gpa.free(view_null_ratios);
    try expectF64SliceApproxOrNan(&.{ 0.0, 1.0 / 3.0, 0.0 }, view_null_ratios);
    const view_valid_ratios = try view.columnValidRatios(gpa);
    defer gpa.free(view_valid_ratios);
    try expectF64SliceApproxOrNan(&.{ 1.0, 2.0 / 3.0, 1.0 }, view_valid_ratios);
    const view_nullable_mask = try view.columnNullableMask(gpa);
    defer gpa.free(view_nullable_mask);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false }, view_nullable_mask);
    try std.testing.expectEqual(@as(usize, 1), view.nullableColumnCount());
    try std.testing.expectEqual(@as(usize, 2), view.nonNullableColumnCount());
    const view_has_nulls = try view.columnHasNullsMask(gpa);
    defer gpa.free(view_has_nulls);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false }, view_has_nulls);
    try std.testing.expectEqual(@as(usize, 1), view.columnsWithNullsCount());
    try std.testing.expectEqual(@as(usize, 2), view.columnsWithoutNullsCount());
    const view_data_nbytes = try view.columnDataNbytes(gpa);
    defer gpa.free(view_data_nbytes);
    try std.testing.expectEqualSlices(usize, &.{ sales_column_view.dataNbytes(), units_column_view.dataNbytes(), active_column_view.dataNbytes() }, view_data_nbytes);
    const view_validity_nbytes = try view.columnValidityNbytes(gpa);
    defer gpa.free(view_validity_nbytes);
    try std.testing.expectEqualSlices(usize, &.{ sales_column_view.validityNbytes(), units_column_view.validityNbytes(), active_column_view.validityNbytes() }, view_validity_nbytes);
    const view_total_nbytes = try view.columnTotalNbytes(gpa);
    defer gpa.free(view_total_nbytes);
    try std.testing.expectEqualSlices(usize, &.{ sales_column_view.totalNbytes(), units_column_view.totalNbytes(), active_column_view.totalNbytes() }, view_total_nbytes);
    try std.testing.expectEqual(sales_column_view.dataNbytes() + units_column_view.dataNbytes() + active_column_view.dataNbytes(), view.dataNbytes());
    try std.testing.expectEqual(view.dataNbytes(), view.dataMemoryUsage());
    try std.testing.expectEqual(sales_column_view.validityNbytes() + units_column_view.validityNbytes() + active_column_view.validityNbytes(), view.validityNbytes());
    try std.testing.expectEqual(view.validityNbytes(), view.validityMemoryUsage());
    try std.testing.expectEqual(view.dataNbytes() + view.validityNbytes(), view.totalNbytes());
    try std.testing.expectEqual(view.totalNbytes(), view.memoryUsage());
    try std.testing.expectEqual(view.totalNbytes(), view.estimatedSize());
    const sales_schema = try view.columnSchema("sales");
    try std.testing.expect(std.mem.eql(u8, "sales", sales_schema.name));
    try std.testing.expectEqual(DeviceDType.f64, sales_schema.dtype);
    try std.testing.expectEqual(@as(usize, 3), sales_schema.rows);
    try std.testing.expectEqual(@as(usize, 3), sales_schema.len());
    try std.testing.expectEqual(sales_schema.len(), sales_schema.rowCount());
    try std.testing.expectEqual(sales_schema.len(), sales_schema.height());
    try std.testing.expectEqual(sales_schema.len(), sales_schema.nRows());
    try std.testing.expectEqual(@as(usize, 3), sales_schema.shape().rows);
    try std.testing.expect(sales_schema.shapeEquals(3));
    try std.testing.expect(sales_schema.hasShape(3));
    try std.testing.expect(!sales_schema.isEmpty());
    try std.testing.expect(sales_schema.isNonEmpty());
    try std.testing.expect(sales_schema.hasRows());
    try std.testing.expectEqual(sales_schema.len(), sales_schema.cellCount());
    try std.testing.expect(sales_schema.isFloat());
    try std.testing.expect(!sales_schema.isComplex());
    try std.testing.expect(!sales_schema.nullableColumn());
    try std.testing.expect(sales_schema.allValid());
    try std.testing.expect(!sales_schema.anyNull());
    try std.testing.expect(sales_schema.anyValid());
    try std.testing.expectEqual(sales_column_view.dataNbytes(), sales_schema.dataMemoryUsage());
    try std.testing.expectEqual(sales_column_view.dataPtr(), sales_schema.dataPtr());
    try std.testing.expect(!sales_schema.hasValidity());
    try std.testing.expectEqual(sales_column_view.totalNbytes(), sales_schema.totalNbytes());
    const units_view_schema = try view.columnSchemaAt(1);
    try std.testing.expect(std.mem.eql(u8, "units", units_view_schema.name));
    try std.testing.expect(units_view_schema.nullableColumn());
    try std.testing.expect(units_view_schema.hasNulls());
    try std.testing.expectEqual(@as(usize, 1), units_view_schema.null_count);
    try std.testing.expectEqual(units_column_view.totalNbytes(), units_view_schema.estimatedSize());
    try std.testing.expect(units_view_schema.sameDevice(sales_schema));
    try std.testing.expect(units_view_schema.sameLength(sales_schema));
    try std.testing.expect(units_view_schema.sameShape(sales_schema));
    try std.testing.expect(!units_view_schema.sameDType(sales_schema));
    try std.testing.expect(!units_view_schema.sameNullability(sales_schema));
    try std.testing.expect(!units_view_schema.sameStorage(sales_schema));
    try std.testing.expect(units_view_schema.schemaEquals(units_schema));
    try std.testing.expect(!sales_schema.schemaEquals(units_view_schema));
    const view_schema = try view.schema(gpa);
    defer gpa.free(view_schema);
    try std.testing.expectEqual(@as(usize, 3), view_schema.len);
    try std.testing.expectEqual(DeviceDType.bool, view_schema[2].dtype);
    const view_schema_summary = try view.schemaSummary(gpa);
    defer gpa.free(view_schema_summary);
    try std.testing.expectEqual(@as(usize, 3), view_schema_summary.len);
    try std.testing.expectError(error.ColumnNotFound, view.columnSchema("missing"));
    try std.testing.expectError(error.IndexOutOfBounds, view.columnSchemaAt(3));
    try std.testing.expectError(error.ColumnNotFound, view.columnDType("missing"));
    try std.testing.expectError(error.IndexOutOfBounds, view.columnDTypeAt(3));

    var selected = try table.select(&.{"sales"});
    defer selected.deinit();
    try std.testing.expectEqual(@as(usize, 1), selected.width());
    try std.testing.expectEqual(DeviceDType.f64, try selected.columnDType("sales"));
    var selected_view = try selected.view();
    defer selected_view.deinit();
    try std.testing.expect(view.sameDevice(selected_view));
    try std.testing.expect(view.sameHeight(selected_view));
    try std.testing.expect(!view.sameWidth(selected_view));
    try std.testing.expect(!view.sameShape(selected_view));
    try std.testing.expect(view.sameStorage(view));
    try std.testing.expect(!view.sameStorage(selected_view));
    try std.testing.expect(!view.schemaEquals(selected_view));

    var table_clone_view = try table_clone.view();
    defer table_clone_view.deinit();
    try std.testing.expect(view.schemaEquals(table_clone_view));
    try std.testing.expect(view.sameSchema(table_clone_view));
    try std.testing.expect(view.schemaCompatible(table_clone_view));

    var reordered_view = try reordered.view();
    defer reordered_view.deinit();
    try std.testing.expect(!view.schemaEquals(reordered_view));

    var different_nulls_view = try different_nulls.view();
    defer different_nulls_view.deinit();
    try std.testing.expect(!view.schemaEquals(different_nulls_view));

    var duplicate_view = try duplicate_name_table.view();
    defer duplicate_view.deinit();
    try std.testing.expect(!duplicate_view.columnNamesUnique());
    try std.testing.expect(duplicate_view.hasDuplicateColumnNames());
    try std.testing.expectEqual(@as(usize, 1), duplicate_view.duplicateColumnNameCount());

    var no_columns = try table.select(&.{});
    defer no_columns.deinit();
    try std.testing.expect(no_columns.isEmpty());
    try std.testing.expect(no_columns.hasRows());
    try std.testing.expect(!no_columns.hasColumns());
    try std.testing.expect(!table.sameShape(no_columns));
    try std.testing.expect(table.sameHeight(no_columns));
    try std.testing.expect(!table.sameWidth(no_columns));

    var no_rows = try table.head(0);
    defer no_rows.deinit();
    try std.testing.expect(no_rows.isEmpty());
    try std.testing.expect(!no_rows.hasRows());
    try std.testing.expect(no_rows.hasColumns());
    try std.testing.expect(!table.sameShape(no_rows));
    try std.testing.expect(!table.sameHeight(no_rows));
    try std.testing.expect(table.sameWidth(no_rows));

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

    var except_units = try table.selectExcept(&.{"units"});
    defer except_units.deinit();
    try std.testing.expectEqual(@as(usize, 2), except_units.width());
    try std.testing.expectEqual(@as(?usize, 0), except_units.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), except_units.columnIndex("active"));

    var all_except_sales = try table.selectAllExcept(&.{"sales"});
    defer all_except_sales.deinit();
    try std.testing.expectEqual(@as(usize, 2), all_except_sales.width());
    try std.testing.expectEqual(@as(?usize, 0), all_except_sales.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, 1), all_except_sales.columnIndex("active"));

    var exclude_active = try table.excludeColumns(&.{"active"});
    defer exclude_active.deinit();
    try std.testing.expectEqual(@as(usize, 2), exclude_active.width());
    try std.testing.expectEqual(@as(?usize, 0), exclude_active.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), exclude_active.columnIndex("units"));

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
    var units_filled_expr = try table.withColumnFillNull("units_filled", "units", i64, -1);
    defer units_filled_expr.deinit();
    const original_units_validity = try (try units_filled_expr.column("units")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(original_units_validity);
    const units_filled_values = try (try units_filled_expr.column("units_filled")).i64.toOwnedSlice(gpa);
    defer gpa.free(units_filled_values);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, original_units_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, -1, 3 }, units_filled_values);
    try std.testing.expectEqual(@as(usize, 0), (try units_filled_expr.column("units_filled")).nullCount());
    try std.testing.expectError(error.TypeUnsupported, table.withColumnFillNull("bad_units_filled", "units", f64, 0.0));

    var directional_units = try DeviceColumn.fromSliceWithValidity(i64, gpa, &.{ 10, 20, 30, 40 }, &.{ false, true, false, false }, .cpu);
    defer directional_units.deinit();
    var directional_table = try DeviceDataFrame.init(gpa, &.{.{ .name = "units", .data = directional_units }});
    defer directional_table.deinit();
    var forward_filled = try directional_table.withColumnFillNullForward("units_ffill", "units");
    defer forward_filled.deinit();
    const forward_values = try (try forward_filled.column("units_ffill")).i64.toOwnedSlice(gpa);
    defer gpa.free(forward_values);
    const forward_validity = try (try forward_filled.column("units_ffill")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(forward_validity);
    try std.testing.expectEqualSlices(i64, &.{ 10, 20, 20, 20 }, forward_values);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, forward_validity);
    var backward_filled = try directional_table.withColumnFillNullBackward("units_bfill", "units");
    defer backward_filled.deinit();
    const backward_values = try (try backward_filled.column("units_bfill")).i64.toOwnedSlice(gpa);
    defer gpa.free(backward_values);
    const backward_validity = try (try backward_filled.column("units_bfill")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(backward_validity);
    try std.testing.expectEqualSlices(i64, &.{ 20, 20, 30, 40 }, backward_values);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false }, backward_validity);

    var units_null_if_expr = try table.withColumnNullIf("units_without_one", "units", i64, 1);
    defer units_null_if_expr.deinit();
    const units_without_one = try (try units_null_if_expr.column("units_without_one")).i64.toOwnedSlice(gpa);
    defer gpa.free(units_without_one);
    const units_without_one_validity = try (try units_null_if_expr.column("units_without_one")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(units_without_one_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3 }, units_without_one);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true }, units_without_one_validity);
    var units_null_if_in_place = try table.nullIfColumn("units", i64, 3);
    defer units_null_if_in_place.deinit();
    const units_null_if_in_place_validity = try (try units_null_if_in_place.column("units")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(units_null_if_in_place_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false }, units_null_if_in_place_validity);

    var units_null_if_values_expr = try table.withColumnNullIfValues("units_without_values", "units", i64, &.{ 1, 3 });
    defer units_null_if_values_expr.deinit();
    const units_without_values_validity = try (try units_null_if_values_expr.column("units_without_values")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(units_without_values_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false }, units_without_values_validity);

    var units_null_if_values_in_place = try table.nullIfValuesColumn("units", i64, &.{1});
    defer units_null_if_values_in_place.deinit();
    const units_null_if_values_in_place_validity = try (try units_null_if_values_in_place.column("units")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(units_null_if_values_in_place_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true }, units_null_if_values_in_place_validity);
    try std.testing.expectError(error.TypeUnsupported, table.withColumnNullIf("bad_null_if", "units", f64, 1.0));
    try std.testing.expectError(error.TypeUnsupported, table.withColumnNullIfValues("bad_null_if_values", "units", f64, &.{1.0}));
    try std.testing.expectError(error.TypeUnsupported, table.fillNullColumn("units", f64, 0.0));
    try std.testing.expectError(error.ColumnNotFound, table.fillNullColumn("missing", i64, 0));

    var fallback_units_col = try DeviceColumn.fromSliceWithValidity(i64, gpa, &.{ 10, 20, 30 }, &.{ true, true, false }, .cpu);
    defer fallback_units_col.deinit();
    var fallback_table = try table.withColumn("fallback_units", fallback_units_col);
    defer fallback_table.deinit();
    var empty_fallback_units_col = try DeviceColumn.fromSliceWithValidity(i64, gpa, &.{ 10, 20, 30 }, &.{ false, false, false }, .cpu);
    defer empty_fallback_units_col.deinit();
    var second_fallback_units_col = try DeviceColumn.fromSliceWithValidity(i64, gpa, &.{ 100, 200, 300 }, &.{ false, true, true }, .cpu);
    defer second_fallback_units_col.deinit();
    var multi_fallback_table = try fallback_table.withColumn("empty_fallback_units", empty_fallback_units_col);
    defer multi_fallback_table.deinit();
    var multi_source_table = try multi_fallback_table.withColumn("second_fallback_units", second_fallback_units_col);
    defer multi_source_table.deinit();
    var coalesced_units = try fallback_table.coalesceColumns("units", "fallback_units", "units_coalesced");
    defer coalesced_units.deinit();
    try std.testing.expectEqual(DeviceDType.i64, try coalesced_units.columnDType("units_coalesced"));
    try std.testing.expectEqual(@as(usize, 0), (try coalesced_units.column("units_coalesced")).nullCount());
    const coalesced_values = try (try coalesced_units.column("units_coalesced")).i64.toOwnedSlice(gpa);
    defer gpa.free(coalesced_values);
    try std.testing.expectEqualSlices(i64, &.{ 1, 20, 3 }, coalesced_values);
    var coalesced_many_units = try multi_source_table.coalesceColumnsMany(&.{ "units", "empty_fallback_units", "second_fallback_units" }, "units_coalesced_many");
    defer coalesced_many_units.deinit();
    try std.testing.expectEqual(@as(usize, 0), (try coalesced_many_units.column("units_coalesced_many")).nullCount());
    const coalesced_many_values = try (try coalesced_many_units.column("units_coalesced_many")).i64.toOwnedSlice(gpa);
    defer gpa.free(coalesced_many_values);
    try std.testing.expectEqualSlices(i64, &.{ 1, 200, 3 }, coalesced_many_values);
    var coalesced_alias_units = try multi_source_table.coalesceManyColumns(&.{ "empty_fallback_units", "second_fallback_units" }, "units_coalesced_alias");
    defer coalesced_alias_units.deinit();
    const coalesced_alias = try (try coalesced_alias_units.column("units_coalesced_alias")).i64.toOwnedSlice(gpa);
    defer gpa.free(coalesced_alias);
    const coalesced_alias_validity = try (try coalesced_alias_units.column("units_coalesced_alias")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(coalesced_alias_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 200, 300 }, coalesced_alias);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true }, coalesced_alias_validity);
    try std.testing.expectError(error.LengthMismatch, multi_source_table.coalesceFirstValidColumns(&.{}, "bad_empty_coalesce"));
    try std.testing.expectError(error.TypeMismatch, multi_source_table.coalesceColumnsMany(&.{ "units", "sales" }, "bad_type_coalesce"));
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

    var row_any_nulls = try table.withRowAnyNull(&.{ "sales", "units", "active" }, "row_any_null");
    defer row_any_nulls.deinit();
    const row_any_null = try (try row_any_nulls.column("row_any_null")).bool.toOwnedSlice(gpa);
    defer gpa.free(row_any_null);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false }, row_any_null);

    var row_all_nulls = try table.withRowAllNull(&.{ "sales", "units", "active" }, "row_all_null");
    defer row_all_nulls.deinit();
    const row_all_null = try (try row_all_nulls.column("row_all_null")).bool.toOwnedSlice(gpa);
    defer gpa.free(row_all_null);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false }, row_all_null);

    var row_any_valids = try table.withRowAnyValid(&.{ "sales", "units", "active" }, "row_any_valid");
    defer row_any_valids.deinit();
    const row_any_valid = try (try row_any_valids.column("row_any_valid")).bool.toOwnedSlice(gpa);
    defer gpa.free(row_any_valid);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true }, row_any_valid);

    var row_all_valids = try table.withRowAllValid(&.{ "sales", "units", "active" }, "row_all_valid");
    defer row_all_valids.deinit();
    const row_all_valid = try (try row_all_valids.column("row_all_valid")).bool.toOwnedSlice(gpa);
    defer gpa.free(row_all_valid);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, row_all_valid);

    var row_cum_valid_counts = try table.withRowCumulativeValidCount(
        &.{ "sales", "units", "active" },
        &.{ "sales_cum_valid", "units_cum_valid", "active_cum_valid" },
    );
    defer row_cum_valid_counts.deinit();
    try std.testing.expectEqual(DeviceDType.i64, try row_cum_valid_counts.columnDType("sales_cum_valid"));
    const sales_cum_valid = try (try row_cum_valid_counts.column("sales_cum_valid")).i64.toOwnedSlice(gpa);
    defer gpa.free(sales_cum_valid);
    const units_cum_valid = try (try row_cum_valid_counts.column("units_cum_valid")).i64.toOwnedSlice(gpa);
    defer gpa.free(units_cum_valid);
    const active_cum_valid = try (try row_cum_valid_counts.column("active_cum_valid")).i64.toOwnedSlice(gpa);
    defer gpa.free(active_cum_valid);
    try std.testing.expectEqualSlices(i64, &.{ 1, 1, 1 }, sales_cum_valid);
    try std.testing.expectEqualSlices(i64, &.{ 2, 1, 2 }, units_cum_valid);
    try std.testing.expectEqualSlices(i64, &.{ 3, 2, 3 }, active_cum_valid);

    var row_cum_any_nulls = try table.withRowCumulativeAnyNull(
        &.{ "sales", "units", "active" },
        &.{ "sales_cum_any_null", "units_cum_any_null", "active_cum_any_null" },
    );
    defer row_cum_any_nulls.deinit();
    const sales_cum_any_null = try (try row_cum_any_nulls.column("sales_cum_any_null")).bool.toOwnedSlice(gpa);
    defer gpa.free(sales_cum_any_null);
    const units_cum_any_null = try (try row_cum_any_nulls.column("units_cum_any_null")).bool.toOwnedSlice(gpa);
    defer gpa.free(units_cum_any_null);
    const active_cum_any_null = try (try row_cum_any_nulls.column("active_cum_any_null")).bool.toOwnedSlice(gpa);
    defer gpa.free(active_cum_any_null);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false }, sales_cum_any_null);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false }, units_cum_any_null);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false }, active_cum_any_null);

    var row_prefix_all_valids = try table.withRowPrefixAllValid(
        &.{ "sales", "units", "active" },
        &.{ "sales_prefix_all_valid", "units_prefix_all_valid", "active_prefix_all_valid" },
    );
    defer row_prefix_all_valids.deinit();
    const sales_prefix_all_valid = try (try row_prefix_all_valids.column("sales_prefix_all_valid")).bool.toOwnedSlice(gpa);
    defer gpa.free(sales_prefix_all_valid);
    const units_prefix_all_valid = try (try row_prefix_all_valids.column("units_prefix_all_valid")).bool.toOwnedSlice(gpa);
    defer gpa.free(units_prefix_all_valid);
    const active_prefix_all_valid = try (try row_prefix_all_valids.column("active_prefix_all_valid")).bool.toOwnedSlice(gpa);
    defer gpa.free(active_prefix_all_valid);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true }, sales_prefix_all_valid);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, units_prefix_all_valid);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, active_prefix_all_valid);

    var row_cum_null_counts = try table.withRowPrefixNullCount(
        &.{ "sales", "units", "active" },
        &.{ "sales_cum_null", "units_cum_null", "active_cum_null" },
    );
    defer row_cum_null_counts.deinit();
    const sales_cum_null = try (try row_cum_null_counts.column("sales_cum_null")).i64.toOwnedSlice(gpa);
    defer gpa.free(sales_cum_null);
    const units_cum_null = try (try row_cum_null_counts.column("units_cum_null")).i64.toOwnedSlice(gpa);
    defer gpa.free(units_cum_null);
    const active_cum_null = try (try row_cum_null_counts.column("active_cum_null")).i64.toOwnedSlice(gpa);
    defer gpa.free(active_cum_null);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0 }, sales_cum_null);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0 }, units_cum_null);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0 }, active_cum_null);

    var row_cum_valid_ratios = try table.withRowCumulativeValidRatio(
        &.{ "sales", "units", "active" },
        &.{ "sales_cum_valid_ratio", "units_cum_valid_ratio", "active_cum_valid_ratio" },
    );
    defer row_cum_valid_ratios.deinit();
    const sales_cum_valid_ratio = try (try row_cum_valid_ratios.column("sales_cum_valid_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(sales_cum_valid_ratio);
    const units_cum_valid_ratio = try (try row_cum_valid_ratios.column("units_cum_valid_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(units_cum_valid_ratio);
    const active_cum_valid_ratio = try (try row_cum_valid_ratios.column("active_cum_valid_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(active_cum_valid_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 1.0, 1.0 }, sales_cum_valid_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 0.5, 1.0 }, units_cum_valid_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 2.0 / 3.0, 1.0 }, active_cum_valid_ratio);

    var row_cum_null_ratios = try table.withRowPrefixNullRatio(
        &.{ "sales", "units", "active" },
        &.{ "sales_cum_null_ratio", "units_cum_null_ratio", "active_cum_null_ratio" },
    );
    defer row_cum_null_ratios.deinit();
    const sales_cum_null_ratio = try (try row_cum_null_ratios.column("sales_cum_null_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(sales_cum_null_ratio);
    const units_cum_null_ratio = try (try row_cum_null_ratios.column("units_cum_null_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(units_cum_null_ratio);
    const active_cum_null_ratio = try (try row_cum_null_ratios.column("active_cum_null_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(active_cum_null_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.0 }, sales_cum_null_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.5, 0.0 }, units_cum_null_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 1.0 / 3.0, 0.0 }, active_cum_null_ratio);
    try std.testing.expectError(error.LengthMismatch, table.withRowPrefixValidCount(&.{"sales"}, &.{ "sales_cum_valid", "extra_cum_valid" }));
    try std.testing.expectError(error.LengthMismatch, table.withRowPrefixValidRatio(&.{"sales"}, &.{ "sales_cum_valid_ratio", "extra_cum_valid_ratio" }));

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

    var row_cum_first_valid = try table.withRowCumulativeFirstValidIndex(
        &.{ "sales", "units", "active" },
        &.{ "sales_first_valid", "units_first_valid", "active_first_valid" },
    );
    defer row_cum_first_valid.deinit();
    const sales_first_valid_column = try row_cum_first_valid.column("sales_first_valid");
    try std.testing.expect(sales_first_valid_column.i64.nullable());
    const sales_first_valid = try sales_first_valid_column.i64.toOwnedSlice(gpa);
    defer gpa.free(sales_first_valid);
    const sales_first_valid_validity = try sales_first_valid_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(sales_first_valid_validity);
    const units_first_valid = try (try row_cum_first_valid.column("units_first_valid")).i64.toOwnedSlice(gpa);
    defer gpa.free(units_first_valid);
    const active_first_valid = try (try row_cum_first_valid.column("active_first_valid")).i64.toOwnedSlice(gpa);
    defer gpa.free(active_first_valid);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0 }, sales_first_valid);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0 }, units_first_valid);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0 }, active_first_valid);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true }, sales_first_valid_validity);

    var row_cum_last_null = try table.withRowPrefixLastNullIndex(
        &.{ "sales", "units", "active" },
        &.{ "sales_last_null", "units_last_null", "active_last_null" },
    );
    defer row_cum_last_null.deinit();
    const sales_last_null_column = try row_cum_last_null.column("sales_last_null");
    const sales_last_null = try sales_last_null_column.i64.toOwnedSlice(gpa);
    defer gpa.free(sales_last_null);
    const sales_last_null_validity = try sales_last_null_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(sales_last_null_validity);
    const units_last_null_column = try row_cum_last_null.column("units_last_null");
    const units_last_null = try units_last_null_column.i64.toOwnedSlice(gpa);
    defer gpa.free(units_last_null);
    const units_last_null_validity = try units_last_null_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(units_last_null_validity);
    const active_last_null = try (try row_cum_last_null.column("active_last_null")).i64.toOwnedSlice(gpa);
    defer gpa.free(active_last_null);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0 }, sales_last_null);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false }, sales_last_null_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0 }, units_last_null);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false }, units_last_null_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0 }, active_last_null);
    try std.testing.expectError(error.LengthMismatch, table.withRowPrefixFirstValidIndex(&.{"sales"}, &.{ "sales_first_valid", "extra_first_valid" }));

    var bool_alt = try DeviceColumn.fromSliceWithValidity(bool, gpa, &.{ false, true, false }, &.{ true, false, true }, .cpu);
    defer bool_alt.deinit();
    var bool_table = try table.withColumn("bool_alt", bool_alt);
    defer bool_table.deinit();
    var row_cum_true_counts = try bool_table.withRowCumulativeTrueCount(&.{ "active", "bool_alt" }, &.{ "active_cum_true", "alt_cum_true" });
    defer row_cum_true_counts.deinit();
    const active_cum_true = try (try row_cum_true_counts.column("active_cum_true")).i64.toOwnedSlice(gpa);
    defer gpa.free(active_cum_true);
    const alt_cum_true = try (try row_cum_true_counts.column("alt_cum_true")).i64.toOwnedSlice(gpa);
    defer gpa.free(alt_cum_true);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 1 }, active_cum_true);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 1 }, alt_cum_true);
    var row_cum_false_counts = try bool_table.withRowPrefixFalseCount(&.{ "active", "bool_alt" }, &.{ "active_cum_false", "alt_cum_false" });
    defer row_cum_false_counts.deinit();
    const active_cum_false = try (try row_cum_false_counts.column("active_cum_false")).i64.toOwnedSlice(gpa);
    defer gpa.free(active_cum_false);
    const alt_cum_false = try (try row_cum_false_counts.column("alt_cum_false")).i64.toOwnedSlice(gpa);
    defer gpa.free(alt_cum_false);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0 }, active_cum_false);
    try std.testing.expectEqualSlices(i64, &.{ 1, 1, 1 }, alt_cum_false);
    var row_cum_true_ratios = try bool_table.withRowCumulativeTrueRatio(&.{ "active", "bool_alt" }, &.{ "active_cum_true_ratio", "alt_cum_true_ratio" });
    defer row_cum_true_ratios.deinit();
    const active_cum_true_ratio = try (try row_cum_true_ratios.column("active_cum_true_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(active_cum_true_ratio);
    const alt_cum_true_ratio = try (try row_cum_true_ratios.column("alt_cum_true_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(alt_cum_true_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 0.0, 1.0 }, active_cum_true_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 0.5, 0.0, 0.5 }, alt_cum_true_ratio);
    var row_cum_false_ratios = try bool_table.withRowPrefixFalseRatio(&.{ "active", "bool_alt" }, &.{ "active_cum_false_ratio", "alt_cum_false_ratio" });
    defer row_cum_false_ratios.deinit();
    const active_cum_false_ratio = try (try row_cum_false_ratios.column("active_cum_false_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(active_cum_false_ratio);
    const alt_cum_false_ratio = try (try row_cum_false_ratios.column("alt_cum_false_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(alt_cum_false_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 1.0, 0.0 }, active_cum_false_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 0.5, 0.5, 0.5 }, alt_cum_false_ratio);
    try std.testing.expectError(error.LengthMismatch, bool_table.withRowPrefixTrueCount(&.{"active"}, &.{ "active_cum_true", "extra_cum_true" }));
    var row_cum_any_true = try bool_table.withRowCumulativeAnyTrue(&.{ "active", "bool_alt" }, &.{ "active_cum_any_true", "alt_cum_any_true" });
    defer row_cum_any_true.deinit();
    const active_cum_any_true_column = try row_cum_any_true.column("active_cum_any_true");
    try std.testing.expect(active_cum_any_true_column.bool.nullable());
    const active_cum_any_true = try active_cum_any_true_column.bool.toOwnedSlice(gpa);
    defer gpa.free(active_cum_any_true);
    const active_cum_any_true_validity = try active_cum_any_true_column.bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(active_cum_any_true_validity);
    const alt_cum_any_true_column = try row_cum_any_true.column("alt_cum_any_true");
    const alt_cum_any_true = try alt_cum_any_true_column.bool.toOwnedSlice(gpa);
    defer gpa.free(alt_cum_any_true);
    const alt_cum_any_true_validity = try alt_cum_any_true_column.bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(alt_cum_any_true_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, active_cum_any_true);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, alt_cum_any_true);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true }, active_cum_any_true_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, alt_cum_any_true_validity);

    var row_cum_all_true = try bool_table.withRowPrefixAllTrue(&.{ "active", "bool_alt" }, &.{ "active_cum_all_true", "alt_cum_all_true" });
    defer row_cum_all_true.deinit();
    const active_cum_all_true = try (try row_cum_all_true.column("active_cum_all_true")).bool.toOwnedSlice(gpa);
    defer gpa.free(active_cum_all_true);
    const alt_cum_all_true_column = try row_cum_all_true.column("alt_cum_all_true");
    const alt_cum_all_true = try alt_cum_all_true_column.bool.toOwnedSlice(gpa);
    defer gpa.free(alt_cum_all_true);
    const alt_cum_all_true_validity = try alt_cum_all_true_column.bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(alt_cum_all_true_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, active_cum_all_true);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false }, alt_cum_all_true);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, alt_cum_all_true_validity);

    var row_cum_any_false = try bool_table.withRowCumulativeAnyFalse(&.{ "active", "bool_alt" }, &.{ "active_cum_any_false", "alt_cum_any_false" });
    defer row_cum_any_false.deinit();
    const alt_cum_any_false = try (try row_cum_any_false.column("alt_cum_any_false")).bool.toOwnedSlice(gpa);
    defer gpa.free(alt_cum_any_false);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, alt_cum_any_false);

    var row_cum_all_false = try bool_table.withRowPrefixAllFalse(&.{ "active", "bool_alt" }, &.{ "active_cum_all_false", "alt_cum_all_false" });
    defer row_cum_all_false.deinit();
    const alt_cum_all_false = try (try row_cum_all_false.column("alt_cum_all_false")).bool.toOwnedSlice(gpa);
    defer gpa.free(alt_cum_all_false);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false }, alt_cum_all_false);

    try std.testing.expectError(error.LengthMismatch, bool_table.withRowPrefixTrueRatio(&.{"active"}, &.{ "active_cum_true_ratio", "extra_cum_true_ratio" }));
    try std.testing.expectError(error.LengthMismatch, bool_table.withRowPrefixAnyTrue(&.{"active"}, &.{ "active_cum_any_true", "extra_cum_any_true" }));
    try std.testing.expectError(error.TypeMismatch, table.withRowCumulativeTrueCount(&.{"sales"}, &.{"sales_cum_true"}));
    var row_cum_first_true = try bool_table.withRowCumulativeFirstTrueIndex(&.{ "active", "bool_alt" }, &.{ "active_first_true", "alt_first_true" });
    defer row_cum_first_true.deinit();
    const active_first_true = try (try row_cum_first_true.column("active_first_true")).i64.toOwnedSlice(gpa);
    defer gpa.free(active_first_true);
    const alt_first_true_column = try row_cum_first_true.column("alt_first_true");
    const alt_first_true = try alt_first_true_column.i64.toOwnedSlice(gpa);
    defer gpa.free(alt_first_true);
    const alt_first_true_validity = try alt_first_true_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(alt_first_true_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0 }, active_first_true);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0 }, alt_first_true);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, alt_first_true_validity);

    var row_cum_last_false = try bool_table.withRowPrefixLastFalseIndex(&.{ "active", "bool_alt" }, &.{ "active_last_false", "alt_last_false" });
    defer row_cum_last_false.deinit();
    const active_last_false = try (try row_cum_last_false.column("active_last_false")).i64.toOwnedSlice(gpa);
    defer gpa.free(active_last_false);
    const alt_last_false_column = try row_cum_last_false.column("alt_last_false");
    const alt_last_false = try alt_last_false_column.i64.toOwnedSlice(gpa);
    defer gpa.free(alt_last_false);
    const alt_last_false_validity = try alt_last_false_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(alt_last_false_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0 }, active_last_false);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 1 }, alt_last_false);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true }, alt_last_false_validity);

    try std.testing.expectError(error.TypeMismatch, table.withRowCumulativeTrueRatio(&.{"sales"}, &.{"sales_cum_true_ratio"}));
    try std.testing.expectError(error.TypeMismatch, table.withRowCumulativeAnyTrue(&.{"sales"}, &.{"sales_cum_any_true"}));
    try std.testing.expectError(error.LengthMismatch, bool_table.withRowPrefixFirstTrueIndex(&.{"active"}, &.{ "active_first_true", "extra_first_true" }));
    try std.testing.expectError(error.TypeMismatch, table.withRowCumulativeFirstTrueIndex(&.{"sales"}, &.{"sales_first_true"}));

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

    var row_weighted_pair_weight_sum_table = try validity_table.withRowWeightedPairWeightSum(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_pair_weight_sum");
    defer row_weighted_pair_weight_sum_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_pair_weight_sum_table, gpa, "row_weighted_pair_weight_sum", &.{ 1.0, 1.0, 0.0, 5.0 }, &.{ true, true, false, true });

    var row_weighted_pair_positive_count_table = try validity_table.withRowWeightedPairPositiveCount(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_pair_positive_count");
    defer row_weighted_pair_positive_count_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_pair_positive_count_table, gpa, "row_weighted_pair_positive_count", &.{ 1.0, 1.0, 0.0, 2.0 }, &.{ true, true, false, true });

    var row_weighted_pair_effective_n_table = try validity_table.withRowWeightedPairEffectiveCount(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_pair_effective_n");
    defer row_weighted_pair_effective_n_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_pair_effective_n_table, gpa, "row_weighted_pair_effective_n", &.{ 1.0, 1.0, 0.0, 25.0 / 17.0 }, &.{ true, true, false, true });

    var row_cum_weighted_pair_weight_sum_table = try validity_table.withRowCumulativeWeightedPairWeightSum(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_pair_cum_weight_sum", "b_row_weighted_pair_cum_weight_sum" });
    defer row_cum_weighted_pair_weight_sum_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_pair_weight_sum_table, gpa, "a_row_weighted_pair_cum_weight_sum", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_pair_weight_sum_table, gpa, "b_row_weighted_pair_cum_weight_sum", &.{ 0.0, 1.0, 0.0, 5.0 }, &.{ false, true, false, true });

    var row_cum_weighted_pair_positive_count_table = try validity_table.withRowCumWeightedPairPositiveCount(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_pair_cum_positive_count", "b_row_weighted_pair_cum_positive_count" });
    defer row_cum_weighted_pair_positive_count_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_pair_positive_count_table, gpa, "a_row_weighted_pair_cum_positive_count", &.{ 1.0, 0.0, 0.0, 1.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_pair_positive_count_table, gpa, "b_row_weighted_pair_cum_positive_count", &.{ 0.0, 1.0, 0.0, 2.0 }, &.{ false, true, false, true });

    var row_cum_weighted_pair_effective_n_table = try validity_table.withRowPrefixWeightedPairEffectiveCount(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_pair_cum_effective_n", "b_row_weighted_pair_cum_effective_n" });
    defer row_cum_weighted_pair_effective_n_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_pair_effective_n_table, gpa, "a_row_weighted_pair_cum_effective_n", &.{ 1.0, 0.0, 0.0, 1.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_pair_effective_n_table, gpa, "b_row_weighted_pair_cum_effective_n", &.{ 0.0, 1.0, 0.0, 25.0 / 17.0 }, &.{ false, true, false, true });

    var row_cum_weighted_dot_table = try validity_table.withRowPrefixWeightedDot(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumdot", "b_row_weighted_cumdot" });
    defer row_cum_weighted_dot_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_dot_table, gpa, "a_row_weighted_cumdot", &.{ 1.0, 0.0, 0.0, 64.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_dot_table, gpa, "b_row_weighted_cumdot", &.{ 0.0, 20.0, 0.0, 104.0 }, &.{ false, true, false, true });

    var row_cum_weighted_cosine_table = try validity_table.withRowCumWeightedCosine(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumcosine", "b_row_weighted_cumcosine" });
    defer row_cum_weighted_cosine_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_cosine_table, gpa, "a_row_weighted_cumcosine", &.{ 1.0, 0.0, 0.0, 1.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_cosine_table, gpa, "b_row_weighted_cumcosine", &.{ 0.0, 1.0, 0.0, 104.0 / std.math.sqrt(@as(f64, 1664.0 * 65.0)) }, &.{ false, true, false, true });

    var row_cum_weighted_sqdist_table = try validity_table.withRowPrefixWeightedSquaredEuclideanDistance(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumsqdist", "b_row_weighted_cumsqdist" });
    defer row_cum_weighted_sqdist_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_sqdist_table, gpa, "a_row_weighted_cumsqdist", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_sqdist_table, gpa, "b_row_weighted_cumsqdist", &.{ 0.0, 361.0, 0.0, 1521.0 }, &.{ false, true, false, true });

    var row_cum_weighted_l2_distance_table = try validity_table.withRowCumWeightedL2Distance(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cuml2", "b_row_weighted_cuml2" });
    defer row_cum_weighted_l2_distance_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_l2_distance_table, gpa, "a_row_weighted_cuml2", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_l2_distance_table, gpa, "b_row_weighted_cuml2", &.{ 0.0, 19.0, 0.0, 39.0 }, &.{ false, true, false, true });

    var row_cum_weighted_l1_distance_table = try validity_table.withRowPrefixWeightedL1Distance(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cuml1dist", "b_row_weighted_cuml1dist" });
    defer row_cum_weighted_l1_distance_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_l1_distance_table, gpa, "a_row_weighted_cuml1dist", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_l1_distance_table, gpa, "b_row_weighted_cuml1dist", &.{ 0.0, 19.0, 0.0, 39.0 }, &.{ false, true, false, true });

    var row_cum_weighted_chebyshev_table = try validity_table.withRowPrefixWeightedChebyshevDistance(&.{ "a", "b" }, &.{ "wb", "wa" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumchebyshev", "b_row_weighted_cumchebyshev" });
    defer row_cum_weighted_chebyshev_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_chebyshev_table, gpa, "a_row_weighted_cumchebyshev", &.{ 1.0, 0.0, 0.0, 3.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_chebyshev_table, gpa, "b_row_weighted_cumchebyshev", &.{ 0.0, 18.0, 0.0, 36.0 }, &.{ false, true, false, true });

    var row_cum_weighted_canberra_table = try validity_table.withRowPrefixWeightedCanberraDistance(&.{ "a", "b" }, &.{ "wb", "wa" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumcanberra", "b_row_weighted_cumcanberra" });
    defer row_cum_weighted_canberra_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_canberra_table, gpa, "a_row_weighted_cumcanberra", &.{ 1.0 / 3.0, 0.0, 0.0, 12.0 / 5.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_canberra_table, gpa, "b_row_weighted_cumcanberra", &.{ 0.0, 9.0 / 11.0, 0.0, 177.0 / 55.0 }, &.{ false, true, false, true });

    var row_cum_weighted_bray_table = try validity_table.withRowPrefixWeightedBrayCurtisDistance(&.{ "a", "b" }, &.{ "wb", "wa" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumbray", "b_row_weighted_cumbray" });
    defer row_cum_weighted_bray_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_bray_table, gpa, "a_row_weighted_cumbray", &.{ 1.0 / 3.0, 0.0, 0.0, 3.0 / 5.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_bray_table, gpa, "b_row_weighted_cumbray", &.{ 0.0, 9.0 / 11.0, 0.0, 3.0 / 4.0 }, &.{ false, true, false, true });

    var row_cum_weighted_bias_table = try validity_table.withRowPrefixWeightedBias(&.{ "a", "b" }, &.{ "wb", "wa" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumbias", "b_row_weighted_cumbias" });
    defer row_cum_weighted_bias_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_bias_table, gpa, "a_row_weighted_cumbias", &.{ -1.0, 0.0, 0.0, 3.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_bias_table, gpa, "b_row_weighted_cumbias", &.{ 0.0, 18.0, 0.0, 48.0 / 5.0 }, &.{ false, true, false, true });

    var row_cum_weighted_mae_table = try validity_table.withRowPrefixWeightedMAE(&.{ "a", "b" }, &.{ "wb", "wa" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cummae", "b_row_weighted_cummae" });
    defer row_cum_weighted_mae_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_mae_table, gpa, "a_row_weighted_cummae", &.{ 1.0, 0.0, 0.0, 3.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_mae_table, gpa, "b_row_weighted_cummae", &.{ 0.0, 18.0, 0.0, 48.0 / 5.0 }, &.{ false, true, false, true });

    var row_cum_weighted_mse_table = try validity_table.withRowPrefixWeightedMSE(&.{ "a", "b" }, &.{ "wb", "wa" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cummse", "b_row_weighted_cummse" });
    defer row_cum_weighted_mse_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_mse_table, gpa, "a_row_weighted_cummse", &.{ 1.0, 0.0, 0.0, 9.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_mse_table, gpa, "b_row_weighted_cummse", &.{ 0.0, 324.0, 0.0, 1332.0 / 5.0 }, &.{ false, true, false, true });

    var row_cum_weighted_rmse_table = try validity_table.withRowPrefixWeightedRMSE(&.{ "a", "b" }, &.{ "wb", "wa" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumrmse", "b_row_weighted_cumrmse" });
    defer row_cum_weighted_rmse_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_rmse_table, gpa, "a_row_weighted_cumrmse", &.{ 1.0, 0.0, 0.0, 3.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_rmse_table, gpa, "b_row_weighted_cumrmse", &.{ 0.0, 18.0, 0.0, std.math.sqrt(@as(f64, 1332.0 / 5.0)) }, &.{ false, true, false, true });

    var row_cum_weighted_mape_table = try validity_table.withRowPrefixWeightedMAPE(&.{ "a", "b" }, &.{ "wb", "wa" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cummape", "b_row_weighted_cummape" });
    defer row_cum_weighted_mape_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_mape_table, gpa, "a_row_weighted_cummape", &.{ 1.0, 0.0, 0.0, 3.0 / 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_mape_table, gpa, "b_row_weighted_cummape", &.{ 0.0, 9.0 / 10.0, 0.0, 39.0 / 50.0 }, &.{ false, true, false, true });

    var row_cum_weighted_smape_table = try validity_table.withRowPrefixWeightedSMAPE(&.{ "a", "b" }, &.{ "wb", "wa" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumsmape", "b_row_weighted_cumsmape" });
    defer row_cum_weighted_smape_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_smape_table, gpa, "a_row_weighted_cumsmape", &.{ 2.0 / 3.0, 0.0, 0.0, 6.0 / 5.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_smape_table, gpa, "b_row_weighted_cumsmape", &.{ 0.0, 18.0 / 11.0, 0.0, 354.0 / 275.0 }, &.{ false, true, false, true });

    var row_cum_weighted_cov_table = try validity_table.withRowPrefixWeightedCovariance(&.{ "a", "b" }, &.{ "wb", "wa" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumcov", "b_row_weighted_cumcov" }, 0.0);
    defer row_cum_weighted_cov_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_cov_table, gpa, "a_row_weighted_cumcov", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_cov_table, gpa, "b_row_weighted_cumcov", &.{ 0.0, 0.0, 0.0, 432.0 / 25.0 }, &.{ false, true, false, true });
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowCumulativeWeightedPairWeightSum(&.{"a"}, &.{"wa"}, &.{"wa"}, &.{ "a_row_weighted_pair_cum_weight_sum", "extra_row_weighted_pair_cum_weight_sum" }));
    try std.testing.expectError(error.InvalidShape, validity_table.withRowCumulativeWeightedCovariance(&.{ "a", "b" }, &.{ "wb", "wa" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumcov", "b_row_weighted_cumcov" }, -1.0));

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

    var row_cum_argmin_table = try validity_table.withRowCumulativeArgMin(
        &.{ "a", "b", "wa", "wb" },
        &.{ "a_cum_argmin", "b_cum_argmin", "wa_cum_argmin", "wb_cum_argmin" },
    );
    defer row_cum_argmin_table.deinit();
    const a_cum_argmin_column = try row_cum_argmin_table.column("a_cum_argmin");
    try std.testing.expect(a_cum_argmin_column.i64.nullable());
    const a_cum_argmin = try a_cum_argmin_column.i64.toOwnedSlice(gpa);
    defer gpa.free(a_cum_argmin);
    const a_cum_argmin_validity = try a_cum_argmin_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(a_cum_argmin_validity);
    const b_cum_argmin = try (try row_cum_argmin_table.column("b_cum_argmin")).i64.toOwnedSlice(gpa);
    defer gpa.free(b_cum_argmin);
    const wb_cum_argmin = try (try row_cum_argmin_table.column("wb_cum_argmin")).i64.toOwnedSlice(gpa);
    defer gpa.free(wb_cum_argmin);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0 }, a_cum_argmin);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, a_cum_argmin_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 0 }, b_cum_argmin);
    try std.testing.expectEqualSlices(i64, &.{ 0, 3, 2, 3 }, wb_cum_argmin);

    var row_cum_argmax_table = try validity_table.withRowCumulativeArgMax(
        &.{ "a", "b", "wa", "wb" },
        &.{ "a_cum_argmax", "b_cum_argmax", "wa_cum_argmax", "wb_cum_argmax" },
    );
    defer row_cum_argmax_table.deinit();
    const b_cum_argmax = try (try row_cum_argmax_table.column("b_cum_argmax")).i64.toOwnedSlice(gpa);
    defer gpa.free(b_cum_argmax);
    const wb_cum_argmax = try (try row_cum_argmax_table.column("wb_cum_argmax")).i64.toOwnedSlice(gpa);
    defer gpa.free(wb_cum_argmax);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 1 }, b_cum_argmax);
    try std.testing.expectEqualSlices(i64, &.{ 3, 1, 3, 1 }, wb_cum_argmax);
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowPrefixArgMin(&.{"a"}, &.{ "a_cum_argmin", "extra_cum_argmin" }));

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

    var row_quantile_range_table = try validity_table.withRowQuantileRange(&.{ "a", "b" }, "row_quantile_range", 0.2, 0.8);
    defer row_quantile_range_table.deinit();
    const row_quantile_range_column = try row_quantile_range_table.column("row_quantile_range");
    try std.testing.expect(row_quantile_range_column.f64.nullable());
    const row_quantile_range = try row_quantile_range_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_quantile_range);
    const row_quantile_range_validity = try row_quantile_range_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_quantile_range_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_quantile_range[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_quantile_range[1], 1e-12);
    try std.testing.expectEqual(@as(f64, 0.0), row_quantile_range[2]);
    try std.testing.expectApproxEqAbs(@as(f64, 21.6), row_quantile_range[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_quantile_range_validity);

    var row_trimmed_mean_table = try validity_table.withRowTrimmedMean(&.{ "a", "b" }, "row_trimmed_mean", 0.25);
    defer row_trimmed_mean_table.deinit();
    const row_trimmed_mean_column = try row_trimmed_mean_table.column("row_trimmed_mean");
    try std.testing.expect(row_trimmed_mean_column.f64.nullable());
    const row_trimmed_mean = try row_trimmed_mean_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_trimmed_mean);
    const row_trimmed_mean_validity = try row_trimmed_mean_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_trimmed_mean_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_trimmed_mean[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), row_trimmed_mean[1], 1e-12);
    try std.testing.expectEqual(@as(f64, 0.0), row_trimmed_mean[2]);
    try std.testing.expectApproxEqAbs(@as(f64, 22.0), row_trimmed_mean[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_trimmed_mean_validity);

    var row_winsorized_mean_table = try validity_table.withRowWinsorizedMean(&.{ "a", "b" }, "row_winsorized_mean", 0.25);
    defer row_winsorized_mean_table.deinit();
    const row_winsorized_mean_column = try row_winsorized_mean_table.column("row_winsorized_mean");
    try std.testing.expect(row_winsorized_mean_column.f64.nullable());
    const row_winsorized_mean = try row_winsorized_mean_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_winsorized_mean);
    const row_winsorized_mean_validity = try row_winsorized_mean_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_winsorized_mean_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_winsorized_mean[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), row_winsorized_mean[1], 1e-12);
    try std.testing.expectEqual(@as(f64, 0.0), row_winsorized_mean[2]);
    try std.testing.expectApproxEqAbs(@as(f64, 22.0), row_winsorized_mean[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_winsorized_mean_validity);

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

    var row_idr_table = try validity_table.withRowInterdecileRange(&.{ "a", "b" }, "row_idr");
    defer row_idr_table.deinit();
    const row_idr_column = try row_idr_table.column("row_idr");
    try std.testing.expect(row_idr_column.f64.nullable());
    const row_idr = try row_idr_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_idr);
    const row_idr_validity = try row_idr_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_idr_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_idr[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_idr[1], 1e-12);
    try std.testing.expectEqual(@as(f64, 0.0), row_idr[2]);
    try std.testing.expectApproxEqAbs(@as(f64, 28.8), row_idr[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_idr_validity);

    var row_midhinge_table = try validity_table.withRowMidhinge(&.{ "a", "b" }, "row_midhinge");
    defer row_midhinge_table.deinit();
    const row_midhinge = try (try row_midhinge_table.column("row_midhinge")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_midhinge);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 22.0 }, row_midhinge);

    var row_trimean_table = try validity_table.withRowTrimean(&.{ "a", "b" }, "row_trimean");
    defer row_trimean_table.deinit();
    const row_trimean_column = try row_trimean_table.column("row_trimean");
    try std.testing.expect(row_trimean_column.f64.nullable());
    const row_trimean = try row_trimean_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_trimean);
    const row_trimean_validity = try row_trimean_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_trimean_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 22.0 }, row_trimean);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_trimean_validity);

    var row_bowley_table = try validity_table.withRowBowleySkewness(&.{ "a", "b" }, "row_bowley");
    defer row_bowley_table.deinit();
    const row_bowley_column = try row_bowley_table.column("row_bowley");
    try std.testing.expect(row_bowley_column.f64.nullable());
    const row_bowley = try row_bowley_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_bowley);
    const row_bowley_validity = try row_bowley_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_bowley_validity);
    try std.testing.expect(std.math.isNan(row_bowley[0]));
    try std.testing.expect(std.math.isNan(row_bowley[1]));
    try std.testing.expectEqual(@as(f64, 0.0), row_bowley[2]);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_bowley[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_bowley_validity);

    var row_qcd_table = try validity_table.withRowQuartileCoeffDispersion(&.{ "a", "b" }, "row_qcd");
    defer row_qcd_table.deinit();
    const row_qcd_column = try row_qcd_table.column("row_qcd");
    try std.testing.expect(row_qcd_column.f64.nullable());
    const row_qcd = try row_qcd_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_qcd);
    const row_qcd_validity = try row_qcd_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_qcd_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_qcd[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_qcd[1], 1e-12);
    try std.testing.expectEqual(@as(f64, 0.0), row_qcd[2]);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0 / 22.0), row_qcd[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_qcd_validity);

    var row_kelley_table = try validity_table.withRowKelleySkewness(&.{ "a", "b" }, "row_kelley");
    defer row_kelley_table.deinit();
    const row_kelley_column = try row_kelley_table.column("row_kelley");
    try std.testing.expect(row_kelley_column.f64.nullable());
    const row_kelley = try row_kelley_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_kelley);
    const row_kelley_validity = try row_kelley_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_kelley_validity);
    try std.testing.expect(std.math.isNan(row_kelley[0]));
    try std.testing.expect(std.math.isNan(row_kelley[1]));
    try std.testing.expectEqual(@as(f64, 0.0), row_kelley[2]);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_kelley[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_kelley_validity);

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

    var row_mode_margin_table = try validity_table.withRowModeMargin(&.{ "a", "b", "wa" }, "row_mode_margin");
    defer row_mode_margin_table.deinit();
    const row_mode_margin = try (try row_mode_margin_table.column("row_mode_margin")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_mode_margin);
    try std.testing.expectEqualSlices(i64, &.{ 2, 0, 1, 1 }, row_mode_margin);
    var row_mode_margin_ratio_table = try validity_table.withRowModeMarginRatio(&.{ "a", "b", "wa" }, "row_mode_margin_ratio");
    defer row_mode_margin_ratio_table.deinit();
    const row_mode_margin_ratio = try (try row_mode_margin_ratio_table.column("row_mode_margin_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_mode_margin_ratio);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_mode_margin_ratio[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_mode_margin_ratio[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_mode_margin_ratio[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), row_mode_margin_ratio[3], 1e-12);

    var row_cummode_table = try validity_table.withRowCumulativeMode(
        &.{ "a", "b", "wa", "wb" },
        &.{ "a_cummode", "b_cummode", "wa_cummode", "wb_cummode" },
    );
    defer row_cummode_table.deinit();
    const a_cummode_column = try row_cummode_table.column("a_cummode");
    try std.testing.expect(a_cummode_column.f64.nullable());
    const a_cummode = try a_cummode_column.f64.toOwnedSlice(gpa);
    defer gpa.free(a_cummode);
    const a_cummode_validity = try a_cummode_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(a_cummode_validity);
    const b_cummode = try (try row_cummode_table.column("b_cummode")).f64.toOwnedSlice(gpa);
    defer gpa.free(b_cummode);
    const wb_cummode = try (try row_cummode_table.column("wb_cummode")).f64.toOwnedSlice(gpa);
    defer gpa.free(wb_cummode);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 0.0, 0.0, 4.0 }, a_cummode);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, a_cummode_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 4.0 }, b_cummode);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 3.0, 4.0 }, wb_cummode);
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowPrefixMode(&.{"a"}, &.{ "a_cummode", "extra_cummode" }));

    var row_cummode_count_table = try validity_table.withRowCumulativeModeCount(
        &.{ "a", "b", "wa", "wb" },
        &.{ "a_cummode_count", "b_cummode_count", "wa_cummode_count", "wb_cummode_count" },
    );
    defer row_cummode_count_table.deinit();
    const b_cummode_count = try (try row_cummode_count_table.column("b_cummode_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(b_cummode_count);
    const wb_cummode_count = try (try row_cummode_count_table.column("wb_cummode_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(wb_cummode_count);
    try std.testing.expectEqualSlices(i64, &.{ 1, 1, 0, 1 }, b_cummode_count);
    try std.testing.expectEqualSlices(i64, &.{ 2, 1, 1, 2 }, wb_cummode_count);

    var row_cummode_ratio_table = try validity_table.withRowPrefixModeRatio(
        &.{ "a", "b", "wa", "wb" },
        &.{ "a_cummode_ratio", "b_cummode_ratio", "wa_cummode_ratio", "wb_cummode_ratio" },
    );
    defer row_cummode_ratio_table.deinit();
    const b_cummode_ratio = try (try row_cummode_ratio_table.column("b_cummode_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(b_cummode_ratio);
    const wb_cummode_ratio = try (try row_cummode_ratio_table.column("wb_cummode_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(wb_cummode_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 1.0, 0.0, 0.5 }, b_cummode_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 2.0 / 3.0, 1.0 / 3.0, 0.5, 0.5 }, wb_cummode_ratio);
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowPrefixModeCount(&.{"a"}, &.{ "a_cummode_count", "extra_cummode_count" }));

    var row_cummode_margin_table = try validity_table.withRowCumulativeModeMargin(
        &.{ "a", "b", "wa", "wb" },
        &.{ "a_cummode_margin", "b_cummode_margin", "wa_cummode_margin", "wb_cummode_margin" },
    );
    defer row_cummode_margin_table.deinit();
    const b_cummode_margin = try (try row_cummode_margin_table.column("b_cummode_margin")).i64.toOwnedSlice(gpa);
    defer gpa.free(b_cummode_margin);
    const wb_cummode_margin = try (try row_cummode_margin_table.column("wb_cummode_margin")).i64.toOwnedSlice(gpa);
    defer gpa.free(wb_cummode_margin);
    try std.testing.expectEqualSlices(i64, &.{ 1, 1, 0, 0 }, b_cummode_margin);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 0, 1 }, wb_cummode_margin);

    var row_cummode_margin_ratio_table = try validity_table.withRowPrefixModeMarginRatio(
        &.{ "a", "b", "wa", "wb" },
        &.{ "a_cummode_margin_ratio", "b_cummode_margin_ratio", "wa_cummode_margin_ratio", "wb_cummode_margin_ratio" },
    );
    defer row_cummode_margin_ratio_table.deinit();
    const b_cummode_margin_ratio = try (try row_cummode_margin_ratio_table.column("b_cummode_margin_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(b_cummode_margin_ratio);
    const wb_cummode_margin_ratio = try (try row_cummode_margin_ratio_table.column("wb_cummode_margin_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(wb_cummode_margin_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 1.0, 0.0, 0.0 }, b_cummode_margin_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 1.0 / 3.0, 0.0, 0.0, 0.25 }, wb_cummode_margin_ratio);
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowPrefixModeMargin(&.{"a"}, &.{ "a_cummode_margin", "extra_cummode_margin" }));

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

    var row_weighted_sum_table = try validity_table.withRowWeightedSum(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_sum");
    defer row_weighted_sum_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_sum_table, gpa, "row_weighted_sum", &.{ 1.0, 20.0, 0.0, 56.0 }, &.{ true, true, false, true });

    var row_cum_weighted_sum_table = try validity_table.withRowCumulativeWeightedSum(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumsum", "b_row_weighted_cumsum" });
    defer row_cum_weighted_sum_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_sum_table, gpa, "a_row_weighted_cumsum", &.{ 1.0, 0.0, 0.0, 16.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_sum_table, gpa, "b_row_weighted_cumsum", &.{ 0.0, 20.0, 0.0, 56.0 }, &.{ false, true, false, true });

    var row_cum_weighted_mean_table = try validity_table.withRowPrefixWeightedAverage(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cummean", "b_row_weighted_cummean" });
    defer row_cum_weighted_mean_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_mean_table, gpa, "a_row_weighted_cummean", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_mean_table, gpa, "b_row_weighted_cummean", &.{ 0.0, 20.0, 0.0, 56.0 / 5.0 }, &.{ false, true, false, true });

    var row_cum_weighted_mean_square_table = try validity_table.withRowPrefixWeightedMeanSq(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cummeansq", "b_row_weighted_cummeansq" });
    defer row_cum_weighted_mean_square_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_mean_square_table, gpa, "a_row_weighted_cummeansq", &.{ 1.0, 0.0, 0.0, 16.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_mean_square_table, gpa, "b_row_weighted_cummeansq", &.{ 0.0, 400.0, 0.0, 1664.0 / 5.0 }, &.{ false, true, false, true });

    var row_cum_weighted_rms_table = try validity_table.withRowCumWeightedRMS(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumrms", "b_row_weighted_cumrms" });
    defer row_cum_weighted_rms_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_rms_table, gpa, "a_row_weighted_cumrms", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_rms_table, gpa, "b_row_weighted_cumrms", &.{ 0.0, 20.0, 0.0, std.math.sqrt(@as(f64, 1664.0 / 5.0)) }, &.{ false, true, false, true });

    var row_cum_weighted_mean_abs_table = try validity_table.withRowCumulativeWeightedMeanAbs(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cummeanabs", "b_row_weighted_cummeanabs" });
    defer row_cum_weighted_mean_abs_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_mean_abs_table, gpa, "a_row_weighted_cummeanabs", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_mean_abs_table, gpa, "b_row_weighted_cummeanabs", &.{ 0.0, 20.0, 0.0, 56.0 / 5.0 }, &.{ false, true, false, true });

    var row_cum_weighted_l1_table = try validity_table.withRowCumWeightedL1(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cuml1", "b_row_weighted_cuml1" });
    defer row_cum_weighted_l1_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_l1_table, gpa, "a_row_weighted_cuml1", &.{ 1.0, 0.0, 0.0, 16.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_l1_table, gpa, "b_row_weighted_cuml1", &.{ 0.0, 20.0, 0.0, 56.0 }, &.{ false, true, false, true });

    var row_cum_weighted_l2_table = try validity_table.withRowPrefixWeightedL2Norm(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cuml2", "b_row_weighted_cuml2" });
    defer row_cum_weighted_l2_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_l2_table, gpa, "a_row_weighted_cuml2", &.{ 1.0, 0.0, 0.0, 8.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_l2_table, gpa, "b_row_weighted_cuml2", &.{ 0.0, 20.0, 0.0, std.math.sqrt(@as(f64, 1664.0)) }, &.{ false, true, false, true });

    var row_cum_weighted_min_table = try validity_table.withRowCumulativeWeightedMin(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cummin", "b_row_weighted_cummin" });
    defer row_cum_weighted_min_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_min_table, gpa, "a_row_weighted_cummin", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_min_table, gpa, "b_row_weighted_cummin", &.{ 0.0, 20.0, 0.0, 4.0 }, &.{ false, true, false, true });

    var row_cum_weighted_max_table = try validity_table.withRowPrefixWeightedMax(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cummax", "b_row_weighted_cummax" });
    defer row_cum_weighted_max_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_max_table, gpa, "a_row_weighted_cummax", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_max_table, gpa, "b_row_weighted_cummax", &.{ 0.0, 20.0, 0.0, 40.0 }, &.{ false, true, false, true });

    var row_cum_weighted_max_abs_table = try validity_table.withRowCumWeightedMaxAbs(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cummaxabs", "b_row_weighted_cummaxabs" });
    defer row_cum_weighted_max_abs_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_max_abs_table, gpa, "a_row_weighted_cummaxabs", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_max_abs_table, gpa, "b_row_weighted_cummaxabs", &.{ 0.0, 20.0, 0.0, 40.0 }, &.{ false, true, false, true });

    var row_cum_weighted_min_abs_table = try validity_table.withRowPrefixWeightedMinAbs(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumminabs", "b_row_weighted_cumminabs" });
    defer row_cum_weighted_min_abs_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_min_abs_table, gpa, "a_row_weighted_cumminabs", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_min_abs_table, gpa, "b_row_weighted_cumminabs", &.{ 0.0, 20.0, 0.0, 4.0 }, &.{ false, true, false, true });

    var row_cum_weighted_range_table = try validity_table.withRowCumulativeWeightedRange(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumrange", "b_row_weighted_cumrange" });
    defer row_cum_weighted_range_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_range_table, gpa, "a_row_weighted_cumrange", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_range_table, gpa, "b_row_weighted_cumrange", &.{ 0.0, 0.0, 0.0, 36.0 }, &.{ false, true, false, true });

    var row_cum_weighted_midrange_table = try validity_table.withRowCumWeightedMidrange(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cummidrange", "b_row_weighted_cummidrange" });
    defer row_cum_weighted_midrange_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_midrange_table, gpa, "a_row_weighted_cummidrange", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_midrange_table, gpa, "b_row_weighted_cummidrange", &.{ 0.0, 20.0, 0.0, 22.0 }, &.{ false, true, false, true });

    var row_cum_weighted_range_coeff_table = try validity_table.withRowPrefixWeightedRangeCoefficient(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumrange_coeff", "b_row_weighted_cumrange_coeff" });
    defer row_cum_weighted_range_coeff_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_range_coeff_table, gpa, "a_row_weighted_cumrange_coeff", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_range_coeff_table, gpa, "b_row_weighted_cumrange_coeff", &.{ 0.0, 0.0, 0.0, 9.0 / 11.0 }, &.{ false, true, false, true });

    var row_cum_weighted_product_table = try validity_table.withRowCumWeightedProd(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumprod", "b_row_weighted_cumprod" });
    defer row_cum_weighted_product_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_product_table, gpa, "a_row_weighted_cumprod", &.{ 1.0, 0.0, 0.0, 256.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_product_table, gpa, "b_row_weighted_cumprod", &.{ 0.0, 20.0, 0.0, std.math.exp(4.0 * std.math.log(f64, std.math.e, @as(f64, 4.0)) + std.math.log(f64, std.math.e, @as(f64, 40.0))) }, &.{ false, true, false, true });

    var row_cum_weighted_geo_table = try validity_table.withRowPrefixWeightedGeoMean(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumgeo", "b_row_weighted_cumgeo" });
    defer row_cum_weighted_geo_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_geo_table, gpa, "a_row_weighted_cumgeo", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_geo_table, gpa, "b_row_weighted_cumgeo", &.{ 0.0, 20.0, 0.0, std.math.exp((4.0 * std.math.log(f64, std.math.e, @as(f64, 4.0)) + std.math.log(f64, std.math.e, @as(f64, 40.0))) / 5.0) }, &.{ false, true, false, true });

    var row_cum_weighted_harmonic_table = try validity_table.withRowCumWeightedHarmonicMean(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumharmonic", "b_row_weighted_cumharmonic" });
    defer row_cum_weighted_harmonic_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_harmonic_table, gpa, "a_row_weighted_cumharmonic", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_harmonic_table, gpa, "b_row_weighted_cumharmonic", &.{ 0.0, 20.0, 0.0, 5.0 / (4.0 / 4.0 + 1.0 / 40.0) }, &.{ false, true, false, true });

    var row_cum_weighted_logsumexp_table = try validity_table.withRowPrefixWeightedLogSumExp(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumlogsumexp", "b_row_weighted_cumlogsumexp" });
    defer row_cum_weighted_logsumexp_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_logsumexp_table, gpa, "a_row_weighted_cumlogsumexp", &.{ 1.0, 0.0, 0.0, 4.0 + std.math.log(f64, std.math.e, @as(f64, 4.0)) }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_logsumexp_table, gpa, "b_row_weighted_cumlogsumexp", &.{ 0.0, 20.0, 0.0, 40.0 + std.math.log1p(@as(f64, 4.0) * std.math.exp(@as(f64, -36.0))) }, &.{ false, true, false, true });

    var row_cum_weighted_logmeanexp_table = try validity_table.withRowCumWeightedLogmeanexp(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumlogmeanexp", "b_row_weighted_cumlogmeanexp" });
    defer row_cum_weighted_logmeanexp_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_logmeanexp_table, gpa, "a_row_weighted_cumlogmeanexp", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_logmeanexp_table, gpa, "b_row_weighted_cumlogmeanexp", &.{ 0.0, 20.0, 0.0, 40.0 + std.math.log1p(@as(f64, 4.0) * std.math.exp(@as(f64, -36.0))) - std.math.log(f64, std.math.e, @as(f64, 5.0)) }, &.{ false, true, false, true });

    var row_cum_weighted_variance_table = try validity_table.withRowCumWeightedVar(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumvar", "b_row_weighted_cumvar" }, 0.0);
    defer row_cum_weighted_variance_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_variance_table, gpa, "a_row_weighted_cumvar", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_variance_table, gpa, "b_row_weighted_cumvar", &.{ 0.0, 0.0, 0.0, 207.36 }, &.{ false, true, false, true });

    var row_cum_weighted_stddev_table = try validity_table.withRowPrefixWeightedStd(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumstd", "b_row_weighted_cumstd" }, 0.0);
    defer row_cum_weighted_stddev_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_stddev_table, gpa, "a_row_weighted_cumstd", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_stddev_table, gpa, "b_row_weighted_cumstd", &.{ 0.0, 0.0, 0.0, 14.4 }, &.{ false, true, false, true });

    var row_cum_weighted_sem_table = try validity_table.withRowCumulativeWeightedSem(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumsem", "b_row_weighted_cumsem" }, 0.0);
    defer row_cum_weighted_sem_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_sem_table, gpa, "a_row_weighted_cumsem", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_sem_table, gpa, "b_row_weighted_cumsem", &.{ 0.0, 0.0, 0.0, std.math.sqrt(@as(f64, 207.36 / 5.0)) }, &.{ false, true, false, true });

    var row_cum_weighted_cv_table = try validity_table.withRowCumWeightedCV(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumcv", "b_row_weighted_cumcv" }, 0.0);
    defer row_cum_weighted_cv_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_cv_table, gpa, "a_row_weighted_cumcv", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_cv_table, gpa, "b_row_weighted_cumcv", &.{ 0.0, 0.0, 0.0, 9.0 / 7.0 }, &.{ false, true, false, true });

    var row_cum_weighted_fano_table = try validity_table.withRowPrefixWeightedFano(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumfano", "b_row_weighted_cumfano" }, 0.0);
    defer row_cum_weighted_fano_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_fano_table, gpa, "a_row_weighted_cumfano", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_fano_table, gpa, "b_row_weighted_cumfano", &.{ 0.0, 0.0, 0.0, 648.0 / 35.0 }, &.{ false, true, false, true });

    var row_cum_weighted_skew_table = try validity_table.withRowCumWeightedSkew(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumskew", "b_row_weighted_cumskew" });
    defer row_cum_weighted_skew_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_skew_table, gpa, "a_row_weighted_cumskew", &.{ std.math.nan(f64), 0.0, 0.0, std.math.nan(f64) }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_skew_table, gpa, "b_row_weighted_cumskew", &.{ 0.0, std.math.nan(f64), 0.0, 1.5 }, &.{ false, true, false, true });

    var row_cum_weighted_kurt_table = try validity_table.withRowPrefixWeightedKurtosis(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumkurt", "b_row_weighted_cumkurt" });
    defer row_cum_weighted_kurt_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_kurt_table, gpa, "a_row_weighted_cumkurt", &.{ std.math.nan(f64), 0.0, 0.0, std.math.nan(f64) }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_kurt_table, gpa, "b_row_weighted_cumkurt", &.{ 0.0, std.math.nan(f64), 0.0, 0.25 }, &.{ false, true, false, true });

    var row_cum_weighted_quantile_table = try validity_table.withRowPrefixWeightedQuantile(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumquantile", "b_row_weighted_cumquantile" }, 0.9);
    defer row_cum_weighted_quantile_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_quantile_table, gpa, "a_row_weighted_cumquantile", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_quantile_table, gpa, "b_row_weighted_cumquantile", &.{ 0.0, 20.0, 0.0, 40.0 }, &.{ false, true, false, true });

    var row_cum_weighted_median_table = try validity_table.withRowCumWeightedMedian(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cummedian", "b_row_weighted_cummedian" });
    defer row_cum_weighted_median_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_median_table, gpa, "a_row_weighted_cummedian", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_median_table, gpa, "b_row_weighted_cummedian", &.{ 0.0, 20.0, 0.0, 4.0 }, &.{ false, true, false, true });

    var row_cum_weighted_iqr_table = try validity_table.withRowPrefixWeightedIQR(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumiqr", "b_row_weighted_cumiqr" });
    defer row_cum_weighted_iqr_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_iqr_table, gpa, "a_row_weighted_cumiqr", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_iqr_table, gpa, "b_row_weighted_cumiqr", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ false, true, false, true });

    var row_cum_weighted_mad_table = try validity_table.withRowPrefixWeightedMAD(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cummad", "b_row_weighted_cummad" });
    defer row_cum_weighted_mad_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_mad_table, gpa, "a_row_weighted_cummad", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_mad_table, gpa, "b_row_weighted_cummad", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ false, true, false, true });

    var row_cum_weighted_trimmed_table = try validity_table.withRowPrefixWeightedTrimmedMean(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumtrimmed", "b_row_weighted_cumtrimmed" }, 0.25);
    defer row_cum_weighted_trimmed_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_trimmed_table, gpa, "a_row_weighted_cumtrimmed", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_trimmed_table, gpa, "b_row_weighted_cumtrimmed", &.{ 0.0, 20.0, 0.0, 4.0 }, &.{ false, true, false, true });

    var row_cum_weighted_winsor_table = try validity_table.withRowCumWeightedWinsorizedMean(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumwinsor", "b_row_weighted_cumwinsor" }, 0.25);
    defer row_cum_weighted_winsor_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_winsor_table, gpa, "a_row_weighted_cumwinsor", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_winsor_table, gpa, "b_row_weighted_cumwinsor", &.{ 0.0, 20.0, 0.0, 4.0 }, &.{ false, true, false, true });

    var row_cum_weighted_idr_table = try validity_table.withRowCumWeightedIDR(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumidr", "b_row_weighted_cumidr" });
    defer row_cum_weighted_idr_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_idr_table, gpa, "a_row_weighted_cumidr", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_idr_table, gpa, "b_row_weighted_cumidr", &.{ 0.0, 0.0, 0.0, 36.0 }, &.{ false, true, false, true });

    var row_cum_weighted_midhinge_table = try validity_table.withRowPrefixWeightedMidhinge(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cummidhinge", "b_row_weighted_cummidhinge" });
    defer row_cum_weighted_midhinge_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_midhinge_table, gpa, "a_row_weighted_cummidhinge", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_midhinge_table, gpa, "b_row_weighted_cummidhinge", &.{ 0.0, 20.0, 0.0, 4.0 }, &.{ false, true, false, true });

    var row_cum_weighted_trimean_table = try validity_table.withRowCumWeightedTrimean(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumtrimean", "b_row_weighted_cumtrimean" });
    defer row_cum_weighted_trimean_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_trimean_table, gpa, "a_row_weighted_cumtrimean", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_trimean_table, gpa, "b_row_weighted_cumtrimean", &.{ 0.0, 20.0, 0.0, 4.0 }, &.{ false, true, false, true });

    var row_cum_weighted_bowley_table = try validity_table.withRowPrefixWeightedBowleySkew(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumbowley", "b_row_weighted_cumbowley" });
    defer row_cum_weighted_bowley_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_bowley_table, gpa, "a_row_weighted_cumbowley", &.{ std.math.nan(f64), 0.0, 0.0, std.math.nan(f64) }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_bowley_table, gpa, "b_row_weighted_cumbowley", &.{ 0.0, std.math.nan(f64), 0.0, std.math.nan(f64) }, &.{ false, true, false, true });

    var row_cum_weighted_qcd_table = try validity_table.withRowCumWeightedQCD(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumqcd", "b_row_weighted_cumqcd" });
    defer row_cum_weighted_qcd_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_qcd_table, gpa, "a_row_weighted_cumqcd", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_qcd_table, gpa, "b_row_weighted_cumqcd", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ false, true, false, true });

    var row_cum_weighted_kelley_table = try validity_table.withRowPrefixWeightedKelleySkew(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumkelley", "b_row_weighted_cumkelley" });
    defer row_cum_weighted_kelley_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_kelley_table, gpa, "a_row_weighted_cumkelley", &.{ std.math.nan(f64), 0.0, 0.0, std.math.nan(f64) }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_kelley_table, gpa, "b_row_weighted_cumkelley", &.{ 0.0, std.math.nan(f64), 0.0, 1.0 }, &.{ false, true, false, true });

    var row_cum_weighted_mode_table = try validity_table.withRowPrefixWeightedMode(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cummode", "b_row_weighted_cummode" });
    defer row_cum_weighted_mode_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_mode_table, gpa, "a_row_weighted_cummode", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_mode_table, gpa, "b_row_weighted_cummode", &.{ 0.0, 20.0, 0.0, 4.0 }, &.{ false, true, false, true });

    var row_cum_weighted_mode_weight_table = try validity_table.withRowCumWeightedModeWeight(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cummode_weight", "b_row_weighted_cummode_weight" });
    defer row_cum_weighted_mode_weight_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_mode_weight_table, gpa, "a_row_weighted_cummode_weight", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_mode_weight_table, gpa, "b_row_weighted_cummode_weight", &.{ 0.0, 1.0, 0.0, 4.0 }, &.{ false, true, false, true });

    var row_cum_weighted_mode_ratio_table = try validity_table.withRowPrefixWeightedModeRatio(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cummode_ratio", "b_row_weighted_cummode_ratio" });
    defer row_cum_weighted_mode_ratio_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_mode_ratio_table, gpa, "a_row_weighted_cummode_ratio", &.{ 1.0, 0.0, 0.0, 1.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_mode_ratio_table, gpa, "b_row_weighted_cummode_ratio", &.{ 0.0, 1.0, 0.0, 4.0 / 5.0 }, &.{ false, true, false, true });

    var row_cum_weighted_mode_margin_table = try validity_table.withRowCumWeightedModeMargin(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cummode_margin", "b_row_weighted_cummode_margin" });
    defer row_cum_weighted_mode_margin_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_mode_margin_table, gpa, "a_row_weighted_cummode_margin", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_mode_margin_table, gpa, "b_row_weighted_cummode_margin", &.{ 0.0, 1.0, 0.0, 3.0 }, &.{ false, true, false, true });

    var row_cum_weighted_mode_margin_ratio_table = try validity_table.withRowPrefixWeightedModeMarginRatio(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cummode_margin_ratio", "b_row_weighted_cummode_margin_ratio" });
    defer row_cum_weighted_mode_margin_ratio_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_mode_margin_ratio_table, gpa, "a_row_weighted_cummode_margin_ratio", &.{ 1.0, 0.0, 0.0, 1.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_mode_margin_ratio_table, gpa, "b_row_weighted_cummode_margin_ratio", &.{ 0.0, 1.0, 0.0, 3.0 / 5.0 }, &.{ false, true, false, true });

    const weighted_prefix_entropy = -(@as(f64, 4.0 / 5.0) * std.math.log(f64, std.math.e, @as(f64, 4.0 / 5.0)) + @as(f64, 1.0 / 5.0) * std.math.log(f64, std.math.e, @as(f64, 1.0 / 5.0)));
    var row_cum_weighted_entropy_table = try validity_table.withRowCumWeightedEntropy(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumentropy", "b_row_weighted_cumentropy" });
    defer row_cum_weighted_entropy_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_entropy_table, gpa, "a_row_weighted_cumentropy", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_entropy_table, gpa, "b_row_weighted_cumentropy", &.{ 0.0, 0.0, 0.0, weighted_prefix_entropy }, &.{ false, true, false, true });

    var row_cum_weighted_gini_table = try validity_table.withRowPrefixWeightedGini(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumgini", "b_row_weighted_cumgini" });
    defer row_cum_weighted_gini_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_gini_table, gpa, "a_row_weighted_cumgini", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_gini_table, gpa, "b_row_weighted_cumgini", &.{ 0.0, 0.0, 0.0, 8.0 / 25.0 }, &.{ false, true, false, true });

    var row_cum_weighted_perplexity_table = try validity_table.withRowCumWeightedPerplexity(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumperplexity", "b_row_weighted_cumperplexity" });
    defer row_cum_weighted_perplexity_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_perplexity_table, gpa, "a_row_weighted_cumperplexity", &.{ 1.0, 0.0, 0.0, 1.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_perplexity_table, gpa, "b_row_weighted_cumperplexity", &.{ 0.0, 1.0, 0.0, std.math.exp(weighted_prefix_entropy) }, &.{ false, true, false, true });

    var row_cum_weighted_inverse_table = try validity_table.withRowPrefixWeightedInverseSimpson(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cuminverse", "b_row_weighted_cuminverse" });
    defer row_cum_weighted_inverse_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_inverse_table, gpa, "a_row_weighted_cuminverse", &.{ 1.0, 0.0, 0.0, 1.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_inverse_table, gpa, "b_row_weighted_cuminverse", &.{ 0.0, 1.0, 0.0, 25.0 / 17.0 }, &.{ false, true, false, true });

    var row_cum_weighted_concentration_table = try validity_table.withRowCumWeightedConcentration(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumconcentration", "b_row_weighted_cumconcentration" });
    defer row_cum_weighted_concentration_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_concentration_table, gpa, "a_row_weighted_cumconcentration", &.{ 1.0, 0.0, 0.0, 1.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_concentration_table, gpa, "b_row_weighted_cumconcentration", &.{ 0.0, 1.0, 0.0, 17.0 / 25.0 }, &.{ false, true, false, true });

    var row_cum_weighted_evenness_table = try validity_table.withRowPrefixWeightedEvenness(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumevenness", "b_row_weighted_cumevenness" });
    defer row_cum_weighted_evenness_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_evenness_table, gpa, "a_row_weighted_cumevenness", &.{ 1.0, 0.0, 0.0, 1.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_evenness_table, gpa, "b_row_weighted_cumevenness", &.{ 0.0, 1.0, 0.0, weighted_prefix_entropy / std.math.log(f64, std.math.e, @as(f64, 2.0)) }, &.{ false, true, false, true });

    var row_cum_weighted_mean_abs_dev_table = try validity_table.withRowCumWeightedMeanAbsDev(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cum_mean_abs_dev", "b_row_weighted_cum_mean_abs_dev" });
    defer row_cum_weighted_mean_abs_dev_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_mean_abs_dev_table, gpa, "a_row_weighted_cum_mean_abs_dev", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_mean_abs_dev_table, gpa, "b_row_weighted_cum_mean_abs_dev", &.{ 0.0, 0.0, 0.0, 288.0 / 25.0 }, &.{ false, true, false, true });

    var row_cum_weighted_mad_ratio_table = try validity_table.withRowPrefixWeightedMadRatio(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cum_mad_ratio", "b_row_weighted_cum_mad_ratio" });
    defer row_cum_weighted_mad_ratio_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_mad_ratio_table, gpa, "a_row_weighted_cum_mad_ratio", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_mad_ratio_table, gpa, "b_row_weighted_cum_mad_ratio", &.{ 0.0, 0.0, 0.0, 36.0 / 35.0 }, &.{ false, true, false, true });

    var row_cum_weighted_gini_mean_diff_table = try validity_table.withRowCumWeightedGiniMeanDiff(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cum_gini_mean_diff", "b_row_weighted_cum_gini_mean_diff" });
    defer row_cum_weighted_gini_mean_diff_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_gini_mean_diff_table, gpa, "a_row_weighted_cum_gini_mean_diff", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_gini_mean_diff_table, gpa, "b_row_weighted_cum_gini_mean_diff", &.{ 0.0, 0.0, 0.0, 36.0 }, &.{ false, true, false, true });

    var row_cum_weighted_gini_coeff_table = try validity_table.withRowPrefixWeightedGiniCoeff(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cum_gini_coeff", "b_row_weighted_cum_gini_coeff" });
    defer row_cum_weighted_gini_coeff_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_gini_coeff_table, gpa, "a_row_weighted_cum_gini_coeff", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_gini_coeff_table, gpa, "b_row_weighted_cum_gini_coeff", &.{ 0.0, 0.0, 0.0, 45.0 / 28.0 }, &.{ false, true, false, true });

    var row_cum_weighted_weight_sum_table = try validity_table.withRowCumulativeWeightedWeightSum(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cum_weight_sum", "b_row_weighted_cum_weight_sum" });
    defer row_cum_weighted_weight_sum_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_weight_sum_table, gpa, "a_row_weighted_cum_weight_sum", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_weight_sum_table, gpa, "b_row_weighted_cum_weight_sum", &.{ 0.0, 1.0, 0.0, 5.0 }, &.{ false, true, false, true });

    var row_cum_weighted_positive_count_table = try validity_table.withRowCumWeightedPositiveCount(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cum_positive_count", "b_row_weighted_cum_positive_count" });
    defer row_cum_weighted_positive_count_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_positive_count_table, gpa, "a_row_weighted_cum_positive_count", &.{ 1.0, 0.0, 0.0, 1.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_positive_count_table, gpa, "b_row_weighted_cum_positive_count", &.{ 0.0, 1.0, 0.0, 2.0 }, &.{ false, true, false, true });

    var row_cum_weighted_effective_n_table = try validity_table.withRowPrefixWeightedEffectiveCount(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cum_effective_n", "b_row_weighted_cum_effective_n" });
    defer row_cum_weighted_effective_n_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_effective_n_table, gpa, "a_row_weighted_cum_effective_n", &.{ 1.0, 0.0, 0.0, 1.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(row_cum_weighted_effective_n_table, gpa, "b_row_weighted_cum_effective_n", &.{ 0.0, 1.0, 0.0, 25.0 / 17.0 }, &.{ false, true, false, true });
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowCumulativeWeightedMean(&.{"a"}, &.{"wa"}, &.{ "a_row_weighted_cummean", "extra_row_weighted_cummean" }));
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowCumulativeWeightedMeanSquare(&.{"a"}, &.{"wa"}, &.{ "a_row_weighted_cummeansq", "extra_row_weighted_cummeansq" }));
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowCumulativeWeightedRange(&.{"a"}, &.{"wa"}, &.{ "a_row_weighted_cumrange", "extra_row_weighted_cumrange" }));
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowCumulativeWeightedProduct(&.{"a"}, &.{"wa"}, &.{ "a_row_weighted_cumprod", "extra_row_weighted_cumprod" }));
    try std.testing.expectError(error.InvalidShape, validity_table.withRowCumulativeWeightedVariance(&.{"a"}, &.{"wa"}, &.{"a_row_weighted_cumvar"}, -1.0));
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowCumulativeWeightedSkewness(&.{"a"}, &.{"wa"}, &.{ "a_row_weighted_cumskew", "extra_row_weighted_cumskew" }));
    try std.testing.expectError(error.InvalidShape, validity_table.withRowCumulativeWeightedQuantile(&.{"a"}, &.{"wa"}, &.{"a_row_weighted_cumquantile"}, 1.5));
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowCumulativeWeightedMedian(&.{"a"}, &.{"wa"}, &.{ "a_row_weighted_cummedian", "extra_row_weighted_cummedian" }));
    try std.testing.expectError(error.InvalidShape, validity_table.withRowCumulativeWeightedTrimmedMean(&.{"a"}, &.{"wa"}, &.{"a_row_weighted_cumtrimmed"}, 0.5));
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowCumulativeWeightedMidhinge(&.{"a"}, &.{"wa"}, &.{ "a_row_weighted_cummidhinge", "extra_row_weighted_cummidhinge" }));
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowCumulativeWeightedMode(&.{"a"}, &.{"wa"}, &.{ "a_row_weighted_cummode", "extra_row_weighted_cummode" }));
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowCumulativeWeightedModeWeight(&.{"a"}, &.{"wa"}, &.{ "a_row_weighted_cummode_weight", "extra_row_weighted_cummode_weight" }));
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowCumulativeWeightedEntropy(&.{"a"}, &.{"wa"}, &.{ "a_row_weighted_cumentropy", "extra_row_weighted_cumentropy" }));
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowCumulativeWeightedMeanAbsDev(&.{"a"}, &.{"wa"}, &.{ "a_row_weighted_cum_mean_abs_dev", "extra_row_weighted_cum_mean_abs_dev" }));
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowCumulativeWeightedWeightSum(&.{"a"}, &.{"wa"}, &.{ "a_row_weighted_cum_weight_sum", "extra_row_weighted_cum_weight_sum" }));

    var row_weighted_weight_sum_table = try validity_table.withRowWeightedWeightSum(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_weight_sum");
    defer row_weighted_weight_sum_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_weight_sum_table, gpa, "row_weighted_weight_sum", &.{ 1.0, 1.0, 0.0, 5.0 }, &.{ true, true, false, true });

    var row_weighted_positive_count_table = try validity_table.withRowWeightedPositiveCount(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_positive_count");
    defer row_weighted_positive_count_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_positive_count_table, gpa, "row_weighted_positive_count", &.{ 1.0, 1.0, 0.0, 2.0 }, &.{ true, true, false, true });

    var row_weighted_effective_n_table = try validity_table.withRowWeightedEffectiveCount(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_effective_n");
    defer row_weighted_effective_n_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_effective_n_table, gpa, "row_weighted_effective_n", &.{ 1.0, 1.0, 0.0, 25.0 / 17.0 }, &.{ true, true, false, true });

    var row_weighted_mean_square_table = try validity_table.withRowWeightedMeanSq(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_mean_square");
    defer row_weighted_mean_square_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_mean_square_table, gpa, "row_weighted_mean_square", &.{ 1.0, 400.0, 0.0, 1664.0 / 5.0 }, &.{ true, true, false, true });

    var row_weighted_rms_table = try validity_table.withRowWeightedRMS(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_rms");
    defer row_weighted_rms_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_rms_table, gpa, "row_weighted_rms", &.{ 1.0, 20.0, 0.0, std.math.sqrt(@as(f64, 1664.0 / 5.0)) }, &.{ true, true, false, true });

    var row_weighted_mean_abs_table = try validity_table.withRowWeightedMeanAbs(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_mean_abs");
    defer row_weighted_mean_abs_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_mean_abs_table, gpa, "row_weighted_mean_abs", &.{ 1.0, 20.0, 0.0, 56.0 / 5.0 }, &.{ true, true, false, true });

    var row_weighted_l1_table = try validity_table.withRowWeightedL1(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_l1");
    defer row_weighted_l1_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_l1_table, gpa, "row_weighted_l1", &.{ 1.0, 20.0, 0.0, 56.0 }, &.{ true, true, false, true });

    var row_weighted_l2_table = try validity_table.withRowWeightedL2Norm(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_l2");
    defer row_weighted_l2_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_l2_table, gpa, "row_weighted_l2", &.{ 1.0, 20.0, 0.0, std.math.sqrt(@as(f64, 1664.0)) }, &.{ true, true, false, true });

    var row_weighted_min_table = try validity_table.withRowWeightedMinimum(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_min");
    defer row_weighted_min_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_min_table, gpa, "row_weighted_min", &.{ 1.0, 20.0, 0.0, 4.0 }, &.{ true, true, false, true });

    var row_weighted_max_table = try validity_table.withRowWeightedMaximum(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_max");
    defer row_weighted_max_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_max_table, gpa, "row_weighted_max", &.{ 1.0, 20.0, 0.0, 40.0 }, &.{ true, true, false, true });

    var row_weighted_max_abs_table = try validity_table.withRowWeightedMaximumAbs(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_max_abs");
    defer row_weighted_max_abs_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_max_abs_table, gpa, "row_weighted_max_abs", &.{ 1.0, 20.0, 0.0, 40.0 }, &.{ true, true, false, true });

    var row_weighted_min_abs_table = try validity_table.withRowWeightedMinimumAbs(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_min_abs");
    defer row_weighted_min_abs_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_min_abs_table, gpa, "row_weighted_min_abs", &.{ 1.0, 20.0, 0.0, 4.0 }, &.{ true, true, false, true });

    var row_weighted_range_table = try validity_table.withRowWeightedRange(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_range");
    defer row_weighted_range_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_range_table, gpa, "row_weighted_range", &.{ 0.0, 0.0, 0.0, 36.0 }, &.{ true, true, false, true });

    var row_weighted_midrange_table = try validity_table.withRowWeightedMidrange(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_midrange");
    defer row_weighted_midrange_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_midrange_table, gpa, "row_weighted_midrange", &.{ 1.0, 20.0, 0.0, 22.0 }, &.{ true, true, false, true });

    var row_weighted_range_coeff_table = try validity_table.withRowWeightedRangeCoefficient(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_range_coeff");
    defer row_weighted_range_coeff_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_range_coeff_table, gpa, "row_weighted_range_coeff", &.{ 0.0, 0.0, 0.0, 9.0 / 11.0 }, &.{ true, true, false, true });

    var row_weighted_product_table = try validity_table.withRowWeightedProduct(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_product");
    defer row_weighted_product_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_product_table, gpa, "row_weighted_product", &.{ 1.0, 20.0, 0.0, std.math.exp(4.0 * std.math.log(f64, std.math.e, @as(f64, 4.0)) + std.math.log(f64, std.math.e, @as(f64, 40.0))) }, &.{ true, true, false, true });

    var row_weighted_geo_table = try validity_table.withRowWeightedGeoMean(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_geo");
    defer row_weighted_geo_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_geo_table, gpa, "row_weighted_geo", &.{ 1.0, 20.0, 0.0, std.math.exp((4.0 * std.math.log(f64, std.math.e, @as(f64, 4.0)) + std.math.log(f64, std.math.e, @as(f64, 40.0))) / 5.0) }, &.{ true, true, false, true });

    var row_weighted_harmonic_table = try validity_table.withRowWeightedHarmonicMean(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_harmonic");
    defer row_weighted_harmonic_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_harmonic_table, gpa, "row_weighted_harmonic", &.{ 1.0, 20.0, 0.0, 5.0 / (4.0 / 4.0 + 1.0 / 40.0) }, &.{ true, true, false, true });

    var row_weighted_logsumexp_table = try validity_table.withRowWeightedLogsumexp(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_logsumexp");
    defer row_weighted_logsumexp_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_logsumexp_table, gpa, "row_weighted_logsumexp", &.{ 1.0, 20.0, 0.0, 40.0 + std.math.log1p(@as(f64, 4.0) * std.math.exp(@as(f64, -36.0))) }, &.{ true, true, false, true });

    var row_weighted_logmeanexp_table = try validity_table.withRowWeightedLogmeanexp(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_logmeanexp");
    defer row_weighted_logmeanexp_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_logmeanexp_table, gpa, "row_weighted_logmeanexp", &.{ 1.0, 20.0, 0.0, 40.0 + std.math.log1p(@as(f64, 4.0) * std.math.exp(@as(f64, -36.0))) - std.math.log(f64, std.math.e, @as(f64, 5.0)) }, &.{ true, true, false, true });

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

    var row_weighted_trimmed_table = try validity_table.withRowWeightedTrimmedMean(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_trimmed", 0.25);
    defer row_weighted_trimmed_table.deinit();
    const row_weighted_trimmed_column = try row_weighted_trimmed_table.column("row_weighted_trimmed");
    try std.testing.expect(row_weighted_trimmed_column.f64.nullable());
    const row_weighted_trimmed = try row_weighted_trimmed_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_trimmed);
    const row_weighted_trimmed_validity = try row_weighted_trimmed_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_trimmed_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 4.0 }, row_weighted_trimmed);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_trimmed_validity);

    var row_weighted_winsorized_table = try validity_table.withRowWeightedWinsorizedMean(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_winsorized", 0.25);
    defer row_weighted_winsorized_table.deinit();
    const row_weighted_winsorized_column = try row_weighted_winsorized_table.column("row_weighted_winsorized");
    try std.testing.expect(row_weighted_winsorized_column.f64.nullable());
    const row_weighted_winsorized = try row_weighted_winsorized_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_winsorized);
    const row_weighted_winsorized_validity = try row_weighted_winsorized_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_winsorized_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 4.0 }, row_weighted_winsorized);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_winsorized_validity);

    var row_weighted_idr_table = try validity_table.withRowWeightedInterdecileRange(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_idr");
    defer row_weighted_idr_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_idr_table, gpa, "row_weighted_idr", &.{ 0.0, 0.0, 0.0, 36.0 }, &.{ true, true, false, true });

    var row_weighted_midhinge_table = try validity_table.withRowWeightedMidhinge(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_midhinge");
    defer row_weighted_midhinge_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_midhinge_table, gpa, "row_weighted_midhinge", &.{ 1.0, 20.0, 0.0, 4.0 }, &.{ true, true, false, true });

    var row_weighted_trimean_table = try validity_table.withRowWeightedTrimean(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_trimean");
    defer row_weighted_trimean_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_trimean_table, gpa, "row_weighted_trimean", &.{ 1.0, 20.0, 0.0, 4.0 }, &.{ true, true, false, true });

    var row_weighted_bowley_table = try validity_table.withRowWeightedBowleySkewness(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_bowley");
    defer row_weighted_bowley_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_bowley_table, gpa, "row_weighted_bowley", &.{ std.math.nan(f64), std.math.nan(f64), 0.0, std.math.nan(f64) }, &.{ true, true, false, true });

    var row_weighted_qcd_table = try validity_table.withRowWeightedQcd(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_qcd");
    defer row_weighted_qcd_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_qcd_table, gpa, "row_weighted_qcd", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, true, false, true });

    var row_weighted_kelley_table = try validity_table.withRowWeightedKelleySkewness(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_kelley");
    defer row_weighted_kelley_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_kelley_table, gpa, "row_weighted_kelley", &.{ std.math.nan(f64), std.math.nan(f64), 0.0, 1.0 }, &.{ true, true, false, true });

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

    var row_weighted_mode_margin_table = try validity_table.withRowWeightedModeMargin(&.{ "a", "b", "wa" }, &.{ "wb", "wa", "wb" }, "row_weighted_mode_margin");
    defer row_weighted_mode_margin_table.deinit();
    const row_weighted_mode_margin = try (try row_weighted_mode_margin_table.column("row_weighted_mode_margin")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_mode_margin);
    try std.testing.expectEqualSlices(f64, &.{ 4.0, 1.0, 5.0, 2.0 }, row_weighted_mode_margin);
    var row_weighted_mode_margin_ratio_table = try validity_table.withRowWeightedModeMarginRatio(&.{ "a", "b", "wa" }, &.{ "wb", "wa", "wb" }, "row_weighted_mode_margin_ratio");
    defer row_weighted_mode_margin_ratio_table.deinit();
    const row_weighted_mode_margin_ratio = try (try row_weighted_mode_margin_ratio_table.column("row_weighted_mode_margin_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_mode_margin_ratio);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_weighted_mode_margin_ratio[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), row_weighted_mode_margin_ratio[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_weighted_mode_margin_ratio[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), row_weighted_mode_margin_ratio[3], 1e-12);

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

    var row_weighted_mean_abs_dev_table = try validity_table.withRowWeightedMeanAbsDev(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_mean_abs_dev");
    defer row_weighted_mean_abs_dev_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_mean_abs_dev_table, gpa, "row_weighted_mean_abs_dev", &.{ 0.0, 0.0, 0.0, 11.52 }, &.{ true, true, false, true });

    var row_weighted_mad_ratio_table = try validity_table.withRowWeightedMadRatio(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_mad_ratio");
    defer row_weighted_mad_ratio_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_mad_ratio_table, gpa, "row_weighted_mad_ratio", &.{ 0.0, 0.0, 0.0, 36.0 / 35.0 }, &.{ true, true, false, true });

    var row_weighted_gini_mean_diff_table = try validity_table.withRowWeightedGiniMeanDiff(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_gini_mean_diff");
    defer row_weighted_gini_mean_diff_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_gini_mean_diff_table, gpa, "row_weighted_gini_mean_diff", &.{ 0.0, 0.0, 0.0, 36.0 }, &.{ true, true, false, true });

    var row_weighted_gini_coeff_table = try validity_table.withRowWeightedGiniCoeff(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_gini_coeff");
    defer row_weighted_gini_coeff_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_gini_coeff_table, gpa, "row_weighted_gini_coeff", &.{ 0.0, 0.0, 0.0, 45.0 / 28.0 }, &.{ true, true, false, true });

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

    var row_weighted_sem_table = try validity_table.withRowWeightedSEM(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_sem", 0.0);
    defer row_weighted_sem_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_sem_table, gpa, "row_weighted_sem", &.{ 0.0, 0.0, 0.0, std.math.sqrt(@as(f64, 207.36 / 5.0)) }, &.{ true, true, false, true });

    var row_weighted_cv_table = try validity_table.withRowWeightedCV(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_cv", 0.0);
    defer row_weighted_cv_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_cv_table, gpa, "row_weighted_cv", &.{ 0.0, 0.0, 0.0, 9.0 / 7.0 }, &.{ true, true, false, true });

    var row_weighted_fano_table = try validity_table.withRowWeightedFano(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_fano", 0.0);
    defer row_weighted_fano_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_fano_table, gpa, "row_weighted_fano", &.{ 0.0, 0.0, 0.0, 648.0 / 35.0 }, &.{ true, true, false, true });

    const row_weighted_skew3 = std.math.sqrt(@as(f64, 5.0)) * @as(f64, 22394.88) / std.math.pow(f64, @as(f64, 1036.8), 1.5);
    var row_weighted_skew_table = try validity_table.withRowWeightedSkewness(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_skew");
    defer row_weighted_skew_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_skew_table, gpa, "row_weighted_skew", &.{ std.math.nan(f64), std.math.nan(f64), 0.0, row_weighted_skew3 }, &.{ true, true, false, true });

    const row_weighted_kurt3 = @as(f64, 5.0) * @as(f64, 698720.256) / (@as(f64, 1036.8) * @as(f64, 1036.8)) - 3.0;
    var row_weighted_kurt_table = try validity_table.withRowWeightedKurtosis(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_kurt");
    defer row_weighted_kurt_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_kurt_table, gpa, "row_weighted_kurt", &.{ std.math.nan(f64), std.math.nan(f64), 0.0, row_weighted_kurt3 }, &.{ true, true, false, true });

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

    var row_weighted_dot_table = try validity_table.withRowWeightedDot(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_dot");
    defer row_weighted_dot_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_dot_table, gpa, "row_weighted_dot", &.{ 1.0, 20.0, 0.0, 104.0 }, &.{ true, true, false, true });

    var row_weighted_cosine_table = try validity_table.withRowWeightedCosine(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_cosine");
    defer row_weighted_cosine_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_cosine_table, gpa, "row_weighted_cosine", &.{ 1.0, 1.0, 0.0, 104.0 / (std.math.sqrt(@as(f64, 1664.0)) * std.math.sqrt(@as(f64, 65.0))) }, &.{ true, true, false, true });

    var row_weighted_sqdist_table = try validity_table.withRowWeightedSquaredDistance(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_sqdist");
    defer row_weighted_sqdist_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_sqdist_table, gpa, "row_weighted_sqdist", &.{ 0.0, 361.0, 0.0, 1521.0 }, &.{ true, true, false, true });

    var row_weighted_euclidean_table = try validity_table.withRowWeightedL2Distance(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_euclidean");
    defer row_weighted_euclidean_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_euclidean_table, gpa, "row_weighted_euclidean", &.{ 0.0, 19.0, 0.0, 39.0 }, &.{ true, true, false, true });

    var row_weighted_manhattan_table = try validity_table.withRowWeightedL1Distance(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_manhattan");
    defer row_weighted_manhattan_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_manhattan_table, gpa, "row_weighted_manhattan", &.{ 0.0, 19.0, 0.0, 39.0 }, &.{ true, true, false, true });

    var row_weighted_chebyshev_table = try validity_table.withRowWeightedChebyshevDistance(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_chebyshev");
    defer row_weighted_chebyshev_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_chebyshev_table, gpa, "row_weighted_chebyshev", &.{ 0.0, 19.0, 0.0, 39.0 }, &.{ true, true, false, true });

    var row_weighted_canberra_table = try validity_table.withRowWeightedCanberraDistance(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_canberra");
    defer row_weighted_canberra_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_canberra_table, gpa, "row_weighted_canberra", &.{ 0.0, 19.0 / 21.0, 0.0, 39.0 / 41.0 }, &.{ true, true, false, true });

    var row_weighted_bray_table = try validity_table.withRowWeightedBrayCurtisDistance(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_bray");
    defer row_weighted_bray_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_bray_table, gpa, "row_weighted_bray", &.{ 0.0, 19.0 / 21.0, 0.0, 39.0 / 73.0 }, &.{ true, true, false, true });

    var row_weighted_bias_table = try validity_table.withRowWeightedBias(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_bias");
    defer row_weighted_bias_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_bias_table, gpa, "row_weighted_bias", &.{ 0.0, 19.0, 0.0, 39.0 / 5.0 }, &.{ true, true, false, true });

    var row_weighted_mae_table = try validity_table.withRowWeightedMAE(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_mae");
    defer row_weighted_mae_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_mae_table, gpa, "row_weighted_mae", &.{ 0.0, 19.0, 0.0, 39.0 / 5.0 }, &.{ true, true, false, true });

    var row_weighted_mse_table = try validity_table.withRowWeightedMSE(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_mse");
    defer row_weighted_mse_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_mse_table, gpa, "row_weighted_mse", &.{ 0.0, 361.0, 0.0, 1521.0 / 5.0 }, &.{ true, true, false, true });

    var row_weighted_rmse_table = try validity_table.withRowWeightedRMSE(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_rmse");
    defer row_weighted_rmse_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_rmse_table, gpa, "row_weighted_rmse", &.{ 0.0, 19.0, 0.0, std.math.sqrt(@as(f64, 1521.0 / 5.0)) }, &.{ true, true, false, true });

    var row_weighted_mape_table = try validity_table.withRowWeightedMAPE(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_mape");
    defer row_weighted_mape_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_mape_table, gpa, "row_weighted_mape", &.{ 0.0, 19.0 / 20.0, 0.0, 39.0 / 200.0 }, &.{ true, true, false, true });

    var row_weighted_smape_table = try validity_table.withRowWeightedSMAPE(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_smape");
    defer row_weighted_smape_table.deinit();
    try expectF64ColumnApproxOrNanWithValidity(row_weighted_smape_table, gpa, "row_weighted_smape", &.{ 0.0, 38.0 / 21.0, 0.0, 78.0 / 205.0 }, &.{ true, true, false, true });

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
    try std.testing.expectError(error.InvalidShape, validity_table.withRowWeightedTrimmedMean(&.{ "a", "b" }, &.{ "wa", "wb" }, "bad_row_weighted_trimmed", 0.5));
    try std.testing.expectError(error.InvalidShape, validity_table.withRowWeightedWinsorizedMean(&.{ "a", "b" }, &.{ "wa", "wb" }, "bad_row_weighted_winsorized", -0.01));
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

    var row_cum_distinct_table = try validity_table.withRowCumulativeDistinctCount(
        &.{ "a", "b", "wa", "wb" },
        &.{ "a_cum_distinct", "b_cum_distinct", "wa_cum_distinct", "wb_cum_distinct" },
    );
    defer row_cum_distinct_table.deinit();
    const a_cum_distinct = try (try row_cum_distinct_table.column("a_cum_distinct")).i64.toOwnedSlice(gpa);
    defer gpa.free(a_cum_distinct);
    const b_cum_distinct = try (try row_cum_distinct_table.column("b_cum_distinct")).i64.toOwnedSlice(gpa);
    defer gpa.free(b_cum_distinct);
    const wa_cum_distinct = try (try row_cum_distinct_table.column("wa_cum_distinct")).i64.toOwnedSlice(gpa);
    defer gpa.free(wa_cum_distinct);
    const wb_cum_distinct = try (try row_cum_distinct_table.column("wb_cum_distinct")).i64.toOwnedSlice(gpa);
    defer gpa.free(wb_cum_distinct);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 0, 1 }, a_cum_distinct);
    try std.testing.expectEqualSlices(i64, &.{ 1, 1, 0, 2 }, b_cum_distinct);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 1, 2 }, wa_cum_distinct);
    try std.testing.expectEqualSlices(i64, &.{ 2, 3, 2, 3 }, wb_cum_distinct);
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowPrefixNUnique(&.{"a"}, &.{ "a_cum_unique", "extra_cum_unique" }));
    try std.testing.expectError(error.InvalidShape, validity_table.withRowQuantile(&.{ "a", "b" }, "bad_row_quantile", 1.5));
    try std.testing.expectError(error.InvalidShape, validity_table.withRowQuantileRange(&.{ "a", "b" }, "bad_row_quantile_range", 0.8, 0.2));
    try std.testing.expectError(error.InvalidShape, validity_table.withRowTrimmedMean(&.{ "a", "b" }, "bad_row_trimmed_mean", 0.5));
    try std.testing.expectError(error.InvalidShape, validity_table.withRowWinsorizedMean(&.{ "a", "b" }, "bad_row_winsorized_mean", 0.5));

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

    var row_logsumexp_table = try validity_table.withRowLogSumExp(&.{ "a", "b" }, "row_logsumexp");
    defer row_logsumexp_table.deinit();
    const row_logsumexp_column = try row_logsumexp_table.column("row_logsumexp");
    try std.testing.expect(row_logsumexp_column.f64.nullable());
    const row_logsumexp = try row_logsumexp_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_logsumexp);
    const row_logsumexp_validity = try row_logsumexp_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_logsumexp_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_logsumexp[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), row_logsumexp[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_logsumexp[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 40.0) + std.math.log1p(std.math.exp(@as(f64, -36.0))), row_logsumexp[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_logsumexp_validity);

    var row_logmeanexp_table = try validity_table.withRowLogMeanExp(&.{ "a", "b" }, "row_logmeanexp");
    defer row_logmeanexp_table.deinit();
    const row_logmeanexp_column = try row_logmeanexp_table.column("row_logmeanexp");
    try std.testing.expect(row_logmeanexp_column.f64.nullable());
    const row_logmeanexp = try row_logmeanexp_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_logmeanexp);
    const row_logmeanexp_validity = try row_logmeanexp_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_logmeanexp_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_logmeanexp[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), row_logmeanexp[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_logmeanexp[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 40.0) + std.math.log1p(std.math.exp(@as(f64, -36.0))) - std.math.ln2, row_logmeanexp[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_logmeanexp_validity);

    var row_centered_table = try validity_table.withRowCentered(&.{ "a", "b" }, &.{ "a_centered", "b_centered" });
    defer row_centered_table.deinit();
    const row_a_centered_column = try row_centered_table.column("a_centered");
    const row_b_centered_column = try row_centered_table.column("b_centered");
    try std.testing.expect(row_a_centered_column.f64.nullable());
    try std.testing.expect(row_b_centered_column.f64.nullable());
    const row_a_centered = try row_a_centered_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_centered);
    const row_b_centered = try row_b_centered_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_centered);
    const row_a_centered_validity = try row_a_centered_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_a_centered_validity);
    const row_b_centered_validity = try row_b_centered_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_centered_validity);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.0, -18.0 }, row_a_centered);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.0, 18.0 }, row_b_centered);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, row_a_centered_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_centered_validity);

    var row_zscore_table = try validity_table.withRowZScore(&.{ "a", "b" }, &.{ "a_zscore", "b_zscore" });
    defer row_zscore_table.deinit();
    const row_a_zscore = try (try row_zscore_table.column("a_zscore")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_zscore);
    const row_b_zscore = try (try row_zscore_table.column("b_zscore")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_zscore);
    const row_a_zscore_validity = try (try row_zscore_table.column("a_zscore")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_a_zscore_validity);
    const row_b_zscore_validity = try (try row_zscore_table.column("b_zscore")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_zscore_validity);
    try std.testing.expect(std.math.isNan(row_a_zscore[0]));
    try std.testing.expect(std.math.isNan(row_b_zscore[1]));
    try std.testing.expectApproxEqAbs(@as(f64, -1.0), row_a_zscore[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_b_zscore[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, row_a_zscore_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_zscore_validity);

    var row_dense_rank_table = try validity_table.withRowDenseRank(&.{ "a", "b", "wa", "wb" }, &.{ "a_row_dense_rank", "b_row_dense_rank", "wa_row_dense_rank", "wb_row_dense_rank" });
    defer row_dense_rank_table.deinit();
    const row_a_dense_rank_column = try row_dense_rank_table.column("a_row_dense_rank");
    try std.testing.expectEqual(DeviceDType.i64, row_a_dense_rank_column.dtype());
    try std.testing.expect(row_a_dense_rank_column.i64.nullable());
    const row_a_dense_rank = try row_a_dense_rank_column.i64.toOwnedSlice(gpa);
    defer gpa.free(row_a_dense_rank);
    const row_a_dense_rank_validity = try row_a_dense_rank_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_a_dense_rank_validity);
    const row_b_dense_rank_column = try row_dense_rank_table.column("b_row_dense_rank");
    try std.testing.expect(row_b_dense_rank_column.i64.nullable());
    const row_b_dense_rank = try row_b_dense_rank_column.i64.toOwnedSlice(gpa);
    defer gpa.free(row_b_dense_rank);
    const row_b_dense_rank_validity = try row_b_dense_rank_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_dense_rank_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 0, 2 }, row_a_dense_rank);
    try std.testing.expectEqualSlices(i64, &.{ 0, 3, 0, 3 }, row_b_dense_rank);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, row_a_dense_rank_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_dense_rank_validity);

    var row_ordinal_rank_table = try validity_table.withRowOrdinalRank(&.{ "a", "b", "wa", "wb" }, &.{ "a_row_ordinal_rank", "b_row_ordinal_rank", "wa_row_ordinal_rank", "wb_row_ordinal_rank" });
    defer row_ordinal_rank_table.deinit();
    const row_a_ordinal_rank_column = try row_ordinal_rank_table.column("a_row_ordinal_rank");
    try std.testing.expectEqual(DeviceDType.i64, row_a_ordinal_rank_column.dtype());
    try std.testing.expect(row_a_ordinal_rank_column.i64.nullable());
    const row_a_ordinal_rank = try row_a_ordinal_rank_column.i64.toOwnedSlice(gpa);
    defer gpa.free(row_a_ordinal_rank);
    const row_a_ordinal_rank_validity = try row_a_ordinal_rank_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_a_ordinal_rank_validity);
    const row_b_ordinal_rank_column = try row_ordinal_rank_table.column("b_row_ordinal_rank");
    try std.testing.expect(row_b_ordinal_rank_column.i64.nullable());
    const row_b_ordinal_rank = try row_b_ordinal_rank_column.i64.toOwnedSlice(gpa);
    defer gpa.free(row_b_ordinal_rank);
    const row_b_ordinal_rank_validity = try row_b_ordinal_rank_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_ordinal_rank_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 0, 2 }, row_a_ordinal_rank);
    try std.testing.expectEqualSlices(i64, &.{ 0, 3, 0, 4 }, row_b_ordinal_rank);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, row_a_ordinal_rank_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_ordinal_rank_validity);

    var row_average_rank_table = try validity_table.withRowAverageRank(&.{ "a", "b", "wa", "wb" }, &.{ "a_row_average_rank", "b_row_average_rank", "wa_row_average_rank", "wb_row_average_rank" });
    defer row_average_rank_table.deinit();
    const row_a_average_rank_column = try row_average_rank_table.column("a_row_average_rank");
    try std.testing.expectEqual(DeviceDType.f64, row_a_average_rank_column.dtype());
    try std.testing.expect(row_a_average_rank_column.f64.nullable());
    const row_a_average_rank = try row_a_average_rank_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_average_rank);
    const row_a_average_rank_validity = try row_a_average_rank_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_a_average_rank_validity);
    const row_b_average_rank_column = try row_average_rank_table.column("b_row_average_rank");
    try std.testing.expect(row_b_average_rank_column.f64.nullable());
    const row_b_average_rank = try row_b_average_rank_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_average_rank);
    const row_b_average_rank_validity = try row_b_average_rank_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_average_rank_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), row_a_average_rank[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), row_a_average_rank[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), row_b_average_rank[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), row_b_average_rank[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, row_a_average_rank_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_average_rank_validity);

    var row_competition_rank_table = try validity_table.withRowCompetitionRank(&.{ "a", "b", "wa", "wb" }, &.{ "a_row_competition_rank", "b_row_competition_rank", "wa_row_competition_rank", "wb_row_competition_rank" });
    defer row_competition_rank_table.deinit();
    const row_a_competition_rank_column = try row_competition_rank_table.column("a_row_competition_rank");
    try std.testing.expectEqual(DeviceDType.i64, row_a_competition_rank_column.dtype());
    try std.testing.expect(row_a_competition_rank_column.i64.nullable());
    const row_a_competition_rank = try row_a_competition_rank_column.i64.toOwnedSlice(gpa);
    defer gpa.free(row_a_competition_rank);
    const row_a_competition_rank_validity = try row_a_competition_rank_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_a_competition_rank_validity);
    const row_b_competition_rank_column = try row_competition_rank_table.column("b_row_competition_rank");
    try std.testing.expect(row_b_competition_rank_column.i64.nullable());
    const row_b_competition_rank = try row_b_competition_rank_column.i64.toOwnedSlice(gpa);
    defer gpa.free(row_b_competition_rank);
    const row_b_competition_rank_validity = try row_b_competition_rank_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_competition_rank_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 0, 2 }, row_a_competition_rank);
    try std.testing.expectEqualSlices(i64, &.{ 0, 3, 0, 4 }, row_b_competition_rank);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, row_a_competition_rank_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_competition_rank_validity);

    var row_percent_rank_table = try validity_table.withRowPercentRank(&.{ "a", "b", "wa", "wb" }, &.{ "a_row_percent_rank", "b_row_percent_rank", "wa_row_percent_rank", "wb_row_percent_rank" });
    defer row_percent_rank_table.deinit();
    const row_a_percent_rank_column = try row_percent_rank_table.column("a_row_percent_rank");
    try std.testing.expectEqual(DeviceDType.f64, row_a_percent_rank_column.dtype());
    try std.testing.expect(row_a_percent_rank_column.f64.nullable());
    const row_a_percent_rank = try row_a_percent_rank_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_percent_rank);
    const row_a_percent_rank_validity = try row_a_percent_rank_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_a_percent_rank_validity);
    const row_b_percent_rank_column = try row_percent_rank_table.column("b_row_percent_rank");
    try std.testing.expect(row_b_percent_rank_column.f64.nullable());
    const row_b_percent_rank = try row_b_percent_rank_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_percent_rank);
    const row_b_percent_rank_validity = try row_b_percent_rank_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_percent_rank_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_a_percent_rank[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), row_a_percent_rank[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_b_percent_rank[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_b_percent_rank[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, row_a_percent_rank_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_percent_rank_validity);

    var row_cume_dist_table = try validity_table.withRowCumeDist(&.{ "a", "b", "wa", "wb" }, &.{ "a_row_cume", "b_row_cume", "wa_row_cume", "wb_row_cume" });
    defer row_cume_dist_table.deinit();
    const row_a_cume_column = try row_cume_dist_table.column("a_row_cume");
    try std.testing.expectEqual(DeviceDType.f64, row_a_cume_column.dtype());
    try std.testing.expect(row_a_cume_column.f64.nullable());
    const row_a_cume = try row_a_cume_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_cume);
    const row_a_cume_validity = try row_a_cume_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_a_cume_validity);
    const row_b_cume_column = try row_cume_dist_table.column("b_row_cume");
    try std.testing.expect(row_b_cume_column.f64.nullable());
    const row_b_cume = try row_b_cume_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_cume);
    const row_b_cume_validity = try row_b_cume_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_cume_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), row_a_cume[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), row_a_cume[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_b_cume[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_b_cume[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, row_a_cume_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_cume_validity);

    var row_cumsum_table = try validity_table.withRowCumulativeSum(&.{ "a", "b", "wa", "wb" }, &.{ "a_row_cumsum", "b_row_cumsum", "wa_row_cumsum", "wb_row_cumsum" });
    defer row_cumsum_table.deinit();
    const row_a_cumsum_column = try row_cumsum_table.column("a_row_cumsum");
    try std.testing.expectEqual(DeviceDType.f64, row_a_cumsum_column.dtype());
    try std.testing.expect(row_a_cumsum_column.f64.nullable());
    const row_a_cumsum = try row_a_cumsum_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_cumsum);
    const row_a_cumsum_validity = try row_a_cumsum_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_a_cumsum_validity);
    const row_b_cumsum_column = try row_cumsum_table.column("b_row_cumsum");
    try std.testing.expect(row_b_cumsum_column.f64.nullable());
    const row_b_cumsum = try row_b_cumsum_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_cumsum);
    const row_b_cumsum_validity = try row_b_cumsum_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_cumsum_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 0.0, 0.0, 4.0 }, row_a_cumsum);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 20.0, 0.0, 44.0 }, row_b_cumsum);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, row_a_cumsum_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_cumsum_validity);

    var row_cummean_table = try validity_table.withRowCumulativeMean(&.{ "a", "b", "wa", "wb" }, &.{ "a_row_cummean", "b_row_cummean", "wa_row_cummean", "wb_row_cummean" });
    defer row_cummean_table.deinit();
    const row_a_cummean_column = try row_cummean_table.column("a_row_cummean");
    try std.testing.expectEqual(DeviceDType.f64, row_a_cummean_column.dtype());
    try std.testing.expect(row_a_cummean_column.f64.nullable());
    const row_a_cummean = try row_a_cummean_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_cummean);
    const row_a_cummean_validity = try row_a_cummean_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_a_cummean_validity);
    const row_b_cummean_column = try row_cummean_table.column("b_row_cummean");
    try std.testing.expect(row_b_cummean_column.f64.nullable());
    const row_b_cummean = try row_b_cummean_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_cummean);
    const row_b_cummean_validity = try row_b_cummean_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_cummean_validity);
    const row_wb_cummean = try (try row_cummean_table.column("wb_row_cummean")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cummean);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 0.0, 0.0, 4.0 }, row_a_cummean);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 20.0, 0.0, 22.0 }, row_b_cummean);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0 / 3.0), row_wb_cummean[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 23.0 / 3.0), row_wb_cummean[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), row_wb_cummean[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 12.25), row_wb_cummean[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, row_a_cummean_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_cummean_validity);

    var row_cum_lse_table = try validity_table.withRowCumulativeLogSumExp(&.{ "a", "b", "wa", "wb" }, &.{ "a_row_cumlse", "b_row_cumlse", "wa_row_cumlse", "wb_row_cumlse" });
    defer row_cum_lse_table.deinit();
    const row_b_cumlse = try (try row_cum_lse_table.column("b_row_cumlse")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_cumlse);
    const row_wb_cumlse = try (try row_cum_lse_table.column("wb_row_cumlse")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cumlse);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), row_b_cumlse[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 40.0) + std.math.log1p(std.math.exp(@as(f64, -36.0))), row_b_cumlse[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0) + std.math.log1p(@as(f64, 2.0) * std.math.exp(@as(f64, -1.0))), row_wb_cumlse[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0) + std.math.log1p(std.math.exp(@as(f64, -18.0)) + std.math.exp(@as(f64, -19.0))), row_wb_cumlse[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0) + std.math.log1p(std.math.exp(@as(f64, -2.0))), row_wb_cumlse[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 40.0) + std.math.log1p(@as(f64, 2.0) * std.math.exp(@as(f64, -36.0)) + std.math.exp(@as(f64, -39.0))), row_wb_cumlse[3], 1e-12);

    var row_cum_lme_table = try validity_table.withRowPrefixLogMeanExp(&.{ "a", "b", "wa", "wb" }, &.{ "a_row_cumlme", "b_row_cumlme", "wa_row_cumlme", "wb_row_cumlme" });
    defer row_cum_lme_table.deinit();
    const row_wb_cumlme = try (try row_cum_lme_table.column("wb_row_cumlme")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cumlme);
    try std.testing.expectApproxEqAbs(row_wb_cumlse[0] - std.math.log(f64, std.math.e, 3.0), row_wb_cumlme[0], 1e-12);
    try std.testing.expectApproxEqAbs(row_wb_cumlse[1] - std.math.log(f64, std.math.e, 3.0), row_wb_cumlme[1], 1e-12);
    try std.testing.expectApproxEqAbs(row_wb_cumlse[2] - std.math.ln2, row_wb_cumlme[2], 1e-12);
    try std.testing.expectApproxEqAbs(row_wb_cumlse[3] - std.math.log(f64, std.math.e, 4.0), row_wb_cumlme[3], 1e-12);

    var row_cumgeo_table = try validity_table.withRowCumulativeGeometricMean(&.{ "a", "b", "wa", "wb" }, &.{ "a_row_cumgeo", "b_row_cumgeo", "wa_row_cumgeo", "wb_row_cumgeo" });
    defer row_cumgeo_table.deinit();
    const row_wb_cumgeo = try (try row_cumgeo_table.column("wb_row_cumgeo")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cumgeo);
    try std.testing.expectApproxEqAbs(std.math.pow(f64, 2.0, 1.0 / 3.0), row_wb_cumgeo[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.pow(f64, 40.0, 1.0 / 3.0), row_wb_cumgeo[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 15.0)), row_wb_cumgeo[2], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.pow(f64, 640.0, 0.25), row_wb_cumgeo[3], 1e-12);

    var row_cumharm_table = try validity_table.withRowPrefixHarmonicMean(&.{ "a", "b", "wa", "wb" }, &.{ "a_row_cumharm", "b_row_cumharm", "wa_row_cumharm", "wb_row_cumharm" });
    defer row_cumharm_table.deinit();
    const row_wb_cumharm = try (try row_cumharm_table.column("wb_row_cumharm")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cumharm);
    try std.testing.expectApproxEqAbs(@as(f64, 1.2), row_wb_cumharm[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 60.0 / 31.0), row_wb_cumharm[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 30.0 / 8.0), row_wb_cumharm[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 160.0 / 61.0), row_wb_cumharm[3], 1e-12);

    var row_cumvar_table = try validity_table.withRowCumulativeVariance(&.{ "a", "b", "wa", "wb" }, &.{ "a_row_cumvar", "b_row_cumvar", "wa_row_cumvar", "wb_row_cumvar" }, 0.0);
    defer row_cumvar_table.deinit();
    const row_b_cumvar_column = try row_cumvar_table.column("b_row_cumvar");
    try std.testing.expectEqual(DeviceDType.f64, row_b_cumvar_column.dtype());
    try std.testing.expect(row_b_cumvar_column.f64.nullable());
    const row_b_cumvar = try row_b_cumvar_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_cumvar);
    const row_b_cumvar_validity = try row_b_cumvar_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_cumvar_validity);
    const row_wb_cumvar = try (try row_cumvar_table.column("wb_row_cumvar")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cumvar);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.0, 324.0 }, row_b_cumvar);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 9.0), row_wb_cumvar[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 686.0 / 9.0), row_wb_cumvar[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_wb_cumvar[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4131.0 / 16.0), row_wb_cumvar[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_cumvar_validity);

    var row_cumstd_table = try validity_table.withRowPrefixStddev(&.{ "a", "b", "wa", "wb" }, &.{ "a_row_cumstd", "b_row_cumstd", "wa_row_cumstd", "wb_row_cumstd" }, 0.0);
    defer row_cumstd_table.deinit();
    const row_b_cumstd_column = try row_cumstd_table.column("b_row_cumstd");
    try std.testing.expect(row_b_cumstd_column.f64.nullable());
    const row_b_cumstd = try row_b_cumstd_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_cumstd);
    const row_b_cumstd_validity = try row_b_cumstd_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_cumstd_validity);
    const row_wb_cumstd = try (try row_cumstd_table.column("wb_row_cumstd")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cumstd);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.0, 18.0 }, row_b_cumstd);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 2.0 / 9.0)), row_wb_cumstd[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 686.0 / 9.0)), row_wb_cumstd[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_wb_cumstd[2], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 4131.0 / 16.0)), row_wb_cumstd[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_cumstd_validity);

    var row_cumsem_table = try validity_table.withRowCumulativeSem(&.{ "a", "b", "wa", "wb" }, &.{ "a_row_cumsem", "b_row_cumsem", "wa_row_cumsem", "wb_row_cumsem" }, 0.0);
    defer row_cumsem_table.deinit();
    const row_wb_cumsem = try (try row_cumsem_table.column("wb_row_cumsem")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cumsem);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 2.0 / 9.0)) / std.math.sqrt(@as(f64, 3.0)), row_wb_cumsem[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 686.0 / 9.0)) / std.math.sqrt(@as(f64, 3.0)), row_wb_cumsem[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0) / std.math.sqrt(@as(f64, 2.0)), row_wb_cumsem[2], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 4131.0 / 16.0)) / 2.0, row_wb_cumsem[3], 1e-12);

    var row_cumcv_table = try validity_table.withRowPrefixCv(&.{ "a", "b", "wa", "wb" }, &.{ "a_row_cumcv", "b_row_cumcv", "wa_row_cumcv", "wb_row_cumcv" }, 0.0);
    defer row_cumcv_table.deinit();
    const row_wb_cumcv = try (try row_cumcv_table.column("wb_row_cumcv")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cumcv);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 2.0 / 9.0)) / @as(f64, 4.0 / 3.0), row_wb_cumcv[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 686.0 / 9.0)) / @as(f64, 23.0 / 3.0), row_wb_cumcv[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), row_wb_cumcv[2], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 4131.0 / 16.0)) / @as(f64, 12.25), row_wb_cumcv[3], 1e-12);

    var row_cumfano_table = try validity_table.withRowCumulativeIndexOfDispersion(&.{ "a", "b", "wa", "wb" }, &.{ "a_row_cumfano", "b_row_cumfano", "wa_row_cumfano", "wb_row_cumfano" }, 0.0);
    defer row_cumfano_table.deinit();
    const row_wb_cumfano = try (try row_cumfano_table.column("wb_row_cumfano")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cumfano);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 6.0), row_wb_cumfano[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 686.0 / 69.0), row_wb_cumfano[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), row_wb_cumfano[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4131.0 / 196.0), row_wb_cumfano[3], 1e-12);

    var row_cumskew_table = try validity_table.withRowCumulativeSkewness(&.{ "a", "b", "wa", "wb" }, &.{ "a_row_cumskew", "b_row_cumskew", "wa_row_cumskew", "wb_row_cumskew" });
    defer row_cumskew_table.deinit();
    const row_wb_cumskew = try (try row_cumskew_table.column("wb_row_cumskew")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cumskew);
    try std.testing.expectApproxEqAbs(@as(f64, 0.7071067811865479), row_wb_cumskew[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.7001554400787792), row_wb_cumskew[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_wb_cumskew[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.134257375254061), row_wb_cumskew[3], 1e-12);

    var row_cumkurt_table = try validity_table.withRowPrefixKurtosis(&.{ "a", "b", "wa", "wb" }, &.{ "a_row_cumkurt", "b_row_cumkurt", "wa_row_cumkurt", "wb_row_cumkurt" });
    defer row_cumkurt_table.deinit();
    const row_wb_cumkurt = try (try row_cumkurt_table.column("wb_row_cumkurt")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cumkurt);
    try std.testing.expectApproxEqAbs(@as(f64, -1.5), row_wb_cumkurt[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -1.5), row_wb_cumkurt[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -2.0), row_wb_cumkurt[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.6812479530664843), row_wb_cumkurt[3], 1e-12);

    var row_cumrms_table = try validity_table.withRowCumulativeRms(&.{ "a", "b", "wa", "wb" }, &.{ "a_row_cumrms", "b_row_cumrms", "wa_row_cumrms", "wb_row_cumrms" });
    defer row_cumrms_table.deinit();
    const row_b_cumrms = try (try row_cumrms_table.column("b_row_cumrms")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_cumrms);
    const row_wb_cumrms = try (try row_cumrms_table.column("wb_row_cumrms")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cumrms);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 20.0, 0.0, std.math.sqrt(@as(f64, 808.0)) }, row_b_cumrms);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 2.0)), row_wb_cumrms[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 135.0)), row_wb_cumrms[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 17.0)), row_wb_cumrms[2], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 1633.0 / 4.0)), row_wb_cumrms[3], 1e-12);

    var row_cummeanabs_table = try validity_table.withRowPrefixMeanAbs(&.{ "a", "b", "wa", "wb" }, &.{ "a_row_cummeanabs", "b_row_cummeanabs", "wa_row_cummeanabs", "wb_row_cummeanabs" });
    defer row_cummeanabs_table.deinit();
    const row_b_cummeanabs_column = try row_cummeanabs_table.column("b_row_cummeanabs");
    try std.testing.expect(row_b_cummeanabs_column.f64.nullable());
    const row_b_cummeanabs = try row_b_cummeanabs_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_cummeanabs);
    const row_b_cummeanabs_validity = try row_b_cummeanabs_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_cummeanabs_validity);
    const row_wb_cummeanabs = try (try row_cummeanabs_table.column("wb_row_cummeanabs")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cummeanabs);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 20.0, 0.0, 22.0 }, row_b_cummeanabs);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_cummeanabs_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0 / 3.0), row_wb_cummeanabs[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 23.0 / 3.0), row_wb_cummeanabs[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), row_wb_cummeanabs[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 49.0 / 4.0), row_wb_cummeanabs[3], 1e-12);

    var row_cummeansq_table = try validity_table.withRowCumulativeMeanSquared(&.{ "a", "b", "wa", "wb" }, &.{ "a_row_cummeansq", "b_row_cummeansq", "wa_row_cummeansq", "wb_row_cummeansq" });
    defer row_cummeansq_table.deinit();
    const row_b_cummeansq = try (try row_cummeansq_table.column("b_row_cummeansq")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_cummeansq);
    const row_wb_cummeansq = try (try row_cummeansq_table.column("wb_row_cummeansq")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cummeansq);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 400.0, 0.0, 808.0 }, row_b_cummeansq);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), row_wb_cummeansq[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 135.0), row_wb_cummeansq[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 17.0), row_wb_cummeansq[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1633.0 / 4.0), row_wb_cummeansq[3], 1e-12);

    var row_cummaxabs_table = try validity_table.withRowPrefixLInfNorm(&.{ "a", "b", "wa", "wb" }, &.{ "a_row_cummaxabs", "b_row_cummaxabs", "wa_row_cummaxabs", "wb_row_cummaxabs" });
    defer row_cummaxabs_table.deinit();
    const row_b_cummaxabs_column = try row_cummaxabs_table.column("b_row_cummaxabs");
    try std.testing.expect(row_b_cummaxabs_column.f64.nullable());
    const row_b_cummaxabs = try row_b_cummaxabs_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_cummaxabs);
    const row_b_cummaxabs_validity = try row_b_cummaxabs_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_cummaxabs_validity);
    const row_wb_cummaxabs = try (try row_cummaxabs_table.column("wb_row_cummaxabs")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cummaxabs);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 20.0, 0.0, 40.0 }, row_b_cummaxabs);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_cummaxabs_validity);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 20.0, 5.0, 40.0 }, row_wb_cummaxabs);

    var row_cumminabs_table = try validity_table.withRowCumulativeMinAbsolute(&.{ "a", "b", "wa", "wb" }, &.{ "a_row_cumminabs", "b_row_cumminabs", "wa_row_cumminabs", "wb_row_cumminabs" });
    defer row_cumminabs_table.deinit();
    const row_wb_cumminabs = try (try row_cumminabs_table.column("wb_row_cumminabs")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cumminabs);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 1.0, 3.0, 1.0 }, row_wb_cumminabs);

    var row_cuml1_table = try validity_table.withRowPrefixL1Norm(&.{ "a", "b", "wa", "wb" }, &.{ "a_row_cuml1", "b_row_cuml1", "wa_row_cuml1", "wb_row_cuml1" });
    defer row_cuml1_table.deinit();
    const row_wb_cuml1 = try (try row_cuml1_table.column("wb_row_cuml1")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cuml1);
    try std.testing.expectEqualSlices(f64, &.{ 4.0, 23.0, 8.0, 49.0 }, row_wb_cuml1);

    var row_cuml2_table = try validity_table.withRowCumulativeL2Norm(&.{ "a", "b", "wa", "wb" }, &.{ "a_row_cuml2", "b_row_cuml2", "wa_row_cuml2", "wb_row_cuml2" });
    defer row_cuml2_table.deinit();
    const row_wb_cuml2 = try (try row_cuml2_table.column("wb_row_cuml2")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cuml2);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 6.0)), row_wb_cuml2[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 405.0)), row_wb_cuml2[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 34.0)), row_wb_cuml2[2], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 1633.0)), row_wb_cuml2[3], 1e-12);

    try std.testing.expectError(error.LengthMismatch, validity_table.withRowPrefixLogsumexp(&.{"a"}, &.{ "a_row_cumlse", "extra_row_cumlse" }));
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowCumGeoMean(&.{"a"}, &.{ "a_row_cumgeo", "extra_row_cumgeo" }));
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowCumVar(&.{"a"}, &.{ "a_row_cumvar", "extra_row_cumvar" }, 0.0));
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowPrefixSkew(&.{"a"}, &.{ "a_row_cumskew", "extra_row_cumskew" }));
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowPrefixMeanSquared(&.{"a"}, &.{ "a_row_cummeansq", "extra_row_cummeansq" }));
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowCumLInfNorm(&.{"a"}, &.{ "a_row_cummaxabs", "extra_row_cummaxabs" }));
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowCumL2Norm(&.{"a"}, &.{ "a_row_cuml2", "extra_row_cuml2" }));
    try std.testing.expectError(error.InvalidShape, validity_table.withRowPrefixStd(&.{"a"}, &.{"a_row_cumstd"}, -1.0));

    var row_cumprod_table = try validity_table.withRowCumulativeProduct(&.{ "a", "b", "wa", "wb" }, &.{ "a_row_cumprod", "b_row_cumprod", "wa_row_cumprod", "wb_row_cumprod" });
    defer row_cumprod_table.deinit();
    const row_a_cumprod_column = try row_cumprod_table.column("a_row_cumprod");
    try std.testing.expectEqual(DeviceDType.f64, row_a_cumprod_column.dtype());
    try std.testing.expect(row_a_cumprod_column.f64.nullable());
    const row_a_cumprod = try row_a_cumprod_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_cumprod);
    const row_a_cumprod_validity = try row_a_cumprod_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_a_cumprod_validity);
    const row_b_cumprod_column = try row_cumprod_table.column("b_row_cumprod");
    try std.testing.expect(row_b_cumprod_column.f64.nullable());
    const row_b_cumprod = try row_b_cumprod_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_cumprod);
    const row_b_cumprod_validity = try row_b_cumprod_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_cumprod_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 0.0, 0.0, 4.0 }, row_a_cumprod);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 20.0, 0.0, 160.0 }, row_b_cumprod);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, row_a_cumprod_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_cumprod_validity);

    var row_cummax_table = try validity_table.withRowCumulativeMax(&.{ "a", "b", "wa", "wb" }, &.{ "a_row_cummax", "b_row_cummax", "wa_row_cummax", "wb_row_cummax" });
    defer row_cummax_table.deinit();
    const row_a_cummax_column = try row_cummax_table.column("a_row_cummax");
    try std.testing.expectEqual(DeviceDType.f64, row_a_cummax_column.dtype());
    try std.testing.expect(row_a_cummax_column.f64.nullable());
    const row_a_cummax = try row_a_cummax_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_cummax);
    const row_a_cummax_validity = try row_a_cummax_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_a_cummax_validity);
    const row_b_cummax_column = try row_cummax_table.column("b_row_cummax");
    try std.testing.expect(row_b_cummax_column.f64.nullable());
    const row_b_cummax = try row_b_cummax_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_cummax);
    const row_b_cummax_validity = try row_b_cummax_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_cummax_validity);
    const row_wa_cummax = try (try row_cummax_table.column("wa_row_cummax")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wa_cummax);
    const row_wb_cummax = try (try row_cummax_table.column("wb_row_cummax")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cummax);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 0.0, 0.0, 4.0 }, row_a_cummax);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 20.0, 0.0, 40.0 }, row_b_cummax);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 3.0, 40.0 }, row_wa_cummax);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 20.0, 5.0, 40.0 }, row_wb_cummax);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, row_a_cummax_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_cummax_validity);

    var row_cummin_table = try validity_table.withRowCumulativeMin(&.{ "a", "b", "wa", "wb" }, &.{ "a_row_cummin", "b_row_cummin", "wa_row_cummin", "wb_row_cummin" });
    defer row_cummin_table.deinit();
    const row_a_cummin_column = try row_cummin_table.column("a_row_cummin");
    try std.testing.expectEqual(DeviceDType.f64, row_a_cummin_column.dtype());
    try std.testing.expect(row_a_cummin_column.f64.nullable());
    const row_a_cummin = try row_a_cummin_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_cummin);
    const row_a_cummin_validity = try row_a_cummin_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_a_cummin_validity);
    const row_b_cummin_column = try row_cummin_table.column("b_row_cummin");
    try std.testing.expect(row_b_cummin_column.f64.nullable());
    const row_b_cummin = try row_b_cummin_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_cummin);
    const row_b_cummin_validity = try row_b_cummin_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_cummin_validity);
    const row_wa_cummin = try (try row_cummin_table.column("wa_row_cummin")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wa_cummin);
    const row_wb_cummin = try (try row_cummin_table.column("wb_row_cummin")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cummin);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 0.0, 0.0, 4.0 }, row_a_cummin);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 20.0, 0.0, 4.0 }, row_b_cummin);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 2.0, 3.0, 4.0 }, row_wa_cummin);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 1.0, 3.0, 1.0 }, row_wb_cummin);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, row_a_cummin_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_cummin_validity);

    var row_cumrange_table = try validity_table.withRowCumulativeRange(&.{ "a", "b", "wa", "wb" }, &.{ "a_row_cumrange", "b_row_cumrange", "wa_row_cumrange", "wb_row_cumrange" });
    defer row_cumrange_table.deinit();
    const row_a_cumrange_column = try row_cumrange_table.column("a_row_cumrange");
    try std.testing.expectEqual(DeviceDType.f64, row_a_cumrange_column.dtype());
    try std.testing.expect(row_a_cumrange_column.f64.nullable());
    const row_a_cumrange = try row_a_cumrange_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_cumrange);
    const row_a_cumrange_validity = try row_a_cumrange_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_a_cumrange_validity);
    const row_b_cumrange_column = try row_cumrange_table.column("b_row_cumrange");
    try std.testing.expect(row_b_cumrange_column.f64.nullable());
    const row_b_cumrange = try row_b_cumrange_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_cumrange);
    const row_b_cumrange_validity = try row_b_cumrange_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_cumrange_validity);
    const row_wa_cumrange = try (try row_cumrange_table.column("wa_row_cumrange")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wa_cumrange);
    const row_wb_cumrange = try (try row_cumrange_table.column("wb_row_cumrange")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cumrange);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.0, 0.0 }, row_a_cumrange);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.0, 36.0 }, row_b_cumrange);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 18.0, 0.0, 36.0 }, row_wa_cumrange);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 19.0, 2.0, 39.0 }, row_wb_cumrange);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, row_a_cumrange_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_cumrange_validity);

    var row_robust_zscore_table = try validity_table.withRowRobustZScore(&.{ "a", "b" }, &.{ "a_robust_zscore", "b_robust_zscore" });
    defer row_robust_zscore_table.deinit();
    const row_a_robust_zscore_column = try row_robust_zscore_table.column("a_robust_zscore");
    const row_b_robust_zscore_column = try row_robust_zscore_table.column("b_robust_zscore");
    try std.testing.expect(row_a_robust_zscore_column.f64.nullable());
    try std.testing.expect(row_b_robust_zscore_column.f64.nullable());
    const row_a_robust_zscore = try row_a_robust_zscore_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_robust_zscore);
    const row_b_robust_zscore = try row_b_robust_zscore_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_robust_zscore);
    const row_a_robust_zscore_validity = try row_a_robust_zscore_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_a_robust_zscore_validity);
    const row_b_robust_zscore_validity = try row_b_robust_zscore_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_robust_zscore_validity);
    try std.testing.expect(std.math.isNan(row_a_robust_zscore[0]));
    try std.testing.expect(std.math.isNan(row_b_robust_zscore[1]));
    try std.testing.expectApproxEqAbs(-@as(f64, 0.6744897501960817), row_a_robust_zscore[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.6744897501960817), row_b_robust_zscore[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, row_a_robust_zscore_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_robust_zscore_validity);

    var row_iqr_outlier_table = try validity_table.withRowIqrOutlier(
        &.{ "a", "b", "wa", "wb" },
        &.{ "a_iqr_outlier", "b_iqr_outlier", "wa_iqr_outlier", "wb_iqr_outlier" },
    );
    defer row_iqr_outlier_table.deinit();
    const row_b_iqr_outlier_column = try row_iqr_outlier_table.column("b_iqr_outlier");
    try std.testing.expectEqual(DeviceDType.bool, row_b_iqr_outlier_column.dtype());
    try std.testing.expect(row_b_iqr_outlier_column.bool.nullable());
    const row_b_iqr_outlier = try row_b_iqr_outlier_column.bool.toOwnedSlice(gpa);
    defer gpa.free(row_b_iqr_outlier);
    const row_b_iqr_outlier_validity = try row_b_iqr_outlier_column.bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_iqr_outlier_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, true }, row_b_iqr_outlier);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_iqr_outlier_validity);

    var row_tukey_winsor_table = try validity_table.withRowTukeyWinsorize(
        &.{ "a", "b", "wa", "wb" },
        &.{ "a_tukey_winsor", "b_tukey_winsor", "wa_tukey_winsor", "wb_tukey_winsor" },
    );
    defer row_tukey_winsor_table.deinit();
    const row_b_tukey_winsor_column = try row_tukey_winsor_table.column("b_tukey_winsor");
    try std.testing.expectEqual(DeviceDType.f64, row_b_tukey_winsor_column.dtype());
    try std.testing.expect(row_b_tukey_winsor_column.f64.nullable());
    const row_b_tukey_winsor = try row_b_tukey_winsor_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_tukey_winsor);
    const row_b_tukey_winsor_validity = try row_b_tukey_winsor_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_tukey_winsor_validity);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 20.0, 0.0, 27.625 }, row_b_tukey_winsor);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_tukey_winsor_validity);

    var row_max_indicator_table = try validity_table.withRowMaxIndicator(
        &.{ "a", "b", "wa", "wb" },
        &.{ "a_row_is_max", "b_row_is_max", "wa_row_is_max", "wb_row_is_max" },
    );
    defer row_max_indicator_table.deinit();
    const row_a_is_max_column = try row_max_indicator_table.column("a_row_is_max");
    try std.testing.expectEqual(DeviceDType.bool, row_a_is_max_column.dtype());
    try std.testing.expect(row_a_is_max_column.bool.nullable());
    const row_a_is_max = try row_a_is_max_column.bool.toOwnedSlice(gpa);
    defer gpa.free(row_a_is_max);
    const row_a_is_max_validity = try row_a_is_max_column.bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_a_is_max_validity);
    const row_b_is_max_column = try row_max_indicator_table.column("b_row_is_max");
    try std.testing.expect(row_b_is_max_column.bool.nullable());
    const row_b_is_max = try row_b_is_max_column.bool.toOwnedSlice(gpa);
    defer gpa.free(row_b_is_max);
    const row_b_is_max_validity = try row_b_is_max_column.bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_is_max_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false }, row_a_is_max);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_is_max);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, row_a_is_max_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_is_max_validity);

    var row_min_indicator_table = try validity_table.withRowMinIndicator(
        &.{ "a", "b", "wa", "wb" },
        &.{ "a_row_is_min", "b_row_is_min", "wa_row_is_min", "wb_row_is_min" },
    );
    defer row_min_indicator_table.deinit();
    const row_a_is_min_column = try row_min_indicator_table.column("a_row_is_min");
    try std.testing.expectEqual(DeviceDType.bool, row_a_is_min_column.dtype());
    try std.testing.expect(row_a_is_min_column.bool.nullable());
    const row_a_is_min = try row_a_is_min_column.bool.toOwnedSlice(gpa);
    defer gpa.free(row_a_is_min);
    const row_a_is_min_validity = try row_a_is_min_column.bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_a_is_min_validity);
    const row_b_is_min_column = try row_min_indicator_table.column("b_row_is_min");
    try std.testing.expect(row_b_is_min_column.bool.nullable());
    const row_b_is_min = try row_b_is_min_column.bool.toOwnedSlice(gpa);
    defer gpa.free(row_b_is_min);
    const row_b_is_min_validity = try row_b_is_min_column.bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_is_min_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false }, row_a_is_min);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false }, row_b_is_min);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, row_a_is_min_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_is_min_validity);

    var row_minmax_table = try validity_table.withRowMinMaxScale(&.{ "a", "b" }, &.{ "a_minmax", "b_minmax" });
    defer row_minmax_table.deinit();
    const row_a_minmax = try (try row_minmax_table.column("a_minmax")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_minmax);
    const row_b_minmax = try (try row_minmax_table.column("b_minmax")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_minmax);
    try std.testing.expect(std.math.isNan(row_a_minmax[0]));
    try std.testing.expect(std.math.isNan(row_b_minmax[1]));
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_a_minmax[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_b_minmax[3], 1e-12);

    var row_l2_unit_table = try validity_table.withRowL2Normalize(&.{ "a", "b" }, &.{ "a_l2_unit", "b_l2_unit" });
    defer row_l2_unit_table.deinit();
    const row_a_l2 = try (try row_l2_unit_table.column("a_l2_unit")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_l2);
    const row_b_l2 = try (try row_l2_unit_table.column("b_l2_unit")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_l2);
    const row3_l2_norm = std.math.sqrt(@as(f64, 1616.0));
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_a_l2[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_b_l2[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0) / row3_l2_norm, row_a_l2[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 40.0) / row3_l2_norm, row_b_l2[3], 1e-12);

    var row_l1_unit_table = try validity_table.withRowL1Normalize(&.{ "a", "b" }, &.{ "a_l1_unit", "b_l1_unit" });
    defer row_l1_unit_table.deinit();
    const row_a_l1 = try (try row_l1_unit_table.column("a_l1_unit")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_l1);
    const row_b_l1 = try (try row_l1_unit_table.column("b_l1_unit")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_l1);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_a_l1[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_b_l1[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 11.0), row_a_l1[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0 / 11.0), row_b_l1[3], 1e-12);

    var row_share_table = try validity_table.withRowSumNormalize(&.{ "a", "b" }, &.{ "a_share", "b_share" });
    defer row_share_table.deinit();
    const row_a_share = try (try row_share_table.column("a_share")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_share);
    const row_b_share = try (try row_share_table.column("b_share")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_share);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_a_share[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_b_share[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 11.0), row_a_share[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0 / 11.0), row_b_share[3], 1e-12);

    var row_mean_ratio_table = try validity_table.withRowMeanNormalize(&.{ "a", "b" }, &.{ "a_mean_ratio", "b_mean_ratio" });
    defer row_mean_ratio_table.deinit();
    const row_a_mean_ratio_column = try row_mean_ratio_table.column("a_mean_ratio");
    const row_b_mean_ratio_column = try row_mean_ratio_table.column("b_mean_ratio");
    try std.testing.expect(row_a_mean_ratio_column.f64.nullable());
    try std.testing.expect(row_b_mean_ratio_column.f64.nullable());
    const row_a_mean_ratio = try row_a_mean_ratio_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_mean_ratio);
    const row_b_mean_ratio = try row_b_mean_ratio_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_mean_ratio);
    const row_a_mean_ratio_validity = try row_a_mean_ratio_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_a_mean_ratio_validity);
    const row_b_mean_ratio_validity = try row_b_mean_ratio_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_mean_ratio_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_a_mean_ratio[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_b_mean_ratio[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 11.0), row_a_mean_ratio[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0 / 11.0), row_b_mean_ratio[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, row_a_mean_ratio_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_mean_ratio_validity);

    var mean_zero_left = try DeviceColumn.fromSlice(f64, gpa, &.{ 1.0, 0.0 }, .cpu);
    defer mean_zero_left.deinit();
    var mean_zero_right = try DeviceColumn.fromSlice(f64, gpa, &.{ -1.0, 0.0 }, .cpu);
    defer mean_zero_right.deinit();
    var mean_zero_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "left", .data = mean_zero_left },
        .{ .name = "right", .data = mean_zero_right },
    });
    defer mean_zero_table.deinit();
    var mean_zero_ratio_table = try mean_zero_table.withRowMeanNormalized(&.{ "left", "right" }, &.{ "left_mean_ratio", "right_mean_ratio" });
    defer mean_zero_ratio_table.deinit();
    const left_mean_ratio = try (try mean_zero_ratio_table.column("left_mean_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(left_mean_ratio);
    const right_mean_ratio = try (try mean_zero_ratio_table.column("right_mean_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(right_mean_ratio);
    try std.testing.expect(std.math.isNan(left_mean_ratio[0]));
    try std.testing.expect(std.math.isNan(right_mean_ratio[0]));
    try std.testing.expect(std.math.isNan(left_mean_ratio[1]));
    try std.testing.expect(std.math.isNan(right_mean_ratio[1]));

    var row_maxabs_table = try validity_table.withRowMaxAbsNormalize(&.{ "a", "b" }, &.{ "a_maxabs", "b_maxabs" });
    defer row_maxabs_table.deinit();
    const row_a_maxabs = try (try row_maxabs_table.column("a_maxabs")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_maxabs);
    const row_b_maxabs = try (try row_maxabs_table.column("b_maxabs")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_maxabs);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_a_maxabs[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_b_maxabs[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.1), row_a_maxabs[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_b_maxabs[3], 1e-12);
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowCentered(&.{"a"}, &.{ "a_centered", "extra_centered" }));
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowFractionalRanks(&.{"a"}, &.{ "a_row_average_rank", "extra_row_average_rank" }));
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowOrdinalRanks(&.{"a"}, &.{ "a_row_ordinal_rank", "extra_row_ordinal_rank" }));
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowDenseRanks(&.{"a"}, &.{ "a_row_dense_rank", "extra_row_dense_rank" }));
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowMinRanks(&.{"a"}, &.{ "a_row_min_rank", "extra_row_min_rank" }));
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowPercentileRanks(&.{"a"}, &.{ "a_row_percent_rank", "extra_row_percent_rank" }));
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowCumulativeDistribution(&.{"a"}, &.{ "a_row_cume", "extra_row_cume" }));
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowPrefixSum(&.{"a"}, &.{ "a_row_cumsum", "extra_row_cumsum" }));
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowPrefixAvg(&.{"a"}, &.{ "a_row_cummean", "extra_row_cummean" }));
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowPrefixProduct(&.{"a"}, &.{ "a_row_cumprod", "extra_row_cumprod" }));
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowPrefixMax(&.{"a"}, &.{ "a_row_cummax", "extra_row_cummax" }));
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowPrefixMin(&.{"a"}, &.{ "a_row_cummin", "extra_row_cummin" }));
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowPrefixPtp(&.{"a"}, &.{ "a_row_cumrange", "extra_row_cumrange" }));
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowRobustZscore(&.{"a"}, &.{ "a_robust_zscore", "extra_robust_zscore" }));
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowTukeyOutliers(&.{"a"}, &.{ "a_iqr_outlier", "extra_iqr_outlier" }));
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowIqrWinsorized(&.{"a"}, &.{ "a_tukey_winsor", "extra_tukey_winsor" }));
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowMaxMask(&.{"a"}, &.{ "a_row_is_max", "extra_row_is_max" }));
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowMinMask(&.{"a"}, &.{ "a_row_is_min", "extra_row_is_min" }));
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowMeanNormalize(&.{"a"}, &.{ "a_mean_ratio", "extra_mean_ratio" }));

    var row_softmax_table = try validity_table.withRowSoftmax(&.{ "a", "b" }, &.{ "a_softmax", "b_softmax" });
    defer row_softmax_table.deinit();
    const row_a_softmax_column = try row_softmax_table.column("a_softmax");
    const row_b_softmax_column = try row_softmax_table.column("b_softmax");
    try std.testing.expect(row_a_softmax_column.f64.nullable());
    try std.testing.expect(row_b_softmax_column.f64.nullable());
    const row_a_softmax = try row_a_softmax_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_softmax);
    const row_b_softmax = try row_b_softmax_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_softmax);
    const row_a_softmax_validity = try row_a_softmax_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_a_softmax_validity);
    const row_b_softmax_validity = try row_b_softmax_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_softmax_validity);
    const row3_a_softmax = std.math.exp(@as(f64, -36.0)) / (@as(f64, 1.0) + std.math.exp(@as(f64, -36.0)));
    const row3_b_softmax = @as(f64, 1.0) / (@as(f64, 1.0) + std.math.exp(@as(f64, -36.0)));
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_a_softmax[0], 1e-12);
    try std.testing.expectApproxEqAbs(row3_a_softmax, row_a_softmax[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_b_softmax[1], 1e-12);
    try std.testing.expectApproxEqAbs(row3_b_softmax, row_b_softmax[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, row_a_softmax_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_softmax_validity);

    var row_log_softmax_table = try validity_table.withRowLogSoftmax(&.{ "a", "b" }, &.{ "a_log_softmax", "b_log_softmax" });
    defer row_log_softmax_table.deinit();
    const row_a_log_softmax_column = try row_log_softmax_table.column("a_log_softmax");
    const row_b_log_softmax_column = try row_log_softmax_table.column("b_log_softmax");
    const row_a_log_softmax = try row_a_log_softmax_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_log_softmax);
    const row_b_log_softmax = try row_b_log_softmax_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_log_softmax);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_a_log_softmax[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log(f64, std.math.e, row3_a_softmax), row_a_log_softmax[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_b_log_softmax[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log(f64, std.math.e, row3_b_softmax), row_b_log_softmax[3], 1e-12);

    var row_softmin_table = try validity_table.withRowSoftmin(&.{ "a", "b" }, &.{ "a_softmin", "b_softmin" });
    defer row_softmin_table.deinit();
    const row_a_softmin = try (try row_softmin_table.column("a_softmin")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_softmin);
    const row_b_softmin = try (try row_softmin_table.column("b_softmin")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_softmin);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_a_softmin[0], 1e-12);
    try std.testing.expectApproxEqAbs(row3_b_softmax, row_a_softmin[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_b_softmin[1], 1e-12);
    try std.testing.expectApproxEqAbs(row3_a_softmax, row_b_softmin[3], 1e-12);

    var row_log_softmin_table = try validity_table.withRowLogSoftmin(&.{ "a", "b" }, &.{ "a_log_softmin", "b_log_softmin" });
    defer row_log_softmin_table.deinit();
    const row_a_log_softmin = try (try row_log_softmin_table.column("a_log_softmin")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_log_softmin);
    const row_b_log_softmin = try (try row_log_softmin_table.column("b_log_softmin")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_log_softmin);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_a_log_softmin[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log(f64, std.math.e, row3_b_softmax), row_a_log_softmin[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_b_log_softmin[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log(f64, std.math.e, row3_a_softmax), row_b_log_softmin[3], 1e-12);

    var row_softmax_entropy_table = try validity_table.withRowSoftmaxEntropy(&.{ "a", "b" }, "row_softmax_entropy");
    defer row_softmax_entropy_table.deinit();
    const row_softmax_entropy_column = try row_softmax_entropy_table.column("row_softmax_entropy");
    try std.testing.expect(row_softmax_entropy_column.f64.nullable());
    const row_softmax_entropy = try row_softmax_entropy_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_softmax_entropy);
    const row_softmax_entropy_validity = try row_softmax_entropy_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_softmax_entropy_validity);
    const row3_softmax_entropy = -(row3_a_softmax * std.math.log(f64, std.math.e, row3_a_softmax) + row3_b_softmax * std.math.log(f64, std.math.e, row3_b_softmax));
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_softmax_entropy[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_softmax_entropy[1], 1e-12);
    try std.testing.expectApproxEqAbs(row3_softmax_entropy, row_softmax_entropy[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_softmax_entropy_validity);

    var row_softmax_perplexity_table = try validity_table.withRowSoftmaxPerplexity(&.{ "a", "b" }, "row_softmax_perplexity");
    defer row_softmax_perplexity_table.deinit();
    const row_softmax_perplexity = try (try row_softmax_perplexity_table.column("row_softmax_perplexity")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_softmax_perplexity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_softmax_perplexity[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_softmax_perplexity[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.exp(row3_softmax_entropy), row_softmax_perplexity[3], 1e-12);

    var row_softmax_confidence_table = try validity_table.withRowSoftmaxConfidence(&.{ "a", "b" }, "row_softmax_confidence");
    defer row_softmax_confidence_table.deinit();
    const row_softmax_confidence = try (try row_softmax_confidence_table.column("row_softmax_confidence")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_softmax_confidence);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_softmax_confidence[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_softmax_confidence[1], 1e-12);
    try std.testing.expectApproxEqAbs(row3_b_softmax, row_softmax_confidence[3], 1e-12);

    var row_softmax_margin_table = try validity_table.withRowSoftmaxMargin(&.{ "a", "b" }, "row_softmax_margin");
    defer row_softmax_margin_table.deinit();
    const row_softmax_margin = try (try row_softmax_margin_table.column("row_softmax_margin")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_softmax_margin);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_softmax_margin[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_softmax_margin[1], 1e-12);
    try std.testing.expectApproxEqAbs(row3_b_softmax - row3_a_softmax, row_softmax_margin[3], 1e-12);

    var row_softmax_evenness_table = try validity_table.withRowSoftmaxEvenness(&.{ "a", "b" }, "row_softmax_evenness");
    defer row_softmax_evenness_table.deinit();
    const row_softmax_evenness = try (try row_softmax_evenness_table.column("row_softmax_evenness")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_softmax_evenness);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_softmax_evenness[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_softmax_evenness[1], 1e-12);
    try std.testing.expectApproxEqAbs(row3_softmax_entropy / std.math.ln2, row_softmax_evenness[3], 1e-12);

    var row_softmax_concentration_table = try validity_table.withRowSoftmaxConcentration(&.{ "a", "b" }, "row_softmax_concentration");
    defer row_softmax_concentration_table.deinit();
    const row_softmax_concentration = try (try row_softmax_concentration_table.column("row_softmax_concentration")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_softmax_concentration);
    const row3_concentration = row3_a_softmax * row3_a_softmax + row3_b_softmax * row3_b_softmax;
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_softmax_concentration[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_softmax_concentration[1], 1e-12);
    try std.testing.expectApproxEqAbs(row3_concentration, row_softmax_concentration[3], 1e-12);

    var row_softmax_gini_table = try validity_table.withRowSoftmaxGiniImpurity(&.{ "a", "b" }, "row_softmax_gini");
    defer row_softmax_gini_table.deinit();
    const row_softmax_gini = try (try row_softmax_gini_table.column("row_softmax_gini")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_softmax_gini);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_softmax_gini[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_softmax_gini[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0) - row3_concentration, row_softmax_gini[3], 1e-12);

    var row_softmax_normalized_hhi_table = try validity_table.withRowSoftmaxNormalizedHhi(&.{ "a", "b" }, "row_softmax_normalized_hhi");
    defer row_softmax_normalized_hhi_table.deinit();
    const row_softmax_normalized_hhi = try (try row_softmax_normalized_hhi_table.column("row_softmax_normalized_hhi")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_softmax_normalized_hhi);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_softmax_normalized_hhi[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_softmax_normalized_hhi[1], 1e-12);
    try std.testing.expectApproxEqAbs((row3_concentration - 0.5) / 0.5, row_softmax_normalized_hhi[3], 1e-12);

    var row_softmax_inverse_table = try validity_table.withRowSoftmaxInverseSimpson(&.{ "a", "b" }, "row_softmax_inverse");
    defer row_softmax_inverse_table.deinit();
    const row_softmax_inverse = try (try row_softmax_inverse_table.column("row_softmax_inverse")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_softmax_inverse);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_softmax_inverse[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_softmax_inverse[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0) / row3_concentration, row_softmax_inverse[3], 1e-12);

    var row_softmax_simpson_evenness_table = try validity_table.withRowSoftmaxSimpsonEvenness(&.{ "a", "b" }, "row_softmax_simpson_evenness");
    defer row_softmax_simpson_evenness_table.deinit();
    const row_softmax_simpson_evenness = try (try row_softmax_simpson_evenness_table.column("row_softmax_simpson_evenness")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_softmax_simpson_evenness);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_softmax_simpson_evenness[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_softmax_simpson_evenness[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0) / (row3_concentration * 2.0), row_softmax_simpson_evenness[3], 1e-12);

    var row_logit_margin_table = try validity_table.withRowLogitMargin(&.{ "a", "b" }, "row_logit_margin");
    defer row_logit_margin_table.deinit();
    const row_logit_margin = try (try row_logit_margin_table.column("row_logit_margin")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_logit_margin);
    try std.testing.expect(std.math.isPositiveInf(row_logit_margin[0]));
    try std.testing.expect(std.math.isPositiveInf(row_logit_margin[1]));
    try std.testing.expectApproxEqAbs(@as(f64, 36.0), row_logit_margin[3], 1e-12);
    try std.testing.expectError(error.LengthMismatch, validity_table.withRowSoftmax(&.{"a"}, &.{ "a_softmax", "extra_softmax" }));

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

    var row_magnitude_geo_table = try validity_table.withRowMagnitudeGeometricMean(&.{ "a", "b" }, "row_magnitude_geo");
    defer row_magnitude_geo_table.deinit();
    const row_magnitude_geo_column = try row_magnitude_geo_table.column("row_magnitude_geo");
    try std.testing.expect(row_magnitude_geo_column.f64.nullable());
    const row_magnitude_geo = try row_magnitude_geo_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_geo);
    const row_magnitude_geo_validity = try row_magnitude_geo_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_geo_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_magnitude_geo[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), row_magnitude_geo[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_magnitude_geo[2], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 160.0)), row_magnitude_geo[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_magnitude_geo_validity);

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

    var row_magnitude_ptp_table = try validity_table.withRowMagnitudePtp(&.{ "a", "b" }, "row_magnitude_ptp");
    defer row_magnitude_ptp_table.deinit();
    const row_magnitude_ptp_column = try row_magnitude_ptp_table.column("row_magnitude_ptp");
    try std.testing.expect(row_magnitude_ptp_column.f64.nullable());
    const row_magnitude_ptp = try row_magnitude_ptp_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_ptp);
    const row_magnitude_ptp_validity = try row_magnitude_ptp_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_ptp_validity);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.0, 36.0 }, row_magnitude_ptp);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_magnitude_ptp_validity);

    var row_midrange_table = try validity_table.withRowMidrange(&.{ "a", "b" }, "row_midrange");
    defer row_midrange_table.deinit();
    const row_midrange = try (try row_midrange_table.column("row_midrange")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_midrange);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 22.0 }, row_midrange);

    var row_magnitude_midrange_table = try validity_table.withRowMagnitudeMidrange(&.{ "a", "b" }, "row_magnitude_midrange");
    defer row_magnitude_midrange_table.deinit();
    const row_magnitude_midrange = try (try row_magnitude_midrange_table.column("row_magnitude_midrange")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_midrange);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 22.0 }, row_magnitude_midrange);

    var row_range_coeff_table = try validity_table.withRowRangeCoeff(&.{ "a", "b" }, "row_range_coeff");
    defer row_range_coeff_table.deinit();
    const row_range_coeff_column = try row_range_coeff_table.column("row_range_coeff");
    try std.testing.expect(row_range_coeff_column.f64.nullable());
    const row_range_coeff = try row_range_coeff_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_range_coeff);
    const row_range_coeff_validity = try row_range_coeff_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_range_coeff_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_range_coeff[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_range_coeff[1], 1e-12);
    try std.testing.expectEqual(@as(f64, 0.0), row_range_coeff[2]);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0 / 11.0), row_range_coeff[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_range_coeff_validity);

    var row_magnitude_range_coeff_table = try validity_table.withRowMagnitudeRangeCoeff(&.{ "a", "b" }, "row_magnitude_range_coeff");
    defer row_magnitude_range_coeff_table.deinit();
    const row_magnitude_range_coeff_column = try row_magnitude_range_coeff_table.column("row_magnitude_range_coeff");
    try std.testing.expect(row_magnitude_range_coeff_column.f64.nullable());
    const row_magnitude_range_coeff = try row_magnitude_range_coeff_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_range_coeff);
    const row_magnitude_range_coeff_validity = try row_magnitude_range_coeff_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_range_coeff_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_magnitude_range_coeff[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_magnitude_range_coeff[1], 1e-12);
    try std.testing.expectEqual(@as(f64, 0.0), row_magnitude_range_coeff[2]);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0 / 11.0), row_magnitude_range_coeff[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_magnitude_range_coeff_validity);

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

    var row_hhi_table = try validity_table.withRowHhi(&.{ "a", "b" }, "row_hhi");
    defer row_hhi_table.deinit();
    const row_hhi_column = try row_hhi_table.column("row_hhi");
    try std.testing.expect(row_hhi_column.f64.nullable());
    const row_hhi = try row_hhi_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_hhi);
    const row_hhi_validity = try row_hhi_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_hhi_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_hhi[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_hhi[1], 1e-12);
    try std.testing.expectEqual(@as(f64, 0.0), row_hhi[2]);
    try std.testing.expectApproxEqAbs(@as(f64, 101.0 / 121.0), row_hhi[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_hhi_validity);

    var row_magnitude_normalized_hhi_table = try validity_table.withRowMagnitudeNormalizedHhi(&.{ "a", "b" }, "row_magnitude_normalized_hhi");
    defer row_magnitude_normalized_hhi_table.deinit();
    const row_magnitude_normalized_hhi_column = try row_magnitude_normalized_hhi_table.column("row_magnitude_normalized_hhi");
    try std.testing.expect(row_magnitude_normalized_hhi_column.f64.nullable());
    const row_magnitude_normalized_hhi = try row_magnitude_normalized_hhi_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_normalized_hhi);
    const row_magnitude_normalized_hhi_validity = try row_magnitude_normalized_hhi_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_normalized_hhi_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_magnitude_normalized_hhi[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_magnitude_normalized_hhi[1], 1e-12);
    try std.testing.expectEqual(@as(f64, 0.0), row_magnitude_normalized_hhi[2]);
    try std.testing.expectApproxEqAbs(@as(f64, 81.0 / 121.0), row_magnitude_normalized_hhi[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_magnitude_normalized_hhi_validity);

    var row_magnitude_sparsity_table = try validity_table.withRowMagnitudeSparsity(&.{ "a", "b" }, "row_magnitude_sparsity");
    defer row_magnitude_sparsity_table.deinit();
    const row_magnitude_sparsity_column = try row_magnitude_sparsity_table.column("row_magnitude_sparsity");
    try std.testing.expect(row_magnitude_sparsity_column.f64.nullable());
    const row_magnitude_sparsity = try row_magnitude_sparsity_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_sparsity);
    const row_magnitude_sparsity_validity = try row_magnitude_sparsity_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_sparsity_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_magnitude_sparsity[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_magnitude_sparsity[1], 1e-12);
    try std.testing.expectEqual(@as(f64, 0.0), row_magnitude_sparsity[2]);
    try std.testing.expectApproxEqAbs((std.math.sqrt(@as(f64, 2.0)) - @as(f64, 11.0) / std.math.sqrt(@as(f64, 101.0))) / (std.math.sqrt(@as(f64, 2.0)) - 1.0), row_magnitude_sparsity[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_magnitude_sparsity_validity);

    var row_magnitude_inverse_table = try validity_table.withRowMagnitudeInverseSimpson(&.{ "a", "b" }, "row_magnitude_inverse");
    defer row_magnitude_inverse_table.deinit();
    const row_magnitude_inverse_column = try row_magnitude_inverse_table.column("row_magnitude_inverse");
    try std.testing.expect(row_magnitude_inverse_column.f64.nullable());
    const row_magnitude_inverse = try row_magnitude_inverse_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_inverse);
    const row_magnitude_inverse_validity = try row_magnitude_inverse_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_inverse_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_magnitude_inverse[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_magnitude_inverse[1], 1e-12);
    try std.testing.expectEqual(@as(f64, 0.0), row_magnitude_inverse[2]);
    try std.testing.expectApproxEqAbs(@as(f64, 121.0 / 101.0), row_magnitude_inverse[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_magnitude_inverse_validity);

    var row_magnitude_simpson_evenness_table = try validity_table.withRowMagnitudeSimpsonEvenness(&.{ "a", "b" }, "row_magnitude_simpson_evenness");
    defer row_magnitude_simpson_evenness_table.deinit();
    const row_magnitude_simpson_evenness_column = try row_magnitude_simpson_evenness_table.column("row_magnitude_simpson_evenness");
    try std.testing.expect(row_magnitude_simpson_evenness_column.f64.nullable());
    const row_magnitude_simpson_evenness = try row_magnitude_simpson_evenness_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_simpson_evenness);
    const row_magnitude_simpson_evenness_validity = try row_magnitude_simpson_evenness_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_simpson_evenness_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_magnitude_simpson_evenness[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_magnitude_simpson_evenness[1], 1e-12);
    try std.testing.expectEqual(@as(f64, 0.0), row_magnitude_simpson_evenness[2]);
    try std.testing.expectApproxEqAbs(@as(f64, 121.0 / 202.0), row_magnitude_simpson_evenness[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_magnitude_simpson_evenness_validity);

    var row_magnitude_dominance_table = try validity_table.withRowMagnitudeDominance(&.{ "a", "b" }, "row_magnitude_dominance");
    defer row_magnitude_dominance_table.deinit();
    const row_magnitude_dominance_column = try row_magnitude_dominance_table.column("row_magnitude_dominance");
    try std.testing.expect(row_magnitude_dominance_column.f64.nullable());
    const row_magnitude_dominance = try row_magnitude_dominance_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_dominance);
    const row_magnitude_dominance_validity = try row_magnitude_dominance_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_dominance_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_magnitude_dominance[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_magnitude_dominance[1], 1e-12);
    try std.testing.expectEqual(@as(f64, 0.0), row_magnitude_dominance[2]);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0 / 11.0), row_magnitude_dominance[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_magnitude_dominance_validity);

    var row_magnitude_margin_table = try validity_table.withRowMagnitudeDominanceMargin(&.{ "a", "b" }, "row_magnitude_margin");
    defer row_magnitude_margin_table.deinit();
    const row_magnitude_margin_column = try row_magnitude_margin_table.column("row_magnitude_margin");
    try std.testing.expect(row_magnitude_margin_column.f64.nullable());
    const row_magnitude_margin = try row_magnitude_margin_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_margin);
    const row_magnitude_margin_validity = try row_magnitude_margin_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_margin_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_magnitude_margin[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_magnitude_margin[1], 1e-12);
    try std.testing.expectEqual(@as(f64, 0.0), row_magnitude_margin[2]);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0 / 11.0), row_magnitude_margin[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_magnitude_margin_validity);

    var row_magnitude_entropy_table = try validity_table.withRowMagnitudeEntropy(&.{ "a", "b" }, "row_magnitude_entropy");
    defer row_magnitude_entropy_table.deinit();
    const row_magnitude_entropy_column = try row_magnitude_entropy_table.column("row_magnitude_entropy");
    try std.testing.expect(row_magnitude_entropy_column.f64.nullable());
    const row_magnitude_entropy = try row_magnitude_entropy_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_entropy);
    const row_magnitude_entropy_validity = try row_magnitude_entropy_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_entropy_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_magnitude_entropy[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_magnitude_entropy[1], 1e-12);
    try std.testing.expectEqual(@as(f64, 0.0), row_magnitude_entropy[2]);
    try std.testing.expectApproxEqAbs(-(@as(f64, 1.0 / 11.0) * std.math.log(f64, std.math.e, @as(f64, 1.0 / 11.0)) + @as(f64, 10.0 / 11.0) * std.math.log(f64, std.math.e, @as(f64, 10.0 / 11.0))), row_magnitude_entropy[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_magnitude_entropy_validity);

    var row_magnitude_perplexity_table = try validity_table.withRowMagnitudePerplexity(&.{ "a", "b" }, "row_magnitude_perplexity");
    defer row_magnitude_perplexity_table.deinit();
    const row_magnitude_perplexity_column = try row_magnitude_perplexity_table.column("row_magnitude_perplexity");
    try std.testing.expect(row_magnitude_perplexity_column.f64.nullable());
    const row_magnitude_perplexity = try row_magnitude_perplexity_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_perplexity);
    const row_magnitude_perplexity_validity = try row_magnitude_perplexity_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_perplexity_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_magnitude_perplexity[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_magnitude_perplexity[1], 1e-12);
    try std.testing.expectEqual(@as(f64, 0.0), row_magnitude_perplexity[2]);
    try std.testing.expectApproxEqAbs(std.math.exp(-(@as(f64, 1.0 / 11.0) * std.math.log(f64, std.math.e, @as(f64, 1.0 / 11.0)) + @as(f64, 10.0 / 11.0) * std.math.log(f64, std.math.e, @as(f64, 10.0 / 11.0)))), row_magnitude_perplexity[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_magnitude_perplexity_validity);

    var row_magnitude_evenness_table = try validity_table.withRowMagnitudeEvenness(&.{ "a", "b" }, "row_magnitude_evenness");
    defer row_magnitude_evenness_table.deinit();
    const row_magnitude_evenness_column = try row_magnitude_evenness_table.column("row_magnitude_evenness");
    try std.testing.expect(row_magnitude_evenness_column.f64.nullable());
    const row_magnitude_evenness = try row_magnitude_evenness_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_evenness);
    const row_magnitude_evenness_validity = try row_magnitude_evenness_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_evenness_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_magnitude_evenness[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_magnitude_evenness[1], 1e-12);
    try std.testing.expectEqual(@as(f64, 0.0), row_magnitude_evenness[2]);
    try std.testing.expectApproxEqAbs(-(@as(f64, 1.0 / 11.0) * std.math.log(f64, std.math.e, @as(f64, 1.0 / 11.0)) + @as(f64, 10.0 / 11.0) * std.math.log(f64, std.math.e, @as(f64, 10.0 / 11.0))) / std.math.log(f64, std.math.e, @as(f64, 2.0)), row_magnitude_evenness[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_magnitude_evenness_validity);

    var row_mean_abs_dev_table = try validity_table.withRowMeanAbsDev(&.{ "a", "b" }, "row_mean_abs_dev");
    defer row_mean_abs_dev_table.deinit();
    const row_mean_abs_dev_column = try row_mean_abs_dev_table.column("row_mean_abs_dev");
    try std.testing.expect(row_mean_abs_dev_column.f64.nullable());
    const row_mean_abs_dev = try row_mean_abs_dev_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_mean_abs_dev);
    const row_mean_abs_dev_validity = try row_mean_abs_dev_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_mean_abs_dev_validity);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.0, 18.0 }, row_mean_abs_dev);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_mean_abs_dev_validity);

    var row_gini_mean_diff_table = try validity_table.withRowGiniMeanDiff(&.{ "a", "b" }, "row_gini_mean_diff");
    defer row_gini_mean_diff_table.deinit();
    const row_gini_mean_diff_column = try row_gini_mean_diff_table.column("row_gini_mean_diff");
    try std.testing.expect(row_gini_mean_diff_column.f64.nullable());
    const row_gini_mean_diff = try row_gini_mean_diff_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_gini_mean_diff);
    const row_gini_mean_diff_validity = try row_gini_mean_diff_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_gini_mean_diff_validity);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.0, 36.0 }, row_gini_mean_diff);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_gini_mean_diff_validity);

    var row_gini_coeff_table = try validity_table.withRowGiniCoefficient(&.{ "a", "b" }, "row_gini_coeff");
    defer row_gini_coeff_table.deinit();
    const row_gini_coeff_column = try row_gini_coeff_table.column("row_gini_coeff");
    try std.testing.expect(row_gini_coeff_column.f64.nullable());
    const row_gini_coeff = try row_gini_coeff_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_gini_coeff);
    const row_gini_coeff_validity = try row_gini_coeff_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_gini_coeff_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_gini_coeff[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_gini_coeff[1], 1e-12);
    try std.testing.expectEqual(@as(f64, 0.0), row_gini_coeff[2]);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0 / 11.0), row_gini_coeff[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_gini_coeff_validity);

    var row_mad_ratio_table = try validity_table.withRowMeanAbsDevRatio(&.{ "a", "b" }, "row_mad_ratio");
    defer row_mad_ratio_table.deinit();
    const row_mad_ratio_column = try row_mad_ratio_table.column("row_mad_ratio");
    try std.testing.expect(row_mad_ratio_column.f64.nullable());
    const row_mad_ratio = try row_mad_ratio_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_mad_ratio);
    const row_mad_ratio_validity = try row_mad_ratio_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_mad_ratio_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_mad_ratio[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_mad_ratio[1], 1e-12);
    try std.testing.expectEqual(@as(f64, 0.0), row_mad_ratio[2]);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0 / 11.0), row_mad_ratio[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_mad_ratio_validity);

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

    var row_magnitude_cv_table = try validity_table.withRowMagnitudeCv(&.{ "a", "b" }, "row_magnitude_cv", 0.0);
    defer row_magnitude_cv_table.deinit();
    const row_magnitude_cv_column = try row_magnitude_cv_table.column("row_magnitude_cv");
    try std.testing.expect(row_magnitude_cv_column.f64.nullable());
    const row_magnitude_cv = try row_magnitude_cv_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_cv);
    const row_magnitude_cv_validity = try row_magnitude_cv_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_cv_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_magnitude_cv[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_magnitude_cv[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_magnitude_cv[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 18.0 / 22.0), row_magnitude_cv[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_magnitude_cv_validity);

    var row_fano_table = try validity_table.withRowFano(&.{ "a", "b" }, "row_fano", 0.0);
    defer row_fano_table.deinit();
    const row_fano_column = try row_fano_table.column("row_fano");
    try std.testing.expect(row_fano_column.f64.nullable());
    const row_fano = try row_fano_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_fano);
    const row_fano_validity = try row_fano_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_fano_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_fano[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_fano[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_fano[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 162.0 / 11.0), row_fano[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_fano_validity);

    var row_magnitude_fano_table = try validity_table.withRowMagnitudeFano(&.{ "a", "b" }, "row_magnitude_fano", 0.0);
    defer row_magnitude_fano_table.deinit();
    const row_magnitude_fano_column = try row_magnitude_fano_table.column("row_magnitude_fano");
    try std.testing.expect(row_magnitude_fano_column.f64.nullable());
    const row_magnitude_fano = try row_magnitude_fano_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_fano);
    const row_magnitude_fano_validity = try row_magnitude_fano_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_fano_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_magnitude_fano[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_magnitude_fano[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_magnitude_fano[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 162.0 / 11.0), row_magnitude_fano[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_magnitude_fano_validity);

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

    var left_nullable = try DeviceColumn.fromSliceWithValidity(i64, gpa, &.{ 1, 2, 3, 4 }, &.{ false, true, false, true }, .cpu);
    defer left_nullable.deinit();
    var right_nullable = try DeviceColumn.fromSliceWithValidity(i64, gpa, &.{ 10, 20, 30, 40 }, &.{ false, false, true, true }, .cpu);
    defer right_nullable.deinit();
    var all_null_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "left", .data = left_nullable },
        .{ .name = "right", .data = right_nullable },
    });
    defer all_null_table.deinit();
    var dropped_all_nulls = try all_null_table.dropAllNulls(&.{ "left", "right" });
    defer dropped_all_nulls.deinit();
    const dropped_all_left = try (try dropped_all_nulls.column("left")).i64.toOwnedSlice(gpa);
    defer gpa.free(dropped_all_left);
    try std.testing.expectEqualSlices(i64, &.{ 2, 3, 4 }, dropped_all_left);
    var only_all_nulls = try all_null_table.filterAllNulls(&.{ "left", "right" });
    defer only_all_nulls.deinit();
    const only_all_left_validity = try (try only_all_nulls.column("left")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(only_all_left_validity);
    try std.testing.expectEqualSlices(bool, &.{false}, only_all_left_validity);

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

    var stripped_prefix_names = try prefixed_names.stripColumnNamePrefix("src_");
    defer stripped_prefix_names.deinit();
    try std.testing.expectEqual(@as(?usize, 0), stripped_prefix_names.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), stripped_prefix_names.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, 2), stripped_prefix_names.columnIndex("active"));

    var stripped_suffix_names = try suffixed_names.stripColumnNameSuffix("_raw");
    defer stripped_suffix_names.deinit();
    try std.testing.expectEqual(@as(?usize, 0), stripped_suffix_names.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), stripped_suffix_names.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, 2), stripped_suffix_names.columnIndex("active"));

    var replaced_prefix_names = try prefixed_names.replaceColumnNamePrefix("src_", "raw_");
    defer replaced_prefix_names.deinit();
    try std.testing.expectEqual(@as(?usize, 0), replaced_prefix_names.columnIndex("raw_sales"));
    try std.testing.expectEqual(@as(?usize, 1), replaced_prefix_names.columnIndex("raw_units"));
    try std.testing.expectEqual(@as(?usize, 2), replaced_prefix_names.columnIndex("raw_active"));

    var replaced_suffix_names = try suffixed_names.replaceColumnNameSuffix("_raw", "_clean");
    defer replaced_suffix_names.deinit();
    try std.testing.expectEqual(@as(?usize, 0), replaced_suffix_names.columnIndex("sales_clean"));
    try std.testing.expectEqual(@as(?usize, 1), replaced_suffix_names.columnIndex("units_clean"));
    try std.testing.expectEqual(@as(?usize, 2), replaced_suffix_names.columnIndex("active_clean"));
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

    var limited = try table.limit(2);
    defer limited.deinit();
    try std.testing.expectEqual(@as(usize, 2), limited.height());
    var first_row = try table.firstRow();
    defer first_row.deinit();
    const first_row_sales = try (try first_row.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(first_row_sales);
    try std.testing.expectEqualSlices(f64, &.{2.0}, first_row_sales);
    var last_row = try table.lastRow();
    defer last_row.deinit();
    const last_row_sales = try (try last_row.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(last_row_sales);
    try std.testing.expectEqualSlices(f64, &.{5.0}, last_row_sales);
    var offset_rows = try table.offset(1);
    defer offset_rows.deinit();
    const offset_sales = try (try offset_rows.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(offset_sales);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 5.0 }, offset_sales);
    var slice_len = try table.sliceRowsLen(1, 1);
    defer slice_len.deinit();
    const slice_len_sales = try (try slice_len.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(slice_len_sales);
    try std.testing.expectEqualSlices(f64, &.{3.0}, slice_len_sales);

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

    var shuffled = try table.shuffleRows(1234);
    defer shuffled.deinit();
    try std.testing.expectEqual(table.height(), shuffled.height());
    var shuffled_again = try table.shuffleRows(1234);
    defer shuffled_again.deinit();
    const shuffled_sales = try (try shuffled.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(shuffled_sales);
    const shuffled_again_sales = try (try shuffled_again.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(shuffled_again_sales);
    try std.testing.expectEqualSlices(f64, shuffled_sales, shuffled_again_sales);

    var sampled_fraction = try table.sampleRowsFraction(0.5, 1234);
    defer sampled_fraction.deinit();
    try std.testing.expectEqual(@as(usize, 1), sampled_fraction.height());
    var sampled_frac_alias = try table.sampleFrac(0.5, 1234);
    defer sampled_frac_alias.deinit();
    try std.testing.expectEqual(@as(usize, 1), sampled_frac_alias.height());
    var sampled_fraction_full = try table.sampleRowsFraction(1.0, 1234);
    defer sampled_fraction_full.deinit();
    try std.testing.expectEqual(table.height(), sampled_fraction_full.height());
    try std.testing.expectError(error.InvalidShape, table.sampleRowsFraction(1.1, 1234));

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

    var sampled_fraction_replacement = try table.sampleRowsFractionWithReplacement(1.5, 4321);
    defer sampled_fraction_replacement.deinit();
    try std.testing.expectEqual(@as(usize, 4), sampled_fraction_replacement.height());
    var sampled_frac_replacement_alias = try table.sampleFracWithReplacement(1.5, 4321);
    defer sampled_frac_replacement_alias.deinit();
    try std.testing.expectEqual(@as(usize, 4), sampled_frac_replacement_alias.height());

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

test "device dataframe derives row magnitude coefficient of variation for signed rows" {
    const gpa = std.testing.allocator;
    var a = try DeviceColumn.fromSlice(f64, gpa, &.{ -2.0, -3.0, 0.0 }, .cpu);
    defer a.deinit();
    var b = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 9.0, 0.0 }, .cpu);
    defer b.deinit();
    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "a", .data = a },
        .{ .name = "b", .data = b },
    });
    defer table.deinit();

    var ordinary_geo = try table.withRowGeometricMean(&.{ "a", "b" }, "row_geo");
    defer ordinary_geo.deinit();
    const row_geo = try (try ordinary_geo.column("row_geo")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_geo);
    try std.testing.expect(std.math.isNan(row_geo[0]));
    try std.testing.expect(std.math.isNan(row_geo[1]));
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_geo[2], 1e-12);

    var signed_magnitude_geo = try table.withRowMagnitudeGeometricMean(&.{ "a", "b" }, "row_magnitude_geo");
    defer signed_magnitude_geo.deinit();
    const signed_row_magnitude_geo = try (try signed_magnitude_geo.column("row_magnitude_geo")).f64.toOwnedSlice(gpa);
    defer gpa.free(signed_row_magnitude_geo);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), signed_row_magnitude_geo[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 27.0)), signed_row_magnitude_geo[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), signed_row_magnitude_geo[2], 1e-12);

    var ordinary_variance = try table.withRowVariance(&.{ "a", "b" }, "row_variance", 0.0);
    defer ordinary_variance.deinit();
    const row_variance = try (try ordinary_variance.column("row_variance")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_variance);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), row_variance[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 36.0), row_variance[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_variance[2], 1e-12);

    var magnitude_variance = try table.withRowMagnitudeVariance(&.{ "a", "b" }, "row_magnitude_variance", 0.0);
    defer magnitude_variance.deinit();
    const row_magnitude_variance = try (try magnitude_variance.column("row_magnitude_variance")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_variance);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_magnitude_variance[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0), row_magnitude_variance[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_magnitude_variance[2], 1e-12);

    var magnitude_stddev = try table.withRowMagnitudeStddev(&.{ "a", "b" }, "row_magnitude_stddev", 0.0);
    defer magnitude_stddev.deinit();
    const row_magnitude_stddev = try (try magnitude_stddev.column("row_magnitude_stddev")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_stddev);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_magnitude_stddev[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), row_magnitude_stddev[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_magnitude_stddev[2], 1e-12);

    var magnitude_sem = try table.withRowMagnitudeSem(&.{ "a", "b" }, "row_magnitude_sem", 0.0);
    defer magnitude_sem.deinit();
    const row_magnitude_sem = try (try magnitude_sem.column("row_magnitude_sem")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_sem);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_magnitude_sem[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0 / std.math.sqrt(@as(f64, 2.0))), row_magnitude_sem[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_magnitude_sem[2], 1e-12);

    var magnitude_skew = try table.withRowMagnitudeSkewness(&.{ "a", "b" }, "row_magnitude_skew");
    defer magnitude_skew.deinit();
    const row_magnitude_skew = try (try magnitude_skew.column("row_magnitude_skew")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_skew);
    try std.testing.expect(std.math.isNan(row_magnitude_skew[0]));
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_magnitude_skew[1], 1e-12);
    try std.testing.expect(std.math.isNan(row_magnitude_skew[2]));

    var magnitude_kurt = try table.withRowMagnitudeKurtosis(&.{ "a", "b" }, "row_magnitude_kurt");
    defer magnitude_kurt.deinit();
    const row_magnitude_kurt = try (try magnitude_kurt.column("row_magnitude_kurt")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_kurt);
    try std.testing.expect(std.math.isNan(row_magnitude_kurt[0]));
    try std.testing.expectApproxEqAbs(@as(f64, -2.0), row_magnitude_kurt[1], 1e-12);
    try std.testing.expect(std.math.isNan(row_magnitude_kurt[2]));

    var ordinary = try table.withRowCv(&.{ "a", "b" }, "row_cv", 0.0);
    defer ordinary.deinit();
    const row_cv = try (try ordinary.column("row_cv")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_cv);
    try std.testing.expect(std.math.isNan(row_cv[0]));
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), row_cv[1], 1e-12);
    try std.testing.expect(std.math.isNan(row_cv[2]));

    var magnitude = try table.withRowMagnitudeCv(&.{ "a", "b" }, "row_magnitude_cv", 0.0);
    defer magnitude.deinit();
    const row_magnitude_cv = try (try magnitude.column("row_magnitude_cv")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_cv);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_magnitude_cv[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), row_magnitude_cv[1], 1e-12);
    try std.testing.expect(std.math.isNan(row_magnitude_cv[2]));

    var ordinary_fano = try table.withRowFano(&.{ "a", "b" }, "row_fano", 0.0);
    defer ordinary_fano.deinit();
    const row_fano = try (try ordinary_fano.column("row_fano")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_fano);
    try std.testing.expect(std.math.isNan(row_fano[0]));
    try std.testing.expectApproxEqAbs(@as(f64, 12.0), row_fano[1], 1e-12);
    try std.testing.expect(std.math.isNan(row_fano[2]));

    var magnitude_fano = try table.withRowMagnitudeFano(&.{ "a", "b" }, "row_magnitude_fano", 0.0);
    defer magnitude_fano.deinit();
    const row_magnitude_fano = try (try magnitude_fano.column("row_magnitude_fano")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_fano);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_magnitude_fano[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), row_magnitude_fano[1], 1e-12);
    try std.testing.expect(std.math.isNan(row_magnitude_fano[2]));
}

test "device dataframe derives stable row logsumexp for extreme logits" {
    const gpa = std.testing.allocator;
    var low = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1000.0, -std.math.inf(f64), std.math.nan(f64), 5.0 }, &.{ true, true, true, false }, .cpu);
    defer low.deinit();
    var high = try DeviceColumn.fromSlice(f64, gpa, &.{ 1001.0, -std.math.inf(f64), 1.0, 7.0 }, .cpu);
    defer high.deinit();
    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "low", .data = low },
        .{ .name = "high", .data = high },
    });
    defer table.deinit();

    var lse_table = try table.withRowLogSumExp(&.{ "low", "high" }, "row_logsumexp");
    defer lse_table.deinit();
    const lse_column = try lse_table.column("row_logsumexp");
    try std.testing.expect(lse_column.f64.nullable());
    const lse = try lse_column.f64.toOwnedSlice(gpa);
    defer gpa.free(lse);
    const lse_validity = try lse_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lse_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1001.0) + std.math.log1p(std.math.exp(@as(f64, -1.0))), lse[0], 1e-12);
    try std.testing.expect(std.math.isNegativeInf(lse[1]));
    try std.testing.expect(std.math.isNan(lse[2]));
    try std.testing.expectApproxEqAbs(@as(f64, 7.0), lse[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, lse_validity);

    var lme_table = try table.withRowLogMeanExp(&.{ "low", "high" }, "row_logmeanexp");
    defer lme_table.deinit();
    const lme_column = try lme_table.column("row_logmeanexp");
    try std.testing.expect(lme_column.f64.nullable());
    const lme = try lme_column.f64.toOwnedSlice(gpa);
    defer gpa.free(lme);
    const lme_validity = try lme_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lme_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1001.0) + std.math.log1p(std.math.exp(@as(f64, -1.0))) - std.math.ln2, lme[0], 1e-12);
    try std.testing.expect(std.math.isNegativeInf(lme[1]));
    try std.testing.expect(std.math.isNan(lme[2]));
    try std.testing.expectApproxEqAbs(@as(f64, 7.0), lme[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, lme_validity);

    var softmax_table = try table.withRowSoftmax(&.{ "low", "high" }, &.{ "low_prob", "high_prob" });
    defer softmax_table.deinit();
    const low_prob_column = try softmax_table.column("low_prob");
    const high_prob_column = try softmax_table.column("high_prob");
    try std.testing.expect(low_prob_column.f64.nullable());
    try std.testing.expect(high_prob_column.f64.nullable());
    const low_prob = try low_prob_column.f64.toOwnedSlice(gpa);
    defer gpa.free(low_prob);
    const high_prob = try high_prob_column.f64.toOwnedSlice(gpa);
    defer gpa.free(high_prob);
    const low_prob_validity = try low_prob_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(low_prob_validity);
    const high_prob_validity = try high_prob_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(high_prob_validity);
    const expected_low0 = std.math.exp(@as(f64, -1.0)) / (@as(f64, 1.0) + std.math.exp(@as(f64, -1.0)));
    const expected_high0 = @as(f64, 1.0) / (@as(f64, 1.0) + std.math.exp(@as(f64, -1.0)));
    try std.testing.expectApproxEqAbs(expected_low0, low_prob[0], 1e-12);
    try std.testing.expectApproxEqAbs(expected_high0, high_prob[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), low_prob[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), high_prob[1], 1e-12);
    try std.testing.expect(std.math.isNan(low_prob[2]));
    try std.testing.expect(std.math.isNan(high_prob[2]));
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), high_prob[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false }, low_prob_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, high_prob_validity);

    var log_softmax_table = try table.withRowLogSoftmax(&.{ "low", "high" }, &.{ "low_log_prob", "high_log_prob" });
    defer log_softmax_table.deinit();
    const low_log_prob = try (try log_softmax_table.column("low_log_prob")).f64.toOwnedSlice(gpa);
    defer gpa.free(low_log_prob);
    const high_log_prob = try (try log_softmax_table.column("high_log_prob")).f64.toOwnedSlice(gpa);
    defer gpa.free(high_log_prob);
    try std.testing.expectApproxEqAbs(std.math.log(f64, std.math.e, expected_low0), low_log_prob[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log(f64, std.math.e, expected_high0), high_log_prob[0], 1e-12);
    try std.testing.expectApproxEqAbs(-std.math.ln2, low_log_prob[1], 1e-12);
    try std.testing.expectApproxEqAbs(-std.math.ln2, high_log_prob[1], 1e-12);
    try std.testing.expect(std.math.isNan(low_log_prob[2]));
    try std.testing.expect(std.math.isNan(high_log_prob[2]));
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), high_log_prob[3], 1e-12);

    var softmin_table = try table.withRowSoftmin(&.{ "low", "high" }, &.{ "low_softmin", "high_softmin" });
    defer softmin_table.deinit();
    const low_softmin = try (try softmin_table.column("low_softmin")).f64.toOwnedSlice(gpa);
    defer gpa.free(low_softmin);
    const high_softmin = try (try softmin_table.column("high_softmin")).f64.toOwnedSlice(gpa);
    defer gpa.free(high_softmin);
    try std.testing.expectApproxEqAbs(expected_high0, low_softmin[0], 1e-12);
    try std.testing.expectApproxEqAbs(expected_low0, high_softmin[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), low_softmin[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), high_softmin[1], 1e-12);
    try std.testing.expect(std.math.isNan(low_softmin[2]));
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), high_softmin[3], 1e-12);

    var log_softmin_table = try table.withRowLogSoftmin(&.{ "low", "high" }, &.{ "low_log_softmin", "high_log_softmin" });
    defer log_softmin_table.deinit();
    const low_log_softmin = try (try log_softmin_table.column("low_log_softmin")).f64.toOwnedSlice(gpa);
    defer gpa.free(low_log_softmin);
    const high_log_softmin = try (try log_softmin_table.column("high_log_softmin")).f64.toOwnedSlice(gpa);
    defer gpa.free(high_log_softmin);
    try std.testing.expectApproxEqAbs(std.math.log(f64, std.math.e, expected_high0), low_log_softmin[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log(f64, std.math.e, expected_low0), high_log_softmin[0], 1e-12);
    try std.testing.expectApproxEqAbs(-std.math.ln2, low_log_softmin[1], 1e-12);
    try std.testing.expectApproxEqAbs(-std.math.ln2, high_log_softmin[1], 1e-12);
    try std.testing.expect(std.math.isNan(low_log_softmin[2]));
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), high_log_softmin[3], 1e-12);

    var entropy_table = try table.withRowSoftmaxEntropy(&.{ "low", "high" }, "row_softmax_entropy");
    defer entropy_table.deinit();
    const entropy_column = try entropy_table.column("row_softmax_entropy");
    try std.testing.expect(entropy_column.f64.nullable());
    const entropy = try entropy_column.f64.toOwnedSlice(gpa);
    defer gpa.free(entropy);
    const entropy_validity = try entropy_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(entropy_validity);
    const expected_entropy0 = -(expected_low0 * std.math.log(f64, std.math.e, expected_low0) + expected_high0 * std.math.log(f64, std.math.e, expected_high0));
    try std.testing.expectApproxEqAbs(expected_entropy0, entropy[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.ln2, entropy[1], 1e-12);
    try std.testing.expect(std.math.isNan(entropy[2]));
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), entropy[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, entropy_validity);

    var perplexity_table = try table.withRowSoftmaxPerplexity(&.{ "low", "high" }, "row_softmax_perplexity");
    defer perplexity_table.deinit();
    const perplexity = try (try perplexity_table.column("row_softmax_perplexity")).f64.toOwnedSlice(gpa);
    defer gpa.free(perplexity);
    try std.testing.expectApproxEqAbs(std.math.exp(expected_entropy0), perplexity[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), perplexity[1], 1e-12);
    try std.testing.expect(std.math.isNan(perplexity[2]));
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), perplexity[3], 1e-12);

    var confidence_table = try table.withRowSoftmaxConfidence(&.{ "low", "high" }, "row_softmax_confidence");
    defer confidence_table.deinit();
    const confidence = try (try confidence_table.column("row_softmax_confidence")).f64.toOwnedSlice(gpa);
    defer gpa.free(confidence);
    try std.testing.expectApproxEqAbs(expected_high0, confidence[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), confidence[1], 1e-12);
    try std.testing.expect(std.math.isNan(confidence[2]));
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), confidence[3], 1e-12);

    var margin_table = try table.withRowSoftmaxMargin(&.{ "low", "high" }, "row_softmax_margin");
    defer margin_table.deinit();
    const margin = try (try margin_table.column("row_softmax_margin")).f64.toOwnedSlice(gpa);
    defer gpa.free(margin);
    try std.testing.expectApproxEqAbs(expected_high0 - expected_low0, margin[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), margin[1], 1e-12);
    try std.testing.expect(std.math.isNan(margin[2]));
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), margin[3], 1e-12);

    var evenness_table = try table.withRowSoftmaxNormalizedEntropy(&.{ "low", "high" }, "row_softmax_evenness");
    defer evenness_table.deinit();
    const evenness = try (try evenness_table.column("row_softmax_evenness")).f64.toOwnedSlice(gpa);
    defer gpa.free(evenness);
    try std.testing.expectApproxEqAbs(expected_entropy0 / std.math.ln2, evenness[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), evenness[1], 1e-12);
    try std.testing.expect(std.math.isNan(evenness[2]));
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), evenness[3], 1e-12);

    var concentration_table = try table.withRowSoftmaxConcentration(&.{ "low", "high" }, "row_softmax_concentration");
    defer concentration_table.deinit();
    const concentration = try (try concentration_table.column("row_softmax_concentration")).f64.toOwnedSlice(gpa);
    defer gpa.free(concentration);
    const expected_concentration0 = expected_low0 * expected_low0 + expected_high0 * expected_high0;
    try std.testing.expectApproxEqAbs(expected_concentration0, concentration[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), concentration[1], 1e-12);
    try std.testing.expect(std.math.isNan(concentration[2]));
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), concentration[3], 1e-12);

    var gini_table = try table.withRowSoftmaxGini(&.{ "low", "high" }, "row_softmax_gini");
    defer gini_table.deinit();
    const gini = try (try gini_table.column("row_softmax_gini")).f64.toOwnedSlice(gpa);
    defer gpa.free(gini);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0) - expected_concentration0, gini[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), gini[1], 1e-12);
    try std.testing.expect(std.math.isNan(gini[2]));
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), gini[3], 1e-12);

    var normalized_hhi_table = try table.withRowSoftmaxNhhi(&.{ "low", "high" }, "row_softmax_normalized_hhi");
    defer normalized_hhi_table.deinit();
    const normalized_hhi = try (try normalized_hhi_table.column("row_softmax_normalized_hhi")).f64.toOwnedSlice(gpa);
    defer gpa.free(normalized_hhi);
    try std.testing.expectApproxEqAbs((expected_concentration0 - 0.5) / 0.5, normalized_hhi[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), normalized_hhi[1], 1e-12);
    try std.testing.expect(std.math.isNan(normalized_hhi[2]));
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), normalized_hhi[3], 1e-12);

    var inverse_table = try table.withRowSoftmaxInverseSimpson(&.{ "low", "high" }, "row_softmax_inverse");
    defer inverse_table.deinit();
    const inverse = try (try inverse_table.column("row_softmax_inverse")).f64.toOwnedSlice(gpa);
    defer gpa.free(inverse);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0) / expected_concentration0, inverse[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), inverse[1], 1e-12);
    try std.testing.expect(std.math.isNan(inverse[2]));
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), inverse[3], 1e-12);

    var simpson_evenness_table = try table.withRowSoftmaxSimpsonEven(&.{ "low", "high" }, "row_softmax_simpson_evenness");
    defer simpson_evenness_table.deinit();
    const simpson_evenness = try (try simpson_evenness_table.column("row_softmax_simpson_evenness")).f64.toOwnedSlice(gpa);
    defer gpa.free(simpson_evenness);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0) / (expected_concentration0 * 2.0), simpson_evenness[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), simpson_evenness[1], 1e-12);
    try std.testing.expect(std.math.isNan(simpson_evenness[2]));
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), simpson_evenness[3], 1e-12);

    var logit_margin_table = try table.withRowLogitMargin(&.{ "low", "high" }, "row_logit_margin");
    defer logit_margin_table.deinit();
    const logit_margin = try (try logit_margin_table.column("row_logit_margin")).f64.toOwnedSlice(gpa);
    defer gpa.free(logit_margin);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), logit_margin[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), logit_margin[1], 1e-12);
    try std.testing.expect(std.math.isNan(logit_margin[2]));
    try std.testing.expect(std.math.isPositiveInf(logit_margin[3]));
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

    var row_any_zero = try table.withRowAnyZero(&.{ "metric", "id", "flag" }, "row_any_zero");
    defer row_any_zero.deinit();
    const row_any_zero_values = try (try row_any_zero.column("row_any_zero")).bool.toOwnedSlice(gpa);
    defer gpa.free(row_any_zero_values);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false, false, true }, row_any_zero_values);

    var row_all_non_zero = try table.withRowAllNonZero(&.{ "metric", "id", "flag" }, "row_all_non_zero");
    defer row_all_non_zero.deinit();
    const row_all_non_zero_values = try (try row_all_non_zero.column("row_all_non_zero")).bool.toOwnedSlice(gpa);
    defer gpa.free(row_all_non_zero_values);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, true, true, false }, row_all_non_zero_values);

    var row_cum_any_zero = try table.withRowCumulativeAnyZero(&.{ "metric", "id", "flag" }, &.{ "metric_cum_any_zero", "id_cum_any_zero", "flag_cum_any_zero" });
    defer row_cum_any_zero.deinit();
    const metric_cum_any_zero = try (try row_cum_any_zero.column("metric_cum_any_zero")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_any_zero);
    const metric_cum_any_zero_validity = try (try row_cum_any_zero.column("metric_cum_any_zero")).bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_any_zero_validity);
    const flag_cum_any_zero = try (try row_cum_any_zero.column("flag_cum_any_zero")).bool.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_any_zero);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false, false, false }, metric_cum_any_zero);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, true, false }, metric_cum_any_zero_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false, false, true }, flag_cum_any_zero);

    var row_prefix_all_non_zero = try table.withRowPrefixAllNonZero(&.{ "metric", "id", "flag" }, &.{ "metric_prefix_all_nonzero", "id_prefix_all_nonzero", "flag_prefix_all_nonzero" });
    defer row_prefix_all_non_zero.deinit();
    const flag_prefix_all_nonzero = try (try row_prefix_all_non_zero.column("flag_prefix_all_nonzero")).bool.toOwnedSlice(gpa);
    defer gpa.free(flag_prefix_all_nonzero);
    const flag_prefix_all_nonzero_validity = try (try row_prefix_all_non_zero.column("flag_prefix_all_nonzero")).bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(flag_prefix_all_nonzero_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, true, true, false }, flag_prefix_all_nonzero);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, true, true }, flag_prefix_all_nonzero_validity);

    var row_first_zero_indices = try table.withRowFirstZeroIndex(&.{ "metric", "id", "flag" }, "row_first_zero_index");
    defer row_first_zero_indices.deinit();
    const row_first_zero_column = try row_first_zero_indices.column("row_first_zero_index");
    try std.testing.expect(row_first_zero_column.i64.nullable());
    const row_first_zero_index = try row_first_zero_column.i64.toOwnedSlice(gpa);
    defer gpa.free(row_first_zero_index);
    const row_first_zero_validity = try row_first_zero_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_first_zero_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 0, 0, 1 }, row_first_zero_index);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false, false, true }, row_first_zero_validity);

    var row_last_non_zero_indices = try table.withRowLastNonZeroIndex(&.{ "metric", "id", "flag" }, "row_last_nonzero_index");
    defer row_last_non_zero_indices.deinit();
    const row_last_nonzero_column = try row_last_non_zero_indices.column("row_last_nonzero_index");
    try std.testing.expect(row_last_nonzero_column.i64.nullable());
    const row_last_nonzero_index = try row_last_nonzero_column.i64.toOwnedSlice(gpa);
    defer gpa.free(row_last_nonzero_index);
    const row_last_nonzero_validity = try row_last_nonzero_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_last_nonzero_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 2, 0, 2, 2, 0 }, row_last_nonzero_index);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true, false }, row_last_nonzero_validity);

    var row_cum_first_zero_indices = try table.withRowCumulativeFirstZeroIndex(&.{ "metric", "id", "flag" }, &.{ "metric_cum_first_zero", "id_cum_first_zero", "flag_cum_first_zero" });
    defer row_cum_first_zero_indices.deinit();
    const metric_cum_first_zero_column = try row_cum_first_zero_indices.column("metric_cum_first_zero");
    try std.testing.expect(metric_cum_first_zero_column.i64.nullable());
    const metric_cum_first_zero = try metric_cum_first_zero_column.i64.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_first_zero);
    const metric_cum_first_zero_validity = try metric_cum_first_zero_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_first_zero_validity);
    const flag_cum_first_zero = try (try row_cum_first_zero_indices.column("flag_cum_first_zero")).i64.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_first_zero);
    const flag_cum_first_zero_validity = try (try row_cum_first_zero_indices.column("flag_cum_first_zero")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_first_zero_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0, 0, 0 }, metric_cum_first_zero);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false, false, false }, metric_cum_first_zero_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 0, 0, 1 }, flag_cum_first_zero);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false, false, true }, flag_cum_first_zero_validity);

    var row_prefix_last_nonzero_indices = try table.withRowPrefixLastNonZeroIndex(&.{ "metric", "id", "flag" }, &.{ "metric_prefix_last_nonzero", "id_prefix_last_nonzero", "flag_prefix_last_nonzero" });
    defer row_prefix_last_nonzero_indices.deinit();
    const id_prefix_last_nonzero = try (try row_prefix_last_nonzero_indices.column("id_prefix_last_nonzero")).i64.toOwnedSlice(gpa);
    defer gpa.free(id_prefix_last_nonzero);
    const id_prefix_last_nonzero_validity = try (try row_prefix_last_nonzero_indices.column("id_prefix_last_nonzero")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(id_prefix_last_nonzero_validity);
    const flag_prefix_last_nonzero = try (try row_prefix_last_nonzero_indices.column("flag_prefix_last_nonzero")).i64.toOwnedSlice(gpa);
    defer gpa.free(flag_prefix_last_nonzero);
    const flag_prefix_last_nonzero_validity = try (try row_prefix_last_nonzero_indices.column("flag_prefix_last_nonzero")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(flag_prefix_last_nonzero_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 1, 1, 0 }, id_prefix_last_nonzero);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true, false }, id_prefix_last_nonzero_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 2, 0, 2, 2, 0 }, flag_prefix_last_nonzero);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true, false }, flag_prefix_last_nonzero_validity);

    var row_cum_zero_counts = try table.withRowCumulativeZeroCount(&.{ "metric", "id", "flag" }, &.{ "metric_cum_zero", "id_cum_zero", "flag_cum_zero" });
    defer row_cum_zero_counts.deinit();
    const id_cum_zero = try (try row_cum_zero_counts.column("id_cum_zero")).i64.toOwnedSlice(gpa);
    defer gpa.free(id_cum_zero);
    const flag_cum_zero = try (try row_cum_zero_counts.column("flag_cum_zero")).i64.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_zero);
    try std.testing.expectEqualSlices(i64, &.{ 2, 1, 1, 0, 0, 1 }, id_cum_zero);
    try std.testing.expectEqualSlices(i64, &.{ 3, 1, 2, 0, 0, 2 }, flag_cum_zero);

    var row_cum_non_zero_ratios = try table.withRowPrefixNonZeroRatio(&.{ "metric", "id", "flag" }, &.{ "metric_cum_nonzero", "id_cum_nonzero", "flag_cum_nonzero" });
    defer row_cum_non_zero_ratios.deinit();
    const metric_cum_nonzero = try (try row_cum_non_zero_ratios.column("metric_cum_nonzero")).f64.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_nonzero);
    const flag_cum_nonzero = try (try row_cum_non_zero_ratios.column("flag_cum_nonzero")).f64.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_nonzero);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 1.0, 1.0, 1.0, 0.0 }, metric_cum_nonzero);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 2.0 / 3.0, 1.0 / 3.0, 1.0, 1.0, 0.0 }, flag_cum_nonzero);

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
    try std.testing.expectError(error.ColumnNotFound, table.withRowAnyZero(&.{"missing"}, "bad_any_zero"));
    try std.testing.expectError(error.LengthMismatch, table.withRowPrefixAllNonZero(&.{"metric"}, &.{ "metric_all_nonzero", "extra_all_nonzero" }));
    try std.testing.expectError(error.ColumnNotFound, table.withRowZeroCount(&.{"missing"}, "bad_zero_count"));
    try std.testing.expectError(error.ColumnNotFound, table.withRowFirstZeroIndex(&.{"missing"}, "bad_zero_index"));
    try std.testing.expectError(error.ColumnNotFound, table.withRowCumulativeFirstZeroIndex(&.{"missing"}, &.{"bad_first_zero"}));
    try std.testing.expectError(error.LengthMismatch, table.withRowPrefixLastNonZeroIndex(&.{"metric"}, &.{ "metric_last_nonzero", "extra_last_nonzero" }));
    try std.testing.expectError(error.LengthMismatch, table.withRowCumZeroRatio(&.{"metric"}, &.{ "metric_cum_zero", "extra_cum_zero" }));
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

    var row_any_positive = try table.withRowAnyPositive(&.{ "metric", "id", "unsigned", "flag" }, "row_any_positive");
    defer row_any_positive.deinit();
    const row_any_positive_values = try (try row_any_positive.column("row_any_positive")).bool.toOwnedSlice(gpa);
    defer gpa.free(row_any_positive_values);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true, true, true, true }, row_any_positive_values);

    var row_any_signbit = try table.withRowAnySignBit(&.{ "metric", "id", "unsigned", "flag" }, "row_any_signbit");
    defer row_any_signbit.deinit();
    const row_any_signbit_values = try (try row_any_signbit.column("row_any_signbit")).bool.toOwnedSlice(gpa);
    defer gpa.free(row_any_signbit_values);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true, false, false, true, false }, row_any_signbit_values);

    var row_any_positive_zero = try table.withRowAnyPositiveZero(&.{ "metric", "id", "unsigned", "flag" }, "row_any_positive_zero");
    defer row_any_positive_zero.deinit();
    const row_any_positive_zero_values = try (try row_any_positive_zero.column("row_any_positive_zero")).bool.toOwnedSlice(gpa);
    defer gpa.free(row_any_positive_zero_values);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, false, false, false, false }, row_any_positive_zero_values);

    var row_any_negative_zero = try table.withRowAnyNegativeZero(&.{ "metric", "id", "unsigned", "flag" }, "row_any_negative_zero");
    defer row_any_negative_zero.deinit();
    const row_any_negative_zero_values = try (try row_any_negative_zero.column("row_any_negative_zero")).bool.toOwnedSlice(gpa);
    defer gpa.free(row_any_negative_zero_values);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false, false, false, false, false }, row_any_negative_zero_values);

    var row_first_positive_zero_indices = try table.withRowFirstPositiveZeroIndex(&.{ "metric", "id", "unsigned", "flag" }, "row_first_positive_zero_index");
    defer row_first_positive_zero_indices.deinit();
    const row_first_positive_zero = try (try row_first_positive_zero_indices.column("row_first_positive_zero_index")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_first_positive_zero);
    const row_first_positive_zero_validity = try (try row_first_positive_zero_indices.column("row_first_positive_zero_index")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_first_positive_zero_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0, 0, 0, 0, 0 }, row_first_positive_zero);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, false, false, false, false }, row_first_positive_zero_validity);

    var row_last_signbit_indices = try table.withRowLastSignBitIndex(&.{ "metric", "id", "unsigned", "flag" }, "row_last_signbit_index");
    defer row_last_signbit_indices.deinit();
    const row_last_signbit = try (try row_last_signbit_indices.column("row_last_signbit_index")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_last_signbit);
    const row_last_signbit_validity = try (try row_last_signbit_indices.column("row_last_signbit_index")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_last_signbit_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 0, 1, 0, 0, 1, 0 }, row_last_signbit);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true, false, false, true, false }, row_last_signbit_validity);

    var row_first_positive_indices = try table.withRowFirstPositiveIndex(&.{ "metric", "id", "unsigned", "flag" }, "row_first_positive_index");
    defer row_first_positive_indices.deinit();
    const row_first_positive = try (try row_first_positive_indices.column("row_first_positive_index")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_first_positive);
    const row_first_positive_validity = try (try row_first_positive_indices.column("row_first_positive_index")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_first_positive_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 2, 1, 0, 1, 0, 2, 1 }, row_first_positive);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true, true, true, true }, row_first_positive_validity);

    var row_last_negative_indices = try table.withRowLastNegativeIndex(&.{ "metric", "id", "unsigned", "flag" }, "row_last_negative_index");
    defer row_last_negative_indices.deinit();
    const row_last_negative = try (try row_last_negative_indices.column("row_last_negative_index")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_last_negative);
    const row_last_negative_validity = try (try row_last_negative_indices.column("row_last_negative_index")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_last_negative_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 0, 1, 0, 0, 1, 0 }, row_last_negative);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true, false, false, true, false }, row_last_negative_validity);

    var row_cum_first_positive_indices = try table.withRowCumulativeFirstPositiveIndex(&.{ "metric", "id", "unsigned", "flag" }, &.{ "metric_cum_first_positive", "id_cum_first_positive", "unsigned_cum_first_positive", "flag_cum_first_positive" });
    defer row_cum_first_positive_indices.deinit();
    const metric_cum_first_positive = try (try row_cum_first_positive_indices.column("metric_cum_first_positive")).i64.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_first_positive);
    const metric_cum_first_positive_validity = try (try row_cum_first_positive_indices.column("metric_cum_first_positive")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_first_positive_validity);
    const flag_cum_first_positive = try (try row_cum_first_positive_indices.column("flag_cum_first_positive")).i64.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_first_positive);
    const flag_cum_first_positive_validity = try (try row_cum_first_positive_indices.column("flag_cum_first_positive")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_first_positive_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0, 0, 0, 0, 0 }, metric_cum_first_positive);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, true, false, true, false, false }, metric_cum_first_positive_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 2, 1, 0, 1, 0, 2, 1 }, flag_cum_first_positive);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true, true, true, true }, flag_cum_first_positive_validity);

    var row_prefix_last_signbit_indices = try table.withRowPrefixLastSignBitIndex(&.{ "metric", "id", "unsigned", "flag" }, &.{ "metric_prefix_last_signbit", "id_prefix_last_signbit", "unsigned_prefix_last_signbit", "flag_prefix_last_signbit" });
    defer row_prefix_last_signbit_indices.deinit();
    const id_prefix_last_signbit = try (try row_prefix_last_signbit_indices.column("id_prefix_last_signbit")).i64.toOwnedSlice(gpa);
    defer gpa.free(id_prefix_last_signbit);
    const id_prefix_last_signbit_validity = try (try row_prefix_last_signbit_indices.column("id_prefix_last_signbit")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(id_prefix_last_signbit_validity);
    const flag_prefix_last_signbit = try (try row_prefix_last_signbit_indices.column("flag_prefix_last_signbit")).i64.toOwnedSlice(gpa);
    defer gpa.free(flag_prefix_last_signbit);
    const flag_prefix_last_signbit_validity = try (try row_prefix_last_signbit_indices.column("flag_prefix_last_signbit")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(flag_prefix_last_signbit_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 0, 1, 0, 0, 1, 0 }, id_prefix_last_signbit);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true, false, false, true, false }, id_prefix_last_signbit_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 0, 1, 0, 0, 1, 0 }, flag_prefix_last_signbit);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true, false, false, true, false }, flag_prefix_last_signbit_validity);

    var row_cum_last_negative_indices = try table.withRowCumulativeLastNegativeIndex(&.{ "metric", "id", "unsigned", "flag" }, &.{ "metric_cum_last_negative", "id_cum_last_negative", "unsigned_cum_last_negative", "flag_cum_last_negative" });
    defer row_cum_last_negative_indices.deinit();
    const metric_cum_last_negative = try (try row_cum_last_negative_indices.column("metric_cum_last_negative")).i64.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_last_negative);
    const metric_cum_last_negative_validity = try (try row_cum_last_negative_indices.column("metric_cum_last_negative")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_last_negative_validity);
    const flag_cum_last_negative = try (try row_cum_last_negative_indices.column("flag_cum_last_negative")).i64.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_last_negative);
    const flag_cum_last_negative_validity = try (try row_cum_last_negative_indices.column("flag_cum_last_negative")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_last_negative_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0, 0, 0, 0, 0 }, metric_cum_last_negative);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false, false, false, true, false }, metric_cum_last_negative_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 0, 1, 0, 0, 1, 0 }, flag_cum_last_negative);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true, false, false, true, false }, flag_cum_last_negative_validity);

    var row_cum_positive_counts = try table.withRowPrefixPositiveCount(&.{ "metric", "id", "unsigned", "flag" }, &.{ "metric_cum_positive", "id_cum_positive", "unsigned_cum_positive", "flag_cum_positive" });
    defer row_cum_positive_counts.deinit();
    const metric_cum_positive = try (try row_cum_positive_counts.column("metric_cum_positive")).i64.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_positive);
    const unsigned_cum_positive = try (try row_cum_positive_counts.column("unsigned_cum_positive")).i64.toOwnedSlice(gpa);
    defer gpa.free(unsigned_cum_positive);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 1, 0, 1, 0, 0 }, metric_cum_positive);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 1, 2, 1, 2, 1, 1 }, unsigned_cum_positive);

    var row_cum_negative_ratios = try table.withRowCumulativeNegativeRatio(&.{ "metric", "id", "unsigned", "flag" }, &.{ "metric_cum_negative", "id_cum_negative", "unsigned_cum_negative", "flag_cum_negative" });
    defer row_cum_negative_ratios.deinit();
    const id_cum_negative = try (try row_cum_negative_ratios.column("id_cum_negative")).f64.toOwnedSlice(gpa);
    defer gpa.free(id_cum_negative);
    const flag_cum_negative = try (try row_cum_negative_ratios.column("flag_cum_negative")).f64.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_negative);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0 }, id_cum_negative);
    try std.testing.expectEqualSlices(f64, &.{ 0.5, 0.0, 0.0, 0.25, 0.0, 0.0, 0.5, 0.0 }, flag_cum_negative);

    var row_cum_any_positive = try table.withRowCumulativeAnyPositive(&.{ "metric", "id", "unsigned", "flag" }, &.{ "metric_cum_any_positive", "id_cum_any_positive", "unsigned_cum_any_positive", "flag_cum_any_positive" });
    defer row_cum_any_positive.deinit();
    const metric_cum_any_positive = try (try row_cum_any_positive.column("metric_cum_any_positive")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_any_positive);
    const metric_cum_any_positive_validity = try (try row_cum_any_positive.column("metric_cum_any_positive")).bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_any_positive_validity);
    const flag_cum_any_positive = try (try row_cum_any_positive.column("flag_cum_any_positive")).bool.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_any_positive);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, true, false, true, false, false }, metric_cum_any_positive);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, true, true, true, false }, metric_cum_any_positive_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true, true, true, true }, flag_cum_any_positive);

    var row_prefix_all_signbit = try table.withRowPrefixAllSignBit(&.{ "metric", "id", "unsigned", "flag" }, &.{ "metric_prefix_all_signbit", "id_prefix_all_signbit", "unsigned_prefix_all_signbit", "flag_prefix_all_signbit" });
    defer row_prefix_all_signbit.deinit();
    const metric_prefix_all_signbit = try (try row_prefix_all_signbit.column("metric_prefix_all_signbit")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_prefix_all_signbit);
    const metric_prefix_all_signbit_validity = try (try row_prefix_all_signbit.column("metric_prefix_all_signbit")).bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(metric_prefix_all_signbit_validity);
    const flag_prefix_all_signbit = try (try row_prefix_all_signbit.column("flag_prefix_all_signbit")).bool.toOwnedSlice(gpa);
    defer gpa.free(flag_prefix_all_signbit);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false, false, false, true, false }, metric_prefix_all_signbit);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, true, true, true, false }, metric_prefix_all_signbit_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false, false, false, false, false }, flag_prefix_all_signbit);

    var row_cum_any_positive_zero = try table.withRowCumulativeAnyPositiveZero(&.{ "metric", "id", "unsigned", "flag" }, &.{ "metric_cum_any_poszero", "id_cum_any_poszero", "unsigned_cum_any_poszero", "flag_cum_any_poszero" });
    defer row_cum_any_positive_zero.deinit();
    const flag_cum_any_poszero = try (try row_cum_any_positive_zero.column("flag_cum_any_poszero")).bool.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_any_poszero);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, false, false, false, false }, flag_cum_any_poszero);

    var row_prefix_all_negative = try table.withRowPrefixAllNegative(&.{ "metric", "id", "unsigned", "flag" }, &.{ "metric_prefix_all_negative", "id_prefix_all_negative", "unsigned_prefix_all_negative", "flag_prefix_all_negative" });
    defer row_prefix_all_negative.deinit();
    const flag_prefix_all_negative = try (try row_prefix_all_negative.column("flag_prefix_all_negative")).bool.toOwnedSlice(gpa);
    defer gpa.free(flag_prefix_all_negative);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false, false, false, false, false }, flag_prefix_all_negative);

    var row_cum_first_positive_zero_indices = try table.withRowCumulativeFirstPositiveZeroIndex(&.{ "metric", "id", "unsigned", "flag" }, &.{ "metric_cum_first_poszero", "id_cum_first_poszero", "unsigned_cum_first_poszero", "flag_cum_first_poszero" });
    defer row_cum_first_positive_zero_indices.deinit();
    const metric_cum_first_poszero = try (try row_cum_first_positive_zero_indices.column("metric_cum_first_poszero")).i64.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_first_poszero);
    const metric_cum_first_poszero_validity = try (try row_cum_first_positive_zero_indices.column("metric_cum_first_poszero")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_first_poszero_validity);
    const flag_cum_first_poszero = try (try row_cum_first_positive_zero_indices.column("flag_cum_first_poszero")).i64.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_first_poszero);
    const flag_cum_first_poszero_validity = try (try row_cum_first_positive_zero_indices.column("flag_cum_first_poszero")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_first_poszero_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0, 0, 0, 0, 0 }, metric_cum_first_poszero);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, false, false, false, false }, metric_cum_first_poszero_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0, 0, 0, 0, 0 }, flag_cum_first_poszero);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, false, false, false, false }, flag_cum_first_poszero_validity);

    var row_prefix_last_negative_zero_indices = try table.withRowPrefixLastNegativeZeroIndex(&.{ "metric", "id", "unsigned", "flag" }, &.{ "metric_prefix_last_negzero", "id_prefix_last_negzero", "unsigned_prefix_last_negzero", "flag_prefix_last_negzero" });
    defer row_prefix_last_negative_zero_indices.deinit();
    const metric_prefix_last_negzero = try (try row_prefix_last_negative_zero_indices.column("metric_prefix_last_negzero")).i64.toOwnedSlice(gpa);
    defer gpa.free(metric_prefix_last_negzero);
    const metric_prefix_last_negzero_validity = try (try row_prefix_last_negative_zero_indices.column("metric_prefix_last_negzero")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(metric_prefix_last_negzero_validity);
    const flag_prefix_last_negzero = try (try row_prefix_last_negative_zero_indices.column("flag_prefix_last_negzero")).i64.toOwnedSlice(gpa);
    defer gpa.free(flag_prefix_last_negzero);
    const flag_prefix_last_negzero_validity = try (try row_prefix_last_negative_zero_indices.column("flag_prefix_last_negzero")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(flag_prefix_last_negzero_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0, 0, 0, 0, 0 }, metric_prefix_last_negzero);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false, false, false, false, false }, metric_prefix_last_negzero_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0, 0, 0, 0, 0 }, flag_prefix_last_negzero);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false, false, false, false, false }, flag_prefix_last_negzero_validity);

    var row_cum_positive_zero_counts = try table.withRowCumulativePositiveZeroCount(&.{ "metric", "id", "unsigned", "flag" }, &.{ "metric_cum_poszero", "id_cum_poszero", "unsigned_cum_poszero", "flag_cum_poszero" });
    defer row_cum_positive_zero_counts.deinit();
    const metric_cum_poszero = try (try row_cum_positive_zero_counts.column("metric_cum_poszero")).i64.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_poszero);
    const flag_cum_poszero = try (try row_cum_positive_zero_counts.column("flag_cum_poszero")).i64.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_poszero);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 0, 0, 0, 0, 0 }, metric_cum_poszero);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 0, 0, 0, 0, 0 }, flag_cum_poszero);

    var row_cum_negative_zero_ratios = try table.withRowPrefixNegativeZeroRatio(&.{ "metric", "id", "unsigned", "flag" }, &.{ "metric_cum_negzero", "id_cum_negzero", "unsigned_cum_negzero", "flag_cum_negzero" });
    defer row_cum_negative_zero_ratios.deinit();
    const metric_cum_negzero = try (try row_cum_negative_zero_ratios.column("metric_cum_negzero")).f64.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_negzero);
    const flag_cum_negzero = try (try row_cum_negative_zero_ratios.column("flag_cum_negzero")).f64.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_negzero);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, metric_cum_negzero);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, flag_cum_negzero);

    var row_cum_signbit_ratios = try table.withRowCumulativeSignBitRatio(&.{ "metric", "id", "unsigned", "flag" }, &.{ "metric_cum_signbit", "id_cum_signbit", "unsigned_cum_signbit", "flag_cum_signbit" });
    defer row_cum_signbit_ratios.deinit();
    const metric_cum_signbit = try (try row_cum_signbit_ratios.column("metric_cum_signbit")).f64.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_signbit);
    const flag_cum_signbit = try (try row_cum_signbit_ratios.column("flag_cum_signbit")).f64.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_signbit);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 }, metric_cum_signbit);
    try std.testing.expectEqualSlices(f64, &.{ 0.5, 0.25, 0.0, 0.25, 0.0, 0.0, 0.5, 0.0 }, flag_cum_signbit);

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
    try std.testing.expectError(error.ColumnNotFound, table.withRowCumulativeFirstPositiveZeroIndex(&.{"missing"}, &.{"bad_poszero_index"}));
    try std.testing.expectError(error.LengthMismatch, table.withRowPrefixLastNegativeZeroIndex(&.{"metric"}, &.{ "metric_last_negzero", "extra_last_negzero" }));
    try std.testing.expectError(error.ColumnNotFound, table.withRowAnyPositiveZero(&.{"missing"}, "bad_any_positive_zero"));
    try std.testing.expectError(error.LengthMismatch, table.withRowPrefixAllSignBit(&.{"metric"}, &.{ "metric_all_signbit", "extra_all_signbit" }));
    try std.testing.expectError(error.ColumnNotFound, table.withRowPositiveZeroCount(&.{"missing"}, "bad_positive_zero_count"));
    try std.testing.expectError(error.ColumnNotFound, table.withRowPositiveCount(&.{"missing"}, "bad_positive_count"));
    try std.testing.expectError(error.ColumnNotFound, table.withRowFirstPositiveIndex(&.{"missing"}, "bad_positive_index"));
    try std.testing.expectError(error.ColumnNotFound, table.withRowCumulativeFirstPositiveIndex(&.{"missing"}, &.{"bad_positive_index"}));
    try std.testing.expectError(error.LengthMismatch, table.withRowPrefixLastSignBitIndex(&.{"metric"}, &.{ "metric_last_signbit", "extra_last_signbit" }));
    try std.testing.expectError(error.ColumnNotFound, table.withRowFirstSignBitIndex(&.{"missing"}, "bad_signbit_index"));
    try std.testing.expectError(error.LengthMismatch, table.withRowPrefixNegativeCount(&.{"metric"}, &.{ "metric_cum_negative", "extra_cum_negative" }));
    try std.testing.expectError(error.LengthMismatch, table.withRowPrefixSignBitRatio(&.{"metric"}, &.{ "metric_cum_signbit", "extra_cum_signbit" }));
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

    var row_any_nan = try table.withRowAnyNaN(&.{ "metric", "id" }, "row_any_nan");
    defer row_any_nan.deinit();
    const row_any_nan_values = try (try row_any_nan.column("row_any_nan")).bool.toOwnedSlice(gpa);
    defer gpa.free(row_any_nan_values);
    const row_any_nan_validity = try (try row_any_nan.column("row_any_nan")).bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_any_nan_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false }, row_any_nan_values);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, row_any_nan_validity);

    var row_all_finite = try table.withRowAllFinite(&.{ "metric", "id" }, "row_all_finite");
    defer row_all_finite.deinit();
    const row_all_finite_values = try (try row_all_finite.column("row_all_finite")).bool.toOwnedSlice(gpa);
    defer gpa.free(row_all_finite_values);
    const row_all_finite_validity = try (try row_all_finite.column("row_all_finite")).bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_all_finite_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, row_all_finite_values);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, row_all_finite_validity);

    var row_cum_any_nan = try table.withRowCumulativeAnyNaN(&.{ "id", "metric" }, &.{ "id_cum_any_nan", "metric_cum_any_nan" });
    defer row_cum_any_nan.deinit();
    const metric_cum_any_nan = try (try row_cum_any_nan.column("metric_cum_any_nan")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_any_nan);
    const metric_cum_any_nan_validity = try (try row_cum_any_nan.column("metric_cum_any_nan")).bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_any_nan_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false }, metric_cum_any_nan);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, metric_cum_any_nan_validity);

    var row_prefix_all_finite = try table.withRowPrefixAllFinite(&.{ "metric", "id" }, &.{ "metric_prefix_all_finite", "id_prefix_all_finite" });
    defer row_prefix_all_finite.deinit();
    const id_prefix_all_finite = try (try row_prefix_all_finite.column("id_prefix_all_finite")).bool.toOwnedSlice(gpa);
    defer gpa.free(id_prefix_all_finite);
    const id_prefix_all_finite_validity = try (try row_prefix_all_finite.column("id_prefix_all_finite")).bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(id_prefix_all_finite_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, id_prefix_all_finite);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, id_prefix_all_finite_validity);

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

    var row_first_nan_indices = try table.withRowFirstNaNIndex(&.{ "metric", "id" }, "row_first_nan_index");
    defer row_first_nan_indices.deinit();
    const row_first_nan = try (try row_first_nan_indices.column("row_first_nan_index")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_first_nan);
    const row_first_nan_validity = try (try row_first_nan_indices.column("row_first_nan_index")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_first_nan_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0 }, row_first_nan);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false }, row_first_nan_validity);

    var row_last_inf_indices = try table.withRowLastInfIndex(&.{ "metric", "id" }, "row_last_inf_index");
    defer row_last_inf_indices.deinit();
    const row_last_inf = try (try row_last_inf_indices.column("row_last_inf_index")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_last_inf);
    const row_last_inf_validity = try (try row_last_inf_indices.column("row_last_inf_index")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_last_inf_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0 }, row_last_inf);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false }, row_last_inf_validity);

    var signed_inf_metric = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ std.math.inf(f64), std.math.inf(f64), -std.math.inf(f64), std.math.nan(f64), 5.0 }, &.{ true, true, true, true, false }, .cpu);
    defer signed_inf_metric.deinit();
    var signed_inf_peer = try DeviceColumn.fromSlice(f64, gpa, &.{ -std.math.inf(f64), std.math.inf(f64), -std.math.inf(f64), -std.math.inf(f64), std.math.inf(f64) }, .cpu);
    defer signed_inf_peer.deinit();
    var signed_inf_id = try DeviceColumn.fromSlice(i64, gpa, &.{ 10, 20, 30, 40, 50 }, .cpu);
    defer signed_inf_id.deinit();
    var signed_inf_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "metric", .data = signed_inf_metric },
        .{ .name = "peer", .data = signed_inf_peer },
        .{ .name = "id", .data = signed_inf_id },
    });
    defer signed_inf_table.deinit();

    var row_first_positive_inf_indices = try signed_inf_table.withRowFirstPositiveInfIndex(&.{ "metric", "peer", "id" }, "row_first_positive_inf_index");
    defer row_first_positive_inf_indices.deinit();
    const row_first_positive_inf = try (try row_first_positive_inf_indices.column("row_first_positive_inf_index")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_first_positive_inf);
    const row_first_positive_inf_validity = try (try row_first_positive_inf_indices.column("row_first_positive_inf_index")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_first_positive_inf_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0, 1 }, row_first_positive_inf);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false, true }, row_first_positive_inf_validity);

    var row_last_positive_inf_indices = try signed_inf_table.withRowLastPositiveInfIndex(&.{ "metric", "peer", "id" }, "row_last_positive_inf_index");
    defer row_last_positive_inf_indices.deinit();
    const row_last_positive_inf = try (try row_last_positive_inf_indices.column("row_last_positive_inf_index")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_last_positive_inf);
    const row_last_positive_inf_validity = try (try row_last_positive_inf_indices.column("row_last_positive_inf_index")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_last_positive_inf_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 0, 1 }, row_last_positive_inf);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false, true }, row_last_positive_inf_validity);

    var row_first_negative_inf_indices = try signed_inf_table.withRowFirstNegativeInfIndex(&.{ "metric", "peer", "id" }, "row_first_negative_inf_index");
    defer row_first_negative_inf_indices.deinit();
    const row_first_negative_inf = try (try row_first_negative_inf_indices.column("row_first_negative_inf_index")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_first_negative_inf);
    const row_first_negative_inf_validity = try (try row_first_negative_inf_indices.column("row_first_negative_inf_index")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_first_negative_inf_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 0, 1, 0 }, row_first_negative_inf);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, true, false }, row_first_negative_inf_validity);

    var row_last_negative_inf_indices = try signed_inf_table.withRowLastNegativeInfIndex(&.{ "metric", "peer", "id" }, "row_last_negative_inf_index");
    defer row_last_negative_inf_indices.deinit();
    const row_last_negative_inf = try (try row_last_negative_inf_indices.column("row_last_negative_inf_index")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_last_negative_inf);
    const row_last_negative_inf_validity = try (try row_last_negative_inf_indices.column("row_last_negative_inf_index")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_last_negative_inf_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 1, 1, 0 }, row_last_negative_inf);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, true, false }, row_last_negative_inf_validity);

    var row_any_positive_inf = try signed_inf_table.withRowAnyPositiveInf(&.{ "metric", "peer" }, "row_any_positive_inf");
    defer row_any_positive_inf.deinit();
    const row_any_positive_inf_values = try (try row_any_positive_inf.column("row_any_positive_inf")).bool.toOwnedSlice(gpa);
    defer gpa.free(row_any_positive_inf_values);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false, true }, row_any_positive_inf_values);

    var row_prefix_all_negative_inf = try signed_inf_table.withRowPrefixAllNegativeInf(&.{ "metric", "peer" }, &.{ "metric_prefix_all_negative_inf", "peer_prefix_all_negative_inf" });
    defer row_prefix_all_negative_inf.deinit();
    const peer_prefix_all_negative_inf = try (try row_prefix_all_negative_inf.column("peer_prefix_all_negative_inf")).bool.toOwnedSlice(gpa);
    defer gpa.free(peer_prefix_all_negative_inf);
    const peer_prefix_all_negative_inf_validity = try (try row_prefix_all_negative_inf.column("peer_prefix_all_negative_inf")).bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(peer_prefix_all_negative_inf_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, false }, peer_prefix_all_negative_inf);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, true }, peer_prefix_all_negative_inf_validity);

    var row_cum_first_nan_indices = try table.withRowCumulativeFirstNaNIndex(&.{ "id", "metric" }, &.{ "id_cum_first_nan", "metric_cum_first_nan" });
    defer row_cum_first_nan_indices.deinit();
    const metric_cum_first_nan = try (try row_cum_first_nan_indices.column("metric_cum_first_nan")).i64.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_first_nan);
    const metric_cum_first_nan_validity = try (try row_cum_first_nan_indices.column("metric_cum_first_nan")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_first_nan_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 0 }, metric_cum_first_nan);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false }, metric_cum_first_nan_validity);

    var row_prefix_last_inf_indices = try table.withRowPrefixLastInfIndex(&.{ "id", "metric" }, &.{ "id_prefix_last_inf", "metric_prefix_last_inf" });
    defer row_prefix_last_inf_indices.deinit();
    const metric_prefix_last_inf = try (try row_prefix_last_inf_indices.column("metric_prefix_last_inf")).i64.toOwnedSlice(gpa);
    defer gpa.free(metric_prefix_last_inf);
    const metric_prefix_last_inf_validity = try (try row_prefix_last_inf_indices.column("metric_prefix_last_inf")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(metric_prefix_last_inf_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 0 }, metric_prefix_last_inf);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false }, metric_prefix_last_inf_validity);

    var row_cum_first_finite_indices = try table.withRowCumulativeFirstFiniteIndex(&.{ "metric", "id" }, &.{ "metric_cum_first_finite", "id_cum_first_finite" });
    defer row_cum_first_finite_indices.deinit();
    const id_cum_first_finite = try (try row_cum_first_finite_indices.column("id_cum_first_finite")).i64.toOwnedSlice(gpa);
    defer gpa.free(id_cum_first_finite);
    const id_cum_first_finite_validity = try (try row_cum_first_finite_indices.column("id_cum_first_finite")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(id_cum_first_finite_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 1, 1 }, id_cum_first_finite);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, id_cum_first_finite_validity);

    var row_prefix_last_negative_inf_indices = try signed_inf_table.withRowPrefixLastNegativeInfIndex(&.{ "metric", "peer" }, &.{ "metric_prefix_last_negative_inf", "peer_prefix_last_negative_inf" });
    defer row_prefix_last_negative_inf_indices.deinit();
    const peer_prefix_last_negative_inf = try (try row_prefix_last_negative_inf_indices.column("peer_prefix_last_negative_inf")).i64.toOwnedSlice(gpa);
    defer gpa.free(peer_prefix_last_negative_inf);
    const peer_prefix_last_negative_inf_validity = try (try row_prefix_last_negative_inf_indices.column("peer_prefix_last_negative_inf")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(peer_prefix_last_negative_inf_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 1, 1, 0 }, peer_prefix_last_negative_inf);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, true, false }, peer_prefix_last_negative_inf_validity);

    var row_cum_nan_counts = try table.withRowCumulativeNaNCount(&.{ "metric", "id" }, &.{ "metric_cum_nan", "id_cum_nan" });
    defer row_cum_nan_counts.deinit();
    const id_cum_nan = try (try row_cum_nan_counts.column("id_cum_nan")).i64.toOwnedSlice(gpa);
    defer gpa.free(id_cum_nan);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 0 }, id_cum_nan);

    var row_cum_inf_counts = try table.withRowPrefixInfCount(&.{ "metric", "id" }, &.{ "metric_cum_inf", "id_cum_inf" });
    defer row_cum_inf_counts.deinit();
    const id_cum_inf = try (try row_cum_inf_counts.column("id_cum_inf")).i64.toOwnedSlice(gpa);
    defer gpa.free(id_cum_inf);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 0 }, id_cum_inf);

    var row_cum_finite_ratios = try table.withRowCumulativeFiniteRatio(&.{ "metric", "id" }, &.{ "metric_cum_finite_ratio", "id_cum_finite_ratio" });
    defer row_cum_finite_ratios.deinit();
    const id_cum_finite_ratio = try (try row_cum_finite_ratios.column("id_cum_finite_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(id_cum_finite_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 0.5, 0.5, 0.5 }, id_cum_finite_ratio);

    var row_first_finite_indices = try table.withRowFirstFiniteIndex(&.{ "metric", "id" }, "row_first_finite_index");
    defer row_first_finite_indices.deinit();
    const row_first_finite = try (try row_first_finite_indices.column("row_first_finite_index")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_first_finite);
    const row_first_finite_validity = try (try row_first_finite_indices.column("row_first_finite_index")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_first_finite_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 1, 1 }, row_first_finite);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, row_first_finite_validity);

    var row_last_non_finite_indices = try table.withRowLastNonFiniteIndex(&.{ "metric", "id" }, "row_last_non_finite_index");
    defer row_last_non_finite_indices.deinit();
    const row_last_non_finite = try (try row_last_non_finite_indices.column("row_last_non_finite_index")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_last_non_finite);
    const row_last_non_finite_validity = try (try row_last_non_finite_indices.column("row_last_non_finite_index")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_last_non_finite_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0 }, row_last_non_finite);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, false }, row_last_non_finite_validity);

    var row_cum_non_finite_ratios = try table.withRowPrefixNonFiniteRatio(&.{ "metric", "id" }, &.{ "metric_cum_non_finite_ratio", "id_cum_non_finite_ratio" });
    defer row_cum_non_finite_ratios.deinit();
    const metric_cum_non_finite_ratio = try (try row_cum_non_finite_ratios.column("metric_cum_non_finite_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_non_finite_ratio);
    const id_cum_non_finite_ratio = try (try row_cum_non_finite_ratios.column("id_cum_non_finite_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(id_cum_non_finite_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 1.0, 1.0, 0.0 }, metric_cum_non_finite_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.5, 0.5, 0.0 }, id_cum_non_finite_ratio);

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
    try std.testing.expectError(error.ColumnNotFound, table.withRowAnyNaN(&.{"missing"}, "bad_any_nan"));
    try std.testing.expectError(error.LengthMismatch, table.withRowPrefixAllFinite(&.{"metric"}, &.{ "metric_all_finite", "extra_all_finite" }));
    try std.testing.expectError(error.ColumnNotFound, table.withRowNaNCount(&.{"missing"}, "bad_count"));
    try std.testing.expectError(error.ColumnNotFound, table.withRowNaNRatio(&.{"missing"}, "bad_ratio"));
    try std.testing.expectError(error.ColumnNotFound, table.withRowFirstNaNIndex(&.{"missing"}, "bad_nan_index"));
    try std.testing.expectError(error.ColumnNotFound, signed_inf_table.withRowFirstPositiveInfIndex(&.{"missing"}, "bad_positive_inf_index"));
    try std.testing.expectError(error.ColumnNotFound, table.withRowFirstFiniteIndex(&.{"missing"}, "bad_finite_index"));
    try std.testing.expectError(error.LengthMismatch, table.withRowPrefixNonFiniteRatio(&.{"metric"}, &.{ "metric_cum_non_finite", "extra_cum_non_finite" }));
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

    var index_metric = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, 0.0, std.math.floatTrueMin(f64), std.math.inf(f64), -2.0 }, &.{ true, true, true, true, false }, .cpu);
    defer index_metric.deinit();
    var index_peer = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 3.0, std.math.floatTrueMin(f64), 4.0, std.math.floatTrueMin(f64) }, .cpu);
    defer index_peer.deinit();
    var index_id = try DeviceColumn.fromSlice(i64, gpa, &.{ 10, 20, 30, 40, 50 }, .cpu);
    defer index_id.deinit();
    var index_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "metric", .data = index_metric },
        .{ .name = "peer", .data = index_peer },
        .{ .name = "id", .data = index_id },
    });
    defer index_table.deinit();

    var row_first_normal_indices = try index_table.withRowFirstNormalIndex(&.{ "metric", "peer", "id" }, "row_first_normal_index");
    defer row_first_normal_indices.deinit();
    const row_first_normal = try (try row_first_normal_indices.column("row_first_normal_index")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_first_normal);
    const row_first_normal_validity = try (try row_first_normal_indices.column("row_first_normal_index")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_first_normal_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 1, 0 }, row_first_normal);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true, false }, row_first_normal_validity);

    var row_last_normal_indices = try index_table.withRowLastNormalIndex(&.{ "metric", "peer", "id" }, "row_last_normal_index");
    defer row_last_normal_indices.deinit();
    const row_last_normal = try (try row_last_normal_indices.column("row_last_normal_index")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_last_normal);
    const row_last_normal_validity = try (try row_last_normal_indices.column("row_last_normal_index")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_last_normal_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 1, 0, 1, 0 }, row_last_normal);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true, false }, row_last_normal_validity);

    var row_first_subnormal_indices = try index_table.withRowFirstSubnormalIndex(&.{ "metric", "peer", "id" }, "row_first_subnormal_index");
    defer row_first_subnormal_indices.deinit();
    const row_first_subnormal = try (try row_first_subnormal_indices.column("row_first_subnormal_index")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_first_subnormal);
    const row_first_subnormal_validity = try (try row_first_subnormal_indices.column("row_first_subnormal_index")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_first_subnormal_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0, 1 }, row_first_subnormal);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, true }, row_first_subnormal_validity);

    var row_last_subnormal_indices = try index_table.withRowLastSubnormalIndex(&.{ "metric", "peer", "id" }, "row_last_subnormal_index");
    defer row_last_subnormal_indices.deinit();
    const row_last_subnormal = try (try row_last_subnormal_indices.column("row_last_subnormal_index")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_last_subnormal);
    const row_last_subnormal_validity = try (try row_last_subnormal_indices.column("row_last_subnormal_index")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_last_subnormal_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 0, 1 }, row_last_subnormal);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, true }, row_last_subnormal_validity);

    var row_any_normal = try table.withRowAnyNormal(&.{ "metric", "id" }, "row_any_normal");
    defer row_any_normal.deinit();
    const row_any_normal_values = try (try row_any_normal.column("row_any_normal")).bool.toOwnedSlice(gpa);
    defer gpa.free(row_any_normal_values);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false, false }, row_any_normal_values);

    var row_prefix_any_subnormal = try index_table.withRowPrefixAnySubnormal(&.{ "metric", "peer" }, &.{ "metric_prefix_any_subnormal", "peer_prefix_any_subnormal" });
    defer row_prefix_any_subnormal.deinit();
    const peer_prefix_any_subnormal = try (try row_prefix_any_subnormal.column("peer_prefix_any_subnormal")).bool.toOwnedSlice(gpa);
    defer gpa.free(peer_prefix_any_subnormal);
    const peer_prefix_any_subnormal_validity = try (try row_prefix_any_subnormal.column("peer_prefix_any_subnormal")).bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(peer_prefix_any_subnormal_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, true }, peer_prefix_any_subnormal);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, true }, peer_prefix_any_subnormal_validity);

    var row_cum_first_normal_indices = try table.withRowCumulativeFirstNormalIndex(&.{ "metric", "id" }, &.{ "metric_cum_first_normal", "id_cum_first_normal" });
    defer row_cum_first_normal_indices.deinit();
    const id_cum_first_normal = try (try row_cum_first_normal_indices.column("id_cum_first_normal")).i64.toOwnedSlice(gpa);
    defer gpa.free(id_cum_first_normal);
    const id_cum_first_normal_validity = try (try row_cum_first_normal_indices.column("id_cum_first_normal")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(id_cum_first_normal_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0, 0 }, id_cum_first_normal);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false, false }, id_cum_first_normal_validity);

    var row_prefix_last_subnormal_indices = try index_table.withRowPrefixLastSubnormalIndex(&.{ "metric", "peer" }, &.{ "metric_prefix_last_subnormal", "peer_prefix_last_subnormal" });
    defer row_prefix_last_subnormal_indices.deinit();
    const peer_prefix_last_subnormal = try (try row_prefix_last_subnormal_indices.column("peer_prefix_last_subnormal")).i64.toOwnedSlice(gpa);
    defer gpa.free(peer_prefix_last_subnormal);
    const peer_prefix_last_subnormal_validity = try (try row_prefix_last_subnormal_indices.column("peer_prefix_last_subnormal")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(peer_prefix_last_subnormal_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 0, 1 }, peer_prefix_last_subnormal);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, true }, peer_prefix_last_subnormal_validity);

    var row_cum_normal_counts = try table.withRowCumulativeNormalCount(&.{ "metric", "id" }, &.{ "metric_cum_normal", "id_cum_normal" });
    defer row_cum_normal_counts.deinit();
    const id_cum_normal = try (try row_cum_normal_counts.column("id_cum_normal")).i64.toOwnedSlice(gpa);
    defer gpa.free(id_cum_normal);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 0, 0, 0 }, id_cum_normal);

    var row_cum_subnormal_ratios = try table.withRowPrefixSubnormalRatio(&.{ "metric", "id" }, &.{ "metric_cum_subnormal", "id_cum_subnormal" });
    defer row_cum_subnormal_ratios.deinit();
    const metric_cum_subnormal = try (try row_cum_subnormal_ratios.column("metric_cum_subnormal")).f64.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_subnormal);
    const id_cum_subnormal = try (try row_cum_subnormal_ratios.column("id_cum_subnormal")).f64.toOwnedSlice(gpa);
    defer gpa.free(id_cum_subnormal);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 1.0, 0.0, 0.0 }, metric_cum_subnormal);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.5, 0.0, 0.0 }, id_cum_subnormal);

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
    try std.testing.expectError(error.ColumnNotFound, table.withRowAnyNormal(&.{"missing"}, "bad_any_normal"));
    try std.testing.expectError(error.LengthMismatch, table.withRowPrefixAnySubnormal(&.{"metric"}, &.{ "metric_any_subnormal", "extra_any_subnormal" }));
    try std.testing.expectError(error.ColumnNotFound, table.withRowNormalCount(&.{"missing"}, "bad_count"));
    try std.testing.expectError(error.ColumnNotFound, table.withRowSubnormalCount(&.{"missing"}, "bad_subnormal_count"));
    try std.testing.expectError(error.ColumnNotFound, index_table.withRowFirstNormalIndex(&.{"missing"}, "bad_normal_index"));
    try std.testing.expectError(error.ColumnNotFound, index_table.withRowFirstSubnormalIndex(&.{"missing"}, "bad_subnormal_index"));
    try std.testing.expectError(error.LengthMismatch, table.withRowPrefixSubnormalRatio(&.{"metric"}, &.{ "metric_cum_subnormal", "extra_cum_subnormal" }));
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

    var row_cum_positive_inf_counts = try table.withRowPrefixPositiveInfCount(&.{"metric"}, &.{"metric_cum_pos_inf"});
    defer row_cum_positive_inf_counts.deinit();
    const metric_cum_pos_inf = try (try row_cum_positive_inf_counts.column("metric_cum_pos_inf")).i64.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_pos_inf);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 0, 0 }, metric_cum_pos_inf);

    var row_cum_negative_inf_ratios = try table.withRowCumulativeNegativeInfRatio(&.{"metric"}, &.{"metric_cum_neg_inf"});
    defer row_cum_negative_inf_ratios.deinit();
    const metric_cum_neg_inf = try (try row_cum_negative_inf_ratios.column("metric_cum_neg_inf")).f64.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_neg_inf);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 1.0, 0.0, 0.0 }, metric_cum_neg_inf);

    try std.testing.expectError(error.ColumnNotFound, table.dropPositiveInfsColumn("missing"));
    try std.testing.expectError(error.ColumnNotFound, table.filterNegativeInfsColumn("missing"));
    try std.testing.expectError(error.ColumnNotFound, table.withRowPositiveInfCount(&.{"missing"}, "bad_count"));
    try std.testing.expectError(error.LengthMismatch, table.withRowCumPositiveInfRatio(&.{"metric"}, &.{ "metric_cum_pos_inf", "extra_cum_pos_inf" }));
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

    var globbed = try table.selectByNameGlob("*_q?");
    defer globbed.deinit();
    try std.testing.expectEqual(@as(usize, 3), globbed.width());
    try std.testing.expectEqual(@as(?usize, 0), globbed.columnIndex("sales_q1"));
    try std.testing.expectEqual(@as(?usize, 1), globbed.columnIndex("sales_q2"));
    try std.testing.expectEqual(@as(?usize, 2), globbed.columnIndex("cost_q2"));
    try std.testing.expectEqual(@as(?usize, null), globbed.columnIndex("active_flag"));

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

    var drop_globbed = try table.dropByNameGlob("*_q?");
    defer drop_globbed.deinit();
    try std.testing.expectEqual(@as(usize, 2), drop_globbed.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_globbed.columnIndex("active_flag"));
    try std.testing.expectEqual(@as(?usize, 1), drop_globbed.columnIndex("region_code"));
    try std.testing.expectEqual(@as(?usize, null), drop_globbed.columnIndex("sales_q1"));

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
    const table_schema = try table.schema(gpa);
    defer gpa.free(table_schema);
    try std.testing.expect(vectra.ArrowExport.DataFrame.Arrow.hasProjection(table, &.{ "sales", "active" }));
    try std.testing.expect(!vectra.ArrowExport.DataFrame.Arrow.hasProjection(table, &.{"missing"}));
    const grouped_arrow_fields = try vectra.ArrowExport.DataFrame.Arrow.toFields(table, gpa);
    defer {
        for (grouped_arrow_fields) |*field| field.deinit(gpa);
        gpa.free(grouped_arrow_fields);
    }
    try std.testing.expectEqual(@as(usize, 3), grouped_arrow_fields.len);
    const arrow_fields = try table.toArrowFields(gpa);
    defer {
        for (arrow_fields) |*field| field.deinit(gpa);
        gpa.free(arrow_fields);
    }
    var sales_field = try sales.toArrowField(gpa, "sales");
    defer sales_field.deinit(gpa);
    try std.testing.expect(std.mem.eql(u8, "sales", sales_field.name));
    try std.testing.expect(sales_field.data_type.eql(arrow_fields[0].data_type));
    var sales_field_alias = try vectra.ArrowExport.Column.toArrowField(sales, gpa, "sales");
    defer sales_field_alias.deinit(gpa);
    try std.testing.expect(sales_field_alias.data_type.eql(sales_field.data_type));
    const sales_arrow_dtype = try vectra.ArrowExport.Column.arrowDataType(sales);
    try std.testing.expect(sales_arrow_dtype.eql(arrow_fields[0].data_type));
    var sales_arrow_array = try vectra.ArrowExport.Column.toArrowArray(sales, gpa);
    defer sales_arrow_array.deinit(gpa);
    try std.testing.expectEqual(@as(?f64, 2.0), sales_arrow_array.float64.value(0));
    try std.testing.expectEqual(@as(usize, 3), arrow_fields.len);
    try std.testing.expect(std.mem.eql(u8, table_schema[0].name, arrow_fields[0].name));
    try std.testing.expectEqual(table_schema[1].nullableColumn(), arrow_fields[1].nullable);
    const units_schema_dtype = try vectra.ArrowExport.ColumnSchema.arrowDataType(table_schema[1]);
    try std.testing.expect(units_schema_dtype.eql(arrow_fields[1].data_type));
    var units_schema_field = try vectra.ArrowExport.ColumnSchema.toArrowField(table_schema[1], gpa);
    defer units_schema_field.deinit(gpa);
    try std.testing.expect(std.mem.eql(u8, "units", units_schema_field.name));
    try std.testing.expectEqual(table_schema[1].nullableColumn(), units_schema_field.nullable);
    try std.testing.expect(units_schema_field.data_type.eql(arrow_fields[1].data_type));
    var table_view = try table.view();
    defer table_view.deinit();
    const sales_view_dtype = try vectra.ArrowExport.ColumnView.arrowDataType(try table_view.column("sales"));
    try std.testing.expect(sales_view_dtype.eql(arrow_fields[0].data_type));
    var sales_view_field = try vectra.ArrowExport.ColumnView.toArrowField(try table_view.column("sales"), gpa, "sales");
    defer sales_view_field.deinit(gpa);
    try std.testing.expect(std.mem.eql(u8, "sales", sales_view_field.name));
    try std.testing.expect(sales_view_field.data_type.eql(arrow_fields[0].data_type));
    const view_arrow_fields = try vectra.ArrowExport.DataFrameView.toArrowFields(table_view, gpa);
    defer {
        for (view_arrow_fields) |*field| field.deinit(gpa);
        gpa.free(view_arrow_fields);
    }
    try std.testing.expectEqual(@as(usize, 3), view_arrow_fields.len);
    try std.testing.expect(std.mem.eql(u8, table_schema[0].name, view_arrow_fields[0].name));
    try std.testing.expect(vectra.ArrowExport.DataFrameView.hasArrowProjection(table_view, &.{ "active", "sales" }));
    try std.testing.expect(!vectra.ArrowExport.DataFrameView.hasArrowProjection(table_view, &.{"missing"}));
    const projected_view_arrow_fields = try vectra.ArrowExport.DataFrameView.toArrowFieldsProjection(table_view, gpa, &.{ "active", "sales" });
    defer {
        for (projected_view_arrow_fields) |*field| field.deinit(gpa);
        gpa.free(projected_view_arrow_fields);
    }
    try std.testing.expectEqual(@as(usize, 2), projected_view_arrow_fields.len);
    try std.testing.expect(std.mem.eql(u8, "active", projected_view_arrow_fields[0].name));
    try std.testing.expect(std.mem.eql(u8, "sales", projected_view_arrow_fields[1].name));
    var projected_view_arrow_schema = try vectra.ArrowExport.DataFrameView.toArrowSchemaProjection(table_view, gpa, &.{ "active", "sales" });
    defer projected_view_arrow_schema.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 2), projected_view_arrow_schema.fieldCount());
    try std.testing.expect(std.mem.eql(u8, "active", projected_view_arrow_schema.fields[0].name));
    try std.testing.expect(std.mem.eql(u8, "sales", projected_view_arrow_schema.fields[1].name));
    try std.testing.expectError(error.ColumnNotFound, vectra.ArrowExport.DataFrameView.toArrowFieldsProjection(table_view, gpa, &.{"missing"}));
    try std.testing.expectError(error.ColumnNotFound, vectra.ArrowExport.DataFrameView.toArrowSchemaProjection(table_view, gpa, &.{"missing"}));
    var view_arrow_schema = try vectra.ArrowExport.DataFrameView.toArrowSchema(table_view, gpa);
    defer view_arrow_schema.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 3), view_arrow_schema.fieldCount());
    try std.testing.expectEqual(table_schema[1].nullableColumn(), view_arrow_schema.fields[1].nullable);
    var lazy_table = try DeviceLazyFrame.init(gpa, table);
    defer lazy_table.deinit();
    var lazy_table_arrow_schema = try lazy_table.toArrowSchema(gpa);
    defer lazy_table_arrow_schema.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 3), lazy_table_arrow_schema.fieldCount());
    const lazy_table_arrow_fields = try lazy_table.toArrowFields(gpa);
    defer {
        for (lazy_table_arrow_fields) |*field| field.deinit(gpa);
        gpa.free(lazy_table_arrow_fields);
    }
    try std.testing.expectEqualStrings("sales", lazy_table_arrow_fields[0].name);
    try std.testing.expect(lazy_table.hasArrowProjection(&.{ "active", "sales" }));
    try std.testing.expect(!lazy_table.hasArrowProjection(&.{"missing"}));
    try std.testing.expect(lazy_table.hasSchemaProjection(&.{ "active", "sales" }));
    try std.testing.expect(!lazy_table.hasSchemaProjection(&.{"missing"}));
    const lazy_table_projected_schemas = try lazy_table.columnSchemasProjection(gpa, &.{ "active", "sales" });
    defer gpa.free(lazy_table_projected_schemas);
    try std.testing.expectEqual(@as(usize, 2), lazy_table_projected_schemas.len);
    try std.testing.expectEqualStrings("active", lazy_table_projected_schemas[0].name);
    try std.testing.expectEqual(vectra.DeviceDType.bool, lazy_table_projected_schemas[0].dtype);
    try std.testing.expectEqualStrings("sales", lazy_table_projected_schemas[1].name);
    const lazy_table_schema_projection = try lazy_table.schemaProjection(gpa, &.{"units"});
    defer gpa.free(lazy_table_schema_projection);
    try std.testing.expectEqual(@as(usize, 1), lazy_table_schema_projection.len);
    try std.testing.expectEqualStrings("units", lazy_table_schema_projection[0].name);
    const lazy_table_schema_summary_projection = try lazy_table.schemaSummaryProjection(gpa, &.{"sales"});
    defer gpa.free(lazy_table_schema_summary_projection);
    try std.testing.expectEqual(@as(usize, 1), lazy_table_schema_summary_projection.len);
    try std.testing.expectEqualStrings("sales", lazy_table_schema_summary_projection[0].name);
    const lazy_table_projected_dtypes = try lazy_table.columnDTypesProjection(gpa, &.{ "active", "sales" });
    defer gpa.free(lazy_table_projected_dtypes);
    try std.testing.expectEqualSlices(vectra.DeviceDType, &.{ .bool, .f64 }, lazy_table_projected_dtypes);
    const lazy_table_projected_dtype_names = try lazy_table.dtypeNamesProjection(gpa, &.{ "active", "sales" });
    defer gpa.free(lazy_table_projected_dtype_names);
    try std.testing.expectEqualStrings("bool", lazy_table_projected_dtype_names[0]);
    try std.testing.expectEqualStrings("f64", lazy_table_projected_dtype_names[1]);
    const lazy_table_projected_dtype_bytes = try lazy_table.columnDTypeByteSizesProjection(gpa, &.{ "active", "sales" });
    defer gpa.free(lazy_table_projected_dtype_bytes);
    try std.testing.expectEqualSlices(usize, &.{ 1, 8 }, lazy_table_projected_dtype_bytes);
    const lazy_table_projected_dtype_bits = try lazy_table.columnDTypeBitSizesProjection(gpa, &.{ "active", "sales" });
    defer gpa.free(lazy_table_projected_dtype_bits);
    try std.testing.expectEqualSlices(usize, &.{ 8, 64 }, lazy_table_projected_dtype_bits);
    const lazy_table_projected_numeric = try lazy_table.columnDTypeClassMaskProjection(gpa, &.{ "active", "sales" }, .numeric);
    defer gpa.free(lazy_table_projected_numeric);
    try std.testing.expectEqualSlices(bool, &.{ false, true }, lazy_table_projected_numeric);
    try std.testing.expectEqual(@as(usize, 1), try lazy_table.numericColumnCountProjection(&.{ "active", "sales" }));
    try std.testing.expectEqual(@as(usize, 1), try lazy_table.floatColumnCountProjection(&.{ "active", "sales" }));
    try std.testing.expectEqual(@as(usize, 0), try lazy_table.integerColumnCountProjection(&.{ "active", "sales" }));
    try std.testing.expectEqual(@as(usize, 1), try lazy_table.boolColumnCountProjection(&.{ "active", "sales" }));
    const lazy_table_projected_nullable = try lazy_table.columnNullableMaskProjection(gpa, &.{ "units", "active" });
    defer gpa.free(lazy_table_projected_nullable);
    try std.testing.expectEqualSlices(bool, &.{ true, false }, lazy_table_projected_nullable);
    try std.testing.expectEqual(@as(usize, 1), try lazy_table.nullableColumnCountProjection(&.{ "units", "active" }));
    try std.testing.expectEqual(@as(usize, 1), try lazy_table.nonNullableColumnCountProjection(&.{ "units", "active" }));
    try std.testing.expectError(error.ColumnNotFound, lazy_table.columnSchemasProjection(gpa, &.{"missing"}));
    try std.testing.expectError(error.ColumnNotFound, lazy_table.columnDTypesProjection(gpa, &.{"missing"}));
    try std.testing.expectError(error.ColumnNotFound, lazy_table.columnDTypeByteSizesProjection(gpa, &.{"missing"}));
    try std.testing.expectError(error.ColumnNotFound, lazy_table.columnDTypeClassMaskProjection(gpa, &.{"missing"}, .numeric));
    try std.testing.expectError(error.ColumnNotFound, lazy_table.columnNullableMaskProjection(gpa, &.{"missing"}));
    const lazy_table_projected_fields = try lazy_table.toArrowFieldsProjection(gpa, &.{ "active", "sales" });
    defer {
        for (lazy_table_projected_fields) |*field| field.deinit(gpa);
        gpa.free(lazy_table_projected_fields);
    }
    try std.testing.expectEqual(@as(usize, 2), lazy_table_projected_fields.len);
    try std.testing.expectEqualStrings("active", lazy_table_projected_fields[0].name);
    try std.testing.expectEqualStrings("sales", lazy_table_projected_fields[1].name);
    const grouped_lazy_table_projected_fields = try vectra.ArrowExport.LazyFrame.Arrow.toFieldsProjection(&lazy_table, gpa, &.{"units"});
    defer {
        for (grouped_lazy_table_projected_fields) |*field| field.deinit(gpa);
        gpa.free(grouped_lazy_table_projected_fields);
    }
    try std.testing.expectEqual(@as(usize, 1), grouped_lazy_table_projected_fields.len);
    try std.testing.expectEqualStrings("units", grouped_lazy_table_projected_fields[0].name);
    var lazy_table_projected_schema = try lazy_table.toArrowSchemaProjection(gpa, &.{ "active", "sales" });
    defer lazy_table_projected_schema.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 2), lazy_table_projected_schema.fieldCount());
    try std.testing.expectEqualStrings("active", lazy_table_projected_schema.fields[0].name);
    try std.testing.expectEqualStrings("sales", lazy_table_projected_schema.fields[1].name);
    try std.testing.expect(vectra.ArrowExport.LazyFrame.hasArrowProjection(&lazy_table, &.{"sales"}));
    var grouped_lazy_table_schema = try vectra.DeviceLazyFrameArrow.toArrowSchemaProjection(&lazy_table, gpa, &.{"units"});
    defer grouped_lazy_table_schema.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 1), grouped_lazy_table_schema.fieldCount());
    try std.testing.expectEqualStrings("units", grouped_lazy_table_schema.fields[0].name);
    try std.testing.expectError(error.ColumnNotFound, lazy_table.toArrowFieldsProjection(gpa, &.{"missing"}));
    try std.testing.expectError(error.ColumnNotFound, lazy_table.toArrowSchemaProjection(gpa, &.{"missing"}));
    try std.testing.expect(std.mem.eql(u8, table_schema[0].name, schema.fields[0].name));
    try std.testing.expectEqual(table_schema[1].nullableColumn(), schema.fields[1].nullable);
    try std.testing.expectEqual(table_schema[2].nullableColumn(), schema.fields[2].nullable);
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
    var batch_roundtrip = try vectra.ArrowExport.DataFrame.Arrow.fromRecordBatch(gpa, batch, .cpu);
    defer batch_roundtrip.deinit();
    try std.testing.expectEqual(table.height(), batch_roundtrip.height());
    try std.testing.expect(batch_roundtrip.schemaEquals(table));
    var batch_projection = try vectra.ArrowExport.DataFrame.Arrow.fromRecordBatchProjection(gpa, batch, &.{ "sales", "active" }, .cpu);
    defer batch_projection.deinit();
    try std.testing.expectEqual(@as(usize, 2), batch_projection.width());
    try std.testing.expectEqual(@as(?usize, 0), batch_projection.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), batch_projection.columnIndex("active"));
    try std.testing.expectError(error.ColumnNotFound, vectra.ArrowExport.DataFrame.fromArrowRecordBatchProjection(gpa, batch, &.{"missing"}, .cpu));

    var arrow_table = try table.toArrowTable(gpa);
    defer arrow_table.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 1), arrow_table.batchCount());
    try std.testing.expectEqual(@as(usize, 3), arrow_table.row_count);
    try std.testing.expectEqual(@as(?usize, 1), arrow_table.columnIndexByName("units"));
    var grouped_arrow_table = try vectra.ArrowExport.DataFrame.Arrow.toTable(table, gpa);
    defer grouped_arrow_table.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 3), grouped_arrow_table.row_count);
    var table_roundtrip = try vectra.ArrowExport.DataFrame.Arrow.fromTable(gpa, grouped_arrow_table, .cpu);
    defer table_roundtrip.deinit();
    try std.testing.expectEqual(table.height(), table_roundtrip.height());
    try std.testing.expect(table_roundtrip.schemaEquals(table));
    var table_projection = try vectra.ArrowExport.DataFrame.Arrow.fromTableProjection(gpa, grouped_arrow_table, &.{"units"}, .cpu);
    defer table_projection.deinit();
    try std.testing.expectEqual(@as(usize, 1), table_projection.width());
    try std.testing.expectEqual(@as(?usize, 0), table_projection.columnIndex("units"));
    try std.testing.expectError(error.ColumnNotFound, vectra.ArrowExport.DataFrame.fromArrowTableProjection(gpa, grouped_arrow_table, &.{"missing"}, .cpu));
    const grouped_parquet_bytes = try vectra.ArrowExport.DataFrame.Parquet.toBytes(table, gpa);
    defer gpa.free(grouped_parquet_bytes);
    var tmp_parquet_dir = std.testing.tmpDir(.{});
    defer tmp_parquet_dir.cleanup();
    try vectra.ArrowExport.DataFrame.Parquet.writeFileInDir(table, tmp_parquet_dir.dir, std.testing.io, "frame.parquet");
    var file_roundtrip_df = try vectra.ArrowExport.DataFrame.Parquet.fromFileInDir(gpa, tmp_parquet_dir.dir, std.testing.io, "frame.parquet", .limited(1024 * 1024), .cpu);
    defer file_roundtrip_df.deinit();
    try std.testing.expect(file_roundtrip_df.schemaEquals(table));
    var file_pruned_df = try vectra.ArrowExport.DataFrame.Parquet.fromFilePrunedInDir(gpa, tmp_parquet_dir.dir, std.testing.io, "frame.parquet", .limited(1024 * 1024), "sales", .{ .f64 = .{ .min = 0.0 } }, .cpu);
    defer file_pruned_df.deinit();
    try std.testing.expect(file_pruned_df.schemaEquals(table));
    var tmp_scan_dir = std.testing.tmpDir(.{});
    defer tmp_scan_dir.cleanup();
    try tmp_scan_dir.dir.writeFile(std.testing.io, .{ .sub_path = "scan.parquet", .data = grouped_parquet_bytes });
    var file_scan = try vectra.ArrowExport.ParquetScan.Lifecycle.fromFileInDir(gpa, tmp_scan_dir.dir, std.testing.io, "scan.parquet", .limited(1024 * 1024), .cpu);
    defer file_scan.deinit();
    try std.testing.expectEqual(grouped_parquet_bytes.len, vectra.ArrowExport.ParquetScan.Source.sourceNbytes(file_scan));
    try std.testing.expectEqual(table.height(), try vectra.ArrowExport.ParquetScan.File.rowCount(file_scan));
    const lazy_owned_scan_bytes = try gpa.dupe(u8, grouped_parquet_bytes);
    var owned_lazy_scan = DeviceLazyFrame.scanParquetOwnedBytes(gpa, lazy_owned_scan_bytes, .cpu);
    defer owned_lazy_scan.deinit();
    try std.testing.expectEqualStrings("parquet_scan", owned_lazy_scan.sourceName());
    try std.testing.expect(owned_lazy_scan.isParquetScanSource());
    try std.testing.expect(!owned_lazy_scan.isDataFrameSource());
    try std.testing.expectEqual(@as(usize, 0), owned_lazy_scan.opCount());
    try std.testing.expectEqual(owned_lazy_scan.opCount(), owned_lazy_scan.rawOpCount());
    try std.testing.expectEqual(@as(usize, 0), try owned_lazy_scan.optimizedOpCount());
    try std.testing.expect(owned_lazy_scan.isOptimizedNoOp());
    try std.testing.expectEqual(table.height(), try owned_lazy_scan.rowCount());
    try std.testing.expectEqual(table.width(), try owned_lazy_scan.columnCount());
    const owned_lazy_names = try owned_lazy_scan.columnNames(gpa);
    defer {
        for (owned_lazy_names) |name| gpa.free(name);
        gpa.free(owned_lazy_names);
    }
    try std.testing.expectEqualStrings("sales", owned_lazy_names[0]);
    try std.testing.expect(owned_lazy_scan.columnNamesUnique());
    try std.testing.expect(!owned_lazy_scan.hasDuplicateColumnNames());
    try std.testing.expectEqual(@as(usize, 0), owned_lazy_scan.duplicateColumnNameCount());
    const owned_lazy_labels = try owned_lazy_scan.columnLabels(gpa);
    defer {
        for (owned_lazy_labels) |label| gpa.free(label);
        gpa.free(owned_lazy_labels);
    }
    try std.testing.expectEqualStrings("sales", owned_lazy_labels[0]);
    try std.testing.expectEqualStrings("active", owned_lazy_labels[2]);
    try std.testing.expectEqual(@as(?usize, 1), try owned_lazy_scan.columnIndex("units"));
    try std.testing.expect((try owned_lazy_scan.columnIndex("missing")) == null);
    const owned_lazy_name = (try owned_lazy_scan.columnNameAt(gpa, 1)).?;
    defer gpa.free(owned_lazy_name);
    try std.testing.expectEqualStrings("units", owned_lazy_name);
    try std.testing.expect(owned_lazy_scan.hasAllColumns(&.{ "sales", "units" }));
    try std.testing.expect(owned_lazy_scan.hasAnyColumn(&.{ "missing", "active" }));
    try std.testing.expect(!owned_lazy_scan.hasColumn("missing"));
    const lazy_dtypes = try owned_lazy_scan.columnDTypes(gpa);
    defer gpa.free(lazy_dtypes);
    try std.testing.expectEqualSlices(vectra.DeviceDType, &.{ .f64, .i64, .bool }, lazy_dtypes);
    const lazy_dtype_names = try owned_lazy_scan.dtypeNames(gpa);
    defer gpa.free(lazy_dtype_names);
    try std.testing.expectEqualStrings("f64", lazy_dtype_names[0]);
    try std.testing.expectEqual(vectra.DeviceDType.i64, try owned_lazy_scan.columnDType("units"));
    try std.testing.expectEqual(@as(?vectra.DeviceDType, .bool), try owned_lazy_scan.columnDTypeAt(2));
    const lazy_dtype_byte_sizes = try owned_lazy_scan.columnDTypeByteSizes(gpa);
    defer gpa.free(lazy_dtype_byte_sizes);
    try std.testing.expectEqualSlices(usize, &.{ 8, 8, 1 }, lazy_dtype_byte_sizes);
    const lazy_dtype_bit_sizes = try owned_lazy_scan.columnDTypeBitSizes(gpa);
    defer gpa.free(lazy_dtype_bit_sizes);
    try std.testing.expectEqualSlices(usize, &.{ 64, 64, 8 }, lazy_dtype_bit_sizes);
    const lazy_numeric_mask = try owned_lazy_scan.columnDTypeClassMask(gpa, .numeric);
    defer gpa.free(lazy_numeric_mask);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false }, lazy_numeric_mask);
    const lazy_scan_has_nulls_mask = try owned_lazy_scan.columnHasNullsMask(gpa);
    defer gpa.free(lazy_scan_has_nulls_mask);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false }, lazy_scan_has_nulls_mask);
    const lazy_scan_projected_has_nulls_mask = try owned_lazy_scan.columnHasNullsMaskProjection(gpa, &.{ "units", "active" });
    defer gpa.free(lazy_scan_projected_has_nulls_mask);
    try std.testing.expectEqualSlices(bool, &.{ true, false }, lazy_scan_projected_has_nulls_mask);
    try std.testing.expectEqual(@as(usize, 1), try owned_lazy_scan.columnsWithNullsCount());
    try std.testing.expectEqual(@as(usize, 2), try owned_lazy_scan.columnsWithoutNullsCount());
    try std.testing.expectEqual(@as(usize, 1), try owned_lazy_scan.columnsWithNullsCountProjection(&.{ "units", "active" }));
    try std.testing.expectEqual(@as(usize, 1), try owned_lazy_scan.columnsWithoutNullsCountProjection(&.{ "units", "active" }));
    const lazy_scan_float_mask = try owned_lazy_scan.columnIsFloatMask(gpa);
    defer gpa.free(lazy_scan_float_mask);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false }, lazy_scan_float_mask);
    const lazy_scan_bool_mask = try owned_lazy_scan.columnIsBoolMask(gpa);
    defer gpa.free(lazy_scan_bool_mask);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true }, lazy_scan_bool_mask);
    try std.testing.expectEqual(@as(usize, 2), try owned_lazy_scan.columnDTypeClassCount(.numeric));
    try std.testing.expectEqual(@as(usize, 2), try owned_lazy_scan.numericColumnCount());
    try std.testing.expectEqual(@as(usize, 2), try owned_lazy_scan.realColumnCount());
    try std.testing.expectEqual(@as(usize, 1), try owned_lazy_scan.floatColumnCount());
    try std.testing.expectEqual(@as(usize, 1), try owned_lazy_scan.integerColumnCount());
    try std.testing.expectEqual(@as(usize, 1), try owned_lazy_scan.signedIntegerColumnCount());
    try std.testing.expectEqual(@as(usize, 0), try owned_lazy_scan.unsignedIntegerColumnCount());
    try std.testing.expectEqual(@as(usize, 1), try owned_lazy_scan.boolColumnCount());
    try std.testing.expectEqual(@as(usize, 0), try owned_lazy_scan.complexColumnCount());
    try std.testing.expectEqual(@as(?bool, false), try owned_lazy_scan.columnNullableAt(0));
    try std.testing.expectEqual(@as(?bool, true), try owned_lazy_scan.columnNullableAt(1));
    try std.testing.expect((try owned_lazy_scan.columnNullableAt(99)) == null);
    try std.testing.expect(try owned_lazy_scan.columnNullable("units"));
    try std.testing.expectError(error.ColumnNotFound, owned_lazy_scan.columnNullable("missing"));
    const lazy_nullable_mask = try owned_lazy_scan.columnNullableMask(gpa);
    defer gpa.free(lazy_nullable_mask);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false }, lazy_nullable_mask);
    try std.testing.expectEqual(@as(usize, 1), try owned_lazy_scan.nullableColumnCount());
    try std.testing.expectEqual(@as(usize, 2), try owned_lazy_scan.nonNullableColumnCount());
    const lazy_scan_null_counts = try owned_lazy_scan.columnNullCounts(gpa);
    defer gpa.free(lazy_scan_null_counts);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 0 }, lazy_scan_null_counts);
    const lazy_scan_valid_counts = try owned_lazy_scan.columnValidCounts(gpa);
    defer gpa.free(lazy_scan_valid_counts);
    try std.testing.expectEqualSlices(usize, &.{ 3, 2, 3 }, lazy_scan_valid_counts);
    try std.testing.expectEqual(@as(usize, 1), try owned_lazy_scan.nullCount());
    try std.testing.expectEqual(@as(usize, 8), try owned_lazy_scan.validCount());
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 9.0), try owned_lazy_scan.nullRatio(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 8.0 / 9.0), try owned_lazy_scan.validRatio(), 1e-12);
    const lazy_scan_null_ratios = try owned_lazy_scan.columnNullRatios(gpa);
    defer gpa.free(lazy_scan_null_ratios);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), lazy_scan_null_ratios[1], 1e-12);
    const lazy_scan_valid_ratios = try owned_lazy_scan.columnValidRatios(gpa);
    defer gpa.free(lazy_scan_valid_ratios);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), lazy_scan_valid_ratios[1], 1e-12);
    try std.testing.expect(owned_lazy_scan.hasNullableColumns());
    try std.testing.expect(!owned_lazy_scan.allColumnsNullable());
    const lazy_schema_at = (try owned_lazy_scan.columnSchemaAt(1)).?;
    try std.testing.expectEqual(vectra.DeviceDType.i64, lazy_schema_at.dtype);
    try std.testing.expectEqualStrings("", lazy_schema_at.name);
    try std.testing.expectEqual(@as(usize, 1), lazy_schema_at.nullCount());
    try std.testing.expectEqual(@as(usize, 2), lazy_schema_at.validCount());
    try std.testing.expect((try owned_lazy_scan.columnSchemaAt(99)) == null);
    const lazy_units_schema = try owned_lazy_scan.columnSchema("units");
    try std.testing.expectEqual(vectra.DeviceDType.i64, lazy_units_schema.dtype);
    try std.testing.expectEqualStrings("units", lazy_units_schema.name);
    try std.testing.expectEqual(@as(usize, 1), lazy_units_schema.nullCount());
    try std.testing.expectEqual(@as(usize, 2), lazy_units_schema.validCount());
    try std.testing.expect(lazy_units_schema.sameNullability(lazy_schema_at));
    try std.testing.expectError(error.ColumnNotFound, owned_lazy_scan.columnSchema("missing"));
    const lazy_schema_summary = try owned_lazy_scan.schemaSummary(gpa);
    defer gpa.free(lazy_schema_summary);
    try std.testing.expectEqual(@as(usize, 3), lazy_schema_summary.len);
    try std.testing.expectEqual(vectra.DeviceDType.f64, lazy_schema_summary[0].dtype);
    var lazy_scan_arrow_schema = try owned_lazy_scan.toArrowSchema(gpa);
    defer lazy_scan_arrow_schema.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 3), lazy_scan_arrow_schema.fieldCount());
    const lazy_scan_arrow_fields = try owned_lazy_scan.toArrowFields(gpa);
    defer {
        for (lazy_scan_arrow_fields) |*field| field.deinit(gpa);
        gpa.free(lazy_scan_arrow_fields);
    }
    try std.testing.expectEqualStrings("units", lazy_scan_arrow_fields[1].name);
    try std.testing.expect(owned_lazy_scan.hasArrowProjection(&.{ "active", "sales" }));
    try std.testing.expect(!owned_lazy_scan.hasArrowProjection(&.{"missing"}));
    try std.testing.expect(owned_lazy_scan.hasSchemaProjection(&.{ "active", "sales" }));
    try std.testing.expect(!owned_lazy_scan.hasSchemaProjection(&.{"missing"}));
    const lazy_scan_projected_schemas = try owned_lazy_scan.columnSchemasProjection(gpa, &.{ "active", "sales" });
    defer gpa.free(lazy_scan_projected_schemas);
    try std.testing.expectEqual(@as(usize, 2), lazy_scan_projected_schemas.len);
    try std.testing.expectEqualStrings("active", lazy_scan_projected_schemas[0].name);
    try std.testing.expectEqual(vectra.DeviceDType.bool, lazy_scan_projected_schemas[0].dtype);
    try std.testing.expectEqualStrings("sales", lazy_scan_projected_schemas[1].name);
    const lazy_scan_schema_projection = try owned_lazy_scan.schemaProjection(gpa, &.{"units"});
    defer gpa.free(lazy_scan_schema_projection);
    try std.testing.expectEqual(@as(usize, 1), lazy_scan_schema_projection.len);
    try std.testing.expectEqualStrings("units", lazy_scan_schema_projection[0].name);
    const lazy_scan_schema_summary_projection = try owned_lazy_scan.schemaSummaryProjection(gpa, &.{"sales"});
    defer gpa.free(lazy_scan_schema_summary_projection);
    try std.testing.expectEqual(@as(usize, 1), lazy_scan_schema_summary_projection.len);
    try std.testing.expectEqualStrings("sales", lazy_scan_schema_summary_projection[0].name);
    const lazy_scan_projected_dtypes = try owned_lazy_scan.columnDTypesProjection(gpa, &.{ "active", "sales" });
    defer gpa.free(lazy_scan_projected_dtypes);
    try std.testing.expectEqualSlices(vectra.DeviceDType, &.{ .bool, .f64 }, lazy_scan_projected_dtypes);
    const lazy_scan_projected_dtype_names = try owned_lazy_scan.dtypeNamesProjection(gpa, &.{ "active", "sales" });
    defer gpa.free(lazy_scan_projected_dtype_names);
    try std.testing.expectEqualStrings("bool", lazy_scan_projected_dtype_names[0]);
    try std.testing.expectEqualStrings("f64", lazy_scan_projected_dtype_names[1]);
    const lazy_scan_projected_dtype_bytes = try owned_lazy_scan.columnDTypeByteSizesProjection(gpa, &.{ "active", "sales" });
    defer gpa.free(lazy_scan_projected_dtype_bytes);
    try std.testing.expectEqualSlices(usize, &.{ 1, 8 }, lazy_scan_projected_dtype_bytes);
    const lazy_scan_projected_dtype_bits = try owned_lazy_scan.columnDTypeBitSizesProjection(gpa, &.{ "active", "sales" });
    defer gpa.free(lazy_scan_projected_dtype_bits);
    try std.testing.expectEqualSlices(usize, &.{ 8, 64 }, lazy_scan_projected_dtype_bits);
    const lazy_scan_projected_numeric = try owned_lazy_scan.columnDTypeClassMaskProjection(gpa, &.{ "active", "sales" }, .numeric);
    defer gpa.free(lazy_scan_projected_numeric);
    try std.testing.expectEqualSlices(bool, &.{ false, true }, lazy_scan_projected_numeric);
    try std.testing.expectEqual(@as(usize, 1), try owned_lazy_scan.numericColumnCountProjection(&.{ "active", "sales" }));
    try std.testing.expectEqual(@as(usize, 1), try owned_lazy_scan.floatColumnCountProjection(&.{ "active", "sales" }));
    try std.testing.expectEqual(@as(usize, 0), try owned_lazy_scan.integerColumnCountProjection(&.{ "active", "sales" }));
    try std.testing.expectEqual(@as(usize, 1), try owned_lazy_scan.boolColumnCountProjection(&.{ "active", "sales" }));
    const lazy_scan_projected_nullable = try owned_lazy_scan.columnNullableMaskProjection(gpa, &.{ "units", "active" });
    defer gpa.free(lazy_scan_projected_nullable);
    try std.testing.expectEqualSlices(bool, &.{ true, false }, lazy_scan_projected_nullable);
    try std.testing.expectEqual(@as(usize, 1), try owned_lazy_scan.nullableColumnCountProjection(&.{ "units", "active" }));
    try std.testing.expectEqual(@as(usize, 1), try owned_lazy_scan.nonNullableColumnCountProjection(&.{ "units", "active" }));
    const lazy_scan_projected_null_counts = try owned_lazy_scan.columnNullCountsProjection(gpa, &.{ "units", "active" });
    defer gpa.free(lazy_scan_projected_null_counts);
    try std.testing.expectEqualSlices(usize, &.{ 1, 0 }, lazy_scan_projected_null_counts);
    const lazy_scan_projected_valid_counts = try owned_lazy_scan.columnValidCountsProjection(gpa, &.{ "units", "active" });
    defer gpa.free(lazy_scan_projected_valid_counts);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, lazy_scan_projected_valid_counts);
    try std.testing.expectEqual(@as(usize, 1), try owned_lazy_scan.nullCountProjection(&.{ "units", "active" }));
    try std.testing.expectEqual(@as(usize, 5), try owned_lazy_scan.validCountProjection(&.{ "units", "active" }));
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 6.0), try owned_lazy_scan.nullRatioProjection(&.{ "units", "active" }), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 6.0), try owned_lazy_scan.validRatioProjection(&.{ "units", "active" }), 1e-12);
    const lazy_scan_projected_null_ratios = try owned_lazy_scan.columnNullRatiosProjection(gpa, &.{ "units", "active" });
    defer gpa.free(lazy_scan_projected_null_ratios);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), lazy_scan_projected_null_ratios[0], 1e-12);
    const lazy_scan_projected_valid_ratios = try owned_lazy_scan.columnValidRatiosProjection(gpa, &.{ "units", "active" });
    defer gpa.free(lazy_scan_projected_valid_ratios);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), lazy_scan_projected_valid_ratios[0], 1e-12);
    try std.testing.expectError(error.ColumnNotFound, owned_lazy_scan.columnSchemasProjection(gpa, &.{"missing"}));
    try std.testing.expectError(error.ColumnNotFound, owned_lazy_scan.columnDTypesProjection(gpa, &.{"missing"}));
    try std.testing.expectError(error.ColumnNotFound, owned_lazy_scan.columnDTypeByteSizesProjection(gpa, &.{"missing"}));
    try std.testing.expectError(error.ColumnNotFound, owned_lazy_scan.columnDTypeClassMaskProjection(gpa, &.{"missing"}, .numeric));
    try std.testing.expectError(error.ColumnNotFound, owned_lazy_scan.columnNullableMaskProjection(gpa, &.{"missing"}));
    try std.testing.expectError(error.ColumnNotFound, owned_lazy_scan.columnNullCountsProjection(gpa, &.{"missing"}));
    const lazy_scan_projected_fields = try owned_lazy_scan.toArrowFieldsProjection(gpa, &.{ "active", "sales" });
    defer {
        for (lazy_scan_projected_fields) |*field| field.deinit(gpa);
        gpa.free(lazy_scan_projected_fields);
    }
    try std.testing.expectEqual(@as(usize, 2), lazy_scan_projected_fields.len);
    try std.testing.expectEqualStrings("active", lazy_scan_projected_fields[0].name);
    try std.testing.expectEqualStrings("sales", lazy_scan_projected_fields[1].name);
    const grouped_lazy_scan_fields = try vectra.ArrowExport.LazyFrame.toArrowFieldsProjection(&owned_lazy_scan, gpa, &.{"units"});
    defer {
        for (grouped_lazy_scan_fields) |*field| field.deinit(gpa);
        gpa.free(grouped_lazy_scan_fields);
    }
    try std.testing.expectEqual(@as(usize, 1), grouped_lazy_scan_fields.len);
    try std.testing.expectEqualStrings("units", grouped_lazy_scan_fields[0].name);
    var lazy_scan_projected_schema = try owned_lazy_scan.toArrowSchemaProjection(gpa, &.{ "active", "sales" });
    defer lazy_scan_projected_schema.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 2), lazy_scan_projected_schema.fieldCount());
    try std.testing.expectEqualStrings("active", lazy_scan_projected_schema.fields[0].name);
    try std.testing.expectEqualStrings("sales", lazy_scan_projected_schema.fields[1].name);
    try std.testing.expect(vectra.ArrowExport.LazyFrame.Arrow.hasProjection(&owned_lazy_scan, &.{"sales"}));
    var grouped_lazy_scan_schema = try vectra.DeviceLazyFrameArrow.Arrow.toSchemaProjection(&owned_lazy_scan, gpa, &.{"units"});
    defer grouped_lazy_scan_schema.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 1), grouped_lazy_scan_schema.fieldCount());
    try std.testing.expectEqualStrings("units", grouped_lazy_scan_schema.fields[0].name);
    try std.testing.expectError(error.ColumnNotFound, owned_lazy_scan.toArrowFieldsProjection(gpa, &.{"missing"}));
    try std.testing.expectError(error.ColumnNotFound, owned_lazy_scan.toArrowSchemaProjection(gpa, &.{"missing"}));
    try std.testing.expect(owned_lazy_scan.schemaEqualsSchemas(lazy_schema_summary));
    try std.testing.expect(owned_lazy_scan.sameSchemaSchemas(lazy_schema_summary));
    try std.testing.expect(owned_lazy_scan.schemaCompatibleSchemas(lazy_schema_summary));
    try std.testing.expect(owned_lazy_scan.schemaEquals(&owned_lazy_scan));
    try std.testing.expect(owned_lazy_scan.sameSchema(&owned_lazy_scan));
    try std.testing.expect(owned_lazy_scan.schemaCompatible(&owned_lazy_scan));
    var eager_lazy_for_schema = try DeviceLazyFrame.init(gpa, table);
    defer eager_lazy_for_schema.deinit();
    try std.testing.expect(!owned_lazy_scan.schemaEquals(&eager_lazy_for_schema));
    try std.testing.expect(owned_lazy_scan.schemaCompatible(&eager_lazy_for_schema));
    const lazy_schema_alias = try owned_lazy_scan.schema(gpa);
    defer gpa.free(lazy_schema_alias);
    try std.testing.expectEqual(@as(usize, 3), lazy_schema_alias.len);
    const lazy_column_schemas = try owned_lazy_scan.columnSchemas(gpa);
    defer gpa.free(lazy_column_schemas);
    try std.testing.expectEqual(@as(usize, 3), lazy_column_schemas.len);
    try std.testing.expect(lazy_column_schemas[1].schemaEquals(lazy_schema_summary[1]));
    try std.testing.expectEqual(grouped_parquet_bytes.len, owned_lazy_scan.sourceNbytes());
    try std.testing.expectEqual(grouped_parquet_bytes.len, owned_lazy_scan.sourceByteCount());
    try std.testing.expectEqual(grouped_parquet_bytes.len, owned_lazy_scan.nbytes());
    try std.testing.expectEqual(grouped_parquet_bytes.len, owned_lazy_scan.byteCount());
    try std.testing.expect(owned_lazy_scan.hasBytes());
    try std.testing.expectEqual(grouped_parquet_bytes.len, owned_lazy_scan.ownedNbytes());
    try std.testing.expectEqual(owned_lazy_scan.ownedNbytes(), owned_lazy_scan.memoryUsage());
    try std.testing.expectEqual(owned_lazy_scan.ownedNbytes(), owned_lazy_scan.estimatedSize());
    try std.testing.expectEqual(table.height() * table.width(), try owned_lazy_scan.cellCount());
    const owned_lazy_shape = try owned_lazy_scan.shape();
    try std.testing.expectEqual(table.height(), owned_lazy_shape.rows);
    try std.testing.expectEqual(table.width(), owned_lazy_shape.cols);
    try std.testing.expectEqual(table.height(), try owned_lazy_scan.height());
    try std.testing.expectEqual(table.width(), try owned_lazy_scan.cols());
    try std.testing.expect(owned_lazy_scan.hasShape(table.height(), table.width()));
    try std.testing.expect(owned_lazy_scan.shapeEquals(table.height(), table.width()));
    try std.testing.expect(owned_lazy_scan.sameHeight(&owned_lazy_scan));
    try std.testing.expect(owned_lazy_scan.sameWidth(&owned_lazy_scan));
    try std.testing.expect(owned_lazy_scan.sameShape(&owned_lazy_scan));
    try std.testing.expect(!owned_lazy_scan.isEmpty());
    try std.testing.expect(owned_lazy_scan.isNonEmpty());
    try std.testing.expectEqualStrings("cpu", owned_lazy_scan.deviceBackendName());
    try std.testing.expect(owned_lazy_scan.isCpu());
    try std.testing.expect(owned_lazy_scan.isHostBacked());
    try std.testing.expect(!owned_lazy_scan.isCudaBacked());
    try std.testing.expect(!owned_lazy_scan.isMpsBacked());
    try std.testing.expect(!owned_lazy_scan.isAcceleratorBacked());
    try std.testing.expect(!owned_lazy_scan.isRemoteBacked());
    try std.testing.expect(!owned_lazy_scan.isDeviceBacked());
    try std.testing.expect(owned_lazy_scan.isDeviceAvailable());
    try std.testing.expect(owned_lazy_scan.sameDevice(&owned_lazy_scan));
    try std.testing.expect(owned_lazy_scan.sameStorage(&owned_lazy_scan));
    try std.testing.expect(owned_lazy_scan.sharesStorage(&owned_lazy_scan));
    try std.testing.expect(owned_lazy_scan.sameSource(&owned_lazy_scan));
    try std.testing.expect(owned_lazy_scan.sharesSource(&owned_lazy_scan));
    var eager_lazy_for_storage = try DeviceLazyFrame.init(gpa, table);
    defer eager_lazy_for_storage.deinit();
    try std.testing.expect(!owned_lazy_scan.sameStorage(&eager_lazy_for_storage));
    var owned_lazy_rows = try owned_lazy_scan.collect();
    defer owned_lazy_rows.deinit();
    try std.testing.expectEqual(table.height(), owned_lazy_rows.height());
    var file_lazy_scan = try DeviceLazyFrame.scanParquetFileInDir(gpa, tmp_scan_dir.dir, std.testing.io, "scan.parquet", .limited(1024 * 1024), .cpu);
    defer file_lazy_scan.deinit();
    try file_lazy_scan.select(&.{"sales"});
    try std.testing.expectEqual(@as(usize, 1), file_lazy_scan.opCount());
    try std.testing.expectEqual(file_lazy_scan.opCount(), file_lazy_scan.rawOpCount());
    try std.testing.expectEqual(@as(usize, 1), try file_lazy_scan.optimizedOpCount());
    try std.testing.expect(!file_lazy_scan.isOptimizedNoOp());
    var file_lazy_pushdown = try file_lazy_scan.scanPushdownSummary();
    defer file_lazy_pushdown.deinit();
    try std.testing.expect(file_lazy_scan.hasScanPushdown());
    try std.testing.expect(file_lazy_scan.usesScanPushdown());
    try std.testing.expect(file_lazy_scan.usesScanPushdownCollect());
    try std.testing.expect(file_lazy_pushdown.hasProjection());
    try std.testing.expectEqual(@as(usize, 1), file_lazy_pushdown.projectionColumnCount());
    try std.testing.expect(file_lazy_pushdown.projectionNamesUnique());
    try std.testing.expect(!file_lazy_pushdown.hasDuplicateProjectionNames());
    try std.testing.expectEqual(@as(usize, 0), file_lazy_pushdown.duplicateProjectionNameCount());
    try std.testing.expect(file_lazy_pushdown.projectionContains("sales"));
    try std.testing.expect(file_lazy_pushdown.hasAllProjectionNames(&.{"sales"}));
    try std.testing.expect(file_lazy_pushdown.hasAnyProjectionName(&.{ "missing", "sales" }));
    try std.testing.expect(!file_lazy_pushdown.hasAnyProjectionName(&.{"missing"}));
    const file_lazy_pushdown_explain = try file_lazy_pushdown.explain(gpa);
    defer gpa.free(file_lazy_pushdown_explain);
    try std.testing.expect(std.mem.indexOf(u8, file_lazy_pushdown_explain, "projection=[sales]") != null);
    try std.testing.expect(file_lazy_pushdown.projectsColumn("sales"));
    try std.testing.expect(!file_lazy_pushdown.projectsColumn("units"));
    try std.testing.expect(!file_lazy_pushdown.hasPredicate());
    try std.testing.expect(file_lazy_pushdown.pushdownMetadataNbytes() >= "sales".len);
    try std.testing.expectEqual(file_lazy_pushdown.pushdownMetadataNbytes(), file_lazy_pushdown.memoryUsage());
    try std.testing.expectEqual(file_lazy_pushdown.pushdownMetadataNbytes(), file_lazy_pushdown.estimatedSize());
    const file_lazy_pushdown_summary = file_lazy_pushdown.summary();
    try std.testing.expect(file_lazy_pushdown_summary.hasProjection());
    try std.testing.expectEqual(file_lazy_pushdown.projectionColumnCount(), file_lazy_pushdown_summary.projectionColumnCount());
    var file_lazy_owned_summary = try file_lazy_pushdown.summaryOwned(gpa);
    defer file_lazy_owned_summary.deinit();
    try std.testing.expect(file_lazy_owned_summary.summary().projectionContains("sales"));
    var file_lazy_owned_frame_summary = try file_lazy_scan.scanPushdownSummaryOwned(gpa);
    defer file_lazy_owned_frame_summary.deinit();
    try std.testing.expect(file_lazy_owned_frame_summary.summary().projectionContains("sales"));
    const file_lazy_summary = try file_lazy_scan.explainSummary(gpa);
    defer gpa.free(file_lazy_summary);
    try std.testing.expect(std.mem.indexOf(u8, file_lazy_summary, "DeviceLazyFrame(raw_ops=1, optimized_ops=1, source=parquet_scan)") != null);
    var filtered_lazy_scan = try DeviceLazyFrame.scanParquetFileInDir(gpa, tmp_scan_dir.dir, std.testing.io, "scan.parquet", .limited(1024 * 1024), .cpu);
    defer filtered_lazy_scan.deinit();
    try filtered_lazy_scan.filterColumnScalar("sales", f64, 0.0, .ge);
    var filtered_pushdown = try filtered_lazy_scan.scanPushdownSummary();
    defer filtered_pushdown.deinit();
    try std.testing.expect(filtered_pushdown.hasPredicate());
    try std.testing.expectEqualStrings("sales", filtered_pushdown.predicateColumn().?);
    try std.testing.expect(filtered_pushdown.hasPredicateFor("sales"));
    try std.testing.expect(filtered_pushdown.hasRangePredicate());
    try std.testing.expect(filtered_pushdown.rangePredicate() != null);
    try std.testing.expect(filtered_pushdown.hasRangePredicateFor("sales"));
    try std.testing.expectEqual(vectra.DeviceDType.f64, filtered_pushdown.rangePredicateDType().?);
    try std.testing.expect(filtered_pushdown.nullPredicate() == null);
    var filtered_owned_summary = try filtered_pushdown.summaryOwned(gpa);
    defer filtered_owned_summary.deinit();
    const filtered_owned_value = filtered_owned_summary.summary();
    try std.testing.expect(filtered_owned_value.hasRangePredicateFor("sales"));
    try std.testing.expectEqualStrings("sales", filtered_owned_value.rangePredicateColumn().?);
    try std.testing.expectEqual(table.height(), try file_lazy_scan.rowCount());
    try std.testing.expectEqual(@as(usize, 3), try file_lazy_scan.columnCount());
    var file_lazy_rows = try file_lazy_scan.collect();
    defer file_lazy_rows.deinit();
    try std.testing.expectEqual(table.height(), file_lazy_rows.height());
    var parquet_roundtrip = try vectra.ArrowExport.DataFrame.Parquet.fromBytes(gpa, grouped_parquet_bytes, .cpu);
    defer parquet_roundtrip.deinit();
    try std.testing.expectEqual(table.height(), parquet_roundtrip.height());
    try std.testing.expect(parquet_roundtrip.schemaEquals(table));
    var grouped_pruned = try vectra.ArrowExport.DataFrame.fromParquetBytesPruned(
        gpa,
        grouped_parquet_bytes,
        "sales",
        .{ .f64 = .{ .min = 0.0 } },
        .cpu,
    );
    defer grouped_pruned.deinit();
    try std.testing.expectEqual(table.height(), grouped_pruned.height());
    try std.testing.expect(grouped_pruned.schemaEquals(table));
    try std.testing.expectError(error.UnsupportedParquetSchema, vectra.ArrowExport.DataFrame.fromParquetBytesPruned(
        gpa,
        grouped_parquet_bytes,
        "missing",
        .{ .f64 = .{ .min = 0.0 } },
        .cpu,
    ));
    const owned_scan_bytes = try gpa.dupe(u8, grouped_parquet_bytes);
    var owned_scan = vectra.ArrowExport.ParquetScan.Lifecycle.initOwnedBytes(gpa, owned_scan_bytes, .cpu);
    defer owned_scan.deinit();
    try std.testing.expectEqual(grouped_parquet_bytes.len, vectra.ArrowExport.ParquetScan.sourceNbytes(owned_scan));
    const moved_scan_bytes = vectra.ArrowExport.ParquetScan.Lifecycle.moveBytes(&owned_scan);
    defer gpa.free(moved_scan_bytes);
    try std.testing.expectEqual(grouped_parquet_bytes.len, moved_scan_bytes.len);
    try std.testing.expectEqual(@as(usize, 0), vectra.ArrowExport.ParquetScan.sourceNbytes(owned_scan));
    var grouped_scan = try vectra.ArrowExport.ParquetScan.Lifecycle.init(gpa, grouped_parquet_bytes, .cpu);
    defer grouped_scan.deinit();
    try std.testing.expect(vectra.ArrowExport.ParquetScan.Device.deviceValue(grouped_scan).sameDevice(.cpu));
    try std.testing.expectEqual(grouped_parquet_bytes.len, vectra.ArrowExport.ParquetScan.Source.sourceNbytes(grouped_scan));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.deviceBackend(grouped_scan) == .cpu);
    try std.testing.expectEqualStrings("cpu", vectra.ArrowExport.ParquetScan.deviceBackendName(grouped_scan));
    try std.testing.expectEqual(@as(usize, 0), vectra.ArrowExport.ParquetScan.deviceIndex(grouped_scan));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.isCpu(grouped_scan));
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.isCuda(grouped_scan));
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.isMps(grouped_scan));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.isHostBacked(grouped_scan));
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.isCudaBacked(grouped_scan));
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.isMpsBacked(grouped_scan));
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.isAcceleratorBacked(grouped_scan));
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.isRemoteBacked(grouped_scan));
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.isDeviceBacked(grouped_scan));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.isDeviceAvailable(grouped_scan));
    try std.testing.expectEqual(grouped_parquet_bytes.len, vectra.ArrowExport.ParquetScan.sourceNbytes(grouped_scan));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.sourcePtr(grouped_scan) != 0);
    try std.testing.expectEqual(vectra.ArrowExport.ParquetScan.sourcePtr(grouped_scan), vectra.ArrowExport.ParquetScan.dataPtr(grouped_scan));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.hasSourcePtr(grouped_scan));
    try std.testing.expectEqual(vectra.ArrowExport.ParquetScan.sourcePtr(grouped_scan) + grouped_parquet_bytes.len, vectra.ArrowExport.ParquetScan.sourceEndPtr(grouped_scan));
    const parquet_file_summary: vectra.DeviceParquetFileSummary = try vectra.ArrowExport.ParquetScan.parquetFileSummary(grouped_scan);
    try std.testing.expectEqual(table.height(), parquet_file_summary.rowCount());
    try std.testing.expectEqual(table.height(), parquet_file_summary.nRows());
    try std.testing.expectEqual(table.height(), parquet_file_summary.rowGroupRowCount());
    try std.testing.expectEqual(@as(usize, 1), parquet_file_summary.rowGroupCount());
    try std.testing.expectEqual(table.width(), parquet_file_summary.columnChunkCount());
    try std.testing.expect(parquet_file_summary.hasRows());
    try std.testing.expect(parquet_file_summary.hasRowGroups());
    try std.testing.expect(parquet_file_summary.hasColumns());
    try std.testing.expect(parquet_file_summary.allColumnsHaveMetadata());
    try std.testing.expect(!parquet_file_summary.anyColumnsMissingMetadata());
    try std.testing.expectEqual(parquet_file_summary.totalUncompressedNbytes(), parquet_file_summary.totalNbytes());
    try std.testing.expectEqual(parquet_file_summary.totalNbytes(), parquet_file_summary.memoryUsage());
    try std.testing.expectEqual(parquet_file_summary.totalNbytes(), parquet_file_summary.estimatedSize());
    try std.testing.expect(parquet_file_summary.anyColumnsHavePageIndex());
    try std.testing.expect(parquet_file_summary.allColumnsHavePageIndex());
    try std.testing.expect(parquet_file_summary.anyColumnsHaveBloomFilter());
    try std.testing.expect(parquet_file_summary.allBloomFiltersSized());
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), parquet_file_summary.metadataCoverageRatio(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), parquet_file_summary.missingMetadataRatio(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), parquet_file_summary.columnIndexCoverageRatio(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), parquet_file_summary.offsetIndexCoverageRatio(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), parquet_file_summary.pageIndexCoverageRatio(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), parquet_file_summary.bloomFilterCoverageRatio(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), parquet_file_summary.sizedBloomFilterCoverageRatio(), 1e-12);
    try std.testing.expect(parquet_file_summary.compressionRatio() > 0.0);
    try std.testing.expectEqual(table.height(), try vectra.ArrowExport.ParquetScan.File.rowCount(grouped_scan));
    try std.testing.expectEqual(table.height(), try vectra.ArrowExport.ParquetScan.rowCount(grouped_scan));
    try std.testing.expectEqual(table.height(), try vectra.ArrowExport.ParquetScan.nRows(grouped_scan));
    try std.testing.expectEqual(@as(usize, 1), try vectra.ArrowExport.ParquetScan.rowGroupCount(grouped_scan));
    try std.testing.expectEqual(table.width(), try vectra.ArrowExport.ParquetScan.parquetColumnChunkCount(grouped_scan));
    try std.testing.expectEqual(table.width(), try vectra.ArrowExport.ParquetScan.columnCount(grouped_scan));
    try std.testing.expectEqual(table.width(), try vectra.ArrowExport.ParquetScan.width(grouped_scan));
    try std.testing.expectEqual(table.width(), try vectra.ArrowExport.ParquetScan.cols(grouped_scan));
    try std.testing.expectEqual(table.width(), try vectra.ArrowExport.ParquetScan.nCols(grouped_scan));
    try std.testing.expectEqual(table.height() * table.width(), try vectra.ArrowExport.ParquetScan.cellCount(grouped_scan));
    const scan_shape = try vectra.ArrowExport.ParquetScan.shape(grouped_scan);
    try std.testing.expectEqual(table.height(), scan_shape.rows);
    try std.testing.expectEqual(table.width(), scan_shape.cols);
    try std.testing.expect(vectra.ArrowExport.ParquetScan.hasRows(grouped_scan));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.hasColumns(grouped_scan));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.hasShape(grouped_scan, table.height(), table.width()));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.shapeEquals(grouped_scan, table.height(), table.width()));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.sameHeight(grouped_scan, grouped_scan));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.sameWidth(grouped_scan, grouped_scan));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.sameShape(grouped_scan, grouped_scan));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.File.sameRowGroups(grouped_scan, grouped_scan));
    try std.testing.expectEqual(parquet_file_summary.totalNbytes(), try vectra.ArrowExport.ParquetScan.parquetTotalNbytes(grouped_scan));
    try std.testing.expectEqual(parquet_file_summary.totalCompressedNbytes(), try vectra.ArrowExport.ParquetScan.parquetTotalCompressedNbytes(grouped_scan));
    try std.testing.expectEqual(parquet_file_summary.totalUncompressedNbytes(), try vectra.ArrowExport.ParquetScan.parquetTotalUncompressedNbytes(grouped_scan));
    try std.testing.expectApproxEqAbs(parquet_file_summary.compressionRatio(), try vectra.ArrowExport.ParquetScan.parquetCompressionRatio(grouped_scan), 1e-12);
    try std.testing.expectApproxEqAbs(parquet_file_summary.metadataCoverageRatio(), try vectra.ArrowExport.ParquetScan.parquetMetadataCoverageRatio(grouped_scan), 1e-12);
    try std.testing.expectApproxEqAbs(parquet_file_summary.pageIndexCoverageRatio(), try vectra.ArrowExport.ParquetScan.parquetPageIndexCoverageRatio(grouped_scan), 1e-12);
    try std.testing.expect(vectra.ArrowExport.ParquetScan.hasRowGroups(grouped_scan));
    const grouped_scan_source: vectra.DeviceParquetScanSourceRange = vectra.ArrowExport.ParquetScan.sourceRange(grouped_scan);
    try std.testing.expectEqual(vectra.ArrowExport.ParquetScan.sourcePtr(grouped_scan), grouped_scan_source.sourcePtr());
    try std.testing.expectEqual(vectra.ArrowExport.ParquetScan.sourceNbytes(grouped_scan), grouped_scan_source.sourceNbytes());
    try std.testing.expectEqual(vectra.ArrowExport.ParquetScan.sourceEndPtr(grouped_scan), grouped_scan_source.sourceEndPtr());
    try std.testing.expect(grouped_scan_source.hasPtr());
    try std.testing.expect(grouped_scan_source.isNonEmpty());
    try std.testing.expect(vectra.ArrowExport.ParquetScan.hasArrowProjection(grouped_scan, &.{ "sales", "units" }));
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.hasArrowProjection(grouped_scan, &.{"missing"}));
    try std.testing.expectEqual(@as(usize, 3), try vectra.ArrowExport.ParquetScan.Arrow.arrowFieldCount(grouped_scan));
    try std.testing.expectEqual(@as(usize, 3), try vectra.ArrowExport.ParquetScan.arrowFieldCount(grouped_scan));
    const scan_field_name = (try vectra.ArrowExport.ParquetScan.arrowFieldNameAt(grouped_scan, gpa, 1)).?;
    defer gpa.free(scan_field_name);
    try std.testing.expectEqualStrings("units", scan_field_name);
    try std.testing.expect((try vectra.ArrowExport.ParquetScan.arrowFieldNameAt(grouped_scan, gpa, 99)) == null);
    const scan_field_names = try vectra.ArrowExport.ParquetScan.arrowFieldNames(grouped_scan, gpa);
    defer {
        for (scan_field_names) |name| gpa.free(name);
        gpa.free(scan_field_names);
    }
    try std.testing.expectEqual(@as(usize, 3), scan_field_names.len);
    try std.testing.expectEqualStrings("sales", scan_field_names[0]);
    try std.testing.expectEqual(@as(?usize, 2), try vectra.ArrowExport.ParquetScan.arrowFieldIndex(grouped_scan, "active"));
    try std.testing.expect(try vectra.ArrowExport.ParquetScan.arrowFieldIndex(grouped_scan, "missing") == null);
    try std.testing.expect(vectra.ArrowExport.ParquetScan.hasArrowField(grouped_scan, "units"));
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.hasArrowField(grouped_scan, "missing"));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.hasAllArrowFields(grouped_scan, &.{ "sales", "active" }));
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.hasAllArrowFields(grouped_scan, &.{ "sales", "missing" }));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.hasAnyArrowField(grouped_scan, &.{ "missing", "active" }));
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.hasAnyArrowField(grouped_scan, &.{"missing"}));
    try std.testing.expectEqual(@as(?vectra.DeviceDType, .f64), try vectra.ArrowExport.ParquetScan.arrowFieldDTypeAt(grouped_scan, 0));
    try std.testing.expect((try vectra.ArrowExport.ParquetScan.arrowFieldDTypeAt(grouped_scan, 99)) == null);
    try std.testing.expectEqual(vectra.DeviceDType.i64, try vectra.ArrowExport.ParquetScan.arrowFieldDType(grouped_scan, "units"));
    try std.testing.expectError(error.ColumnNotFound, vectra.ArrowExport.ParquetScan.arrowFieldDType(grouped_scan, "missing"));
    const scan_dtypes = try vectra.ArrowExport.ParquetScan.arrowFieldDTypes(grouped_scan, gpa);
    defer gpa.free(scan_dtypes);
    try std.testing.expectEqualSlices(vectra.DeviceDType, &.{ .f64, .i64, .bool }, scan_dtypes);
    const scan_dtype_names = try vectra.ArrowExport.ParquetScan.arrowFieldDTypeNames(grouped_scan, gpa);
    defer gpa.free(scan_dtype_names);
    try std.testing.expectEqualStrings("f64", scan_dtype_names[0]);
    try std.testing.expectEqualStrings("i64", scan_dtype_names[1]);
    try std.testing.expectEqualStrings("bool", scan_dtype_names[2]);
    const scan_dtype_bytes = try vectra.ArrowExport.ParquetScan.arrowFieldDTypeByteSizes(grouped_scan, gpa);
    defer gpa.free(scan_dtype_bytes);
    try std.testing.expectEqualSlices(usize, &.{ 8, 8, 1 }, scan_dtype_bytes);
    const scan_dtype_bits = try vectra.ArrowExport.ParquetScan.arrowFieldDTypeBitSizes(grouped_scan, gpa);
    defer gpa.free(scan_dtype_bits);
    try std.testing.expectEqualSlices(usize, &.{ 64, 64, 8 }, scan_dtype_bits);
    const scan_numeric_mask = try vectra.ArrowExport.ParquetScan.arrowFieldDTypeClassMask(grouped_scan, gpa, .numeric);
    defer gpa.free(scan_numeric_mask);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false }, scan_numeric_mask);
    try std.testing.expectEqual(@as(usize, 2), try vectra.ArrowExport.ParquetScan.numericArrowFieldCount(grouped_scan));
    try std.testing.expectEqual(@as(usize, 1), try vectra.ArrowExport.ParquetScan.floatArrowFieldCount(grouped_scan));
    try std.testing.expectEqual(@as(usize, 1), try vectra.ArrowExport.ParquetScan.integerArrowFieldCount(grouped_scan));
    try std.testing.expectEqual(@as(usize, 1), try vectra.ArrowExport.ParquetScan.boolArrowFieldCount(grouped_scan));
    try std.testing.expectEqual(@as(?bool, false), try vectra.ArrowExport.ParquetScan.arrowFieldNullableAt(grouped_scan, 0));
    try std.testing.expectEqual(@as(?bool, true), try vectra.ArrowExport.ParquetScan.arrowFieldNullableAt(grouped_scan, 1));
    try std.testing.expect((try vectra.ArrowExport.ParquetScan.arrowFieldNullableAt(grouped_scan, 99)) == null);
    try std.testing.expect(try vectra.ArrowExport.ParquetScan.arrowFieldNullable(grouped_scan, "units"));
    try std.testing.expectError(error.ColumnNotFound, vectra.ArrowExport.ParquetScan.arrowFieldNullable(grouped_scan, "missing"));
    const scan_nullable_mask = try vectra.ArrowExport.ParquetScan.arrowFieldNullableMask(grouped_scan, gpa);
    defer gpa.free(scan_nullable_mask);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false }, scan_nullable_mask);
    try std.testing.expectEqual(@as(usize, 1), try vectra.ArrowExport.ParquetScan.nullableArrowFieldCount(grouped_scan));
    try std.testing.expectEqual(@as(usize, 2), try vectra.ArrowExport.ParquetScan.nonNullableArrowFieldCount(grouped_scan));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.hasNullableArrowFields(grouped_scan));
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.allArrowFieldsNullable(grouped_scan));
    try std.testing.expectEqual(@as(usize, 1), try vectra.ArrowExport.ParquetScan.arrowFieldNullCount(grouped_scan, "units"));
    try std.testing.expectEqual(@as(usize, 2), try vectra.ArrowExport.ParquetScan.arrowFieldValidCount(grouped_scan, "units"));
    try std.testing.expectError(error.ColumnNotFound, vectra.ArrowExport.ParquetScan.arrowFieldNullCount(grouped_scan, "missing"));
    const scan_null_counts = try vectra.ArrowExport.ParquetScan.arrowFieldNullCounts(grouped_scan, gpa);
    defer gpa.free(scan_null_counts);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 0 }, scan_null_counts);
    const scan_valid_counts = try vectra.ArrowExport.ParquetScan.arrowFieldValidCounts(grouped_scan, gpa);
    defer gpa.free(scan_valid_counts);
    try std.testing.expectEqualSlices(usize, &.{ 3, 2, 3 }, scan_valid_counts);
    try std.testing.expectEqual(@as(usize, 1), try vectra.ArrowExport.ParquetScan.arrowNullCount(grouped_scan));
    try std.testing.expectEqual(@as(usize, 8), try vectra.ArrowExport.ParquetScan.arrowValidCount(grouped_scan));
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 9.0), try vectra.ArrowExport.ParquetScan.arrowNullRatio(grouped_scan), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 8.0 / 9.0), try vectra.ArrowExport.ParquetScan.arrowValidRatio(grouped_scan), 1e-12);
    const scan_null_ratios = try vectra.ArrowExport.ParquetScan.arrowFieldNullRatios(grouped_scan, gpa);
    defer gpa.free(scan_null_ratios);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), scan_null_ratios[1], 1e-12);
    const scan_valid_ratios = try vectra.ArrowExport.ParquetScan.arrowFieldValidRatios(grouped_scan, gpa);
    defer gpa.free(scan_valid_ratios);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), scan_valid_ratios[1], 1e-12);
    const scan_units_schema = try vectra.ArrowExport.ParquetScan.arrowColumnSchema(grouped_scan, "units");
    try std.testing.expectEqual(vectra.DeviceDType.i64, scan_units_schema.dtype);
    try std.testing.expectEqual(table.height(), scan_units_schema.len());
    try std.testing.expect(scan_units_schema.nullableColumn());
    try std.testing.expectEqual(@as(usize, 1), scan_units_schema.nullCount());
    try std.testing.expectEqual(@as(usize, 2), scan_units_schema.validCount());
    try std.testing.expect(scan_units_schema.isCpu());
    try std.testing.expectError(error.ColumnNotFound, vectra.ArrowExport.ParquetScan.arrowColumnSchema(grouped_scan, "missing"));
    const scan_schema_at = (try vectra.ArrowExport.ParquetScan.arrowColumnSchemaAt(grouped_scan, 0)).?;
    try std.testing.expectEqual(vectra.DeviceDType.f64, scan_schema_at.dtype);
    try std.testing.expectEqualStrings("", scan_schema_at.name);
    try std.testing.expect((try vectra.ArrowExport.ParquetScan.arrowColumnSchemaAt(grouped_scan, 99)) == null);
    const scan_column_schemas = try vectra.ArrowExport.ParquetScan.arrowColumnSchemas(grouped_scan, gpa);
    defer gpa.free(scan_column_schemas);
    try std.testing.expectEqual(@as(usize, 3), scan_column_schemas.len);
    try std.testing.expectEqualStrings("", scan_column_schemas[0].name);
    try std.testing.expectEqual(vectra.DeviceDType.bool, scan_column_schemas[2].dtype);
    const scan_schema_summary = try vectra.ArrowExport.ParquetScan.arrowSchemaSummary(grouped_scan, gpa);
    defer gpa.free(scan_schema_summary);
    try std.testing.expectEqual(@as(usize, 3), scan_schema_summary.len);
    try std.testing.expectEqual(vectra.DeviceDType.f64, scan_schema_summary[0].dtype);
    try std.testing.expectEqual(table.height(), scan_schema_summary[0].len());
    try std.testing.expect(vectra.ArrowExport.ParquetScan.Arrow.arrowSchemaEquals(grouped_scan, grouped_scan));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.arrowSameSchema(grouped_scan, grouped_scan));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.arrowSchemaCompatible(grouped_scan, grouped_scan));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.arrowSchemaEqualsSchemas(grouped_scan, scan_schema_summary));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.Arrow.arrowSchemaEqualsFrame(grouped_scan, table));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.arrowSameSchemaFrame(grouped_scan, table));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.arrowSchemaCompatibleFrame(grouped_scan, table));
    var scan_arrow_schema = try vectra.ArrowExport.ParquetScan.toArrowSchema(grouped_scan, gpa);
    defer scan_arrow_schema.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 3), scan_arrow_schema.fieldCount());
    try std.testing.expectEqual(@as(?usize, 0), scan_arrow_schema.fieldIndexByName("sales"));
    const scan_arrow_fields = try vectra.ArrowExport.ParquetScan.toArrowFields(grouped_scan, gpa);
    defer {
        for (scan_arrow_fields) |*field| field.deinit(gpa);
        gpa.free(scan_arrow_fields);
    }
    try std.testing.expectEqual(@as(usize, 3), scan_arrow_fields.len);
    try std.testing.expectEqualStrings("units", scan_arrow_fields[1].name);
    try std.testing.expectEqual(grouped_parquet_bytes.len, vectra.ArrowExport.ParquetScan.sourceByteCount(grouped_scan));
    try std.testing.expectEqual(grouped_parquet_bytes.len, vectra.ArrowExport.ParquetScan.nbytes(grouped_scan));
    try std.testing.expectEqual(grouped_parquet_bytes.len, vectra.ArrowExport.ParquetScan.byteCount(grouped_scan));
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.isEmpty(grouped_scan));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.isNonEmpty(grouped_scan));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.hasBytes(grouped_scan));
    try std.testing.expectEqual(@as(usize, 0), vectra.ArrowExport.ParquetScan.projectionMetadataNbytes(grouped_scan));
    try std.testing.expectEqual(@as(usize, 0), vectra.ArrowExport.ParquetScan.predicateMetadataNbytes(grouped_scan));
    try std.testing.expectEqual(@as(usize, 0), vectra.ArrowExport.ParquetScan.pushdownMetadataNbytes(grouped_scan));
    try std.testing.expectEqual(grouped_parquet_bytes.len, vectra.ArrowExport.ParquetScan.ownedNbytes(grouped_scan));
    try std.testing.expectEqual(vectra.ArrowExport.ParquetScan.ownedNbytes(grouped_scan), vectra.ArrowExport.ParquetScan.memoryUsage(grouped_scan));
    try std.testing.expectEqual(vectra.ArrowExport.ParquetScan.ownedNbytes(grouped_scan), vectra.ArrowExport.ParquetScan.estimatedSize(grouped_scan));
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.hasPushdown(grouped_scan));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.projectionNameAt(grouped_scan, 0) == null);
    try std.testing.expect(vectra.ArrowExport.ParquetScan.projectionIndex(grouped_scan, "sales") == null);
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.projectionContains(grouped_scan, "sales"));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.projectsColumn(grouped_scan, "sales"));
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.hasPredicate(grouped_scan));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.predicateColumn(grouped_scan) == null);
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.hasPredicateFor(grouped_scan, "sales"));
    try vectra.ArrowExport.ParquetScan.Pushdown.select(&grouped_scan, &.{"sales"});
    try vectra.ArrowExport.ParquetScan.Pushdown.appendSelect(&grouped_scan, &.{ "units", "sales" });
    try std.testing.expectEqual(@as(usize, 2), vectra.ArrowExport.ParquetScan.projectionColumnCount(grouped_scan));
    try vectra.ArrowExport.ParquetScan.Pushdown.dropSelected(&grouped_scan, &.{"missing"});
    try std.testing.expectEqual(@as(usize, 2), vectra.ArrowExport.ParquetScan.projectionColumnCount(grouped_scan));
    try vectra.ArrowExport.ParquetScan.Pushdown.intersectSelect(&grouped_scan, &.{ "units", "sales", "extra" });
    try std.testing.expectEqual(@as(usize, 2), vectra.ArrowExport.ParquetScan.projectionColumnCount(grouped_scan));
    try vectra.ArrowExport.ParquetScan.Pushdown.validateProjection(grouped_scan);
    try vectra.ArrowExport.ParquetScan.validatePushdown(grouped_scan);
    try std.testing.expect(vectra.ArrowExport.ParquetScan.pushdownValid(grouped_scan));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.hasProjection(grouped_scan));
    try std.testing.expectEqual(@as(usize, 2), vectra.ArrowExport.ParquetScan.projectionColumnCount(grouped_scan));
    try std.testing.expect(std.mem.eql(u8, "sales", vectra.ArrowExport.ParquetScan.projectionNames(grouped_scan)[0]));
    try std.testing.expectEqualStrings("sales", vectra.ArrowExport.ParquetScan.projectionNameAt(grouped_scan, 0).?);
    try std.testing.expect(vectra.ArrowExport.ParquetScan.projectionNameAt(grouped_scan, 2) == null);
    try std.testing.expectEqual(@as(?usize, 1), vectra.ArrowExport.ParquetScan.projectionIndex(grouped_scan, "units"));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.projectionContains(grouped_scan, "sales"));
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.projectionContains(grouped_scan, "missing"));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.projectionNamesUnique(grouped_scan));
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.hasDuplicateProjectionNames(grouped_scan));
    try std.testing.expectEqual(@as(usize, 0), vectra.ArrowExport.ParquetScan.duplicateProjectionNameCount(grouped_scan));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.hasAllProjectionNames(grouped_scan, &.{ "sales", "units" }));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.hasAnyProjectionName(grouped_scan, &.{ "missing", "units" }));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.projectsColumn(grouped_scan, "units"));
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.projectsColumn(grouped_scan, "missing"));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.hasPushdown(grouped_scan));
    try std.testing.expectEqual(@as(usize, 2), try vectra.ArrowExport.ParquetScan.columnCount(grouped_scan));
    try std.testing.expectEqual(table.height() * 2, try vectra.ArrowExport.ParquetScan.cellCount(grouped_scan));
    const projected_scan_shape = try vectra.ArrowExport.ParquetScan.shape(grouped_scan);
    try std.testing.expectEqual(table.height(), projected_scan_shape.rows);
    try std.testing.expectEqual(@as(usize, 2), projected_scan_shape.cols);
    try std.testing.expect(vectra.ArrowExport.ParquetScan.hasShape(grouped_scan, table.height(), 2));
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.shapeEquals(grouped_scan, table.height(), table.width()));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.sameHeight(grouped_scan, grouped_scan));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.sameWidth(grouped_scan, grouped_scan));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.sameShape(grouped_scan, grouped_scan));
    try std.testing.expectEqual(@as(usize, 2), try vectra.ArrowExport.ParquetScan.arrowFieldCount(grouped_scan));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.hasArrowField(grouped_scan, "units"));
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.hasArrowField(grouped_scan, "active"));
    try std.testing.expectEqual(@as(?usize, 1), try vectra.ArrowExport.ParquetScan.arrowFieldIndex(grouped_scan, "units"));
    const projected_scan_dtypes = try vectra.ArrowExport.ParquetScan.arrowFieldDTypes(grouped_scan, gpa);
    defer gpa.free(projected_scan_dtypes);
    try std.testing.expectEqualSlices(vectra.DeviceDType, &.{ .f64, .i64 }, projected_scan_dtypes);
    try std.testing.expectEqual(@as(usize, 2), try vectra.ArrowExport.ParquetScan.numericArrowFieldCount(grouped_scan));
    try std.testing.expectEqual(@as(usize, 1), try vectra.ArrowExport.ParquetScan.floatArrowFieldCount(grouped_scan));
    try std.testing.expectEqual(@as(usize, 1), try vectra.ArrowExport.ParquetScan.integerArrowFieldCount(grouped_scan));
    try std.testing.expectEqual(@as(usize, 0), try vectra.ArrowExport.ParquetScan.boolArrowFieldCount(grouped_scan));
    try std.testing.expectEqual(@as(usize, 1), try vectra.ArrowExport.ParquetScan.nullableArrowFieldCount(grouped_scan));
    try std.testing.expectEqual(@as(usize, 1), try vectra.ArrowExport.ParquetScan.nonNullableArrowFieldCount(grouped_scan));
    const projected_scan_null_counts = try vectra.ArrowExport.ParquetScan.arrowFieldNullCounts(grouped_scan, gpa);
    defer gpa.free(projected_scan_null_counts);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, projected_scan_null_counts);
    const projected_scan_valid_counts = try vectra.ArrowExport.ParquetScan.arrowFieldValidCounts(grouped_scan, gpa);
    defer gpa.free(projected_scan_valid_counts);
    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, projected_scan_valid_counts);
    try std.testing.expectEqual(@as(usize, 1), try vectra.ArrowExport.ParquetScan.arrowNullCount(grouped_scan));
    try std.testing.expectEqual(@as(usize, 5), try vectra.ArrowExport.ParquetScan.arrowValidCount(grouped_scan));
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 6.0), try vectra.ArrowExport.ParquetScan.arrowNullRatio(grouped_scan), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 6.0), try vectra.ArrowExport.ParquetScan.arrowValidRatio(grouped_scan), 1e-12);
    const projected_column_schemas = try vectra.ArrowExport.ParquetScan.arrowColumnSchemas(grouped_scan, gpa);
    defer gpa.free(projected_column_schemas);
    try std.testing.expectEqual(@as(usize, 2), projected_column_schemas.len);
    try std.testing.expectEqualStrings("sales", projected_column_schemas[0].name);
    try std.testing.expectEqual(vectra.DeviceDType.f64, projected_column_schemas[0].dtype);
    try std.testing.expectEqual(vectra.DeviceDType.i64, projected_column_schemas[1].dtype);
    try std.testing.expect(vectra.ArrowExport.ParquetScan.arrowSchemaEqualsSchemas(grouped_scan, projected_column_schemas));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.arrowSchemaEquals(grouped_scan, grouped_scan));
    var projected_scan_schema = try vectra.ArrowExport.ParquetScan.toArrowSchema(grouped_scan, gpa);
    defer projected_scan_schema.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 2), projected_scan_schema.fieldCount());
    try std.testing.expectEqualStrings("sales", projected_scan_schema.fields[0].name);
    try std.testing.expectEqualStrings("units", projected_scan_schema.fields[1].name);
    var explicit_scan_schema = try vectra.ArrowExport.ParquetScan.toArrowSchemaProjection(grouped_scan, gpa, &.{"active"});
    defer explicit_scan_schema.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 1), explicit_scan_schema.fieldCount());
    try std.testing.expectEqualStrings("active", explicit_scan_schema.fields[0].name);
    const explicit_scan_column_schemas = try vectra.ArrowExport.ParquetScan.Arrow.arrowColumnSchemasProjection(grouped_scan, gpa, &.{"active"});
    defer gpa.free(explicit_scan_column_schemas);
    try std.testing.expectEqual(@as(usize, 1), explicit_scan_column_schemas.len);
    try std.testing.expectEqualStrings("active", explicit_scan_column_schemas[0].name);
    try std.testing.expectEqual(vectra.DeviceDType.bool, explicit_scan_column_schemas[0].dtype);
    try std.testing.expectEqual(@as(usize, 0), explicit_scan_column_schemas[0].nullCount());
    try std.testing.expectEqual(@as(usize, 3), explicit_scan_column_schemas[0].validCount());
    const explicit_scan_schema_summary = try vectra.ArrowExport.ParquetScan.arrowSchemaSummaryProjection(grouped_scan, gpa, &.{"units"});
    defer gpa.free(explicit_scan_schema_summary);
    try std.testing.expectEqual(@as(usize, 1), explicit_scan_schema_summary.len);
    try std.testing.expectEqualStrings("units", explicit_scan_schema_summary[0].name);
    const explicit_scan_dtypes = try vectra.ArrowExport.ParquetScan.Arrow.arrowFieldDTypesProjection(grouped_scan, gpa, &.{ "active", "sales" });
    defer gpa.free(explicit_scan_dtypes);
    try std.testing.expectEqualSlices(vectra.DeviceDType, &.{ .bool, .f64 }, explicit_scan_dtypes);
    const explicit_scan_dtype_names = try vectra.ArrowExport.ParquetScan.arrowFieldDTypeNamesProjection(grouped_scan, gpa, &.{ "active", "sales" });
    defer gpa.free(explicit_scan_dtype_names);
    try std.testing.expectEqualStrings("bool", explicit_scan_dtype_names[0]);
    try std.testing.expectEqualStrings("f64", explicit_scan_dtype_names[1]);
    const explicit_scan_dtype_bytes = try vectra.ArrowExport.ParquetScan.arrowFieldDTypeByteSizesProjection(grouped_scan, gpa, &.{ "active", "sales" });
    defer gpa.free(explicit_scan_dtype_bytes);
    try std.testing.expectEqualSlices(usize, &.{ 1, 8 }, explicit_scan_dtype_bytes);
    const explicit_scan_dtype_bits = try vectra.ArrowExport.ParquetScan.Arrow.arrowFieldDTypeBitSizesProjection(grouped_scan, gpa, &.{ "active", "sales" });
    defer gpa.free(explicit_scan_dtype_bits);
    try std.testing.expectEqualSlices(usize, &.{ 8, 64 }, explicit_scan_dtype_bits);
    const explicit_scan_numeric = try vectra.ArrowExport.ParquetScan.arrowFieldDTypeClassMaskProjection(grouped_scan, gpa, &.{ "active", "sales" }, .numeric);
    defer gpa.free(explicit_scan_numeric);
    try std.testing.expectEqualSlices(bool, &.{ false, true }, explicit_scan_numeric);
    try std.testing.expectEqual(@as(usize, 1), try vectra.ArrowExport.ParquetScan.numericArrowFieldCountProjection(grouped_scan, &.{ "active", "sales" }));
    try std.testing.expectEqual(@as(usize, 1), try vectra.ArrowExport.ParquetScan.floatArrowFieldCountProjection(grouped_scan, &.{ "active", "sales" }));
    try std.testing.expectEqual(@as(usize, 0), try vectra.ArrowExport.ParquetScan.integerArrowFieldCountProjection(grouped_scan, &.{ "active", "sales" }));
    try std.testing.expectEqual(@as(usize, 1), try vectra.ArrowExport.ParquetScan.boolArrowFieldCountProjection(grouped_scan, &.{ "active", "sales" }));
    const explicit_scan_nullable = try vectra.ArrowExport.ParquetScan.arrowFieldNullableMaskProjection(grouped_scan, gpa, &.{ "units", "active" });
    defer gpa.free(explicit_scan_nullable);
    try std.testing.expectEqualSlices(bool, &.{ true, false }, explicit_scan_nullable);
    try std.testing.expectEqual(@as(usize, 1), try vectra.ArrowExport.ParquetScan.nullableArrowFieldCountProjection(grouped_scan, &.{ "units", "active" }));
    try std.testing.expectEqual(@as(usize, 1), try vectra.ArrowExport.ParquetScan.nonNullableArrowFieldCountProjection(grouped_scan, &.{ "units", "active" }));
    const explicit_scan_null_counts = try vectra.ArrowExport.ParquetScan.arrowFieldNullCountsProjection(grouped_scan, gpa, &.{ "units", "active" });
    defer gpa.free(explicit_scan_null_counts);
    try std.testing.expectEqualSlices(usize, &.{ 1, 0 }, explicit_scan_null_counts);
    const explicit_scan_valid_counts = try vectra.ArrowExport.ParquetScan.Arrow.arrowFieldValidCountsProjection(grouped_scan, gpa, &.{ "units", "active" });
    defer gpa.free(explicit_scan_valid_counts);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, explicit_scan_valid_counts);
    const explicit_scan_null_ratios = try vectra.ArrowExport.ParquetScan.arrowFieldNullRatiosProjection(grouped_scan, gpa, &.{ "units", "active" });
    defer gpa.free(explicit_scan_null_ratios);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), explicit_scan_null_ratios[0], 1e-12);
    const explicit_scan_valid_ratios = try vectra.ArrowExport.ParquetScan.arrowFieldValidRatiosProjection(grouped_scan, gpa, &.{ "units", "active" });
    defer gpa.free(explicit_scan_valid_ratios);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), explicit_scan_valid_ratios[0], 1e-12);
    try std.testing.expectEqual(@as(usize, 1), try vectra.ArrowExport.ParquetScan.arrowNullCountProjection(grouped_scan, &.{ "units", "active" }));
    try std.testing.expectEqual(@as(usize, 5), try vectra.ArrowExport.ParquetScan.arrowValidCountProjection(grouped_scan, &.{ "units", "active" }));
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 6.0), try vectra.ArrowExport.ParquetScan.arrowNullRatioProjection(grouped_scan, &.{ "units", "active" }), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 6.0), try vectra.ArrowExport.ParquetScan.arrowValidRatioProjection(grouped_scan, &.{ "units", "active" }), 1e-12);
    try std.testing.expectError(error.ColumnNotFound, vectra.ArrowExport.ParquetScan.toArrowSchemaProjection(grouped_scan, gpa, &.{"missing"}));
    try std.testing.expectError(error.ColumnNotFound, vectra.ArrowExport.ParquetScan.arrowColumnSchemasProjection(grouped_scan, gpa, &.{"missing"}));
    try std.testing.expectError(error.ColumnNotFound, vectra.ArrowExport.ParquetScan.arrowFieldDTypesProjection(grouped_scan, gpa, &.{"missing"}));
    try std.testing.expectError(error.ColumnNotFound, vectra.ArrowExport.ParquetScan.arrowFieldDTypeByteSizesProjection(grouped_scan, gpa, &.{"missing"}));
    try std.testing.expectError(error.ColumnNotFound, vectra.ArrowExport.ParquetScan.arrowFieldDTypeClassMaskProjection(grouped_scan, gpa, &.{"missing"}, .numeric));
    try std.testing.expectError(error.ColumnNotFound, vectra.ArrowExport.ParquetScan.arrowFieldNullableMaskProjection(grouped_scan, gpa, &.{"missing"}));
    try std.testing.expectError(error.ColumnNotFound, vectra.ArrowExport.ParquetScan.arrowFieldNullCountsProjection(grouped_scan, gpa, &.{"missing"}));
    var invalid_scan = try vectra.ArrowExport.ParquetScan.clone(grouped_scan);
    defer invalid_scan.deinit();
    vectra.ArrowExport.ParquetScan.clearPushdown(&invalid_scan);
    try vectra.ArrowExport.ParquetScan.select(&invalid_scan, &.{"missing"});
    try std.testing.expectError(error.ColumnNotFound, vectra.ArrowExport.ParquetScan.validateProjection(invalid_scan));
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.pushdownValid(invalid_scan));
    var duplicate_scan = try vectra.ArrowExport.ParquetScan.clone(grouped_scan);
    defer duplicate_scan.deinit();
    vectra.ArrowExport.ParquetScan.clearPushdown(&duplicate_scan);
    try vectra.ArrowExport.ParquetScan.select(&duplicate_scan, &.{ "sales", "sales" });
    try std.testing.expect(vectra.ArrowExport.ParquetScan.hasDuplicateProjectionNames(duplicate_scan));
    try std.testing.expectError(error.ColumnNotFound, vectra.ArrowExport.ParquetScan.validateProjection(duplicate_scan));
    const projected_scan_fields = try vectra.ArrowExport.ParquetScan.toArrowFields(grouped_scan, gpa);
    defer {
        for (projected_scan_fields) |*field| field.deinit(gpa);
        gpa.free(projected_scan_fields);
    }
    try std.testing.expectEqual(@as(usize, 2), projected_scan_fields.len);
    try std.testing.expectEqualStrings("units", projected_scan_fields[1].name);
    const explicit_scan_fields = try vectra.ArrowExport.ParquetScan.toArrowFieldsProjection(grouped_scan, gpa, &.{"sales"});
    defer {
        for (explicit_scan_fields) |*field| field.deinit(gpa);
        gpa.free(explicit_scan_fields);
    }
    try std.testing.expectEqual(@as(usize, 1), explicit_scan_fields.len);
    try std.testing.expectEqualStrings("sales", explicit_scan_fields[0].name);
    const projection_metadata_nbytes = 2 * @sizeOf([]const u8) + "sales".len + "units".len;
    try std.testing.expectEqual(@as(usize, projection_metadata_nbytes), vectra.ArrowExport.ParquetScan.projectionMetadataNbytes(grouped_scan));
    try vectra.ArrowExport.ParquetScan.Pushdown.whereMin(&grouped_scan, "sales", f64, 0.0);
    try vectra.ArrowExport.ParquetScan.validatePredicate(grouped_scan);
    try vectra.ArrowExport.ParquetScan.validatePushdown(grouped_scan);
    try vectra.ArrowExport.ParquetScan.Pushdown.validateCollect(grouped_scan);
    try std.testing.expect(vectra.ArrowExport.ParquetScan.pushdownValid(grouped_scan));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.collectValid(grouped_scan));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.hasPredicate(grouped_scan));
    try std.testing.expectEqualStrings("sales", vectra.ArrowExport.ParquetScan.predicateColumn(grouped_scan).?);
    try std.testing.expect(vectra.ArrowExport.ParquetScan.hasPredicateFor(grouped_scan, "sales"));
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.hasPredicateFor(grouped_scan, "units"));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.hasRangePredicate(grouped_scan));
    try std.testing.expect(std.mem.eql(u8, "sales", vectra.ArrowExport.ParquetScan.rangePredicateColumn(grouped_scan).?));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.hasRangePredicateFor(grouped_scan, "sales"));
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.hasRangePredicateFor(grouped_scan, "units"));
    try std.testing.expectEqual(vectra.DeviceDType.f64, vectra.ArrowExport.ParquetScan.rangePredicateDType(grouped_scan).?);
    try std.testing.expect(vectra.ArrowExport.ParquetScan.rangePredicate(grouped_scan).?.f64.min.? == 0.0);
    try vectra.ArrowExport.ParquetScan.whereMax(&grouped_scan, "sales", f64, 10.0);
    try std.testing.expect(vectra.ArrowExport.ParquetScan.rangePredicate(grouped_scan).?.f64.max.? == 10.0);
    try vectra.ArrowExport.ParquetScan.whereBetween(&grouped_scan, "sales", f64, 0.0, 10.0);
    try std.testing.expect(vectra.ArrowExport.ParquetScan.rangePredicate(grouped_scan).?.f64.min.? == 0.0);
    try std.testing.expect(vectra.ArrowExport.ParquetScan.rangePredicate(grouped_scan).?.f64.max.? == 10.0);
    try vectra.ArrowExport.ParquetScan.Pushdown.whereGe(&grouped_scan, "sales", f64, 0.0);
    try std.testing.expect(vectra.ArrowExport.ParquetScan.rangePredicate(grouped_scan).?.f64.min.? == 0.0);
    try vectra.ArrowExport.ParquetScan.whereLe(&grouped_scan, "sales", f64, 10.0);
    try std.testing.expect(vectra.ArrowExport.ParquetScan.rangePredicate(grouped_scan).?.f64.max.? == 10.0);
    try vectra.ArrowExport.ParquetScan.whereGt(&grouped_scan, "sales", f64, 1.0);
    try std.testing.expect(vectra.ArrowExport.ParquetScan.rangePredicate(grouped_scan).?.f64.min.? == 1.0);
    try vectra.ArrowExport.ParquetScan.whereLt(&grouped_scan, "sales", f64, 9.0);
    try std.testing.expect(vectra.ArrowExport.ParquetScan.rangePredicate(grouped_scan).?.f64.max.? == 9.0);
    try vectra.ArrowExport.ParquetScan.whereEq(&grouped_scan, "sales", f64, 2.0);
    try std.testing.expect(vectra.ArrowExport.ParquetScan.rangePredicate(grouped_scan).?.f64.min.? == 2.0);
    try std.testing.expect(vectra.ArrowExport.ParquetScan.rangePredicate(grouped_scan).?.f64.max.? == 2.0);
    try vectra.ArrowExport.ParquetScan.whereBool(&grouped_scan, "active", true);
    try std.testing.expectEqual(vectra.DeviceDType.bool, vectra.ArrowExport.ParquetScan.rangePredicateDType(grouped_scan).?);
    try std.testing.expect(vectra.ArrowExport.ParquetScan.rangePredicate(grouped_scan).?.bool.min.?);
    try std.testing.expect(vectra.ArrowExport.ParquetScan.rangePredicate(grouped_scan).?.bool.max.?);
    try vectra.ArrowExport.ParquetScan.whereEq(&grouped_scan, "sales", f64, 2.0);
    var wrong_dtype_scan = try vectra.ArrowExport.ParquetScan.clone(grouped_scan);
    defer wrong_dtype_scan.deinit();
    try vectra.ArrowExport.ParquetScan.whereRange(&wrong_dtype_scan, "sales", .{ .i64 = .{ .min = 0 } });
    try std.testing.expectError(error.TypeMismatch, vectra.ArrowExport.ParquetScan.validatePredicate(wrong_dtype_scan));
    var inverted_range_scan = try vectra.ArrowExport.ParquetScan.clone(grouped_scan);
    defer inverted_range_scan.deinit();
    try vectra.ArrowExport.ParquetScan.whereBetween(&inverted_range_scan, "sales", f64, 10.0, 0.0);
    try std.testing.expectError(error.TypeMismatch, vectra.ArrowExport.ParquetScan.validatePredicate(inverted_range_scan));
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.pushdownValid(wrong_dtype_scan));
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.Pushdown.pushdownValid(wrong_dtype_scan));
    try std.testing.expectError(error.TypeMismatch, vectra.ArrowExport.ParquetScan.validateCollect(wrong_dtype_scan));
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.collectValid(wrong_dtype_scan));
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.hasNullPredicate(grouped_scan));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.nullPredicateColumn(grouped_scan) == null);
    try std.testing.expect(vectra.ArrowExport.ParquetScan.nullPredicateWantNulls(grouped_scan) == null);
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.hasNullPredicateFor(grouped_scan, "sales"));
    try std.testing.expectEqual(@as(usize, "sales".len), vectra.ArrowExport.ParquetScan.rangePredicateMetadataNbytes(grouped_scan));
    try std.testing.expectEqual(@as(usize, 0), vectra.ArrowExport.ParquetScan.nullPredicateMetadataNbytes(grouped_scan));
    try std.testing.expectEqual(@as(usize, "sales".len), vectra.ArrowExport.ParquetScan.predicateMetadataNbytes(grouped_scan));
    try std.testing.expectEqual(@as(usize, projection_metadata_nbytes + "sales".len), vectra.ArrowExport.ParquetScan.pushdownMetadataNbytes(grouped_scan));
    try std.testing.expectEqual(grouped_parquet_bytes.len + projection_metadata_nbytes + "sales".len, vectra.ArrowExport.ParquetScan.ownedNbytes(grouped_scan));
    const range_summary: vectra.DeviceParquetScanPushdownSummary = vectra.ArrowExport.ParquetScan.pushdownSummary(grouped_scan);
    try std.testing.expect(range_summary.hasPushdown());
    try std.testing.expect(range_summary.isNonEmpty());
    try std.testing.expect(range_summary.hasProjection());
    try std.testing.expectEqual(@as(usize, 2), range_summary.projectionColumnCount());
    try std.testing.expectEqualStrings("sales", range_summary.projectionNameAt(0).?);
    try std.testing.expectEqual(@as(?usize, 1), range_summary.projectionIndex("units"));
    try std.testing.expect(range_summary.projectsColumn("sales"));
    try std.testing.expect(!range_summary.projectsColumn("missing"));
    try std.testing.expect(range_summary.hasPredicateFor("sales"));
    try std.testing.expect(range_summary.hasRangePredicateFor("sales"));
    try std.testing.expectEqual(vectra.DeviceDType.f64, range_summary.rangePredicateDType().?);
    try std.testing.expect(!range_summary.hasNullPredicate());
    try std.testing.expectEqual(vectra.ArrowExport.ParquetScan.pushdownMetadataNbytes(grouped_scan), range_summary.pushdownMetadataNbytes());
    const scan_summary: vectra.DeviceParquetScanSummary = vectra.ArrowExport.ParquetScan.summary(grouped_scan);
    try std.testing.expect(scan_summary.deviceValue().sameDevice(.cpu));
    try std.testing.expect(scan_summary.isCpu());
    try std.testing.expect(scan_summary.isHostBacked());
    try std.testing.expect(!scan_summary.isDeviceBacked());
    try std.testing.expect(scan_summary.isDeviceAvailable());
    try std.testing.expectEqual(grouped_parquet_bytes.len, scan_summary.sourceNbytes());
    try std.testing.expectEqual(vectra.ArrowExport.ParquetScan.sourcePtr(grouped_scan), scan_summary.sourcePtr());
    try std.testing.expectEqual(scan_summary.sourcePtr(), scan_summary.dataPtr());
    try std.testing.expect(scan_summary.hasSourcePtr());
    try std.testing.expectEqual(scan_summary.sourcePtr() + scan_summary.sourceNbytes(), scan_summary.sourceEndPtr());
    const scan_source_range = scan_summary.sourceRange();
    try std.testing.expectEqual(scan_summary.sourcePtr(), scan_source_range.ptr);
    try std.testing.expectEqual(scan_summary.sourceNbytes(), scan_source_range.nbytes);
    try std.testing.expectEqual(scan_summary.sourceEndPtr(), scan_source_range.endPtr());
    try std.testing.expect(scan_source_range.isNonEmpty());
    try std.testing.expect(scan_summary.sharesSource(scan_summary));
    try std.testing.expect(scan_summary.sameSource(scan_summary));
    try std.testing.expect(scan_summary.sharesStorage(scan_summary));
    try std.testing.expect(scan_summary.sameStorage(scan_summary));
    try std.testing.expect(scan_summary.mayOverlap(scan_summary));
    try std.testing.expectEqual(grouped_parquet_bytes.len, scan_summary.nbytes());
    try std.testing.expect(scan_summary.hasBytes());
    try std.testing.expect(scan_summary.isNonEmpty());
    try std.testing.expect(scan_summary.hasPushdown());
    try std.testing.expectEqual(vectra.ArrowExport.ParquetScan.ownedNbytes(grouped_scan), scan_summary.ownedNbytes());
    try std.testing.expectEqual(scan_summary.ownedNbytes(), scan_summary.memoryUsage());
    try std.testing.expect(scan_summary.pushdownSummary().hasRangePredicateFor("sales"));
    var scan_cpu_clone = try vectra.ArrowExport.ParquetScan.cpu(grouped_scan);
    defer scan_cpu_clone.deinit();
    try std.testing.expect(vectra.ArrowExport.ParquetScan.isCpu(scan_cpu_clone));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.sameDevice(grouped_scan, scan_cpu_clone));
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.sameStorage(grouped_scan, scan_cpu_clone));
    try std.testing.expectEqual(vectra.ArrowExport.ParquetScan.ownedNbytes(grouped_scan), vectra.ArrowExport.ParquetScan.ownedNbytes(scan_cpu_clone));
    var scan_with_cpu = try vectra.ArrowExport.ParquetScan.withDevice(grouped_scan, .cpu);
    defer scan_with_cpu.deinit();
    try std.testing.expect(vectra.ArrowExport.ParquetScan.isCpu(scan_with_cpu));
    try vectra.ArrowExport.ParquetScan.setDevice(&scan_with_cpu, .cpu);
    try std.testing.expect(vectra.ArrowExport.ParquetScan.deviceValue(scan_with_cpu).sameDevice(.cpu));
    try vectra.ArrowExport.ParquetScan.retarget(&scan_with_cpu, .cpu);
    try std.testing.expect(vectra.ArrowExport.ParquetScan.deviceValue(scan_with_cpu).sameDevice(.cpu));
    try vectra.ArrowExport.ParquetScan.Pushdown.whereIsNotNull(&grouped_scan, "units");
    try std.testing.expect(vectra.ArrowExport.ParquetScan.hasPredicate(grouped_scan));
    try std.testing.expectEqualStrings("units", vectra.ArrowExport.ParquetScan.predicateColumn(grouped_scan).?);
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.hasRangePredicate(grouped_scan));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.rangePredicateColumn(grouped_scan) == null);
    try std.testing.expect(vectra.ArrowExport.ParquetScan.rangePredicate(grouped_scan) == null);
    try std.testing.expect(vectra.ArrowExport.ParquetScan.rangePredicateDType(grouped_scan) == null);
    try std.testing.expect(vectra.ArrowExport.ParquetScan.hasNullPredicate(grouped_scan));
    try std.testing.expectEqualStrings("units", vectra.ArrowExport.ParquetScan.nullPredicateColumn(grouped_scan).?);
    try std.testing.expectEqual(@as(?bool, false), vectra.ArrowExport.ParquetScan.nullPredicateWantNulls(grouped_scan));
    try vectra.ArrowExport.ParquetScan.whereIsNull(&grouped_scan, "units");
    try std.testing.expectEqual(@as(?bool, true), vectra.ArrowExport.ParquetScan.nullPredicateWantNulls(grouped_scan));
    try vectra.ArrowExport.ParquetScan.whereNotNull(&grouped_scan, "units");
    try std.testing.expectEqual(@as(?bool, false), vectra.ArrowExport.ParquetScan.nullPredicateWantNulls(grouped_scan));
    try vectra.ArrowExport.ParquetScan.validatePredicate(grouped_scan);
    try std.testing.expect(vectra.ArrowExport.ParquetScan.hasNullPredicateFor(grouped_scan, "units"));
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.hasNullPredicateFor(grouped_scan, "sales"));
    var non_nullable_null_scan = try vectra.ArrowExport.ParquetScan.clone(grouped_scan);
    defer non_nullable_null_scan.deinit();
    try vectra.ArrowExport.ParquetScan.whereNull(&non_nullable_null_scan, "sales", true);
    try std.testing.expectError(error.TypeMismatch, vectra.ArrowExport.ParquetScan.validatePredicate(non_nullable_null_scan));
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.pushdownValid(non_nullable_null_scan));
    try std.testing.expectEqual(@as(usize, 0), vectra.ArrowExport.ParquetScan.rangePredicateMetadataNbytes(grouped_scan));
    try std.testing.expectEqual(@as(usize, "units".len), vectra.ArrowExport.ParquetScan.nullPredicateMetadataNbytes(grouped_scan));
    vectra.ArrowExport.ParquetScan.clearNullPredicate(&grouped_scan);
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.hasPredicate(grouped_scan));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.hasProjection(grouped_scan));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.hasPushdown(grouped_scan));
    try vectra.ArrowExport.ParquetScan.whereRange(&grouped_scan, "sales", .{ .f64 = .{ .min = 0.0 } });
    vectra.ArrowExport.ParquetScan.clearRangePredicate(&grouped_scan);
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.hasPredicate(grouped_scan));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.hasProjection(grouped_scan));
    try vectra.ArrowExport.ParquetScan.whereRange(&grouped_scan, "sales", .{ .f64 = .{ .min = 0.0 } });
    vectra.ArrowExport.ParquetScan.clearPredicate(&grouped_scan);
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.hasPredicate(grouped_scan));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.hasProjection(grouped_scan));
    try vectra.ArrowExport.ParquetScan.dropSelected(&grouped_scan, &.{"units"});
    try std.testing.expectEqual(@as(usize, 1), vectra.ArrowExport.ParquetScan.projectionColumnCount(grouped_scan));
    vectra.ArrowExport.ParquetScan.selectAll(&grouped_scan);
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.hasProjection(grouped_scan));
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.hasPushdown(grouped_scan));
    try vectra.ArrowExport.ParquetScan.selectExcept(&grouped_scan, &.{"active"});
    try std.testing.expectEqual(@as(usize, 2), vectra.ArrowExport.ParquetScan.projectionColumnCount(grouped_scan));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.projectionContains(grouped_scan, "sales"));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.projectionContains(grouped_scan, "units"));
    try vectra.ArrowExport.ParquetScan.select(&grouped_scan, &.{ "sales", "units" });
    try vectra.ArrowExport.ParquetScan.whereRange(&grouped_scan, "sales", .{ .f64 = .{ .min = 0.0 } });
    const grouped_scan_explain = try vectra.ArrowExport.ParquetScan.explain(grouped_scan, gpa);
    defer gpa.free(grouped_scan_explain);
    try std.testing.expect(std.mem.indexOf(u8, grouped_scan_explain, "pushdown") != null);
    const grouped_scan_summary_explain = try vectra.ArrowExport.ParquetScan.Lifecycle.explainSummary(grouped_scan, gpa);
    defer gpa.free(grouped_scan_summary_explain);
    try std.testing.expect(std.mem.indexOf(u8, grouped_scan_summary_explain, "rows=") != null);
    try std.testing.expect(std.mem.indexOf(u8, grouped_scan_summary_explain, "valid=true") != null);
    var grouped_scan_clone = try vectra.ArrowExport.ParquetScan.clone(grouped_scan);
    defer grouped_scan_clone.deinit();
    try std.testing.expect(vectra.ArrowExport.ParquetScan.sameDevice(grouped_scan, grouped_scan_clone));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.sharesSource(grouped_scan, grouped_scan));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.sameSource(grouped_scan, grouped_scan));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.sameStorage(grouped_scan, grouped_scan));
    try std.testing.expect(vectra.ArrowExport.ParquetScan.mayOverlap(grouped_scan, grouped_scan));
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.sharesSource(grouped_scan, grouped_scan_clone));
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.sameStorage(grouped_scan, grouped_scan_clone));
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.sourceMayOverlap(grouped_scan, grouped_scan_clone));
    try std.testing.expectEqual(vectra.ArrowExport.ParquetScan.ownedNbytes(grouped_scan), vectra.ArrowExport.ParquetScan.ownedNbytes(grouped_scan_clone));
    vectra.ArrowExport.ParquetScan.clearPushdown(&grouped_scan_clone);
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.hasPushdown(grouped_scan_clone));
    try std.testing.expectEqual(vectra.ArrowExport.ParquetScan.sourceNbytes(grouped_scan_clone), vectra.ArrowExport.ParquetScan.ownedNbytes(grouped_scan_clone));
    try vectra.ArrowExport.ParquetScan.select(&grouped_scan_clone, &.{"sales"});
    try vectra.ArrowExport.ParquetScan.whereNull(&grouped_scan_clone, "units", true);
    try std.testing.expect(vectra.ArrowExport.ParquetScan.hasPushdown(grouped_scan_clone));
    vectra.ArrowExport.ParquetScan.resetPushdown(&grouped_scan_clone);
    try std.testing.expect(!vectra.ArrowExport.ParquetScan.hasPushdown(grouped_scan_clone));
    var grouped_scan_rows = try vectra.ArrowExport.ParquetScan.collect(grouped_scan);
    defer grouped_scan_rows.deinit();
    try std.testing.expectEqual(table.height(), grouped_scan_rows.height());
    var grouped_scan_lazy = try vectra.ArrowExport.ParquetScan.lazy(grouped_scan);
    defer grouped_scan_lazy.deinit();
}

test "device dataframe preserves zero-column row count through boltha arrow" {
    const gpa = std.testing.allocator;

    var table = try DeviceDataFrame.initEmpty(gpa, 3, .cpu);
    defer table.deinit();

    var batch = try table.toArrowRecordBatch(gpa);
    defer batch.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 3), batch.row_count);
    try std.testing.expectEqual(@as(usize, 0), batch.columnCount());

    var arrow_table = try table.toArrowTable(gpa);
    defer arrow_table.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 3), arrow_table.row_count);
    try std.testing.expectEqual(@as(usize, 1), arrow_table.batchCount());
    try std.testing.expectEqual(@as(usize, 0), arrow_table.columnCount());

    var roundtrip = try DeviceDataFrame.fromArrowRecordBatch(gpa, batch, .cpu);
    defer roundtrip.deinit();
    try std.testing.expectEqual(@as(usize, 3), roundtrip.height());
    try std.testing.expectEqual(@as(usize, 0), roundtrip.width());
}

test "device dataframe round-trips Vectra extension dtypes through boltha arrow" {
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

    var schema = try table.toArrowSchema(gpa);
    defer schema.deinit(gpa);
    try std.testing.expect(schema.fields[0].data_type.eql(.{ .fixed_size_binary = 2 }));
    try std.testing.expect(schema.fields[1].data_type.eql(.{ .fixed_size_binary = 8 }));
    try std.testing.expect(schema.fields[2].data_type.eql(.{ .fixed_size_binary = 16 }));
    try std.testing.expectEqualStrings("vectra.bfloat16", schema.fields[0].extensionTypeName().?);
    try std.testing.expectEqualStrings("vectra.complex64", schema.fields[1].extensionTypeName().?);
    try std.testing.expectEqualStrings("vectra.complex128", schema.fields[2].extensionTypeName().?);

    var batch = try table.toArrowRecordBatch(gpa);
    defer batch.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 3), batch.row_count);
    try std.testing.expectEqual(@as(usize, 2), batch.columns[0].fixed_size_binary.byte_width);
    try std.testing.expectEqual(@as(usize, 8), batch.columns[1].fixed_size_binary.byte_width);
    try std.testing.expectEqual(@as(?[]const u8, null), batch.columns[1].fixed_size_binary.value(1));

    var restored = try DeviceDataFrame.fromArrowRecordBatch(gpa, batch, .cpu);
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
    var all_nan_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ std.math.nan(f64), std.math.nan(f64) }, .cpu);
    defer all_nan_metric.deinit();
    var all_nan_table = try DeviceDataFrame.init(gpa, &.{.{ .name = "metric", .data = all_nan_metric }});
    defer all_nan_table.deinit();
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
    try std.testing.expect(!try table.anyZeroColumn("sales"));
    try std.testing.expect(!try table.allZeroColumn("sales"));
    try std.testing.expect(try table.anyNonzeroColumn("sales"));
    try std.testing.expect(try table.allNonzeroColumn("sales"));
    try std.testing.expect(try table.anyNonZeroColumn("units"));
    try std.testing.expect(try table.allNonZeroColumn("units"));
    try std.testing.expect(try table.anyPositiveColumn("sales"));
    try std.testing.expect(try table.allPositiveColumn("sales"));
    try std.testing.expect(!try table.anyNegativeColumn("sales"));
    try std.testing.expect(!try table.allNegativeColumn("sales"));
    try std.testing.expect(try signed_zero_table.anyZeroColumn("metric"));
    try std.testing.expect(!try signed_zero_table.allZeroColumn("metric"));
    try std.testing.expect(try signed_zero_table.anyPositiveZeroColumn("metric"));
    try std.testing.expect(!try signed_zero_table.allPositiveZeroColumn("metric"));
    try std.testing.expect(try signed_zero_table.anyNegativeZeroColumn("metric"));
    try std.testing.expect(!try signed_zero_table.allNegativeZeroColumn("metric"));
    try std.testing.expect(try sign_metric_table.anySignBitColumn("metric"));
    try std.testing.expect(!try sign_metric_table.allSignBitColumn("metric"));
    try std.testing.expect(try sign_metric_table.anyNegativeColumn("metric"));
    try std.testing.expect(!try sign_metric_table.allNegativeColumn("metric"));
    try std.testing.expect(try nan_close_source.anyNanColumn("metric"));
    try std.testing.expect(try nan_close_source.anyNaNColumn("metric"));
    try std.testing.expect(!try nan_close_source.allNanColumn("metric"));
    try std.testing.expect(!try nan_close_source.allNaNColumn("metric"));
    try std.testing.expect(try all_nan_table.allNanColumn("metric"));
    try std.testing.expect(try all_nan_table.allNaNColumn("metric"));
    try std.testing.expect(try signed_inf_table.anyInfColumn("metric"));
    try std.testing.expect(!try signed_inf_table.allInfColumn("metric"));
    try std.testing.expect(try signed_inf_table.anyPositiveInfColumn("metric"));
    try std.testing.expect(!try signed_inf_table.allPositiveInfColumn("metric"));
    try std.testing.expect(try signed_inf_table.anyNegativeInfColumn("metric"));
    try std.testing.expect(!try signed_inf_table.allNegativeInfColumn("metric"));
    try std.testing.expect(try signed_inf_table.anyFiniteColumn("metric"));
    try std.testing.expect(!try signed_inf_table.allFiniteColumn("metric"));
    try std.testing.expect(try signed_inf_table.anyNonFiniteColumn("metric"));
    try std.testing.expect(!try signed_inf_table.allNonFiniteColumn("metric"));
    try std.testing.expect(try all_nan_table.allNonFiniteColumn("metric"));
    try std.testing.expect(try ieee_class_table.anyNormalColumn("metric"));
    try std.testing.expect(!try ieee_class_table.allNormalColumn("metric"));
    try std.testing.expect(try ieee_class_table.anySubnormalColumn("metric"));
    try std.testing.expect(!try ieee_class_table.allSubnormalColumn("metric"));
    try std.testing.expect(!try all_null_metric_table.anyFiniteColumn("metric"));
    try std.testing.expect(!try all_null_metric_table.allFiniteColumn("metric"));
    var filled_nan_expr = try nan_close_source.withColumnFillNaN("metric_no_nan", "metric", f64, -1.0);
    defer filled_nan_expr.deinit();
    const metric_no_nan = try (try filled_nan_expr.column("metric_no_nan")).f64.toOwnedSlice(gpa);
    defer gpa.free(metric_no_nan);
    try std.testing.expectEqual(@as(f64, -1.0), metric_no_nan[0]);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), metric_no_nan[1], 1e-12);
    var filled_inf_expr = try signed_inf_table.withColumnFillInf("metric_no_inf", "metric", f64, 7.0);
    defer filled_inf_expr.deinit();
    const metric_no_inf = try (try filled_inf_expr.column("metric_no_inf")).f64.toOwnedSlice(gpa);
    defer gpa.free(metric_no_inf);
    try std.testing.expectEqualSlices(f64, &.{ 7.0, 7.0, 1.0 }, metric_no_inf);
    var filled_pos_inf_expr = try signed_inf_table.withColumnFillPositiveInf("metric_no_pos_inf", "metric", f64, 8.0);
    defer filled_pos_inf_expr.deinit();
    const metric_no_pos_inf = try (try filled_pos_inf_expr.column("metric_no_pos_inf")).f64.toOwnedSlice(gpa);
    defer gpa.free(metric_no_pos_inf);
    try std.testing.expectEqual(@as(f64, 8.0), metric_no_pos_inf[0]);
    try std.testing.expect(std.math.isNegativeInf(metric_no_pos_inf[1]));
    var filled_neg_inf_expr = try signed_inf_table.withColumnFillNegativeInf("metric_no_neg_inf", "metric", f64, -8.0);
    defer filled_neg_inf_expr.deinit();
    const metric_no_neg_inf = try (try filled_neg_inf_expr.column("metric_no_neg_inf")).f64.toOwnedSlice(gpa);
    defer gpa.free(metric_no_neg_inf);
    try std.testing.expect(std.math.isPositiveInf(metric_no_neg_inf[0]));
    try std.testing.expectEqual(@as(f64, -8.0), metric_no_neg_inf[1]);
    var filled_zero_expr = try signed_zero_table.withColumnFillZero("metric_no_zero", "metric", f64, 9.0);
    defer filled_zero_expr.deinit();
    const metric_no_zero = try (try filled_zero_expr.column("metric_no_zero")).f64.toOwnedSlice(gpa);
    defer gpa.free(metric_no_zero);
    try std.testing.expectEqualSlices(f64, &.{ 9.0, 9.0, 1.0 }, metric_no_zero);
    var filled_pos_zero_expr = try signed_zero_table.withColumnFillPositiveZero("metric_no_poszero", "metric", f64, 5.0);
    defer filled_pos_zero_expr.deinit();
    const metric_no_poszero = try (try filled_pos_zero_expr.column("metric_no_poszero")).f64.toOwnedSlice(gpa);
    defer gpa.free(metric_no_poszero);
    try std.testing.expectEqual(@as(f64, 5.0), metric_no_poszero[0]);
    try std.testing.expectEqual(@as(f64, -0.0), metric_no_poszero[1]);
    var filled_neg_zero_expr = try signed_zero_table.withColumnFillNegativeZero("metric_no_negzero", "metric", f64, -5.0);
    defer filled_neg_zero_expr.deinit();
    const metric_no_negzero = try (try filled_neg_zero_expr.column("metric_no_negzero")).f64.toOwnedSlice(gpa);
    defer gpa.free(metric_no_negzero);
    try std.testing.expectEqual(@as(f64, 0.0), metric_no_negzero[0]);
    try std.testing.expectEqual(@as(f64, -5.0), metric_no_negzero[1]);
    var filled_nonzero_expr = try signed_zero_table.withColumnFillNonZero("metric_no_nonzero", "metric", f64, 11.0);
    defer filled_nonzero_expr.deinit();
    const metric_no_nonzero = try (try filled_nonzero_expr.column("metric_no_nonzero")).f64.toOwnedSlice(gpa);
    defer gpa.free(metric_no_nonzero);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, -0.0, 11.0 }, metric_no_nonzero);
    var filled_positive_expr = try sign_metric_table.withColumnFillPositive("metric_no_positive", "metric", f64, 12.0);
    defer filled_positive_expr.deinit();
    const metric_no_positive = try (try filled_positive_expr.column("metric_no_positive")).f64.toOwnedSlice(gpa);
    defer gpa.free(metric_no_positive);
    try std.testing.expectEqualSlices(f64, &.{ -2.0, 0.0, 12.0, -0.0 }, metric_no_positive);
    var filled_signbit_expr = try sign_metric_table.withColumnFillSignBit("metric_no_signbit", "metric", f64, 13.0);
    defer filled_signbit_expr.deinit();
    const metric_no_signbit = try (try filled_signbit_expr.column("metric_no_signbit")).f64.toOwnedSlice(gpa);
    defer gpa.free(metric_no_signbit);
    try std.testing.expectEqualSlices(f64, &.{ 13.0, 0.0, 3.0, 13.0 }, metric_no_signbit);
    var filled_negative_expr = try sign_metric_table.withColumnFillNegative("metric_no_negative", "metric", f64, 14.0);
    defer filled_negative_expr.deinit();
    const metric_no_negative = try (try filled_negative_expr.column("metric_no_negative")).f64.toOwnedSlice(gpa);
    defer gpa.free(metric_no_negative);
    try std.testing.expectEqualSlices(f64, &.{ 14.0, 0.0, 3.0, -0.0 }, metric_no_negative);
    var filled_finite_expr = try signed_inf_table.withColumnFillFinite("metric_no_finite", "metric", f64, 15.0);
    defer filled_finite_expr.deinit();
    const metric_no_finite = try (try filled_finite_expr.column("metric_no_finite")).f64.toOwnedSlice(gpa);
    defer gpa.free(metric_no_finite);
    try std.testing.expect(std.math.isPositiveInf(metric_no_finite[0]));
    try std.testing.expect(std.math.isNegativeInf(metric_no_finite[1]));
    try std.testing.expectEqual(@as(f64, 15.0), metric_no_finite[2]);
    var filled_normal_expr = try ieee_class_table.withColumnFillNormal("metric_no_normal", "metric", f64, 16.0);
    defer filled_normal_expr.deinit();
    const metric_no_normal = try (try filled_normal_expr.column("metric_no_normal")).f64.toOwnedSlice(gpa);
    defer gpa.free(metric_no_normal);
    try std.testing.expectEqual(@as(f64, 16.0), metric_no_normal[0]);
    try std.testing.expectApproxEqAbs(std.math.floatMin(f64) / 2.0, metric_no_normal[1], 0.0);
    var filled_subnormal_expr = try ieee_class_table.withColumnFillSubnormal("metric_no_subnormal", "metric", f64, 17.0);
    defer filled_subnormal_expr.deinit();
    const metric_no_subnormal = try (try filled_subnormal_expr.column("metric_no_subnormal")).f64.toOwnedSlice(gpa);
    defer gpa.free(metric_no_subnormal);
    try std.testing.expectApproxEqAbs(std.math.floatMin(f64), metric_no_subnormal[0], 0.0);
    try std.testing.expectEqual(@as(f64, 17.0), metric_no_subnormal[1]);
    var filled_nonfinite_expr = try signed_inf_table.withColumnFillNonFinite("metric_no_nonfinite", "metric", f64, 18.0);
    defer filled_nonfinite_expr.deinit();
    const metric_no_nonfinite = try (try filled_nonfinite_expr.column("metric_no_nonfinite")).f64.toOwnedSlice(gpa);
    defer gpa.free(metric_no_nonfinite);
    try std.testing.expectEqualSlices(f64, &.{ 18.0, 18.0, 1.0 }, metric_no_nonfinite);
    try std.testing.expectError(error.TypeUnsupported, table.withColumnFillNaN("bad_fill_nan", "units", f64, 0.0));
    var null_if_nan_expr = try nan_close_source.withColumnNullIfNaN("metric_nan_null", "metric");
    defer null_if_nan_expr.deinit();
    const metric_nan_null_validity = try (try null_if_nan_expr.column("metric_nan_null")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(metric_nan_null_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true }, metric_nan_null_validity);
    var null_if_inf_expr = try signed_inf_table.withColumnNullIfInf("metric_inf_null", "metric");
    defer null_if_inf_expr.deinit();
    const metric_inf_null_validity = try (try null_if_inf_expr.column("metric_inf_null")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(metric_inf_null_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true }, metric_inf_null_validity);
    var null_if_zero_expr = try signed_zero_table.withColumnNullIfZero("metric_zero_null", "metric");
    defer null_if_zero_expr.deinit();
    const metric_zero_null_validity = try (try null_if_zero_expr.column("metric_zero_null")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(metric_zero_null_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true }, metric_zero_null_validity);
    var null_if_signbit_expr = try sign_metric_table.withColumnNullIfSignBit("metric_signbit_null", "metric");
    defer null_if_signbit_expr.deinit();
    const metric_signbit_null_validity = try (try null_if_signbit_expr.column("metric_signbit_null")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(metric_signbit_null_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, false }, metric_signbit_null_validity);
    var null_if_finite_expr = try ieee_class_table.withColumnNullIfFinite("metric_finite_null", "metric");
    defer null_if_finite_expr.deinit();
    const metric_finite_null_validity = try (try null_if_finite_expr.column("metric_finite_null")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(metric_finite_null_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, true }, metric_finite_null_validity);
    var null_if_nonfinite_expr = try signed_inf_table.withColumnNullIfNonFinite("metric_nonfinite_null", "metric");
    defer null_if_nonfinite_expr.deinit();
    const metric_nonfinite_null_validity = try (try null_if_nonfinite_expr.column("metric_nonfinite_null")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(metric_nonfinite_null_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true }, metric_nonfinite_null_validity);
    try std.testing.expectError(error.ColumnNotFound, table.nanCountColumn("missing"));
    try std.testing.expectError(error.ColumnNotFound, table.anyZeroColumn("missing"));
    try std.testing.expectEqual(@as(usize, 0), try table.nullCountColumn("sales"));
    try std.testing.expectEqual(@as(usize, 3), try table.validCountColumn("sales"));
    try std.testing.expect(!try table.anyNullColumn("sales"));
    try std.testing.expect(!try table.allNullColumn("sales"));
    try std.testing.expect(try table.anyValidColumn("sales"));
    try std.testing.expect(try table.allValidColumn("sales"));
    try std.testing.expectEqual(@as(?usize, 0), try table.firstValidIndexColumn("sales"));
    try std.testing.expectEqual(@as(?usize, 2), try table.lastValidIndexColumn("sales"));
    try std.testing.expectEqual(@as(?usize, null), try table.firstNullIndexColumn("sales"));
    try std.testing.expectEqual(@as(?usize, null), try table.lastNullIndexColumn("sales"));
    try std.testing.expectEqual(@as(usize, 1), try table.nullCountColumn("units"));
    try std.testing.expectEqual(@as(usize, 2), try table.validCountColumn("units"));
    try std.testing.expect(try table.anyNullColumn("units"));
    try std.testing.expect(!try table.allNullColumn("units"));
    try std.testing.expect(try table.anyValidColumn("units"));
    try std.testing.expect(!try table.allValidColumn("units"));
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
    try std.testing.expect(try nullable_sales_table.anyNullColumn("metric"));
    try std.testing.expect(!try nullable_sales_table.allNullColumn("metric"));
    try std.testing.expect(try nullable_sales_table.anyValidColumn("metric"));
    try std.testing.expect(!try nullable_sales_table.allValidColumn("metric"));
    try std.testing.expect(try all_null_metric_table.anyNullColumn("metric"));
    try std.testing.expect(try all_null_metric_table.allNullColumn("metric"));
    try std.testing.expect(!try all_null_metric_table.anyValidColumn("metric"));
    try std.testing.expect(!try all_null_metric_table.allValidColumn("metric"));
    try std.testing.expectEqual(@as(?usize, null), try all_null_metric_table.firstValidIndexColumn("metric"));
    try std.testing.expectEqual(@as(?usize, null), try all_null_metric_table.lastValidIndexColumn("metric"));
    try std.testing.expectEqual(@as(?usize, 0), try all_null_metric_table.firstNullIndexColumn("metric"));
    try std.testing.expectEqual(@as(?usize, 1), try all_null_metric_table.lastNullIndexColumn("metric"));
    try std.testing.expectEqual(DeviceScalar{ .f64 = 1.0 }, try all_null_metric_table.nullRatioColumn("metric"));
    var empty_metric = try DeviceColumn.fromSlice(f64, gpa, &.{}, .cpu);
    defer empty_metric.deinit();
    var empty_metric_table = try DeviceDataFrame.init(gpa, &.{.{ .name = "metric", .data = empty_metric }});
    defer empty_metric_table.deinit();
    try std.testing.expect(!try empty_metric_table.anyNullColumn("metric"));
    try std.testing.expect(try empty_metric_table.allNullColumn("metric"));
    try std.testing.expect(!try empty_metric_table.anyValidColumn("metric"));
    try std.testing.expect(try empty_metric_table.allValidColumn("metric"));
    try std.testing.expectError(error.ColumnNotFound, table.nullCountColumn("missing"));
    try std.testing.expectError(error.ColumnNotFound, table.anyNullColumn("missing"));
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
    try std.testing.expect(try rounding_type_table.anyTrueColumn("active"));
    try std.testing.expect(!try rounding_type_table.allTrueColumn("active"));
    try std.testing.expect(try rounding_type_table.anyFalseColumn("active"));
    try std.testing.expect(!try rounding_type_table.allFalseColumn("active"));
    var active_not_column = try rounding_type_table.logicalNotColumn("active");
    defer active_not_column.deinit();
    const active_not_values = try active_not_column.bool.toOwnedSlice(gpa);
    defer gpa.free(active_not_values);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false }, active_not_values);
    var active_not_table = try rounding_type_table.withColumnLogicalNot("active_not", "active");
    defer active_not_table.deinit();
    const active_not = try (try active_not_table.column("active_not")).bool.toOwnedSlice(gpa);
    defer gpa.free(active_not);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false }, active_not);
    var active_not_alias_table = try rounding_type_table.withColumnNot("active_not_alias", "active");
    defer active_not_alias_table.deinit();
    const active_not_alias = try (try active_not_alias_table.column("active_not_alias")).bool.toOwnedSlice(gpa);
    defer gpa.free(active_not_alias);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false }, active_not_alias);
    try std.testing.expectEqual(@as(usize, 2), try rounding_type_table.countTrueColumn("active"));
    try std.testing.expectEqual(@as(usize, 1), try rounding_type_table.countFalseColumn("active"));
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), (try rounding_type_table.trueRatioColumn("active")).f64, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), (try rounding_type_table.falseRatioColumn("active")).f64, 1e-12);
    try std.testing.expectEqual(@as(?usize, 0), try rounding_type_table.firstTrueIndexColumn("active"));
    try std.testing.expectEqual(@as(?usize, 2), try rounding_type_table.lastTrueIndexColumn("active"));
    try std.testing.expectEqual(@as(?usize, 1), try rounding_type_table.firstFalseIndexColumn("active"));
    try std.testing.expectEqual(@as(?usize, 1), try rounding_type_table.lastFalseIndexColumn("active"));
    try std.testing.expectError(error.TypeUnsupported, table.anyColumn("sales"));
    try std.testing.expectError(error.TypeUnsupported, table.withColumnLogicalNot("bad_not", "sales"));
    try std.testing.expectError(error.TypeUnsupported, table.anyFalseColumn("sales"));
    try std.testing.expectError(error.TypeUnsupported, table.trueRatioColumn("sales"));
    try std.testing.expectError(error.TypeUnsupported, table.firstTrueIndexColumn("sales"));

    var nullable_bool = try DeviceColumn.fromSliceWithValidity(bool, gpa, &.{ false, true, false }, &.{ true, false, true }, .cpu);
    defer nullable_bool.deinit();
    var nullable_bool_table = try DeviceDataFrame.init(gpa, &.{.{ .name = "flag", .data = nullable_bool }});
    defer nullable_bool_table.deinit();
    try std.testing.expect(!try nullable_bool_table.anyColumn("flag"));
    try std.testing.expect(!try nullable_bool_table.allColumn("flag"));
    try std.testing.expect(!try nullable_bool_table.anyTrueColumn("flag"));
    try std.testing.expect(!try nullable_bool_table.allTrueColumn("flag"));
    try std.testing.expect(try nullable_bool_table.anyFalseColumn("flag"));
    try std.testing.expect(try nullable_bool_table.allFalseColumn("flag"));
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
    try std.testing.expect(!try all_null_bool_table.anyTrueColumn("flag"));
    try std.testing.expect(!try all_null_bool_table.allTrueColumn("flag"));
    try std.testing.expect(!try all_null_bool_table.anyFalseColumn("flag"));
    try std.testing.expect(!try all_null_bool_table.allFalseColumn("flag"));
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

    var metric_isin_values_table = try where_table.withColumnIsInValues("metric_isin_values", "metric", f64, &.{ 2.0, 5.0 });
    defer metric_isin_values_table.deinit();
    const metric_isin_values = try (try metric_isin_values_table.column("metric_isin_values")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_isin_values);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true }, metric_isin_values);

    var metric_notin_values_table = try where_table.withColumnIsInValuesInverted("metric_notin_values", "metric", f64, &.{ 2.0, 5.0 });
    defer metric_notin_values_table.deinit();
    const metric_notin_values_column = try (try metric_notin_values_table.column("metric_notin_values")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_notin_values_column);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false }, metric_notin_values_column);

    var metric_isin_filtered = try where_table.filterIsInColumn("metric", "needles");
    defer metric_isin_filtered.deinit();
    const metric_isin_filtered_values = try (try metric_isin_filtered.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(metric_isin_filtered_values);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 5.0 }, metric_isin_filtered_values);

    var metric_notin_filtered = try where_table.filterNotInColumn("metric", "needles");
    defer metric_notin_filtered.deinit();
    const metric_notin_filtered_values = try (try metric_notin_filtered.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(metric_notin_filtered_values);
    try std.testing.expectEqualSlices(f64, &.{-1.0}, metric_notin_filtered_values);

    var metric_values_filtered = try where_table.filterIsInValues("metric", f64, &.{ 2.0, 5.0 });
    defer metric_values_filtered.deinit();
    const metric_values_filtered_values = try (try metric_values_filtered.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(metric_values_filtered_values);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 5.0 }, metric_values_filtered_values);

    var metric_values_notin = try where_table.filterNotInValues("metric", f64, &.{ 2.0, 5.0 });
    defer metric_values_notin.deinit();
    const metric_values_notin_values = try (try metric_values_notin.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(metric_values_notin_values);
    try std.testing.expectEqualSlices(f64, &.{-1.0}, metric_values_notin_values);

    var metric_drop_values = try where_table.dropIsInValues("metric", f64, &.{ 2.0, 5.0 });
    defer metric_drop_values.deinit();
    const metric_drop_values_values = try (try metric_drop_values.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(metric_drop_values_values);
    try std.testing.expectEqualSlices(f64, &.{-1.0}, metric_drop_values_values);

    var metric_drop_not_values = try where_table.dropNotInValues("metric", f64, &.{ 2.0, 5.0 });
    defer metric_drop_not_values.deinit();
    const metric_drop_not_values_values = try (try metric_drop_not_values.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(metric_drop_not_values_values);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 5.0 }, metric_drop_not_values_values);

    var metric_drop_isin = try where_table.dropIsInColumn("metric", "needles");
    defer metric_drop_isin.deinit();
    const metric_drop_isin_values = try (try metric_drop_isin.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(metric_drop_isin_values);
    try std.testing.expectEqualSlices(f64, &.{-1.0}, metric_drop_isin_values);

    var metric_drop_notin = try where_table.dropNotInColumn("metric", "needles");
    defer metric_drop_notin.deinit();
    const metric_drop_notin_values = try (try metric_drop_notin.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(metric_drop_notin_values);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 5.0 }, metric_drop_notin_values);
    try std.testing.expectError(error.TypeUnsupported, where_table.filterIsInColumn("metric", "mask"));
    try std.testing.expectError(error.TypeUnsupported, where_table.filterIsInValues("metric", bool, &.{true}));

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

    var ratio_between_table = try inverse_trig_table.withColumnBetween("ratio_between", "ratio", f64, -0.5, 0.0);
    defer ratio_between_table.deinit();
    const ratio_between = try (try ratio_between_table.column("ratio_between")).bool.toOwnedSlice(gpa);
    defer gpa.free(ratio_between);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false }, ratio_between);
    var ratio_between_exclusive_table = try inverse_trig_table.withColumnBetweenExclusive("ratio_between_exclusive", "ratio", f64, -0.5, 0.5);
    defer ratio_between_exclusive_table.deinit();
    const ratio_between_exclusive = try (try ratio_between_exclusive_table.column("ratio_between_exclusive")).bool.toOwnedSlice(gpa);
    defer gpa.free(ratio_between_exclusive);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false }, ratio_between_exclusive);
    var ratio_between_left_table = try inverse_trig_table.withColumnBetweenLeftClosed("ratio_between_left", "ratio", f64, -0.5, 0.5);
    defer ratio_between_left_table.deinit();
    const ratio_between_left = try (try ratio_between_left_table.column("ratio_between_left")).bool.toOwnedSlice(gpa);
    defer gpa.free(ratio_between_left);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false }, ratio_between_left);
    var ratio_between_right_table = try inverse_trig_table.withColumnBetweenRightClosed("ratio_between_right", "ratio", f64, -0.5, 0.5);
    defer ratio_between_right_table.deinit();
    const ratio_between_right = try (try ratio_between_right_table.column("ratio_between_right")).bool.toOwnedSlice(gpa);
    defer gpa.free(ratio_between_right);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true }, ratio_between_right);
    var ratio_not_between_table = try inverse_trig_table.withColumnNotBetween("ratio_not_between", "ratio", f64, -0.5, 0.0);
    defer ratio_not_between_table.deinit();
    const ratio_not_between = try (try ratio_not_between_table.column("ratio_not_between")).bool.toOwnedSlice(gpa);
    defer gpa.free(ratio_not_between);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true }, ratio_not_between);
    var ratio_outside_table = try inverse_trig_table.withColumnOutside("ratio_outside", "ratio", f64, -0.5, 0.5);
    defer ratio_outside_table.deinit();
    const ratio_outside = try (try ratio_outside_table.column("ratio_outside")).bool.toOwnedSlice(gpa);
    defer gpa.free(ratio_outside);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false }, ratio_outside);
    var ratio_not_between_open_table = try inverse_trig_table.withColumnNotBetweenExclusive("ratio_not_between_open", "ratio", f64, -0.5, 0.5);
    defer ratio_not_between_open_table.deinit();
    const ratio_not_between_open = try (try ratio_not_between_open_table.column("ratio_not_between_open")).bool.toOwnedSlice(gpa);
    defer gpa.free(ratio_not_between_open);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, ratio_not_between_open);
    try std.testing.expectError(error.TypeUnsupported, rounding_type_table.withColumnBetween("bad_between", "active", f64, -0.25, 0.25));
    try std.testing.expectError(error.ColumnNotFound, inverse_trig_table.withColumnBetween("missing_between", "missing", f64, -0.25, 0.25));

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

    var scalar_filtered = try table.filterColumnScalar("sales", f64, 2.5, .gt);
    defer scalar_filtered.deinit();
    const scalar_filtered_sales = try (try scalar_filtered.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(scalar_filtered_sales);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 5.0 }, scalar_filtered_sales);

    var scalar_dropped = try table.dropColumnScalar("sales", f64, 2.5, .gt);
    defer scalar_dropped.deinit();
    const scalar_dropped_sales = try (try scalar_dropped.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(scalar_dropped_sales);
    try std.testing.expectEqualSlices(f64, &.{2.0}, scalar_dropped_sales);
    try std.testing.expectError(error.TypeUnsupported, table.filterColumnScalar("units", f64, 0.0, .gt));

    var between_filtered = try table.filterBetweenColumn("sales", f64, 3.0, 5.0);
    defer between_filtered.deinit();
    const between_filtered_sales = try (try between_filtered.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(between_filtered_sales);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 5.0 }, between_filtered_sales);

    var outside_filtered = try table.filterOutsideColumn("sales", f64, 3.0, 5.0);
    defer outside_filtered.deinit();
    const outside_filtered_sales = try (try outside_filtered.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(outside_filtered_sales);
    try std.testing.expectEqualSlices(f64, &.{2.0}, outside_filtered_sales);

    var drop_between = try table.dropBetweenColumn("sales", f64, 3.0, 5.0);
    defer drop_between.deinit();
    const drop_between_sales = try (try drop_between.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(drop_between_sales);
    try std.testing.expectEqualSlices(f64, &.{2.0}, drop_between_sales);

    var drop_outside = try table.dropOutsideColumn("sales", f64, 3.0, 5.0);
    defer drop_outside.deinit();
    const drop_outside_sales = try (try drop_outside.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(drop_outside_sales);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 5.0 }, drop_outside_sales);
    try std.testing.expectError(error.TypeUnsupported, table.filterBetweenColumn("units", f64, 0.0, 1.0));

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
