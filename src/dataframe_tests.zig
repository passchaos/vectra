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
}

test "device dataframe sorts by device column keys" {
    const gpa = std.testing.allocator;

    var score = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 3.0, 1.0, 2.0, 4.0 }, &.{ true, true, false, true }, .cpu);
    defer score.deinit();
    var id = try DeviceColumn.fromSlice(i64, gpa, &.{ 30, 10, 20, 40 }, .cpu);
    defer id.deinit();
    var flag = try DeviceColumn.fromSlice(bool, gpa, &.{ true, false, true, false }, .cpu);
    defer flag.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "score", .data = score },
        .{ .name = "id", .data = id },
        .{ .name = "flag", .data = flag },
    });
    defer table.deinit();

    const asc = try table.argsortBy("score", .{ .descending = false, .nulls = .last });
    defer gpa.free(asc);
    try std.testing.expectEqualSlices(usize, &.{ 1, 0, 3, 2 }, asc);

    var sorted = try table.sortBy("score", .{ .descending = false, .nulls = .last });
    defer sorted.deinit();
    const sorted_id = try sorted.column("id");
    const sorted_id_values = try sorted_id.i64.toOwnedSlice(gpa);
    defer gpa.free(sorted_id_values);
    try std.testing.expectEqualSlices(i64, &.{ 10, 30, 40, 20 }, sorted_id_values);

    var desc_nulls_first = try table.sortBy("score", .{ .descending = true, .nulls = .first });
    defer desc_nulls_first.deinit();
    const desc_id = try desc_nulls_first.column("id");
    const desc_id_values = try desc_id.i64.toOwnedSlice(gpa);
    defer gpa.free(desc_id_values);
    try std.testing.expectEqualSlices(i64, &.{ 20, 40, 30, 10 }, desc_id_values);

    var bool_sorted = try table.sortBy("flag", .{});
    defer bool_sorted.deinit();
    const bool_sorted_id = try bool_sorted.column("id");
    const bool_sorted_id_values = try bool_sorted_id.i64.toOwnedSlice(gpa);
    defer gpa.free(bool_sorted_id_values);
    try std.testing.expectEqualSlices(i64, &.{ 10, 40, 30, 20 }, bool_sorted_id_values);

    var tied_score = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 10.0, 20.0, 20.0, 30.0, 0.0 }, &.{ true, true, true, true, false }, .cpu);
    defer tied_score.deinit();
    var tied_id = try DeviceColumn.fromSlice(i64, gpa, &.{ 1, 2, 3, 4, 5 }, .cpu);
    defer tied_id.deinit();
    var tied_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "score", .data = tied_score },
        .{ .name = "id", .data = tied_id },
    });
    defer tied_table.deinit();

    var ranks = try tied_table.rankProfileBy("score", "score", .{ .descending = false, .nulls = .last });
    defer ranks.deinit();
    try std.testing.expectEqual(@as(usize, 7), ranks.width());
    const ordinal = try (try ranks.column("score_ordinal_rank")).i64.toOwnedSlice(gpa);
    defer gpa.free(ordinal);
    const competition = try (try ranks.column("score_competition_rank")).i64.toOwnedSlice(gpa);
    defer gpa.free(competition);
    const dense_rank = try (try ranks.column("score_dense_rank")).i64.toOwnedSlice(gpa);
    defer gpa.free(dense_rank);
    const percent_rank = try (try ranks.column("score_percent_rank")).f64.toOwnedSlice(gpa);
    defer gpa.free(percent_rank);
    const cume_dist = try (try ranks.column("score_cume_dist")).f64.toOwnedSlice(gpa);
    defer gpa.free(cume_dist);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4, 5 }, ordinal);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 2, 4, 5 }, competition);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 2, 3, 4 }, dense_rank);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), percent_rank[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), percent_rank[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), percent_rank[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), percent_rank[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), percent_rank[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.2), cume_dist[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.6), cume_dist[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.6), cume_dist[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.8), cume_dist[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), cume_dist[4], 1e-12);

    var desc_ranks = try tied_table.rankProfileBy("score", "score_desc", .{ .descending = true, .nulls = .first });
    defer desc_ranks.deinit();
    const desc_competition = try (try desc_ranks.column("score_desc_competition_rank")).i64.toOwnedSlice(gpa);
    defer gpa.free(desc_competition);
    const desc_cume_dist = try (try desc_ranks.column("score_desc_cume_dist")).f64.toOwnedSlice(gpa);
    defer gpa.free(desc_cume_dist);
    try std.testing.expectEqualSlices(i64, &.{ 5, 3, 3, 2, 1 }, desc_competition);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), desc_cume_dist[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.8), desc_cume_dist[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.8), desc_cume_dist[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.4), desc_cume_dist[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.2), desc_cume_dist[4], 1e-12);

    var rolling_ranks = try tied_table.rollingRankProfile("score", "score_roll", .{ .window = 3, .min_periods = 2 });
    defer rolling_ranks.deinit();
    try std.testing.expectEqual(@as(usize, 6), rolling_ranks.width());
    const rolling_rank_count = try (try rolling_ranks.column("score_roll_rolling_rank_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(rolling_rank_count);
    const rolling_rank = try (try rolling_ranks.column("score_roll_rolling_rank")).i64.toOwnedSlice(gpa);
    defer gpa.free(rolling_rank);
    const rolling_percent_rank = try (try rolling_ranks.column("score_roll_rolling_percent_rank")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_percent_rank);
    const rolling_rank_cume = try (try rolling_ranks.column("score_roll_rolling_cume_dist")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_rank_cume);
    const rolling_rank_validity = try (try rolling_ranks.column("score_roll_rolling_rank")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(rolling_rank_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 3, 2 }, rolling_rank_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, false }, rolling_rank_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 2, 2, 3, 0 }, rolling_rank);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_percent_rank[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_percent_rank[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_percent_rank[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_rank_cume[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_rank_cume[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_rank_cume[3], 1e-12);

    var expanding_ranks = try tied_table.expandingRankProfile("score", "score_expand", .{ .min_periods = 2 });
    defer expanding_ranks.deinit();
    try std.testing.expectEqual(@as(usize, 6), expanding_ranks.width());
    const expanding_rank_count = try (try expanding_ranks.column("score_expand_expanding_rank_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(expanding_rank_count);
    const expanding_rank = try (try expanding_ranks.column("score_expand_expanding_rank")).i64.toOwnedSlice(gpa);
    defer gpa.free(expanding_rank);
    const expanding_percent_rank = try (try expanding_ranks.column("score_expand_expanding_percent_rank")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_percent_rank);
    const expanding_rank_cume = try (try expanding_ranks.column("score_expand_expanding_cume_dist")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_rank_cume);
    const expanding_rank_validity = try (try expanding_ranks.column("score_expand_expanding_rank")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(expanding_rank_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4, 4 }, expanding_rank_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, false }, expanding_rank_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 2, 2, 4, 0 }, expanding_rank);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), expanding_percent_rank[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_percent_rank[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), expanding_percent_rank[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), expanding_rank_cume[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), expanding_rank_cume[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), expanding_rank_cume[3], 1e-12);

    var rolling_sales = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, 2.0, 100.0, 4.0, 5.0 }, &.{ true, true, false, true, true }, .cpu);
    defer rolling_sales.deinit();
    var rolling_id = try DeviceColumn.fromSlice(i64, gpa, &.{ 1, 2, 3, 4, 5 }, .cpu);
    defer rolling_id.deinit();
    var rolling_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "sales", .data = rolling_sales },
        .{ .name = "id", .data = rolling_id },
    });
    defer rolling_table.deinit();

    var rolling = try rolling_table.rollingProfile("sales", "sales", .{ .window = 3, .min_periods = 2 });
    defer rolling.deinit();
    try std.testing.expectEqual(@as(usize, 7), rolling.width());
    const rolling_count = try (try rolling.column("sales_rolling_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(rolling_count);
    const rolling_sum = try (try rolling.column("sales_rolling_sum")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_sum);
    const rolling_mean = try (try rolling.column("sales_rolling_mean")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_mean);
    const rolling_variance = try (try rolling.column("sales_rolling_variance")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_variance);
    const rolling_stddev = try (try rolling.column("sales_rolling_stddev")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_stddev);
    const rolling_validity = try (try rolling.column("sales_rolling_mean")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(rolling_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 2, 2, 2 }, rolling_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true }, rolling_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), rolling_sum[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), rolling_sum[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 6.0), rolling_sum[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0), rolling_sum[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), rolling_mean[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), rolling_mean[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), rolling_mean[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.5), rolling_mean[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), rolling_variance[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), rolling_variance[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_variance[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), rolling_variance[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_stddev[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_stddev[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_stddev[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_stddev[4], 1e-12);

    var rolling_moments = try rolling_table.rollingMomentProfile("sales", "sales", .{ .window = 3, .min_periods = 2 });
    defer rolling_moments.deinit();
    try std.testing.expectEqual(@as(usize, 7), rolling_moments.width());
    const rolling_moment_count = try (try rolling_moments.column("sales_rolling_moment_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(rolling_moment_count);
    const rolling_m3 = try (try rolling_moments.column("sales_rolling_m3")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_m3);
    const rolling_m4 = try (try rolling_moments.column("sales_rolling_m4")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_m4);
    const rolling_skewness = try (try rolling_moments.column("sales_rolling_skewness")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_skewness);
    const rolling_kurtosis = try (try rolling_moments.column("sales_rolling_kurtosis")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_kurtosis);
    const rolling_moment_validity = try (try rolling_moments.column("sales_rolling_skewness")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(rolling_moment_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 2, 2, 2 }, rolling_moment_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true }, rolling_moment_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), rolling_m3[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), rolling_m3[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0625), rolling_m4[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_m4[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), rolling_skewness[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), rolling_skewness[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -2.0), rolling_kurtosis[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -2.0), rolling_kurtosis[3], 1e-12);

    var rolling_robust = try rolling_table.rollingRobustProfile("sales", "sales", .{ .window = 3, .min_periods = 2 });
    defer rolling_robust.deinit();
    try std.testing.expectEqual(@as(usize, 6), rolling_robust.width());
    const rolling_median_centered = try (try rolling_robust.column("sales_rolling_median_centered")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_median_centered);
    const rolling_mad_zscore = try (try rolling_robust.column("sales_rolling_mad_zscore")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_mad_zscore);
    const rolling_iqr_outlier = try (try rolling_robust.column("sales_rolling_iqr_outlier")).bool.toOwnedSlice(gpa);
    defer gpa.free(rolling_iqr_outlier);
    const rolling_winsorized = try (try rolling_robust.column("sales_rolling_winsorized")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_winsorized);
    const rolling_robust_validity = try (try rolling_robust.column("sales_rolling_median_centered")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(rolling_robust_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true, true }, rolling_robust_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_median_centered[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_median_centered[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_median_centered[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.6744897501960817), rolling_mad_zscore[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.6744897501960817), rolling_mad_zscore[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.6744897501960817), rolling_mad_zscore[4], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false, false }, rolling_iqr_outlier);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), rolling_winsorized[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), rolling_winsorized[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), rolling_winsorized[4], 1e-12);

    var ema = try rolling_table.emaProfile("sales", "sales", .{ .alpha = 0.5, .min_periods = 2 });
    defer ema.deinit();
    try std.testing.expectEqual(@as(usize, 5), ema.width());
    const ema_values = try (try ema.column("sales_ema")).f64.toOwnedSlice(gpa);
    defer gpa.free(ema_values);
    const ema_residual = try (try ema.column("sales_ema_residual")).f64.toOwnedSlice(gpa);
    defer gpa.free(ema_residual);
    const ema_ratio = try (try ema.column("sales_ema_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(ema_ratio);
    const ema_validity = try (try ema.column("sales_ema")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(ema_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true, true }, ema_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), ema_values[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.75), ema_values[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.875), ema_values[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), ema_residual[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.25), ema_residual[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.125), ema_residual[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 1.5), ema_ratio[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0 / 2.75), ema_ratio[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 3.875), ema_ratio[4], 1e-12);

    var rolling_range = try rolling_table.rollingRangeProfile("sales", "sales", .{ .window = 3, .min_periods = 2 });
    defer rolling_range.deinit();
    try std.testing.expectEqual(@as(usize, 6), rolling_range.width());
    const rolling_low = try (try rolling_range.column("sales_rolling_low")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_low);
    const rolling_high = try (try rolling_range.column("sales_rolling_high")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_high);
    const rolling_spread = try (try rolling_range.column("sales_rolling_range")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_spread);
    const rolling_position = try (try rolling_range.column("sales_rolling_position")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_position);
    const rolling_range_validity = try (try rolling_range.column("sales_rolling_range")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(rolling_range_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true, true }, rolling_range_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_low[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), rolling_low[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), rolling_low[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), rolling_high[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), rolling_high[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), rolling_high[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_spread[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), rolling_spread[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_spread[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_position[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_position[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_position[4], 1e-12);

    var rolling_normalized = try rolling_table.rollingNormalizeProfile("sales", "sales", .{ .window = 3, .min_periods = 2 });
    defer rolling_normalized.deinit();
    try std.testing.expectEqual(@as(usize, 5), rolling_normalized.width());
    const rolling_centered = try (try rolling_normalized.column("sales_rolling_centered")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_centered);
    const rolling_zscore = try (try rolling_normalized.column("sales_rolling_zscore")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_zscore);
    const rolling_minmax = try (try rolling_normalized.column("sales_rolling_minmax")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_minmax);
    const rolling_norm_validity = try (try rolling_normalized.column("sales_rolling_zscore")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(rolling_norm_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true, true }, rolling_norm_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_centered[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_centered[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_centered[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_zscore[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_zscore[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_zscore[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_minmax[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_minmax[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_minmax[4], 1e-12);

    var rolling_quantiles = try rolling_table.rollingQuantileProfile("sales", "sales", .{ .window = 3, .min_periods = 2 });
    defer rolling_quantiles.deinit();
    try std.testing.expectEqual(@as(usize, 6), rolling_quantiles.width());
    const rolling_q1 = try (try rolling_quantiles.column("sales_rolling_q1")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_q1);
    const rolling_median = try (try rolling_quantiles.column("sales_rolling_median")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_median);
    const rolling_q3 = try (try rolling_quantiles.column("sales_rolling_q3")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_q3);
    const rolling_iqr = try (try rolling_quantiles.column("sales_rolling_iqr")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_iqr);
    const rolling_quantile_validity = try (try rolling_quantiles.column("sales_rolling_median")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(rolling_quantile_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true, true }, rolling_quantile_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.25), rolling_q1[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), rolling_q1[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.25), rolling_q1[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), rolling_median[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), rolling_median[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.5), rolling_median[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.75), rolling_q3[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.5), rolling_q3[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.75), rolling_q3[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_iqr[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_iqr[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_iqr[4], 1e-12);

    var expanding_quantiles = try rolling_table.expandingQuantileProfile("sales", "sales", .{ .min_periods = 2 });
    defer expanding_quantiles.deinit();
    try std.testing.expectEqual(@as(usize, 6), expanding_quantiles.width());
    const expanding_q1 = try (try expanding_quantiles.column("sales_expanding_q1")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_q1);
    const expanding_median = try (try expanding_quantiles.column("sales_expanding_median")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_median);
    const expanding_q3 = try (try expanding_quantiles.column("sales_expanding_q3")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_q3);
    const expanding_iqr = try (try expanding_quantiles.column("sales_expanding_iqr")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_iqr);
    const expanding_quantile_validity = try (try expanding_quantiles.column("sales_expanding_median")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(expanding_quantile_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true }, expanding_quantile_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.25), expanding_q1[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.25), expanding_q1[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), expanding_q1[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.75), expanding_q1[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), expanding_median[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), expanding_median[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), expanding_median[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), expanding_median[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.75), expanding_q3[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.75), expanding_q3[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), expanding_q3[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.25), expanding_q3[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_iqr[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_iqr[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), expanding_iqr[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), expanding_iqr[4], 1e-12);

    var lag_source = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 10.0, 0.0, 15.0, 20.0, 99.0 }, &.{ true, true, true, true, false }, .cpu);
    defer lag_source.deinit();
    var lag_id = try DeviceColumn.fromSlice(i64, gpa, &.{ 1, 2, 3, 4, 5 }, .cpu);
    defer lag_id.deinit();
    var lag_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "sales", .data = lag_source },
        .{ .name = "id", .data = lag_id },
    });
    defer lag_table.deinit();

    var lagged = try lag_table.lagProfile("sales", "sales", .{ .periods = 2 });
    defer lagged.deinit();
    try std.testing.expectEqual(@as(usize, 5), lagged.width());
    const lag_values = try (try lagged.column("sales_lag")).f64.toOwnedSlice(gpa);
    defer gpa.free(lag_values);
    const diff_values = try (try lagged.column("sales_diff")).f64.toOwnedSlice(gpa);
    defer gpa.free(diff_values);
    const pct_values = try (try lagged.column("sales_pct_change")).f64.toOwnedSlice(gpa);
    defer gpa.free(pct_values);
    const lag_validity = try (try lagged.column("sales_lag")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lag_validity);
    const diff_validity = try (try lagged.column("sales_diff")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(diff_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, true, true }, lag_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, true, false }, diff_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), lag_values[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lag_values[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 15.0), lag_values[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), diff_values[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), diff_values[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), pct_values[2], 1e-12);
    try std.testing.expect(std.math.isNan(pct_values[3]));

    var leaded = try lag_table.leadProfile("sales", "sales", .{ .periods = 2 });
    defer leaded.deinit();
    try std.testing.expectEqual(@as(usize, 5), leaded.width());
    const lead_values = try (try leaded.column("sales_lead")).f64.toOwnedSlice(gpa);
    defer gpa.free(lead_values);
    const forward_diff = try (try leaded.column("sales_forward_diff")).f64.toOwnedSlice(gpa);
    defer gpa.free(forward_diff);
    const forward_pct = try (try leaded.column("sales_forward_pct_change")).f64.toOwnedSlice(gpa);
    defer gpa.free(forward_pct);
    const lead_validity = try (try leaded.column("sales_lead")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lead_validity);
    const forward_validity = try (try leaded.column("sales_forward_diff")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(forward_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false, false }, lead_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false, false }, forward_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 15.0), lead_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), lead_values[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), forward_diff[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), forward_diff[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), forward_pct[0], 1e-12);
    try std.testing.expect(std.math.isNan(forward_pct[1]));

    var expanding = try lag_table.expandingProfile("sales", "sales", .{ .min_periods = 2 });
    defer expanding.deinit();
    try std.testing.expectEqual(@as(usize, 7), expanding.width());
    const expanding_count = try (try expanding.column("sales_expanding_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(expanding_count);
    const expanding_sum = try (try expanding.column("sales_expanding_sum")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_sum);
    const expanding_mean = try (try expanding.column("sales_expanding_mean")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_mean);
    const expanding_min = try (try expanding.column("sales_expanding_min")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_min);
    const expanding_max = try (try expanding.column("sales_expanding_max")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_max);
    const expanding_validity = try (try expanding.column("sales_expanding_mean")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(expanding_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4, 4 }, expanding_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true }, expanding_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), expanding_sum[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 25.0), expanding_sum[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 45.0), expanding_sum[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 45.0), expanding_sum[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), expanding_mean[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 25.0 / 3.0), expanding_mean[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 11.25), expanding_mean[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 11.25), expanding_mean[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), expanding_min[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), expanding_min[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), expanding_max[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), expanding_max[4], 1e-12);

    var expanding_normalized = try lag_table.expandingNormalizeProfile("sales", "sales", .{ .min_periods = 2 });
    defer expanding_normalized.deinit();
    try std.testing.expectEqual(@as(usize, 5), expanding_normalized.width());
    const expanding_centered = try (try expanding_normalized.column("sales_expanding_centered")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_centered);
    const expanding_zscore = try (try expanding_normalized.column("sales_expanding_zscore")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_zscore);
    const expanding_minmax = try (try expanding_normalized.column("sales_expanding_minmax")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_minmax);
    const expanding_normalized_validity = try (try expanding_normalized.column("sales_expanding_zscore")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(expanding_normalized_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, false }, expanding_normalized_validity);
    try std.testing.expectApproxEqAbs(@as(f64, -5.0), expanding_centered[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0 / 3.0), expanding_centered[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 8.75), expanding_centered[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -1.0), expanding_zscore[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0690449676496976), expanding_zscore[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.1832159566199232), expanding_zscore[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), expanding_minmax[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), expanding_minmax[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), expanding_minmax[3], 1e-12);

    var expanding_moments = try lag_table.expandingMomentProfile("sales", "sales", .{ .min_periods = 2 });
    defer expanding_moments.deinit();
    try std.testing.expectEqual(@as(usize, 7), expanding_moments.width());
    const expanding_moment_count = try (try expanding_moments.column("sales_expanding_moment_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(expanding_moment_count);
    const expanding_m3 = try (try expanding_moments.column("sales_expanding_m3")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_m3);
    const expanding_m4 = try (try expanding_moments.column("sales_expanding_m4")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_m4);
    const expanding_skewness = try (try expanding_moments.column("sales_expanding_skewness")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_skewness);
    const expanding_kurtosis = try (try expanding_moments.column("sales_expanding_kurtosis")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_kurtosis);
    const expanding_moment_validity = try (try expanding_moments.column("sales_expanding_skewness")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(expanding_moment_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4, 4 }, expanding_moment_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true }, expanding_moment_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), expanding_m3[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -92.59259259259267), expanding_m3[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -175.78125), expanding_m3[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 625.0), expanding_m4[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2268.5185185185187), expanding_m4[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5520.01953125), expanding_m4[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), expanding_skewness[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.3818017741606065), expanding_skewness[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.43465075957466565), expanding_skewness[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -2.0), expanding_kurtosis[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -1.5), expanding_kurtosis[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -1.1542857142857144), expanding_kurtosis[3], 1e-12);

    var clipped = try lag_table.clipProfile("sales", "sales", .{ .lower = 5.0, .upper = 15.0 });
    defer clipped.deinit();
    try std.testing.expectEqual(@as(usize, 6), clipped.width());
    const clipped_values = try (try clipped.column("sales_clipped")).f64.toOwnedSlice(gpa);
    defer gpa.free(clipped_values);
    const below_values = try (try clipped.column("sales_below")).bool.toOwnedSlice(gpa);
    defer gpa.free(below_values);
    const above_values = try (try clipped.column("sales_above")).bool.toOwnedSlice(gpa);
    defer gpa.free(above_values);
    const in_range_values = try (try clipped.column("sales_in_range")).bool.toOwnedSlice(gpa);
    defer gpa.free(in_range_values);
    const clip_validity = try (try clipped.column("sales_clipped")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(clip_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, false }, clip_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), clipped_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), clipped_values[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 15.0), clipped_values[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 15.0), clipped_values[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false, false }, below_values);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, true, false }, above_values);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, false, false }, in_range_values);

    var rolling_clipped = try lag_table.rollingClipProfile("sales", "sales", .{ .lower = 5.0, .upper = 15.0 }, .{ .window = 3, .min_periods = 2 });
    defer rolling_clipped.deinit();
    try std.testing.expectEqual(@as(usize, 8), rolling_clipped.width());
    const rolling_clip_count = try (try rolling_clipped.column("sales_rolling_clip_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(rolling_clip_count);
    const rolling_mean_clipped = try (try rolling_clipped.column("sales_rolling_mean_clipped")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_mean_clipped);
    const rolling_clipped_rate = try (try rolling_clipped.column("sales_rolling_clipped_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_clipped_rate);
    const rolling_clip_below_rate = try (try rolling_clipped.column("sales_rolling_clip_below_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_clip_below_rate);
    const rolling_clip_above_rate = try (try rolling_clipped.column("sales_rolling_clip_above_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_clip_above_rate);
    const rolling_in_range_rate = try (try rolling_clipped.column("sales_rolling_clip_in_range_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_in_range_rate);
    const rolling_clip_validity = try (try rolling_clipped.column("sales_rolling_mean_clipped")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(rolling_clip_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 3, 2 }, rolling_clip_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true }, rolling_clip_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 7.5), rolling_mean_clipped[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), rolling_mean_clipped[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 35.0 / 3.0), rolling_mean_clipped[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 15.0), rolling_mean_clipped[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_clipped_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), rolling_clipped_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), rolling_clipped_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_clipped_rate[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_clip_below_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), rolling_clip_below_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), rolling_clip_below_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), rolling_clip_below_rate[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), rolling_clip_above_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), rolling_clip_above_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), rolling_clip_above_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_clip_above_rate[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_in_range_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), rolling_in_range_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), rolling_in_range_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_in_range_rate[4], 1e-12);

    var expanding_clipped = try lag_table.expandingClipProfile("sales", "sales", .{ .lower = 5.0, .upper = 15.0 }, .{ .min_periods = 2 });
    defer expanding_clipped.deinit();
    try std.testing.expectEqual(@as(usize, 8), expanding_clipped.width());
    const expanding_clip_count = try (try expanding_clipped.column("sales_expanding_clip_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(expanding_clip_count);
    const expanding_mean_clipped = try (try expanding_clipped.column("sales_expanding_mean_clipped")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_mean_clipped);
    const expanding_clipped_rate = try (try expanding_clipped.column("sales_expanding_clipped_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_clipped_rate);
    const expanding_clip_below_rate = try (try expanding_clipped.column("sales_expanding_clip_below_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_clip_below_rate);
    const expanding_clip_above_rate = try (try expanding_clipped.column("sales_expanding_clip_above_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_clip_above_rate);
    const expanding_clip_in_range_rate = try (try expanding_clipped.column("sales_expanding_clip_in_range_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_clip_in_range_rate);
    const expanding_clip_validity = try (try expanding_clipped.column("sales_expanding_mean_clipped")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(expanding_clip_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4, 4 }, expanding_clip_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true }, expanding_clip_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 7.5), expanding_mean_clipped[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), expanding_mean_clipped[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 11.25), expanding_mean_clipped[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 11.25), expanding_mean_clipped[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_clipped_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), expanding_clipped_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_clipped_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_clipped_rate[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_clip_below_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), expanding_clip_below_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), expanding_clip_below_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), expanding_clip_below_rate[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), expanding_clip_above_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), expanding_clip_above_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), expanding_clip_above_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), expanding_clip_above_rate[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_clip_in_range_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), expanding_clip_in_range_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_clip_in_range_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_clip_in_range_rate[4], 1e-12);

    var thresholded = try lag_table.thresholdProfile("sales", "sales", .{ .threshold = 10.0 });
    defer thresholded.deinit();
    try std.testing.expectEqual(@as(usize, 7), thresholded.width());
    const threshold_distance = try (try thresholded.column("sales_distance")).f64.toOwnedSlice(gpa);
    defer gpa.free(threshold_distance);
    const threshold_abs_distance = try (try thresholded.column("sales_abs_distance")).f64.toOwnedSlice(gpa);
    defer gpa.free(threshold_abs_distance);
    const threshold_above = try (try thresholded.column("sales_above")).bool.toOwnedSlice(gpa);
    defer gpa.free(threshold_above);
    const threshold_below = try (try thresholded.column("sales_below")).bool.toOwnedSlice(gpa);
    defer gpa.free(threshold_below);
    const threshold_at = try (try thresholded.column("sales_at")).bool.toOwnedSlice(gpa);
    defer gpa.free(threshold_at);
    const threshold_validity = try (try thresholded.column("sales_distance")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(threshold_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, false }, threshold_validity);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, -10.0, 5.0, 10.0, 0.0 }, threshold_distance);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 10.0, 5.0, 10.0, 0.0 }, threshold_abs_distance);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, true, false }, threshold_above);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false, false }, threshold_below);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false, false }, threshold_at);

    var rolling_thresholded = try lag_table.rollingThresholdProfile("sales", "sales", 10.0, .{ .window = 3, .min_periods = 2 });
    defer rolling_thresholded.deinit();
    try std.testing.expectEqual(@as(usize, 8), rolling_thresholded.width());
    const rolling_threshold_count = try (try rolling_thresholded.column("sales_rolling_threshold_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(rolling_threshold_count);
    const rolling_mean_distance = try (try rolling_thresholded.column("sales_rolling_mean_distance")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_mean_distance);
    const rolling_mean_abs_distance = try (try rolling_thresholded.column("sales_rolling_mean_abs_distance")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_mean_abs_distance);
    const rolling_above_rate = try (try rolling_thresholded.column("sales_rolling_above_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_above_rate);
    const rolling_below_rate = try (try rolling_thresholded.column("sales_rolling_below_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_below_rate);
    const rolling_at_rate = try (try rolling_thresholded.column("sales_rolling_at_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_at_rate);
    const rolling_threshold_validity = try (try rolling_thresholded.column("sales_rolling_mean_distance")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(rolling_threshold_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 3, 2 }, rolling_threshold_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true }, rolling_threshold_validity);
    try std.testing.expectApproxEqAbs(@as(f64, -5.0), rolling_mean_distance[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -5.0 / 3.0), rolling_mean_distance[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 3.0), rolling_mean_distance[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 7.5), rolling_mean_distance[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), rolling_mean_abs_distance[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), rolling_mean_abs_distance[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 25.0 / 3.0), rolling_mean_abs_distance[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 7.5), rolling_mean_abs_distance[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), rolling_above_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), rolling_above_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), rolling_above_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_above_rate[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_below_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), rolling_below_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), rolling_below_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), rolling_below_rate[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_at_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), rolling_at_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), rolling_at_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), rolling_at_rate[4], 1e-12);

    var expanding_thresholded = try lag_table.expandingThresholdProfile("sales", "sales", 10.0, .{ .min_periods = 2 });
    defer expanding_thresholded.deinit();
    try std.testing.expectEqual(@as(usize, 8), expanding_thresholded.width());
    const expanding_threshold_count = try (try expanding_thresholded.column("sales_expanding_threshold_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(expanding_threshold_count);
    const expanding_mean_distance = try (try expanding_thresholded.column("sales_expanding_mean_distance")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_mean_distance);
    const expanding_mean_abs_distance = try (try expanding_thresholded.column("sales_expanding_mean_abs_distance")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_mean_abs_distance);
    const expanding_above_rate = try (try expanding_thresholded.column("sales_expanding_above_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_above_rate);
    const expanding_below_rate = try (try expanding_thresholded.column("sales_expanding_below_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_below_rate);
    const expanding_at_rate = try (try expanding_thresholded.column("sales_expanding_at_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_at_rate);
    const expanding_threshold_validity = try (try expanding_thresholded.column("sales_expanding_mean_distance")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(expanding_threshold_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4, 4 }, expanding_threshold_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true }, expanding_threshold_validity);
    try std.testing.expectApproxEqAbs(@as(f64, -5.0), expanding_mean_distance[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -5.0 / 3.0), expanding_mean_distance[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.25), expanding_mean_distance[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.25), expanding_mean_distance[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), expanding_mean_abs_distance[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), expanding_mean_abs_distance[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 6.25), expanding_mean_abs_distance[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 6.25), expanding_mean_abs_distance[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), expanding_above_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), expanding_above_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_above_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_above_rate[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_below_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), expanding_below_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), expanding_below_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), expanding_below_rate[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_at_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), expanding_at_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), expanding_at_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), expanding_at_rate[4], 1e-12);

    var scaled = try lag_table.standardizeProfile("sales", "sales", .{ .min_periods = 3 });
    defer scaled.deinit();
    try std.testing.expectEqual(@as(usize, 5), scaled.width());
    const centered = try (try scaled.column("sales_centered")).f64.toOwnedSlice(gpa);
    defer gpa.free(centered);
    const zscore = try (try scaled.column("sales_zscore")).f64.toOwnedSlice(gpa);
    defer gpa.free(zscore);
    const minmax = try (try scaled.column("sales_minmax")).f64.toOwnedSlice(gpa);
    defer gpa.free(minmax);
    const scaled_validity = try (try scaled.column("sales_zscore")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(scaled_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, false }, scaled_validity);
    try std.testing.expectApproxEqAbs(@as(f64, -1.25), centered[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -11.25), centered[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.75), centered[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 8.75), centered[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.1690308509457033), zscore[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -1.5212776585113297), zscore[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.50709255283711), zscore[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.1832159566199232), zscore[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), minmax[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), minmax[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), minmax[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), minmax[3], 1e-12);

    var robust_source = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, 2.0, 3.0, 100.0, 0.0 }, &.{ true, true, true, true, false }, .cpu);
    defer robust_source.deinit();
    var robust_id = try DeviceColumn.fromSlice(i64, gpa, &.{ 1, 2, 3, 4, 5 }, .cpu);
    defer robust_id.deinit();
    var robust_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "value", .data = robust_source },
        .{ .name = "id", .data = robust_id },
    });
    defer robust_table.deinit();

    var robust = try robust_table.robustProfile("value", "value", .{ .min_periods = 4 });
    defer robust.deinit();
    try std.testing.expectEqual(@as(usize, 6), robust.width());
    const median_centered = try (try robust.column("value_median_centered")).f64.toOwnedSlice(gpa);
    defer gpa.free(median_centered);
    const mad_zscore = try (try robust.column("value_mad_zscore")).f64.toOwnedSlice(gpa);
    defer gpa.free(mad_zscore);
    const iqr_outlier = try (try robust.column("value_iqr_outlier")).bool.toOwnedSlice(gpa);
    defer gpa.free(iqr_outlier);
    const winsorized = try (try robust.column("value_winsorized")).f64.toOwnedSlice(gpa);
    defer gpa.free(winsorized);
    const robust_validity = try (try robust.column("value_mad_zscore")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(robust_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, false }, robust_validity);
    try std.testing.expectApproxEqAbs(@as(f64, -1.5), median_centered[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.5), median_centered[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), median_centered[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 97.5), median_centered[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -1.0117346252941226), mad_zscore[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.33724487509804085), mad_zscore[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.33724487509804085), mad_zscore[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 65.76275064411797), mad_zscore[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, true, false }, iqr_outlier);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), winsorized[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), winsorized[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), winsorized[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 65.5), winsorized[3], 1e-12);

    var expanding_robust = try robust_table.expandingRobustProfile("value", "value", .{ .min_periods = 2 });
    defer expanding_robust.deinit();
    try std.testing.expectEqual(@as(usize, 6), expanding_robust.width());
    const expanding_median_centered = try (try expanding_robust.column("value_expanding_median_centered")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_median_centered);
    const expanding_mad_zscore = try (try expanding_robust.column("value_expanding_mad_zscore")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_mad_zscore);
    const expanding_iqr_outlier = try (try expanding_robust.column("value_expanding_iqr_outlier")).bool.toOwnedSlice(gpa);
    defer gpa.free(expanding_iqr_outlier);
    const expanding_winsorized = try (try expanding_robust.column("value_expanding_winsorized")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_winsorized);
    const expanding_robust_validity = try (try expanding_robust.column("value_expanding_mad_zscore")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(expanding_robust_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, false }, expanding_robust_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_median_centered[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), expanding_median_centered[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 97.5), expanding_median_centered[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.6744897501960817), expanding_mad_zscore[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.6744897501960817), expanding_mad_zscore[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 65.76275064411797), expanding_mad_zscore[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, true, false }, expanding_iqr_outlier);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), expanding_winsorized[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), expanding_winsorized[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 65.5), expanding_winsorized[3], 1e-12);

    var equity = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 100.0, 120.0, 90.0, 130.0, 80.0, 0.0 }, &.{ true, true, true, true, true, false }, .cpu);
    defer equity.deinit();
    var equity_id = try DeviceColumn.fromSlice(i64, gpa, &.{ 1, 2, 3, 4, 5, 6 }, .cpu);
    defer equity_id.deinit();
    var equity_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "equity", .data = equity },
        .{ .name = "id", .data = equity_id },
    });
    defer equity_table.deinit();

    var drawdown = try equity_table.drawdownProfile("equity", "equity", .{ .min_periods = 2 });
    defer drawdown.deinit();
    try std.testing.expectEqual(@as(usize, 5), drawdown.width());
    const running_peak = try (try drawdown.column("equity_running_peak")).f64.toOwnedSlice(gpa);
    defer gpa.free(running_peak);
    const drawdown_values = try (try drawdown.column("equity_drawdown")).f64.toOwnedSlice(gpa);
    defer gpa.free(drawdown_values);
    const drawdown_pct = try (try drawdown.column("equity_drawdown_pct")).f64.toOwnedSlice(gpa);
    defer gpa.free(drawdown_pct);
    const drawdown_validity = try (try drawdown.column("equity_drawdown")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(drawdown_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true, false }, drawdown_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 120.0), running_peak[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 120.0), running_peak[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 130.0), running_peak[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 130.0), running_peak[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), drawdown_values[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -30.0), drawdown_values[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), drawdown_values[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -50.0), drawdown_values[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), drawdown_pct[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.25), drawdown_pct[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), drawdown_pct[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -50.0 / 130.0), drawdown_pct[4], 1e-12);

    var rolling_drawdown = try equity_table.rollingDrawdownProfile("equity", "equity", .{ .window = 3, .min_periods = 2 });
    defer rolling_drawdown.deinit();
    try std.testing.expectEqual(@as(usize, 6), rolling_drawdown.width());
    const rolling_peak = try (try rolling_drawdown.column("equity_rolling_peak")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_peak);
    const rolling_drawdown_values = try (try rolling_drawdown.column("equity_rolling_drawdown")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_drawdown_values);
    const rolling_drawdown_pct = try (try rolling_drawdown.column("equity_rolling_drawdown_pct")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_drawdown_pct);
    const rolling_peak_age = try (try rolling_drawdown.column("equity_rolling_peak_age")).i64.toOwnedSlice(gpa);
    defer gpa.free(rolling_peak_age);
    const rolling_drawdown_validity = try (try rolling_drawdown.column("equity_rolling_drawdown")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(rolling_drawdown_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true, false }, rolling_drawdown_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 120.0), rolling_peak[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 120.0), rolling_peak[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 130.0), rolling_peak[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 130.0), rolling_peak[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), rolling_drawdown_values[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -30.0), rolling_drawdown_values[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), rolling_drawdown_values[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -50.0), rolling_drawdown_values[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), rolling_drawdown_pct[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.25), rolling_drawdown_pct[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), rolling_drawdown_pct[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -50.0 / 130.0), rolling_drawdown_pct[4], 1e-12);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 0, 1, 0 }, rolling_peak_age);

    var extrema = try equity_table.extremaProfile("equity", "equity", .{ .min_periods = 2 });
    defer extrema.deinit();
    try std.testing.expectEqual(@as(usize, 6), extrema.width());
    const running_low = try (try extrema.column("equity_running_low")).f64.toOwnedSlice(gpa);
    defer gpa.free(running_low);
    const running_high = try (try extrema.column("equity_running_high")).f64.toOwnedSlice(gpa);
    defer gpa.free(running_high);
    const new_low = try (try extrema.column("equity_new_low")).bool.toOwnedSlice(gpa);
    defer gpa.free(new_low);
    const new_high = try (try extrema.column("equity_new_high")).bool.toOwnedSlice(gpa);
    defer gpa.free(new_high);
    const extrema_validity = try (try extrema.column("equity_running_low")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(extrema_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true, false }, extrema_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 100.0), running_low[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 90.0), running_low[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 90.0), running_low[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 80.0), running_low[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 120.0), running_high[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 120.0), running_high[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 130.0), running_high[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 130.0), running_high[4], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, true, false }, new_low);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true, false, false }, new_high);

    var trend_source = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, 3.0, 2.0, 2.0, 5.0, 0.0, 4.0 }, &.{ true, true, true, true, true, false, true }, .cpu);
    defer trend_source.deinit();
    var trend_id = try DeviceColumn.fromSlice(i64, gpa, &.{ 1, 2, 3, 4, 5, 6, 7 }, .cpu);
    defer trend_id.deinit();
    var trend_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "price", .data = trend_source },
        .{ .name = "id", .data = trend_id },
    });
    defer trend_table.deinit();

    var trend = try trend_table.trendProfile("price", "price", .{ .periods = 1 });
    defer trend.deinit();
    try std.testing.expectEqual(@as(usize, 7), trend.width());
    const trend_values = try (try trend.column("price_trend")).i64.toOwnedSlice(gpa);
    defer gpa.free(trend_values);
    const up_streak = try (try trend.column("price_up_streak")).i64.toOwnedSlice(gpa);
    defer gpa.free(up_streak);
    const down_streak = try (try trend.column("price_down_streak")).i64.toOwnedSlice(gpa);
    defer gpa.free(down_streak);
    const flat_streak = try (try trend.column("price_flat_streak")).i64.toOwnedSlice(gpa);
    defer gpa.free(flat_streak);
    const reversal = try (try trend.column("price_reversal")).bool.toOwnedSlice(gpa);
    defer gpa.free(reversal);
    const trend_validity = try (try trend.column("price_trend")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(trend_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true, false, false }, trend_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, -1, 0, 1, 0, 0 }, trend_values);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 0, 1, 0, 0 }, up_streak);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 0, 0, 0, 0 }, down_streak);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 1, 0, 0, 0 }, flat_streak);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, true, false, false }, reversal);

    var rolling_trends = try trend_table.rollingTrendProfile("price", "price", .{ .periods = 1 }, .{ .window = 3, .min_periods = 2 });
    defer rolling_trends.deinit();
    try std.testing.expectEqual(@as(usize, 7), rolling_trends.width());
    const rolling_trend_count = try (try rolling_trends.column("price_rolling_trend_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(rolling_trend_count);
    const rolling_up_rate = try (try rolling_trends.column("price_rolling_up_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_up_rate);
    const rolling_down_rate = try (try rolling_trends.column("price_rolling_down_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_down_rate);
    const rolling_flat_rate = try (try rolling_trends.column("price_rolling_flat_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_flat_rate);
    const rolling_reversal_rate = try (try rolling_trends.column("price_rolling_reversal_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_reversal_rate);
    const rolling_trend_validity = try (try rolling_trends.column("price_rolling_up_rate")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(rolling_trend_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 2, 3, 3, 2, 1 }, rolling_trend_count);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, true, true, true, false }, rolling_trend_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_up_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), rolling_up_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), rolling_up_rate[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_up_rate[5], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_down_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), rolling_down_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), rolling_flat_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), rolling_reversal_rate[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_reversal_rate[5], 1e-12);

    var expanding_trends = try trend_table.expandingTrendProfile("price", "price", .{ .periods = 1 }, .{ .min_periods = 2 });
    defer expanding_trends.deinit();
    try std.testing.expectEqual(@as(usize, 7), expanding_trends.width());
    const expanding_trend_count = try (try expanding_trends.column("price_expanding_trend_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(expanding_trend_count);
    const expanding_up_rate = try (try expanding_trends.column("price_expanding_up_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_up_rate);
    const expanding_down_rate = try (try expanding_trends.column("price_expanding_down_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_down_rate);
    const expanding_flat_rate = try (try expanding_trends.column("price_expanding_flat_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_flat_rate);
    const expanding_reversal_rate = try (try expanding_trends.column("price_expanding_reversal_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_reversal_rate);
    const expanding_trend_validity = try (try expanding_trends.column("price_expanding_up_rate")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(expanding_trend_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 2, 3, 4, 4, 4 }, expanding_trend_count);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, true, true, true, true }, expanding_trend_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_up_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), expanding_up_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_up_rate[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_down_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), expanding_down_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), expanding_down_rate[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), expanding_flat_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), expanding_flat_rate[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_reversal_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), expanding_reversal_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_reversal_rate[4], 1e-12);

    var changes = try trend_table.changePointProfile("price", "price", 2.0, .{ .periods = 1 });
    defer changes.deinit();
    try std.testing.expectEqual(@as(usize, 6), changes.width());
    const change_delta = try (try changes.column("price_change_delta")).f64.toOwnedSlice(gpa);
    defer gpa.free(change_delta);
    const change_abs_delta = try (try changes.column("price_change_abs_delta")).f64.toOwnedSlice(gpa);
    defer gpa.free(change_abs_delta);
    const change_pct = try (try changes.column("price_change_pct")).f64.toOwnedSlice(gpa);
    defer gpa.free(change_pct);
    const change_point = try (try changes.column("price_change_point")).bool.toOwnedSlice(gpa);
    defer gpa.free(change_point);
    const change_validity = try (try changes.column("price_change_delta")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(change_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true, false, false }, change_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), change_delta[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -1.0), change_delta[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), change_delta[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), change_delta[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), change_abs_delta[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), change_abs_delta[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), change_abs_delta[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), change_abs_delta[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), change_pct[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -1.0 / 3.0), change_pct[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), change_pct[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), change_pct[4], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false, true, false, false }, change_point);

    var rolling_changes = try trend_table.rollingChangePointProfile("price", "price", 2.0, .{ .periods = 1 }, .{ .window = 3, .min_periods = 2 });
    defer rolling_changes.deinit();
    try std.testing.expectEqual(@as(usize, 7), rolling_changes.width());
    const rolling_change_count = try (try rolling_changes.column("price_rolling_change_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(rolling_change_count);
    const rolling_change_point_count = try (try rolling_changes.column("price_rolling_change_point_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(rolling_change_point_count);
    const rolling_change_rate = try (try rolling_changes.column("price_rolling_change_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_change_rate);
    const rolling_mean_abs_delta = try (try rolling_changes.column("price_rolling_mean_abs_delta")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_mean_abs_delta);
    const rolling_max_abs_delta = try (try rolling_changes.column("price_rolling_max_abs_delta")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_max_abs_delta);
    const rolling_change_validity = try (try rolling_changes.column("price_rolling_change_rate")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(rolling_change_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 2, 3, 3, 2, 1 }, rolling_change_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 1, 1, 1, 1, 1 }, rolling_change_point_count);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, true, true, true, false }, rolling_change_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_change_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), rolling_change_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), rolling_change_rate[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_change_rate[5], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), rolling_mean_abs_delta[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_mean_abs_delta[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0 / 3.0), rolling_mean_abs_delta[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), rolling_mean_abs_delta[5], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), rolling_max_abs_delta[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), rolling_max_abs_delta[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), rolling_max_abs_delta[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), rolling_max_abs_delta[5], 1e-12);

    var expanding_changes = try trend_table.expandingChangePointProfile("price", "price", 2.0, .{ .periods = 1 }, .{ .min_periods = 2 });
    defer expanding_changes.deinit();
    try std.testing.expectEqual(@as(usize, 7), expanding_changes.width());
    const expanding_change_count = try (try expanding_changes.column("price_expanding_change_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(expanding_change_count);
    const expanding_change_point_count = try (try expanding_changes.column("price_expanding_change_point_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(expanding_change_point_count);
    const expanding_change_rate = try (try expanding_changes.column("price_expanding_change_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_change_rate);
    const expanding_mean_abs_delta = try (try expanding_changes.column("price_expanding_mean_abs_delta")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_mean_abs_delta);
    const expanding_max_abs_delta = try (try expanding_changes.column("price_expanding_max_abs_delta")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_max_abs_delta);
    const expanding_change_validity = try (try expanding_changes.column("price_expanding_change_rate")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(expanding_change_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 2, 3, 4, 4, 4 }, expanding_change_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 1, 1, 2, 2, 2 }, expanding_change_point_count);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, true, true, true, true }, expanding_change_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_change_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), expanding_change_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_change_rate[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), expanding_mean_abs_delta[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), expanding_mean_abs_delta[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), expanding_mean_abs_delta[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), expanding_max_abs_delta[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), expanding_max_abs_delta[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), expanding_max_abs_delta[4], 1e-12);

    var signed_values_col = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ -1.0, -2.0, 0.0, 3.0, -4.0, 0.0, 5.0 }, &.{ true, true, true, true, true, false, true }, .cpu);
    defer signed_values_col.deinit();
    var signed_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "signal", .data = signed_values_col },
    });
    defer signed_table.deinit();
    var sign = try signed_table.signProfile("signal", "signal", .{ .periods = 1 });
    defer sign.deinit();
    try std.testing.expectEqual(@as(usize, 6), sign.width());
    const sign_values = try (try sign.column("signal_sign")).i64.toOwnedSlice(gpa);
    defer gpa.free(sign_values);
    const sign_flip = try (try sign.column("signal_sign_flip")).bool.toOwnedSlice(gpa);
    defer gpa.free(sign_flip);
    const positive_streak = try (try sign.column("signal_positive_streak")).i64.toOwnedSlice(gpa);
    defer gpa.free(positive_streak);
    const negative_streak = try (try sign.column("signal_negative_streak")).i64.toOwnedSlice(gpa);
    defer gpa.free(negative_streak);
    const zero_streak = try (try sign.column("signal_zero_streak")).i64.toOwnedSlice(gpa);
    defer gpa.free(zero_streak);
    const sign_validity = try (try sign.column("signal_sign")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(sign_validity);
    const flip_validity = try (try sign.column("signal_sign_flip")).bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(flip_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, true, false, true }, sign_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true, false, false }, flip_validity);
    try std.testing.expectEqualSlices(i64, &.{ -1, -1, 0, 1, -1, 0, 1 }, sign_values);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, true, true, false, false }, sign_flip);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 1, 0, 0, 1 }, positive_streak);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 0, 0, 1, 0, 0 }, negative_streak);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 0, 0, 0, 0 }, zero_streak);

    var rolling_sign = try signed_table.rollingSignProfile("signal", "signal", .{ .periods = 1 }, .{ .window = 3, .min_periods = 2 });
    defer rolling_sign.deinit();
    try std.testing.expectEqual(@as(usize, 6), rolling_sign.width());
    const rolling_sign_count = try (try rolling_sign.column("signal_rolling_sign_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(rolling_sign_count);
    const rolling_positive_rate = try (try rolling_sign.column("signal_rolling_positive_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_positive_rate);
    const rolling_negative_rate = try (try rolling_sign.column("signal_rolling_negative_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_negative_rate);
    const rolling_zero_rate = try (try rolling_sign.column("signal_rolling_zero_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_zero_rate);
    const rolling_flip_rate = try (try rolling_sign.column("signal_rolling_sign_flip_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_flip_rate);
    const rolling_sign_rate_validity = try (try rolling_sign.column("signal_rolling_positive_rate")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(rolling_sign_rate_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 3, 3, 2, 2 }, rolling_sign_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true, true, true }, rolling_sign_rate_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), rolling_positive_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), rolling_positive_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), rolling_positive_rate[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_positive_rate[6], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_negative_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), rolling_negative_rate[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), rolling_zero_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), rolling_zero_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), rolling_flip_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_flip_rate[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_flip_rate[6], 1e-12);

    var expanding_sign = try signed_table.expandingSignProfile("signal", "signal", .{ .periods = 1 }, .{ .min_periods = 2 });
    defer expanding_sign.deinit();
    try std.testing.expectEqual(@as(usize, 6), expanding_sign.width());
    const expanding_sign_count = try (try expanding_sign.column("signal_expanding_sign_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(expanding_sign_count);
    const expanding_positive_rate = try (try expanding_sign.column("signal_expanding_positive_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_positive_rate);
    const expanding_negative_rate = try (try expanding_sign.column("signal_expanding_negative_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_negative_rate);
    const expanding_zero_rate = try (try expanding_sign.column("signal_expanding_zero_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_zero_rate);
    const expanding_flip_rate = try (try expanding_sign.column("signal_expanding_sign_flip_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_flip_rate);
    const expanding_sign_rate_validity = try (try expanding_sign.column("signal_expanding_positive_rate")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(expanding_sign_rate_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4, 5, 5, 6 }, expanding_sign_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true, true, true }, expanding_sign_rate_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), expanding_positive_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), expanding_positive_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.2), expanding_positive_rate[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), expanding_positive_rate[6], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), expanding_negative_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_negative_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.6), expanding_negative_rate[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_negative_rate[6], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), expanding_zero_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.2), expanding_zero_rate[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), expanding_flip_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_flip_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.6), expanding_flip_rate[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_flip_rate[6], 1e-12);

    var validity = try trend_table.validityProfile("price", "price");
    defer validity.deinit();
    try std.testing.expectEqual(@as(usize, 6), validity.width());
    const is_null = try (try validity.column("price_is_null")).bool.toOwnedSlice(gpa);
    defer gpa.free(is_null);
    const is_valid = try (try validity.column("price_is_valid")).bool.toOwnedSlice(gpa);
    defer gpa.free(is_valid);
    const valid_streak = try (try validity.column("price_valid_streak")).i64.toOwnedSlice(gpa);
    defer gpa.free(valid_streak);
    const null_streak = try (try validity.column("price_null_streak")).i64.toOwnedSlice(gpa);
    defer gpa.free(null_streak);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false, false, true, false }, is_null);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, true, false, true }, is_valid);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4, 5, 0, 1 }, valid_streak);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0, 0, 1, 0 }, null_streak);

    var rolling_quality = try trend_table.rollingValidityProfile("price", "price", .{ .window = 3, .min_periods = 2 });
    defer rolling_quality.deinit();
    try std.testing.expectEqual(@as(usize, 7), rolling_quality.width());
    const rolling_validity_count = try (try rolling_quality.column("price_rolling_validity_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(rolling_validity_count);
    const rolling_valid_count = try (try rolling_quality.column("price_rolling_valid_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(rolling_valid_count);
    const rolling_null_count = try (try rolling_quality.column("price_rolling_null_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(rolling_null_count);
    const rolling_valid_rate = try (try rolling_quality.column("price_rolling_valid_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_valid_rate);
    const rolling_null_rate = try (try rolling_quality.column("price_rolling_null_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_null_rate);
    const rolling_validity_rate_validity = try (try rolling_quality.column("price_rolling_valid_rate")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(rolling_validity_rate_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 3, 3, 3, 3 }, rolling_validity_count);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 3, 3, 2, 2 }, rolling_valid_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0, 0, 1, 1 }, rolling_null_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true, true, true }, rolling_validity_rate_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_valid_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_valid_rate[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), rolling_valid_rate[5], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), rolling_valid_rate[6], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), rolling_null_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), rolling_null_rate[5], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), rolling_null_rate[6], 1e-12);

    var expanding_quality = try trend_table.expandingValidityProfile("price", "price", .{ .min_periods = 2 });
    defer expanding_quality.deinit();
    try std.testing.expectEqual(@as(usize, 7), expanding_quality.width());
    const expanding_validity_count = try (try expanding_quality.column("price_expanding_validity_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(expanding_validity_count);
    const expanding_valid_count = try (try expanding_quality.column("price_expanding_valid_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(expanding_valid_count);
    const expanding_null_count = try (try expanding_quality.column("price_expanding_null_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(expanding_null_count);
    const expanding_valid_rate = try (try expanding_quality.column("price_expanding_valid_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_valid_rate);
    const expanding_null_rate = try (try expanding_quality.column("price_expanding_null_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_null_rate);
    const expanding_validity_rate_validity = try (try expanding_quality.column("price_expanding_valid_rate")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(expanding_validity_rate_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4, 5, 6, 7 }, expanding_validity_count);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4, 5, 5, 6 }, expanding_valid_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0, 0, 1, 1 }, expanding_null_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true, true, true }, expanding_validity_rate_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), expanding_valid_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), expanding_valid_rate[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 6.0), expanding_valid_rate[5], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 6.0 / 7.0), expanding_valid_rate[6], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), expanding_null_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 6.0), expanding_null_rate[5], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 7.0), expanding_null_rate[6], 1e-12);

    var actual_label = try DeviceColumn.fromSliceWithValidity(bool, gpa, &.{ true, false, true, false, true }, &.{ true, true, true, false, true }, .cpu);
    defer actual_label.deinit();
    var predicted_label = try DeviceColumn.fromSlice(bool, gpa, &.{ true, true, false, false, true }, .cpu);
    defer predicted_label.deinit();
    var label_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "actual", .data = actual_label },
        .{ .name = "predicted", .data = predicted_label },
    });
    defer label_table.deinit();

    var classes = try label_table.classificationProfile("actual", "predicted", "cls");
    defer classes.deinit();
    try std.testing.expectEqual(@as(usize, 7), classes.width());
    const tp = try (try classes.column("cls_tp")).bool.toOwnedSlice(gpa);
    defer gpa.free(tp);
    const fp = try (try classes.column("cls_fp")).bool.toOwnedSlice(gpa);
    defer gpa.free(fp);
    const tn = try (try classes.column("cls_tn")).bool.toOwnedSlice(gpa);
    defer gpa.free(tn);
    const fn_values = try (try classes.column("cls_fn")).bool.toOwnedSlice(gpa);
    defer gpa.free(fn_values);
    const correct = try (try classes.column("cls_correct")).bool.toOwnedSlice(gpa);
    defer gpa.free(correct);
    const class_validity = try (try classes.column("cls_correct")).bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(class_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false, true }, class_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false, true }, tp);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false, false }, fp);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false, false }, tn);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, false }, fn_values);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false, true }, correct);

    var rolling_classes = try label_table.rollingClassificationProfile("actual", "predicted", "cls_roll", .{ .window = 3, .min_periods = 2 });
    defer rolling_classes.deinit();
    try std.testing.expectEqual(@as(usize, 10), rolling_classes.width());
    const rolling_class_count = try (try rolling_classes.column("cls_roll_rolling_class_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(rolling_class_count);
    const rolling_tp_count = try (try rolling_classes.column("cls_roll_rolling_tp_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(rolling_tp_count);
    const rolling_fp_count = try (try rolling_classes.column("cls_roll_rolling_fp_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(rolling_fp_count);
    const rolling_tn_count = try (try rolling_classes.column("cls_roll_rolling_tn_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(rolling_tn_count);
    const rolling_fn_count = try (try rolling_classes.column("cls_roll_rolling_fn_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(rolling_fn_count);
    const rolling_accuracy = try (try rolling_classes.column("cls_roll_rolling_accuracy")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_accuracy);
    const rolling_precision = try (try rolling_classes.column("cls_roll_rolling_precision")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_precision);
    const rolling_recall = try (try rolling_classes.column("cls_roll_rolling_recall")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_recall);
    const rolling_class_validity = try (try rolling_classes.column("cls_roll_rolling_accuracy")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(rolling_class_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 2, 2 }, rolling_class_count);
    try std.testing.expectEqualSlices(i64, &.{ 1, 1, 1, 0, 1 }, rolling_tp_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 1, 1, 0 }, rolling_fp_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0, 0 }, rolling_tn_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 1, 1 }, rolling_fn_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true }, rolling_class_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_accuracy[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), rolling_accuracy[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), rolling_accuracy[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_accuracy[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_precision[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_precision[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), rolling_precision[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_precision[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_recall[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_recall[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), rolling_recall[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_recall[4], 1e-12);

    var expanding_classes = try label_table.expandingClassificationProfile("actual", "predicted", "cls_expand", .{ .min_periods = 2 });
    defer expanding_classes.deinit();
    try std.testing.expectEqual(@as(usize, 10), expanding_classes.width());
    const expanding_class_count = try (try expanding_classes.column("cls_expand_expanding_class_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(expanding_class_count);
    const expanding_tp_count = try (try expanding_classes.column("cls_expand_expanding_tp_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(expanding_tp_count);
    const expanding_fp_count = try (try expanding_classes.column("cls_expand_expanding_fp_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(expanding_fp_count);
    const expanding_tn_count = try (try expanding_classes.column("cls_expand_expanding_tn_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(expanding_tn_count);
    const expanding_fn_count = try (try expanding_classes.column("cls_expand_expanding_fn_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(expanding_fn_count);
    const expanding_accuracy = try (try expanding_classes.column("cls_expand_expanding_accuracy")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_accuracy);
    const expanding_precision = try (try expanding_classes.column("cls_expand_expanding_precision")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_precision);
    const expanding_recall = try (try expanding_classes.column("cls_expand_expanding_recall")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_recall);
    const expanding_class_validity = try (try expanding_classes.column("cls_expand_expanding_accuracy")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(expanding_class_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 3, 4 }, expanding_class_count);
    try std.testing.expectEqualSlices(i64, &.{ 1, 1, 1, 1, 2 }, expanding_tp_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 1, 1, 1 }, expanding_fp_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0, 0 }, expanding_tn_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 1, 1 }, expanding_fn_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true }, expanding_class_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_accuracy[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), expanding_accuracy[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_accuracy[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_precision[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_precision[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), expanding_precision[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), expanding_recall[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_recall[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), expanding_recall[4], 1e-12);

    var transitions = try label_table.boolTransitionProfile("actual", "actual", .{ .periods = 1 });
    defer transitions.deinit();
    try std.testing.expectEqual(@as(usize, 7), transitions.width());
    const rising = try (try transitions.column("actual_rising")).bool.toOwnedSlice(gpa);
    defer gpa.free(rising);
    const falling = try (try transitions.column("actual_falling")).bool.toOwnedSlice(gpa);
    defer gpa.free(falling);
    const toggled = try (try transitions.column("actual_toggled")).bool.toOwnedSlice(gpa);
    defer gpa.free(toggled);
    const true_streak = try (try transitions.column("actual_true_streak")).i64.toOwnedSlice(gpa);
    defer gpa.free(true_streak);
    const false_streak = try (try transitions.column("actual_false_streak")).i64.toOwnedSlice(gpa);
    defer gpa.free(false_streak);
    const transition_validity = try (try transitions.column("actual_toggled")).bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(transition_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, false, false }, transition_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, false }, rising);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false, false }, falling);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, false, false }, toggled);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 1, 0, 1 }, true_streak);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 0, 0 }, false_streak);

    var rolling_transitions = try label_table.rollingBoolTransitionProfile("actual", "actual", .{ .periods = 1 }, .{ .window = 3, .min_periods = 2 });
    defer rolling_transitions.deinit();
    try std.testing.expectEqual(@as(usize, 9), rolling_transitions.width());
    const rolling_transition_count = try (try rolling_transitions.column("actual_rolling_transition_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(rolling_transition_count);
    const rolling_rising_count = try (try rolling_transitions.column("actual_rolling_rising_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(rolling_rising_count);
    const rolling_falling_count = try (try rolling_transitions.column("actual_rolling_falling_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(rolling_falling_count);
    const rolling_toggle_count = try (try rolling_transitions.column("actual_rolling_toggle_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(rolling_toggle_count);
    const rolling_rising_rate = try (try rolling_transitions.column("actual_rolling_rising_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_rising_rate);
    const rolling_falling_rate = try (try rolling_transitions.column("actual_rolling_falling_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_falling_rate);
    const rolling_toggle_rate = try (try rolling_transitions.column("actual_rolling_toggle_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_toggle_rate);
    const rolling_transition_validity = try (try rolling_transitions.column("actual_rolling_toggle_rate")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(rolling_transition_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 2, 2, 1 }, rolling_transition_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 1, 1 }, rolling_rising_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 1, 1, 0 }, rolling_falling_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 2, 2, 1 }, rolling_toggle_count);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, true, false }, rolling_transition_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_rising_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_rising_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_falling_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_falling_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_toggle_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_toggle_rate[3], 1e-12);

    var expanding_transitions = try label_table.expandingBoolTransitionProfile("actual", "actual", .{ .periods = 1 }, .{ .min_periods = 2 });
    defer expanding_transitions.deinit();
    try std.testing.expectEqual(@as(usize, 9), expanding_transitions.width());
    const expanding_transition_count = try (try expanding_transitions.column("actual_expanding_transition_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(expanding_transition_count);
    const expanding_rising_count = try (try expanding_transitions.column("actual_expanding_rising_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(expanding_rising_count);
    const expanding_falling_count = try (try expanding_transitions.column("actual_expanding_falling_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(expanding_falling_count);
    const expanding_toggle_count = try (try expanding_transitions.column("actual_expanding_toggle_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(expanding_toggle_count);
    const expanding_rising_rate = try (try expanding_transitions.column("actual_expanding_rising_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_rising_rate);
    const expanding_falling_rate = try (try expanding_transitions.column("actual_expanding_falling_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_falling_rate);
    const expanding_toggle_rate = try (try expanding_transitions.column("actual_expanding_toggle_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_toggle_rate);
    const expanding_transition_validity = try (try expanding_transitions.column("actual_expanding_toggle_rate")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(expanding_transition_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 2, 2, 2 }, expanding_transition_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 1, 1 }, expanding_rising_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 1, 1, 1 }, expanding_falling_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 2, 2, 2 }, expanding_toggle_count);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, true, true }, expanding_transition_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_rising_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_rising_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_rising_rate[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_falling_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_falling_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_falling_rate[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), expanding_toggle_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), expanding_toggle_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), expanding_toggle_rate[4], 1e-12);

    var fast = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, 3.0, 2.0, 5.0, 4.0, 6.0 }, &.{ true, true, true, true, false, true }, .cpu);
    defer fast.deinit();
    var slow = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 2.0, 2.0, 4.0, 5.0, 0.0 }, .cpu);
    defer slow.deinit();
    var signal_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "fast", .data = fast },
        .{ .name = "slow", .data = slow },
    });
    defer signal_table.deinit();

    var cross = try signal_table.crossoverProfile("fast", "slow", "fast_slow", .{ .periods = 1 });
    defer cross.deinit();
    try std.testing.expectEqual(@as(usize, 6), cross.width());
    const spread = try (try cross.column("fast_slow_spread")).f64.toOwnedSlice(gpa);
    defer gpa.free(spread);
    const ratio = try (try cross.column("fast_slow_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio);
    const cross_above = try (try cross.column("fast_slow_cross_above")).bool.toOwnedSlice(gpa);
    defer gpa.free(cross_above);
    const cross_below = try (try cross.column("fast_slow_cross_below")).bool.toOwnedSlice(gpa);
    defer gpa.free(cross_below);
    const spread_validity = try (try cross.column("fast_slow_spread")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(spread_validity);
    const cross_validity = try (try cross.column("fast_slow_cross_above")).bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(cross_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, false, true }, spread_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, false, false }, cross_validity);
    try std.testing.expectApproxEqAbs(@as(f64, -1.0), spread[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), spread[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), spread[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), spread[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 6.0), spread[5], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), ratio[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), ratio[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), ratio[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.25), ratio[3], 1e-12);
    try std.testing.expect(std.math.isNan(ratio[5]));
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true, false, false }, cross_above);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false, false, false }, cross_below);

    var rolling_cross = try signal_table.rollingCrossoverProfile("fast", "slow", "fast_slow", .{ .periods = 1 }, .{ .window = 3, .min_periods = 2 });
    defer rolling_cross.deinit();
    try std.testing.expectEqual(@as(usize, 8), rolling_cross.width());
    const rolling_cross_count = try (try rolling_cross.column("fast_slow_rolling_cross_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(rolling_cross_count);
    const rolling_cross_above_count = try (try rolling_cross.column("fast_slow_rolling_cross_above_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(rolling_cross_above_count);
    const rolling_cross_below_count = try (try rolling_cross.column("fast_slow_rolling_cross_below_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(rolling_cross_below_count);
    const rolling_cross_above_rate = try (try rolling_cross.column("fast_slow_rolling_cross_above_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_cross_above_rate);
    const rolling_cross_below_rate = try (try rolling_cross.column("fast_slow_rolling_cross_below_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_cross_below_rate);
    const rolling_mean_abs_spread = try (try rolling_cross.column("fast_slow_rolling_mean_abs_spread")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_mean_abs_spread);
    const rolling_cross_validity = try (try rolling_cross.column("fast_slow_rolling_cross_above_rate")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(rolling_cross_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 3, 2, 2 }, rolling_cross_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 1, 2, 1, 1 }, rolling_cross_above_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0, 0, 0 }, rolling_cross_below_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true, true }, rolling_cross_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_cross_above_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), rolling_cross_above_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), rolling_cross_above_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_cross_above_rate[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_cross_above_rate[5], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), rolling_cross_below_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_mean_abs_spread[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), rolling_mean_abs_spread[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), rolling_mean_abs_spread[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_mean_abs_spread[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.5), rolling_mean_abs_spread[5], 1e-12);

    var expanding_cross = try signal_table.expandingCrossoverProfile("fast", "slow", "fast_slow", .{ .periods = 1 }, .{ .min_periods = 2 });
    defer expanding_cross.deinit();
    try std.testing.expectEqual(@as(usize, 8), expanding_cross.width());
    const expanding_cross_count = try (try expanding_cross.column("fast_slow_expanding_cross_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(expanding_cross_count);
    const expanding_cross_above_count = try (try expanding_cross.column("fast_slow_expanding_cross_above_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(expanding_cross_above_count);
    const expanding_cross_below_count = try (try expanding_cross.column("fast_slow_expanding_cross_below_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(expanding_cross_below_count);
    const expanding_cross_above_rate = try (try expanding_cross.column("fast_slow_expanding_cross_above_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_cross_above_rate);
    const expanding_cross_below_rate = try (try expanding_cross.column("fast_slow_expanding_cross_below_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_cross_below_rate);
    const expanding_mean_abs_spread = try (try expanding_cross.column("fast_slow_expanding_mean_abs_spread")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_mean_abs_spread);
    const expanding_cross_validity = try (try expanding_cross.column("fast_slow_expanding_cross_above_rate")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(expanding_cross_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4, 4, 5 }, expanding_cross_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 1, 2, 2, 2 }, expanding_cross_above_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0, 0, 0 }, expanding_cross_below_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true, true }, expanding_cross_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_cross_above_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), expanding_cross_above_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_cross_above_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_cross_above_rate[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.4), expanding_cross_above_rate[5], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), expanding_cross_below_rate[5], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), expanding_mean_abs_spread[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), expanding_mean_abs_spread[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), expanding_mean_abs_spread[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), expanding_mean_abs_spread[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.8), expanding_mean_abs_spread[5], 1e-12);

    var corr = try signal_table.rollingCorrelationProfile("fast", "slow", "fast_slow", .{ .window = 3, .min_periods = 2 });
    defer corr.deinit();
    try std.testing.expectEqual(@as(usize, 6), corr.width());
    const pair_count = try (try corr.column("fast_slow_rolling_pair_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(pair_count);
    const covariance = try (try corr.column("fast_slow_rolling_covariance")).f64.toOwnedSlice(gpa);
    defer gpa.free(covariance);
    const correlation = try (try corr.column("fast_slow_rolling_correlation")).f64.toOwnedSlice(gpa);
    defer gpa.free(correlation);
    const beta = try (try corr.column("fast_slow_rolling_beta")).f64.toOwnedSlice(gpa);
    defer gpa.free(beta);
    const corr_validity = try (try corr.column("fast_slow_rolling_correlation")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(corr_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 3, 2, 2 }, pair_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true, true }, corr_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), covariance[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), covariance[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.1111111111111107), covariance[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), covariance[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -1.0), covariance[5], 1e-12);
    try std.testing.expect(std.math.isNan(correlation[1]));
    try std.testing.expect(std.math.isNan(correlation[2]));
    try std.testing.expectApproxEqAbs(@as(f64, 0.944911182523068), correlation[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), correlation[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -1.0), correlation[5], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), beta[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.714285714285715), beta[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), beta[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -4.0), beta[5], 1e-12);

    var fit_x = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, 2.0, 3.0, 4.0, 5.0 }, &.{ true, true, true, true, false }, .cpu);
    defer fit_x.deinit();
    var fit_y = try DeviceColumn.fromSlice(f64, gpa, &.{ 3.0, 5.0, 8.0, 9.0, 0.0 }, .cpu);
    defer fit_y.deinit();
    var fit_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "x", .data = fit_x },
        .{ .name = "y", .data = fit_y },
    });
    defer fit_table.deinit();

    var fitted_table = try fit_table.linearFitProfile("x", "y", "xy", .{ .min_periods = 3 });
    defer fitted_table.deinit();
    try std.testing.expectEqual(@as(usize, 6), fitted_table.width());
    const fitted = try (try fitted_table.column("xy_fitted")).f64.toOwnedSlice(gpa);
    defer gpa.free(fitted);
    const residual = try (try fitted_table.column("xy_residual")).f64.toOwnedSlice(gpa);
    defer gpa.free(residual);
    const residual_z = try (try fitted_table.column("xy_residual_zscore")).f64.toOwnedSlice(gpa);
    defer gpa.free(residual_z);
    const slope_values = try (try fitted_table.column("xy_slope")).f64.toOwnedSlice(gpa);
    defer gpa.free(slope_values);
    const fit_validity = try (try fitted_table.column("xy_fitted")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(fit_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, false }, fit_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 3.1), fitted[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.2), fitted[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 7.3), fitted[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 9.4), fitted[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.1), residual[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.2), residual[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.7), residual[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.4), residual[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.23904572186687895), residual_z[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.4780914437337579), residual_z[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.6733200530681511), residual_z[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.9561828874675167), residual_z[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.1), slope_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.1), slope_values[3], 1e-12);

    var rolling_fit = try fit_table.rollingLinearFitProfile("x", "y", "xy", .{ .window = 3, .min_periods = 2 });
    defer rolling_fit.deinit();
    try std.testing.expectEqual(@as(usize, 8), rolling_fit.width());
    const rolling_fit_count = try (try rolling_fit.column("xy_rolling_pair_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(rolling_fit_count);
    const rolling_slope = try (try rolling_fit.column("xy_rolling_slope")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_slope);
    const rolling_intercept = try (try rolling_fit.column("xy_rolling_intercept")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_intercept);
    const rolling_fitted = try (try rolling_fit.column("xy_rolling_fitted")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_fitted);
    const rolling_residual = try (try rolling_fit.column("xy_rolling_residual")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_residual);
    const rolling_residual_z = try (try rolling_fit.column("xy_rolling_residual_zscore")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_residual_z);
    const rolling_slope_validity = try (try rolling_fit.column("xy_rolling_slope")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(rolling_slope_validity);
    const rolling_fitted_validity = try (try rolling_fit.column("xy_rolling_fitted")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(rolling_fitted_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 3, 2 }, rolling_fit_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true }, rolling_slope_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, false }, rolling_fitted_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), rolling_slope[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), rolling_slope[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), rolling_slope[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_slope[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_intercept[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), rolling_intercept[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0 / 3.0), rolling_intercept[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), rolling_intercept[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), rolling_fitted[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 47.0 / 6.0), rolling_fitted[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 28.0 / 3.0), rolling_fitted[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), rolling_residual[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 6.0), rolling_residual[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -1.0 / 3.0), rolling_residual[3], 1e-12);
    try std.testing.expect(std.math.isNan(rolling_residual_z[1]));
    try std.testing.expectApproxEqAbs(@as(f64, 0.7071067811865475), rolling_residual_z[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.7071067811865475), rolling_residual_z[3], 1e-12);

    var expanding_corr = try fit_table.expandingCorrelationProfile("x", "y", "xy_corr", .{ .min_periods = 2 });
    defer expanding_corr.deinit();
    try std.testing.expectEqual(@as(usize, 6), expanding_corr.width());
    const expanding_pair_count = try (try expanding_corr.column("xy_corr_expanding_pair_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(expanding_pair_count);
    const expanding_cov = try (try expanding_corr.column("xy_corr_expanding_covariance")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_cov);
    const expanding_corr_values = try (try expanding_corr.column("xy_corr_expanding_correlation")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_corr_values);
    const expanding_beta = try (try expanding_corr.column("xy_corr_expanding_beta")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_beta);
    const expanding_corr_validity = try (try expanding_corr.column("xy_corr_expanding_correlation")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(expanding_corr_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4, 4 }, expanding_pair_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true }, expanding_corr_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_cov[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.6666666666666679), expanding_cov[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.625), expanding_cov[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), expanding_corr_values[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.9933992677987834), expanding_corr_values[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.9844951849708403), expanding_corr_values[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), expanding_beta[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), expanding_beta[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.1), expanding_beta[3], 1e-12);

    var expanding_fit = try fit_table.expandingLinearFitProfile("x", "y", "xy_fit", .{ .min_periods = 2 });
    defer expanding_fit.deinit();
    try std.testing.expectEqual(@as(usize, 8), expanding_fit.width());
    const expanding_fit_count = try (try expanding_fit.column("xy_fit_expanding_pair_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(expanding_fit_count);
    const expanding_slope = try (try expanding_fit.column("xy_fit_expanding_slope")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_slope);
    const expanding_intercept = try (try expanding_fit.column("xy_fit_expanding_intercept")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_intercept);
    const expanding_fitted = try (try expanding_fit.column("xy_fit_expanding_fitted")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_fitted);
    const expanding_residual = try (try expanding_fit.column("xy_fit_expanding_residual")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_residual);
    const expanding_residual_z = try (try expanding_fit.column("xy_fit_expanding_residual_zscore")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_residual_z);
    const expanding_fit_validity = try (try expanding_fit.column("xy_fit_expanding_slope")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(expanding_fit_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4, 4 }, expanding_fit_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true }, expanding_fit_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), expanding_slope[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), expanding_slope[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.1), expanding_slope[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), expanding_intercept[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), expanding_intercept[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), expanding_intercept[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), expanding_fitted[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 47.0 / 6.0), expanding_fitted[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 9.4), expanding_fitted[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), expanding_residual[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 6.0), expanding_residual[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.4), expanding_residual[3], 1e-12);
    try std.testing.expect(std.math.isNan(expanding_residual_z[1]));
    try std.testing.expectApproxEqAbs(@as(f64, 0.7071067811865491), expanding_residual_z[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.9561828874675161), expanding_residual_z[3], 1e-12);

    var errors = try fit_table.errorProfile("y", "x", "yx");
    defer errors.deinit();
    try std.testing.expectEqual(@as(usize, 7), errors.width());
    const error_values = try (try errors.column("yx_error")).f64.toOwnedSlice(gpa);
    defer gpa.free(error_values);
    const abs_error_values = try (try errors.column("yx_abs_error")).f64.toOwnedSlice(gpa);
    defer gpa.free(abs_error_values);
    const squared_error_values = try (try errors.column("yx_squared_error")).f64.toOwnedSlice(gpa);
    defer gpa.free(squared_error_values);
    const ape_values = try (try errors.column("yx_ape")).f64.toOwnedSlice(gpa);
    defer gpa.free(ape_values);
    const smape_values = try (try errors.column("yx_smape")).f64.toOwnedSlice(gpa);
    defer gpa.free(smape_values);
    const error_validity = try (try errors.column("yx_error")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(error_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, false }, error_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), error_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), error_values[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), error_values[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), error_values[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), abs_error_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 25.0), squared_error_values[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), ape_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 9.0), ape_values[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), smape_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0 / 11.0), smape_values[2], 1e-12);

    var rolling_errors = try fit_table.rollingErrorProfile("y", "x", "yx_roll", .{ .window = 3, .min_periods = 2 });
    defer rolling_errors.deinit();
    try std.testing.expectEqual(@as(usize, 7), rolling_errors.width());
    const rolling_error_count = try (try rolling_errors.column("yx_roll_rolling_error_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(rolling_error_count);
    const rolling_mae = try (try rolling_errors.column("yx_roll_rolling_mae")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_mae);
    const rolling_rmse = try (try rolling_errors.column("yx_roll_rolling_rmse")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_rmse);
    const rolling_mape = try (try rolling_errors.column("yx_roll_rolling_mape")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_mape);
    const rolling_smape = try (try rolling_errors.column("yx_roll_rolling_smape")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_smape);
    const rolling_error_validity = try (try rolling_errors.column("yx_roll_rolling_mae")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(rolling_error_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 3, 2 }, rolling_error_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true }, rolling_error_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), rolling_mae[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0 / 3.0), rolling_mae[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 13.0 / 3.0), rolling_mae[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.5495097567963922), rolling_rmse[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.559026084010437), rolling_rmse[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.43471156521669), rolling_rmse[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.6333333333333333), rolling_mape[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.6305555555555555), rolling_mape[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5935185185185184), rolling_mape[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.9285714285714286), rolling_smape[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.9220779220779219), rolling_smape[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.8451548451548452), rolling_smape[3], 1e-12);

    var expanding_errors = try fit_table.expandingErrorProfile("y", "x", "yx_expand", .{ .min_periods = 2 });
    defer expanding_errors.deinit();
    try std.testing.expectEqual(@as(usize, 7), expanding_errors.width());
    const expanding_error_count = try (try expanding_errors.column("yx_expand_expanding_error_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(expanding_error_count);
    const expanding_mae = try (try expanding_errors.column("yx_expand_expanding_mae")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_mae);
    const expanding_rmse = try (try expanding_errors.column("yx_expand_expanding_rmse")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_rmse);
    const expanding_mape = try (try expanding_errors.column("yx_expand_expanding_mape")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_mape);
    const expanding_smape = try (try expanding_errors.column("yx_expand_expanding_smape")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_smape);
    const expanding_error_validity = try (try expanding_errors.column("yx_expand_expanding_mae")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(expanding_error_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4, 4 }, expanding_error_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true }, expanding_error_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), expanding_mae[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0 / 3.0), expanding_mae[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.75), expanding_mae[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.5495097567963922), expanding_rmse[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.559026084010437), expanding_rmse[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.968626966596886), expanding_rmse[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.6333333333333333), expanding_mape[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.6305555555555555), expanding_mape[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.6118055555555555), expanding_mape[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.9285714285714286), expanding_smape[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.9220779220779219), expanding_smape[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.8838661338661339), expanding_smape[3], 1e-12);

    var bucketed = try signal_table.bucketProfile("fast", "fast", .{ .buckets = 3, .lower_quantile = 0.34, .upper_quantile = 0.84 });
    defer bucketed.deinit();
    try std.testing.expectEqual(@as(usize, 6), bucketed.width());
    const ecdf = try (try bucketed.column("fast_ecdf")).f64.toOwnedSlice(gpa);
    defer gpa.free(ecdf);
    const bucket = try (try bucketed.column("fast_bucket")).i64.toOwnedSlice(gpa);
    defer gpa.free(bucket);
    const lower_tail = try (try bucketed.column("fast_lower_tail")).bool.toOwnedSlice(gpa);
    defer gpa.free(lower_tail);
    const upper_tail = try (try bucketed.column("fast_upper_tail")).bool.toOwnedSlice(gpa);
    defer gpa.free(upper_tail);
    const bucket_validity = try (try bucketed.column("fast_bucket")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(bucket_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, false, true }, bucket_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.2), ecdf[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.6), ecdf[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.4), ecdf[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.8), ecdf[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), ecdf[5], 1e-12);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 1, 0, 2 }, bucket);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false, false, false }, lower_tail);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false, false, true }, upper_tail);
}

test "device dataframe rolling bool profile handles nullable windows" {
    const gpa = std.testing.allocator;

    var active = try DeviceColumn.fromSliceWithValidity(bool, gpa, &.{ true, false, true, true, true, false }, &.{ true, true, false, true, true, true }, .cpu);
    defer active.deinit();
    var id = try DeviceColumn.fromSlice(i64, gpa, &.{ 1, 2, 3, 4, 5, 6 }, .cpu);
    defer id.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "active", .data = active },
        .{ .name = "id", .data = id },
    });
    defer table.deinit();

    var profiled = try table.rollingBoolProfile("active", "active", .{ .window = 3, .min_periods = 2 });
    defer profiled.deinit();
    try std.testing.expectEqual(@as(usize, 7), profiled.width());

    const true_count = try (try profiled.column("active_rolling_true_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(true_count);
    const false_count = try (try profiled.column("active_rolling_false_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(false_count);
    const true_rate = try (try profiled.column("active_rolling_true_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(true_rate);
    const rolling_any = try (try profiled.column("active_rolling_any")).bool.toOwnedSlice(gpa);
    defer gpa.free(rolling_any);
    const rolling_all = try (try profiled.column("active_rolling_all")).bool.toOwnedSlice(gpa);
    defer gpa.free(rolling_all);
    const bool_validity = try (try profiled.column("active_rolling_true_rate")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(bool_validity);

    try std.testing.expectEqualSlices(i64, &.{ 1, 1, 1, 1, 2, 2 }, true_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 1, 1, 0, 1 }, false_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true, true, true }, bool_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), true_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), true_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), true_rate[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), true_rate[5], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true, true, true }, rolling_any);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false, true, false }, rolling_all);

    var expanding_bool = try table.expandingBoolProfile("active", "active", .{ .min_periods = 2 });
    defer expanding_bool.deinit();
    try std.testing.expectEqual(@as(usize, 7), expanding_bool.width());
    const expanding_true_count = try (try expanding_bool.column("active_expanding_true_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(expanding_true_count);
    const expanding_false_count = try (try expanding_bool.column("active_expanding_false_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(expanding_false_count);
    const expanding_true_rate = try (try expanding_bool.column("active_expanding_true_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_true_rate);
    const expanding_any = try (try expanding_bool.column("active_expanding_any")).bool.toOwnedSlice(gpa);
    defer gpa.free(expanding_any);
    const expanding_all = try (try expanding_bool.column("active_expanding_all")).bool.toOwnedSlice(gpa);
    defer gpa.free(expanding_all);
    const expanding_bool_validity = try (try expanding_bool.column("active_expanding_true_rate")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(expanding_bool_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 1, 1, 2, 3, 3 }, expanding_true_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 1, 1, 1, 2 }, expanding_false_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true, true, true }, expanding_bool_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_true_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), expanding_true_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), expanding_true_rate[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.6), expanding_true_rate[5], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true, true, true }, expanding_any);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false, false, false }, expanding_all);
}

test "device dataframe groupby aggregations on fixed-width columns" {
    const gpa = std.testing.allocator;

    var key = try DeviceColumn.fromSliceWithValidity(i32, gpa, &.{ 1, 2, 1, 3, 2, 1 }, &.{ true, true, true, false, true, true }, .cpu);
    defer key.deinit();
    var sales = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 2.0, 3.0, 5.0, 7.0, 11.0, 13.0 }, &.{ true, true, false, true, true, true }, .cpu);
    defer sales.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "store", .data = key },
        .{ .name = "sales", .data = sales },
    });
    defer table.deinit();

    var counted = try table.groupByCount("store", "rows");
    defer counted.deinit();
    try std.testing.expectEqual(@as(usize, 2), counted.height());
    const count_keys = try (try counted.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(count_keys);
    const counts = try (try counted.column("rows")).i64.toOwnedSlice(gpa);
    defer gpa.free(counts);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2 }, count_keys);
    try std.testing.expectEqualSlices(i64, &.{ 3, 2 }, counts);

    var summed = try table.groupBySum("store", "sales", "sales_sum");
    defer summed.deinit();
    try std.testing.expectEqual(@as(usize, 2), summed.height());
    const sum_keys = try (try summed.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(sum_keys);
    const sums = try (try summed.column("sales_sum")).f64.toOwnedSlice(gpa);
    defer gpa.free(sums);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2 }, sum_keys);
    try std.testing.expectEqualSlices(f64, &.{ 15.0, 14.0 }, sums);

    var mins = try table.groupByMin("store", "sales", "sales_min");
    defer mins.deinit();
    const min_values = try (try mins.column("sales_min")).f64.toOwnedSlice(gpa);
    defer gpa.free(min_values);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 3.0 }, min_values);

    var maxes = try table.groupByMax("store", "sales", "sales_max");
    defer maxes.deinit();
    const max_values = try (try maxes.column("sales_max")).f64.toOwnedSlice(gpa);
    defer gpa.free(max_values);
    try std.testing.expectEqualSlices(f64, &.{ 13.0, 11.0 }, max_values);

    var means = try table.groupByMean("store", "sales", "sales_mean");
    defer means.deinit();
    const mean_values = try (try means.column("sales_mean")).f64.toOwnedSlice(gpa);
    defer gpa.free(mean_values);
    try std.testing.expectEqualSlices(f64, &.{ 7.5, 7.0 }, mean_values);

    var stats = try table.groupByStats("store", "sales", "sales");
    defer stats.deinit();
    try std.testing.expectEqual(@as(usize, 6), stats.width());
    const stats_keys = try (try stats.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(stats_keys);
    const stats_counts = try (try stats.column("sales_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(stats_counts);
    const stats_sums = try (try stats.column("sales_sum")).f64.toOwnedSlice(gpa);
    defer gpa.free(stats_sums);
    const stats_mins = try (try stats.column("sales_min")).f64.toOwnedSlice(gpa);
    defer gpa.free(stats_mins);
    const stats_maxes = try (try stats.column("sales_max")).f64.toOwnedSlice(gpa);
    defer gpa.free(stats_maxes);
    const stats_means = try (try stats.column("sales_mean")).f64.toOwnedSlice(gpa);
    defer gpa.free(stats_means);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2 }, stats_keys);
    try std.testing.expectEqualSlices(i64, &.{ 2, 2 }, stats_counts);
    try std.testing.expectEqualSlices(f64, &.{ 15.0, 14.0 }, stats_sums);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 3.0 }, stats_mins);
    try std.testing.expectEqualSlices(f64, &.{ 13.0, 11.0 }, stats_maxes);
    try std.testing.expectEqualSlices(f64, &.{ 7.5, 7.0 }, stats_means);

    var profile = try table.groupByProfile("store", "sales", "sales");
    defer profile.deinit();
    try std.testing.expectEqual(@as(usize, 8), profile.width());
    const profile_keys = try (try profile.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(profile_keys);
    const profile_counts = try (try profile.column("sales_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(profile_counts);
    const profile_sums = try (try profile.column("sales_sum")).f64.toOwnedSlice(gpa);
    defer gpa.free(profile_sums);
    const profile_variances = try (try profile.column("sales_variance")).f64.toOwnedSlice(gpa);
    defer gpa.free(profile_variances);
    const profile_stddevs = try (try profile.column("sales_stddev")).f64.toOwnedSlice(gpa);
    defer gpa.free(profile_stddevs);
    const profile_skewnesses = try (try profile.column("sales_skewness")).f64.toOwnedSlice(gpa);
    defer gpa.free(profile_skewnesses);
    const profile_kurtoses = try (try profile.column("sales_kurtosis")).f64.toOwnedSlice(gpa);
    defer gpa.free(profile_kurtoses);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2 }, profile_keys);
    try std.testing.expectEqualSlices(i64, &.{ 2, 2 }, profile_counts);
    try std.testing.expectEqualSlices(f64, &.{ 15.0, 14.0 }, profile_sums);
    try std.testing.expectApproxEqAbs(@as(f64, 30.25), profile_variances[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 16.0), profile_variances[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.5), profile_stddevs[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), profile_stddevs[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), profile_skewnesses[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), profile_skewnesses[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -2.0), profile_kurtoses[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -2.0), profile_kurtoses[1], 1e-12);

    var keyed = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 1, 1, 2, 2, 2 }, .cpu);
    defer keyed.deinit();
    var day = try DeviceColumn.fromSlice(i32, gpa, &.{ 10, 10, 11, 10, 10, 11 }, .cpu);
    defer day.deinit();
    var amount = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, 2.0, 9.0, 4.0, 6.0, 12.0 }, &.{ true, true, true, true, false, true }, .cpu);
    defer amount.deinit();
    var multi = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "store", .data = keyed },
        .{ .name = "day", .data = day },
        .{ .name = "amount", .data = amount },
    });
    defer multi.deinit();

    var multi_stats = try multi.groupByStatsOn(&.{ "store", "day" }, "amount", "amount");
    defer multi_stats.deinit();
    try std.testing.expectEqual(@as(usize, 7), multi_stats.width());
    try std.testing.expectEqual(@as(usize, 4), multi_stats.height());
    const ms_store = try (try multi_stats.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(ms_store);
    const ms_day = try (try multi_stats.column("day")).i32.toOwnedSlice(gpa);
    defer gpa.free(ms_day);
    const ms_count = try (try multi_stats.column("amount_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(ms_count);
    const ms_sum = try (try multi_stats.column("amount_sum")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_sum);
    const ms_mean = try (try multi_stats.column("amount_mean")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_mean);
    try std.testing.expectEqualSlices(i32, &.{ 1, 1, 2, 2 }, ms_store);
    try std.testing.expectEqualSlices(i32, &.{ 10, 11, 10, 11 }, ms_day);
    try std.testing.expectEqualSlices(i64, &.{ 2, 1, 1, 1 }, ms_count);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 9.0, 4.0, 12.0 }, ms_sum);
    try std.testing.expectEqualSlices(f64, &.{ 1.5, 9.0, 4.0, 12.0 }, ms_mean);

    var multi_profile = try multi.groupByProfileOn(&.{ "store", "day" }, "amount", "amount");
    defer multi_profile.deinit();
    try std.testing.expectEqual(@as(usize, 9), multi_profile.width());
    try std.testing.expectEqual(@as(usize, 4), multi_profile.height());
    const mp_store = try (try multi_profile.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(mp_store);
    const mp_day = try (try multi_profile.column("day")).i32.toOwnedSlice(gpa);
    defer gpa.free(mp_day);
    const mp_count = try (try multi_profile.column("amount_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(mp_count);
    const mp_variance = try (try multi_profile.column("amount_variance")).f64.toOwnedSlice(gpa);
    defer gpa.free(mp_variance);
    const mp_stddev = try (try multi_profile.column("amount_stddev")).f64.toOwnedSlice(gpa);
    defer gpa.free(mp_stddev);
    const mp_skewness = try (try multi_profile.column("amount_skewness")).f64.toOwnedSlice(gpa);
    defer gpa.free(mp_skewness);
    const mp_kurtosis = try (try multi_profile.column("amount_kurtosis")).f64.toOwnedSlice(gpa);
    defer gpa.free(mp_kurtosis);
    try std.testing.expectEqualSlices(i32, &.{ 1, 1, 2, 2 }, mp_store);
    try std.testing.expectEqualSlices(i32, &.{ 10, 11, 10, 11 }, mp_day);
    try std.testing.expectEqualSlices(i64, &.{ 2, 1, 1, 1 }, mp_count);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), mp_variance[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), mp_stddev[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), mp_skewness[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -2.0), mp_kurtosis[0], 1e-12);
    try std.testing.expect(std.math.isNan(mp_skewness[1]));
    try std.testing.expect(std.math.isNan(mp_kurtosis[1]));
}

test "device dataframe inner joins on fixed-width keys" {
    const gpa = std.testing.allocator;

    var left_id = try DeviceColumn.fromSliceWithValidity(i32, gpa, &.{ 1, 2, 3, 2, 9 }, &.{ true, true, true, true, false }, .cpu);
    defer left_id.deinit();
    var left_value = try DeviceColumn.fromSlice(f64, gpa, &.{ 10.0, 20.0, 30.0, 21.0, 90.0 }, .cpu);
    defer left_value.deinit();

    var right_id = try DeviceColumn.fromSliceWithValidity(i32, gpa, &.{ 2, 3, 2, 4, 9 }, &.{ true, true, true, true, false }, .cpu);
    defer right_id.deinit();
    var right_value = try DeviceColumn.fromSlice(f64, gpa, &.{ 200.0, 300.0, 201.0, 400.0, 900.0 }, .cpu);
    defer right_value.deinit();
    var right_label = try DeviceColumn.fromSlice(i64, gpa, &.{ 20, 30, 21, 40, 90 }, .cpu);
    defer right_label.deinit();

    var left = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "id", .data = left_id },
        .{ .name = "value", .data = left_value },
    });
    defer left.deinit();
    var right = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "id", .data = right_id },
        .{ .name = "value", .data = right_value },
        .{ .name = "label", .data = right_label },
    });
    defer right.deinit();

    var joined = try left.innerJoin(right, "id", "id", .{});
    defer joined.deinit();
    try std.testing.expectEqual(@as(usize, 5), joined.height());
    try std.testing.expectEqual(@as(usize, 4), joined.width());
    try std.testing.expectEqual(DeviceDType.f64, try joined.columnDType("value"));
    try std.testing.expectEqual(DeviceDType.f64, try joined.columnDType("value_right"));

    const ids = try (try joined.column("id")).i32.toOwnedSlice(gpa);
    defer gpa.free(ids);
    const left_values = try (try joined.column("value")).f64.toOwnedSlice(gpa);
    defer gpa.free(left_values);
    const right_values = try (try joined.column("value_right")).f64.toOwnedSlice(gpa);
    defer gpa.free(right_values);
    const labels = try (try joined.column("label")).i64.toOwnedSlice(gpa);
    defer gpa.free(labels);

    try std.testing.expectEqualSlices(i32, &.{ 2, 2, 3, 2, 2 }, ids);
    try std.testing.expectEqualSlices(f64, &.{ 20.0, 20.0, 30.0, 21.0, 21.0 }, left_values);
    try std.testing.expectEqualSlices(f64, &.{ 200.0, 201.0, 300.0, 200.0, 201.0 }, right_values);
    try std.testing.expectEqualSlices(i64, &.{ 20, 21, 30, 20, 21 }, labels);
}

test "device dataframe inner joins on multiple fixed-width keys" {
    const gpa = std.testing.allocator;

    var left_store = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 1, 2, 2 }, .cpu);
    defer left_store.deinit();
    var left_day = try DeviceColumn.fromSlice(i32, gpa, &.{ 10, 11, 10, 12 }, .cpu);
    defer left_day.deinit();
    var left_sales = try DeviceColumn.fromSlice(f64, gpa, &.{ 100.0, 110.0, 200.0, 220.0 }, .cpu);
    defer left_sales.deinit();

    var right_store = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 2, 2, 3 }, .cpu);
    defer right_store.deinit();
    var right_day = try DeviceColumn.fromSlice(i32, gpa, &.{ 10, 10, 13, 10 }, .cpu);
    defer right_day.deinit();
    var right_region = try DeviceColumn.fromSlice(i64, gpa, &.{ 7, 8, 9, 10 }, .cpu);
    defer right_region.deinit();

    var left = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "store", .data = left_store },
        .{ .name = "day", .data = left_day },
        .{ .name = "sales", .data = left_sales },
    });
    defer left.deinit();
    var right = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "store", .data = right_store },
        .{ .name = "day", .data = right_day },
        .{ .name = "region", .data = right_region },
    });
    defer right.deinit();

    var joined = try left.innerJoinOn(right, &.{ "store", "day" }, &.{ "store", "day" }, .{});
    defer joined.deinit();
    try std.testing.expectEqual(@as(usize, 2), joined.height());
    try std.testing.expectEqual(@as(usize, 4), joined.width());

    const stores = try (try joined.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(stores);
    const days = try (try joined.column("day")).i32.toOwnedSlice(gpa);
    defer gpa.free(days);
    const sales = try (try joined.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(sales);
    const regions = try (try joined.column("region")).i64.toOwnedSlice(gpa);
    defer gpa.free(regions);

    try std.testing.expectEqualSlices(i32, &.{ 1, 2 }, stores);
    try std.testing.expectEqualSlices(i32, &.{ 10, 10 }, days);
    try std.testing.expectEqualSlices(f64, &.{ 100.0, 200.0 }, sales);
    try std.testing.expectEqualSlices(i64, &.{ 7, 8 }, regions);
}

test "device dataframe left joins on multiple fixed-width keys" {
    const gpa = std.testing.allocator;

    var left_store = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 1, 2, 2 }, .cpu);
    defer left_store.deinit();
    var left_day = try DeviceColumn.fromSliceWithValidity(i32, gpa, &.{ 10, 11, 10, 12 }, &.{ true, true, true, false }, .cpu);
    defer left_day.deinit();
    var left_sales = try DeviceColumn.fromSlice(f64, gpa, &.{ 100.0, 110.0, 200.0, 220.0 }, .cpu);
    defer left_sales.deinit();

    var right_store = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 2, 3 }, .cpu);
    defer right_store.deinit();
    var right_day = try DeviceColumn.fromSlice(i32, gpa, &.{ 10, 10, 10 }, .cpu);
    defer right_day.deinit();
    var right_region = try DeviceColumn.fromSlice(i64, gpa, &.{ 7, 8, 10 }, .cpu);
    defer right_region.deinit();

    var left = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "store", .data = left_store },
        .{ .name = "day", .data = left_day },
        .{ .name = "sales", .data = left_sales },
    });
    defer left.deinit();
    var right = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "store", .data = right_store },
        .{ .name = "day", .data = right_day },
        .{ .name = "region", .data = right_region },
    });
    defer right.deinit();

    var joined = try left.leftJoinOn(right, &.{ "store", "day" }, &.{ "store", "day" }, .{});
    defer joined.deinit();
    try std.testing.expectEqual(@as(usize, 4), joined.height());
    try std.testing.expectEqual(@as(usize, 4), joined.width());

    const stores = try (try joined.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(stores);
    const days = try (try joined.column("day")).i32.toOwnedSlice(gpa);
    defer gpa.free(days);
    const day_validity = try (try joined.column("day")).i32.validity.?.toOwnedSlice(gpa);
    defer gpa.free(day_validity);
    const regions = try (try joined.column("region")).i64.toOwnedSlice(gpa);
    defer gpa.free(regions);
    const region_validity = try (try joined.column("region")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(region_validity);

    try std.testing.expectEqualSlices(i32, &.{ 1, 1, 2, 2 }, stores);
    try std.testing.expectEqualSlices(i32, &.{ 10, 11, 10, 12 }, days);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false }, day_validity);
    try std.testing.expectEqualSlices(i64, &.{ 7, 0, 8, 0 }, regions);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, false }, region_validity);
}

test "device dataframe full joins on multiple fixed-width keys" {
    const gpa = std.testing.allocator;

    var left_store = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 1, 2 }, .cpu);
    defer left_store.deinit();
    var left_day = try DeviceColumn.fromSliceWithValidity(i32, gpa, &.{ 10, 11, 10 }, &.{ true, true, false }, .cpu);
    defer left_day.deinit();
    var left_sales = try DeviceColumn.fromSlice(f64, gpa, &.{ 100.0, 110.0, 200.0 }, .cpu);
    defer left_sales.deinit();

    var right_store = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 2, 3 }, .cpu);
    defer right_store.deinit();
    var right_day = try DeviceColumn.fromSlice(i32, gpa, &.{ 10, 10, 10 }, .cpu);
    defer right_day.deinit();
    var right_region = try DeviceColumn.fromSlice(i64, gpa, &.{ 7, 8, 9 }, .cpu);
    defer right_region.deinit();

    var left = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "store", .data = left_store },
        .{ .name = "day", .data = left_day },
        .{ .name = "sales", .data = left_sales },
    });
    defer left.deinit();
    var right = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "store", .data = right_store },
        .{ .name = "day", .data = right_day },
        .{ .name = "region", .data = right_region },
    });
    defer right.deinit();

    var joined = try left.fullJoinOn(right, &.{ "store", "day" }, &.{ "store", "day" }, .{});
    defer joined.deinit();
    try std.testing.expectEqual(@as(usize, 5), joined.height());
    try std.testing.expectEqual(@as(usize, 4), joined.width());

    const stores = try (try joined.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(stores);
    const days = try (try joined.column("day")).i32.toOwnedSlice(gpa);
    defer gpa.free(days);
    const day_validity = try (try joined.column("day")).i32.validity.?.toOwnedSlice(gpa);
    defer gpa.free(day_validity);
    const sales = try (try joined.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(sales);
    const sales_validity = try (try joined.column("sales")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(sales_validity);
    const regions = try (try joined.column("region")).i64.toOwnedSlice(gpa);
    defer gpa.free(regions);
    const region_validity = try (try joined.column("region")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(region_validity);

    try std.testing.expectEqualSlices(i32, &.{ 1, 1, 2, 2, 3 }, stores);
    try std.testing.expectEqualSlices(i32, &.{ 10, 11, 0, 10, 10 }, days);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true, true }, day_validity);
    try std.testing.expectEqualSlices(f64, &.{ 100.0, 110.0, 200.0, 0.0, 0.0 }, sales);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false, false }, sales_validity);
    try std.testing.expectEqualSlices(i64, &.{ 7, 0, 0, 8, 9 }, regions);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true, true }, region_validity);
}

test "device dataframe semi and anti join on multiple fixed-width keys" {
    const gpa = std.testing.allocator;

    var left_store = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 1, 2, 2, 3 }, .cpu);
    defer left_store.deinit();
    var left_day = try DeviceColumn.fromSliceWithValidity(i32, gpa, &.{ 10, 11, 10, 12, 10 }, &.{ true, true, true, false, true }, .cpu);
    defer left_day.deinit();
    var left_sales = try DeviceColumn.fromSlice(f64, gpa, &.{ 100.0, 110.0, 200.0, 220.0, 300.0 }, .cpu);
    defer left_sales.deinit();

    var right_store = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 2, 4 }, .cpu);
    defer right_store.deinit();
    var right_day = try DeviceColumn.fromSlice(i32, gpa, &.{ 10, 10, 10 }, .cpu);
    defer right_day.deinit();

    var left = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "store", .data = left_store },
        .{ .name = "day", .data = left_day },
        .{ .name = "sales", .data = left_sales },
    });
    defer left.deinit();
    var right = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "store", .data = right_store },
        .{ .name = "day", .data = right_day },
    });
    defer right.deinit();

    var semi = try left.semiJoinOn(right, &.{ "store", "day" }, &.{ "store", "day" });
    defer semi.deinit();
    const semi_store = try (try semi.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(semi_store);
    const semi_sales = try (try semi.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(semi_sales);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2 }, semi_store);
    try std.testing.expectEqualSlices(f64, &.{ 100.0, 200.0 }, semi_sales);

    var anti = try left.antiJoinOn(right, &.{ "store", "day" }, &.{ "store", "day" });
    defer anti.deinit();
    const anti_store = try (try anti.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(anti_store);
    const anti_sales = try (try anti.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(anti_sales);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2, 3 }, anti_store);
    try std.testing.expectEqualSlices(f64, &.{ 110.0, 220.0, 300.0 }, anti_sales);
}

test "device dataframe left joins with nullable unmatched right payloads" {
    const gpa = std.testing.allocator;

    var left_id = try DeviceColumn.fromSliceWithValidity(i32, gpa, &.{ 1, 2, 3, 2, 9 }, &.{ true, true, true, true, false }, .cpu);
    defer left_id.deinit();
    var left_value = try DeviceColumn.fromSlice(f64, gpa, &.{ 10.0, 20.0, 30.0, 21.0, 90.0 }, .cpu);
    defer left_value.deinit();

    var right_id = try DeviceColumn.fromSlice(i32, gpa, &.{ 2, 3 }, .cpu);
    defer right_id.deinit();
    var right_value = try DeviceColumn.fromSlice(f64, gpa, &.{ 200.0, 300.0 }, .cpu);
    defer right_value.deinit();

    var left = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "id", .data = left_id },
        .{ .name = "value", .data = left_value },
    });
    defer left.deinit();
    var right = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "id", .data = right_id },
        .{ .name = "value", .data = right_value },
    });
    defer right.deinit();

    var joined = try left.leftJoin(right, "id", "id", .{});
    defer joined.deinit();
    try std.testing.expectEqual(@as(usize, 5), joined.height());
    try std.testing.expectEqual(DeviceDType.f64, try joined.columnDType("value_right"));

    const ids = try (try joined.column("id")).i32.toOwnedSlice(gpa);
    defer gpa.free(ids);
    const right_values = try (try joined.column("value_right")).f64.toOwnedSlice(gpa);
    defer gpa.free(right_values);
    const right_validity = try (try joined.column("value_right")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(right_validity);

    try std.testing.expectEqualSlices(i32, &.{ 1, 2, 3, 2, 9 }, ids);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 200.0, 300.0, 200.0, 0.0 }, right_values);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, false }, right_validity);
}

test "device dataframe semi and anti joins filter left rows" {
    const gpa = std.testing.allocator;

    var left_id = try DeviceColumn.fromSliceWithValidity(i32, gpa, &.{ 1, 2, 3, 2, 9 }, &.{ true, true, true, true, false }, .cpu);
    defer left_id.deinit();
    var left_value = try DeviceColumn.fromSlice(f64, gpa, &.{ 10.0, 20.0, 30.0, 21.0, 90.0 }, .cpu);
    defer left_value.deinit();

    var right_id = try DeviceColumn.fromSlice(i32, gpa, &.{ 2, 4 }, .cpu);
    defer right_id.deinit();
    var right_value = try DeviceColumn.fromSlice(f64, gpa, &.{ 200.0, 400.0 }, .cpu);
    defer right_value.deinit();

    var left = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "id", .data = left_id },
        .{ .name = "value", .data = left_value },
    });
    defer left.deinit();
    var right = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "id", .data = right_id },
        .{ .name = "value", .data = right_value },
    });
    defer right.deinit();

    var semi = try left.semiJoin(right, "id", "id");
    defer semi.deinit();
    try std.testing.expectEqual(@as(usize, 2), semi.height());
    const semi_ids = try (try semi.column("id")).i32.toOwnedSlice(gpa);
    defer gpa.free(semi_ids);
    const semi_values = try (try semi.column("value")).f64.toOwnedSlice(gpa);
    defer gpa.free(semi_values);
    try std.testing.expectEqualSlices(i32, &.{ 2, 2 }, semi_ids);
    try std.testing.expectEqualSlices(f64, &.{ 20.0, 21.0 }, semi_values);

    var anti = try left.antiJoin(right, "id", "id");
    defer anti.deinit();
    try std.testing.expectEqual(@as(usize, 3), anti.height());
    const anti_ids = try (try anti.column("id")).i32.toOwnedSlice(gpa);
    defer gpa.free(anti_ids);
    const anti_values = try (try anti.column("value")).f64.toOwnedSlice(gpa);
    defer gpa.free(anti_values);
    try std.testing.expectEqualSlices(i32, &.{ 1, 3, 9 }, anti_ids);
    try std.testing.expectEqualSlices(f64, &.{ 10.0, 30.0, 90.0 }, anti_values);
}

test "device dataframe full joins with nullable payloads from both sides" {
    const gpa = std.testing.allocator;

    var left_id = try DeviceColumn.fromSliceWithValidity(i32, gpa, &.{ 1, 2, 3, 9 }, &.{ true, true, true, false }, .cpu);
    defer left_id.deinit();
    var left_value = try DeviceColumn.fromSlice(f64, gpa, &.{ 10.0, 20.0, 30.0, 90.0 }, .cpu);
    defer left_value.deinit();

    var right_id = try DeviceColumn.fromSliceWithValidity(i32, gpa, &.{ 2, 4, 9 }, &.{ true, true, false }, .cpu);
    defer right_id.deinit();
    var right_value = try DeviceColumn.fromSlice(f64, gpa, &.{ 200.0, 400.0, 900.0 }, .cpu);
    defer right_value.deinit();

    var left = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "id", .data = left_id },
        .{ .name = "value", .data = left_value },
    });
    defer left.deinit();
    var right = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "id", .data = right_id },
        .{ .name = "value", .data = right_value },
    });
    defer right.deinit();

    var joined = try left.fullJoin(right, "id", "id", .{});
    defer joined.deinit();
    try std.testing.expectEqual(@as(usize, 6), joined.height());

    const ids = try (try joined.column("id")).i32.toOwnedSlice(gpa);
    defer gpa.free(ids);
    const id_validity = try (try joined.column("id")).i32.validity.?.toOwnedSlice(gpa);
    defer gpa.free(id_validity);
    const left_values = try (try joined.column("value")).f64.toOwnedSlice(gpa);
    defer gpa.free(left_values);
    const left_validity = try (try joined.column("value")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(left_validity);
    const right_values = try (try joined.column("value_right")).f64.toOwnedSlice(gpa);
    defer gpa.free(right_values);
    const right_validity = try (try joined.column("value_right")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(right_validity);

    try std.testing.expectEqualSlices(i32, &.{ 1, 2, 3, 0, 4, 0 }, ids);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false, true, false }, id_validity);
    try std.testing.expectEqualSlices(f64, &.{ 10.0, 20.0, 30.0, 90.0, 0.0, 0.0 }, left_values);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, false, false }, left_validity);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 200.0, 0.0, 0.0, 400.0, 900.0 }, right_values);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false, true, true }, right_validity);
}

test "device dataframe asof joins with previous next and nearest strategies" {
    const gpa = std.testing.allocator;

    var left_time = try DeviceColumn.fromSliceWithValidity(i64, gpa, &.{ 1, 5, 8, 12, 20 }, &.{ true, true, true, true, false }, .cpu);
    defer left_time.deinit();
    var left_value = try DeviceColumn.fromSlice(f64, gpa, &.{ 10.0, 50.0, 80.0, 120.0, 200.0 }, .cpu);
    defer left_value.deinit();

    var right_time = try DeviceColumn.fromSliceWithValidity(i64, gpa, &.{ 2, 6, 10, 30 }, &.{ true, true, true, false }, .cpu);
    defer right_time.deinit();
    var quote = try DeviceColumn.fromSlice(i64, gpa, &.{ 20, 60, 100, 300 }, .cpu);
    defer quote.deinit();

    var left = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "time", .data = left_time },
        .{ .name = "value", .data = left_value },
    });
    defer left.deinit();
    var right = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "time", .data = right_time },
        .{ .name = "quote", .data = quote },
    });
    defer right.deinit();

    var previous = try left.asofJoin(right, "time", "time", .{ .strategy = .previous });
    defer previous.deinit();
    const previous_quote = try (try previous.column("quote")).i64.toOwnedSlice(gpa);
    defer gpa.free(previous_quote);
    const previous_validity = try (try previous.column("quote")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(previous_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 20, 60, 100, 0 }, previous_quote);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, false }, previous_validity);

    var next = try left.asofJoin(right, "time", "time", .{ .strategy = .next });
    defer next.deinit();
    const next_quote = try (try next.column("quote")).i64.toOwnedSlice(gpa);
    defer gpa.free(next_quote);
    const next_validity = try (try next.column("quote")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(next_validity);
    try std.testing.expectEqualSlices(i64, &.{ 20, 60, 100, 0, 0 }, next_quote);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false, false }, next_validity);

    var nearest = try left.asofJoin(right, "time", "time", .{ .strategy = .nearest });
    defer nearest.deinit();
    const nearest_quote = try (try nearest.column("quote")).i64.toOwnedSlice(gpa);
    defer gpa.free(nearest_quote);
    const nearest_validity = try (try nearest.column("quote")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(nearest_validity);
    try std.testing.expectEqualSlices(i64, &.{ 20, 60, 60, 100, 0 }, nearest_quote);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, false }, nearest_validity);
}

test "device dataframe concatenates rows eagerly and lazily" {
    const gpa = std.testing.allocator;

    var left_id = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 2 }, .cpu);
    defer left_id.deinit();
    var left_value = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 10.0, 20.0 }, &.{ true, false }, .cpu);
    defer left_value.deinit();
    var right_id = try DeviceColumn.fromSlice(i32, gpa, &.{ 3, 4 }, .cpu);
    defer right_id.deinit();
    var right_value = try DeviceColumn.fromSlice(f64, gpa, &.{ 30.0, 40.0 }, .cpu);
    defer right_value.deinit();

    var left = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "id", .data = left_id },
        .{ .name = "value", .data = left_value },
    });
    defer left.deinit();
    var right = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "id", .data = right_id },
        .{ .name = "value", .data = right_value },
    });
    defer right.deinit();

    var stacked = try left.concatRows(right);
    defer stacked.deinit();
    try std.testing.expectEqual(@as(usize, 4), stacked.height());
    try std.testing.expectEqual(@as(usize, 2), stacked.width());
    const stacked_ids = try (try stacked.column("id")).i32.toOwnedSlice(gpa);
    defer gpa.free(stacked_ids);
    const stacked_values = try (try stacked.column("value")).f64.toOwnedSlice(gpa);
    defer gpa.free(stacked_values);
    const stacked_validity = try (try stacked.column("value")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(stacked_validity);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2, 3, 4 }, stacked_ids);
    try std.testing.expectEqualSlices(f64, &.{ 10.0, 20.0, 30.0, 40.0 }, stacked_values);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, true }, stacked_validity);

    var plan = try DeviceLazyFrame.init(gpa, left);
    defer plan.deinit();
    try plan.concatRows(right);
    try plan.filterColumnScalar("id", i32, 2, .ge);
    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "concat_rows(rows=2, cols=2)") != null);
    var lazy_stacked = try plan.collect();
    defer lazy_stacked.deinit();
    try std.testing.expectEqual(@as(usize, 3), lazy_stacked.height());
    const lazy_ids = try (try lazy_stacked.column("id")).i32.toOwnedSlice(gpa);
    defer gpa.free(lazy_ids);
    try std.testing.expectEqualSlices(i32, &.{ 2, 3, 4 }, lazy_ids);
}

test "device dataframe drops duplicate rows eagerly and lazily" {
    const gpa = std.testing.allocator;

    var id = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 1, 2, 2, 3 }, .cpu);
    defer id.deinit();
    var value = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 10.0, 99.0, 20.0, 21.0, 30.0 }, &.{ true, true, true, true, false }, .cpu);
    defer value.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "id", .data = id },
        .{ .name = "value", .data = value },
    });
    defer table.deinit();

    var distinct = try table.distinctOn(&.{"id"});
    defer distinct.deinit();
    try std.testing.expectEqual(@as(usize, 3), distinct.height());
    const distinct_ids = try (try distinct.column("id")).i32.toOwnedSlice(gpa);
    defer gpa.free(distinct_ids);
    const distinct_values = try (try distinct.column("value")).f64.toOwnedSlice(gpa);
    defer gpa.free(distinct_values);
    const distinct_validity = try (try distinct.column("value")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(distinct_validity);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2, 3 }, distinct_ids);
    try std.testing.expectEqualSlices(f64, &.{ 10.0, 20.0, 30.0 }, distinct_values);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false }, distinct_validity);

    var full_distinct = try table.distinctRows();
    defer full_distinct.deinit();
    try std.testing.expectEqual(@as(usize, 4), full_distinct.height());

    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.distinctOn(&.{"id"});
    try plan.select(&.{ "id", "value" });
    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "distinct_on([id])") != null);
    var lazy_distinct = try plan.collect();
    defer lazy_distinct.deinit();
    try std.testing.expectEqual(@as(usize, 3), lazy_distinct.height());
    const lazy_ids = try (try lazy_distinct.column("id")).i32.toOwnedSlice(gpa);
    defer gpa.free(lazy_ids);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2, 3 }, lazy_ids);
}
test "device lazy frame collects staged select filter sort and limit operations" {
    const gpa = std.testing.allocator;

    var sales = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 3.0, 5.0, 7.0 }, .cpu);
    defer sales.deinit();
    var units = try DeviceColumn.fromSlice(i64, gpa, &.{ 1, 2, 3, 4 }, .cpu);
    defer units.deinit();
    var active = try DeviceColumn.fromSlice(bool, gpa, &.{ true, false, true, true }, .cpu);
    defer active.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "sales", .data = sales },
        .{ .name = "units", .data = units },
        .{ .name = "active", .data = active },
    });
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

    var lag_plan = try DeviceLazyFrame.init(gpa, table);
    defer lag_plan.deinit();
    try lag_plan.lagProfile("sales", "sales", .{ .periods = 1 });
    try lag_plan.select(&.{ "sales", "sales_lag", "sales_diff", "sales_pct_change" });
    const lag_explain = try lag_plan.explain(gpa);
    defer gpa.free(lag_explain);
    try std.testing.expect(std.mem.indexOf(u8, lag_explain, "lag_profile(sales") != null);
    var lagged = try lag_plan.collect();
    defer lagged.deinit();
    try std.testing.expectEqual(@as(usize, 4), lagged.height());
    try std.testing.expectEqual(@as(usize, 4), lagged.width());
    const lazy_lag = try (try lagged.column("sales_lag")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_lag);
    const lazy_diff = try (try lagged.column("sales_diff")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_diff);
    const lazy_pct = try (try lagged.column("sales_pct_change")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_pct);
    const lazy_lag_validity = try (try lagged.column("sales_lag")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_lag_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_lag_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_lag[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), lazy_lag[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), lazy_lag[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_diff[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_diff[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_diff[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_pct[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), lazy_pct[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.4), lazy_pct[3], 1e-12);

    var lead_plan = try DeviceLazyFrame.init(gpa, table);
    defer lead_plan.deinit();
    try lead_plan.leadProfile("sales", "sales", .{ .periods = 1 });
    try lead_plan.select(&.{ "sales", "sales_lead", "sales_forward_diff", "sales_forward_pct_change" });
    const lead_explain = try lead_plan.explain(gpa);
    defer gpa.free(lead_explain);
    try std.testing.expect(std.mem.indexOf(u8, lead_explain, "lead_profile(sales") != null);
    var leaded = try lead_plan.collect();
    defer leaded.deinit();
    try std.testing.expectEqual(@as(usize, 4), leaded.height());
    try std.testing.expectEqual(@as(usize, 4), leaded.width());
    const lazy_lead = try (try leaded.column("sales_lead")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_lead);
    const lazy_forward_diff = try (try leaded.column("sales_forward_diff")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_forward_diff);
    const lazy_forward_pct = try (try leaded.column("sales_forward_pct_change")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_forward_pct);
    const lazy_lead_validity = try (try leaded.column("sales_lead")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_lead_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false }, lazy_lead_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), lazy_lead[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), lazy_lead[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 7.0), lazy_lead[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_forward_diff[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_forward_diff[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_forward_diff[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_forward_pct[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), lazy_forward_pct[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.4), lazy_forward_pct[2], 1e-12);

    var clip_plan = try DeviceLazyFrame.init(gpa, table);
    defer clip_plan.deinit();
    try clip_plan.clipProfile("sales", "sales", .{ .lower = 3.0, .upper = 5.0 });
    try clip_plan.select(&.{ "sales", "sales_clipped", "sales_below", "sales_above", "sales_in_range" });
    const clip_explain = try clip_plan.explain(gpa);
    defer gpa.free(clip_explain);
    try std.testing.expect(std.mem.indexOf(u8, clip_explain, "clip_profile(sales") != null);
    var lazy_clip = try clip_plan.collect();
    defer lazy_clip.deinit();
    try std.testing.expectEqual(@as(usize, 4), lazy_clip.height());
    try std.testing.expectEqual(@as(usize, 5), lazy_clip.width());
    const lazy_clipped = try (try lazy_clip.column("sales_clipped")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_clipped);
    const lazy_below = try (try lazy_clip.column("sales_below")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_below);
    const lazy_above = try (try lazy_clip.column("sales_above")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_above);
    const lazy_in_range = try (try lazy_clip.column("sales_in_range")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_in_range);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 3.0, 5.0, 5.0 }, lazy_clipped);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false }, lazy_below);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, true }, lazy_above);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, false }, lazy_in_range);

    var rolling_clip_plan = try DeviceLazyFrame.init(gpa, table);
    defer rolling_clip_plan.deinit();
    try rolling_clip_plan.rollingClipProfile("sales", "sales", .{ .lower = 3.0, .upper = 5.0 }, .{ .window = 2, .min_periods = 2 });
    try rolling_clip_plan.select(&.{ "sales", "sales_rolling_clip_count", "sales_rolling_mean_clipped", "sales_rolling_clipped_rate", "sales_rolling_clip_below_rate", "sales_rolling_clip_above_rate", "sales_rolling_clip_in_range_rate" });
    const rolling_clip_explain = try rolling_clip_plan.explain(gpa);
    defer gpa.free(rolling_clip_explain);
    try std.testing.expect(std.mem.indexOf(u8, rolling_clip_explain, "rolling_clip_profile(sales") != null);
    var lazy_rolling_clip = try rolling_clip_plan.collect();
    defer lazy_rolling_clip.deinit();
    try std.testing.expectEqual(@as(usize, 4), lazy_rolling_clip.height());
    try std.testing.expectEqual(@as(usize, 7), lazy_rolling_clip.width());
    const lazy_rolling_clip_count = try (try lazy_rolling_clip.column("sales_rolling_clip_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_clip_count);
    const lazy_rolling_mean_clipped = try (try lazy_rolling_clip.column("sales_rolling_mean_clipped")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_mean_clipped);
    const lazy_rolling_clipped_rate = try (try lazy_rolling_clip.column("sales_rolling_clipped_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_clipped_rate);
    const lazy_rolling_clip_below_rate = try (try lazy_rolling_clip.column("sales_rolling_clip_below_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_clip_below_rate);
    const lazy_rolling_clip_above_rate = try (try lazy_rolling_clip.column("sales_rolling_clip_above_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_clip_above_rate);
    const lazy_rolling_clip_in_range_rate = try (try lazy_rolling_clip.column("sales_rolling_clip_in_range_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_clip_in_range_rate);
    const lazy_rolling_clip_validity = try (try lazy_rolling_clip.column("sales_rolling_mean_clipped")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_clip_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 2, 2 }, lazy_rolling_clip_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_rolling_clip_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), lazy_rolling_mean_clipped[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), lazy_rolling_mean_clipped[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), lazy_rolling_mean_clipped[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_rolling_clipped_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_rolling_clipped_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_rolling_clipped_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_rolling_clip_below_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_rolling_clip_below_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_rolling_clip_below_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_rolling_clip_above_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_rolling_clip_above_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_rolling_clip_above_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_rolling_clip_in_range_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_clip_in_range_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_rolling_clip_in_range_rate[3], 1e-12);

    var expanding_clip_plan = try DeviceLazyFrame.init(gpa, table);
    defer expanding_clip_plan.deinit();
    try expanding_clip_plan.expandingClipProfile("sales", "sales", .{ .lower = 3.0, .upper = 5.0 }, .{ .min_periods = 2 });
    try expanding_clip_plan.select(&.{ "sales", "sales_expanding_clip_count", "sales_expanding_mean_clipped", "sales_expanding_clipped_rate", "sales_expanding_clip_below_rate", "sales_expanding_clip_above_rate", "sales_expanding_clip_in_range_rate" });
    const expanding_clip_explain = try expanding_clip_plan.explain(gpa);
    defer gpa.free(expanding_clip_explain);
    try std.testing.expect(std.mem.indexOf(u8, expanding_clip_explain, "expanding_clip_profile(sales") != null);
    var lazy_expanding_clip = try expanding_clip_plan.collect();
    defer lazy_expanding_clip.deinit();
    try std.testing.expectEqual(@as(usize, 4), lazy_expanding_clip.height());
    try std.testing.expectEqual(@as(usize, 7), lazy_expanding_clip.width());
    const lazy_expanding_clip_count = try (try lazy_expanding_clip.column("sales_expanding_clip_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_clip_count);
    const lazy_expanding_mean_clipped = try (try lazy_expanding_clip.column("sales_expanding_mean_clipped")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_mean_clipped);
    const lazy_expanding_clipped_rate = try (try lazy_expanding_clip.column("sales_expanding_clipped_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_clipped_rate);
    const lazy_expanding_clip_below_rate = try (try lazy_expanding_clip.column("sales_expanding_clip_below_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_clip_below_rate);
    const lazy_expanding_clip_above_rate = try (try lazy_expanding_clip.column("sales_expanding_clip_above_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_clip_above_rate);
    const lazy_expanding_clip_in_range_rate = try (try lazy_expanding_clip.column("sales_expanding_clip_in_range_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_clip_in_range_rate);
    const lazy_expanding_clip_validity = try (try lazy_expanding_clip.column("sales_expanding_mean_clipped")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_clip_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4 }, lazy_expanding_clip_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_expanding_clip_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), lazy_expanding_mean_clipped[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 11.0 / 3.0), lazy_expanding_mean_clipped[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), lazy_expanding_mean_clipped[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_expanding_clipped_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), lazy_expanding_clipped_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_expanding_clipped_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_expanding_clip_below_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), lazy_expanding_clip_below_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), lazy_expanding_clip_below_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_expanding_clip_above_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_expanding_clip_above_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), lazy_expanding_clip_above_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_expanding_clip_in_range_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), lazy_expanding_clip_in_range_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_expanding_clip_in_range_rate[3], 1e-12);

    var threshold_plan = try DeviceLazyFrame.init(gpa, table);
    defer threshold_plan.deinit();
    try threshold_plan.thresholdProfile("sales", "sales", .{ .threshold = 5.0 });
    try threshold_plan.select(&.{ "sales", "sales_distance", "sales_abs_distance", "sales_above", "sales_below", "sales_at" });
    const threshold_explain = try threshold_plan.explain(gpa);
    defer gpa.free(threshold_explain);
    try std.testing.expect(std.mem.indexOf(u8, threshold_explain, "threshold_profile(sales") != null);
    var lazy_threshold = try threshold_plan.collect();
    defer lazy_threshold.deinit();
    try std.testing.expectEqual(@as(usize, 4), lazy_threshold.height());
    try std.testing.expectEqual(@as(usize, 6), lazy_threshold.width());
    const lazy_distance = try (try lazy_threshold.column("sales_distance")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_distance);
    const lazy_abs_distance = try (try lazy_threshold.column("sales_abs_distance")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_abs_distance);
    const lazy_above_threshold = try (try lazy_threshold.column("sales_above")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_above_threshold);
    const lazy_below_threshold = try (try lazy_threshold.column("sales_below")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_below_threshold);
    const lazy_at_threshold = try (try lazy_threshold.column("sales_at")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_at_threshold);
    try std.testing.expectEqualSlices(f64, &.{ -3.0, -2.0, 0.0, 2.0 }, lazy_distance);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 2.0, 0.0, 2.0 }, lazy_abs_distance);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, true }, lazy_above_threshold);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false }, lazy_below_threshold);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false }, lazy_at_threshold);

    var rolling_threshold_plan = try DeviceLazyFrame.init(gpa, table);
    defer rolling_threshold_plan.deinit();
    try rolling_threshold_plan.rollingThresholdProfile("sales", "sales", 4.0, .{ .window = 2, .min_periods = 2 });
    try rolling_threshold_plan.select(&.{ "sales", "sales_rolling_threshold_count", "sales_rolling_mean_distance", "sales_rolling_mean_abs_distance", "sales_rolling_above_rate", "sales_rolling_below_rate", "sales_rolling_at_rate" });
    const rolling_threshold_explain = try rolling_threshold_plan.explain(gpa);
    defer gpa.free(rolling_threshold_explain);
    try std.testing.expect(std.mem.indexOf(u8, rolling_threshold_explain, "rolling_threshold_profile(sales") != null);
    var lazy_rolling_threshold = try rolling_threshold_plan.collect();
    defer lazy_rolling_threshold.deinit();
    try std.testing.expectEqual(@as(usize, 4), lazy_rolling_threshold.height());
    try std.testing.expectEqual(@as(usize, 7), lazy_rolling_threshold.width());
    const lazy_rolling_threshold_count = try (try lazy_rolling_threshold.column("sales_rolling_threshold_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_threshold_count);
    const lazy_rolling_mean_distance = try (try lazy_rolling_threshold.column("sales_rolling_mean_distance")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_mean_distance);
    const lazy_rolling_mean_abs_distance = try (try lazy_rolling_threshold.column("sales_rolling_mean_abs_distance")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_mean_abs_distance);
    const lazy_rolling_above_rate = try (try lazy_rolling_threshold.column("sales_rolling_above_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_above_rate);
    const lazy_rolling_below_rate = try (try lazy_rolling_threshold.column("sales_rolling_below_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_below_rate);
    const lazy_rolling_at_rate = try (try lazy_rolling_threshold.column("sales_rolling_at_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_at_rate);
    const lazy_rolling_threshold_validity = try (try lazy_rolling_threshold.column("sales_rolling_mean_distance")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_threshold_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 2, 2 }, lazy_rolling_threshold_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_rolling_threshold_validity);
    try std.testing.expectApproxEqAbs(@as(f64, -1.5), lazy_rolling_mean_distance[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_rolling_mean_distance[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_rolling_mean_distance[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), lazy_rolling_mean_abs_distance[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_mean_abs_distance[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_rolling_mean_abs_distance[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_rolling_above_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_rolling_above_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_above_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_below_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_rolling_below_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_rolling_below_rate[3], 1e-12);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.0, 0.0 }, lazy_rolling_at_rate);

    var expanding_threshold_plan = try DeviceLazyFrame.init(gpa, table);
    defer expanding_threshold_plan.deinit();
    try expanding_threshold_plan.expandingThresholdProfile("sales", "sales", 4.0, .{ .min_periods = 2 });
    try expanding_threshold_plan.select(&.{ "sales", "sales_expanding_threshold_count", "sales_expanding_mean_distance", "sales_expanding_mean_abs_distance", "sales_expanding_above_rate", "sales_expanding_below_rate", "sales_expanding_at_rate" });
    const expanding_threshold_explain = try expanding_threshold_plan.explain(gpa);
    defer gpa.free(expanding_threshold_explain);
    try std.testing.expect(std.mem.indexOf(u8, expanding_threshold_explain, "expanding_threshold_profile(sales") != null);
    var lazy_expanding_threshold = try expanding_threshold_plan.collect();
    defer lazy_expanding_threshold.deinit();
    try std.testing.expectEqual(@as(usize, 4), lazy_expanding_threshold.height());
    try std.testing.expectEqual(@as(usize, 7), lazy_expanding_threshold.width());
    const lazy_expanding_threshold_count = try (try lazy_expanding_threshold.column("sales_expanding_threshold_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_threshold_count);
    const lazy_expanding_mean_distance = try (try lazy_expanding_threshold.column("sales_expanding_mean_distance")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_mean_distance);
    const lazy_expanding_mean_abs_distance = try (try lazy_expanding_threshold.column("sales_expanding_mean_abs_distance")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_mean_abs_distance);
    const lazy_expanding_above_rate = try (try lazy_expanding_threshold.column("sales_expanding_above_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_above_rate);
    const lazy_expanding_below_rate = try (try lazy_expanding_threshold.column("sales_expanding_below_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_below_rate);
    const lazy_expanding_at_rate = try (try lazy_expanding_threshold.column("sales_expanding_at_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_at_rate);
    const lazy_expanding_threshold_validity = try (try lazy_expanding_threshold.column("sales_expanding_mean_distance")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_threshold_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4 }, lazy_expanding_threshold_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_expanding_threshold_validity);
    try std.testing.expectApproxEqAbs(@as(f64, -1.5), lazy_expanding_mean_distance[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -2.0 / 3.0), lazy_expanding_mean_distance[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), lazy_expanding_mean_distance[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), lazy_expanding_mean_abs_distance[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0 / 3.0), lazy_expanding_mean_abs_distance[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.75), lazy_expanding_mean_abs_distance[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_expanding_above_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), lazy_expanding_above_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_expanding_above_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_expanding_below_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), lazy_expanding_below_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_expanding_below_rate[3], 1e-12);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.0, 0.0 }, lazy_expanding_at_rate);

    var expanding_plan = try DeviceLazyFrame.init(gpa, table);
    defer expanding_plan.deinit();
    try expanding_plan.expandingProfile("sales", "sales", .{ .min_periods = 2 });
    try expanding_plan.select(&.{ "sales", "sales_expanding_count", "sales_expanding_mean", "sales_expanding_max" });
    const expanding_explain = try expanding_plan.explain(gpa);
    defer gpa.free(expanding_explain);
    try std.testing.expect(std.mem.indexOf(u8, expanding_explain, "expanding_profile(sales") != null);
    var expanding = try expanding_plan.collect();
    defer expanding.deinit();
    try std.testing.expectEqual(@as(usize, 4), expanding.height());
    try std.testing.expectEqual(@as(usize, 4), expanding.width());
    const expanding_count = try (try expanding.column("sales_expanding_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(expanding_count);
    const expanding_mean = try (try expanding.column("sales_expanding_mean")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_mean);
    const expanding_max = try (try expanding.column("sales_expanding_max")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_max);
    const expanding_validity = try (try expanding.column("sales_expanding_mean")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(expanding_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4 }, expanding_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, expanding_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), expanding_mean[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0 / 3.0), expanding_mean[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.25), expanding_mean[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), expanding_max[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), expanding_max[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 7.0), expanding_max[3], 1e-12);

    var expanding_norm_plan = try DeviceLazyFrame.init(gpa, table);
    defer expanding_norm_plan.deinit();
    try expanding_norm_plan.expandingNormalizeProfile("sales", "sales", .{ .min_periods = 2 });
    try expanding_norm_plan.select(&.{ "sales", "sales_expanding_centered", "sales_expanding_zscore", "sales_expanding_minmax" });
    const expanding_norm_explain = try expanding_norm_plan.explain(gpa);
    defer gpa.free(expanding_norm_explain);
    try std.testing.expect(std.mem.indexOf(u8, expanding_norm_explain, "expanding_normalize_profile(sales") != null);
    var expanding_norm = try expanding_norm_plan.collect();
    defer expanding_norm.deinit();
    try std.testing.expectEqual(@as(usize, 4), expanding_norm.height());
    try std.testing.expectEqual(@as(usize, 4), expanding_norm.width());
    const lazy_expanding_centered = try (try expanding_norm.column("sales_expanding_centered")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_centered);
    const lazy_expanding_zscore = try (try expanding_norm.column("sales_expanding_zscore")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_zscore);
    const lazy_expanding_minmax = try (try expanding_norm.column("sales_expanding_minmax")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_minmax);
    const lazy_expanding_norm_validity = try (try expanding_norm.column("sales_expanding_zscore")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_norm_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_expanding_norm_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_expanding_centered[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 3.0), lazy_expanding_centered[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.75), lazy_expanding_centered[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_expanding_zscore[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.3363062095621219), lazy_expanding_zscore[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.4320780207890627), lazy_expanding_zscore[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_expanding_minmax[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_expanding_minmax[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_expanding_minmax[3], 1e-12);

    var expanding_moment_plan = try DeviceLazyFrame.init(gpa, table);
    defer expanding_moment_plan.deinit();
    try expanding_moment_plan.expandingMomentProfile("sales", "sales", .{ .min_periods = 2 });
    try expanding_moment_plan.select(&.{ "sales", "sales_expanding_moment_count", "sales_expanding_m3", "sales_expanding_m4", "sales_expanding_skewness", "sales_expanding_kurtosis" });
    const expanding_moment_explain = try expanding_moment_plan.explain(gpa);
    defer gpa.free(expanding_moment_explain);
    try std.testing.expect(std.mem.indexOf(u8, expanding_moment_explain, "expanding_moment_profile(sales") != null);
    var expanding_moments = try expanding_moment_plan.collect();
    defer expanding_moments.deinit();
    try std.testing.expectEqual(@as(usize, 4), expanding_moments.height());
    try std.testing.expectEqual(@as(usize, 6), expanding_moments.width());
    const lazy_expanding_moment_count = try (try expanding_moments.column("sales_expanding_moment_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_moment_count);
    const lazy_expanding_m3 = try (try expanding_moments.column("sales_expanding_m3")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_m3);
    const lazy_expanding_m4 = try (try expanding_moments.column("sales_expanding_m4")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_m4);
    const lazy_expanding_skewness = try (try expanding_moments.column("sales_expanding_skewness")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_skewness);
    const lazy_expanding_kurtosis = try (try expanding_moments.column("sales_expanding_kurtosis")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_kurtosis);
    const lazy_expanding_moment_validity = try (try expanding_moments.column("sales_expanding_skewness")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_moment_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4 }, lazy_expanding_moment_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_expanding_moment_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_expanding_m3[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.7407407407407399), lazy_expanding_m3[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.96875), lazy_expanding_m3[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0625), lazy_expanding_m4[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.6296296296296293), lazy_expanding_m4[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 21.39453125), lazy_expanding_m4[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_expanding_skewness[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.3818017741606058), lazy_expanding_skewness[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.27803055565396284), lazy_expanding_skewness[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -2.0), lazy_expanding_kurtosis[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -1.5), lazy_expanding_kurtosis[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -1.4266015512783683), lazy_expanding_kurtosis[3], 1e-12);

    var standardize_plan = try DeviceLazyFrame.init(gpa, table);
    defer standardize_plan.deinit();
    try standardize_plan.standardizeProfile("sales", "sales", .{});
    try standardize_plan.select(&.{ "sales", "sales_centered", "sales_zscore", "sales_minmax" });
    const standardize_explain = try standardize_plan.explain(gpa);
    defer gpa.free(standardize_explain);
    try std.testing.expect(std.mem.indexOf(u8, standardize_explain, "standardize_profile(sales") != null);
    var standardized = try standardize_plan.collect();
    defer standardized.deinit();
    try std.testing.expectEqual(@as(usize, 4), standardized.height());
    try std.testing.expectEqual(@as(usize, 4), standardized.width());
    const lazy_centered = try (try standardized.column("sales_centered")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_centered);
    const lazy_zscore = try (try standardized.column("sales_zscore")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_zscore);
    const lazy_minmax = try (try standardized.column("sales_minmax")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_minmax);
    const lazy_standardized_validity = try (try standardized.column("sales_zscore")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_standardized_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, lazy_standardized_validity);
    try std.testing.expectApproxEqAbs(@as(f64, -2.25), lazy_centered[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -1.25), lazy_centered[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), lazy_centered[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.75), lazy_centered[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -1.171700198827415), lazy_zscore[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.6509445549041194), lazy_zscore[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.39056673294247163), lazy_zscore[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.4320780207890627), lazy_zscore[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_minmax[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.2), lazy_minmax[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.6), lazy_minmax[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_minmax[3], 1e-12);

    var robust_plan = try DeviceLazyFrame.init(gpa, table);
    defer robust_plan.deinit();
    try robust_plan.robustProfile("sales", "sales", .{});
    try robust_plan.select(&.{ "sales", "sales_median_centered", "sales_mad_zscore", "sales_iqr_outlier", "sales_winsorized" });
    const robust_explain = try robust_plan.explain(gpa);
    defer gpa.free(robust_explain);
    try std.testing.expect(std.mem.indexOf(u8, robust_explain, "robust_profile(sales") != null);
    var robust = try robust_plan.collect();
    defer robust.deinit();
    try std.testing.expectEqual(@as(usize, 4), robust.height());
    try std.testing.expectEqual(@as(usize, 5), robust.width());
    const lazy_median_centered = try (try robust.column("sales_median_centered")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_median_centered);
    const lazy_mad_zscore = try (try robust.column("sales_mad_zscore")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_mad_zscore);
    const lazy_iqr_outlier = try (try robust.column("sales_iqr_outlier")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_iqr_outlier);
    const lazy_winsorized = try (try robust.column("sales_winsorized")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_winsorized);
    try std.testing.expectApproxEqAbs(@as(f64, -2.0), lazy_median_centered[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -1.0), lazy_median_centered[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_median_centered[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), lazy_median_centered[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.8993196669281089), lazy_mad_zscore[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.44965983346405447), lazy_mad_zscore[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.44965983346405447), lazy_mad_zscore[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.3489795003921634), lazy_mad_zscore[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false }, lazy_iqr_outlier);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 3.0, 5.0, 7.0 }, lazy_winsorized);

    var expanding_robust_plan = try DeviceLazyFrame.init(gpa, table);
    defer expanding_robust_plan.deinit();
    try expanding_robust_plan.expandingRobustProfile("sales", "sales", .{ .min_periods = 2 });
    try expanding_robust_plan.select(&.{ "sales", "sales_expanding_median_centered", "sales_expanding_mad_zscore", "sales_expanding_iqr_outlier", "sales_expanding_winsorized" });
    const expanding_robust_explain = try expanding_robust_plan.explain(gpa);
    defer gpa.free(expanding_robust_explain);
    try std.testing.expect(std.mem.indexOf(u8, expanding_robust_explain, "expanding_robust_profile(sales") != null);
    var expanding_robust = try expanding_robust_plan.collect();
    defer expanding_robust.deinit();
    try std.testing.expectEqual(@as(usize, 4), expanding_robust.height());
    try std.testing.expectEqual(@as(usize, 5), expanding_robust.width());
    const lazy_expanding_median_centered = try (try expanding_robust.column("sales_expanding_median_centered")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_median_centered);
    const lazy_expanding_mad_zscore = try (try expanding_robust.column("sales_expanding_mad_zscore")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_mad_zscore);
    const lazy_expanding_iqr_outlier = try (try expanding_robust.column("sales_expanding_iqr_outlier")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_iqr_outlier);
    const lazy_expanding_winsorized = try (try expanding_robust.column("sales_expanding_winsorized")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_winsorized);
    const lazy_expanding_robust_validity = try (try expanding_robust.column("sales_expanding_mad_zscore")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_robust_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_expanding_robust_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_expanding_median_centered[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_expanding_median_centered[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), lazy_expanding_median_centered[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.6744897501960817), lazy_expanding_mad_zscore[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.3489795003921634), lazy_expanding_mad_zscore[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.3489795003921634), lazy_expanding_mad_zscore[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false }, lazy_expanding_iqr_outlier);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), lazy_expanding_winsorized[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), lazy_expanding_winsorized[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 7.0), lazy_expanding_winsorized[3], 1e-12);

    var drawdown_plan = try DeviceLazyFrame.init(gpa, table);
    defer drawdown_plan.deinit();
    try drawdown_plan.drawdownProfile("sales", "sales", .{ .min_periods = 2 });
    try drawdown_plan.select(&.{ "sales", "sales_running_peak", "sales_drawdown", "sales_drawdown_pct" });
    const drawdown_explain = try drawdown_plan.explain(gpa);
    defer gpa.free(drawdown_explain);
    try std.testing.expect(std.mem.indexOf(u8, drawdown_explain, "drawdown_profile(sales") != null);
    var drawdown = try drawdown_plan.collect();
    defer drawdown.deinit();
    try std.testing.expectEqual(@as(usize, 4), drawdown.height());
    try std.testing.expectEqual(@as(usize, 4), drawdown.width());
    const lazy_peak = try (try drawdown.column("sales_running_peak")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_peak);
    const lazy_drawdown = try (try drawdown.column("sales_drawdown")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_drawdown);
    const lazy_drawdown_pct = try (try drawdown.column("sales_drawdown_pct")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_drawdown_pct);
    const lazy_drawdown_validity = try (try drawdown.column("sales_drawdown")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_drawdown_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_drawdown_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), lazy_peak[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), lazy_peak[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 7.0), lazy_peak[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_drawdown[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_drawdown[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_drawdown[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_drawdown_pct[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_drawdown_pct[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_drawdown_pct[3], 1e-12);

    var rolling_drawdown_plan = try DeviceLazyFrame.init(gpa, table);
    defer rolling_drawdown_plan.deinit();
    try rolling_drawdown_plan.rollingDrawdownProfile("sales", "sales", .{ .window = 2, .min_periods = 1 });
    try rolling_drawdown_plan.select(&.{ "sales", "sales_rolling_peak", "sales_rolling_drawdown", "sales_rolling_drawdown_pct", "sales_rolling_peak_age" });
    const rolling_drawdown_explain = try rolling_drawdown_plan.explain(gpa);
    defer gpa.free(rolling_drawdown_explain);
    try std.testing.expect(std.mem.indexOf(u8, rolling_drawdown_explain, "rolling_drawdown_profile(sales") != null);
    var lazy_rolling_drawdown = try rolling_drawdown_plan.collect();
    defer lazy_rolling_drawdown.deinit();
    try std.testing.expectEqual(@as(usize, 4), lazy_rolling_drawdown.height());
    try std.testing.expectEqual(@as(usize, 5), lazy_rolling_drawdown.width());
    const lazy_rolling_peak = try (try lazy_rolling_drawdown.column("sales_rolling_peak")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_peak);
    const lazy_rolling_drawdown_values = try (try lazy_rolling_drawdown.column("sales_rolling_drawdown")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_drawdown_values);
    const lazy_rolling_drawdown_pct = try (try lazy_rolling_drawdown.column("sales_rolling_drawdown_pct")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_drawdown_pct);
    const lazy_rolling_peak_age = try (try lazy_rolling_drawdown.column("sales_rolling_peak_age")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_peak_age);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 3.0, 5.0, 7.0 }, lazy_rolling_peak);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.0, 0.0 }, lazy_rolling_drawdown_values);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.0, 0.0 }, lazy_rolling_drawdown_pct);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0 }, lazy_rolling_peak_age);

    var extrema_plan = try DeviceLazyFrame.init(gpa, table);
    defer extrema_plan.deinit();
    try extrema_plan.extremaProfile("sales", "sales", .{ .min_periods = 1 });
    try extrema_plan.select(&.{ "sales", "sales_running_low", "sales_running_high", "sales_new_low", "sales_new_high" });
    const extrema_explain = try extrema_plan.explain(gpa);
    defer gpa.free(extrema_explain);
    try std.testing.expect(std.mem.indexOf(u8, extrema_explain, "extrema_profile(sales") != null);
    var lazy_extrema = try extrema_plan.collect();
    defer lazy_extrema.deinit();
    try std.testing.expectEqual(@as(usize, 4), lazy_extrema.height());
    try std.testing.expectEqual(@as(usize, 5), lazy_extrema.width());
    const lazy_running_low = try (try lazy_extrema.column("sales_running_low")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_running_low);
    const lazy_running_high = try (try lazy_extrema.column("sales_running_high")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_running_high);
    const lazy_new_low = try (try lazy_extrema.column("sales_new_low")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_new_low);
    const lazy_new_high = try (try lazy_extrema.column("sales_new_high")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_new_high);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 2.0, 2.0, 2.0 }, lazy_running_low);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 3.0, 5.0, 7.0 }, lazy_running_high);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false }, lazy_new_low);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, lazy_new_high);

    var trend_plan = try DeviceLazyFrame.init(gpa, table);
    defer trend_plan.deinit();
    try trend_plan.trendProfile("sales", "sales", .{ .periods = 1 });
    try trend_plan.select(&.{ "sales", "sales_trend", "sales_up_streak", "sales_reversal" });
    const trend_explain = try trend_plan.explain(gpa);
    defer gpa.free(trend_explain);
    try std.testing.expect(std.mem.indexOf(u8, trend_explain, "trend_profile(sales") != null);
    var trend = try trend_plan.collect();
    defer trend.deinit();
    try std.testing.expectEqual(@as(usize, 4), trend.height());
    try std.testing.expectEqual(@as(usize, 4), trend.width());
    const lazy_trend = try (try trend.column("sales_trend")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_trend);
    const lazy_up_streak = try (try trend.column("sales_up_streak")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_up_streak);
    const lazy_reversal = try (try trend.column("sales_reversal")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_reversal);
    const lazy_trend_validity = try (try trend.column("sales_trend")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_trend_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_trend_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 1, 1 }, lazy_trend);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 2, 3 }, lazy_up_streak);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false }, lazy_reversal);

    var rolling_trend_plan = try DeviceLazyFrame.init(gpa, table);
    defer rolling_trend_plan.deinit();
    try rolling_trend_plan.rollingTrendProfile("sales", "sales", .{ .periods = 1 }, .{ .window = 2, .min_periods = 1 });
    try rolling_trend_plan.select(&.{ "sales", "sales_rolling_trend_count", "sales_rolling_up_rate", "sales_rolling_down_rate", "sales_rolling_flat_rate", "sales_rolling_reversal_rate" });
    const rolling_trend_explain = try rolling_trend_plan.explain(gpa);
    defer gpa.free(rolling_trend_explain);
    try std.testing.expect(std.mem.indexOf(u8, rolling_trend_explain, "rolling_trend_profile(sales") != null);
    var lazy_rolling_trend = try rolling_trend_plan.collect();
    defer lazy_rolling_trend.deinit();
    try std.testing.expectEqual(@as(usize, 4), lazy_rolling_trend.height());
    try std.testing.expectEqual(@as(usize, 6), lazy_rolling_trend.width());
    const lazy_rolling_trend_count = try (try lazy_rolling_trend.column("sales_rolling_trend_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_trend_count);
    const lazy_rolling_up_rate = try (try lazy_rolling_trend.column("sales_rolling_up_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_up_rate);
    const lazy_rolling_down_rate = try (try lazy_rolling_trend.column("sales_rolling_down_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_down_rate);
    const lazy_rolling_flat_rate = try (try lazy_rolling_trend.column("sales_rolling_flat_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_flat_rate);
    const lazy_rolling_reversal_rate = try (try lazy_rolling_trend.column("sales_rolling_reversal_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_reversal_rate);
    const lazy_rolling_trend_validity = try (try lazy_rolling_trend.column("sales_rolling_up_rate")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_trend_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 2, 2 }, lazy_rolling_trend_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_rolling_trend_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_up_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_up_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_up_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_rolling_down_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_rolling_flat_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_rolling_reversal_rate[3], 1e-12);

    var expanding_trend_plan = try DeviceLazyFrame.init(gpa, table);
    defer expanding_trend_plan.deinit();
    try expanding_trend_plan.expandingTrendProfile("sales", "sales", .{ .periods = 1 }, .{ .min_periods = 1 });
    try expanding_trend_plan.select(&.{ "sales", "sales_expanding_trend_count", "sales_expanding_up_rate", "sales_expanding_down_rate", "sales_expanding_flat_rate", "sales_expanding_reversal_rate" });
    const expanding_trend_explain = try expanding_trend_plan.explain(gpa);
    defer gpa.free(expanding_trend_explain);
    try std.testing.expect(std.mem.indexOf(u8, expanding_trend_explain, "expanding_trend_profile(sales") != null);
    var lazy_expanding_trend = try expanding_trend_plan.collect();
    defer lazy_expanding_trend.deinit();
    try std.testing.expectEqual(@as(usize, 4), lazy_expanding_trend.height());
    try std.testing.expectEqual(@as(usize, 6), lazy_expanding_trend.width());
    const lazy_expanding_trend_count = try (try lazy_expanding_trend.column("sales_expanding_trend_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_trend_count);
    const lazy_expanding_up_rate = try (try lazy_expanding_trend.column("sales_expanding_up_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_up_rate);
    const lazy_expanding_down_rate = try (try lazy_expanding_trend.column("sales_expanding_down_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_down_rate);
    const lazy_expanding_flat_rate = try (try lazy_expanding_trend.column("sales_expanding_flat_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_flat_rate);
    const lazy_expanding_reversal_rate = try (try lazy_expanding_trend.column("sales_expanding_reversal_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_reversal_rate);
    const lazy_expanding_trend_validity = try (try lazy_expanding_trend.column("sales_expanding_up_rate")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_trend_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 2, 3 }, lazy_expanding_trend_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_expanding_trend_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_expanding_up_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_expanding_up_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_expanding_up_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_expanding_down_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_expanding_flat_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_expanding_reversal_rate[3], 1e-12);

    var change_plan = try DeviceLazyFrame.init(gpa, table);
    defer change_plan.deinit();
    try change_plan.changePointProfile("sales", "sales", 2.0, .{ .periods = 1 });
    try change_plan.select(&.{ "sales", "sales_change_delta", "sales_change_abs_delta", "sales_change_pct", "sales_change_point" });
    const change_explain = try change_plan.explain(gpa);
    defer gpa.free(change_explain);
    try std.testing.expect(std.mem.indexOf(u8, change_explain, "change_point_profile(sales") != null);
    var lazy_change = try change_plan.collect();
    defer lazy_change.deinit();
    try std.testing.expectEqual(@as(usize, 4), lazy_change.height());
    try std.testing.expectEqual(@as(usize, 5), lazy_change.width());
    const lazy_change_delta = try (try lazy_change.column("sales_change_delta")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_change_delta);
    const lazy_change_abs_delta = try (try lazy_change.column("sales_change_abs_delta")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_change_abs_delta);
    const lazy_change_pct = try (try lazy_change.column("sales_change_pct")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_change_pct);
    const lazy_change_point = try (try lazy_change.column("sales_change_point")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_change_point);
    const lazy_change_validity = try (try lazy_change.column("sales_change_delta")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_change_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_change_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_change_delta[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_change_delta[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_change_delta[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_change_abs_delta[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_change_abs_delta[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_change_abs_delta[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_change_pct[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), lazy_change_pct[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.4), lazy_change_pct[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, true }, lazy_change_point);

    var rolling_change_plan = try DeviceLazyFrame.init(gpa, table);
    defer rolling_change_plan.deinit();
    try rolling_change_plan.rollingChangePointProfile("sales", "sales", 2.0, .{ .periods = 1 }, .{ .window = 2, .min_periods = 1 });
    try rolling_change_plan.select(&.{ "sales", "sales_rolling_change_count", "sales_rolling_change_point_count", "sales_rolling_change_rate", "sales_rolling_mean_abs_delta", "sales_rolling_max_abs_delta" });
    const rolling_change_explain = try rolling_change_plan.explain(gpa);
    defer gpa.free(rolling_change_explain);
    try std.testing.expect(std.mem.indexOf(u8, rolling_change_explain, "rolling_change_point_profile(sales") != null);
    var lazy_rolling_change = try rolling_change_plan.collect();
    defer lazy_rolling_change.deinit();
    try std.testing.expectEqual(@as(usize, 4), lazy_rolling_change.height());
    try std.testing.expectEqual(@as(usize, 6), lazy_rolling_change.width());
    const lazy_rolling_change_count = try (try lazy_rolling_change.column("sales_rolling_change_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_change_count);
    const lazy_rolling_change_point_count = try (try lazy_rolling_change.column("sales_rolling_change_point_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_change_point_count);
    const lazy_rolling_change_rate = try (try lazy_rolling_change.column("sales_rolling_change_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_change_rate);
    const lazy_rolling_mean_abs_delta = try (try lazy_rolling_change.column("sales_rolling_mean_abs_delta")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_mean_abs_delta);
    const lazy_rolling_max_abs_delta = try (try lazy_rolling_change.column("sales_rolling_max_abs_delta")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_max_abs_delta);
    const lazy_rolling_change_validity = try (try lazy_rolling_change.column("sales_rolling_change_rate")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_change_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 2, 2 }, lazy_rolling_change_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 2 }, lazy_rolling_change_point_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_rolling_change_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_rolling_change_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_rolling_change_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_change_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_mean_abs_delta[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), lazy_rolling_mean_abs_delta[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_rolling_mean_abs_delta[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_max_abs_delta[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_rolling_max_abs_delta[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_rolling_max_abs_delta[3], 1e-12);

    var expanding_change_plan = try DeviceLazyFrame.init(gpa, table);
    defer expanding_change_plan.deinit();
    try expanding_change_plan.expandingChangePointProfile("sales", "sales", 2.0, .{ .periods = 1 }, .{ .min_periods = 1 });
    try expanding_change_plan.select(&.{ "sales", "sales_expanding_change_count", "sales_expanding_change_point_count", "sales_expanding_change_rate", "sales_expanding_mean_abs_delta", "sales_expanding_max_abs_delta" });
    const expanding_change_explain = try expanding_change_plan.explain(gpa);
    defer gpa.free(expanding_change_explain);
    try std.testing.expect(std.mem.indexOf(u8, expanding_change_explain, "expanding_change_point_profile(sales") != null);
    var lazy_expanding_change = try expanding_change_plan.collect();
    defer lazy_expanding_change.deinit();
    try std.testing.expectEqual(@as(usize, 4), lazy_expanding_change.height());
    try std.testing.expectEqual(@as(usize, 6), lazy_expanding_change.width());
    const lazy_expanding_change_count = try (try lazy_expanding_change.column("sales_expanding_change_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_change_count);
    const lazy_expanding_change_point_count = try (try lazy_expanding_change.column("sales_expanding_change_point_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_change_point_count);
    const lazy_expanding_change_rate = try (try lazy_expanding_change.column("sales_expanding_change_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_change_rate);
    const lazy_expanding_mean_abs_delta = try (try lazy_expanding_change.column("sales_expanding_mean_abs_delta")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_mean_abs_delta);
    const lazy_expanding_max_abs_delta = try (try lazy_expanding_change.column("sales_expanding_max_abs_delta")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_max_abs_delta);
    const lazy_expanding_change_validity = try (try lazy_expanding_change.column("sales_expanding_change_rate")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_change_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 2, 3 }, lazy_expanding_change_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 2 }, lazy_expanding_change_point_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_expanding_change_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_expanding_change_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_expanding_change_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), lazy_expanding_change_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_expanding_mean_abs_delta[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), lazy_expanding_mean_abs_delta[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 3.0), lazy_expanding_mean_abs_delta[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_expanding_max_abs_delta[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_expanding_max_abs_delta[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_expanding_max_abs_delta[3], 1e-12);

    var sign_plan = try DeviceLazyFrame.init(gpa, table);
    defer sign_plan.deinit();
    try sign_plan.withColumnScalar("sales_minus4", "sales", f64, 4.0, .sub);
    try sign_plan.signProfile("sales_minus4", "sales", .{ .periods = 1 });
    try sign_plan.select(&.{ "sales_minus4", "sales_sign", "sales_sign_flip", "sales_positive_streak", "sales_negative_streak" });
    const sign_explain = try sign_plan.explain(gpa);
    defer gpa.free(sign_explain);
    try std.testing.expect(std.mem.indexOf(u8, sign_explain, "sign_profile(sales_minus4") != null);
    var lazy_sign = try sign_plan.collect();
    defer lazy_sign.deinit();
    try std.testing.expectEqual(@as(usize, 4), lazy_sign.height());
    try std.testing.expectEqual(@as(usize, 5), lazy_sign.width());
    const lazy_sign_values = try (try lazy_sign.column("sales_sign")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_sign_values);
    const lazy_sign_flip = try (try lazy_sign.column("sales_sign_flip")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_sign_flip);
    const lazy_positive_streak = try (try lazy_sign.column("sales_positive_streak")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_positive_streak);
    const lazy_negative_streak = try (try lazy_sign.column("sales_negative_streak")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_negative_streak);
    try std.testing.expectEqualSlices(i64, &.{ -1, -1, 1, 1 }, lazy_sign_values);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false }, lazy_sign_flip);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 2 }, lazy_positive_streak);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 0, 0 }, lazy_negative_streak);

    var rolling_sign_plan = try DeviceLazyFrame.init(gpa, table);
    defer rolling_sign_plan.deinit();
    try rolling_sign_plan.withColumnScalar("sales_minus4", "sales", f64, 4.0, .sub);
    try rolling_sign_plan.rollingSignProfile("sales_minus4", "sales", .{ .periods = 1 }, .{ .window = 2, .min_periods = 1 });
    try rolling_sign_plan.select(&.{ "sales_minus4", "sales_rolling_sign_count", "sales_rolling_positive_rate", "sales_rolling_negative_rate", "sales_rolling_zero_rate", "sales_rolling_sign_flip_rate" });
    const rolling_sign_explain = try rolling_sign_plan.explain(gpa);
    defer gpa.free(rolling_sign_explain);
    try std.testing.expect(std.mem.indexOf(u8, rolling_sign_explain, "rolling_sign_profile(sales_minus4") != null);
    var lazy_rolling_sign = try rolling_sign_plan.collect();
    defer lazy_rolling_sign.deinit();
    try std.testing.expectEqual(@as(usize, 4), lazy_rolling_sign.height());
    try std.testing.expectEqual(@as(usize, 6), lazy_rolling_sign.width());
    const lazy_rolling_sign_count = try (try lazy_rolling_sign.column("sales_rolling_sign_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_sign_count);
    const lazy_rolling_positive_rate = try (try lazy_rolling_sign.column("sales_rolling_positive_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_positive_rate);
    const lazy_rolling_negative_rate = try (try lazy_rolling_sign.column("sales_rolling_negative_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_negative_rate);
    const lazy_rolling_zero_rate = try (try lazy_rolling_sign.column("sales_rolling_zero_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_zero_rate);
    const lazy_rolling_sign_flip_rate = try (try lazy_rolling_sign.column("sales_rolling_sign_flip_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_sign_flip_rate);
    const lazy_rolling_sign_validity = try (try lazy_rolling_sign.column("sales_rolling_positive_rate")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_sign_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 2, 2 }, lazy_rolling_sign_count);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, lazy_rolling_sign_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_rolling_positive_rate[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_rolling_positive_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_rolling_positive_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_positive_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_negative_rate[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_negative_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_rolling_negative_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_rolling_negative_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_rolling_zero_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_rolling_sign_flip_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_rolling_sign_flip_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_rolling_sign_flip_rate[3], 1e-12);

    var expanding_sign_plan = try DeviceLazyFrame.init(gpa, table);
    defer expanding_sign_plan.deinit();
    try expanding_sign_plan.withColumnScalar("sales_minus4", "sales", f64, 4.0, .sub);
    try expanding_sign_plan.expandingSignProfile("sales_minus4", "sales", .{ .periods = 1 }, .{ .min_periods = 1 });
    try expanding_sign_plan.select(&.{ "sales_minus4", "sales_expanding_sign_count", "sales_expanding_positive_rate", "sales_expanding_negative_rate", "sales_expanding_zero_rate", "sales_expanding_sign_flip_rate" });
    const expanding_sign_explain = try expanding_sign_plan.explain(gpa);
    defer gpa.free(expanding_sign_explain);
    try std.testing.expect(std.mem.indexOf(u8, expanding_sign_explain, "expanding_sign_profile(sales_minus4") != null);
    var lazy_expanding_sign = try expanding_sign_plan.collect();
    defer lazy_expanding_sign.deinit();
    try std.testing.expectEqual(@as(usize, 4), lazy_expanding_sign.height());
    try std.testing.expectEqual(@as(usize, 6), lazy_expanding_sign.width());
    const lazy_expanding_sign_count = try (try lazy_expanding_sign.column("sales_expanding_sign_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_sign_count);
    const lazy_expanding_positive_rate = try (try lazy_expanding_sign.column("sales_expanding_positive_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_positive_rate);
    const lazy_expanding_negative_rate = try (try lazy_expanding_sign.column("sales_expanding_negative_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_negative_rate);
    const lazy_expanding_zero_rate = try (try lazy_expanding_sign.column("sales_expanding_zero_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_zero_rate);
    const lazy_expanding_sign_flip_rate = try (try lazy_expanding_sign.column("sales_expanding_sign_flip_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_sign_flip_rate);
    const lazy_expanding_sign_validity = try (try lazy_expanding_sign.column("sales_expanding_positive_rate")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_sign_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4 }, lazy_expanding_sign_count);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, lazy_expanding_sign_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_expanding_positive_rate[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_expanding_positive_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), lazy_expanding_positive_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_expanding_positive_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_expanding_negative_rate[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_expanding_negative_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), lazy_expanding_negative_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_expanding_negative_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_expanding_zero_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_expanding_sign_flip_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), lazy_expanding_sign_flip_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), lazy_expanding_sign_flip_rate[3], 1e-12);

    var validity_plan = try DeviceLazyFrame.init(gpa, table);
    defer validity_plan.deinit();
    try validity_plan.validityProfile("sales", "sales");
    try validity_plan.select(&.{ "sales", "sales_is_null", "sales_is_valid", "sales_valid_streak", "sales_null_streak" });
    const validity_explain = try validity_plan.explain(gpa);
    defer gpa.free(validity_explain);
    try std.testing.expect(std.mem.indexOf(u8, validity_explain, "validity_profile(sales") != null);
    var lazy_validity = try validity_plan.collect();
    defer lazy_validity.deinit();
    try std.testing.expectEqual(@as(usize, 4), lazy_validity.height());
    try std.testing.expectEqual(@as(usize, 5), lazy_validity.width());
    const lazy_is_null = try (try lazy_validity.column("sales_is_null")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_is_null);
    const lazy_is_valid = try (try lazy_validity.column("sales_is_valid")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_is_valid);
    const lazy_valid_streak = try (try lazy_validity.column("sales_valid_streak")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_valid_streak);
    const lazy_null_streak = try (try lazy_validity.column("sales_null_streak")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_null_streak);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false }, lazy_is_null);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, lazy_is_valid);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4 }, lazy_valid_streak);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0 }, lazy_null_streak);

    var nullable_quality = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, 2.0, 3.0, 4.0 }, &.{ true, false, true, true }, .cpu);
    defer nullable_quality.deinit();
    var quality_table = try DeviceDataFrame.init(gpa, &.{.{ .name = "quality", .data = nullable_quality }});
    defer quality_table.deinit();
    var rolling_validity_plan = try DeviceLazyFrame.init(gpa, quality_table);
    defer rolling_validity_plan.deinit();
    try rolling_validity_plan.rollingValidityProfile("quality", "quality", .{ .window = 2, .min_periods = 2 });
    try rolling_validity_plan.select(&.{ "quality", "quality_rolling_validity_count", "quality_rolling_valid_count", "quality_rolling_null_count", "quality_rolling_valid_rate", "quality_rolling_null_rate" });
    const rolling_validity_explain = try rolling_validity_plan.explain(gpa);
    defer gpa.free(rolling_validity_explain);
    try std.testing.expect(std.mem.indexOf(u8, rolling_validity_explain, "rolling_validity_profile(quality") != null);
    var lazy_rolling_validity = try rolling_validity_plan.collect();
    defer lazy_rolling_validity.deinit();
    try std.testing.expectEqual(@as(usize, 4), lazy_rolling_validity.height());
    try std.testing.expectEqual(@as(usize, 6), lazy_rolling_validity.width());
    const lazy_rolling_validity_count = try (try lazy_rolling_validity.column("quality_rolling_validity_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_validity_count);
    const lazy_rolling_valid_count = try (try lazy_rolling_validity.column("quality_rolling_valid_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_valid_count);
    const lazy_rolling_null_count = try (try lazy_rolling_validity.column("quality_rolling_null_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_null_count);
    const lazy_rolling_valid_rate = try (try lazy_rolling_validity.column("quality_rolling_valid_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_valid_rate);
    const lazy_rolling_null_rate = try (try lazy_rolling_validity.column("quality_rolling_null_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_null_rate);
    const lazy_rolling_validity_validity = try (try lazy_rolling_validity.column("quality_rolling_valid_rate")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_validity_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 2, 2 }, lazy_rolling_validity_count);
    try std.testing.expectEqualSlices(i64, &.{ 1, 1, 1, 2 }, lazy_rolling_valid_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 1, 0 }, lazy_rolling_null_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_rolling_validity_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_rolling_valid_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_rolling_valid_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_valid_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_rolling_null_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_rolling_null_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_rolling_null_rate[3], 1e-12);

    var expanding_validity_plan = try DeviceLazyFrame.init(gpa, quality_table);
    defer expanding_validity_plan.deinit();
    try expanding_validity_plan.expandingValidityProfile("quality", "quality", .{ .min_periods = 2 });
    try expanding_validity_plan.select(&.{ "quality", "quality_expanding_validity_count", "quality_expanding_valid_count", "quality_expanding_null_count", "quality_expanding_valid_rate", "quality_expanding_null_rate" });
    const expanding_validity_explain = try expanding_validity_plan.explain(gpa);
    defer gpa.free(expanding_validity_explain);
    try std.testing.expect(std.mem.indexOf(u8, expanding_validity_explain, "expanding_validity_profile(quality") != null);
    var lazy_expanding_validity = try expanding_validity_plan.collect();
    defer lazy_expanding_validity.deinit();
    try std.testing.expectEqual(@as(usize, 4), lazy_expanding_validity.height());
    try std.testing.expectEqual(@as(usize, 6), lazy_expanding_validity.width());
    const lazy_expanding_validity_count = try (try lazy_expanding_validity.column("quality_expanding_validity_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_validity_count);
    const lazy_expanding_valid_count = try (try lazy_expanding_validity.column("quality_expanding_valid_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_valid_count);
    const lazy_expanding_null_count = try (try lazy_expanding_validity.column("quality_expanding_null_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_null_count);
    const lazy_expanding_valid_rate = try (try lazy_expanding_validity.column("quality_expanding_valid_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_valid_rate);
    const lazy_expanding_null_rate = try (try lazy_expanding_validity.column("quality_expanding_null_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_null_rate);
    const lazy_expanding_validity_validity = try (try lazy_expanding_validity.column("quality_expanding_valid_rate")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_validity_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4 }, lazy_expanding_validity_count);
    try std.testing.expectEqualSlices(i64, &.{ 1, 1, 2, 3 }, lazy_expanding_valid_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 1, 1 }, lazy_expanding_null_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_expanding_validity_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_expanding_valid_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), lazy_expanding_valid_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), lazy_expanding_valid_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_expanding_null_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), lazy_expanding_null_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), lazy_expanding_null_rate[3], 1e-12);

    var class_plan = try DeviceLazyFrame.init(gpa, table);
    defer class_plan.deinit();
    try class_plan.withColumnCompareScalar("predicted_active", "sales", f64, 4.0, .gt);
    try class_plan.classificationProfile("active", "predicted_active", "active_cls");
    try class_plan.select(&.{ "active", "predicted_active", "active_cls_tp", "active_cls_fp", "active_cls_tn", "active_cls_fn", "active_cls_correct" });
    const class_explain = try class_plan.explain(gpa);
    defer gpa.free(class_explain);
    try std.testing.expect(std.mem.indexOf(u8, class_explain, "classification_profile(actual=active, predicted=predicted_active") != null);
    var classed = try class_plan.collect();
    defer classed.deinit();
    try std.testing.expectEqual(@as(usize, 4), classed.height());
    try std.testing.expectEqual(@as(usize, 7), classed.width());
    const lazy_tp = try (try classed.column("active_cls_tp")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_tp);
    const lazy_fp = try (try classed.column("active_cls_fp")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_fp);
    const lazy_tn = try (try classed.column("active_cls_tn")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_tn);
    const lazy_fn = try (try classed.column("active_cls_fn")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_fn);
    const lazy_correct = try (try classed.column("active_cls_correct")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_correct);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, true }, lazy_tp);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false }, lazy_fp);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false }, lazy_tn);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false }, lazy_fn);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_correct);

    var rolling_class_plan = try DeviceLazyFrame.init(gpa, table);
    defer rolling_class_plan.deinit();
    try rolling_class_plan.withColumnCompareScalar("predicted_active", "sales", f64, 4.0, .gt);
    try rolling_class_plan.rollingClassificationProfile("active", "predicted_active", "active_cls", .{ .window = 2, .min_periods = 2 });
    try rolling_class_plan.select(&.{ "active", "predicted_active", "active_cls_rolling_class_count", "active_cls_rolling_tp_count", "active_cls_rolling_fp_count", "active_cls_rolling_tn_count", "active_cls_rolling_fn_count", "active_cls_rolling_accuracy", "active_cls_rolling_precision", "active_cls_rolling_recall" });
    const rolling_class_explain = try rolling_class_plan.explain(gpa);
    defer gpa.free(rolling_class_explain);
    try std.testing.expect(std.mem.indexOf(u8, rolling_class_explain, "rolling_classification_profile(actual=active, predicted=predicted_active") != null);
    var lazy_rolling_class = try rolling_class_plan.collect();
    defer lazy_rolling_class.deinit();
    try std.testing.expectEqual(@as(usize, 4), lazy_rolling_class.height());
    try std.testing.expectEqual(@as(usize, 10), lazy_rolling_class.width());
    const lazy_rolling_class_count = try (try lazy_rolling_class.column("active_cls_rolling_class_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_class_count);
    const lazy_rolling_tp_count = try (try lazy_rolling_class.column("active_cls_rolling_tp_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_tp_count);
    const lazy_rolling_fp_count = try (try lazy_rolling_class.column("active_cls_rolling_fp_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_fp_count);
    const lazy_rolling_tn_count = try (try lazy_rolling_class.column("active_cls_rolling_tn_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_tn_count);
    const lazy_rolling_fn_count = try (try lazy_rolling_class.column("active_cls_rolling_fn_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_fn_count);
    const lazy_rolling_accuracy = try (try lazy_rolling_class.column("active_cls_rolling_accuracy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_accuracy);
    const lazy_rolling_precision = try (try lazy_rolling_class.column("active_cls_rolling_precision")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_precision);
    const lazy_rolling_recall = try (try lazy_rolling_class.column("active_cls_rolling_recall")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_recall);
    const lazy_rolling_class_validity = try (try lazy_rolling_class.column("active_cls_rolling_accuracy")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_class_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 2, 2 }, lazy_rolling_class_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 2 }, lazy_rolling_tp_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0 }, lazy_rolling_fp_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 1, 0 }, lazy_rolling_tn_count);
    try std.testing.expectEqualSlices(i64, &.{ 1, 1, 0, 0 }, lazy_rolling_fn_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_rolling_class_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_rolling_accuracy[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_accuracy[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_accuracy[3], 1e-12);
    try std.testing.expect(std.math.isNan(lazy_rolling_precision[1]));
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_precision[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_precision[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_rolling_recall[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_recall[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_recall[3], 1e-12);

    var expanding_class_plan = try DeviceLazyFrame.init(gpa, table);
    defer expanding_class_plan.deinit();
    try expanding_class_plan.withColumnCompareScalar("predicted_active", "sales", f64, 4.0, .gt);
    try expanding_class_plan.expandingClassificationProfile("active", "predicted_active", "active_cls", .{ .min_periods = 2 });
    try expanding_class_plan.select(&.{ "active", "predicted_active", "active_cls_expanding_class_count", "active_cls_expanding_tp_count", "active_cls_expanding_fp_count", "active_cls_expanding_tn_count", "active_cls_expanding_fn_count", "active_cls_expanding_accuracy", "active_cls_expanding_precision", "active_cls_expanding_recall" });
    const expanding_class_explain = try expanding_class_plan.explain(gpa);
    defer gpa.free(expanding_class_explain);
    try std.testing.expect(std.mem.indexOf(u8, expanding_class_explain, "expanding_classification_profile(actual=active, predicted=predicted_active") != null);
    var lazy_expanding_class = try expanding_class_plan.collect();
    defer lazy_expanding_class.deinit();
    try std.testing.expectEqual(@as(usize, 4), lazy_expanding_class.height());
    try std.testing.expectEqual(@as(usize, 10), lazy_expanding_class.width());
    const lazy_expanding_class_count = try (try lazy_expanding_class.column("active_cls_expanding_class_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_class_count);
    const lazy_expanding_tp_count = try (try lazy_expanding_class.column("active_cls_expanding_tp_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_tp_count);
    const lazy_expanding_fp_count = try (try lazy_expanding_class.column("active_cls_expanding_fp_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_fp_count);
    const lazy_expanding_tn_count = try (try lazy_expanding_class.column("active_cls_expanding_tn_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_tn_count);
    const lazy_expanding_fn_count = try (try lazy_expanding_class.column("active_cls_expanding_fn_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_fn_count);
    const lazy_expanding_accuracy = try (try lazy_expanding_class.column("active_cls_expanding_accuracy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_accuracy);
    const lazy_expanding_precision = try (try lazy_expanding_class.column("active_cls_expanding_precision")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_precision);
    const lazy_expanding_recall = try (try lazy_expanding_class.column("active_cls_expanding_recall")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_recall);
    const lazy_expanding_class_validity = try (try lazy_expanding_class.column("active_cls_expanding_accuracy")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_class_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4 }, lazy_expanding_class_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 2 }, lazy_expanding_tp_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0 }, lazy_expanding_fp_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 1, 1 }, lazy_expanding_tn_count);
    try std.testing.expectEqualSlices(i64, &.{ 1, 1, 1, 1 }, lazy_expanding_fn_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_expanding_class_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_expanding_accuracy[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), lazy_expanding_accuracy[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), lazy_expanding_accuracy[3], 1e-12);
    try std.testing.expect(std.math.isNan(lazy_expanding_precision[1]));
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_expanding_precision[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_expanding_precision[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_expanding_recall[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_expanding_recall[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), lazy_expanding_recall[3], 1e-12);

    var transition_plan = try DeviceLazyFrame.init(gpa, table);
    defer transition_plan.deinit();
    try transition_plan.boolTransitionProfile("active", "active", .{ .periods = 1 });
    try transition_plan.select(&.{ "active", "active_rising", "active_falling", "active_toggled", "active_true_streak", "active_false_streak" });
    const transition_explain = try transition_plan.explain(gpa);
    defer gpa.free(transition_explain);
    try std.testing.expect(std.mem.indexOf(u8, transition_explain, "bool_transition_profile(active") != null);
    var lazy_transition = try transition_plan.collect();
    defer lazy_transition.deinit();
    try std.testing.expectEqual(@as(usize, 4), lazy_transition.height());
    try std.testing.expectEqual(@as(usize, 6), lazy_transition.width());
    const lazy_rising = try (try lazy_transition.column("active_rising")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_rising);
    const lazy_falling = try (try lazy_transition.column("active_falling")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_falling);
    const lazy_toggled = try (try lazy_transition.column("active_toggled")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_toggled);
    const lazy_true_streak = try (try lazy_transition.column("active_true_streak")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_true_streak);
    const lazy_false_streak = try (try lazy_transition.column("active_false_streak")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_false_streak);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false }, lazy_rising);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false }, lazy_falling);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, false }, lazy_toggled);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 1, 2 }, lazy_true_streak);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 0 }, lazy_false_streak);

    var rolling_transition_plan = try DeviceLazyFrame.init(gpa, table);
    defer rolling_transition_plan.deinit();
    try rolling_transition_plan.rollingBoolTransitionProfile("active", "active", .{ .periods = 1 }, .{ .window = 2, .min_periods = 1 });
    try rolling_transition_plan.select(&.{ "active", "active_rolling_transition_count", "active_rolling_rising_count", "active_rolling_falling_count", "active_rolling_toggle_count", "active_rolling_rising_rate", "active_rolling_falling_rate", "active_rolling_toggle_rate" });
    const rolling_transition_explain = try rolling_transition_plan.explain(gpa);
    defer gpa.free(rolling_transition_explain);
    try std.testing.expect(std.mem.indexOf(u8, rolling_transition_explain, "rolling_bool_transition_profile(active") != null);
    var lazy_rolling_transition = try rolling_transition_plan.collect();
    defer lazy_rolling_transition.deinit();
    try std.testing.expectEqual(@as(usize, 4), lazy_rolling_transition.height());
    try std.testing.expectEqual(@as(usize, 8), lazy_rolling_transition.width());
    const lazy_rolling_transition_count = try (try lazy_rolling_transition.column("active_rolling_transition_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_transition_count);
    const lazy_rolling_rising_count = try (try lazy_rolling_transition.column("active_rolling_rising_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_rising_count);
    const lazy_rolling_falling_count = try (try lazy_rolling_transition.column("active_rolling_falling_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_falling_count);
    const lazy_rolling_toggle_count = try (try lazy_rolling_transition.column("active_rolling_toggle_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_toggle_count);
    const lazy_rolling_rising_rate = try (try lazy_rolling_transition.column("active_rolling_rising_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_rising_rate);
    const lazy_rolling_falling_rate = try (try lazy_rolling_transition.column("active_rolling_falling_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_falling_rate);
    const lazy_rolling_toggle_rate = try (try lazy_rolling_transition.column("active_rolling_toggle_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_toggle_rate);
    const lazy_rolling_transition_validity = try (try lazy_rolling_transition.column("active_rolling_toggle_rate")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_transition_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 2, 2 }, lazy_rolling_transition_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 1 }, lazy_rolling_rising_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 1, 0 }, lazy_rolling_falling_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 2, 1 }, lazy_rolling_toggle_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_rolling_transition_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_rolling_rising_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_rolling_rising_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_rolling_rising_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_falling_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_rolling_falling_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_rolling_falling_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_toggle_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_toggle_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_rolling_toggle_rate[3], 1e-12);

    var expanding_transition_plan = try DeviceLazyFrame.init(gpa, table);
    defer expanding_transition_plan.deinit();
    try expanding_transition_plan.expandingBoolTransitionProfile("active", "active", .{ .periods = 1 }, .{ .min_periods = 2 });
    try expanding_transition_plan.select(&.{ "active", "active_expanding_transition_count", "active_expanding_rising_count", "active_expanding_falling_count", "active_expanding_toggle_count", "active_expanding_rising_rate", "active_expanding_falling_rate", "active_expanding_toggle_rate" });
    const expanding_transition_explain = try expanding_transition_plan.explain(gpa);
    defer gpa.free(expanding_transition_explain);
    try std.testing.expect(std.mem.indexOf(u8, expanding_transition_explain, "expanding_bool_transition_profile(active") != null);
    var lazy_expanding_transition = try expanding_transition_plan.collect();
    defer lazy_expanding_transition.deinit();
    try std.testing.expectEqual(@as(usize, 4), lazy_expanding_transition.height());
    try std.testing.expectEqual(@as(usize, 8), lazy_expanding_transition.width());
    const lazy_expanding_transition_count = try (try lazy_expanding_transition.column("active_expanding_transition_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_transition_count);
    const lazy_expanding_rising_count = try (try lazy_expanding_transition.column("active_expanding_rising_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_rising_count);
    const lazy_expanding_falling_count = try (try lazy_expanding_transition.column("active_expanding_falling_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_falling_count);
    const lazy_expanding_toggle_count = try (try lazy_expanding_transition.column("active_expanding_toggle_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_toggle_count);
    const lazy_expanding_rising_rate = try (try lazy_expanding_transition.column("active_expanding_rising_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_rising_rate);
    const lazy_expanding_falling_rate = try (try lazy_expanding_transition.column("active_expanding_falling_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_falling_rate);
    const lazy_expanding_toggle_rate = try (try lazy_expanding_transition.column("active_expanding_toggle_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_toggle_rate);
    const lazy_expanding_transition_validity = try (try lazy_expanding_transition.column("active_expanding_toggle_rate")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_transition_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 2, 3 }, lazy_expanding_transition_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 1 }, lazy_expanding_rising_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 1, 1 }, lazy_expanding_falling_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 2, 2 }, lazy_expanding_toggle_count);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, true }, lazy_expanding_transition_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_expanding_rising_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), lazy_expanding_rising_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_expanding_falling_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), lazy_expanding_falling_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_expanding_toggle_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), lazy_expanding_toggle_rate[3], 1e-12);

    var crossover_plan = try DeviceLazyFrame.init(gpa, table);
    defer crossover_plan.deinit();
    try crossover_plan.withColumnScalar("units_f64", "sales", f64, 1.0, .sub);
    try crossover_plan.crossoverProfile("sales", "units_f64", "sales_units", .{ .periods = 1 });
    try crossover_plan.select(&.{ "sales", "units_f64", "sales_units_spread", "sales_units_ratio", "sales_units_cross_above", "sales_units_cross_below" });
    const crossover_explain = try crossover_plan.explain(gpa);
    defer gpa.free(crossover_explain);
    try std.testing.expect(std.mem.indexOf(u8, crossover_explain, "crossover_profile(sales,units_f64") != null);
    var crossover = try crossover_plan.collect();
    defer crossover.deinit();
    try std.testing.expectEqual(@as(usize, 4), crossover.height());
    try std.testing.expectEqual(@as(usize, 6), crossover.width());
    const lazy_spread = try (try crossover.column("sales_units_spread")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_spread);
    const lazy_ratio = try (try crossover.column("sales_units_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ratio);
    const lazy_cross_above = try (try crossover.column("sales_units_cross_above")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_cross_above);
    const lazy_cross_below = try (try crossover.column("sales_units_cross_below")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_cross_below);
    const lazy_cross_validity = try (try crossover.column("sales_units_cross_above")).bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_cross_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_cross_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_spread[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_spread[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_spread[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_spread[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_ratio[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), lazy_ratio[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.25), lazy_ratio[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 7.0 / 6.0), lazy_ratio[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false }, lazy_cross_above);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false }, lazy_cross_below);

    var rolling_crossover_plan = try DeviceLazyFrame.init(gpa, table);
    defer rolling_crossover_plan.deinit();
    try rolling_crossover_plan.withColumnScalar("sales_minus4", "sales", f64, 4.0, .sub);
    try rolling_crossover_plan.withColumnScalar("zero_sales", "sales", f64, 0.0, .mul);
    try rolling_crossover_plan.rollingCrossoverProfile("sales_minus4", "zero_sales", "sales_zero", .{ .periods = 1 }, .{ .window = 2, .min_periods = 2 });
    try rolling_crossover_plan.select(&.{ "sales", "sales_minus4", "zero_sales", "sales_zero_rolling_cross_count", "sales_zero_rolling_cross_above_count", "sales_zero_rolling_cross_below_count", "sales_zero_rolling_cross_above_rate", "sales_zero_rolling_cross_below_rate", "sales_zero_rolling_mean_abs_spread" });
    const rolling_crossover_explain = try rolling_crossover_plan.explain(gpa);
    defer gpa.free(rolling_crossover_explain);
    try std.testing.expect(std.mem.indexOf(u8, rolling_crossover_explain, "rolling_crossover_profile(sales_minus4,zero_sales") != null);
    var rolling_crossover = try rolling_crossover_plan.collect();
    defer rolling_crossover.deinit();
    try std.testing.expectEqual(@as(usize, 4), rolling_crossover.height());
    try std.testing.expectEqual(@as(usize, 9), rolling_crossover.width());
    const lazy_rolling_cross_count = try (try rolling_crossover.column("sales_zero_rolling_cross_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_cross_count);
    const lazy_rolling_cross_above_count = try (try rolling_crossover.column("sales_zero_rolling_cross_above_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_cross_above_count);
    const lazy_rolling_cross_below_count = try (try rolling_crossover.column("sales_zero_rolling_cross_below_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_cross_below_count);
    const lazy_rolling_cross_above_rate = try (try rolling_crossover.column("sales_zero_rolling_cross_above_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_cross_above_rate);
    const lazy_rolling_cross_below_rate = try (try rolling_crossover.column("sales_zero_rolling_cross_below_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_cross_below_rate);
    const lazy_rolling_mean_abs_spread = try (try rolling_crossover.column("sales_zero_rolling_mean_abs_spread")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_mean_abs_spread);
    const lazy_rolling_crossover_validity = try (try rolling_crossover.column("sales_zero_rolling_cross_above_rate")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_crossover_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 2, 2 }, lazy_rolling_cross_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 1 }, lazy_rolling_cross_above_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0 }, lazy_rolling_cross_below_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_rolling_crossover_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_rolling_cross_above_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_rolling_cross_above_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_rolling_cross_above_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_rolling_cross_below_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), lazy_rolling_mean_abs_spread[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_mean_abs_spread[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_rolling_mean_abs_spread[3], 1e-12);

    var expanding_crossover_plan = try DeviceLazyFrame.init(gpa, table);
    defer expanding_crossover_plan.deinit();
    try expanding_crossover_plan.withColumnScalar("sales_minus4", "sales", f64, 4.0, .sub);
    try expanding_crossover_plan.withColumnScalar("zero_sales", "sales", f64, 0.0, .mul);
    try expanding_crossover_plan.expandingCrossoverProfile("sales_minus4", "zero_sales", "sales_zero", .{ .periods = 1 }, .{ .min_periods = 2 });
    try expanding_crossover_plan.select(&.{ "sales", "sales_minus4", "zero_sales", "sales_zero_expanding_cross_count", "sales_zero_expanding_cross_above_count", "sales_zero_expanding_cross_below_count", "sales_zero_expanding_cross_above_rate", "sales_zero_expanding_cross_below_rate", "sales_zero_expanding_mean_abs_spread" });
    const expanding_crossover_explain = try expanding_crossover_plan.explain(gpa);
    defer gpa.free(expanding_crossover_explain);
    try std.testing.expect(std.mem.indexOf(u8, expanding_crossover_explain, "expanding_crossover_profile(sales_minus4,zero_sales") != null);
    var expanding_crossover = try expanding_crossover_plan.collect();
    defer expanding_crossover.deinit();
    try std.testing.expectEqual(@as(usize, 4), expanding_crossover.height());
    try std.testing.expectEqual(@as(usize, 9), expanding_crossover.width());
    const lazy_expanding_cross_count = try (try expanding_crossover.column("sales_zero_expanding_cross_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_cross_count);
    const lazy_expanding_cross_above_count = try (try expanding_crossover.column("sales_zero_expanding_cross_above_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_cross_above_count);
    const lazy_expanding_cross_below_count = try (try expanding_crossover.column("sales_zero_expanding_cross_below_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_cross_below_count);
    const lazy_expanding_cross_above_rate = try (try expanding_crossover.column("sales_zero_expanding_cross_above_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_cross_above_rate);
    const lazy_expanding_cross_below_rate = try (try expanding_crossover.column("sales_zero_expanding_cross_below_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_cross_below_rate);
    const lazy_expanding_mean_abs_spread = try (try expanding_crossover.column("sales_zero_expanding_mean_abs_spread")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_mean_abs_spread);
    const lazy_expanding_crossover_validity = try (try expanding_crossover.column("sales_zero_expanding_cross_above_rate")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_crossover_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4 }, lazy_expanding_cross_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 1 }, lazy_expanding_cross_above_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0 }, lazy_expanding_cross_below_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_expanding_crossover_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_expanding_cross_above_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), lazy_expanding_cross_above_rate[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), lazy_expanding_cross_above_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_expanding_cross_below_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), lazy_expanding_mean_abs_spread[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0 / 3.0), lazy_expanding_mean_abs_spread[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.75), lazy_expanding_mean_abs_spread[3], 1e-12);

    var fit_plan = try DeviceLazyFrame.init(gpa, table);
    defer fit_plan.deinit();
    try fit_plan.withColumnScalar("sales_minus1", "sales", f64, 1.0, .sub);
    try fit_plan.linearFitProfile("sales_minus1", "sales", "sales_fit", .{});
    try fit_plan.select(&.{ "sales", "sales_minus1", "sales_fit_fitted", "sales_fit_residual", "sales_fit_residual_zscore", "sales_fit_slope" });
    const fit_explain = try fit_plan.explain(gpa);
    defer gpa.free(fit_explain);
    try std.testing.expect(std.mem.indexOf(u8, fit_explain, "linear_fit_profile(sales_minus1->sales") != null);
    var fit = try fit_plan.collect();
    defer fit.deinit();
    try std.testing.expectEqual(@as(usize, 4), fit.height());
    try std.testing.expectEqual(@as(usize, 6), fit.width());
    const lazy_fitted = try (try fit.column("sales_fit_fitted")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_fitted);
    const lazy_fit_residual = try (try fit.column("sales_fit_residual")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_fit_residual);
    const lazy_fit_residual_z = try (try fit.column("sales_fit_residual_zscore")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_fit_residual_z);
    const lazy_fit_slope = try (try fit.column("sales_fit_slope")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_fit_slope);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_fitted[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), lazy_fitted[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), lazy_fitted[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 7.0), lazy_fitted[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_fit_residual[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_fit_residual[3], 1e-12);
    try std.testing.expect(std.math.isNan(lazy_fit_residual_z[0]));
    try std.testing.expect(std.math.isNan(lazy_fit_residual_z[3]));
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_fit_slope[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_fit_slope[3], 1e-12);

    var error_plan = try DeviceLazyFrame.init(gpa, table);
    defer error_plan.deinit();
    try error_plan.withColumnScalar("sales_minus1", "sales", f64, 1.0, .sub);
    try error_plan.errorProfile("sales", "sales_minus1", "sales_err");
    try error_plan.select(&.{ "sales", "sales_err_error", "sales_err_abs_error", "sales_err_squared_error", "sales_err_ape", "sales_err_smape" });
    const error_explain = try error_plan.explain(gpa);
    defer gpa.free(error_explain);
    try std.testing.expect(std.mem.indexOf(u8, error_explain, "error_profile(actual=sales, predicted=sales_minus1") != null);
    var lazy_errors = try error_plan.collect();
    defer lazy_errors.deinit();
    try std.testing.expectEqual(@as(usize, 4), lazy_errors.height());
    try std.testing.expectEqual(@as(usize, 6), lazy_errors.width());
    const lazy_error = try (try lazy_errors.column("sales_err_error")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_error);
    const lazy_abs_error = try (try lazy_errors.column("sales_err_abs_error")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_abs_error);
    const lazy_squared_error = try (try lazy_errors.column("sales_err_squared_error")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_squared_error);
    const lazy_ape = try (try lazy_errors.column("sales_err_ape")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ape);
    const lazy_smape = try (try lazy_errors.column("sales_err_smape")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_smape);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 1.0, 1.0, 1.0 }, lazy_error);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 1.0, 1.0, 1.0 }, lazy_abs_error);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 1.0, 1.0, 1.0 }, lazy_squared_error);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_ape[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), lazy_ape[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.2), lazy_ape[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 7.0), lazy_ape[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), lazy_smape[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 5.0), lazy_smape[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 9.0), lazy_smape[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 13.0), lazy_smape[3], 1e-12);

    var rolling_error_plan = try DeviceLazyFrame.init(gpa, table);
    defer rolling_error_plan.deinit();
    try rolling_error_plan.withColumnScalar("sales_minus1", "sales", f64, 1.0, .sub);
    try rolling_error_plan.rollingErrorProfile("sales", "sales_minus1", "sales_err", .{ .window = 2, .min_periods = 2 });
    try rolling_error_plan.select(&.{ "sales", "sales_err_rolling_error_count", "sales_err_rolling_mae", "sales_err_rolling_rmse", "sales_err_rolling_mape", "sales_err_rolling_smape" });
    const rolling_error_explain = try rolling_error_plan.explain(gpa);
    defer gpa.free(rolling_error_explain);
    try std.testing.expect(std.mem.indexOf(u8, rolling_error_explain, "rolling_error_profile(actual=sales, predicted=sales_minus1") != null);
    var rolling_errors = try rolling_error_plan.collect();
    defer rolling_errors.deinit();
    try std.testing.expectEqual(@as(usize, 4), rolling_errors.height());
    try std.testing.expectEqual(@as(usize, 6), rolling_errors.width());
    const lazy_rolling_error_count = try (try rolling_errors.column("sales_err_rolling_error_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_error_count);
    const lazy_rolling_mae = try (try rolling_errors.column("sales_err_rolling_mae")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_mae);
    const lazy_rolling_rmse = try (try rolling_errors.column("sales_err_rolling_rmse")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_rmse);
    const lazy_rolling_mape = try (try rolling_errors.column("sales_err_rolling_mape")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_mape);
    const lazy_rolling_smape = try (try rolling_errors.column("sales_err_rolling_smape")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_smape);
    const lazy_rolling_error_validity = try (try rolling_errors.column("sales_err_rolling_mae")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_error_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 2, 2 }, lazy_rolling_error_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_rolling_error_validity);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 1.0, 1.0, 1.0 }, lazy_rolling_mae);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 1.0, 1.0, 1.0 }, lazy_rolling_rmse);
    try std.testing.expectApproxEqAbs(@as(f64, (0.5 + 1.0 / 3.0) / 2.0), lazy_rolling_mape[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, (1.0 / 3.0 + 0.2) / 2.0), lazy_rolling_mape[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, (0.2 + 1.0 / 7.0) / 2.0), lazy_rolling_mape[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, (2.0 / 3.0 + 2.0 / 5.0) / 2.0), lazy_rolling_smape[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, (2.0 / 5.0 + 2.0 / 9.0) / 2.0), lazy_rolling_smape[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, (2.0 / 9.0 + 2.0 / 13.0) / 2.0), lazy_rolling_smape[3], 1e-12);

    var expanding_error_plan = try DeviceLazyFrame.init(gpa, table);
    defer expanding_error_plan.deinit();
    try expanding_error_plan.withColumnScalar("sales_minus1", "sales", f64, 1.0, .sub);
    try expanding_error_plan.expandingErrorProfile("sales", "sales_minus1", "sales_err", .{ .min_periods = 2 });
    try expanding_error_plan.select(&.{ "sales", "sales_err_expanding_error_count", "sales_err_expanding_mae", "sales_err_expanding_rmse", "sales_err_expanding_mape", "sales_err_expanding_smape" });
    const expanding_error_explain = try expanding_error_plan.explain(gpa);
    defer gpa.free(expanding_error_explain);
    try std.testing.expect(std.mem.indexOf(u8, expanding_error_explain, "expanding_error_profile(actual=sales, predicted=sales_minus1") != null);
    var expanding_errors = try expanding_error_plan.collect();
    defer expanding_errors.deinit();
    try std.testing.expectEqual(@as(usize, 4), expanding_errors.height());
    try std.testing.expectEqual(@as(usize, 6), expanding_errors.width());
    const lazy_expanding_error_count = try (try expanding_errors.column("sales_err_expanding_error_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_error_count);
    const lazy_expanding_mae = try (try expanding_errors.column("sales_err_expanding_mae")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_mae);
    const lazy_expanding_rmse = try (try expanding_errors.column("sales_err_expanding_rmse")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_rmse);
    const lazy_expanding_mape = try (try expanding_errors.column("sales_err_expanding_mape")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_mape);
    const lazy_expanding_smape = try (try expanding_errors.column("sales_err_expanding_smape")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_smape);
    const lazy_expanding_error_validity = try (try expanding_errors.column("sales_err_expanding_mae")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_error_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4 }, lazy_expanding_error_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_expanding_error_validity);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 1.0, 1.0, 1.0 }, lazy_expanding_mae);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 1.0, 1.0, 1.0 }, lazy_expanding_rmse);
    try std.testing.expectApproxEqAbs(@as(f64, (0.5 + 1.0 / 3.0) / 2.0), lazy_expanding_mape[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, (0.5 + 1.0 / 3.0 + 0.2) / 3.0), lazy_expanding_mape[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, (0.5 + 1.0 / 3.0 + 0.2 + 1.0 / 7.0) / 4.0), lazy_expanding_mape[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, (2.0 / 3.0 + 2.0 / 5.0) / 2.0), lazy_expanding_smape[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, (2.0 / 3.0 + 2.0 / 5.0 + 2.0 / 9.0) / 3.0), lazy_expanding_smape[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, (2.0 / 3.0 + 2.0 / 5.0 + 2.0 / 9.0 + 2.0 / 13.0) / 4.0), lazy_expanding_smape[3], 1e-12);

    var corr_plan = try DeviceLazyFrame.init(gpa, table);
    defer corr_plan.deinit();
    try corr_plan.withColumnScalar("sales_minus1", "sales", f64, 1.0, .sub);
    try corr_plan.rollingCorrelationProfile("sales_minus1", "sales", "sales_corr", .{ .window = 2, .min_periods = 2 });
    try corr_plan.select(&.{ "sales", "sales_corr_rolling_pair_count", "sales_corr_rolling_covariance", "sales_corr_rolling_correlation", "sales_corr_rolling_beta" });
    const corr_explain = try corr_plan.explain(gpa);
    defer gpa.free(corr_explain);
    try std.testing.expect(std.mem.indexOf(u8, corr_explain, "rolling_correlation_profile(sales_minus1,sales") != null);
    var rolling_corr = try corr_plan.collect();
    defer rolling_corr.deinit();
    try std.testing.expectEqual(@as(usize, 4), rolling_corr.height());
    try std.testing.expectEqual(@as(usize, 5), rolling_corr.width());
    const lazy_pair_count = try (try rolling_corr.column("sales_corr_rolling_pair_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_pair_count);
    const lazy_covariance = try (try rolling_corr.column("sales_corr_rolling_covariance")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_covariance);
    const lazy_correlation = try (try rolling_corr.column("sales_corr_rolling_correlation")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_correlation);
    const lazy_beta = try (try rolling_corr.column("sales_corr_rolling_beta")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_beta);
    const lazy_corr_validity = try (try rolling_corr.column("sales_corr_rolling_correlation")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_corr_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 2, 2 }, lazy_pair_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_corr_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), lazy_covariance[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_covariance[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_covariance[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_correlation[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_correlation[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_correlation[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_beta[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_beta[3], 1e-12);

    var expanding_corr_plan = try DeviceLazyFrame.init(gpa, table);
    defer expanding_corr_plan.deinit();
    try expanding_corr_plan.withColumnScalar("sales_minus1", "sales", f64, 1.0, .sub);
    try expanding_corr_plan.expandingCorrelationProfile("sales_minus1", "sales", "sales_corr", .{ .min_periods = 2 });
    try expanding_corr_plan.select(&.{ "sales", "sales_corr_expanding_pair_count", "sales_corr_expanding_covariance", "sales_corr_expanding_correlation", "sales_corr_expanding_beta" });
    const expanding_corr_explain = try expanding_corr_plan.explain(gpa);
    defer gpa.free(expanding_corr_explain);
    try std.testing.expect(std.mem.indexOf(u8, expanding_corr_explain, "expanding_correlation_profile(sales_minus1,sales") != null);
    var expanding_corr = try expanding_corr_plan.collect();
    defer expanding_corr.deinit();
    try std.testing.expectEqual(@as(usize, 4), expanding_corr.height());
    try std.testing.expectEqual(@as(usize, 5), expanding_corr.width());
    const lazy_expanding_pair_count = try (try expanding_corr.column("sales_corr_expanding_pair_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_pair_count);
    const lazy_expanding_covariance = try (try expanding_corr.column("sales_corr_expanding_covariance")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_covariance);
    const lazy_expanding_correlation = try (try expanding_corr.column("sales_corr_expanding_correlation")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_correlation);
    const lazy_expanding_beta = try (try expanding_corr.column("sales_corr_expanding_beta")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_beta);
    const lazy_expanding_corr_validity = try (try expanding_corr.column("sales_corr_expanding_correlation")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_corr_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4 }, lazy_expanding_pair_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_expanding_corr_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), lazy_expanding_covariance[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5555555555555554), lazy_expanding_covariance[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.6875), lazy_expanding_covariance[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_expanding_correlation[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_expanding_correlation[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_expanding_correlation[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_expanding_beta[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_expanding_beta[3], 1e-12);

    var expanding_fit_plan = try DeviceLazyFrame.init(gpa, table);
    defer expanding_fit_plan.deinit();
    try expanding_fit_plan.withColumnScalar("sales_minus1", "sales", f64, 1.0, .sub);
    try expanding_fit_plan.expandingLinearFitProfile("sales_minus1", "sales", "sales_fit", .{ .min_periods = 2 });
    try expanding_fit_plan.select(&.{ "sales", "sales_fit_expanding_pair_count", "sales_fit_expanding_slope", "sales_fit_expanding_intercept", "sales_fit_expanding_fitted", "sales_fit_expanding_residual" });
    const expanding_fit_explain = try expanding_fit_plan.explain(gpa);
    defer gpa.free(expanding_fit_explain);
    try std.testing.expect(std.mem.indexOf(u8, expanding_fit_explain, "expanding_linear_fit_profile(sales_minus1->sales") != null);
    var expanding_fit = try expanding_fit_plan.collect();
    defer expanding_fit.deinit();
    try std.testing.expectEqual(@as(usize, 4), expanding_fit.height());
    try std.testing.expectEqual(@as(usize, 6), expanding_fit.width());
    const lazy_expanding_fit_count = try (try expanding_fit.column("sales_fit_expanding_pair_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_fit_count);
    const lazy_expanding_fit_slope = try (try expanding_fit.column("sales_fit_expanding_slope")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_fit_slope);
    const lazy_expanding_fit_intercept = try (try expanding_fit.column("sales_fit_expanding_intercept")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_fit_intercept);
    const lazy_expanding_fit_fitted = try (try expanding_fit.column("sales_fit_expanding_fitted")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_fit_fitted);
    const lazy_expanding_fit_residual = try (try expanding_fit.column("sales_fit_expanding_residual")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_fit_residual);
    const lazy_expanding_fit_validity = try (try expanding_fit.column("sales_fit_expanding_slope")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_fit_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4 }, lazy_expanding_fit_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_expanding_fit_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_expanding_fit_slope[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_expanding_fit_slope[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_expanding_fit_slope[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_expanding_fit_intercept[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_expanding_fit_intercept[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), lazy_expanding_fit_fitted[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), lazy_expanding_fit_fitted[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 7.0), lazy_expanding_fit_fitted[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_expanding_fit_residual[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_expanding_fit_residual[3], 1e-12);

    var rolling_fit_plan = try DeviceLazyFrame.init(gpa, table);
    defer rolling_fit_plan.deinit();
    try rolling_fit_plan.withColumnScalar("sales_minus1", "sales", f64, 1.0, .sub);
    try rolling_fit_plan.rollingLinearFitProfile("sales_minus1", "sales", "sales_fit", .{ .window = 2, .min_periods = 2 });
    try rolling_fit_plan.select(&.{ "sales", "sales_fit_rolling_pair_count", "sales_fit_rolling_slope", "sales_fit_rolling_intercept", "sales_fit_rolling_fitted", "sales_fit_rolling_residual" });
    const rolling_fit_explain = try rolling_fit_plan.explain(gpa);
    defer gpa.free(rolling_fit_explain);
    try std.testing.expect(std.mem.indexOf(u8, rolling_fit_explain, "rolling_linear_fit_profile(sales_minus1->sales") != null);
    var rolling_fit = try rolling_fit_plan.collect();
    defer rolling_fit.deinit();
    try std.testing.expectEqual(@as(usize, 4), rolling_fit.height());
    try std.testing.expectEqual(@as(usize, 6), rolling_fit.width());
    const lazy_fit_count = try (try rolling_fit.column("sales_fit_rolling_pair_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_fit_count);
    const lazy_rolling_fit_slope = try (try rolling_fit.column("sales_fit_rolling_slope")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_fit_slope);
    const lazy_rolling_fit_intercept = try (try rolling_fit.column("sales_fit_rolling_intercept")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_fit_intercept);
    const lazy_rolling_fit_fitted = try (try rolling_fit.column("sales_fit_rolling_fitted")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_fit_fitted);
    const lazy_rolling_fit_residual = try (try rolling_fit.column("sales_fit_rolling_residual")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_fit_residual);
    const lazy_fit_validity = try (try rolling_fit.column("sales_fit_rolling_slope")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_fit_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 2, 2 }, lazy_fit_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_fit_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_fit_slope[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_fit_slope[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_fit_slope[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_fit_intercept[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_fit_intercept[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_fit_intercept[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), lazy_rolling_fit_fitted[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), lazy_rolling_fit_fitted[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 7.0), lazy_rolling_fit_fitted[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_rolling_fit_residual[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_rolling_fit_residual[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_rolling_fit_residual[3], 1e-12);

    var bucket_plan = try DeviceLazyFrame.init(gpa, table);
    defer bucket_plan.deinit();
    try bucket_plan.bucketProfile("sales", "sales", .{ .buckets = 2, .lower_quantile = 0.25, .upper_quantile = 0.75 });
    try bucket_plan.select(&.{ "sales", "sales_ecdf", "sales_bucket", "sales_lower_tail", "sales_upper_tail" });
    const bucket_explain = try bucket_plan.explain(gpa);
    defer gpa.free(bucket_explain);
    try std.testing.expect(std.mem.indexOf(u8, bucket_explain, "bucket_profile(sales") != null);
    var bucketed = try bucket_plan.collect();
    defer bucketed.deinit();
    try std.testing.expectEqual(@as(usize, 4), bucketed.height());
    try std.testing.expectEqual(@as(usize, 5), bucketed.width());
    const lazy_ecdf = try (try bucketed.column("sales_ecdf")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ecdf);
    const lazy_bucket = try (try bucketed.column("sales_bucket")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_bucket);
    const lazy_lower_tail = try (try bucketed.column("sales_lower_tail")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_lower_tail);
    const lazy_upper_tail = try (try bucketed.column("sales_upper_tail")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_upper_tail);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), lazy_ecdf[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_ecdf[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), lazy_ecdf[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_ecdf[3], 1e-12);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 1 }, lazy_bucket);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false }, lazy_lower_tail);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, true }, lazy_upper_tail);
}

test "device lazy frame collects groupby aggregations" {
    const gpa = std.testing.allocator;

    var store = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 1, 2, 2, 2 }, .cpu);
    defer store.deinit();
    var day = try DeviceColumn.fromSlice(i32, gpa, &.{ 10, 10, 10, 11, 11 }, .cpu);
    defer day.deinit();
    var sales = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 3.0, 5.0, 7.0, 11.0 }, .cpu);
    defer sales.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "store", .data = store },
        .{ .name = "day", .data = day },
        .{ .name = "sales", .data = sales },
    });
    defer table.deinit();

    var sum_plan = try DeviceLazyFrame.init(gpa, table);
    defer sum_plan.deinit();
    try sum_plan.filterColumnScalar("sales", f64, 2.5, .gt);
    try sum_plan.groupBySum("store", "sales", "sales_sum");
    const sum_explain = try sum_plan.explain(gpa);
    defer gpa.free(sum_explain);
    try std.testing.expect(std.mem.indexOf(u8, sum_explain, "group_by_sum(store") != null);
    var summed = try sum_plan.collect();
    defer summed.deinit();
    try std.testing.expectEqual(@as(usize, 2), summed.height());
    try std.testing.expectEqual(@as(usize, 2), summed.width());
    const sum_store = try (try summed.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(sum_store);
    const sum_values = try (try summed.column("sales_sum")).f64.toOwnedSlice(gpa);
    defer gpa.free(sum_values);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2 }, sum_store);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 23.0 }, sum_values);

    var stats_plan = try DeviceLazyFrame.init(gpa, table);
    defer stats_plan.deinit();
    try stats_plan.groupByStatsOn(&.{ "store", "day" }, "sales", "sales");
    const stats_explain = try stats_plan.explain(gpa);
    defer gpa.free(stats_explain);
    try std.testing.expect(std.mem.indexOf(u8, stats_explain, "group_by_stats_on([store,day]") != null);
    var stats = try stats_plan.collect();
    defer stats.deinit();
    try std.testing.expectEqual(@as(usize, 3), stats.height());
    try std.testing.expectEqual(@as(usize, 7), stats.width());
    const stats_store = try (try stats.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(stats_store);
    const stats_day = try (try stats.column("day")).i32.toOwnedSlice(gpa);
    defer gpa.free(stats_day);
    const stats_count = try (try stats.column("sales_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(stats_count);
    const stats_sum = try (try stats.column("sales_sum")).f64.toOwnedSlice(gpa);
    defer gpa.free(stats_sum);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2, 2 }, stats_store);
    try std.testing.expectEqualSlices(i32, &.{ 10, 10, 11 }, stats_day);
    try std.testing.expectEqualSlices(i64, &.{ 2, 1, 2 }, stats_count);
    try std.testing.expectEqualSlices(f64, &.{ 5.0, 5.0, 18.0 }, stats_sum);

    var profile_plan = try DeviceLazyFrame.init(gpa, table);
    defer profile_plan.deinit();
    try profile_plan.groupByProfile("store", "sales", "sales");
    const profile_explain = try profile_plan.explain(gpa);
    defer gpa.free(profile_explain);
    try std.testing.expect(std.mem.indexOf(u8, profile_explain, "group_by_profile(store") != null);
    var profile = try profile_plan.collect();
    defer profile.deinit();
    try std.testing.expectEqual(@as(usize, 2), profile.height());
    try std.testing.expectEqual(@as(usize, 8), profile.width());
    const profile_count = try (try profile.column("sales_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(profile_count);
    const profile_variance = try (try profile.column("sales_variance")).f64.toOwnedSlice(gpa);
    defer gpa.free(profile_variance);
    const profile_skewness = try (try profile.column("sales_skewness")).f64.toOwnedSlice(gpa);
    defer gpa.free(profile_skewness);
    const profile_kurtosis = try (try profile.column("sales_kurtosis")).f64.toOwnedSlice(gpa);
    defer gpa.free(profile_kurtosis);
    try std.testing.expectEqualSlices(i64, &.{ 2, 3 }, profile_count);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), profile_variance[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 6.222222222222222), profile_variance[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), profile_skewness[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.3818017741606059), profile_skewness[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -2.0), profile_kurtosis[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -1.5), profile_kurtosis[1], 1e-12);
}

test "device lazy frame collects multi-key joins" {
    const gpa = std.testing.allocator;

    var left_store = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 1, 2, 3 }, .cpu);
    defer left_store.deinit();
    var left_day = try DeviceColumn.fromSlice(i32, gpa, &.{ 10, 11, 10, 10 }, .cpu);
    defer left_day.deinit();
    var sales = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 3.0, 5.0, 7.0 }, .cpu);
    defer sales.deinit();
    var right_store = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 2, 9 }, .cpu);
    defer right_store.deinit();
    var right_day = try DeviceColumn.fromSlice(i32, gpa, &.{ 11, 10, 10 }, .cpu);
    defer right_day.deinit();
    var region = try DeviceColumn.fromSlice(i64, gpa, &.{ 100, 200, 900 }, .cpu);
    defer region.deinit();

    var left = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "store", .data = left_store },
        .{ .name = "day", .data = left_day },
        .{ .name = "sales", .data = sales },
    });
    defer left.deinit();
    var right = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "store", .data = right_store },
        .{ .name = "day", .data = right_day },
        .{ .name = "region", .data = region },
    });
    defer right.deinit();

    var joined_plan = try DeviceLazyFrame.init(gpa, left);
    defer joined_plan.deinit();
    try joined_plan.filterColumnScalar("sales", f64, 2.5, .gt);
    try joined_plan.innerJoinOn(right, &.{ "store", "day" }, &.{ "store", "day" }, .{});
    try joined_plan.select(&.{ "store", "day", "sales", "region" });
    const joined_explain = try joined_plan.explain(gpa);
    defer gpa.free(joined_explain);
    try std.testing.expect(std.mem.indexOf(u8, joined_explain, "inner_join_on(left=[store,day]") != null);
    var joined = try joined_plan.collect();
    defer joined.deinit();
    try std.testing.expectEqual(@as(usize, 2), joined.height());
    try std.testing.expectEqual(@as(usize, 4), joined.width());
    const joined_store = try (try joined.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(joined_store);
    const joined_sales = try (try joined.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(joined_sales);
    const joined_region = try (try joined.column("region")).i64.toOwnedSlice(gpa);
    defer gpa.free(joined_region);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2 }, joined_store);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 5.0 }, joined_sales);
    try std.testing.expectEqualSlices(i64, &.{ 100, 200 }, joined_region);

    var anti_plan = try DeviceLazyFrame.init(gpa, left);
    defer anti_plan.deinit();
    try anti_plan.antiJoinOn(right, &.{ "store", "day" }, &.{ "store", "day" });
    var anti = try anti_plan.collect();
    defer anti.deinit();
    try std.testing.expectEqual(@as(usize, 2), anti.height());
    const anti_store = try (try anti.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(anti_store);
    try std.testing.expectEqualSlices(i32, &.{ 1, 3 }, anti_store);
}

test "device lazy frame collects asof joins" {
    const gpa = std.testing.allocator;

    var left_time = try DeviceColumn.fromSlice(i64, gpa, &.{ 1, 5, 8, 12, 20 }, .cpu);
    defer left_time.deinit();
    var value = try DeviceColumn.fromSlice(f64, gpa, &.{ 10.0, 50.0, 80.0, 120.0, 200.0 }, .cpu);
    defer value.deinit();
    var right_time = try DeviceColumn.fromSliceWithValidity(i64, gpa, &.{ 2, 6, 10, 30 }, &.{ true, true, true, false }, .cpu);
    defer right_time.deinit();
    var quote = try DeviceColumn.fromSlice(i64, gpa, &.{ 20, 60, 100, 300 }, .cpu);
    defer quote.deinit();

    var left = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "time", .data = left_time },
        .{ .name = "value", .data = value },
    });
    defer left.deinit();
    var right = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "time", .data = right_time },
        .{ .name = "quote", .data = quote },
    });
    defer right.deinit();

    var plan = try DeviceLazyFrame.init(gpa, left);
    defer plan.deinit();
    try plan.filterColumnScalar("time", i64, 4, .ge);
    try plan.asofJoin(right, "time", "time", .{ .strategy = .nearest });
    try plan.select(&.{ "time", "value", "quote" });
    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "asof_join(time->time, strategy=nearest)") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 4), result.height());
    try std.testing.expectEqual(@as(usize, 3), result.width());
    const result_time = try (try result.column("time")).i64.toOwnedSlice(gpa);
    defer gpa.free(result_time);
    const result_quote = try (try result.column("quote")).i64.toOwnedSlice(gpa);
    defer gpa.free(result_quote);
    try std.testing.expectEqual(@as(usize, 0), (try result.column("quote")).nullCount());
    try std.testing.expectEqualSlices(i64, &.{ 5, 8, 12, 20 }, result_time);
    try std.testing.expectEqualSlices(i64, &.{ 60, 60, 100, 100 }, result_quote);
}
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
