//! Device-aware DataFrame usage with cuDF-like owning columns and table views.
//!
//! The same API works for CPU, CUDA, and MPS devices.  This example keeps the
//! default CPU device so it can run on every development machine:
//!   zig build example-device-dataframe

const std = @import("std");
const vx = @import("vectra");

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;

    var sales = try vx.DeviceColumn.fromSlice(f64, allocator, &.{ 2.0, 3.0, 5.0 }, vx.cpu);
    defer sales.deinit();
    var units = try vx.DeviceColumn.fromSliceWithValidity(i64, allocator, &.{ 1, 2, 3 }, &.{ true, false, true }, vx.cpu);
    defer units.deinit();

    var df = try vx.DeviceDataFrame.init(allocator, &.{
        .{ .name = "sales", .data = sales },
        .{ .name = "units", .data = units },
    });
    defer df.deinit();

    var view = try df.view();
    defer view.deinit();

    var arrow_batch = try df.toArrowRecordBatch(allocator);
    defer arrow_batch.deinit(allocator);

    var expensive = try df.compareColumnScalar("sales", f64, 2.5, .gt);
    defer expensive.deinit();
    var doubled_sales = try df.binaryColumnScalar("sales", f64, 2.0, .mul);
    defer doubled_sales.deinit();
    var lazy = try vx.DeviceLazyFrame.init(allocator, df);
    defer lazy.deinit();
    try lazy.withColumnScalar("sales_x2", "sales", f64, 2.0, .mul);
    try lazy.withColumnCompareScalar("expensive", "sales_x2", f64, 6.0, .gt);
    try lazy.filterColumnScalar("sales", f64, 2.5, .gt);
    try lazy.sortBy("sales", .{ .descending = true });
    try lazy.select(&.{ "sales", "units", "sales_x2", "expensive" });
    try lazy.head(2);
    const lazy_explain = try lazy.explain(allocator);
    defer allocator.free(lazy_explain);
    var lazy_result = try lazy.collect();
    defer lazy_result.deinit();
    var lazy_group = try vx.DeviceLazyFrame.init(allocator, df);
    defer lazy_group.deinit();
    try lazy_group.filterColumnScalar("sales", f64, 1.0, .gt);
    try lazy_group.groupBySum("units", "sales", "lazy_sales_sum");
    const lazy_group_explain = try lazy_group.explain(allocator);
    defer allocator.free(lazy_group_explain);
    var lazy_grouped = try lazy_group.collect();
    defer lazy_grouped.deinit();

    var filtered = try df.filter(&.{ true, false, true });
    defer filtered.deinit();
    const filtered_units = try filtered.column("units");
    var expression_filtered = try df.filterColumnMask(expensive);
    defer expression_filtered.deinit();
    var sorted = try df.sortBy("sales", .{ .descending = true });
    defer sorted.deinit();
    const sorted_sales = try sorted.column("sales");
    var top2 = try df.topKBy("sales", 2, .{ .descending = true });
    defer top2.deinit();
    var grouped = try df.groupByCount("units", "rows");
    defer grouped.deinit();
    var summed = try df.groupBySum("units", "sales", "sales_sum");
    defer summed.deinit();
    var minned = try df.groupByMin("units", "sales", "sales_min");
    defer minned.deinit();
    var maxed = try df.groupByMax("units", "sales", "sales_max");
    defer maxed.deinit();
    var meaned = try df.groupByMean("units", "sales", "sales_mean");
    defer meaned.deinit();
    var stats = try df.groupByStats("units", "sales", "sales");
    defer stats.deinit();
    var stats_on = try df.groupByStatsOn(&.{"units"}, "sales", "sales");
    defer stats_on.deinit();
    var lookup_units = try vx.DeviceColumn.fromSliceWithValidity(i64, allocator, &.{ 1, 3, 99 }, &.{ true, true, false }, vx.cpu);
    defer lookup_units.deinit();
    var region = try vx.DeviceColumn.fromSlice(i64, allocator, &.{ 10, 30, 990 }, vx.cpu);
    defer region.deinit();
    var lookup = try vx.DeviceDataFrame.init(allocator, &.{
        .{ .name = "units", .data = lookup_units },
        .{ .name = "region", .data = region },
    });
    defer lookup.deinit();
    var joined = try df.innerJoin(lookup, "units", "units", .{});
    defer joined.deinit();
    var joined_on = try df.innerJoinOn(lookup, &.{"units"}, &.{"units"}, .{});
    defer joined_on.deinit();
    var left_joined = try df.leftJoin(lookup, "units", "units", .{});
    defer left_joined.deinit();
    var left_joined_on = try df.leftJoinOn(lookup, &.{"units"}, &.{"units"}, .{});
    defer left_joined_on.deinit();
    var full_joined = try df.fullJoin(lookup, "units", "units", .{});
    defer full_joined.deinit();
    var full_joined_on = try df.fullJoinOn(lookup, &.{"units"}, &.{"units"}, .{});
    defer full_joined_on.deinit();
    var semi_joined = try df.semiJoin(lookup, "units", "units");
    defer semi_joined.deinit();
    var semi_joined_on = try df.semiJoinOn(lookup, &.{"units"}, &.{"units"});
    defer semi_joined_on.deinit();
    var anti_joined = try df.antiJoin(lookup, "units", "units");
    defer anti_joined.deinit();
    var anti_joined_on = try df.antiJoinOn(lookup, &.{"units"}, &.{"units"});
    defer anti_joined_on.deinit();
    var asof_joined = try df.asofJoin(lookup, "units", "units", .{ .strategy = .nearest });
    defer asof_joined.deinit();
    const parquet_bytes = try df.toParquetBytes(allocator);
    defer allocator.free(parquet_bytes);
    var parquet_roundtrip = try vx.DeviceDataFrame.fromParquetBytes(allocator, parquet_bytes, vx.cpu);
    defer parquet_roundtrip.deinit();
    var parquet_pruned = try vx.DeviceDataFrame.fromParquetBytesPruned(
        allocator,
        parquet_bytes,
        "sales",
        .{ .f64 = .{ .min = 4.0 } },
        vx.cpu,
    );
    defer parquet_pruned.deinit();
    var parquet_scan = try vx.DeviceParquetScan.init(allocator, parquet_bytes, vx.cpu);
    defer parquet_scan.deinit();
    try parquet_scan.whereRange("sales", .{ .f64 = .{ .min = 4.0 } });
    try parquet_scan.select(&.{ "sales", "units" });
    const parquet_scan_plan = try parquet_scan.explain(allocator);
    defer allocator.free(parquet_scan_plan);
    var parquet_scanned = try parquet_scan.collect();
    defer parquet_scanned.deinit();

    var lazy_scan = try vx.DeviceLazyFrame.scanParquetBytes(allocator, parquet_bytes, vx.cpu);
    defer lazy_scan.deinit();
    try lazy_scan.filterColumnScalar("sales", f64, 4.0, .ge);
    try lazy_scan.select(&.{ "sales", "units" });
    const lazy_scan_plan = try lazy_scan.explain(allocator);
    defer allocator.free(lazy_scan_plan);
    var lazy_scanned = try lazy_scan.collect();
    defer lazy_scanned.deinit();

    var legacy = try filtered.toDataFrame();
    defer legacy.deinit();

    try std.testing.expectEqual(@as(usize, 3), df.height());
    try std.testing.expectEqual(@as(usize, 2), df.width());
    try std.testing.expectEqual(vx.DeviceDType.f64, view.columns[0].dtype);
    try std.testing.expectEqual(vx.DeviceValidityEncoding.bool_mask, view.columns[1].validity_encoding);
    try std.testing.expectEqual(@as(usize, 1), view.columns[1].null_count);
    try std.testing.expectEqual(@as(usize, 3), arrow_batch.row_count);
    try std.testing.expectEqual(@as(?usize, 1), arrow_batch.columnIndexByName("units"));
    try std.testing.expectEqual(@as(?i64, null), arrow_batch.columns[1].int64.value(1));
    try std.testing.expectEqual(@as(usize, 2), lazy_result.height());
    try std.testing.expectEqual(@as(usize, 4), lazy_result.width());
    try std.testing.expect(lazy_explain.len != 0);
    try std.testing.expect(std.mem.indexOf(u8, lazy_explain, "with_column_scalar(sales_x2") != null);
    const lazy_sales_x2 = try (try lazy_result.column("sales_x2")).f64.toOwnedSlice(allocator);
    defer allocator.free(lazy_sales_x2);
    try std.testing.expectEqualSlices(f64, &.{ 10.0, 6.0 }, lazy_sales_x2);
    try std.testing.expectEqual(@as(usize, 2), lazy_grouped.height());
    try std.testing.expect(std.mem.indexOf(u8, lazy_group_explain, "group_by_sum(units") != null);
    try std.testing.expectEqual(@as(usize, 2), expression_filtered.height());
    const doubled_values = try doubled_sales.f64.toOwnedSlice(allocator);
    defer allocator.free(doubled_values);
    try std.testing.expectEqualSlices(f64, &.{ 4.0, 6.0, 10.0 }, doubled_values);
    const sorted_values = try sorted_sales.f64.toOwnedSlice(allocator);
    defer allocator.free(sorted_values);
    try std.testing.expectEqualSlices(f64, &.{ 5.0, 3.0, 2.0 }, sorted_values);
    try std.testing.expectEqual(@as(usize, 2), top2.height());
    try std.testing.expectEqual(@as(usize, 2), grouped.height());
    try std.testing.expectEqual(@as(usize, 2), summed.height());
    try std.testing.expectEqual(@as(usize, 2), minned.height());
    try std.testing.expectEqual(@as(usize, 2), maxed.height());
    try std.testing.expectEqual(@as(usize, 2), meaned.height());
    try std.testing.expectEqual(@as(usize, 6), stats.width());
    try std.testing.expectEqual(@as(usize, 6), stats_on.width());
    try std.testing.expectEqual(@as(usize, 2), joined.height());
    try std.testing.expectEqual(@as(usize, 2), joined_on.height());
    try std.testing.expectEqual(df.height(), left_joined.height());
    try std.testing.expectEqual(df.height(), left_joined_on.height());
    try std.testing.expectEqual(@as(usize, 4), full_joined.height());
    try std.testing.expectEqual(@as(usize, 4), full_joined_on.height());
    try std.testing.expectEqual(@as(usize, 2), semi_joined.height());
    try std.testing.expectEqual(@as(usize, 2), semi_joined_on.height());
    try std.testing.expectEqual(@as(usize, 1), anti_joined.height());
    try std.testing.expectEqual(@as(usize, 1), anti_joined_on.height());
    try std.testing.expectEqual(df.height(), asof_joined.height());
    try std.testing.expectEqual(df.height(), parquet_roundtrip.height());
    try std.testing.expectEqual(df.width(), parquet_roundtrip.width());
    try std.testing.expectEqual(df.height(), parquet_pruned.height());
    try std.testing.expectEqual(@as(usize, 2), parquet_scanned.width());
    try std.testing.expect(parquet_scan_plan.len != 0);
    try std.testing.expectEqual(@as(usize, 2), lazy_scanned.width());
    try std.testing.expect(std.mem.indexOf(u8, lazy_scan_plan, "scan_pushdown: range=sales, projection=[sales,units]") != null);
    try std.testing.expectEqual(@as(usize, 2), filtered.height());
    try std.testing.expectEqual(@as(usize, 0), filtered_units.nullCount());
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 5.0 }, legacy.columns[0].f64);
    try std.testing.expectEqualSlices(i64, &.{ 1, 3 }, legacy.columns[1].i64);

    var stdout_buffer: [2048]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout.interface.print(
        \\
        \\{{
        \\  "example": "device_dataframe",
        \\  "device": "{s}",
        \\  "shape": [{d},{d}],
        \\  "sales_dtype": "{s}",
        \\  "units_nulls": {d},
        \\  "arrow_rows": {d},
        \\  "arrow_columns": {d},
        \\  "lazy_rows": {d},
        \\  "lazy_width": {d},
        \\  "lazy_plan_bytes": {d},
        \\  "lazy_grouped_rows": {d},
        \\  "expression_filtered_rows": {d},
        \\  "doubled_sales_last": {d:.1},
        \\  "sorted_sales_first": {d:.1},
        \\  "top2_rows": {d},
        \\  "grouped_rows": {d},
        \\  "summed_rows": {d},
        \\  "minned_rows": {d},
        \\  "maxed_rows": {d},
        \\  "meaned_rows": {d},
        \\  "stats_width": {d},
        \\  "stats_on_width": {d},
        \\
    , .{
        df.device.backendName(),
        df.height(),
        df.width(),
        view.columns[0].dtype.name(),
        view.columns[1].null_count,
        arrow_batch.row_count,
        arrow_batch.columnCount(),
        lazy_result.height(),
        lazy_result.width(),
        lazy_explain.len,
        lazy_grouped.height(),
        expression_filtered.height(),
        doubled_values[2],
        sorted_values[0],
        top2.height(),
        grouped.height(),
        summed.height(),
        minned.height(),
        maxed.height(),
        meaned.height(),
        stats.width(),
        stats_on.width(),
    });
    try stdout.interface.print(
        \\  "joined_rows": {d},
        \\  "joined_on_rows": {d},
        \\  "left_joined_rows": {d},
        \\  "left_joined_on_rows": {d},
        \\  "full_joined_rows": {d},
        \\  "full_joined_on_rows": {d},
        \\  "semi_joined_rows": {d},
        \\  "semi_joined_on_rows": {d},
        \\  "anti_joined_rows": {d},
        \\  "anti_joined_on_rows": {d},
        \\  "asof_joined_rows": {d},
        \\  "parquet_bytes": {d},
        \\  "parquet_rows": {d},
        \\  "parquet_pruned_rows": {d},
        \\  "parquet_scan_width": {d},
        \\  "lazy_scan_width": {d},
        \\  "filtered_rows": {d},
        \\  "filtered_units_nulls": {d},
        \\  "ok": true
        \\}}
        \\
    , .{
        joined.height(),
        joined_on.height(),
        left_joined.height(),
        left_joined_on.height(),
        full_joined.height(),
        full_joined_on.height(),
        semi_joined.height(),
        semi_joined_on.height(),
        anti_joined.height(),
        anti_joined_on.height(),
        asof_joined.height(),
        parquet_bytes.len,
        parquet_roundtrip.height(),
        parquet_pruned.height(),
        parquet_scanned.width(),
        lazy_scanned.width(),
        filtered.height(),
        filtered_units.nullCount(),
    });
    try stdout.interface.flush();
}
