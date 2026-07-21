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

    var filtered = try df.filter(&.{ true, false, true });
    defer filtered.deinit();
    const filtered_units = try filtered.column("units");
    var expression_filtered = try df.filterColumnMask(expensive);
    defer expression_filtered.deinit();

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
    try std.testing.expectEqual(@as(usize, 2), expression_filtered.height());
    const doubled_values = try doubled_sales.f64.toOwnedSlice(allocator);
    defer allocator.free(doubled_values);
    try std.testing.expectEqualSlices(f64, &.{ 4.0, 6.0, 10.0 }, doubled_values);
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
        \\  "expression_filtered_rows": {d},
        \\  "doubled_sales_last": {d:.1},
        \\  "filtered_rows": {d},
        \\  "filtered_units_nulls": {d},
        \\  "ok": true
        \\}}
        \\
    , .{
        df.device.backendName(),
        df.height(),
        df.width(),
        view.columns[0].dtype.name(),
        view.columns[1].null_count,
        arrow_batch.row_count,
        arrow_batch.columnCount(),
        expression_filtered.height(),
        doubled_values[2],
        filtered.height(),
        filtered_units.nullCount(),
    });
    try stdout.interface.flush();
}
