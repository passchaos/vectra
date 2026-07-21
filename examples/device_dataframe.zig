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

    var filtered = try df.filter(&.{ true, false, true });
    defer filtered.deinit();
    const filtered_units = try filtered.column("units");

    var legacy = try filtered.toDataFrame();
    defer legacy.deinit();

    try std.testing.expectEqual(@as(usize, 3), df.height());
    try std.testing.expectEqual(@as(usize, 2), df.width());
    try std.testing.expectEqual(vx.DeviceDType.f64, view.columns[0].dtype);
    try std.testing.expectEqual(vx.DeviceValidityEncoding.bool_mask, view.columns[1].validity_encoding);
    try std.testing.expectEqual(@as(usize, 1), view.columns[1].null_count);
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
        filtered.height(),
        filtered_units.nullCount(),
    });
    try stdout.interface.flush();
}
