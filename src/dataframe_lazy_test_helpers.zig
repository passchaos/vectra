const std = @import("std");
const vectra = @import("vectra");

const DeviceColumn = vectra.DeviceColumn;
const DeviceDataFrame = vectra.DeviceDataFrame;

pub fn lazyCollectTable(gpa: std.mem.Allocator) !DeviceDataFrame {
    var sales = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 3.0, 5.0, 7.0 }, .cpu);
    defer sales.deinit();
    var units = try DeviceColumn.fromSlice(i64, gpa, &.{ 1, 2, 3, 4 }, .cpu);
    defer units.deinit();
    var active = try DeviceColumn.fromSlice(bool, gpa, &.{ true, false, true, true }, .cpu);
    defer active.deinit();

    return DeviceDataFrame.init(gpa, &.{
        .{ .name = "sales", .data = sales },
        .{ .name = "units", .data = units },
        .{ .name = "active", .data = active },
    });
}
