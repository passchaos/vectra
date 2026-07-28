const std = @import("std");
const vectra = @import("vectra");

const DeviceColumn = vectra.DeviceColumn;
const DeviceDataFrame = vectra.DeviceDataFrame;
const DeviceLazyFrame = vectra.DeviceLazyFrame;

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
