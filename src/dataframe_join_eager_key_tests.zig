const std = @import("std");
const vectra = @import("vectra");

const DeviceColumn = vectra.DeviceColumn;
const DeviceDataFrame = vectra.DeviceDataFrame;
const DeviceDType = vectra.DeviceDType;

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
