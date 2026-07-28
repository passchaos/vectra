const std = @import("std");
const vectra = @import("vectra");

const DeviceColumn = vectra.DeviceColumn;
const DeviceDataFrame = vectra.DeviceDataFrame;
const DeviceLazyFrame = vectra.DeviceLazyFrame;

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
