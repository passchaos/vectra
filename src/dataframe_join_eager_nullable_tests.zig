const std = @import("std");
const vectra = @import("vectra");

const DeviceColumn = vectra.DeviceColumn;
const DeviceDataFrame = vectra.DeviceDataFrame;
const DeviceDType = vectra.DeviceDType;

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
