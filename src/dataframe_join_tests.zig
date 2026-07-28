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
