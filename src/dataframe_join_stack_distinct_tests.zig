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

    var distinct_last = try table.distinctOnLast(&.{"id"});
    defer distinct_last.deinit();
    const distinct_last_values = try (try distinct_last.column("value")).f64.toOwnedSlice(gpa);
    defer gpa.free(distinct_last_values);
    try std.testing.expectEqualSlices(f64, &.{ 99.0, 21.0, 30.0 }, distinct_last_values);

    var distinct_none = try table.dropDuplicatesOnNone(&.{"id"});
    defer distinct_none.deinit();
    try std.testing.expectEqual(@as(usize, 1), distinct_none.height());
    const distinct_none_ids = try (try distinct_none.column("id")).i32.toOwnedSlice(gpa);
    defer gpa.free(distinct_none_ids);
    const distinct_none_values = try (try distinct_none.column("value")).f64.toOwnedSlice(gpa);
    defer gpa.free(distinct_none_values);
    const distinct_none_validity = try (try distinct_none.column("value")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(distinct_none_validity);
    try std.testing.expectEqualSlices(i32, &.{3}, distinct_none_ids);
    try std.testing.expectEqualSlices(f64, &.{30.0}, distinct_none_values);
    try std.testing.expectEqualSlices(bool, &.{false}, distinct_none_validity);

    var full_distinct = try table.distinctRows();
    defer full_distinct.deinit();
    try std.testing.expectEqual(@as(usize, 4), full_distinct.height());

    var full_distinct_none = try table.distinctRowsNone();
    defer full_distinct_none.deinit();
    try std.testing.expectEqual(@as(usize, 4), full_distinct_none.height());

    var duplicate_flags = try table.withRowIsDuplicated(&.{"id"}, "id_is_duplicated");
    defer duplicate_flags.deinit();
    const id_is_duplicated = try (try duplicate_flags.column("id_is_duplicated")).bool.toOwnedSlice(gpa);
    defer gpa.free(id_is_duplicated);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, false }, id_is_duplicated);

    var unique_flags = try table.withRowIsUnique(&.{"id"}, "id_is_unique");
    defer unique_flags.deinit();
    const id_is_unique = try (try unique_flags.column("id_is_unique")).bool.toOwnedSlice(gpa);
    defer gpa.free(id_is_unique);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false, true }, id_is_unique);

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

    var last_plan = try DeviceLazyFrame.init(gpa, table);
    defer last_plan.deinit();
    try last_plan.dropDuplicatesOnLast(&.{"id"});
    try last_plan.select(&.{ "id", "value" });
    const last_explained = try last_plan.explain(gpa);
    defer gpa.free(last_explained);
    try std.testing.expect(std.mem.indexOf(u8, last_explained, "distinct_on_last([id])") != null);
    var lazy_last = try last_plan.collect();
    defer lazy_last.deinit();
    const lazy_last_values = try (try lazy_last.column("value")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_last_values);
    try std.testing.expectEqualSlices(f64, &.{ 99.0, 21.0, 30.0 }, lazy_last_values);

    var none_plan = try DeviceLazyFrame.init(gpa, table);
    defer none_plan.deinit();
    try none_plan.dropDuplicatesOnNone(&.{"id"});
    try none_plan.select(&.{ "id", "value" });
    const none_explained = try none_plan.explain(gpa);
    defer gpa.free(none_explained);
    try std.testing.expect(std.mem.indexOf(u8, none_explained, "distinct_on_none([id])") != null);
    var lazy_none = try none_plan.collect();
    defer lazy_none.deinit();
    try std.testing.expectEqual(@as(usize, 1), lazy_none.height());
    const lazy_none_ids = try (try lazy_none.column("id")).i32.toOwnedSlice(gpa);
    defer gpa.free(lazy_none_ids);
    const lazy_none_values = try (try lazy_none.column("value")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_none_values);
    const lazy_none_validity = try (try lazy_none.column("value")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_none_validity);
    try std.testing.expectEqualSlices(i32, &.{3}, lazy_none_ids);
    try std.testing.expectEqualSlices(f64, &.{30.0}, lazy_none_values);
    try std.testing.expectEqualSlices(bool, &.{false}, lazy_none_validity);

    var rows_none_plan = try DeviceLazyFrame.init(gpa, table);
    defer rows_none_plan.deinit();
    try rows_none_plan.distinctRowsNone();
    const rows_none_explained = try rows_none_plan.explain(gpa);
    defer gpa.free(rows_none_explained);
    try std.testing.expect(std.mem.indexOf(u8, rows_none_explained, "distinct_rows_none") != null);
    var lazy_rows_none = try rows_none_plan.collect();
    defer lazy_rows_none.deinit();
    try std.testing.expectEqual(@as(usize, 4), lazy_rows_none.height());

    var duplicate_mask_plan = try DeviceLazyFrame.init(gpa, table);
    defer duplicate_mask_plan.deinit();
    try duplicate_mask_plan.withRowIsDuplicated(&.{"id"}, "id_is_duplicated");
    try duplicate_mask_plan.withRowIsUnique(&.{"id"}, "id_is_unique");
    try duplicate_mask_plan.select(&.{ "id_is_duplicated", "id_is_unique" });
    const duplicate_mask_explained = try duplicate_mask_plan.explain(gpa);
    defer gpa.free(duplicate_mask_explained);
    try std.testing.expect(std.mem.indexOf(u8, duplicate_mask_explained, "row_is_duplicated([id]->id_is_duplicated)") != null);
    try std.testing.expect(std.mem.indexOf(u8, duplicate_mask_explained, "row_is_unique([id]->id_is_unique)") != null);
    var lazy_duplicate_flags = try duplicate_mask_plan.collect();
    defer lazy_duplicate_flags.deinit();
    const lazy_id_is_duplicated = try (try lazy_duplicate_flags.column("id_is_duplicated")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_id_is_duplicated);
    const lazy_id_is_unique = try (try lazy_duplicate_flags.column("id_is_unique")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_id_is_unique);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, false }, lazy_id_is_duplicated);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false, true }, lazy_id_is_unique);
}
