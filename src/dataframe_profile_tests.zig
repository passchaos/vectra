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
const DeviceLazyWeightedGroupByAggregation = vectra.DeviceLazyWeightedGroupByAggregation;
const DeviceLazyPairGroupByAggregation = vectra.DeviceLazyPairGroupByAggregation;
const DeviceLazyWeightedPairGroupByAggregation = vectra.DeviceLazyWeightedPairGroupByAggregation;

fn expectApproxOrNan(expected: f64, actual: f64) !void {
    if (std.math.isNan(expected)) {
        try std.testing.expect(std.math.isNan(actual));
    } else {
        try std.testing.expectApproxEqAbs(expected, actual, 1e-12);
    }
}

fn expectF64SliceApproxOrNan(expected: []const f64, actual: []const f64) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |expected_item, actual_item| {
        try expectApproxOrNan(expected_item, actual_item);
    }
}

fn expectF64ColumnApproxOrNan(frame: anytype, allocator: std.mem.Allocator, name: []const u8, expected: []const f64) !void {
    const values = try (try frame.column(name)).f64.toOwnedSlice(allocator);
    defer allocator.free(values);
    try expectF64SliceApproxOrNan(expected, values);
}

fn expectNullableI64Column(frame: anytype, allocator: std.mem.Allocator, name: []const u8, expected_values: []const i64, expected_validity: []const bool) !void {
    const column = try frame.column(name);
    const values = try column.i64.toOwnedSlice(allocator);
    defer allocator.free(values);
    const validity = try column.i64.validity.?.toOwnedSlice(allocator);
    defer allocator.free(validity);
    try std.testing.expectEqualSlices(i64, expected_values, values);
    try std.testing.expectEqualSlices(bool, expected_validity, validity);
}

fn expectNullableBoolColumn(frame: anytype, allocator: std.mem.Allocator, name: []const u8, expected_values: []const bool, expected_validity: []const bool) !void {
    const column = try frame.column(name);
    const values = try column.bool.toOwnedSlice(allocator);
    defer allocator.free(values);
    const validity = try column.bool.validity.?.toOwnedSlice(allocator);
    defer allocator.free(validity);
    try std.testing.expectEqualSlices(bool, expected_values, values);
    try std.testing.expectEqualSlices(bool, expected_validity, validity);
}

fn expectF64ColumnWithValidity(frame: anytype, allocator: std.mem.Allocator, name: []const u8, expected_values: []const f64, expected_validity: []const bool) !void {
    const column = try frame.column(name);
    const values = try column.f64.toOwnedSlice(allocator);
    defer allocator.free(values);
    try std.testing.expectEqualSlices(f64, expected_values, values);
    if (column.f64.validity) |mask| {
        const validity = try mask.toOwnedSlice(allocator);
        defer allocator.free(validity);
        try std.testing.expectEqualSlices(bool, expected_validity, validity);
    } else {
        for (expected_validity) |valid| try std.testing.expect(valid);
    }
}

fn expectF64ColumnApproxOrNanWithValidity(frame: anytype, allocator: std.mem.Allocator, name: []const u8, expected_values: []const f64, expected_validity: []const bool) !void {
    const column = try frame.column(name);
    const values = try column.f64.toOwnedSlice(allocator);
    defer allocator.free(values);
    try expectF64SliceApproxOrNan(expected_values, values);
    if (column.f64.validity) |mask| {
        const validity = try mask.toOwnedSlice(allocator);
        defer allocator.free(validity);
        try std.testing.expectEqualSlices(bool, expected_validity, validity);
    } else {
        for (expected_validity) |valid| try std.testing.expect(valid);
    }
}

test "device dataframe rolling bool profile handles nullable windows" {
    const gpa = std.testing.allocator;

    var active = try DeviceColumn.fromSliceWithValidity(bool, gpa, &.{ true, false, true, true, true, false }, &.{ true, true, false, true, true, true }, .cpu);
    defer active.deinit();
    var id = try DeviceColumn.fromSlice(i64, gpa, &.{ 1, 2, 3, 4, 5, 6 }, .cpu);
    defer id.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "active", .data = active },
        .{ .name = "id", .data = id },
    });
    defer table.deinit();

    var profiled = try table.rollingBoolProfile("active", "active", .{ .window = 3, .min_periods = 2 });
    defer profiled.deinit();
    try std.testing.expectEqual(@as(usize, 7), profiled.width());

    const true_count = try (try profiled.column("active_rolling_true_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(true_count);
    const false_count = try (try profiled.column("active_rolling_false_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(false_count);
    const true_rate = try (try profiled.column("active_rolling_true_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(true_rate);
    const rolling_any = try (try profiled.column("active_rolling_any")).bool.toOwnedSlice(gpa);
    defer gpa.free(rolling_any);
    const rolling_all = try (try profiled.column("active_rolling_all")).bool.toOwnedSlice(gpa);
    defer gpa.free(rolling_all);
    const bool_validity = try (try profiled.column("active_rolling_true_rate")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(bool_validity);

    try std.testing.expectEqualSlices(i64, &.{ 1, 1, 1, 1, 2, 2 }, true_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 1, 1, 0, 1 }, false_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true, true, true }, bool_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), true_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), true_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), true_rate[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), true_rate[5], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true, true, true }, rolling_any);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false, true, false }, rolling_all);

    var expanding_bool = try table.expandingBoolProfile("active", "active", .{ .min_periods = 2 });
    defer expanding_bool.deinit();
    try std.testing.expectEqual(@as(usize, 7), expanding_bool.width());
    const expanding_true_count = try (try expanding_bool.column("active_expanding_true_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(expanding_true_count);
    const expanding_false_count = try (try expanding_bool.column("active_expanding_false_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(expanding_false_count);
    const expanding_true_rate = try (try expanding_bool.column("active_expanding_true_rate")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_true_rate);
    const expanding_any = try (try expanding_bool.column("active_expanding_any")).bool.toOwnedSlice(gpa);
    defer gpa.free(expanding_any);
    const expanding_all = try (try expanding_bool.column("active_expanding_all")).bool.toOwnedSlice(gpa);
    defer gpa.free(expanding_all);
    const expanding_bool_validity = try (try expanding_bool.column("active_expanding_true_rate")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(expanding_bool_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 1, 1, 2, 3, 3 }, expanding_true_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 1, 1, 1, 2 }, expanding_false_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true, true, true }, expanding_bool_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_true_rate[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), expanding_true_rate[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), expanding_true_rate[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.6), expanding_true_rate[5], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true, true, true }, expanding_any);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false, false, false }, expanding_all);
}

test "device dataframe groupby aggregations on fixed-width columns" {
    @setEvalBranchQuota(4000);
    const gpa = std.testing.allocator;

    var bool_key = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 1, 2, 2, 3 }, .cpu);
    defer bool_key.deinit();
    var bool_day = try DeviceColumn.fromSlice(i32, gpa, &.{ 10, 10, 10, 11, 11 }, .cpu);
    defer bool_day.deinit();
    var active_grouped = try DeviceColumn.fromSliceWithValidity(bool, gpa, &.{ false, true, false, true, true }, &.{ true, true, true, true, false }, .cpu);
    defer active_grouped.deinit();
    var bool_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "store", .data = bool_key },
        .{ .name = "day", .data = bool_day },
        .{ .name = "active", .data = active_grouped },
    });
    defer bool_table.deinit();

    var any_active = try bool_table.groupByAny("store", "active", "any_active");
    defer any_active.deinit();
    const any_active_values = try (try any_active.column("any_active")).bool.toOwnedSlice(gpa);
    defer gpa.free(any_active_values);
    try std.testing.expectEqualSlices(bool, &.{ true, true }, any_active_values);

    var all_active = try bool_table.groupByAll("store", "active", "all_active");
    defer all_active.deinit();
    const all_active_values = try (try all_active.column("all_active")).bool.toOwnedSlice(gpa);
    defer gpa.free(all_active_values);
    try std.testing.expectEqualSlices(bool, &.{ false, false }, all_active_values);

    var any_active_on = try bool_table.groupByAnyOn(&.{ "store", "day" }, "active", "any_active_on");
    defer any_active_on.deinit();
    const any_active_on_values = try (try any_active_on.column("any_active_on")).bool.toOwnedSlice(gpa);
    defer gpa.free(any_active_on_values);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, any_active_on_values);

    var all_active_on = try bool_table.groupByAllOn(&.{ "store", "day" }, "active", "all_active_on");
    defer all_active_on.deinit();
    const all_active_on_values = try (try all_active_on.column("all_active_on")).bool.toOwnedSlice(gpa);
    defer gpa.free(all_active_on_values);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true }, all_active_on_values);

    var active_true_counts = try bool_table.groupByTrueCount("store", "active", "active_true_count");
    defer active_true_counts.deinit();
    const active_true_count_values = try (try active_true_counts.column("active_true_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(active_true_count_values);
    try std.testing.expectEqualSlices(i64, &.{ 1, 1 }, active_true_count_values);

    var active_false_counts = try bool_table.groupByFalseCount("store", "active", "active_false_count");
    defer active_false_counts.deinit();
    const active_false_count_values = try (try active_false_counts.column("active_false_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(active_false_count_values);
    try std.testing.expectEqualSlices(i64, &.{ 1, 1 }, active_false_count_values);

    var active_true_ratios = try bool_table.groupByTrueRatio("store", "active", "active_true_ratio");
    defer active_true_ratios.deinit();
    const active_true_ratio_values = try (try active_true_ratios.column("active_true_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(active_true_ratio_values);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), active_true_ratio_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), active_true_ratio_values[1], 1e-12);

    var active_false_ratios = try bool_table.groupByFalseRatio("store", "active", "active_false_ratio");
    defer active_false_ratios.deinit();
    const active_false_ratio_values = try (try active_false_ratios.column("active_false_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(active_false_ratio_values);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), active_false_ratio_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), active_false_ratio_values[1], 1e-12);

    var active_true_counts_on = try bool_table.groupByTrueCountOn(&.{ "store", "day" }, "active", "active_true_count_on");
    defer active_true_counts_on.deinit();
    const active_true_count_on_values = try (try active_true_counts_on.column("active_true_count_on")).i64.toOwnedSlice(gpa);
    defer gpa.free(active_true_count_on_values);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 1 }, active_true_count_on_values);

    var active_false_ratios_on = try bool_table.groupByFalseRatioOn(&.{ "store", "day" }, "active", "active_false_ratio_on");
    defer active_false_ratios_on.deinit();
    const active_false_ratio_on_values = try (try active_false_ratios_on.column("active_false_ratio_on")).f64.toOwnedSlice(gpa);
    defer gpa.free(active_false_ratio_on_values);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), active_false_ratio_on_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), active_false_ratio_on_values[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), active_false_ratio_on_values[2], 1e-12);

    var active_cum_any = try bool_table.withGroupCumulativeAny("store", "active", "active_cum_any");
    defer active_cum_any.deinit();
    try expectNullableBoolColumn(active_cum_any, gpa, "active_cum_any", &.{ false, true, false, true, false }, &.{ true, true, true, true, false });

    var active_cum_all = try bool_table.withGroupCumulativeAll("store", "active", "active_cum_all");
    defer active_cum_all.deinit();
    try expectNullableBoolColumn(active_cum_all, gpa, "active_cum_all", &.{ false, false, false, false, false }, &.{ true, true, true, true, false });

    var active_cum_true_count = try bool_table.withGroupCumulativeTrueCount("store", "active", "active_cum_true_count");
    defer active_cum_true_count.deinit();
    try expectNullableI64Column(active_cum_true_count, gpa, "active_cum_true_count", &.{ 0, 1, 0, 1, 0 }, &.{ true, true, true, true, false });

    var active_cum_false_count = try bool_table.withGroupCumulativeFalseCount("store", "active", "active_cum_false_count");
    defer active_cum_false_count.deinit();
    try expectNullableI64Column(active_cum_false_count, gpa, "active_cum_false_count", &.{ 1, 1, 1, 1, 0 }, &.{ true, true, true, true, false });

    var active_cum_true_ratio = try bool_table.withGroupCumulativeTrueRatio("store", "active", "active_cum_true_ratio");
    defer active_cum_true_ratio.deinit();
    try expectF64ColumnWithValidity(active_cum_true_ratio, gpa, "active_cum_true_ratio", &.{ 0.0, 0.5, 0.0, 0.5, 0.0 }, &.{ true, true, true, true, false });

    var active_cum_false_ratio = try bool_table.withGroupCumulativeFalseRatio("store", "active", "active_cum_false_ratio");
    defer active_cum_false_ratio.deinit();
    try expectF64ColumnWithValidity(active_cum_false_ratio, gpa, "active_cum_false_ratio", &.{ 1.0, 0.5, 1.0, 0.5, 0.0 }, &.{ true, true, true, true, false });

    var active_cum_first_true = try bool_table.withGroupCumulativeFirstTrueIndex("store", "active", "active_cum_first_true");
    defer active_cum_first_true.deinit();
    try expectNullableI64Column(active_cum_first_true, gpa, "active_cum_first_true", &.{ 0, 1, 0, 3, 0 }, &.{ false, true, false, true, false });

    var active_cum_last_true = try bool_table.withGroupCumulativeLastTrueIndex("store", "active", "active_cum_last_true");
    defer active_cum_last_true.deinit();
    try expectNullableI64Column(active_cum_last_true, gpa, "active_cum_last_true", &.{ 0, 1, 0, 3, 0 }, &.{ false, true, false, true, false });

    var active_cum_first_false = try bool_table.withGroupCumulativeFirstFalseIndex("store", "active", "active_cum_first_false");
    defer active_cum_first_false.deinit();
    try expectNullableI64Column(active_cum_first_false, gpa, "active_cum_first_false", &.{ 0, 0, 2, 2, 0 }, &.{ true, true, true, true, false });

    var active_cum_last_false = try bool_table.withGroupCumulativeLastFalseIndex("store", "active", "active_cum_last_false");
    defer active_cum_last_false.deinit();
    try expectNullableI64Column(active_cum_last_false, gpa, "active_cum_last_false", &.{ 0, 0, 2, 2, 0 }, &.{ true, true, true, true, false });

    var active_first_true_indices = try bool_table.groupByFirstTrueIndex("store", "active", "active_first_true_index");
    defer active_first_true_indices.deinit();
    try expectNullableI64Column(active_first_true_indices, gpa, "active_first_true_index", &.{ 1, 3, 0 }, &.{ true, true, false });

    var active_last_true_indices = try bool_table.groupByLastTrueIndex("store", "active", "active_last_true_index");
    defer active_last_true_indices.deinit();
    try expectNullableI64Column(active_last_true_indices, gpa, "active_last_true_index", &.{ 1, 3, 0 }, &.{ true, true, false });

    var active_first_false_indices = try bool_table.groupByFirstFalseIndex("store", "active", "active_first_false_index");
    defer active_first_false_indices.deinit();
    try expectNullableI64Column(active_first_false_indices, gpa, "active_first_false_index", &.{ 0, 2, 0 }, &.{ true, true, false });

    var active_last_false_indices = try bool_table.groupByLastFalseIndex("store", "active", "active_last_false_index");
    defer active_last_false_indices.deinit();
    try expectNullableI64Column(active_last_false_indices, gpa, "active_last_false_index", &.{ 0, 2, 0 }, &.{ true, true, false });

    var active_valids = try bool_table.groupByValidCount("store", "active", "active_valid_count");
    defer active_valids.deinit();
    const active_valid_values = try (try active_valids.column("active_valid_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(active_valid_values);
    try std.testing.expectEqualSlices(i64, &.{ 2, 2, 0 }, active_valid_values);

    var active_nulls = try bool_table.groupByNullCount("store", "active", "active_null_count");
    defer active_nulls.deinit();
    const active_null_values = try (try active_nulls.column("active_null_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(active_null_values);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1 }, active_null_values);

    var active_valids_on = try bool_table.groupByValidCountOn(&.{ "store", "day" }, "active", "active_valid_count_on");
    defer active_valids_on.deinit();
    const active_valid_on_values = try (try active_valids_on.column("active_valid_count_on")).i64.toOwnedSlice(gpa);
    defer gpa.free(active_valid_on_values);
    try std.testing.expectEqualSlices(i64, &.{ 2, 1, 1, 0 }, active_valid_on_values);

    var active_nulls_on = try bool_table.groupByNullCountOn(&.{ "store", "day" }, "active", "active_null_count_on");
    defer active_nulls_on.deinit();
    const active_null_on_values = try (try active_nulls_on.column("active_null_count_on")).i64.toOwnedSlice(gpa);
    defer gpa.free(active_null_on_values);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 1 }, active_null_on_values);

    var active_any_valid = try bool_table.groupByAnyValid("store", "active", "active_any_valid");
    defer active_any_valid.deinit();
    const active_any_valid_values = try (try active_any_valid.column("active_any_valid")).bool.toOwnedSlice(gpa);
    defer gpa.free(active_any_valid_values);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false }, active_any_valid_values);

    var active_all_valid = try bool_table.groupByAllValid("store", "active", "active_all_valid");
    defer active_all_valid.deinit();
    const active_all_valid_values = try (try active_all_valid.column("active_all_valid")).bool.toOwnedSlice(gpa);
    defer gpa.free(active_all_valid_values);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false }, active_all_valid_values);

    var active_any_null = try bool_table.groupByAnyNull("store", "active", "active_any_null");
    defer active_any_null.deinit();
    const active_any_null_values = try (try active_any_null.column("active_any_null")).bool.toOwnedSlice(gpa);
    defer gpa.free(active_any_null_values);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true }, active_any_null_values);

    var active_all_null = try bool_table.groupByAllNull("store", "active", "active_all_null");
    defer active_all_null.deinit();
    const active_all_null_values = try (try active_all_null.column("active_all_null")).bool.toOwnedSlice(gpa);
    defer gpa.free(active_all_null_values);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true }, active_all_null_values);

    var active_first_valid_indices = try bool_table.groupByFirstValidIndex("store", "active", "active_first_valid_index");
    defer active_first_valid_indices.deinit();
    try expectNullableI64Column(active_first_valid_indices, gpa, "active_first_valid_index", &.{ 0, 2, 0 }, &.{ true, true, false });

    var active_last_valid_indices = try bool_table.groupByLastValidIndex("store", "active", "active_last_valid_index");
    defer active_last_valid_indices.deinit();
    try expectNullableI64Column(active_last_valid_indices, gpa, "active_last_valid_index", &.{ 1, 3, 0 }, &.{ true, true, false });

    var active_first_null_indices = try bool_table.groupByFirstNullIndex("store", "active", "active_first_null_index");
    defer active_first_null_indices.deinit();
    try expectNullableI64Column(active_first_null_indices, gpa, "active_first_null_index", &.{ 0, 0, 4 }, &.{ false, false, true });

    var active_last_null_indices = try bool_table.groupByLastNullIndex("store", "active", "active_last_null_index");
    defer active_last_null_indices.deinit();
    try expectNullableI64Column(active_last_null_indices, gpa, "active_last_null_index", &.{ 0, 0, 4 }, &.{ false, false, true });

    var active_valid_ratios = try bool_table.groupByValidRatio("store", "active", "active_valid_ratio");
    defer active_valid_ratios.deinit();
    const active_valid_ratio_values = try (try active_valid_ratios.column("active_valid_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(active_valid_ratio_values);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), active_valid_ratio_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), active_valid_ratio_values[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), active_valid_ratio_values[2], 1e-12);

    var active_null_ratios = try bool_table.groupByNullRatio("store", "active", "active_null_ratio");
    defer active_null_ratios.deinit();
    const active_null_ratio_values = try (try active_null_ratios.column("active_null_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(active_null_ratio_values);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), active_null_ratio_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), active_null_ratio_values[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), active_null_ratio_values[2], 1e-12);

    var active_null_ratios_on = try bool_table.groupByNullRatioOn(&.{ "store", "day" }, "active", "active_null_ratio_on");
    defer active_null_ratios_on.deinit();
    const active_null_ratio_on_values = try (try active_null_ratios_on.column("active_null_ratio_on")).f64.toOwnedSlice(gpa);
    defer gpa.free(active_null_ratio_on_values);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), active_null_ratio_on_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), active_null_ratio_on_values[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), active_null_ratio_on_values[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), active_null_ratio_on_values[3], 1e-12);

    const smallest_subnormal: f64 = @bitCast(@as(u64, 1));
    var quality_key = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 1, 1, 1, 2, 2, 2, 3, 3, 4 }, .cpu);
    defer quality_key.deinit();
    var quality_day = try DeviceColumn.fromSlice(i32, gpa, &.{ 10, 10, 11, 11, 10, 10, 11, 10, 10, 10 }, .cpu);
    defer quality_day.deinit();
    var quality_values_buffer = [_]f64{
        1.0,
        std.math.nan(f64),
        std.math.inf(f64),
        -std.math.inf(f64),
        0.0,
        smallest_subnormal,
        5.0,
        std.math.nan(f64),
        std.math.inf(f64),
        42.0,
    };
    var quality_value = try DeviceColumn.fromSliceWithValidity(f64, gpa, &quality_values_buffer, &.{ true, true, true, true, true, true, false, false, true, false }, .cpu);
    defer quality_value.deinit();
    var quality_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "bucket", .data = quality_key },
        .{ .name = "day", .data = quality_day },
        .{ .name = "metric", .data = quality_value },
    });
    defer quality_table.deinit();

    var quality_index_key = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3 }, .cpu);
    defer quality_index_key.deinit();
    var quality_index_values_buffer = [_]f64{
        std.math.nan(f64),
        std.math.inf(f64),
        -std.math.inf(f64),
        std.math.nan(f64),
        std.math.inf(f64),
        1.0,
        smallest_subnormal,
        5.0,
        -std.math.inf(f64),
        std.math.inf(f64),
        7.0,
    };
    var quality_index_value = try DeviceColumn.fromSliceWithValidity(f64, gpa, &quality_index_values_buffer, &.{ true, true, true, true, true, true, true, true, true, true, false }, .cpu);
    defer quality_index_value.deinit();
    var quality_index_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "bucket", .data = quality_index_key },
        .{ .name = "metric", .data = quality_index_value },
    });
    defer quality_index_table.deinit();

    var metric_nan_counts = try quality_table.groupByNaNCount("bucket", "metric", "metric_nan_count");
    defer metric_nan_counts.deinit();
    const metric_nan_count_values = try (try metric_nan_counts.column("metric_nan_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(metric_nan_count_values);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 0, 0 }, metric_nan_count_values);

    var metric_inf_counts = try quality_table.groupByInfCount("bucket", "metric", "metric_inf_count");
    defer metric_inf_counts.deinit();
    const metric_inf_count_values = try (try metric_inf_counts.column("metric_inf_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(metric_inf_count_values);
    try std.testing.expectEqualSlices(i64, &.{ 2, 0, 1, 0 }, metric_inf_count_values);

    var metric_positive_inf_counts = try quality_table.groupByPositiveInfCount("bucket", "metric", "metric_positive_inf_count");
    defer metric_positive_inf_counts.deinit();
    const metric_positive_inf_count_values = try (try metric_positive_inf_counts.column("metric_positive_inf_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(metric_positive_inf_count_values);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 1, 0 }, metric_positive_inf_count_values);

    var metric_negative_inf_counts = try quality_table.groupByNegativeInfCount("bucket", "metric", "metric_negative_inf_count");
    defer metric_negative_inf_counts.deinit();
    const metric_negative_inf_count_values = try (try metric_negative_inf_counts.column("metric_negative_inf_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(metric_negative_inf_count_values);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 0, 0 }, metric_negative_inf_count_values);

    var metric_finite_counts = try quality_table.groupByFiniteCount("bucket", "metric", "metric_finite_count");
    defer metric_finite_counts.deinit();
    const metric_finite_count_values = try (try metric_finite_counts.column("metric_finite_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(metric_finite_count_values);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 0, 0 }, metric_finite_count_values);

    var metric_normal_counts = try quality_table.groupByNormalCount("bucket", "metric", "metric_normal_count");
    defer metric_normal_counts.deinit();
    const metric_normal_count_values = try (try metric_normal_counts.column("metric_normal_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(metric_normal_count_values);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 0, 0 }, metric_normal_count_values);

    var metric_subnormal_counts = try quality_table.groupBySubnormalCount("bucket", "metric", "metric_subnormal_count");
    defer metric_subnormal_counts.deinit();
    const metric_subnormal_count_values = try (try metric_subnormal_counts.column("metric_subnormal_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(metric_subnormal_count_values);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 0 }, metric_subnormal_count_values);

    var metric_non_finite_counts = try quality_table.groupByNonFiniteCount("bucket", "metric", "metric_non_finite_count");
    defer metric_non_finite_counts.deinit();
    const metric_non_finite_count_values = try (try metric_non_finite_counts.column("metric_non_finite_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(metric_non_finite_count_values);
    try std.testing.expectEqualSlices(i64, &.{ 3, 0, 1, 0 }, metric_non_finite_count_values);

    const quality_cum_validity = [_]bool{ true, true, true, true, true, true, false, false, true, false };
    const metric_cum_nan_count_expected = [_]i64{ 0, 1, 1, 1, 0, 0, 0, 0, 0, 0 };
    const metric_cum_inf_count_expected = [_]i64{ 0, 0, 1, 2, 0, 0, 0, 0, 1, 0 };
    const metric_cum_positive_inf_count_expected = [_]i64{ 0, 0, 1, 1, 0, 0, 0, 0, 1, 0 };
    const metric_cum_negative_inf_count_expected = [_]i64{ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 };
    const metric_cum_finite_count_expected = [_]i64{ 1, 1, 1, 1, 1, 2, 0, 0, 0, 0 };
    const metric_cum_normal_count_expected = [_]i64{ 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 };
    const metric_cum_subnormal_count_expected = [_]i64{ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 };
    const metric_cum_non_finite_count_expected = [_]i64{ 0, 1, 2, 3, 0, 0, 0, 0, 1, 0 };
    const metric_cum_zero_count_expected = [_]i64{ 0, 0, 0, 0, 1, 1, 0, 0, 0, 0 };
    const metric_cum_positive_zero_count_expected = [_]i64{ 0, 0, 0, 0, 1, 1, 0, 0, 0, 0 };
    const metric_cum_negative_zero_count_expected = [_]i64{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    const metric_cum_non_zero_count_expected = [_]i64{ 1, 2, 3, 4, 0, 1, 0, 0, 1, 0 };
    const metric_cum_positive_count_expected = [_]i64{ 1, 1, 2, 2, 0, 1, 0, 0, 1, 0 };
    const metric_cum_signbit_count_expected = [_]i64{ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 };
    const metric_cum_negative_count_expected = [_]i64{ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 };
    const metric_cum_nan_ratio_expected = [_]f64{ 0.0, 0.5, 1.0 / 3.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    const metric_cum_inf_ratio_expected = [_]f64{ 0.0, 0.0, 1.0 / 3.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 };
    const metric_cum_positive_inf_ratio_expected = [_]f64{ 0.0, 0.0, 1.0 / 3.0, 0.25, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 };
    const metric_cum_negative_inf_ratio_expected = [_]f64{ 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    const metric_cum_finite_ratio_expected = [_]f64{ 1.0, 0.5, 1.0 / 3.0, 0.25, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0 };
    const metric_cum_normal_ratio_expected = [_]f64{ 1.0, 0.5, 1.0 / 3.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    const metric_cum_subnormal_ratio_expected = [_]f64{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0 };
    const metric_cum_non_finite_ratio_expected = [_]f64{ 0.0, 0.5, 2.0 / 3.0, 0.75, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 };
    const metric_cum_zero_ratio_expected = [_]f64{ 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0 };
    const metric_cum_positive_zero_ratio_expected = [_]f64{ 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0 };
    const metric_cum_negative_zero_ratio_expected = [_]f64{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    const metric_cum_non_zero_ratio_expected = [_]f64{ 1.0, 1.0, 1.0, 1.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0 };
    const metric_cum_positive_ratio_expected = [_]f64{ 1.0, 0.5, 2.0 / 3.0, 0.5, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0 };
    const metric_cum_signbit_ratio_expected = [_]f64{ 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    const metric_cum_negative_ratio_expected = [_]f64{ 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

    var metric_cum_nan_count = try quality_table.withGroupCumulativeNaNCount("bucket", "metric", "metric_cum_nan_count");
    defer metric_cum_nan_count.deinit();
    try expectNullableI64Column(metric_cum_nan_count, gpa, "metric_cum_nan_count", &metric_cum_nan_count_expected, &quality_cum_validity);

    var metric_cum_inf_count = try quality_table.withGroupCumulativeInfCount("bucket", "metric", "metric_cum_inf_count");
    defer metric_cum_inf_count.deinit();
    try expectNullableI64Column(metric_cum_inf_count, gpa, "metric_cum_inf_count", &metric_cum_inf_count_expected, &quality_cum_validity);

    var metric_cum_positive_inf_count = try quality_table.withGroupCumulativePositiveInfCount("bucket", "metric", "metric_cum_positive_inf_count");
    defer metric_cum_positive_inf_count.deinit();
    try expectNullableI64Column(metric_cum_positive_inf_count, gpa, "metric_cum_positive_inf_count", &metric_cum_positive_inf_count_expected, &quality_cum_validity);

    var metric_cum_negative_inf_count = try quality_table.withGroupCumulativeNegativeInfCount("bucket", "metric", "metric_cum_negative_inf_count");
    defer metric_cum_negative_inf_count.deinit();
    try expectNullableI64Column(metric_cum_negative_inf_count, gpa, "metric_cum_negative_inf_count", &metric_cum_negative_inf_count_expected, &quality_cum_validity);

    var metric_cum_finite_count = try quality_table.withGroupCumulativeFiniteCount("bucket", "metric", "metric_cum_finite_count");
    defer metric_cum_finite_count.deinit();
    try expectNullableI64Column(metric_cum_finite_count, gpa, "metric_cum_finite_count", &metric_cum_finite_count_expected, &quality_cum_validity);

    var metric_cum_normal_count = try quality_table.withGroupCumulativeNormalCount("bucket", "metric", "metric_cum_normal_count");
    defer metric_cum_normal_count.deinit();
    try expectNullableI64Column(metric_cum_normal_count, gpa, "metric_cum_normal_count", &metric_cum_normal_count_expected, &quality_cum_validity);

    var metric_cum_subnormal_count = try quality_table.withGroupCumulativeSubnormalCount("bucket", "metric", "metric_cum_subnormal_count");
    defer metric_cum_subnormal_count.deinit();
    try expectNullableI64Column(metric_cum_subnormal_count, gpa, "metric_cum_subnormal_count", &metric_cum_subnormal_count_expected, &quality_cum_validity);

    var metric_cum_non_finite_count = try quality_table.withGroupCumulativeNonFiniteCount("bucket", "metric", "metric_cum_non_finite_count");
    defer metric_cum_non_finite_count.deinit();
    try expectNullableI64Column(metric_cum_non_finite_count, gpa, "metric_cum_non_finite_count", &metric_cum_non_finite_count_expected, &quality_cum_validity);

    var metric_cum_zero_count = try quality_table.withGroupCumulativeZeroCount("bucket", "metric", "metric_cum_zero_count");
    defer metric_cum_zero_count.deinit();
    try expectNullableI64Column(metric_cum_zero_count, gpa, "metric_cum_zero_count", &metric_cum_zero_count_expected, &quality_cum_validity);

    var metric_cum_positive_zero_count = try quality_table.withGroupCumulativePositiveZeroCount("bucket", "metric", "metric_cum_positive_zero_count");
    defer metric_cum_positive_zero_count.deinit();
    try expectNullableI64Column(metric_cum_positive_zero_count, gpa, "metric_cum_positive_zero_count", &metric_cum_positive_zero_count_expected, &quality_cum_validity);

    var metric_cum_negative_zero_count = try quality_table.withGroupCumulativeNegativeZeroCount("bucket", "metric", "metric_cum_negative_zero_count");
    defer metric_cum_negative_zero_count.deinit();
    try expectNullableI64Column(metric_cum_negative_zero_count, gpa, "metric_cum_negative_zero_count", &metric_cum_negative_zero_count_expected, &quality_cum_validity);

    var metric_cum_non_zero_count = try quality_table.withGroupCumulativeNonZeroCount("bucket", "metric", "metric_cum_non_zero_count");
    defer metric_cum_non_zero_count.deinit();
    try expectNullableI64Column(metric_cum_non_zero_count, gpa, "metric_cum_non_zero_count", &metric_cum_non_zero_count_expected, &quality_cum_validity);

    var metric_cum_positive_count = try quality_table.withGroupCumulativePositiveCount("bucket", "metric", "metric_cum_positive_count");
    defer metric_cum_positive_count.deinit();
    try expectNullableI64Column(metric_cum_positive_count, gpa, "metric_cum_positive_count", &metric_cum_positive_count_expected, &quality_cum_validity);

    var metric_cum_signbit_count = try quality_table.withGroupCumulativeSignBitCount("bucket", "metric", "metric_cum_signbit_count");
    defer metric_cum_signbit_count.deinit();
    try expectNullableI64Column(metric_cum_signbit_count, gpa, "metric_cum_signbit_count", &metric_cum_signbit_count_expected, &quality_cum_validity);

    var metric_cum_negative_count = try quality_table.withGroupCumulativeNegativeCount("bucket", "metric", "metric_cum_negative_count");
    defer metric_cum_negative_count.deinit();
    try expectNullableI64Column(metric_cum_negative_count, gpa, "metric_cum_negative_count", &metric_cum_negative_count_expected, &quality_cum_validity);

    var metric_cum_nan_ratio = try quality_table.withGroupCumulativeNaNRatio("bucket", "metric", "metric_cum_nan_ratio");
    defer metric_cum_nan_ratio.deinit();
    try expectF64ColumnWithValidity(metric_cum_nan_ratio, gpa, "metric_cum_nan_ratio", &metric_cum_nan_ratio_expected, &quality_cum_validity);

    var metric_cum_inf_ratio = try quality_table.withGroupCumulativeInfRatio("bucket", "metric", "metric_cum_inf_ratio");
    defer metric_cum_inf_ratio.deinit();
    try expectF64ColumnWithValidity(metric_cum_inf_ratio, gpa, "metric_cum_inf_ratio", &metric_cum_inf_ratio_expected, &quality_cum_validity);

    var metric_cum_positive_inf_ratio = try quality_table.withGroupCumulativePositiveInfRatio("bucket", "metric", "metric_cum_positive_inf_ratio");
    defer metric_cum_positive_inf_ratio.deinit();
    try expectF64ColumnWithValidity(metric_cum_positive_inf_ratio, gpa, "metric_cum_positive_inf_ratio", &metric_cum_positive_inf_ratio_expected, &quality_cum_validity);

    var metric_cum_negative_inf_ratio = try quality_table.withGroupCumulativeNegativeInfRatio("bucket", "metric", "metric_cum_negative_inf_ratio");
    defer metric_cum_negative_inf_ratio.deinit();
    try expectF64ColumnWithValidity(metric_cum_negative_inf_ratio, gpa, "metric_cum_negative_inf_ratio", &metric_cum_negative_inf_ratio_expected, &quality_cum_validity);

    var metric_cum_finite_ratio = try quality_table.withGroupCumulativeFiniteRatio("bucket", "metric", "metric_cum_finite_ratio");
    defer metric_cum_finite_ratio.deinit();
    try expectF64ColumnWithValidity(metric_cum_finite_ratio, gpa, "metric_cum_finite_ratio", &metric_cum_finite_ratio_expected, &quality_cum_validity);

    var metric_cum_normal_ratio = try quality_table.withGroupCumulativeNormalRatio("bucket", "metric", "metric_cum_normal_ratio");
    defer metric_cum_normal_ratio.deinit();
    try expectF64ColumnWithValidity(metric_cum_normal_ratio, gpa, "metric_cum_normal_ratio", &metric_cum_normal_ratio_expected, &quality_cum_validity);

    var metric_cum_subnormal_ratio = try quality_table.withGroupCumulativeSubnormalRatio("bucket", "metric", "metric_cum_subnormal_ratio");
    defer metric_cum_subnormal_ratio.deinit();
    try expectF64ColumnWithValidity(metric_cum_subnormal_ratio, gpa, "metric_cum_subnormal_ratio", &metric_cum_subnormal_ratio_expected, &quality_cum_validity);

    var metric_cum_non_finite_ratio = try quality_table.withGroupCumulativeNonFiniteRatio("bucket", "metric", "metric_cum_non_finite_ratio");
    defer metric_cum_non_finite_ratio.deinit();
    try expectF64ColumnWithValidity(metric_cum_non_finite_ratio, gpa, "metric_cum_non_finite_ratio", &metric_cum_non_finite_ratio_expected, &quality_cum_validity);

    var metric_cum_zero_ratio = try quality_table.withGroupCumulativeZeroRatio("bucket", "metric", "metric_cum_zero_ratio");
    defer metric_cum_zero_ratio.deinit();
    try expectF64ColumnWithValidity(metric_cum_zero_ratio, gpa, "metric_cum_zero_ratio", &metric_cum_zero_ratio_expected, &quality_cum_validity);

    var metric_cum_positive_zero_ratio = try quality_table.withGroupCumulativePositiveZeroRatio("bucket", "metric", "metric_cum_positive_zero_ratio");
    defer metric_cum_positive_zero_ratio.deinit();
    try expectF64ColumnWithValidity(metric_cum_positive_zero_ratio, gpa, "metric_cum_positive_zero_ratio", &metric_cum_positive_zero_ratio_expected, &quality_cum_validity);

    var metric_cum_negative_zero_ratio = try quality_table.withGroupCumulativeNegativeZeroRatio("bucket", "metric", "metric_cum_negative_zero_ratio");
    defer metric_cum_negative_zero_ratio.deinit();
    try expectF64ColumnWithValidity(metric_cum_negative_zero_ratio, gpa, "metric_cum_negative_zero_ratio", &metric_cum_negative_zero_ratio_expected, &quality_cum_validity);

    var metric_cum_non_zero_ratio = try quality_table.withGroupCumulativeNonZeroRatio("bucket", "metric", "metric_cum_non_zero_ratio");
    defer metric_cum_non_zero_ratio.deinit();
    try expectF64ColumnWithValidity(metric_cum_non_zero_ratio, gpa, "metric_cum_non_zero_ratio", &metric_cum_non_zero_ratio_expected, &quality_cum_validity);

    var metric_cum_positive_ratio = try quality_table.withGroupCumulativePositiveRatio("bucket", "metric", "metric_cum_positive_ratio");
    defer metric_cum_positive_ratio.deinit();
    try expectF64ColumnWithValidity(metric_cum_positive_ratio, gpa, "metric_cum_positive_ratio", &metric_cum_positive_ratio_expected, &quality_cum_validity);

    var metric_cum_signbit_ratio = try quality_table.withGroupCumulativeSignBitRatio("bucket", "metric", "metric_cum_signbit_ratio");
    defer metric_cum_signbit_ratio.deinit();
    try expectF64ColumnWithValidity(metric_cum_signbit_ratio, gpa, "metric_cum_signbit_ratio", &metric_cum_signbit_ratio_expected, &quality_cum_validity);

    var metric_cum_negative_ratio = try quality_table.withGroupCumulativeNegativeRatio("bucket", "metric", "metric_cum_negative_ratio");
    defer metric_cum_negative_ratio.deinit();
    try expectF64ColumnWithValidity(metric_cum_negative_ratio, gpa, "metric_cum_negative_ratio", &metric_cum_negative_ratio_expected, &quality_cum_validity);

    const ratio_nan = std.math.nan(f64);
    const metric_nan_ratio_expected = [_]f64{ 0.25, 0.0, 0.0, ratio_nan };
    var metric_nan_ratios = try quality_table.groupByNaNRatio("bucket", "metric", "metric_nan_ratio");
    defer metric_nan_ratios.deinit();
    try expectF64ColumnApproxOrNan(metric_nan_ratios, gpa, "metric_nan_ratio", &metric_nan_ratio_expected);

    const metric_inf_ratio_expected = [_]f64{ 0.5, 0.0, 1.0, ratio_nan };
    var metric_inf_ratios = try quality_table.groupByInfRatio("bucket", "metric", "metric_inf_ratio");
    defer metric_inf_ratios.deinit();
    try expectF64ColumnApproxOrNan(metric_inf_ratios, gpa, "metric_inf_ratio", &metric_inf_ratio_expected);

    const metric_positive_inf_ratio_expected = [_]f64{ 0.25, 0.0, 1.0, ratio_nan };
    var metric_positive_inf_ratios = try quality_table.groupByPositiveInfRatio("bucket", "metric", "metric_positive_inf_ratio");
    defer metric_positive_inf_ratios.deinit();
    try expectF64ColumnApproxOrNan(metric_positive_inf_ratios, gpa, "metric_positive_inf_ratio", &metric_positive_inf_ratio_expected);

    const metric_negative_inf_ratio_expected = [_]f64{ 0.25, 0.0, 0.0, ratio_nan };
    var metric_negative_inf_ratios = try quality_table.groupByNegativeInfRatio("bucket", "metric", "metric_negative_inf_ratio");
    defer metric_negative_inf_ratios.deinit();
    try expectF64ColumnApproxOrNan(metric_negative_inf_ratios, gpa, "metric_negative_inf_ratio", &metric_negative_inf_ratio_expected);

    const metric_finite_ratio_expected = [_]f64{ 0.25, 1.0, 0.0, ratio_nan };
    var metric_finite_ratios = try quality_table.groupByFiniteRatio("bucket", "metric", "metric_finite_ratio");
    defer metric_finite_ratios.deinit();
    try expectF64ColumnApproxOrNan(metric_finite_ratios, gpa, "metric_finite_ratio", &metric_finite_ratio_expected);

    const metric_normal_ratio_expected = [_]f64{ 0.25, 0.0, 0.0, ratio_nan };
    var metric_normal_ratios = try quality_table.groupByNormalRatio("bucket", "metric", "metric_normal_ratio");
    defer metric_normal_ratios.deinit();
    try expectF64ColumnApproxOrNan(metric_normal_ratios, gpa, "metric_normal_ratio", &metric_normal_ratio_expected);

    const metric_subnormal_ratio_expected = [_]f64{ 0.0, 0.5, 0.0, ratio_nan };
    var metric_subnormal_ratios = try quality_table.groupBySubnormalRatio("bucket", "metric", "metric_subnormal_ratio");
    defer metric_subnormal_ratios.deinit();
    try expectF64ColumnApproxOrNan(metric_subnormal_ratios, gpa, "metric_subnormal_ratio", &metric_subnormal_ratio_expected);

    const metric_non_finite_ratio_expected = [_]f64{ 0.75, 0.0, 1.0, ratio_nan };
    var metric_non_finite_ratios = try quality_table.groupByNonFiniteRatio("bucket", "metric", "metric_non_finite_ratio");
    defer metric_non_finite_ratios.deinit();
    try expectF64ColumnApproxOrNan(metric_non_finite_ratios, gpa, "metric_non_finite_ratio", &metric_non_finite_ratio_expected);

    var metric_zero_counts = try quality_table.groupByZeroCount("bucket", "metric", "metric_zero_count");
    defer metric_zero_counts.deinit();
    const metric_zero_count_values = try (try metric_zero_counts.column("metric_zero_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(metric_zero_count_values);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 0 }, metric_zero_count_values);

    const metric_zero_ratio_expected = [_]f64{ 0.0, 0.5, 0.0, ratio_nan };
    var metric_zero_ratios = try quality_table.groupByZeroRatio("bucket", "metric", "metric_zero_ratio");
    defer metric_zero_ratios.deinit();
    try expectF64ColumnApproxOrNan(metric_zero_ratios, gpa, "metric_zero_ratio", &metric_zero_ratio_expected);

    var metric_non_zero_counts = try quality_table.groupByNonZeroCount("bucket", "metric", "metric_non_zero_count");
    defer metric_non_zero_counts.deinit();
    const metric_non_zero_count_values = try (try metric_non_zero_counts.column("metric_non_zero_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(metric_non_zero_count_values);
    try std.testing.expectEqualSlices(i64, &.{ 4, 1, 1, 0 }, metric_non_zero_count_values);

    const metric_non_zero_ratio_expected = [_]f64{ 1.0, 0.5, 1.0, ratio_nan };
    var metric_non_zero_ratios = try quality_table.groupByNonZeroRatio("bucket", "metric", "metric_non_zero_ratio");
    defer metric_non_zero_ratios.deinit();
    try expectF64ColumnApproxOrNan(metric_non_zero_ratios, gpa, "metric_non_zero_ratio", &metric_non_zero_ratio_expected);

    var metric_positive_counts = try quality_table.groupByPositiveCount("bucket", "metric", "metric_positive_count");
    defer metric_positive_counts.deinit();
    const metric_positive_count_values = try (try metric_positive_counts.column("metric_positive_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(metric_positive_count_values);
    try std.testing.expectEqualSlices(i64, &.{ 2, 1, 1, 0 }, metric_positive_count_values);

    const metric_positive_ratio_expected = [_]f64{ 0.5, 0.5, 1.0, ratio_nan };
    var metric_positive_ratios = try quality_table.groupByPositiveRatio("bucket", "metric", "metric_positive_ratio");
    defer metric_positive_ratios.deinit();
    try expectF64ColumnApproxOrNan(metric_positive_ratios, gpa, "metric_positive_ratio", &metric_positive_ratio_expected);

    var metric_signbit_counts = try quality_table.groupBySignBitCount("bucket", "metric", "metric_signbit_count");
    defer metric_signbit_counts.deinit();
    const metric_signbit_count_values = try (try metric_signbit_counts.column("metric_signbit_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(metric_signbit_count_values);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 0, 0 }, metric_signbit_count_values);

    const metric_signbit_ratio_expected = [_]f64{ 0.25, 0.0, 0.0, ratio_nan };
    var metric_signbit_ratios = try quality_table.groupBySignBitRatio("bucket", "metric", "metric_signbit_ratio");
    defer metric_signbit_ratios.deinit();
    try expectF64ColumnApproxOrNan(metric_signbit_ratios, gpa, "metric_signbit_ratio", &metric_signbit_ratio_expected);

    var metric_negative_counts = try quality_table.groupByNegativeCount("bucket", "metric", "metric_negative_count");
    defer metric_negative_counts.deinit();
    const metric_negative_count_values = try (try metric_negative_counts.column("metric_negative_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(metric_negative_count_values);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 0, 0 }, metric_negative_count_values);

    const metric_negative_ratio_expected = [_]f64{ 0.25, 0.0, 0.0, ratio_nan };
    var metric_negative_ratios = try quality_table.groupByNegativeRatio("bucket", "metric", "metric_negative_ratio");
    defer metric_negative_ratios.deinit();
    try expectF64ColumnApproxOrNan(metric_negative_ratios, gpa, "metric_negative_ratio", &metric_negative_ratio_expected);

    var first_nan_indices = try quality_index_table.groupByFirstNaNIndex("bucket", "metric", "first_nan_index");
    defer first_nan_indices.deinit();
    try expectNullableI64Column(first_nan_indices, gpa, "first_nan_index", &.{ 0, 0, 0 }, &.{ true, false, false });

    var last_nan_indices = try quality_index_table.groupByLastNaNIndex("bucket", "metric", "last_nan_index");
    defer last_nan_indices.deinit();
    try expectNullableI64Column(last_nan_indices, gpa, "last_nan_index", &.{ 3, 0, 0 }, &.{ true, false, false });

    var first_inf_indices = try quality_index_table.groupByFirstInfIndex("bucket", "metric", "first_inf_index");
    defer first_inf_indices.deinit();
    try expectNullableI64Column(first_inf_indices, gpa, "first_inf_index", &.{ 1, 8, 0 }, &.{ true, true, false });

    var last_inf_indices = try quality_index_table.groupByLastInfIndex("bucket", "metric", "last_inf_index");
    defer last_inf_indices.deinit();
    try expectNullableI64Column(last_inf_indices, gpa, "last_inf_index", &.{ 4, 9, 0 }, &.{ true, true, false });

    var first_positive_inf_indices = try quality_index_table.groupByFirstPositiveInfIndex("bucket", "metric", "first_positive_inf_index");
    defer first_positive_inf_indices.deinit();
    try expectNullableI64Column(first_positive_inf_indices, gpa, "first_positive_inf_index", &.{ 1, 9, 0 }, &.{ true, true, false });

    var last_positive_inf_indices = try quality_index_table.groupByLastPositiveInfIndex("bucket", "metric", "last_positive_inf_index");
    defer last_positive_inf_indices.deinit();
    try expectNullableI64Column(last_positive_inf_indices, gpa, "last_positive_inf_index", &.{ 4, 9, 0 }, &.{ true, true, false });

    var first_negative_inf_indices = try quality_index_table.groupByFirstNegativeInfIndex("bucket", "metric", "first_negative_inf_index");
    defer first_negative_inf_indices.deinit();
    try expectNullableI64Column(first_negative_inf_indices, gpa, "first_negative_inf_index", &.{ 2, 8, 0 }, &.{ true, true, false });

    var last_negative_inf_indices = try quality_index_table.groupByLastNegativeInfIndex("bucket", "metric", "last_negative_inf_index");
    defer last_negative_inf_indices.deinit();
    try expectNullableI64Column(last_negative_inf_indices, gpa, "last_negative_inf_index", &.{ 2, 8, 0 }, &.{ true, true, false });

    var first_finite_indices = try quality_index_table.groupByFirstFiniteIndex("bucket", "metric", "first_finite_index");
    defer first_finite_indices.deinit();
    try expectNullableI64Column(first_finite_indices, gpa, "first_finite_index", &.{ 5, 7, 0 }, &.{ true, true, false });

    var last_finite_indices = try quality_index_table.groupByLastFiniteIndex("bucket", "metric", "last_finite_index");
    defer last_finite_indices.deinit();
    try expectNullableI64Column(last_finite_indices, gpa, "last_finite_index", &.{ 6, 7, 0 }, &.{ true, true, false });

    var first_normal_indices = try quality_index_table.groupByFirstNormalIndex("bucket", "metric", "first_normal_index");
    defer first_normal_indices.deinit();
    try expectNullableI64Column(first_normal_indices, gpa, "first_normal_index", &.{ 5, 7, 0 }, &.{ true, true, false });

    var last_normal_indices = try quality_index_table.groupByLastNormalIndex("bucket", "metric", "last_normal_index");
    defer last_normal_indices.deinit();
    try expectNullableI64Column(last_normal_indices, gpa, "last_normal_index", &.{ 5, 7, 0 }, &.{ true, true, false });

    var first_subnormal_indices = try quality_index_table.groupByFirstSubnormalIndex("bucket", "metric", "first_subnormal_index");
    defer first_subnormal_indices.deinit();
    try expectNullableI64Column(first_subnormal_indices, gpa, "first_subnormal_index", &.{ 6, 0, 0 }, &.{ true, false, false });

    var last_subnormal_indices = try quality_index_table.groupByLastSubnormalIndex("bucket", "metric", "last_subnormal_index");
    defer last_subnormal_indices.deinit();
    try expectNullableI64Column(last_subnormal_indices, gpa, "last_subnormal_index", &.{ 6, 0, 0 }, &.{ true, false, false });

    var first_non_finite_indices = try quality_index_table.groupByFirstNonFiniteIndex("bucket", "metric", "first_non_finite_index");
    defer first_non_finite_indices.deinit();
    try expectNullableI64Column(first_non_finite_indices, gpa, "first_non_finite_index", &.{ 0, 8, 0 }, &.{ true, true, false });

    var last_non_finite_indices = try quality_index_table.groupByLastNonFiniteIndex("bucket", "metric", "last_non_finite_index");
    defer last_non_finite_indices.deinit();
    try expectNullableI64Column(last_non_finite_indices, gpa, "last_non_finite_index", &.{ 4, 9, 0 }, &.{ true, true, false });

    var signed_zero_key = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 1, 1, 1, 2, 2, 2, 3 }, .cpu);
    defer signed_zero_key.deinit();
    var signed_zero_values_buffer = [_]f64{ 0.0, -0.0, 1.0, 0.0, -0.0, 0.0, -0.0, 5.0 };
    var signed_zero_value = try DeviceColumn.fromSliceWithValidity(f64, gpa, &signed_zero_values_buffer, &.{ true, true, true, true, true, true, true, false }, .cpu);
    defer signed_zero_value.deinit();
    var signed_zero_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "bucket", .data = signed_zero_key },
        .{ .name = "metric", .data = signed_zero_value },
    });
    defer signed_zero_table.deinit();

    var positive_zero_counts = try signed_zero_table.groupByPositiveZeroCount("bucket", "metric", "positive_zero_count");
    defer positive_zero_counts.deinit();
    const positive_zero_count_values = try (try positive_zero_counts.column("positive_zero_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(positive_zero_count_values);
    try std.testing.expectEqualSlices(i64, &.{ 2, 1, 0 }, positive_zero_count_values);

    var negative_zero_counts = try signed_zero_table.groupByNegativeZeroCount("bucket", "metric", "negative_zero_count");
    defer negative_zero_counts.deinit();
    const negative_zero_count_values = try (try negative_zero_counts.column("negative_zero_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(negative_zero_count_values);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 0 }, negative_zero_count_values);

    const positive_zero_ratio_expected = [_]f64{ 0.5, 1.0 / 3.0, ratio_nan };
    var positive_zero_ratios = try signed_zero_table.groupByPositiveZeroRatio("bucket", "metric", "positive_zero_ratio");
    defer positive_zero_ratios.deinit();
    try expectF64ColumnApproxOrNan(positive_zero_ratios, gpa, "positive_zero_ratio", &positive_zero_ratio_expected);

    const negative_zero_ratio_expected = [_]f64{ 0.25, 2.0 / 3.0, ratio_nan };
    var negative_zero_ratios = try signed_zero_table.groupByNegativeZeroRatio("bucket", "metric", "negative_zero_ratio");
    defer negative_zero_ratios.deinit();
    try expectF64ColumnApproxOrNan(negative_zero_ratios, gpa, "negative_zero_ratio", &negative_zero_ratio_expected);

    var first_positive_zero_indices = try signed_zero_table.groupByFirstPositiveZeroIndex("bucket", "metric", "first_positive_zero_index");
    defer first_positive_zero_indices.deinit();
    const first_positive_zero_index_values = try (try first_positive_zero_indices.column("first_positive_zero_index")).i64.toOwnedSlice(gpa);
    defer gpa.free(first_positive_zero_index_values);
    const first_positive_zero_index_validity = try (try first_positive_zero_indices.column("first_positive_zero_index")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(first_positive_zero_index_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 5, 0 }, first_positive_zero_index_values);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false }, first_positive_zero_index_validity);

    var last_positive_zero_indices = try signed_zero_table.groupByLastPositiveZeroIndex("bucket", "metric", "last_positive_zero_index");
    defer last_positive_zero_indices.deinit();
    const last_positive_zero_index_values = try (try last_positive_zero_indices.column("last_positive_zero_index")).i64.toOwnedSlice(gpa);
    defer gpa.free(last_positive_zero_index_values);
    const last_positive_zero_index_validity = try (try last_positive_zero_indices.column("last_positive_zero_index")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(last_positive_zero_index_validity);
    try std.testing.expectEqualSlices(i64, &.{ 3, 5, 0 }, last_positive_zero_index_values);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false }, last_positive_zero_index_validity);

    var first_negative_zero_indices = try signed_zero_table.groupByFirstNegativeZeroIndex("bucket", "metric", "first_negative_zero_index");
    defer first_negative_zero_indices.deinit();
    const first_negative_zero_index_values = try (try first_negative_zero_indices.column("first_negative_zero_index")).i64.toOwnedSlice(gpa);
    defer gpa.free(first_negative_zero_index_values);
    const first_negative_zero_index_validity = try (try first_negative_zero_indices.column("first_negative_zero_index")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(first_negative_zero_index_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 4, 0 }, first_negative_zero_index_values);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false }, first_negative_zero_index_validity);

    var last_negative_zero_indices = try signed_zero_table.groupByLastNegativeZeroIndex("bucket", "metric", "last_negative_zero_index");
    defer last_negative_zero_indices.deinit();
    const last_negative_zero_index_values = try (try last_negative_zero_indices.column("last_negative_zero_index")).i64.toOwnedSlice(gpa);
    defer gpa.free(last_negative_zero_index_values);
    const last_negative_zero_index_validity = try (try last_negative_zero_indices.column("last_negative_zero_index")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(last_negative_zero_index_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 6, 0 }, last_negative_zero_index_values);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false }, last_negative_zero_index_validity);

    var sign_index_key = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 1, 1, 1, 2, 2, 2, 3 }, .cpu);
    defer sign_index_key.deinit();
    var sign_index_values_buffer = [_]f64{ 0.0, -2.0, 3.0, -4.0, 5.0, 0.0, -0.0, 9.0 };
    var sign_index_value = try DeviceColumn.fromSliceWithValidity(f64, gpa, &sign_index_values_buffer, &.{ true, true, true, true, true, true, true, false }, .cpu);
    defer sign_index_value.deinit();
    var sign_index_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "bucket", .data = sign_index_key },
        .{ .name = "metric", .data = sign_index_value },
    });
    defer sign_index_table.deinit();

    var first_zero_indices = try sign_index_table.groupByFirstZeroIndex("bucket", "metric", "first_zero_index");
    defer first_zero_indices.deinit();
    try expectNullableI64Column(first_zero_indices, gpa, "first_zero_index", &.{ 0, 5, 0 }, &.{ true, true, false });

    var last_zero_indices = try sign_index_table.groupByLastZeroIndex("bucket", "metric", "last_zero_index");
    defer last_zero_indices.deinit();
    try expectNullableI64Column(last_zero_indices, gpa, "last_zero_index", &.{ 0, 6, 0 }, &.{ true, true, false });

    var first_non_zero_indices = try sign_index_table.groupByFirstNonZeroIndex("bucket", "metric", "first_non_zero_index");
    defer first_non_zero_indices.deinit();
    try expectNullableI64Column(first_non_zero_indices, gpa, "first_non_zero_index", &.{ 1, 4, 0 }, &.{ true, true, false });

    var last_non_zero_indices = try sign_index_table.groupByLastNonZeroIndex("bucket", "metric", "last_non_zero_index");
    defer last_non_zero_indices.deinit();
    try expectNullableI64Column(last_non_zero_indices, gpa, "last_non_zero_index", &.{ 3, 4, 0 }, &.{ true, true, false });

    var first_positive_indices = try sign_index_table.groupByFirstPositiveIndex("bucket", "metric", "first_positive_index");
    defer first_positive_indices.deinit();
    try expectNullableI64Column(first_positive_indices, gpa, "first_positive_index", &.{ 2, 4, 0 }, &.{ true, true, false });

    var last_positive_indices = try sign_index_table.groupByLastPositiveIndex("bucket", "metric", "last_positive_index");
    defer last_positive_indices.deinit();
    try expectNullableI64Column(last_positive_indices, gpa, "last_positive_index", &.{ 2, 4, 0 }, &.{ true, true, false });

    var first_signbit_indices = try sign_index_table.groupByFirstSignBitIndex("bucket", "metric", "first_signbit_index");
    defer first_signbit_indices.deinit();
    try expectNullableI64Column(first_signbit_indices, gpa, "first_signbit_index", &.{ 1, 6, 0 }, &.{ true, true, false });

    var last_signbit_indices = try sign_index_table.groupByLastSignBitIndex("bucket", "metric", "last_signbit_index");
    defer last_signbit_indices.deinit();
    try expectNullableI64Column(last_signbit_indices, gpa, "last_signbit_index", &.{ 3, 6, 0 }, &.{ true, true, false });

    var first_negative_indices = try sign_index_table.groupByFirstNegativeIndex("bucket", "metric", "first_negative_index");
    defer first_negative_indices.deinit();
    try expectNullableI64Column(first_negative_indices, gpa, "first_negative_index", &.{ 1, 0, 0 }, &.{ true, false, false });

    var last_negative_indices = try sign_index_table.groupByLastNegativeIndex("bucket", "metric", "last_negative_index");
    defer last_negative_indices.deinit();
    try expectNullableI64Column(last_negative_indices, gpa, "last_negative_index", &.{ 3, 0, 0 }, &.{ true, false, false });

    // Cumulative quality-index transforms preserve the prefix state per group while
    // keeping null value rows nullable, matching the grouped cumulative count/ratio semantics.
    var metric_cum_first_nan_index = try quality_index_table.withGroupCumulativeFirstNaNIndex("bucket", "metric", "metric_cum_first_nan_index");
    defer metric_cum_first_nan_index.deinit();
    try expectNullableI64Column(metric_cum_first_nan_index, gpa, "metric_cum_first_nan_index", &.{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, &.{ true, true, true, true, true, true, true, false, false, false, false });

    var metric_cum_last_nan_index = try quality_index_table.withGroupCumulativeLastNaNIndex("bucket", "metric", "metric_cum_last_nan_index");
    defer metric_cum_last_nan_index.deinit();
    try expectNullableI64Column(metric_cum_last_nan_index, gpa, "metric_cum_last_nan_index", &.{ 0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 0 }, &.{ true, true, true, true, true, true, true, false, false, false, false });

    var metric_cum_first_inf_index = try quality_index_table.withGroupCumulativeFirstInfIndex("bucket", "metric", "metric_cum_first_inf_index");
    defer metric_cum_first_inf_index.deinit();
    try expectNullableI64Column(metric_cum_first_inf_index, gpa, "metric_cum_first_inf_index", &.{ 0, 1, 1, 1, 1, 1, 1, 0, 8, 8, 0 }, &.{ false, true, true, true, true, true, true, false, true, true, false });

    var metric_cum_last_inf_index = try quality_index_table.withGroupCumulativeLastInfIndex("bucket", "metric", "metric_cum_last_inf_index");
    defer metric_cum_last_inf_index.deinit();
    try expectNullableI64Column(metric_cum_last_inf_index, gpa, "metric_cum_last_inf_index", &.{ 0, 1, 2, 2, 4, 4, 4, 0, 8, 9, 0 }, &.{ false, true, true, true, true, true, true, false, true, true, false });

    var metric_cum_first_positive_inf_index = try quality_index_table.withGroupCumulativeFirstPositiveInfIndex("bucket", "metric", "metric_cum_first_positive_inf_index");
    defer metric_cum_first_positive_inf_index.deinit();
    try expectNullableI64Column(metric_cum_first_positive_inf_index, gpa, "metric_cum_first_positive_inf_index", &.{ 0, 1, 1, 1, 1, 1, 1, 0, 0, 9, 0 }, &.{ false, true, true, true, true, true, true, false, false, true, false });

    var metric_cum_last_positive_inf_index = try quality_index_table.withGroupCumulativeLastPositiveInfIndex("bucket", "metric", "metric_cum_last_positive_inf_index");
    defer metric_cum_last_positive_inf_index.deinit();
    try expectNullableI64Column(metric_cum_last_positive_inf_index, gpa, "metric_cum_last_positive_inf_index", &.{ 0, 1, 1, 1, 4, 4, 4, 0, 0, 9, 0 }, &.{ false, true, true, true, true, true, true, false, false, true, false });

    var metric_cum_first_negative_inf_index = try quality_index_table.withGroupCumulativeFirstNegativeInfIndex("bucket", "metric", "metric_cum_first_negative_inf_index");
    defer metric_cum_first_negative_inf_index.deinit();
    try expectNullableI64Column(metric_cum_first_negative_inf_index, gpa, "metric_cum_first_negative_inf_index", &.{ 0, 0, 2, 2, 2, 2, 2, 0, 8, 8, 0 }, &.{ false, false, true, true, true, true, true, false, true, true, false });

    var metric_cum_last_negative_inf_index = try quality_index_table.withGroupCumulativeLastNegativeInfIndex("bucket", "metric", "metric_cum_last_negative_inf_index");
    defer metric_cum_last_negative_inf_index.deinit();
    try expectNullableI64Column(metric_cum_last_negative_inf_index, gpa, "metric_cum_last_negative_inf_index", &.{ 0, 0, 2, 2, 2, 2, 2, 0, 8, 8, 0 }, &.{ false, false, true, true, true, true, true, false, true, true, false });

    var metric_cum_first_finite_index = try quality_index_table.withGroupCumulativeFirstFiniteIndex("bucket", "metric", "metric_cum_first_finite_index");
    defer metric_cum_first_finite_index.deinit();
    try expectNullableI64Column(metric_cum_first_finite_index, gpa, "metric_cum_first_finite_index", &.{ 0, 0, 0, 0, 0, 5, 5, 7, 7, 7, 0 }, &.{ false, false, false, false, false, true, true, true, true, true, false });

    var metric_cum_last_finite_index = try quality_index_table.withGroupCumulativeLastFiniteIndex("bucket", "metric", "metric_cum_last_finite_index");
    defer metric_cum_last_finite_index.deinit();
    try expectNullableI64Column(metric_cum_last_finite_index, gpa, "metric_cum_last_finite_index", &.{ 0, 0, 0, 0, 0, 5, 6, 7, 7, 7, 0 }, &.{ false, false, false, false, false, true, true, true, true, true, false });

    var metric_cum_first_normal_index = try quality_index_table.withGroupCumulativeFirstNormalIndex("bucket", "metric", "metric_cum_first_normal_index");
    defer metric_cum_first_normal_index.deinit();
    try expectNullableI64Column(metric_cum_first_normal_index, gpa, "metric_cum_first_normal_index", &.{ 0, 0, 0, 0, 0, 5, 5, 7, 7, 7, 0 }, &.{ false, false, false, false, false, true, true, true, true, true, false });

    var metric_cum_last_normal_index = try quality_index_table.withGroupCumulativeLastNormalIndex("bucket", "metric", "metric_cum_last_normal_index");
    defer metric_cum_last_normal_index.deinit();
    try expectNullableI64Column(metric_cum_last_normal_index, gpa, "metric_cum_last_normal_index", &.{ 0, 0, 0, 0, 0, 5, 5, 7, 7, 7, 0 }, &.{ false, false, false, false, false, true, true, true, true, true, false });

    var metric_cum_first_subnormal_index = try quality_index_table.withGroupCumulativeFirstSubnormalIndex("bucket", "metric", "metric_cum_first_subnormal_index");
    defer metric_cum_first_subnormal_index.deinit();
    try expectNullableI64Column(metric_cum_first_subnormal_index, gpa, "metric_cum_first_subnormal_index", &.{ 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0 }, &.{ false, false, false, false, false, false, true, false, false, false, false });

    var metric_cum_last_subnormal_index = try quality_index_table.withGroupCumulativeLastSubnormalIndex("bucket", "metric", "metric_cum_last_subnormal_index");
    defer metric_cum_last_subnormal_index.deinit();
    try expectNullableI64Column(metric_cum_last_subnormal_index, gpa, "metric_cum_last_subnormal_index", &.{ 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0 }, &.{ false, false, false, false, false, false, true, false, false, false, false });

    var metric_cum_first_non_finite_index = try quality_index_table.withGroupCumulativeFirstNonFiniteIndex("bucket", "metric", "metric_cum_first_non_finite_index");
    defer metric_cum_first_non_finite_index.deinit();
    try expectNullableI64Column(metric_cum_first_non_finite_index, gpa, "metric_cum_first_non_finite_index", &.{ 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 0 }, &.{ true, true, true, true, true, true, true, false, true, true, false });

    var metric_cum_last_non_finite_index = try quality_index_table.withGroupCumulativeLastNonFiniteIndex("bucket", "metric", "metric_cum_last_non_finite_index");
    defer metric_cum_last_non_finite_index.deinit();
    try expectNullableI64Column(metric_cum_last_non_finite_index, gpa, "metric_cum_last_non_finite_index", &.{ 0, 1, 2, 3, 4, 4, 4, 0, 8, 9, 0 }, &.{ true, true, true, true, true, true, true, false, true, true, false });

    var metric_cum_first_non_zero_index = try quality_index_table.withGroupCumulativeFirstNonZeroIndex("bucket", "metric", "metric_cum_first_non_zero_index");
    defer metric_cum_first_non_zero_index.deinit();
    try expectNullableI64Column(metric_cum_first_non_zero_index, gpa, "metric_cum_first_non_zero_index", &.{ 0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 0 }, &.{ true, true, true, true, true, true, true, true, true, true, false });

    var metric_cum_last_non_zero_index = try quality_index_table.withGroupCumulativeLastNonZeroIndex("bucket", "metric", "metric_cum_last_non_zero_index");
    defer metric_cum_last_non_zero_index.deinit();
    try expectNullableI64Column(metric_cum_last_non_zero_index, gpa, "metric_cum_last_non_zero_index", &.{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0 }, &.{ true, true, true, true, true, true, true, true, true, true, false });

    var metric_cum_first_positive_index = try quality_index_table.withGroupCumulativeFirstPositiveIndex("bucket", "metric", "metric_cum_first_positive_index");
    defer metric_cum_first_positive_index.deinit();
    try expectNullableI64Column(metric_cum_first_positive_index, gpa, "metric_cum_first_positive_index", &.{ 0, 1, 1, 1, 1, 1, 1, 7, 7, 7, 0 }, &.{ false, true, true, true, true, true, true, true, true, true, false });

    var metric_cum_last_positive_index = try quality_index_table.withGroupCumulativeLastPositiveIndex("bucket", "metric", "metric_cum_last_positive_index");
    defer metric_cum_last_positive_index.deinit();
    try expectNullableI64Column(metric_cum_last_positive_index, gpa, "metric_cum_last_positive_index", &.{ 0, 1, 1, 1, 4, 5, 6, 7, 7, 9, 0 }, &.{ false, true, true, true, true, true, true, true, true, true, false });

    var metric_cum_first_signbit_index = try quality_index_table.withGroupCumulativeFirstSignBitIndex("bucket", "metric", "metric_cum_first_signbit_index");
    defer metric_cum_first_signbit_index.deinit();
    try expectNullableI64Column(metric_cum_first_signbit_index, gpa, "metric_cum_first_signbit_index", &.{ 0, 0, 2, 2, 2, 2, 2, 0, 8, 8, 0 }, &.{ false, false, true, true, true, true, true, false, true, true, false });

    var metric_cum_last_signbit_index = try quality_index_table.withGroupCumulativeLastSignBitIndex("bucket", "metric", "metric_cum_last_signbit_index");
    defer metric_cum_last_signbit_index.deinit();
    try expectNullableI64Column(metric_cum_last_signbit_index, gpa, "metric_cum_last_signbit_index", &.{ 0, 0, 2, 2, 2, 2, 2, 0, 8, 8, 0 }, &.{ false, false, true, true, true, true, true, false, true, true, false });

    var metric_cum_first_negative_index = try quality_index_table.withGroupCumulativeFirstNegativeIndex("bucket", "metric", "metric_cum_first_negative_index");
    defer metric_cum_first_negative_index.deinit();
    try expectNullableI64Column(metric_cum_first_negative_index, gpa, "metric_cum_first_negative_index", &.{ 0, 0, 2, 2, 2, 2, 2, 0, 8, 8, 0 }, &.{ false, false, true, true, true, true, true, false, true, true, false });

    var metric_cum_last_negative_index = try quality_index_table.withGroupCumulativeLastNegativeIndex("bucket", "metric", "metric_cum_last_negative_index");
    defer metric_cum_last_negative_index.deinit();
    try expectNullableI64Column(metric_cum_last_negative_index, gpa, "metric_cum_last_negative_index", &.{ 0, 0, 2, 2, 2, 2, 2, 0, 8, 8, 0 }, &.{ false, false, true, true, true, true, true, false, true, true, false });

    var metric_cum_first_zero_index = try signed_zero_table.withGroupCumulativeFirstZeroIndex("bucket", "metric", "metric_cum_first_zero_index");
    defer metric_cum_first_zero_index.deinit();
    try expectNullableI64Column(metric_cum_first_zero_index, gpa, "metric_cum_first_zero_index", &.{ 0, 0, 0, 0, 4, 4, 4, 0 }, &.{ true, true, true, true, true, true, true, false });

    var metric_cum_last_zero_index = try signed_zero_table.withGroupCumulativeLastZeroIndex("bucket", "metric", "metric_cum_last_zero_index");
    defer metric_cum_last_zero_index.deinit();
    try expectNullableI64Column(metric_cum_last_zero_index, gpa, "metric_cum_last_zero_index", &.{ 0, 1, 1, 3, 4, 5, 6, 0 }, &.{ true, true, true, true, true, true, true, false });

    var metric_cum_first_positive_zero_index = try signed_zero_table.withGroupCumulativeFirstPositiveZeroIndex("bucket", "metric", "metric_cum_first_positive_zero_index");
    defer metric_cum_first_positive_zero_index.deinit();
    try expectNullableI64Column(metric_cum_first_positive_zero_index, gpa, "metric_cum_first_positive_zero_index", &.{ 0, 0, 0, 0, 0, 5, 5, 0 }, &.{ true, true, true, true, false, true, true, false });

    var metric_cum_last_positive_zero_index = try signed_zero_table.withGroupCumulativeLastPositiveZeroIndex("bucket", "metric", "metric_cum_last_positive_zero_index");
    defer metric_cum_last_positive_zero_index.deinit();
    try expectNullableI64Column(metric_cum_last_positive_zero_index, gpa, "metric_cum_last_positive_zero_index", &.{ 0, 0, 0, 3, 0, 5, 5, 0 }, &.{ true, true, true, true, false, true, true, false });

    var metric_cum_first_negative_zero_index = try signed_zero_table.withGroupCumulativeFirstNegativeZeroIndex("bucket", "metric", "metric_cum_first_negative_zero_index");
    defer metric_cum_first_negative_zero_index.deinit();
    try expectNullableI64Column(metric_cum_first_negative_zero_index, gpa, "metric_cum_first_negative_zero_index", &.{ 0, 1, 1, 1, 4, 4, 4, 0 }, &.{ false, true, true, true, true, true, true, false });

    var metric_cum_last_negative_zero_index = try signed_zero_table.withGroupCumulativeLastNegativeZeroIndex("bucket", "metric", "metric_cum_last_negative_zero_index");
    defer metric_cum_last_negative_zero_index.deinit();
    try expectNullableI64Column(metric_cum_last_negative_zero_index, gpa, "metric_cum_last_negative_zero_index", &.{ 0, 1, 1, 1, 4, 4, 6, 0 }, &.{ false, true, true, true, true, true, true, false });

    var metric_nan_counts_on = try quality_table.groupByNaNCountOn(&.{ "bucket", "day" }, "metric", "metric_nan_count_on");
    defer metric_nan_counts_on.deinit();
    const metric_nan_count_on_values = try (try metric_nan_counts_on.column("metric_nan_count_on")).i64.toOwnedSlice(gpa);
    defer gpa.free(metric_nan_count_on_values);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 0, 0, 0, 0 }, metric_nan_count_on_values);

    const metric_finite_ratio_on_expected = [_]f64{ 0.5, 0.0, 1.0, ratio_nan, 0.0, ratio_nan };
    var metric_finite_ratios_on = try quality_table.groupByFiniteRatioOn(&.{ "bucket", "day" }, "metric", "metric_finite_ratio_on");
    defer metric_finite_ratios_on.deinit();
    try expectF64ColumnApproxOrNan(metric_finite_ratios_on, gpa, "metric_finite_ratio_on", &metric_finite_ratio_on_expected);

    var nan_count_plan = try DeviceLazyFrame.init(gpa, quality_table);
    defer nan_count_plan.deinit();
    try nan_count_plan.groupByNaNCountOn(&.{"bucket"}, "metric", "metric_nan_count_lazy");
    const nan_count_explained = try nan_count_plan.explain(gpa);
    defer gpa.free(nan_count_explained);
    try std.testing.expect(std.mem.indexOf(u8, nan_count_explained, "group_by_nan_count_on([bucket], value=metric -> metric_nan_count_lazy)") != null);
    var lazy_nan_count = try nan_count_plan.collect();
    defer lazy_nan_count.deinit();
    const lazy_nan_count_values = try (try lazy_nan_count.column("metric_nan_count_lazy")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_nan_count_values);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 0, 0 }, lazy_nan_count_values);

    var cumulative_quality_plan = try DeviceLazyFrame.init(gpa, quality_table);
    defer cumulative_quality_plan.deinit();
    try cumulative_quality_plan.withGroupCumulativeNaNCount("bucket", "metric", "metric_cum_nan_count_lazy");
    try cumulative_quality_plan.withGroupCumulativeNaNRatio("bucket", "metric", "metric_cum_nan_ratio_lazy");
    try cumulative_quality_plan.withGroupCumulativeInfCount("bucket", "metric", "metric_cum_inf_count_lazy");
    try cumulative_quality_plan.withGroupCumulativeInfRatio("bucket", "metric", "metric_cum_inf_ratio_lazy");
    try cumulative_quality_plan.withGroupCumulativePositiveInfCount("bucket", "metric", "metric_cum_positive_inf_count_lazy");
    try cumulative_quality_plan.withGroupCumulativePositiveInfRatio("bucket", "metric", "metric_cum_positive_inf_ratio_lazy");
    try cumulative_quality_plan.withGroupCumulativeNegativeInfCount("bucket", "metric", "metric_cum_negative_inf_count_lazy");
    try cumulative_quality_plan.withGroupCumulativeNegativeInfRatio("bucket", "metric", "metric_cum_negative_inf_ratio_lazy");
    try cumulative_quality_plan.withGroupCumulativeFiniteCount("bucket", "metric", "metric_cum_finite_count_lazy");
    try cumulative_quality_plan.withGroupCumulativeFiniteRatio("bucket", "metric", "metric_cum_finite_ratio_lazy");
    try cumulative_quality_plan.withGroupCumulativeNormalCount("bucket", "metric", "metric_cum_normal_count_lazy");
    try cumulative_quality_plan.withGroupCumulativeNormalRatio("bucket", "metric", "metric_cum_normal_ratio_lazy");
    try cumulative_quality_plan.withGroupCumulativeSubnormalCount("bucket", "metric", "metric_cum_subnormal_count_lazy");
    try cumulative_quality_plan.withGroupCumulativeSubnormalRatio("bucket", "metric", "metric_cum_subnormal_ratio_lazy");
    try cumulative_quality_plan.withGroupCumulativeNonFiniteCount("bucket", "metric", "metric_cum_non_finite_count_lazy");
    try cumulative_quality_plan.withGroupCumulativeNonFiniteRatio("bucket", "metric", "metric_cum_non_finite_ratio_lazy");
    try cumulative_quality_plan.withGroupCumulativeZeroCount("bucket", "metric", "metric_cum_zero_count_lazy");
    try cumulative_quality_plan.withGroupCumulativeZeroRatio("bucket", "metric", "metric_cum_zero_ratio_lazy");
    try cumulative_quality_plan.withGroupCumulativePositiveZeroCount("bucket", "metric", "metric_cum_positive_zero_count_lazy");
    try cumulative_quality_plan.withGroupCumulativePositiveZeroRatio("bucket", "metric", "metric_cum_positive_zero_ratio_lazy");
    try cumulative_quality_plan.withGroupCumulativeNegativeZeroCount("bucket", "metric", "metric_cum_negative_zero_count_lazy");
    try cumulative_quality_plan.withGroupCumulativeNegativeZeroRatio("bucket", "metric", "metric_cum_negative_zero_ratio_lazy");
    try cumulative_quality_plan.withGroupCumulativeNonZeroCount("bucket", "metric", "metric_cum_non_zero_count_lazy");
    try cumulative_quality_plan.withGroupCumulativeNonZeroRatio("bucket", "metric", "metric_cum_non_zero_ratio_lazy");
    try cumulative_quality_plan.withGroupCumulativePositiveCount("bucket", "metric", "metric_cum_positive_count_lazy");
    try cumulative_quality_plan.withGroupCumulativePositiveRatio("bucket", "metric", "metric_cum_positive_ratio_lazy");
    try cumulative_quality_plan.withGroupCumulativeSignBitCount("bucket", "metric", "metric_cum_signbit_count_lazy");
    try cumulative_quality_plan.withGroupCumulativeSignBitRatio("bucket", "metric", "metric_cum_signbit_ratio_lazy");
    try cumulative_quality_plan.withGroupCumulativeNegativeCount("bucket", "metric", "metric_cum_negative_count_lazy");
    try cumulative_quality_plan.withGroupCumulativeNegativeRatio("bucket", "metric", "metric_cum_negative_ratio_lazy");
    const cumulative_quality_explained = try cumulative_quality_plan.explain(gpa);
    defer gpa.free(cumulative_quality_explained);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_quality_explained, "group_cumulative_nan_count([bucket], value=metric->metric_cum_nan_count_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_quality_explained, "group_cumulative_nan_ratio([bucket], value=metric->metric_cum_nan_ratio_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_quality_explained, "group_cumulative_inf_count([bucket], value=metric->metric_cum_inf_count_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_quality_explained, "group_cumulative_inf_ratio([bucket], value=metric->metric_cum_inf_ratio_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_quality_explained, "group_cumulative_positive_inf_count([bucket], value=metric->metric_cum_positive_inf_count_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_quality_explained, "group_cumulative_positive_inf_ratio([bucket], value=metric->metric_cum_positive_inf_ratio_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_quality_explained, "group_cumulative_negative_inf_count([bucket], value=metric->metric_cum_negative_inf_count_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_quality_explained, "group_cumulative_negative_inf_ratio([bucket], value=metric->metric_cum_negative_inf_ratio_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_quality_explained, "group_cumulative_finite_count([bucket], value=metric->metric_cum_finite_count_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_quality_explained, "group_cumulative_finite_ratio([bucket], value=metric->metric_cum_finite_ratio_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_quality_explained, "group_cumulative_normal_count([bucket], value=metric->metric_cum_normal_count_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_quality_explained, "group_cumulative_normal_ratio([bucket], value=metric->metric_cum_normal_ratio_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_quality_explained, "group_cumulative_subnormal_count([bucket], value=metric->metric_cum_subnormal_count_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_quality_explained, "group_cumulative_subnormal_ratio([bucket], value=metric->metric_cum_subnormal_ratio_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_quality_explained, "group_cumulative_non_finite_count([bucket], value=metric->metric_cum_non_finite_count_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_quality_explained, "group_cumulative_non_finite_ratio([bucket], value=metric->metric_cum_non_finite_ratio_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_quality_explained, "group_cumulative_zero_count([bucket], value=metric->metric_cum_zero_count_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_quality_explained, "group_cumulative_zero_ratio([bucket], value=metric->metric_cum_zero_ratio_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_quality_explained, "group_cumulative_positive_zero_count([bucket], value=metric->metric_cum_positive_zero_count_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_quality_explained, "group_cumulative_positive_zero_ratio([bucket], value=metric->metric_cum_positive_zero_ratio_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_quality_explained, "group_cumulative_negative_zero_count([bucket], value=metric->metric_cum_negative_zero_count_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_quality_explained, "group_cumulative_negative_zero_ratio([bucket], value=metric->metric_cum_negative_zero_ratio_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_quality_explained, "group_cumulative_non_zero_count([bucket], value=metric->metric_cum_non_zero_count_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_quality_explained, "group_cumulative_non_zero_ratio([bucket], value=metric->metric_cum_non_zero_ratio_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_quality_explained, "group_cumulative_positive_count([bucket], value=metric->metric_cum_positive_count_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_quality_explained, "group_cumulative_positive_ratio([bucket], value=metric->metric_cum_positive_ratio_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_quality_explained, "group_cumulative_signbit_count([bucket], value=metric->metric_cum_signbit_count_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_quality_explained, "group_cumulative_signbit_ratio([bucket], value=metric->metric_cum_signbit_ratio_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_quality_explained, "group_cumulative_negative_count([bucket], value=metric->metric_cum_negative_count_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_quality_explained, "group_cumulative_negative_ratio([bucket], value=metric->metric_cum_negative_ratio_lazy)") != null);
    var lazy_cumulative_quality = try cumulative_quality_plan.collect();
    defer lazy_cumulative_quality.deinit();
    try expectNullableI64Column(lazy_cumulative_quality, gpa, "metric_cum_nan_count_lazy", &metric_cum_nan_count_expected, &quality_cum_validity);
    try expectF64ColumnWithValidity(lazy_cumulative_quality, gpa, "metric_cum_nan_ratio_lazy", &metric_cum_nan_ratio_expected, &quality_cum_validity);
    try expectNullableI64Column(lazy_cumulative_quality, gpa, "metric_cum_inf_count_lazy", &metric_cum_inf_count_expected, &quality_cum_validity);
    try expectF64ColumnWithValidity(lazy_cumulative_quality, gpa, "metric_cum_inf_ratio_lazy", &metric_cum_inf_ratio_expected, &quality_cum_validity);
    try expectNullableI64Column(lazy_cumulative_quality, gpa, "metric_cum_positive_inf_count_lazy", &metric_cum_positive_inf_count_expected, &quality_cum_validity);
    try expectF64ColumnWithValidity(lazy_cumulative_quality, gpa, "metric_cum_positive_inf_ratio_lazy", &metric_cum_positive_inf_ratio_expected, &quality_cum_validity);
    try expectNullableI64Column(lazy_cumulative_quality, gpa, "metric_cum_negative_inf_count_lazy", &metric_cum_negative_inf_count_expected, &quality_cum_validity);
    try expectF64ColumnWithValidity(lazy_cumulative_quality, gpa, "metric_cum_negative_inf_ratio_lazy", &metric_cum_negative_inf_ratio_expected, &quality_cum_validity);
    try expectNullableI64Column(lazy_cumulative_quality, gpa, "metric_cum_finite_count_lazy", &metric_cum_finite_count_expected, &quality_cum_validity);
    try expectF64ColumnWithValidity(lazy_cumulative_quality, gpa, "metric_cum_finite_ratio_lazy", &metric_cum_finite_ratio_expected, &quality_cum_validity);
    try expectNullableI64Column(lazy_cumulative_quality, gpa, "metric_cum_normal_count_lazy", &metric_cum_normal_count_expected, &quality_cum_validity);
    try expectF64ColumnWithValidity(lazy_cumulative_quality, gpa, "metric_cum_normal_ratio_lazy", &metric_cum_normal_ratio_expected, &quality_cum_validity);
    try expectNullableI64Column(lazy_cumulative_quality, gpa, "metric_cum_subnormal_count_lazy", &metric_cum_subnormal_count_expected, &quality_cum_validity);
    try expectF64ColumnWithValidity(lazy_cumulative_quality, gpa, "metric_cum_subnormal_ratio_lazy", &metric_cum_subnormal_ratio_expected, &quality_cum_validity);
    try expectNullableI64Column(lazy_cumulative_quality, gpa, "metric_cum_non_finite_count_lazy", &metric_cum_non_finite_count_expected, &quality_cum_validity);
    try expectF64ColumnWithValidity(lazy_cumulative_quality, gpa, "metric_cum_non_finite_ratio_lazy", &metric_cum_non_finite_ratio_expected, &quality_cum_validity);
    try expectNullableI64Column(lazy_cumulative_quality, gpa, "metric_cum_zero_count_lazy", &metric_cum_zero_count_expected, &quality_cum_validity);
    try expectF64ColumnWithValidity(lazy_cumulative_quality, gpa, "metric_cum_zero_ratio_lazy", &metric_cum_zero_ratio_expected, &quality_cum_validity);
    try expectNullableI64Column(lazy_cumulative_quality, gpa, "metric_cum_positive_zero_count_lazy", &metric_cum_positive_zero_count_expected, &quality_cum_validity);
    try expectF64ColumnWithValidity(lazy_cumulative_quality, gpa, "metric_cum_positive_zero_ratio_lazy", &metric_cum_positive_zero_ratio_expected, &quality_cum_validity);
    try expectNullableI64Column(lazy_cumulative_quality, gpa, "metric_cum_negative_zero_count_lazy", &metric_cum_negative_zero_count_expected, &quality_cum_validity);
    try expectF64ColumnWithValidity(lazy_cumulative_quality, gpa, "metric_cum_negative_zero_ratio_lazy", &metric_cum_negative_zero_ratio_expected, &quality_cum_validity);
    try expectNullableI64Column(lazy_cumulative_quality, gpa, "metric_cum_non_zero_count_lazy", &metric_cum_non_zero_count_expected, &quality_cum_validity);
    try expectF64ColumnWithValidity(lazy_cumulative_quality, gpa, "metric_cum_non_zero_ratio_lazy", &metric_cum_non_zero_ratio_expected, &quality_cum_validity);
    try expectNullableI64Column(lazy_cumulative_quality, gpa, "metric_cum_positive_count_lazy", &metric_cum_positive_count_expected, &quality_cum_validity);
    try expectF64ColumnWithValidity(lazy_cumulative_quality, gpa, "metric_cum_positive_ratio_lazy", &metric_cum_positive_ratio_expected, &quality_cum_validity);
    try expectNullableI64Column(lazy_cumulative_quality, gpa, "metric_cum_signbit_count_lazy", &metric_cum_signbit_count_expected, &quality_cum_validity);
    try expectF64ColumnWithValidity(lazy_cumulative_quality, gpa, "metric_cum_signbit_ratio_lazy", &metric_cum_signbit_ratio_expected, &quality_cum_validity);
    try expectNullableI64Column(lazy_cumulative_quality, gpa, "metric_cum_negative_count_lazy", &metric_cum_negative_count_expected, &quality_cum_validity);
    try expectF64ColumnWithValidity(lazy_cumulative_quality, gpa, "metric_cum_negative_ratio_lazy", &metric_cum_negative_ratio_expected, &quality_cum_validity);

    var cumulative_quality_index_plan = try DeviceLazyFrame.init(gpa, quality_index_table);
    defer cumulative_quality_index_plan.deinit();
    try cumulative_quality_index_plan.withGroupCumulativeFirstNaNIndex("bucket", "metric", "metric_cum_first_nan_index_lazy");
    try cumulative_quality_index_plan.withGroupCumulativeLastInfIndex("bucket", "metric", "metric_cum_last_inf_index_lazy");
    try cumulative_quality_index_plan.withGroupCumulativeFirstPositiveInfIndex("bucket", "metric", "metric_cum_first_positive_inf_index_lazy");
    try cumulative_quality_index_plan.withGroupCumulativeLastFiniteIndex("bucket", "metric", "metric_cum_last_finite_index_lazy");
    try cumulative_quality_index_plan.withGroupCumulativeFirstSubnormalIndex("bucket", "metric", "metric_cum_first_subnormal_index_lazy");
    try cumulative_quality_index_plan.withGroupCumulativeLastNonFiniteIndex("bucket", "metric", "metric_cum_last_non_finite_index_lazy");
    try cumulative_quality_index_plan.withGroupCumulativeFirstNonZeroIndex("bucket", "metric", "metric_cum_first_non_zero_index_lazy");
    try cumulative_quality_index_plan.withGroupCumulativeLastPositiveIndex("bucket", "metric", "metric_cum_last_positive_index_lazy");
    try cumulative_quality_index_plan.withGroupCumulativeFirstSignBitIndex("bucket", "metric", "metric_cum_first_signbit_index_lazy");
    try cumulative_quality_index_plan.withGroupCumulativeLastNegativeIndex("bucket", "metric", "metric_cum_last_negative_index_lazy");
    const cumulative_quality_index_explained = try cumulative_quality_index_plan.explain(gpa);
    defer gpa.free(cumulative_quality_index_explained);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_quality_index_explained, "group_cumulative_first_nan_index([bucket], value=metric->metric_cum_first_nan_index_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_quality_index_explained, "group_cumulative_last_inf_index([bucket], value=metric->metric_cum_last_inf_index_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_quality_index_explained, "group_cumulative_first_positive_inf_index([bucket], value=metric->metric_cum_first_positive_inf_index_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_quality_index_explained, "group_cumulative_last_finite_index([bucket], value=metric->metric_cum_last_finite_index_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_quality_index_explained, "group_cumulative_first_subnormal_index([bucket], value=metric->metric_cum_first_subnormal_index_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_quality_index_explained, "group_cumulative_last_non_finite_index([bucket], value=metric->metric_cum_last_non_finite_index_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_quality_index_explained, "group_cumulative_first_non_zero_index([bucket], value=metric->metric_cum_first_non_zero_index_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_quality_index_explained, "group_cumulative_last_positive_index([bucket], value=metric->metric_cum_last_positive_index_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_quality_index_explained, "group_cumulative_first_signbit_index([bucket], value=metric->metric_cum_first_signbit_index_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_quality_index_explained, "group_cumulative_last_negative_index([bucket], value=metric->metric_cum_last_negative_index_lazy)") != null);
    var lazy_cumulative_quality_index = try cumulative_quality_index_plan.collect();
    defer lazy_cumulative_quality_index.deinit();
    try expectNullableI64Column(lazy_cumulative_quality_index, gpa, "metric_cum_first_nan_index_lazy", &.{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, &.{ true, true, true, true, true, true, true, false, false, false, false });
    try expectNullableI64Column(lazy_cumulative_quality_index, gpa, "metric_cum_last_inf_index_lazy", &.{ 0, 1, 2, 2, 4, 4, 4, 0, 8, 9, 0 }, &.{ false, true, true, true, true, true, true, false, true, true, false });
    try expectNullableI64Column(lazy_cumulative_quality_index, gpa, "metric_cum_first_positive_inf_index_lazy", &.{ 0, 1, 1, 1, 1, 1, 1, 0, 0, 9, 0 }, &.{ false, true, true, true, true, true, true, false, false, true, false });
    try expectNullableI64Column(lazy_cumulative_quality_index, gpa, "metric_cum_last_finite_index_lazy", &.{ 0, 0, 0, 0, 0, 5, 6, 7, 7, 7, 0 }, &.{ false, false, false, false, false, true, true, true, true, true, false });
    try expectNullableI64Column(lazy_cumulative_quality_index, gpa, "metric_cum_first_subnormal_index_lazy", &.{ 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0 }, &.{ false, false, false, false, false, false, true, false, false, false, false });
    try expectNullableI64Column(lazy_cumulative_quality_index, gpa, "metric_cum_last_non_finite_index_lazy", &.{ 0, 1, 2, 3, 4, 4, 4, 0, 8, 9, 0 }, &.{ true, true, true, true, true, true, true, false, true, true, false });
    try expectNullableI64Column(lazy_cumulative_quality_index, gpa, "metric_cum_first_non_zero_index_lazy", &.{ 0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 0 }, &.{ true, true, true, true, true, true, true, true, true, true, false });
    try expectNullableI64Column(lazy_cumulative_quality_index, gpa, "metric_cum_last_positive_index_lazy", &.{ 0, 1, 1, 1, 4, 5, 6, 7, 7, 9, 0 }, &.{ false, true, true, true, true, true, true, true, true, true, false });
    try expectNullableI64Column(lazy_cumulative_quality_index, gpa, "metric_cum_first_signbit_index_lazy", &.{ 0, 0, 2, 2, 2, 2, 2, 0, 8, 8, 0 }, &.{ false, false, true, true, true, true, true, false, true, true, false });
    try expectNullableI64Column(lazy_cumulative_quality_index, gpa, "metric_cum_last_negative_index_lazy", &.{ 0, 0, 2, 2, 2, 2, 2, 0, 8, 8, 0 }, &.{ false, false, true, true, true, true, true, false, true, true, false });

    var cumulative_zero_index_plan = try DeviceLazyFrame.init(gpa, signed_zero_table);
    defer cumulative_zero_index_plan.deinit();
    try cumulative_zero_index_plan.withGroupCumulativeFirstZeroIndex("bucket", "metric", "metric_cum_first_zero_index_lazy");
    try cumulative_zero_index_plan.withGroupCumulativeLastPositiveZeroIndex("bucket", "metric", "metric_cum_last_positive_zero_index_lazy");
    try cumulative_zero_index_plan.withGroupCumulativeFirstNegativeZeroIndex("bucket", "metric", "metric_cum_first_negative_zero_index_lazy");
    const cumulative_zero_index_explained = try cumulative_zero_index_plan.explain(gpa);
    defer gpa.free(cumulative_zero_index_explained);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_zero_index_explained, "group_cumulative_first_zero_index([bucket], value=metric->metric_cum_first_zero_index_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_zero_index_explained, "group_cumulative_last_positive_zero_index([bucket], value=metric->metric_cum_last_positive_zero_index_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_zero_index_explained, "group_cumulative_first_negative_zero_index([bucket], value=metric->metric_cum_first_negative_zero_index_lazy)") != null);
    var lazy_cumulative_zero_index = try cumulative_zero_index_plan.collect();
    defer lazy_cumulative_zero_index.deinit();
    try expectNullableI64Column(lazy_cumulative_zero_index, gpa, "metric_cum_first_zero_index_lazy", &.{ 0, 0, 0, 0, 4, 4, 4, 0 }, &.{ true, true, true, true, true, true, true, false });
    try expectNullableI64Column(lazy_cumulative_zero_index, gpa, "metric_cum_last_positive_zero_index_lazy", &.{ 0, 0, 0, 3, 0, 5, 5, 0 }, &.{ true, true, true, true, false, true, true, false });
    try expectNullableI64Column(lazy_cumulative_zero_index, gpa, "metric_cum_first_negative_zero_index_lazy", &.{ 0, 1, 1, 1, 4, 4, 4, 0 }, &.{ false, true, true, true, true, true, true, false });

    var last_inf_index_plan = try DeviceLazyFrame.init(gpa, quality_index_table);
    defer last_inf_index_plan.deinit();
    try last_inf_index_plan.groupByLastInfIndex("bucket", "metric", "last_inf_index_lazy");
    const last_inf_index_explained = try last_inf_index_plan.explain(gpa);
    defer gpa.free(last_inf_index_explained);
    try std.testing.expect(std.mem.indexOf(u8, last_inf_index_explained, "group_by_last_inf_index(bucket, value=metric -> last_inf_index_lazy)") != null);
    var lazy_last_inf_index = try last_inf_index_plan.collect();
    defer lazy_last_inf_index.deinit();
    try expectNullableI64Column(lazy_last_inf_index, gpa, "last_inf_index_lazy", &.{ 4, 9, 0 }, &.{ true, true, false });

    var first_finite_index_plan = try DeviceLazyFrame.init(gpa, quality_index_table);
    defer first_finite_index_plan.deinit();
    try first_finite_index_plan.groupByFirstFiniteIndex("bucket", "metric", "first_finite_index_lazy");
    const first_finite_index_explained = try first_finite_index_plan.explain(gpa);
    defer gpa.free(first_finite_index_explained);
    try std.testing.expect(std.mem.indexOf(u8, first_finite_index_explained, "group_by_first_finite_index(bucket, value=metric -> first_finite_index_lazy)") != null);
    var lazy_first_finite_index = try first_finite_index_plan.collect();
    defer lazy_first_finite_index.deinit();
    try expectNullableI64Column(lazy_first_finite_index, gpa, "first_finite_index_lazy", &.{ 5, 7, 0 }, &.{ true, true, false });

    var finite_ratio_plan = try DeviceLazyFrame.init(gpa, quality_table);
    defer finite_ratio_plan.deinit();
    try finite_ratio_plan.groupByFiniteRatio("bucket", "metric", "metric_finite_ratio_lazy");
    const finite_ratio_explained = try finite_ratio_plan.explain(gpa);
    defer gpa.free(finite_ratio_explained);
    try std.testing.expect(std.mem.indexOf(u8, finite_ratio_explained, "group_by_finite_ratio(bucket, value=metric -> metric_finite_ratio_lazy)") != null);
    var lazy_finite_ratio = try finite_ratio_plan.collect();
    defer lazy_finite_ratio.deinit();
    try expectF64ColumnApproxOrNan(lazy_finite_ratio, gpa, "metric_finite_ratio_lazy", &metric_finite_ratio_expected);

    var non_finite_ratio_plan = try DeviceLazyFrame.init(gpa, quality_table);
    defer non_finite_ratio_plan.deinit();
    try non_finite_ratio_plan.groupByNonFiniteRatioOn(&.{"bucket"}, "metric", "metric_non_finite_ratio_lazy");
    const non_finite_ratio_explained = try non_finite_ratio_plan.explain(gpa);
    defer gpa.free(non_finite_ratio_explained);
    try std.testing.expect(std.mem.indexOf(u8, non_finite_ratio_explained, "group_by_non_finite_ratio_on([bucket], value=metric -> metric_non_finite_ratio_lazy)") != null);
    var lazy_non_finite_ratio = try non_finite_ratio_plan.collect();
    defer lazy_non_finite_ratio.deinit();
    try expectF64ColumnApproxOrNan(lazy_non_finite_ratio, gpa, "metric_non_finite_ratio_lazy", &metric_non_finite_ratio_expected);

    var zero_count_plan = try DeviceLazyFrame.init(gpa, quality_table);
    defer zero_count_plan.deinit();
    try zero_count_plan.groupByZeroCount("bucket", "metric", "metric_zero_count_lazy");
    const zero_count_explained = try zero_count_plan.explain(gpa);
    defer gpa.free(zero_count_explained);
    try std.testing.expect(std.mem.indexOf(u8, zero_count_explained, "group_by_zero_count(bucket, value=metric -> metric_zero_count_lazy)") != null);
    var lazy_zero_count = try zero_count_plan.collect();
    defer lazy_zero_count.deinit();
    const lazy_zero_count_values = try (try lazy_zero_count.column("metric_zero_count_lazy")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_zero_count_values);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 0 }, lazy_zero_count_values);

    var positive_ratio_plan = try DeviceLazyFrame.init(gpa, quality_table);
    defer positive_ratio_plan.deinit();
    try positive_ratio_plan.groupByPositiveRatioOn(&.{"bucket"}, "metric", "metric_positive_ratio_lazy");
    const positive_ratio_explained = try positive_ratio_plan.explain(gpa);
    defer gpa.free(positive_ratio_explained);
    try std.testing.expect(std.mem.indexOf(u8, positive_ratio_explained, "group_by_positive_ratio_on([bucket], value=metric -> metric_positive_ratio_lazy)") != null);
    var lazy_positive_ratio = try positive_ratio_plan.collect();
    defer lazy_positive_ratio.deinit();
    try expectF64ColumnApproxOrNan(lazy_positive_ratio, gpa, "metric_positive_ratio_lazy", &metric_positive_ratio_expected);

    var positive_zero_ratio_plan = try DeviceLazyFrame.init(gpa, signed_zero_table);
    defer positive_zero_ratio_plan.deinit();
    try positive_zero_ratio_plan.groupByPositiveZeroRatio("bucket", "metric", "positive_zero_ratio_lazy");
    const positive_zero_ratio_explained = try positive_zero_ratio_plan.explain(gpa);
    defer gpa.free(positive_zero_ratio_explained);
    try std.testing.expect(std.mem.indexOf(u8, positive_zero_ratio_explained, "group_by_positive_zero_ratio(bucket, value=metric -> positive_zero_ratio_lazy)") != null);
    var lazy_positive_zero_ratio = try positive_zero_ratio_plan.collect();
    defer lazy_positive_zero_ratio.deinit();
    try expectF64ColumnApproxOrNan(lazy_positive_zero_ratio, gpa, "positive_zero_ratio_lazy", &positive_zero_ratio_expected);

    var last_negative_zero_index_plan = try DeviceLazyFrame.init(gpa, signed_zero_table);
    defer last_negative_zero_index_plan.deinit();
    try last_negative_zero_index_plan.groupByLastNegativeZeroIndex("bucket", "metric", "last_negative_zero_index_lazy");
    const last_negative_zero_index_explained = try last_negative_zero_index_plan.explain(gpa);
    defer gpa.free(last_negative_zero_index_explained);
    try std.testing.expect(std.mem.indexOf(u8, last_negative_zero_index_explained, "group_by_last_negative_zero_index(bucket, value=metric -> last_negative_zero_index_lazy)") != null);
    var lazy_last_negative_zero_index = try last_negative_zero_index_plan.collect();
    defer lazy_last_negative_zero_index.deinit();
    const lazy_last_negative_zero_index_values = try (try lazy_last_negative_zero_index.column("last_negative_zero_index_lazy")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_last_negative_zero_index_values);
    const lazy_last_negative_zero_index_validity = try (try lazy_last_negative_zero_index.column("last_negative_zero_index_lazy")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_last_negative_zero_index_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 6, 0 }, lazy_last_negative_zero_index_values);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false }, lazy_last_negative_zero_index_validity);

    var first_signbit_index_plan = try DeviceLazyFrame.init(gpa, sign_index_table);
    defer first_signbit_index_plan.deinit();
    try first_signbit_index_plan.groupByFirstSignBitIndex("bucket", "metric", "first_signbit_index_lazy");
    const first_signbit_index_explained = try first_signbit_index_plan.explain(gpa);
    defer gpa.free(first_signbit_index_explained);
    try std.testing.expect(std.mem.indexOf(u8, first_signbit_index_explained, "group_by_first_signbit_index(bucket, value=metric -> first_signbit_index_lazy)") != null);
    var lazy_first_signbit_index = try first_signbit_index_plan.collect();
    defer lazy_first_signbit_index.deinit();
    try expectNullableI64Column(lazy_first_signbit_index, gpa, "first_signbit_index_lazy", &.{ 1, 6, 0 }, &.{ true, true, false });

    var any_active_plan = try DeviceLazyFrame.init(gpa, bool_table);
    defer any_active_plan.deinit();
    try any_active_plan.groupByAnyOn(&.{ "store", "day" }, "active", "any_active_lazy");
    const any_active_explained = try any_active_plan.explain(gpa);
    defer gpa.free(any_active_explained);
    try std.testing.expect(std.mem.indexOf(u8, any_active_explained, "group_by_any_on([store,day], value=active -> any_active_lazy)") != null);
    var lazy_any_active = try any_active_plan.collect();
    defer lazy_any_active.deinit();
    const lazy_any_active_values = try (try lazy_any_active.column("any_active_lazy")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_any_active_values);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, lazy_any_active_values);

    var cumulative_bool_plan = try DeviceLazyFrame.init(gpa, bool_table);
    defer cumulative_bool_plan.deinit();
    try cumulative_bool_plan.withGroupCumulativeAny("store", "active", "active_cum_any_lazy");
    try cumulative_bool_plan.withGroupCumulativeAll("store", "active", "active_cum_all_lazy");
    try cumulative_bool_plan.withGroupCumulativeTrueCount("store", "active", "active_cum_true_count_lazy");
    try cumulative_bool_plan.withGroupCumulativeFalseCount("store", "active", "active_cum_false_count_lazy");
    try cumulative_bool_plan.withGroupCumulativeTrueRatio("store", "active", "active_cum_true_ratio_lazy");
    try cumulative_bool_plan.withGroupCumulativeFalseRatio("store", "active", "active_cum_false_ratio_lazy");
    try cumulative_bool_plan.withGroupCumulativeFirstTrueIndex("store", "active", "active_cum_first_true_lazy");
    try cumulative_bool_plan.withGroupCumulativeLastTrueIndex("store", "active", "active_cum_last_true_lazy");
    try cumulative_bool_plan.withGroupCumulativeFirstFalseIndex("store", "active", "active_cum_first_false_lazy");
    try cumulative_bool_plan.withGroupCumulativeLastFalseIndex("store", "active", "active_cum_last_false_lazy");
    const cumulative_bool_explained = try cumulative_bool_plan.explain(gpa);
    defer gpa.free(cumulative_bool_explained);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_bool_explained, "group_cumulative_any([store], value=active->active_cum_any_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_bool_explained, "group_cumulative_all([store], value=active->active_cum_all_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_bool_explained, "group_cumulative_true_count([store], value=active->active_cum_true_count_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_bool_explained, "group_cumulative_false_count([store], value=active->active_cum_false_count_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_bool_explained, "group_cumulative_true_ratio([store], value=active->active_cum_true_ratio_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_bool_explained, "group_cumulative_false_ratio([store], value=active->active_cum_false_ratio_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_bool_explained, "group_cumulative_first_true_index([store], value=active->active_cum_first_true_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_bool_explained, "group_cumulative_last_true_index([store], value=active->active_cum_last_true_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_bool_explained, "group_cumulative_first_false_index([store], value=active->active_cum_first_false_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_bool_explained, "group_cumulative_last_false_index([store], value=active->active_cum_last_false_lazy)") != null);
    var lazy_cumulative_bool = try cumulative_bool_plan.collect();
    defer lazy_cumulative_bool.deinit();
    try expectNullableBoolColumn(lazy_cumulative_bool, gpa, "active_cum_any_lazy", &.{ false, true, false, true, false }, &.{ true, true, true, true, false });
    try expectNullableBoolColumn(lazy_cumulative_bool, gpa, "active_cum_all_lazy", &.{ false, false, false, false, false }, &.{ true, true, true, true, false });
    try expectNullableI64Column(lazy_cumulative_bool, gpa, "active_cum_true_count_lazy", &.{ 0, 1, 0, 1, 0 }, &.{ true, true, true, true, false });
    try expectNullableI64Column(lazy_cumulative_bool, gpa, "active_cum_false_count_lazy", &.{ 1, 1, 1, 1, 0 }, &.{ true, true, true, true, false });
    try expectF64ColumnWithValidity(lazy_cumulative_bool, gpa, "active_cum_true_ratio_lazy", &.{ 0.0, 0.5, 0.0, 0.5, 0.0 }, &.{ true, true, true, true, false });
    try expectF64ColumnWithValidity(lazy_cumulative_bool, gpa, "active_cum_false_ratio_lazy", &.{ 1.0, 0.5, 1.0, 0.5, 0.0 }, &.{ true, true, true, true, false });
    try expectNullableI64Column(lazy_cumulative_bool, gpa, "active_cum_first_true_lazy", &.{ 0, 1, 0, 3, 0 }, &.{ false, true, false, true, false });
    try expectNullableI64Column(lazy_cumulative_bool, gpa, "active_cum_last_true_lazy", &.{ 0, 1, 0, 3, 0 }, &.{ false, true, false, true, false });
    try expectNullableI64Column(lazy_cumulative_bool, gpa, "active_cum_first_false_lazy", &.{ 0, 0, 2, 2, 0 }, &.{ true, true, true, true, false });
    try expectNullableI64Column(lazy_cumulative_bool, gpa, "active_cum_last_false_lazy", &.{ 0, 0, 2, 2, 0 }, &.{ true, true, true, true, false });

    var null_count_plan = try DeviceLazyFrame.init(gpa, bool_table);
    defer null_count_plan.deinit();
    try null_count_plan.groupByNullCountOn(&.{ "store", "day" }, "active", "active_null_count_lazy");
    const null_count_explained = try null_count_plan.explain(gpa);
    defer gpa.free(null_count_explained);
    try std.testing.expect(std.mem.indexOf(u8, null_count_explained, "group_by_null_count_on([store,day], value=active -> active_null_count_lazy)") != null);
    var lazy_null_counts = try null_count_plan.collect();
    defer lazy_null_counts.deinit();
    const lazy_null_count_values = try (try lazy_null_counts.column("active_null_count_lazy")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_null_count_values);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 1 }, lazy_null_count_values);

    var null_ratio_plan = try DeviceLazyFrame.init(gpa, bool_table);
    defer null_ratio_plan.deinit();
    try null_ratio_plan.groupByNullRatioOn(&.{ "store", "day" }, "active", "active_null_ratio_lazy");
    const null_ratio_explained = try null_ratio_plan.explain(gpa);
    defer gpa.free(null_ratio_explained);
    try std.testing.expect(std.mem.indexOf(u8, null_ratio_explained, "group_by_null_ratio_on([store,day], value=active -> active_null_ratio_lazy)") != null);
    var lazy_null_ratios = try null_ratio_plan.collect();
    defer lazy_null_ratios.deinit();
    const lazy_null_ratio_values = try (try lazy_null_ratios.column("active_null_ratio_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_null_ratio_values);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_null_ratio_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_null_ratio_values[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_null_ratio_values[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_null_ratio_values[3], 1e-12);

    var any_null_plan = try DeviceLazyFrame.init(gpa, bool_table);
    defer any_null_plan.deinit();
    try any_null_plan.groupByAnyNullOn(&.{ "store", "day" }, "active", "active_any_null_lazy");
    const any_null_explained = try any_null_plan.explain(gpa);
    defer gpa.free(any_null_explained);
    try std.testing.expect(std.mem.indexOf(u8, any_null_explained, "group_by_any_null_on([store,day], value=active -> active_any_null_lazy)") != null);
    var lazy_any_null = try any_null_plan.collect();
    defer lazy_any_null.deinit();
    const lazy_any_null_values = try (try lazy_any_null.column("active_any_null_lazy")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_any_null_values);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, true }, lazy_any_null_values);

    var last_valid_index_plan = try DeviceLazyFrame.init(gpa, bool_table);
    defer last_valid_index_plan.deinit();
    try last_valid_index_plan.groupByLastValidIndex("store", "active", "active_last_valid_index_lazy");
    const last_valid_index_explained = try last_valid_index_plan.explain(gpa);
    defer gpa.free(last_valid_index_explained);
    try std.testing.expect(std.mem.indexOf(u8, last_valid_index_explained, "group_by_last_valid_index(store, value=active -> active_last_valid_index_lazy)") != null);
    var lazy_last_valid_index = try last_valid_index_plan.collect();
    defer lazy_last_valid_index.deinit();
    try expectNullableI64Column(lazy_last_valid_index, gpa, "active_last_valid_index_lazy", &.{ 1, 3, 0 }, &.{ true, true, false });

    var true_ratio_plan = try DeviceLazyFrame.init(gpa, bool_table);
    defer true_ratio_plan.deinit();
    try true_ratio_plan.groupByTrueRatioOn(&.{ "store", "day" }, "active", "active_true_ratio_lazy");
    const true_ratio_explained = try true_ratio_plan.explain(gpa);
    defer gpa.free(true_ratio_explained);
    try std.testing.expect(std.mem.indexOf(u8, true_ratio_explained, "group_by_true_ratio_on([store,day], value=active -> active_true_ratio_lazy)") != null);
    var lazy_true_ratios = try true_ratio_plan.collect();
    defer lazy_true_ratios.deinit();
    const lazy_true_ratio_values = try (try lazy_true_ratios.column("active_true_ratio_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_true_ratio_values);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_true_ratio_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_true_ratio_values[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_true_ratio_values[2], 1e-12);

    var first_false_index_plan = try DeviceLazyFrame.init(gpa, bool_table);
    defer first_false_index_plan.deinit();
    try first_false_index_plan.groupByFirstFalseIndex("store", "active", "active_first_false_index_lazy");
    const first_false_index_explained = try first_false_index_plan.explain(gpa);
    defer gpa.free(first_false_index_explained);
    try std.testing.expect(std.mem.indexOf(u8, first_false_index_explained, "group_by_first_false_index(store, value=active -> active_first_false_index_lazy)") != null);
    var lazy_first_false_index = try first_false_index_plan.collect();
    defer lazy_first_false_index.deinit();
    try expectNullableI64Column(lazy_first_false_index, gpa, "active_first_false_index_lazy", &.{ 0, 2, 0 }, &.{ true, true, false });

    try std.testing.expectError(error.TypeUnsupported, bool_table.groupByAny("store", "day", "bad_any"));

    var key = try DeviceColumn.fromSliceWithValidity(i32, gpa, &.{ 1, 2, 1, 3, 2, 1 }, &.{ true, true, true, false, true, true }, .cpu);
    defer key.deinit();
    var sales = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 2.0, 3.0, 5.0, 7.0, 11.0, 13.0 }, &.{ true, true, false, true, true, true }, .cpu);
    defer sales.deinit();
    var delta = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ -2.0, -3.0, -5.0, 7.0, -11.0, 13.0 }, &.{ true, true, false, true, true, true }, .cpu);
    defer delta.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "store", .data = key },
        .{ .name = "sales", .data = sales },
        .{ .name = "delta", .data = delta },
    });
    defer table.deinit();

    var counted = try table.groupByCount("store", "rows");
    defer counted.deinit();
    try std.testing.expectEqual(@as(usize, 2), counted.height());
    const count_keys = try (try counted.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(count_keys);
    const counts = try (try counted.column("rows")).i64.toOwnedSlice(gpa);
    defer gpa.free(counts);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2 }, count_keys);
    try std.testing.expectEqualSlices(i64, &.{ 3, 2 }, counts);

    var group_ids = try table.withGroupId("store", "store_group_id");
    defer group_ids.deinit();
    try expectNullableI64Column(group_ids, gpa, "store_group_id", &.{ 0, 1, 0, 0, 1, 0 }, &.{ true, true, true, false, true, true });

    var group_first_row_indices = try table.withGroupFirstRowIndex("store", "store_first_row_index");
    defer group_first_row_indices.deinit();
    try expectNullableI64Column(group_first_row_indices, gpa, "store_first_row_index", &.{ 0, 1, 0, 0, 1, 0 }, &.{ true, true, true, false, true, true });

    var group_last_row_indices = try table.withGroupLastRowIndex("store", "store_last_row_index");
    defer group_last_row_indices.deinit();
    try expectNullableI64Column(group_last_row_indices, gpa, "store_last_row_index", &.{ 5, 4, 5, 0, 4, 5 }, &.{ true, true, true, false, true, true });

    var group_is_first_rows = try table.withGroupIsFirstRow("store", "store_is_first_row");
    defer group_is_first_rows.deinit();
    try expectNullableBoolColumn(group_is_first_rows, gpa, "store_is_first_row", &.{ true, true, false, false, false, false }, &.{ true, true, true, false, true, true });

    var group_is_last_rows = try table.withGroupIsLastRow("store", "store_is_last_row");
    defer group_is_last_rows.deinit();
    try expectNullableBoolColumn(group_is_last_rows, gpa, "store_is_last_row", &.{ false, false, false, false, true, true }, &.{ true, true, true, false, true, true });

    var store_singletons = try table.withGroupIsSingleton("store", "store_is_singleton");
    defer store_singletons.deinit();
    try expectNullableBoolColumn(store_singletons, gpa, "store_is_singleton", &.{ false, false, false, false, false, false }, &.{ true, true, true, false, true, true });

    var store_sales_singletons = try table.withGroupIsSingletonOn(&.{ "store", "sales" }, "store_sales_is_singleton");
    defer store_sales_singletons.deinit();
    try expectNullableBoolColumn(store_sales_singletons, gpa, "store_sales_is_singleton", &.{ true, true, false, false, true, true }, &.{ true, true, false, false, true, true });

    var store_duplicated_groups = try table.withGroupIsDuplicated("store", "store_is_duplicated_group");
    defer store_duplicated_groups.deinit();
    try expectNullableBoolColumn(store_duplicated_groups, gpa, "store_is_duplicated_group", &.{ true, true, true, false, true, true }, &.{ true, true, true, false, true, true });

    var store_sales_duplicated_groups = try table.withGroupIsDuplicatedOn(&.{ "store", "sales" }, "store_sales_is_duplicated_group");
    defer store_sales_duplicated_groups.deinit();
    try expectNullableBoolColumn(store_sales_duplicated_groups, gpa, "store_sales_is_duplicated_group", &.{ false, false, false, false, false, false }, &.{ true, true, false, false, true, true });

    var group_cume_dist = try table.withGroupCumeDist("store", "store_cume_dist");
    defer group_cume_dist.deinit();
    try expectF64ColumnWithValidity(group_cume_dist, gpa, "store_cume_dist", &.{ 1.0 / 3.0, 0.5, 2.0 / 3.0, 0.0, 1.0, 1.0 }, &.{ true, true, true, false, true, true });

    var group_percent_rank = try table.withGroupPercentRank("store", "store_percent_rank");
    defer group_percent_rank.deinit();
    try expectF64ColumnWithValidity(group_percent_rank, gpa, "store_percent_rank", &.{ 0.0, 0.0, 0.5, 0.0, 1.0, 1.0 }, &.{ true, true, true, false, true, true });

    var group_reverse_cume_dist = try table.withGroupReverseCumeDist("store", "store_reverse_cume_dist");
    defer group_reverse_cume_dist.deinit();
    try expectF64ColumnWithValidity(group_reverse_cume_dist, gpa, "store_reverse_cume_dist", &.{ 1.0, 1.0, 2.0 / 3.0, 0.0, 0.5, 1.0 / 3.0 }, &.{ true, true, true, false, true, true });

    var group_reverse_percent_rank = try table.withGroupReversePercentRank("store", "store_reverse_percent_rank");
    defer group_reverse_percent_rank.deinit();
    try expectF64ColumnWithValidity(group_reverse_percent_rank, gpa, "store_reverse_percent_rank", &.{ 1.0, 1.0, 0.5, 0.0, 0.0, 0.0 }, &.{ true, true, true, false, true, true });

    var group_lagged_sales = try table.withGroupLag("store", "sales", "store_sales_lag", 1);
    defer group_lagged_sales.deinit();
    try expectF64ColumnWithValidity(group_lagged_sales, gpa, "store_sales_lag", &.{ 0.0, 0.0, 2.0, 0.0, 3.0, 5.0 }, &.{ false, false, true, false, true, false });

    var group_lead_sales = try table.withGroupLead("store", "sales", "store_sales_lead", 1);
    defer group_lead_sales.deinit();
    try expectF64ColumnWithValidity(group_lead_sales, gpa, "store_sales_lead", &.{ 5.0, 11.0, 13.0, 0.0, 0.0, 0.0 }, &.{ false, true, true, false, false, false });

    var group_lagged_sales_on = try table.withGroupLagOn(&.{ "store", "sales" }, "sales", "store_sales_lag_on", 1);
    defer group_lagged_sales_on.deinit();
    try expectF64ColumnWithValidity(group_lagged_sales_on, gpa, "store_sales_lag_on", &.{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, &.{ false, false, false, false, false, false });

    var group_first_sales = try table.withGroupFirstRowValue("store", "sales", "store_sales_first_row_value");
    defer group_first_sales.deinit();
    try expectF64ColumnWithValidity(group_first_sales, gpa, "store_sales_first_row_value", &.{ 2.0, 3.0, 2.0, 0.0, 3.0, 2.0 }, &.{ true, true, true, false, true, true });

    var group_last_sales = try table.withGroupLastRowValue("store", "sales", "store_sales_last_row_value");
    defer group_last_sales.deinit();
    try expectF64ColumnWithValidity(group_last_sales, gpa, "store_sales_last_row_value", &.{ 13.0, 11.0, 13.0, 0.0, 11.0, 13.0 }, &.{ true, true, true, false, true, true });

    var group_nth_sales = try table.withGroupNthRowValue("store", "sales", "store_sales_nth_row_value", 1);
    defer group_nth_sales.deinit();
    try expectF64ColumnWithValidity(group_nth_sales, gpa, "store_sales_nth_row_value", &.{ 5.0, 11.0, 5.0, 0.0, 11.0, 5.0 }, &.{ false, true, false, false, true, false });

    var group_first_valid_sales = try table.withGroupFirstValidValue("store", "sales", "store_sales_first_valid");
    defer group_first_valid_sales.deinit();
    try expectF64ColumnWithValidity(group_first_valid_sales, gpa, "store_sales_first_valid", &.{ 2.0, 3.0, 2.0, 0.0, 3.0, 2.0 }, &.{ true, true, true, false, true, true });

    var group_last_valid_sales = try table.withGroupLastValidValue("store", "sales", "store_sales_last_valid");
    defer group_last_valid_sales.deinit();
    try expectF64ColumnWithValidity(group_last_valid_sales, gpa, "store_sales_last_valid", &.{ 13.0, 11.0, 13.0, 0.0, 11.0, 13.0 }, &.{ true, true, true, false, true, true });

    var group_nth_valid_sales = try table.withGroupNthValidValue("store", "sales", "store_sales_nth_valid", 1);
    defer group_nth_valid_sales.deinit();
    try expectF64ColumnWithValidity(group_nth_valid_sales, gpa, "store_sales_nth_valid", &.{ 13.0, 11.0, 13.0, 0.0, 11.0, 13.0 }, &.{ true, true, true, false, true, true });

    var group_forward_filled_sales = try table.withGroupFillNullForward("store", "sales", "store_sales_ffill");
    defer group_forward_filled_sales.deinit();
    try expectF64ColumnWithValidity(group_forward_filled_sales, gpa, "store_sales_ffill", &.{ 2.0, 3.0, 2.0, 0.0, 11.0, 13.0 }, &.{ true, true, true, false, true, true });

    var group_backward_filled_sales = try table.withGroupFillNullBackward("store", "sales", "store_sales_bfill");
    defer group_backward_filled_sales.deinit();
    try expectF64ColumnWithValidity(group_backward_filled_sales, gpa, "store_sales_bfill", &.{ 2.0, 3.0, 13.0, 0.0, 11.0, 13.0 }, &.{ true, true, true, false, true, true });

    var group_cum_valid_sales = try table.withGroupCumulativeValidCount("store", "sales", "store_sales_cum_valid");
    defer group_cum_valid_sales.deinit();
    try expectNullableI64Column(group_cum_valid_sales, gpa, "store_sales_cum_valid", &.{ 1, 1, 1, 0, 2, 2 }, &.{ true, true, true, false, true, true });

    var group_cum_null_sales = try table.withGroupCumulativeNullCount("store", "sales", "store_sales_cum_null");
    defer group_cum_null_sales.deinit();
    try expectNullableI64Column(group_cum_null_sales, gpa, "store_sales_cum_null", &.{ 0, 0, 1, 0, 0, 1 }, &.{ true, true, true, false, true, true });

    var group_cum_valid_ratio_sales = try table.withGroupCumulativeValidRatio("store", "sales", "store_sales_cum_valid_ratio");
    defer group_cum_valid_ratio_sales.deinit();
    try expectF64ColumnWithValidity(group_cum_valid_ratio_sales, gpa, "store_sales_cum_valid_ratio", &.{ 1.0, 1.0, 0.5, 0.0, 1.0, 2.0 / 3.0 }, &.{ true, true, true, false, true, true });

    var group_cum_null_ratio_sales = try table.withGroupCumulativeNullRatio("store", "sales", "store_sales_cum_null_ratio");
    defer group_cum_null_ratio_sales.deinit();
    try expectF64ColumnWithValidity(group_cum_null_ratio_sales, gpa, "store_sales_cum_null_ratio", &.{ 0.0, 0.0, 0.5, 0.0, 0.0, 1.0 / 3.0 }, &.{ true, true, true, false, true, true });

    var distinct_key = try DeviceColumn.fromSliceWithValidity(i32, gpa, &.{ 1, 1, 1, 1, 2, 2, 2, 3 }, &.{ true, true, true, true, true, true, true, false }, .cpu);
    defer distinct_key.deinit();
    var distinct_value = try DeviceColumn.fromSliceWithValidity(i32, gpa, &.{ 5, 5, 7, 5, 1, 2, 1, 9 }, &.{ true, true, true, false, true, true, true, true }, .cpu);
    defer distinct_value.deinit();
    var distinct_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "bucket", .data = distinct_key },
        .{ .name = "label", .data = distinct_value },
    });
    defer distinct_table.deinit();

    var group_cum_distinct_label = try distinct_table.withGroupCumulativeDistinctCount("bucket", "label", "label_cum_distinct");
    defer group_cum_distinct_label.deinit();
    try expectNullableI64Column(group_cum_distinct_label, gpa, "label_cum_distinct", &.{ 1, 1, 2, 0, 1, 2, 2, 0 }, &.{ true, true, true, false, true, true, true, false });

    var group_cum_nunique_label = try distinct_table.withGroupCumNUnique("bucket", "label", "label_cum_n_unique");
    defer group_cum_nunique_label.deinit();
    try expectNullableI64Column(group_cum_nunique_label, gpa, "label_cum_n_unique", &.{ 1, 1, 2, 0, 1, 2, 2, 0 }, &.{ true, true, true, false, true, true, true, false });

    var cumulative_distinct_plan = try DeviceLazyFrame.init(gpa, distinct_table);
    defer cumulative_distinct_plan.deinit();
    try cumulative_distinct_plan.withGroupCumulativeDistinctCount("bucket", "label", "label_cum_distinct_lazy");
    try cumulative_distinct_plan.withGroupCumulativeNUnique("bucket", "label", "label_cum_n_unique_lazy");
    const cumulative_distinct_explained = try cumulative_distinct_plan.explain(gpa);
    defer gpa.free(cumulative_distinct_explained);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_distinct_explained, "group_cumulative_distinct_count([bucket], value=label->label_cum_distinct_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_distinct_explained, "group_cumulative_n_unique([bucket], value=label->label_cum_n_unique_lazy)") != null);
    var lazy_cumulative_distinct = try cumulative_distinct_plan.collect();
    defer lazy_cumulative_distinct.deinit();
    try expectNullableI64Column(lazy_cumulative_distinct, gpa, "label_cum_distinct_lazy", &.{ 1, 1, 2, 0, 1, 2, 2, 0 }, &.{ true, true, true, false, true, true, true, false });
    try expectNullableI64Column(lazy_cumulative_distinct, gpa, "label_cum_n_unique_lazy", &.{ 1, 1, 2, 0, 1, 2, 2, 0 }, &.{ true, true, true, false, true, true, true, false });

    const group_cum_mode_values_expected = [_]i32{ 5, 5, 5, 0, 1, 1, 1, 0 };
    const group_cum_mode_validity_expected = [_]bool{ true, true, true, false, true, true, true, false };
    const group_cum_mode_count_expected = [_]i64{ 1, 2, 2, 0, 1, 1, 2, 0 };
    const group_cum_mode_ratio_expected = [_]f64{ 1.0, 1.0, 2.0 / 3.0, 0.0, 1.0, 0.5, 2.0 / 3.0, 0.0 };
    const group_cum_mode_margin_expected = [_]i64{ 1, 2, 1, 0, 1, 0, 1, 0 };
    const group_cum_mode_margin_ratio_expected = [_]f64{ 1.0, 1.0, 1.0 / 3.0, 0.0, 1.0, 0.0, 1.0 / 3.0, 0.0 };

    var group_cum_mode_label = try distinct_table.withGroupCumulativeMode("bucket", "label", "label_cum_mode");
    defer group_cum_mode_label.deinit();
    const group_cum_mode_label_values = try (try group_cum_mode_label.column("label_cum_mode")).i32.toOwnedSlice(gpa);
    defer gpa.free(group_cum_mode_label_values);
    const group_cum_mode_label_validity = try (try group_cum_mode_label.column("label_cum_mode")).i32.validity.?.toOwnedSlice(gpa);
    defer gpa.free(group_cum_mode_label_validity);
    try std.testing.expectEqualSlices(i32, &group_cum_mode_values_expected, group_cum_mode_label_values);
    try std.testing.expectEqualSlices(bool, &group_cum_mode_validity_expected, group_cum_mode_label_validity);

    var group_cum_mode_count_label = try distinct_table.withGroupCumulativeModeCount("bucket", "label", "label_cum_mode_count");
    defer group_cum_mode_count_label.deinit();
    try expectNullableI64Column(group_cum_mode_count_label, gpa, "label_cum_mode_count", &group_cum_mode_count_expected, &group_cum_mode_validity_expected);

    var group_cum_mode_ratio_label = try distinct_table.withGroupCumulativeModeRatio("bucket", "label", "label_cum_mode_ratio");
    defer group_cum_mode_ratio_label.deinit();
    try expectF64ColumnWithValidity(group_cum_mode_ratio_label, gpa, "label_cum_mode_ratio", &group_cum_mode_ratio_expected, &group_cum_mode_validity_expected);

    var group_cum_mode_margin_label = try distinct_table.withGroupCumulativeModeMargin("bucket", "label", "label_cum_mode_margin");
    defer group_cum_mode_margin_label.deinit();
    try expectNullableI64Column(group_cum_mode_margin_label, gpa, "label_cum_mode_margin", &group_cum_mode_margin_expected, &group_cum_mode_validity_expected);

    var group_cum_mode_margin_ratio_label = try distinct_table.withGroupCumulativeModeMarginRatio("bucket", "label", "label_cum_mode_margin_ratio");
    defer group_cum_mode_margin_ratio_label.deinit();
    try expectF64ColumnWithValidity(group_cum_mode_margin_ratio_label, gpa, "label_cum_mode_margin_ratio", &group_cum_mode_margin_ratio_expected, &group_cum_mode_validity_expected);

    var cumulative_mode_plan = try DeviceLazyFrame.init(gpa, distinct_table);
    defer cumulative_mode_plan.deinit();
    try cumulative_mode_plan.withGroupCumulativeMode("bucket", "label", "label_cum_mode_lazy");
    try cumulative_mode_plan.withGroupCumulativeModeCount("bucket", "label", "label_cum_mode_count_lazy");
    try cumulative_mode_plan.withGroupCumulativeModeRatio("bucket", "label", "label_cum_mode_ratio_lazy");
    try cumulative_mode_plan.withGroupCumulativeModeMargin("bucket", "label", "label_cum_mode_margin_lazy");
    try cumulative_mode_plan.withGroupCumulativeModeMarginRatio("bucket", "label", "label_cum_mode_margin_ratio_lazy");
    const cumulative_mode_explained = try cumulative_mode_plan.explain(gpa);
    defer gpa.free(cumulative_mode_explained);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_mode_explained, "group_cumulative_mode([bucket], value=label->label_cum_mode_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_mode_explained, "group_cumulative_mode_count([bucket], value=label->label_cum_mode_count_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_mode_explained, "group_cumulative_mode_ratio([bucket], value=label->label_cum_mode_ratio_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_mode_explained, "group_cumulative_mode_margin([bucket], value=label->label_cum_mode_margin_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_mode_explained, "group_cumulative_mode_margin_ratio([bucket], value=label->label_cum_mode_margin_ratio_lazy)") != null);
    var lazy_cumulative_mode = try cumulative_mode_plan.collect();
    defer lazy_cumulative_mode.deinit();
    const lazy_group_cum_mode_values = try (try lazy_cumulative_mode.column("label_cum_mode_lazy")).i32.toOwnedSlice(gpa);
    defer gpa.free(lazy_group_cum_mode_values);
    const lazy_group_cum_mode_validity = try (try lazy_cumulative_mode.column("label_cum_mode_lazy")).i32.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_group_cum_mode_validity);
    try std.testing.expectEqualSlices(i32, &group_cum_mode_values_expected, lazy_group_cum_mode_values);
    try std.testing.expectEqualSlices(bool, &group_cum_mode_validity_expected, lazy_group_cum_mode_validity);
    try expectNullableI64Column(lazy_cumulative_mode, gpa, "label_cum_mode_count_lazy", &group_cum_mode_count_expected, &group_cum_mode_validity_expected);
    try expectF64ColumnWithValidity(lazy_cumulative_mode, gpa, "label_cum_mode_ratio_lazy", &group_cum_mode_ratio_expected, &group_cum_mode_validity_expected);
    try expectNullableI64Column(lazy_cumulative_mode, gpa, "label_cum_mode_margin_lazy", &group_cum_mode_margin_expected, &group_cum_mode_validity_expected);
    try expectF64ColumnWithValidity(lazy_cumulative_mode, gpa, "label_cum_mode_margin_ratio_lazy", &group_cum_mode_margin_ratio_expected, &group_cum_mode_validity_expected);

    const group_cum_entropy_expected = [_]f64{ 0.0, 0.0, -((2.0 / 3.0) * std.math.log(f64, std.math.e, 2.0 / 3.0) + (1.0 / 3.0) * std.math.log(f64, std.math.e, 1.0 / 3.0)), 0.0, 0.0, std.math.log(f64, std.math.e, 2.0), -((2.0 / 3.0) * std.math.log(f64, std.math.e, 2.0 / 3.0) + (1.0 / 3.0) * std.math.log(f64, std.math.e, 1.0 / 3.0)), 0.0 };
    const group_cum_gini_expected = [_]f64{ 0.0, 0.0, 4.0 / 9.0, 0.0, 0.0, 0.5, 4.0 / 9.0, 0.0 };
    const group_cum_perplexity_expected = [_]f64{ 1.0, 1.0, std.math.exp(group_cum_entropy_expected[2]), 0.0, 1.0, 2.0, std.math.exp(group_cum_entropy_expected[6]), 0.0 };
    const group_cum_inverse_simpson_expected = [_]f64{ 1.0, 1.0, 9.0 / 5.0, 0.0, 1.0, 2.0, 9.0 / 5.0, 0.0 };
    const group_cum_concentration_expected = [_]f64{ 1.0, 1.0, 5.0 / 9.0, 0.0, 1.0, 0.5, 5.0 / 9.0, 0.0 };
    const group_cum_evenness_expected = [_]f64{ 1.0, 1.0, group_cum_entropy_expected[2] / std.math.log(f64, std.math.e, 2.0), 0.0, 1.0, 1.0, group_cum_entropy_expected[6] / std.math.log(f64, std.math.e, 2.0), 0.0 };

    var group_cum_entropy_label = try distinct_table.withGroupCumulativeEntropy("bucket", "label", "label_cum_entropy");
    defer group_cum_entropy_label.deinit();
    try expectF64ColumnApproxOrNanWithValidity(group_cum_entropy_label, gpa, "label_cum_entropy", &group_cum_entropy_expected, &group_cum_mode_validity_expected);

    var group_cum_gini_label = try distinct_table.withGroupCumulativeGiniImpurity("bucket", "label", "label_cum_gini");
    defer group_cum_gini_label.deinit();
    try expectF64ColumnApproxOrNanWithValidity(group_cum_gini_label, gpa, "label_cum_gini", &group_cum_gini_expected, &group_cum_mode_validity_expected);

    var group_cum_perplexity_label = try distinct_table.withGroupCumulativePerplexity("bucket", "label", "label_cum_perplexity");
    defer group_cum_perplexity_label.deinit();
    try expectF64ColumnApproxOrNanWithValidity(group_cum_perplexity_label, gpa, "label_cum_perplexity", &group_cum_perplexity_expected, &group_cum_mode_validity_expected);

    var group_cum_inverse_label = try distinct_table.withGroupCumulativeInverseSimpson("bucket", "label", "label_cum_inverse_simpson");
    defer group_cum_inverse_label.deinit();
    try expectF64ColumnApproxOrNanWithValidity(group_cum_inverse_label, gpa, "label_cum_inverse_simpson", &group_cum_inverse_simpson_expected, &group_cum_mode_validity_expected);

    var group_cum_concentration_label = try distinct_table.withGroupCumulativeConcentration("bucket", "label", "label_cum_concentration");
    defer group_cum_concentration_label.deinit();
    try expectF64ColumnApproxOrNanWithValidity(group_cum_concentration_label, gpa, "label_cum_concentration", &group_cum_concentration_expected, &group_cum_mode_validity_expected);

    var group_cum_evenness_label = try distinct_table.withGroupCumulativeEvenness("bucket", "label", "label_cum_evenness");
    defer group_cum_evenness_label.deinit();
    try expectF64ColumnApproxOrNanWithValidity(group_cum_evenness_label, gpa, "label_cum_evenness", &group_cum_evenness_expected, &group_cum_mode_validity_expected);

    var cumulative_distribution_plan = try DeviceLazyFrame.init(gpa, distinct_table);
    defer cumulative_distribution_plan.deinit();
    try cumulative_distribution_plan.withGroupCumulativeEntropy("bucket", "label", "label_cum_entropy_lazy");
    try cumulative_distribution_plan.withGroupCumulativeGini("bucket", "label", "label_cum_gini_lazy");
    try cumulative_distribution_plan.withGroupCumulativePerplexity("bucket", "label", "label_cum_perplexity_lazy");
    try cumulative_distribution_plan.withGroupCumulativeInverseSimpson("bucket", "label", "label_cum_inverse_simpson_lazy");
    try cumulative_distribution_plan.withGroupCumulativeConcentration("bucket", "label", "label_cum_concentration_lazy");
    try cumulative_distribution_plan.withGroupCumulativeEvenness("bucket", "label", "label_cum_evenness_lazy");
    const cumulative_distribution_explained = try cumulative_distribution_plan.explain(gpa);
    defer gpa.free(cumulative_distribution_explained);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_distribution_explained, "group_cumulative_entropy([bucket], value=label->label_cum_entropy_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_distribution_explained, "group_cumulative_gini_impurity([bucket], value=label->label_cum_gini_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_distribution_explained, "group_cumulative_perplexity([bucket], value=label->label_cum_perplexity_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_distribution_explained, "group_cumulative_inverse_simpson([bucket], value=label->label_cum_inverse_simpson_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_distribution_explained, "group_cumulative_simpson_concentration([bucket], value=label->label_cum_concentration_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_distribution_explained, "group_cumulative_evenness([bucket], value=label->label_cum_evenness_lazy)") != null);
    var lazy_cumulative_distribution = try cumulative_distribution_plan.collect();
    defer lazy_cumulative_distribution.deinit();
    try expectF64ColumnApproxOrNanWithValidity(lazy_cumulative_distribution, gpa, "label_cum_entropy_lazy", &group_cum_entropy_expected, &group_cum_mode_validity_expected);
    try expectF64ColumnApproxOrNanWithValidity(lazy_cumulative_distribution, gpa, "label_cum_gini_lazy", &group_cum_gini_expected, &group_cum_mode_validity_expected);
    try expectF64ColumnApproxOrNanWithValidity(lazy_cumulative_distribution, gpa, "label_cum_perplexity_lazy", &group_cum_perplexity_expected, &group_cum_mode_validity_expected);
    try expectF64ColumnApproxOrNanWithValidity(lazy_cumulative_distribution, gpa, "label_cum_inverse_simpson_lazy", &group_cum_inverse_simpson_expected, &group_cum_mode_validity_expected);
    try expectF64ColumnApproxOrNanWithValidity(lazy_cumulative_distribution, gpa, "label_cum_concentration_lazy", &group_cum_concentration_expected, &group_cum_mode_validity_expected);
    try expectF64ColumnApproxOrNanWithValidity(lazy_cumulative_distribution, gpa, "label_cum_evenness_lazy", &group_cum_evenness_expected, &group_cum_mode_validity_expected);

    const group_cum_mad_expected = [_]f64{ 0.0, 0.0, 8.0 / 9.0, 0.0, 0.0, 0.5, 4.0 / 9.0, 0.0 };
    const group_cum_mad_ratio_expected = [_]f64{ 0.0, 0.0, (8.0 / 9.0) / (17.0 / 3.0), 0.0, 0.0, 1.0 / 3.0, (4.0 / 9.0) / (4.0 / 3.0), 0.0 };
    const group_cum_gini_mean_diff_expected = [_]f64{ 0.0, 0.0, 4.0 / 3.0, 0.0, 0.0, 1.0, 2.0 / 3.0, 0.0 };
    const group_cum_gini_coeff_expected = [_]f64{ 0.0, 0.0, (4.0 / 3.0) / (2.0 * (17.0 / 3.0)), 0.0, 0.0, 1.0 / 3.0, (2.0 / 3.0) / (2.0 * (4.0 / 3.0)), 0.0 };

    var group_cum_mad_label = try distinct_table.withGroupCumulativeMeanAbsDev("bucket", "label", "label_cum_mad");
    defer group_cum_mad_label.deinit();
    try expectF64ColumnApproxOrNanWithValidity(group_cum_mad_label, gpa, "label_cum_mad", &group_cum_mad_expected, &group_cum_mode_validity_expected);

    var group_cum_mad_ratio_label = try distinct_table.withGroupCumulativeMeanAbsDevRatio("bucket", "label", "label_cum_mad_ratio");
    defer group_cum_mad_ratio_label.deinit();
    try expectF64ColumnApproxOrNanWithValidity(group_cum_mad_ratio_label, gpa, "label_cum_mad_ratio", &group_cum_mad_ratio_expected, &group_cum_mode_validity_expected);

    var group_cum_gini_mean_diff_label = try distinct_table.withGroupCumulativeGiniMeanDiff("bucket", "label", "label_cum_gini_mean_diff");
    defer group_cum_gini_mean_diff_label.deinit();
    try expectF64ColumnApproxOrNanWithValidity(group_cum_gini_mean_diff_label, gpa, "label_cum_gini_mean_diff", &group_cum_gini_mean_diff_expected, &group_cum_mode_validity_expected);

    var group_cum_gini_coeff_label = try distinct_table.withGroupCumulativeGiniCoeff("bucket", "label", "label_cum_gini_coeff");
    defer group_cum_gini_coeff_label.deinit();
    try expectF64ColumnApproxOrNanWithValidity(group_cum_gini_coeff_label, gpa, "label_cum_gini_coeff", &group_cum_gini_coeff_expected, &group_cum_mode_validity_expected);

    var cumulative_inequality_plan = try DeviceLazyFrame.init(gpa, distinct_table);
    defer cumulative_inequality_plan.deinit();
    try cumulative_inequality_plan.withGroupCumulativeMeanAbsDev("bucket", "label", "label_cum_mad_lazy");
    try cumulative_inequality_plan.withGroupCumulativeMeanAbsDevRatio("bucket", "label", "label_cum_mad_ratio_lazy");
    try cumulative_inequality_plan.withGroupCumulativeGiniMeanDiff("bucket", "label", "label_cum_gini_mean_diff_lazy");
    try cumulative_inequality_plan.withGroupCumulativeGiniCoefficient("bucket", "label", "label_cum_gini_coeff_lazy");
    const cumulative_inequality_explained = try cumulative_inequality_plan.explain(gpa);
    defer gpa.free(cumulative_inequality_explained);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_inequality_explained, "group_cumulative_mean_abs_dev([bucket], value=label->label_cum_mad_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_inequality_explained, "group_cumulative_mean_abs_dev_ratio([bucket], value=label->label_cum_mad_ratio_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_inequality_explained, "group_cumulative_gini_mean_diff([bucket], value=label->label_cum_gini_mean_diff_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_inequality_explained, "group_cumulative_gini_coefficient([bucket], value=label->label_cum_gini_coeff_lazy)") != null);
    var lazy_cumulative_inequality = try cumulative_inequality_plan.collect();
    defer lazy_cumulative_inequality.deinit();
    try expectF64ColumnApproxOrNanWithValidity(lazy_cumulative_inequality, gpa, "label_cum_mad_lazy", &group_cum_mad_expected, &group_cum_mode_validity_expected);
    try expectF64ColumnApproxOrNanWithValidity(lazy_cumulative_inequality, gpa, "label_cum_mad_ratio_lazy", &group_cum_mad_ratio_expected, &group_cum_mode_validity_expected);
    try expectF64ColumnApproxOrNanWithValidity(lazy_cumulative_inequality, gpa, "label_cum_gini_mean_diff_lazy", &group_cum_gini_mean_diff_expected, &group_cum_mode_validity_expected);
    try expectF64ColumnApproxOrNanWithValidity(lazy_cumulative_inequality, gpa, "label_cum_gini_coeff_lazy", &group_cum_gini_coeff_expected, &group_cum_mode_validity_expected);

    const group_cum_median_expected = [_]f64{ 5.0, 5.0, 5.0, 0.0, 1.0, 1.5, 1.0, 0.0 };

    var group_cum_median_label = try distinct_table.withGroupCumulativeMedian("bucket", "label", "label_cum_median");
    defer group_cum_median_label.deinit();
    try expectF64ColumnWithValidity(group_cum_median_label, gpa, "label_cum_median", &group_cum_median_expected, &group_cum_mode_validity_expected);

    var cumulative_median_plan = try DeviceLazyFrame.init(gpa, distinct_table);
    defer cumulative_median_plan.deinit();
    try cumulative_median_plan.withGroupCumulativeMedian("bucket", "label", "label_cum_median_lazy");
    const cumulative_median_explained = try cumulative_median_plan.explain(gpa);
    defer gpa.free(cumulative_median_explained);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_median_explained, "group_cumulative_median([bucket], value=label->label_cum_median_lazy)") != null);
    var lazy_cumulative_median = try cumulative_median_plan.collect();
    defer lazy_cumulative_median.deinit();
    try expectF64ColumnWithValidity(lazy_cumulative_median, gpa, "label_cum_median_lazy", &group_cum_median_expected, &group_cum_mode_validity_expected);

    const group_cum_q25_expected = [_]f64{ 5.0, 5.0, 5.0, 0.0, 1.0, 1.25, 1.0, 0.0 };

    var group_cum_q25_label = try distinct_table.withGroupCumulativeQuantile("bucket", "label", "label_cum_q25", 0.25);
    defer group_cum_q25_label.deinit();
    try expectF64ColumnWithValidity(group_cum_q25_label, gpa, "label_cum_q25", &group_cum_q25_expected, &group_cum_mode_validity_expected);
    try std.testing.expectError(error.InvalidShape, distinct_table.withGroupCumulativeQuantile("bucket", "label", "bad_q", 1.5));

    var cumulative_quantile_plan = try DeviceLazyFrame.init(gpa, distinct_table);
    defer cumulative_quantile_plan.deinit();
    try cumulative_quantile_plan.withGroupCumulativeQuantile("bucket", "label", "label_cum_q25_lazy", 0.25);
    const cumulative_quantile_explained = try cumulative_quantile_plan.explain(gpa);
    defer gpa.free(cumulative_quantile_explained);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_quantile_explained, "group_cumulative_quantile([bucket], value=label, q=0.25->label_cum_q25_lazy)") != null);
    var lazy_cumulative_quantile = try cumulative_quantile_plan.collect();
    defer lazy_cumulative_quantile.deinit();
    try expectF64ColumnWithValidity(lazy_cumulative_quantile, gpa, "label_cum_q25_lazy", &group_cum_q25_expected, &group_cum_mode_validity_expected);

    const group_cum_iqr_expected = [_]f64{ 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.0 };

    var group_cum_iqr_label = try distinct_table.withGroupCumulativeIqr("bucket", "label", "label_cum_iqr");
    defer group_cum_iqr_label.deinit();
    try expectF64ColumnWithValidity(group_cum_iqr_label, gpa, "label_cum_iqr", &group_cum_iqr_expected, &group_cum_mode_validity_expected);

    var cumulative_iqr_plan = try DeviceLazyFrame.init(gpa, distinct_table);
    defer cumulative_iqr_plan.deinit();
    try cumulative_iqr_plan.withGroupCumulativeIQR("bucket", "label", "label_cum_iqr_lazy");
    const cumulative_iqr_explained = try cumulative_iqr_plan.explain(gpa);
    defer gpa.free(cumulative_iqr_explained);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_iqr_explained, "group_cumulative_iqr([bucket], value=label->label_cum_iqr_lazy)") != null);
    var lazy_cumulative_iqr = try cumulative_iqr_plan.collect();
    defer lazy_cumulative_iqr.deinit();
    try expectF64ColumnWithValidity(lazy_cumulative_iqr, gpa, "label_cum_iqr_lazy", &group_cum_iqr_expected, &group_cum_mode_validity_expected);

    const group_cum_median_abs_dev_expected = [_]f64{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0 };

    var group_cum_median_abs_dev_label = try distinct_table.withGroupCumulativeMad("bucket", "label", "label_cum_median_abs_dev");
    defer group_cum_median_abs_dev_label.deinit();
    try expectF64ColumnWithValidity(group_cum_median_abs_dev_label, gpa, "label_cum_median_abs_dev", &group_cum_median_abs_dev_expected, &group_cum_mode_validity_expected);
    try std.testing.expectError(error.TypeUnsupported, bool_table.withGroupCumulativeMad("store", "active", "bad_mad"));

    var cumulative_mad_plan = try DeviceLazyFrame.init(gpa, distinct_table);
    defer cumulative_mad_plan.deinit();
    try cumulative_mad_plan.withGroupCumulativeMAD("bucket", "label", "label_cum_median_abs_dev_lazy");
    const cumulative_mad_explained = try cumulative_mad_plan.explain(gpa);
    defer gpa.free(cumulative_mad_explained);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_mad_explained, "group_cumulative_mad([bucket], value=label->label_cum_median_abs_dev_lazy)") != null);
    var lazy_cumulative_mad = try cumulative_mad_plan.collect();
    defer lazy_cumulative_mad.deinit();
    try expectF64ColumnWithValidity(lazy_cumulative_mad, gpa, "label_cum_median_abs_dev_lazy", &group_cum_median_abs_dev_expected, &group_cum_mode_validity_expected);

    const group_cum_idr_expected = [_]f64{ 0.0, 0.0, 8.0 / 5.0, 0.0, 0.0, 4.0 / 5.0, 4.0 / 5.0, 0.0 };

    var group_cum_idr_label = try distinct_table.withGroupCumulativeInterdecileRange("bucket", "label", "label_cum_idr");
    defer group_cum_idr_label.deinit();
    try expectF64ColumnApproxOrNanWithValidity(group_cum_idr_label, gpa, "label_cum_idr", &group_cum_idr_expected, &group_cum_mode_validity_expected);

    var cumulative_idr_plan = try DeviceLazyFrame.init(gpa, distinct_table);
    defer cumulative_idr_plan.deinit();
    try cumulative_idr_plan.withGroupCumulativeIDR("bucket", "label", "label_cum_idr_lazy");
    const cumulative_idr_explained = try cumulative_idr_plan.explain(gpa);
    defer gpa.free(cumulative_idr_explained);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_idr_explained, "group_cumulative_interdecile_range([bucket], value=label->label_cum_idr_lazy)") != null);
    var lazy_cumulative_idr = try cumulative_idr_plan.collect();
    defer lazy_cumulative_idr.deinit();
    try expectF64ColumnApproxOrNanWithValidity(lazy_cumulative_idr, gpa, "label_cum_idr_lazy", &group_cum_idr_expected, &group_cum_mode_validity_expected);

    const group_cum_midhinge_expected = [_]f64{ 5.0, 5.0, 5.5, 0.0, 1.0, 1.5, 1.25, 0.0 };

    var group_cum_midhinge_label = try distinct_table.withGroupCumulativeMidhinge("bucket", "label", "label_cum_midhinge");
    defer group_cum_midhinge_label.deinit();
    try expectF64ColumnWithValidity(group_cum_midhinge_label, gpa, "label_cum_midhinge", &group_cum_midhinge_expected, &group_cum_mode_validity_expected);

    var cumulative_midhinge_plan = try DeviceLazyFrame.init(gpa, distinct_table);
    defer cumulative_midhinge_plan.deinit();
    try cumulative_midhinge_plan.withGroupCumMidhinge("bucket", "label", "label_cum_midhinge_lazy");
    const cumulative_midhinge_explained = try cumulative_midhinge_plan.explain(gpa);
    defer gpa.free(cumulative_midhinge_explained);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_midhinge_explained, "group_cumulative_midhinge([bucket], value=label->label_cum_midhinge_lazy)") != null);
    var lazy_cumulative_midhinge = try cumulative_midhinge_plan.collect();
    defer lazy_cumulative_midhinge.deinit();
    try expectF64ColumnWithValidity(lazy_cumulative_midhinge, gpa, "label_cum_midhinge_lazy", &group_cum_midhinge_expected, &group_cum_mode_validity_expected);

    const group_cum_trimean_expected = [_]f64{ 5.0, 5.0, 5.25, 0.0, 1.0, 1.5, 1.125, 0.0 };

    var group_cum_trimean_label = try distinct_table.withGroupCumulativeTrimean("bucket", "label", "label_cum_trimean");
    defer group_cum_trimean_label.deinit();
    try expectF64ColumnWithValidity(group_cum_trimean_label, gpa, "label_cum_trimean", &group_cum_trimean_expected, &group_cum_mode_validity_expected);

    var cumulative_trimean_plan = try DeviceLazyFrame.init(gpa, distinct_table);
    defer cumulative_trimean_plan.deinit();
    try cumulative_trimean_plan.withGroupCumTrimean("bucket", "label", "label_cum_trimean_lazy");
    const cumulative_trimean_explained = try cumulative_trimean_plan.explain(gpa);
    defer gpa.free(cumulative_trimean_explained);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_trimean_explained, "group_cumulative_trimean([bucket], value=label->label_cum_trimean_lazy)") != null);
    var lazy_cumulative_trimean = try cumulative_trimean_plan.collect();
    defer lazy_cumulative_trimean.deinit();
    try expectF64ColumnWithValidity(lazy_cumulative_trimean, gpa, "label_cum_trimean_lazy", &group_cum_trimean_expected, &group_cum_mode_validity_expected);

    const group_cum_bowley_expected = [_]f64{ std.math.nan(f64), std.math.nan(f64), 1.0, 0.0, std.math.nan(f64), 0.0, 1.0, 0.0 };

    var group_cum_bowley_label = try distinct_table.withGroupCumulativeBowleySkewness("bucket", "label", "label_cum_bowley");
    defer group_cum_bowley_label.deinit();
    try expectF64ColumnApproxOrNanWithValidity(group_cum_bowley_label, gpa, "label_cum_bowley", &group_cum_bowley_expected, &group_cum_mode_validity_expected);

    var cumulative_bowley_plan = try DeviceLazyFrame.init(gpa, distinct_table);
    defer cumulative_bowley_plan.deinit();
    try cumulative_bowley_plan.withGroupCumBowleySkew("bucket", "label", "label_cum_bowley_lazy");
    const cumulative_bowley_explained = try cumulative_bowley_plan.explain(gpa);
    defer gpa.free(cumulative_bowley_explained);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_bowley_explained, "group_cumulative_bowley_skewness([bucket], value=label->label_cum_bowley_lazy)") != null);
    var lazy_cumulative_bowley = try cumulative_bowley_plan.collect();
    defer lazy_cumulative_bowley.deinit();
    try expectF64ColumnApproxOrNanWithValidity(lazy_cumulative_bowley, gpa, "label_cum_bowley_lazy", &group_cum_bowley_expected, &group_cum_mode_validity_expected);

    const group_cum_qcd_expected = [_]f64{ 0.0, 0.0, 1.0 / 11.0, 0.0, 0.0, 1.0 / 6.0, 1.0 / 5.0, 0.0 };

    var group_cum_qcd_label = try distinct_table.withGroupCumulativeQuartileCoeffDispersion("bucket", "label", "label_cum_qcd");
    defer group_cum_qcd_label.deinit();
    try expectF64ColumnApproxOrNanWithValidity(group_cum_qcd_label, gpa, "label_cum_qcd", &group_cum_qcd_expected, &group_cum_mode_validity_expected);

    var cumulative_qcd_plan = try DeviceLazyFrame.init(gpa, distinct_table);
    defer cumulative_qcd_plan.deinit();
    try cumulative_qcd_plan.withGroupCumulativeQCD("bucket", "label", "label_cum_qcd_lazy");
    const cumulative_qcd_explained = try cumulative_qcd_plan.explain(gpa);
    defer gpa.free(cumulative_qcd_explained);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_qcd_explained, "group_cumulative_quartile_coeff_dispersion([bucket], value=label->label_cum_qcd_lazy)") != null);
    var lazy_cumulative_qcd = try cumulative_qcd_plan.collect();
    defer lazy_cumulative_qcd.deinit();
    try expectF64ColumnApproxOrNanWithValidity(lazy_cumulative_qcd, gpa, "label_cum_qcd_lazy", &group_cum_qcd_expected, &group_cum_mode_validity_expected);

    const group_cum_kelley_expected = [_]f64{ std.math.nan(f64), std.math.nan(f64), 1.0, 0.0, std.math.nan(f64), 0.0, 1.0, 0.0 };

    var group_cum_kelley_label = try distinct_table.withGroupCumulativeKelleySkewness("bucket", "label", "label_cum_kelley");
    defer group_cum_kelley_label.deinit();
    try expectF64ColumnApproxOrNanWithValidity(group_cum_kelley_label, gpa, "label_cum_kelley", &group_cum_kelley_expected, &group_cum_mode_validity_expected);

    var cumulative_kelley_plan = try DeviceLazyFrame.init(gpa, distinct_table);
    defer cumulative_kelley_plan.deinit();
    try cumulative_kelley_plan.withGroupCumKelleySkew("bucket", "label", "label_cum_kelley_lazy");
    const cumulative_kelley_explained = try cumulative_kelley_plan.explain(gpa);
    defer gpa.free(cumulative_kelley_explained);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_kelley_explained, "group_cumulative_kelley_skewness([bucket], value=label->label_cum_kelley_lazy)") != null);
    var lazy_cumulative_kelley = try cumulative_kelley_plan.collect();
    defer lazy_cumulative_kelley.deinit();
    try expectF64ColumnApproxOrNanWithValidity(lazy_cumulative_kelley, gpa, "label_cum_kelley_lazy", &group_cum_kelley_expected, &group_cum_mode_validity_expected);

    var trim_key = try DeviceColumn.fromSliceWithValidity(i32, gpa, &.{ 1, 1, 1, 1, 1, 2, 2 }, &.{ true, true, true, true, true, true, false }, .cpu);
    defer trim_key.deinit();
    var trim_value = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, 2.0, 100.0, 4.0, 9.0, 10.0, 20.0 }, &.{ true, true, true, true, false, true, true }, .cpu);
    defer trim_value.deinit();
    var trim_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "trim_bucket", .data = trim_key },
        .{ .name = "trim_value", .data = trim_value },
    });
    defer trim_table.deinit();
    const group_cum_trimmed_expected = [_]f64{ 1.0, 1.5, 103.0 / 3.0, 3.0, 0.0, 10.0, 0.0 };
    const group_cum_trimmed_validity = [_]bool{ true, true, true, true, false, true, false };

    var group_cum_trimmed = try trim_table.withGroupCumulativeTrimmedMean("trim_bucket", "trim_value", "trim_value_cum_trimmed", 0.25);
    defer group_cum_trimmed.deinit();
    try expectF64ColumnApproxOrNanWithValidity(group_cum_trimmed, gpa, "trim_value_cum_trimmed", &group_cum_trimmed_expected, &group_cum_trimmed_validity);
    try std.testing.expectError(error.InvalidShape, trim_table.withGroupCumulativeTrimmedMean("trim_bucket", "trim_value", "bad_trimmed", 0.5));

    var cumulative_trimmed_plan = try DeviceLazyFrame.init(gpa, trim_table);
    defer cumulative_trimmed_plan.deinit();
    try cumulative_trimmed_plan.withGroupCumTrimmedMean("trim_bucket", "trim_value", "trim_value_cum_trimmed_lazy", 0.25);
    const cumulative_trimmed_explained = try cumulative_trimmed_plan.explain(gpa);
    defer gpa.free(cumulative_trimmed_explained);
    try std.testing.expect(std.mem.indexOf(u8, cumulative_trimmed_explained, "group_cumulative_trimmed_mean([trim_bucket], value=trim_value, trim_fraction=0.25->trim_value_cum_trimmed_lazy)") != null);
    var lazy_cumulative_trimmed = try cumulative_trimmed_plan.collect();
    defer lazy_cumulative_trimmed.deinit();
    try expectF64ColumnApproxOrNanWithValidity(lazy_cumulative_trimmed, gpa, "trim_value_cum_trimmed_lazy", &group_cum_trimmed_expected, &group_cum_trimmed_validity);

    var group_cum_sum_sales = try table.withGroupCumulativeSum("store", "sales", "store_sales_cum_sum");
    defer group_cum_sum_sales.deinit();
    try expectF64ColumnWithValidity(group_cum_sum_sales, gpa, "store_sales_cum_sum", &.{ 2.0, 3.0, 0.0, 0.0, 14.0, 15.0 }, &.{ true, true, false, false, true, true });

    var group_cum_mean_sales = try table.withGroupCumulativeMean("store", "sales", "store_sales_cum_mean");
    defer group_cum_mean_sales.deinit();
    try expectF64ColumnWithValidity(group_cum_mean_sales, gpa, "store_sales_cum_mean", &.{ 2.0, 3.0, 0.0, 0.0, 7.0, 7.5 }, &.{ true, true, false, false, true, true });

    var group_cum_product_sales = try table.withGroupCumulativeProduct("store", "sales", "store_sales_cum_product");
    defer group_cum_product_sales.deinit();
    try expectF64ColumnWithValidity(group_cum_product_sales, gpa, "store_sales_cum_product", &.{ 2.0, 3.0, 0.0, 0.0, 33.0, 26.0 }, &.{ true, true, false, false, true, true });

    var group_cum_min_sales = try table.withGroupCumulativeMin("store", "sales", "store_sales_cum_min");
    defer group_cum_min_sales.deinit();
    try expectF64ColumnWithValidity(group_cum_min_sales, gpa, "store_sales_cum_min", &.{ 2.0, 3.0, 0.0, 0.0, 3.0, 2.0 }, &.{ true, true, false, false, true, true });

    var group_cum_max_sales = try table.withGroupCumulativeMax("store", "sales", "store_sales_cum_max");
    defer group_cum_max_sales.deinit();
    try expectF64ColumnWithValidity(group_cum_max_sales, gpa, "store_sales_cum_max", &.{ 2.0, 3.0, 0.0, 0.0, 11.0, 13.0 }, &.{ true, true, false, false, true, true });

    var group_cum_variance_sales = try table.withGroupCumulativeVariance("store", "sales", "store_sales_cum_variance");
    defer group_cum_variance_sales.deinit();
    try expectF64ColumnWithValidity(group_cum_variance_sales, gpa, "store_sales_cum_variance", &.{ 0.0, 0.0, 0.0, 0.0, 16.0, 30.25 }, &.{ true, true, false, false, true, true });

    var group_cum_stddev_sales = try table.withGroupCumulativeStddev("store", "sales", "store_sales_cum_stddev");
    defer group_cum_stddev_sales.deinit();
    try expectF64ColumnWithValidity(group_cum_stddev_sales, gpa, "store_sales_cum_stddev", &.{ 0.0, 0.0, 0.0, 0.0, 4.0, 5.5 }, &.{ true, true, false, false, true, true });

    var group_cum_sem_sales = try table.withGroupCumulativeSem("store", "sales", "store_sales_cum_sem");
    defer group_cum_sem_sales.deinit();
    try expectF64ColumnApproxOrNanWithValidity(group_cum_sem_sales, gpa, "store_sales_cum_sem", &.{ 0.0, 0.0, 0.0, 0.0, 4.0 / std.math.sqrt(@as(f64, 2.0)), 5.5 / std.math.sqrt(@as(f64, 2.0)) }, &.{ true, true, false, false, true, true });

    var group_cum_cv_sales = try table.withGroupCumulativeCv("store", "sales", "store_sales_cum_cv");
    defer group_cum_cv_sales.deinit();
    try expectF64ColumnApproxOrNanWithValidity(group_cum_cv_sales, gpa, "store_sales_cum_cv", &.{ 0.0, 0.0, 0.0, 0.0, 4.0 / 7.0, 11.0 / 15.0 }, &.{ true, true, false, false, true, true });

    var group_cum_fano_sales = try table.withGroupCumulativeFano("store", "sales", "store_sales_cum_fano");
    defer group_cum_fano_sales.deinit();
    try expectF64ColumnApproxOrNanWithValidity(group_cum_fano_sales, gpa, "store_sales_cum_fano", &.{ 0.0, 0.0, 0.0, 0.0, 16.0 / 7.0, 121.0 / 30.0 }, &.{ true, true, false, false, true, true });

    var group_cum_skew_sales = try table.withGroupCumulativeSkewness("store", "sales", "store_sales_cum_skewness");
    defer group_cum_skew_sales.deinit();
    try expectF64ColumnApproxOrNanWithValidity(group_cum_skew_sales, gpa, "store_sales_cum_skewness", &.{ std.math.nan(f64), std.math.nan(f64), 0.0, 0.0, 0.0, 0.0 }, &.{ true, true, false, false, true, true });

    var group_cum_kurt_sales = try table.withGroupCumulativeKurtosis("store", "sales", "store_sales_cum_kurtosis");
    defer group_cum_kurt_sales.deinit();
    try expectF64ColumnApproxOrNanWithValidity(group_cum_kurt_sales, gpa, "store_sales_cum_kurtosis", &.{ std.math.nan(f64), std.math.nan(f64), 0.0, 0.0, -2.0, -2.0 }, &.{ true, true, false, false, true, true });

    var group_cum_mean_abs_delta = try table.withGroupCumulativeMeanAbs("store", "delta", "store_delta_cum_mean_abs");
    defer group_cum_mean_abs_delta.deinit();
    try expectF64ColumnWithValidity(group_cum_mean_abs_delta, gpa, "store_delta_cum_mean_abs", &.{ 2.0, 3.0, 0.0, 0.0, 7.0, 7.5 }, &.{ true, true, false, false, true, true });

    var group_cum_mean_square_delta = try table.withGroupCumulativeMeanSquare("store", "delta", "store_delta_cum_mean_square");
    defer group_cum_mean_square_delta.deinit();
    try expectF64ColumnWithValidity(group_cum_mean_square_delta, gpa, "store_delta_cum_mean_square", &.{ 4.0, 9.0, 0.0, 0.0, 65.0, 86.5 }, &.{ true, true, false, false, true, true });

    var group_cum_rms_delta = try table.withGroupCumulativeRms("store", "delta", "store_delta_cum_rms");
    defer group_cum_rms_delta.deinit();
    try expectF64ColumnApproxOrNanWithValidity(group_cum_rms_delta, gpa, "store_delta_cum_rms", &.{ 2.0, 3.0, 0.0, 0.0, std.math.sqrt(@as(f64, 65.0)), std.math.sqrt(@as(f64, 86.5)) }, &.{ true, true, false, false, true, true });

    var group_cum_max_abs_delta = try table.withGroupCumulativeMaxAbs("store", "delta", "store_delta_cum_max_abs");
    defer group_cum_max_abs_delta.deinit();
    try expectF64ColumnWithValidity(group_cum_max_abs_delta, gpa, "store_delta_cum_max_abs", &.{ 2.0, 3.0, 0.0, 0.0, 11.0, 13.0 }, &.{ true, true, false, false, true, true });

    var group_cum_min_abs_delta = try table.withGroupCumulativeMinAbs("store", "delta", "store_delta_cum_min_abs");
    defer group_cum_min_abs_delta.deinit();
    try expectF64ColumnWithValidity(group_cum_min_abs_delta, gpa, "store_delta_cum_min_abs", &.{ 2.0, 3.0, 0.0, 0.0, 3.0, 2.0 }, &.{ true, true, false, false, true, true });

    var group_cum_l1_delta = try table.withGroupCumulativeL1Norm("store", "delta", "store_delta_cum_l1");
    defer group_cum_l1_delta.deinit();
    try expectF64ColumnWithValidity(group_cum_l1_delta, gpa, "store_delta_cum_l1", &.{ 2.0, 3.0, 0.0, 0.0, 14.0, 15.0 }, &.{ true, true, false, false, true, true });

    var group_cum_l2_delta = try table.withGroupCumulativeL2Norm("store", "delta", "store_delta_cum_l2");
    defer group_cum_l2_delta.deinit();
    try expectF64ColumnApproxOrNanWithValidity(group_cum_l2_delta, gpa, "store_delta_cum_l2", &.{ 2.0, 3.0, 0.0, 0.0, std.math.sqrt(@as(f64, 130.0)), std.math.sqrt(@as(f64, 173.0)) }, &.{ true, true, false, false, true, true });

    var group_cum_range_delta = try table.withGroupCumulativeRange("store", "delta", "store_delta_cum_range");
    defer group_cum_range_delta.deinit();
    try expectF64ColumnWithValidity(group_cum_range_delta, gpa, "store_delta_cum_range", &.{ 0.0, 0.0, 0.0, 0.0, 8.0, 15.0 }, &.{ true, true, false, false, true, true });

    var group_cum_midrange_delta = try table.withGroupCumulativeMidrange("store", "delta", "store_delta_cum_midrange");
    defer group_cum_midrange_delta.deinit();
    try expectF64ColumnWithValidity(group_cum_midrange_delta, gpa, "store_delta_cum_midrange", &.{ -2.0, -3.0, 0.0, 0.0, -7.0, 5.5 }, &.{ true, true, false, false, true, true });

    var group_cum_range_coeff_delta = try table.withGroupCumulativeRangeCoeff("store", "delta", "store_delta_cum_range_coeff");
    defer group_cum_range_coeff_delta.deinit();
    try expectF64ColumnApproxOrNanWithValidity(group_cum_range_coeff_delta, gpa, "store_delta_cum_range_coeff", &.{ -0.0, -0.0, 0.0, 0.0, -8.0 / 14.0, 15.0 / 11.0 }, &.{ true, true, false, false, true, true });

    var group_cum_logsumexp_sales = try table.withGroupCumulativeLogSumExp("store", "sales", "store_sales_cum_logsumexp");
    defer group_cum_logsumexp_sales.deinit();
    try expectF64ColumnApproxOrNanWithValidity(group_cum_logsumexp_sales, gpa, "store_sales_cum_logsumexp", &.{ 2.0, 3.0, 0.0, 0.0, 11.0 + std.math.log(f64, std.math.e, 1.0 + std.math.exp(-8.0)), 13.0 + std.math.log(f64, std.math.e, 1.0 + std.math.exp(-11.0)) }, &.{ true, true, false, false, true, true });

    var group_cum_logmeanexp_sales = try table.withGroupCumulativeLogMeanExp("store", "sales", "store_sales_cum_logmeanexp");
    defer group_cum_logmeanexp_sales.deinit();
    try expectF64ColumnApproxOrNanWithValidity(group_cum_logmeanexp_sales, gpa, "store_sales_cum_logmeanexp", &.{ 2.0, 3.0, 0.0, 0.0, 11.0 + std.math.log(f64, std.math.e, 1.0 + std.math.exp(-8.0)) - std.math.log(f64, std.math.e, 2.0), 13.0 + std.math.log(f64, std.math.e, 1.0 + std.math.exp(-11.0)) - std.math.log(f64, std.math.e, 2.0) }, &.{ true, true, false, false, true, true });

    var group_cum_geometric_sales = try table.withGroupCumulativeGeometricMean("store", "sales", "store_sales_cum_geometric");
    defer group_cum_geometric_sales.deinit();
    try expectF64ColumnApproxOrNanWithValidity(group_cum_geometric_sales, gpa, "store_sales_cum_geometric", &.{ 2.0, 3.0, 0.0, 0.0, std.math.sqrt(@as(f64, 33.0)), std.math.sqrt(@as(f64, 26.0)) }, &.{ true, true, false, false, true, true });

    var group_cum_harmonic_sales = try table.withGroupCumulativeHarmonicMean("store", "sales", "store_sales_cum_harmonic");
    defer group_cum_harmonic_sales.deinit();
    try expectF64ColumnApproxOrNanWithValidity(group_cum_harmonic_sales, gpa, "store_sales_cum_harmonic", &.{ 2.0, 3.0, 0.0, 0.0, 33.0 / 7.0, 52.0 / 15.0 }, &.{ true, true, false, false, true, true });

    var group_cum_argmin_sales = try table.withGroupCumulativeArgMin("store", "sales", "store_sales_cum_argmin");
    defer group_cum_argmin_sales.deinit();
    try expectNullableI64Column(group_cum_argmin_sales, gpa, "store_sales_cum_argmin", &.{ 0, 1, 0, 0, 1, 0 }, &.{ true, true, false, false, true, true });

    var group_cum_argmax_sales = try table.withGroupCumulativeArgMax("store", "sales", "store_sales_cum_argmax");
    defer group_cum_argmax_sales.deinit();
    try expectNullableI64Column(group_cum_argmax_sales, gpa, "store_sales_cum_argmax", &.{ 0, 1, 0, 0, 4, 5 }, &.{ true, true, false, false, true, true });

    var group_cum_first_valid_sales = try table.withGroupCumulativeFirstValidIndex("store", "sales", "store_sales_cum_first_valid_index");
    defer group_cum_first_valid_sales.deinit();
    try expectNullableI64Column(group_cum_first_valid_sales, gpa, "store_sales_cum_first_valid_index", &.{ 0, 1, 0, 0, 1, 0 }, &.{ true, true, true, false, true, true });

    var group_cum_last_valid_sales = try table.withGroupCumulativeLastValidIndex("store", "sales", "store_sales_cum_last_valid_index");
    defer group_cum_last_valid_sales.deinit();
    try expectNullableI64Column(group_cum_last_valid_sales, gpa, "store_sales_cum_last_valid_index", &.{ 0, 1, 0, 0, 4, 5 }, &.{ true, true, true, false, true, true });

    var group_cum_first_null_sales = try table.withGroupCumulativeFirstNullIndex("store", "sales", "store_sales_cum_first_null_index");
    defer group_cum_first_null_sales.deinit();
    try expectNullableI64Column(group_cum_first_null_sales, gpa, "store_sales_cum_first_null_index", &.{ 0, 0, 2, 0, 0, 2 }, &.{ false, false, true, false, false, true });

    var group_cum_last_null_sales = try table.withGroupCumulativeLastNullIndex("store", "sales", "store_sales_cum_last_null_index");
    defer group_cum_last_null_sales.deinit();
    try expectNullableI64Column(group_cum_last_null_sales, gpa, "store_sales_cum_last_null_index", &.{ 0, 0, 2, 0, 0, 2 }, &.{ false, false, true, false, false, true });

    var group_row_numbers = try table.withGroupRowNumber("store", "store_row_number");
    defer group_row_numbers.deinit();
    try expectNullableI64Column(group_row_numbers, gpa, "store_row_number", &.{ 0, 0, 1, 0, 1, 2 }, &.{ true, true, true, false, true, true });

    var group_sizes = try table.withGroupSize("store", "store_group_size");
    defer group_sizes.deinit();
    try expectNullableI64Column(group_sizes, gpa, "store_group_size", &.{ 3, 2, 3, 0, 2, 3 }, &.{ true, true, true, false, true, true });

    var group_reverse_row_numbers = try table.withGroupReverseRowNumber("store", "store_reverse_row_number");
    defer group_reverse_row_numbers.deinit();
    try expectNullableI64Column(group_reverse_row_numbers, gpa, "store_reverse_row_number", &.{ 2, 1, 1, 0, 0, 0 }, &.{ true, true, true, false, true, true });

    var value_counts = try table.valueCounts("store");
    defer value_counts.deinit();
    const value_count_keys = try (try value_counts.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(value_count_keys);
    const value_count_values = try (try value_counts.column("count")).i64.toOwnedSlice(gpa);
    defer gpa.free(value_count_values);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2 }, value_count_keys);
    try std.testing.expectEqualSlices(i64, &.{ 3, 2 }, value_count_values);

    var named_value_counts = try table.valueCountsAs("store", "rows_named");
    defer named_value_counts.deinit();
    const named_counts = try (try named_value_counts.column("rows_named")).i64.toOwnedSlice(gpa);
    defer gpa.free(named_counts);
    try std.testing.expectEqualSlices(i64, &.{ 3, 2 }, named_counts);

    var sorted_value_counts = try table.valueCountsSorted("store");
    defer sorted_value_counts.deinit();
    const sorted_value_count_keys = try (try sorted_value_counts.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(sorted_value_count_keys);
    const sorted_value_count_values = try (try sorted_value_counts.column("count")).i64.toOwnedSlice(gpa);
    defer gpa.free(sorted_value_count_values);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2 }, sorted_value_count_keys);
    try std.testing.expectEqualSlices(i64, &.{ 3, 2 }, sorted_value_count_values);

    var head_rows = try table.groupByHeadRows("store", 2);
    defer head_rows.deinit();
    const head_row_keys = try (try head_rows.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(head_row_keys);
    const head_row_sales = try (try head_rows.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(head_row_sales);
    const head_row_sales_validity = try (try head_rows.column("sales")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(head_row_sales_validity);
    try std.testing.expectEqualSlices(i32, &.{ 1, 1, 2, 2 }, head_row_keys);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 5.0, 3.0, 11.0 }, head_row_sales);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, true }, head_row_sales_validity);

    var tail_rows = try table.groupByTailRows("store", 1);
    defer tail_rows.deinit();
    const tail_row_keys = try (try tail_rows.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(tail_row_keys);
    const tail_row_sales = try (try tail_rows.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(tail_row_sales);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2 }, tail_row_keys);
    try std.testing.expectEqualSlices(f64, &.{ 13.0, 11.0 }, tail_row_sales);

    var slice_rows = try table.groupBySliceRows("store", 1, 2);
    defer slice_rows.deinit();
    const slice_row_keys = try (try slice_rows.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(slice_row_keys);
    const slice_row_sales = try (try slice_rows.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(slice_row_sales);
    const slice_row_sales_validity = try (try slice_rows.column("sales")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(slice_row_sales_validity);
    try std.testing.expectEqualSlices(i32, &.{ 1, 1, 2 }, slice_row_keys);
    try std.testing.expectEqualSlices(f64, &.{ 5.0, 13.0, 11.0 }, slice_row_sales);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true }, slice_row_sales_validity);

    var stepped_slice_rows = try table.groupBySliceRowsStep("store", 0, 3, 2);
    defer stepped_slice_rows.deinit();
    const stepped_slice_keys = try (try stepped_slice_rows.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(stepped_slice_keys);
    try std.testing.expectEqualSlices(i32, &.{ 1, 1, 2 }, stepped_slice_keys);
    try expectF64ColumnWithValidity(stepped_slice_rows, gpa, "sales", &.{ 2.0, 13.0, 3.0 }, &.{ true, true, true });
    try std.testing.expectError(error.InvalidShape, table.groupBySliceRowsStep("store", 0, 2, 0));

    var signed_slice_rows = try table.groupBySliceRowsSigned("store", -2, 2);
    defer signed_slice_rows.deinit();
    const signed_slice_keys = try (try signed_slice_rows.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(signed_slice_keys);
    try std.testing.expectEqualSlices(i32, &.{ 1, 1, 2, 2 }, signed_slice_keys);
    try expectF64ColumnWithValidity(signed_slice_rows, gpa, "sales", &.{ 5.0, 13.0, 3.0, 11.0 }, &.{ false, true, true, true });
    try std.testing.expectError(error.IndexOutOfBounds, table.groupBySliceRowsSigned("store", -4, 1));

    var top_sales_rows = try table.groupByTopRows("store", "sales", 2, .{ .descending = true, .nulls = .last });
    defer top_sales_rows.deinit();
    const top_sales_keys = try (try top_sales_rows.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(top_sales_keys);
    const top_sales_values = try (try top_sales_rows.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(top_sales_values);
    try std.testing.expectEqualSlices(i32, &.{ 1, 1, 2, 2 }, top_sales_keys);
    try std.testing.expectEqualSlices(f64, &.{ 13.0, 2.0, 11.0, 3.0 }, top_sales_values);

    var bottom_sales_rows = try table.groupByBottomRows("store", "sales", 1, .{ .descending = true, .nulls = .last });
    defer bottom_sales_rows.deinit();
    const bottom_sales_keys = try (try bottom_sales_rows.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(bottom_sales_keys);
    const bottom_sales_values = try (try bottom_sales_rows.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(bottom_sales_values);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2 }, bottom_sales_keys);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 3.0 }, bottom_sales_values);

    var sorted_row_key = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 1, 1, 2, 2, 2 }, .cpu);
    defer sorted_row_key.deinit();
    var sorted_row_score = try DeviceColumn.fromSlice(i32, gpa, &.{ 10, 10, 7, 5, 5, 9 }, .cpu);
    defer sorted_row_score.deinit();
    var sorted_row_tie = try DeviceColumn.fromSlice(i32, gpa, &.{ 2, 1, 3, 3, 1, 2 }, .cpu);
    defer sorted_row_tie.deinit();
    var sorted_row_id = try DeviceColumn.fromSlice(i32, gpa, &.{ 10, 11, 12, 20, 21, 22 }, .cpu);
    defer sorted_row_id.deinit();
    var sorted_row_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "bucket", .data = sorted_row_key },
        .{ .name = "score", .data = sorted_row_score },
        .{ .name = "tie", .data = sorted_row_tie },
        .{ .name = "id", .data = sorted_row_id },
    });
    defer sorted_row_table.deinit();

    const sorted_row_options = [_]vectra.DeviceSortOptions{ .{ .descending = true }, .{} };
    var top_sorted_rows = try sorted_row_table.groupByTopRowsByColumns("bucket", &.{ "score", "tie" }, 2, &sorted_row_options);
    defer top_sorted_rows.deinit();
    const top_sorted_ids = try (try top_sorted_rows.column("id")).i32.toOwnedSlice(gpa);
    defer gpa.free(top_sorted_ids);
    try std.testing.expectEqualSlices(i32, &.{ 11, 10, 22, 21 }, top_sorted_ids);

    var bottom_sorted_rows = try sorted_row_table.groupByBottomRowsByColumns("bucket", &.{ "score", "tie" }, 1, &sorted_row_options);
    defer bottom_sorted_rows.deinit();
    const bottom_sorted_ids = try (try bottom_sorted_rows.column("id")).i32.toOwnedSlice(gpa);
    defer gpa.free(bottom_sorted_ids);
    try std.testing.expectEqualSlices(i32, &.{ 12, 20 }, bottom_sorted_ids);

    var value_counts_plan = try DeviceLazyFrame.init(gpa, table);
    defer value_counts_plan.deinit();
    try value_counts_plan.valueCountsSortedAs("store", "rows_lazy");
    const value_counts_explained = try value_counts_plan.explain(gpa);
    defer gpa.free(value_counts_explained);
    try std.testing.expect(std.mem.indexOf(u8, value_counts_explained, "group_by_count(store -> rows_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, value_counts_explained, "sort_by(rows_lazy, desc=true)") != null);
    var lazy_value_counts = try value_counts_plan.collect();
    defer lazy_value_counts.deinit();
    const lazy_value_count_keys = try (try lazy_value_counts.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(lazy_value_count_keys);
    const lazy_value_count_values = try (try lazy_value_counts.column("rows_lazy")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_value_count_values);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2 }, lazy_value_count_keys);
    try std.testing.expectEqualSlices(i64, &.{ 3, 2 }, lazy_value_count_values);

    var group_id_plan = try DeviceLazyFrame.init(gpa, table);
    defer group_id_plan.deinit();
    try group_id_plan.withGroupId("store", "store_group_id_lazy");
    const group_id_explained = try group_id_plan.explain(gpa);
    defer gpa.free(group_id_explained);
    try std.testing.expect(std.mem.indexOf(u8, group_id_explained, "group_id([store]->store_group_id_lazy)") != null);
    var lazy_group_ids = try group_id_plan.collect();
    defer lazy_group_ids.deinit();
    try expectNullableI64Column(lazy_group_ids, gpa, "store_group_id_lazy", &.{ 0, 1, 0, 0, 1, 0 }, &.{ true, true, true, false, true, true });

    var group_boundary_plan = try DeviceLazyFrame.init(gpa, table);
    defer group_boundary_plan.deinit();
    try group_boundary_plan.withGroupFirstRowIndex("store", "store_first_row_index_lazy");
    try group_boundary_plan.withGroupLastRowIndex("store", "store_last_row_index_lazy");
    const group_boundary_explained = try group_boundary_plan.explain(gpa);
    defer gpa.free(group_boundary_explained);
    try std.testing.expect(std.mem.indexOf(u8, group_boundary_explained, "group_first_row_index([store]->store_first_row_index_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_boundary_explained, "group_last_row_index([store]->store_last_row_index_lazy)") != null);
    var lazy_group_boundaries = try group_boundary_plan.collect();
    defer lazy_group_boundaries.deinit();
    try expectNullableI64Column(lazy_group_boundaries, gpa, "store_first_row_index_lazy", &.{ 0, 1, 0, 0, 1, 0 }, &.{ true, true, true, false, true, true });
    try expectNullableI64Column(lazy_group_boundaries, gpa, "store_last_row_index_lazy", &.{ 5, 4, 5, 0, 4, 5 }, &.{ true, true, true, false, true, true });

    var group_boundary_flag_plan = try DeviceLazyFrame.init(gpa, table);
    defer group_boundary_flag_plan.deinit();
    try group_boundary_flag_plan.withGroupIsFirstRow("store", "store_is_first_row_lazy");
    try group_boundary_flag_plan.withGroupIsLastRow("store", "store_is_last_row_lazy");
    const group_boundary_flag_explained = try group_boundary_flag_plan.explain(gpa);
    defer gpa.free(group_boundary_flag_explained);
    try std.testing.expect(std.mem.indexOf(u8, group_boundary_flag_explained, "group_is_first_row([store]->store_is_first_row_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_boundary_flag_explained, "group_is_last_row([store]->store_is_last_row_lazy)") != null);
    var lazy_group_boundary_flags = try group_boundary_flag_plan.collect();
    defer lazy_group_boundary_flags.deinit();
    try expectNullableBoolColumn(lazy_group_boundary_flags, gpa, "store_is_first_row_lazy", &.{ true, true, false, false, false, false }, &.{ true, true, true, false, true, true });
    try expectNullableBoolColumn(lazy_group_boundary_flags, gpa, "store_is_last_row_lazy", &.{ false, false, false, false, true, true }, &.{ true, true, true, false, true, true });

    var group_singleton_plan = try DeviceLazyFrame.init(gpa, table);
    defer group_singleton_plan.deinit();
    try group_singleton_plan.withGroupIsSingletonOn(&.{ "store", "sales" }, "store_sales_is_singleton_lazy");
    try group_singleton_plan.withGroupIsDuplicated("store", "store_is_duplicated_group_lazy");
    const group_singleton_explained = try group_singleton_plan.explain(gpa);
    defer gpa.free(group_singleton_explained);
    try std.testing.expect(std.mem.indexOf(u8, group_singleton_explained, "group_is_singleton([store,sales]->store_sales_is_singleton_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_singleton_explained, "group_is_duplicated([store]->store_is_duplicated_group_lazy)") != null);
    var lazy_group_singletons = try group_singleton_plan.collect();
    defer lazy_group_singletons.deinit();
    try expectNullableBoolColumn(lazy_group_singletons, gpa, "store_sales_is_singleton_lazy", &.{ true, true, false, false, true, true }, &.{ true, true, false, false, true, true });
    try expectNullableBoolColumn(lazy_group_singletons, gpa, "store_is_duplicated_group_lazy", &.{ true, true, true, false, true, true }, &.{ true, true, true, false, true, true });

    var group_cume_dist_plan = try DeviceLazyFrame.init(gpa, table);
    defer group_cume_dist_plan.deinit();
    try group_cume_dist_plan.withGroupCumeDist("store", "store_cume_dist_lazy");
    try group_cume_dist_plan.withGroupPercentRank("store", "store_percent_rank_lazy");
    try group_cume_dist_plan.withGroupReverseCumeDist("store", "store_reverse_cume_dist_lazy");
    try group_cume_dist_plan.withGroupReversePercentRank("store", "store_reverse_percent_rank_lazy");
    try group_cume_dist_plan.withGroupLag("store", "sales", "store_sales_lag_lazy", 1);
    try group_cume_dist_plan.withGroupLead("store", "sales", "store_sales_lead_lazy", 1);
    try group_cume_dist_plan.withGroupFirstRowValue("store", "sales", "store_sales_first_lazy");
    try group_cume_dist_plan.withGroupLastRowValue("store", "sales", "store_sales_last_lazy");
    try group_cume_dist_plan.withGroupNthRowValue("store", "sales", "store_sales_nth_lazy", 1);
    try group_cume_dist_plan.withGroupFirstValidValue("store", "sales", "store_sales_first_valid_lazy");
    try group_cume_dist_plan.withGroupLastValidValue("store", "sales", "store_sales_last_valid_lazy");
    try group_cume_dist_plan.withGroupNthValidValue("store", "sales", "store_sales_nth_valid_lazy", 1);
    try group_cume_dist_plan.withGroupFillNullForward("store", "sales", "store_sales_ffill_lazy");
    try group_cume_dist_plan.withGroupFillNullBackward("store", "sales", "store_sales_bfill_lazy");
    try group_cume_dist_plan.withGroupCumulativeValidCount("store", "sales", "store_sales_cum_valid_lazy");
    try group_cume_dist_plan.withGroupCumulativeNullCount("store", "sales", "store_sales_cum_null_lazy");
    try group_cume_dist_plan.withGroupCumulativeValidRatio("store", "sales", "store_sales_cum_valid_ratio_lazy");
    try group_cume_dist_plan.withGroupCumulativeNullRatio("store", "sales", "store_sales_cum_null_ratio_lazy");
    try group_cume_dist_plan.withGroupCumulativeSum("store", "sales", "store_sales_cum_sum_lazy");
    try group_cume_dist_plan.withGroupCumulativeMean("store", "sales", "store_sales_cum_mean_lazy");
    try group_cume_dist_plan.withGroupCumulativeProduct("store", "sales", "store_sales_cum_product_lazy");
    try group_cume_dist_plan.withGroupCumulativeMin("store", "sales", "store_sales_cum_min_lazy");
    try group_cume_dist_plan.withGroupCumulativeMax("store", "sales", "store_sales_cum_max_lazy");
    try group_cume_dist_plan.withGroupCumulativeVariance("store", "sales", "store_sales_cum_variance_lazy");
    try group_cume_dist_plan.withGroupCumulativeStddev("store", "sales", "store_sales_cum_stddev_lazy");
    try group_cume_dist_plan.withGroupCumulativeSem("store", "sales", "store_sales_cum_sem_lazy");
    try group_cume_dist_plan.withGroupCumulativeCv("store", "sales", "store_sales_cum_cv_lazy");
    try group_cume_dist_plan.withGroupCumulativeFano("store", "sales", "store_sales_cum_fano_lazy");
    try group_cume_dist_plan.withGroupCumulativeSkewness("store", "sales", "store_sales_cum_skewness_lazy");
    try group_cume_dist_plan.withGroupCumulativeKurtosis("store", "sales", "store_sales_cum_kurtosis_lazy");
    try group_cume_dist_plan.withGroupCumulativeMeanAbs("store", "delta", "store_delta_cum_mean_abs_lazy");
    try group_cume_dist_plan.withGroupCumulativeMeanSquare("store", "delta", "store_delta_cum_mean_square_lazy");
    try group_cume_dist_plan.withGroupCumulativeRms("store", "delta", "store_delta_cum_rms_lazy");
    try group_cume_dist_plan.withGroupCumulativeMaxAbs("store", "delta", "store_delta_cum_max_abs_lazy");
    try group_cume_dist_plan.withGroupCumulativeMinAbs("store", "delta", "store_delta_cum_min_abs_lazy");
    try group_cume_dist_plan.withGroupCumulativeL1Norm("store", "delta", "store_delta_cum_l1_lazy");
    try group_cume_dist_plan.withGroupCumulativeL2Norm("store", "delta", "store_delta_cum_l2_lazy");
    try group_cume_dist_plan.withGroupCumulativeRange("store", "delta", "store_delta_cum_range_lazy");
    try group_cume_dist_plan.withGroupCumulativeMidrange("store", "delta", "store_delta_cum_midrange_lazy");
    try group_cume_dist_plan.withGroupCumulativeRangeCoeff("store", "delta", "store_delta_cum_range_coeff_lazy");
    try group_cume_dist_plan.withGroupCumulativeLogSumExp("store", "sales", "store_sales_cum_logsumexp_lazy");
    try group_cume_dist_plan.withGroupCumulativeLogMeanExp("store", "sales", "store_sales_cum_logmeanexp_lazy");
    try group_cume_dist_plan.withGroupCumulativeGeometricMean("store", "sales", "store_sales_cum_geometric_lazy");
    try group_cume_dist_plan.withGroupCumulativeHarmonicMean("store", "sales", "store_sales_cum_harmonic_lazy");
    try group_cume_dist_plan.withGroupCumulativeArgMin("store", "sales", "store_sales_cum_argmin_lazy");
    try group_cume_dist_plan.withGroupCumulativeArgMax("store", "sales", "store_sales_cum_argmax_lazy");
    try group_cume_dist_plan.withGroupCumulativeFirstValidIndex("store", "sales", "store_sales_cum_first_valid_index_lazy");
    try group_cume_dist_plan.withGroupCumulativeLastValidIndex("store", "sales", "store_sales_cum_last_valid_index_lazy");
    try group_cume_dist_plan.withGroupCumulativeFirstNullIndex("store", "sales", "store_sales_cum_first_null_index_lazy");
    try group_cume_dist_plan.withGroupCumulativeLastNullIndex("store", "sales", "store_sales_cum_last_null_index_lazy");
    const group_cume_dist_explained = try group_cume_dist_plan.explain(gpa);
    defer gpa.free(group_cume_dist_explained);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_cume_dist([store]->store_cume_dist_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_percent_rank([store]->store_percent_rank_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_reverse_cume_dist([store]->store_reverse_cume_dist_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_reverse_percent_rank([store]->store_reverse_percent_rank_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_lag([store], value=sales, offset=1->store_sales_lag_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_lead([store], value=sales, offset=1->store_sales_lead_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_first_row_value([store], value=sales->store_sales_first_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_last_row_value([store], value=sales->store_sales_last_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_nth_row_value([store], value=sales, n=1->store_sales_nth_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_first_valid_value([store], value=sales->store_sales_first_valid_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_last_valid_value([store], value=sales->store_sales_last_valid_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_nth_valid_value([store], value=sales, n=1->store_sales_nth_valid_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_fill_null_forward([store], value=sales->store_sales_ffill_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_fill_null_backward([store], value=sales->store_sales_bfill_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_cumulative_valid_count([store], value=sales->store_sales_cum_valid_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_cumulative_null_count([store], value=sales->store_sales_cum_null_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_cumulative_valid_ratio([store], value=sales->store_sales_cum_valid_ratio_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_cumulative_null_ratio([store], value=sales->store_sales_cum_null_ratio_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_cumulative_sum([store], value=sales->store_sales_cum_sum_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_cumulative_mean([store], value=sales->store_sales_cum_mean_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_cumulative_product([store], value=sales->store_sales_cum_product_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_cumulative_min([store], value=sales->store_sales_cum_min_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_cumulative_max([store], value=sales->store_sales_cum_max_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_cumulative_variance([store], value=sales->store_sales_cum_variance_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_cumulative_stddev([store], value=sales->store_sales_cum_stddev_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_cumulative_sem([store], value=sales->store_sales_cum_sem_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_cumulative_cv([store], value=sales->store_sales_cum_cv_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_cumulative_fano([store], value=sales->store_sales_cum_fano_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_cumulative_skewness([store], value=sales->store_sales_cum_skewness_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_cumulative_kurtosis([store], value=sales->store_sales_cum_kurtosis_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_cumulative_mean_abs([store], value=delta->store_delta_cum_mean_abs_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_cumulative_mean_square([store], value=delta->store_delta_cum_mean_square_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_cumulative_rms([store], value=delta->store_delta_cum_rms_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_cumulative_max_abs([store], value=delta->store_delta_cum_max_abs_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_cumulative_min_abs([store], value=delta->store_delta_cum_min_abs_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_cumulative_l1_norm([store], value=delta->store_delta_cum_l1_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_cumulative_l2_norm([store], value=delta->store_delta_cum_l2_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_cumulative_range([store], value=delta->store_delta_cum_range_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_cumulative_midrange([store], value=delta->store_delta_cum_midrange_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_cumulative_range_coeff([store], value=delta->store_delta_cum_range_coeff_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_cumulative_logsumexp([store], value=sales->store_sales_cum_logsumexp_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_cumulative_logmeanexp([store], value=sales->store_sales_cum_logmeanexp_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_cumulative_geometric_mean([store], value=sales->store_sales_cum_geometric_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_cumulative_harmonic_mean([store], value=sales->store_sales_cum_harmonic_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_cumulative_argmin([store], value=sales->store_sales_cum_argmin_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_cumulative_argmax([store], value=sales->store_sales_cum_argmax_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_cumulative_first_valid_index([store], value=sales->store_sales_cum_first_valid_index_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_cumulative_last_valid_index([store], value=sales->store_sales_cum_last_valid_index_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_cumulative_first_null_index([store], value=sales->store_sales_cum_first_null_index_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, group_cume_dist_explained, "group_cumulative_last_null_index([store], value=sales->store_sales_cum_last_null_index_lazy)") != null);
    var lazy_group_cume_dist = try group_cume_dist_plan.collect();
    defer lazy_group_cume_dist.deinit();
    try expectF64ColumnWithValidity(lazy_group_cume_dist, gpa, "store_cume_dist_lazy", &.{ 1.0 / 3.0, 0.5, 2.0 / 3.0, 0.0, 1.0, 1.0 }, &.{ true, true, true, false, true, true });
    try expectF64ColumnWithValidity(lazy_group_cume_dist, gpa, "store_percent_rank_lazy", &.{ 0.0, 0.0, 0.5, 0.0, 1.0, 1.0 }, &.{ true, true, true, false, true, true });
    try expectF64ColumnWithValidity(lazy_group_cume_dist, gpa, "store_reverse_cume_dist_lazy", &.{ 1.0, 1.0, 2.0 / 3.0, 0.0, 0.5, 1.0 / 3.0 }, &.{ true, true, true, false, true, true });
    try expectF64ColumnWithValidity(lazy_group_cume_dist, gpa, "store_reverse_percent_rank_lazy", &.{ 1.0, 1.0, 0.5, 0.0, 0.0, 0.0 }, &.{ true, true, true, false, true, true });
    try expectF64ColumnWithValidity(lazy_group_cume_dist, gpa, "store_sales_lag_lazy", &.{ 0.0, 0.0, 2.0, 0.0, 3.0, 5.0 }, &.{ false, false, true, false, true, false });
    try expectF64ColumnWithValidity(lazy_group_cume_dist, gpa, "store_sales_lead_lazy", &.{ 5.0, 11.0, 13.0, 0.0, 0.0, 0.0 }, &.{ false, true, true, false, false, false });
    try expectF64ColumnWithValidity(lazy_group_cume_dist, gpa, "store_sales_first_lazy", &.{ 2.0, 3.0, 2.0, 0.0, 3.0, 2.0 }, &.{ true, true, true, false, true, true });
    try expectF64ColumnWithValidity(lazy_group_cume_dist, gpa, "store_sales_last_lazy", &.{ 13.0, 11.0, 13.0, 0.0, 11.0, 13.0 }, &.{ true, true, true, false, true, true });
    try expectF64ColumnWithValidity(lazy_group_cume_dist, gpa, "store_sales_nth_lazy", &.{ 5.0, 11.0, 5.0, 0.0, 11.0, 5.0 }, &.{ false, true, false, false, true, false });
    try expectF64ColumnWithValidity(lazy_group_cume_dist, gpa, "store_sales_first_valid_lazy", &.{ 2.0, 3.0, 2.0, 0.0, 3.0, 2.0 }, &.{ true, true, true, false, true, true });
    try expectF64ColumnWithValidity(lazy_group_cume_dist, gpa, "store_sales_last_valid_lazy", &.{ 13.0, 11.0, 13.0, 0.0, 11.0, 13.0 }, &.{ true, true, true, false, true, true });
    try expectF64ColumnWithValidity(lazy_group_cume_dist, gpa, "store_sales_nth_valid_lazy", &.{ 13.0, 11.0, 13.0, 0.0, 11.0, 13.0 }, &.{ true, true, true, false, true, true });
    try expectF64ColumnWithValidity(lazy_group_cume_dist, gpa, "store_sales_ffill_lazy", &.{ 2.0, 3.0, 2.0, 0.0, 11.0, 13.0 }, &.{ true, true, true, false, true, true });
    try expectF64ColumnWithValidity(lazy_group_cume_dist, gpa, "store_sales_bfill_lazy", &.{ 2.0, 3.0, 13.0, 0.0, 11.0, 13.0 }, &.{ true, true, true, false, true, true });
    try expectNullableI64Column(lazy_group_cume_dist, gpa, "store_sales_cum_valid_lazy", &.{ 1, 1, 1, 0, 2, 2 }, &.{ true, true, true, false, true, true });
    try expectNullableI64Column(lazy_group_cume_dist, gpa, "store_sales_cum_null_lazy", &.{ 0, 0, 1, 0, 0, 1 }, &.{ true, true, true, false, true, true });
    try expectF64ColumnWithValidity(lazy_group_cume_dist, gpa, "store_sales_cum_valid_ratio_lazy", &.{ 1.0, 1.0, 0.5, 0.0, 1.0, 2.0 / 3.0 }, &.{ true, true, true, false, true, true });
    try expectF64ColumnWithValidity(lazy_group_cume_dist, gpa, "store_sales_cum_null_ratio_lazy", &.{ 0.0, 0.0, 0.5, 0.0, 0.0, 1.0 / 3.0 }, &.{ true, true, true, false, true, true });
    try expectF64ColumnWithValidity(lazy_group_cume_dist, gpa, "store_sales_cum_sum_lazy", &.{ 2.0, 3.0, 0.0, 0.0, 14.0, 15.0 }, &.{ true, true, false, false, true, true });
    try expectF64ColumnWithValidity(lazy_group_cume_dist, gpa, "store_sales_cum_mean_lazy", &.{ 2.0, 3.0, 0.0, 0.0, 7.0, 7.5 }, &.{ true, true, false, false, true, true });
    try expectF64ColumnWithValidity(lazy_group_cume_dist, gpa, "store_sales_cum_product_lazy", &.{ 2.0, 3.0, 0.0, 0.0, 33.0, 26.0 }, &.{ true, true, false, false, true, true });
    try expectF64ColumnWithValidity(lazy_group_cume_dist, gpa, "store_sales_cum_min_lazy", &.{ 2.0, 3.0, 0.0, 0.0, 3.0, 2.0 }, &.{ true, true, false, false, true, true });
    try expectF64ColumnWithValidity(lazy_group_cume_dist, gpa, "store_sales_cum_max_lazy", &.{ 2.0, 3.0, 0.0, 0.0, 11.0, 13.0 }, &.{ true, true, false, false, true, true });
    try expectF64ColumnWithValidity(lazy_group_cume_dist, gpa, "store_sales_cum_variance_lazy", &.{ 0.0, 0.0, 0.0, 0.0, 16.0, 30.25 }, &.{ true, true, false, false, true, true });
    try expectF64ColumnWithValidity(lazy_group_cume_dist, gpa, "store_sales_cum_stddev_lazy", &.{ 0.0, 0.0, 0.0, 0.0, 4.0, 5.5 }, &.{ true, true, false, false, true, true });
    try expectF64ColumnApproxOrNanWithValidity(lazy_group_cume_dist, gpa, "store_sales_cum_sem_lazy", &.{ 0.0, 0.0, 0.0, 0.0, 4.0 / std.math.sqrt(@as(f64, 2.0)), 5.5 / std.math.sqrt(@as(f64, 2.0)) }, &.{ true, true, false, false, true, true });
    try expectF64ColumnApproxOrNanWithValidity(lazy_group_cume_dist, gpa, "store_sales_cum_cv_lazy", &.{ 0.0, 0.0, 0.0, 0.0, 4.0 / 7.0, 11.0 / 15.0 }, &.{ true, true, false, false, true, true });
    try expectF64ColumnApproxOrNanWithValidity(lazy_group_cume_dist, gpa, "store_sales_cum_fano_lazy", &.{ 0.0, 0.0, 0.0, 0.0, 16.0 / 7.0, 121.0 / 30.0 }, &.{ true, true, false, false, true, true });
    try expectF64ColumnApproxOrNanWithValidity(lazy_group_cume_dist, gpa, "store_sales_cum_skewness_lazy", &.{ std.math.nan(f64), std.math.nan(f64), 0.0, 0.0, 0.0, 0.0 }, &.{ true, true, false, false, true, true });
    try expectF64ColumnApproxOrNanWithValidity(lazy_group_cume_dist, gpa, "store_sales_cum_kurtosis_lazy", &.{ std.math.nan(f64), std.math.nan(f64), 0.0, 0.0, -2.0, -2.0 }, &.{ true, true, false, false, true, true });
    try expectF64ColumnWithValidity(lazy_group_cume_dist, gpa, "store_delta_cum_mean_abs_lazy", &.{ 2.0, 3.0, 0.0, 0.0, 7.0, 7.5 }, &.{ true, true, false, false, true, true });
    try expectF64ColumnWithValidity(lazy_group_cume_dist, gpa, "store_delta_cum_mean_square_lazy", &.{ 4.0, 9.0, 0.0, 0.0, 65.0, 86.5 }, &.{ true, true, false, false, true, true });
    try expectF64ColumnApproxOrNanWithValidity(lazy_group_cume_dist, gpa, "store_delta_cum_rms_lazy", &.{ 2.0, 3.0, 0.0, 0.0, std.math.sqrt(@as(f64, 65.0)), std.math.sqrt(@as(f64, 86.5)) }, &.{ true, true, false, false, true, true });
    try expectF64ColumnWithValidity(lazy_group_cume_dist, gpa, "store_delta_cum_max_abs_lazy", &.{ 2.0, 3.0, 0.0, 0.0, 11.0, 13.0 }, &.{ true, true, false, false, true, true });
    try expectF64ColumnWithValidity(lazy_group_cume_dist, gpa, "store_delta_cum_min_abs_lazy", &.{ 2.0, 3.0, 0.0, 0.0, 3.0, 2.0 }, &.{ true, true, false, false, true, true });
    try expectF64ColumnWithValidity(lazy_group_cume_dist, gpa, "store_delta_cum_l1_lazy", &.{ 2.0, 3.0, 0.0, 0.0, 14.0, 15.0 }, &.{ true, true, false, false, true, true });
    try expectF64ColumnApproxOrNanWithValidity(lazy_group_cume_dist, gpa, "store_delta_cum_l2_lazy", &.{ 2.0, 3.0, 0.0, 0.0, std.math.sqrt(@as(f64, 130.0)), std.math.sqrt(@as(f64, 173.0)) }, &.{ true, true, false, false, true, true });
    try expectF64ColumnWithValidity(lazy_group_cume_dist, gpa, "store_delta_cum_range_lazy", &.{ 0.0, 0.0, 0.0, 0.0, 8.0, 15.0 }, &.{ true, true, false, false, true, true });
    try expectF64ColumnWithValidity(lazy_group_cume_dist, gpa, "store_delta_cum_midrange_lazy", &.{ -2.0, -3.0, 0.0, 0.0, -7.0, 5.5 }, &.{ true, true, false, false, true, true });
    try expectF64ColumnApproxOrNanWithValidity(lazy_group_cume_dist, gpa, "store_delta_cum_range_coeff_lazy", &.{ -0.0, -0.0, 0.0, 0.0, -8.0 / 14.0, 15.0 / 11.0 }, &.{ true, true, false, false, true, true });
    try expectF64ColumnApproxOrNanWithValidity(lazy_group_cume_dist, gpa, "store_sales_cum_logsumexp_lazy", &.{ 2.0, 3.0, 0.0, 0.0, 11.0 + std.math.log(f64, std.math.e, 1.0 + std.math.exp(-8.0)), 13.0 + std.math.log(f64, std.math.e, 1.0 + std.math.exp(-11.0)) }, &.{ true, true, false, false, true, true });
    try expectF64ColumnApproxOrNanWithValidity(lazy_group_cume_dist, gpa, "store_sales_cum_logmeanexp_lazy", &.{ 2.0, 3.0, 0.0, 0.0, 11.0 + std.math.log(f64, std.math.e, 1.0 + std.math.exp(-8.0)) - std.math.log(f64, std.math.e, 2.0), 13.0 + std.math.log(f64, std.math.e, 1.0 + std.math.exp(-11.0)) - std.math.log(f64, std.math.e, 2.0) }, &.{ true, true, false, false, true, true });
    try expectF64ColumnApproxOrNanWithValidity(lazy_group_cume_dist, gpa, "store_sales_cum_geometric_lazy", &.{ 2.0, 3.0, 0.0, 0.0, std.math.sqrt(@as(f64, 33.0)), std.math.sqrt(@as(f64, 26.0)) }, &.{ true, true, false, false, true, true });
    try expectF64ColumnApproxOrNanWithValidity(lazy_group_cume_dist, gpa, "store_sales_cum_harmonic_lazy", &.{ 2.0, 3.0, 0.0, 0.0, 33.0 / 7.0, 52.0 / 15.0 }, &.{ true, true, false, false, true, true });
    try expectNullableI64Column(lazy_group_cume_dist, gpa, "store_sales_cum_argmin_lazy", &.{ 0, 1, 0, 0, 1, 0 }, &.{ true, true, false, false, true, true });
    try expectNullableI64Column(lazy_group_cume_dist, gpa, "store_sales_cum_argmax_lazy", &.{ 0, 1, 0, 0, 4, 5 }, &.{ true, true, false, false, true, true });
    try expectNullableI64Column(lazy_group_cume_dist, gpa, "store_sales_cum_first_valid_index_lazy", &.{ 0, 1, 0, 0, 1, 0 }, &.{ true, true, true, false, true, true });
    try expectNullableI64Column(lazy_group_cume_dist, gpa, "store_sales_cum_last_valid_index_lazy", &.{ 0, 1, 0, 0, 4, 5 }, &.{ true, true, true, false, true, true });
    try expectNullableI64Column(lazy_group_cume_dist, gpa, "store_sales_cum_first_null_index_lazy", &.{ 0, 0, 2, 0, 0, 2 }, &.{ false, false, true, false, false, true });
    try expectNullableI64Column(lazy_group_cume_dist, gpa, "store_sales_cum_last_null_index_lazy", &.{ 0, 0, 2, 0, 0, 2 }, &.{ false, false, true, false, false, true });

    var group_row_number_plan = try DeviceLazyFrame.init(gpa, table);
    defer group_row_number_plan.deinit();
    try group_row_number_plan.withGroupRowNumber("store", "store_row_number_lazy");
    const group_row_number_explained = try group_row_number_plan.explain(gpa);
    defer gpa.free(group_row_number_explained);
    try std.testing.expect(std.mem.indexOf(u8, group_row_number_explained, "group_row_number([store]->store_row_number_lazy)") != null);
    var lazy_group_row_numbers = try group_row_number_plan.collect();
    defer lazy_group_row_numbers.deinit();
    try expectNullableI64Column(lazy_group_row_numbers, gpa, "store_row_number_lazy", &.{ 0, 0, 1, 0, 1, 2 }, &.{ true, true, true, false, true, true });

    var group_size_plan = try DeviceLazyFrame.init(gpa, table);
    defer group_size_plan.deinit();
    try group_size_plan.withGroupSize("store", "store_group_size_lazy");
    const group_size_explained = try group_size_plan.explain(gpa);
    defer gpa.free(group_size_explained);
    try std.testing.expect(std.mem.indexOf(u8, group_size_explained, "group_size([store]->store_group_size_lazy)") != null);
    var lazy_group_sizes = try group_size_plan.collect();
    defer lazy_group_sizes.deinit();
    try expectNullableI64Column(lazy_group_sizes, gpa, "store_group_size_lazy", &.{ 3, 2, 3, 0, 2, 3 }, &.{ true, true, true, false, true, true });

    var group_reverse_row_plan = try DeviceLazyFrame.init(gpa, table);
    defer group_reverse_row_plan.deinit();
    try group_reverse_row_plan.withGroupReverseRowNumber("store", "store_reverse_row_number_lazy");
    const group_reverse_row_explained = try group_reverse_row_plan.explain(gpa);
    defer gpa.free(group_reverse_row_explained);
    try std.testing.expect(std.mem.indexOf(u8, group_reverse_row_explained, "group_reverse_row_number([store]->store_reverse_row_number_lazy)") != null);
    var lazy_group_reverse_rows = try group_reverse_row_plan.collect();
    defer lazy_group_reverse_rows.deinit();
    try expectNullableI64Column(lazy_group_reverse_rows, gpa, "store_reverse_row_number_lazy", &.{ 2, 1, 1, 0, 0, 0 }, &.{ true, true, true, false, true, true });

    var tail_rows_plan = try DeviceLazyFrame.init(gpa, table);
    defer tail_rows_plan.deinit();
    try tail_rows_plan.groupByTailRows("store", 1);
    const tail_rows_explained = try tail_rows_plan.explain(gpa);
    defer gpa.free(tail_rows_explained);
    try std.testing.expect(std.mem.indexOf(u8, tail_rows_explained, "group_by_tail_rows(store, n=1)") != null);
    var lazy_tail_rows = try tail_rows_plan.collect();
    defer lazy_tail_rows.deinit();
    const lazy_tail_keys = try (try lazy_tail_rows.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(lazy_tail_keys);
    const lazy_tail_sales = try (try lazy_tail_rows.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_tail_sales);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2 }, lazy_tail_keys);
    try std.testing.expectEqualSlices(f64, &.{ 13.0, 11.0 }, lazy_tail_sales);

    var slice_rows_plan = try DeviceLazyFrame.init(gpa, table);
    defer slice_rows_plan.deinit();
    try slice_rows_plan.groupBySliceRows("store", 1, 2);
    const slice_rows_explained = try slice_rows_plan.explain(gpa);
    defer gpa.free(slice_rows_explained);
    try std.testing.expect(std.mem.indexOf(u8, slice_rows_explained, "group_by_slice_rows(store, start=1, length=2)") != null);
    var lazy_slice_rows = try slice_rows_plan.collect();
    defer lazy_slice_rows.deinit();
    const lazy_slice_keys = try (try lazy_slice_rows.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(lazy_slice_keys);
    const lazy_slice_sales = try (try lazy_slice_rows.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_slice_sales);
    const lazy_slice_sales_validity = try (try lazy_slice_rows.column("sales")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_slice_sales_validity);
    try std.testing.expectEqualSlices(i32, &.{ 1, 1, 2 }, lazy_slice_keys);
    try std.testing.expectEqualSlices(f64, &.{ 5.0, 13.0, 11.0 }, lazy_slice_sales);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true }, lazy_slice_sales_validity);

    var stepped_slice_plan = try DeviceLazyFrame.init(gpa, table);
    defer stepped_slice_plan.deinit();
    try stepped_slice_plan.groupBySliceRowsStep("store", 0, 3, 2);
    const stepped_slice_explained = try stepped_slice_plan.explain(gpa);
    defer gpa.free(stepped_slice_explained);
    try std.testing.expect(std.mem.indexOf(u8, stepped_slice_explained, "group_by_slice_rows_step(store, start=0, length=3, step=2)") != null);
    var lazy_stepped_slice = try stepped_slice_plan.collect();
    defer lazy_stepped_slice.deinit();
    const lazy_stepped_keys = try (try lazy_stepped_slice.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(lazy_stepped_keys);
    const lazy_stepped_sales = try (try lazy_stepped_slice.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_stepped_sales);
    try std.testing.expectEqualSlices(i32, &.{ 1, 1, 2 }, lazy_stepped_keys);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 13.0, 3.0 }, lazy_stepped_sales);

    var signed_slice_plan = try DeviceLazyFrame.init(gpa, table);
    defer signed_slice_plan.deinit();
    try signed_slice_plan.groupBySliceRowsSigned("store", -2, 2);
    const signed_slice_explained = try signed_slice_plan.explain(gpa);
    defer gpa.free(signed_slice_explained);
    try std.testing.expect(std.mem.indexOf(u8, signed_slice_explained, "group_by_slice_rows_signed_step(store, start=-2, length=2, step=1)") != null);
    var lazy_signed_slice = try signed_slice_plan.collect();
    defer lazy_signed_slice.deinit();
    const lazy_signed_keys = try (try lazy_signed_slice.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(lazy_signed_keys);
    try std.testing.expectEqualSlices(i32, &.{ 1, 1, 2, 2 }, lazy_signed_keys);
    try expectF64ColumnWithValidity(lazy_signed_slice, gpa, "sales", &.{ 5.0, 13.0, 3.0, 11.0 }, &.{ false, true, true, true });

    var top_rows_plan = try DeviceLazyFrame.init(gpa, table);
    defer top_rows_plan.deinit();
    try top_rows_plan.groupByTopRows("store", "sales", 2, .{ .descending = true, .nulls = .last });
    const top_rows_explained = try top_rows_plan.explain(gpa);
    defer gpa.free(top_rows_explained);
    try std.testing.expect(std.mem.indexOf(u8, top_rows_explained, "group_by_top_rows(store, sort=sales, n=2, desc=true)") != null);
    var lazy_top_rows = try top_rows_plan.collect();
    defer lazy_top_rows.deinit();
    const lazy_top_keys = try (try lazy_top_rows.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(lazy_top_keys);
    const lazy_top_sales = try (try lazy_top_rows.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_top_sales);
    try std.testing.expectEqualSlices(i32, &.{ 1, 1, 2, 2 }, lazy_top_keys);
    try std.testing.expectEqualSlices(f64, &.{ 13.0, 2.0, 11.0, 3.0 }, lazy_top_sales);

    var top_sorted_plan = try DeviceLazyFrame.init(gpa, sorted_row_table);
    defer top_sorted_plan.deinit();
    try top_sorted_plan.groupByTopRowsByColumns("bucket", &.{ "score", "tie" }, 2, &sorted_row_options);
    const top_sorted_explained = try top_sorted_plan.explain(gpa);
    defer gpa.free(top_sorted_explained);
    try std.testing.expect(std.mem.indexOf(u8, top_sorted_explained, "group_by_top_rows_by_columns(bucket, sort=[score,tie], n=2)") != null);
    var lazy_top_sorted = try top_sorted_plan.collect();
    defer lazy_top_sorted.deinit();
    const lazy_top_sorted_ids = try (try lazy_top_sorted.column("id")).i32.toOwnedSlice(gpa);
    defer gpa.free(lazy_top_sorted_ids);
    try std.testing.expectEqualSlices(i32, &.{ 11, 10, 22, 21 }, lazy_top_sorted_ids);

    var summed = try table.groupBySum("store", "sales", "sales_sum");
    defer summed.deinit();
    try std.testing.expectEqual(@as(usize, 2), summed.height());
    const sum_keys = try (try summed.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(sum_keys);
    const sums = try (try summed.column("sales_sum")).f64.toOwnedSlice(gpa);
    defer gpa.free(sums);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2 }, sum_keys);
    try std.testing.expectEqualSlices(f64, &.{ 15.0, 14.0 }, sums);

    var producted = try table.groupByProd("store", "sales", "sales_prod");
    defer producted.deinit();
    const products = try (try producted.column("sales_prod")).f64.toOwnedSlice(gpa);
    defer gpa.free(products);
    try std.testing.expectEqualSlices(f64, &.{ 26.0, 33.0 }, products);

    var mins = try table.groupByMin("store", "sales", "sales_min");
    defer mins.deinit();
    const min_values = try (try mins.column("sales_min")).f64.toOwnedSlice(gpa);
    defer gpa.free(min_values);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 3.0 }, min_values);

    var maxes = try table.groupByMax("store", "sales", "sales_max");
    defer maxes.deinit();
    const max_values = try (try maxes.column("sales_max")).f64.toOwnedSlice(gpa);
    defer gpa.free(max_values);
    try std.testing.expectEqualSlices(f64, &.{ 13.0, 11.0 }, max_values);

    var means = try table.groupByMean("store", "sales", "sales_mean");
    defer means.deinit();
    const mean_values = try (try means.column("sales_mean")).f64.toOwnedSlice(gpa);
    defer gpa.free(mean_values);
    try std.testing.expectEqualSlices(f64, &.{ 7.5, 7.0 }, mean_values);

    var first_sales = try table.groupByFirst("store", "sales", "sales_first");
    defer first_sales.deinit();
    const first_sales_values = try (try first_sales.column("sales_first")).f64.toOwnedSlice(gpa);
    defer gpa.free(first_sales_values);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 3.0 }, first_sales_values);

    var last_sales = try table.groupByLast("store", "sales", "sales_last");
    defer last_sales.deinit();
    const last_sales_values = try (try last_sales.column("sales_last")).f64.toOwnedSlice(gpa);
    defer gpa.free(last_sales_values);
    try std.testing.expectEqualSlices(f64, &.{ 13.0, 11.0 }, last_sales_values);

    var first_sales_rows = try table.groupByFirstRow("store", "sales", "sales_first_row");
    defer first_sales_rows.deinit();
    try expectF64ColumnWithValidity(first_sales_rows, gpa, "sales_first_row", &.{ 2.0, 3.0 }, &.{ true, true });

    var last_sales_rows = try table.groupByLastRow("store", "sales", "sales_last_row");
    defer last_sales_rows.deinit();
    try expectF64ColumnWithValidity(last_sales_rows, gpa, "sales_last_row", &.{ 13.0, 11.0 }, &.{ true, true });

    var row_key = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 1, 1, 2, 2, 2 }, .cpu);
    defer row_key.deinit();
    var row_values = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 10.0, 11.0, 12.0, 20.0, 21.0, 22.0 }, &.{ false, true, true, true, true, false }, .cpu);
    defer row_values.deinit();
    var row_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "bucket", .data = row_key },
        .{ .name = "metric", .data = row_values },
    });
    defer row_table.deinit();

    var first_valid_metrics = try row_table.groupByFirst("bucket", "metric", "metric_first_valid");
    defer first_valid_metrics.deinit();
    try expectF64ColumnApproxOrNan(first_valid_metrics, gpa, "metric_first_valid", &.{ 11.0, 20.0 });

    var first_row_metrics = try row_table.groupByFirstRow("bucket", "metric", "metric_first_row");
    defer first_row_metrics.deinit();
    try expectF64ColumnWithValidity(first_row_metrics, gpa, "metric_first_row", &.{ 10.0, 20.0 }, &.{ false, true });

    var last_valid_metrics = try row_table.groupByLast("bucket", "metric", "metric_last_valid");
    defer last_valid_metrics.deinit();
    try expectF64ColumnApproxOrNan(last_valid_metrics, gpa, "metric_last_valid", &.{ 12.0, 21.0 });

    var last_row_metrics = try row_table.groupByLastRow("bucket", "metric", "metric_last_row");
    defer last_row_metrics.deinit();
    try expectF64ColumnWithValidity(last_row_metrics, gpa, "metric_last_row", &.{ 12.0, 22.0 }, &.{ true, false });

    var nth_valid_metrics = try row_table.groupByNth("bucket", "metric", "metric_nth_valid", 1);
    defer nth_valid_metrics.deinit();
    try expectF64ColumnWithValidity(nth_valid_metrics, gpa, "metric_nth_valid", &.{ 12.0, 21.0 }, &.{ true, true });

    var nth_row_metrics = try row_table.groupByNthRow("bucket", "metric", "metric_nth_row", 1);
    defer nth_row_metrics.deinit();
    try expectF64ColumnWithValidity(nth_row_metrics, gpa, "metric_nth_row", &.{ 11.0, 21.0 }, &.{ true, true });

    var nth_oob_metrics = try row_table.groupByNth("bucket", "metric", "metric_nth_oob", 2);
    defer nth_oob_metrics.deinit();
    try expectF64ColumnWithValidity(nth_oob_metrics, gpa, "metric_nth_oob", &.{ 0.0, 0.0 }, &.{ false, false });

    var nth_valid_indices = try row_table.groupByNthIndex("bucket", "metric", "metric_nth_valid_index", 1);
    defer nth_valid_indices.deinit();
    try expectNullableI64Column(nth_valid_indices, gpa, "metric_nth_valid_index", &.{ 2, 4 }, &.{ true, true });

    var nth_row_indices = try row_table.groupByNthRowIndex("bucket", "metric", "metric_nth_row_index", 1);
    defer nth_row_indices.deinit();
    try expectNullableI64Column(nth_row_indices, gpa, "metric_nth_row_index", &.{ 1, 4 }, &.{ true, true });

    var nth_index_oob = try row_table.groupByNthIndex("bucket", "metric", "metric_nth_index_oob", 2);
    defer nth_index_oob.deinit();
    try expectNullableI64Column(nth_index_oob, gpa, "metric_nth_index_oob", &.{ 0, 0 }, &.{ false, false });

    var last_row_plan = try DeviceLazyFrame.init(gpa, row_table);
    defer last_row_plan.deinit();
    try last_row_plan.groupByLastRow("bucket", "metric", "metric_last_row_lazy");
    const last_row_explained = try last_row_plan.explain(gpa);
    defer gpa.free(last_row_explained);
    try std.testing.expect(std.mem.indexOf(u8, last_row_explained, "group_by_last_row(bucket, value=metric -> metric_last_row_lazy)") != null);
    var lazy_last_row = try last_row_plan.collect();
    defer lazy_last_row.deinit();
    try expectF64ColumnWithValidity(lazy_last_row, gpa, "metric_last_row_lazy", &.{ 12.0, 22.0 }, &.{ true, false });

    var nth_row_plan = try DeviceLazyFrame.init(gpa, row_table);
    defer nth_row_plan.deinit();
    try nth_row_plan.groupByNthRow("bucket", "metric", "metric_nth_row_lazy", 1);
    const nth_row_explained = try nth_row_plan.explain(gpa);
    defer gpa.free(nth_row_explained);
    try std.testing.expect(std.mem.indexOf(u8, nth_row_explained, "group_by_nth_row(bucket, value=metric, n=1 -> metric_nth_row_lazy)") != null);
    var lazy_nth_row = try nth_row_plan.collect();
    defer lazy_nth_row.deinit();
    try expectF64ColumnWithValidity(lazy_nth_row, gpa, "metric_nth_row_lazy", &.{ 11.0, 21.0 }, &.{ true, true });

    var nth_row_index_plan = try DeviceLazyFrame.init(gpa, row_table);
    defer nth_row_index_plan.deinit();
    try nth_row_index_plan.groupByNthRowIndex("bucket", "metric", "metric_nth_row_index_lazy", 1);
    const nth_row_index_explained = try nth_row_index_plan.explain(gpa);
    defer gpa.free(nth_row_index_explained);
    try std.testing.expect(std.mem.indexOf(u8, nth_row_index_explained, "group_by_nth_row_index(bucket, value=metric, n=1 -> metric_nth_row_index_lazy)") != null);
    var lazy_nth_row_index = try nth_row_index_plan.collect();
    defer lazy_nth_row_index.deinit();
    try expectNullableI64Column(lazy_nth_row_index, gpa, "metric_nth_row_index_lazy", &.{ 1, 4 }, &.{ true, true });

    var unique_sales = try table.groupByNUnique("store", "sales", "sales_n_unique");
    defer unique_sales.deinit();
    const unique_sales_values = try (try unique_sales.column("sales_n_unique")).i64.toOwnedSlice(gpa);
    defer gpa.free(unique_sales_values);
    try std.testing.expectEqualSlices(i64, &.{ 2, 2 }, unique_sales_values);

    var modal_sales = try table.groupByMode("store", "sales", "sales_mode");
    defer modal_sales.deinit();
    const modal_sales_values = try (try modal_sales.column("sales_mode")).f64.toOwnedSlice(gpa);
    defer gpa.free(modal_sales_values);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 3.0 }, modal_sales_values);

    var mode_diag_key = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 1, 1, 1, 2, 2, 2, 2, 2 }, .cpu);
    defer mode_diag_key.deinit();
    var mode_diag_value = try DeviceColumn.fromSlice(i32, gpa, &.{ 5, 5, 7, 8, 1, 1, 2, 2, 3 }, .cpu);
    defer mode_diag_value.deinit();
    var mode_diag_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "bucket", .data = mode_diag_key },
        .{ .name = "label", .data = mode_diag_value },
    });
    defer mode_diag_table.deinit();

    var mode_diag_mode = try mode_diag_table.groupByMode("bucket", "label", "label_mode");
    defer mode_diag_mode.deinit();
    const mode_diag_mode_values = try (try mode_diag_mode.column("label_mode")).i32.toOwnedSlice(gpa);
    defer gpa.free(mode_diag_mode_values);
    try std.testing.expectEqualSlices(i32, &.{ 5, 1 }, mode_diag_mode_values);

    var mode_count = try mode_diag_table.groupByModeCount("bucket", "label", "label_mode_count");
    defer mode_count.deinit();
    const mode_count_values = try (try mode_count.column("label_mode_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(mode_count_values);
    try std.testing.expectEqualSlices(i64, &.{ 2, 2 }, mode_count_values);

    var mode_ratio = try mode_diag_table.groupByModeRatioOn(&.{"bucket"}, "label", "label_mode_ratio");
    defer mode_ratio.deinit();
    const mode_ratio_values = try (try mode_ratio.column("label_mode_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(mode_ratio_values);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), mode_ratio_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.4), mode_ratio_values[1], 1e-12);

    var mode_margin = try mode_diag_table.groupByModeMargin("bucket", "label", "label_mode_margin");
    defer mode_margin.deinit();
    const mode_margin_values = try (try mode_margin.column("label_mode_margin")).i64.toOwnedSlice(gpa);
    defer gpa.free(mode_margin_values);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0 }, mode_margin_values);

    var mode_margin_ratio = try mode_diag_table.groupByModeMarginRatioOn(&.{"bucket"}, "label", "label_mode_margin_ratio");
    defer mode_margin_ratio.deinit();
    const mode_margin_ratio_values = try (try mode_margin_ratio.column("label_mode_margin_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(mode_margin_ratio_values);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), mode_margin_ratio_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), mode_margin_ratio_values[1], 1e-12);

    const group_entropy_1 = -(@as(f64, 0.5) * std.math.log(f64, std.math.e, @as(f64, 0.5)) + 2.0 * @as(f64, 0.25) * std.math.log(f64, std.math.e, @as(f64, 0.25)));
    const group_entropy_2 = -(2.0 * @as(f64, 0.4) * std.math.log(f64, std.math.e, @as(f64, 0.4)) + @as(f64, 0.2) * std.math.log(f64, std.math.e, @as(f64, 0.2)));
    const group_concentration_1 = @as(f64, 0.5) * @as(f64, 0.5) + 2.0 * @as(f64, 0.25) * @as(f64, 0.25);
    const group_concentration_2 = 2.0 * @as(f64, 0.4) * @as(f64, 0.4) + @as(f64, 0.2) * @as(f64, 0.2);
    const group_distinct_log = std.math.log(f64, std.math.e, @as(f64, 3.0));

    var entropy_group = try mode_diag_table.groupByEntropy("bucket", "label", "label_entropy");
    defer entropy_group.deinit();
    const entropy_group_values = try (try entropy_group.column("label_entropy")).f64.toOwnedSlice(gpa);
    defer gpa.free(entropy_group_values);
    try std.testing.expectApproxEqAbs(group_entropy_1, entropy_group_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(group_entropy_2, entropy_group_values[1], 1e-12);

    var gini_group = try mode_diag_table.groupByGiniOn(&.{"bucket"}, "label", "label_gini");
    defer gini_group.deinit();
    const gini_group_values = try (try gini_group.column("label_gini")).f64.toOwnedSlice(gpa);
    defer gpa.free(gini_group_values);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0) - group_concentration_1, gini_group_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0) - group_concentration_2, gini_group_values[1], 1e-12);

    var perplexity_group = try mode_diag_table.groupByPerplexity("bucket", "label", "label_perplexity");
    defer perplexity_group.deinit();
    const perplexity_group_values = try (try perplexity_group.column("label_perplexity")).f64.toOwnedSlice(gpa);
    defer gpa.free(perplexity_group_values);
    try std.testing.expectApproxEqAbs(std.math.exp(group_entropy_1), perplexity_group_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.exp(group_entropy_2), perplexity_group_values[1], 1e-12);

    var inverse_group = try mode_diag_table.groupByInverseSimpsonOn(&.{"bucket"}, "label", "label_inverse");
    defer inverse_group.deinit();
    const inverse_group_values = try (try inverse_group.column("label_inverse")).f64.toOwnedSlice(gpa);
    defer gpa.free(inverse_group_values);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0) / group_concentration_1, inverse_group_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0) / group_concentration_2, inverse_group_values[1], 1e-12);

    var concentration_group = try mode_diag_table.groupByConcentration("bucket", "label", "label_concentration");
    defer concentration_group.deinit();
    const concentration_group_values = try (try concentration_group.column("label_concentration")).f64.toOwnedSlice(gpa);
    defer gpa.free(concentration_group_values);
    try std.testing.expectApproxEqAbs(group_concentration_1, concentration_group_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(group_concentration_2, concentration_group_values[1], 1e-12);

    var evenness_group = try mode_diag_table.groupByEvennessOn(&.{"bucket"}, "label", "label_evenness");
    defer evenness_group.deinit();
    const evenness_group_values = try (try evenness_group.column("label_evenness")).f64.toOwnedSlice(gpa);
    defer gpa.free(evenness_group_values);
    try std.testing.expectApproxEqAbs(group_entropy_1 / group_distinct_log, evenness_group_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(group_entropy_2 / group_distinct_log, evenness_group_values[1], 1e-12);

    var gini_mean_diff_group = try mode_diag_table.groupByGiniMeanDiff("bucket", "label", "label_gini_mean_diff");
    defer gini_mean_diff_group.deinit();
    const gini_mean_diff_values = try (try gini_mean_diff_group.column("label_gini_mean_diff")).f64.toOwnedSlice(gpa);
    defer gpa.free(gini_mean_diff_values);
    try std.testing.expectApproxEqAbs(@as(f64, 11.0 / 6.0), gini_mean_diff_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), gini_mean_diff_values[1], 1e-12);

    var gini_coeff_group = try mode_diag_table.groupByGiniCoeffOn(&.{"bucket"}, "label", "label_gini_coeff");
    defer gini_coeff_group.deinit();
    const gini_coeff_values = try (try gini_coeff_group.column("label_gini_coeff")).f64.toOwnedSlice(gpa);
    defer gpa.free(gini_coeff_values);
    try std.testing.expectApproxEqAbs(@as(f64, 11.0 / 75.0), gini_coeff_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 18.0), gini_coeff_values[1], 1e-12);

    var mean_abs_dev_group = try mode_diag_table.groupByMeanAbsDev("bucket", "label", "label_mean_abs_dev");
    defer mean_abs_dev_group.deinit();
    const mean_abs_dev_values = try (try mean_abs_dev_group.column("label_mean_abs_dev")).f64.toOwnedSlice(gpa);
    defer gpa.free(mean_abs_dev_values);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 4.0), mean_abs_dev_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 16.0 / 25.0), mean_abs_dev_values[1], 1e-12);

    var mean_abs_dev_ratio_group = try mode_diag_table.groupByMeanAbsDevRatioOn(&.{"bucket"}, "label", "label_mean_abs_dev_ratio");
    defer mean_abs_dev_ratio_group.deinit();
    const mean_abs_dev_ratio_values = try (try mean_abs_dev_ratio_group.column("label_mean_abs_dev_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(mean_abs_dev_ratio_values);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 5.0), mean_abs_dev_ratio_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 16.0 / 45.0), mean_abs_dev_ratio_values[1], 1e-12);

    var zero_gini_key = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 1 }, .cpu);
    defer zero_gini_key.deinit();
    var zero_gini_value = try DeviceColumn.fromSlice(f64, gpa, &.{ -1.0, 1.0 }, .cpu);
    defer zero_gini_value.deinit();
    var zero_gini_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "bucket", .data = zero_gini_key },
        .{ .name = "value", .data = zero_gini_value },
    });
    defer zero_gini_table.deinit();
    var zero_gini_coeff = try zero_gini_table.groupByGiniCoefficient("bucket", "value", "value_gini_coeff");
    defer zero_gini_coeff.deinit();
    const zero_gini_coeff_values = try (try zero_gini_coeff.column("value_gini_coeff")).f64.toOwnedSlice(gpa);
    defer gpa.free(zero_gini_coeff_values);
    try std.testing.expect(std.math.isNan(zero_gini_coeff_values[0]));

    var zero_mean_abs_dev_ratio = try zero_gini_table.groupByMeanAbsDevRatio("bucket", "value", "value_mean_abs_dev_ratio");
    defer zero_mean_abs_dev_ratio.deinit();
    const zero_mean_abs_dev_ratio_values = try (try zero_mean_abs_dev_ratio.column("value_mean_abs_dev_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(zero_mean_abs_dev_ratio_values);
    try std.testing.expect(std.math.isNan(zero_mean_abs_dev_ratio_values[0]));

    var weighted_key = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 1, 1, 2, 2, 2, 3, 3 }, .cpu);
    defer weighted_key.deinit();
    var weighted_day = try DeviceColumn.fromSlice(i32, gpa, &.{ 10, 10, 11, 10, 10, 11, 10, 10 }, .cpu);
    defer weighted_day.deinit();
    var weighted_value = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 10.0, 20.0, 30.0, 5.0, 15.0, 100.0, 7.0, 9.0 }, &.{ true, true, true, true, true, false, true, true }, .cpu);
    defer weighted_value.deinit();
    var weighted_lhs = try DeviceColumn.fromSlice(f64, gpa, &.{ 1.0, 2.0, 4.0, 1.0, 3.0, 100.0, 7.0, 9.0 }, .cpu);
    defer weighted_lhs.deinit();
    var weighted_rhs = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 2.0, 1.0, 8.0, 2.0, 0.0, 100.0, -1.0, -2.0 }, &.{ true, true, true, true, true, false, true, true }, .cpu);
    defer weighted_rhs.deinit();
    var weighted_weight = try DeviceColumn.fromSlice(f64, gpa, &.{ 1.0, 3.0, 2.0, 1.0, 1.0, 10.0, 0.0, 0.0 }, .cpu);
    defer weighted_weight.deinit();
    var weighted_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "bucket", .data = weighted_key },
        .{ .name = "day", .data = weighted_day },
        .{ .name = "value", .data = weighted_value },
        .{ .name = "lhs", .data = weighted_lhs },
        .{ .name = "rhs", .data = weighted_rhs },
        .{ .name = "weight", .data = weighted_weight },
    });
    defer weighted_table.deinit();

    var weighted_mean = try weighted_table.groupByWeightedMean("bucket", "value", "weight", "value_weighted_mean");
    defer weighted_mean.deinit();
    const weighted_mean_values = try (try weighted_mean.column("value_weighted_mean")).f64.toOwnedSlice(gpa);
    defer gpa.free(weighted_mean_values);
    try std.testing.expectApproxEqAbs(@as(f64, 65.0 / 3.0), weighted_mean_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), weighted_mean_values[1], 1e-12);
    try std.testing.expect(std.math.isNan(weighted_mean_values[2]));

    var weighted_variance = try weighted_table.groupByWeightedVarOn(&.{"bucket"}, "value", "weight", "value_weighted_variance");
    defer weighted_variance.deinit();
    const weighted_variance_values = try (try weighted_variance.column("value_weighted_variance")).f64.toOwnedSlice(gpa);
    defer gpa.free(weighted_variance_values);
    try std.testing.expectApproxEqAbs(@as(f64, 425.0 / 9.0), weighted_variance_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 25.0), weighted_variance_values[1], 1e-12);
    try std.testing.expect(std.math.isNan(weighted_variance_values[2]));

    var weighted_stddev = try weighted_table.groupByWeightedStddev("bucket", "value", "weight", "value_weighted_stddev");
    defer weighted_stddev.deinit();
    const weighted_stddev_values = try (try weighted_stddev.column("value_weighted_stddev")).f64.toOwnedSlice(gpa);
    defer gpa.free(weighted_stddev_values);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 425.0 / 9.0)), weighted_stddev_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), weighted_stddev_values[1], 1e-12);
    try std.testing.expect(std.math.isNan(weighted_stddev_values[2]));

    var weighted_mean_on = try weighted_table.groupByWeightedMeanOn(&.{ "bucket", "day" }, "value", "weight", "value_weighted_mean_on");
    defer weighted_mean_on.deinit();
    const weighted_mean_on_values = try (try weighted_mean_on.column("value_weighted_mean_on")).f64.toOwnedSlice(gpa);
    defer gpa.free(weighted_mean_on_values);
    try std.testing.expectApproxEqAbs(@as(f64, 17.5), weighted_mean_on_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 30.0), weighted_mean_on_values[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), weighted_mean_on_values[2], 1e-12);
    try std.testing.expect(std.math.isNan(weighted_mean_on_values[3]));

    var weighted_quantile = try weighted_table.groupByWeightedQuantile("bucket", "value", "weight", "value_weighted_q75", 0.75);
    defer weighted_quantile.deinit();
    const weighted_quantile_values = try (try weighted_quantile.column("value_weighted_q75")).f64.toOwnedSlice(gpa);
    defer gpa.free(weighted_quantile_values);
    try std.testing.expectApproxEqAbs(@as(f64, 30.0), weighted_quantile_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 15.0), weighted_quantile_values[1], 1e-12);
    try std.testing.expect(std.math.isNan(weighted_quantile_values[2]));

    var weighted_median = try weighted_table.groupByWeightedMedian("bucket", "value", "weight", "value_weighted_median");
    defer weighted_median.deinit();
    const weighted_median_values = try (try weighted_median.column("value_weighted_median")).f64.toOwnedSlice(gpa);
    defer gpa.free(weighted_median_values);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), weighted_median_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), weighted_median_values[1], 1e-12);
    try std.testing.expect(std.math.isNan(weighted_median_values[2]));

    var weighted_iqr = try weighted_table.groupByWeightedIQROn(&.{"bucket"}, "value", "weight", "value_weighted_iqr");
    defer weighted_iqr.deinit();
    const weighted_iqr_values = try (try weighted_iqr.column("value_weighted_iqr")).f64.toOwnedSlice(gpa);
    defer gpa.free(weighted_iqr_values);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), weighted_iqr_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), weighted_iqr_values[1], 1e-12);
    try std.testing.expect(std.math.isNan(weighted_iqr_values[2]));

    var weighted_mad = try weighted_table.groupByWeightedMad("bucket", "value", "weight", "value_weighted_mad");
    defer weighted_mad.deinit();
    const weighted_mad_values = try (try weighted_mad.column("value_weighted_mad")).f64.toOwnedSlice(gpa);
    defer gpa.free(weighted_mad_values);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), weighted_mad_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), weighted_mad_values[1], 1e-12);
    try std.testing.expect(std.math.isNan(weighted_mad_values[2]));

    const weighted_nan = std.math.nan(f64);
    const weighted_mode_expected = [_]f64{ 20.0, 5.0, weighted_nan };
    const weighted_mode_weight_expected = [_]f64{ 3.0, 1.0, weighted_nan };
    const weighted_mode_ratio_expected = [_]f64{ 0.5, 0.5, weighted_nan };
    const weighted_mode_margin_expected = [_]f64{ 1.0, 0.0, weighted_nan };
    const weighted_mode_margin_ratio_expected = [_]f64{ 1.0 / 6.0, 0.0, weighted_nan };
    const weighted_entropy_1 = -(@as(f64, 1.0 / 6.0) * std.math.log(f64, std.math.e, @as(f64, 1.0 / 6.0)) + @as(f64, 0.5) * std.math.log(f64, std.math.e, @as(f64, 0.5)) + @as(f64, 1.0 / 3.0) * std.math.log(f64, std.math.e, @as(f64, 1.0 / 3.0)));
    const weighted_entropy_2 = std.math.log(f64, std.math.e, @as(f64, 2.0));
    const weighted_concentration_1 = @as(f64, 7.0 / 18.0);
    const weighted_concentration_2 = @as(f64, 0.5);
    const weighted_entropy_expected = [_]f64{ weighted_entropy_1, weighted_entropy_2, weighted_nan };
    const weighted_gini_expected = [_]f64{ @as(f64, 1.0) - weighted_concentration_1, @as(f64, 1.0) - weighted_concentration_2, weighted_nan };
    const weighted_perplexity_expected = [_]f64{ std.math.exp(weighted_entropy_1), std.math.exp(weighted_entropy_2), weighted_nan };
    const weighted_inverse_simpson_expected = [_]f64{ @as(f64, 1.0) / weighted_concentration_1, @as(f64, 1.0) / weighted_concentration_2, weighted_nan };
    const weighted_concentration_expected = [_]f64{ weighted_concentration_1, weighted_concentration_2, weighted_nan };
    const weighted_evenness_expected = [_]f64{ weighted_entropy_1 / std.math.log(f64, std.math.e, @as(f64, 3.0)), 1.0, weighted_nan };

    var weighted_mode = try weighted_table.groupByWeightedMode("bucket", "value", "weight", "value_weighted_mode");
    defer weighted_mode.deinit();
    try expectF64ColumnApproxOrNan(weighted_mode, gpa, "value_weighted_mode", &weighted_mode_expected);

    var weighted_mode_weight = try weighted_table.groupByWeightedModeWeight("bucket", "value", "weight", "value_weighted_mode_weight");
    defer weighted_mode_weight.deinit();
    try expectF64ColumnApproxOrNan(weighted_mode_weight, gpa, "value_weighted_mode_weight", &weighted_mode_weight_expected);

    var weighted_mode_ratio = try weighted_table.groupByWeightedModeRatio("bucket", "value", "weight", "value_weighted_mode_ratio");
    defer weighted_mode_ratio.deinit();
    try expectF64ColumnApproxOrNan(weighted_mode_ratio, gpa, "value_weighted_mode_ratio", &weighted_mode_ratio_expected);

    var weighted_mode_margin = try weighted_table.groupByWeightedModeMargin("bucket", "value", "weight", "value_weighted_mode_margin");
    defer weighted_mode_margin.deinit();
    try expectF64ColumnApproxOrNan(weighted_mode_margin, gpa, "value_weighted_mode_margin", &weighted_mode_margin_expected);

    var weighted_mode_margin_ratio = try weighted_table.groupByWeightedModeMarginRatio("bucket", "value", "weight", "value_weighted_mode_margin_ratio");
    defer weighted_mode_margin_ratio.deinit();
    try expectF64ColumnApproxOrNan(weighted_mode_margin_ratio, gpa, "value_weighted_mode_margin_ratio", &weighted_mode_margin_ratio_expected);

    var weighted_entropy = try weighted_table.groupByWeightedEntropy("bucket", "value", "weight", "value_weighted_entropy");
    defer weighted_entropy.deinit();
    try expectF64ColumnApproxOrNan(weighted_entropy, gpa, "value_weighted_entropy", &weighted_entropy_expected);

    var weighted_gini = try weighted_table.groupByWeightedGini("bucket", "value", "weight", "value_weighted_gini");
    defer weighted_gini.deinit();
    try expectF64ColumnApproxOrNan(weighted_gini, gpa, "value_weighted_gini", &weighted_gini_expected);

    var weighted_perplexity = try weighted_table.groupByWeightedPerplexity("bucket", "value", "weight", "value_weighted_perplexity");
    defer weighted_perplexity.deinit();
    try expectF64ColumnApproxOrNan(weighted_perplexity, gpa, "value_weighted_perplexity", &weighted_perplexity_expected);

    var weighted_inverse = try weighted_table.groupByWeightedInverseSimpson("bucket", "value", "weight", "value_weighted_inverse");
    defer weighted_inverse.deinit();
    try expectF64ColumnApproxOrNan(weighted_inverse, gpa, "value_weighted_inverse", &weighted_inverse_simpson_expected);

    var weighted_concentration = try weighted_table.groupByWeightedConcentration("bucket", "value", "weight", "value_weighted_concentration");
    defer weighted_concentration.deinit();
    try expectF64ColumnApproxOrNan(weighted_concentration, gpa, "value_weighted_concentration", &weighted_concentration_expected);

    var weighted_evenness = try weighted_table.groupByWeightedEvenness("bucket", "value", "weight", "value_weighted_evenness");
    defer weighted_evenness.deinit();
    try expectF64ColumnApproxOrNan(weighted_evenness, gpa, "value_weighted_evenness", &weighted_evenness_expected);

    var pair_count = try weighted_table.groupByPairCount("bucket", "lhs", "rhs", "lhs_rhs_pair_count");
    defer pair_count.deinit();
    const pair_count_values = try (try pair_count.column("lhs_rhs_pair_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(pair_count_values);
    try std.testing.expectEqualSlices(i64, &.{ 3, 2, 2 }, pair_count_values);

    const pair_dot_expected = [_]f64{ 36.0, 2.0, -25.0 };
    const pair_cos_expected = [_]f64{ 36.0 / std.math.sqrt(@as(f64, 21.0 * 69.0)), 1.0 / std.math.sqrt(@as(f64, 10.0)), -25.0 / std.math.sqrt(@as(f64, 650.0)) };
    const pair_sqdist_expected = [_]f64{ 18.0, 10.0, 185.0 };
    const pair_euclidean_expected = [_]f64{ std.math.sqrt(@as(f64, 18.0)), std.math.sqrt(@as(f64, 10.0)), std.math.sqrt(@as(f64, 185.0)) };
    const pair_manhattan_expected = [_]f64{ 6.0, 4.0, 19.0 };
    const pair_chebyshev_expected = [_]f64{ 4.0, 3.0, 11.0 };
    const pair_canberra_expected = [_]f64{ 1.0, 4.0 / 3.0, 2.0 };
    const pair_bray_expected = [_]f64{ 1.0 / 3.0, 2.0 / 3.0, 1.0 };
    const pair_mean_error_expected = [_]f64{ -4.0 / 3.0, 1.0, 19.0 / 2.0 };
    const pair_mae_expected = [_]f64{ 2.0, 2.0, 19.0 / 2.0 };
    const pair_mse_expected = [_]f64{ 6.0, 5.0, 185.0 / 2.0 };
    const pair_rmse_expected = [_]f64{ std.math.sqrt(@as(f64, 6.0)), std.math.sqrt(@as(f64, 5.0)), std.math.sqrt(@as(f64, 185.0 / 2.0)) };
    const pair_mape_expected = [_]f64{ 5.0 / 6.0, 1.0, 149.0 / 126.0 };
    const pair_smape_expected = [_]f64{ 2.0 / 3.0, 4.0 / 3.0, 2.0 };
    const pair_cov_expected = [_]f64{ 31.0 / 9.0, -1.0, -0.5 };
    const pair_corr_expected = [_]f64{ 31.0 / std.math.sqrt(@as(f64, 1204.0)), -1.0, -1.0 };
    const pair_beta_expected = [_]f64{ 31.0 / 14.0, -1.0, -0.5 };

    var dot = try weighted_table.groupByDot("bucket", "lhs", "rhs", "lhs_rhs_dot");
    defer dot.deinit();
    try expectF64ColumnApproxOrNan(dot, gpa, "lhs_rhs_dot", &pair_dot_expected);

    var cosine = try weighted_table.groupByCosine("bucket", "lhs", "rhs", "lhs_rhs_cos");
    defer cosine.deinit();
    try expectF64ColumnApproxOrNan(cosine, gpa, "lhs_rhs_cos", &pair_cos_expected);

    var squared_euclidean = try weighted_table.groupBySquaredEuclideanDistance("bucket", "lhs", "rhs", "lhs_rhs_sqdist");
    defer squared_euclidean.deinit();
    try expectF64ColumnApproxOrNan(squared_euclidean, gpa, "lhs_rhs_sqdist", &pair_sqdist_expected);

    var euclidean = try weighted_table.groupByEuclideanDistance("bucket", "lhs", "rhs", "lhs_rhs_euclidean");
    defer euclidean.deinit();
    try expectF64ColumnApproxOrNan(euclidean, gpa, "lhs_rhs_euclidean", &pair_euclidean_expected);

    var manhattan = try weighted_table.groupByManhattanDistance("bucket", "lhs", "rhs", "lhs_rhs_manhattan");
    defer manhattan.deinit();
    try expectF64ColumnApproxOrNan(manhattan, gpa, "lhs_rhs_manhattan", &pair_manhattan_expected);

    var chebyshev = try weighted_table.groupByChebyshevDistance("bucket", "lhs", "rhs", "lhs_rhs_chebyshev");
    defer chebyshev.deinit();
    try expectF64ColumnApproxOrNan(chebyshev, gpa, "lhs_rhs_chebyshev", &pair_chebyshev_expected);

    var canberra = try weighted_table.groupByCanberraDistance("bucket", "lhs", "rhs", "lhs_rhs_canberra");
    defer canberra.deinit();
    try expectF64ColumnApproxOrNan(canberra, gpa, "lhs_rhs_canberra", &pair_canberra_expected);

    var bray = try weighted_table.groupByBrayCurtisDistance("bucket", "lhs", "rhs", "lhs_rhs_bray");
    defer bray.deinit();
    try expectF64ColumnApproxOrNan(bray, gpa, "lhs_rhs_bray", &pair_bray_expected);

    var mean_error = try weighted_table.groupByMeanError("bucket", "lhs", "rhs", "lhs_rhs_mean_error");
    defer mean_error.deinit();
    try expectF64ColumnApproxOrNan(mean_error, gpa, "lhs_rhs_mean_error", &pair_mean_error_expected);

    var mae = try weighted_table.groupByMae("bucket", "lhs", "rhs", "lhs_rhs_mae");
    defer mae.deinit();
    try expectF64ColumnApproxOrNan(mae, gpa, "lhs_rhs_mae", &pair_mae_expected);

    var mse = try weighted_table.groupByMse("bucket", "lhs", "rhs", "lhs_rhs_mse");
    defer mse.deinit();
    try expectF64ColumnApproxOrNan(mse, gpa, "lhs_rhs_mse", &pair_mse_expected);

    var rmse = try weighted_table.groupByRmse("bucket", "lhs", "rhs", "lhs_rhs_rmse");
    defer rmse.deinit();
    try expectF64ColumnApproxOrNan(rmse, gpa, "lhs_rhs_rmse", &pair_rmse_expected);

    var mape = try weighted_table.groupByMape("bucket", "lhs", "rhs", "lhs_rhs_mape");
    defer mape.deinit();
    try expectF64ColumnApproxOrNan(mape, gpa, "lhs_rhs_mape", &pair_mape_expected);

    var smape = try weighted_table.groupBySmape("bucket", "lhs", "rhs", "lhs_rhs_smape");
    defer smape.deinit();
    try expectF64ColumnApproxOrNan(smape, gpa, "lhs_rhs_smape", &pair_smape_expected);

    var covariance = try weighted_table.groupByCov("bucket", "lhs", "rhs", "lhs_rhs_cov");
    defer covariance.deinit();
    try expectF64ColumnApproxOrNan(covariance, gpa, "lhs_rhs_cov", &pair_cov_expected);

    var correlation = try weighted_table.groupByCorr("bucket", "lhs", "rhs", "lhs_rhs_corr");
    defer correlation.deinit();
    try expectF64ColumnApproxOrNan(correlation, gpa, "lhs_rhs_corr", &pair_corr_expected);

    var beta = try weighted_table.groupByBeta("bucket", "lhs", "rhs", "lhs_rhs_beta");
    defer beta.deinit();
    try expectF64ColumnApproxOrNan(beta, gpa, "lhs_rhs_beta", &pair_beta_expected);

    const weighted_pair_dot_expected = [_]f64{ 72.0, 2.0, weighted_nan };
    const weighted_pair_cos_expected = [_]f64{ 72.0 / std.math.sqrt(@as(f64, 45.0 * 135.0)), 1.0 / std.math.sqrt(@as(f64, 10.0)), weighted_nan };
    const weighted_pair_sqdist_expected = [_]f64{ 36.0, 10.0, weighted_nan };
    const weighted_pair_euclidean_expected = [_]f64{ 6.0, std.math.sqrt(@as(f64, 10.0)), weighted_nan };
    const weighted_pair_manhattan_expected = [_]f64{ 12.0, 4.0, weighted_nan };
    const weighted_pair_chebyshev_expected = [_]f64{ 4.0, 3.0, weighted_nan };
    const weighted_pair_canberra_expected = [_]f64{ 2.0, 4.0 / 3.0, weighted_nan };
    const weighted_pair_bray_expected = [_]f64{ 1.0 / 3.0, 2.0 / 3.0, weighted_nan };
    const weighted_pair_mean_error_expected = [_]f64{ -1.0, 1.0, weighted_nan };
    const weighted_pair_mae_expected = [_]f64{ 2.0, 2.0, weighted_nan };
    const weighted_pair_mse_expected = [_]f64{ 6.0, 5.0, weighted_nan };
    const weighted_pair_rmse_expected = [_]f64{ std.math.sqrt(@as(f64, 6.0)), std.math.sqrt(@as(f64, 5.0)), weighted_nan };
    const weighted_pair_mape_expected = [_]f64{ 0.75, 1.0, weighted_nan };
    const weighted_pair_smape_expected = [_]f64{ 2.0 / 3.0, 4.0 / 3.0, weighted_nan };
    const weighted_pair_cov_expected = [_]f64{ 13.0 / 4.0, -1.0, weighted_nan };
    const weighted_pair_corr_expected = [_]f64{ 39.0 / std.math.sqrt(@as(f64, 1845.0)), -1.0, weighted_nan };
    const weighted_pair_beta_expected = [_]f64{ 13.0 / 5.0, -1.0, weighted_nan };

    var weighted_dot = try weighted_table.groupByWeightedDot("bucket", "lhs", "rhs", "weight", "lhs_rhs_weighted_dot");
    defer weighted_dot.deinit();
    try expectF64ColumnApproxOrNan(weighted_dot, gpa, "lhs_rhs_weighted_dot", &weighted_pair_dot_expected);

    var weighted_cosine = try weighted_table.groupByWeightedCosine("bucket", "lhs", "rhs", "weight", "lhs_rhs_weighted_cos");
    defer weighted_cosine.deinit();
    try expectF64ColumnApproxOrNan(weighted_cosine, gpa, "lhs_rhs_weighted_cos", &weighted_pair_cos_expected);

    var weighted_sqdist = try weighted_table.groupByWeightedSquaredEuclideanDistance("bucket", "lhs", "rhs", "weight", "lhs_rhs_weighted_sqdist");
    defer weighted_sqdist.deinit();
    try expectF64ColumnApproxOrNan(weighted_sqdist, gpa, "lhs_rhs_weighted_sqdist", &weighted_pair_sqdist_expected);

    var weighted_euclidean = try weighted_table.groupByWeightedEuclideanDistance("bucket", "lhs", "rhs", "weight", "lhs_rhs_weighted_euclidean");
    defer weighted_euclidean.deinit();
    try expectF64ColumnApproxOrNan(weighted_euclidean, gpa, "lhs_rhs_weighted_euclidean", &weighted_pair_euclidean_expected);

    var weighted_manhattan = try weighted_table.groupByWeightedManhattanDistance("bucket", "lhs", "rhs", "weight", "lhs_rhs_weighted_manhattan");
    defer weighted_manhattan.deinit();
    try expectF64ColumnApproxOrNan(weighted_manhattan, gpa, "lhs_rhs_weighted_manhattan", &weighted_pair_manhattan_expected);

    var weighted_chebyshev = try weighted_table.groupByWeightedChebyshevDistance("bucket", "lhs", "rhs", "weight", "lhs_rhs_weighted_chebyshev");
    defer weighted_chebyshev.deinit();
    try expectF64ColumnApproxOrNan(weighted_chebyshev, gpa, "lhs_rhs_weighted_chebyshev", &weighted_pair_chebyshev_expected);

    var weighted_canberra = try weighted_table.groupByWeightedCanberraDistance("bucket", "lhs", "rhs", "weight", "lhs_rhs_weighted_canberra");
    defer weighted_canberra.deinit();
    try expectF64ColumnApproxOrNan(weighted_canberra, gpa, "lhs_rhs_weighted_canberra", &weighted_pair_canberra_expected);

    var weighted_bray = try weighted_table.groupByWeightedBrayCurtisDistance("bucket", "lhs", "rhs", "weight", "lhs_rhs_weighted_bray");
    defer weighted_bray.deinit();
    try expectF64ColumnApproxOrNan(weighted_bray, gpa, "lhs_rhs_weighted_bray", &weighted_pair_bray_expected);

    var weighted_mean_error = try weighted_table.groupByWeightedMeanError("bucket", "lhs", "rhs", "weight", "lhs_rhs_weighted_mean_error");
    defer weighted_mean_error.deinit();
    try expectF64ColumnApproxOrNan(weighted_mean_error, gpa, "lhs_rhs_weighted_mean_error", &weighted_pair_mean_error_expected);

    var weighted_mae = try weighted_table.groupByWeightedMae("bucket", "lhs", "rhs", "weight", "lhs_rhs_weighted_mae");
    defer weighted_mae.deinit();
    try expectF64ColumnApproxOrNan(weighted_mae, gpa, "lhs_rhs_weighted_mae", &weighted_pair_mae_expected);

    var weighted_mse = try weighted_table.groupByWeightedMse("bucket", "lhs", "rhs", "weight", "lhs_rhs_weighted_mse");
    defer weighted_mse.deinit();
    try expectF64ColumnApproxOrNan(weighted_mse, gpa, "lhs_rhs_weighted_mse", &weighted_pair_mse_expected);

    var weighted_rmse = try weighted_table.groupByWeightedRmse("bucket", "lhs", "rhs", "weight", "lhs_rhs_weighted_rmse");
    defer weighted_rmse.deinit();
    try expectF64ColumnApproxOrNan(weighted_rmse, gpa, "lhs_rhs_weighted_rmse", &weighted_pair_rmse_expected);

    var weighted_mape = try weighted_table.groupByWeightedMape("bucket", "lhs", "rhs", "weight", "lhs_rhs_weighted_mape");
    defer weighted_mape.deinit();
    try expectF64ColumnApproxOrNan(weighted_mape, gpa, "lhs_rhs_weighted_mape", &weighted_pair_mape_expected);

    var weighted_smape = try weighted_table.groupByWeightedSmape("bucket", "lhs", "rhs", "weight", "lhs_rhs_weighted_smape");
    defer weighted_smape.deinit();
    try expectF64ColumnApproxOrNan(weighted_smape, gpa, "lhs_rhs_weighted_smape", &weighted_pair_smape_expected);

    var weighted_covariance = try weighted_table.groupByWeightedCov("bucket", "lhs", "rhs", "weight", "lhs_rhs_weighted_cov", 0.0);
    defer weighted_covariance.deinit();
    try expectF64ColumnApproxOrNan(weighted_covariance, gpa, "lhs_rhs_weighted_cov", &weighted_pair_cov_expected);

    var weighted_correlation = try weighted_table.groupByWeightedCorr("bucket", "lhs", "rhs", "weight", "lhs_rhs_weighted_corr", 0.0);
    defer weighted_correlation.deinit();
    try expectF64ColumnApproxOrNan(weighted_correlation, gpa, "lhs_rhs_weighted_corr", &weighted_pair_corr_expected);

    var weighted_beta = try weighted_table.groupByWeightedBeta("bucket", "lhs", "rhs", "weight", "lhs_rhs_weighted_beta", 0.0);
    defer weighted_beta.deinit();
    try expectF64ColumnApproxOrNan(weighted_beta, gpa, "lhs_rhs_weighted_beta", &weighted_pair_beta_expected);

    var weighted_median_on = try weighted_table.groupByWeightedMedianOn(&.{ "bucket", "day" }, "value", "weight", "value_weighted_median_on");
    defer weighted_median_on.deinit();
    const weighted_median_on_values = try (try weighted_median_on.column("value_weighted_median_on")).f64.toOwnedSlice(gpa);
    defer gpa.free(weighted_median_on_values);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), weighted_median_on_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 30.0), weighted_median_on_values[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), weighted_median_on_values[2], 1e-12);
    try std.testing.expect(std.math.isNan(weighted_median_on_values[3]));

    const weighted_mode_on_expected = [_]f64{ 20.0, 30.0, 5.0, weighted_nan };
    var weighted_mode_on = try weighted_table.groupByWeightedModeOn(&.{ "bucket", "day" }, "value", "weight", "value_weighted_mode_on");
    defer weighted_mode_on.deinit();
    try expectF64ColumnApproxOrNan(weighted_mode_on, gpa, "value_weighted_mode_on", &weighted_mode_on_expected);

    const pair_count_on_expected = [_]i64{ 2, 1, 2, 0, 2 };
    var pair_count_on = try weighted_table.groupByPairCountOn(&.{ "bucket", "day" }, "lhs", "rhs", "lhs_rhs_pair_count_on");
    defer pair_count_on.deinit();
    const pair_count_on_values = try (try pair_count_on.column("lhs_rhs_pair_count_on")).i64.toOwnedSlice(gpa);
    defer gpa.free(pair_count_on_values);
    try std.testing.expectEqualSlices(i64, &pair_count_on_expected, pair_count_on_values);

    const pair_cov_on_expected = [_]f64{ -0.25, 0.0, -1.0, weighted_nan, -0.5 };
    var covariance_on = try weighted_table.groupByCovarianceOn(&.{ "bucket", "day" }, "lhs", "rhs", "lhs_rhs_cov_on");
    defer covariance_on.deinit();
    try expectF64ColumnApproxOrNan(covariance_on, gpa, "lhs_rhs_cov_on", &pair_cov_on_expected);

    const weighted_cov_on_expected = [_]f64{ -3.0 / 16.0, 0.0, -1.0, weighted_nan };
    var weighted_covariance_on = try weighted_table.groupByWeightedCovarianceOn(&.{ "bucket", "day" }, "lhs", "rhs", "weight", "lhs_rhs_weighted_cov_on", 0.0);
    defer weighted_covariance_on.deinit();
    try expectF64ColumnApproxOrNan(weighted_covariance_on, gpa, "lhs_rhs_weighted_cov_on", &weighted_cov_on_expected);

    try std.testing.expectError(error.InvalidShape, weighted_table.groupByWeightedQuantile("bucket", "value", "weight", "bad_weighted_q", 1.5));
    try std.testing.expectError(error.InvalidShape, weighted_table.groupByWeightedCovariance("bucket", "lhs", "rhs", "weight", "bad_weighted_cov", -1.0));

    var negative_weight_key = try DeviceColumn.fromSlice(i32, gpa, &.{1}, .cpu);
    defer negative_weight_key.deinit();
    var negative_weight_value = try DeviceColumn.fromSlice(f64, gpa, &.{1.0}, .cpu);
    defer negative_weight_value.deinit();
    var negative_weight = try DeviceColumn.fromSlice(f64, gpa, &.{-1.0}, .cpu);
    defer negative_weight.deinit();
    var negative_weight_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "bucket", .data = negative_weight_key },
        .{ .name = "value", .data = negative_weight_value },
        .{ .name = "weight", .data = negative_weight },
    });
    defer negative_weight_table.deinit();
    try std.testing.expectError(error.InvalidShape, negative_weight_table.groupByWeightedMean("bucket", "value", "weight", "bad_weighted_mean"));
    try std.testing.expectError(error.InvalidShape, negative_weight_table.groupByWeightedCovariance("bucket", "value", "value", "weight", "bad_weighted_cov", 0.0));

    var weighted_mean_plan = try DeviceLazyFrame.init(gpa, weighted_table);
    defer weighted_mean_plan.deinit();
    try weighted_mean_plan.groupByWeightedMeanOn(&.{"bucket"}, "value", "weight", "value_weighted_mean_lazy");
    const weighted_mean_explained = try weighted_mean_plan.explain(gpa);
    defer gpa.free(weighted_mean_explained);
    try std.testing.expect(std.mem.indexOf(u8, weighted_mean_explained, "group_by_weighted_mean_on([bucket], value=value, weight=weight -> value_weighted_mean_lazy)") != null);
    var lazy_weighted_mean = try weighted_mean_plan.collect();
    defer lazy_weighted_mean.deinit();
    const lazy_weighted_mean_values = try (try lazy_weighted_mean.column("value_weighted_mean_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_weighted_mean_values);
    try std.testing.expectApproxEqAbs(@as(f64, 65.0 / 3.0), lazy_weighted_mean_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), lazy_weighted_mean_values[1], 1e-12);
    try std.testing.expect(std.math.isNan(lazy_weighted_mean_values[2]));

    var weighted_std_plan = try DeviceLazyFrame.init(gpa, weighted_table);
    defer weighted_std_plan.deinit();
    try weighted_std_plan.groupByWeightedStddev("bucket", "value", "weight", "value_weighted_stddev_lazy");
    const weighted_std_explained = try weighted_std_plan.explain(gpa);
    defer gpa.free(weighted_std_explained);
    try std.testing.expect(std.mem.indexOf(u8, weighted_std_explained, "group_by_weighted_stddev(bucket, value=value, weight=weight -> value_weighted_stddev_lazy)") != null);
    var lazy_weighted_std = try weighted_std_plan.collect();
    defer lazy_weighted_std.deinit();
    const lazy_weighted_std_values = try (try lazy_weighted_std.column("value_weighted_stddev_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_weighted_std_values);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 425.0 / 9.0)), lazy_weighted_std_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), lazy_weighted_std_values[1], 1e-12);
    try std.testing.expect(std.math.isNan(lazy_weighted_std_values[2]));

    var weighted_quantile_plan = try DeviceLazyFrame.init(gpa, weighted_table);
    defer weighted_quantile_plan.deinit();
    try weighted_quantile_plan.groupByWeightedQuantileOn(&.{"bucket"}, "value", "weight", "value_weighted_q75_lazy", 0.75);
    const weighted_quantile_explained = try weighted_quantile_plan.explain(gpa);
    defer gpa.free(weighted_quantile_explained);
    try std.testing.expect(std.mem.indexOf(u8, weighted_quantile_explained, "group_by_weighted_quantile_on([bucket], value=value, weight=weight, q=0.75 -> value_weighted_q75_lazy)") != null);
    var lazy_weighted_quantile = try weighted_quantile_plan.collect();
    defer lazy_weighted_quantile.deinit();
    const lazy_weighted_quantile_values = try (try lazy_weighted_quantile.column("value_weighted_q75_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_weighted_quantile_values);
    try std.testing.expectApproxEqAbs(@as(f64, 30.0), lazy_weighted_quantile_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 15.0), lazy_weighted_quantile_values[1], 1e-12);
    try std.testing.expect(std.math.isNan(lazy_weighted_quantile_values[2]));

    var weighted_median_plan = try DeviceLazyFrame.init(gpa, weighted_table);
    defer weighted_median_plan.deinit();
    try weighted_median_plan.groupByWeightedMedian("bucket", "value", "weight", "value_weighted_median_lazy");
    const weighted_median_explained = try weighted_median_plan.explain(gpa);
    defer gpa.free(weighted_median_explained);
    try std.testing.expect(std.mem.indexOf(u8, weighted_median_explained, "group_by_weighted_median(bucket, value=value, weight=weight -> value_weighted_median_lazy)") != null);
    var lazy_weighted_median = try weighted_median_plan.collect();
    defer lazy_weighted_median.deinit();
    const lazy_weighted_median_values = try (try lazy_weighted_median.column("value_weighted_median_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_weighted_median_values);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), lazy_weighted_median_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), lazy_weighted_median_values[1], 1e-12);
    try std.testing.expect(std.math.isNan(lazy_weighted_median_values[2]));

    var weighted_iqr_plan = try DeviceLazyFrame.init(gpa, weighted_table);
    defer weighted_iqr_plan.deinit();
    try weighted_iqr_plan.groupByWeightedIqr("bucket", "value", "weight", "value_weighted_iqr_lazy");
    const weighted_iqr_explained = try weighted_iqr_plan.explain(gpa);
    defer gpa.free(weighted_iqr_explained);
    try std.testing.expect(std.mem.indexOf(u8, weighted_iqr_explained, "group_by_weighted_iqr(bucket, value=value, weight=weight -> value_weighted_iqr_lazy)") != null);
    var lazy_weighted_iqr = try weighted_iqr_plan.collect();
    defer lazy_weighted_iqr.deinit();
    const lazy_weighted_iqr_values = try (try lazy_weighted_iqr.column("value_weighted_iqr_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_weighted_iqr_values);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), lazy_weighted_iqr_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), lazy_weighted_iqr_values[1], 1e-12);
    try std.testing.expect(std.math.isNan(lazy_weighted_iqr_values[2]));

    var weighted_mad_plan = try DeviceLazyFrame.init(gpa, weighted_table);
    defer weighted_mad_plan.deinit();
    try weighted_mad_plan.groupByWeightedMad("bucket", "value", "weight", "value_weighted_mad_lazy");
    const weighted_mad_explained = try weighted_mad_plan.explain(gpa);
    defer gpa.free(weighted_mad_explained);
    try std.testing.expect(std.mem.indexOf(u8, weighted_mad_explained, "group_by_weighted_mad(bucket, value=value, weight=weight -> value_weighted_mad_lazy)") != null);
    var lazy_weighted_mad = try weighted_mad_plan.collect();
    defer lazy_weighted_mad.deinit();
    const lazy_weighted_mad_values = try (try lazy_weighted_mad.column("value_weighted_mad_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_weighted_mad_values);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_weighted_mad_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_weighted_mad_values[1], 1e-12);
    try std.testing.expect(std.math.isNan(lazy_weighted_mad_values[2]));

    const weighted_lazy_cases = [_]struct {
        method: DeviceLazyWeightedGroupByAggregation,
        output_name: []const u8,
        explain: []const u8,
        expected: []const f64,
    }{
        .{ .method = .weighted_mode, .output_name = "value_weighted_mode_lazy", .explain = "group_by_weighted_mode(bucket, value=value, weight=weight -> value_weighted_mode_lazy)", .expected = &weighted_mode_expected },
        .{ .method = .weighted_mode_weight, .output_name = "value_weighted_mode_weight_lazy", .explain = "group_by_weighted_mode_weight(bucket, value=value, weight=weight -> value_weighted_mode_weight_lazy)", .expected = &weighted_mode_weight_expected },
        .{ .method = .weighted_mode_ratio, .output_name = "value_weighted_mode_ratio_lazy", .explain = "group_by_weighted_mode_ratio(bucket, value=value, weight=weight -> value_weighted_mode_ratio_lazy)", .expected = &weighted_mode_ratio_expected },
        .{ .method = .weighted_mode_margin, .output_name = "value_weighted_mode_margin_lazy", .explain = "group_by_weighted_mode_margin(bucket, value=value, weight=weight -> value_weighted_mode_margin_lazy)", .expected = &weighted_mode_margin_expected },
        .{ .method = .weighted_mode_margin_ratio, .output_name = "value_weighted_mode_margin_ratio_lazy", .explain = "group_by_weighted_mode_margin_ratio(bucket, value=value, weight=weight -> value_weighted_mode_margin_ratio_lazy)", .expected = &weighted_mode_margin_ratio_expected },
        .{ .method = .weighted_entropy, .output_name = "value_weighted_entropy_lazy", .explain = "group_by_weighted_entropy(bucket, value=value, weight=weight -> value_weighted_entropy_lazy)", .expected = &weighted_entropy_expected },
        .{ .method = .weighted_gini_impurity, .output_name = "value_weighted_gini_lazy", .explain = "group_by_weighted_gini_impurity(bucket, value=value, weight=weight -> value_weighted_gini_lazy)", .expected = &weighted_gini_expected },
        .{ .method = .weighted_perplexity, .output_name = "value_weighted_perplexity_lazy", .explain = "group_by_weighted_perplexity(bucket, value=value, weight=weight -> value_weighted_perplexity_lazy)", .expected = &weighted_perplexity_expected },
        .{ .method = .weighted_inverse_simpson, .output_name = "value_weighted_inverse_lazy", .explain = "group_by_weighted_inverse_simpson(bucket, value=value, weight=weight -> value_weighted_inverse_lazy)", .expected = &weighted_inverse_simpson_expected },
        .{ .method = .weighted_simpson_concentration, .output_name = "value_weighted_concentration_lazy", .explain = "group_by_weighted_simpson_concentration(bucket, value=value, weight=weight -> value_weighted_concentration_lazy)", .expected = &weighted_concentration_expected },
        .{ .method = .weighted_evenness, .output_name = "value_weighted_evenness_lazy", .explain = "group_by_weighted_evenness(bucket, value=value, weight=weight -> value_weighted_evenness_lazy)", .expected = &weighted_evenness_expected },
    };
    for (weighted_lazy_cases) |case| {
        var plan = try DeviceLazyFrame.init(gpa, weighted_table);
        defer plan.deinit();
        try switch (case.method) {
            .weighted_mode => plan.groupByWeightedMode("bucket", "value", "weight", case.output_name),
            .weighted_mode_weight => plan.groupByWeightedModeWeight("bucket", "value", "weight", case.output_name),
            .weighted_mode_ratio => plan.groupByWeightedModeRatio("bucket", "value", "weight", case.output_name),
            .weighted_mode_margin => plan.groupByWeightedModeMargin("bucket", "value", "weight", case.output_name),
            .weighted_mode_margin_ratio => plan.groupByWeightedModeMarginRatio("bucket", "value", "weight", case.output_name),
            .weighted_entropy => plan.groupByWeightedEntropy("bucket", "value", "weight", case.output_name),
            .weighted_gini_impurity => plan.groupByWeightedGini("bucket", "value", "weight", case.output_name),
            .weighted_perplexity => plan.groupByWeightedPerplexity("bucket", "value", "weight", case.output_name),
            .weighted_inverse_simpson => plan.groupByWeightedInverseSimpson("bucket", "value", "weight", case.output_name),
            .weighted_simpson_concentration => plan.groupByWeightedConcentration("bucket", "value", "weight", case.output_name),
            .weighted_evenness => plan.groupByWeightedEvenness("bucket", "value", "weight", case.output_name),
            .weighted_mean, .weighted_variance, .weighted_stddev, .weighted_quantile, .weighted_median, .weighted_iqr, .weighted_mad => unreachable,
        };
        const explained = try plan.explain(gpa);
        defer gpa.free(explained);
        try std.testing.expect(std.mem.indexOf(u8, explained, case.explain) != null);
        var collected = try plan.collect();
        defer collected.deinit();
        try expectF64ColumnApproxOrNan(collected, gpa, case.output_name, case.expected);
    }

    var weighted_mode_on_plan = try DeviceLazyFrame.init(gpa, weighted_table);
    defer weighted_mode_on_plan.deinit();
    try weighted_mode_on_plan.groupByWeightedModeOn(&.{"bucket"}, "value", "weight", "value_weighted_mode_on_lazy");
    const weighted_mode_on_explained = try weighted_mode_on_plan.explain(gpa);
    defer gpa.free(weighted_mode_on_explained);
    try std.testing.expect(std.mem.indexOf(u8, weighted_mode_on_explained, "group_by_weighted_mode_on([bucket], value=value, weight=weight -> value_weighted_mode_on_lazy)") != null);
    var lazy_weighted_mode_on = try weighted_mode_on_plan.collect();
    defer lazy_weighted_mode_on.deinit();
    try expectF64ColumnApproxOrNan(lazy_weighted_mode_on, gpa, "value_weighted_mode_on_lazy", &weighted_mode_expected);

    var pair_count_plan = try DeviceLazyFrame.init(gpa, weighted_table);
    defer pair_count_plan.deinit();
    try pair_count_plan.groupByPairCountOn(&.{"bucket"}, "lhs", "rhs", "lhs_rhs_pair_count_lazy");
    const pair_count_explained = try pair_count_plan.explain(gpa);
    defer gpa.free(pair_count_explained);
    try std.testing.expect(std.mem.indexOf(u8, pair_count_explained, "group_by_pair_count_on([bucket], lhs=lhs, rhs=rhs -> lhs_rhs_pair_count_lazy)") != null);
    var lazy_pair_count = try pair_count_plan.collect();
    defer lazy_pair_count.deinit();
    const lazy_pair_count_values = try (try lazy_pair_count.column("lhs_rhs_pair_count_lazy")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_pair_count_values);
    try std.testing.expectEqualSlices(i64, &.{ 3, 2, 2 }, lazy_pair_count_values);

    const pair_lazy_cases = [_]struct {
        method: DeviceLazyPairGroupByAggregation,
        output_name: []const u8,
        explain: []const u8,
        expected: []const f64,
    }{
        .{ .method = .dot, .output_name = "lhs_rhs_dot_lazy", .explain = "group_by_dot(bucket, lhs=lhs, rhs=rhs -> lhs_rhs_dot_lazy)", .expected = &pair_dot_expected },
        .{ .method = .cosine_similarity, .output_name = "lhs_rhs_cos_lazy", .explain = "group_by_cosine_similarity(bucket, lhs=lhs, rhs=rhs -> lhs_rhs_cos_lazy)", .expected = &pair_cos_expected },
        .{ .method = .squared_euclidean_distance, .output_name = "lhs_rhs_sqdist_lazy", .explain = "group_by_squared_euclidean_distance(bucket, lhs=lhs, rhs=rhs -> lhs_rhs_sqdist_lazy)", .expected = &pair_sqdist_expected },
        .{ .method = .euclidean_distance, .output_name = "lhs_rhs_euclidean_lazy", .explain = "group_by_euclidean_distance(bucket, lhs=lhs, rhs=rhs -> lhs_rhs_euclidean_lazy)", .expected = &pair_euclidean_expected },
        .{ .method = .manhattan_distance, .output_name = "lhs_rhs_manhattan_lazy", .explain = "group_by_manhattan_distance(bucket, lhs=lhs, rhs=rhs -> lhs_rhs_manhattan_lazy)", .expected = &pair_manhattan_expected },
        .{ .method = .chebyshev_distance, .output_name = "lhs_rhs_chebyshev_lazy", .explain = "group_by_chebyshev_distance(bucket, lhs=lhs, rhs=rhs -> lhs_rhs_chebyshev_lazy)", .expected = &pair_chebyshev_expected },
        .{ .method = .canberra_distance, .output_name = "lhs_rhs_canberra_lazy", .explain = "group_by_canberra_distance(bucket, lhs=lhs, rhs=rhs -> lhs_rhs_canberra_lazy)", .expected = &pair_canberra_expected },
        .{ .method = .bray_curtis_distance, .output_name = "lhs_rhs_bray_lazy", .explain = "group_by_bray_curtis_distance(bucket, lhs=lhs, rhs=rhs -> lhs_rhs_bray_lazy)", .expected = &pair_bray_expected },
        .{ .method = .mean_error, .output_name = "lhs_rhs_mean_error_lazy", .explain = "group_by_mean_error(bucket, lhs=lhs, rhs=rhs -> lhs_rhs_mean_error_lazy)", .expected = &pair_mean_error_expected },
        .{ .method = .mae, .output_name = "lhs_rhs_mae_lazy", .explain = "group_by_mae(bucket, lhs=lhs, rhs=rhs -> lhs_rhs_mae_lazy)", .expected = &pair_mae_expected },
        .{ .method = .mse, .output_name = "lhs_rhs_mse_lazy", .explain = "group_by_mse(bucket, lhs=lhs, rhs=rhs -> lhs_rhs_mse_lazy)", .expected = &pair_mse_expected },
        .{ .method = .rmse, .output_name = "lhs_rhs_rmse_lazy", .explain = "group_by_rmse(bucket, lhs=lhs, rhs=rhs -> lhs_rhs_rmse_lazy)", .expected = &pair_rmse_expected },
        .{ .method = .mape, .output_name = "lhs_rhs_mape_lazy", .explain = "group_by_mape(bucket, lhs=lhs, rhs=rhs -> lhs_rhs_mape_lazy)", .expected = &pair_mape_expected },
        .{ .method = .smape, .output_name = "lhs_rhs_smape_lazy", .explain = "group_by_smape(bucket, lhs=lhs, rhs=rhs -> lhs_rhs_smape_lazy)", .expected = &pair_smape_expected },
        .{ .method = .covariance, .output_name = "lhs_rhs_cov_lazy", .explain = "group_by_covariance(bucket, lhs=lhs, rhs=rhs -> lhs_rhs_cov_lazy)", .expected = &pair_cov_expected },
        .{ .method = .correlation, .output_name = "lhs_rhs_corr_lazy", .explain = "group_by_correlation(bucket, lhs=lhs, rhs=rhs -> lhs_rhs_corr_lazy)", .expected = &pair_corr_expected },
        .{ .method = .beta, .output_name = "lhs_rhs_beta_lazy", .explain = "group_by_beta(bucket, lhs=lhs, rhs=rhs -> lhs_rhs_beta_lazy)", .expected = &pair_beta_expected },
        .{ .method = .pair_count, .output_name = "", .explain = "", .expected = &.{} },
    };
    for (pair_lazy_cases) |case| {
        if (case.method == .pair_count) continue;
        var plan = try DeviceLazyFrame.init(gpa, weighted_table);
        defer plan.deinit();
        try switch (case.method) {
            .dot => plan.groupByDot("bucket", "lhs", "rhs", case.output_name),
            .cosine_similarity => plan.groupByCosine("bucket", "lhs", "rhs", case.output_name),
            .squared_euclidean_distance => plan.groupBySquaredEuclideanDistance("bucket", "lhs", "rhs", case.output_name),
            .euclidean_distance => plan.groupByEuclideanDistance("bucket", "lhs", "rhs", case.output_name),
            .manhattan_distance => plan.groupByManhattanDistance("bucket", "lhs", "rhs", case.output_name),
            .chebyshev_distance => plan.groupByChebyshevDistance("bucket", "lhs", "rhs", case.output_name),
            .canberra_distance => plan.groupByCanberraDistance("bucket", "lhs", "rhs", case.output_name),
            .bray_curtis_distance => plan.groupByBrayCurtisDistance("bucket", "lhs", "rhs", case.output_name),
            .mean_error => plan.groupByMeanError("bucket", "lhs", "rhs", case.output_name),
            .mae => plan.groupByMae("bucket", "lhs", "rhs", case.output_name),
            .mse => plan.groupByMse("bucket", "lhs", "rhs", case.output_name),
            .rmse => plan.groupByRmse("bucket", "lhs", "rhs", case.output_name),
            .mape => plan.groupByMape("bucket", "lhs", "rhs", case.output_name),
            .smape => plan.groupBySmape("bucket", "lhs", "rhs", case.output_name),
            .covariance => plan.groupByCov("bucket", "lhs", "rhs", case.output_name),
            .correlation => plan.groupByCorr("bucket", "lhs", "rhs", case.output_name),
            .beta => plan.groupByBeta("bucket", "lhs", "rhs", case.output_name),
            .pair_count => unreachable,
        };
        const explained = try plan.explain(gpa);
        defer gpa.free(explained);
        try std.testing.expect(std.mem.indexOf(u8, explained, case.explain) != null);
        var collected = try plan.collect();
        defer collected.deinit();
        try expectF64ColumnApproxOrNan(collected, gpa, case.output_name, case.expected);
    }

    var covariance_on_plan = try DeviceLazyFrame.init(gpa, weighted_table);
    defer covariance_on_plan.deinit();
    try covariance_on_plan.groupByCovarianceOn(&.{"bucket"}, "lhs", "rhs", "lhs_rhs_cov_on_lazy");
    const covariance_on_explained = try covariance_on_plan.explain(gpa);
    defer gpa.free(covariance_on_explained);
    try std.testing.expect(std.mem.indexOf(u8, covariance_on_explained, "group_by_covariance_on([bucket], lhs=lhs, rhs=rhs -> lhs_rhs_cov_on_lazy)") != null);
    var lazy_covariance_on = try covariance_on_plan.collect();
    defer lazy_covariance_on.deinit();
    try expectF64ColumnApproxOrNan(lazy_covariance_on, gpa, "lhs_rhs_cov_on_lazy", &pair_cov_expected);

    const weighted_pair_lazy_cases = [_]struct {
        method: DeviceLazyWeightedPairGroupByAggregation,
        output_name: []const u8,
        explain: []const u8,
        expected: []const f64,
    }{
        .{ .method = .weighted_dot, .output_name = "lhs_rhs_weighted_dot_lazy", .explain = "group_by_weighted_dot(bucket, lhs=lhs, rhs=rhs, weight=weight -> lhs_rhs_weighted_dot_lazy)", .expected = &weighted_pair_dot_expected },
        .{ .method = .weighted_cosine_similarity, .output_name = "lhs_rhs_weighted_cos_lazy", .explain = "group_by_weighted_cosine_similarity(bucket, lhs=lhs, rhs=rhs, weight=weight -> lhs_rhs_weighted_cos_lazy)", .expected = &weighted_pair_cos_expected },
        .{ .method = .weighted_squared_euclidean_distance, .output_name = "lhs_rhs_weighted_sqdist_lazy", .explain = "group_by_weighted_squared_euclidean_distance(bucket, lhs=lhs, rhs=rhs, weight=weight -> lhs_rhs_weighted_sqdist_lazy)", .expected = &weighted_pair_sqdist_expected },
        .{ .method = .weighted_euclidean_distance, .output_name = "lhs_rhs_weighted_euclidean_lazy", .explain = "group_by_weighted_euclidean_distance(bucket, lhs=lhs, rhs=rhs, weight=weight -> lhs_rhs_weighted_euclidean_lazy)", .expected = &weighted_pair_euclidean_expected },
        .{ .method = .weighted_manhattan_distance, .output_name = "lhs_rhs_weighted_manhattan_lazy", .explain = "group_by_weighted_manhattan_distance(bucket, lhs=lhs, rhs=rhs, weight=weight -> lhs_rhs_weighted_manhattan_lazy)", .expected = &weighted_pair_manhattan_expected },
        .{ .method = .weighted_chebyshev_distance, .output_name = "lhs_rhs_weighted_chebyshev_lazy", .explain = "group_by_weighted_chebyshev_distance(bucket, lhs=lhs, rhs=rhs, weight=weight -> lhs_rhs_weighted_chebyshev_lazy)", .expected = &weighted_pair_chebyshev_expected },
        .{ .method = .weighted_canberra_distance, .output_name = "lhs_rhs_weighted_canberra_lazy", .explain = "group_by_weighted_canberra_distance(bucket, lhs=lhs, rhs=rhs, weight=weight -> lhs_rhs_weighted_canberra_lazy)", .expected = &weighted_pair_canberra_expected },
        .{ .method = .weighted_bray_curtis_distance, .output_name = "lhs_rhs_weighted_bray_lazy", .explain = "group_by_weighted_bray_curtis_distance(bucket, lhs=lhs, rhs=rhs, weight=weight -> lhs_rhs_weighted_bray_lazy)", .expected = &weighted_pair_bray_expected },
        .{ .method = .weighted_mean_error, .output_name = "lhs_rhs_weighted_mean_error_lazy", .explain = "group_by_weighted_mean_error(bucket, lhs=lhs, rhs=rhs, weight=weight -> lhs_rhs_weighted_mean_error_lazy)", .expected = &weighted_pair_mean_error_expected },
        .{ .method = .weighted_mae, .output_name = "lhs_rhs_weighted_mae_lazy", .explain = "group_by_weighted_mae(bucket, lhs=lhs, rhs=rhs, weight=weight -> lhs_rhs_weighted_mae_lazy)", .expected = &weighted_pair_mae_expected },
        .{ .method = .weighted_mse, .output_name = "lhs_rhs_weighted_mse_lazy", .explain = "group_by_weighted_mse(bucket, lhs=lhs, rhs=rhs, weight=weight -> lhs_rhs_weighted_mse_lazy)", .expected = &weighted_pair_mse_expected },
        .{ .method = .weighted_rmse, .output_name = "lhs_rhs_weighted_rmse_lazy", .explain = "group_by_weighted_rmse(bucket, lhs=lhs, rhs=rhs, weight=weight -> lhs_rhs_weighted_rmse_lazy)", .expected = &weighted_pair_rmse_expected },
        .{ .method = .weighted_mape, .output_name = "lhs_rhs_weighted_mape_lazy", .explain = "group_by_weighted_mape(bucket, lhs=lhs, rhs=rhs, weight=weight -> lhs_rhs_weighted_mape_lazy)", .expected = &weighted_pair_mape_expected },
        .{ .method = .weighted_smape, .output_name = "lhs_rhs_weighted_smape_lazy", .explain = "group_by_weighted_smape(bucket, lhs=lhs, rhs=rhs, weight=weight -> lhs_rhs_weighted_smape_lazy)", .expected = &weighted_pair_smape_expected },
        .{ .method = .weighted_covariance, .output_name = "lhs_rhs_weighted_cov_lazy", .explain = "group_by_weighted_covariance(bucket, lhs=lhs, rhs=rhs, weight=weight -> lhs_rhs_weighted_cov_lazy)", .expected = &weighted_pair_cov_expected },
        .{ .method = .weighted_correlation, .output_name = "lhs_rhs_weighted_corr_lazy", .explain = "group_by_weighted_correlation(bucket, lhs=lhs, rhs=rhs, weight=weight -> lhs_rhs_weighted_corr_lazy)", .expected = &weighted_pair_corr_expected },
        .{ .method = .weighted_beta, .output_name = "lhs_rhs_weighted_beta_lazy", .explain = "group_by_weighted_beta(bucket, lhs=lhs, rhs=rhs, weight=weight -> lhs_rhs_weighted_beta_lazy)", .expected = &weighted_pair_beta_expected },
    };
    for (weighted_pair_lazy_cases) |case| {
        var plan = try DeviceLazyFrame.init(gpa, weighted_table);
        defer plan.deinit();
        try switch (case.method) {
            .weighted_dot => plan.groupByWeightedDot("bucket", "lhs", "rhs", "weight", case.output_name),
            .weighted_cosine_similarity => plan.groupByWeightedCosine("bucket", "lhs", "rhs", "weight", case.output_name),
            .weighted_squared_euclidean_distance => plan.groupByWeightedSquaredEuclideanDistance("bucket", "lhs", "rhs", "weight", case.output_name),
            .weighted_euclidean_distance => plan.groupByWeightedEuclideanDistance("bucket", "lhs", "rhs", "weight", case.output_name),
            .weighted_manhattan_distance => plan.groupByWeightedManhattanDistance("bucket", "lhs", "rhs", "weight", case.output_name),
            .weighted_chebyshev_distance => plan.groupByWeightedChebyshevDistance("bucket", "lhs", "rhs", "weight", case.output_name),
            .weighted_canberra_distance => plan.groupByWeightedCanberraDistance("bucket", "lhs", "rhs", "weight", case.output_name),
            .weighted_bray_curtis_distance => plan.groupByWeightedBrayCurtisDistance("bucket", "lhs", "rhs", "weight", case.output_name),
            .weighted_mean_error => plan.groupByWeightedMeanError("bucket", "lhs", "rhs", "weight", case.output_name),
            .weighted_mae => plan.groupByWeightedMae("bucket", "lhs", "rhs", "weight", case.output_name),
            .weighted_mse => plan.groupByWeightedMse("bucket", "lhs", "rhs", "weight", case.output_name),
            .weighted_rmse => plan.groupByWeightedRmse("bucket", "lhs", "rhs", "weight", case.output_name),
            .weighted_mape => plan.groupByWeightedMape("bucket", "lhs", "rhs", "weight", case.output_name),
            .weighted_smape => plan.groupByWeightedSmape("bucket", "lhs", "rhs", "weight", case.output_name),
            .weighted_covariance => plan.groupByWeightedCov("bucket", "lhs", "rhs", "weight", case.output_name, 0.0),
            .weighted_correlation => plan.groupByWeightedCorr("bucket", "lhs", "rhs", "weight", case.output_name, 0.0),
            .weighted_beta => plan.groupByWeightedBeta("bucket", "lhs", "rhs", "weight", case.output_name, 0.0),
        };
        const explained = try plan.explain(gpa);
        defer gpa.free(explained);
        try std.testing.expect(std.mem.indexOf(u8, explained, case.explain) != null);
        var collected = try plan.collect();
        defer collected.deinit();
        try expectF64ColumnApproxOrNan(collected, gpa, case.output_name, case.expected);
    }

    var weighted_cov_on_plan = try DeviceLazyFrame.init(gpa, weighted_table);
    defer weighted_cov_on_plan.deinit();
    try weighted_cov_on_plan.groupByWeightedCovarianceOn(&.{"bucket"}, "lhs", "rhs", "weight", "lhs_rhs_weighted_cov_on_lazy", 0.0);
    const weighted_cov_on_explained = try weighted_cov_on_plan.explain(gpa);
    defer gpa.free(weighted_cov_on_explained);
    try std.testing.expect(std.mem.indexOf(u8, weighted_cov_on_explained, "group_by_weighted_covariance_on([bucket], lhs=lhs, rhs=rhs, weight=weight -> lhs_rhs_weighted_cov_on_lazy)") != null);
    var lazy_weighted_cov_on = try weighted_cov_on_plan.collect();
    defer lazy_weighted_cov_on.deinit();
    try expectF64ColumnApproxOrNan(lazy_weighted_cov_on, gpa, "lhs_rhs_weighted_cov_on_lazy", &weighted_pair_cov_expected);

    var mode_count_plan = try DeviceLazyFrame.init(gpa, mode_diag_table);
    defer mode_count_plan.deinit();
    try mode_count_plan.groupByModeCount("bucket", "label", "label_mode_count_lazy");
    const mode_count_explained = try mode_count_plan.explain(gpa);
    defer gpa.free(mode_count_explained);
    try std.testing.expect(std.mem.indexOf(u8, mode_count_explained, "group_by_mode_count(bucket, value=label -> label_mode_count_lazy)") != null);
    var lazy_mode_count = try mode_count_plan.collect();
    defer lazy_mode_count.deinit();
    const lazy_mode_count_values = try (try lazy_mode_count.column("label_mode_count_lazy")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_mode_count_values);
    try std.testing.expectEqualSlices(i64, &.{ 2, 2 }, lazy_mode_count_values);

    var mode_ratio_plan = try DeviceLazyFrame.init(gpa, mode_diag_table);
    defer mode_ratio_plan.deinit();
    try mode_ratio_plan.groupByModeRatioOn(&.{"bucket"}, "label", "label_mode_ratio_lazy");
    const mode_ratio_explained = try mode_ratio_plan.explain(gpa);
    defer gpa.free(mode_ratio_explained);
    try std.testing.expect(std.mem.indexOf(u8, mode_ratio_explained, "group_by_mode_ratio_on([bucket], value=label -> label_mode_ratio_lazy)") != null);
    var lazy_mode_ratio = try mode_ratio_plan.collect();
    defer lazy_mode_ratio.deinit();
    const lazy_mode_ratio_values = try (try lazy_mode_ratio.column("label_mode_ratio_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_mode_ratio_values);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_mode_ratio_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.4), lazy_mode_ratio_values[1], 1e-12);

    var mode_margin_plan = try DeviceLazyFrame.init(gpa, mode_diag_table);
    defer mode_margin_plan.deinit();
    try mode_margin_plan.groupByModeMargin("bucket", "label", "label_mode_margin_lazy");
    const mode_margin_explained = try mode_margin_plan.explain(gpa);
    defer gpa.free(mode_margin_explained);
    try std.testing.expect(std.mem.indexOf(u8, mode_margin_explained, "group_by_mode_margin(bucket, value=label -> label_mode_margin_lazy)") != null);
    var lazy_mode_margin = try mode_margin_plan.collect();
    defer lazy_mode_margin.deinit();
    const lazy_mode_margin_values = try (try lazy_mode_margin.column("label_mode_margin_lazy")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_mode_margin_values);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0 }, lazy_mode_margin_values);

    var mode_margin_ratio_plan = try DeviceLazyFrame.init(gpa, mode_diag_table);
    defer mode_margin_ratio_plan.deinit();
    try mode_margin_ratio_plan.groupByModeMarginRatioOn(&.{"bucket"}, "label", "label_mode_margin_ratio_lazy");
    const mode_margin_ratio_explained = try mode_margin_ratio_plan.explain(gpa);
    defer gpa.free(mode_margin_ratio_explained);
    try std.testing.expect(std.mem.indexOf(u8, mode_margin_ratio_explained, "group_by_mode_margin_ratio_on([bucket], value=label -> label_mode_margin_ratio_lazy)") != null);
    var lazy_mode_margin_ratio = try mode_margin_ratio_plan.collect();
    defer lazy_mode_margin_ratio.deinit();
    const lazy_mode_margin_ratio_values = try (try lazy_mode_margin_ratio.column("label_mode_margin_ratio_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_mode_margin_ratio_values);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), lazy_mode_margin_ratio_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_mode_margin_ratio_values[1], 1e-12);

    var entropy_group_plan = try DeviceLazyFrame.init(gpa, mode_diag_table);
    defer entropy_group_plan.deinit();
    try entropy_group_plan.groupByEntropyOn(&.{"bucket"}, "label", "label_entropy_lazy");
    const entropy_group_explained = try entropy_group_plan.explain(gpa);
    defer gpa.free(entropy_group_explained);
    try std.testing.expect(std.mem.indexOf(u8, entropy_group_explained, "group_by_entropy_on([bucket], value=label -> label_entropy_lazy)") != null);
    var lazy_entropy_group = try entropy_group_plan.collect();
    defer lazy_entropy_group.deinit();
    const lazy_entropy_group_values = try (try lazy_entropy_group.column("label_entropy_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_entropy_group_values);
    try std.testing.expectApproxEqAbs(group_entropy_1, lazy_entropy_group_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(group_entropy_2, lazy_entropy_group_values[1], 1e-12);

    var gini_group_plan = try DeviceLazyFrame.init(gpa, mode_diag_table);
    defer gini_group_plan.deinit();
    try gini_group_plan.groupByGiniOn(&.{"bucket"}, "label", "label_gini_lazy");
    const gini_group_explained = try gini_group_plan.explain(gpa);
    defer gpa.free(gini_group_explained);
    try std.testing.expect(std.mem.indexOf(u8, gini_group_explained, "group_by_gini_impurity_on([bucket], value=label -> label_gini_lazy)") != null);
    var lazy_gini_group = try gini_group_plan.collect();
    defer lazy_gini_group.deinit();
    const lazy_gini_group_values = try (try lazy_gini_group.column("label_gini_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_gini_group_values);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0) - group_concentration_1, lazy_gini_group_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0) - group_concentration_2, lazy_gini_group_values[1], 1e-12);

    var perplexity_group_plan = try DeviceLazyFrame.init(gpa, mode_diag_table);
    defer perplexity_group_plan.deinit();
    try perplexity_group_plan.groupByPerplexity("bucket", "label", "label_perplexity_lazy");
    const perplexity_group_explained = try perplexity_group_plan.explain(gpa);
    defer gpa.free(perplexity_group_explained);
    try std.testing.expect(std.mem.indexOf(u8, perplexity_group_explained, "group_by_perplexity(bucket, value=label -> label_perplexity_lazy)") != null);
    var lazy_perplexity_group = try perplexity_group_plan.collect();
    defer lazy_perplexity_group.deinit();
    const lazy_perplexity_group_values = try (try lazy_perplexity_group.column("label_perplexity_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_perplexity_group_values);
    try std.testing.expectApproxEqAbs(std.math.exp(group_entropy_1), lazy_perplexity_group_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.exp(group_entropy_2), lazy_perplexity_group_values[1], 1e-12);

    var inverse_group_plan = try DeviceLazyFrame.init(gpa, mode_diag_table);
    defer inverse_group_plan.deinit();
    try inverse_group_plan.groupByInverseSimpsonOn(&.{"bucket"}, "label", "label_inverse_lazy");
    const inverse_group_explained = try inverse_group_plan.explain(gpa);
    defer gpa.free(inverse_group_explained);
    try std.testing.expect(std.mem.indexOf(u8, inverse_group_explained, "group_by_inverse_simpson_on([bucket], value=label -> label_inverse_lazy)") != null);
    var lazy_inverse_group = try inverse_group_plan.collect();
    defer lazy_inverse_group.deinit();
    const lazy_inverse_group_values = try (try lazy_inverse_group.column("label_inverse_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_inverse_group_values);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0) / group_concentration_1, lazy_inverse_group_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0) / group_concentration_2, lazy_inverse_group_values[1], 1e-12);

    var concentration_group_plan = try DeviceLazyFrame.init(gpa, mode_diag_table);
    defer concentration_group_plan.deinit();
    try concentration_group_plan.groupByConcentration("bucket", "label", "label_concentration_lazy");
    const concentration_group_explained = try concentration_group_plan.explain(gpa);
    defer gpa.free(concentration_group_explained);
    try std.testing.expect(std.mem.indexOf(u8, concentration_group_explained, "group_by_simpson_concentration(bucket, value=label -> label_concentration_lazy)") != null);
    var lazy_concentration_group = try concentration_group_plan.collect();
    defer lazy_concentration_group.deinit();
    const lazy_concentration_group_values = try (try lazy_concentration_group.column("label_concentration_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_concentration_group_values);
    try std.testing.expectApproxEqAbs(group_concentration_1, lazy_concentration_group_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(group_concentration_2, lazy_concentration_group_values[1], 1e-12);

    var evenness_group_plan = try DeviceLazyFrame.init(gpa, mode_diag_table);
    defer evenness_group_plan.deinit();
    try evenness_group_plan.groupByEvennessOn(&.{"bucket"}, "label", "label_evenness_lazy");
    const evenness_group_explained = try evenness_group_plan.explain(gpa);
    defer gpa.free(evenness_group_explained);
    try std.testing.expect(std.mem.indexOf(u8, evenness_group_explained, "group_by_evenness_on([bucket], value=label -> label_evenness_lazy)") != null);
    var lazy_evenness_group = try evenness_group_plan.collect();
    defer lazy_evenness_group.deinit();
    const lazy_evenness_group_values = try (try lazy_evenness_group.column("label_evenness_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_evenness_group_values);
    try std.testing.expectApproxEqAbs(group_entropy_1 / group_distinct_log, lazy_evenness_group_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(group_entropy_2 / group_distinct_log, lazy_evenness_group_values[1], 1e-12);

    var gini_mean_diff_plan = try DeviceLazyFrame.init(gpa, mode_diag_table);
    defer gini_mean_diff_plan.deinit();
    try gini_mean_diff_plan.groupByGiniMeanDiffOn(&.{"bucket"}, "label", "label_gini_mean_diff_lazy");
    const gini_mean_diff_explained = try gini_mean_diff_plan.explain(gpa);
    defer gpa.free(gini_mean_diff_explained);
    try std.testing.expect(std.mem.indexOf(u8, gini_mean_diff_explained, "group_by_gini_mean_diff_on([bucket], value=label -> label_gini_mean_diff_lazy)") != null);
    var lazy_gini_mean_diff = try gini_mean_diff_plan.collect();
    defer lazy_gini_mean_diff.deinit();
    const lazy_gini_mean_diff_values = try (try lazy_gini_mean_diff.column("label_gini_mean_diff_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_gini_mean_diff_values);
    try std.testing.expectApproxEqAbs(@as(f64, 11.0 / 6.0), lazy_gini_mean_diff_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_gini_mean_diff_values[1], 1e-12);

    var gini_coeff_plan = try DeviceLazyFrame.init(gpa, mode_diag_table);
    defer gini_coeff_plan.deinit();
    try gini_coeff_plan.groupByGiniCoefficient("bucket", "label", "label_gini_coeff_lazy");
    const gini_coeff_explained = try gini_coeff_plan.explain(gpa);
    defer gpa.free(gini_coeff_explained);
    try std.testing.expect(std.mem.indexOf(u8, gini_coeff_explained, "group_by_gini_coefficient(bucket, value=label -> label_gini_coeff_lazy)") != null);
    var lazy_gini_coeff = try gini_coeff_plan.collect();
    defer lazy_gini_coeff.deinit();
    const lazy_gini_coeff_values = try (try lazy_gini_coeff.column("label_gini_coeff_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_gini_coeff_values);
    try std.testing.expectApproxEqAbs(@as(f64, 11.0 / 75.0), lazy_gini_coeff_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 18.0), lazy_gini_coeff_values[1], 1e-12);

    var mean_abs_dev_plan = try DeviceLazyFrame.init(gpa, mode_diag_table);
    defer mean_abs_dev_plan.deinit();
    try mean_abs_dev_plan.groupByMeanAbsDevOn(&.{"bucket"}, "label", "label_mean_abs_dev_lazy");
    const mean_abs_dev_explained = try mean_abs_dev_plan.explain(gpa);
    defer gpa.free(mean_abs_dev_explained);
    try std.testing.expect(std.mem.indexOf(u8, mean_abs_dev_explained, "group_by_mean_abs_dev_on([bucket], value=label -> label_mean_abs_dev_lazy)") != null);
    var lazy_mean_abs_dev = try mean_abs_dev_plan.collect();
    defer lazy_mean_abs_dev.deinit();
    const lazy_mean_abs_dev_values = try (try lazy_mean_abs_dev.column("label_mean_abs_dev_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_mean_abs_dev_values);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 4.0), lazy_mean_abs_dev_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 16.0 / 25.0), lazy_mean_abs_dev_values[1], 1e-12);

    var mean_abs_dev_ratio_plan = try DeviceLazyFrame.init(gpa, mode_diag_table);
    defer mean_abs_dev_ratio_plan.deinit();
    try mean_abs_dev_ratio_plan.groupByMeanAbsDevRatio("bucket", "label", "label_mean_abs_dev_ratio_lazy");
    const mean_abs_dev_ratio_explained = try mean_abs_dev_ratio_plan.explain(gpa);
    defer gpa.free(mean_abs_dev_ratio_explained);
    try std.testing.expect(std.mem.indexOf(u8, mean_abs_dev_ratio_explained, "group_by_mean_abs_dev_ratio(bucket, value=label -> label_mean_abs_dev_ratio_lazy)") != null);
    var lazy_mean_abs_dev_ratio = try mean_abs_dev_ratio_plan.collect();
    defer lazy_mean_abs_dev_ratio.deinit();
    const lazy_mean_abs_dev_ratio_values = try (try lazy_mean_abs_dev_ratio.column("label_mean_abs_dev_ratio_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_mean_abs_dev_ratio_values);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 5.0), lazy_mean_abs_dev_ratio_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 16.0 / 45.0), lazy_mean_abs_dev_ratio_values[1], 1e-12);

    var median_sales = try table.groupByMedian("store", "sales", "sales_median");
    defer median_sales.deinit();
    const median_sales_values = try (try median_sales.column("sales_median")).f64.toOwnedSlice(gpa);
    defer gpa.free(median_sales_values);
    try std.testing.expectEqualSlices(f64, &.{ 7.5, 7.0 }, median_sales_values);

    var q1_sales = try table.groupByQuantile("store", "sales", "sales_q1", 0.25);
    defer q1_sales.deinit();
    const q1_sales_values = try (try q1_sales.column("sales_q1")).f64.toOwnedSlice(gpa);
    defer gpa.free(q1_sales_values);
    try std.testing.expectEqualSlices(f64, &.{ 4.75, 5.0 }, q1_sales_values);

    var tail_key = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 1, 1, 1, 1, 2, 2, 2, 2, 2 }, .cpu);
    defer tail_key.deinit();
    var tail_value = try DeviceColumn.fromSlice(f64, gpa, &.{ 1.0, 2.0, 3.0, 100.0, 200.0, 10.0, 20.0, 30.0, 40.0, 1000.0 }, .cpu);
    defer tail_value.deinit();
    var tail_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "bucket", .data = tail_key },
        .{ .name = "value", .data = tail_value },
    });
    defer tail_table.deinit();

    var trimmed_tail = try tail_table.groupByTrimmedMean("bucket", "value", "value_trimmed", 0.2);
    defer trimmed_tail.deinit();
    const trimmed_tail_keys = try (try trimmed_tail.column("bucket")).i32.toOwnedSlice(gpa);
    defer gpa.free(trimmed_tail_keys);
    const trimmed_tail_values = try (try trimmed_tail.column("value_trimmed")).f64.toOwnedSlice(gpa);
    defer gpa.free(trimmed_tail_values);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2 }, trimmed_tail_keys);
    try std.testing.expectApproxEqAbs(@as(f64, 35.0), trimmed_tail_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 30.0), trimmed_tail_values[1], 1e-12);

    var winsorized_tail = try tail_table.groupByWinsorizedMeanOn(&.{"bucket"}, "value", "value_winsorized", 0.2);
    defer winsorized_tail.deinit();
    const winsorized_tail_values = try (try winsorized_tail.column("value_winsorized")).f64.toOwnedSlice(gpa);
    defer gpa.free(winsorized_tail_values);
    try std.testing.expectApproxEqAbs(@as(f64, 41.4), winsorized_tail_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 30.0), winsorized_tail_values[1], 1e-12);

    try std.testing.expectError(error.InvalidShape, tail_table.groupByTrimmedMean("bucket", "value", "bad_trimmed", 0.5));
    try std.testing.expectError(error.InvalidShape, tail_table.groupByWinsorizedMean("bucket", "value", "bad_winsorized", -0.01));

    var trimmed_tail_plan = try DeviceLazyFrame.init(gpa, tail_table);
    defer trimmed_tail_plan.deinit();
    try trimmed_tail_plan.groupByTrimmedMeanOn(&.{"bucket"}, "value", "value_trimmed_lazy", 0.2);
    const trimmed_tail_explained = try trimmed_tail_plan.explain(gpa);
    defer gpa.free(trimmed_tail_explained);
    try std.testing.expect(std.mem.indexOf(u8, trimmed_tail_explained, "group_by_trimmed_mean_on([bucket], value=value, trim_fraction=0.2 -> value_trimmed_lazy)") != null);
    var lazy_trimmed_tail = try trimmed_tail_plan.collect();
    defer lazy_trimmed_tail.deinit();
    const lazy_trimmed_tail_values = try (try lazy_trimmed_tail.column("value_trimmed_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_trimmed_tail_values);
    try std.testing.expectApproxEqAbs(@as(f64, 35.0), lazy_trimmed_tail_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 30.0), lazy_trimmed_tail_values[1], 1e-12);

    var winsorized_tail_plan = try DeviceLazyFrame.init(gpa, tail_table);
    defer winsorized_tail_plan.deinit();
    try winsorized_tail_plan.groupByWinsorizedMeanOn(&.{"bucket"}, "value", "value_winsorized_lazy", 0.2);
    const winsorized_tail_explained = try winsorized_tail_plan.explain(gpa);
    defer gpa.free(winsorized_tail_explained);
    try std.testing.expect(std.mem.indexOf(u8, winsorized_tail_explained, "group_by_winsorized_mean_on([bucket], value=value, winsor_fraction=0.2 -> value_winsorized_lazy)") != null);
    var lazy_winsorized_tail = try winsorized_tail_plan.collect();
    defer lazy_winsorized_tail.deinit();
    const lazy_winsorized_tail_values = try (try lazy_winsorized_tail.column("value_winsorized_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_winsorized_tail_values);
    try std.testing.expectApproxEqAbs(@as(f64, 41.4), lazy_winsorized_tail_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 30.0), lazy_winsorized_tail_values[1], 1e-12);

    var invalid_trimmed_tail_plan = try DeviceLazyFrame.init(gpa, tail_table);
    defer invalid_trimmed_tail_plan.deinit();
    try invalid_trimmed_tail_plan.groupByTrimmedMean("bucket", "value", "bad_trimmed_lazy", 0.5);
    try std.testing.expectError(error.InvalidShape, invalid_trimmed_tail_plan.collect());

    var invalid_winsorized_tail_plan = try DeviceLazyFrame.init(gpa, tail_table);
    defer invalid_winsorized_tail_plan.deinit();
    try invalid_winsorized_tail_plan.groupByWinsorizedMean("bucket", "value", "bad_winsorized_lazy", -0.01);
    try std.testing.expectError(error.InvalidShape, invalid_winsorized_tail_plan.collect());

    var variance_sales = try table.groupByVariance("store", "sales", "sales_variance_simple");
    defer variance_sales.deinit();
    const variance_sales_values = try (try variance_sales.column("sales_variance_simple")).f64.toOwnedSlice(gpa);
    defer gpa.free(variance_sales_values);
    try std.testing.expectApproxEqAbs(@as(f64, 30.25), variance_sales_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 16.0), variance_sales_values[1], 1e-12);

    var stddev_sales = try table.groupByStddev("store", "sales", "sales_stddev_simple");
    defer stddev_sales.deinit();
    const stddev_sales_values = try (try stddev_sales.column("sales_stddev_simple")).f64.toOwnedSlice(gpa);
    defer gpa.free(stddev_sales_values);
    try std.testing.expectApproxEqAbs(@as(f64, 5.5), stddev_sales_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), stddev_sales_values[1], 1e-12);

    var sem_sales = try table.groupBySem("store", "sales", "sales_sem_simple");
    defer sem_sales.deinit();
    const sem_sales_values = try (try sem_sales.column("sales_sem_simple")).f64.toOwnedSlice(gpa);
    defer gpa.free(sem_sales_values);
    try std.testing.expectApproxEqAbs(@as(f64, 5.5 / std.math.sqrt(2.0)), sem_sales_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0 / std.math.sqrt(2.0)), sem_sales_values[1], 1e-12);

    var cv_sales = try table.groupByCv("store", "sales", "sales_cv_simple");
    defer cv_sales.deinit();
    const cv_sales_values = try (try cv_sales.column("sales_cv_simple")).f64.toOwnedSlice(gpa);
    defer gpa.free(cv_sales_values);
    try std.testing.expectApproxEqAbs(@as(f64, 5.5 / 7.5), cv_sales_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0 / 7.0), cv_sales_values[1], 1e-12);

    var fano_sales = try table.groupByIndexOfDispersion("store", "sales", "sales_fano_simple");
    defer fano_sales.deinit();
    const fano_sales_values = try (try fano_sales.column("sales_fano_simple")).f64.toOwnedSlice(gpa);
    defer gpa.free(fano_sales_values);
    try std.testing.expectApproxEqAbs(@as(f64, 30.25 / 7.5), fano_sales_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 16.0 / 7.0), fano_sales_values[1], 1e-12);

    var zero_mean_key = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 1 }, .cpu);
    defer zero_mean_key.deinit();
    var zero_mean_value = try DeviceColumn.fromSlice(f64, gpa, &.{ -1.0, 1.0 }, .cpu);
    defer zero_mean_value.deinit();
    var zero_mean_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "bucket", .data = zero_mean_key },
        .{ .name = "value", .data = zero_mean_value },
    });
    defer zero_mean_table.deinit();
    var zero_mean_fano = try zero_mean_table.groupByFano("bucket", "value", "value_fano");
    defer zero_mean_fano.deinit();
    const zero_mean_fano_values = try (try zero_mean_fano.column("value_fano")).f64.toOwnedSlice(gpa);
    defer gpa.free(zero_mean_fano_values);
    try std.testing.expect(std.math.isNan(zero_mean_fano_values[0]));

    var magnitude_key = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 1, 2, 2 }, .cpu);
    defer magnitude_key.deinit();
    var signed_delta = try DeviceColumn.fromSlice(f64, gpa, &.{ -3.0, 4.0, -5.0, 12.0 }, .cpu);
    defer signed_delta.deinit();
    var magnitude_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "bucket", .data = magnitude_key },
        .{ .name = "delta", .data = signed_delta },
    });
    defer magnitude_table.deinit();

    var mean_abs_delta = try magnitude_table.groupByMeanAbs("bucket", "delta", "delta_mean_abs");
    defer mean_abs_delta.deinit();
    const mean_abs_delta_values = try (try mean_abs_delta.column("delta_mean_abs")).f64.toOwnedSlice(gpa);
    defer gpa.free(mean_abs_delta_values);
    try std.testing.expectApproxEqAbs(@as(f64, 3.5), mean_abs_delta_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 8.5), mean_abs_delta_values[1], 1e-12);

    var mean_square_delta = try magnitude_table.groupByMeanSq("bucket", "delta", "delta_mean_square");
    defer mean_square_delta.deinit();
    const mean_square_delta_values = try (try mean_square_delta.column("delta_mean_square")).f64.toOwnedSlice(gpa);
    defer gpa.free(mean_square_delta_values);
    try std.testing.expectApproxEqAbs(@as(f64, 12.5), mean_square_delta_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 84.5), mean_square_delta_values[1], 1e-12);

    var rms_delta = try magnitude_table.groupByRMS("bucket", "delta", "delta_rms");
    defer rms_delta.deinit();
    const rms_delta_values = try (try rms_delta.column("delta_rms")).f64.toOwnedSlice(gpa);
    defer gpa.free(rms_delta_values);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 12.5)), rms_delta_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 84.5)), rms_delta_values[1], 1e-12);

    var l1_delta = try magnitude_table.groupByL1Norm("bucket", "delta", "delta_l1");
    defer l1_delta.deinit();
    const l1_delta_values = try (try l1_delta.column("delta_l1")).f64.toOwnedSlice(gpa);
    defer gpa.free(l1_delta_values);
    try std.testing.expectApproxEqAbs(@as(f64, 7.0), l1_delta_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 17.0), l1_delta_values[1], 1e-12);

    var l2_delta = try magnitude_table.groupByL2Norm("bucket", "delta", "delta_l2");
    defer l2_delta.deinit();
    const l2_delta_values = try (try l2_delta.column("delta_l2")).f64.toOwnedSlice(gpa);
    defer gpa.free(l2_delta_values);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), l2_delta_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 13.0), l2_delta_values[1], 1e-12);

    var max_abs_delta = try magnitude_table.groupByMaxAbs("bucket", "delta", "delta_max_abs");
    defer max_abs_delta.deinit();
    const max_abs_delta_values = try (try max_abs_delta.column("delta_max_abs")).f64.toOwnedSlice(gpa);
    defer gpa.free(max_abs_delta_values);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), max_abs_delta_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 12.0), max_abs_delta_values[1], 1e-12);

    var min_abs_delta = try magnitude_table.groupByMinAbs("bucket", "delta", "delta_min_abs");
    defer min_abs_delta.deinit();
    const min_abs_delta_values = try (try min_abs_delta.column("delta_min_abs")).f64.toOwnedSlice(gpa);
    defer gpa.free(min_abs_delta_values);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), min_abs_delta_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), min_abs_delta_values[1], 1e-12);

    var hhi_delta = try magnitude_table.groupByHerfindahl("bucket", "delta", "delta_hhi");
    defer hhi_delta.deinit();
    const hhi_delta_values = try (try hhi_delta.column("delta_hhi")).f64.toOwnedSlice(gpa);
    defer gpa.free(hhi_delta_values);
    try std.testing.expectApproxEqAbs(@as(f64, 25.0 / 49.0), hhi_delta_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 169.0 / 289.0), hhi_delta_values[1], 1e-12);

    var normalized_hhi_delta = try magnitude_table.groupByAbsNormalizedHhi("bucket", "delta", "delta_normalized_hhi");
    defer normalized_hhi_delta.deinit();
    const normalized_hhi_delta_values = try (try normalized_hhi_delta.column("delta_normalized_hhi")).f64.toOwnedSlice(gpa);
    defer gpa.free(normalized_hhi_delta_values);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 49.0), normalized_hhi_delta_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 49.0 / 289.0), normalized_hhi_delta_values[1], 1e-12);

    var sparsity_delta = try magnitude_table.groupByMagnitudeSparsity("bucket", "delta", "delta_sparsity");
    defer sparsity_delta.deinit();
    const sparsity_delta_values = try (try sparsity_delta.column("delta_sparsity")).f64.toOwnedSlice(gpa);
    defer gpa.free(sparsity_delta_values);
    try std.testing.expectApproxEqAbs((std.math.sqrt(@as(f64, 2.0)) - @as(f64, 7.0 / 5.0)) / (std.math.sqrt(@as(f64, 2.0)) - 1.0), sparsity_delta_values[0], 1e-12);
    try std.testing.expectApproxEqAbs((std.math.sqrt(@as(f64, 2.0)) - @as(f64, 17.0 / 13.0)) / (std.math.sqrt(@as(f64, 2.0)) - 1.0), sparsity_delta_values[1], 1e-12);

    var inverse_delta = try magnitude_table.groupByAbsInverseSimpson("bucket", "delta", "delta_inverse");
    defer inverse_delta.deinit();
    const inverse_delta_values = try (try inverse_delta.column("delta_inverse")).f64.toOwnedSlice(gpa);
    defer gpa.free(inverse_delta_values);
    try std.testing.expectApproxEqAbs(@as(f64, 49.0 / 25.0), inverse_delta_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 289.0 / 169.0), inverse_delta_values[1], 1e-12);

    var simpson_evenness_delta = try magnitude_table.groupByMagnitudeSimpsonEvenness("bucket", "delta", "delta_simpson_evenness");
    defer simpson_evenness_delta.deinit();
    const simpson_evenness_delta_values = try (try simpson_evenness_delta.column("delta_simpson_evenness")).f64.toOwnedSlice(gpa);
    defer gpa.free(simpson_evenness_delta_values);
    try std.testing.expectApproxEqAbs(@as(f64, 49.0 / 50.0), simpson_evenness_delta_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 289.0 / 338.0), simpson_evenness_delta_values[1], 1e-12);

    var dominance_delta = try magnitude_table.groupByAbsDominance("bucket", "delta", "delta_dominance");
    defer dominance_delta.deinit();
    const dominance_delta_values = try (try dominance_delta.column("delta_dominance")).f64.toOwnedSlice(gpa);
    defer gpa.free(dominance_delta_values);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0 / 7.0), dominance_delta_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 12.0 / 17.0), dominance_delta_values[1], 1e-12);

    var dominance_margin_delta = try magnitude_table.groupByMagnitudeDominanceMargin("bucket", "delta", "delta_dominance_margin");
    defer dominance_margin_delta.deinit();
    const dominance_margin_delta_values = try (try dominance_margin_delta.column("delta_dominance_margin")).f64.toOwnedSlice(gpa);
    defer gpa.free(dominance_margin_delta_values);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 7.0), dominance_margin_delta_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 7.0 / 17.0), dominance_margin_delta_values[1], 1e-12);

    const magnitude_entropy_1 = std.math.log(f64, std.math.e, @as(f64, 7.0)) - (@as(f64, 3.0) * std.math.log(f64, std.math.e, @as(f64, 3.0)) + @as(f64, 4.0) * std.math.log(f64, std.math.e, @as(f64, 4.0))) / @as(f64, 7.0);
    const magnitude_entropy_2 = std.math.log(f64, std.math.e, @as(f64, 17.0)) - (@as(f64, 5.0) * std.math.log(f64, std.math.e, @as(f64, 5.0)) + @as(f64, 12.0) * std.math.log(f64, std.math.e, @as(f64, 12.0))) / @as(f64, 17.0);

    var magnitude_entropy_delta = try magnitude_table.groupByAbsEntropy("bucket", "delta", "delta_magnitude_entropy");
    defer magnitude_entropy_delta.deinit();
    const magnitude_entropy_delta_values = try (try magnitude_entropy_delta.column("delta_magnitude_entropy")).f64.toOwnedSlice(gpa);
    defer gpa.free(magnitude_entropy_delta_values);
    try std.testing.expectApproxEqAbs(magnitude_entropy_1, magnitude_entropy_delta_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(magnitude_entropy_2, magnitude_entropy_delta_values[1], 1e-12);

    var magnitude_perplexity_delta = try magnitude_table.groupByMagnitudePerplexity("bucket", "delta", "delta_magnitude_perplexity");
    defer magnitude_perplexity_delta.deinit();
    const magnitude_perplexity_delta_values = try (try magnitude_perplexity_delta.column("delta_magnitude_perplexity")).f64.toOwnedSlice(gpa);
    defer gpa.free(magnitude_perplexity_delta_values);
    try std.testing.expectApproxEqAbs(std.math.exp(magnitude_entropy_1), magnitude_perplexity_delta_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.exp(magnitude_entropy_2), magnitude_perplexity_delta_values[1], 1e-12);

    var magnitude_evenness_delta = try magnitude_table.groupByAbsEvenness("bucket", "delta", "delta_magnitude_evenness");
    defer magnitude_evenness_delta.deinit();
    const magnitude_evenness_delta_values = try (try magnitude_evenness_delta.column("delta_magnitude_evenness")).f64.toOwnedSlice(gpa);
    defer gpa.free(magnitude_evenness_delta_values);
    try std.testing.expectApproxEqAbs(magnitude_entropy_1 / std.math.log(f64, std.math.e, @as(f64, 2.0)), magnitude_evenness_delta_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(magnitude_entropy_2 / std.math.log(f64, std.math.e, @as(f64, 2.0)), magnitude_evenness_delta_values[1], 1e-12);

    var magnitude_zero_key = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 1 }, .cpu);
    defer magnitude_zero_key.deinit();
    var magnitude_zero_value = try DeviceColumn.fromSlice(f64, gpa, &.{ 0.0, 0.0 }, .cpu);
    defer magnitude_zero_value.deinit();
    var magnitude_zero_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "bucket", .data = magnitude_zero_key },
        .{ .name = "value", .data = magnitude_zero_value },
    });
    defer magnitude_zero_table.deinit();
    var zero_hhi = try magnitude_zero_table.groupByHhi("bucket", "value", "value_hhi");
    defer zero_hhi.deinit();
    const zero_hhi_values = try (try zero_hhi.column("value_hhi")).f64.toOwnedSlice(gpa);
    defer gpa.free(zero_hhi_values);
    try std.testing.expect(std.math.isNan(zero_hhi_values[0]));

    var magnitude_lazy_plan = try DeviceLazyFrame.init(gpa, magnitude_table);
    defer magnitude_lazy_plan.deinit();
    try magnitude_lazy_plan.groupByHhiOn(&.{"bucket"}, "delta", "delta_hhi_lazy");
    const magnitude_lazy_explained = try magnitude_lazy_plan.explain(gpa);
    defer gpa.free(magnitude_lazy_explained);
    try std.testing.expect(std.mem.indexOf(u8, magnitude_lazy_explained, "group_by_hhi_on([bucket], value=delta -> delta_hhi_lazy)") != null);
    var lazy_hhi_delta = try magnitude_lazy_plan.collect();
    defer lazy_hhi_delta.deinit();
    const lazy_hhi_delta_values = try (try lazy_hhi_delta.column("delta_hhi_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_hhi_delta_values);
    try std.testing.expectApproxEqAbs(@as(f64, 25.0 / 49.0), lazy_hhi_delta_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 169.0 / 289.0), lazy_hhi_delta_values[1], 1e-12);

    var magnitude_alias_plan = try DeviceLazyFrame.init(gpa, magnitude_table);
    defer magnitude_alias_plan.deinit();
    try magnitude_alias_plan.groupByAbsNormalizedHhi("bucket", "delta", "delta_normalized_hhi_lazy");
    try magnitude_alias_plan.groupByMagnitudeSparsity("bucket", "delta", "delta_sparsity_lazy");
    try magnitude_alias_plan.groupByAbsInverseSimpson("bucket", "delta", "delta_inverse_lazy");
    try magnitude_alias_plan.groupByMagnitudeSimpsonEvenness("bucket", "delta", "delta_simpson_evenness_lazy");
    try magnitude_alias_plan.groupByAbsDominance("bucket", "delta", "delta_dominance_lazy");
    try magnitude_alias_plan.groupByMagnitudeDominanceMargin("bucket", "delta", "delta_dominance_margin_lazy");
    try magnitude_alias_plan.groupByAbsEntropy("bucket", "delta", "delta_magnitude_entropy_lazy");
    try magnitude_alias_plan.groupByMagnitudePerplexity("bucket", "delta", "delta_magnitude_perplexity_lazy");
    try magnitude_alias_plan.groupByAbsEvenness("bucket", "delta", "delta_magnitude_evenness_lazy");
    const magnitude_alias_explained = try magnitude_alias_plan.explain(gpa);
    defer gpa.free(magnitude_alias_explained);
    try std.testing.expect(std.mem.indexOf(u8, magnitude_alias_explained, "group_by_magnitude_normalized_hhi(bucket, value=delta -> delta_normalized_hhi_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, magnitude_alias_explained, "group_by_magnitude_sparsity(bucket, value=delta -> delta_sparsity_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, magnitude_alias_explained, "group_by_magnitude_evenness(bucket, value=delta -> delta_magnitude_evenness_lazy)") != null);

    var magnitude_variance_delta = try magnitude_table.groupByAbsVariance("bucket", "delta", "delta_magnitude_variance");
    defer magnitude_variance_delta.deinit();
    const magnitude_variance_delta_values = try (try magnitude_variance_delta.column("delta_magnitude_variance")).f64.toOwnedSlice(gpa);
    defer gpa.free(magnitude_variance_delta_values);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), magnitude_variance_delta_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 12.25), magnitude_variance_delta_values[1], 1e-12);

    var magnitude_std_delta = try magnitude_table.groupByMagnitudeStddev("bucket", "delta", "delta_magnitude_stddev");
    defer magnitude_std_delta.deinit();
    const magnitude_std_delta_values = try (try magnitude_std_delta.column("delta_magnitude_stddev")).f64.toOwnedSlice(gpa);
    defer gpa.free(magnitude_std_delta_values);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), magnitude_std_delta_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.5), magnitude_std_delta_values[1], 1e-12);

    var magnitude_sem_delta = try magnitude_table.groupByAbsSem("bucket", "delta", "delta_magnitude_sem");
    defer magnitude_sem_delta.deinit();
    const magnitude_sem_delta_values = try (try magnitude_sem_delta.column("delta_magnitude_sem")).f64.toOwnedSlice(gpa);
    defer gpa.free(magnitude_sem_delta_values);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5 / std.math.sqrt(2.0)), magnitude_sem_delta_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.5 / std.math.sqrt(2.0)), magnitude_sem_delta_values[1], 1e-12);

    var magnitude_cv_delta = try magnitude_table.groupByAbsCv("bucket", "delta", "delta_magnitude_cv");
    defer magnitude_cv_delta.deinit();
    const magnitude_cv_delta_values = try (try magnitude_cv_delta.column("delta_magnitude_cv")).f64.toOwnedSlice(gpa);
    defer gpa.free(magnitude_cv_delta_values);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 7.0), magnitude_cv_delta_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 7.0 / 17.0), magnitude_cv_delta_values[1], 1e-12);

    var magnitude_fano_delta = try magnitude_table.groupByMagnitudeIndexOfDispersion("bucket", "delta", "delta_magnitude_fano");
    defer magnitude_fano_delta.deinit();
    const magnitude_fano_delta_values = try (try magnitude_fano_delta.column("delta_magnitude_fano")).f64.toOwnedSlice(gpa);
    defer gpa.free(magnitude_fano_delta_values);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 14.0), magnitude_fano_delta_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 49.0 / 34.0), magnitude_fano_delta_values[1], 1e-12);

    var magnitude_skew_delta = try magnitude_table.groupByAbsSkewness("bucket", "delta", "delta_magnitude_skew");
    defer magnitude_skew_delta.deinit();
    const magnitude_skew_delta_values = try (try magnitude_skew_delta.column("delta_magnitude_skew")).f64.toOwnedSlice(gpa);
    defer gpa.free(magnitude_skew_delta_values);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), magnitude_skew_delta_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), magnitude_skew_delta_values[1], 1e-12);

    var magnitude_kurt_delta = try magnitude_table.groupByMagnitudeKurtosis("bucket", "delta", "delta_magnitude_kurt");
    defer magnitude_kurt_delta.deinit();
    const magnitude_kurt_delta_values = try (try magnitude_kurt_delta.column("delta_magnitude_kurt")).f64.toOwnedSlice(gpa);
    defer gpa.free(magnitude_kurt_delta_values);
    try std.testing.expectApproxEqAbs(@as(f64, -2.0), magnitude_kurt_delta_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -2.0), magnitude_kurt_delta_values[1], 1e-12);

    var magnitude_variance_plan = try DeviceLazyFrame.init(gpa, magnitude_table);
    defer magnitude_variance_plan.deinit();
    try magnitude_variance_plan.groupByMagnitudeVarianceOn(&.{"bucket"}, "delta", "delta_magnitude_variance_lazy");
    const magnitude_variance_explained = try magnitude_variance_plan.explain(gpa);
    defer gpa.free(magnitude_variance_explained);
    try std.testing.expect(std.mem.indexOf(u8, magnitude_variance_explained, "group_by_magnitude_variance_on([bucket], value=delta -> delta_magnitude_variance_lazy)") != null);
    var lazy_magnitude_variance = try magnitude_variance_plan.collect();
    defer lazy_magnitude_variance.deinit();
    const lazy_magnitude_variance_values = try (try lazy_magnitude_variance.column("delta_magnitude_variance_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_magnitude_variance_values);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), lazy_magnitude_variance_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 12.25), lazy_magnitude_variance_values[1], 1e-12);

    var magnitude_moment_alias_plan = try DeviceLazyFrame.init(gpa, magnitude_table);
    defer magnitude_moment_alias_plan.deinit();
    try magnitude_moment_alias_plan.groupByAbsStddev("bucket", "delta", "delta_magnitude_stddev_lazy");
    try magnitude_moment_alias_plan.groupByMagnitudeSem("bucket", "delta", "delta_magnitude_sem_lazy");
    try magnitude_moment_alias_plan.groupByAbsCv("bucket", "delta", "delta_magnitude_cv_lazy");
    try magnitude_moment_alias_plan.groupByAbsFano("bucket", "delta", "delta_magnitude_fano_lazy");
    try magnitude_moment_alias_plan.groupByMagnitudeSkewness("bucket", "delta", "delta_magnitude_skew_lazy");
    try magnitude_moment_alias_plan.groupByAbsKurtosis("bucket", "delta", "delta_magnitude_kurt_lazy");
    const magnitude_moment_alias_explained = try magnitude_moment_alias_plan.explain(gpa);
    defer gpa.free(magnitude_moment_alias_explained);
    try std.testing.expect(std.mem.indexOf(u8, magnitude_moment_alias_explained, "group_by_magnitude_stddev(bucket, value=delta -> delta_magnitude_stddev_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, magnitude_moment_alias_explained, "group_by_magnitude_cv(bucket, value=delta -> delta_magnitude_cv_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, magnitude_moment_alias_explained, "group_by_magnitude_kurtosis(bucket, value=delta -> delta_magnitude_kurt_lazy)") != null);

    var range_key = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 1, 2, 2, 3, 3 }, .cpu);
    defer range_key.deinit();
    var range_delta = try DeviceColumn.fromSlice(f64, gpa, &.{ -3.0, 4.0, -5.0, 12.0, -2.0, 2.0 }, .cpu);
    defer range_delta.deinit();
    var range_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "bucket", .data = range_key },
        .{ .name = "delta", .data = range_delta },
    });
    defer range_table.deinit();

    var ptp_delta = try range_table.groupByPeakToPeak("bucket", "delta", "delta_ptp");
    defer ptp_delta.deinit();
    const ptp_delta_values = try (try ptp_delta.column("delta_ptp")).f64.toOwnedSlice(gpa);
    defer gpa.free(ptp_delta_values);
    try std.testing.expectApproxEqAbs(@as(f64, 7.0), ptp_delta_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 17.0), ptp_delta_values[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), ptp_delta_values[2], 1e-12);

    var midrange_delta = try range_table.groupByMidrange("bucket", "delta", "delta_midrange");
    defer midrange_delta.deinit();
    const midrange_delta_values = try (try midrange_delta.column("delta_midrange")).f64.toOwnedSlice(gpa);
    defer gpa.free(midrange_delta_values);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), midrange_delta_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.5), midrange_delta_values[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), midrange_delta_values[2], 1e-12);

    var range_coeff_delta = try range_table.groupByRangeCoefficient("bucket", "delta", "delta_range_coeff");
    defer range_coeff_delta.deinit();
    const range_coeff_delta_values = try (try range_coeff_delta.column("delta_range_coeff")).f64.toOwnedSlice(gpa);
    defer gpa.free(range_coeff_delta_values);
    try std.testing.expectApproxEqAbs(@as(f64, 7.0), range_coeff_delta_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 17.0 / 7.0), range_coeff_delta_values[1], 1e-12);
    try std.testing.expect(std.math.isNan(range_coeff_delta_values[2]));

    var mean_key = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 1, 2, 2, 3, 3, 4, 4 }, .cpu);
    defer mean_key.deinit();
    var ratio = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 8.0, 1.0, 4.0, 0.0, 5.0, -1.0, 4.0 }, .cpu);
    defer ratio.deinit();
    var mean_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "bucket", .data = mean_key },
        .{ .name = "ratio", .data = ratio },
    });
    defer mean_table.deinit();

    var geometric_ratio = try mean_table.groupByGeometricMean("bucket", "ratio", "ratio_geometric");
    defer geometric_ratio.deinit();
    const geometric_ratio_values = try (try geometric_ratio.column("ratio_geometric")).f64.toOwnedSlice(gpa);
    defer gpa.free(geometric_ratio_values);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), geometric_ratio_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), geometric_ratio_values[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), geometric_ratio_values[2], 1e-12);
    try std.testing.expect(std.math.isNan(geometric_ratio_values[3]));

    var harmonic_ratio = try mean_table.groupByHarmonicMean("bucket", "ratio", "ratio_harmonic");
    defer harmonic_ratio.deinit();
    const harmonic_ratio_values = try (try harmonic_ratio.column("ratio_harmonic")).f64.toOwnedSlice(gpa);
    defer gpa.free(harmonic_ratio_values);
    try std.testing.expectApproxEqAbs(@as(f64, 16.0 / 5.0), harmonic_ratio_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 8.0 / 5.0), harmonic_ratio_values[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), harmonic_ratio_values[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -8.0 / 3.0), harmonic_ratio_values[3], 1e-12);

    var log_key = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 1, 2, 2, 3, 3 }, .cpu);
    defer log_key.deinit();
    var logit = try DeviceColumn.fromSlice(f64, gpa, &.{ 1000.0, 1001.0, -std.math.inf(f64), -std.math.inf(f64), std.math.nan(f64), 1.0 }, .cpu);
    defer logit.deinit();
    var log_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "bucket", .data = log_key },
        .{ .name = "logit", .data = logit },
    });
    defer log_table.deinit();

    var logsumexp_logits = try log_table.groupByLogSumExp("bucket", "logit", "logit_logsumexp");
    defer logsumexp_logits.deinit();
    const logsumexp_logit_values = try (try logsumexp_logits.column("logit_logsumexp")).f64.toOwnedSlice(gpa);
    defer gpa.free(logsumexp_logit_values);
    try std.testing.expectApproxEqAbs(@as(f64, 1001.0) + std.math.log1p(std.math.exp(@as(f64, -1.0))), logsumexp_logit_values[0], 1e-12);
    try std.testing.expect(std.math.isNegativeInf(logsumexp_logit_values[1]));
    try std.testing.expect(std.math.isNan(logsumexp_logit_values[2]));

    var logmeanexp_logits = try log_table.groupByLogMeanExp("bucket", "logit", "logit_logmeanexp");
    defer logmeanexp_logits.deinit();
    const logmeanexp_logit_values = try (try logmeanexp_logits.column("logit_logmeanexp")).f64.toOwnedSlice(gpa);
    defer gpa.free(logmeanexp_logit_values);
    try std.testing.expectApproxEqAbs(@as(f64, 1001.0) + std.math.log1p(std.math.exp(@as(f64, -1.0))) - std.math.ln2, logmeanexp_logit_values[0], 1e-12);
    try std.testing.expect(std.math.isNegativeInf(logmeanexp_logit_values[1]));
    try std.testing.expect(std.math.isNan(logmeanexp_logit_values[2]));

    var skew_sales = try table.groupBySkewness("store", "sales", "sales_skewness_simple");
    defer skew_sales.deinit();
    const skew_sales_values = try (try skew_sales.column("sales_skewness_simple")).f64.toOwnedSlice(gpa);
    defer gpa.free(skew_sales_values);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), skew_sales_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), skew_sales_values[1], 1e-12);

    var kurt_sales = try table.groupByKurtosis("store", "sales", "sales_kurtosis_simple");
    defer kurt_sales.deinit();
    const kurt_sales_values = try (try kurt_sales.column("sales_kurtosis_simple")).f64.toOwnedSlice(gpa);
    defer gpa.free(kurt_sales_values);
    try std.testing.expectApproxEqAbs(@as(f64, -2.0), kurt_sales_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -2.0), kurt_sales_values[1], 1e-12);

    var stats = try table.groupByStats("store", "sales", "sales");
    defer stats.deinit();
    try std.testing.expectEqual(@as(usize, 6), stats.width());
    const stats_keys = try (try stats.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(stats_keys);
    const stats_counts = try (try stats.column("sales_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(stats_counts);
    const stats_sums = try (try stats.column("sales_sum")).f64.toOwnedSlice(gpa);
    defer gpa.free(stats_sums);
    const stats_mins = try (try stats.column("sales_min")).f64.toOwnedSlice(gpa);
    defer gpa.free(stats_mins);
    const stats_maxes = try (try stats.column("sales_max")).f64.toOwnedSlice(gpa);
    defer gpa.free(stats_maxes);
    const stats_means = try (try stats.column("sales_mean")).f64.toOwnedSlice(gpa);
    defer gpa.free(stats_means);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2 }, stats_keys);
    try std.testing.expectEqualSlices(i64, &.{ 2, 2 }, stats_counts);
    try std.testing.expectEqualSlices(f64, &.{ 15.0, 14.0 }, stats_sums);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 3.0 }, stats_mins);
    try std.testing.expectEqualSlices(f64, &.{ 13.0, 11.0 }, stats_maxes);
    try std.testing.expectEqualSlices(f64, &.{ 7.5, 7.0 }, stats_means);

    var profile = try table.groupByProfile("store", "sales", "sales");
    defer profile.deinit();
    try std.testing.expectEqual(@as(usize, 8), profile.width());
    const profile_keys = try (try profile.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(profile_keys);
    const profile_counts = try (try profile.column("sales_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(profile_counts);
    const profile_sums = try (try profile.column("sales_sum")).f64.toOwnedSlice(gpa);
    defer gpa.free(profile_sums);
    const profile_variances = try (try profile.column("sales_variance")).f64.toOwnedSlice(gpa);
    defer gpa.free(profile_variances);
    const profile_stddevs = try (try profile.column("sales_stddev")).f64.toOwnedSlice(gpa);
    defer gpa.free(profile_stddevs);
    const profile_skewnesses = try (try profile.column("sales_skewness")).f64.toOwnedSlice(gpa);
    defer gpa.free(profile_skewnesses);
    const profile_kurtoses = try (try profile.column("sales_kurtosis")).f64.toOwnedSlice(gpa);
    defer gpa.free(profile_kurtoses);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2 }, profile_keys);
    try std.testing.expectEqualSlices(i64, &.{ 2, 2 }, profile_counts);
    try std.testing.expectEqualSlices(f64, &.{ 15.0, 14.0 }, profile_sums);
    try std.testing.expectApproxEqAbs(@as(f64, 30.25), profile_variances[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 16.0), profile_variances[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.5), profile_stddevs[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), profile_stddevs[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), profile_skewnesses[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), profile_skewnesses[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -2.0), profile_kurtoses[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -2.0), profile_kurtoses[1], 1e-12);

    var keyed = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 1, 1, 2, 2, 2 }, .cpu);
    defer keyed.deinit();
    var day = try DeviceColumn.fromSlice(i32, gpa, &.{ 10, 10, 11, 10, 10, 11 }, .cpu);
    defer day.deinit();
    var amount = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, 2.0, 9.0, 4.0, 6.0, 12.0 }, &.{ true, true, true, true, false, true }, .cpu);
    defer amount.deinit();
    var multi = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "store", .data = keyed },
        .{ .name = "day", .data = day },
        .{ .name = "amount", .data = amount },
    });
    defer multi.deinit();

    var multi_counts = try multi.groupByCountOn(&.{ "store", "day" }, "rows");
    defer multi_counts.deinit();
    try std.testing.expectEqual(@as(usize, 3), multi_counts.width());
    try std.testing.expectEqual(@as(usize, 4), multi_counts.height());
    const mc_store = try (try multi_counts.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(mc_store);
    const mc_day = try (try multi_counts.column("day")).i32.toOwnedSlice(gpa);
    defer gpa.free(mc_day);
    const mc_rows = try (try multi_counts.column("rows")).i64.toOwnedSlice(gpa);
    defer gpa.free(mc_rows);
    try std.testing.expectEqualSlices(i32, &.{ 1, 1, 2, 2 }, mc_store);
    try std.testing.expectEqualSlices(i32, &.{ 10, 11, 10, 11 }, mc_day);
    try std.testing.expectEqualSlices(i64, &.{ 2, 1, 2, 1 }, mc_rows);

    var multi_value_counts = try multi.valueCountsOnAs(&.{ "store", "day" }, "freq");
    defer multi_value_counts.deinit();
    const mvc_rows = try (try multi_value_counts.column("freq")).i64.toOwnedSlice(gpa);
    defer gpa.free(mvc_rows);
    try std.testing.expectEqualSlices(i64, &.{ 2, 1, 2, 1 }, mvc_rows);

    var multi_sorted_counts = try multi.valueCountsOnSortedAs(&.{ "store", "day" }, "freq");
    defer multi_sorted_counts.deinit();
    const msc_rows = try (try multi_sorted_counts.column("freq")).i64.toOwnedSlice(gpa);
    defer gpa.free(msc_rows);
    try std.testing.expectEqualSlices(i64, &.{ 2, 2, 1, 1 }, msc_rows);

    var multi_sum = try multi.groupBySumOn(&.{ "store", "day" }, "amount", "amount_sum");
    defer multi_sum.deinit();
    const ms_simple_sum = try (try multi_sum.column("amount_sum")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_sum);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 9.0, 4.0, 12.0 }, ms_simple_sum);

    var multi_prod = try multi.groupByProdOn(&.{ "store", "day" }, "amount", "amount_prod");
    defer multi_prod.deinit();
    const ms_simple_prod = try (try multi_prod.column("amount_prod")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_prod);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 9.0, 4.0, 12.0 }, ms_simple_prod);

    var multi_min = try multi.groupByMinOn(&.{ "store", "day" }, "amount", "amount_min");
    defer multi_min.deinit();
    const ms_simple_min = try (try multi_min.column("amount_min")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_min);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 9.0, 4.0, 12.0 }, ms_simple_min);

    var multi_max = try multi.groupByMaxOn(&.{ "store", "day" }, "amount", "amount_max");
    defer multi_max.deinit();
    const ms_simple_max = try (try multi_max.column("amount_max")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_max);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 9.0, 4.0, 12.0 }, ms_simple_max);

    var multi_mean = try multi.groupByMeanOn(&.{ "store", "day" }, "amount", "amount_mean");
    defer multi_mean.deinit();
    const ms_simple_mean = try (try multi_mean.column("amount_mean")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_mean);
    try std.testing.expectEqualSlices(f64, &.{ 1.5, 9.0, 4.0, 12.0 }, ms_simple_mean);

    var multi_first = try multi.groupByFirstOn(&.{ "store", "day" }, "amount", "amount_first");
    defer multi_first.deinit();
    const ms_simple_first = try (try multi_first.column("amount_first")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_first);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 9.0, 4.0, 12.0 }, ms_simple_first);

    var multi_last = try multi.groupByLastOn(&.{ "store", "day" }, "amount", "amount_last");
    defer multi_last.deinit();
    const ms_simple_last = try (try multi_last.column("amount_last")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_last);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 9.0, 4.0, 12.0 }, ms_simple_last);

    var multi_argmin = try multi.groupByArgMinOn(&.{ "store", "day" }, "amount", "amount_argmin");
    defer multi_argmin.deinit();
    const ms_simple_argmin = try (try multi_argmin.column("amount_argmin")).i64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_argmin);
    try std.testing.expectEqualSlices(i64, &.{ 0, 2, 3, 5 }, ms_simple_argmin);

    var multi_argmax = try multi.groupByArgMaxOn(&.{ "store", "day" }, "amount", "amount_argmax");
    defer multi_argmax.deinit();
    const ms_simple_argmax = try (try multi_argmax.column("amount_argmax")).i64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_argmax);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 5 }, ms_simple_argmax);

    var multi_unique = try multi.groupByNUniqueOn(&.{ "store", "day" }, "amount", "amount_n_unique");
    defer multi_unique.deinit();
    const ms_simple_unique = try (try multi_unique.column("amount_n_unique")).i64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_unique);
    try std.testing.expectEqualSlices(i64, &.{ 2, 1, 1, 1 }, ms_simple_unique);

    var multi_mode = try multi.groupByModeOn(&.{ "store", "day" }, "amount", "amount_mode");
    defer multi_mode.deinit();
    const ms_simple_mode = try (try multi_mode.column("amount_mode")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_mode);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 9.0, 4.0, 12.0 }, ms_simple_mode);

    var multi_median = try multi.groupByMedianOn(&.{ "store", "day" }, "amount", "amount_median");
    defer multi_median.deinit();
    const ms_simple_median = try (try multi_median.column("amount_median")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_median);
    try std.testing.expectEqualSlices(f64, &.{ 1.5, 9.0, 4.0, 12.0 }, ms_simple_median);

    var multi_q1 = try multi.groupByQuantileOn(&.{ "store", "day" }, "amount", "amount_q1", 0.25);
    defer multi_q1.deinit();
    const ms_simple_q1 = try (try multi_q1.column("amount_q1")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_q1);
    try std.testing.expectEqualSlices(f64, &.{ 1.25, 9.0, 4.0, 12.0 }, ms_simple_q1);

    var multi_iqr = try multi.groupByIQROn(&.{ "store", "day" }, "amount", "amount_iqr_simple");
    defer multi_iqr.deinit();
    const ms_simple_iqr = try (try multi_iqr.column("amount_iqr_simple")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_iqr);
    try std.testing.expectEqualSlices(f64, &.{ 0.5, 0.0, 0.0, 0.0 }, ms_simple_iqr);

    var multi_mad = try multi.groupByMedianAbsDevOn(&.{ "store", "day" }, "amount", "amount_mad_simple");
    defer multi_mad.deinit();
    const ms_simple_mad = try (try multi_mad.column("amount_mad_simple")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_mad);
    try std.testing.expectEqualSlices(f64, &.{ 0.5, 0.0, 0.0, 0.0 }, ms_simple_mad);

    var multi_idr = try multi.groupByIDROn(&.{ "store", "day" }, "amount", "amount_idr_simple");
    defer multi_idr.deinit();
    const ms_simple_idr = try (try multi_idr.column("amount_idr_simple")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_idr);
    try std.testing.expectApproxEqAbs(@as(f64, 0.8), ms_simple_idr[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ms_simple_idr[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ms_simple_idr[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ms_simple_idr[3], 1e-12);

    var multi_midhinge = try multi.groupByMidhingeOn(&.{ "store", "day" }, "amount", "amount_midhinge_simple");
    defer multi_midhinge.deinit();
    const ms_simple_midhinge = try (try multi_midhinge.column("amount_midhinge_simple")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_midhinge);
    try std.testing.expectEqualSlices(f64, &.{ 1.5, 9.0, 4.0, 12.0 }, ms_simple_midhinge);

    var multi_trimean = try multi.groupByTrimeanOn(&.{ "store", "day" }, "amount", "amount_trimean_simple");
    defer multi_trimean.deinit();
    const ms_simple_trimean = try (try multi_trimean.column("amount_trimean_simple")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_trimean);
    try std.testing.expectEqualSlices(f64, &.{ 1.5, 9.0, 4.0, 12.0 }, ms_simple_trimean);

    var multi_bowley = try multi.groupByBowleySkewOn(&.{ "store", "day" }, "amount", "amount_bowley_simple");
    defer multi_bowley.deinit();
    const ms_simple_bowley = try (try multi_bowley.column("amount_bowley_simple")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_bowley);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ms_simple_bowley[0], 1e-12);
    try std.testing.expect(std.math.isNan(ms_simple_bowley[1]));
    try std.testing.expect(std.math.isNan(ms_simple_bowley[2]));
    try std.testing.expect(std.math.isNan(ms_simple_bowley[3]));

    var multi_qcd = try multi.groupByQcdOn(&.{ "store", "day" }, "amount", "amount_qcd_simple");
    defer multi_qcd.deinit();
    const ms_simple_qcd = try (try multi_qcd.column("amount_qcd_simple")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_qcd);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 6.0), ms_simple_qcd[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ms_simple_qcd[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ms_simple_qcd[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ms_simple_qcd[3], 1e-12);

    var multi_kelley = try multi.groupByKelleySkewOn(&.{ "store", "day" }, "amount", "amount_kelley_simple");
    defer multi_kelley.deinit();
    const ms_simple_kelley = try (try multi_kelley.column("amount_kelley_simple")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_kelley);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ms_simple_kelley[0], 1e-12);
    try std.testing.expect(std.math.isNan(ms_simple_kelley[1]));
    try std.testing.expect(std.math.isNan(ms_simple_kelley[2]));
    try std.testing.expect(std.math.isNan(ms_simple_kelley[3]));

    var multi_variance = try multi.groupByVarianceOn(&.{ "store", "day" }, "amount", "amount_variance_simple");
    defer multi_variance.deinit();
    const ms_simple_variance = try (try multi_variance.column("amount_variance_simple")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_variance);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), ms_simple_variance[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ms_simple_variance[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ms_simple_variance[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ms_simple_variance[3], 1e-12);

    var multi_stddev = try multi.groupByStddevOn(&.{ "store", "day" }, "amount", "amount_stddev_simple");
    defer multi_stddev.deinit();
    const ms_simple_stddev = try (try multi_stddev.column("amount_stddev_simple")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_stddev);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), ms_simple_stddev[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ms_simple_stddev[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ms_simple_stddev[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ms_simple_stddev[3], 1e-12);

    var multi_sem = try multi.groupBySemOn(&.{ "store", "day" }, "amount", "amount_sem_simple");
    defer multi_sem.deinit();
    const ms_simple_sem = try (try multi_sem.column("amount_sem_simple")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_sem);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5 / std.math.sqrt(2.0)), ms_simple_sem[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ms_simple_sem[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ms_simple_sem[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ms_simple_sem[3], 1e-12);

    var multi_cv = try multi.groupByCvOn(&.{ "store", "day" }, "amount", "amount_cv_simple");
    defer multi_cv.deinit();
    const ms_simple_cv = try (try multi_cv.column("amount_cv_simple")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_cv);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5 / 1.5), ms_simple_cv[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ms_simple_cv[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ms_simple_cv[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ms_simple_cv[3], 1e-12);

    var multi_fano = try multi.groupByFanoOn(&.{ "store", "day" }, "amount", "amount_fano_simple");
    defer multi_fano.deinit();
    const ms_simple_fano = try (try multi_fano.column("amount_fano_simple")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_fano);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25 / 1.5), ms_simple_fano[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ms_simple_fano[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ms_simple_fano[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ms_simple_fano[3], 1e-12);

    var multi_mean_abs = try multi.groupByMeanAbsOn(&.{ "store", "day" }, "amount", "amount_mean_abs_simple");
    defer multi_mean_abs.deinit();
    const ms_simple_mean_abs = try (try multi_mean_abs.column("amount_mean_abs_simple")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_mean_abs);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), ms_simple_mean_abs[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0), ms_simple_mean_abs[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), ms_simple_mean_abs[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 12.0), ms_simple_mean_abs[3], 1e-12);

    var multi_mean_abs_dev = try multi.groupByMeanAbsDevOn(&.{ "store", "day" }, "amount", "amount_mean_abs_dev_simple");
    defer multi_mean_abs_dev.deinit();
    const ms_simple_mean_abs_dev = try (try multi_mean_abs_dev.column("amount_mean_abs_dev_simple")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_mean_abs_dev);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), ms_simple_mean_abs_dev[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ms_simple_mean_abs_dev[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ms_simple_mean_abs_dev[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ms_simple_mean_abs_dev[3], 1e-12);

    var multi_mean_abs_dev_ratio = try multi.groupByMeanAbsDevRatioOn(&.{ "store", "day" }, "amount", "amount_mean_abs_dev_ratio_simple");
    defer multi_mean_abs_dev_ratio.deinit();
    const ms_simple_mean_abs_dev_ratio = try (try multi_mean_abs_dev_ratio.column("amount_mean_abs_dev_ratio_simple")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_mean_abs_dev_ratio);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), ms_simple_mean_abs_dev_ratio[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ms_simple_mean_abs_dev_ratio[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ms_simple_mean_abs_dev_ratio[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ms_simple_mean_abs_dev_ratio[3], 1e-12);

    var multi_mean_square = try multi.groupByMeanSquareOn(&.{ "store", "day" }, "amount", "amount_mean_square_simple");
    defer multi_mean_square.deinit();
    const ms_simple_mean_square = try (try multi_mean_square.column("amount_mean_square_simple")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_mean_square);
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), ms_simple_mean_square[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 81.0), ms_simple_mean_square[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 16.0), ms_simple_mean_square[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 144.0), ms_simple_mean_square[3], 1e-12);

    var multi_rms = try multi.groupByRmsOn(&.{ "store", "day" }, "amount", "amount_rms_simple");
    defer multi_rms.deinit();
    const ms_simple_rms = try (try multi_rms.column("amount_rms_simple")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_rms);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 2.5)), ms_simple_rms[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0), ms_simple_rms[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), ms_simple_rms[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 12.0), ms_simple_rms[3], 1e-12);

    var multi_l1 = try multi.groupByL1NormOn(&.{ "store", "day" }, "amount", "amount_l1_simple");
    defer multi_l1.deinit();
    const ms_simple_l1 = try (try multi_l1.column("amount_l1_simple")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_l1);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), ms_simple_l1[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0), ms_simple_l1[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), ms_simple_l1[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 12.0), ms_simple_l1[3], 1e-12);

    var multi_l2 = try multi.groupByL2NormOn(&.{ "store", "day" }, "amount", "amount_l2_simple");
    defer multi_l2.deinit();
    const ms_simple_l2 = try (try multi_l2.column("amount_l2_simple")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_l2);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 5.0)), ms_simple_l2[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0), ms_simple_l2[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), ms_simple_l2[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 12.0), ms_simple_l2[3], 1e-12);

    var multi_max_abs = try multi.groupByMaxAbsOn(&.{ "store", "day" }, "amount", "amount_max_abs_simple");
    defer multi_max_abs.deinit();
    const ms_simple_max_abs = try (try multi_max_abs.column("amount_max_abs_simple")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_max_abs);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), ms_simple_max_abs[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0), ms_simple_max_abs[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), ms_simple_max_abs[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 12.0), ms_simple_max_abs[3], 1e-12);

    var multi_min_abs = try multi.groupByMinAbsOn(&.{ "store", "day" }, "amount", "amount_min_abs_simple");
    defer multi_min_abs.deinit();
    const ms_simple_min_abs = try (try multi_min_abs.column("amount_min_abs_simple")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_min_abs);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), ms_simple_min_abs[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0), ms_simple_min_abs[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), ms_simple_min_abs[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 12.0), ms_simple_min_abs[3], 1e-12);

    var multi_geometric = try multi.groupByGeometricMeanOn(&.{ "store", "day" }, "amount", "amount_geometric_simple");
    defer multi_geometric.deinit();
    const ms_simple_geometric = try (try multi_geometric.column("amount_geometric_simple")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_geometric);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 2.0)), ms_simple_geometric[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0), ms_simple_geometric[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), ms_simple_geometric[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 12.0), ms_simple_geometric[3], 1e-12);

    var multi_harmonic = try multi.groupByHarmonicMeanOn(&.{ "store", "day" }, "amount", "amount_harmonic_simple");
    defer multi_harmonic.deinit();
    const ms_simple_harmonic = try (try multi_harmonic.column("amount_harmonic_simple")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_harmonic);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0 / 3.0), ms_simple_harmonic[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0), ms_simple_harmonic[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), ms_simple_harmonic[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 12.0), ms_simple_harmonic[3], 1e-12);

    var multi_logsumexp = try multi.groupByLogSumExpOn(&.{ "store", "day" }, "amount", "amount_logsumexp_simple");
    defer multi_logsumexp.deinit();
    const ms_simple_logsumexp = try (try multi_logsumexp.column("amount_logsumexp_simple")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_logsumexp);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0) + std.math.log1p(std.math.exp(@as(f64, -1.0))), ms_simple_logsumexp[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0), ms_simple_logsumexp[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), ms_simple_logsumexp[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 12.0), ms_simple_logsumexp[3], 1e-12);

    var multi_logmeanexp = try multi.groupByLogMeanExpOn(&.{ "store", "day" }, "amount", "amount_logmeanexp_simple");
    defer multi_logmeanexp.deinit();
    const ms_simple_logmeanexp = try (try multi_logmeanexp.column("amount_logmeanexp_simple")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_logmeanexp);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0) + std.math.log1p(std.math.exp(@as(f64, -1.0))) - std.math.ln2, ms_simple_logmeanexp[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0), ms_simple_logmeanexp[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), ms_simple_logmeanexp[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 12.0), ms_simple_logmeanexp[3], 1e-12);

    var multi_ptp = try multi.groupByPTPOn(&.{ "store", "day" }, "amount", "amount_ptp_simple");
    defer multi_ptp.deinit();
    const ms_simple_ptp = try (try multi_ptp.column("amount_ptp_simple")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_ptp);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), ms_simple_ptp[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ms_simple_ptp[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ms_simple_ptp[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ms_simple_ptp[3], 1e-12);

    var multi_midrange = try multi.groupByMidrangeOn(&.{ "store", "day" }, "amount", "amount_midrange_simple");
    defer multi_midrange.deinit();
    const ms_simple_midrange = try (try multi_midrange.column("amount_midrange_simple")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_midrange);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), ms_simple_midrange[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0), ms_simple_midrange[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), ms_simple_midrange[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 12.0), ms_simple_midrange[3], 1e-12);

    var multi_range_coeff = try multi.groupByRangeCoeffOn(&.{ "store", "day" }, "amount", "amount_range_coeff_simple");
    defer multi_range_coeff.deinit();
    const ms_simple_range_coeff = try (try multi_range_coeff.column("amount_range_coeff_simple")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_range_coeff);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), ms_simple_range_coeff[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ms_simple_range_coeff[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ms_simple_range_coeff[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ms_simple_range_coeff[3], 1e-12);

    var multi_skew = try multi.groupBySkewnessOn(&.{ "store", "day" }, "amount", "amount_skewness_simple");
    defer multi_skew.deinit();
    const ms_simple_skew = try (try multi_skew.column("amount_skewness_simple")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_skew);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ms_simple_skew[0], 1e-12);
    try std.testing.expect(std.math.isNan(ms_simple_skew[1]));
    try std.testing.expect(std.math.isNan(ms_simple_skew[2]));
    try std.testing.expect(std.math.isNan(ms_simple_skew[3]));

    var multi_kurt = try multi.groupByKurtosisOn(&.{ "store", "day" }, "amount", "amount_kurtosis_simple");
    defer multi_kurt.deinit();
    const ms_simple_kurt = try (try multi_kurt.column("amount_kurtosis_simple")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_simple_kurt);
    try std.testing.expectApproxEqAbs(@as(f64, -2.0), ms_simple_kurt[0], 1e-12);
    try std.testing.expect(std.math.isNan(ms_simple_kurt[1]));
    try std.testing.expect(std.math.isNan(ms_simple_kurt[2]));
    try std.testing.expect(std.math.isNan(ms_simple_kurt[3]));

    try std.testing.expectError(error.InvalidShape, multi.groupByQuantileOn(&.{ "store", "day" }, "amount", "bad_q", 1.5));

    var multi_counts_plan = try DeviceLazyFrame.init(gpa, multi);
    defer multi_counts_plan.deinit();
    try multi_counts_plan.valueCountsOnSortedAs(&.{ "store", "day" }, "freq_lazy");
    const multi_counts_explained = try multi_counts_plan.explain(gpa);
    defer gpa.free(multi_counts_explained);
    try std.testing.expect(std.mem.indexOf(u8, multi_counts_explained, "group_by_count_on([store,day] -> freq_lazy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, multi_counts_explained, "sort_by(freq_lazy, desc=true)") != null);
    var lazy_multi_counts = try multi_counts_plan.collect();
    defer lazy_multi_counts.deinit();
    const lazy_mc_rows = try (try lazy_multi_counts.column("freq_lazy")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_mc_rows);
    try std.testing.expectEqualSlices(i64, &.{ 2, 2, 1, 1 }, lazy_mc_rows);

    var multi_mean_plan = try DeviceLazyFrame.init(gpa, multi);
    defer multi_mean_plan.deinit();
    try multi_mean_plan.groupByMeanOn(&.{ "store", "day" }, "amount", "amount_mean_lazy");
    const multi_mean_explained = try multi_mean_plan.explain(gpa);
    defer gpa.free(multi_mean_explained);
    try std.testing.expect(std.mem.indexOf(u8, multi_mean_explained, "group_by_mean_on([store,day], value=amount -> amount_mean_lazy)") != null);
    var lazy_multi_mean = try multi_mean_plan.collect();
    defer lazy_multi_mean.deinit();
    const lazy_ms_mean = try (try lazy_multi_mean.column("amount_mean_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ms_mean);
    try std.testing.expectEqualSlices(f64, &.{ 1.5, 9.0, 4.0, 12.0 }, lazy_ms_mean);

    var multi_prod_plan = try DeviceLazyFrame.init(gpa, multi);
    defer multi_prod_plan.deinit();
    try multi_prod_plan.groupByProdOn(&.{ "store", "day" }, "amount", "amount_prod_lazy");
    const multi_prod_explained = try multi_prod_plan.explain(gpa);
    defer gpa.free(multi_prod_explained);
    try std.testing.expect(std.mem.indexOf(u8, multi_prod_explained, "group_by_prod_on([store,day], value=amount -> amount_prod_lazy)") != null);
    var lazy_multi_prod = try multi_prod_plan.collect();
    defer lazy_multi_prod.deinit();
    const lazy_ms_prod = try (try lazy_multi_prod.column("amount_prod_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ms_prod);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 9.0, 4.0, 12.0 }, lazy_ms_prod);

    var multi_last_plan = try DeviceLazyFrame.init(gpa, multi);
    defer multi_last_plan.deinit();
    try multi_last_plan.groupByLastOn(&.{ "store", "day" }, "amount", "amount_last_lazy");
    const multi_last_explained = try multi_last_plan.explain(gpa);
    defer gpa.free(multi_last_explained);
    try std.testing.expect(std.mem.indexOf(u8, multi_last_explained, "group_by_last_on([store,day], value=amount -> amount_last_lazy)") != null);
    var lazy_multi_last = try multi_last_plan.collect();
    defer lazy_multi_last.deinit();
    const lazy_ms_last = try (try lazy_multi_last.column("amount_last_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ms_last);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 9.0, 4.0, 12.0 }, lazy_ms_last);

    var multi_argmax_plan = try DeviceLazyFrame.init(gpa, multi);
    defer multi_argmax_plan.deinit();
    try multi_argmax_plan.groupByArgMaxOn(&.{ "store", "day" }, "amount", "amount_argmax_lazy");
    const multi_argmax_explained = try multi_argmax_plan.explain(gpa);
    defer gpa.free(multi_argmax_explained);
    try std.testing.expect(std.mem.indexOf(u8, multi_argmax_explained, "group_by_argmax_on([store,day], value=amount -> amount_argmax_lazy)") != null);
    var lazy_multi_argmax = try multi_argmax_plan.collect();
    defer lazy_multi_argmax.deinit();
    const lazy_ms_argmax = try (try lazy_multi_argmax.column("amount_argmax_lazy")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ms_argmax);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 5 }, lazy_ms_argmax);

    var multi_unique_plan = try DeviceLazyFrame.init(gpa, multi);
    defer multi_unique_plan.deinit();
    try multi_unique_plan.groupByNUniqueOn(&.{ "store", "day" }, "amount", "amount_n_unique_lazy");
    const multi_unique_explained = try multi_unique_plan.explain(gpa);
    defer gpa.free(multi_unique_explained);
    try std.testing.expect(std.mem.indexOf(u8, multi_unique_explained, "group_by_n_unique_on([store,day], value=amount -> amount_n_unique_lazy)") != null);
    var lazy_multi_unique = try multi_unique_plan.collect();
    defer lazy_multi_unique.deinit();
    const lazy_ms_unique = try (try lazy_multi_unique.column("amount_n_unique_lazy")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ms_unique);
    try std.testing.expectEqualSlices(i64, &.{ 2, 1, 1, 1 }, lazy_ms_unique);

    var multi_mode_plan = try DeviceLazyFrame.init(gpa, multi);
    defer multi_mode_plan.deinit();
    try multi_mode_plan.groupByModeOn(&.{ "store", "day" }, "amount", "amount_mode_lazy");
    const multi_mode_explained = try multi_mode_plan.explain(gpa);
    defer gpa.free(multi_mode_explained);
    try std.testing.expect(std.mem.indexOf(u8, multi_mode_explained, "group_by_mode_on([store,day], value=amount -> amount_mode_lazy)") != null);
    var lazy_multi_mode = try multi_mode_plan.collect();
    defer lazy_multi_mode.deinit();
    const lazy_ms_mode = try (try lazy_multi_mode.column("amount_mode_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ms_mode);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 9.0, 4.0, 12.0 }, lazy_ms_mode);

    var multi_median_plan = try DeviceLazyFrame.init(gpa, multi);
    defer multi_median_plan.deinit();
    try multi_median_plan.groupByMedianOn(&.{ "store", "day" }, "amount", "amount_median_lazy");
    const multi_median_explained = try multi_median_plan.explain(gpa);
    defer gpa.free(multi_median_explained);
    try std.testing.expect(std.mem.indexOf(u8, multi_median_explained, "group_by_median_on([store,day], value=amount -> amount_median_lazy)") != null);
    var lazy_multi_median = try multi_median_plan.collect();
    defer lazy_multi_median.deinit();
    const lazy_ms_median = try (try lazy_multi_median.column("amount_median_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ms_median);
    try std.testing.expectEqualSlices(f64, &.{ 1.5, 9.0, 4.0, 12.0 }, lazy_ms_median);

    var multi_q1_plan = try DeviceLazyFrame.init(gpa, multi);
    defer multi_q1_plan.deinit();
    try multi_q1_plan.groupByQuantileOn(&.{ "store", "day" }, "amount", "amount_q1_lazy", 0.25);
    const multi_q1_explained = try multi_q1_plan.explain(gpa);
    defer gpa.free(multi_q1_explained);
    try std.testing.expect(std.mem.indexOf(u8, multi_q1_explained, "group_by_quantile_on([store,day], value=amount, q=0.25 -> amount_q1_lazy)") != null);
    var lazy_multi_q1 = try multi_q1_plan.collect();
    defer lazy_multi_q1.deinit();
    const lazy_ms_q1 = try (try lazy_multi_q1.column("amount_q1_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ms_q1);
    try std.testing.expectEqualSlices(f64, &.{ 1.25, 9.0, 4.0, 12.0 }, lazy_ms_q1);

    var multi_iqr_plan = try DeviceLazyFrame.init(gpa, multi);
    defer multi_iqr_plan.deinit();
    try multi_iqr_plan.groupByIqrOn(&.{ "store", "day" }, "amount", "amount_iqr_lazy");
    const multi_iqr_explained = try multi_iqr_plan.explain(gpa);
    defer gpa.free(multi_iqr_explained);
    try std.testing.expect(std.mem.indexOf(u8, multi_iqr_explained, "group_by_iqr_on([store,day], value=amount -> amount_iqr_lazy)") != null);
    var lazy_multi_iqr = try multi_iqr_plan.collect();
    defer lazy_multi_iqr.deinit();
    const lazy_ms_iqr = try (try lazy_multi_iqr.column("amount_iqr_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ms_iqr);
    try std.testing.expectEqualSlices(f64, &.{ 0.5, 0.0, 0.0, 0.0 }, lazy_ms_iqr);

    var multi_mad_plan = try DeviceLazyFrame.init(gpa, multi);
    defer multi_mad_plan.deinit();
    try multi_mad_plan.groupByMADOn(&.{ "store", "day" }, "amount", "amount_mad_lazy");
    const multi_mad_explained = try multi_mad_plan.explain(gpa);
    defer gpa.free(multi_mad_explained);
    try std.testing.expect(std.mem.indexOf(u8, multi_mad_explained, "group_by_mad_on([store,day], value=amount -> amount_mad_lazy)") != null);
    var lazy_multi_mad = try multi_mad_plan.collect();
    defer lazy_multi_mad.deinit();
    const lazy_ms_mad = try (try lazy_multi_mad.column("amount_mad_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ms_mad);
    try std.testing.expectEqualSlices(f64, &.{ 0.5, 0.0, 0.0, 0.0 }, lazy_ms_mad);

    var multi_idr_plan = try DeviceLazyFrame.init(gpa, multi);
    defer multi_idr_plan.deinit();
    try multi_idr_plan.groupByIdrOn(&.{ "store", "day" }, "amount", "amount_idr_lazy");
    const multi_idr_explained = try multi_idr_plan.explain(gpa);
    defer gpa.free(multi_idr_explained);
    try std.testing.expect(std.mem.indexOf(u8, multi_idr_explained, "group_by_interdecile_range_on([store,day], value=amount -> amount_idr_lazy)") != null);
    var lazy_multi_idr = try multi_idr_plan.collect();
    defer lazy_multi_idr.deinit();
    const lazy_ms_idr = try (try lazy_multi_idr.column("amount_idr_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ms_idr);
    try std.testing.expectApproxEqAbs(@as(f64, 0.8), lazy_ms_idr[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_ms_idr[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_ms_idr[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_ms_idr[3], 1e-12);

    var multi_qcd_plan = try DeviceLazyFrame.init(gpa, multi);
    defer multi_qcd_plan.deinit();
    try multi_qcd_plan.groupByQcdOn(&.{ "store", "day" }, "amount", "amount_qcd_lazy");
    const multi_qcd_explained = try multi_qcd_plan.explain(gpa);
    defer gpa.free(multi_qcd_explained);
    try std.testing.expect(std.mem.indexOf(u8, multi_qcd_explained, "group_by_quartile_coeff_dispersion_on([store,day], value=amount -> amount_qcd_lazy)") != null);
    var lazy_multi_qcd = try multi_qcd_plan.collect();
    defer lazy_multi_qcd.deinit();
    const lazy_ms_qcd = try (try lazy_multi_qcd.column("amount_qcd_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ms_qcd);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 6.0), lazy_ms_qcd[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_ms_qcd[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_ms_qcd[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_ms_qcd[3], 1e-12);

    var multi_kelley_plan = try DeviceLazyFrame.init(gpa, multi);
    defer multi_kelley_plan.deinit();
    try multi_kelley_plan.groupByKelleySkewnessOn(&.{ "store", "day" }, "amount", "amount_kelley_lazy");
    const multi_kelley_explained = try multi_kelley_plan.explain(gpa);
    defer gpa.free(multi_kelley_explained);
    try std.testing.expect(std.mem.indexOf(u8, multi_kelley_explained, "group_by_kelley_skewness_on([store,day], value=amount -> amount_kelley_lazy)") != null);
    var lazy_multi_kelley = try multi_kelley_plan.collect();
    defer lazy_multi_kelley.deinit();
    const lazy_ms_kelley = try (try lazy_multi_kelley.column("amount_kelley_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ms_kelley);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_ms_kelley[0], 1e-12);
    try std.testing.expect(std.math.isNan(lazy_ms_kelley[1]));
    try std.testing.expect(std.math.isNan(lazy_ms_kelley[2]));
    try std.testing.expect(std.math.isNan(lazy_ms_kelley[3]));

    var multi_variance_plan = try DeviceLazyFrame.init(gpa, multi);
    defer multi_variance_plan.deinit();
    try multi_variance_plan.groupByVarianceOn(&.{ "store", "day" }, "amount", "amount_variance_lazy");
    const multi_variance_explained = try multi_variance_plan.explain(gpa);
    defer gpa.free(multi_variance_explained);
    try std.testing.expect(std.mem.indexOf(u8, multi_variance_explained, "group_by_variance_on([store,day], value=amount -> amount_variance_lazy)") != null);
    var lazy_multi_variance = try multi_variance_plan.collect();
    defer lazy_multi_variance.deinit();
    const lazy_ms_variance = try (try lazy_multi_variance.column("amount_variance_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ms_variance);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), lazy_ms_variance[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_ms_variance[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_ms_variance[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_ms_variance[3], 1e-12);

    var multi_sem_plan = try DeviceLazyFrame.init(gpa, multi);
    defer multi_sem_plan.deinit();
    try multi_sem_plan.groupBySemOn(&.{ "store", "day" }, "amount", "amount_sem_lazy");
    const multi_sem_explained = try multi_sem_plan.explain(gpa);
    defer gpa.free(multi_sem_explained);
    try std.testing.expect(std.mem.indexOf(u8, multi_sem_explained, "group_by_sem_on([store,day], value=amount -> amount_sem_lazy)") != null);
    var lazy_multi_sem = try multi_sem_plan.collect();
    defer lazy_multi_sem.deinit();
    const lazy_ms_sem = try (try lazy_multi_sem.column("amount_sem_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ms_sem);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5 / std.math.sqrt(2.0)), lazy_ms_sem[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_ms_sem[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_ms_sem[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_ms_sem[3], 1e-12);

    var multi_cv_plan = try DeviceLazyFrame.init(gpa, multi);
    defer multi_cv_plan.deinit();
    try multi_cv_plan.groupByCvOn(&.{ "store", "day" }, "amount", "amount_cv_lazy");
    const multi_cv_explained = try multi_cv_plan.explain(gpa);
    defer gpa.free(multi_cv_explained);
    try std.testing.expect(std.mem.indexOf(u8, multi_cv_explained, "group_by_cv_on([store,day], value=amount -> amount_cv_lazy)") != null);
    var lazy_multi_cv = try multi_cv_plan.collect();
    defer lazy_multi_cv.deinit();
    const lazy_ms_cv = try (try lazy_multi_cv.column("amount_cv_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ms_cv);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5 / 1.5), lazy_ms_cv[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_ms_cv[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_ms_cv[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_ms_cv[3], 1e-12);

    var multi_fano_plan = try DeviceLazyFrame.init(gpa, multi);
    defer multi_fano_plan.deinit();
    try multi_fano_plan.groupByIndexOfDispersionOn(&.{ "store", "day" }, "amount", "amount_fano_lazy");
    const multi_fano_explained = try multi_fano_plan.explain(gpa);
    defer gpa.free(multi_fano_explained);
    try std.testing.expect(std.mem.indexOf(u8, multi_fano_explained, "group_by_fano_on([store,day], value=amount -> amount_fano_lazy)") != null);
    var lazy_multi_fano = try multi_fano_plan.collect();
    defer lazy_multi_fano.deinit();
    const lazy_ms_fano = try (try lazy_multi_fano.column("amount_fano_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ms_fano);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25 / 1.5), lazy_ms_fano[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_ms_fano[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_ms_fano[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_ms_fano[3], 1e-12);

    var multi_mean_square_plan = try DeviceLazyFrame.init(gpa, multi);
    defer multi_mean_square_plan.deinit();
    try multi_mean_square_plan.groupByMeanSqOn(&.{ "store", "day" }, "amount", "amount_mean_square_lazy");
    const multi_mean_square_explained = try multi_mean_square_plan.explain(gpa);
    defer gpa.free(multi_mean_square_explained);
    try std.testing.expect(std.mem.indexOf(u8, multi_mean_square_explained, "group_by_mean_square_on([store,day], value=amount -> amount_mean_square_lazy)") != null);
    var lazy_multi_mean_square = try multi_mean_square_plan.collect();
    defer lazy_multi_mean_square.deinit();
    const lazy_ms_mean_square = try (try lazy_multi_mean_square.column("amount_mean_square_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ms_mean_square);
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), lazy_ms_mean_square[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 81.0), lazy_ms_mean_square[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 16.0), lazy_ms_mean_square[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 144.0), lazy_ms_mean_square[3], 1e-12);

    var multi_mean_abs_plan = try DeviceLazyFrame.init(gpa, multi);
    defer multi_mean_abs_plan.deinit();
    try multi_mean_abs_plan.groupByMeanAbsOn(&.{ "store", "day" }, "amount", "amount_mean_abs_lazy");
    const multi_mean_abs_explained = try multi_mean_abs_plan.explain(gpa);
    defer gpa.free(multi_mean_abs_explained);
    try std.testing.expect(std.mem.indexOf(u8, multi_mean_abs_explained, "group_by_mean_abs_on([store,day], value=amount -> amount_mean_abs_lazy)") != null);
    var lazy_multi_mean_abs = try multi_mean_abs_plan.collect();
    defer lazy_multi_mean_abs.deinit();
    const lazy_ms_mean_abs = try (try lazy_multi_mean_abs.column("amount_mean_abs_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ms_mean_abs);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), lazy_ms_mean_abs[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0), lazy_ms_mean_abs[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), lazy_ms_mean_abs[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 12.0), lazy_ms_mean_abs[3], 1e-12);

    var multi_mean_abs_dev_plan = try DeviceLazyFrame.init(gpa, multi);
    defer multi_mean_abs_dev_plan.deinit();
    try multi_mean_abs_dev_plan.groupByMeanAbsDevOn(&.{ "store", "day" }, "amount", "amount_mean_abs_dev_lazy");
    const multi_mean_abs_dev_explained = try multi_mean_abs_dev_plan.explain(gpa);
    defer gpa.free(multi_mean_abs_dev_explained);
    try std.testing.expect(std.mem.indexOf(u8, multi_mean_abs_dev_explained, "group_by_mean_abs_dev_on([store,day], value=amount -> amount_mean_abs_dev_lazy)") != null);
    var lazy_multi_mean_abs_dev = try multi_mean_abs_dev_plan.collect();
    defer lazy_multi_mean_abs_dev.deinit();
    const lazy_ms_mean_abs_dev = try (try lazy_multi_mean_abs_dev.column("amount_mean_abs_dev_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ms_mean_abs_dev);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_ms_mean_abs_dev[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_ms_mean_abs_dev[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_ms_mean_abs_dev[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_ms_mean_abs_dev[3], 1e-12);

    var multi_mean_abs_dev_ratio_plan = try DeviceLazyFrame.init(gpa, multi);
    defer multi_mean_abs_dev_ratio_plan.deinit();
    try multi_mean_abs_dev_ratio_plan.groupByMeanAbsDevRatioOn(&.{ "store", "day" }, "amount", "amount_mean_abs_dev_ratio_lazy");
    const multi_mean_abs_dev_ratio_explained = try multi_mean_abs_dev_ratio_plan.explain(gpa);
    defer gpa.free(multi_mean_abs_dev_ratio_explained);
    try std.testing.expect(std.mem.indexOf(u8, multi_mean_abs_dev_ratio_explained, "group_by_mean_abs_dev_ratio_on([store,day], value=amount -> amount_mean_abs_dev_ratio_lazy)") != null);
    var lazy_multi_mean_abs_dev_ratio = try multi_mean_abs_dev_ratio_plan.collect();
    defer lazy_multi_mean_abs_dev_ratio.deinit();
    const lazy_ms_mean_abs_dev_ratio = try (try lazy_multi_mean_abs_dev_ratio.column("amount_mean_abs_dev_ratio_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ms_mean_abs_dev_ratio);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), lazy_ms_mean_abs_dev_ratio[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_ms_mean_abs_dev_ratio[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_ms_mean_abs_dev_ratio[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_ms_mean_abs_dev_ratio[3], 1e-12);

    var multi_max_abs_plan = try DeviceLazyFrame.init(gpa, multi);
    defer multi_max_abs_plan.deinit();
    try multi_max_abs_plan.groupByMaxAbsOn(&.{ "store", "day" }, "amount", "amount_max_abs_lazy");
    const multi_max_abs_explained = try multi_max_abs_plan.explain(gpa);
    defer gpa.free(multi_max_abs_explained);
    try std.testing.expect(std.mem.indexOf(u8, multi_max_abs_explained, "group_by_max_abs_on([store,day], value=amount -> amount_max_abs_lazy)") != null);
    var lazy_multi_max_abs = try multi_max_abs_plan.collect();
    defer lazy_multi_max_abs.deinit();
    const lazy_ms_max_abs = try (try lazy_multi_max_abs.column("amount_max_abs_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ms_max_abs);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_ms_max_abs[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0), lazy_ms_max_abs[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), lazy_ms_max_abs[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 12.0), lazy_ms_max_abs[3], 1e-12);

    var multi_l2_plan = try DeviceLazyFrame.init(gpa, multi);
    defer multi_l2_plan.deinit();
    try multi_l2_plan.groupByL2NormOn(&.{ "store", "day" }, "amount", "amount_l2_lazy");
    const multi_l2_explained = try multi_l2_plan.explain(gpa);
    defer gpa.free(multi_l2_explained);
    try std.testing.expect(std.mem.indexOf(u8, multi_l2_explained, "group_by_l2_norm_on([store,day], value=amount -> amount_l2_lazy)") != null);
    var lazy_multi_l2 = try multi_l2_plan.collect();
    defer lazy_multi_l2.deinit();
    const lazy_ms_l2 = try (try lazy_multi_l2.column("amount_l2_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ms_l2);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 5.0)), lazy_ms_l2[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0), lazy_ms_l2[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), lazy_ms_l2[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 12.0), lazy_ms_l2[3], 1e-12);

    var multi_geometric_plan = try DeviceLazyFrame.init(gpa, multi);
    defer multi_geometric_plan.deinit();
    try multi_geometric_plan.groupByGeoMeanOn(&.{ "store", "day" }, "amount", "amount_geometric_lazy");
    const multi_geometric_explained = try multi_geometric_plan.explain(gpa);
    defer gpa.free(multi_geometric_explained);
    try std.testing.expect(std.mem.indexOf(u8, multi_geometric_explained, "group_by_geometric_mean_on([store,day], value=amount -> amount_geometric_lazy)") != null);
    var lazy_multi_geometric = try multi_geometric_plan.collect();
    defer lazy_multi_geometric.deinit();
    const lazy_ms_geometric = try (try lazy_multi_geometric.column("amount_geometric_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ms_geometric);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 2.0)), lazy_ms_geometric[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0), lazy_ms_geometric[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), lazy_ms_geometric[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 12.0), lazy_ms_geometric[3], 1e-12);

    var multi_harmonic_plan = try DeviceLazyFrame.init(gpa, multi);
    defer multi_harmonic_plan.deinit();
    try multi_harmonic_plan.groupByHarmonicMeanOn(&.{ "store", "day" }, "amount", "amount_harmonic_lazy");
    const multi_harmonic_explained = try multi_harmonic_plan.explain(gpa);
    defer gpa.free(multi_harmonic_explained);
    try std.testing.expect(std.mem.indexOf(u8, multi_harmonic_explained, "group_by_harmonic_mean_on([store,day], value=amount -> amount_harmonic_lazy)") != null);
    var lazy_multi_harmonic = try multi_harmonic_plan.collect();
    defer lazy_multi_harmonic.deinit();
    const lazy_ms_harmonic = try (try lazy_multi_harmonic.column("amount_harmonic_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ms_harmonic);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0 / 3.0), lazy_ms_harmonic[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0), lazy_ms_harmonic[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), lazy_ms_harmonic[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 12.0), lazy_ms_harmonic[3], 1e-12);

    var multi_logsumexp_plan = try DeviceLazyFrame.init(gpa, multi);
    defer multi_logsumexp_plan.deinit();
    try multi_logsumexp_plan.groupByLogsumexpOn(&.{ "store", "day" }, "amount", "amount_logsumexp_lazy");
    const multi_logsumexp_explained = try multi_logsumexp_plan.explain(gpa);
    defer gpa.free(multi_logsumexp_explained);
    try std.testing.expect(std.mem.indexOf(u8, multi_logsumexp_explained, "group_by_logsumexp_on([store,day], value=amount -> amount_logsumexp_lazy)") != null);
    var lazy_multi_logsumexp = try multi_logsumexp_plan.collect();
    defer lazy_multi_logsumexp.deinit();
    const lazy_ms_logsumexp = try (try lazy_multi_logsumexp.column("amount_logsumexp_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ms_logsumexp);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0) + std.math.log1p(std.math.exp(@as(f64, -1.0))), lazy_ms_logsumexp[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0), lazy_ms_logsumexp[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), lazy_ms_logsumexp[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 12.0), lazy_ms_logsumexp[3], 1e-12);

    var multi_logmeanexp_plan = try DeviceLazyFrame.init(gpa, multi);
    defer multi_logmeanexp_plan.deinit();
    try multi_logmeanexp_plan.groupByLogMeanExpOn(&.{ "store", "day" }, "amount", "amount_logmeanexp_lazy");
    const multi_logmeanexp_explained = try multi_logmeanexp_plan.explain(gpa);
    defer gpa.free(multi_logmeanexp_explained);
    try std.testing.expect(std.mem.indexOf(u8, multi_logmeanexp_explained, "group_by_logmeanexp_on([store,day], value=amount -> amount_logmeanexp_lazy)") != null);
    var lazy_multi_logmeanexp = try multi_logmeanexp_plan.collect();
    defer lazy_multi_logmeanexp.deinit();
    const lazy_ms_logmeanexp = try (try lazy_multi_logmeanexp.column("amount_logmeanexp_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ms_logmeanexp);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0) + std.math.log1p(std.math.exp(@as(f64, -1.0))) - std.math.ln2, lazy_ms_logmeanexp[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0), lazy_ms_logmeanexp[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), lazy_ms_logmeanexp[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 12.0), lazy_ms_logmeanexp[3], 1e-12);

    var multi_ptp_plan = try DeviceLazyFrame.init(gpa, multi);
    defer multi_ptp_plan.deinit();
    try multi_ptp_plan.groupByPTPOn(&.{ "store", "day" }, "amount", "amount_ptp_lazy");
    const multi_ptp_explained = try multi_ptp_plan.explain(gpa);
    defer gpa.free(multi_ptp_explained);
    try std.testing.expect(std.mem.indexOf(u8, multi_ptp_explained, "group_by_ptp_on([store,day], value=amount -> amount_ptp_lazy)") != null);
    var lazy_multi_ptp = try multi_ptp_plan.collect();
    defer lazy_multi_ptp.deinit();
    const lazy_ms_ptp = try (try lazy_multi_ptp.column("amount_ptp_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ms_ptp);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_ms_ptp[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_ms_ptp[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_ms_ptp[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_ms_ptp[3], 1e-12);

    var multi_range_coeff_plan = try DeviceLazyFrame.init(gpa, multi);
    defer multi_range_coeff_plan.deinit();
    try multi_range_coeff_plan.groupByRangeCoefficientOn(&.{ "store", "day" }, "amount", "amount_range_coeff_lazy");
    const multi_range_coeff_explained = try multi_range_coeff_plan.explain(gpa);
    defer gpa.free(multi_range_coeff_explained);
    try std.testing.expect(std.mem.indexOf(u8, multi_range_coeff_explained, "group_by_range_coeff_on([store,day], value=amount -> amount_range_coeff_lazy)") != null);
    var lazy_multi_range_coeff = try multi_range_coeff_plan.collect();
    defer lazy_multi_range_coeff.deinit();
    const lazy_ms_range_coeff = try (try lazy_multi_range_coeff.column("amount_range_coeff_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ms_range_coeff);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), lazy_ms_range_coeff[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_ms_range_coeff[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_ms_range_coeff[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_ms_range_coeff[3], 1e-12);

    var multi_skew_plan = try DeviceLazyFrame.init(gpa, multi);
    defer multi_skew_plan.deinit();
    try multi_skew_plan.groupBySkewnessOn(&.{ "store", "day" }, "amount", "amount_skewness_lazy");
    const multi_skew_explained = try multi_skew_plan.explain(gpa);
    defer gpa.free(multi_skew_explained);
    try std.testing.expect(std.mem.indexOf(u8, multi_skew_explained, "group_by_skewness_on([store,day], value=amount -> amount_skewness_lazy)") != null);
    var lazy_multi_skew = try multi_skew_plan.collect();
    defer lazy_multi_skew.deinit();
    const lazy_ms_skew = try (try lazy_multi_skew.column("amount_skewness_lazy")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ms_skew);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_ms_skew[0], 1e-12);
    try std.testing.expect(std.math.isNan(lazy_ms_skew[1]));
    try std.testing.expect(std.math.isNan(lazy_ms_skew[2]));
    try std.testing.expect(std.math.isNan(lazy_ms_skew[3]));

    var multi_stats = try multi.groupByStatsOn(&.{ "store", "day" }, "amount", "amount");
    defer multi_stats.deinit();
    try std.testing.expectEqual(@as(usize, 7), multi_stats.width());
    try std.testing.expectEqual(@as(usize, 4), multi_stats.height());
    const ms_store = try (try multi_stats.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(ms_store);
    const ms_day = try (try multi_stats.column("day")).i32.toOwnedSlice(gpa);
    defer gpa.free(ms_day);
    const ms_count = try (try multi_stats.column("amount_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(ms_count);
    const ms_sum = try (try multi_stats.column("amount_sum")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_sum);
    const ms_mean = try (try multi_stats.column("amount_mean")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_mean);
    try std.testing.expectEqualSlices(i32, &.{ 1, 1, 2, 2 }, ms_store);
    try std.testing.expectEqualSlices(i32, &.{ 10, 11, 10, 11 }, ms_day);
    try std.testing.expectEqualSlices(i64, &.{ 2, 1, 1, 1 }, ms_count);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 9.0, 4.0, 12.0 }, ms_sum);
    try std.testing.expectEqualSlices(f64, &.{ 1.5, 9.0, 4.0, 12.0 }, ms_mean);

    var multi_profile = try multi.groupByProfileOn(&.{ "store", "day" }, "amount", "amount");
    defer multi_profile.deinit();
    try std.testing.expect(multi_profile.width() == 9);
    try std.testing.expect(multi_profile.height() == 4);
    const mp_store = try (try multi_profile.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(mp_store);
    const mp_day = try (try multi_profile.column("day")).i32.toOwnedSlice(gpa);
    defer gpa.free(mp_day);
    const mp_count = try (try multi_profile.column("amount_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(mp_count);
    const mp_variance = try (try multi_profile.column("amount_variance")).f64.toOwnedSlice(gpa);
    defer gpa.free(mp_variance);
    const mp_stddev = try (try multi_profile.column("amount_stddev")).f64.toOwnedSlice(gpa);
    defer gpa.free(mp_stddev);
    const mp_skewness = try (try multi_profile.column("amount_skewness")).f64.toOwnedSlice(gpa);
    defer gpa.free(mp_skewness);
    const mp_kurtosis = try (try multi_profile.column("amount_kurtosis")).f64.toOwnedSlice(gpa);
    defer gpa.free(mp_kurtosis);
    try std.testing.expectEqualSlices(i32, &.{ 1, 1, 2, 2 }, mp_store);
    try std.testing.expectEqualSlices(i32, &.{ 10, 11, 10, 11 }, mp_day);
    try std.testing.expectEqualSlices(i64, &.{ 2, 1, 1, 1 }, mp_count);
    const expected_mp_variance: f64 = 0.25;
    const expected_mp_stddev: f64 = 0.5;
    const expected_mp_skewness: f64 = 0.0;
    const expected_mp_kurtosis: f64 = -2.0;
    try std.testing.expect(@abs(mp_variance[0] - expected_mp_variance) <= 1e-12);
    try std.testing.expect(@abs(mp_stddev[0] - expected_mp_stddev) <= 1e-12);
    try std.testing.expect(@abs(mp_skewness[0] - expected_mp_skewness) <= 1e-12);
    try std.testing.expect(@abs(mp_kurtosis[0] - expected_mp_kurtosis) <= 1e-12);
    try std.testing.expect(std.math.isNan(mp_skewness[1]));
    try std.testing.expect(std.math.isNan(mp_kurtosis[1]));
}
