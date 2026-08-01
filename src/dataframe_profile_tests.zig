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

    try std.testing.expectError(error.TypeUnsupported, bool_table.groupByAny("store", "day", "bad_any"));

    var key = try DeviceColumn.fromSliceWithValidity(i32, gpa, &.{ 1, 2, 1, 3, 2, 1 }, &.{ true, true, true, false, true, true }, .cpu);
    defer key.deinit();
    var sales = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 2.0, 3.0, 5.0, 7.0, 11.0, 13.0 }, &.{ true, true, false, true, true, true }, .cpu);
    defer sales.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "store", .data = key },
        .{ .name = "sales", .data = sales },
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
    try std.testing.expectEqual(@as(usize, 9), multi_profile.width());
    try std.testing.expectEqual(@as(usize, 4), multi_profile.height());
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
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), mp_variance[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), mp_stddev[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), mp_skewness[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -2.0), mp_kurtosis[0], 1e-12);
    try std.testing.expect(std.math.isNan(mp_skewness[1]));
    try std.testing.expect(std.math.isNan(mp_kurtosis[1]));
}
