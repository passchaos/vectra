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

test "device dataframe sorts and rank profiles" {
    const gpa = std.testing.allocator;

    var score = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 3.0, 1.0, 2.0, 4.0 }, &.{ true, true, false, true }, .cpu);
    defer score.deinit();
    var id = try DeviceColumn.fromSlice(i64, gpa, &.{ 30, 10, 20, 40 }, .cpu);
    defer id.deinit();
    var flag = try DeviceColumn.fromSlice(bool, gpa, &.{ true, false, true, false }, .cpu);
    defer flag.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "score", .data = score },
        .{ .name = "id", .data = id },
        .{ .name = "flag", .data = flag },
    });
    defer table.deinit();

    const asc = try table.argsortBy("score", .{ .descending = false, .nulls = .last });
    defer gpa.free(asc);
    try std.testing.expectEqualSlices(usize, &.{ 1, 0, 3, 2 }, asc);

    var sorted = try table.sortBy("score", .{ .descending = false, .nulls = .last });
    defer sorted.deinit();
    const sorted_id = try sorted.column("id");
    const sorted_id_values = try sorted_id.i64.toOwnedSlice(gpa);
    defer gpa.free(sorted_id_values);
    try std.testing.expectEqualSlices(i64, &.{ 10, 30, 40, 20 }, sorted_id_values);

    var desc_nulls_first = try table.sortBy("score", .{ .descending = true, .nulls = .first });
    defer desc_nulls_first.deinit();
    const desc_id = try desc_nulls_first.column("id");
    const desc_id_values = try desc_id.i64.toOwnedSlice(gpa);
    defer gpa.free(desc_id_values);
    try std.testing.expectEqualSlices(i64, &.{ 20, 40, 30, 10 }, desc_id_values);

    var bool_sorted = try table.sortBy("flag", .{});
    defer bool_sorted.deinit();
    const bool_sorted_id = try bool_sorted.column("id");
    const bool_sorted_id_values = try bool_sorted_id.i64.toOwnedSlice(gpa);
    defer gpa.free(bool_sorted_id_values);
    try std.testing.expectEqualSlices(i64, &.{ 10, 40, 30, 20 }, bool_sorted_id_values);

    var group = try DeviceColumn.fromSlice(i64, gpa, &.{ 1, 1, 2, 1, 2 }, .cpu);
    defer group.deinit();
    var multi_score = try DeviceColumn.fromSlice(i64, gpa, &.{ 2, 1, 1, 3, 0 }, .cpu);
    defer multi_score.deinit();
    var multi_id = try DeviceColumn.fromSlice(i64, gpa, &.{ 0, 1, 2, 3, 4 }, .cpu);
    defer multi_id.deinit();
    var multi_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "group", .data = group },
        .{ .name = "score", .data = multi_score },
        .{ .name = "id", .data = multi_id },
    });
    defer multi_table.deinit();

    const multi_order = try multi_table.argsortByColumns(&.{ "group", "score" }, &.{ .{ .descending = false }, .{ .descending = true } });
    defer gpa.free(multi_order);
    try std.testing.expectEqualSlices(usize, &.{ 3, 0, 1, 2, 4 }, multi_order);

    var multi_sorted = try multi_table.sortByColumns(&.{ "group", "score" }, &.{ .{ .descending = false }, .{ .descending = true } });
    defer multi_sorted.deinit();
    const multi_sorted_id = try (try multi_sorted.column("id")).i64.toOwnedSlice(gpa);
    defer gpa.free(multi_sorted_id);
    try std.testing.expectEqualSlices(i64, &.{ 3, 0, 1, 2, 4 }, multi_sorted_id);

    var multi_top2 = try multi_table.topKByColumns(&.{ "group", "score" }, 2, &.{ .{ .descending = false }, .{ .descending = true } });
    defer multi_top2.deinit();
    const multi_top2_id = try (try multi_top2.column("id")).i64.toOwnedSlice(gpa);
    defer gpa.free(multi_top2_id);
    try std.testing.expectEqualSlices(i64, &.{ 3, 0 }, multi_top2_id);

    var bottom2 = try table.bottomKBy("score", 2, .{ .nulls = .last });
    defer bottom2.deinit();
    const bottom2_id = try (try bottom2.column("id")).i64.toOwnedSlice(gpa);
    defer gpa.free(bottom2_id);
    try std.testing.expectEqualSlices(i64, &.{ 10, 30 }, bottom2_id);

    var multi_bottom2 = try multi_table.bottomKByColumns(&.{ "group", "score" }, 2, &.{ .{ .descending = false }, .{ .descending = false } });
    defer multi_bottom2.deinit();
    const multi_bottom2_id = try (try multi_bottom2.column("id")).i64.toOwnedSlice(gpa);
    defer gpa.free(multi_bottom2_id);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0 }, multi_bottom2_id);
    try std.testing.expectError(error.LengthMismatch, multi_table.sortByColumns(&.{"group"}, &.{ .{ .descending = false }, .{ .descending = true } }));

    var tied_score = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 10.0, 20.0, 20.0, 30.0, 0.0 }, &.{ true, true, true, true, false }, .cpu);
    defer tied_score.deinit();
    var tied_id = try DeviceColumn.fromSlice(i64, gpa, &.{ 1, 2, 3, 4, 5 }, .cpu);
    defer tied_id.deinit();
    var tied_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "score", .data = tied_score },
        .{ .name = "id", .data = tied_id },
    });
    defer tied_table.deinit();

    var ranks = try tied_table.rankProfileBy("score", "score", .{ .descending = false, .nulls = .last });
    defer ranks.deinit();
    try std.testing.expectEqual(@as(usize, 7), ranks.width());
    const ordinal = try (try ranks.column("score_ordinal_rank")).i64.toOwnedSlice(gpa);
    defer gpa.free(ordinal);
    const competition = try (try ranks.column("score_competition_rank")).i64.toOwnedSlice(gpa);
    defer gpa.free(competition);
    const dense_rank = try (try ranks.column("score_dense_rank")).i64.toOwnedSlice(gpa);
    defer gpa.free(dense_rank);
    const percent_rank = try (try ranks.column("score_percent_rank")).f64.toOwnedSlice(gpa);
    defer gpa.free(percent_rank);
    const cume_dist = try (try ranks.column("score_cume_dist")).f64.toOwnedSlice(gpa);
    defer gpa.free(cume_dist);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4, 5 }, ordinal);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 2, 4, 5 }, competition);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 2, 3, 4 }, dense_rank);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), percent_rank[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), percent_rank[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), percent_rank[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), percent_rank[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), percent_rank[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.2), cume_dist[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.6), cume_dist[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.6), cume_dist[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.8), cume_dist[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), cume_dist[4], 1e-12);

    var desc_ranks = try tied_table.rankProfileBy("score", "score_desc", .{ .descending = true, .nulls = .first });
    defer desc_ranks.deinit();
    const desc_competition = try (try desc_ranks.column("score_desc_competition_rank")).i64.toOwnedSlice(gpa);
    defer gpa.free(desc_competition);
    const desc_cume_dist = try (try desc_ranks.column("score_desc_cume_dist")).f64.toOwnedSlice(gpa);
    defer gpa.free(desc_cume_dist);
    try std.testing.expectEqualSlices(i64, &.{ 5, 3, 3, 2, 1 }, desc_competition);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), desc_cume_dist[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.8), desc_cume_dist[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.8), desc_cume_dist[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.4), desc_cume_dist[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.2), desc_cume_dist[4], 1e-12);

    var rolling_ranks = try tied_table.rollingRankProfile("score", "score_roll", .{ .window = 3, .min_periods = 2 });
    defer rolling_ranks.deinit();
    try std.testing.expectEqual(@as(usize, 6), rolling_ranks.width());
    const rolling_rank_count = try (try rolling_ranks.column("score_roll_rolling_rank_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(rolling_rank_count);
    const rolling_rank = try (try rolling_ranks.column("score_roll_rolling_rank")).i64.toOwnedSlice(gpa);
    defer gpa.free(rolling_rank);
    const rolling_percent_rank = try (try rolling_ranks.column("score_roll_rolling_percent_rank")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_percent_rank);
    const rolling_rank_cume = try (try rolling_ranks.column("score_roll_rolling_cume_dist")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_rank_cume);
    const rolling_rank_validity = try (try rolling_ranks.column("score_roll_rolling_rank")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(rolling_rank_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 3, 2 }, rolling_rank_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, false }, rolling_rank_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 2, 2, 3, 0 }, rolling_rank);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_percent_rank[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_percent_rank[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_percent_rank[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_rank_cume[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_rank_cume[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_rank_cume[3], 1e-12);

    var expanding_ranks = try tied_table.expandingRankProfile("score", "score_expand", .{ .min_periods = 2 });
    defer expanding_ranks.deinit();
    try std.testing.expectEqual(@as(usize, 6), expanding_ranks.width());
    const expanding_rank_count = try (try expanding_ranks.column("score_expand_expanding_rank_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(expanding_rank_count);
    const expanding_rank = try (try expanding_ranks.column("score_expand_expanding_rank")).i64.toOwnedSlice(gpa);
    defer gpa.free(expanding_rank);
    const expanding_percent_rank = try (try expanding_ranks.column("score_expand_expanding_percent_rank")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_percent_rank);
    const expanding_rank_cume = try (try expanding_ranks.column("score_expand_expanding_cume_dist")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_rank_cume);
    const expanding_rank_validity = try (try expanding_ranks.column("score_expand_expanding_rank")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(expanding_rank_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4, 4 }, expanding_rank_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, false }, expanding_rank_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 2, 2, 4, 0 }, expanding_rank);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), expanding_percent_rank[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_percent_rank[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), expanding_percent_rank[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), expanding_rank_cume[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), expanding_rank_cume[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), expanding_rank_cume[3], 1e-12);
}
