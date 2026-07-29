const std = @import("std");
const vectra = @import("vectra");

const DeviceColumn = vectra.DeviceColumn;
const DeviceDataFrame = vectra.DeviceDataFrame;

test "device dataframe robust profile methods" {
    const gpa = std.testing.allocator;

    var robust_source = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, 2.0, 3.0, 100.0, 0.0 }, &.{ true, true, true, true, false }, .cpu);
    defer robust_source.deinit();
    var robust_id = try DeviceColumn.fromSlice(i64, gpa, &.{ 1, 2, 3, 4, 5 }, .cpu);
    defer robust_id.deinit();
    var robust_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "value", .data = robust_source },
        .{ .name = "id", .data = robust_id },
    });
    defer robust_table.deinit();

    var robust = try robust_table.robustProfile("value", "value", .{ .min_periods = 4 });
    defer robust.deinit();
    try std.testing.expectEqual(@as(usize, 6), robust.width());
    const median_centered = try (try robust.column("value_median_centered")).f64.toOwnedSlice(gpa);
    defer gpa.free(median_centered);
    const mad_zscore = try (try robust.column("value_mad_zscore")).f64.toOwnedSlice(gpa);
    defer gpa.free(mad_zscore);
    const iqr_outlier = try (try robust.column("value_iqr_outlier")).bool.toOwnedSlice(gpa);
    defer gpa.free(iqr_outlier);
    const winsorized = try (try robust.column("value_winsorized")).f64.toOwnedSlice(gpa);
    defer gpa.free(winsorized);
    const robust_validity = try (try robust.column("value_mad_zscore")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(robust_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, false }, robust_validity);
    try std.testing.expectApproxEqAbs(@as(f64, -1.5), median_centered[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.5), median_centered[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), median_centered[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 97.5), median_centered[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -1.0117346252941226), mad_zscore[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.33724487509804085), mad_zscore[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.33724487509804085), mad_zscore[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 65.76275064411797), mad_zscore[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, true, false }, iqr_outlier);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), winsorized[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), winsorized[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), winsorized[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 65.5), winsorized[3], 1e-12);

    var expanding_robust = try robust_table.expandingRobustProfile("value", "value", .{ .min_periods = 2 });
    defer expanding_robust.deinit();
    try std.testing.expectEqual(@as(usize, 6), expanding_robust.width());
    const expanding_median_centered = try (try expanding_robust.column("value_expanding_median_centered")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_median_centered);
    const expanding_mad_zscore = try (try expanding_robust.column("value_expanding_mad_zscore")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_mad_zscore);
    const expanding_iqr_outlier = try (try expanding_robust.column("value_expanding_iqr_outlier")).bool.toOwnedSlice(gpa);
    defer gpa.free(expanding_iqr_outlier);
    const expanding_winsorized = try (try expanding_robust.column("value_expanding_winsorized")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_winsorized);
    const expanding_robust_validity = try (try expanding_robust.column("value_expanding_mad_zscore")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(expanding_robust_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, false }, expanding_robust_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), expanding_median_centered[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), expanding_median_centered[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 97.5), expanding_median_centered[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.6744897501960817), expanding_mad_zscore[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.6744897501960817), expanding_mad_zscore[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 65.76275064411797), expanding_mad_zscore[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, true, false }, expanding_iqr_outlier);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), expanding_winsorized[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), expanding_winsorized[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 65.5), expanding_winsorized[3], 1e-12);
}
