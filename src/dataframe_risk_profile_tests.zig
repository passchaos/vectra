const std = @import("std");
const vectra = @import("vectra");

const DeviceColumn = vectra.DeviceColumn;
const DeviceDataFrame = vectra.DeviceDataFrame;

test "device dataframe risk profile methods" {
    const gpa = std.testing.allocator;

    var equity = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 100.0, 120.0, 90.0, 130.0, 80.0, 0.0 }, &.{ true, true, true, true, true, false }, .cpu);
    defer equity.deinit();
    var equity_id = try DeviceColumn.fromSlice(i64, gpa, &.{ 1, 2, 3, 4, 5, 6 }, .cpu);
    defer equity_id.deinit();
    var equity_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "equity", .data = equity },
        .{ .name = "id", .data = equity_id },
    });
    defer equity_table.deinit();

    var drawdown = try equity_table.drawdownProfile("equity", "equity", .{ .min_periods = 2 });
    defer drawdown.deinit();
    try std.testing.expectEqual(@as(usize, 5), drawdown.width());
    const running_peak = try (try drawdown.column("equity_running_peak")).f64.toOwnedSlice(gpa);
    defer gpa.free(running_peak);
    const drawdown_values = try (try drawdown.column("equity_drawdown")).f64.toOwnedSlice(gpa);
    defer gpa.free(drawdown_values);
    const drawdown_pct = try (try drawdown.column("equity_drawdown_pct")).f64.toOwnedSlice(gpa);
    defer gpa.free(drawdown_pct);
    const drawdown_validity = try (try drawdown.column("equity_drawdown")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(drawdown_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true, false }, drawdown_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 120.0), running_peak[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 120.0), running_peak[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 130.0), running_peak[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 130.0), running_peak[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), drawdown_values[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -30.0), drawdown_values[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), drawdown_values[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -50.0), drawdown_values[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), drawdown_pct[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.25), drawdown_pct[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), drawdown_pct[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -50.0 / 130.0), drawdown_pct[4], 1e-12);

    var rolling_drawdown = try equity_table.rollingDrawdownProfile("equity", "equity", .{ .window = 3, .min_periods = 2 });
    defer rolling_drawdown.deinit();
    try std.testing.expectEqual(@as(usize, 6), rolling_drawdown.width());
    const rolling_peak = try (try rolling_drawdown.column("equity_rolling_peak")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_peak);
    const rolling_drawdown_values = try (try rolling_drawdown.column("equity_rolling_drawdown")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_drawdown_values);
    const rolling_drawdown_pct = try (try rolling_drawdown.column("equity_rolling_drawdown_pct")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_drawdown_pct);
    const rolling_peak_age = try (try rolling_drawdown.column("equity_rolling_peak_age")).i64.toOwnedSlice(gpa);
    defer gpa.free(rolling_peak_age);
    const rolling_drawdown_validity = try (try rolling_drawdown.column("equity_rolling_drawdown")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(rolling_drawdown_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true, false }, rolling_drawdown_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 120.0), rolling_peak[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 120.0), rolling_peak[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 130.0), rolling_peak[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 130.0), rolling_peak[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), rolling_drawdown_values[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -30.0), rolling_drawdown_values[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), rolling_drawdown_values[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -50.0), rolling_drawdown_values[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), rolling_drawdown_pct[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.25), rolling_drawdown_pct[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), rolling_drawdown_pct[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -50.0 / 130.0), rolling_drawdown_pct[4], 1e-12);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 0, 1, 0 }, rolling_peak_age);

    var extrema = try equity_table.extremaProfile("equity", "equity", .{ .min_periods = 2 });
    defer extrema.deinit();
    try std.testing.expectEqual(@as(usize, 6), extrema.width());
    const running_low = try (try extrema.column("equity_running_low")).f64.toOwnedSlice(gpa);
    defer gpa.free(running_low);
    const running_high = try (try extrema.column("equity_running_high")).f64.toOwnedSlice(gpa);
    defer gpa.free(running_high);
    const new_low = try (try extrema.column("equity_new_low")).bool.toOwnedSlice(gpa);
    defer gpa.free(new_low);
    const new_high = try (try extrema.column("equity_new_high")).bool.toOwnedSlice(gpa);
    defer gpa.free(new_high);
    const extrema_validity = try (try extrema.column("equity_running_low")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(extrema_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true, false }, extrema_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 100.0), running_low[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 90.0), running_low[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 90.0), running_low[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 80.0), running_low[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 120.0), running_high[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 120.0), running_high[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 130.0), running_high[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 130.0), running_high[4], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, true, false }, new_low);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true, false, false }, new_high);
}
