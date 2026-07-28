const std = @import("std");
const vectra = @import("vectra");

const DeviceLazyFrame = vectra.DeviceLazyFrame;
const lazyCollectTable = @import("dataframe_lazy_test_helpers.zig").lazyCollectTable;

test "device lazy frame collects lag operations" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();
    var lag_plan = try DeviceLazyFrame.init(gpa, table);
    defer lag_plan.deinit();
    try lag_plan.lagProfile("sales", "sales", .{ .periods = 1 });
    try lag_plan.select(&.{ "sales", "sales_lag", "sales_diff", "sales_pct_change" });
    const lag_explain = try lag_plan.explain(gpa);
    defer gpa.free(lag_explain);
    try std.testing.expect(std.mem.indexOf(u8, lag_explain, "lag_profile(sales") != null);
    var lagged = try lag_plan.collect();
    defer lagged.deinit();
    try std.testing.expectEqual(@as(usize, 4), lagged.height());
    try std.testing.expectEqual(@as(usize, 4), lagged.width());
    const lazy_lag = try (try lagged.column("sales_lag")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_lag);
    const lazy_diff = try (try lagged.column("sales_diff")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_diff);
    const lazy_pct = try (try lagged.column("sales_pct_change")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_pct);
    const lazy_lag_validity = try (try lagged.column("sales_lag")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_lag_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_lag_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_lag[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), lazy_lag[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), lazy_lag[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_diff[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_diff[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_diff[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_pct[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), lazy_pct[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.4), lazy_pct[3], 1e-12);
}

test "device lazy frame collects lead operations" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();
    var lead_plan = try DeviceLazyFrame.init(gpa, table);
    defer lead_plan.deinit();
    try lead_plan.leadProfile("sales", "sales", .{ .periods = 1 });
    try lead_plan.select(&.{ "sales", "sales_lead", "sales_forward_diff", "sales_forward_pct_change" });
    const lead_explain = try lead_plan.explain(gpa);
    defer gpa.free(lead_explain);
    try std.testing.expect(std.mem.indexOf(u8, lead_explain, "lead_profile(sales") != null);
    var leaded = try lead_plan.collect();
    defer leaded.deinit();
    try std.testing.expectEqual(@as(usize, 4), leaded.height());
    try std.testing.expectEqual(@as(usize, 4), leaded.width());
    const lazy_lead = try (try leaded.column("sales_lead")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_lead);
    const lazy_forward_diff = try (try leaded.column("sales_forward_diff")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_forward_diff);
    const lazy_forward_pct = try (try leaded.column("sales_forward_pct_change")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_forward_pct);
    const lazy_lead_validity = try (try leaded.column("sales_lead")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_lead_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false }, lazy_lead_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), lazy_lead[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), lazy_lead[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 7.0), lazy_lead[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_forward_diff[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_forward_diff[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_forward_diff[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_forward_pct[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), lazy_forward_pct[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.4), lazy_forward_pct[2], 1e-12);
}
