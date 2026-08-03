const std = @import("std");
const vectra = @import("vectra");

const DeviceLazyFrame = vectra.DeviceLazyFrame;
const lazyCollectTable = @import("test_helpers.zig").lazyCollectTable;

test "device lazy frame collects bucket operations" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();
    var bucket_plan = try DeviceLazyFrame.init(gpa, table);
    defer bucket_plan.deinit();
    try bucket_plan.bucketProfile("sales", "sales", .{ .buckets = 2, .lower_quantile = 0.25, .upper_quantile = 0.75 });
    try bucket_plan.select(&.{ "sales", "sales_ecdf", "sales_bucket", "sales_lower_tail", "sales_upper_tail" });
    const bucket_explain = try bucket_plan.explain(gpa);
    defer gpa.free(bucket_explain);
    try std.testing.expect(std.mem.indexOf(u8, bucket_explain, "bucket_profile(sales") != null);
    var bucketed = try bucket_plan.collect();
    defer bucketed.deinit();
    try std.testing.expectEqual(@as(usize, 4), bucketed.height());
    try std.testing.expectEqual(@as(usize, 5), bucketed.width());
    const lazy_ecdf = try (try bucketed.column("sales_ecdf")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ecdf);
    const lazy_bucket = try (try bucketed.column("sales_bucket")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_bucket);
    const lazy_lower_tail = try (try bucketed.column("sales_lower_tail")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_lower_tail);
    const lazy_upper_tail = try (try bucketed.column("sales_upper_tail")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_upper_tail);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), lazy_ecdf[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_ecdf[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), lazy_ecdf[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_ecdf[3], 1e-12);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 1 }, lazy_bucket);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false }, lazy_lower_tail);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, true }, lazy_upper_tail);
}
