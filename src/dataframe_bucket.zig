const std = @import("std");

pub const BucketMetrics = struct {
    allocator: std.mem.Allocator,
    ecdf: []f64,
    buckets: []i64,
    lower_tail: []bool,
    upper_tail: []bool,
    validity: []bool,

    pub fn deinit(self: *BucketMetrics) void {
        self.allocator.free(self.ecdf);
        self.allocator.free(self.buckets);
        self.allocator.free(self.lower_tail);
        self.allocator.free(self.upper_tail);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

fn validate(values_len: usize, maybe_validity: ?[]const bool, buckets: usize, lower_quantile: f64, upper_quantile: f64, min_periods: usize) error{ InvalidShape, LengthMismatch }!void {
    if (buckets == 0 or min_periods == 0) return error.InvalidShape;
    if (lower_quantile < 0 or lower_quantile > 1) return error.InvalidShape;
    if (upper_quantile < 0 or upper_quantile > 1) return error.InvalidShape;
    if (lower_quantile > upper_quantile) return error.InvalidShape;
    if (maybe_validity) |validity| {
        if (validity.len != values_len) return error.LengthMismatch;
    }
}

fn rowValid(maybe_validity: ?[]const bool, row: usize) bool {
    return if (maybe_validity) |mask| mask[row] else true;
}

pub fn bucketProfile(
    allocator: std.mem.Allocator,
    order: []const usize,
    maybe_validity: ?[]const bool,
    keysTie: *const fn (usize, usize) bool,
    buckets_count: usize,
    lower_quantile: f64,
    upper_quantile: f64,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!BucketMetrics {
    try validate(order.len, maybe_validity, buckets_count, lower_quantile, upper_quantile, min_periods);

    const ecdf = try allocator.alloc(f64, order.len);
    errdefer allocator.free(ecdf);
    const buckets = try allocator.alloc(i64, order.len);
    errdefer allocator.free(buckets);
    const lower_tail = try allocator.alloc(bool, order.len);
    errdefer allocator.free(lower_tail);
    const upper_tail = try allocator.alloc(bool, order.len);
    errdefer allocator.free(upper_tail);
    const validity = try allocator.alloc(bool, order.len);
    errdefer allocator.free(validity);

    @memset(ecdf, 0);
    @memset(buckets, 0);
    @memset(lower_tail, false);
    @memset(upper_tail, false);
    @memset(validity, false);

    var valid_count: usize = 0;
    for (0..order.len) |row| {
        if (rowValid(maybe_validity, row)) valid_count += 1;
    }

    if (valid_count >= min_periods and valid_count != 0) {
        var group_start: usize = 0;
        while (group_start < valid_count) {
            var group_end = group_start + 1;
            while (group_end < valid_count and keysTie(order[group_start], order[group_end])) {
                group_end += 1;
            }

            const rank_position = group_end; // right-continuous ECDF, 1-based.
            const ecdf_value = @as(f64, @floatFromInt(rank_position)) / @as(f64, @floatFromInt(valid_count));
            var bucket_index = @divFloor((rank_position - 1) * buckets_count, valid_count);
            if (bucket_index >= buckets_count) bucket_index = buckets_count - 1;
            const is_lower = ecdf_value <= lower_quantile;
            const is_upper = ecdf_value >= upper_quantile;

            for (order[group_start..group_end]) |row| {
                ecdf[row] = ecdf_value;
                buckets[row] = @intCast(bucket_index);
                lower_tail[row] = is_lower;
                upper_tail[row] = is_upper;
                validity[row] = true;
            }
            group_start = group_end;
        }
    }

    return .{
        .allocator = allocator,
        .ecdf = ecdf,
        .buckets = buckets,
        .lower_tail = lower_tail,
        .upper_tail = upper_tail,
        .validity = validity,
    };
}
