const std = @import("std");

pub const MomentProfile = struct {
    count: i64 = 0,
    sum: f64 = 0,
    mean: f64 = 0,
    m2: f64 = 0,
    m3: f64 = 0,
    m4: f64 = 0,

    pub fn update(self: *MomentProfile, value: f64) void {
        const previous_count = self.count;
        self.count += 1;

        const n: f64 = @floatFromInt(self.count);
        const previous_n: f64 = @floatFromInt(previous_count);
        const delta = value - self.mean;
        const delta_n = delta / n;
        const delta_n2 = delta_n * delta_n;
        const term1 = delta * delta_n * previous_n;
        const previous_m2 = self.m2;
        const previous_m3 = self.m3;

        self.mean += delta_n;
        self.m4 += term1 * delta_n2 * (n * n - 3.0 * n + 3.0) + 6.0 * delta_n2 * previous_m2 - 4.0 * delta_n * previous_m3;
        self.m3 += term1 * delta_n * (n - 2.0) - 3.0 * delta_n * previous_m2;
        self.m2 += term1;
        self.sum += value;
    }

    pub fn variance(self: MomentProfile) f64 {
        if (self.count == 0) return std.math.nan(f64);
        return self.m2 / @as(f64, @floatFromInt(self.count));
    }

    pub fn stddev(self: MomentProfile) f64 {
        return std.math.sqrt(self.variance());
    }

    pub fn skewness(self: MomentProfile) f64 {
        if (self.count < 2 or self.m2 == 0) return std.math.nan(f64);
        const n: f64 = @floatFromInt(self.count);
        return std.math.sqrt(n) * self.m3 / std.math.pow(f64, self.m2, 1.5);
    }

    pub fn kurtosis(self: MomentProfile) f64 {
        if (self.count < 2 or self.m2 == 0) return std.math.nan(f64);
        const n: f64 = @floatFromInt(self.count);
        return n * self.m4 / (self.m2 * self.m2) - 3.0;
    }
};

pub const MetricSlices = struct {
    allocator: std.mem.Allocator,
    counts: []i64,
    sums: []f64,
    means: []f64,
    variances: []f64,
    stddevs: []f64,
    skewnesses: []f64,
    kurtoses: []f64,

    pub fn deinit(self: *MetricSlices) void {
        self.allocator.free(self.counts);
        self.allocator.free(self.sums);
        self.allocator.free(self.means);
        self.allocator.free(self.variances);
        self.allocator.free(self.stddevs);
        self.allocator.free(self.skewnesses);
        self.allocator.free(self.kurtoses);
        self.* = undefined;
    }
};

pub fn materializeMetrics(allocator: std.mem.Allocator, profiles: []const MomentProfile) std.mem.Allocator.Error!MetricSlices {
    const counts = try allocator.alloc(i64, profiles.len);
    errdefer allocator.free(counts);
    const sums = try allocator.alloc(f64, profiles.len);
    errdefer allocator.free(sums);
    const means = try allocator.alloc(f64, profiles.len);
    errdefer allocator.free(means);
    const variances = try allocator.alloc(f64, profiles.len);
    errdefer allocator.free(variances);
    const stddevs = try allocator.alloc(f64, profiles.len);
    errdefer allocator.free(stddevs);
    const skewnesses = try allocator.alloc(f64, profiles.len);
    errdefer allocator.free(skewnesses);
    const kurtoses = try allocator.alloc(f64, profiles.len);
    errdefer allocator.free(kurtoses);

    for (profiles, 0..) |profile, i| {
        counts[i] = profile.count;
        sums[i] = profile.sum;
        means[i] = profile.mean;
        variances[i] = profile.variance();
        stddevs[i] = profile.stddev();
        skewnesses[i] = profile.skewness();
        kurtoses[i] = profile.kurtosis();
    }

    return .{
        .allocator = allocator,
        .counts = counts,
        .sums = sums,
        .means = means,
        .variances = variances,
        .stddevs = stddevs,
        .skewnesses = skewnesses,
        .kurtoses = kurtoses,
    };
}
