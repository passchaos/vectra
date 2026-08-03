//! Grouped moment metric accumulation and dataframe materialization helpers.

const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const dataframe_device_column_mod = @import("dataframe/device_column.zig");
const names_mod = @import("dataframe_names.zig");

const DeviceColumn = dataframe_device_column_mod.DeviceColumn;

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

    pub fn sem(self: MomentProfile) f64 {
        if (self.count == 0) return std.math.nan(f64);
        return self.stddev() / std.math.sqrt(@as(f64, @floatFromInt(self.count)));
    }

    pub fn cv(self: MomentProfile) f64 {
        if (self.count == 0) return std.math.nan(f64);
        return self.stddev() / self.mean;
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

pub fn initProfileDataFrame(
    comptime DeviceDataFrame: type,
    allocator: std.mem.Allocator,
    key_names: []const []const u8,
    output_prefix: []const u8,
    key_columns: []const DeviceColumn,
    metrics: MetricSlices,
    device_value: array_mod.Device,
) (array_mod.ArrayError || std.mem.Allocator.Error || error{ LengthMismatch, TypeMismatch, TypeUnsupported, InvalidDevice })!DeviceDataFrame {
    if (key_columns.len != key_names.len) return error.LengthMismatch;
    const rows = metrics.counts.len;
    const names = try names_mod.profileOutputNames(allocator, key_names, output_prefix);
    defer names_mod.freeProfileOutputNames(allocator, names, key_names.len);

    var columns = try allocator.alloc(DeviceColumn, key_names.len + 7);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        allocator.free(columns);
    }

    for (key_columns) |key_col| {
        if (key_col.len() != rows) return error.LengthMismatch;
        columns[initialized] = try key_col.clone();
        initialized += 1;
    }
    columns[initialized] = try DeviceColumn.fromSlice(i64, allocator, metrics.counts, device_value);
    initialized += 1;
    columns[initialized] = try DeviceColumn.fromSlice(f64, allocator, metrics.sums, device_value);
    initialized += 1;
    columns[initialized] = try DeviceColumn.fromSlice(f64, allocator, metrics.means, device_value);
    initialized += 1;
    columns[initialized] = try DeviceColumn.fromSlice(f64, allocator, metrics.variances, device_value);
    initialized += 1;
    columns[initialized] = try DeviceColumn.fromSlice(f64, allocator, metrics.stddevs, device_value);
    initialized += 1;
    columns[initialized] = try DeviceColumn.fromSlice(f64, allocator, metrics.skewnesses, device_value);
    initialized += 1;
    columns[initialized] = try DeviceColumn.fromSlice(f64, allocator, metrics.kurtoses, device_value);
    initialized += 1;

    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, allocator, names, columns, rows, device_value);
}
