const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const dataframe_device_column_mod = @import("dataframe_device_column.zig");
const numeric_mod = @import("dataframe_numeric.zig");
const options_mod = @import("dataframe_options.zig");
const validity_mod = @import("dataframe_validity.zig");

const DeviceColumn = dataframe_device_column_mod.DeviceColumn;
const DeviceTypedColumn = dataframe_device_column_mod.DeviceTypedColumn;
const DeviceGroupByAggregation = options_mod.DeviceGroupByAggregation;
const findGroupIndex = numeric_mod.findGroupIndex;
const compareSortValues = numeric_mod.compareSortValues;
const castToF64 = numeric_mod.castToF64;
const validityValues = validity_mod.validityValues;

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

pub fn groupByCountTyped(
    comptime DeviceDataFrame: type,
    comptime K: type,
    allocator: std.mem.Allocator,
    key_name: []const u8,
    output_name: []const u8,
    key: DeviceTypedColumn(K),
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{ LengthMismatch, InvalidDevice })!DeviceDataFrame {
    const keys = try key.values.toOwnedSlice(allocator);
    defer allocator.free(keys);
    const maybe_key_validity = try validityValues(key, allocator);
    defer if (maybe_key_validity) |validity| allocator.free(validity);

    var unique_keys: std.ArrayList(K) = .empty;
    defer unique_keys.deinit(allocator);
    var counts: std.ArrayList(i64) = .empty;
    defer counts.deinit(allocator);

    for (keys, 0..) |key_value, row| {
        if (maybe_key_validity) |validity| {
            if (!validity[row]) continue;
        }
        const group_index = findGroupIndex(K, unique_keys.items, key_value) orelse blk: {
            try unique_keys.append(allocator, key_value);
            try counts.append(allocator, 0);
            break :blk unique_keys.items.len - 1;
        };
        counts.items[group_index] += 1;
    }

    const key_col = try DeviceColumn.fromSlice(K, allocator, unique_keys.items, device_value);
    const count_col = try DeviceColumn.fromSlice(i64, allocator, counts.items, device_value);
    return initAggregatedDataFrame(DeviceDataFrame, allocator, key_name, key_col, output_name, count_col, device_value);
}
pub fn initAggregatedDataFrame(
    comptime DeviceDataFrame: type,
    allocator: std.mem.Allocator,
    key_name: []const u8,
    key_col: DeviceColumn,
    output_name: []const u8,
    value_col: DeviceColumn,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{ LengthMismatch, InvalidDevice })!DeviceDataFrame {
    var owned_key = key_col;
    errdefer owned_key.deinit();
    const rows = owned_key.len();
    var owned_value = value_col;
    errdefer owned_value.deinit();
    if (owned_value.len() != rows) return error.LengthMismatch;
    const names = [_][]const u8{ key_name, output_name };
    const columns = try allocator.alloc(DeviceColumn, 2);
    errdefer allocator.free(columns);
    columns[0] = owned_key;
    columns[1] = owned_value;
    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, allocator, &names, columns, rows, device_value);
}

pub fn groupByNumericDispatchKey(
    comptime DeviceDataFrame: type,
    op: DeviceGroupByAggregation,
    allocator: std.mem.Allocator,
    key_name: []const u8,
    output_name: []const u8,
    key: DeviceColumn,
    value: DeviceColumn,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, TypeUnsupported, InvalidDevice })!DeviceDataFrame {
    return switch (key) {
        .bool => |typed| groupByNumericDispatchValue(DeviceDataFrame, op, bool, allocator, key_name, output_name, typed, value, device_value),
        .i8 => |typed| groupByNumericDispatchValue(DeviceDataFrame, op, i8, allocator, key_name, output_name, typed, value, device_value),
        .i16 => |typed| groupByNumericDispatchValue(DeviceDataFrame, op, i16, allocator, key_name, output_name, typed, value, device_value),
        .i32 => |typed| groupByNumericDispatchValue(DeviceDataFrame, op, i32, allocator, key_name, output_name, typed, value, device_value),
        .i64 => |typed| groupByNumericDispatchValue(DeviceDataFrame, op, i64, allocator, key_name, output_name, typed, value, device_value),
        .u8 => |typed| groupByNumericDispatchValue(DeviceDataFrame, op, u8, allocator, key_name, output_name, typed, value, device_value),
        .u16 => |typed| groupByNumericDispatchValue(DeviceDataFrame, op, u16, allocator, key_name, output_name, typed, value, device_value),
        .u32 => |typed| groupByNumericDispatchValue(DeviceDataFrame, op, u32, allocator, key_name, output_name, typed, value, device_value),
        .u64 => |typed| groupByNumericDispatchValue(DeviceDataFrame, op, u64, allocator, key_name, output_name, typed, value, device_value),
        .usize => |typed| groupByNumericDispatchValue(DeviceDataFrame, op, usize, allocator, key_name, output_name, typed, value, device_value),
        .isize => |typed| groupByNumericDispatchValue(DeviceDataFrame, op, isize, allocator, key_name, output_name, typed, value, device_value),
        .f16 => |typed| groupByNumericDispatchValue(DeviceDataFrame, op, f16, allocator, key_name, output_name, typed, value, device_value),
        .f32 => |typed| groupByNumericDispatchValue(DeviceDataFrame, op, f32, allocator, key_name, output_name, typed, value, device_value),
        .f64 => |typed| groupByNumericDispatchValue(DeviceDataFrame, op, f64, allocator, key_name, output_name, typed, value, device_value),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn groupByNumericDispatchValue(
    comptime DeviceDataFrame: type,
    op: DeviceGroupByAggregation,
    comptime K: type,
    allocator: std.mem.Allocator,
    key_name: []const u8,
    output_name: []const u8,
    key: DeviceTypedColumn(K),
    value: DeviceColumn,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, TypeUnsupported, InvalidDevice })!DeviceDataFrame {
    return switch (value) {
        .i8 => |typed| groupByNumericTyped(DeviceDataFrame, op, K, i8, allocator, key_name, output_name, key, typed, device_value),
        .i16 => |typed| groupByNumericTyped(DeviceDataFrame, op, K, i16, allocator, key_name, output_name, key, typed, device_value),
        .i32 => |typed| groupByNumericTyped(DeviceDataFrame, op, K, i32, allocator, key_name, output_name, key, typed, device_value),
        .i64 => |typed| groupByNumericTyped(DeviceDataFrame, op, K, i64, allocator, key_name, output_name, key, typed, device_value),
        .u8 => |typed| groupByNumericTyped(DeviceDataFrame, op, K, u8, allocator, key_name, output_name, key, typed, device_value),
        .u16 => |typed| groupByNumericTyped(DeviceDataFrame, op, K, u16, allocator, key_name, output_name, key, typed, device_value),
        .u32 => |typed| groupByNumericTyped(DeviceDataFrame, op, K, u32, allocator, key_name, output_name, key, typed, device_value),
        .u64 => |typed| groupByNumericTyped(DeviceDataFrame, op, K, u64, allocator, key_name, output_name, key, typed, device_value),
        .usize => |typed| groupByNumericTyped(DeviceDataFrame, op, K, usize, allocator, key_name, output_name, key, typed, device_value),
        .isize => |typed| groupByNumericTyped(DeviceDataFrame, op, K, isize, allocator, key_name, output_name, key, typed, device_value),
        .f16 => |typed| groupByNumericTyped(DeviceDataFrame, op, K, f16, allocator, key_name, output_name, key, typed, device_value),
        .f32 => |typed| groupByNumericTyped(DeviceDataFrame, op, K, f32, allocator, key_name, output_name, key, typed, device_value),
        .f64 => |typed| groupByNumericTyped(DeviceDataFrame, op, K, f64, allocator, key_name, output_name, key, typed, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn groupByNumericTyped(
    comptime DeviceDataFrame: type,
    op: DeviceGroupByAggregation,
    comptime K: type,
    comptime V: type,
    allocator: std.mem.Allocator,
    key_name: []const u8,
    output_name: []const u8,
    key: DeviceTypedColumn(K),
    value: DeviceTypedColumn(V),
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, TypeUnsupported, InvalidDevice })!DeviceDataFrame {
    if (key.len() != value.len()) return error.LengthMismatch;
    if (!key.device().sameDevice(value.device())) return error.InvalidDevice;

    const keys = try key.values.toOwnedSlice(allocator);
    defer allocator.free(keys);
    const values = try value.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_key_validity = try validityValues(key, allocator);
    defer if (maybe_key_validity) |validity| allocator.free(validity);
    const maybe_value_validity = try validityValues(value, allocator);
    defer if (maybe_value_validity) |validity| allocator.free(validity);

    var unique_keys: std.ArrayList(K) = .empty;
    defer unique_keys.deinit(allocator);
    var aggregates: std.ArrayList(V) = .empty;
    defer aggregates.deinit(allocator);

    for (keys, values, 0..) |key_value, value_item, row| {
        if (maybe_key_validity) |validity| {
            if (!validity[row]) continue;
        }
        if (maybe_value_validity) |validity| {
            if (!validity[row]) continue;
        }
        const maybe_group_index = findGroupIndex(K, unique_keys.items, key_value);
        if (maybe_group_index == null) {
            try unique_keys.append(allocator, key_value);
            try aggregates.append(allocator, value_item);
            continue;
        }
        const group_index = maybe_group_index.?;
        switch (op) {
            .sum => aggregates.items[group_index] += value_item,
            .min => {
                if (compareSortValues(V, value_item, aggregates.items[group_index]) < 0) aggregates.items[group_index] = value_item;
            },
            .max => {
                if (compareSortValues(V, value_item, aggregates.items[group_index]) > 0) aggregates.items[group_index] = value_item;
            },
        }
    }

    const key_col = try DeviceColumn.fromSlice(K, allocator, unique_keys.items, device_value);
    const aggregate_col = try DeviceColumn.fromSlice(V, allocator, aggregates.items, device_value);
    return initAggregatedDataFrame(DeviceDataFrame, allocator, key_name, key_col, output_name, aggregate_col, device_value);
}
pub fn groupByMeanDispatchKey(
    comptime DeviceDataFrame: type,
    allocator: std.mem.Allocator,
    key_name: []const u8,
    output_name: []const u8,
    key: DeviceColumn,
    value: DeviceColumn,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, TypeUnsupported, InvalidDevice })!DeviceDataFrame {
    return switch (key) {
        .bool => |typed| groupByMeanDispatchValue(DeviceDataFrame, bool, allocator, key_name, output_name, typed, value, device_value),
        .i8 => |typed| groupByMeanDispatchValue(DeviceDataFrame, i8, allocator, key_name, output_name, typed, value, device_value),
        .i16 => |typed| groupByMeanDispatchValue(DeviceDataFrame, i16, allocator, key_name, output_name, typed, value, device_value),
        .i32 => |typed| groupByMeanDispatchValue(DeviceDataFrame, i32, allocator, key_name, output_name, typed, value, device_value),
        .i64 => |typed| groupByMeanDispatchValue(DeviceDataFrame, i64, allocator, key_name, output_name, typed, value, device_value),
        .u8 => |typed| groupByMeanDispatchValue(DeviceDataFrame, u8, allocator, key_name, output_name, typed, value, device_value),
        .u16 => |typed| groupByMeanDispatchValue(DeviceDataFrame, u16, allocator, key_name, output_name, typed, value, device_value),
        .u32 => |typed| groupByMeanDispatchValue(DeviceDataFrame, u32, allocator, key_name, output_name, typed, value, device_value),
        .u64 => |typed| groupByMeanDispatchValue(DeviceDataFrame, u64, allocator, key_name, output_name, typed, value, device_value),
        .usize => |typed| groupByMeanDispatchValue(DeviceDataFrame, usize, allocator, key_name, output_name, typed, value, device_value),
        .isize => |typed| groupByMeanDispatchValue(DeviceDataFrame, isize, allocator, key_name, output_name, typed, value, device_value),
        .f16 => |typed| groupByMeanDispatchValue(DeviceDataFrame, f16, allocator, key_name, output_name, typed, value, device_value),
        .f32 => |typed| groupByMeanDispatchValue(DeviceDataFrame, f32, allocator, key_name, output_name, typed, value, device_value),
        .f64 => |typed| groupByMeanDispatchValue(DeviceDataFrame, f64, allocator, key_name, output_name, typed, value, device_value),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn groupByMeanDispatchValue(
    comptime DeviceDataFrame: type,
    comptime K: type,
    allocator: std.mem.Allocator,
    key_name: []const u8,
    output_name: []const u8,
    key: DeviceTypedColumn(K),
    value: DeviceColumn,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, TypeUnsupported, InvalidDevice })!DeviceDataFrame {
    return switch (value) {
        .i8 => |typed| groupByMeanTyped(DeviceDataFrame, K, i8, allocator, key_name, output_name, key, typed, device_value),
        .i16 => |typed| groupByMeanTyped(DeviceDataFrame, K, i16, allocator, key_name, output_name, key, typed, device_value),
        .i32 => |typed| groupByMeanTyped(DeviceDataFrame, K, i32, allocator, key_name, output_name, key, typed, device_value),
        .i64 => |typed| groupByMeanTyped(DeviceDataFrame, K, i64, allocator, key_name, output_name, key, typed, device_value),
        .u8 => |typed| groupByMeanTyped(DeviceDataFrame, K, u8, allocator, key_name, output_name, key, typed, device_value),
        .u16 => |typed| groupByMeanTyped(DeviceDataFrame, K, u16, allocator, key_name, output_name, key, typed, device_value),
        .u32 => |typed| groupByMeanTyped(DeviceDataFrame, K, u32, allocator, key_name, output_name, key, typed, device_value),
        .u64 => |typed| groupByMeanTyped(DeviceDataFrame, K, u64, allocator, key_name, output_name, key, typed, device_value),
        .usize => |typed| groupByMeanTyped(DeviceDataFrame, K, usize, allocator, key_name, output_name, key, typed, device_value),
        .isize => |typed| groupByMeanTyped(DeviceDataFrame, K, isize, allocator, key_name, output_name, key, typed, device_value),
        .f16 => |typed| groupByMeanTyped(DeviceDataFrame, K, f16, allocator, key_name, output_name, key, typed, device_value),
        .f32 => |typed| groupByMeanTyped(DeviceDataFrame, K, f32, allocator, key_name, output_name, key, typed, device_value),
        .f64 => |typed| groupByMeanTyped(DeviceDataFrame, K, f64, allocator, key_name, output_name, key, typed, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn groupByMeanTyped(
    comptime DeviceDataFrame: type,
    comptime K: type,
    comptime V: type,
    allocator: std.mem.Allocator,
    key_name: []const u8,
    output_name: []const u8,
    key: DeviceTypedColumn(K),
    value: DeviceTypedColumn(V),
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch, TypeUnsupported, InvalidDevice })!DeviceDataFrame {
    if (key.len() != value.len()) return error.LengthMismatch;
    if (!key.device().sameDevice(value.device())) return error.InvalidDevice;

    const keys = try key.values.toOwnedSlice(allocator);
    defer allocator.free(keys);
    const values = try value.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_key_validity = try validityValues(key, allocator);
    defer if (maybe_key_validity) |validity| allocator.free(validity);
    const maybe_value_validity = try validityValues(value, allocator);
    defer if (maybe_value_validity) |validity| allocator.free(validity);

    var unique_keys: std.ArrayList(K) = .empty;
    defer unique_keys.deinit(allocator);
    var sums: std.ArrayList(f64) = .empty;
    defer sums.deinit(allocator);
    var counts: std.ArrayList(usize) = .empty;
    defer counts.deinit(allocator);

    for (keys, values, 0..) |key_value, value_item, row| {
        if (maybe_key_validity) |validity| {
            if (!validity[row]) continue;
        }
        if (maybe_value_validity) |validity| {
            if (!validity[row]) continue;
        }
        const group_index = findGroupIndex(K, unique_keys.items, key_value) orelse blk: {
            try unique_keys.append(allocator, key_value);
            try sums.append(allocator, 0);
            try counts.append(allocator, 0);
            break :blk unique_keys.items.len - 1;
        };
        sums.items[group_index] += castToF64(V, value_item);
        counts.items[group_index] += 1;
    }

    const means = try allocator.alloc(f64, sums.items.len);
    defer allocator.free(means);
    for (sums.items, counts.items, means) |sum_value, count, *slot| {
        slot.* = sum_value / @as(f64, @floatFromInt(count));
    }

    const key_col = try DeviceColumn.fromSlice(K, allocator, unique_keys.items, device_value);
    const mean_col = try DeviceColumn.fromSlice(f64, allocator, means, device_value);
    return initAggregatedDataFrame(DeviceDataFrame, allocator, key_name, key_col, output_name, mean_col, device_value);
}
