//! Multi-key group-by materializers for device dataframes. Keeping these
//! dispatch helpers here keeps the public dataframe facade smaller while the
//! generic `DeviceDataFrame` parameter preserves the original API shape.

const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const dataframe_device_column_mod = @import("dataframe_device_column.zig");
const group_profile_mod = @import("dataframe_group_profile.zig");
const keys_mod = @import("dataframe_keys.zig");
const names_mod = @import("dataframe_names.zig");
const numeric_mod = @import("dataframe_numeric.zig");
const options_mod = @import("dataframe_options.zig");
const validity_mod = @import("dataframe_validity.zig");

const DeviceColumn = dataframe_device_column_mod.DeviceColumn;
const DeviceTypedColumn = dataframe_device_column_mod.DeviceTypedColumn;
const DeviceGroupByAggregation = options_mod.DeviceGroupByAggregation;
const MomentProfile = group_profile_mod.MomentProfile;
const compareSortValues = numeric_mod.compareSortValues;
const castToF64 = numeric_mod.castToF64;
const groupKeyEqual = numeric_mod.groupKeyEqual;
const rowHasValidKeys = keys_mod.rowHasValidKeys;
const columnRowValid = keys_mod.columnRowValid;
const findMultiKeyGroupIndex = keys_mod.findMultiKeyGroupIndex;
const validityValues = validity_mod.validityValues;

const GroupByOnError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    TypeUnsupported,
    UnsupportedType,
    InvalidCsv,
    EmptyDataFrame,
    InvalidDevice,
    IndexOutOfBounds,
};

const GroupByMomentAggregation = enum {
    variance,
    magnitude_variance,
    stddev,
    magnitude_stddev,
    sem,
    magnitude_sem,
    cv,
    magnitude_cv,
    fano,
    magnitude_fano,
    skewness,
    magnitude_skewness,
    kurtosis,
    magnitude_kurtosis,
};

const GroupByRealAggregation = enum {
    mean_abs,
    mean_square,
    rms,
    l1_norm,
    l2_norm,
    max_abs,
    min_abs,
    geometric_mean,
    harmonic_mean,
    logsumexp,
    logmeanexp,
    ptp,
    midrange,
    range_coeff,
    hhi,
    magnitude_normalized_hhi,
    magnitude_sparsity,
    magnitude_inverse_simpson,
    magnitude_simpson_evenness,
    magnitude_dominance,
    magnitude_dominance_margin,
    magnitude_entropy,
    magnitude_perplexity,
    magnitude_evenness,
};

const GroupByRobustAggregation = enum {
    iqr,
    mad,
    trimmed_mean,
    winsorized_mean,
    interdecile_range,
    midhinge,
    trimean,
    bowley_skewness,
    quartile_coeff_dispersion,
    kelley_skewness,
};

const GroupByModeDiagnostic = enum {
    count,
    ratio,
    margin,
    margin_ratio,
};

const GroupByDistributionAggregation = enum {
    entropy,
    gini_impurity,
    perplexity,
    inverse_simpson,
    simpson_concentration,
    evenness,
};

const GroupByInequalityAggregation = enum {
    mean_abs_dev,
    mean_abs_dev_ratio,
    gini_mean_diff,
    gini_coefficient,
};

const GroupByBoolAggregation = enum {
    any,
    all,
    true_count,
    false_count,
    true_ratio,
    false_ratio,
};

const GroupByValidityAggregation = enum {
    valid_count,
    null_count,
    valid_ratio,
    null_ratio,
};

const GroupByArgAggregation = enum {
    argmin,
    argmax,
};

const GroupByWeightedAggregation = enum {
    weighted_mean,
    weighted_variance,
    weighted_stddev,
    weighted_quantile,
    weighted_median,
    weighted_iqr,
    weighted_mad,
};

const OwnedGroupRealColumn = struct {
    allocator: std.mem.Allocator,
    values: []f64,
    validity: ?[]bool,

    fn deinit(self: *OwnedGroupRealColumn) void {
        self.allocator.free(self.values);
        if (self.validity) |validity| self.allocator.free(validity);
        self.* = undefined;
    }
};

fn ownedGroupRealColumn(allocator: std.mem.Allocator, column: DeviceColumn) GroupByOnError!OwnedGroupRealColumn {
    return switch (column) {
        .i8 => |typed| ownedGroupRealColumnTyped(i8, allocator, typed),
        .i16 => |typed| ownedGroupRealColumnTyped(i16, allocator, typed),
        .i32 => |typed| ownedGroupRealColumnTyped(i32, allocator, typed),
        .i64 => |typed| ownedGroupRealColumnTyped(i64, allocator, typed),
        .u8 => |typed| ownedGroupRealColumnTyped(u8, allocator, typed),
        .u16 => |typed| ownedGroupRealColumnTyped(u16, allocator, typed),
        .u32 => |typed| ownedGroupRealColumnTyped(u32, allocator, typed),
        .u64 => |typed| ownedGroupRealColumnTyped(u64, allocator, typed),
        .usize => |typed| ownedGroupRealColumnTyped(usize, allocator, typed),
        .isize => |typed| ownedGroupRealColumnTyped(isize, allocator, typed),
        .f16 => |typed| ownedGroupRealColumnTyped(f16, allocator, typed),
        .f32 => |typed| ownedGroupRealColumnTyped(f32, allocator, typed),
        .f64 => |typed| ownedGroupRealColumnTyped(f64, allocator, typed),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn ownedGroupRealColumnTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
) GroupByOnError!OwnedGroupRealColumn {
    const raw_values = try column.values.toOwnedSlice(allocator);
    defer allocator.free(raw_values);
    const values = try allocator.alloc(f64, raw_values.len);
    errdefer allocator.free(values);
    for (raw_values, values) |raw, *slot| slot.* = castToF64(T, raw);
    const maybe_validity = try validityValues(column, allocator);
    errdefer if (maybe_validity) |validity| allocator.free(validity);
    return .{
        .allocator = allocator,
        .values = values,
        .validity = maybe_validity,
    };
}

const GroupWeightedValue = struct {
    value: f64,
    weight: f64,
};

fn groupWeightedValueLess(_: void, lhs: GroupWeightedValue, rhs: GroupWeightedValue) bool {
    return groupByQuantileLess({}, lhs.value, rhs.value);
}

fn groupWeightedQuantileFromSorted(sorted: []const GroupWeightedValue, q: f64, total_weight: f64) f64 {
    const threshold = q * total_weight;
    var cumulative: f64 = 0.0;
    for (sorted) |item| {
        cumulative += item.weight;
        if (cumulative >= threshold) return item.value;
    }
    return sorted[sorted.len - 1].value;
}

fn groupWeightedQuantileFromRows(
    allocator: std.mem.Allocator,
    rows: []const usize,
    values: []const f64,
    weights: []const f64,
    q: f64,
    subtract_q: ?f64,
) std.mem.Allocator.Error!f64 {
    const scratch = try allocator.alloc(GroupWeightedValue, rows.len);
    defer allocator.free(scratch);

    var total_weight: f64 = 0.0;
    for (rows, 0..) |row, index| {
        const weight = weights[row];
        scratch[index] = .{ .value = values[row], .weight = weight };
        total_weight += weight;
    }
    if (rows.len == 0 or !(total_weight > 0.0)) return std.math.nan(f64);

    std.sort.insertion(GroupWeightedValue, scratch, {}, groupWeightedValueLess);
    const hi = groupWeightedQuantileFromSorted(scratch, q, total_weight);
    return if (subtract_q) |lo_q| hi - groupWeightedQuantileFromSorted(scratch, lo_q, total_weight) else hi;
}

fn groupWeightedMadFromRows(
    allocator: std.mem.Allocator,
    rows: []const usize,
    values: []const f64,
    weights: []const f64,
) std.mem.Allocator.Error!f64 {
    const scratch = try allocator.alloc(GroupWeightedValue, rows.len);
    defer allocator.free(scratch);

    var total_weight: f64 = 0.0;
    for (rows, 0..) |row, index| {
        const weight = weights[row];
        scratch[index] = .{ .value = values[row], .weight = weight };
        total_weight += weight;
    }
    if (rows.len == 0 or !(total_weight > 0.0)) return std.math.nan(f64);

    std.sort.insertion(GroupWeightedValue, scratch, {}, groupWeightedValueLess);
    const center = groupWeightedQuantileFromSorted(scratch, 0.5, total_weight);
    for (scratch) |*item| item.value = @abs(item.value - center);
    std.sort.insertion(GroupWeightedValue, scratch, {}, groupWeightedValueLess);
    return groupWeightedQuantileFromSorted(scratch, 0.5, total_weight);
}

pub fn groupByStatsOnDispatchValue(
    comptime DeviceDataFrame: type,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_prefix: []const u8,
    value: DeviceColumn,
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    return switch (value) {
        .i8 => |typed| groupByStatsOnTyped(DeviceDataFrame, i8, allocator, frame, key_names, output_prefix, typed, device_value),
        .i16 => |typed| groupByStatsOnTyped(DeviceDataFrame, i16, allocator, frame, key_names, output_prefix, typed, device_value),
        .i32 => |typed| groupByStatsOnTyped(DeviceDataFrame, i32, allocator, frame, key_names, output_prefix, typed, device_value),
        .i64 => |typed| groupByStatsOnTyped(DeviceDataFrame, i64, allocator, frame, key_names, output_prefix, typed, device_value),
        .u8 => |typed| groupByStatsOnTyped(DeviceDataFrame, u8, allocator, frame, key_names, output_prefix, typed, device_value),
        .u16 => |typed| groupByStatsOnTyped(DeviceDataFrame, u16, allocator, frame, key_names, output_prefix, typed, device_value),
        .u32 => |typed| groupByStatsOnTyped(DeviceDataFrame, u32, allocator, frame, key_names, output_prefix, typed, device_value),
        .u64 => |typed| groupByStatsOnTyped(DeviceDataFrame, u64, allocator, frame, key_names, output_prefix, typed, device_value),
        .usize => |typed| groupByStatsOnTyped(DeviceDataFrame, usize, allocator, frame, key_names, output_prefix, typed, device_value),
        .isize => |typed| groupByStatsOnTyped(DeviceDataFrame, isize, allocator, frame, key_names, output_prefix, typed, device_value),
        .f16 => |typed| groupByStatsOnTyped(DeviceDataFrame, f16, allocator, frame, key_names, output_prefix, typed, device_value),
        .f32 => |typed| groupByStatsOnTyped(DeviceDataFrame, f32, allocator, frame, key_names, output_prefix, typed, device_value),
        .f64 => |typed| groupByStatsOnTyped(DeviceDataFrame, f64, allocator, frame, key_names, output_prefix, typed, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn groupByStatsOnTyped(
    comptime DeviceDataFrame: type,
    comptime V: type,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_prefix: []const u8,
    value: DeviceTypedColumn(V),
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    if (frame.rows != value.len()) return error.LengthMismatch;
    const values = try value.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_value_validity = try validityValues(value, allocator);
    defer if (maybe_value_validity) |validity| allocator.free(validity);

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(allocator);
    var counts: std.ArrayList(i64) = .empty;
    defer counts.deinit(allocator);
    var sums: std.ArrayList(V) = .empty;
    defer sums.deinit(allocator);
    var mins: std.ArrayList(V) = .empty;
    defer mins.deinit(allocator);
    var maxes: std.ArrayList(V) = .empty;
    defer maxes.deinit(allocator);
    var mean_sums: std.ArrayList(f64) = .empty;
    defer mean_sums.deinit(allocator);

    for (values, 0..) |value_item, row| {
        if (maybe_value_validity) |validity| {
            if (!validity[row]) continue;
        }
        if (!try rowHasValidKeys(allocator, frame, key_names, row)) continue;
        const maybe_group_index = try findMultiKeyGroupIndex(allocator, frame, key_names, representative_rows.items, row);
        if (maybe_group_index == null) {
            try representative_rows.append(allocator, row);
            try counts.append(allocator, 1);
            try sums.append(allocator, value_item);
            try mins.append(allocator, value_item);
            try maxes.append(allocator, value_item);
            try mean_sums.append(allocator, castToF64(V, value_item));
            continue;
        }
        const group_index = maybe_group_index.?;
        counts.items[group_index] += 1;
        sums.items[group_index] += value_item;
        if (compareSortValues(V, value_item, mins.items[group_index]) < 0) mins.items[group_index] = value_item;
        if (compareSortValues(V, value_item, maxes.items[group_index]) > 0) maxes.items[group_index] = value_item;
        mean_sums.items[group_index] += castToF64(V, value_item);
    }

    const means = try allocator.alloc(f64, counts.items.len);
    defer allocator.free(means);
    for (mean_sums.items, counts.items, means) |sum_value, count, *slot| {
        slot.* = sum_value / @as(f64, @floatFromInt(count));
    }

    const output_names = try names_mod.statsOutputNames(allocator, "", output_prefix);
    defer names_mod.freeStatsOutputNames(allocator, output_names);
    const total_cols = key_names.len + 5;
    var names = try allocator.alloc([]const u8, total_cols);
    defer allocator.free(names);
    var columns = try allocator.alloc(DeviceColumn, total_cols);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        allocator.free(columns);
    }

    for (key_names) |key_name| {
        names[initialized] = key_name;
        columns[initialized] = try (try frame.column(key_name)).take(representative_rows.items);
        initialized += 1;
    }
    names[initialized] = output_names[1];
    columns[initialized] = try DeviceColumn.fromSlice(i64, allocator, counts.items, device_value);
    initialized += 1;
    names[initialized] = output_names[2];
    columns[initialized] = try DeviceColumn.fromSlice(V, allocator, sums.items, device_value);
    initialized += 1;
    names[initialized] = output_names[3];
    columns[initialized] = try DeviceColumn.fromSlice(V, allocator, mins.items, device_value);
    initialized += 1;
    names[initialized] = output_names[4];
    columns[initialized] = try DeviceColumn.fromSlice(V, allocator, maxes.items, device_value);
    initialized += 1;
    names[initialized] = output_names[5];
    columns[initialized] = try DeviceColumn.fromSlice(f64, allocator, means, device_value);
    initialized += 1;
    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, allocator, names, columns, representative_rows.items.len, device_value);
}

pub fn groupByCountOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(frame.allocator);
    var counts: std.ArrayList(i64) = .empty;
    defer counts.deinit(frame.allocator);

    for (0..frame.rows) |row| {
        if (!try rowHasValidKeys(frame.allocator, frame, key_names, row)) continue;
        const group_index = (try findMultiKeyGroupIndex(frame.allocator, frame, key_names, representative_rows.items, row)) orelse blk: {
            try representative_rows.append(frame.allocator, row);
            try counts.append(frame.allocator, 0);
            break :blk representative_rows.items.len - 1;
        };
        counts.items[group_index] += 1;
    }

    const count_column = try DeviceColumn.fromSlice(i64, frame.allocator, counts.items, frame.device);
    return initMultiKeyAggregatedDataFrame(DeviceDataFrame, frame, key_names, representative_rows.items, output_name, count_column);
}

pub fn groupByNumericOn(
    comptime DeviceDataFrame: type,
    op: DeviceGroupByAggregation,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);
    const value = try frame.column(value_name);
    return groupByNumericOnDispatchValue(DeviceDataFrame, op, frame.allocator, frame, key_names, output_name, value.*, frame.device);
}

pub fn groupByMeanOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);
    const value = try frame.column(value_name);
    return groupByMeanOnDispatchValue(DeviceDataFrame, frame.allocator, frame, key_names, output_name, value.*, frame.device);
}

fn groupByTakeOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
    comptime keep_last: bool,
) GroupByOnError!DeviceDataFrame {
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);
    const value = try frame.column(value_name);

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(frame.allocator);
    var value_rows: std.ArrayList(usize) = .empty;
    defer value_rows.deinit(frame.allocator);

    for (0..frame.rows) |row| {
        if (!try rowHasValidKeys(frame.allocator, frame, key_names, row)) continue;
        if (!try columnRowValid(frame.allocator, value.*, row)) continue;
        const maybe_group_index = try findMultiKeyGroupIndex(frame.allocator, frame, key_names, representative_rows.items, row);
        if (maybe_group_index) |group_index| {
            if (keep_last) value_rows.items[group_index] = row;
        } else {
            try representative_rows.append(frame.allocator, row);
            try value_rows.append(frame.allocator, row);
        }
    }

    const value_column = try value.take(value_rows.items);
    return initMultiKeyAggregatedDataFrame(DeviceDataFrame, frame, key_names, representative_rows.items, output_name, value_column);
}

pub fn groupByFirstOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByTakeOn(DeviceDataFrame, frame, key_names, value_name, output_name, false);
}

pub fn groupByLastOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByTakeOn(DeviceDataFrame, frame, key_names, value_name, output_name, true);
}

pub fn groupByNUniqueOnDispatchValue(
    comptime DeviceDataFrame: type,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
    value: DeviceColumn,
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    return switch (value) {
        .bool => |typed| groupByNUniqueOnTyped(DeviceDataFrame, bool, allocator, frame, key_names, output_name, typed, device_value),
        .i8 => |typed| groupByNUniqueOnTyped(DeviceDataFrame, i8, allocator, frame, key_names, output_name, typed, device_value),
        .i16 => |typed| groupByNUniqueOnTyped(DeviceDataFrame, i16, allocator, frame, key_names, output_name, typed, device_value),
        .i32 => |typed| groupByNUniqueOnTyped(DeviceDataFrame, i32, allocator, frame, key_names, output_name, typed, device_value),
        .i64 => |typed| groupByNUniqueOnTyped(DeviceDataFrame, i64, allocator, frame, key_names, output_name, typed, device_value),
        .u8 => |typed| groupByNUniqueOnTyped(DeviceDataFrame, u8, allocator, frame, key_names, output_name, typed, device_value),
        .u16 => |typed| groupByNUniqueOnTyped(DeviceDataFrame, u16, allocator, frame, key_names, output_name, typed, device_value),
        .u32 => |typed| groupByNUniqueOnTyped(DeviceDataFrame, u32, allocator, frame, key_names, output_name, typed, device_value),
        .u64 => |typed| groupByNUniqueOnTyped(DeviceDataFrame, u64, allocator, frame, key_names, output_name, typed, device_value),
        .usize => |typed| groupByNUniqueOnTyped(DeviceDataFrame, usize, allocator, frame, key_names, output_name, typed, device_value),
        .isize => |typed| groupByNUniqueOnTyped(DeviceDataFrame, isize, allocator, frame, key_names, output_name, typed, device_value),
        .f16 => |typed| groupByNUniqueOnTyped(DeviceDataFrame, f16, allocator, frame, key_names, output_name, typed, device_value),
        .f32 => |typed| groupByNUniqueOnTyped(DeviceDataFrame, f32, allocator, frame, key_names, output_name, typed, device_value),
        .f64 => |typed| groupByNUniqueOnTyped(DeviceDataFrame, f64, allocator, frame, key_names, output_name, typed, device_value),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn groupByNUniqueOnTyped(
    comptime DeviceDataFrame: type,
    comptime V: type,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
    value: DeviceTypedColumn(V),
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    if (frame.rows != value.len()) return error.LengthMismatch;
    const values = try value.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_value_validity = try validityValues(value, allocator);
    defer if (maybe_value_validity) |validity| allocator.free(validity);

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(allocator);
    var distinct_value_rows: std.ArrayList(std.ArrayList(usize)) = .empty;
    defer {
        for (distinct_value_rows.items) |*rows| rows.deinit(allocator);
        distinct_value_rows.deinit(allocator);
    }

    for (values, 0..) |value_item, row| {
        if (maybe_value_validity) |validity| {
            if (!validity[row]) continue;
        }
        if (!try rowHasValidKeys(allocator, frame, key_names, row)) continue;
        const group_index = (try findMultiKeyGroupIndex(allocator, frame, key_names, representative_rows.items, row)) orelse blk: {
            try representative_rows.append(allocator, row);
            try distinct_value_rows.append(allocator, .empty);
            break :blk representative_rows.items.len - 1;
        };

        var seen = false;
        for (distinct_value_rows.items[group_index].items) |previous_row| {
            if (groupKeyEqual(V, values[previous_row], value_item)) {
                seen = true;
                break;
            }
        }
        if (!seen) try distinct_value_rows.items[group_index].append(allocator, row);
    }

    const counts = try allocator.alloc(i64, distinct_value_rows.items.len);
    defer allocator.free(counts);
    for (distinct_value_rows.items, counts) |rows, *slot| {
        slot.* = @intCast(rows.items.len);
    }

    const count_column = try DeviceColumn.fromSlice(i64, allocator, counts, device_value);
    return initMultiKeyAggregatedDataFrame(DeviceDataFrame, frame, key_names, representative_rows.items, output_name, count_column);
}

pub fn groupByNUniqueOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);
    const value = try frame.column(value_name);
    return groupByNUniqueOnDispatchValue(DeviceDataFrame, frame.allocator, frame, key_names, output_name, value.*, frame.device);
}

pub fn groupByModeOnDispatchValue(
    comptime DeviceDataFrame: type,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
    value: DeviceColumn,
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    return switch (value) {
        .bool => |typed| groupByModeOnTyped(DeviceDataFrame, bool, allocator, frame, key_names, output_name, typed, device_value),
        .i8 => |typed| groupByModeOnTyped(DeviceDataFrame, i8, allocator, frame, key_names, output_name, typed, device_value),
        .i16 => |typed| groupByModeOnTyped(DeviceDataFrame, i16, allocator, frame, key_names, output_name, typed, device_value),
        .i32 => |typed| groupByModeOnTyped(DeviceDataFrame, i32, allocator, frame, key_names, output_name, typed, device_value),
        .i64 => |typed| groupByModeOnTyped(DeviceDataFrame, i64, allocator, frame, key_names, output_name, typed, device_value),
        .u8 => |typed| groupByModeOnTyped(DeviceDataFrame, u8, allocator, frame, key_names, output_name, typed, device_value),
        .u16 => |typed| groupByModeOnTyped(DeviceDataFrame, u16, allocator, frame, key_names, output_name, typed, device_value),
        .u32 => |typed| groupByModeOnTyped(DeviceDataFrame, u32, allocator, frame, key_names, output_name, typed, device_value),
        .u64 => |typed| groupByModeOnTyped(DeviceDataFrame, u64, allocator, frame, key_names, output_name, typed, device_value),
        .usize => |typed| groupByModeOnTyped(DeviceDataFrame, usize, allocator, frame, key_names, output_name, typed, device_value),
        .isize => |typed| groupByModeOnTyped(DeviceDataFrame, isize, allocator, frame, key_names, output_name, typed, device_value),
        .f16 => |typed| groupByModeOnTyped(DeviceDataFrame, f16, allocator, frame, key_names, output_name, typed, device_value),
        .f32 => |typed| groupByModeOnTyped(DeviceDataFrame, f32, allocator, frame, key_names, output_name, typed, device_value),
        .f64 => |typed| groupByModeOnTyped(DeviceDataFrame, f64, allocator, frame, key_names, output_name, typed, device_value),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

const GroupModeStats = struct {
    row: usize,
    count: usize,
    second_count: usize,
    total_count: usize,
};

fn groupModeStats(comptime V: type, values: []const V, rows: []const usize) GroupModeStats {
    var best_row: usize = rows[0];
    var best_count: usize = 0;
    var second_count: usize = 0;
    for (rows, 0..) |candidate_row, candidate_index| {
        var seen = false;
        for (rows[0..candidate_index]) |previous_row| {
            if (groupKeyEqual(V, values[previous_row], values[candidate_row])) {
                seen = true;
                break;
            }
        }
        if (seen) continue;

        var count: usize = 0;
        for (rows[candidate_index..]) |match_row| {
            if (groupKeyEqual(V, values[candidate_row], values[match_row])) count += 1;
        }
        // Keep the first distinct value as the mode on exact ties, matching
        // the public `groupByMode` contract while still tracking the tied
        // runner-up frequency so margin diagnostics expose ambiguity.
        if (count > best_count) {
            second_count = best_count;
            best_row = candidate_row;
            best_count = count;
        } else if (count > second_count) {
            second_count = count;
        }
    }
    return .{
        .row = best_row,
        .count = best_count,
        .second_count = second_count,
        .total_count = rows.len,
    };
}

fn groupByModeOnTyped(
    comptime DeviceDataFrame: type,
    comptime V: type,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
    value: DeviceTypedColumn(V),
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    if (frame.rows != value.len()) return error.LengthMismatch;
    const values = try value.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_value_validity = try validityValues(value, allocator);
    defer if (maybe_value_validity) |validity| allocator.free(validity);

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(allocator);
    var group_value_rows: std.ArrayList(std.ArrayList(usize)) = .empty;
    defer {
        for (group_value_rows.items) |*rows| rows.deinit(allocator);
        group_value_rows.deinit(allocator);
    }

    for (values, 0..) |_, row| {
        if (maybe_value_validity) |validity| {
            if (!validity[row]) continue;
        }
        if (!try rowHasValidKeys(allocator, frame, key_names, row)) continue;
        const group_index = (try findMultiKeyGroupIndex(allocator, frame, key_names, representative_rows.items, row)) orelse blk: {
            try representative_rows.append(allocator, row);
            try group_value_rows.append(allocator, .empty);
            break :blk representative_rows.items.len - 1;
        };
        try group_value_rows.items[group_index].append(allocator, row);
    }

    const mode_rows = try allocator.alloc(usize, group_value_rows.items.len);
    defer allocator.free(mode_rows);
    for (group_value_rows.items, mode_rows) |rows, *slot| {
        slot.* = groupModeStats(V, values, rows.items).row;
    }

    const mode_values = try allocator.alloc(V, mode_rows.len);
    defer allocator.free(mode_values);
    for (mode_rows, mode_values) |row, *slot| slot.* = values[row];

    const mode_column = try DeviceColumn.fromSlice(V, allocator, mode_values, device_value);
    return initMultiKeyAggregatedDataFrame(DeviceDataFrame, frame, key_names, representative_rows.items, output_name, mode_column);
}

pub fn groupByModeDiagnosticOnDispatchValue(
    comptime DeviceDataFrame: type,
    aggregation: GroupByModeDiagnostic,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
    value: DeviceColumn,
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    return switch (value) {
        .bool => |typed| groupByModeDiagnosticOnTyped(DeviceDataFrame, bool, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .i8 => |typed| groupByModeDiagnosticOnTyped(DeviceDataFrame, i8, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .i16 => |typed| groupByModeDiagnosticOnTyped(DeviceDataFrame, i16, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .i32 => |typed| groupByModeDiagnosticOnTyped(DeviceDataFrame, i32, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .i64 => |typed| groupByModeDiagnosticOnTyped(DeviceDataFrame, i64, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u8 => |typed| groupByModeDiagnosticOnTyped(DeviceDataFrame, u8, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u16 => |typed| groupByModeDiagnosticOnTyped(DeviceDataFrame, u16, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u32 => |typed| groupByModeDiagnosticOnTyped(DeviceDataFrame, u32, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u64 => |typed| groupByModeDiagnosticOnTyped(DeviceDataFrame, u64, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .usize => |typed| groupByModeDiagnosticOnTyped(DeviceDataFrame, usize, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .isize => |typed| groupByModeDiagnosticOnTyped(DeviceDataFrame, isize, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .f16 => |typed| groupByModeDiagnosticOnTyped(DeviceDataFrame, f16, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .f32 => |typed| groupByModeDiagnosticOnTyped(DeviceDataFrame, f32, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .f64 => |typed| groupByModeDiagnosticOnTyped(DeviceDataFrame, f64, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn groupByModeDiagnosticOnTyped(
    comptime DeviceDataFrame: type,
    comptime V: type,
    aggregation: GroupByModeDiagnostic,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
    value: DeviceTypedColumn(V),
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    if (frame.rows != value.len()) return error.LengthMismatch;
    const values = try value.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_value_validity = try validityValues(value, allocator);
    defer if (maybe_value_validity) |validity| allocator.free(validity);

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(allocator);
    var group_value_rows: std.ArrayList(std.ArrayList(usize)) = .empty;
    defer {
        for (group_value_rows.items) |*rows| rows.deinit(allocator);
        group_value_rows.deinit(allocator);
    }

    for (values, 0..) |_, row| {
        if (maybe_value_validity) |validity| {
            if (!validity[row]) continue;
        }
        if (!try rowHasValidKeys(allocator, frame, key_names, row)) continue;
        const group_index = (try findMultiKeyGroupIndex(allocator, frame, key_names, representative_rows.items, row)) orelse blk: {
            try representative_rows.append(allocator, row);
            try group_value_rows.append(allocator, .empty);
            break :blk representative_rows.items.len - 1;
        };
        try group_value_rows.items[group_index].append(allocator, row);
    }

    const output_column: DeviceColumn = switch (aggregation) {
        .count, .margin => blk: {
            const out = try allocator.alloc(i64, group_value_rows.items.len);
            defer allocator.free(out);
            for (group_value_rows.items, out) |rows, *slot| {
                const stats = groupModeStats(V, values, rows.items);
                slot.* = @intCast(if (aggregation == .count) stats.count else stats.count - stats.second_count);
            }
            break :blk try DeviceColumn.fromSlice(i64, allocator, out, device_value);
        },
        .ratio, .margin_ratio => blk: {
            const out = try allocator.alloc(f64, group_value_rows.items.len);
            defer allocator.free(out);
            for (group_value_rows.items, out) |rows, *slot| {
                const stats = groupModeStats(V, values, rows.items);
                const numerator = if (aggregation == .ratio) stats.count else stats.count - stats.second_count;
                slot.* = @as(f64, @floatFromInt(numerator)) / @as(f64, @floatFromInt(stats.total_count));
            }
            break :blk try DeviceColumn.fromSlice(f64, allocator, out, device_value);
        },
    };
    return initMultiKeyAggregatedDataFrame(DeviceDataFrame, frame, key_names, representative_rows.items, output_name, output_column);
}

fn groupByModeDiagnosticOn(
    comptime DeviceDataFrame: type,
    aggregation: GroupByModeDiagnostic,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);
    const value = try frame.column(value_name);
    return groupByModeDiagnosticOnDispatchValue(DeviceDataFrame, aggregation, frame.allocator, frame, key_names, output_name, value.*, frame.device);
}

pub fn groupByModeOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);
    const value = try frame.column(value_name);
    return groupByModeOnDispatchValue(DeviceDataFrame, frame.allocator, frame, key_names, output_name, value.*, frame.device);
}

pub fn groupByModeCountOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByModeDiagnosticOn(DeviceDataFrame, .count, frame, key_names, value_name, output_name);
}

pub fn groupByModeRatioOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByModeDiagnosticOn(DeviceDataFrame, .ratio, frame, key_names, value_name, output_name);
}

pub fn groupByModeMarginOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByModeDiagnosticOn(DeviceDataFrame, .margin, frame, key_names, value_name, output_name);
}

pub fn groupByModeMarginRatioOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByModeDiagnosticOn(DeviceDataFrame, .margin_ratio, frame, key_names, value_name, output_name);
}

pub fn groupByDistributionOnDispatchValue(
    comptime DeviceDataFrame: type,
    aggregation: GroupByDistributionAggregation,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
    value: DeviceColumn,
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    return switch (value) {
        .bool => |typed| groupByDistributionOnTyped(DeviceDataFrame, bool, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .i8 => |typed| groupByDistributionOnTyped(DeviceDataFrame, i8, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .i16 => |typed| groupByDistributionOnTyped(DeviceDataFrame, i16, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .i32 => |typed| groupByDistributionOnTyped(DeviceDataFrame, i32, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .i64 => |typed| groupByDistributionOnTyped(DeviceDataFrame, i64, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u8 => |typed| groupByDistributionOnTyped(DeviceDataFrame, u8, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u16 => |typed| groupByDistributionOnTyped(DeviceDataFrame, u16, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u32 => |typed| groupByDistributionOnTyped(DeviceDataFrame, u32, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u64 => |typed| groupByDistributionOnTyped(DeviceDataFrame, u64, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .usize => |typed| groupByDistributionOnTyped(DeviceDataFrame, usize, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .isize => |typed| groupByDistributionOnTyped(DeviceDataFrame, isize, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .f16 => |typed| groupByDistributionOnTyped(DeviceDataFrame, f16, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .f32 => |typed| groupByDistributionOnTyped(DeviceDataFrame, f32, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .f64 => |typed| groupByDistributionOnTyped(DeviceDataFrame, f64, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn distributionMetric(
    comptime V: type,
    aggregation: GroupByDistributionAggregation,
    values: []const V,
    rows: []const usize,
) f64 {
    var entropy: f64 = 0.0;
    var sum_probability_sq: f64 = 0.0;
    var distinct_count: usize = 0;
    const total = @as(f64, @floatFromInt(rows.len));
    for (rows, 0..) |candidate_row, candidate_index| {
        var seen = false;
        for (rows[0..candidate_index]) |previous_row| {
            if (groupKeyEqual(V, values[previous_row], values[candidate_row])) {
                seen = true;
                break;
            }
        }
        if (seen) continue;

        var count: usize = 0;
        for (rows[candidate_index..]) |match_row| {
            if (groupKeyEqual(V, values[candidate_row], values[match_row])) count += 1;
        }
        distinct_count += 1;
        const probability = @as(f64, @floatFromInt(count)) / total;
        sum_probability_sq += probability * probability;
        entropy -= probability * std.math.log(f64, std.math.e, probability);
    }

    return switch (aggregation) {
        .entropy => entropy,
        .gini_impurity => 1.0 - sum_probability_sq,
        .perplexity => std.math.exp(entropy),
        .inverse_simpson => 1.0 / sum_probability_sq,
        .simpson_concentration => sum_probability_sq,
        .evenness => if (distinct_count <= 1) 1.0 else entropy / std.math.log(f64, std.math.e, @as(f64, @floatFromInt(distinct_count))),
    };
}

fn groupByDistributionOnTyped(
    comptime DeviceDataFrame: type,
    comptime V: type,
    aggregation: GroupByDistributionAggregation,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
    value: DeviceTypedColumn(V),
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    if (frame.rows != value.len()) return error.LengthMismatch;
    const values = try value.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_value_validity = try validityValues(value, allocator);
    defer if (maybe_value_validity) |validity| allocator.free(validity);

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(allocator);
    var group_value_rows: std.ArrayList(std.ArrayList(usize)) = .empty;
    defer {
        for (group_value_rows.items) |*rows| rows.deinit(allocator);
        group_value_rows.deinit(allocator);
    }

    for (values, 0..) |_, row| {
        if (maybe_value_validity) |validity| {
            if (!validity[row]) continue;
        }
        if (!try rowHasValidKeys(allocator, frame, key_names, row)) continue;
        const group_index = (try findMultiKeyGroupIndex(allocator, frame, key_names, representative_rows.items, row)) orelse blk: {
            try representative_rows.append(allocator, row);
            try group_value_rows.append(allocator, .empty);
            break :blk representative_rows.items.len - 1;
        };
        try group_value_rows.items[group_index].append(allocator, row);
    }

    const out = try allocator.alloc(f64, group_value_rows.items.len);
    defer allocator.free(out);
    for (group_value_rows.items, out) |rows, *slot| {
        slot.* = distributionMetric(V, aggregation, values, rows.items);
    }

    const output_column = try DeviceColumn.fromSlice(f64, allocator, out, device_value);
    return initMultiKeyAggregatedDataFrame(DeviceDataFrame, frame, key_names, representative_rows.items, output_name, output_column);
}

fn groupByDistributionOn(
    comptime DeviceDataFrame: type,
    aggregation: GroupByDistributionAggregation,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);
    const value = try frame.column(value_name);
    return groupByDistributionOnDispatchValue(DeviceDataFrame, aggregation, frame.allocator, frame, key_names, output_name, value.*, frame.device);
}

pub fn groupByEntropyOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByDistributionOn(DeviceDataFrame, .entropy, frame, key_names, value_name, output_name);
}

pub fn groupByGiniImpurityOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByDistributionOn(DeviceDataFrame, .gini_impurity, frame, key_names, value_name, output_name);
}

pub fn groupByPerplexityOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByDistributionOn(DeviceDataFrame, .perplexity, frame, key_names, value_name, output_name);
}

pub fn groupByInverseSimpsonOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByDistributionOn(DeviceDataFrame, .inverse_simpson, frame, key_names, value_name, output_name);
}

pub fn groupBySimpsonConcentrationOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByDistributionOn(DeviceDataFrame, .simpson_concentration, frame, key_names, value_name, output_name);
}

pub fn groupByEvennessOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByDistributionOn(DeviceDataFrame, .evenness, frame, key_names, value_name, output_name);
}

pub fn groupByInequalityOnDispatchValue(
    comptime DeviceDataFrame: type,
    aggregation: GroupByInequalityAggregation,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
    value: DeviceColumn,
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    return switch (value) {
        .i8 => |typed| groupByInequalityOnTyped(DeviceDataFrame, i8, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .i16 => |typed| groupByInequalityOnTyped(DeviceDataFrame, i16, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .i32 => |typed| groupByInequalityOnTyped(DeviceDataFrame, i32, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .i64 => |typed| groupByInequalityOnTyped(DeviceDataFrame, i64, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u8 => |typed| groupByInequalityOnTyped(DeviceDataFrame, u8, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u16 => |typed| groupByInequalityOnTyped(DeviceDataFrame, u16, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u32 => |typed| groupByInequalityOnTyped(DeviceDataFrame, u32, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u64 => |typed| groupByInequalityOnTyped(DeviceDataFrame, u64, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .usize => |typed| groupByInequalityOnTyped(DeviceDataFrame, usize, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .isize => |typed| groupByInequalityOnTyped(DeviceDataFrame, isize, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .f16 => |typed| groupByInequalityOnTyped(DeviceDataFrame, f16, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .f32 => |typed| groupByInequalityOnTyped(DeviceDataFrame, f32, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .f64 => |typed| groupByInequalityOnTyped(DeviceDataFrame, f64, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

const GroupGiniStats = struct {
    mean: f64,
    mean_diff: f64,
};

const GroupMeanAbsDevStats = struct {
    mean: f64,
    mean_abs_dev: f64,
};

fn groupMeanAbsDevStats(comptime V: type, values: []const V, rows: []const usize) GroupMeanAbsDevStats {
    if (rows.len == 0) return .{ .mean = std.math.nan(f64), .mean_abs_dev = std.math.nan(f64) };

    var total: f64 = 0.0;
    for (rows) |row| total += castToF64(V, values[row]);
    const mean = total / @as(f64, @floatFromInt(rows.len));

    var deviation_sum: f64 = 0.0;
    for (rows) |row| deviation_sum += @abs(castToF64(V, values[row]) - mean);

    return .{
        .mean = mean,
        .mean_abs_dev = deviation_sum / @as(f64, @floatFromInt(rows.len)),
    };
}

fn groupGiniStats(comptime V: type, values: []const V, rows: []const usize) GroupGiniStats {
    if (rows.len == 0) return .{ .mean = std.math.nan(f64), .mean_diff = std.math.nan(f64) };

    var total: f64 = 0.0;
    for (rows) |row| total += castToF64(V, values[row]);
    const mean = total / @as(f64, @floatFromInt(rows.len));

    var pair_sum: f64 = 0.0;
    var pair_count: usize = 0;
    for (rows, 0..) |lhs_row, lhs_index| {
        const lhs = castToF64(V, values[lhs_row]);
        for (rows[lhs_index + 1 ..]) |rhs_row| {
            pair_sum += @abs(lhs - castToF64(V, values[rhs_row]));
            pair_count += 1;
        }
    }

    // Match the existing row-wise contract: a singleton group has zero mean
    // pairwise difference, while the normalized coefficient below still
    // reports NaN for zero-mean groups because the denominator is undefined.
    const mean_diff = if (pair_count == 0) 0.0 else pair_sum / @as(f64, @floatFromInt(pair_count));
    return .{ .mean = mean, .mean_diff = mean_diff };
}

fn groupByInequalityOnTyped(
    comptime DeviceDataFrame: type,
    comptime V: type,
    aggregation: GroupByInequalityAggregation,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
    value: DeviceTypedColumn(V),
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    if (frame.rows != value.len()) return error.LengthMismatch;
    const values = try value.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_value_validity = try validityValues(value, allocator);
    defer if (maybe_value_validity) |validity| allocator.free(validity);

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(allocator);
    var group_value_rows: std.ArrayList(std.ArrayList(usize)) = .empty;
    defer {
        for (group_value_rows.items) |*rows| rows.deinit(allocator);
        group_value_rows.deinit(allocator);
    }

    for (values, 0..) |_, row| {
        if (maybe_value_validity) |validity| {
            if (!validity[row]) continue;
        }
        if (!try rowHasValidKeys(allocator, frame, key_names, row)) continue;
        const group_index = (try findMultiKeyGroupIndex(allocator, frame, key_names, representative_rows.items, row)) orelse blk: {
            try representative_rows.append(allocator, row);
            try group_value_rows.append(allocator, .empty);
            break :blk representative_rows.items.len - 1;
        };
        try group_value_rows.items[group_index].append(allocator, row);
    }

    const out = try allocator.alloc(f64, group_value_rows.items.len);
    defer allocator.free(out);
    for (group_value_rows.items, out) |rows, *slot| {
        slot.* = switch (aggregation) {
            .mean_abs_dev => groupMeanAbsDevStats(V, values, rows.items).mean_abs_dev,
            .mean_abs_dev_ratio => blk: {
                const stats = groupMeanAbsDevStats(V, values, rows.items);
                break :blk if (stats.mean == 0.0) std.math.nan(f64) else stats.mean_abs_dev / @abs(stats.mean);
            },
            .gini_mean_diff => groupGiniStats(V, values, rows.items).mean_diff,
            .gini_coefficient => blk: {
                const stats = groupGiniStats(V, values, rows.items);
                break :blk if (stats.mean == 0.0) std.math.nan(f64) else stats.mean_diff / (2.0 * @abs(stats.mean));
            },
        };
    }

    const output_column = try DeviceColumn.fromSlice(f64, allocator, out, device_value);
    return initMultiKeyAggregatedDataFrame(DeviceDataFrame, frame, key_names, representative_rows.items, output_name, output_column);
}

fn groupByInequalityOn(
    comptime DeviceDataFrame: type,
    aggregation: GroupByInequalityAggregation,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);
    const value = try frame.column(value_name);
    return groupByInequalityOnDispatchValue(DeviceDataFrame, aggregation, frame.allocator, frame, key_names, output_name, value.*, frame.device);
}

pub fn groupByGiniMeanDiffOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByInequalityOn(DeviceDataFrame, .gini_mean_diff, frame, key_names, value_name, output_name);
}

pub fn groupByMeanAbsDevOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByInequalityOn(DeviceDataFrame, .mean_abs_dev, frame, key_names, value_name, output_name);
}

pub fn groupByMeanAbsDevRatioOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByInequalityOn(DeviceDataFrame, .mean_abs_dev_ratio, frame, key_names, value_name, output_name);
}

pub fn groupByGiniCoefficientOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByInequalityOn(DeviceDataFrame, .gini_coefficient, frame, key_names, value_name, output_name);
}

pub fn groupByWeightedOn(
    comptime DeviceDataFrame: type,
    aggregation: GroupByWeightedAggregation,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    weight_name: []const u8,
    output_name: []const u8,
    q: f64,
) GroupByOnError!DeviceDataFrame {
    if (key_names.len == 0) return error.LengthMismatch;
    if (aggregation == .weighted_quantile and (std.math.isNan(q) or q < 0.0 or q > 1.0)) return error.InvalidShape;
    for (key_names) |key_name| _ = try frame.column(key_name);
    const value_column = try frame.column(value_name);
    const weight_column = try frame.column(weight_name);

    var values = try ownedGroupRealColumn(frame.allocator, value_column.*);
    defer values.deinit();
    var weights = try ownedGroupRealColumn(frame.allocator, weight_column.*);
    defer weights.deinit();
    if (frame.rows != values.values.len or frame.rows != weights.values.len) return error.LengthMismatch;

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(frame.allocator);
    var weight_sums: std.ArrayList(f64) = .empty;
    defer weight_sums.deinit(frame.allocator);
    var weighted_sums: std.ArrayList(f64) = .empty;
    defer weighted_sums.deinit(frame.allocator);
    var weighted_square_sums: std.ArrayList(f64) = .empty;
    defer weighted_square_sums.deinit(frame.allocator);
    var group_value_rows: std.ArrayList(std.ArrayList(usize)) = .empty;
    defer {
        for (group_value_rows.items) |*rows| rows.deinit(frame.allocator);
        group_value_rows.deinit(frame.allocator);
    }

    for (0..frame.rows) |row| {
        if (values.validity) |validity| {
            if (!validity[row]) continue;
        }
        if (weights.validity) |validity| {
            if (!validity[row]) continue;
        }
        const weight = weights.values[row];
        if (!try rowHasValidKeys(frame.allocator, frame, key_names, row)) continue;
        if (weight < 0.0) return error.InvalidShape;
        const group_index = (try findMultiKeyGroupIndex(frame.allocator, frame, key_names, representative_rows.items, row)) orelse blk: {
            try representative_rows.append(frame.allocator, row);
            try weight_sums.append(frame.allocator, 0.0);
            try weighted_sums.append(frame.allocator, 0.0);
            try weighted_square_sums.append(frame.allocator, 0.0);
            try group_value_rows.append(frame.allocator, .empty);
            break :blk representative_rows.items.len - 1;
        };
        const value = values.values[row];
        weight_sums.items[group_index] += weight;
        weighted_sums.items[group_index] += value * weight;
        weighted_square_sums.items[group_index] += value * value * weight;
        try group_value_rows.items[group_index].append(frame.allocator, row);
    }

    const out = try frame.allocator.alloc(f64, representative_rows.items.len);
    defer frame.allocator.free(out);
    for (weight_sums.items, weighted_sums.items, weighted_square_sums.items, group_value_rows.items, out) |weight_sum, weighted_sum, weighted_square_sum, rows, *slot| {
        if (weight_sum == 0.0) {
            slot.* = std.math.nan(f64);
            continue;
        }
        slot.* = switch (aggregation) {
            .weighted_mean => weighted_sum / weight_sum,
            .weighted_variance, .weighted_stddev => blk: {
                var centered_square_sum = weighted_square_sum - weighted_sum * weighted_sum / weight_sum;
                if (centered_square_sum < 0.0 and centered_square_sum > -1e-12) centered_square_sum = 0.0;
                const variance = centered_square_sum / weight_sum;
                break :blk if (aggregation == .weighted_stddev) std.math.sqrt(variance) else variance;
            },
            .weighted_quantile => try groupWeightedQuantileFromRows(frame.allocator, rows.items, values.values, weights.values, q, null),
            .weighted_median => try groupWeightedQuantileFromRows(frame.allocator, rows.items, values.values, weights.values, 0.5, null),
            .weighted_iqr => try groupWeightedQuantileFromRows(frame.allocator, rows.items, values.values, weights.values, 0.75, 0.25),
            .weighted_mad => try groupWeightedMadFromRows(frame.allocator, rows.items, values.values, weights.values),
        };
    }

    const output_column = try DeviceColumn.fromSlice(f64, frame.allocator, out, frame.device);
    return initMultiKeyAggregatedDataFrame(DeviceDataFrame, frame, key_names, representative_rows.items, output_name, output_column);
}

pub fn groupByWeightedMeanOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    weight_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByWeightedOn(DeviceDataFrame, .weighted_mean, frame, key_names, value_name, weight_name, output_name, 0.5);
}

pub fn groupByWeightedVarianceOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    weight_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByWeightedOn(DeviceDataFrame, .weighted_variance, frame, key_names, value_name, weight_name, output_name, 0.5);
}

pub fn groupByWeightedStddevOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    weight_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByWeightedOn(DeviceDataFrame, .weighted_stddev, frame, key_names, value_name, weight_name, output_name, 0.5);
}

pub fn groupByWeightedQuantileOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    weight_name: []const u8,
    output_name: []const u8,
    q: f64,
) GroupByOnError!DeviceDataFrame {
    return groupByWeightedOn(DeviceDataFrame, .weighted_quantile, frame, key_names, value_name, weight_name, output_name, q);
}

pub fn groupByWeightedMedianOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    weight_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByWeightedOn(DeviceDataFrame, .weighted_median, frame, key_names, value_name, weight_name, output_name, 0.5);
}

pub fn groupByWeightedIqrOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    weight_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByWeightedOn(DeviceDataFrame, .weighted_iqr, frame, key_names, value_name, weight_name, output_name, 0.5);
}

pub fn groupByWeightedMadOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    weight_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByWeightedOn(DeviceDataFrame, .weighted_mad, frame, key_names, value_name, weight_name, output_name, 0.5);
}

pub fn groupByMomentOnDispatchValue(
    comptime DeviceDataFrame: type,
    aggregation: GroupByMomentAggregation,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
    value: DeviceColumn,
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    return switch (value) {
        .i8 => |typed| groupByMomentOnTyped(DeviceDataFrame, i8, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .i16 => |typed| groupByMomentOnTyped(DeviceDataFrame, i16, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .i32 => |typed| groupByMomentOnTyped(DeviceDataFrame, i32, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .i64 => |typed| groupByMomentOnTyped(DeviceDataFrame, i64, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u8 => |typed| groupByMomentOnTyped(DeviceDataFrame, u8, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u16 => |typed| groupByMomentOnTyped(DeviceDataFrame, u16, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u32 => |typed| groupByMomentOnTyped(DeviceDataFrame, u32, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u64 => |typed| groupByMomentOnTyped(DeviceDataFrame, u64, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .usize => |typed| groupByMomentOnTyped(DeviceDataFrame, usize, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .isize => |typed| groupByMomentOnTyped(DeviceDataFrame, isize, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .f16 => |typed| groupByMomentOnTyped(DeviceDataFrame, f16, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .f32 => |typed| groupByMomentOnTyped(DeviceDataFrame, f32, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .f64 => |typed| groupByMomentOnTyped(DeviceDataFrame, f64, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn groupByMomentOnTyped(
    comptime DeviceDataFrame: type,
    comptime V: type,
    aggregation: GroupByMomentAggregation,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
    value: DeviceTypedColumn(V),
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    if (frame.rows != value.len()) return error.LengthMismatch;
    const values = try value.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_value_validity = try validityValues(value, allocator);
    defer if (maybe_value_validity) |validity| allocator.free(validity);

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(allocator);
    var profiles: std.ArrayList(MomentProfile) = .empty;
    defer profiles.deinit(allocator);

    for (values, 0..) |value_item, row| {
        if (maybe_value_validity) |validity| {
            if (!validity[row]) continue;
        }
        if (!try rowHasValidKeys(allocator, frame, key_names, row)) continue;
        const group_index = (try findMultiKeyGroupIndex(allocator, frame, key_names, representative_rows.items, row)) orelse blk: {
            try representative_rows.append(allocator, row);
            try profiles.append(allocator, .{});
            break :blk representative_rows.items.len - 1;
        };
        const value_f64 = castToF64(V, value_item);
        const profile_value = switch (aggregation) {
            .magnitude_variance,
            .magnitude_stddev,
            .magnitude_sem,
            .magnitude_cv,
            .magnitude_fano,
            .magnitude_skewness,
            .magnitude_kurtosis,
            => @abs(value_f64),
            else => value_f64,
        };
        profiles.items[group_index].update(profile_value);
    }

    const values_out = try allocator.alloc(f64, profiles.items.len);
    defer allocator.free(values_out);
    for (profiles.items, values_out) |profile, *slot| {
        slot.* = switch (aggregation) {
            .variance, .magnitude_variance => profile.variance(),
            .stddev, .magnitude_stddev => profile.stddev(),
            .sem, .magnitude_sem => profile.sem(),
            .cv, .magnitude_cv => profile.cv(),
            .fano, .magnitude_fano => if (profile.mean == 0.0) std.math.nan(f64) else profile.variance() / profile.mean,
            .skewness, .magnitude_skewness => profile.skewness(),
            .kurtosis, .magnitude_kurtosis => profile.kurtosis(),
        };
    }

    const output_column = try DeviceColumn.fromSlice(f64, allocator, values_out, device_value);
    return initMultiKeyAggregatedDataFrame(DeviceDataFrame, frame, key_names, representative_rows.items, output_name, output_column);
}

fn groupByMomentOn(
    comptime DeviceDataFrame: type,
    aggregation: GroupByMomentAggregation,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);
    const value = try frame.column(value_name);
    return groupByMomentOnDispatchValue(DeviceDataFrame, aggregation, frame.allocator, frame, key_names, output_name, value.*, frame.device);
}

pub fn groupByVarianceOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByMomentOn(DeviceDataFrame, .variance, frame, key_names, value_name, output_name);
}

pub fn groupByStddevOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByMomentOn(DeviceDataFrame, .stddev, frame, key_names, value_name, output_name);
}

pub fn groupBySemOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByMomentOn(DeviceDataFrame, .sem, frame, key_names, value_name, output_name);
}

pub fn groupByCvOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByMomentOn(DeviceDataFrame, .cv, frame, key_names, value_name, output_name);
}

pub fn groupByFanoOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByMomentOn(DeviceDataFrame, .fano, frame, key_names, value_name, output_name);
}

pub fn groupBySkewnessOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByMomentOn(DeviceDataFrame, .skewness, frame, key_names, value_name, output_name);
}

pub fn groupByKurtosisOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByMomentOn(DeviceDataFrame, .kurtosis, frame, key_names, value_name, output_name);
}

pub fn groupByMagnitudeVarianceOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByMomentOn(DeviceDataFrame, .magnitude_variance, frame, key_names, value_name, output_name);
}

pub fn groupByMagnitudeStddevOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByMomentOn(DeviceDataFrame, .magnitude_stddev, frame, key_names, value_name, output_name);
}

pub fn groupByMagnitudeSemOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByMomentOn(DeviceDataFrame, .magnitude_sem, frame, key_names, value_name, output_name);
}

pub fn groupByMagnitudeCvOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByMomentOn(DeviceDataFrame, .magnitude_cv, frame, key_names, value_name, output_name);
}

pub fn groupByMagnitudeFanoOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByMomentOn(DeviceDataFrame, .magnitude_fano, frame, key_names, value_name, output_name);
}

pub fn groupByMagnitudeSkewnessOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByMomentOn(DeviceDataFrame, .magnitude_skewness, frame, key_names, value_name, output_name);
}

pub fn groupByMagnitudeKurtosisOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByMomentOn(DeviceDataFrame, .magnitude_kurtosis, frame, key_names, value_name, output_name);
}

pub fn groupByRealOnDispatchValue(
    comptime DeviceDataFrame: type,
    aggregation: GroupByRealAggregation,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
    value: DeviceColumn,
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    return switch (value) {
        .i8 => |typed| groupByRealOnTyped(DeviceDataFrame, i8, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .i16 => |typed| groupByRealOnTyped(DeviceDataFrame, i16, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .i32 => |typed| groupByRealOnTyped(DeviceDataFrame, i32, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .i64 => |typed| groupByRealOnTyped(DeviceDataFrame, i64, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u8 => |typed| groupByRealOnTyped(DeviceDataFrame, u8, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u16 => |typed| groupByRealOnTyped(DeviceDataFrame, u16, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u32 => |typed| groupByRealOnTyped(DeviceDataFrame, u32, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u64 => |typed| groupByRealOnTyped(DeviceDataFrame, u64, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .usize => |typed| groupByRealOnTyped(DeviceDataFrame, usize, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .isize => |typed| groupByRealOnTyped(DeviceDataFrame, isize, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .f16 => |typed| groupByRealOnTyped(DeviceDataFrame, f16, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .f32 => |typed| groupByRealOnTyped(DeviceDataFrame, f32, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .f64 => |typed| groupByRealOnTyped(DeviceDataFrame, f64, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn groupByRealOnTyped(
    comptime DeviceDataFrame: type,
    comptime V: type,
    aggregation: GroupByRealAggregation,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
    value: DeviceTypedColumn(V),
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    if (frame.rows != value.len()) return error.LengthMismatch;
    const values = try value.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_value_validity = try validityValues(value, allocator);
    defer if (maybe_value_validity) |validity| allocator.free(validity);

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(allocator);
    var totals: std.ArrayList(f64) = .empty;
    defer totals.deinit(allocator);
    var counts: std.ArrayList(i64) = .empty;
    defer counts.deinit(allocator);
    var zero_seen: std.ArrayList(bool) = .empty;
    defer zero_seen.deinit(allocator);
    var aux_values: std.ArrayList(f64) = .empty;
    defer aux_values.deinit(allocator);
    var secondary_values: std.ArrayList(f64) = .empty;
    defer secondary_values.deinit(allocator);

    for (values, 0..) |value_item, row| {
        if (maybe_value_validity) |validity| {
            if (!validity[row]) continue;
        }
        if (!try rowHasValidKeys(allocator, frame, key_names, row)) continue;
        const group_index = (try findMultiKeyGroupIndex(allocator, frame, key_names, representative_rows.items, row)) orelse blk: {
            try representative_rows.append(allocator, row);
            try totals.append(allocator, 0.0);
            try counts.append(allocator, 0);
            try zero_seen.append(allocator, false);
            try aux_values.append(allocator, 0.0);
            try secondary_values.append(allocator, 0.0);
            break :blk representative_rows.items.len - 1;
        };
        const value_f64 = castToF64(V, value_item);
        switch (aggregation) {
            .mean_abs, .l1_norm => totals.items[group_index] += @abs(value_f64),
            .mean_square, .rms, .l2_norm => totals.items[group_index] += value_f64 * value_f64,
            .hhi, .magnitude_normalized_hhi, .magnitude_sparsity, .magnitude_inverse_simpson, .magnitude_simpson_evenness => {
                const magnitude = @abs(value_f64);
                totals.items[group_index] += magnitude;
                aux_values.items[group_index] += magnitude * magnitude;
            },
            .magnitude_dominance => {
                const magnitude = @abs(value_f64);
                totals.items[group_index] += magnitude;
                if (counts.items[group_index] == 0 or std.math.isNan(magnitude) or (!std.math.isNan(aux_values.items[group_index]) and magnitude > aux_values.items[group_index])) {
                    aux_values.items[group_index] = magnitude;
                }
            },
            .magnitude_dominance_margin => {
                const magnitude = @abs(value_f64);
                totals.items[group_index] += magnitude;
                if (counts.items[group_index] == 0 or std.math.isNan(magnitude)) {
                    aux_values.items[group_index] = magnitude;
                } else if (!std.math.isNan(aux_values.items[group_index])) {
                    if (magnitude > aux_values.items[group_index]) {
                        secondary_values.items[group_index] = aux_values.items[group_index];
                        aux_values.items[group_index] = magnitude;
                    } else if (magnitude > secondary_values.items[group_index]) {
                        secondary_values.items[group_index] = magnitude;
                    }
                }
            },
            .magnitude_entropy, .magnitude_perplexity, .magnitude_evenness => {
                const magnitude = @abs(value_f64);
                totals.items[group_index] += magnitude;
                if (magnitude > 0.0) aux_values.items[group_index] += magnitude * std.math.log(f64, std.math.e, magnitude);
            },
            .max_abs => {
                const magnitude = @abs(value_f64);
                if (counts.items[group_index] == 0 or std.math.isNan(magnitude) or (!std.math.isNan(totals.items[group_index]) and magnitude > totals.items[group_index])) {
                    totals.items[group_index] = magnitude;
                }
            },
            .min_abs => {
                const magnitude = @abs(value_f64);
                if (counts.items[group_index] == 0 or std.math.isNan(magnitude) or (!std.math.isNan(totals.items[group_index]) and magnitude < totals.items[group_index])) {
                    totals.items[group_index] = magnitude;
                }
            },
            .geometric_mean => {
                if (value_f64 < 0.0) {
                    totals.items[group_index] = std.math.nan(f64);
                } else if (value_f64 == 0.0 and !std.math.isNan(totals.items[group_index])) {
                    zero_seen.items[group_index] = true;
                    totals.items[group_index] = 0.0;
                } else if (!zero_seen.items[group_index] and !std.math.isNan(totals.items[group_index])) {
                    totals.items[group_index] += std.math.log(f64, std.math.e, value_f64);
                }
            },
            .harmonic_mean => {
                if (value_f64 == 0.0 and !std.math.isNan(totals.items[group_index])) {
                    totals.items[group_index] = std.math.inf(f64);
                } else if (!std.math.isInf(totals.items[group_index])) {
                    totals.items[group_index] += 1.0 / value_f64;
                }
            },
            .logsumexp, .logmeanexp => {
                if (std.math.isNan(value_f64)) {
                    totals.items[group_index] = std.math.nan(f64);
                    aux_values.items[group_index] = std.math.nan(f64);
                } else if (counts.items[group_index] == 0) {
                    aux_values.items[group_index] = value_f64;
                    totals.items[group_index] = 1.0;
                } else if (!std.math.isNan(totals.items[group_index])) {
                    if (std.math.isPositiveInf(aux_values.items[group_index])) {
                        totals.items[group_index] = 1.0;
                    } else if (std.math.isPositiveInf(value_f64)) {
                        aux_values.items[group_index] = value_f64;
                        totals.items[group_index] = 1.0;
                    } else if (value_f64 > aux_values.items[group_index]) {
                        totals.items[group_index] = totals.items[group_index] * std.math.exp(aux_values.items[group_index] - value_f64) + 1.0;
                        aux_values.items[group_index] = value_f64;
                    } else if (!(std.math.isNegativeInf(aux_values.items[group_index]) and std.math.isNegativeInf(value_f64))) {
                        totals.items[group_index] += std.math.exp(value_f64 - aux_values.items[group_index]);
                    }
                }
            },
            .ptp, .midrange, .range_coeff => {
                if (counts.items[group_index] == 0) {
                    totals.items[group_index] = value_f64;
                    aux_values.items[group_index] = value_f64;
                } else if (std.math.isNan(value_f64)) {
                    totals.items[group_index] = value_f64;
                    aux_values.items[group_index] = value_f64;
                } else if (!std.math.isNan(totals.items[group_index])) {
                    if (value_f64 < totals.items[group_index]) totals.items[group_index] = value_f64;
                    if (value_f64 > aux_values.items[group_index]) aux_values.items[group_index] = value_f64;
                }
            },
        }
        counts.items[group_index] += 1;
    }

    const out = try allocator.alloc(f64, totals.items.len);
    defer allocator.free(out);
    for (totals.items, counts.items, zero_seen.items, aux_values.items, secondary_values.items, out) |total, count, has_zero, aux_value, secondary_value, *slot| {
        slot.* = switch (aggregation) {
            .mean_abs => total / @as(f64, @floatFromInt(count)),
            .mean_square => total / @as(f64, @floatFromInt(count)),
            .rms => std.math.sqrt(total / @as(f64, @floatFromInt(count))),
            .l1_norm => total,
            .l2_norm => std.math.sqrt(total),
            .max_abs, .min_abs => total,
            .geometric_mean => if (std.math.isNan(total)) std.math.nan(f64) else if (has_zero) 0.0 else std.math.exp(total / @as(f64, @floatFromInt(count))),
            .harmonic_mean => if (std.math.isInf(total)) 0.0 else @as(f64, @floatFromInt(count)) / total,
            .logsumexp, .logmeanexp => blk: {
                if (std.math.isNan(total) or std.math.isNan(aux_value)) break :blk std.math.nan(f64);
                if (std.math.isPositiveInf(aux_value) or std.math.isNegativeInf(aux_value)) break :blk aux_value;
                var result = aux_value + std.math.log(f64, std.math.e, total);
                if (aggregation == .logmeanexp) result -= std.math.log(f64, std.math.e, @as(f64, @floatFromInt(count)));
                break :blk result;
            },
            .ptp => aux_value - total,
            .midrange => (total + aux_value) / 2.0,
            .range_coeff => blk: {
                const denominator = aux_value + total;
                break :blk if (denominator == 0.0) std.math.nan(f64) else (aux_value - total) / denominator;
            },
            .hhi => if (total == 0.0) std.math.nan(f64) else aux_value / (total * total),
            .magnitude_normalized_hhi => blk: {
                if (total == 0.0) break :blk std.math.nan(f64);
                if (count <= 1) break :blk 1.0;
                const concentration = aux_value / (total * total);
                const uniform_floor = 1.0 / @as(f64, @floatFromInt(count));
                break :blk (concentration - uniform_floor) / (1.0 - uniform_floor);
            },
            .magnitude_sparsity => blk: {
                if (total == 0.0 or aux_value == 0.0) break :blk std.math.nan(f64);
                if (count <= 1) break :blk 1.0;
                const sqrt_count = std.math.sqrt(@as(f64, @floatFromInt(count)));
                const l1_over_l2 = total / std.math.sqrt(aux_value);
                break :blk (sqrt_count - l1_over_l2) / (sqrt_count - 1.0);
            },
            .magnitude_inverse_simpson => if (total == 0.0 or aux_value == 0.0) std.math.nan(f64) else (total * total) / aux_value,
            .magnitude_simpson_evenness => if (total == 0.0 or aux_value == 0.0) std.math.nan(f64) else (total * total) / (aux_value * @as(f64, @floatFromInt(count))),
            .magnitude_dominance => if (total == 0.0) std.math.nan(f64) else aux_value / total,
            .magnitude_dominance_margin => if (total == 0.0) std.math.nan(f64) else (aux_value - secondary_value) / total,
            .magnitude_entropy => if (total == 0.0) std.math.nan(f64) else std.math.log(f64, std.math.e, total) - aux_value / total,
            .magnitude_perplexity => if (total == 0.0) std.math.nan(f64) else std.math.exp(std.math.log(f64, std.math.e, total) - aux_value / total),
            .magnitude_evenness => if (count <= 1) 1.0 else if (total == 0.0) std.math.nan(f64) else (std.math.log(f64, std.math.e, total) - aux_value / total) / std.math.log(f64, std.math.e, @as(f64, @floatFromInt(count))),
        };
    }

    const output_column = try DeviceColumn.fromSlice(f64, allocator, out, device_value);
    return initMultiKeyAggregatedDataFrame(DeviceDataFrame, frame, key_names, representative_rows.items, output_name, output_column);
}

fn groupByRealOn(
    comptime DeviceDataFrame: type,
    aggregation: GroupByRealAggregation,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);
    const value = try frame.column(value_name);
    return groupByRealOnDispatchValue(DeviceDataFrame, aggregation, frame.allocator, frame, key_names, output_name, value.*, frame.device);
}

pub fn groupByMeanAbsOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRealOn(DeviceDataFrame, .mean_abs, frame, key_names, value_name, output_name);
}

pub fn groupByMeanSquareOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRealOn(DeviceDataFrame, .mean_square, frame, key_names, value_name, output_name);
}

pub fn groupByRmsOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRealOn(DeviceDataFrame, .rms, frame, key_names, value_name, output_name);
}

pub fn groupByL1NormOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRealOn(DeviceDataFrame, .l1_norm, frame, key_names, value_name, output_name);
}

pub fn groupByL2NormOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRealOn(DeviceDataFrame, .l2_norm, frame, key_names, value_name, output_name);
}

pub fn groupByMaxAbsOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRealOn(DeviceDataFrame, .max_abs, frame, key_names, value_name, output_name);
}

pub fn groupByMinAbsOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRealOn(DeviceDataFrame, .min_abs, frame, key_names, value_name, output_name);
}

pub fn groupByHhiOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRealOn(DeviceDataFrame, .hhi, frame, key_names, value_name, output_name);
}

pub fn groupByMagnitudeNormalizedHhiOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRealOn(DeviceDataFrame, .magnitude_normalized_hhi, frame, key_names, value_name, output_name);
}

pub fn groupByMagnitudeSparsityOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRealOn(DeviceDataFrame, .magnitude_sparsity, frame, key_names, value_name, output_name);
}

pub fn groupByMagnitudeInverseSimpsonOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRealOn(DeviceDataFrame, .magnitude_inverse_simpson, frame, key_names, value_name, output_name);
}

pub fn groupByMagnitudeSimpsonEvennessOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRealOn(DeviceDataFrame, .magnitude_simpson_evenness, frame, key_names, value_name, output_name);
}

pub fn groupByMagnitudeDominanceOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRealOn(DeviceDataFrame, .magnitude_dominance, frame, key_names, value_name, output_name);
}

pub fn groupByMagnitudeDominanceMarginOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRealOn(DeviceDataFrame, .magnitude_dominance_margin, frame, key_names, value_name, output_name);
}

pub fn groupByMagnitudeEntropyOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRealOn(DeviceDataFrame, .magnitude_entropy, frame, key_names, value_name, output_name);
}

pub fn groupByMagnitudePerplexityOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRealOn(DeviceDataFrame, .magnitude_perplexity, frame, key_names, value_name, output_name);
}

pub fn groupByMagnitudeEvennessOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRealOn(DeviceDataFrame, .magnitude_evenness, frame, key_names, value_name, output_name);
}

pub fn groupByGeometricMeanOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRealOn(DeviceDataFrame, .geometric_mean, frame, key_names, value_name, output_name);
}

pub fn groupByHarmonicMeanOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRealOn(DeviceDataFrame, .harmonic_mean, frame, key_names, value_name, output_name);
}

pub fn groupByLogSumExpOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRealOn(DeviceDataFrame, .logsumexp, frame, key_names, value_name, output_name);
}

pub fn groupByLogMeanExpOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRealOn(DeviceDataFrame, .logmeanexp, frame, key_names, value_name, output_name);
}

pub fn groupByPtpOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRealOn(DeviceDataFrame, .ptp, frame, key_names, value_name, output_name);
}

pub fn groupByMidrangeOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRealOn(DeviceDataFrame, .midrange, frame, key_names, value_name, output_name);
}

pub fn groupByRangeCoeffOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRealOn(DeviceDataFrame, .range_coeff, frame, key_names, value_name, output_name);
}

fn groupByQuantileLess(_: void, lhs: f64, rhs: f64) bool {
    const lhs_nan = std.math.isNan(lhs);
    const rhs_nan = std.math.isNan(rhs);
    if (lhs_nan != rhs_nan) return !lhs_nan;
    if (lhs_nan and rhs_nan) return false;
    return lhs < rhs;
}

fn quantileFromSorted(sorted_values: []const f64, q: f64) f64 {
    const max_index = sorted_values.len - 1;
    const position = q * @as(f64, @floatFromInt(max_index));
    const lower_float = @floor(position);
    const lower: usize = @intFromFloat(lower_float);
    const upper = @min(lower + 1, max_index);
    const weight = position - lower_float;
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight;
}

fn validateTailFraction(fraction: f64) GroupByOnError!void {
    if (std.math.isNan(fraction) or fraction < 0.0 or fraction >= 0.5) return error.InvalidShape;
}

fn groupByRobustUsesTailFraction(aggregation: GroupByRobustAggregation) bool {
    return switch (aggregation) {
        .trimmed_mean, .winsorized_mean => true,
        else => false,
    };
}

fn tailCount(len: usize, fraction: f64) usize {
    return @intFromFloat(@floor(@as(f64, @floatFromInt(len)) * fraction));
}

fn trimmedMeanFromSorted(sorted_values: []const f64, trim_fraction: f64) f64 {
    const trim_count = tailCount(sorted_values.len, trim_fraction);
    const trimmed = sorted_values[trim_count .. sorted_values.len - trim_count];
    var total: f64 = 0.0;
    for (trimmed) |value| total += value;
    return total / @as(f64, @floatFromInt(trimmed.len));
}

fn winsorizedMeanFromSorted(sorted_values: []const f64, winsor_fraction: f64) f64 {
    const winsor_count = tailCount(sorted_values.len, winsor_fraction);
    const lower = sorted_values[winsor_count];
    const upper = sorted_values[sorted_values.len - winsor_count - 1];
    var total: f64 = 0.0;
    for (sorted_values) |value| total += @min(@max(value, lower), upper);
    return total / @as(f64, @floatFromInt(sorted_values.len));
}

pub fn groupByQuantileOnDispatchValue(
    comptime DeviceDataFrame: type,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
    value: DeviceColumn,
    device_value: array_mod.Device,
    q: f64,
) GroupByOnError!DeviceDataFrame {
    if (std.math.isNan(q) or q < 0.0 or q > 1.0) return error.InvalidShape;
    return switch (value) {
        .i8 => |typed| groupByQuantileOnTyped(DeviceDataFrame, i8, allocator, frame, key_names, output_name, typed, device_value, q),
        .i16 => |typed| groupByQuantileOnTyped(DeviceDataFrame, i16, allocator, frame, key_names, output_name, typed, device_value, q),
        .i32 => |typed| groupByQuantileOnTyped(DeviceDataFrame, i32, allocator, frame, key_names, output_name, typed, device_value, q),
        .i64 => |typed| groupByQuantileOnTyped(DeviceDataFrame, i64, allocator, frame, key_names, output_name, typed, device_value, q),
        .u8 => |typed| groupByQuantileOnTyped(DeviceDataFrame, u8, allocator, frame, key_names, output_name, typed, device_value, q),
        .u16 => |typed| groupByQuantileOnTyped(DeviceDataFrame, u16, allocator, frame, key_names, output_name, typed, device_value, q),
        .u32 => |typed| groupByQuantileOnTyped(DeviceDataFrame, u32, allocator, frame, key_names, output_name, typed, device_value, q),
        .u64 => |typed| groupByQuantileOnTyped(DeviceDataFrame, u64, allocator, frame, key_names, output_name, typed, device_value, q),
        .usize => |typed| groupByQuantileOnTyped(DeviceDataFrame, usize, allocator, frame, key_names, output_name, typed, device_value, q),
        .isize => |typed| groupByQuantileOnTyped(DeviceDataFrame, isize, allocator, frame, key_names, output_name, typed, device_value, q),
        .f16 => |typed| groupByQuantileOnTyped(DeviceDataFrame, f16, allocator, frame, key_names, output_name, typed, device_value, q),
        .f32 => |typed| groupByQuantileOnTyped(DeviceDataFrame, f32, allocator, frame, key_names, output_name, typed, device_value, q),
        .f64 => |typed| groupByQuantileOnTyped(DeviceDataFrame, f64, allocator, frame, key_names, output_name, typed, device_value, q),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn groupByQuantileOnTyped(
    comptime DeviceDataFrame: type,
    comptime V: type,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
    value: DeviceTypedColumn(V),
    device_value: array_mod.Device,
    q: f64,
) GroupByOnError!DeviceDataFrame {
    if (frame.rows != value.len()) return error.LengthMismatch;
    const values = try value.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_value_validity = try validityValues(value, allocator);
    defer if (maybe_value_validity) |validity| allocator.free(validity);

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(allocator);
    var group_values: std.ArrayList(std.ArrayList(f64)) = .empty;
    defer {
        for (group_values.items) |*rows| rows.deinit(allocator);
        group_values.deinit(allocator);
    }

    for (values, 0..) |value_item, row| {
        if (maybe_value_validity) |validity| {
            if (!validity[row]) continue;
        }
        if (!try rowHasValidKeys(allocator, frame, key_names, row)) continue;
        const group_index = (try findMultiKeyGroupIndex(allocator, frame, key_names, representative_rows.items, row)) orelse blk: {
            try representative_rows.append(allocator, row);
            try group_values.append(allocator, .empty);
            break :blk representative_rows.items.len - 1;
        };
        try group_values.items[group_index].append(allocator, castToF64(V, value_item));
    }

    const quantiles = try allocator.alloc(f64, group_values.items.len);
    defer allocator.free(quantiles);
    for (group_values.items, quantiles) |values_for_group, *slot| {
        std.sort.insertion(f64, values_for_group.items, {}, groupByQuantileLess);
        slot.* = quantileFromSorted(values_for_group.items, q);
    }

    const quantile_column = try DeviceColumn.fromSlice(f64, allocator, quantiles, device_value);
    return initMultiKeyAggregatedDataFrame(DeviceDataFrame, frame, key_names, representative_rows.items, output_name, quantile_column);
}

pub fn groupByQuantileOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
    q: f64,
) GroupByOnError!DeviceDataFrame {
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);
    const value = try frame.column(value_name);
    return groupByQuantileOnDispatchValue(DeviceDataFrame, frame.allocator, frame, key_names, output_name, value.*, frame.device, q);
}

pub fn groupByMedianOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByQuantileOn(DeviceDataFrame, frame, key_names, value_name, output_name, 0.5);
}

pub fn groupByRobustOnDispatchValue(
    comptime DeviceDataFrame: type,
    aggregation: GroupByRobustAggregation,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
    value: DeviceColumn,
    device_value: array_mod.Device,
    fraction: f64,
) GroupByOnError!DeviceDataFrame {
    if (groupByRobustUsesTailFraction(aggregation)) try validateTailFraction(fraction);
    return switch (value) {
        .i8 => |typed| groupByRobustOnTyped(DeviceDataFrame, i8, aggregation, allocator, frame, key_names, output_name, typed, device_value, fraction),
        .i16 => |typed| groupByRobustOnTyped(DeviceDataFrame, i16, aggregation, allocator, frame, key_names, output_name, typed, device_value, fraction),
        .i32 => |typed| groupByRobustOnTyped(DeviceDataFrame, i32, aggregation, allocator, frame, key_names, output_name, typed, device_value, fraction),
        .i64 => |typed| groupByRobustOnTyped(DeviceDataFrame, i64, aggregation, allocator, frame, key_names, output_name, typed, device_value, fraction),
        .u8 => |typed| groupByRobustOnTyped(DeviceDataFrame, u8, aggregation, allocator, frame, key_names, output_name, typed, device_value, fraction),
        .u16 => |typed| groupByRobustOnTyped(DeviceDataFrame, u16, aggregation, allocator, frame, key_names, output_name, typed, device_value, fraction),
        .u32 => |typed| groupByRobustOnTyped(DeviceDataFrame, u32, aggregation, allocator, frame, key_names, output_name, typed, device_value, fraction),
        .u64 => |typed| groupByRobustOnTyped(DeviceDataFrame, u64, aggregation, allocator, frame, key_names, output_name, typed, device_value, fraction),
        .usize => |typed| groupByRobustOnTyped(DeviceDataFrame, usize, aggregation, allocator, frame, key_names, output_name, typed, device_value, fraction),
        .isize => |typed| groupByRobustOnTyped(DeviceDataFrame, isize, aggregation, allocator, frame, key_names, output_name, typed, device_value, fraction),
        .f16 => |typed| groupByRobustOnTyped(DeviceDataFrame, f16, aggregation, allocator, frame, key_names, output_name, typed, device_value, fraction),
        .f32 => |typed| groupByRobustOnTyped(DeviceDataFrame, f32, aggregation, allocator, frame, key_names, output_name, typed, device_value, fraction),
        .f64 => |typed| groupByRobustOnTyped(DeviceDataFrame, f64, aggregation, allocator, frame, key_names, output_name, typed, device_value, fraction),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn groupByRobustOnTyped(
    comptime DeviceDataFrame: type,
    comptime V: type,
    aggregation: GroupByRobustAggregation,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
    value: DeviceTypedColumn(V),
    device_value: array_mod.Device,
    fraction: f64,
) GroupByOnError!DeviceDataFrame {
    if (frame.rows != value.len()) return error.LengthMismatch;
    const values = try value.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_value_validity = try validityValues(value, allocator);
    defer if (maybe_value_validity) |validity| allocator.free(validity);

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(allocator);
    var group_values: std.ArrayList(std.ArrayList(f64)) = .empty;
    defer {
        for (group_values.items) |*rows| rows.deinit(allocator);
        group_values.deinit(allocator);
    }

    for (values, 0..) |value_item, row| {
        if (maybe_value_validity) |validity| {
            if (!validity[row]) continue;
        }
        if (!try rowHasValidKeys(allocator, frame, key_names, row)) continue;
        const group_index = (try findMultiKeyGroupIndex(allocator, frame, key_names, representative_rows.items, row)) orelse blk: {
            try representative_rows.append(allocator, row);
            try group_values.append(allocator, .empty);
            break :blk representative_rows.items.len - 1;
        };
        try group_values.items[group_index].append(allocator, castToF64(V, value_item));
    }

    const out = try allocator.alloc(f64, group_values.items.len);
    defer allocator.free(out);
    for (group_values.items, out) |values_for_group, *slot| {
        std.sort.insertion(f64, values_for_group.items, {}, groupByQuantileLess);
        slot.* = switch (aggregation) {
            .iqr => quantileFromSorted(values_for_group.items, 0.75) - quantileFromSorted(values_for_group.items, 0.25),
            .mad => blk: {
                const center = quantileFromSorted(values_for_group.items, 0.5);
                for (values_for_group.items) |*item| item.* = @abs(item.* - center);
                std.sort.insertion(f64, values_for_group.items, {}, groupByQuantileLess);
                break :blk quantileFromSorted(values_for_group.items, 0.5);
            },
            .trimmed_mean => trimmedMeanFromSorted(values_for_group.items, fraction),
            .winsorized_mean => winsorizedMeanFromSorted(values_for_group.items, fraction),
            .interdecile_range => quantileFromSorted(values_for_group.items, 0.9) - quantileFromSorted(values_for_group.items, 0.1),
            .midhinge => (quantileFromSorted(values_for_group.items, 0.25) + quantileFromSorted(values_for_group.items, 0.75)) / 2.0,
            .trimean => (quantileFromSorted(values_for_group.items, 0.25) + 2.0 * quantileFromSorted(values_for_group.items, 0.5) + quantileFromSorted(values_for_group.items, 0.75)) / 4.0,
            .bowley_skewness => blk: {
                const q1 = quantileFromSorted(values_for_group.items, 0.25);
                const median = quantileFromSorted(values_for_group.items, 0.5);
                const q3 = quantileFromSorted(values_for_group.items, 0.75);
                const iqr = q3 - q1;
                break :blk if (iqr == 0.0) std.math.nan(f64) else (q3 + q1 - 2.0 * median) / iqr;
            },
            .quartile_coeff_dispersion => blk: {
                const q1 = quantileFromSorted(values_for_group.items, 0.25);
                const q3 = quantileFromSorted(values_for_group.items, 0.75);
                const denominator = q3 + q1;
                break :blk if (denominator == 0.0) std.math.nan(f64) else (q3 - q1) / denominator;
            },
            .kelley_skewness => blk: {
                const p10 = quantileFromSorted(values_for_group.items, 0.1);
                const median = quantileFromSorted(values_for_group.items, 0.5);
                const p90 = quantileFromSorted(values_for_group.items, 0.9);
                const spread = p90 - p10;
                break :blk if (spread == 0.0) std.math.nan(f64) else (p90 + p10 - 2.0 * median) / spread;
            },
        };
    }

    const output_column = try DeviceColumn.fromSlice(f64, allocator, out, device_value);
    return initMultiKeyAggregatedDataFrame(DeviceDataFrame, frame, key_names, representative_rows.items, output_name, output_column);
}

fn groupByRobustOn(
    comptime DeviceDataFrame: type,
    aggregation: GroupByRobustAggregation,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
    fraction: f64,
) GroupByOnError!DeviceDataFrame {
    if (groupByRobustUsesTailFraction(aggregation)) try validateTailFraction(fraction);
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);
    const value = try frame.column(value_name);
    return groupByRobustOnDispatchValue(DeviceDataFrame, aggregation, frame.allocator, frame, key_names, output_name, value.*, frame.device, fraction);
}

pub fn groupByIqrOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRobustOn(DeviceDataFrame, .iqr, frame, key_names, value_name, output_name, 0.0);
}

pub fn groupByMadOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRobustOn(DeviceDataFrame, .mad, frame, key_names, value_name, output_name, 0.0);
}

pub fn groupByTrimmedMeanOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
    trim_fraction: f64,
) GroupByOnError!DeviceDataFrame {
    return groupByRobustOn(DeviceDataFrame, .trimmed_mean, frame, key_names, value_name, output_name, trim_fraction);
}

pub fn groupByWinsorizedMeanOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
    winsor_fraction: f64,
) GroupByOnError!DeviceDataFrame {
    return groupByRobustOn(DeviceDataFrame, .winsorized_mean, frame, key_names, value_name, output_name, winsor_fraction);
}

pub fn groupByInterdecileRangeOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRobustOn(DeviceDataFrame, .interdecile_range, frame, key_names, value_name, output_name, 0.0);
}

pub fn groupByMidhingeOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRobustOn(DeviceDataFrame, .midhinge, frame, key_names, value_name, output_name, 0.0);
}

pub fn groupByTrimeanOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRobustOn(DeviceDataFrame, .trimean, frame, key_names, value_name, output_name, 0.0);
}

pub fn groupByBowleySkewnessOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRobustOn(DeviceDataFrame, .bowley_skewness, frame, key_names, value_name, output_name, 0.0);
}

pub fn groupByQuartileCoeffDispersionOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRobustOn(DeviceDataFrame, .quartile_coeff_dispersion, frame, key_names, value_name, output_name, 0.0);
}

pub fn groupByKelleySkewnessOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRobustOn(DeviceDataFrame, .kelley_skewness, frame, key_names, value_name, output_name, 0.0);
}

fn groupByBoolOn(
    comptime DeviceDataFrame: type,
    aggregation: GroupByBoolAggregation,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);
    const value = try frame.column(value_name);
    if (value.* != .bool) return error.TypeUnsupported;

    const values = try value.bool.values.toOwnedSlice(frame.allocator);
    defer frame.allocator.free(values);
    const maybe_value_validity = try validityValues(value.bool, frame.allocator);
    defer if (maybe_value_validity) |validity| frame.allocator.free(validity);

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(frame.allocator);
    var true_counts: std.ArrayList(i64) = .empty;
    defer true_counts.deinit(frame.allocator);
    var false_counts: std.ArrayList(i64) = .empty;
    defer false_counts.deinit(frame.allocator);

    for (values, 0..) |value_item, row| {
        if (maybe_value_validity) |validity| {
            if (!validity[row]) continue;
        }
        if (!try rowHasValidKeys(frame.allocator, frame, key_names, row)) continue;
        const group_index = (try findMultiKeyGroupIndex(frame.allocator, frame, key_names, representative_rows.items, row)) orelse blk: {
            try representative_rows.append(frame.allocator, row);
            try true_counts.append(frame.allocator, 0);
            try false_counts.append(frame.allocator, 0);
            break :blk representative_rows.items.len - 1;
        };
        if (value_item) {
            true_counts.items[group_index] += 1;
        } else {
            false_counts.items[group_index] += 1;
        }
    }

    const output_column: DeviceColumn = switch (aggregation) {
        .any, .all => blk: {
            const outputs = try frame.allocator.alloc(bool, true_counts.items.len);
            defer frame.allocator.free(outputs);
            for (true_counts.items, false_counts.items, outputs) |true_count, false_count, *slot| {
                slot.* = switch (aggregation) {
                    .any => true_count != 0,
                    .all => false_count == 0,
                    else => unreachable,
                };
            }
            break :blk try DeviceColumn.fromSlice(bool, frame.allocator, outputs, frame.device);
        },
        .true_count => try DeviceColumn.fromSlice(i64, frame.allocator, true_counts.items, frame.device),
        .false_count => try DeviceColumn.fromSlice(i64, frame.allocator, false_counts.items, frame.device),
        .true_ratio, .false_ratio => blk: {
            const ratios = try frame.allocator.alloc(f64, true_counts.items.len);
            defer frame.allocator.free(ratios);
            for (true_counts.items, false_counts.items, ratios) |true_count, false_count, *slot| {
                const valid_count = true_count + false_count;
                if (valid_count == 0) {
                    slot.* = std.math.nan(f64);
                    continue;
                }
                const numerator = switch (aggregation) {
                    .true_ratio => true_count,
                    .false_ratio => false_count,
                    else => unreachable,
                };
                slot.* = @as(f64, @floatFromInt(numerator)) / @as(f64, @floatFromInt(valid_count));
            }
            break :blk try DeviceColumn.fromSlice(f64, frame.allocator, ratios, frame.device);
        },
    };
    return initMultiKeyAggregatedDataFrame(DeviceDataFrame, frame, key_names, representative_rows.items, output_name, output_column);
}

pub fn groupByAnyOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByBoolOn(DeviceDataFrame, .any, frame, key_names, value_name, output_name);
}

pub fn groupByAllOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByBoolOn(DeviceDataFrame, .all, frame, key_names, value_name, output_name);
}

pub fn groupByTrueCountOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByBoolOn(DeviceDataFrame, .true_count, frame, key_names, value_name, output_name);
}

pub fn groupByFalseCountOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByBoolOn(DeviceDataFrame, .false_count, frame, key_names, value_name, output_name);
}

pub fn groupByTrueRatioOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByBoolOn(DeviceDataFrame, .true_ratio, frame, key_names, value_name, output_name);
}

pub fn groupByFalseRatioOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByBoolOn(DeviceDataFrame, .false_ratio, frame, key_names, value_name, output_name);
}

fn groupByValidityCountOn(
    comptime DeviceDataFrame: type,
    aggregation: GroupByValidityAggregation,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);
    const value = try frame.column(value_name);

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(frame.allocator);
    var valid_counts: std.ArrayList(i64) = .empty;
    defer valid_counts.deinit(frame.allocator);
    var null_counts: std.ArrayList(i64) = .empty;
    defer null_counts.deinit(frame.allocator);

    for (0..frame.rows) |row| {
        if (!try rowHasValidKeys(frame.allocator, frame, key_names, row)) continue;
        const value_valid = try columnRowValid(frame.allocator, value.*, row);
        const group_index = (try findMultiKeyGroupIndex(frame.allocator, frame, key_names, representative_rows.items, row)) orelse blk: {
            try representative_rows.append(frame.allocator, row);
            try valid_counts.append(frame.allocator, 0);
            try null_counts.append(frame.allocator, 0);
            break :blk representative_rows.items.len - 1;
        };
        if (value_valid) {
            valid_counts.items[group_index] += 1;
        } else {
            null_counts.items[group_index] += 1;
        }
    }

    const output_column: DeviceColumn = switch (aggregation) {
        .valid_count => try DeviceColumn.fromSlice(i64, frame.allocator, valid_counts.items, frame.device),
        .null_count => try DeviceColumn.fromSlice(i64, frame.allocator, null_counts.items, frame.device),
        .valid_ratio, .null_ratio => blk: {
            const ratios = try frame.allocator.alloc(f64, valid_counts.items.len);
            defer frame.allocator.free(ratios);
            for (valid_counts.items, null_counts.items, ratios) |valid_count, null_count, *slot| {
                const total_count = valid_count + null_count;
                if (total_count == 0) {
                    slot.* = std.math.nan(f64);
                    continue;
                }
                const numerator = switch (aggregation) {
                    .valid_ratio => valid_count,
                    .null_ratio => null_count,
                    else => unreachable,
                };
                slot.* = @as(f64, @floatFromInt(numerator)) / @as(f64, @floatFromInt(total_count));
            }
            break :blk try DeviceColumn.fromSlice(f64, frame.allocator, ratios, frame.device);
        },
    };
    return initMultiKeyAggregatedDataFrame(DeviceDataFrame, frame, key_names, representative_rows.items, output_name, output_column);
}

pub fn groupByValidCountOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByValidityCountOn(DeviceDataFrame, .valid_count, frame, key_names, value_name, output_name);
}

pub fn groupByNullCountOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByValidityCountOn(DeviceDataFrame, .null_count, frame, key_names, value_name, output_name);
}

pub fn groupByValidRatioOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByValidityCountOn(DeviceDataFrame, .valid_ratio, frame, key_names, value_name, output_name);
}

pub fn groupByNullRatioOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByValidityCountOn(DeviceDataFrame, .null_ratio, frame, key_names, value_name, output_name);
}

pub fn groupByArgOnDispatchValue(
    comptime DeviceDataFrame: type,
    aggregation: GroupByArgAggregation,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
    value: DeviceColumn,
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    return switch (value) {
        .i8 => |typed| groupByArgOnTyped(DeviceDataFrame, i8, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .i16 => |typed| groupByArgOnTyped(DeviceDataFrame, i16, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .i32 => |typed| groupByArgOnTyped(DeviceDataFrame, i32, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .i64 => |typed| groupByArgOnTyped(DeviceDataFrame, i64, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u8 => |typed| groupByArgOnTyped(DeviceDataFrame, u8, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u16 => |typed| groupByArgOnTyped(DeviceDataFrame, u16, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u32 => |typed| groupByArgOnTyped(DeviceDataFrame, u32, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u64 => |typed| groupByArgOnTyped(DeviceDataFrame, u64, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .usize => |typed| groupByArgOnTyped(DeviceDataFrame, usize, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .isize => |typed| groupByArgOnTyped(DeviceDataFrame, isize, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .f16 => |typed| groupByArgOnTyped(DeviceDataFrame, f16, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .f32 => |typed| groupByArgOnTyped(DeviceDataFrame, f32, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .f64 => |typed| groupByArgOnTyped(DeviceDataFrame, f64, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn groupByArgOnTyped(
    comptime DeviceDataFrame: type,
    comptime V: type,
    aggregation: GroupByArgAggregation,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
    value: DeviceTypedColumn(V),
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    if (frame.rows != value.len()) return error.LengthMismatch;
    const values = try value.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_value_validity = try validityValues(value, allocator);
    defer if (maybe_value_validity) |validity| allocator.free(validity);

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(allocator);
    var best_rows: std.ArrayList(usize) = .empty;
    defer best_rows.deinit(allocator);

    for (values, 0..) |value_item, row| {
        if (maybe_value_validity) |validity| {
            if (!validity[row]) continue;
        }
        if (!try rowHasValidKeys(allocator, frame, key_names, row)) continue;
        const group_index = (try findMultiKeyGroupIndex(allocator, frame, key_names, representative_rows.items, row)) orelse blk: {
            try representative_rows.append(allocator, row);
            try best_rows.append(allocator, row);
            break :blk representative_rows.items.len - 1;
        };
        const best_row = best_rows.items[group_index];
        const better = switch (aggregation) {
            .argmin => compareSortValues(V, value_item, values[best_row]) < 0,
            .argmax => compareSortValues(V, value_item, values[best_row]) > 0,
        };
        if (better) best_rows.items[group_index] = row;
    }

    const out = try allocator.alloc(i64, best_rows.items.len);
    defer allocator.free(out);
    for (best_rows.items, out) |row, *slot| slot.* = @intCast(row);

    const output_column = try DeviceColumn.fromSlice(i64, allocator, out, device_value);
    return initMultiKeyAggregatedDataFrame(DeviceDataFrame, frame, key_names, representative_rows.items, output_name, output_column);
}

fn groupByArgOn(
    comptime DeviceDataFrame: type,
    aggregation: GroupByArgAggregation,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);
    const value = try frame.column(value_name);
    return groupByArgOnDispatchValue(DeviceDataFrame, aggregation, frame.allocator, frame, key_names, output_name, value.*, frame.device);
}

pub fn groupByArgMinOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByArgOn(DeviceDataFrame, .argmin, frame, key_names, value_name, output_name);
}

pub fn groupByArgMaxOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByArgOn(DeviceDataFrame, .argmax, frame, key_names, value_name, output_name);
}

fn initMultiKeyAggregatedDataFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    representative_rows: []const usize,
    output_name: []const u8,
    value_column: DeviceColumn,
) GroupByOnError!DeviceDataFrame {
    var owned_value = value_column;
    var value_moved = false;
    errdefer if (!value_moved) owned_value.deinit();
    if (owned_value.len() != representative_rows.len) return error.LengthMismatch;
    if (!owned_value.device().sameDevice(frame.device)) return error.InvalidDevice;

    const total_cols = key_names.len + 1;
    var names = try frame.allocator.alloc([]const u8, total_cols);
    defer frame.allocator.free(names);
    var columns = try frame.allocator.alloc(DeviceColumn, total_cols);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }

    for (key_names) |key_name| {
        names[initialized] = key_name;
        columns[initialized] = try (try frame.column(key_name)).take(representative_rows);
        initialized += 1;
    }
    names[initialized] = output_name;
    columns[initialized] = owned_value;
    value_moved = true;
    initialized += 1;
    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, names, columns, representative_rows.len, frame.device);
}

pub fn groupByNumericOnDispatchValue(
    comptime DeviceDataFrame: type,
    op: DeviceGroupByAggregation,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
    value: DeviceColumn,
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    return switch (value) {
        .i8 => |typed| groupByNumericOnTyped(DeviceDataFrame, op, i8, allocator, frame, key_names, output_name, typed, device_value),
        .i16 => |typed| groupByNumericOnTyped(DeviceDataFrame, op, i16, allocator, frame, key_names, output_name, typed, device_value),
        .i32 => |typed| groupByNumericOnTyped(DeviceDataFrame, op, i32, allocator, frame, key_names, output_name, typed, device_value),
        .i64 => |typed| groupByNumericOnTyped(DeviceDataFrame, op, i64, allocator, frame, key_names, output_name, typed, device_value),
        .u8 => |typed| groupByNumericOnTyped(DeviceDataFrame, op, u8, allocator, frame, key_names, output_name, typed, device_value),
        .u16 => |typed| groupByNumericOnTyped(DeviceDataFrame, op, u16, allocator, frame, key_names, output_name, typed, device_value),
        .u32 => |typed| groupByNumericOnTyped(DeviceDataFrame, op, u32, allocator, frame, key_names, output_name, typed, device_value),
        .u64 => |typed| groupByNumericOnTyped(DeviceDataFrame, op, u64, allocator, frame, key_names, output_name, typed, device_value),
        .usize => |typed| groupByNumericOnTyped(DeviceDataFrame, op, usize, allocator, frame, key_names, output_name, typed, device_value),
        .isize => |typed| groupByNumericOnTyped(DeviceDataFrame, op, isize, allocator, frame, key_names, output_name, typed, device_value),
        .f16 => |typed| groupByNumericOnTyped(DeviceDataFrame, op, f16, allocator, frame, key_names, output_name, typed, device_value),
        .f32 => |typed| groupByNumericOnTyped(DeviceDataFrame, op, f32, allocator, frame, key_names, output_name, typed, device_value),
        .f64 => |typed| groupByNumericOnTyped(DeviceDataFrame, op, f64, allocator, frame, key_names, output_name, typed, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn groupByNumericOnTyped(
    comptime DeviceDataFrame: type,
    op: DeviceGroupByAggregation,
    comptime V: type,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
    value: DeviceTypedColumn(V),
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    if (frame.rows != value.len()) return error.LengthMismatch;
    const values = try value.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_value_validity = try validityValues(value, allocator);
    defer if (maybe_value_validity) |validity| allocator.free(validity);

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(allocator);
    var aggregates: std.ArrayList(V) = .empty;
    defer aggregates.deinit(allocator);

    for (values, 0..) |value_item, row| {
        if (maybe_value_validity) |validity| {
            if (!validity[row]) continue;
        }
        if (!try rowHasValidKeys(allocator, frame, key_names, row)) continue;
        const maybe_group_index = try findMultiKeyGroupIndex(allocator, frame, key_names, representative_rows.items, row);
        if (maybe_group_index == null) {
            try representative_rows.append(allocator, row);
            try aggregates.append(allocator, value_item);
            continue;
        }
        const group_index = maybe_group_index.?;
        switch (op) {
            .sum => aggregates.items[group_index] += value_item,
            .prod => aggregates.items[group_index] *= value_item,
            .min => {
                if (compareSortValues(V, value_item, aggregates.items[group_index]) < 0) aggregates.items[group_index] = value_item;
            },
            .max => {
                if (compareSortValues(V, value_item, aggregates.items[group_index]) > 0) aggregates.items[group_index] = value_item;
            },
        }
    }

    const aggregate_column = try DeviceColumn.fromSlice(V, allocator, aggregates.items, device_value);
    return initMultiKeyAggregatedDataFrame(DeviceDataFrame, frame, key_names, representative_rows.items, output_name, aggregate_column);
}

pub fn groupByMeanOnDispatchValue(
    comptime DeviceDataFrame: type,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
    value: DeviceColumn,
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    return switch (value) {
        .i8 => |typed| groupByMeanOnTyped(DeviceDataFrame, i8, allocator, frame, key_names, output_name, typed, device_value),
        .i16 => |typed| groupByMeanOnTyped(DeviceDataFrame, i16, allocator, frame, key_names, output_name, typed, device_value),
        .i32 => |typed| groupByMeanOnTyped(DeviceDataFrame, i32, allocator, frame, key_names, output_name, typed, device_value),
        .i64 => |typed| groupByMeanOnTyped(DeviceDataFrame, i64, allocator, frame, key_names, output_name, typed, device_value),
        .u8 => |typed| groupByMeanOnTyped(DeviceDataFrame, u8, allocator, frame, key_names, output_name, typed, device_value),
        .u16 => |typed| groupByMeanOnTyped(DeviceDataFrame, u16, allocator, frame, key_names, output_name, typed, device_value),
        .u32 => |typed| groupByMeanOnTyped(DeviceDataFrame, u32, allocator, frame, key_names, output_name, typed, device_value),
        .u64 => |typed| groupByMeanOnTyped(DeviceDataFrame, u64, allocator, frame, key_names, output_name, typed, device_value),
        .usize => |typed| groupByMeanOnTyped(DeviceDataFrame, usize, allocator, frame, key_names, output_name, typed, device_value),
        .isize => |typed| groupByMeanOnTyped(DeviceDataFrame, isize, allocator, frame, key_names, output_name, typed, device_value),
        .f16 => |typed| groupByMeanOnTyped(DeviceDataFrame, f16, allocator, frame, key_names, output_name, typed, device_value),
        .f32 => |typed| groupByMeanOnTyped(DeviceDataFrame, f32, allocator, frame, key_names, output_name, typed, device_value),
        .f64 => |typed| groupByMeanOnTyped(DeviceDataFrame, f64, allocator, frame, key_names, output_name, typed, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn groupByMeanOnTyped(
    comptime DeviceDataFrame: type,
    comptime V: type,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
    value: DeviceTypedColumn(V),
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    if (frame.rows != value.len()) return error.LengthMismatch;
    const values = try value.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_value_validity = try validityValues(value, allocator);
    defer if (maybe_value_validity) |validity| allocator.free(validity);

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(allocator);
    var sums: std.ArrayList(f64) = .empty;
    defer sums.deinit(allocator);
    var counts: std.ArrayList(i64) = .empty;
    defer counts.deinit(allocator);

    for (values, 0..) |value_item, row| {
        if (maybe_value_validity) |validity| {
            if (!validity[row]) continue;
        }
        if (!try rowHasValidKeys(allocator, frame, key_names, row)) continue;
        const group_index = (try findMultiKeyGroupIndex(allocator, frame, key_names, representative_rows.items, row)) orelse blk: {
            try representative_rows.append(allocator, row);
            try sums.append(allocator, 0);
            try counts.append(allocator, 0);
            break :blk representative_rows.items.len - 1;
        };
        sums.items[group_index] += castToF64(V, value_item);
        counts.items[group_index] += 1;
    }

    const means = try allocator.alloc(f64, sums.items.len);
    defer allocator.free(means);
    for (sums.items, counts.items, means) |sum_value, count, *slot| {
        slot.* = sum_value / @as(f64, @floatFromInt(count));
    }

    const mean_column = try DeviceColumn.fromSlice(f64, allocator, means, device_value);
    return initMultiKeyAggregatedDataFrame(DeviceDataFrame, frame, key_names, representative_rows.items, output_name, mean_column);
}

pub fn groupByProfileOnDispatchValue(
    comptime DeviceDataFrame: type,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_prefix: []const u8,
    value: DeviceColumn,
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    return switch (value) {
        .i8 => |typed| groupByProfileOnTyped(DeviceDataFrame, i8, allocator, frame, key_names, output_prefix, typed, device_value),
        .i16 => |typed| groupByProfileOnTyped(DeviceDataFrame, i16, allocator, frame, key_names, output_prefix, typed, device_value),
        .i32 => |typed| groupByProfileOnTyped(DeviceDataFrame, i32, allocator, frame, key_names, output_prefix, typed, device_value),
        .i64 => |typed| groupByProfileOnTyped(DeviceDataFrame, i64, allocator, frame, key_names, output_prefix, typed, device_value),
        .u8 => |typed| groupByProfileOnTyped(DeviceDataFrame, u8, allocator, frame, key_names, output_prefix, typed, device_value),
        .u16 => |typed| groupByProfileOnTyped(DeviceDataFrame, u16, allocator, frame, key_names, output_prefix, typed, device_value),
        .u32 => |typed| groupByProfileOnTyped(DeviceDataFrame, u32, allocator, frame, key_names, output_prefix, typed, device_value),
        .u64 => |typed| groupByProfileOnTyped(DeviceDataFrame, u64, allocator, frame, key_names, output_prefix, typed, device_value),
        .usize => |typed| groupByProfileOnTyped(DeviceDataFrame, usize, allocator, frame, key_names, output_prefix, typed, device_value),
        .isize => |typed| groupByProfileOnTyped(DeviceDataFrame, isize, allocator, frame, key_names, output_prefix, typed, device_value),
        .f16 => |typed| groupByProfileOnTyped(DeviceDataFrame, f16, allocator, frame, key_names, output_prefix, typed, device_value),
        .f32 => |typed| groupByProfileOnTyped(DeviceDataFrame, f32, allocator, frame, key_names, output_prefix, typed, device_value),
        .f64 => |typed| groupByProfileOnTyped(DeviceDataFrame, f64, allocator, frame, key_names, output_prefix, typed, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn groupByProfileOnTyped(
    comptime DeviceDataFrame: type,
    comptime V: type,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_prefix: []const u8,
    value: DeviceTypedColumn(V),
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    if (frame.rows != value.len()) return error.LengthMismatch;
    const values = try value.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_value_validity = try validityValues(value, allocator);
    defer if (maybe_value_validity) |validity| allocator.free(validity);

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(allocator);
    var profiles: std.ArrayList(MomentProfile) = .empty;
    defer profiles.deinit(allocator);

    for (values, 0..) |value_item, row| {
        if (maybe_value_validity) |validity| {
            if (!validity[row]) continue;
        }
        if (!try rowHasValidKeys(allocator, frame, key_names, row)) continue;
        const maybe_group_index = try findMultiKeyGroupIndex(allocator, frame, key_names, representative_rows.items, row);
        const group_index = maybe_group_index orelse blk: {
            try representative_rows.append(allocator, row);
            try profiles.append(allocator, .{});
            break :blk representative_rows.items.len - 1;
        };
        profiles.items[group_index].update(castToF64(V, value_item));
    }

    var metrics = try group_profile_mod.materializeMetrics(allocator, profiles.items);
    defer metrics.deinit();
    var key_columns = try allocator.alloc(DeviceColumn, key_names.len);
    var initialized: usize = 0;
    defer {
        for (key_columns[0..initialized]) |*col| col.deinit();
        allocator.free(key_columns);
    }
    for (key_names, key_columns) |key_name, *slot| {
        slot.* = try (try frame.column(key_name)).take(representative_rows.items);
        initialized += 1;
    }

    return group_profile_mod.initProfileDataFrame(DeviceDataFrame, allocator, key_names, output_prefix, key_columns, metrics, device_value);
}

pub fn groupByStatsOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_prefix: []const u8,
) GroupByOnError!DeviceDataFrame {
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);
    const value = try frame.column(value_name);
    return groupByStatsOnDispatchValue(DeviceDataFrame, frame.allocator, frame, key_names, output_prefix, value.*, frame.device);
}

pub fn groupByProfileOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_prefix: []const u8,
) GroupByOnError!DeviceDataFrame {
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);
    const value = try frame.column(value_name);
    return groupByProfileOnDispatchValue(DeviceDataFrame, frame.allocator, frame, key_names, output_prefix, value.*, frame.device);
}
