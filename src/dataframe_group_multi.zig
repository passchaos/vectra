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
const rank_mod = @import("dataframe_rank.zig");
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

const GroupByBoolIndexAggregation = enum {
    first_true_index,
    last_true_index,
    first_false_index,
    last_false_index,
};

const GroupByValidityAggregation = enum {
    any_valid,
    all_valid,
    any_null,
    all_null,
    valid_count,
    null_count,
    valid_ratio,
    null_ratio,
};

const GroupByValidityIndexAggregation = enum {
    first_valid_index,
    last_valid_index,
    first_null_index,
    last_null_index,
};

const GroupByNumericQualityAggregation = enum {
    nan_count,
    nan_ratio,
    inf_count,
    inf_ratio,
    positive_inf_count,
    positive_inf_ratio,
    negative_inf_count,
    negative_inf_ratio,
    finite_count,
    finite_ratio,
    normal_count,
    normal_ratio,
    subnormal_count,
    subnormal_ratio,
    non_finite_count,
    non_finite_ratio,
    zero_count,
    zero_ratio,
    positive_zero_count,
    positive_zero_ratio,
    negative_zero_count,
    negative_zero_ratio,
    non_zero_count,
    non_zero_ratio,
    positive_count,
    positive_ratio,
    signbit_count,
    signbit_ratio,
    negative_count,
    negative_ratio,
};

const GroupByNumericQualityIndexAggregation = enum {
    first_nan_index,
    last_nan_index,
    first_inf_index,
    last_inf_index,
    first_positive_inf_index,
    last_positive_inf_index,
    first_negative_inf_index,
    last_negative_inf_index,
    first_finite_index,
    last_finite_index,
    first_normal_index,
    last_normal_index,
    first_subnormal_index,
    last_subnormal_index,
    first_non_finite_index,
    last_non_finite_index,
    first_zero_index,
    last_zero_index,
    first_positive_zero_index,
    last_positive_zero_index,
    first_negative_zero_index,
    last_negative_zero_index,
    first_non_zero_index,
    last_non_zero_index,
    first_positive_index,
    last_positive_index,
    first_signbit_index,
    last_signbit_index,
    first_negative_index,
    last_negative_index,
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
    weighted_mode,
    weighted_mode_weight,
    weighted_mode_ratio,
    weighted_mode_margin,
    weighted_mode_margin_ratio,
    weighted_entropy,
    weighted_gini_impurity,
    weighted_perplexity,
    weighted_inverse_simpson,
    weighted_simpson_concentration,
    weighted_evenness,
};

const GroupByPairAggregation = enum {
    dot,
    cosine_similarity,
    squared_euclidean_distance,
    euclidean_distance,
    manhattan_distance,
    chebyshev_distance,
    canberra_distance,
    bray_curtis_distance,
    mean_error,
    mae,
    mse,
    rmse,
    mape,
    smape,
    covariance,
    correlation,
    beta,
};

const GroupByWeightedPairAggregation = enum {
    weighted_dot,
    weighted_cosine_similarity,
    weighted_squared_euclidean_distance,
    weighted_euclidean_distance,
    weighted_manhattan_distance,
    weighted_chebyshev_distance,
    weighted_canberra_distance,
    weighted_bray_curtis_distance,
    weighted_mean_error,
    weighted_mae,
    weighted_mse,
    weighted_rmse,
    weighted_mape,
    weighted_smape,
    weighted_covariance,
    weighted_correlation,
    weighted_beta,
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

fn groupColumnValidityValues(allocator: std.mem.Allocator, column: DeviceColumn) GroupByOnError!?[]bool {
    return switch (column) {
        inline else => |typed| try validityValues(typed, allocator),
    };
}

fn groupNumericQualityIsRatio(aggregation: GroupByNumericQualityAggregation) bool {
    return switch (aggregation) {
        .nan_ratio,
        .inf_ratio,
        .positive_inf_ratio,
        .negative_inf_ratio,
        .finite_ratio,
        .normal_ratio,
        .subnormal_ratio,
        .non_finite_ratio,
        .zero_ratio,
        .positive_zero_ratio,
        .negative_zero_ratio,
        .non_zero_ratio,
        .positive_ratio,
        .signbit_ratio,
        .negative_ratio,
        => true,
        else => false,
    };
}

fn groupNumericQualityMatchesTyped(comptime T: type, value: T, aggregation: GroupByNumericQualityAggregation) bool {
    return switch (@typeInfo(T)) {
        .float, .comptime_float => switch (aggregation) {
            .nan_count, .nan_ratio => std.math.isNan(value),
            .inf_count, .inf_ratio => std.math.isInf(value),
            .positive_inf_count, .positive_inf_ratio => std.math.isPositiveInf(value),
            .negative_inf_count, .negative_inf_ratio => std.math.isNegativeInf(value),
            .finite_count, .finite_ratio => std.math.isFinite(value),
            .normal_count, .normal_ratio => std.math.isNormal(value),
            .subnormal_count, .subnormal_ratio => std.math.isFinite(value) and value != 0.0 and !std.math.isNormal(value),
            .non_finite_count, .non_finite_ratio => !std.math.isFinite(value),
            .zero_count, .zero_ratio => value == 0.0,
            .positive_zero_count, .positive_zero_ratio => value == 0.0 and !std.math.signbit(value),
            .negative_zero_count, .negative_zero_ratio => value == 0.0 and std.math.signbit(value),
            .non_zero_count, .non_zero_ratio => value != 0.0,
            .positive_count, .positive_ratio => value > 0.0,
            .signbit_count, .signbit_ratio => std.math.signbit(value),
            .negative_count, .negative_ratio => value < 0.0,
        },
        .int => |info| switch (aggregation) {
            .finite_count, .finite_ratio => true,
            .zero_count, .zero_ratio => value == 0,
            .non_zero_count, .non_zero_ratio => value != 0,
            .positive_count, .positive_ratio => value > 0,
            .signbit_count, .signbit_ratio => info.signedness == .signed and value < 0,
            .negative_count, .negative_ratio => info.signedness == .signed and value < 0,
            else => false,
        },
        .comptime_int => switch (aggregation) {
            .finite_count, .finite_ratio => true,
            .zero_count, .zero_ratio => value == 0,
            .non_zero_count, .non_zero_ratio => value != 0,
            .positive_count, .positive_ratio => value > 0,
            .signbit_count, .signbit_ratio => value < 0,
            .negative_count, .negative_ratio => value < 0,
            else => false,
        },
        else => false,
    };
}

fn groupNumericQualityIndexPredicate(aggregation: GroupByNumericQualityIndexAggregation) GroupByNumericQualityAggregation {
    return switch (aggregation) {
        .first_nan_index, .last_nan_index => .nan_count,
        .first_inf_index, .last_inf_index => .inf_count,
        .first_positive_inf_index, .last_positive_inf_index => .positive_inf_count,
        .first_negative_inf_index, .last_negative_inf_index => .negative_inf_count,
        .first_finite_index, .last_finite_index => .finite_count,
        .first_normal_index, .last_normal_index => .normal_count,
        .first_subnormal_index, .last_subnormal_index => .subnormal_count,
        .first_non_finite_index, .last_non_finite_index => .non_finite_count,
        .first_zero_index, .last_zero_index => .zero_count,
        .first_positive_zero_index, .last_positive_zero_index => .positive_zero_count,
        .first_negative_zero_index, .last_negative_zero_index => .negative_zero_count,
        .first_non_zero_index, .last_non_zero_index => .non_zero_count,
        .first_positive_index, .last_positive_index => .positive_count,
        .first_signbit_index, .last_signbit_index => .signbit_count,
        .first_negative_index, .last_negative_index => .negative_count,
    };
}

fn groupNumericQualityIndexKeepsLast(aggregation: GroupByNumericQualityIndexAggregation) bool {
    return switch (aggregation) {
        .first_nan_index,
        .first_inf_index,
        .first_positive_inf_index,
        .first_negative_inf_index,
        .first_finite_index,
        .first_normal_index,
        .first_subnormal_index,
        .first_non_finite_index,
        .first_zero_index,
        .first_positive_zero_index,
        .first_negative_zero_index,
        .first_non_zero_index,
        .first_positive_index,
        .first_signbit_index,
        .first_negative_index,
        => false,
        .last_nan_index,
        .last_inf_index,
        .last_positive_inf_index,
        .last_negative_inf_index,
        .last_finite_index,
        .last_normal_index,
        .last_subnormal_index,
        .last_non_finite_index,
        .last_zero_index,
        .last_positive_zero_index,
        .last_negative_zero_index,
        .last_non_zero_index,
        .last_positive_index,
        .last_signbit_index,
        .last_negative_index,
        => true,
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

const GroupWeightedModeStats = struct {
    mode: f64,
    mode_weight: f64,
    second_weight: f64,
    total_weight: f64,
    distinct_positive_weight_count: usize,
    entropy: f64,
    sum_probability_sq: f64,
};

fn groupWeightedValueLess(_: void, lhs: GroupWeightedValue, rhs: GroupWeightedValue) bool {
    return groupByQuantileLess({}, lhs.value, rhs.value);
}

fn groupWeightedValueEqual(lhs: f64, rhs: f64) bool {
    return (std.math.isNan(lhs) and std.math.isNan(rhs)) or lhs == rhs;
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

fn groupWeightedModeStats(rows: []const usize, values: []const f64, weights: []const f64) GroupWeightedModeStats {
    var total_weight: f64 = 0.0;
    for (rows) |row| total_weight += weights[row];
    if (rows.len == 0 or !(total_weight > 0.0)) {
        return .{
            .mode = std.math.nan(f64),
            .mode_weight = 0.0,
            .second_weight = 0.0,
            .total_weight = total_weight,
            .distinct_positive_weight_count = 0,
            .entropy = std.math.nan(f64),
            .sum_probability_sq = std.math.nan(f64),
        };
    }

    var found = false;
    var best_value: f64 = 0.0;
    var best_weight: f64 = 0.0;
    var second_weight: f64 = 0.0;
    var entropy: f64 = 0.0;
    var sum_probability_sq: f64 = 0.0;
    var distinct_positive_weight_count: usize = 0;

    for (rows, 0..) |candidate_row, candidate_index| {
        const candidate = values[candidate_row];
        var seen = false;
        for (rows[0..candidate_index]) |previous_row| {
            if (groupWeightedValueEqual(values[previous_row], candidate)) {
                seen = true;
                break;
            }
        }
        if (seen) continue;

        var candidate_weight: f64 = 0.0;
        for (rows[candidate_index..]) |match_row| {
            if (groupWeightedValueEqual(candidate, values[match_row])) candidate_weight += weights[match_row];
        }

        // Preserve unweighted grouped mode's stable first-tie semantics: equal
        // winning weights leave the first distinct group value as the mode while
        // still counting the tie as the second-best weight for margin metrics.
        if (!found or candidate_weight > best_weight) {
            second_weight = best_weight;
            best_weight = candidate_weight;
            best_value = candidate;
            found = true;
        } else if (candidate_weight > second_weight) {
            second_weight = candidate_weight;
        }

        if (candidate_weight > 0.0) {
            const probability = candidate_weight / total_weight;
            entropy -= probability * std.math.log(f64, std.math.e, probability);
            sum_probability_sq += probability * probability;
            distinct_positive_weight_count += 1;
        }
    }

    return .{
        .mode = best_value,
        .mode_weight = best_weight,
        .second_weight = second_weight,
        .total_weight = total_weight,
        .distinct_positive_weight_count = distinct_positive_weight_count,
        .entropy = entropy,
        .sum_probability_sq = sum_probability_sq,
    };
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

fn groupByLimitRowsOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    n: usize,
    comptime keep_tail: bool,
) GroupByOnError!DeviceDataFrame {
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);
    if (n == 0) return dataframe_array_mod.takeRows(DeviceDataFrame, frame, &.{});

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(frame.allocator);
    var groups: std.ArrayList(std.ArrayList(usize)) = .empty;
    defer {
        for (groups.items) |*group| group.deinit(frame.allocator);
        groups.deinit(frame.allocator);
    }

    for (0..frame.rows) |row| {
        if (!try rowHasValidKeys(frame.allocator, frame, key_names, row)) continue;
        const group_index = (try findMultiKeyGroupIndex(frame.allocator, frame, key_names, representative_rows.items, row)) orelse blk: {
            try representative_rows.append(frame.allocator, row);
            var rows: std.ArrayList(usize) = .empty;
            errdefer rows.deinit(frame.allocator);
            try groups.append(frame.allocator, rows);
            break :blk groups.items.len - 1;
        };
        if (keep_tail) {
            const group = &groups.items[group_index];
            if (group.items.len == n) _ = group.orderedRemove(0);
            try group.append(frame.allocator, row);
        } else if (groups.items[group_index].items.len < n) {
            try groups.items[group_index].append(frame.allocator, row);
        }
    }

    var row_indices: std.ArrayList(usize) = .empty;
    defer row_indices.deinit(frame.allocator);
    for (groups.items) |group| try row_indices.appendSlice(frame.allocator, group.items);
    return dataframe_array_mod.takeRows(DeviceDataFrame, frame, row_indices.items);
}

fn RowSortContext(comptime T: type) type {
    return struct {
        values: []const T,
        validity: ?[]const bool,
        options: options_mod.DeviceSortOptions,
    };
}

fn rowLessByTyped(comptime T: type, context: RowSortContext(T), lhs: usize, rhs: usize) bool {
    const lhs_valid = if (context.validity) |validity| validity[lhs] else true;
    const rhs_valid = if (context.validity) |validity| validity[rhs] else true;
    if (!lhs_valid or !rhs_valid) {
        if (lhs_valid == rhs_valid) return lhs < rhs;
        const null_first = context.options.nulls == .first;
        return if (!lhs_valid and rhs_valid) null_first else !null_first;
    }
    const cmp = compareSortValues(T, context.values[lhs], context.values[rhs]);
    if (cmp == 0) return lhs < rhs;
    return if (context.options.descending) cmp > 0 else cmp < 0;
}

fn groupByTopRowsOnTyped(
    comptime DeviceDataFrame: type,
    comptime T: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    n: usize,
    options: options_mod.DeviceSortOptions,
    sort_column: DeviceTypedColumn(T),
) GroupByOnError!DeviceDataFrame {
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);
    if (frame.rows != sort_column.len()) return error.LengthMismatch;
    if (n == 0) return dataframe_array_mod.takeRows(DeviceDataFrame, frame, &.{});

    const sort_values = try sort_column.values.toOwnedSlice(frame.allocator);
    defer frame.allocator.free(sort_values);
    const maybe_validity = try validityValues(sort_column, frame.allocator);
    defer if (maybe_validity) |validity| frame.allocator.free(validity);

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(frame.allocator);
    var groups: std.ArrayList(std.ArrayList(usize)) = .empty;
    defer {
        for (groups.items) |*group| group.deinit(frame.allocator);
        groups.deinit(frame.allocator);
    }

    for (0..frame.rows) |row| {
        if (!try rowHasValidKeys(frame.allocator, frame, key_names, row)) continue;
        const group_index = (try findMultiKeyGroupIndex(frame.allocator, frame, key_names, representative_rows.items, row)) orelse blk: {
            try representative_rows.append(frame.allocator, row);
            var rows: std.ArrayList(usize) = .empty;
            errdefer rows.deinit(frame.allocator);
            try groups.append(frame.allocator, rows);
            break :blk groups.items.len - 1;
        };
        try groups.items[group_index].append(frame.allocator, row);
    }

    const context: RowSortContext(T) = .{
        .values = sort_values,
        .validity = maybe_validity,
        .options = options,
    };
    var row_indices: std.ArrayList(usize) = .empty;
    defer row_indices.deinit(frame.allocator);
    for (groups.items) |*group| {
        std.mem.sort(usize, group.items, context, struct {
            fn less(ctx: RowSortContext(T), lhs: usize, rhs: usize) bool {
                return rowLessByTyped(T, ctx, lhs, rhs);
            }
        }.less);
        try row_indices.appendSlice(frame.allocator, group.items[0..@min(n, group.items.len)]);
    }
    return dataframe_array_mod.takeRows(DeviceDataFrame, frame, row_indices.items);
}

fn groupByTopRowsDispatchSortColumn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    sort_name: []const u8,
    n: usize,
    options: options_mod.DeviceSortOptions,
) GroupByOnError!DeviceDataFrame {
    const sort_column = try frame.column(sort_name);
    return switch (sort_column.*) {
        .bool => |typed| groupByTopRowsOnTyped(DeviceDataFrame, bool, frame, key_names, n, options, typed),
        .i8 => |typed| groupByTopRowsOnTyped(DeviceDataFrame, i8, frame, key_names, n, options, typed),
        .i16 => |typed| groupByTopRowsOnTyped(DeviceDataFrame, i16, frame, key_names, n, options, typed),
        .i32 => |typed| groupByTopRowsOnTyped(DeviceDataFrame, i32, frame, key_names, n, options, typed),
        .i64 => |typed| groupByTopRowsOnTyped(DeviceDataFrame, i64, frame, key_names, n, options, typed),
        .u8 => |typed| groupByTopRowsOnTyped(DeviceDataFrame, u8, frame, key_names, n, options, typed),
        .u16 => |typed| groupByTopRowsOnTyped(DeviceDataFrame, u16, frame, key_names, n, options, typed),
        .u32 => |typed| groupByTopRowsOnTyped(DeviceDataFrame, u32, frame, key_names, n, options, typed),
        .u64 => |typed| groupByTopRowsOnTyped(DeviceDataFrame, u64, frame, key_names, n, options, typed),
        .usize => |typed| groupByTopRowsOnTyped(DeviceDataFrame, usize, frame, key_names, n, options, typed),
        .isize => |typed| groupByTopRowsOnTyped(DeviceDataFrame, isize, frame, key_names, n, options, typed),
        .f16 => |typed| groupByTopRowsOnTyped(DeviceDataFrame, f16, frame, key_names, n, options, typed),
        .f32 => |typed| groupByTopRowsOnTyped(DeviceDataFrame, f32, frame, key_names, n, options, typed),
        .f64 => |typed| groupByTopRowsOnTyped(DeviceDataFrame, f64, frame, key_names, n, options, typed),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

pub fn groupByHeadRowsOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    n: usize,
) GroupByOnError!DeviceDataFrame {
    return groupByLimitRowsOn(DeviceDataFrame, frame, key_names, n, false);
}

pub fn groupByTailRowsOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    n: usize,
) GroupByOnError!DeviceDataFrame {
    return groupByLimitRowsOn(DeviceDataFrame, frame, key_names, n, true);
}

pub fn groupByTopRowsOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    sort_name: []const u8,
    n: usize,
    options: options_mod.DeviceSortOptions,
) GroupByOnError!DeviceDataFrame {
    return groupByTopRowsDispatchSortColumn(DeviceDataFrame, frame, key_names, sort_name, n, options);
}

pub fn groupByBottomRowsOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    sort_name: []const u8,
    n: usize,
    options: options_mod.DeviceSortOptions,
) GroupByOnError!DeviceDataFrame {
    var bottom_options = options;
    bottom_options.descending = !bottom_options.descending;
    return groupByTopRowsDispatchSortColumn(DeviceDataFrame, frame, key_names, sort_name, n, bottom_options);
}

fn groupByTopRowsByColumnsCoreOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    sort_names: []const []const u8,
    n: usize,
    options_values: []const options_mod.DeviceSortOptions,
) GroupByOnError!DeviceDataFrame {
    if (key_names.len == 0 or sort_names.len == 0) return error.LengthMismatch;
    if (sort_names.len != options_values.len) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);
    for (sort_names) |sort_name| _ = try frame.column(sort_name);
    if (n == 0) return dataframe_array_mod.takeRows(DeviceDataFrame, frame, &.{});

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(frame.allocator);
    var groups: std.ArrayList(std.ArrayList(usize)) = .empty;
    defer {
        for (groups.items) |*group| group.deinit(frame.allocator);
        groups.deinit(frame.allocator);
    }

    for (0..frame.rows) |row| {
        if (!try rowHasValidKeys(frame.allocator, frame, key_names, row)) continue;
        const group_index = (try findMultiKeyGroupIndex(frame.allocator, frame, key_names, representative_rows.items, row)) orelse blk: {
            try representative_rows.append(frame.allocator, row);
            var rows: std.ArrayList(usize) = .empty;
            errdefer rows.deinit(frame.allocator);
            try groups.append(frame.allocator, rows);
            break :blk groups.items.len - 1;
        };
        try groups.items[group_index].append(frame.allocator, row);
    }

    var row_indices: std.ArrayList(usize) = .empty;
    defer row_indices.deinit(frame.allocator);
    for (groups.items) |group| {
        var group_frame = try dataframe_array_mod.takeRows(DeviceDataFrame, frame, group.items);
        defer group_frame.deinit();
        const local_order = try rank_mod.argsortByColumns(group_frame, sort_names, options_values);
        defer frame.allocator.free(local_order);
        for (local_order[0..@min(n, local_order.len)]) |local_row| {
            try row_indices.append(frame.allocator, group.items[local_row]);
        }
    }
    return dataframe_array_mod.takeRows(DeviceDataFrame, frame, row_indices.items);
}

pub fn groupByTopRowsByColumnsOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    sort_names: []const []const u8,
    n: usize,
    options_values: []const options_mod.DeviceSortOptions,
) GroupByOnError!DeviceDataFrame {
    return groupByTopRowsByColumnsCoreOn(DeviceDataFrame, frame, key_names, sort_names, n, options_values);
}

pub fn groupByBottomRowsByColumnsOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    sort_names: []const []const u8,
    n: usize,
    options_values: []const options_mod.DeviceSortOptions,
) GroupByOnError!DeviceDataFrame {
    const bottom_options = try frame.allocator.dupe(options_mod.DeviceSortOptions, options_values);
    defer frame.allocator.free(bottom_options);
    for (bottom_options) |*option| option.descending = !option.descending;
    return groupByTopRowsByColumnsCoreOn(DeviceDataFrame, frame, key_names, sort_names, n, bottom_options);
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

fn groupByRowValueOn(
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

fn groupByNthValueOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
    n: usize,
    comptime skip_nulls: bool,
) GroupByOnError!DeviceDataFrame {
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);
    const value = try frame.column(value_name);

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(frame.allocator);
    var value_rows: std.ArrayList(?usize) = .empty;
    defer value_rows.deinit(frame.allocator);
    var seen_counts: std.ArrayList(usize) = .empty;
    defer seen_counts.deinit(frame.allocator);
    var found_values: std.ArrayList(bool) = .empty;
    defer found_values.deinit(frame.allocator);

    for (0..frame.rows) |row| {
        if (!try rowHasValidKeys(frame.allocator, frame, key_names, row)) continue;
        if (skip_nulls and !try columnRowValid(frame.allocator, value.*, row)) continue;
        const group_index = (try findMultiKeyGroupIndex(frame.allocator, frame, key_names, representative_rows.items, row)) orelse blk: {
            try representative_rows.append(frame.allocator, row);
            try value_rows.append(frame.allocator, null);
            try seen_counts.append(frame.allocator, 0);
            try found_values.append(frame.allocator, false);
            break :blk representative_rows.items.len - 1;
        };
        if (!found_values.items[group_index] and seen_counts.items[group_index] == n) {
            value_rows.items[group_index] = row;
            found_values.items[group_index] = true;
        }
        seen_counts.items[group_index] += 1;
    }

    const value_column = try value.takeOptional(value_rows.items);
    return initMultiKeyAggregatedDataFrame(DeviceDataFrame, frame, key_names, representative_rows.items, output_name, value_column);
}

fn groupByNthIndexCoreOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
    n: usize,
    comptime skip_nulls: bool,
) GroupByOnError!DeviceDataFrame {
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);
    const value = try frame.column(value_name);

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(frame.allocator);
    var index_values: std.ArrayList(i64) = .empty;
    defer index_values.deinit(frame.allocator);
    var output_validity: std.ArrayList(bool) = .empty;
    defer output_validity.deinit(frame.allocator);
    var seen_counts: std.ArrayList(usize) = .empty;
    defer seen_counts.deinit(frame.allocator);
    var found_values: std.ArrayList(bool) = .empty;
    defer found_values.deinit(frame.allocator);

    for (0..frame.rows) |row| {
        if (!try rowHasValidKeys(frame.allocator, frame, key_names, row)) continue;
        if (skip_nulls and !try columnRowValid(frame.allocator, value.*, row)) continue;
        const group_index = (try findMultiKeyGroupIndex(frame.allocator, frame, key_names, representative_rows.items, row)) orelse blk: {
            try representative_rows.append(frame.allocator, row);
            try index_values.append(frame.allocator, 0);
            try output_validity.append(frame.allocator, false);
            try seen_counts.append(frame.allocator, 0);
            try found_values.append(frame.allocator, false);
            break :blk representative_rows.items.len - 1;
        };
        if (!found_values.items[group_index] and seen_counts.items[group_index] == n) {
            index_values.items[group_index] = @intCast(row);
            output_validity.items[group_index] = true;
            found_values.items[group_index] = true;
        }
        seen_counts.items[group_index] += 1;
    }

    const output_column = try DeviceColumn.fromSliceWithValidity(i64, frame.allocator, index_values.items, output_validity.items, frame.device);
    return initMultiKeyAggregatedDataFrame(DeviceDataFrame, frame, key_names, representative_rows.items, output_name, output_column);
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

pub fn groupByFirstRowOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRowValueOn(DeviceDataFrame, frame, key_names, value_name, output_name, false);
}

pub fn groupByLastRowOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByRowValueOn(DeviceDataFrame, frame, key_names, value_name, output_name, true);
}

pub fn groupByNthOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
    n: usize,
) GroupByOnError!DeviceDataFrame {
    return groupByNthValueOn(DeviceDataFrame, frame, key_names, value_name, output_name, n, true);
}

pub fn groupByNthRowOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
    n: usize,
) GroupByOnError!DeviceDataFrame {
    return groupByNthValueOn(DeviceDataFrame, frame, key_names, value_name, output_name, n, false);
}

pub fn groupByNthIndexOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
    n: usize,
) GroupByOnError!DeviceDataFrame {
    return groupByNthIndexCoreOn(DeviceDataFrame, frame, key_names, value_name, output_name, n, true);
}

pub fn groupByNthRowIndexOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
    n: usize,
) GroupByOnError!DeviceDataFrame {
    return groupByNthIndexCoreOn(DeviceDataFrame, frame, key_names, value_name, output_name, n, false);
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
        if (!(weight_sum > 0.0)) {
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
            .weighted_mode => groupWeightedModeStats(rows.items, values.values, weights.values).mode,
            .weighted_mode_weight => groupWeightedModeStats(rows.items, values.values, weights.values).mode_weight,
            .weighted_mode_ratio => blk: {
                const stats = groupWeightedModeStats(rows.items, values.values, weights.values);
                break :blk stats.mode_weight / stats.total_weight;
            },
            .weighted_mode_margin => blk: {
                const stats = groupWeightedModeStats(rows.items, values.values, weights.values);
                break :blk stats.mode_weight - stats.second_weight;
            },
            .weighted_mode_margin_ratio => blk: {
                const stats = groupWeightedModeStats(rows.items, values.values, weights.values);
                break :blk (stats.mode_weight - stats.second_weight) / stats.total_weight;
            },
            .weighted_entropy => groupWeightedModeStats(rows.items, values.values, weights.values).entropy,
            .weighted_gini_impurity => 1.0 - groupWeightedModeStats(rows.items, values.values, weights.values).sum_probability_sq,
            .weighted_perplexity => std.math.exp(groupWeightedModeStats(rows.items, values.values, weights.values).entropy),
            .weighted_inverse_simpson => blk: {
                const concentration = groupWeightedModeStats(rows.items, values.values, weights.values).sum_probability_sq;
                break :blk if (concentration == 0.0) std.math.nan(f64) else 1.0 / concentration;
            },
            .weighted_simpson_concentration => groupWeightedModeStats(rows.items, values.values, weights.values).sum_probability_sq,
            .weighted_evenness => blk: {
                const stats = groupWeightedModeStats(rows.items, values.values, weights.values);
                break :blk if (stats.distinct_positive_weight_count <= 1) 1.0 else stats.entropy / std.math.log(f64, std.math.e, @as(f64, @floatFromInt(stats.distinct_positive_weight_count)));
            },
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

pub fn groupByWeightedModeOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    weight_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByWeightedOn(DeviceDataFrame, .weighted_mode, frame, key_names, value_name, weight_name, output_name, 0.5);
}

pub fn groupByWeightedModeWeightOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    weight_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByWeightedOn(DeviceDataFrame, .weighted_mode_weight, frame, key_names, value_name, weight_name, output_name, 0.5);
}

pub fn groupByWeightedModeRatioOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    weight_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByWeightedOn(DeviceDataFrame, .weighted_mode_ratio, frame, key_names, value_name, weight_name, output_name, 0.5);
}

pub fn groupByWeightedModeMarginOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    weight_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByWeightedOn(DeviceDataFrame, .weighted_mode_margin, frame, key_names, value_name, weight_name, output_name, 0.5);
}

pub fn groupByWeightedModeMarginRatioOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    weight_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByWeightedOn(DeviceDataFrame, .weighted_mode_margin_ratio, frame, key_names, value_name, weight_name, output_name, 0.5);
}

pub fn groupByWeightedEntropyOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    weight_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByWeightedOn(DeviceDataFrame, .weighted_entropy, frame, key_names, value_name, weight_name, output_name, 0.5);
}

pub fn groupByWeightedGiniImpurityOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    weight_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByWeightedOn(DeviceDataFrame, .weighted_gini_impurity, frame, key_names, value_name, weight_name, output_name, 0.5);
}

pub fn groupByWeightedPerplexityOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    weight_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByWeightedOn(DeviceDataFrame, .weighted_perplexity, frame, key_names, value_name, weight_name, output_name, 0.5);
}

pub fn groupByWeightedInverseSimpsonOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    weight_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByWeightedOn(DeviceDataFrame, .weighted_inverse_simpson, frame, key_names, value_name, weight_name, output_name, 0.5);
}

pub fn groupByWeightedSimpsonConcentrationOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    weight_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByWeightedOn(DeviceDataFrame, .weighted_simpson_concentration, frame, key_names, value_name, weight_name, output_name, 0.5);
}

pub fn groupByWeightedEvennessOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    weight_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByWeightedOn(DeviceDataFrame, .weighted_evenness, frame, key_names, value_name, weight_name, output_name, 0.5);
}

pub fn groupByPairCountOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);
    const lhs_column = try frame.column(lhs_name);
    const rhs_column = try frame.column(rhs_name);
    if (frame.rows != lhs_column.*.len() or frame.rows != rhs_column.*.len()) return error.LengthMismatch;
    const lhs_validity = try groupColumnValidityValues(frame.allocator, lhs_column.*);
    defer if (lhs_validity) |validity| frame.allocator.free(validity);
    const rhs_validity = try groupColumnValidityValues(frame.allocator, rhs_column.*);
    defer if (rhs_validity) |validity| frame.allocator.free(validity);

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(frame.allocator);
    var pair_counts: std.ArrayList(i64) = .empty;
    defer pair_counts.deinit(frame.allocator);

    for (0..frame.rows) |row| {
        if (!try rowHasValidKeys(frame.allocator, frame, key_names, row)) continue;
        const group_index = (try findMultiKeyGroupIndex(frame.allocator, frame, key_names, representative_rows.items, row)) orelse blk: {
            try representative_rows.append(frame.allocator, row);
            try pair_counts.append(frame.allocator, 0);
            break :blk representative_rows.items.len - 1;
        };
        const lhs_valid = if (lhs_validity) |validity| validity[row] else true;
        const rhs_valid = if (rhs_validity) |validity| validity[row] else true;
        if (lhs_valid and rhs_valid) {
            pair_counts.items[group_index] += 1;
        }
    }

    const output_column = try DeviceColumn.fromSlice(i64, frame.allocator, pair_counts.items, frame.device);
    return initMultiKeyAggregatedDataFrame(DeviceDataFrame, frame, key_names, representative_rows.items, output_name, output_column);
}

pub fn groupByPairOn(
    comptime DeviceDataFrame: type,
    aggregation: GroupByPairAggregation,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);
    const lhs_column = try frame.column(lhs_name);
    const rhs_column = try frame.column(rhs_name);

    var lhs = try ownedGroupRealColumn(frame.allocator, lhs_column.*);
    defer lhs.deinit();
    var rhs = try ownedGroupRealColumn(frame.allocator, rhs_column.*);
    defer rhs.deinit();
    if (frame.rows != lhs.values.len or frame.rows != rhs.values.len) return error.LengthMismatch;

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(frame.allocator);
    var pair_counts: std.ArrayList(i64) = .empty;
    defer pair_counts.deinit(frame.allocator);
    var lhs_sums: std.ArrayList(f64) = .empty;
    defer lhs_sums.deinit(frame.allocator);
    var rhs_sums: std.ArrayList(f64) = .empty;
    defer rhs_sums.deinit(frame.allocator);
    var lhs_square_sums: std.ArrayList(f64) = .empty;
    defer lhs_square_sums.deinit(frame.allocator);
    var rhs_square_sums: std.ArrayList(f64) = .empty;
    defer rhs_square_sums.deinit(frame.allocator);
    var cross_sums: std.ArrayList(f64) = .empty;
    defer cross_sums.deinit(frame.allocator);
    var manhattan_sums: std.ArrayList(f64) = .empty;
    defer manhattan_sums.deinit(frame.allocator);
    var chebyshev_values: std.ArrayList(f64) = .empty;
    defer chebyshev_values.deinit(frame.allocator);
    var canberra_sums: std.ArrayList(f64) = .empty;
    defer canberra_sums.deinit(frame.allocator);
    var bray_curtis_denominators: std.ArrayList(f64) = .empty;
    defer bray_curtis_denominators.deinit(frame.allocator);
    var mape_sums: std.ArrayList(f64) = .empty;
    defer mape_sums.deinit(frame.allocator);
    var smape_sums: std.ArrayList(f64) = .empty;
    defer smape_sums.deinit(frame.allocator);
    var signed_error_sums: std.ArrayList(f64) = .empty;
    defer signed_error_sums.deinit(frame.allocator);

    for (0..frame.rows) |row| {
        if (!try rowHasValidKeys(frame.allocator, frame, key_names, row)) continue;
        const group_index = (try findMultiKeyGroupIndex(frame.allocator, frame, key_names, representative_rows.items, row)) orelse blk: {
            try representative_rows.append(frame.allocator, row);
            try pair_counts.append(frame.allocator, 0);
            try lhs_sums.append(frame.allocator, 0.0);
            try rhs_sums.append(frame.allocator, 0.0);
            try lhs_square_sums.append(frame.allocator, 0.0);
            try rhs_square_sums.append(frame.allocator, 0.0);
            try cross_sums.append(frame.allocator, 0.0);
            try manhattan_sums.append(frame.allocator, 0.0);
            try chebyshev_values.append(frame.allocator, 0.0);
            try canberra_sums.append(frame.allocator, 0.0);
            try bray_curtis_denominators.append(frame.allocator, 0.0);
            try mape_sums.append(frame.allocator, 0.0);
            try smape_sums.append(frame.allocator, 0.0);
            try signed_error_sums.append(frame.allocator, 0.0);
            break :blk representative_rows.items.len - 1;
        };

        if (lhs.validity) |validity| {
            if (!validity[row]) continue;
        }
        if (rhs.validity) |validity| {
            if (!validity[row]) continue;
        }
        const lhs_value = lhs.values[row];
        const rhs_value = rhs.values[row];
        const signed_error = lhs_value - rhs_value;
        const abs_error = @abs(signed_error);
        const abs_lhs = @abs(lhs_value);
        const abs_rhs = @abs(rhs_value);
        const abs_sum = abs_lhs + abs_rhs;
        pair_counts.items[group_index] += 1;
        lhs_sums.items[group_index] += lhs_value;
        rhs_sums.items[group_index] += rhs_value;
        lhs_square_sums.items[group_index] += lhs_value * lhs_value;
        rhs_square_sums.items[group_index] += rhs_value * rhs_value;
        cross_sums.items[group_index] += lhs_value * rhs_value;
        manhattan_sums.items[group_index] += abs_error;
        chebyshev_values.items[group_index] = @max(chebyshev_values.items[group_index], abs_error);
        // Match row-wise paired metrics: zero/zero coordinates do not
        // contribute to Canberra, while Bray-Curtis surfaces NaN when the
        // entire group's absolute denominator is zero.
        canberra_sums.items[group_index] += if (abs_sum == 0.0) 0.0 else abs_error / abs_sum;
        bray_curtis_denominators.items[group_index] += abs_sum;
        mape_sums.items[group_index] += if (lhs_value == 0.0) std.math.nan(f64) else abs_error / abs_lhs;
        smape_sums.items[group_index] += if (abs_sum == 0.0) std.math.nan(f64) else 2.0 * abs_error / abs_sum;
        signed_error_sums.items[group_index] += signed_error;
    }

    const out = try frame.allocator.alloc(f64, representative_rows.items.len);
    defer frame.allocator.free(out);
    for (
        pair_counts.items,
        lhs_sums.items,
        rhs_sums.items,
        lhs_square_sums.items,
        rhs_square_sums.items,
        cross_sums.items,
        manhattan_sums.items,
        chebyshev_values.items,
        canberra_sums.items,
        bray_curtis_denominators.items,
        mape_sums.items,
        smape_sums.items,
        signed_error_sums.items,
        out,
    ) |pair_count, lhs_sum, rhs_sum, lhs_square_sum, rhs_square_sum, cross_sum, manhattan_sum, chebyshev_value, canberra_sum, bray_curtis_denominator, mape_sum, smape_sum, signed_error_sum, *slot| {
        if (pair_count == 0) {
            slot.* = std.math.nan(f64);
            continue;
        }
        const count_f64: f64 = @floatFromInt(pair_count);
        const mean_lhs = lhs_sum / count_f64;
        const mean_rhs = rhs_sum / count_f64;
        const covariance = cross_sum / count_f64 - mean_lhs * mean_rhs;
        const squared_distance = lhs_square_sum + rhs_square_sum - 2.0 * cross_sum;
        var lhs_variance = lhs_square_sum / count_f64 - mean_lhs * mean_lhs;
        var rhs_variance = rhs_square_sum / count_f64 - mean_rhs * mean_rhs;
        if (lhs_variance < 0.0 and lhs_variance > -1e-12) lhs_variance = 0.0;
        if (rhs_variance < 0.0 and rhs_variance > -1e-12) rhs_variance = 0.0;
        slot.* = switch (aggregation) {
            .dot => cross_sum,
            .cosine_similarity => if (lhs_square_sum == 0.0 or rhs_square_sum == 0.0) std.math.nan(f64) else cross_sum / (std.math.sqrt(lhs_square_sum) * std.math.sqrt(rhs_square_sum)),
            .squared_euclidean_distance => squared_distance,
            .euclidean_distance => std.math.sqrt(squared_distance),
            .manhattan_distance => manhattan_sum,
            .chebyshev_distance => chebyshev_value,
            .canberra_distance => canberra_sum,
            .bray_curtis_distance => if (bray_curtis_denominator == 0.0) std.math.nan(f64) else manhattan_sum / bray_curtis_denominator,
            .mean_error => signed_error_sum / count_f64,
            .mae => manhattan_sum / count_f64,
            .mse => squared_distance / count_f64,
            .rmse => std.math.sqrt(squared_distance / count_f64),
            .mape => mape_sum / count_f64,
            .smape => smape_sum / count_f64,
            .covariance => covariance,
            .correlation => if (lhs_variance == 0.0 or rhs_variance == 0.0) std.math.nan(f64) else covariance / std.math.sqrt(lhs_variance * rhs_variance),
            .beta => if (lhs_variance == 0.0) std.math.nan(f64) else covariance / lhs_variance,
        };
    }

    const output_column = try DeviceColumn.fromSlice(f64, frame.allocator, out, frame.device);
    return initMultiKeyAggregatedDataFrame(DeviceDataFrame, frame, key_names, representative_rows.items, output_name, output_column);
}

pub fn groupByDotOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByPairOn(DeviceDataFrame, .dot, frame, key_names, lhs_name, rhs_name, output_name);
}

pub fn groupByCosineSimilarityOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByPairOn(DeviceDataFrame, .cosine_similarity, frame, key_names, lhs_name, rhs_name, output_name);
}

pub fn groupBySquaredEuclideanDistanceOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByPairOn(DeviceDataFrame, .squared_euclidean_distance, frame, key_names, lhs_name, rhs_name, output_name);
}

pub fn groupByEuclideanDistanceOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByPairOn(DeviceDataFrame, .euclidean_distance, frame, key_names, lhs_name, rhs_name, output_name);
}

pub fn groupByManhattanDistanceOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByPairOn(DeviceDataFrame, .manhattan_distance, frame, key_names, lhs_name, rhs_name, output_name);
}

pub fn groupByChebyshevDistanceOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByPairOn(DeviceDataFrame, .chebyshev_distance, frame, key_names, lhs_name, rhs_name, output_name);
}

pub fn groupByCanberraDistanceOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByPairOn(DeviceDataFrame, .canberra_distance, frame, key_names, lhs_name, rhs_name, output_name);
}

pub fn groupByBrayCurtisDistanceOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByPairOn(DeviceDataFrame, .bray_curtis_distance, frame, key_names, lhs_name, rhs_name, output_name);
}

pub fn groupByMeanErrorOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByPairOn(DeviceDataFrame, .mean_error, frame, key_names, lhs_name, rhs_name, output_name);
}

pub fn groupByMaeOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByPairOn(DeviceDataFrame, .mae, frame, key_names, lhs_name, rhs_name, output_name);
}

pub fn groupByMseOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByPairOn(DeviceDataFrame, .mse, frame, key_names, lhs_name, rhs_name, output_name);
}

pub fn groupByRmseOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByPairOn(DeviceDataFrame, .rmse, frame, key_names, lhs_name, rhs_name, output_name);
}

pub fn groupByMapeOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByPairOn(DeviceDataFrame, .mape, frame, key_names, lhs_name, rhs_name, output_name);
}

pub fn groupBySmapeOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByPairOn(DeviceDataFrame, .smape, frame, key_names, lhs_name, rhs_name, output_name);
}

pub fn groupByCovarianceOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByPairOn(DeviceDataFrame, .covariance, frame, key_names, lhs_name, rhs_name, output_name);
}

pub fn groupByCorrelationOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByPairOn(DeviceDataFrame, .correlation, frame, key_names, lhs_name, rhs_name, output_name);
}

pub fn groupByBetaOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByPairOn(DeviceDataFrame, .beta, frame, key_names, lhs_name, rhs_name, output_name);
}

pub fn groupByWeightedPairOn(
    comptime DeviceDataFrame: type,
    aggregation: GroupByWeightedPairAggregation,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
    weight_name: []const u8,
    output_name: []const u8,
    correction: f64,
) GroupByOnError!DeviceDataFrame {
    if (std.math.isNan(correction) or correction < 0.0) return error.InvalidShape;
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);
    const lhs_column = try frame.column(lhs_name);
    const rhs_column = try frame.column(rhs_name);
    const weight_column = try frame.column(weight_name);

    var lhs = try ownedGroupRealColumn(frame.allocator, lhs_column.*);
    defer lhs.deinit();
    var rhs = try ownedGroupRealColumn(frame.allocator, rhs_column.*);
    defer rhs.deinit();
    var weights = try ownedGroupRealColumn(frame.allocator, weight_column.*);
    defer weights.deinit();
    if (frame.rows != lhs.values.len or frame.rows != rhs.values.len or frame.rows != weights.values.len) return error.LengthMismatch;

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(frame.allocator);
    var weight_sums: std.ArrayList(f64) = .empty;
    defer weight_sums.deinit(frame.allocator);
    var lhs_sums: std.ArrayList(f64) = .empty;
    defer lhs_sums.deinit(frame.allocator);
    var rhs_sums: std.ArrayList(f64) = .empty;
    defer rhs_sums.deinit(frame.allocator);
    var lhs_square_sums: std.ArrayList(f64) = .empty;
    defer lhs_square_sums.deinit(frame.allocator);
    var rhs_square_sums: std.ArrayList(f64) = .empty;
    defer rhs_square_sums.deinit(frame.allocator);
    var cross_sums: std.ArrayList(f64) = .empty;
    defer cross_sums.deinit(frame.allocator);
    var weighted_abs_error_sums: std.ArrayList(f64) = .empty;
    defer weighted_abs_error_sums.deinit(frame.allocator);
    var chebyshev_values: std.ArrayList(f64) = .empty;
    defer chebyshev_values.deinit(frame.allocator);
    var weighted_canberra_sums: std.ArrayList(f64) = .empty;
    defer weighted_canberra_sums.deinit(frame.allocator);
    var weighted_bray_denominators: std.ArrayList(f64) = .empty;
    defer weighted_bray_denominators.deinit(frame.allocator);
    var weighted_mape_sums: std.ArrayList(f64) = .empty;
    defer weighted_mape_sums.deinit(frame.allocator);
    var weighted_smape_sums: std.ArrayList(f64) = .empty;
    defer weighted_smape_sums.deinit(frame.allocator);

    for (0..frame.rows) |row| {
        if (lhs.validity) |validity| {
            if (!validity[row]) continue;
        }
        if (rhs.validity) |validity| {
            if (!validity[row]) continue;
        }
        if (weights.validity) |validity| {
            if (!validity[row]) continue;
        }
        if (!try rowHasValidKeys(frame.allocator, frame, key_names, row)) continue;
        const weight = weights.values[row];
        if (weight < 0.0) return error.InvalidShape;
        const group_index = (try findMultiKeyGroupIndex(frame.allocator, frame, key_names, representative_rows.items, row)) orelse blk: {
            try representative_rows.append(frame.allocator, row);
            try weight_sums.append(frame.allocator, 0.0);
            try lhs_sums.append(frame.allocator, 0.0);
            try rhs_sums.append(frame.allocator, 0.0);
            try lhs_square_sums.append(frame.allocator, 0.0);
            try rhs_square_sums.append(frame.allocator, 0.0);
            try cross_sums.append(frame.allocator, 0.0);
            try weighted_abs_error_sums.append(frame.allocator, 0.0);
            try chebyshev_values.append(frame.allocator, 0.0);
            try weighted_canberra_sums.append(frame.allocator, 0.0);
            try weighted_bray_denominators.append(frame.allocator, 0.0);
            try weighted_mape_sums.append(frame.allocator, 0.0);
            try weighted_smape_sums.append(frame.allocator, 0.0);
            break :blk representative_rows.items.len - 1;
        };
        const lhs_value = lhs.values[row];
        const rhs_value = rhs.values[row];
        const signed_error = lhs_value - rhs_value;
        const abs_error = @abs(signed_error);
        const abs_lhs = @abs(lhs_value);
        const abs_rhs = @abs(rhs_value);
        const abs_sum = abs_lhs + abs_rhs;
        weight_sums.items[group_index] += weight;
        lhs_sums.items[group_index] += weight * lhs_value;
        rhs_sums.items[group_index] += weight * rhs_value;
        lhs_square_sums.items[group_index] += weight * lhs_value * lhs_value;
        rhs_square_sums.items[group_index] += weight * rhs_value * rhs_value;
        cross_sums.items[group_index] += weight * lhs_value * rhs_value;
        weighted_abs_error_sums.items[group_index] += weight * abs_error;
        if (weight == 0.0) continue;
        chebyshev_values.items[group_index] = @max(chebyshev_values.items[group_index], abs_error);
        weighted_canberra_sums.items[group_index] += if (abs_sum == 0.0) 0.0 else weight * abs_error / abs_sum;
        weighted_bray_denominators.items[group_index] += weight * abs_sum;
        weighted_mape_sums.items[group_index] += if (lhs_value == 0.0) std.math.nan(f64) else weight * abs_error / abs_lhs;
        weighted_smape_sums.items[group_index] += if (abs_sum == 0.0) std.math.nan(f64) else weight * 2.0 * abs_error / abs_sum;
    }

    const out = try frame.allocator.alloc(f64, representative_rows.items.len);
    defer frame.allocator.free(out);
    for (
        weight_sums.items,
        lhs_sums.items,
        rhs_sums.items,
        lhs_square_sums.items,
        rhs_square_sums.items,
        cross_sums.items,
        weighted_abs_error_sums.items,
        chebyshev_values.items,
        weighted_canberra_sums.items,
        weighted_bray_denominators.items,
        weighted_mape_sums.items,
        weighted_smape_sums.items,
        out,
    ) |weight_sum, lhs_sum, rhs_sum, lhs_square_sum, rhs_square_sum, cross_sum, weighted_abs_error_sum, chebyshev_value, weighted_canberra_sum, weighted_bray_denominator, weighted_mape_sum, weighted_smape_sum, *slot| {
        if (!(weight_sum > 0.0)) {
            slot.* = std.math.nan(f64);
            continue;
        }
        const denominator = weight_sum - correction;
        var lhs_centered = lhs_square_sum - lhs_sum * lhs_sum / weight_sum;
        var rhs_centered = rhs_square_sum - rhs_sum * rhs_sum / weight_sum;
        const cross_centered = cross_sum - lhs_sum * rhs_sum / weight_sum;
        const squared_distance = lhs_square_sum + rhs_square_sum - 2.0 * cross_sum;
        if (lhs_centered < 0.0 and lhs_centered > -1e-12) lhs_centered = 0.0;
        if (rhs_centered < 0.0 and rhs_centered > -1e-12) rhs_centered = 0.0;
        slot.* = switch (aggregation) {
            .weighted_dot => cross_sum,
            .weighted_cosine_similarity => if (lhs_square_sum == 0.0 or rhs_square_sum == 0.0) std.math.nan(f64) else cross_sum / (std.math.sqrt(lhs_square_sum) * std.math.sqrt(rhs_square_sum)),
            .weighted_squared_euclidean_distance => squared_distance,
            .weighted_euclidean_distance => std.math.sqrt(squared_distance),
            .weighted_manhattan_distance => weighted_abs_error_sum,
            .weighted_chebyshev_distance => chebyshev_value,
            .weighted_canberra_distance => weighted_canberra_sum,
            .weighted_bray_curtis_distance => if (weighted_bray_denominator == 0.0) std.math.nan(f64) else weighted_abs_error_sum / weighted_bray_denominator,
            .weighted_mean_error => (lhs_sum - rhs_sum) / weight_sum,
            .weighted_mae => weighted_abs_error_sum / weight_sum,
            .weighted_mse => squared_distance / weight_sum,
            .weighted_rmse => std.math.sqrt(squared_distance / weight_sum),
            .weighted_mape => weighted_mape_sum / weight_sum,
            .weighted_smape => weighted_smape_sum / weight_sum,
            .weighted_covariance => if (denominator <= 0.0) std.math.nan(f64) else cross_centered / denominator,
            .weighted_correlation => if (denominator <= 0.0 or lhs_centered == 0.0 or rhs_centered == 0.0) std.math.nan(f64) else cross_centered / std.math.sqrt(lhs_centered * rhs_centered),
            .weighted_beta => if (denominator <= 0.0 or lhs_centered == 0.0) std.math.nan(f64) else cross_centered / lhs_centered,
        };
    }

    const output_column = try DeviceColumn.fromSlice(f64, frame.allocator, out, frame.device);
    return initMultiKeyAggregatedDataFrame(DeviceDataFrame, frame, key_names, representative_rows.items, output_name, output_column);
}

pub fn groupByWeightedDotOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
    weight_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByWeightedPairOn(DeviceDataFrame, .weighted_dot, frame, key_names, lhs_name, rhs_name, weight_name, output_name, 0.0);
}

pub fn groupByWeightedCosineSimilarityOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
    weight_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByWeightedPairOn(DeviceDataFrame, .weighted_cosine_similarity, frame, key_names, lhs_name, rhs_name, weight_name, output_name, 0.0);
}

pub fn groupByWeightedSquaredEuclideanDistanceOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
    weight_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByWeightedPairOn(DeviceDataFrame, .weighted_squared_euclidean_distance, frame, key_names, lhs_name, rhs_name, weight_name, output_name, 0.0);
}

pub fn groupByWeightedEuclideanDistanceOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
    weight_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByWeightedPairOn(DeviceDataFrame, .weighted_euclidean_distance, frame, key_names, lhs_name, rhs_name, weight_name, output_name, 0.0);
}

pub fn groupByWeightedManhattanDistanceOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
    weight_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByWeightedPairOn(DeviceDataFrame, .weighted_manhattan_distance, frame, key_names, lhs_name, rhs_name, weight_name, output_name, 0.0);
}

pub fn groupByWeightedChebyshevDistanceOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
    weight_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByWeightedPairOn(DeviceDataFrame, .weighted_chebyshev_distance, frame, key_names, lhs_name, rhs_name, weight_name, output_name, 0.0);
}

pub fn groupByWeightedCanberraDistanceOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
    weight_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByWeightedPairOn(DeviceDataFrame, .weighted_canberra_distance, frame, key_names, lhs_name, rhs_name, weight_name, output_name, 0.0);
}

pub fn groupByWeightedBrayCurtisDistanceOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
    weight_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByWeightedPairOn(DeviceDataFrame, .weighted_bray_curtis_distance, frame, key_names, lhs_name, rhs_name, weight_name, output_name, 0.0);
}

pub fn groupByWeightedMeanErrorOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
    weight_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByWeightedPairOn(DeviceDataFrame, .weighted_mean_error, frame, key_names, lhs_name, rhs_name, weight_name, output_name, 0.0);
}

pub fn groupByWeightedMaeOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
    weight_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByWeightedPairOn(DeviceDataFrame, .weighted_mae, frame, key_names, lhs_name, rhs_name, weight_name, output_name, 0.0);
}

pub fn groupByWeightedMseOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
    weight_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByWeightedPairOn(DeviceDataFrame, .weighted_mse, frame, key_names, lhs_name, rhs_name, weight_name, output_name, 0.0);
}

pub fn groupByWeightedRmseOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
    weight_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByWeightedPairOn(DeviceDataFrame, .weighted_rmse, frame, key_names, lhs_name, rhs_name, weight_name, output_name, 0.0);
}

pub fn groupByWeightedMapeOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
    weight_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByWeightedPairOn(DeviceDataFrame, .weighted_mape, frame, key_names, lhs_name, rhs_name, weight_name, output_name, 0.0);
}

pub fn groupByWeightedSmapeOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
    weight_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByWeightedPairOn(DeviceDataFrame, .weighted_smape, frame, key_names, lhs_name, rhs_name, weight_name, output_name, 0.0);
}

pub fn groupByWeightedCovarianceOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
    weight_name: []const u8,
    output_name: []const u8,
    correction: f64,
) GroupByOnError!DeviceDataFrame {
    return groupByWeightedPairOn(DeviceDataFrame, .weighted_covariance, frame, key_names, lhs_name, rhs_name, weight_name, output_name, correction);
}

pub fn groupByWeightedCorrelationOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
    weight_name: []const u8,
    output_name: []const u8,
    correction: f64,
) GroupByOnError!DeviceDataFrame {
    return groupByWeightedPairOn(DeviceDataFrame, .weighted_correlation, frame, key_names, lhs_name, rhs_name, weight_name, output_name, correction);
}

pub fn groupByWeightedBetaOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
    weight_name: []const u8,
    output_name: []const u8,
    correction: f64,
) GroupByOnError!DeviceDataFrame {
    return groupByWeightedPairOn(DeviceDataFrame, .weighted_beta, frame, key_names, lhs_name, rhs_name, weight_name, output_name, correction);
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

fn groupByBoolIndexOn(
    comptime DeviceDataFrame: type,
    aggregation: GroupByBoolIndexAggregation,
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

    const match_value = switch (aggregation) {
        .first_true_index, .last_true_index => true,
        .first_false_index, .last_false_index => false,
    };
    const keep_last = switch (aggregation) {
        .first_true_index, .first_false_index => false,
        .last_true_index, .last_false_index => true,
    };

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(frame.allocator);
    var index_values: std.ArrayList(i64) = .empty;
    defer index_values.deinit(frame.allocator);
    var output_validity: std.ArrayList(bool) = .empty;
    defer output_validity.deinit(frame.allocator);

    for (values, 0..) |value_item, row| {
        if (!try rowHasValidKeys(frame.allocator, frame, key_names, row)) continue;
        const group_index = (try findMultiKeyGroupIndex(frame.allocator, frame, key_names, representative_rows.items, row)) orelse blk: {
            try representative_rows.append(frame.allocator, row);
            try index_values.append(frame.allocator, 0);
            try output_validity.append(frame.allocator, false);
            break :blk representative_rows.items.len - 1;
        };
        if (maybe_value_validity) |validity| {
            if (!validity[row]) continue;
        }
        if (value_item != match_value) continue;
        if (keep_last or !output_validity.items[group_index]) {
            index_values.items[group_index] = @intCast(row);
            output_validity.items[group_index] = true;
        }
    }

    const output_column = try DeviceColumn.fromSliceWithValidity(i64, frame.allocator, index_values.items, output_validity.items, frame.device);
    return initMultiKeyAggregatedDataFrame(DeviceDataFrame, frame, key_names, representative_rows.items, output_name, output_column);
}

pub fn groupByFirstTrueIndexOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByBoolIndexOn(DeviceDataFrame, .first_true_index, frame, key_names, value_name, output_name);
}

pub fn groupByLastTrueIndexOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByBoolIndexOn(DeviceDataFrame, .last_true_index, frame, key_names, value_name, output_name);
}

pub fn groupByFirstFalseIndexOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByBoolIndexOn(DeviceDataFrame, .first_false_index, frame, key_names, value_name, output_name);
}

pub fn groupByLastFalseIndexOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByBoolIndexOn(DeviceDataFrame, .last_false_index, frame, key_names, value_name, output_name);
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
        .any_valid, .all_valid, .any_null, .all_null => blk: {
            const outputs = try frame.allocator.alloc(bool, valid_counts.items.len);
            defer frame.allocator.free(outputs);
            for (valid_counts.items, null_counts.items, outputs) |valid_count, null_count, *slot| {
                slot.* = switch (aggregation) {
                    .any_valid => valid_count != 0,
                    .all_valid => null_count == 0,
                    .any_null => null_count != 0,
                    .all_null => valid_count == 0,
                    else => unreachable,
                };
            }
            break :blk try DeviceColumn.fromSlice(bool, frame.allocator, outputs, frame.device);
        },
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

pub fn groupByAnyValidOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByValidityCountOn(DeviceDataFrame, .any_valid, frame, key_names, value_name, output_name);
}

pub fn groupByAllValidOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByValidityCountOn(DeviceDataFrame, .all_valid, frame, key_names, value_name, output_name);
}

pub fn groupByAnyNullOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByValidityCountOn(DeviceDataFrame, .any_null, frame, key_names, value_name, output_name);
}

pub fn groupByAllNullOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByValidityCountOn(DeviceDataFrame, .all_null, frame, key_names, value_name, output_name);
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

fn groupByValidityIndexOn(
    comptime DeviceDataFrame: type,
    aggregation: GroupByValidityIndexAggregation,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);
    const value = try frame.column(value_name);
    const maybe_value_validity = try groupColumnValidityValues(frame.allocator, value.*);
    defer if (maybe_value_validity) |validity| frame.allocator.free(validity);

    const match_valid = switch (aggregation) {
        .first_valid_index, .last_valid_index => true,
        .first_null_index, .last_null_index => false,
    };
    const keep_last = switch (aggregation) {
        .first_valid_index, .first_null_index => false,
        .last_valid_index, .last_null_index => true,
    };

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(frame.allocator);
    var index_values: std.ArrayList(i64) = .empty;
    defer index_values.deinit(frame.allocator);
    var output_validity: std.ArrayList(bool) = .empty;
    defer output_validity.deinit(frame.allocator);

    for (0..frame.rows) |row| {
        if (!try rowHasValidKeys(frame.allocator, frame, key_names, row)) continue;
        const group_index = (try findMultiKeyGroupIndex(frame.allocator, frame, key_names, representative_rows.items, row)) orelse blk: {
            try representative_rows.append(frame.allocator, row);
            try index_values.append(frame.allocator, 0);
            try output_validity.append(frame.allocator, false);
            break :blk representative_rows.items.len - 1;
        };
        const value_valid = if (maybe_value_validity) |validity| validity[row] else true;
        if (value_valid != match_valid) continue;
        if (keep_last or !output_validity.items[group_index]) {
            index_values.items[group_index] = @intCast(row);
            output_validity.items[group_index] = true;
        }
    }

    const output_column = try DeviceColumn.fromSliceWithValidity(i64, frame.allocator, index_values.items, output_validity.items, frame.device);
    return initMultiKeyAggregatedDataFrame(DeviceDataFrame, frame, key_names, representative_rows.items, output_name, output_column);
}

pub fn groupByFirstValidIndexOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByValidityIndexOn(DeviceDataFrame, .first_valid_index, frame, key_names, value_name, output_name);
}

pub fn groupByLastValidIndexOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByValidityIndexOn(DeviceDataFrame, .last_valid_index, frame, key_names, value_name, output_name);
}

pub fn groupByFirstNullIndexOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByValidityIndexOn(DeviceDataFrame, .first_null_index, frame, key_names, value_name, output_name);
}

pub fn groupByLastNullIndexOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByValidityIndexOn(DeviceDataFrame, .last_null_index, frame, key_names, value_name, output_name);
}

pub fn groupByNumericQualityOnDispatchValue(
    comptime DeviceDataFrame: type,
    aggregation: GroupByNumericQualityAggregation,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
    value: DeviceColumn,
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    return switch (value) {
        .i8 => |typed| groupByNumericQualityOnTyped(DeviceDataFrame, i8, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .i16 => |typed| groupByNumericQualityOnTyped(DeviceDataFrame, i16, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .i32 => |typed| groupByNumericQualityOnTyped(DeviceDataFrame, i32, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .i64 => |typed| groupByNumericQualityOnTyped(DeviceDataFrame, i64, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u8 => |typed| groupByNumericQualityOnTyped(DeviceDataFrame, u8, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u16 => |typed| groupByNumericQualityOnTyped(DeviceDataFrame, u16, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u32 => |typed| groupByNumericQualityOnTyped(DeviceDataFrame, u32, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u64 => |typed| groupByNumericQualityOnTyped(DeviceDataFrame, u64, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .usize => |typed| groupByNumericQualityOnTyped(DeviceDataFrame, usize, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .isize => |typed| groupByNumericQualityOnTyped(DeviceDataFrame, isize, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .f16 => |typed| groupByNumericQualityOnTyped(DeviceDataFrame, f16, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .f32 => |typed| groupByNumericQualityOnTyped(DeviceDataFrame, f32, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .f64 => |typed| groupByNumericQualityOnTyped(DeviceDataFrame, f64, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn groupByNumericQualityOnTyped(
    comptime DeviceDataFrame: type,
    comptime V: type,
    aggregation: GroupByNumericQualityAggregation,
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
    var match_counts: std.ArrayList(i64) = .empty;
    defer match_counts.deinit(allocator);
    var valid_counts: std.ArrayList(i64) = .empty;
    defer valid_counts.deinit(allocator);

    for (values, 0..) |value_item, row| {
        if (!try rowHasValidKeys(allocator, frame, key_names, row)) continue;
        const group_index = (try findMultiKeyGroupIndex(allocator, frame, key_names, representative_rows.items, row)) orelse blk: {
            try representative_rows.append(allocator, row);
            try match_counts.append(allocator, 0);
            try valid_counts.append(allocator, 0);
            break :blk representative_rows.items.len - 1;
        };
        if (maybe_value_validity) |validity| {
            if (!validity[row]) continue;
        }
        valid_counts.items[group_index] += 1;
        if (groupNumericQualityMatchesTyped(V, value_item, aggregation)) match_counts.items[group_index] += 1;
    }

    const output_column: DeviceColumn = if (groupNumericQualityIsRatio(aggregation)) blk: {
        const ratios = try allocator.alloc(f64, representative_rows.items.len);
        defer allocator.free(ratios);
        for (match_counts.items, valid_counts.items, ratios) |match_count, valid_count, *slot| {
            slot.* = if (valid_count == 0) std.math.nan(f64) else @as(f64, @floatFromInt(match_count)) / @as(f64, @floatFromInt(valid_count));
        }
        break :blk try DeviceColumn.fromSlice(f64, allocator, ratios, device_value);
    } else try DeviceColumn.fromSlice(i64, allocator, match_counts.items, device_value);
    return initMultiKeyAggregatedDataFrame(DeviceDataFrame, frame, key_names, representative_rows.items, output_name, output_column);
}

fn groupByNumericQualityOn(
    comptime DeviceDataFrame: type,
    aggregation: GroupByNumericQualityAggregation,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);
    const value = try frame.column(value_name);
    return groupByNumericQualityOnDispatchValue(DeviceDataFrame, aggregation, frame.allocator, frame, key_names, output_name, value.*, frame.device);
}

pub fn groupByNumericQualityIndexOnDispatchValue(
    comptime DeviceDataFrame: type,
    aggregation: GroupByNumericQualityIndexAggregation,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_name: []const u8,
    value: DeviceColumn,
    device_value: array_mod.Device,
) GroupByOnError!DeviceDataFrame {
    return switch (value) {
        .i8 => |typed| groupByNumericQualityIndexOnTyped(DeviceDataFrame, i8, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .i16 => |typed| groupByNumericQualityIndexOnTyped(DeviceDataFrame, i16, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .i32 => |typed| groupByNumericQualityIndexOnTyped(DeviceDataFrame, i32, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .i64 => |typed| groupByNumericQualityIndexOnTyped(DeviceDataFrame, i64, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u8 => |typed| groupByNumericQualityIndexOnTyped(DeviceDataFrame, u8, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u16 => |typed| groupByNumericQualityIndexOnTyped(DeviceDataFrame, u16, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u32 => |typed| groupByNumericQualityIndexOnTyped(DeviceDataFrame, u32, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .u64 => |typed| groupByNumericQualityIndexOnTyped(DeviceDataFrame, u64, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .usize => |typed| groupByNumericQualityIndexOnTyped(DeviceDataFrame, usize, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .isize => |typed| groupByNumericQualityIndexOnTyped(DeviceDataFrame, isize, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .f16 => |typed| groupByNumericQualityIndexOnTyped(DeviceDataFrame, f16, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .f32 => |typed| groupByNumericQualityIndexOnTyped(DeviceDataFrame, f32, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .f64 => |typed| groupByNumericQualityIndexOnTyped(DeviceDataFrame, f64, aggregation, allocator, frame, key_names, output_name, typed, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn groupByNumericQualityIndexOnTyped(
    comptime DeviceDataFrame: type,
    comptime V: type,
    aggregation: GroupByNumericQualityIndexAggregation,
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

    const predicate = groupNumericQualityIndexPredicate(aggregation);
    const keep_last = groupNumericQualityIndexKeepsLast(aggregation);
    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(allocator);
    var index_values: std.ArrayList(i64) = .empty;
    defer index_values.deinit(allocator);
    var output_validity: std.ArrayList(bool) = .empty;
    defer output_validity.deinit(allocator);

    for (values, 0..) |value_item, row| {
        if (!try rowHasValidKeys(allocator, frame, key_names, row)) continue;
        const group_index = (try findMultiKeyGroupIndex(allocator, frame, key_names, representative_rows.items, row)) orelse blk: {
            try representative_rows.append(allocator, row);
            try index_values.append(allocator, 0);
            try output_validity.append(allocator, false);
            break :blk representative_rows.items.len - 1;
        };
        if (maybe_value_validity) |validity| {
            if (!validity[row]) continue;
        }
        if (!groupNumericQualityMatchesTyped(V, value_item, predicate)) continue;
        if (keep_last or !output_validity.items[group_index]) {
            index_values.items[group_index] = @intCast(row);
            output_validity.items[group_index] = true;
        }
    }

    const output_column = try DeviceColumn.fromSliceWithValidity(i64, allocator, index_values.items, output_validity.items, device_value);
    return initMultiKeyAggregatedDataFrame(DeviceDataFrame, frame, key_names, representative_rows.items, output_name, output_column);
}

fn groupByNumericQualityIndexOn(
    comptime DeviceDataFrame: type,
    aggregation: GroupByNumericQualityIndexAggregation,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |key_name| _ = try frame.column(key_name);
    const value = try frame.column(value_name);
    return groupByNumericQualityIndexOnDispatchValue(DeviceDataFrame, aggregation, frame.allocator, frame, key_names, output_name, value.*, frame.device);
}

pub fn groupByNaNCountOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityOn(DeviceDataFrame, .nan_count, frame, key_names, value_name, output_name);
}

pub fn groupByNaNRatioOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityOn(DeviceDataFrame, .nan_ratio, frame, key_names, value_name, output_name);
}

pub fn groupByInfCountOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityOn(DeviceDataFrame, .inf_count, frame, key_names, value_name, output_name);
}

pub fn groupByInfRatioOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityOn(DeviceDataFrame, .inf_ratio, frame, key_names, value_name, output_name);
}

pub fn groupByPositiveInfCountOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityOn(DeviceDataFrame, .positive_inf_count, frame, key_names, value_name, output_name);
}

pub fn groupByPositiveInfRatioOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityOn(DeviceDataFrame, .positive_inf_ratio, frame, key_names, value_name, output_name);
}

pub fn groupByNegativeInfCountOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityOn(DeviceDataFrame, .negative_inf_count, frame, key_names, value_name, output_name);
}

pub fn groupByNegativeInfRatioOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityOn(DeviceDataFrame, .negative_inf_ratio, frame, key_names, value_name, output_name);
}

pub fn groupByFirstNaNIndexOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityIndexOn(DeviceDataFrame, .first_nan_index, frame, key_names, value_name, output_name);
}

pub fn groupByLastNaNIndexOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityIndexOn(DeviceDataFrame, .last_nan_index, frame, key_names, value_name, output_name);
}

pub fn groupByFirstInfIndexOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityIndexOn(DeviceDataFrame, .first_inf_index, frame, key_names, value_name, output_name);
}

pub fn groupByLastInfIndexOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityIndexOn(DeviceDataFrame, .last_inf_index, frame, key_names, value_name, output_name);
}

pub fn groupByFirstPositiveInfIndexOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityIndexOn(DeviceDataFrame, .first_positive_inf_index, frame, key_names, value_name, output_name);
}

pub fn groupByLastPositiveInfIndexOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityIndexOn(DeviceDataFrame, .last_positive_inf_index, frame, key_names, value_name, output_name);
}

pub fn groupByFirstNegativeInfIndexOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityIndexOn(DeviceDataFrame, .first_negative_inf_index, frame, key_names, value_name, output_name);
}

pub fn groupByLastNegativeInfIndexOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityIndexOn(DeviceDataFrame, .last_negative_inf_index, frame, key_names, value_name, output_name);
}

pub fn groupByFiniteCountOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityOn(DeviceDataFrame, .finite_count, frame, key_names, value_name, output_name);
}

pub fn groupByFiniteRatioOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityOn(DeviceDataFrame, .finite_ratio, frame, key_names, value_name, output_name);
}

pub fn groupByFirstFiniteIndexOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityIndexOn(DeviceDataFrame, .first_finite_index, frame, key_names, value_name, output_name);
}

pub fn groupByLastFiniteIndexOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityIndexOn(DeviceDataFrame, .last_finite_index, frame, key_names, value_name, output_name);
}

pub fn groupByNormalCountOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityOn(DeviceDataFrame, .normal_count, frame, key_names, value_name, output_name);
}

pub fn groupByNormalRatioOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityOn(DeviceDataFrame, .normal_ratio, frame, key_names, value_name, output_name);
}

pub fn groupByFirstNormalIndexOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityIndexOn(DeviceDataFrame, .first_normal_index, frame, key_names, value_name, output_name);
}

pub fn groupByLastNormalIndexOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityIndexOn(DeviceDataFrame, .last_normal_index, frame, key_names, value_name, output_name);
}

pub fn groupBySubnormalCountOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityOn(DeviceDataFrame, .subnormal_count, frame, key_names, value_name, output_name);
}

pub fn groupBySubnormalRatioOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityOn(DeviceDataFrame, .subnormal_ratio, frame, key_names, value_name, output_name);
}

pub fn groupByFirstSubnormalIndexOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityIndexOn(DeviceDataFrame, .first_subnormal_index, frame, key_names, value_name, output_name);
}

pub fn groupByLastSubnormalIndexOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityIndexOn(DeviceDataFrame, .last_subnormal_index, frame, key_names, value_name, output_name);
}

pub fn groupByNonFiniteCountOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityOn(DeviceDataFrame, .non_finite_count, frame, key_names, value_name, output_name);
}

pub fn groupByNonFiniteRatioOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityOn(DeviceDataFrame, .non_finite_ratio, frame, key_names, value_name, output_name);
}

pub fn groupByFirstNonFiniteIndexOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityIndexOn(DeviceDataFrame, .first_non_finite_index, frame, key_names, value_name, output_name);
}

pub fn groupByLastNonFiniteIndexOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityIndexOn(DeviceDataFrame, .last_non_finite_index, frame, key_names, value_name, output_name);
}

pub fn groupByZeroCountOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityOn(DeviceDataFrame, .zero_count, frame, key_names, value_name, output_name);
}

pub fn groupByZeroRatioOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityOn(DeviceDataFrame, .zero_ratio, frame, key_names, value_name, output_name);
}

pub fn groupByFirstZeroIndexOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityIndexOn(DeviceDataFrame, .first_zero_index, frame, key_names, value_name, output_name);
}

pub fn groupByLastZeroIndexOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityIndexOn(DeviceDataFrame, .last_zero_index, frame, key_names, value_name, output_name);
}

pub fn groupByPositiveZeroCountOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityOn(DeviceDataFrame, .positive_zero_count, frame, key_names, value_name, output_name);
}

pub fn groupByPositiveZeroRatioOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityOn(DeviceDataFrame, .positive_zero_ratio, frame, key_names, value_name, output_name);
}

pub fn groupByNegativeZeroCountOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityOn(DeviceDataFrame, .negative_zero_count, frame, key_names, value_name, output_name);
}

pub fn groupByNegativeZeroRatioOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityOn(DeviceDataFrame, .negative_zero_ratio, frame, key_names, value_name, output_name);
}

pub fn groupByFirstPositiveZeroIndexOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityIndexOn(DeviceDataFrame, .first_positive_zero_index, frame, key_names, value_name, output_name);
}

pub fn groupByLastPositiveZeroIndexOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityIndexOn(DeviceDataFrame, .last_positive_zero_index, frame, key_names, value_name, output_name);
}

pub fn groupByFirstNegativeZeroIndexOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityIndexOn(DeviceDataFrame, .first_negative_zero_index, frame, key_names, value_name, output_name);
}

pub fn groupByLastNegativeZeroIndexOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityIndexOn(DeviceDataFrame, .last_negative_zero_index, frame, key_names, value_name, output_name);
}

pub fn groupByNonZeroCountOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityOn(DeviceDataFrame, .non_zero_count, frame, key_names, value_name, output_name);
}

pub fn groupByNonZeroRatioOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityOn(DeviceDataFrame, .non_zero_ratio, frame, key_names, value_name, output_name);
}

pub fn groupByFirstNonZeroIndexOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityIndexOn(DeviceDataFrame, .first_non_zero_index, frame, key_names, value_name, output_name);
}

pub fn groupByLastNonZeroIndexOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityIndexOn(DeviceDataFrame, .last_non_zero_index, frame, key_names, value_name, output_name);
}

pub fn groupByPositiveCountOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityOn(DeviceDataFrame, .positive_count, frame, key_names, value_name, output_name);
}

pub fn groupByPositiveRatioOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityOn(DeviceDataFrame, .positive_ratio, frame, key_names, value_name, output_name);
}

pub fn groupByFirstPositiveIndexOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityIndexOn(DeviceDataFrame, .first_positive_index, frame, key_names, value_name, output_name);
}

pub fn groupByLastPositiveIndexOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityIndexOn(DeviceDataFrame, .last_positive_index, frame, key_names, value_name, output_name);
}

pub fn groupBySignBitCountOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityOn(DeviceDataFrame, .signbit_count, frame, key_names, value_name, output_name);
}

pub fn groupBySignBitRatioOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityOn(DeviceDataFrame, .signbit_ratio, frame, key_names, value_name, output_name);
}

pub fn groupByFirstSignBitIndexOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityIndexOn(DeviceDataFrame, .first_signbit_index, frame, key_names, value_name, output_name);
}

pub fn groupByLastSignBitIndexOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityIndexOn(DeviceDataFrame, .last_signbit_index, frame, key_names, value_name, output_name);
}

pub fn groupByNegativeCountOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityOn(DeviceDataFrame, .negative_count, frame, key_names, value_name, output_name);
}

pub fn groupByNegativeRatioOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityOn(DeviceDataFrame, .negative_ratio, frame, key_names, value_name, output_name);
}

pub fn groupByFirstNegativeIndexOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityIndexOn(DeviceDataFrame, .first_negative_index, frame, key_names, value_name, output_name);
}

pub fn groupByLastNegativeIndexOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByOnError!DeviceDataFrame {
    return groupByNumericQualityIndexOn(DeviceDataFrame, .last_negative_index, frame, key_names, value_name, output_name);
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
