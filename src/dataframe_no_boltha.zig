const std = @import("std");
const array_mod = @import("array.zig");
const host_mod = @import("dataframe_no_boltha_host.zig");
const options_mod = @import("dataframe_no_boltha_options.zig");

pub const DataError = host_mod.DataError;
pub const DType = host_mod.DType;
pub const Column = host_mod.Column;
pub const ColumnDef = host_mod.ColumnDef;
pub const DataFrame = host_mod.DataFrame;
pub const dataframe = host_mod.dataframe;

pub const DeviceDType = array_mod.DType;
pub const DeviceDataError = DataError || array_mod.ArrayError;
pub const ArrowInteropError = DeviceDataError || error{FeatureUnavailable};
pub const ParquetInteropError = ArrowInteropError;

pub const DeviceValidityEncoding = options_mod.DeviceValidityEncoding;
pub const DeviceColumnBinaryOp = options_mod.DeviceColumnBinaryOp;
pub const DeviceColumnCompareOp = options_mod.DeviceColumnCompareOp;
pub const DeviceColumnLogicalOp = options_mod.DeviceColumnLogicalOp;
pub const DeviceDTypeClass = options_mod.DeviceDTypeClass;
pub const DeviceScalar = options_mod.DeviceScalar;
pub const DeviceGroupByAggregation = options_mod.DeviceGroupByAggregation;
pub const NullPlacement = options_mod.NullPlacement;
pub const DeviceSortOptions = options_mod.DeviceSortOptions;
pub const DeviceClipOptions = options_mod.DeviceClipOptions;
pub const DeviceThresholdOptions = options_mod.DeviceThresholdOptions;
pub const DeviceRollingOptions = options_mod.DeviceRollingOptions;
pub const DeviceLagOptions = options_mod.DeviceLagOptions;
pub const DeviceExpandingOptions = options_mod.DeviceExpandingOptions;
pub const DeviceExpandingRankOptions = options_mod.DeviceExpandingRankOptions;
pub const DeviceStandardizeOptions = options_mod.DeviceStandardizeOptions;
pub const DeviceRobustOptions = options_mod.DeviceRobustOptions;
pub const DeviceDrawdownOptions = options_mod.DeviceDrawdownOptions;
pub const DeviceExtremaOptions = options_mod.DeviceExtremaOptions;
pub const DeviceTrendOptions = options_mod.DeviceTrendOptions;
pub const DeviceCrossoverOptions = options_mod.DeviceCrossoverOptions;
pub const DeviceBucketOptions = options_mod.DeviceBucketOptions;
pub const DeviceEmaOptions = options_mod.DeviceEmaOptions;
pub const DeviceLinearFitOptions = options_mod.DeviceLinearFitOptions;
pub const DeviceRollingCorrelationOptions = options_mod.DeviceRollingCorrelationOptions;
pub const DeviceRollingRankOptions = options_mod.DeviceRollingRankOptions;
pub const DeviceRollingRobustOptions = options_mod.DeviceRollingRobustOptions;
pub const DeviceJoinOptions = options_mod.DeviceJoinOptions;
pub const AsofStrategy = options_mod.AsofStrategy;
pub const DeviceAsofOptions = options_mod.DeviceAsofOptions;
pub const Range = options_mod.Range;
pub const ParquetRangePredicate = options_mod.ParquetRangePredicate;
pub const DeviceParquetRangeFilter = options_mod.DeviceParquetRangeFilter;
pub const DeviceParquetNullFilter = options_mod.DeviceParquetNullFilter;

pub const DeviceColumnView = struct {
    dtype: DeviceDType,
    rows: usize,
    device: array_mod.Device,
    data_ptr: u64,
    data_nbytes: usize,
    validity_ptr: ?u64 = null,
    validity_nbytes: usize = 0,
    null_count: usize = 0,
    validity_encoding: DeviceValidityEncoding = .none,

    pub fn len(self: DeviceColumnView) usize {
        return self.rows;
    }

    pub fn dtypeName(self: DeviceColumnView) []const u8 {
        return self.dtype.name();
    }

    pub fn dtypeByteSize(self: DeviceColumnView) usize {
        return self.dtype.byteSize();
    }

    pub fn dtypeBitSize(self: DeviceColumnView) usize {
        return self.dtype.bitSize();
    }

    pub fn nullable(self: DeviceColumnView) bool {
        return self.validity_ptr != null;
    }

    pub fn hasNulls(self: DeviceColumnView) bool {
        return self.null_count != 0;
    }

    pub fn nullCount(self: DeviceColumnView) usize {
        return self.null_count;
    }

    pub fn validCount(self: DeviceColumnView) usize {
        return self.rows - self.null_count;
    }

    fn ratioFromCount(count: usize, rows: usize) f64 {
        if (rows == 0) return std.math.nan(f64);
        return @as(f64, @floatFromInt(count)) / @as(f64, @floatFromInt(rows));
    }

    pub fn nullRatio(self: DeviceColumnView) f64 {
        return ratioFromCount(self.null_count, self.rows);
    }

    pub fn validRatio(self: DeviceColumnView) f64 {
        return ratioFromCount(self.validCount(), self.rows);
    }

    pub fn dataNbytes(self: DeviceColumnView) usize {
        return self.data_nbytes;
    }

    pub fn dataMemoryUsage(self: DeviceColumnView) usize {
        return self.data_nbytes;
    }

    pub fn validityNbytes(self: DeviceColumnView) usize {
        return self.validity_nbytes;
    }

    pub fn validityMemoryUsage(self: DeviceColumnView) usize {
        return self.validity_nbytes;
    }

    pub fn totalNbytes(self: DeviceColumnView) usize {
        return self.data_nbytes + self.validity_nbytes;
    }

    pub fn memoryUsage(self: DeviceColumnView) usize {
        return self.totalNbytes();
    }

    pub fn estimatedSize(self: DeviceColumnView) usize {
        return self.totalNbytes();
    }

    pub fn isCpu(self: DeviceColumnView) bool {
        return self.device.isCpu();
    }

    pub fn isCuda(self: DeviceColumnView) bool {
        return self.device.isCuda();
    }

    pub fn isMps(self: DeviceColumnView) bool {
        return self.device.isMps();
    }

    pub fn isDeviceBacked(self: DeviceColumnView) bool {
        return !self.device.isCpu();
    }

    pub fn deviceBackendName(self: DeviceColumnView) []const u8 {
        return self.device.backendName();
    }
};

pub const DeviceDataFrameView = struct {
    allocator: std.mem.Allocator,
    names: []const []const u8,
    columns: []DeviceColumnView,
    rows: usize,
    device: array_mod.Device,

    pub fn deinit(self: *DeviceDataFrameView) void {
        if (self.columns.len != 0) self.allocator.free(self.columns);
        self.* = undefined;
    }

    pub fn height(self: DeviceDataFrameView) usize {
        return self.rows;
    }

    pub fn rowCount(self: DeviceDataFrameView) usize {
        return self.height();
    }

    pub fn nRows(self: DeviceDataFrameView) usize {
        return self.height();
    }

    pub fn width(self: DeviceDataFrameView) usize {
        return self.columns.len;
    }

    pub fn columnCount(self: DeviceDataFrameView) usize {
        return self.width();
    }

    pub fn cols(self: DeviceDataFrameView) usize {
        return self.width();
    }

    pub fn nCols(self: DeviceDataFrameView) usize {
        return self.width();
    }

    pub fn shape(self: DeviceDataFrameView) struct { rows: usize, cols: usize } {
        return .{ .rows = self.rows, .cols = self.columns.len };
    }

    pub fn columnNullCounts(self: DeviceDataFrameView, allocator: std.mem.Allocator) std.mem.Allocator.Error![]usize {
        const out = try allocator.alloc(usize, self.columns.len);
        for (self.columns, out) |column_value, *slot| slot.* = column_value.nullCount();
        return out;
    }

    pub fn columnValidCounts(self: DeviceDataFrameView, allocator: std.mem.Allocator) std.mem.Allocator.Error![]usize {
        const out = try allocator.alloc(usize, self.columns.len);
        for (self.columns, out) |column_value, *slot| slot.* = column_value.validCount();
        return out;
    }

    pub fn nullCount(self: DeviceDataFrameView) usize {
        var count: usize = 0;
        for (self.columns) |column_value| count += column_value.nullCount();
        return count;
    }

    pub fn validCount(self: DeviceDataFrameView) usize {
        var count: usize = 0;
        for (self.columns) |column_value| count += column_value.validCount();
        return count;
    }

    pub fn cellCount(self: DeviceDataFrameView) usize {
        return self.rows * self.columns.len;
    }

    fn ratioFromCount(count: usize, rows: usize) f64 {
        if (rows == 0) return std.math.nan(f64);
        return @as(f64, @floatFromInt(count)) / @as(f64, @floatFromInt(rows));
    }

    pub fn nullRatio(self: DeviceDataFrameView) f64 {
        return ratioFromCount(self.nullCount(), self.cellCount());
    }

    pub fn validRatio(self: DeviceDataFrameView) f64 {
        return ratioFromCount(self.validCount(), self.cellCount());
    }

    pub fn columnNullRatios(self: DeviceDataFrameView, allocator: std.mem.Allocator) std.mem.Allocator.Error![]f64 {
        const out = try allocator.alloc(f64, self.columns.len);
        for (self.columns, out) |column_value, *slot| slot.* = ratioFromCount(column_value.nullCount(), self.rows);
        return out;
    }

    pub fn columnValidRatios(self: DeviceDataFrameView, allocator: std.mem.Allocator) std.mem.Allocator.Error![]f64 {
        const out = try allocator.alloc(f64, self.columns.len);
        for (self.columns, out) |column_value, *slot| slot.* = ratioFromCount(column_value.validCount(), self.rows);
        return out;
    }

    pub fn columnNullableMask(self: DeviceDataFrameView, allocator: std.mem.Allocator) std.mem.Allocator.Error![]bool {
        const out = try allocator.alloc(bool, self.columns.len);
        for (self.columns, out) |column_value, *slot| slot.* = column_value.nullable();
        return out;
    }

    pub fn nullableColumnCount(self: DeviceDataFrameView) usize {
        var count: usize = 0;
        for (self.columns) |column_value| {
            if (column_value.nullable()) count += 1;
        }
        return count;
    }

    pub fn nonNullableColumnCount(self: DeviceDataFrameView) usize {
        return self.columns.len - self.nullableColumnCount();
    }

    pub fn columnHasNullsMask(self: DeviceDataFrameView, allocator: std.mem.Allocator) std.mem.Allocator.Error![]bool {
        const out = try allocator.alloc(bool, self.columns.len);
        for (self.columns, out) |column_value, *slot| slot.* = column_value.hasNulls();
        return out;
    }

    pub fn columnsWithNullsCount(self: DeviceDataFrameView) usize {
        var count: usize = 0;
        for (self.columns) |column_value| {
            if (column_value.hasNulls()) count += 1;
        }
        return count;
    }

    pub fn columnsWithoutNullsCount(self: DeviceDataFrameView) usize {
        return self.columns.len - self.columnsWithNullsCount();
    }

    pub fn columnDataNbytes(self: DeviceDataFrameView, allocator: std.mem.Allocator) std.mem.Allocator.Error![]usize {
        const out = try allocator.alloc(usize, self.columns.len);
        for (self.columns, out) |column_value, *slot| slot.* = column_value.dataNbytes();
        return out;
    }

    pub fn columnDataMemoryUsage(self: DeviceDataFrameView, allocator: std.mem.Allocator) std.mem.Allocator.Error![]usize {
        return self.columnDataNbytes(allocator);
    }

    pub fn columnValidityNbytes(self: DeviceDataFrameView, allocator: std.mem.Allocator) std.mem.Allocator.Error![]usize {
        const out = try allocator.alloc(usize, self.columns.len);
        for (self.columns, out) |column_value, *slot| slot.* = column_value.validityNbytes();
        return out;
    }

    pub fn columnValidityMemoryUsage(self: DeviceDataFrameView, allocator: std.mem.Allocator) std.mem.Allocator.Error![]usize {
        return self.columnValidityNbytes(allocator);
    }

    pub fn columnTotalNbytes(self: DeviceDataFrameView, allocator: std.mem.Allocator) std.mem.Allocator.Error![]usize {
        const out = try allocator.alloc(usize, self.columns.len);
        for (self.columns, out) |column_value, *slot| slot.* = column_value.totalNbytes();
        return out;
    }

    pub fn columnMemoryUsage(self: DeviceDataFrameView, allocator: std.mem.Allocator) std.mem.Allocator.Error![]usize {
        return self.columnTotalNbytes(allocator);
    }

    pub fn dataNbytes(self: DeviceDataFrameView) usize {
        var total: usize = 0;
        for (self.columns) |column_value| total += column_value.dataNbytes();
        return total;
    }

    pub fn dataMemoryUsage(self: DeviceDataFrameView) usize {
        return self.dataNbytes();
    }

    pub fn validityNbytes(self: DeviceDataFrameView) usize {
        var total: usize = 0;
        for (self.columns) |column_value| total += column_value.validityNbytes();
        return total;
    }

    pub fn validityMemoryUsage(self: DeviceDataFrameView) usize {
        return self.validityNbytes();
    }

    pub fn totalNbytes(self: DeviceDataFrameView) usize {
        return self.dataNbytes() + self.validityNbytes();
    }

    pub fn memoryUsage(self: DeviceDataFrameView) usize {
        return self.totalNbytes();
    }

    pub fn estimatedSize(self: DeviceDataFrameView) usize {
        return self.totalNbytes();
    }

    pub fn isEmpty(self: DeviceDataFrameView) bool {
        return self.rows == 0 or self.columns.len == 0;
    }

    pub fn isNonEmpty(self: DeviceDataFrameView) bool {
        return !self.isEmpty();
    }

    pub fn hasRows(self: DeviceDataFrameView) bool {
        return self.rows != 0;
    }

    pub fn hasColumns(self: DeviceDataFrameView) bool {
        return self.columns.len != 0;
    }

    pub fn isCpu(self: DeviceDataFrameView) bool {
        return self.device.isCpu();
    }

    pub fn isCuda(self: DeviceDataFrameView) bool {
        return self.device.isCuda();
    }

    pub fn isMps(self: DeviceDataFrameView) bool {
        return self.device.isMps();
    }

    pub fn isDeviceBacked(self: DeviceDataFrameView) bool {
        return !self.isCpu();
    }

    pub fn deviceBackendName(self: DeviceDataFrameView) []const u8 {
        return self.device.backendName();
    }

    pub fn sameDevice(self: DeviceDataFrameView, other: DeviceDataFrameView) bool {
        return self.device.sameDevice(other.device);
    }

    pub fn sameShape(self: DeviceDataFrameView, other: DeviceDataFrameView) bool {
        return self.rows == other.rows and self.columns.len == other.columns.len;
    }

    pub fn shapeEquals(self: DeviceDataFrameView, rows: usize, columns: usize) bool {
        return self.rows == rows and self.columns.len == columns;
    }

    pub fn hasShape(self: DeviceDataFrameView, rows: usize, columns: usize) bool {
        return self.shapeEquals(rows, columns);
    }

    pub fn sameHeight(self: DeviceDataFrameView, other: DeviceDataFrameView) bool {
        return self.rows == other.rows;
    }

    pub fn sameWidth(self: DeviceDataFrameView, other: DeviceDataFrameView) bool {
        return self.columns.len == other.columns.len;
    }

    pub fn columnNames(self: DeviceDataFrameView) []const []const u8 {
        return self.names;
    }

    pub fn columnLabels(self: DeviceDataFrameView) []const []const u8 {
        return self.columnNames();
    }

    pub fn columnIndex(self: DeviceDataFrameView, name: []const u8) ?usize {
        for (self.names, 0..) |existing, i| {
            if (std.mem.eql(u8, existing, name)) return i;
        }
        return null;
    }

    pub fn hasColumn(self: DeviceDataFrameView, name: []const u8) bool {
        return self.columnIndex(name) != null;
    }

    pub fn hasAllColumns(self: DeviceDataFrameView, names: []const []const u8) bool {
        for (names) |name| {
            if (!self.hasColumn(name)) return false;
        }
        return true;
    }

    pub fn hasAnyColumn(self: DeviceDataFrameView, names: []const []const u8) bool {
        for (names) |name| {
            if (self.hasColumn(name)) return true;
        }
        return false;
    }

    pub fn column(self: DeviceDataFrameView, name: []const u8) DataError!DeviceColumnView {
        const idx = self.columnIndex(name) orelse return error.ColumnNotFound;
        return self.columns[idx];
    }

    pub fn columnView(self: DeviceDataFrameView, name: []const u8) DataError!DeviceColumnView {
        return self.column(name);
    }

    pub fn columnAt(self: DeviceDataFrameView, index: usize) DeviceDataError!DeviceColumnView {
        if (index >= self.columns.len) return error.IndexOutOfBounds;
        return self.columns[index];
    }

    pub fn columnViewAt(self: DeviceDataFrameView, index: usize) DeviceDataError!DeviceColumnView {
        return self.columnAt(index);
    }

    pub fn columnNameAt(self: DeviceDataFrameView, index: usize) DeviceDataError![]const u8 {
        if (index >= self.names.len) return error.IndexOutOfBounds;
        return self.names[index];
    }

    pub fn columnDType(self: DeviceDataFrameView, name: []const u8) DataError!DeviceDType {
        const idx = self.columnIndex(name) orelse return error.ColumnNotFound;
        return self.columns[idx].dtype;
    }

    pub fn columnDTypeAt(self: DeviceDataFrameView, index: usize) DeviceDataError!DeviceDType {
        const column_value = try self.columnAt(index);
        return column_value.dtype;
    }
};

pub fn DeviceTypedColumn(comptime T: type) type {
    return struct {
        data: array_mod.Array(T),
        validity: ?array_mod.Array(bool) = null,
        null_count: usize = 0,

        pub fn deinit(self: *@This()) void {
            self.data.deinit();
            if (self.validity) |*validity| validity.deinit();
            self.* = undefined;
        }
    };
}

pub const DeviceColumn = union(DeviceDType) {
    f32: DeviceTypedColumn(f32),
    f64: DeviceTypedColumn(f64),
    i8: DeviceTypedColumn(i8),
    i16: DeviceTypedColumn(i16),
    i32: DeviceTypedColumn(i32),
    i64: DeviceTypedColumn(i64),
    u8: DeviceTypedColumn(u8),
    u16: DeviceTypedColumn(u16),
    u32: DeviceTypedColumn(u32),
    u64: DeviceTypedColumn(u64),
    usize: DeviceTypedColumn(usize),
    bool: DeviceTypedColumn(bool),
    bf16: DeviceTypedColumn(array_mod.BFloat16),
    f16: DeviceTypedColumn(f16),
    c64: DeviceTypedColumn(array_mod.Complex64),
    c128: DeviceTypedColumn(array_mod.Complex128),
    isize: DeviceTypedColumn(isize),
};

pub const DeviceColumnDef = struct {
    name: []const u8,
    data: DeviceColumn,
};

pub const DeviceColumnSchema = struct {
    name: []const u8,
    dtype: DeviceDType,
    rows: usize,
    nullable: bool,
    null_count: usize,
    valid_count: usize,
    data_nbytes: usize,
    validity_nbytes: usize,
    total_nbytes: usize,
    device: array_mod.Device,

    pub fn nullableColumn(self: @This()) bool {
        return self.nullable;
    }

    pub fn hasNulls(self: @This()) bool {
        return self.null_count != 0;
    }

    pub fn allValid(self: @This()) bool {
        return self.null_count == 0;
    }

    fn ratioFromSchemaCount(count: usize, rows: usize) f64 {
        if (rows == 0) return std.math.nan(f64);
        return @as(f64, @floatFromInt(count)) / @as(f64, @floatFromInt(rows));
    }

    pub fn nullRatio(self: @This()) f64 {
        return ratioFromSchemaCount(self.null_count, self.rows);
    }

    pub fn validRatio(self: @This()) f64 {
        return ratioFromSchemaCount(self.valid_count, self.rows);
    }

    pub fn dataMemoryUsage(self: @This()) usize {
        return self.data_nbytes;
    }

    pub fn validityMemoryUsage(self: @This()) usize {
        return self.validity_nbytes;
    }

    pub fn memoryUsage(self: @This()) usize {
        return self.total_nbytes;
    }

    pub fn estimatedSize(self: @This()) usize {
        return self.total_nbytes;
    }

    pub fn isCpu(self: @This()) bool {
        return self.device.isCpu();
    }

    pub fn isCuda(self: @This()) bool {
        return self.device.isCuda();
    }

    pub fn isMps(self: @This()) bool {
        return self.device.isMps();
    }
};

pub const DeviceLazyGroupByAggregation = enum {
    sum,
    prod,
    min,
    max,
    mean,
    first,
    last,
    first_row,
    last_row,
    nth,
    nth_row,
    nth_index,
    nth_row_index,
    n_unique,
    mode,
    mode_count,
    mode_ratio,
    mode_margin,
    mode_margin_ratio,
    entropy,
    gini_impurity,
    perplexity,
    inverse_simpson,
    simpson_concentration,
    evenness,
    gini_mean_diff,
    gini_coefficient,
    mean_abs_dev,
    mean_abs_dev_ratio,
    median,
    quantile,
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
    variance,
    stddev,
    sem,
    cv,
    fano,
    skewness,
    kurtosis,
    magnitude_variance,
    magnitude_stddev,
    magnitude_sem,
    magnitude_cv,
    magnitude_fano,
    magnitude_skewness,
    magnitude_kurtosis,
    mean_abs,
    mean_square,
    rms,
    l1_norm,
    l2_norm,
    max_abs,
    min_abs,
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
    geometric_mean,
    harmonic_mean,
    logsumexp,
    logmeanexp,
    ptp,
    midrange,
    range_coeff,
    any,
    all,
    true_count,
    false_count,
    true_ratio,
    false_ratio,
    first_true_index,
    last_true_index,
    first_false_index,
    last_false_index,
    any_valid,
    all_valid,
    any_null,
    all_null,
    valid_count,
    null_count,
    valid_ratio,
    null_ratio,
    first_valid_index,
    last_valid_index,
    first_null_index,
    last_null_index,
    nan_count,
    nan_ratio,
    inf_count,
    inf_ratio,
    positive_inf_count,
    positive_inf_ratio,
    negative_inf_count,
    negative_inf_ratio,
    first_nan_index,
    last_nan_index,
    first_inf_index,
    last_inf_index,
    first_positive_inf_index,
    last_positive_inf_index,
    first_negative_inf_index,
    last_negative_inf_index,
    finite_count,
    finite_ratio,
    first_finite_index,
    last_finite_index,
    normal_count,
    normal_ratio,
    first_normal_index,
    last_normal_index,
    subnormal_count,
    subnormal_ratio,
    first_subnormal_index,
    last_subnormal_index,
    non_finite_count,
    non_finite_ratio,
    first_non_finite_index,
    last_non_finite_index,
    zero_count,
    zero_ratio,
    first_zero_index,
    last_zero_index,
    positive_zero_count,
    positive_zero_ratio,
    negative_zero_count,
    negative_zero_ratio,
    first_positive_zero_index,
    last_positive_zero_index,
    first_negative_zero_index,
    last_negative_zero_index,
    non_zero_count,
    non_zero_ratio,
    first_non_zero_index,
    last_non_zero_index,
    positive_count,
    positive_ratio,
    first_positive_index,
    last_positive_index,
    signbit_count,
    signbit_ratio,
    first_signbit_index,
    last_signbit_index,
    negative_count,
    negative_ratio,
    first_negative_index,
    last_negative_index,
    argmin,
    argmax,
};

pub const DeviceLazyWeightedGroupByAggregation = enum {
    weighted_sum,
    weighted_product,
    weighted_weight_sum,
    weighted_positive_count,
    weighted_effective_n,
    weighted_mean,
    weighted_mean_square,
    weighted_rms,
    weighted_min,
    weighted_max,
    weighted_mean_abs,
    weighted_l1_norm,
    weighted_l2_norm,
    weighted_max_abs,
    weighted_min_abs,
    weighted_geometric_mean,
    weighted_harmonic_mean,
    weighted_logsumexp,
    weighted_logmeanexp,
    weighted_range,
    weighted_midrange,
    weighted_range_coeff,
    weighted_variance,
    weighted_stddev,
    weighted_sem,
    weighted_cv,
    weighted_fano,
    weighted_skewness,
    weighted_kurtosis,
    weighted_quantile,
    weighted_median,
    weighted_iqr,
    weighted_mad,
    weighted_trimmed_mean,
    weighted_winsorized_mean,
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

pub const DeviceLazyPairGroupByAggregation = enum {
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
    pair_count,
    covariance,
    correlation,
    beta,
};

pub const DeviceLazyWeightedPairGroupByAggregation = enum {
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

pub const DeviceLazyJoinKind = enum {
    inner,
    left,
    full,
    semi,
    anti,
};

pub const DeviceLazyOp = union(enum) {
    unsupported: void,
};

pub const DeviceLazySource = union(enum) {
    unsupported: void,
};

pub const DeviceLazyFrame = struct {
    pub fn init(_: std.mem.Allocator, _: DeviceDataFrame) DeviceDataError!DeviceLazyFrame {
        return error.FeatureUnavailable;
    }

    pub fn scanParquetBytes(_: std.mem.Allocator, _: []const u8, _: array_mod.Device) ParquetInteropError!DeviceLazyFrame {
        return error.FeatureUnavailable;
    }

    pub fn filterIsInValuesColumn(_: *DeviceLazyFrame, _: []const u8, _: DeviceColumn) DeviceDataError!void {
        return error.FeatureUnavailable;
    }

    pub fn filterNotInValuesColumn(_: *DeviceLazyFrame, _: []const u8, _: DeviceColumn) DeviceDataError!void {
        return error.FeatureUnavailable;
    }
};

pub const DeviceParquetScan = struct {
    pub fn init(_: std.mem.Allocator, _: []const u8, _: array_mod.Device) ParquetInteropError!DeviceParquetScan {
        return error.FeatureUnavailable;
    }

    pub fn deinit(_: *DeviceParquetScan) void {}

    pub fn clone(_: DeviceParquetScan) ParquetInteropError!DeviceParquetScan {
        return error.FeatureUnavailable;
    }

    pub fn lazy(_: DeviceParquetScan) ParquetInteropError!DeviceLazyFrame {
        return error.FeatureUnavailable;
    }

    pub fn select(_: *DeviceParquetScan, _: []const []const u8) ParquetInteropError!void {
        return error.FeatureUnavailable;
    }

    pub fn whereRange(_: *DeviceParquetScan, _: []const u8, _: ParquetRangePredicate) ParquetInteropError!void {
        return error.FeatureUnavailable;
    }

    pub fn whereNull(_: *DeviceParquetScan, _: []const u8, _: bool) ParquetInteropError!void {
        return error.FeatureUnavailable;
    }

    pub fn collect(_: DeviceParquetScan) ParquetInteropError!DeviceDataFrame {
        return error.FeatureUnavailable;
    }

    pub fn explain(_: DeviceParquetScan, _: std.mem.Allocator) ParquetInteropError![]u8 {
        return error.FeatureUnavailable;
    }
};

pub const DeviceDataFrame = struct {
    fn unavailableSlice(comptime T: type) DeviceDataError![]T {
        return error.FeatureUnavailable;
    }

    pub fn height(_: DeviceDataFrame) usize {
        return 0;
    }

    pub fn rowCount(_: DeviceDataFrame) usize {
        return 0;
    }

    pub fn nRows(_: DeviceDataFrame) usize {
        return 0;
    }

    pub fn width(_: DeviceDataFrame) usize {
        return 0;
    }

    pub fn columnCount(_: DeviceDataFrame) usize {
        return 0;
    }

    pub fn cols(_: DeviceDataFrame) usize {
        return 0;
    }

    pub fn nCols(_: DeviceDataFrame) usize {
        return 0;
    }

    pub fn columnLabels(_: DeviceDataFrame) []const []const u8 {
        return &.{};
    }

    pub fn columnNames(_: DeviceDataFrame) []const []const u8 {
        return &.{};
    }

    pub fn columnNamesUnique(_: DeviceDataFrame) bool {
        return true;
    }

    pub fn hasDuplicateColumnNames(_: DeviceDataFrame) bool {
        return false;
    }

    pub fn duplicateColumnNameCount(_: DeviceDataFrame) usize {
        return 0;
    }

    pub fn columnDTypes(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]DeviceDType {
        return unavailableSlice(DeviceDType);
    }

    pub fn dtypes(self: DeviceDataFrame, allocator: std.mem.Allocator) DeviceDataError![]DeviceDType {
        return self.columnDTypes(allocator);
    }

    pub fn columnDTypeNames(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![][]const u8 {
        return unavailableSlice([]const u8);
    }

    pub fn dtypeNames(self: DeviceDataFrame, allocator: std.mem.Allocator) DeviceDataError![][]const u8 {
        return self.columnDTypeNames(allocator);
    }

    pub fn columnDTypeByteSizes(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]usize {
        return unavailableSlice(usize);
    }

    pub fn columnDTypeBitSizes(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]usize {
        return unavailableSlice(usize);
    }

    pub fn columnDTypeClassMask(_: DeviceDataFrame, _: std.mem.Allocator, _: DeviceDTypeClass) DeviceDataError![]bool {
        return unavailableSlice(bool);
    }

    pub fn columnDTypeClassCount(_: DeviceDataFrame, _: DeviceDTypeClass) usize {
        return 0;
    }

    pub fn numericColumnCount(self: DeviceDataFrame) usize {
        return self.columnDTypeClassCount(.numeric);
    }

    pub fn realColumnCount(self: DeviceDataFrame) usize {
        return self.columnDTypeClassCount(.real);
    }

    pub fn floatColumnCount(self: DeviceDataFrame) usize {
        return self.columnDTypeClassCount(.float);
    }

    pub fn integerColumnCount(self: DeviceDataFrame) usize {
        return self.columnDTypeClassCount(.integer);
    }

    pub fn signedIntegerColumnCount(self: DeviceDataFrame) usize {
        return self.columnDTypeClassCount(.signed_integer);
    }

    pub fn unsignedIntegerColumnCount(self: DeviceDataFrame) usize {
        return self.columnDTypeClassCount(.unsigned_integer);
    }

    pub fn boolColumnCount(self: DeviceDataFrame) usize {
        return self.columnDTypeClassCount(.bool);
    }

    pub fn complexColumnCount(self: DeviceDataFrame) usize {
        return self.columnDTypeClassCount(.complex);
    }

    pub fn columnIsNumericMask(self: DeviceDataFrame, allocator: std.mem.Allocator) DeviceDataError![]bool {
        return self.columnDTypeClassMask(allocator, .numeric);
    }

    pub fn columnIsRealMask(self: DeviceDataFrame, allocator: std.mem.Allocator) DeviceDataError![]bool {
        return self.columnDTypeClassMask(allocator, .real);
    }

    pub fn columnIsFloatMask(self: DeviceDataFrame, allocator: std.mem.Allocator) DeviceDataError![]bool {
        return self.columnDTypeClassMask(allocator, .float);
    }

    pub fn columnIsIntegerMask(self: DeviceDataFrame, allocator: std.mem.Allocator) DeviceDataError![]bool {
        return self.columnDTypeClassMask(allocator, .integer);
    }

    pub fn columnIsSignedIntegerMask(self: DeviceDataFrame, allocator: std.mem.Allocator) DeviceDataError![]bool {
        return self.columnDTypeClassMask(allocator, .signed_integer);
    }

    pub fn columnIsUnsignedIntegerMask(self: DeviceDataFrame, allocator: std.mem.Allocator) DeviceDataError![]bool {
        return self.columnDTypeClassMask(allocator, .unsigned_integer);
    }

    pub fn columnIsBoolMask(self: DeviceDataFrame, allocator: std.mem.Allocator) DeviceDataError![]bool {
        return self.columnDTypeClassMask(allocator, .bool);
    }

    pub fn columnIsComplexMask(self: DeviceDataFrame, allocator: std.mem.Allocator) DeviceDataError![]bool {
        return self.columnDTypeClassMask(allocator, .complex);
    }

    pub fn columnNullCounts(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]usize {
        return unavailableSlice(usize);
    }

    pub fn columnValidCounts(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]usize {
        return unavailableSlice(usize);
    }

    pub fn nullCount(_: DeviceDataFrame) usize {
        return 0;
    }

    pub fn validCount(_: DeviceDataFrame) usize {
        return 0;
    }

    pub fn cellCount(_: DeviceDataFrame) usize {
        return 0;
    }

    fn ratioFromCount(count: usize, rows: usize) f64 {
        _ = count;
        if (rows == 0) return std.math.nan(f64);
        return 0.0;
    }

    pub fn nullRatio(_: DeviceDataFrame) f64 {
        return ratioFromCount(0, 0);
    }

    pub fn validRatio(_: DeviceDataFrame) f64 {
        return ratioFromCount(0, 0);
    }

    pub fn columnNullRatios(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]f64 {
        return unavailableSlice(f64);
    }

    pub fn columnValidRatios(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]f64 {
        return unavailableSlice(f64);
    }

    pub fn columnDistinctCounts(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]usize {
        return unavailableSlice(usize);
    }

    pub fn columnNUniqueCounts(self: DeviceDataFrame, allocator: std.mem.Allocator) DeviceDataError![]usize {
        return self.columnDistinctCounts(allocator);
    }

    pub fn columnNUnique(self: DeviceDataFrame, allocator: std.mem.Allocator) DeviceDataError![]usize {
        return self.columnDistinctCounts(allocator);
    }

    pub fn columnDuplicateCounts(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]usize {
        return unavailableSlice(usize);
    }

    pub fn columnRepeatedCounts(self: DeviceDataFrame, allocator: std.mem.Allocator) DeviceDataError![]usize {
        return self.columnDuplicateCounts(allocator);
    }

    pub fn columnDistinctRatios(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]f64 {
        return unavailableSlice(f64);
    }

    pub fn columnNUniqueRatios(self: DeviceDataFrame, allocator: std.mem.Allocator) DeviceDataError![]f64 {
        return self.columnDistinctRatios(allocator);
    }

    pub fn columnDuplicateRatios(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]f64 {
        return unavailableSlice(f64);
    }

    pub fn columnIsUniqueMask(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]bool {
        return unavailableSlice(bool);
    }

    pub fn columnHasDuplicatesMask(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]bool {
        return unavailableSlice(bool);
    }

    pub fn columnHasDuplicateValues(self: DeviceDataFrame, allocator: std.mem.Allocator) DeviceDataError![]bool {
        return self.columnHasDuplicatesMask(allocator);
    }

    pub fn columnNullableMask(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]bool {
        return unavailableSlice(bool);
    }

    pub fn nullableColumnCount(_: DeviceDataFrame) usize {
        return 0;
    }

    pub fn nonNullableColumnCount(_: DeviceDataFrame) usize {
        return 0;
    }

    pub fn columnHasNullsMask(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]bool {
        return unavailableSlice(bool);
    }

    pub fn columnsWithNullsCount(_: DeviceDataFrame) usize {
        return 0;
    }

    pub fn columnsWithoutNullsCount(_: DeviceDataFrame) usize {
        return 0;
    }

    pub fn columnDataNbytes(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]usize {
        return unavailableSlice(usize);
    }

    pub fn columnDataMemoryUsage(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]usize {
        return error.FeatureUnavailable;
    }

    pub fn columnValidityNbytes(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]usize {
        return unavailableSlice(usize);
    }

    pub fn columnValidityMemoryUsage(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]usize {
        return error.FeatureUnavailable;
    }

    pub fn columnTotalNbytes(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]usize {
        return unavailableSlice(usize);
    }

    pub fn columnMemoryUsage(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]usize {
        return error.FeatureUnavailable;
    }

    pub fn dataNbytes(_: DeviceDataFrame) usize {
        return 0;
    }

    pub fn dataMemoryUsage(_: DeviceDataFrame) usize {
        return 0;
    }

    pub fn validityNbytes(_: DeviceDataFrame) usize {
        return 0;
    }

    pub fn validityMemoryUsage(_: DeviceDataFrame) usize {
        return 0;
    }

    pub fn totalNbytes(_: DeviceDataFrame) usize {
        return 0;
    }

    pub fn memoryUsage(_: DeviceDataFrame) usize {
        return 0;
    }

    pub fn estimatedSize(_: DeviceDataFrame) usize {
        return 0;
    }

    pub fn columnSchemaAt(_: DeviceDataFrame, _: usize) DeviceDataError!DeviceColumnSchema {
        return error.FeatureUnavailable;
    }

    pub fn columnSchema(_: DeviceDataFrame, _: []const u8) DataError!DeviceColumnSchema {
        return error.FeatureUnavailable;
    }

    pub fn columnSchemas(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]DeviceColumnSchema {
        return error.FeatureUnavailable;
    }

    pub fn schema(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]DeviceColumnSchema {
        return error.FeatureUnavailable;
    }

    pub fn schemaSummary(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]DeviceColumnSchema {
        return error.FeatureUnavailable;
    }

    pub fn isEmpty(_: DeviceDataFrame) bool {
        return true;
    }

    pub fn isNonEmpty(_: DeviceDataFrame) bool {
        return false;
    }

    pub fn hasRows(_: DeviceDataFrame) bool {
        return false;
    }

    pub fn hasColumns(_: DeviceDataFrame) bool {
        return false;
    }

    pub fn isCpu(_: DeviceDataFrame) bool {
        return true;
    }

    pub fn isCuda(_: DeviceDataFrame) bool {
        return false;
    }

    pub fn isMps(_: DeviceDataFrame) bool {
        return false;
    }

    pub fn isDeviceBacked(_: DeviceDataFrame) bool {
        return false;
    }

    pub fn deviceBackendName(_: DeviceDataFrame) []const u8 {
        return "cpu";
    }

    pub fn sameDevice(_: DeviceDataFrame, _: DeviceDataFrame) bool {
        return true;
    }

    pub fn hasColumn(_: DeviceDataFrame, _: []const u8) bool {
        return false;
    }

    pub fn hasAllColumns(_: DeviceDataFrame, names: []const []const u8) bool {
        return names.len == 0;
    }

    pub fn hasAnyColumn(_: DeviceDataFrame, _: []const []const u8) bool {
        return false;
    }

    pub fn shape(_: DeviceDataFrame) struct { rows: usize, cols: usize } {
        return .{ .rows = 0, .cols = 0 };
    }

    pub fn sameShape(_: DeviceDataFrame, _: DeviceDataFrame) bool {
        return true;
    }

    pub fn shapeEquals(_: DeviceDataFrame, rows: usize, columns: usize) bool {
        return rows == 0 and columns == 0;
    }

    pub fn hasShape(self: DeviceDataFrame, rows: usize, columns: usize) bool {
        return self.shapeEquals(rows, columns);
    }

    pub fn sameHeight(_: DeviceDataFrame, _: DeviceDataFrame) bool {
        return true;
    }

    pub fn sameWidth(_: DeviceDataFrame, _: DeviceDataFrame) bool {
        return true;
    }

    pub fn columnIndex(_: DeviceDataFrame, _: []const u8) ?usize {
        return null;
    }

    pub fn column(_: *const DeviceDataFrame, _: []const u8) DataError!*const DeviceColumn {
        return error.ColumnNotFound;
    }

    pub fn columnAt(_: *const DeviceDataFrame, _: usize) DeviceDataError!*const DeviceColumn {
        return error.IndexOutOfBounds;
    }

    pub fn columnView(_: *const DeviceDataFrame, _: []const u8) DataError!DeviceColumnView {
        return error.ColumnNotFound;
    }

    pub fn columnViewAt(_: *const DeviceDataFrame, _: usize) DeviceDataError!DeviceColumnView {
        return error.IndexOutOfBounds;
    }

    pub fn columnNameAt(_: DeviceDataFrame, _: usize) DeviceDataError![]const u8 {
        return error.IndexOutOfBounds;
    }

    pub fn columnDType(_: DeviceDataFrame, _: []const u8) DataError!DeviceDType {
        return error.ColumnNotFound;
    }

    pub fn columnDTypeAt(_: DeviceDataFrame, _: usize) DeviceDataError!DeviceDType {
        return error.IndexOutOfBounds;
    }

    pub fn init(_: std.mem.Allocator, _: []const DeviceColumnDef) DeviceDataError!DeviceDataFrame {
        return error.FeatureUnavailable;
    }

    pub fn initEmpty(_: std.mem.Allocator, _: usize, _: array_mod.Device) DeviceDataError!DeviceDataFrame {
        return error.FeatureUnavailable;
    }

    pub fn fromDataFrame(_: std.mem.Allocator, _: DataFrame, _: array_mod.Device) DeviceDataError!DeviceDataFrame {
        return error.FeatureUnavailable;
    }

    pub fn fromParquetBytes(_: std.mem.Allocator, _: []const u8, _: array_mod.Device) ParquetInteropError!DeviceDataFrame {
        return error.FeatureUnavailable;
    }

    pub fn fromParquetBytesPruned(
        _: std.mem.Allocator,
        _: []const u8,
        _: []const u8,
        _: ParquetRangePredicate,
        _: array_mod.Device,
    ) ParquetInteropError!DeviceDataFrame {
        return error.FeatureUnavailable;
    }
};

pub fn deviceDataFrame(allocator: std.mem.Allocator, defs: []const DeviceColumnDef) DeviceDataError!DeviceDataFrame {
    return DeviceDataFrame.init(allocator, defs);
}
