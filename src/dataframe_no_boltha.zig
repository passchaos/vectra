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

    pub fn nullable(self: DeviceColumnView) bool {
        return self.validity_ptr != null;
    }

    pub fn hasNulls(self: DeviceColumnView) bool {
        return self.null_count != 0;
    }

    pub fn isDeviceBacked(self: DeviceColumnView) bool {
        return !self.device.isCpu();
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

    pub fn width(self: DeviceDataFrameView) usize {
        return self.columns.len;
    }

    pub fn shape(self: DeviceDataFrameView) struct { rows: usize, cols: usize } {
        return .{ .rows = self.rows, .cols = self.columns.len };
    }

    pub fn columnIndex(self: DeviceDataFrameView, name: []const u8) ?usize {
        for (self.names, 0..) |existing, i| {
            if (std.mem.eql(u8, existing, name)) return i;
        }
        return null;
    }

    pub fn column(self: DeviceDataFrameView, name: []const u8) DataError!DeviceColumnView {
        const idx = self.columnIndex(name) orelse return error.ColumnNotFound;
        return self.columns[idx];
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

pub const DeviceLazyGroupByAggregation = enum {
    sum,
    prod,
    min,
    max,
    mean,
    first,
    last,
    n_unique,
    mode,
    median,
    quantile,
    variance,
    stddev,
    skewness,
    kurtosis,
    any,
    all,
    true_count,
    false_count,
    true_ratio,
    false_ratio,
    valid_count,
    null_count,
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
};

pub const DeviceParquetScan = struct {
    pub fn init(_: std.mem.Allocator, _: []const u8, _: array_mod.Device) ParquetInteropError!DeviceParquetScan {
        return error.FeatureUnavailable;
    }
};

pub const DeviceDataFrame = struct {
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
        _: []const DeviceParquetRangeFilter,
        _: array_mod.Device,
    ) ParquetInteropError!DeviceDataFrame {
        return error.FeatureUnavailable;
    }
};

pub fn deviceDataFrame(allocator: std.mem.Allocator, defs: []const DeviceColumnDef) DeviceDataError!DeviceDataFrame {
    return DeviceDataFrame.init(allocator, defs);
}
