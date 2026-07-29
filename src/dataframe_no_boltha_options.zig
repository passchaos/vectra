//! Fallback dataframe option and predicate types used when Boltha is disabled.

const array_mod = @import("array.zig");

const DeviceDType = array_mod.DType;

pub const DeviceValidityEncoding = enum {
    none,
    bool_mask,
};

pub const DeviceColumnBinaryOp = enum {
    add,
    sub,
    mul,
    div,
};

pub const DeviceColumnCompareOp = enum {
    eq,
    ne,
    gt,
    ge,
    lt,
    le,
};

pub const DeviceScalar = union(DeviceDType) {
    f32: f32,
    f64: f64,
    i8: i8,
    i16: i16,
    i32: i32,
    i64: i64,
    u8: u8,
    u16: u16,
    u32: u32,
    u64: u64,
    usize: usize,
    bool: bool,
    bf16: array_mod.BFloat16,
    f16: f16,
    c64: array_mod.Complex64,
    c128: array_mod.Complex128,
    isize: isize,

    pub fn init(comptime T: type, value: T) DeviceScalar {
        const tag = comptime DeviceDType.of(T);
        return @unionInit(DeviceScalar, @tagName(tag), value);
    }
};

pub const DeviceGroupByAggregation = enum {
    sum,
    min,
    max,
};

pub const NullPlacement = enum {
    first,
    last,
};

pub const DeviceSortOptions = struct {
    descending: bool = false,
    nulls: NullPlacement = .last,
};

pub const DeviceClipOptions = struct {
    lower: f64,
    upper: f64,
};

pub const DeviceThresholdOptions = struct {
    threshold: f64,
};

pub const DeviceRollingOptions = struct {
    window: usize,
    min_periods: ?usize = null,
};

pub const DeviceLagOptions = struct {
    periods: usize = 1,
};

pub const DeviceExpandingOptions = struct {
    min_periods: usize = 1,
};

pub const DeviceExpandingRankOptions = struct {
    min_periods: usize = 1,
    descending: bool = false,
};

pub const DeviceStandardizeOptions = struct {
    min_periods: usize = 1,
};

pub const DeviceRobustOptions = struct {
    min_periods: usize = 1,
    iqr_multiplier: f64 = 1.5,
};

pub const DeviceDrawdownOptions = struct {
    min_periods: usize = 1,
};

pub const DeviceExtremaOptions = struct {
    min_periods: usize = 1,
};

pub const DeviceTrendOptions = struct {
    periods: usize = 1,
};

pub const DeviceCrossoverOptions = struct {
    periods: usize = 1,
};

pub const DeviceBucketOptions = struct {
    buckets: usize = 10,
    lower_quantile: f64 = 0.05,
    upper_quantile: f64 = 0.95,
    min_periods: usize = 1,
};

pub const DeviceEmaOptions = struct {
    alpha: f64,
    min_periods: usize = 1,
};

pub const DeviceLinearFitOptions = struct {
    min_periods: usize = 2,
};

pub const DeviceRollingCorrelationOptions = struct {
    window: usize,
    min_periods: ?usize = null,
};

pub const DeviceRollingRankOptions = struct {
    window: usize,
    min_periods: ?usize = null,
    descending: bool = false,
};

pub const DeviceRollingRobustOptions = struct {
    window: usize,
    min_periods: ?usize = null,
    iqr_multiplier: f64 = 1.5,
};

pub const DeviceJoinOptions = struct {
    right_suffix: []const u8 = "_right",
};

pub const AsofStrategy = enum {
    previous,
    next,
    nearest,
};

pub const DeviceAsofOptions = struct {
    strategy: AsofStrategy = .previous,
    right_suffix: []const u8 = "_right",
};

pub fn Range(comptime T: type) type {
    return struct {
        min: ?T = null,
        max: ?T = null,
    };
}

pub const ParquetRangePredicate = union(DeviceDType) {
    f32: Range(f32),
    f64: Range(f64),
    i8: Range(i8),
    i16: Range(i16),
    i32: Range(i32),
    i64: Range(i64),
    u8: Range(u8),
    u16: Range(u16),
    u32: Range(u32),
    u64: Range(u64),
    usize: Range(usize),
    bool: Range(bool),
    bf16: Range(array_mod.BFloat16),
    f16: Range(f16),
    c64: Range(array_mod.Complex64),
    c128: Range(array_mod.Complex128),
    isize: Range(isize),
};

pub const DeviceParquetRangeFilter = struct {
    column: []const u8,
    predicate: ParquetRangePredicate,
};
