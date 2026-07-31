const array_mod = @import("array.zig");

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

pub const DeviceColumnLogicalOp = enum {
    @"and",
    @"or",
    xor,
};

pub const DeviceDTypeClass = enum {
    numeric,
    real,
    float,
    integer,
    signed_integer,
    unsigned_integer,
    complex,
    bool,

    pub fn matches(self: DeviceDTypeClass, dtype: array_mod.DType) bool {
        return switch (self) {
            .numeric => dtype.isNumeric(),
            .real => dtype.isReal(),
            .float => dtype.isFloat(),
            .integer => dtype.isInteger(),
            .signed_integer => dtype.isSigned(),
            .unsigned_integer => dtype.isUnsigned(),
            .complex => dtype.isComplex(),
            .bool => dtype.isBool(),
        };
    }
};

pub const DeviceScalar = union(array_mod.DType) {
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
        const tag = comptime array_mod.DType.of(T);
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

pub const DeviceRollingOptions = struct {
    /// Trailing, row-count based window width including the current row.
    window: usize,
    /// Minimum valid observations required to mark rolling metrics as valid.
    /// When omitted, Vectra follows strict fixed-window semantics and requires
    /// a full `window` of non-null observations.
    min_periods: ?usize = null,
};

pub const DeviceLagOptions = struct {
    /// Number of rows to look backward when deriving lag, diff, and pct-change.
    periods: usize = 1,
};

pub const DeviceExpandingOptions = struct {
    /// Minimum valid observations required to mark cumulative metrics as valid.
    min_periods: usize = 1,
};

pub const DeviceExpandingRankOptions = struct {
    /// Minimum valid observations required to mark expanding rank metrics.
    min_periods: usize = 1,
    /// Rank descending instead of ascending within the expanding prefix.
    descending: bool = false,
};

pub const DeviceStandardizeOptions = struct {
    /// Minimum valid observations required to compute global scaling metrics.
    min_periods: usize = 1,
};

pub const DeviceRobustOptions = struct {
    /// Minimum valid observations required to compute median/MAD/IQR metrics.
    min_periods: usize = 1,
    /// Tukey fence multiplier used for outlier flags and winsorization bounds.
    iqr_multiplier: f64 = 1.5,
};

pub const DeviceDrawdownOptions = struct {
    /// Minimum valid observations required to mark drawdown metrics as valid.
    min_periods: usize = 1,
};

pub const DeviceExtremaOptions = struct {
    /// Minimum valid observations required to mark extrema metrics as valid.
    min_periods: usize = 1,
};

pub const DeviceTrendOptions = struct {
    /// Number of rows to look backward for trend deltas.
    periods: usize = 1,
};

pub const DeviceCrossoverOptions = struct {
    /// Number of rows to look backward when detecting spread sign crosses.
    periods: usize = 1,
};

pub const DeviceBucketOptions = struct {
    /// Number of equal-frequency buckets to assign, zero-based.
    buckets: usize = 10,
    /// Lower-tail quantile threshold for the tail flag.
    lower_quantile: f64 = 0.05,
    /// Upper-tail quantile threshold for the tail flag.
    upper_quantile: f64 = 0.95,
    /// Minimum valid observations required to compute distribution features.
    min_periods: usize = 1,
};

pub const DeviceEmaOptions = struct {
    /// Exponential smoothing factor in (0, 1].
    alpha: f64,
    /// Minimum valid observations required before EMA-derived metrics are valid.
    min_periods: usize = 1,
};

pub const DeviceLinearFitOptions = struct {
    /// Minimum valid observation pairs required to compute fit diagnostics.
    min_periods: usize = 2,
};

pub const DeviceClipOptions = struct {
    lower: f64,
    upper: f64,
};

pub const DeviceThresholdOptions = struct {
    threshold: f64,
};

pub const DeviceRollingCorrelationOptions = struct {
    /// Trailing, row-count based window width including the current row.
    window: usize,
    /// Minimum valid observation pairs required to mark correlation metrics.
    /// When omitted, Vectra requires a full `window` of valid pairs.
    min_periods: ?usize = null,
};

pub const DeviceRollingRankOptions = struct {
    /// Trailing, row-count based window width including the current row.
    window: usize,
    /// Minimum valid observations required to mark rolling rank metrics.
    /// When omitted, Vectra requires a full `window` of valid observations.
    min_periods: ?usize = null,
    /// Rank descending instead of ascending within each trailing window.
    descending: bool = false,
};

pub const DeviceRollingRobustOptions = struct {
    /// Trailing, row-count based window width including the current row.
    window: usize,
    /// Minimum valid observations required to mark robust metrics.
    /// When omitted, Vectra requires a full `window` of valid observations.
    min_periods: ?usize = null,
    /// Tukey fence multiplier used for outlier flags and winsorization bounds.
    iqr_multiplier: f64 = 1.5,
};

pub const ParquetRangePredicate = union(array_mod.DType) {
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

pub fn Range(comptime T: type) type {
    return struct {
        min: ?T = null,
        max: ?T = null,
    };
}
