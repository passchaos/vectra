const std = @import("std");
const series_mod = @import("series.zig");
const array_mod = @import("array.zig");
const boltha = @import("boltha");

pub const DataError = series_mod.DataError;
pub const DType = enum { f64, i64, bool, string };

pub const Column = union(DType) {
    f64: []const f64,
    i64: []const i64,
    bool: []const bool,
    string: []const []const u8,

    pub fn len(self: Column) usize {
        return switch (self) {
            .f64 => |v| v.len,
            .i64 => |v| v.len,
            .bool => |v| v.len,
            .string => |v| v.len,
        };
    }

    pub fn dtype(self: Column) DType {
        return std.meta.activeTag(self);
    }
};

pub const ColumnDef = struct {
    name: []const u8,
    data: Column,
};

pub const DeviceDType = array_mod.DType;
pub const DeviceDataError = DataError || array_mod.ArrayError;
pub const ArrowInteropError = DeviceDataError || boltha.arrow.ArrayError || boltha.arrow.RecordBatchError || boltha.arrow.TableError;
pub const ParquetInteropError = ArrowInteropError || boltha.parquet.SimpleError;

/// Vectra's portable validity representation for device dataframe columns.
///
/// cuDF uses Arrow-compatible packed bitmasks.  Vectra starts one abstraction
/// level higher and keeps validity as a `Array(bool)` so the dataframe wrapper
/// can work across CPU, CUDA, and MPS storage immediately.  A future Arrow ABI
/// bridge can add a packed-bitmask view without changing the owning column/table
/// shape introduced here.
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

pub fn Range(comptime T: type) type {
    return struct {
        min: ?T = null,
        max: ?T = null,
    };
}

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

/// Non-owning table metadata modeled after cuDF's `table_view`.
///
/// The view does not own column storage or names; it only owns the small
/// `columns` metadata slice allocated by `DeviceDataFrame.view()`.  Users may pass
/// this compact description to backend bridges without copying column buffers.
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
        const Self = @This();
        pub const Scalar = T;

        values: array_mod.Array(T),
        validity: ?array_mod.Array(bool) = null,
        null_count: usize = 0,

        pub fn init(values: array_mod.Array(T), validity: ?array_mod.Array(bool), null_count: usize) array_mod.ArrayError!Self {
            if (values.shape.len != 1) return error.InvalidShape;
            if (validity) |mask| {
                if (mask.shape.len != 1 or mask.shape[0] != values.shape[0]) return error.ShapeMismatch;
                if (!mask.device.sameDevice(values.device)) return error.InvalidDevice;
            }
            return .{ .values = values, .validity = validity, .null_count = null_count };
        }

        pub fn fromSlice(allocator: std.mem.Allocator, values: []const T, device_value: array_mod.Device) array_mod.ArrayError!Self {
            const value_array = try array_mod.Array(T).fromSliceOn(allocator, values, &.{values.len}, device_value);
            errdefer {
                var cleanup = value_array;
                cleanup.deinit();
            }
            return Self.init(value_array, null, 0);
        }

        pub fn fromSliceWithValidity(
            allocator: std.mem.Allocator,
            values: []const T,
            validity_values: []const bool,
            device_value: array_mod.Device,
        ) array_mod.ArrayError!Self {
            if (validity_values.len != values.len) return error.ShapeMismatch;
            const value_array = try array_mod.Array(T).fromSliceOn(allocator, values, &.{values.len}, device_value);
            errdefer {
                var cleanup = value_array;
                cleanup.deinit();
            }
            const validity_array = try array_mod.Array(bool).fromSliceOn(allocator, validity_values, &.{validity_values.len}, device_value);
            errdefer {
                var cleanup = validity_array;
                cleanup.deinit();
            }
            return Self.init(value_array, validity_array, countNulls(validity_values));
        }

        pub fn deinit(self: *Self) void {
            self.values.deinit();
            if (self.validity) |*mask| mask.deinit();
            self.* = undefined;
        }

        pub fn len(self: Self) usize {
            return self.values.shape[0];
        }

        pub fn dtype(self: Self) DeviceDType {
            _ = self;
            return DeviceDType.of(T);
        }

        pub fn device(self: Self) array_mod.Device {
            return self.values.device;
        }

        pub fn nullable(self: Self) bool {
            return self.validity != null;
        }

        pub fn hasNulls(self: Self) bool {
            return self.null_count != 0;
        }

        pub fn dataNbytes(self: Self) usize {
            return self.values.nbytes();
        }

        pub fn view(self: Self) DeviceColumnView {
            const validity_ptr: ?u64 = if (self.validity) |mask| @intFromPtr(mask.dataPtr()) else null;
            return .{
                .dtype = DeviceDType.of(T),
                .rows = self.len(),
                .device = self.device(),
                .data_ptr = @intFromPtr(self.values.dataPtr()),
                .data_nbytes = self.values.nbytes(),
                .validity_ptr = validity_ptr,
                .validity_nbytes = if (self.validity) |mask| mask.nbytes() else 0,
                .null_count = self.null_count,
                .validity_encoding = if (validity_ptr != null) .bool_mask else .none,
            };
        }

        pub fn clone(self: Self) array_mod.ArrayError!Self {
            var values = try self.values.clone();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn to(self: Self, device_value: array_mod.Device) array_mod.ArrayError!Self {
            var values = try self.values.to(device_value);
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.to(device_value);
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn cpu(self: Self) array_mod.ArrayError!Self {
            return self.to(.cpu);
        }

        pub fn cuda(self: Self, index: usize) array_mod.ArrayError!Self {
            return self.to(array_mod.Device.cuda(index));
        }

        pub fn mps(self: Self, index: usize) array_mod.ArrayError!Self {
            return self.to(array_mod.Device.mps(index));
        }

        pub fn sliceRows(self: Self, start: usize, stop: usize) array_mod.ArrayError!Self {
            const end = @min(stop, self.len());
            const begin = @min(start, end);
            var values = try sliceArray1d(T, self.values, begin, end);
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try sliceArray1d(bool, mask, begin, end);
            const nulls = if (validity) |mask| try countNullsInArray(mask) else 0;
            return .{ .values = values, .validity = validity, .null_count = nulls };
        }

        pub fn take(self: Self, row_indices: []const usize) array_mod.ArrayError!Self {
            var values = try takeArray1d(T, self.values, row_indices);
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try takeArray1d(bool, mask, row_indices);
            const nulls = if (validity) |mask| try countNullsInArray(mask) else 0;
            return .{ .values = values, .validity = validity, .null_count = nulls };
        }

        pub fn takeOptional(self: Self, row_indices: []const ?usize) array_mod.ArrayError!Self {
            const host_values = try self.values.toOwnedSlice(self.values.allocator);
            defer self.values.allocator.free(host_values);
            const maybe_validity = try validityValues(self, self.values.allocator);
            defer if (maybe_validity) |validity| self.values.allocator.free(validity);

            const values = try self.values.allocator.alloc(T, row_indices.len);
            defer self.values.allocator.free(values);
            const validity_values = try self.values.allocator.alloc(bool, row_indices.len);
            defer self.values.allocator.free(validity_values);
            for (row_indices, values, validity_values) |maybe_idx, *value_slot, *valid_slot| {
                if (maybe_idx) |idx| {
                    if (idx >= host_values.len) return error.IndexOutOfBounds;
                    value_slot.* = host_values[idx];
                    valid_slot.* = if (maybe_validity) |validity| validity[idx] else true;
                } else {
                    value_slot.* = zeroValue(T);
                    valid_slot.* = false;
                }
            }
            var value_array = try array_mod.Array(T).fromSliceOn(self.values.allocator, values, &.{row_indices.len}, self.device());
            errdefer value_array.deinit();
            if (countNulls(validity_values) == 0) return .{ .values = value_array, .validity = null, .null_count = 0 };
            var validity_array = try array_mod.Array(bool).fromSliceOn(self.values.allocator, validity_values, &.{row_indices.len}, self.device());
            errdefer validity_array.deinit();
            return .{ .values = value_array, .validity = validity_array, .null_count = countNulls(validity_values) };
        }

        pub fn filter(self: Self, mask: []const bool) array_mod.ArrayError!Self {
            if (mask.len != self.len()) return error.ShapeMismatch;
            const row_indices = try rowIndicesFromMask(self.values.allocator, mask);
            defer self.values.allocator.free(row_indices);
            return self.take(row_indices);
        }

        pub fn binary(self: Self, other: Self, op: DeviceColumnBinaryOp) array_mod.ArrayError!Self {
            if (comptime T == bool) return error.TypeUnsupported;
            try requireCompatibleColumnArrays(T, self.values, other.values);
            var values = switch (op) {
                .add => try self.values.add(other.values),
                .sub => try self.values.sub(other.values),
                .mul => try self.values.mul(other.values),
                .div => if (comptime isIntegerColumnType(T)) return error.TypeUnsupported else try self.values.div(other.values),
            };
            errdefer values.deinit();
            var validity = try combineValidityMasks(self.values.allocator, self.validity, other.validity, self.len(), self.device());
            errdefer if (validity) |*mask| mask.deinit();
            const nulls = if (validity) |mask| try countNullsInArray(mask) else 0;
            return .{ .values = values, .validity = validity, .null_count = nulls };
        }

        pub fn binaryScalar(self: Self, scalar: T, op: DeviceColumnBinaryOp) array_mod.ArrayError!Self {
            if (comptime T == bool) return error.TypeUnsupported;
            var values = switch (op) {
                .add => try self.values.addScalar(scalar),
                .sub => try self.values.subScalar(scalar),
                .mul => try self.values.mulScalar(scalar),
                .div => if (comptime isIntegerColumnType(T)) return error.TypeUnsupported else try self.values.divScalar(scalar),
            };
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn compare(self: Self, other: Self, op: DeviceColumnCompareOp) array_mod.ArrayError!DeviceTypedColumn(bool) {
            try requireCompatibleColumnArrays(T, self.values, other.values);
            if (comptime !isOrderedColumnType(T)) {
                var values = switch (op) {
                    .eq => try self.values.equal(other.values),
                    .ne => try self.values.notEqual(other.values),
                    .gt, .ge, .lt, .le => return error.TypeUnsupported,
                };
                errdefer values.deinit();
                var validity = try combineValidityMasks(self.values.allocator, self.validity, other.validity, self.len(), self.device());
                errdefer if (validity) |*mask| mask.deinit();
                const nulls = if (validity) |mask| try countNullsInArray(mask) else 0;
                return .{ .values = values, .validity = validity, .null_count = nulls };
            }
            var values = switch (op) {
                .eq => try self.values.equal(other.values),
                .ne => try self.values.notEqual(other.values),
                .gt => try self.values.greater(other.values),
                .ge => try self.values.greaterEqual(other.values),
                .lt => try self.values.less(other.values),
                .le => try self.values.lessEqual(other.values),
            };
            errdefer values.deinit();
            var validity = try combineValidityMasks(self.values.allocator, self.validity, other.validity, self.len(), self.device());
            errdefer if (validity) |*mask| mask.deinit();
            const nulls = if (validity) |mask| try countNullsInArray(mask) else 0;
            return .{ .values = values, .validity = validity, .null_count = nulls };
        }

        pub fn compareScalar(self: Self, scalar: T, op: DeviceColumnCompareOp) array_mod.ArrayError!DeviceTypedColumn(bool) {
            if (comptime !isOrderedColumnType(T)) {
                var values = switch (op) {
                    .eq => try self.values.equalScalar(scalar),
                    .ne => try self.values.notEqualScalar(scalar),
                    .gt, .ge, .lt, .le => return error.TypeUnsupported,
                };
                errdefer values.deinit();
                var validity: ?array_mod.Array(bool) = null;
                errdefer if (validity) |*mask| mask.deinit();
                if (self.validity) |mask| validity = try mask.clone();
                return .{ .values = values, .validity = validity, .null_count = self.null_count };
            }
            var values = switch (op) {
                .eq => try self.values.equalScalar(scalar),
                .ne => try self.values.notEqualScalar(scalar),
                .gt => try self.values.greaterScalar(scalar),
                .ge => try self.values.greaterEqualScalar(scalar),
                .lt => try self.values.lessScalar(scalar),
                .le => try self.values.lessEqualScalar(scalar),
            };
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn toOwnedSlice(self: Self, allocator: std.mem.Allocator) array_mod.ArrayError![]T {
            return self.values.toOwnedSlice(allocator);
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

    pub fn fromSlice(comptime T: type, allocator: std.mem.Allocator, values: []const T, device_value: array_mod.Device) array_mod.ArrayError!DeviceColumn {
        const tag = comptime DeviceDType.of(T);
        const typed = try DeviceTypedColumn(T).fromSlice(allocator, values, device_value);
        return @unionInit(DeviceColumn, @tagName(tag), typed);
    }

    pub fn fromSliceWithValidity(
        comptime T: type,
        allocator: std.mem.Allocator,
        values: []const T,
        validity_values: []const bool,
        device_value: array_mod.Device,
    ) array_mod.ArrayError!DeviceColumn {
        const tag = comptime DeviceDType.of(T);
        const typed = try DeviceTypedColumn(T).fromSliceWithValidity(allocator, values, validity_values, device_value);
        return @unionInit(DeviceColumn, @tagName(tag), typed);
    }

    pub fn deinit(self: *DeviceColumn) void {
        switch (self.*) {
            inline else => |*typed| typed.deinit(),
        }
        self.* = undefined;
    }

    pub fn len(self: DeviceColumn) usize {
        return switch (self) {
            inline else => |typed| typed.len(),
        };
    }

    pub fn dtype(self: DeviceColumn) DeviceDType {
        return std.meta.activeTag(self);
    }

    pub fn device(self: DeviceColumn) array_mod.Device {
        return switch (self) {
            inline else => |typed| typed.device(),
        };
    }

    pub fn nullable(self: DeviceColumn) bool {
        return switch (self) {
            inline else => |typed| typed.nullable(),
        };
    }

    pub fn hasNulls(self: DeviceColumn) bool {
        return switch (self) {
            inline else => |typed| typed.hasNulls(),
        };
    }

    pub fn nullCount(self: DeviceColumn) usize {
        return switch (self) {
            inline else => |typed| typed.null_count,
        };
    }

    pub fn dataNbytes(self: DeviceColumn) usize {
        return switch (self) {
            inline else => |typed| typed.dataNbytes(),
        };
    }

    pub fn view(self: DeviceColumn) DeviceColumnView {
        return switch (self) {
            inline else => |typed| typed.view(),
        };
    }

    pub fn clone(self: DeviceColumn) array_mod.ArrayError!DeviceColumn {
        return switch (self) {
            inline else => |typed, tag| @unionInit(DeviceColumn, @tagName(tag), try typed.clone()),
        };
    }

    pub fn to(self: DeviceColumn, device_value: array_mod.Device) array_mod.ArrayError!DeviceColumn {
        return switch (self) {
            inline else => |typed, tag| @unionInit(DeviceColumn, @tagName(tag), try typed.to(device_value)),
        };
    }

    pub fn cpu(self: DeviceColumn) array_mod.ArrayError!DeviceColumn {
        return self.to(.cpu);
    }

    pub fn cuda(self: DeviceColumn, index: usize) array_mod.ArrayError!DeviceColumn {
        return self.to(array_mod.Device.cuda(index));
    }

    pub fn mps(self: DeviceColumn, index: usize) array_mod.ArrayError!DeviceColumn {
        return self.to(array_mod.Device.mps(index));
    }

    pub fn sliceRows(self: DeviceColumn, start: usize, stop: usize) array_mod.ArrayError!DeviceColumn {
        return switch (self) {
            inline else => |typed, tag| @unionInit(DeviceColumn, @tagName(tag), try typed.sliceRows(start, stop)),
        };
    }

    pub fn take(self: DeviceColumn, row_indices: []const usize) array_mod.ArrayError!DeviceColumn {
        return switch (self) {
            inline else => |typed, tag| @unionInit(DeviceColumn, @tagName(tag), try typed.take(row_indices)),
        };
    }

    pub fn takeOptional(self: DeviceColumn, row_indices: []const ?usize) array_mod.ArrayError!DeviceColumn {
        return switch (self) {
            inline else => |typed, tag| @unionInit(DeviceColumn, @tagName(tag), try typed.takeOptional(row_indices)),
        };
    }

    pub fn filter(self: DeviceColumn, mask: []const bool) array_mod.ArrayError!DeviceColumn {
        return switch (self) {
            inline else => |typed, tag| @unionInit(DeviceColumn, @tagName(tag), try typed.filter(mask)),
        };
    }

    pub fn argsort(self: DeviceColumn, allocator: std.mem.Allocator, options_value: DeviceSortOptions) DeviceDataError![]usize {
        return switch (self) {
            .bool => |typed| try argsortTypedColumn(bool, typed, allocator, options_value),
            .i8 => |typed| try argsortTypedColumn(i8, typed, allocator, options_value),
            .i16 => |typed| try argsortTypedColumn(i16, typed, allocator, options_value),
            .i32 => |typed| try argsortTypedColumn(i32, typed, allocator, options_value),
            .i64 => |typed| try argsortTypedColumn(i64, typed, allocator, options_value),
            .u8 => |typed| try argsortTypedColumn(u8, typed, allocator, options_value),
            .u16 => |typed| try argsortTypedColumn(u16, typed, allocator, options_value),
            .u32 => |typed| try argsortTypedColumn(u32, typed, allocator, options_value),
            .u64 => |typed| try argsortTypedColumn(u64, typed, allocator, options_value),
            .usize => |typed| try argsortTypedColumn(usize, typed, allocator, options_value),
            .isize => |typed| try argsortTypedColumn(isize, typed, allocator, options_value),
            .f16 => |typed| try argsortTypedColumn(f16, typed, allocator, options_value),
            .f32 => |typed| try argsortTypedColumn(f32, typed, allocator, options_value),
            .f64 => |typed| try argsortTypedColumn(f64, typed, allocator, options_value),
            .bf16, .c64, .c128 => error.TypeUnsupported,
        };
    }

    pub fn binary(self: DeviceColumn, other: DeviceColumn, op: DeviceColumnBinaryOp) array_mod.ArrayError!DeviceColumn {
        if (self.dtype() != other.dtype()) return error.TypeUnsupported;
        if (!self.device().sameDevice(other.device())) return error.InvalidDevice;
        return switch (self) {
            inline else => |typed, tag| @unionInit(DeviceColumn, @tagName(tag), try typed.binary(@field(other, @tagName(tag)), op)),
        };
    }

    pub fn add(self: DeviceColumn, other: DeviceColumn) array_mod.ArrayError!DeviceColumn {
        return self.binary(other, .add);
    }

    pub fn sub(self: DeviceColumn, other: DeviceColumn) array_mod.ArrayError!DeviceColumn {
        return self.binary(other, .sub);
    }

    pub fn mul(self: DeviceColumn, other: DeviceColumn) array_mod.ArrayError!DeviceColumn {
        return self.binary(other, .mul);
    }

    pub fn div(self: DeviceColumn, other: DeviceColumn) array_mod.ArrayError!DeviceColumn {
        return self.binary(other, .div);
    }

    pub fn binaryScalar(self: DeviceColumn, comptime T: type, scalar: T, op: DeviceColumnBinaryOp) array_mod.ArrayError!DeviceColumn {
        if (self.dtype() != DeviceDType.of(T)) return error.TypeUnsupported;
        const tag = comptime DeviceDType.of(T);
        return @unionInit(DeviceColumn, @tagName(tag), try @field(self, @tagName(tag)).binaryScalar(scalar, op));
    }

    pub fn addScalar(self: DeviceColumn, comptime T: type, scalar: T) array_mod.ArrayError!DeviceColumn {
        return self.binaryScalar(T, scalar, .add);
    }

    pub fn subScalar(self: DeviceColumn, comptime T: type, scalar: T) array_mod.ArrayError!DeviceColumn {
        return self.binaryScalar(T, scalar, .sub);
    }

    pub fn mulScalar(self: DeviceColumn, comptime T: type, scalar: T) array_mod.ArrayError!DeviceColumn {
        return self.binaryScalar(T, scalar, .mul);
    }

    pub fn divScalar(self: DeviceColumn, comptime T: type, scalar: T) array_mod.ArrayError!DeviceColumn {
        return self.binaryScalar(T, scalar, .div);
    }

    pub fn compare(self: DeviceColumn, other: DeviceColumn, op: DeviceColumnCompareOp) array_mod.ArrayError!DeviceColumn {
        if (self.dtype() != other.dtype()) return error.TypeUnsupported;
        if (!self.device().sameDevice(other.device())) return error.InvalidDevice;
        return switch (self) {
            .bool => |typed| .{ .bool = try typed.compare(other.bool, op) },
            .i8 => |typed| .{ .bool = try typed.compare(other.i8, op) },
            .i16 => |typed| .{ .bool = try typed.compare(other.i16, op) },
            .i32 => |typed| .{ .bool = try typed.compare(other.i32, op) },
            .i64 => |typed| .{ .bool = try typed.compare(other.i64, op) },
            .u8 => |typed| .{ .bool = try typed.compare(other.u8, op) },
            .u16 => |typed| .{ .bool = try typed.compare(other.u16, op) },
            .u32 => |typed| .{ .bool = try typed.compare(other.u32, op) },
            .u64 => |typed| .{ .bool = try typed.compare(other.u64, op) },
            .usize => |typed| .{ .bool = try typed.compare(other.usize, op) },
            .isize => |typed| .{ .bool = try typed.compare(other.isize, op) },
            .f16 => |typed| .{ .bool = try typed.compare(other.f16, op) },
            .f32 => |typed| .{ .bool = try typed.compare(other.f32, op) },
            .f64 => |typed| .{ .bool = try typed.compare(other.f64, op) },
            .bf16, .c64, .c128 => error.TypeUnsupported,
        };
    }

    pub fn equal(self: DeviceColumn, other: DeviceColumn) array_mod.ArrayError!DeviceColumn {
        return self.compare(other, .eq);
    }

    pub fn notEqual(self: DeviceColumn, other: DeviceColumn) array_mod.ArrayError!DeviceColumn {
        return self.compare(other, .ne);
    }

    pub fn greater(self: DeviceColumn, other: DeviceColumn) array_mod.ArrayError!DeviceColumn {
        return self.compare(other, .gt);
    }

    pub fn greaterEqual(self: DeviceColumn, other: DeviceColumn) array_mod.ArrayError!DeviceColumn {
        return self.compare(other, .ge);
    }

    pub fn less(self: DeviceColumn, other: DeviceColumn) array_mod.ArrayError!DeviceColumn {
        return self.compare(other, .lt);
    }

    pub fn lessEqual(self: DeviceColumn, other: DeviceColumn) array_mod.ArrayError!DeviceColumn {
        return self.compare(other, .le);
    }

    pub fn compareScalar(self: DeviceColumn, comptime T: type, scalar: T, op: DeviceColumnCompareOp) array_mod.ArrayError!DeviceColumn {
        if (self.dtype() != DeviceDType.of(T)) return error.TypeUnsupported;
        const tag = comptime DeviceDType.of(T);
        return .{ .bool = try @field(self, @tagName(tag)).compareScalar(scalar, op) };
    }

    pub fn equalScalar(self: DeviceColumn, comptime T: type, scalar: T) array_mod.ArrayError!DeviceColumn {
        return self.compareScalar(T, scalar, .eq);
    }

    pub fn notEqualScalar(self: DeviceColumn, comptime T: type, scalar: T) array_mod.ArrayError!DeviceColumn {
        return self.compareScalar(T, scalar, .ne);
    }

    pub fn greaterScalar(self: DeviceColumn, comptime T: type, scalar: T) array_mod.ArrayError!DeviceColumn {
        return self.compareScalar(T, scalar, .gt);
    }

    pub fn greaterEqualScalar(self: DeviceColumn, comptime T: type, scalar: T) array_mod.ArrayError!DeviceColumn {
        return self.compareScalar(T, scalar, .ge);
    }

    pub fn lessScalar(self: DeviceColumn, comptime T: type, scalar: T) array_mod.ArrayError!DeviceColumn {
        return self.compareScalar(T, scalar, .lt);
    }

    pub fn lessEqualScalar(self: DeviceColumn, comptime T: type, scalar: T) array_mod.ArrayError!DeviceColumn {
        return self.compareScalar(T, scalar, .le);
    }

    pub fn arrowDataType(self: DeviceColumn) ArrowInteropError!boltha.arrow.DataType {
        return deviceDTypeToArrowDataType(self.dtype());
    }

    pub fn toArrowArray(self: DeviceColumn, allocator: std.mem.Allocator) ArrowInteropError!boltha.arrow.AnyArray {
        return switch (self) {
            .bool => |typed| try boolColumnToArrow(typed, allocator),
            .i8 => |typed| try primitiveColumnToArrow(i8, "int8", typed, allocator),
            .i16 => |typed| try primitiveColumnToArrow(i16, "int16", typed, allocator),
            .i32 => |typed| try primitiveColumnToArrow(i32, "int32", typed, allocator),
            .i64 => |typed| try primitiveColumnToArrow(i64, "int64", typed, allocator),
            .u8 => |typed| try primitiveColumnToArrow(u8, "uint8", typed, allocator),
            .u16 => |typed| try primitiveColumnToArrow(u16, "uint16", typed, allocator),
            .u32 => |typed| try primitiveColumnToArrow(u32, "uint32", typed, allocator),
            .u64 => |typed| try primitiveColumnToArrow(u64, "uint64", typed, allocator),
            .f16 => |typed| try primitiveColumnToArrow(f16, "float16", typed, allocator),
            .f32 => |typed| try primitiveColumnToArrow(f32, "float32", typed, allocator),
            .f64 => |typed| try primitiveColumnToArrow(f64, "float64", typed, allocator),
            .usize => |typed| try indexColumnToArrow(usize, typed, allocator),
            .isize => |typed| try indexColumnToArrow(isize, typed, allocator),
            .bf16, .c64, .c128 => error.TypeUnsupported,
        };
    }
};

pub const DeviceColumnDef = struct {
    name: []const u8,
    data: DeviceColumn,
};

pub const DeviceLazyGroupByAggregation = enum {
    sum,
    min,
    max,
    mean,
};

pub const DeviceLazyJoinKind = enum {
    inner,
    left,
    full,
    semi,
    anti,
};

pub const DeviceLazyOp = union(enum) {
    select: [][]const u8,
    with_column_binary: struct {
        name: []const u8,
        lhs_name: []const u8,
        rhs_name: []const u8,
        op: DeviceColumnBinaryOp,
    },
    with_column_scalar: struct {
        name: []const u8,
        input_name: []const u8,
        op: DeviceColumnBinaryOp,
        scalar: DeviceScalar,
    },
    with_column_compare: struct {
        name: []const u8,
        lhs_name: []const u8,
        rhs_name: []const u8,
        op: DeviceColumnCompareOp,
    },
    with_column_compare_scalar: struct {
        name: []const u8,
        input_name: []const u8,
        op: DeviceColumnCompareOp,
        scalar: DeviceScalar,
    },
    filter_mask: DeviceColumn,
    filter_scalar: struct {
        name: []const u8,
        op: DeviceColumnCompareOp,
        scalar: DeviceScalar,
    },
    group_by_count: struct {
        key_name: []const u8,
        output_name: []const u8,
    },
    group_by_value: struct {
        key_name: []const u8,
        value_name: []const u8,
        output_name: []const u8,
        aggregation: DeviceLazyGroupByAggregation,
    },
    group_by_stats: struct {
        key_name: []const u8,
        value_name: []const u8,
        output_prefix: []const u8,
    },
    group_by_stats_on: struct {
        key_names: [][]const u8,
        value_name: []const u8,
        output_prefix: []const u8,
    },
    group_by_profile: struct {
        key_name: []const u8,
        value_name: []const u8,
        output_prefix: []const u8,
    },
    group_by_profile_on: struct {
        key_names: [][]const u8,
        value_name: []const u8,
        output_prefix: []const u8,
    },
    join_on: struct {
        kind: DeviceLazyJoinKind,
        right: DeviceDataFrame,
        left_key_names: [][]const u8,
        right_key_names: [][]const u8,
        options: DeviceJoinOptions,
    },
    asof_join: struct {
        right: DeviceDataFrame,
        left_key_name: []const u8,
        right_key_name: []const u8,
        options: DeviceAsofOptions,
    },
    concat_rows: DeviceDataFrame,
    distinct_rows,
    distinct_on: [][]const u8,
    sort_by: struct {
        name: []const u8,
        options: DeviceSortOptions,
    },
    top_k: struct {
        name: []const u8,
        options: DeviceSortOptions,
        k: usize,
    },
    rank_profile_by: struct {
        name: []const u8,
        output_prefix: []const u8,
        options: DeviceSortOptions,
    },
    rolling_profile: struct {
        name: []const u8,
        output_prefix: []const u8,
        options: DeviceRollingOptions,
    },
    rolling_range_profile: struct {
        name: []const u8,
        output_prefix: []const u8,
        options: DeviceRollingOptions,
    },
    rolling_normalize_profile: struct {
        name: []const u8,
        output_prefix: []const u8,
        options: DeviceRollingOptions,
    },
    rolling_quantile_profile: struct {
        name: []const u8,
        output_prefix: []const u8,
        options: DeviceRollingOptions,
    },
    lag_profile: struct {
        name: []const u8,
        output_prefix: []const u8,
        options: DeviceLagOptions,
    },
    lead_profile: struct {
        name: []const u8,
        output_prefix: []const u8,
        options: DeviceLagOptions,
    },
    clip_profile: struct {
        name: []const u8,
        output_prefix: []const u8,
        options: DeviceClipOptions,
    },
    threshold_profile: struct {
        name: []const u8,
        output_prefix: []const u8,
        options: DeviceThresholdOptions,
    },
    expanding_profile: struct {
        name: []const u8,
        output_prefix: []const u8,
        options: DeviceExpandingOptions,
    },
    standardize_profile: struct {
        name: []const u8,
        output_prefix: []const u8,
        options: DeviceStandardizeOptions,
    },
    robust_profile: struct {
        name: []const u8,
        output_prefix: []const u8,
        options: DeviceRobustOptions,
    },
    drawdown_profile: struct {
        name: []const u8,
        output_prefix: []const u8,
        options: DeviceDrawdownOptions,
    },
    extrema_profile: struct {
        name: []const u8,
        output_prefix: []const u8,
        options: DeviceExtremaOptions,
    },
    trend_profile: struct {
        name: []const u8,
        output_prefix: []const u8,
        options: DeviceTrendOptions,
    },
    sign_profile: struct {
        name: []const u8,
        output_prefix: []const u8,
        options: DeviceTrendOptions,
    },
    crossover_profile: struct {
        lhs_name: []const u8,
        rhs_name: []const u8,
        output_prefix: []const u8,
        options: DeviceCrossoverOptions,
    },
    bucket_profile: struct {
        name: []const u8,
        output_prefix: []const u8,
        options: DeviceBucketOptions,
    },
    ema_profile: struct {
        name: []const u8,
        output_prefix: []const u8,
        options: DeviceEmaOptions,
    },
    linear_fit_profile: struct {
        x_name: []const u8,
        y_name: []const u8,
        output_prefix: []const u8,
        options: DeviceLinearFitOptions,
    },
    error_profile: struct {
        actual_name: []const u8,
        predicted_name: []const u8,
        output_prefix: []const u8,
    },
    classification_profile: struct {
        actual_name: []const u8,
        predicted_name: []const u8,
        output_prefix: []const u8,
    },
    rolling_correlation_profile: struct {
        x_name: []const u8,
        y_name: []const u8,
        output_prefix: []const u8,
        options: DeviceRollingCorrelationOptions,
    },
    validity_profile: struct {
        name: []const u8,
        output_prefix: []const u8,
    },
    head: usize,
    tail: usize,

    fn deinit(self: *DeviceLazyOp, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .select => |names| {
                for (names) |name| allocator.free(name);
                allocator.free(names);
            },
            .with_column_binary => |expr| {
                allocator.free(expr.name);
                allocator.free(expr.lhs_name);
                allocator.free(expr.rhs_name);
            },
            .with_column_scalar => |expr| {
                allocator.free(expr.name);
                allocator.free(expr.input_name);
            },
            .with_column_compare => |expr| {
                allocator.free(expr.name);
                allocator.free(expr.lhs_name);
                allocator.free(expr.rhs_name);
            },
            .with_column_compare_scalar => |expr| {
                allocator.free(expr.name);
                allocator.free(expr.input_name);
            },
            .filter_mask => |*mask| mask.deinit(),
            .filter_scalar => |filter_op| allocator.free(filter_op.name),
            .group_by_count => |group| {
                allocator.free(group.key_name);
                allocator.free(group.output_name);
            },
            .group_by_value => |group| {
                allocator.free(group.key_name);
                allocator.free(group.value_name);
                allocator.free(group.output_name);
            },
            .group_by_stats => |group| {
                allocator.free(group.key_name);
                allocator.free(group.value_name);
                allocator.free(group.output_prefix);
            },
            .group_by_stats_on => |group| {
                freeNameList(allocator, group.key_names);
                allocator.free(group.value_name);
                allocator.free(group.output_prefix);
            },
            .group_by_profile => |group| {
                allocator.free(group.key_name);
                allocator.free(group.value_name);
                allocator.free(group.output_prefix);
            },
            .group_by_profile_on => |group| {
                freeNameList(allocator, group.key_names);
                allocator.free(group.value_name);
                allocator.free(group.output_prefix);
            },
            .join_on => |*join| {
                join.right.deinit();
                freeNameList(allocator, join.left_key_names);
                freeNameList(allocator, join.right_key_names);
                allocator.free(join.options.right_suffix);
            },
            .asof_join => |*join| {
                join.right.deinit();
                allocator.free(join.left_key_name);
                allocator.free(join.right_key_name);
                allocator.free(join.options.right_suffix);
            },
            .concat_rows => |*right| right.deinit(),
            .distinct_on => |names| freeNameList(allocator, names),
            .sort_by => |sort| allocator.free(sort.name),
            .top_k => |top| allocator.free(top.name),
            .rank_profile_by => |rank| {
                allocator.free(rank.name);
                allocator.free(rank.output_prefix);
            },
            .rolling_profile => |rolling| {
                allocator.free(rolling.name);
                allocator.free(rolling.output_prefix);
            },
            .rolling_range_profile => |rolling| {
                allocator.free(rolling.name);
                allocator.free(rolling.output_prefix);
            },
            .rolling_normalize_profile => |rolling| {
                allocator.free(rolling.name);
                allocator.free(rolling.output_prefix);
            },
            .rolling_quantile_profile => |rolling| {
                allocator.free(rolling.name);
                allocator.free(rolling.output_prefix);
            },
            .lag_profile => |lag| {
                allocator.free(lag.name);
                allocator.free(lag.output_prefix);
            },
            .lead_profile => |lead| {
                allocator.free(lead.name);
                allocator.free(lead.output_prefix);
            },
            .clip_profile => |clip| {
                allocator.free(clip.name);
                allocator.free(clip.output_prefix);
            },
            .threshold_profile => |threshold| {
                allocator.free(threshold.name);
                allocator.free(threshold.output_prefix);
            },
            .expanding_profile => |expanding| {
                allocator.free(expanding.name);
                allocator.free(expanding.output_prefix);
            },
            .standardize_profile => |standardize| {
                allocator.free(standardize.name);
                allocator.free(standardize.output_prefix);
            },
            .robust_profile => |robust| {
                allocator.free(robust.name);
                allocator.free(robust.output_prefix);
            },
            .drawdown_profile => |drawdown| {
                allocator.free(drawdown.name);
                allocator.free(drawdown.output_prefix);
            },
            .extrema_profile => |extrema| {
                allocator.free(extrema.name);
                allocator.free(extrema.output_prefix);
            },
            .trend_profile => |trend| {
                allocator.free(trend.name);
                allocator.free(trend.output_prefix);
            },
            .sign_profile => |sign| {
                allocator.free(sign.name);
                allocator.free(sign.output_prefix);
            },
            .crossover_profile => |cross| {
                allocator.free(cross.lhs_name);
                allocator.free(cross.rhs_name);
                allocator.free(cross.output_prefix);
            },
            .bucket_profile => |bucket| {
                allocator.free(bucket.name);
                allocator.free(bucket.output_prefix);
            },
            .ema_profile => |ema| {
                allocator.free(ema.name);
                allocator.free(ema.output_prefix);
            },
            .linear_fit_profile => |fit| {
                allocator.free(fit.x_name);
                allocator.free(fit.y_name);
                allocator.free(fit.output_prefix);
            },
            .error_profile => |err| {
                allocator.free(err.actual_name);
                allocator.free(err.predicted_name);
                allocator.free(err.output_prefix);
            },
            .classification_profile => |class| {
                allocator.free(class.actual_name);
                allocator.free(class.predicted_name);
                allocator.free(class.output_prefix);
            },
            .rolling_correlation_profile => |corr| {
                allocator.free(corr.x_name);
                allocator.free(corr.y_name);
                allocator.free(corr.output_prefix);
            },
            .validity_profile => |validity| {
                allocator.free(validity.name);
                allocator.free(validity.output_prefix);
            },
            .distinct_rows, .head, .tail => {},
        }
        self.* = undefined;
    }

    fn clone(self: DeviceLazyOp, allocator: std.mem.Allocator) DeviceDataError!DeviceLazyOp {
        return switch (self) {
            .select => |names| blk: {
                const owned = try allocator.alloc([]const u8, names.len);
                errdefer allocator.free(owned);
                var initialized: usize = 0;
                errdefer {
                    for (owned[0..initialized]) |name| allocator.free(name);
                }
                for (names, owned) |name, *slot| {
                    slot.* = try allocator.dupe(u8, name);
                    initialized += 1;
                }
                break :blk .{ .select = owned };
            },
            .with_column_binary => |expr| blk: {
                const name = try allocator.dupe(u8, expr.name);
                errdefer allocator.free(name);
                const lhs_name = try allocator.dupe(u8, expr.lhs_name);
                errdefer allocator.free(lhs_name);
                const rhs_name = try allocator.dupe(u8, expr.rhs_name);
                errdefer allocator.free(rhs_name);
                break :blk .{ .with_column_binary = .{
                    .name = name,
                    .lhs_name = lhs_name,
                    .rhs_name = rhs_name,
                    .op = expr.op,
                } };
            },
            .with_column_scalar => |expr| blk: {
                const name = try allocator.dupe(u8, expr.name);
                errdefer allocator.free(name);
                const input_name = try allocator.dupe(u8, expr.input_name);
                errdefer allocator.free(input_name);
                break :blk .{ .with_column_scalar = .{
                    .name = name,
                    .input_name = input_name,
                    .op = expr.op,
                    .scalar = expr.scalar,
                } };
            },
            .with_column_compare => |expr| blk: {
                const name = try allocator.dupe(u8, expr.name);
                errdefer allocator.free(name);
                const lhs_name = try allocator.dupe(u8, expr.lhs_name);
                errdefer allocator.free(lhs_name);
                const rhs_name = try allocator.dupe(u8, expr.rhs_name);
                errdefer allocator.free(rhs_name);
                break :blk .{ .with_column_compare = .{
                    .name = name,
                    .lhs_name = lhs_name,
                    .rhs_name = rhs_name,
                    .op = expr.op,
                } };
            },
            .with_column_compare_scalar => |expr| blk: {
                const name = try allocator.dupe(u8, expr.name);
                errdefer allocator.free(name);
                const input_name = try allocator.dupe(u8, expr.input_name);
                errdefer allocator.free(input_name);
                break :blk .{ .with_column_compare_scalar = .{
                    .name = name,
                    .input_name = input_name,
                    .op = expr.op,
                    .scalar = expr.scalar,
                } };
            },
            .filter_mask => |mask| .{ .filter_mask = try mask.clone() },
            .filter_scalar => |filter_op| .{ .filter_scalar = .{
                .name = try allocator.dupe(u8, filter_op.name),
                .op = filter_op.op,
                .scalar = filter_op.scalar,
            } },
            .group_by_count => |group| blk: {
                const key_name = try allocator.dupe(u8, group.key_name);
                errdefer allocator.free(key_name);
                const output_name = try allocator.dupe(u8, group.output_name);
                errdefer allocator.free(output_name);
                break :blk .{ .group_by_count = .{
                    .key_name = key_name,
                    .output_name = output_name,
                } };
            },
            .group_by_value => |group| blk: {
                const key_name = try allocator.dupe(u8, group.key_name);
                errdefer allocator.free(key_name);
                const value_name = try allocator.dupe(u8, group.value_name);
                errdefer allocator.free(value_name);
                const output_name = try allocator.dupe(u8, group.output_name);
                errdefer allocator.free(output_name);
                break :blk .{ .group_by_value = .{
                    .key_name = key_name,
                    .value_name = value_name,
                    .output_name = output_name,
                    .aggregation = group.aggregation,
                } };
            },
            .group_by_stats => |group| blk: {
                const key_name = try allocator.dupe(u8, group.key_name);
                errdefer allocator.free(key_name);
                const value_name = try allocator.dupe(u8, group.value_name);
                errdefer allocator.free(value_name);
                const output_prefix = try allocator.dupe(u8, group.output_prefix);
                errdefer allocator.free(output_prefix);
                break :blk .{ .group_by_stats = .{
                    .key_name = key_name,
                    .value_name = value_name,
                    .output_prefix = output_prefix,
                } };
            },
            .group_by_stats_on => |group| blk: {
                const key_names = try cloneNameList(allocator, group.key_names);
                errdefer freeNameList(allocator, key_names);
                const value_name = try allocator.dupe(u8, group.value_name);
                errdefer allocator.free(value_name);
                const output_prefix = try allocator.dupe(u8, group.output_prefix);
                errdefer allocator.free(output_prefix);
                break :blk .{ .group_by_stats_on = .{
                    .key_names = key_names,
                    .value_name = value_name,
                    .output_prefix = output_prefix,
                } };
            },
            .group_by_profile => |group| blk: {
                const key_name = try allocator.dupe(u8, group.key_name);
                errdefer allocator.free(key_name);
                const value_name = try allocator.dupe(u8, group.value_name);
                errdefer allocator.free(value_name);
                const output_prefix = try allocator.dupe(u8, group.output_prefix);
                errdefer allocator.free(output_prefix);
                break :blk .{ .group_by_profile = .{
                    .key_name = key_name,
                    .value_name = value_name,
                    .output_prefix = output_prefix,
                } };
            },
            .group_by_profile_on => |group| blk: {
                const key_names = try cloneNameList(allocator, group.key_names);
                errdefer freeNameList(allocator, key_names);
                const value_name = try allocator.dupe(u8, group.value_name);
                errdefer allocator.free(value_name);
                const output_prefix = try allocator.dupe(u8, group.output_prefix);
                errdefer allocator.free(output_prefix);
                break :blk .{ .group_by_profile_on = .{
                    .key_names = key_names,
                    .value_name = value_name,
                    .output_prefix = output_prefix,
                } };
            },
            .join_on => |join| blk: {
                var right = try join.right.clone();
                errdefer right.deinit();
                const left_key_names = try cloneNameList(allocator, join.left_key_names);
                errdefer freeNameList(allocator, left_key_names);
                const right_key_names = try cloneNameList(allocator, join.right_key_names);
                errdefer freeNameList(allocator, right_key_names);
                const right_suffix = try allocator.dupe(u8, join.options.right_suffix);
                errdefer allocator.free(right_suffix);
                break :blk .{ .join_on = .{
                    .kind = join.kind,
                    .right = right,
                    .left_key_names = left_key_names,
                    .right_key_names = right_key_names,
                    .options = .{ .right_suffix = right_suffix },
                } };
            },
            .asof_join => |join| blk: {
                var right = try join.right.clone();
                errdefer right.deinit();
                const left_key_name = try allocator.dupe(u8, join.left_key_name);
                errdefer allocator.free(left_key_name);
                const right_key_name = try allocator.dupe(u8, join.right_key_name);
                errdefer allocator.free(right_key_name);
                const right_suffix = try allocator.dupe(u8, join.options.right_suffix);
                errdefer allocator.free(right_suffix);
                break :blk .{ .asof_join = .{
                    .right = right,
                    .left_key_name = left_key_name,
                    .right_key_name = right_key_name,
                    .options = .{
                        .strategy = join.options.strategy,
                        .right_suffix = right_suffix,
                    },
                } };
            },
            .concat_rows => |right| .{ .concat_rows = try right.clone() },
            .distinct_rows => .{ .distinct_rows = {} },
            .distinct_on => |names| .{ .distinct_on = try cloneNameList(allocator, names) },
            .sort_by => |sort| .{ .sort_by = .{
                .name = try allocator.dupe(u8, sort.name),
                .options = sort.options,
            } },
            .top_k => |top| .{ .top_k = .{
                .name = try allocator.dupe(u8, top.name),
                .options = top.options,
                .k = top.k,
            } },
            .rank_profile_by => |rank| blk: {
                const name = try allocator.dupe(u8, rank.name);
                errdefer allocator.free(name);
                const output_prefix = try allocator.dupe(u8, rank.output_prefix);
                errdefer allocator.free(output_prefix);
                break :blk .{ .rank_profile_by = .{
                    .name = name,
                    .output_prefix = output_prefix,
                    .options = rank.options,
                } };
            },
            .rolling_profile => |rolling| blk: {
                const name = try allocator.dupe(u8, rolling.name);
                errdefer allocator.free(name);
                const output_prefix = try allocator.dupe(u8, rolling.output_prefix);
                errdefer allocator.free(output_prefix);
                break :blk .{ .rolling_profile = .{
                    .name = name,
                    .output_prefix = output_prefix,
                    .options = rolling.options,
                } };
            },
            .rolling_range_profile => |rolling| blk: {
                const name = try allocator.dupe(u8, rolling.name);
                errdefer allocator.free(name);
                const output_prefix = try allocator.dupe(u8, rolling.output_prefix);
                errdefer allocator.free(output_prefix);
                break :blk .{ .rolling_range_profile = .{
                    .name = name,
                    .output_prefix = output_prefix,
                    .options = rolling.options,
                } };
            },
            .rolling_normalize_profile => |rolling| blk: {
                const name = try allocator.dupe(u8, rolling.name);
                errdefer allocator.free(name);
                const output_prefix = try allocator.dupe(u8, rolling.output_prefix);
                errdefer allocator.free(output_prefix);
                break :blk .{ .rolling_normalize_profile = .{
                    .name = name,
                    .output_prefix = output_prefix,
                    .options = rolling.options,
                } };
            },
            .rolling_quantile_profile => |rolling| blk: {
                const name = try allocator.dupe(u8, rolling.name);
                errdefer allocator.free(name);
                const output_prefix = try allocator.dupe(u8, rolling.output_prefix);
                errdefer allocator.free(output_prefix);
                break :blk .{ .rolling_quantile_profile = .{
                    .name = name,
                    .output_prefix = output_prefix,
                    .options = rolling.options,
                } };
            },
            .lag_profile => |lag| blk: {
                const name = try allocator.dupe(u8, lag.name);
                errdefer allocator.free(name);
                const output_prefix = try allocator.dupe(u8, lag.output_prefix);
                errdefer allocator.free(output_prefix);
                break :blk .{ .lag_profile = .{
                    .name = name,
                    .output_prefix = output_prefix,
                    .options = lag.options,
                } };
            },
            .lead_profile => |lead| blk: {
                const name = try allocator.dupe(u8, lead.name);
                errdefer allocator.free(name);
                const output_prefix = try allocator.dupe(u8, lead.output_prefix);
                errdefer allocator.free(output_prefix);
                break :blk .{ .lead_profile = .{
                    .name = name,
                    .output_prefix = output_prefix,
                    .options = lead.options,
                } };
            },
            .clip_profile => |clip| blk: {
                const name = try allocator.dupe(u8, clip.name);
                errdefer allocator.free(name);
                const output_prefix = try allocator.dupe(u8, clip.output_prefix);
                errdefer allocator.free(output_prefix);
                break :blk .{ .clip_profile = .{
                    .name = name,
                    .output_prefix = output_prefix,
                    .options = clip.options,
                } };
            },
            .threshold_profile => |threshold| blk: {
                const name = try allocator.dupe(u8, threshold.name);
                errdefer allocator.free(name);
                const output_prefix = try allocator.dupe(u8, threshold.output_prefix);
                errdefer allocator.free(output_prefix);
                break :blk .{ .threshold_profile = .{
                    .name = name,
                    .output_prefix = output_prefix,
                    .options = threshold.options,
                } };
            },
            .expanding_profile => |expanding| blk: {
                const name = try allocator.dupe(u8, expanding.name);
                errdefer allocator.free(name);
                const output_prefix = try allocator.dupe(u8, expanding.output_prefix);
                errdefer allocator.free(output_prefix);
                break :blk .{ .expanding_profile = .{
                    .name = name,
                    .output_prefix = output_prefix,
                    .options = expanding.options,
                } };
            },
            .standardize_profile => |standardize| blk: {
                const name = try allocator.dupe(u8, standardize.name);
                errdefer allocator.free(name);
                const output_prefix = try allocator.dupe(u8, standardize.output_prefix);
                errdefer allocator.free(output_prefix);
                break :blk .{ .standardize_profile = .{
                    .name = name,
                    .output_prefix = output_prefix,
                    .options = standardize.options,
                } };
            },
            .robust_profile => |robust| blk: {
                const name = try allocator.dupe(u8, robust.name);
                errdefer allocator.free(name);
                const output_prefix = try allocator.dupe(u8, robust.output_prefix);
                errdefer allocator.free(output_prefix);
                break :blk .{ .robust_profile = .{
                    .name = name,
                    .output_prefix = output_prefix,
                    .options = robust.options,
                } };
            },
            .drawdown_profile => |drawdown| blk: {
                const name = try allocator.dupe(u8, drawdown.name);
                errdefer allocator.free(name);
                const output_prefix = try allocator.dupe(u8, drawdown.output_prefix);
                errdefer allocator.free(output_prefix);
                break :blk .{ .drawdown_profile = .{
                    .name = name,
                    .output_prefix = output_prefix,
                    .options = drawdown.options,
                } };
            },
            .extrema_profile => |extrema| blk: {
                const name = try allocator.dupe(u8, extrema.name);
                errdefer allocator.free(name);
                const output_prefix = try allocator.dupe(u8, extrema.output_prefix);
                errdefer allocator.free(output_prefix);
                break :blk .{ .extrema_profile = .{
                    .name = name,
                    .output_prefix = output_prefix,
                    .options = extrema.options,
                } };
            },
            .trend_profile => |trend| blk: {
                const name = try allocator.dupe(u8, trend.name);
                errdefer allocator.free(name);
                const output_prefix = try allocator.dupe(u8, trend.output_prefix);
                errdefer allocator.free(output_prefix);
                break :blk .{ .trend_profile = .{
                    .name = name,
                    .output_prefix = output_prefix,
                    .options = trend.options,
                } };
            },
            .sign_profile => |sign| blk: {
                const name = try allocator.dupe(u8, sign.name);
                errdefer allocator.free(name);
                const output_prefix = try allocator.dupe(u8, sign.output_prefix);
                errdefer allocator.free(output_prefix);
                break :blk .{ .sign_profile = .{
                    .name = name,
                    .output_prefix = output_prefix,
                    .options = sign.options,
                } };
            },
            .crossover_profile => |cross| blk: {
                const lhs_name = try allocator.dupe(u8, cross.lhs_name);
                errdefer allocator.free(lhs_name);
                const rhs_name = try allocator.dupe(u8, cross.rhs_name);
                errdefer allocator.free(rhs_name);
                const output_prefix = try allocator.dupe(u8, cross.output_prefix);
                errdefer allocator.free(output_prefix);
                break :blk .{ .crossover_profile = .{
                    .lhs_name = lhs_name,
                    .rhs_name = rhs_name,
                    .output_prefix = output_prefix,
                    .options = cross.options,
                } };
            },
            .bucket_profile => |bucket| blk: {
                const name = try allocator.dupe(u8, bucket.name);
                errdefer allocator.free(name);
                const output_prefix = try allocator.dupe(u8, bucket.output_prefix);
                errdefer allocator.free(output_prefix);
                break :blk .{ .bucket_profile = .{
                    .name = name,
                    .output_prefix = output_prefix,
                    .options = bucket.options,
                } };
            },
            .ema_profile => |ema| blk: {
                const name = try allocator.dupe(u8, ema.name);
                errdefer allocator.free(name);
                const output_prefix = try allocator.dupe(u8, ema.output_prefix);
                errdefer allocator.free(output_prefix);
                break :blk .{ .ema_profile = .{
                    .name = name,
                    .output_prefix = output_prefix,
                    .options = ema.options,
                } };
            },
            .linear_fit_profile => |fit| blk: {
                const x_name = try allocator.dupe(u8, fit.x_name);
                errdefer allocator.free(x_name);
                const y_name = try allocator.dupe(u8, fit.y_name);
                errdefer allocator.free(y_name);
                const output_prefix = try allocator.dupe(u8, fit.output_prefix);
                errdefer allocator.free(output_prefix);
                break :blk .{ .linear_fit_profile = .{
                    .x_name = x_name,
                    .y_name = y_name,
                    .output_prefix = output_prefix,
                    .options = fit.options,
                } };
            },
            .error_profile => |err| blk: {
                const actual_name = try allocator.dupe(u8, err.actual_name);
                errdefer allocator.free(actual_name);
                const predicted_name = try allocator.dupe(u8, err.predicted_name);
                errdefer allocator.free(predicted_name);
                const output_prefix = try allocator.dupe(u8, err.output_prefix);
                errdefer allocator.free(output_prefix);
                break :blk .{ .error_profile = .{
                    .actual_name = actual_name,
                    .predicted_name = predicted_name,
                    .output_prefix = output_prefix,
                } };
            },
            .classification_profile => |class| blk: {
                const actual_name = try allocator.dupe(u8, class.actual_name);
                errdefer allocator.free(actual_name);
                const predicted_name = try allocator.dupe(u8, class.predicted_name);
                errdefer allocator.free(predicted_name);
                const output_prefix = try allocator.dupe(u8, class.output_prefix);
                errdefer allocator.free(output_prefix);
                break :blk .{ .classification_profile = .{
                    .actual_name = actual_name,
                    .predicted_name = predicted_name,
                    .output_prefix = output_prefix,
                } };
            },
            .rolling_correlation_profile => |corr| blk: {
                const x_name = try allocator.dupe(u8, corr.x_name);
                errdefer allocator.free(x_name);
                const y_name = try allocator.dupe(u8, corr.y_name);
                errdefer allocator.free(y_name);
                const output_prefix = try allocator.dupe(u8, corr.output_prefix);
                errdefer allocator.free(output_prefix);
                break :blk .{ .rolling_correlation_profile = .{
                    .x_name = x_name,
                    .y_name = y_name,
                    .output_prefix = output_prefix,
                    .options = corr.options,
                } };
            },
            .validity_profile => |validity| blk: {
                const name = try allocator.dupe(u8, validity.name);
                errdefer allocator.free(name);
                const output_prefix = try allocator.dupe(u8, validity.output_prefix);
                errdefer allocator.free(output_prefix);
                break :blk .{ .validity_profile = .{
                    .name = name,
                    .output_prefix = output_prefix,
                } };
            },
            .head => |n| .{ .head = n },
            .tail => |n| .{ .tail = n },
        };
    }
};

pub const DeviceLazySource = union(enum) {
    dataframe: DeviceDataFrame,
    parquet_scan: DeviceParquetScan,

    fn deinit(self: *DeviceLazySource) void {
        switch (self.*) {
            .dataframe => |*frame| frame.deinit(),
            .parquet_scan => |*scan| scan.deinit(),
        }
        self.* = undefined;
    }

    fn clone(self: DeviceLazySource) DeviceDataError!DeviceLazySource {
        return switch (self) {
            .dataframe => |frame| .{ .dataframe = try frame.clone() },
            .parquet_scan => |scan| .{ .parquet_scan = try scan.clone() },
        };
    }

    fn name(self: DeviceLazySource) []const u8 {
        return switch (self) {
            .dataframe => "dataframe",
            .parquet_scan => "parquet_scan",
        };
    }
};

/// A compact eager-backed lazy plan for `DeviceDataFrame`.
///
/// Polars' lazy API is valuable because it gives the planner a concrete list of
/// projections, filters, and ordering operations before execution.  Vectra keeps
/// the plan small and still executes through the existing `DeviceDataFrame`
/// methods in `collect()`, but scan sources are represented explicitly so the
/// planner can push conservative Parquet row-group pruning and column projection
/// toward Boltha before materializing CPU/CUDA/MPS columns.  That gives callers a
/// stable API today and gives Axiom a single future lowering boundary for
/// fusing/reordering dataframe operations across CPU/CUDA/MPS.
pub const DeviceLazyFrame = struct {
    allocator: std.mem.Allocator,
    source: DeviceLazySource,
    ops: std.ArrayList(DeviceLazyOp) = .empty,

    pub fn init(allocator: std.mem.Allocator, source: DeviceDataFrame) DeviceDataError!DeviceLazyFrame {
        return .{
            .allocator = allocator,
            .source = .{ .dataframe = try source.clone() },
        };
    }

    pub fn initParquetScan(allocator: std.mem.Allocator, scan: DeviceParquetScan) DeviceDataError!DeviceLazyFrame {
        return .{
            .allocator = allocator,
            .source = .{ .parquet_scan = try scan.clone() },
        };
    }

    pub fn scanParquetBytes(allocator: std.mem.Allocator, bytes: []const u8, device_value: array_mod.Device) DeviceDataError!DeviceLazyFrame {
        return .{
            .allocator = allocator,
            .source = .{ .parquet_scan = try DeviceParquetScan.init(allocator, bytes, device_value) },
        };
    }

    pub fn clone(self: DeviceLazyFrame) DeviceDataError!DeviceLazyFrame {
        var cloned = DeviceLazyFrame{
            .allocator = self.allocator,
            .source = try self.source.clone(),
        };
        errdefer cloned.source.deinit();
        errdefer deinitLazyOps(self.allocator, &cloned.ops);
        for (self.ops.items) |op| {
            var cloned_op = try op.clone(self.allocator);
            errdefer cloned_op.deinit(self.allocator);
            try cloned.ops.append(self.allocator, cloned_op);
        }
        return cloned;
    }

    pub fn deinit(self: *DeviceLazyFrame) void {
        self.source.deinit();
        for (self.ops.items) |*op| op.deinit(self.allocator);
        self.ops.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn select(self: *DeviceLazyFrame, names: []const []const u8) DeviceDataError!void {
        const owned = try self.allocator.alloc([]const u8, names.len);
        errdefer self.allocator.free(owned);
        var initialized: usize = 0;
        errdefer {
            for (owned[0..initialized]) |name| self.allocator.free(name);
        }
        for (names, owned) |name, *slot| {
            slot.* = try self.allocator.dupe(u8, name);
            initialized += 1;
        }
        try self.ops.append(self.allocator, .{ .select = owned });
    }

    pub fn filter(self: *DeviceLazyFrame, mask: DeviceColumn) DeviceDataError!void {
        try self.ops.append(self.allocator, .{ .filter_mask = try mask.clone() });
    }

    pub fn withColumnBinary(self: *DeviceLazyFrame, name: []const u8, lhs_name: []const u8, rhs_name: []const u8, op: DeviceColumnBinaryOp) DeviceDataError!void {
        const owned_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned_name);
        const owned_lhs = try self.allocator.dupe(u8, lhs_name);
        errdefer self.allocator.free(owned_lhs);
        const owned_rhs = try self.allocator.dupe(u8, rhs_name);
        errdefer self.allocator.free(owned_rhs);
        try self.ops.append(self.allocator, .{ .with_column_binary = .{
            .name = owned_name,
            .lhs_name = owned_lhs,
            .rhs_name = owned_rhs,
            .op = op,
        } });
    }

    pub fn withColumnScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, scalar: T, op: DeviceColumnBinaryOp) DeviceDataError!void {
        const owned_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned_name);
        const owned_input = try self.allocator.dupe(u8, input_name);
        errdefer self.allocator.free(owned_input);
        try self.ops.append(self.allocator, .{ .with_column_scalar = .{
            .name = owned_name,
            .input_name = owned_input,
            .op = op,
            .scalar = DeviceScalar.init(T, scalar),
        } });
    }

    pub fn withColumnCompare(self: *DeviceLazyFrame, name: []const u8, lhs_name: []const u8, rhs_name: []const u8, op: DeviceColumnCompareOp) DeviceDataError!void {
        const owned_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned_name);
        const owned_lhs = try self.allocator.dupe(u8, lhs_name);
        errdefer self.allocator.free(owned_lhs);
        const owned_rhs = try self.allocator.dupe(u8, rhs_name);
        errdefer self.allocator.free(owned_rhs);
        try self.ops.append(self.allocator, .{ .with_column_compare = .{
            .name = owned_name,
            .lhs_name = owned_lhs,
            .rhs_name = owned_rhs,
            .op = op,
        } });
    }

    pub fn withColumnCompareScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, scalar: T, op: DeviceColumnCompareOp) DeviceDataError!void {
        const owned_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned_name);
        const owned_input = try self.allocator.dupe(u8, input_name);
        errdefer self.allocator.free(owned_input);
        try self.ops.append(self.allocator, .{ .with_column_compare_scalar = .{
            .name = owned_name,
            .input_name = owned_input,
            .op = op,
            .scalar = DeviceScalar.init(T, scalar),
        } });
    }

    pub fn groupByCount(self: *DeviceLazyFrame, key_name: []const u8, output_name: []const u8) DeviceDataError!void {
        const owned_key = try self.allocator.dupe(u8, key_name);
        errdefer self.allocator.free(owned_key);
        const owned_output = try self.allocator.dupe(u8, output_name);
        errdefer self.allocator.free(owned_output);
        try self.ops.append(self.allocator, .{ .group_by_count = .{
            .key_name = owned_key,
            .output_name = owned_output,
        } });
    }

    pub fn groupByValue(self: *DeviceLazyFrame, key_name: []const u8, value_name: []const u8, output_name: []const u8, aggregation: DeviceLazyGroupByAggregation) DeviceDataError!void {
        const owned_key = try self.allocator.dupe(u8, key_name);
        errdefer self.allocator.free(owned_key);
        const owned_value = try self.allocator.dupe(u8, value_name);
        errdefer self.allocator.free(owned_value);
        const owned_output = try self.allocator.dupe(u8, output_name);
        errdefer self.allocator.free(owned_output);
        try self.ops.append(self.allocator, .{ .group_by_value = .{
            .key_name = owned_key,
            .value_name = owned_value,
            .output_name = owned_output,
            .aggregation = aggregation,
        } });
    }

    pub fn groupBySum(self: *DeviceLazyFrame, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
        return self.groupByValue(key_name, value_name, output_name, .sum);
    }

    pub fn groupByMin(self: *DeviceLazyFrame, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
        return self.groupByValue(key_name, value_name, output_name, .min);
    }

    pub fn groupByMax(self: *DeviceLazyFrame, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
        return self.groupByValue(key_name, value_name, output_name, .max);
    }

    pub fn groupByMean(self: *DeviceLazyFrame, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
        return self.groupByValue(key_name, value_name, output_name, .mean);
    }

    pub fn groupByStats(self: *DeviceLazyFrame, key_name: []const u8, value_name: []const u8, output_prefix: []const u8) DeviceDataError!void {
        const owned_key = try self.allocator.dupe(u8, key_name);
        errdefer self.allocator.free(owned_key);
        const owned_value = try self.allocator.dupe(u8, value_name);
        errdefer self.allocator.free(owned_value);
        const owned_prefix = try self.allocator.dupe(u8, output_prefix);
        errdefer self.allocator.free(owned_prefix);
        try self.ops.append(self.allocator, .{ .group_by_stats = .{
            .key_name = owned_key,
            .value_name = owned_value,
            .output_prefix = owned_prefix,
        } });
    }

    pub fn groupByStatsOn(self: *DeviceLazyFrame, key_names: []const []const u8, value_name: []const u8, output_prefix: []const u8) DeviceDataError!void {
        const owned_keys = try cloneNameList(self.allocator, key_names);
        errdefer freeNameList(self.allocator, owned_keys);
        const owned_value = try self.allocator.dupe(u8, value_name);
        errdefer self.allocator.free(owned_value);
        const owned_prefix = try self.allocator.dupe(u8, output_prefix);
        errdefer self.allocator.free(owned_prefix);
        try self.ops.append(self.allocator, .{ .group_by_stats_on = .{
            .key_names = owned_keys,
            .value_name = owned_value,
            .output_prefix = owned_prefix,
        } });
    }

    pub fn groupByProfile(self: *DeviceLazyFrame, key_name: []const u8, value_name: []const u8, output_prefix: []const u8) DeviceDataError!void {
        const owned_key = try self.allocator.dupe(u8, key_name);
        errdefer self.allocator.free(owned_key);
        const owned_value = try self.allocator.dupe(u8, value_name);
        errdefer self.allocator.free(owned_value);
        const owned_prefix = try self.allocator.dupe(u8, output_prefix);
        errdefer self.allocator.free(owned_prefix);
        try self.ops.append(self.allocator, .{ .group_by_profile = .{
            .key_name = owned_key,
            .value_name = owned_value,
            .output_prefix = owned_prefix,
        } });
    }

    pub fn groupByProfileOn(self: *DeviceLazyFrame, key_names: []const []const u8, value_name: []const u8, output_prefix: []const u8) DeviceDataError!void {
        const owned_keys = try cloneNameList(self.allocator, key_names);
        errdefer freeNameList(self.allocator, owned_keys);
        const owned_value = try self.allocator.dupe(u8, value_name);
        errdefer self.allocator.free(owned_value);
        const owned_prefix = try self.allocator.dupe(u8, output_prefix);
        errdefer self.allocator.free(owned_prefix);
        try self.ops.append(self.allocator, .{ .group_by_profile_on = .{
            .key_names = owned_keys,
            .value_name = owned_value,
            .output_prefix = owned_prefix,
        } });
    }

    pub fn joinOn(
        self: *DeviceLazyFrame,
        right: DeviceDataFrame,
        left_key_names: []const []const u8,
        right_key_names: []const []const u8,
        kind: DeviceLazyJoinKind,
        options_value: DeviceJoinOptions,
    ) DeviceDataError!void {
        if (left_key_names.len == 0 or left_key_names.len != right_key_names.len) return error.LengthMismatch;
        var owned_right = try right.clone();
        errdefer owned_right.deinit();
        const owned_left_keys = try cloneNameList(self.allocator, left_key_names);
        errdefer freeNameList(self.allocator, owned_left_keys);
        const owned_right_keys = try cloneNameList(self.allocator, right_key_names);
        errdefer freeNameList(self.allocator, owned_right_keys);
        const owned_suffix = try self.allocator.dupe(u8, options_value.right_suffix);
        errdefer self.allocator.free(owned_suffix);
        try self.ops.append(self.allocator, .{ .join_on = .{
            .kind = kind,
            .right = owned_right,
            .left_key_names = owned_left_keys,
            .right_key_names = owned_right_keys,
            .options = .{ .right_suffix = owned_suffix },
        } });
    }

    pub fn innerJoinOn(self: *DeviceLazyFrame, right: DeviceDataFrame, left_key_names: []const []const u8, right_key_names: []const []const u8, options_value: DeviceJoinOptions) DeviceDataError!void {
        return self.joinOn(right, left_key_names, right_key_names, .inner, options_value);
    }

    pub fn leftJoinOn(self: *DeviceLazyFrame, right: DeviceDataFrame, left_key_names: []const []const u8, right_key_names: []const []const u8, options_value: DeviceJoinOptions) DeviceDataError!void {
        return self.joinOn(right, left_key_names, right_key_names, .left, options_value);
    }

    pub fn fullJoinOn(self: *DeviceLazyFrame, right: DeviceDataFrame, left_key_names: []const []const u8, right_key_names: []const []const u8, options_value: DeviceJoinOptions) DeviceDataError!void {
        return self.joinOn(right, left_key_names, right_key_names, .full, options_value);
    }

    pub fn semiJoinOn(self: *DeviceLazyFrame, right: DeviceDataFrame, left_key_names: []const []const u8, right_key_names: []const []const u8) DeviceDataError!void {
        return self.joinOn(right, left_key_names, right_key_names, .semi, .{});
    }

    pub fn antiJoinOn(self: *DeviceLazyFrame, right: DeviceDataFrame, left_key_names: []const []const u8, right_key_names: []const []const u8) DeviceDataError!void {
        return self.joinOn(right, left_key_names, right_key_names, .anti, .{});
    }

    pub fn asofJoin(
        self: *DeviceLazyFrame,
        right: DeviceDataFrame,
        left_key_name: []const u8,
        right_key_name: []const u8,
        options_value: DeviceAsofOptions,
    ) DeviceDataError!void {
        var owned_right = try right.clone();
        errdefer owned_right.deinit();
        const owned_left_key = try self.allocator.dupe(u8, left_key_name);
        errdefer self.allocator.free(owned_left_key);
        const owned_right_key = try self.allocator.dupe(u8, right_key_name);
        errdefer self.allocator.free(owned_right_key);
        const owned_suffix = try self.allocator.dupe(u8, options_value.right_suffix);
        errdefer self.allocator.free(owned_suffix);
        try self.ops.append(self.allocator, .{ .asof_join = .{
            .right = owned_right,
            .left_key_name = owned_left_key,
            .right_key_name = owned_right_key,
            .options = .{
                .strategy = options_value.strategy,
                .right_suffix = owned_suffix,
            },
        } });
    }

    pub fn concatRows(self: *DeviceLazyFrame, right: DeviceDataFrame) DeviceDataError!void {
        var owned_right = try right.clone();
        errdefer owned_right.deinit();
        try self.ops.append(self.allocator, .{ .concat_rows = owned_right });
    }

    pub fn appendRows(self: *DeviceLazyFrame, right: DeviceDataFrame) DeviceDataError!void {
        return self.concatRows(right);
    }

    pub fn vstack(self: *DeviceLazyFrame, right: DeviceDataFrame) DeviceDataError!void {
        return self.concatRows(right);
    }

    pub fn distinctRows(self: *DeviceLazyFrame) DeviceDataError!void {
        try self.ops.append(self.allocator, .{ .distinct_rows = {} });
    }

    pub fn distinctOn(self: *DeviceLazyFrame, key_names: []const []const u8) DeviceDataError!void {
        if (key_names.len == 0) return error.LengthMismatch;
        try self.ops.append(self.allocator, .{ .distinct_on = try cloneNameList(self.allocator, key_names) });
    }

    pub fn dropDuplicates(self: *DeviceLazyFrame) DeviceDataError!void {
        return self.distinctRows();
    }

    pub fn dropDuplicatesOn(self: *DeviceLazyFrame, key_names: []const []const u8) DeviceDataError!void {
        return self.distinctOn(key_names);
    }

    pub fn uniqueRows(self: *DeviceLazyFrame) DeviceDataError!void {
        return self.distinctRows();
    }

    pub fn filterColumnScalar(self: *DeviceLazyFrame, name: []const u8, comptime T: type, scalar: T, op: DeviceColumnCompareOp) DeviceDataError!void {
        try self.ops.append(self.allocator, .{ .filter_scalar = .{
            .name = try self.allocator.dupe(u8, name),
            .op = op,
            .scalar = DeviceScalar.init(T, scalar),
        } });
    }

    pub fn sortBy(self: *DeviceLazyFrame, name: []const u8, options_value: DeviceSortOptions) DeviceDataError!void {
        try self.ops.append(self.allocator, .{ .sort_by = .{
            .name = try self.allocator.dupe(u8, name),
            .options = options_value,
        } });
    }

    pub fn rankProfileBy(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceSortOptions) DeviceDataError!void {
        const owned_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned_name);
        const owned_prefix = try self.allocator.dupe(u8, output_prefix);
        errdefer self.allocator.free(owned_prefix);
        try self.ops.append(self.allocator, .{ .rank_profile_by = .{
            .name = owned_name,
            .output_prefix = owned_prefix,
            .options = options_value,
        } });
    }

    pub fn rollingProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingOptions) DeviceDataError!void {
        const owned_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned_name);
        const owned_prefix = try self.allocator.dupe(u8, output_prefix);
        errdefer self.allocator.free(owned_prefix);
        try self.ops.append(self.allocator, .{ .rolling_profile = .{
            .name = owned_name,
            .output_prefix = owned_prefix,
            .options = options_value,
        } });
    }

    pub fn rollingRangeProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingOptions) DeviceDataError!void {
        const owned_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned_name);
        const owned_prefix = try self.allocator.dupe(u8, output_prefix);
        errdefer self.allocator.free(owned_prefix);
        try self.ops.append(self.allocator, .{ .rolling_range_profile = .{
            .name = owned_name,
            .output_prefix = owned_prefix,
            .options = options_value,
        } });
    }

    pub fn rollingNormalizeProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingOptions) DeviceDataError!void {
        const owned_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned_name);
        const owned_prefix = try self.allocator.dupe(u8, output_prefix);
        errdefer self.allocator.free(owned_prefix);
        try self.ops.append(self.allocator, .{ .rolling_normalize_profile = .{
            .name = owned_name,
            .output_prefix = owned_prefix,
            .options = options_value,
        } });
    }

    pub fn rollingQuantileProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingOptions) DeviceDataError!void {
        const owned_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned_name);
        const owned_prefix = try self.allocator.dupe(u8, output_prefix);
        errdefer self.allocator.free(owned_prefix);
        try self.ops.append(self.allocator, .{ .rolling_quantile_profile = .{
            .name = owned_name,
            .output_prefix = owned_prefix,
            .options = options_value,
        } });
    }

    pub fn lagProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceLagOptions) DeviceDataError!void {
        const owned_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned_name);
        const owned_prefix = try self.allocator.dupe(u8, output_prefix);
        errdefer self.allocator.free(owned_prefix);
        try self.ops.append(self.allocator, .{ .lag_profile = .{
            .name = owned_name,
            .output_prefix = owned_prefix,
            .options = options_value,
        } });
    }

    pub fn leadProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceLagOptions) DeviceDataError!void {
        const owned_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned_name);
        const owned_prefix = try self.allocator.dupe(u8, output_prefix);
        errdefer self.allocator.free(owned_prefix);
        try self.ops.append(self.allocator, .{ .lead_profile = .{
            .name = owned_name,
            .output_prefix = owned_prefix,
            .options = options_value,
        } });
    }

    pub fn clipProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceClipOptions) DeviceDataError!void {
        const owned_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned_name);
        const owned_prefix = try self.allocator.dupe(u8, output_prefix);
        errdefer self.allocator.free(owned_prefix);
        try self.ops.append(self.allocator, .{ .clip_profile = .{
            .name = owned_name,
            .output_prefix = owned_prefix,
            .options = options_value,
        } });
    }

    pub fn thresholdProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceThresholdOptions) DeviceDataError!void {
        const owned_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned_name);
        const owned_prefix = try self.allocator.dupe(u8, output_prefix);
        errdefer self.allocator.free(owned_prefix);
        try self.ops.append(self.allocator, .{ .threshold_profile = .{
            .name = owned_name,
            .output_prefix = owned_prefix,
            .options = options_value,
        } });
    }

    pub fn expandingProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceExpandingOptions) DeviceDataError!void {
        const owned_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned_name);
        const owned_prefix = try self.allocator.dupe(u8, output_prefix);
        errdefer self.allocator.free(owned_prefix);
        try self.ops.append(self.allocator, .{ .expanding_profile = .{
            .name = owned_name,
            .output_prefix = owned_prefix,
            .options = options_value,
        } });
    }

    pub fn standardizeProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceStandardizeOptions) DeviceDataError!void {
        const owned_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned_name);
        const owned_prefix = try self.allocator.dupe(u8, output_prefix);
        errdefer self.allocator.free(owned_prefix);
        try self.ops.append(self.allocator, .{ .standardize_profile = .{
            .name = owned_name,
            .output_prefix = owned_prefix,
            .options = options_value,
        } });
    }

    pub fn robustProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceRobustOptions) DeviceDataError!void {
        const owned_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned_name);
        const owned_prefix = try self.allocator.dupe(u8, output_prefix);
        errdefer self.allocator.free(owned_prefix);
        try self.ops.append(self.allocator, .{ .robust_profile = .{
            .name = owned_name,
            .output_prefix = owned_prefix,
            .options = options_value,
        } });
    }

    pub fn drawdownProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceDrawdownOptions) DeviceDataError!void {
        const owned_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned_name);
        const owned_prefix = try self.allocator.dupe(u8, output_prefix);
        errdefer self.allocator.free(owned_prefix);
        try self.ops.append(self.allocator, .{ .drawdown_profile = .{
            .name = owned_name,
            .output_prefix = owned_prefix,
            .options = options_value,
        } });
    }

    pub fn extremaProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceExtremaOptions) DeviceDataError!void {
        const owned_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned_name);
        const owned_prefix = try self.allocator.dupe(u8, output_prefix);
        errdefer self.allocator.free(owned_prefix);
        try self.ops.append(self.allocator, .{ .extrema_profile = .{
            .name = owned_name,
            .output_prefix = owned_prefix,
            .options = options_value,
        } });
    }

    pub fn trendProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceTrendOptions) DeviceDataError!void {
        const owned_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned_name);
        const owned_prefix = try self.allocator.dupe(u8, output_prefix);
        errdefer self.allocator.free(owned_prefix);
        try self.ops.append(self.allocator, .{ .trend_profile = .{
            .name = owned_name,
            .output_prefix = owned_prefix,
            .options = options_value,
        } });
    }

    pub fn signProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceTrendOptions) DeviceDataError!void {
        const owned_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned_name);
        const owned_prefix = try self.allocator.dupe(u8, output_prefix);
        errdefer self.allocator.free(owned_prefix);
        try self.ops.append(self.allocator, .{ .sign_profile = .{
            .name = owned_name,
            .output_prefix = owned_prefix,
            .options = options_value,
        } });
    }

    pub fn crossoverProfile(
        self: *DeviceLazyFrame,
        lhs_name: []const u8,
        rhs_name: []const u8,
        output_prefix: []const u8,
        options_value: DeviceCrossoverOptions,
    ) DeviceDataError!void {
        const owned_lhs = try self.allocator.dupe(u8, lhs_name);
        errdefer self.allocator.free(owned_lhs);
        const owned_rhs = try self.allocator.dupe(u8, rhs_name);
        errdefer self.allocator.free(owned_rhs);
        const owned_prefix = try self.allocator.dupe(u8, output_prefix);
        errdefer self.allocator.free(owned_prefix);
        try self.ops.append(self.allocator, .{ .crossover_profile = .{
            .lhs_name = owned_lhs,
            .rhs_name = owned_rhs,
            .output_prefix = owned_prefix,
            .options = options_value,
        } });
    }

    pub fn bucketProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceBucketOptions) DeviceDataError!void {
        const owned_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned_name);
        const owned_prefix = try self.allocator.dupe(u8, output_prefix);
        errdefer self.allocator.free(owned_prefix);
        try self.ops.append(self.allocator, .{ .bucket_profile = .{
            .name = owned_name,
            .output_prefix = owned_prefix,
            .options = options_value,
        } });
    }

    pub fn emaProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceEmaOptions) DeviceDataError!void {
        const owned_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned_name);
        const owned_prefix = try self.allocator.dupe(u8, output_prefix);
        errdefer self.allocator.free(owned_prefix);
        try self.ops.append(self.allocator, .{ .ema_profile = .{
            .name = owned_name,
            .output_prefix = owned_prefix,
            .options = options_value,
        } });
    }

    pub fn linearFitProfile(
        self: *DeviceLazyFrame,
        x_name: []const u8,
        y_name: []const u8,
        output_prefix: []const u8,
        options_value: DeviceLinearFitOptions,
    ) DeviceDataError!void {
        const owned_x = try self.allocator.dupe(u8, x_name);
        errdefer self.allocator.free(owned_x);
        const owned_y = try self.allocator.dupe(u8, y_name);
        errdefer self.allocator.free(owned_y);
        const owned_prefix = try self.allocator.dupe(u8, output_prefix);
        errdefer self.allocator.free(owned_prefix);
        try self.ops.append(self.allocator, .{ .linear_fit_profile = .{
            .x_name = owned_x,
            .y_name = owned_y,
            .output_prefix = owned_prefix,
            .options = options_value,
        } });
    }

    pub fn errorProfile(
        self: *DeviceLazyFrame,
        actual_name: []const u8,
        predicted_name: []const u8,
        output_prefix: []const u8,
    ) DeviceDataError!void {
        const owned_actual = try self.allocator.dupe(u8, actual_name);
        errdefer self.allocator.free(owned_actual);
        const owned_predicted = try self.allocator.dupe(u8, predicted_name);
        errdefer self.allocator.free(owned_predicted);
        const owned_prefix = try self.allocator.dupe(u8, output_prefix);
        errdefer self.allocator.free(owned_prefix);
        try self.ops.append(self.allocator, .{ .error_profile = .{
            .actual_name = owned_actual,
            .predicted_name = owned_predicted,
            .output_prefix = owned_prefix,
        } });
    }

    pub fn classificationProfile(
        self: *DeviceLazyFrame,
        actual_name: []const u8,
        predicted_name: []const u8,
        output_prefix: []const u8,
    ) DeviceDataError!void {
        const owned_actual = try self.allocator.dupe(u8, actual_name);
        errdefer self.allocator.free(owned_actual);
        const owned_predicted = try self.allocator.dupe(u8, predicted_name);
        errdefer self.allocator.free(owned_predicted);
        const owned_prefix = try self.allocator.dupe(u8, output_prefix);
        errdefer self.allocator.free(owned_prefix);
        try self.ops.append(self.allocator, .{ .classification_profile = .{
            .actual_name = owned_actual,
            .predicted_name = owned_predicted,
            .output_prefix = owned_prefix,
        } });
    }

    pub fn rollingCorrelationProfile(
        self: *DeviceLazyFrame,
        x_name: []const u8,
        y_name: []const u8,
        output_prefix: []const u8,
        options_value: DeviceRollingCorrelationOptions,
    ) DeviceDataError!void {
        const owned_x = try self.allocator.dupe(u8, x_name);
        errdefer self.allocator.free(owned_x);
        const owned_y = try self.allocator.dupe(u8, y_name);
        errdefer self.allocator.free(owned_y);
        const owned_prefix = try self.allocator.dupe(u8, output_prefix);
        errdefer self.allocator.free(owned_prefix);
        try self.ops.append(self.allocator, .{ .rolling_correlation_profile = .{
            .x_name = owned_x,
            .y_name = owned_y,
            .output_prefix = owned_prefix,
            .options = options_value,
        } });
    }

    pub fn validityProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8) DeviceDataError!void {
        const owned_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned_name);
        const owned_prefix = try self.allocator.dupe(u8, output_prefix);
        errdefer self.allocator.free(owned_prefix);
        try self.ops.append(self.allocator, .{ .validity_profile = .{
            .name = owned_name,
            .output_prefix = owned_prefix,
        } });
    }

    pub fn head(self: *DeviceLazyFrame, n: usize) DeviceDataError!void {
        try self.ops.append(self.allocator, .{ .head = n });
    }

    pub fn tail(self: *DeviceLazyFrame, n: usize) DeviceDataError!void {
        try self.ops.append(self.allocator, .{ .tail = n });
    }

    pub fn collect(self: DeviceLazyFrame) ParquetInteropError!DeviceDataFrame {
        var optimized = try self.optimizedOps();
        defer deinitLazyOps(self.allocator, &optimized);
        var current = try self.collectSource(optimized.items);
        errdefer current.deinit();
        for (optimized.items) |op| {
            const next = switch (op) {
                .select => |names| try current.select(names),
                .with_column_binary => |expr| blk: {
                    var column_value = try current.binaryColumns(expr.lhs_name, expr.rhs_name, expr.op);
                    defer column_value.deinit();
                    break :blk try current.withColumn(expr.name, column_value);
                },
                .with_column_scalar => |expr| blk: {
                    var column_value = try current.binaryColumnScalarWithDeviceScalar(expr.input_name, expr.scalar, expr.op);
                    defer column_value.deinit();
                    break :blk try current.withColumn(expr.name, column_value);
                },
                .with_column_compare => |expr| blk: {
                    var column_value = try current.compareColumns(expr.lhs_name, expr.rhs_name, expr.op);
                    defer column_value.deinit();
                    break :blk try current.withColumn(expr.name, column_value);
                },
                .with_column_compare_scalar => |expr| blk: {
                    var column_value = try current.compareColumnScalarWithDeviceScalar(expr.input_name, expr.scalar, expr.op);
                    defer column_value.deinit();
                    break :blk try current.withColumn(expr.name, column_value);
                },
                .filter_mask => |mask| try current.filterColumnMask(mask),
                .filter_scalar => |filter_op| blk: {
                    var mask = try current.compareColumnScalarWithDeviceScalar(filter_op.name, filter_op.scalar, filter_op.op);
                    defer mask.deinit();
                    break :blk try current.filterColumnMask(mask);
                },
                .group_by_count => |group| try current.groupByCount(group.key_name, group.output_name),
                .group_by_value => |group| switch (group.aggregation) {
                    .sum => try current.groupBySum(group.key_name, group.value_name, group.output_name),
                    .min => try current.groupByMin(group.key_name, group.value_name, group.output_name),
                    .max => try current.groupByMax(group.key_name, group.value_name, group.output_name),
                    .mean => try current.groupByMean(group.key_name, group.value_name, group.output_name),
                },
                .group_by_stats => |group| try current.groupByStats(group.key_name, group.value_name, group.output_prefix),
                .group_by_stats_on => |group| try current.groupByStatsOn(group.key_names, group.value_name, group.output_prefix),
                .group_by_profile => |group| try current.groupByProfile(group.key_name, group.value_name, group.output_prefix),
                .group_by_profile_on => |group| try current.groupByProfileOn(group.key_names, group.value_name, group.output_prefix),
                .join_on => |join| switch (join.kind) {
                    .inner => try current.innerJoinOn(join.right, join.left_key_names, join.right_key_names, join.options),
                    .left => try current.leftJoinOn(join.right, join.left_key_names, join.right_key_names, join.options),
                    .full => try current.fullJoinOn(join.right, join.left_key_names, join.right_key_names, join.options),
                    .semi => try current.semiJoinOn(join.right, join.left_key_names, join.right_key_names),
                    .anti => try current.antiJoinOn(join.right, join.left_key_names, join.right_key_names),
                },
                .asof_join => |join| try current.asofJoin(join.right, join.left_key_name, join.right_key_name, join.options),
                .concat_rows => |right| try current.concatRows(right),
                .distinct_rows => try current.distinctRows(),
                .distinct_on => |names| try current.distinctOn(names),
                .sort_by => |sort| try current.sortBy(sort.name, sort.options),
                .top_k => |top| try current.topKBy(top.name, top.k, top.options),
                .rank_profile_by => |rank| try current.rankProfileBy(rank.name, rank.output_prefix, rank.options),
                .rolling_profile => |rolling| try current.rollingProfile(rolling.name, rolling.output_prefix, rolling.options),
                .rolling_range_profile => |rolling| try current.rollingRangeProfile(rolling.name, rolling.output_prefix, rolling.options),
                .rolling_normalize_profile => |rolling| try current.rollingNormalizeProfile(rolling.name, rolling.output_prefix, rolling.options),
                .rolling_quantile_profile => |rolling| try current.rollingQuantileProfile(rolling.name, rolling.output_prefix, rolling.options),
                .lag_profile => |lag| try current.lagProfile(lag.name, lag.output_prefix, lag.options),
                .lead_profile => |lead| try current.leadProfile(lead.name, lead.output_prefix, lead.options),
                .clip_profile => |clip| try current.clipProfile(clip.name, clip.output_prefix, clip.options),
                .threshold_profile => |threshold| try current.thresholdProfile(threshold.name, threshold.output_prefix, threshold.options),
                .expanding_profile => |expanding| try current.expandingProfile(expanding.name, expanding.output_prefix, expanding.options),
                .standardize_profile => |standardize| try current.standardizeProfile(standardize.name, standardize.output_prefix, standardize.options),
                .robust_profile => |robust| try current.robustProfile(robust.name, robust.output_prefix, robust.options),
                .drawdown_profile => |drawdown| try current.drawdownProfile(drawdown.name, drawdown.output_prefix, drawdown.options),
                .extrema_profile => |extrema| try current.extremaProfile(extrema.name, extrema.output_prefix, extrema.options),
                .trend_profile => |trend| try current.trendProfile(trend.name, trend.output_prefix, trend.options),
                .sign_profile => |sign| try current.signProfile(sign.name, sign.output_prefix, sign.options),
                .crossover_profile => |cross| try current.crossoverProfile(cross.lhs_name, cross.rhs_name, cross.output_prefix, cross.options),
                .bucket_profile => |bucket| try current.bucketProfile(bucket.name, bucket.output_prefix, bucket.options),
                .ema_profile => |ema| try current.emaProfile(ema.name, ema.output_prefix, ema.options),
                .linear_fit_profile => |fit| try current.linearFitProfile(fit.x_name, fit.y_name, fit.output_prefix, fit.options),
                .error_profile => |err| try current.errorProfile(err.actual_name, err.predicted_name, err.output_prefix),
                .classification_profile => |class| try current.classificationProfile(class.actual_name, class.predicted_name, class.output_prefix),
                .rolling_correlation_profile => |corr| try current.rollingCorrelationProfile(corr.x_name, corr.y_name, corr.output_prefix, corr.options),
                .validity_profile => |validity| try current.validityProfile(validity.name, validity.output_prefix),
                .head => |n| try current.head(n),
                .tail => |n| try current.tail(n),
            };
            current.deinit();
            current = next;
        }
        return current;
    }

    pub fn explain(self: DeviceLazyFrame, allocator: std.mem.Allocator) DeviceDataError![]u8 {
        var optimized = try self.optimizedOps();
        defer deinitLazyOps(self.allocator, &optimized);
        var aw: std.Io.Writer.Allocating = .init(allocator);
        errdefer aw.deinit();
        try aw.writer.print("DeviceLazyFrame(raw_ops={d}, optimized_ops={d}, source={s})\n", .{ self.ops.items.len, optimized.items.len, self.source.name() });
        if (self.source == .parquet_scan) {
            var pushdown = try planLazyScanPushdown(self.allocator, optimized.items);
            defer pushdown.deinit();
            try aw.writer.print("  scan_pushdown: ", .{});
            try formatLazyScanPushdown(&aw.writer, pushdown);
            try aw.writer.print("\n", .{});
        }
        for (optimized.items, 0..) |op, i| {
            try aw.writer.print("  {d}: ", .{i});
            try formatLazyOp(&aw.writer, op);
            try aw.writer.print("\n", .{});
        }
        return aw.toOwnedSlice();
    }

    fn optimizedOps(self: DeviceLazyFrame) DeviceDataError!std.ArrayList(DeviceLazyOp) {
        var optimized: std.ArrayList(DeviceLazyOp) = .empty;
        errdefer deinitLazyOps(self.allocator, &optimized);
        for (self.ops.items) |op| {
            switch (op) {
                .select => |names| {
                    if (optimized.items.len != 0 and optimized.items[optimized.items.len - 1] == .select) {
                        const previous = optimized.items[optimized.items.len - 1].select;
                        if (allNamesIn(names, previous)) {
                            optimized.items[optimized.items.len - 1].deinit(self.allocator);
                            var cloned_op = try op.clone(self.allocator);
                            errdefer cloned_op.deinit(self.allocator);
                            optimized.items[optimized.items.len - 1] = cloned_op;
                            continue;
                        }
                    }
                },
                .head => |n| {
                    if (optimized.items.len != 0 and optimized.items[optimized.items.len - 1] == .sort_by) {
                        const sort = optimized.items[optimized.items.len - 1].sort_by;
                        const name = try self.allocator.dupe(u8, sort.name);
                        optimized.items[optimized.items.len - 1].deinit(self.allocator);
                        optimized.items[optimized.items.len - 1] = .{ .top_k = .{
                            .name = name,
                            .options = sort.options,
                            .k = n,
                        } };
                        continue;
                    }
                    if (optimized.items.len != 0 and optimized.items[optimized.items.len - 1] == .top_k) {
                        const top = optimized.items[optimized.items.len - 1].top_k;
                        optimized.items[optimized.items.len - 1] = .{ .top_k = .{
                            .name = top.name,
                            .options = top.options,
                            .k = @min(top.k, n),
                        } };
                        continue;
                    }
                    if (optimized.items.len != 0 and optimized.items[optimized.items.len - 1] == .head) {
                        const prev = optimized.items[optimized.items.len - 1].head;
                        optimized.items[optimized.items.len - 1] = .{ .head = @min(prev, n) };
                        continue;
                    }
                },
                .tail => |n| {
                    if (optimized.items.len != 0 and optimized.items[optimized.items.len - 1] == .tail) {
                        const prev = optimized.items[optimized.items.len - 1].tail;
                        optimized.items[optimized.items.len - 1] = .{ .tail = @min(prev, n) };
                        continue;
                    }
                },
                else => {},
            }
            var cloned_op = try op.clone(self.allocator);
            errdefer cloned_op.deinit(self.allocator);
            try optimized.append(self.allocator, cloned_op);
        }
        return optimized;
    }

    fn collectSource(self: DeviceLazyFrame, ops: []const DeviceLazyOp) ParquetInteropError!DeviceDataFrame {
        return switch (self.source) {
            .dataframe => |frame| try frame.clone(),
            .parquet_scan => |scan| blk: {
                var scan_plan = try scan.clone();
                defer scan_plan.deinit();

                var pushdown = try planLazyScanPushdown(self.allocator, ops);
                defer pushdown.deinit();
                if (pushdown.range_predicate) |predicate| {
                    try scan_plan.whereRange(predicate.column, predicate.predicate);
                }
                if (pushdown.projection) |names| {
                    try scan_plan.select(names);
                }

                break :blk try scan_plan.collect();
            },
        };
    }
};

pub const DeviceParquetScan = struct {
    allocator: std.mem.Allocator,
    bytes: []u8,
    device: array_mod.Device,
    projection: ?[][]const u8 = null,
    range_predicate: ?DeviceParquetRangeFilter = null,

    pub fn init(allocator: std.mem.Allocator, bytes: []const u8, device_value: array_mod.Device) std.mem.Allocator.Error!DeviceParquetScan {
        return .{
            .allocator = allocator,
            .bytes = try allocator.dupe(u8, bytes),
            .device = device_value,
        };
    }

    pub fn deinit(self: *DeviceParquetScan) void {
        self.allocator.free(self.bytes);
        if (self.projection) |names| freeNameList(self.allocator, names);
        if (self.range_predicate) |predicate| self.allocator.free(predicate.column);
        self.* = undefined;
    }

    pub fn clone(self: DeviceParquetScan) std.mem.Allocator.Error!DeviceParquetScan {
        var cloned = try DeviceParquetScan.init(self.allocator, self.bytes, self.device);
        errdefer cloned.deinit();
        if (self.projection) |names| try cloned.select(names);
        if (self.range_predicate) |predicate| try cloned.whereRange(predicate.column, predicate.predicate);
        return cloned;
    }

    pub fn lazy(self: DeviceParquetScan) DeviceDataError!DeviceLazyFrame {
        return DeviceLazyFrame.initParquetScan(self.allocator, self);
    }

    pub fn select(self: *DeviceParquetScan, names: []const []const u8) std.mem.Allocator.Error!void {
        if (self.projection) |old| freeNameList(self.allocator, old);
        self.projection = try cloneNameList(self.allocator, names);
    }

    pub fn whereRange(self: *DeviceParquetScan, column: []const u8, predicate: ParquetRangePredicate) std.mem.Allocator.Error!void {
        if (self.range_predicate) |old| self.allocator.free(old.column);
        self.range_predicate = .{
            .column = try self.allocator.dupe(u8, column),
            .predicate = predicate,
        };
    }

    pub fn collect(self: DeviceParquetScan) ParquetInteropError!DeviceDataFrame {
        var table = if (self.range_predicate) |predicate|
            try readBolthaTableWithRangePruning(self.allocator, self.bytes, predicate.column, predicate.predicate)
        else
            try boltha.parquet.readTable(self.allocator, self.bytes);
        defer table.deinit(self.allocator);

        if (self.projection) |names| {
            return DeviceDataFrame.fromArrowTableProjection(self.allocator, table, names, self.device);
        }
        return DeviceDataFrame.fromArrowTable(self.allocator, table, self.device);
    }

    pub fn explain(self: DeviceParquetScan, allocator: std.mem.Allocator) (std.mem.Allocator.Error || std.Io.Writer.Error)![]u8 {
        var aw: std.Io.Writer.Allocating = .init(allocator);
        errdefer aw.deinit();
        try aw.writer.print("DeviceParquetScan(bytes={d}, device={s}", .{ self.bytes.len, self.device.backendName() });
        if (self.range_predicate) |predicate| try aw.writer.print(", range={s}", .{predicate.column});
        if (self.projection) |names| {
            try aw.writer.print(", projection=[", .{});
            for (names, 0..) |name, i| {
                if (i != 0) try aw.writer.print(",", .{});
                try aw.writer.print("{s}", .{name});
            }
            try aw.writer.print("]", .{});
        }
        try aw.writer.print(")\n", .{});
        return aw.toOwnedSlice();
    }
};

fn deinitLazyOps(allocator: std.mem.Allocator, ops: *std.ArrayList(DeviceLazyOp)) void {
    for (ops.items) |*op| op.deinit(allocator);
    ops.deinit(allocator);
}

fn cloneNameList(allocator: std.mem.Allocator, names: []const []const u8) std.mem.Allocator.Error![][]const u8 {
    const owned = try allocator.alloc([]const u8, names.len);
    errdefer allocator.free(owned);
    var initialized: usize = 0;
    errdefer {
        for (owned[0..initialized]) |name| allocator.free(name);
    }
    for (names, owned) |name, *slot| {
        slot.* = try allocator.dupe(u8, name);
        initialized += 1;
    }
    return owned;
}

fn freeNameList(allocator: std.mem.Allocator, names: [][]const u8) void {
    for (names) |name| allocator.free(name);
    allocator.free(names);
}

const LazyScanPushdown = struct {
    allocator: std.mem.Allocator,
    projection: ?[][]const u8 = null,
    range_predicate: ?DeviceParquetRangeFilter = null,

    fn deinit(self: *LazyScanPushdown) void {
        if (self.projection) |names| freeNameList(self.allocator, names);
        if (self.range_predicate) |predicate| self.allocator.free(predicate.column);
        self.* = undefined;
    }
};

fn planLazyScanPushdown(allocator: std.mem.Allocator, ops: []const DeviceLazyOp) DeviceDataError!LazyScanPushdown {
    var required_names: std.ArrayList([]const u8) = .empty;
    errdefer required_names.deinit(allocator);
    errdefer freeOwnedNameItems(allocator, required_names.items);
    var derived_names: std.ArrayList([]const u8) = .empty;
    defer derived_names.deinit(allocator);

    var saw_select = false;
    var projection_blocked = false;
    var range_predicate: ?DeviceParquetRangeFilter = null;
    errdefer if (range_predicate) |predicate| allocator.free(predicate.column);

    op_loop: for (ops) |op| {
        switch (op) {
            .select => |names| {
                saw_select = true;
                for (names) |name| {
                    if (!nameInBorrowedList(name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, name);
                    }
                }
            },
            .with_column_binary => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                try appendOwnedNameUnique(allocator, &required_names, expr.lhs_name);
                try appendOwnedNameUnique(allocator, &required_names, expr.rhs_name);
            },
            .with_column_scalar => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
            },
            .with_column_compare => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                try appendOwnedNameUnique(allocator, &required_names, expr.lhs_name);
                try appendOwnedNameUnique(allocator, &required_names, expr.rhs_name);
            },
            .with_column_compare_scalar => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
            },
            .group_by_count => |group| {
                if (!nameInBorrowedList(group.key_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, group.key_name);
                }
                saw_select = true;
                break :op_loop;
            },
            .group_by_value => |group| {
                if (!nameInBorrowedList(group.key_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, group.key_name);
                }
                if (!nameInBorrowedList(group.value_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, group.value_name);
                }
                saw_select = true;
                break :op_loop;
            },
            .group_by_stats => |group| {
                if (!nameInBorrowedList(group.key_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, group.key_name);
                }
                if (!nameInBorrowedList(group.value_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, group.value_name);
                }
                saw_select = true;
                break :op_loop;
            },
            .group_by_stats_on => |group| {
                for (group.key_names) |key_name| {
                    if (!nameInBorrowedList(key_name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, key_name);
                    }
                }
                if (!nameInBorrowedList(group.value_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, group.value_name);
                }
                saw_select = true;
                break :op_loop;
            },
            .group_by_profile => |group| {
                if (!nameInBorrowedList(group.key_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, group.key_name);
                }
                if (!nameInBorrowedList(group.value_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, group.value_name);
                }
                saw_select = true;
                break :op_loop;
            },
            .group_by_profile_on => |group| {
                for (group.key_names) |key_name| {
                    if (!nameInBorrowedList(key_name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, key_name);
                    }
                }
                if (!nameInBorrowedList(group.value_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, group.value_name);
                }
                saw_select = true;
                break :op_loop;
            },
            .join_on => |join| {
                // A join changes the output schema by adding right-side payload
                // columns.  Without source schema metadata at this planning
                // layer, a later select cannot be safely split into left-source
                // columns vs. right payload columns.  Keep row-group predicate
                // pruning, but conservatively disable Parquet projection
                // pushdown for the left source rather than risk dropping a left
                // payload or requesting a right column from the source scan.
                projection_blocked = true;
                for (join.left_key_names) |key_name| {
                    if (!nameInBorrowedList(key_name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, key_name);
                    }
                }
                break :op_loop;
            },
            .asof_join => |join| {
                projection_blocked = true;
                if (!nameInBorrowedList(join.left_key_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, join.left_key_name);
                }
                break :op_loop;
            },
            .concat_rows => {
                break :op_loop;
            },
            .distinct_rows => {
                projection_blocked = true;
            },
            .distinct_on => |names| {
                for (names) |name| {
                    if (!nameInBorrowedList(name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, name);
                    }
                }
            },
            .filter_scalar => |filter_op| {
                const filter_depends_on_source = !nameInBorrowedList(filter_op.name, derived_names.items);
                if (filter_depends_on_source) try appendOwnedNameUnique(allocator, &required_names, filter_op.name);
                if (filter_depends_on_source and range_predicate == null) {
                    if (parquetRangePredicateFromScalar(filter_op.scalar, filter_op.op)) |predicate| {
                        range_predicate = .{
                            .column = try allocator.dupe(u8, filter_op.name),
                            .predicate = predicate,
                        };
                    }
                }
            },
            .sort_by => |sort| {
                if (!nameInBorrowedList(sort.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, sort.name);
            },
            .top_k => |top| {
                if (!nameInBorrowedList(top.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, top.name);
            },
            .rank_profile_by => |rank| {
                // A rank profile appends derived rank/window columns while
                // preserving the rest of the input table.  Without source schema
                // metadata here, a later select cannot be split safely into
                // source columns vs. rank-derived columns, so keep scalar
                // predicate pruning but avoid Parquet projection pushdown.
                projection_blocked = true;
                if (!nameInBorrowedList(rank.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, rank.name);
                break :op_loop;
            },
            .rolling_profile => |rolling| {
                // Rolling profiles append several derived columns and preserve
                // the existing table, so projection pushdown needs schema
                // awareness to avoid dropping later-selected source columns.
                projection_blocked = true;
                if (!nameInBorrowedList(rolling.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, rolling.name);
                break :op_loop;
            },
            .rolling_range_profile => |rolling| {
                // Rolling range profiles append low/high/range/position fields
                // and preserve the input table, so projection pushdown needs
                // schema awareness before it can safely pass this operation.
                projection_blocked = true;
                if (!nameInBorrowedList(rolling.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, rolling.name);
                break :op_loop;
            },
            .rolling_normalize_profile => |rolling| {
                // Rolling normalize profiles append window-local scaling fields
                // and preserve the input table, so projection pushdown needs
                // derived-field schema awareness before it can pass through.
                projection_blocked = true;
                if (!nameInBorrowedList(rolling.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, rolling.name);
                break :op_loop;
            },
            .rolling_quantile_profile => |rolling| {
                // Rolling quantile profiles append window distribution fields and
                // preserve the input table, so projection pushdown needs
                // generated-field schema awareness before it can pass through.
                projection_blocked = true;
                if (!nameInBorrowedList(rolling.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, rolling.name);
                break :op_loop;
            },
            .lag_profile => |lag| {
                // Lag profiles append multiple derived columns and preserve the
                // input schema.  Like rank/rolling profiles, keep scan predicate
                // pruning but avoid unsafe projection pushdown until the planner
                // has schema-level knowledge of derived vs. source fields.
                projection_blocked = true;
                if (!nameInBorrowedList(lag.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, lag.name);
                break :op_loop;
            },
            .lead_profile => |lead| {
                // Lead profiles append forward-looking derived columns and
                // preserve the input schema. Keep scan predicates, but block
                // projection pushdown across generated lead fields.
                projection_blocked = true;
                if (!nameInBorrowedList(lead.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, lead.name);
                break :op_loop;
            },
            .clip_profile => |clip| {
                // Clip profiles append cleaning diagnostics and preserve the
                // source column, so projection pushdown must wait for generated
                // field schema awareness.
                projection_blocked = true;
                if (!nameInBorrowedList(clip.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, clip.name);
                break :op_loop;
            },
            .threshold_profile => |threshold| {
                projection_blocked = true;
                if (!nameInBorrowedList(threshold.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, threshold.name);
                break :op_loop;
            },
            .expanding_profile => |expanding| {
                // Expanding profiles append cumulative derived columns while
                // preserving source columns.  Keep the source dependency for
                // scans, but do not push projection through this schema-changing
                // operation until planner schema metadata is richer.
                projection_blocked = true;
                if (!nameInBorrowedList(expanding.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, expanding.name);
                break :op_loop;
            },
            .standardize_profile => |standardize| {
                // Standardization adds derived scale columns while retaining the
                // input schema. It depends on the whole source column, so keep
                // predicate pruning but avoid unsafe projection pushdown until
                // derived-field schema metadata is available.
                projection_blocked = true;
                if (!nameInBorrowedList(standardize.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, standardize.name);
                break :op_loop;
            },
            .robust_profile => |robust| {
                // Robust profiles append median/MAD/IQR-derived columns while
                // preserving the input table. Keep predicate pruning but avoid
                // projection pushdown until derived-field schema tracking can
                // distinguish source and generated columns.
                projection_blocked = true;
                if (!nameInBorrowedList(robust.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, robust.name);
                break :op_loop;
            },
            .drawdown_profile => |drawdown| {
                // Drawdown profiles append sequence-derived columns while
                // preserving source fields. Keep scan predicates, but avoid
                // projection pushdown until source-vs-derived schema tracking is
                // rich enough to safely split later selects.
                projection_blocked = true;
                if (!nameInBorrowedList(drawdown.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, drawdown.name);
                break :op_loop;
            },
            .extrema_profile => |extrema| {
                projection_blocked = true;
                if (!nameInBorrowedList(extrema.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, extrema.name);
                break :op_loop;
            },
            .trend_profile => |trend| {
                // Trend profiles are row-order dependent and append several
                // derived columns. Preserve scan predicates, but block Parquet
                // projection pushdown until the planner can reason about
                // generated fields separately from source fields.
                projection_blocked = true;
                if (!nameInBorrowedList(trend.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, trend.name);
                break :op_loop;
            },
            .sign_profile => |sign| {
                projection_blocked = true;
                if (!nameInBorrowedList(sign.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, sign.name);
                break :op_loop;
            },
            .crossover_profile => |cross| {
                // Crossover profiles depend on two source columns and append
                // several signal columns. Keep scan predicates but block
                // projection pushdown until derived-field schema tracking can
                // safely split source and generated columns.
                projection_blocked = true;
                if (!nameInBorrowedList(cross.lhs_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, cross.lhs_name);
                if (!nameInBorrowedList(cross.rhs_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, cross.rhs_name);
                break :op_loop;
            },
            .bucket_profile => |bucket| {
                // Bucket profiles depend on the whole source distribution and
                // append several derived fields, so keep predicates but block
                // projection pushdown until generated-field schema metadata is
                // tracked explicitly.
                projection_blocked = true;
                if (!nameInBorrowedList(bucket.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, bucket.name);
                break :op_loop;
            },
            .ema_profile => |ema| {
                // EMA profiles are order-dependent and append derived columns,
                // so keep predicate pruning but block projection pushdown until
                // generated-field schema metadata is explicit.
                projection_blocked = true;
                if (!nameInBorrowedList(ema.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, ema.name);
                break :op_loop;
            },
            .linear_fit_profile => |fit| {
                // Linear-fit profiles depend on two source columns and append
                // model diagnostics. Keep predicate pruning but block projection
                // pushdown until generated-field schema metadata is explicit.
                projection_blocked = true;
                if (!nameInBorrowedList(fit.x_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, fit.x_name);
                if (!nameInBorrowedList(fit.y_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, fit.y_name);
                break :op_loop;
            },
            .error_profile => |err| {
                projection_blocked = true;
                if (!nameInBorrowedList(err.actual_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, err.actual_name);
                if (!nameInBorrowedList(err.predicted_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, err.predicted_name);
                break :op_loop;
            },
            .classification_profile => |class| {
                projection_blocked = true;
                if (!nameInBorrowedList(class.actual_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, class.actual_name);
                if (!nameInBorrowedList(class.predicted_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, class.predicted_name);
                break :op_loop;
            },
            .rolling_correlation_profile => |corr| {
                // Rolling correlation profiles depend on two source columns and
                // append several window diagnostics. Keep predicate pruning but
                // block projection pushdown until generated-field schema
                // metadata is explicit.
                projection_blocked = true;
                if (!nameInBorrowedList(corr.x_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, corr.x_name);
                if (!nameInBorrowedList(corr.y_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, corr.y_name);
                break :op_loop;
            },
            .validity_profile => |validity| {
                // Validity profiles are schema-changing data-quality diagnostics
                // over one source column. Keep source dependency for scans and
                // avoid projection pushdown across generated validity fields.
                projection_blocked = true;
                if (!nameInBorrowedList(validity.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, validity.name);
                break :op_loop;
            },
            .filter_mask, .head, .tail => {},
        }
    }

    const projection = if (saw_select and !projection_blocked) blk: {
        const owned = try required_names.toOwnedSlice(allocator);
        required_names = .empty;
        break :blk owned;
    } else null;
    if (projection == null) freeOwnedNameItems(allocator, required_names.items);
    required_names.deinit(allocator);

    const out = LazyScanPushdown{
        .allocator = allocator,
        .projection = projection,
        .range_predicate = range_predicate,
    };
    range_predicate = null;
    return out;
}

fn appendOwnedNameUnique(allocator: std.mem.Allocator, names: *std.ArrayList([]const u8), name: []const u8) std.mem.Allocator.Error!void {
    for (names.items) |existing| {
        if (std.mem.eql(u8, existing, name)) return;
    }
    const owned = try allocator.dupe(u8, name);
    errdefer allocator.free(owned);
    try names.append(allocator, owned);
}

fn appendBorrowedNameUnique(allocator: std.mem.Allocator, names: *std.ArrayList([]const u8), name: []const u8) std.mem.Allocator.Error!void {
    if (nameInBorrowedList(name, names.items)) return;
    try names.append(allocator, name);
}

fn nameInBorrowedList(name: []const u8, names: []const []const u8) bool {
    for (names) |existing| {
        if (std.mem.eql(u8, existing, name)) return true;
    }
    return false;
}

fn freeOwnedNameItems(allocator: std.mem.Allocator, names: []const []const u8) void {
    for (names) |name| allocator.free(name);
}

fn parquetRangePredicateFromScalar(scalar: DeviceScalar, op: DeviceColumnCompareOp) ?ParquetRangePredicate {
    return switch (scalar) {
        .bool => |value| blk: {
            const exact = switch (op) {
                .eq => value,
                .ne => !value,
                .gt, .ge, .lt, .le => break :blk null,
            };
            break :blk .{ .bool = .{ .min = exact, .max = exact } };
        },
        .i8 => |value| if (rangeFromScalarPredicate(i8, value, op)) |range| .{ .i8 = range } else null,
        .i16 => |value| if (rangeFromScalarPredicate(i16, value, op)) |range| .{ .i16 = range } else null,
        .i32 => |value| if (rangeFromScalarPredicate(i32, value, op)) |range| .{ .i32 = range } else null,
        .i64 => |value| if (rangeFromScalarPredicate(i64, value, op)) |range| .{ .i64 = range } else null,
        .u8 => |value| if (rangeFromScalarPredicate(u8, value, op)) |range| .{ .u8 = range } else null,
        .u16 => |value| if (rangeFromScalarPredicate(u16, value, op)) |range| .{ .u16 = range } else null,
        .u32 => |value| if (rangeFromScalarPredicate(u32, value, op)) |range| .{ .u32 = range } else null,
        .u64 => |value| if (rangeFromScalarPredicate(u64, value, op)) |range| .{ .u64 = range } else null,
        .usize => |value| if (rangeFromScalarPredicate(usize, value, op)) |range| .{ .usize = range } else null,
        .isize => |value| if (rangeFromScalarPredicate(isize, value, op)) |range| .{ .isize = range } else null,
        .f16 => |value| if (rangeFromScalarPredicate(f16, value, op)) |range| .{ .f16 = range } else null,
        .f32 => |value| if (rangeFromScalarPredicate(f32, value, op)) |range| .{ .f32 = range } else null,
        .f64 => |value| if (rangeFromScalarPredicate(f64, value, op)) |range| .{ .f64 = range } else null,
        .bf16, .c64, .c128 => null,
    };
}

fn rangeFromScalarPredicate(comptime T: type, value: T, op: DeviceColumnCompareOp) ?Range(T) {
    if (comptime @typeInfo(T) == .float) {
        if (std.math.isNan(value)) return null;
    }
    return switch (op) {
        .eq => .{ .min = value, .max = value },
        .gt, .ge => .{ .min = value },
        .lt, .le => .{ .max = value },
        .ne => null,
    };
}

fn allNamesIn(names: []const []const u8, allowed: []const []const u8) bool {
    for (names) |name| {
        var found = false;
        for (allowed) |candidate| {
            if (std.mem.eql(u8, name, candidate)) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    return true;
}

fn formatLazyScanPushdown(writer: *std.Io.Writer, pushdown: LazyScanPushdown) std.Io.Writer.Error!void {
    var printed = false;
    if (pushdown.range_predicate) |predicate| {
        try writer.print("range={s}", .{predicate.column});
        printed = true;
    }
    if (pushdown.projection) |names| {
        if (printed) try writer.print(", ", .{});
        try writer.print("projection=[", .{});
        for (names, 0..) |name, i| {
            if (i != 0) try writer.print(",", .{});
            try writer.print("{s}", .{name});
        }
        try writer.print("]", .{});
        printed = true;
    }
    if (!printed) try writer.print("none", .{});
}

fn formatLazyOp(writer: *std.Io.Writer, op: DeviceLazyOp) std.Io.Writer.Error!void {
    switch (op) {
        .select => |names| {
            try writer.print("select[", .{});
            for (names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("]", .{});
        },
        .filter_mask => |mask| try writer.print("filter_mask(dtype={s}, rows={d})", .{ mask.dtype().name(), mask.len() }),
        .filter_scalar => |filter_op| try writer.print("filter_scalar({s}, op={s}, dtype={s})", .{ filter_op.name, @tagName(filter_op.op), @tagName(filter_op.scalar) }),
        .with_column_binary => |expr| try writer.print("with_column_binary({s}={s} {s} {s})", .{ expr.name, expr.lhs_name, @tagName(expr.op), expr.rhs_name }),
        .with_column_scalar => |expr| try writer.print("with_column_scalar({s}={s} {s} scalar:{s})", .{ expr.name, expr.input_name, @tagName(expr.op), @tagName(expr.scalar) }),
        .with_column_compare => |expr| try writer.print("with_column_compare({s}={s} {s} {s})", .{ expr.name, expr.lhs_name, @tagName(expr.op), expr.rhs_name }),
        .with_column_compare_scalar => |expr| try writer.print("with_column_compare_scalar({s}={s} {s} scalar:{s})", .{ expr.name, expr.input_name, @tagName(expr.op), @tagName(expr.scalar) }),
        .group_by_count => |group| try writer.print("group_by_count({s} -> {s})", .{ group.key_name, group.output_name }),
        .group_by_value => |group| try writer.print("group_by_{s}({s}, value={s} -> {s})", .{ @tagName(group.aggregation), group.key_name, group.value_name, group.output_name }),
        .group_by_stats => |group| try writer.print("group_by_stats({s}, value={s}, prefix={s})", .{ group.key_name, group.value_name, group.output_prefix }),
        .group_by_stats_on => |group| {
            try writer.print("group_by_stats_on([", .{});
            for (group.key_names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("], value={s}, prefix={s})", .{ group.value_name, group.output_prefix });
        },
        .group_by_profile => |group| try writer.print("group_by_profile({s}, value={s}, prefix={s})", .{ group.key_name, group.value_name, group.output_prefix }),
        .group_by_profile_on => |group| {
            try writer.print("group_by_profile_on([", .{});
            for (group.key_names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("], value={s}, prefix={s})", .{ group.value_name, group.output_prefix });
        },
        .join_on => |join| {
            try writer.print("{s}_join_on(left=[", .{@tagName(join.kind)});
            for (join.left_key_names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("], right=[", .{});
            for (join.right_key_names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("])", .{});
        },
        .asof_join => |join| try writer.print("asof_join({s}->{s}, strategy={s})", .{ join.left_key_name, join.right_key_name, @tagName(join.options.strategy) }),
        .concat_rows => |right| try writer.print("concat_rows(rows={d}, cols={d})", .{ right.height(), right.width() }),
        .distinct_rows => try writer.print("distinct_rows", .{}),
        .distinct_on => |names| {
            try writer.print("distinct_on([", .{});
            for (names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("])", .{});
        },
        .sort_by => |sort| try writer.print("sort_by({s}, desc={})", .{ sort.name, sort.options.descending }),
        .top_k => |top| try writer.print("top_k({s}, k={d}, desc={})", .{ top.name, top.k, top.options.descending }),
        .rank_profile_by => |rank| try writer.print("rank_profile_by({s}, prefix={s}, desc={})", .{ rank.name, rank.output_prefix, rank.options.descending }),
        .rolling_profile => |rolling| try writer.print("rolling_profile({s}, prefix={s}, window={d})", .{ rolling.name, rolling.output_prefix, rolling.options.window }),
        .rolling_range_profile => |rolling| try writer.print("rolling_range_profile({s}, prefix={s}, window={d})", .{ rolling.name, rolling.output_prefix, rolling.options.window }),
        .rolling_normalize_profile => |rolling| try writer.print("rolling_normalize_profile({s}, prefix={s}, window={d})", .{ rolling.name, rolling.output_prefix, rolling.options.window }),
        .rolling_quantile_profile => |rolling| try writer.print("rolling_quantile_profile({s}, prefix={s}, window={d})", .{ rolling.name, rolling.output_prefix, rolling.options.window }),
        .lag_profile => |lag| try writer.print("lag_profile({s}, prefix={s}, periods={d})", .{ lag.name, lag.output_prefix, lag.options.periods }),
        .lead_profile => |lead| try writer.print("lead_profile({s}, prefix={s}, periods={d})", .{ lead.name, lead.output_prefix, lead.options.periods }),
        .clip_profile => |clip| try writer.print("clip_profile({s}, prefix={s}, [{d},{d}])", .{ clip.name, clip.output_prefix, clip.options.lower, clip.options.upper }),
        .threshold_profile => |threshold| try writer.print("threshold_profile({s}, prefix={s}, threshold={d})", .{ threshold.name, threshold.output_prefix, threshold.options.threshold }),
        .expanding_profile => |expanding| try writer.print("expanding_profile({s}, prefix={s}, min_periods={d})", .{ expanding.name, expanding.output_prefix, expanding.options.min_periods }),
        .standardize_profile => |standardize| try writer.print("standardize_profile({s}, prefix={s}, min_periods={d})", .{ standardize.name, standardize.output_prefix, standardize.options.min_periods }),
        .robust_profile => |robust| try writer.print("robust_profile({s}, prefix={s}, min_periods={d})", .{ robust.name, robust.output_prefix, robust.options.min_periods }),
        .drawdown_profile => |drawdown| try writer.print("drawdown_profile({s}, prefix={s}, min_periods={d})", .{ drawdown.name, drawdown.output_prefix, drawdown.options.min_periods }),
        .extrema_profile => |extrema| try writer.print("extrema_profile({s}, prefix={s}, min_periods={d})", .{ extrema.name, extrema.output_prefix, extrema.options.min_periods }),
        .trend_profile => |trend| try writer.print("trend_profile({s}, prefix={s}, periods={d})", .{ trend.name, trend.output_prefix, trend.options.periods }),
        .sign_profile => |sign| try writer.print("sign_profile({s}, prefix={s}, periods={d})", .{ sign.name, sign.output_prefix, sign.options.periods }),
        .crossover_profile => |cross| try writer.print("crossover_profile({s},{s}, prefix={s}, periods={d})", .{ cross.lhs_name, cross.rhs_name, cross.output_prefix, cross.options.periods }),
        .bucket_profile => |bucket| try writer.print("bucket_profile({s}, prefix={s}, buckets={d})", .{ bucket.name, bucket.output_prefix, bucket.options.buckets }),
        .ema_profile => |ema| try writer.print("ema_profile({s}, prefix={s}, alpha={d})", .{ ema.name, ema.output_prefix, ema.options.alpha }),
        .linear_fit_profile => |fit| try writer.print("linear_fit_profile({s}->{s}, prefix={s})", .{ fit.x_name, fit.y_name, fit.output_prefix }),
        .error_profile => |err| try writer.print("error_profile(actual={s}, predicted={s}, prefix={s})", .{ err.actual_name, err.predicted_name, err.output_prefix }),
        .classification_profile => |class| try writer.print("classification_profile(actual={s}, predicted={s}, prefix={s})", .{ class.actual_name, class.predicted_name, class.output_prefix }),
        .rolling_correlation_profile => |corr| try writer.print("rolling_correlation_profile({s},{s}, prefix={s}, window={d})", .{ corr.x_name, corr.y_name, corr.output_prefix, corr.options.window }),
        .validity_profile => |validity| try writer.print("validity_profile({s}, prefix={s})", .{ validity.name, validity.output_prefix }),
        .head => |n| try writer.print("head({d})", .{n}),
        .tail => |n| try writer.print("tail({d})", .{n}),
    }
}

/// Owning fixed-width dataframe that can keep every column on the same Vectra
/// device.
///
/// This is intentionally an owning/table wrapper rather than a CUDA-only API:
/// `.cpu`, `.cuda(index)`, and `.mps(index)` use the same metadata and column
/// invariants.  CUDA/MPS row slicing and host-mask filtering currently
/// materialize through host memory because Vectra has not yet grown
/// dataframe-specific gather/compact kernels; preserving the operation behind
/// this API gives those kernels a single integration point later.
pub const DeviceDataFrame = struct {
    allocator: std.mem.Allocator,
    names: [][]const u8,
    columns: []DeviceColumn,
    rows: usize,
    device: array_mod.Device,

    pub fn init(allocator: std.mem.Allocator, defs: []const DeviceColumnDef) DeviceDataError!DeviceDataFrame {
        if (defs.len == 0) return DeviceDataFrame.initEmpty(allocator, 0, .cpu);
        const rows = defs[0].data.len();
        const device_value = defs[0].data.device();
        for (defs) |def| {
            if (def.data.len() != rows) return error.LengthMismatch;
            if (!def.data.device().sameDevice(device_value)) return error.InvalidDevice;
        }

        var names = try allocator.alloc([]const u8, defs.len);
        errdefer allocator.free(names);
        var columns = try allocator.alloc(DeviceColumn, defs.len);
        errdefer allocator.free(columns);

        var initialized: usize = 0;
        errdefer {
            for (0..initialized) |i| {
                allocator.free(names[i]);
                columns[i].deinit();
            }
        }

        for (defs, 0..) |def, i| {
            names[i] = try allocator.dupe(u8, def.name);
            columns[i] = try def.data.clone();
            initialized += 1;
        }

        return .{ .allocator = allocator, .names = names, .columns = columns, .rows = rows, .device = device_value };
    }

    pub fn initEmpty(allocator: std.mem.Allocator, rows: usize, device_value: array_mod.Device) DeviceDataError!DeviceDataFrame {
        if (!device_value.isAvailable()) return error.InvalidDevice;
        return .{ .allocator = allocator, .names = &.{}, .columns = &.{}, .rows = rows, .device = device_value };
    }

    pub fn fromDataFrame(allocator: std.mem.Allocator, frame: DataFrame, device_value: array_mod.Device) DeviceDataError!DeviceDataFrame {
        if (!device_value.isAvailable()) return error.InvalidDevice;
        if (frame.columns.len == 0) return DeviceDataFrame.initEmpty(allocator, frame.rows, device_value);
        var defs = try allocator.alloc(DeviceColumnDef, frame.columns.len);
        defer allocator.free(defs);
        var initialized: usize = 0;
        defer {
            for (defs[0..initialized]) |*def| def.data.deinit();
        }
        for (frame.names, frame.columns, 0..) |name, col, i| {
            defs[i].name = name;
            defs[i].data = switch (col) {
                .f64 => |values| try DeviceColumn.fromSlice(f64, allocator, values, device_value),
                .i64 => |values| try DeviceColumn.fromSlice(i64, allocator, values, device_value),
                .bool => |values| try DeviceColumn.fromSlice(bool, allocator, values, device_value),
                .string => return error.TypeUnsupported,
            };
            initialized += 1;
        }
        return DeviceDataFrame.init(allocator, defs);
    }

    pub fn deinit(self: *DeviceDataFrame) void {
        for (self.names) |name| self.allocator.free(name);
        for (self.columns) |*col| col.deinit();
        if (self.names.len != 0) self.allocator.free(self.names);
        if (self.columns.len != 0) self.allocator.free(self.columns);
        self.* = undefined;
    }

    pub fn clone(self: DeviceDataFrame) DeviceDataError!DeviceDataFrame {
        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        return initDeviceDataFrameFromOwnedColumns(self.allocator, self.names, columns, self.rows, self.device);
    }

    pub fn height(self: DeviceDataFrame) usize {
        return self.rows;
    }

    pub fn width(self: DeviceDataFrame) usize {
        return self.columns.len;
    }

    pub fn shape(self: DeviceDataFrame) struct { rows: usize, cols: usize } {
        return .{ .rows = self.rows, .cols = self.columns.len };
    }

    pub fn columnIndex(self: DeviceDataFrame, name: []const u8) ?usize {
        for (self.names, 0..) |existing, i| {
            if (std.mem.eql(u8, existing, name)) return i;
        }
        return null;
    }

    pub fn column(self: *const DeviceDataFrame, name: []const u8) DataError!*const DeviceColumn {
        const idx = self.columnIndex(name) orelse return error.ColumnNotFound;
        return &self.columns[idx];
    }

    pub fn columnDType(self: DeviceDataFrame, name: []const u8) DataError!DeviceDType {
        const idx = self.columnIndex(name) orelse return error.ColumnNotFound;
        return self.columns[idx].dtype();
    }

    pub fn binaryColumns(self: DeviceDataFrame, lhs_name: []const u8, rhs_name: []const u8, op: DeviceColumnBinaryOp) DeviceDataError!DeviceColumn {
        const lhs = try self.column(lhs_name);
        const rhs = try self.column(rhs_name);
        return lhs.binary(rhs.*, op);
    }

    pub fn addColumns(self: DeviceDataFrame, lhs_name: []const u8, rhs_name: []const u8) DeviceDataError!DeviceColumn {
        return self.binaryColumns(lhs_name, rhs_name, .add);
    }

    pub fn subColumns(self: DeviceDataFrame, lhs_name: []const u8, rhs_name: []const u8) DeviceDataError!DeviceColumn {
        return self.binaryColumns(lhs_name, rhs_name, .sub);
    }

    pub fn mulColumns(self: DeviceDataFrame, lhs_name: []const u8, rhs_name: []const u8) DeviceDataError!DeviceColumn {
        return self.binaryColumns(lhs_name, rhs_name, .mul);
    }

    pub fn divColumns(self: DeviceDataFrame, lhs_name: []const u8, rhs_name: []const u8) DeviceDataError!DeviceColumn {
        return self.binaryColumns(lhs_name, rhs_name, .div);
    }

    pub fn binaryColumnScalar(self: DeviceDataFrame, name: []const u8, comptime T: type, scalar: T, op: DeviceColumnBinaryOp) DeviceDataError!DeviceColumn {
        const col = try self.column(name);
        return col.binaryScalar(T, scalar, op);
    }

    pub fn binaryColumnScalarWithDeviceScalar(self: DeviceDataFrame, name: []const u8, scalar: DeviceScalar, op: DeviceColumnBinaryOp) DeviceDataError!DeviceColumn {
        const col = try self.column(name);
        return switch (scalar) {
            .i8 => |value| col.binaryScalar(i8, value, op),
            .i16 => |value| col.binaryScalar(i16, value, op),
            .i32 => |value| col.binaryScalar(i32, value, op),
            .i64 => |value| col.binaryScalar(i64, value, op),
            .u8 => |value| col.binaryScalar(u8, value, op),
            .u16 => |value| col.binaryScalar(u16, value, op),
            .u32 => |value| col.binaryScalar(u32, value, op),
            .u64 => |value| col.binaryScalar(u64, value, op),
            .usize => |value| col.binaryScalar(usize, value, op),
            .isize => |value| col.binaryScalar(isize, value, op),
            .f16 => |value| col.binaryScalar(f16, value, op),
            .f32 => |value| col.binaryScalar(f32, value, op),
            .f64 => |value| col.binaryScalar(f64, value, op),
            .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
        };
    }

    pub fn compareColumns(self: DeviceDataFrame, lhs_name: []const u8, rhs_name: []const u8, op: DeviceColumnCompareOp) DeviceDataError!DeviceColumn {
        const lhs = try self.column(lhs_name);
        const rhs = try self.column(rhs_name);
        return lhs.compare(rhs.*, op);
    }

    pub fn compareColumnScalar(self: DeviceDataFrame, name: []const u8, comptime T: type, scalar: T, op: DeviceColumnCompareOp) DeviceDataError!DeviceColumn {
        const col = try self.column(name);
        return col.compareScalar(T, scalar, op);
    }

    pub fn compareColumnScalarWithDeviceScalar(self: DeviceDataFrame, name: []const u8, scalar: DeviceScalar, op: DeviceColumnCompareOp) DeviceDataError!DeviceColumn {
        const col = try self.column(name);
        return switch (scalar) {
            .bool => |value| col.compareScalar(bool, value, op),
            .i8 => |value| col.compareScalar(i8, value, op),
            .i16 => |value| col.compareScalar(i16, value, op),
            .i32 => |value| col.compareScalar(i32, value, op),
            .i64 => |value| col.compareScalar(i64, value, op),
            .u8 => |value| col.compareScalar(u8, value, op),
            .u16 => |value| col.compareScalar(u16, value, op),
            .u32 => |value| col.compareScalar(u32, value, op),
            .u64 => |value| col.compareScalar(u64, value, op),
            .usize => |value| col.compareScalar(usize, value, op),
            .isize => |value| col.compareScalar(isize, value, op),
            .f16 => |value| col.compareScalar(f16, value, op),
            .f32 => |value| col.compareScalar(f32, value, op),
            .f64 => |value| col.compareScalar(f64, value, op),
            .bf16, .c64, .c128 => error.TypeUnsupported,
        };
    }

    pub fn filterColumnMask(self: DeviceDataFrame, mask: DeviceColumn) DeviceDataError!DeviceDataFrame {
        const typed_mask = switch (mask) {
            .bool => |typed| typed,
            else => return error.TypeMismatch,
        };
        if (!typed_mask.device().sameDevice(self.device)) return error.InvalidDevice;
        if (typed_mask.len() != self.rows) return error.LengthMismatch;
        const host_values = try typed_mask.values.toOwnedSlice(self.allocator);
        defer self.allocator.free(host_values);
        if (typed_mask.validity) |validity_array| {
            const host_validity = try validity_array.toOwnedSlice(self.allocator);
            defer self.allocator.free(host_validity);
            const host_mask = try self.allocator.alloc(bool, self.rows);
            defer self.allocator.free(host_mask);
            for (host_values, host_validity, host_mask) |value, valid, *slot| {
                // Match dataframe query semantics used by Polars/Arrow engines:
                // a null predicate does not select the row.  Keeping the mask
                // resolution here makes scalar-filter and Parquet statistics
                // pushdown conservative even when predicate columns are nullable.
                slot.* = valid and value;
            }
            return self.filter(host_mask);
        }
        return self.filter(host_values);
    }

    /// Export a Boltha/Arrow schema for the fixed-width device dataframe.
    ///
    /// Polars keeps Arrow as the explicit columnar interchange boundary:
    /// dataframe fields describe logical dtype and nullability, while each
    /// column exports its Arrow array independently.  Vectra follows the same
    /// split here, but the source columns may live on CPU, CUDA, or MPS; array
    /// bytes are downloaded/materialized only when `toArrowRecordBatch()` is
    /// called.
    pub fn toArrowSchema(self: DeviceDataFrame, allocator: std.mem.Allocator) ArrowInteropError!boltha.arrow.Schema {
        var fields = try allocator.alloc(boltha.arrow.Field, self.columns.len);
        defer allocator.free(fields);
        var initialized: usize = 0;
        defer {
            for (fields[0..initialized]) |*field| field.deinit(allocator);
        }

        for (self.names, self.columns, 0..) |name, col, i| {
            fields[i] = try boltha.arrow.Field.init(
                allocator,
                name,
                try col.arrowDataType(),
                col.nullable(),
            );
            initialized += 1;
        }
        return boltha.arrow.Schema.init(allocator, fields);
    }

    /// Materialize the dataframe as a single Boltha/Arrow record batch.
    ///
    /// This is the host-side interoperability boundary for Arrow IPC/Parquet
    /// and external consumers.  Device-resident columns remain authoritative in
    /// `DeviceDataFrame`; export downloads through `Array.toOwnedSlice()` so the
    /// same method works on CPU, CUDA, and MPS.  A zero-column table with a
    /// non-zero row count is representable by `DeviceDataFrameView`, but Boltha's
    /// current `RecordBatch` has no explicit row-count constructor for that
    /// case, so export rejects it instead of silently dropping rows.
    pub fn toArrowRecordBatch(self: DeviceDataFrame, allocator: std.mem.Allocator) ArrowInteropError!boltha.arrow.RecordBatch {
        if (self.columns.len == 0 and self.rows != 0) return error.TypeUnsupported;

        var schema = try self.toArrowSchema(allocator);
        errdefer schema.deinit(allocator);

        const columns = try allocator.alloc(boltha.arrow.AnyArray, self.columns.len);
        errdefer allocator.free(columns);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*column_value| column_value.deinit(allocator);
        }

        for (self.columns, columns) |col, *slot| {
            slot.* = try col.toArrowArray(allocator);
            initialized += 1;
        }
        return boltha.arrow.RecordBatch.initOwned(schema, columns);
    }

    pub fn toArrowTable(self: DeviceDataFrame, allocator: std.mem.Allocator) ArrowInteropError!boltha.arrow.Table {
        var batch = try self.toArrowRecordBatch(allocator);
        errdefer batch.deinit(allocator);

        var schema = try batch.schema.clone(allocator);
        errdefer schema.deinit(allocator);

        const batches = try allocator.alloc(boltha.arrow.RecordBatch, 1);
        errdefer allocator.free(batches);
        batches[0] = batch;
        return boltha.arrow.Table.initOwned(schema, batches);
    }

    pub fn toParquetBytes(self: DeviceDataFrame, allocator: std.mem.Allocator) ParquetInteropError![]u8 {
        var batch = try self.toArrowRecordBatch(allocator);
        defer batch.deinit(allocator);
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(allocator);
        try boltha.parquet.writeRecordBatch(allocator, &out, batch);
        return out.toOwnedSlice(allocator);
    }

    pub fn fromParquetBytes(allocator: std.mem.Allocator, bytes: []const u8, device_value: array_mod.Device) ParquetInteropError!DeviceDataFrame {
        var batch = try boltha.parquet.readRecordBatch(allocator, bytes);
        defer batch.deinit(allocator);
        return DeviceDataFrame.fromArrowRecordBatch(allocator, batch, device_value);
    }

    pub fn fromParquetBytesPruned(
        allocator: std.mem.Allocator,
        bytes: []const u8,
        column_name: []const u8,
        predicate: ParquetRangePredicate,
        device_value: array_mod.Device,
    ) ParquetInteropError!DeviceDataFrame {
        var table = try readBolthaTableWithRangePruning(allocator, bytes, column_name, predicate);
        defer table.deinit(allocator);
        return DeviceDataFrame.fromArrowTable(allocator, table, device_value);
    }

    pub fn fromArrowTable(allocator: std.mem.Allocator, table: boltha.arrow.Table, device_value: array_mod.Device) ArrowInteropError!DeviceDataFrame {
        if (table.batches.len == 0) return emptyFromArrowSchema(allocator, table.schema, table.row_count, device_value);
        var out = try DeviceDataFrame.fromArrowRecordBatch(allocator, table.batches[0], device_value);
        errdefer out.deinit();
        for (table.batches[1..]) |batch| {
            var next = try DeviceDataFrame.fromArrowRecordBatch(allocator, batch, device_value);
            defer next.deinit();
            const combined = try concatDeviceDataFramesRows(out, next);
            out.deinit();
            out = combined;
        }
        return out;
    }

    pub fn fromArrowTableProjection(
        allocator: std.mem.Allocator,
        table: boltha.arrow.Table,
        wanted_names: []const []const u8,
        device_value: array_mod.Device,
    ) ArrowInteropError!DeviceDataFrame {
        if (wanted_names.len == 0) return DeviceDataFrame.initEmpty(allocator, table.row_count, device_value);
        if (table.batches.len == 0) return emptyFromArrowSchemaProjection(allocator, table.schema, table.row_count, wanted_names, device_value);

        // Projection is applied while crossing the Arrow -> DeviceDataFrame
        // boundary.  Boltha's current simple Parquet reader still decodes full
        // row groups, but dropped columns are not uploaded/materialized into
        // Vectra CPU/CUDA/MPS arrays.  That keeps the public scan plan aligned
        // with Polars-style projection pushdown while leaving a narrow seam for
        // a future Boltha column-projection reader.
        var out = try DeviceDataFrame.fromArrowRecordBatchProjection(allocator, table.batches[0], wanted_names, device_value);
        errdefer out.deinit();
        for (table.batches[1..]) |batch| {
            var next = try DeviceDataFrame.fromArrowRecordBatchProjection(allocator, batch, wanted_names, device_value);
            defer next.deinit();
            const combined = try concatDeviceDataFramesRows(out, next);
            out.deinit();
            out = combined;
        }
        return out;
    }

    pub fn fromArrowRecordBatch(allocator: std.mem.Allocator, batch: boltha.arrow.RecordBatch, device_value: array_mod.Device) ArrowInteropError!DeviceDataFrame {
        if (!device_value.isAvailable()) return error.InvalidDevice;
        var defs = try allocator.alloc(DeviceColumnDef, batch.columns.len);
        defer allocator.free(defs);
        var initialized: usize = 0;
        defer {
            for (defs[0..initialized]) |*def| def.data.deinit();
        }
        for (batch.schema.fields, batch.columns, 0..) |field, arrow_column, i| {
            defs[i] = .{
                .name = field.name,
                .data = try deviceColumnFromArrowArray(allocator, arrow_column, device_value),
            };
            initialized += 1;
        }
        return DeviceDataFrame.init(allocator, defs);
    }

    pub fn fromArrowRecordBatchProjection(
        allocator: std.mem.Allocator,
        batch: boltha.arrow.RecordBatch,
        wanted_names: []const []const u8,
        device_value: array_mod.Device,
    ) ArrowInteropError!DeviceDataFrame {
        if (!device_value.isAvailable()) return error.InvalidDevice;
        if (wanted_names.len == 0) return DeviceDataFrame.initEmpty(allocator, batch.row_count, device_value);

        var defs = try allocator.alloc(DeviceColumnDef, wanted_names.len);
        defer allocator.free(defs);
        var initialized: usize = 0;
        defer {
            for (defs[0..initialized]) |*def| def.data.deinit();
        }
        for (wanted_names, 0..) |name, i| {
            const column_index = batch.schema.fieldIndexByName(name) orelse return error.ColumnNotFound;
            defs[i] = .{
                .name = batch.schema.fields[column_index].name,
                .data = try deviceColumnFromArrowArray(allocator, batch.columns[column_index], device_value),
            };
            initialized += 1;
        }
        return DeviceDataFrame.init(allocator, defs);
    }

    pub fn view(self: DeviceDataFrame) DeviceDataError!DeviceDataFrameView {
        const columns = try self.allocator.alloc(DeviceColumnView, self.columns.len);
        errdefer self.allocator.free(columns);
        for (self.columns, columns) |col, *slot| slot.* = col.view();
        return .{
            .allocator = self.allocator,
            .names = self.names,
            .columns = columns,
            .rows = self.rows,
            .device = self.device,
        };
    }

    pub fn select(self: DeviceDataFrame, wanted_names: []const []const u8) DeviceDataError!DeviceDataFrame {
        if (wanted_names.len == 0) return DeviceDataFrame.initEmpty(self.allocator, self.rows, self.device);
        var columns = try self.allocator.alloc(DeviceColumn, wanted_names.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (wanted_names, 0..) |name, i| {
            const source = try self.column(name);
            columns[i] = try source.clone();
            initialized += 1;
        }
        return initDeviceDataFrameFromOwnedColumns(self.allocator, wanted_names, columns, self.rows, self.device);
    }

    pub fn withColumn(self: DeviceDataFrame, name: []const u8, data: DeviceColumn) DeviceDataError!DeviceDataFrame {
        if (data.len() != self.rows) return error.LengthMismatch;
        if (!data.device().sameDevice(self.device)) return error.InvalidDevice;
        var source_names = try self.allocator.alloc([]const u8, self.columns.len + 1);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |existing, i| source_names[i] = existing;
        source_names[self.columns.len] = name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + 1);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        columns[self.columns.len] = try data.clone();
        initialized += 1;
        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn head(self: DeviceDataFrame, n: usize) DeviceDataError!DeviceDataFrame {
        return self.sliceRows(0, @min(n, self.rows));
    }

    pub fn tail(self: DeviceDataFrame, n: usize) DeviceDataError!DeviceDataFrame {
        const count = @min(n, self.rows);
        return self.sliceRows(self.rows - count, self.rows);
    }

    pub fn sliceRows(self: DeviceDataFrame, start: usize, stop: usize) DeviceDataError!DeviceDataFrame {
        const end = @min(stop, self.rows);
        const begin = @min(start, end);
        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.sliceRows(begin, end);
            initialized += 1;
        }
        return initDeviceDataFrameFromOwnedColumns(self.allocator, self.names, columns, end - begin, self.device);
    }

    pub fn take(self: DeviceDataFrame, row_indices: []const usize) DeviceDataError!DeviceDataFrame {
        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.take(row_indices);
            initialized += 1;
        }
        return initDeviceDataFrameFromOwnedColumns(self.allocator, self.names, columns, row_indices.len, self.device);
    }

    pub fn concatRows(self: DeviceDataFrame, other: DeviceDataFrame) DeviceDataError!DeviceDataFrame {
        return concatDeviceDataFramesRows(self, other);
    }

    pub fn appendRows(self: DeviceDataFrame, other: DeviceDataFrame) DeviceDataError!DeviceDataFrame {
        return self.concatRows(other);
    }

    pub fn vstack(self: DeviceDataFrame, other: DeviceDataFrame) DeviceDataError!DeviceDataFrame {
        return self.concatRows(other);
    }

    pub fn distinctRows(self: DeviceDataFrame) DeviceDataError!DeviceDataFrame {
        return self.distinctOn(self.names);
    }

    pub fn distinctOn(self: DeviceDataFrame, key_names: []const []const u8) DeviceDataError!DeviceDataFrame {
        const indices = try distinctRowIndices(self.allocator, self, key_names);
        defer self.allocator.free(indices);
        return self.take(indices);
    }

    pub fn dropDuplicates(self: DeviceDataFrame) DeviceDataError!DeviceDataFrame {
        return self.distinctRows();
    }

    pub fn dropDuplicatesOn(self: DeviceDataFrame, key_names: []const []const u8) DeviceDataError!DeviceDataFrame {
        return self.distinctOn(key_names);
    }

    pub fn uniqueRows(self: DeviceDataFrame) DeviceDataError!DeviceDataFrame {
        return self.distinctRows();
    }

    pub fn argsortBy(self: DeviceDataFrame, name: []const u8, options_value: DeviceSortOptions) DeviceDataError![]usize {
        const sort_key = try self.column(name);
        return sort_key.argsort(self.allocator, options_value);
    }

    pub fn sortBy(self: DeviceDataFrame, name: []const u8, options_value: DeviceSortOptions) DeviceDataError!DeviceDataFrame {
        const order = try self.argsortBy(name, options_value);
        defer self.allocator.free(order);
        return self.take(order);
    }

    pub fn sortByColumn(self: DeviceDataFrame, name: []const u8, options_value: DeviceSortOptions) DeviceDataError!DeviceDataFrame {
        return self.sortBy(name, options_value);
    }

    pub fn topKBy(self: DeviceDataFrame, name: []const u8, k: usize, options_value: DeviceSortOptions) DeviceDataError!DeviceDataFrame {
        var sorted = try self.sortBy(name, options_value);
        defer sorted.deinit();
        return sorted.head(k);
    }

    pub fn rankProfileBy(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceSortOptions) DeviceDataError!DeviceDataFrame {
        const rank_key = try self.column(name);
        var rank_columns = try rankProfileColumnsByKey(self.allocator, rank_key.*, options_value, self.device, self.rows);
        var rank_columns_transferred: usize = 0;
        errdefer {
            for (rank_columns[rank_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + rank_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var rank_names = try rankProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, rank_names[0..]);
        for (rank_names, 0..) |rank_name, i| source_names[self.columns.len + i] = rank_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + rank_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&rank_columns) |*rank_col| {
            columns[initialized] = rank_col.*;
            initialized += 1;
            rank_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn rollingProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingOptions) DeviceDataError!DeviceDataFrame {
        const rolling_value = try self.column(name);
        var rolling_columns = try rollingProfileColumnsByValue(self.allocator, rolling_value.*, options_value, self.device, self.rows);
        var rolling_columns_transferred: usize = 0;
        errdefer {
            for (rolling_columns[rolling_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + rolling_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var rolling_names = try rollingProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, rolling_names[0..]);
        for (rolling_names, 0..) |rolling_name, i| source_names[self.columns.len + i] = rolling_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + rolling_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&rolling_columns) |*rolling_col| {
            columns[initialized] = rolling_col.*;
            initialized += 1;
            rolling_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn rollingRangeProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingOptions) DeviceDataError!DeviceDataFrame {
        const rolling_value = try self.column(name);
        var rolling_columns = try rollingRangeProfileColumnsByValue(self.allocator, rolling_value.*, options_value, self.device, self.rows);
        var rolling_columns_transferred: usize = 0;
        errdefer {
            for (rolling_columns[rolling_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + rolling_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var rolling_names = try rollingRangeProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, rolling_names[0..]);
        for (rolling_names, 0..) |rolling_name, i| source_names[self.columns.len + i] = rolling_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + rolling_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&rolling_columns) |*rolling_col| {
            columns[initialized] = rolling_col.*;
            initialized += 1;
            rolling_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn rollingNormalizeProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingOptions) DeviceDataError!DeviceDataFrame {
        const rolling_value = try self.column(name);
        var rolling_columns = try rollingNormalizeProfileColumnsByValue(self.allocator, rolling_value.*, options_value, self.device, self.rows);
        var rolling_columns_transferred: usize = 0;
        errdefer {
            for (rolling_columns[rolling_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + rolling_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var rolling_names = try rollingNormalizeProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, rolling_names[0..]);
        for (rolling_names, 0..) |rolling_name, i| source_names[self.columns.len + i] = rolling_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + rolling_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&rolling_columns) |*rolling_col| {
            columns[initialized] = rolling_col.*;
            initialized += 1;
            rolling_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn rollingQuantileProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingOptions) DeviceDataError!DeviceDataFrame {
        const rolling_value = try self.column(name);
        var rolling_columns = try rollingQuantileProfileColumnsByValue(self.allocator, rolling_value.*, options_value, self.device, self.rows);
        var rolling_columns_transferred: usize = 0;
        errdefer {
            for (rolling_columns[rolling_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + rolling_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var rolling_names = try rollingQuantileProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, rolling_names[0..]);
        for (rolling_names, 0..) |rolling_name, i| source_names[self.columns.len + i] = rolling_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + rolling_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&rolling_columns) |*rolling_col| {
            columns[initialized] = rolling_col.*;
            initialized += 1;
            rolling_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn lagProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceLagOptions) DeviceDataError!DeviceDataFrame {
        const lag_value = try self.column(name);
        var lag_columns = try lagProfileColumnsByValue(self.allocator, lag_value.*, options_value, self.device, self.rows);
        var lag_columns_transferred: usize = 0;
        errdefer {
            for (lag_columns[lag_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + lag_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var lag_names = try lagProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, lag_names[0..]);
        for (lag_names, 0..) |lag_name, i| source_names[self.columns.len + i] = lag_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + lag_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&lag_columns) |*lag_col| {
            columns[initialized] = lag_col.*;
            initialized += 1;
            lag_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn leadProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceLagOptions) DeviceDataError!DeviceDataFrame {
        const lead_value = try self.column(name);
        var lead_columns = try leadProfileColumnsByValue(self.allocator, lead_value.*, options_value, self.device, self.rows);
        var lead_columns_transferred: usize = 0;
        errdefer {
            for (lead_columns[lead_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + lead_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var lead_names = try leadProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, lead_names[0..]);
        for (lead_names, 0..) |lead_name, i| source_names[self.columns.len + i] = lead_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + lead_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&lead_columns) |*lead_col| {
            columns[initialized] = lead_col.*;
            initialized += 1;
            lead_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn clipProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceClipOptions) DeviceDataError!DeviceDataFrame {
        const clip_value = try self.column(name);
        var clip_columns = try clipProfileColumnsByValue(self.allocator, clip_value.*, options_value, self.device, self.rows);
        var clip_columns_transferred: usize = 0;
        errdefer {
            for (clip_columns[clip_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + clip_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var clip_names = try clipProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, clip_names[0..]);
        for (clip_names, 0..) |clip_name, i| source_names[self.columns.len + i] = clip_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + clip_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&clip_columns) |*clip_col| {
            columns[initialized] = clip_col.*;
            initialized += 1;
            clip_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn thresholdProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceThresholdOptions) DeviceDataError!DeviceDataFrame {
        const threshold_value = try self.column(name);
        var threshold_columns = try thresholdProfileColumnsByValue(self.allocator, threshold_value.*, options_value, self.device, self.rows);
        var threshold_columns_transferred: usize = 0;
        errdefer {
            for (threshold_columns[threshold_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + threshold_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var threshold_names = try thresholdProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, threshold_names[0..]);
        for (threshold_names, 0..) |threshold_name, i| source_names[self.columns.len + i] = threshold_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + threshold_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&threshold_columns) |*threshold_col| {
            columns[initialized] = threshold_col.*;
            initialized += 1;
            threshold_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn expandingProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceExpandingOptions) DeviceDataError!DeviceDataFrame {
        const expanding_value = try self.column(name);
        var expanding_columns = try expandingProfileColumnsByValue(self.allocator, expanding_value.*, options_value, self.device, self.rows);
        var expanding_columns_transferred: usize = 0;
        errdefer {
            for (expanding_columns[expanding_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + expanding_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var expanding_names = try expandingProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, expanding_names[0..]);
        for (expanding_names, 0..) |expanding_name, i| source_names[self.columns.len + i] = expanding_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + expanding_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&expanding_columns) |*expanding_col| {
            columns[initialized] = expanding_col.*;
            initialized += 1;
            expanding_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn standardizeProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceStandardizeOptions) DeviceDataError!DeviceDataFrame {
        const standardize_value = try self.column(name);
        var standardize_columns = try standardizeProfileColumnsByValue(self.allocator, standardize_value.*, options_value, self.device, self.rows);
        var standardize_columns_transferred: usize = 0;
        errdefer {
            for (standardize_columns[standardize_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + standardize_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var standardize_names = try standardizeProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, standardize_names[0..]);
        for (standardize_names, 0..) |standardize_name, i| source_names[self.columns.len + i] = standardize_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + standardize_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&standardize_columns) |*standardize_col| {
            columns[initialized] = standardize_col.*;
            initialized += 1;
            standardize_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn robustProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceRobustOptions) DeviceDataError!DeviceDataFrame {
        const robust_value = try self.column(name);
        var robust_columns = try robustProfileColumnsByValue(self.allocator, robust_value.*, options_value, self.device, self.rows);
        var robust_columns_transferred: usize = 0;
        errdefer {
            for (robust_columns[robust_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + robust_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var robust_names = try robustProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, robust_names[0..]);
        for (robust_names, 0..) |robust_name, i| source_names[self.columns.len + i] = robust_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + robust_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&robust_columns) |*robust_col| {
            columns[initialized] = robust_col.*;
            initialized += 1;
            robust_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn drawdownProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceDrawdownOptions) DeviceDataError!DeviceDataFrame {
        const drawdown_value = try self.column(name);
        var drawdown_columns = try drawdownProfileColumnsByValue(self.allocator, drawdown_value.*, options_value, self.device, self.rows);
        var drawdown_columns_transferred: usize = 0;
        errdefer {
            for (drawdown_columns[drawdown_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + drawdown_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var drawdown_names = try drawdownProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, drawdown_names[0..]);
        for (drawdown_names, 0..) |drawdown_name, i| source_names[self.columns.len + i] = drawdown_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + drawdown_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&drawdown_columns) |*drawdown_col| {
            columns[initialized] = drawdown_col.*;
            initialized += 1;
            drawdown_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn extremaProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceExtremaOptions) DeviceDataError!DeviceDataFrame {
        const extrema_value = try self.column(name);
        var extrema_columns = try extremaProfileColumnsByValue(self.allocator, extrema_value.*, options_value, self.device, self.rows);
        var extrema_columns_transferred: usize = 0;
        errdefer {
            for (extrema_columns[extrema_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + extrema_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var extrema_names = try extremaProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, extrema_names[0..]);
        for (extrema_names, 0..) |extrema_name, i| source_names[self.columns.len + i] = extrema_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + extrema_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&extrema_columns) |*extrema_col| {
            columns[initialized] = extrema_col.*;
            initialized += 1;
            extrema_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn trendProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceTrendOptions) DeviceDataError!DeviceDataFrame {
        const trend_value = try self.column(name);
        var trend_columns = try trendProfileColumnsByValue(self.allocator, trend_value.*, options_value, self.device, self.rows);
        var trend_columns_transferred: usize = 0;
        errdefer {
            for (trend_columns[trend_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + trend_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var trend_names = try trendProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, trend_names[0..]);
        for (trend_names, 0..) |trend_name, i| source_names[self.columns.len + i] = trend_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + trend_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&trend_columns) |*trend_col| {
            columns[initialized] = trend_col.*;
            initialized += 1;
            trend_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn signProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceTrendOptions) DeviceDataError!DeviceDataFrame {
        const sign_value = try self.column(name);
        var sign_columns = try signProfileColumnsByValue(self.allocator, sign_value.*, options_value, self.device, self.rows);
        var sign_columns_transferred: usize = 0;
        errdefer {
            for (sign_columns[sign_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + sign_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var sign_names = try signProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, sign_names[0..]);
        for (sign_names, 0..) |sign_name, i| source_names[self.columns.len + i] = sign_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + sign_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&sign_columns) |*sign_col| {
            columns[initialized] = sign_col.*;
            initialized += 1;
            sign_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn crossoverProfile(
        self: DeviceDataFrame,
        lhs_name: []const u8,
        rhs_name: []const u8,
        output_prefix: []const u8,
        options_value: DeviceCrossoverOptions,
    ) DeviceDataError!DeviceDataFrame {
        const lhs = try self.column(lhs_name);
        const rhs = try self.column(rhs_name);
        if (lhs.dtype() != rhs.dtype()) return error.TypeMismatch;
        var cross_columns = try crossoverProfileColumnsByValue(self.allocator, lhs.*, rhs.*, options_value, self.device, self.rows);
        var cross_columns_transferred: usize = 0;
        errdefer {
            for (cross_columns[cross_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + cross_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var cross_names = try crossoverProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, cross_names[0..]);
        for (cross_names, 0..) |cross_name, i| source_names[self.columns.len + i] = cross_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + cross_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&cross_columns) |*cross_col| {
            columns[initialized] = cross_col.*;
            initialized += 1;
            cross_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn bucketProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceBucketOptions) DeviceDataError!DeviceDataFrame {
        const bucket_value = try self.column(name);
        var bucket_columns = try bucketProfileColumnsByValue(self.allocator, bucket_value.*, options_value, self.device, self.rows);
        var bucket_columns_transferred: usize = 0;
        errdefer {
            for (bucket_columns[bucket_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + bucket_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var bucket_names = try bucketProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, bucket_names[0..]);
        for (bucket_names, 0..) |bucket_name, i| source_names[self.columns.len + i] = bucket_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + bucket_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&bucket_columns) |*bucket_col| {
            columns[initialized] = bucket_col.*;
            initialized += 1;
            bucket_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn emaProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceEmaOptions) DeviceDataError!DeviceDataFrame {
        const ema_value = try self.column(name);
        var ema_columns = try emaProfileColumnsByValue(self.allocator, ema_value.*, options_value, self.device, self.rows);
        var ema_columns_transferred: usize = 0;
        errdefer {
            for (ema_columns[ema_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + ema_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var ema_names = try emaProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, ema_names[0..]);
        for (ema_names, 0..) |ema_name, i| source_names[self.columns.len + i] = ema_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + ema_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&ema_columns) |*ema_col| {
            columns[initialized] = ema_col.*;
            initialized += 1;
            ema_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn linearFitProfile(
        self: DeviceDataFrame,
        x_name: []const u8,
        y_name: []const u8,
        output_prefix: []const u8,
        options_value: DeviceLinearFitOptions,
    ) DeviceDataError!DeviceDataFrame {
        const x = try self.column(x_name);
        const y = try self.column(y_name);
        if (x.dtype() != y.dtype()) return error.TypeMismatch;
        var fit_columns = try linearFitProfileColumnsByValue(self.allocator, x.*, y.*, options_value, self.device, self.rows);
        var fit_columns_transferred: usize = 0;
        errdefer {
            for (fit_columns[fit_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + fit_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var fit_names = try linearFitProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, fit_names[0..]);
        for (fit_names, 0..) |fit_name, i| source_names[self.columns.len + i] = fit_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + fit_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&fit_columns) |*fit_col| {
            columns[initialized] = fit_col.*;
            initialized += 1;
            fit_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn errorProfile(
        self: DeviceDataFrame,
        actual_name: []const u8,
        predicted_name: []const u8,
        output_prefix: []const u8,
    ) DeviceDataError!DeviceDataFrame {
        const actual = try self.column(actual_name);
        const predicted = try self.column(predicted_name);
        if (actual.dtype() != predicted.dtype()) return error.TypeMismatch;
        var error_columns = try errorProfileColumnsByValue(self.allocator, actual.*, predicted.*, self.device, self.rows);
        var error_columns_transferred: usize = 0;
        errdefer {
            for (error_columns[error_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + error_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var error_names = try errorProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, error_names[0..]);
        for (error_names, 0..) |error_name, i| source_names[self.columns.len + i] = error_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + error_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&error_columns) |*error_col| {
            columns[initialized] = error_col.*;
            initialized += 1;
            error_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn classificationProfile(
        self: DeviceDataFrame,
        actual_name: []const u8,
        predicted_name: []const u8,
        output_prefix: []const u8,
    ) DeviceDataError!DeviceDataFrame {
        const actual = try self.column(actual_name);
        const predicted = try self.column(predicted_name);
        if (actual.dtype() != .bool or predicted.dtype() != .bool) return error.TypeMismatch;
        var class_columns = try classificationProfileColumns(self.allocator, actual.bool, predicted.bool, self.device, self.rows);
        var class_columns_transferred: usize = 0;
        errdefer {
            for (class_columns[class_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + class_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var class_names = try classificationProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, class_names[0..]);
        for (class_names, 0..) |class_name, i| source_names[self.columns.len + i] = class_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + class_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&class_columns) |*class_col| {
            columns[initialized] = class_col.*;
            initialized += 1;
            class_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn rollingCorrelationProfile(
        self: DeviceDataFrame,
        x_name: []const u8,
        y_name: []const u8,
        output_prefix: []const u8,
        options_value: DeviceRollingCorrelationOptions,
    ) DeviceDataError!DeviceDataFrame {
        const x = try self.column(x_name);
        const y = try self.column(y_name);
        if (x.dtype() != y.dtype()) return error.TypeMismatch;
        var corr_columns = try rollingCorrelationProfileColumnsByValue(self.allocator, x.*, y.*, options_value, self.device, self.rows);
        var corr_columns_transferred: usize = 0;
        errdefer {
            for (corr_columns[corr_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + corr_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var corr_names = try rollingCorrelationProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, corr_names[0..]);
        for (corr_names, 0..) |corr_name, i| source_names[self.columns.len + i] = corr_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + corr_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&corr_columns) |*corr_col| {
            columns[initialized] = corr_col.*;
            initialized += 1;
            corr_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn validityProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8) DeviceDataError!DeviceDataFrame {
        const source = try self.column(name);
        var validity_columns = try validityProfileColumnsByValue(self.allocator, source.*, self.device, self.rows);
        var validity_columns_transferred: usize = 0;
        errdefer {
            for (validity_columns[validity_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + validity_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var validity_names = try validityProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, validity_names[0..]);
        for (validity_names, 0..) |validity_name, i| source_names[self.columns.len + i] = validity_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + validity_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&validity_columns) |*validity_col| {
            columns[initialized] = validity_col.*;
            initialized += 1;
            validity_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn groupByCount(self: DeviceDataFrame, key_name: []const u8, output_name: []const u8) DeviceDataError!DeviceDataFrame {
        const key = try self.column(key_name);
        return switch (key.*) {
            .bool => |typed| groupByCountTyped(bool, self.allocator, key_name, output_name, typed, self.device),
            .i8 => |typed| groupByCountTyped(i8, self.allocator, key_name, output_name, typed, self.device),
            .i16 => |typed| groupByCountTyped(i16, self.allocator, key_name, output_name, typed, self.device),
            .i32 => |typed| groupByCountTyped(i32, self.allocator, key_name, output_name, typed, self.device),
            .i64 => |typed| groupByCountTyped(i64, self.allocator, key_name, output_name, typed, self.device),
            .u8 => |typed| groupByCountTyped(u8, self.allocator, key_name, output_name, typed, self.device),
            .u16 => |typed| groupByCountTyped(u16, self.allocator, key_name, output_name, typed, self.device),
            .u32 => |typed| groupByCountTyped(u32, self.allocator, key_name, output_name, typed, self.device),
            .u64 => |typed| groupByCountTyped(u64, self.allocator, key_name, output_name, typed, self.device),
            .usize => |typed| groupByCountTyped(usize, self.allocator, key_name, output_name, typed, self.device),
            .isize => |typed| groupByCountTyped(isize, self.allocator, key_name, output_name, typed, self.device),
            .f16 => |typed| groupByCountTyped(f16, self.allocator, key_name, output_name, typed, self.device),
            .f32 => |typed| groupByCountTyped(f32, self.allocator, key_name, output_name, typed, self.device),
            .f64 => |typed| groupByCountTyped(f64, self.allocator, key_name, output_name, typed, self.device),
            .bf16, .c64, .c128 => error.TypeUnsupported,
        };
    }

    pub fn groupBySum(self: DeviceDataFrame, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!DeviceDataFrame {
        const key = try self.column(key_name);
        const value = try self.column(value_name);
        return groupByNumericDispatchKey(.sum, self.allocator, key_name, output_name, key.*, value.*, self.device);
    }

    pub fn groupByMin(self: DeviceDataFrame, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!DeviceDataFrame {
        const key = try self.column(key_name);
        const value = try self.column(value_name);
        return groupByNumericDispatchKey(.min, self.allocator, key_name, output_name, key.*, value.*, self.device);
    }

    pub fn groupByMax(self: DeviceDataFrame, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!DeviceDataFrame {
        const key = try self.column(key_name);
        const value = try self.column(value_name);
        return groupByNumericDispatchKey(.max, self.allocator, key_name, output_name, key.*, value.*, self.device);
    }

    pub fn groupByMean(self: DeviceDataFrame, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!DeviceDataFrame {
        const key = try self.column(key_name);
        const value = try self.column(value_name);
        return groupByMeanDispatchKey(self.allocator, key_name, output_name, key.*, value.*, self.device);
    }

    pub fn groupByStats(self: DeviceDataFrame, key_name: []const u8, value_name: []const u8, output_prefix: []const u8) DeviceDataError!DeviceDataFrame {
        const key = try self.column(key_name);
        const value = try self.column(value_name);
        return groupByStatsDispatchKey(self.allocator, key_name, output_prefix, key.*, value.*, self.device);
    }

    pub fn groupByStatsOn(self: DeviceDataFrame, key_names: []const []const u8, value_name: []const u8, output_prefix: []const u8) DeviceDataError!DeviceDataFrame {
        if (key_names.len == 0) return error.LengthMismatch;
        for (key_names) |key_name| _ = try self.column(key_name);
        const value = try self.column(value_name);
        return groupByStatsOnDispatchValue(self.allocator, self, key_names, output_prefix, value.*, self.device);
    }

    pub fn groupByProfile(self: DeviceDataFrame, key_name: []const u8, value_name: []const u8, output_prefix: []const u8) DeviceDataError!DeviceDataFrame {
        const key = try self.column(key_name);
        const value = try self.column(value_name);
        return groupByProfileDispatchKey(self.allocator, key_name, output_prefix, key.*, value.*, self.device);
    }

    pub fn groupByProfileOn(self: DeviceDataFrame, key_names: []const []const u8, value_name: []const u8, output_prefix: []const u8) DeviceDataError!DeviceDataFrame {
        if (key_names.len == 0) return error.LengthMismatch;
        for (key_names) |key_name| _ = try self.column(key_name);
        const value = try self.column(value_name);
        return groupByProfileOnDispatchValue(self.allocator, self, key_names, output_prefix, value.*, self.device);
    }

    pub fn innerJoin(
        self: DeviceDataFrame,
        right: DeviceDataFrame,
        left_key_name: []const u8,
        right_key_name: []const u8,
        options_value: DeviceJoinOptions,
    ) DeviceDataError!DeviceDataFrame {
        if (!self.device.sameDevice(right.device)) return error.InvalidDevice;
        const left_key = try self.column(left_key_name);
        const right_key = try right.column(right_key_name);
        if (left_key.dtype() != right_key.dtype()) return error.TypeMismatch;

        var pair = try innerJoinRowIndices(self.allocator, left_key.*, right_key.*);
        defer pair.deinit();

        var left_rows = try takeOptionalRows(self, pair.left);
        defer left_rows.deinit();
        var right_rows = try takeOptionalRows(right, pair.right);
        defer right_rows.deinit();

        return concatJoinedTables(self.allocator, left_rows, right_rows, right_key_name, options_value);
    }

    pub fn innerJoinOn(
        self: DeviceDataFrame,
        right: DeviceDataFrame,
        left_key_names: []const []const u8,
        right_key_names: []const []const u8,
        options_value: DeviceJoinOptions,
    ) DeviceDataError!DeviceDataFrame {
        if (!self.device.sameDevice(right.device)) return error.InvalidDevice;
        if (left_key_names.len == 0 or left_key_names.len != right_key_names.len) return error.LengthMismatch;
        for (left_key_names, right_key_names) |left_name, right_name| {
            const left_key = try self.column(left_name);
            const right_key = try right.column(right_name);
            if (left_key.dtype() != right_key.dtype()) return error.TypeMismatch;
        }

        var pair = try innerJoinRowIndicesMulti(self.allocator, self, right, left_key_names, right_key_names);
        defer pair.deinit();

        var left_rows = try takeOptionalRows(self, pair.left);
        defer left_rows.deinit();
        var right_rows = try takeOptionalRows(right, pair.right);
        defer right_rows.deinit();

        return concatJoinedTablesExcludingKeys(self.allocator, left_rows, right_rows, right_key_names, options_value);
    }

    pub fn leftJoin(
        self: DeviceDataFrame,
        right: DeviceDataFrame,
        left_key_name: []const u8,
        right_key_name: []const u8,
        options_value: DeviceJoinOptions,
    ) DeviceDataError!DeviceDataFrame {
        if (!self.device.sameDevice(right.device)) return error.InvalidDevice;
        const left_key = try self.column(left_key_name);
        const right_key = try right.column(right_key_name);
        if (left_key.dtype() != right_key.dtype()) return error.TypeMismatch;

        var pair = try leftJoinRowIndices(self.allocator, left_key.*, right_key.*);
        defer pair.deinit();

        var left_rows = try takeOptionalRows(self, pair.left);
        defer left_rows.deinit();
        var right_rows = try takeOptionalRows(right, pair.right);
        defer right_rows.deinit();

        return concatJoinedTables(self.allocator, left_rows, right_rows, right_key_name, options_value);
    }

    pub fn leftJoinOn(
        self: DeviceDataFrame,
        right: DeviceDataFrame,
        left_key_names: []const []const u8,
        right_key_names: []const []const u8,
        options_value: DeviceJoinOptions,
    ) DeviceDataError!DeviceDataFrame {
        if (!self.device.sameDevice(right.device)) return error.InvalidDevice;
        if (left_key_names.len == 0 or left_key_names.len != right_key_names.len) return error.LengthMismatch;
        for (left_key_names, right_key_names) |left_name, right_name| {
            const left_key = try self.column(left_name);
            const right_key = try right.column(right_name);
            if (left_key.dtype() != right_key.dtype()) return error.TypeMismatch;
        }

        var pair = try leftJoinRowIndicesMulti(self.allocator, self, right, left_key_names, right_key_names);
        defer pair.deinit();

        var left_rows = try takeOptionalRows(self, pair.left);
        defer left_rows.deinit();
        var right_rows = try takeOptionalRows(right, pair.right);
        defer right_rows.deinit();

        return concatJoinedTablesExcludingKeys(self.allocator, left_rows, right_rows, right_key_names, options_value);
    }

    pub fn fullJoin(
        self: DeviceDataFrame,
        right: DeviceDataFrame,
        left_key_name: []const u8,
        right_key_name: []const u8,
        options_value: DeviceJoinOptions,
    ) DeviceDataError!DeviceDataFrame {
        if (!self.device.sameDevice(right.device)) return error.InvalidDevice;
        const left_key = try self.column(left_key_name);
        const right_key = try right.column(right_key_name);
        if (left_key.dtype() != right_key.dtype()) return error.TypeMismatch;

        var pair = try fullJoinRowIndices(self.allocator, left_key.*, right_key.*);
        defer pair.deinit();

        var left_rows = try takeOptionalRows(self, pair.left);
        defer left_rows.deinit();
        var right_rows = try takeOptionalRows(right, pair.right);
        defer right_rows.deinit();

        return concatFullJoinedTables(self.allocator, left_rows, right_rows, left_key_name, right_key_name, options_value);
    }

    pub fn fullJoinOn(
        self: DeviceDataFrame,
        right: DeviceDataFrame,
        left_key_names: []const []const u8,
        right_key_names: []const []const u8,
        options_value: DeviceJoinOptions,
    ) DeviceDataError!DeviceDataFrame {
        if (!self.device.sameDevice(right.device)) return error.InvalidDevice;
        if (left_key_names.len == 0 or left_key_names.len != right_key_names.len) return error.LengthMismatch;
        for (left_key_names, right_key_names) |left_name, right_name| {
            const left_key = try self.column(left_name);
            const right_key = try right.column(right_name);
            if (left_key.dtype() != right_key.dtype()) return error.TypeMismatch;
        }

        var pair = try fullJoinRowIndicesMulti(self.allocator, self, right, left_key_names, right_key_names);
        defer pair.deinit();

        var left_rows = try takeOptionalRows(self, pair.left);
        defer left_rows.deinit();
        var right_rows = try takeOptionalRows(right, pair.right);
        defer right_rows.deinit();

        return concatFullJoinedTablesOn(self.allocator, left_rows, right_rows, left_key_names, right_key_names, options_value);
    }

    pub fn semiJoin(
        self: DeviceDataFrame,
        right: DeviceDataFrame,
        left_key_name: []const u8,
        right_key_name: []const u8,
    ) DeviceDataError!DeviceDataFrame {
        const indices = try self.semiAntiJoinIndices(right, left_key_name, right_key_name, true);
        defer self.allocator.free(indices);
        return self.take(indices);
    }

    pub fn semiJoinOn(
        self: DeviceDataFrame,
        right: DeviceDataFrame,
        left_key_names: []const []const u8,
        right_key_names: []const []const u8,
    ) DeviceDataError!DeviceDataFrame {
        const indices = try self.semiAntiJoinIndicesOn(right, left_key_names, right_key_names, true);
        defer self.allocator.free(indices);
        return self.take(indices);
    }

    pub fn antiJoin(
        self: DeviceDataFrame,
        right: DeviceDataFrame,
        left_key_name: []const u8,
        right_key_name: []const u8,
    ) DeviceDataError!DeviceDataFrame {
        const indices = try self.semiAntiJoinIndices(right, left_key_name, right_key_name, false);
        defer self.allocator.free(indices);
        return self.take(indices);
    }

    pub fn antiJoinOn(
        self: DeviceDataFrame,
        right: DeviceDataFrame,
        left_key_names: []const []const u8,
        right_key_names: []const []const u8,
    ) DeviceDataError!DeviceDataFrame {
        const indices = try self.semiAntiJoinIndicesOn(right, left_key_names, right_key_names, false);
        defer self.allocator.free(indices);
        return self.take(indices);
    }

    pub fn asofJoin(
        self: DeviceDataFrame,
        right: DeviceDataFrame,
        left_key_name: []const u8,
        right_key_name: []const u8,
        options_value: DeviceAsofOptions,
    ) DeviceDataError!DeviceDataFrame {
        if (!self.device.sameDevice(right.device)) return error.InvalidDevice;
        const left_key = try self.column(left_key_name);
        const right_key = try right.column(right_key_name);
        if (left_key.dtype() != right_key.dtype()) return error.TypeMismatch;

        const right_indices = try asofRightRowIndices(self.allocator, left_key.*, right_key.*, options_value.strategy);
        defer self.allocator.free(right_indices);
        var right_rows = try takeOptionalRows(right, right_indices);
        defer right_rows.deinit();

        return concatJoinedTables(self.allocator, self, right_rows, right_key_name, .{ .right_suffix = options_value.right_suffix });
    }

    fn semiAntiJoinIndices(
        self: DeviceDataFrame,
        right: DeviceDataFrame,
        left_key_name: []const u8,
        right_key_name: []const u8,
        keep_matches: bool,
    ) DeviceDataError![]usize {
        if (!self.device.sameDevice(right.device)) return error.InvalidDevice;
        const left_key = try self.column(left_key_name);
        const right_key = try right.column(right_key_name);
        if (left_key.dtype() != right_key.dtype()) return error.TypeMismatch;
        return semiAntiJoinRowIndices(self.allocator, left_key.*, right_key.*, keep_matches);
    }

    fn semiAntiJoinIndicesOn(
        self: DeviceDataFrame,
        right: DeviceDataFrame,
        left_key_names: []const []const u8,
        right_key_names: []const []const u8,
        keep_matches: bool,
    ) DeviceDataError![]usize {
        if (!self.device.sameDevice(right.device)) return error.InvalidDevice;
        if (left_key_names.len == 0 or left_key_names.len != right_key_names.len) return error.LengthMismatch;
        for (left_key_names, right_key_names) |left_name, right_name| {
            const left_key = try self.column(left_name);
            const right_key = try right.column(right_name);
            if (left_key.dtype() != right_key.dtype()) return error.TypeMismatch;
        }
        return semiAntiJoinRowIndicesMulti(self.allocator, self, right, left_key_names, right_key_names, keep_matches);
    }

    pub fn filter(self: DeviceDataFrame, mask: []const bool) DeviceDataError!DeviceDataFrame {
        if (mask.len != self.rows) return error.LengthMismatch;
        const row_indices = try rowIndicesFromMask(self.allocator, mask);
        defer self.allocator.free(row_indices);
        return self.take(row_indices);
    }

    pub fn to(self: DeviceDataFrame, device_value: array_mod.Device) DeviceDataError!DeviceDataFrame {
        if (!device_value.isAvailable()) return error.InvalidDevice;
        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.to(device_value);
            initialized += 1;
        }
        return initDeviceDataFrameFromOwnedColumns(self.allocator, self.names, columns, self.rows, device_value);
    }

    pub fn cpu(self: DeviceDataFrame) DeviceDataError!DeviceDataFrame {
        return self.to(.cpu);
    }

    pub fn cuda(self: DeviceDataFrame, index: usize) DeviceDataError!DeviceDataFrame {
        return self.to(array_mod.Device.cuda(index));
    }

    pub fn mps(self: DeviceDataFrame, index: usize) DeviceDataError!DeviceDataFrame {
        return self.to(array_mod.Device.mps(index));
    }

    pub fn toDataFrame(self: DeviceDataFrame) DeviceDataError!DataFrame {
        var defs = try self.allocator.alloc(ColumnDef, self.columns.len);
        defer self.allocator.free(defs);
        var initialized: usize = 0;
        defer {
            for (defs[0..initialized]) |def| freeColumn(self.allocator, def.data);
        }

        for (self.names, self.columns, 0..) |name, col, i| {
            if (col.hasNulls()) return error.TypeUnsupported;
            defs[i].name = name;
            defs[i].data = switch (col) {
                .f64 => |typed| .{ .f64 = try typed.toOwnedSlice(self.allocator) },
                .i64 => |typed| .{ .i64 = try typed.toOwnedSlice(self.allocator) },
                .bool => |typed| .{ .bool = try typed.toOwnedSlice(self.allocator) },
                else => return error.TypeUnsupported,
            };
            initialized += 1;
        }
        return DataFrame.init(self.allocator, defs);
    }
};

fn countNulls(validity_values: []const bool) usize {
    var nulls: usize = 0;
    for (validity_values) |valid| {
        if (!valid) nulls += 1;
    }
    return nulls;
}

fn countNullsInArray(mask: array_mod.Array(bool)) array_mod.ArrayError!usize {
    const values = try mask.toOwnedSlice(mask.allocator);
    defer mask.allocator.free(values);
    return countNulls(values);
}

fn argsortTypedColumn(comptime T: type, column: DeviceTypedColumn(T), allocator: std.mem.Allocator, options_value: DeviceSortOptions) DeviceDataError![]usize {
    const values = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const order = try allocator.alloc(usize, values.len);
    for (order, 0..) |*slot, i| slot.* = i;

    const Ctx = struct {
        values: []const T,
        validity: ?[]const bool,
        options: DeviceSortOptions,

        fn isValid(ctx: @This(), index: usize) bool {
            return if (ctx.validity) |validity| validity[index] else true;
        }

        fn lessThan(ctx: @This(), a: usize, b: usize) bool {
            const a_valid = ctx.isValid(a);
            const b_valid = ctx.isValid(b);
            if (a_valid != b_valid) {
                return switch (ctx.options.nulls) {
                    .first => !a_valid,
                    .last => a_valid,
                };
            }
            if (!a_valid and !b_valid) return a < b;

            const cmp = compareSortValues(T, ctx.values[a], ctx.values[b]);
            if (cmp == 0) return a < b;
            return if (ctx.options.descending) cmp > 0 else cmp < 0;
        }
    };

    std.sort.insertion(usize, order, Ctx{
        .values = values,
        .validity = maybe_validity,
        .options = options_value,
    }, Ctx.lessThan);
    return order;
}

fn compareSortValues(comptime T: type, lhs: T, rhs: T) i8 {
    if (comptime T == bool) {
        if (lhs == rhs) return 0;
        return if (!lhs and rhs) -1 else 1;
    }
    return switch (@typeInfo(T)) {
        .int, .comptime_int => if (lhs < rhs) -1 else if (rhs < lhs) 1 else 0,
        .float, .comptime_float => compareFloatSortValues(T, lhs, rhs),
        else => @compileError("sort requires bool or ordered numeric column values"),
    };
}

fn compareFloatSortValues(comptime T: type, lhs: T, rhs: T) i8 {
    const lhs_nan = std.math.isNan(lhs);
    const rhs_nan = std.math.isNan(rhs);
    if (lhs_nan != rhs_nan) return if (lhs_nan) 1 else -1;
    if (lhs_nan and rhs_nan) return 0;
    if (lhs < rhs) return -1;
    if (rhs < lhs) return 1;
    return 0;
}

const RankProfileColumnCount = 5;

fn rankProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![RankProfileColumnCount][]const u8 {
    var names: [RankProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{
        "ordinal_rank",
        "competition_rank",
        "dense_rank",
        "percent_rank",
        "cume_dist",
    };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn rankProfileColumnsByKey(
    allocator: std.mem.Allocator,
    key: DeviceColumn,
    options_value: DeviceSortOptions,
    device_value: array_mod.Device,
    rows: usize,
) DeviceDataError![RankProfileColumnCount]DeviceColumn {
    if (key.len() != rows) return error.LengthMismatch;
    return switch (key) {
        .bool => |typed| rankProfileColumnsTyped(bool, allocator, typed, options_value, device_value),
        .i8 => |typed| rankProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| rankProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| rankProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| rankProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| rankProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| rankProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| rankProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| rankProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| rankProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| rankProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| rankProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| rankProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| rankProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn rankProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceSortOptions,
    device_value: array_mod.Device,
) DeviceDataError![RankProfileColumnCount]DeviceColumn {
    const rows = column.len();
    const values = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);
    const order = try argsortTypedColumn(T, column, allocator, options_value);
    defer allocator.free(order);

    const ordinal = try allocator.alloc(i64, rows);
    defer allocator.free(ordinal);
    const competition = try allocator.alloc(i64, rows);
    defer allocator.free(competition);
    const dense = try allocator.alloc(i64, rows);
    defer allocator.free(dense);
    const percent = try allocator.alloc(f64, rows);
    defer allocator.free(percent);
    const cume = try allocator.alloc(f64, rows);
    defer allocator.free(cume);

    var group_start: usize = 0;
    var dense_rank: i64 = 0;
    while (group_start < rows) {
        var group_end = group_start + 1;
        while (group_end < rows and rankKeysTie(T, values, maybe_validity, order[group_start], order[group_end])) {
            group_end += 1;
        }

        dense_rank += 1;
        const competition_rank: i64 = @intCast(group_start + 1);
        const percent_rank: f64 = if (rows <= 1) 0 else @as(f64, @floatFromInt(group_start)) / @as(f64, @floatFromInt(rows - 1));
        const cume_dist: f64 = if (rows == 0) std.math.nan(f64) else @as(f64, @floatFromInt(group_end)) / @as(f64, @floatFromInt(rows));

        for (order[group_start..group_end], group_start..) |row, sorted_position| {
            ordinal[row] = @intCast(sorted_position + 1);
            competition[row] = competition_rank;
            dense[row] = dense_rank;
            percent[row] = percent_rank;
            cume[row] = cume_dist;
        }
        group_start = group_end;
    }

    var columns: [RankProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, ordinal, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSlice(i64, allocator, competition, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSlice(i64, allocator, dense, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSlice(f64, allocator, percent, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSlice(f64, allocator, cume, device_value);
    initialized += 1;
    return columns;
}

fn rankKeysTie(comptime T: type, values: []const T, maybe_validity: ?[]const bool, lhs: usize, rhs: usize) bool {
    const lhs_valid = if (maybe_validity) |validity| validity[lhs] else true;
    const rhs_valid = if (maybe_validity) |validity| validity[rhs] else true;
    if (lhs_valid != rhs_valid) return false;
    if (!lhs_valid) return true;
    return compareSortValues(T, values[lhs], values[rhs]) == 0;
}

const RollingProfileColumnCount = 5;

fn rollingProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![RollingProfileColumnCount][]const u8 {
    var names: [RollingProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "rolling_count", "rolling_sum", "rolling_mean", "rolling_variance", "rolling_stddev" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn rollingProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
    rows: usize,
) DeviceDataError![RollingProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| rollingProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| rollingProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| rollingProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| rollingProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| rollingProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| rollingProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| rollingProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| rollingProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| rollingProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| rollingProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| rollingProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| rollingProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| rollingProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn rollingProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
) DeviceDataError![RollingProfileColumnCount]DeviceColumn {
    if (options_value.window == 0) return error.InvalidShape;
    const min_periods = options_value.min_periods orelse options_value.window;
    if (min_periods == 0 or min_periods > options_value.window) return error.InvalidShape;

    const values = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values.len;
    const counts = try allocator.alloc(i64, rows);
    defer allocator.free(counts);
    const sums = try allocator.alloc(f64, rows);
    defer allocator.free(sums);
    const means = try allocator.alloc(f64, rows);
    defer allocator.free(means);
    const variances = try allocator.alloc(f64, rows);
    defer allocator.free(variances);
    const stddevs = try allocator.alloc(f64, rows);
    defer allocator.free(stddevs);
    const validity = try allocator.alloc(bool, rows);
    defer allocator.free(validity);

    var running_sum: f64 = 0;
    var running_sum_sq: f64 = 0;
    var running_count: usize = 0;
    for (values, 0..) |value_item, row| {
        const valid = if (maybe_validity) |mask| mask[row] else true;
        if (valid) {
            const x = castToF64(T, value_item);
            running_sum += x;
            running_sum_sq += x * x;
            running_count += 1;
        }
        if (row >= options_value.window) {
            const evict_row = row - options_value.window;
            const evict_valid = if (maybe_validity) |mask| mask[evict_row] else true;
            if (evict_valid) {
                const x = castToF64(T, values[evict_row]);
                running_sum -= x;
                running_sum_sq -= x * x;
                running_count -= 1;
            }
        }

        counts[row] = @intCast(running_count);
        const has_enough = running_count >= min_periods;
        validity[row] = has_enough;
        if (has_enough) {
            const n: f64 = @floatFromInt(running_count);
            const mean = running_sum / n;
            const raw_variance = running_sum_sq / n - mean * mean;
            const variance = if (raw_variance < 0) 0 else raw_variance;
            sums[row] = running_sum;
            means[row] = mean;
            variances[row] = variance;
            stddevs[row] = std.math.sqrt(variance);
        } else {
            sums[row] = 0;
            means[row] = 0;
            variances[row] = 0;
            stddevs[row] = 0;
        }
    }

    var columns: [RollingProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, sums, validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, means, validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, variances, validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, stddevs, validity, device_value);
    initialized += 1;
    return columns;
}

const RollingRangeProfileColumnCount = 4;

fn rollingRangeProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![RollingRangeProfileColumnCount][]const u8 {
    var names: [RollingRangeProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "rolling_low", "rolling_high", "rolling_range", "rolling_position" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn rollingRangeProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
    rows: usize,
) DeviceDataError![RollingRangeProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| rollingRangeProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| rollingRangeProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| rollingRangeProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| rollingRangeProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| rollingRangeProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| rollingRangeProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| rollingRangeProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| rollingRangeProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| rollingRangeProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| rollingRangeProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| rollingRangeProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| rollingRangeProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| rollingRangeProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn rollingRangeProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
) DeviceDataError![RollingRangeProfileColumnCount]DeviceColumn {
    if (options_value.window == 0) return error.InvalidShape;
    const min_periods = options_value.min_periods orelse options_value.window;
    if (min_periods == 0 or min_periods > options_value.window) return error.InvalidShape;

    const values = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values.len;
    const lows = try allocator.alloc(f64, rows);
    defer allocator.free(lows);
    const highs = try allocator.alloc(f64, rows);
    defer allocator.free(highs);
    const ranges = try allocator.alloc(f64, rows);
    defer allocator.free(ranges);
    const positions = try allocator.alloc(f64, rows);
    defer allocator.free(positions);
    const validity = try allocator.alloc(bool, rows);
    defer allocator.free(validity);

    // This intentionally recomputes each small trailing window in host memory,
    // matching the other dataframe profile APIs' current gather/materialize
    // strategy while preserving a single future lowering seam for device rolling
    // min/max kernels.
    for (values, 0..) |value_item, row| {
        const start = if (row + 1 > options_value.window) row + 1 - options_value.window else 0;
        var count: usize = 0;
        var low: f64 = 0;
        var high: f64 = 0;
        for (start..row + 1) |window_row| {
            const row_valid = if (maybe_validity) |mask| mask[window_row] else true;
            if (!row_valid) continue;
            const x = castToF64(T, values[window_row]);
            if (count == 0) {
                low = x;
                high = x;
            } else {
                if (x < low) low = x;
                if (x > high) high = x;
            }
            count += 1;
        }

        const current_valid = if (maybe_validity) |mask| mask[row] else true;
        const has_enough = current_valid and count >= min_periods;
        validity[row] = has_enough;
        if (has_enough) {
            const current = castToF64(T, value_item);
            const range = high - low;
            lows[row] = low;
            highs[row] = high;
            ranges[row] = range;
            positions[row] = if (range == 0) std.math.nan(f64) else (current - low) / range;
        } else {
            lows[row] = 0;
            highs[row] = 0;
            ranges[row] = 0;
            positions[row] = 0;
        }
    }

    var columns: [RollingRangeProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, lows, validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, highs, validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, ranges, validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, positions, validity, device_value);
    initialized += 1;
    return columns;
}

const RollingNormalizeProfileColumnCount = 3;

fn rollingNormalizeProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![RollingNormalizeProfileColumnCount][]const u8 {
    var names: [RollingNormalizeProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "rolling_centered", "rolling_zscore", "rolling_minmax" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn rollingNormalizeProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
    rows: usize,
) DeviceDataError![RollingNormalizeProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| rollingNormalizeProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| rollingNormalizeProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| rollingNormalizeProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| rollingNormalizeProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| rollingNormalizeProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| rollingNormalizeProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| rollingNormalizeProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| rollingNormalizeProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| rollingNormalizeProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| rollingNormalizeProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| rollingNormalizeProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| rollingNormalizeProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| rollingNormalizeProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn rollingNormalizeProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
) DeviceDataError![RollingNormalizeProfileColumnCount]DeviceColumn {
    if (options_value.window == 0) return error.InvalidShape;
    const min_periods = options_value.min_periods orelse options_value.window;
    if (min_periods == 0 or min_periods > options_value.window) return error.InvalidShape;

    const values = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values.len;
    const centered = try allocator.alloc(f64, rows);
    defer allocator.free(centered);
    const zscores = try allocator.alloc(f64, rows);
    defer allocator.free(zscores);
    const minmax = try allocator.alloc(f64, rows);
    defer allocator.free(minmax);
    const validity = try allocator.alloc(bool, rows);
    defer allocator.free(validity);

    for (values, 0..) |value_item, row| {
        const start = if (row + 1 > options_value.window) row + 1 - options_value.window else 0;
        var count: usize = 0;
        var sum: f64 = 0;
        var sum_sq: f64 = 0;
        var low: f64 = 0;
        var high: f64 = 0;
        for (start..row + 1) |window_row| {
            const row_valid = if (maybe_validity) |mask| mask[window_row] else true;
            if (!row_valid) continue;
            const x = castToF64(T, values[window_row]);
            if (count == 0) {
                low = x;
                high = x;
            } else {
                if (x < low) low = x;
                if (x > high) high = x;
            }
            sum += x;
            sum_sq += x * x;
            count += 1;
        }

        const current_valid = if (maybe_validity) |mask| mask[row] else true;
        const has_enough = current_valid and count >= min_periods;
        validity[row] = has_enough;
        if (has_enough) {
            const x = castToF64(T, value_item);
            const n: f64 = @floatFromInt(count);
            const mean = sum / n;
            const raw_variance = sum_sq / n - mean * mean;
            const variance = if (raw_variance < 0) 0 else raw_variance;
            const stddev = std.math.sqrt(variance);
            const range = high - low;
            centered[row] = x - mean;
            zscores[row] = if (stddev == 0) std.math.nan(f64) else (x - mean) / stddev;
            minmax[row] = if (range == 0) std.math.nan(f64) else (x - low) / range;
        } else {
            centered[row] = 0;
            zscores[row] = 0;
            minmax[row] = 0;
        }
    }

    var columns: [RollingNormalizeProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, centered, validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, zscores, validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, minmax, validity, device_value);
    initialized += 1;
    return columns;
}

const RollingQuantileProfileColumnCount = 4;

fn rollingQuantileProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![RollingQuantileProfileColumnCount][]const u8 {
    var names: [RollingQuantileProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "rolling_q1", "rolling_median", "rolling_q3", "rolling_iqr" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn rollingQuantileProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
    rows: usize,
) DeviceDataError![RollingQuantileProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| rollingQuantileProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| rollingQuantileProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| rollingQuantileProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| rollingQuantileProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| rollingQuantileProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| rollingQuantileProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| rollingQuantileProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| rollingQuantileProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| rollingQuantileProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| rollingQuantileProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| rollingQuantileProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| rollingQuantileProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| rollingQuantileProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn rollingQuantileProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
) DeviceDataError![RollingQuantileProfileColumnCount]DeviceColumn {
    if (options_value.window == 0) return error.InvalidShape;
    const min_periods = options_value.min_periods orelse options_value.window;
    if (min_periods == 0 or min_periods > options_value.window) return error.InvalidShape;

    const values = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values.len;
    const q1_values = try allocator.alloc(f64, rows);
    defer allocator.free(q1_values);
    const medians = try allocator.alloc(f64, rows);
    defer allocator.free(medians);
    const q3_values = try allocator.alloc(f64, rows);
    defer allocator.free(q3_values);
    const iqrs = try allocator.alloc(f64, rows);
    defer allocator.free(iqrs);
    const validity = try allocator.alloc(bool, rows);
    defer allocator.free(validity);
    const scratch = try allocator.alloc(f64, options_value.window);
    defer allocator.free(scratch);

    for (0..rows) |row| {
        const start = if (row + 1 > options_value.window) row + 1 - options_value.window else 0;
        var count: usize = 0;
        for (start..row + 1) |window_row| {
            const row_valid = if (maybe_validity) |mask| mask[window_row] else true;
            if (!row_valid) continue;
            scratch[count] = castToF64(T, values[window_row]);
            count += 1;
        }

        const current_valid = if (maybe_validity) |mask| mask[row] else true;
        const has_enough = current_valid and count >= min_periods;
        validity[row] = has_enough;
        if (has_enough) {
            const window_values = scratch[0..count];
            std.sort.insertion(f64, window_values, {}, struct {
                fn lessThan(_: void, lhs: f64, rhs: f64) bool {
                    return compareFloatSortValues(f64, lhs, rhs) < 0;
                }
            }.lessThan);
            const q1 = quantileSorted(window_values, 0.25);
            const median = quantileSorted(window_values, 0.5);
            const q3 = quantileSorted(window_values, 0.75);
            q1_values[row] = q1;
            medians[row] = median;
            q3_values[row] = q3;
            iqrs[row] = q3 - q1;
        } else {
            q1_values[row] = 0;
            medians[row] = 0;
            q3_values[row] = 0;
            iqrs[row] = 0;
        }
    }

    var columns: [RollingQuantileProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, q1_values, validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, medians, validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, q3_values, validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, iqrs, validity, device_value);
    initialized += 1;
    return columns;
}

const LagProfileColumnCount = 3;

fn lagProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![LagProfileColumnCount][]const u8 {
    var names: [LagProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "lag", "diff", "pct_change" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn lagProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceLagOptions,
    device_value: array_mod.Device,
    rows: usize,
) DeviceDataError![LagProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| lagProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| lagProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| lagProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| lagProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| lagProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| lagProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| lagProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| lagProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| lagProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| lagProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| lagProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| lagProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| lagProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn lagProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceLagOptions,
    device_value: array_mod.Device,
) DeviceDataError![LagProfileColumnCount]DeviceColumn {
    if (options_value.periods == 0) return error.InvalidShape;

    const values = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values.len;
    const lag_values = try allocator.alloc(f64, rows);
    defer allocator.free(lag_values);
    const diff_values = try allocator.alloc(f64, rows);
    defer allocator.free(diff_values);
    const pct_values = try allocator.alloc(f64, rows);
    defer allocator.free(pct_values);
    const lag_validity = try allocator.alloc(bool, rows);
    defer allocator.free(lag_validity);
    const change_validity = try allocator.alloc(bool, rows);
    defer allocator.free(change_validity);

    // This deliberately emits lag, absolute difference, and percent change in
    // one pass because feature-engineering pipelines commonly request all three
    // together.  The shape mirrors dataframe engines such as Polars/Pandas while
    // keeping a single future lowering seam for device-side shift kernels.
    for (values, 0..) |value_item, row| {
        const row_valid = if (maybe_validity) |mask| mask[row] else true;
        if (row < options_value.periods) {
            lag_values[row] = 0;
            diff_values[row] = 0;
            pct_values[row] = 0;
            lag_validity[row] = false;
            change_validity[row] = false;
            continue;
        }

        const lag_row = row - options_value.periods;
        const lag_row_valid = if (maybe_validity) |mask| mask[lag_row] else true;
        const previous = castToF64(T, values[lag_row]);
        const current = castToF64(T, value_item);
        lag_values[row] = previous;
        lag_validity[row] = lag_row_valid;

        const can_change = row_valid and lag_row_valid;
        change_validity[row] = can_change;
        if (can_change) {
            const diff = current - previous;
            diff_values[row] = diff;
            pct_values[row] = if (previous == 0) std.math.nan(f64) else diff / previous;
        } else {
            diff_values[row] = 0;
            pct_values[row] = 0;
        }
    }

    var columns: [LagProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, lag_values, lag_validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, diff_values, change_validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, pct_values, change_validity, device_value);
    initialized += 1;
    return columns;
}

const LeadProfileColumnCount = 3;

fn leadProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![LeadProfileColumnCount][]const u8 {
    var names: [LeadProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "lead", "forward_diff", "forward_pct_change" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn leadProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceLagOptions,
    device_value: array_mod.Device,
    rows: usize,
) DeviceDataError![LeadProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| leadProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| leadProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| leadProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| leadProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| leadProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| leadProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| leadProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| leadProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| leadProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| leadProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| leadProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| leadProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| leadProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn leadProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceLagOptions,
    device_value: array_mod.Device,
) DeviceDataError![LeadProfileColumnCount]DeviceColumn {
    if (options_value.periods == 0) return error.InvalidShape;

    const values = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values.len;
    const lead_values = try allocator.alloc(f64, rows);
    defer allocator.free(lead_values);
    const diff_values = try allocator.alloc(f64, rows);
    defer allocator.free(diff_values);
    const pct_values = try allocator.alloc(f64, rows);
    defer allocator.free(pct_values);
    const lead_validity = try allocator.alloc(bool, rows);
    defer allocator.free(lead_validity);
    const change_validity = try allocator.alloc(bool, rows);
    defer allocator.free(change_validity);

    for (values, 0..) |value_item, row| {
        const lead_row = row + options_value.periods;
        const row_valid = if (maybe_validity) |mask| mask[row] else true;
        if (lead_row >= rows) {
            lead_values[row] = 0;
            diff_values[row] = 0;
            pct_values[row] = 0;
            lead_validity[row] = false;
            change_validity[row] = false;
            continue;
        }

        const lead_row_valid = if (maybe_validity) |mask| mask[lead_row] else true;
        const current = castToF64(T, value_item);
        const future = castToF64(T, values[lead_row]);
        lead_values[row] = future;
        lead_validity[row] = lead_row_valid;

        const can_change = row_valid and lead_row_valid;
        change_validity[row] = can_change;
        if (can_change) {
            const diff = future - current;
            diff_values[row] = diff;
            pct_values[row] = if (current == 0) std.math.nan(f64) else diff / current;
        } else {
            diff_values[row] = 0;
            pct_values[row] = 0;
        }
    }

    var columns: [LeadProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, lead_values, lead_validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, diff_values, change_validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, pct_values, change_validity, device_value);
    initialized += 1;
    return columns;
}

const ClipProfileColumnCount = 4;

fn clipProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![ClipProfileColumnCount][]const u8 {
    var names: [ClipProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "clipped", "below", "above", "in_range" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn clipProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceClipOptions,
    device_value: array_mod.Device,
    rows: usize,
) DeviceDataError![ClipProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| clipProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| clipProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| clipProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| clipProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| clipProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| clipProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| clipProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| clipProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| clipProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| clipProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| clipProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| clipProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| clipProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn clipProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceClipOptions,
    device_value: array_mod.Device,
) DeviceDataError![ClipProfileColumnCount]DeviceColumn {
    if (options_value.lower > options_value.upper) return error.InvalidShape;

    const values = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values.len;
    const clipped = try allocator.alloc(f64, rows);
    defer allocator.free(clipped);
    const below = try allocator.alloc(bool, rows);
    defer allocator.free(below);
    const above = try allocator.alloc(bool, rows);
    defer allocator.free(above);
    const in_range = try allocator.alloc(bool, rows);
    defer allocator.free(in_range);
    const validity = try allocator.alloc(bool, rows);
    defer allocator.free(validity);

    for (values, 0..) |value_item, row| {
        const valid = if (maybe_validity) |mask| mask[row] else true;
        validity[row] = valid;
        if (valid) {
            const x = castToF64(T, value_item);
            below[row] = x < options_value.lower;
            above[row] = x > options_value.upper;
            in_range[row] = !below[row] and !above[row];
            clipped[row] = @min(@max(x, options_value.lower), options_value.upper);
        } else {
            below[row] = false;
            above[row] = false;
            in_range[row] = false;
            clipped[row] = 0;
        }
    }

    var columns: [ClipProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, clipped, validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(bool, allocator, below, validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(bool, allocator, above, validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(bool, allocator, in_range, validity, device_value);
    initialized += 1;
    return columns;
}

const ThresholdProfileColumnCount = 5;

fn thresholdProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![ThresholdProfileColumnCount][]const u8 {
    var names: [ThresholdProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "distance", "abs_distance", "above", "below", "at" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn thresholdProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceThresholdOptions,
    device_value: array_mod.Device,
    rows: usize,
) DeviceDataError![ThresholdProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| thresholdProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| thresholdProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| thresholdProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| thresholdProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| thresholdProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| thresholdProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| thresholdProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| thresholdProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| thresholdProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| thresholdProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| thresholdProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| thresholdProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| thresholdProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn thresholdProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceThresholdOptions,
    device_value: array_mod.Device,
) DeviceDataError![ThresholdProfileColumnCount]DeviceColumn {
    const values = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values.len;
    const distances = try allocator.alloc(f64, rows);
    defer allocator.free(distances);
    const abs_distances = try allocator.alloc(f64, rows);
    defer allocator.free(abs_distances);
    const above = try allocator.alloc(bool, rows);
    defer allocator.free(above);
    const below = try allocator.alloc(bool, rows);
    defer allocator.free(below);
    const at = try allocator.alloc(bool, rows);
    defer allocator.free(at);
    const validity = try allocator.alloc(bool, rows);
    defer allocator.free(validity);

    for (values, 0..) |value_item, row| {
        const valid = if (maybe_validity) |mask| mask[row] else true;
        validity[row] = valid;
        if (valid) {
            const distance = castToF64(T, value_item) - options_value.threshold;
            distances[row] = distance;
            abs_distances[row] = @abs(distance);
            above[row] = distance > 0;
            below[row] = distance < 0;
            at[row] = distance == 0;
        } else {
            distances[row] = 0;
            abs_distances[row] = 0;
            above[row] = false;
            below[row] = false;
            at[row] = false;
        }
    }

    var columns: [ThresholdProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, distances, validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, abs_distances, validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(bool, allocator, above, validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(bool, allocator, below, validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(bool, allocator, at, validity, device_value);
    initialized += 1;
    return columns;
}

const ExpandingProfileColumnCount = 5;

fn expandingProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![ExpandingProfileColumnCount][]const u8 {
    var names: [ExpandingProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "expanding_count", "expanding_sum", "expanding_mean", "expanding_min", "expanding_max" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn expandingProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
    rows: usize,
) DeviceDataError![ExpandingProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| expandingProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| expandingProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| expandingProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| expandingProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| expandingProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| expandingProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| expandingProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| expandingProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| expandingProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| expandingProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| expandingProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| expandingProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| expandingProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn expandingProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
) DeviceDataError![ExpandingProfileColumnCount]DeviceColumn {
    if (options_value.min_periods == 0) return error.InvalidShape;

    const values = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values.len;
    const counts = try allocator.alloc(i64, rows);
    defer allocator.free(counts);
    const sums = try allocator.alloc(f64, rows);
    defer allocator.free(sums);
    const means = try allocator.alloc(f64, rows);
    defer allocator.free(means);
    const mins = try allocator.alloc(f64, rows);
    defer allocator.free(mins);
    const maxes = try allocator.alloc(f64, rows);
    defer allocator.free(maxes);
    const metric_validity = try allocator.alloc(bool, rows);
    defer allocator.free(metric_validity);

    var running_count: usize = 0;
    var running_sum: f64 = 0;
    var running_min: f64 = 0;
    var running_max: f64 = 0;
    for (values, 0..) |value_item, row| {
        const valid = if (maybe_validity) |mask| mask[row] else true;
        if (valid) {
            const x = castToF64(T, value_item);
            if (running_count == 0) {
                running_min = x;
                running_max = x;
            } else {
                if (x < running_min) running_min = x;
                if (x > running_max) running_max = x;
            }
            running_sum += x;
            running_count += 1;
        }

        counts[row] = @intCast(running_count);
        const has_enough = running_count >= options_value.min_periods;
        metric_validity[row] = has_enough;
        if (has_enough) {
            sums[row] = running_sum;
            means[row] = running_sum / @as(f64, @floatFromInt(running_count));
            mins[row] = running_min;
            maxes[row] = running_max;
        } else {
            sums[row] = 0;
            means[row] = 0;
            mins[row] = 0;
            maxes[row] = 0;
        }
    }

    var columns: [ExpandingProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, sums, metric_validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, means, metric_validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, mins, metric_validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, maxes, metric_validity, device_value);
    initialized += 1;
    return columns;
}

const StandardizeProfileColumnCount = 3;

fn standardizeProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![StandardizeProfileColumnCount][]const u8 {
    var names: [StandardizeProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "centered", "zscore", "minmax" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn standardizeProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceStandardizeOptions,
    device_value: array_mod.Device,
    rows: usize,
) DeviceDataError![StandardizeProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| standardizeProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| standardizeProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| standardizeProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| standardizeProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| standardizeProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| standardizeProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| standardizeProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| standardizeProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| standardizeProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| standardizeProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| standardizeProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| standardizeProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| standardizeProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn standardizeProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceStandardizeOptions,
    device_value: array_mod.Device,
) DeviceDataError![StandardizeProfileColumnCount]DeviceColumn {
    if (options_value.min_periods == 0) return error.InvalidShape;

    const values = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    var count: usize = 0;
    var sum: f64 = 0;
    var sum_sq: f64 = 0;
    var min_value: f64 = 0;
    var max_value: f64 = 0;
    for (values, 0..) |value_item, row| {
        const valid = if (maybe_validity) |mask| mask[row] else true;
        if (!valid) continue;
        const x = castToF64(T, value_item);
        if (count == 0) {
            min_value = x;
            max_value = x;
        } else {
            if (x < min_value) min_value = x;
            if (x > max_value) max_value = x;
        }
        sum += x;
        sum_sq += x * x;
        count += 1;
    }

    const rows = values.len;
    const centered = try allocator.alloc(f64, rows);
    defer allocator.free(centered);
    const zscores = try allocator.alloc(f64, rows);
    defer allocator.free(zscores);
    const minmax = try allocator.alloc(f64, rows);
    defer allocator.free(minmax);
    const validity = try allocator.alloc(bool, rows);
    defer allocator.free(validity);

    const has_enough = count >= options_value.min_periods;
    const mean = if (count == 0) 0 else sum / @as(f64, @floatFromInt(count));
    const raw_variance = if (count == 0) 0 else sum_sq / @as(f64, @floatFromInt(count)) - mean * mean;
    const variance = if (raw_variance < 0) 0 else raw_variance;
    const stddev = std.math.sqrt(variance);
    const range = max_value - min_value;

    // Generate common whole-column scaling features in a single pass over the
    // materialized values. This mirrors dataframe feature-engineering pipelines
    // that ask for centered, z-score, and min-max forms together while leaving a
    // single future lowering seam for device-side normalization kernels.
    for (values, 0..) |value_item, row| {
        const row_valid = if (maybe_validity) |mask| mask[row] else true;
        const valid = row_valid and has_enough;
        validity[row] = valid;
        if (valid) {
            const x = castToF64(T, value_item);
            const delta = x - mean;
            centered[row] = delta;
            zscores[row] = if (stddev == 0) std.math.nan(f64) else delta / stddev;
            minmax[row] = if (range == 0) std.math.nan(f64) else (x - min_value) / range;
        } else {
            centered[row] = 0;
            zscores[row] = 0;
            minmax[row] = 0;
        }
    }

    var columns: [StandardizeProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, centered, validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, zscores, validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, minmax, validity, device_value);
    initialized += 1;
    return columns;
}

const RobustProfileColumnCount = 4;

fn robustProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![RobustProfileColumnCount][]const u8 {
    var names: [RobustProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "median_centered", "mad_zscore", "iqr_outlier", "winsorized" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn robustProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceRobustOptions,
    device_value: array_mod.Device,
    rows: usize,
) DeviceDataError![RobustProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| robustProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| robustProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| robustProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| robustProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| robustProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| robustProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| robustProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| robustProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| robustProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| robustProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| robustProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| robustProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| robustProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn robustProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceRobustOptions,
    device_value: array_mod.Device,
) DeviceDataError![RobustProfileColumnCount]DeviceColumn {
    if (options_value.min_periods == 0) return error.InvalidShape;

    const values = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    var valid_count: usize = 0;
    for (values, 0..) |_, row| {
        const valid = if (maybe_validity) |mask| mask[row] else true;
        if (valid) valid_count += 1;
    }

    const valid_values = try allocator.alloc(f64, valid_count);
    defer allocator.free(valid_values);
    var write: usize = 0;
    for (values, 0..) |value_item, row| {
        const valid = if (maybe_validity) |mask| mask[row] else true;
        if (!valid) continue;
        valid_values[write] = castToF64(T, value_item);
        write += 1;
    }
    std.sort.insertion(f64, valid_values, {}, struct {
        fn lessThan(_: void, lhs: f64, rhs: f64) bool {
            return compareFloatSortValues(f64, lhs, rhs) < 0;
        }
    }.lessThan);

    const has_enough = valid_count >= options_value.min_periods;
    const median = if (valid_count == 0) 0 else quantileSorted(valid_values, 0.5);
    const q1 = if (valid_count == 0) 0 else quantileSorted(valid_values, 0.25);
    const q3 = if (valid_count == 0) 0 else quantileSorted(valid_values, 0.75);
    const iqr = q3 - q1;
    const lower_fence = q1 - options_value.iqr_multiplier * iqr;
    const upper_fence = q3 + options_value.iqr_multiplier * iqr;

    const deviations = try allocator.alloc(f64, valid_count);
    defer allocator.free(deviations);
    for (valid_values, deviations) |value, *slot| slot.* = @abs(value - median);
    std.sort.insertion(f64, deviations, {}, struct {
        fn lessThan(_: void, lhs: f64, rhs: f64) bool {
            return compareFloatSortValues(f64, lhs, rhs) < 0;
        }
    }.lessThan);
    const mad = if (valid_count == 0) 0 else quantileSorted(deviations, 0.5);

    const rows = values.len;
    const centered = try allocator.alloc(f64, rows);
    defer allocator.free(centered);
    const mad_zscore = try allocator.alloc(f64, rows);
    defer allocator.free(mad_zscore);
    const outlier = try allocator.alloc(bool, rows);
    defer allocator.free(outlier);
    const winsorized = try allocator.alloc(f64, rows);
    defer allocator.free(winsorized);
    const metric_validity = try allocator.alloc(bool, rows);
    defer allocator.free(metric_validity);

    // Robust profiles use order statistics instead of mean/stddev, giving
    // feature-engineering pipelines less outlier-sensitive columns while keeping
    // null propagation and a future device quantile/winsorization lowering seam.
    for (values, 0..) |value_item, row| {
        const row_valid = if (maybe_validity) |mask| mask[row] else true;
        const valid = row_valid and has_enough;
        metric_validity[row] = valid;
        if (valid) {
            const value = castToF64(T, value_item);
            centered[row] = value - median;
            mad_zscore[row] = if (mad == 0) std.math.nan(f64) else 0.6744897501960817 * centered[row] / mad;
            outlier[row] = value < lower_fence or value > upper_fence;
            winsorized[row] = @min(@max(value, lower_fence), upper_fence);
        } else {
            centered[row] = 0;
            mad_zscore[row] = 0;
            outlier[row] = false;
            winsorized[row] = 0;
        }
    }

    var columns: [RobustProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, centered, metric_validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, mad_zscore, metric_validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(bool, allocator, outlier, metric_validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, winsorized, metric_validity, device_value);
    initialized += 1;
    return columns;
}

fn quantileSorted(values: []const f64, probability: f64) f64 {
    if (values.len == 0) return std.math.nan(f64);
    if (values.len == 1) return values[0];
    const position = probability * @as(f64, @floatFromInt(values.len - 1));
    const lower: usize = @intFromFloat(@floor(position));
    const upper: usize = if (lower + 1 < values.len and position > @as(f64, @floatFromInt(lower))) lower + 1 else lower;
    const fraction = position - @as(f64, @floatFromInt(lower));
    return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

const DrawdownProfileColumnCount = 3;

fn drawdownProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![DrawdownProfileColumnCount][]const u8 {
    var names: [DrawdownProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "running_peak", "drawdown", "drawdown_pct" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn drawdownProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceDrawdownOptions,
    device_value: array_mod.Device,
    rows: usize,
) DeviceDataError![DrawdownProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| drawdownProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| drawdownProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| drawdownProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| drawdownProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| drawdownProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| drawdownProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| drawdownProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| drawdownProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| drawdownProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| drawdownProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| drawdownProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| drawdownProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| drawdownProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn drawdownProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceDrawdownOptions,
    device_value: array_mod.Device,
) DeviceDataError![DrawdownProfileColumnCount]DeviceColumn {
    if (options_value.min_periods == 0) return error.InvalidShape;

    const values = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values.len;
    const running_peak = try allocator.alloc(f64, rows);
    defer allocator.free(running_peak);
    const drawdown = try allocator.alloc(f64, rows);
    defer allocator.free(drawdown);
    const drawdown_pct = try allocator.alloc(f64, rows);
    defer allocator.free(drawdown_pct);
    const metric_validity = try allocator.alloc(bool, rows);
    defer allocator.free(metric_validity);

    var valid_count: usize = 0;
    var peak: f64 = 0;
    // Drawdown is inherently order-sensitive, so null rows do not advance the
    // running peak and their derived metrics are null. The output remains in the
    // original row order and gives risk/time-series pipelines a compact seam for
    // later device-side prefix-max lowering.
    for (values, 0..) |value_item, row| {
        const row_valid = if (maybe_validity) |mask| mask[row] else true;
        if (row_valid) {
            const current = castToF64(T, value_item);
            if (valid_count == 0 or current > peak) peak = current;
            valid_count += 1;

            const has_enough = valid_count >= options_value.min_periods;
            metric_validity[row] = has_enough;
            if (has_enough) {
                const dd = current - peak;
                running_peak[row] = peak;
                drawdown[row] = dd;
                drawdown_pct[row] = if (peak == 0) std.math.nan(f64) else dd / peak;
            } else {
                running_peak[row] = 0;
                drawdown[row] = 0;
                drawdown_pct[row] = 0;
            }
        } else {
            metric_validity[row] = false;
            running_peak[row] = 0;
            drawdown[row] = 0;
            drawdown_pct[row] = 0;
        }
    }

    var columns: [DrawdownProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, running_peak, metric_validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, drawdown, metric_validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, drawdown_pct, metric_validity, device_value);
    initialized += 1;
    return columns;
}

const ExtremaProfileColumnCount = 4;

fn extremaProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![ExtremaProfileColumnCount][]const u8 {
    var names: [ExtremaProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "running_low", "running_high", "new_low", "new_high" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn extremaProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceExtremaOptions,
    device_value: array_mod.Device,
    rows: usize,
) DeviceDataError![ExtremaProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| extremaProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| extremaProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| extremaProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| extremaProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| extremaProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| extremaProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| extremaProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| extremaProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| extremaProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| extremaProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| extremaProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| extremaProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| extremaProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn extremaProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceExtremaOptions,
    device_value: array_mod.Device,
) DeviceDataError![ExtremaProfileColumnCount]DeviceColumn {
    if (options_value.min_periods == 0) return error.InvalidShape;

    const values = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values.len;
    const running_low = try allocator.alloc(f64, rows);
    defer allocator.free(running_low);
    const running_high = try allocator.alloc(f64, rows);
    defer allocator.free(running_high);
    const new_low = try allocator.alloc(bool, rows);
    defer allocator.free(new_low);
    const new_high = try allocator.alloc(bool, rows);
    defer allocator.free(new_high);
    const metric_validity = try allocator.alloc(bool, rows);
    defer allocator.free(metric_validity);

    var seen: usize = 0;
    var low: f64 = 0;
    var high: f64 = 0;
    for (values, 0..) |value_item, row| {
        const valid = if (maybe_validity) |mask| mask[row] else true;
        if (!valid) {
            running_low[row] = 0;
            running_high[row] = 0;
            new_low[row] = false;
            new_high[row] = false;
            metric_validity[row] = false;
            continue;
        }
        const value = castToF64(T, value_item);
        const first = seen == 0;
        const is_new_low = first or value < low;
        const is_new_high = first or value > high;
        if (first) {
            low = value;
            high = value;
        } else {
            if (is_new_low) low = value;
            if (is_new_high) high = value;
        }
        seen += 1;
        const has_enough = seen >= options_value.min_periods;
        metric_validity[row] = has_enough;
        if (has_enough) {
            running_low[row] = low;
            running_high[row] = high;
            new_low[row] = is_new_low;
            new_high[row] = is_new_high;
        } else {
            running_low[row] = 0;
            running_high[row] = 0;
            new_low[row] = false;
            new_high[row] = false;
        }
    }

    var columns: [ExtremaProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, running_low, metric_validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, running_high, metric_validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(bool, allocator, new_low, metric_validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(bool, allocator, new_high, metric_validity, device_value);
    initialized += 1;
    return columns;
}

const TrendProfileColumnCount = 5;

fn trendProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![TrendProfileColumnCount][]const u8 {
    var names: [TrendProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "trend", "up_streak", "down_streak", "flat_streak", "reversal" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn trendProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceTrendOptions,
    device_value: array_mod.Device,
    rows: usize,
) DeviceDataError![TrendProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| trendProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| trendProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| trendProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| trendProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| trendProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| trendProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| trendProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| trendProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| trendProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| trendProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| trendProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| trendProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| trendProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn trendProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceTrendOptions,
    device_value: array_mod.Device,
) DeviceDataError![TrendProfileColumnCount]DeviceColumn {
    if (options_value.periods == 0) return error.InvalidShape;

    const values = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values.len;
    const trends = try allocator.alloc(i64, rows);
    defer allocator.free(trends);
    const up_streak = try allocator.alloc(i64, rows);
    defer allocator.free(up_streak);
    const down_streak = try allocator.alloc(i64, rows);
    defer allocator.free(down_streak);
    const flat_streak = try allocator.alloc(i64, rows);
    defer allocator.free(flat_streak);
    const reversal = try allocator.alloc(bool, rows);
    defer allocator.free(reversal);
    const metric_validity = try allocator.alloc(bool, rows);
    defer allocator.free(metric_validity);

    var current_up: i64 = 0;
    var current_down: i64 = 0;
    var current_flat: i64 = 0;
    var previous_nonzero_trend: i64 = 0;

    // Trend profile turns an ordered value series into direction and streak
    // state.  Null or insufficient-history rows reset streak state so downstream
    // models do not accidentally bridge over missing observations.
    for (values, 0..) |value_item, row| {
        if (row < options_value.periods) {
            trends[row] = 0;
            up_streak[row] = 0;
            down_streak[row] = 0;
            flat_streak[row] = 0;
            reversal[row] = false;
            metric_validity[row] = false;
            current_up = 0;
            current_down = 0;
            current_flat = 0;
            previous_nonzero_trend = 0;
            continue;
        }

        const previous_row = row - options_value.periods;
        const row_valid = if (maybe_validity) |mask| mask[row] else true;
        const previous_valid = if (maybe_validity) |mask| mask[previous_row] else true;
        const valid = row_valid and previous_valid;
        metric_validity[row] = valid;
        if (!valid) {
            trends[row] = 0;
            up_streak[row] = 0;
            down_streak[row] = 0;
            flat_streak[row] = 0;
            reversal[row] = false;
            current_up = 0;
            current_down = 0;
            current_flat = 0;
            previous_nonzero_trend = 0;
            continue;
        }

        const current = castToF64(T, value_item);
        const previous = castToF64(T, values[previous_row]);
        const trend: i64 = if (current > previous) 1 else if (current < previous) -1 else 0;
        trends[row] = trend;
        switch (trend) {
            1 => {
                current_up += 1;
                current_down = 0;
                current_flat = 0;
            },
            -1 => {
                current_down += 1;
                current_up = 0;
                current_flat = 0;
            },
            else => {
                current_flat += 1;
                current_up = 0;
                current_down = 0;
            },
        }
        up_streak[row] = current_up;
        down_streak[row] = current_down;
        flat_streak[row] = current_flat;
        reversal[row] = trend != 0 and previous_nonzero_trend != 0 and trend != previous_nonzero_trend;
        if (trend != 0) previous_nonzero_trend = trend;
    }

    var columns: [TrendProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(i64, allocator, trends, metric_validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(i64, allocator, up_streak, metric_validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(i64, allocator, down_streak, metric_validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(i64, allocator, flat_streak, metric_validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(bool, allocator, reversal, metric_validity, device_value);
    initialized += 1;
    return columns;
}

const SignProfileColumnCount = 5;

fn signProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![SignProfileColumnCount][]const u8 {
    var names: [SignProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "sign", "sign_flip", "positive_streak", "negative_streak", "zero_streak" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn signProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceTrendOptions,
    device_value: array_mod.Device,
    rows: usize,
) DeviceDataError![SignProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| signProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| signProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| signProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| signProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| signProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| signProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| signProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| signProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| signProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| signProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| signProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| signProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| signProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn signProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceTrendOptions,
    device_value: array_mod.Device,
) DeviceDataError![SignProfileColumnCount]DeviceColumn {
    if (options_value.periods == 0) return error.InvalidShape;

    const values = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values.len;
    const signs = try allocator.alloc(i64, rows);
    defer allocator.free(signs);
    const flips = try allocator.alloc(bool, rows);
    defer allocator.free(flips);
    const positive_streak = try allocator.alloc(i64, rows);
    defer allocator.free(positive_streak);
    const negative_streak = try allocator.alloc(i64, rows);
    defer allocator.free(negative_streak);
    const zero_streak = try allocator.alloc(i64, rows);
    defer allocator.free(zero_streak);
    const sign_validity = try allocator.alloc(bool, rows);
    defer allocator.free(sign_validity);
    const flip_validity = try allocator.alloc(bool, rows);
    defer allocator.free(flip_validity);

    var pos: i64 = 0;
    var neg: i64 = 0;
    var zero: i64 = 0;
    for (values, 0..) |value_item, row| {
        const valid = if (maybe_validity) |mask| mask[row] else true;
        sign_validity[row] = valid;
        if (!valid) {
            signs[row] = 0;
            flips[row] = false;
            positive_streak[row] = 0;
            negative_streak[row] = 0;
            zero_streak[row] = 0;
            flip_validity[row] = false;
            pos = 0;
            neg = 0;
            zero = 0;
            continue;
        }

        const x = castToF64(T, value_item);
        const sign: i64 = if (x > 0) 1 else if (x < 0) -1 else 0;
        signs[row] = sign;
        switch (sign) {
            1 => {
                pos += 1;
                neg = 0;
                zero = 0;
            },
            -1 => {
                neg += 1;
                pos = 0;
                zero = 0;
            },
            else => {
                zero += 1;
                pos = 0;
                neg = 0;
            },
        }
        positive_streak[row] = pos;
        negative_streak[row] = neg;
        zero_streak[row] = zero;

        if (row < options_value.periods) {
            flips[row] = false;
            flip_validity[row] = false;
        } else {
            const previous_row = row - options_value.periods;
            const previous_valid = if (maybe_validity) |mask| mask[previous_row] else true;
            flip_validity[row] = previous_valid;
            if (previous_valid) {
                const previous_sign: i64 = if (castToF64(T, values[previous_row]) > 0) 1 else if (castToF64(T, values[previous_row]) < 0) -1 else 0;
                flips[row] = sign != previous_sign;
            } else {
                flips[row] = false;
            }
        }
    }

    var columns: [SignProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(i64, allocator, signs, sign_validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(bool, allocator, flips, flip_validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(i64, allocator, positive_streak, sign_validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(i64, allocator, negative_streak, sign_validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(i64, allocator, zero_streak, sign_validity, device_value);
    initialized += 1;
    return columns;
}

const CrossoverProfileColumnCount = 4;

fn crossoverProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![CrossoverProfileColumnCount][]const u8 {
    var names: [CrossoverProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "spread", "ratio", "cross_above", "cross_below" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn crossoverProfileColumnsByValue(
    allocator: std.mem.Allocator,
    lhs: DeviceColumn,
    rhs: DeviceColumn,
    options_value: DeviceCrossoverOptions,
    device_value: array_mod.Device,
    rows: usize,
) DeviceDataError![CrossoverProfileColumnCount]DeviceColumn {
    if (lhs.len() != rows or rhs.len() != rows) return error.LengthMismatch;
    if (lhs.dtype() != rhs.dtype()) return error.TypeMismatch;
    return switch (lhs) {
        .i8 => |typed| crossoverProfileColumnsTyped(i8, allocator, typed, rhs.i8, options_value, device_value),
        .i16 => |typed| crossoverProfileColumnsTyped(i16, allocator, typed, rhs.i16, options_value, device_value),
        .i32 => |typed| crossoverProfileColumnsTyped(i32, allocator, typed, rhs.i32, options_value, device_value),
        .i64 => |typed| crossoverProfileColumnsTyped(i64, allocator, typed, rhs.i64, options_value, device_value),
        .u8 => |typed| crossoverProfileColumnsTyped(u8, allocator, typed, rhs.u8, options_value, device_value),
        .u16 => |typed| crossoverProfileColumnsTyped(u16, allocator, typed, rhs.u16, options_value, device_value),
        .u32 => |typed| crossoverProfileColumnsTyped(u32, allocator, typed, rhs.u32, options_value, device_value),
        .u64 => |typed| crossoverProfileColumnsTyped(u64, allocator, typed, rhs.u64, options_value, device_value),
        .usize => |typed| crossoverProfileColumnsTyped(usize, allocator, typed, rhs.usize, options_value, device_value),
        .isize => |typed| crossoverProfileColumnsTyped(isize, allocator, typed, rhs.isize, options_value, device_value),
        .f16 => |typed| crossoverProfileColumnsTyped(f16, allocator, typed, rhs.f16, options_value, device_value),
        .f32 => |typed| crossoverProfileColumnsTyped(f32, allocator, typed, rhs.f32, options_value, device_value),
        .f64 => |typed| crossoverProfileColumnsTyped(f64, allocator, typed, rhs.f64, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn crossoverProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    lhs: DeviceTypedColumn(T),
    rhs: DeviceTypedColumn(T),
    options_value: DeviceCrossoverOptions,
    device_value: array_mod.Device,
) DeviceDataError![CrossoverProfileColumnCount]DeviceColumn {
    if (options_value.periods == 0) return error.InvalidShape;
    if (lhs.len() != rhs.len()) return error.LengthMismatch;
    if (!lhs.device().sameDevice(rhs.device())) return error.InvalidDevice;

    const lhs_values = try lhs.values.toOwnedSlice(allocator);
    defer allocator.free(lhs_values);
    const rhs_values = try rhs.values.toOwnedSlice(allocator);
    defer allocator.free(rhs_values);
    const maybe_lhs_validity = try validityValues(lhs, allocator);
    defer if (maybe_lhs_validity) |validity| allocator.free(validity);
    const maybe_rhs_validity = try validityValues(rhs, allocator);
    defer if (maybe_rhs_validity) |validity| allocator.free(validity);

    const rows = lhs_values.len;
    const spreads = try allocator.alloc(f64, rows);
    defer allocator.free(spreads);
    const ratios = try allocator.alloc(f64, rows);
    defer allocator.free(ratios);
    const cross_above = try allocator.alloc(bool, rows);
    defer allocator.free(cross_above);
    const cross_below = try allocator.alloc(bool, rows);
    defer allocator.free(cross_below);
    const spread_validity = try allocator.alloc(bool, rows);
    defer allocator.free(spread_validity);
    const cross_validity = try allocator.alloc(bool, rows);
    defer allocator.free(cross_validity);

    // Crossover profiles combine pairwise spread/ratio features with
    // sign-change events in a single pass.  The implementation materializes
    // values today, but the API boundary matches the signal kernels that a
    // future device backend can lower without changing user code.
    for (lhs_values, rhs_values, 0..) |lhs_value, rhs_value, row| {
        const lhs_valid = if (maybe_lhs_validity) |mask| mask[row] else true;
        const rhs_valid = if (maybe_rhs_validity) |mask| mask[row] else true;
        const current_valid = lhs_valid and rhs_valid;
        spread_validity[row] = current_valid;
        if (current_valid) {
            const left = castToF64(T, lhs_value);
            const right = castToF64(T, rhs_value);
            const spread = left - right;
            spreads[row] = spread;
            ratios[row] = if (right == 0) std.math.nan(f64) else left / right;
        } else {
            spreads[row] = 0;
            ratios[row] = 0;
        }

        if (row < options_value.periods) {
            cross_above[row] = false;
            cross_below[row] = false;
            cross_validity[row] = false;
            continue;
        }

        const previous_row = row - options_value.periods;
        const previous_lhs_valid = if (maybe_lhs_validity) |mask| mask[previous_row] else true;
        const previous_rhs_valid = if (maybe_rhs_validity) |mask| mask[previous_row] else true;
        const event_valid = current_valid and previous_lhs_valid and previous_rhs_valid;
        cross_validity[row] = event_valid;
        if (event_valid) {
            const current_spread = castToF64(T, lhs_value) - castToF64(T, rhs_value);
            const previous_spread = castToF64(T, lhs_values[previous_row]) - castToF64(T, rhs_values[previous_row]);
            cross_above[row] = previous_spread <= 0 and current_spread > 0;
            cross_below[row] = previous_spread >= 0 and current_spread < 0;
        } else {
            cross_above[row] = false;
            cross_below[row] = false;
        }
    }

    var columns: [CrossoverProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, spreads, spread_validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, ratios, spread_validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(bool, allocator, cross_above, cross_validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(bool, allocator, cross_below, cross_validity, device_value);
    initialized += 1;
    return columns;
}

const BucketProfileColumnCount = 4;

fn bucketProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![BucketProfileColumnCount][]const u8 {
    var names: [BucketProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "ecdf", "bucket", "lower_tail", "upper_tail" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn bucketProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceBucketOptions,
    device_value: array_mod.Device,
    rows: usize,
) DeviceDataError![BucketProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .bool => |typed| bucketProfileColumnsTyped(bool, allocator, typed, options_value, device_value),
        .i8 => |typed| bucketProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| bucketProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| bucketProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| bucketProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| bucketProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| bucketProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| bucketProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| bucketProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| bucketProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| bucketProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| bucketProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| bucketProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| bucketProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn bucketProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceBucketOptions,
    device_value: array_mod.Device,
) DeviceDataError![BucketProfileColumnCount]DeviceColumn {
    if (options_value.buckets == 0 or options_value.min_periods == 0) return error.InvalidShape;
    if (options_value.lower_quantile < 0 or options_value.lower_quantile > 1) return error.InvalidShape;
    if (options_value.upper_quantile < 0 or options_value.upper_quantile > 1) return error.InvalidShape;
    if (options_value.lower_quantile > options_value.upper_quantile) return error.InvalidShape;

    const values = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const order = try argsortTypedColumn(T, column, allocator, .{ .descending = false, .nulls = .last });
    defer allocator.free(order);

    var valid_count: usize = 0;
    for (values, 0..) |_, row| {
        const valid = if (maybe_validity) |mask| mask[row] else true;
        if (valid) valid_count += 1;
    }

    const rows = values.len;
    const ecdf = try allocator.alloc(f64, rows);
    defer allocator.free(ecdf);
    const buckets = try allocator.alloc(i64, rows);
    defer allocator.free(buckets);
    const lower_tail = try allocator.alloc(bool, rows);
    defer allocator.free(lower_tail);
    const upper_tail = try allocator.alloc(bool, rows);
    defer allocator.free(upper_tail);
    const metric_validity = try allocator.alloc(bool, rows);
    defer allocator.free(metric_validity);

    @memset(ecdf, 0);
    @memset(buckets, 0);
    @memset(lower_tail, false);
    @memset(upper_tail, false);
    @memset(metric_validity, false);

    if (valid_count >= options_value.min_periods and valid_count != 0) {
        var group_start: usize = 0;
        while (group_start < valid_count) {
            var group_end = group_start + 1;
            while (group_end < valid_count and bucketKeysTie(T, values, order[group_start], order[group_end])) {
                group_end += 1;
            }

            const rank_position = group_end; // right-continuous ECDF, 1-based.
            const ecdf_value = @as(f64, @floatFromInt(rank_position)) / @as(f64, @floatFromInt(valid_count));
            var bucket_index = @divFloor((rank_position - 1) * options_value.buckets, valid_count);
            if (bucket_index >= options_value.buckets) bucket_index = options_value.buckets - 1;
            const is_lower = ecdf_value <= options_value.lower_quantile;
            const is_upper = ecdf_value >= options_value.upper_quantile;

            for (order[group_start..group_end]) |row| {
                ecdf[row] = ecdf_value;
                buckets[row] = @intCast(bucket_index);
                lower_tail[row] = is_lower;
                upper_tail[row] = is_upper;
                metric_validity[row] = true;
            }
            group_start = group_end;
        }
    }

    var columns: [BucketProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, ecdf, metric_validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(i64, allocator, buckets, metric_validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(bool, allocator, lower_tail, metric_validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(bool, allocator, upper_tail, metric_validity, device_value);
    initialized += 1;
    return columns;
}

fn bucketKeysTie(comptime T: type, values: []const T, lhs: usize, rhs: usize) bool {
    if (comptime T == bool) return values[lhs] == values[rhs];
    return compareSortValues(T, values[lhs], values[rhs]) == 0;
}

const EmaProfileColumnCount = 3;

fn emaProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![EmaProfileColumnCount][]const u8 {
    var names: [EmaProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "ema", "ema_residual", "ema_ratio" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn emaProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceEmaOptions,
    device_value: array_mod.Device,
    rows: usize,
) DeviceDataError![EmaProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| emaProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| emaProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| emaProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| emaProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| emaProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| emaProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| emaProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| emaProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| emaProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| emaProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| emaProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| emaProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| emaProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn emaProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceEmaOptions,
    device_value: array_mod.Device,
) DeviceDataError![EmaProfileColumnCount]DeviceColumn {
    if (options_value.alpha <= 0 or options_value.alpha > 1 or options_value.min_periods == 0) return error.InvalidShape;

    const values = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values.len;
    const ema_values = try allocator.alloc(f64, rows);
    defer allocator.free(ema_values);
    const residuals = try allocator.alloc(f64, rows);
    defer allocator.free(residuals);
    const ratios = try allocator.alloc(f64, rows);
    defer allocator.free(ratios);
    const metric_validity = try allocator.alloc(bool, rows);
    defer allocator.free(metric_validity);

    var seen: usize = 0;
    var ema: f64 = 0;
    // Null observations do not update EMA state. This keeps sequence gaps from
    // biasing the exponential smoother while preserving row-aligned nullable
    // outputs for downstream feature engineering.
    for (values, 0..) |value_item, row| {
        const row_valid = if (maybe_validity) |mask| mask[row] else true;
        if (!row_valid) {
            ema_values[row] = 0;
            residuals[row] = 0;
            ratios[row] = 0;
            metric_validity[row] = false;
            continue;
        }

        const x = castToF64(T, value_item);
        if (seen == 0) {
            ema = x;
        } else {
            ema = options_value.alpha * x + (1.0 - options_value.alpha) * ema;
        }
        seen += 1;

        const has_enough = seen >= options_value.min_periods;
        metric_validity[row] = has_enough;
        if (has_enough) {
            ema_values[row] = ema;
            residuals[row] = x - ema;
            ratios[row] = if (ema == 0) std.math.nan(f64) else x / ema;
        } else {
            ema_values[row] = 0;
            residuals[row] = 0;
            ratios[row] = 0;
        }
    }

    var columns: [EmaProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, ema_values, metric_validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, residuals, metric_validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, ratios, metric_validity, device_value);
    initialized += 1;
    return columns;
}

const LinearFitProfileColumnCount = 4;

fn linearFitProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![LinearFitProfileColumnCount][]const u8 {
    var names: [LinearFitProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "fitted", "residual", "residual_zscore", "slope" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn linearFitProfileColumnsByValue(
    allocator: std.mem.Allocator,
    x: DeviceColumn,
    y: DeviceColumn,
    options_value: DeviceLinearFitOptions,
    device_value: array_mod.Device,
    rows: usize,
) DeviceDataError![LinearFitProfileColumnCount]DeviceColumn {
    if (x.len() != rows or y.len() != rows) return error.LengthMismatch;
    if (x.dtype() != y.dtype()) return error.TypeMismatch;
    return switch (x) {
        .i8 => |typed| linearFitProfileColumnsTyped(i8, allocator, typed, y.i8, options_value, device_value),
        .i16 => |typed| linearFitProfileColumnsTyped(i16, allocator, typed, y.i16, options_value, device_value),
        .i32 => |typed| linearFitProfileColumnsTyped(i32, allocator, typed, y.i32, options_value, device_value),
        .i64 => |typed| linearFitProfileColumnsTyped(i64, allocator, typed, y.i64, options_value, device_value),
        .u8 => |typed| linearFitProfileColumnsTyped(u8, allocator, typed, y.u8, options_value, device_value),
        .u16 => |typed| linearFitProfileColumnsTyped(u16, allocator, typed, y.u16, options_value, device_value),
        .u32 => |typed| linearFitProfileColumnsTyped(u32, allocator, typed, y.u32, options_value, device_value),
        .u64 => |typed| linearFitProfileColumnsTyped(u64, allocator, typed, y.u64, options_value, device_value),
        .usize => |typed| linearFitProfileColumnsTyped(usize, allocator, typed, y.usize, options_value, device_value),
        .isize => |typed| linearFitProfileColumnsTyped(isize, allocator, typed, y.isize, options_value, device_value),
        .f16 => |typed| linearFitProfileColumnsTyped(f16, allocator, typed, y.f16, options_value, device_value),
        .f32 => |typed| linearFitProfileColumnsTyped(f32, allocator, typed, y.f32, options_value, device_value),
        .f64 => |typed| linearFitProfileColumnsTyped(f64, allocator, typed, y.f64, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn linearFitProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    x_column: DeviceTypedColumn(T),
    y_column: DeviceTypedColumn(T),
    options_value: DeviceLinearFitOptions,
    device_value: array_mod.Device,
) DeviceDataError![LinearFitProfileColumnCount]DeviceColumn {
    if (options_value.min_periods == 0) return error.InvalidShape;
    if (x_column.len() != y_column.len()) return error.LengthMismatch;
    if (!x_column.device().sameDevice(y_column.device())) return error.InvalidDevice;

    const xs = try x_column.values.toOwnedSlice(allocator);
    defer allocator.free(xs);
    const ys = try y_column.values.toOwnedSlice(allocator);
    defer allocator.free(ys);
    const maybe_x_validity = try validityValues(x_column, allocator);
    defer if (maybe_x_validity) |validity| allocator.free(validity);
    const maybe_y_validity = try validityValues(y_column, allocator);
    defer if (maybe_y_validity) |validity| allocator.free(validity);

    var count: usize = 0;
    var sum_x: f64 = 0;
    var sum_y: f64 = 0;
    var sum_xx: f64 = 0;
    var sum_xy: f64 = 0;
    for (xs, ys, 0..) |x_value, y_value, row| {
        const valid = (if (maybe_x_validity) |mask| mask[row] else true) and (if (maybe_y_validity) |mask| mask[row] else true);
        if (!valid) continue;
        const x = castToF64(T, x_value);
        const y = castToF64(T, y_value);
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
        count += 1;
    }

    const rows = xs.len;
    const fitted = try allocator.alloc(f64, rows);
    defer allocator.free(fitted);
    const residuals = try allocator.alloc(f64, rows);
    defer allocator.free(residuals);
    const residual_z = try allocator.alloc(f64, rows);
    defer allocator.free(residual_z);
    const slopes = try allocator.alloc(f64, rows);
    defer allocator.free(slopes);
    const metric_validity = try allocator.alloc(bool, rows);
    defer allocator.free(metric_validity);

    const has_fit = count >= options_value.min_periods;
    const denom = @as(f64, @floatFromInt(count)) * sum_xx - sum_x * sum_x;
    const slope = if (has_fit and denom != 0) (@as(f64, @floatFromInt(count)) * sum_xy - sum_x * sum_y) / denom else std.math.nan(f64);
    const intercept = if (has_fit and !std.math.isNan(slope)) (sum_y - slope * sum_x) / @as(f64, @floatFromInt(count)) else std.math.nan(f64);

    var residual_sum_sq: f64 = 0;
    if (has_fit and !std.math.isNan(slope)) {
        for (xs, ys, 0..) |x_value, y_value, row| {
            const valid = (if (maybe_x_validity) |mask| mask[row] else true) and (if (maybe_y_validity) |mask| mask[row] else true);
            if (!valid) continue;
            const fit = intercept + slope * castToF64(T, x_value);
            const residual = castToF64(T, y_value) - fit;
            residual_sum_sq += residual * residual;
        }
    }
    const residual_std = if (count == 0 or std.math.isNan(slope)) std.math.nan(f64) else std.math.sqrt(residual_sum_sq / @as(f64, @floatFromInt(count)));

    // Fit one global y = intercept + slope*x model and emit row-aligned
    // diagnostics. This keeps model diagnostics in the dataframe API while
    // leaving a future backend seam for regression kernels.
    for (xs, ys, 0..) |x_value, y_value, row| {
        const row_valid = (if (maybe_x_validity) |mask| mask[row] else true) and (if (maybe_y_validity) |mask| mask[row] else true);
        const valid = row_valid and has_fit;
        metric_validity[row] = valid;
        if (valid) {
            const fit = intercept + slope * castToF64(T, x_value);
            const residual = castToF64(T, y_value) - fit;
            fitted[row] = fit;
            residuals[row] = residual;
            residual_z[row] = if (residual_std == 0 or std.math.isNan(residual_std)) std.math.nan(f64) else residual / residual_std;
            slopes[row] = slope;
        } else {
            fitted[row] = 0;
            residuals[row] = 0;
            residual_z[row] = 0;
            slopes[row] = 0;
        }
    }

    var columns: [LinearFitProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, fitted, metric_validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, residuals, metric_validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, residual_z, metric_validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, slopes, metric_validity, device_value);
    initialized += 1;
    return columns;
}

const ErrorProfileColumnCount = 5;

fn errorProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![ErrorProfileColumnCount][]const u8 {
    var names: [ErrorProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "error", "abs_error", "squared_error", "ape", "smape" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn errorProfileColumnsByValue(
    allocator: std.mem.Allocator,
    actual: DeviceColumn,
    predicted: DeviceColumn,
    device_value: array_mod.Device,
    rows: usize,
) DeviceDataError![ErrorProfileColumnCount]DeviceColumn {
    if (actual.len() != rows or predicted.len() != rows) return error.LengthMismatch;
    if (actual.dtype() != predicted.dtype()) return error.TypeMismatch;
    return switch (actual) {
        .i8 => |typed| errorProfileColumnsTyped(i8, allocator, typed, predicted.i8, device_value),
        .i16 => |typed| errorProfileColumnsTyped(i16, allocator, typed, predicted.i16, device_value),
        .i32 => |typed| errorProfileColumnsTyped(i32, allocator, typed, predicted.i32, device_value),
        .i64 => |typed| errorProfileColumnsTyped(i64, allocator, typed, predicted.i64, device_value),
        .u8 => |typed| errorProfileColumnsTyped(u8, allocator, typed, predicted.u8, device_value),
        .u16 => |typed| errorProfileColumnsTyped(u16, allocator, typed, predicted.u16, device_value),
        .u32 => |typed| errorProfileColumnsTyped(u32, allocator, typed, predicted.u32, device_value),
        .u64 => |typed| errorProfileColumnsTyped(u64, allocator, typed, predicted.u64, device_value),
        .usize => |typed| errorProfileColumnsTyped(usize, allocator, typed, predicted.usize, device_value),
        .isize => |typed| errorProfileColumnsTyped(isize, allocator, typed, predicted.isize, device_value),
        .f16 => |typed| errorProfileColumnsTyped(f16, allocator, typed, predicted.f16, device_value),
        .f32 => |typed| errorProfileColumnsTyped(f32, allocator, typed, predicted.f32, device_value),
        .f64 => |typed| errorProfileColumnsTyped(f64, allocator, typed, predicted.f64, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn errorProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    actual_column: DeviceTypedColumn(T),
    predicted_column: DeviceTypedColumn(T),
    device_value: array_mod.Device,
) DeviceDataError![ErrorProfileColumnCount]DeviceColumn {
    if (actual_column.len() != predicted_column.len()) return error.LengthMismatch;
    if (!actual_column.device().sameDevice(predicted_column.device())) return error.InvalidDevice;

    const actual_values = try actual_column.values.toOwnedSlice(allocator);
    defer allocator.free(actual_values);
    const predicted_values = try predicted_column.values.toOwnedSlice(allocator);
    defer allocator.free(predicted_values);
    const maybe_actual_validity = try validityValues(actual_column, allocator);
    defer if (maybe_actual_validity) |validity| allocator.free(validity);
    const maybe_predicted_validity = try validityValues(predicted_column, allocator);
    defer if (maybe_predicted_validity) |validity| allocator.free(validity);

    const rows = actual_values.len;
    const errors = try allocator.alloc(f64, rows);
    defer allocator.free(errors);
    const abs_errors = try allocator.alloc(f64, rows);
    defer allocator.free(abs_errors);
    const squared_errors = try allocator.alloc(f64, rows);
    defer allocator.free(squared_errors);
    const ape = try allocator.alloc(f64, rows);
    defer allocator.free(ape);
    const smape = try allocator.alloc(f64, rows);
    defer allocator.free(smape);
    const metric_validity = try allocator.alloc(bool, rows);
    defer allocator.free(metric_validity);

    for (actual_values, predicted_values, 0..) |actual_value, predicted_value, row| {
        const valid = (if (maybe_actual_validity) |mask| mask[row] else true) and (if (maybe_predicted_validity) |mask| mask[row] else true);
        metric_validity[row] = valid;
        if (valid) {
            const actual = castToF64(T, actual_value);
            const predicted = castToF64(T, predicted_value);
            const err = actual - predicted;
            const abs_err = @abs(err);
            const denom = @abs(actual) + @abs(predicted);
            errors[row] = err;
            abs_errors[row] = abs_err;
            squared_errors[row] = err * err;
            ape[row] = if (actual == 0) std.math.nan(f64) else abs_err / @abs(actual);
            smape[row] = if (denom == 0) std.math.nan(f64) else 2.0 * abs_err / denom;
        } else {
            errors[row] = 0;
            abs_errors[row] = 0;
            squared_errors[row] = 0;
            ape[row] = 0;
            smape[row] = 0;
        }
    }

    var columns: [ErrorProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, errors, metric_validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, abs_errors, metric_validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, squared_errors, metric_validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, ape, metric_validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, smape, metric_validity, device_value);
    initialized += 1;
    return columns;
}

const ClassificationProfileColumnCount = 5;

fn classificationProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![ClassificationProfileColumnCount][]const u8 {
    var names: [ClassificationProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "tp", "fp", "tn", "fn", "correct" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn classificationProfileColumns(
    allocator: std.mem.Allocator,
    actual: DeviceTypedColumn(bool),
    predicted: DeviceTypedColumn(bool),
    device_value: array_mod.Device,
    rows: usize,
) DeviceDataError![ClassificationProfileColumnCount]DeviceColumn {
    if (actual.len() != rows or predicted.len() != rows) return error.LengthMismatch;
    if (!actual.device().sameDevice(predicted.device())) return error.InvalidDevice;

    const actual_values = try actual.values.toOwnedSlice(allocator);
    defer allocator.free(actual_values);
    const predicted_values = try predicted.values.toOwnedSlice(allocator);
    defer allocator.free(predicted_values);
    const maybe_actual_validity = try validityValues(actual, allocator);
    defer if (maybe_actual_validity) |validity| allocator.free(validity);
    const maybe_predicted_validity = try validityValues(predicted, allocator);
    defer if (maybe_predicted_validity) |validity| allocator.free(validity);

    const tp = try allocator.alloc(bool, rows);
    defer allocator.free(tp);
    const fp = try allocator.alloc(bool, rows);
    defer allocator.free(fp);
    const tn = try allocator.alloc(bool, rows);
    defer allocator.free(tn);
    const fn_values = try allocator.alloc(bool, rows);
    defer allocator.free(fn_values);
    const correct = try allocator.alloc(bool, rows);
    defer allocator.free(correct);
    const metric_validity = try allocator.alloc(bool, rows);
    defer allocator.free(metric_validity);

    for (actual_values, predicted_values, 0..) |actual_value, predicted_value, row| {
        const valid = (if (maybe_actual_validity) |mask| mask[row] else true) and (if (maybe_predicted_validity) |mask| mask[row] else true);
        metric_validity[row] = valid;
        if (valid) {
            tp[row] = actual_value and predicted_value;
            fp[row] = !actual_value and predicted_value;
            tn[row] = !actual_value and !predicted_value;
            fn_values[row] = actual_value and !predicted_value;
            correct[row] = actual_value == predicted_value;
        } else {
            tp[row] = false;
            fp[row] = false;
            tn[row] = false;
            fn_values[row] = false;
            correct[row] = false;
        }
    }

    var columns: [ClassificationProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(bool, allocator, tp, metric_validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(bool, allocator, fp, metric_validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(bool, allocator, tn, metric_validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(bool, allocator, fn_values, metric_validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(bool, allocator, correct, metric_validity, device_value);
    initialized += 1;
    return columns;
}

const RollingCorrelationProfileColumnCount = 4;

fn rollingCorrelationProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![RollingCorrelationProfileColumnCount][]const u8 {
    var names: [RollingCorrelationProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "rolling_pair_count", "rolling_covariance", "rolling_correlation", "rolling_beta" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn rollingCorrelationProfileColumnsByValue(
    allocator: std.mem.Allocator,
    x: DeviceColumn,
    y: DeviceColumn,
    options_value: DeviceRollingCorrelationOptions,
    device_value: array_mod.Device,
    rows: usize,
) DeviceDataError![RollingCorrelationProfileColumnCount]DeviceColumn {
    if (x.len() != rows or y.len() != rows) return error.LengthMismatch;
    if (x.dtype() != y.dtype()) return error.TypeMismatch;
    return switch (x) {
        .i8 => |typed| rollingCorrelationProfileColumnsTyped(i8, allocator, typed, y.i8, options_value, device_value),
        .i16 => |typed| rollingCorrelationProfileColumnsTyped(i16, allocator, typed, y.i16, options_value, device_value),
        .i32 => |typed| rollingCorrelationProfileColumnsTyped(i32, allocator, typed, y.i32, options_value, device_value),
        .i64 => |typed| rollingCorrelationProfileColumnsTyped(i64, allocator, typed, y.i64, options_value, device_value),
        .u8 => |typed| rollingCorrelationProfileColumnsTyped(u8, allocator, typed, y.u8, options_value, device_value),
        .u16 => |typed| rollingCorrelationProfileColumnsTyped(u16, allocator, typed, y.u16, options_value, device_value),
        .u32 => |typed| rollingCorrelationProfileColumnsTyped(u32, allocator, typed, y.u32, options_value, device_value),
        .u64 => |typed| rollingCorrelationProfileColumnsTyped(u64, allocator, typed, y.u64, options_value, device_value),
        .usize => |typed| rollingCorrelationProfileColumnsTyped(usize, allocator, typed, y.usize, options_value, device_value),
        .isize => |typed| rollingCorrelationProfileColumnsTyped(isize, allocator, typed, y.isize, options_value, device_value),
        .f16 => |typed| rollingCorrelationProfileColumnsTyped(f16, allocator, typed, y.f16, options_value, device_value),
        .f32 => |typed| rollingCorrelationProfileColumnsTyped(f32, allocator, typed, y.f32, options_value, device_value),
        .f64 => |typed| rollingCorrelationProfileColumnsTyped(f64, allocator, typed, y.f64, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn rollingCorrelationProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    x_column: DeviceTypedColumn(T),
    y_column: DeviceTypedColumn(T),
    options_value: DeviceRollingCorrelationOptions,
    device_value: array_mod.Device,
) DeviceDataError![RollingCorrelationProfileColumnCount]DeviceColumn {
    if (options_value.window == 0) return error.InvalidShape;
    const min_periods = options_value.min_periods orelse options_value.window;
    if (min_periods == 0 or min_periods > options_value.window) return error.InvalidShape;
    if (x_column.len() != y_column.len()) return error.LengthMismatch;
    if (!x_column.device().sameDevice(y_column.device())) return error.InvalidDevice;

    const xs = try x_column.values.toOwnedSlice(allocator);
    defer allocator.free(xs);
    const ys = try y_column.values.toOwnedSlice(allocator);
    defer allocator.free(ys);
    const maybe_x_validity = try validityValues(x_column, allocator);
    defer if (maybe_x_validity) |validity| allocator.free(validity);
    const maybe_y_validity = try validityValues(y_column, allocator);
    defer if (maybe_y_validity) |validity| allocator.free(validity);

    const rows = xs.len;
    const pair_counts = try allocator.alloc(i64, rows);
    defer allocator.free(pair_counts);
    const covariances = try allocator.alloc(f64, rows);
    defer allocator.free(covariances);
    const correlations = try allocator.alloc(f64, rows);
    defer allocator.free(correlations);
    const betas = try allocator.alloc(f64, rows);
    defer allocator.free(betas);
    const metric_validity = try allocator.alloc(bool, rows);
    defer allocator.free(metric_validity);

    // Recompute each trailing window in host memory, mirroring the existing
    // dataframe rolling profile APIs while exposing a stable seam for future
    // device-side rolling covariance/correlation kernels.
    for (xs, 0..) |_, row| {
        const start = if (row + 1 > options_value.window) row + 1 - options_value.window else 0;
        var count: usize = 0;
        var sum_x: f64 = 0;
        var sum_y: f64 = 0;
        var sum_xx: f64 = 0;
        var sum_yy: f64 = 0;
        var sum_xy: f64 = 0;
        for (start..row + 1) |window_row| {
            const valid = (if (maybe_x_validity) |mask| mask[window_row] else true) and (if (maybe_y_validity) |mask| mask[window_row] else true);
            if (!valid) continue;
            const x = castToF64(T, xs[window_row]);
            const y = castToF64(T, ys[window_row]);
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_yy += y * y;
            sum_xy += x * y;
            count += 1;
        }

        pair_counts[row] = @intCast(count);
        const has_enough = count >= min_periods;
        metric_validity[row] = has_enough;
        if (has_enough) {
            const n: f64 = @floatFromInt(count);
            const mean_x = sum_x / n;
            const mean_y = sum_y / n;
            const cov = sum_xy / n - mean_x * mean_y;
            const var_x_raw = sum_xx / n - mean_x * mean_x;
            const var_y_raw = sum_yy / n - mean_y * mean_y;
            const var_x = if (var_x_raw < 0) 0 else var_x_raw;
            const var_y = if (var_y_raw < 0) 0 else var_y_raw;
            covariances[row] = cov;
            correlations[row] = if (var_x == 0 or var_y == 0) std.math.nan(f64) else cov / std.math.sqrt(var_x * var_y);
            betas[row] = if (var_x == 0) std.math.nan(f64) else cov / var_x;
        } else {
            covariances[row] = 0;
            correlations[row] = 0;
            betas[row] = 0;
        }
    }

    var columns: [RollingCorrelationProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, pair_counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, covariances, metric_validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, correlations, metric_validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, betas, metric_validity, device_value);
    initialized += 1;
    return columns;
}

const ValidityProfileColumnCount = 4;

fn validityProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![ValidityProfileColumnCount][]const u8 {
    var names: [ValidityProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "is_null", "is_valid", "valid_streak", "null_streak" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn validityProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    device_value: array_mod.Device,
    rows: usize,
) DeviceDataError![ValidityProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        inline else => |typed| validityProfileColumnsTyped(allocator, typed, device_value),
    };
}

fn validityProfileColumnsTyped(
    allocator: std.mem.Allocator,
    column: anytype,
    device_value: array_mod.Device,
) DeviceDataError![ValidityProfileColumnCount]DeviceColumn {
    const rows = column.len();
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const is_null = try allocator.alloc(bool, rows);
    defer allocator.free(is_null);
    const is_valid = try allocator.alloc(bool, rows);
    defer allocator.free(is_valid);
    const valid_streak = try allocator.alloc(i64, rows);
    defer allocator.free(valid_streak);
    const null_streak = try allocator.alloc(i64, rows);
    defer allocator.free(null_streak);

    var current_valid_streak: i64 = 0;
    var current_null_streak: i64 = 0;
    for (0..rows) |row| {
        const valid = if (maybe_validity) |validity| validity[row] else true;
        is_valid[row] = valid;
        is_null[row] = !valid;
        if (valid) {
            current_valid_streak += 1;
            current_null_streak = 0;
        } else {
            current_null_streak += 1;
            current_valid_streak = 0;
        }
        valid_streak[row] = current_valid_streak;
        null_streak[row] = current_null_streak;
    }

    var columns: [ValidityProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(bool, allocator, is_null, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSlice(bool, allocator, is_valid, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSlice(i64, allocator, valid_streak, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSlice(i64, allocator, null_streak, device_value);
    initialized += 1;
    return columns;
}

fn groupByCountTyped(
    comptime K: type,
    allocator: std.mem.Allocator,
    key_name: []const u8,
    output_name: []const u8,
    key: DeviceTypedColumn(K),
    device_value: array_mod.Device,
) DeviceDataError!DeviceDataFrame {
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
    return initAggregatedDataFrame(allocator, key_name, key_col, output_name, count_col, device_value);
}

fn groupByNumericDispatchKey(
    op: DeviceGroupByAggregation,
    allocator: std.mem.Allocator,
    key_name: []const u8,
    output_name: []const u8,
    key: DeviceColumn,
    value: DeviceColumn,
    device_value: array_mod.Device,
) DeviceDataError!DeviceDataFrame {
    return switch (key) {
        .bool => |typed| groupByNumericDispatchValue(op, bool, allocator, key_name, output_name, typed, value, device_value),
        .i8 => |typed| groupByNumericDispatchValue(op, i8, allocator, key_name, output_name, typed, value, device_value),
        .i16 => |typed| groupByNumericDispatchValue(op, i16, allocator, key_name, output_name, typed, value, device_value),
        .i32 => |typed| groupByNumericDispatchValue(op, i32, allocator, key_name, output_name, typed, value, device_value),
        .i64 => |typed| groupByNumericDispatchValue(op, i64, allocator, key_name, output_name, typed, value, device_value),
        .u8 => |typed| groupByNumericDispatchValue(op, u8, allocator, key_name, output_name, typed, value, device_value),
        .u16 => |typed| groupByNumericDispatchValue(op, u16, allocator, key_name, output_name, typed, value, device_value),
        .u32 => |typed| groupByNumericDispatchValue(op, u32, allocator, key_name, output_name, typed, value, device_value),
        .u64 => |typed| groupByNumericDispatchValue(op, u64, allocator, key_name, output_name, typed, value, device_value),
        .usize => |typed| groupByNumericDispatchValue(op, usize, allocator, key_name, output_name, typed, value, device_value),
        .isize => |typed| groupByNumericDispatchValue(op, isize, allocator, key_name, output_name, typed, value, device_value),
        .f16 => |typed| groupByNumericDispatchValue(op, f16, allocator, key_name, output_name, typed, value, device_value),
        .f32 => |typed| groupByNumericDispatchValue(op, f32, allocator, key_name, output_name, typed, value, device_value),
        .f64 => |typed| groupByNumericDispatchValue(op, f64, allocator, key_name, output_name, typed, value, device_value),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn groupByNumericDispatchValue(
    op: DeviceGroupByAggregation,
    comptime K: type,
    allocator: std.mem.Allocator,
    key_name: []const u8,
    output_name: []const u8,
    key: DeviceTypedColumn(K),
    value: DeviceColumn,
    device_value: array_mod.Device,
) DeviceDataError!DeviceDataFrame {
    return switch (value) {
        .i8 => |typed| groupByNumericTyped(op, K, i8, allocator, key_name, output_name, key, typed, device_value),
        .i16 => |typed| groupByNumericTyped(op, K, i16, allocator, key_name, output_name, key, typed, device_value),
        .i32 => |typed| groupByNumericTyped(op, K, i32, allocator, key_name, output_name, key, typed, device_value),
        .i64 => |typed| groupByNumericTyped(op, K, i64, allocator, key_name, output_name, key, typed, device_value),
        .u8 => |typed| groupByNumericTyped(op, K, u8, allocator, key_name, output_name, key, typed, device_value),
        .u16 => |typed| groupByNumericTyped(op, K, u16, allocator, key_name, output_name, key, typed, device_value),
        .u32 => |typed| groupByNumericTyped(op, K, u32, allocator, key_name, output_name, key, typed, device_value),
        .u64 => |typed| groupByNumericTyped(op, K, u64, allocator, key_name, output_name, key, typed, device_value),
        .usize => |typed| groupByNumericTyped(op, K, usize, allocator, key_name, output_name, key, typed, device_value),
        .isize => |typed| groupByNumericTyped(op, K, isize, allocator, key_name, output_name, key, typed, device_value),
        .f16 => |typed| groupByNumericTyped(op, K, f16, allocator, key_name, output_name, key, typed, device_value),
        .f32 => |typed| groupByNumericTyped(op, K, f32, allocator, key_name, output_name, key, typed, device_value),
        .f64 => |typed| groupByNumericTyped(op, K, f64, allocator, key_name, output_name, key, typed, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn groupByNumericTyped(
    op: DeviceGroupByAggregation,
    comptime K: type,
    comptime V: type,
    allocator: std.mem.Allocator,
    key_name: []const u8,
    output_name: []const u8,
    key: DeviceTypedColumn(K),
    value: DeviceTypedColumn(V),
    device_value: array_mod.Device,
) DeviceDataError!DeviceDataFrame {
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
    return initAggregatedDataFrame(allocator, key_name, key_col, output_name, aggregate_col, device_value);
}

fn groupByMeanDispatchKey(
    allocator: std.mem.Allocator,
    key_name: []const u8,
    output_name: []const u8,
    key: DeviceColumn,
    value: DeviceColumn,
    device_value: array_mod.Device,
) DeviceDataError!DeviceDataFrame {
    return switch (key) {
        .bool => |typed| groupByMeanDispatchValue(bool, allocator, key_name, output_name, typed, value, device_value),
        .i8 => |typed| groupByMeanDispatchValue(i8, allocator, key_name, output_name, typed, value, device_value),
        .i16 => |typed| groupByMeanDispatchValue(i16, allocator, key_name, output_name, typed, value, device_value),
        .i32 => |typed| groupByMeanDispatchValue(i32, allocator, key_name, output_name, typed, value, device_value),
        .i64 => |typed| groupByMeanDispatchValue(i64, allocator, key_name, output_name, typed, value, device_value),
        .u8 => |typed| groupByMeanDispatchValue(u8, allocator, key_name, output_name, typed, value, device_value),
        .u16 => |typed| groupByMeanDispatchValue(u16, allocator, key_name, output_name, typed, value, device_value),
        .u32 => |typed| groupByMeanDispatchValue(u32, allocator, key_name, output_name, typed, value, device_value),
        .u64 => |typed| groupByMeanDispatchValue(u64, allocator, key_name, output_name, typed, value, device_value),
        .usize => |typed| groupByMeanDispatchValue(usize, allocator, key_name, output_name, typed, value, device_value),
        .isize => |typed| groupByMeanDispatchValue(isize, allocator, key_name, output_name, typed, value, device_value),
        .f16 => |typed| groupByMeanDispatchValue(f16, allocator, key_name, output_name, typed, value, device_value),
        .f32 => |typed| groupByMeanDispatchValue(f32, allocator, key_name, output_name, typed, value, device_value),
        .f64 => |typed| groupByMeanDispatchValue(f64, allocator, key_name, output_name, typed, value, device_value),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn groupByMeanDispatchValue(
    comptime K: type,
    allocator: std.mem.Allocator,
    key_name: []const u8,
    output_name: []const u8,
    key: DeviceTypedColumn(K),
    value: DeviceColumn,
    device_value: array_mod.Device,
) DeviceDataError!DeviceDataFrame {
    return switch (value) {
        .i8 => |typed| groupByMeanTyped(K, i8, allocator, key_name, output_name, key, typed, device_value),
        .i16 => |typed| groupByMeanTyped(K, i16, allocator, key_name, output_name, key, typed, device_value),
        .i32 => |typed| groupByMeanTyped(K, i32, allocator, key_name, output_name, key, typed, device_value),
        .i64 => |typed| groupByMeanTyped(K, i64, allocator, key_name, output_name, key, typed, device_value),
        .u8 => |typed| groupByMeanTyped(K, u8, allocator, key_name, output_name, key, typed, device_value),
        .u16 => |typed| groupByMeanTyped(K, u16, allocator, key_name, output_name, key, typed, device_value),
        .u32 => |typed| groupByMeanTyped(K, u32, allocator, key_name, output_name, key, typed, device_value),
        .u64 => |typed| groupByMeanTyped(K, u64, allocator, key_name, output_name, key, typed, device_value),
        .usize => |typed| groupByMeanTyped(K, usize, allocator, key_name, output_name, key, typed, device_value),
        .isize => |typed| groupByMeanTyped(K, isize, allocator, key_name, output_name, key, typed, device_value),
        .f16 => |typed| groupByMeanTyped(K, f16, allocator, key_name, output_name, key, typed, device_value),
        .f32 => |typed| groupByMeanTyped(K, f32, allocator, key_name, output_name, key, typed, device_value),
        .f64 => |typed| groupByMeanTyped(K, f64, allocator, key_name, output_name, key, typed, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn groupByMeanTyped(
    comptime K: type,
    comptime V: type,
    allocator: std.mem.Allocator,
    key_name: []const u8,
    output_name: []const u8,
    key: DeviceTypedColumn(K),
    value: DeviceTypedColumn(V),
    device_value: array_mod.Device,
) DeviceDataError!DeviceDataFrame {
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
    return initAggregatedDataFrame(allocator, key_name, key_col, output_name, mean_col, device_value);
}

fn groupByStatsDispatchKey(
    allocator: std.mem.Allocator,
    key_name: []const u8,
    output_prefix: []const u8,
    key: DeviceColumn,
    value: DeviceColumn,
    device_value: array_mod.Device,
) DeviceDataError!DeviceDataFrame {
    return switch (key) {
        .bool => |typed| groupByStatsDispatchValue(bool, allocator, key_name, output_prefix, typed, value, device_value),
        .i8 => |typed| groupByStatsDispatchValue(i8, allocator, key_name, output_prefix, typed, value, device_value),
        .i16 => |typed| groupByStatsDispatchValue(i16, allocator, key_name, output_prefix, typed, value, device_value),
        .i32 => |typed| groupByStatsDispatchValue(i32, allocator, key_name, output_prefix, typed, value, device_value),
        .i64 => |typed| groupByStatsDispatchValue(i64, allocator, key_name, output_prefix, typed, value, device_value),
        .u8 => |typed| groupByStatsDispatchValue(u8, allocator, key_name, output_prefix, typed, value, device_value),
        .u16 => |typed| groupByStatsDispatchValue(u16, allocator, key_name, output_prefix, typed, value, device_value),
        .u32 => |typed| groupByStatsDispatchValue(u32, allocator, key_name, output_prefix, typed, value, device_value),
        .u64 => |typed| groupByStatsDispatchValue(u64, allocator, key_name, output_prefix, typed, value, device_value),
        .usize => |typed| groupByStatsDispatchValue(usize, allocator, key_name, output_prefix, typed, value, device_value),
        .isize => |typed| groupByStatsDispatchValue(isize, allocator, key_name, output_prefix, typed, value, device_value),
        .f16 => |typed| groupByStatsDispatchValue(f16, allocator, key_name, output_prefix, typed, value, device_value),
        .f32 => |typed| groupByStatsDispatchValue(f32, allocator, key_name, output_prefix, typed, value, device_value),
        .f64 => |typed| groupByStatsDispatchValue(f64, allocator, key_name, output_prefix, typed, value, device_value),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn groupByStatsDispatchValue(
    comptime K: type,
    allocator: std.mem.Allocator,
    key_name: []const u8,
    output_prefix: []const u8,
    key: DeviceTypedColumn(K),
    value: DeviceColumn,
    device_value: array_mod.Device,
) DeviceDataError!DeviceDataFrame {
    return switch (value) {
        .i8 => |typed| groupByStatsTyped(K, i8, allocator, key_name, output_prefix, key, typed, device_value),
        .i16 => |typed| groupByStatsTyped(K, i16, allocator, key_name, output_prefix, key, typed, device_value),
        .i32 => |typed| groupByStatsTyped(K, i32, allocator, key_name, output_prefix, key, typed, device_value),
        .i64 => |typed| groupByStatsTyped(K, i64, allocator, key_name, output_prefix, key, typed, device_value),
        .u8 => |typed| groupByStatsTyped(K, u8, allocator, key_name, output_prefix, key, typed, device_value),
        .u16 => |typed| groupByStatsTyped(K, u16, allocator, key_name, output_prefix, key, typed, device_value),
        .u32 => |typed| groupByStatsTyped(K, u32, allocator, key_name, output_prefix, key, typed, device_value),
        .u64 => |typed| groupByStatsTyped(K, u64, allocator, key_name, output_prefix, key, typed, device_value),
        .usize => |typed| groupByStatsTyped(K, usize, allocator, key_name, output_prefix, key, typed, device_value),
        .isize => |typed| groupByStatsTyped(K, isize, allocator, key_name, output_prefix, key, typed, device_value),
        .f16 => |typed| groupByStatsTyped(K, f16, allocator, key_name, output_prefix, key, typed, device_value),
        .f32 => |typed| groupByStatsTyped(K, f32, allocator, key_name, output_prefix, key, typed, device_value),
        .f64 => |typed| groupByStatsTyped(K, f64, allocator, key_name, output_prefix, key, typed, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn groupByStatsTyped(
    comptime K: type,
    comptime V: type,
    allocator: std.mem.Allocator,
    key_name: []const u8,
    output_prefix: []const u8,
    key: DeviceTypedColumn(K),
    value: DeviceTypedColumn(V),
    device_value: array_mod.Device,
) DeviceDataError!DeviceDataFrame {
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

    var key_col = try DeviceColumn.fromSlice(K, allocator, unique_keys.items, device_value);
    errdefer key_col.deinit();
    var count_col = try DeviceColumn.fromSlice(i64, allocator, counts.items, device_value);
    errdefer count_col.deinit();
    var sum_col = try DeviceColumn.fromSlice(V, allocator, sums.items, device_value);
    errdefer sum_col.deinit();
    var min_col = try DeviceColumn.fromSlice(V, allocator, mins.items, device_value);
    errdefer min_col.deinit();
    var max_col = try DeviceColumn.fromSlice(V, allocator, maxes.items, device_value);
    errdefer max_col.deinit();
    var mean_col = try DeviceColumn.fromSlice(f64, allocator, means, device_value);
    errdefer mean_col.deinit();

    const names = try statsOutputNames(allocator, key_name, output_prefix);
    defer freeStatsOutputNames(allocator, names);
    const columns = try allocator.alloc(DeviceColumn, 6);
    errdefer allocator.free(columns);
    columns[0] = key_col;
    columns[1] = count_col;
    columns[2] = sum_col;
    columns[3] = min_col;
    columns[4] = max_col;
    columns[5] = mean_col;
    return initDeviceDataFrameFromOwnedColumns(allocator, names, columns, unique_keys.items.len, device_value);
}

fn statsOutputNames(allocator: std.mem.Allocator, key_name: []const u8, prefix: []const u8) std.mem.Allocator.Error![]const []const u8 {
    const names = try allocator.alloc([]const u8, 6);
    errdefer allocator.free(names);
    names[0] = key_name;
    names[1] = try std.fmt.allocPrint(allocator, "{s}_count", .{prefix});
    errdefer allocator.free(names[1]);
    names[2] = try std.fmt.allocPrint(allocator, "{s}_sum", .{prefix});
    errdefer allocator.free(names[2]);
    names[3] = try std.fmt.allocPrint(allocator, "{s}_min", .{prefix});
    errdefer allocator.free(names[3]);
    names[4] = try std.fmt.allocPrint(allocator, "{s}_max", .{prefix});
    errdefer allocator.free(names[4]);
    names[5] = try std.fmt.allocPrint(allocator, "{s}_mean", .{prefix});
    return names;
}

fn freeStatsOutputNames(allocator: std.mem.Allocator, names: []const []const u8) void {
    for (names[1..]) |name| allocator.free(name);
    allocator.free(names);
}

fn groupByStatsOnDispatchValue(
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_prefix: []const u8,
    value: DeviceColumn,
    device_value: array_mod.Device,
) DeviceDataError!DeviceDataFrame {
    return switch (value) {
        .i8 => |typed| groupByStatsOnTyped(i8, allocator, frame, key_names, output_prefix, typed, device_value),
        .i16 => |typed| groupByStatsOnTyped(i16, allocator, frame, key_names, output_prefix, typed, device_value),
        .i32 => |typed| groupByStatsOnTyped(i32, allocator, frame, key_names, output_prefix, typed, device_value),
        .i64 => |typed| groupByStatsOnTyped(i64, allocator, frame, key_names, output_prefix, typed, device_value),
        .u8 => |typed| groupByStatsOnTyped(u8, allocator, frame, key_names, output_prefix, typed, device_value),
        .u16 => |typed| groupByStatsOnTyped(u16, allocator, frame, key_names, output_prefix, typed, device_value),
        .u32 => |typed| groupByStatsOnTyped(u32, allocator, frame, key_names, output_prefix, typed, device_value),
        .u64 => |typed| groupByStatsOnTyped(u64, allocator, frame, key_names, output_prefix, typed, device_value),
        .usize => |typed| groupByStatsOnTyped(usize, allocator, frame, key_names, output_prefix, typed, device_value),
        .isize => |typed| groupByStatsOnTyped(isize, allocator, frame, key_names, output_prefix, typed, device_value),
        .f16 => |typed| groupByStatsOnTyped(f16, allocator, frame, key_names, output_prefix, typed, device_value),
        .f32 => |typed| groupByStatsOnTyped(f32, allocator, frame, key_names, output_prefix, typed, device_value),
        .f64 => |typed| groupByStatsOnTyped(f64, allocator, frame, key_names, output_prefix, typed, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn groupByStatsOnTyped(
    comptime V: type,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_prefix: []const u8,
    value: DeviceTypedColumn(V),
    device_value: array_mod.Device,
) DeviceDataError!DeviceDataFrame {
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

    const output_names = try statsOutputNames(allocator, "", output_prefix);
    defer freeStatsOutputNames(allocator, output_names);
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
    return initDeviceDataFrameFromOwnedColumns(allocator, names, columns, representative_rows.items.len, device_value);
}

const GroupedMomentProfile = struct {
    count: i64 = 0,
    sum: f64 = 0,
    mean: f64 = 0,
    m2: f64 = 0,
    m3: f64 = 0,
    m4: f64 = 0,

    fn update(self: *GroupedMomentProfile, value: f64) void {
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

    fn variance(self: GroupedMomentProfile) f64 {
        if (self.count == 0) return std.math.nan(f64);
        return self.m2 / @as(f64, @floatFromInt(self.count));
    }

    fn stddev(self: GroupedMomentProfile) f64 {
        return std.math.sqrt(self.variance());
    }

    fn skewness(self: GroupedMomentProfile) f64 {
        if (self.count < 2 or self.m2 == 0) return std.math.nan(f64);
        const n: f64 = @floatFromInt(self.count);
        return std.math.sqrt(n) * self.m3 / std.math.pow(f64, self.m2, 1.5);
    }

    fn kurtosis(self: GroupedMomentProfile) f64 {
        if (self.count < 2 or self.m2 == 0) return std.math.nan(f64);
        const n: f64 = @floatFromInt(self.count);
        return n * self.m4 / (self.m2 * self.m2) - 3.0;
    }
};

const ProfileMetricSlices = struct {
    allocator: std.mem.Allocator,
    counts: []i64,
    sums: []f64,
    means: []f64,
    variances: []f64,
    stddevs: []f64,
    skewnesses: []f64,
    kurtoses: []f64,

    fn deinit(self: *ProfileMetricSlices) void {
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

fn materializeProfileMetrics(allocator: std.mem.Allocator, profiles: []const GroupedMomentProfile) std.mem.Allocator.Error!ProfileMetricSlices {
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

fn groupByProfileDispatchKey(
    allocator: std.mem.Allocator,
    key_name: []const u8,
    output_prefix: []const u8,
    key: DeviceColumn,
    value: DeviceColumn,
    device_value: array_mod.Device,
) DeviceDataError!DeviceDataFrame {
    return switch (key) {
        .bool => |typed| groupByProfileDispatchValue(bool, allocator, key_name, output_prefix, typed, value, device_value),
        .i8 => |typed| groupByProfileDispatchValue(i8, allocator, key_name, output_prefix, typed, value, device_value),
        .i16 => |typed| groupByProfileDispatchValue(i16, allocator, key_name, output_prefix, typed, value, device_value),
        .i32 => |typed| groupByProfileDispatchValue(i32, allocator, key_name, output_prefix, typed, value, device_value),
        .i64 => |typed| groupByProfileDispatchValue(i64, allocator, key_name, output_prefix, typed, value, device_value),
        .u8 => |typed| groupByProfileDispatchValue(u8, allocator, key_name, output_prefix, typed, value, device_value),
        .u16 => |typed| groupByProfileDispatchValue(u16, allocator, key_name, output_prefix, typed, value, device_value),
        .u32 => |typed| groupByProfileDispatchValue(u32, allocator, key_name, output_prefix, typed, value, device_value),
        .u64 => |typed| groupByProfileDispatchValue(u64, allocator, key_name, output_prefix, typed, value, device_value),
        .usize => |typed| groupByProfileDispatchValue(usize, allocator, key_name, output_prefix, typed, value, device_value),
        .isize => |typed| groupByProfileDispatchValue(isize, allocator, key_name, output_prefix, typed, value, device_value),
        .f16 => |typed| groupByProfileDispatchValue(f16, allocator, key_name, output_prefix, typed, value, device_value),
        .f32 => |typed| groupByProfileDispatchValue(f32, allocator, key_name, output_prefix, typed, value, device_value),
        .f64 => |typed| groupByProfileDispatchValue(f64, allocator, key_name, output_prefix, typed, value, device_value),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn groupByProfileDispatchValue(
    comptime K: type,
    allocator: std.mem.Allocator,
    key_name: []const u8,
    output_prefix: []const u8,
    key: DeviceTypedColumn(K),
    value: DeviceColumn,
    device_value: array_mod.Device,
) DeviceDataError!DeviceDataFrame {
    return switch (value) {
        .i8 => |typed| groupByProfileTyped(K, i8, allocator, key_name, output_prefix, key, typed, device_value),
        .i16 => |typed| groupByProfileTyped(K, i16, allocator, key_name, output_prefix, key, typed, device_value),
        .i32 => |typed| groupByProfileTyped(K, i32, allocator, key_name, output_prefix, key, typed, device_value),
        .i64 => |typed| groupByProfileTyped(K, i64, allocator, key_name, output_prefix, key, typed, device_value),
        .u8 => |typed| groupByProfileTyped(K, u8, allocator, key_name, output_prefix, key, typed, device_value),
        .u16 => |typed| groupByProfileTyped(K, u16, allocator, key_name, output_prefix, key, typed, device_value),
        .u32 => |typed| groupByProfileTyped(K, u32, allocator, key_name, output_prefix, key, typed, device_value),
        .u64 => |typed| groupByProfileTyped(K, u64, allocator, key_name, output_prefix, key, typed, device_value),
        .usize => |typed| groupByProfileTyped(K, usize, allocator, key_name, output_prefix, key, typed, device_value),
        .isize => |typed| groupByProfileTyped(K, isize, allocator, key_name, output_prefix, key, typed, device_value),
        .f16 => |typed| groupByProfileTyped(K, f16, allocator, key_name, output_prefix, key, typed, device_value),
        .f32 => |typed| groupByProfileTyped(K, f32, allocator, key_name, output_prefix, key, typed, device_value),
        .f64 => |typed| groupByProfileTyped(K, f64, allocator, key_name, output_prefix, key, typed, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn groupByProfileTyped(
    comptime K: type,
    comptime V: type,
    allocator: std.mem.Allocator,
    key_name: []const u8,
    output_prefix: []const u8,
    key: DeviceTypedColumn(K),
    value: DeviceTypedColumn(V),
    device_value: array_mod.Device,
) DeviceDataError!DeviceDataFrame {
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
    var profiles: std.ArrayList(GroupedMomentProfile) = .empty;
    defer profiles.deinit(allocator);

    // Keep all moment-derived metrics in one pass over each group.  Besides
    // being cheaper than issuing many independent group-bys, this preserves one
    // API seam for a future Axiom grouped-moment kernel on CPU/CUDA/MPS.
    for (keys, values, 0..) |key_value, value_item, row| {
        if (maybe_key_validity) |validity| {
            if (!validity[row]) continue;
        }
        if (maybe_value_validity) |validity| {
            if (!validity[row]) continue;
        }
        const group_index = findGroupIndex(K, unique_keys.items, key_value) orelse blk: {
            try unique_keys.append(allocator, key_value);
            try profiles.append(allocator, .{});
            break :blk unique_keys.items.len - 1;
        };
        profiles.items[group_index].update(castToF64(V, value_item));
    }

    var metrics = try materializeProfileMetrics(allocator, profiles.items);
    defer metrics.deinit();
    var key_col = try DeviceColumn.fromSlice(K, allocator, unique_keys.items, device_value);
    defer key_col.deinit();
    return initProfileDataFrame(allocator, &.{key_name}, output_prefix, &.{key_col}, metrics, device_value);
}

fn groupByProfileOnDispatchValue(
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_prefix: []const u8,
    value: DeviceColumn,
    device_value: array_mod.Device,
) DeviceDataError!DeviceDataFrame {
    return switch (value) {
        .i8 => |typed| groupByProfileOnTyped(i8, allocator, frame, key_names, output_prefix, typed, device_value),
        .i16 => |typed| groupByProfileOnTyped(i16, allocator, frame, key_names, output_prefix, typed, device_value),
        .i32 => |typed| groupByProfileOnTyped(i32, allocator, frame, key_names, output_prefix, typed, device_value),
        .i64 => |typed| groupByProfileOnTyped(i64, allocator, frame, key_names, output_prefix, typed, device_value),
        .u8 => |typed| groupByProfileOnTyped(u8, allocator, frame, key_names, output_prefix, typed, device_value),
        .u16 => |typed| groupByProfileOnTyped(u16, allocator, frame, key_names, output_prefix, typed, device_value),
        .u32 => |typed| groupByProfileOnTyped(u32, allocator, frame, key_names, output_prefix, typed, device_value),
        .u64 => |typed| groupByProfileOnTyped(u64, allocator, frame, key_names, output_prefix, typed, device_value),
        .usize => |typed| groupByProfileOnTyped(usize, allocator, frame, key_names, output_prefix, typed, device_value),
        .isize => |typed| groupByProfileOnTyped(isize, allocator, frame, key_names, output_prefix, typed, device_value),
        .f16 => |typed| groupByProfileOnTyped(f16, allocator, frame, key_names, output_prefix, typed, device_value),
        .f32 => |typed| groupByProfileOnTyped(f32, allocator, frame, key_names, output_prefix, typed, device_value),
        .f64 => |typed| groupByProfileOnTyped(f64, allocator, frame, key_names, output_prefix, typed, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn groupByProfileOnTyped(
    comptime V: type,
    allocator: std.mem.Allocator,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
    output_prefix: []const u8,
    value: DeviceTypedColumn(V),
    device_value: array_mod.Device,
) DeviceDataError!DeviceDataFrame {
    if (frame.rows != value.len()) return error.LengthMismatch;
    const values = try value.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_value_validity = try validityValues(value, allocator);
    defer if (maybe_value_validity) |validity| allocator.free(validity);

    var representative_rows: std.ArrayList(usize) = .empty;
    defer representative_rows.deinit(allocator);
    var profiles: std.ArrayList(GroupedMomentProfile) = .empty;
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

    var metrics = try materializeProfileMetrics(allocator, profiles.items);
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

    return initProfileDataFrame(allocator, key_names, output_prefix, key_columns, metrics, device_value);
}

fn profileOutputNames(allocator: std.mem.Allocator, key_names: []const []const u8, prefix: []const u8) std.mem.Allocator.Error![]const []const u8 {
    const names = try allocator.alloc([]const u8, key_names.len + 7);
    errdefer allocator.free(names);
    for (key_names, 0..) |key_name, i| names[i] = key_name;
    var initialized: usize = 0;
    errdefer {
        for (names[key_names.len .. key_names.len + initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "count", "sum", "mean", "variance", "stddev", "skewness", "kurtosis" };
    for (suffixes, 0..) |suffix, i| {
        names[key_names.len + i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn freeProfileOutputNames(allocator: std.mem.Allocator, names: []const []const u8, key_count: usize) void {
    for (names[key_count..]) |name| allocator.free(name);
    allocator.free(names);
}

fn initProfileDataFrame(
    allocator: std.mem.Allocator,
    key_names: []const []const u8,
    output_prefix: []const u8,
    key_columns: []const DeviceColumn,
    metrics: ProfileMetricSlices,
    device_value: array_mod.Device,
) DeviceDataError!DeviceDataFrame {
    if (key_columns.len != key_names.len) return error.LengthMismatch;
    const rows = metrics.counts.len;
    const names = try profileOutputNames(allocator, key_names, output_prefix);
    defer freeProfileOutputNames(allocator, names, key_names.len);

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

    return initDeviceDataFrameFromOwnedColumns(allocator, names, columns, rows, device_value);
}

fn distinctRowIndices(allocator: std.mem.Allocator, frame: DeviceDataFrame, key_names: []const []const u8) DeviceDataError![]usize {
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |name| _ = try frame.column(name);

    var representatives: std.ArrayList(usize) = .empty;
    errdefer representatives.deinit(allocator);

    // Preserve first-seen row order, matching the common stable
    // `drop_duplicates(keep=first)` dataframe behavior.  The current
    // implementation deliberately routes through the same row-comparison helper
    // used by multi-key joins/grouping so null-key rows are skipped and future
    // Axiom hash-distinct lowering has a single API seam to replace.
    for (0..frame.rows) |row| {
        if (!try rowHasValidKeys(allocator, frame, key_names, row)) continue;
        const maybe_seen = try findMultiKeyGroupIndex(allocator, frame, key_names, representatives.items, row);
        if (maybe_seen == null) try representatives.append(allocator, row);
    }

    return representatives.toOwnedSlice(allocator);
}

fn rowHasValidKeys(allocator: std.mem.Allocator, frame: DeviceDataFrame, key_names: []const []const u8, row: usize) DeviceDataError!bool {
    for (key_names) |key_name| {
        const key = try frame.column(key_name);
        if (!try columnRowValid(allocator, key.*, row)) return false;
    }
    return true;
}

fn columnRowValid(allocator: std.mem.Allocator, column: DeviceColumn, row: usize) DeviceDataError!bool {
    return switch (column) {
        inline else => |typed| blk: {
            if (row >= typed.len()) return error.IndexOutOfBounds;
            const maybe_validity = try validityValues(typed, allocator);
            defer if (maybe_validity) |validity| allocator.free(validity);
            break :blk if (maybe_validity) |validity| validity[row] else true;
        },
    };
}

fn findMultiKeyGroupIndex(allocator: std.mem.Allocator, frame: DeviceDataFrame, key_names: []const []const u8, representatives: []const usize, row: usize) DeviceDataError!?usize {
    for (representatives, 0..) |representative, i| {
        if (try rowsMatchAllKeys(allocator, frame, frame, key_names, key_names, representative, row)) return i;
    }
    return null;
}

fn initAggregatedDataFrame(
    allocator: std.mem.Allocator,
    key_name: []const u8,
    key_col: DeviceColumn,
    output_name: []const u8,
    value_col: DeviceColumn,
    device_value: array_mod.Device,
) DeviceDataError!DeviceDataFrame {
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
    return initDeviceDataFrameFromOwnedColumns(allocator, &names, columns, rows, device_value);
}

fn findGroupIndex(comptime T: type, keys: []const T, value: T) ?usize {
    for (keys, 0..) |candidate, i| {
        if (groupKeyEqual(T, candidate, value)) return i;
    }
    return null;
}

fn groupKeyEqual(comptime T: type, lhs: T, rhs: T) bool {
    if (comptime @typeInfo(T) == .float) {
        const lhs_nan = std.math.isNan(lhs);
        const rhs_nan = std.math.isNan(rhs);
        return if (lhs_nan or rhs_nan) lhs_nan and rhs_nan else lhs == rhs;
    }
    return lhs == rhs;
}

fn castToF64(comptime T: type, value: T) f64 {
    return switch (@typeInfo(T)) {
        .float, .comptime_float => @floatCast(value),
        .int, .comptime_int => @floatFromInt(value),
        else => @compileError("mean requires numeric values"),
    };
}

const JoinRowIndexPair = struct {
    allocator: std.mem.Allocator,
    left: []?usize,
    right: []?usize,

    fn deinit(self: *JoinRowIndexPair) void {
        self.allocator.free(self.left);
        self.allocator.free(self.right);
        self.* = undefined;
    }
};

fn innerJoinRowIndices(allocator: std.mem.Allocator, left: DeviceColumn, right: DeviceColumn) DeviceDataError!JoinRowIndexPair {
    return switch (left) {
        .bool => |typed| innerJoinRowIndicesTyped(bool, allocator, typed, right.bool),
        .i8 => |typed| innerJoinRowIndicesTyped(i8, allocator, typed, right.i8),
        .i16 => |typed| innerJoinRowIndicesTyped(i16, allocator, typed, right.i16),
        .i32 => |typed| innerJoinRowIndicesTyped(i32, allocator, typed, right.i32),
        .i64 => |typed| innerJoinRowIndicesTyped(i64, allocator, typed, right.i64),
        .u8 => |typed| innerJoinRowIndicesTyped(u8, allocator, typed, right.u8),
        .u16 => |typed| innerJoinRowIndicesTyped(u16, allocator, typed, right.u16),
        .u32 => |typed| innerJoinRowIndicesTyped(u32, allocator, typed, right.u32),
        .u64 => |typed| innerJoinRowIndicesTyped(u64, allocator, typed, right.u64),
        .usize => |typed| innerJoinRowIndicesTyped(usize, allocator, typed, right.usize),
        .isize => |typed| innerJoinRowIndicesTyped(isize, allocator, typed, right.isize),
        .f16 => |typed| innerJoinRowIndicesTyped(f16, allocator, typed, right.f16),
        .f32 => |typed| innerJoinRowIndicesTyped(f32, allocator, typed, right.f32),
        .f64 => |typed| innerJoinRowIndicesTyped(f64, allocator, typed, right.f64),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn innerJoinRowIndicesMulti(
    allocator: std.mem.Allocator,
    left: DeviceDataFrame,
    right: DeviceDataFrame,
    left_key_names: []const []const u8,
    right_key_names: []const []const u8,
) DeviceDataError!JoinRowIndexPair {
    var left_indices: std.ArrayList(?usize) = .empty;
    errdefer left_indices.deinit(allocator);
    var right_indices: std.ArrayList(?usize) = .empty;
    errdefer right_indices.deinit(allocator);

    // This is intentionally expressed as a row-pair builder, mirroring the
    // single-key join path and cuDF's hash-join output shape.  The current
    // implementation materializes key columns through `Array.toOwnedSlice()` in
    // `columnsRowsEqual`; the API boundary is what future Axiom lowering will
    // replace with a multi-key hash table/probe kernel.
    for (0..left.rows) |left_i| {
        for (0..right.rows) |right_i| {
            if (try rowsMatchAllKeys(allocator, left, right, left_key_names, right_key_names, left_i, right_i)) {
                try left_indices.append(allocator, left_i);
                try right_indices.append(allocator, right_i);
            }
        }
    }

    const owned_left = try left_indices.toOwnedSlice(allocator);
    left_indices = .empty;
    errdefer allocator.free(owned_left);
    const owned_right = try right_indices.toOwnedSlice(allocator);
    right_indices = .empty;
    return .{
        .allocator = allocator,
        .left = owned_left,
        .right = owned_right,
    };
}

fn leftJoinRowIndicesMulti(
    allocator: std.mem.Allocator,
    left: DeviceDataFrame,
    right: DeviceDataFrame,
    left_key_names: []const []const u8,
    right_key_names: []const []const u8,
) DeviceDataError!JoinRowIndexPair {
    var left_indices: std.ArrayList(?usize) = .empty;
    errdefer left_indices.deinit(allocator);
    var right_indices: std.ArrayList(?usize) = .empty;
    errdefer right_indices.deinit(allocator);

    for (0..left.rows) |left_i| {
        var matched = false;
        for (0..right.rows) |right_i| {
            if (try rowsMatchAllKeys(allocator, left, right, left_key_names, right_key_names, left_i, right_i)) {
                try left_indices.append(allocator, left_i);
                try right_indices.append(allocator, right_i);
                matched = true;
            }
        }
        if (!matched) {
            try left_indices.append(allocator, left_i);
            try right_indices.append(allocator, null);
        }
    }

    const owned_left = try left_indices.toOwnedSlice(allocator);
    left_indices = .empty;
    errdefer allocator.free(owned_left);
    const owned_right = try right_indices.toOwnedSlice(allocator);
    right_indices = .empty;
    return .{
        .allocator = allocator,
        .left = owned_left,
        .right = owned_right,
    };
}

fn fullJoinRowIndicesMulti(
    allocator: std.mem.Allocator,
    left: DeviceDataFrame,
    right: DeviceDataFrame,
    left_key_names: []const []const u8,
    right_key_names: []const []const u8,
) DeviceDataError!JoinRowIndexPair {
    var left_indices: std.ArrayList(?usize) = .empty;
    errdefer left_indices.deinit(allocator);
    var right_indices: std.ArrayList(?usize) = .empty;
    errdefer right_indices.deinit(allocator);
    const right_matched = try allocator.alloc(bool, right.rows);
    defer allocator.free(right_matched);
    @memset(right_matched, false);

    for (0..left.rows) |left_i| {
        var matched = false;
        for (0..right.rows) |right_i| {
            if (try rowsMatchAllKeys(allocator, left, right, left_key_names, right_key_names, left_i, right_i)) {
                try left_indices.append(allocator, left_i);
                try right_indices.append(allocator, right_i);
                right_matched[right_i] = true;
                matched = true;
            }
        }
        if (!matched) {
            try left_indices.append(allocator, left_i);
            try right_indices.append(allocator, null);
        }
    }

    for (0..right.rows) |right_i| {
        if (!right_matched[right_i]) {
            try left_indices.append(allocator, null);
            try right_indices.append(allocator, right_i);
        }
    }

    const owned_left = try left_indices.toOwnedSlice(allocator);
    left_indices = .empty;
    errdefer allocator.free(owned_left);
    const owned_right = try right_indices.toOwnedSlice(allocator);
    right_indices = .empty;
    return .{
        .allocator = allocator,
        .left = owned_left,
        .right = owned_right,
    };
}

fn rowsMatchAllKeys(
    allocator: std.mem.Allocator,
    left: DeviceDataFrame,
    right: DeviceDataFrame,
    left_key_names: []const []const u8,
    right_key_names: []const []const u8,
    left_i: usize,
    right_i: usize,
) DeviceDataError!bool {
    for (left_key_names, right_key_names) |left_name, right_name| {
        const left_key = try left.column(left_name);
        const right_key = try right.column(right_name);
        if (!try columnsRowsEqual(allocator, left_key.*, right_key.*, left_i, right_i)) return false;
    }
    return true;
}

fn columnsRowsEqual(
    allocator: std.mem.Allocator,
    left: DeviceColumn,
    right: DeviceColumn,
    left_i: usize,
    right_i: usize,
) DeviceDataError!bool {
    if (left.dtype() != right.dtype()) return error.TypeMismatch;
    return switch (left) {
        .bool => |typed| columnsRowsEqualTyped(bool, allocator, typed, right.bool, left_i, right_i),
        .i8 => |typed| columnsRowsEqualTyped(i8, allocator, typed, right.i8, left_i, right_i),
        .i16 => |typed| columnsRowsEqualTyped(i16, allocator, typed, right.i16, left_i, right_i),
        .i32 => |typed| columnsRowsEqualTyped(i32, allocator, typed, right.i32, left_i, right_i),
        .i64 => |typed| columnsRowsEqualTyped(i64, allocator, typed, right.i64, left_i, right_i),
        .u8 => |typed| columnsRowsEqualTyped(u8, allocator, typed, right.u8, left_i, right_i),
        .u16 => |typed| columnsRowsEqualTyped(u16, allocator, typed, right.u16, left_i, right_i),
        .u32 => |typed| columnsRowsEqualTyped(u32, allocator, typed, right.u32, left_i, right_i),
        .u64 => |typed| columnsRowsEqualTyped(u64, allocator, typed, right.u64, left_i, right_i),
        .usize => |typed| columnsRowsEqualTyped(usize, allocator, typed, right.usize, left_i, right_i),
        .isize => |typed| columnsRowsEqualTyped(isize, allocator, typed, right.isize, left_i, right_i),
        .f16 => |typed| columnsRowsEqualTyped(f16, allocator, typed, right.f16, left_i, right_i),
        .f32 => |typed| columnsRowsEqualTyped(f32, allocator, typed, right.f32, left_i, right_i),
        .f64 => |typed| columnsRowsEqualTyped(f64, allocator, typed, right.f64, left_i, right_i),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn columnsRowsEqualTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    left: DeviceTypedColumn(T),
    right: DeviceTypedColumn(T),
    left_i: usize,
    right_i: usize,
) DeviceDataError!bool {
    if (!left.device().sameDevice(right.device())) return error.InvalidDevice;
    if (left_i >= left.len() or right_i >= right.len()) return error.IndexOutOfBounds;
    const left_validity = try validityValues(left, allocator);
    defer if (left_validity) |validity| allocator.free(validity);
    const right_validity = try validityValues(right, allocator);
    defer if (right_validity) |validity| allocator.free(validity);
    if (left_validity) |validity| {
        if (!validity[left_i]) return false;
    }
    if (right_validity) |validity| {
        if (!validity[right_i]) return false;
    }
    const left_values = try left.values.toOwnedSlice(allocator);
    defer allocator.free(left_values);
    const right_values = try right.values.toOwnedSlice(allocator);
    defer allocator.free(right_values);
    return groupKeyEqual(T, left_values[left_i], right_values[right_i]);
}

fn asofRightRowIndices(allocator: std.mem.Allocator, left: DeviceColumn, right: DeviceColumn, strategy: AsofStrategy) DeviceDataError![]?usize {
    return switch (left) {
        .i8 => |typed| asofRightRowIndicesTyped(i8, allocator, typed, right.i8, strategy),
        .i16 => |typed| asofRightRowIndicesTyped(i16, allocator, typed, right.i16, strategy),
        .i32 => |typed| asofRightRowIndicesTyped(i32, allocator, typed, right.i32, strategy),
        .i64 => |typed| asofRightRowIndicesTyped(i64, allocator, typed, right.i64, strategy),
        .u8 => |typed| asofRightRowIndicesTyped(u8, allocator, typed, right.u8, strategy),
        .u16 => |typed| asofRightRowIndicesTyped(u16, allocator, typed, right.u16, strategy),
        .u32 => |typed| asofRightRowIndicesTyped(u32, allocator, typed, right.u32, strategy),
        .u64 => |typed| asofRightRowIndicesTyped(u64, allocator, typed, right.u64, strategy),
        .usize => |typed| asofRightRowIndicesTyped(usize, allocator, typed, right.usize, strategy),
        .isize => |typed| asofRightRowIndicesTyped(isize, allocator, typed, right.isize, strategy),
        .f16 => |typed| asofRightRowIndicesTyped(f16, allocator, typed, right.f16, strategy),
        .f32 => |typed| asofRightRowIndicesTyped(f32, allocator, typed, right.f32, strategy),
        .f64 => |typed| asofRightRowIndicesTyped(f64, allocator, typed, right.f64, strategy),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn asofRightRowIndicesTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    left: DeviceTypedColumn(T),
    right: DeviceTypedColumn(T),
    strategy: AsofStrategy,
) DeviceDataError![]?usize {
    if (!left.device().sameDevice(right.device())) return error.InvalidDevice;
    const left_values = try left.values.toOwnedSlice(allocator);
    defer allocator.free(left_values);
    const right_values = try right.values.toOwnedSlice(allocator);
    defer allocator.free(right_values);
    const maybe_left_validity = try validityValues(left, allocator);
    defer if (maybe_left_validity) |validity| allocator.free(validity);
    const maybe_right_validity = try validityValues(right, allocator);
    defer if (maybe_right_validity) |validity| allocator.free(validity);

    const indices = try allocator.alloc(?usize, left_values.len);
    for (left_values, indices, 0..) |left_value, *slot, left_i| {
        slot.* = null;
        if (maybe_left_validity) |validity| {
            if (!validity[left_i]) continue;
        }
        var best: ?usize = null;
        for (right_values, 0..) |right_value, right_i| {
            if (maybe_right_validity) |validity| {
                if (!validity[right_i]) continue;
            }
            switch (strategy) {
                .previous => {
                    if (compareSortValues(T, right_value, left_value) <= 0 and (best == null or compareSortValues(T, right_value, right_values[best.?]) > 0)) best = right_i;
                },
                .next => {
                    if (compareSortValues(T, right_value, left_value) >= 0 and (best == null or compareSortValues(T, right_value, right_values[best.?]) < 0)) best = right_i;
                },
                .nearest => {
                    if (best == null or asofDistance(T, left_value, right_value) < asofDistance(T, left_value, right_values[best.?])) best = right_i;
                },
            }
        }
        slot.* = best;
    }
    return indices;
}

fn asofDistance(comptime T: type, lhs: T, rhs: T) f64 {
    return @abs(castToF64(T, lhs) - castToF64(T, rhs));
}

fn leftJoinRowIndices(allocator: std.mem.Allocator, left: DeviceColumn, right: DeviceColumn) DeviceDataError!JoinRowIndexPair {
    return switch (left) {
        .bool => |typed| leftJoinRowIndicesTyped(bool, allocator, typed, right.bool),
        .i8 => |typed| leftJoinRowIndicesTyped(i8, allocator, typed, right.i8),
        .i16 => |typed| leftJoinRowIndicesTyped(i16, allocator, typed, right.i16),
        .i32 => |typed| leftJoinRowIndicesTyped(i32, allocator, typed, right.i32),
        .i64 => |typed| leftJoinRowIndicesTyped(i64, allocator, typed, right.i64),
        .u8 => |typed| leftJoinRowIndicesTyped(u8, allocator, typed, right.u8),
        .u16 => |typed| leftJoinRowIndicesTyped(u16, allocator, typed, right.u16),
        .u32 => |typed| leftJoinRowIndicesTyped(u32, allocator, typed, right.u32),
        .u64 => |typed| leftJoinRowIndicesTyped(u64, allocator, typed, right.u64),
        .usize => |typed| leftJoinRowIndicesTyped(usize, allocator, typed, right.usize),
        .isize => |typed| leftJoinRowIndicesTyped(isize, allocator, typed, right.isize),
        .f16 => |typed| leftJoinRowIndicesTyped(f16, allocator, typed, right.f16),
        .f32 => |typed| leftJoinRowIndicesTyped(f32, allocator, typed, right.f32),
        .f64 => |typed| leftJoinRowIndicesTyped(f64, allocator, typed, right.f64),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn fullJoinRowIndices(allocator: std.mem.Allocator, left: DeviceColumn, right: DeviceColumn) DeviceDataError!JoinRowIndexPair {
    return switch (left) {
        .bool => |typed| fullJoinRowIndicesTyped(bool, allocator, typed, right.bool),
        .i8 => |typed| fullJoinRowIndicesTyped(i8, allocator, typed, right.i8),
        .i16 => |typed| fullJoinRowIndicesTyped(i16, allocator, typed, right.i16),
        .i32 => |typed| fullJoinRowIndicesTyped(i32, allocator, typed, right.i32),
        .i64 => |typed| fullJoinRowIndicesTyped(i64, allocator, typed, right.i64),
        .u8 => |typed| fullJoinRowIndicesTyped(u8, allocator, typed, right.u8),
        .u16 => |typed| fullJoinRowIndicesTyped(u16, allocator, typed, right.u16),
        .u32 => |typed| fullJoinRowIndicesTyped(u32, allocator, typed, right.u32),
        .u64 => |typed| fullJoinRowIndicesTyped(u64, allocator, typed, right.u64),
        .usize => |typed| fullJoinRowIndicesTyped(usize, allocator, typed, right.usize),
        .isize => |typed| fullJoinRowIndicesTyped(isize, allocator, typed, right.isize),
        .f16 => |typed| fullJoinRowIndicesTyped(f16, allocator, typed, right.f16),
        .f32 => |typed| fullJoinRowIndicesTyped(f32, allocator, typed, right.f32),
        .f64 => |typed| fullJoinRowIndicesTyped(f64, allocator, typed, right.f64),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn semiAntiJoinRowIndices(allocator: std.mem.Allocator, left: DeviceColumn, right: DeviceColumn, keep_matches: bool) DeviceDataError![]usize {
    return switch (left) {
        .bool => |typed| semiAntiJoinRowIndicesTyped(bool, allocator, typed, right.bool, keep_matches),
        .i8 => |typed| semiAntiJoinRowIndicesTyped(i8, allocator, typed, right.i8, keep_matches),
        .i16 => |typed| semiAntiJoinRowIndicesTyped(i16, allocator, typed, right.i16, keep_matches),
        .i32 => |typed| semiAntiJoinRowIndicesTyped(i32, allocator, typed, right.i32, keep_matches),
        .i64 => |typed| semiAntiJoinRowIndicesTyped(i64, allocator, typed, right.i64, keep_matches),
        .u8 => |typed| semiAntiJoinRowIndicesTyped(u8, allocator, typed, right.u8, keep_matches),
        .u16 => |typed| semiAntiJoinRowIndicesTyped(u16, allocator, typed, right.u16, keep_matches),
        .u32 => |typed| semiAntiJoinRowIndicesTyped(u32, allocator, typed, right.u32, keep_matches),
        .u64 => |typed| semiAntiJoinRowIndicesTyped(u64, allocator, typed, right.u64, keep_matches),
        .usize => |typed| semiAntiJoinRowIndicesTyped(usize, allocator, typed, right.usize, keep_matches),
        .isize => |typed| semiAntiJoinRowIndicesTyped(isize, allocator, typed, right.isize, keep_matches),
        .f16 => |typed| semiAntiJoinRowIndicesTyped(f16, allocator, typed, right.f16, keep_matches),
        .f32 => |typed| semiAntiJoinRowIndicesTyped(f32, allocator, typed, right.f32, keep_matches),
        .f64 => |typed| semiAntiJoinRowIndicesTyped(f64, allocator, typed, right.f64, keep_matches),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn semiAntiJoinRowIndicesMulti(
    allocator: std.mem.Allocator,
    left: DeviceDataFrame,
    right: DeviceDataFrame,
    left_key_names: []const []const u8,
    right_key_names: []const []const u8,
    keep_matches: bool,
) DeviceDataError![]usize {
    var indices: std.ArrayList(usize) = .empty;
    errdefer indices.deinit(allocator);

    for (0..left.rows) |left_i| {
        var matched = false;
        for (0..right.rows) |right_i| {
            if (try rowsMatchAllKeys(allocator, left, right, left_key_names, right_key_names, left_i, right_i)) {
                matched = true;
                break;
            }
        }
        if (matched == keep_matches) try indices.append(allocator, left_i);
    }

    return indices.toOwnedSlice(allocator);
}

fn innerJoinRowIndicesTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    left: DeviceTypedColumn(T),
    right: DeviceTypedColumn(T),
) DeviceDataError!JoinRowIndexPair {
    if (!left.device().sameDevice(right.device())) return error.InvalidDevice;
    const left_values = try left.values.toOwnedSlice(allocator);
    defer allocator.free(left_values);
    const right_values = try right.values.toOwnedSlice(allocator);
    defer allocator.free(right_values);
    const maybe_left_validity = try validityValues(left, allocator);
    defer if (maybe_left_validity) |validity| allocator.free(validity);
    const maybe_right_validity = try validityValues(right, allocator);
    defer if (maybe_right_validity) |validity| allocator.free(validity);

    var left_indices: std.ArrayList(?usize) = .empty;
    errdefer left_indices.deinit(allocator);
    var right_indices: std.ArrayList(?usize) = .empty;
    errdefer right_indices.deinit(allocator);

    for (left_values, 0..) |left_value, left_i| {
        if (maybe_left_validity) |validity| {
            if (!validity[left_i]) continue;
        }
        for (right_values, 0..) |right_value, right_i| {
            if (maybe_right_validity) |validity| {
                if (!validity[right_i]) continue;
            }
            if (groupKeyEqual(T, left_value, right_value)) {
                try left_indices.append(allocator, left_i);
                try right_indices.append(allocator, right_i);
            }
        }
    }

    const owned_left = try left_indices.toOwnedSlice(allocator);
    left_indices = .empty;
    errdefer allocator.free(owned_left);
    const owned_right = try right_indices.toOwnedSlice(allocator);
    right_indices = .empty;
    return .{
        .allocator = allocator,
        .left = owned_left,
        .right = owned_right,
    };
}

fn semiAntiJoinRowIndicesTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    left: DeviceTypedColumn(T),
    right: DeviceTypedColumn(T),
    keep_matches: bool,
) DeviceDataError![]usize {
    if (!left.device().sameDevice(right.device())) return error.InvalidDevice;
    const left_values = try left.values.toOwnedSlice(allocator);
    defer allocator.free(left_values);
    const right_values = try right.values.toOwnedSlice(allocator);
    defer allocator.free(right_values);
    const maybe_left_validity = try validityValues(left, allocator);
    defer if (maybe_left_validity) |validity| allocator.free(validity);
    const maybe_right_validity = try validityValues(right, allocator);
    defer if (maybe_right_validity) |validity| allocator.free(validity);

    var indices: std.ArrayList(usize) = .empty;
    errdefer indices.deinit(allocator);
    for (left_values, 0..) |left_value, left_i| {
        const left_valid = if (maybe_left_validity) |validity| validity[left_i] else true;
        var matched = false;
        if (left_valid) {
            for (right_values, 0..) |right_value, right_i| {
                if (maybe_right_validity) |validity| {
                    if (!validity[right_i]) continue;
                }
                if (groupKeyEqual(T, left_value, right_value)) {
                    matched = true;
                    break;
                }
            }
        }
        if (matched == keep_matches) try indices.append(allocator, left_i);
    }
    return indices.toOwnedSlice(allocator);
}

fn leftJoinRowIndicesTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    left: DeviceTypedColumn(T),
    right: DeviceTypedColumn(T),
) DeviceDataError!JoinRowIndexPair {
    if (!left.device().sameDevice(right.device())) return error.InvalidDevice;
    const left_values = try left.values.toOwnedSlice(allocator);
    defer allocator.free(left_values);
    const right_values = try right.values.toOwnedSlice(allocator);
    defer allocator.free(right_values);
    const maybe_left_validity = try validityValues(left, allocator);
    defer if (maybe_left_validity) |validity| allocator.free(validity);
    const maybe_right_validity = try validityValues(right, allocator);
    defer if (maybe_right_validity) |validity| allocator.free(validity);

    var left_indices: std.ArrayList(?usize) = .empty;
    errdefer left_indices.deinit(allocator);
    var right_indices: std.ArrayList(?usize) = .empty;
    errdefer right_indices.deinit(allocator);

    for (left_values, 0..) |left_value, left_i| {
        var matched = false;
        const left_valid = if (maybe_left_validity) |validity| validity[left_i] else true;
        if (left_valid) {
            for (right_values, 0..) |right_value, right_i| {
                if (maybe_right_validity) |validity| {
                    if (!validity[right_i]) continue;
                }
                if (groupKeyEqual(T, left_value, right_value)) {
                    try left_indices.append(allocator, left_i);
                    try right_indices.append(allocator, right_i);
                    matched = true;
                }
            }
        }
        if (!matched) {
            try left_indices.append(allocator, left_i);
            try right_indices.append(allocator, null);
        }
    }

    const owned_left = try left_indices.toOwnedSlice(allocator);
    left_indices = .empty;
    errdefer allocator.free(owned_left);
    const owned_right = try right_indices.toOwnedSlice(allocator);
    right_indices = .empty;
    return .{
        .allocator = allocator,
        .left = owned_left,
        .right = owned_right,
    };
}

fn fullJoinRowIndicesTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    left: DeviceTypedColumn(T),
    right: DeviceTypedColumn(T),
) DeviceDataError!JoinRowIndexPair {
    if (!left.device().sameDevice(right.device())) return error.InvalidDevice;
    const left_values = try left.values.toOwnedSlice(allocator);
    defer allocator.free(left_values);
    const right_values = try right.values.toOwnedSlice(allocator);
    defer allocator.free(right_values);
    const maybe_left_validity = try validityValues(left, allocator);
    defer if (maybe_left_validity) |validity| allocator.free(validity);
    const maybe_right_validity = try validityValues(right, allocator);
    defer if (maybe_right_validity) |validity| allocator.free(validity);

    var left_indices: std.ArrayList(?usize) = .empty;
    errdefer left_indices.deinit(allocator);
    var right_indices: std.ArrayList(?usize) = .empty;
    errdefer right_indices.deinit(allocator);
    const right_matched = try allocator.alloc(bool, right_values.len);
    defer allocator.free(right_matched);
    @memset(right_matched, false);

    for (left_values, 0..) |left_value, left_i| {
        var matched = false;
        const left_valid = if (maybe_left_validity) |validity| validity[left_i] else true;
        if (left_valid) {
            for (right_values, 0..) |right_value, right_i| {
                if (maybe_right_validity) |validity| {
                    if (!validity[right_i]) continue;
                }
                if (groupKeyEqual(T, left_value, right_value)) {
                    try left_indices.append(allocator, left_i);
                    try right_indices.append(allocator, right_i);
                    right_matched[right_i] = true;
                    matched = true;
                }
            }
        }
        if (!matched) {
            try left_indices.append(allocator, left_i);
            try right_indices.append(allocator, null);
        }
    }

    for (right_values, 0..) |_, right_i| {
        if (maybe_right_validity) |validity| {
            if (!validity[right_i]) {
                try left_indices.append(allocator, null);
                try right_indices.append(allocator, right_i);
                continue;
            }
        }
        if (!right_matched[right_i]) {
            try left_indices.append(allocator, null);
            try right_indices.append(allocator, right_i);
        }
    }

    const owned_left = try left_indices.toOwnedSlice(allocator);
    left_indices = .empty;
    errdefer allocator.free(owned_left);
    const owned_right = try right_indices.toOwnedSlice(allocator);
    right_indices = .empty;
    return .{
        .allocator = allocator,
        .left = owned_left,
        .right = owned_right,
    };
}

fn takeOptionalRows(input: DeviceDataFrame, row_indices: []const ?usize) DeviceDataError!DeviceDataFrame {
    var columns = try input.allocator.alloc(DeviceColumn, input.columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        input.allocator.free(columns);
    }
    for (input.columns, 0..) |col, i| {
        columns[i] = try col.takeOptional(row_indices);
        initialized += 1;
    }
    return initDeviceDataFrameFromOwnedColumns(input.allocator, input.names, columns, row_indices.len, input.device);
}

fn concatJoinedTables(
    allocator: std.mem.Allocator,
    left: DeviceDataFrame,
    right: DeviceDataFrame,
    right_key_name: []const u8,
    options_value: DeviceJoinOptions,
) DeviceDataError!DeviceDataFrame {
    return concatJoinedTablesExcludingKeys(allocator, left, right, &.{right_key_name}, options_value);
}

fn concatJoinedTablesExcludingKeys(
    allocator: std.mem.Allocator,
    left: DeviceDataFrame,
    right: DeviceDataFrame,
    right_key_names: []const []const u8,
    options_value: DeviceJoinOptions,
) DeviceDataError!DeviceDataFrame {
    if (!left.device.sameDevice(right.device)) return error.InvalidDevice;
    if (left.rows != right.rows) return error.LengthMismatch;

    const total_cols = left.columns.len + right.columns.len - rightExcludedKeyCount(right, right_key_names);
    var names = try allocator.alloc([]const u8, total_cols);
    defer allocator.free(names);
    var temporary_names: std.ArrayList([]const u8) = .empty;
    defer {
        for (temporary_names.items) |name| allocator.free(name);
        temporary_names.deinit(allocator);
    }
    var columns = try allocator.alloc(DeviceColumn, total_cols);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        allocator.free(columns);
    }

    for (left.names, left.columns) |name, col| {
        names[initialized] = name;
        columns[initialized] = try col.clone();
        initialized += 1;
    }

    for (right.names, right.columns) |name, col| {
        if (nameInList(name, right_key_names)) continue;
        if (nameNeedsSuffix(left, name)) {
            const suffixed = try suffixedNameTemp(allocator, name, options_value.right_suffix);
            errdefer allocator.free(suffixed);
            try temporary_names.append(allocator, suffixed);
            names[initialized] = suffixed;
        } else {
            names[initialized] = name;
        }
        columns[initialized] = try col.clone();
        initialized += 1;
    }

    return initDeviceDataFrameFromOwnedColumns(allocator, names, columns, left.rows, left.device);
}

fn rightExcludedKeyCount(right: DeviceDataFrame, names: []const []const u8) usize {
    var count: usize = 0;
    for (right.names) |name| {
        if (nameInList(name, names)) count += 1;
    }
    return count;
}

fn nameInList(name: []const u8, names: []const []const u8) bool {
    for (names) |candidate| {
        if (std.mem.eql(u8, name, candidate)) return true;
    }
    return false;
}

fn concatFullJoinedTables(
    allocator: std.mem.Allocator,
    left: DeviceDataFrame,
    right: DeviceDataFrame,
    left_key_name: []const u8,
    right_key_name: []const u8,
    options_value: DeviceJoinOptions,
) DeviceDataError!DeviceDataFrame {
    return concatFullJoinedTablesOn(allocator, left, right, &.{left_key_name}, &.{right_key_name}, options_value);
}

fn concatFullJoinedTablesOn(
    allocator: std.mem.Allocator,
    left: DeviceDataFrame,
    right: DeviceDataFrame,
    left_key_names: []const []const u8,
    right_key_names: []const []const u8,
    options_value: DeviceJoinOptions,
) DeviceDataError!DeviceDataFrame {
    if (!left.device.sameDevice(right.device)) return error.InvalidDevice;
    if (left.rows != right.rows) return error.LengthMismatch;
    if (left_key_names.len == 0 or left_key_names.len != right_key_names.len) return error.LengthMismatch;
    for (left_key_names, right_key_names) |left_name, right_name| {
        const left_key = try left.column(left_name);
        const right_key = try right.column(right_name);
        if (left_key.dtype() != right_key.dtype()) return error.TypeMismatch;
    }

    const total_cols = left.columns.len + right.columns.len - rightExcludedKeyCount(right, right_key_names);
    var names = try allocator.alloc([]const u8, total_cols);
    defer allocator.free(names);
    var temporary_names: std.ArrayList([]const u8) = .empty;
    defer {
        for (temporary_names.items) |name| allocator.free(name);
        temporary_names.deinit(allocator);
    }
    var columns = try allocator.alloc(DeviceColumn, total_cols);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        allocator.free(columns);
    }

    for (left.names, left.columns, 0..) |name, col, i| {
        names[initialized] = name;
        columns[initialized] = if (leftKeyRightIndex(left, right, left_key_names, right_key_names, i)) |right_key_index|
            try coalesceJoinKeys(col, right.columns[right_key_index])
        else
            try col.clone();
        initialized += 1;
    }

    for (right.names, right.columns, 0..) |name, col, i| {
        if (rightKeyIndexInList(right, right_key_names, i)) continue;
        if (nameNeedsSuffix(left, name)) {
            const suffixed = try suffixedNameTemp(allocator, name, options_value.right_suffix);
            errdefer allocator.free(suffixed);
            try temporary_names.append(allocator, suffixed);
            names[initialized] = suffixed;
        } else {
            names[initialized] = name;
        }
        columns[initialized] = try col.clone();
        initialized += 1;
    }

    return initDeviceDataFrameFromOwnedColumns(allocator, names, columns, left.rows, left.device);
}

fn leftKeyRightIndex(
    left: DeviceDataFrame,
    right: DeviceDataFrame,
    left_key_names: []const []const u8,
    right_key_names: []const []const u8,
    left_index: usize,
) ?usize {
    for (left_key_names, right_key_names) |left_name, right_name| {
        const candidate = left.columnIndex(left_name) orelse continue;
        if (candidate == left_index) return right.columnIndex(right_name);
    }
    return null;
}

fn rightKeyIndexInList(right: DeviceDataFrame, right_key_names: []const []const u8, right_index: usize) bool {
    for (right_key_names) |right_name| {
        const candidate = right.columnIndex(right_name) orelse continue;
        if (candidate == right_index) return true;
    }
    return false;
}

fn coalesceJoinKeys(left: DeviceColumn, right: DeviceColumn) DeviceDataError!DeviceColumn {
    if (left.dtype() != right.dtype()) return error.TypeMismatch;
    return switch (left) {
        .bool => |typed| .{ .bool = try coalesceTypedJoinKeys(bool, typed, right.bool) },
        .i8 => |typed| .{ .i8 = try coalesceTypedJoinKeys(i8, typed, right.i8) },
        .i16 => |typed| .{ .i16 = try coalesceTypedJoinKeys(i16, typed, right.i16) },
        .i32 => |typed| .{ .i32 = try coalesceTypedJoinKeys(i32, typed, right.i32) },
        .i64 => |typed| .{ .i64 = try coalesceTypedJoinKeys(i64, typed, right.i64) },
        .u8 => |typed| .{ .u8 = try coalesceTypedJoinKeys(u8, typed, right.u8) },
        .u16 => |typed| .{ .u16 = try coalesceTypedJoinKeys(u16, typed, right.u16) },
        .u32 => |typed| .{ .u32 = try coalesceTypedJoinKeys(u32, typed, right.u32) },
        .u64 => |typed| .{ .u64 = try coalesceTypedJoinKeys(u64, typed, right.u64) },
        .usize => |typed| .{ .usize = try coalesceTypedJoinKeys(usize, typed, right.usize) },
        .isize => |typed| .{ .isize = try coalesceTypedJoinKeys(isize, typed, right.isize) },
        .f16 => |typed| .{ .f16 = try coalesceTypedJoinKeys(f16, typed, right.f16) },
        .f32 => |typed| .{ .f32 = try coalesceTypedJoinKeys(f32, typed, right.f32) },
        .f64 => |typed| .{ .f64 = try coalesceTypedJoinKeys(f64, typed, right.f64) },
        .bf16 => |typed| .{ .bf16 = try coalesceTypedJoinKeys(array_mod.BFloat16, typed, right.bf16) },
        .c64 => |typed| .{ .c64 = try coalesceTypedJoinKeys(array_mod.Complex64, typed, right.c64) },
        .c128 => |typed| .{ .c128 = try coalesceTypedJoinKeys(array_mod.Complex128, typed, right.c128) },
    };
}

fn coalesceTypedJoinKeys(comptime T: type, left: DeviceTypedColumn(T), right: DeviceTypedColumn(T)) DeviceDataError!DeviceTypedColumn(T) {
    if (!left.device().sameDevice(right.device())) return error.InvalidDevice;
    if (left.len() != right.len()) return error.LengthMismatch;
    const allocator = left.values.allocator;
    const left_values = try left.values.toOwnedSlice(allocator);
    defer allocator.free(left_values);
    const right_values = try right.values.toOwnedSlice(allocator);
    defer allocator.free(right_values);
    const maybe_left_validity = try validityValues(left, allocator);
    defer if (maybe_left_validity) |validity| allocator.free(validity);
    const maybe_right_validity = try validityValues(right, allocator);
    defer if (maybe_right_validity) |validity| allocator.free(validity);

    const values = try allocator.alloc(T, left_values.len);
    defer allocator.free(values);
    const validity = try allocator.alloc(bool, left_values.len);
    defer allocator.free(validity);
    for (values, validity, 0..) |*value_slot, *valid_slot, i| {
        const left_valid = if (maybe_left_validity) |mask| mask[i] else true;
        const right_valid = if (maybe_right_validity) |mask| mask[i] else true;
        if (left_valid) {
            value_slot.* = left_values[i];
            valid_slot.* = true;
        } else if (right_valid) {
            value_slot.* = right_values[i];
            valid_slot.* = true;
        } else {
            value_slot.* = zeroValue(T);
            valid_slot.* = false;
        }
    }
    if (countNulls(validity) == 0) return DeviceTypedColumn(T).fromSlice(allocator, values, left.device());
    return DeviceTypedColumn(T).fromSliceWithValidity(allocator, values, validity, left.device());
}

fn nameNeedsSuffix(left: DeviceDataFrame, name: []const u8) bool {
    return left.columnIndex(name) != null;
}

fn suffixedNameTemp(allocator: std.mem.Allocator, name: []const u8, suffix: []const u8) std.mem.Allocator.Error![]const u8 {
    var buffer: std.ArrayList(u8) = .empty;
    errdefer buffer.deinit(allocator);
    try buffer.appendSlice(allocator, name);
    try buffer.appendSlice(allocator, suffix);
    return buffer.toOwnedSlice(allocator);
}

fn requireCompatibleColumnArrays(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!void {
    if (!lhs.device.sameDevice(rhs.device)) return error.InvalidDevice;
    if (lhs.shape.len != 1 or rhs.shape.len != 1 or lhs.shape[0] != rhs.shape[0]) return error.ShapeMismatch;
}

fn combineValidityMasks(
    _: std.mem.Allocator,
    lhs: ?array_mod.Array(bool),
    rhs: ?array_mod.Array(bool),
    rows: usize,
    device_value: array_mod.Device,
) array_mod.ArrayError!?array_mod.Array(bool) {
    if (lhs == null and rhs == null) return null;
    if (lhs) |mask| {
        if (!mask.device.sameDevice(device_value) or mask.shape.len != 1 or mask.shape[0] != rows) return error.InvalidDevice;
    }
    if (rhs) |mask| {
        if (!mask.device.sameDevice(device_value) or mask.shape.len != 1 or mask.shape[0] != rows) return error.InvalidDevice;
    }
    if (lhs == null) return try rhs.?.clone();
    if (rhs == null) return try lhs.?.clone();
    const lhs_values = try lhs.?.toOwnedSlice(lhs.?.allocator);
    defer lhs.?.allocator.free(lhs_values);
    const rhs_values = try rhs.?.toOwnedSlice(rhs.?.allocator);
    defer rhs.?.allocator.free(rhs_values);
    const out_values = try lhs.?.allocator.alloc(bool, rows);
    defer lhs.?.allocator.free(out_values);
    for (lhs_values, rhs_values, out_values) |left_valid, right_valid, *slot| {
        slot.* = left_valid and right_valid;
    }
    return try array_mod.Array(bool).fromSliceOn(lhs.?.allocator, out_values, &.{rows}, device_value);
}

fn isIntegerColumnType(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .int, .comptime_int => true,
        else => false,
    };
}

fn isOrderedColumnType(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .int, .float, .comptime_int, .comptime_float => true,
        else => false,
    };
}

fn deviceDTypeToArrowDataType(dtype: DeviceDType) ArrowInteropError!boltha.arrow.DataType {
    return switch (dtype) {
        .bool => .bool,
        .i8 => .{ .int = .{ .bit_width = 8, .signed = true } },
        .i16 => .{ .int = .{ .bit_width = 16, .signed = true } },
        .i32 => .{ .int = .{ .bit_width = 32, .signed = true } },
        .i64, .isize => .{ .int = .{ .bit_width = 64, .signed = true } },
        .u8 => .{ .int = .{ .bit_width = 8, .signed = false } },
        .u16 => .{ .int = .{ .bit_width = 16, .signed = false } },
        .u32 => .{ .int = .{ .bit_width = 32, .signed = false } },
        .u64, .usize => .{ .int = .{ .bit_width = 64, .signed = false } },
        .f16 => .{ .floating_point = .half },
        .f32 => .{ .floating_point = .single },
        .f64 => .{ .floating_point = .double },
        // Boltha already models Arrow primitive/fixed/nested types. Vectra's
        // BFloat16 and complex values need explicit logical-extension metadata
        // before they can be exported without losing semantics, so keep them
        // rejected rather than pretending they are plain fixed-size binaries.
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn validityValues(column: anytype, allocator: std.mem.Allocator) array_mod.ArrayError!?[]bool {
    const mask = column.validity orelse return null;
    return try mask.toOwnedSlice(allocator);
}

fn primitiveColumnToArrow(
    comptime T: type,
    comptime tag_name: []const u8,
    column: DeviceTypedColumn(T),
    allocator: std.mem.Allocator,
) ArrowInteropError!boltha.arrow.AnyArray {
    const values = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const primitive = if (maybe_validity) |validity| blk: {
        const optional_values = try allocator.alloc(?T, values.len);
        defer allocator.free(optional_values);
        for (values, validity, optional_values) |value, valid, *slot| {
            slot.* = if (valid) value else null;
        }
        break :blk try boltha.arrow.PrimitiveArray(T).fromOptionalSlice(allocator, optional_values, zeroValue(T));
    } else try boltha.arrow.PrimitiveArray(T).fromSlice(allocator, values);
    return @unionInit(boltha.arrow.AnyArray, tag_name, primitive);
}

fn boolColumnToArrow(column: DeviceTypedColumn(bool), allocator: std.mem.Allocator) ArrowInteropError!boltha.arrow.AnyArray {
    const values = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const array_value = if (maybe_validity) |validity| blk: {
        const optional_values = try allocator.alloc(?bool, values.len);
        defer allocator.free(optional_values);
        for (values, validity, optional_values) |value, valid, *slot| {
            slot.* = if (valid) value else null;
        }
        break :blk try boltha.arrow.BooleanArray.fromOptionalSlice(allocator, optional_values);
    } else try boltha.arrow.BooleanArray.fromSlice(allocator, values);
    return .{ .boolean = array_value };
}

fn deviceColumnFromArrowArray(allocator: std.mem.Allocator, column: boltha.arrow.AnyArray, device_value: array_mod.Device) ArrowInteropError!DeviceColumn {
    return switch (column) {
        .boolean => |array| boolDeviceColumnFromArrow(allocator, array, device_value),
        .int8 => |array| primitiveDeviceColumnFromArrow(i8, allocator, array, device_value),
        .uint8 => |array| primitiveDeviceColumnFromArrow(u8, allocator, array, device_value),
        .int16 => |array| primitiveDeviceColumnFromArrow(i16, allocator, array, device_value),
        .uint16 => |array| primitiveDeviceColumnFromArrow(u16, allocator, array, device_value),
        .int32 => |array| primitiveDeviceColumnFromArrow(i32, allocator, array, device_value),
        .uint32 => |array| primitiveDeviceColumnFromArrow(u32, allocator, array, device_value),
        .int64 => |array| primitiveDeviceColumnFromArrow(i64, allocator, array, device_value),
        .uint64 => |array| primitiveDeviceColumnFromArrow(u64, allocator, array, device_value),
        .float16 => |array| primitiveDeviceColumnFromArrow(f16, allocator, array, device_value),
        .float32 => |array| primitiveDeviceColumnFromArrow(f32, allocator, array, device_value),
        .float64 => |array| primitiveDeviceColumnFromArrow(f64, allocator, array, device_value),
        else => error.TypeUnsupported,
    };
}

fn emptyFromArrowSchema(allocator: std.mem.Allocator, schema: boltha.arrow.Schema, rows: usize, device_value: array_mod.Device) ArrowInteropError!DeviceDataFrame {
    if (rows != 0) return error.TypeUnsupported;
    var defs = try allocator.alloc(DeviceColumnDef, schema.fields.len);
    defer allocator.free(defs);
    var initialized: usize = 0;
    defer {
        for (defs[0..initialized]) |*def| def.data.deinit();
    }
    for (schema.fields, 0..) |field, i| {
        defs[i] = .{
            .name = field.name,
            .data = try emptyDeviceColumnFromArrowType(allocator, field.data_type, device_value),
        };
        initialized += 1;
    }
    return DeviceDataFrame.init(allocator, defs);
}

fn emptyFromArrowSchemaProjection(
    allocator: std.mem.Allocator,
    schema: boltha.arrow.Schema,
    rows: usize,
    wanted_names: []const []const u8,
    device_value: array_mod.Device,
) ArrowInteropError!DeviceDataFrame {
    if (rows != 0) return error.TypeUnsupported;
    var defs = try allocator.alloc(DeviceColumnDef, wanted_names.len);
    defer allocator.free(defs);
    var initialized: usize = 0;
    defer {
        for (defs[0..initialized]) |*def| def.data.deinit();
    }
    for (wanted_names, 0..) |name, i| {
        const column_index = schema.fieldIndexByName(name) orelse return error.ColumnNotFound;
        const field = schema.fields[column_index];
        defs[i] = .{
            .name = field.name,
            .data = try emptyDeviceColumnFromArrowType(allocator, field.data_type, device_value),
        };
        initialized += 1;
    }
    return DeviceDataFrame.init(allocator, defs);
}

fn emptyDeviceColumnFromArrowType(allocator: std.mem.Allocator, dtype: boltha.arrow.DataType, device_value: array_mod.Device) ArrowInteropError!DeviceColumn {
    return switch (dtype) {
        .bool => DeviceColumn.fromSlice(bool, allocator, &.{}, device_value),
        .int => |info| if (info.signed) switch (info.bit_width) {
            8 => DeviceColumn.fromSlice(i8, allocator, &.{}, device_value),
            16 => DeviceColumn.fromSlice(i16, allocator, &.{}, device_value),
            32 => DeviceColumn.fromSlice(i32, allocator, &.{}, device_value),
            64 => DeviceColumn.fromSlice(i64, allocator, &.{}, device_value),
            else => error.TypeUnsupported,
        } else switch (info.bit_width) {
            8 => DeviceColumn.fromSlice(u8, allocator, &.{}, device_value),
            16 => DeviceColumn.fromSlice(u16, allocator, &.{}, device_value),
            32 => DeviceColumn.fromSlice(u32, allocator, &.{}, device_value),
            64 => DeviceColumn.fromSlice(u64, allocator, &.{}, device_value),
            else => error.TypeUnsupported,
        },
        .floating_point => |fp| switch (fp) {
            .half => DeviceColumn.fromSlice(f16, allocator, &.{}, device_value),
            .single => DeviceColumn.fromSlice(f32, allocator, &.{}, device_value),
            .double => DeviceColumn.fromSlice(f64, allocator, &.{}, device_value),
        },
        else => error.TypeUnsupported,
    };
}

fn readBolthaTableWithRangePruning(
    allocator: std.mem.Allocator,
    bytes: []const u8,
    column_name: []const u8,
    predicate: ParquetRangePredicate,
) ParquetInteropError!boltha.arrow.Table {
    return switch (predicate) {
        .bool => |range| readBolthaTableWithBoolRangePruning(allocator, bytes, column_name, range),
        .i8 => |range| boltha.parquet.readTableWithInt8Pruning(allocator, bytes, column_name, .{ .min = optionalCast(i32, range.min), .max = optionalCast(i32, range.max) }),
        .i16 => |range| boltha.parquet.readTableWithInt16Pruning(allocator, bytes, column_name, .{ .min = optionalCast(i32, range.min), .max = optionalCast(i32, range.max) }),
        .i32 => |range| boltha.parquet.readTableWithInt32Pruning(allocator, bytes, column_name, .{ .min = range.min, .max = range.max }),
        .i64 => |range| boltha.parquet.readTableWithInt64Pruning(allocator, bytes, column_name, .{ .min = range.min, .max = range.max }),
        .isize => |range| boltha.parquet.readTableWithInt64Pruning(allocator, bytes, column_name, .{ .min = optionalCast(i64, range.min), .max = optionalCast(i64, range.max) }),
        .u8 => |range| boltha.parquet.readTableWithUInt8Pruning(allocator, bytes, column_name, .{ .min = optionalCast(u32, range.min), .max = optionalCast(u32, range.max) }),
        .u16 => |range| boltha.parquet.readTableWithUInt16Pruning(allocator, bytes, column_name, .{ .min = optionalCast(u32, range.min), .max = optionalCast(u32, range.max) }),
        .u32 => |range| boltha.parquet.readTableWithUInt32Pruning(allocator, bytes, column_name, .{ .min = range.min, .max = range.max }),
        .u64 => |range| boltha.parquet.readTableWithUInt64Pruning(allocator, bytes, column_name, .{ .min = range.min, .max = range.max }),
        .usize => |range| boltha.parquet.readTableWithUInt64Pruning(allocator, bytes, column_name, .{ .min = optionalCast(u64, range.min), .max = optionalCast(u64, range.max) }),
        .f16 => |range| boltha.parquet.readTableWithFloat16Pruning(allocator, bytes, column_name, .{ .min = range.min, .max = range.max }),
        .f32 => |range| boltha.parquet.readTableWithFloatPruning(allocator, bytes, column_name, .{ .min = range.min, .max = range.max }),
        .f64 => |range| boltha.parquet.readTableWithDoublePruning(allocator, bytes, column_name, .{ .min = range.min, .max = range.max }),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn readBolthaTableWithBoolRangePruning(
    allocator: std.mem.Allocator,
    bytes: []const u8,
    column_name: []const u8,
    range: Range(bool),
) ParquetInteropError!boltha.arrow.Table {
    if (range.min) |min_value| {
        if (range.max) |max_value| {
            if (min_value == max_value) {
                return boltha.parquet.readTableWithBooleanPruning(allocator, bytes, column_name, .{ .value = min_value });
            }
            if (!min_value and max_value) return boltha.parquet.readTable(allocator, bytes);
            return emptyBolthaTableForParquetBytes(allocator, bytes);
        }
        return if (min_value)
            boltha.parquet.readTableWithBooleanPruning(allocator, bytes, column_name, .{ .value = true })
        else
            boltha.parquet.readTable(allocator, bytes);
    }
    if (range.max) |max_value| {
        return if (!max_value)
            boltha.parquet.readTableWithBooleanPruning(allocator, bytes, column_name, .{ .value = false })
        else
            boltha.parquet.readTable(allocator, bytes);
    }
    return boltha.parquet.readTable(allocator, bytes);
}

fn emptyBolthaTableForParquetBytes(allocator: std.mem.Allocator, bytes: []const u8) ParquetInteropError!boltha.arrow.Table {
    var schema = try boltha.parquet.readSchema(allocator, bytes);
    errdefer schema.deinit(allocator);
    const batches = try allocator.alloc(boltha.arrow.RecordBatch, 0);
    errdefer allocator.free(batches);
    return boltha.arrow.Table.initOwned(schema, batches);
}

fn optionalCast(comptime T: type, value: anytype) ?T {
    const unwrapped = value orelse return null;
    return std.math.cast(T, unwrapped) orelse unreachable;
}

fn concatDeviceDataFramesRows(first: DeviceDataFrame, second: DeviceDataFrame) DeviceDataError!DeviceDataFrame {
    if (!first.device.sameDevice(second.device)) return error.InvalidDevice;
    if (first.columns.len != second.columns.len) return error.LengthMismatch;
    for (first.names, second.names, first.columns, second.columns) |first_name, second_name, first_col, second_col| {
        if (!std.mem.eql(u8, first_name, second_name)) return error.ColumnNotFound;
        if (first_col.dtype() != second_col.dtype()) return error.TypeMismatch;
    }

    var columns = try first.allocator.alloc(DeviceColumn, first.columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        first.allocator.free(columns);
    }
    for (first.columns, second.columns, 0..) |first_col, second_col, i| {
        columns[i] = try concatDeviceColumns(first_col, second_col);
        initialized += 1;
    }
    return initDeviceDataFrameFromOwnedColumns(first.allocator, first.names, columns, first.rows + second.rows, first.device);
}

fn concatDeviceColumns(first: DeviceColumn, second: DeviceColumn) DeviceDataError!DeviceColumn {
    if (first.dtype() != second.dtype()) return error.TypeMismatch;
    return switch (first) {
        .bool => |typed| .{ .bool = try concatTypedColumns(bool, typed, second.bool) },
        .i8 => |typed| .{ .i8 = try concatTypedColumns(i8, typed, second.i8) },
        .i16 => |typed| .{ .i16 = try concatTypedColumns(i16, typed, second.i16) },
        .i32 => |typed| .{ .i32 = try concatTypedColumns(i32, typed, second.i32) },
        .i64 => |typed| .{ .i64 = try concatTypedColumns(i64, typed, second.i64) },
        .u8 => |typed| .{ .u8 = try concatTypedColumns(u8, typed, second.u8) },
        .u16 => |typed| .{ .u16 = try concatTypedColumns(u16, typed, second.u16) },
        .u32 => |typed| .{ .u32 = try concatTypedColumns(u32, typed, second.u32) },
        .u64 => |typed| .{ .u64 = try concatTypedColumns(u64, typed, second.u64) },
        .usize => |typed| .{ .usize = try concatTypedColumns(usize, typed, second.usize) },
        .isize => |typed| .{ .isize = try concatTypedColumns(isize, typed, second.isize) },
        .f16 => |typed| .{ .f16 = try concatTypedColumns(f16, typed, second.f16) },
        .f32 => |typed| .{ .f32 = try concatTypedColumns(f32, typed, second.f32) },
        .f64 => |typed| .{ .f64 = try concatTypedColumns(f64, typed, second.f64) },
        .bf16 => |typed| .{ .bf16 = try concatTypedColumns(array_mod.BFloat16, typed, second.bf16) },
        .c64 => |typed| .{ .c64 = try concatTypedColumns(array_mod.Complex64, typed, second.c64) },
        .c128 => |typed| .{ .c128 = try concatTypedColumns(array_mod.Complex128, typed, second.c128) },
    };
}

fn concatTypedColumns(comptime T: type, first: DeviceTypedColumn(T), second: DeviceTypedColumn(T)) DeviceDataError!DeviceTypedColumn(T) {
    if (!first.device().sameDevice(second.device())) return error.InvalidDevice;
    const allocator = first.values.allocator;
    const first_values = try first.values.toOwnedSlice(allocator);
    defer allocator.free(first_values);
    const second_values = try second.values.toOwnedSlice(allocator);
    defer allocator.free(second_values);
    const values = try allocator.alloc(T, first_values.len + second_values.len);
    defer allocator.free(values);
    @memcpy(values[0..first_values.len], first_values);
    @memcpy(values[first_values.len..], second_values);

    const first_validity = try validityValues(first, allocator);
    defer if (first_validity) |validity| allocator.free(validity);
    const second_validity = try validityValues(second, allocator);
    defer if (second_validity) |validity| allocator.free(validity);

    if (first_validity == null and second_validity == null) return DeviceTypedColumn(T).fromSlice(allocator, values, first.device());
    const validity = try allocator.alloc(bool, values.len);
    defer allocator.free(validity);
    for (validity[0..first_values.len], 0..) |*slot, i| slot.* = if (first_validity) |mask| mask[i] else true;
    for (validity[first_values.len..], 0..) |*slot, i| slot.* = if (second_validity) |mask| mask[i] else true;
    return DeviceTypedColumn(T).fromSliceWithValidity(allocator, values, validity, first.device());
}

fn primitiveDeviceColumnFromArrow(
    comptime T: type,
    allocator: std.mem.Allocator,
    arrow_array: boltha.arrow.PrimitiveArray(T),
    device_value: array_mod.Device,
) ArrowInteropError!DeviceColumn {
    if (arrow_array.null_count == 0) return DeviceColumn.fromSlice(T, allocator, arrow_array.values, device_value);

    const validity = try allocator.alloc(bool, arrow_array.values.len);
    defer allocator.free(validity);
    for (validity, 0..) |*slot, i| slot.* = !arrow_array.isNull(i);
    return DeviceColumn.fromSliceWithValidity(T, allocator, arrow_array.values, validity, device_value);
}

fn boolDeviceColumnFromArrow(allocator: std.mem.Allocator, arrow_array: boltha.arrow.BooleanArray, device_value: array_mod.Device) ArrowInteropError!DeviceColumn {
    const values = try allocator.alloc(bool, arrow_array.len());
    defer allocator.free(values);
    const validity = try allocator.alloc(bool, arrow_array.len());
    defer allocator.free(validity);
    for (values, validity, 0..) |*value_slot, *valid_slot, i| {
        if (arrow_array.value(i)) |value| {
            value_slot.* = value;
            valid_slot.* = true;
        } else {
            value_slot.* = false;
            valid_slot.* = false;
        }
    }
    if (arrow_array.null_count == 0) return DeviceColumn.fromSlice(bool, allocator, values, device_value);
    return DeviceColumn.fromSliceWithValidity(bool, allocator, values, validity, device_value);
}

fn indexColumnToArrow(comptime T: type, column: DeviceTypedColumn(T), allocator: std.mem.Allocator) ArrowInteropError!boltha.arrow.AnyArray {
    const values = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    if (comptime T == usize) {
        const converted = try allocator.alloc(u64, values.len);
        defer allocator.free(converted);
        for (values, converted) |value, *slot| {
            slot.* = std.math.cast(u64, value) orelse return error.TypeUnsupported;
        }
        if (maybe_validity) |validity| {
            const optional_values = try allocator.alloc(?u64, values.len);
            defer allocator.free(optional_values);
            for (converted, validity, optional_values) |value, valid, *slot| {
                slot.* = if (valid) value else null;
            }
            return .{ .uint64 = try boltha.arrow.PrimitiveArray(u64).fromOptionalSlice(allocator, optional_values, 0) };
        }
        return .{ .uint64 = try boltha.arrow.PrimitiveArray(u64).fromSlice(allocator, converted) };
    }

    if (comptime T == isize) {
        const converted = try allocator.alloc(i64, values.len);
        defer allocator.free(converted);
        for (values, converted) |value, *slot| {
            slot.* = std.math.cast(i64, value) orelse return error.TypeUnsupported;
        }
        if (maybe_validity) |validity| {
            const optional_values = try allocator.alloc(?i64, values.len);
            defer allocator.free(optional_values);
            for (converted, validity, optional_values) |value, valid, *slot| {
                slot.* = if (valid) value else null;
            }
            return .{ .int64 = try boltha.arrow.PrimitiveArray(i64).fromOptionalSlice(allocator, optional_values, 0) };
        }
        return .{ .int64 = try boltha.arrow.PrimitiveArray(i64).fromSlice(allocator, converted) };
    }

    unreachable;
}

fn zeroValue(comptime T: type) T {
    if (comptime T == bool) return false;
    if (comptime T == array_mod.BFloat16) return array_mod.BFloat16.fromF32(0);
    if (comptime T == array_mod.Complex64) return .{ .re = 0, .im = 0 };
    if (comptime T == array_mod.Complex128) return .{ .re = 0, .im = 0 };
    return switch (@typeInfo(T)) {
        .int, .float => 0,
        else => @compileError("zeroValue only supports primitive numeric Arrow values"),
    };
}

fn rowIndicesFromMask(allocator: std.mem.Allocator, mask: []const bool) array_mod.ArrayError![]usize {
    var count: usize = 0;
    for (mask) |keep| {
        if (keep) count += 1;
    }
    const indices = try allocator.alloc(usize, count);
    var write: usize = 0;
    for (mask, 0..) |keep, i| {
        if (keep) {
            indices[write] = i;
            write += 1;
        }
    }
    return indices;
}

fn sliceArray1d(comptime T: type, values: array_mod.Array(T), start: usize, stop: usize) array_mod.ArrayError!array_mod.Array(T) {
    if (values.shape.len != 1) return error.InvalidShape;
    if (start > values.shape[0] or stop < start or stop > values.shape[0]) return error.IndexOutOfBounds;
    const host_values = try values.toOwnedSlice(values.allocator);
    defer values.allocator.free(host_values);
    return array_mod.Array(T).fromSliceOn(values.allocator, host_values[start..stop], &.{stop - start}, values.device);
}

fn takeArray1d(comptime T: type, values: array_mod.Array(T), row_indices: []const usize) array_mod.ArrayError!array_mod.Array(T) {
    if (values.shape.len != 1) return error.InvalidShape;
    const host_values = try values.toOwnedSlice(values.allocator);
    defer values.allocator.free(host_values);
    const out_values = try values.allocator.alloc(T, row_indices.len);
    defer values.allocator.free(out_values);
    for (row_indices, out_values) |idx, *slot| {
        if (idx >= host_values.len) return error.IndexOutOfBounds;
        slot.* = host_values[idx];
    }
    return array_mod.Array(T).fromSliceOn(values.allocator, out_values, &.{row_indices.len}, values.device);
}

fn initDeviceDataFrameFromOwnedColumns(
    allocator: std.mem.Allocator,
    source_names: []const []const u8,
    columns: []DeviceColumn,
    rows: usize,
    device_value: array_mod.Device,
) DeviceDataError!DeviceDataFrame {
    if (source_names.len != columns.len) return error.LengthMismatch;
    for (columns) |col| {
        if (col.len() != rows) return error.LengthMismatch;
        if (!col.device().sameDevice(device_value)) return error.InvalidDevice;
    }

    var names = try allocator.alloc([]const u8, source_names.len);
    errdefer allocator.free(names);
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    for (source_names, names) |source, *slot| {
        slot.* = try allocator.dupe(u8, source);
        initialized += 1;
    }
    return .{ .allocator = allocator, .names = names, .columns = columns, .rows = rows, .device = device_value };
}

pub const DataFrame = struct {
    allocator: std.mem.Allocator,
    names: [][]const u8,
    columns: []Column,
    rows: usize,

    pub fn init(allocator: std.mem.Allocator, defs: []const ColumnDef) DataError!DataFrame {
        if (defs.len == 0) {
            return .{ .allocator = allocator, .names = &.{}, .columns = &.{}, .rows = 0 };
        }
        const rows = defs[0].data.len();
        for (defs) |def| {
            if (def.data.len() != rows) return error.LengthMismatch;
        }

        var names = try allocator.alloc([]const u8, defs.len);
        errdefer allocator.free(names);
        var columns = try allocator.alloc(Column, defs.len);
        errdefer allocator.free(columns);

        var initialized: usize = 0;
        errdefer {
            for (0..initialized) |i| {
                allocator.free(names[i]);
                freeColumn(allocator, columns[i]);
            }
        }

        for (defs, 0..) |def, i| {
            names[i] = try allocator.dupe(u8, def.name);
            columns[i] = try cloneColumn(allocator, def.data);
            initialized += 1;
        }
        return .{ .allocator = allocator, .names = names, .columns = columns, .rows = rows };
    }

    pub fn deinit(self: *DataFrame) void {
        for (self.names) |name| self.allocator.free(name);
        for (self.columns) |col| freeColumn(self.allocator, col);
        if (self.names.len != 0) self.allocator.free(self.names);
        if (self.columns.len != 0) self.allocator.free(self.columns);
        self.* = undefined;
    }

    pub fn height(self: DataFrame) usize {
        return self.rows;
    }

    pub fn width(self: DataFrame) usize {
        return self.columns.len;
    }

    pub fn shape(self: DataFrame) struct { rows: usize, cols: usize } {
        return .{ .rows = self.rows, .cols = self.columns.len };
    }

    pub fn columnIndex(self: DataFrame, name: []const u8) ?usize {
        for (self.names, 0..) |n, i| {
            if (std.mem.eql(u8, n, name)) return i;
        }
        return null;
    }

    pub fn column(self: DataFrame, name: []const u8) DataError!Column {
        const idx = self.columnIndex(name) orelse return error.ColumnNotFound;
        return self.columns[idx];
    }

    pub fn series(self: DataFrame, comptime T: type, name: []const u8) DataError!series_mod.Series(T) {
        const col = try self.column(name);
        return switch (col) {
            .f64 => |v| if (T == f64) series_mod.Series(T).init(self.allocator, name, v) else error.TypeMismatch,
            .i64 => |v| if (T == i64) series_mod.Series(T).init(self.allocator, name, v) else error.TypeMismatch,
            .bool => |v| if (T == bool) series_mod.Series(T).init(self.allocator, name, v) else error.TypeMismatch,
            .string => error.TypeMismatch,
        };
    }

    pub fn select(self: DataFrame, names: []const []const u8) DataError!DataFrame {
        var defs = try self.allocator.alloc(ColumnDef, names.len);
        defer self.allocator.free(defs);
        for (names, 0..) |name, i| {
            defs[i] = .{ .name = name, .data = try self.column(name) };
        }
        return DataFrame.init(self.allocator, defs);
    }

    pub fn withColumn(self: DataFrame, name: []const u8, data: Column) DataError!DataFrame {
        if (self.rows != data.len()) return error.LengthMismatch;
        var defs = try self.allocator.alloc(ColumnDef, self.columns.len + 1);
        defer self.allocator.free(defs);
        for (self.names, self.columns, 0..) |n, c, i| defs[i] = .{ .name = n, .data = c };
        defs[self.columns.len] = .{ .name = name, .data = data };
        return DataFrame.init(self.allocator, defs);
    }

    pub fn filter(self: DataFrame, mask: []const bool) DataError!DataFrame {
        if (mask.len != self.rows) return error.LengthMismatch;
        var defs = try self.allocator.alloc(ColumnDef, self.columns.len);
        defer self.allocator.free(defs);
        for (self.names, self.columns, 0..) |name, col, i| {
            defs[i] = .{ .name = name, .data = try filterColumn(self.allocator, col, mask) };
        }
        defer {
            for (defs) |def| freeColumn(self.allocator, def.data);
        }
        return DataFrame.init(self.allocator, defs);
    }

    pub fn head(self: DataFrame, n: usize) DataError!DataFrame {
        return self.sliceRows(0, @min(n, self.rows));
    }

    pub fn tail(self: DataFrame, n: usize) DataError!DataFrame {
        const count = @min(n, self.rows);
        return self.sliceRows(self.rows - count, self.rows);
    }

    pub fn sliceRows(self: DataFrame, start: usize, stop: usize) DataError!DataFrame {
        const end = @min(stop, self.rows);
        const begin = @min(start, end);
        var defs = try self.allocator.alloc(ColumnDef, self.columns.len);
        defer self.allocator.free(defs);
        for (self.names, self.columns, 0..) |name, col, i| defs[i] = .{ .name = name, .data = sliceColumn(col, begin, end) };
        return DataFrame.init(self.allocator, defs);
    }

    pub fn sortBy(self: DataFrame, name: []const u8, descending: bool) DataError!DataFrame {
        const idx = self.columnIndex(name) orelse return error.ColumnNotFound;
        const order = try self.allocator.alloc(usize, self.rows);
        defer self.allocator.free(order);
        for (order, 0..) |*slot, i| slot.* = i;
        const Ctx = struct {
            col: Column,
            desc: bool,
            fn lessThan(ctx: @This(), a: usize, b: usize) bool {
                const less = switch (ctx.col) {
                    .f64 => |v| v[a] < v[b],
                    .i64 => |v| v[a] < v[b],
                    .bool => |v| !v[a] and v[b],
                    .string => |v| std.mem.lessThan(u8, v[a], v[b]),
                };
                return if (ctx.desc) !less else less;
            }
        };
        std.sort.insertion(usize, order, Ctx{ .col = self.columns[idx], .desc = descending }, Ctx.lessThan);
        return self.take(order);
    }

    pub fn take(self: DataFrame, row_indices: []const usize) DataError!DataFrame {
        var defs = try self.allocator.alloc(ColumnDef, self.columns.len);
        defer self.allocator.free(defs);
        for (self.names, self.columns, 0..) |name, col, i| {
            defs[i] = .{ .name = name, .data = try takeColumn(self.allocator, col, row_indices) };
        }
        defer {
            for (defs) |def| freeColumn(self.allocator, def.data);
        }
        return DataFrame.init(self.allocator, defs);
    }

    pub fn groupBySum(self: DataFrame, key_name: []const u8, value_name: []const u8) DataError!DataFrame {
        const key_col = try self.column(key_name);
        const value_col = try self.column(value_name);
        if (std.meta.activeTag(key_col) != .string or std.meta.activeTag(value_col) != .f64) return error.TypeMismatch;
        var keys: std.ArrayList([]const u8) = .empty;
        defer keys.deinit(self.allocator);
        var sums: std.ArrayList(f64) = .empty;
        defer sums.deinit(self.allocator);

        for (key_col.string, value_col.f64) |key, value| {
            var found: ?usize = null;
            for (keys.items, 0..) |existing, i| {
                if (std.mem.eql(u8, existing, key)) {
                    found = i;
                    break;
                }
            }
            if (found) |i| {
                sums.items[i] += value;
            } else {
                try keys.append(self.allocator, key);
                try sums.append(self.allocator, value);
            }
        }

        return DataFrame.init(self.allocator, &.{
            .{ .name = key_name, .data = .{ .string = keys.items } },
            .{ .name = value_name, .data = .{ .f64 = sums.items } },
        });
    }

    pub fn describe(self: DataFrame) DataError!DataFrame {
        const stat_names = [_][]const u8{ "count", "mean", "min", "max" };
        var defs_list: std.ArrayList(ColumnDef) = .empty;
        defer defs_list.deinit(self.allocator);
        try defs_list.append(self.allocator, .{ .name = "stat", .data = .{ .string = stat_names[0..] } });
        for (self.names, self.columns) |name, col| {
            switch (col) {
                .f64 => |v| {
                    const vals = try describeF64(self.allocator, v);
                    try defs_list.append(self.allocator, .{ .name = name, .data = .{ .f64 = vals } });
                },
                .i64 => |v| {
                    const vals = try describeI64(self.allocator, v);
                    try defs_list.append(self.allocator, .{ .name = name, .data = .{ .f64 = vals } });
                },
                else => {},
            }
        }
        defer {
            for (defs_list.items[1..]) |def| freeColumn(self.allocator, def.data);
        }
        return DataFrame.init(self.allocator, defs_list.items);
    }

    pub fn toArray(self: DataFrame, comptime T: type, names: []const []const u8) (DataError || array_mod.ArrayError)!array_mod.Array(T) {
        var values = try self.allocator.alloc(T, self.rows * names.len);
        defer self.allocator.free(values);
        for (0..self.rows) |r| {
            for (names, 0..) |name, c| {
                const col = try self.column(name);
                values[r * names.len + c] = switch (col) {
                    .f64 => |v| if (T == f64) v[r] else @as(T, @intFromFloat(v[r])),
                    .i64 => |v| if (T == i64) v[r] else @as(T, @floatFromInt(v[r])),
                    .bool => |v| if (T == bool) v[r] else if (v[r]) 1 else 0,
                    .string => return error.TypeMismatch,
                };
            }
        }
        return array_mod.Array(T).fromSlice(self.allocator, values, &.{ self.rows, names.len });
    }

    pub fn toNDArray(self: DataFrame, comptime T: type, names: []const []const u8) (DataError || array_mod.ArrayError)!array_mod.NDArray(T) {
        return self.toArray(T, names);
    }

    pub fn readCsv(allocator: std.mem.Allocator, text: []const u8, has_header: bool) DataError!DataFrame {
        var lines = std.mem.splitScalar(u8, text, '\n');
        const first_raw = lines.next() orelse return error.InvalidCsv;
        const first = std.mem.trim(u8, first_raw, "\r ");
        if (first.len == 0) return error.InvalidCsv;
        var headers_list: std.ArrayList([]const u8) = .empty;
        defer {
            for (headers_list.items) |h| allocator.free(h);
            headers_list.deinit(allocator);
        }

        var first_values: ?[][]const u8 = null;
        defer if (first_values) |vals| {
            for (vals) |v| allocator.free(v);
            allocator.free(vals);
        };

        if (has_header) {
            try splitCsvLineOwned(allocator, first, &headers_list);
        } else {
            var vals_list: std.ArrayList([]const u8) = .empty;
            defer vals_list.deinit(allocator);
            try splitCsvLineOwned(allocator, first, &vals_list);
            first_values = try vals_list.toOwnedSlice(allocator);
            for (0..first_values.?.len) |i| {
                var aw: std.Io.Writer.Allocating = .init(allocator);
                defer aw.deinit();
                try aw.writer.print("col{}", .{i});
                try headers_list.append(allocator, try aw.toOwnedSlice());
            }
        }

        const cols = headers_list.items.len;
        var string_cols = try allocator.alloc(std.ArrayList([]const u8), cols);
        defer allocator.free(string_cols);
        for (string_cols) |*list| list.* = .empty;
        defer {
            for (string_cols) |*list| {
                for (list.items) |cell| allocator.free(cell);
                list.deinit(allocator);
            }
        }

        if (first_values) |vals| {
            if (vals.len != cols) return error.InvalidCsv;
            for (vals, 0..) |v, i| try string_cols[i].append(allocator, try allocator.dupe(u8, v));
        }
        while (lines.next()) |raw| {
            const line = std.mem.trim(u8, raw, "\r ");
            if (line.len == 0) continue;
            var vals_list: std.ArrayList([]const u8) = .empty;
            defer {
                for (vals_list.items) |v| allocator.free(v);
                vals_list.deinit(allocator);
            }
            try splitCsvLineOwned(allocator, line, &vals_list);
            if (vals_list.items.len != cols) return error.InvalidCsv;
            for (vals_list.items, 0..) |v, i| try string_cols[i].append(allocator, try allocator.dupe(u8, v));
        }

        var defs = try allocator.alloc(ColumnDef, cols);
        defer allocator.free(defs);
        for (headers_list.items, string_cols, 0..) |header, cells, i| {
            defs[i] = .{ .name = header, .data = try inferColumn(allocator, cells.items) };
        }
        defer {
            for (defs) |def| freeColumn(allocator, def.data);
        }
        return DataFrame.init(allocator, defs);
    }

    pub fn writeCsv(self: DataFrame, allocator: std.mem.Allocator) DataError![]u8 {
        var aw: std.Io.Writer.Allocating = .init(allocator);
        errdefer aw.deinit();
        for (self.names, 0..) |name, i| {
            if (i != 0) try aw.writer.print(",", .{});
            try aw.writer.print("{s}", .{name});
        }
        try aw.writer.print("\n", .{});
        for (0..self.rows) |r| {
            for (self.columns, 0..) |col, c| {
                if (c != 0) try aw.writer.print(",", .{});
                try printCell(&aw.writer, col, r);
            }
            try aw.writer.print("\n", .{});
        }
        return aw.toOwnedSlice();
    }

    pub fn print(self: DataFrame, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        try writer.print("DataFrame(shape=({}, {}))\n", .{ self.rows, self.columns.len });
        for (self.names, 0..) |name, i| {
            if (i != 0) try writer.print("\t", .{});
            try writer.print("{s}", .{name});
        }
        try writer.print("\n", .{});
        const limit = @min(self.rows, 12);
        for (0..limit) |r| {
            for (self.columns, 0..) |col, c| {
                if (c != 0) try writer.print("\t", .{});
                try printCell(writer, col, r);
            }
            try writer.print("\n", .{});
        }
        if (self.rows > limit) try writer.print("...\n", .{});
    }
};

fn cloneColumn(allocator: std.mem.Allocator, col: Column) DataError!Column {
    return switch (col) {
        .f64 => |v| .{ .f64 = try allocator.dupe(f64, v) },
        .i64 => |v| .{ .i64 = try allocator.dupe(i64, v) },
        .bool => |v| .{ .bool = try allocator.dupe(bool, v) },
        .string => |v| blk: {
            var out = try allocator.alloc([]const u8, v.len);
            errdefer allocator.free(out);
            var initialized: usize = 0;
            errdefer {
                for (out[0..initialized]) |s| allocator.free(s);
            }
            for (v, 0..) |s, i| {
                out[i] = try allocator.dupe(u8, s);
                initialized += 1;
            }
            break :blk .{ .string = out };
        },
    };
}

fn freeColumn(allocator: std.mem.Allocator, col: Column) void {
    switch (col) {
        .f64 => |v| allocator.free(v),
        .i64 => |v| allocator.free(v),
        .bool => |v| allocator.free(v),
        .string => |v| {
            for (v) |s| allocator.free(s);
            allocator.free(v);
        },
    }
}

fn filterColumn(allocator: std.mem.Allocator, col: Column, mask: []const bool) DataError!Column {
    var count: usize = 0;
    for (mask) |keep| {
        if (keep) count += 1;
    }
    return switch (col) {
        .f64 => |v| blk: {
            var out = try allocator.alloc(f64, count);
            var w: usize = 0;
            for (v, mask) |x, keep| if (keep) {
                out[w] = x;
                w += 1;
            };
            break :blk .{ .f64 = out };
        },
        .i64 => |v| blk: {
            var out = try allocator.alloc(i64, count);
            var w: usize = 0;
            for (v, mask) |x, keep| if (keep) {
                out[w] = x;
                w += 1;
            };
            break :blk .{ .i64 = out };
        },
        .bool => |v| blk: {
            var out = try allocator.alloc(bool, count);
            var w: usize = 0;
            for (v, mask) |x, keep| if (keep) {
                out[w] = x;
                w += 1;
            };
            break :blk .{ .bool = out };
        },
        .string => |v| blk: {
            var out = try allocator.alloc([]const u8, count);
            errdefer allocator.free(out);
            var w: usize = 0;
            errdefer {
                for (out[0..w]) |s| allocator.free(s);
            }
            for (v, mask) |x, keep| if (keep) {
                out[w] = try allocator.dupe(u8, x);
                w += 1;
            };
            break :blk .{ .string = out };
        },
    };
}

fn sliceColumn(col: Column, start: usize, stop: usize) Column {
    return switch (col) {
        .f64 => |v| .{ .f64 = v[start..stop] },
        .i64 => |v| .{ .i64 = v[start..stop] },
        .bool => |v| .{ .bool = v[start..stop] },
        .string => |v| .{ .string = v[start..stop] },
    };
}

fn takeColumn(allocator: std.mem.Allocator, col: Column, indices: []const usize) DataError!Column {
    return switch (col) {
        .f64 => |v| blk: {
            const out = try allocator.alloc(f64, indices.len);
            for (indices, out) |idx, *slot| slot.* = v[idx];
            break :blk .{ .f64 = out };
        },
        .i64 => |v| blk: {
            const out = try allocator.alloc(i64, indices.len);
            for (indices, out) |idx, *slot| slot.* = v[idx];
            break :blk .{ .i64 = out };
        },
        .bool => |v| blk: {
            const out = try allocator.alloc(bool, indices.len);
            for (indices, out) |idx, *slot| slot.* = v[idx];
            break :blk .{ .bool = out };
        },
        .string => |v| blk: {
            var out = try allocator.alloc([]const u8, indices.len);
            errdefer allocator.free(out);
            var initialized: usize = 0;
            errdefer {
                for (out[0..initialized]) |s| allocator.free(s);
            }
            for (indices, out) |idx, *slot| {
                slot.* = try allocator.dupe(u8, v[idx]);
                initialized += 1;
            }
            break :blk .{ .string = out };
        },
    };
}

fn describeF64(allocator: std.mem.Allocator, v: []const f64) DataError![]f64 {
    var out = try allocator.alloc(f64, 4);
    if (v.len == 0) {
        @memset(out, std.math.nan(f64));
        out[0] = 0;
        return out;
    }
    var total: f64 = 0;
    var min_v = v[0];
    var max_v = v[0];
    for (v) |x| {
        total += x;
        if (x < min_v) min_v = x;
        if (x > max_v) max_v = x;
    }
    out[0] = @floatFromInt(v.len);
    out[1] = total / @as(f64, @floatFromInt(v.len));
    out[2] = min_v;
    out[3] = max_v;
    return out;
}

fn describeI64(allocator: std.mem.Allocator, v: []const i64) DataError![]f64 {
    const tmp = try allocator.alloc(f64, v.len);
    defer allocator.free(tmp);
    for (v, tmp) |x, *slot| slot.* = @floatFromInt(x);
    return describeF64(allocator, tmp);
}

fn splitCsvLineOwned(allocator: std.mem.Allocator, line: []const u8, out: *std.ArrayList([]const u8)) DataError!void {
    var it = std.mem.splitScalar(u8, line, ',');
    while (it.next()) |cell| {
        try out.append(allocator, try allocator.dupe(u8, std.mem.trim(u8, cell, " \t\r\"")));
    }
}

fn inferColumn(allocator: std.mem.Allocator, cells: []const []const u8) DataError!Column {
    const i64_values = try allocator.alloc(i64, cells.len);
    var all_i64 = true;
    for (cells, i64_values) |cell, *slot| {
        slot.* = std.fmt.parseInt(i64, cell, 10) catch {
            all_i64 = false;
            break;
        };
    }
    if (all_i64) return .{ .i64 = i64_values };
    allocator.free(i64_values);

    const f64_values = try allocator.alloc(f64, cells.len);
    var all_f64 = true;
    for (cells, f64_values) |cell, *slot| {
        slot.* = std.fmt.parseFloat(f64, cell) catch {
            all_f64 = false;
            break;
        };
    }
    if (all_f64) return .{ .f64 = f64_values };
    allocator.free(f64_values);

    const bool_values = try allocator.alloc(bool, cells.len);
    var all_bool = true;
    for (cells, bool_values) |cell, *slot| {
        if (std.ascii.eqlIgnoreCase(cell, "true")) {
            slot.* = true;
        } else if (std.ascii.eqlIgnoreCase(cell, "false")) {
            slot.* = false;
        } else {
            all_bool = false;
            break;
        }
    }
    if (all_bool) return .{ .bool = bool_values };
    allocator.free(bool_values);

    var strings = try allocator.alloc([]const u8, cells.len);
    errdefer allocator.free(strings);
    var initialized: usize = 0;
    errdefer {
        for (strings[0..initialized]) |s| allocator.free(s);
    }
    for (cells, strings) |cell, *slot| {
        slot.* = try allocator.dupe(u8, cell);
        initialized += 1;
    }
    return .{ .string = strings };
}

fn printCell(writer: *std.Io.Writer, col: Column, row: usize) std.Io.Writer.Error!void {
    switch (col) {
        .f64 => |v| try writer.print("{}", .{v[row]}),
        .i64 => |v| try writer.print("{}", .{v[row]}),
        .bool => |v| try writer.print("{}", .{v[row]}),
        .string => |v| try writer.print("{s}", .{v[row]}),
    }
}

pub fn dataframe(allocator: std.mem.Allocator, defs: []const ColumnDef) DataError!DataFrame {
    return DataFrame.init(allocator, defs);
}

pub fn deviceDataFrame(allocator: std.mem.Allocator, defs: []const DeviceColumnDef) DeviceDataError!DeviceDataFrame {
    return DeviceDataFrame.init(allocator, defs);
}

test "dataframe select filter groupby and csv" {
    const gpa = std.testing.allocator;
    var df = try DataFrame.init(gpa, &.{
        .{ .name = "city", .data = .{ .string = &.{ "hz", "bj", "hz" } } },
        .{ .name = "sales", .data = .{ .f64 = &.{ 2.0, 3.0, 5.0 } } },
        .{ .name = "units", .data = .{ .i64 = &.{ 1, 2, 3 } } },
    });
    defer df.deinit();
    try std.testing.expectEqual(@as(usize, 3), df.height());
    var filtered = try df.filter(&.{ true, false, true });
    defer filtered.deinit();
    try std.testing.expectEqual(@as(usize, 2), filtered.height());
    var grouped = try df.groupBySum("city", "sales");
    defer grouped.deinit();
    try std.testing.expectEqual(@as(usize, 2), grouped.height());
    var desc = try df.describe();
    defer desc.deinit();
    try std.testing.expectEqual(@as(usize, 4), desc.height());
    const csv = try df.writeCsv(gpa);
    defer gpa.free(csv);
    var parsed = try DataFrame.readCsv(gpa, csv, true);
    defer parsed.deinit();
    try std.testing.expectEqual(df.height(), parsed.height());
}

test "device dataframe owns fixed-width columns on a shared device" {
    const gpa = std.testing.allocator;

    var sales = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 3.0, 5.0 }, .cpu);
    defer sales.deinit();
    var units = try DeviceColumn.fromSliceWithValidity(i64, gpa, &.{ 1, 2, 3 }, &.{ true, false, true }, .cpu);
    defer units.deinit();
    var active = try DeviceColumn.fromSlice(bool, gpa, &.{ true, false, true }, .cpu);
    defer active.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "sales", .data = sales },
        .{ .name = "units", .data = units },
        .{ .name = "active", .data = active },
    });
    defer table.deinit();

    try std.testing.expectEqual(@as(usize, 3), table.height());
    try std.testing.expectEqual(@as(usize, 3), table.width());
    try std.testing.expect(table.device.isCpu());
    try std.testing.expectEqual(DeviceDType.i64, try table.columnDType("units"));

    const units_col = try table.column("units");
    try std.testing.expect(units_col.nullable());
    try std.testing.expect(units_col.hasNulls());
    try std.testing.expectEqual(@as(usize, 1), units_col.nullCount());

    var view = try table.view();
    defer view.deinit();
    try std.testing.expectEqual(@as(usize, 3), view.height());
    try std.testing.expectEqual(DeviceDType.f64, view.columns[0].dtype);
    try std.testing.expectEqual(DeviceValidityEncoding.bool_mask, view.columns[1].validity_encoding);
    try std.testing.expect(view.columns[0].data_ptr != 0);

    var selected = try table.select(&.{"sales"});
    defer selected.deinit();
    try std.testing.expectEqual(@as(usize, 1), selected.width());
    try std.testing.expectEqual(DeviceDType.f64, try selected.columnDType("sales"));

    var head = try table.head(2);
    defer head.deinit();
    try std.testing.expectEqual(@as(usize, 2), head.height());
    const head_units = try head.column("units");
    try std.testing.expectEqual(@as(usize, 1), head_units.nullCount());

    var filtered = try table.filter(&.{ true, false, true });
    defer filtered.deinit();
    try std.testing.expectEqual(@as(usize, 2), filtered.height());
    const filtered_units = try filtered.column("units");
    try std.testing.expectEqual(@as(usize, 0), filtered_units.nullCount());
}

test "device dataframe round-trips legacy dataframe fixed-width columns" {
    const gpa = std.testing.allocator;
    var legacy = try DataFrame.init(gpa, &.{
        .{ .name = "sales", .data = .{ .f64 = &.{ 2.0, 3.0, 5.0 } } },
        .{ .name = "units", .data = .{ .i64 = &.{ 1, 2, 3 } } },
        .{ .name = "active", .data = .{ .bool = &.{ true, false, true } } },
    });
    defer legacy.deinit();

    var device_table = try DeviceDataFrame.fromDataFrame(gpa, legacy, .cpu);
    defer device_table.deinit();
    try std.testing.expectEqual(@as(usize, 3), device_table.height());
    try std.testing.expectEqual(DeviceDType.f64, try device_table.columnDType("sales"));

    var roundtrip = try device_table.toDataFrame();
    defer roundtrip.deinit();
    try std.testing.expectEqual(legacy.height(), roundtrip.height());
    try std.testing.expectEqualSlices(f64, legacy.columns[0].f64, roundtrip.columns[0].f64);
    try std.testing.expectEqualSlices(i64, legacy.columns[1].i64, roundtrip.columns[1].i64);
    try std.testing.expectEqualSlices(bool, legacy.columns[2].bool, roundtrip.columns[2].bool);
}

test "device dataframe exports boltha arrow record batch" {
    const gpa = std.testing.allocator;

    var sales = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 3.0, 5.0 }, .cpu);
    defer sales.deinit();
    var units = try DeviceColumn.fromSliceWithValidity(i64, gpa, &.{ 1, 2, 3 }, &.{ true, false, true }, .cpu);
    defer units.deinit();
    var active = try DeviceColumn.fromSlice(bool, gpa, &.{ true, false, true }, .cpu);
    defer active.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "sales", .data = sales },
        .{ .name = "units", .data = units },
        .{ .name = "active", .data = active },
    });
    defer table.deinit();

    var schema = try table.toArrowSchema(gpa);
    defer schema.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 3), schema.fieldCount());
    try std.testing.expectEqual(@as(?usize, 0), schema.fieldIndexByName("sales"));
    try std.testing.expect(schema.fields[0].data_type.eql(.{ .floating_point = .double }));
    try std.testing.expect(schema.fields[1].nullable);
    try std.testing.expect(schema.fields[1].data_type.eql(.{ .int = .{ .bit_width = 64, .signed = true } }));
    try std.testing.expect(schema.fields[2].data_type.eql(.bool));

    var batch = try table.toArrowRecordBatch(gpa);
    defer batch.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 3), batch.row_count);
    try std.testing.expectEqual(@as(usize, 3), batch.columnCount());
    try std.testing.expectEqual(@as(?f64, 2.0), batch.columns[0].float64.value(0));
    try std.testing.expectEqual(@as(?i64, 1), batch.columns[1].int64.value(0));
    try std.testing.expectEqual(@as(?i64, null), batch.columns[1].int64.value(1));
    try std.testing.expectEqual(@as(?bool, true), batch.columns[2].boolean.value(0));
    try std.testing.expectEqual(@as(usize, 1), batch.columns[1].nullCount());

    var arrow_table = try table.toArrowTable(gpa);
    defer arrow_table.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 1), arrow_table.batchCount());
    try std.testing.expectEqual(@as(usize, 3), arrow_table.row_count);
    try std.testing.expectEqual(@as(?usize, 1), arrow_table.columnIndexByName("units"));
}

test "device dataframe eager column expressions and boolean mask filtering" {
    const gpa = std.testing.allocator;

    var sales = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 3.0, 5.0 }, .cpu);
    defer sales.deinit();
    var cost = try DeviceColumn.fromSlice(f64, gpa, &.{ 1.0, 1.5, 2.0 }, .cpu);
    defer cost.deinit();
    var units = try DeviceColumn.fromSliceWithValidity(i64, gpa, &.{ 1, 2, 3 }, &.{ true, false, true }, .cpu);
    defer units.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "sales", .data = sales },
        .{ .name = "cost", .data = cost },
        .{ .name = "units", .data = units },
    });
    defer table.deinit();

    var margin = try table.subColumns("sales", "cost");
    defer margin.deinit();
    const margin_values = try margin.f64.toOwnedSlice(gpa);
    defer gpa.free(margin_values);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 1.5, 3.0 }, margin_values);

    var doubled = try table.binaryColumnScalar("sales", f64, 2.0, .mul);
    defer doubled.deinit();
    const doubled_values = try doubled.f64.toOwnedSlice(gpa);
    defer gpa.free(doubled_values);
    try std.testing.expectEqualSlices(f64, &.{ 4.0, 6.0, 10.0 }, doubled_values);

    var mask = try table.compareColumnScalar("sales", f64, 2.5, .gt);
    defer mask.deinit();
    try std.testing.expectEqual(DeviceDType.bool, mask.dtype());
    const mask_values = try mask.bool.toOwnedSlice(gpa);
    defer gpa.free(mask_values);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true }, mask_values);

    var filtered = try table.filterColumnMask(mask);
    defer filtered.deinit();
    try std.testing.expectEqual(@as(usize, 2), filtered.height());
    const filtered_sales = try filtered.column("sales");
    const filtered_sales_values = try filtered_sales.f64.toOwnedSlice(gpa);
    defer gpa.free(filtered_sales_values);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 5.0 }, filtered_sales_values);

    var units_mask = try table.compareColumnScalar("units", i64, 1, .gt);
    defer units_mask.deinit();
    try std.testing.expectEqual(@as(usize, 1), units_mask.bool.null_count);
    var nullable_mask_filtered = try table.filterColumnMask(units_mask);
    defer nullable_mask_filtered.deinit();
    try std.testing.expectEqual(@as(usize, 1), nullable_mask_filtered.height());
    const nullable_mask_sales = try (try nullable_mask_filtered.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(nullable_mask_sales);
    try std.testing.expectEqualSlices(f64, &.{5.0}, nullable_mask_sales);
}

test "device dataframe sorts by device column keys" {
    const gpa = std.testing.allocator;

    var score = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 3.0, 1.0, 2.0, 4.0 }, &.{ true, true, false, true }, .cpu);
    defer score.deinit();
    var id = try DeviceColumn.fromSlice(i64, gpa, &.{ 30, 10, 20, 40 }, .cpu);
    defer id.deinit();
    var flag = try DeviceColumn.fromSlice(bool, gpa, &.{ true, false, true, false }, .cpu);
    defer flag.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "score", .data = score },
        .{ .name = "id", .data = id },
        .{ .name = "flag", .data = flag },
    });
    defer table.deinit();

    const asc = try table.argsortBy("score", .{ .descending = false, .nulls = .last });
    defer gpa.free(asc);
    try std.testing.expectEqualSlices(usize, &.{ 1, 0, 3, 2 }, asc);

    var sorted = try table.sortBy("score", .{ .descending = false, .nulls = .last });
    defer sorted.deinit();
    const sorted_id = try sorted.column("id");
    const sorted_id_values = try sorted_id.i64.toOwnedSlice(gpa);
    defer gpa.free(sorted_id_values);
    try std.testing.expectEqualSlices(i64, &.{ 10, 30, 40, 20 }, sorted_id_values);

    var desc_nulls_first = try table.sortBy("score", .{ .descending = true, .nulls = .first });
    defer desc_nulls_first.deinit();
    const desc_id = try desc_nulls_first.column("id");
    const desc_id_values = try desc_id.i64.toOwnedSlice(gpa);
    defer gpa.free(desc_id_values);
    try std.testing.expectEqualSlices(i64, &.{ 20, 40, 30, 10 }, desc_id_values);

    var bool_sorted = try table.sortBy("flag", .{});
    defer bool_sorted.deinit();
    const bool_sorted_id = try bool_sorted.column("id");
    const bool_sorted_id_values = try bool_sorted_id.i64.toOwnedSlice(gpa);
    defer gpa.free(bool_sorted_id_values);
    try std.testing.expectEqualSlices(i64, &.{ 10, 40, 30, 20 }, bool_sorted_id_values);

    var tied_score = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 10.0, 20.0, 20.0, 30.0, 0.0 }, &.{ true, true, true, true, false }, .cpu);
    defer tied_score.deinit();
    var tied_id = try DeviceColumn.fromSlice(i64, gpa, &.{ 1, 2, 3, 4, 5 }, .cpu);
    defer tied_id.deinit();
    var tied_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "score", .data = tied_score },
        .{ .name = "id", .data = tied_id },
    });
    defer tied_table.deinit();

    var ranks = try tied_table.rankProfileBy("score", "score", .{ .descending = false, .nulls = .last });
    defer ranks.deinit();
    try std.testing.expectEqual(@as(usize, 7), ranks.width());
    const ordinal = try (try ranks.column("score_ordinal_rank")).i64.toOwnedSlice(gpa);
    defer gpa.free(ordinal);
    const competition = try (try ranks.column("score_competition_rank")).i64.toOwnedSlice(gpa);
    defer gpa.free(competition);
    const dense_rank = try (try ranks.column("score_dense_rank")).i64.toOwnedSlice(gpa);
    defer gpa.free(dense_rank);
    const percent_rank = try (try ranks.column("score_percent_rank")).f64.toOwnedSlice(gpa);
    defer gpa.free(percent_rank);
    const cume_dist = try (try ranks.column("score_cume_dist")).f64.toOwnedSlice(gpa);
    defer gpa.free(cume_dist);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4, 5 }, ordinal);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 2, 4, 5 }, competition);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 2, 3, 4 }, dense_rank);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), percent_rank[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), percent_rank[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), percent_rank[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), percent_rank[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), percent_rank[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.2), cume_dist[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.6), cume_dist[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.6), cume_dist[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.8), cume_dist[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), cume_dist[4], 1e-12);

    var desc_ranks = try tied_table.rankProfileBy("score", "score_desc", .{ .descending = true, .nulls = .first });
    defer desc_ranks.deinit();
    const desc_competition = try (try desc_ranks.column("score_desc_competition_rank")).i64.toOwnedSlice(gpa);
    defer gpa.free(desc_competition);
    const desc_cume_dist = try (try desc_ranks.column("score_desc_cume_dist")).f64.toOwnedSlice(gpa);
    defer gpa.free(desc_cume_dist);
    try std.testing.expectEqualSlices(i64, &.{ 5, 3, 3, 2, 1 }, desc_competition);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), desc_cume_dist[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.8), desc_cume_dist[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.8), desc_cume_dist[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.4), desc_cume_dist[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.2), desc_cume_dist[4], 1e-12);

    var rolling_sales = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, 2.0, 100.0, 4.0, 5.0 }, &.{ true, true, false, true, true }, .cpu);
    defer rolling_sales.deinit();
    var rolling_id = try DeviceColumn.fromSlice(i64, gpa, &.{ 1, 2, 3, 4, 5 }, .cpu);
    defer rolling_id.deinit();
    var rolling_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "sales", .data = rolling_sales },
        .{ .name = "id", .data = rolling_id },
    });
    defer rolling_table.deinit();

    var rolling = try rolling_table.rollingProfile("sales", "sales", .{ .window = 3, .min_periods = 2 });
    defer rolling.deinit();
    try std.testing.expectEqual(@as(usize, 7), rolling.width());
    const rolling_count = try (try rolling.column("sales_rolling_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(rolling_count);
    const rolling_sum = try (try rolling.column("sales_rolling_sum")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_sum);
    const rolling_mean = try (try rolling.column("sales_rolling_mean")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_mean);
    const rolling_variance = try (try rolling.column("sales_rolling_variance")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_variance);
    const rolling_stddev = try (try rolling.column("sales_rolling_stddev")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_stddev);
    const rolling_validity = try (try rolling.column("sales_rolling_mean")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(rolling_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 2, 2, 2 }, rolling_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true }, rolling_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), rolling_sum[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), rolling_sum[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 6.0), rolling_sum[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0), rolling_sum[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), rolling_mean[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), rolling_mean[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), rolling_mean[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.5), rolling_mean[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), rolling_variance[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), rolling_variance[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_variance[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), rolling_variance[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_stddev[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_stddev[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_stddev[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_stddev[4], 1e-12);

    var ema = try rolling_table.emaProfile("sales", "sales", .{ .alpha = 0.5, .min_periods = 2 });
    defer ema.deinit();
    try std.testing.expectEqual(@as(usize, 5), ema.width());
    const ema_values = try (try ema.column("sales_ema")).f64.toOwnedSlice(gpa);
    defer gpa.free(ema_values);
    const ema_residual = try (try ema.column("sales_ema_residual")).f64.toOwnedSlice(gpa);
    defer gpa.free(ema_residual);
    const ema_ratio = try (try ema.column("sales_ema_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(ema_ratio);
    const ema_validity = try (try ema.column("sales_ema")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(ema_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true, true }, ema_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), ema_values[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.75), ema_values[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.875), ema_values[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), ema_residual[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.25), ema_residual[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.125), ema_residual[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 1.5), ema_ratio[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0 / 2.75), ema_ratio[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 3.875), ema_ratio[4], 1e-12);

    var rolling_range = try rolling_table.rollingRangeProfile("sales", "sales", .{ .window = 3, .min_periods = 2 });
    defer rolling_range.deinit();
    try std.testing.expectEqual(@as(usize, 6), rolling_range.width());
    const rolling_low = try (try rolling_range.column("sales_rolling_low")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_low);
    const rolling_high = try (try rolling_range.column("sales_rolling_high")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_high);
    const rolling_spread = try (try rolling_range.column("sales_rolling_range")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_spread);
    const rolling_position = try (try rolling_range.column("sales_rolling_position")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_position);
    const rolling_range_validity = try (try rolling_range.column("sales_rolling_range")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(rolling_range_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true, true }, rolling_range_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_low[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), rolling_low[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), rolling_low[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), rolling_high[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), rolling_high[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), rolling_high[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_spread[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), rolling_spread[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_spread[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_position[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_position[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_position[4], 1e-12);

    var rolling_normalized = try rolling_table.rollingNormalizeProfile("sales", "sales", .{ .window = 3, .min_periods = 2 });
    defer rolling_normalized.deinit();
    try std.testing.expectEqual(@as(usize, 5), rolling_normalized.width());
    const rolling_centered = try (try rolling_normalized.column("sales_rolling_centered")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_centered);
    const rolling_zscore = try (try rolling_normalized.column("sales_rolling_zscore")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_zscore);
    const rolling_minmax = try (try rolling_normalized.column("sales_rolling_minmax")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_minmax);
    const rolling_norm_validity = try (try rolling_normalized.column("sales_rolling_zscore")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(rolling_norm_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true, true }, rolling_norm_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_centered[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_centered[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_centered[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_zscore[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_zscore[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_zscore[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_minmax[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_minmax[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_minmax[4], 1e-12);

    var rolling_quantiles = try rolling_table.rollingQuantileProfile("sales", "sales", .{ .window = 3, .min_periods = 2 });
    defer rolling_quantiles.deinit();
    try std.testing.expectEqual(@as(usize, 6), rolling_quantiles.width());
    const rolling_q1 = try (try rolling_quantiles.column("sales_rolling_q1")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_q1);
    const rolling_median = try (try rolling_quantiles.column("sales_rolling_median")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_median);
    const rolling_q3 = try (try rolling_quantiles.column("sales_rolling_q3")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_q3);
    const rolling_iqr = try (try rolling_quantiles.column("sales_rolling_iqr")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_iqr);
    const rolling_quantile_validity = try (try rolling_quantiles.column("sales_rolling_median")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(rolling_quantile_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true, true }, rolling_quantile_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.25), rolling_q1[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), rolling_q1[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.25), rolling_q1[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), rolling_median[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), rolling_median[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.5), rolling_median[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.75), rolling_q3[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.5), rolling_q3[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.75), rolling_q3[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_iqr[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_iqr[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_iqr[4], 1e-12);

    var lag_source = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 10.0, 0.0, 15.0, 20.0, 99.0 }, &.{ true, true, true, true, false }, .cpu);
    defer lag_source.deinit();
    var lag_id = try DeviceColumn.fromSlice(i64, gpa, &.{ 1, 2, 3, 4, 5 }, .cpu);
    defer lag_id.deinit();
    var lag_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "sales", .data = lag_source },
        .{ .name = "id", .data = lag_id },
    });
    defer lag_table.deinit();

    var lagged = try lag_table.lagProfile("sales", "sales", .{ .periods = 2 });
    defer lagged.deinit();
    try std.testing.expectEqual(@as(usize, 5), lagged.width());
    const lag_values = try (try lagged.column("sales_lag")).f64.toOwnedSlice(gpa);
    defer gpa.free(lag_values);
    const diff_values = try (try lagged.column("sales_diff")).f64.toOwnedSlice(gpa);
    defer gpa.free(diff_values);
    const pct_values = try (try lagged.column("sales_pct_change")).f64.toOwnedSlice(gpa);
    defer gpa.free(pct_values);
    const lag_validity = try (try lagged.column("sales_lag")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lag_validity);
    const diff_validity = try (try lagged.column("sales_diff")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(diff_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, true, true }, lag_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, true, false }, diff_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), lag_values[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lag_values[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 15.0), lag_values[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), diff_values[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), diff_values[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), pct_values[2], 1e-12);
    try std.testing.expect(std.math.isNan(pct_values[3]));

    var leaded = try lag_table.leadProfile("sales", "sales", .{ .periods = 2 });
    defer leaded.deinit();
    try std.testing.expectEqual(@as(usize, 5), leaded.width());
    const lead_values = try (try leaded.column("sales_lead")).f64.toOwnedSlice(gpa);
    defer gpa.free(lead_values);
    const forward_diff = try (try leaded.column("sales_forward_diff")).f64.toOwnedSlice(gpa);
    defer gpa.free(forward_diff);
    const forward_pct = try (try leaded.column("sales_forward_pct_change")).f64.toOwnedSlice(gpa);
    defer gpa.free(forward_pct);
    const lead_validity = try (try leaded.column("sales_lead")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lead_validity);
    const forward_validity = try (try leaded.column("sales_forward_diff")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(forward_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false, false }, lead_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false, false }, forward_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 15.0), lead_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), lead_values[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), forward_diff[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), forward_diff[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), forward_pct[0], 1e-12);
    try std.testing.expect(std.math.isNan(forward_pct[1]));

    var expanding = try lag_table.expandingProfile("sales", "sales", .{ .min_periods = 2 });
    defer expanding.deinit();
    try std.testing.expectEqual(@as(usize, 7), expanding.width());
    const expanding_count = try (try expanding.column("sales_expanding_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(expanding_count);
    const expanding_sum = try (try expanding.column("sales_expanding_sum")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_sum);
    const expanding_mean = try (try expanding.column("sales_expanding_mean")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_mean);
    const expanding_min = try (try expanding.column("sales_expanding_min")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_min);
    const expanding_max = try (try expanding.column("sales_expanding_max")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_max);
    const expanding_validity = try (try expanding.column("sales_expanding_mean")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(expanding_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4, 4 }, expanding_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true }, expanding_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), expanding_sum[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 25.0), expanding_sum[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 45.0), expanding_sum[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 45.0), expanding_sum[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), expanding_mean[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 25.0 / 3.0), expanding_mean[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 11.25), expanding_mean[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 11.25), expanding_mean[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), expanding_min[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), expanding_min[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), expanding_max[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), expanding_max[4], 1e-12);

    var clipped = try lag_table.clipProfile("sales", "sales", .{ .lower = 5.0, .upper = 15.0 });
    defer clipped.deinit();
    try std.testing.expectEqual(@as(usize, 6), clipped.width());
    const clipped_values = try (try clipped.column("sales_clipped")).f64.toOwnedSlice(gpa);
    defer gpa.free(clipped_values);
    const below_values = try (try clipped.column("sales_below")).bool.toOwnedSlice(gpa);
    defer gpa.free(below_values);
    const above_values = try (try clipped.column("sales_above")).bool.toOwnedSlice(gpa);
    defer gpa.free(above_values);
    const in_range_values = try (try clipped.column("sales_in_range")).bool.toOwnedSlice(gpa);
    defer gpa.free(in_range_values);
    const clip_validity = try (try clipped.column("sales_clipped")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(clip_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, false }, clip_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), clipped_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), clipped_values[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 15.0), clipped_values[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 15.0), clipped_values[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false, false }, below_values);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, true, false }, above_values);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, false, false }, in_range_values);

    var thresholded = try lag_table.thresholdProfile("sales", "sales", .{ .threshold = 10.0 });
    defer thresholded.deinit();
    try std.testing.expectEqual(@as(usize, 7), thresholded.width());
    const threshold_distance = try (try thresholded.column("sales_distance")).f64.toOwnedSlice(gpa);
    defer gpa.free(threshold_distance);
    const threshold_abs_distance = try (try thresholded.column("sales_abs_distance")).f64.toOwnedSlice(gpa);
    defer gpa.free(threshold_abs_distance);
    const threshold_above = try (try thresholded.column("sales_above")).bool.toOwnedSlice(gpa);
    defer gpa.free(threshold_above);
    const threshold_below = try (try thresholded.column("sales_below")).bool.toOwnedSlice(gpa);
    defer gpa.free(threshold_below);
    const threshold_at = try (try thresholded.column("sales_at")).bool.toOwnedSlice(gpa);
    defer gpa.free(threshold_at);
    const threshold_validity = try (try thresholded.column("sales_distance")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(threshold_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, false }, threshold_validity);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, -10.0, 5.0, 10.0, 0.0 }, threshold_distance);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 10.0, 5.0, 10.0, 0.0 }, threshold_abs_distance);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, true, false }, threshold_above);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false, false }, threshold_below);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false, false }, threshold_at);

    var scaled = try lag_table.standardizeProfile("sales", "sales", .{ .min_periods = 3 });
    defer scaled.deinit();
    try std.testing.expectEqual(@as(usize, 5), scaled.width());
    const centered = try (try scaled.column("sales_centered")).f64.toOwnedSlice(gpa);
    defer gpa.free(centered);
    const zscore = try (try scaled.column("sales_zscore")).f64.toOwnedSlice(gpa);
    defer gpa.free(zscore);
    const minmax = try (try scaled.column("sales_minmax")).f64.toOwnedSlice(gpa);
    defer gpa.free(minmax);
    const scaled_validity = try (try scaled.column("sales_zscore")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(scaled_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, false }, scaled_validity);
    try std.testing.expectApproxEqAbs(@as(f64, -1.25), centered[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -11.25), centered[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.75), centered[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 8.75), centered[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.1690308509457033), zscore[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -1.5212776585113297), zscore[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.50709255283711), zscore[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.1832159566199232), zscore[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), minmax[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), minmax[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), minmax[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), minmax[3], 1e-12);

    var robust_source = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, 2.0, 3.0, 100.0, 0.0 }, &.{ true, true, true, true, false }, .cpu);
    defer robust_source.deinit();
    var robust_id = try DeviceColumn.fromSlice(i64, gpa, &.{ 1, 2, 3, 4, 5 }, .cpu);
    defer robust_id.deinit();
    var robust_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "value", .data = robust_source },
        .{ .name = "id", .data = robust_id },
    });
    defer robust_table.deinit();

    var robust = try robust_table.robustProfile("value", "value", .{ .min_periods = 4 });
    defer robust.deinit();
    try std.testing.expectEqual(@as(usize, 6), robust.width());
    const median_centered = try (try robust.column("value_median_centered")).f64.toOwnedSlice(gpa);
    defer gpa.free(median_centered);
    const mad_zscore = try (try robust.column("value_mad_zscore")).f64.toOwnedSlice(gpa);
    defer gpa.free(mad_zscore);
    const iqr_outlier = try (try robust.column("value_iqr_outlier")).bool.toOwnedSlice(gpa);
    defer gpa.free(iqr_outlier);
    const winsorized = try (try robust.column("value_winsorized")).f64.toOwnedSlice(gpa);
    defer gpa.free(winsorized);
    const robust_validity = try (try robust.column("value_mad_zscore")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(robust_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, false }, robust_validity);
    try std.testing.expectApproxEqAbs(@as(f64, -1.5), median_centered[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.5), median_centered[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), median_centered[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 97.5), median_centered[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -1.0117346252941226), mad_zscore[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.33724487509804085), mad_zscore[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.33724487509804085), mad_zscore[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 65.76275064411797), mad_zscore[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, true, false }, iqr_outlier);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), winsorized[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), winsorized[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), winsorized[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 65.5), winsorized[3], 1e-12);

    var equity = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 100.0, 120.0, 90.0, 130.0, 80.0, 0.0 }, &.{ true, true, true, true, true, false }, .cpu);
    defer equity.deinit();
    var equity_id = try DeviceColumn.fromSlice(i64, gpa, &.{ 1, 2, 3, 4, 5, 6 }, .cpu);
    defer equity_id.deinit();
    var equity_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "equity", .data = equity },
        .{ .name = "id", .data = equity_id },
    });
    defer equity_table.deinit();

    var drawdown = try equity_table.drawdownProfile("equity", "equity", .{ .min_periods = 2 });
    defer drawdown.deinit();
    try std.testing.expectEqual(@as(usize, 5), drawdown.width());
    const running_peak = try (try drawdown.column("equity_running_peak")).f64.toOwnedSlice(gpa);
    defer gpa.free(running_peak);
    const drawdown_values = try (try drawdown.column("equity_drawdown")).f64.toOwnedSlice(gpa);
    defer gpa.free(drawdown_values);
    const drawdown_pct = try (try drawdown.column("equity_drawdown_pct")).f64.toOwnedSlice(gpa);
    defer gpa.free(drawdown_pct);
    const drawdown_validity = try (try drawdown.column("equity_drawdown")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(drawdown_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true, false }, drawdown_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 120.0), running_peak[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 120.0), running_peak[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 130.0), running_peak[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 130.0), running_peak[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), drawdown_values[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -30.0), drawdown_values[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), drawdown_values[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -50.0), drawdown_values[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), drawdown_pct[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.25), drawdown_pct[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), drawdown_pct[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -50.0 / 130.0), drawdown_pct[4], 1e-12);

    var extrema = try equity_table.extremaProfile("equity", "equity", .{ .min_periods = 2 });
    defer extrema.deinit();
    try std.testing.expectEqual(@as(usize, 6), extrema.width());
    const running_low = try (try extrema.column("equity_running_low")).f64.toOwnedSlice(gpa);
    defer gpa.free(running_low);
    const running_high = try (try extrema.column("equity_running_high")).f64.toOwnedSlice(gpa);
    defer gpa.free(running_high);
    const new_low = try (try extrema.column("equity_new_low")).bool.toOwnedSlice(gpa);
    defer gpa.free(new_low);
    const new_high = try (try extrema.column("equity_new_high")).bool.toOwnedSlice(gpa);
    defer gpa.free(new_high);
    const extrema_validity = try (try extrema.column("equity_running_low")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(extrema_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true, false }, extrema_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 100.0), running_low[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 90.0), running_low[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 90.0), running_low[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 80.0), running_low[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 120.0), running_high[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 120.0), running_high[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 130.0), running_high[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 130.0), running_high[4], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, true, false }, new_low);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true, false, false }, new_high);

    var trend_source = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, 3.0, 2.0, 2.0, 5.0, 0.0, 4.0 }, &.{ true, true, true, true, true, false, true }, .cpu);
    defer trend_source.deinit();
    var trend_id = try DeviceColumn.fromSlice(i64, gpa, &.{ 1, 2, 3, 4, 5, 6, 7 }, .cpu);
    defer trend_id.deinit();
    var trend_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "price", .data = trend_source },
        .{ .name = "id", .data = trend_id },
    });
    defer trend_table.deinit();

    var trend = try trend_table.trendProfile("price", "price", .{ .periods = 1 });
    defer trend.deinit();
    try std.testing.expectEqual(@as(usize, 7), trend.width());
    const trend_values = try (try trend.column("price_trend")).i64.toOwnedSlice(gpa);
    defer gpa.free(trend_values);
    const up_streak = try (try trend.column("price_up_streak")).i64.toOwnedSlice(gpa);
    defer gpa.free(up_streak);
    const down_streak = try (try trend.column("price_down_streak")).i64.toOwnedSlice(gpa);
    defer gpa.free(down_streak);
    const flat_streak = try (try trend.column("price_flat_streak")).i64.toOwnedSlice(gpa);
    defer gpa.free(flat_streak);
    const reversal = try (try trend.column("price_reversal")).bool.toOwnedSlice(gpa);
    defer gpa.free(reversal);
    const trend_validity = try (try trend.column("price_trend")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(trend_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true, false, false }, trend_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, -1, 0, 1, 0, 0 }, trend_values);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 0, 1, 0, 0 }, up_streak);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 0, 0, 0, 0 }, down_streak);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 1, 0, 0, 0 }, flat_streak);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, true, false, false }, reversal);

    var signed_values_col = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ -1.0, -2.0, 0.0, 3.0, -4.0, 0.0, 5.0 }, &.{ true, true, true, true, true, false, true }, .cpu);
    defer signed_values_col.deinit();
    var signed_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "signal", .data = signed_values_col },
    });
    defer signed_table.deinit();
    var sign = try signed_table.signProfile("signal", "signal", .{ .periods = 1 });
    defer sign.deinit();
    try std.testing.expectEqual(@as(usize, 6), sign.width());
    const sign_values = try (try sign.column("signal_sign")).i64.toOwnedSlice(gpa);
    defer gpa.free(sign_values);
    const sign_flip = try (try sign.column("signal_sign_flip")).bool.toOwnedSlice(gpa);
    defer gpa.free(sign_flip);
    const positive_streak = try (try sign.column("signal_positive_streak")).i64.toOwnedSlice(gpa);
    defer gpa.free(positive_streak);
    const negative_streak = try (try sign.column("signal_negative_streak")).i64.toOwnedSlice(gpa);
    defer gpa.free(negative_streak);
    const zero_streak = try (try sign.column("signal_zero_streak")).i64.toOwnedSlice(gpa);
    defer gpa.free(zero_streak);
    const sign_validity = try (try sign.column("signal_sign")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(sign_validity);
    const flip_validity = try (try sign.column("signal_sign_flip")).bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(flip_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, true, false, true }, sign_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true, false, false }, flip_validity);
    try std.testing.expectEqualSlices(i64, &.{ -1, -1, 0, 1, -1, 0, 1 }, sign_values);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, true, true, false, false }, sign_flip);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 1, 0, 0, 1 }, positive_streak);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 0, 0, 1, 0, 0 }, negative_streak);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 0, 0, 0, 0 }, zero_streak);

    var validity = try trend_table.validityProfile("price", "price");
    defer validity.deinit();
    try std.testing.expectEqual(@as(usize, 6), validity.width());
    const is_null = try (try validity.column("price_is_null")).bool.toOwnedSlice(gpa);
    defer gpa.free(is_null);
    const is_valid = try (try validity.column("price_is_valid")).bool.toOwnedSlice(gpa);
    defer gpa.free(is_valid);
    const valid_streak = try (try validity.column("price_valid_streak")).i64.toOwnedSlice(gpa);
    defer gpa.free(valid_streak);
    const null_streak = try (try validity.column("price_null_streak")).i64.toOwnedSlice(gpa);
    defer gpa.free(null_streak);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false, false, true, false }, is_null);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, true, false, true }, is_valid);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4, 5, 0, 1 }, valid_streak);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0, 0, 1, 0 }, null_streak);

    var actual_label = try DeviceColumn.fromSliceWithValidity(bool, gpa, &.{ true, false, true, false, true }, &.{ true, true, true, false, true }, .cpu);
    defer actual_label.deinit();
    var predicted_label = try DeviceColumn.fromSlice(bool, gpa, &.{ true, true, false, false, true }, .cpu);
    defer predicted_label.deinit();
    var label_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "actual", .data = actual_label },
        .{ .name = "predicted", .data = predicted_label },
    });
    defer label_table.deinit();

    var classes = try label_table.classificationProfile("actual", "predicted", "cls");
    defer classes.deinit();
    try std.testing.expectEqual(@as(usize, 7), classes.width());
    const tp = try (try classes.column("cls_tp")).bool.toOwnedSlice(gpa);
    defer gpa.free(tp);
    const fp = try (try classes.column("cls_fp")).bool.toOwnedSlice(gpa);
    defer gpa.free(fp);
    const tn = try (try classes.column("cls_tn")).bool.toOwnedSlice(gpa);
    defer gpa.free(tn);
    const fn_values = try (try classes.column("cls_fn")).bool.toOwnedSlice(gpa);
    defer gpa.free(fn_values);
    const correct = try (try classes.column("cls_correct")).bool.toOwnedSlice(gpa);
    defer gpa.free(correct);
    const class_validity = try (try classes.column("cls_correct")).bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(class_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false, true }, class_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false, true }, tp);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false, false }, fp);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false, false }, tn);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, false }, fn_values);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false, true }, correct);

    var fast = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, 3.0, 2.0, 5.0, 4.0, 6.0 }, &.{ true, true, true, true, false, true }, .cpu);
    defer fast.deinit();
    var slow = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 2.0, 2.0, 4.0, 5.0, 0.0 }, .cpu);
    defer slow.deinit();
    var signal_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "fast", .data = fast },
        .{ .name = "slow", .data = slow },
    });
    defer signal_table.deinit();

    var cross = try signal_table.crossoverProfile("fast", "slow", "fast_slow", .{ .periods = 1 });
    defer cross.deinit();
    try std.testing.expectEqual(@as(usize, 6), cross.width());
    const spread = try (try cross.column("fast_slow_spread")).f64.toOwnedSlice(gpa);
    defer gpa.free(spread);
    const ratio = try (try cross.column("fast_slow_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(ratio);
    const cross_above = try (try cross.column("fast_slow_cross_above")).bool.toOwnedSlice(gpa);
    defer gpa.free(cross_above);
    const cross_below = try (try cross.column("fast_slow_cross_below")).bool.toOwnedSlice(gpa);
    defer gpa.free(cross_below);
    const spread_validity = try (try cross.column("fast_slow_spread")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(spread_validity);
    const cross_validity = try (try cross.column("fast_slow_cross_above")).bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(cross_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, false, true }, spread_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, false, false }, cross_validity);
    try std.testing.expectApproxEqAbs(@as(f64, -1.0), spread[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), spread[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), spread[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), spread[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 6.0), spread[5], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), ratio[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), ratio[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), ratio[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.25), ratio[3], 1e-12);
    try std.testing.expect(std.math.isNan(ratio[5]));
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true, false, false }, cross_above);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false, false, false }, cross_below);

    var corr = try signal_table.rollingCorrelationProfile("fast", "slow", "fast_slow", .{ .window = 3, .min_periods = 2 });
    defer corr.deinit();
    try std.testing.expectEqual(@as(usize, 6), corr.width());
    const pair_count = try (try corr.column("fast_slow_rolling_pair_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(pair_count);
    const covariance = try (try corr.column("fast_slow_rolling_covariance")).f64.toOwnedSlice(gpa);
    defer gpa.free(covariance);
    const correlation = try (try corr.column("fast_slow_rolling_correlation")).f64.toOwnedSlice(gpa);
    defer gpa.free(correlation);
    const beta = try (try corr.column("fast_slow_rolling_beta")).f64.toOwnedSlice(gpa);
    defer gpa.free(beta);
    const corr_validity = try (try corr.column("fast_slow_rolling_correlation")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(corr_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 3, 2, 2 }, pair_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true, true }, corr_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), covariance[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), covariance[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.1111111111111107), covariance[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), covariance[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -1.0), covariance[5], 1e-12);
    try std.testing.expect(std.math.isNan(correlation[1]));
    try std.testing.expect(std.math.isNan(correlation[2]));
    try std.testing.expectApproxEqAbs(@as(f64, 0.944911182523068), correlation[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), correlation[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -1.0), correlation[5], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), beta[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.714285714285715), beta[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), beta[4], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -4.0), beta[5], 1e-12);

    var fit_x = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, 2.0, 3.0, 4.0, 5.0 }, &.{ true, true, true, true, false }, .cpu);
    defer fit_x.deinit();
    var fit_y = try DeviceColumn.fromSlice(f64, gpa, &.{ 3.0, 5.0, 8.0, 9.0, 0.0 }, .cpu);
    defer fit_y.deinit();
    var fit_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "x", .data = fit_x },
        .{ .name = "y", .data = fit_y },
    });
    defer fit_table.deinit();

    var fitted_table = try fit_table.linearFitProfile("x", "y", "xy", .{ .min_periods = 3 });
    defer fitted_table.deinit();
    try std.testing.expectEqual(@as(usize, 6), fitted_table.width());
    const fitted = try (try fitted_table.column("xy_fitted")).f64.toOwnedSlice(gpa);
    defer gpa.free(fitted);
    const residual = try (try fitted_table.column("xy_residual")).f64.toOwnedSlice(gpa);
    defer gpa.free(residual);
    const residual_z = try (try fitted_table.column("xy_residual_zscore")).f64.toOwnedSlice(gpa);
    defer gpa.free(residual_z);
    const slope_values = try (try fitted_table.column("xy_slope")).f64.toOwnedSlice(gpa);
    defer gpa.free(slope_values);
    const fit_validity = try (try fitted_table.column("xy_fitted")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(fit_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, false }, fit_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 3.1), fitted[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.2), fitted[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 7.3), fitted[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 9.4), fitted[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.1), residual[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.2), residual[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.7), residual[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.4), residual[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.23904572186687895), residual_z[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.4780914437337579), residual_z[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.6733200530681511), residual_z[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.9561828874675167), residual_z[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.1), slope_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.1), slope_values[3], 1e-12);

    var errors = try fit_table.errorProfile("y", "x", "yx");
    defer errors.deinit();
    try std.testing.expectEqual(@as(usize, 7), errors.width());
    const error_values = try (try errors.column("yx_error")).f64.toOwnedSlice(gpa);
    defer gpa.free(error_values);
    const abs_error_values = try (try errors.column("yx_abs_error")).f64.toOwnedSlice(gpa);
    defer gpa.free(abs_error_values);
    const squared_error_values = try (try errors.column("yx_squared_error")).f64.toOwnedSlice(gpa);
    defer gpa.free(squared_error_values);
    const ape_values = try (try errors.column("yx_ape")).f64.toOwnedSlice(gpa);
    defer gpa.free(ape_values);
    const smape_values = try (try errors.column("yx_smape")).f64.toOwnedSlice(gpa);
    defer gpa.free(smape_values);
    const error_validity = try (try errors.column("yx_error")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(error_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, false }, error_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), error_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), error_values[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), error_values[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), error_values[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), abs_error_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 25.0), squared_error_values[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), ape_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 9.0), ape_values[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), smape_values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0 / 11.0), smape_values[2], 1e-12);

    var bucketed = try signal_table.bucketProfile("fast", "fast", .{ .buckets = 3, .lower_quantile = 0.34, .upper_quantile = 0.84 });
    defer bucketed.deinit();
    try std.testing.expectEqual(@as(usize, 6), bucketed.width());
    const ecdf = try (try bucketed.column("fast_ecdf")).f64.toOwnedSlice(gpa);
    defer gpa.free(ecdf);
    const bucket = try (try bucketed.column("fast_bucket")).i64.toOwnedSlice(gpa);
    defer gpa.free(bucket);
    const lower_tail = try (try bucketed.column("fast_lower_tail")).bool.toOwnedSlice(gpa);
    defer gpa.free(lower_tail);
    const upper_tail = try (try bucketed.column("fast_upper_tail")).bool.toOwnedSlice(gpa);
    defer gpa.free(upper_tail);
    const bucket_validity = try (try bucketed.column("fast_bucket")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(bucket_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, false, true }, bucket_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.2), ecdf[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.6), ecdf[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.4), ecdf[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.8), ecdf[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), ecdf[5], 1e-12);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 1, 0, 2 }, bucket);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false, false, false }, lower_tail);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false, false, true }, upper_tail);
}

test "device dataframe groupby aggregations on fixed-width columns" {
    const gpa = std.testing.allocator;

    var key = try DeviceColumn.fromSliceWithValidity(i32, gpa, &.{ 1, 2, 1, 3, 2, 1 }, &.{ true, true, true, false, true, true }, .cpu);
    defer key.deinit();
    var sales = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 2.0, 3.0, 5.0, 7.0, 11.0, 13.0 }, &.{ true, true, false, true, true, true }, .cpu);
    defer sales.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "store", .data = key },
        .{ .name = "sales", .data = sales },
    });
    defer table.deinit();

    var counted = try table.groupByCount("store", "rows");
    defer counted.deinit();
    try std.testing.expectEqual(@as(usize, 2), counted.height());
    const count_keys = try (try counted.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(count_keys);
    const counts = try (try counted.column("rows")).i64.toOwnedSlice(gpa);
    defer gpa.free(counts);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2 }, count_keys);
    try std.testing.expectEqualSlices(i64, &.{ 3, 2 }, counts);

    var summed = try table.groupBySum("store", "sales", "sales_sum");
    defer summed.deinit();
    try std.testing.expectEqual(@as(usize, 2), summed.height());
    const sum_keys = try (try summed.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(sum_keys);
    const sums = try (try summed.column("sales_sum")).f64.toOwnedSlice(gpa);
    defer gpa.free(sums);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2 }, sum_keys);
    try std.testing.expectEqualSlices(f64, &.{ 15.0, 14.0 }, sums);

    var mins = try table.groupByMin("store", "sales", "sales_min");
    defer mins.deinit();
    const min_values = try (try mins.column("sales_min")).f64.toOwnedSlice(gpa);
    defer gpa.free(min_values);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 3.0 }, min_values);

    var maxes = try table.groupByMax("store", "sales", "sales_max");
    defer maxes.deinit();
    const max_values = try (try maxes.column("sales_max")).f64.toOwnedSlice(gpa);
    defer gpa.free(max_values);
    try std.testing.expectEqualSlices(f64, &.{ 13.0, 11.0 }, max_values);

    var means = try table.groupByMean("store", "sales", "sales_mean");
    defer means.deinit();
    const mean_values = try (try means.column("sales_mean")).f64.toOwnedSlice(gpa);
    defer gpa.free(mean_values);
    try std.testing.expectEqualSlices(f64, &.{ 7.5, 7.0 }, mean_values);

    var stats = try table.groupByStats("store", "sales", "sales");
    defer stats.deinit();
    try std.testing.expectEqual(@as(usize, 6), stats.width());
    const stats_keys = try (try stats.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(stats_keys);
    const stats_counts = try (try stats.column("sales_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(stats_counts);
    const stats_sums = try (try stats.column("sales_sum")).f64.toOwnedSlice(gpa);
    defer gpa.free(stats_sums);
    const stats_mins = try (try stats.column("sales_min")).f64.toOwnedSlice(gpa);
    defer gpa.free(stats_mins);
    const stats_maxes = try (try stats.column("sales_max")).f64.toOwnedSlice(gpa);
    defer gpa.free(stats_maxes);
    const stats_means = try (try stats.column("sales_mean")).f64.toOwnedSlice(gpa);
    defer gpa.free(stats_means);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2 }, stats_keys);
    try std.testing.expectEqualSlices(i64, &.{ 2, 2 }, stats_counts);
    try std.testing.expectEqualSlices(f64, &.{ 15.0, 14.0 }, stats_sums);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 3.0 }, stats_mins);
    try std.testing.expectEqualSlices(f64, &.{ 13.0, 11.0 }, stats_maxes);
    try std.testing.expectEqualSlices(f64, &.{ 7.5, 7.0 }, stats_means);

    var profile = try table.groupByProfile("store", "sales", "sales");
    defer profile.deinit();
    try std.testing.expectEqual(@as(usize, 8), profile.width());
    const profile_keys = try (try profile.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(profile_keys);
    const profile_counts = try (try profile.column("sales_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(profile_counts);
    const profile_sums = try (try profile.column("sales_sum")).f64.toOwnedSlice(gpa);
    defer gpa.free(profile_sums);
    const profile_variances = try (try profile.column("sales_variance")).f64.toOwnedSlice(gpa);
    defer gpa.free(profile_variances);
    const profile_stddevs = try (try profile.column("sales_stddev")).f64.toOwnedSlice(gpa);
    defer gpa.free(profile_stddevs);
    const profile_skewnesses = try (try profile.column("sales_skewness")).f64.toOwnedSlice(gpa);
    defer gpa.free(profile_skewnesses);
    const profile_kurtoses = try (try profile.column("sales_kurtosis")).f64.toOwnedSlice(gpa);
    defer gpa.free(profile_kurtoses);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2 }, profile_keys);
    try std.testing.expectEqualSlices(i64, &.{ 2, 2 }, profile_counts);
    try std.testing.expectEqualSlices(f64, &.{ 15.0, 14.0 }, profile_sums);
    try std.testing.expectApproxEqAbs(@as(f64, 30.25), profile_variances[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 16.0), profile_variances[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.5), profile_stddevs[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), profile_stddevs[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), profile_skewnesses[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), profile_skewnesses[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -2.0), profile_kurtoses[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -2.0), profile_kurtoses[1], 1e-12);

    var keyed = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 1, 1, 2, 2, 2 }, .cpu);
    defer keyed.deinit();
    var day = try DeviceColumn.fromSlice(i32, gpa, &.{ 10, 10, 11, 10, 10, 11 }, .cpu);
    defer day.deinit();
    var amount = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, 2.0, 9.0, 4.0, 6.0, 12.0 }, &.{ true, true, true, true, false, true }, .cpu);
    defer amount.deinit();
    var multi = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "store", .data = keyed },
        .{ .name = "day", .data = day },
        .{ .name = "amount", .data = amount },
    });
    defer multi.deinit();

    var multi_stats = try multi.groupByStatsOn(&.{ "store", "day" }, "amount", "amount");
    defer multi_stats.deinit();
    try std.testing.expectEqual(@as(usize, 7), multi_stats.width());
    try std.testing.expectEqual(@as(usize, 4), multi_stats.height());
    const ms_store = try (try multi_stats.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(ms_store);
    const ms_day = try (try multi_stats.column("day")).i32.toOwnedSlice(gpa);
    defer gpa.free(ms_day);
    const ms_count = try (try multi_stats.column("amount_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(ms_count);
    const ms_sum = try (try multi_stats.column("amount_sum")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_sum);
    const ms_mean = try (try multi_stats.column("amount_mean")).f64.toOwnedSlice(gpa);
    defer gpa.free(ms_mean);
    try std.testing.expectEqualSlices(i32, &.{ 1, 1, 2, 2 }, ms_store);
    try std.testing.expectEqualSlices(i32, &.{ 10, 11, 10, 11 }, ms_day);
    try std.testing.expectEqualSlices(i64, &.{ 2, 1, 1, 1 }, ms_count);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 9.0, 4.0, 12.0 }, ms_sum);
    try std.testing.expectEqualSlices(f64, &.{ 1.5, 9.0, 4.0, 12.0 }, ms_mean);

    var multi_profile = try multi.groupByProfileOn(&.{ "store", "day" }, "amount", "amount");
    defer multi_profile.deinit();
    try std.testing.expectEqual(@as(usize, 9), multi_profile.width());
    try std.testing.expectEqual(@as(usize, 4), multi_profile.height());
    const mp_store = try (try multi_profile.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(mp_store);
    const mp_day = try (try multi_profile.column("day")).i32.toOwnedSlice(gpa);
    defer gpa.free(mp_day);
    const mp_count = try (try multi_profile.column("amount_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(mp_count);
    const mp_variance = try (try multi_profile.column("amount_variance")).f64.toOwnedSlice(gpa);
    defer gpa.free(mp_variance);
    const mp_stddev = try (try multi_profile.column("amount_stddev")).f64.toOwnedSlice(gpa);
    defer gpa.free(mp_stddev);
    const mp_skewness = try (try multi_profile.column("amount_skewness")).f64.toOwnedSlice(gpa);
    defer gpa.free(mp_skewness);
    const mp_kurtosis = try (try multi_profile.column("amount_kurtosis")).f64.toOwnedSlice(gpa);
    defer gpa.free(mp_kurtosis);
    try std.testing.expectEqualSlices(i32, &.{ 1, 1, 2, 2 }, mp_store);
    try std.testing.expectEqualSlices(i32, &.{ 10, 11, 10, 11 }, mp_day);
    try std.testing.expectEqualSlices(i64, &.{ 2, 1, 1, 1 }, mp_count);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), mp_variance[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), mp_stddev[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), mp_skewness[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -2.0), mp_kurtosis[0], 1e-12);
    try std.testing.expect(std.math.isNan(mp_skewness[1]));
    try std.testing.expect(std.math.isNan(mp_kurtosis[1]));
}

test "device dataframe inner joins on fixed-width keys" {
    const gpa = std.testing.allocator;

    var left_id = try DeviceColumn.fromSliceWithValidity(i32, gpa, &.{ 1, 2, 3, 2, 9 }, &.{ true, true, true, true, false }, .cpu);
    defer left_id.deinit();
    var left_value = try DeviceColumn.fromSlice(f64, gpa, &.{ 10.0, 20.0, 30.0, 21.0, 90.0 }, .cpu);
    defer left_value.deinit();

    var right_id = try DeviceColumn.fromSliceWithValidity(i32, gpa, &.{ 2, 3, 2, 4, 9 }, &.{ true, true, true, true, false }, .cpu);
    defer right_id.deinit();
    var right_value = try DeviceColumn.fromSlice(f64, gpa, &.{ 200.0, 300.0, 201.0, 400.0, 900.0 }, .cpu);
    defer right_value.deinit();
    var right_label = try DeviceColumn.fromSlice(i64, gpa, &.{ 20, 30, 21, 40, 90 }, .cpu);
    defer right_label.deinit();

    var left = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "id", .data = left_id },
        .{ .name = "value", .data = left_value },
    });
    defer left.deinit();
    var right = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "id", .data = right_id },
        .{ .name = "value", .data = right_value },
        .{ .name = "label", .data = right_label },
    });
    defer right.deinit();

    var joined = try left.innerJoin(right, "id", "id", .{});
    defer joined.deinit();
    try std.testing.expectEqual(@as(usize, 5), joined.height());
    try std.testing.expectEqual(@as(usize, 4), joined.width());
    try std.testing.expectEqual(DeviceDType.f64, try joined.columnDType("value"));
    try std.testing.expectEqual(DeviceDType.f64, try joined.columnDType("value_right"));

    const ids = try (try joined.column("id")).i32.toOwnedSlice(gpa);
    defer gpa.free(ids);
    const left_values = try (try joined.column("value")).f64.toOwnedSlice(gpa);
    defer gpa.free(left_values);
    const right_values = try (try joined.column("value_right")).f64.toOwnedSlice(gpa);
    defer gpa.free(right_values);
    const labels = try (try joined.column("label")).i64.toOwnedSlice(gpa);
    defer gpa.free(labels);

    try std.testing.expectEqualSlices(i32, &.{ 2, 2, 3, 2, 2 }, ids);
    try std.testing.expectEqualSlices(f64, &.{ 20.0, 20.0, 30.0, 21.0, 21.0 }, left_values);
    try std.testing.expectEqualSlices(f64, &.{ 200.0, 201.0, 300.0, 200.0, 201.0 }, right_values);
    try std.testing.expectEqualSlices(i64, &.{ 20, 21, 30, 20, 21 }, labels);
}

test "device dataframe inner joins on multiple fixed-width keys" {
    const gpa = std.testing.allocator;

    var left_store = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 1, 2, 2 }, .cpu);
    defer left_store.deinit();
    var left_day = try DeviceColumn.fromSlice(i32, gpa, &.{ 10, 11, 10, 12 }, .cpu);
    defer left_day.deinit();
    var left_sales = try DeviceColumn.fromSlice(f64, gpa, &.{ 100.0, 110.0, 200.0, 220.0 }, .cpu);
    defer left_sales.deinit();

    var right_store = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 2, 2, 3 }, .cpu);
    defer right_store.deinit();
    var right_day = try DeviceColumn.fromSlice(i32, gpa, &.{ 10, 10, 13, 10 }, .cpu);
    defer right_day.deinit();
    var right_region = try DeviceColumn.fromSlice(i64, gpa, &.{ 7, 8, 9, 10 }, .cpu);
    defer right_region.deinit();

    var left = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "store", .data = left_store },
        .{ .name = "day", .data = left_day },
        .{ .name = "sales", .data = left_sales },
    });
    defer left.deinit();
    var right = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "store", .data = right_store },
        .{ .name = "day", .data = right_day },
        .{ .name = "region", .data = right_region },
    });
    defer right.deinit();

    var joined = try left.innerJoinOn(right, &.{ "store", "day" }, &.{ "store", "day" }, .{});
    defer joined.deinit();
    try std.testing.expectEqual(@as(usize, 2), joined.height());
    try std.testing.expectEqual(@as(usize, 4), joined.width());

    const stores = try (try joined.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(stores);
    const days = try (try joined.column("day")).i32.toOwnedSlice(gpa);
    defer gpa.free(days);
    const sales = try (try joined.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(sales);
    const regions = try (try joined.column("region")).i64.toOwnedSlice(gpa);
    defer gpa.free(regions);

    try std.testing.expectEqualSlices(i32, &.{ 1, 2 }, stores);
    try std.testing.expectEqualSlices(i32, &.{ 10, 10 }, days);
    try std.testing.expectEqualSlices(f64, &.{ 100.0, 200.0 }, sales);
    try std.testing.expectEqualSlices(i64, &.{ 7, 8 }, regions);
}

test "device dataframe left joins on multiple fixed-width keys" {
    const gpa = std.testing.allocator;

    var left_store = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 1, 2, 2 }, .cpu);
    defer left_store.deinit();
    var left_day = try DeviceColumn.fromSliceWithValidity(i32, gpa, &.{ 10, 11, 10, 12 }, &.{ true, true, true, false }, .cpu);
    defer left_day.deinit();
    var left_sales = try DeviceColumn.fromSlice(f64, gpa, &.{ 100.0, 110.0, 200.0, 220.0 }, .cpu);
    defer left_sales.deinit();

    var right_store = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 2, 3 }, .cpu);
    defer right_store.deinit();
    var right_day = try DeviceColumn.fromSlice(i32, gpa, &.{ 10, 10, 10 }, .cpu);
    defer right_day.deinit();
    var right_region = try DeviceColumn.fromSlice(i64, gpa, &.{ 7, 8, 10 }, .cpu);
    defer right_region.deinit();

    var left = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "store", .data = left_store },
        .{ .name = "day", .data = left_day },
        .{ .name = "sales", .data = left_sales },
    });
    defer left.deinit();
    var right = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "store", .data = right_store },
        .{ .name = "day", .data = right_day },
        .{ .name = "region", .data = right_region },
    });
    defer right.deinit();

    var joined = try left.leftJoinOn(right, &.{ "store", "day" }, &.{ "store", "day" }, .{});
    defer joined.deinit();
    try std.testing.expectEqual(@as(usize, 4), joined.height());
    try std.testing.expectEqual(@as(usize, 4), joined.width());

    const stores = try (try joined.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(stores);
    const days = try (try joined.column("day")).i32.toOwnedSlice(gpa);
    defer gpa.free(days);
    const day_validity = try (try joined.column("day")).i32.validity.?.toOwnedSlice(gpa);
    defer gpa.free(day_validity);
    const regions = try (try joined.column("region")).i64.toOwnedSlice(gpa);
    defer gpa.free(regions);
    const region_validity = try (try joined.column("region")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(region_validity);

    try std.testing.expectEqualSlices(i32, &.{ 1, 1, 2, 2 }, stores);
    try std.testing.expectEqualSlices(i32, &.{ 10, 11, 10, 12 }, days);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false }, day_validity);
    try std.testing.expectEqualSlices(i64, &.{ 7, 0, 8, 0 }, regions);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, false }, region_validity);
}

test "device dataframe full joins on multiple fixed-width keys" {
    const gpa = std.testing.allocator;

    var left_store = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 1, 2 }, .cpu);
    defer left_store.deinit();
    var left_day = try DeviceColumn.fromSliceWithValidity(i32, gpa, &.{ 10, 11, 10 }, &.{ true, true, false }, .cpu);
    defer left_day.deinit();
    var left_sales = try DeviceColumn.fromSlice(f64, gpa, &.{ 100.0, 110.0, 200.0 }, .cpu);
    defer left_sales.deinit();

    var right_store = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 2, 3 }, .cpu);
    defer right_store.deinit();
    var right_day = try DeviceColumn.fromSlice(i32, gpa, &.{ 10, 10, 10 }, .cpu);
    defer right_day.deinit();
    var right_region = try DeviceColumn.fromSlice(i64, gpa, &.{ 7, 8, 9 }, .cpu);
    defer right_region.deinit();

    var left = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "store", .data = left_store },
        .{ .name = "day", .data = left_day },
        .{ .name = "sales", .data = left_sales },
    });
    defer left.deinit();
    var right = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "store", .data = right_store },
        .{ .name = "day", .data = right_day },
        .{ .name = "region", .data = right_region },
    });
    defer right.deinit();

    var joined = try left.fullJoinOn(right, &.{ "store", "day" }, &.{ "store", "day" }, .{});
    defer joined.deinit();
    try std.testing.expectEqual(@as(usize, 5), joined.height());
    try std.testing.expectEqual(@as(usize, 4), joined.width());

    const stores = try (try joined.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(stores);
    const days = try (try joined.column("day")).i32.toOwnedSlice(gpa);
    defer gpa.free(days);
    const day_validity = try (try joined.column("day")).i32.validity.?.toOwnedSlice(gpa);
    defer gpa.free(day_validity);
    const sales = try (try joined.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(sales);
    const sales_validity = try (try joined.column("sales")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(sales_validity);
    const regions = try (try joined.column("region")).i64.toOwnedSlice(gpa);
    defer gpa.free(regions);
    const region_validity = try (try joined.column("region")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(region_validity);

    try std.testing.expectEqualSlices(i32, &.{ 1, 1, 2, 2, 3 }, stores);
    try std.testing.expectEqualSlices(i32, &.{ 10, 11, 0, 10, 10 }, days);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true, true }, day_validity);
    try std.testing.expectEqualSlices(f64, &.{ 100.0, 110.0, 200.0, 0.0, 0.0 }, sales);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false, false }, sales_validity);
    try std.testing.expectEqualSlices(i64, &.{ 7, 0, 0, 8, 9 }, regions);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true, true }, region_validity);
}

test "device dataframe semi and anti join on multiple fixed-width keys" {
    const gpa = std.testing.allocator;

    var left_store = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 1, 2, 2, 3 }, .cpu);
    defer left_store.deinit();
    var left_day = try DeviceColumn.fromSliceWithValidity(i32, gpa, &.{ 10, 11, 10, 12, 10 }, &.{ true, true, true, false, true }, .cpu);
    defer left_day.deinit();
    var left_sales = try DeviceColumn.fromSlice(f64, gpa, &.{ 100.0, 110.0, 200.0, 220.0, 300.0 }, .cpu);
    defer left_sales.deinit();

    var right_store = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 2, 4 }, .cpu);
    defer right_store.deinit();
    var right_day = try DeviceColumn.fromSlice(i32, gpa, &.{ 10, 10, 10 }, .cpu);
    defer right_day.deinit();

    var left = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "store", .data = left_store },
        .{ .name = "day", .data = left_day },
        .{ .name = "sales", .data = left_sales },
    });
    defer left.deinit();
    var right = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "store", .data = right_store },
        .{ .name = "day", .data = right_day },
    });
    defer right.deinit();

    var semi = try left.semiJoinOn(right, &.{ "store", "day" }, &.{ "store", "day" });
    defer semi.deinit();
    const semi_store = try (try semi.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(semi_store);
    const semi_sales = try (try semi.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(semi_sales);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2 }, semi_store);
    try std.testing.expectEqualSlices(f64, &.{ 100.0, 200.0 }, semi_sales);

    var anti = try left.antiJoinOn(right, &.{ "store", "day" }, &.{ "store", "day" });
    defer anti.deinit();
    const anti_store = try (try anti.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(anti_store);
    const anti_sales = try (try anti.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(anti_sales);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2, 3 }, anti_store);
    try std.testing.expectEqualSlices(f64, &.{ 110.0, 220.0, 300.0 }, anti_sales);
}

test "device dataframe left joins with nullable unmatched right payloads" {
    const gpa = std.testing.allocator;

    var left_id = try DeviceColumn.fromSliceWithValidity(i32, gpa, &.{ 1, 2, 3, 2, 9 }, &.{ true, true, true, true, false }, .cpu);
    defer left_id.deinit();
    var left_value = try DeviceColumn.fromSlice(f64, gpa, &.{ 10.0, 20.0, 30.0, 21.0, 90.0 }, .cpu);
    defer left_value.deinit();

    var right_id = try DeviceColumn.fromSlice(i32, gpa, &.{ 2, 3 }, .cpu);
    defer right_id.deinit();
    var right_value = try DeviceColumn.fromSlice(f64, gpa, &.{ 200.0, 300.0 }, .cpu);
    defer right_value.deinit();

    var left = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "id", .data = left_id },
        .{ .name = "value", .data = left_value },
    });
    defer left.deinit();
    var right = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "id", .data = right_id },
        .{ .name = "value", .data = right_value },
    });
    defer right.deinit();

    var joined = try left.leftJoin(right, "id", "id", .{});
    defer joined.deinit();
    try std.testing.expectEqual(@as(usize, 5), joined.height());
    try std.testing.expectEqual(DeviceDType.f64, try joined.columnDType("value_right"));

    const ids = try (try joined.column("id")).i32.toOwnedSlice(gpa);
    defer gpa.free(ids);
    const right_values = try (try joined.column("value_right")).f64.toOwnedSlice(gpa);
    defer gpa.free(right_values);
    const right_validity = try (try joined.column("value_right")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(right_validity);

    try std.testing.expectEqualSlices(i32, &.{ 1, 2, 3, 2, 9 }, ids);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 200.0, 300.0, 200.0, 0.0 }, right_values);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, false }, right_validity);
}

test "device dataframe semi and anti joins filter left rows" {
    const gpa = std.testing.allocator;

    var left_id = try DeviceColumn.fromSliceWithValidity(i32, gpa, &.{ 1, 2, 3, 2, 9 }, &.{ true, true, true, true, false }, .cpu);
    defer left_id.deinit();
    var left_value = try DeviceColumn.fromSlice(f64, gpa, &.{ 10.0, 20.0, 30.0, 21.0, 90.0 }, .cpu);
    defer left_value.deinit();

    var right_id = try DeviceColumn.fromSlice(i32, gpa, &.{ 2, 4 }, .cpu);
    defer right_id.deinit();
    var right_value = try DeviceColumn.fromSlice(f64, gpa, &.{ 200.0, 400.0 }, .cpu);
    defer right_value.deinit();

    var left = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "id", .data = left_id },
        .{ .name = "value", .data = left_value },
    });
    defer left.deinit();
    var right = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "id", .data = right_id },
        .{ .name = "value", .data = right_value },
    });
    defer right.deinit();

    var semi = try left.semiJoin(right, "id", "id");
    defer semi.deinit();
    try std.testing.expectEqual(@as(usize, 2), semi.height());
    const semi_ids = try (try semi.column("id")).i32.toOwnedSlice(gpa);
    defer gpa.free(semi_ids);
    const semi_values = try (try semi.column("value")).f64.toOwnedSlice(gpa);
    defer gpa.free(semi_values);
    try std.testing.expectEqualSlices(i32, &.{ 2, 2 }, semi_ids);
    try std.testing.expectEqualSlices(f64, &.{ 20.0, 21.0 }, semi_values);

    var anti = try left.antiJoin(right, "id", "id");
    defer anti.deinit();
    try std.testing.expectEqual(@as(usize, 3), anti.height());
    const anti_ids = try (try anti.column("id")).i32.toOwnedSlice(gpa);
    defer gpa.free(anti_ids);
    const anti_values = try (try anti.column("value")).f64.toOwnedSlice(gpa);
    defer gpa.free(anti_values);
    try std.testing.expectEqualSlices(i32, &.{ 1, 3, 9 }, anti_ids);
    try std.testing.expectEqualSlices(f64, &.{ 10.0, 30.0, 90.0 }, anti_values);
}

test "device dataframe full joins with nullable payloads from both sides" {
    const gpa = std.testing.allocator;

    var left_id = try DeviceColumn.fromSliceWithValidity(i32, gpa, &.{ 1, 2, 3, 9 }, &.{ true, true, true, false }, .cpu);
    defer left_id.deinit();
    var left_value = try DeviceColumn.fromSlice(f64, gpa, &.{ 10.0, 20.0, 30.0, 90.0 }, .cpu);
    defer left_value.deinit();

    var right_id = try DeviceColumn.fromSliceWithValidity(i32, gpa, &.{ 2, 4, 9 }, &.{ true, true, false }, .cpu);
    defer right_id.deinit();
    var right_value = try DeviceColumn.fromSlice(f64, gpa, &.{ 200.0, 400.0, 900.0 }, .cpu);
    defer right_value.deinit();

    var left = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "id", .data = left_id },
        .{ .name = "value", .data = left_value },
    });
    defer left.deinit();
    var right = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "id", .data = right_id },
        .{ .name = "value", .data = right_value },
    });
    defer right.deinit();

    var joined = try left.fullJoin(right, "id", "id", .{});
    defer joined.deinit();
    try std.testing.expectEqual(@as(usize, 6), joined.height());

    const ids = try (try joined.column("id")).i32.toOwnedSlice(gpa);
    defer gpa.free(ids);
    const id_validity = try (try joined.column("id")).i32.validity.?.toOwnedSlice(gpa);
    defer gpa.free(id_validity);
    const left_values = try (try joined.column("value")).f64.toOwnedSlice(gpa);
    defer gpa.free(left_values);
    const left_validity = try (try joined.column("value")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(left_validity);
    const right_values = try (try joined.column("value_right")).f64.toOwnedSlice(gpa);
    defer gpa.free(right_values);
    const right_validity = try (try joined.column("value_right")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(right_validity);

    try std.testing.expectEqualSlices(i32, &.{ 1, 2, 3, 0, 4, 0 }, ids);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false, true, false }, id_validity);
    try std.testing.expectEqualSlices(f64, &.{ 10.0, 20.0, 30.0, 90.0, 0.0, 0.0 }, left_values);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, false, false }, left_validity);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 200.0, 0.0, 0.0, 400.0, 900.0 }, right_values);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false, true, true }, right_validity);
}

test "device dataframe asof joins with previous next and nearest strategies" {
    const gpa = std.testing.allocator;

    var left_time = try DeviceColumn.fromSliceWithValidity(i64, gpa, &.{ 1, 5, 8, 12, 20 }, &.{ true, true, true, true, false }, .cpu);
    defer left_time.deinit();
    var left_value = try DeviceColumn.fromSlice(f64, gpa, &.{ 10.0, 50.0, 80.0, 120.0, 200.0 }, .cpu);
    defer left_value.deinit();

    var right_time = try DeviceColumn.fromSliceWithValidity(i64, gpa, &.{ 2, 6, 10, 30 }, &.{ true, true, true, false }, .cpu);
    defer right_time.deinit();
    var quote = try DeviceColumn.fromSlice(i64, gpa, &.{ 20, 60, 100, 300 }, .cpu);
    defer quote.deinit();

    var left = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "time", .data = left_time },
        .{ .name = "value", .data = left_value },
    });
    defer left.deinit();
    var right = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "time", .data = right_time },
        .{ .name = "quote", .data = quote },
    });
    defer right.deinit();

    var previous = try left.asofJoin(right, "time", "time", .{ .strategy = .previous });
    defer previous.deinit();
    const previous_quote = try (try previous.column("quote")).i64.toOwnedSlice(gpa);
    defer gpa.free(previous_quote);
    const previous_validity = try (try previous.column("quote")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(previous_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 20, 60, 100, 0 }, previous_quote);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, false }, previous_validity);

    var next = try left.asofJoin(right, "time", "time", .{ .strategy = .next });
    defer next.deinit();
    const next_quote = try (try next.column("quote")).i64.toOwnedSlice(gpa);
    defer gpa.free(next_quote);
    const next_validity = try (try next.column("quote")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(next_validity);
    try std.testing.expectEqualSlices(i64, &.{ 20, 60, 100, 0, 0 }, next_quote);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false, false }, next_validity);

    var nearest = try left.asofJoin(right, "time", "time", .{ .strategy = .nearest });
    defer nearest.deinit();
    const nearest_quote = try (try nearest.column("quote")).i64.toOwnedSlice(gpa);
    defer gpa.free(nearest_quote);
    const nearest_validity = try (try nearest.column("quote")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(nearest_validity);
    try std.testing.expectEqualSlices(i64, &.{ 20, 60, 60, 100, 0 }, nearest_quote);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, false }, nearest_validity);
}

test "device dataframe concatenates rows eagerly and lazily" {
    const gpa = std.testing.allocator;

    var left_id = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 2 }, .cpu);
    defer left_id.deinit();
    var left_value = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 10.0, 20.0 }, &.{ true, false }, .cpu);
    defer left_value.deinit();
    var right_id = try DeviceColumn.fromSlice(i32, gpa, &.{ 3, 4 }, .cpu);
    defer right_id.deinit();
    var right_value = try DeviceColumn.fromSlice(f64, gpa, &.{ 30.0, 40.0 }, .cpu);
    defer right_value.deinit();

    var left = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "id", .data = left_id },
        .{ .name = "value", .data = left_value },
    });
    defer left.deinit();
    var right = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "id", .data = right_id },
        .{ .name = "value", .data = right_value },
    });
    defer right.deinit();

    var stacked = try left.concatRows(right);
    defer stacked.deinit();
    try std.testing.expectEqual(@as(usize, 4), stacked.height());
    try std.testing.expectEqual(@as(usize, 2), stacked.width());
    const stacked_ids = try (try stacked.column("id")).i32.toOwnedSlice(gpa);
    defer gpa.free(stacked_ids);
    const stacked_values = try (try stacked.column("value")).f64.toOwnedSlice(gpa);
    defer gpa.free(stacked_values);
    const stacked_validity = try (try stacked.column("value")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(stacked_validity);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2, 3, 4 }, stacked_ids);
    try std.testing.expectEqualSlices(f64, &.{ 10.0, 20.0, 30.0, 40.0 }, stacked_values);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, true }, stacked_validity);

    var plan = try DeviceLazyFrame.init(gpa, left);
    defer plan.deinit();
    try plan.concatRows(right);
    try plan.filterColumnScalar("id", i32, 2, .ge);
    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "concat_rows(rows=2, cols=2)") != null);
    var lazy_stacked = try plan.collect();
    defer lazy_stacked.deinit();
    try std.testing.expectEqual(@as(usize, 3), lazy_stacked.height());
    const lazy_ids = try (try lazy_stacked.column("id")).i32.toOwnedSlice(gpa);
    defer gpa.free(lazy_ids);
    try std.testing.expectEqualSlices(i32, &.{ 2, 3, 4 }, lazy_ids);
}

test "device dataframe drops duplicate rows eagerly and lazily" {
    const gpa = std.testing.allocator;

    var id = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 1, 2, 2, 3 }, .cpu);
    defer id.deinit();
    var value = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 10.0, 99.0, 20.0, 21.0, 30.0 }, &.{ true, true, true, true, false }, .cpu);
    defer value.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "id", .data = id },
        .{ .name = "value", .data = value },
    });
    defer table.deinit();

    var distinct = try table.distinctOn(&.{"id"});
    defer distinct.deinit();
    try std.testing.expectEqual(@as(usize, 3), distinct.height());
    const distinct_ids = try (try distinct.column("id")).i32.toOwnedSlice(gpa);
    defer gpa.free(distinct_ids);
    const distinct_values = try (try distinct.column("value")).f64.toOwnedSlice(gpa);
    defer gpa.free(distinct_values);
    const distinct_validity = try (try distinct.column("value")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(distinct_validity);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2, 3 }, distinct_ids);
    try std.testing.expectEqualSlices(f64, &.{ 10.0, 20.0, 30.0 }, distinct_values);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false }, distinct_validity);

    var full_distinct = try table.distinctRows();
    defer full_distinct.deinit();
    try std.testing.expectEqual(@as(usize, 4), full_distinct.height());

    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.distinctOn(&.{"id"});
    try plan.select(&.{ "id", "value" });
    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "distinct_on([id])") != null);
    var lazy_distinct = try plan.collect();
    defer lazy_distinct.deinit();
    try std.testing.expectEqual(@as(usize, 3), lazy_distinct.height());
    const lazy_ids = try (try lazy_distinct.column("id")).i32.toOwnedSlice(gpa);
    defer gpa.free(lazy_ids);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2, 3 }, lazy_ids);
}
test "device lazy frame collects staged select filter sort and limit operations" {
    const gpa = std.testing.allocator;

    var sales = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 3.0, 5.0, 7.0 }, .cpu);
    defer sales.deinit();
    var units = try DeviceColumn.fromSlice(i64, gpa, &.{ 1, 2, 3, 4 }, .cpu);
    defer units.deinit();
    var active = try DeviceColumn.fromSlice(bool, gpa, &.{ true, false, true, true }, .cpu);
    defer active.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "sales", .data = sales },
        .{ .name = "units", .data = units },
        .{ .name = "active", .data = active },
    });
    defer table.deinit();

    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.withColumnScalar("sales_x2", "sales", f64, 2.0, .mul);
    try plan.withColumnCompareScalar("big_sale", "sales_x2", f64, 10.0, .gt);
    try plan.filterColumnScalar("sales", f64, 2.5, .gt);
    try plan.sortBy("sales", .{ .descending = true });
    try plan.select(&.{ "sales", "units", "sales_x2", "big_sale", "active" });
    try plan.select(&.{ "sales", "units", "sales_x2", "big_sale" });
    try plan.head(3);
    try plan.head(2);

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "raw_ops=8") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "optimized_ops=6") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_scalar(sales_x2") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_compare_scalar(big_sale") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "filter_scalar(sales") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 2), result.height());
    try std.testing.expectEqual(@as(usize, 4), result.width());
    const result_sales = try (try result.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales);
    const result_units = try (try result.column("units")).i64.toOwnedSlice(gpa);
    defer gpa.free(result_units);
    const result_sales_x2 = try (try result.column("sales_x2")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_x2);
    const result_big_sale = try (try result.column("big_sale")).bool.toOwnedSlice(gpa);
    defer gpa.free(result_big_sale);
    try std.testing.expectEqualSlices(f64, &.{ 7.0, 5.0 }, result_sales);
    try std.testing.expectEqualSlices(i64, &.{ 4, 3 }, result_units);
    try std.testing.expectEqualSlices(f64, &.{ 14.0, 10.0 }, result_sales_x2);
    try std.testing.expectEqualSlices(bool, &.{ true, false }, result_big_sale);

    var topk_plan = try DeviceLazyFrame.init(gpa, table);
    defer topk_plan.deinit();
    try topk_plan.sortBy("sales", .{ .descending = true });
    try topk_plan.head(2);
    const topk_explain = try topk_plan.explain(gpa);
    defer gpa.free(topk_explain);
    try std.testing.expect(std.mem.indexOf(u8, topk_explain, "top_k(sales, k=2") != null);
    var topk = try topk_plan.collect();
    defer topk.deinit();
    const topk_sales = try (try topk.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(topk_sales);
    try std.testing.expectEqualSlices(f64, &.{ 7.0, 5.0 }, topk_sales);

    var rank_plan = try DeviceLazyFrame.init(gpa, table);
    defer rank_plan.deinit();
    try rank_plan.rankProfileBy("sales", "sales_rank", .{ .descending = true });
    try rank_plan.select(&.{ "sales", "sales_rank_ordinal_rank", "sales_rank_percent_rank", "sales_rank_cume_dist" });
    const rank_explain = try rank_plan.explain(gpa);
    defer gpa.free(rank_explain);
    try std.testing.expect(std.mem.indexOf(u8, rank_explain, "rank_profile_by(sales") != null);
    var ranked = try rank_plan.collect();
    defer ranked.deinit();
    try std.testing.expectEqual(@as(usize, 4), ranked.height());
    try std.testing.expectEqual(@as(usize, 4), ranked.width());
    const ranked_ordinal = try (try ranked.column("sales_rank_ordinal_rank")).i64.toOwnedSlice(gpa);
    defer gpa.free(ranked_ordinal);
    const ranked_percent = try (try ranked.column("sales_rank_percent_rank")).f64.toOwnedSlice(gpa);
    defer gpa.free(ranked_percent);
    const ranked_cume = try (try ranked.column("sales_rank_cume_dist")).f64.toOwnedSlice(gpa);
    defer gpa.free(ranked_cume);
    try std.testing.expectEqualSlices(i64, &.{ 4, 3, 2, 1 }, ranked_ordinal);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), ranked_percent[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), ranked_percent[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), ranked_percent[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ranked_percent[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), ranked_cume[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), ranked_cume[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), ranked_cume[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), ranked_cume[3], 1e-12);

    var rolling_plan = try DeviceLazyFrame.init(gpa, table);
    defer rolling_plan.deinit();
    try rolling_plan.rollingProfile("sales", "sales", .{ .window = 2, .min_periods = 1 });
    try rolling_plan.select(&.{ "sales", "sales_rolling_mean", "sales_rolling_stddev" });
    const rolling_explain = try rolling_plan.explain(gpa);
    defer gpa.free(rolling_explain);
    try std.testing.expect(std.mem.indexOf(u8, rolling_explain, "rolling_profile(sales") != null);
    var rolling = try rolling_plan.collect();
    defer rolling.deinit();
    try std.testing.expectEqual(@as(usize, 4), rolling.height());
    try std.testing.expectEqual(@as(usize, 3), rolling.width());
    const rolling_mean = try (try rolling.column("sales_rolling_mean")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_mean);
    const rolling_stddev = try (try rolling.column("sales_rolling_stddev")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolling_stddev);
    const rolling_validity = try (try rolling.column("sales_rolling_mean")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(rolling_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, rolling_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), rolling_mean[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), rolling_mean[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), rolling_mean[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 6.0), rolling_mean[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), rolling_stddev[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rolling_stddev[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_stddev[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rolling_stddev[3], 1e-12);

    var ema_plan = try DeviceLazyFrame.init(gpa, table);
    defer ema_plan.deinit();
    try ema_plan.emaProfile("sales", "sales", .{ .alpha = 0.5, .min_periods = 1 });
    try ema_plan.select(&.{ "sales", "sales_ema", "sales_ema_residual", "sales_ema_ratio" });
    const ema_explain = try ema_plan.explain(gpa);
    defer gpa.free(ema_explain);
    try std.testing.expect(std.mem.indexOf(u8, ema_explain, "ema_profile(sales") != null);
    var ema = try ema_plan.collect();
    defer ema.deinit();
    try std.testing.expectEqual(@as(usize, 4), ema.height());
    try std.testing.expectEqual(@as(usize, 4), ema.width());
    const lazy_ema = try (try ema.column("sales_ema")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ema);
    const lazy_ema_residual = try (try ema.column("sales_ema_residual")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ema_residual);
    const lazy_ema_ratio = try (try ema.column("sales_ema_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ema_ratio);
    const lazy_ema_validity = try (try ema.column("sales_ema")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_ema_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, lazy_ema_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_ema[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), lazy_ema[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.75), lazy_ema[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.375), lazy_ema[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_ema_residual[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_ema_residual[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.25), lazy_ema_residual[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.625), lazy_ema_residual[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_ema_ratio[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.2), lazy_ema_ratio[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 3.75), lazy_ema_ratio[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 7.0 / 5.375), lazy_ema_ratio[3], 1e-12);

    var rolling_range_plan = try DeviceLazyFrame.init(gpa, table);
    defer rolling_range_plan.deinit();
    try rolling_range_plan.rollingRangeProfile("sales", "sales", .{ .window = 2, .min_periods = 1 });
    try rolling_range_plan.select(&.{ "sales", "sales_rolling_low", "sales_rolling_high", "sales_rolling_position" });
    const rolling_range_explain = try rolling_range_plan.explain(gpa);
    defer gpa.free(rolling_range_explain);
    try std.testing.expect(std.mem.indexOf(u8, rolling_range_explain, "rolling_range_profile(sales") != null);
    var rolling_range = try rolling_range_plan.collect();
    defer rolling_range.deinit();
    try std.testing.expectEqual(@as(usize, 4), rolling_range.height());
    try std.testing.expectEqual(@as(usize, 4), rolling_range.width());
    const lazy_rolling_low = try (try rolling_range.column("sales_rolling_low")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_low);
    const lazy_rolling_high = try (try rolling_range.column("sales_rolling_high")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_high);
    const lazy_rolling_position = try (try rolling_range.column("sales_rolling_position")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_position);
    const lazy_rolling_range_validity = try (try rolling_range.column("sales_rolling_position")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_range_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, lazy_rolling_range_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_rolling_low[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_rolling_low[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), lazy_rolling_low[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), lazy_rolling_low[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_rolling_high[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), lazy_rolling_high[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), lazy_rolling_high[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 7.0), lazy_rolling_high[3], 1e-12);
    try std.testing.expect(std.math.isNan(lazy_rolling_position[0]));
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_position[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_position[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_position[3], 1e-12);

    var rolling_norm_plan = try DeviceLazyFrame.init(gpa, table);
    defer rolling_norm_plan.deinit();
    try rolling_norm_plan.rollingNormalizeProfile("sales", "sales", .{ .window = 2, .min_periods = 1 });
    try rolling_norm_plan.select(&.{ "sales", "sales_rolling_centered", "sales_rolling_zscore", "sales_rolling_minmax" });
    const rolling_norm_explain = try rolling_norm_plan.explain(gpa);
    defer gpa.free(rolling_norm_explain);
    try std.testing.expect(std.mem.indexOf(u8, rolling_norm_explain, "rolling_normalize_profile(sales") != null);
    var rolling_norm = try rolling_norm_plan.collect();
    defer rolling_norm.deinit();
    try std.testing.expectEqual(@as(usize, 4), rolling_norm.height());
    try std.testing.expectEqual(@as(usize, 4), rolling_norm.width());
    const lazy_rolling_centered = try (try rolling_norm.column("sales_rolling_centered")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_centered);
    const lazy_rolling_zscore = try (try rolling_norm.column("sales_rolling_zscore")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_zscore);
    const lazy_rolling_minmax = try (try rolling_norm.column("sales_rolling_minmax")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_minmax);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_rolling_centered[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_rolling_centered[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_centered[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_centered[3], 1e-12);
    try std.testing.expect(std.math.isNan(lazy_rolling_zscore[0]));
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_zscore[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_zscore[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_zscore[3], 1e-12);
    try std.testing.expect(std.math.isNan(lazy_rolling_minmax[0]));
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_minmax[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_minmax[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_minmax[3], 1e-12);

    var rolling_quantile_plan = try DeviceLazyFrame.init(gpa, table);
    defer rolling_quantile_plan.deinit();
    try rolling_quantile_plan.rollingQuantileProfile("sales", "sales", .{ .window = 2, .min_periods = 1 });
    try rolling_quantile_plan.select(&.{ "sales", "sales_rolling_q1", "sales_rolling_median", "sales_rolling_q3", "sales_rolling_iqr" });
    const rolling_quantile_explain = try rolling_quantile_plan.explain(gpa);
    defer gpa.free(rolling_quantile_explain);
    try std.testing.expect(std.mem.indexOf(u8, rolling_quantile_explain, "rolling_quantile_profile(sales") != null);
    var rolling_quantile = try rolling_quantile_plan.collect();
    defer rolling_quantile.deinit();
    try std.testing.expectEqual(@as(usize, 4), rolling_quantile.height());
    try std.testing.expectEqual(@as(usize, 5), rolling_quantile.width());
    const lazy_rolling_q1 = try (try rolling_quantile.column("sales_rolling_q1")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_q1);
    const lazy_rolling_median = try (try rolling_quantile.column("sales_rolling_median")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_median);
    const lazy_rolling_q3 = try (try rolling_quantile.column("sales_rolling_q3")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_q3);
    const lazy_rolling_iqr = try (try rolling_quantile.column("sales_rolling_iqr")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_iqr);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_rolling_q1[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.25), lazy_rolling_q1[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.5), lazy_rolling_q1[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.5), lazy_rolling_q1[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_rolling_median[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), lazy_rolling_median[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), lazy_rolling_median[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 6.0), lazy_rolling_median[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_rolling_q3[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.75), lazy_rolling_q3[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.5), lazy_rolling_q3[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 6.5), lazy_rolling_q3[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_rolling_iqr[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_rolling_iqr[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_iqr[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_rolling_iqr[3], 1e-12);

    var lag_plan = try DeviceLazyFrame.init(gpa, table);
    defer lag_plan.deinit();
    try lag_plan.lagProfile("sales", "sales", .{ .periods = 1 });
    try lag_plan.select(&.{ "sales", "sales_lag", "sales_diff", "sales_pct_change" });
    const lag_explain = try lag_plan.explain(gpa);
    defer gpa.free(lag_explain);
    try std.testing.expect(std.mem.indexOf(u8, lag_explain, "lag_profile(sales") != null);
    var lagged = try lag_plan.collect();
    defer lagged.deinit();
    try std.testing.expectEqual(@as(usize, 4), lagged.height());
    try std.testing.expectEqual(@as(usize, 4), lagged.width());
    const lazy_lag = try (try lagged.column("sales_lag")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_lag);
    const lazy_diff = try (try lagged.column("sales_diff")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_diff);
    const lazy_pct = try (try lagged.column("sales_pct_change")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_pct);
    const lazy_lag_validity = try (try lagged.column("sales_lag")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_lag_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_lag_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_lag[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), lazy_lag[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), lazy_lag[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_diff[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_diff[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_diff[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_pct[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), lazy_pct[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.4), lazy_pct[3], 1e-12);

    var lead_plan = try DeviceLazyFrame.init(gpa, table);
    defer lead_plan.deinit();
    try lead_plan.leadProfile("sales", "sales", .{ .periods = 1 });
    try lead_plan.select(&.{ "sales", "sales_lead", "sales_forward_diff", "sales_forward_pct_change" });
    const lead_explain = try lead_plan.explain(gpa);
    defer gpa.free(lead_explain);
    try std.testing.expect(std.mem.indexOf(u8, lead_explain, "lead_profile(sales") != null);
    var leaded = try lead_plan.collect();
    defer leaded.deinit();
    try std.testing.expectEqual(@as(usize, 4), leaded.height());
    try std.testing.expectEqual(@as(usize, 4), leaded.width());
    const lazy_lead = try (try leaded.column("sales_lead")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_lead);
    const lazy_forward_diff = try (try leaded.column("sales_forward_diff")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_forward_diff);
    const lazy_forward_pct = try (try leaded.column("sales_forward_pct_change")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_forward_pct);
    const lazy_lead_validity = try (try leaded.column("sales_lead")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_lead_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false }, lazy_lead_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), lazy_lead[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), lazy_lead[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 7.0), lazy_lead[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_forward_diff[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_forward_diff[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_forward_diff[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_forward_pct[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), lazy_forward_pct[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.4), lazy_forward_pct[2], 1e-12);

    var clip_plan = try DeviceLazyFrame.init(gpa, table);
    defer clip_plan.deinit();
    try clip_plan.clipProfile("sales", "sales", .{ .lower = 3.0, .upper = 5.0 });
    try clip_plan.select(&.{ "sales", "sales_clipped", "sales_below", "sales_above", "sales_in_range" });
    const clip_explain = try clip_plan.explain(gpa);
    defer gpa.free(clip_explain);
    try std.testing.expect(std.mem.indexOf(u8, clip_explain, "clip_profile(sales") != null);
    var lazy_clip = try clip_plan.collect();
    defer lazy_clip.deinit();
    try std.testing.expectEqual(@as(usize, 4), lazy_clip.height());
    try std.testing.expectEqual(@as(usize, 5), lazy_clip.width());
    const lazy_clipped = try (try lazy_clip.column("sales_clipped")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_clipped);
    const lazy_below = try (try lazy_clip.column("sales_below")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_below);
    const lazy_above = try (try lazy_clip.column("sales_above")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_above);
    const lazy_in_range = try (try lazy_clip.column("sales_in_range")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_in_range);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 3.0, 5.0, 5.0 }, lazy_clipped);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false }, lazy_below);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, true }, lazy_above);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, false }, lazy_in_range);

    var threshold_plan = try DeviceLazyFrame.init(gpa, table);
    defer threshold_plan.deinit();
    try threshold_plan.thresholdProfile("sales", "sales", .{ .threshold = 5.0 });
    try threshold_plan.select(&.{ "sales", "sales_distance", "sales_abs_distance", "sales_above", "sales_below", "sales_at" });
    const threshold_explain = try threshold_plan.explain(gpa);
    defer gpa.free(threshold_explain);
    try std.testing.expect(std.mem.indexOf(u8, threshold_explain, "threshold_profile(sales") != null);
    var lazy_threshold = try threshold_plan.collect();
    defer lazy_threshold.deinit();
    try std.testing.expectEqual(@as(usize, 4), lazy_threshold.height());
    try std.testing.expectEqual(@as(usize, 6), lazy_threshold.width());
    const lazy_distance = try (try lazy_threshold.column("sales_distance")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_distance);
    const lazy_abs_distance = try (try lazy_threshold.column("sales_abs_distance")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_abs_distance);
    const lazy_above_threshold = try (try lazy_threshold.column("sales_above")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_above_threshold);
    const lazy_below_threshold = try (try lazy_threshold.column("sales_below")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_below_threshold);
    const lazy_at_threshold = try (try lazy_threshold.column("sales_at")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_at_threshold);
    try std.testing.expectEqualSlices(f64, &.{ -3.0, -2.0, 0.0, 2.0 }, lazy_distance);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 2.0, 0.0, 2.0 }, lazy_abs_distance);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, true }, lazy_above_threshold);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false }, lazy_below_threshold);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false }, lazy_at_threshold);

    var expanding_plan = try DeviceLazyFrame.init(gpa, table);
    defer expanding_plan.deinit();
    try expanding_plan.expandingProfile("sales", "sales", .{ .min_periods = 2 });
    try expanding_plan.select(&.{ "sales", "sales_expanding_count", "sales_expanding_mean", "sales_expanding_max" });
    const expanding_explain = try expanding_plan.explain(gpa);
    defer gpa.free(expanding_explain);
    try std.testing.expect(std.mem.indexOf(u8, expanding_explain, "expanding_profile(sales") != null);
    var expanding = try expanding_plan.collect();
    defer expanding.deinit();
    try std.testing.expectEqual(@as(usize, 4), expanding.height());
    try std.testing.expectEqual(@as(usize, 4), expanding.width());
    const expanding_count = try (try expanding.column("sales_expanding_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(expanding_count);
    const expanding_mean = try (try expanding.column("sales_expanding_mean")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_mean);
    const expanding_max = try (try expanding.column("sales_expanding_max")).f64.toOwnedSlice(gpa);
    defer gpa.free(expanding_max);
    const expanding_validity = try (try expanding.column("sales_expanding_mean")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(expanding_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4 }, expanding_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, expanding_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), expanding_mean[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0 / 3.0), expanding_mean[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.25), expanding_mean[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), expanding_max[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), expanding_max[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 7.0), expanding_max[3], 1e-12);

    var standardize_plan = try DeviceLazyFrame.init(gpa, table);
    defer standardize_plan.deinit();
    try standardize_plan.standardizeProfile("sales", "sales", .{});
    try standardize_plan.select(&.{ "sales", "sales_centered", "sales_zscore", "sales_minmax" });
    const standardize_explain = try standardize_plan.explain(gpa);
    defer gpa.free(standardize_explain);
    try std.testing.expect(std.mem.indexOf(u8, standardize_explain, "standardize_profile(sales") != null);
    var standardized = try standardize_plan.collect();
    defer standardized.deinit();
    try std.testing.expectEqual(@as(usize, 4), standardized.height());
    try std.testing.expectEqual(@as(usize, 4), standardized.width());
    const lazy_centered = try (try standardized.column("sales_centered")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_centered);
    const lazy_zscore = try (try standardized.column("sales_zscore")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_zscore);
    const lazy_minmax = try (try standardized.column("sales_minmax")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_minmax);
    const lazy_standardized_validity = try (try standardized.column("sales_zscore")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_standardized_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, lazy_standardized_validity);
    try std.testing.expectApproxEqAbs(@as(f64, -2.25), lazy_centered[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -1.25), lazy_centered[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), lazy_centered[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.75), lazy_centered[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -1.171700198827415), lazy_zscore[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.6509445549041194), lazy_zscore[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.39056673294247163), lazy_zscore[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.4320780207890627), lazy_zscore[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_minmax[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.2), lazy_minmax[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.6), lazy_minmax[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_minmax[3], 1e-12);

    var robust_plan = try DeviceLazyFrame.init(gpa, table);
    defer robust_plan.deinit();
    try robust_plan.robustProfile("sales", "sales", .{});
    try robust_plan.select(&.{ "sales", "sales_median_centered", "sales_mad_zscore", "sales_iqr_outlier", "sales_winsorized" });
    const robust_explain = try robust_plan.explain(gpa);
    defer gpa.free(robust_explain);
    try std.testing.expect(std.mem.indexOf(u8, robust_explain, "robust_profile(sales") != null);
    var robust = try robust_plan.collect();
    defer robust.deinit();
    try std.testing.expectEqual(@as(usize, 4), robust.height());
    try std.testing.expectEqual(@as(usize, 5), robust.width());
    const lazy_median_centered = try (try robust.column("sales_median_centered")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_median_centered);
    const lazy_mad_zscore = try (try robust.column("sales_mad_zscore")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_mad_zscore);
    const lazy_iqr_outlier = try (try robust.column("sales_iqr_outlier")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_iqr_outlier);
    const lazy_winsorized = try (try robust.column("sales_winsorized")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_winsorized);
    try std.testing.expectApproxEqAbs(@as(f64, -2.0), lazy_median_centered[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -1.0), lazy_median_centered[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_median_centered[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), lazy_median_centered[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.8993196669281089), lazy_mad_zscore[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.44965983346405447), lazy_mad_zscore[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.44965983346405447), lazy_mad_zscore[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.3489795003921634), lazy_mad_zscore[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false }, lazy_iqr_outlier);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 3.0, 5.0, 7.0 }, lazy_winsorized);

    var drawdown_plan = try DeviceLazyFrame.init(gpa, table);
    defer drawdown_plan.deinit();
    try drawdown_plan.drawdownProfile("sales", "sales", .{ .min_periods = 2 });
    try drawdown_plan.select(&.{ "sales", "sales_running_peak", "sales_drawdown", "sales_drawdown_pct" });
    const drawdown_explain = try drawdown_plan.explain(gpa);
    defer gpa.free(drawdown_explain);
    try std.testing.expect(std.mem.indexOf(u8, drawdown_explain, "drawdown_profile(sales") != null);
    var drawdown = try drawdown_plan.collect();
    defer drawdown.deinit();
    try std.testing.expectEqual(@as(usize, 4), drawdown.height());
    try std.testing.expectEqual(@as(usize, 4), drawdown.width());
    const lazy_peak = try (try drawdown.column("sales_running_peak")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_peak);
    const lazy_drawdown = try (try drawdown.column("sales_drawdown")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_drawdown);
    const lazy_drawdown_pct = try (try drawdown.column("sales_drawdown_pct")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_drawdown_pct);
    const lazy_drawdown_validity = try (try drawdown.column("sales_drawdown")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_drawdown_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_drawdown_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), lazy_peak[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), lazy_peak[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 7.0), lazy_peak[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_drawdown[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_drawdown[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_drawdown[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_drawdown_pct[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_drawdown_pct[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_drawdown_pct[3], 1e-12);

    var extrema_plan = try DeviceLazyFrame.init(gpa, table);
    defer extrema_plan.deinit();
    try extrema_plan.extremaProfile("sales", "sales", .{ .min_periods = 1 });
    try extrema_plan.select(&.{ "sales", "sales_running_low", "sales_running_high", "sales_new_low", "sales_new_high" });
    const extrema_explain = try extrema_plan.explain(gpa);
    defer gpa.free(extrema_explain);
    try std.testing.expect(std.mem.indexOf(u8, extrema_explain, "extrema_profile(sales") != null);
    var lazy_extrema = try extrema_plan.collect();
    defer lazy_extrema.deinit();
    try std.testing.expectEqual(@as(usize, 4), lazy_extrema.height());
    try std.testing.expectEqual(@as(usize, 5), lazy_extrema.width());
    const lazy_running_low = try (try lazy_extrema.column("sales_running_low")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_running_low);
    const lazy_running_high = try (try lazy_extrema.column("sales_running_high")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_running_high);
    const lazy_new_low = try (try lazy_extrema.column("sales_new_low")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_new_low);
    const lazy_new_high = try (try lazy_extrema.column("sales_new_high")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_new_high);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 2.0, 2.0, 2.0 }, lazy_running_low);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 3.0, 5.0, 7.0 }, lazy_running_high);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false }, lazy_new_low);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, lazy_new_high);

    var trend_plan = try DeviceLazyFrame.init(gpa, table);
    defer trend_plan.deinit();
    try trend_plan.trendProfile("sales", "sales", .{ .periods = 1 });
    try trend_plan.select(&.{ "sales", "sales_trend", "sales_up_streak", "sales_reversal" });
    const trend_explain = try trend_plan.explain(gpa);
    defer gpa.free(trend_explain);
    try std.testing.expect(std.mem.indexOf(u8, trend_explain, "trend_profile(sales") != null);
    var trend = try trend_plan.collect();
    defer trend.deinit();
    try std.testing.expectEqual(@as(usize, 4), trend.height());
    try std.testing.expectEqual(@as(usize, 4), trend.width());
    const lazy_trend = try (try trend.column("sales_trend")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_trend);
    const lazy_up_streak = try (try trend.column("sales_up_streak")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_up_streak);
    const lazy_reversal = try (try trend.column("sales_reversal")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_reversal);
    const lazy_trend_validity = try (try trend.column("sales_trend")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_trend_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_trend_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 1, 1 }, lazy_trend);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 2, 3 }, lazy_up_streak);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false }, lazy_reversal);

    var sign_plan = try DeviceLazyFrame.init(gpa, table);
    defer sign_plan.deinit();
    try sign_plan.withColumnScalar("sales_minus4", "sales", f64, 4.0, .sub);
    try sign_plan.signProfile("sales_minus4", "sales", .{ .periods = 1 });
    try sign_plan.select(&.{ "sales_minus4", "sales_sign", "sales_sign_flip", "sales_positive_streak", "sales_negative_streak" });
    const sign_explain = try sign_plan.explain(gpa);
    defer gpa.free(sign_explain);
    try std.testing.expect(std.mem.indexOf(u8, sign_explain, "sign_profile(sales_minus4") != null);
    var lazy_sign = try sign_plan.collect();
    defer lazy_sign.deinit();
    try std.testing.expectEqual(@as(usize, 4), lazy_sign.height());
    try std.testing.expectEqual(@as(usize, 5), lazy_sign.width());
    const lazy_sign_values = try (try lazy_sign.column("sales_sign")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_sign_values);
    const lazy_sign_flip = try (try lazy_sign.column("sales_sign_flip")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_sign_flip);
    const lazy_positive_streak = try (try lazy_sign.column("sales_positive_streak")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_positive_streak);
    const lazy_negative_streak = try (try lazy_sign.column("sales_negative_streak")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_negative_streak);
    try std.testing.expectEqualSlices(i64, &.{ -1, -1, 1, 1 }, lazy_sign_values);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false }, lazy_sign_flip);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 2 }, lazy_positive_streak);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 0, 0 }, lazy_negative_streak);

    var validity_plan = try DeviceLazyFrame.init(gpa, table);
    defer validity_plan.deinit();
    try validity_plan.validityProfile("sales", "sales");
    try validity_plan.select(&.{ "sales", "sales_is_null", "sales_is_valid", "sales_valid_streak", "sales_null_streak" });
    const validity_explain = try validity_plan.explain(gpa);
    defer gpa.free(validity_explain);
    try std.testing.expect(std.mem.indexOf(u8, validity_explain, "validity_profile(sales") != null);
    var lazy_validity = try validity_plan.collect();
    defer lazy_validity.deinit();
    try std.testing.expectEqual(@as(usize, 4), lazy_validity.height());
    try std.testing.expectEqual(@as(usize, 5), lazy_validity.width());
    const lazy_is_null = try (try lazy_validity.column("sales_is_null")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_is_null);
    const lazy_is_valid = try (try lazy_validity.column("sales_is_valid")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_is_valid);
    const lazy_valid_streak = try (try lazy_validity.column("sales_valid_streak")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_valid_streak);
    const lazy_null_streak = try (try lazy_validity.column("sales_null_streak")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_null_streak);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false }, lazy_is_null);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, lazy_is_valid);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4 }, lazy_valid_streak);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0 }, lazy_null_streak);

    var class_plan = try DeviceLazyFrame.init(gpa, table);
    defer class_plan.deinit();
    try class_plan.withColumnCompareScalar("predicted_active", "sales", f64, 4.0, .gt);
    try class_plan.classificationProfile("active", "predicted_active", "active_cls");
    try class_plan.select(&.{ "active", "predicted_active", "active_cls_tp", "active_cls_fp", "active_cls_tn", "active_cls_fn", "active_cls_correct" });
    const class_explain = try class_plan.explain(gpa);
    defer gpa.free(class_explain);
    try std.testing.expect(std.mem.indexOf(u8, class_explain, "classification_profile(actual=active, predicted=predicted_active") != null);
    var classed = try class_plan.collect();
    defer classed.deinit();
    try std.testing.expectEqual(@as(usize, 4), classed.height());
    try std.testing.expectEqual(@as(usize, 7), classed.width());
    const lazy_tp = try (try classed.column("active_cls_tp")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_tp);
    const lazy_fp = try (try classed.column("active_cls_fp")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_fp);
    const lazy_tn = try (try classed.column("active_cls_tn")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_tn);
    const lazy_fn = try (try classed.column("active_cls_fn")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_fn);
    const lazy_correct = try (try classed.column("active_cls_correct")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_correct);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, true }, lazy_tp);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false }, lazy_fp);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false }, lazy_tn);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false }, lazy_fn);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_correct);

    var crossover_plan = try DeviceLazyFrame.init(gpa, table);
    defer crossover_plan.deinit();
    try crossover_plan.withColumnScalar("units_f64", "sales", f64, 1.0, .sub);
    try crossover_plan.crossoverProfile("sales", "units_f64", "sales_units", .{ .periods = 1 });
    try crossover_plan.select(&.{ "sales", "units_f64", "sales_units_spread", "sales_units_ratio", "sales_units_cross_above", "sales_units_cross_below" });
    const crossover_explain = try crossover_plan.explain(gpa);
    defer gpa.free(crossover_explain);
    try std.testing.expect(std.mem.indexOf(u8, crossover_explain, "crossover_profile(sales,units_f64") != null);
    var crossover = try crossover_plan.collect();
    defer crossover.deinit();
    try std.testing.expectEqual(@as(usize, 4), crossover.height());
    try std.testing.expectEqual(@as(usize, 6), crossover.width());
    const lazy_spread = try (try crossover.column("sales_units_spread")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_spread);
    const lazy_ratio = try (try crossover.column("sales_units_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ratio);
    const lazy_cross_above = try (try crossover.column("sales_units_cross_above")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_cross_above);
    const lazy_cross_below = try (try crossover.column("sales_units_cross_below")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_cross_below);
    const lazy_cross_validity = try (try crossover.column("sales_units_cross_above")).bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_cross_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_cross_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_spread[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_spread[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_spread[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_spread[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_ratio[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), lazy_ratio[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.25), lazy_ratio[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 7.0 / 6.0), lazy_ratio[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false }, lazy_cross_above);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false }, lazy_cross_below);

    var fit_plan = try DeviceLazyFrame.init(gpa, table);
    defer fit_plan.deinit();
    try fit_plan.withColumnScalar("sales_minus1", "sales", f64, 1.0, .sub);
    try fit_plan.linearFitProfile("sales_minus1", "sales", "sales_fit", .{});
    try fit_plan.select(&.{ "sales", "sales_minus1", "sales_fit_fitted", "sales_fit_residual", "sales_fit_residual_zscore", "sales_fit_slope" });
    const fit_explain = try fit_plan.explain(gpa);
    defer gpa.free(fit_explain);
    try std.testing.expect(std.mem.indexOf(u8, fit_explain, "linear_fit_profile(sales_minus1->sales") != null);
    var fit = try fit_plan.collect();
    defer fit.deinit();
    try std.testing.expectEqual(@as(usize, 4), fit.height());
    try std.testing.expectEqual(@as(usize, 6), fit.width());
    const lazy_fitted = try (try fit.column("sales_fit_fitted")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_fitted);
    const lazy_fit_residual = try (try fit.column("sales_fit_residual")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_fit_residual);
    const lazy_fit_residual_z = try (try fit.column("sales_fit_residual_zscore")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_fit_residual_z);
    const lazy_fit_slope = try (try fit.column("sales_fit_slope")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_fit_slope);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), lazy_fitted[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), lazy_fitted[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), lazy_fitted[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 7.0), lazy_fitted[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_fit_residual[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_fit_residual[3], 1e-12);
    try std.testing.expect(std.math.isNan(lazy_fit_residual_z[0]));
    try std.testing.expect(std.math.isNan(lazy_fit_residual_z[3]));
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_fit_slope[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_fit_slope[3], 1e-12);

    var error_plan = try DeviceLazyFrame.init(gpa, table);
    defer error_plan.deinit();
    try error_plan.withColumnScalar("sales_minus1", "sales", f64, 1.0, .sub);
    try error_plan.errorProfile("sales", "sales_minus1", "sales_err");
    try error_plan.select(&.{ "sales", "sales_err_error", "sales_err_abs_error", "sales_err_squared_error", "sales_err_ape", "sales_err_smape" });
    const error_explain = try error_plan.explain(gpa);
    defer gpa.free(error_explain);
    try std.testing.expect(std.mem.indexOf(u8, error_explain, "error_profile(actual=sales, predicted=sales_minus1") != null);
    var lazy_errors = try error_plan.collect();
    defer lazy_errors.deinit();
    try std.testing.expectEqual(@as(usize, 4), lazy_errors.height());
    try std.testing.expectEqual(@as(usize, 6), lazy_errors.width());
    const lazy_error = try (try lazy_errors.column("sales_err_error")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_error);
    const lazy_abs_error = try (try lazy_errors.column("sales_err_abs_error")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_abs_error);
    const lazy_squared_error = try (try lazy_errors.column("sales_err_squared_error")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_squared_error);
    const lazy_ape = try (try lazy_errors.column("sales_err_ape")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ape);
    const lazy_smape = try (try lazy_errors.column("sales_err_smape")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_smape);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 1.0, 1.0, 1.0 }, lazy_error);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 1.0, 1.0, 1.0 }, lazy_abs_error);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 1.0, 1.0, 1.0 }, lazy_squared_error);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_ape[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), lazy_ape[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.2), lazy_ape[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 7.0), lazy_ape[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), lazy_smape[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 5.0), lazy_smape[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 9.0), lazy_smape[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 13.0), lazy_smape[3], 1e-12);

    var corr_plan = try DeviceLazyFrame.init(gpa, table);
    defer corr_plan.deinit();
    try corr_plan.withColumnScalar("sales_minus1", "sales", f64, 1.0, .sub);
    try corr_plan.rollingCorrelationProfile("sales_minus1", "sales", "sales_corr", .{ .window = 2, .min_periods = 2 });
    try corr_plan.select(&.{ "sales", "sales_corr_rolling_pair_count", "sales_corr_rolling_covariance", "sales_corr_rolling_correlation", "sales_corr_rolling_beta" });
    const corr_explain = try corr_plan.explain(gpa);
    defer gpa.free(corr_explain);
    try std.testing.expect(std.mem.indexOf(u8, corr_explain, "rolling_correlation_profile(sales_minus1,sales") != null);
    var rolling_corr = try corr_plan.collect();
    defer rolling_corr.deinit();
    try std.testing.expectEqual(@as(usize, 4), rolling_corr.height());
    try std.testing.expectEqual(@as(usize, 5), rolling_corr.width());
    const lazy_pair_count = try (try rolling_corr.column("sales_corr_rolling_pair_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_pair_count);
    const lazy_covariance = try (try rolling_corr.column("sales_corr_rolling_covariance")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_covariance);
    const lazy_correlation = try (try rolling_corr.column("sales_corr_rolling_correlation")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_correlation);
    const lazy_beta = try (try rolling_corr.column("sales_corr_rolling_beta")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_beta);
    const lazy_corr_validity = try (try rolling_corr.column("sales_corr_rolling_correlation")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_corr_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 2, 2 }, lazy_pair_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_corr_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), lazy_covariance[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_covariance[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_covariance[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_correlation[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_correlation[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_correlation[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_beta[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_beta[3], 1e-12);

    var bucket_plan = try DeviceLazyFrame.init(gpa, table);
    defer bucket_plan.deinit();
    try bucket_plan.bucketProfile("sales", "sales", .{ .buckets = 2, .lower_quantile = 0.25, .upper_quantile = 0.75 });
    try bucket_plan.select(&.{ "sales", "sales_ecdf", "sales_bucket", "sales_lower_tail", "sales_upper_tail" });
    const bucket_explain = try bucket_plan.explain(gpa);
    defer gpa.free(bucket_explain);
    try std.testing.expect(std.mem.indexOf(u8, bucket_explain, "bucket_profile(sales") != null);
    var bucketed = try bucket_plan.collect();
    defer bucketed.deinit();
    try std.testing.expectEqual(@as(usize, 4), bucketed.height());
    try std.testing.expectEqual(@as(usize, 5), bucketed.width());
    const lazy_ecdf = try (try bucketed.column("sales_ecdf")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_ecdf);
    const lazy_bucket = try (try bucketed.column("sales_bucket")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_bucket);
    const lazy_lower_tail = try (try bucketed.column("sales_lower_tail")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_lower_tail);
    const lazy_upper_tail = try (try bucketed.column("sales_upper_tail")).bool.toOwnedSlice(gpa);
    defer gpa.free(lazy_upper_tail);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), lazy_ecdf[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_ecdf[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), lazy_ecdf[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lazy_ecdf[3], 1e-12);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 1 }, lazy_bucket);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false }, lazy_lower_tail);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, true }, lazy_upper_tail);
}

test "device lazy frame collects groupby aggregations" {
    const gpa = std.testing.allocator;

    var store = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 1, 2, 2, 2 }, .cpu);
    defer store.deinit();
    var day = try DeviceColumn.fromSlice(i32, gpa, &.{ 10, 10, 10, 11, 11 }, .cpu);
    defer day.deinit();
    var sales = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 3.0, 5.0, 7.0, 11.0 }, .cpu);
    defer sales.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "store", .data = store },
        .{ .name = "day", .data = day },
        .{ .name = "sales", .data = sales },
    });
    defer table.deinit();

    var sum_plan = try DeviceLazyFrame.init(gpa, table);
    defer sum_plan.deinit();
    try sum_plan.filterColumnScalar("sales", f64, 2.5, .gt);
    try sum_plan.groupBySum("store", "sales", "sales_sum");
    const sum_explain = try sum_plan.explain(gpa);
    defer gpa.free(sum_explain);
    try std.testing.expect(std.mem.indexOf(u8, sum_explain, "group_by_sum(store") != null);
    var summed = try sum_plan.collect();
    defer summed.deinit();
    try std.testing.expectEqual(@as(usize, 2), summed.height());
    try std.testing.expectEqual(@as(usize, 2), summed.width());
    const sum_store = try (try summed.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(sum_store);
    const sum_values = try (try summed.column("sales_sum")).f64.toOwnedSlice(gpa);
    defer gpa.free(sum_values);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2 }, sum_store);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 23.0 }, sum_values);

    var stats_plan = try DeviceLazyFrame.init(gpa, table);
    defer stats_plan.deinit();
    try stats_plan.groupByStatsOn(&.{ "store", "day" }, "sales", "sales");
    const stats_explain = try stats_plan.explain(gpa);
    defer gpa.free(stats_explain);
    try std.testing.expect(std.mem.indexOf(u8, stats_explain, "group_by_stats_on([store,day]") != null);
    var stats = try stats_plan.collect();
    defer stats.deinit();
    try std.testing.expectEqual(@as(usize, 3), stats.height());
    try std.testing.expectEqual(@as(usize, 7), stats.width());
    const stats_store = try (try stats.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(stats_store);
    const stats_day = try (try stats.column("day")).i32.toOwnedSlice(gpa);
    defer gpa.free(stats_day);
    const stats_count = try (try stats.column("sales_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(stats_count);
    const stats_sum = try (try stats.column("sales_sum")).f64.toOwnedSlice(gpa);
    defer gpa.free(stats_sum);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2, 2 }, stats_store);
    try std.testing.expectEqualSlices(i32, &.{ 10, 10, 11 }, stats_day);
    try std.testing.expectEqualSlices(i64, &.{ 2, 1, 2 }, stats_count);
    try std.testing.expectEqualSlices(f64, &.{ 5.0, 5.0, 18.0 }, stats_sum);

    var profile_plan = try DeviceLazyFrame.init(gpa, table);
    defer profile_plan.deinit();
    try profile_plan.groupByProfile("store", "sales", "sales");
    const profile_explain = try profile_plan.explain(gpa);
    defer gpa.free(profile_explain);
    try std.testing.expect(std.mem.indexOf(u8, profile_explain, "group_by_profile(store") != null);
    var profile = try profile_plan.collect();
    defer profile.deinit();
    try std.testing.expectEqual(@as(usize, 2), profile.height());
    try std.testing.expectEqual(@as(usize, 8), profile.width());
    const profile_count = try (try profile.column("sales_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(profile_count);
    const profile_variance = try (try profile.column("sales_variance")).f64.toOwnedSlice(gpa);
    defer gpa.free(profile_variance);
    const profile_skewness = try (try profile.column("sales_skewness")).f64.toOwnedSlice(gpa);
    defer gpa.free(profile_skewness);
    const profile_kurtosis = try (try profile.column("sales_kurtosis")).f64.toOwnedSlice(gpa);
    defer gpa.free(profile_kurtosis);
    try std.testing.expectEqualSlices(i64, &.{ 2, 3 }, profile_count);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), profile_variance[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 6.222222222222222), profile_variance[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), profile_skewness[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.3818017741606059), profile_skewness[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -2.0), profile_kurtosis[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -1.5), profile_kurtosis[1], 1e-12);
}

test "device lazy frame collects multi-key joins" {
    const gpa = std.testing.allocator;

    var left_store = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 1, 2, 3 }, .cpu);
    defer left_store.deinit();
    var left_day = try DeviceColumn.fromSlice(i32, gpa, &.{ 10, 11, 10, 10 }, .cpu);
    defer left_day.deinit();
    var sales = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 3.0, 5.0, 7.0 }, .cpu);
    defer sales.deinit();
    var right_store = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 2, 9 }, .cpu);
    defer right_store.deinit();
    var right_day = try DeviceColumn.fromSlice(i32, gpa, &.{ 11, 10, 10 }, .cpu);
    defer right_day.deinit();
    var region = try DeviceColumn.fromSlice(i64, gpa, &.{ 100, 200, 900 }, .cpu);
    defer region.deinit();

    var left = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "store", .data = left_store },
        .{ .name = "day", .data = left_day },
        .{ .name = "sales", .data = sales },
    });
    defer left.deinit();
    var right = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "store", .data = right_store },
        .{ .name = "day", .data = right_day },
        .{ .name = "region", .data = region },
    });
    defer right.deinit();

    var joined_plan = try DeviceLazyFrame.init(gpa, left);
    defer joined_plan.deinit();
    try joined_plan.filterColumnScalar("sales", f64, 2.5, .gt);
    try joined_plan.innerJoinOn(right, &.{ "store", "day" }, &.{ "store", "day" }, .{});
    try joined_plan.select(&.{ "store", "day", "sales", "region" });
    const joined_explain = try joined_plan.explain(gpa);
    defer gpa.free(joined_explain);
    try std.testing.expect(std.mem.indexOf(u8, joined_explain, "inner_join_on(left=[store,day]") != null);
    var joined = try joined_plan.collect();
    defer joined.deinit();
    try std.testing.expectEqual(@as(usize, 2), joined.height());
    try std.testing.expectEqual(@as(usize, 4), joined.width());
    const joined_store = try (try joined.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(joined_store);
    const joined_sales = try (try joined.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(joined_sales);
    const joined_region = try (try joined.column("region")).i64.toOwnedSlice(gpa);
    defer gpa.free(joined_region);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2 }, joined_store);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 5.0 }, joined_sales);
    try std.testing.expectEqualSlices(i64, &.{ 100, 200 }, joined_region);

    var anti_plan = try DeviceLazyFrame.init(gpa, left);
    defer anti_plan.deinit();
    try anti_plan.antiJoinOn(right, &.{ "store", "day" }, &.{ "store", "day" });
    var anti = try anti_plan.collect();
    defer anti.deinit();
    try std.testing.expectEqual(@as(usize, 2), anti.height());
    const anti_store = try (try anti.column("store")).i32.toOwnedSlice(gpa);
    defer gpa.free(anti_store);
    try std.testing.expectEqualSlices(i32, &.{ 1, 3 }, anti_store);
}

test "device lazy frame collects asof joins" {
    const gpa = std.testing.allocator;

    var left_time = try DeviceColumn.fromSlice(i64, gpa, &.{ 1, 5, 8, 12, 20 }, .cpu);
    defer left_time.deinit();
    var value = try DeviceColumn.fromSlice(f64, gpa, &.{ 10.0, 50.0, 80.0, 120.0, 200.0 }, .cpu);
    defer value.deinit();
    var right_time = try DeviceColumn.fromSliceWithValidity(i64, gpa, &.{ 2, 6, 10, 30 }, &.{ true, true, true, false }, .cpu);
    defer right_time.deinit();
    var quote = try DeviceColumn.fromSlice(i64, gpa, &.{ 20, 60, 100, 300 }, .cpu);
    defer quote.deinit();

    var left = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "time", .data = left_time },
        .{ .name = "value", .data = value },
    });
    defer left.deinit();
    var right = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "time", .data = right_time },
        .{ .name = "quote", .data = quote },
    });
    defer right.deinit();

    var plan = try DeviceLazyFrame.init(gpa, left);
    defer plan.deinit();
    try plan.filterColumnScalar("time", i64, 4, .ge);
    try plan.asofJoin(right, "time", "time", .{ .strategy = .nearest });
    try plan.select(&.{ "time", "value", "quote" });
    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "asof_join(time->time, strategy=nearest)") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 4), result.height());
    try std.testing.expectEqual(@as(usize, 3), result.width());
    const result_time = try (try result.column("time")).i64.toOwnedSlice(gpa);
    defer gpa.free(result_time);
    const result_quote = try (try result.column("quote")).i64.toOwnedSlice(gpa);
    defer gpa.free(result_quote);
    try std.testing.expectEqual(@as(usize, 0), (try result.column("quote")).nullCount());
    try std.testing.expectEqualSlices(i64, &.{ 5, 8, 12, 20 }, result_time);
    try std.testing.expectEqualSlices(i64, &.{ 60, 60, 100, 100 }, result_quote);
}
test "device dataframe round-trips through boltha parquet" {
    const gpa = std.testing.allocator;

    var id = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 2, 3 }, .cpu);
    defer id.deinit();
    var sales = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 2.0, 3.0, 5.0 }, &.{ true, false, true }, .cpu);
    defer sales.deinit();
    var active = try DeviceColumn.fromSlice(bool, gpa, &.{ true, false, true }, .cpu);
    defer active.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "id", .data = id },
        .{ .name = "sales", .data = sales },
        .{ .name = "active", .data = active },
    });
    defer table.deinit();

    const bytes = try table.toParquetBytes(gpa);
    defer gpa.free(bytes);
    try std.testing.expect(bytes.len > 0);

    var restored = try DeviceDataFrame.fromParquetBytes(gpa, bytes, .cpu);
    defer restored.deinit();
    try std.testing.expectEqual(table.height(), restored.height());
    try std.testing.expectEqual(table.width(), restored.width());
    try std.testing.expectEqual(DeviceDType.i32, try restored.columnDType("id"));
    try std.testing.expectEqual(DeviceDType.f64, try restored.columnDType("sales"));
    try std.testing.expectEqual(DeviceDType.bool, try restored.columnDType("active"));

    const ids = try (try restored.column("id")).i32.toOwnedSlice(gpa);
    defer gpa.free(ids);
    const sales_values = try (try restored.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(sales_values);
    const sales_validity = try (try restored.column("sales")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(sales_validity);
    const active_values = try (try restored.column("active")).bool.toOwnedSlice(gpa);
    defer gpa.free(active_values);

    try std.testing.expectEqualSlices(i32, &.{ 1, 2, 3 }, ids);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 0.0, 5.0 }, sales_values);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, sales_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, active_values);
}

test "device dataframe reads boltha parquet with range pruning" {
    const gpa = std.testing.allocator;

    var id_field = try boltha.arrow.Field.init(gpa, "id", .{ .int = .{ .bit_width = 32, .signed = true } }, false);
    defer id_field.deinit(gpa);
    var sales_field = try boltha.arrow.Field.init(gpa, "sales", .{ .floating_point = .double }, false);
    defer sales_field.deinit(gpa);
    const schema = try boltha.arrow.Schema.init(gpa, &.{ id_field, sales_field });

    const batches = try gpa.alloc(boltha.arrow.RecordBatch, 2);
    const schema0 = try boltha.arrow.Schema.init(gpa, &.{ id_field, sales_field });
    const cols0 = try gpa.alloc(boltha.arrow.AnyArray, 2);
    cols0[0] = .{ .int32 = try boltha.arrow.PrimitiveArray(i32).fromSlice(gpa, &.{ 1, 2 }) };
    cols0[1] = .{ .float64 = try boltha.arrow.PrimitiveArray(f64).fromSlice(gpa, &.{ 10.0, 20.0 }) };
    batches[0] = try boltha.arrow.RecordBatch.initOwned(schema0, cols0);
    const schema1 = try boltha.arrow.Schema.init(gpa, &.{ id_field, sales_field });
    const cols1 = try gpa.alloc(boltha.arrow.AnyArray, 2);
    cols1[0] = .{ .int32 = try boltha.arrow.PrimitiveArray(i32).fromSlice(gpa, &.{ 100, 101 }) };
    cols1[1] = .{ .float64 = try boltha.arrow.PrimitiveArray(f64).fromSlice(gpa, &.{ 1000.0, 1010.0 }) };
    batches[1] = try boltha.arrow.RecordBatch.initOwned(schema1, cols1);

    var arrow_table = try boltha.arrow.Table.initOwned(schema, batches);
    defer arrow_table.deinit(gpa);
    var parquet_bytes: std.ArrayList(u8) = .empty;
    defer parquet_bytes.deinit(gpa);
    try boltha.parquet.writeTable(gpa, &parquet_bytes, arrow_table);

    var pruned = try DeviceDataFrame.fromParquetBytesPruned(
        gpa,
        parquet_bytes.items,
        "id",
        .{ .i32 = .{ .min = 100, .max = 101 } },
        .cpu,
    );
    defer pruned.deinit();
    try std.testing.expectEqual(@as(usize, 2), pruned.height());
    const ids = try (try pruned.column("id")).i32.toOwnedSlice(gpa);
    defer gpa.free(ids);
    const sales_values = try (try pruned.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(sales_values);
    try std.testing.expectEqualSlices(i32, &.{ 100, 101 }, ids);
    try std.testing.expectEqualSlices(f64, &.{ 1000.0, 1010.0 }, sales_values);

    var empty = try DeviceDataFrame.fromParquetBytesPruned(
        gpa,
        parquet_bytes.items,
        "id",
        .{ .i32 = .{ .min = 10_000 } },
        .cpu,
    );
    defer empty.deinit();
    try std.testing.expectEqual(@as(usize, 0), empty.height());
    try std.testing.expectEqual(@as(usize, 2), empty.width());
}

test "device parquet scan pushes range predicate and projection into collect" {
    const gpa = std.testing.allocator;

    var id = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 2, 3 }, .cpu);
    defer id.deinit();
    var sales = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 3.0, 5.0 }, .cpu);
    defer sales.deinit();
    var active = try DeviceColumn.fromSlice(bool, gpa, &.{ true, false, true }, .cpu);
    defer active.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "id", .data = id },
        .{ .name = "sales", .data = sales },
        .{ .name = "active", .data = active },
    });
    defer table.deinit();

    const bytes = try table.toParquetBytes(gpa);
    defer gpa.free(bytes);

    var scan = try DeviceParquetScan.init(gpa, bytes, .cpu);
    defer scan.deinit();
    try scan.whereRange("id", .{ .i32 = .{ .min = 2, .max = 3 } });
    try scan.select(&.{ "id", "sales" });

    const explain = try scan.explain(gpa);
    defer gpa.free(explain);
    try std.testing.expect(std.mem.indexOf(u8, explain, "range=id") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "projection=[id,sales]") != null);

    var result = try scan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 3), result.height());
    try std.testing.expectEqual(@as(usize, 2), result.width());
    try std.testing.expectEqual(DeviceDType.i32, try result.columnDType("id"));
    try std.testing.expectEqual(DeviceDType.f64, try result.columnDType("sales"));
    try std.testing.expectEqual(@as(?usize, null), result.columnIndex("active"));
}

test "device lazy frame pushes scalar filters and projection into parquet scan source" {
    const gpa = std.testing.allocator;

    var id = try DeviceColumn.fromSlice(i32, gpa, &.{ 1, 2, 3 }, .cpu);
    defer id.deinit();
    var sales = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 3.0, 5.0 }, .cpu);
    defer sales.deinit();
    var active = try DeviceColumn.fromSlice(bool, gpa, &.{ true, false, true }, .cpu);
    defer active.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "id", .data = id },
        .{ .name = "sales", .data = sales },
        .{ .name = "active", .data = active },
    });
    defer table.deinit();

    const bytes = try table.toParquetBytes(gpa);
    defer gpa.free(bytes);

    var lazy_scan = try DeviceLazyFrame.scanParquetBytes(gpa, bytes, .cpu);
    defer lazy_scan.deinit();
    try lazy_scan.withColumnScalar("sales_x2", "sales", f64, 2.0, .mul);
    try lazy_scan.filterColumnScalar("sales", f64, 2.5, .gt);
    try lazy_scan.select(&.{ "sales_x2", "id" });

    const explain = try lazy_scan.explain(gpa);
    defer gpa.free(explain);
    try std.testing.expect(std.mem.indexOf(u8, explain, "source=parquet_scan") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "with_column_scalar(sales_x2") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain, "scan_pushdown: range=sales, projection=[sales,id]") != null);

    var result = try lazy_scan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 2), result.height());
    try std.testing.expectEqual(@as(usize, 2), result.width());
    try std.testing.expectEqual(@as(?usize, null), result.columnIndex("active"));
    try std.testing.expectEqual(@as(?usize, null), result.columnIndex("sales"));
    const result_sales_x2 = try (try result.column("sales_x2")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_x2);
    try std.testing.expectEqualSlices(f64, &.{ 6.0, 10.0 }, result_sales_x2);
}
