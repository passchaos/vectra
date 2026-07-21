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
