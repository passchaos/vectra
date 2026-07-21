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
            inline else => |typed, tag| .{ .bool = try typed.compare(@field(other, @tagName(tag)), op) },
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

    pub fn compareColumns(self: DeviceDataFrame, lhs_name: []const u8, rhs_name: []const u8, op: DeviceColumnCompareOp) DeviceDataError!DeviceColumn {
        const lhs = try self.column(lhs_name);
        const rhs = try self.column(rhs_name);
        return lhs.compare(rhs.*, op);
    }

    pub fn compareColumnScalar(self: DeviceDataFrame, name: []const u8, comptime T: type, scalar: T, op: DeviceColumnCompareOp) DeviceDataError!DeviceColumn {
        const col = try self.column(name);
        return col.compareScalar(T, scalar, op);
    }

    pub fn filterColumnMask(self: DeviceDataFrame, mask: DeviceColumn) DeviceDataError!DeviceDataFrame {
        const typed_mask = switch (mask) {
            .bool => |typed| typed,
            else => return error.TypeMismatch,
        };
        if (!typed_mask.device().sameDevice(self.device)) return error.InvalidDevice;
        if (typed_mask.len() != self.rows) return error.LengthMismatch;
        if (typed_mask.hasNulls()) return error.TypeUnsupported;
        const host_mask = try typed_mask.values.toOwnedSlice(self.allocator);
        defer self.allocator.free(host_mask);
        return self.filter(host_mask);
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
        return groupBySumDispatchKey(self.allocator, key_name, output_name, key.*, value.*, self.device);
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

        var left_rows = try self.take(pair.left);
        defer left_rows.deinit();
        var right_rows = try right.take(pair.right);
        defer right_rows.deinit();

        return concatJoinedTables(self.allocator, left_rows, right_rows, right_key_name, options_value);
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

fn groupBySumDispatchKey(
    allocator: std.mem.Allocator,
    key_name: []const u8,
    output_name: []const u8,
    key: DeviceColumn,
    value: DeviceColumn,
    device_value: array_mod.Device,
) DeviceDataError!DeviceDataFrame {
    return switch (key) {
        .bool => |typed| groupBySumDispatchValue(bool, allocator, key_name, output_name, typed, value, device_value),
        .i8 => |typed| groupBySumDispatchValue(i8, allocator, key_name, output_name, typed, value, device_value),
        .i16 => |typed| groupBySumDispatchValue(i16, allocator, key_name, output_name, typed, value, device_value),
        .i32 => |typed| groupBySumDispatchValue(i32, allocator, key_name, output_name, typed, value, device_value),
        .i64 => |typed| groupBySumDispatchValue(i64, allocator, key_name, output_name, typed, value, device_value),
        .u8 => |typed| groupBySumDispatchValue(u8, allocator, key_name, output_name, typed, value, device_value),
        .u16 => |typed| groupBySumDispatchValue(u16, allocator, key_name, output_name, typed, value, device_value),
        .u32 => |typed| groupBySumDispatchValue(u32, allocator, key_name, output_name, typed, value, device_value),
        .u64 => |typed| groupBySumDispatchValue(u64, allocator, key_name, output_name, typed, value, device_value),
        .usize => |typed| groupBySumDispatchValue(usize, allocator, key_name, output_name, typed, value, device_value),
        .isize => |typed| groupBySumDispatchValue(isize, allocator, key_name, output_name, typed, value, device_value),
        .f16 => |typed| groupBySumDispatchValue(f16, allocator, key_name, output_name, typed, value, device_value),
        .f32 => |typed| groupBySumDispatchValue(f32, allocator, key_name, output_name, typed, value, device_value),
        .f64 => |typed| groupBySumDispatchValue(f64, allocator, key_name, output_name, typed, value, device_value),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn groupBySumDispatchValue(
    comptime K: type,
    allocator: std.mem.Allocator,
    key_name: []const u8,
    output_name: []const u8,
    key: DeviceTypedColumn(K),
    value: DeviceColumn,
    device_value: array_mod.Device,
) DeviceDataError!DeviceDataFrame {
    return switch (value) {
        .i8 => |typed| groupBySumTyped(K, i8, allocator, key_name, output_name, key, typed, device_value),
        .i16 => |typed| groupBySumTyped(K, i16, allocator, key_name, output_name, key, typed, device_value),
        .i32 => |typed| groupBySumTyped(K, i32, allocator, key_name, output_name, key, typed, device_value),
        .i64 => |typed| groupBySumTyped(K, i64, allocator, key_name, output_name, key, typed, device_value),
        .u8 => |typed| groupBySumTyped(K, u8, allocator, key_name, output_name, key, typed, device_value),
        .u16 => |typed| groupBySumTyped(K, u16, allocator, key_name, output_name, key, typed, device_value),
        .u32 => |typed| groupBySumTyped(K, u32, allocator, key_name, output_name, key, typed, device_value),
        .u64 => |typed| groupBySumTyped(K, u64, allocator, key_name, output_name, key, typed, device_value),
        .usize => |typed| groupBySumTyped(K, usize, allocator, key_name, output_name, key, typed, device_value),
        .isize => |typed| groupBySumTyped(K, isize, allocator, key_name, output_name, key, typed, device_value),
        .f16 => |typed| groupBySumTyped(K, f16, allocator, key_name, output_name, key, typed, device_value),
        .f32 => |typed| groupBySumTyped(K, f32, allocator, key_name, output_name, key, typed, device_value),
        .f64 => |typed| groupBySumTyped(K, f64, allocator, key_name, output_name, key, typed, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

fn groupBySumTyped(
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
    var sums: std.ArrayList(V) = .empty;
    defer sums.deinit(allocator);

    for (keys, values, 0..) |key_value, value_item, row| {
        if (maybe_key_validity) |validity| {
            if (!validity[row]) continue;
        }
        if (maybe_value_validity) |validity| {
            if (!validity[row]) continue;
        }
        const group_index = findGroupIndex(K, unique_keys.items, key_value) orelse blk: {
            try unique_keys.append(allocator, key_value);
            try sums.append(allocator, zeroValue(V));
            break :blk unique_keys.items.len - 1;
        };
        sums.items[group_index] += value_item;
    }

    const key_col = try DeviceColumn.fromSlice(K, allocator, unique_keys.items, device_value);
    const sum_col = try DeviceColumn.fromSlice(V, allocator, sums.items, device_value);
    return initAggregatedDataFrame(allocator, key_name, key_col, output_name, sum_col, device_value);
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

const JoinRowIndexPair = struct {
    allocator: std.mem.Allocator,
    left: []usize,
    right: []usize,

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

    var left_indices: std.ArrayList(usize) = .empty;
    errdefer left_indices.deinit(allocator);
    var right_indices: std.ArrayList(usize) = .empty;
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

fn concatJoinedTables(
    allocator: std.mem.Allocator,
    left: DeviceDataFrame,
    right: DeviceDataFrame,
    right_key_name: []const u8,
    options_value: DeviceJoinOptions,
) DeviceDataError!DeviceDataFrame {
    if (!left.device.sameDevice(right.device)) return error.InvalidDevice;
    if (left.rows != right.rows) return error.LengthMismatch;

    const total_cols = left.columns.len + right.columns.len - @as(usize, @intFromBool(right.columnIndex(right_key_name) != null));
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
        if (std.mem.eql(u8, name, right_key_name)) continue;
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
    try std.testing.expectError(error.TypeUnsupported, table.filterColumnMask(units_mask));
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

test "device dataframe groupby count and sum fixed-width columns" {
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
