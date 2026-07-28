const std = @import("std");
const series_mod = @import("series.zig");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const dataframe_arrow_mod = @import("dataframe_arrow.zig");
const dataframe_view_mod = @import("dataframe_view.zig");
const numeric_mod = @import("dataframe_numeric.zig");
const options_mod = @import("dataframe_options.zig");
const validity_mod = @import("dataframe_validity.zig");
const boltha = @import("boltha");

const DeviceDType = array_mod.DType;
const DeviceColumnView = dataframe_view_mod.DeviceColumnView;
const DeviceColumnBinaryOp = options_mod.DeviceColumnBinaryOp;
const DeviceColumnCompareOp = options_mod.DeviceColumnCompareOp;
const DeviceSortOptions = options_mod.DeviceSortOptions;
const countNulls = validity_mod.countNulls;
const countNullsInArray = validity_mod.countNullsInArray;
const validityValues = validity_mod.validityValues;
const requireCompatibleColumnArrays = dataframe_array_mod.requireCompatibleColumnArrays;
const combineValidityMasks = dataframe_array_mod.combineValidityMasks;
const zeroValue = dataframe_array_mod.zeroValue;
const rowIndicesFromMask = dataframe_array_mod.rowIndicesFromMask;
const sliceArray1d = dataframe_array_mod.sliceArray1d;
const takeArray1d = dataframe_array_mod.takeArray1d;
const isIntegerColumnType = numeric_mod.isIntegerColumnType;
const isOrderedColumnType = numeric_mod.isOrderedColumnType;
const compareSortValues = numeric_mod.compareSortValues;
const DeviceDataError = series_mod.DataError || array_mod.ArrayError;
const ArrowInteropError = DeviceDataError || boltha.arrow.ArrayError || boltha.arrow.RecordBatchError || boltha.arrow.TableError;
const deviceDTypeToArrowDataType = dataframe_arrow_mod.deviceDTypeToArrowDataType;
const primitiveColumnToArrow = dataframe_arrow_mod.primitiveColumnToArrow;
const boolColumnToArrow = dataframe_arrow_mod.boolColumnToArrow;
const indexColumnToArrow = dataframe_arrow_mod.indexColumnToArrow;

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

pub fn argsortTypedColumn(comptime T: type, column: DeviceTypedColumn(T), allocator: std.mem.Allocator, options_value: DeviceSortOptions) array_mod.ArrayError![]usize {
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
