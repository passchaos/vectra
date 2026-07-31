//! Generic typed device column storage and per-type kernels.
//!
//! This module intentionally depends only on low-level array, numeric, view,
//! option, and validity-core helpers. Keeping it independent from the tagged
//! `DeviceColumn` union prevents import cycles while allowing the public facade
//! in `dataframe_device_column.zig` to re-export the same `DeviceTypedColumn`
//! API.

const std = @import("std");
const array_mod = @import("array.zig");
const array_helpers_mod = @import("dataframe_array_helpers.zig");
const dataframe_view_mod = @import("dataframe_view.zig");
const numeric_mod = @import("dataframe_numeric.zig");
const options_mod = @import("dataframe_options.zig");
const validity_core_mod = @import("dataframe_validity_core.zig");

const DeviceDType = array_mod.DType;
const DeviceColumnView = dataframe_view_mod.DeviceColumnView;
const DeviceColumnBinaryOp = options_mod.DeviceColumnBinaryOp;
const DeviceColumnCompareOp = options_mod.DeviceColumnCompareOp;
const countNulls = validity_core_mod.countNulls;
const countNullsInArray = validity_core_mod.countNullsInArray;
const validityValues = validity_core_mod.validityValues;
const requireCompatibleColumnArrays = array_helpers_mod.requireCompatibleColumnArrays;
const combineValidityMasks = array_helpers_mod.combineValidityMasks;
const zeroValue = array_helpers_mod.zeroValue;
const rowIndicesFromMask = array_helpers_mod.rowIndicesFromMask;
const sliceArray1d = array_helpers_mod.sliceArray1d;
const takeArray1d = array_helpers_mod.takeArray1d;
const isIntegerColumnType = numeric_mod.isIntegerColumnType;
const isOrderedColumnType = numeric_mod.isOrderedColumnType;

fn isComplexColumnType(comptime T: type) bool {
    return T == array_mod.Complex64 or T == array_mod.Complex128;
}

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

        pub fn cast(self: Self, comptime U: type) array_mod.ArrayError!DeviceTypedColumn(U) {
            var values = try self.values.astype(U);
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

        pub fn fillNull(self: Self, value: T) array_mod.ArrayError!Self {
            const maybe_validity = try validityValues(self, self.values.allocator);
            defer if (maybe_validity) |validity| self.values.allocator.free(validity);
            if (maybe_validity == null) return self.clone();

            const host_values = try self.values.toOwnedSlice(self.values.allocator);
            defer self.values.allocator.free(host_values);
            for (host_values, maybe_validity.?) |*slot, valid| {
                if (!valid) slot.* = value;
            }
            return Self.fromSlice(self.values.allocator, host_values, self.device());
        }

        pub fn abs(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool) return error.TypeUnsupported;
            var values = try self.values.abs();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn neg(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool) return error.TypeUnsupported;
            var values = try self.values.neg();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn square(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool) return error.TypeUnsupported;
            var values = try self.values.square();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn reciprocal(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or isIntegerColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.reciprocal();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn sign(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.sign();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn sqrt(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or isIntegerColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.sqrt();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn rsqrt(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or isIntegerColumnType(T) or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.rsqrt();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn cbrt(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or isIntegerColumnType(T) or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.cbrt();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn floor(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.floor();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn ceil(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.ceil();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn round(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.round();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn trunc(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.trunc();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn deg2rad(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or isIntegerColumnType(T) or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.deg2rad();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn rad2deg(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or isIntegerColumnType(T) or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.rad2deg();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn expit(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or isIntegerColumnType(T) or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.expit();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn logit(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or T == f16 or isIntegerColumnType(T) or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.logit();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn softplus(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or T == f16 or T == array_mod.BFloat16 or isIntegerColumnType(T) or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.softplus();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn logsigmoid(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or T == f16 or isIntegerColumnType(T) or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.logsigmoid();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn relu(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.relu();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn leakyRelu(self: Self, negative_slope: T) array_mod.ArrayError!Self {
            if (comptime T == bool or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.leakyRelu(negative_slope);
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn relu6(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.relu6();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn powScalar(self: Self, scalar: T) array_mod.ArrayError!Self {
            if (comptime T == bool or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.powScalar(scalar);
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn floorDivScalar(self: Self, scalar: T) array_mod.ArrayError!Self {
            if (comptime T == bool or T == array_mod.BFloat16 or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.floorDivScalar(scalar);
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn modScalar(self: Self, scalar: T) array_mod.ArrayError!Self {
            if (comptime T == bool or T == array_mod.BFloat16 or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.modScalar(scalar);
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn remainderScalar(self: Self, scalar: T) array_mod.ArrayError!Self {
            return self.modScalar(scalar);
        }

        pub fn logAddExpScalar(self: Self, scalar: T) array_mod.ArrayError!Self {
            if (comptime T == bool or T == f16 or T == array_mod.BFloat16 or isIntegerColumnType(T) or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.logAddExpScalar(scalar);
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn logAddExp2Scalar(self: Self, scalar: T) array_mod.ArrayError!Self {
            if (comptime T == bool or T == f16 or T == array_mod.BFloat16 or isIntegerColumnType(T) or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.logAddExp2Scalar(scalar);
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn xlogyScalar(self: Self, scalar: T) array_mod.ArrayError!Self {
            if (comptime T == bool or T == f16 or T == array_mod.BFloat16 or isIntegerColumnType(T) or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.xlogyScalar(scalar);
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn fmaxScalar(self: Self, scalar: T) array_mod.ArrayError!Self {
            if (comptime T == bool or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.fmaxScalar(scalar);
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn fminScalar(self: Self, scalar: T) array_mod.ArrayError!Self {
            if (comptime T == bool or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.fminScalar(scalar);
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn hypotScalar(self: Self, scalar: T) array_mod.ArrayError!Self {
            if (comptime T == bool or T == array_mod.BFloat16 or isIntegerColumnType(T) or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.hypotScalar(scalar);
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn atan2Scalar(self: Self, scalar: T) array_mod.ArrayError!Self {
            if (comptime T == bool or T == f16 or T == array_mod.BFloat16 or isIntegerColumnType(T) or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.atan2Scalar(scalar);
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn nextAfterScalar(self: Self, scalar: T) array_mod.ArrayError!Self {
            if (comptime T == bool or T == array_mod.BFloat16 or isIntegerColumnType(T) or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.nextAfterScalar(scalar);
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn copysignScalar(self: Self, scalar: T) array_mod.ArrayError!Self {
            if (comptime T == bool or T == array_mod.BFloat16 or isIntegerColumnType(T) or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.copysignScalar(scalar);
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn heavisideScalar(self: Self, value_at_zero: T) array_mod.ArrayError!Self {
            if (comptime T == bool or T == array_mod.BFloat16 or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.heavisideScalar(value_at_zero);
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn ldexpScalar(self: Self, exponent: i32) array_mod.ArrayError!Self {
            if (comptime T == bool or T == array_mod.BFloat16 or isIntegerColumnType(T) or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.ldexpScalar(exponent);
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn threshold(self: Self, threshold_value: T, replacement_value: T) array_mod.ArrayError!Self {
            if (comptime T == bool or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.threshold(threshold_value, replacement_value);
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn hardtanh(self: Self, min_value: T, max_value: T) array_mod.ArrayError!Self {
            if (comptime T == bool or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.hardtanh(min_value, max_value);
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn maximumScalar(self: Self, scalar: T) array_mod.ArrayError!Self {
            if (comptime T == bool or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.maximumScalar(scalar);
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn minimumScalar(self: Self, scalar: T) array_mod.ArrayError!Self {
            if (comptime T == bool or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.minimumScalar(scalar);
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn clipMin(self: Self, min_value: T) array_mod.ArrayError!Self {
            return self.maximumScalar(min_value);
        }

        pub fn clipMax(self: Self, max_value: T) array_mod.ArrayError!Self {
            return self.minimumScalar(max_value);
        }

        pub fn hardshrink(self: Self, lambd: T) array_mod.ArrayError!Self {
            if (comptime T == bool or isIntegerColumnType(T) or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.hardshrink(lambd);
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn softshrink(self: Self, lambd: T) array_mod.ArrayError!Self {
            if (comptime T == bool or isIntegerColumnType(T) or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.softshrink(lambd);
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn tanhshrink(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or isIntegerColumnType(T) or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.tanhshrink();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn elu(self: Self, alpha: T) array_mod.ArrayError!Self {
            if (comptime T == bool or isIntegerColumnType(T) or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.elu(alpha);
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn celu(self: Self, alpha: T) array_mod.ArrayError!Self {
            if (comptime T == bool or isIntegerColumnType(T) or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.celu(alpha);
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn softsign(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or T == array_mod.BFloat16 or isIntegerColumnType(T) or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.softsign();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn hardsigmoid(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or isIntegerColumnType(T) or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.hardsigmoid();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn hardswish(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or isIntegerColumnType(T) or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.hardswish();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn silu(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or isIntegerColumnType(T) or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.silu();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn swish(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or isIntegerColumnType(T) or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.swish();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn mish(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or T == f16 or isIntegerColumnType(T) or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.mish();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn gelu(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or isIntegerColumnType(T) or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.gelu();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn selu(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or isIntegerColumnType(T) or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.selu();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn exp(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or isIntegerColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.exp();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn exp2(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or isIntegerColumnType(T) or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.exp2();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn expm1(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or isIntegerColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.expm1();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn sin(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or isIntegerColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.sin();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn cos(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or isIntegerColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.cos();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn tan(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or isIntegerColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.tan();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn asin(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or isIntegerColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.asin();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn acos(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or isIntegerColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.acos();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn atan(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or isIntegerColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.atan();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn sinh(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or T == f16 or isIntegerColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.sinh();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn cosh(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or T == f16 or isIntegerColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.cosh();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn tanh(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or isIntegerColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.tanh();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn asinh(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or isIntegerColumnType(T) or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.asinh();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn acosh(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or isIntegerColumnType(T) or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.acosh();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn atanh(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or isIntegerColumnType(T) or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.atanh();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn log(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or isIntegerColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.log();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn log1p(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or isIntegerColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.log1p();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn lgamma(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or isIntegerColumnType(T) or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.lgamma();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn sinc(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or isIntegerColumnType(T) or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.sinc();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn log2(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or isIntegerColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.log2();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn log10(self: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or isIntegerColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.log10();
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
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

        pub fn lerpScalar(self: Self, end: Self, weight: T) array_mod.ArrayError!Self {
            if (comptime T == bool or isIntegerColumnType(T) or isComplexColumnType(T)) return error.TypeUnsupported;
            try requireCompatibleColumnArrays(T, self.values, end.values);
            var values = try self.values.lerpScalar(end.values, weight);
            errdefer values.deinit();
            var validity = try combineValidityMasks(self.values.allocator, self.validity, end.validity, self.len(), self.device());
            errdefer if (validity) |*mask| mask.deinit();
            const nulls = if (validity) |mask| try countNullsInArray(mask) else 0;
            return .{ .values = values, .validity = validity, .null_count = nulls };
        }

        pub fn fusedTernaryScalar(self: Self, input1: Self, input2: Self, value: T, comptime op: enum { addcmul, addcdiv }) array_mod.ArrayError!Self {
            if (comptime T == bool or isComplexColumnType(T)) return error.TypeUnsupported;
            if (comptime op == .addcdiv and isIntegerColumnType(T)) return error.TypeUnsupported;
            try requireCompatibleColumnArrays(T, self.values, input1.values);
            try requireCompatibleColumnArrays(T, self.values, input2.values);
            var values = switch (op) {
                .addcmul => try self.values.addcmul(input1.values, input2.values, value),
                .addcdiv => try self.values.addcdiv(input1.values, input2.values, value),
            };
            errdefer values.deinit();
            var validity = try combineValidityMasks(self.values.allocator, self.validity, input1.validity, self.len(), self.device());
            errdefer if (validity) |*mask| mask.deinit();
            if (input2.validity) |mask| {
                var combined = try combineValidityMasks(self.values.allocator, validity, mask, self.len(), self.device());
                errdefer if (combined) |*combined_mask| combined_mask.deinit();
                if (validity) |*old_mask| old_mask.deinit();
                validity = combined;
            }
            const nulls = if (validity) |mask| try countNullsInArray(mask) else 0;
            return .{ .values = values, .validity = validity, .null_count = nulls };
        }

        pub fn clipArray(self: Self, min_values: Self, max_values: Self) array_mod.ArrayError!Self {
            if (comptime T == bool or isComplexColumnType(T)) return error.TypeUnsupported;
            try requireCompatibleColumnArrays(T, self.values, min_values.values);
            try requireCompatibleColumnArrays(T, self.values, max_values.values);
            var values = try self.values.clipArray(min_values.values, max_values.values);
            errdefer values.deinit();
            var validity = try combineValidityMasks(self.values.allocator, self.validity, min_values.validity, self.len(), self.device());
            errdefer if (validity) |*mask| mask.deinit();
            if (max_values.validity) |mask| {
                var combined = try combineValidityMasks(self.values.allocator, validity, mask, self.len(), self.device());
                errdefer if (combined) |*combined_mask| combined_mask.deinit();
                if (validity) |*old_mask| old_mask.deinit();
                validity = combined;
            }
            const nulls = if (validity) |mask| try countNullsInArray(mask) else 0;
            return .{ .values = values, .validity = validity, .null_count = nulls };
        }

        pub fn isinColumn(self: Self, test_elements: Self, invert: bool) array_mod.ArrayError!DeviceTypedColumn(bool) {
            if (comptime T == array_mod.BFloat16 or isComplexColumnType(T)) return error.TypeUnsupported;
            if (!self.device().sameDevice(test_elements.device())) return error.InvalidDevice;
            var candidates: array_mod.Array(T) = undefined;
            var owns_candidates = false;
            if (test_elements.validity) |validity_mask| {
                const raw_values = try test_elements.values.toOwnedSlice(self.values.allocator);
                defer self.values.allocator.free(raw_values);
                const validity_values = try validity_mask.toOwnedSlice(self.values.allocator);
                defer self.values.allocator.free(validity_values);
                var valid_count: usize = 0;
                for (validity_values) |valid| valid_count += @intFromBool(valid);
                const filtered = try self.values.allocator.alloc(T, valid_count);
                defer self.values.allocator.free(filtered);
                var write: usize = 0;
                for (raw_values, validity_values) |value, valid| {
                    if (valid) {
                        filtered[write] = value;
                        write += 1;
                    }
                }
                candidates = try array_mod.Array(T).fromSliceOn(self.values.allocator, filtered, &.{filtered.len}, self.device());
                owns_candidates = true;
            } else {
                candidates = test_elements.values;
            }
            defer if (owns_candidates) candidates.deinit();

            var values = try self.values.isin(candidates, invert);
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn whereScalar(self: Self, mask_column: DeviceTypedColumn(bool), other_value: T) array_mod.ArrayError!Self {
            if (!self.device().sameDevice(mask_column.device())) return error.InvalidDevice;
            if (mask_column.values.shape.len != 1 or mask_column.len() != self.len()) return error.ShapeMismatch;
            var values = try self.values.whereScalar(mask_column.values, other_value);
            errdefer values.deinit();
            var validity = try combineValidityMasks(self.values.allocator, self.validity, mask_column.validity, self.len(), self.device());
            errdefer if (validity) |*mask| mask.deinit();
            const nulls = if (validity) |mask| try countNullsInArray(mask) else 0;
            return .{ .values = values, .validity = validity, .null_count = nulls };
        }

        pub fn whereColumn(self: Self, mask_column: DeviceTypedColumn(bool), other: Self) array_mod.ArrayError!Self {
            if (!self.device().sameDevice(mask_column.device())) return error.InvalidDevice;
            if (mask_column.values.shape.len != 1 or mask_column.len() != self.len()) return error.ShapeMismatch;
            try requireCompatibleColumnArrays(T, self.values, other.values);
            var values = try self.values.where(mask_column.values, other.values);
            errdefer values.deinit();
            var validity = try combineValidityMasks(self.values.allocator, self.validity, mask_column.validity, self.len(), self.device());
            errdefer if (validity) |*mask| mask.deinit();
            if (other.validity) |mask| {
                var combined = try combineValidityMasks(self.values.allocator, validity, mask, self.len(), self.device());
                errdefer if (combined) |*combined_mask| combined_mask.deinit();
                if (validity) |*old_mask| old_mask.deinit();
                validity = combined;
            }
            const nulls = if (validity) |mask| try countNullsInArray(mask) else 0;
            return .{ .values = values, .validity = validity, .null_count = nulls };
        }

        pub fn maskedPutScalar(self: Self, mask_column: DeviceTypedColumn(bool), value: T) array_mod.ArrayError!Self {
            if (!self.device().sameDevice(mask_column.device())) return error.InvalidDevice;
            if (mask_column.values.shape.len != 1 or mask_column.len() != self.len()) return error.ShapeMismatch;
            var values = try self.values.maskedPutScalar(mask_column.values, value);
            errdefer values.deinit();
            var validity = try combineValidityMasks(self.values.allocator, self.validity, mask_column.validity, self.len(), self.device());
            errdefer if (validity) |*mask| mask.deinit();
            const nulls = if (validity) |mask| try countNullsInArray(mask) else 0;
            return .{ .values = values, .validity = validity, .null_count = nulls };
        }

        pub fn putFlat(self: Self, row_indices: []const usize, value_column: Self) array_mod.ArrayError!Self {
            if (!self.device().sameDevice(value_column.device())) return error.InvalidDevice;
            if (value_column.len() != 1 and value_column.len() != row_indices.len) return error.ShapeMismatch;
            var indices = try array_mod.Array(usize).fromSliceOn(self.values.allocator, row_indices, &.{row_indices.len}, self.device());
            defer indices.deinit();
            var values = try self.values.putFlat(indices, value_column.values);
            errdefer values.deinit();

            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity != null or value_column.validity != null) {
                const allocator = self.values.allocator;
                const validity_values = try allocator.alloc(bool, self.len());
                defer allocator.free(validity_values);
                if (self.validity) |mask| {
                    const existing = try mask.toOwnedSlice(allocator);
                    defer allocator.free(existing);
                    @memcpy(validity_values, existing);
                } else {
                    @memset(validity_values, true);
                }

                const value_validity = if (value_column.validity) |mask| try mask.toOwnedSlice(allocator) else null;
                defer if (value_validity) |mask| allocator.free(mask);
                const scalar_value_column = value_column.len() == 1;
                // Mirror Array.putFlat's sequential write contract for null
                // metadata as well as values. Duplicate indices are therefore
                // resolved by the last incoming value, matching the data write.
                for (row_indices, 0..) |row_index, i| {
                    if (row_index >= self.len()) return error.IndexOutOfBounds;
                    const value_index = if (scalar_value_column) 0 else i;
                    validity_values[row_index] = if (value_validity) |mask| mask[value_index] else true;
                }
                validity = try array_mod.Array(bool).fromSliceOn(allocator, validity_values, &.{validity_values.len}, self.device());
            }
            const nulls = if (validity) |mask| try countNullsInArray(mask) else 0;
            return .{ .values = values, .validity = validity, .null_count = nulls };
        }

        pub fn putFlatScalar(self: Self, row_indices: []const usize, value: T) array_mod.ArrayError!Self {
            var indices = try array_mod.Array(usize).fromSliceOn(self.values.allocator, row_indices, &.{row_indices.len}, self.device());
            defer indices.deinit();
            var values = try self.values.putFlatScalar(indices, value);
            errdefer values.deinit();

            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| {
                const validity_values = try mask.toOwnedSlice(self.values.allocator);
                defer self.values.allocator.free(validity_values);
                for (row_indices) |row_index| {
                    if (row_index >= self.len()) return error.IndexOutOfBounds;
                    validity_values[row_index] = true;
                }
                validity = try array_mod.Array(bool).fromSliceOn(self.values.allocator, validity_values, &.{validity_values.len}, self.device());
            }
            const nulls = if (validity) |mask| try countNullsInArray(mask) else 0;
            return .{ .values = values, .validity = validity, .null_count = nulls };
        }

        pub fn putFlatScalarMode(self: Self, row_indices: []const usize, value: T, mode: array_mod.IndexMode) array_mod.ArrayError!Self {
            var indices = try array_mod.Array(usize).fromSliceOn(self.values.allocator, row_indices, &.{row_indices.len}, self.device());
            defer indices.deinit();
            var values = try self.values.putFlatScalarMode(indices, value, mode);
            errdefer values.deinit();

            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| {
                if (self.len() == 0 and row_indices.len != 0) return error.IndexOutOfBounds;
                const validity_values = try mask.toOwnedSlice(self.values.allocator);
                defer self.values.allocator.free(validity_values);
                for (row_indices) |row_index| {
                    const normalized = switch (mode) {
                        .raise => if (row_index >= self.len()) return error.IndexOutOfBounds else row_index,
                        .wrap => row_index % self.len(),
                        .clip => @min(row_index, self.len() - 1),
                    };
                    validity_values[normalized] = true;
                }
                validity = try array_mod.Array(bool).fromSliceOn(self.values.allocator, validity_values, &.{validity_values.len}, self.device());
            }
            const nulls = if (validity) |mask| try countNullsInArray(mask) else 0;
            return .{ .values = values, .validity = validity, .null_count = nulls };
        }

        pub fn putFlatScalarSigned(self: Self, row_indices: []const isize, value: T) array_mod.ArrayError!Self {
            var indices = try array_mod.Array(isize).fromSliceOn(self.values.allocator, row_indices, &.{row_indices.len}, self.device());
            defer indices.deinit();
            var values = try self.values.putFlatScalarSigned(indices, value);
            errdefer values.deinit();

            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| {
                const validity_values = try mask.toOwnedSlice(self.values.allocator);
                defer self.values.allocator.free(validity_values);
                const signed_len: isize = @intCast(self.len());
                for (row_indices) |row_index| {
                    const normalized = if (row_index < 0) signed_len + row_index else row_index;
                    if (normalized < 0 or normalized >= signed_len) return error.IndexOutOfBounds;
                    validity_values[@intCast(normalized)] = true;
                }
                validity = try array_mod.Array(bool).fromSliceOn(self.values.allocator, validity_values, &.{validity_values.len}, self.device());
            }
            const nulls = if (validity) |mask| try countNullsInArray(mask) else 0;
            return .{ .values = values, .validity = validity, .null_count = nulls };
        }

        pub fn logicalScalar(self: Self, scalar: bool, comptime op: enum { @"and", @"or", xor }) array_mod.ArrayError!Self {
            if (comptime T != bool) return error.TypeUnsupported;
            var values = switch (op) {
                .@"and" => try self.values.logicalAndScalar(scalar),
                .@"or" => try self.values.logicalOrScalar(scalar),
                .xor => try self.values.logicalXorScalar(scalar),
            };
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn logical(self: Self, other: Self, comptime op: enum { @"and", @"or", xor }) array_mod.ArrayError!Self {
            if (comptime T != bool) return error.TypeUnsupported;
            try requireCompatibleColumnArrays(T, self.values, other.values);
            var values = switch (op) {
                .@"and" => try self.values.logicalAnd(other.values),
                .@"or" => try self.values.logicalOr(other.values),
                .xor => try self.values.logicalXor(other.values),
            };
            errdefer values.deinit();
            var validity = try combineValidityMasks(self.values.allocator, self.validity, other.validity, self.len(), self.device());
            errdefer if (validity) |*mask| mask.deinit();
            const nulls = if (validity) |mask| try countNullsInArray(mask) else 0;
            return .{ .values = values, .validity = validity, .null_count = nulls };
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

        pub fn iscloseScalar(self: Self, scalar: T, rtol: T, atol: T, equal_nan: bool) array_mod.ArrayError!DeviceTypedColumn(bool) {
            if (comptime T == bool or isIntegerColumnType(T) or isComplexColumnType(T)) return error.TypeUnsupported;
            var values = try self.values.iscloseScalarEqualNan(scalar, rtol, atol, equal_nan);
            errdefer values.deinit();
            var validity: ?array_mod.Array(bool) = null;
            errdefer if (validity) |*mask| mask.deinit();
            if (self.validity) |mask| validity = try mask.clone();
            return .{ .values = values, .validity = validity, .null_count = self.null_count };
        }

        pub fn allcloseScalar(self: Self, scalar: T, rtol: T, atol: T, equal_nan: bool) array_mod.ArrayError!bool {
            if (comptime T == bool or isIntegerColumnType(T) or isComplexColumnType(T)) return error.TypeUnsupported;
            // Treat nullable columns conservatively: a column with unknown
            // values is not "all close" to a scalar unless callers first
            // impute/filter it explicitly.
            if (self.hasNulls()) return false;
            return self.values.allcloseScalarEqualNan(scalar, rtol, atol, equal_nan);
        }

        fn isZeroValue(value: T) bool {
            if (comptime T == array_mod.BFloat16) return value.eql(zeroValue(T));
            if (comptime T == array_mod.Complex64 or T == array_mod.Complex128) return value.re == 0 and value.im == 0;
            return value == zeroValue(T);
        }

        fn addValue(lhs: T, rhs: T) T {
            if (comptime T == array_mod.BFloat16) return lhs.add(rhs);
            if (comptime T == array_mod.Complex64 or T == array_mod.Complex128) return lhs.add(rhs);
            return lhs + rhs;
        }

        pub fn sum(self: Self) array_mod.ArrayError!T {
            if (comptime T == bool) return error.TypeUnsupported;
            const values = try self.values.toOwnedSlice(self.values.allocator);
            defer self.values.allocator.free(values);
            const maybe_validity = try validityValues(self, self.values.allocator);
            defer if (maybe_validity) |validity| self.values.allocator.free(validity);
            var total = zeroValue(T);
            for (values, 0..) |value, row| {
                if (maybe_validity) |validity| {
                    if (!validity[row]) continue;
                }
                total = addValue(total, value);
            }
            return total;
        }

        pub fn countNonzero(self: Self) array_mod.ArrayError!usize {
            const values = try self.values.toOwnedSlice(self.values.allocator);
            defer self.values.allocator.free(values);
            const maybe_validity = try validityValues(self, self.values.allocator);
            defer if (maybe_validity) |validity| self.values.allocator.free(validity);
            var count: usize = 0;
            for (values, 0..) |value, row| {
                if (maybe_validity) |validity| {
                    if (!validity[row]) continue;
                }
                if (!isZeroValue(value)) count += 1;
            }
            return count;
        }

        pub fn any(self: Self) array_mod.ArrayError!bool {
            if (comptime T != bool) return error.TypeUnsupported;
            const values = try self.values.toOwnedSlice(self.values.allocator);
            defer self.values.allocator.free(values);
            const maybe_validity = try validityValues(self, self.values.allocator);
            defer if (maybe_validity) |validity| self.values.allocator.free(validity);
            for (values, 0..) |value, row| {
                if (maybe_validity) |validity| {
                    if (!validity[row]) continue;
                }
                if (value) return true;
            }
            return false;
        }

        pub fn all(self: Self) array_mod.ArrayError!bool {
            if (comptime T != bool) return error.TypeUnsupported;
            const values = try self.values.toOwnedSlice(self.values.allocator);
            defer self.values.allocator.free(values);
            const maybe_validity = try validityValues(self, self.values.allocator);
            defer if (maybe_validity) |validity| self.values.allocator.free(validity);
            for (values, 0..) |value, row| {
                if (maybe_validity) |validity| {
                    if (!validity[row]) continue;
                }
                if (!value) return false;
            }
            return true;
        }

        pub fn countTrue(self: Self) array_mod.ArrayError!usize {
            if (comptime T != bool) return error.TypeUnsupported;
            return self.countNonzero();
        }

        pub fn countFalse(self: Self) array_mod.ArrayError!usize {
            if (comptime T != bool) return error.TypeUnsupported;
            const values = try self.values.toOwnedSlice(self.values.allocator);
            defer self.values.allocator.free(values);
            const maybe_validity = try validityValues(self, self.values.allocator);
            defer if (maybe_validity) |validity| self.values.allocator.free(validity);
            var count: usize = 0;
            for (values, 0..) |value, row| {
                if (maybe_validity) |validity| {
                    if (!validity[row]) continue;
                }
                if (!value) count += 1;
            }
            return count;
        }

        pub fn toOwnedSlice(self: Self, allocator: std.mem.Allocator) array_mod.ArrayError![]T {
            return self.values.toOwnedSlice(allocator);
        }
    };
}
