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
const oneValue = array_helpers_mod.oneValue;
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
        const RealStats = struct {
            count: usize,
            mean: f64,
            m2: f64,
            m3: f64,
            m4: f64,
        };

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

        fn isPositiveZeroValue(value: T) bool {
            if (comptime T == array_mod.BFloat16) {
                const widened = value.toF32();
                return widened == 0 and !std.math.signbit(widened);
            }
            if (comptime T == array_mod.Complex64 or T == array_mod.Complex128) {
                return value.re == 0 and value.im == 0 and !std.math.signbit(value.re) and !std.math.signbit(value.im);
            }
            return switch (@typeInfo(T)) {
                .float => value == 0 and !std.math.signbit(value),
                else => false,
            };
        }

        fn isNegativeZeroValue(value: T) bool {
            if (comptime T == array_mod.BFloat16) {
                const widened = value.toF32();
                return widened == 0 and std.math.signbit(widened);
            }
            if (comptime T == array_mod.Complex64 or T == array_mod.Complex128) {
                return value.re == 0 and value.im == 0 and (std.math.signbit(value.re) or std.math.signbit(value.im));
            }
            return switch (@typeInfo(T)) {
                .float => value == 0 and std.math.signbit(value),
                else => false,
            };
        }

        fn isPositiveValue(value: T) bool {
            if (comptime T == array_mod.BFloat16) return value.toF32() > 0;
            if (comptime T == array_mod.Complex64 or T == array_mod.Complex128) return false;
            return switch (@typeInfo(T)) {
                .float, .comptime_float, .int, .comptime_int => value > 0,
                else => false,
            };
        }

        fn isNegativeValue(value: T) bool {
            if (comptime T == array_mod.BFloat16) return value.toF32() < 0;
            if (comptime T == array_mod.Complex64 or T == array_mod.Complex128) return false;
            return switch (@typeInfo(T)) {
                .float, .comptime_float, .int, .comptime_int => value < 0,
                else => false,
            };
        }

        fn isSignBitValue(value: T) bool {
            if (comptime T == array_mod.BFloat16) return std.math.signbit(value.toF32());
            if (comptime T == array_mod.Complex64 or T == array_mod.Complex128) return false;
            return switch (@typeInfo(T)) {
                .float, .comptime_float => std.math.signbit(value),
                .int, .comptime_int => value < 0,
                else => false,
            };
        }

        fn isNanValue(value: T) bool {
            if (comptime T == array_mod.BFloat16) return std.math.isNan(value.toF32());
            if (comptime T == array_mod.Complex64 or T == array_mod.Complex128) return std.math.isNan(value.re) or std.math.isNan(value.im);
            return switch (@typeInfo(T)) {
                .float => std.math.isNan(value),
                else => false,
            };
        }

        fn isInfValue(value: T) bool {
            if (comptime T == array_mod.BFloat16) return std.math.isInf(value.toF32());
            if (comptime T == array_mod.Complex64 or T == array_mod.Complex128) return std.math.isInf(value.re) or std.math.isInf(value.im);
            return switch (@typeInfo(T)) {
                .float => std.math.isInf(value),
                else => false,
            };
        }

        fn isPositiveInfValue(value: T) bool {
            if (comptime T == array_mod.BFloat16) return std.math.isPositiveInf(value.toF32());
            if (comptime T == array_mod.Complex64 or T == array_mod.Complex128) return std.math.isPositiveInf(value.re) or std.math.isPositiveInf(value.im);
            return switch (@typeInfo(T)) {
                .float => std.math.isPositiveInf(value),
                else => false,
            };
        }

        fn isNegativeInfValue(value: T) bool {
            if (comptime T == array_mod.BFloat16) return std.math.isNegativeInf(value.toF32());
            if (comptime T == array_mod.Complex64 or T == array_mod.Complex128) return std.math.isNegativeInf(value.re) or std.math.isNegativeInf(value.im);
            return switch (@typeInfo(T)) {
                .float => std.math.isNegativeInf(value),
                else => false,
            };
        }

        fn isFiniteValue(value: T) bool {
            if (comptime T == array_mod.BFloat16) return std.math.isFinite(value.toF32());
            if (comptime T == array_mod.Complex64 or T == array_mod.Complex128) return std.math.isFinite(value.re) and std.math.isFinite(value.im);
            return switch (@typeInfo(T)) {
                .float => std.math.isFinite(value),
                else => true,
            };
        }

        fn isNormalValue(value: T) bool {
            if (comptime T == array_mod.BFloat16) return std.math.isNormal(value.toF32());
            if (comptime T == array_mod.Complex64 or T == array_mod.Complex128) return std.math.isNormal(value.re) and std.math.isNormal(value.im);
            return switch (@typeInfo(T)) {
                .float => std.math.isNormal(value),
                else => false,
            };
        }

        fn isSubnormalFloat(comptime F: type, value: F) bool {
            return std.math.isFinite(value) and !std.math.isNormal(value) and value != 0;
        }

        fn isSubnormalValue(value: T) bool {
            if (comptime T == array_mod.BFloat16) return isSubnormalFloat(f32, value.toF32());
            if (comptime T == array_mod.Complex64 or T == array_mod.Complex128) {
                return isSubnormalFloat(@TypeOf(value.re), value.re) or isSubnormalFloat(@TypeOf(value.im), value.im);
            }
            return switch (@typeInfo(T)) {
                .float => isSubnormalFloat(T, value),
                else => false,
            };
        }

        fn floatKeyEqual(comptime F: type, lhs: F, rhs: F) bool {
            const lhs_nan = std.math.isNan(lhs);
            const rhs_nan = std.math.isNan(rhs);
            return if (lhs_nan or rhs_nan) lhs_nan and rhs_nan else lhs == rhs;
        }

        fn distinctValueEqual(lhs: T, rhs: T) bool {
            if (comptime T == array_mod.BFloat16) return floatKeyEqual(f32, lhs.toF32(), rhs.toF32());
            if (comptime T == array_mod.Complex64) {
                return floatKeyEqual(f32, lhs.re, rhs.re) and floatKeyEqual(f32, lhs.im, rhs.im);
            }
            if (comptime T == array_mod.Complex128) {
                return floatKeyEqual(f64, lhs.re, rhs.re) and floatKeyEqual(f64, lhs.im, rhs.im);
            }
            return switch (@typeInfo(T)) {
                .float => floatKeyEqual(T, lhs, rhs),
                else => lhs == rhs,
            };
        }

        fn addValue(lhs: T, rhs: T) T {
            if (comptime T == array_mod.BFloat16) return lhs.add(rhs);
            if (comptime T == array_mod.Complex64 or T == array_mod.Complex128) return lhs.add(rhs);
            return lhs + rhs;
        }

        fn mulValue(lhs: T, rhs: T) T {
            if (comptime T == array_mod.BFloat16) return lhs.mul(rhs);
            if (comptime T == array_mod.Complex64 or T == array_mod.Complex128) return lhs.mul(rhs);
            return lhs * rhs;
        }

        fn subValue(lhs: T, rhs: T) T {
            if (comptime T == array_mod.BFloat16) return lhs.sub(rhs);
            if (comptime T == array_mod.Complex64 or T == array_mod.Complex128) return lhs.sub(rhs);
            return lhs - rhs;
        }

        fn divByCount(value: T, count: usize) T {
            if (comptime T == array_mod.BFloat16) return value.div(array_mod.BFloat16.fromF64(@floatFromInt(count)));
            return value / @as(T, @floatFromInt(count));
        }

        fn lessValue(lhs: T, rhs: T) bool {
            if (comptime T == array_mod.BFloat16) return lhs.lt(rhs);
            return lhs < rhs;
        }

        fn realValueToF64(value: T) f64 {
            if (comptime T == array_mod.BFloat16) return value.toF64();
            return switch (@typeInfo(T)) {
                .float, .comptime_float => @floatCast(value),
                .int, .comptime_int => @floatFromInt(value),
                else => @compileError("realValueToF64 requires a real numeric column value"),
            };
        }

        fn quantileLess(_: void, lhs: f64, rhs: f64) bool {
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

        fn realStats(self: Self) array_mod.ArrayError!RealStats {
            if (comptime T == bool or isComplexColumnType(T)) return error.TypeUnsupported;
            const values = try self.values.toOwnedSlice(self.values.allocator);
            defer self.values.allocator.free(values);
            const maybe_validity = try validityValues(self, self.values.allocator);
            defer if (maybe_validity) |validity| self.values.allocator.free(validity);

            var stats: RealStats = .{ .count = 0, .mean = 0.0, .m2 = 0.0, .m3 = 0.0, .m4 = 0.0 };
            for (values, 0..) |value, row| {
                if (maybe_validity) |validity| {
                    if (!validity[row]) continue;
                }
                // Online central moment accumulation avoids the worst
                // cancellation from raw-power sums while providing variance,
                // skewness, and excess kurtosis from one pass over nullable
                // rows. The formulas match the grouped/window moment helpers
                // so scalar and profile APIs report the same population
                // moments.
                const previous_count = stats.count;
                stats.count += 1;

                const n: f64 = @floatFromInt(stats.count);
                const previous_n: f64 = @floatFromInt(previous_count);
                const x = realValueToF64(value);
                const delta = x - stats.mean;
                const delta_n = delta / n;
                const delta_n2 = delta_n * delta_n;
                const term1 = delta * delta_n * previous_n;
                const previous_m2 = stats.m2;
                const previous_m3 = stats.m3;

                stats.mean += delta_n;
                stats.m4 += term1 * delta_n2 * (n * n - 3.0 * n + 3.0) + 6.0 * delta_n2 * previous_m2 - 4.0 * delta_n * previous_m3;
                stats.m3 += term1 * delta_n * (n - 2.0) - 3.0 * delta_n * previous_m2;
                stats.m2 += term1;
            }
            if (stats.count == 0) return error.EmptyArray;
            return stats;
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

        pub fn prod(self: Self) array_mod.ArrayError!T {
            if (comptime T == bool) return error.TypeUnsupported;
            const values = try self.values.toOwnedSlice(self.values.allocator);
            defer self.values.allocator.free(values);
            const maybe_validity = try validityValues(self, self.values.allocator);
            defer if (maybe_validity) |validity| self.values.allocator.free(validity);
            // Match array product semantics for empty/all-null inputs: the
            // multiplicative identity is the neutral result after skipping
            // nullable rows, just as sum starts from zero.
            var total = oneValue(T);
            for (values, 0..) |value, row| {
                if (maybe_validity) |validity| {
                    if (!validity[row]) continue;
                }
                total = mulValue(total, value);
            }
            return total;
        }

        pub fn mean(self: Self) array_mod.ArrayError!T {
            if (comptime T == bool or isIntegerColumnType(T) or isComplexColumnType(T)) return error.TypeUnsupported;
            const values = try self.values.toOwnedSlice(self.values.allocator);
            defer self.values.allocator.free(values);
            const maybe_validity = try validityValues(self, self.values.allocator);
            defer if (maybe_validity) |validity| self.values.allocator.free(validity);
            var total = zeroValue(T);
            var count: usize = 0;
            for (values, 0..) |value, row| {
                if (maybe_validity) |validity| {
                    if (!validity[row]) continue;
                }
                total = addValue(total, value);
                count += 1;
            }
            if (count == 0) return error.EmptyArray;
            return divByCount(total, count);
        }

        pub fn min(self: Self) array_mod.ArrayError!T {
            if (comptime T == bool or isComplexColumnType(T)) return error.TypeUnsupported;
            const values = try self.values.toOwnedSlice(self.values.allocator);
            defer self.values.allocator.free(values);
            const maybe_validity = try validityValues(self, self.values.allocator);
            defer if (maybe_validity) |validity| self.values.allocator.free(validity);
            var found = false;
            var best = zeroValue(T);
            for (values, 0..) |value, row| {
                if (maybe_validity) |validity| {
                    if (!validity[row]) continue;
                }
                if (!found or lessValue(value, best)) {
                    best = value;
                    found = true;
                }
            }
            if (!found) return error.EmptyArray;
            return best;
        }

        pub fn max(self: Self) array_mod.ArrayError!T {
            if (comptime T == bool or isComplexColumnType(T)) return error.TypeUnsupported;
            const values = try self.values.toOwnedSlice(self.values.allocator);
            defer self.values.allocator.free(values);
            const maybe_validity = try validityValues(self, self.values.allocator);
            defer if (maybe_validity) |validity| self.values.allocator.free(validity);
            var found = false;
            var best = zeroValue(T);
            for (values, 0..) |value, row| {
                if (maybe_validity) |validity| {
                    if (!validity[row]) continue;
                }
                if (!found or lessValue(best, value)) {
                    best = value;
                    found = true;
                }
            }
            if (!found) return error.EmptyArray;
            return best;
        }

        pub fn ptp(self: Self) array_mod.ArrayError!T {
            if (comptime T == bool or isComplexColumnType(T)) return error.TypeUnsupported;
            const values = try self.values.toOwnedSlice(self.values.allocator);
            defer self.values.allocator.free(values);
            const maybe_validity = try validityValues(self, self.values.allocator);
            defer if (maybe_validity) |validity| self.values.allocator.free(validity);
            var found = false;
            var low = zeroValue(T);
            var high = zeroValue(T);
            for (values, 0..) |value, row| {
                if (maybe_validity) |validity| {
                    if (!validity[row]) continue;
                }
                if (!found) {
                    low = value;
                    high = value;
                    found = true;
                    continue;
                }
                if (lessValue(value, low)) low = value;
                if (lessValue(high, value)) high = value;
            }
            if (!found) return error.EmptyArray;
            return subValue(high, low);
        }

        fn argExtreme(self: Self, comptime want_max: bool) array_mod.ArrayError!usize {
            if (comptime T == bool or isComplexColumnType(T)) return error.TypeUnsupported;
            const values = try self.values.toOwnedSlice(self.values.allocator);
            defer self.values.allocator.free(values);
            const maybe_validity = try validityValues(self, self.values.allocator);
            defer if (maybe_validity) |validity| self.values.allocator.free(validity);
            var found = false;
            var best = zeroValue(T);
            var best_row: usize = 0;
            for (values, 0..) |value, row| {
                if (maybe_validity) |validity| {
                    if (!validity[row]) continue;
                }
                if (!found) {
                    best = value;
                    best_row = row;
                    found = true;
                    continue;
                }
                // Use strict comparisons so ties keep the first physical row,
                // matching stable argmin/argmax expectations while still
                // returning source-table row positions after nulls are skipped.
                const better = if (want_max) lessValue(best, value) else lessValue(value, best);
                if (better) {
                    best = value;
                    best_row = row;
                }
            }
            if (!found) return error.EmptyArray;
            return best_row;
        }

        pub fn argmin(self: Self) array_mod.ArrayError!usize {
            return self.argExtreme(false);
        }

        pub fn argmax(self: Self) array_mod.ArrayError!usize {
            return self.argExtreme(true);
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

        fn countMatching(self: Self, comptime predicate: fn (T) bool) array_mod.ArrayError!usize {
            const values = try self.values.toOwnedSlice(self.values.allocator);
            defer self.values.allocator.free(values);
            const maybe_validity = try validityValues(self, self.values.allocator);
            defer if (maybe_validity) |validity| self.values.allocator.free(validity);
            var count: usize = 0;
            for (values, 0..) |value, row| {
                if (maybe_validity) |validity| {
                    if (!validity[row]) continue;
                }
                if (predicate(value)) count += 1;
            }
            return count;
        }

        fn firstMatchingIndex(self: Self, comptime predicate: fn (T) bool) array_mod.ArrayError!?usize {
            const values = try self.values.toOwnedSlice(self.values.allocator);
            defer self.values.allocator.free(values);
            const maybe_validity = try validityValues(self, self.values.allocator);
            defer if (maybe_validity) |validity| self.values.allocator.free(validity);
            for (values, 0..) |value, row| {
                if (maybe_validity) |validity| {
                    if (!validity[row]) continue;
                }
                if (predicate(value)) return row;
            }
            return null;
        }

        fn lastMatchingIndex(self: Self, comptime predicate: fn (T) bool) array_mod.ArrayError!?usize {
            const values = try self.values.toOwnedSlice(self.values.allocator);
            defer self.values.allocator.free(values);
            const maybe_validity = try validityValues(self, self.values.allocator);
            defer if (maybe_validity) |validity| self.values.allocator.free(validity);
            var row = values.len;
            while (row > 0) {
                row -= 1;
                if (maybe_validity) |validity| {
                    if (!validity[row]) continue;
                }
                if (predicate(values[row])) return row;
            }
            return null;
        }

        pub fn countNan(self: Self) array_mod.ArrayError!usize {
            return self.countMatching(isNanValue);
        }

        pub fn countPositiveZero(self: Self) array_mod.ArrayError!usize {
            return self.countMatching(isPositiveZeroValue);
        }

        pub fn countNegativeZero(self: Self) array_mod.ArrayError!usize {
            return self.countMatching(isNegativeZeroValue);
        }

        pub fn countPositive(self: Self) array_mod.ArrayError!usize {
            return self.countMatching(isPositiveValue);
        }

        pub fn countNegative(self: Self) array_mod.ArrayError!usize {
            return self.countMatching(isNegativeValue);
        }

        pub fn countSignBit(self: Self) array_mod.ArrayError!usize {
            return self.countMatching(isSignBitValue);
        }

        pub fn countInf(self: Self) array_mod.ArrayError!usize {
            return self.countMatching(isInfValue);
        }

        pub fn countPositiveInf(self: Self) array_mod.ArrayError!usize {
            return self.countMatching(isPositiveInfValue);
        }

        pub fn countNegativeInf(self: Self) array_mod.ArrayError!usize {
            return self.countMatching(isNegativeInfValue);
        }

        pub fn countFinite(self: Self) array_mod.ArrayError!usize {
            return self.countMatching(isFiniteValue);
        }

        pub fn countNonFinite(self: Self) array_mod.ArrayError!usize {
            return self.countMatching(struct {
                fn f(value: T) bool {
                    return !isFiniteValue(value);
                }
            }.f);
        }

        pub fn countNormal(self: Self) array_mod.ArrayError!usize {
            return self.countMatching(isNormalValue);
        }

        pub fn countSubnormal(self: Self) array_mod.ArrayError!usize {
            return self.countMatching(isSubnormalValue);
        }

        pub fn firstZeroIndex(self: Self) array_mod.ArrayError!?usize {
            return self.firstMatchingIndex(isZeroValue);
        }

        pub fn lastZeroIndex(self: Self) array_mod.ArrayError!?usize {
            return self.lastMatchingIndex(isZeroValue);
        }

        pub fn firstPositiveZeroIndex(self: Self) array_mod.ArrayError!?usize {
            return self.firstMatchingIndex(isPositiveZeroValue);
        }

        pub fn lastPositiveZeroIndex(self: Self) array_mod.ArrayError!?usize {
            return self.lastMatchingIndex(isPositiveZeroValue);
        }

        pub fn firstNegativeZeroIndex(self: Self) array_mod.ArrayError!?usize {
            return self.firstMatchingIndex(isNegativeZeroValue);
        }

        pub fn lastNegativeZeroIndex(self: Self) array_mod.ArrayError!?usize {
            return self.lastMatchingIndex(isNegativeZeroValue);
        }

        pub fn firstNonzeroIndex(self: Self) array_mod.ArrayError!?usize {
            return self.firstMatchingIndex(struct {
                fn f(value: T) bool {
                    return !isZeroValue(value);
                }
            }.f);
        }

        pub fn lastNonzeroIndex(self: Self) array_mod.ArrayError!?usize {
            return self.lastMatchingIndex(struct {
                fn f(value: T) bool {
                    return !isZeroValue(value);
                }
            }.f);
        }

        pub fn firstPositiveIndex(self: Self) array_mod.ArrayError!?usize {
            return self.firstMatchingIndex(isPositiveValue);
        }

        pub fn lastPositiveIndex(self: Self) array_mod.ArrayError!?usize {
            return self.lastMatchingIndex(isPositiveValue);
        }

        pub fn firstNegativeIndex(self: Self) array_mod.ArrayError!?usize {
            return self.firstMatchingIndex(isNegativeValue);
        }

        pub fn lastNegativeIndex(self: Self) array_mod.ArrayError!?usize {
            return self.lastMatchingIndex(isNegativeValue);
        }

        pub fn firstSignBitIndex(self: Self) array_mod.ArrayError!?usize {
            return self.firstMatchingIndex(isSignBitValue);
        }

        pub fn lastSignBitIndex(self: Self) array_mod.ArrayError!?usize {
            return self.lastMatchingIndex(isSignBitValue);
        }

        pub fn firstNanIndex(self: Self) array_mod.ArrayError!?usize {
            return self.firstMatchingIndex(isNanValue);
        }

        pub fn lastNanIndex(self: Self) array_mod.ArrayError!?usize {
            return self.lastMatchingIndex(isNanValue);
        }

        pub fn firstInfIndex(self: Self) array_mod.ArrayError!?usize {
            return self.firstMatchingIndex(isInfValue);
        }

        pub fn lastInfIndex(self: Self) array_mod.ArrayError!?usize {
            return self.lastMatchingIndex(isInfValue);
        }

        pub fn firstPositiveInfIndex(self: Self) array_mod.ArrayError!?usize {
            return self.firstMatchingIndex(isPositiveInfValue);
        }

        pub fn lastPositiveInfIndex(self: Self) array_mod.ArrayError!?usize {
            return self.lastMatchingIndex(isPositiveInfValue);
        }

        pub fn firstNegativeInfIndex(self: Self) array_mod.ArrayError!?usize {
            return self.firstMatchingIndex(isNegativeInfValue);
        }

        pub fn lastNegativeInfIndex(self: Self) array_mod.ArrayError!?usize {
            return self.lastMatchingIndex(isNegativeInfValue);
        }

        pub fn firstFiniteIndex(self: Self) array_mod.ArrayError!?usize {
            return self.firstMatchingIndex(isFiniteValue);
        }

        pub fn lastFiniteIndex(self: Self) array_mod.ArrayError!?usize {
            return self.lastMatchingIndex(isFiniteValue);
        }

        pub fn firstNonFiniteIndex(self: Self) array_mod.ArrayError!?usize {
            return self.firstMatchingIndex(struct {
                fn f(value: T) bool {
                    return !isFiniteValue(value);
                }
            }.f);
        }

        pub fn lastNonFiniteIndex(self: Self) array_mod.ArrayError!?usize {
            return self.lastMatchingIndex(struct {
                fn f(value: T) bool {
                    return !isFiniteValue(value);
                }
            }.f);
        }

        pub fn firstValidIndex(self: Self) array_mod.ArrayError!?usize {
            if (!self.nullable()) return if (self.len() == 0) null else 0;
            const validity = try validityValues(self, self.values.allocator);
            defer if (validity) |mask| self.values.allocator.free(mask);
            for (validity.?, 0..) |is_valid, row| {
                if (is_valid) return row;
            }
            return null;
        }

        pub fn lastValidIndex(self: Self) array_mod.ArrayError!?usize {
            if (!self.nullable()) return if (self.len() == 0) null else self.len() - 1;
            const validity = try validityValues(self, self.values.allocator);
            defer if (validity) |mask| self.values.allocator.free(mask);
            var row = validity.?.len;
            while (row > 0) {
                row -= 1;
                if (validity.?[row]) return row;
            }
            return null;
        }

        pub fn firstNullIndex(self: Self) array_mod.ArrayError!?usize {
            if (!self.nullable() or self.null_count == 0) return null;
            const validity = try validityValues(self, self.values.allocator);
            defer if (validity) |mask| self.values.allocator.free(mask);
            for (validity.?, 0..) |is_valid, row| {
                if (!is_valid) return row;
            }
            return null;
        }

        pub fn lastNullIndex(self: Self) array_mod.ArrayError!?usize {
            if (!self.nullable() or self.null_count == 0) return null;
            const validity = try validityValues(self, self.values.allocator);
            defer if (validity) |mask| self.values.allocator.free(mask);
            var row = validity.?.len;
            while (row > 0) {
                row -= 1;
                if (!validity.?[row]) return row;
            }
            return null;
        }

        pub fn countDistinct(self: Self) array_mod.ArrayError!usize {
            const values = try self.values.toOwnedSlice(self.values.allocator);
            defer self.values.allocator.free(values);
            const maybe_validity = try validityValues(self, self.values.allocator);
            defer if (maybe_validity) |validity| self.values.allocator.free(validity);
            var distinct_count: usize = 0;
            for (values, 0..) |value, row| {
                if (maybe_validity) |validity| {
                    if (!validity[row]) continue;
                }
                var seen = false;
                // This intentionally stays as a small host-side baseline until
                // a per-device hash-set kernel exists. It preserves dataframe
                // null semantics now (skip null rows) and NaN-key semantics
                // used by joins/groups (all NaNs of the same dtype compare
                // equal for distinct counting).
                for (values[0..row], 0..) |previous, previous_row| {
                    if (maybe_validity) |validity| {
                        if (!validity[previous_row]) continue;
                    }
                    if (distinctValueEqual(previous, value)) {
                        seen = true;
                        break;
                    }
                }
                if (!seen) distinct_count += 1;
            }
            return distinct_count;
        }

        pub fn nUnique(self: Self) array_mod.ArrayError!usize {
            return self.countDistinct();
        }

        pub fn modeValue(self: Self) array_mod.ArrayError!T {
            const values = try self.values.toOwnedSlice(self.values.allocator);
            defer self.values.allocator.free(values);
            const maybe_validity = try validityValues(self, self.values.allocator);
            defer if (maybe_validity) |validity| self.values.allocator.free(validity);
            var found = false;
            var best = zeroValue(T);
            var best_count: usize = 0;
            for (values, 0..) |value, row| {
                if (maybe_validity) |validity| {
                    if (!validity[row]) continue;
                }

                var seen = false;
                for (values[0..row], 0..) |previous, previous_row| {
                    if (maybe_validity) |validity| {
                        if (!validity[previous_row]) continue;
                    }
                    if (distinctValueEqual(previous, value)) {
                        seen = true;
                        break;
                    }
                }
                if (seen) continue;

                var count: usize = 0;
                for (values[row..], row..) |candidate, candidate_row| {
                    if (maybe_validity) |validity| {
                        if (!validity[candidate_row]) continue;
                    }
                    if (distinctValueEqual(value, candidate)) count += 1;
                }

                // Ties intentionally keep the first distinct valid value. That
                // makes `mode` deterministic for unsorted dataframe columns and
                // matches the stable "first occurrence wins" policy used by
                // argmin/argmax above.
                if (!found or count > best_count) {
                    best = value;
                    best_count = count;
                    found = true;
                }
            }
            if (!found) return error.EmptyArray;
            return best;
        }

        pub fn quantile(self: Self, q: f64) array_mod.ArrayError!f64 {
            if (comptime T == bool or isComplexColumnType(T)) return error.TypeUnsupported;
            if (std.math.isNan(q) or q < 0.0 or q > 1.0) return error.InvalidShape;
            const values = try self.values.toOwnedSlice(self.values.allocator);
            defer self.values.allocator.free(values);
            const maybe_validity = try validityValues(self, self.values.allocator);
            defer if (maybe_validity) |validity| self.values.allocator.free(validity);
            const scratch = try self.values.allocator.alloc(f64, values.len);
            defer self.values.allocator.free(scratch);
            var count: usize = 0;
            for (values, 0..) |value, row| {
                if (maybe_validity) |validity| {
                    if (!validity[row]) continue;
                }
                scratch[count] = realValueToF64(value);
                count += 1;
            }
            if (count == 0) return error.EmptyArray;
            std.sort.insertion(f64, scratch[0..count], {}, quantileLess);
            return quantileFromSorted(scratch[0..count], q);
        }

        pub fn median(self: Self) array_mod.ArrayError!f64 {
            return self.quantile(0.5);
        }

        pub fn variance(self: Self, correction: f64) array_mod.ArrayError!f64 {
            if (std.math.isNan(correction) or correction < 0.0) return error.InvalidShape;
            const stats = try self.realStats();
            const denom = @as(f64, @floatFromInt(stats.count)) - correction;
            if (denom <= 0.0) return std.math.nan(f64);
            return stats.m2 / denom;
        }

        pub fn stddev(self: Self, correction: f64) array_mod.ArrayError!f64 {
            return std.math.sqrt(try self.variance(correction));
        }

        pub fn sem(self: Self, correction: f64) array_mod.ArrayError!f64 {
            if (std.math.isNan(correction) or correction < 0.0) return error.InvalidShape;
            const stats = try self.realStats();
            const denom = @as(f64, @floatFromInt(stats.count)) - correction;
            if (denom <= 0.0) return std.math.nan(f64);
            const stddev_value = std.math.sqrt(stats.m2 / denom);
            return stddev_value / std.math.sqrt(@as(f64, @floatFromInt(stats.count)));
        }

        pub fn cv(self: Self, correction: f64) array_mod.ArrayError!f64 {
            if (std.math.isNan(correction) or correction < 0.0) return error.InvalidShape;
            const stats = try self.realStats();
            const denom = @as(f64, @floatFromInt(stats.count)) - correction;
            if (denom <= 0.0) return std.math.nan(f64);
            return std.math.sqrt(stats.m2 / denom) / stats.mean;
        }

        pub fn skewness(self: Self) array_mod.ArrayError!f64 {
            const stats = try self.realStats();
            if (stats.count < 2 or stats.m2 == 0.0) return std.math.nan(f64);
            const n: f64 = @floatFromInt(stats.count);
            return std.math.sqrt(n) * stats.m3 / std.math.pow(f64, stats.m2, 1.5);
        }

        pub fn kurtosis(self: Self) array_mod.ArrayError!f64 {
            const stats = try self.realStats();
            if (stats.count < 2 or stats.m2 == 0.0) return std.math.nan(f64);
            const n: f64 = @floatFromInt(stats.count);
            return n * stats.m4 / (stats.m2 * stats.m2) - 3.0;
        }

        pub fn meanAbs(self: Self) array_mod.ArrayError!f64 {
            if (comptime T == bool or isComplexColumnType(T)) return error.TypeUnsupported;
            const values = try self.values.toOwnedSlice(self.values.allocator);
            defer self.values.allocator.free(values);
            const maybe_validity = try validityValues(self, self.values.allocator);
            defer if (maybe_validity) |validity| self.values.allocator.free(validity);
            var total: f64 = 0.0;
            var count: usize = 0;
            for (values, 0..) |value, row| {
                if (maybe_validity) |validity| {
                    if (!validity[row]) continue;
                }
                total += @abs(realValueToF64(value));
                count += 1;
            }
            if (count == 0) return error.EmptyArray;
            return total / @as(f64, @floatFromInt(count));
        }

        pub fn rms(self: Self) array_mod.ArrayError!f64 {
            if (comptime T == bool or isComplexColumnType(T)) return error.TypeUnsupported;
            const values = try self.values.toOwnedSlice(self.values.allocator);
            defer self.values.allocator.free(values);
            const maybe_validity = try validityValues(self, self.values.allocator);
            defer if (maybe_validity) |validity| self.values.allocator.free(validity);
            var total: f64 = 0.0;
            var count: usize = 0;
            for (values, 0..) |value, row| {
                if (maybe_validity) |validity| {
                    if (!validity[row]) continue;
                }
                const x = realValueToF64(value);
                total += x * x;
                count += 1;
            }
            if (count == 0) return error.EmptyArray;
            return std.math.sqrt(total / @as(f64, @floatFromInt(count)));
        }

        pub fn l1Norm(self: Self) array_mod.ArrayError!f64 {
            if (comptime T == bool or isComplexColumnType(T)) return error.TypeUnsupported;
            const values = try self.values.toOwnedSlice(self.values.allocator);
            defer self.values.allocator.free(values);
            const maybe_validity = try validityValues(self, self.values.allocator);
            defer if (maybe_validity) |validity| self.values.allocator.free(validity);
            var total: f64 = 0.0;
            var count: usize = 0;
            for (values, 0..) |value, row| {
                if (maybe_validity) |validity| {
                    if (!validity[row]) continue;
                }
                total += @abs(realValueToF64(value));
                count += 1;
            }
            if (count == 0) return error.EmptyArray;
            return total;
        }

        pub fn l2Norm(self: Self) array_mod.ArrayError!f64 {
            if (comptime T == bool or isComplexColumnType(T)) return error.TypeUnsupported;
            const values = try self.values.toOwnedSlice(self.values.allocator);
            defer self.values.allocator.free(values);
            const maybe_validity = try validityValues(self, self.values.allocator);
            defer if (maybe_validity) |validity| self.values.allocator.free(validity);
            var total: f64 = 0.0;
            var count: usize = 0;
            for (values, 0..) |value, row| {
                if (maybe_validity) |validity| {
                    if (!validity[row]) continue;
                }
                const x = realValueToF64(value);
                total += x * x;
                count += 1;
            }
            if (count == 0) return error.EmptyArray;
            return std.math.sqrt(total);
        }

        pub fn geometricMean(self: Self) array_mod.ArrayError!f64 {
            if (comptime T == bool or isComplexColumnType(T)) return error.TypeUnsupported;
            const values = try self.values.toOwnedSlice(self.values.allocator);
            defer self.values.allocator.free(values);
            const maybe_validity = try validityValues(self, self.values.allocator);
            defer if (maybe_validity) |validity| self.values.allocator.free(validity);
            var log_total: f64 = 0.0;
            var count: usize = 0;
            for (values, 0..) |value, row| {
                if (maybe_validity) |validity| {
                    if (!validity[row]) continue;
                }
                const x = realValueToF64(value);
                if (x < 0.0) return std.math.nan(f64);
                if (x == 0.0) return 0.0;
                log_total += std.math.log(f64, std.math.e, x);
                count += 1;
            }
            if (count == 0) return error.EmptyArray;
            return std.math.exp(log_total / @as(f64, @floatFromInt(count)));
        }

        pub fn harmonicMean(self: Self) array_mod.ArrayError!f64 {
            if (comptime T == bool or isComplexColumnType(T)) return error.TypeUnsupported;
            const values = try self.values.toOwnedSlice(self.values.allocator);
            defer self.values.allocator.free(values);
            const maybe_validity = try validityValues(self, self.values.allocator);
            defer if (maybe_validity) |validity| self.values.allocator.free(validity);
            var reciprocal_total: f64 = 0.0;
            var count: usize = 0;
            for (values, 0..) |value, row| {
                if (maybe_validity) |validity| {
                    if (!validity[row]) continue;
                }
                const x = realValueToF64(value);
                if (x == 0.0) return 0.0;
                reciprocal_total += 1.0 / x;
                count += 1;
            }
            if (count == 0) return error.EmptyArray;
            return @as(f64, @floatFromInt(count)) / reciprocal_total;
        }

        pub fn mad(self: Self) array_mod.ArrayError!f64 {
            if (comptime T == bool or isComplexColumnType(T)) return error.TypeUnsupported;
            const center = try self.median();
            const values = try self.values.toOwnedSlice(self.values.allocator);
            defer self.values.allocator.free(values);
            const maybe_validity = try validityValues(self, self.values.allocator);
            defer if (maybe_validity) |validity| self.values.allocator.free(validity);
            const scratch = try self.values.allocator.alloc(f64, values.len);
            defer self.values.allocator.free(scratch);
            var count: usize = 0;
            for (values, 0..) |value, row| {
                if (maybe_validity) |validity| {
                    if (!validity[row]) continue;
                }
                scratch[count] = @abs(realValueToF64(value) - center);
                count += 1;
            }
            if (count == 0) return error.EmptyArray;
            std.sort.insertion(f64, scratch[0..count], {}, quantileLess);
            return quantileFromSorted(scratch[0..count], 0.5);
        }

        pub fn iqr(self: Self) array_mod.ArrayError!f64 {
            return (try self.quantile(0.75)) - (try self.quantile(0.25));
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
