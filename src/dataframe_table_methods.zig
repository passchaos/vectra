//! Eager DeviceDataFrame expression, row, and sort method wrappers.

const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const dataframe_view_mod = @import("dataframe_view.zig");
const expr_mod = @import("dataframe_expr.zig");
const keys_mod = @import("dataframe_keys.zig");
const rank_mod = @import("dataframe_rank.zig");
const options_mod = @import("dataframe_options.zig");
const series_mod = @import("series.zig");

const DeviceDataError = series_mod.DataError || array_mod.ArrayError;
const DeviceColumnBinaryOp = options_mod.DeviceColumnBinaryOp;
const DeviceColumnCompareOp = options_mod.DeviceColumnCompareOp;
const DeviceDTypeClass = options_mod.DeviceDTypeClass;
const DeviceScalar = options_mod.DeviceScalar;
const DeviceSortOptions = options_mod.DeviceSortOptions;

fn FrameType(comptime Frame: type) type {
    return switch (@typeInfo(Frame)) {
        .pointer => |ptr| ptr.child,
        else => Frame,
    };
}

fn frameValue(self: anytype) FrameType(@TypeOf(self)) {
    return switch (@typeInfo(@TypeOf(self))) {
        .pointer => self.*,
        else => self,
    };
}

pub fn binaryColumns(self: anytype, lhs_name: []const u8, rhs_name: []const u8, op: DeviceColumnBinaryOp) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.binaryColumns(frameValue(self), lhs_name, rhs_name, op);
}

pub fn addColumns(self: anytype, lhs_name: []const u8, rhs_name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return self.binaryColumns(lhs_name, rhs_name, .add);
}

pub fn subColumns(self: anytype, lhs_name: []const u8, rhs_name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return self.binaryColumns(lhs_name, rhs_name, .sub);
}

pub fn mulColumns(self: anytype, lhs_name: []const u8, rhs_name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return self.binaryColumns(lhs_name, rhs_name, .mul);
}

pub fn divColumns(self: anytype, lhs_name: []const u8, rhs_name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return self.binaryColumns(lhs_name, rhs_name, .div);
}

pub fn binaryColumnScalar(self: anytype, name: []const u8, comptime T: type, scalar: T, op: DeviceColumnBinaryOp) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.binaryColumnScalar(frameValue(self), name, T, scalar, op);
}

pub fn binaryColumnScalarWithDeviceScalar(self: anytype, name: []const u8, scalar: DeviceScalar, op: DeviceColumnBinaryOp) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.binaryColumnScalarWithDeviceScalar(frameValue(self), name, scalar, op);
}

pub fn compareColumns(self: anytype, lhs_name: []const u8, rhs_name: []const u8, op: DeviceColumnCompareOp) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.compareColumns(frameValue(self), lhs_name, rhs_name, op);
}

pub fn compareColumnScalar(self: anytype, name: []const u8, comptime T: type, scalar: T, op: DeviceColumnCompareOp) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.compareColumnScalar(frameValue(self), name, T, scalar, op);
}

pub fn compareColumnScalarWithDeviceScalar(self: anytype, name: []const u8, scalar: DeviceScalar, op: DeviceColumnCompareOp) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.compareColumnScalarWithDeviceScalar(frameValue(self), name, scalar, op);
}

pub fn filterColumnMask(self: anytype, mask: @TypeOf(frameValue(self).columns[0])) DeviceDataError!FrameType(@TypeOf(self)) {
    return expr_mod.filterColumnMask(FrameType(@TypeOf(self)), frameValue(self), mask);
}

pub fn filterColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return expr_mod.filterColumn(FrameType(@TypeOf(self)), frameValue(self), name);
}

pub fn view(self: anytype) DeviceDataError!dataframe_view_mod.DeviceDataFrameView {
    return dataframe_array_mod.view(dataframe_view_mod.DeviceDataFrameView, dataframe_view_mod.DeviceColumnView, frameValue(self));
}

pub fn select(self: anytype, wanted_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.select(FrameType(@TypeOf(self)), frameValue(self), wanted_names);
}

pub fn selectByDTypes(self: anytype, dtypes: []const array_mod.DType) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectByDTypes(FrameType(@TypeOf(self)), frameValue(self), dtypes);
}

pub fn selectByDTypeClass(self: anytype, class: DeviceDTypeClass) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectByDTypeClass(FrameType(@TypeOf(self)), frameValue(self), class);
}

pub fn selectNumeric(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return selectByDTypeClass(self, .numeric);
}

pub fn selectReal(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return selectByDTypeClass(self, .real);
}

pub fn selectFloat(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return selectByDTypeClass(self, .float);
}

pub fn selectInteger(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return selectByDTypeClass(self, .integer);
}

pub fn selectBool(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return selectByDTypeClass(self, .bool);
}

pub fn withColumn(self: anytype, name: []const u8, data: @TypeOf(frameValue(self).columns[0])) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), name, data);
}

pub fn castColumn(self: anytype, name: []const u8, dtype_value: array_mod.DType) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.castColumn(FrameType(@TypeOf(self)), frameValue(self), name, dtype_value);
}

pub fn fillNullColumn(self: anytype, name: []const u8, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.fillNullColumn(FrameType(@TypeOf(self)), frameValue(self), name, DeviceScalar.init(T, value));
}

pub fn fillNullColumnWithScalar(self: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.fillNullColumn(FrameType(@TypeOf(self)), frameValue(self), name, scalar);
}

pub fn withColumnLiteral(self: anytype, name: []const u8, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnLiteral(FrameType(@TypeOf(self)), frameValue(self), name, T, value);
}

pub fn withColumnLiteralScalar(self: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnLiteralScalar(FrameType(@TypeOf(self)), frameValue(self), name, scalar);
}

pub fn withRowIndex(self: anytype, name: []const u8, offset: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowIndex(FrameType(@TypeOf(self)), frameValue(self), name, offset);
}

pub fn renameColumn(self: anytype, old_name: []const u8, new_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.renameColumn(FrameType(@TypeOf(self)), frameValue(self), old_name, new_name);
}

pub fn dropColumns(self: anytype, names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropColumns(FrameType(@TypeOf(self)), frameValue(self), names);
}

pub fn dropColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropColumn(FrameType(@TypeOf(self)), frameValue(self), name);
}

pub fn dropNulls(self: anytype, names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropNulls(FrameType(@TypeOf(self)), frameValue(self), names);
}

pub fn dropNullsOn(self: anytype, names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dropNulls(self, names);
}

pub fn dropNullsColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dropNulls(self, &.{name});
}

pub fn head(self: anytype, n: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return sliceRows(self, 0, @min(n, self.rows));
}

pub fn tail(self: anytype, n: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    const count = @min(n, self.rows);
    return sliceRows(self, self.rows - count, self.rows);
}

pub fn sliceRows(self: anytype, start: usize, stop: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.sliceRows(FrameType(@TypeOf(self)), frameValue(self), start, stop);
}

pub fn take(self: anytype, row_indices: []const usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.takeRows(FrameType(@TypeOf(self)), frameValue(self), row_indices);
}

pub fn concatRows(self: anytype, other: FrameType(@TypeOf(self))) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.concatDeviceDataFramesRows(FrameType(@TypeOf(self)), frameValue(self), other);
}

pub fn appendRows(self: anytype, other: FrameType(@TypeOf(self))) DeviceDataError!FrameType(@TypeOf(self)) {
    return concatRows(self, other);
}

pub fn vstack(self: anytype, other: FrameType(@TypeOf(self))) DeviceDataError!FrameType(@TypeOf(self)) {
    return concatRows(self, other);
}

pub fn distinctRows(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return keys_mod.distinctRows(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn distinctOn(self: anytype, key_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return keys_mod.distinctOn(FrameType(@TypeOf(self)), frameValue(self), key_names);
}

pub fn dropDuplicates(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return distinctRows(self);
}

pub fn dropDuplicatesOn(self: anytype, key_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return distinctOn(self, key_names);
}

pub fn uniqueRows(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return distinctRows(self);
}

pub fn argsortBy(self: anytype, name: []const u8, options_value: DeviceSortOptions) DeviceDataError![]usize {
    return rank_mod.argsortBy(frameValue(self), name, options_value);
}

pub fn sortBy(self: anytype, name: []const u8, options_value: DeviceSortOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return rank_mod.sortBy(FrameType(@TypeOf(self)), frameValue(self), name, options_value);
}

pub fn sortByColumn(self: anytype, name: []const u8, options_value: DeviceSortOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return sortBy(self, name, options_value);
}

pub fn topKBy(self: anytype, name: []const u8, k: usize, options_value: DeviceSortOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return rank_mod.topKBy(FrameType(@TypeOf(self)), frameValue(self), name, k, options_value);
}

pub fn rankProfileBy(self: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceSortOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return rank_mod.rankProfileBy(FrameType(@TypeOf(self)), frameValue(self), name, output_prefix, options_value);
}

pub fn filter(self: anytype, mask: []const bool) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.filterRows(FrameType(@TypeOf(self)), frameValue(self), mask);
}

pub fn to(self: anytype, device_value: array_mod.Device) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.toDevice(FrameType(@TypeOf(self)), frameValue(self), device_value);
}

pub fn cpu(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return to(self, .cpu);
}

pub fn cuda(self: anytype, index: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return to(self, array_mod.Device.cuda(index));
}

pub fn mps(self: anytype, index: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return to(self, array_mod.Device.mps(index));
}
