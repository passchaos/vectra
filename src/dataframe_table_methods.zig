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

pub fn selectByColumnIndices(self: anytype, indices: []const usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectByColumnIndices(FrameType(@TypeOf(self)), frameValue(self), indices);
}

pub fn selectColumnRange(self: anytype, start: usize, stop: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectColumnRange(FrameType(@TypeOf(self)), frameValue(self), start, stop);
}

pub fn selectFirstColumns(self: anytype, n: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return selectColumnRange(self, 0, @min(n, frameValue(self).columns.len));
}

pub fn selectLastColumns(self: anytype, n: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    const count = @min(n, frameValue(self).columns.len);
    return selectColumnRange(self, frameValue(self).columns.len - count, frameValue(self).columns.len);
}

pub fn dropByColumnIndices(self: anytype, indices: []const usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropByColumnIndices(FrameType(@TypeOf(self)), frameValue(self), indices);
}

pub fn dropColumnRange(self: anytype, start: usize, stop: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropColumnRange(FrameType(@TypeOf(self)), frameValue(self), start, stop);
}

pub fn dropFirstColumns(self: anytype, n: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return dropColumnRange(self, 0, @min(n, frameValue(self).columns.len));
}

pub fn dropLastColumns(self: anytype, n: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    const count = @min(n, frameValue(self).columns.len);
    return dropColumnRange(self, frameValue(self).columns.len - count, frameValue(self).columns.len);
}

pub fn reverseColumns(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.reverseColumns(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn sortColumnsByName(self: anytype, descending: bool) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.sortColumnsByName(FrameType(@TypeOf(self)), frameValue(self), descending);
}

pub fn selectByNamePrefix(self: anytype, prefix: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectByNamePrefix(FrameType(@TypeOf(self)), frameValue(self), prefix);
}

pub fn selectByNameSuffix(self: anytype, suffix: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectByNameSuffix(FrameType(@TypeOf(self)), frameValue(self), suffix);
}

pub fn selectByNameContains(self: anytype, needle: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectByNameContains(FrameType(@TypeOf(self)), frameValue(self), needle);
}

pub fn dropByNamePrefix(self: anytype, prefix: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropByNamePrefix(FrameType(@TypeOf(self)), frameValue(self), prefix);
}

pub fn dropByNameSuffix(self: anytype, suffix: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropByNameSuffix(FrameType(@TypeOf(self)), frameValue(self), suffix);
}

pub fn dropByNameContains(self: anytype, needle: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropByNameContains(FrameType(@TypeOf(self)), frameValue(self), needle);
}

pub fn selectByDTypes(self: anytype, dtypes: []const array_mod.DType) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectByDTypes(FrameType(@TypeOf(self)), frameValue(self), dtypes);
}

pub fn selectByDTypeClass(self: anytype, class: DeviceDTypeClass) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectByDTypeClass(FrameType(@TypeOf(self)), frameValue(self), class);
}

pub fn dropByDTypes(self: anytype, dtypes: []const array_mod.DType) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropByDTypes(FrameType(@TypeOf(self)), frameValue(self), dtypes);
}

pub fn dropByDTypeClass(self: anytype, class: DeviceDTypeClass) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropByDTypeClass(FrameType(@TypeOf(self)), frameValue(self), class);
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

pub fn dropNumeric(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dropByDTypeClass(self, .numeric);
}

pub fn dropReal(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dropByDTypeClass(self, .real);
}

pub fn dropFloat(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dropByDTypeClass(self, .float);
}

pub fn dropInteger(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dropByDTypeClass(self, .integer);
}

pub fn dropBool(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dropByDTypeClass(self, .bool);
}

pub fn selectNullableColumns(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectNullableColumns(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn selectNonNullableColumns(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectNonNullableColumns(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn selectColumnsWithNulls(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectColumnsWithNulls(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn selectColumnsWithoutNulls(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectColumnsWithoutNulls(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn dropNullableColumns(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropNullableColumns(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn dropNonNullableColumns(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropNonNullableColumns(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn dropColumnsWithNulls(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropColumnsWithNulls(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn dropColumnsWithoutNulls(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropColumnsWithoutNulls(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn withColumn(self: anytype, name: []const u8, data: @TypeOf(frameValue(self).columns[0])) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), name, data);
}

pub fn withColumnAt(self: anytype, name: []const u8, data: @TypeOf(frameValue(self).columns[0]), target_index: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnAt(FrameType(@TypeOf(self)), frameValue(self), name, data, target_index);
}

pub fn withColumnBefore(self: anytype, name: []const u8, data: @TypeOf(frameValue(self).columns[0]), before_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnBefore(FrameType(@TypeOf(self)), frameValue(self), name, data, before_name);
}

pub fn withColumnAfter(self: anytype, name: []const u8, data: @TypeOf(frameValue(self).columns[0]), after_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnAfter(FrameType(@TypeOf(self)), frameValue(self), name, data, after_name);
}

pub fn copyColumn(self: anytype, source_name: []const u8, new_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.copyColumn(FrameType(@TypeOf(self)), frameValue(self), source_name, new_name);
}

pub fn copyColumnAt(self: anytype, source_name: []const u8, new_name: []const u8, target_index: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.copyColumnAt(FrameType(@TypeOf(self)), frameValue(self), source_name, new_name, target_index);
}

pub fn copyColumnBefore(self: anytype, source_name: []const u8, new_name: []const u8, before_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.copyColumnBefore(FrameType(@TypeOf(self)), frameValue(self), source_name, new_name, before_name);
}

pub fn copyColumnAfter(self: anytype, source_name: []const u8, new_name: []const u8, after_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.copyColumnAfter(FrameType(@TypeOf(self)), frameValue(self), source_name, new_name, after_name);
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

pub fn fillNaNColumn(self: anytype, name: []const u8, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.fillNaNColumn(FrameType(@TypeOf(self)), frameValue(self), name, DeviceScalar.init(T, value));
}

pub fn fillNaNColumnWithScalar(self: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.fillNaNColumn(FrameType(@TypeOf(self)), frameValue(self), name, scalar);
}

pub fn coalesceColumns(self: anytype, primary_name: []const u8, fallback_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.coalesceColumns(FrameType(@TypeOf(self)), frameValue(self), primary_name, fallback_name, output_name);
}

pub fn isNullColumn(self: anytype, name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.isNullColumn(FrameType(@TypeOf(self)), frameValue(self), name, output_name);
}

pub fn isValidColumn(self: anytype, name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.isValidColumn(FrameType(@TypeOf(self)), frameValue(self), name, output_name);
}

pub fn isNanColumn(self: anytype, name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.isNanColumn(FrameType(@TypeOf(self)), frameValue(self), name, output_name);
}

pub fn isFiniteColumn(self: anytype, name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.isFiniteColumn(FrameType(@TypeOf(self)), frameValue(self), name, output_name);
}

pub fn withRowNullCount(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowNullCount(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowValidCount(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowValidCount(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withColumnLiteral(self: anytype, name: []const u8, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnLiteral(FrameType(@TypeOf(self)), frameValue(self), name, T, value);
}

pub fn withColumnLiteralScalar(self: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnLiteralScalar(FrameType(@TypeOf(self)), frameValue(self), name, scalar);
}

pub fn withColumnLiteralAt(self: anytype, name: []const u8, comptime T: type, value: T, target_index: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnLiteralAt(FrameType(@TypeOf(self)), frameValue(self), name, T, value, target_index);
}

pub fn withColumnLiteralBefore(self: anytype, name: []const u8, comptime T: type, value: T, before_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnLiteralBefore(FrameType(@TypeOf(self)), frameValue(self), name, T, value, before_name);
}

pub fn withColumnLiteralAfter(self: anytype, name: []const u8, comptime T: type, value: T, after_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnLiteralAfter(FrameType(@TypeOf(self)), frameValue(self), name, T, value, after_name);
}

pub fn withColumnLiteralScalarAt(self: anytype, name: []const u8, scalar: DeviceScalar, target_index: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnLiteralScalarAt(FrameType(@TypeOf(self)), frameValue(self), name, scalar, target_index);
}

pub fn withColumnLiteralScalarBefore(self: anytype, name: []const u8, scalar: DeviceScalar, before_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnLiteralScalarBefore(FrameType(@TypeOf(self)), frameValue(self), name, scalar, before_name);
}

pub fn withColumnLiteralScalarAfter(self: anytype, name: []const u8, scalar: DeviceScalar, after_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnLiteralScalarAfter(FrameType(@TypeOf(self)), frameValue(self), name, scalar, after_name);
}

pub fn withRowIndex(self: anytype, name: []const u8, offset: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowIndex(FrameType(@TypeOf(self)), frameValue(self), name, offset);
}

pub fn renameColumn(self: anytype, old_name: []const u8, new_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.renameColumn(FrameType(@TypeOf(self)), frameValue(self), old_name, new_name);
}

pub fn renameColumns(self: anytype, old_names: []const []const u8, new_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.renameColumns(FrameType(@TypeOf(self)), frameValue(self), old_names, new_names);
}

pub fn addColumnNamePrefix(self: anytype, prefix: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.addColumnNamePrefix(FrameType(@TypeOf(self)), frameValue(self), prefix);
}

pub fn addColumnNameSuffix(self: anytype, suffix: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.addColumnNameSuffix(FrameType(@TypeOf(self)), frameValue(self), suffix);
}

pub fn moveColumn(self: anytype, name: []const u8, target_index: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.moveColumn(FrameType(@TypeOf(self)), frameValue(self), name, target_index);
}

pub fn moveColumnBefore(self: anytype, name: []const u8, before_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.moveColumnBefore(FrameType(@TypeOf(self)), frameValue(self), name, before_name);
}

pub fn moveColumnAfter(self: anytype, name: []const u8, after_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.moveColumnAfter(FrameType(@TypeOf(self)), frameValue(self), name, after_name);
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

pub fn filterNullsColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.filterNullsColumn(FrameType(@TypeOf(self)), frameValue(self), name);
}

pub fn dropNaNs(self: anytype, names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropNaNs(FrameType(@TypeOf(self)), frameValue(self), names);
}

pub fn dropNaNsOn(self: anytype, names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dropNaNs(self, names);
}

pub fn dropNaNsColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dropNaNs(self, &.{name});
}

pub fn filterNaNsColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.filterNaNsColumn(FrameType(@TypeOf(self)), frameValue(self), name);
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

pub fn dropRows(self: anytype, row_indices: []const usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropRows(FrameType(@TypeOf(self)), frameValue(self), row_indices);
}

pub fn dropRowRange(self: anytype, start: usize, stop: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropRowRange(FrameType(@TypeOf(self)), frameValue(self), start, stop);
}

pub fn dropFirstRows(self: anytype, n: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return dropRowRange(self, 0, @min(n, frameValue(self).rows));
}

pub fn dropLastRows(self: anytype, n: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    const count = @min(n, frameValue(self).rows);
    return dropRowRange(self, frameValue(self).rows - count, frameValue(self).rows);
}

pub fn sliceRowsStep(self: anytype, start: usize, stop: usize, step: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.sliceRowsStep(FrameType(@TypeOf(self)), frameValue(self), start, stop, step);
}

pub fn sliceStep(self: anytype, start: usize, len: usize, step: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    const stop = std.math.add(usize, start, len) catch return error.InvalidShape;
    return sliceRowsStep(self, start, stop, step);
}

pub fn take(self: anytype, row_indices: []const usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.takeRows(FrameType(@TypeOf(self)), frameValue(self), row_indices);
}

pub fn sampleRows(self: anytype, count: usize, seed: u64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.sampleRows(FrameType(@TypeOf(self)), frameValue(self), count, seed);
}

pub fn sampleRowsWithReplacement(self: anytype, count: usize, seed: u64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.sampleRowsWithReplacement(FrameType(@TypeOf(self)), frameValue(self), count, seed);
}

pub fn strideRows(self: anytype, start: usize, step: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.strideRows(FrameType(@TypeOf(self)), frameValue(self), start, step);
}

pub fn reverseRows(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.reverseRows(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn reverse(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return reverseRows(self);
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
