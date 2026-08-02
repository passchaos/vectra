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
const validity_mod = @import("dataframe_validity_core.zig");

const DeviceDataError = series_mod.DataError || array_mod.ArrayError;
const DeviceColumnBinaryOp = options_mod.DeviceColumnBinaryOp;
const DeviceColumnCompareOp = options_mod.DeviceColumnCompareOp;
const DeviceColumnLogicalOp = options_mod.DeviceColumnLogicalOp;
const DeviceDTypeClass = options_mod.DeviceDTypeClass;
const DeviceScalar = options_mod.DeviceScalar;
const DeviceSortOptions = options_mod.DeviceSortOptions;

fn FrameType(comptime Frame: type) type {
    @setEvalBranchQuota(4000);
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

pub fn unaryColumnAbs(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnAbs(frameValue(self), name);
}

pub fn withColumnAbs(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnAbs(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnNeg(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnNeg(frameValue(self), name);
}

pub fn withColumnNeg(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnNeg(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnNegative(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return withColumnNeg(self, output_name, input_name);
}

pub fn unaryColumnSquare(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnSquare(frameValue(self), name);
}

pub fn withColumnSquare(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnSquare(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnReciprocal(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnReciprocal(frameValue(self), name);
}

pub fn withColumnReciprocal(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnReciprocal(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnSign(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnSign(frameValue(self), name);
}

pub fn withColumnSign(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnSign(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnSqrt(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnSqrt(frameValue(self), name);
}

pub fn withColumnSqrt(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnSqrt(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnRsqrt(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnRsqrt(frameValue(self), name);
}

pub fn withColumnRsqrt(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnRsqrt(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnCbrt(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnCbrt(frameValue(self), name);
}

pub fn withColumnCbrt(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnCbrt(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnFloor(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnFloor(frameValue(self), name);
}

pub fn withColumnFloor(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnFloor(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnCeil(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnCeil(frameValue(self), name);
}

pub fn withColumnCeil(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnCeil(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnRound(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnRound(frameValue(self), name);
}

pub fn withColumnRound(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnRound(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnTrunc(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnTrunc(frameValue(self), name);
}

pub fn withColumnTrunc(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnTrunc(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnDeg2rad(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnDeg2rad(frameValue(self), name);
}

pub fn withColumnDeg2rad(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnDeg2rad(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnRad2deg(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnRad2deg(frameValue(self), name);
}

pub fn withColumnRad2deg(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnRad2deg(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnExpit(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnExpit(frameValue(self), name);
}

pub fn withColumnExpit(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnExpit(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnLogit(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnLogit(frameValue(self), name);
}

pub fn withColumnLogit(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnLogit(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnSoftplus(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnSoftplus(frameValue(self), name);
}

pub fn withColumnSoftplus(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnSoftplus(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnLogsigmoid(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnLogsigmoid(frameValue(self), name);
}

pub fn withColumnLogsigmoid(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnLogsigmoid(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnRelu(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnRelu(frameValue(self), name);
}

pub fn withColumnRelu(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnRelu(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnLeakyRelu(self: anytype, name: []const u8, comptime T: type, negative_slope: T) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnLeakyRelu(frameValue(self), name, T, negative_slope);
}

pub fn unaryColumnLeakyReluWithDeviceScalar(self: anytype, name: []const u8, negative_slope: DeviceScalar) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnLeakyReluWithDeviceScalar(frameValue(self), name, negative_slope);
}

pub fn withColumnLeakyRelu(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, negative_slope: T) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnLeakyRelu(self, input_name, T, negative_slope);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnLeakyReluWithDeviceScalar(self: anytype, output_name: []const u8, input_name: []const u8, negative_slope: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnLeakyReluWithDeviceScalar(self, input_name, negative_slope);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnRelu6(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnRelu6(frameValue(self), name);
}

pub fn withColumnRelu6(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnRelu6(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnPowScalar(self: anytype, name: []const u8, comptime T: type, exponent: T) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnPowScalar(frameValue(self), name, T, exponent);
}

pub fn unaryColumnPowWithDeviceScalar(self: anytype, name: []const u8, exponent: DeviceScalar) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnPowWithDeviceScalar(frameValue(self), name, exponent);
}

pub fn withColumnPowScalar(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, exponent: T) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnPowScalar(self, input_name, T, exponent);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnPowWithDeviceScalar(self: anytype, output_name: []const u8, input_name: []const u8, exponent: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnPowWithDeviceScalar(self, input_name, exponent);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnFloorDivScalar(self: anytype, name: []const u8, comptime T: type, scalar: T) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnFloorDivScalar(frameValue(self), name, T, scalar);
}

pub fn unaryColumnFloorDivWithDeviceScalar(self: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnFloorDivWithDeviceScalar(frameValue(self), name, scalar);
}

pub fn withColumnFloorDivScalar(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, scalar: T) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnFloorDivScalar(self, input_name, T, scalar);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnFloorDivWithDeviceScalar(self: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnFloorDivWithDeviceScalar(self, input_name, scalar);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnModScalar(self: anytype, name: []const u8, comptime T: type, scalar: T) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnModScalar(frameValue(self), name, T, scalar);
}

pub fn unaryColumnModWithDeviceScalar(self: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnModWithDeviceScalar(frameValue(self), name, scalar);
}

pub fn withColumnModScalar(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, scalar: T) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnModScalar(self, input_name, T, scalar);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnModWithDeviceScalar(self: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnModWithDeviceScalar(self, input_name, scalar);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnRemainderScalar(self: anytype, name: []const u8, comptime T: type, scalar: T) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnRemainderScalar(frameValue(self), name, T, scalar);
}

pub fn unaryColumnRemainderWithDeviceScalar(self: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnRemainderWithDeviceScalar(frameValue(self), name, scalar);
}

pub fn withColumnRemainderScalar(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, scalar: T) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnRemainderScalar(self, input_name, T, scalar);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnRemainderWithDeviceScalar(self: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnRemainderWithDeviceScalar(self, input_name, scalar);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnLogAddExpScalar(self: anytype, name: []const u8, comptime T: type, scalar: T) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnLogAddExpScalar(frameValue(self), name, T, scalar);
}

pub fn unaryColumnLogAddExpWithDeviceScalar(self: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnLogAddExpWithDeviceScalar(frameValue(self), name, scalar);
}

pub fn withColumnLogAddExpScalar(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, scalar: T) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnLogAddExpScalar(self, input_name, T, scalar);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnLogAddExpWithDeviceScalar(self: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnLogAddExpWithDeviceScalar(self, input_name, scalar);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnLogAddExp2Scalar(self: anytype, name: []const u8, comptime T: type, scalar: T) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnLogAddExp2Scalar(frameValue(self), name, T, scalar);
}

pub fn unaryColumnLogAddExp2WithDeviceScalar(self: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnLogAddExp2WithDeviceScalar(frameValue(self), name, scalar);
}

pub fn withColumnLogAddExp2Scalar(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, scalar: T) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnLogAddExp2Scalar(self, input_name, T, scalar);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnLogAddExp2WithDeviceScalar(self: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnLogAddExp2WithDeviceScalar(self, input_name, scalar);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnXlogyScalar(self: anytype, name: []const u8, comptime T: type, scalar: T) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnXlogyScalar(frameValue(self), name, T, scalar);
}

pub fn unaryColumnXlogyWithDeviceScalar(self: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnXlogyWithDeviceScalar(frameValue(self), name, scalar);
}

pub fn withColumnXlogyScalar(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, scalar: T) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnXlogyScalar(self, input_name, T, scalar);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnXlogyWithDeviceScalar(self: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnXlogyWithDeviceScalar(self, input_name, scalar);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnFmaxScalar(self: anytype, name: []const u8, comptime T: type, scalar: T) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnFmaxScalar(frameValue(self), name, T, scalar);
}

pub fn unaryColumnFmaxWithDeviceScalar(self: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnFmaxWithDeviceScalar(frameValue(self), name, scalar);
}

pub fn withColumnFmaxScalar(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, scalar: T) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnFmaxScalar(self, input_name, T, scalar);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnFmaxWithDeviceScalar(self: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnFmaxWithDeviceScalar(self, input_name, scalar);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnFminScalar(self: anytype, name: []const u8, comptime T: type, scalar: T) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnFminScalar(frameValue(self), name, T, scalar);
}

pub fn unaryColumnFminWithDeviceScalar(self: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnFminWithDeviceScalar(frameValue(self), name, scalar);
}

pub fn withColumnFminScalar(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, scalar: T) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnFminScalar(self, input_name, T, scalar);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnFminWithDeviceScalar(self: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnFminWithDeviceScalar(self, input_name, scalar);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnHypotScalar(self: anytype, name: []const u8, comptime T: type, scalar: T) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnHypotScalar(frameValue(self), name, T, scalar);
}

pub fn unaryColumnHypotWithDeviceScalar(self: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnHypotWithDeviceScalar(frameValue(self), name, scalar);
}

pub fn withColumnHypotScalar(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, scalar: T) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnHypotScalar(self, input_name, T, scalar);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnHypotWithDeviceScalar(self: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnHypotWithDeviceScalar(self, input_name, scalar);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnAtan2Scalar(self: anytype, name: []const u8, comptime T: type, scalar: T) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnAtan2Scalar(frameValue(self), name, T, scalar);
}

pub fn unaryColumnAtan2WithDeviceScalar(self: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnAtan2WithDeviceScalar(frameValue(self), name, scalar);
}

pub fn withColumnAtan2Scalar(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, scalar: T) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnAtan2Scalar(self, input_name, T, scalar);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnAtan2WithDeviceScalar(self: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnAtan2WithDeviceScalar(self, input_name, scalar);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnNextAfterScalar(self: anytype, name: []const u8, comptime T: type, scalar: T) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnNextAfterScalar(frameValue(self), name, T, scalar);
}

pub fn unaryColumnNextAfterWithDeviceScalar(self: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnNextAfterWithDeviceScalar(frameValue(self), name, scalar);
}

pub fn withColumnNextAfterScalar(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, scalar: T) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnNextAfterScalar(self, input_name, T, scalar);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnNextAfterWithDeviceScalar(self: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnNextAfterWithDeviceScalar(self, input_name, scalar);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnCopysignScalar(self: anytype, name: []const u8, comptime T: type, scalar: T) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnCopysignScalar(frameValue(self), name, T, scalar);
}

pub fn unaryColumnCopysignWithDeviceScalar(self: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnCopysignWithDeviceScalar(frameValue(self), name, scalar);
}

pub fn withColumnCopysignScalar(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, scalar: T) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnCopysignScalar(self, input_name, T, scalar);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnCopysignWithDeviceScalar(self: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnCopysignWithDeviceScalar(self, input_name, scalar);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnHeavisideScalar(self: anytype, name: []const u8, comptime T: type, value_at_zero: T) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnHeavisideScalar(frameValue(self), name, T, value_at_zero);
}

pub fn unaryColumnHeavisideWithDeviceScalar(self: anytype, name: []const u8, value_at_zero: DeviceScalar) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnHeavisideWithDeviceScalar(frameValue(self), name, value_at_zero);
}

pub fn withColumnHeavisideScalar(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, value_at_zero: T) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnHeavisideScalar(self, input_name, T, value_at_zero);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnHeavisideWithDeviceScalar(self: anytype, output_name: []const u8, input_name: []const u8, value_at_zero: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnHeavisideWithDeviceScalar(self, input_name, value_at_zero);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnLdexpScalar(self: anytype, name: []const u8, exponent: i32) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnLdexpScalar(frameValue(self), name, exponent);
}

pub fn withColumnLdexpScalar(self: anytype, output_name: []const u8, input_name: []const u8, exponent: i32) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnLdexpScalar(self, input_name, exponent);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnThreshold(
    self: anytype,
    name: []const u8,
    comptime T: type,
    threshold_value: T,
    replacement_value: T,
) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnThreshold(frameValue(self), name, T, threshold_value, replacement_value);
}

pub fn unaryColumnThresholdWithDeviceScalars(
    self: anytype,
    name: []const u8,
    threshold_value: DeviceScalar,
    replacement_value: DeviceScalar,
) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnThresholdWithDeviceScalars(frameValue(self), name, threshold_value, replacement_value);
}

pub fn withColumnThreshold(
    self: anytype,
    output_name: []const u8,
    input_name: []const u8,
    comptime T: type,
    threshold_value: T,
    replacement_value: T,
) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnThreshold(self, input_name, T, threshold_value, replacement_value);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnThresholdWithDeviceScalars(
    self: anytype,
    output_name: []const u8,
    input_name: []const u8,
    threshold_value: DeviceScalar,
    replacement_value: DeviceScalar,
) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnThresholdWithDeviceScalars(self, input_name, threshold_value, replacement_value);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnHardtanh(
    self: anytype,
    name: []const u8,
    comptime T: type,
    min_value: T,
    max_value: T,
) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnHardtanh(frameValue(self), name, T, min_value, max_value);
}

pub fn unaryColumnHardtanhWithDeviceScalars(
    self: anytype,
    name: []const u8,
    min_value: DeviceScalar,
    max_value: DeviceScalar,
) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnHardtanhWithDeviceScalars(frameValue(self), name, min_value, max_value);
}

pub fn withColumnHardtanh(
    self: anytype,
    output_name: []const u8,
    input_name: []const u8,
    comptime T: type,
    min_value: T,
    max_value: T,
) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnHardtanh(self, input_name, T, min_value, max_value);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnHardtanhWithDeviceScalars(
    self: anytype,
    output_name: []const u8,
    input_name: []const u8,
    min_value: DeviceScalar,
    max_value: DeviceScalar,
) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnHardtanhWithDeviceScalars(self, input_name, min_value, max_value);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnMaximumScalar(self: anytype, name: []const u8, comptime T: type, scalar: T) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnMaximumScalar(frameValue(self), name, T, scalar);
}

pub fn unaryColumnMaximumWithDeviceScalar(self: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnMaximumWithDeviceScalar(frameValue(self), name, scalar);
}

pub fn withColumnMaximumScalar(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, scalar: T) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnMaximumScalar(self, input_name, T, scalar);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnMaximumWithDeviceScalar(self: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnMaximumWithDeviceScalar(self, input_name, scalar);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnMinimumScalar(self: anytype, name: []const u8, comptime T: type, scalar: T) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnMinimumScalar(frameValue(self), name, T, scalar);
}

pub fn unaryColumnMinimumWithDeviceScalar(self: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnMinimumWithDeviceScalar(frameValue(self), name, scalar);
}

pub fn withColumnMinimumScalar(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, scalar: T) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnMinimumScalar(self, input_name, T, scalar);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnMinimumWithDeviceScalar(self: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnMinimumWithDeviceScalar(self, input_name, scalar);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnClipMin(self: anytype, name: []const u8, comptime T: type, min_value: T) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnClipMin(frameValue(self), name, T, min_value);
}

pub fn unaryColumnClipMinWithDeviceScalar(self: anytype, name: []const u8, min_value: DeviceScalar) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnClipMinWithDeviceScalar(frameValue(self), name, min_value);
}

pub fn withColumnClipMin(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, min_value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnClipMin(self, input_name, T, min_value);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnClipMinWithDeviceScalar(self: anytype, output_name: []const u8, input_name: []const u8, min_value: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnClipMinWithDeviceScalar(self, input_name, min_value);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnClipMax(self: anytype, name: []const u8, comptime T: type, max_value: T) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnClipMax(frameValue(self), name, T, max_value);
}

pub fn unaryColumnClipMaxWithDeviceScalar(self: anytype, name: []const u8, max_value: DeviceScalar) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnClipMaxWithDeviceScalar(frameValue(self), name, max_value);
}

pub fn withColumnClipMax(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, max_value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnClipMax(self, input_name, T, max_value);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnClipMaxWithDeviceScalar(self: anytype, output_name: []const u8, input_name: []const u8, max_value: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnClipMaxWithDeviceScalar(self, input_name, max_value);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnHardshrink(self: anytype, name: []const u8, comptime T: type, lambd: T) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnHardshrink(frameValue(self), name, T, lambd);
}

pub fn unaryColumnHardshrinkWithDeviceScalar(self: anytype, name: []const u8, lambd: DeviceScalar) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnHardshrinkWithDeviceScalar(frameValue(self), name, lambd);
}

pub fn withColumnHardshrink(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, lambd: T) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnHardshrink(self, input_name, T, lambd);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnHardshrinkWithDeviceScalar(self: anytype, output_name: []const u8, input_name: []const u8, lambd: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnHardshrinkWithDeviceScalar(self, input_name, lambd);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnSoftshrink(self: anytype, name: []const u8, comptime T: type, lambd: T) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnSoftshrink(frameValue(self), name, T, lambd);
}

pub fn unaryColumnSoftshrinkWithDeviceScalar(self: anytype, name: []const u8, lambd: DeviceScalar) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnSoftshrinkWithDeviceScalar(frameValue(self), name, lambd);
}

pub fn withColumnSoftshrink(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, lambd: T) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnSoftshrink(self, input_name, T, lambd);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnSoftshrinkWithDeviceScalar(self: anytype, output_name: []const u8, input_name: []const u8, lambd: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnSoftshrinkWithDeviceScalar(self, input_name, lambd);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnTanhshrink(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnTanhshrink(frameValue(self), name);
}

pub fn withColumnTanhshrink(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnTanhshrink(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnElu(self: anytype, name: []const u8, comptime T: type, alpha: T) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnElu(frameValue(self), name, T, alpha);
}

pub fn unaryColumnEluWithDeviceScalar(self: anytype, name: []const u8, alpha: DeviceScalar) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnEluWithDeviceScalar(frameValue(self), name, alpha);
}

pub fn withColumnElu(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, alpha: T) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnElu(self, input_name, T, alpha);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnEluWithDeviceScalar(self: anytype, output_name: []const u8, input_name: []const u8, alpha: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnEluWithDeviceScalar(self, input_name, alpha);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnCelu(self: anytype, name: []const u8, comptime T: type, alpha: T) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnCelu(frameValue(self), name, T, alpha);
}

pub fn unaryColumnCeluWithDeviceScalar(self: anytype, name: []const u8, alpha: DeviceScalar) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnCeluWithDeviceScalar(frameValue(self), name, alpha);
}

pub fn withColumnCelu(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, alpha: T) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnCelu(self, input_name, T, alpha);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnCeluWithDeviceScalar(self: anytype, output_name: []const u8, input_name: []const u8, alpha: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnCeluWithDeviceScalar(self, input_name, alpha);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnSoftsign(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnSoftsign(frameValue(self), name);
}

pub fn withColumnSoftsign(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnSoftsign(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnHardsigmoid(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnHardsigmoid(frameValue(self), name);
}

pub fn withColumnHardsigmoid(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnHardsigmoid(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnHardswish(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnHardswish(frameValue(self), name);
}

pub fn withColumnHardswish(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnHardswish(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnSilu(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnSilu(frameValue(self), name);
}

pub fn withColumnSilu(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnSilu(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnSwish(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnSwish(frameValue(self), name);
}

pub fn withColumnSwish(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnSwish(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnMish(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnMish(frameValue(self), name);
}

pub fn withColumnMish(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnMish(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnGelu(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnGelu(frameValue(self), name);
}

pub fn withColumnGelu(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnGelu(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnSelu(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnSelu(frameValue(self), name);
}

pub fn withColumnSelu(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnSelu(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnExp(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnExp(frameValue(self), name);
}

pub fn withColumnExp(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnExp(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnExp2(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnExp2(frameValue(self), name);
}

pub fn withColumnExp2(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnExp2(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnExpm1(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnExpm1(frameValue(self), name);
}

pub fn withColumnExpm1(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnExpm1(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnSin(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnSin(frameValue(self), name);
}

pub fn withColumnSin(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnSin(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnCos(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnCos(frameValue(self), name);
}

pub fn withColumnCos(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnCos(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnTan(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnTan(frameValue(self), name);
}

pub fn withColumnTan(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnTan(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnAsin(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnAsin(frameValue(self), name);
}

pub fn withColumnAsin(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnAsin(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnAcos(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnAcos(frameValue(self), name);
}

pub fn withColumnAcos(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnAcos(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnAtan(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnAtan(frameValue(self), name);
}

pub fn withColumnAtan(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnAtan(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnSinh(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnSinh(frameValue(self), name);
}

pub fn withColumnSinh(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnSinh(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnCosh(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnCosh(frameValue(self), name);
}

pub fn withColumnCosh(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnCosh(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnTanh(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnTanh(frameValue(self), name);
}

pub fn withColumnTanh(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnTanh(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnAsinh(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnAsinh(frameValue(self), name);
}

pub fn withColumnAsinh(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnAsinh(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnAcosh(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnAcosh(frameValue(self), name);
}

pub fn withColumnAcosh(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnAcosh(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnAtanh(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnAtanh(frameValue(self), name);
}

pub fn withColumnAtanh(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnAtanh(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnLog(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnLog(frameValue(self), name);
}

pub fn withColumnLog(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnLog(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnLog1p(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnLog1p(frameValue(self), name);
}

pub fn withColumnLog1p(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnLog1p(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnLgamma(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnLgamma(frameValue(self), name);
}

pub fn withColumnLgamma(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnLgamma(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnSinc(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnSinc(frameValue(self), name);
}

pub fn withColumnSinc(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnSinc(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnLog2(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnLog2(frameValue(self), name);
}

pub fn withColumnLog2(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnLog2(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn unaryColumnLog10(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.unaryColumnLog10(frameValue(self), name);
}

pub fn withColumnLog10(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try unaryColumnLog10(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
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

pub fn lerpColumnsScalar(self: anytype, lhs_name: []const u8, rhs_name: []const u8, comptime T: type, weight: T) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.lerpColumnsScalar(frameValue(self), lhs_name, rhs_name, T, weight);
}

pub fn lerpColumnsWithDeviceScalar(self: anytype, lhs_name: []const u8, rhs_name: []const u8, weight: DeviceScalar) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.lerpColumnsWithDeviceScalar(frameValue(self), lhs_name, rhs_name, weight);
}

pub fn withColumnLerpScalar(self: anytype, output_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, comptime T: type, weight: T) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try lerpColumnsScalar(self, lhs_name, rhs_name, T, weight);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnLerpWithDeviceScalar(self: anytype, output_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try lerpColumnsWithDeviceScalar(self, lhs_name, rhs_name, weight);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn addcmulColumnsScalar(self: anytype, base_name: []const u8, input1_name: []const u8, input2_name: []const u8, comptime T: type, value: T) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.addcmulColumnsScalar(frameValue(self), base_name, input1_name, input2_name, T, value);
}

pub fn addcmulColumnsWithDeviceScalar(self: anytype, base_name: []const u8, input1_name: []const u8, input2_name: []const u8, value: DeviceScalar) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.addcmulColumnsWithDeviceScalar(frameValue(self), base_name, input1_name, input2_name, value);
}

pub fn withColumnAddcmulScalar(self: anytype, output_name: []const u8, base_name: []const u8, input1_name: []const u8, input2_name: []const u8, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try addcmulColumnsScalar(self, base_name, input1_name, input2_name, T, value);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnAddcmulWithDeviceScalar(self: anytype, output_name: []const u8, base_name: []const u8, input1_name: []const u8, input2_name: []const u8, value: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try addcmulColumnsWithDeviceScalar(self, base_name, input1_name, input2_name, value);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn addcdivColumnsScalar(self: anytype, base_name: []const u8, input1_name: []const u8, input2_name: []const u8, comptime T: type, value: T) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.addcdivColumnsScalar(frameValue(self), base_name, input1_name, input2_name, T, value);
}

pub fn addcdivColumnsWithDeviceScalar(self: anytype, base_name: []const u8, input1_name: []const u8, input2_name: []const u8, value: DeviceScalar) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.addcdivColumnsWithDeviceScalar(frameValue(self), base_name, input1_name, input2_name, value);
}

pub fn withColumnAddcdivScalar(self: anytype, output_name: []const u8, base_name: []const u8, input1_name: []const u8, input2_name: []const u8, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try addcdivColumnsScalar(self, base_name, input1_name, input2_name, T, value);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnAddcdivWithDeviceScalar(self: anytype, output_name: []const u8, base_name: []const u8, input1_name: []const u8, input2_name: []const u8, value: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try addcdivColumnsWithDeviceScalar(self, base_name, input1_name, input2_name, value);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn clipArrayColumns(self: anytype, input_name: []const u8, min_name: []const u8, max_name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.clipArrayColumns(frameValue(self), input_name, min_name, max_name);
}

pub fn withColumnClipArray(self: anytype, output_name: []const u8, input_name: []const u8, min_name: []const u8, max_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try clipArrayColumns(self, input_name, min_name, max_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn whereColumnScalar(self: anytype, input_name: []const u8, mask_name: []const u8, comptime T: type, other_value: T) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.whereColumnScalar(frameValue(self), input_name, mask_name, T, other_value);
}

pub fn whereColumnWithDeviceScalar(self: anytype, input_name: []const u8, mask_name: []const u8, other_value: DeviceScalar) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.whereColumnWithDeviceScalar(frameValue(self), input_name, mask_name, other_value);
}

pub fn withColumnWhereScalar(self: anytype, output_name: []const u8, input_name: []const u8, mask_name: []const u8, comptime T: type, other_value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try whereColumnScalar(self, input_name, mask_name, T, other_value);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnWhereWithDeviceScalar(self: anytype, output_name: []const u8, input_name: []const u8, mask_name: []const u8, other_value: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try whereColumnWithDeviceScalar(self, input_name, mask_name, other_value);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn whereColumns(self: anytype, input_name: []const u8, mask_name: []const u8, other_name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.whereColumns(frameValue(self), input_name, mask_name, other_name);
}

pub fn withColumnWhere(self: anytype, output_name: []const u8, input_name: []const u8, mask_name: []const u8, other_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try whereColumns(self, input_name, mask_name, other_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn isinColumns(self: anytype, input_name: []const u8, test_name: []const u8, invert: bool) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.isinColumns(frameValue(self), input_name, test_name, invert);
}

pub fn isinColumnValuesWithDeviceColumn(self: anytype, input_name: []const u8, test_values: @TypeOf(frameValue(self).columns[0]), invert: bool) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    const input = try frameValue(self).column(input_name);
    return input.isinColumn(test_values, invert);
}

pub fn isinColumnValues(self: anytype, input_name: []const u8, comptime T: type, values: []const T, invert: bool) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    const DeviceColumnType = @TypeOf(frameValue(self).columns[0]);
    const frame = frameValue(self);
    var test_values = try DeviceColumnType.fromSlice(T, frame.allocator, values, frame.device);
    defer test_values.deinit();
    return isinColumnValuesWithDeviceColumn(self, input_name, test_values, invert);
}

pub fn withColumnIsIn(self: anytype, output_name: []const u8, input_name: []const u8, test_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try isinColumns(self, input_name, test_name, false);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnIsInInverted(self: anytype, output_name: []const u8, input_name: []const u8, test_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try isinColumns(self, input_name, test_name, true);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub const withColumnIsin = withColumnIsIn;
pub const withColumnIsinInverted = withColumnIsInInverted;

pub fn withColumnIsInValues(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, values: []const T) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try isinColumnValues(self, input_name, T, values, false);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnIsInValuesInverted(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, values: []const T) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try isinColumnValues(self, input_name, T, values, true);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub const withColumnIsinValues = withColumnIsInValues;
pub const withColumnIsinValuesInverted = withColumnIsInValuesInverted;

pub fn maskedPutColumnScalar(self: anytype, input_name: []const u8, mask_name: []const u8, comptime T: type, value: T) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.maskedPutColumnScalar(frameValue(self), input_name, mask_name, T, value);
}

pub fn maskedPutColumnWithDeviceScalar(self: anytype, input_name: []const u8, mask_name: []const u8, value: DeviceScalar) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.maskedPutColumnWithDeviceScalar(frameValue(self), input_name, mask_name, value);
}

pub fn withColumnMaskedPutScalar(self: anytype, output_name: []const u8, input_name: []const u8, mask_name: []const u8, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try maskedPutColumnScalar(self, input_name, mask_name, T, value);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnMaskedPutWithDeviceScalar(self: anytype, output_name: []const u8, input_name: []const u8, mask_name: []const u8, value: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try maskedPutColumnWithDeviceScalar(self, input_name, mask_name, value);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnPutMaskScalar(self: anytype, output_name: []const u8, input_name: []const u8, mask_name: []const u8, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return withColumnMaskedPutScalar(self, output_name, input_name, mask_name, T, value);
}

pub fn withColumnPutMaskWithDeviceScalar(self: anytype, output_name: []const u8, input_name: []const u8, mask_name: []const u8, value: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    return withColumnMaskedPutWithDeviceScalar(self, output_name, input_name, mask_name, value);
}

pub fn putFlatColumnScalar(self: anytype, input_name: []const u8, row_indices: []const usize, comptime T: type, value: T) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.putFlatColumnScalar(frameValue(self), input_name, row_indices, T, value);
}

pub fn putFlatColumnWithDeviceScalar(self: anytype, input_name: []const u8, row_indices: []const usize, value: DeviceScalar) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.putFlatColumnWithDeviceScalar(frameValue(self), input_name, row_indices, value);
}

pub fn withColumnPutFlatScalar(self: anytype, output_name: []const u8, input_name: []const u8, row_indices: []const usize, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try putFlatColumnScalar(self, input_name, row_indices, T, value);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnPutFlatWithDeviceScalar(self: anytype, output_name: []const u8, input_name: []const u8, row_indices: []const usize, value: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try putFlatColumnWithDeviceScalar(self, input_name, row_indices, value);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn putFlatColumns(self: anytype, input_name: []const u8, row_indices: []const usize, value_name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.putFlatColumns(frameValue(self), input_name, row_indices, value_name);
}

pub fn withColumnPutFlat(self: anytype, output_name: []const u8, input_name: []const u8, row_indices: []const usize, value_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try putFlatColumns(self, input_name, row_indices, value_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnIndexPut(self: anytype, output_name: []const u8, input_name: []const u8, row_indices: []const usize, value_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return withColumnPutFlat(self, output_name, input_name, row_indices, value_name);
}

pub fn withColumnIndexPutScalar(self: anytype, output_name: []const u8, input_name: []const u8, row_indices: []const usize, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return withColumnPutFlatScalar(self, output_name, input_name, row_indices, T, value);
}

pub fn withColumnIndexPutWithDeviceScalar(self: anytype, output_name: []const u8, input_name: []const u8, row_indices: []const usize, value: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    return withColumnPutFlatWithDeviceScalar(self, output_name, input_name, row_indices, value);
}

pub fn putFlatColumnScalarMode(self: anytype, input_name: []const u8, row_indices: []const usize, comptime T: type, value: T, mode: array_mod.IndexMode) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.putFlatColumnScalarMode(frameValue(self), input_name, row_indices, T, value, mode);
}

pub fn putFlatColumnModeWithDeviceScalar(self: anytype, input_name: []const u8, row_indices: []const usize, value: DeviceScalar, mode: array_mod.IndexMode) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.putFlatColumnModeWithDeviceScalar(frameValue(self), input_name, row_indices, value, mode);
}

pub fn withColumnPutFlatScalarMode(self: anytype, output_name: []const u8, input_name: []const u8, row_indices: []const usize, comptime T: type, value: T, mode: array_mod.IndexMode) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try putFlatColumnScalarMode(self, input_name, row_indices, T, value, mode);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnPutFlatModeWithDeviceScalar(self: anytype, output_name: []const u8, input_name: []const u8, row_indices: []const usize, value: DeviceScalar, mode: array_mod.IndexMode) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try putFlatColumnModeWithDeviceScalar(self, input_name, row_indices, value, mode);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn putFlatColumnScalarSigned(self: anytype, input_name: []const u8, row_indices: []const isize, comptime T: type, value: T) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.putFlatColumnScalarSigned(frameValue(self), input_name, row_indices, T, value);
}

pub fn putFlatColumnSignedWithDeviceScalar(self: anytype, input_name: []const u8, row_indices: []const isize, value: DeviceScalar) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.putFlatColumnSignedWithDeviceScalar(frameValue(self), input_name, row_indices, value);
}

pub fn withColumnPutFlatScalarSigned(self: anytype, output_name: []const u8, input_name: []const u8, row_indices: []const isize, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try putFlatColumnScalarSigned(self, input_name, row_indices, T, value);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnPutFlatSignedWithDeviceScalar(self: anytype, output_name: []const u8, input_name: []const u8, row_indices: []const isize, value: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try putFlatColumnSignedWithDeviceScalar(self, input_name, row_indices, value);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnIndexPutScalarSigned(self: anytype, output_name: []const u8, input_name: []const u8, row_indices: []const isize, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return withColumnPutFlatScalarSigned(self, output_name, input_name, row_indices, T, value);
}

pub fn withColumnIndexPutSignedWithDeviceScalar(self: anytype, output_name: []const u8, input_name: []const u8, row_indices: []const isize, value: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    return withColumnPutFlatSignedWithDeviceScalar(self, output_name, input_name, row_indices, value);
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

pub fn betweenColumnScalar(self: anytype, name: []const u8, comptime T: type, lower: T, upper: T, lower_inclusive: bool, upper_inclusive: bool) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.betweenColumnScalar(frameValue(self), name, T, lower, upper, lower_inclusive, upper_inclusive);
}

pub fn betweenColumnWithDeviceScalars(self: anytype, name: []const u8, lower: DeviceScalar, upper: DeviceScalar, lower_inclusive: bool, upper_inclusive: bool) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.betweenColumnWithDeviceScalars(frameValue(self), name, lower, upper, lower_inclusive, upper_inclusive);
}

pub fn withColumnBetween(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, lower: T, upper: T) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try betweenColumnScalar(self, input_name, T, lower, upper, true, true);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnIsBetween(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, lower: T, upper: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return withColumnBetween(self, output_name, input_name, T, lower, upper);
}

pub fn withColumnBetweenClosed(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, lower: T, upper: T, lower_inclusive: bool, upper_inclusive: bool) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try betweenColumnScalar(self, input_name, T, lower, upper, lower_inclusive, upper_inclusive);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnBetweenWithDeviceScalars(self: anytype, output_name: []const u8, input_name: []const u8, lower: DeviceScalar, upper: DeviceScalar, lower_inclusive: bool, upper_inclusive: bool) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try betweenColumnWithDeviceScalars(self, input_name, lower, upper, lower_inclusive, upper_inclusive);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnBetweenExclusive(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, lower: T, upper: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return withColumnBetweenClosed(self, output_name, input_name, T, lower, upper, false, false);
}

pub fn withColumnBetweenLeftClosed(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, lower: T, upper: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return withColumnBetweenClosed(self, output_name, input_name, T, lower, upper, true, false);
}

pub fn withColumnBetweenRightClosed(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, lower: T, upper: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return withColumnBetweenClosed(self, output_name, input_name, T, lower, upper, false, true);
}

pub fn notBetweenColumnScalar(self: anytype, name: []const u8, comptime T: type, lower: T, upper: T, lower_inclusive: bool, upper_inclusive: bool) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.notBetweenColumnScalar(frameValue(self), name, T, lower, upper, lower_inclusive, upper_inclusive);
}

pub fn notBetweenColumnWithDeviceScalars(self: anytype, name: []const u8, lower: DeviceScalar, upper: DeviceScalar, lower_inclusive: bool, upper_inclusive: bool) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.notBetweenColumnWithDeviceScalars(frameValue(self), name, lower, upper, lower_inclusive, upper_inclusive);
}

pub fn withColumnNotBetween(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, lower: T, upper: T) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try notBetweenColumnScalar(self, input_name, T, lower, upper, true, true);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnOutside(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, lower: T, upper: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return withColumnNotBetween(self, output_name, input_name, T, lower, upper);
}

pub fn withColumnNotBetweenClosed(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, lower: T, upper: T, lower_inclusive: bool, upper_inclusive: bool) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try notBetweenColumnScalar(self, input_name, T, lower, upper, lower_inclusive, upper_inclusive);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnNotBetweenWithDeviceScalars(self: anytype, output_name: []const u8, input_name: []const u8, lower: DeviceScalar, upper: DeviceScalar, lower_inclusive: bool, upper_inclusive: bool) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try notBetweenColumnWithDeviceScalars(self, input_name, lower, upper, lower_inclusive, upper_inclusive);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnNotBetweenExclusive(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, lower: T, upper: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return withColumnNotBetweenClosed(self, output_name, input_name, T, lower, upper, false, false);
}

pub fn withColumnNotBetweenLeftClosed(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, lower: T, upper: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return withColumnNotBetweenClosed(self, output_name, input_name, T, lower, upper, true, false);
}

pub fn withColumnNotBetweenRightClosed(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, lower: T, upper: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return withColumnNotBetweenClosed(self, output_name, input_name, T, lower, upper, false, true);
}

pub fn iscloseColumnScalar(self: anytype, name: []const u8, comptime T: type, scalar: T, rtol: T, atol: T) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.iscloseColumnScalar(frameValue(self), name, T, scalar, rtol, atol, false);
}

pub fn iscloseColumnScalarEqualNan(self: anytype, name: []const u8, comptime T: type, scalar: T, rtol: T, atol: T, equal_nan: bool) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.iscloseColumnScalar(frameValue(self), name, T, scalar, rtol, atol, equal_nan);
}

pub fn iscloseColumnWithDeviceScalars(self: anytype, name: []const u8, scalar: DeviceScalar, rtol: DeviceScalar, atol: DeviceScalar) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.iscloseColumnWithDeviceScalars(frameValue(self), name, scalar, rtol, atol, false);
}

pub fn iscloseColumnWithDeviceScalarsEqualNan(self: anytype, name: []const u8, scalar: DeviceScalar, rtol: DeviceScalar, atol: DeviceScalar, equal_nan: bool) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.iscloseColumnWithDeviceScalars(frameValue(self), name, scalar, rtol, atol, equal_nan);
}

pub fn withColumnIscloseScalar(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, scalar: T, rtol: T, atol: T) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try iscloseColumnScalar(self, input_name, T, scalar, rtol, atol);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnIscloseScalarEqualNan(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, scalar: T, rtol: T, atol: T, equal_nan: bool) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try iscloseColumnScalarEqualNan(self, input_name, T, scalar, rtol, atol, equal_nan);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnIscloseWithDeviceScalars(self: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar, rtol: DeviceScalar, atol: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try iscloseColumnWithDeviceScalars(self, input_name, scalar, rtol, atol);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnIscloseWithDeviceScalarsEqualNan(self: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar, rtol: DeviceScalar, atol: DeviceScalar, equal_nan: bool) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try iscloseColumnWithDeviceScalarsEqualNan(self, input_name, scalar, rtol, atol, equal_nan);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn allcloseColumnScalar(self: anytype, name: []const u8, comptime T: type, scalar: T, rtol: T, atol: T) DeviceDataError!bool {
    return expr_mod.allcloseColumnScalar(frameValue(self), name, T, scalar, rtol, atol, false);
}

pub fn allcloseColumnScalarEqualNan(self: anytype, name: []const u8, comptime T: type, scalar: T, rtol: T, atol: T, equal_nan: bool) DeviceDataError!bool {
    return expr_mod.allcloseColumnScalar(frameValue(self), name, T, scalar, rtol, atol, equal_nan);
}

pub fn allcloseColumnWithDeviceScalars(self: anytype, name: []const u8, scalar: DeviceScalar, rtol: DeviceScalar, atol: DeviceScalar) DeviceDataError!bool {
    return expr_mod.allcloseColumnWithDeviceScalars(frameValue(self), name, scalar, rtol, atol, false);
}

pub fn allcloseColumnWithDeviceScalarsEqualNan(self: anytype, name: []const u8, scalar: DeviceScalar, rtol: DeviceScalar, atol: DeviceScalar, equal_nan: bool) DeviceDataError!bool {
    return expr_mod.allcloseColumnWithDeviceScalars(frameValue(self), name, scalar, rtol, atol, equal_nan);
}

pub fn countNonzeroColumn(self: anytype, name: []const u8) DeviceDataError!usize {
    return expr_mod.countNonzeroColumn(frameValue(self), name);
}

pub fn zeroCountColumn(self: anytype, name: []const u8) DeviceDataError!usize {
    return expr_mod.zeroCountColumn(frameValue(self), name);
}

pub fn countZeroColumn(self: anytype, name: []const u8) DeviceDataError!usize {
    return zeroCountColumn(self, name);
}

pub fn nanCountColumn(self: anytype, name: []const u8) DeviceDataError!usize {
    return expr_mod.nanCountColumn(frameValue(self), name);
}

pub fn positiveZeroCountColumn(self: anytype, name: []const u8) DeviceDataError!usize {
    return expr_mod.positiveZeroCountColumn(frameValue(self), name);
}

pub fn negativeZeroCountColumn(self: anytype, name: []const u8) DeviceDataError!usize {
    return expr_mod.negativeZeroCountColumn(frameValue(self), name);
}

pub fn positiveCountColumn(self: anytype, name: []const u8) DeviceDataError!usize {
    return expr_mod.positiveCountColumn(frameValue(self), name);
}

pub fn negativeCountColumn(self: anytype, name: []const u8) DeviceDataError!usize {
    return expr_mod.negativeCountColumn(frameValue(self), name);
}

pub fn signBitCountColumn(self: anytype, name: []const u8) DeviceDataError!usize {
    return expr_mod.signBitCountColumn(frameValue(self), name);
}

pub fn infCountColumn(self: anytype, name: []const u8) DeviceDataError!usize {
    return expr_mod.infCountColumn(frameValue(self), name);
}

pub fn positiveInfCountColumn(self: anytype, name: []const u8) DeviceDataError!usize {
    return expr_mod.positiveInfCountColumn(frameValue(self), name);
}

pub fn negativeInfCountColumn(self: anytype, name: []const u8) DeviceDataError!usize {
    return expr_mod.negativeInfCountColumn(frameValue(self), name);
}

pub fn finiteCountColumn(self: anytype, name: []const u8) DeviceDataError!usize {
    return expr_mod.finiteCountColumn(frameValue(self), name);
}

pub fn nonFiniteCountColumn(self: anytype, name: []const u8) DeviceDataError!usize {
    return expr_mod.nonFiniteCountColumn(frameValue(self), name);
}

pub fn normalCountColumn(self: anytype, name: []const u8) DeviceDataError!usize {
    return expr_mod.normalCountColumn(frameValue(self), name);
}

pub fn subnormalCountColumn(self: anytype, name: []const u8) DeviceDataError!usize {
    return expr_mod.subnormalCountColumn(frameValue(self), name);
}

pub fn anyZeroColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return expr_mod.anyZeroColumn(frameValue(self), name);
}

pub fn allZeroColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return expr_mod.allZeroColumn(frameValue(self), name);
}

pub fn anyNonzeroColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return expr_mod.anyNonzeroColumn(frameValue(self), name);
}

pub fn anyNonZeroColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return anyNonzeroColumn(self, name);
}

pub fn allNonzeroColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return expr_mod.allNonzeroColumn(frameValue(self), name);
}

pub fn allNonZeroColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return allNonzeroColumn(self, name);
}

pub fn anyPositiveZeroColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return expr_mod.anyPositiveZeroColumn(frameValue(self), name);
}

pub fn allPositiveZeroColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return expr_mod.allPositiveZeroColumn(frameValue(self), name);
}

pub fn anyNegativeZeroColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return expr_mod.anyNegativeZeroColumn(frameValue(self), name);
}

pub fn allNegativeZeroColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return expr_mod.allNegativeZeroColumn(frameValue(self), name);
}

pub fn anyPositiveColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return expr_mod.anyPositiveColumn(frameValue(self), name);
}

pub fn allPositiveColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return expr_mod.allPositiveColumn(frameValue(self), name);
}

pub fn anyNegativeColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return expr_mod.anyNegativeColumn(frameValue(self), name);
}

pub fn allNegativeColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return expr_mod.allNegativeColumn(frameValue(self), name);
}

pub fn anySignBitColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return expr_mod.anySignBitColumn(frameValue(self), name);
}

pub fn allSignBitColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return expr_mod.allSignBitColumn(frameValue(self), name);
}

pub fn anyNanColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return expr_mod.anyNanColumn(frameValue(self), name);
}

pub fn anyNaNColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return anyNanColumn(self, name);
}

pub fn allNanColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return expr_mod.allNanColumn(frameValue(self), name);
}

pub fn allNaNColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return allNanColumn(self, name);
}

pub fn anyInfColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return expr_mod.anyInfColumn(frameValue(self), name);
}

pub fn allInfColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return expr_mod.allInfColumn(frameValue(self), name);
}

pub fn anyPositiveInfColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return expr_mod.anyPositiveInfColumn(frameValue(self), name);
}

pub fn allPositiveInfColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return expr_mod.allPositiveInfColumn(frameValue(self), name);
}

pub fn anyNegativeInfColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return expr_mod.anyNegativeInfColumn(frameValue(self), name);
}

pub fn allNegativeInfColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return expr_mod.allNegativeInfColumn(frameValue(self), name);
}

pub fn anyFiniteColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return expr_mod.anyFiniteColumn(frameValue(self), name);
}

pub fn allFiniteColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return expr_mod.allFiniteColumn(frameValue(self), name);
}

pub fn anyNonFiniteColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return expr_mod.anyNonFiniteColumn(frameValue(self), name);
}

pub fn allNonFiniteColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return expr_mod.allNonFiniteColumn(frameValue(self), name);
}

pub fn anyNormalColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return expr_mod.anyNormalColumn(frameValue(self), name);
}

pub fn allNormalColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return expr_mod.allNormalColumn(frameValue(self), name);
}

pub fn anySubnormalColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return expr_mod.anySubnormalColumn(frameValue(self), name);
}

pub fn allSubnormalColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return expr_mod.allSubnormalColumn(frameValue(self), name);
}

pub fn zeroRatioColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return expr_mod.zeroRatioColumn(frameValue(self), name);
}

pub fn nonzeroRatioColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return expr_mod.nonzeroRatioColumn(frameValue(self), name);
}

pub fn nonZeroRatioColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return nonzeroRatioColumn(self, name);
}

pub fn positiveZeroRatioColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return expr_mod.positiveZeroRatioColumn(frameValue(self), name);
}

pub fn negativeZeroRatioColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return expr_mod.negativeZeroRatioColumn(frameValue(self), name);
}

pub fn positiveRatioColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return expr_mod.positiveRatioColumn(frameValue(self), name);
}

pub fn negativeRatioColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return expr_mod.negativeRatioColumn(frameValue(self), name);
}

pub fn signBitRatioColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return expr_mod.signBitRatioColumn(frameValue(self), name);
}

pub fn nanRatioColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return expr_mod.nanRatioColumn(frameValue(self), name);
}

pub fn infRatioColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return expr_mod.infRatioColumn(frameValue(self), name);
}

pub fn positiveInfRatioColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return expr_mod.positiveInfRatioColumn(frameValue(self), name);
}

pub fn negativeInfRatioColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return expr_mod.negativeInfRatioColumn(frameValue(self), name);
}

pub fn finiteRatioColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return expr_mod.finiteRatioColumn(frameValue(self), name);
}

pub fn nonFiniteRatioColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return expr_mod.nonFiniteRatioColumn(frameValue(self), name);
}

pub fn normalRatioColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return expr_mod.normalRatioColumn(frameValue(self), name);
}

pub fn subnormalRatioColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return expr_mod.subnormalRatioColumn(frameValue(self), name);
}

pub fn firstZeroIndexColumn(self: anytype, name: []const u8) DeviceDataError!?usize {
    return expr_mod.firstZeroIndexColumn(frameValue(self), name);
}

pub fn lastZeroIndexColumn(self: anytype, name: []const u8) DeviceDataError!?usize {
    return expr_mod.lastZeroIndexColumn(frameValue(self), name);
}

pub fn firstPositiveZeroIndexColumn(self: anytype, name: []const u8) DeviceDataError!?usize {
    return expr_mod.firstPositiveZeroIndexColumn(frameValue(self), name);
}

pub fn lastPositiveZeroIndexColumn(self: anytype, name: []const u8) DeviceDataError!?usize {
    return expr_mod.lastPositiveZeroIndexColumn(frameValue(self), name);
}

pub fn firstNegativeZeroIndexColumn(self: anytype, name: []const u8) DeviceDataError!?usize {
    return expr_mod.firstNegativeZeroIndexColumn(frameValue(self), name);
}

pub fn lastNegativeZeroIndexColumn(self: anytype, name: []const u8) DeviceDataError!?usize {
    return expr_mod.lastNegativeZeroIndexColumn(frameValue(self), name);
}

pub fn firstNonzeroIndexColumn(self: anytype, name: []const u8) DeviceDataError!?usize {
    return expr_mod.firstNonzeroIndexColumn(frameValue(self), name);
}

pub fn lastNonzeroIndexColumn(self: anytype, name: []const u8) DeviceDataError!?usize {
    return expr_mod.lastNonzeroIndexColumn(frameValue(self), name);
}

pub fn firstPositiveIndexColumn(self: anytype, name: []const u8) DeviceDataError!?usize {
    return expr_mod.firstPositiveIndexColumn(frameValue(self), name);
}

pub fn lastPositiveIndexColumn(self: anytype, name: []const u8) DeviceDataError!?usize {
    return expr_mod.lastPositiveIndexColumn(frameValue(self), name);
}

pub fn firstNegativeIndexColumn(self: anytype, name: []const u8) DeviceDataError!?usize {
    return expr_mod.firstNegativeIndexColumn(frameValue(self), name);
}

pub fn lastNegativeIndexColumn(self: anytype, name: []const u8) DeviceDataError!?usize {
    return expr_mod.lastNegativeIndexColumn(frameValue(self), name);
}

pub fn firstSignBitIndexColumn(self: anytype, name: []const u8) DeviceDataError!?usize {
    return expr_mod.firstSignBitIndexColumn(frameValue(self), name);
}

pub fn lastSignBitIndexColumn(self: anytype, name: []const u8) DeviceDataError!?usize {
    return expr_mod.lastSignBitIndexColumn(frameValue(self), name);
}

pub fn firstNanIndexColumn(self: anytype, name: []const u8) DeviceDataError!?usize {
    return expr_mod.firstNanIndexColumn(frameValue(self), name);
}

pub fn firstNaNIndexColumn(self: anytype, name: []const u8) DeviceDataError!?usize {
    return firstNanIndexColumn(self, name);
}

pub fn lastNanIndexColumn(self: anytype, name: []const u8) DeviceDataError!?usize {
    return expr_mod.lastNanIndexColumn(frameValue(self), name);
}

pub fn lastNaNIndexColumn(self: anytype, name: []const u8) DeviceDataError!?usize {
    return lastNanIndexColumn(self, name);
}

pub fn firstInfIndexColumn(self: anytype, name: []const u8) DeviceDataError!?usize {
    return expr_mod.firstInfIndexColumn(frameValue(self), name);
}

pub fn lastInfIndexColumn(self: anytype, name: []const u8) DeviceDataError!?usize {
    return expr_mod.lastInfIndexColumn(frameValue(self), name);
}

pub fn firstPositiveInfIndexColumn(self: anytype, name: []const u8) DeviceDataError!?usize {
    return expr_mod.firstPositiveInfIndexColumn(frameValue(self), name);
}

pub fn lastPositiveInfIndexColumn(self: anytype, name: []const u8) DeviceDataError!?usize {
    return expr_mod.lastPositiveInfIndexColumn(frameValue(self), name);
}

pub fn firstNegativeInfIndexColumn(self: anytype, name: []const u8) DeviceDataError!?usize {
    return expr_mod.firstNegativeInfIndexColumn(frameValue(self), name);
}

pub fn lastNegativeInfIndexColumn(self: anytype, name: []const u8) DeviceDataError!?usize {
    return expr_mod.lastNegativeInfIndexColumn(frameValue(self), name);
}

pub fn firstFiniteIndexColumn(self: anytype, name: []const u8) DeviceDataError!?usize {
    return expr_mod.firstFiniteIndexColumn(frameValue(self), name);
}

pub fn lastFiniteIndexColumn(self: anytype, name: []const u8) DeviceDataError!?usize {
    return expr_mod.lastFiniteIndexColumn(frameValue(self), name);
}

pub fn firstNormalIndexColumn(self: anytype, name: []const u8) DeviceDataError!?usize {
    return expr_mod.firstNormalIndexColumn(frameValue(self), name);
}

pub fn lastNormalIndexColumn(self: anytype, name: []const u8) DeviceDataError!?usize {
    return expr_mod.lastNormalIndexColumn(frameValue(self), name);
}

pub fn firstSubnormalIndexColumn(self: anytype, name: []const u8) DeviceDataError!?usize {
    return expr_mod.firstSubnormalIndexColumn(frameValue(self), name);
}

pub fn lastSubnormalIndexColumn(self: anytype, name: []const u8) DeviceDataError!?usize {
    return expr_mod.lastSubnormalIndexColumn(frameValue(self), name);
}

pub fn firstNonFiniteIndexColumn(self: anytype, name: []const u8) DeviceDataError!?usize {
    return expr_mod.firstNonFiniteIndexColumn(frameValue(self), name);
}

pub fn lastNonFiniteIndexColumn(self: anytype, name: []const u8) DeviceDataError!?usize {
    return expr_mod.lastNonFiniteIndexColumn(frameValue(self), name);
}

pub fn firstValidIndexColumn(self: anytype, name: []const u8) DeviceDataError!?usize {
    return expr_mod.firstValidIndexColumn(frameValue(self), name);
}

pub fn lastValidIndexColumn(self: anytype, name: []const u8) DeviceDataError!?usize {
    return expr_mod.lastValidIndexColumn(frameValue(self), name);
}

pub fn firstNullIndexColumn(self: anytype, name: []const u8) DeviceDataError!?usize {
    return expr_mod.firstNullIndexColumn(frameValue(self), name);
}

pub fn lastNullIndexColumn(self: anytype, name: []const u8) DeviceDataError!?usize {
    return expr_mod.lastNullIndexColumn(frameValue(self), name);
}

pub fn countDistinctColumn(self: anytype, name: []const u8) DeviceDataError!usize {
    return expr_mod.countDistinctColumn(frameValue(self), name);
}

pub fn nUniqueColumn(self: anytype, name: []const u8) DeviceDataError!usize {
    return expr_mod.nUniqueColumn(frameValue(self), name);
}

pub fn nullCountColumn(self: anytype, name: []const u8) DeviceDataError!usize {
    return expr_mod.nullCountColumn(frameValue(self), name);
}

pub fn validCountColumn(self: anytype, name: []const u8) DeviceDataError!usize {
    return expr_mod.validCountColumn(frameValue(self), name);
}

pub fn anyNullColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return expr_mod.anyNullColumn(frameValue(self), name);
}

pub fn allNullColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return expr_mod.allNullColumn(frameValue(self), name);
}

pub fn anyValidColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return expr_mod.anyValidColumn(frameValue(self), name);
}

pub fn allValidColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return expr_mod.allValidColumn(frameValue(self), name);
}

pub fn nullRatioColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return expr_mod.nullRatioColumn(frameValue(self), name);
}

pub fn validRatioColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return expr_mod.validRatioColumn(frameValue(self), name);
}

pub fn modeColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return expr_mod.modeColumn(frameValue(self), name);
}

pub fn sumColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return expr_mod.sumColumn(frameValue(self), name);
}

pub fn prodColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return expr_mod.prodColumn(frameValue(self), name);
}

pub fn meanColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return expr_mod.meanColumn(frameValue(self), name);
}

pub fn quantileColumn(self: anytype, name: []const u8, q: f64) DeviceDataError!DeviceScalar {
    return expr_mod.quantileColumn(frameValue(self), name, q);
}

pub fn medianColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return expr_mod.medianColumn(frameValue(self), name);
}

pub fn varianceColumn(self: anytype, name: []const u8, correction: f64) DeviceDataError!DeviceScalar {
    return expr_mod.varianceColumn(frameValue(self), name, correction);
}

pub fn varColumn(self: anytype, name: []const u8, correction: f64) DeviceDataError!DeviceScalar {
    return varianceColumn(self, name, correction);
}

pub fn stddevColumn(self: anytype, name: []const u8, correction: f64) DeviceDataError!DeviceScalar {
    return expr_mod.stddevColumn(frameValue(self), name, correction);
}

pub fn stdColumn(self: anytype, name: []const u8, correction: f64) DeviceDataError!DeviceScalar {
    return stddevColumn(self, name, correction);
}

pub fn semColumn(self: anytype, name: []const u8, correction: f64) DeviceDataError!DeviceScalar {
    return expr_mod.semColumn(frameValue(self), name, correction);
}

pub fn cvColumn(self: anytype, name: []const u8, correction: f64) DeviceDataError!DeviceScalar {
    return expr_mod.cvColumn(frameValue(self), name, correction);
}

pub fn skewnessColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return expr_mod.skewnessColumn(frameValue(self), name);
}

pub fn skewColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return skewnessColumn(self, name);
}

pub fn kurtosisColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return expr_mod.kurtosisColumn(frameValue(self), name);
}

pub fn kurtColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return kurtosisColumn(self, name);
}

pub fn meanAbsColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return expr_mod.meanAbsColumn(frameValue(self), name);
}

pub fn rmsColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return expr_mod.rmsColumn(frameValue(self), name);
}

pub fn l1NormColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return expr_mod.l1NormColumn(frameValue(self), name);
}

pub fn l2NormColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return expr_mod.l2NormColumn(frameValue(self), name);
}

pub fn geometricMeanColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return expr_mod.geometricMeanColumn(frameValue(self), name);
}

pub fn geoMeanColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return geometricMeanColumn(self, name);
}

pub fn harmonicMeanColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return expr_mod.harmonicMeanColumn(frameValue(self), name);
}

pub fn harmMeanColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return harmonicMeanColumn(self, name);
}

pub fn madColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return expr_mod.madColumn(frameValue(self), name);
}

pub fn medianAbsDevColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return madColumn(self, name);
}

pub fn iqrColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return expr_mod.iqrColumn(frameValue(self), name);
}

pub fn minColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return expr_mod.minColumn(frameValue(self), name);
}

pub fn maxColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return expr_mod.maxColumn(frameValue(self), name);
}

pub fn ptpColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return expr_mod.ptpColumn(frameValue(self), name);
}

pub fn argminColumn(self: anytype, name: []const u8) DeviceDataError!usize {
    return expr_mod.argminColumn(frameValue(self), name);
}

pub fn argmaxColumn(self: anytype, name: []const u8) DeviceDataError!usize {
    return expr_mod.argmaxColumn(frameValue(self), name);
}

pub fn anyColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return expr_mod.anyColumn(frameValue(self), name);
}

pub fn allColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return expr_mod.allColumn(frameValue(self), name);
}

pub fn anyTrueColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return expr_mod.anyTrueColumn(frameValue(self), name);
}

pub fn allTrueColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return expr_mod.allTrueColumn(frameValue(self), name);
}

pub fn anyFalseColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return expr_mod.anyFalseColumn(frameValue(self), name);
}

pub fn allFalseColumn(self: anytype, name: []const u8) DeviceDataError!bool {
    return expr_mod.allFalseColumn(frameValue(self), name);
}

pub fn countTrueColumn(self: anytype, name: []const u8) DeviceDataError!usize {
    return expr_mod.countTrueColumn(frameValue(self), name);
}

pub fn countFalseColumn(self: anytype, name: []const u8) DeviceDataError!usize {
    return expr_mod.countFalseColumn(frameValue(self), name);
}

pub fn trueRatioColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return expr_mod.trueRatioColumn(frameValue(self), name);
}

pub fn falseRatioColumn(self: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    return expr_mod.falseRatioColumn(frameValue(self), name);
}

pub fn firstTrueIndexColumn(self: anytype, name: []const u8) DeviceDataError!?usize {
    return expr_mod.firstTrueIndexColumn(frameValue(self), name);
}

pub fn lastTrueIndexColumn(self: anytype, name: []const u8) DeviceDataError!?usize {
    return expr_mod.lastTrueIndexColumn(frameValue(self), name);
}

pub fn firstFalseIndexColumn(self: anytype, name: []const u8) DeviceDataError!?usize {
    return expr_mod.firstFalseIndexColumn(frameValue(self), name);
}

pub fn lastFalseIndexColumn(self: anytype, name: []const u8) DeviceDataError!?usize {
    return expr_mod.lastFalseIndexColumn(frameValue(self), name);
}

pub fn logicalColumnScalar(self: anytype, name: []const u8, scalar: bool, op: DeviceColumnLogicalOp) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.logicalColumnScalar(frameValue(self), name, scalar, op);
}

pub fn logicalNotColumn(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.logicalNotColumn(frameValue(self), name);
}

pub fn notColumn(self: anytype, name: []const u8) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.notColumn(frameValue(self), name);
}

pub fn withColumnLogicalScalar(self: anytype, output_name: []const u8, input_name: []const u8, scalar: bool, op: DeviceColumnLogicalOp) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try logicalColumnScalar(self, input_name, scalar, op);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnLogicalAndScalar(self: anytype, output_name: []const u8, input_name: []const u8, scalar: bool) DeviceDataError!FrameType(@TypeOf(self)) {
    return withColumnLogicalScalar(self, output_name, input_name, scalar, .@"and");
}

pub fn withColumnLogicalOrScalar(self: anytype, output_name: []const u8, input_name: []const u8, scalar: bool) DeviceDataError!FrameType(@TypeOf(self)) {
    return withColumnLogicalScalar(self, output_name, input_name, scalar, .@"or");
}

pub fn withColumnLogicalXorScalar(self: anytype, output_name: []const u8, input_name: []const u8, scalar: bool) DeviceDataError!FrameType(@TypeOf(self)) {
    return withColumnLogicalScalar(self, output_name, input_name, scalar, .xor);
}

pub fn withColumnLogicalNot(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try logicalNotColumn(self, input_name);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnNot(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return withColumnLogicalNot(self, output_name, input_name);
}

pub fn logicalColumns(self: anytype, lhs_name: []const u8, rhs_name: []const u8, op: DeviceColumnLogicalOp) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.logicalColumns(frameValue(self), lhs_name, rhs_name, op);
}

pub fn withColumnLogical(self: anytype, output_name: []const u8, lhs_name: []const u8, rhs_name: []const u8, op: DeviceColumnLogicalOp) DeviceDataError!FrameType(@TypeOf(self)) {
    var column = try logicalColumns(self, lhs_name, rhs_name, op);
    defer column.deinit();
    return dataframe_array_mod.withColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, column);
}

pub fn withColumnLogicalAnd(self: anytype, output_name: []const u8, lhs_name: []const u8, rhs_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return withColumnLogical(self, output_name, lhs_name, rhs_name, .@"and");
}

pub fn withColumnLogicalOr(self: anytype, output_name: []const u8, lhs_name: []const u8, rhs_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return withColumnLogical(self, output_name, lhs_name, rhs_name, .@"or");
}

pub fn withColumnLogicalXor(self: anytype, output_name: []const u8, lhs_name: []const u8, rhs_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return withColumnLogical(self, output_name, lhs_name, rhs_name, .xor);
}

pub fn filterColumnMask(self: anytype, mask: @TypeOf(frameValue(self).columns[0])) DeviceDataError!FrameType(@TypeOf(self)) {
    return expr_mod.filterColumnMask(FrameType(@TypeOf(self)), frameValue(self), mask);
}

pub fn dropColumnMask(self: anytype, mask: @TypeOf(frameValue(self).columns[0])) DeviceDataError!FrameType(@TypeOf(self)) {
    return expr_mod.dropColumnMask(FrameType(@TypeOf(self)), frameValue(self), mask);
}

pub fn filterColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return expr_mod.filterColumn(FrameType(@TypeOf(self)), frameValue(self), name);
}

pub fn filterColumnScalarWithDeviceScalar(self: anytype, name: []const u8, scalar: DeviceScalar, op: DeviceColumnCompareOp) DeviceDataError!FrameType(@TypeOf(self)) {
    var mask = try compareColumnScalarWithDeviceScalar(self, name, scalar, op);
    defer mask.deinit();
    return filterColumnMask(self, mask);
}

pub fn filterColumnScalar(self: anytype, name: []const u8, comptime T: type, scalar: T, op: DeviceColumnCompareOp) DeviceDataError!FrameType(@TypeOf(self)) {
    return filterColumnScalarWithDeviceScalar(self, name, DeviceScalar.init(T, scalar), op);
}

pub fn dropColumnScalarWithDeviceScalar(self: anytype, name: []const u8, scalar: DeviceScalar, op: DeviceColumnCompareOp) DeviceDataError!FrameType(@TypeOf(self)) {
    var mask = try compareColumnScalarWithDeviceScalar(self, name, scalar, op);
    defer mask.deinit();
    return dropColumnMask(self, mask);
}

pub fn dropColumnScalar(self: anytype, name: []const u8, comptime T: type, scalar: T, op: DeviceColumnCompareOp) DeviceDataError!FrameType(@TypeOf(self)) {
    return dropColumnScalarWithDeviceScalar(self, name, DeviceScalar.init(T, scalar), op);
}

fn filterIsInColumnMode(self: anytype, input_name: []const u8, test_name: []const u8, invert: bool) DeviceDataError!FrameType(@TypeOf(self)) {
    var mask = try isinColumns(self, input_name, test_name, invert);
    defer mask.deinit();
    return filterColumnMask(self, mask);
}

pub fn filterIsInColumn(self: anytype, input_name: []const u8, test_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return filterIsInColumnMode(self, input_name, test_name, false);
}

pub fn filterNotInColumn(self: anytype, input_name: []const u8, test_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return filterIsInColumnMode(self, input_name, test_name, true);
}

pub const filterIsinColumn = filterIsInColumn;
pub const filterIsInColumnInverted = filterNotInColumn;
pub const filterIsinColumnInverted = filterNotInColumn;

fn filterIsInValuesMode(self: anytype, input_name: []const u8, comptime T: type, values: []const T, invert: bool) DeviceDataError!FrameType(@TypeOf(self)) {
    var mask = try isinColumnValues(self, input_name, T, values, invert);
    defer mask.deinit();
    return filterColumnMask(self, mask);
}

pub fn filterIsInValues(self: anytype, input_name: []const u8, comptime T: type, values: []const T) DeviceDataError!FrameType(@TypeOf(self)) {
    return filterIsInValuesMode(self, input_name, T, values, false);
}

pub fn filterNotInValues(self: anytype, input_name: []const u8, comptime T: type, values: []const T) DeviceDataError!FrameType(@TypeOf(self)) {
    return filterIsInValuesMode(self, input_name, T, values, true);
}

pub const filterIsinValues = filterIsInValues;
pub const filterIsInValuesInverted = filterNotInValues;
pub const filterIsinValuesInverted = filterNotInValues;

pub fn dropIsInColumn(self: anytype, input_name: []const u8, test_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return filterNotInColumn(self, input_name, test_name);
}

pub fn dropNotInColumn(self: anytype, input_name: []const u8, test_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return filterIsInColumn(self, input_name, test_name);
}

pub const dropIsinColumn = dropIsInColumn;
pub const dropIsInColumnInverted = dropNotInColumn;
pub const dropIsinColumnInverted = dropNotInColumn;

pub fn dropIsInValues(self: anytype, input_name: []const u8, comptime T: type, values: []const T) DeviceDataError!FrameType(@TypeOf(self)) {
    return filterNotInValues(self, input_name, T, values);
}

pub fn dropNotInValues(self: anytype, input_name: []const u8, comptime T: type, values: []const T) DeviceDataError!FrameType(@TypeOf(self)) {
    return filterIsInValues(self, input_name, T, values);
}

pub const dropIsinValues = dropIsInValues;
pub const dropIsInValuesInverted = dropNotInValues;
pub const dropIsinValuesInverted = dropNotInValues;

pub fn filterBetweenColumnWithDeviceScalars(self: anytype, name: []const u8, lower: DeviceScalar, upper: DeviceScalar, lower_inclusive: bool, upper_inclusive: bool) DeviceDataError!FrameType(@TypeOf(self)) {
    var mask = try betweenColumnWithDeviceScalars(self, name, lower, upper, lower_inclusive, upper_inclusive);
    defer mask.deinit();
    return filterColumnMask(self, mask);
}

pub fn filterBetweenColumnClosed(self: anytype, name: []const u8, comptime T: type, lower: T, upper: T, lower_inclusive: bool, upper_inclusive: bool) DeviceDataError!FrameType(@TypeOf(self)) {
    return filterBetweenColumnWithDeviceScalars(self, name, DeviceScalar.init(T, lower), DeviceScalar.init(T, upper), lower_inclusive, upper_inclusive);
}

pub fn filterBetweenColumn(self: anytype, name: []const u8, comptime T: type, lower: T, upper: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return filterBetweenColumnClosed(self, name, T, lower, upper, true, true);
}

pub fn filterOutsideColumnWithDeviceScalars(self: anytype, name: []const u8, lower: DeviceScalar, upper: DeviceScalar, lower_inclusive: bool, upper_inclusive: bool) DeviceDataError!FrameType(@TypeOf(self)) {
    var mask = try notBetweenColumnWithDeviceScalars(self, name, lower, upper, lower_inclusive, upper_inclusive);
    defer mask.deinit();
    return filterColumnMask(self, mask);
}

pub fn filterOutsideColumnClosed(self: anytype, name: []const u8, comptime T: type, lower: T, upper: T, lower_inclusive: bool, upper_inclusive: bool) DeviceDataError!FrameType(@TypeOf(self)) {
    return filterOutsideColumnWithDeviceScalars(self, name, DeviceScalar.init(T, lower), DeviceScalar.init(T, upper), lower_inclusive, upper_inclusive);
}

pub fn filterOutsideColumn(self: anytype, name: []const u8, comptime T: type, lower: T, upper: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return filterOutsideColumnClosed(self, name, T, lower, upper, true, true);
}

pub fn dropBetweenColumn(self: anytype, name: []const u8, comptime T: type, lower: T, upper: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return filterOutsideColumn(self, name, T, lower, upper);
}

pub fn dropOutsideColumn(self: anytype, name: []const u8, comptime T: type, lower: T, upper: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return filterBetweenColumn(self, name, T, lower, upper);
}

pub fn dropRowsByColumnMask(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return expr_mod.dropRowsByColumnMask(FrameType(@TypeOf(self)), frameValue(self), name);
}

pub fn whereIndicesColumn(self: anytype, mask_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.whereIndicesColumn(FrameType(@TypeOf(self)), frameValue(self), mask_name, output_name);
}

pub const argwhereColumn = whereIndicesColumn;

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

pub fn selectByNameGlob(self: anytype, pattern: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectByNameGlob(FrameType(@TypeOf(self)), frameValue(self), pattern);
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

pub fn dropByNameGlob(self: anytype, pattern: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropByNameGlob(FrameType(@TypeOf(self)), frameValue(self), pattern);
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

pub fn selectColumnsWithNaNs(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectColumnsWithNaNs(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn selectColumnsWithoutNaNs(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectColumnsWithoutNaNs(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn dropColumnsWithNaNs(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropColumnsWithNaNs(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn dropColumnsWithoutNaNs(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropColumnsWithoutNaNs(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn selectColumnsWithInfs(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectColumnsWithInfs(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn selectColumnsWithoutInfs(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectColumnsWithoutInfs(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn dropColumnsWithInfs(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropColumnsWithInfs(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn dropColumnsWithoutInfs(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropColumnsWithoutInfs(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn selectColumnsWithPositiveInfs(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectColumnsWithPositiveInfs(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn selectColumnsWithoutPositiveInfs(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectColumnsWithoutPositiveInfs(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn dropColumnsWithPositiveInfs(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropColumnsWithPositiveInfs(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn dropColumnsWithoutPositiveInfs(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropColumnsWithoutPositiveInfs(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn selectColumnsWithNegativeInfs(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectColumnsWithNegativeInfs(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn selectColumnsWithoutNegativeInfs(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectColumnsWithoutNegativeInfs(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn dropColumnsWithNegativeInfs(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropColumnsWithNegativeInfs(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn dropColumnsWithoutNegativeInfs(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropColumnsWithoutNegativeInfs(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn selectColumnsWithZeros(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectColumnsWithZeros(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn selectColumnsWithoutZeros(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectColumnsWithoutZeros(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn dropColumnsWithZeros(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropColumnsWithZeros(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn dropColumnsWithoutZeros(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropColumnsWithoutZeros(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn selectColumnsWithPositiveZeros(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectColumnsWithPositiveZeros(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn selectColumnsWithoutPositiveZeros(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectColumnsWithoutPositiveZeros(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn dropColumnsWithPositiveZeros(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropColumnsWithPositiveZeros(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn dropColumnsWithoutPositiveZeros(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropColumnsWithoutPositiveZeros(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn selectColumnsWithNegativeZeros(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectColumnsWithNegativeZeros(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn selectColumnsWithoutNegativeZeros(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectColumnsWithoutNegativeZeros(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn dropColumnsWithNegativeZeros(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropColumnsWithNegativeZeros(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn dropColumnsWithoutNegativeZeros(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropColumnsWithoutNegativeZeros(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn selectColumnsWithNonZeros(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectColumnsWithNonZeros(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn selectColumnsWithoutNonZeros(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectColumnsWithoutNonZeros(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn dropColumnsWithNonZeros(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropColumnsWithNonZeros(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn dropColumnsWithoutNonZeros(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropColumnsWithoutNonZeros(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn selectColumnsWithPositives(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectColumnsWithPositives(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn selectColumnsWithoutPositives(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectColumnsWithoutPositives(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn dropColumnsWithPositives(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropColumnsWithPositives(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn dropColumnsWithoutPositives(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropColumnsWithoutPositives(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn selectColumnsWithSignBits(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectColumnsWithSignBits(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn selectColumnsWithoutSignBits(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectColumnsWithoutSignBits(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn dropColumnsWithSignBits(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropColumnsWithSignBits(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn dropColumnsWithoutSignBits(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropColumnsWithoutSignBits(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn selectColumnsWithNegatives(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectColumnsWithNegatives(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn selectColumnsWithoutNegatives(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectColumnsWithoutNegatives(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn dropColumnsWithNegatives(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropColumnsWithNegatives(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn dropColumnsWithoutNegatives(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropColumnsWithoutNegatives(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn selectColumnsWithFinites(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectColumnsWithFinites(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn selectColumnsWithoutFinites(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectColumnsWithoutFinites(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn dropColumnsWithFinites(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropColumnsWithFinites(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn dropColumnsWithoutFinites(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropColumnsWithoutFinites(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn selectColumnsWithNormals(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectColumnsWithNormals(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn selectColumnsWithoutNormals(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectColumnsWithoutNormals(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn dropColumnsWithNormals(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropColumnsWithNormals(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn dropColumnsWithoutNormals(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropColumnsWithoutNormals(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn selectColumnsWithSubnormals(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectColumnsWithSubnormals(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn selectColumnsWithoutSubnormals(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectColumnsWithoutSubnormals(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn dropColumnsWithSubnormals(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropColumnsWithSubnormals(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn dropColumnsWithoutSubnormals(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropColumnsWithoutSubnormals(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn selectColumnsWithNonFinites(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectColumnsWithNonFinites(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn selectColumnsWithoutNonFinites(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.selectColumnsWithoutNonFinites(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn dropColumnsWithNonFinites(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropColumnsWithNonFinites(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn dropColumnsWithoutNonFinites(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropColumnsWithoutNonFinites(FrameType(@TypeOf(self)), frameValue(self));
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

pub fn withColumnFillNull(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnFillNull(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name, DeviceScalar.init(T, value));
}

pub fn withColumnFillNullScalar(self: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnFillNullScalar(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name, scalar);
}

pub fn fillNullForwardColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.fillNullForwardColumn(FrameType(@TypeOf(self)), frameValue(self), name);
}

pub fn fillNullBackwardColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.fillNullBackwardColumn(FrameType(@TypeOf(self)), frameValue(self), name);
}

pub fn withColumnFillNullForward(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnFillNullForward(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name);
}

pub fn withColumnFillNullBackward(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnFillNullBackward(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name);
}

pub fn nullIfColumn(self: anytype, name: []const u8, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.nullIfColumn(FrameType(@TypeOf(self)), frameValue(self), name, DeviceScalar.init(T, value));
}

pub fn nullIfColumnScalar(self: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.nullIfColumnScalar(FrameType(@TypeOf(self)), frameValue(self), name, scalar);
}

pub fn nullIfValuesColumnWithDeviceColumn(self: anytype, name: []const u8, test_values: @TypeOf(frameValue(self).columns[0])) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.nullIfValuesColumnWithDeviceColumn(FrameType(@TypeOf(self)), frameValue(self), name, test_values);
}

pub fn nullIfValuesColumn(self: anytype, name: []const u8, comptime T: type, values: []const T) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.nullIfValuesColumn(FrameType(@TypeOf(self)), frameValue(self), name, T, values);
}

pub fn withColumnNullIf(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnNullIf(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name, DeviceScalar.init(T, value));
}

pub fn withColumnNullIfScalar(self: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnNullIfScalar(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name, scalar);
}

pub fn withColumnNullIfValuesWithDeviceColumn(self: anytype, output_name: []const u8, input_name: []const u8, test_values: @TypeOf(frameValue(self).columns[0])) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnNullIfValuesWithDeviceColumn(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name, test_values);
}

pub fn withColumnNullIfValues(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, values: []const T) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnNullIfValues(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name, T, values);
}

pub fn nullIfNaNColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.nullIfNaNColumn(FrameType(@TypeOf(self)), frameValue(self), name);
}

pub fn withColumnNullIfNaN(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnNullIfNaN(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name);
}

pub fn nullIfInfColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.nullIfInfColumn(FrameType(@TypeOf(self)), frameValue(self), name);
}

pub fn withColumnNullIfInf(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnNullIfInf(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name);
}

pub fn nullIfPositiveInfColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.nullIfPositiveInfColumn(FrameType(@TypeOf(self)), frameValue(self), name);
}

pub fn withColumnNullIfPositiveInf(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnNullIfPositiveInf(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name);
}

pub fn nullIfNegativeInfColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.nullIfNegativeInfColumn(FrameType(@TypeOf(self)), frameValue(self), name);
}

pub fn withColumnNullIfNegativeInf(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnNullIfNegativeInf(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name);
}

pub fn nullIfZeroColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.nullIfZeroColumn(FrameType(@TypeOf(self)), frameValue(self), name);
}

pub fn withColumnNullIfZero(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnNullIfZero(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name);
}

pub fn nullIfPositiveZeroColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.nullIfPositiveZeroColumn(FrameType(@TypeOf(self)), frameValue(self), name);
}

pub fn withColumnNullIfPositiveZero(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnNullIfPositiveZero(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name);
}

pub fn nullIfNegativeZeroColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.nullIfNegativeZeroColumn(FrameType(@TypeOf(self)), frameValue(self), name);
}

pub fn withColumnNullIfNegativeZero(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnNullIfNegativeZero(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name);
}

pub fn nullIfNonZeroColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.nullIfNonZeroColumn(FrameType(@TypeOf(self)), frameValue(self), name);
}

pub fn withColumnNullIfNonZero(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnNullIfNonZero(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name);
}

pub fn nullIfPositiveColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.nullIfPositiveColumn(FrameType(@TypeOf(self)), frameValue(self), name);
}

pub fn withColumnNullIfPositive(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnNullIfPositive(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name);
}

pub fn nullIfSignBitColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.nullIfSignBitColumn(FrameType(@TypeOf(self)), frameValue(self), name);
}

pub fn withColumnNullIfSignBit(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnNullIfSignBit(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name);
}

pub fn nullIfNegativeColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.nullIfNegativeColumn(FrameType(@TypeOf(self)), frameValue(self), name);
}

pub fn withColumnNullIfNegative(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnNullIfNegative(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name);
}

pub fn nullIfFiniteColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.nullIfFiniteColumn(FrameType(@TypeOf(self)), frameValue(self), name);
}

pub fn withColumnNullIfFinite(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnNullIfFinite(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name);
}

pub fn nullIfNormalColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.nullIfNormalColumn(FrameType(@TypeOf(self)), frameValue(self), name);
}

pub fn withColumnNullIfNormal(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnNullIfNormal(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name);
}

pub fn nullIfSubnormalColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.nullIfSubnormalColumn(FrameType(@TypeOf(self)), frameValue(self), name);
}

pub fn withColumnNullIfSubnormal(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnNullIfSubnormal(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name);
}

pub fn nullIfNonFiniteColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.nullIfNonFiniteColumn(FrameType(@TypeOf(self)), frameValue(self), name);
}

pub fn withColumnNullIfNonFinite(self: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnNullIfNonFinite(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name);
}

pub fn withColumnFillNaN(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnFillNaN(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name, DeviceScalar.init(T, value));
}

pub fn withColumnFillNaNScalar(self: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnFillNaNScalar(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name, scalar);
}

pub fn withColumnFillInf(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnFillInf(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name, DeviceScalar.init(T, value));
}

pub fn withColumnFillInfScalar(self: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnFillInfScalar(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name, scalar);
}

pub fn withColumnFillPositiveInf(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnFillPositiveInf(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name, DeviceScalar.init(T, value));
}

pub fn withColumnFillPositiveInfScalar(self: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnFillPositiveInfScalar(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name, scalar);
}

pub fn withColumnFillNegativeInf(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnFillNegativeInf(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name, DeviceScalar.init(T, value));
}

pub fn withColumnFillNegativeInfScalar(self: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnFillNegativeInfScalar(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name, scalar);
}

pub fn withColumnFillZero(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnFillZero(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name, DeviceScalar.init(T, value));
}

pub fn withColumnFillZeroScalar(self: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnFillZeroScalar(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name, scalar);
}

pub fn withColumnFillPositiveZero(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnFillPositiveZero(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name, DeviceScalar.init(T, value));
}

pub fn withColumnFillPositiveZeroScalar(self: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnFillPositiveZeroScalar(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name, scalar);
}

pub fn withColumnFillNegativeZero(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnFillNegativeZero(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name, DeviceScalar.init(T, value));
}

pub fn withColumnFillNegativeZeroScalar(self: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnFillNegativeZeroScalar(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name, scalar);
}

pub fn withColumnFillNonZero(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnFillNonZero(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name, DeviceScalar.init(T, value));
}

pub fn withColumnFillNonZeroScalar(self: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnFillNonZeroScalar(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name, scalar);
}

pub fn withColumnFillPositive(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnFillPositive(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name, DeviceScalar.init(T, value));
}

pub fn withColumnFillPositiveScalar(self: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnFillPositiveScalar(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name, scalar);
}

pub fn withColumnFillSignBit(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnFillSignBit(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name, DeviceScalar.init(T, value));
}

pub fn withColumnFillSignBitScalar(self: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnFillSignBitScalar(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name, scalar);
}

pub fn withColumnFillNegative(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnFillNegative(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name, DeviceScalar.init(T, value));
}

pub fn withColumnFillNegativeScalar(self: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnFillNegativeScalar(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name, scalar);
}

pub fn withColumnFillFinite(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnFillFinite(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name, DeviceScalar.init(T, value));
}

pub fn withColumnFillFiniteScalar(self: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnFillFiniteScalar(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name, scalar);
}

pub fn withColumnFillNormal(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnFillNormal(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name, DeviceScalar.init(T, value));
}

pub fn withColumnFillNormalScalar(self: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnFillNormalScalar(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name, scalar);
}

pub fn withColumnFillSubnormal(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnFillSubnormal(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name, DeviceScalar.init(T, value));
}

pub fn withColumnFillSubnormalScalar(self: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnFillSubnormalScalar(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name, scalar);
}

pub fn withColumnFillNonFinite(self: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnFillNonFinite(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name, DeviceScalar.init(T, value));
}

pub fn withColumnFillNonFiniteScalar(self: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withColumnFillNonFiniteScalar(FrameType(@TypeOf(self)), frameValue(self), output_name, input_name, scalar);
}

pub fn fillNaNColumn(self: anytype, name: []const u8, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.fillNaNColumn(FrameType(@TypeOf(self)), frameValue(self), name, DeviceScalar.init(T, value));
}

pub fn fillNaNColumnWithScalar(self: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.fillNaNColumn(FrameType(@TypeOf(self)), frameValue(self), name, scalar);
}

pub fn fillInfColumn(self: anytype, name: []const u8, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.fillInfColumn(FrameType(@TypeOf(self)), frameValue(self), name, DeviceScalar.init(T, value));
}

pub fn fillInfColumnWithScalar(self: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.fillInfColumn(FrameType(@TypeOf(self)), frameValue(self), name, scalar);
}

pub fn fillPositiveInfColumn(self: anytype, name: []const u8, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.fillPositiveInfColumn(FrameType(@TypeOf(self)), frameValue(self), name, DeviceScalar.init(T, value));
}

pub fn fillPositiveInfColumnWithScalar(self: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.fillPositiveInfColumn(FrameType(@TypeOf(self)), frameValue(self), name, scalar);
}

pub fn fillNegativeInfColumn(self: anytype, name: []const u8, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.fillNegativeInfColumn(FrameType(@TypeOf(self)), frameValue(self), name, DeviceScalar.init(T, value));
}

pub fn fillNegativeInfColumnWithScalar(self: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.fillNegativeInfColumn(FrameType(@TypeOf(self)), frameValue(self), name, scalar);
}

pub fn fillZeroColumn(self: anytype, name: []const u8, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.fillZeroColumn(FrameType(@TypeOf(self)), frameValue(self), name, DeviceScalar.init(T, value));
}

pub fn fillZeroColumnWithScalar(self: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.fillZeroColumn(FrameType(@TypeOf(self)), frameValue(self), name, scalar);
}

pub fn fillPositiveZeroColumn(self: anytype, name: []const u8, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.fillPositiveZeroColumn(FrameType(@TypeOf(self)), frameValue(self), name, DeviceScalar.init(T, value));
}

pub fn fillPositiveZeroColumnWithScalar(self: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.fillPositiveZeroColumn(FrameType(@TypeOf(self)), frameValue(self), name, scalar);
}

pub fn fillNegativeZeroColumn(self: anytype, name: []const u8, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.fillNegativeZeroColumn(FrameType(@TypeOf(self)), frameValue(self), name, DeviceScalar.init(T, value));
}

pub fn fillNegativeZeroColumnWithScalar(self: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.fillNegativeZeroColumn(FrameType(@TypeOf(self)), frameValue(self), name, scalar);
}

pub fn fillNonZeroColumn(self: anytype, name: []const u8, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.fillNonZeroColumn(FrameType(@TypeOf(self)), frameValue(self), name, DeviceScalar.init(T, value));
}

pub fn fillNonZeroColumnWithScalar(self: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.fillNonZeroColumn(FrameType(@TypeOf(self)), frameValue(self), name, scalar);
}

pub fn fillPositiveColumn(self: anytype, name: []const u8, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.fillPositiveColumn(FrameType(@TypeOf(self)), frameValue(self), name, DeviceScalar.init(T, value));
}

pub fn fillPositiveColumnWithScalar(self: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.fillPositiveColumn(FrameType(@TypeOf(self)), frameValue(self), name, scalar);
}

pub fn fillSignBitColumn(self: anytype, name: []const u8, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.fillSignBitColumn(FrameType(@TypeOf(self)), frameValue(self), name, DeviceScalar.init(T, value));
}

pub fn fillSignBitColumnWithScalar(self: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.fillSignBitColumn(FrameType(@TypeOf(self)), frameValue(self), name, scalar);
}

pub fn fillNegativeColumn(self: anytype, name: []const u8, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.fillNegativeColumn(FrameType(@TypeOf(self)), frameValue(self), name, DeviceScalar.init(T, value));
}

pub fn fillNegativeColumnWithScalar(self: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.fillNegativeColumn(FrameType(@TypeOf(self)), frameValue(self), name, scalar);
}

pub fn fillFiniteColumn(self: anytype, name: []const u8, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.fillFiniteColumn(FrameType(@TypeOf(self)), frameValue(self), name, DeviceScalar.init(T, value));
}

pub fn fillFiniteColumnWithScalar(self: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.fillFiniteColumn(FrameType(@TypeOf(self)), frameValue(self), name, scalar);
}

pub fn fillNormalColumn(self: anytype, name: []const u8, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.fillNormalColumn(FrameType(@TypeOf(self)), frameValue(self), name, DeviceScalar.init(T, value));
}

pub fn fillNormalColumnWithScalar(self: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.fillNormalColumn(FrameType(@TypeOf(self)), frameValue(self), name, scalar);
}

pub fn fillSubnormalColumn(self: anytype, name: []const u8, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.fillSubnormalColumn(FrameType(@TypeOf(self)), frameValue(self), name, DeviceScalar.init(T, value));
}

pub fn fillSubnormalColumnWithScalar(self: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.fillSubnormalColumn(FrameType(@TypeOf(self)), frameValue(self), name, scalar);
}

pub fn fillNonFiniteColumn(self: anytype, name: []const u8, comptime T: type, value: T) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.fillNonFiniteColumn(FrameType(@TypeOf(self)), frameValue(self), name, DeviceScalar.init(T, value));
}

pub fn fillNonFiniteColumnWithScalar(self: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.fillNonFiniteColumn(FrameType(@TypeOf(self)), frameValue(self), name, scalar);
}

pub fn coalesceColumns(self: anytype, primary_name: []const u8, fallback_name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.coalesceColumns(FrameType(@TypeOf(self)), frameValue(self), primary_name, fallback_name, output_name);
}

pub fn coalesceColumnsMany(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.coalesceColumnsMany(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn coalesceManyColumns(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.coalesceManyColumns(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn coalesceFirstValidColumns(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.coalesceFirstValidColumns(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
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

pub fn isZeroColumn(self: anytype, name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.isZeroColumn(FrameType(@TypeOf(self)), frameValue(self), name, output_name);
}

pub fn isPositiveZeroColumn(self: anytype, name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.isPositiveZeroColumn(FrameType(@TypeOf(self)), frameValue(self), name, output_name);
}

pub fn isNegativeZeroColumn(self: anytype, name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.isNegativeZeroColumn(FrameType(@TypeOf(self)), frameValue(self), name, output_name);
}

pub fn isNonZeroColumn(self: anytype, name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.isNonZeroColumn(FrameType(@TypeOf(self)), frameValue(self), name, output_name);
}

pub fn isPositiveColumn(self: anytype, name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.isPositiveColumn(FrameType(@TypeOf(self)), frameValue(self), name, output_name);
}

pub fn isSignBitColumn(self: anytype, name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.isSignBitColumn(FrameType(@TypeOf(self)), frameValue(self), name, output_name);
}

pub fn isNegativeColumn(self: anytype, name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.isNegativeColumn(FrameType(@TypeOf(self)), frameValue(self), name, output_name);
}

pub fn isFiniteColumn(self: anytype, name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.isFiniteColumn(FrameType(@TypeOf(self)), frameValue(self), name, output_name);
}

pub fn isNormalColumn(self: anytype, name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.isNormalColumn(FrameType(@TypeOf(self)), frameValue(self), name, output_name);
}

pub fn isSubnormalColumn(self: anytype, name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.isSubnormalColumn(FrameType(@TypeOf(self)), frameValue(self), name, output_name);
}

pub fn isNonFiniteColumn(self: anytype, name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.isNonFiniteColumn(FrameType(@TypeOf(self)), frameValue(self), name, output_name);
}

pub fn isInfColumn(self: anytype, name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.isInfColumn(FrameType(@TypeOf(self)), frameValue(self), name, output_name);
}

pub fn isPositiveInfColumn(self: anytype, name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.isPositiveInfColumn(FrameType(@TypeOf(self)), frameValue(self), name, output_name);
}

pub fn isNegativeInfColumn(self: anytype, name: []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.isNegativeInfColumn(FrameType(@TypeOf(self)), frameValue(self), name, output_name);
}

pub fn withRowNullCount(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowNullCount(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowValidCount(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowValidCount(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAnyNull(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAnyNull(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAllNull(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAllNull(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAnyValid(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAnyValid(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAllValid(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAllValid(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowCumulativeAnyNull(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeAnyNull(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumAnyNull(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumAnyNull(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixAnyNull(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixAnyNull(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeAllNull(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeAllNull(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumAllNull(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumAllNull(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixAllNull(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixAllNull(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeAnyValid(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeAnyValid(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumAnyValid(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumAnyValid(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixAnyValid(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixAnyValid(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeAllValid(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeAllValid(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumAllValid(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumAllValid(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixAllValid(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixAllValid(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeNullCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeNullCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumNullCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumNullCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixNullCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixNullCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeValidCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeValidCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumValidCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumValidCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixValidCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixValidCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeNullRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeNullRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumNullRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumNullRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixNullRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixNullRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeValidRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeValidRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumValidRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumValidRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixValidRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixValidRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowNullRatio(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowNullRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowValidRatio(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowValidRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowPairCount(self: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPairCount(FrameType(@TypeOf(self)), frameValue(self), lhs_names, rhs_names, output_name);
}

pub fn withRowFirstValidIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowFirstValidIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowLastValidIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowLastValidIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowFirstNullIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowFirstNullIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowLastNullIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowLastNullIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowCumulativeFirstValidIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeFirstValidIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixFirstValidIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixFirstValidIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeLastValidIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeLastValidIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixLastValidIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixLastValidIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeFirstNullIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeFirstNullIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixFirstNullIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixFirstNullIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeLastNullIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeLastNullIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixLastNullIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixLastNullIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowWeightedMean(self: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowWeightedMean(FrameType(@TypeOf(self)), frameValue(self), value_names, weight_names, output_name);
}

pub fn withRowWeightedWeightSum(self: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowWeightedWeightSum(FrameType(@TypeOf(self)), frameValue(self), value_names, weight_names, output_name);
}

pub fn withRowWeightedPositiveCount(self: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowWeightedPositiveCount(FrameType(@TypeOf(self)), frameValue(self), value_names, weight_names, output_name);
}

pub fn withRowWeightedEffectiveN(self: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowWeightedEffectiveN(FrameType(@TypeOf(self)), frameValue(self), value_names, weight_names, output_name);
}

pub const withRowWeightedEffectiveCount = withRowWeightedEffectiveN;

pub fn withRowWeightedMeanSquare(self: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowWeightedMeanSquare(FrameType(@TypeOf(self)), frameValue(self), value_names, weight_names, output_name);
}

pub fn withRowWeightedRms(self: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowWeightedRms(FrameType(@TypeOf(self)), frameValue(self), value_names, weight_names, output_name);
}

pub fn withRowWeightedMeanAbs(self: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowWeightedMeanAbs(FrameType(@TypeOf(self)), frameValue(self), value_names, weight_names, output_name);
}

pub fn withRowWeightedL1Norm(self: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowWeightedL1Norm(FrameType(@TypeOf(self)), frameValue(self), value_names, weight_names, output_name);
}

pub fn withRowWeightedL2Norm(self: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowWeightedL2Norm(FrameType(@TypeOf(self)), frameValue(self), value_names, weight_names, output_name);
}

pub const withRowWeightedMeanSquared = withRowWeightedMeanSquare;
pub const withRowWeightedMeanSq = withRowWeightedMeanSquare;
pub const withRowWeightedRMS = withRowWeightedRms;
pub const withRowWeightedL1 = withRowWeightedL1Norm;
pub const withRowWeightedL2 = withRowWeightedL2Norm;

pub fn withRowWeightedVariance(self: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowWeightedVariance(FrameType(@TypeOf(self)), frameValue(self), value_names, weight_names, output_name, correction);
}

pub fn withRowWeightedVar(self: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return withRowWeightedVariance(self, value_names, weight_names, output_name, correction);
}

pub fn withRowWeightedStddev(self: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowWeightedStddev(FrameType(@TypeOf(self)), frameValue(self), value_names, weight_names, output_name, correction);
}

pub fn withRowWeightedStd(self: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return withRowWeightedStddev(self, value_names, weight_names, output_name, correction);
}

pub fn withRowWeightedCovariance(self: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowWeightedCovariance(FrameType(@TypeOf(self)), frameValue(self), lhs_names, rhs_names, weight_names, output_name, correction);
}

pub fn withRowWeightedCorrelation(self: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowWeightedCorrelation(FrameType(@TypeOf(self)), frameValue(self), lhs_names, rhs_names, weight_names, output_name, correction);
}

pub fn withRowWeightedBeta(self: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowWeightedBeta(FrameType(@TypeOf(self)), frameValue(self), lhs_names, rhs_names, weight_names, output_name, correction);
}

pub fn withRowWeightedQuantile(self: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, q: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowWeightedQuantile(FrameType(@TypeOf(self)), frameValue(self), value_names, weight_names, output_name, q);
}

pub fn withRowWeightedMedian(self: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowWeightedMedian(FrameType(@TypeOf(self)), frameValue(self), value_names, weight_names, output_name);
}

pub fn withRowWeightedIqr(self: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowWeightedIqr(FrameType(@TypeOf(self)), frameValue(self), value_names, weight_names, output_name);
}

pub fn withRowWeightedMad(self: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowWeightedMad(FrameType(@TypeOf(self)), frameValue(self), value_names, weight_names, output_name);
}

pub fn withRowWeightedTrimmedMean(self: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, trim_fraction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowWeightedTrimmedMean(FrameType(@TypeOf(self)), frameValue(self), value_names, weight_names, output_name, trim_fraction);
}

pub fn withRowWeightedWinsorizedMean(self: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, winsor_fraction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowWeightedWinsorizedMean(FrameType(@TypeOf(self)), frameValue(self), value_names, weight_names, output_name, winsor_fraction);
}

pub fn withRowWeightedInterdecileRange(self: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowWeightedInterdecileRange(FrameType(@TypeOf(self)), frameValue(self), value_names, weight_names, output_name);
}

pub fn withRowWeightedMidhinge(self: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowWeightedMidhinge(FrameType(@TypeOf(self)), frameValue(self), value_names, weight_names, output_name);
}

pub fn withRowWeightedTrimean(self: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowWeightedTrimean(FrameType(@TypeOf(self)), frameValue(self), value_names, weight_names, output_name);
}

pub fn withRowWeightedBowleySkewness(self: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowWeightedBowleySkewness(FrameType(@TypeOf(self)), frameValue(self), value_names, weight_names, output_name);
}

pub fn withRowWeightedQuartileCoeffDispersion(self: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowWeightedQuartileCoeffDispersion(FrameType(@TypeOf(self)), frameValue(self), value_names, weight_names, output_name);
}

pub fn withRowWeightedKelleySkewness(self: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowWeightedKelleySkewness(FrameType(@TypeOf(self)), frameValue(self), value_names, weight_names, output_name);
}

pub const withRowWeightedIdr = withRowWeightedInterdecileRange;
pub const withRowWeightedIDR = withRowWeightedInterdecileRange;
pub const withRowWeightedBowleySkew = withRowWeightedBowleySkewness;
pub const withRowWeightedQcd = withRowWeightedQuartileCoeffDispersion;
pub const withRowWeightedQCD = withRowWeightedQuartileCoeffDispersion;
pub const withRowWeightedKelleySkew = withRowWeightedKelleySkewness;

pub fn withRowWeightedMode(self: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowWeightedMode(FrameType(@TypeOf(self)), frameValue(self), value_names, weight_names, output_name);
}

pub fn withRowWeightedModeWeight(self: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowWeightedModeWeight(FrameType(@TypeOf(self)), frameValue(self), value_names, weight_names, output_name);
}

pub fn withRowWeightedModeRatio(self: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowWeightedModeRatio(FrameType(@TypeOf(self)), frameValue(self), value_names, weight_names, output_name);
}

pub fn withRowWeightedModeMargin(self: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowWeightedModeMargin(FrameType(@TypeOf(self)), frameValue(self), value_names, weight_names, output_name);
}

pub fn withRowWeightedModeMarginRatio(self: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowWeightedModeMarginRatio(FrameType(@TypeOf(self)), frameValue(self), value_names, weight_names, output_name);
}

pub fn withRowWeightedEntropy(self: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowWeightedEntropy(FrameType(@TypeOf(self)), frameValue(self), value_names, weight_names, output_name);
}

pub fn withRowWeightedGiniImpurity(self: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowWeightedGiniImpurity(FrameType(@TypeOf(self)), frameValue(self), value_names, weight_names, output_name);
}

pub fn withRowWeightedPerplexity(self: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowWeightedPerplexity(FrameType(@TypeOf(self)), frameValue(self), value_names, weight_names, output_name);
}

pub fn withRowWeightedInverseSimpson(self: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowWeightedInverseSimpson(FrameType(@TypeOf(self)), frameValue(self), value_names, weight_names, output_name);
}

pub fn withRowWeightedSimpsonConcentration(self: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowWeightedSimpsonConcentration(FrameType(@TypeOf(self)), frameValue(self), value_names, weight_names, output_name);
}

pub fn withRowWeightedEvenness(self: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowWeightedEvenness(FrameType(@TypeOf(self)), frameValue(self), value_names, weight_names, output_name);
}

pub fn withRowWeightedMeanAbsDev(self: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowWeightedMeanAbsDev(FrameType(@TypeOf(self)), frameValue(self), value_names, weight_names, output_name);
}

pub fn withRowWeightedMeanAbsDevRatio(self: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowWeightedMeanAbsDevRatio(FrameType(@TypeOf(self)), frameValue(self), value_names, weight_names, output_name);
}

pub fn withRowWeightedGiniMeanDiff(self: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowWeightedGiniMeanDiff(FrameType(@TypeOf(self)), frameValue(self), value_names, weight_names, output_name);
}

pub fn withRowWeightedGiniCoefficient(self: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowWeightedGiniCoefficient(FrameType(@TypeOf(self)), frameValue(self), value_names, weight_names, output_name);
}

pub const withRowWeightedMeanAbsoluteDeviation = withRowWeightedMeanAbsDev;
pub const withRowWeightedMadRatio = withRowWeightedMeanAbsDevRatio;
pub const withRowWeightedGiniCoeff = withRowWeightedGiniCoefficient;

pub fn withRowDot(self: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowDot(FrameType(@TypeOf(self)), frameValue(self), lhs_names, rhs_names, output_name);
}

pub fn withRowCosineSimilarity(self: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCosineSimilarity(FrameType(@TypeOf(self)), frameValue(self), lhs_names, rhs_names, output_name);
}

pub fn withRowCosine(self: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCosine(FrameType(@TypeOf(self)), frameValue(self), lhs_names, rhs_names, output_name);
}

pub fn withRowSquaredEuclideanDistance(self: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowSquaredEuclideanDistance(FrameType(@TypeOf(self)), frameValue(self), lhs_names, rhs_names, output_name);
}

pub fn withRowEuclideanDistance(self: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowEuclideanDistance(FrameType(@TypeOf(self)), frameValue(self), lhs_names, rhs_names, output_name);
}

pub fn withRowManhattanDistance(self: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowManhattanDistance(FrameType(@TypeOf(self)), frameValue(self), lhs_names, rhs_names, output_name);
}

pub fn withRowChebyshevDistance(self: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowChebyshevDistance(FrameType(@TypeOf(self)), frameValue(self), lhs_names, rhs_names, output_name);
}

pub fn withRowCanberraDistance(self: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCanberraDistance(FrameType(@TypeOf(self)), frameValue(self), lhs_names, rhs_names, output_name);
}

pub fn withRowBrayCurtisDistance(self: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowBrayCurtisDistance(FrameType(@TypeOf(self)), frameValue(self), lhs_names, rhs_names, output_name);
}

pub fn withRowMeanError(self: anytype, actual_names: []const []const u8, predicted_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMeanError(FrameType(@TypeOf(self)), frameValue(self), actual_names, predicted_names, output_name);
}

pub fn withRowBias(self: anytype, actual_names: []const []const u8, predicted_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowBias(FrameType(@TypeOf(self)), frameValue(self), actual_names, predicted_names, output_name);
}

pub fn withRowMae(self: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMae(FrameType(@TypeOf(self)), frameValue(self), lhs_names, rhs_names, output_name);
}

pub fn withRowMse(self: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMse(FrameType(@TypeOf(self)), frameValue(self), lhs_names, rhs_names, output_name);
}

pub fn withRowRmse(self: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowRmse(FrameType(@TypeOf(self)), frameValue(self), lhs_names, rhs_names, output_name);
}

pub fn withRowMape(self: anytype, actual_names: []const []const u8, predicted_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMape(FrameType(@TypeOf(self)), frameValue(self), actual_names, predicted_names, output_name);
}

pub fn withRowSmape(self: anytype, actual_names: []const []const u8, predicted_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowSmape(FrameType(@TypeOf(self)), frameValue(self), actual_names, predicted_names, output_name);
}

pub fn withRowCovariance(self: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCovariance(FrameType(@TypeOf(self)), frameValue(self), lhs_names, rhs_names, output_name);
}

pub fn withRowCorrelation(self: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCorrelation(FrameType(@TypeOf(self)), frameValue(self), lhs_names, rhs_names, output_name);
}

pub fn withRowBeta(self: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowBeta(FrameType(@TypeOf(self)), frameValue(self), lhs_names, rhs_names, output_name);
}

pub fn withRowArgMin(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowArgMin(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowArgMax(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowArgMax(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowCumulativeArgMin(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeArgMin(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumArgMin(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumArgMin(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixArgMin(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixArgMin(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeArgMax(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeArgMax(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumArgMax(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumArgMax(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixArgMax(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixArgMax(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowQuantile(self: anytype, names: []const []const u8, output_name: []const u8, q: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowQuantile(FrameType(@TypeOf(self)), frameValue(self), names, output_name, q);
}

pub fn withRowQuantileRange(self: anytype, names: []const []const u8, output_name: []const u8, low_q: f64, high_q: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowQuantileRange(FrameType(@TypeOf(self)), frameValue(self), names, output_name, low_q, high_q);
}

pub fn withRowTrimmedMean(self: anytype, names: []const []const u8, output_name: []const u8, trim_fraction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowTrimmedMean(FrameType(@TypeOf(self)), frameValue(self), names, output_name, trim_fraction);
}

pub fn withRowWinsorizedMean(self: anytype, names: []const []const u8, output_name: []const u8, winsor_fraction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowWinsorizedMean(FrameType(@TypeOf(self)), frameValue(self), names, output_name, winsor_fraction);
}

pub fn withRowMedian(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMedian(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowIqr(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowIqr(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowInterdecileRange(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowInterdecileRange(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowIdr(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowIdr(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowMidhinge(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMidhinge(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowTrimean(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowTrimean(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowBowleySkewness(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowBowleySkewness(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowBowleySkew(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowBowleySkew(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowQuartileCoeffDispersion(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowQuartileCoeffDispersion(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowQcd(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowQcd(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowKelleySkewness(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowKelleySkewness(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowKelleySkew(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowKelleySkew(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowMad(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMad(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowMedianAbsDev(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMedianAbsDev(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowMode(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMode(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowCumulativeMode(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeMode(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumMode(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumMode(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixMode(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixMode(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeModeCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeModeCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumModeCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumModeCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixModeCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixModeCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeModeRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeModeRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumModeRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumModeRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixModeRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixModeRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeModeMargin(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeModeMargin(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumModeMargin(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumModeMargin(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixModeMargin(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixModeMargin(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeModeMarginRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeModeMarginRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumModeMarginRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumModeMarginRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixModeMarginRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixModeMarginRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowEntropy(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowEntropy(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowGiniImpurity(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowGiniImpurity(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowPerplexity(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPerplexity(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowInverseSimpson(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowInverseSimpson(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowSimpsonConcentration(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowSimpsonConcentration(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowEvenness(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowEvenness(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowModeCount(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowModeCount(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowModeRatio(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowModeRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowModeMargin(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowModeMargin(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowModeMarginRatio(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowModeMarginRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowCountDistinct(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCountDistinct(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowNUnique(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowNUnique(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowCumulativeDistinctCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeDistinctCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumDistinctCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumDistinctCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixDistinctCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixDistinctCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeNUnique(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeNUnique(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixNUnique(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixNUnique(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowSum(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowSum(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowMean(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMean(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowLogSumExp(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowLogSumExp(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowLogsumexp(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowLogsumexp(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowLogMeanExp(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowLogMeanExp(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowLogmeanexp(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowLogmeanexp(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowCentered(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCentered(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowDemean(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowDemean(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowZScore(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowZScore(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowZscore(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowZscore(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowStandardize(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowStandardize(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowRobustZScore(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowRobustZScore(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowRobustZscore(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowRobustZscore(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowMadZScore(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMadZScore(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowMadZscore(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMadZscore(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowAverageRank(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAverageRank(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowAverageRanks(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAverageRanks(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowAvgRank(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAvgRank(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowAvgRanks(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAvgRanks(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowFractionalRank(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowFractionalRank(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowFractionalRanks(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowFractionalRanks(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowOrdinalRank(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowOrdinalRank(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowOrdinalRanks(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowOrdinalRanks(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowDenseRank(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowDenseRank(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowDenseRanks(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowDenseRanks(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCompetitionRank(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCompetitionRank(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCompetitionRanks(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCompetitionRanks(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowMinRank(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMinRank(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowMinRanks(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMinRanks(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPercentRank(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPercentRank(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPercentRanks(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPercentRanks(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPercentileRank(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPercentileRank(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPercentileRanks(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPercentileRanks(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumeDist(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumeDist(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumeDistribution(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumeDistribution(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeDistribution(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeDistribution(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeSum(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeSum(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumsum(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumsum(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumSum(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumSum(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixSum(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixSum(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeMean(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeMean(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCummean(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCummean(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumMean(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumMean(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixMean(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixMean(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeAverage(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeAverage(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumAverage(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumAverage(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumAvg(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumAvg(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixAverage(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixAverage(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixAvg(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixAvg(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeLogSumExp(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeLogSumExp(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeLogsumexp(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeLogsumexp(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumLogSumExp(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumLogSumExp(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumLogsumexp(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumLogsumexp(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixLogSumExp(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixLogSumExp(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixLogsumexp(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixLogsumexp(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeLogMeanExp(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeLogMeanExp(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeLogmeanexp(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeLogmeanexp(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumLogMeanExp(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumLogMeanExp(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumLogmeanexp(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumLogmeanexp(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixLogMeanExp(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixLogMeanExp(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixLogmeanexp(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixLogmeanexp(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeGeometricMean(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeGeometricMean(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeGeoMean(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeGeoMean(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumGeometricMean(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumGeometricMean(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumGeoMean(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumGeoMean(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixGeometricMean(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixGeometricMean(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixGeoMean(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixGeoMean(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeHarmonicMean(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeHarmonicMean(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeHarmMean(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeHarmMean(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumHarmonicMean(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumHarmonicMean(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumHarmMean(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumHarmMean(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixHarmonicMean(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixHarmonicMean(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixHarmMean(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixHarmMean(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeVariance(self: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeVariance(FrameType(@TypeOf(self)), frameValue(self), names, output_names, correction);
}

pub fn withRowCumulativeVar(self: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeVar(FrameType(@TypeOf(self)), frameValue(self), names, output_names, correction);
}

pub fn withRowCumVariance(self: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumVariance(FrameType(@TypeOf(self)), frameValue(self), names, output_names, correction);
}

pub fn withRowCumVar(self: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumVar(FrameType(@TypeOf(self)), frameValue(self), names, output_names, correction);
}

pub fn withRowPrefixVariance(self: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixVariance(FrameType(@TypeOf(self)), frameValue(self), names, output_names, correction);
}

pub fn withRowPrefixVar(self: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixVar(FrameType(@TypeOf(self)), frameValue(self), names, output_names, correction);
}

pub fn withRowCumulativeStddev(self: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeStddev(FrameType(@TypeOf(self)), frameValue(self), names, output_names, correction);
}

pub fn withRowCumulativeStd(self: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeStd(FrameType(@TypeOf(self)), frameValue(self), names, output_names, correction);
}

pub fn withRowCumStddev(self: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumStddev(FrameType(@TypeOf(self)), frameValue(self), names, output_names, correction);
}

pub fn withRowCumStd(self: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumStd(FrameType(@TypeOf(self)), frameValue(self), names, output_names, correction);
}

pub fn withRowPrefixStddev(self: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixStddev(FrameType(@TypeOf(self)), frameValue(self), names, output_names, correction);
}

pub fn withRowPrefixStd(self: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixStd(FrameType(@TypeOf(self)), frameValue(self), names, output_names, correction);
}

pub fn withRowCumulativeSem(self: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeSem(FrameType(@TypeOf(self)), frameValue(self), names, output_names, correction);
}

pub fn withRowCumSem(self: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumSem(FrameType(@TypeOf(self)), frameValue(self), names, output_names, correction);
}

pub fn withRowPrefixSem(self: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixSem(FrameType(@TypeOf(self)), frameValue(self), names, output_names, correction);
}

pub fn withRowCumulativeCv(self: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeCv(FrameType(@TypeOf(self)), frameValue(self), names, output_names, correction);
}

pub fn withRowCumCv(self: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumCv(FrameType(@TypeOf(self)), frameValue(self), names, output_names, correction);
}

pub fn withRowPrefixCv(self: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixCv(FrameType(@TypeOf(self)), frameValue(self), names, output_names, correction);
}

pub fn withRowCumulativeFano(self: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeFano(FrameType(@TypeOf(self)), frameValue(self), names, output_names, correction);
}

pub fn withRowCumFano(self: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumFano(FrameType(@TypeOf(self)), frameValue(self), names, output_names, correction);
}

pub fn withRowPrefixFano(self: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixFano(FrameType(@TypeOf(self)), frameValue(self), names, output_names, correction);
}

pub fn withRowCumulativeIndexOfDispersion(self: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeIndexOfDispersion(FrameType(@TypeOf(self)), frameValue(self), names, output_names, correction);
}

pub fn withRowCumIndexOfDispersion(self: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumIndexOfDispersion(FrameType(@TypeOf(self)), frameValue(self), names, output_names, correction);
}

pub fn withRowPrefixIndexOfDispersion(self: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixIndexOfDispersion(FrameType(@TypeOf(self)), frameValue(self), names, output_names, correction);
}

pub fn withRowCumulativeSkewness(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeSkewness(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeSkew(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeSkew(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumSkewness(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumSkewness(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumSkew(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumSkew(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixSkewness(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixSkewness(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixSkew(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixSkew(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeKurtosis(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeKurtosis(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeKurt(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeKurt(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumKurtosis(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumKurtosis(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumKurt(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumKurt(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixKurtosis(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixKurtosis(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixKurt(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixKurt(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeRms(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeRms(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumRms(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumRms(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixRms(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixRms(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeMeanAbs(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeMeanAbs(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeMeanAbsolute(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeMeanAbsolute(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumMeanAbs(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumMeanAbs(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumMeanAbsolute(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumMeanAbsolute(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixMeanAbs(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixMeanAbs(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixMeanAbsolute(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixMeanAbsolute(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeMeanSquare(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeMeanSquare(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeMeanSquared(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeMeanSquared(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumMeanSquare(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumMeanSquare(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumMeanSquared(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumMeanSquared(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixMeanSquare(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixMeanSquare(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixMeanSquared(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixMeanSquared(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeMaxAbs(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeMaxAbs(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeMaxAbsolute(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeMaxAbsolute(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeLInfNorm(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeLInfNorm(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeLinfNorm(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeLinfNorm(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumMaxAbs(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumMaxAbs(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumMaxAbsolute(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumMaxAbsolute(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumLInfNorm(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumLInfNorm(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumLinfNorm(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumLinfNorm(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixMaxAbs(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixMaxAbs(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixMaxAbsolute(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixMaxAbsolute(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixLInfNorm(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixLInfNorm(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixLinfNorm(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixLinfNorm(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeMinAbs(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeMinAbs(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeMinAbsolute(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeMinAbsolute(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumMinAbs(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumMinAbs(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumMinAbsolute(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumMinAbsolute(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixMinAbs(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixMinAbs(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixMinAbsolute(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixMinAbsolute(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeL1Norm(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeL1Norm(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumL1Norm(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumL1Norm(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixL1Norm(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixL1Norm(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeL2Norm(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeL2Norm(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumL2Norm(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumL2Norm(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixL2Norm(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixL2Norm(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeProduct(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeProduct(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumprod(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumprod(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumProd(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumProd(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixProduct(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixProduct(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeMax(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeMax(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCummax(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCummax(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumMax(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumMax(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixMax(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixMax(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeMin(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeMin(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCummin(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCummin(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumMin(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumMin(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixMin(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixMin(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeRange(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeRange(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumRange(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumRange(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixRange(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixRange(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativePtp(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativePtp(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumPtp(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumPtp(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixPtp(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixPtp(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowIqrOutlier(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowIqrOutlier(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowIqrOutliers(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowIqrOutliers(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowTukeyOutlier(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowTukeyOutlier(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowTukeyOutliers(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowTukeyOutliers(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowMaxIndicator(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMaxIndicator(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowMaxIndicators(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMaxIndicators(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowIsMax(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowIsMax(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowMaxMask(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMaxMask(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowMinIndicator(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMinIndicator(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowMinIndicators(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMinIndicators(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowIsMin(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowIsMin(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowMinMask(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMinMask(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowTukeyWinsorize(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowTukeyWinsorize(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowTukeyWinsorized(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowTukeyWinsorized(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowIqrWinsorize(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowIqrWinsorize(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowIqrWinsorized(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowIqrWinsorized(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowMinMaxScale(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMinMaxScale(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowMinmaxScale(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMinmaxScale(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowL2Normalize(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowL2Normalize(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowL2Normalized(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowL2Normalized(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowL1Normalize(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowL1Normalize(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowL1Normalized(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowL1Normalized(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowSumNormalize(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowSumNormalize(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowProportion(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowProportion(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowShare(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowShare(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowMeanNormalize(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMeanNormalize(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowMeanNormalized(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMeanNormalized(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowMeanRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMeanRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowMaxAbsNormalize(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMaxAbsNormalize(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowMaxabsNormalize(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMaxabsNormalize(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowLInfNormalize(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowLInfNormalize(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowLinfNormalize(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowLinfNormalize(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowSoftmax(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowSoftmax(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowLogSoftmax(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowLogSoftmax(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowLogsoftmax(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowLogsoftmax(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowSoftmin(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowSoftmin(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowLogSoftmin(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowLogSoftmin(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowLogsoftmin(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowLogsoftmin(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowSoftmaxEntropy(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowSoftmaxEntropy(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowSoftmaxPerplexity(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowSoftmaxPerplexity(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowSoftmaxConfidence(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowSoftmaxConfidence(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowSoftmaxMargin(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowSoftmaxMargin(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowSoftmaxEvenness(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowSoftmaxEvenness(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowSoftmaxNormalizedEntropy(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowSoftmaxNormalizedEntropy(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowSoftmaxConcentration(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowSoftmaxConcentration(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowSoftmaxNormalizedHhi(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowSoftmaxNormalizedHhi(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowSoftmaxNormalizedHHI(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowSoftmaxNormalizedHHI(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowSoftmaxNhhi(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowSoftmaxNhhi(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowSoftmaxGiniImpurity(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowSoftmaxGiniImpurity(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowSoftmaxGini(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowSoftmaxGini(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowSoftmaxInverseSimpson(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowSoftmaxInverseSimpson(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowSoftmaxSimpsonEvenness(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowSoftmaxSimpsonEvenness(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowSoftmaxSimpsonEven(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowSoftmaxSimpsonEven(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowLogitMargin(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowLogitMargin(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowGeometricMean(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowGeometricMean(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowGeoMean(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowGeoMean(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowMagnitudeGeometricMean(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMagnitudeGeometricMean(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAbsGeometricMean(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAbsGeometricMean(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowMagnitudeGeoMean(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMagnitudeGeoMean(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAbsGeoMean(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAbsGeoMean(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowHarmonicMean(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowHarmonicMean(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowHarmMean(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowHarmMean(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowProd(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowProd(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowMin(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMin(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowMax(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMax(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowPtp(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPtp(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowMagnitudePtp(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMagnitudePtp(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAbsPtp(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAbsPtp(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowMagnitudePeakToPeak(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMagnitudePeakToPeak(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAbsPeakToPeak(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAbsPeakToPeak(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowMidrange(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMidrange(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowMagnitudeMidrange(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMagnitudeMidrange(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAbsMidrange(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAbsMidrange(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowRangeCoeff(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowRangeCoeff(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowRangeCoefficient(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowRangeCoefficient(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowMagnitudeRangeCoeff(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMagnitudeRangeCoeff(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAbsRangeCoeff(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAbsRangeCoeff(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowMagnitudeRangeCoefficient(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMagnitudeRangeCoefficient(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAbsRangeCoefficient(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAbsRangeCoefficient(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowMeanAbs(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMeanAbs(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowHhi(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowHhi(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowHerfindahl(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowHerfindahl(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowHerfindahlHirschman(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowHerfindahlHirschman(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowMagnitudeNormalizedHhi(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMagnitudeNormalizedHhi(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAbsNormalizedHhi(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAbsNormalizedHhi(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowMagnitudeSparsity(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMagnitudeSparsity(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAbsSparsity(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAbsSparsity(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowMagnitudeInverseSimpson(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMagnitudeInverseSimpson(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAbsInverseSimpson(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAbsInverseSimpson(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowMagnitudeSimpsonEvenness(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMagnitudeSimpsonEvenness(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAbsSimpsonEvenness(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAbsSimpsonEvenness(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowMagnitudeDominance(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMagnitudeDominance(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAbsDominance(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAbsDominance(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowMagnitudeDominanceMargin(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMagnitudeDominanceMargin(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAbsDominanceMargin(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAbsDominanceMargin(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowMagnitudeEntropy(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMagnitudeEntropy(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAbsEntropy(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAbsEntropy(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowMagnitudePerplexity(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMagnitudePerplexity(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAbsPerplexity(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAbsPerplexity(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowMagnitudeEvenness(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMagnitudeEvenness(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAbsEvenness(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAbsEvenness(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowMeanAbsDev(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMeanAbsDev(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowGiniMeanDiff(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowGiniMeanDiff(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowGiniCoefficient(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowGiniCoefficient(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowGiniCoeff(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowGiniCoeff(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowMeanAbsDevRatio(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMeanAbsDevRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowRms(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowRms(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowL1Norm(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowL1Norm(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowL2Norm(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowL2Norm(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowVariance(self: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowVariance(FrameType(@TypeOf(self)), frameValue(self), names, output_name, correction);
}

pub fn withRowVar(self: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return withRowVariance(self, names, output_name, correction);
}

pub fn withRowMagnitudeVariance(self: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMagnitudeVariance(FrameType(@TypeOf(self)), frameValue(self), names, output_name, correction);
}

pub fn withRowAbsVariance(self: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAbsVariance(FrameType(@TypeOf(self)), frameValue(self), names, output_name, correction);
}

pub fn withRowMagnitudeVar(self: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMagnitudeVar(FrameType(@TypeOf(self)), frameValue(self), names, output_name, correction);
}

pub fn withRowAbsVar(self: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAbsVar(FrameType(@TypeOf(self)), frameValue(self), names, output_name, correction);
}

pub fn withRowStddev(self: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowStddev(FrameType(@TypeOf(self)), frameValue(self), names, output_name, correction);
}

pub fn withRowStd(self: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return withRowStddev(self, names, output_name, correction);
}

pub fn withRowMagnitudeStddev(self: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMagnitudeStddev(FrameType(@TypeOf(self)), frameValue(self), names, output_name, correction);
}

pub fn withRowAbsStddev(self: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAbsStddev(FrameType(@TypeOf(self)), frameValue(self), names, output_name, correction);
}

pub fn withRowMagnitudeStd(self: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMagnitudeStd(FrameType(@TypeOf(self)), frameValue(self), names, output_name, correction);
}

pub fn withRowAbsStd(self: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAbsStd(FrameType(@TypeOf(self)), frameValue(self), names, output_name, correction);
}

pub fn withRowSem(self: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowSem(FrameType(@TypeOf(self)), frameValue(self), names, output_name, correction);
}

pub fn withRowMagnitudeSem(self: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMagnitudeSem(FrameType(@TypeOf(self)), frameValue(self), names, output_name, correction);
}

pub fn withRowAbsSem(self: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAbsSem(FrameType(@TypeOf(self)), frameValue(self), names, output_name, correction);
}

pub fn withRowCv(self: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCv(FrameType(@TypeOf(self)), frameValue(self), names, output_name, correction);
}

pub fn withRowMagnitudeCv(self: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMagnitudeCv(FrameType(@TypeOf(self)), frameValue(self), names, output_name, correction);
}

pub fn withRowAbsCv(self: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAbsCv(FrameType(@TypeOf(self)), frameValue(self), names, output_name, correction);
}

pub fn withRowMagnitudeFano(self: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMagnitudeFano(FrameType(@TypeOf(self)), frameValue(self), names, output_name, correction);
}

pub fn withRowAbsFano(self: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAbsFano(FrameType(@TypeOf(self)), frameValue(self), names, output_name, correction);
}

pub fn withRowMagnitudeIndexOfDispersion(self: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMagnitudeIndexOfDispersion(FrameType(@TypeOf(self)), frameValue(self), names, output_name, correction);
}

pub fn withRowAbsIndexOfDispersion(self: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAbsIndexOfDispersion(FrameType(@TypeOf(self)), frameValue(self), names, output_name, correction);
}

pub fn withRowFano(self: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowFano(FrameType(@TypeOf(self)), frameValue(self), names, output_name, correction);
}

pub fn withRowIndexOfDispersion(self: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowIndexOfDispersion(FrameType(@TypeOf(self)), frameValue(self), names, output_name, correction);
}

pub fn withRowSkewness(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowSkewness(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowSkew(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowSkew(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowMagnitudeSkewness(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMagnitudeSkewness(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAbsSkewness(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAbsSkewness(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowMagnitudeSkew(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMagnitudeSkew(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAbsSkew(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAbsSkew(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowKurtosis(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowKurtosis(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowKurt(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowKurt(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowMagnitudeKurtosis(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMagnitudeKurtosis(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAbsKurtosis(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAbsKurtosis(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowMagnitudeKurt(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowMagnitudeKurt(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAbsKurt(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAbsKurt(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowTrueCount(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowTrueCount(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowFalseCount(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowFalseCount(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowCumulativeTrueCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeTrueCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumTrueCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumTrueCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixTrueCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixTrueCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeFalseCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeFalseCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumFalseCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumFalseCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixFalseCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixFalseCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeTrueRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeTrueRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumTrueRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumTrueRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixTrueRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixTrueRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeFalseRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeFalseRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumFalseRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumFalseRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixFalseRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixFalseRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowAnyTrue(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAnyTrue(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAllTrue(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAllTrue(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAnyFalse(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAnyFalse(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAllFalse(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAllFalse(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowCumulativeAnyTrue(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeAnyTrue(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumAnyTrue(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumAnyTrue(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixAnyTrue(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixAnyTrue(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeAllTrue(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeAllTrue(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumAllTrue(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumAllTrue(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixAllTrue(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixAllTrue(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeAnyFalse(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeAnyFalse(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumAnyFalse(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumAnyFalse(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixAnyFalse(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixAnyFalse(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeAllFalse(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeAllFalse(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumAllFalse(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumAllFalse(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixAllFalse(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixAllFalse(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowFirstTrueIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowFirstTrueIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowLastTrueIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowLastTrueIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowFirstFalseIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowFirstFalseIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowLastFalseIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowLastFalseIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowCumulativeFirstTrueIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeFirstTrueIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixFirstTrueIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixFirstTrueIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeLastTrueIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeLastTrueIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixLastTrueIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixLastTrueIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeFirstFalseIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeFirstFalseIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixFirstFalseIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixFirstFalseIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeLastFalseIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeLastFalseIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixLastFalseIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixLastFalseIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowTrueRatio(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowTrueRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowFalseRatio(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowFalseRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowNaNCount(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowNaNCount(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowNaNRatio(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowNaNRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowNanRatio(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return withRowNaNRatio(self, names, output_name);
}

pub fn withRowInfCount(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowInfCount(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowInfRatio(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowInfRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowPositiveInfCount(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPositiveInfCount(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowNegativeInfCount(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowNegativeInfCount(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowPositiveInfRatio(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPositiveInfRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowNegativeInfRatio(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowNegativeInfRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowZeroCount(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowZeroCount(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowZeroRatio(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowZeroRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowPositiveZeroCount(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPositiveZeroCount(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowNegativeZeroCount(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowNegativeZeroCount(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowPositiveZeroRatio(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPositiveZeroRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowNegativeZeroRatio(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowNegativeZeroRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowNonZeroCount(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowNonZeroCount(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowNonZeroRatio(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowNonZeroRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAnyZero(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAnyZero(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAllZero(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAllZero(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowCumulativeAnyZero(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeAnyZero(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumAnyZero(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumAnyZero(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixAnyZero(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixAnyZero(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeAllZero(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeAllZero(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumAllZero(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumAllZero(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixAllZero(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixAllZero(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowAnyNonZero(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAnyNonZero(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAllNonZero(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAllNonZero(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowCumulativeAnyNonZero(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeAnyNonZero(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumAnyNonZero(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumAnyNonZero(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixAnyNonZero(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixAnyNonZero(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeAllNonZero(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeAllNonZero(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumAllNonZero(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumAllNonZero(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixAllNonZero(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixAllNonZero(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowAnyPositiveZero(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAnyPositiveZero(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAllPositiveZero(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAllPositiveZero(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowCumulativeAnyPositiveZero(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeAnyPositiveZero(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumAnyPositiveZero(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumAnyPositiveZero(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixAnyPositiveZero(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixAnyPositiveZero(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeAllPositiveZero(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeAllPositiveZero(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumAllPositiveZero(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumAllPositiveZero(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixAllPositiveZero(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixAllPositiveZero(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowAnyNegativeZero(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAnyNegativeZero(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAllNegativeZero(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAllNegativeZero(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowCumulativeAnyNegativeZero(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeAnyNegativeZero(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumAnyNegativeZero(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumAnyNegativeZero(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixAnyNegativeZero(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixAnyNegativeZero(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeAllNegativeZero(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeAllNegativeZero(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumAllNegativeZero(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumAllNegativeZero(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixAllNegativeZero(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixAllNegativeZero(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowAnyPositive(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAnyPositive(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAllPositive(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAllPositive(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowCumulativeAnyPositive(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeAnyPositive(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumAnyPositive(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumAnyPositive(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixAnyPositive(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixAnyPositive(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeAllPositive(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeAllPositive(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumAllPositive(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumAllPositive(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixAllPositive(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixAllPositive(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowAnySignBit(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAnySignBit(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAllSignBit(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAllSignBit(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowCumulativeAnySignBit(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeAnySignBit(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumAnySignBit(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumAnySignBit(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixAnySignBit(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixAnySignBit(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeAllSignBit(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeAllSignBit(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumAllSignBit(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumAllSignBit(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixAllSignBit(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixAllSignBit(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowAnyNegative(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAnyNegative(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAllNegative(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAllNegative(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowCumulativeAnyNegative(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeAnyNegative(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumAnyNegative(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumAnyNegative(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixAnyNegative(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixAnyNegative(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeAllNegative(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeAllNegative(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumAllNegative(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumAllNegative(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixAllNegative(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixAllNegative(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowAnyNaN(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAnyNaN(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAllNaN(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAllNaN(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowCumulativeAnyNaN(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeAnyNaN(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumAnyNaN(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumAnyNaN(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixAnyNaN(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixAnyNaN(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeAllNaN(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeAllNaN(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumAllNaN(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumAllNaN(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixAllNaN(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixAllNaN(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowAnyInf(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAnyInf(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAllInf(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAllInf(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowCumulativeAnyInf(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeAnyInf(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumAnyInf(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumAnyInf(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixAnyInf(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixAnyInf(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeAllInf(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeAllInf(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumAllInf(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumAllInf(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixAllInf(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixAllInf(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowAnyPositiveInf(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAnyPositiveInf(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAllPositiveInf(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAllPositiveInf(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowCumulativeAnyPositiveInf(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeAnyPositiveInf(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumAnyPositiveInf(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumAnyPositiveInf(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixAnyPositiveInf(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixAnyPositiveInf(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeAllPositiveInf(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeAllPositiveInf(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumAllPositiveInf(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumAllPositiveInf(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixAllPositiveInf(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixAllPositiveInf(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowAnyNegativeInf(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAnyNegativeInf(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAllNegativeInf(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAllNegativeInf(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowCumulativeAnyNegativeInf(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeAnyNegativeInf(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumAnyNegativeInf(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumAnyNegativeInf(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixAnyNegativeInf(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixAnyNegativeInf(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeAllNegativeInf(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeAllNegativeInf(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumAllNegativeInf(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumAllNegativeInf(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixAllNegativeInf(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixAllNegativeInf(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowAnyFinite(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAnyFinite(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAllFinite(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAllFinite(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowCumulativeAnyFinite(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeAnyFinite(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumAnyFinite(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumAnyFinite(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixAnyFinite(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixAnyFinite(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeAllFinite(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeAllFinite(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumAllFinite(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumAllFinite(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixAllFinite(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixAllFinite(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowAnyNormal(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAnyNormal(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAllNormal(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAllNormal(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowCumulativeAnyNormal(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeAnyNormal(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumAnyNormal(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumAnyNormal(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixAnyNormal(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixAnyNormal(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeAllNormal(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeAllNormal(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumAllNormal(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumAllNormal(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixAllNormal(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixAllNormal(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowAnySubnormal(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAnySubnormal(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAllSubnormal(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAllSubnormal(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowCumulativeAnySubnormal(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeAnySubnormal(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumAnySubnormal(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumAnySubnormal(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixAnySubnormal(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixAnySubnormal(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeAllSubnormal(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeAllSubnormal(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumAllSubnormal(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumAllSubnormal(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixAllSubnormal(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixAllSubnormal(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowAnyNonFinite(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAnyNonFinite(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowAllNonFinite(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowAllNonFinite(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowCumulativeAnyNonFinite(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeAnyNonFinite(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumAnyNonFinite(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumAnyNonFinite(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixAnyNonFinite(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixAnyNonFinite(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeAllNonFinite(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeAllNonFinite(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumAllNonFinite(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumAllNonFinite(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixAllNonFinite(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixAllNonFinite(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowFirstNaNIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowFirstNaNIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowFirstNanIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowFirstNanIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowLastNaNIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowLastNaNIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowLastNanIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowLastNanIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowFirstInfIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowFirstInfIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowLastInfIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowLastInfIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowFirstPositiveInfIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowFirstPositiveInfIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowLastPositiveInfIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowLastPositiveInfIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowFirstNegativeInfIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowFirstNegativeInfIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowLastNegativeInfIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowLastNegativeInfIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowFirstPositiveZeroIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowFirstPositiveZeroIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowLastPositiveZeroIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowLastPositiveZeroIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowFirstNegativeZeroIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowFirstNegativeZeroIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowLastNegativeZeroIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowLastNegativeZeroIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowFirstSignBitIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowFirstSignBitIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowLastSignBitIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowLastSignBitIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowFirstFiniteIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowFirstFiniteIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowLastFiniteIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowLastFiniteIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowFirstNormalIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowFirstNormalIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowLastNormalIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowLastNormalIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowFirstSubnormalIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowFirstSubnormalIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowLastSubnormalIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowLastSubnormalIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowFirstNonFiniteIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowFirstNonFiniteIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowFirstNonfiniteIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowFirstNonfiniteIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowLastNonFiniteIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowLastNonFiniteIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowLastNonfiniteIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowLastNonfiniteIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowFirstZeroIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowFirstZeroIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowLastZeroIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowLastZeroIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowFirstNonZeroIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowFirstNonZeroIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowFirstNonzeroIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowFirstNonzeroIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowLastNonZeroIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowLastNonZeroIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowLastNonzeroIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowLastNonzeroIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowFirstPositiveIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowFirstPositiveIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowLastPositiveIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowLastPositiveIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowFirstNegativeIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowFirstNegativeIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowLastNegativeIndex(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowLastNegativeIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowPositiveCount(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPositiveCount(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowPositiveRatio(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPositiveRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowSignBitCount(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowSignBitCount(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowSignBitRatio(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowSignBitRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowNegativeCount(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowNegativeCount(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowNegativeRatio(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowNegativeRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowCumulativePositiveZeroCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativePositiveZeroCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumPositiveZeroCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumPositiveZeroCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixPositiveZeroCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixPositiveZeroCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativePositiveZeroRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativePositiveZeroRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumPositiveZeroRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumPositiveZeroRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixPositiveZeroRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixPositiveZeroRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeNegativeZeroCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeNegativeZeroCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumNegativeZeroCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumNegativeZeroCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixNegativeZeroCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixNegativeZeroCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeNegativeZeroRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeNegativeZeroRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumNegativeZeroRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumNegativeZeroRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixNegativeZeroRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixNegativeZeroRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeSignBitCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeSignBitCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumSignBitCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumSignBitCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixSignBitCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixSignBitCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeSignBitRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeSignBitRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumSignBitRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumSignBitRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixSignBitRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixSignBitRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeNaNCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeNaNCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumNaNCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumNaNCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixNaNCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixNaNCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeNaNRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeNaNRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumNaNRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumNaNRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixNaNRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixNaNRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeInfCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeInfCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumInfCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumInfCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixInfCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixInfCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeInfRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeInfRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumInfRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumInfRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixInfRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixInfRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativePositiveInfCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativePositiveInfCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumPositiveInfCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumPositiveInfCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixPositiveInfCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixPositiveInfCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativePositiveInfRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativePositiveInfRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumPositiveInfRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumPositiveInfRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixPositiveInfRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixPositiveInfRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeNegativeInfCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeNegativeInfCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumNegativeInfCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumNegativeInfCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixNegativeInfCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixNegativeInfCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeNegativeInfRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeNegativeInfRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumNegativeInfRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumNegativeInfRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixNegativeInfRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixNegativeInfRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeFiniteCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeFiniteCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumFiniteCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumFiniteCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixFiniteCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixFiniteCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeFiniteRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeFiniteRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumFiniteRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumFiniteRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixFiniteRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixFiniteRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeNormalCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeNormalCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumNormalCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumNormalCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixNormalCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixNormalCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeNormalRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeNormalRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumNormalRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumNormalRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixNormalRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixNormalRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeSubnormalCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeSubnormalCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumSubnormalCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumSubnormalCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixSubnormalCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixSubnormalCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeSubnormalRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeSubnormalRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumSubnormalRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumSubnormalRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixSubnormalRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixSubnormalRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeNonFiniteCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeNonFiniteCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumNonFiniteCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumNonFiniteCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixNonFiniteCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixNonFiniteCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeNonFiniteRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeNonFiniteRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumNonFiniteRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumNonFiniteRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixNonFiniteRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixNonFiniteRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeZeroCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeZeroCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumZeroCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumZeroCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixZeroCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixZeroCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeFirstNaNIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeFirstNaNIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixFirstNaNIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixFirstNaNIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeLastNaNIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeLastNaNIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixLastNaNIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixLastNaNIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeFirstInfIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeFirstInfIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixFirstInfIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixFirstInfIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeLastInfIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeLastInfIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixLastInfIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixLastInfIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeFirstPositiveInfIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeFirstPositiveInfIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixFirstPositiveInfIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixFirstPositiveInfIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeLastPositiveInfIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeLastPositiveInfIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixLastPositiveInfIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixLastPositiveInfIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeFirstNegativeInfIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeFirstNegativeInfIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixFirstNegativeInfIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixFirstNegativeInfIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeLastNegativeInfIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeLastNegativeInfIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixLastNegativeInfIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixLastNegativeInfIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeFirstFiniteIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeFirstFiniteIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixFirstFiniteIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixFirstFiniteIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeLastFiniteIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeLastFiniteIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixLastFiniteIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixLastFiniteIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeFirstNormalIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeFirstNormalIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixFirstNormalIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixFirstNormalIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeLastNormalIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeLastNormalIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixLastNormalIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixLastNormalIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeFirstSubnormalIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeFirstSubnormalIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixFirstSubnormalIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixFirstSubnormalIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeLastSubnormalIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeLastSubnormalIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixLastSubnormalIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixLastSubnormalIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeFirstNonFiniteIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeFirstNonFiniteIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixFirstNonFiniteIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixFirstNonFiniteIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeLastNonFiniteIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeLastNonFiniteIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixLastNonFiniteIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixLastNonFiniteIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeFirstZeroIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeFirstZeroIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixFirstZeroIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixFirstZeroIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeLastZeroIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeLastZeroIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixLastZeroIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixLastZeroIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeFirstPositiveZeroIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeFirstPositiveZeroIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixFirstPositiveZeroIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixFirstPositiveZeroIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeLastPositiveZeroIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeLastPositiveZeroIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixLastPositiveZeroIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixLastPositiveZeroIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeFirstNegativeZeroIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeFirstNegativeZeroIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixFirstNegativeZeroIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixFirstNegativeZeroIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeLastNegativeZeroIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeLastNegativeZeroIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixLastNegativeZeroIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixLastNegativeZeroIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeNonZeroCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeNonZeroCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumNonZeroCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumNonZeroCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixNonZeroCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixNonZeroCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeFirstNonZeroIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeFirstNonZeroIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeFirstNonzeroIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeFirstNonzeroIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixFirstNonZeroIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixFirstNonZeroIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixFirstNonzeroIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixFirstNonzeroIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeLastNonZeroIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeLastNonZeroIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeLastNonzeroIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeLastNonzeroIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixLastNonZeroIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixLastNonZeroIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixLastNonzeroIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixLastNonzeroIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeFirstPositiveIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeFirstPositiveIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixFirstPositiveIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixFirstPositiveIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeLastPositiveIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeLastPositiveIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixLastPositiveIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixLastPositiveIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeFirstSignBitIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeFirstSignBitIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixFirstSignBitIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixFirstSignBitIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeLastSignBitIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeLastSignBitIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixLastSignBitIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixLastSignBitIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeFirstNegativeIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeFirstNegativeIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixFirstNegativeIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixFirstNegativeIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeLastNegativeIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeLastNegativeIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixLastNegativeIndex(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixLastNegativeIndex(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativePositiveCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativePositiveCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumPositiveCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumPositiveCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixPositiveCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixPositiveCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeNegativeCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeNegativeCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumNegativeCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumNegativeCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixNegativeCount(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixNegativeCount(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeZeroRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeZeroRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumZeroRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumZeroRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixZeroRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixZeroRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeNonZeroRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeNonZeroRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumNonZeroRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumNonZeroRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixNonZeroRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixNonZeroRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativePositiveRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativePositiveRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumPositiveRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumPositiveRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixPositiveRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixPositiveRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumulativeNegativeRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumulativeNegativeRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowCumNegativeRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowCumNegativeRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowPrefixNegativeRatio(self: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPrefixNegativeRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_names);
}

pub fn withRowFiniteCount(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowFiniteCount(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowFiniteRatio(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowFiniteRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowNormalCount(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowNormalCount(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowNormalRatio(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowNormalRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowSubnormalCount(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowSubnormalCount(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowSubnormalRatio(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowSubnormalRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowNonFiniteCount(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowNonFiniteCount(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowNonFiniteRatio(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowNonFiniteRatio(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
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

pub fn withRowIndex(self: anytype, name: []const u8, row_offset: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowIndex(FrameType(@TypeOf(self)), frameValue(self), name, row_offset);
}

fn optionalValidityBit(validity: ?[]const bool, index: usize) bool {
    return if (validity) |mask| mask[index] else true;
}

fn typedColumnsEqual(allocator: std.mem.Allocator, left: anytype, right: @TypeOf(left)) DeviceDataError!bool {
    if (left.len() != right.len()) return false;
    const left_validity = try validity_mod.validityValues(left, allocator);
    defer if (left_validity) |mask| allocator.free(mask);
    const right_validity = try validity_mod.validityValues(right, allocator);
    defer if (right_validity) |mask| allocator.free(mask);
    const left_values = try left.toOwnedSlice(allocator);
    defer allocator.free(left_values);
    const right_values = try right.toOwnedSlice(allocator);
    defer allocator.free(right_values);

    for (left_values, right_values, 0..) |left_value, right_value, row| {
        const left_valid = optionalValidityBit(left_validity, row);
        const right_valid = optionalValidityBit(right_validity, row);
        if (left_valid != right_valid) return false;
        if (left_valid and !std.meta.eql(left_value, right_value)) return false;
    }
    return true;
}

fn realValueAsF64(comptime T: type, value: T) f64 {
    if (comptime T == array_mod.BFloat16) return value.toF64();
    return @floatCast(value);
}

fn scalarAllClose(comptime T: type, left: T, right: T, rtol: f64, atol: f64, equal_nan: bool) bool {
    if (comptime T == array_mod.BFloat16 or @typeInfo(T) == .float) {
        const left_value = realValueAsF64(T, left);
        const right_value = realValueAsF64(T, right);
        const left_nan = std.math.isNan(left_value);
        const right_nan = std.math.isNan(right_value);
        if (left_nan or right_nan) return equal_nan and left_nan and right_nan;
        if (left_value == right_value) return true;
        if (!std.math.isFinite(left_value) or !std.math.isFinite(right_value)) return false;
        return @abs(left_value - right_value) <= atol + rtol * @abs(right_value);
    }
    return std.meta.eql(left, right);
}

fn typedColumnsAllClose(allocator: std.mem.Allocator, left: anytype, right: @TypeOf(left), rtol: f64, atol: f64, equal_nan: bool) DeviceDataError!bool {
    if (left.len() != right.len()) return false;
    const left_validity = try validity_mod.validityValues(left, allocator);
    defer if (left_validity) |mask| allocator.free(mask);
    const right_validity = try validity_mod.validityValues(right, allocator);
    defer if (right_validity) |mask| allocator.free(mask);
    const left_values = try left.toOwnedSlice(allocator);
    defer allocator.free(left_values);
    const right_values = try right.toOwnedSlice(allocator);
    defer allocator.free(right_values);

    for (left_values, right_values, 0..) |left_value, right_value, row| {
        const left_valid = optionalValidityBit(left_validity, row);
        const right_valid = optionalValidityBit(right_validity, row);
        if (left_valid != right_valid) return false;
        if (left_valid and !scalarAllClose(@TypeOf(left_value), left_value, right_value, rtol, atol, equal_nan)) return false;
    }
    return true;
}

fn columnsEqual(allocator: std.mem.Allocator, left: anytype, right: @TypeOf(left)) DeviceDataError!bool {
    if (left.dtype() != right.dtype()) return false;
    return switch (left) {
        inline else => |typed, tag| try typedColumnsEqual(allocator, typed, @field(right, @tagName(tag))),
    };
}

fn columnsAllClose(allocator: std.mem.Allocator, left: anytype, right: @TypeOf(left), rtol: f64, atol: f64, equal_nan: bool) DeviceDataError!bool {
    if (left.dtype() != right.dtype()) return false;
    return switch (left) {
        inline else => |typed, tag| try typedColumnsAllClose(allocator, typed, @field(right, @tagName(tag)), rtol, atol, equal_nan),
    };
}

pub fn equals(self: anytype, other: FrameType(@TypeOf(self))) DeviceDataError!bool {
    const left = frameValue(self);
    if (left.rows != other.rows or left.names.len != other.names.len) return false;
    for (left.names, other.names, left.columns, other.columns) |left_name, right_name, left_column, right_column| {
        if (!std.mem.eql(u8, left_name, right_name)) return false;
        if (!try columnsEqual(left.allocator, left_column, right_column)) return false;
    }
    return true;
}

pub const frameEquals = equals;

pub fn allClose(self: anytype, other: FrameType(@TypeOf(self)), rtol: f64, atol: f64) DeviceDataError!bool {
    return allCloseEqualNan(self, other, rtol, atol, false);
}

pub fn allCloseEqualNan(self: anytype, other: FrameType(@TypeOf(self)), rtol: f64, atol: f64, equal_nan: bool) DeviceDataError!bool {
    if (std.math.isNan(rtol) or std.math.isNan(atol) or rtol < 0.0 or atol < 0.0) return error.InvalidShape;
    const left = frameValue(self);
    if (!schemaEquals(left, other) or left.rows != other.rows) return false;
    for (left.columns, other.columns) |left_column, right_column| {
        if (!try columnsAllClose(left.allocator, left_column, right_column, rtol, atol, equal_nan)) return false;
    }
    return true;
}

pub const frameAllClose = allClose;

pub fn schemaEquals(self: anytype, other: FrameType(@TypeOf(self))) bool {
    const left = frameValue(self);
    if (left.names.len != other.names.len) return false;
    for (left.names, other.names, left.columns, other.columns) |left_name, right_name, left_column, right_column| {
        if (!std.mem.eql(u8, left_name, right_name)) return false;
        if (left_column.dtype() != right_column.dtype()) return false;
        if (left_column.nullable() != right_column.nullable()) return false;
    }
    return true;
}

pub const sameSchema = schemaEquals;
pub const schemaCompatible = schemaEquals;

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

pub fn stripColumnNamePrefix(self: anytype, prefix: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.stripColumnNamePrefix(FrameType(@TypeOf(self)), frameValue(self), prefix);
}

pub const removeColumnNamePrefix = stripColumnNamePrefix;

pub fn stripColumnNameSuffix(self: anytype, suffix: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.stripColumnNameSuffix(FrameType(@TypeOf(self)), frameValue(self), suffix);
}

pub const removeColumnNameSuffix = stripColumnNameSuffix;

pub fn replaceColumnNamePrefix(self: anytype, old_prefix: []const u8, new_prefix: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.replaceColumnNamePrefix(FrameType(@TypeOf(self)), frameValue(self), old_prefix, new_prefix);
}

pub fn replaceColumnNameSuffix(self: anytype, old_suffix: []const u8, new_suffix: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.replaceColumnNameSuffix(FrameType(@TypeOf(self)), frameValue(self), old_suffix, new_suffix);
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

pub fn selectExcept(self: anytype, names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dropColumns(self, names);
}

pub const selectAllExcept = selectExcept;
pub const excludeColumns = selectExcept;

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

pub fn dropAllNulls(self: anytype, names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropAllNulls(FrameType(@TypeOf(self)), frameValue(self), names);
}

pub fn dropAllNullsOn(self: anytype, names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dropAllNulls(self, names);
}

pub fn filterAllNulls(self: anytype, names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.filterAllNulls(FrameType(@TypeOf(self)), frameValue(self), names);
}

pub fn filterAllNullsOn(self: anytype, names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return filterAllNulls(self, names);
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

pub fn dropInfs(self: anytype, names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropInfs(FrameType(@TypeOf(self)), frameValue(self), names);
}

pub fn dropInfsOn(self: anytype, names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dropInfs(self, names);
}

pub fn dropInfsColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dropInfs(self, &.{name});
}

pub fn filterInfsColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.filterInfsColumn(FrameType(@TypeOf(self)), frameValue(self), name);
}

pub fn dropPositiveInfs(self: anytype, names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropPositiveInfs(FrameType(@TypeOf(self)), frameValue(self), names);
}

pub fn dropPositiveInfsOn(self: anytype, names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dropPositiveInfs(self, names);
}

pub fn dropPositiveInfsColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dropPositiveInfs(self, &.{name});
}

pub fn filterPositiveInfsColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.filterPositiveInfsColumn(FrameType(@TypeOf(self)), frameValue(self), name);
}

pub fn dropNegativeInfs(self: anytype, names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropNegativeInfs(FrameType(@TypeOf(self)), frameValue(self), names);
}

pub fn dropNegativeInfsOn(self: anytype, names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dropNegativeInfs(self, names);
}

pub fn dropNegativeInfsColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dropNegativeInfs(self, &.{name});
}

pub fn filterNegativeInfsColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.filterNegativeInfsColumn(FrameType(@TypeOf(self)), frameValue(self), name);
}

pub fn dropZeros(self: anytype, names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropZeros(FrameType(@TypeOf(self)), frameValue(self), names);
}

pub fn dropZerosOn(self: anytype, names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dropZeros(self, names);
}

pub fn dropZerosColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dropZeros(self, &.{name});
}

pub fn filterZerosColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.filterZerosColumn(FrameType(@TypeOf(self)), frameValue(self), name);
}

pub fn dropPositiveZeros(self: anytype, names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropPositiveZeros(FrameType(@TypeOf(self)), frameValue(self), names);
}

pub fn dropPositiveZerosOn(self: anytype, names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dropPositiveZeros(self, names);
}

pub fn dropPositiveZerosColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dropPositiveZeros(self, &.{name});
}

pub fn filterPositiveZerosColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.filterPositiveZerosColumn(FrameType(@TypeOf(self)), frameValue(self), name);
}

pub fn dropNegativeZeros(self: anytype, names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropNegativeZeros(FrameType(@TypeOf(self)), frameValue(self), names);
}

pub fn dropNegativeZerosOn(self: anytype, names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dropNegativeZeros(self, names);
}

pub fn dropNegativeZerosColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dropNegativeZeros(self, &.{name});
}

pub fn filterNegativeZerosColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.filterNegativeZerosColumn(FrameType(@TypeOf(self)), frameValue(self), name);
}

pub fn dropNonZeros(self: anytype, names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropNonZeros(FrameType(@TypeOf(self)), frameValue(self), names);
}

pub fn dropNonZerosOn(self: anytype, names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dropNonZeros(self, names);
}

pub fn dropNonZerosColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dropNonZeros(self, &.{name});
}

pub fn filterNonZerosColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.filterNonZerosColumn(FrameType(@TypeOf(self)), frameValue(self), name);
}

pub fn dropPositives(self: anytype, names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropPositives(FrameType(@TypeOf(self)), frameValue(self), names);
}

pub fn dropPositivesOn(self: anytype, names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dropPositives(self, names);
}

pub fn dropPositivesColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dropPositives(self, &.{name});
}

pub fn filterPositivesColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.filterPositivesColumn(FrameType(@TypeOf(self)), frameValue(self), name);
}

pub fn dropSignBits(self: anytype, names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropSignBits(FrameType(@TypeOf(self)), frameValue(self), names);
}

pub fn dropSignBitsOn(self: anytype, names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dropSignBits(self, names);
}

pub fn dropSignBitsColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dropSignBits(self, &.{name});
}

pub fn filterSignBitsColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.filterSignBitsColumn(FrameType(@TypeOf(self)), frameValue(self), name);
}

pub fn dropNegatives(self: anytype, names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropNegatives(FrameType(@TypeOf(self)), frameValue(self), names);
}

pub fn dropNegativesOn(self: anytype, names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dropNegatives(self, names);
}

pub fn dropNegativesColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dropNegatives(self, &.{name});
}

pub fn filterNegativesColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.filterNegativesColumn(FrameType(@TypeOf(self)), frameValue(self), name);
}

pub fn dropFinites(self: anytype, names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropFinites(FrameType(@TypeOf(self)), frameValue(self), names);
}

pub fn dropFinitesOn(self: anytype, names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dropFinites(self, names);
}

pub fn dropFinitesColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dropFinites(self, &.{name});
}

pub fn filterFinitesColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.filterFinitesColumn(FrameType(@TypeOf(self)), frameValue(self), name);
}

pub fn dropNormals(self: anytype, names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropNormals(FrameType(@TypeOf(self)), frameValue(self), names);
}

pub fn dropNormalsOn(self: anytype, names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dropNormals(self, names);
}

pub fn dropNormalsColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dropNormals(self, &.{name});
}

pub fn filterNormalsColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.filterNormalsColumn(FrameType(@TypeOf(self)), frameValue(self), name);
}

pub fn dropSubnormals(self: anytype, names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropSubnormals(FrameType(@TypeOf(self)), frameValue(self), names);
}

pub fn dropSubnormalsOn(self: anytype, names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dropSubnormals(self, names);
}

pub fn dropSubnormalsColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dropSubnormals(self, &.{name});
}

pub fn filterSubnormalsColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.filterSubnormalsColumn(FrameType(@TypeOf(self)), frameValue(self), name);
}

pub fn dropNonFinites(self: anytype, names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropNonFinites(FrameType(@TypeOf(self)), frameValue(self), names);
}

pub fn dropNonFinitesOn(self: anytype, names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dropNonFinites(self, names);
}

pub fn dropNonFinitesColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dropNonFinites(self, &.{name});
}

pub fn filterNonFinitesColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.filterNonFinitesColumn(FrameType(@TypeOf(self)), frameValue(self), name);
}

pub fn head(self: anytype, n: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return sliceRows(self, 0, @min(n, self.rows));
}

pub fn limit(self: anytype, n: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return head(self, n);
}

pub fn firstRow(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return head(self, 1);
}

pub fn tail(self: anytype, n: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    const count = @min(n, self.rows);
    return sliceRows(self, self.rows - count, self.rows);
}

pub fn lastRow(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return tail(self, 1);
}

pub fn sliceRows(self: anytype, start: usize, stop: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.sliceRows(FrameType(@TypeOf(self)), frameValue(self), start, stop);
}

pub fn sliceRowsLen(self: anytype, start: usize, length: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    const stop = std.math.add(usize, start, length) catch return error.InvalidShape;
    return sliceRows(self, start, stop);
}

pub fn offset(self: anytype, n: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return sliceRows(self, n, std.math.maxInt(usize));
}

pub fn sliceRowsSigned(self: anytype, start: isize, length: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.sliceRowsSigned(FrameType(@TypeOf(self)), frameValue(self), start, length);
}

pub fn sliceSigned(self: anytype, start: isize, length: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return sliceRowsSigned(self, start, length);
}

pub fn sliceRowsSignedStep(self: anytype, start: isize, stop: isize, step: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.sliceRowsSignedStep(FrameType(@TypeOf(self)), frameValue(self), start, stop, step);
}

pub fn sliceSignedStep(self: anytype, start: isize, stop: isize, step: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return sliceRowsSignedStep(self, start, stop, step);
}

pub fn dropRows(self: anytype, row_indices: []const usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropRows(FrameType(@TypeOf(self)), frameValue(self), row_indices);
}

pub fn dropRowsMode(self: anytype, row_indices: []const usize, mode: array_mod.IndexMode) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropRowsMode(FrameType(@TypeOf(self)), frameValue(self), row_indices, mode);
}

pub fn dropRowsSigned(self: anytype, row_indices: []const isize) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropRowsSigned(FrameType(@TypeOf(self)), frameValue(self), row_indices);
}

pub fn dropRowsSignedMode(self: anytype, row_indices: []const isize, mode: array_mod.IndexMode) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropRowsSignedMode(FrameType(@TypeOf(self)), frameValue(self), row_indices, mode);
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

pub fn takeMode(self: anytype, row_indices: []const usize, mode: array_mod.IndexMode) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.takeRowsMode(FrameType(@TypeOf(self)), frameValue(self), row_indices, mode);
}

pub fn takeSigned(self: anytype, row_indices: []const isize) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.takeRowsSigned(FrameType(@TypeOf(self)), frameValue(self), row_indices);
}

pub fn takeSignedMode(self: anytype, row_indices: []const isize, mode: array_mod.IndexMode) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.takeRowsSignedMode(FrameType(@TypeOf(self)), frameValue(self), row_indices, mode);
}

pub fn takeOptional(self: anytype, row_indices: []const ?usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.takeOptionalRows(FrameType(@TypeOf(self)), frameValue(self), row_indices);
}

pub fn takeOptionalRows(self: anytype, row_indices: []const ?usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return takeOptional(self, row_indices);
}

pub fn takeByColumn(self: anytype, index_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.takeRowsByColumn(FrameType(@TypeOf(self)), frameValue(self), index_name);
}

pub fn takeByColumnMode(self: anytype, index_name: []const u8, mode: array_mod.IndexMode) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.takeRowsByColumnMode(FrameType(@TypeOf(self)), frameValue(self), index_name, mode);
}

pub fn takeRowsByColumn(self: anytype, index_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return takeByColumn(self, index_name);
}

pub fn takeRowsByColumnMode(self: anytype, index_name: []const u8, mode: array_mod.IndexMode) DeviceDataError!FrameType(@TypeOf(self)) {
    return takeByColumnMode(self, index_name, mode);
}

pub fn dropRowsByColumn(self: anytype, index_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropRowsByColumn(FrameType(@TypeOf(self)), frameValue(self), index_name);
}

pub fn dropRowsByColumnMode(self: anytype, index_name: []const u8, mode: array_mod.IndexMode) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.dropRowsByColumnMode(FrameType(@TypeOf(self)), frameValue(self), index_name, mode);
}

pub fn repeatRows(self: anytype, repeat_count: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.repeatRows(FrameType(@TypeOf(self)), frameValue(self), repeat_count);
}

pub fn tileRows(self: anytype, tile_count: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.tileRows(FrameType(@TypeOf(self)), frameValue(self), tile_count);
}

pub fn repeatRowsByColumn(self: anytype, count_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.repeatRowsByColumn(FrameType(@TypeOf(self)), frameValue(self), count_name);
}

pub fn sampleRows(self: anytype, count: usize, seed: u64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.sampleRows(FrameType(@TypeOf(self)), frameValue(self), count, seed);
}

pub fn shuffleRows(self: anytype, seed: u64) DeviceDataError!FrameType(@TypeOf(self)) {
    return sampleRows(self, frameValue(self).rows, seed);
}

pub fn sampleRowsFraction(self: anytype, fraction: f64, seed: u64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.sampleRowsFraction(FrameType(@TypeOf(self)), frameValue(self), fraction, seed);
}

pub const sampleFrac = sampleRowsFraction;

pub fn sampleRowsWithReplacement(self: anytype, count: usize, seed: u64) DeviceDataError!FrameType(@TypeOf(self)) {
    @setEvalBranchQuota(2000);
    return dataframe_array_mod.sampleRowsWithReplacement(FrameType(@TypeOf(self)), frameValue(self), count, seed);
}

pub fn sampleRowsFractionWithReplacement(self: anytype, fraction: f64, seed: u64) DeviceDataError!FrameType(@TypeOf(self)) {
    @setEvalBranchQuota(2000);
    return dataframe_array_mod.sampleRowsFractionWithReplacement(FrameType(@TypeOf(self)), frameValue(self), fraction, seed);
}

pub const sampleFracWithReplacement = sampleRowsFractionWithReplacement;

pub fn strideRows(self: anytype, start: usize, step: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.strideRows(FrameType(@TypeOf(self)), frameValue(self), start, step);
}

pub fn reverseRows(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.reverseRows(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn rollRows(self: anytype, shift: isize) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.rollRows(FrameType(@TypeOf(self)), frameValue(self), shift);
}

pub fn shiftRows(self: anytype, shift: isize) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.shiftRows(FrameType(@TypeOf(self)), frameValue(self), shift);
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

pub fn concatColumns(self: anytype, other: FrameType(@TypeOf(self))) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.concatDeviceDataFramesColumns(FrameType(@TypeOf(self)), frameValue(self), other);
}

pub fn appendColumns(self: anytype, other: FrameType(@TypeOf(self))) DeviceDataError!FrameType(@TypeOf(self)) {
    return concatColumns(self, other);
}

pub fn hstack(self: anytype, other: FrameType(@TypeOf(self))) DeviceDataError!FrameType(@TypeOf(self)) {
    return concatColumns(self, other);
}

pub fn distinctRows(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return keys_mod.distinctRows(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn distinctRowsLast(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return keys_mod.distinctRowsLast(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn distinctRowsNone(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return keys_mod.distinctRowsNone(FrameType(@TypeOf(self)), frameValue(self));
}

pub fn distinctOn(self: anytype, key_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return keys_mod.distinctOn(FrameType(@TypeOf(self)), frameValue(self), key_names);
}

pub fn distinctOnLast(self: anytype, key_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return keys_mod.distinctOnLast(FrameType(@TypeOf(self)), frameValue(self), key_names);
}

pub fn distinctOnNone(self: anytype, key_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return keys_mod.distinctOnNone(FrameType(@TypeOf(self)), frameValue(self), key_names);
}

fn countMatchingRows(self: anytype, key_names: []const []const u8, comptime mark_unique: bool) DeviceDataError!usize {
    const frame = frameValue(self);
    const mask = try keys_mod.rowDuplicateMask(frame.allocator, frame, key_names, mark_unique);
    defer frame.allocator.free(mask);
    var count: usize = 0;
    for (mask) |matched| {
        if (matched) count += 1;
    }
    return count;
}

pub fn distinctRowCount(self: anytype) DeviceDataError!usize {
    return distinctRowCountOn(self, frameValue(self).names);
}

pub fn distinctRowCountOn(self: anytype, key_names: []const []const u8) DeviceDataError!usize {
    const frame = frameValue(self);
    const indices = try keys_mod.distinctRowIndices(frame.allocator, frame, key_names);
    defer frame.allocator.free(indices);
    return indices.len;
}

pub fn uniqueRowCount(self: anytype) DeviceDataError!usize {
    return uniqueRowCountOn(self, frameValue(self).names);
}

pub fn uniqueRowCountOn(self: anytype, key_names: []const []const u8) DeviceDataError!usize {
    return countMatchingRows(self, key_names, true);
}

fn rowRatio(count: usize, rows: usize) f64 {
    if (rows == 0) return std.math.nan(f64);
    return @as(f64, @floatFromInt(count)) / @as(f64, @floatFromInt(rows));
}

pub fn distinctRowRatio(self: anytype) DeviceDataError!f64 {
    return rowRatio(try distinctRowCount(self), frameValue(self).rows);
}

pub fn distinctRowRatioOn(self: anytype, key_names: []const []const u8) DeviceDataError!f64 {
    return rowRatio(try distinctRowCountOn(self, key_names), frameValue(self).rows);
}

pub fn uniqueRowRatio(self: anytype) DeviceDataError!f64 {
    return rowRatio(try uniqueRowCount(self), frameValue(self).rows);
}

pub fn uniqueRowRatioOn(self: anytype, key_names: []const []const u8) DeviceDataError!f64 {
    return rowRatio(try uniqueRowCountOn(self, key_names), frameValue(self).rows);
}

pub fn duplicateRowCount(self: anytype) DeviceDataError!usize {
    return duplicateRowCountOn(self, frameValue(self).names);
}

pub fn duplicateRowCountOn(self: anytype, key_names: []const []const u8) DeviceDataError!usize {
    return countMatchingRows(self, key_names, false);
}

pub fn duplicateRowRatio(self: anytype) DeviceDataError!f64 {
    return rowRatio(try duplicateRowCount(self), frameValue(self).rows);
}

pub fn duplicateRowRatioOn(self: anytype, key_names: []const []const u8) DeviceDataError!f64 {
    return rowRatio(try duplicateRowCountOn(self, key_names), frameValue(self).rows);
}

pub fn hasDuplicateRows(self: anytype) DeviceDataError!bool {
    return try duplicateRowCount(self) != 0;
}

pub fn hasDuplicateRowsOn(self: anytype, key_names: []const []const u8) DeviceDataError!bool {
    return try duplicateRowCountOn(self, key_names) != 0;
}

pub fn withRowIsDuplicated(self: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return keys_mod.withRowIsDuplicated(FrameType(@TypeOf(self)), frameValue(self), key_names, output_name);
}

pub fn withRowIsUnique(self: anytype, key_names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return keys_mod.withRowIsUnique(FrameType(@TypeOf(self)), frameValue(self), key_names, output_name);
}

pub fn dropDuplicates(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return distinctRows(self);
}

pub fn dropDuplicatesOn(self: anytype, key_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return distinctOn(self, key_names);
}

pub fn dropDuplicatesLast(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return distinctRowsLast(self);
}

pub fn dropDuplicatesOnLast(self: anytype, key_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return distinctOnLast(self, key_names);
}

pub fn dropDuplicatesNone(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return distinctRowsNone(self);
}

pub fn dropDuplicatesOnNone(self: anytype, key_names: []const []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return distinctOnNone(self, key_names);
}

pub fn uniqueRows(self: anytype) DeviceDataError!FrameType(@TypeOf(self)) {
    return distinctRows(self);
}

pub fn argsortBy(self: anytype, name: []const u8, options_value: DeviceSortOptions) DeviceDataError![]usize {
    return rank_mod.argsortBy(frameValue(self), name, options_value);
}

pub fn isSortedBy(self: anytype, name: []const u8, options_value: DeviceSortOptions) DeviceDataError!bool {
    return rank_mod.isSortedBy(frameValue(self), name, options_value);
}

pub fn isSortedByColumn(self: anytype, name: []const u8, options_value: DeviceSortOptions) DeviceDataError!bool {
    return isSortedBy(self, name, options_value);
}

pub fn sortBy(self: anytype, name: []const u8, options_value: DeviceSortOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return rank_mod.sortBy(FrameType(@TypeOf(self)), frameValue(self), name, options_value);
}

pub fn sortByColumn(self: anytype, name: []const u8, options_value: DeviceSortOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return sortBy(self, name, options_value);
}

pub fn argsortByColumns(self: anytype, names: []const []const u8, options_values: []const DeviceSortOptions) DeviceDataError![]usize {
    return rank_mod.argsortByColumns(frameValue(self), names, options_values);
}

pub fn isSortedByColumns(self: anytype, names: []const []const u8, options_values: []const DeviceSortOptions) DeviceDataError!bool {
    return rank_mod.isSortedByColumns(frameValue(self), names, options_values);
}

pub fn sortByColumns(self: anytype, names: []const []const u8, options_values: []const DeviceSortOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return rank_mod.sortByColumns(FrameType(@TypeOf(self)), frameValue(self), names, options_values);
}

pub fn topKByColumns(self: anytype, names: []const []const u8, k: usize, options_values: []const DeviceSortOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return rank_mod.topKByColumns(FrameType(@TypeOf(self)), frameValue(self), names, k, options_values);
}

pub fn topKBy(self: anytype, name: []const u8, k: usize, options_value: DeviceSortOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return rank_mod.topKBy(FrameType(@TypeOf(self)), frameValue(self), name, k, options_value);
}

pub fn bottomKByColumns(self: anytype, names: []const []const u8, k: usize, options_values: []const DeviceSortOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return topKByColumns(self, names, k, options_values);
}

pub fn bottomKBy(self: anytype, name: []const u8, k: usize, options_value: DeviceSortOptions) DeviceDataError!FrameType(@TypeOf(self)) {
    return topKBy(self, name, k, options_value);
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
