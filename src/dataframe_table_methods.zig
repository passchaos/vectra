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
const DeviceColumnLogicalOp = options_mod.DeviceColumnLogicalOp;
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

pub fn firstNonzeroIndexColumn(self: anytype, name: []const u8) DeviceDataError!?usize {
    return expr_mod.firstNonzeroIndexColumn(frameValue(self), name);
}

pub fn lastNonzeroIndexColumn(self: anytype, name: []const u8) DeviceDataError!?usize {
    return expr_mod.lastNonzeroIndexColumn(frameValue(self), name);
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

pub fn countTrueColumn(self: anytype, name: []const u8) DeviceDataError!usize {
    return expr_mod.countTrueColumn(frameValue(self), name);
}

pub fn countFalseColumn(self: anytype, name: []const u8) DeviceDataError!usize {
    return expr_mod.countFalseColumn(frameValue(self), name);
}

pub fn logicalColumnScalar(self: anytype, name: []const u8, scalar: bool, op: DeviceColumnLogicalOp) DeviceDataError!@TypeOf(frameValue(self).columns[0]) {
    return expr_mod.logicalColumnScalar(frameValue(self), name, scalar, op);
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

pub fn filterColumn(self: anytype, name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return expr_mod.filterColumn(FrameType(@TypeOf(self)), frameValue(self), name);
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

pub fn withRowNaNCount(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowNaNCount(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowInfCount(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowInfCount(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowPositiveInfCount(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPositiveInfCount(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowNegativeInfCount(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowNegativeInfCount(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowZeroCount(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowZeroCount(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowPositiveZeroCount(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPositiveZeroCount(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowNegativeZeroCount(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowNegativeZeroCount(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowNonZeroCount(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowNonZeroCount(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowPositiveCount(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowPositiveCount(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowSignBitCount(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowSignBitCount(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowNegativeCount(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowNegativeCount(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowFiniteCount(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowFiniteCount(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowNormalCount(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowNormalCount(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowSubnormalCount(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowSubnormalCount(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
}

pub fn withRowNonFiniteCount(self: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.withRowNonFiniteCount(FrameType(@TypeOf(self)), frameValue(self), names, output_name);
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

pub fn tail(self: anytype, n: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    const count = @min(n, self.rows);
    return sliceRows(self, self.rows - count, self.rows);
}

pub fn sliceRows(self: anytype, start: usize, stop: usize) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.sliceRows(FrameType(@TypeOf(self)), frameValue(self), start, stop);
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

pub fn sampleRowsWithReplacement(self: anytype, count: usize, seed: u64) DeviceDataError!FrameType(@TypeOf(self)) {
    return dataframe_array_mod.sampleRowsWithReplacement(FrameType(@TypeOf(self)), frameValue(self), count, seed);
}

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
