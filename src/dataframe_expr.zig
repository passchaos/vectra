//! Device dataframe column expression helpers.
//!
//! These helpers keep expression dispatch and nullable predicate semantics out
//! of the public dataframe facade while preserving the same `DeviceDataFrame`
//! method surface.

const std = @import("std");
const array_mod = @import("array.zig");
const options_mod = @import("dataframe_options.zig");
const series_mod = @import("series.zig");

const DeviceColumnBinaryOp = options_mod.DeviceColumnBinaryOp;
const DeviceColumnCompareOp = options_mod.DeviceColumnCompareOp;
const DeviceColumnLogicalOp = options_mod.DeviceColumnLogicalOp;
const DeviceScalar = options_mod.DeviceScalar;
const DeviceDataError = series_mod.DataError || array_mod.ArrayError;

pub fn unaryColumnAbs(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.abs();
}

pub fn unaryColumnNeg(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.neg();
}

pub fn unaryColumnSquare(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.square();
}

pub fn unaryColumnReciprocal(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.reciprocal();
}

pub fn unaryColumnSign(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.sign();
}

pub fn unaryColumnSqrt(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.sqrt();
}

pub fn unaryColumnRsqrt(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.rsqrt();
}

pub fn unaryColumnCbrt(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.cbrt();
}

pub fn unaryColumnFloor(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.floor();
}

pub fn unaryColumnCeil(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.ceil();
}

pub fn unaryColumnRound(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.round();
}

pub fn unaryColumnTrunc(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.trunc();
}

pub fn unaryColumnDeg2rad(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.deg2rad();
}

pub fn unaryColumnRad2deg(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.rad2deg();
}

pub fn unaryColumnExpit(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.expit();
}

pub fn unaryColumnLogit(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.logit();
}

pub fn unaryColumnSoftplus(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.softplus();
}

pub fn unaryColumnLogsigmoid(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.logsigmoid();
}

pub fn unaryColumnRelu(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.relu();
}

pub fn unaryColumnLeakyRelu(frame: anytype, name: []const u8, comptime T: type, negative_slope: T) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.leakyRelu(T, negative_slope);
}

pub fn unaryColumnLeakyReluWithDeviceScalar(frame: anytype, name: []const u8, negative_slope: DeviceScalar) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.leakyReluWithDeviceScalar(negative_slope);
}

pub fn unaryColumnRelu6(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.relu6();
}

pub fn unaryColumnPowScalar(frame: anytype, name: []const u8, comptime T: type, exponent: T) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.powScalar(T, exponent);
}

pub fn unaryColumnPowWithDeviceScalar(frame: anytype, name: []const u8, exponent: DeviceScalar) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.powWithDeviceScalar(exponent);
}

pub fn unaryColumnFloorDivScalar(frame: anytype, name: []const u8, comptime T: type, scalar: T) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.floorDivScalar(T, scalar);
}

pub fn unaryColumnFloorDivWithDeviceScalar(frame: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.floorDivWithDeviceScalar(scalar);
}

pub fn unaryColumnModScalar(frame: anytype, name: []const u8, comptime T: type, scalar: T) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.modScalar(T, scalar);
}

pub fn unaryColumnModWithDeviceScalar(frame: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.modWithDeviceScalar(scalar);
}

pub fn unaryColumnRemainderScalar(frame: anytype, name: []const u8, comptime T: type, scalar: T) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.remainderScalar(T, scalar);
}

pub fn unaryColumnRemainderWithDeviceScalar(frame: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.remainderWithDeviceScalar(scalar);
}

pub fn unaryColumnLogAddExpScalar(frame: anytype, name: []const u8, comptime T: type, scalar: T) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.logAddExpScalar(T, scalar);
}

pub fn unaryColumnLogAddExpWithDeviceScalar(frame: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.logAddExpWithDeviceScalar(scalar);
}

pub fn unaryColumnLogAddExp2Scalar(frame: anytype, name: []const u8, comptime T: type, scalar: T) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.logAddExp2Scalar(T, scalar);
}

pub fn unaryColumnLogAddExp2WithDeviceScalar(frame: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.logAddExp2WithDeviceScalar(scalar);
}

pub fn unaryColumnXlogyScalar(frame: anytype, name: []const u8, comptime T: type, scalar: T) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.xlogyScalar(T, scalar);
}

pub fn unaryColumnXlogyWithDeviceScalar(frame: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.xlogyWithDeviceScalar(scalar);
}

pub fn unaryColumnFmaxScalar(frame: anytype, name: []const u8, comptime T: type, scalar: T) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.fmaxScalar(T, scalar);
}

pub fn unaryColumnFmaxWithDeviceScalar(frame: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.fmaxWithDeviceScalar(scalar);
}

pub fn unaryColumnFminScalar(frame: anytype, name: []const u8, comptime T: type, scalar: T) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.fminScalar(T, scalar);
}

pub fn unaryColumnFminWithDeviceScalar(frame: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.fminWithDeviceScalar(scalar);
}

pub fn unaryColumnHypotScalar(frame: anytype, name: []const u8, comptime T: type, scalar: T) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.hypotScalar(T, scalar);
}

pub fn unaryColumnHypotWithDeviceScalar(frame: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.hypotWithDeviceScalar(scalar);
}

pub fn unaryColumnAtan2Scalar(frame: anytype, name: []const u8, comptime T: type, scalar: T) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.atan2Scalar(T, scalar);
}

pub fn unaryColumnAtan2WithDeviceScalar(frame: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.atan2WithDeviceScalar(scalar);
}

pub fn unaryColumnNextAfterScalar(frame: anytype, name: []const u8, comptime T: type, scalar: T) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.nextAfterScalar(T, scalar);
}

pub fn unaryColumnNextAfterWithDeviceScalar(frame: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.nextAfterWithDeviceScalar(scalar);
}

pub fn unaryColumnCopysignScalar(frame: anytype, name: []const u8, comptime T: type, scalar: T) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.copysignScalar(T, scalar);
}

pub fn unaryColumnCopysignWithDeviceScalar(frame: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.copysignWithDeviceScalar(scalar);
}

pub fn unaryColumnHeavisideScalar(frame: anytype, name: []const u8, comptime T: type, value_at_zero: T) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.heavisideScalar(T, value_at_zero);
}

pub fn unaryColumnHeavisideWithDeviceScalar(frame: anytype, name: []const u8, value_at_zero: DeviceScalar) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.heavisideWithDeviceScalar(value_at_zero);
}

pub fn unaryColumnLdexpScalar(frame: anytype, name: []const u8, exponent: i32) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.ldexpScalar(exponent);
}

pub fn unaryColumnThreshold(frame: anytype, name: []const u8, comptime T: type, threshold_value: T, replacement_value: T) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.threshold(T, threshold_value, replacement_value);
}

pub fn unaryColumnThresholdWithDeviceScalars(frame: anytype, name: []const u8, threshold_value: DeviceScalar, replacement_value: DeviceScalar) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.thresholdWithDeviceScalars(threshold_value, replacement_value);
}

pub fn unaryColumnHardtanh(frame: anytype, name: []const u8, comptime T: type, min_value: T, max_value: T) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.hardtanh(T, min_value, max_value);
}

pub fn unaryColumnHardtanhWithDeviceScalars(frame: anytype, name: []const u8, min_value: DeviceScalar, max_value: DeviceScalar) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.hardtanhWithDeviceScalars(min_value, max_value);
}

pub fn unaryColumnMaximumScalar(frame: anytype, name: []const u8, comptime T: type, scalar: T) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.maximumScalar(T, scalar);
}

pub fn unaryColumnMaximumWithDeviceScalar(frame: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.maximumWithDeviceScalar(scalar);
}

pub fn unaryColumnMinimumScalar(frame: anytype, name: []const u8, comptime T: type, scalar: T) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.minimumScalar(T, scalar);
}

pub fn unaryColumnMinimumWithDeviceScalar(frame: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.minimumWithDeviceScalar(scalar);
}

pub fn unaryColumnClipMin(frame: anytype, name: []const u8, comptime T: type, min_value: T) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.clipMin(T, min_value);
}

pub fn unaryColumnClipMinWithDeviceScalar(frame: anytype, name: []const u8, min_value: DeviceScalar) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.clipMinWithDeviceScalar(min_value);
}

pub fn unaryColumnClipMax(frame: anytype, name: []const u8, comptime T: type, max_value: T) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.clipMax(T, max_value);
}

pub fn unaryColumnClipMaxWithDeviceScalar(frame: anytype, name: []const u8, max_value: DeviceScalar) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.clipMaxWithDeviceScalar(max_value);
}

pub fn unaryColumnHardshrink(frame: anytype, name: []const u8, comptime T: type, lambd: T) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.hardshrink(T, lambd);
}

pub fn unaryColumnHardshrinkWithDeviceScalar(frame: anytype, name: []const u8, lambd: DeviceScalar) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.hardshrinkWithDeviceScalar(lambd);
}

pub fn unaryColumnSoftshrink(frame: anytype, name: []const u8, comptime T: type, lambd: T) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.softshrink(T, lambd);
}

pub fn unaryColumnSoftshrinkWithDeviceScalar(frame: anytype, name: []const u8, lambd: DeviceScalar) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.softshrinkWithDeviceScalar(lambd);
}

pub fn unaryColumnTanhshrink(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.tanhshrink();
}

pub fn unaryColumnElu(frame: anytype, name: []const u8, comptime T: type, alpha: T) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.elu(T, alpha);
}

pub fn unaryColumnEluWithDeviceScalar(frame: anytype, name: []const u8, alpha: DeviceScalar) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.eluWithDeviceScalar(alpha);
}

pub fn unaryColumnCelu(frame: anytype, name: []const u8, comptime T: type, alpha: T) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.celu(T, alpha);
}

pub fn unaryColumnCeluWithDeviceScalar(frame: anytype, name: []const u8, alpha: DeviceScalar) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.celuWithDeviceScalar(alpha);
}

pub fn unaryColumnSoftsign(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.softsign();
}

pub fn unaryColumnHardsigmoid(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.hardsigmoid();
}

pub fn unaryColumnHardswish(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.hardswish();
}

pub fn unaryColumnSilu(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.silu();
}

pub fn unaryColumnSwish(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.swish();
}

pub fn unaryColumnMish(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.mish();
}

pub fn unaryColumnGelu(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.gelu();
}

pub fn unaryColumnSelu(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.selu();
}

pub fn unaryColumnExp(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.exp();
}

pub fn unaryColumnExp2(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.exp2();
}

pub fn unaryColumnExpm1(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.expm1();
}

pub fn unaryColumnSin(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.sin();
}

pub fn unaryColumnCos(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.cos();
}

pub fn unaryColumnTan(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.tan();
}

pub fn unaryColumnAsin(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.asin();
}

pub fn unaryColumnAcos(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.acos();
}

pub fn unaryColumnAtan(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.atan();
}

pub fn unaryColumnSinh(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.sinh();
}

pub fn unaryColumnCosh(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.cosh();
}

pub fn unaryColumnTanh(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.tanh();
}

pub fn unaryColumnAsinh(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.asinh();
}

pub fn unaryColumnAcosh(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.acosh();
}

pub fn unaryColumnAtanh(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.atanh();
}

pub fn unaryColumnLog(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.log();
}

pub fn unaryColumnLog1p(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.log1p();
}

pub fn unaryColumnLgamma(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.lgamma();
}

pub fn unaryColumnSinc(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.sinc();
}

pub fn unaryColumnLog2(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.log2();
}

pub fn unaryColumnLog10(frame: anytype, name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.log10();
}

pub fn binaryColumns(frame: anytype, lhs_name: []const u8, rhs_name: []const u8, op: DeviceColumnBinaryOp) DeviceDataError!@TypeOf(frame.columns[0]) {
    const lhs = try frame.column(lhs_name);
    const rhs = try frame.column(rhs_name);
    return lhs.binary(rhs.*, op);
}

pub fn binaryColumnScalar(frame: anytype, name: []const u8, comptime T: type, scalar: T, op: DeviceColumnBinaryOp) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.binaryScalar(T, scalar, op);
}

pub fn binaryColumnScalarWithDeviceScalar(frame: anytype, name: []const u8, scalar: DeviceScalar, op: DeviceColumnBinaryOp) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
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

pub fn lerpColumnsScalar(frame: anytype, lhs_name: []const u8, rhs_name: []const u8, comptime T: type, weight: T) DeviceDataError!@TypeOf(frame.columns[0]) {
    const lhs = try frame.column(lhs_name);
    const rhs = try frame.column(rhs_name);
    return lhs.lerpScalar(rhs.*, T, weight);
}

pub fn lerpColumnsWithDeviceScalar(frame: anytype, lhs_name: []const u8, rhs_name: []const u8, weight: DeviceScalar) DeviceDataError!@TypeOf(frame.columns[0]) {
    const lhs = try frame.column(lhs_name);
    const rhs = try frame.column(rhs_name);
    return lhs.lerpWithDeviceScalar(rhs.*, weight);
}

pub fn addcmulColumnsScalar(frame: anytype, base_name: []const u8, input1_name: []const u8, input2_name: []const u8, comptime T: type, value: T) DeviceDataError!@TypeOf(frame.columns[0]) {
    const base = try frame.column(base_name);
    const input1 = try frame.column(input1_name);
    const input2 = try frame.column(input2_name);
    return base.addcmulScalar(input1.*, input2.*, T, value);
}

pub fn addcmulColumnsWithDeviceScalar(frame: anytype, base_name: []const u8, input1_name: []const u8, input2_name: []const u8, value: DeviceScalar) DeviceDataError!@TypeOf(frame.columns[0]) {
    const base = try frame.column(base_name);
    const input1 = try frame.column(input1_name);
    const input2 = try frame.column(input2_name);
    return base.addcmulWithDeviceScalar(input1.*, input2.*, value);
}

pub fn addcdivColumnsScalar(frame: anytype, base_name: []const u8, input1_name: []const u8, input2_name: []const u8, comptime T: type, value: T) DeviceDataError!@TypeOf(frame.columns[0]) {
    const base = try frame.column(base_name);
    const input1 = try frame.column(input1_name);
    const input2 = try frame.column(input2_name);
    return base.addcdivScalar(input1.*, input2.*, T, value);
}

pub fn addcdivColumnsWithDeviceScalar(frame: anytype, base_name: []const u8, input1_name: []const u8, input2_name: []const u8, value: DeviceScalar) DeviceDataError!@TypeOf(frame.columns[0]) {
    const base = try frame.column(base_name);
    const input1 = try frame.column(input1_name);
    const input2 = try frame.column(input2_name);
    return base.addcdivWithDeviceScalar(input1.*, input2.*, value);
}

pub fn clipArrayColumns(frame: anytype, input_name: []const u8, min_name: []const u8, max_name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const input = try frame.column(input_name);
    const min_values = try frame.column(min_name);
    const max_values = try frame.column(max_name);
    return input.clipArray(min_values.*, max_values.*);
}

pub fn whereColumnScalar(frame: anytype, input_name: []const u8, mask_name: []const u8, comptime T: type, other_value: T) DeviceDataError!@TypeOf(frame.columns[0]) {
    const input = try frame.column(input_name);
    const mask = try frame.column(mask_name);
    return input.whereScalar(mask.*, T, other_value);
}

pub fn whereColumnWithDeviceScalar(frame: anytype, input_name: []const u8, mask_name: []const u8, other_value: DeviceScalar) DeviceDataError!@TypeOf(frame.columns[0]) {
    const input = try frame.column(input_name);
    const mask = try frame.column(mask_name);
    return input.whereWithDeviceScalar(mask.*, other_value);
}

pub fn whereColumns(frame: anytype, input_name: []const u8, mask_name: []const u8, other_name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const input = try frame.column(input_name);
    const mask = try frame.column(mask_name);
    const other = try frame.column(other_name);
    return input.whereColumn(mask.*, other.*);
}

pub fn isinColumns(frame: anytype, input_name: []const u8, test_name: []const u8, invert: bool) DeviceDataError!@TypeOf(frame.columns[0]) {
    const input = try frame.column(input_name);
    const test_elements = try frame.column(test_name);
    return input.isinColumn(test_elements.*, invert);
}

pub fn maskedPutColumnScalar(frame: anytype, input_name: []const u8, mask_name: []const u8, comptime T: type, value: T) DeviceDataError!@TypeOf(frame.columns[0]) {
    const input = try frame.column(input_name);
    const mask = try frame.column(mask_name);
    return input.maskedPutScalar(mask.*, T, value);
}

pub fn maskedPutColumnWithDeviceScalar(frame: anytype, input_name: []const u8, mask_name: []const u8, value: DeviceScalar) DeviceDataError!@TypeOf(frame.columns[0]) {
    const input = try frame.column(input_name);
    const mask = try frame.column(mask_name);
    return input.maskedPutWithDeviceScalar(mask.*, value);
}

pub fn putFlatColumnScalar(frame: anytype, input_name: []const u8, row_indices: []const usize, comptime T: type, value: T) DeviceDataError!@TypeOf(frame.columns[0]) {
    const input = try frame.column(input_name);
    return input.putFlatScalar(row_indices, T, value);
}

pub fn putFlatColumnWithDeviceScalar(frame: anytype, input_name: []const u8, row_indices: []const usize, value: DeviceScalar) DeviceDataError!@TypeOf(frame.columns[0]) {
    const input = try frame.column(input_name);
    return input.putFlatWithDeviceScalar(row_indices, value);
}

pub fn putFlatColumns(frame: anytype, input_name: []const u8, row_indices: []const usize, value_name: []const u8) DeviceDataError!@TypeOf(frame.columns[0]) {
    const input = try frame.column(input_name);
    const values = try frame.column(value_name);
    return input.putFlat(row_indices, values.*);
}

pub fn putFlatColumnScalarMode(frame: anytype, input_name: []const u8, row_indices: []const usize, comptime T: type, value: T, mode: array_mod.IndexMode) DeviceDataError!@TypeOf(frame.columns[0]) {
    const input = try frame.column(input_name);
    return input.putFlatScalarMode(row_indices, T, value, mode);
}

pub fn putFlatColumnModeWithDeviceScalar(frame: anytype, input_name: []const u8, row_indices: []const usize, value: DeviceScalar, mode: array_mod.IndexMode) DeviceDataError!@TypeOf(frame.columns[0]) {
    const input = try frame.column(input_name);
    return input.putFlatModeWithDeviceScalar(row_indices, value, mode);
}

pub fn putFlatColumnScalarSigned(frame: anytype, input_name: []const u8, row_indices: []const isize, comptime T: type, value: T) DeviceDataError!@TypeOf(frame.columns[0]) {
    const input = try frame.column(input_name);
    return input.putFlatScalarSigned(row_indices, T, value);
}

pub fn putFlatColumnSignedWithDeviceScalar(frame: anytype, input_name: []const u8, row_indices: []const isize, value: DeviceScalar) DeviceDataError!@TypeOf(frame.columns[0]) {
    const input = try frame.column(input_name);
    return input.putFlatSignedWithDeviceScalar(row_indices, value);
}

pub fn compareColumns(frame: anytype, lhs_name: []const u8, rhs_name: []const u8, op: DeviceColumnCompareOp) DeviceDataError!@TypeOf(frame.columns[0]) {
    const lhs = try frame.column(lhs_name);
    const rhs = try frame.column(rhs_name);
    return lhs.compare(rhs.*, op);
}

pub fn compareColumnScalar(frame: anytype, name: []const u8, comptime T: type, scalar: T, op: DeviceColumnCompareOp) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.compareScalar(T, scalar, op);
}

pub fn compareColumnScalarWithDeviceScalar(frame: anytype, name: []const u8, scalar: DeviceScalar, op: DeviceColumnCompareOp) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
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

pub fn iscloseColumnScalar(
    frame: anytype,
    name: []const u8,
    comptime T: type,
    scalar: T,
    rtol: T,
    atol: T,
    equal_nan: bool,
) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.iscloseScalar(T, scalar, rtol, atol, equal_nan);
}

pub fn iscloseColumnWithDeviceScalars(
    frame: anytype,
    name: []const u8,
    scalar: DeviceScalar,
    rtol: DeviceScalar,
    atol: DeviceScalar,
    equal_nan: bool,
) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return col.iscloseWithDeviceScalars(scalar, rtol, atol, equal_nan);
}

pub fn allcloseColumnScalar(
    frame: anytype,
    name: []const u8,
    comptime T: type,
    scalar: T,
    rtol: T,
    atol: T,
    equal_nan: bool,
) DeviceDataError!bool {
    const col = try frame.column(name);
    return col.allcloseScalar(T, scalar, rtol, atol, equal_nan);
}

pub fn allcloseColumnWithDeviceScalars(
    frame: anytype,
    name: []const u8,
    scalar: DeviceScalar,
    rtol: DeviceScalar,
    atol: DeviceScalar,
    equal_nan: bool,
) DeviceDataError!bool {
    const col = try frame.column(name);
    return col.allcloseWithDeviceScalars(scalar, rtol, atol, equal_nan);
}

pub fn countNonzeroColumn(frame: anytype, name: []const u8) DeviceDataError!usize {
    const col = try frame.column(name);
    return col.countNonzero();
}

pub fn zeroCountColumn(frame: anytype, name: []const u8) DeviceDataError!usize {
    const col = try frame.column(name);
    return col.validCount() - try col.countNonzero();
}

pub fn countZeroColumn(frame: anytype, name: []const u8) DeviceDataError!usize {
    return zeroCountColumn(frame, name);
}

pub fn nanCountColumn(frame: anytype, name: []const u8) DeviceDataError!usize {
    const col = try frame.column(name);
    return col.countNan();
}

pub fn positiveZeroCountColumn(frame: anytype, name: []const u8) DeviceDataError!usize {
    const col = try frame.column(name);
    return col.countPositiveZero();
}

pub fn negativeZeroCountColumn(frame: anytype, name: []const u8) DeviceDataError!usize {
    const col = try frame.column(name);
    return col.countNegativeZero();
}

pub fn infCountColumn(frame: anytype, name: []const u8) DeviceDataError!usize {
    const col = try frame.column(name);
    return col.countInf();
}

pub fn positiveInfCountColumn(frame: anytype, name: []const u8) DeviceDataError!usize {
    const col = try frame.column(name);
    return col.countPositiveInf();
}

pub fn negativeInfCountColumn(frame: anytype, name: []const u8) DeviceDataError!usize {
    const col = try frame.column(name);
    return col.countNegativeInf();
}

pub fn finiteCountColumn(frame: anytype, name: []const u8) DeviceDataError!usize {
    const col = try frame.column(name);
    return col.countFinite();
}

pub fn nonFiniteCountColumn(frame: anytype, name: []const u8) DeviceDataError!usize {
    const col = try frame.column(name);
    return col.countNonFinite();
}

fn ratioFromValidCount(count: usize, valid_count: usize) DeviceScalar {
    if (valid_count == 0) return .{ .f64 = std.math.nan(f64) };
    return .{ .f64 = @as(f64, @floatFromInt(count)) / @as(f64, @floatFromInt(valid_count)) };
}

pub fn zeroRatioColumn(frame: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    const col = try frame.column(name);
    const valid_count = col.validCount();
    return ratioFromValidCount(valid_count - try col.countNonzero(), valid_count);
}

pub fn nonzeroRatioColumn(frame: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    const col = try frame.column(name);
    return ratioFromValidCount(try col.countNonzero(), col.validCount());
}

pub fn positiveZeroRatioColumn(frame: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    const col = try frame.column(name);
    return ratioFromValidCount(try col.countPositiveZero(), col.validCount());
}

pub fn negativeZeroRatioColumn(frame: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    const col = try frame.column(name);
    return ratioFromValidCount(try col.countNegativeZero(), col.validCount());
}

pub fn nanRatioColumn(frame: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    const col = try frame.column(name);
    return ratioFromValidCount(try col.countNan(), col.validCount());
}

pub fn infRatioColumn(frame: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    const col = try frame.column(name);
    return ratioFromValidCount(try col.countInf(), col.validCount());
}

pub fn positiveInfRatioColumn(frame: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    const col = try frame.column(name);
    return ratioFromValidCount(try col.countPositiveInf(), col.validCount());
}

pub fn negativeInfRatioColumn(frame: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    const col = try frame.column(name);
    return ratioFromValidCount(try col.countNegativeInf(), col.validCount());
}

pub fn finiteRatioColumn(frame: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    const col = try frame.column(name);
    return ratioFromValidCount(try col.countFinite(), col.validCount());
}

pub fn nonFiniteRatioColumn(frame: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    const col = try frame.column(name);
    return ratioFromValidCount(try col.countNonFinite(), col.validCount());
}

pub fn firstValidIndexColumn(frame: anytype, name: []const u8) DeviceDataError!?usize {
    const col = try frame.column(name);
    return col.firstValidIndex();
}

pub fn lastValidIndexColumn(frame: anytype, name: []const u8) DeviceDataError!?usize {
    const col = try frame.column(name);
    return col.lastValidIndex();
}

pub fn firstNullIndexColumn(frame: anytype, name: []const u8) DeviceDataError!?usize {
    const col = try frame.column(name);
    return col.firstNullIndex();
}

pub fn lastNullIndexColumn(frame: anytype, name: []const u8) DeviceDataError!?usize {
    const col = try frame.column(name);
    return col.lastNullIndex();
}

pub fn countDistinctColumn(frame: anytype, name: []const u8) DeviceDataError!usize {
    const col = try frame.column(name);
    return col.countDistinct();
}

pub fn nUniqueColumn(frame: anytype, name: []const u8) DeviceDataError!usize {
    const col = try frame.column(name);
    return col.nUnique();
}

pub fn nullCountColumn(frame: anytype, name: []const u8) DeviceDataError!usize {
    const col = try frame.column(name);
    return col.nullCount();
}

pub fn validCountColumn(frame: anytype, name: []const u8) DeviceDataError!usize {
    const col = try frame.column(name);
    return col.validCount();
}

pub fn nullRatioColumn(frame: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    const col = try frame.column(name);
    return .{ .f64 = col.nullRatio() };
}

pub fn validRatioColumn(frame: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    const col = try frame.column(name);
    return .{ .f64 = col.validRatio() };
}

pub fn modeColumn(frame: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    const col = try frame.column(name);
    return col.mode();
}

pub fn sumColumn(frame: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    const col = try frame.column(name);
    return col.sum();
}

pub fn prodColumn(frame: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    const col = try frame.column(name);
    return col.prod();
}

pub fn meanColumn(frame: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    const col = try frame.column(name);
    return col.mean();
}

pub fn quantileColumn(frame: anytype, name: []const u8, q: f64) DeviceDataError!DeviceScalar {
    const col = try frame.column(name);
    return col.quantile(q);
}

pub fn medianColumn(frame: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    const col = try frame.column(name);
    return col.median();
}

pub fn varianceColumn(frame: anytype, name: []const u8, correction: f64) DeviceDataError!DeviceScalar {
    const col = try frame.column(name);
    return col.variance(correction);
}

pub fn stddevColumn(frame: anytype, name: []const u8, correction: f64) DeviceDataError!DeviceScalar {
    const col = try frame.column(name);
    return col.stddev(correction);
}

pub fn semColumn(frame: anytype, name: []const u8, correction: f64) DeviceDataError!DeviceScalar {
    const col = try frame.column(name);
    return col.sem(correction);
}

pub fn cvColumn(frame: anytype, name: []const u8, correction: f64) DeviceDataError!DeviceScalar {
    const col = try frame.column(name);
    return col.cv(correction);
}

pub fn skewnessColumn(frame: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    const col = try frame.column(name);
    return col.skewness();
}

pub fn kurtosisColumn(frame: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    const col = try frame.column(name);
    return col.kurtosis();
}

pub fn meanAbsColumn(frame: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    const col = try frame.column(name);
    return col.meanAbs();
}

pub fn rmsColumn(frame: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    const col = try frame.column(name);
    return col.rms();
}

pub fn l1NormColumn(frame: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    const col = try frame.column(name);
    return col.l1Norm();
}

pub fn l2NormColumn(frame: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    const col = try frame.column(name);
    return col.l2Norm();
}

pub fn geometricMeanColumn(frame: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    const col = try frame.column(name);
    return col.geometricMean();
}

pub fn harmonicMeanColumn(frame: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    const col = try frame.column(name);
    return col.harmonicMean();
}

pub fn madColumn(frame: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    const col = try frame.column(name);
    return col.mad();
}

pub fn iqrColumn(frame: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    const col = try frame.column(name);
    return col.iqr();
}

pub fn minColumn(frame: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    const col = try frame.column(name);
    return col.min();
}

pub fn maxColumn(frame: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    const col = try frame.column(name);
    return col.max();
}

pub fn ptpColumn(frame: anytype, name: []const u8) DeviceDataError!DeviceScalar {
    const col = try frame.column(name);
    return col.ptp();
}

pub fn argminColumn(frame: anytype, name: []const u8) DeviceDataError!usize {
    const col = try frame.column(name);
    return col.argmin();
}

pub fn argmaxColumn(frame: anytype, name: []const u8) DeviceDataError!usize {
    const col = try frame.column(name);
    return col.argmax();
}

pub fn anyColumn(frame: anytype, name: []const u8) DeviceDataError!bool {
    const col = try frame.column(name);
    return col.any();
}

pub fn allColumn(frame: anytype, name: []const u8) DeviceDataError!bool {
    const col = try frame.column(name);
    return col.all();
}

pub fn countTrueColumn(frame: anytype, name: []const u8) DeviceDataError!usize {
    const col = try frame.column(name);
    return col.countTrue();
}

pub fn countFalseColumn(frame: anytype, name: []const u8) DeviceDataError!usize {
    const col = try frame.column(name);
    return col.countFalse();
}

pub fn logicalColumnScalar(frame: anytype, name: []const u8, scalar: bool, op: DeviceColumnLogicalOp) DeviceDataError!@TypeOf(frame.columns[0]) {
    const col = try frame.column(name);
    return switch (op) {
        .@"and" => col.logicalAndScalar(scalar),
        .@"or" => col.logicalOrScalar(scalar),
        .xor => col.logicalXorScalar(scalar),
    };
}

pub fn logicalColumns(frame: anytype, lhs_name: []const u8, rhs_name: []const u8, op: DeviceColumnLogicalOp) DeviceDataError!@TypeOf(frame.columns[0]) {
    const lhs = try frame.column(lhs_name);
    const rhs = try frame.column(rhs_name);
    return lhs.logical(rhs.*, op);
}

pub fn filterColumnMask(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    mask: anytype,
) DeviceDataError!DeviceDataFrame {
    const typed_mask = switch (mask) {
        .bool => |typed| typed,
        else => return error.TypeMismatch,
    };
    if (!typed_mask.device().sameDevice(frame.device)) return error.InvalidDevice;
    if (typed_mask.len() != frame.rows) return error.LengthMismatch;
    const host_values = try typed_mask.values.toOwnedSlice(frame.allocator);
    defer frame.allocator.free(host_values);
    if (typed_mask.validity) |validity_array| {
        const host_validity = try validity_array.toOwnedSlice(frame.allocator);
        defer frame.allocator.free(host_validity);
        const host_mask = try frame.allocator.alloc(bool, frame.rows);
        defer frame.allocator.free(host_mask);
        for (host_values, host_validity, host_mask) |value, valid, *slot| {
            // Null predicate rows follow dataframe query semantics: they do not
            // select the row, matching Arrow/Polars-style filter behavior.
            slot.* = valid and value;
        }
        return frame.filter(host_mask);
    }
    return frame.filter(host_values);
}

pub fn filterColumn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
) DeviceDataError!DeviceDataFrame {
    const mask = try frame.column(name);
    return filterColumnMask(DeviceDataFrame, frame, mask.*);
}

pub fn dropRowsByColumnMask(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
) DeviceDataError!DeviceDataFrame {
    const mask = try frame.column(name);
    const typed_mask = switch (mask.*) {
        .bool => |typed| typed,
        else => return error.TypeMismatch,
    };
    if (!typed_mask.device().sameDevice(frame.device)) return error.InvalidDevice;
    if (typed_mask.len() != frame.rows) return error.LengthMismatch;
    const host_values = try typed_mask.values.toOwnedSlice(frame.allocator);
    defer frame.allocator.free(host_values);
    const host_validity = if (typed_mask.validity) |validity_array| try validity_array.toOwnedSlice(frame.allocator) else null;
    defer if (host_validity) |validity| frame.allocator.free(validity);
    const keep_mask = try frame.allocator.alloc(bool, frame.rows);
    defer frame.allocator.free(keep_mask);
    for (host_values, keep_mask, 0..) |value, *slot, row| {
        // Match filter semantics for nullable predicates: a null mask row is
        // treated as false, so it is not dropped by this complementary API.
        const should_drop = if (host_validity) |validity| validity[row] and value else value;
        slot.* = !should_drop;
    }
    return frame.filter(keep_mask);
}
