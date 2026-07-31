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
