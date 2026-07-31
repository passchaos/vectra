//! Arithmetic and comparison helpers for tagged device columns.

const array_mod = @import("array.zig");
const options_mod = @import("dataframe_options.zig");
const std = @import("std");

const DeviceColumnBinaryOp = options_mod.DeviceColumnBinaryOp;
const DeviceColumnCompareOp = options_mod.DeviceColumnCompareOp;

fn columnValue(self: anytype) switch (@typeInfo(@TypeOf(self))) {
    .pointer => |ptr| ptr.child,
    else => @TypeOf(self),
} {
    return switch (@typeInfo(@TypeOf(self))) {
        .pointer => self.*,
        else => self,
    };
}

fn ColumnType(comptime Self: type) type {
    return switch (@typeInfo(Self)) {
        .pointer => |ptr| ptr.child,
        else => Self,
    };
}

fn castFloatLeakyReluSlopeToInt(comptime Dst: type, value: anytype) array_mod.ArrayError!Dst {
    const as_f128: f128 = @floatCast(value);
    if (!std.math.isFinite(as_f128) or @trunc(as_f128) != as_f128) return error.TypeUnsupported;
    const min_value: f128 = @floatFromInt(std.math.minInt(Dst));
    const max_value: f128 = @floatFromInt(std.math.maxInt(Dst));
    if (as_f128 < min_value or as_f128 > max_value) return error.TypeUnsupported;
    return @intFromFloat(as_f128);
}

fn castLeakyReluSlope(comptime Src: type, comptime Dst: type, value: Src) array_mod.ArrayError!Dst {
    if (comptime Dst == array_mod.BFloat16) {
        return switch (@typeInfo(Src)) {
            .int, .comptime_int => array_mod.BFloat16.fromF32(@floatFromInt(value)),
            .float, .comptime_float => array_mod.BFloat16.fromF32(@floatCast(value)),
            .@"struct" => if (comptime Src == array_mod.BFloat16) value else error.TypeUnsupported,
            else => error.TypeUnsupported,
        };
    }
    return switch (@typeInfo(Dst)) {
        .int => switch (@typeInfo(Src)) {
            .int, .comptime_int => std.math.cast(Dst, value) orelse error.TypeUnsupported,
            // Integer leaky-ReLU columns keep their dtype.  Accept a float slope
            // only when it is exactly representable as that integer type so a
            // lazy DeviceScalar cannot silently truncate a fractional slope.
            .float, .comptime_float => castFloatLeakyReluSlopeToInt(Dst, value),
            .@"struct" => if (comptime Src == array_mod.BFloat16) castFloatLeakyReluSlopeToInt(Dst, value.toF32()) else error.TypeUnsupported,
            else => error.TypeUnsupported,
        },
        .float => switch (@typeInfo(Src)) {
            .int, .comptime_int => @floatFromInt(value),
            .float, .comptime_float => @floatCast(value),
            .@"struct" => if (comptime Src == array_mod.BFloat16) @floatCast(value.toF32()) else error.TypeUnsupported,
            else => error.TypeUnsupported,
        },
        else => error.TypeUnsupported,
    };
}

pub fn abs(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.abs()),
    };
}

pub fn neg(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .u8, .u16, .u32, .u64, .usize => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.neg()),
    };
}

pub fn square(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.square()),
    };
}

pub fn reciprocal(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.reciprocal()),
    };
}

pub fn sign(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.sign()),
    };
}

pub fn sqrt(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.sqrt()),
    };
}

pub fn rsqrt(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.rsqrt()),
    };
}

pub fn cbrt(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.cbrt()),
    };
}

pub fn floor(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.floor()),
    };
}

pub fn ceil(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.ceil()),
    };
}

pub fn round(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.round()),
    };
}

pub fn trunc(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.trunc()),
    };
}

pub fn deg2rad(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.deg2rad()),
    };
}

pub fn rad2deg(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.rad2deg()),
    };
}

pub fn expit(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.expit()),
    };
}

pub fn logit(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .f16, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.logit()),
    };
}

pub fn softplus(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .bf16, .f16, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.softplus()),
    };
}

pub fn logsigmoid(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .f16, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.logsigmoid()),
    };
}

pub fn relu(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.relu()),
    };
}

pub fn leakyRelu(self: anytype, comptime T: type, negative_slope: T) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.leakyRelu(try castLeakyReluSlope(T, @TypeOf(typed).Scalar, negative_slope))),
    };
}

pub fn leakyReluWithDeviceScalar(self: anytype, scalar: options_mod.DeviceScalar) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    return switch (scalar) {
        inline .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .bf16, .f16, .f32, .f64 => |value| leakyRelu(self, @TypeOf(value), value),
        .bool, .c64, .c128 => error.TypeUnsupported,
    };
}

pub fn relu6(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.relu6()),
    };
}

pub fn softsign(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .bf16, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.softsign()),
    };
}

pub fn hardsigmoid(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.hardsigmoid()),
    };
}

pub fn hardswish(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.hardswish()),
    };
}

pub fn silu(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.silu()),
    };
}

pub fn swish(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.swish()),
    };
}

pub fn mish(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .f16, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.mish()),
    };
}

pub fn gelu(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.gelu()),
    };
}

pub fn selu(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.selu()),
    };
}

pub fn exp(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.exp()),
    };
}

pub fn exp2(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.exp2()),
    };
}

pub fn expm1(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.expm1()),
    };
}

pub fn sin(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.sin()),
    };
}

pub fn cos(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.cos()),
    };
}

pub fn tan(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.tan()),
    };
}

pub fn asin(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.asin()),
    };
}

pub fn acos(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.acos()),
    };
}

pub fn atan(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.atan()),
    };
}

pub fn sinh(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.sinh()),
    };
}

pub fn cosh(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.cosh()),
    };
}

pub fn tanh(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.tanh()),
    };
}

pub fn asinh(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.asinh()),
    };
}

pub fn acosh(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.acosh()),
    };
}

pub fn atanh(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.atanh()),
    };
}

pub fn log(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.log()),
    };
}

pub fn log1p(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.log1p()),
    };
}

pub fn lgamma(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.lgamma()),
    };
}

pub fn sinc(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.sinc()),
    };
}

pub fn log2(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.log2()),
    };
}

pub fn log10(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.log10()),
    };
}

pub fn binary(self: anytype, other: ColumnType(@TypeOf(self)), op: DeviceColumnBinaryOp) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    if (value.dtype() != other.dtype()) return error.TypeUnsupported;
    if (!value.device().sameDevice(other.device())) return error.InvalidDevice;
    return switch (value) {
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.binary(@field(other, @tagName(tag)), op)),
    };
}

pub fn add(self: anytype, other: ColumnType(@TypeOf(self))) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    return binary(self, other, .add);
}

pub fn sub(self: anytype, other: ColumnType(@TypeOf(self))) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    return binary(self, other, .sub);
}

pub fn mul(self: anytype, other: ColumnType(@TypeOf(self))) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    return binary(self, other, .mul);
}

pub fn div(self: anytype, other: ColumnType(@TypeOf(self))) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    return binary(self, other, .div);
}

pub fn binaryScalar(self: anytype, comptime T: type, scalar: T, op: DeviceColumnBinaryOp) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    if (value.dtype() != array_mod.DType.of(T)) return error.TypeUnsupported;
    const tag = comptime array_mod.DType.of(T);
    return @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try @field(value, @tagName(tag)).binaryScalar(scalar, op));
}

pub fn addScalar(self: anytype, comptime T: type, scalar: T) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    return binaryScalar(self, T, scalar, .add);
}

pub fn subScalar(self: anytype, comptime T: type, scalar: T) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    return binaryScalar(self, T, scalar, .sub);
}

pub fn mulScalar(self: anytype, comptime T: type, scalar: T) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    return binaryScalar(self, T, scalar, .mul);
}

pub fn divScalar(self: anytype, comptime T: type, scalar: T) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    return binaryScalar(self, T, scalar, .div);
}

pub fn compare(self: anytype, other: ColumnType(@TypeOf(self)), op: DeviceColumnCompareOp) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    if (value.dtype() != other.dtype()) return error.TypeUnsupported;
    if (!value.device().sameDevice(other.device())) return error.InvalidDevice;
    return switch (value) {
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

pub fn equal(self: anytype, other: ColumnType(@TypeOf(self))) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    return compare(self, other, .eq);
}

pub fn notEqual(self: anytype, other: ColumnType(@TypeOf(self))) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    return compare(self, other, .ne);
}

pub fn greater(self: anytype, other: ColumnType(@TypeOf(self))) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    return compare(self, other, .gt);
}

pub fn greaterEqual(self: anytype, other: ColumnType(@TypeOf(self))) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    return compare(self, other, .ge);
}

pub fn less(self: anytype, other: ColumnType(@TypeOf(self))) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    return compare(self, other, .lt);
}

pub fn lessEqual(self: anytype, other: ColumnType(@TypeOf(self))) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    return compare(self, other, .le);
}

pub fn compareScalar(self: anytype, comptime T: type, scalar: T, op: DeviceColumnCompareOp) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    if (value.dtype() != array_mod.DType.of(T)) return error.TypeUnsupported;
    const tag = comptime array_mod.DType.of(T);
    return .{ .bool = try @field(value, @tagName(tag)).compareScalar(scalar, op) };
}

pub fn equalScalar(self: anytype, comptime T: type, scalar: T) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    return compareScalar(self, T, scalar, .eq);
}

pub fn notEqualScalar(self: anytype, comptime T: type, scalar: T) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    return compareScalar(self, T, scalar, .ne);
}

pub fn greaterScalar(self: anytype, comptime T: type, scalar: T) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    return compareScalar(self, T, scalar, .gt);
}

pub fn greaterEqualScalar(self: anytype, comptime T: type, scalar: T) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    return compareScalar(self, T, scalar, .ge);
}

pub fn lessScalar(self: anytype, comptime T: type, scalar: T) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    return compareScalar(self, T, scalar, .lt);
}

pub fn lessEqualScalar(self: anytype, comptime T: type, scalar: T) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    return compareScalar(self, T, scalar, .le);
}
