//! Arithmetic and comparison helpers for tagged device columns.

const array_mod = @import("array.zig");
const options_mod = @import("dataframe_options.zig");
const std = @import("std");

const DeviceColumnBinaryOp = options_mod.DeviceColumnBinaryOp;
const DeviceColumnCompareOp = options_mod.DeviceColumnCompareOp;
const DeviceColumnLogicalOp = options_mod.DeviceColumnLogicalOp;

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

fn castFloatScalarToInt(comptime Dst: type, value: anytype) array_mod.ArrayError!Dst {
    const as_f128: f128 = @floatCast(value);
    if (!std.math.isFinite(as_f128) or @trunc(as_f128) != as_f128) return error.TypeUnsupported;
    const min_value: f128 = @floatFromInt(std.math.minInt(Dst));
    const max_value: f128 = @floatFromInt(std.math.maxInt(Dst));
    if (as_f128 < min_value or as_f128 > max_value) return error.TypeUnsupported;
    return @intFromFloat(as_f128);
}

fn castNumericScalar(comptime Src: type, comptime Dst: type, value: Src) array_mod.ArrayError!Dst {
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
            // Parameterized dataframe ops keep the column dtype.  Accept a
            // float scalar for an integer column only when it is exactly
            // representable as that integer type so a lazy DeviceScalar cannot
            // silently truncate fractional parameters.
            .float, .comptime_float => castFloatScalarToInt(Dst, value),
            .@"struct" => if (comptime Src == array_mod.BFloat16) castFloatScalarToInt(Dst, value.toF32()) else error.TypeUnsupported,
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

fn castDeviceScalar(comptime Dst: type, scalar: options_mod.DeviceScalar) array_mod.ArrayError!Dst {
    return switch (scalar) {
        inline .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .bf16, .f16, .f32, .f64 => |value| castNumericScalar(@TypeOf(value), Dst, value),
        .bool, .c64, .c128 => error.TypeUnsupported,
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
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.leakyRelu(try castNumericScalar(T, @TypeOf(typed).Scalar, negative_slope))),
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

pub fn powScalar(self: anytype, comptime T: type, exponent: T) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.powScalar(try castNumericScalar(T, @TypeOf(typed).Scalar, exponent))),
    };
}

pub fn powWithDeviceScalar(self: anytype, exponent: options_mod.DeviceScalar) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.powScalar(try castDeviceScalar(@TypeOf(typed).Scalar, exponent))),
    };
}

pub fn floorDivScalar(self: anytype, comptime T: type, scalar: T) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.floorDivScalar(try castNumericScalar(T, @TypeOf(typed).Scalar, scalar))),
    };
}

pub fn floorDivWithDeviceScalar(self: anytype, scalar: options_mod.DeviceScalar) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.floorDivScalar(try castDeviceScalar(@TypeOf(typed).Scalar, scalar))),
    };
}

pub fn modScalar(self: anytype, comptime T: type, scalar: T) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.modScalar(try castNumericScalar(T, @TypeOf(typed).Scalar, scalar))),
    };
}

pub fn modWithDeviceScalar(self: anytype, scalar: options_mod.DeviceScalar) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.modScalar(try castDeviceScalar(@TypeOf(typed).Scalar, scalar))),
    };
}

pub fn remainderScalar(self: anytype, comptime T: type, scalar: T) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    return modScalar(self, T, scalar);
}

pub fn remainderWithDeviceScalar(self: anytype, scalar: options_mod.DeviceScalar) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    return modWithDeviceScalar(self, scalar);
}

pub fn logAddExpScalar(self: anytype, comptime T: type, scalar: T) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.logAddExpScalar(try castNumericScalar(T, @TypeOf(typed).Scalar, scalar))),
    };
}

pub fn logAddExpWithDeviceScalar(self: anytype, scalar: options_mod.DeviceScalar) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.logAddExpScalar(try castDeviceScalar(@TypeOf(typed).Scalar, scalar))),
    };
}

pub fn logAddExp2Scalar(self: anytype, comptime T: type, scalar: T) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.logAddExp2Scalar(try castNumericScalar(T, @TypeOf(typed).Scalar, scalar))),
    };
}

pub fn logAddExp2WithDeviceScalar(self: anytype, scalar: options_mod.DeviceScalar) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.logAddExp2Scalar(try castDeviceScalar(@TypeOf(typed).Scalar, scalar))),
    };
}

pub fn xlogyScalar(self: anytype, comptime T: type, scalar: T) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.xlogyScalar(try castNumericScalar(T, @TypeOf(typed).Scalar, scalar))),
    };
}

pub fn xlogyWithDeviceScalar(self: anytype, scalar: options_mod.DeviceScalar) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.xlogyScalar(try castDeviceScalar(@TypeOf(typed).Scalar, scalar))),
    };
}

pub fn fmaxScalar(self: anytype, comptime T: type, scalar: T) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.fmaxScalar(try castNumericScalar(T, @TypeOf(typed).Scalar, scalar))),
    };
}

pub fn fmaxWithDeviceScalar(self: anytype, scalar: options_mod.DeviceScalar) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.fmaxScalar(try castDeviceScalar(@TypeOf(typed).Scalar, scalar))),
    };
}

pub fn fminScalar(self: anytype, comptime T: type, scalar: T) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.fminScalar(try castNumericScalar(T, @TypeOf(typed).Scalar, scalar))),
    };
}

pub fn fminWithDeviceScalar(self: anytype, scalar: options_mod.DeviceScalar) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.fminScalar(try castDeviceScalar(@TypeOf(typed).Scalar, scalar))),
    };
}

pub fn hypotScalar(self: anytype, comptime T: type, scalar: T) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.hypotScalar(try castNumericScalar(T, @TypeOf(typed).Scalar, scalar))),
    };
}

pub fn hypotWithDeviceScalar(self: anytype, scalar: options_mod.DeviceScalar) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.hypotScalar(try castDeviceScalar(@TypeOf(typed).Scalar, scalar))),
    };
}

pub fn atan2Scalar(self: anytype, comptime T: type, scalar: T) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.atan2Scalar(try castNumericScalar(T, @TypeOf(typed).Scalar, scalar))),
    };
}

pub fn atan2WithDeviceScalar(self: anytype, scalar: options_mod.DeviceScalar) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.atan2Scalar(try castDeviceScalar(@TypeOf(typed).Scalar, scalar))),
    };
}

pub fn nextAfterScalar(self: anytype, comptime T: type, scalar: T) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.nextAfterScalar(try castNumericScalar(T, @TypeOf(typed).Scalar, scalar))),
    };
}

pub fn nextAfterWithDeviceScalar(self: anytype, scalar: options_mod.DeviceScalar) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.nextAfterScalar(try castDeviceScalar(@TypeOf(typed).Scalar, scalar))),
    };
}

pub fn copysignScalar(self: anytype, comptime T: type, scalar: T) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.copysignScalar(try castNumericScalar(T, @TypeOf(typed).Scalar, scalar))),
    };
}

pub fn copysignWithDeviceScalar(self: anytype, scalar: options_mod.DeviceScalar) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.copysignScalar(try castDeviceScalar(@TypeOf(typed).Scalar, scalar))),
    };
}

pub fn heavisideScalar(self: anytype, comptime T: type, value_at_zero: T) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.heavisideScalar(try castNumericScalar(T, @TypeOf(typed).Scalar, value_at_zero))),
    };
}

pub fn heavisideWithDeviceScalar(self: anytype, value_at_zero: options_mod.DeviceScalar) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.heavisideScalar(try castDeviceScalar(@TypeOf(typed).Scalar, value_at_zero))),
    };
}

pub fn ldexpScalar(self: anytype, exponent: i32) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.ldexpScalar(exponent)),
    };
}

pub fn threshold(self: anytype, comptime T: type, threshold_value: T, replacement_value: T) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.threshold(
            try castNumericScalar(T, @TypeOf(typed).Scalar, threshold_value),
            try castNumericScalar(T, @TypeOf(typed).Scalar, replacement_value),
        )),
    };
}

pub fn thresholdWithDeviceScalars(self: anytype, threshold_value: options_mod.DeviceScalar, replacement_value: options_mod.DeviceScalar) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.threshold(
            try castDeviceScalar(@TypeOf(typed).Scalar, threshold_value),
            try castDeviceScalar(@TypeOf(typed).Scalar, replacement_value),
        )),
    };
}

pub fn hardtanh(self: anytype, comptime T: type, min_value: T, max_value: T) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.hardtanh(
            try castNumericScalar(T, @TypeOf(typed).Scalar, min_value),
            try castNumericScalar(T, @TypeOf(typed).Scalar, max_value),
        )),
    };
}

pub fn hardtanhWithDeviceScalars(self: anytype, min_value: options_mod.DeviceScalar, max_value: options_mod.DeviceScalar) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.hardtanh(
            try castDeviceScalar(@TypeOf(typed).Scalar, min_value),
            try castDeviceScalar(@TypeOf(typed).Scalar, max_value),
        )),
    };
}

pub fn maximumScalar(self: anytype, comptime T: type, scalar: T) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.maximumScalar(try castNumericScalar(T, @TypeOf(typed).Scalar, scalar))),
    };
}

pub fn maximumWithDeviceScalar(self: anytype, scalar: options_mod.DeviceScalar) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.maximumScalar(try castDeviceScalar(@TypeOf(typed).Scalar, scalar))),
    };
}

pub fn minimumScalar(self: anytype, comptime T: type, scalar: T) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.minimumScalar(try castNumericScalar(T, @TypeOf(typed).Scalar, scalar))),
    };
}

pub fn minimumWithDeviceScalar(self: anytype, scalar: options_mod.DeviceScalar) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.minimumScalar(try castDeviceScalar(@TypeOf(typed).Scalar, scalar))),
    };
}

pub fn clipMin(self: anytype, comptime T: type, min_value: T) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    return maximumScalar(self, T, min_value);
}

pub fn clipMinWithDeviceScalar(self: anytype, min_value: options_mod.DeviceScalar) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    return maximumWithDeviceScalar(self, min_value);
}

pub fn clipMax(self: anytype, comptime T: type, max_value: T) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    return minimumScalar(self, T, max_value);
}

pub fn clipMaxWithDeviceScalar(self: anytype, max_value: options_mod.DeviceScalar) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    return minimumWithDeviceScalar(self, max_value);
}

pub fn hardshrink(self: anytype, comptime T: type, lambd: T) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.hardshrink(try castNumericScalar(T, @TypeOf(typed).Scalar, lambd))),
    };
}

pub fn hardshrinkWithDeviceScalar(self: anytype, scalar: options_mod.DeviceScalar) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    return switch (scalar) {
        inline .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .bf16, .f16, .f32, .f64 => |value| hardshrink(self, @TypeOf(value), value),
        .bool, .c64, .c128 => error.TypeUnsupported,
    };
}

pub fn softshrink(self: anytype, comptime T: type, lambd: T) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.softshrink(try castNumericScalar(T, @TypeOf(typed).Scalar, lambd))),
    };
}

pub fn softshrinkWithDeviceScalar(self: anytype, scalar: options_mod.DeviceScalar) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    return switch (scalar) {
        inline .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .bf16, .f16, .f32, .f64 => |value| softshrink(self, @TypeOf(value), value),
        .bool, .c64, .c128 => error.TypeUnsupported,
    };
}

pub fn tanhshrink(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.tanhshrink()),
    };
}

pub fn elu(self: anytype, comptime T: type, alpha: T) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.elu(try castNumericScalar(T, @TypeOf(typed).Scalar, alpha))),
    };
}

pub fn eluWithDeviceScalar(self: anytype, scalar: options_mod.DeviceScalar) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    return switch (scalar) {
        inline .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .bf16, .f16, .f32, .f64 => |value| elu(self, @TypeOf(value), value),
        .bool, .c64, .c128 => error.TypeUnsupported,
    };
}

pub fn celu(self: anytype, comptime T: type, alpha: T) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.celu(try castNumericScalar(T, @TypeOf(typed).Scalar, alpha))),
    };
}

pub fn celuWithDeviceScalar(self: anytype, scalar: options_mod.DeviceScalar) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    return switch (scalar) {
        inline .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .bf16, .f16, .f32, .f64 => |value| celu(self, @TypeOf(value), value),
        .bool, .c64, .c128 => error.TypeUnsupported,
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

pub fn lerpScalar(self: anytype, end: ColumnType(@TypeOf(self)), comptime T: type, weight: T) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    if (value.dtype() != end.dtype()) return error.TypeUnsupported;
    if (!value.device().sameDevice(end.device())) return error.InvalidDevice;
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.lerpScalar(@field(end, @tagName(tag)), try castNumericScalar(T, @TypeOf(typed).Scalar, weight))),
    };
}

pub fn lerpWithDeviceScalar(self: anytype, end: ColumnType(@TypeOf(self)), weight: options_mod.DeviceScalar) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    if (value.dtype() != end.dtype()) return error.TypeUnsupported;
    if (!value.device().sameDevice(end.device())) return error.InvalidDevice;
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.lerpScalar(@field(end, @tagName(tag)), try castDeviceScalar(@TypeOf(typed).Scalar, weight))),
    };
}

pub fn addcmulScalar(self: anytype, input1: ColumnType(@TypeOf(self)), input2: ColumnType(@TypeOf(self)), comptime T: type, value: T) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const column = columnValue(self);
    if (column.dtype() != input1.dtype() or column.dtype() != input2.dtype()) return error.TypeUnsupported;
    if (!column.device().sameDevice(input1.device()) or !column.device().sameDevice(input2.device())) return error.InvalidDevice;
    return switch (column) {
        .bool, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.fusedTernaryScalar(
            @field(input1, @tagName(tag)),
            @field(input2, @tagName(tag)),
            try castNumericScalar(T, @TypeOf(typed).Scalar, value),
            .addcmul,
        )),
    };
}

pub fn addcmulWithDeviceScalar(self: anytype, input1: ColumnType(@TypeOf(self)), input2: ColumnType(@TypeOf(self)), value: options_mod.DeviceScalar) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const column = columnValue(self);
    if (column.dtype() != input1.dtype() or column.dtype() != input2.dtype()) return error.TypeUnsupported;
    if (!column.device().sameDevice(input1.device()) or !column.device().sameDevice(input2.device())) return error.InvalidDevice;
    return switch (column) {
        .bool, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.fusedTernaryScalar(
            @field(input1, @tagName(tag)),
            @field(input2, @tagName(tag)),
            try castDeviceScalar(@TypeOf(typed).Scalar, value),
            .addcmul,
        )),
    };
}

pub fn addcdivScalar(self: anytype, input1: ColumnType(@TypeOf(self)), input2: ColumnType(@TypeOf(self)), comptime T: type, value: T) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const column = columnValue(self);
    if (column.dtype() != input1.dtype() or column.dtype() != input2.dtype()) return error.TypeUnsupported;
    if (!column.device().sameDevice(input1.device()) or !column.device().sameDevice(input2.device())) return error.InvalidDevice;
    return switch (column) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.fusedTernaryScalar(
            @field(input1, @tagName(tag)),
            @field(input2, @tagName(tag)),
            try castNumericScalar(T, @TypeOf(typed).Scalar, value),
            .addcdiv,
        )),
    };
}

pub fn addcdivWithDeviceScalar(self: anytype, input1: ColumnType(@TypeOf(self)), input2: ColumnType(@TypeOf(self)), value: options_mod.DeviceScalar) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const column = columnValue(self);
    if (column.dtype() != input1.dtype() or column.dtype() != input2.dtype()) return error.TypeUnsupported;
    if (!column.device().sameDevice(input1.device()) or !column.device().sameDevice(input2.device())) return error.InvalidDevice;
    return switch (column) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.fusedTernaryScalar(
            @field(input1, @tagName(tag)),
            @field(input2, @tagName(tag)),
            try castDeviceScalar(@TypeOf(typed).Scalar, value),
            .addcdiv,
        )),
    };
}

pub fn clipArray(self: anytype, min_values: ColumnType(@TypeOf(self)), max_values: ColumnType(@TypeOf(self))) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const column = columnValue(self);
    if (column.dtype() != min_values.dtype() or column.dtype() != max_values.dtype()) return error.TypeUnsupported;
    if (!column.device().sameDevice(min_values.device()) or !column.device().sameDevice(max_values.device())) return error.InvalidDevice;
    return switch (column) {
        .bool, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.clipArray(
            @field(min_values, @tagName(tag)),
            @field(max_values, @tagName(tag)),
        )),
    };
}

pub fn whereScalar(self: anytype, mask: ColumnType(@TypeOf(self)), comptime T: type, other_value: T) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const column = columnValue(self);
    if (mask != .bool) return error.TypeUnsupported;
    if (column.dtype() != array_mod.DType.of(T)) return error.TypeUnsupported;
    if (!column.device().sameDevice(mask.device())) return error.InvalidDevice;
    const tag = comptime array_mod.DType.of(T);
    return @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try @field(column, @tagName(tag)).whereScalar(mask.bool, other_value));
}

pub fn whereWithDeviceScalar(self: anytype, mask: ColumnType(@TypeOf(self)), other_value: options_mod.DeviceScalar) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    return switch (other_value) {
        .bool => |value| whereScalar(self, mask, bool, value),
        .i8 => |value| whereScalar(self, mask, i8, value),
        .i16 => |value| whereScalar(self, mask, i16, value),
        .i32 => |value| whereScalar(self, mask, i32, value),
        .i64 => |value| whereScalar(self, mask, i64, value),
        .u8 => |value| whereScalar(self, mask, u8, value),
        .u16 => |value| whereScalar(self, mask, u16, value),
        .u32 => |value| whereScalar(self, mask, u32, value),
        .u64 => |value| whereScalar(self, mask, u64, value),
        .usize => |value| whereScalar(self, mask, usize, value),
        .isize => |value| whereScalar(self, mask, isize, value),
        .f16 => |value| whereScalar(self, mask, f16, value),
        .f32 => |value| whereScalar(self, mask, f32, value),
        .f64 => |value| whereScalar(self, mask, f64, value),
        .bf16 => |value| whereScalar(self, mask, array_mod.BFloat16, value),
        .c64 => |value| whereScalar(self, mask, array_mod.Complex64, value),
        .c128 => |value| whereScalar(self, mask, array_mod.Complex128, value),
    };
}

pub fn whereColumn(self: anytype, mask: ColumnType(@TypeOf(self)), other: ColumnType(@TypeOf(self))) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const column = columnValue(self);
    if (mask != .bool) return error.TypeUnsupported;
    if (column.dtype() != other.dtype()) return error.TypeUnsupported;
    if (!column.device().sameDevice(mask.device()) or !column.device().sameDevice(other.device())) return error.InvalidDevice;
    return switch (column) {
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.whereColumn(mask.bool, @field(other, @tagName(tag)))),
    };
}

pub fn isinColumn(self: anytype, test_elements: ColumnType(@TypeOf(self)), invert: bool) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const column = columnValue(self);
    if (column.dtype() != test_elements.dtype()) return error.TypeUnsupported;
    if (!column.device().sameDevice(test_elements.device())) return error.InvalidDevice;
    return switch (column) {
        inline else => |typed, tag| .{ .bool = try typed.isinColumn(@field(test_elements, @tagName(tag)), invert) },
    };
}

pub fn maskedPutScalar(self: anytype, mask: ColumnType(@TypeOf(self)), comptime T: type, value: T) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const column = columnValue(self);
    if (mask != .bool) return error.TypeUnsupported;
    if (column.dtype() != array_mod.DType.of(T)) return error.TypeUnsupported;
    if (!column.device().sameDevice(mask.device())) return error.InvalidDevice;
    const tag = comptime array_mod.DType.of(T);
    return @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try @field(column, @tagName(tag)).maskedPutScalar(mask.bool, value));
}

pub fn maskedPutWithDeviceScalar(self: anytype, mask: ColumnType(@TypeOf(self)), value: options_mod.DeviceScalar) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    return switch (value) {
        .bool => |scalar| maskedPutScalar(self, mask, bool, scalar),
        .i8 => |scalar| maskedPutScalar(self, mask, i8, scalar),
        .i16 => |scalar| maskedPutScalar(self, mask, i16, scalar),
        .i32 => |scalar| maskedPutScalar(self, mask, i32, scalar),
        .i64 => |scalar| maskedPutScalar(self, mask, i64, scalar),
        .u8 => |scalar| maskedPutScalar(self, mask, u8, scalar),
        .u16 => |scalar| maskedPutScalar(self, mask, u16, scalar),
        .u32 => |scalar| maskedPutScalar(self, mask, u32, scalar),
        .u64 => |scalar| maskedPutScalar(self, mask, u64, scalar),
        .usize => |scalar| maskedPutScalar(self, mask, usize, scalar),
        .isize => |scalar| maskedPutScalar(self, mask, isize, scalar),
        .f16 => |scalar| maskedPutScalar(self, mask, f16, scalar),
        .f32 => |scalar| maskedPutScalar(self, mask, f32, scalar),
        .f64 => |scalar| maskedPutScalar(self, mask, f64, scalar),
        .bf16 => |scalar| maskedPutScalar(self, mask, array_mod.BFloat16, scalar),
        .c64 => |scalar| maskedPutScalar(self, mask, array_mod.Complex64, scalar),
        .c128 => |scalar| maskedPutScalar(self, mask, array_mod.Complex128, scalar),
    };
}

pub fn putFlat(self: anytype, row_indices: []const usize, values: ColumnType(@TypeOf(self))) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const column = columnValue(self);
    if (column.dtype() != values.dtype()) return error.TypeUnsupported;
    if (!column.device().sameDevice(values.device())) return error.InvalidDevice;
    return switch (column) {
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.putFlat(row_indices, @field(values, @tagName(tag)))),
    };
}

pub fn putFlatScalar(self: anytype, row_indices: []const usize, comptime T: type, value: T) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const column = columnValue(self);
    if (column.dtype() != array_mod.DType.of(T)) return error.TypeUnsupported;
    const tag = comptime array_mod.DType.of(T);
    return @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try @field(column, @tagName(tag)).putFlatScalar(row_indices, value));
}

pub fn putFlatWithDeviceScalar(self: anytype, row_indices: []const usize, value: options_mod.DeviceScalar) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    return switch (value) {
        .bool => |scalar| putFlatScalar(self, row_indices, bool, scalar),
        .i8 => |scalar| putFlatScalar(self, row_indices, i8, scalar),
        .i16 => |scalar| putFlatScalar(self, row_indices, i16, scalar),
        .i32 => |scalar| putFlatScalar(self, row_indices, i32, scalar),
        .i64 => |scalar| putFlatScalar(self, row_indices, i64, scalar),
        .u8 => |scalar| putFlatScalar(self, row_indices, u8, scalar),
        .u16 => |scalar| putFlatScalar(self, row_indices, u16, scalar),
        .u32 => |scalar| putFlatScalar(self, row_indices, u32, scalar),
        .u64 => |scalar| putFlatScalar(self, row_indices, u64, scalar),
        .usize => |scalar| putFlatScalar(self, row_indices, usize, scalar),
        .isize => |scalar| putFlatScalar(self, row_indices, isize, scalar),
        .f16 => |scalar| putFlatScalar(self, row_indices, f16, scalar),
        .f32 => |scalar| putFlatScalar(self, row_indices, f32, scalar),
        .f64 => |scalar| putFlatScalar(self, row_indices, f64, scalar),
        .bf16 => |scalar| putFlatScalar(self, row_indices, array_mod.BFloat16, scalar),
        .c64 => |scalar| putFlatScalar(self, row_indices, array_mod.Complex64, scalar),
        .c128 => |scalar| putFlatScalar(self, row_indices, array_mod.Complex128, scalar),
    };
}

pub fn putFlatScalarMode(self: anytype, row_indices: []const usize, comptime T: type, value: T, mode: array_mod.IndexMode) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const column = columnValue(self);
    if (column.dtype() != array_mod.DType.of(T)) return error.TypeUnsupported;
    const tag = comptime array_mod.DType.of(T);
    return @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try @field(column, @tagName(tag)).putFlatScalarMode(row_indices, value, mode));
}

pub fn putFlatModeWithDeviceScalar(self: anytype, row_indices: []const usize, value: options_mod.DeviceScalar, mode: array_mod.IndexMode) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    return switch (value) {
        .bool => |scalar| putFlatScalarMode(self, row_indices, bool, scalar, mode),
        .i8 => |scalar| putFlatScalarMode(self, row_indices, i8, scalar, mode),
        .i16 => |scalar| putFlatScalarMode(self, row_indices, i16, scalar, mode),
        .i32 => |scalar| putFlatScalarMode(self, row_indices, i32, scalar, mode),
        .i64 => |scalar| putFlatScalarMode(self, row_indices, i64, scalar, mode),
        .u8 => |scalar| putFlatScalarMode(self, row_indices, u8, scalar, mode),
        .u16 => |scalar| putFlatScalarMode(self, row_indices, u16, scalar, mode),
        .u32 => |scalar| putFlatScalarMode(self, row_indices, u32, scalar, mode),
        .u64 => |scalar| putFlatScalarMode(self, row_indices, u64, scalar, mode),
        .usize => |scalar| putFlatScalarMode(self, row_indices, usize, scalar, mode),
        .isize => |scalar| putFlatScalarMode(self, row_indices, isize, scalar, mode),
        .f16 => |scalar| putFlatScalarMode(self, row_indices, f16, scalar, mode),
        .f32 => |scalar| putFlatScalarMode(self, row_indices, f32, scalar, mode),
        .f64 => |scalar| putFlatScalarMode(self, row_indices, f64, scalar, mode),
        .bf16 => |scalar| putFlatScalarMode(self, row_indices, array_mod.BFloat16, scalar, mode),
        .c64 => |scalar| putFlatScalarMode(self, row_indices, array_mod.Complex64, scalar, mode),
        .c128 => |scalar| putFlatScalarMode(self, row_indices, array_mod.Complex128, scalar, mode),
    };
}

pub fn putFlatScalarSigned(self: anytype, row_indices: []const isize, comptime T: type, value: T) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const column = columnValue(self);
    if (column.dtype() != array_mod.DType.of(T)) return error.TypeUnsupported;
    const tag = comptime array_mod.DType.of(T);
    return @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try @field(column, @tagName(tag)).putFlatScalarSigned(row_indices, value));
}

pub fn putFlatSignedWithDeviceScalar(self: anytype, row_indices: []const isize, value: options_mod.DeviceScalar) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    return switch (value) {
        .bool => |scalar| putFlatScalarSigned(self, row_indices, bool, scalar),
        .i8 => |scalar| putFlatScalarSigned(self, row_indices, i8, scalar),
        .i16 => |scalar| putFlatScalarSigned(self, row_indices, i16, scalar),
        .i32 => |scalar| putFlatScalarSigned(self, row_indices, i32, scalar),
        .i64 => |scalar| putFlatScalarSigned(self, row_indices, i64, scalar),
        .u8 => |scalar| putFlatScalarSigned(self, row_indices, u8, scalar),
        .u16 => |scalar| putFlatScalarSigned(self, row_indices, u16, scalar),
        .u32 => |scalar| putFlatScalarSigned(self, row_indices, u32, scalar),
        .u64 => |scalar| putFlatScalarSigned(self, row_indices, u64, scalar),
        .usize => |scalar| putFlatScalarSigned(self, row_indices, usize, scalar),
        .isize => |scalar| putFlatScalarSigned(self, row_indices, isize, scalar),
        .f16 => |scalar| putFlatScalarSigned(self, row_indices, f16, scalar),
        .f32 => |scalar| putFlatScalarSigned(self, row_indices, f32, scalar),
        .f64 => |scalar| putFlatScalarSigned(self, row_indices, f64, scalar),
        .bf16 => |scalar| putFlatScalarSigned(self, row_indices, array_mod.BFloat16, scalar),
        .c64 => |scalar| putFlatScalarSigned(self, row_indices, array_mod.Complex64, scalar),
        .c128 => |scalar| putFlatScalarSigned(self, row_indices, array_mod.Complex128, scalar),
    };
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

pub fn iscloseScalar(
    self: anytype,
    comptime T: type,
    scalar: T,
    rtol: T,
    atol: T,
    equal_nan: bool,
) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed| .{ .bool = try typed.iscloseScalar(
            try castNumericScalar(T, @TypeOf(typed).Scalar, scalar),
            try castNumericScalar(T, @TypeOf(typed).Scalar, rtol),
            try castNumericScalar(T, @TypeOf(typed).Scalar, atol),
            equal_nan,
        ) },
    };
}

pub fn iscloseWithDeviceScalars(
    self: anytype,
    scalar: options_mod.DeviceScalar,
    rtol: options_mod.DeviceScalar,
    atol: options_mod.DeviceScalar,
    equal_nan: bool,
) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed| .{ .bool = try typed.iscloseScalar(
            try castDeviceScalar(@TypeOf(typed).Scalar, scalar),
            try castDeviceScalar(@TypeOf(typed).Scalar, rtol),
            try castDeviceScalar(@TypeOf(typed).Scalar, atol),
            equal_nan,
        ) },
    };
}

pub fn allcloseScalar(
    self: anytype,
    comptime T: type,
    scalar: T,
    rtol: T,
    atol: T,
    equal_nan: bool,
) array_mod.ArrayError!bool {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed| try typed.allcloseScalar(
            try castNumericScalar(T, @TypeOf(typed).Scalar, scalar),
            try castNumericScalar(T, @TypeOf(typed).Scalar, rtol),
            try castNumericScalar(T, @TypeOf(typed).Scalar, atol),
            equal_nan,
        ),
    };
}

pub fn allcloseWithDeviceScalars(
    self: anytype,
    scalar: options_mod.DeviceScalar,
    rtol: options_mod.DeviceScalar,
    atol: options_mod.DeviceScalar,
    equal_nan: bool,
) array_mod.ArrayError!bool {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed| try typed.allcloseScalar(
            try castDeviceScalar(@TypeOf(typed).Scalar, scalar),
            try castDeviceScalar(@TypeOf(typed).Scalar, rtol),
            try castDeviceScalar(@TypeOf(typed).Scalar, atol),
            equal_nan,
        ),
    };
}

pub fn countNonzero(self: anytype) array_mod.ArrayError!usize {
    const value = columnValue(self);
    return switch (value) {
        inline else => |typed| try typed.countNonzero(),
    };
}

pub fn countDistinct(self: anytype) array_mod.ArrayError!usize {
    const value = columnValue(self);
    return switch (value) {
        inline else => |typed| try typed.countDistinct(),
    };
}

pub fn nUnique(self: anytype) array_mod.ArrayError!usize {
    return countDistinct(self);
}

pub fn modeValue(self: anytype) array_mod.ArrayError!options_mod.DeviceScalar {
    const value = columnValue(self);
    return switch (value) {
        inline else => |typed| options_mod.DeviceScalar.init(@TypeOf(typed).Scalar, try typed.modeValue()),
    };
}

pub fn sum(self: anytype) array_mod.ArrayError!options_mod.DeviceScalar {
    const value = columnValue(self);
    return switch (value) {
        .bool => error.TypeUnsupported,
        inline else => |typed| options_mod.DeviceScalar.init(@TypeOf(typed).Scalar, try typed.sum()),
    };
}

pub fn prod(self: anytype) array_mod.ArrayError!options_mod.DeviceScalar {
    const value = columnValue(self);
    return switch (value) {
        .bool => error.TypeUnsupported,
        inline else => |typed| options_mod.DeviceScalar.init(@TypeOf(typed).Scalar, try typed.prod()),
    };
}

pub fn mean(self: anytype) array_mod.ArrayError!options_mod.DeviceScalar {
    const value = columnValue(self);
    return switch (value) {
        .f16 => |typed| options_mod.DeviceScalar.init(f16, try typed.mean()),
        .f32 => |typed| options_mod.DeviceScalar.init(f32, try typed.mean()),
        .f64 => |typed| options_mod.DeviceScalar.init(f64, try typed.mean()),
        .bf16 => |typed| options_mod.DeviceScalar.init(array_mod.BFloat16, try typed.mean()),
        else => error.TypeUnsupported,
    };
}

pub fn quantile(self: anytype, q: f64) array_mod.ArrayError!options_mod.DeviceScalar {
    const value = columnValue(self);
    return switch (value) {
        .bool, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed| .{ .f64 = try typed.quantile(q) },
    };
}

pub fn median(self: anytype) array_mod.ArrayError!options_mod.DeviceScalar {
    return quantile(self, 0.5);
}

pub fn variance(self: anytype, correction: f64) array_mod.ArrayError!options_mod.DeviceScalar {
    const value = columnValue(self);
    return switch (value) {
        .bool, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed| .{ .f64 = try typed.variance(correction) },
    };
}

pub fn stddev(self: anytype, correction: f64) array_mod.ArrayError!options_mod.DeviceScalar {
    const value = columnValue(self);
    return switch (value) {
        .bool, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed| .{ .f64 = try typed.stddev(correction) },
    };
}

pub fn sem(self: anytype, correction: f64) array_mod.ArrayError!options_mod.DeviceScalar {
    const value = columnValue(self);
    return switch (value) {
        .bool, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed| .{ .f64 = try typed.sem(correction) },
    };
}

pub fn cv(self: anytype, correction: f64) array_mod.ArrayError!options_mod.DeviceScalar {
    const value = columnValue(self);
    return switch (value) {
        .bool, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed| .{ .f64 = try typed.cv(correction) },
    };
}

pub fn skewness(self: anytype) array_mod.ArrayError!options_mod.DeviceScalar {
    const value = columnValue(self);
    return switch (value) {
        .bool, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed| .{ .f64 = try typed.skewness() },
    };
}

pub fn kurtosis(self: anytype) array_mod.ArrayError!options_mod.DeviceScalar {
    const value = columnValue(self);
    return switch (value) {
        .bool, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed| .{ .f64 = try typed.kurtosis() },
    };
}

pub fn meanAbs(self: anytype) array_mod.ArrayError!options_mod.DeviceScalar {
    const value = columnValue(self);
    return switch (value) {
        .bool, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed| .{ .f64 = try typed.meanAbs() },
    };
}

pub fn rms(self: anytype) array_mod.ArrayError!options_mod.DeviceScalar {
    const value = columnValue(self);
    return switch (value) {
        .bool, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed| .{ .f64 = try typed.rms() },
    };
}

pub fn min(self: anytype) array_mod.ArrayError!options_mod.DeviceScalar {
    const value = columnValue(self);
    return switch (value) {
        .bool, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed| options_mod.DeviceScalar.init(@TypeOf(typed).Scalar, try typed.min()),
    };
}

pub fn max(self: anytype) array_mod.ArrayError!options_mod.DeviceScalar {
    const value = columnValue(self);
    return switch (value) {
        .bool, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed| options_mod.DeviceScalar.init(@TypeOf(typed).Scalar, try typed.max()),
    };
}

pub fn ptp(self: anytype) array_mod.ArrayError!options_mod.DeviceScalar {
    const value = columnValue(self);
    return switch (value) {
        .bool, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed| options_mod.DeviceScalar.init(@TypeOf(typed).Scalar, try typed.ptp()),
    };
}

pub fn argmin(self: anytype) array_mod.ArrayError!usize {
    const value = columnValue(self);
    return switch (value) {
        .bool, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed| try typed.argmin(),
    };
}

pub fn argmax(self: anytype) array_mod.ArrayError!usize {
    const value = columnValue(self);
    return switch (value) {
        .bool, .c64, .c128 => error.TypeUnsupported,
        inline else => |typed| try typed.argmax(),
    };
}

pub fn any(self: anytype) array_mod.ArrayError!bool {
    const value = columnValue(self);
    return switch (value) {
        .bool => |typed| try typed.any(),
        else => error.TypeUnsupported,
    };
}

pub fn all(self: anytype) array_mod.ArrayError!bool {
    const value = columnValue(self);
    return switch (value) {
        .bool => |typed| try typed.all(),
        else => error.TypeUnsupported,
    };
}

pub fn countTrue(self: anytype) array_mod.ArrayError!usize {
    const value = columnValue(self);
    return switch (value) {
        .bool => |typed| try typed.countTrue(),
        else => error.TypeUnsupported,
    };
}

pub fn countFalse(self: anytype) array_mod.ArrayError!usize {
    const value = columnValue(self);
    return switch (value) {
        .bool => |typed| try typed.countFalse(),
        else => error.TypeUnsupported,
    };
}

pub fn logicalAndScalar(self: anytype, scalar: bool) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool => |typed| .{ .bool = try typed.logicalScalar(scalar, .@"and") },
        else => error.TypeUnsupported,
    };
}

pub fn logicalOrScalar(self: anytype, scalar: bool) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool => |typed| .{ .bool = try typed.logicalScalar(scalar, .@"or") },
        else => error.TypeUnsupported,
    };
}

pub fn logicalXorScalar(self: anytype, scalar: bool) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool => |typed| .{ .bool = try typed.logicalScalar(scalar, .xor) },
        else => error.TypeUnsupported,
    };
}

pub fn logical(self: anytype, other: ColumnType(@TypeOf(self)), op: DeviceColumnLogicalOp) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    if (value.dtype() != other.dtype()) return error.TypeUnsupported;
    if (!value.device().sameDevice(other.device())) return error.InvalidDevice;
    return switch (value) {
        .bool => |typed| .{ .bool = switch (op) {
            .@"and" => try typed.logical(other.bool, .@"and"),
            .@"or" => try typed.logical(other.bool, .@"or"),
            .xor => try typed.logical(other.bool, .xor),
        } },
        else => error.TypeUnsupported,
    };
}

pub fn logicalAnd(self: anytype, other: ColumnType(@TypeOf(self))) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    return logical(self, other, .@"and");
}

pub fn logicalOr(self: anytype, other: ColumnType(@TypeOf(self))) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    return logical(self, other, .@"or");
}

pub fn logicalXor(self: anytype, other: ColumnType(@TypeOf(self))) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    return logical(self, other, .xor);
}
