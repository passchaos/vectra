//! Arithmetic and comparison helpers for tagged device columns.

const array_mod = @import("array.zig");
const options_mod = @import("dataframe_options.zig");

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

pub fn sqrt(self: anytype) array_mod.ArrayError!ColumnType(@TypeOf(self)) {
    const value = columnValue(self);
    return switch (value) {
        .bool, .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize => error.TypeUnsupported,
        inline else => |typed, tag| @unionInit(ColumnType(@TypeOf(self)), @tagName(tag), try typed.sqrt()),
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
