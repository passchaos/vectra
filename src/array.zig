const std = @import("std");
const alea = @import("alea");

pub const Complex64 = std.math.Complex(f32);
pub const Complex128 = std.math.Complex(f64);

pub const BFloat16 = struct {
    bits: u16,

    pub fn fromF32(value: f32) BFloat16 {
        const raw: u32 = @bitCast(value);
        const rounding_bias = @as(u32, 0x7fff) + ((raw >> 16) & 1);
        return .{ .bits = @intCast((raw +% rounding_bias) >> 16) };
    }

    pub fn fromF64(value: f64) BFloat16 {
        return BFloat16.fromF32(@floatCast(value));
    }

    pub fn toF32(self: BFloat16) f32 {
        return @bitCast(@as(u32, self.bits) << 16);
    }

    pub fn toF64(self: BFloat16) f64 {
        return @floatCast(self.toF32());
    }

    pub fn add(self: BFloat16, other: BFloat16) BFloat16 {
        return BFloat16.fromF32(self.toF32() + other.toF32());
    }

    pub fn sub(self: BFloat16, other: BFloat16) BFloat16 {
        return BFloat16.fromF32(self.toF32() - other.toF32());
    }

    pub fn mul(self: BFloat16, other: BFloat16) BFloat16 {
        return BFloat16.fromF32(self.toF32() * other.toF32());
    }

    pub fn div(self: BFloat16, other: BFloat16) BFloat16 {
        return BFloat16.fromF32(self.toF32() / other.toF32());
    }

    pub fn neg(self: BFloat16) BFloat16 {
        return .{ .bits = self.bits ^ 0x8000 };
    }

    pub fn abs(self: BFloat16) BFloat16 {
        return .{ .bits = self.bits & 0x7fff };
    }

    pub fn eql(self: BFloat16, other: BFloat16) bool {
        return self.toF32() == other.toF32();
    }

    pub fn lt(self: BFloat16, other: BFloat16) bool {
        return self.toF32() < other.toF32();
    }

    pub fn lte(self: BFloat16, other: BFloat16) bool {
        return self.toF32() <= other.toF32();
    }
};

pub const Backend = enum {
    cpu,
    cuda,
};

pub const Device = struct {
    backend: Backend = .cpu,
    index: usize = 0,

    pub const cpu: Device = .{ .backend = .cpu, .index = 0 };

    pub fn cuda(index: usize) Device {
        return .{ .backend = .cuda, .index = index };
    }

    pub fn isAvailable(self: Device) bool {
        return switch (self.backend) {
            .cpu => true,
            // CUDA is represented in the public API so users can write PyTorch-like
            // code today. A future backend can make this true without changing array
            // call sites.
            .cuda => false,
        };
    }
};

pub const DType = enum {
    f32,
    f64,
    i8,
    i16,
    i32,
    i64,
    u8,
    u16,
    u32,
    u64,
    usize,
    bool,
    bf16,
    f16,
    c64,
    c128,
    isize,

    pub fn of(comptime T: type) DType {
        return switch (T) {
            BFloat16 => .bf16,
            f16 => .f16,
            f32 => .f32,
            f64 => .f64,
            Complex64 => .c64,
            Complex128 => .c128,
            i8 => .i8,
            i16 => .i16,
            i32 => .i32,
            i64 => .i64,
            u8 => .u8,
            u16 => .u16,
            u32 => .u32,
            u64 => .u64,
            usize => .usize,
            isize => .isize,
            bool => .bool,
            else => @compileError("unsupported Vectra dtype: " ++ @typeName(T)),
        };
    }

    pub fn name(self: DType) []const u8 {
        return switch (self) {
            .bf16 => "bf16",
            .f16 => "f16",
            .f32 => "f32",
            .f64 => "f64",
            .i8 => "i8",
            .i16 => "i16",
            .i32 => "i32",
            .i64 => "i64",
            .u8 => "u8",
            .u16 => "u16",
            .u32 => "u32",
            .u64 => "u64",
            .usize => "usize",
            .bool => "bool",
            .c64 => "complex64",
            .c128 => "complex128",
            .isize => "isize",
        };
    }

    pub fn byteSize(self: DType) usize {
        return switch (self) {
            .bool, .i8, .u8 => 1,
            .bf16, .f16, .i16, .u16 => 2,
            .f32, .i32, .u32 => 4,
            .f64, .i64, .u64 => 8,
            .usize => @sizeOf(usize),
            .c64 => 8,
            .c128 => 16,
            .isize => @sizeOf(isize),
        };
    }

    pub fn isFloat(self: DType) bool {
        return switch (self) {
            .bf16, .f16, .f32, .f64 => true,
            else => false,
        };
    }

    pub fn isInteger(self: DType) bool {
        return switch (self) {
            .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize => true,
            else => false,
        };
    }

    pub fn isSigned(self: DType) bool {
        return switch (self) {
            .i8, .i16, .i32, .i64, .isize => true,
            else => false,
        };
    }

    pub fn isBool(self: DType) bool {
        return self == .bool;
    }

    pub fn isComplex(self: DType) bool {
        return self == .c64 or self == .c128;
    }

    pub fn bitSize(self: DType) usize {
        return self.byteSize() * 8;
    }

    pub fn canCast(self: DType, target: DType) bool {
        if (self == target) return true;
        if (self.isComplex()) return target.isComplex();
        if (target.isComplex()) return self.isBool() or self.isInteger() or self.isFloat();
        if (self.isBool() and target.isBool()) return true;
        if (self.isBool()) return target.isInteger() or target.isFloat();
        if (target.isBool()) return self.isInteger() or self.isFloat();
        return (self.isInteger() or self.isFloat()) and (target.isInteger() or target.isFloat());
    }

    fn floatRank(self: DType) usize {
        return switch (self) {
            .bf16, .f16 => 1,
            .f32 => 2,
            .f64 => 3,
            else => 0,
        };
    }

    fn intRank(self: DType) usize {
        return switch (self) {
            .bool => 0,
            .i8, .u8 => 1,
            .i16, .u16 => 2,
            .i32, .u32 => 3,
            .i64, .u64, .usize, .isize => 4,
            else => 0,
        };
    }

    fn complexRank(self: DType) usize {
        return switch (self) {
            .c64 => 1,
            .c128 => 2,
            else => 0,
        };
    }

    pub fn promote(a: DType, b: DType) DType {
        if (a == b) return a;
        if (a.isComplex() or b.isComplex()) {
            if (a.complexRank() >= 2 or b.complexRank() >= 2 or a == .f64 or b == .f64) return .c128;
            return .c64;
        }
        if (a.isBool()) return b;
        if (b.isBool()) return a;
        if (a.isFloat() or b.isFloat()) {
            const rank = @max(a.floatRank(), b.floatRank());
            if (rank >= 3) return .f64;
            if (rank == 2) return .f32;
            return .f16;
        }
        var rank = @max(a.intRank(), b.intRank());
        const a_unsigned = a.isInteger() and !a.isSigned();
        const b_unsigned = b.isInteger() and !b.isSigned();
        const mixed_signedness = (a.isSigned() and b_unsigned) or (b.isSigned() and a_unsigned);
        if (mixed_signedness) {
            const signed_rank = if (a.isSigned()) a.intRank() else b.intRank();
            const unsigned_rank = if (a_unsigned) a.intRank() else b.intRank();
            if (unsigned_rank >= signed_rank) rank += 1;
            if (rank > 4) return .f64;
        }
        const signed = a.isSigned() or b.isSigned();
        if (rank <= 1) return if (signed) .i8 else .u8;
        if (rank == 2) return if (signed) .i16 else .u16;
        if (rank == 3) return if (signed) .i32 else .u32;
        return if (signed) .i64 else .u64;
    }

    pub fn result(a: DType, b: DType) DType {
        return DType.promote(a, b);
    }

    pub fn Type(self: DType) type {
        return switch (self) {
            .bf16 => BFloat16,
            .f16 => f16,
            .f32 => f32,
            .f64 => f64,
            .i8 => i8,
            .i16 => i16,
            .i32 => i32,
            .i64 => i64,
            .u8 => u8,
            .u16 => u16,
            .u32 => u32,
            .u64 => u64,
            .usize => usize,
            .bool => bool,
            .c64 => Complex64,
            .c128 => Complex128,
            .isize => isize,
        };
    }

    pub fn tag(self: DType) u8 {
        return @intFromEnum(self);
    }

    pub fn fromTag(tag_value: u8) ?DType {
        return switch (tag_value) {
            @intFromEnum(DType.bf16) => .bf16,
            @intFromEnum(DType.f16) => .f16,
            @intFromEnum(DType.f32) => .f32,
            @intFromEnum(DType.f64) => .f64,
            @intFromEnum(DType.i8) => .i8,
            @intFromEnum(DType.i16) => .i16,
            @intFromEnum(DType.i32) => .i32,
            @intFromEnum(DType.i64) => .i64,
            @intFromEnum(DType.u8) => .u8,
            @intFromEnum(DType.u16) => .u16,
            @intFromEnum(DType.u32) => .u32,
            @intFromEnum(DType.u64) => .u64,
            @intFromEnum(DType.usize) => .usize,
            @intFromEnum(DType.bool) => .bool,
            @intFromEnum(DType.c64) => .c64,
            @intFromEnum(DType.c128) => .c128,
            @intFromEnum(DType.isize) => .isize,
            else => null,
        };
    }
};

pub const Archive = struct {
    pub const magic = [_]u8{ 'V', 'X', 'A', 'R', 'R', '0', '1', 0 };
    pub const version: u8 = 1;
};

pub fn canCastDType(from: DType, to: DType) bool {
    return from.canCast(to);
}

pub fn promoteDType(a: DType, b: DType) DType {
    return DType.promote(a, b);
}

pub fn resultDType(a: DType, b: DType) DType {
    return DType.result(a, b);
}

pub fn promoteType(comptime A: type, comptime B: type) type {
    return DType.promote(DType.of(A), DType.of(B)).Type();
}

pub const ArrayError = error{
    ShapeMismatch,
    InvalidShape,
    InvalidAxis,
    InvalidDevice,
    InvalidPermutation,
    IndexOutOfBounds,
    NonMatrixArray,
    NonVectorArray,
    EmptyArray,
    TypeUnsupported,
} || std.mem.Allocator.Error;

pub const Shape = struct {
    allocator: std.mem.Allocator,
    dims: []usize,

    pub fn init(allocator: std.mem.Allocator, dims: []const usize) ArrayError!Shape {
        const owned = try allocator.dupe(usize, dims);
        return .{ .allocator = allocator, .dims = owned };
    }

    pub fn deinit(self: *Shape) void {
        self.allocator.free(self.dims);
        self.* = undefined;
    }

    pub fn len(self: Shape) usize {
        return self.dims.len;
    }

    pub fn eql(self: Shape, other: Shape) bool {
        return std.mem.eql(usize, self.dims, other.dims);
    }

    pub fn numel(self: Shape) ArrayError!usize {
        return numelFrom(self.dims);
    }
};

pub fn numelFrom(dims: []const usize) ArrayError!usize {
    if (dims.len == 0) return 1;
    var n: usize = 1;
    for (dims) |d| {
        n = std.math.mul(usize, n, d) catch return error.InvalidShape;
    }
    return n;
}

pub fn stridesFor(allocator: std.mem.Allocator, dims: []const usize) ArrayError![]usize {
    const strides = try allocator.alloc(usize, dims.len);
    var stride: usize = 1;
    var i = dims.len;
    while (i > 0) {
        i -= 1;
        strides[i] = stride;
        stride = std.math.mul(usize, stride, dims[i]) catch return error.InvalidShape;
    }
    return strides;
}

fn isFloat(comptime T: type) bool {
    if (comptime T == BFloat16) return true;
    return @typeInfo(T) == .float;
}

fn isComplex(comptime T: type) bool {
    return T == Complex64 or T == Complex128;
}

fn isNumeric(comptime T: type) bool {
    if (comptime T == BFloat16) return true;
    if (comptime isComplex(T)) return true;
    return switch (@typeInfo(T)) {
        .int, .float, .comptime_int, .comptime_float => true,
        else => false,
    };
}

fn ensureNumeric(comptime T: type) void {
    if (comptime !isNumeric(T)) @compileError("operation requires a numeric array, got " ++ @typeName(T));
}

fn ensureFloat(comptime T: type) void {
    if (comptime !isFloat(T)) @compileError("operation requires a floating-point array, got " ++ @typeName(T));
}

fn ensureComplex(comptime T: type) void {
    if (comptime !isComplex(T)) @compileError("operation requires a complex array, got " ++ @typeName(T));
}

fn complexRealType(comptime T: type) type {
    if (comptime T == Complex64) return f32;
    if (comptime T == Complex128) return f64;
    @compileError("operation requires a complex array, got " ++ @typeName(T));
}

fn complexTypeForReal(comptime T: type) type {
    if (comptime T == f64) return Complex128;
    if (comptime T == f32 or T == f16 or T == BFloat16) return Complex64;
    return Complex64;
}

fn realTypeForComplex(comptime T: type) type {
    if (comptime T == Complex128) return f64;
    return f32;
}

fn ensureOrderable(comptime T: type) void {
    switch (@typeInfo(T)) {
        .bool, .int, .float, .comptime_int, .comptime_float => {},
        else => @compileError("ordering requires a bool or numeric array, got " ++ @typeName(T)),
    }
}

fn lessValue(comptime T: type, a: T, b: T) bool {
    if (comptime T == BFloat16) return a.lt(b);
    return switch (@typeInfo(T)) {
        .bool => !a and b,
        .int, .float, .comptime_int, .comptime_float => a < b,
        else => @compileError("ordering requires a bool or numeric array, got " ++ @typeName(T)),
    };
}

fn castValue(comptime T: type, value: anytype) T {
    const V = @TypeOf(value);
    if (comptime T == BFloat16) {
        if (comptime V == BFloat16) return value;
        if (comptime isComplex(V)) @compileError("cannot cast " ++ @typeName(V) ++ " to BFloat16");
        return switch (@typeInfo(V)) {
            .float, .comptime_float => BFloat16.fromF32(@floatCast(value)),
            .int, .comptime_int => BFloat16.fromF32(@floatFromInt(value)),
            .bool => BFloat16.fromF32(if (value) 1 else 0),
            else => @compileError("cannot cast " ++ @typeName(V) ++ " to BFloat16"),
        };
    }
    if (comptime isComplex(T)) {
        const Real = @TypeOf(@as(T, undefined).re);
        if (comptime isComplex(V)) return T.init(castValue(Real, value.re), castValue(Real, value.im));
        if (comptime V == BFloat16) return T.init(castValue(Real, value.toF32()), 0);
        return T.init(castValue(Real, value), 0);
    }
    return switch (@typeInfo(T)) {
        .float => switch (@typeInfo(V)) {
            .float, .comptime_float => @floatCast(value),
            .int, .comptime_int => @floatFromInt(value),
            .@"struct" => if (comptime V == BFloat16) @floatCast(value.toF32()) else @compileError("cannot cast " ++ @typeName(V) ++ " to " ++ @typeName(T)),
            .bool => if (value) 1 else 0,
            else => @compileError("cannot cast " ++ @typeName(V) ++ " to " ++ @typeName(T)),
        },
        .int => switch (@typeInfo(V)) {
            .int, .comptime_int => @intCast(value),
            .float, .comptime_float => @intFromFloat(value),
            .@"struct" => if (comptime V == BFloat16) @intFromFloat(value.toF32()) else @compileError("cannot cast " ++ @typeName(V) ++ " to " ++ @typeName(T)),
            .bool => if (value) 1 else 0,
            else => @compileError("cannot cast " ++ @typeName(V) ++ " to " ++ @typeName(T)),
        },
        .bool => switch (@typeInfo(V)) {
            .bool => value,
            .int, .comptime_int => value != 0,
            .float, .comptime_float => value != 0,
            else => @compileError("cannot cast " ++ @typeName(V) ++ " to bool"),
        },
        else => @compileError("unsupported array scalar type: " ++ @typeName(T)),
    };
}

fn zero(comptime T: type) T {
    return switch (@typeInfo(T)) {
        .bool => false,
        else => castValue(T, 0),
    };
}

fn one(comptime T: type) T {
    return switch (@typeInfo(T)) {
        .bool => true,
        else => castValue(T, 1),
    };
}

fn addValue(comptime T: type, a: T, b: T) T {
    if (comptime T == BFloat16) return a.add(b);
    if (comptime isComplex(T)) return a.add(b);
    return switch (@typeInfo(T)) {
        .bool => a or b,
        else => a + b,
    };
}

fn subValue(comptime T: type, a: T, b: T) T {
    if (comptime T == BFloat16) return a.sub(b);
    if (comptime isComplex(T)) return a.sub(b);
    return a - b;
}

fn mulValue(comptime T: type, a: T, b: T) T {
    if (comptime T == BFloat16) return a.mul(b);
    if (comptime isComplex(T)) return a.mul(b);
    return switch (@typeInfo(T)) {
        .bool => a and b,
        else => a * b,
    };
}

fn divValue(comptime T: type, a: T, b: T) T {
    if (comptime T == BFloat16) return a.div(b);
    if (comptime isComplex(T)) return a.div(b);
    return a / b;
}

fn negValue(comptime T: type, a: T) T {
    if (comptime T == BFloat16) return a.neg();
    if (comptime isComplex(T)) return a.neg();
    return -a;
}

fn absValue(comptime T: type, value: T) T {
    if (comptime T == BFloat16) return value.abs();
    if (comptime isComplex(T)) return T.init(value.magnitude(), 0);
    return switch (@typeInfo(T)) {
        .int => if (@typeInfo(T).int.signedness == .signed and value < 0) -value else value,
        .float => @abs(value),
        else => @compileError("abs requires a numeric array"),
    };
}

fn normalizeDim(dim: isize, rank: usize) ArrayError!usize {
    const signed_rank: isize = @intCast(rank);
    const normalized = if (dim < 0) signed_rank + dim else dim;
    if (normalized < 0 or normalized >= signed_rank) return error.InvalidAxis;
    return @intCast(normalized);
}

fn canonicalAxis(axis: usize, rank: usize) ArrayError!usize {
    if (axis >= rank) return error.InvalidAxis;
    return axis;
}

fn normalizeUniqueAxes(allocator: std.mem.Allocator, axes: []const isize, rank: usize) ArrayError![]usize {
    const normalized = try allocator.alloc(usize, axes.len);
    errdefer allocator.free(normalized);
    var seen = try allocator.alloc(bool, rank);
    defer allocator.free(seen);
    @memset(seen, false);
    for (axes, 0..) |axis_index, i| {
        const axis = try normalizeDim(axis_index, rank);
        if (seen[axis]) return error.InvalidAxis;
        seen[axis] = true;
        normalized[i] = axis;
    }
    return normalized;
}

fn movedimManyAxes(allocator: std.mem.Allocator, rank: usize, sources: []const isize, destinations: []const isize) ArrayError![]usize {
    if (sources.len != destinations.len) return error.ShapeMismatch;
    const normalized_sources = try normalizeUniqueAxes(allocator, sources, rank);
    defer allocator.free(normalized_sources);
    const normalized_destinations = try normalizeUniqueAxes(allocator, destinations, rank);
    defer allocator.free(normalized_destinations);

    var is_source = try allocator.alloc(bool, rank);
    defer allocator.free(is_source);
    @memset(is_source, false);
    for (normalized_sources) |axis| is_source[axis] = true;

    const remaining = try allocator.alloc(usize, rank - sources.len);
    defer allocator.free(remaining);
    var remaining_len: usize = 0;
    for (0..rank) |axis| {
        if (is_source[axis]) continue;
        remaining[remaining_len] = axis;
        remaining_len += 1;
    }

    const axes = try allocator.alloc(usize, rank);
    errdefer allocator.free(axes);
    var read_remaining: usize = 0;
    for (axes, 0..) |*slot, out_axis| {
        var placed = false;
        for (normalized_sources, normalized_destinations) |source_axis, destination_axis| {
            if (destination_axis == out_axis) {
                slot.* = source_axis;
                placed = true;
                break;
            }
        }
        if (!placed) {
            slot.* = remaining[read_remaining];
            read_remaining += 1;
        }
    }
    return axes;
}

fn product(dims: []const usize) usize {
    var out: usize = 1;
    for (dims) |d| out *= d;
    return out;
}

fn normalizeIndex(index: isize, len: usize) ArrayError!usize {
    const signed_len: isize = @intCast(len);
    const normalized = if (index < 0) signed_len + index else index;
    if (normalized < 0 or normalized >= signed_len) return error.IndexOutOfBounds;
    return @intCast(normalized);
}

fn unravelIndexInto(mut_index: usize, dims: []const usize, out: []usize) void {
    var idx = mut_index;
    var i = dims.len;
    while (i > 0) {
        i -= 1;
        if (dims[i] == 0) {
            out[i] = 0;
        } else {
            out[i] = idx % dims[i];
            idx /= dims[i];
        }
    }
}

fn ravelIndex(indices: []const usize, strides: []const usize) usize {
    var offset: usize = 0;
    for (indices, strides) |idx, stride_value| {
        offset += idx * stride_value;
    }
    return offset;
}

fn broadcastShape(allocator: std.mem.Allocator, a: []const usize, b: []const usize) ArrayError![]usize {
    const rank = @max(a.len, b.len);
    const out = try allocator.alloc(usize, rank);
    errdefer allocator.free(out);

    var i: usize = 0;
    while (i < rank) : (i += 1) {
        const ai: usize = if (i >= rank - a.len) a[i - (rank - a.len)] else 1;
        const bi: usize = if (i >= rank - b.len) b[i - (rank - b.len)] else 1;
        if (ai == bi or ai == 1 or bi == 1) {
            out[i] = @max(ai, bi);
        } else {
            return error.ShapeMismatch;
        }
    }
    return out;
}

fn broadcastOffset(out_multi: []const usize, out_rank: usize, in_shape: []const usize, in_strides: []const usize) usize {
    if (in_shape.len == 0) return 0;
    const offset_rank = out_rank - in_shape.len;
    var offset: usize = 0;
    for (in_shape, in_strides, 0..) |dim, stride, i| {
        const coord = if (dim == 1) 0 else out_multi[offset_rank + i];
        offset += coord * stride;
    }
    return offset;
}

fn broadcastAxisExtent(out_rank: usize, in_shape: []const usize, axis: usize) ?usize {
    const leading = out_rank - in_shape.len;
    if (axis < leading) return null;
    return in_shape[axis - leading];
}

pub const Slice = struct {
    start: isize = 0,
    stop: ?isize = null,
    step: isize = 1,
};

pub const ScatterReduce = enum {
    sum,
    prod,
    min,
    max,
};

pub const SearchSide = enum {
    left,
    right,
};

pub const IndexMode = enum {
    raise,
    wrap,
    clip,
};

pub const MeshGridIndexing = enum {
    xy,
    ij,
};

pub const ConvMode = enum {
    full,
    same,
    valid,
};

fn normalizeSlice(s: Slice, len: usize) ArrayError!struct { start: usize, stop: usize, step: usize, count: usize } {
    if (s.step <= 0) return error.InvalidShape;
    const length: isize = @intCast(len);
    var start = if (s.start < 0) length + s.start else s.start;
    var stop = if (s.stop) |v| if (v < 0) length + v else v else length;
    if (start < 0) start = 0;
    if (stop < 0) stop = 0;
    if (start > length) start = length;
    if (stop > length) stop = length;
    const u_start: usize = @intCast(start);
    const u_stop: usize = @intCast(stop);
    const u_step: usize = @intCast(s.step);
    const count = if (u_stop <= u_start) 0 else (u_stop - u_start + u_step - 1) / u_step;
    return .{ .start = u_start, .stop = u_stop, .step = u_step, .count = count };
}

fn inferredShape(allocator: std.mem.Allocator, dims: []const isize, element_count: usize) ArrayError![]usize {
    if (dims.len == 0) {
        if (element_count != 1) return error.ShapeMismatch;
        return allocator.alloc(usize, 0);
    }
    const out = try allocator.alloc(usize, dims.len);
    errdefer allocator.free(out);
    var inferred_axis: ?usize = null;
    var known_product: usize = 1;
    for (dims, 0..) |dim_value, i| {
        if (dim_value == -1) {
            if (inferred_axis != null) return error.InvalidShape;
            inferred_axis = i;
            out[i] = 1;
        } else if (dim_value < 0) {
            return error.InvalidShape;
        } else {
            const dim: usize = @intCast(dim_value);
            out[i] = dim;
            known_product = std.math.mul(usize, known_product, dim) catch return error.InvalidShape;
        }
    }
    if (inferred_axis) |axis| {
        if (known_product == 0 or element_count % known_product != 0) return error.ShapeMismatch;
        out[axis] = element_count / known_product;
    } else if (known_product != element_count) {
        return error.ShapeMismatch;
    }
    return out;
}

fn flattenShape(
    allocator: std.mem.Allocator,
    shape: []const usize,
    start_axis: isize,
    end_axis: isize,
) ArrayError![]usize {
    if (shape.len == 0) return error.InvalidAxis;
    const start = try normalizeDim(start_axis, shape.len);
    const end = try normalizeDim(end_axis, shape.len);
    if (start > end) return error.InvalidAxis;
    const out_rank = shape.len - (end - start);
    const out = try allocator.alloc(usize, out_rank);
    errdefer allocator.free(out);
    var write: usize = 0;
    for (shape[0..start]) |dim| {
        out[write] = dim;
        write += 1;
    }
    out[write] = product(shape[start .. end + 1]);
    write += 1;
    for (shape[end + 1 ..]) |dim| {
        out[write] = dim;
        write += 1;
    }
    return out;
}

fn unflattenShape(
    allocator: std.mem.Allocator,
    shape: []const usize,
    axis_index: isize,
    dims: []const usize,
) ArrayError![]usize {
    if (shape.len == 0) return error.InvalidAxis;
    const axis = try normalizeDim(axis_index, shape.len);
    const expanded = try numelFrom(dims);
    if (expanded != shape[axis]) return error.ShapeMismatch;
    const out = try allocator.alloc(usize, shape.len - 1 + dims.len);
    errdefer allocator.free(out);
    var write: usize = 0;
    for (shape[0..axis]) |dim| {
        out[write] = dim;
        write += 1;
    }
    for (dims) |dim| {
        out[write] = dim;
        write += 1;
    }
    for (shape[axis + 1 ..]) |dim| {
        out[write] = dim;
        write += 1;
    }
    return out;
}

fn validateStridedBounds(data_len: usize, offset: usize, dims: []const usize, stride_values: []const usize) ArrayError!void {
    if (dims.len != stride_values.len) return error.InvalidShape;
    if (offset > data_len) return error.IndexOutOfBounds;
    var empty = false;
    var max_offset = offset;
    for (dims, stride_values) |dim, stride_value| {
        if (dim == 0) {
            empty = true;
            continue;
        }
        const span = std.math.mul(usize, dim - 1, stride_value) catch return error.InvalidShape;
        max_offset = std.math.add(usize, max_offset, span) catch return error.InvalidShape;
    }
    if (empty) return;
    if (max_offset >= data_len) return error.IndexOutOfBounds;
}

pub fn ArrayView(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        data: []T,
        shape: []usize,
        strides: []usize,
        offset: usize = 0,
        device: Device = .cpu,

        pub const Scalar = T;
        pub const dtype = DType.of(T);

        pub fn fromArray(input: Array(T)) ArrayError!Self {
            const shape = try input.allocator.dupe(usize, input.shape);
            errdefer input.allocator.free(shape);
            const strides = try input.allocator.dupe(usize, input.strides);
            return .{
                .allocator = input.allocator,
                .data = input.data,
                .shape = shape,
                .strides = strides,
                .offset = 0,
                .device = input.device,
            };
        }

        pub fn init(
            allocator: std.mem.Allocator,
            data: []T,
            dims: []const usize,
            stride_values: []const usize,
            offset: usize,
            device: Device,
        ) ArrayError!Self {
            if (dims.len != stride_values.len) return error.InvalidShape;
            const shape = try allocator.dupe(usize, dims);
            errdefer allocator.free(shape);
            const strides = try allocator.dupe(usize, stride_values);
            return .{
                .allocator = allocator,
                .data = data,
                .shape = shape,
                .strides = strides,
                .offset = offset,
                .device = device,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.shape);
            self.allocator.free(self.strides);
            self.* = undefined;
        }

        pub fn clone(self: Self) ArrayError!Self {
            return Self.init(self.allocator, self.data, self.shape, self.strides, self.offset, self.device);
        }

        pub fn ndim(self: Self) usize {
            return self.shape.len;
        }

        pub fn dim(self: Self) usize {
            return self.ndim();
        }

        pub fn rank(self: Self) usize {
            return self.ndim();
        }

        pub fn numDims(self: Self) usize {
            return self.ndim();
        }

        pub fn numel(self: Self) usize {
            return product(self.shape);
        }

        pub fn nelement(self: Self) usize {
            return self.numel();
        }

        pub fn isEmpty(self: Self) bool {
            return self.numel() == 0;
        }

        pub fn size(self: Self, axis_opt: ?isize) ArrayError!usize {
            if (axis_opt) |axis_index| return self.shape[try normalizeDim(axis_index, self.shape.len)];
            return self.numel();
        }

        pub fn shapeAt(self: Self, axis_index: isize) ArrayError!usize {
            return self.shape[try normalizeDim(axis_index, self.shape.len)];
        }

        pub fn len(self: Self) ArrayError!usize {
            if (self.shape.len == 0) return error.InvalidShape;
            return self.shape[0];
        }

        pub fn stride(self: Self, axis_index: isize) ArrayError!usize {
            return self.strides[try normalizeDim(axis_index, self.shape.len)];
        }

        pub fn strideAt(self: Self, axis_index: isize) ArrayError!usize {
            return self.stride(axis_index);
        }

        pub fn elementSize(self: Self) usize {
            _ = self;
            return @sizeOf(T);
        }

        pub fn nbytes(self: Self) usize {
            return self.numel() * @sizeOf(T);
        }

        pub fn sameShape(self: Self, other: Self) bool {
            return std.mem.eql(usize, self.shape, other.shape);
        }

        pub fn isScalar(self: Self) bool {
            return self.shape.len == 0 or (self.shape.len == 1 and self.shape[0] == 1);
        }

        pub fn isContiguous(self: Self) bool {
            var expected: usize = 1;
            var i = self.shape.len;
            while (i > 0) {
                i -= 1;
                if (self.strides[i] != expected) return false;
                expected *= self.shape[i];
            }
            return true;
        }

        pub fn is_contiguous(self: Self) bool {
            return self.isContiguous();
        }

        fn offsetOf(self: Self, indices: []const usize) ArrayError!usize {
            if (indices.len != self.shape.len) return error.InvalidShape;
            var offset = self.offset;
            for (indices, self.shape, self.strides) |idx, extent, stride_value| {
                if (idx >= extent) return error.IndexOutOfBounds;
                offset += idx * stride_value;
            }
            return offset;
        }

        fn offsetOfSigned(self: Self, indices: []const isize) ArrayError!usize {
            if (indices.len != self.shape.len) return error.InvalidShape;
            var offset = self.offset;
            for (indices, self.shape, self.strides) |idx, extent, stride_value| {
                offset += (try normalizeIndex(idx, extent)) * stride_value;
            }
            return offset;
        }

        pub fn get(self: Self, indices: []const usize) ArrayError!T {
            return self.data[try self.offsetOf(indices)];
        }

        pub fn getSigned(self: Self, indices: []const isize) ArrayError!T {
            return self.data[try self.offsetOfSigned(indices)];
        }

        pub fn at(self: Self, indices: []const usize) ArrayError!T {
            return self.get(indices);
        }

        pub fn atSigned(self: Self, indices: []const isize) ArrayError!T {
            return self.getSigned(indices);
        }

        pub fn set(self: Self, indices: []const usize, value: T) ArrayError!void {
            self.data[try self.offsetOf(indices)] = value;
        }

        pub fn setSigned(self: Self, indices: []const isize, value: T) ArrayError!void {
            self.data[try self.offsetOfSigned(indices)] = value;
        }

        pub fn put(self: Self, indices: []const usize, value: T) ArrayError!void {
            return self.set(indices, value);
        }

        pub fn putSigned(self: Self, indices: []const isize, value: T) ArrayError!void {
            return self.setSigned(indices, value);
        }

        pub fn item(self: Self) ArrayError!T {
            if (!self.isScalar()) return error.ShapeMismatch;
            if (self.numel() == 0) return error.EmptyArray;
            return self.data[self.offset];
        }

        pub fn toArray(self: Self) ArrayError!Array(T) {
            var out = try Array(T).empty(self.allocator, self.shape);
            errdefer out.deinit();
            if (out.data.len == 0) return out;
            const multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(multi);
            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, self.shape, multi);
                slot.* = self.data[self.offset + ravelIndex(multi, self.strides)];
            }
            return out;
        }

        pub fn contiguous(self: Self) ArrayError!Array(T) {
            return self.toArray();
        }

        pub fn emptyLike(self: Self) ArrayError!Array(T) {
            return Array(T).empty(self.allocator, self.shape);
        }

        pub fn zerosLike(self: Self) ArrayError!Array(T) {
            return Array(T).zeros(self.allocator, self.shape);
        }

        pub fn onesLike(self: Self) ArrayError!Array(T) {
            return Array(T).ones(self.allocator, self.shape);
        }

        pub fn fullLike(self: Self, value: T) ArrayError!Array(T) {
            return Array(T).full(self.allocator, self.shape, value);
        }

        pub fn newEmpty(self: Self, dims: []const usize) ArrayError!Array(T) {
            return Array(T).empty(self.allocator, dims);
        }

        pub fn new_empty(self: Self, dims: []const usize) ArrayError!Array(T) {
            return self.newEmpty(dims);
        }

        pub fn newZeros(self: Self, dims: []const usize) ArrayError!Array(T) {
            return Array(T).zeros(self.allocator, dims);
        }

        pub fn new_zeros(self: Self, dims: []const usize) ArrayError!Array(T) {
            return self.newZeros(dims);
        }

        pub fn newOnes(self: Self, dims: []const usize) ArrayError!Array(T) {
            return Array(T).ones(self.allocator, dims);
        }

        pub fn new_ones(self: Self, dims: []const usize) ArrayError!Array(T) {
            return self.newOnes(dims);
        }

        pub fn newFull(self: Self, dims: []const usize, value: T) ArrayError!Array(T) {
            return Array(T).full(self.allocator, dims, value);
        }

        pub fn new_full(self: Self, dims: []const usize, value: T) ArrayError!Array(T) {
            return self.newFull(dims, value);
        }

        pub fn astype(self: Self, comptime U: type) ArrayError!Array(U) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.astype(U);
        }

        pub fn to(self: Self, device: Device) ArrayError!Self {
            if (!device.isAvailable()) return error.InvalidDevice;
            var out = try self.clone();
            out.device = device;
            return out;
        }

        pub fn cpu(self: Self) ArrayError!Self {
            return self.to(.cpu);
        }

        pub fn cuda(self: Self, index: usize) ArrayError!Self {
            return self.to(Device.cuda(index));
        }

        pub fn asStrided(self: Self, dims: []const usize, stride_values: []const usize, offset: usize) ArrayError!Self {
            const view_offset = std.math.add(usize, self.offset, offset) catch return error.InvalidShape;
            try validateStridedBounds(self.data.len, view_offset, dims, stride_values);
            return Self.init(self.allocator, self.data, dims, stride_values, view_offset, self.device);
        }

        pub fn unfold(self: Self, axis_index: isize, window_size: usize, step: usize) ArrayError!Self {
            if (window_size == 0 or step == 0) return error.InvalidShape;
            if (self.shape.len == 0) return error.InvalidAxis;
            const axis = try normalizeDim(axis_index, self.shape.len);
            if (window_size > self.shape[axis]) return error.InvalidShape;
            const window_count = (self.shape[axis] - window_size) / step + 1;
            const dims = try self.allocator.alloc(usize, self.shape.len + 1);
            defer self.allocator.free(dims);
            const stride_values = try self.allocator.alloc(usize, self.strides.len + 1);
            defer self.allocator.free(stride_values);
            for (self.shape[0..axis], 0..) |extent, i| dims[i] = extent;
            dims[axis] = window_count;
            for (self.shape[axis + 1 ..], axis + 1..) |extent, i| dims[i] = extent;
            dims[dims.len - 1] = window_size;
            for (self.strides[0..axis], 0..) |stride_value, i| stride_values[i] = stride_value;
            stride_values[axis] = self.strides[axis] * step;
            for (self.strides[axis + 1 ..], axis + 1..) |stride_value, i| stride_values[i] = stride_value;
            stride_values[stride_values.len - 1] = self.strides[axis];
            return self.asStrided(dims, stride_values, 0);
        }

        fn broadcastOffsetOf(self: Self, out_multi: []const usize, out_rank: usize) usize {
            return self.offset + broadcastOffset(out_multi, out_rank, self.shape, self.strides);
        }

        fn opAdd(a: T, b: T) T {
            return addValue(T, a, b);
        }

        fn opSub(a: T, b: T) T {
            return subValue(T, a, b);
        }

        fn opMul(a: T, b: T) T {
            return mulValue(T, a, b);
        }

        fn opDiv(a: T, b: T) T {
            return divValue(T, a, b);
        }

        fn opNeg(a: T) T {
            return negValue(T, a);
        }

        fn opAbs(a: T) T {
            return absValue(T, a);
        }

        fn unary(self: Self, comptime op: fn (T) T) ArrayError!Array(T) {
            var out = try Array(T).empty(self.allocator, self.shape);
            errdefer out.deinit();
            if (out.data.len == 0) return out;
            const multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(multi);
            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, self.shape, multi);
                slot.* = op(self.data[self.offset + ravelIndex(multi, self.strides)]);
            }
            return out;
        }

        fn binaryScalar(self: Self, scalar: T, comptime op: fn (T, T) T) ArrayError!Array(T) {
            var out = try Array(T).empty(self.allocator, self.shape);
            errdefer out.deinit();
            if (out.data.len == 0) return out;
            const multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(multi);
            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, self.shape, multi);
                slot.* = op(self.data[self.offset + ravelIndex(multi, self.strides)], scalar);
            }
            return out;
        }

        fn binaryView(self: Self, other: Self, comptime op: fn (T, T) T) ArrayError!Array(T) {
            const out_shape = try broadcastShape(self.allocator, self.shape, other.shape);
            defer self.allocator.free(out_shape);
            var out = try Array(T).empty(self.allocator, out_shape);
            errdefer out.deinit();
            if (out.data.len == 0) return out;
            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, out_shape, out_multi);
                slot.* = op(self.data[self.broadcastOffsetOf(out_multi, out_shape.len)], other.data[other.broadcastOffsetOf(out_multi, out_shape.len)]);
            }
            return out;
        }

        fn compareView(self: Self, other: Self, comptime op: fn (T, T) bool) ArrayError!Array(bool) {
            const out_shape = try broadcastShape(self.allocator, self.shape, other.shape);
            defer self.allocator.free(out_shape);
            var out = try Array(bool).empty(self.allocator, out_shape);
            errdefer out.deinit();
            if (out.data.len == 0) return out;
            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, out_shape, out_multi);
                slot.* = op(self.data[self.broadcastOffsetOf(out_multi, out_shape.len)], other.data[other.broadcastOffsetOf(out_multi, out_shape.len)]);
            }
            return out;
        }

        fn compareScalar(self: Self, scalar: T, comptime op: fn (T, T) bool) ArrayError!Array(bool) {
            var out = try Array(bool).empty(self.allocator, self.shape);
            errdefer out.deinit();
            if (out.data.len == 0) return out;
            const multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(multi);
            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, self.shape, multi);
                slot.* = op(self.data[self.offset + ravelIndex(multi, self.strides)], scalar);
            }
            return out;
        }

        pub fn fill(self: Self, value: T) ArrayError!void {
            const multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(multi);
            for (0..self.numel()) |flat| {
                unravelIndexInto(flat, self.shape, multi);
                self.data[self.offset + ravelIndex(multi, self.strides)] = value;
            }
        }

        pub fn copyFromView(self: Self, source: Self) ArrayError!void {
            const out_shape = try broadcastShape(self.allocator, self.shape, source.shape);
            defer self.allocator.free(out_shape);
            if (!std.mem.eql(usize, out_shape, self.shape)) return error.ShapeMismatch;
            const multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(multi);
            for (0..self.numel()) |flat| {
                unravelIndexInto(flat, self.shape, multi);
                self.data[self.offset + ravelIndex(multi, self.strides)] = source.data[source.broadcastOffsetOf(multi, self.shape.len)];
            }
        }

        pub fn copyFrom(self: Self, source: Self) ArrayError!void {
            return self.copyFromView(source);
        }

        pub fn copyFromArray(self: Self, source: Array(T)) ArrayError!void {
            var source_view = try source.asView();
            defer source_view.deinit();
            return self.copyFromView(source_view);
        }

        fn viewValueAtFlat(source: Self, flat: usize, scratch: []usize) T {
            if (source.numel() == 1) return source.data[source.offset];
            unravelIndexInto(flat, source.shape, scratch);
            return source.data[source.offset + ravelIndex(scratch, source.strides)];
        }

        pub fn maskedFill(self: Self, mask: Array(bool), value: T) ArrayError!void {
            const out_shape = try broadcastShape(self.allocator, self.shape, mask.shape);
            defer self.allocator.free(out_shape);
            if (!std.mem.eql(usize, out_shape, self.shape)) return error.ShapeMismatch;
            const multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(multi);
            for (0..self.numel()) |flat| {
                unravelIndexInto(flat, self.shape, multi);
                if (mask.data[broadcastOffset(multi, self.shape.len, mask.shape, mask.strides)]) {
                    self.data[self.offset + ravelIndex(multi, self.strides)] = value;
                }
            }
        }

        pub fn maskedFillAssign(self: Self, mask: Array(bool), value: T) ArrayError!void {
            return self.maskedFill(mask, value);
        }

        pub fn maskedCopyFromView(self: Self, mask: Array(bool), values: Self) ArrayError!void {
            const out_shape = try broadcastShape(self.allocator, self.shape, mask.shape);
            defer self.allocator.free(out_shape);
            if (!std.mem.eql(usize, out_shape, self.shape)) return error.ShapeMismatch;
            const multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(multi);
            var count: usize = 0;
            for (0..self.numel()) |flat| {
                unravelIndexInto(flat, self.shape, multi);
                if (mask.data[broadcastOffset(multi, self.shape.len, mask.shape, mask.strides)]) count += 1;
            }
            if (values.numel() != 1 and values.numel() != count) return error.ShapeMismatch;
            const value_multi = try self.allocator.alloc(usize, values.shape.len);
            defer self.allocator.free(value_multi);
            var write: usize = 0;
            for (0..self.numel()) |flat| {
                unravelIndexInto(flat, self.shape, multi);
                if (mask.data[broadcastOffset(multi, self.shape.len, mask.shape, mask.strides)]) {
                    self.data[self.offset + ravelIndex(multi, self.strides)] = viewValueAtFlat(values, write, value_multi);
                    write += 1;
                }
            }
        }

        pub fn maskedCopyFrom(self: Self, mask: Array(bool), values: Self) ArrayError!void {
            return self.maskedCopyFromView(mask, values);
        }

        pub fn maskedCopyFromArray(self: Self, mask: Array(bool), values: Array(T)) ArrayError!void {
            var values_view = try values.asView();
            defer values_view.deinit();
            return self.maskedCopyFromView(mask, values_view);
        }

        pub fn copyWhereFromView(self: Self, mask: Array(bool), source: Self) ArrayError!void {
            const tmp_shape = try broadcastShape(self.allocator, self.shape, mask.shape);
            defer self.allocator.free(tmp_shape);
            const out_shape = try broadcastShape(self.allocator, tmp_shape, source.shape);
            defer self.allocator.free(out_shape);
            if (!std.mem.eql(usize, out_shape, self.shape)) return error.ShapeMismatch;
            const multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(multi);
            for (0..self.numel()) |flat| {
                unravelIndexInto(flat, self.shape, multi);
                if (mask.data[broadcastOffset(multi, self.shape.len, mask.shape, mask.strides)]) {
                    self.data[self.offset + ravelIndex(multi, self.strides)] = source.data[source.broadcastOffsetOf(multi, self.shape.len)];
                }
            }
        }

        pub fn copyWhereAssign(self: Self, mask: Array(bool), source: Self) ArrayError!void {
            return self.copyWhereFromView(mask, source);
        }

        pub fn copyWhereFromArray(self: Self, mask: Array(bool), source: Array(T)) ArrayError!void {
            var source_view = try source.asView();
            defer source_view.deinit();
            return self.copyWhereFromView(mask, source_view);
        }

        pub fn copyWhereAssignView(self: Self, mask: Array(bool), source: Self) ArrayError!void {
            return self.copyWhereFromView(mask, source);
        }

        fn assignView(self: Self, source: Self, comptime op: fn (T, T) T) ArrayError!void {
            const out_shape = try broadcastShape(self.allocator, self.shape, source.shape);
            defer self.allocator.free(out_shape);
            if (!std.mem.eql(usize, out_shape, self.shape)) return error.ShapeMismatch;
            const multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(multi);
            for (0..self.numel()) |flat| {
                unravelIndexInto(flat, self.shape, multi);
                const dst_offset = self.offset + ravelIndex(multi, self.strides);
                const src_offset = source.broadcastOffsetOf(multi, self.shape.len);
                self.data[dst_offset] = op(self.data[dst_offset], source.data[src_offset]);
            }
        }

        fn assignArray(self: Self, source: Array(T), comptime op: fn (T, T) T) ArrayError!void {
            var source_view = try source.asView();
            defer source_view.deinit();
            return self.assignView(source_view, op);
        }

        fn assignScalar(self: Self, scalar: T, comptime op: fn (T, T) T) ArrayError!void {
            const multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(multi);
            for (0..self.numel()) |flat| {
                unravelIndexInto(flat, self.shape, multi);
                const dst_offset = self.offset + ravelIndex(multi, self.strides);
                self.data[dst_offset] = op(self.data[dst_offset], scalar);
            }
        }

        pub fn addAssign(self: Self, source: Self) ArrayError!void {
            ensureNumeric(T);
            return self.assignView(source, opAdd);
        }

        pub fn addAssignView(self: Self, source: Self) ArrayError!void {
            return self.addAssign(source);
        }

        pub fn subAssign(self: Self, source: Self) ArrayError!void {
            ensureNumeric(T);
            return self.assignView(source, opSub);
        }

        pub fn subAssignView(self: Self, source: Self) ArrayError!void {
            return self.subAssign(source);
        }

        pub fn mulAssign(self: Self, source: Self) ArrayError!void {
            ensureNumeric(T);
            return self.assignView(source, opMul);
        }

        pub fn mulAssignView(self: Self, source: Self) ArrayError!void {
            return self.mulAssign(source);
        }

        pub fn divAssign(self: Self, source: Self) ArrayError!void {
            ensureNumeric(T);
            return self.assignView(source, opDiv);
        }

        pub fn divAssignView(self: Self, source: Self) ArrayError!void {
            return self.divAssign(source);
        }

        pub fn addAssignArray(self: Self, source: Array(T)) ArrayError!void {
            ensureNumeric(T);
            return self.assignArray(source, opAdd);
        }

        pub fn subAssignArray(self: Self, source: Array(T)) ArrayError!void {
            ensureNumeric(T);
            return self.assignArray(source, opSub);
        }

        pub fn mulAssignArray(self: Self, source: Array(T)) ArrayError!void {
            ensureNumeric(T);
            return self.assignArray(source, opMul);
        }

        pub fn divAssignArray(self: Self, source: Array(T)) ArrayError!void {
            ensureNumeric(T);
            return self.assignArray(source, opDiv);
        }

        pub fn addScalarAssign(self: Self, scalar: T) ArrayError!void {
            ensureNumeric(T);
            return self.assignScalar(scalar, opAdd);
        }

        pub fn subScalarAssign(self: Self, scalar: T) ArrayError!void {
            ensureNumeric(T);
            return self.assignScalar(scalar, opSub);
        }

        pub fn mulScalarAssign(self: Self, scalar: T) ArrayError!void {
            ensureNumeric(T);
            return self.assignScalar(scalar, opMul);
        }

        pub fn divScalarAssign(self: Self, scalar: T) ArrayError!void {
            ensureNumeric(T);
            return self.assignScalar(scalar, opDiv);
        }

        pub fn neg(self: Self) ArrayError!Array(T) {
            ensureNumeric(T);
            return self.unary(opNeg);
        }

        pub fn negative(self: Self) ArrayError!Array(T) {
            return self.neg();
        }

        pub fn positive(self: Self) ArrayError!Array(T) {
            ensureNumeric(T);
            return self.toArray();
        }

        pub fn abs(self: Self) ArrayError!Array(T) {
            ensureNumeric(T);
            return self.unary(opAbs);
        }

        pub fn absolute(self: Self) ArrayError!Array(T) {
            return self.abs();
        }

        pub fn fabs(self: Self) ArrayError!Array(T) {
            return self.abs();
        }

        pub fn add(self: Self, other: Self) ArrayError!Array(T) {
            ensureNumeric(T);
            return self.binaryView(other, opAdd);
        }

        pub fn sub(self: Self, other: Self) ArrayError!Array(T) {
            ensureNumeric(T);
            return self.binaryView(other, opSub);
        }

        pub fn mul(self: Self, other: Self) ArrayError!Array(T) {
            ensureNumeric(T);
            return self.binaryView(other, opMul);
        }

        pub fn div(self: Self, other: Self) ArrayError!Array(T) {
            ensureNumeric(T);
            return self.binaryView(other, opDiv);
        }

        fn ownedBinary(self: Self, other: Self, comptime method: fn (Array(T), Array(T)) ArrayError!Array(T)) ArrayError!Array(T) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            var rhs = try other.toArray();
            defer rhs.deinit();
            return method(lhs, rhs);
        }

        fn ownedUnary(self: Self, comptime R: type, comptime method: fn (Array(T)) ArrayError!R) ArrayError!R {
            var owned = try self.toArray();
            defer owned.deinit();
            return method(owned);
        }

        fn ownedWith(self: Self, arg: anytype, comptime R: type, comptime method: anytype) ArrayError!R {
            var owned = try self.toArray();
            defer owned.deinit();
            return method(owned, arg);
        }

        fn ownedWith2(self: Self, arg1: anytype, arg2: anytype, comptime R: type, comptime method: anytype) ArrayError!R {
            var owned = try self.toArray();
            defer owned.deinit();
            return method(owned, arg1, arg2);
        }

        fn ownedWith3(self: Self, arg1: anytype, arg2: anytype, arg3: anytype, comptime R: type, comptime method: anytype) ArrayError!R {
            var owned = try self.toArray();
            defer owned.deinit();
            return method(owned, arg1, arg2, arg3);
        }

        fn ownedTernary(self: Self, second: Self, third: Self, comptime method: anytype) ArrayError!Array(T) {
            var first_owned = try self.toArray();
            defer first_owned.deinit();
            var second_owned = try second.toArray();
            defer second_owned.deinit();
            var third_owned = try third.toArray();
            defer third_owned.deinit();
            return method(first_owned, second_owned, third_owned);
        }

        fn ownedTernaryScalar(self: Self, second: Self, third: Self, scalar: T, comptime method: anytype) ArrayError!Array(T) {
            var first_owned = try self.toArray();
            defer first_owned.deinit();
            var second_owned = try second.toArray();
            defer second_owned.deinit();
            var third_owned = try third.toArray();
            defer third_owned.deinit();
            return method(first_owned, second_owned, third_owned, scalar);
        }

        pub fn addArray(self: Self, other: Array(T)) ArrayError!Array(T) {
            var other_view = try other.asView();
            defer other_view.deinit();
            return self.add(other_view);
        }

        pub fn subArray(self: Self, other: Array(T)) ArrayError!Array(T) {
            var other_view = try other.asView();
            defer other_view.deinit();
            return self.sub(other_view);
        }

        pub fn mulArray(self: Self, other: Array(T)) ArrayError!Array(T) {
            var other_view = try other.asView();
            defer other_view.deinit();
            return self.mul(other_view);
        }

        pub fn divArray(self: Self, other: Array(T)) ArrayError!Array(T) {
            var other_view = try other.asView();
            defer other_view.deinit();
            return self.div(other_view);
        }

        pub fn pow(self: Self, other: Self) ArrayError!Array(T) {
            return self.ownedBinary(other, Array(T).pow);
        }

        pub fn powArray(self: Self, other: Array(T)) ArrayError!Array(T) {
            var other_view = try other.asView();
            defer other_view.deinit();
            return self.pow(other_view);
        }

        pub fn floorDiv(self: Self, other: Self) ArrayError!Array(T) {
            return self.ownedBinary(other, Array(T).floorDiv);
        }

        pub fn floorDivArray(self: Self, other: Array(T)) ArrayError!Array(T) {
            var other_view = try other.asView();
            defer other_view.deinit();
            return self.floorDiv(other_view);
        }

        pub fn mod(self: Self, other: Self) ArrayError!Array(T) {
            return self.ownedBinary(other, Array(T).mod);
        }

        pub fn modArray(self: Self, other: Array(T)) ArrayError!Array(T) {
            var other_view = try other.asView();
            defer other_view.deinit();
            return self.mod(other_view);
        }

        pub fn remainder(self: Self, other: Self) ArrayError!Array(T) {
            return self.mod(other);
        }

        pub fn remainderArray(self: Self, other: Array(T)) ArrayError!Array(T) {
            return self.modArray(other);
        }

        pub fn maximum(self: Self, other: Self) ArrayError!Array(T) {
            return self.ownedBinary(other, Array(T).maximum);
        }

        pub fn maximumArray(self: Self, other: Array(T)) ArrayError!Array(T) {
            var other_view = try other.asView();
            defer other_view.deinit();
            return self.maximum(other_view);
        }

        pub fn minimum(self: Self, other: Self) ArrayError!Array(T) {
            return self.ownedBinary(other, Array(T).minimum);
        }

        pub fn minimumArray(self: Self, other: Array(T)) ArrayError!Array(T) {
            var other_view = try other.asView();
            defer other_view.deinit();
            return self.minimum(other_view);
        }

        pub fn fmax(self: Self, other: Self) ArrayError!Array(T) {
            return self.ownedBinary(other, Array(T).fmax);
        }

        pub fn fmaxArray(self: Self, other: Array(T)) ArrayError!Array(T) {
            var other_view = try other.asView();
            defer other_view.deinit();
            return self.fmax(other_view);
        }

        pub fn fmin(self: Self, other: Self) ArrayError!Array(T) {
            return self.ownedBinary(other, Array(T).fmin);
        }

        pub fn fminArray(self: Self, other: Array(T)) ArrayError!Array(T) {
            var other_view = try other.asView();
            defer other_view.deinit();
            return self.fmin(other_view);
        }

        pub fn addPromote(self: Self, comptime U: type, other: Array(U)) ArrayError!Array(promoteType(T, U)) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            return lhs.addPromote(U, other);
        }

        pub fn subPromote(self: Self, comptime U: type, other: Array(U)) ArrayError!Array(promoteType(T, U)) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            return lhs.subPromote(U, other);
        }

        pub fn mulPromote(self: Self, comptime U: type, other: Array(U)) ArrayError!Array(promoteType(T, U)) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            return lhs.mulPromote(U, other);
        }

        pub fn divPromote(self: Self, comptime U: type, other: Array(U)) ArrayError!Array(promoteType(T, U)) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            return lhs.divPromote(U, other);
        }

        pub fn maximumPromote(self: Self, comptime U: type, other: Array(U)) ArrayError!Array(promoteType(T, U)) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            return lhs.maximumPromote(U, other);
        }

        pub fn minimumPromote(self: Self, comptime U: type, other: Array(U)) ArrayError!Array(promoteType(T, U)) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            return lhs.minimumPromote(U, other);
        }

        pub fn hypot(self: Self, other: Self) ArrayError!Array(T) {
            return self.ownedBinary(other, Array(T).hypot);
        }

        pub fn hypotArray(self: Self, other: Array(T)) ArrayError!Array(T) {
            var other_view = try other.asView();
            defer other_view.deinit();
            return self.hypot(other_view);
        }

        pub fn atan2(self: Self, other: Self) ArrayError!Array(T) {
            return self.ownedBinary(other, Array(T).atan2);
        }

        pub fn arctan2(self: Self, other: Self) ArrayError!Array(T) {
            return self.atan2(other);
        }

        pub fn atan2Array(self: Self, other: Array(T)) ArrayError!Array(T) {
            var other_view = try other.asView();
            defer other_view.deinit();
            return self.atan2(other_view);
        }

        pub fn arctan2Array(self: Self, other: Array(T)) ArrayError!Array(T) {
            return self.atan2Array(other);
        }

        pub fn nextAfter(self: Self, other: Self) ArrayError!Array(T) {
            return self.ownedBinary(other, Array(T).nextAfter);
        }

        pub fn nextafter(self: Self, other: Self) ArrayError!Array(T) {
            return self.nextAfter(other);
        }

        pub fn copysign(self: Self, sign_values: Self) ArrayError!Array(T) {
            return self.ownedBinary(sign_values, Array(T).copysign);
        }

        pub fn heaviside(self: Self, values_at_zero: Self) ArrayError!Array(T) {
            return self.ownedBinary(values_at_zero, Array(T).heaviside);
        }

        pub fn logAddExp(self: Self, other: Self) ArrayError!Array(T) {
            return self.ownedBinary(other, Array(T).logAddExp);
        }

        pub fn logaddexp(self: Self, other: Self) ArrayError!Array(T) {
            return self.logAddExp(other);
        }

        pub fn logAddExp2(self: Self, other: Self) ArrayError!Array(T) {
            return self.ownedBinary(other, Array(T).logAddExp2);
        }

        pub fn logaddexp2(self: Self, other: Self) ArrayError!Array(T) {
            return self.logAddExp2(other);
        }

        pub fn xlogy(self: Self, other: Self) ArrayError!Array(T) {
            return self.ownedBinary(other, Array(T).xlogy);
        }

        pub fn lerp(self: Self, end: Self, weight: Self) ArrayError!Array(T) {
            return self.ownedTernary(end, weight, Array(T).lerp);
        }

        pub fn lerpArray(self: Self, end: Array(T), weight: Array(T)) ArrayError!Array(T) {
            var first_owned = try self.toArray();
            defer first_owned.deinit();
            return first_owned.lerp(end, weight);
        }

        pub fn lerpScalar(self: Self, end: Self, weight: T) ArrayError!Array(T) {
            var first_owned = try self.toArray();
            defer first_owned.deinit();
            var end_owned = try end.toArray();
            defer end_owned.deinit();
            return first_owned.lerpScalar(end_owned, weight);
        }

        pub fn addcmul(self: Self, input1: Self, input2: Self, value: T) ArrayError!Array(T) {
            return self.ownedTernaryScalar(input1, input2, value, Array(T).addcmul);
        }

        pub fn addCMul(self: Self, input1: Self, input2: Self, value: T) ArrayError!Array(T) {
            return self.addcmul(input1, input2, value);
        }

        pub fn addcmulArray(self: Self, input1: Array(T), input2: Array(T), value: T) ArrayError!Array(T) {
            var first_owned = try self.toArray();
            defer first_owned.deinit();
            return first_owned.addcmul(input1, input2, value);
        }

        pub fn addcdiv(self: Self, input1: Self, input2: Self, value: T) ArrayError!Array(T) {
            return self.ownedTernaryScalar(input1, input2, value, Array(T).addcdiv);
        }

        pub fn addCDiv(self: Self, input1: Self, input2: Self, value: T) ArrayError!Array(T) {
            return self.addcdiv(input1, input2, value);
        }

        pub fn addcdivArray(self: Self, input1: Array(T), input2: Array(T), value: T) ArrayError!Array(T) {
            var first_owned = try self.toArray();
            defer first_owned.deinit();
            return first_owned.addcdiv(input1, input2, value);
        }

        pub fn clipArray(self: Self, min_values: Self, max_values: Self) ArrayError!Array(T) {
            var values = try self.toArray();
            defer values.deinit();
            var min_owned = try min_values.toArray();
            defer min_owned.deinit();
            var max_owned = try max_values.toArray();
            defer max_owned.deinit();
            return values.clipArray(min_owned, max_owned);
        }

        pub fn addScalar(self: Self, scalar: T) ArrayError!Array(T) {
            ensureNumeric(T);
            return self.binaryScalar(scalar, opAdd);
        }

        pub fn subScalar(self: Self, scalar: T) ArrayError!Array(T) {
            ensureNumeric(T);
            return self.binaryScalar(scalar, opSub);
        }

        pub fn mulScalar(self: Self, scalar: T) ArrayError!Array(T) {
            ensureNumeric(T);
            return self.binaryScalar(scalar, opMul);
        }

        pub fn divScalar(self: Self, scalar: T) ArrayError!Array(T) {
            ensureNumeric(T);
            return self.binaryScalar(scalar, opDiv);
        }

        pub fn powScalar(self: Self, scalar: T) ArrayError!Array(T) {
            return self.ownedWith(scalar, Array(T), Array(T).powScalar);
        }

        pub fn floorDivScalar(self: Self, scalar: T) ArrayError!Array(T) {
            return self.ownedWith(scalar, Array(T), Array(T).floorDivScalar);
        }

        pub fn modScalar(self: Self, scalar: T) ArrayError!Array(T) {
            return self.ownedWith(scalar, Array(T), Array(T).modScalar);
        }

        pub fn remainderScalar(self: Self, scalar: T) ArrayError!Array(T) {
            return self.modScalar(scalar);
        }

        pub fn maximumScalar(self: Self, scalar: T) ArrayError!Array(T) {
            return self.ownedWith(scalar, Array(T), Array(T).maximumScalar);
        }

        pub fn minimumScalar(self: Self, scalar: T) ArrayError!Array(T) {
            return self.ownedWith(scalar, Array(T), Array(T).minimumScalar);
        }

        pub fn clipMin(self: Self, min_value: T) ArrayError!Array(T) {
            return self.maximumScalar(min_value);
        }

        pub fn clampMin(self: Self, min_value: T) ArrayError!Array(T) {
            return self.clipMin(min_value);
        }

        pub fn clipMax(self: Self, max_value: T) ArrayError!Array(T) {
            return self.minimumScalar(max_value);
        }

        pub fn clampMax(self: Self, max_value: T) ArrayError!Array(T) {
            return self.clipMax(max_value);
        }

        pub fn fmaxScalar(self: Self, scalar: T) ArrayError!Array(T) {
            return self.ownedWith(scalar, Array(T), Array(T).fmaxScalar);
        }

        pub fn fminScalar(self: Self, scalar: T) ArrayError!Array(T) {
            return self.ownedWith(scalar, Array(T), Array(T).fminScalar);
        }

        pub fn hypotScalar(self: Self, scalar: T) ArrayError!Array(T) {
            return self.ownedWith(scalar, Array(T), Array(T).hypotScalar);
        }

        pub fn atan2Scalar(self: Self, scalar: T) ArrayError!Array(T) {
            return self.ownedWith(scalar, Array(T), Array(T).atan2Scalar);
        }

        pub fn arctan2Scalar(self: Self, scalar: T) ArrayError!Array(T) {
            return self.atan2Scalar(scalar);
        }

        pub fn nextAfterScalar(self: Self, scalar: T) ArrayError!Array(T) {
            return self.ownedWith(scalar, Array(T), Array(T).nextAfterScalar);
        }

        pub fn nextafterScalar(self: Self, scalar: T) ArrayError!Array(T) {
            return self.nextAfterScalar(scalar);
        }

        pub fn copysignScalar(self: Self, scalar: T) ArrayError!Array(T) {
            return self.ownedWith(scalar, Array(T), Array(T).copysignScalar);
        }

        pub fn heavisideScalar(self: Self, value_at_zero: T) ArrayError!Array(T) {
            return self.ownedWith(value_at_zero, Array(T), Array(T).heavisideScalar);
        }

        pub fn logAddExpScalar(self: Self, scalar: T) ArrayError!Array(T) {
            return self.ownedWith(scalar, Array(T), Array(T).logAddExpScalar);
        }

        pub fn logaddexpScalar(self: Self, scalar: T) ArrayError!Array(T) {
            return self.logAddExpScalar(scalar);
        }

        pub fn logAddExp2Scalar(self: Self, scalar: T) ArrayError!Array(T) {
            return self.ownedWith(scalar, Array(T), Array(T).logAddExp2Scalar);
        }

        pub fn logaddexp2Scalar(self: Self, scalar: T) ArrayError!Array(T) {
            return self.logAddExp2Scalar(scalar);
        }

        pub fn xlogyScalar(self: Self, scalar: T) ArrayError!Array(T) {
            return self.ownedWith(scalar, Array(T), Array(T).xlogyScalar);
        }

        pub fn ldexpScalar(self: Self, exponent: i32) ArrayError!Array(T) {
            return self.ownedWith(exponent, Array(T), Array(T).ldexpScalar);
        }

        pub fn eq(self: Self, other: Self) ArrayError!Array(bool) {
            return self.compareView(other, struct {
                fn f(a: T, b: T) bool {
                    return a == b;
                }
            }.f);
        }

        pub fn gt(self: Self, other: Self) ArrayError!Array(bool) {
            ensureNumeric(T);
            return self.compareView(other, struct {
                fn f(a: T, b: T) bool {
                    return lessValue(T, b, a);
                }
            }.f);
        }

        pub fn lt(self: Self, other: Self) ArrayError!Array(bool) {
            ensureNumeric(T);
            return self.compareView(other, struct {
                fn f(a: T, b: T) bool {
                    return lessValue(T, a, b);
                }
            }.f);
        }

        pub fn eqScalar(self: Self, scalar: T) ArrayError!Array(bool) {
            return self.compareScalar(scalar, struct {
                fn f(a: T, b: T) bool {
                    return a == b;
                }
            }.f);
        }

        pub fn equalScalar(self: Self, scalar: T) ArrayError!Array(bool) {
            return self.eqScalar(scalar);
        }

        pub fn gtScalar(self: Self, scalar: T) ArrayError!Array(bool) {
            ensureNumeric(T);
            return self.compareScalar(scalar, struct {
                fn f(a: T, b: T) bool {
                    return lessValue(T, b, a);
                }
            }.f);
        }

        pub fn greaterScalar(self: Self, scalar: T) ArrayError!Array(bool) {
            return self.gtScalar(scalar);
        }

        pub fn ltScalar(self: Self, scalar: T) ArrayError!Array(bool) {
            ensureNumeric(T);
            return self.compareScalar(scalar, struct {
                fn f(a: T, b: T) bool {
                    return lessValue(T, a, b);
                }
            }.f);
        }

        pub fn lessScalar(self: Self, scalar: T) ArrayError!Array(bool) {
            return self.ltScalar(scalar);
        }

        pub fn ne(self: Self, other: Self) ArrayError!Array(bool) {
            return self.compareView(other, struct {
                fn f(a: T, b: T) bool {
                    return a != b;
                }
            }.f);
        }

        pub fn notEqual(self: Self, other: Self) ArrayError!Array(bool) {
            return self.ne(other);
        }

        pub fn ge(self: Self, other: Self) ArrayError!Array(bool) {
            ensureNumeric(T);
            return self.compareView(other, struct {
                fn f(a: T, b: T) bool {
                    return !lessValue(T, a, b);
                }
            }.f);
        }

        pub fn greaterEqual(self: Self, other: Self) ArrayError!Array(bool) {
            return self.ge(other);
        }

        pub fn le(self: Self, other: Self) ArrayError!Array(bool) {
            ensureNumeric(T);
            return self.compareView(other, struct {
                fn f(a: T, b: T) bool {
                    return !lessValue(T, b, a);
                }
            }.f);
        }

        pub fn lessEqual(self: Self, other: Self) ArrayError!Array(bool) {
            return self.le(other);
        }

        pub fn equal(self: Self, other: Self) ArrayError!Array(bool) {
            return self.eq(other);
        }

        pub fn greater(self: Self, other: Self) ArrayError!Array(bool) {
            return self.gt(other);
        }

        pub fn less(self: Self, other: Self) ArrayError!Array(bool) {
            return self.lt(other);
        }

        pub fn neScalar(self: Self, scalar: T) ArrayError!Array(bool) {
            return self.compareScalar(scalar, struct {
                fn f(a: T, b: T) bool {
                    return a != b;
                }
            }.f);
        }

        pub fn notEqualScalar(self: Self, scalar: T) ArrayError!Array(bool) {
            return self.neScalar(scalar);
        }

        pub fn geScalar(self: Self, scalar: T) ArrayError!Array(bool) {
            ensureNumeric(T);
            return self.compareScalar(scalar, struct {
                fn f(a: T, b: T) bool {
                    return !lessValue(T, a, b);
                }
            }.f);
        }

        pub fn greaterEqualScalar(self: Self, scalar: T) ArrayError!Array(bool) {
            return self.geScalar(scalar);
        }

        pub fn leScalar(self: Self, scalar: T) ArrayError!Array(bool) {
            ensureNumeric(T);
            return self.compareScalar(scalar, struct {
                fn f(a: T, b: T) bool {
                    return !lessValue(T, b, a);
                }
            }.f);
        }

        pub fn lessEqualScalar(self: Self, scalar: T) ArrayError!Array(bool) {
            return self.leScalar(scalar);
        }

        pub fn square(self: Self) ArrayError!Array(T) {
            return self.ownedUnary(Array(T), Array(T).square);
        }

        pub fn reciprocal(self: Self) ArrayError!Array(T) {
            return self.ownedUnary(Array(T), Array(T).reciprocal);
        }

        pub fn sign(self: Self) ArrayError!Array(T) {
            return self.ownedUnary(Array(T), Array(T).sign);
        }

        pub fn signbit(self: Self) ArrayError!Array(bool) {
            return self.ownedUnary(Array(bool), Array(T).signbit);
        }

        pub fn exp(self: Self) ArrayError!Array(T) {
            return self.ownedUnary(Array(T), Array(T).exp);
        }

        pub fn exp2(self: Self) ArrayError!Array(T) {
            return self.ownedUnary(Array(T), Array(T).exp2);
        }

        pub fn expm1(self: Self) ArrayError!Array(T) {
            return self.ownedUnary(Array(T), Array(T).expm1);
        }

        pub fn log(self: Self) ArrayError!Array(T) {
            return self.ownedUnary(Array(T), Array(T).log);
        }

        pub fn log2(self: Self) ArrayError!Array(T) {
            return self.ownedUnary(Array(T), Array(T).log2);
        }

        pub fn log10(self: Self) ArrayError!Array(T) {
            return self.ownedUnary(Array(T), Array(T).log10);
        }

        pub fn log1p(self: Self) ArrayError!Array(T) {
            return self.ownedUnary(Array(T), Array(T).log1p);
        }

        pub fn lgamma(self: Self) ArrayError!Array(T) {
            return self.ownedUnary(Array(T), Array(T).lgamma);
        }

        pub fn gammaln(self: Self) ArrayError!Array(T) {
            return self.lgamma();
        }

        pub fn logGamma(self: Self) ArrayError!Array(T) {
            return self.lgamma();
        }

        pub fn loggamma(self: Self) ArrayError!Array(T) {
            return self.lgamma();
        }

        pub fn sqrt(self: Self) ArrayError!Array(T) {
            return self.ownedUnary(Array(T), Array(T).sqrt);
        }

        pub fn rsqrt(self: Self) ArrayError!Array(T) {
            return self.ownedUnary(Array(T), Array(T).rsqrt);
        }

        pub fn cbrt(self: Self) ArrayError!Array(T) {
            return self.ownedUnary(Array(T), Array(T).cbrt);
        }

        pub fn floor(self: Self) ArrayError!Array(T) {
            return self.ownedUnary(Array(T), Array(T).floor);
        }

        pub fn ceil(self: Self) ArrayError!Array(T) {
            return self.ownedUnary(Array(T), Array(T).ceil);
        }

        pub fn round(self: Self) ArrayError!Array(T) {
            return self.ownedUnary(Array(T), Array(T).round);
        }

        pub fn trunc(self: Self) ArrayError!Array(T) {
            return self.ownedUnary(Array(T), Array(T).trunc);
        }

        pub fn deg2rad(self: Self) ArrayError!Array(T) {
            return self.ownedUnary(Array(T), Array(T).deg2rad);
        }

        pub fn rad2deg(self: Self) ArrayError!Array(T) {
            return self.ownedUnary(Array(T), Array(T).rad2deg);
        }

        pub fn radians(self: Self) ArrayError!Array(T) {
            return self.deg2rad();
        }

        pub fn degrees(self: Self) ArrayError!Array(T) {
            return self.rad2deg();
        }

        pub fn sinc(self: Self) ArrayError!Array(T) {
            return self.ownedUnary(Array(T), Array(T).sinc);
        }

        pub fn sin(self: Self) ArrayError!Array(T) {
            return self.ownedUnary(Array(T), Array(T).sin);
        }

        pub fn cos(self: Self) ArrayError!Array(T) {
            return self.ownedUnary(Array(T), Array(T).cos);
        }

        pub fn tan(self: Self) ArrayError!Array(T) {
            return self.ownedUnary(Array(T), Array(T).tan);
        }

        pub fn asin(self: Self) ArrayError!Array(T) {
            return self.ownedUnary(Array(T), Array(T).asin);
        }

        pub fn arcsin(self: Self) ArrayError!Array(T) {
            return self.asin();
        }

        pub fn acos(self: Self) ArrayError!Array(T) {
            return self.ownedUnary(Array(T), Array(T).acos);
        }

        pub fn arccos(self: Self) ArrayError!Array(T) {
            return self.acos();
        }

        pub fn atan(self: Self) ArrayError!Array(T) {
            return self.ownedUnary(Array(T), Array(T).atan);
        }

        pub fn arctan(self: Self) ArrayError!Array(T) {
            return self.atan();
        }

        pub fn sinh(self: Self) ArrayError!Array(T) {
            return self.ownedUnary(Array(T), Array(T).sinh);
        }

        pub fn cosh(self: Self) ArrayError!Array(T) {
            return self.ownedUnary(Array(T), Array(T).cosh);
        }

        pub fn tanh(self: Self) ArrayError!Array(T) {
            return self.ownedUnary(Array(T), Array(T).tanh);
        }

        pub fn asinh(self: Self) ArrayError!Array(T) {
            return self.ownedUnary(Array(T), Array(T).asinh);
        }

        pub fn arcsinh(self: Self) ArrayError!Array(T) {
            return self.asinh();
        }

        pub fn acosh(self: Self) ArrayError!Array(T) {
            return self.ownedUnary(Array(T), Array(T).acosh);
        }

        pub fn arccosh(self: Self) ArrayError!Array(T) {
            return self.acosh();
        }

        pub fn atanh(self: Self) ArrayError!Array(T) {
            return self.ownedUnary(Array(T), Array(T).atanh);
        }

        pub fn arctanh(self: Self) ArrayError!Array(T) {
            return self.atanh();
        }

        pub fn relu(self: Self) ArrayError!Array(T) {
            return self.ownedUnary(Array(T), Array(T).relu);
        }

        pub fn leakyRelu(self: Self, negative_slope: T) ArrayError!Array(T) {
            return self.ownedWith(negative_slope, Array(T), Array(T).leakyRelu);
        }

        pub fn sigmoid(self: Self) ArrayError!Array(T) {
            return self.ownedUnary(Array(T), Array(T).sigmoid);
        }

        pub fn expit(self: Self) ArrayError!Array(T) {
            return self.ownedUnary(Array(T), Array(T).expit);
        }

        pub fn logit(self: Self) ArrayError!Array(T) {
            return self.ownedUnary(Array(T), Array(T).logit);
        }

        pub fn softplus(self: Self) ArrayError!Array(T) {
            return self.ownedUnary(Array(T), Array(T).softplus);
        }

        pub fn softsign(self: Self) ArrayError!Array(T) {
            return self.ownedUnary(Array(T), Array(T).softsign);
        }

        pub fn gelu(self: Self) ArrayError!Array(T) {
            return self.ownedUnary(Array(T), Array(T).gelu);
        }

        pub fn clip(self: Self, min_value: T, max_value: T) ArrayError!Array(T) {
            return self.ownedWith2(min_value, max_value, Array(T), Array(T).clip);
        }

        pub fn clamp(self: Self, min_value: T, max_value: T) ArrayError!Array(T) {
            return self.clip(min_value, max_value);
        }

        pub fn isNan(self: Self) ArrayError!Array(bool) {
            return self.ownedUnary(Array(bool), Array(T).isNan);
        }

        pub fn isnan(self: Self) ArrayError!Array(bool) {
            return self.isNan();
        }

        pub fn isInf(self: Self) ArrayError!Array(bool) {
            return self.ownedUnary(Array(bool), Array(T).isInf);
        }

        pub fn isinf(self: Self) ArrayError!Array(bool) {
            return self.isInf();
        }

        pub fn isPosInf(self: Self) ArrayError!Array(bool) {
            return self.ownedUnary(Array(bool), Array(T).isPosInf);
        }

        pub fn isposinf(self: Self) ArrayError!Array(bool) {
            return self.isPosInf();
        }

        pub fn isNegInf(self: Self) ArrayError!Array(bool) {
            return self.ownedUnary(Array(bool), Array(T).isNegInf);
        }

        pub fn isneginf(self: Self) ArrayError!Array(bool) {
            return self.isNegInf();
        }

        pub fn isFinite(self: Self) ArrayError!Array(bool) {
            return self.ownedUnary(Array(bool), Array(T).isFinite);
        }

        pub fn isfinite(self: Self) ArrayError!Array(bool) {
            return self.isFinite();
        }

        pub fn isNormal(self: Self) ArrayError!Array(bool) {
            return self.ownedUnary(Array(bool), Array(T).isNormal);
        }

        pub fn isnormal(self: Self) ArrayError!Array(bool) {
            return self.isNormal();
        }

        pub fn isReal(self: Self) ArrayError!Array(bool) {
            return self.ownedUnary(Array(bool), Array(T).isReal);
        }

        pub fn isreal(self: Self) ArrayError!Array(bool) {
            return self.isReal();
        }

        pub fn iscomplex(self: Self) ArrayError!Array(bool) {
            return self.ownedUnary(Array(bool), Array(T).iscomplex);
        }

        pub fn logicalNot(self: Self) ArrayError!Array(T) {
            return self.ownedUnary(Array(T), Array(T).logicalNot);
        }

        pub fn logicalAnd(self: Self, other: Self) ArrayError!Array(T) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            var rhs = try other.toArray();
            defer rhs.deinit();
            return lhs.logicalAnd(rhs);
        }

        pub fn logicalAndArray(self: Self, other: Array(T)) ArrayError!Array(T) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            return lhs.logicalAnd(other);
        }

        pub fn logicalAndScalar(self: Self, scalar: bool) ArrayError!Array(T) {
            return self.ownedWith(scalar, Array(T), Array(T).logicalAndScalar);
        }

        pub fn logicalOr(self: Self, other: Self) ArrayError!Array(T) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            var rhs = try other.toArray();
            defer rhs.deinit();
            return lhs.logicalOr(rhs);
        }

        pub fn logicalOrArray(self: Self, other: Array(T)) ArrayError!Array(T) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            return lhs.logicalOr(other);
        }

        pub fn logicalOrScalar(self: Self, scalar: bool) ArrayError!Array(T) {
            return self.ownedWith(scalar, Array(T), Array(T).logicalOrScalar);
        }

        pub fn logicalXor(self: Self, other: Self) ArrayError!Array(T) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            var rhs = try other.toArray();
            defer rhs.deinit();
            return lhs.logicalXor(rhs);
        }

        pub fn logicalXorArray(self: Self, other: Array(T)) ArrayError!Array(T) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            return lhs.logicalXor(other);
        }

        pub fn logicalXorScalar(self: Self, scalar: bool) ArrayError!Array(T) {
            return self.ownedWith(scalar, Array(T), Array(T).logicalXorScalar);
        }

        pub fn isclose(self: Self, other: Self, rtol: T, atol: T) ArrayError!Array(bool) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            var rhs = try other.toArray();
            defer rhs.deinit();
            return lhs.isclose(rhs, rtol, atol);
        }

        pub fn allclose(self: Self, other: Self, rtol: T, atol: T) ArrayError!bool {
            var lhs = try self.toArray();
            defer lhs.deinit();
            var rhs = try other.toArray();
            defer rhs.deinit();
            return lhs.allclose(rhs, rtol, atol);
        }

        pub fn iscloseScalar(self: Self, scalar: T, rtol: T, atol: T) ArrayError!Array(bool) {
            return self.ownedWith3(scalar, rtol, atol, Array(bool), Array(T).iscloseScalar);
        }

        pub fn allcloseScalar(self: Self, scalar: T, rtol: T, atol: T) ArrayError!bool {
            return self.ownedWith3(scalar, rtol, atol, bool, Array(T).allcloseScalar);
        }

        fn reducedShape(self: Self, axis: usize, keepdims: bool) ArrayError![]usize {
            var out_shape = try self.allocator.alloc(usize, if (keepdims) self.shape.len else self.shape.len - 1);
            if (keepdims) {
                @memcpy(out_shape, self.shape);
                out_shape[axis] = 1;
            } else {
                for (self.shape[0..axis], 0..) |extent, i| out_shape[i] = extent;
                for (self.shape[axis + 1 ..], axis..) |extent, i| out_shape[i] = extent;
            }
            return out_shape;
        }

        fn mapReducedToInput(axis: usize, keepdims: bool, out_multi: []const usize, in_multi: []usize) void {
            if (keepdims) {
                @memcpy(in_multi, out_multi);
            } else {
                for (out_multi[0..axis], 0..) |coord, i| in_multi[i] = coord;
                for (out_multi[axis..], axis + 1..) |coord, i| in_multi[i] = coord;
            }
        }

        fn keepDimsAllOnes(allocator: std.mem.Allocator, rank_count: usize) ArrayError![]usize {
            const dims = try allocator.alloc(usize, rank_count);
            @memset(dims, 1);
            return dims;
        }

        fn reduce(self: Self, axis_opt: ?isize, keepdims: bool, init_value: T, comptime op: fn (T, T) T) ArrayError!Array(T) {
            if (axis_opt == null) {
                var total = init_value;
                const multi = try self.allocator.alloc(usize, self.shape.len);
                defer self.allocator.free(multi);
                for (0..self.numel()) |flat| {
                    unravelIndexInto(flat, self.shape, multi);
                    total = op(total, self.data[self.offset + ravelIndex(multi, self.strides)]);
                }
                if (keepdims) {
                    const out_shape = try keepDimsAllOnes(self.allocator, self.shape.len);
                    defer self.allocator.free(out_shape);
                    return Array(T).fromSlice(self.allocator, &.{total}, out_shape);
                }
                return Array(T).fromSlice(self.allocator, &.{total}, &.{});
            }

            const axis = try normalizeDim(axis_opt.?, self.shape.len);
            const out_shape = try self.reducedShape(axis, keepdims);
            defer self.allocator.free(out_shape);
            var out = try Array(T).full(self.allocator, out_shape, init_value);
            errdefer out.deinit();
            if (out.data.len == 0) return out;
            var in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);
            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            for (0..self.numel()) |flat| {
                unravelIndexInto(flat, self.shape, in_multi);
                if (keepdims) {
                    @memcpy(out_multi, in_multi);
                    out_multi[axis] = 0;
                } else {
                    for (in_multi[0..axis], 0..) |coord, i| out_multi[i] = coord;
                    for (in_multi[axis + 1 ..], axis..) |coord, i| out_multi[i] = coord;
                }
                const out_index = ravelIndex(out_multi, out.strides);
                out.data[out_index] = op(out.data[out_index], self.data[self.offset + ravelIndex(in_multi, self.strides)]);
            }
            return out;
        }

        fn reduceFirst(self: Self, axis_opt: ?isize, keepdims: bool, comptime op: fn (T, T) T) ArrayError!Array(T) {
            if (self.numel() == 0) return error.EmptyArray;
            if (axis_opt == null) {
                const multi = try self.allocator.alloc(usize, self.shape.len);
                defer self.allocator.free(multi);
                unravelIndexInto(0, self.shape, multi);
                var total = self.data[self.offset + ravelIndex(multi, self.strides)];
                for (1..self.numel()) |flat| {
                    unravelIndexInto(flat, self.shape, multi);
                    total = op(total, self.data[self.offset + ravelIndex(multi, self.strides)]);
                }
                if (keepdims) {
                    const out_shape = try keepDimsAllOnes(self.allocator, self.shape.len);
                    defer self.allocator.free(out_shape);
                    return Array(T).fromSlice(self.allocator, &.{total}, out_shape);
                }
                return Array(T).fromSlice(self.allocator, &.{total}, &.{});
            }

            const axis = try normalizeDim(axis_opt.?, self.shape.len);
            if (self.shape[axis] == 0) return error.EmptyArray;
            const out_shape = try self.reducedShape(axis, keepdims);
            defer self.allocator.free(out_shape);
            var out = try Array(T).empty(self.allocator, out_shape);
            errdefer out.deinit();
            if (out.data.len == 0) return out;
            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            var in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);
            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, out_shape, out_multi);
                mapReducedToInput(axis, keepdims, out_multi, in_multi);
                in_multi[axis] = 0;
                var acc = self.data[self.offset + ravelIndex(in_multi, self.strides)];
                for (1..self.shape[axis]) |axis_i| {
                    in_multi[axis] = axis_i;
                    acc = op(acc, self.data[self.offset + ravelIndex(in_multi, self.strides)]);
                }
                slot.* = acc;
            }
            return out;
        }

        pub fn sum(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Array(T) {
            ensureNumeric(T);
            return self.reduce(axis_opt, keepdims, zero(T), opAdd);
        }

        pub fn prod(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Array(T) {
            ensureNumeric(T);
            return self.reduce(axis_opt, keepdims, one(T), opMul);
        }

        pub fn min(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Array(T) {
            ensureNumeric(T);
            return self.reduceFirst(axis_opt, keepdims, struct {
                fn f(a: T, b: T) T {
                    return if (lessValue(T, b, a)) b else a;
                }
            }.f);
        }

        pub fn amin(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Array(T) {
            return self.min(axis_opt, keepdims);
        }

        pub fn max(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Array(T) {
            ensureNumeric(T);
            return self.reduceFirst(axis_opt, keepdims, struct {
                fn f(a: T, b: T) T {
                    return if (lessValue(T, a, b)) b else a;
                }
            }.f);
        }

        pub fn amax(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Array(T) {
            return self.max(axis_opt, keepdims);
        }

        pub fn ptp(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Array(T) {
            var max_values = try self.max(axis_opt, keepdims);
            defer max_values.deinit();
            var min_values = try self.min(axis_opt, keepdims);
            defer min_values.deinit();
            return max_values.sub(min_values);
        }

        pub fn mean(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Array(T) {
            ensureFloat(T);
            const divisor: T = if (axis_opt) |axis_index| blk: {
                const axis = try normalizeDim(axis_index, self.shape.len);
                if (self.shape[axis] == 0) return error.EmptyArray;
                break :blk castValue(T, self.shape[axis]);
            } else blk: {
                if (self.numel() == 0) return error.EmptyArray;
                break :blk castValue(T, self.numel());
            };
            const out = try self.sum(axis_opt, keepdims);
            for (out.data) |*value| value.* /= divisor;
            return out;
        }

        pub fn variance(self: Self, axis_opt: ?isize, keepdims: bool, correction: T) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.variance(axis_opt, keepdims, correction);
        }

        pub fn stddev(self: Self, axis_opt: ?isize, keepdims: bool, correction: T) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.stddev(axis_opt, keepdims, correction);
        }

        pub fn median(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.median(axis_opt, keepdims);
        }

        pub fn quantile(self: Self, q: T, axis_opt: ?isize, keepdims: bool) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.quantile(q, axis_opt, keepdims);
        }

        pub fn percentile(self: Self, p: T, axis_opt: ?isize, keepdims: bool) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.percentile(p, axis_opt, keepdims);
        }

        pub fn average(self: Self, weights: ?Array(T), axis_opt: ?isize, keepdims: bool) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.average(weights, axis_opt, keepdims);
        }

        pub fn weightedMean(self: Self, weights: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            var owned_weights = try weights.toArray();
            defer owned_weights.deinit();
            return owned.weightedMean(owned_weights, axis_opt, keepdims);
        }

        pub fn weightedMeanArray(self: Self, weights: Array(T), axis_opt: ?isize, keepdims: bool) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.weightedMean(weights, axis_opt, keepdims);
        }

        pub fn weightedVariance(self: Self, weights: Self, axis_opt: ?isize, keepdims: bool, correction: T) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            var owned_weights = try weights.toArray();
            defer owned_weights.deinit();
            return owned.weightedVariance(owned_weights, axis_opt, keepdims, correction);
        }

        pub fn weightedVarianceArray(self: Self, weights: Array(T), axis_opt: ?isize, keepdims: bool, correction: T) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.weightedVariance(weights, axis_opt, keepdims, correction);
        }

        pub fn weightedVar(self: Self, weights: Self, axis_opt: ?isize, keepdims: bool, correction: T) ArrayError!Array(T) {
            return self.weightedVariance(weights, axis_opt, keepdims, correction);
        }

        pub fn weightedVarArray(self: Self, weights: Array(T), axis_opt: ?isize, keepdims: bool, correction: T) ArrayError!Array(T) {
            return self.weightedVarianceArray(weights, axis_opt, keepdims, correction);
        }

        pub fn weightedStddev(self: Self, weights: Self, axis_opt: ?isize, keepdims: bool, correction: T) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            var owned_weights = try weights.toArray();
            defer owned_weights.deinit();
            return owned.weightedStddev(owned_weights, axis_opt, keepdims, correction);
        }

        pub fn weightedStddevArray(self: Self, weights: Array(T), axis_opt: ?isize, keepdims: bool, correction: T) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.weightedStddev(weights, axis_opt, keepdims, correction);
        }

        pub fn weightedStd(self: Self, weights: Self, axis_opt: ?isize, keepdims: bool, correction: T) ArrayError!Array(T) {
            return self.weightedStddev(weights, axis_opt, keepdims, correction);
        }

        pub fn weightedStdArray(self: Self, weights: Array(T), axis_opt: ?isize, keepdims: bool, correction: T) ArrayError!Array(T) {
            return self.weightedStddevArray(weights, axis_opt, keepdims, correction);
        }

        pub fn weightedQuantile(self: Self, weights: Self, q: T, axis_opt: ?isize, keepdims: bool) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            var owned_weights = try weights.toArray();
            defer owned_weights.deinit();
            return owned.weightedQuantile(owned_weights, q, axis_opt, keepdims);
        }

        pub fn weightedQuantileArray(self: Self, weights: Array(T), q: T, axis_opt: ?isize, keepdims: bool) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.weightedQuantile(weights, q, axis_opt, keepdims);
        }

        pub fn weightedMedian(self: Self, weights: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            var owned_weights = try weights.toArray();
            defer owned_weights.deinit();
            return owned.weightedMedian(owned_weights, axis_opt, keepdims);
        }

        pub fn weightedMedianArray(self: Self, weights: Array(T), axis_opt: ?isize, keepdims: bool) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.weightedMedian(weights, axis_opt, keepdims);
        }

        pub fn nansum(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.nansum(axis_opt, keepdims);
        }

        pub fn nanmean(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.nanmean(axis_opt, keepdims);
        }

        pub fn nanvar(self: Self, axis_opt: ?isize, keepdims: bool, correction: T) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.nanvar(axis_opt, keepdims, correction);
        }

        pub fn nanstd(self: Self, axis_opt: ?isize, keepdims: bool, correction: T) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.nanstd(axis_opt, keepdims, correction);
        }

        pub fn nanmin(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.nanmin(axis_opt, keepdims);
        }

        pub fn nanmax(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.nanmax(axis_opt, keepdims);
        }

        pub fn nanmedian(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.nanmedian(axis_opt, keepdims);
        }

        pub fn nanquantile(self: Self, q: T, axis_opt: ?isize, keepdims: bool) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.nanquantile(q, axis_opt, keepdims);
        }

        pub fn nanpercentile(self: Self, p: T, axis_opt: ?isize, keepdims: bool) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.nanpercentile(p, axis_opt, keepdims);
        }

        pub fn nanToNum(self: Self, nan_value: T, posinf_value: T, neginf_value: T) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.nanToNum(nan_value, posinf_value, neginf_value);
        }

        pub fn nan_to_num(self: Self, nan_value: T, posinf_value: T, neginf_value: T) ArrayError!Array(T) {
            return self.nanToNum(nan_value, posinf_value, neginf_value);
        }

        pub fn nanToNumDefault(self: Self) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.nanToNumDefault();
        }

        pub fn nan_to_num_default(self: Self) ArrayError!Array(T) {
            return self.nanToNumDefault();
        }

        pub fn cumsum(self: Self) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.cumsum();
        }

        pub fn cumprod(self: Self) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.cumprod();
        }

        pub fn cummax(self: Self) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.cummax();
        }

        pub fn cummin(self: Self) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.cummin();
        }

        pub fn cumsumAxis(self: Self, axis_index: isize) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.cumsumAxis(axis_index);
        }

        pub fn cumprodAxis(self: Self, axis_index: isize) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.cumprodAxis(axis_index);
        }

        pub fn cummaxAxis(self: Self, axis_index: isize) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.cummaxAxis(axis_index);
        }

        pub fn cumminAxis(self: Self, axis_index: isize) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.cumminAxis(axis_index);
        }

        pub fn logcumsumexp(self: Self) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.logcumsumexp();
        }

        pub fn logcumsumexpAxis(self: Self, axis_index: isize) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.logcumsumexpAxis(axis_index);
        }

        pub fn diff(self: Self, axis_index: isize, n: usize) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.diff(axis_index, n);
        }

        pub fn trapezoid(self: Self, x_values: ?Array(T), dx: T, axis_index: isize) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.trapezoid(x_values, dx, axis_index);
        }

        pub fn trapz(self: Self, x_values: ?Array(T), dx: T, axis_index: isize) ArrayError!Array(T) {
            return self.trapezoid(x_values, dx, axis_index);
        }

        pub fn gradient(self: Self, x_values: ?Array(T), dx: T, axis_index: isize) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.gradient(x_values, dx, axis_index);
        }

        pub fn argmax(self: Self) ArrayError!usize {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.argmax();
        }

        pub fn argmin(self: Self) ArrayError!usize {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.argmin();
        }

        pub fn argmaxAxis(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Array(usize) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.argmaxAxis(axis_opt, keepdims);
        }

        pub fn argminAxis(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Array(usize) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.argminAxis(axis_opt, keepdims);
        }

        pub fn nanargmax(self: Self) ArrayError!usize {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.nanargmax();
        }

        pub fn nanargmin(self: Self) ArrayError!usize {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.nanargmin();
        }

        pub fn nanargmaxAxis(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Array(usize) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.nanargmaxAxis(axis_opt, keepdims);
        }

        pub fn nanargminAxis(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Array(usize) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.nanargminAxis(axis_opt, keepdims);
        }

        pub fn materializedApply(self: Self, comptime U: type, comptime method: fn (Array(T)) ArrayError!Array(U)) ArrayError!Array(U) {
            var owned = try self.toArray();
            defer owned.deinit();
            return method(owned);
        }

        pub fn softmax(self: Self, axis_index: isize) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.softmax(axis_index);
        }

        pub fn logSoftmax(self: Self, axis_index: isize) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.logSoftmax(axis_index);
        }

        pub fn norm(self: Self, p: T, axis_opt: ?isize, keepdims: bool) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.norm(p, axis_opt, keepdims);
        }

        pub fn logsumexp(self: Self, axis_index: isize, keepdims: bool) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.logsumexp(axis_index, keepdims);
        }

        pub fn log_softmax(self: Self, axis_index: isize) ArrayError!Array(T) {
            return self.logSoftmax(axis_index);
        }

        pub fn cov(self: Self, rowvar: bool, correction: T) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.cov(rowvar, correction);
        }

        pub fn corrcoef(self: Self, rowvar: bool) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.corrcoef(rowvar);
        }

        pub fn weightedCov(self: Self, weights: Array(T), rowvar: bool, correction: T) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.weightedCov(weights, rowvar, correction);
        }

        pub fn weightedCorrcoef(self: Self, weights: Array(T), rowvar: bool) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.weightedCorrcoef(weights, rowvar);
        }

        pub fn nanCov(self: Self, rowvar: bool, correction: T) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.nanCov(rowvar, correction);
        }

        pub fn nanCorrcoef(self: Self, rowvar: bool) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.nanCorrcoef(rowvar);
        }

        pub fn sort(self: Self, axis_opt: ?isize) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.sort(axis_opt);
        }

        pub fn sortBy(self: Self, axis_opt: ?isize, descending: bool) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.sortBy(axis_opt, descending);
        }

        pub fn sortDescending(self: Self, axis_opt: ?isize) ArrayError!Array(T) {
            return self.sortBy(axis_opt, true);
        }

        pub fn argsort(self: Self) ArrayError!Array(usize) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.argsort();
        }

        pub fn argsortAxis(self: Self, axis_opt: ?isize, descending: bool) ArrayError!Array(usize) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.argsortAxis(axis_opt, descending);
        }

        pub fn argsortDescending(self: Self) ArrayError!Array(usize) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.argsortDescending();
        }

        pub fn sortWithIndices(self: Self, axis_opt: ?isize, descending: bool) ArrayError!Array(T).SortResult {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.sortWithIndices(axis_opt, descending);
        }

        pub fn partition(self: Self, kth: usize, axis_opt: ?isize, descending: bool) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.partition(kth, axis_opt, descending);
        }

        pub fn argpartition(self: Self, kth: usize, axis_opt: ?isize, descending: bool) ArrayError!Array(usize) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.argpartition(kth, axis_opt, descending);
        }

        pub fn topk(self: Self, k: usize, axis_opt: ?isize, largest: bool, sorted: bool) ArrayError!Array(T).TopK {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.topk(k, axis_opt, largest, sorted);
        }

        pub fn take(self: Self, indices: Array(usize), axis_opt: ?isize) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.take(indices, axis_opt);
        }

        pub fn takeSigned(self: Self, indices: Array(isize), axis_opt: ?isize) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.takeSigned(indices, axis_opt);
        }

        pub fn takeMode(self: Self, indices: Array(usize), axis_opt: ?isize, mode: IndexMode) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.takeMode(indices, axis_opt, mode);
        }

        pub fn indexSelect(self: Self, axis_index: isize, indices: Array(usize)) ArrayError!Array(T) {
            return self.take(indices, axis_index);
        }

        pub fn indexSelectSigned(self: Self, axis_index: isize, indices: Array(isize)) ArrayError!Array(T) {
            return self.takeSigned(indices, axis_index);
        }

        pub fn gather(self: Self, axis_index: isize, indices: Array(usize)) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.gather(axis_index, indices);
        }

        pub fn gatherSigned(self: Self, axis_index: isize, indices: Array(isize)) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.gatherSigned(axis_index, indices);
        }

        pub fn takeAlongAxis(self: Self, indices: Array(usize), axis_index: isize) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.takeAlongAxis(indices, axis_index);
        }

        pub fn takeAlongAxisSigned(self: Self, indices: Array(isize), axis_index: isize) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.takeAlongAxisSigned(indices, axis_index);
        }

        pub fn putAlongAxis(self: Self, indices: Array(usize), src: Array(T), axis_index: isize) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.putAlongAxis(indices, src, axis_index);
        }

        pub fn maskedSelect(self: Self, mask: Array(bool)) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.maskedSelect(mask);
        }

        pub fn maskedScatter(self: Self, mask: Array(bool), src: Array(T)) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.maskedScatter(mask, src);
        }

        pub fn maskedPut(self: Self, mask: Array(bool), values: Array(T)) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.maskedPut(mask, values);
        }

        pub fn putMask(self: Self, mask: Array(bool), values: Array(T)) ArrayError!Array(T) {
            return self.maskedPut(mask, values);
        }

        pub fn maskedPutScalar(self: Self, mask: Array(bool), value: T) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.maskedPutScalar(mask, value);
        }

        pub fn putMaskScalar(self: Self, mask: Array(bool), value: T) ArrayError!Array(T) {
            return self.maskedPutScalar(mask, value);
        }

        pub fn copyWhere(self: Self, mask: Array(bool), src: Array(T)) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.copyWhere(mask, src);
        }

        pub fn compress(self: Self, condition: Array(bool), axis_opt: ?isize) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.compress(condition, axis_opt);
        }

        pub fn where(self: Self, mask: Array(bool), other: Self) ArrayError!Array(T) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            var rhs = try other.toArray();
            defer rhs.deinit();
            return lhs.where(mask, rhs);
        }

        pub fn whereArray(self: Self, mask: Array(bool), other: Array(T)) ArrayError!Array(T) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            return lhs.where(mask, other);
        }

        pub fn whereScalar(self: Self, mask: Array(bool), other_value: T) ArrayError!Array(T) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            return lhs.whereScalar(mask, other_value);
        }

        pub fn repeat(self: Self, repeats: usize, axis_index: isize) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.repeat(repeats, axis_index);
        }

        pub fn repeatInterleave(self: Self, repeats: Array(usize), axis_opt: ?isize) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.repeatInterleave(repeats, axis_opt);
        }

        pub fn repeatInterleaveScalar(self: Self, repeat_count: usize, axis_opt: ?isize) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.repeatInterleaveScalar(repeat_count, axis_opt);
        }

        pub fn tile(self: Self, repeats: []const usize) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.tile(repeats);
        }

        pub fn flip(self: Self, axis_index: isize) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.flip(axis_index);
        }

        pub fn flipAxes(self: Self, axes: []const isize) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.flipAxes(axes);
        }

        pub fn roll(self: Self, shift: isize, axis_index: isize) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.roll(shift, axis_index);
        }

        pub fn rollAxes(self: Self, shifts: []const isize, axes: []const isize) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.rollAxes(shifts, axes);
        }

        pub fn rot90(self: Self, k: isize, axes: [2]isize) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.rot90(k, axes);
        }

        pub fn padConstant(self: Self, before: []const usize, after: []const usize, value: T) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.padConstant(before, after, value);
        }

        pub fn padEdge(self: Self, before: []const usize, after: []const usize) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.padEdge(before, after);
        }

        pub fn padReflect(self: Self, before: []const usize, after: []const usize) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.padReflect(before, after);
        }

        pub fn padWrap(self: Self, before: []const usize, after: []const usize) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.padWrap(before, after);
        }

        pub fn padSymmetric(self: Self, before: []const usize, after: []const usize) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.padSymmetric(before, after);
        }

        pub fn slice1d(self: Self, slice_value: Slice) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.slice1d(slice_value);
        }

        pub fn split(self: Self, split_size: usize, axis_index: isize) ArrayError!Array(T).SplitResult {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.split(split_size, axis_index);
        }

        pub fn splitWithSizes(self: Self, sizes: []const usize, axis_index: isize) ArrayError!Array(T).SplitResult {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.splitWithSizes(sizes, axis_index);
        }

        pub fn split_with_sizes(self: Self, sizes: []const usize, axis_index: isize) ArrayError!Array(T).SplitResult {
            return self.splitWithSizes(sizes, axis_index);
        }

        pub fn splitAtIndices(self: Self, indices: []const usize, axis_index: isize) ArrayError!Array(T).SplitResult {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.splitAtIndices(indices, axis_index);
        }

        pub fn split_at_indices(self: Self, indices: []const usize, axis_index: isize) ArrayError!Array(T).SplitResult {
            return self.splitAtIndices(indices, axis_index);
        }

        pub fn chunk(self: Self, chunks: usize, axis_index: isize) ArrayError!Array(T).SplitResult {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.chunk(chunks, axis_index);
        }

        pub fn unbind(self: Self, axis_index: isize) ArrayError!Array(T).SplitResult {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.unbind(axis_index);
        }

        pub fn countNonzero(self: Self) usize {
            var count: usize = 0;
            const multi = self.allocator.alloc(usize, self.shape.len) catch return 0;
            defer self.allocator.free(multi);
            for (0..self.numel()) |flat| {
                unravelIndexInto(flat, self.shape, multi);
                if (self.data[self.offset + ravelIndex(multi, self.strides)] != zero(T)) count += 1;
            }
            return count;
        }

        pub fn count_nonzero(self: Self) usize {
            return self.countNonzero();
        }

        pub fn countNonzeroAxis(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Array(usize) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.countNonzeroAxis(axis_opt, keepdims);
        }

        pub fn all(self: Self) bool {
            var owned = self.toArray() catch return false;
            defer owned.deinit();
            return owned.all();
        }

        pub fn any(self: Self) bool {
            var owned = self.toArray() catch return false;
            defer owned.deinit();
            return owned.any();
        }

        pub fn allAxis(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.allAxis(axis_opt, keepdims);
        }

        pub fn anyAxis(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.anyAxis(axis_opt, keepdims);
        }

        pub fn flatNonzero(self: Self) ArrayError!Array(usize) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.flatNonzero();
        }

        pub fn nonzero(self: Self) ArrayError!Array(usize) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.nonzero();
        }

        pub fn argwhere(self: Self) ArrayError!Array(usize) {
            return self.nonzero();
        }

        pub fn whereIndices(self: Self) ArrayError!Array(usize) {
            return self.nonzero();
        }

        pub fn putFlat(self: Self, indices: Array(usize), values: Array(T)) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.putFlat(indices, values);
        }

        pub fn putFlatMode(self: Self, indices: Array(usize), values: Array(T), mode: IndexMode) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.putFlatMode(indices, values, mode);
        }

        pub fn putFlatScalar(self: Self, indices: Array(usize), value: T) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.putFlatScalar(indices, value);
        }

        pub fn putFlatScalarMode(self: Self, indices: Array(usize), value: T, mode: IndexMode) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.putFlatScalarMode(indices, value, mode);
        }

        pub fn putFlatSigned(self: Self, indices: Array(isize), values: Array(T)) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.putFlatSigned(indices, values);
        }

        pub fn putFlatScalarSigned(self: Self, indices: Array(isize), value: T) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.putFlatScalarSigned(indices, value);
        }

        pub fn indexPut(self: Self, indices: Array(usize), values: Array(T)) ArrayError!Array(T) {
            return self.putFlat(indices, values);
        }

        pub fn indexPutScalar(self: Self, indices: Array(usize), value: T) ArrayError!Array(T) {
            return self.putFlatScalar(indices, value);
        }

        pub fn ravelCoords(self: Self, coords: Array(usize)) ArrayError!Array(usize) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.ravelCoords(coords);
        }

        pub fn unravelFlat(self: Self, indices: Array(usize)) ArrayError!Array(usize) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.unravelFlat(indices);
        }

        pub fn takeCoords(self: Self, coords: Array(usize)) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.takeCoords(coords);
        }

        pub fn putCoords(self: Self, coords: Array(usize), values: Array(T)) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.putCoords(coords, values);
        }

        pub fn putCoordsScalar(self: Self, coords: Array(usize), value: T) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.putCoordsScalar(coords, value);
        }

        pub fn ravelMultiIndex(self: Self, indices: []const Array(usize)) ArrayError!Array(usize) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.ravelMultiIndex(indices);
        }

        pub fn takeMultiIndex(self: Self, indices: []const Array(usize)) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.takeMultiIndex(indices);
        }

        pub fn putMultiIndex(self: Self, indices: []const Array(usize), values: Array(T)) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.putMultiIndex(indices, values);
        }

        pub fn putMultiIndexScalar(self: Self, indices: []const Array(usize), value: T) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.putMultiIndexScalar(indices, value);
        }

        pub fn scatter(self: Self, axis_index: isize, indices: Array(usize), src: Array(T)) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.scatter(axis_index, indices, src);
        }

        pub fn scatterScalar(self: Self, axis_index: isize, indices: Array(usize), value: T) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.scatterScalar(axis_index, indices, value);
        }

        pub fn scatterReduce(self: Self, axis_index: isize, indices: Array(usize), src: Array(T), reduction: ScatterReduce) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.scatterReduce(axis_index, indices, src, reduction);
        }

        pub fn scatterAdd(self: Self, axis_index: isize, indices: Array(usize), src: Array(T)) ArrayError!Array(T) {
            return self.scatterReduce(axis_index, indices, src, .sum);
        }

        pub fn scatterReduceScalar(self: Self, axis_index: isize, indices: Array(usize), value: T, reduction: ScatterReduce) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.scatterReduceScalar(axis_index, indices, value, reduction);
        }

        pub fn scatterAddScalar(self: Self, axis_index: isize, indices: Array(usize), value: T) ArrayError!Array(T) {
            return self.scatterReduceScalar(axis_index, indices, value, .sum);
        }

        pub fn unique(self: Self) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.unique();
        }

        pub fn uniqueWithCounts(self: Self) ArrayError!Array(T).UniqueCounts {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.uniqueWithCounts();
        }

        pub fn union1d(self: Self, other: Self) ArrayError!Array(T) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            var rhs = try other.toArray();
            defer rhs.deinit();
            return lhs.union1d(rhs);
        }

        pub fn union1dArray(self: Self, other: Array(T)) ArrayError!Array(T) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            return lhs.union1d(other);
        }

        pub fn intersect1d(self: Self, other: Self) ArrayError!Array(T) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            var rhs = try other.toArray();
            defer rhs.deinit();
            return lhs.intersect1d(rhs);
        }

        pub fn intersect1dArray(self: Self, other: Array(T)) ArrayError!Array(T) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            return lhs.intersect1d(other);
        }

        pub fn setdiff1d(self: Self, other: Self) ArrayError!Array(T) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            var rhs = try other.toArray();
            defer rhs.deinit();
            return lhs.setdiff1d(rhs);
        }

        pub fn setdiff1dArray(self: Self, other: Array(T)) ArrayError!Array(T) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            return lhs.setdiff1d(other);
        }

        pub fn setxor1d(self: Self, other: Self) ArrayError!Array(T) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            var rhs = try other.toArray();
            defer rhs.deinit();
            return lhs.setxor1d(rhs);
        }

        pub fn setxor1dArray(self: Self, other: Array(T)) ArrayError!Array(T) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            return lhs.setxor1d(other);
        }

        pub fn bincount(self: Self, minlength: usize) ArrayError!Array(usize) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.bincount(minlength);
        }

        pub fn bincountWeighted(self: Self, comptime W: type, weights: Array(W), minlength: usize) ArrayError!Array(W) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.bincountWeighted(W, weights, minlength);
        }

        pub fn searchsorted(self: Self, values: Array(T), side: SearchSide) ArrayError!Array(usize) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.searchsorted(values, side);
        }

        pub fn bucketize(self: Self, boundaries: Array(T), side: SearchSide) ArrayError!Array(usize) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.bucketize(boundaries, side);
        }

        pub fn digitize(self: Self, bins: Array(T), right: bool) ArrayError!Array(usize) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.digitize(bins, right);
        }

        pub fn isin(self: Self, test_elements: Array(T), invert: bool) ArrayError!Array(bool) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.isin(test_elements, invert);
        }

        pub fn histogram(self: Self, bins: usize, range: ?Array(T).HistogramRange) ArrayError!Array(T).HistogramResult {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.histogram(bins, range);
        }

        pub fn matmul(self: Self, other: Self) ArrayError!Array(T) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            var rhs = try other.toArray();
            defer rhs.deinit();
            return lhs.matmul(rhs);
        }

        pub fn matmulArray(self: Self, other: Array(T)) ArrayError!Array(T) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            return lhs.matmul(other);
        }

        pub fn mm(self: Self, other: Self) ArrayError!Array(T) {
            return self.matmul(other);
        }

        pub fn bmm(self: Self, other: Self) ArrayError!Array(T) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            var rhs = try other.toArray();
            defer rhs.deinit();
            return lhs.bmm(rhs);
        }

        pub fn bmmArray(self: Self, other: Array(T)) ArrayError!Array(T) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            return lhs.bmm(other);
        }

        pub fn matvec(self: Self, vector: Self) ArrayError!Array(T) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            var rhs = try vector.toArray();
            defer rhs.deinit();
            return lhs.matvec(rhs);
        }

        pub fn matvecArray(self: Self, vector: Array(T)) ArrayError!Array(T) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            return lhs.matvec(vector);
        }

        pub fn dot(self: Self, other: Self) ArrayError!Array(T) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            var rhs = try other.toArray();
            defer rhs.deinit();
            return lhs.dot(rhs);
        }

        pub fn vdot(self: Self, other: Self) ArrayError!Array(T) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            var rhs = try other.toArray();
            defer rhs.deinit();
            return lhs.vdot(rhs);
        }

        pub fn vdotArray(self: Self, other: Array(T)) ArrayError!Array(T) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            return lhs.vdot(other);
        }

        pub fn vecdot(self: Self, other: Self, axis_index: isize) ArrayError!Array(T) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            var rhs = try other.toArray();
            defer rhs.deinit();
            return lhs.vecdot(rhs, axis_index);
        }

        pub fn vecdotArray(self: Self, other: Array(T), axis_index: isize) ArrayError!Array(T) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            return lhs.vecdot(other, axis_index);
        }

        pub fn inner(self: Self, other: Self) ArrayError!Array(T) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            var rhs = try other.toArray();
            defer rhs.deinit();
            return lhs.inner(rhs);
        }

        pub fn innerArray(self: Self, other: Array(T)) ArrayError!Array(T) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            return lhs.inner(other);
        }

        pub fn outer(self: Self, other: Self) ArrayError!Array(T) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            var rhs = try other.toArray();
            defer rhs.deinit();
            return lhs.outer(rhs);
        }

        pub fn cross(self: Self, other: Self, axis_index: isize) ArrayError!Array(T) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            var rhs = try other.toArray();
            defer rhs.deinit();
            return lhs.cross(rhs, axis_index);
        }

        pub fn crossArray(self: Self, other: Array(T), axis_index: isize) ArrayError!Array(T) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            return lhs.cross(other, axis_index);
        }

        pub fn contractAxes(self: Self, other: Self, axes_self: []const usize, axes_other: []const usize) ArrayError!Array(T) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            var rhs = try other.toArray();
            defer rhs.deinit();
            return lhs.contractAxes(rhs, axes_self, axes_other);
        }

        pub fn contractAxesArray(self: Self, other: Array(T), axes_self: []const usize, axes_other: []const usize) ArrayError!Array(T) {
            var lhs = try self.toArray();
            defer lhs.deinit();
            return lhs.contractAxes(other, axes_self, axes_other);
        }

        pub fn trace(self: Self) ArrayError!T {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.trace();
        }

        pub fn traceOffset(self: Self, offset: isize) ArrayError!T {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.traceOffset(offset);
        }

        pub fn diagonal(self: Self, offset: isize) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.diagonal(offset);
        }

        pub fn diag(self: Self, offset: isize) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.diag(offset);
        }

        pub fn diagflat(self: Self, offset: isize) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.diagflat(offset);
        }

        pub fn triu(self: Self, diagonal_offset: isize) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.triu(diagonal_offset);
        }

        pub fn tril(self: Self, diagonal_offset: isize) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.tril(diagonal_offset);
        }

        pub fn real(self: Self) ArrayError!Array(complexRealType(T)) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.real();
        }

        pub fn imag(self: Self) ArrayError!Array(complexRealType(T)) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.imag();
        }

        pub fn conjugate(self: Self) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.conjugate();
        }

        pub fn conj(self: Self) ArrayError!Array(T) {
            return self.conjugate();
        }

        pub fn magnitude(self: Self) ArrayError!Array(complexRealType(T)) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.magnitude();
        }

        pub fn absComplex(self: Self) ArrayError!Array(complexRealType(T)) {
            return self.magnitude();
        }

        pub fn angle(self: Self) ArrayError!Array(complexRealType(T)) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.angle();
        }

        pub fn phase(self: Self) ArrayError!Array(complexRealType(T)) {
            return self.angle();
        }

        pub fn fft(self: Self) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.fft();
        }

        pub fn ifft(self: Self) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.ifft();
        }

        pub fn rfft(self: Self) ArrayError!Array(complexTypeForReal(T)) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.rfft();
        }

        pub fn irfft(self: Self, output_len: ?usize) ArrayError!Array(realTypeForComplex(T)) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.irfft(output_len);
        }

        pub fn fftAxis(self: Self, axis_index: isize) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.fftAxis(axis_index);
        }

        pub fn ifftAxis(self: Self, axis_index: isize) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.ifftAxis(axis_index);
        }

        pub fn fftAxes(self: Self, axes: []const isize) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.fftAxes(axes);
        }

        pub fn ifftAxes(self: Self, axes: []const isize) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.ifftAxes(axes);
        }

        pub fn fft2(self: Self) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.fft2();
        }

        pub fn ifft2(self: Self) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.ifft2();
        }

        pub fn ldexp(self: Self, exponents: Array(i32)) ArrayError!Array(T) {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.ldexp(exponents);
        }

        pub fn frexp(self: Self) ArrayError!Array(T).FrexpResult {
            var owned = try self.toArray();
            defer owned.deinit();
            return owned.frexp();
        }

        pub fn convolve1d(self: Self, kernel: Self, mode: ConvMode) ArrayError!Array(T) {
            var signal_values = try self.toArray();
            defer signal_values.deinit();
            var kernel_values = try kernel.toArray();
            defer kernel_values.deinit();
            return signal_values.convolve1d(kernel_values, mode);
        }

        pub fn convolve1dArray(self: Self, kernel: Array(T), mode: ConvMode) ArrayError!Array(T) {
            var signal_values = try self.toArray();
            defer signal_values.deinit();
            return signal_values.convolve1d(kernel, mode);
        }

        pub fn correlate1d(self: Self, kernel: Self, mode: ConvMode) ArrayError!Array(T) {
            var signal_values = try self.toArray();
            defer signal_values.deinit();
            var kernel_values = try kernel.toArray();
            defer kernel_values.deinit();
            return signal_values.correlate1d(kernel_values, mode);
        }

        pub fn correlate1dArray(self: Self, kernel: Array(T), mode: ConvMode) ArrayError!Array(T) {
            var signal_values = try self.toArray();
            defer signal_values.deinit();
            return signal_values.correlate1d(kernel, mode);
        }

        pub fn convolve2d(self: Self, kernel: Self, mode: ConvMode) ArrayError!Array(T) {
            var image_values = try self.toArray();
            defer image_values.deinit();
            var kernel_values = try kernel.toArray();
            defer kernel_values.deinit();
            return image_values.convolve2d(kernel_values, mode);
        }

        pub fn convolve2dArray(self: Self, kernel: Array(T), mode: ConvMode) ArrayError!Array(T) {
            var image_values = try self.toArray();
            defer image_values.deinit();
            return image_values.convolve2d(kernel, mode);
        }

        pub fn correlate2d(self: Self, kernel: Self, mode: ConvMode) ArrayError!Array(T) {
            var image_values = try self.toArray();
            defer image_values.deinit();
            var kernel_values = try kernel.toArray();
            defer kernel_values.deinit();
            return image_values.correlate2d(kernel_values, mode);
        }

        pub fn correlate2dArray(self: Self, kernel: Array(T), mode: ConvMode) ArrayError!Array(T) {
            var image_values = try self.toArray();
            defer image_values.deinit();
            return image_values.correlate2d(kernel, mode);
        }

        pub fn reshape(self: Self, dims: []const usize) ArrayError!Self {
            if (!self.isContiguous()) return error.InvalidShape;
            const n = try numelFrom(dims);
            if (n != self.numel()) return error.ShapeMismatch;
            const strides = try stridesFor(self.allocator, dims);
            defer self.allocator.free(strides);
            return Self.init(self.allocator, self.data, dims, strides, self.offset, self.device);
        }

        pub fn reshapeInfer(self: Self, dims: []const isize) ArrayError!Self {
            const inferred = try inferredShape(self.allocator, dims, self.numel());
            defer self.allocator.free(inferred);
            return self.reshape(inferred);
        }

        pub fn reshapeAs(self: Self, other: Self) ArrayError!Self {
            return self.reshape(other.shape);
        }

        pub fn reshapeAsArray(self: Self, other: Array(T)) ArrayError!Self {
            return self.reshape(other.shape);
        }

        pub fn reshape_as(self: Self, other: Self) ArrayError!Self {
            return self.reshapeAs(other);
        }

        pub fn view(self: Self, dims: []const usize) ArrayError!Self {
            return self.reshape(dims);
        }

        pub fn viewInfer(self: Self, dims: []const isize) ArrayError!Self {
            return self.reshapeInfer(dims);
        }

        pub fn viewAs(self: Self, other: Self) ArrayError!Self {
            return self.view(other.shape);
        }

        pub fn viewAsArray(self: Self, other: Array(T)) ArrayError!Self {
            return self.view(other.shape);
        }

        pub fn view_as(self: Self, other: Self) ArrayError!Self {
            return self.viewAs(other);
        }

        pub fn flatten(self: Self) ArrayError!Self {
            return self.reshape(&.{self.numel()});
        }

        pub fn flattenAxes(self: Self, start_axis: isize, end_axis: isize) ArrayError!Self {
            const dims = try flattenShape(self.allocator, self.shape, start_axis, end_axis);
            defer self.allocator.free(dims);
            return self.reshape(dims);
        }

        pub fn flattenRange(self: Self, start_axis: isize, end_axis: isize) ArrayError!Self {
            return self.flattenAxes(start_axis, end_axis);
        }

        pub fn flatten_range(self: Self, start_axis: isize, end_axis: isize) ArrayError!Self {
            return self.flattenAxes(start_axis, end_axis);
        }

        pub fn flattenFrom(self: Self, start_axis: isize) ArrayError!Self {
            return self.flattenAxes(start_axis, -1);
        }

        pub fn flatten_from(self: Self, start_axis: isize) ArrayError!Self {
            return self.flattenFrom(start_axis);
        }

        pub fn ravel(self: Self) ArrayError!Self {
            return self.flatten();
        }

        pub fn atLeast1d(self: Self) ArrayError!Self {
            if (self.shape.len >= 1) return self.clone();
            return self.unsqueeze(0);
        }

        pub fn atLeast2d(self: Self) ArrayError!Self {
            return switch (self.shape.len) {
                0 => blk: {
                    var one_d = try self.unsqueeze(0);
                    defer one_d.deinit();
                    break :blk one_d.unsqueeze(0);
                },
                1 => self.unsqueeze(0),
                else => self.clone(),
            };
        }

        pub fn atLeast3d(self: Self) ArrayError!Self {
            return switch (self.shape.len) {
                0 => blk: {
                    var one_d = try self.unsqueeze(0);
                    defer one_d.deinit();
                    var two_d = try one_d.unsqueeze(0);
                    defer two_d.deinit();
                    break :blk two_d.unsqueeze(0);
                },
                1 => blk: {
                    var two_d = try self.unsqueeze(0);
                    defer two_d.deinit();
                    break :blk two_d.unsqueeze(2);
                },
                2 => self.unsqueeze(2),
                else => self.clone(),
            };
        }

        pub fn unflatten(self: Self, axis_index: isize, dims: []const usize) ArrayError!Self {
            const out_shape = try unflattenShape(self.allocator, self.shape, axis_index, dims);
            defer self.allocator.free(out_shape);
            return self.reshape(out_shape);
        }

        pub fn sliceAxis(self: Self, axis_index: isize, slice_value: Slice) ArrayError!Self {
            if (self.shape.len == 0) return error.InvalidAxis;
            const axis = try normalizeDim(axis_index, self.shape.len);
            const ns = try normalizeSlice(slice_value, self.shape[axis]);
            const shape = try self.allocator.dupe(usize, self.shape);
            errdefer self.allocator.free(shape);
            const strides = try self.allocator.dupe(usize, self.strides);
            shape[axis] = ns.count;
            strides[axis] *= ns.step;
            return .{
                .allocator = self.allocator,
                .data = self.data,
                .shape = shape,
                .strides = strides,
                .offset = self.offset + ns.start * self.strides[axis],
                .device = self.device,
            };
        }

        pub fn slice(self: Self, slices: []const Slice) ArrayError!Self {
            if (slices.len != self.shape.len) return error.ShapeMismatch;
            var current = try self.clone();
            errdefer current.deinit();
            for (slices, 0..) |slice_value, axis| {
                const next = try current.sliceAxis(@intCast(axis), slice_value);
                current.deinit();
                current = next;
            }
            return current;
        }

        pub fn narrow(self: Self, axis_index: isize, start: usize, length: usize) ArrayError!Self {
            const axis = try normalizeDim(axis_index, self.shape.len);
            if (start > self.shape[axis] or start + length > self.shape[axis]) return error.IndexOutOfBounds;
            return self.sliceAxis(axis_index, .{ .start = @intCast(start), .stop = @intCast(start + length), .step = 1 });
        }

        pub fn select(self: Self, axis_index: isize, index: usize) ArrayError!Self {
            if (self.shape.len == 0) return error.InvalidAxis;
            const axis = try normalizeDim(axis_index, self.shape.len);
            if (index >= self.shape[axis]) return error.IndexOutOfBounds;
            const shape = try self.allocator.alloc(usize, self.shape.len - 1);
            errdefer self.allocator.free(shape);
            const strides = try self.allocator.alloc(usize, self.strides.len - 1);
            for (self.shape[0..axis], 0..) |extent, i| shape[i] = extent;
            for (self.shape[axis + 1 ..], axis..) |extent, i| shape[i] = extent;
            for (self.strides[0..axis], 0..) |stride_value, i| strides[i] = stride_value;
            for (self.strides[axis + 1 ..], axis..) |stride_value, i| strides[i] = stride_value;
            return .{
                .allocator = self.allocator,
                .data = self.data,
                .shape = shape,
                .strides = strides,
                .offset = self.offset + index * self.strides[axis],
                .device = self.device,
            };
        }

        pub fn selectSigned(self: Self, axis_index: isize, index: isize) ArrayError!Self {
            const axis = try normalizeDim(axis_index, self.shape.len);
            return self.select(axis_index, try normalizeIndex(index, self.shape[axis]));
        }

        pub fn squeeze(self: Self, axis_opt: ?isize) ArrayError!Self {
            var shape_list: std.ArrayList(usize) = .empty;
            defer shape_list.deinit(self.allocator);
            var stride_list: std.ArrayList(usize) = .empty;
            defer stride_list.deinit(self.allocator);
            if (axis_opt) |axis_index| {
                const axis = try normalizeDim(axis_index, self.shape.len);
                for (self.shape, self.strides, 0..) |extent, stride_value, i| {
                    if (i == axis and extent == 1) continue;
                    try shape_list.append(self.allocator, extent);
                    try stride_list.append(self.allocator, stride_value);
                }
            } else {
                for (self.shape, self.strides) |extent, stride_value| {
                    if (extent == 1) continue;
                    try shape_list.append(self.allocator, extent);
                    try stride_list.append(self.allocator, stride_value);
                }
            }
            return Self.init(self.allocator, self.data, shape_list.items, stride_list.items, self.offset, self.device);
        }

        pub fn squeezeDim(self: Self, axis_index: isize) ArrayError!Self {
            return self.squeeze(axis_index);
        }

        pub fn squeeze_dim(self: Self, axis_index: isize) ArrayError!Self {
            return self.squeezeDim(axis_index);
        }

        pub fn squeezeAxes(self: Self, axes: []const isize) ArrayError!Self {
            if (axes.len == 0) return self.clone();
            const normalized_axes = try normalizeUniqueAxes(self.allocator, axes, self.shape.len);
            defer self.allocator.free(normalized_axes);
            var squeeze_mask = try self.allocator.alloc(bool, self.shape.len);
            defer self.allocator.free(squeeze_mask);
            @memset(squeeze_mask, false);
            for (normalized_axes) |axis| {
                if (self.shape[axis] != 1) return error.ShapeMismatch;
                squeeze_mask[axis] = true;
            }
            var shape_list: std.ArrayList(usize) = .empty;
            defer shape_list.deinit(self.allocator);
            var stride_list: std.ArrayList(usize) = .empty;
            defer stride_list.deinit(self.allocator);
            for (self.shape, self.strides, 0..) |extent, stride_value, axis| {
                if (squeeze_mask[axis]) continue;
                try shape_list.append(self.allocator, extent);
                try stride_list.append(self.allocator, stride_value);
            }
            return Self.init(self.allocator, self.data, shape_list.items, stride_list.items, self.offset, self.device);
        }

        pub fn squeeze_axes(self: Self, axes: []const isize) ArrayError!Self {
            return self.squeezeAxes(axes);
        }

        pub fn unsqueeze(self: Self, axis_index: isize) ArrayError!Self {
            const rank_count = self.shape.len + 1;
            const axis = if (axis_index < 0) blk: {
                const signed_rank: isize = @intCast(rank_count);
                const normalized = signed_rank + axis_index;
                if (normalized < 0 or normalized >= signed_rank) return error.InvalidAxis;
                break :blk @as(usize, @intCast(normalized));
            } else try canonicalAxis(@intCast(axis_index), rank_count);
            const shape = try self.allocator.alloc(usize, rank_count);
            errdefer self.allocator.free(shape);
            const strides = try self.allocator.alloc(usize, rank_count);
            for (self.shape[0..axis], 0..) |extent, i| shape[i] = extent;
            shape[axis] = 1;
            for (self.shape[axis..], axis + 1..) |extent, i| shape[i] = extent;
            for (self.strides[0..axis], 0..) |stride_value, i| strides[i] = stride_value;
            strides[axis] = 0;
            for (self.strides[axis..], axis + 1..) |stride_value, i| strides[i] = stride_value;
            return .{
                .allocator = self.allocator,
                .data = self.data,
                .shape = shape,
                .strides = strides,
                .offset = self.offset,
                .device = self.device,
            };
        }

        pub fn unsqueezeDim(self: Self, axis_index: isize) ArrayError!Self {
            return self.unsqueeze(axis_index);
        }

        pub fn unsqueeze_dim(self: Self, axis_index: isize) ArrayError!Self {
            return self.unsqueezeDim(axis_index);
        }

        pub fn broadcastTo(self: Self, dims: []const usize) ArrayError!Self {
            const out_shape = try broadcastShape(self.allocator, self.shape, dims);
            errdefer self.allocator.free(out_shape);
            if (!std.mem.eql(usize, out_shape, dims)) return error.ShapeMismatch;
            const out_strides = try self.allocator.alloc(usize, dims.len);
            const leading = dims.len - self.shape.len;
            for (out_strides, 0..) |*slot, axis| {
                if (axis < leading) {
                    slot.* = 0;
                } else {
                    const in_axis = axis - leading;
                    slot.* = if (self.shape[in_axis] == 1 and dims[axis] != 1) 0 else self.strides[in_axis];
                }
            }
            return .{
                .allocator = self.allocator,
                .data = self.data,
                .shape = out_shape,
                .strides = out_strides,
                .offset = self.offset,
                .device = self.device,
            };
        }

        pub fn expand(self: Self, dims: []const usize) ArrayError!Self {
            return self.broadcastTo(dims);
        }

        pub fn expandAs(self: Self, other: Self) ArrayError!Self {
            return self.expand(other.shape);
        }

        pub fn expandAsArray(self: Self, other: Array(T)) ArrayError!Self {
            return self.expand(other.shape);
        }

        pub fn expandAsView(self: Self, other: Self) ArrayError!Self {
            return self.expand(other.shape);
        }

        pub fn expand_as(self: Self, other: Self) ArrayError!Self {
            return self.expandAs(other);
        }

        pub fn permute(self: Self, axes: []const usize) ArrayError!Self {
            if (axes.len != self.shape.len) return error.InvalidPermutation;
            var seen = try self.allocator.alloc(bool, axes.len);
            defer self.allocator.free(seen);
            @memset(seen, false);
            const shape = try self.allocator.alloc(usize, axes.len);
            errdefer self.allocator.free(shape);
            const strides = try self.allocator.alloc(usize, axes.len);
            errdefer self.allocator.free(strides);
            for (axes, 0..) |axis, i| {
                if (axis >= axes.len or seen[axis]) return error.InvalidPermutation;
                seen[axis] = true;
                shape[i] = self.shape[axis];
                strides[i] = self.strides[axis];
            }
            return .{
                .allocator = self.allocator,
                .data = self.data,
                .shape = shape,
                .strides = strides,
                .offset = self.offset,
                .device = self.device,
            };
        }

        pub fn swapaxes(self: Self, dim0: isize, dim1: isize) ArrayError!Self {
            const a0 = try normalizeDim(dim0, self.shape.len);
            const a1 = try normalizeDim(dim1, self.shape.len);
            var axes = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(axes);
            for (axes, 0..) |*slot, i| slot.* = i;
            std.mem.swap(usize, &axes[a0], &axes[a1]);
            return self.permute(axes);
        }

        pub fn swapDims(self: Self, dim0: isize, dim1: isize) ArrayError!Self {
            return self.swapaxes(dim0, dim1);
        }

        pub fn swap_dims(self: Self, dim0: isize, dim1: isize) ArrayError!Self {
            return self.swapDims(dim0, dim1);
        }

        pub fn movedim(self: Self, source: isize, destination: isize) ArrayError!Self {
            const src = try normalizeDim(source, self.shape.len);
            const dst = try normalizeDim(destination, self.shape.len);
            const axes = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(axes);
            var remaining = try self.allocator.alloc(usize, self.shape.len - 1);
            defer self.allocator.free(remaining);
            var remaining_len: usize = 0;
            for (0..self.shape.len) |i| {
                if (i == src) continue;
                remaining[remaining_len] = i;
                remaining_len += 1;
            }
            var read: usize = 0;
            for (axes, 0..) |*slot, out_i| {
                if (out_i == dst) {
                    slot.* = src;
                } else {
                    slot.* = remaining[read];
                    read += 1;
                }
            }
            return self.permute(axes);
        }

        pub fn moveaxis(self: Self, source: isize, destination: isize) ArrayError!Self {
            return self.movedim(source, destination);
        }

        pub fn moveaxes(self: Self, sources: []const isize, destinations: []const isize) ArrayError!Self {
            const axes = try movedimManyAxes(self.allocator, self.shape.len, sources, destinations);
            defer self.allocator.free(axes);
            return self.permute(axes);
        }

        pub fn move_axes(self: Self, sources: []const isize, destinations: []const isize) ArrayError!Self {
            return self.moveaxes(sources, destinations);
        }

        pub fn transpose(self: Self) ArrayError!Self {
            if (self.shape.len != 2) return error.NonMatrixArray;
            return self.swapaxes(0, 1);
        }

        pub fn T_(self: Self) ArrayError!Self {
            return self.transpose();
        }
    };
}

pub fn Array(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        data: []T,
        shape: []usize,
        strides: []usize,
        device: Device = .cpu,

        pub const Scalar = T;
        pub const dtype = DType.of(T);

        pub fn init(allocator: std.mem.Allocator, dims: []const usize) ArrayError!Self {
            const n = try numelFrom(dims);
            const values = try allocator.alloc(T, n);
            @memset(values, zero(T));
            const shape = try allocator.dupe(usize, dims);
            errdefer allocator.free(shape);
            const strides = try stridesFor(allocator, shape);
            return .{ .allocator = allocator, .data = values, .shape = shape, .strides = strides };
        }

        pub fn full(allocator: std.mem.Allocator, dims: []const usize, value: T) ArrayError!Self {
            const out = try Self.init(allocator, dims);
            @memset(out.data, value);
            return out;
        }

        pub fn zeros(allocator: std.mem.Allocator, dims: []const usize) ArrayError!Self {
            return Self.full(allocator, dims, zero(T));
        }

        pub fn ones(allocator: std.mem.Allocator, dims: []const usize) ArrayError!Self {
            return Self.full(allocator, dims, one(T));
        }

        pub fn empty(allocator: std.mem.Allocator, dims: []const usize) ArrayError!Self {
            const n = try numelFrom(dims);
            const values = try allocator.alloc(T, n);
            const shape = try allocator.dupe(usize, dims);
            errdefer allocator.free(shape);
            const strides = try stridesFor(allocator, shape);
            return .{ .allocator = allocator, .data = values, .shape = shape, .strides = strides };
        }

        pub fn fromScalar(allocator: std.mem.Allocator, value: T) ArrayError!Self {
            return Self.fromSlice(allocator, &.{value}, &.{});
        }

        pub fn emptyLike(self: Self) ArrayError!Self {
            return Self.empty(self.allocator, self.shape);
        }

        pub fn zerosLike(self: Self) ArrayError!Self {
            return Self.zeros(self.allocator, self.shape);
        }

        pub fn onesLike(self: Self) ArrayError!Self {
            return Self.ones(self.allocator, self.shape);
        }

        pub fn fullLike(self: Self, value: T) ArrayError!Self {
            return Self.full(self.allocator, self.shape, value);
        }

        pub fn newEmpty(self: Self, dims: []const usize) ArrayError!Self {
            return Self.empty(self.allocator, dims);
        }

        pub fn new_empty(self: Self, dims: []const usize) ArrayError!Self {
            return self.newEmpty(dims);
        }

        pub fn newZeros(self: Self, dims: []const usize) ArrayError!Self {
            return Self.zeros(self.allocator, dims);
        }

        pub fn new_zeros(self: Self, dims: []const usize) ArrayError!Self {
            return self.newZeros(dims);
        }

        pub fn newOnes(self: Self, dims: []const usize) ArrayError!Self {
            return Self.ones(self.allocator, dims);
        }

        pub fn new_ones(self: Self, dims: []const usize) ArrayError!Self {
            return self.newOnes(dims);
        }

        pub fn newFull(self: Self, dims: []const usize, value: T) ArrayError!Self {
            return Self.full(self.allocator, dims, value);
        }

        pub fn new_full(self: Self, dims: []const usize, value: T) ArrayError!Self {
            return self.newFull(dims, value);
        }

        pub fn arange(allocator: std.mem.Allocator, start: T, stop: T, step: T) ArrayError!Self {
            ensureNumeric(T);
            if (step == zero(T)) return error.InvalidShape;
            var count: usize = 0;
            var value = start;
            if (step > zero(T)) {
                while (value < stop) : (value += step) count += 1;
            } else {
                while (value > stop) : (value += step) count += 1;
            }
            const out = try Self.empty(allocator, &.{count});
            value = start;
            for (out.data) |*slot| {
                slot.* = value;
                value += step;
            }
            return out;
        }

        pub fn linspace(allocator: std.mem.Allocator, start: T, stop: T, count: usize) ArrayError!Self {
            ensureNumeric(T);
            var out = try Self.empty(allocator, &.{count});
            if (count == 0) return out;
            if (count == 1) {
                out.data[0] = start;
                return out;
            }
            const denom: T = castValue(T, count - 1);
            const step = (stop - start) / denom;
            for (out.data, 0..) |*slot, i| {
                slot.* = start + step * castValue(T, i);
            }
            return out;
        }

        pub fn logspace(allocator: std.mem.Allocator, start: T, stop: T, count: usize, base: T) ArrayError!Self {
            ensureFloat(T);
            if (!(base > zero(T))) return error.InvalidShape;
            var exponents = try Self.linspace(allocator, start, stop, count);
            defer exponents.deinit();
            const out = try Self.empty(allocator, &.{count});
            for (exponents.data, out.data) |exponent, *slot| slot.* = std.math.pow(T, base, exponent);
            return out;
        }

        pub fn geomspace(allocator: std.mem.Allocator, start: T, stop: T, count: usize) ArrayError!Self {
            ensureFloat(T);
            if (!(start > zero(T)) or !(stop > zero(T))) return error.InvalidShape;
            if (count == 0) return Self.empty(allocator, &.{0});
            var exponents = try Self.linspace(allocator, std.math.log(T, std.math.e, start), std.math.log(T, std.math.e, stop), count);
            defer exponents.deinit();
            const out = try Self.empty(allocator, &.{count});
            for (exponents.data, out.data) |exponent, *slot| slot.* = std.math.exp(exponent);
            return out;
        }

        pub const MeshGrid2 = struct {
            x: Self,
            y: Self,

            pub fn deinit(self: *@This()) void {
                self.x.deinit();
                self.y.deinit();
                self.* = undefined;
            }
        };

        pub fn meshgrid(x_values: Self, y_values: Self, indexing: MeshGridIndexing) ArrayError!MeshGrid2 {
            if (x_values.shape.len != 1 or y_values.shape.len != 1) return error.NonVectorArray;
            const x_len = x_values.shape[0];
            const y_len = y_values.shape[0];
            const dims = switch (indexing) {
                .ij => [_]usize{ x_len, y_len },
                .xy => [_]usize{ y_len, x_len },
            };
            var x_grid = try Self.empty(x_values.allocator, dims[0..]);
            errdefer x_grid.deinit();
            var y_grid = try Self.empty(x_values.allocator, dims[0..]);
            errdefer y_grid.deinit();
            for (0..dims[0]) |r| {
                for (0..dims[1]) |c| {
                    const out_index = r * dims[1] + c;
                    switch (indexing) {
                        .ij => {
                            x_grid.data[out_index] = x_values.data[r];
                            y_grid.data[out_index] = y_values.data[c];
                        },
                        .xy => {
                            x_grid.data[out_index] = x_values.data[c];
                            y_grid.data[out_index] = y_values.data[r];
                        },
                    }
                }
            }
            return .{ .x = x_grid, .y = y_grid };
        }

        pub fn rand(allocator: std.mem.Allocator, dims: []const usize, seed: u64) ArrayError!Self {
            return Self.uniform(allocator, dims, zero(T), one(T), seed);
        }

        pub fn permutation(allocator: std.mem.Allocator, n: usize, seed: u64) ArrayError!Self {
            if (comptime T != usize) @compileError("permutation requires Array(usize)");
            var out = try Self.empty(allocator, &.{n});
            errdefer out.deinit();
            for (out.data, 0..) |*slot, i| slot.* = i;
            var engine = alea.ScalarPrng.init(seed);
            const rng = alea.Rng.init(&engine);
            rng.shuffle(usize, out.data);
            return out;
        }

        pub fn fromSlice(allocator: std.mem.Allocator, values: []const T, dims: []const usize) ArrayError!Self {
            const n = try numelFrom(dims);
            if (values.len != n) return error.ShapeMismatch;
            const data = try allocator.dupe(T, values);
            errdefer allocator.free(data);
            const shape = try allocator.dupe(usize, dims);
            errdefer allocator.free(shape);
            const strides = try stridesFor(allocator, shape);
            return .{ .allocator = allocator, .data = data, .shape = shape, .strides = strides };
        }

        pub fn fromNested2D(allocator: std.mem.Allocator, comptime rows: usize, comptime cols: usize, values: [rows][cols]T) ArrayError!Self {
            var out = try Self.empty(allocator, &.{ rows, cols });
            for (0..rows) |r| {
                @memcpy(out.data[r * cols ..][0..cols], values[r][0..]);
            }
            return out;
        }

        pub fn eye(allocator: std.mem.Allocator, n: usize) ArrayError!Self {
            const out = try Self.zeros(allocator, &.{ n, n });
            for (0..n) |i| out.data[i * n + i] = one(T);
            return out;
        }

        pub fn identity(allocator: std.mem.Allocator, n: usize) ArrayError!Self {
            return Self.eye(allocator, n);
        }

        pub fn eyeRect(allocator: std.mem.Allocator, rows: usize, cols: usize, diagonal_offset: isize) ArrayError!Self {
            const out = try Self.zeros(allocator, &.{ rows, cols });
            const start_row: usize = if (diagonal_offset < 0) @intCast(-diagonal_offset) else 0;
            const start_col: usize = if (diagonal_offset > 0) @intCast(diagonal_offset) else 0;
            if (start_row >= rows or start_col >= cols) return out;
            const count = @min(rows - start_row, cols - start_col);
            for (0..count) |i| out.data[(start_row + i) * cols + start_col + i] = one(T);
            return out;
        }

        pub fn randn(allocator: std.mem.Allocator, dims: []const usize, seed: u64) ArrayError!Self {
            return Self.normal(allocator, dims, zero(T), one(T), seed);
        }

        pub fn shuffle(self: Self, seed: u64) ArrayError!Self {
            var out = try self.clone();
            errdefer out.deinit();
            var engine = alea.ScalarPrng.init(seed);
            const rng = alea.Rng.init(&engine);
            rng.shuffle(T, out.data);
            return out;
        }

        pub fn shuffleInPlace(self: Self, seed: u64) void {
            var engine = alea.ScalarPrng.init(seed);
            const rng = alea.Rng.init(&engine);
            rng.shuffle(T, self.data);
        }

        pub fn choice(self: Self, count: usize, replacement: bool, seed: u64) ArrayError!Self {
            if (count == 0) return Self.empty(self.allocator, &.{0});
            if (self.data.len == 0) return error.EmptyArray;
            var engine = alea.ScalarPrng.init(seed);
            const rng = alea.Rng.init(&engine);
            if (replacement) {
                var out = try Self.empty(self.allocator, &.{count});
                errdefer out.deinit();
                for (out.data) |*slot| {
                    const idx = rng.intRangeLessThan(usize, 0, self.data.len);
                    slot.* = self.data[idx];
                }
                return out;
            }
            if (count > self.data.len) return error.ShapeMismatch;
            const sampled = rng.sampleWithoutReplacement(T, self.allocator, self.data, count) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                else => return error.InvalidShape,
            };
            errdefer self.allocator.free(sampled);
            const shape = try self.allocator.dupe(usize, &.{count});
            errdefer self.allocator.free(shape);
            const strides = try stridesFor(self.allocator, shape);
            return .{ .allocator = self.allocator, .data = sampled, .shape = shape, .strides = strides, .device = self.device };
        }

        pub fn choiceWeighted(self: Self, weights: Array(f64), count: usize, seed: u64) ArrayError!Self {
            if (weights.data.len != self.data.len) return error.ShapeMismatch;
            if (count == 0) return Self.empty(self.allocator, &.{0});
            if (self.data.len == 0) return error.EmptyArray;
            var engine = alea.ScalarPrng.init(seed);
            const rng = alea.Rng.init(&engine);
            const indices = rng.weightedIndexBatchChecked(self.allocator, count, weights.data) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                else => return error.InvalidShape,
            };
            defer self.allocator.free(indices);
            var out = try Self.empty(self.allocator, &.{count});
            errdefer out.deinit();
            for (out.data, indices) |*slot, idx| slot.* = self.data[idx];
            return out;
        }

        pub fn dirichlet(allocator: std.mem.Allocator, alpha: []const T, samples: usize, seed: u64) ArrayError!Self {
            ensureFloat(T);
            const distribution = alea.distributions.Dirichlet(T).init(alpha) catch return error.InvalidShape;
            var out = try Self.empty(allocator, &.{ samples, alpha.len });
            errdefer out.deinit();
            var engine = alea.ScalarPrng.init(seed);
            const rng = alea.Rng.init(&engine);
            distribution.sampleManyIntoChecked(rng, out.data) catch return error.InvalidShape;
            return out;
        }

        pub fn multinomial(allocator: std.mem.Allocator, trials: u64, probabilities: []const f64, samples: usize, seed: u64) ArrayError!Self {
            if (comptime T != u64) @compileError("multinomial requires Array(u64)");
            const distribution = alea.distributions.Multinomial.init(trials, probabilities) catch return error.InvalidShape;
            var out = try Self.empty(allocator, &.{ samples, probabilities.len });
            errdefer out.deinit();
            var engine = alea.ScalarPrng.init(seed);
            const rng = alea.Rng.init(&engine);
            distribution.sampleManyIntoChecked(rng, out.data) catch return error.InvalidShape;
            return out;
        }

        pub fn uniform(allocator: std.mem.Allocator, dims: []const usize, low: T, high: T, seed: u64) ArrayError!Self {
            if (comptime !isNumeric(T)) @compileError("uniform requires a numeric array type");
            if (low > high) return error.InvalidShape;
            var engine = alea.ScalarPrng.init(seed);
            const rng = alea.Rng.init(&engine);
            const out = try Self.empty(allocator, dims);
            for (out.data) |*slot| slot.* = alea.distributions.uniform(rng, T, low, high);
            return out;
        }

        pub fn normal(allocator: std.mem.Allocator, dims: []const usize, mean_value: T, stddev_value: T, seed: u64) ArrayError!Self {
            ensureFloat(T);
            if (stddev_value < zero(T)) return error.InvalidShape;
            var engine = alea.ScalarPrng.init(seed);
            const rng = alea.Rng.init(&engine);
            const out = try Self.empty(allocator, dims);
            for (out.data) |*slot| slot.* = alea.distributions.normal(rng, T, mean_value, stddev_value);
            return out;
        }

        pub fn randint(allocator: std.mem.Allocator, dims: []const usize, low: T, high: T, seed: u64) ArrayError!Self {
            if (comptime @typeInfo(T) != .int) @compileError("randint requires an integer array type");
            return Self.uniform(allocator, dims, low, high, seed);
        }

        pub fn bernoulli(allocator: std.mem.Allocator, dims: []const usize, p: f64, seed: u64) ArrayError!Self {
            if (comptime T != bool) @compileError("bernoulli requires Array(bool)");
            if (p < 0 or p > 1) return error.InvalidShape;
            var engine = alea.ScalarPrng.init(seed);
            const rng = alea.Rng.init(&engine);
            const out = try Self.empty(allocator, dims);
            for (out.data) |*slot| slot.* = alea.distributions.bernoulli(rng, p);
            return out;
        }

        fn randomFromAlea(
            allocator: std.mem.Allocator,
            dims: []const usize,
            seed: u64,
            context: anytype,
            comptime sampler: anytype,
        ) ArrayError!Self {
            ensureFloat(T);
            var engine = alea.ScalarPrng.init(seed);
            const rng = alea.Rng.init(&engine);
            var out = try Self.empty(allocator, dims);
            errdefer out.deinit();
            if (out.data.len == 0) {
                _ = try sampler(rng, context);
                return out;
            }
            for (out.data) |*slot| slot.* = try sampler(rng, context);
            return out;
        }

        pub fn exponential(allocator: std.mem.Allocator, dims: []const usize, rate: T, seed: u64) ArrayError!Self {
            return randomFromAlea(allocator, dims, seed, rate, struct {
                fn f(rng: alea.Rng, value: T) ArrayError!T {
                    return alea.distributions.exponentialChecked(rng, T, value) catch error.InvalidShape;
                }
            }.f);
        }

        pub fn gamma(allocator: std.mem.Allocator, dims: []const usize, shape_param: T, scale: T, seed: u64) ArrayError!Self {
            return randomFromAlea(allocator, dims, seed, .{ .shape = shape_param, .scale = scale }, struct {
                fn f(rng: alea.Rng, params: anytype) ArrayError!T {
                    return alea.distributions.gammaChecked(rng, T, params.shape, params.scale) catch error.InvalidShape;
                }
            }.f);
        }

        pub fn beta(allocator: std.mem.Allocator, dims: []const usize, alpha: T, beta_param: T, seed: u64) ArrayError!Self {
            return randomFromAlea(allocator, dims, seed, .{ .alpha = alpha, .beta_param = beta_param }, struct {
                fn f(rng: alea.Rng, params: anytype) ArrayError!T {
                    return alea.distributions.betaChecked(rng, T, params.alpha, params.beta_param) catch error.InvalidShape;
                }
            }.f);
        }

        pub fn halfNormal(allocator: std.mem.Allocator, dims: []const usize, scale: T, seed: u64) ArrayError!Self {
            return randomFromAlea(allocator, dims, seed, scale, struct {
                fn f(rng: alea.Rng, value: T) ArrayError!T {
                    return alea.distributions.halfNormalChecked(rng, T, value) catch error.InvalidShape;
                }
            }.f);
        }

        pub fn chiSquared(allocator: std.mem.Allocator, dims: []const usize, dof: T, seed: u64) ArrayError!Self {
            return randomFromAlea(allocator, dims, seed, dof, struct {
                fn f(rng: alea.Rng, value: T) ArrayError!T {
                    return alea.distributions.chiSquaredChecked(rng, T, value) catch error.InvalidShape;
                }
            }.f);
        }

        pub fn chi(allocator: std.mem.Allocator, dims: []const usize, dof: T, seed: u64) ArrayError!Self {
            return randomFromAlea(allocator, dims, seed, dof, struct {
                fn f(rng: alea.Rng, value: T) ArrayError!T {
                    return alea.distributions.chiChecked(rng, T, value) catch error.InvalidShape;
                }
            }.f);
        }

        pub fn erlang(allocator: std.mem.Allocator, dims: []const usize, shape: u64, scale: T, seed: u64) ArrayError!Self {
            return randomFromAlea(allocator, dims, seed, .{ .shape = shape, .scale = scale }, struct {
                fn f(rng: alea.Rng, params: anytype) ArrayError!T {
                    return alea.distributions.erlangChecked(rng, T, params.shape, params.scale) catch error.InvalidShape;
                }
            }.f);
        }

        pub fn fisherF(allocator: std.mem.Allocator, dims: []const usize, d1: T, d2: T, seed: u64) ArrayError!Self {
            return randomFromAlea(allocator, dims, seed, .{ .d1 = d1, .d2 = d2 }, struct {
                fn f(rng: alea.Rng, params: anytype) ArrayError!T {
                    return alea.distributions.fisherFChecked(rng, T, params.d1, params.d2) catch error.InvalidShape;
                }
            }.f);
        }

        pub fn triangular(allocator: std.mem.Allocator, dims: []const usize, min_value: T, mode_value: T, max_value: T, seed: u64) ArrayError!Self {
            return randomFromAlea(allocator, dims, seed, .{ .min_value = min_value, .mode_value = mode_value, .max_value = max_value }, struct {
                fn f(rng: alea.Rng, params: anytype) ArrayError!T {
                    return alea.distributions.triangularChecked(rng, T, params.min_value, params.mode_value, params.max_value) catch error.InvalidShape;
                }
            }.f);
        }

        pub fn arcsine(allocator: std.mem.Allocator, dims: []const usize, min_value: T, max_value: T, seed: u64) ArrayError!Self {
            return randomFromAlea(allocator, dims, seed, .{ .min_value = min_value, .max_value = max_value }, struct {
                fn f(rng: alea.Rng, params: anytype) ArrayError!T {
                    return alea.distributions.arcsineChecked(rng, T, params.min_value, params.max_value) catch error.InvalidShape;
                }
            }.f);
        }

        pub fn logistic(allocator: std.mem.Allocator, dims: []const usize, location: T, scale: T, seed: u64) ArrayError!Self {
            return randomFromAlea(allocator, dims, seed, .{ .location = location, .scale = scale }, struct {
                fn f(rng: alea.Rng, params: anytype) ArrayError!T {
                    return alea.distributions.logisticChecked(rng, T, params.location, params.scale) catch error.InvalidShape;
                }
            }.f);
        }

        pub fn logLogistic(allocator: std.mem.Allocator, dims: []const usize, scale: T, shape: T, seed: u64) ArrayError!Self {
            return randomFromAlea(allocator, dims, seed, .{ .scale = scale, .shape = shape }, struct {
                fn f(rng: alea.Rng, params: anytype) ArrayError!T {
                    return alea.distributions.logLogisticChecked(rng, T, params.scale, params.shape) catch error.InvalidShape;
                }
            }.f);
        }

        pub fn kumaraswamy(allocator: std.mem.Allocator, dims: []const usize, alpha: T, beta_param: T, seed: u64) ArrayError!Self {
            return randomFromAlea(allocator, dims, seed, .{ .alpha = alpha, .beta_param = beta_param }, struct {
                fn f(rng: alea.Rng, params: anytype) ArrayError!T {
                    return alea.distributions.kumaraswamyChecked(rng, T, params.alpha, params.beta_param) catch error.InvalidShape;
                }
            }.f);
        }

        pub fn powerFunction(allocator: std.mem.Allocator, dims: []const usize, min_value: T, max_value: T, shape: T, seed: u64) ArrayError!Self {
            return randomFromAlea(allocator, dims, seed, .{ .min_value = min_value, .max_value = max_value, .shape = shape }, struct {
                fn f(rng: alea.Rng, params: anytype) ArrayError!T {
                    return alea.distributions.powerFunctionChecked(rng, T, params.min_value, params.max_value, params.shape) catch error.InvalidShape;
                }
            }.f);
        }

        pub fn rayleigh(allocator: std.mem.Allocator, dims: []const usize, scale: T, seed: u64) ArrayError!Self {
            return randomFromAlea(allocator, dims, seed, scale, struct {
                fn f(rng: alea.Rng, value: T) ArrayError!T {
                    return alea.distributions.rayleighChecked(rng, T, value) catch error.InvalidShape;
                }
            }.f);
        }

        pub fn maxwell(allocator: std.mem.Allocator, dims: []const usize, scale: T, seed: u64) ArrayError!Self {
            return randomFromAlea(allocator, dims, seed, scale, struct {
                fn f(rng: alea.Rng, value: T) ArrayError!T {
                    return alea.distributions.maxwellChecked(rng, T, value) catch error.InvalidShape;
                }
            }.f);
        }

        pub fn pareto(allocator: std.mem.Allocator, dims: []const usize, scale: T, shape: T, seed: u64) ArrayError!Self {
            return randomFromAlea(allocator, dims, seed, .{ .scale = scale, .shape = shape }, struct {
                fn f(rng: alea.Rng, params: anytype) ArrayError!T {
                    return alea.distributions.paretoChecked(rng, T, params.scale, params.shape) catch error.InvalidShape;
                }
            }.f);
        }

        pub fn gumbel(allocator: std.mem.Allocator, dims: []const usize, location: T, scale: T, seed: u64) ArrayError!Self {
            return randomFromAlea(allocator, dims, seed, .{ .location = location, .scale = scale }, struct {
                fn f(rng: alea.Rng, params: anytype) ArrayError!T {
                    return alea.distributions.gumbelChecked(rng, T, params.location, params.scale) catch error.InvalidShape;
                }
            }.f);
        }

        pub fn frechet(allocator: std.mem.Allocator, dims: []const usize, location: T, scale: T, shape: T, seed: u64) ArrayError!Self {
            return randomFromAlea(allocator, dims, seed, .{ .location = location, .scale = scale, .shape = shape }, struct {
                fn f(rng: alea.Rng, params: anytype) ArrayError!T {
                    return alea.distributions.frechetChecked(rng, T, params.location, params.scale, params.shape) catch error.InvalidShape;
                }
            }.f);
        }

        pub fn skewNormal(allocator: std.mem.Allocator, dims: []const usize, location: T, scale: T, shape: T, seed: u64) ArrayError!Self {
            return randomFromAlea(allocator, dims, seed, .{ .location = location, .scale = scale, .shape = shape }, struct {
                fn f(rng: alea.Rng, params: anytype) ArrayError!T {
                    return alea.distributions.skewNormalChecked(rng, T, params.location, params.scale, params.shape) catch error.InvalidShape;
                }
            }.f);
        }

        pub fn pert(allocator: std.mem.Allocator, dims: []const usize, min_value: T, mode_value: T, max_value: T, shape: T, seed: u64) ArrayError!Self {
            return randomFromAlea(allocator, dims, seed, .{ .min_value = min_value, .mode_value = mode_value, .max_value = max_value, .shape = shape }, struct {
                fn f(rng: alea.Rng, params: anytype) ArrayError!T {
                    return alea.distributions.pertChecked(rng, T, params.min_value, params.mode_value, params.max_value, params.shape) catch error.InvalidShape;
                }
            }.f);
        }

        pub fn inverseGaussian(allocator: std.mem.Allocator, dims: []const usize, mean_value: T, shape: T, seed: u64) ArrayError!Self {
            return randomFromAlea(allocator, dims, seed, .{ .mean_value = mean_value, .shape = shape }, struct {
                fn f(rng: alea.Rng, params: anytype) ArrayError!T {
                    return alea.distributions.inverseGaussianChecked(rng, T, params.mean_value, params.shape) catch error.InvalidShape;
                }
            }.f);
        }

        pub fn normalInverseGaussian(allocator: std.mem.Allocator, dims: []const usize, alpha: T, beta_param: T, seed: u64) ArrayError!Self {
            return randomFromAlea(allocator, dims, seed, .{ .alpha = alpha, .beta_param = beta_param }, struct {
                fn f(rng: alea.Rng, params: anytype) ArrayError!T {
                    return alea.distributions.normalInverseGaussianChecked(rng, T, params.alpha, params.beta_param) catch error.InvalidShape;
                }
            }.f);
        }

        pub fn poisson(allocator: std.mem.Allocator, dims: []const usize, lambda: f64, seed: u64) ArrayError!Self {
            if (comptime T != u64) @compileError("poisson requires Array(u64)");
            if (!(lambda >= 0)) return error.InvalidShape;
            var engine = alea.ScalarPrng.init(seed);
            const rng = alea.Rng.init(&engine);
            var out = try Self.empty(allocator, dims);
            errdefer out.deinit();
            for (out.data) |*slot| slot.* = alea.distributions.poissonChecked(rng, lambda) catch return error.InvalidShape;
            return out;
        }

        pub fn lognormal(allocator: std.mem.Allocator, dims: []const usize, mean_value: T, stddev_value: T, seed: u64) ArrayError!Self {
            return randomFromAlea(allocator, dims, seed, .{ .mean_value = mean_value, .stddev_value = stddev_value }, struct {
                fn f(rng: alea.Rng, params: anytype) ArrayError!T {
                    return alea.distributions.logNormalChecked(rng, T, params.mean_value, params.stddev_value) catch error.InvalidShape;
                }
            }.f);
        }

        pub fn studentT(allocator: std.mem.Allocator, dims: []const usize, dof: T, seed: u64) ArrayError!Self {
            return randomFromAlea(allocator, dims, seed, dof, struct {
                fn f(rng: alea.Rng, value: T) ArrayError!T {
                    return alea.distributions.studentTChecked(rng, T, value) catch error.InvalidShape;
                }
            }.f);
        }

        pub fn cauchy(allocator: std.mem.Allocator, dims: []const usize, median_value: T, scale: T, seed: u64) ArrayError!Self {
            return randomFromAlea(allocator, dims, seed, .{ .median_value = median_value, .scale = scale }, struct {
                fn f(rng: alea.Rng, params: anytype) ArrayError!T {
                    return alea.distributions.cauchyChecked(rng, T, params.median_value, params.scale) catch error.InvalidShape;
                }
            }.f);
        }

        pub fn laplace(allocator: std.mem.Allocator, dims: []const usize, location: T, scale: T, seed: u64) ArrayError!Self {
            return randomFromAlea(allocator, dims, seed, .{ .location = location, .scale = scale }, struct {
                fn f(rng: alea.Rng, params: anytype) ArrayError!T {
                    return alea.distributions.laplaceChecked(rng, T, params.location, params.scale) catch error.InvalidShape;
                }
            }.f);
        }

        pub fn weibull(allocator: std.mem.Allocator, dims: []const usize, scale: T, shape_param: T, seed: u64) ArrayError!Self {
            return randomFromAlea(allocator, dims, seed, .{ .scale = scale, .shape_param = shape_param }, struct {
                fn f(rng: alea.Rng, params: anytype) ArrayError!T {
                    return alea.distributions.weibullChecked(rng, T, params.scale, params.shape_param) catch error.InvalidShape;
                }
            }.f);
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.data);
            self.allocator.free(self.shape);
            self.allocator.free(self.strides);
            self.* = undefined;
        }

        pub fn clone(self: Self) ArrayError!Self {
            return Self.fromSlice(self.allocator, self.data, self.shape);
        }

        pub fn astype(self: Self, comptime U: type) ArrayError!Array(U) {
            const out = try Array(U).empty(self.allocator, self.shape);
            for (self.data, out.data) |v, *slot| {
                slot.* = castValue(U, v);
            }
            return out;
        }

        pub fn real(self: Self) ArrayError!Array(complexRealType(T)) {
            ensureComplex(T);
            const Real = complexRealType(T);
            var out = try Array(Real).empty(self.allocator, self.shape);
            errdefer out.deinit();
            for (self.data, out.data) |value, *slot| slot.* = value.re;
            return out;
        }

        pub fn imag(self: Self) ArrayError!Array(complexRealType(T)) {
            ensureComplex(T);
            const Real = complexRealType(T);
            var out = try Array(Real).empty(self.allocator, self.shape);
            errdefer out.deinit();
            for (self.data, out.data) |value, *slot| slot.* = value.im;
            return out;
        }

        pub fn conjugate(self: Self) ArrayError!Self {
            ensureComplex(T);
            return self.unary(struct {
                fn f(value: T) T {
                    return value.conjugate();
                }
            }.f);
        }

        pub fn conj(self: Self) ArrayError!Self {
            return self.conjugate();
        }

        pub fn magnitude(self: Self) ArrayError!Array(complexRealType(T)) {
            ensureComplex(T);
            const Real = complexRealType(T);
            var out = try Array(Real).empty(self.allocator, self.shape);
            errdefer out.deinit();
            for (self.data, out.data) |value, *slot| slot.* = value.magnitude();
            return out;
        }

        pub fn absComplex(self: Self) ArrayError!Array(complexRealType(T)) {
            return self.magnitude();
        }

        pub fn angle(self: Self) ArrayError!Array(complexRealType(T)) {
            ensureComplex(T);
            const Real = complexRealType(T);
            var out = try Array(Real).empty(self.allocator, self.shape);
            errdefer out.deinit();
            for (self.data, out.data) |value, *slot| slot.* = std.math.atan2(value.im, value.re);
            return out;
        }

        pub fn phase(self: Self) ArrayError!Array(complexRealType(T)) {
            return self.angle();
        }

        fn fftWithSign(self: Self, inverse: bool) ArrayError!Self {
            ensureComplex(T);
            if (self.shape.len != 1) return error.NonVectorArray;
            const n = self.shape[0];
            var out = try Self.empty(self.allocator, self.shape);
            errdefer out.deinit();
            if (n == 0) return out;
            const Real = complexRealType(T);
            const n_real: Real = @floatFromInt(n);
            const direction: Real = if (inverse) 1 else -1;
            for (0..n) |k| {
                var acc = zero(T);
                const k_real: Real = @floatFromInt(k);
                for (0..n) |j| {
                    const j_real: Real = @floatFromInt(j);
                    const phase_angle = direction * castValue(Real, 2.0 * std.math.pi) * k_real * j_real / n_real;
                    const twiddle = T.init(@cos(phase_angle), @sin(phase_angle));
                    acc = acc.add(self.data[j].mul(twiddle));
                }
                out.data[k] = if (inverse) acc.div(T.init(n_real, 0)) else acc;
            }
            return out;
        }

        fn fftAxisWithSign(self: Self, axis_index: isize, inverse: bool) ArrayError!Self {
            ensureComplex(T);
            if (self.shape.len == 0) return error.InvalidAxis;
            const axis = try normalizeDim(axis_index, self.shape.len);
            const n = self.shape[axis];
            var out = try Self.empty(self.allocator, self.shape);
            errdefer out.deinit();
            if (out.data.len == 0) return out;

            const Real = complexRealType(T);
            const n_real: Real = @floatFromInt(n);
            const direction: Real = if (inverse) 1 else -1;
            const out_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(out_multi);
            var in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);

            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, self.shape, out_multi);
                const k = out_multi[axis];
                const k_real: Real = @floatFromInt(k);
                var acc = zero(T);
                @memcpy(in_multi, out_multi);
                for (0..n) |j| {
                    in_multi[axis] = j;
                    const j_real: Real = @floatFromInt(j);
                    const phase_angle = direction * castValue(Real, 2.0 * std.math.pi) * k_real * j_real / n_real;
                    const twiddle = T.init(@cos(phase_angle), @sin(phase_angle));
                    acc = acc.add(self.data[ravelIndex(in_multi, self.strides)].mul(twiddle));
                }
                slot.* = if (inverse) acc.div(T.init(n_real, 0)) else acc;
            }
            return out;
        }

        pub fn fft(self: Self) ArrayError!Self {
            return self.fftWithSign(false);
        }

        pub fn ifft(self: Self) ArrayError!Self {
            return self.fftWithSign(true);
        }

        pub fn rfft(self: Self) ArrayError!Array(complexTypeForReal(T)) {
            ensureFloat(T);
            if (self.shape.len != 1) return error.NonVectorArray;
            const C = complexTypeForReal(T);
            const Real = complexRealType(C);
            const n = self.shape[0];
            const out_len = n / 2 + 1;
            var out = try Array(C).empty(self.allocator, &.{out_len});
            errdefer out.deinit();
            if (n == 0) return out;
            const n_real: Real = @floatFromInt(n);
            for (0..out_len) |k| {
                const k_real: Real = @floatFromInt(k);
                var acc = C.init(0, 0);
                for (0..n) |j| {
                    const j_real: Real = @floatFromInt(j);
                    const phase_angle = -castValue(Real, 2.0 * std.math.pi) * k_real * j_real / n_real;
                    const twiddle = C.init(@cos(phase_angle), @sin(phase_angle));
                    acc = acc.add(C.init(castValue(Real, self.data[j]), 0).mul(twiddle));
                }
                out.data[k] = acc;
            }
            return out;
        }

        pub fn irfft(self: Self, output_len: ?usize) ArrayError!Array(realTypeForComplex(T)) {
            ensureComplex(T);
            if (self.shape.len != 1) return error.NonVectorArray;
            const Real = realTypeForComplex(T);
            const n = output_len orelse if (self.data.len == 0) 0 else (self.data.len - 1) * 2;
            if (self.data.len != n / 2 + 1) return error.ShapeMismatch;
            var out = try Array(Real).empty(self.allocator, &.{n});
            errdefer out.deinit();
            if (n == 0) return out;
            const n_real: Real = @floatFromInt(n);
            for (0..n) |j| {
                const j_real: Real = @floatFromInt(j);
                var acc = zero(T);
                for (0..n) |k| {
                    const spectrum_value = if (k < self.data.len)
                        self.data[k]
                    else
                        self.data[n - k].conjugate();
                    const k_real: Real = @floatFromInt(k);
                    const phase_angle = castValue(Real, 2.0 * std.math.pi) * k_real * j_real / n_real;
                    const twiddle = T.init(@cos(phase_angle), @sin(phase_angle));
                    acc = acc.add(spectrum_value.mul(twiddle));
                }
                out.data[j] = acc.re / n_real;
            }
            return out;
        }

        pub fn fftAxis(self: Self, axis_index: isize) ArrayError!Self {
            return self.fftAxisWithSign(axis_index, false);
        }

        pub fn ifftAxis(self: Self, axis_index: isize) ArrayError!Self {
            return self.fftAxisWithSign(axis_index, true);
        }

        fn fftAxesWithSign(self: Self, axes: []const isize, inverse: bool) ArrayError!Self {
            ensureComplex(T);
            var current = try self.clone();
            errdefer current.deinit();
            for (axes) |axis_index| {
                const next = try current.fftAxisWithSign(axis_index, inverse);
                current.deinit();
                current = next;
            }
            return current;
        }

        pub fn fftAxes(self: Self, axes: []const isize) ArrayError!Self {
            return self.fftAxesWithSign(axes, false);
        }

        pub fn ifftAxes(self: Self, axes: []const isize) ArrayError!Self {
            return self.fftAxesWithSign(axes, true);
        }

        pub fn fft2(self: Self) ArrayError!Self {
            if (self.shape.len < 2) return error.InvalidAxis;
            return self.fftAxes(&.{ -2, -1 });
        }

        pub fn ifft2(self: Self) ArrayError!Self {
            if (self.shape.len < 2) return error.InvalidAxis;
            return self.ifftAxes(&.{ -2, -1 });
        }

        fn trimConvolutionResult(full_result_in: Self, left_len: usize, right_len: usize, mode: ConvMode) ArrayError!Self {
            var full_result = full_result_in;
            switch (mode) {
                .full => return full_result,
                .same, .valid => {
                    const out_len: usize = switch (mode) {
                        .same => left_len,
                        .valid => if (left_len >= right_len) left_len - right_len + 1 else right_len - left_len + 1,
                        .full => unreachable,
                    };
                    const start: usize = switch (mode) {
                        .same => (full_result.data.len - out_len) / 2,
                        .valid => @min(left_len, right_len) - 1,
                        .full => unreachable,
                    };
                    var out = try Self.empty(full_result.allocator, &.{out_len});
                    errdefer out.deinit();
                    @memcpy(out.data, full_result.data[start..][0..out_len]);
                    full_result.deinit();
                    return out;
                },
            }
        }

        pub fn convolve1d(self: Self, kernel: Self, mode: ConvMode) ArrayError!Self {
            ensureNumeric(T);
            if (self.shape.len != 1 or kernel.shape.len != 1) return error.NonVectorArray;
            if (self.data.len == 0 or kernel.data.len == 0) return error.EmptyArray;
            const out_len = self.data.len + kernel.data.len - 1;
            var full_result = try Self.full(self.allocator, &.{out_len}, zero(T));
            errdefer full_result.deinit();
            for (0..out_len) |out_i| {
                var acc = zero(T);
                for (0..self.data.len) |signal_i| {
                    if (out_i < signal_i) continue;
                    const kernel_i = out_i - signal_i;
                    if (kernel_i >= kernel.data.len) continue;
                    acc = addValue(T, acc, mulValue(T, self.data[signal_i], kernel.data[kernel_i]));
                }
                full_result.data[out_i] = acc;
            }
            return trimConvolutionResult(full_result, self.data.len, kernel.data.len, mode);
        }

        pub fn correlate1d(self: Self, kernel: Self, mode: ConvMode) ArrayError!Self {
            ensureNumeric(T);
            if (self.shape.len != 1 or kernel.shape.len != 1) return error.NonVectorArray;
            if (self.data.len == 0 or kernel.data.len == 0) return error.EmptyArray;
            const out_len = self.data.len + kernel.data.len - 1;
            var full_result = try Self.full(self.allocator, &.{out_len}, zero(T));
            errdefer full_result.deinit();
            const lag_offset = kernel.data.len - 1;
            for (0..out_len) |out_i| {
                const lag: isize = @as(isize, @intCast(out_i)) - @as(isize, @intCast(lag_offset));
                var acc = zero(T);
                for (0..kernel.data.len) |kernel_i| {
                    const signal_i_signed = @as(isize, @intCast(kernel_i)) + lag;
                    if (signal_i_signed < 0) continue;
                    const signal_i: usize = @intCast(signal_i_signed);
                    if (signal_i >= self.data.len) continue;
                    acc = addValue(T, acc, mulValue(T, self.data[signal_i], kernel.data[kernel_i]));
                }
                full_result.data[out_i] = acc;
            }
            return trimConvolutionResult(full_result, self.data.len, kernel.data.len, mode);
        }

        fn trimConvolution2DResult(full_result_in: Self, input_shape: []const usize, kernel_shape: []const usize, mode: ConvMode) ArrayError!Self {
            var full_result = full_result_in;
            switch (mode) {
                .full => return full_result,
                .same, .valid => {
                    const out_rows: usize = switch (mode) {
                        .same => input_shape[0],
                        .valid => if (input_shape[0] >= kernel_shape[0]) input_shape[0] - kernel_shape[0] + 1 else kernel_shape[0] - input_shape[0] + 1,
                        .full => unreachable,
                    };
                    const out_cols: usize = switch (mode) {
                        .same => input_shape[1],
                        .valid => if (input_shape[1] >= kernel_shape[1]) input_shape[1] - kernel_shape[1] + 1 else kernel_shape[1] - input_shape[1] + 1,
                        .full => unreachable,
                    };
                    const start_row: usize = switch (mode) {
                        .same => (full_result.shape[0] - out_rows) / 2,
                        .valid => @min(input_shape[0], kernel_shape[0]) - 1,
                        .full => unreachable,
                    };
                    const start_col: usize = switch (mode) {
                        .same => (full_result.shape[1] - out_cols) / 2,
                        .valid => @min(input_shape[1], kernel_shape[1]) - 1,
                        .full => unreachable,
                    };
                    var out = try Self.empty(full_result.allocator, &.{ out_rows, out_cols });
                    errdefer out.deinit();
                    for (0..out_rows) |row| {
                        const src_start = (start_row + row) * full_result.shape[1] + start_col;
                        @memcpy(out.data[row * out_cols ..][0..out_cols], full_result.data[src_start..][0..out_cols]);
                    }
                    full_result.deinit();
                    return out;
                },
            }
        }

        pub fn convolve2d(self: Self, kernel: Self, mode: ConvMode) ArrayError!Self {
            ensureNumeric(T);
            if (self.shape.len != 2 or kernel.shape.len != 2) return error.NonMatrixArray;
            if (self.data.len == 0 or kernel.data.len == 0) return error.EmptyArray;
            const rows = self.shape[0];
            const cols = self.shape[1];
            const kernel_rows = kernel.shape[0];
            const kernel_cols = kernel.shape[1];
            var full_result = try Self.full(self.allocator, &.{ rows + kernel_rows - 1, cols + kernel_cols - 1 }, zero(T));
            errdefer full_result.deinit();
            for (0..full_result.shape[0]) |out_row| {
                for (0..full_result.shape[1]) |out_col| {
                    var acc = zero(T);
                    for (0..rows) |row| {
                        if (out_row < row) continue;
                        const kernel_row = out_row - row;
                        if (kernel_row >= kernel_rows) continue;
                        for (0..cols) |col| {
                            if (out_col < col) continue;
                            const kernel_col = out_col - col;
                            if (kernel_col >= kernel_cols) continue;
                            acc = addValue(T, acc, mulValue(T, self.data[row * cols + col], kernel.data[kernel_row * kernel_cols + kernel_col]));
                        }
                    }
                    full_result.data[out_row * full_result.shape[1] + out_col] = acc;
                }
            }
            return trimConvolution2DResult(full_result, self.shape, kernel.shape, mode);
        }

        pub fn correlate2d(self: Self, kernel: Self, mode: ConvMode) ArrayError!Self {
            ensureNumeric(T);
            if (self.shape.len != 2 or kernel.shape.len != 2) return error.NonMatrixArray;
            if (self.data.len == 0 or kernel.data.len == 0) return error.EmptyArray;
            const rows = self.shape[0];
            const cols = self.shape[1];
            const kernel_rows = kernel.shape[0];
            const kernel_cols = kernel.shape[1];
            var full_result = try Self.full(self.allocator, &.{ rows + kernel_rows - 1, cols + kernel_cols - 1 }, zero(T));
            errdefer full_result.deinit();
            const row_lag_offset = kernel_rows - 1;
            const col_lag_offset = kernel_cols - 1;
            for (0..full_result.shape[0]) |out_row| {
                const row_lag: isize = @as(isize, @intCast(out_row)) - @as(isize, @intCast(row_lag_offset));
                for (0..full_result.shape[1]) |out_col| {
                    const col_lag: isize = @as(isize, @intCast(out_col)) - @as(isize, @intCast(col_lag_offset));
                    var acc = zero(T);
                    for (0..kernel_rows) |kernel_row| {
                        const row_signed = @as(isize, @intCast(kernel_row)) + row_lag;
                        if (row_signed < 0) continue;
                        const row: usize = @intCast(row_signed);
                        if (row >= rows) continue;
                        for (0..kernel_cols) |kernel_col| {
                            const col_signed = @as(isize, @intCast(kernel_col)) + col_lag;
                            if (col_signed < 0) continue;
                            const col: usize = @intCast(col_signed);
                            if (col >= cols) continue;
                            acc = addValue(T, acc, mulValue(T, self.data[row * cols + col], kernel.data[kernel_row * kernel_cols + kernel_col]));
                        }
                    }
                    full_result.data[out_row * full_result.shape[1] + out_col] = acc;
                }
            }
            return trimConvolution2DResult(full_result, self.shape, kernel.shape, mode);
        }

        pub fn to(self: Self, device: Device) ArrayError!Self {
            if (!device.isAvailable()) return error.InvalidDevice;
            var out = try self.clone();
            out.device = device;
            return out;
        }

        pub fn cpu(self: Self) ArrayError!Self {
            return self.to(.cpu);
        }

        pub fn cuda(self: Self, index: usize) ArrayError!Self {
            return self.to(Device.cuda(index));
        }

        pub fn fill(self: Self, value: T) void {
            @memset(self.data, value);
        }

        pub fn copyFrom(self: Self, source: Self) ArrayError!void {
            var dest_view = try self.asView();
            defer dest_view.deinit();
            var source_view = try source.asView();
            defer source_view.deinit();
            return dest_view.copyFromView(source_view);
        }

        pub fn copyFromView(self: Self, source: ArrayView(T)) ArrayError!void {
            var dest_view = try self.asView();
            defer dest_view.deinit();
            return dest_view.copyFromView(source);
        }

        pub fn maskedFillAssign(self: Self, mask: Array(bool), value: T) ArrayError!void {
            var dest_view = try self.asView();
            defer dest_view.deinit();
            return dest_view.maskedFill(mask, value);
        }

        pub fn maskedCopyFrom(self: Self, mask: Array(bool), values: Self) ArrayError!void {
            var dest_view = try self.asView();
            defer dest_view.deinit();
            return dest_view.maskedCopyFromArray(mask, values);
        }

        pub fn maskedCopyFromView(self: Self, mask: Array(bool), values: ArrayView(T)) ArrayError!void {
            var dest_view = try self.asView();
            defer dest_view.deinit();
            return dest_view.maskedCopyFromView(mask, values);
        }

        pub fn copyWhereAssign(self: Self, mask: Array(bool), source: Self) ArrayError!void {
            var dest_view = try self.asView();
            defer dest_view.deinit();
            return dest_view.copyWhereFromArray(mask, source);
        }

        pub fn copyWhereAssignView(self: Self, mask: Array(bool), source: ArrayView(T)) ArrayError!void {
            var dest_view = try self.asView();
            defer dest_view.deinit();
            return dest_view.copyWhereFromView(mask, source);
        }

        pub fn addAssign(self: Self, source: Self) ArrayError!void {
            var dest_view = try self.asView();
            defer dest_view.deinit();
            return dest_view.addAssignArray(source);
        }

        pub fn subAssign(self: Self, source: Self) ArrayError!void {
            var dest_view = try self.asView();
            defer dest_view.deinit();
            return dest_view.subAssignArray(source);
        }

        pub fn mulAssign(self: Self, source: Self) ArrayError!void {
            var dest_view = try self.asView();
            defer dest_view.deinit();
            return dest_view.mulAssignArray(source);
        }

        pub fn divAssign(self: Self, source: Self) ArrayError!void {
            var dest_view = try self.asView();
            defer dest_view.deinit();
            return dest_view.divAssignArray(source);
        }

        pub fn addAssignView(self: Self, source: ArrayView(T)) ArrayError!void {
            var dest_view = try self.asView();
            defer dest_view.deinit();
            return dest_view.addAssign(source);
        }

        pub fn subAssignView(self: Self, source: ArrayView(T)) ArrayError!void {
            var dest_view = try self.asView();
            defer dest_view.deinit();
            return dest_view.subAssign(source);
        }

        pub fn mulAssignView(self: Self, source: ArrayView(T)) ArrayError!void {
            var dest_view = try self.asView();
            defer dest_view.deinit();
            return dest_view.mulAssign(source);
        }

        pub fn divAssignView(self: Self, source: ArrayView(T)) ArrayError!void {
            var dest_view = try self.asView();
            defer dest_view.deinit();
            return dest_view.divAssign(source);
        }

        pub fn addScalarAssign(self: Self, scalar: T) ArrayError!void {
            var dest_view = try self.asView();
            defer dest_view.deinit();
            return dest_view.addScalarAssign(scalar);
        }

        pub fn subScalarAssign(self: Self, scalar: T) ArrayError!void {
            var dest_view = try self.asView();
            defer dest_view.deinit();
            return dest_view.subScalarAssign(scalar);
        }

        pub fn mulScalarAssign(self: Self, scalar: T) ArrayError!void {
            var dest_view = try self.asView();
            defer dest_view.deinit();
            return dest_view.mulScalarAssign(scalar);
        }

        pub fn divScalarAssign(self: Self, scalar: T) ArrayError!void {
            var dest_view = try self.asView();
            defer dest_view.deinit();
            return dest_view.divScalarAssign(scalar);
        }

        pub fn numel(self: Self) usize {
            return self.data.len;
        }

        pub fn ndim(self: Self) usize {
            return self.shape.len;
        }

        pub fn dim(self: Self) usize {
            return self.ndim();
        }

        pub fn rank(self: Self) usize {
            return self.ndim();
        }

        pub fn numDims(self: Self) usize {
            return self.ndim();
        }

        pub fn size(self: Self, axis_opt: ?isize) ArrayError!usize {
            if (axis_opt) |d| return self.shape[try normalizeDim(d, self.shape.len)];
            return self.numel();
        }

        pub fn nelement(self: Self) usize {
            return self.numel();
        }

        pub fn isEmpty(self: Self) bool {
            return self.numel() == 0;
        }

        pub fn shapeAt(self: Self, axis_index: isize) ArrayError!usize {
            return self.shape[try normalizeDim(axis_index, self.shape.len)];
        }

        pub fn len(self: Self) ArrayError!usize {
            if (self.shape.len == 0) return error.InvalidShape;
            return self.shape[0];
        }

        pub fn stride(self: Self, axis_index: isize) ArrayError!usize {
            return self.strides[try normalizeDim(axis_index, self.shape.len)];
        }

        pub fn strideAt(self: Self, axis_index: isize) ArrayError!usize {
            return self.stride(axis_index);
        }

        pub fn elementSize(self: Self) usize {
            _ = self;
            return @sizeOf(T);
        }

        pub fn nbytes(self: Self) usize {
            return self.numel() * @sizeOf(T);
        }

        pub fn sameShape(self: Self, other: Self) bool {
            return std.mem.eql(usize, self.shape, other.shape);
        }

        pub fn isContiguous(self: Self) bool {
            var expected: usize = 1;
            var i = self.shape.len;
            while (i > 0) {
                i -= 1;
                if (self.strides[i] != expected) return false;
                expected *= self.shape[i];
            }
            return true;
        }

        pub fn is_contiguous(self: Self) bool {
            return self.isContiguous();
        }

        pub fn contiguous(self: Self) ArrayError!Self {
            return self.clone();
        }

        pub fn asView(self: Self) ArrayError!ArrayView(T) {
            return ArrayView(T).fromArray(self);
        }

        pub fn asStrided(self: Self, dims: []const usize, stride_values: []const usize, offset: usize) ArrayError!ArrayView(T) {
            try validateStridedBounds(self.data.len, offset, dims, stride_values);
            return ArrayView(T).init(self.allocator, self.data, dims, stride_values, offset, self.device);
        }

        pub fn unfold(self: Self, axis_index: isize, window_size: usize, step: usize) ArrayError!ArrayView(T) {
            var base = try self.asView();
            defer base.deinit();
            return base.unfold(axis_index, window_size, step);
        }

        pub fn sliceAxisView(self: Self, axis_index: isize, slice_value: Slice) ArrayError!ArrayView(T) {
            var base = try self.asView();
            defer base.deinit();
            return base.sliceAxis(axis_index, slice_value);
        }

        pub fn sliceView(self: Self, slices: []const Slice) ArrayError!ArrayView(T) {
            var base = try self.asView();
            defer base.deinit();
            return base.slice(slices);
        }

        pub fn selectView(self: Self, axis_index: isize, index: usize) ArrayError!ArrayView(T) {
            var base = try self.asView();
            defer base.deinit();
            return base.select(axis_index, index);
        }

        pub fn narrowView(self: Self, axis_index: isize, start: usize, length: usize) ArrayError!ArrayView(T) {
            var base = try self.asView();
            defer base.deinit();
            return base.narrow(axis_index, start, length);
        }

        pub fn permuteView(self: Self, axes: []const usize) ArrayError!ArrayView(T) {
            var base = try self.asView();
            defer base.deinit();
            return base.permute(axes);
        }

        pub fn swapaxesView(self: Self, dim0: isize, dim1: isize) ArrayError!ArrayView(T) {
            var base = try self.asView();
            defer base.deinit();
            return base.swapaxes(dim0, dim1);
        }

        pub fn movedimView(self: Self, source: isize, destination: isize) ArrayError!ArrayView(T) {
            var base = try self.asView();
            defer base.deinit();
            return base.movedim(source, destination);
        }

        pub fn transposeView(self: Self) ArrayError!ArrayView(T) {
            var base = try self.asView();
            defer base.deinit();
            return base.transpose();
        }

        pub fn broadcastView(self: Self, dims: []const usize) ArrayError!ArrayView(T) {
            var base = try self.asView();
            defer base.deinit();
            return base.broadcastTo(dims);
        }

        pub fn expand(self: Self, dims: []const usize) ArrayError!ArrayView(T) {
            return self.expandView(dims);
        }

        pub fn expandView(self: Self, dims: []const usize) ArrayError!ArrayView(T) {
            return self.broadcastView(dims);
        }

        pub fn expandAs(self: Self, other: Self) ArrayError!ArrayView(T) {
            return self.expandView(other.shape);
        }

        pub fn expandAsView(self: Self, other: ArrayView(T)) ArrayError!ArrayView(T) {
            return self.expandView(other.shape);
        }

        pub fn expand_as(self: Self, other: Self) ArrayError!ArrayView(T) {
            return self.expandAs(other);
        }

        pub fn isScalar(self: Self) bool {
            return self.shape.len == 0 or (self.shape.len == 1 and self.shape[0] == 1);
        }

        fn offsetOf(self: Self, indices: []const usize) ArrayError!usize {
            if (indices.len != self.shape.len) return error.InvalidShape;
            var offset: usize = 0;
            for (indices, self.shape, self.strides) |idx, extent, stride_value| {
                if (idx >= extent) return error.IndexOutOfBounds;
                offset += idx * stride_value;
            }
            return offset;
        }

        fn offsetOfSigned(self: Self, indices: []const isize) ArrayError!usize {
            if (indices.len != self.shape.len) return error.InvalidShape;
            var offset: usize = 0;
            for (indices, self.shape, self.strides) |idx, extent, stride_value| {
                offset += (try normalizeIndex(idx, extent)) * stride_value;
            }
            return offset;
        }

        pub fn get(self: Self, indices: []const usize) ArrayError!T {
            return self.data[try self.offsetOf(indices)];
        }

        pub fn getSigned(self: Self, indices: []const isize) ArrayError!T {
            return self.data[try self.offsetOfSigned(indices)];
        }

        pub fn set(self: *Self, indices: []const usize, value: T) ArrayError!void {
            self.data[try self.offsetOf(indices)] = value;
        }

        pub fn setSigned(self: *Self, indices: []const isize, value: T) ArrayError!void {
            self.data[try self.offsetOfSigned(indices)] = value;
        }

        pub fn at(self: Self, indices: []const usize) ArrayError!T {
            return self.get(indices);
        }

        pub fn atSigned(self: Self, indices: []const isize) ArrayError!T {
            return self.getSigned(indices);
        }

        pub fn put(self: *Self, indices: []const usize, value: T) ArrayError!void {
            return self.set(indices, value);
        }

        pub fn putSigned(self: *Self, indices: []const isize, value: T) ArrayError!void {
            return self.setSigned(indices, value);
        }

        pub fn item(self: Self) ArrayError!T {
            if (!self.isScalar()) return error.ShapeMismatch;
            if (self.data.len == 0) return error.EmptyArray;
            return self.data[0];
        }

        pub fn reshape(self: Self, dims: []const usize) ArrayError!Self {
            const n = try numelFrom(dims);
            if (n != self.data.len) return error.ShapeMismatch;
            var out = try self.clone();
            out.allocator.free(out.shape);
            out.allocator.free(out.strides);
            out.shape = try out.allocator.dupe(usize, dims);
            out.strides = try stridesFor(out.allocator, out.shape);
            return out;
        }

        pub fn reshapeInfer(self: Self, dims: []const isize) ArrayError!Self {
            const inferred = try inferredShape(self.allocator, dims, self.data.len);
            defer self.allocator.free(inferred);
            return self.reshape(inferred);
        }

        pub fn reshapeAs(self: Self, other: Self) ArrayError!Self {
            return self.reshape(other.shape);
        }

        pub fn reshapeAsView(self: Self, other: ArrayView(T)) ArrayError!Self {
            return self.reshape(other.shape);
        }

        pub fn reshape_as(self: Self, other: Self) ArrayError!Self {
            return self.reshapeAs(other);
        }

        pub fn flatten(self: Self) ArrayError!Self {
            return self.reshape(&.{self.data.len});
        }

        pub fn flattenAxes(self: Self, start_axis: isize, end_axis: isize) ArrayError!Self {
            const dims = try flattenShape(self.allocator, self.shape, start_axis, end_axis);
            defer self.allocator.free(dims);
            return self.reshape(dims);
        }

        pub fn flattenRange(self: Self, start_axis: isize, end_axis: isize) ArrayError!Self {
            return self.flattenAxes(start_axis, end_axis);
        }

        pub fn flatten_range(self: Self, start_axis: isize, end_axis: isize) ArrayError!Self {
            return self.flattenAxes(start_axis, end_axis);
        }

        pub fn flattenFrom(self: Self, start_axis: isize) ArrayError!Self {
            return self.flattenAxes(start_axis, -1);
        }

        pub fn flatten_from(self: Self, start_axis: isize) ArrayError!Self {
            return self.flattenFrom(start_axis);
        }

        pub fn ravel(self: Self) ArrayError!Self {
            return self.flatten();
        }

        pub fn atLeast1d(self: Self) ArrayError!Self {
            if (self.shape.len >= 1) return self.clone();
            return self.reshape(&.{1});
        }

        pub fn atLeast2d(self: Self) ArrayError!Self {
            return switch (self.shape.len) {
                0 => self.reshape(&.{ 1, 1 }),
                1 => self.reshape(&.{ 1, self.shape[0] }),
                else => self.clone(),
            };
        }

        pub fn atLeast3d(self: Self) ArrayError!Self {
            return switch (self.shape.len) {
                0 => self.reshape(&.{ 1, 1, 1 }),
                1 => self.reshape(&.{ 1, self.shape[0], 1 }),
                2 => self.reshape(&.{ self.shape[0], self.shape[1], 1 }),
                else => self.clone(),
            };
        }

        pub fn view(self: Self, dims: []const usize) ArrayError!Self {
            return self.reshape(dims);
        }

        pub fn viewInfer(self: Self, dims: []const isize) ArrayError!Self {
            return self.reshapeInfer(dims);
        }

        pub fn viewAs(self: Self, other: Self) ArrayError!Self {
            return self.view(other.shape);
        }

        pub fn viewAsView(self: Self, other: ArrayView(T)) ArrayError!Self {
            return self.view(other.shape);
        }

        pub fn view_as(self: Self, other: Self) ArrayError!Self {
            return self.viewAs(other);
        }

        pub fn unflatten(self: Self, axis_index: isize, dims: []const usize) ArrayError!Self {
            const out_shape = try unflattenShape(self.allocator, self.shape, axis_index, dims);
            defer self.allocator.free(out_shape);
            return self.reshape(out_shape);
        }

        pub fn squeeze(self: Self, axis_opt: ?isize) ArrayError!Self {
            var dims_list: std.ArrayList(usize) = .empty;
            defer dims_list.deinit(self.allocator);
            if (axis_opt) |d| {
                const axis = try normalizeDim(d, self.shape.len);
                for (self.shape, 0..) |size_v, i| {
                    if (i == axis) {
                        if (size_v != 1) try dims_list.append(self.allocator, size_v);
                    } else {
                        try dims_list.append(self.allocator, size_v);
                    }
                }
            } else {
                for (self.shape) |size_v| {
                    if (size_v != 1) try dims_list.append(self.allocator, size_v);
                }
            }
            return self.reshape(dims_list.items);
        }

        pub fn squeezeDim(self: Self, axis_index: isize) ArrayError!Self {
            return self.squeeze(axis_index);
        }

        pub fn squeeze_dim(self: Self, axis_index: isize) ArrayError!Self {
            return self.squeezeDim(axis_index);
        }

        pub fn squeezeAxes(self: Self, axes: []const isize) ArrayError!Self {
            if (axes.len == 0) return self.clone();
            const normalized_axes = try normalizeUniqueAxes(self.allocator, axes, self.shape.len);
            defer self.allocator.free(normalized_axes);
            var squeeze_mask = try self.allocator.alloc(bool, self.shape.len);
            defer self.allocator.free(squeeze_mask);
            @memset(squeeze_mask, false);
            for (normalized_axes) |axis| {
                if (self.shape[axis] != 1) return error.ShapeMismatch;
                squeeze_mask[axis] = true;
            }
            var dims_list: std.ArrayList(usize) = .empty;
            defer dims_list.deinit(self.allocator);
            for (self.shape, 0..) |extent, axis| {
                if (squeeze_mask[axis]) continue;
                try dims_list.append(self.allocator, extent);
            }
            return self.reshape(dims_list.items);
        }

        pub fn squeeze_axes(self: Self, axes: []const isize) ArrayError!Self {
            return self.squeezeAxes(axes);
        }

        pub fn unsqueeze(self: Self, axis_index: isize) ArrayError!Self {
            const rank_count = self.shape.len + 1;
            const axis = if (axis_index < 0) blk: {
                const signed_rank: isize = @intCast(rank_count);
                const normalized = signed_rank + axis_index;
                if (normalized < 0 or normalized >= signed_rank) return error.InvalidAxis;
                break :blk @as(usize, @intCast(normalized));
            } else try canonicalAxis(@intCast(axis_index), rank_count);
            var dims = try self.allocator.alloc(usize, rank_count);
            defer self.allocator.free(dims);
            for (self.shape[0..axis], 0..) |d, i| dims[i] = d;
            dims[axis] = 1;
            for (self.shape[axis..], axis + 1..) |d, i| dims[i] = d;
            return self.reshape(dims);
        }

        pub fn unsqueezeDim(self: Self, axis_index: isize) ArrayError!Self {
            return self.unsqueeze(axis_index);
        }

        pub fn unsqueeze_dim(self: Self, axis_index: isize) ArrayError!Self {
            return self.unsqueezeDim(axis_index);
        }

        pub fn broadcastTo(self: Self, dims: []const usize) ArrayError!Self {
            const out_shape = try broadcastShape(self.allocator, self.shape, dims);
            defer self.allocator.free(out_shape);
            if (!std.mem.eql(usize, out_shape, dims)) return error.ShapeMismatch;
            const out = try Self.empty(self.allocator, dims);
            const out_multi = try self.allocator.alloc(usize, dims.len);
            defer self.allocator.free(out_multi);
            for (out.data, 0..) |*slot, i| {
                unravelIndexInto(i, dims, out_multi);
                slot.* = self.data[broadcastOffset(out_multi, dims.len, self.shape, self.strides)];
            }
            return out;
        }

        fn repeatInterleaveTotal(source_len: usize, repeats: Array(usize)) ArrayError!usize {
            if (repeats.data.len != 1 and repeats.data.len != source_len) return error.ShapeMismatch;
            if (repeats.data.len == 1) {
                return std.math.mul(usize, source_len, repeats.data[0]) catch return error.InvalidShape;
            }
            var total: usize = 0;
            for (repeats.data) |repeat_count| {
                total = std.math.add(usize, total, repeat_count) catch return error.InvalidShape;
            }
            return total;
        }

        fn repeatInterleaveCount(repeats: Array(usize), source_index: usize) usize {
            return if (repeats.data.len == 1) repeats.data[0] else repeats.data[source_index];
        }

        pub fn repeat(self: Self, repeats: usize, axis_index: isize) ArrayError!Self {
            if (self.shape.len == 0) return error.InvalidAxis;
            const axis = try normalizeDim(axis_index, self.shape.len);
            var out_shape = try self.allocator.dupe(usize, self.shape);
            defer self.allocator.free(out_shape);
            out_shape[axis] = std.math.mul(usize, out_shape[axis], repeats) catch return error.InvalidShape;
            var out = try Self.empty(self.allocator, out_shape);
            errdefer out.deinit();
            if (out.data.len == 0) return out;
            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            var in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);
            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, out_shape, out_multi);
                @memcpy(in_multi, out_multi);
                in_multi[axis] = out_multi[axis] / repeats;
                slot.* = self.data[ravelIndex(in_multi, self.strides)];
            }
            return out;
        }

        pub fn repeatInterleave(self: Self, repeats: Array(usize), axis_opt: ?isize) ArrayError!Self {
            if (axis_opt == null) {
                const total = try repeatInterleaveTotal(self.data.len, repeats);
                var out = try Self.empty(self.allocator, &.{total});
                errdefer out.deinit();
                var write: usize = 0;
                for (self.data, 0..) |value, source_index| {
                    const repeat_count = repeatInterleaveCount(repeats, source_index);
                    for (0..repeat_count) |_| {
                        out.data[write] = value;
                        write += 1;
                    }
                }
                return out;
            }

            if (self.shape.len == 0) return error.InvalidAxis;
            const axis = try normalizeDim(axis_opt.?, self.shape.len);
            const total = try repeatInterleaveTotal(self.shape[axis], repeats);
            var out_shape = try self.allocator.dupe(usize, self.shape);
            defer self.allocator.free(out_shape);
            out_shape[axis] = total;
            var out = try Self.empty(self.allocator, out_shape);
            errdefer out.deinit();
            if (out.data.len == 0) return out;

            var slice_shape = try self.allocator.alloc(usize, self.shape.len - 1);
            defer self.allocator.free(slice_shape);
            for (self.shape[0..axis], 0..) |extent, i| slice_shape[i] = extent;
            for (self.shape[axis + 1 ..], axis..) |extent, i| slice_shape[i] = extent;
            const slice_count = try numelFrom(slice_shape);
            const slice_multi = try self.allocator.alloc(usize, slice_shape.len);
            defer self.allocator.free(slice_multi);
            var in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);
            var out_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(out_multi);

            for (0..slice_count) |slice_flat| {
                unravelIndexInto(slice_flat, slice_shape, slice_multi);
                for (slice_multi[0..axis], 0..) |coord, i| {
                    in_multi[i] = coord;
                    out_multi[i] = coord;
                }
                for (slice_multi[axis..], axis + 1..) |coord, i| {
                    in_multi[i] = coord;
                    out_multi[i] = coord;
                }

                var out_axis: usize = 0;
                for (0..self.shape[axis]) |source_axis| {
                    in_multi[axis] = source_axis;
                    const value = self.data[ravelIndex(in_multi, self.strides)];
                    const repeat_count = repeatInterleaveCount(repeats, source_axis);
                    for (0..repeat_count) |_| {
                        out_multi[axis] = out_axis;
                        out.data[ravelIndex(out_multi, out.strides)] = value;
                        out_axis += 1;
                    }
                }
            }
            return out;
        }

        pub fn repeatInterleaveScalar(self: Self, repeat_count: usize, axis_opt: ?isize) ArrayError!Self {
            var repeats = try Array(usize).fromScalar(self.allocator, repeat_count);
            defer repeats.deinit();
            return self.repeatInterleave(repeats, axis_opt);
        }

        pub fn sliceAxis(self: Self, axis_index: isize, slice_value: Slice) ArrayError!Self {
            if (self.shape.len == 0) return error.InvalidAxis;
            const axis = try normalizeDim(axis_index, self.shape.len);
            const ns = try normalizeSlice(slice_value, self.shape[axis]);
            var out_shape = try self.allocator.dupe(usize, self.shape);
            defer self.allocator.free(out_shape);
            out_shape[axis] = ns.count;
            const out = try Self.empty(self.allocator, out_shape);
            if (out.data.len == 0) return out;
            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            var in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);
            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, out_shape, out_multi);
                @memcpy(in_multi, out_multi);
                in_multi[axis] = ns.start + out_multi[axis] * ns.step;
                slot.* = self.data[ravelIndex(in_multi, self.strides)];
            }
            return out;
        }

        pub fn slice(self: Self, slices: []const Slice) ArrayError!Self {
            if (slices.len != self.shape.len) return error.ShapeMismatch;
            var current = try self.clone();
            errdefer current.deinit();
            for (slices, 0..) |slice_value, axis| {
                const next = try current.sliceAxis(@intCast(axis), slice_value);
                current.deinit();
                current = next;
            }
            return current;
        }

        pub fn flip(self: Self, axis_index: isize) ArrayError!Self {
            if (self.shape.len == 0) return error.InvalidAxis;
            const axis = try normalizeDim(axis_index, self.shape.len);
            var out = try Self.empty(self.allocator, self.shape);
            errdefer out.deinit();
            if (out.data.len == 0) return out;
            const out_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(out_multi);
            var in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);
            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, self.shape, out_multi);
                @memcpy(in_multi, out_multi);
                in_multi[axis] = self.shape[axis] - 1 - out_multi[axis];
                slot.* = self.data[ravelIndex(in_multi, self.strides)];
            }
            return out;
        }

        pub fn flipAxes(self: Self, axes: []const isize) ArrayError!Self {
            if (axes.len == 0) return self.clone();
            const normalized_axes = try self.allocator.alloc(usize, axes.len);
            defer self.allocator.free(normalized_axes);
            var seen = try self.allocator.alloc(bool, self.shape.len);
            defer self.allocator.free(seen);
            @memset(seen, false);
            for (axes, 0..) |axis_index, i| {
                const axis = try normalizeDim(axis_index, self.shape.len);
                if (seen[axis]) return error.InvalidAxis;
                seen[axis] = true;
                normalized_axes[i] = axis;
            }
            var out = try Self.empty(self.allocator, self.shape);
            errdefer out.deinit();
            if (out.data.len == 0) return out;
            const out_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(out_multi);
            var in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);
            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, self.shape, out_multi);
                @memcpy(in_multi, out_multi);
                for (normalized_axes) |axis| in_multi[axis] = self.shape[axis] - 1 - out_multi[axis];
                slot.* = self.data[ravelIndex(in_multi, self.strides)];
            }
            return out;
        }

        pub fn roll(self: Self, shift: isize, axis_index: isize) ArrayError!Self {
            if (self.shape.len == 0) return error.InvalidAxis;
            const axis = try normalizeDim(axis_index, self.shape.len);
            const len_axis = self.shape[axis];
            if (len_axis == 0) return self.clone();
            const signed_len: isize = @intCast(len_axis);
            const normalized_shift: usize = @intCast(@mod(shift, signed_len));
            var out = try Self.empty(self.allocator, self.shape);
            errdefer out.deinit();
            const out_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(out_multi);
            var in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);
            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, self.shape, out_multi);
                @memcpy(in_multi, out_multi);
                in_multi[axis] = (out_multi[axis] + len_axis - normalized_shift) % len_axis;
                slot.* = self.data[ravelIndex(in_multi, self.strides)];
            }
            return out;
        }

        pub fn rollAxes(self: Self, shifts: []const isize, axes: []const isize) ArrayError!Self {
            if (shifts.len != axes.len) return error.ShapeMismatch;
            if (shifts.len == 0) return self.clone();
            if (self.shape.len == 0) return error.InvalidAxis;
            const normalized_axes = try self.allocator.alloc(usize, axes.len);
            defer self.allocator.free(normalized_axes);
            const normalized_shifts = try self.allocator.alloc(usize, axes.len);
            defer self.allocator.free(normalized_shifts);
            var seen = try self.allocator.alloc(bool, self.shape.len);
            defer self.allocator.free(seen);
            @memset(seen, false);
            for (axes, 0..) |axis_index, i| {
                const axis = try normalizeDim(axis_index, self.shape.len);
                if (seen[axis]) return error.InvalidAxis;
                seen[axis] = true;
                normalized_axes[i] = axis;
                const len_axis = self.shape[axis];
                normalized_shifts[i] = if (len_axis == 0) 0 else @intCast(@mod(shifts[i], @as(isize, @intCast(len_axis))));
            }

            var out = try Self.empty(self.allocator, self.shape);
            errdefer out.deinit();
            if (out.data.len == 0) return out;
            const out_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(out_multi);
            var in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);
            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, self.shape, out_multi);
                @memcpy(in_multi, out_multi);
                for (normalized_axes, normalized_shifts) |axis, normalized_shift| {
                    const len_axis = self.shape[axis];
                    in_multi[axis] = if (len_axis == 0) 0 else (out_multi[axis] + len_axis - normalized_shift) % len_axis;
                }
                slot.* = self.data[ravelIndex(in_multi, self.strides)];
            }
            return out;
        }

        pub fn rot90(self: Self, k: isize, axes_pair: [2]isize) ArrayError!Self {
            if (self.shape.len < 2) return error.InvalidAxis;
            const axis0 = try normalizeDim(axes_pair[0], self.shape.len);
            const axis1 = try normalizeDim(axes_pair[1], self.shape.len);
            if (axis0 == axis1) return error.InvalidAxis;
            const turns: usize = @intCast(@mod(k, 4));
            if (turns == 0) return self.clone();

            var out_shape = try self.allocator.dupe(usize, self.shape);
            defer self.allocator.free(out_shape);
            if (turns == 1 or turns == 3) {
                out_shape[axis0] = self.shape[axis1];
                out_shape[axis1] = self.shape[axis0];
            }

            var out = try Self.empty(self.allocator, out_shape);
            errdefer out.deinit();
            if (out.data.len == 0) return out;

            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            var in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);

            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, out_shape, out_multi);
                @memcpy(in_multi, out_multi);
                switch (turns) {
                    1 => {
                        in_multi[axis0] = out_multi[axis1];
                        in_multi[axis1] = self.shape[axis1] - 1 - out_multi[axis0];
                    },
                    2 => {
                        in_multi[axis0] = self.shape[axis0] - 1 - out_multi[axis0];
                        in_multi[axis1] = self.shape[axis1] - 1 - out_multi[axis1];
                    },
                    3 => {
                        in_multi[axis0] = self.shape[axis0] - 1 - out_multi[axis1];
                        in_multi[axis1] = out_multi[axis0];
                    },
                    else => unreachable,
                }
                slot.* = self.data[ravelIndex(in_multi, self.strides)];
            }
            return out;
        }

        pub fn padConstant(self: Self, before: []const usize, after: []const usize, value: T) ArrayError!Self {
            if (before.len != self.shape.len or after.len != self.shape.len) return error.ShapeMismatch;
            var out_shape = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(out_shape);
            for (self.shape, before, after, 0..) |d, b, a, i| out_shape[i] = d + b + a;
            var out = try Self.full(self.allocator, out_shape, value);
            errdefer out.deinit();
            if (self.data.len == 0) return out;
            const in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);
            var out_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(out_multi);
            for (self.data, 0..) |v, flat| {
                unravelIndexInto(flat, self.shape, in_multi);
                for (in_multi, before, 0..) |coord, b, i| out_multi[i] = coord + b;
                out.data[ravelIndex(out_multi, out.strides)] = v;
            }
            return out;
        }

        fn edgePadCoord(out_coord: usize, before: usize, extent: usize) usize {
            if (out_coord < before) return 0;
            const shifted = out_coord - before;
            return if (shifted >= extent) extent - 1 else shifted;
        }

        fn reflectPadCoord(out_coord: usize, before: usize, extent: usize) usize {
            const period: isize = @intCast(2 * extent - 2);
            var pos: isize = @as(isize, @intCast(out_coord)) - @as(isize, @intCast(before));
            pos = @mod(pos, period);
            const normalized: usize = @intCast(pos);
            return if (normalized < extent) normalized else (2 * extent - 2) - normalized;
        }

        fn wrapPadCoord(out_coord: usize, before: usize, extent: usize) usize {
            const pos: isize = @as(isize, @intCast(out_coord)) - @as(isize, @intCast(before));
            return @intCast(@mod(pos, @as(isize, @intCast(extent))));
        }

        fn symmetricPadCoord(out_coord: usize, before: usize, extent: usize) usize {
            const period: isize = @intCast(2 * extent);
            var pos: isize = @as(isize, @intCast(out_coord)) - @as(isize, @intCast(before));
            pos = @mod(pos, period);
            const normalized: usize = @intCast(pos);
            return if (normalized < extent) normalized else (2 * extent - 1) - normalized;
        }

        pub fn padEdge(self: Self, before: []const usize, after: []const usize) ArrayError!Self {
            if (before.len != self.shape.len or after.len != self.shape.len) return error.ShapeMismatch;
            if (self.data.len == 0) return error.EmptyArray;
            var out_shape = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(out_shape);
            for (self.shape, before, after, 0..) |extent, before_i, after_i, axis| out_shape[axis] = extent + before_i + after_i;
            var out = try Self.empty(self.allocator, out_shape);
            errdefer out.deinit();
            if (out.data.len == 0) return out;
            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            var in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);
            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, out_shape, out_multi);
                for (out_multi, before, self.shape, 0..) |coord, before_i, extent, axis| {
                    in_multi[axis] = edgePadCoord(coord, before_i, extent);
                }
                slot.* = self.data[ravelIndex(in_multi, self.strides)];
            }
            return out;
        }

        pub fn padReflect(self: Self, before: []const usize, after: []const usize) ArrayError!Self {
            if (before.len != self.shape.len or after.len != self.shape.len) return error.ShapeMismatch;
            if (self.data.len == 0) return error.EmptyArray;
            var out_shape = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(out_shape);
            for (self.shape, before, after, 0..) |extent, before_i, after_i, axis| {
                if (extent < 2 and (before_i != 0 or after_i != 0)) return error.InvalidShape;
                if (extent >= 2 and (before_i >= extent or after_i >= extent)) return error.InvalidShape;
                out_shape[axis] = extent + before_i + after_i;
            }
            var out = try Self.empty(self.allocator, out_shape);
            errdefer out.deinit();
            if (out.data.len == 0) return out;
            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            var in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);
            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, out_shape, out_multi);
                for (out_multi, before, self.shape, 0..) |coord, before_i, extent, axis| {
                    in_multi[axis] = if (extent == 1) 0 else reflectPadCoord(coord, before_i, extent);
                }
                slot.* = self.data[ravelIndex(in_multi, self.strides)];
            }
            return out;
        }

        pub fn padWrap(self: Self, before: []const usize, after: []const usize) ArrayError!Self {
            if (before.len != self.shape.len or after.len != self.shape.len) return error.ShapeMismatch;
            if (self.data.len == 0) return error.EmptyArray;
            var out_shape = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(out_shape);
            for (self.shape, before, after, 0..) |extent, before_i, after_i, axis| out_shape[axis] = extent + before_i + after_i;
            var out = try Self.empty(self.allocator, out_shape);
            errdefer out.deinit();
            if (out.data.len == 0) return out;
            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            var in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);
            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, out_shape, out_multi);
                for (out_multi, before, self.shape, 0..) |coord, before_i, extent, axis| {
                    in_multi[axis] = wrapPadCoord(coord, before_i, extent);
                }
                slot.* = self.data[ravelIndex(in_multi, self.strides)];
            }
            return out;
        }

        pub fn padSymmetric(self: Self, before: []const usize, after: []const usize) ArrayError!Self {
            if (before.len != self.shape.len or after.len != self.shape.len) return error.ShapeMismatch;
            if (self.data.len == 0) return error.EmptyArray;
            var out_shape = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(out_shape);
            for (self.shape, before, after, 0..) |extent, before_i, after_i, axis| out_shape[axis] = extent + before_i + after_i;
            var out = try Self.empty(self.allocator, out_shape);
            errdefer out.deinit();
            if (out.data.len == 0) return out;
            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            var in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);
            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, out_shape, out_multi);
                for (out_multi, before, self.shape, 0..) |coord, before_i, extent, axis| {
                    in_multi[axis] = symmetricPadCoord(coord, before_i, extent);
                }
                slot.* = self.data[ravelIndex(in_multi, self.strides)];
            }
            return out;
        }

        pub fn tile(self: Self, repeats: []const usize) ArrayError!Self {
            const out_rank = @max(self.shape.len, repeats.len);
            const out_shape = try self.allocator.alloc(usize, out_rank);
            defer self.allocator.free(out_shape);
            const shape_leading = out_rank - self.shape.len;
            const repeats_leading = out_rank - repeats.len;
            for (out_shape, 0..) |*slot, axis| {
                const extent = if (axis < shape_leading) 1 else self.shape[axis - shape_leading];
                const repeat_count = if (axis < repeats_leading) 1 else repeats[axis - repeats_leading];
                slot.* = std.math.mul(usize, extent, repeat_count) catch return error.InvalidShape;
            }
            const out = try Self.empty(self.allocator, out_shape);
            if (out.data.len == 0) return out;
            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            const in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);
            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, out_shape, out_multi);
                for (in_multi, 0..) |*coord_slot, axis| {
                    const out_axis = shape_leading + axis;
                    const extent = self.shape[axis];
                    coord_slot.* = if (extent == 0) 0 else out_multi[out_axis] % extent;
                }
                slot.* = self.data[ravelIndex(in_multi, self.strides)];
            }
            return out;
        }

        pub fn transpose(self: Self) ArrayError!Self {
            if (self.shape.len != 2) return error.NonMatrixArray;
            const rows = self.shape[0];
            const cols = self.shape[1];
            var out = try Self.empty(self.allocator, &.{ cols, rows });
            for (0..rows) |r| {
                for (0..cols) |c| {
                    out.data[c * rows + r] = self.data[r * cols + c];
                }
            }
            return out;
        }

        pub fn T_(self: Self) ArrayError!Self {
            return self.transpose();
        }

        pub fn swapaxes(self: Self, dim0: isize, dim1: isize) ArrayError!Self {
            const a0 = try normalizeDim(dim0, self.shape.len);
            const a1 = try normalizeDim(dim1, self.shape.len);
            var perm = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(perm);
            for (perm, 0..) |*slot, i| slot.* = i;
            std.mem.swap(usize, &perm[a0], &perm[a1]);
            return self.permute(perm);
        }

        pub fn swapDims(self: Self, dim0: isize, dim1: isize) ArrayError!Self {
            return self.swapaxes(dim0, dim1);
        }

        pub fn swap_dims(self: Self, dim0: isize, dim1: isize) ArrayError!Self {
            return self.swapDims(dim0, dim1);
        }

        pub fn permute(self: Self, axes: []const usize) ArrayError!Self {
            if (axes.len != self.shape.len) return error.InvalidPermutation;
            var seen = try self.allocator.alloc(bool, axes.len);
            defer self.allocator.free(seen);
            @memset(seen, false);
            var out_shape = try self.allocator.alloc(usize, axes.len);
            defer self.allocator.free(out_shape);
            for (axes, 0..) |axis, i| {
                if (axis >= axes.len or seen[axis]) return error.InvalidPermutation;
                seen[axis] = true;
                out_shape[i] = self.shape[axis];
            }
            const out = try Self.empty(self.allocator, out_shape);
            if (out.data.len == 0) return out;
            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            var in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);
            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, out_shape, out_multi);
                for (axes, 0..) |axis, i| in_multi[axis] = out_multi[i];
                slot.* = self.data[ravelIndex(in_multi, self.strides)];
            }
            return out;
        }

        pub fn movedim(self: Self, source: isize, destination: isize) ArrayError!Self {
            const src = try normalizeDim(source, self.shape.len);
            const dst = try normalizeDim(destination, self.shape.len);
            const axes = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(axes);
            var remaining = try self.allocator.alloc(usize, self.shape.len - 1);
            defer self.allocator.free(remaining);
            var remaining_len: usize = 0;
            for (0..self.shape.len) |i| {
                if (i == src) continue;
                remaining[remaining_len] = i;
                remaining_len += 1;
            }
            var read: usize = 0;
            for (axes, 0..) |*slot, out_i| {
                if (out_i == dst) {
                    slot.* = src;
                } else {
                    slot.* = remaining[read];
                    read += 1;
                }
            }
            return self.permute(axes);
        }

        pub fn moveaxis(self: Self, source: isize, destination: isize) ArrayError!Self {
            return self.movedim(source, destination);
        }

        pub fn moveaxes(self: Self, sources: []const isize, destinations: []const isize) ArrayError!Self {
            const axes = try movedimManyAxes(self.allocator, self.shape.len, sources, destinations);
            defer self.allocator.free(axes);
            return self.permute(axes);
        }

        pub fn move_axes(self: Self, sources: []const isize, destinations: []const isize) ArrayError!Self {
            return self.moveaxes(sources, destinations);
        }

        pub fn slice1d(self: Self, slice_value: Slice) ArrayError!Self {
            if (self.shape.len != 1) return error.NonVectorArray;
            const ns = try normalizeSlice(slice_value, self.shape[0]);
            const out = try Self.empty(self.allocator, &.{ns.count});
            var idx = ns.start;
            for (out.data) |*slot| {
                slot.* = self.data[idx];
                idx += ns.step;
            }
            return out;
        }

        pub fn select(self: Self, axis_index: isize, index: usize) ArrayError!Self {
            const axis = try normalizeDim(axis_index, self.shape.len);
            if (index >= self.shape[axis]) return error.IndexOutOfBounds;
            if (self.shape.len == 0) return error.InvalidAxis;

            var out_shape = try self.allocator.alloc(usize, self.shape.len - 1);
            defer self.allocator.free(out_shape);
            for (self.shape[0..axis], 0..) |d, i| out_shape[i] = d;
            for (self.shape[axis + 1 ..], axis..) |d, i| out_shape[i] = d;

            const out = try Self.empty(self.allocator, out_shape);
            if (out.data.len == 0) return out;
            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            var in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);

            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, out_shape, out_multi);
                for (out_multi[0..axis], 0..) |coord, i| in_multi[i] = coord;
                in_multi[axis] = index;
                for (out_multi[axis..], axis + 1..) |coord, i| in_multi[i] = coord;
                slot.* = self.data[ravelIndex(in_multi, self.strides)];
            }
            return out;
        }

        pub fn selectSigned(self: Self, axis_index: isize, index: isize) ArrayError!Self {
            const axis = try normalizeDim(axis_index, self.shape.len);
            return self.select(axis_index, try normalizeIndex(index, self.shape[axis]));
        }

        pub fn narrow(self: Self, axis_index: isize, start: usize, length: usize) ArrayError!Self {
            const axis = try normalizeDim(axis_index, self.shape.len);
            if (start > self.shape[axis] or start + length > self.shape[axis]) return error.IndexOutOfBounds;
            var out_shape = try self.allocator.dupe(usize, self.shape);
            defer self.allocator.free(out_shape);
            out_shape[axis] = length;
            const out = try Self.empty(self.allocator, out_shape);
            if (out.data.len == 0) return out;
            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            var in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);
            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, out_shape, out_multi);
                @memcpy(in_multi, out_multi);
                in_multi[axis] = start + out_multi[axis];
                slot.* = self.data[ravelIndex(in_multi, self.strides)];
            }
            return out;
        }

        pub const SplitResult = struct {
            allocator: std.mem.Allocator,
            items: []Self,

            pub fn deinit(self: *@This()) void {
                for (self.items) |*part| part.deinit();
                self.allocator.free(self.items);
                self.* = undefined;
            }
        };

        pub fn split(self: Self, split_size: usize, axis_index: isize) ArrayError!SplitResult {
            if (split_size == 0) return error.InvalidShape;
            const axis = try normalizeDim(axis_index, self.shape.len);
            const axis_len = self.shape[axis];
            const part_count = if (axis_len == 0) 0 else (axis_len + split_size - 1) / split_size;
            const items = try self.allocator.alloc(Self, part_count);
            errdefer self.allocator.free(items);
            var initialized: usize = 0;
            errdefer {
                for (items[0..initialized]) |*part| part.deinit();
            }
            var start: usize = 0;
            while (start < axis_len) : (start += split_size) {
                const len_part = @min(split_size, axis_len - start);
                items[initialized] = try self.narrow(axis_index, start, len_part);
                initialized += 1;
            }
            return .{ .allocator = self.allocator, .items = items };
        }

        pub fn splitWithSizes(self: Self, sizes: []const usize, axis_index: isize) ArrayError!SplitResult {
            const axis = try normalizeDim(axis_index, self.shape.len);
            const axis_len = self.shape[axis];
            var total: usize = 0;
            for (sizes) |part_len| {
                total = std.math.add(usize, total, part_len) catch return error.InvalidShape;
            }
            if (total != axis_len) return error.ShapeMismatch;

            const items = try self.allocator.alloc(Self, sizes.len);
            errdefer self.allocator.free(items);
            var initialized: usize = 0;
            errdefer {
                for (items[0..initialized]) |*part| part.deinit();
            }
            var start: usize = 0;
            for (sizes, 0..) |part_len, i| {
                items[i] = try self.narrow(axis_index, start, part_len);
                initialized += 1;
                start += part_len;
            }
            return .{ .allocator = self.allocator, .items = items };
        }

        pub fn split_with_sizes(self: Self, sizes: []const usize, axis_index: isize) ArrayError!SplitResult {
            return self.splitWithSizes(sizes, axis_index);
        }

        pub fn splitAtIndices(self: Self, indices: []const usize, axis_index: isize) ArrayError!SplitResult {
            const axis = try normalizeDim(axis_index, self.shape.len);
            const axis_len = self.shape[axis];
            const items = try self.allocator.alloc(Self, indices.len + 1);
            errdefer self.allocator.free(items);
            var initialized: usize = 0;
            errdefer {
                for (items[0..initialized]) |*part| part.deinit();
            }
            var start: usize = 0;
            for (indices, 0..) |stop, i| {
                if (stop < start or stop > axis_len) return error.InvalidShape;
                items[i] = try self.narrow(axis_index, start, stop - start);
                initialized += 1;
                start = stop;
            }
            items[indices.len] = try self.narrow(axis_index, start, axis_len - start);
            return .{ .allocator = self.allocator, .items = items };
        }

        pub fn split_at_indices(self: Self, indices: []const usize, axis_index: isize) ArrayError!SplitResult {
            return self.splitAtIndices(indices, axis_index);
        }

        pub fn chunk(self: Self, chunks: usize, axis_index: isize) ArrayError!SplitResult {
            if (chunks == 0) return error.InvalidShape;
            const axis = try normalizeDim(axis_index, self.shape.len);
            const axis_len = self.shape[axis];
            if (axis_len == 0) {
                const items = try self.allocator.alloc(Self, 0);
                return .{ .allocator = self.allocator, .items = items };
            }
            const split_size = (axis_len + chunks - 1) / chunks;
            return self.split(split_size, axis_index);
        }

        pub fn unbind(self: Self, axis_index: isize) ArrayError!SplitResult {
            const axis = try normalizeDim(axis_index, self.shape.len);
            const axis_len = self.shape[axis];
            const items = try self.allocator.alloc(Self, axis_len);
            errdefer self.allocator.free(items);
            var initialized: usize = 0;
            errdefer {
                for (items[0..initialized]) |*part| part.deinit();
            }
            for (items, 0..) |*part, index| {
                part.* = try self.select(@intCast(axis), index);
                initialized += 1;
            }
            return .{ .allocator = self.allocator, .items = items };
        }

        pub fn take(self: Self, indices: Array(usize), axis_opt: ?isize) ArrayError!Self {
            if (axis_opt == null) {
                const out = try Self.empty(self.allocator, indices.shape);
                for (indices.data, out.data) |idx, *slot| {
                    if (idx >= self.data.len) return error.IndexOutOfBounds;
                    slot.* = self.data[idx];
                }
                return out;
            }
            const axis = try normalizeDim(axis_opt.?, self.shape.len);
            var out_shape = try self.allocator.dupe(usize, self.shape);
            defer self.allocator.free(out_shape);
            out_shape[axis] = indices.data.len;
            const out = try Self.empty(self.allocator, out_shape);
            if (out.data.len == 0) return out;
            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            var in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);
            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, out_shape, out_multi);
                @memcpy(in_multi, out_multi);
                const idx = indices.data[out_multi[axis]];
                if (idx >= self.shape[axis]) return error.IndexOutOfBounds;
                in_multi[axis] = idx;
                slot.* = self.data[ravelIndex(in_multi, self.strides)];
            }
            return out;
        }

        pub fn takeSigned(self: Self, indices: Array(isize), axis_opt: ?isize) ArrayError!Self {
            if (axis_opt == null) {
                var out = try Self.empty(self.allocator, indices.shape);
                errdefer out.deinit();
                for (indices.data, out.data) |idx, *slot| {
                    slot.* = self.data[try normalizeIndex(idx, self.data.len)];
                }
                return out;
            }
            const axis = try normalizeDim(axis_opt.?, self.shape.len);
            var out_shape = try self.allocator.dupe(usize, self.shape);
            defer self.allocator.free(out_shape);
            out_shape[axis] = indices.data.len;
            const out = try Self.empty(self.allocator, out_shape);
            if (out.data.len == 0) return out;
            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            var in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);
            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, out_shape, out_multi);
                @memcpy(in_multi, out_multi);
                in_multi[axis] = try normalizeIndex(indices.data[out_multi[axis]], self.shape[axis]);
                slot.* = self.data[ravelIndex(in_multi, self.strides)];
            }
            return out;
        }

        fn applyIndexMode(idx: usize, extent: usize, mode: IndexMode) ArrayError!usize {
            if (extent == 0) return error.IndexOutOfBounds;
            return switch (mode) {
                .raise => if (idx >= extent) error.IndexOutOfBounds else idx,
                .wrap => idx % extent,
                .clip => @min(idx, extent - 1),
            };
        }

        pub fn takeMode(self: Self, indices: Array(usize), axis_opt: ?isize, mode: IndexMode) ArrayError!Self {
            if (axis_opt == null) {
                const out = try Self.empty(self.allocator, indices.shape);
                for (indices.data, out.data) |idx, *slot| {
                    slot.* = self.data[try applyIndexMode(idx, self.data.len, mode)];
                }
                return out;
            }
            const axis = try normalizeDim(axis_opt.?, self.shape.len);
            var out_shape = try self.allocator.dupe(usize, self.shape);
            defer self.allocator.free(out_shape);
            out_shape[axis] = indices.data.len;
            const out = try Self.empty(self.allocator, out_shape);
            if (out.data.len == 0) return out;
            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            var in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);
            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, out_shape, out_multi);
                @memcpy(in_multi, out_multi);
                in_multi[axis] = try applyIndexMode(indices.data[out_multi[axis]], self.shape[axis], mode);
                slot.* = self.data[ravelIndex(in_multi, self.strides)];
            }
            return out;
        }

        pub fn indexSelect(self: Self, axis_index: isize, indices: Array(usize)) ArrayError!Self {
            return self.take(indices, axis_index);
        }

        pub fn indexSelectSigned(self: Self, axis_index: isize, indices: Array(isize)) ArrayError!Self {
            return self.takeSigned(indices, axis_index);
        }

        pub fn takeAlongAxis(self: Self, indices: Array(usize), axis_index: isize) ArrayError!Self {
            return self.gather(axis_index, indices);
        }

        pub fn takeAlongAxisSigned(self: Self, indices: Array(isize), axis_index: isize) ArrayError!Self {
            return self.gatherSigned(axis_index, indices);
        }

        pub fn putAlongAxis(self: Self, indices: Array(usize), src: Self, axis_index: isize) ArrayError!Self {
            return self.scatter(axis_index, indices, src);
        }

        pub fn maskedSelect(self: Self, mask: Array(bool)) ArrayError!Self {
            const out_shape = try broadcastShape(self.allocator, self.shape, mask.shape);
            defer self.allocator.free(out_shape);
            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            var count: usize = 0;
            for (0..product(out_shape)) |i| {
                unravelIndexInto(i, out_shape, out_multi);
                const mi = broadcastOffset(out_multi, out_shape.len, mask.shape, mask.strides);
                if (mask.data[mi]) count += 1;
            }
            const out = try Self.empty(self.allocator, &.{count});
            var write: usize = 0;
            for (0..product(out_shape)) |i| {
                unravelIndexInto(i, out_shape, out_multi);
                const mi = broadcastOffset(out_multi, out_shape.len, mask.shape, mask.strides);
                if (mask.data[mi]) {
                    const si = broadcastOffset(out_multi, out_shape.len, self.shape, self.strides);
                    out.data[write] = self.data[si];
                    write += 1;
                }
            }
            return out;
        }

        pub fn maskedFill(self: Self, mask: Array(bool), value: T) ArrayError!Self {
            const out_shape = try broadcastShape(self.allocator, self.shape, mask.shape);
            defer self.allocator.free(out_shape);
            var out = try self.broadcastTo(out_shape);
            errdefer out.deinit();
            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            for (out.data, 0..) |*slot, i| {
                unravelIndexInto(i, out_shape, out_multi);
                const mi = broadcastOffset(out_multi, out_shape.len, mask.shape, mask.strides);
                if (mask.data[mi]) slot.* = value;
            }
            return out;
        }

        pub fn maskedScatter(self: Self, mask: Array(bool), src: Self) ArrayError!Self {
            const out_shape = try broadcastShape(self.allocator, self.shape, mask.shape);
            defer self.allocator.free(out_shape);
            var out = try self.broadcastTo(out_shape);
            errdefer out.deinit();
            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            var write: usize = 0;
            for (out.data, 0..) |*slot, i| {
                unravelIndexInto(i, out_shape, out_multi);
                const mi = broadcastOffset(out_multi, out_shape.len, mask.shape, mask.strides);
                if (mask.data[mi]) {
                    if (write >= src.data.len) return error.ShapeMismatch;
                    slot.* = src.data[write];
                    write += 1;
                }
            }
            if (write != src.data.len) return error.ShapeMismatch;
            return out;
        }

        pub fn maskedPut(self: Self, mask: Array(bool), values: Self) ArrayError!Self {
            const out_shape = try broadcastShape(self.allocator, self.shape, mask.shape);
            defer self.allocator.free(out_shape);
            var out = try self.broadcastTo(out_shape);
            errdefer out.deinit();
            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            var count: usize = 0;
            for (0..product(out_shape)) |i| {
                unravelIndexInto(i, out_shape, out_multi);
                const mi = broadcastOffset(out_multi, out_shape.len, mask.shape, mask.strides);
                if (mask.data[mi]) count += 1;
            }
            if (values.data.len != 1 and values.data.len != count) return error.ShapeMismatch;
            var write: usize = 0;
            for (out.data, 0..) |*slot, i| {
                unravelIndexInto(i, out_shape, out_multi);
                const mi = broadcastOffset(out_multi, out_shape.len, mask.shape, mask.strides);
                if (mask.data[mi]) {
                    slot.* = values.data[if (values.data.len == 1) 0 else write];
                    write += 1;
                }
            }
            return out;
        }

        pub fn putMask(self: Self, mask: Array(bool), values: Self) ArrayError!Self {
            return self.maskedPut(mask, values);
        }

        pub fn maskedPutScalar(self: Self, mask: Array(bool), value: T) ArrayError!Self {
            return self.maskedFill(mask, value);
        }

        pub fn putMaskScalar(self: Self, mask: Array(bool), value: T) ArrayError!Self {
            return self.maskedPutScalar(mask, value);
        }

        pub fn copyWhere(self: Self, mask: Array(bool), src: Self) ArrayError!Self {
            return src.where(mask, self);
        }

        pub fn where(self: Self, mask: Array(bool), other: Self) ArrayError!Self {
            return whereMask(mask, self, other);
        }

        pub fn whereScalar(self: Self, mask: Array(bool), other_value: T) ArrayError!Self {
            const out_shape = try broadcastShape(self.allocator, self.shape, mask.shape);
            defer self.allocator.free(out_shape);
            var out = try Self.empty(self.allocator, out_shape);
            errdefer out.deinit();
            if (out.data.len == 0) return out;
            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            for (out.data, 0..) |*slot, i| {
                unravelIndexInto(i, out_shape, out_multi);
                const mi = broadcastOffset(out_multi, out_shape.len, mask.shape, mask.strides);
                const si = broadcastOffset(out_multi, out_shape.len, self.shape, self.strides);
                slot.* = if (mask.data[mi]) self.data[si] else other_value;
            }
            return out;
        }

        pub fn putFlat(self: Self, indices: Array(usize), values: Self) ArrayError!Self {
            if (values.data.len != 1 and values.data.len != indices.data.len) return error.ShapeMismatch;
            var out = try self.clone();
            errdefer out.deinit();
            for (indices.data, 0..) |idx, i| {
                if (idx >= out.data.len) return error.IndexOutOfBounds;
                out.data[idx] = values.data[if (values.data.len == 1) 0 else i];
            }
            return out;
        }

        pub fn putFlatSigned(self: Self, indices: Array(isize), values: Self) ArrayError!Self {
            if (values.data.len != 1 and values.data.len != indices.data.len) return error.ShapeMismatch;
            var out = try self.clone();
            errdefer out.deinit();
            for (indices.data, 0..) |idx, i| {
                out.data[try normalizeIndex(idx, out.data.len)] = values.data[if (values.data.len == 1) 0 else i];
            }
            return out;
        }

        pub fn putFlatMode(self: Self, indices: Array(usize), values: Self, mode: IndexMode) ArrayError!Self {
            if (values.data.len != 1 and values.data.len != indices.data.len) return error.ShapeMismatch;
            var out = try self.clone();
            errdefer out.deinit();
            for (indices.data, 0..) |idx, i| {
                out.data[try applyIndexMode(idx, out.data.len, mode)] = values.data[if (values.data.len == 1) 0 else i];
            }
            return out;
        }

        pub fn putFlatScalar(self: Self, indices: Array(usize), value: T) ArrayError!Self {
            var out = try self.clone();
            errdefer out.deinit();
            for (indices.data) |idx| {
                if (idx >= out.data.len) return error.IndexOutOfBounds;
                out.data[idx] = value;
            }
            return out;
        }

        pub fn putFlatScalarSigned(self: Self, indices: Array(isize), value: T) ArrayError!Self {
            var out = try self.clone();
            errdefer out.deinit();
            for (indices.data) |idx| {
                out.data[try normalizeIndex(idx, out.data.len)] = value;
            }
            return out;
        }

        pub fn putFlatScalarMode(self: Self, indices: Array(usize), value: T, mode: IndexMode) ArrayError!Self {
            var out = try self.clone();
            errdefer out.deinit();
            for (indices.data) |idx| {
                out.data[try applyIndexMode(idx, out.data.len, mode)] = value;
            }
            return out;
        }

        pub fn indexPut(self: Self, indices: Array(usize), values: Self) ArrayError!Self {
            return self.putFlat(indices, values);
        }

        pub fn indexPutScalar(self: Self, indices: Array(usize), value: T) ArrayError!Self {
            return self.putFlatScalar(indices, value);
        }

        pub fn countNonzero(self: Self) usize {
            var count: usize = 0;
            for (self.data) |v| {
                if (v != zero(T)) count += 1;
            }
            return count;
        }

        pub fn count_nonzero(self: Self) usize {
            return self.countNonzero();
        }

        pub fn countNonzeroAxis(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Array(usize) {
            if (axis_opt == null) {
                const count = self.countNonzero();
                if (keepdims) {
                    const out_shape = try keepDimsAllOnes(self.allocator, self.shape.len);
                    defer self.allocator.free(out_shape);
                    return Array(usize).fromSlice(self.allocator, &.{count}, out_shape);
                }
                return Array(usize).fromSlice(self.allocator, &.{count}, &.{});
            }

            const axis = try normalizeDim(axis_opt.?, self.shape.len);
            const out_shape = try self.reducedShape(axis, keepdims);
            defer self.allocator.free(out_shape);
            var out = try Array(usize).zeros(self.allocator, out_shape);
            errdefer out.deinit();
            if (self.data.len == 0) return out;

            const in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);
            var out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            for (self.data, 0..) |value, flat| {
                if (value == zero(T)) continue;
                unravelIndexInto(flat, self.shape, in_multi);
                if (keepdims) {
                    @memcpy(out_multi, in_multi);
                    out_multi[axis] = 0;
                } else {
                    for (in_multi[0..axis], 0..) |coord, i| out_multi[i] = coord;
                    for (in_multi[axis + 1 ..], axis..) |coord, i| out_multi[i] = coord;
                }
                out.data[ravelIndex(out_multi, out.strides)] += 1;
            }
            return out;
        }

        pub fn flatNonzero(self: Self) ArrayError!Array(usize) {
            const count = self.countNonzero();
            const out = try Array(usize).empty(self.allocator, &.{count});
            var write: usize = 0;
            for (self.data, 0..) |value, flat| {
                if (value == zero(T)) continue;
                out.data[write] = flat;
                write += 1;
            }
            return out;
        }

        pub fn nonzero(self: Self) ArrayError!Array(usize) {
            const count = self.countNonzero();
            const out = try Array(usize).empty(self.allocator, &.{ count, self.shape.len });
            if (count == 0 or self.shape.len == 0) return out;
            const multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(multi);
            var write: usize = 0;
            for (self.data, 0..) |v, flat| {
                if (v == zero(T)) continue;
                unravelIndexInto(flat, self.shape, multi);
                @memcpy(out.data[write * self.shape.len ..][0..self.shape.len], multi);
                write += 1;
            }
            return out;
        }

        pub fn argwhere(self: Self) ArrayError!Array(usize) {
            return self.nonzero();
        }

        pub fn whereIndices(self: Self) ArrayError!Array(usize) {
            if (comptime T != bool) @compileError("whereIndices requires Array(bool)");
            return self.nonzero();
        }

        pub fn ravelCoords(self: Self, coords: Array(usize)) ArrayError!Array(usize) {
            if (coords.shape.len == 0 or coords.shape[coords.shape.len - 1] != self.shape.len) return error.ShapeMismatch;
            const out_shape = coords.shape[0 .. coords.shape.len - 1];
            var out = try Array(usize).empty(self.allocator, out_shape);
            errdefer out.deinit();
            for (out.data, 0..) |*slot, row| {
                var offset: usize = 0;
                for (0..self.shape.len) |axis| {
                    const coord = coords.data[row * self.shape.len + axis];
                    if (coord >= self.shape[axis]) return error.IndexOutOfBounds;
                    offset += coord * self.strides[axis];
                }
                slot.* = offset;
            }
            return out;
        }

        pub fn unravelFlat(self: Self, indices: Array(usize)) ArrayError!Array(usize) {
            var out_shape = try self.allocator.alloc(usize, indices.shape.len + 1);
            defer self.allocator.free(out_shape);
            @memcpy(out_shape[0..indices.shape.len], indices.shape);
            out_shape[indices.shape.len] = self.shape.len;
            var out = try Array(usize).empty(self.allocator, out_shape);
            errdefer out.deinit();
            if (self.shape.len == 0) {
                for (indices.data) |idx| {
                    if (idx != 0) return error.IndexOutOfBounds;
                }
                return out;
            }
            const coords = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(coords);
            for (indices.data, 0..) |idx, row| {
                if (idx >= self.data.len) return error.IndexOutOfBounds;
                unravelIndexInto(idx, self.shape, coords);
                @memcpy(out.data[row * self.shape.len ..][0..self.shape.len], coords);
            }
            return out;
        }

        pub fn takeCoords(self: Self, coords: Array(usize)) ArrayError!Self {
            var flat = try self.ravelCoords(coords);
            defer flat.deinit();
            return self.take(flat, null);
        }

        pub fn putCoords(self: Self, coords: Array(usize), values: Self) ArrayError!Self {
            var flat = try self.ravelCoords(coords);
            defer flat.deinit();
            return self.putFlat(flat, values);
        }

        pub fn putCoordsScalar(self: Self, coords: Array(usize), value: T) ArrayError!Self {
            var flat = try self.ravelCoords(coords);
            defer flat.deinit();
            return self.putFlatScalar(flat, value);
        }

        fn multiIndexShape(self: Self, indices: []const Array(usize)) ArrayError![]usize {
            if (indices.len != self.shape.len) return error.ShapeMismatch;
            var out_shape = try self.allocator.dupe(usize, indices[0].shape);
            errdefer self.allocator.free(out_shape);
            for (indices[1..]) |idx_array| {
                const next_shape = try broadcastShape(self.allocator, out_shape, idx_array.shape);
                self.allocator.free(out_shape);
                out_shape = next_shape;
            }
            return out_shape;
        }

        pub fn ravelMultiIndex(self: Self, indices: []const Array(usize)) ArrayError!Array(usize) {
            const out_shape = try self.multiIndexShape(indices);
            defer self.allocator.free(out_shape);
            var out = try Array(usize).empty(self.allocator, out_shape);
            errdefer out.deinit();
            if (out.data.len == 0) return out;
            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, out_shape, out_multi);
                var offset: usize = 0;
                for (indices, self.shape, self.strides) |idx_array, extent, stride_value| {
                    const coord = idx_array.data[broadcastOffset(out_multi, out_shape.len, idx_array.shape, idx_array.strides)];
                    if (coord >= extent) return error.IndexOutOfBounds;
                    offset += coord * stride_value;
                }
                slot.* = offset;
            }
            return out;
        }

        pub fn takeMultiIndex(self: Self, indices: []const Array(usize)) ArrayError!Self {
            var flat = try self.ravelMultiIndex(indices);
            defer flat.deinit();
            return self.take(flat, null);
        }

        pub fn putMultiIndex(self: Self, indices: []const Array(usize), values: Self) ArrayError!Self {
            var flat = try self.ravelMultiIndex(indices);
            defer flat.deinit();
            return self.putFlat(flat, values);
        }

        pub fn putMultiIndexScalar(self: Self, indices: []const Array(usize), value: T) ArrayError!Self {
            var flat = try self.ravelMultiIndex(indices);
            defer flat.deinit();
            return self.putFlatScalar(flat, value);
        }

        pub fn compress(self: Self, condition: Array(bool), axis_opt: ?isize) ArrayError!Self {
            if (condition.shape.len != 1) return error.ShapeMismatch;
            if (axis_opt == null) {
                var flat = try self.flatten();
                defer flat.deinit();
                if (condition.data.len != flat.data.len) return error.ShapeMismatch;
                return flat.maskedSelect(condition);
            }

            const axis = try normalizeDim(axis_opt.?, self.shape.len);
            if (condition.data.len != self.shape[axis]) return error.ShapeMismatch;
            var selected_count: usize = 0;
            for (condition.data) |keep| {
                if (keep) selected_count += 1;
            }
            const selected = try self.allocator.alloc(usize, selected_count);
            defer self.allocator.free(selected);
            var write: usize = 0;
            for (condition.data, 0..) |keep, i| {
                if (keep) {
                    selected[write] = i;
                    write += 1;
                }
            }

            var out_shape = try self.allocator.dupe(usize, self.shape);
            defer self.allocator.free(out_shape);
            out_shape[axis] = selected_count;
            const out = try Self.empty(self.allocator, out_shape);
            if (out.data.len == 0) return out;
            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            var in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);
            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, out_shape, out_multi);
                @memcpy(in_multi, out_multi);
                in_multi[axis] = selected[out_multi[axis]];
                slot.* = self.data[ravelIndex(in_multi, self.strides)];
            }
            return out;
        }

        pub fn gather(self: Self, axis_index: isize, indices: Array(usize)) ArrayError!Self {
            const axis = try normalizeDim(axis_index, self.shape.len);
            if (indices.shape.len != self.shape.len) return error.ShapeMismatch;
            for (indices.shape, self.shape, 0..) |index_dim, self_dim, i| {
                if (i != axis and index_dim > self_dim) return error.ShapeMismatch;
            }

            var out = try Self.empty(self.allocator, indices.shape);
            errdefer out.deinit();
            if (out.data.len == 0) return out;
            const out_multi = try self.allocator.alloc(usize, indices.shape.len);
            defer self.allocator.free(out_multi);
            var in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);

            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, indices.shape, out_multi);
                const selected = indices.data[flat];
                if (selected >= self.shape[axis]) return error.IndexOutOfBounds;
                @memcpy(in_multi, out_multi);
                in_multi[axis] = selected;
                slot.* = self.data[ravelIndex(in_multi, self.strides)];
            }
            return out;
        }

        pub fn gatherSigned(self: Self, axis_index: isize, indices: Array(isize)) ArrayError!Self {
            const axis = try normalizeDim(axis_index, self.shape.len);
            if (indices.shape.len != self.shape.len) return error.ShapeMismatch;
            for (indices.shape, self.shape, 0..) |index_dim, self_dim, i| {
                if (i != axis and index_dim > self_dim) return error.ShapeMismatch;
            }
            var out = try Self.empty(self.allocator, indices.shape);
            errdefer out.deinit();
            if (out.data.len == 0) return out;
            const out_multi = try self.allocator.alloc(usize, indices.shape.len);
            defer self.allocator.free(out_multi);
            var in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);
            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, indices.shape, out_multi);
                @memcpy(in_multi, out_multi);
                in_multi[axis] = try normalizeIndex(indices.data[flat], self.shape[axis]);
                slot.* = self.data[ravelIndex(in_multi, self.strides)];
            }
            return out;
        }

        pub fn scatter(self: Self, axis_index: isize, indices: Array(usize), src: Self) ArrayError!Self {
            const axis = try normalizeDim(axis_index, self.shape.len);
            if (!std.mem.eql(usize, indices.shape, src.shape)) return error.ShapeMismatch;
            if (indices.shape.len != self.shape.len) return error.ShapeMismatch;
            for (indices.shape, self.shape, 0..) |index_dim, self_dim, i| {
                if (i != axis and index_dim > self_dim) return error.ShapeMismatch;
            }

            var out = try self.clone();
            errdefer out.deinit();
            if (indices.data.len == 0) return out;
            const src_multi = try self.allocator.alloc(usize, indices.shape.len);
            defer self.allocator.free(src_multi);
            var dst_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(dst_multi);

            for (src.data, 0..) |value, flat| {
                unravelIndexInto(flat, indices.shape, src_multi);
                const selected = indices.data[flat];
                if (selected >= self.shape[axis]) return error.IndexOutOfBounds;
                @memcpy(dst_multi, src_multi);
                dst_multi[axis] = selected;
                out.data[ravelIndex(dst_multi, out.strides)] = value;
            }
            return out;
        }

        pub fn scatterScalar(self: Self, axis_index: isize, indices: Array(usize), value: T) ArrayError!Self {
            const axis = try normalizeDim(axis_index, self.shape.len);
            if (indices.shape.len != self.shape.len) return error.ShapeMismatch;
            for (indices.shape, self.shape, 0..) |index_dim, self_dim, i| {
                if (i != axis and index_dim > self_dim) return error.ShapeMismatch;
            }

            var out = try self.clone();
            errdefer out.deinit();
            if (indices.data.len == 0) return out;
            const idx_multi = try self.allocator.alloc(usize, indices.shape.len);
            defer self.allocator.free(idx_multi);
            var dst_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(dst_multi);

            for (indices.data, 0..) |selected, flat| {
                if (selected >= self.shape[axis]) return error.IndexOutOfBounds;
                unravelIndexInto(flat, indices.shape, idx_multi);
                @memcpy(dst_multi, idx_multi);
                dst_multi[axis] = selected;
                out.data[ravelIndex(dst_multi, out.strides)] = value;
            }
            return out;
        }

        fn validateScatterShapes(self: Self, axis: usize, indices: Array(usize), src_shape: []const usize) ArrayError!void {
            if (indices.shape.len != self.shape.len or src_shape.len != self.shape.len) return error.ShapeMismatch;
            if (!std.mem.eql(usize, indices.shape, src_shape)) return error.ShapeMismatch;
            for (indices.shape, self.shape, 0..) |index_dim, self_dim, i| {
                if (i != axis and index_dim > self_dim) return error.ShapeMismatch;
            }
        }

        fn applyScatterReduce(current: T, value: T, reduction: ScatterReduce) T {
            return switch (reduction) {
                .sum => addValue(T, current, value),
                .prod => mulValue(T, current, value),
                .min => if (value < current) value else current,
                .max => if (value > current) value else current,
            };
        }

        pub fn scatterReduce(self: Self, axis_index: isize, indices: Array(usize), src: Self, reduction: ScatterReduce) ArrayError!Self {
            ensureNumeric(T);
            const axis = try normalizeDim(axis_index, self.shape.len);
            try self.validateScatterShapes(axis, indices, src.shape);

            var out = try self.clone();
            errdefer out.deinit();
            if (indices.data.len == 0) return out;
            const src_multi = try self.allocator.alloc(usize, indices.shape.len);
            defer self.allocator.free(src_multi);
            var dst_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(dst_multi);

            for (src.data, 0..) |value, flat| {
                unravelIndexInto(flat, indices.shape, src_multi);
                const selected = indices.data[flat];
                if (selected >= self.shape[axis]) return error.IndexOutOfBounds;
                @memcpy(dst_multi, src_multi);
                dst_multi[axis] = selected;
                const out_index = ravelIndex(dst_multi, out.strides);
                out.data[out_index] = applyScatterReduce(out.data[out_index], value, reduction);
            }
            return out;
        }

        pub fn scatterAdd(self: Self, axis_index: isize, indices: Array(usize), src: Self) ArrayError!Self {
            return self.scatterReduce(axis_index, indices, src, .sum);
        }

        pub fn scatterReduceScalar(self: Self, axis_index: isize, indices: Array(usize), value: T, reduction: ScatterReduce) ArrayError!Self {
            ensureNumeric(T);
            const axis = try normalizeDim(axis_index, self.shape.len);
            if (indices.shape.len != self.shape.len) return error.ShapeMismatch;
            for (indices.shape, self.shape, 0..) |index_dim, self_dim, i| {
                if (i != axis and index_dim > self_dim) return error.ShapeMismatch;
            }

            var out = try self.clone();
            errdefer out.deinit();
            if (indices.data.len == 0) return out;
            const idx_multi = try self.allocator.alloc(usize, indices.shape.len);
            defer self.allocator.free(idx_multi);
            var dst_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(dst_multi);

            for (indices.data, 0..) |selected, flat| {
                if (selected >= self.shape[axis]) return error.IndexOutOfBounds;
                unravelIndexInto(flat, indices.shape, idx_multi);
                @memcpy(dst_multi, idx_multi);
                dst_multi[axis] = selected;
                const out_index = ravelIndex(dst_multi, out.strides);
                out.data[out_index] = applyScatterReduce(out.data[out_index], value, reduction);
            }
            return out;
        }

        pub fn scatterAddScalar(self: Self, axis_index: isize, indices: Array(usize), value: T) ArrayError!Self {
            return self.scatterReduceScalar(axis_index, indices, value, .sum);
        }

        fn binaryArray(self: Self, other: Self, comptime op: fn (T, T) T) ArrayError!Self {
            const out_shape = try broadcastShape(self.allocator, self.shape, other.shape);
            defer self.allocator.free(out_shape);
            const out = try Self.empty(self.allocator, out_shape);

            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);

            for (out.data, 0..) |*slot, i| {
                unravelIndexInto(i, out_shape, out_multi);
                const ai = broadcastOffset(out_multi, out_shape.len, self.shape, self.strides);
                const bi = broadcastOffset(out_multi, out_shape.len, other.shape, other.strides);
                slot.* = op(self.data[ai], other.data[bi]);
            }
            return out;
        }

        fn binaryScalar(self: Self, scalar: T, comptime op: fn (T, T) T) ArrayError!Self {
            const out = try Self.empty(self.allocator, self.shape);
            for (self.data, out.data) |v, *slot| slot.* = op(v, scalar);
            return out;
        }

        fn ternaryArray(self: Self, second: Self, third: Self, comptime op: fn (T, T, T) T) ArrayError!Self {
            const tmp_shape = try broadcastShape(self.allocator, self.shape, second.shape);
            defer self.allocator.free(tmp_shape);
            const out_shape = try broadcastShape(self.allocator, tmp_shape, third.shape);
            defer self.allocator.free(out_shape);
            const out = try Self.empty(self.allocator, out_shape);
            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            for (out.data, 0..) |*slot, i| {
                unravelIndexInto(i, out_shape, out_multi);
                const ai = broadcastOffset(out_multi, out_shape.len, self.shape, self.strides);
                const bi = broadcastOffset(out_multi, out_shape.len, second.shape, second.strides);
                const ci = broadcastOffset(out_multi, out_shape.len, third.shape, third.strides);
                slot.* = op(self.data[ai], second.data[bi], third.data[ci]);
            }
            return out;
        }

        fn ternaryArrayScalar(self: Self, second: Self, third: Self, scalar: T, comptime op: fn (T, T, T, T) T) ArrayError!Self {
            const tmp_shape = try broadcastShape(self.allocator, self.shape, second.shape);
            defer self.allocator.free(tmp_shape);
            const out_shape = try broadcastShape(self.allocator, tmp_shape, third.shape);
            defer self.allocator.free(out_shape);
            const out = try Self.empty(self.allocator, out_shape);
            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            for (out.data, 0..) |*slot, i| {
                unravelIndexInto(i, out_shape, out_multi);
                const ai = broadcastOffset(out_multi, out_shape.len, self.shape, self.strides);
                const bi = broadcastOffset(out_multi, out_shape.len, second.shape, second.strides);
                const ci = broadcastOffset(out_multi, out_shape.len, third.shape, third.strides);
                slot.* = op(self.data[ai], second.data[bi], third.data[ci], scalar);
            }
            return out;
        }

        fn binaryArrayScalar(self: Self, other: Self, scalar: T, comptime op: fn (T, T, T) T) ArrayError!Self {
            const out_shape = try broadcastShape(self.allocator, self.shape, other.shape);
            defer self.allocator.free(out_shape);
            const out = try Self.empty(self.allocator, out_shape);
            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            for (out.data, 0..) |*slot, i| {
                unravelIndexInto(i, out_shape, out_multi);
                const ai = broadcastOffset(out_multi, out_shape.len, self.shape, self.strides);
                const bi = broadcastOffset(out_multi, out_shape.len, other.shape, other.strides);
                slot.* = op(self.data[ai], other.data[bi], scalar);
            }
            return out;
        }

        fn unary(self: Self, comptime op: fn (T) T) ArrayError!Self {
            const out = try Self.empty(self.allocator, self.shape);
            for (self.data, out.data) |v, *slot| slot.* = op(v);
            return out;
        }

        fn unaryBool(self: Self, comptime op: fn (T) bool) ArrayError!Array(bool) {
            const out = try Array(bool).empty(self.allocator, self.shape);
            for (self.data, out.data) |v, *slot| slot.* = op(v);
            return out;
        }

        fn opAdd(a: T, b: T) T {
            return addValue(T, a, b);
        }
        fn opSub(a: T, b: T) T {
            return subValue(T, a, b);
        }
        fn opMul(a: T, b: T) T {
            return mulValue(T, a, b);
        }
        fn opDiv(a: T, b: T) T {
            return divValue(T, a, b);
        }
        fn opPow(a: T, b: T) T {
            if (comptime isComplex(T)) return std.math.complex.pow(a, b);
            return std.math.pow(T, a, b);
        }
        fn opFloorDiv(a: T, b: T) T {
            return @divFloor(a, b);
        }
        fn opMod(a: T, b: T) T {
            return @mod(a, b);
        }
        fn opHypot(a: T, b: T) T {
            return std.math.hypot(a, b);
        }
        fn opAtan2(a: T, b: T) T {
            return std.math.atan2(a, b);
        }
        fn opNextAfter(a: T, b: T) T {
            return std.math.nextAfter(T, a, b);
        }
        fn opCopysign(a: T, b: T) T {
            return std.math.copysign(a, b);
        }
        fn opHeaviside(a: T, b: T) T {
            return if (a < zero(T)) zero(T) else if (a > zero(T)) one(T) else b;
        }
        fn opLogAddExp(a: T, b: T) T {
            if (comptime T == BFloat16) {
                const lhs = a.toF32();
                const rhs = b.toF32();
                const max_value = @max(lhs, rhs);
                return BFloat16.fromF32(max_value + std.math.log1p(std.math.exp(-@abs(lhs - rhs))));
            }
            const max_value = @max(a, b);
            return max_value + std.math.log1p(std.math.exp(-@abs(a - b)));
        }
        fn opLogAddExp2(a: T, b: T) T {
            if (comptime T == BFloat16) {
                const lhs = a.toF32();
                const rhs = b.toF32();
                const max_value = @max(lhs, rhs);
                return BFloat16.fromF32(max_value + std.math.log2(@as(f32, 1) + std.math.pow(f32, 2, -@abs(lhs - rhs))));
            }
            const max_value = @max(a, b);
            return max_value + std.math.log2(one(T) + std.math.pow(T, castValue(T, 2), -@abs(a - b)));
        }
        fn opXlogy(a: T, b: T) T {
            return if (a == zero(T)) zero(T) else a * std.math.log(T, std.math.e, b);
        }
        fn opLerp(a: T, b: T, weight: T) T {
            return addValue(T, a, mulValue(T, subValue(T, b, a), weight));
        }
        fn opLerpScalar(a: T, b: T, weight: T) T {
            return opLerp(a, b, weight);
        }
        fn opAddcmul(a: T, b: T, c: T, value: T) T {
            return addValue(T, a, mulValue(T, value, mulValue(T, b, c)));
        }
        fn opAddcdiv(a: T, b: T, c: T, value: T) T {
            return addValue(T, a, mulValue(T, value, divValue(T, b, c)));
        }
        fn opFmax(a: T, b: T) T {
            if (comptime T == BFloat16) {
                if (std.math.isNan(a.toF32())) return b;
                if (std.math.isNan(b.toF32())) return a;
                return if (a.lt(b)) b else a;
            }
            if (comptime isComplex(T)) @compileError("fmax requires an orderable numeric array");
            return switch (@typeInfo(T)) {
                .float => if (std.math.isNan(a)) b else if (std.math.isNan(b)) a else @max(a, b),
                .int, .comptime_int => @max(a, b),
                else => @compileError("fmax requires an orderable numeric array"),
            };
        }
        fn opFmin(a: T, b: T) T {
            if (comptime T == BFloat16) {
                if (std.math.isNan(a.toF32())) return b;
                if (std.math.isNan(b.toF32())) return a;
                return if (a.lt(b)) a else b;
            }
            if (comptime isComplex(T)) @compileError("fmin requires an orderable numeric array");
            return switch (@typeInfo(T)) {
                .float => if (std.math.isNan(a)) b else if (std.math.isNan(b)) a else @min(a, b),
                .int, .comptime_int => @min(a, b),
                else => @compileError("fmin requires an orderable numeric array"),
            };
        }
        fn opCummax(a: T, b: T) T {
            if (comptime isComplex(T)) @compileError("cummax requires an orderable numeric array");
            return if (lessValue(T, a, b)) b else a;
        }
        fn opCummin(a: T, b: T) T {
            if (comptime isComplex(T)) @compileError("cummin requires an orderable numeric array");
            return if (lessValue(T, b, a)) b else a;
        }
        fn opNeg(a: T) T {
            return negValue(T, a);
        }
        fn opAbs(a: T) T {
            return absValue(T, a);
        }
        fn opExp(a: T) T {
            if (comptime isComplex(T)) return std.math.complex.exp(a);
            if (comptime T == BFloat16) return BFloat16.fromF32(std.math.exp(a.toF32()));
            return std.math.exp(a);
        }
        fn opExp2(a: T) T {
            if (comptime T == BFloat16) return BFloat16.fromF32(std.math.exp2(a.toF32()));
            return std.math.exp2(a);
        }
        fn opLog(a: T) T {
            if (comptime isComplex(T)) return std.math.complex.log(a);
            if (comptime T == BFloat16) return BFloat16.fromF32(std.math.log(f32, std.math.e, a.toF32()));
            return std.math.log(T, std.math.e, a);
        }
        fn opLog2(a: T) T {
            if (comptime isComplex(T)) {
                return std.math.complex.log(a).div(T.init(std.math.ln2, 0));
            }
            if (comptime T == BFloat16) return BFloat16.fromF32(std.math.log2(a.toF32()));
            return std.math.log2(a);
        }
        fn opLog10(a: T) T {
            if (comptime isComplex(T)) {
                return std.math.complex.log(a).div(T.init(std.math.ln10, 0));
            }
            if (comptime T == BFloat16) return BFloat16.fromF32(std.math.log10(a.toF32()));
            return std.math.log10(a);
        }
        fn opSqrt(a: T) T {
            if (comptime isComplex(T)) return std.math.complex.sqrt(a);
            if (comptime T == BFloat16) return BFloat16.fromF32(std.math.sqrt(a.toF32()));
            return std.math.sqrt(a);
        }
        fn opRsqrt(a: T) T {
            if (comptime T == BFloat16) return BFloat16.fromF32(@as(f32, 1) / std.math.sqrt(a.toF32()));
            return one(T) / std.math.sqrt(a);
        }
        fn opCbrt(a: T) T {
            if (comptime T == BFloat16) return BFloat16.fromF32(std.math.cbrt(a.toF32()));
            return switch (T) {
                f16 => @floatCast(std.math.cbrt(@as(f32, @floatCast(a)))),
                f32, f64 => std.math.cbrt(a),
                else => @compileError("cbrt requires a real floating-point array"),
            };
        }
        fn opSin(a: T) T {
            if (comptime isComplex(T)) return std.math.complex.sin(a);
            if (comptime T == BFloat16) return BFloat16.fromF32(std.math.sin(a.toF32()));
            return std.math.sin(a);
        }
        fn opCos(a: T) T {
            if (comptime isComplex(T)) return std.math.complex.cos(a);
            if (comptime T == BFloat16) return BFloat16.fromF32(std.math.cos(a.toF32()));
            return std.math.cos(a);
        }
        fn opTan(a: T) T {
            if (comptime isComplex(T)) return std.math.complex.tan(a);
            if (comptime T == BFloat16) return BFloat16.fromF32(std.math.tan(a.toF32()));
            return std.math.tan(a);
        }
        fn opAsin(a: T) T {
            if (comptime isComplex(T)) return std.math.complex.asin(a);
            if (comptime T == BFloat16) return BFloat16.fromF32(std.math.asin(a.toF32()));
            return std.math.asin(a);
        }
        fn opAcos(a: T) T {
            if (comptime isComplex(T)) return std.math.complex.acos(a);
            if (comptime T == BFloat16) return BFloat16.fromF32(std.math.acos(a.toF32()));
            return std.math.acos(a);
        }
        fn opAtan(a: T) T {
            if (comptime isComplex(T)) return std.math.complex.atan(a);
            if (comptime T == BFloat16) return BFloat16.fromF32(std.math.atan(a.toF32()));
            return std.math.atan(a);
        }
        fn opSinh(a: T) T {
            if (comptime isComplex(T)) return std.math.complex.sinh(a);
            if (comptime T == BFloat16) return BFloat16.fromF32(std.math.sinh(a.toF32()));
            return std.math.sinh(a);
        }
        fn opCosh(a: T) T {
            if (comptime isComplex(T)) return std.math.complex.cosh(a);
            if (comptime T == BFloat16) return BFloat16.fromF32(std.math.cosh(a.toF32()));
            return std.math.cosh(a);
        }
        fn opAsinh(a: T) T {
            if (comptime T == BFloat16) return BFloat16.fromF32(std.math.asinh(a.toF32()));
            return switch (T) {
                f16 => @floatCast(std.math.asinh(@as(f32, @floatCast(a)))),
                f32, f64 => std.math.asinh(a),
                else => @compileError("asinh requires a real floating-point array"),
            };
        }
        fn opAcosh(a: T) T {
            if (comptime T == BFloat16) return BFloat16.fromF32(std.math.acosh(a.toF32()));
            return switch (T) {
                f16 => @floatCast(std.math.acosh(@as(f32, @floatCast(a)))),
                f32, f64 => std.math.acosh(a),
                else => @compileError("acosh requires a real floating-point array"),
            };
        }
        fn opAtanh(a: T) T {
            if (comptime T == BFloat16) return BFloat16.fromF32(std.math.atanh(a.toF32()));
            return switch (T) {
                f16 => @floatCast(std.math.atanh(@as(f32, @floatCast(a)))),
                f32, f64 => std.math.atanh(a),
                else => @compileError("atanh requires a real floating-point array"),
            };
        }
        fn opLgamma(a: T) T {
            if (comptime T == BFloat16) return BFloat16.fromF32(std.math.lgamma(f32, a.toF32()));
            return switch (T) {
                f16 => @floatCast(std.math.lgamma(f32, @as(f32, @floatCast(a)))),
                f32, f64 => std.math.lgamma(T, a),
                else => @compileError("lgamma requires a real floating-point array"),
            };
        }
        fn opLog1p(a: T) T {
            if (comptime isComplex(T)) return std.math.complex.log(a.add(one(T)));
            if (comptime T == BFloat16) return BFloat16.fromF32(std.math.log1p(a.toF32()));
            return std.math.log1p(a);
        }
        fn opExpm1(a: T) T {
            if (comptime isComplex(T)) return std.math.complex.exp(a).sub(one(T));
            if (comptime T == BFloat16) return BFloat16.fromF32(std.math.expm1(a.toF32()));
            return std.math.expm1(a);
        }
        fn opSinc(a: T) T {
            if (comptime T == BFloat16) {
                const value = a.toF32();
                if (value == 0) return one(T);
                const scaled = std.math.pi * value;
                return BFloat16.fromF32(std.math.sin(scaled) / scaled);
            }
            const scaled = castValue(T, std.math.pi) * a;
            return if (a == zero(T)) one(T) else std.math.sin(scaled) / scaled;
        }
        fn opLogit(a: T) T {
            if (comptime T == BFloat16) {
                const value = a.toF32();
                return BFloat16.fromF32(std.math.log(f32, std.math.e, value / (@as(f32, 1) - value)));
            }
            return std.math.log(T, std.math.e, a / (one(T) - a));
        }
        fn opExpit(a: T) T {
            if (comptime T == BFloat16) {
                const value = a.toF32();
                return BFloat16.fromF32(@as(f32, 1) / (@as(f32, 1) + std.math.exp(-value)));
            }
            return one(T) / (one(T) + std.math.exp(-a));
        }
        fn opDeg2rad(a: T) T {
            if (comptime T == BFloat16) return BFloat16.fromF32(a.toF32() * @as(f32, @floatCast(std.math.pi / 180.0)));
            return a * castValue(T, std.math.pi / 180.0);
        }
        fn opRad2deg(a: T) T {
            if (comptime T == BFloat16) return BFloat16.fromF32(a.toF32() * @as(f32, @floatCast(180.0 / std.math.pi)));
            return a * castValue(T, 180.0 / std.math.pi);
        }
        fn opFloor(a: T) T {
            if (comptime T == BFloat16) return BFloat16.fromF32(@floor(a.toF32()));
            return switch (@typeInfo(T)) {
                .float => @floor(a),
                .int, .comptime_int => a,
                else => @compileError("floor requires a numeric array"),
            };
        }
        fn opCeil(a: T) T {
            if (comptime T == BFloat16) return BFloat16.fromF32(@ceil(a.toF32()));
            return switch (@typeInfo(T)) {
                .float => @ceil(a),
                .int, .comptime_int => a,
                else => @compileError("ceil requires a numeric array"),
            };
        }
        fn opRound(a: T) T {
            if (comptime T == BFloat16) return BFloat16.fromF32(@round(a.toF32()));
            return switch (@typeInfo(T)) {
                .float => @round(a),
                .int, .comptime_int => a,
                else => @compileError("round requires a numeric array"),
            };
        }
        fn opTrunc(a: T) T {
            if (comptime T == BFloat16) return BFloat16.fromF32(@trunc(a.toF32()));
            return switch (@typeInfo(T)) {
                .float => @trunc(a),
                .int, .comptime_int => a,
                else => @compileError("trunc requires a numeric array"),
            };
        }
        fn opSquare(a: T) T {
            return mulValue(T, a, a);
        }
        fn opReciprocal(a: T) T {
            return divValue(T, one(T), a);
        }
        fn opSign(a: T) T {
            if (comptime T == BFloat16) {
                const value = a.toF32();
                return if (std.math.isNan(value)) a else if (value > 0) one(T) else if (value < 0) negValue(T, one(T)) else zero(T);
            }
            return switch (@typeInfo(T)) {
                .float => if (std.math.isNan(a)) a else if (a > zero(T)) one(T) else if (a < zero(T)) -one(T) else zero(T),
                .int => |info| if (a == 0) zero(T) else if (info.signedness == .signed) (if (a < 0) -one(T) else one(T)) else one(T),
                .comptime_int, .comptime_float => if (a > 0) 1 else if (a < 0) -1 else 0,
                else => @compileError("sign requires a numeric array"),
            };
        }
        fn opIsNan(a: T) bool {
            if (comptime isComplex(T)) return std.math.isNan(a.re) or std.math.isNan(a.im);
            if (comptime T == BFloat16) return std.math.isNan(a.toF32());
            return switch (@typeInfo(T)) {
                .float => std.math.isNan(a),
                .int, .comptime_int => false,
                else => @compileError("isNan requires a numeric array"),
            };
        }
        fn opIsInf(a: T) bool {
            if (comptime isComplex(T)) return std.math.isInf(a.re) or std.math.isInf(a.im);
            if (comptime T == BFloat16) return std.math.isInf(a.toF32());
            return switch (@typeInfo(T)) {
                .float => std.math.isInf(a),
                .int, .comptime_int => false,
                else => @compileError("isInf requires a numeric array"),
            };
        }
        fn opIsPosInf(a: T) bool {
            if (comptime isComplex(T)) return std.math.isPositiveInf(a.re) or std.math.isPositiveInf(a.im);
            if (comptime T == BFloat16) return std.math.isPositiveInf(a.toF32());
            return switch (@typeInfo(T)) {
                .float => std.math.isPositiveInf(a),
                .int, .comptime_int => false,
                else => @compileError("isPosInf requires a numeric array"),
            };
        }
        fn opIsNegInf(a: T) bool {
            if (comptime isComplex(T)) return std.math.isNegativeInf(a.re) or std.math.isNegativeInf(a.im);
            if (comptime T == BFloat16) return std.math.isNegativeInf(a.toF32());
            return switch (@typeInfo(T)) {
                .float => std.math.isNegativeInf(a),
                .int, .comptime_int => false,
                else => @compileError("isNegInf requires a numeric array"),
            };
        }
        fn opIsFinite(a: T) bool {
            if (comptime isComplex(T)) return std.math.isFinite(a.re) and std.math.isFinite(a.im);
            if (comptime T == BFloat16) return std.math.isFinite(a.toF32());
            return switch (@typeInfo(T)) {
                .float => std.math.isFinite(a),
                .int, .comptime_int => true,
                else => @compileError("isFinite requires a numeric array"),
            };
        }
        fn opIsNormal(a: T) bool {
            if (comptime T == BFloat16) return std.math.isNormal(a.toF32());
            return switch (@typeInfo(T)) {
                .float => std.math.isNormal(a),
                else => @compileError("isNormal requires a floating-point array"),
            };
        }
        fn opSignbit(a: T) bool {
            if (comptime T == BFloat16) return std.math.signbit(a.toF32());
            return switch (@typeInfo(T)) {
                .float => std.math.signbit(a),
                .int => |info| if (info.signedness == .signed) a < 0 else false,
                .comptime_int => a < 0,
                else => @compileError("signbit requires a numeric array"),
            };
        }

        pub fn add(self: Self, other: Self) ArrayError!Self {
            ensureNumeric(T);
            return self.binaryArray(other, opAdd);
        }

        pub fn sub(self: Self, other: Self) ArrayError!Self {
            ensureNumeric(T);
            return self.binaryArray(other, opSub);
        }

        pub fn mul(self: Self, other: Self) ArrayError!Self {
            ensureNumeric(T);
            return self.binaryArray(other, opMul);
        }

        pub fn div(self: Self, other: Self) ArrayError!Self {
            ensureNumeric(T);
            return self.binaryArray(other, opDiv);
        }

        pub fn pow(self: Self, other: Self) ArrayError!Self {
            ensureNumeric(T);
            return self.binaryArray(other, opPow);
        }

        pub fn floorDiv(self: Self, other: Self) ArrayError!Self {
            ensureNumeric(T);
            return self.binaryArray(other, opFloorDiv);
        }

        pub fn mod(self: Self, other: Self) ArrayError!Self {
            ensureNumeric(T);
            return self.binaryArray(other, opMod);
        }

        pub fn remainder(self: Self, other: Self) ArrayError!Self {
            return self.mod(other);
        }

        pub fn hypot(self: Self, other: Self) ArrayError!Self {
            ensureFloat(T);
            return self.binaryArray(other, opHypot);
        }

        pub fn atan2(self: Self, other: Self) ArrayError!Self {
            ensureFloat(T);
            return self.binaryArray(other, opAtan2);
        }

        pub fn arctan2(self: Self, other: Self) ArrayError!Self {
            return self.atan2(other);
        }

        pub fn nextAfter(self: Self, other: Self) ArrayError!Self {
            ensureFloat(T);
            return self.binaryArray(other, opNextAfter);
        }

        pub fn nextafter(self: Self, other: Self) ArrayError!Self {
            return self.nextAfter(other);
        }

        pub fn copysign(self: Self, sign_values: Self) ArrayError!Self {
            ensureFloat(T);
            return self.binaryArray(sign_values, opCopysign);
        }

        pub fn heaviside(self: Self, values_at_zero: Self) ArrayError!Self {
            ensureNumeric(T);
            return self.binaryArray(values_at_zero, opHeaviside);
        }

        pub fn logAddExp(self: Self, other: Self) ArrayError!Self {
            ensureFloat(T);
            return self.binaryArray(other, opLogAddExp);
        }

        pub fn logaddexp(self: Self, other: Self) ArrayError!Self {
            return self.logAddExp(other);
        }

        pub fn logAddExp2(self: Self, other: Self) ArrayError!Self {
            ensureFloat(T);
            return self.binaryArray(other, opLogAddExp2);
        }

        pub fn logaddexp2(self: Self, other: Self) ArrayError!Self {
            return self.logAddExp2(other);
        }

        pub fn xlogy(self: Self, other: Self) ArrayError!Self {
            ensureFloat(T);
            return self.binaryArray(other, opXlogy);
        }

        pub fn lerp(self: Self, end: Self, weight: Self) ArrayError!Self {
            ensureFloat(T);
            return self.ternaryArray(end, weight, opLerp);
        }

        pub fn lerpScalar(self: Self, end: Self, weight: T) ArrayError!Self {
            ensureFloat(T);
            return self.binaryArrayScalar(end, weight, opLerpScalar);
        }

        pub fn addcmul(self: Self, input1: Self, input2: Self, value: T) ArrayError!Self {
            ensureNumeric(T);
            return self.ternaryArrayScalar(input1, input2, value, opAddcmul);
        }

        pub fn addCMul(self: Self, input1: Self, input2: Self, value: T) ArrayError!Self {
            return self.addcmul(input1, input2, value);
        }

        pub fn addcdiv(self: Self, input1: Self, input2: Self, value: T) ArrayError!Self {
            ensureNumeric(T);
            return self.ternaryArrayScalar(input1, input2, value, opAddcdiv);
        }

        pub fn addCDiv(self: Self, input1: Self, input2: Self, value: T) ArrayError!Self {
            return self.addcdiv(input1, input2, value);
        }

        pub fn maximum(self: Self, other: Self) ArrayError!Self {
            ensureNumeric(T);
            return self.binaryArray(other, struct {
                fn f(a: T, b: T) T {
                    return if (lessValue(T, a, b)) b else a;
                }
            }.f);
        }

        pub fn minimum(self: Self, other: Self) ArrayError!Self {
            ensureNumeric(T);
            return self.binaryArray(other, struct {
                fn f(a: T, b: T) T {
                    return if (lessValue(T, b, a)) b else a;
                }
            }.f);
        }

        pub fn fmax(self: Self, other: Self) ArrayError!Self {
            ensureNumeric(T);
            return self.binaryArray(other, opFmax);
        }

        pub fn fmin(self: Self, other: Self) ArrayError!Self {
            ensureNumeric(T);
            return self.binaryArray(other, opFmin);
        }

        pub fn addPromote(self: Self, comptime U: type, other: Array(U)) ArrayError!Array(promoteType(T, U)) {
            const P = promoteType(T, U);
            var lhs = try self.astype(P);
            defer lhs.deinit();
            var rhs = try other.astype(P);
            defer rhs.deinit();
            return lhs.add(rhs);
        }

        pub fn subPromote(self: Self, comptime U: type, other: Array(U)) ArrayError!Array(promoteType(T, U)) {
            const P = promoteType(T, U);
            var lhs = try self.astype(P);
            defer lhs.deinit();
            var rhs = try other.astype(P);
            defer rhs.deinit();
            return lhs.sub(rhs);
        }

        pub fn mulPromote(self: Self, comptime U: type, other: Array(U)) ArrayError!Array(promoteType(T, U)) {
            const P = promoteType(T, U);
            var lhs = try self.astype(P);
            defer lhs.deinit();
            var rhs = try other.astype(P);
            defer rhs.deinit();
            return lhs.mul(rhs);
        }

        pub fn divPromote(self: Self, comptime U: type, other: Array(U)) ArrayError!Array(promoteType(T, U)) {
            const P = promoteType(T, U);
            var lhs = try self.astype(P);
            defer lhs.deinit();
            var rhs = try other.astype(P);
            defer rhs.deinit();
            return lhs.div(rhs);
        }

        pub fn maximumPromote(self: Self, comptime U: type, other: Array(U)) ArrayError!Array(promoteType(T, U)) {
            const P = promoteType(T, U);
            var lhs = try self.astype(P);
            defer lhs.deinit();
            var rhs = try other.astype(P);
            defer rhs.deinit();
            return lhs.maximum(rhs);
        }

        pub fn minimumPromote(self: Self, comptime U: type, other: Array(U)) ArrayError!Array(promoteType(T, U)) {
            const P = promoteType(T, U);
            var lhs = try self.astype(P);
            defer lhs.deinit();
            var rhs = try other.astype(P);
            defer rhs.deinit();
            return lhs.minimum(rhs);
        }

        pub fn addScalar(self: Self, scalar: T) ArrayError!Self {
            ensureNumeric(T);
            return self.binaryScalar(scalar, opAdd);
        }

        pub fn subScalar(self: Self, scalar: T) ArrayError!Self {
            ensureNumeric(T);
            return self.binaryScalar(scalar, opSub);
        }

        pub fn mulScalar(self: Self, scalar: T) ArrayError!Self {
            ensureNumeric(T);
            return self.binaryScalar(scalar, opMul);
        }

        pub fn divScalar(self: Self, scalar: T) ArrayError!Self {
            ensureNumeric(T);
            return self.binaryScalar(scalar, opDiv);
        }

        pub fn powScalar(self: Self, scalar: T) ArrayError!Self {
            ensureNumeric(T);
            return self.binaryScalar(scalar, opPow);
        }

        pub fn floorDivScalar(self: Self, scalar: T) ArrayError!Self {
            ensureNumeric(T);
            return self.binaryScalar(scalar, opFloorDiv);
        }

        pub fn modScalar(self: Self, scalar: T) ArrayError!Self {
            ensureNumeric(T);
            return self.binaryScalar(scalar, opMod);
        }

        pub fn remainderScalar(self: Self, scalar: T) ArrayError!Self {
            return self.modScalar(scalar);
        }

        pub fn maximumScalar(self: Self, scalar: T) ArrayError!Self {
            ensureNumeric(T);
            return self.binaryScalar(scalar, struct {
                fn f(a: T, b: T) T {
                    return if (lessValue(T, a, b)) b else a;
                }
            }.f);
        }

        pub fn minimumScalar(self: Self, scalar: T) ArrayError!Self {
            ensureNumeric(T);
            return self.binaryScalar(scalar, struct {
                fn f(a: T, b: T) T {
                    return if (lessValue(T, b, a)) b else a;
                }
            }.f);
        }

        pub fn clipMin(self: Self, min_value: T) ArrayError!Self {
            return self.maximumScalar(min_value);
        }

        pub fn clampMin(self: Self, min_value: T) ArrayError!Self {
            return self.clipMin(min_value);
        }

        pub fn clipMax(self: Self, max_value: T) ArrayError!Self {
            return self.minimumScalar(max_value);
        }

        pub fn clampMax(self: Self, max_value: T) ArrayError!Self {
            return self.clipMax(max_value);
        }

        pub fn fmaxScalar(self: Self, scalar: T) ArrayError!Self {
            ensureNumeric(T);
            return self.binaryScalar(scalar, opFmax);
        }

        pub fn fminScalar(self: Self, scalar: T) ArrayError!Self {
            ensureNumeric(T);
            return self.binaryScalar(scalar, opFmin);
        }

        pub fn hypotScalar(self: Self, scalar: T) ArrayError!Self {
            ensureFloat(T);
            return self.binaryScalar(scalar, opHypot);
        }

        pub fn atan2Scalar(self: Self, scalar: T) ArrayError!Self {
            ensureFloat(T);
            return self.binaryScalar(scalar, opAtan2);
        }

        pub fn arctan2Scalar(self: Self, scalar: T) ArrayError!Self {
            return self.atan2Scalar(scalar);
        }

        pub fn nextAfterScalar(self: Self, scalar: T) ArrayError!Self {
            ensureFloat(T);
            return self.binaryScalar(scalar, opNextAfter);
        }

        pub fn nextafterScalar(self: Self, scalar: T) ArrayError!Self {
            return self.nextAfterScalar(scalar);
        }

        pub fn copysignScalar(self: Self, scalar: T) ArrayError!Self {
            ensureFloat(T);
            return self.binaryScalar(scalar, opCopysign);
        }

        pub fn heavisideScalar(self: Self, value_at_zero: T) ArrayError!Self {
            ensureNumeric(T);
            return self.binaryScalar(value_at_zero, opHeaviside);
        }

        pub fn logAddExpScalar(self: Self, scalar: T) ArrayError!Self {
            ensureFloat(T);
            return self.binaryScalar(scalar, opLogAddExp);
        }

        pub fn logaddexpScalar(self: Self, scalar: T) ArrayError!Self {
            return self.logAddExpScalar(scalar);
        }

        pub fn logAddExp2Scalar(self: Self, scalar: T) ArrayError!Self {
            ensureFloat(T);
            return self.binaryScalar(scalar, opLogAddExp2);
        }

        pub fn logaddexp2Scalar(self: Self, scalar: T) ArrayError!Self {
            return self.logAddExp2Scalar(scalar);
        }

        pub fn xlogyScalar(self: Self, scalar: T) ArrayError!Self {
            ensureFloat(T);
            return self.binaryScalar(scalar, opXlogy);
        }

        pub fn neg(self: Self) ArrayError!Self {
            ensureNumeric(T);
            return self.unary(opNeg);
        }

        pub fn negative(self: Self) ArrayError!Self {
            return self.neg();
        }

        pub fn positive(self: Self) ArrayError!Self {
            ensureNumeric(T);
            return self.clone();
        }

        pub fn abs(self: Self) ArrayError!Self {
            ensureNumeric(T);
            return self.unary(opAbs);
        }

        pub fn absolute(self: Self) ArrayError!Self {
            return self.abs();
        }

        pub fn fabs(self: Self) ArrayError!Self {
            return self.abs();
        }

        pub fn square(self: Self) ArrayError!Self {
            ensureNumeric(T);
            return self.unary(opSquare);
        }

        pub fn reciprocal(self: Self) ArrayError!Self {
            ensureNumeric(T);
            return self.unary(opReciprocal);
        }

        pub fn sign(self: Self) ArrayError!Self {
            ensureNumeric(T);
            return self.unary(opSign);
        }

        pub fn signbit(self: Self) ArrayError!Array(bool) {
            ensureNumeric(T);
            return self.unaryBool(opSignbit);
        }

        pub fn exp(self: Self) ArrayError!Self {
            ensureNumeric(T);
            return self.unary(opExp);
        }

        pub fn exp2(self: Self) ArrayError!Self {
            ensureFloat(T);
            return self.unary(opExp2);
        }

        pub fn expm1(self: Self) ArrayError!Self {
            ensureNumeric(T);
            return self.unary(opExpm1);
        }

        pub fn log(self: Self) ArrayError!Self {
            ensureNumeric(T);
            return self.unary(opLog);
        }

        pub fn log2(self: Self) ArrayError!Self {
            ensureNumeric(T);
            return self.unary(opLog2);
        }

        pub fn log10(self: Self) ArrayError!Self {
            ensureNumeric(T);
            return self.unary(opLog10);
        }

        pub fn log1p(self: Self) ArrayError!Self {
            ensureNumeric(T);
            return self.unary(opLog1p);
        }

        pub fn lgamma(self: Self) ArrayError!Self {
            ensureFloat(T);
            return self.unary(opLgamma);
        }

        pub fn gammaln(self: Self) ArrayError!Self {
            return self.lgamma();
        }

        pub fn logGamma(self: Self) ArrayError!Self {
            return self.lgamma();
        }

        pub fn loggamma(self: Self) ArrayError!Self {
            return self.lgamma();
        }

        pub fn sqrt(self: Self) ArrayError!Self {
            ensureNumeric(T);
            return self.unary(opSqrt);
        }

        pub fn rsqrt(self: Self) ArrayError!Self {
            ensureFloat(T);
            return self.unary(opRsqrt);
        }

        pub fn cbrt(self: Self) ArrayError!Self {
            ensureFloat(T);
            return self.unary(opCbrt);
        }

        pub fn floor(self: Self) ArrayError!Self {
            ensureNumeric(T);
            return self.unary(opFloor);
        }

        pub fn ceil(self: Self) ArrayError!Self {
            ensureNumeric(T);
            return self.unary(opCeil);
        }

        pub fn round(self: Self) ArrayError!Self {
            ensureNumeric(T);
            return self.unary(opRound);
        }

        pub fn trunc(self: Self) ArrayError!Self {
            ensureNumeric(T);
            return self.unary(opTrunc);
        }

        pub fn deg2rad(self: Self) ArrayError!Self {
            ensureFloat(T);
            return self.unary(opDeg2rad);
        }

        pub fn rad2deg(self: Self) ArrayError!Self {
            ensureFloat(T);
            return self.unary(opRad2deg);
        }

        pub fn radians(self: Self) ArrayError!Self {
            return self.deg2rad();
        }

        pub fn degrees(self: Self) ArrayError!Self {
            return self.rad2deg();
        }

        pub fn sinc(self: Self) ArrayError!Self {
            ensureFloat(T);
            return self.unary(opSinc);
        }

        pub fn ldexp(self: Self, exponents: Array(i32)) ArrayError!Self {
            ensureFloat(T);
            const out_shape = try broadcastShape(self.allocator, self.shape, exponents.shape);
            defer self.allocator.free(out_shape);
            const out = try Self.empty(self.allocator, out_shape);
            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            for (out.data, 0..) |*slot, i| {
                unravelIndexInto(i, out_shape, out_multi);
                const ai = broadcastOffset(out_multi, out_shape.len, self.shape, self.strides);
                const ei = broadcastOffset(out_multi, out_shape.len, exponents.shape, exponents.strides);
                slot.* = std.math.ldexp(self.data[ai], exponents.data[ei]);
            }
            return out;
        }

        pub fn ldexpScalar(self: Self, exponent: i32) ArrayError!Self {
            ensureFloat(T);
            const out = try Self.empty(self.allocator, self.shape);
            for (self.data, out.data) |value, *slot| slot.* = std.math.ldexp(value, exponent);
            return out;
        }

        pub const FrexpResult = struct {
            significand: Self,
            exponent: Array(i32),

            pub fn deinit(self: *@This()) void {
                self.significand.deinit();
                self.exponent.deinit();
                self.* = undefined;
            }
        };

        pub fn frexp(self: Self) ArrayError!FrexpResult {
            ensureFloat(T);
            var significand = try Self.empty(self.allocator, self.shape);
            errdefer significand.deinit();
            var exponent = try Array(i32).empty(self.allocator, self.shape);
            errdefer exponent.deinit();
            for (self.data, significand.data, exponent.data) |value, *sig_slot, *exp_slot| {
                const result = std.math.frexp(value);
                sig_slot.* = result.significand;
                exp_slot.* = result.exponent;
            }
            return .{ .significand = significand, .exponent = exponent };
        }

        pub fn sin(self: Self) ArrayError!Self {
            ensureNumeric(T);
            return self.unary(opSin);
        }

        pub fn cos(self: Self) ArrayError!Self {
            ensureNumeric(T);
            return self.unary(opCos);
        }

        pub fn tan(self: Self) ArrayError!Self {
            ensureNumeric(T);
            return self.unary(opTan);
        }

        pub fn asin(self: Self) ArrayError!Self {
            ensureNumeric(T);
            return self.unary(opAsin);
        }

        pub fn arcsin(self: Self) ArrayError!Self {
            return self.asin();
        }

        pub fn acos(self: Self) ArrayError!Self {
            ensureNumeric(T);
            return self.unary(opAcos);
        }

        pub fn arccos(self: Self) ArrayError!Self {
            return self.acos();
        }

        pub fn atan(self: Self) ArrayError!Self {
            ensureNumeric(T);
            return self.unary(opAtan);
        }

        pub fn arctan(self: Self) ArrayError!Self {
            return self.atan();
        }

        pub fn sinh(self: Self) ArrayError!Self {
            ensureNumeric(T);
            return self.unary(opSinh);
        }

        pub fn cosh(self: Self) ArrayError!Self {
            ensureNumeric(T);
            return self.unary(opCosh);
        }

        pub fn tanh(self: Self) ArrayError!Self {
            ensureNumeric(T);
            return self.unary(struct {
                fn f(a: T) T {
                    if (comptime isComplex(T)) return std.math.complex.tanh(a);
                    return std.math.tanh(a);
                }
            }.f);
        }

        pub fn asinh(self: Self) ArrayError!Self {
            ensureFloat(T);
            return self.unary(opAsinh);
        }

        pub fn arcsinh(self: Self) ArrayError!Self {
            return self.asinh();
        }

        pub fn acosh(self: Self) ArrayError!Self {
            ensureFloat(T);
            return self.unary(opAcosh);
        }

        pub fn arccosh(self: Self) ArrayError!Self {
            return self.acosh();
        }

        pub fn atanh(self: Self) ArrayError!Self {
            ensureFloat(T);
            return self.unary(opAtanh);
        }

        pub fn arctanh(self: Self) ArrayError!Self {
            return self.atanh();
        }

        pub fn relu(self: Self) ArrayError!Self {
            ensureNumeric(T);
            return self.unary(struct {
                fn f(a: T) T {
                    return if (a > zero(T)) a else zero(T);
                }
            }.f);
        }

        pub fn leakyRelu(self: Self, negative_slope: T) ArrayError!Self {
            ensureNumeric(T);
            const out = try Self.empty(self.allocator, self.shape);
            for (self.data, out.data) |value, *slot| {
                slot.* = if (value > zero(T)) value else mulValue(T, value, negative_slope);
            }
            return out;
        }

        pub fn sigmoid(self: Self) ArrayError!Self {
            return self.expit();
        }

        pub fn expit(self: Self) ArrayError!Self {
            ensureFloat(T);
            return self.unary(opExpit);
        }

        pub fn logit(self: Self) ArrayError!Self {
            ensureFloat(T);
            return self.unary(opLogit);
        }

        pub fn softplus(self: Self) ArrayError!Self {
            ensureFloat(T);
            return self.unary(struct {
                fn f(a: T) T {
                    return @max(a, zero(T)) + std.math.log1p(std.math.exp(-@abs(a)));
                }
            }.f);
        }

        pub fn softsign(self: Self) ArrayError!Self {
            ensureFloat(T);
            return self.unary(struct {
                fn f(a: T) T {
                    return a / (one(T) + @abs(a));
                }
            }.f);
        }

        pub fn gelu(self: Self) ArrayError!Self {
            ensureFloat(T);
            return self.unary(struct {
                fn f(a: T) T {
                    const cubic = a * a * a;
                    const gelu_arg = castValue(T, @sqrt(2.0 / std.math.pi)) * (a + castValue(T, 0.044715) * cubic);
                    return castValue(T, 0.5) * a * (one(T) + std.math.tanh(gelu_arg));
                }
            }.f);
        }

        pub fn clip(self: Self, min_value: T, max_value: T) ArrayError!Self {
            ensureNumeric(T);
            const out = try Self.empty(self.allocator, self.shape);
            for (self.data, out.data) |v, *slot| slot.* = @min(@max(v, min_value), max_value);
            return out;
        }

        pub fn clamp(self: Self, min_value: T, max_value: T) ArrayError!Self {
            return self.clip(min_value, max_value);
        }

        pub fn isNan(self: Self) ArrayError!Array(bool) {
            ensureNumeric(T);
            return self.unaryBool(opIsNan);
        }

        pub fn isnan(self: Self) ArrayError!Array(bool) {
            return self.isNan();
        }

        pub fn isInf(self: Self) ArrayError!Array(bool) {
            ensureNumeric(T);
            return self.unaryBool(opIsInf);
        }

        pub fn isinf(self: Self) ArrayError!Array(bool) {
            return self.isInf();
        }

        pub fn isPosInf(self: Self) ArrayError!Array(bool) {
            ensureNumeric(T);
            return self.unaryBool(opIsPosInf);
        }

        pub fn isposinf(self: Self) ArrayError!Array(bool) {
            return self.isPosInf();
        }

        pub fn isNegInf(self: Self) ArrayError!Array(bool) {
            ensureNumeric(T);
            return self.unaryBool(opIsNegInf);
        }

        pub fn isneginf(self: Self) ArrayError!Array(bool) {
            return self.isNegInf();
        }

        pub fn isFinite(self: Self) ArrayError!Array(bool) {
            ensureNumeric(T);
            return self.unaryBool(opIsFinite);
        }

        pub fn isfinite(self: Self) ArrayError!Array(bool) {
            return self.isFinite();
        }

        pub fn isNormal(self: Self) ArrayError!Array(bool) {
            ensureFloat(T);
            return self.unaryBool(opIsNormal);
        }

        pub fn isnormal(self: Self) ArrayError!Array(bool) {
            return self.isNormal();
        }

        pub fn isReal(self: Self) ArrayError!Array(bool) {
            var out = try Array(bool).empty(self.allocator, self.shape);
            errdefer out.deinit();
            if (comptime isComplex(T)) {
                for (self.data, out.data) |value, *slot| slot.* = value.im == 0;
            } else {
                @memset(out.data, true);
            }
            return out;
        }

        pub fn isreal(self: Self) ArrayError!Array(bool) {
            return self.isReal();
        }

        pub fn iscomplex(self: Self) ArrayError!Array(bool) {
            var out = try Array(bool).empty(self.allocator, self.shape);
            errdefer out.deinit();
            if (comptime isComplex(T)) {
                for (self.data, out.data) |value, *slot| slot.* = value.im != 0;
            } else {
                @memset(out.data, false);
            }
            return out;
        }

        pub fn logsumexp(self: Self, axis_index: isize, keepdims: bool) ArrayError!Self {
            ensureFloat(T);
            var max_t = try self.max(axis_index, true);
            defer max_t.deinit();
            var shifted = try self.sub(max_t);
            defer shifted.deinit();
            var exp_t = try shifted.exp();
            defer exp_t.deinit();
            var summed = try exp_t.sum(axis_index, true);
            defer summed.deinit();
            var log_summed = try summed.log();
            defer log_summed.deinit();
            var with_max = try log_summed.add(max_t);
            if (keepdims) return with_max;
            errdefer with_max.deinit();
            const squeezed = try with_max.squeeze(axis_index);
            with_max.deinit();
            return squeezed;
        }

        pub fn logSoftmax(self: Self, axis_index: isize) ArrayError!Self {
            ensureFloat(T);
            var lse = try self.logsumexp(axis_index, true);
            defer lse.deinit();
            return self.sub(lse);
        }

        pub fn log_softmax(self: Self, axis_index: isize) ArrayError!Self {
            return self.logSoftmax(axis_index);
        }

        pub fn eq(self: Self, other: Self) ArrayError!Array(bool) {
            return self.compare(other, struct {
                fn f(a: T, b: T) bool {
                    return a == b;
                }
            }.f);
        }

        pub fn equal(self: Self, other: Self) ArrayError!Array(bool) {
            return self.eq(other);
        }

        pub fn gt(self: Self, other: Self) ArrayError!Array(bool) {
            ensureNumeric(T);
            return self.compare(other, struct {
                fn f(a: T, b: T) bool {
                    return lessValue(T, b, a);
                }
            }.f);
        }

        pub fn greater(self: Self, other: Self) ArrayError!Array(bool) {
            return self.gt(other);
        }

        pub fn lt(self: Self, other: Self) ArrayError!Array(bool) {
            ensureNumeric(T);
            return self.compare(other, struct {
                fn f(a: T, b: T) bool {
                    return lessValue(T, a, b);
                }
            }.f);
        }

        pub fn less(self: Self, other: Self) ArrayError!Array(bool) {
            return self.lt(other);
        }

        pub fn ne(self: Self, other: Self) ArrayError!Array(bool) {
            return self.compare(other, struct {
                fn f(a: T, b: T) bool {
                    return a != b;
                }
            }.f);
        }

        pub fn notEqual(self: Self, other: Self) ArrayError!Array(bool) {
            return self.ne(other);
        }

        pub fn ge(self: Self, other: Self) ArrayError!Array(bool) {
            ensureNumeric(T);
            return self.compare(other, struct {
                fn f(a: T, b: T) bool {
                    return !lessValue(T, a, b);
                }
            }.f);
        }

        pub fn greaterEqual(self: Self, other: Self) ArrayError!Array(bool) {
            return self.ge(other);
        }

        pub fn le(self: Self, other: Self) ArrayError!Array(bool) {
            ensureNumeric(T);
            return self.compare(other, struct {
                fn f(a: T, b: T) bool {
                    return !lessValue(T, b, a);
                }
            }.f);
        }

        pub fn lessEqual(self: Self, other: Self) ArrayError!Array(bool) {
            return self.le(other);
        }

        pub fn eqScalar(self: Self, scalar: T) ArrayError!Array(bool) {
            return self.compareScalar(scalar, struct {
                fn f(a: T, b: T) bool {
                    return a == b;
                }
            }.f);
        }

        pub fn equalScalar(self: Self, scalar: T) ArrayError!Array(bool) {
            return self.eqScalar(scalar);
        }

        pub fn neScalar(self: Self, scalar: T) ArrayError!Array(bool) {
            return self.compareScalar(scalar, struct {
                fn f(a: T, b: T) bool {
                    return a != b;
                }
            }.f);
        }

        pub fn notEqualScalar(self: Self, scalar: T) ArrayError!Array(bool) {
            return self.neScalar(scalar);
        }

        pub fn gtScalar(self: Self, scalar: T) ArrayError!Array(bool) {
            ensureNumeric(T);
            return self.compareScalar(scalar, struct {
                fn f(a: T, b: T) bool {
                    return lessValue(T, b, a);
                }
            }.f);
        }

        pub fn greaterScalar(self: Self, scalar: T) ArrayError!Array(bool) {
            return self.gtScalar(scalar);
        }

        pub fn geScalar(self: Self, scalar: T) ArrayError!Array(bool) {
            ensureNumeric(T);
            return self.compareScalar(scalar, struct {
                fn f(a: T, b: T) bool {
                    return !lessValue(T, a, b);
                }
            }.f);
        }

        pub fn greaterEqualScalar(self: Self, scalar: T) ArrayError!Array(bool) {
            return self.geScalar(scalar);
        }

        pub fn ltScalar(self: Self, scalar: T) ArrayError!Array(bool) {
            ensureNumeric(T);
            return self.compareScalar(scalar, struct {
                fn f(a: T, b: T) bool {
                    return lessValue(T, a, b);
                }
            }.f);
        }

        pub fn lessScalar(self: Self, scalar: T) ArrayError!Array(bool) {
            return self.ltScalar(scalar);
        }

        pub fn leScalar(self: Self, scalar: T) ArrayError!Array(bool) {
            ensureNumeric(T);
            return self.compareScalar(scalar, struct {
                fn f(a: T, b: T) bool {
                    return !lessValue(T, b, a);
                }
            }.f);
        }

        pub fn lessEqualScalar(self: Self, scalar: T) ArrayError!Array(bool) {
            return self.leScalar(scalar);
        }

        pub fn allclose(self: Self, other: Self, rtol: T, atol: T) ArrayError!bool {
            ensureFloat(T);
            const out_shape = try broadcastShape(self.allocator, self.shape, other.shape);
            defer self.allocator.free(out_shape);
            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            for (0..product(out_shape)) |i| {
                unravelIndexInto(i, out_shape, out_multi);
                const ai = broadcastOffset(out_multi, out_shape.len, self.shape, self.strides);
                const bi = broadcastOffset(out_multi, out_shape.len, other.shape, other.strides);
                const lhs = self.data[ai];
                const rhs = other.data[bi];
                if (@abs(lhs - rhs) > atol + rtol * @abs(rhs)) return false;
            }
            return true;
        }

        pub fn isclose(self: Self, other: Self, rtol: T, atol: T) ArrayError!Array(bool) {
            ensureFloat(T);
            const out_shape = try broadcastShape(self.allocator, self.shape, other.shape);
            defer self.allocator.free(out_shape);
            const out = try Array(bool).empty(self.allocator, out_shape);
            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            for (out.data, 0..) |*slot, i| {
                unravelIndexInto(i, out_shape, out_multi);
                const ai = broadcastOffset(out_multi, out_shape.len, self.shape, self.strides);
                const bi = broadcastOffset(out_multi, out_shape.len, other.shape, other.strides);
                const lhs = self.data[ai];
                const rhs = other.data[bi];
                slot.* = @abs(lhs - rhs) <= atol + rtol * @abs(rhs);
            }
            return out;
        }

        pub fn iscloseScalar(self: Self, scalar: T, rtol: T, atol: T) ArrayError!Array(bool) {
            ensureFloat(T);
            const out = try Array(bool).empty(self.allocator, self.shape);
            for (self.data, out.data) |value, *slot| {
                slot.* = @abs(value - scalar) <= atol + rtol * @abs(scalar);
            }
            return out;
        }

        pub fn allcloseScalar(self: Self, scalar: T, rtol: T, atol: T) ArrayError!bool {
            ensureFloat(T);
            for (self.data) |value| {
                if (@abs(value - scalar) > atol + rtol * @abs(scalar)) return false;
            }
            return true;
        }

        fn compare(self: Self, other: Self, comptime op: fn (T, T) bool) ArrayError!Array(bool) {
            const out_shape = try broadcastShape(self.allocator, self.shape, other.shape);
            defer self.allocator.free(out_shape);
            const out = try Array(bool).empty(self.allocator, out_shape);

            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);

            for (out.data, 0..) |*slot, i| {
                unravelIndexInto(i, out_shape, out_multi);
                const ai = broadcastOffset(out_multi, out_shape.len, self.shape, self.strides);
                const bi = broadcastOffset(out_multi, out_shape.len, other.shape, other.strides);
                slot.* = op(self.data[ai], other.data[bi]);
            }
            return out;
        }

        fn compareScalar(self: Self, scalar: T, comptime op: fn (T, T) bool) ArrayError!Array(bool) {
            const out = try Array(bool).empty(self.allocator, self.shape);
            for (self.data, out.data) |value, *slot| slot.* = op(value, scalar);
            return out;
        }

        fn whereMask(mask: Array(bool), a: Self, b: Self) ArrayError!Self {
            const tmp_shape = try broadcastShape(a.allocator, a.shape, b.shape);
            defer a.allocator.free(tmp_shape);
            const out_shape = try broadcastShape(a.allocator, tmp_shape, mask.shape);
            defer a.allocator.free(out_shape);
            var out = try Self.empty(a.allocator, out_shape);
            errdefer out.deinit();

            const out_multi = try a.allocator.alloc(usize, out_shape.len);
            defer a.allocator.free(out_multi);
            for (out.data, 0..) |*slot, i| {
                unravelIndexInto(i, out_shape, out_multi);
                const mi = broadcastOffset(out_multi, out_shape.len, mask.shape, mask.strides);
                const ai = broadcastOffset(out_multi, out_shape.len, a.shape, a.strides);
                const bi = broadcastOffset(out_multi, out_shape.len, b.shape, b.strides);
                slot.* = if (mask.data[mi]) a.data[ai] else b.data[bi];
            }
            return out;
        }

        pub fn all(self: Self) bool {
            if (comptime T != bool) @compileError("all requires Array(bool)");
            for (self.data) |v| if (!v) return false;
            return true;
        }

        pub fn any(self: Self) bool {
            if (comptime T != bool) @compileError("any requires Array(bool)");
            for (self.data) |v| if (v) return true;
            return false;
        }

        pub fn allAxis(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Self {
            if (comptime T != bool) @compileError("allAxis requires Array(bool)");
            return self.boolReduce(axis_opt, keepdims, true, struct {
                fn f(a: bool, b: bool) bool {
                    return a and b;
                }
            }.f);
        }

        pub fn anyAxis(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Self {
            if (comptime T != bool) @compileError("anyAxis requires Array(bool)");
            return self.boolReduce(axis_opt, keepdims, false, struct {
                fn f(a: bool, b: bool) bool {
                    return a or b;
                }
            }.f);
        }

        fn boolReduce(self: Self, axis_opt: ?isize, keepdims: bool, init_value: bool, comptime op: fn (bool, bool) bool) ArrayError!Self {
            if (axis_opt == null) {
                var total = init_value;
                for (self.data) |v| total = op(total, v);
                if (keepdims) {
                    const out_shape = try keepDimsAllOnes(self.allocator, self.shape.len);
                    defer self.allocator.free(out_shape);
                    return Self.fromSlice(self.allocator, &.{total}, out_shape);
                }
                return Self.fromSlice(self.allocator, &.{total}, &.{});
            }

            const axis = try normalizeDim(axis_opt.?, self.shape.len);
            const out_shape = try self.reducedShape(axis, keepdims);
            defer self.allocator.free(out_shape);
            var out = try Self.full(self.allocator, out_shape, init_value);
            errdefer out.deinit();
            if (out.data.len == 0) return out;
            var in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);
            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);

            for (self.data, 0..) |v, flat| {
                unravelIndexInto(flat, self.shape, in_multi);
                if (keepdims) {
                    @memcpy(out_multi, in_multi);
                    out_multi[axis] = 0;
                } else {
                    for (in_multi[0..axis], 0..) |coord, i| out_multi[i] = coord;
                    for (in_multi[axis + 1 ..], axis..) |coord, i| out_multi[i] = coord;
                }
                const out_index = ravelIndex(out_multi, out.strides);
                out.data[out_index] = op(out.data[out_index], v);
            }
            return out;
        }

        pub fn logicalNot(self: Self) ArrayError!Self {
            if (comptime T != bool) @compileError("logicalNot requires Array(bool)");
            const out = try Self.empty(self.allocator, self.shape);
            for (self.data, out.data) |v, *slot| slot.* = !v;
            return out;
        }

        pub fn logicalAnd(self: Self, other: Self) ArrayError!Self {
            if (comptime T != bool) @compileError("logicalAnd requires Array(bool)");
            return self.binaryArray(other, struct {
                fn f(a: bool, b: bool) bool {
                    return a and b;
                }
            }.f);
        }

        pub fn logicalAndScalar(self: Self, scalar: bool) ArrayError!Self {
            if (comptime T != bool) @compileError("logicalAndScalar requires Array(bool)");
            return self.binaryScalar(scalar, struct {
                fn f(a: bool, b: bool) bool {
                    return a and b;
                }
            }.f);
        }

        pub fn logicalOr(self: Self, other: Self) ArrayError!Self {
            if (comptime T != bool) @compileError("logicalOr requires Array(bool)");
            return self.binaryArray(other, struct {
                fn f(a: bool, b: bool) bool {
                    return a or b;
                }
            }.f);
        }

        pub fn logicalOrScalar(self: Self, scalar: bool) ArrayError!Self {
            if (comptime T != bool) @compileError("logicalOrScalar requires Array(bool)");
            return self.binaryScalar(scalar, struct {
                fn f(a: bool, b: bool) bool {
                    return a or b;
                }
            }.f);
        }

        pub fn logicalXor(self: Self, other: Self) ArrayError!Self {
            if (comptime T != bool) @compileError("logicalXor requires Array(bool)");
            return self.binaryArray(other, struct {
                fn f(a: bool, b: bool) bool {
                    return a != b;
                }
            }.f);
        }

        pub fn logicalXorScalar(self: Self, scalar: bool) ArrayError!Self {
            if (comptime T != bool) @compileError("logicalXorScalar requires Array(bool)");
            return self.binaryScalar(scalar, struct {
                fn f(a: bool, b: bool) bool {
                    return a != b;
                }
            }.f);
        }

        pub fn sum(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Self {
            ensureNumeric(T);
            return self.reduce(axis_opt, keepdims, zero(T), opAdd);
        }

        pub fn prod(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Self {
            ensureNumeric(T);
            return self.reduce(axis_opt, keepdims, one(T), opMul);
        }

        pub fn min(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Self {
            ensureNumeric(T);
            if (self.data.len == 0) return error.EmptyArray;
            return self.reduceFirst(axis_opt, keepdims, struct {
                fn f(a: T, b: T) T {
                    return if (lessValue(T, b, a)) b else a;
                }
            }.f);
        }

        pub fn amin(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Self {
            return self.min(axis_opt, keepdims);
        }

        pub fn max(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Self {
            ensureNumeric(T);
            if (self.data.len == 0) return error.EmptyArray;
            return self.reduceFirst(axis_opt, keepdims, struct {
                fn f(a: T, b: T) T {
                    return if (lessValue(T, a, b)) b else a;
                }
            }.f);
        }

        pub fn amax(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Self {
            return self.max(axis_opt, keepdims);
        }

        pub fn ptp(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Self {
            var max_values = try self.max(axis_opt, keepdims);
            defer max_values.deinit();
            var min_values = try self.min(axis_opt, keepdims);
            defer min_values.deinit();
            return max_values.sub(min_values);
        }

        fn reducedShape(self: Self, axis: usize, keepdims: bool) ArrayError![]usize {
            var out_shape = try self.allocator.alloc(usize, if (keepdims) self.shape.len else self.shape.len - 1);
            if (keepdims) {
                @memcpy(out_shape, self.shape);
                out_shape[axis] = 1;
            } else {
                for (self.shape[0..axis], 0..) |d, i| out_shape[i] = d;
                for (self.shape[axis + 1 ..], axis..) |d, i| out_shape[i] = d;
            }
            return out_shape;
        }

        fn mapReducedToInput(self: Self, axis: usize, keepdims: bool, out_multi: []const usize, in_multi: []usize) void {
            _ = self;
            if (keepdims) {
                @memcpy(in_multi, out_multi);
            } else {
                for (out_multi[0..axis], 0..) |coord, i| in_multi[i] = coord;
                for (out_multi[axis..], axis + 1..) |coord, i| in_multi[i] = coord;
            }
        }

        fn reduceFirst(self: Self, axis_opt: ?isize, keepdims: bool, comptime op: fn (T, T) T) ArrayError!Self {
            if (self.data.len == 0) return error.EmptyArray;
            if (axis_opt == null) {
                var total = self.data[0];
                for (self.data[1..]) |v| total = op(total, v);
                if (keepdims) {
                    const out_shape = try keepDimsAllOnes(self.allocator, self.shape.len);
                    defer self.allocator.free(out_shape);
                    return Self.fromSlice(self.allocator, &.{total}, out_shape);
                }
                return Self.fromSlice(self.allocator, &.{total}, &.{});
            }

            const axis = try normalizeDim(axis_opt.?, self.shape.len);
            if (self.shape[axis] == 0) return error.EmptyArray;
            const out_shape = try self.reducedShape(axis, keepdims);
            defer self.allocator.free(out_shape);
            var out = try Self.empty(self.allocator, out_shape);
            errdefer out.deinit();
            if (out.data.len == 0) return out;
            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            var in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);

            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, out_shape, out_multi);
                self.mapReducedToInput(axis, keepdims, out_multi, in_multi);
                in_multi[axis] = 0;
                var acc = self.data[ravelIndex(in_multi, self.strides)];
                for (1..self.shape[axis]) |axis_i| {
                    in_multi[axis] = axis_i;
                    acc = op(acc, self.data[ravelIndex(in_multi, self.strides)]);
                }
                slot.* = acc;
            }
            return out;
        }

        fn reduce(self: Self, axis_opt: ?isize, keepdims: bool, init_value: T, comptime op: fn (T, T) T) ArrayError!Self {
            if (axis_opt == null) {
                var total = init_value;
                for (self.data) |v| total = op(total, v);
                if (keepdims) {
                    const out_shape = try keepDimsAllOnes(self.allocator, self.shape.len);
                    defer self.allocator.free(out_shape);
                    return Self.fromSlice(self.allocator, &.{total}, out_shape);
                }
                return Self.fromSlice(self.allocator, &.{total}, &.{});
            }

            const axis = try normalizeDim(axis_opt.?, self.shape.len);
            var out_shape = try self.allocator.alloc(usize, if (keepdims) self.shape.len else self.shape.len - 1);
            defer self.allocator.free(out_shape);
            if (keepdims) {
                @memcpy(out_shape, self.shape);
                out_shape[axis] = 1;
            } else {
                for (self.shape[0..axis], 0..) |d, i| out_shape[i] = d;
                for (self.shape[axis + 1 ..], axis..) |d, i| out_shape[i] = d;
            }

            var out = try Self.full(self.allocator, out_shape, init_value);
            if (out.data.len == 0) return out;
            var in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);
            var out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            const out_strides = out.strides;
            for (self.data, 0..) |v, flat| {
                unravelIndexInto(flat, self.shape, in_multi);
                if (keepdims) {
                    @memcpy(out_multi, in_multi);
                    out_multi[axis] = 0;
                } else {
                    for (in_multi[0..axis], 0..) |coord, i| out_multi[i] = coord;
                    for (in_multi[axis + 1 ..], axis..) |coord, i| out_multi[i] = coord;
                }
                const oi = ravelIndex(out_multi, out_strides);
                out.data[oi] = op(out.data[oi], v);
            }
            return out;
        }

        fn keepDimsAllOnes(allocator: std.mem.Allocator, rank_count: usize) ArrayError![]usize {
            const dims = try allocator.alloc(usize, rank_count);
            @memset(dims, 1);
            return dims;
        }

        pub fn mean(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Self {
            ensureFloat(T);
            const out = try self.sum(axis_opt, keepdims);
            const divisor: T = if (axis_opt) |d| castValue(T, self.shape[try normalizeDim(d, self.shape.len)]) else castValue(T, self.data.len);
            for (out.data) |*v| v.* /= divisor;
            return out;
        }

        pub fn variance(self: Self, axis_opt: ?isize, keepdims: bool, correction: T) ArrayError!Self {
            ensureFloat(T);
            if (axis_opt != null) {
                const axis = try normalizeDim(axis_opt.?, self.shape.len);
                const n = self.shape[axis];
                if (n == 0) return error.EmptyArray;
                var mean_t = try self.mean(axis_opt, true);
                defer mean_t.deinit();
                var out_shape = try self.allocator.alloc(usize, if (keepdims) self.shape.len else self.shape.len - 1);
                defer self.allocator.free(out_shape);
                if (keepdims) {
                    @memcpy(out_shape, self.shape);
                    out_shape[axis] = 1;
                } else {
                    for (self.shape[0..axis], 0..) |d, i| out_shape[i] = d;
                    for (self.shape[axis + 1 ..], axis..) |d, i| out_shape[i] = d;
                }
                var out = try Self.zeros(self.allocator, out_shape);
                var in_multi = try self.allocator.alloc(usize, self.shape.len);
                defer self.allocator.free(in_multi);
                var out_multi = try self.allocator.alloc(usize, out_shape.len);
                defer self.allocator.free(out_multi);
                var mean_multi = try self.allocator.alloc(usize, self.shape.len);
                defer self.allocator.free(mean_multi);
                for (self.data, 0..) |v, flat| {
                    unravelIndexInto(flat, self.shape, in_multi);
                    @memcpy(mean_multi, in_multi);
                    mean_multi[axis] = 0;
                    if (keepdims) {
                        @memcpy(out_multi, in_multi);
                        out_multi[axis] = 0;
                    } else {
                        for (in_multi[0..axis], 0..) |coord, i| out_multi[i] = coord;
                        for (in_multi[axis + 1 ..], axis..) |coord, i| out_multi[i] = coord;
                    }
                    const delta = v - mean_t.data[ravelIndex(mean_multi, mean_t.strides)];
                    const oi = ravelIndex(out_multi, out.strides);
                    out.data[oi] += delta * delta;
                }
                const denom = castValue(T, n) - correction;
                for (out.data) |*v| v.* /= denom;
                return out;
            }

            if (self.data.len == 0) return error.EmptyArray;
            var mean_value: T = zero(T);
            for (self.data) |v| mean_value += v;
            mean_value /= castValue(T, self.data.len);
            var total: T = zero(T);
            for (self.data) |v| {
                const delta = v - mean_value;
                total += delta * delta;
            }
            const denom = castValue(T, self.data.len) - correction;
            const result = total / denom;
            if (keepdims) {
                const out_shape = try keepDimsAllOnes(self.allocator, self.shape.len);
                defer self.allocator.free(out_shape);
                return Self.fromSlice(self.allocator, &.{result}, out_shape);
            }
            return Self.fromSlice(self.allocator, &.{result}, &.{});
        }

        pub fn stddev(self: Self, axis_opt: ?isize, keepdims: bool, correction: T) ArrayError!Self {
            const out = try self.variance(axis_opt, keepdims, correction);
            for (out.data) |*v| v.* = std.math.sqrt(v.*);
            return out;
        }

        fn maxFiniteValue() T {
            if (comptime T == BFloat16) return .{ .bits = 0x7f7f };
            return switch (@typeInfo(T)) {
                .float => std.math.floatMax(T),
                else => @compileError("maxFiniteValue requires a floating-point array"),
            };
        }

        pub fn nanToNum(self: Self, nan_value: T, posinf_value: T, neginf_value: T) ArrayError!Self {
            ensureFloat(T);
            const out = try Self.empty(self.allocator, self.shape);
            for (self.data, out.data) |value, *slot| {
                slot.* = if (opIsNan(value))
                    nan_value
                else if (opIsPosInf(value))
                    posinf_value
                else if (opIsNegInf(value))
                    neginf_value
                else
                    value;
            }
            return out;
        }

        pub fn nan_to_num(self: Self, nan_value: T, posinf_value: T, neginf_value: T) ArrayError!Self {
            return self.nanToNum(nan_value, posinf_value, neginf_value);
        }

        pub fn nanToNumDefault(self: Self) ArrayError!Self {
            const max_value = maxFiniteValue();
            return self.nanToNum(zero(T), max_value, negValue(T, max_value));
        }

        pub fn nan_to_num_default(self: Self) ArrayError!Self {
            return self.nanToNumDefault();
        }

        pub fn nansum(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Self {
            ensureFloat(T);
            if (axis_opt == null) {
                var total = zero(T);
                for (self.data) |value| {
                    if (!std.math.isNan(value)) total += value;
                }
                if (keepdims) {
                    const out_shape = try keepDimsAllOnes(self.allocator, self.shape.len);
                    defer self.allocator.free(out_shape);
                    return Self.fromSlice(self.allocator, &.{total}, out_shape);
                }
                return Self.fromSlice(self.allocator, &.{total}, &.{});
            }

            const axis = try normalizeDim(axis_opt.?, self.shape.len);
            const out_shape = try self.reducedShape(axis, keepdims);
            defer self.allocator.free(out_shape);
            var out = try Self.zeros(self.allocator, out_shape);
            errdefer out.deinit();
            if (out.data.len == 0) return out;

            var in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);
            var out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);

            for (self.data, 0..) |value, flat| {
                if (std.math.isNan(value)) continue;
                unravelIndexInto(flat, self.shape, in_multi);
                if (keepdims) {
                    @memcpy(out_multi, in_multi);
                    out_multi[axis] = 0;
                } else {
                    for (in_multi[0..axis], 0..) |coord, i| out_multi[i] = coord;
                    for (in_multi[axis + 1 ..], axis..) |coord, i| out_multi[i] = coord;
                }
                out.data[ravelIndex(out_multi, out.strides)] += value;
            }
            return out;
        }

        fn nanmeanWithCounts(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!struct { values: Self, counts: Array(usize) } {
            if (axis_opt == null) {
                var total = zero(T);
                var count: usize = 0;
                for (self.data) |value| {
                    if (std.math.isNan(value)) continue;
                    total += value;
                    count += 1;
                }
                const result = if (count == 0) std.math.nan(T) else total / castValue(T, count);
                const out_shape = if (keepdims) try keepDimsAllOnes(self.allocator, self.shape.len) else try self.allocator.dupe(usize, &.{});
                defer self.allocator.free(out_shape);
                var values = try Self.fromSlice(self.allocator, &.{result}, out_shape);
                errdefer values.deinit();
                var counts = try Array(usize).fromSlice(self.allocator, &.{count}, out_shape);
                errdefer counts.deinit();
                return .{ .values = values, .counts = counts };
            }

            const axis = try normalizeDim(axis_opt.?, self.shape.len);
            const out_shape = try self.reducedShape(axis, keepdims);
            defer self.allocator.free(out_shape);
            var values = try Self.zeros(self.allocator, out_shape);
            errdefer values.deinit();
            var counts = try Array(usize).zeros(self.allocator, out_shape);
            errdefer counts.deinit();
            if (values.data.len == 0) return .{ .values = values, .counts = counts };

            var in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);
            var out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);

            for (self.data, 0..) |value, flat| {
                if (std.math.isNan(value)) continue;
                unravelIndexInto(flat, self.shape, in_multi);
                if (keepdims) {
                    @memcpy(out_multi, in_multi);
                    out_multi[axis] = 0;
                } else {
                    for (in_multi[0..axis], 0..) |coord, i| out_multi[i] = coord;
                    for (in_multi[axis + 1 ..], axis..) |coord, i| out_multi[i] = coord;
                }
                const out_index = ravelIndex(out_multi, values.strides);
                values.data[out_index] += value;
                counts.data[out_index] += 1;
            }

            for (values.data, counts.data) |*value, count| {
                value.* = if (count == 0) std.math.nan(T) else value.* / castValue(T, count);
            }
            return .{ .values = values, .counts = counts };
        }

        pub fn nanmean(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Self {
            ensureFloat(T);
            var result = try self.nanmeanWithCounts(axis_opt, keepdims);
            result.counts.deinit();
            return result.values;
        }

        pub fn nanvar(self: Self, axis_opt: ?isize, keepdims: bool, correction: T) ArrayError!Self {
            ensureFloat(T);
            if (axis_opt == null) {
                var mean_value = zero(T);
                var count: usize = 0;
                for (self.data) |value| {
                    if (std.math.isNan(value)) continue;
                    mean_value += value;
                    count += 1;
                }
                if (count == 0) {
                    const result = std.math.nan(T);
                    if (keepdims) {
                        const out_shape = try keepDimsAllOnes(self.allocator, self.shape.len);
                        defer self.allocator.free(out_shape);
                        return Self.fromSlice(self.allocator, &.{result}, out_shape);
                    }
                    return Self.fromSlice(self.allocator, &.{result}, &.{});
                }
                mean_value /= castValue(T, count);
                var total = zero(T);
                for (self.data) |value| {
                    if (std.math.isNan(value)) continue;
                    const delta = value - mean_value;
                    total += delta * delta;
                }
                const denom = castValue(T, count) - correction;
                const result = if (denom > zero(T)) total / denom else std.math.nan(T);
                if (keepdims) {
                    const out_shape = try keepDimsAllOnes(self.allocator, self.shape.len);
                    defer self.allocator.free(out_shape);
                    return Self.fromSlice(self.allocator, &.{result}, out_shape);
                }
                return Self.fromSlice(self.allocator, &.{result}, &.{});
            }

            const axis = try normalizeDim(axis_opt.?, self.shape.len);
            var mean_result = try self.nanmeanWithCounts(axis_opt, true);
            defer mean_result.values.deinit();
            defer mean_result.counts.deinit();

            const out_shape = try self.reducedShape(axis, keepdims);
            defer self.allocator.free(out_shape);
            var out = try Self.zeros(self.allocator, out_shape);
            errdefer out.deinit();
            if (out.data.len == 0) return out;

            var in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);
            var out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            var mean_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(mean_multi);

            for (self.data, 0..) |value, flat| {
                if (std.math.isNan(value)) continue;
                unravelIndexInto(flat, self.shape, in_multi);
                @memcpy(mean_multi, in_multi);
                mean_multi[axis] = 0;
                if (keepdims) {
                    @memcpy(out_multi, in_multi);
                    out_multi[axis] = 0;
                } else {
                    for (in_multi[0..axis], 0..) |coord, i| out_multi[i] = coord;
                    for (in_multi[axis + 1 ..], axis..) |coord, i| out_multi[i] = coord;
                }
                const delta = value - mean_result.values.data[ravelIndex(mean_multi, mean_result.values.strides)];
                out.data[ravelIndex(out_multi, out.strides)] += delta * delta;
            }

            for (out.data, 0..) |*value, flat| {
                unravelIndexInto(flat, out_shape, out_multi);
                if (keepdims) {
                    @memcpy(mean_multi, out_multi);
                } else {
                    for (out_multi[0..axis], 0..) |coord, i| mean_multi[i] = coord;
                    for (out_multi[axis..], axis + 1..) |coord, i| mean_multi[i] = coord;
                }
                mean_multi[axis] = 0;
                const count = mean_result.counts.data[ravelIndex(mean_multi, mean_result.counts.strides)];
                const denom = castValue(T, count) - correction;
                value.* = if (count == 0 or !(denom > zero(T))) std.math.nan(T) else value.* / denom;
            }
            return out;
        }

        pub fn nanstd(self: Self, axis_opt: ?isize, keepdims: bool, correction: T) ArrayError!Self {
            const out = try self.nanvar(axis_opt, keepdims, correction);
            for (out.data) |*value| value.* = std.math.sqrt(value.*);
            return out;
        }

        fn nanExtreme(self: Self, axis_opt: ?isize, keepdims: bool, comptime better: fn (T, T) bool) ArrayError!Self {
            if (axis_opt == null) {
                var found = false;
                var best = zero(T);
                for (self.data) |value| {
                    if (std.math.isNan(value)) continue;
                    if (!found or better(value, best)) {
                        best = value;
                        found = true;
                    }
                }
                const result = if (found) best else std.math.nan(T);
                if (keepdims) {
                    const out_shape = try keepDimsAllOnes(self.allocator, self.shape.len);
                    defer self.allocator.free(out_shape);
                    return Self.fromSlice(self.allocator, &.{result}, out_shape);
                }
                return Self.fromSlice(self.allocator, &.{result}, &.{});
            }

            const axis = try normalizeDim(axis_opt.?, self.shape.len);
            const out_shape = try self.reducedShape(axis, keepdims);
            defer self.allocator.free(out_shape);
            var out = try Self.empty(self.allocator, out_shape);
            errdefer out.deinit();
            const seen = try self.allocator.alloc(bool, out.data.len);
            defer self.allocator.free(seen);
            @memset(seen, false);

            var in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);
            var out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);

            for (self.data, 0..) |value, flat| {
                if (std.math.isNan(value)) continue;
                unravelIndexInto(flat, self.shape, in_multi);
                if (keepdims) {
                    @memcpy(out_multi, in_multi);
                    out_multi[axis] = 0;
                } else {
                    for (in_multi[0..axis], 0..) |coord, i| out_multi[i] = coord;
                    for (in_multi[axis + 1 ..], axis..) |coord, i| out_multi[i] = coord;
                }
                const out_index = ravelIndex(out_multi, out.strides);
                if (!seen[out_index] or better(value, out.data[out_index])) {
                    out.data[out_index] = value;
                    seen[out_index] = true;
                }
            }
            for (out.data, seen) |*value, was_seen| {
                if (!was_seen) value.* = std.math.nan(T);
            }
            return out;
        }

        pub fn nanmin(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Self {
            ensureFloat(T);
            return self.nanExtreme(axis_opt, keepdims, struct {
                fn f(a: T, b: T) bool {
                    return a < b;
                }
            }.f);
        }

        pub fn nanmax(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Self {
            ensureFloat(T);
            return self.nanExtreme(axis_opt, keepdims, struct {
                fn f(a: T, b: T) bool {
                    return a > b;
                }
            }.f);
        }

        fn quantileFromSorted(sorted_values: []const T, q: T) T {
            const max_index = sorted_values.len - 1;
            const position = q * castValue(T, max_index);
            const lower_float = @floor(position);
            const lower: usize = @intFromFloat(lower_float);
            const upper = @min(lower + 1, max_index);
            const weight = position - lower_float;
            return sorted_values[lower] * (one(T) - weight) + sorted_values[upper] * weight;
        }

        pub fn quantile(self: Self, q: T, axis_opt: ?isize, keepdims: bool) ArrayError!Self {
            ensureFloat(T);
            if (q < zero(T) or q > one(T)) return error.InvalidShape;
            if (self.data.len == 0) return error.EmptyArray;
            if (axis_opt == null) {
                var sorted_values = try self.sort(null);
                defer sorted_values.deinit();
                const result = quantileFromSorted(sorted_values.data, q);
                if (keepdims) {
                    const out_shape = try keepDimsAllOnes(self.allocator, self.shape.len);
                    defer self.allocator.free(out_shape);
                    return Self.fromSlice(self.allocator, &.{result}, out_shape);
                }
                return Self.fromSlice(self.allocator, &.{result}, &.{});
            }

            const axis = try normalizeDim(axis_opt.?, self.shape.len);
            if (self.shape[axis] == 0) return error.EmptyArray;
            const out_shape = try self.reducedShape(axis, keepdims);
            defer self.allocator.free(out_shape);
            var out = try Self.empty(self.allocator, out_shape);
            errdefer out.deinit();
            if (out.data.len == 0) return out;

            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            var in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);
            const scratch = try self.allocator.alloc(T, self.shape[axis]);
            defer self.allocator.free(scratch);

            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, out_shape, out_multi);
                self.mapReducedToInput(axis, keepdims, out_multi, in_multi);
                for (scratch, 0..) |*value, axis_i| {
                    in_multi[axis] = axis_i;
                    value.* = self.data[ravelIndex(in_multi, self.strides)];
                }
                std.sort.insertion(T, scratch, {}, struct {
                    fn lessThan(_: void, a: T, b: T) bool {
                        return lessValue(T, a, b);
                    }
                }.lessThan);
                slot.* = quantileFromSorted(scratch, q);
            }
            return out;
        }

        pub fn percentile(self: Self, p: T, axis_opt: ?isize, keepdims: bool) ArrayError!Self {
            ensureFloat(T);
            return self.quantile(p / castValue(T, 100), axis_opt, keepdims);
        }

        pub fn median(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Self {
            ensureFloat(T);
            return self.quantile(castValue(T, 0.5), axis_opt, keepdims);
        }

        fn checkedBroadcastWeights(self: Self, weights: Self) ArrayError!Self {
            const out_shape = try broadcastShape(self.allocator, self.shape, weights.shape);
            defer self.allocator.free(out_shape);
            if (!std.mem.eql(usize, out_shape, self.shape)) return error.ShapeMismatch;
            return weights.broadcastTo(self.shape);
        }

        pub fn weightedMean(self: Self, weights: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Self {
            ensureFloat(T);
            var full_weights = try self.checkedBroadcastWeights(weights);
            defer full_weights.deinit();

            if (axis_opt == null) {
                var total = zero(T);
                var weight_sum = zero(T);
                for (self.data, full_weights.data) |value, weight| {
                    if (weight < zero(T)) return error.InvalidShape;
                    total += value * weight;
                    weight_sum += weight;
                }
                if (!(weight_sum > zero(T))) return error.InvalidShape;
                const result = total / weight_sum;
                if (keepdims) {
                    const out_shape = try keepDimsAllOnes(self.allocator, self.shape.len);
                    defer self.allocator.free(out_shape);
                    return Self.fromSlice(self.allocator, &.{result}, out_shape);
                }
                return Self.fromSlice(self.allocator, &.{result}, &.{});
            }

            const axis = try normalizeDim(axis_opt.?, self.shape.len);
            const out_shape = try self.reducedShape(axis, keepdims);
            defer self.allocator.free(out_shape);
            var totals = try Self.zeros(self.allocator, out_shape);
            errdefer totals.deinit();
            var weight_sums = try Self.zeros(self.allocator, out_shape);
            defer weight_sums.deinit();
            if (totals.data.len == 0) return totals;

            var in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);
            var out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            for (self.data, full_weights.data, 0..) |value, weight, flat| {
                if (weight < zero(T)) return error.InvalidShape;
                unravelIndexInto(flat, self.shape, in_multi);
                if (keepdims) {
                    @memcpy(out_multi, in_multi);
                    out_multi[axis] = 0;
                } else {
                    for (in_multi[0..axis], 0..) |coord, i| out_multi[i] = coord;
                    for (in_multi[axis + 1 ..], axis..) |coord, i| out_multi[i] = coord;
                }
                const out_index = ravelIndex(out_multi, totals.strides);
                totals.data[out_index] += value * weight;
                weight_sums.data[out_index] += weight;
            }

            for (totals.data, weight_sums.data) |*value, weight_sum| {
                if (!(weight_sum > zero(T))) return error.InvalidShape;
                value.* /= weight_sum;
            }
            return totals;
        }

        pub fn average(self: Self, weights: ?Self, axis_opt: ?isize, keepdims: bool) ArrayError!Self {
            ensureFloat(T);
            if (weights) |w| return self.weightedMean(w, axis_opt, keepdims);
            return self.mean(axis_opt, keepdims);
        }

        pub fn weightedVariance(self: Self, weights: Self, axis_opt: ?isize, keepdims: bool, correction: T) ArrayError!Self {
            ensureFloat(T);
            var full_weights = try self.checkedBroadcastWeights(weights);
            defer full_weights.deinit();

            if (axis_opt == null) {
                var total = zero(T);
                var weight_sum = zero(T);
                for (self.data, full_weights.data) |value, weight| {
                    if (weight < zero(T)) return error.InvalidShape;
                    total += value * weight;
                    weight_sum += weight;
                }
                if (!(weight_sum > zero(T))) return error.InvalidShape;
                const mean_value = total / weight_sum;
                var sq_total = zero(T);
                for (self.data, full_weights.data) |value, weight| {
                    const delta = value - mean_value;
                    sq_total += weight * delta * delta;
                }
                const denom = weight_sum - correction;
                if (!(denom > zero(T))) return error.InvalidShape;
                const result = sq_total / denom;
                if (keepdims) {
                    const out_shape = try keepDimsAllOnes(self.allocator, self.shape.len);
                    defer self.allocator.free(out_shape);
                    return Self.fromSlice(self.allocator, &.{result}, out_shape);
                }
                return Self.fromSlice(self.allocator, &.{result}, &.{});
            }

            const axis = try normalizeDim(axis_opt.?, self.shape.len);
            var means = try self.weightedMean(weights, axis_opt, true);
            defer means.deinit();
            const out_shape = try self.reducedShape(axis, keepdims);
            defer self.allocator.free(out_shape);
            var out = try Self.zeros(self.allocator, out_shape);
            errdefer out.deinit();
            var weight_sums = try Self.zeros(self.allocator, out_shape);
            defer weight_sums.deinit();
            if (out.data.len == 0) return out;

            var in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);
            var out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            var mean_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(mean_multi);

            for (self.data, full_weights.data, 0..) |value, weight, flat| {
                if (weight < zero(T)) return error.InvalidShape;
                unravelIndexInto(flat, self.shape, in_multi);
                @memcpy(mean_multi, in_multi);
                mean_multi[axis] = 0;
                if (keepdims) {
                    @memcpy(out_multi, in_multi);
                    out_multi[axis] = 0;
                } else {
                    for (in_multi[0..axis], 0..) |coord, i| out_multi[i] = coord;
                    for (in_multi[axis + 1 ..], axis..) |coord, i| out_multi[i] = coord;
                }
                const out_index = ravelIndex(out_multi, out.strides);
                const delta = value - means.data[ravelIndex(mean_multi, means.strides)];
                out.data[out_index] += weight * delta * delta;
                weight_sums.data[out_index] += weight;
            }

            for (out.data, weight_sums.data) |*value, weight_sum| {
                const denom = weight_sum - correction;
                if (!(denom > zero(T))) return error.InvalidShape;
                value.* /= denom;
            }
            return out;
        }

        pub fn weightedVar(self: Self, weights: Self, axis_opt: ?isize, keepdims: bool, correction: T) ArrayError!Self {
            return self.weightedVariance(weights, axis_opt, keepdims, correction);
        }

        pub fn weightedStddev(self: Self, weights: Self, axis_opt: ?isize, keepdims: bool, correction: T) ArrayError!Self {
            const out = try self.weightedVariance(weights, axis_opt, keepdims, correction);
            for (out.data) |*value| value.* = std.math.sqrt(value.*);
            return out;
        }

        pub fn weightedStd(self: Self, weights: Self, axis_opt: ?isize, keepdims: bool, correction: T) ArrayError!Self {
            return self.weightedStddev(weights, axis_opt, keepdims, correction);
        }

        fn weightedQuantileFromScratch(self: Self, values: []T, weights: []T, count: usize, q: T) ArrayError!T {
            if (count == 0) return error.EmptyArray;
            const order = try self.allocator.alloc(usize, count);
            defer self.allocator.free(order);
            var total_weight = zero(T);
            for (order, 0..) |*slot, i| {
                if (weights[i] < zero(T)) return error.InvalidShape;
                slot.* = i;
                total_weight += weights[i];
            }
            if (!(total_weight > zero(T))) return error.InvalidShape;
            const Ctx = struct {
                values: []const T,
                fn lessThan(ctx: @This(), a: usize, b: usize) bool {
                    return lessValue(T, ctx.values[a], ctx.values[b]);
                }
            };
            std.sort.insertion(usize, order, Ctx{ .values = values[0..count] }, Ctx.lessThan);
            const threshold = q * total_weight;
            var cumulative = zero(T);
            for (order) |idx| {
                cumulative += weights[idx];
                if (cumulative >= threshold) return values[idx];
            }
            return values[order[count - 1]];
        }

        pub fn weightedQuantile(self: Self, weights: Self, q: T, axis_opt: ?isize, keepdims: bool) ArrayError!Self {
            ensureFloat(T);
            if (q < zero(T) or q > one(T)) return error.InvalidShape;
            if (self.data.len == 0) return error.EmptyArray;
            var full_weights = try self.checkedBroadcastWeights(weights);
            defer full_weights.deinit();

            if (axis_opt == null) {
                const values = try self.allocator.dupe(T, self.data);
                defer self.allocator.free(values);
                const weight_values = try self.allocator.dupe(T, full_weights.data);
                defer self.allocator.free(weight_values);
                const result = try self.weightedQuantileFromScratch(values, weight_values, values.len, q);
                if (keepdims) {
                    const out_shape = try keepDimsAllOnes(self.allocator, self.shape.len);
                    defer self.allocator.free(out_shape);
                    return Self.fromSlice(self.allocator, &.{result}, out_shape);
                }
                return Self.fromSlice(self.allocator, &.{result}, &.{});
            }

            const axis = try normalizeDim(axis_opt.?, self.shape.len);
            if (self.shape[axis] == 0) return error.EmptyArray;
            const out_shape = try self.reducedShape(axis, keepdims);
            defer self.allocator.free(out_shape);
            var out = try Self.empty(self.allocator, out_shape);
            errdefer out.deinit();
            if (out.data.len == 0) return out;

            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            var in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);
            const values = try self.allocator.alloc(T, self.shape[axis]);
            defer self.allocator.free(values);
            const weight_values = try self.allocator.alloc(T, self.shape[axis]);
            defer self.allocator.free(weight_values);

            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, out_shape, out_multi);
                self.mapReducedToInput(axis, keepdims, out_multi, in_multi);
                for (0..self.shape[axis]) |axis_i| {
                    in_multi[axis] = axis_i;
                    const source_index = ravelIndex(in_multi, self.strides);
                    values[axis_i] = self.data[source_index];
                    weight_values[axis_i] = full_weights.data[source_index];
                }
                slot.* = try self.weightedQuantileFromScratch(values, weight_values, self.shape[axis], q);
            }
            return out;
        }

        pub fn weightedMedian(self: Self, weights: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Self {
            ensureFloat(T);
            return self.weightedQuantile(weights, castValue(T, 0.5), axis_opt, keepdims);
        }

        pub fn nanquantile(self: Self, q: T, axis_opt: ?isize, keepdims: bool) ArrayError!Self {
            ensureFloat(T);
            if (q < zero(T) or q > one(T)) return error.InvalidShape;
            if (self.data.len == 0) return error.EmptyArray;
            if (axis_opt == null) {
                const scratch = try self.allocator.alloc(T, self.data.len);
                defer self.allocator.free(scratch);
                var count: usize = 0;
                for (self.data) |value| {
                    if (std.math.isNan(value)) continue;
                    scratch[count] = value;
                    count += 1;
                }
                std.sort.insertion(T, scratch[0..count], {}, struct {
                    fn lessThan(_: void, a: T, b: T) bool {
                        return lessValue(T, a, b);
                    }
                }.lessThan);
                const result = if (count == 0) std.math.nan(T) else quantileFromSorted(scratch[0..count], q);
                if (keepdims) {
                    const out_shape = try keepDimsAllOnes(self.allocator, self.shape.len);
                    defer self.allocator.free(out_shape);
                    return Self.fromSlice(self.allocator, &.{result}, out_shape);
                }
                return Self.fromSlice(self.allocator, &.{result}, &.{});
            }

            const axis = try normalizeDim(axis_opt.?, self.shape.len);
            if (self.shape[axis] == 0) return error.EmptyArray;
            const out_shape = try self.reducedShape(axis, keepdims);
            defer self.allocator.free(out_shape);
            var out = try Self.empty(self.allocator, out_shape);
            errdefer out.deinit();
            if (out.data.len == 0) return out;

            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            var in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);
            const scratch = try self.allocator.alloc(T, self.shape[axis]);
            defer self.allocator.free(scratch);

            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, out_shape, out_multi);
                self.mapReducedToInput(axis, keepdims, out_multi, in_multi);
                var count: usize = 0;
                for (0..self.shape[axis]) |axis_i| {
                    in_multi[axis] = axis_i;
                    const value = self.data[ravelIndex(in_multi, self.strides)];
                    if (std.math.isNan(value)) continue;
                    scratch[count] = value;
                    count += 1;
                }
                std.sort.insertion(T, scratch[0..count], {}, struct {
                    fn lessThan(_: void, a: T, b: T) bool {
                        return lessValue(T, a, b);
                    }
                }.lessThan);
                slot.* = if (count == 0) std.math.nan(T) else quantileFromSorted(scratch[0..count], q);
            }
            return out;
        }

        pub fn nanpercentile(self: Self, p: T, axis_opt: ?isize, keepdims: bool) ArrayError!Self {
            ensureFloat(T);
            return self.nanquantile(p / castValue(T, 100), axis_opt, keepdims);
        }

        pub fn nanmedian(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Self {
            ensureFloat(T);
            return self.nanquantile(castValue(T, 0.5), axis_opt, keepdims);
        }

        fn observationValue(self: Self, variable: usize, observation: usize, rowvar: bool) T {
            if (rowvar) return self.data[variable * self.shape[1] + observation];
            return self.data[observation * self.shape[1] + variable];
        }

        pub fn cov(self: Self, rowvar: bool, correction: T) ArrayError!Self {
            ensureFloat(T);
            if (self.data.len == 0) return error.EmptyArray;
            if (self.shape.len == 1) {
                const observations = self.data.len;
                const denom = castValue(T, observations) - correction;
                if (!(denom > zero(T))) return error.InvalidShape;
                return self.variance(null, false, correction);
            }
            if (self.shape.len != 2) return error.NonMatrixArray;
            const variables = if (rowvar) self.shape[0] else self.shape[1];
            const observations = if (rowvar) self.shape[1] else self.shape[0];
            if (variables == 0 or observations == 0) return error.EmptyArray;
            const denom = castValue(T, observations) - correction;
            if (!(denom > zero(T))) return error.InvalidShape;

            const means = try self.allocator.alloc(T, variables);
            defer self.allocator.free(means);
            @memset(means, zero(T));
            for (0..variables) |i| {
                for (0..observations) |j| means[i] += self.observationValue(i, j, rowvar);
                means[i] /= castValue(T, observations);
            }

            var out = try Self.empty(self.allocator, &.{ variables, variables });
            errdefer out.deinit();
            for (0..variables) |i| {
                for (0..variables) |j| {
                    var total = zero(T);
                    for (0..observations) |k| {
                        total += (self.observationValue(i, k, rowvar) - means[i]) * (self.observationValue(j, k, rowvar) - means[j]);
                    }
                    out.data[i * variables + j] = total / denom;
                }
            }
            return out;
        }

        pub fn corrcoef(self: Self, rowvar: bool) ArrayError!Self {
            ensureFloat(T);
            if (self.shape.len == 1) {
                if (self.data.len < 2) return error.InvalidShape;
                return Self.fromSlice(self.allocator, &.{one(T)}, &.{});
            }
            var covariance = try self.cov(rowvar, one(T));
            defer covariance.deinit();
            const variables = covariance.shape[0];
            var out = try Self.empty(self.allocator, covariance.shape);
            errdefer out.deinit();
            for (0..variables) |i| {
                for (0..variables) |j| {
                    const denom = std.math.sqrt(covariance.data[i * variables + i] * covariance.data[j * variables + j]);
                    out.data[i * variables + j] = covariance.data[i * variables + j] / denom;
                }
            }
            return out;
        }

        fn observationWeight(weights: Self, observation: usize, observations: usize) ArrayError!T {
            if (weights.data.len == 1) return weights.data[0];
            if (weights.shape.len != 1 or weights.data.len != observations) return error.ShapeMismatch;
            return weights.data[observation];
        }

        pub fn weightedCov(self: Self, weights: Self, rowvar: bool, correction: T) ArrayError!Self {
            ensureFloat(T);
            if (self.data.len == 0) return error.EmptyArray;
            if (self.shape.len == 1) {
                const observations = self.data.len;
                var weight_sum = zero(T);
                var total = zero(T);
                for (self.data, 0..) |value, i| {
                    const weight = try observationWeight(weights, i, observations);
                    if (weight < zero(T)) return error.InvalidShape;
                    total += value * weight;
                    weight_sum += weight;
                }
                const denom = weight_sum - correction;
                if (!(weight_sum > zero(T)) or !(denom > zero(T))) return error.InvalidShape;
                const mean_value = total / weight_sum;
                var sq_total = zero(T);
                for (self.data, 0..) |value, i| {
                    const weight = try observationWeight(weights, i, observations);
                    const delta = value - mean_value;
                    sq_total += weight * delta * delta;
                }
                return Self.fromSlice(self.allocator, &.{sq_total / denom}, &.{});
            }
            if (self.shape.len != 2) return error.NonMatrixArray;
            const variables = if (rowvar) self.shape[0] else self.shape[1];
            const observations = if (rowvar) self.shape[1] else self.shape[0];
            if (variables == 0 or observations == 0) return error.EmptyArray;

            var weight_sum = zero(T);
            for (0..observations) |obs| {
                const weight = try observationWeight(weights, obs, observations);
                if (weight < zero(T)) return error.InvalidShape;
                weight_sum += weight;
            }
            const denom = weight_sum - correction;
            if (!(weight_sum > zero(T)) or !(denom > zero(T))) return error.InvalidShape;

            const means = try self.allocator.alloc(T, variables);
            defer self.allocator.free(means);
            @memset(means, zero(T));
            for (0..variables) |i| {
                for (0..observations) |obs| means[i] += self.observationValue(i, obs, rowvar) * (try observationWeight(weights, obs, observations));
                means[i] /= weight_sum;
            }

            var out = try Self.empty(self.allocator, &.{ variables, variables });
            errdefer out.deinit();
            for (0..variables) |i| {
                for (0..variables) |j| {
                    var total = zero(T);
                    for (0..observations) |obs| {
                        const weight = try observationWeight(weights, obs, observations);
                        total += weight * (self.observationValue(i, obs, rowvar) - means[i]) * (self.observationValue(j, obs, rowvar) - means[j]);
                    }
                    out.data[i * variables + j] = total / denom;
                }
            }
            return out;
        }

        pub fn weightedCorrcoef(self: Self, weights: Self, rowvar: bool) ArrayError!Self {
            ensureFloat(T);
            if (self.shape.len == 1) {
                if (self.data.len < 2) return error.InvalidShape;
                return Self.fromSlice(self.allocator, &.{one(T)}, &.{});
            }
            var covariance = try self.weightedCov(weights, rowvar, one(T));
            defer covariance.deinit();
            const variables = covariance.shape[0];
            var out = try Self.empty(self.allocator, covariance.shape);
            errdefer out.deinit();
            for (0..variables) |i| {
                for (0..variables) |j| {
                    const denom = std.math.sqrt(covariance.data[i * variables + i] * covariance.data[j * variables + j]);
                    out.data[i * variables + j] = covariance.data[i * variables + j] / denom;
                }
            }
            return out;
        }

        pub fn nanCov(self: Self, rowvar: bool, correction: T) ArrayError!Self {
            ensureFloat(T);
            if (self.data.len == 0) return error.EmptyArray;
            if (self.shape.len == 1) {
                var count: usize = 0;
                var total = zero(T);
                for (self.data) |value| {
                    if (std.math.isNan(value)) continue;
                    total += value;
                    count += 1;
                }
                const denom = castValue(T, count) - correction;
                if (count == 0 or !(denom > zero(T))) return error.InvalidShape;
                const mean_value = total / castValue(T, count);
                var sq_total = zero(T);
                for (self.data) |value| {
                    if (std.math.isNan(value)) continue;
                    const delta = value - mean_value;
                    sq_total += delta * delta;
                }
                return Self.fromSlice(self.allocator, &.{sq_total / denom}, &.{});
            }
            if (self.shape.len != 2) return error.NonMatrixArray;
            const variables = if (rowvar) self.shape[0] else self.shape[1];
            const observations = if (rowvar) self.shape[1] else self.shape[0];
            if (variables == 0 or observations == 0) return error.EmptyArray;

            var out = try Self.empty(self.allocator, &.{ variables, variables });
            errdefer out.deinit();
            for (0..variables) |i| {
                for (0..variables) |j| {
                    var count: usize = 0;
                    var sum_i = zero(T);
                    var sum_j = zero(T);
                    for (0..observations) |obs| {
                        const vi = self.observationValue(i, obs, rowvar);
                        const vj = self.observationValue(j, obs, rowvar);
                        if (std.math.isNan(vi) or std.math.isNan(vj)) continue;
                        sum_i += vi;
                        sum_j += vj;
                        count += 1;
                    }
                    const denom = castValue(T, count) - correction;
                    if (count == 0 or !(denom > zero(T))) return error.InvalidShape;
                    const mean_i = sum_i / castValue(T, count);
                    const mean_j = sum_j / castValue(T, count);
                    var total = zero(T);
                    for (0..observations) |obs| {
                        const vi = self.observationValue(i, obs, rowvar);
                        const vj = self.observationValue(j, obs, rowvar);
                        if (std.math.isNan(vi) or std.math.isNan(vj)) continue;
                        total += (vi - mean_i) * (vj - mean_j);
                    }
                    out.data[i * variables + j] = total / denom;
                }
            }
            return out;
        }

        pub fn nanCorrcoef(self: Self, rowvar: bool) ArrayError!Self {
            ensureFloat(T);
            if (self.shape.len == 1) {
                if (self.data.len < 2) return error.InvalidShape;
                return Self.fromSlice(self.allocator, &.{one(T)}, &.{});
            }
            var covariance = try self.nanCov(rowvar, one(T));
            defer covariance.deinit();
            const variables = covariance.shape[0];
            var out = try Self.empty(self.allocator, covariance.shape);
            errdefer out.deinit();
            for (0..variables) |i| {
                for (0..variables) |j| {
                    const denom = std.math.sqrt(covariance.data[i * variables + i] * covariance.data[j * variables + j]);
                    out.data[i * variables + j] = covariance.data[i * variables + j] / denom;
                }
            }
            return out;
        }

        pub fn norm(self: Self, p: T, axis_opt: ?isize, keepdims: bool) ArrayError!Self {
            ensureFloat(T);
            if (p == zero(T)) return error.InvalidShape;
            var abs_t = try self.abs();
            defer abs_t.deinit();

            if (p == one(T)) {
                return abs_t.sum(axis_opt, keepdims);
            }
            if (p == castValue(T, 2)) {
                var squared = try abs_t.mul(abs_t);
                defer squared.deinit();
                const summed = try squared.sum(axis_opt, keepdims);
                for (summed.data) |*v| v.* = std.math.sqrt(v.*);
                return summed;
            }

            var powered = try abs_t.powScalar(p);
            defer powered.deinit();
            var summed = try powered.sum(axis_opt, keepdims);
            defer summed.deinit();
            return summed.powScalar(one(T) / p);
        }

        pub fn cumsum(self: Self) ArrayError!Self {
            ensureNumeric(T);
            const out = try Self.empty(self.allocator, self.shape);
            var acc = zero(T);
            for (self.data, out.data) |v, *slot| {
                acc = addValue(T, acc, v);
                slot.* = acc;
            }
            return out;
        }

        pub fn cumprod(self: Self) ArrayError!Self {
            ensureNumeric(T);
            const out = try Self.empty(self.allocator, self.shape);
            var acc = one(T);
            for (self.data, out.data) |v, *slot| {
                acc = mulValue(T, acc, v);
                slot.* = acc;
            }
            return out;
        }

        fn cumulativeFlat(self: Self, comptime op: fn (T, T) T) ArrayError!Self {
            ensureNumeric(T);
            const out = try Self.empty(self.allocator, self.shape);
            if (self.data.len == 0) return out;
            var acc = self.data[0];
            out.data[0] = acc;
            for (self.data[1..], out.data[1..]) |value, *slot| {
                acc = op(acc, value);
                slot.* = acc;
            }
            return out;
        }

        pub fn cummax(self: Self) ArrayError!Self {
            ensureNumeric(T);
            return self.cumulativeFlat(opCummax);
        }

        pub fn cummin(self: Self) ArrayError!Self {
            ensureNumeric(T);
            return self.cumulativeFlat(opCummin);
        }

        pub fn cumsumAxis(self: Self, axis_index: isize) ArrayError!Self {
            ensureNumeric(T);
            return self.cumulativeAxis(axis_index, zero(T), opAdd);
        }

        pub fn cumprodAxis(self: Self, axis_index: isize) ArrayError!Self {
            ensureNumeric(T);
            return self.cumulativeAxis(axis_index, one(T), opMul);
        }

        pub fn cummaxAxis(self: Self, axis_index: isize) ArrayError!Self {
            ensureNumeric(T);
            return self.cumulativeAxisFromFirst(axis_index, opCummax);
        }

        pub fn cumminAxis(self: Self, axis_index: isize) ArrayError!Self {
            ensureNumeric(T);
            return self.cumulativeAxisFromFirst(axis_index, opCummin);
        }

        pub fn logcumsumexp(self: Self) ArrayError!Self {
            ensureFloat(T);
            return self.cumulativeFlat(opLogAddExp);
        }

        pub fn logcumsumexpAxis(self: Self, axis_index: isize) ArrayError!Self {
            ensureFloat(T);
            return self.cumulativeAxisFromFirst(axis_index, opLogAddExp);
        }

        fn cumulativeAxis(self: Self, axis_index: isize, init_value: T, comptime op: fn (T, T) T) ArrayError!Self {
            if (self.shape.len == 0) return error.InvalidAxis;
            const axis = try normalizeDim(axis_index, self.shape.len);
            var out = try Self.empty(self.allocator, self.shape);
            errdefer out.deinit();
            if (out.data.len == 0) return out;

            var slice_shape = try self.allocator.alloc(usize, self.shape.len - 1);
            defer self.allocator.free(slice_shape);
            for (self.shape[0..axis], 0..) |d, i| slice_shape[i] = d;
            for (self.shape[axis + 1 ..], axis..) |d, i| slice_shape[i] = d;
            const slice_multi = try self.allocator.alloc(usize, slice_shape.len);
            defer self.allocator.free(slice_multi);
            var full_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(full_multi);

            for (0..product(slice_shape)) |slice_flat| {
                unravelIndexInto(slice_flat, slice_shape, slice_multi);
                for (slice_multi[0..axis], 0..) |coord, i| full_multi[i] = coord;
                for (slice_multi[axis..], axis + 1..) |coord, i| full_multi[i] = coord;
                var acc = init_value;
                for (0..self.shape[axis]) |axis_i| {
                    full_multi[axis] = axis_i;
                    const idx = ravelIndex(full_multi, self.strides);
                    acc = op(acc, self.data[idx]);
                    out.data[idx] = acc;
                }
            }
            return out;
        }

        fn cumulativeAxisFromFirst(self: Self, axis_index: isize, comptime op: fn (T, T) T) ArrayError!Self {
            if (self.shape.len == 0) return error.InvalidAxis;
            const axis = try normalizeDim(axis_index, self.shape.len);
            var out = try Self.empty(self.allocator, self.shape);
            errdefer out.deinit();
            if (out.data.len == 0) return out;

            var slice_shape = try self.allocator.alloc(usize, self.shape.len - 1);
            defer self.allocator.free(slice_shape);
            for (self.shape[0..axis], 0..) |d, i| slice_shape[i] = d;
            for (self.shape[axis + 1 ..], axis..) |d, i| slice_shape[i] = d;
            const slice_multi = try self.allocator.alloc(usize, slice_shape.len);
            defer self.allocator.free(slice_multi);
            var full_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(full_multi);

            for (0..product(slice_shape)) |slice_flat| {
                unravelIndexInto(slice_flat, slice_shape, slice_multi);
                for (slice_multi[0..axis], 0..) |coord, i| full_multi[i] = coord;
                for (slice_multi[axis..], axis + 1..) |coord, i| full_multi[i] = coord;
                if (self.shape[axis] == 0) continue;
                full_multi[axis] = 0;
                var acc = self.data[ravelIndex(full_multi, self.strides)];
                out.data[ravelIndex(full_multi, out.strides)] = acc;
                for (1..self.shape[axis]) |axis_i| {
                    full_multi[axis] = axis_i;
                    const idx = ravelIndex(full_multi, self.strides);
                    acc = op(acc, self.data[idx]);
                    out.data[ravelIndex(full_multi, out.strides)] = acc;
                }
            }
            return out;
        }

        pub fn diff(self: Self, axis_index: isize, n: usize) ArrayError!Self {
            ensureNumeric(T);
            if (n == 0) return self.clone();
            var current = try self.diffOnce(axis_index);
            errdefer current.deinit();
            var i: usize = 1;
            while (i < n) : (i += 1) {
                const next = try current.diffOnce(axis_index);
                current.deinit();
                current = next;
            }
            return current;
        }

        fn diffOnce(self: Self, axis_index: isize) ArrayError!Self {
            if (self.shape.len == 0) return error.InvalidAxis;
            const axis = try normalizeDim(axis_index, self.shape.len);
            var out_shape = try self.allocator.dupe(usize, self.shape);
            defer self.allocator.free(out_shape);
            out_shape[axis] = if (self.shape[axis] == 0) 0 else self.shape[axis] - 1;
            var out = try Self.empty(self.allocator, out_shape);
            errdefer out.deinit();
            if (out.data.len == 0) return out;
            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            const lhs_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(lhs_multi);
            var rhs_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(rhs_multi);
            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, out_shape, out_multi);
                @memcpy(lhs_multi, out_multi);
                @memcpy(rhs_multi, out_multi);
                rhs_multi[axis] = out_multi[axis] + 1;
                slot.* = self.data[ravelIndex(rhs_multi, self.strides)] - self.data[ravelIndex(lhs_multi, self.strides)];
            }
            return out;
        }

        pub fn trapezoid(self: Self, x_values: ?Self, dx: T, axis_index: isize) ArrayError!Self {
            ensureFloat(T);
            if (self.shape.len == 0) return error.InvalidAxis;
            const axis = try normalizeDim(axis_index, self.shape.len);
            const axis_len = self.shape[axis];
            if (x_values) |x| {
                if (x.shape.len != 1 or x.data.len != axis_len) return error.ShapeMismatch;
            }

            var out_shape = try self.allocator.alloc(usize, self.shape.len - 1);
            defer self.allocator.free(out_shape);
            for (self.shape[0..axis], 0..) |extent, i| out_shape[i] = extent;
            for (self.shape[axis + 1 ..], axis..) |extent, i| out_shape[i] = extent;

            var out = try Self.zeros(self.allocator, out_shape);
            errdefer out.deinit();
            if (out.data.len == 0 or axis_len < 2) return out;

            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            var in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);

            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, out_shape, out_multi);
                for (out_multi[0..axis], 0..) |coord, i| in_multi[i] = coord;
                for (out_multi[axis..], axis + 1..) |coord, i| in_multi[i] = coord;
                var total = zero(T);
                for (0..axis_len - 1) |axis_i| {
                    in_multi[axis] = axis_i;
                    const left = self.data[ravelIndex(in_multi, self.strides)];
                    in_multi[axis] = axis_i + 1;
                    const right = self.data[ravelIndex(in_multi, self.strides)];
                    const width = if (x_values) |x| x.data[axis_i + 1] - x.data[axis_i] else dx;
                    total += (left + right) * width / castValue(T, 2);
                }
                slot.* = total;
            }
            return out;
        }

        pub fn trapz(self: Self, x_values: ?Self, dx: T, axis_index: isize) ArrayError!Self {
            return self.trapezoid(x_values, dx, axis_index);
        }

        pub fn gradient(self: Self, x_values: ?Self, dx: T, axis_index: isize) ArrayError!Self {
            ensureFloat(T);
            if (self.shape.len == 0) return error.InvalidAxis;
            const axis = try normalizeDim(axis_index, self.shape.len);
            const axis_len = self.shape[axis];
            if (x_values) |x| {
                if (x.shape.len != 1 or x.data.len != axis_len) return error.ShapeMismatch;
            }

            var out = try Self.zeros(self.allocator, self.shape);
            errdefer out.deinit();
            if (out.data.len == 0 or axis_len < 2) return out;

            var slice_shape = try self.allocator.alloc(usize, self.shape.len - 1);
            defer self.allocator.free(slice_shape);
            for (self.shape[0..axis], 0..) |extent, i| slice_shape[i] = extent;
            for (self.shape[axis + 1 ..], axis..) |extent, i| slice_shape[i] = extent;

            const slice_multi = try self.allocator.alloc(usize, slice_shape.len);
            defer self.allocator.free(slice_multi);
            var in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);

            for (0..product(slice_shape)) |slice_flat| {
                unravelIndexInto(slice_flat, slice_shape, slice_multi);
                for (slice_multi[0..axis], 0..) |coord, i| in_multi[i] = coord;
                for (slice_multi[axis..], axis + 1..) |coord, i| in_multi[i] = coord;

                for (0..axis_len) |axis_i| {
                    in_multi[axis] = axis_i;
                    const out_index = ravelIndex(in_multi, out.strides);
                    const lhs_axis = if (axis_i == 0) 0 else axis_i - 1;
                    const rhs_axis = if (axis_i + 1 >= axis_len) axis_len - 1 else axis_i + 1;
                    in_multi[axis] = lhs_axis;
                    const left = self.data[ravelIndex(in_multi, self.strides)];
                    in_multi[axis] = rhs_axis;
                    const right = self.data[ravelIndex(in_multi, self.strides)];
                    const width = if (x_values) |x| x.data[rhs_axis] - x.data[lhs_axis] else dx * castValue(T, rhs_axis - lhs_axis);
                    out.data[out_index] = (right - left) / width;
                }
            }
            return out;
        }

        pub fn argmax(self: Self) ArrayError!usize {
            ensureNumeric(T);
            if (self.data.len == 0) return error.EmptyArray;
            var best: usize = 0;
            for (self.data[1..], 1..) |v, i| {
                if (lessValue(T, self.data[best], v)) best = i;
            }
            return best;
        }

        pub fn argmin(self: Self) ArrayError!usize {
            ensureNumeric(T);
            if (self.data.len == 0) return error.EmptyArray;
            var best: usize = 0;
            for (self.data[1..], 1..) |v, i| {
                if (lessValue(T, v, self.data[best])) best = i;
            }
            return best;
        }

        pub fn argmaxAxis(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Array(usize) {
            ensureNumeric(T);
            return self.argReduce(axis_opt, keepdims, struct {
                fn better(a: T, b: T) bool {
                    return lessValue(T, b, a);
                }
            }.better);
        }

        pub fn argminAxis(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Array(usize) {
            ensureNumeric(T);
            return self.argReduce(axis_opt, keepdims, struct {
                fn better(a: T, b: T) bool {
                    return lessValue(T, a, b);
                }
            }.better);
        }

        pub fn nanargmax(self: Self) ArrayError!usize {
            ensureFloat(T);
            var found = false;
            var best: usize = 0;
            for (self.data, 0..) |value, i| {
                if (opIsNan(value)) continue;
                if (!found or lessValue(T, self.data[best], value)) {
                    best = i;
                    found = true;
                }
            }
            if (!found) return error.EmptyArray;
            return best;
        }

        pub fn nanargmin(self: Self) ArrayError!usize {
            ensureFloat(T);
            var found = false;
            var best: usize = 0;
            for (self.data, 0..) |value, i| {
                if (opIsNan(value)) continue;
                if (!found or lessValue(T, value, self.data[best])) {
                    best = i;
                    found = true;
                }
            }
            if (!found) return error.EmptyArray;
            return best;
        }

        pub fn nanargmaxAxis(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Array(usize) {
            ensureFloat(T);
            return self.nanArgReduce(axis_opt, keepdims, struct {
                fn better(a: T, b: T) bool {
                    return lessValue(T, b, a);
                }
            }.better);
        }

        pub fn nanargminAxis(self: Self, axis_opt: ?isize, keepdims: bool) ArrayError!Array(usize) {
            ensureFloat(T);
            return self.nanArgReduce(axis_opt, keepdims, struct {
                fn better(a: T, b: T) bool {
                    return lessValue(T, a, b);
                }
            }.better);
        }

        fn argReduce(self: Self, axis_opt: ?isize, keepdims: bool, comptime better: fn (T, T) bool) ArrayError!Array(usize) {
            if (self.data.len == 0) return error.EmptyArray;
            if (axis_opt == null) {
                var best: usize = 0;
                for (self.data[1..], 1..) |v, i| {
                    if (better(v, self.data[best])) best = i;
                }
                if (keepdims) {
                    const out_shape = try keepDimsAllOnes(self.allocator, self.shape.len);
                    defer self.allocator.free(out_shape);
                    return Array(usize).fromSlice(self.allocator, &.{best}, out_shape);
                }
                return Array(usize).fromSlice(self.allocator, &.{best}, &.{});
            }

            const axis = try normalizeDim(axis_opt.?, self.shape.len);
            if (self.shape[axis] == 0) return error.EmptyArray;
            const out_shape = try self.reducedShape(axis, keepdims);
            defer self.allocator.free(out_shape);
            var out = try Array(usize).empty(self.allocator, out_shape);
            errdefer out.deinit();
            if (out.data.len == 0) return out;
            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            var in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);

            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, out_shape, out_multi);
                self.mapReducedToInput(axis, keepdims, out_multi, in_multi);
                var best_axis: usize = 0;
                in_multi[axis] = 0;
                var best_value = self.data[ravelIndex(in_multi, self.strides)];
                for (1..self.shape[axis]) |axis_i| {
                    in_multi[axis] = axis_i;
                    const value = self.data[ravelIndex(in_multi, self.strides)];
                    if (better(value, best_value)) {
                        best_value = value;
                        best_axis = axis_i;
                    }
                }
                slot.* = best_axis;
            }
            return out;
        }

        fn nanArgReduce(self: Self, axis_opt: ?isize, keepdims: bool, comptime better: fn (T, T) bool) ArrayError!Array(usize) {
            if (self.data.len == 0) return error.EmptyArray;
            if (axis_opt == null) {
                var found = false;
                var best: usize = 0;
                var best_value: T = undefined;
                for (self.data, 0..) |value, i| {
                    if (opIsNan(value)) continue;
                    if (!found or better(value, best_value)) {
                        best = i;
                        best_value = value;
                        found = true;
                    }
                }
                if (!found) return error.EmptyArray;
                if (keepdims) {
                    const out_shape = try keepDimsAllOnes(self.allocator, self.shape.len);
                    defer self.allocator.free(out_shape);
                    return Array(usize).fromSlice(self.allocator, &.{best}, out_shape);
                }
                return Array(usize).fromSlice(self.allocator, &.{best}, &.{});
            }

            const axis = try normalizeDim(axis_opt.?, self.shape.len);
            if (self.shape[axis] == 0) return error.EmptyArray;
            const out_shape = try self.reducedShape(axis, keepdims);
            defer self.allocator.free(out_shape);
            var out = try Array(usize).empty(self.allocator, out_shape);
            errdefer out.deinit();
            if (out.data.len == 0) return out;
            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            var in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);

            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, out_shape, out_multi);
                self.mapReducedToInput(axis, keepdims, out_multi, in_multi);
                var found = false;
                var best_axis: usize = 0;
                var best_value: T = undefined;
                for (0..self.shape[axis]) |axis_i| {
                    in_multi[axis] = axis_i;
                    const value = self.data[ravelIndex(in_multi, self.strides)];
                    if (opIsNan(value)) continue;
                    if (!found or better(value, best_value)) {
                        best_value = value;
                        best_axis = axis_i;
                        found = true;
                    }
                }
                if (!found) return error.EmptyArray;
                slot.* = best_axis;
            }
            return out;
        }

        pub const TopK = struct {
            values: Self,
            indices: Array(usize),

            pub fn deinit(self: *@This()) void {
                self.values.deinit();
                self.indices.deinit();
                self.* = undefined;
            }
        };

        pub fn topk(self: Self, k: usize, axis_opt: ?isize, largest: bool, sorted: bool) ArrayError!TopK {
            ensureNumeric(T);
            if (self.data.len == 0 and k > 0) return error.EmptyArray;
            if (axis_opt == null) return self.topkFlat(k, largest, sorted);
            return self.topkAxis(k, try normalizeDim(axis_opt.?, self.shape.len), largest, sorted);
        }

        fn topkFlat(self: Self, k: usize, largest: bool, sorted: bool) ArrayError!TopK {
            if (k > self.data.len) return error.InvalidShape;
            const order = try self.allocator.alloc(usize, self.data.len);
            defer self.allocator.free(order);
            for (order, 0..) |*slot, i| slot.* = i;
            const Ctx = struct {
                data: []const T,
                largest: bool,
                fn lessThan(ctx: @This(), a: usize, b: usize) bool {
                    return if (ctx.largest) ctx.data[a] > ctx.data[b] else ctx.data[a] < ctx.data[b];
                }
            };
            std.sort.insertion(usize, order, Ctx{ .data = self.data, .largest = largest }, Ctx.lessThan);
            if (!sorted) std.sort.insertion(usize, order[0..k], {}, struct {
                fn lessThan(_: void, a: usize, b: usize) bool {
                    return a < b;
                }
            }.lessThan);

            var values = try Self.empty(self.allocator, &.{k});
            errdefer values.deinit();
            var indices = try Array(usize).empty(self.allocator, &.{k});
            errdefer indices.deinit();
            for (0..k) |i| {
                const idx = order[i];
                values.data[i] = self.data[idx];
                indices.data[i] = idx;
            }
            return .{ .values = values, .indices = indices };
        }

        fn topkAxis(self: Self, k: usize, axis: usize, largest: bool, sorted: bool) ArrayError!TopK {
            const axis_len = self.shape[axis];
            if (k > axis_len) return error.InvalidShape;
            var out_shape = try self.allocator.dupe(usize, self.shape);
            defer self.allocator.free(out_shape);
            out_shape[axis] = k;
            var values = try Self.empty(self.allocator, out_shape);
            errdefer values.deinit();
            var indices = try Array(usize).empty(self.allocator, out_shape);
            errdefer indices.deinit();

            var slice_shape = try self.allocator.alloc(usize, self.shape.len - 1);
            defer self.allocator.free(slice_shape);
            for (self.shape[0..axis], 0..) |d, i| slice_shape[i] = d;
            for (self.shape[axis + 1 ..], axis..) |d, i| slice_shape[i] = d;
            var slice_multi = try self.allocator.alloc(usize, slice_shape.len);
            defer self.allocator.free(slice_multi);
            var base_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(base_multi);
            var out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            const order = try self.allocator.alloc(usize, axis_len);
            defer self.allocator.free(order);

            const Ctx = struct {
                array: Self,
                axis: usize,
                base_multi: []const usize,
                largest: bool,

                fn valueAt(ctx: @This(), axis_i: usize) T {
                    var offset: usize = 0;
                    for (ctx.array.shape, ctx.array.strides, 0..) |_, stride_value, dim_i| {
                        const coord = if (dim_i == ctx.axis) axis_i else ctx.base_multi[dim_i];
                        offset += coord * stride_value;
                    }
                    return ctx.array.data[offset];
                }

                fn lessThan(ctx: @This(), a: usize, b: usize) bool {
                    const av = ctx.valueAt(a);
                    const bv = ctx.valueAt(b);
                    return if (ctx.largest) av > bv else av < bv;
                }
            };

            for (0..product(slice_shape)) |slice_flat| {
                unravelIndexInto(slice_flat, slice_shape, slice_multi);
                for (slice_multi[0..axis], 0..) |coord, i| base_multi[i] = coord;
                for (slice_multi[axis..], axis + 1..) |coord, i| base_multi[i] = coord;
                for (order, 0..) |*slot, i| slot.* = i;
                std.sort.insertion(usize, order, Ctx{ .array = self, .axis = axis, .base_multi = base_multi, .largest = largest }, Ctx.lessThan);
                if (!sorted) std.sort.insertion(usize, order[0..k], {}, struct {
                    fn lessThan(_: void, a: usize, b: usize) bool {
                        return a < b;
                    }
                }.lessThan);

                for (0..k) |rank_i| {
                    @memcpy(out_multi, base_multi);
                    out_multi[axis] = rank_i;
                    const out_offset = ravelIndex(out_multi, values.strides);
                    const source_axis = order[rank_i];
                    base_multi[axis] = source_axis;
                    values.data[out_offset] = self.data[ravelIndex(base_multi, self.strides)];
                    indices.data[out_offset] = source_axis;
                }
            }
            return .{ .values = values, .indices = indices };
        }

        pub fn matmul(self: Self, other: Self) ArrayError!Self {
            ensureNumeric(T);
            if (self.shape.len == 0 or other.shape.len == 0) return error.NonMatrixArray;
            const lhs_vec = self.shape.len == 1;
            const rhs_vec = other.shape.len == 1;
            const lhs_k = self.shape[self.shape.len - 1];
            const rhs_k = if (rhs_vec) other.shape[0] else other.shape[other.shape.len - 2];
            if (lhs_k != rhs_k) return error.ShapeMismatch;

            if (lhs_vec and rhs_vec) return self.dot(other);

            const lhs_batch = if (lhs_vec) self.shape[0..0] else self.shape[0 .. self.shape.len - 2];
            const rhs_batch = if (rhs_vec) other.shape[0..0] else other.shape[0 .. other.shape.len - 2];
            const batch_shape = try broadcastShape(self.allocator, lhs_batch, rhs_batch);
            defer self.allocator.free(batch_shape);

            const lhs_m: usize = if (lhs_vec) 1 else self.shape[self.shape.len - 2];
            const rhs_n: usize = if (rhs_vec) 1 else other.shape[other.shape.len - 1];
            const out_rank = batch_shape.len + @as(usize, if (lhs_vec or rhs_vec) 1 else 2);
            var out_shape = try self.allocator.alloc(usize, out_rank);
            defer self.allocator.free(out_shape);
            for (batch_shape, 0..) |extent, idx| out_shape[idx] = extent;
            if (lhs_vec) {
                out_shape[batch_shape.len] = rhs_n;
            } else if (rhs_vec) {
                out_shape[batch_shape.len] = lhs_m;
            } else {
                out_shape[batch_shape.len] = lhs_m;
                out_shape[batch_shape.len + 1] = rhs_n;
            }

            var out = try Self.zeros(self.allocator, out_shape);
            errdefer out.deinit();
            if (out.data.len == 0) return out;

            const batch_multi = try self.allocator.alloc(usize, batch_shape.len);
            defer self.allocator.free(batch_multi);
            const batch_count = product(batch_shape);
            var write_index: usize = 0;
            for (0..batch_count) |batch_flat| {
                unravelIndexInto(batch_flat, batch_shape, batch_multi);
                var lhs_batch_offset: usize = 0;
                if (!lhs_vec) {
                    const lhs_batch_rank = self.shape.len - 2;
                    const leading = batch_shape.len - lhs_batch_rank;
                    for (0..lhs_batch_rank) |axis| {
                        const coord = if (self.shape[axis] == 1) 0 else batch_multi[leading + axis];
                        lhs_batch_offset += coord * self.strides[axis];
                    }
                }
                var rhs_batch_offset: usize = 0;
                if (!rhs_vec) {
                    const rhs_batch_rank = other.shape.len - 2;
                    const leading = batch_shape.len - rhs_batch_rank;
                    for (0..rhs_batch_rank) |axis| {
                        const coord = if (other.shape[axis] == 1) 0 else batch_multi[leading + axis];
                        rhs_batch_offset += coord * other.strides[axis];
                    }
                }

                if (lhs_vec) {
                    for (0..rhs_n) |col| {
                        var acc = zero(T);
                        for (0..lhs_k) |inner_i| {
                            const lhs_value = self.data[inner_i * self.strides[0]];
                            const rhs_value = other.data[rhs_batch_offset + inner_i * other.strides[other.shape.len - 2] + col * other.strides[other.shape.len - 1]];
                            acc = addValue(T, acc, mulValue(T, lhs_value, rhs_value));
                        }
                        out.data[write_index] = acc;
                        write_index += 1;
                    }
                } else if (rhs_vec) {
                    for (0..lhs_m) |row| {
                        var acc = zero(T);
                        for (0..lhs_k) |inner_i| {
                            const lhs_value = self.data[lhs_batch_offset + row * self.strides[self.shape.len - 2] + inner_i * self.strides[self.shape.len - 1]];
                            const rhs_value = other.data[inner_i * other.strides[0]];
                            acc = addValue(T, acc, mulValue(T, lhs_value, rhs_value));
                        }
                        out.data[write_index] = acc;
                        write_index += 1;
                    }
                } else {
                    for (0..lhs_m) |row| {
                        for (0..rhs_n) |col| {
                            var acc = zero(T);
                            for (0..lhs_k) |inner_i| {
                                const lhs_value = self.data[lhs_batch_offset + row * self.strides[self.shape.len - 2] + inner_i * self.strides[self.shape.len - 1]];
                                const rhs_value = other.data[rhs_batch_offset + inner_i * other.strides[other.shape.len - 2] + col * other.strides[other.shape.len - 1]];
                                acc = addValue(T, acc, mulValue(T, lhs_value, rhs_value));
                            }
                            out.data[write_index] = acc;
                            write_index += 1;
                        }
                    }
                }
            }
            return out;
        }

        pub fn mm(self: Self, other: Self) ArrayError!Self {
            return self.matmul(other);
        }

        pub fn bmm(self: Self, other: Self) ArrayError!Self {
            ensureNumeric(T);
            if (self.shape.len != 3 or other.shape.len != 3) return error.NonMatrixArray;
            const batch = self.shape[0];
            if (other.shape[0] != batch or self.shape[2] != other.shape[1]) return error.ShapeMismatch;
            const m = self.shape[1];
            const k = self.shape[2];
            const n = other.shape[2];
            var out = try Self.zeros(self.allocator, &.{ batch, m, n });
            for (0..batch) |b| {
                for (0..m) |i| {
                    for (0..n) |j| {
                        var acc = zero(T);
                        for (0..k) |p| acc = addValue(T, acc, mulValue(T, self.data[b * m * k + i * k + p], other.data[b * k * n + p * n + j]));
                        out.data[b * m * n + i * n + j] = acc;
                    }
                }
            }
            return out;
        }

        pub fn matvec(self: Self, vector: Self) ArrayError!Self {
            ensureNumeric(T);
            if (self.shape.len != 2) return error.NonMatrixArray;
            if (vector.shape.len != 1) return error.NonVectorArray;
            if (self.shape[1] != vector.shape[0]) return error.ShapeMismatch;
            const rows = self.shape[0];
            const cols = self.shape[1];
            const out = try Self.empty(self.allocator, &.{rows});
            for (out.data, 0..) |*slot, row| {
                var acc = zero(T);
                for (0..cols) |col| {
                    acc = addValue(
                        T,
                        acc,
                        mulValue(T, self.data[row * self.strides[0] + col * self.strides[1]], vector.data[col * vector.strides[0]]),
                    );
                }
                slot.* = acc;
            }
            return out;
        }

        pub fn dot(self: Self, other: Self) ArrayError!Self {
            ensureNumeric(T);
            if (self.shape.len != 1 or other.shape.len != 1) return error.NonVectorArray;
            if (self.shape[0] != other.shape[0]) return error.ShapeMismatch;
            var acc = zero(T);
            for (self.data, other.data) |a, b| acc = addValue(T, acc, mulValue(T, a, b));
            return Self.fromSlice(self.allocator, &.{acc}, &.{});
        }

        pub fn vdot(self: Self, other: Self) ArrayError!Self {
            ensureNumeric(T);
            if (self.data.len != other.data.len) return error.ShapeMismatch;
            var acc = zero(T);
            for (self.data, other.data) |a, b| acc = addValue(T, acc, mulValue(T, a, b));
            return Self.fromSlice(self.allocator, &.{acc}, &.{});
        }

        pub fn vecdot(self: Self, other: Self, axis_index: isize) ArrayError!Self {
            ensureNumeric(T);
            var product_out = try self.mul(other);
            defer product_out.deinit();
            return product_out.sum(axis_index, false);
        }

        pub fn inner(self: Self, other: Self) ArrayError!Self {
            ensureNumeric(T);
            if (self.shape.len == 0 or other.shape.len == 0) return self.mul(other);
            const lhs_contract = self.shape[self.shape.len - 1];
            const rhs_contract = other.shape[other.shape.len - 1];
            if (lhs_contract != rhs_contract) return error.ShapeMismatch;

            const out_rank = self.shape.len + other.shape.len - 2;
            const out_shape = try self.allocator.alloc(usize, out_rank);
            defer self.allocator.free(out_shape);
            var write: usize = 0;
            for (self.shape[0 .. self.shape.len - 1]) |extent| {
                out_shape[write] = extent;
                write += 1;
            }
            for (other.shape[0 .. other.shape.len - 1]) |extent| {
                out_shape[write] = extent;
                write += 1;
            }

            var out = try Self.empty(self.allocator, out_shape);
            errdefer out.deinit();
            if (out.data.len == 0) return out;
            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);

            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, out_shape, out_multi);
                var lhs_base: usize = 0;
                for (self.shape[0 .. self.shape.len - 1], self.strides[0 .. self.shape.len - 1], 0..) |_, stride_value, i| {
                    lhs_base += out_multi[i] * stride_value;
                }
                var rhs_base: usize = 0;
                const rhs_start = self.shape.len - 1;
                for (other.shape[0 .. other.shape.len - 1], other.strides[0 .. other.shape.len - 1], 0..) |_, stride_value, i| {
                    rhs_base += out_multi[rhs_start + i] * stride_value;
                }
                var acc = zero(T);
                for (0..lhs_contract) |axis_i| {
                    acc = addValue(
                        T,
                        acc,
                        mulValue(
                            T,
                            self.data[lhs_base + axis_i * self.strides[self.shape.len - 1]],
                            other.data[rhs_base + axis_i * other.strides[other.shape.len - 1]],
                        ),
                    );
                }
                slot.* = acc;
            }
            return out;
        }

        pub fn outer(self: Self, other: Self) ArrayError!Self {
            ensureNumeric(T);
            if (self.shape.len != 1 or other.shape.len != 1) return error.NonVectorArray;
            const out = try Self.empty(self.allocator, &.{ self.shape[0], other.shape[0] });
            for (0..self.shape[0]) |i| {
                for (0..other.shape[0]) |j| {
                    out.data[i * other.shape[0] + j] = mulValue(T, self.data[i], other.data[j]);
                }
            }
            return out;
        }

        pub fn cross(self: Self, other: Self, axis_index: isize) ArrayError!Self {
            ensureNumeric(T);
            const out_shape = try broadcastShape(self.allocator, self.shape, other.shape);
            defer self.allocator.free(out_shape);
            const axis = try normalizeDim(axis_index, out_shape.len);
            if (out_shape[axis] != 3) return error.ShapeMismatch;
            if ((broadcastAxisExtent(out_shape.len, self.shape, axis) orelse return error.ShapeMismatch) != 3) return error.ShapeMismatch;
            if ((broadcastAxisExtent(out_shape.len, other.shape, axis) orelse return error.ShapeMismatch) != 3) return error.ShapeMismatch;

            var out = try Self.empty(self.allocator, out_shape);
            errdefer out.deinit();
            if (out.data.len == 0) return out;
            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);

            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, out_shape, out_multi);
                const component = out_multi[axis];
                out_multi[axis] = 0;
                const a0 = self.data[broadcastOffset(out_multi, out_shape.len, self.shape, self.strides)];
                const b0 = other.data[broadcastOffset(out_multi, out_shape.len, other.shape, other.strides)];
                out_multi[axis] = 1;
                const a1 = self.data[broadcastOffset(out_multi, out_shape.len, self.shape, self.strides)];
                const b1 = other.data[broadcastOffset(out_multi, out_shape.len, other.shape, other.strides)];
                out_multi[axis] = 2;
                const a2 = self.data[broadcastOffset(out_multi, out_shape.len, self.shape, self.strides)];
                const b2 = other.data[broadcastOffset(out_multi, out_shape.len, other.shape, other.strides)];
                out_multi[axis] = component;
                slot.* = switch (component) {
                    0 => a1 * b2 - a2 * b1,
                    1 => a2 * b0 - a0 * b2,
                    else => a0 * b1 - a1 * b0,
                };
            }
            return out;
        }

        pub fn contractAxes(self: Self, other: Self, axes_self: []const usize, axes_other: []const usize) ArrayError!Self {
            ensureNumeric(T);
            if (axes_self.len != axes_other.len) return error.InvalidShape;

            const seen_self = try self.allocator.alloc(bool, self.shape.len);
            defer self.allocator.free(seen_self);
            @memset(seen_self, false);
            const seen_other = try self.allocator.alloc(bool, other.shape.len);
            defer self.allocator.free(seen_other);
            @memset(seen_other, false);

            const contract_shape = try self.allocator.alloc(usize, axes_self.len);
            defer self.allocator.free(contract_shape);
            for (axes_self, axes_other, 0..) |lhs_axis_raw, rhs_axis_raw, i| {
                const lhs_axis = try canonicalAxis(lhs_axis_raw, self.shape.len);
                const rhs_axis = try canonicalAxis(rhs_axis_raw, other.shape.len);
                if (seen_self[lhs_axis] or seen_other[rhs_axis]) return error.InvalidAxis;
                if (self.shape[lhs_axis] != other.shape[rhs_axis]) return error.ShapeMismatch;
                seen_self[lhs_axis] = true;
                seen_other[rhs_axis] = true;
                contract_shape[i] = self.shape[lhs_axis];
            }

            const out_rank = self.shape.len + other.shape.len - axes_self.len * 2;
            const out_shape = try self.allocator.alloc(usize, out_rank);
            defer self.allocator.free(out_shape);
            var out_pos: usize = 0;
            for (self.shape, 0..) |extent, axis| {
                if (!seen_self[axis]) {
                    out_shape[out_pos] = extent;
                    out_pos += 1;
                }
            }
            for (other.shape, 0..) |extent, axis| {
                if (!seen_other[axis]) {
                    out_shape[out_pos] = extent;
                    out_pos += 1;
                }
            }

            var out = try Self.empty(self.allocator, out_shape);
            errdefer out.deinit();
            if (out.data.len == 0) return out;
            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            var self_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(self_multi);
            var other_multi = try self.allocator.alloc(usize, other.shape.len);
            defer self.allocator.free(other_multi);
            const contract_multi = try self.allocator.alloc(usize, contract_shape.len);
            defer self.allocator.free(contract_multi);

            const contract_count = product(contract_shape);
            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, out_shape, out_multi);
                out_pos = 0;
                for (self.shape, 0..) |_, axis| {
                    if (!seen_self[axis]) {
                        self_multi[axis] = out_multi[out_pos];
                        out_pos += 1;
                    }
                }
                for (other.shape, 0..) |_, axis| {
                    if (!seen_other[axis]) {
                        other_multi[axis] = out_multi[out_pos];
                        out_pos += 1;
                    }
                }

                var acc = zero(T);
                for (0..contract_count) |contract_flat| {
                    unravelIndexInto(contract_flat, contract_shape, contract_multi);
                    for (axes_self, axes_other, 0..) |lhs_axis, rhs_axis, contract_axis| {
                        self_multi[lhs_axis] = contract_multi[contract_axis];
                        other_multi[rhs_axis] = contract_multi[contract_axis];
                    }
                    acc = addValue(
                        T,
                        acc,
                        mulValue(T, self.data[ravelIndex(self_multi, self.strides)], other.data[ravelIndex(other_multi, other.strides)]),
                    );
                }
                slot.* = acc;
            }
            return out;
        }

        pub fn diagonal(self: Self, offset: isize) ArrayError!Self {
            if (self.shape.len != 2) return error.NonMatrixArray;
            const rows = self.shape[0];
            const cols = self.shape[1];
            const start_row: usize = if (offset < 0) blk: {
                const offset_abs: usize = @intCast(-offset);
                if (offset_abs >= rows) return Self.empty(self.allocator, &.{0});
                break :blk offset_abs;
            } else 0;
            const start_col: usize = if (offset > 0) blk: {
                const offset_abs: usize = @intCast(offset);
                if (offset_abs >= cols) return Self.empty(self.allocator, &.{0});
                break :blk offset_abs;
            } else 0;
            const count = @min(rows - start_row, cols - start_col);
            const out = try Self.empty(self.allocator, &.{count});
            for (out.data, 0..) |*slot, i| {
                slot.* = self.data[(start_row + i) * cols + start_col + i];
            }
            return out;
        }

        pub fn diag(self: Self, offset: isize) ArrayError!Self {
            if (self.shape.len == 1) return self.diagflat(offset);
            if (self.shape.len == 2) return self.diagonal(offset);
            return error.InvalidShape;
        }

        pub fn diagflat(self: Self, offset: isize) ArrayError!Self {
            var flat = try self.flatten();
            defer flat.deinit();
            const n = flat.data.len;
            const offset_abs: usize = if (offset < 0) @intCast(-offset) else @intCast(offset);
            const matrix_size = n + offset_abs;
            const out = try Self.zeros(self.allocator, &.{ matrix_size, matrix_size });
            const cols = matrix_size;
            for (flat.data, 0..) |value, i| {
                const row = if (offset < 0) i + offset_abs else i;
                const col = if (offset > 0) i + offset_abs else i;
                out.data[row * cols + col] = value;
            }
            return out;
        }

        pub fn trace(self: Self) ArrayError!T {
            return self.traceOffset(0);
        }

        pub fn traceOffset(self: Self, offset: isize) ArrayError!T {
            ensureNumeric(T);
            if (self.shape.len != 2) return error.NonMatrixArray;
            const rows = self.shape[0];
            const cols = self.shape[1];
            const start_row: usize = if (offset < 0) blk: {
                const offset_abs: usize = @intCast(-offset);
                if (offset_abs >= rows) return zero(T);
                break :blk offset_abs;
            } else 0;
            const start_col: usize = if (offset > 0) blk: {
                const offset_abs: usize = @intCast(offset);
                if (offset_abs >= cols) return zero(T);
                break :blk offset_abs;
            } else 0;
            const count = @min(rows - start_row, cols - start_col);
            var total = zero(T);
            for (0..count) |i| total = addValue(T, total, self.data[(start_row + i) * cols + start_col + i]);
            return total;
        }

        pub fn triu(self: Self, diagonal_offset: isize) ArrayError!Self {
            ensureNumeric(T);
            if (self.shape.len != 2) return error.NonMatrixArray;
            const out = try self.clone();
            const rows = self.shape[0];
            const cols = self.shape[1];
            for (0..rows) |r| {
                for (0..cols) |c| {
                    const diag_distance = @as(isize, @intCast(c)) - @as(isize, @intCast(r));
                    if (diag_distance < diagonal_offset) out.data[r * cols + c] = zero(T);
                }
            }
            return out;
        }

        pub fn tril(self: Self, diagonal_offset: isize) ArrayError!Self {
            ensureNumeric(T);
            if (self.shape.len != 2) return error.NonMatrixArray;
            const out = try self.clone();
            const rows = self.shape[0];
            const cols = self.shape[1];
            for (0..rows) |r| {
                for (0..cols) |c| {
                    const diag_distance = @as(isize, @intCast(c)) - @as(isize, @intCast(r));
                    if (diag_distance > diagonal_offset) out.data[r * cols + c] = zero(T);
                }
            }
            return out;
        }

        pub fn softmax(self: Self, axis_index: isize) ArrayError!Self {
            ensureFloat(T);
            const axis = try normalizeDim(axis_index, self.shape.len);
            var max_t = try self.max(@as(isize, @intCast(axis)), true);
            defer max_t.deinit();
            var shifted = try self.sub(max_t);
            defer shifted.deinit();
            var exp_t = try shifted.exp();
            defer exp_t.deinit();
            var denom = try exp_t.sum(@as(isize, @intCast(axis)), true);
            defer denom.deinit();
            return exp_t.div(denom);
        }

        pub const SortResult = struct {
            values: Self,
            indices: Array(usize),

            pub fn deinit(self: *@This()) void {
                self.values.deinit();
                self.indices.deinit();
                self.* = undefined;
            }
        };

        fn sortOrderLess(descending: bool, a: T, b: T) bool {
            return if (descending) lessValue(T, b, a) else lessValue(T, a, b);
        }

        pub fn sort(self: Self, axis_opt: ?isize) ArrayError!Self {
            return self.sortBy(axis_opt, false);
        }

        pub fn sortDescending(self: Self, axis_opt: ?isize) ArrayError!Self {
            return self.sortBy(axis_opt, true);
        }

        pub fn sortBy(self: Self, axis_opt: ?isize, descending: bool) ArrayError!Self {
            var result = try self.sortWithIndices(axis_opt, descending);
            result.indices.deinit();
            return result.values;
        }

        pub fn argsort(self: Self) ArrayError!Array(usize) {
            return self.argsortAxis(null, false);
        }

        pub fn argsortDescending(self: Self) ArrayError!Array(usize) {
            return self.argsortAxis(null, true);
        }

        pub fn argsortAxis(self: Self, axis_opt: ?isize, descending: bool) ArrayError!Array(usize) {
            var result = try self.sortWithIndices(axis_opt, descending);
            result.values.deinit();
            return result.indices;
        }

        pub fn sortWithIndices(self: Self, axis_opt: ?isize, descending: bool) ArrayError!SortResult {
            ensureOrderable(T);
            if (axis_opt == null) {
                var values = try self.flatten();
                errdefer values.deinit();
                var indices = try Array(usize).empty(self.allocator, &.{self.data.len});
                errdefer indices.deinit();
                for (indices.data, 0..) |*slot, i| slot.* = i;
                const Ctx = struct {
                    data: []const T,
                    descending: bool,
                    fn lessThan(ctx: @This(), a: usize, b: usize) bool {
                        return sortOrderLess(ctx.descending, ctx.data[a], ctx.data[b]);
                    }
                };
                std.sort.insertion(usize, indices.data, Ctx{ .data = self.data, .descending = descending }, Ctx.lessThan);
                for (indices.data, values.data) |idx, *slot| slot.* = self.data[idx];
                return .{ .values = values, .indices = indices };
            }

            const axis = try normalizeDim(axis_opt.?, self.shape.len);
            var values = try Self.empty(self.allocator, self.shape);
            errdefer values.deinit();
            var indices = try Array(usize).empty(self.allocator, self.shape);
            errdefer indices.deinit();
            if (values.data.len == 0) return .{ .values = values, .indices = indices };

            const axis_len = self.shape[axis];
            var slice_shape = try self.allocator.alloc(usize, self.shape.len - 1);
            defer self.allocator.free(slice_shape);
            for (self.shape[0..axis], 0..) |d, i| slice_shape[i] = d;
            for (self.shape[axis + 1 ..], axis..) |d, i| slice_shape[i] = d;
            const slice_multi = try self.allocator.alloc(usize, slice_shape.len);
            defer self.allocator.free(slice_multi);
            var base_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(base_multi);
            var out_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(out_multi);
            const order = try self.allocator.alloc(usize, axis_len);
            defer self.allocator.free(order);

            const Ctx = struct {
                array: Self,
                axis: usize,
                base_multi: []const usize,
                descending: bool,

                fn valueAt(ctx: @This(), axis_i: usize) T {
                    var offset: usize = 0;
                    for (ctx.array.shape, ctx.array.strides, 0..) |_, stride_value, dim_i| {
                        const coord = if (dim_i == ctx.axis) axis_i else ctx.base_multi[dim_i];
                        offset += coord * stride_value;
                    }
                    return ctx.array.data[offset];
                }

                fn lessThan(ctx: @This(), a: usize, b: usize) bool {
                    return sortOrderLess(ctx.descending, ctx.valueAt(a), ctx.valueAt(b));
                }
            };

            for (0..product(slice_shape)) |slice_flat| {
                unravelIndexInto(slice_flat, slice_shape, slice_multi);
                for (slice_multi[0..axis], 0..) |coord, i| base_multi[i] = coord;
                for (slice_multi[axis..], axis + 1..) |coord, i| base_multi[i] = coord;
                for (order, 0..) |*slot, i| slot.* = i;
                std.sort.insertion(usize, order, Ctx{ .array = self, .axis = axis, .base_multi = base_multi, .descending = descending }, Ctx.lessThan);

                for (0..axis_len) |rank_i| {
                    @memcpy(out_multi, base_multi);
                    out_multi[axis] = rank_i;
                    const source_axis = order[rank_i];
                    base_multi[axis] = source_axis;
                    const out_offset = ravelIndex(out_multi, values.strides);
                    values.data[out_offset] = self.data[ravelIndex(base_multi, self.strides)];
                    indices.data[out_offset] = source_axis;
                }
            }
            return .{ .values = values, .indices = indices };
        }

        fn partitionLen(self: Self, axis_opt: ?isize) ArrayError!usize {
            if (axis_opt) |axis_index| return self.shape[try normalizeDim(axis_index, self.shape.len)];
            return self.data.len;
        }

        pub fn partition(self: Self, kth: usize, axis_opt: ?isize, descending: bool) ArrayError!Self {
            ensureOrderable(T);
            const len_axis = try self.partitionLen(axis_opt);
            if (kth >= len_axis) return error.InvalidShape;
            // Full sorting is a valid (stronger) partition: the kth item is in the
            // same position it would occupy in a sorted array, and all preceding
            // items compare before all following items. A future kernel can relax
            // this to O(n) selection while keeping the API stable.
            return self.sortBy(axis_opt, descending);
        }

        pub fn argpartition(self: Self, kth: usize, axis_opt: ?isize, descending: bool) ArrayError!Array(usize) {
            ensureOrderable(T);
            const len_axis = try self.partitionLen(axis_opt);
            if (kth >= len_axis) return error.InvalidShape;
            return self.argsortAxis(axis_opt, descending);
        }

        pub const UniqueCounts = struct {
            values: Self,
            counts: Array(usize),

            pub fn deinit(self: *@This()) void {
                self.values.deinit();
                self.counts.deinit();
                self.* = undefined;
            }
        };

        pub fn unique(self: Self) ArrayError!Self {
            if (comptime T != bool) ensureNumeric(T);
            if (self.data.len == 0) return Self.empty(self.allocator, &.{0});
            var flat = try self.flatten();
            defer flat.deinit();
            std.sort.insertion(T, flat.data, {}, struct {
                fn lessThan(_: void, a: T, b: T) bool {
                    return lessValue(T, a, b);
                }
            }.lessThan);
            var count: usize = 1;
            for (flat.data[1..]) |value| if (value != flat.data[count - 1]) {
                flat.data[count] = value;
                count += 1;
            };
            return Self.fromSlice(self.allocator, flat.data[0..count], &.{count});
        }

        pub fn uniqueWithCounts(self: Self) ArrayError!UniqueCounts {
            if (comptime T != bool) ensureNumeric(T);
            if (self.data.len == 0) {
                var values = try Self.empty(self.allocator, &.{0});
                errdefer values.deinit();
                var counts = try Array(usize).empty(self.allocator, &.{0});
                errdefer counts.deinit();
                return .{ .values = values, .counts = counts };
            }

            var flat = try self.flatten();
            defer flat.deinit();
            std.sort.insertion(T, flat.data, {}, struct {
                fn lessThan(_: void, a: T, b: T) bool {
                    return lessValue(T, a, b);
                }
            }.lessThan);

            var distinct: usize = 1;
            var previous = flat.data[0];
            for (flat.data[1..]) |value| {
                if (value != previous) {
                    distinct += 1;
                    previous = value;
                }
            }

            var values = try Self.empty(self.allocator, &.{distinct});
            errdefer values.deinit();
            var counts = try Array(usize).empty(self.allocator, &.{distinct});
            errdefer counts.deinit();

            var write: usize = 0;
            var current = flat.data[0];
            var current_count: usize = 1;
            for (flat.data[1..]) |value| {
                if (value == current) {
                    current_count += 1;
                } else {
                    values.data[write] = current;
                    counts.data[write] = current_count;
                    write += 1;
                    current = value;
                    current_count = 1;
                }
            }
            values.data[write] = current;
            counts.data[write] = current_count;

            return .{ .values = values, .counts = counts };
        }

        fn valueAsIndex(value: T) ArrayError!usize {
            switch (@typeInfo(T)) {
                .int => |info| {
                    if (info.signedness == .signed and value < 0) return error.InvalidShape;
                    return @intCast(value);
                },
                .comptime_int => return @intCast(value),
                else => @compileError("bincount requires an integer array"),
            }
        }

        pub fn bincount(self: Self, minlength: usize) ArrayError!Array(usize) {
            if (comptime @typeInfo(T) != .int) @compileError("bincount requires an integer array");
            var size_out = minlength;
            for (self.data) |value| {
                const idx = try valueAsIndex(value);
                if (idx + 1 > size_out) size_out = idx + 1;
            }
            var out = try Array(usize).zeros(self.allocator, &.{size_out});
            errdefer out.deinit();
            for (self.data) |value| out.data[try valueAsIndex(value)] += 1;
            return out;
        }

        pub fn bincountWeighted(self: Self, comptime W: type, weights: Array(W), minlength: usize) ArrayError!Array(W) {
            if (comptime @typeInfo(T) != .int) @compileError("bincountWeighted requires an integer input array");
            if (comptime !isNumeric(W)) @compileError("bincountWeighted requires numeric weights");
            if (weights.data.len != self.data.len) return error.ShapeMismatch;
            var size_out = minlength;
            for (self.data) |value| {
                const idx = try valueAsIndex(value);
                if (idx + 1 > size_out) size_out = idx + 1;
            }
            var out = try Array(W).zeros(self.allocator, &.{size_out});
            errdefer out.deinit();
            for (self.data, weights.data) |value, weight| out.data[try valueAsIndex(value)] += weight;
            return out;
        }

        pub fn searchsorted(self: Self, values: Self, side: SearchSide) ArrayError!Array(usize) {
            ensureNumeric(T);
            if (self.shape.len != 1) return error.NonVectorArray;
            var out = try Array(usize).empty(self.allocator, values.shape);
            errdefer out.deinit();
            for (values.data, out.data) |needle_value, *slot| {
                var lo: usize = 0;
                var hi: usize = self.data.len;
                while (lo < hi) {
                    const mid = lo + (hi - lo) / 2;
                    const go_right = switch (side) {
                        .left => self.data[mid] < needle_value,
                        .right => self.data[mid] <= needle_value,
                    };
                    if (go_right) lo = mid + 1 else hi = mid;
                }
                slot.* = lo;
            }
            return out;
        }

        pub fn bucketize(self: Self, boundaries: Self, side: SearchSide) ArrayError!Array(usize) {
            return boundaries.searchsorted(self, side);
        }

        pub fn digitize(self: Self, bins: Self, right: bool) ArrayError!Array(usize) {
            return bins.searchsorted(self, if (right) .left else .right);
        }

        pub fn isin(self: Self, test_elements: Self, invert: bool) ArrayError!Array(bool) {
            if (comptime T != bool) ensureNumeric(T);
            var out = try Array(bool).empty(self.allocator, self.shape);
            errdefer out.deinit();
            for (self.data, out.data) |value, *slot| {
                var found = false;
                for (test_elements.data) |candidate| {
                    if (value == candidate) {
                        found = true;
                        break;
                    }
                }
                slot.* = if (invert) !found else found;
            }
            return out;
        }

        pub fn union1d(self: Self, other: Self) ArrayError!Self {
            if (comptime T != bool) ensureNumeric(T);
            const combined = try self.allocator.alloc(T, self.data.len + other.data.len);
            defer self.allocator.free(combined);
            @memcpy(combined[0..self.data.len], self.data);
            @memcpy(combined[self.data.len..], other.data);
            var merged = try Self.fromSlice(self.allocator, combined, &.{combined.len});
            defer merged.deinit();
            return merged.unique();
        }

        pub fn intersect1d(self: Self, other: Self) ArrayError!Self {
            if (comptime T != bool) ensureNumeric(T);
            var left = try self.unique();
            defer left.deinit();
            var mask = try left.isin(other, false);
            defer mask.deinit();
            return left.maskedSelect(mask);
        }

        pub fn setdiff1d(self: Self, other: Self) ArrayError!Self {
            if (comptime T != bool) ensureNumeric(T);
            var left = try self.unique();
            defer left.deinit();
            var mask = try left.isin(other, true);
            defer mask.deinit();
            return left.maskedSelect(mask);
        }

        pub fn setxor1d(self: Self, other: Self) ArrayError!Self {
            if (comptime T != bool) ensureNumeric(T);
            var left_only = try self.setdiff1d(other);
            defer left_only.deinit();
            var right_only = try other.setdiff1d(self);
            defer right_only.deinit();
            return left_only.union1d(right_only);
        }

        pub fn clipArray(self: Self, min_values: Self, max_values: Self) ArrayError!Self {
            ensureNumeric(T);
            var lower = try self.maximum(min_values);
            defer lower.deinit();
            return lower.minimum(max_values);
        }

        pub fn concatenate(allocator: std.mem.Allocator, arrays: []const Self, axis_index: isize) ArrayError!Self {
            if (arrays.len == 0) return error.EmptyArray;
            const rank_count = arrays[0].shape.len;
            const axis = try normalizeDim(axis_index, rank_count);
            var out_shape = try allocator.dupe(usize, arrays[0].shape);
            defer allocator.free(out_shape);
            out_shape[axis] = 0;
            for (arrays) |t| {
                if (t.shape.len != rank_count) return error.ShapeMismatch;
                for (t.shape, 0..) |d, i| {
                    if (i == axis) continue;
                    if (d != arrays[0].shape[i]) return error.ShapeMismatch;
                }
                out_shape[axis] += t.shape[axis];
            }
            const out = try Self.empty(allocator, out_shape);
            if (out.data.len == 0) return out;
            const out_multi = try allocator.alloc(usize, out_shape.len);
            defer allocator.free(out_multi);
            var in_multi = try allocator.alloc(usize, rank_count);
            defer allocator.free(in_multi);
            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, out_shape, out_multi);
                var base: usize = 0;
                var selected: usize = 0;
                while (selected < arrays.len) : (selected += 1) {
                    const next = base + arrays[selected].shape[axis];
                    if (out_multi[axis] < next) break;
                    base = next;
                }
                @memcpy(in_multi, out_multi);
                in_multi[axis] = out_multi[axis] - base;
                slot.* = arrays[selected].data[ravelIndex(in_multi, arrays[selected].strides)];
            }
            return out;
        }

        pub fn cat(allocator: std.mem.Allocator, arrays: []const Self, axis_index: isize) ArrayError!Self {
            return Self.concatenate(allocator, arrays, axis_index);
        }

        pub fn stack(allocator: std.mem.Allocator, arrays: []const Self, axis_index: isize) ArrayError!Self {
            if (arrays.len == 0) return error.EmptyArray;
            const rank_count = arrays[0].shape.len + 1;
            const axis = if (axis_index < 0) blk: {
                const signed_rank: isize = @intCast(rank_count);
                const normalized = signed_rank + axis_index;
                if (normalized < 0 or normalized >= signed_rank) return error.InvalidAxis;
                break :blk @as(usize, @intCast(normalized));
            } else try canonicalAxis(@intCast(axis_index), rank_count);
            const out_shape = try allocator.alloc(usize, rank_count);
            defer allocator.free(out_shape);
            for (arrays[1..]) |t| {
                if (!std.mem.eql(usize, t.shape, arrays[0].shape)) return error.ShapeMismatch;
            }
            for (out_shape, 0..) |*slot, i| {
                slot.* = if (i < axis) arrays[0].shape[i] else if (i == axis) arrays.len else arrays[0].shape[i - 1];
            }
            const out = try Self.empty(allocator, out_shape);
            if (out.data.len == 0) return out;
            const out_multi = try allocator.alloc(usize, out_shape.len);
            defer allocator.free(out_multi);
            var in_multi = try allocator.alloc(usize, arrays[0].shape.len);
            defer allocator.free(in_multi);
            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, out_shape, out_multi);
                const array_index = out_multi[axis];
                for (out_multi[0..axis], 0..) |coord, i| in_multi[i] = coord;
                for (out_multi[axis + 1 ..], axis..) |coord, i| in_multi[i] = coord;
                slot.* = arrays[array_index].data[ravelIndex(in_multi, arrays[array_index].strides)];
            }
            return out;
        }

        fn transformedStack(allocator: std.mem.Allocator, arrays: []const Self, comptime shapeFn: fn (Self, []usize) ArrayError!void, axis_index: isize) ArrayError!Self {
            if (arrays.len == 0) return error.EmptyArray;
            var transformed = try allocator.alloc(Self, arrays.len);
            defer allocator.free(transformed);
            var initialized: usize = 0;
            errdefer {
                for (transformed[0..initialized]) |*part| part.deinit();
            }
            for (arrays, 0..) |input, i| {
                var dims_buf: [4]usize = undefined;
                try shapeFn(input, dims_buf[0..]);
                transformed[i] = try input.reshape(dims_buf[0 .. input.shape.len + 1]);
                initialized += 1;
            }
            var out = try Self.cat(allocator, transformed, axis_index);
            errdefer out.deinit();
            for (transformed[0..initialized]) |*part| part.deinit();
            initialized = 0;
            return out;
        }

        pub fn hstack(allocator: std.mem.Allocator, arrays: []const Self) ArrayError!Self {
            if (arrays.len == 0) return error.EmptyArray;
            const axis: isize = if (arrays[0].shape.len == 1) 0 else 1;
            return Self.cat(allocator, arrays, axis);
        }

        pub fn vstack(allocator: std.mem.Allocator, arrays: []const Self) ArrayError!Self {
            if (arrays.len == 0) return error.EmptyArray;
            if (arrays[0].shape.len != 1) return Self.cat(allocator, arrays, 0);
            return transformedStack(allocator, arrays, struct {
                fn f(input: Self, out: []usize) ArrayError!void {
                    if (input.shape.len != 1) return error.ShapeMismatch;
                    out[0] = 1;
                    out[1] = input.shape[0];
                }
            }.f, 0);
        }

        pub fn columnStack(allocator: std.mem.Allocator, arrays: []const Self) ArrayError!Self {
            if (arrays.len == 0) return error.EmptyArray;
            var transformed = try allocator.alloc(Self, arrays.len);
            defer allocator.free(transformed);
            var initialized: usize = 0;
            errdefer {
                for (transformed[0..initialized]) |*part| part.deinit();
            }
            for (arrays, 0..) |input, i| {
                if (input.shape.len == 1) {
                    transformed[i] = try input.reshape(&.{ input.shape[0], 1 });
                } else if (input.shape.len == 2) {
                    transformed[i] = try input.clone();
                } else {
                    return error.ShapeMismatch;
                }
                initialized += 1;
            }
            var out = try Self.cat(allocator, transformed, 1);
            errdefer out.deinit();
            for (transformed[0..initialized]) |*part| part.deinit();
            initialized = 0;
            return out;
        }

        pub fn dstack(allocator: std.mem.Allocator, arrays: []const Self) ArrayError!Self {
            if (arrays.len == 0) return error.EmptyArray;
            var transformed = try allocator.alloc(Self, arrays.len);
            defer allocator.free(transformed);
            var initialized: usize = 0;
            errdefer {
                for (transformed[0..initialized]) |*part| part.deinit();
            }
            for (arrays, 0..) |input, i| {
                if (input.shape.len == 1) {
                    transformed[i] = try input.reshape(&.{ 1, input.shape[0], 1 });
                } else if (input.shape.len == 2) {
                    transformed[i] = try input.reshape(&.{ input.shape[0], input.shape[1], 1 });
                } else if (input.shape.len == 3) {
                    transformed[i] = try input.clone();
                } else {
                    return error.ShapeMismatch;
                }
                initialized += 1;
            }
            var out = try Self.cat(allocator, transformed, 2);
            errdefer out.deinit();
            for (transformed[0..initialized]) |*part| part.deinit();
            initialized = 0;
            return out;
        }

        pub const HistogramRange = struct { min: T, max: T };

        pub const HistogramResult = struct { counts: Array(usize), edges: Self };

        pub fn histogram(self: Self, bins: usize, range: ?HistogramRange) ArrayError!HistogramResult {
            ensureFloat(T);
            if (bins == 0) return error.InvalidShape;
            if (self.data.len == 0) return error.EmptyArray;
            var min_v: HistogramRange = range orelse .{ .min = self.data[0], .max = self.data[0] };
            if (range == null) {
                for (self.data[1..]) |v| {
                    if (v < min_v.min) min_v.min = v;
                    if (v > min_v.max) min_v.max = v;
                }
            }
            var counts = try Array(usize).zeros(self.allocator, &.{bins});
            errdefer counts.deinit();
            var edges = try Self.linspace(self.allocator, min_v.min, min_v.max, bins + 1);
            errdefer edges.deinit();
            const width = (min_v.max - min_v.min) / castValue(T, bins);
            for (self.data) |v| {
                if (v < min_v.min or v > min_v.max) continue;
                const raw = if (v == min_v.max) bins - 1 else @as(usize, @intFromFloat((v - min_v.min) / width));
                counts.data[raw] += 1;
            }
            return .{ .counts = counts, .edges = edges };
        }

        pub fn toBytes(self: Self, allocator: std.mem.Allocator) ArrayError![]u8 {
            return allocator.dupe(u8, std.mem.sliceAsBytes(self.data));
        }

        pub fn fromBytes(allocator: std.mem.Allocator, bytes: []const u8, dims: []const usize) ArrayError!Self {
            const n = try numelFrom(dims);
            if (bytes.len != n * @sizeOf(T)) return error.InvalidShape;
            const out = try Self.empty(allocator, dims);
            @memcpy(std.mem.sliceAsBytes(out.data), bytes);
            return out;
        }

        pub fn toArchive(self: Self, allocator: std.mem.Allocator) ArrayError![]u8 {
            const header_len = Archive.magic.len + 1 + 1 + 2 + 8 + self.shape.len * 8;
            const data_bytes = std.mem.sliceAsBytes(self.data);
            const out = try allocator.alloc(u8, header_len + data_bytes.len);
            @memcpy(out[0..Archive.magic.len], Archive.magic[0..]);
            var offset: usize = Archive.magic.len;
            out[offset] = Archive.version;
            offset += 1;
            out[offset] = DType.of(T).tag();
            offset += 1;
            std.mem.writeInt(u16, out[offset..][0..2], @intCast(self.shape.len), .little);
            offset += 2;
            std.mem.writeInt(u64, out[offset..][0..8], @intCast(self.data.len), .little);
            offset += 8;
            for (self.shape) |extent| {
                std.mem.writeInt(u64, out[offset..][0..8], @intCast(extent), .little);
                offset += 8;
            }
            @memcpy(out[offset..][0..data_bytes.len], data_bytes);
            return out;
        }

        pub fn saveArchiveToDir(self: Self, dir: std.Io.Dir, io: std.Io, path: []const u8) !void {
            const archive = try self.toArchive(self.allocator);
            defer self.allocator.free(archive);
            var file = try dir.createFile(io, path, .{});
            defer file.close(io);
            try file.writePositionalAll(io, archive, 0);
        }

        pub fn saveArchive(self: Self, io: std.Io, path: []const u8) !void {
            return self.saveArchiveToDir(std.Io.Dir.cwd(), io, path);
        }

        pub fn fromArchive(allocator: std.mem.Allocator, archive: []const u8) ArrayError!Self {
            const min_len = Archive.magic.len + 1 + 1 + 2 + 8;
            if (archive.len < min_len) return error.InvalidShape;
            if (!std.mem.eql(u8, archive[0..Archive.magic.len], Archive.magic[0..])) return error.InvalidShape;
            var offset: usize = Archive.magic.len;
            if (archive[offset] != Archive.version) return error.InvalidShape;
            offset += 1;
            const archived_dtype = DType.fromTag(archive[offset]) orelse return error.InvalidShape;
            if (archived_dtype != DType.of(T)) return error.TypeUnsupported;
            offset += 1;
            const rank_count = std.mem.readInt(u16, archive[offset..][0..2], .little);
            offset += 2;
            const element_count: usize = @intCast(std.mem.readInt(u64, archive[offset..][0..8], .little));
            offset += 8;
            if (archive.len < min_len + @as(usize, rank_count) * 8) return error.InvalidShape;
            const dims = try allocator.alloc(usize, rank_count);
            defer allocator.free(dims);
            for (dims) |*extent| {
                extent.* = @intCast(std.mem.readInt(u64, archive[offset..][0..8], .little));
                offset += 8;
            }
            const n = try numelFrom(dims);
            if (n != element_count) return error.InvalidShape;
            const data_len = n * @sizeOf(T);
            if (archive.len != offset + data_len) return error.InvalidShape;
            return Self.fromBytes(allocator, archive[offset..], dims);
        }

        pub fn loadArchiveFromDir(
            allocator: std.mem.Allocator,
            dir: std.Io.Dir,
            io: std.Io,
            path: []const u8,
            limit: std.Io.Limit,
        ) !Self {
            const archive = try dir.readFileAlloc(io, path, allocator, limit);
            defer allocator.free(archive);
            return Self.fromArchive(allocator, archive);
        }

        pub fn loadArchive(allocator: std.mem.Allocator, io: std.Io, path: []const u8, limit: std.Io.Limit) !Self {
            return Self.loadArchiveFromDir(allocator, std.Io.Dir.cwd(), io, path, limit);
        }

        pub fn print(self: Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
            try writer.print("Array({s}, shape=", .{@typeName(T)});
            try printShape(writer, self.shape);
            try writer.print(", data=", .{});
            try printFlatData(T, writer, self.data);
            try writer.print(")", .{});
        }

        pub fn toOwnedString(self: Self, allocator: std.mem.Allocator) ArrayError![]u8 {
            var aw: std.Io.Writer.Allocating = .init(allocator);
            errdefer aw.deinit();
            self.print(&aw.writer) catch return error.OutOfMemory;
            return aw.toOwnedSlice();
        }
    };
}

pub const NDArray = Array;
pub const NDArrayView = ArrayView;

fn printShape(writer: *std.Io.Writer, shape: []const usize) std.Io.Writer.Error!void {
    try writer.print("(", .{});
    for (shape, 0..) |d, i| {
        if (i != 0) try writer.print(", ", .{});
        try writer.print("{}", .{d});
    }
    if (shape.len == 1) try writer.print(",", .{});
    try writer.print(")", .{});
}

fn printFlatData(comptime T: type, writer: *std.Io.Writer, data: []const T) std.Io.Writer.Error!void {
    try writer.print("[", .{});
    const limit = @min(data.len, 12);
    for (data[0..limit], 0..) |v, i| {
        if (i != 0) try writer.print(", ", .{});
        try writer.print("{}", .{v});
    }
    if (data.len > limit) try writer.print(", ...", .{});
    try writer.print("]", .{});
}

test "array creation, reshape and broadcasting" {
    const gpa = std.testing.allocator;
    var a = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a.deinit();
    var b = try Array(f64).fromSlice(gpa, &.{ 10, 20, 30 }, &.{3});
    defer b.deinit();
    var c = try a.add(b);
    defer c.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, c.shape);
    try std.testing.expectEqualSlices(f64, &.{ 11, 22, 33, 14, 25, 36 }, c.data);
    var flat = try c.flatten();
    defer flat.deinit();
    try std.testing.expectEqualSlices(usize, &.{6}, flat.shape);

    var parts = try c.split(2, 1);
    defer parts.deinit();
    try std.testing.expectEqual(@as(usize, 2), parts.items.len);
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, parts.items[0].shape);
    try std.testing.expectEqualSlices(f64, &.{ 11, 22, 14, 25 }, parts.items[0].data);
    try std.testing.expectEqualSlices(usize, &.{ 2, 1 }, parts.items[1].shape);
    try std.testing.expectEqualSlices(f64, &.{ 33, 36 }, parts.items[1].data);

    var sized_parts = try c.splitWithSizes(&.{ 1, 2 }, 1);
    defer sized_parts.deinit();
    try std.testing.expectEqual(@as(usize, 2), sized_parts.items.len);
    try std.testing.expectEqualSlices(usize, &.{ 2, 1 }, sized_parts.items[0].shape);
    try std.testing.expectEqualSlices(f64, &.{ 11, 14 }, sized_parts.items[0].data);
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, sized_parts.items[1].shape);
    try std.testing.expectEqualSlices(f64, &.{ 22, 33, 25, 36 }, sized_parts.items[1].data);

    var indexed_parts = try c.split_at_indices(&.{ 1, 2 }, 1);
    defer indexed_parts.deinit();
    try std.testing.expectEqual(@as(usize, 3), indexed_parts.items.len);
    try std.testing.expectEqualSlices(f64, &.{ 11, 14 }, indexed_parts.items[0].data);
    try std.testing.expectEqualSlices(f64, &.{ 22, 25 }, indexed_parts.items[1].data);
    try std.testing.expectEqualSlices(f64, &.{ 33, 36 }, indexed_parts.items[2].data);

    var chunks = try c.chunk(2, 0);
    defer chunks.deinit();
    try std.testing.expectEqual(@as(usize, 2), chunks.items.len);
    try std.testing.expectEqualSlices(f64, &.{ 11, 22, 33 }, chunks.items[0].data);
    try std.testing.expectEqualSlices(f64, &.{ 14, 25, 36 }, chunks.items[1].data);
    try std.testing.expectError(error.InvalidShape, c.split(0, 0));
    try std.testing.expectError(error.ShapeMismatch, c.splitWithSizes(&.{ 1, 1 }, 1));
    try std.testing.expectError(error.InvalidShape, c.splitAtIndices(&.{ 2, 1 }, 1));

    var left = try Array(f64).fromSlice(gpa, &.{ 1, 2 }, &.{2});
    defer left.deinit();
    var right = try Array(f64).fromSlice(gpa, &.{ 3, 4 }, &.{2});
    defer right.deinit();
    const vectors = [_]Array(f64){ left, right };
    var h = try Array(f64).hstack(gpa, vectors[0..]);
    defer h.deinit();
    try std.testing.expectEqualSlices(usize, &.{4}, h.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 3, 4 }, h.data);
    var v = try Array(f64).vstack(gpa, vectors[0..]);
    defer v.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, v.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 3, 4 }, v.data);
    var col = try Array(f64).columnStack(gpa, vectors[0..]);
    defer col.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, col.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 2, 4 }, col.data);
    var d = try Array(f64).dstack(gpa, vectors[0..]);
    defer d.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 2 }, d.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 2, 4 }, d.data);
}

test "array binary math wrappers and clamp aliases" {
    const gpa = std.testing.allocator;
    var a = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4 }, &.{ 2, 2 });
    defer a.deinit();
    var b = try Array(f64).fromSlice(gpa, &.{ 10, 20 }, &.{2});
    defer b.deinit();

    var added = try a.add(b);
    defer added.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 11, 22, 13, 24 }, added.data);
    var subbed = try a.sub(b);
    defer subbed.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -9, -18, -7, -16 }, subbed.data);
    var multiplied = try a.mul(b);
    defer multiplied.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 10, 40, 30, 80 }, multiplied.data);
    var divided = try multiplied.div(b);
    defer divided.deinit();
    try std.testing.expectEqualSlices(f64, a.data, divided.data);

    var exponent = try Array(f64).fromSlice(gpa, &.{2}, &.{1});
    defer exponent.deinit();
    var powed = try a.pow(exponent);
    defer powed.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 4, 9, 16 }, powed.data);

    var clamped = try a.clamp(1.5, 3.5);
    defer clamped.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1.5, 2, 3, 3.5 }, clamped.data);
    var clipped = try a.clamp(2, 3);
    defer clipped.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 2, 2, 3, 3 }, clipped.data);
    var clip_min = try a.clipMin(2.5);
    defer clip_min.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 2.5, 2.5, 3, 4 }, clip_min.data);
    var clamp_min = try a.clampMin(2.5);
    defer clamp_min.deinit();
    try std.testing.expectEqualSlices(f64, clip_min.data, clamp_min.data);
    var clip_max = try a.clipMax(2.5);
    defer clip_max.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 2.5, 2.5 }, clip_max.data);
    var clamp_max = try a.clampMax(2.5);
    defer clamp_max.deinit();
    try std.testing.expectEqualSlices(f64, clip_max.data, clamp_max.data);

    var maxed = try a.maximumScalar(2.5);
    defer maxed.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 2.5, 2.5, 3, 4 }, maxed.data);
    var mined = try a.minimumScalar(2.5);
    defer mined.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 2.5, 2.5 }, mined.data);
    var nan_left = try Array(f64).fromSlice(gpa, &.{ std.math.nan(f64), 2, std.math.nan(f64), 4 }, &.{ 2, 2 });
    defer nan_left.deinit();
    var nan_right = try Array(f64).fromSlice(gpa, &.{ 1, std.math.nan(f64), std.math.nan(f64), 5 }, &.{ 2, 2 });
    defer nan_right.deinit();
    var fmaxed = try nan_left.fmax(nan_right);
    defer fmaxed.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1), fmaxed.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2), fmaxed.data[1], 1e-12);
    try std.testing.expect(std.math.isNan(fmaxed.data[2]));
    try std.testing.expectApproxEqAbs(@as(f64, 5), fmaxed.data[3], 1e-12);
    var fmined = try nan_left.fmin(nan_right);
    defer fmined.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1), fmined.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2), fmined.data[1], 1e-12);
    try std.testing.expect(std.math.isNan(fmined.data[2]));
    try std.testing.expectApproxEqAbs(@as(f64, 4), fmined.data[3], 1e-12);
    var fmax_scalar = try nan_left.fmaxScalar(3);
    defer fmax_scalar.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 3), fmax_scalar.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3), fmax_scalar.data[1], 1e-12);
    var fmin_scalar = try nan_left.fminScalar(3);
    defer fmin_scalar.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 3), fmin_scalar.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2), fmin_scalar.data[1], 1e-12);

    var hyp_a = try Array(f64).fromSlice(gpa, &.{ 3, 5 }, &.{2});
    defer hyp_a.deinit();
    var hyp_b = try Array(f64).fromSlice(gpa, &.{ 4, 12 }, &.{2});
    defer hyp_b.deinit();
    var hyp = try hyp_a.hypot(hyp_b);
    defer hyp.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 5, 13 }, hyp.data);
    var hyp_scalar = try hyp_a.hypotScalar(4);
    defer hyp_scalar.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 5), hyp_scalar.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 41)), hyp_scalar.data[1], 1e-12);

    var y = try Array(f64).fromSlice(gpa, &.{ 0, 1 }, &.{2});
    defer y.deinit();
    var x = try Array(f64).fromSlice(gpa, &.{ 1, 1 }, &.{2});
    defer x.deinit();
    var angles = try y.atan2(x);
    defer angles.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0), angles.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.pi / 4.0, angles.data[1], 1e-12);
    var arctan2_angles = try y.arctan2(x);
    defer arctan2_angles.deinit();
    try std.testing.expectEqualSlices(f64, angles.data, arctan2_angles.data);
    var arctan2_scalar = try y.arctan2Scalar(1);
    defer arctan2_scalar.deinit();
    try std.testing.expectEqualSlices(f64, angles.data, arctan2_scalar.data);
    var next_targets = try Array(f64).fromSlice(gpa, &.{ 2, -1 }, &.{2});
    defer next_targets.deinit();
    var next_values = try y.nextAfter(next_targets);
    defer next_values.deinit();
    try std.testing.expect(next_values.data[0] > 0);
    try std.testing.expect(next_values.data[1] < 1);
    var next_scalar = try y.nextafterScalar(2);
    defer next_scalar.deinit();
    try std.testing.expect(next_scalar.data[0] > 0);
    try std.testing.expect(next_scalar.data[1] > 1);

    var magnitudes = try Array(f64).fromSlice(gpa, &.{ -1, 2, -3 }, &.{3});
    defer magnitudes.deinit();
    var signs_for_copy = try Array(f64).fromSlice(gpa, &.{ 4, -5, -6 }, &.{3});
    defer signs_for_copy.deinit();
    var copied = try magnitudes.copysign(signs_for_copy);
    defer copied.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, -2, -3 }, copied.data);
    var copied_scalar = try magnitudes.copysignScalar(-1);
    defer copied_scalar.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -1, -2, -3 }, copied_scalar.data);
    var negated = try magnitudes.negative();
    defer negated.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, -2, 3 }, negated.data);
    var positive_copy = try magnitudes.positive();
    defer positive_copy.deinit();
    try std.testing.expectEqualSlices(f64, magnitudes.data, positive_copy.data);
    var absolute_values = try magnitudes.absolute();
    defer absolute_values.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 3 }, absolute_values.data);
    var fabs_values = try magnitudes.fabs();
    defer fabs_values.deinit();
    try std.testing.expectEqualSlices(f64, absolute_values.data, fabs_values.data);

    var heav = try Array(f64).fromSlice(gpa, &.{ -2, 0, 3 }, &.{3});
    defer heav.deinit();
    var hzero = try Array(f64).fromSlice(gpa, &.{0.5}, &.{1});
    defer hzero.deinit();
    var heav_out = try heav.heaviside(hzero);
    defer heav_out.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 0.5, 1 }, heav_out.data);
    var heav_scalar = try heav.heavisideScalar(0.25);
    defer heav_scalar.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 0.25, 1 }, heav_scalar.data);
    var log_a = try Array(f64).fromSlice(gpa, &.{ 0, 1000 }, &.{2});
    defer log_a.deinit();
    var log_b = try Array(f64).fromSlice(gpa, &.{ 0, 999 }, &.{2});
    defer log_b.deinit();
    var log_add = try log_a.logAddExp(log_b);
    defer log_add.deinit();
    try std.testing.expectApproxEqAbs(std.math.ln2, log_add.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1000) + std.math.log1p(@exp(@as(f64, -1))), log_add.data[1], 1e-12);
    var log_add_scalar = try log_a.logaddexpScalar(0);
    defer log_add_scalar.deinit();
    try std.testing.expectApproxEqAbs(std.math.ln2, log_add_scalar.data[0], 1e-12);
    var log_add2 = try log_a.logAddExp2(log_b);
    defer log_add2.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1), log_add2.data[0], 1e-12);
    var log_add2_scalar = try log_a.logaddexp2Scalar(0);
    defer log_add2_scalar.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1), log_add2_scalar.data[0], 1e-12);
    var xlogy_x = try Array(f64).fromSlice(gpa, &.{ 0, 2 }, &.{2});
    defer xlogy_x.deinit();
    var xlogy_y = try Array(f64).fromSlice(gpa, &.{ 0, std.math.e }, &.{2});
    defer xlogy_y.deinit();
    var xlogy_out = try xlogy_x.xlogy(xlogy_y);
    defer xlogy_out.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 2 }, xlogy_out.data);
    var xlogy_scalar = try xlogy_x.xlogyScalar(std.math.e);
    defer xlogy_scalar.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 2 }, xlogy_scalar.data);

    var fused_end = try Array(f64).fromSlice(gpa, &.{ 11, 22 }, &.{2});
    defer fused_end.deinit();
    var fused_weight = try Array(f64).fromSlice(gpa, &.{ 0, 0.5 }, &.{2});
    defer fused_weight.deinit();
    var lerped = try a.lerp(fused_end, fused_weight);
    defer lerped.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 12, 3, 13 }, lerped.data);
    var lerped_scalar = try a.lerpScalar(fused_end, 0.25);
    defer lerped_scalar.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 3.5, 7, 5, 8.5 }, lerped_scalar.data);
    var fused_input1 = try Array(f64).fromSlice(gpa, &.{ 10, 20 }, &.{2});
    defer fused_input1.deinit();
    var fused_input2 = try Array(f64).fromSlice(gpa, &.{2}, &.{1});
    defer fused_input2.deinit();
    var addcmul_out = try a.addcmul(fused_input1, fused_input2, 0.5);
    defer addcmul_out.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 11, 22, 13, 24 }, addcmul_out.data);
    var addcmul_alias = try a.addCMul(fused_input1, fused_input2, 0.5);
    defer addcmul_alias.deinit();
    try std.testing.expectEqualSlices(f64, addcmul_out.data, addcmul_alias.data);
    var addcdiv_denom = try Array(f64).fromSlice(gpa, &.{ 2, 4 }, &.{2});
    defer addcdiv_denom.deinit();
    var addcdiv_out = try a.addcdiv(fused_input1, addcdiv_denom, 2);
    defer addcdiv_out.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 11, 12, 13, 14 }, addcdiv_out.data);
    var addcdiv_alias = try a.addCDiv(fused_input1, addcdiv_denom, 2);
    defer addcdiv_alias.deinit();
    try std.testing.expectEqualSlices(f64, addcdiv_out.data, addcdiv_alias.data);
    var bad_fused_input = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3 }, &.{3});
    defer bad_fused_input.deinit();
    try std.testing.expectError(error.ShapeMismatch, a.lerp(fused_end, bad_fused_input));

    var ints = try Array(i32).fromSlice(gpa, &.{ -5, 5, 7 }, &.{3});
    defer ints.deinit();
    var divisors = try Array(i32).fromSlice(gpa, &.{ 2, 2, 3 }, &.{3});
    defer divisors.deinit();
    var floor_div = try ints.floorDiv(divisors);
    defer floor_div.deinit();
    try std.testing.expectEqualSlices(i32, &.{ -3, 2, 2 }, floor_div.data);
    var modulo = try ints.mod(divisors);
    defer modulo.deinit();
    try std.testing.expectEqualSlices(i32, &.{ 1, 1, 1 }, modulo.data);
    var rem_scalar = try ints.remainderScalar(4);
    defer rem_scalar.deinit();
    try std.testing.expectEqualSlices(i32, &.{ 3, 1, 3 }, rem_scalar.data);
}

test "array comparison and logical wrappers" {
    const gpa = std.testing.allocator;
    var a = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4 }, &.{ 2, 2 });
    defer a.deinit();
    var b = try Array(f64).fromSlice(gpa, &.{ 1, 0 }, &.{2});
    defer b.deinit();

    var eq_out = try a.equal(b);
    defer eq_out.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false }, eq_out.data);
    var ne_out = try a.notEqual(b);
    defer ne_out.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, ne_out.data);
    var gt_out = try a.greater(b);
    defer gt_out.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, gt_out.data);
    var ge_out = try a.ge(b);
    defer ge_out.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, ge_out.data);
    var lt_out = try a.less(b);
    defer lt_out.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false }, lt_out.data);
    var le_out = try a.le(b);
    defer le_out.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false }, le_out.data);

    var eq_scalar_out = try a.eqScalar(2);
    defer eq_scalar_out.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false }, eq_scalar_out.data);
    var equal_scalar_out = try a.equalScalar(2);
    defer equal_scalar_out.deinit();
    try std.testing.expectEqualSlices(bool, eq_scalar_out.data, equal_scalar_out.data);
    var ge_scalar_out = try a.geScalar(3);
    defer ge_scalar_out.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, true }, ge_scalar_out.data);
    var greater_equal_scalar_out = try a.greaterEqualScalar(3);
    defer greater_equal_scalar_out.deinit();
    try std.testing.expectEqualSlices(bool, ge_scalar_out.data, greater_equal_scalar_out.data);
    var lt_scalar_out = try a.ltScalar(3);
    defer lt_scalar_out.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false }, lt_scalar_out.data);
    var less_scalar_out = try a.lessScalar(3);
    defer less_scalar_out.deinit();
    try std.testing.expectEqualSlices(bool, lt_scalar_out.data, less_scalar_out.data);
    var not_equal_scalar_out = try a.notEqualScalar(2);
    defer not_equal_scalar_out.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, true }, not_equal_scalar_out.data);

    try std.testing.expect(try a.allclose(a, 1e-12, 1e-12));

    var m1 = try Array(bool).fromSlice(gpa, &.{ true, false, true, false }, &.{ 2, 2 });
    defer m1.deinit();
    var m2 = try Array(bool).fromSlice(gpa, &.{ true, true }, &.{2});
    defer m2.deinit();
    var not_out = try m1.logicalNot();
    defer not_out.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, not_out.data);
    var and_out = try m1.logicalAnd(m2);
    defer and_out.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, false }, and_out.data);
    var or_out = try m1.logicalOr(m2);
    defer or_out.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, or_out.data);
    var xor_out = try m1.logicalXor(m2);
    defer xor_out.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, xor_out.data);
    var xor_scalar_out = try m1.logicalXorScalar(true);
    defer xor_scalar_out.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, xor_scalar_out.data);

    var close_target = try Array(f64).fromSlice(gpa, &.{ 1.0, 2.001, 2.9, 4.0 }, &.{ 2, 2 });
    defer close_target.deinit();
    var close_mask = try a.isclose(close_target, 0.0, 0.01);
    defer close_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, close_mask.data);
    try std.testing.expect(!try a.allclose(close_target, 0.0, 0.01));
    var scalar_close = try a.iscloseScalar(2.0, 0.0, 1.0);
    defer scalar_close.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false }, scalar_close.data);
    try std.testing.expect(!try a.allcloseScalar(2.0, 0.0, 1.0));
}

test "array reductions and matmul" {
    const gpa = std.testing.allocator;
    var a = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a.deinit();
    var s0 = try a.sum(0, false);
    defer s0.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 5, 7, 9 }, s0.data);
    var s1 = try a.sum(1, true);
    defer s1.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 1 }, s1.shape);
    try std.testing.expectEqualSlices(f64, &.{ 6, 15 }, s1.data);
    var p0 = try a.prod(0, false);
    defer p0.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 4, 10, 18 }, p0.data);
    var mn = try a.min(null, false);
    defer mn.deinit();
    try std.testing.expectEqualSlices(f64, &.{1}, mn.data);
    var amin_keep = try a.amin(null, true);
    defer amin_keep.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 1 }, amin_keep.shape);
    try std.testing.expectEqualSlices(f64, &.{1}, amin_keep.data);
    var mx = try a.max(1, false);
    defer mx.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 3, 6 }, mx.data);
    var amax_cols = try a.amax(0, false);
    defer amax_cols.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 4, 5, 6 }, amax_cols.data);
    var ptp_cols = try a.ptp(0, false);
    defer ptp_cols.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 3, 3, 3 }, ptp_cols.data);
    var ptp_flat = try a.ptp(null, false);
    defer ptp_flat.deinit();
    try std.testing.expectEqualSlices(f64, &.{5}, ptp_flat.data);
    var cs = try a.cumsum();
    defer cs.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 6, 10, 15, 21 }, cs.data);
    var cp = try a.cumprod();
    defer cp.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 6, 24, 120, 720 }, cp.data);
    var cmax = try a.cummax();
    defer cmax.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 3, 4, 5, 6 }, cmax.data);
    var cmin = try a.cummin();
    defer cmin.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 1, 1, 1, 1, 1 }, cmin.data);
    var log_cumsum_exp = try a.logcumsumexp();
    defer log_cumsum_exp.deinit();
    var running_lse = a.data[0];
    try std.testing.expectApproxEqAbs(running_lse, log_cumsum_exp.data[0], 1e-12);
    for (a.data[1..], log_cumsum_exp.data[1..]) |value, actual| {
        running_lse = @max(running_lse, value) + std.math.log1p(std.math.exp(-@abs(running_lse - value)));
        try std.testing.expectApproxEqAbs(running_lse, actual, 1e-12);
    }
    try std.testing.expectEqual(@as(usize, 5), try a.argmax());
    try std.testing.expectEqual(@as(usize, 0), try a.argmin());
    var arg1 = try a.argmaxAxis(1, false);
    defer arg1.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, arg1.data);
    var t = try a.transpose();
    defer t.deinit();
    var matrix_product = try a.matmul(t);
    defer matrix_product.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, matrix_product.shape);
    try std.testing.expectEqualSlices(f64, &.{ 14, 32, 32, 77 }, matrix_product.data);
}

test "array object generalized matmul semantics" {
    const gpa = std.testing.allocator;
    var v = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3 }, &.{3});
    defer v.deinit();
    var w = try Array(f64).fromSlice(gpa, &.{ 4, 5, 6 }, &.{3});
    defer w.deinit();
    var dot_out = try v.matmul(w);
    defer dot_out.deinit();
    try std.testing.expectEqual(@as(usize, 0), dot_out.shape.len);
    try std.testing.expectEqual(@as(f64, 32), dot_out.data[0]);

    var m = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer m.deinit();
    var mv = try m.matmul(v);
    defer mv.deinit();
    try std.testing.expectEqualSlices(usize, &.{2}, mv.shape);
    try std.testing.expectEqualSlices(f64, &.{ 14, 32 }, mv.data);

    var vm_rhs = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 3, 2 });
    defer vm_rhs.deinit();
    var vm = try v.matmul(vm_rhs);
    defer vm.deinit();
    try std.testing.expectEqualSlices(usize, &.{2}, vm.shape);
    try std.testing.expectEqualSlices(f64, &.{ 22, 28 }, vm.data);
    var vm_rhs_view = try vm_rhs.asView();
    defer vm_rhs_view.deinit();
    var vm_alias = try vm_rhs_view.T_();
    defer vm_alias.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, vm_alias.shape);

    var batch_a = try Array(f64).fromSlice(gpa, &.{
        1, 2, 3, 4,
        5, 6, 7, 8,
    }, &.{ 2, 2, 2 });
    defer batch_a.deinit();
    var batch_b = try Array(f64).fromSlice(gpa, &.{
        1, 0,
        0, 1,
    }, &.{ 1, 2, 2 });
    defer batch_b.deinit();
    var batch_out = try batch_a.matmul(batch_b);
    defer batch_out.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 2 }, batch_out.shape);
    try std.testing.expectEqualSlices(f64, batch_a.data, batch_out.data);

    var left_broadcast = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4 }, &.{ 1, 2, 2 });
    defer left_broadcast.deinit();
    var right_batch = try Array(f64).fromSlice(gpa, &.{
        1, 0, 0, 1,
        2, 0, 0, 2,
    }, &.{ 2, 2, 2 });
    defer right_batch.deinit();
    var broad_out = try left_broadcast.matmul(right_batch);
    defer broad_out.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 2 }, broad_out.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 3, 4, 2, 4, 6, 8 }, broad_out.data);

    var bad_vec = try Array(f64).fromSlice(gpa, &.{ 1, 2 }, &.{2});
    defer bad_vec.deinit();
    try std.testing.expectError(error.ShapeMismatch, v.matmul(bad_vec));
}

test "array contraction and vector algebra helpers" {
    const gpa = std.testing.allocator;
    var a = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a.deinit();
    var x = try Array(f64).fromSlice(gpa, &.{ 10, 20, 30 }, &.{3});
    defer x.deinit();

    var y = try a.matvec(x);
    defer y.deinit();
    try std.testing.expectEqualSlices(usize, &.{2}, y.shape);
    try std.testing.expectEqualSlices(f64, &.{ 140, 320 }, y.data);
    var y_top = try a.matvec(x);
    defer y_top.deinit();
    try std.testing.expectEqualSlices(f64, y.data, y_top.data);
    var a_view = try a.asView();
    defer a_view.deinit();
    var a_t_view = try a_view.T_();
    defer a_t_view.deinit();
    var mm_alias = try a_view.mm(a_t_view);
    defer mm_alias.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, mm_alias.shape);

    var lhs = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3 }, &.{3});
    defer lhs.deinit();
    var rhs = try Array(f64).fromSlice(gpa, &.{ 4, 5, 6 }, &.{3});
    defer rhs.deinit();
    var dot_out = try lhs.dot(rhs);
    defer dot_out.deinit();
    try std.testing.expectEqual(@as(usize, 0), dot_out.shape.len);
    try std.testing.expectEqual(@as(f64, 32), dot_out.data[0]);

    var left_inner = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer left_inner.deinit();
    var right_inner = try Array(f64).fromSlice(gpa, &.{ 10, 20, 30, 40, 50, 60 }, &.{ 2, 3 });
    defer right_inner.deinit();
    var inner_out = try left_inner.inner(right_inner);
    defer inner_out.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, inner_out.shape);
    try std.testing.expectEqualSlices(f64, &.{ 140, 320, 320, 770 }, inner_out.data);

    var vecdot_out = try left_inner.vecdot(right_inner, 1);
    defer vecdot_out.deinit();
    try std.testing.expectEqualSlices(usize, &.{2}, vecdot_out.shape);
    try std.testing.expectEqualSlices(f64, &.{ 140, 770 }, vecdot_out.data);

    var flat_vdot = try left_inner.vdot(right_inner);
    defer flat_vdot.deinit();
    try std.testing.expectEqual(@as(f64, 910), flat_vdot.data[0]);

    var cross_a = try Array(f64).fromSlice(gpa, &.{ 1, 0, 0, 0, 1, 0 }, &.{ 2, 3 });
    defer cross_a.deinit();
    var cross_b = try Array(f64).fromSlice(gpa, &.{ 0, 1, 0 }, &.{ 1, 3 });
    defer cross_b.deinit();
    var cross_out = try cross_a.cross(cross_b, -1);
    defer cross_out.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, cross_out.shape);
    try std.testing.expectEqualSlices(f64, &.{ 0, 0, 1, 0, 0, 0 }, cross_out.data);

    var td_a = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer td_a.deinit();
    var td_b = try Array(f64).fromSlice(gpa, &.{ 7, 8, 9, 10, 11, 12 }, &.{ 3, 2 });
    defer td_b.deinit();
    var td = try td_a.contractAxes(td_b, &.{1}, &.{0});
    defer td.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, td.shape);
    try std.testing.expectEqualSlices(f64, &.{ 58, 64, 139, 154 }, td.data);

    var batch_a = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6, 7, 8 }, &.{ 2, 2, 2 });
    defer batch_a.deinit();
    var batch_b = try Array(f64).fromSlice(gpa, &.{ 1, 0, 0, 1, 2, 0, 0, 2 }, &.{ 2, 2, 2 });
    defer batch_b.deinit();
    var batch_out = try batch_a.bmm(batch_b);
    defer batch_out.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 3, 4, 10, 12, 14, 16 }, batch_out.data);

    var base_matrix = try Array(f64).fromSlice(gpa, &.{ 1, 9, 2, 8, 3, 7, 4, 6, 5, 5 }, &.{ 2, 5 });
    defer base_matrix.deinit();
    var matrix_view = try base_matrix.sliceAxisView(1, .{ .start = 0, .stop = 5, .step = 2 });
    defer matrix_view.deinit();
    var vec = try Array(f64).fromSlice(gpa, &.{ 10, 20, 30 }, &.{3});
    defer vec.deinit();
    var matvec_view = try matrix_view.matvecArray(vec);
    defer matvec_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 140, 340 }, matvec_view.data);
    var rhs_matrix = try Array(f64).fromSlice(gpa, &.{ 1, 0, 0, 1, 2, 1 }, &.{ 3, 2 });
    defer rhs_matrix.deinit();
    var matmul_view = try matrix_view.matmulArray(rhs_matrix);
    defer matmul_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 7, 5, 17, 11 }, matmul_view.data);
    var lhs_view = try matrix_view.select(0, 0);
    defer lhs_view.deinit();
    var rhs_view = try matrix_view.select(0, 1);
    defer rhs_view.deinit();
    var dot_view = try lhs_view.dot(rhs_view);
    defer dot_view.deinit();
    try std.testing.expectEqual(@as(f64, 34), dot_view.data[0]);
    var vdot_view = try matrix_view.vdotArray(a);
    defer vdot_view.deinit();
    try std.testing.expectEqual(@as(f64, 102), vdot_view.data[0]);
    var vecdot_view = try matrix_view.vecdotArray(a, 1);
    defer vecdot_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 14, 88 }, vecdot_view.data);
    var inner_view = try matrix_view.innerArray(a);
    defer inner_view.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, inner_view.shape);
    try std.testing.expectEqualSlices(f64, &.{ 14, 32, 34, 88 }, inner_view.data);
    var outer_view = try lhs_view.outer(rhs_view);
    defer outer_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 7, 6, 5, 14, 12, 10, 21, 18, 15 }, outer_view.data);
    var cross_lhs = try Array(f64).fromSlice(gpa, &.{ 1, 9, 0, 8, 0, 7, 0, 6, 1, 5 }, &.{ 2, 5 });
    defer cross_lhs.deinit();
    var cross_rhs = try Array(f64).fromSlice(gpa, &.{ 0, 9, 1, 8, 0, 7, 0, 6, 0, 5 }, &.{ 2, 5 });
    defer cross_rhs.deinit();
    var cross_lhs_view = try cross_lhs.sliceAxisView(1, .{ .start = 0, .stop = 5, .step = 2 });
    defer cross_lhs_view.deinit();
    var cross_rhs_view = try cross_rhs.sliceAxisView(1, .{ .start = 0, .stop = 5, .step = 2 });
    defer cross_rhs_view.deinit();
    var cross_view = try cross_lhs_view.cross(cross_rhs_view, -1);
    defer cross_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 0, 1, 0, 0, 0 }, cross_view.data);
    var contract_view = try matrix_view.contractAxesArray(td_b, &.{1}, &.{0});
    defer contract_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 58, 64, 158, 176 }, contract_view.data);
    var batch_view = try batch_a.sliceAxisView(0, .{ .start = 0, .stop = 2, .step = 1 });
    defer batch_view.deinit();
    var batch_rhs_view = try batch_b.asView();
    defer batch_rhs_view.deinit();
    var bmm_view = try batch_view.bmm(batch_rhs_view);
    defer bmm_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 3, 4, 10, 12, 14, 16 }, bmm_view.data);
    var bad_vec = try Array(f64).fromSlice(gpa, &.{ 1, 2 }, &.{2});
    defer bad_vec.deinit();
    try std.testing.expectError(error.ShapeMismatch, matrix_view.matvecArray(bad_vec));

    try std.testing.expectError(error.ShapeMismatch, td_a.contractAxes(td_b, &.{0}, &.{0}));
    var bad_cross = try Array(f64).fromSlice(gpa, &.{ 1, 2 }, &.{2});
    defer bad_cross.deinit();
    try std.testing.expectError(error.ShapeMismatch, bad_cross.cross(bad_cross, 0));
}

test "array scipy-like statistics and softmax" {
    const gpa = std.testing.allocator;
    var a = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4 }, &.{4});
    defer a.deinit();
    var mean_value = try a.mean(null, false);
    defer mean_value.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), mean_value.data[0], 1e-12);
    var std_t = try a.stddev(null, false, 0);
    defer std_t.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1.118033988749895), std_t.data[0], 1e-12);
    var mean_top = try a.mean(null, false);
    defer mean_top.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), mean_top.data[0], 1e-12);
    var var_top = try a.variance(null, false, 0);
    defer var_top.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1.25), var_top.data[0], 1e-12);
    var std_top = try a.stddev(null, false, 0);
    defer std_top.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1.118033988749895), std_top.data[0], 1e-12);
    var norm_top = try a.norm(2, null, false);
    defer norm_top.deinit();
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 30)), norm_top.data[0], 1e-12);

    var logits = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 1, 2, 3 }, &.{ 2, 3 });
    defer logits.deinit();
    var probs = try logits.softmax(1);
    defer probs.deinit();
    var row_sums = try probs.sum(1, false);
    defer row_sums.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1), row_sums.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), row_sums.data[1], 1e-12);

    var mask = try Array(bool).fromSlice(gpa, &.{ true, true, false, true }, &.{ 2, 2 });
    defer mask.deinit();
    var all_rows = try mask.allAxis(1, false);
    defer all_rows.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false }, all_rows.data);
    var any_cols = try mask.anyAxis(0, false);
    defer any_cols.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true }, any_cols.data);
}

test "array pytorch numpy shape indexing and layout helpers" {
    const gpa = std.testing.allocator;
    var a = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a.deinit();
    try std.testing.expectEqual(@as(usize, 2), a.ndim());
    try std.testing.expectEqual(@as(usize, 2), a.dim());
    try std.testing.expectEqual(@as(usize, 2), a.rank());
    try std.testing.expectEqual(@as(usize, 2), a.numDims());
    try std.testing.expectEqual(@as(usize, 6), a.numel());
    try std.testing.expectEqual(@as(usize, 6), a.nelement());
    try std.testing.expect(!a.isEmpty());
    try std.testing.expectEqual(@as(usize, 3), try a.size(1));
    try std.testing.expectEqual(@as(usize, 3), try a.shapeAt(-1));
    try std.testing.expectEqual(@as(usize, 3), try a.strideAt(0));
    try std.testing.expectEqual(@as(usize, @sizeOf(f64)), a.elementSize());
    try std.testing.expectEqual(@as(usize, 6 * @sizeOf(f64)), a.nbytes());
    try std.testing.expect(a.is_contiguous());
    try std.testing.expectEqual(@as(f64, 5), try a.at(&.{ 1, 1 }));
    var empty_meta = try Array(f64).zeros(gpa, &.{ 0, 3 });
    defer empty_meta.deinit();
    try std.testing.expect(empty_meta.isEmpty());
    try std.testing.expectEqual(@as(usize, 0), empty_meta.nbytes());
    var new_zeros = try a.newZeros(&.{2});
    defer new_zeros.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 0 }, new_zeros.data);
    var new_full = try a.new_full(&.{ 1, 2 }, 7);
    defer new_full.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 2 }, new_full.shape);
    try std.testing.expectEqualSlices(f64, &.{ 7, 7 }, new_full.data);

    var u = try a.unsqueeze(0);
    defer u.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 3 }, u.shape);
    var u_alias = try a.unsqueeze_dim(-1);
    defer u_alias.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 3, 1 }, u_alias.shape);
    var s2 = try u.squeeze(null);
    defer s2.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, s2.shape);
    var squeezed_alias = try u_alias.squeezeDim(-1);
    defer squeezed_alias.deinit();
    try std.testing.expectEqualSlices(usize, a.shape, squeezed_alias.shape);
    var multi_singletons = try a.reshape(&.{ 1, 2, 1, 3 });
    defer multi_singletons.deinit();
    var squeezed_axes = try multi_singletons.squeeze_axes(&.{ 0, 2 });
    defer squeezed_axes.deinit();
    try std.testing.expectEqualSlices(usize, a.shape, squeezed_axes.shape);
    try std.testing.expectError(error.ShapeMismatch, multi_singletons.squeezeAxes(&.{1}));
    try std.testing.expectError(error.InvalidAxis, multi_singletons.squeezeAxes(&.{ 0, 0 }));

    var p = try a.permute(&.{ 1, 0 });
    defer p.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, p.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 4, 2, 5, 3, 6 }, p.data);

    var n = try a.narrow(1, 1, 2);
    defer n.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, n.shape);
    try std.testing.expectEqualSlices(f64, &.{ 2, 3, 5, 6 }, n.data);
    var unbound_rows = try a.unbind(0);
    defer unbound_rows.deinit();
    try std.testing.expectEqual(@as(usize, 2), unbound_rows.items.len);
    try std.testing.expectEqualSlices(usize, &.{3}, unbound_rows.items[0].shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 3 }, unbound_rows.items[0].data);
    try std.testing.expectEqualSlices(f64, &.{ 4, 5, 6 }, unbound_rows.items[1].data);
    var unbound_cols = try a.unbind(-1);
    defer unbound_cols.deinit();
    try std.testing.expectEqual(@as(usize, 3), unbound_cols.items.len);
    try std.testing.expectEqualSlices(usize, &.{2}, unbound_cols.items[0].shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 4 }, unbound_cols.items[0].data);
    try std.testing.expectEqualSlices(f64, &.{ 3, 6 }, unbound_cols.items[2].data);
    var reshaped = try a.reshape(&.{ 3, 2 });
    defer reshaped.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, reshaped.shape);
    var reshape_template = try Array(f64).empty(gpa, &.{ 1, 6 });
    defer reshape_template.deinit();
    var reshaped_as = try a.reshape_as(reshape_template);
    defer reshaped_as.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 6 }, reshaped_as.shape);
    var viewed = try reshaped.view(&.{ 2, 3 });
    defer viewed.deinit();
    try std.testing.expectEqualSlices(f64, a.data, viewed.data);
    var view_as = try reshaped_as.viewAs(a);
    defer view_as.deinit();
    try std.testing.expectEqualSlices(usize, a.shape, view_as.shape);
    var flat_top = try a.flatten();
    defer flat_top.deinit();
    try std.testing.expectEqualSlices(usize, &.{6}, flat_top.shape);
    var cube = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6, 7, 8 }, &.{ 2, 2, 2 });
    defer cube.deinit();
    var flat_from = try cube.flattenFrom(1);
    defer flat_from.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 4 }, flat_from.shape);
    var flat_range = try cube.flatten_range(0, 1);
    defer flat_range.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 4, 2 }, flat_range.shape);
    var ravel_top = try a.ravel();
    defer ravel_top.deinit();
    try std.testing.expectEqualSlices(f64, flat_top.data, ravel_top.data);
    var scalar = try Array(f64).fromScalar(gpa, 9);
    defer scalar.deinit();
    var scalar_1d = try scalar.atLeast1d();
    defer scalar_1d.deinit();
    try std.testing.expectEqualSlices(usize, &.{1}, scalar_1d.shape);
    try std.testing.expectEqualSlices(f64, &.{9}, scalar_1d.data);
    var scalar_2d = try scalar.atLeast2d();
    defer scalar_2d.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 1 }, scalar_2d.shape);
    try std.testing.expectEqualSlices(f64, &.{9}, scalar_2d.data);
    var scalar_3d = try scalar.atLeast3d();
    defer scalar_3d.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 1, 1 }, scalar_3d.shape);
    try std.testing.expectEqualSlices(f64, &.{9}, scalar_3d.data);
    var matrix_3d = try a.atLeast3d();
    defer matrix_3d.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 3, 1 }, matrix_3d.shape);
    var already_2d = try a.atLeast2d();
    defer already_2d.deinit();
    try std.testing.expectEqualSlices(usize, a.shape, already_2d.shape);
    var transposed_top = try a.transpose();
    defer transposed_top.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 4, 2, 5, 3, 6 }, transposed_top.data);
    var swapped_top = try a.swapaxes(0, 1);
    defer swapped_top.deinit();
    try std.testing.expectEqualSlices(f64, transposed_top.data, swapped_top.data);
    var swapped_dims_top = try a.swap_dims(0, 1);
    defer swapped_dims_top.deinit();
    try std.testing.expectEqualSlices(f64, transposed_top.data, swapped_dims_top.data);
    var moved_top = try u.movedim(0, 2);
    defer moved_top.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 3, 1 }, moved_top.shape);
    var moveaxis_top = try u.moveaxis(0, -1);
    defer moveaxis_top.deinit();
    try std.testing.expectEqualSlices(usize, moved_top.shape, moveaxis_top.shape);
    var moved_many = try cube.move_axes(&.{ 0, 2 }, &.{ 2, 0 });
    defer moved_many.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 2 }, moved_many.shape);
    try std.testing.expectError(error.ShapeMismatch, cube.moveaxes(&.{0}, &.{ 1, 2 }));
    try std.testing.expectError(error.InvalidAxis, cube.moveaxes(&.{ 0, 0 }, &.{ 1, 2 }));
    var selected_top = try a.select(0, 1);
    defer selected_top.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 4, 5, 6 }, selected_top.data);
    var vector_2d = try selected_top.atLeast2d();
    defer vector_2d.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 3 }, vector_2d.shape);
    var vector_3d = try selected_top.atLeast3d();
    defer vector_3d.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 3, 1 }, vector_3d.shape);
    var broadcast_top = try selected_top.broadcastTo(&.{ 2, 3 });
    defer broadcast_top.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 4, 5, 6, 4, 5, 6 }, broadcast_top.data);
    var expanded_top = try selected_top.expand(&.{ 2, 3 });
    defer expanded_top.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, expanded_top.shape);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, expanded_top.strides);
    try std.testing.expectEqual(@as(f64, 6), try expanded_top.get(&.{ 1, 2 }));
    var expanded_top_owned = try expanded_top.toArray();
    defer expanded_top_owned.deinit();
    try std.testing.expectEqualSlices(f64, broadcast_top.data, expanded_top_owned.data);
    var expanded_as_top = try selected_top.expandAs(broadcast_top);
    defer expanded_as_top.deinit();
    try std.testing.expectEqualSlices(usize, broadcast_top.shape, expanded_as_top.shape);
    var expanded_as_alias = try selected_top.expand_as(broadcast_top);
    defer expanded_as_alias.deinit();
    try std.testing.expectEqualSlices(usize, broadcast_top.shape, expanded_as_alias.shape);
    try std.testing.expectError(error.ShapeMismatch, selected_top.expand(&.{ 2, 2 }));
    var repeated_top = try selected_top.repeat(2, 0);
    defer repeated_top.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 4, 4, 5, 5, 6, 6 }, repeated_top.data);
    var tiled_top = try selected_top.tile(&.{2});
    defer tiled_top.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 4, 5, 6, 4, 5, 6 }, tiled_top.data);
    var tiled_short_repeats = try a.tile(&.{2});
    defer tiled_short_repeats.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 6 }, tiled_short_repeats.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6 }, tiled_short_repeats.data);
    var tiled_long_repeats = try selected_top.tile(&.{ 2, 1 });
    defer tiled_long_repeats.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, tiled_long_repeats.shape);
    try std.testing.expectEqualSlices(f64, &.{ 4, 5, 6, 4, 5, 6 }, tiled_long_repeats.data);
}

test "array object style repeat interleave" {
    const gpa = std.testing.allocator;
    var a = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a.deinit();

    var flat_repeats = try Array(usize).fromSlice(gpa, &.{ 0, 2, 1, 3, 0, 1 }, &.{6});
    defer flat_repeats.deinit();
    var flat = try a.repeatInterleave(flat_repeats, null);
    defer flat.deinit();
    try std.testing.expectEqualSlices(usize, &.{7}, flat.shape);
    try std.testing.expectEqualSlices(f64, &.{ 2, 2, 3, 4, 4, 4, 6 }, flat.data);

    var col_repeats = try Array(usize).fromSlice(gpa, &.{ 1, 0, 2 }, &.{3});
    defer col_repeats.deinit();
    var cols = try a.repeatInterleave(col_repeats, 1);
    defer cols.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, cols.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 3, 4, 6, 6 }, cols.data);

    var row_repeats = try Array(usize).fromSlice(gpa, &.{ 2, 0 }, &.{2});
    defer row_repeats.deinit();
    var rows = try a.repeatInterleave(row_repeats, 0);
    defer rows.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, rows.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 3, 1, 2, 3 }, rows.data);

    var scalar_flat = try a.repeatInterleaveScalar(2, null);
    defer scalar_flat.deinit();
    try std.testing.expectEqualSlices(usize, &.{12}, scalar_flat.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6 }, scalar_flat.data);

    var scalar_axis = try a.repeatInterleaveScalar(2, -1);
    defer scalar_axis.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 6 }, scalar_axis.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6 }, scalar_axis.data);

    var zero_axis = try a.repeatInterleaveScalar(0, 1);
    defer zero_axis.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 0 }, zero_axis.shape);
    try std.testing.expectEqual(@as(usize, 0), zero_axis.data.len);

    var view = try a.sliceAxisView(1, .{ .start = 0, .stop = 3, .step = 2 });
    defer view.deinit();
    var view_repeats = try Array(usize).fromSlice(gpa, &.{ 2, 1 }, &.{2});
    defer view_repeats.deinit();
    var view_repeated = try view.repeatInterleave(view_repeats, 1);
    defer view_repeated.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, view_repeated.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 1, 3, 4, 4, 6 }, view_repeated.data);

    var bad_axis_repeats = try Array(usize).fromSlice(gpa, &.{ 1, 2 }, &.{2});
    defer bad_axis_repeats.deinit();
    try std.testing.expectError(error.ShapeMismatch, a.repeatInterleave(bad_axis_repeats, 1));

    var bad_flat_repeats = try Array(usize).fromSlice(gpa, &.{ 1, 1, 1, 1, 1 }, &.{5});
    defer bad_flat_repeats.deinit();
    try std.testing.expectError(error.ShapeMismatch, a.repeatInterleave(bad_flat_repeats, null));
    try std.testing.expectError(error.InvalidAxis, a.repeatInterleaveScalar(1, 2));
}

test "array view materializing shape wrappers" {
    const gpa = std.testing.allocator;
    var a = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6, 7, 8 }, &.{ 2, 4 });
    defer a.deinit();

    var view = try a.sliceAxisView(1, .{ .start = 0, .stop = 4, .step = 2 });
    defer view.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, view.shape);
    try std.testing.expectEqualSlices(usize, &.{ 4, 2 }, view.strides);
    try std.testing.expectEqual(@as(usize, 2), view.dim());
    try std.testing.expectEqual(@as(usize, 2), view.rank());
    try std.testing.expectEqual(@as(usize, 2), view.numDims());
    try std.testing.expectEqual(@as(usize, 4), view.numel());
    try std.testing.expectEqual(@as(usize, 4), view.nelement());
    try std.testing.expect(!view.isEmpty());
    try std.testing.expectEqual(@as(usize, 2), try view.shapeAt(-1));
    try std.testing.expectEqual(@as(usize, 2), try view.strideAt(1));
    try std.testing.expectEqual(@as(usize, @sizeOf(f64)), view.elementSize());
    try std.testing.expectEqual(@as(usize, 4 * @sizeOf(f64)), view.nbytes());
    try std.testing.expect(!view.is_contiguous());
    var view_zeros = try view.zerosLike();
    defer view_zeros.deinit();
    try std.testing.expectEqualSlices(usize, view.shape, view_zeros.shape);
    try std.testing.expectEqualSlices(f64, &.{ 0, 0, 0, 0 }, view_zeros.data);
    var view_full = try view.fullLike(7);
    defer view_full.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 7, 7, 7, 7 }, view_full.data);
    var view_new_ones = try view.new_ones(&.{3});
    defer view_new_ones.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 1, 1 }, view_new_ones.data);
    var contiguous_view = try a.asView();
    defer contiguous_view.deinit();
    var view_shape_template = try Array(f64).empty(gpa, &.{ 4, 2 });
    defer view_shape_template.deinit();
    var view_reshaped = try contiguous_view.reshapeAsArray(view_shape_template);
    defer view_reshaped.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 4, 2 }, view_reshaped.shape);
    var view_back = try view_reshaped.view_as(contiguous_view);
    defer view_back.deinit();
    try std.testing.expectEqualSlices(usize, contiguous_view.shape, view_back.shape);
    var view_flat_from = try contiguous_view.flattenFrom(1);
    defer view_flat_from.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 4 }, view_flat_from.shape);
    var view_flat_range = try contiguous_view.flatten_range(0, 1);
    defer view_flat_range.deinit();
    try std.testing.expectEqualSlices(usize, &.{8}, view_flat_range.shape);
    var view_unsqueezed = try view.unsqueezeDim(0);
    defer view_unsqueezed.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 2 }, view_unsqueezed.shape);
    var view_squeezed = try view_unsqueezed.squeeze_dim(0);
    defer view_squeezed.deinit();
    try std.testing.expectEqualSlices(usize, view.shape, view_squeezed.shape);
    var view_squeezed_axes = try view_unsqueezed.squeezeAxes(&.{0});
    defer view_squeezed_axes.deinit();
    try std.testing.expectEqualSlices(usize, view.shape, view_squeezed_axes.shape);
    var view_moved_many = try view_unsqueezed.moveaxes(&.{ 0, 2 }, &.{ 2, 0 });
    defer view_moved_many.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 1 }, view_moved_many.shape);
    var unbound_view = try view.unbind(1);
    defer unbound_view.deinit();
    try std.testing.expectEqual(@as(usize, 2), unbound_view.items.len);
    try std.testing.expectEqualSlices(usize, &.{2}, unbound_view.items[0].shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 5 }, unbound_view.items[0].data);
    try std.testing.expectEqualSlices(f64, &.{ 3, 7 }, unbound_view.items[1].data);

    var repeated = try view.repeat(2, 1);
    defer repeated.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 4 }, repeated.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 1, 3, 3, 5, 5, 7, 7 }, repeated.data);

    var tiled = try view.tile(&.{ 1, 2 });
    defer tiled.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 4 }, tiled.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 1, 3, 5, 7, 5, 7 }, tiled.data);
    var tiled_short_repeats = try view.tile(&.{2});
    defer tiled_short_repeats.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 4 }, tiled_short_repeats.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 1, 3, 5, 7, 5, 7 }, tiled_short_repeats.data);
    var tiled_long_repeats = try view.tile(&.{ 2, 1, 1 });
    defer tiled_long_repeats.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 2 }, tiled_long_repeats.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 5, 7, 1, 3, 5, 7 }, tiled_long_repeats.data);

    var flipped = try view.flip(1);
    defer flipped.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 3, 1, 7, 5 }, flipped.data);

    var flipped_axes = try view.flipAxes(&.{ 0, 1 });
    defer flipped_axes.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 7, 5, 3, 1 }, flipped_axes.data);

    var rolled = try view.roll(1, 0);
    defer rolled.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 5, 7, 1, 3 }, rolled.data);

    var rolled_axes = try view.rollAxes(&.{ 1, 1 }, &.{ 0, 1 });
    defer rolled_axes.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 7, 5, 3, 1 }, rolled_axes.data);

    var rotated = try view.rot90(1, .{ 0, 1 });
    defer rotated.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, rotated.shape);
    try std.testing.expectEqualSlices(f64, &.{ 3, 7, 1, 5 }, rotated.data);
    var swapped_view = try view.swapDims(0, 1);
    defer swapped_view.deinit();
    var swapped_view_owned = try swapped_view.toArray();
    defer swapped_view_owned.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 5, 3, 7 }, swapped_view_owned.data);
    var moved_view = try view.moveaxis(0, 1);
    defer moved_view.deinit();
    try std.testing.expectEqualSlices(usize, swapped_view.shape, moved_view.shape);

    var padded = try view.padConstant(&.{ 1, 1 }, &.{ 0, 1 }, 0);
    defer padded.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 4 }, padded.shape);
    try std.testing.expectEqualSlices(f64, &.{
        0, 0, 0, 0,
        0, 1, 3, 0,
        0, 5, 7, 0,
    }, padded.data);

    var edge = try view.padEdge(&.{ 1, 1 }, &.{ 0, 0 });
    defer edge.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 3 }, edge.shape);
    try std.testing.expectEqualSlices(f64, &.{
        1, 1, 3,
        1, 1, 3,
        5, 5, 7,
    }, edge.data);

    var reflect = try view.padReflect(&.{ 1, 1 }, &.{ 0, 0 });
    defer reflect.deinit();
    try std.testing.expectEqualSlices(f64, &.{
        7, 5, 7,
        3, 1, 3,
        7, 5, 7,
    }, reflect.data);

    var wrapped = try view.padWrap(&.{ 1, 1 }, &.{ 0, 1 });
    defer wrapped.deinit();
    try std.testing.expectEqualSlices(f64, &.{
        7, 5, 7, 5,
        3, 1, 3, 1,
        7, 5, 7, 5,
    }, wrapped.data);

    var symmetric = try view.padSymmetric(&.{ 1, 1 }, &.{ 0, 0 });
    defer symmetric.deinit();
    try std.testing.expectEqualSlices(f64, &.{
        1, 1, 3,
        1, 1, 3,
        5, 5, 7,
    }, symmetric.data);

    var vector = try a.selectView(0, 1);
    defer vector.deinit();
    var sliced = try vector.slice1d(.{ .start = 1, .stop = 4, .step = 2 });
    defer sliced.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 6, 8 }, sliced.data);

    var parts = try view.split(1, 1);
    defer parts.deinit();
    try std.testing.expectEqual(@as(usize, 2), parts.items.len);
    try std.testing.expectEqualSlices(f64, &.{ 1, 5 }, parts.items[0].data);
    try std.testing.expectEqualSlices(f64, &.{ 3, 7 }, parts.items[1].data);

    var sized_parts = try view.split_with_sizes(&.{ 1, 1 }, 1);
    defer sized_parts.deinit();
    try std.testing.expectEqual(@as(usize, 2), sized_parts.items.len);
    try std.testing.expectEqualSlices(f64, &.{ 1, 5 }, sized_parts.items[0].data);
    try std.testing.expectEqualSlices(f64, &.{ 3, 7 }, sized_parts.items[1].data);

    var indexed_parts = try view.splitAtIndices(&.{1}, 0);
    defer indexed_parts.deinit();
    try std.testing.expectEqual(@as(usize, 2), indexed_parts.items.len);
    try std.testing.expectEqualSlices(f64, &.{ 1, 3 }, indexed_parts.items[0].data);
    try std.testing.expectEqualSlices(f64, &.{ 5, 7 }, indexed_parts.items[1].data);

    var chunks = try view.chunk(2, 0);
    defer chunks.deinit();
    try std.testing.expectEqual(@as(usize, 2), chunks.items.len);
    try std.testing.expectEqualSlices(f64, &.{ 1, 3 }, chunks.items[0].data);
    try std.testing.expectEqualSlices(f64, &.{ 5, 7 }, chunks.items[1].data);

    try std.testing.expectError(error.InvalidAxis, view.flip(2));
    try std.testing.expectError(error.InvalidAxis, view.flipAxes(&.{ 0, 0 }));
    try std.testing.expectError(error.ShapeMismatch, view.rollAxes(&.{1}, &.{ 0, 1 }));
    try std.testing.expectError(error.InvalidAxis, view.rollAxes(&.{ 1, 1 }, &.{ 0, 0 }));
    try std.testing.expectError(error.InvalidAxis, view.rot90(1, .{ 0, 0 }));
    try std.testing.expectError(error.ShapeMismatch, view.padConstant(&.{1}, &.{1}, 0));
    try std.testing.expectError(error.InvalidShape, view.split(0, 1));
    try std.testing.expectError(error.ShapeMismatch, view.splitWithSizes(&.{1}, 1));
    try std.testing.expectError(error.InvalidShape, view.splitAtIndices(&.{3}, 1));
    try std.testing.expectError(error.InvalidShape, view.chunk(0, 1));
}

test "array non contiguous view helpers" {
    const gpa = std.testing.allocator;
    var a = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6, 7, 8 }, &.{ 2, 4 });
    defer a.deinit();

    var base_view = try a.asView();
    defer base_view.deinit();
    try std.testing.expect(base_view.isContiguous());
    try base_view.set(&.{ 1, 2 }, 99);
    try std.testing.expectEqual(@as(f64, 99), a.data[6]);

    var stepped = try a.sliceAxisView(1, .{ .start = 0, .stop = 4, .step = 2 });
    defer stepped.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, stepped.shape);
    try std.testing.expectEqualSlices(usize, &.{ 4, 2 }, stepped.strides);
    try std.testing.expect(!stepped.isContiguous());
    try std.testing.expectEqual(@as(f64, 99), try stepped.get(&.{ 1, 1 }));
    try stepped.set(&.{ 0, 1 }, 30);
    try std.testing.expectEqual(@as(f64, 30), a.data[2]);
    var stepped_owned = try stepped.toArray();
    defer stepped_owned.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 30, 5, 99 }, stepped_owned.data);
    try std.testing.expectError(error.InvalidShape, stepped.reshape(&.{4}));
    try std.testing.expectEqual(@as(usize, 2), try stepped.len());
    try std.testing.expectEqual(@as(usize, 2), try stepped.stride(1));
    try std.testing.expect(stepped.sameShape(stepped));
    try std.testing.expectError(error.InvalidShape, stepped.view(&.{4}));
    try std.testing.expectError(error.InvalidShape, stepped.viewInfer(&.{-1}));
    var base_view_alias = try base_view.view(&.{8});
    defer base_view_alias.deinit();
    try std.testing.expectEqualSlices(usize, &.{8}, base_view_alias.shape);
    var base_view_infer = try base_view.viewInfer(&.{-1});
    defer base_view_infer.deinit();
    try std.testing.expectEqualSlices(usize, &.{8}, base_view_infer.shape);
    var stepped_t_alias = try stepped.T_();
    defer stepped_t_alias.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, stepped_t_alias.shape);
    var stepped_clamped = try stepped.clamp(2, 50);
    defer stepped_clamped.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 2, 30, 5, 50 }, stepped_clamped.data);
    var stepped_eq = try stepped.equal(stepped);
    defer stepped_eq.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, stepped_eq.data);
    var stepped_clamped_view = try stepped_clamped.asView();
    defer stepped_clamped_view.deinit();
    var stepped_gt = try stepped.greater(stepped_clamped_view);
    defer stepped_gt.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, true }, stepped_gt.data);
    var stepped_lt = try stepped.less(stepped_clamped_view);
    defer stepped_lt.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false }, stepped_lt.data);
    var stepped_equal_scalar = try stepped.equalScalar(30);
    defer stepped_equal_scalar.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false }, stepped_equal_scalar.data);
    var stepped_greater_scalar = try stepped.greaterScalar(40);
    defer stepped_greater_scalar.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, true }, stepped_greater_scalar.data);
    var stepped_not_equal_scalar = try stepped.notEqualScalar(30);
    defer stepped_not_equal_scalar.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, true }, stepped_not_equal_scalar.data);
    try std.testing.expectEqual(@as(usize, 4), stepped.countNonzero());
    try std.testing.expectEqual(@as(usize, 4), stepped.count_nonzero());
    var stepped_count0 = try stepped.countNonzeroAxis(0, false);
    defer stepped_count0.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, stepped_count0.data);
    var stepped_count1_keep = try stepped.countNonzeroAxis(1, true);
    defer stepped_count1_keep.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 1 }, stepped_count1_keep.shape);
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, stepped_count1_keep.data);

    var selected = try a.selectView(0, 1);
    defer selected.deinit();
    try std.testing.expectEqualSlices(usize, &.{4}, selected.shape);
    try selected.set(&.{0}, 50);
    try std.testing.expectEqual(@as(f64, 50), a.data[4]);

    var broadcasted = try a.broadcastView(&.{ 3, 2, 4 });
    defer broadcasted.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 2, 4 }, broadcasted.shape);
    try std.testing.expectEqualSlices(usize, &.{ 0, 4, 1 }, broadcasted.strides);
    try std.testing.expectEqual(@as(f64, 99), try broadcasted.get(&.{ 2, 1, 2 }));

    var selected_broadcast = try selected.broadcastTo(&.{ 2, 4 });
    defer selected_broadcast.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 4 }, selected_broadcast.shape);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, selected_broadcast.strides);
    try selected_broadcast.set(&.{ 1, 3 }, 80);
    try std.testing.expectEqual(@as(f64, 80), a.data[7]);
    var selected_expanded = try selected.expand(&.{ 2, 4 });
    defer selected_expanded.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 4 }, selected_expanded.shape);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, selected_expanded.strides);
    try std.testing.expectEqual(@as(f64, 99), try selected_expanded.get(&.{ 1, 2 }));
    var selected_expanded_as_array = try selected.expandAsArray(a);
    defer selected_expanded_as_array.deinit();
    try std.testing.expectEqualSlices(usize, a.shape, selected_expanded_as_array.shape);
    var selected_expanded_as_view = try selected.expandAs(selected_broadcast);
    defer selected_expanded_as_view.deinit();
    try std.testing.expectEqualSlices(usize, selected_broadcast.shape, selected_expanded_as_view.shape);

    var scalar = try Array(f64).fromScalar(gpa, 5);
    defer scalar.deinit();
    var scalar_view = try scalar.asView();
    defer scalar_view.deinit();
    var scalar_view_1d = try scalar_view.atLeast1d();
    defer scalar_view_1d.deinit();
    try std.testing.expectEqualSlices(usize, &.{1}, scalar_view_1d.shape);
    try std.testing.expectEqualSlices(usize, &.{0}, scalar_view_1d.strides);
    try std.testing.expectEqual(@as(f64, 5), try scalar_view_1d.get(&.{0}));
    var scalar_view_2d = try scalar_view.atLeast2d();
    defer scalar_view_2d.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 1 }, scalar_view_2d.shape);
    try std.testing.expectEqualSlices(usize, &.{ 0, 0 }, scalar_view_2d.strides);
    var selected_2d = try selected.atLeast2d();
    defer selected_2d.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 4 }, selected_2d.shape);
    var selected_3d = try selected.atLeast3d();
    defer selected_3d.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 4, 1 }, selected_3d.shape);
    try selected_3d.set(&.{ 0, 0, 0 }, 70);
    try std.testing.expectEqual(@as(f64, 70), a.data[4]);
    try selected_3d.set(&.{ 0, 0, 0 }, 50);
    try std.testing.expectEqual(@as(f64, 50), a.data[4]);
    var stepped_3d = try stepped.atLeast3d();
    defer stepped_3d.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 1 }, stepped_3d.shape);

    var transposed = try a.transposeView();
    defer transposed.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 4, 2 }, transposed.shape);
    try std.testing.expectEqualSlices(usize, &.{ 1, 4 }, transposed.strides);
    try std.testing.expect(!transposed.isContiguous());
    var transposed_owned = try transposed.toArray();
    defer transposed_owned.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 50, 2, 6, 30, 99, 4, 80 }, transposed_owned.data);
    var swapped = try a.swapaxesView(0, 1);
    defer swapped.deinit();
    try std.testing.expectEqualSlices(usize, transposed.shape, swapped.shape);
    try std.testing.expectEqualSlices(usize, transposed.strides, swapped.strides);
    var moved_view = try a.movedimView(0, 1);
    defer moved_view.deinit();
    try std.testing.expectEqualSlices(usize, transposed.shape, moved_view.shape);
    try std.testing.expectEqualSlices(usize, transposed.strides, moved_view.strides);

    var narrowed = try a.narrowView(1, 1, 2);
    defer narrowed.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, narrowed.shape);
    try std.testing.expectEqual(@as(f64, 6), try narrowed.get(&.{ 1, 0 }));

    var stepped_plus = try stepped.addScalar(1);
    defer stepped_plus.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 2, 31, 51, 100 }, stepped_plus.data);
    var stepped_negative = try stepped.negative();
    defer stepped_negative.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -1, -30, -50, -99 }, stepped_negative.data);
    var stepped_positive = try stepped.positive();
    defer stepped_positive.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 30, 50, 99 }, stepped_positive.data);
    var stepped_absolute = try stepped_negative.absolute();
    defer stepped_absolute.deinit();
    try std.testing.expectEqualSlices(f64, stepped_positive.data, stepped_absolute.data);
    var stepped_fabs = try stepped_negative.fabs();
    defer stepped_fabs.deinit();
    try std.testing.expectEqualSlices(f64, stepped_positive.data, stepped_fabs.data);
    var ldexp_exponents = try Array(i32).fromSlice(gpa, &.{ 1, 0 }, &.{ 1, 2 });
    defer ldexp_exponents.deinit();
    var stepped_ldexp = try stepped.ldexp(ldexp_exponents);
    defer stepped_ldexp.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 2, 30, 100, 99 }, stepped_ldexp.data);
    var stepped_frexp = try stepped.frexp();
    defer stepped_frexp.deinit();
    try std.testing.expectEqualSlices(usize, stepped.shape, stepped_frexp.significand.shape);
    try std.testing.expectEqualSlices(usize, stepped.shape, stepped_frexp.exponent.shape);

    var stepped_sum = try stepped.sum(1, false);
    defer stepped_sum.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 31, 149 }, stepped_sum.data);

    var stepped_mean = try stepped.mean(null, false);
    defer stepped_mean.deinit();
    try std.testing.expectEqual(@as(usize, 0), stepped_mean.shape.len);
    try std.testing.expectApproxEqAbs(@as(f64, 45), stepped_mean.data[0], 1e-12);

    var stepped_mask = try stepped.gtScalar(40);
    defer stepped_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, true }, stepped_mask.data);

    var selected_scaled = try selected.mulScalar(2);
    defer selected_scaled.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 100, 12, 198, 160 }, selected_scaled.data);
    var shifted_view = try stepped.subScalar(10);
    defer shifted_view.deinit();
    var shifted_view_view = try shifted_view.asView();
    defer shifted_view_view.deinit();
    var view_softplus = try shifted_view_view.softplus();
    defer view_softplus.deinit();
    try std.testing.expectApproxEqAbs(std.math.log1p(@exp(@as(f64, -9))), view_softplus.data[0], 1e-12);
    var view_softsign = try shifted_view_view.softsign();
    defer view_softsign.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, -9.0 / 10.0), view_softsign.data[0], 1e-12);
    var view_leaky = try shifted_view_view.leakyRelu(0.2);
    defer view_leaky.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, -1.8), view_leaky.data[0], 1e-12);
    var view_gelu = try shifted_view_view.gelu();
    defer view_gelu.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, -0.0), view_gelu.data[0], 1e-3);

    var math_rhs = try Array(f64).fromSlice(gpa, &.{ 2, 3 }, &.{ 1, 2 });
    defer math_rhs.deinit();
    var math_rhs_view = try math_rhs.asView();
    defer math_rhs_view.deinit();
    var pow_view = try stepped.pow(math_rhs_view);
    defer pow_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 27000, 2500, 970299 }, pow_view.data);
    var floor_div_view = try stepped.floorDiv(math_rhs_view);
    defer floor_div_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 10, 25, 33 }, floor_div_view.data);
    var mod_view = try stepped.mod(math_rhs_view);
    defer mod_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 0, 0, 0 }, mod_view.data);
    var max_view = try stepped.maximum(math_rhs_view);
    defer max_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 2, 30, 50, 99 }, max_view.data);
    var min_view = try stepped.minimum(math_rhs_view);
    defer min_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 2, 3 }, min_view.data);
    var nan_pair = try Array(f64).fromSlice(gpa, &.{ std.math.nan(f64), 40 }, &.{ 1, 2 });
    defer nan_pair.deinit();
    var nan_pair_view = try nan_pair.asView();
    defer nan_pair_view.deinit();
    var fmax_view = try stepped.fmax(nan_pair_view);
    defer fmax_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 40, 50, 99 }, fmax_view.data);
    var fmin_view = try stepped.fmin(nan_pair_view);
    defer fmin_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 30, 50, 40 }, fmin_view.data);
    var fmax_scalar_view = try nan_pair_view.fmaxScalar(10);
    defer fmax_scalar_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 10, 40 }, fmax_scalar_view.data);
    var fmin_scalar_view = try nan_pair_view.fminScalar(10);
    defer fmin_scalar_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 10, 10 }, fmin_scalar_view.data);
    var hypot_view = try stepped.hypot(math_rhs_view);
    defer hypot_view.deinit();
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 5)), hypot_view.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 909)), hypot_view.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 2504)), hypot_view.data[2], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 9810)), hypot_view.data[3], 1e-12);
    var atan2_view = try stepped.atan2(math_rhs_view);
    defer atan2_view.deinit();
    try std.testing.expectApproxEqAbs(std.math.atan2(@as(f64, 1), @as(f64, 2)), atan2_view.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.atan2(@as(f64, 30), @as(f64, 3)), atan2_view.data[1], 1e-12);
    var arctan2_view = try stepped.arctan2(math_rhs_view);
    defer arctan2_view.deinit();
    try std.testing.expectEqualSlices(f64, atan2_view.data, arctan2_view.data);
    var log_view_rhs = try Array(f64).fromSlice(gpa, &.{ 0, 1 }, &.{ 1, 2 });
    defer log_view_rhs.deinit();
    var log_view_rhs_view = try log_view_rhs.asView();
    defer log_view_rhs_view.deinit();
    var log_add_view = try stepped.logaddexp(log_view_rhs_view);
    defer log_add_view.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1) + std.math.log1p(@exp(@as(f64, -1))), log_add_view.data[0], 1e-12);
    var log_add2_view = try stepped.logaddexp2(log_view_rhs_view);
    defer log_add2_view.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1) + std.math.log2(@as(f64, 1) + std.math.pow(f64, 2, -1)), log_add2_view.data[0], 1e-12);
    var xlogy_rhs = try log_view_rhs_view.addScalar(std.math.e);
    defer xlogy_rhs.deinit();
    var xlogy_rhs_view = try xlogy_rhs.asView();
    defer xlogy_rhs_view.deinit();
    var xlogy_view = try stepped.xlogy(xlogy_rhs_view);
    defer xlogy_view.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1), xlogy_view.data[0], 1e-12);
    var xlogy_scalar_view = try stepped.xlogyScalar(std.math.e);
    defer xlogy_scalar_view.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1), xlogy_scalar_view.data[0], 1e-12);
    var lerp_end = try Array(f64).fromSlice(gpa, &.{ 11, 130 }, &.{ 1, 2 });
    defer lerp_end.deinit();
    var lerp_weight = try Array(f64).fromSlice(gpa, &.{ 0, 0.5 }, &.{ 1, 2 });
    defer lerp_weight.deinit();
    var lerp_end_view = try lerp_end.asView();
    defer lerp_end_view.deinit();
    var lerp_weight_view = try lerp_weight.asView();
    defer lerp_weight_view.deinit();
    var lerp_view = try stepped.lerp(lerp_end_view, lerp_weight_view);
    defer lerp_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 80, 50, 114.5 }, lerp_view.data);
    var lerp_scalar_view = try stepped.lerpScalar(lerp_end_view, 0.25);
    defer lerp_scalar_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 3.5, 55, 40.25, 106.75 }, lerp_scalar_view.data);
    var fused_input1 = try Array(f64).fromSlice(gpa, &.{ 10, 20 }, &.{ 1, 2 });
    defer fused_input1.deinit();
    var fused_input2 = try Array(f64).fromSlice(gpa, &.{2}, &.{1});
    defer fused_input2.deinit();
    var fused_input1_view = try fused_input1.asView();
    defer fused_input1_view.deinit();
    var fused_input2_view = try fused_input2.asView();
    defer fused_input2_view.deinit();
    var addcmul_view = try stepped.addcmul(fused_input1_view, fused_input2_view, 0.5);
    defer addcmul_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 11, 50, 60, 119 }, addcmul_view.data);
    var addcmul_alias_view = try stepped.addCMul(fused_input1_view, fused_input2_view, 0.5);
    defer addcmul_alias_view.deinit();
    try std.testing.expectEqualSlices(f64, addcmul_view.data, addcmul_alias_view.data);
    var fused_denom = try Array(f64).fromSlice(gpa, &.{ 2, 4 }, &.{ 1, 2 });
    defer fused_denom.deinit();
    var fused_denom_view = try fused_denom.asView();
    defer fused_denom_view.deinit();
    var addcdiv_view = try stepped.addcdiv(fused_input1_view, fused_denom_view, 2);
    defer addcdiv_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 11, 40, 60, 109 }, addcdiv_view.data);
    var addcdiv_alias_view = try stepped.addCDiv(fused_input1_view, fused_denom_view, 2);
    defer addcdiv_alias_view.deinit();
    try std.testing.expectEqualSlices(f64, addcdiv_view.data, addcdiv_alias_view.data);
    var clip_lo = try Array(f64).fromSlice(gpa, &.{ 2, 10 }, &.{ 1, 2 });
    defer clip_lo.deinit();
    var clip_hi = try Array(f64).fromSlice(gpa, &.{ 4, 50 }, &.{ 1, 2 });
    defer clip_hi.deinit();
    var clip_lo_view = try clip_lo.asView();
    defer clip_lo_view.deinit();
    var clip_hi_view = try clip_hi.asView();
    defer clip_hi_view.deinit();
    var clipped_view = try stepped.clipArray(clip_lo_view, clip_hi_view);
    defer clipped_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 2, 30, 4, 50 }, clipped_view.data);
    var lse_view = try stepped.logsumexp(1, false);
    defer lse_view.deinit();
    try std.testing.expectEqualSlices(usize, &.{2}, lse_view.shape);
    try std.testing.expectApproxEqAbs(@as(f64, 30) + std.math.log1p(@exp(@as(f64, 1 - 30))), lse_view.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 99) + std.math.log1p(@exp(@as(f64, 50 - 99))), lse_view.data[1], 1e-12);
    var stats_view = try stepped.cov(false, 1);
    defer stats_view.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, stats_view.shape);
    try std.testing.expectApproxEqAbs(@as(f64, 1200.5), stats_view.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2380.5), stats_view.data[3], 1e-12);
    var corr_view = try stepped.corrcoef(false);
    defer corr_view.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, corr_view.shape);
    try std.testing.expectApproxEqAbs(@as(f64, 1), corr_view.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), corr_view.data[3], 1e-12);
    var weight_view = try Array(f64).fromSlice(gpa, &.{ 1, 2 }, &.{2});
    defer weight_view.deinit();
    var weighted_cov_view = try stepped.weightedCov(weight_view, false, 0);
    defer weighted_cov_view.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, weighted_cov_view.shape);
    var weighted_corr_view = try stepped.weightedCorrcoef(weight_view, false);
    defer weighted_corr_view.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, weighted_corr_view.shape);
    var special_values = try Array(f64).fromSlice(gpa, &.{ 1, std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64) }, &.{ 2, 2 });
    defer special_values.deinit();
    var special_view = try special_values.asView();
    defer special_view.deinit();
    var cleaned_view = try special_view.nanToNum(0, 9, -9);
    defer cleaned_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 0, 9, -9 }, cleaned_view.data);
    var default_cleaned_view = try special_view.nanToNumDefault();
    defer default_cleaned_view.deinit();
    try std.testing.expectEqual(@as(f64, 1), default_cleaned_view.data[0]);
    try std.testing.expectEqual(@as(f64, 0), default_cleaned_view.data[1]);
    try std.testing.expect(default_cleaned_view.data[2] > 1e300);
    try std.testing.expect(default_cleaned_view.data[3] < -1e300);
    var nan_stats = try Array(f64).fromSlice(gpa, &.{ 1, 2, std.math.nan(f64), 4, 5, 6 }, &.{ 2, 3 });
    defer nan_stats.deinit();
    var nan_stats_view = try nan_stats.transposeView();
    defer nan_stats_view.deinit();
    var nan_cov_view = try nan_stats_view.nanCov(false, 1);
    defer nan_cov_view.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, nan_cov_view.shape);
    var nan_corr_view = try nan_stats_view.nanCorrcoef(false);
    defer nan_corr_view.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, nan_corr_view.shape);
    try std.testing.expectEqual(@as(usize, 5), try nan_stats_view.nanargmax());
    try std.testing.expectEqual(@as(usize, 0), try nan_stats_view.nanargmin());
    var nan_argmax_view = try nan_stats_view.nanargmaxAxis(0, false);
    defer nan_argmax_view.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 2 }, nan_argmax_view.data);
    var nan_argmin_view = try nan_stats_view.nanargminAxis(1, false);
    defer nan_argmin_view.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 0, 1 }, nan_argmin_view.data);
    var pow_scalar_view = try stepped.powScalar(2);
    defer pow_scalar_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 900, 2500, 9801 }, pow_scalar_view.data);
    var floor_div_scalar_view = try stepped.floorDivScalar(4);
    defer floor_div_scalar_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 7, 12, 24 }, floor_div_scalar_view.data);
    var mod_scalar_view = try stepped.modScalar(7);
    defer mod_scalar_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 1, 1 }, mod_scalar_view.data);
    var max_scalar_view = try stepped.maximumScalar(10);
    defer max_scalar_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 10, 30, 50, 99 }, max_scalar_view.data);
    var min_scalar_view = try stepped.minimumScalar(10);
    defer min_scalar_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 10, 10, 10 }, min_scalar_view.data);
    var clip_min_view = try stepped.clipMin(10);
    defer clip_min_view.deinit();
    try std.testing.expectEqualSlices(f64, max_scalar_view.data, clip_min_view.data);
    var clamp_min_view = try stepped.clampMin(10);
    defer clamp_min_view.deinit();
    try std.testing.expectEqualSlices(f64, max_scalar_view.data, clamp_min_view.data);
    var clip_max_view = try stepped.clipMax(10);
    defer clip_max_view.deinit();
    try std.testing.expectEqualSlices(f64, min_scalar_view.data, clip_max_view.data);
    var clamp_max_view = try stepped.clampMax(10);
    defer clamp_max_view.deinit();
    try std.testing.expectEqualSlices(f64, min_scalar_view.data, clamp_max_view.data);
    var hypot_scalar_view = try stepped.hypotScalar(4);
    defer hypot_scalar_view.deinit();
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 17)), hypot_scalar_view.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 916)), hypot_scalar_view.data[1], 1e-12);
    var atan2_scalar_view = try stepped.atan2Scalar(2);
    defer atan2_scalar_view.deinit();
    try std.testing.expectApproxEqAbs(std.math.atan2(@as(f64, 1), @as(f64, 2)), atan2_scalar_view.data[0], 1e-12);
    var arctan2_scalar_view = try stepped.arctan2Scalar(2);
    defer arctan2_scalar_view.deinit();
    try std.testing.expectEqualSlices(f64, atan2_scalar_view.data, arctan2_scalar_view.data);
    var next_scalar_view = try stepped.nextafterScalar(100);
    defer next_scalar_view.deinit();
    try std.testing.expect(next_scalar_view.data[0] > stepped.data[0]);
    var copysign_scalar_view = try stepped.copysignScalar(-1);
    defer copysign_scalar_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -1, -30, -50, -99 }, copysign_scalar_view.data);
    var heaviside_source = try Array(f64).fromSlice(gpa, &.{ -1, 0, 2, 0 }, &.{ 2, 2 });
    defer heaviside_source.deinit();
    var heaviside_view = try heaviside_source.asView();
    defer heaviside_view.deinit();
    var heaviside_scalar_view = try heaviside_view.heavisideScalar(0.25);
    defer heaviside_scalar_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 0.25, 1, 0.25 }, heaviside_scalar_view.data);
    var ldexp_scalar_view = try stepped.ldexpScalar(1);
    defer ldexp_scalar_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 2, 60, 100, 198 }, ldexp_scalar_view.data);
    var bad_rhs = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3 }, &.{3});
    defer bad_rhs.deinit();
    var bad_rhs_view = try bad_rhs.asView();
    defer bad_rhs_view.deinit();
    try std.testing.expectError(error.ShapeMismatch, stepped.maximum(bad_rhs_view));

    var replacement = try Array(f64).fromSlice(gpa, &.{ 7, 8 }, &.{ 1, 2 });
    defer replacement.deinit();
    try stepped.copyFromArray(replacement);
    try std.testing.expectEqual(@as(f64, 7), a.data[0]);
    try std.testing.expectEqual(@as(f64, 8), a.data[2]);
    try std.testing.expectEqual(@as(f64, 7), a.data[4]);
    try std.testing.expectEqual(@as(f64, 8), a.data[6]);

    try narrowed.fill(-1);
    try std.testing.expectEqualSlices(f64, &.{ 7, -1, -1, 4, 7, -1, -1, 80 }, a.data);
}

test "array view object indexing wrappers" {
    const gpa = std.testing.allocator;
    var a = try Array(f64).fromSlice(gpa, &.{ 1, 0, 3, 4, 0, 6 }, &.{ 2, 3 });
    defer a.deinit();
    var view = try a.transposeView();
    defer view.deinit();

    var take_idx = try Array(usize).fromSlice(gpa, &.{ 0, 5, 2 }, &.{3});
    defer take_idx.deinit();
    var flat_take = try view.take(take_idx, null);
    defer flat_take.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 6, 0 }, flat_take.data);

    var signed_idx = try Array(isize).fromSlice(gpa, &.{ -1, 0 }, &.{2});
    defer signed_idx.deinit();
    var signed_take = try view.indexSelectSigned(0, signed_idx);
    defer signed_take.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, signed_take.shape);
    try std.testing.expectEqualSlices(f64, &.{ 3, 6, 1, 4 }, signed_take.data);

    var gather_idx = try Array(isize).fromSlice(gpa, &.{ -1, 0, 1, 0, -1, 1 }, &.{ 3, 2 });
    defer gather_idx.deinit();
    var gathered = try view.gatherSigned(1, gather_idx);
    defer gathered.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 4, 1, 0, 0, 6, 6 }, gathered.data);

    var mask = try Array(bool).fromSlice(gpa, &.{ true, false, false, true, true, false }, &.{ 3, 2 });
    defer mask.deinit();
    var masked = try view.maskedSelect(mask);
    defer masked.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 0, 3 }, masked.data);

    var rows = try Array(bool).fromSlice(gpa, &.{ true, false, true }, &.{3});
    defer rows.deinit();
    var compressed = try view.compress(rows, 0);
    defer compressed.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, compressed.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 4, 3, 6 }, compressed.data);

    try std.testing.expectEqual(@as(usize, 4), view.countNonzero());
    var nz = try view.nonzero();
    defer nz.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 4, 2 }, nz.shape);
    try std.testing.expectEqualSlices(usize, &.{ 0, 0, 0, 1, 2, 0, 2, 1 }, nz.data);

    var sorted_values = try Array(f64).fromSlice(gpa, &.{ 1, 2, 2, 4 }, &.{4});
    defer sorted_values.deinit();
    var sorted_view = try sorted_values.asStrided(&.{4}, &.{1}, 0);
    defer sorted_view.deinit();
    var probes = try Array(f64).fromSlice(gpa, &.{ 2, 3 }, &.{2});
    defer probes.deinit();
    var positions = try sorted_view.searchsorted(probes, .right);
    defer positions.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 3 }, positions.data);

    var needles = try Array(f64).fromSlice(gpa, &.{ 3, 6 }, &.{2});
    defer needles.deinit();
    var members = try view.isin(needles, false);
    defer members.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false, true, true }, members.data);
}

test "array view object statistics wrappers" {
    const gpa = std.testing.allocator;
    const nan = std.math.nan(f64);
    var a = try Array(f64).fromSlice(gpa, &.{
        1, 2, nan,
        4, 5, 6,
    }, &.{ 2, 3 });
    defer a.deinit();
    var view = try a.transposeView();
    defer view.deinit();

    var sum0 = try view.sum(0, false);
    defer sum0.deinit();
    try std.testing.expect(std.math.isNan(sum0.data[0]));
    try std.testing.expectEqual(@as(f64, 15), sum0.data[1]);
    var mean1 = try view.mean(1, false);
    defer mean1.deinit();
    try std.testing.expectEqual(@as(f64, 2.5), mean1.data[0]);
    try std.testing.expectEqual(@as(f64, 3.5), mean1.data[1]);
    try std.testing.expect(std.math.isNan(mean1.data[2]));
    var var1 = try view.variance(1, false, 0);
    defer var1.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 2.25), var1.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.25), var1.data[1], 1e-12);

    var median0 = try view.median(0, false);
    defer median0.deinit();
    try std.testing.expect(std.math.isNan(median0.data[0]));
    try std.testing.expectEqual(@as(f64, 5), median0.data[1]);
    var q0 = try view.quantile(0.5, 0, true);
    defer q0.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 2 }, q0.shape);
    try std.testing.expect(std.math.isNan(q0.data[0]));
    try std.testing.expectEqual(@as(f64, 5), q0.data[1]);

    var nmedian0 = try view.nanmedian(0, false);
    defer nmedian0.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1.5, 5 }, nmedian0.data);

    var nsum0 = try view.nansum(0, false);
    defer nsum0.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 3, 15 }, nsum0.data);
    var nmean1 = try view.nanmean(1, false);
    defer nmean1.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 2.5, 3.5, 6 }, nmean1.data);
    var nmax0 = try view.nanmax(0, false);
    defer nmax0.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 2, 6 }, nmax0.data);
    var amin0 = try view.amin(0, false);
    defer amin0.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 4 }, amin0.data);
    var amax1 = try view.amax(1, true);
    defer amax1.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 1 }, amax1.shape);
    try std.testing.expectEqual(@as(f64, 4), amax1.data[0]);
    try std.testing.expectEqual(@as(f64, 5), amax1.data[1]);
    try std.testing.expect(std.math.isNan(amax1.data[2]));
    var range1 = try view.ptp(1, false);
    defer range1.deinit();
    try std.testing.expectEqual(@as(f64, 3), range1.data[0]);
    try std.testing.expectEqual(@as(f64, 3), range1.data[1]);
    try std.testing.expect(std.math.isNan(range1.data[2]));

    var cums = try view.cumsumAxis(0);
    defer cums.deinit();
    try std.testing.expectEqual(@as(f64, 1), cums.data[0]);
    try std.testing.expectEqual(@as(f64, 4), cums.data[1]);
    try std.testing.expectEqual(@as(f64, 3), cums.data[2]);
    try std.testing.expectEqual(@as(f64, 9), cums.data[3]);
    try std.testing.expect(std.math.isNan(cums.data[4]));
    try std.testing.expectEqual(@as(f64, 15), cums.data[5]);
    var cummax_view = try view.cummaxAxis(0);
    defer cummax_view.deinit();
    try std.testing.expectEqual(@as(f64, 1), cummax_view.data[0]);
    try std.testing.expectEqual(@as(f64, 4), cummax_view.data[1]);
    try std.testing.expectEqual(@as(f64, 2), cummax_view.data[2]);
    try std.testing.expectEqual(@as(f64, 5), cummax_view.data[3]);
    try std.testing.expectEqual(@as(f64, 2), cummax_view.data[4]);
    try std.testing.expectEqual(@as(f64, 6), cummax_view.data[5]);
    var cummin_view = try view.cumminAxis(0);
    defer cummin_view.deinit();
    try std.testing.expectEqual(@as(f64, 1), cummin_view.data[0]);
    try std.testing.expectEqual(@as(f64, 4), cummin_view.data[1]);
    try std.testing.expectEqual(@as(f64, 1), cummin_view.data[2]);
    try std.testing.expectEqual(@as(f64, 4), cummin_view.data[3]);
    try std.testing.expectEqual(@as(f64, 1), cummin_view.data[4]);
    try std.testing.expectEqual(@as(f64, 4), cummin_view.data[5]);
    var log_cumsum_exp_view = try view.logcumsumexpAxis(0);
    defer log_cumsum_exp_view.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1), log_cumsum_exp_view.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4), log_cumsum_exp_view.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2) + std.math.log1p(std.math.exp(@as(f64, -1))), log_cumsum_exp_view.data[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5) + std.math.log1p(std.math.exp(@as(f64, -1))), log_cumsum_exp_view.data[3], 1e-12);
    try std.testing.expect(std.math.isNan(log_cumsum_exp_view.data[4]));
    try std.testing.expectApproxEqAbs(@as(f64, 6) + std.math.log1p(std.math.exp(@as(f64, -1)) + std.math.exp(@as(f64, -2))), log_cumsum_exp_view.data[5], 1e-12);
    var d = try view.diff(0, 1);
    defer d.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, d.shape);
    try std.testing.expectEqual(@as(f64, 1), d.data[0]);
    try std.testing.expectEqual(@as(f64, 1), d.data[1]);
    try std.testing.expect(std.math.isNan(d.data[2]));
    try std.testing.expectEqual(@as(f64, 1), d.data[3]);

    try std.testing.expectEqual(@as(usize, 5), try view.argmax());
    var argmin0 = try view.argminAxis(0, false);
    defer argmin0.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 0 }, argmin0.data);
}

test "array view object unary predicate wrappers" {
    const gpa = std.testing.allocator;
    var a = try Array(f64).fromSlice(gpa, &.{ -1.0, 0.0, 1.0, 2.0, std.math.inf(f64), std.math.nan(f64) }, &.{ 2, 3 });
    defer a.deinit();
    var view = try a.transposeView();
    defer view.deinit();

    var clipped = try view.clip(0, 2);
    defer clipped.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 2, 0, 2, 1, 0 }, clipped.data);

    var relu_out = try view.relu();
    defer relu_out.deinit();
    try std.testing.expectEqual(@as(f64, 0), relu_out.data[0]);
    try std.testing.expectEqual(@as(f64, 2), relu_out.data[1]);
    try std.testing.expect(std.math.isPositiveInf(relu_out.data[3]));

    var finite = try view.isFinite();
    defer finite.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false, true, false }, finite.data);
    var normal = try view.isNormal();
    defer normal.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false, true, false }, normal.data);
    var normal_alias = try view.isnormal();
    defer normal_alias.deinit();
    try std.testing.expectEqualSlices(bool, normal.data, normal_alias.data);
    var real_mask = try view.isreal();
    defer real_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, true, true }, real_mask.data);
    var complex_mask = try view.iscomplex();
    defer complex_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false, false, false }, complex_mask.data);
    var nan_mask = try view.isnan();
    defer nan_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false, false, true }, nan_mask.data);

    var positive = try view.gtScalar(0);
    defer positive.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true, true, false }, positive.data);

    var square_out = try view.square();
    defer square_out.deinit();
    try std.testing.expectEqual(@as(f64, 1), square_out.data[0]);
    try std.testing.expectEqual(@as(f64, 4), square_out.data[1]);
    var exp_out = try view.exp();
    defer exp_out.deinit();
    try std.testing.expectApproxEqAbs(std.math.exp(@as(f64, -1)), exp_out.data[0], 1e-12);
    var exp2_out = try view.exp2();
    defer exp2_out.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), exp2_out.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4), exp2_out.data[1], 1e-12);
    var lgamma_values = try view.lgamma();
    defer lgamma_values.deinit();
    try std.testing.expect(std.math.isPositiveInf(lgamma_values.data[0]));
    try std.testing.expectApproxEqAbs(@as(f64, 0), lgamma_values.data[1], 1e-12);
    try std.testing.expect(std.math.isPositiveInf(lgamma_values.data[2]));
    try std.testing.expect(std.math.isPositiveInf(lgamma_values.data[3]));
    try std.testing.expectApproxEqAbs(@as(f64, 0), lgamma_values.data[4], 1e-12);
    try std.testing.expect(std.math.isNan(lgamma_values.data[5]));
    var gammaln_values = try view.gammaln();
    defer gammaln_values.deinit();
    try std.testing.expect(std.math.isPositiveInf(gammaln_values.data[0]));
    try std.testing.expectApproxEqAbs(lgamma_values.data[1], gammaln_values.data[1], 1e-12);
    var log_gamma_values = try view.logGamma();
    defer log_gamma_values.deinit();
    try std.testing.expect(std.math.isPositiveInf(log_gamma_values.data[0]));
    try std.testing.expectApproxEqAbs(lgamma_values.data[4], log_gamma_values.data[4], 1e-12);
    var degree_values = try Array(f64).fromSlice(gpa, &.{ 0, 90, 180, 45 }, &.{ 2, 2 });
    defer degree_values.deinit();
    var degree_view = try degree_values.transposeView();
    defer degree_view.deinit();
    var radians_view = try degree_view.radians();
    defer radians_view.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0), radians_view.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.pi, radians_view.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.pi / 2.0, radians_view.data[2], 1e-12);
    var degrees_roundtrip_view = try radians_view.degrees();
    defer degrees_roundtrip_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 180, 90, 45 }, degrees_roundtrip_view.data);
    var unit_values = try Array(f64).fromSlice(gpa, &.{ 0, 1, 0.5, -0.5 }, &.{ 2, 2 });
    defer unit_values.deinit();
    var unit_view = try unit_values.transposeView();
    defer unit_view.deinit();
    var arcsin_view = try unit_view.arcsin();
    defer arcsin_view.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0), arcsin_view.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.asin(@as(f64, 0.5)), arcsin_view.data[1], 1e-12);
    var arccos_view = try unit_view.arccos();
    defer arccos_view.deinit();
    try std.testing.expectApproxEqAbs(std.math.pi / 2.0, arccos_view.data[0], 1e-12);
    var arctan_view = try unit_view.arctan();
    defer arctan_view.deinit();
    try std.testing.expectApproxEqAbs(std.math.atan(@as(f64, -0.5)), arctan_view.data[3], 1e-12);
    var view_sigmoid = try view.sigmoid();
    defer view_sigmoid.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1) / (@as(f64, 1) + @exp(@as(f64, 1))), view_sigmoid.data[0], 1e-12);
    var view_expit = try view.expit();
    defer view_expit.deinit();
    for (view_sigmoid.data, view_expit.data) |expected, actual| {
        if (std.math.isNan(expected)) {
            try std.testing.expect(std.math.isNan(actual));
        } else {
            try std.testing.expectApproxEqAbs(expected, actual, 1e-12);
        }
    }

    var probs = try Array(f64).fromSlice(gpa, &.{ 0.25, 0.5, 0.75, 0.125 }, &.{ 2, 2 });
    defer probs.deinit();
    var probs_view = try probs.transposeView();
    defer probs_view.deinit();
    var view_logit = try probs_view.logit();
    defer view_logit.deinit();
    try std.testing.expectApproxEqAbs(-std.math.log(f64, std.math.e, @as(f64, 3)), view_logit.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log(f64, std.math.e, @as(f64, 3)), view_logit.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0), view_logit.data[2], 1e-12);
    try std.testing.expectApproxEqAbs(-std.math.log(f64, std.math.e, @as(f64, 7)), view_logit.data[3], 1e-12);

    var sinc_values = try Array(f64).fromSlice(gpa, &.{ -1, 0, 0.5, 1 }, &.{ 2, 2 });
    defer sinc_values.deinit();
    var sinc_view = try sinc_values.transposeView();
    defer sinc_view.deinit();
    var view_sinc = try sinc_view.sinc();
    defer view_sinc.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0), view_sinc.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2) / std.math.pi, view_sinc.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), view_sinc.data[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0), view_sinc.data[3], 1e-12);

    var finite_values = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4 }, &.{ 2, 2 });
    defer finite_values.deinit();
    var finite_view = try finite_values.transposeView();
    defer finite_view.deinit();
    var sqrt_out = try finite_view.sqrt();
    defer sqrt_out.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1), sqrt_out.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2), sqrt_out.data[3], 1e-12);
    var rsqrt_out = try finite_view.rsqrt();
    defer rsqrt_out.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1), rsqrt_out.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), rsqrt_out.data[3], 1e-12);
    var cbrt_out = try finite_view.cbrt();
    defer cbrt_out.deinit();
    try std.testing.expectApproxEqAbs(std.math.cbrt(@as(f64, 3)), cbrt_out.data[1], 1e-12);

    var hyper_values = try Array(f64).fromSlice(gpa, &.{ 0, 0.5, 1, 2 }, &.{ 2, 2 });
    defer hyper_values.deinit();
    var hyper_view = try hyper_values.transposeView();
    defer hyper_view.deinit();
    var asinh_out = try hyper_view.asinh();
    defer asinh_out.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0), asinh_out.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.asinh(@as(f64, 1)), asinh_out.data[1], 1e-12);
    var arcsinh_out = try hyper_view.arcsinh();
    defer arcsinh_out.deinit();
    try std.testing.expectEqualSlices(f64, asinh_out.data, arcsinh_out.data);
    var acosh_out = try hyper_view.acosh();
    defer acosh_out.deinit();
    try std.testing.expect(std.math.isNan(acosh_out.data[0]));
    try std.testing.expectApproxEqAbs(@as(f64, 0), acosh_out.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.acosh(@as(f64, 2)), acosh_out.data[3], 1e-12);
    var arccosh_out = try hyper_view.arccosh();
    defer arccosh_out.deinit();
    try std.testing.expect(std.math.isNan(arccosh_out.data[0]));
    try std.testing.expectApproxEqAbs(acosh_out.data[1], arccosh_out.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(acosh_out.data[3], arccosh_out.data[3], 1e-12);
    var atanh_out = try hyper_view.atanh();
    defer atanh_out.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0), atanh_out.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.atanh(@as(f64, 0.5)), atanh_out.data[2], 1e-12);
    try std.testing.expect(std.math.isNan(atanh_out.data[3]));
    var arctanh_out = try hyper_view.arctanh();
    defer arctanh_out.deinit();
    try std.testing.expectApproxEqAbs(atanh_out.data[0], arctanh_out.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(atanh_out.data[2], arctanh_out.data[2], 1e-12);
    try std.testing.expect(std.math.isNan(arctanh_out.data[3]));
    var close = try finite_view.isclose(finite_view, 0, 0);
    defer close.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, close.data);
    try std.testing.expect(try finite_view.allclose(finite_view, 0, 0));
    var view_close_scalar = try finite_view.iscloseScalar(2, 0, 1);
    defer view_close_scalar.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false }, view_close_scalar.data);
    try std.testing.expect(!try finite_view.allcloseScalar(2, 0, 1));
}

test "array view object math sort and linalg wrappers" {
    const gpa = std.testing.allocator;
    var a = try Array(f64).fromSlice(gpa, &.{ 3, 1, 2, 6, 4, 5 }, &.{ 2, 3 });
    defer a.deinit();
    var view = try a.transposeView();
    defer view.deinit();

    var sorted = try view.sort(1);
    defer sorted.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, sorted.shape);
    try std.testing.expectEqualSlices(f64, &.{ 3, 6, 1, 4, 2, 5 }, sorted.data);

    var args = try view.argsortAxis(0, false);
    defer args.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, args.shape);
    try std.testing.expectEqualSlices(usize, &.{ 1, 1, 2, 2, 0, 0 }, args.data);

    var top = try view.topk(1, 0, true, true);
    defer top.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 2 }, top.values.shape);
    try std.testing.expectEqualSlices(f64, &.{ 3, 6 }, top.values.data);

    var soft = try view.softmax(0);
    defer soft.deinit();
    var soft_sums = try soft.sum(0, false);
    defer soft_sums.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1), soft_sums.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), soft_sums.data[1], 1e-12);

    var normed = try view.norm(2, 0, false);
    defer normed.deinit();
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 14)), normed.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 77)), normed.data[1], 1e-12);

    var rhs = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer rhs.deinit();
    var matmul_out = try view.matmulArray(rhs);
    defer matmul_out.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 3 }, matmul_out.shape);
    try std.testing.expectEqualSlices(f64, &.{ 27, 36, 45, 17, 22, 27, 22, 29, 36 }, matmul_out.data);

    const trace_value = try view.trace();
    try std.testing.expectEqual(@as(f64, 7), trace_value);

    const C = Complex64;
    var complex_values = try Array(C).fromSlice(gpa, &.{ C.init(1, 2), C.init(3, -4), C.init(-1, 1), C.init(2, 0) }, &.{ 2, 2 });
    defer complex_values.deinit();
    var complex_view = try complex_values.transposeView();
    defer complex_view.deinit();
    var real_part = try complex_view.real();
    defer real_part.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, -1, 3, 2 }, real_part.data);
    var conj_part = try complex_view.conj();
    defer conj_part.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, -2), conj_part.data[0].im, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4), conj_part.data[2].im, 1e-6);
    var magnitudes = try complex_view.magnitude();
    defer magnitudes.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, @sqrt(5.0)), magnitudes.data[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5), magnitudes.data[2], 1e-6);
    var abs_complex = try complex_view.absComplex();
    defer abs_complex.deinit();
    try std.testing.expectEqualSlices(f32, magnitudes.data, abs_complex.data);
    var angles = try complex_view.angle();
    defer angles.deinit();
    try std.testing.expectApproxEqAbs(std.math.atan2(@as(f32, 2), @as(f32, 1)), angles.data[0], 1e-6);
    try std.testing.expectApproxEqAbs(std.math.atan2(@as(f32, 1), @as(f32, -1)), angles.data[1], 1e-6);
    try std.testing.expectApproxEqAbs(std.math.atan2(@as(f32, -4), @as(f32, 3)), angles.data[2], 1e-6);
    var phases = try complex_view.phase();
    defer phases.deinit();
    try std.testing.expectEqualSlices(f32, angles.data, phases.data);
    var complex_real_mask = try complex_view.isReal();
    defer complex_real_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, true }, complex_real_mask.data);
    var complex_value_mask = try complex_view.iscomplex();
    defer complex_value_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false }, complex_value_mask.data);
}

test "array object unfold sliding-window views" {
    const gpa = std.testing.allocator;
    var v = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4, 5 }, &.{5});
    defer v.deinit();

    var windows = try v.unfold(0, 3, 1);
    defer windows.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 3 }, windows.shape);
    try std.testing.expectEqualSlices(usize, &.{ 1, 1 }, windows.strides);
    var owned = try windows.toArray();
    defer owned.deinit();
    try std.testing.expectEqualSlices(f64, &.{
        1, 2, 3,
        2, 3, 4,
        3, 4, 5,
    }, owned.data);

    try windows.set(&.{ 0, 1 }, 20);
    try std.testing.expectEqual(@as(f64, 20), v.data[1]);
    try std.testing.expectEqual(@as(f64, 20), try windows.get(&.{ 1, 0 }));

    var stepped = try v.unfold(0, 2, 2);
    defer stepped.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, stepped.shape);
    try std.testing.expectEqualSlices(usize, &.{ 2, 1 }, stepped.strides);
    var stepped_owned = try stepped.toArray();
    defer stepped_owned.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 20, 3, 4 }, stepped_owned.data);

    var m = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer m.deinit();
    var col_windows = try m.unfold(1, 2, 1);
    defer col_windows.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 2 }, col_windows.shape);
    try std.testing.expectEqualSlices(usize, &.{ 3, 1, 1 }, col_windows.strides);
    var col_owned = try col_windows.toArray();
    defer col_owned.deinit();
    try std.testing.expectEqualSlices(f64, &.{
        1, 2,
        2, 3,
        4, 5,
        5, 6,
    }, col_owned.data);

    var base_view = try m.asView();
    defer base_view.deinit();
    var row_windows = try base_view.unfold(0, 2, 1);
    defer row_windows.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 3, 2 }, row_windows.shape);
    try std.testing.expectEqualSlices(usize, &.{ 3, 1, 3 }, row_windows.strides);

    try std.testing.expectError(error.InvalidShape, v.unfold(0, 0, 1));
    try std.testing.expectError(error.InvalidShape, v.unfold(0, 2, 0));
    try std.testing.expectError(error.InvalidShape, v.unfold(0, 99, 1));
    try std.testing.expectError(error.InvalidAxis, v.unfold(1, 2, 1));
}

test "array object asStrided view helpers" {
    const gpa = std.testing.allocator;
    var base = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{6});
    defer base.deinit();

    var windows = try base.asStrided(&.{ 4, 3 }, &.{ 1, 1 }, 0);
    defer windows.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 4, 3 }, windows.shape);
    try std.testing.expectEqualSlices(usize, &.{ 1, 1 }, windows.strides);
    var owned = try windows.toArray();
    defer owned.deinit();
    try std.testing.expectEqualSlices(f64, &.{
        1, 2, 3,
        2, 3, 4,
        3, 4, 5,
        4, 5, 6,
    }, owned.data);

    try windows.set(&.{ 0, 1 }, 20);
    try std.testing.expectEqual(@as(f64, 20), base.data[1]);
    try std.testing.expectEqual(@as(f64, 20), try windows.get(&.{ 1, 0 }));

    var shifted = try base.asStrided(&.{ 2, 2 }, &.{ 2, 1 }, 1);
    defer shifted.deinit();
    var shifted_owned = try shifted.toArray();
    defer shifted_owned.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 20, 3, 4, 5 }, shifted_owned.data);

    var base_view = try base.asView();
    defer base_view.deinit();
    var every_other = try base_view.asStrided(&.{3}, &.{2}, 0);
    defer every_other.deinit();
    try std.testing.expectEqualSlices(usize, &.{2}, every_other.strides);
    var every_other_owned = try every_other.toArray();
    defer every_other_owned.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 5 }, every_other_owned.data);

    try std.testing.expectError(error.IndexOutOfBounds, base.asStrided(&.{ 4, 3 }, &.{ 2, 1 }, 0));
    try std.testing.expectError(error.IndexOutOfBounds, base.asStrided(&.{1}, &.{1}, 99));
    try std.testing.expectError(error.InvalidShape, base.asStrided(&.{ 2, 2 }, &.{1}, 0));
}

test "array object shape inference helpers" {
    const gpa = std.testing.allocator;
    var a = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, &.{ 2, 2, 3 });
    defer a.deinit();

    var inferred = try a.reshapeInfer(&.{ 3, -1, 2 });
    defer inferred.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 2, 2 }, inferred.shape);
    try std.testing.expectEqualSlices(f64, a.data, inferred.data);

    var viewed = try inferred.viewInfer(&.{ -1, 3 });
    defer viewed.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 4, 3 }, viewed.shape);

    var flat_axes = try a.flattenAxes(1, 2);
    defer flat_axes.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 6 }, flat_axes.shape);
    try std.testing.expectEqualSlices(f64, a.data, flat_axes.data);

    var unflat = try flat_axes.unflatten(1, &.{ 2, 3 });
    defer unflat.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 3 }, unflat.shape);
    try std.testing.expectEqualSlices(f64, a.data, unflat.data);

    var base_view = try a.asView();
    defer base_view.deinit();
    var view_flat = try base_view.flattenAxes(0, 1);
    defer view_flat.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 4, 3 }, view_flat.shape);
    var view_unflat = try view_flat.unflatten(0, &.{ 2, 2 });
    defer view_unflat.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 3 }, view_unflat.shape);

    var stepped = try a.sliceAxisView(2, .{ .start = 0, .stop = 3, .step = 2 });
    defer stepped.deinit();
    try std.testing.expectError(error.InvalidShape, stepped.reshapeInfer(&.{-1}));
    try std.testing.expectError(error.InvalidShape, a.reshapeInfer(&.{ -1, -1 }));
    try std.testing.expectError(error.ShapeMismatch, a.reshapeInfer(&.{ 5, -1 }));
    try std.testing.expectError(error.ShapeMismatch, flat_axes.unflatten(1, &.{ 4, 4 }));
    try std.testing.expectError(error.InvalidAxis, a.flattenAxes(2, 1));
}

test "array object in-place assignment helpers" {
    const gpa = std.testing.allocator;
    var base = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer base.deinit();

    try base.addScalarAssign(10);
    try std.testing.expectEqualSlices(f64, &.{ 11, 12, 13, 14, 15, 16 }, base.data);

    var row = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3 }, &.{ 1, 3 });
    defer row.deinit();
    try base.subAssign(row);
    try std.testing.expectEqualSlices(f64, &.{ 10, 10, 10, 13, 13, 13 }, base.data);

    var cols = try base.sliceAxisView(1, .{ .start = 0, .stop = 3, .step = 2 });
    defer cols.deinit();
    try cols.mulScalarAssign(2);
    try std.testing.expectEqualSlices(f64, &.{ 20, 10, 20, 26, 13, 26 }, base.data);

    var patch = try Array(f64).fromSlice(gpa, &.{ 5, 6 }, &.{ 1, 2 });
    defer patch.deinit();
    try cols.copyFromArray(patch);
    try std.testing.expectEqualSlices(f64, &.{ 5, 10, 6, 5, 13, 6 }, base.data);

    var selected = try base.selectView(0, 1);
    defer selected.deinit();
    var selected_delta = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3 }, &.{3});
    defer selected_delta.deinit();
    try selected.addAssignArray(selected_delta);
    try std.testing.expectEqualSlices(f64, &.{ 5, 10, 6, 6, 15, 9 }, base.data);

    var copied = try Array(f64).zeros(gpa, &.{ 2, 3 });
    defer copied.deinit();
    try copied.copyFrom(base);
    try std.testing.expectEqualSlices(f64, base.data, copied.data);
    copied.fill(0);
    try std.testing.expectEqualSlices(f64, &.{ 0, 0, 0, 0, 0, 0 }, copied.data);

    var bad_source = try Array(f64).ones(gpa, &.{ 2, 2 });
    defer bad_source.deinit();
    try std.testing.expectError(error.ShapeMismatch, copied.copyFrom(bad_source));
}

test "array object masked in-place assignment helpers" {
    const gpa = std.testing.allocator;
    var base = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6, 7, 8 }, &.{ 2, 4 });
    defer base.deinit();
    var view_mask = try Array(bool).fromSlice(gpa, &.{ true, false, false, true }, &.{ 2, 2 });
    defer view_mask.deinit();

    var cols = try base.sliceAxisView(1, .{ .start = 0, .stop = 4, .step = 2 });
    defer cols.deinit();
    try cols.maskedFill(view_mask, -1);
    try std.testing.expectEqualSlices(f64, &.{ -1, 2, 3, 4, 5, 6, -1, 8 }, base.data);

    var replacements = try Array(f64).fromSlice(gpa, &.{ 10, 20 }, &.{2});
    defer replacements.deinit();
    try cols.maskedCopyFromArray(view_mask, replacements);
    try std.testing.expectEqualSlices(f64, &.{ 10, 2, 3, 4, 5, 6, 20, 8 }, base.data);

    var source_rows = try Array(f64).fromSlice(gpa, &.{ 100, 200 }, &.{ 2, 1 });
    defer source_rows.deinit();
    try cols.copyWhereFromArray(view_mask, source_rows);
    try std.testing.expectEqualSlices(f64, &.{ 100, 2, 3, 4, 5, 6, 200, 8 }, base.data);

    var full = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer full.deinit();
    var mask = try Array(bool).fromSlice(gpa, &.{ false, true, false, true, false, true }, &.{ 2, 3 });
    defer mask.deinit();
    try full.maskedFillAssign(mask, 0);
    try std.testing.expectEqualSlices(f64, &.{ 1, 0, 3, 0, 5, 0 }, full.data);

    var masked_values = try Array(f64).fromSlice(gpa, &.{ 7, 8, 9 }, &.{3});
    defer masked_values.deinit();
    try full.maskedCopyFrom(mask, masked_values);
    try std.testing.expectEqualSlices(f64, &.{ 1, 7, 3, 8, 5, 9 }, full.data);

    var where_src = try Array(f64).fromSlice(gpa, &.{ 10, 20 }, &.{ 2, 1 });
    defer where_src.deinit();
    try full.copyWhereAssign(mask, where_src);
    try std.testing.expectEqualSlices(f64, &.{ 1, 10, 3, 20, 5, 20 }, full.data);

    var bad_values = try Array(f64).ones(gpa, &.{2});
    defer bad_values.deinit();
    try std.testing.expectError(error.ShapeMismatch, full.maskedCopyFrom(mask, bad_values));
}

test "array scalar signed indexing helpers" {
    const gpa = std.testing.allocator;
    var a = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a.deinit();

    try std.testing.expectEqual(@as(f64, 6), try a.getSigned(&.{ -1, -1 }));
    try std.testing.expectEqual(@as(f64, 4), try a.atSigned(&.{ -1, 0 }));
    try a.setSigned(&.{ -1, -2 }, 50);
    try std.testing.expectEqual(@as(f64, 50), a.data[4]);
    try a.putSigned(&.{ 0, -1 }, 30);
    try std.testing.expectEqual(@as(f64, 30), a.data[2]);

    var last_row = try a.selectSigned(0, -1);
    defer last_row.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 4, 50, 6 }, last_row.data);

    var view = try a.transposeView();
    defer view.deinit();
    try std.testing.expectEqual(@as(f64, 6), try view.getSigned(&.{ -1, -1 }));
    try view.setSigned(&.{ -1, 0 }, 60);
    try std.testing.expectEqual(@as(f64, 60), a.data[2]);
    var selected = try view.selectSigned(0, -1);
    defer selected.deinit();
    try std.testing.expectEqualSlices(usize, &.{2}, selected.shape);
    var selected_owned = try selected.toArray();
    defer selected_owned.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 60, 6 }, selected_owned.data);

    try std.testing.expectError(error.IndexOutOfBounds, a.getSigned(&.{ -3, 0 }));
}

test "array signed negative indexing helpers" {
    const gpa = std.testing.allocator;
    var a = try Array(f64).fromSlice(gpa, &.{ 10, 11, 12, 20, 21, 22 }, &.{ 2, 3 });
    defer a.deinit();

    var flat_idx = try Array(isize).fromSlice(gpa, &.{ -1, 0, -3 }, &.{3});
    defer flat_idx.deinit();
    var flat_taken = try a.takeSigned(flat_idx, null);
    defer flat_taken.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 22, 10, 20 }, flat_taken.data);

    var col_idx = try Array(isize).fromSlice(gpa, &.{ -1, 0 }, &.{2});
    defer col_idx.deinit();
    var selected = try a.indexSelectSigned(1, col_idx);
    defer selected.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, selected.shape);
    try std.testing.expectEqualSlices(f64, &.{ 12, 10, 22, 20 }, selected.data);

    var gather_idx = try Array(isize).fromSlice(gpa, &.{ -1, 0, -2, 0, -1, 1 }, &.{ 2, 3 });
    defer gather_idx.deinit();
    var gathered = try a.gatherSigned(1, gather_idx);
    defer gathered.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 12, 10, 11, 20, 22, 21 }, gathered.data);
    var gathered_alias = try a.takeAlongAxisSigned(gather_idx, 1);
    defer gathered_alias.deinit();
    try std.testing.expectEqualSlices(f64, gathered.data, gathered_alias.data);

    var values = try Array(f64).fromSlice(gpa, &.{ 100, 200 }, &.{2});
    defer values.deinit();
    var put_idx = try Array(isize).fromSlice(gpa, &.{ -1, -6 }, &.{2});
    defer put_idx.deinit();
    var put = try a.putFlatSigned(put_idx, values);
    defer put.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 200, 11, 12, 20, 21, 100 }, put.data);
    var scalar_put = try a.putFlatScalarSigned(put_idx, -5);
    defer scalar_put.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -5, 11, 12, 20, 21, -5 }, scalar_put.data);

    var bad = try Array(isize).fromSlice(gpa, &.{-7}, &.{1});
    defer bad.deinit();
    try std.testing.expectError(error.IndexOutOfBounds, a.takeSigned(bad, null));
}

test "array take mask stack cat and neural helpers" {
    const gpa = std.testing.allocator;
    var a = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a.deinit();
    var idx = try Array(usize).fromSlice(gpa, &.{ 2, 0 }, &.{2});
    defer idx.deinit();
    var picked = try a.indexSelect(1, idx);
    defer picked.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, picked.shape);
    try std.testing.expectEqualSlices(f64, &.{ 3, 1, 6, 4 }, picked.data);
    var picked_top = try a.indexSelect(1, idx);
    defer picked_top.deinit();
    try std.testing.expectEqualSlices(f64, picked.data, picked_top.data);

    var wrap_idx = try Array(usize).fromSlice(gpa, &.{ 0, 7 }, &.{2});
    defer wrap_idx.deinit();
    var take_wrapped = try a.takeMode(wrap_idx, null, .wrap);
    defer take_wrapped.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 2 }, take_wrapped.data);
    var clip_idx = try Array(usize).fromSlice(gpa, &.{ 0, 99 }, &.{2});
    defer clip_idx.deinit();
    var take_clipped = try a.takeMode(clip_idx, 1, .clip);
    defer take_clipped.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, take_clipped.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 4, 6 }, take_clipped.data);

    var mask = try Array(bool).fromSlice(gpa, &.{ true, false, true, false, true, false }, &.{ 2, 3 });
    defer mask.deinit();
    var masked = try a.maskedSelect(mask);
    defer masked.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 5 }, masked.data);

    const pieces = [_]Array(f64){ a, a };
    var st = try Array(f64).stack(gpa, pieces[0..], 1);
    defer st.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 3 }, st.shape);
    var ca = try Array(f64).cat(gpa, pieces[0..], 0);
    defer ca.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 4, 3 }, ca.shape);

    var shifted = try a.subScalar(3);
    defer shifted.deinit();
    var relu_out = try shifted.relu();
    defer relu_out.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 0, 0, 1, 2, 3 }, relu_out.data);
    var leaky_out = try shifted.leakyRelu(0.1);
    defer leaky_out.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -0.2, -0.1, 0, 1, 2, 3 }, leaky_out.data);
    var softplus_out = try shifted.softplus();
    defer softplus_out.deinit();
    try std.testing.expectApproxEqAbs(std.math.log1p(@exp(@as(f64, -2))), softplus_out.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3) + std.math.log1p(@exp(@as(f64, -3))), softplus_out.data[5], 1e-12);
    var softsign_out = try shifted.softsign();
    defer softsign_out.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, -2.0 / 3.0), softsign_out.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0 / 4.0), softsign_out.data[5], 1e-12);
    var gelu_out = try shifted.gelu();
    defer gelu_out.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, -0.04540230591222494), gelu_out.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.996362607918227), gelu_out.data[5], 1e-12);
    var sigmoid_out = try shifted.sigmoid();
    defer sigmoid_out.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1) / (@as(f64, 1) + @exp(@as(f64, 2))), sigmoid_out.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), sigmoid_out.data[2], 1e-12);
    var expit_out = try shifted.expit();
    defer expit_out.deinit();
    try std.testing.expectEqualSlices(f64, sigmoid_out.data, expit_out.data);
    var probs_for_logit = try Array(f64).fromSlice(gpa, &.{ 0, 0.25, 0.5, 0.75, 1 }, &.{5});
    defer probs_for_logit.deinit();
    var logits_out = try probs_for_logit.logit();
    defer logits_out.deinit();
    try std.testing.expect(std.math.isNegativeInf(logits_out.data[0]));
    try std.testing.expectApproxEqAbs(-std.math.log(f64, std.math.e, @as(f64, 3)), logits_out.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0), logits_out.data[2], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log(f64, std.math.e, @as(f64, 3)), logits_out.data[3], 1e-12);
    try std.testing.expect(std.math.isPositiveInf(logits_out.data[4]));
    var sinc_in = try Array(f64).fromSlice(gpa, &.{ -1, 0, 0.5, 1, 2 }, &.{5});
    defer sinc_in.deinit();
    var sinc_out = try sinc_in.sinc();
    defer sinc_out.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0), sinc_out.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), sinc_out.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2) / std.math.pi, sinc_out.data[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0), sinc_out.data[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0), sinc_out.data[4], 1e-12);
    var cs = try a.cumsum();
    defer cs.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 6, 10, 15, 21 }, cs.data);
    try std.testing.expectEqual(@as(usize, 5), try a.argmax());
}

test "array advanced indexing mutation helpers" {
    const gpa = std.testing.allocator;
    var a = try Array(f64).fromSlice(gpa, &.{ 1, 0, 3, 0, 5, 6 }, &.{ 2, 3 });
    defer a.deinit();

    var flat_idx = try a.flatNonzero();
    defer flat_idx.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 4, 5 }, flat_idx.data);
    try std.testing.expectEqual(@as(usize, 4), a.count_nonzero());
    var count_flat = try a.countNonzeroAxis(null, false);
    defer count_flat.deinit();
    try std.testing.expectEqual(@as(usize, 0), count_flat.shape.len);
    try std.testing.expectEqualSlices(usize, &.{4}, count_flat.data);
    var count_cols = try a.countNonzeroAxis(0, false);
    defer count_cols.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 1, 2 }, count_cols.data);
    var count_rows_keep = try a.countNonzeroAxis(1, true);
    defer count_rows_keep.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 1 }, count_rows_keep.shape);
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, count_rows_keep.data);

    var coords = try a.argwhere();
    defer coords.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 4, 2 }, coords.shape);
    try std.testing.expectEqualSlices(usize, &.{ 0, 0, 0, 2, 1, 1, 1, 2 }, coords.data);
    var flat_from_coords = try a.ravelCoords(coords);
    defer flat_from_coords.deinit();
    try std.testing.expectEqualSlices(usize, flat_idx.data, flat_from_coords.data);
    var coords_roundtrip = try a.unravelFlat(flat_from_coords);
    defer coords_roundtrip.deinit();
    try std.testing.expectEqualSlices(usize, coords.data, coords_roundtrip.data);
    var coord_values = try a.takeCoords(coords);
    defer coord_values.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 5, 6 }, coord_values.data);
    var coord_replacements = try Array(f64).fromSlice(gpa, &.{ 10, 30, 50, 60 }, &.{4});
    defer coord_replacements.deinit();
    var coord_put = try a.putCoords(coords, coord_replacements);
    defer coord_put.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 10, 0, 30, 0, 50, 60 }, coord_put.data);
    var coord_scalar_put = try a.putCoordsScalar(coords, -5);
    defer coord_scalar_put.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -5, 0, -5, 0, -5, -5 }, coord_scalar_put.data);
    var grid_coords = try Array(usize).fromSlice(gpa, &.{
        0, 0, 0, 2,
        1, 1, 1, 2,
    }, &.{ 2, 2, 2 });
    defer grid_coords.deinit();
    var grid_flat = try a.ravelCoords(grid_coords);
    defer grid_flat.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, grid_flat.shape);
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 4, 5 }, grid_flat.data);
    var grid_values = try a.takeCoords(grid_coords);
    defer grid_values.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, grid_values.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 5, 6 }, grid_values.data);
    var grid_coords_roundtrip = try a.unravelFlat(grid_flat);
    defer grid_coords_roundtrip.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 2 }, grid_coords_roundtrip.shape);
    try std.testing.expectEqualSlices(usize, grid_coords.data, grid_coords_roundtrip.data);
    var grid_replacements = try Array(f64).fromSlice(gpa, &.{ 10, 30, 50, 60 }, &.{ 2, 2 });
    defer grid_replacements.deinit();
    var grid_put = try a.putCoords(grid_coords, grid_replacements);
    defer grid_put.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 10, 0, 30, 0, 50, 60 }, grid_put.data);
    var row_indices = try Array(usize).fromSlice(gpa, &.{ 0, 1 }, &.{ 2, 1 });
    defer row_indices.deinit();
    var col_indices = try Array(usize).fromSlice(gpa, &.{ 0, 2 }, &.{ 1, 2 });
    defer col_indices.deinit();
    const multi_indices = [_]Array(usize){ row_indices, col_indices };
    var flat_multi = try a.ravelMultiIndex(multi_indices[0..]);
    defer flat_multi.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, flat_multi.shape);
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 3, 5 }, flat_multi.data);
    var multi_values = try a.takeMultiIndex(multi_indices[0..]);
    defer multi_values.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, multi_values.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 0, 6 }, multi_values.data);
    var multi_replacements = try Array(f64).fromSlice(gpa, &.{ 10, 30, 40, 60 }, &.{ 2, 2 });
    defer multi_replacements.deinit();
    var multi_put = try a.putMultiIndex(multi_indices[0..], multi_replacements);
    defer multi_put.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 10, 0, 30, 40, 5, 60 }, multi_put.data);
    var multi_scalar_put = try a.putMultiIndexScalar(multi_indices[0..], -9);
    defer multi_scalar_put.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -9, 0, -9, -9, 5, -9 }, multi_scalar_put.data);
    var bad_coords = try Array(usize).fromSlice(gpa, &.{ 2, 0 }, &.{ 1, 2 });
    defer bad_coords.deinit();
    try std.testing.expectError(error.IndexOutOfBounds, a.takeCoords(bad_coords));

    var cond = try Array(bool).fromSlice(gpa, &.{ true, false, true }, &.{3});
    defer cond.deinit();
    var compressed_cols = try a.compress(cond, 1);
    defer compressed_cols.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, compressed_cols.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 0, 6 }, compressed_cols.data);

    var flat_cond = try Array(bool).fromSlice(gpa, &.{ true, false, false, true, true, false }, &.{6});
    defer flat_cond.deinit();
    var compressed_flat = try a.compress(flat_cond, null);
    defer compressed_flat.deinit();
    try std.testing.expectEqualSlices(usize, &.{3}, compressed_flat.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 0, 5 }, compressed_flat.data);

    var mask = try Array(bool).fromSlice(gpa, &.{ true, false, true, false, true, false }, &.{ 2, 3 });
    defer mask.deinit();
    var mask_values = try Array(f64).fromSlice(gpa, &.{ 10, 20, 30 }, &.{3});
    defer mask_values.deinit();
    var mask_put = try a.maskedPut(mask, mask_values);
    defer mask_put.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 10, 0, 20, 0, 30, 6 }, mask_put.data);
    var mask_alias = try a.putMask(mask, mask_values);
    defer mask_alias.deinit();
    try std.testing.expectEqualSlices(f64, mask_put.data, mask_alias.data);

    var mask_scalar = try a.maskedPutScalar(mask, -1);
    defer mask_scalar.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -1, 0, -1, 0, -1, 6 }, mask_scalar.data);
    var mask_scalar_alias = try a.putMaskScalar(mask, -2);
    defer mask_scalar_alias.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -2, 0, -2, 0, -2, 6 }, mask_scalar_alias.data);
    var mask_coords = try mask.whereIndices();
    defer mask_coords.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, mask_coords.shape);
    try std.testing.expectEqualSlices(usize, &.{ 0, 0, 0, 2, 1, 1 }, mask_coords.data);
    var copy_src = try Array(f64).full(gpa, &.{ 2, 3 }, 42);
    defer copy_src.deinit();
    var copied_where = try a.copyWhere(mask, copy_src);
    defer copied_where.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 42, 0, 42, 0, 42, 6 }, copied_where.data);
    var where_other = try Array(f64).fromSlice(gpa, &.{ 10, 20, 30 }, &.{ 1, 3 });
    defer where_other.deinit();
    var where_out = try a.where(mask, where_other);
    defer where_out.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, where_out.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 20, 3, 10, 5, 30 }, where_out.data);
    var where_scalar = try a.whereScalar(mask, -1);
    defer where_scalar.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, -1, 3, -1, 5, -1 }, where_scalar.data);
    var row_view = try a.selectView(0, 0);
    defer row_view.deinit();
    var view_where = try row_view.whereArray(mask, copy_src);
    defer view_where.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, view_where.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 42, 3, 42, 0, 42 }, view_where.data);
    var view_where_scalar = try row_view.whereScalar(mask, -2);
    defer view_where_scalar.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, -2, 3, -2, 0, -2 }, view_where_scalar.data);
    var other_view = try where_other.broadcastView(&.{ 2, 3 });
    defer other_view.deinit();
    var view_where_view = try row_view.where(mask, other_view);
    defer view_where_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 20, 3, 10, 0, 30 }, view_where_view.data);
    var bad_where_other = try Array(f64).zeros(gpa, &.{2});
    defer bad_where_other.deinit();
    try std.testing.expectError(error.ShapeMismatch, a.where(mask, bad_where_other));

    var view = try a.sliceAxisView(1, .{ .start = 0, .stop = 3, .step = 2 });
    defer view.deinit();
    var view_indices = try Array(usize).fromSlice(gpa, &.{ 1, 0, 0, 1 }, &.{ 2, 2 });
    defer view_indices.deinit();
    var view_taken = try view.takeAlongAxis(view_indices, 1);
    defer view_taken.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 3, 1, 0, 6 }, view_taken.data);
    var signed_view_indices = try Array(isize).fromSlice(gpa, &.{ -1, 0, 0, -1 }, &.{ 2, 2 });
    defer signed_view_indices.deinit();
    var signed_view_taken = try view.takeAlongAxisSigned(signed_view_indices, 1);
    defer signed_view_taken.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 3, 1, 0, 6 }, signed_view_taken.data);
    var put_values_view = try Array(f64).fromSlice(gpa, &.{ 7, 8 }, &.{2});
    defer put_values_view.deinit();
    var flat_put_indices = try Array(usize).fromSlice(gpa, &.{ 1, 2 }, &.{2});
    defer flat_put_indices.deinit();
    var flat_put_view = try view.putFlat(flat_put_indices, put_values_view);
    defer flat_put_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 7, 8, 6 }, flat_put_view.data);
    var flat_put_scalar_view = try view.putFlatScalar(flat_put_indices, -3);
    defer flat_put_scalar_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, -3, -3, 6 }, flat_put_scalar_view.data);
    var signed_flat_indices = try Array(isize).fromSlice(gpa, &.{-1}, &.{1});
    defer signed_flat_indices.deinit();
    var signed_flat_put_view = try view.putFlatScalarSigned(signed_flat_indices, 99);
    defer signed_flat_put_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 0, 99 }, signed_flat_put_view.data);
    var wrap_indices = try Array(usize).fromSlice(gpa, &.{5}, &.{1});
    defer wrap_indices.deinit();
    var wrapped_put_view = try view.putFlatScalarMode(wrap_indices, 42, .wrap);
    defer wrapped_put_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 42, 0, 6 }, wrapped_put_view.data);
    var coord_indices = try Array(usize).fromSlice(gpa, &.{ 0, 1, 1, 0 }, &.{ 2, 2 });
    defer coord_indices.deinit();
    var raveled_view_coords = try view.ravelCoords(coord_indices);
    defer raveled_view_coords.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 2 }, raveled_view_coords.data);
    var coord_values_view = try view.takeCoords(coord_indices);
    defer coord_values_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 3, 0 }, coord_values_view.data);
    var coord_put_values = try Array(f64).fromSlice(gpa, &.{ 11, 22 }, &.{2});
    defer coord_put_values.deinit();
    var coord_put_view = try view.putCoords(coord_indices, coord_put_values);
    defer coord_put_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 11, 22, 6 }, coord_put_view.data);
    var coord_scalar_put_view = try view.putCoordsScalar(coord_indices, -4);
    defer coord_scalar_put_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, -4, -4, 6 }, coord_scalar_put_view.data);
    var row_idx_view = try Array(usize).fromSlice(gpa, &.{ 0, 1 }, &.{ 2, 1 });
    defer row_idx_view.deinit();
    var col_idx_view = try Array(usize).fromSlice(gpa, &.{ 0, 1 }, &.{ 1, 2 });
    defer col_idx_view.deinit();
    const view_multi_indices = [_]Array(usize){ row_idx_view, col_idx_view };
    var multi_values_view = try view.takeMultiIndex(view_multi_indices[0..]);
    defer multi_values_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 0, 6 }, multi_values_view.data);
    var multi_put_values = try Array(f64).fromSlice(gpa, &.{ 31, 32, 33, 34 }, &.{ 2, 2 });
    defer multi_put_values.deinit();
    var multi_put_view = try view.putMultiIndex(view_multi_indices[0..], multi_put_values);
    defer multi_put_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 31, 32, 33, 34 }, multi_put_view.data);
    var scatter_idx_view = try Array(usize).fromSlice(gpa, &.{ 1, 0, 0, 1 }, &.{ 2, 2 });
    defer scatter_idx_view.deinit();
    var scatter_src_view = try Array(f64).fromSlice(gpa, &.{ 10, 20, 30, 40 }, &.{ 2, 2 });
    defer scatter_src_view.deinit();
    var scattered_view = try view.scatter(1, scatter_idx_view, scatter_src_view);
    defer scattered_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 20, 10, 30, 40 }, scattered_view.data);
    var scatter_added_view = try view.scatterAddScalar(1, scatter_idx_view, 2);
    defer scatter_added_view.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 3, 5, 2, 8 }, scatter_added_view.data);

    var put_idx = try Array(usize).fromSlice(gpa, &.{ 1, 4 }, &.{2});
    defer put_idx.deinit();
    var put_values = try Array(f64).fromSlice(gpa, &.{ 11, 44 }, &.{2});
    defer put_values.deinit();
    var put_flat = try a.putFlat(put_idx, put_values);
    defer put_flat.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 11, 3, 0, 44, 6 }, put_flat.data);

    var put_scalar = try a.putFlatScalar(put_idx, 7);
    defer put_scalar.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 7, 3, 0, 7, 6 }, put_scalar.data);

    var index_put = try a.indexPut(put_idx, put_values);
    defer index_put.deinit();
    try std.testing.expectEqualSlices(f64, put_flat.data, index_put.data);

    var index_put_scalar = try a.indexPutScalar(put_idx, 9);
    defer index_put_scalar.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 9, 3, 0, 9, 6 }, index_put_scalar.data);

    var mode_idx = try Array(usize).fromSlice(gpa, &.{ 1, 9 }, &.{2});
    defer mode_idx.deinit();
    var mode_values = try Array(f64).fromSlice(gpa, &.{ 11, 99 }, &.{2});
    defer mode_values.deinit();
    var put_wrapped = try a.putFlatMode(mode_idx, mode_values, .wrap);
    defer put_wrapped.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 11, 3, 99, 5, 6 }, put_wrapped.data);
    var put_clipped = try a.putFlatScalarMode(mode_idx, -7, .clip);
    defer put_clipped.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, -7, 3, 0, 5, -7 }, put_clipped.data);

    var bad_values = try Array(f64).fromSlice(gpa, &.{ 1, 2 }, &.{2});
    defer bad_values.deinit();
    try std.testing.expectError(error.ShapeMismatch, a.maskedPut(mask, bad_values));
    var bad_indices = try Array(usize).fromSlice(gpa, &.{6}, &.{1});
    defer bad_indices.deinit();
    try std.testing.expectError(error.IndexOutOfBounds, a.putFlatScalar(bad_indices, 1));
}

test "array extended unary math and predicates" {
    const gpa = std.testing.allocator;
    var x = try Array(f64).fromSlice(gpa, &.{ -1.7, -0.2, 0.0, 0.2, 1.7 }, &.{5});
    defer x.deinit();

    var floored = try x.floor();
    defer floored.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -2, -1, 0, 0, 1 }, floored.data);
    var ceiled = try x.ceil();
    defer ceiled.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -1, 0, 0, 1, 2 }, ceiled.data);
    var rounded = try x.round();
    defer rounded.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -2, 0, 0, 0, 2 }, rounded.data);
    var truncated = try x.trunc();
    defer truncated.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -1, 0, 0, 0, 1 }, truncated.data);

    var signs = try x.sign();
    defer signs.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -1, -1, 0, 1, 1 }, signs.data);
    var bits = try x.signbit();
    defer bits.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false, false }, bits.data);

    var sq = try x.square();
    defer sq.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 2.89), sq.data[0], 1e-12);
    var denom = try Array(f64).fromSlice(gpa, &.{ 2, -4 }, &.{2});
    defer denom.deinit();
    var recip = try denom.reciprocal();
    defer recip.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0.5, -0.25 }, recip.data);
    var sqrt_inputs = try Array(f64).fromSlice(gpa, &.{ 1, 4, 9 }, &.{3});
    defer sqrt_inputs.deinit();
    var rsqrt_inputs = try sqrt_inputs.rsqrt();
    defer rsqrt_inputs.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 0.5, 1.0 / 3.0 }, rsqrt_inputs.data);
    var cbrt_inputs = try Array(f64).fromSlice(gpa, &.{ -8, 0, 27 }, &.{3});
    defer cbrt_inputs.deinit();
    var cbrt_values = try cbrt_inputs.cbrt();
    defer cbrt_values.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, -2), cbrt_values.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0), cbrt_values.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3), cbrt_values.data[2], 1e-12);

    var stable = try Array(f64).fromSlice(gpa, &.{ 0, 1 }, &.{2});
    defer stable.deinit();
    var exp2_values = try stable.exp2();
    defer exp2_values.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 2 }, exp2_values.data);
    var e1 = try stable.expm1();
    defer e1.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0), e1.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.e - 1, e1.data[1], 1e-12);
    var l1 = try stable.log1p();
    defer l1.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0), l1.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.ln2, l1.data[1], 1e-12);
    var lg = try stable.lgamma();
    defer lg.deinit();
    try std.testing.expectEqualSlices(f64, &.{ std.math.inf(f64), 0 }, lg.data);
    var gln = try stable.gammaln();
    defer gln.deinit();
    try std.testing.expectEqualSlices(f64, lg.data, gln.data);
    var log_gamma = try stable.logGamma();
    defer log_gamma.deinit();
    try std.testing.expectEqualSlices(f64, lg.data, log_gamma.data);
    var powers = try Array(f64).fromSlice(gpa, &.{ 1, 10, 100 }, &.{3});
    defer powers.deinit();
    var log2_out = try powers.log2();
    defer log2_out.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0), log2_out.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log2(@as(f64, 10)), log2_out.data[1], 1e-12);
    var log10_out = try powers.log10();
    defer log10_out.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 1, 2 }, log10_out.data);

    var degrees = try Array(f64).fromSlice(gpa, &.{ 0, 90, 180 }, &.{3});
    defer degrees.deinit();
    var radians = try degrees.deg2rad();
    defer radians.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0), radians.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.pi / 2.0, radians.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.pi, radians.data[2], 1e-12);
    var radians_alias = try degrees.radians();
    defer radians_alias.deinit();
    try std.testing.expectEqualSlices(f64, radians.data, radians_alias.data);
    var roundtrip_degrees = try radians.rad2deg();
    defer roundtrip_degrees.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0), roundtrip_degrees.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 90), roundtrip_degrees.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 180), roundtrip_degrees.data[2], 1e-12);
    var degrees_alias = try radians.degrees();
    defer degrees_alias.deinit();
    try std.testing.expectEqualSlices(f64, roundtrip_degrees.data, degrees_alias.data);

    var significands = try Array(f64).fromSlice(gpa, &.{ 0.5, 0.75, -0.5 }, &.{3});
    defer significands.deinit();
    var exponents = try Array(i32).fromSlice(gpa, &.{ 1, 2, 3 }, &.{3});
    defer exponents.deinit();
    var ld = try significands.ldexp(exponents);
    defer ld.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, -4 }, ld.data);
    var ld_scalar = try significands.ldexpScalar(2);
    defer ld_scalar.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 2, 3, -2 }, ld_scalar.data);
    var frexp_result = try ld.frexp();
    defer frexp_result.deinit();
    try std.testing.expectEqualSlices(f64, significands.data, frexp_result.significand.data);
    try std.testing.expectEqualSlices(i32, exponents.data, frexp_result.exponent.data);

    var angles = try Array(f64).fromSlice(gpa, &.{ 0, std.math.pi / 2.0 }, &.{2});
    defer angles.deinit();
    var sine = try angles.sin();
    defer sine.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0), sine.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), sine.data[1], 1e-12);
    var cosine = try angles.cos();
    defer cosine.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1), cosine.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0), cosine.data[1], 1e-12);
    var tangent = try angles.tan();
    defer tangent.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0), tangent.data[0], 1e-12);

    var unit = try Array(f64).fromSlice(gpa, &.{ 0, 1 }, &.{2});
    defer unit.deinit();
    var arcs = try unit.asin();
    defer arcs.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0), arcs.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.pi / 2.0, arcs.data[1], 1e-12);
    var arcs_alias = try unit.arcsin();
    defer arcs_alias.deinit();
    try std.testing.expectEqualSlices(f64, arcs.data, arcs_alias.data);
    var arcc = try unit.acos();
    defer arcc.deinit();
    try std.testing.expectApproxEqAbs(std.math.pi / 2.0, arcc.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0), arcc.data[1], 1e-12);
    var arcc_alias = try unit.arccos();
    defer arcc_alias.deinit();
    try std.testing.expectEqualSlices(f64, arcc.data, arcc_alias.data);
    var arct = try unit.atan();
    defer arct.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0), arct.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.pi / 4.0, arct.data[1], 1e-12);
    var arct_alias = try unit.arctan();
    defer arct_alias.deinit();
    try std.testing.expectEqualSlices(f64, arct.data, arct_alias.data);

    var hyp = try Array(f64).fromSlice(gpa, &.{ 0, 1 }, &.{2});
    defer hyp.deinit();
    var sh = try hyp.sinh();
    defer sh.deinit();
    var ch = try hyp.cosh();
    defer ch.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0), sh.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), ch.data[0], 1e-12);
    var inverse_hyp = try Array(f64).fromSlice(gpa, &.{ 0, 0.5, 1, 2 }, &.{4});
    defer inverse_hyp.deinit();
    var asinh_values = try inverse_hyp.asinh();
    defer asinh_values.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0), asinh_values.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.asinh(@as(f64, 2)), asinh_values.data[3], 1e-12);
    var arcsinh_values = try inverse_hyp.arcsinh();
    defer arcsinh_values.deinit();
    try std.testing.expectEqualSlices(f64, asinh_values.data, arcsinh_values.data);
    var acosh_values = try inverse_hyp.acosh();
    defer acosh_values.deinit();
    try std.testing.expect(std.math.isNan(acosh_values.data[0]));
    try std.testing.expectApproxEqAbs(@as(f64, 0), acosh_values.data[2], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.acosh(@as(f64, 2)), acosh_values.data[3], 1e-12);
    var arccosh_values = try inverse_hyp.arccosh();
    defer arccosh_values.deinit();
    try std.testing.expect(std.math.isNan(arccosh_values.data[0]));
    try std.testing.expectApproxEqAbs(acosh_values.data[2], arccosh_values.data[2], 1e-12);
    try std.testing.expectApproxEqAbs(acosh_values.data[3], arccosh_values.data[3], 1e-12);
    var atanh_values = try inverse_hyp.atanh();
    defer atanh_values.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0), atanh_values.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.atanh(@as(f64, 0.5)), atanh_values.data[1], 1e-12);
    try std.testing.expect(std.math.isPositiveInf(atanh_values.data[2]));
    try std.testing.expect(std.math.isNan(atanh_values.data[3]));
    var arctanh_values = try inverse_hyp.arctanh();
    defer arctanh_values.deinit();
    try std.testing.expectApproxEqAbs(atanh_values.data[0], arctanh_values.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(atanh_values.data[1], arctanh_values.data[1], 1e-12);
    try std.testing.expect(std.math.isPositiveInf(arctanh_values.data[2]));
    try std.testing.expect(std.math.isNan(arctanh_values.data[3]));

    var special = try Array(f64).fromSlice(gpa, &.{ 1, std.math.inf(f64), std.math.nan(f64) }, &.{3});
    defer special.deinit();
    var finite_mask = try special.isFinite();
    defer finite_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false, false }, finite_mask.data);
    var normal_mask = try special.isNormal();
    defer normal_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false, false }, normal_mask.data);
    var normal_alias = try special.isnormal();
    defer normal_alias.deinit();
    try std.testing.expectEqualSlices(bool, normal_mask.data, normal_alias.data);
    var inf_mask = try special.isinf();
    defer inf_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true, false }, inf_mask.data);
    var posinf_mask = try special.isPosInf();
    defer posinf_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true, false }, posinf_mask.data);
    var neg_special = try Array(f64).fromSlice(gpa, &.{ -std.math.inf(f64), 1, std.math.inf(f64) }, &.{3});
    defer neg_special.deinit();
    var neginf_mask = try neg_special.isneginf();
    defer neginf_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false, false }, neginf_mask.data);
    var special_view = try neg_special.asView();
    defer special_view.deinit();
    var view_posinf = try special_view.isposinf();
    defer view_posinf.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, false, true }, view_posinf.data);
    var view_neginf = try special_view.isNegInf();
    defer view_neginf.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false, false }, view_neginf.data);
    var nan_mask = try special.isnan();
    defer nan_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, false, true }, nan_mask.data);

    var ints = try Array(i32).fromSlice(gpa, &.{ -2, 0, 7 }, &.{3});
    defer ints.deinit();
    var int_sign = try ints.sign();
    defer int_sign.deinit();
    try std.testing.expectEqualSlices(i32, &.{ -1, 0, 1 }, int_sign.data);
    var int_finite = try ints.isfinite();
    defer int_finite.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, true }, int_finite.data);
}

test "array gather scatter and scalar scatter" {
    const gpa = std.testing.allocator;
    var a = try Array(f64).fromSlice(gpa, &.{ 10, 11, 12, 20, 21, 22 }, &.{ 2, 3 });
    defer a.deinit();
    var idx = try Array(usize).fromSlice(gpa, &.{ 2, 1, 0, 0, 2, 1 }, &.{ 2, 3 });
    defer idx.deinit();

    var gathered = try a.gather(1, idx);
    defer gathered.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 12, 11, 10, 20, 22, 21 }, gathered.data);
    var gathered_top = try a.gather(1, idx);
    defer gathered_top.deinit();
    try std.testing.expectEqualSlices(f64, gathered.data, gathered_top.data);

    var base = try Array(f64).zeros(gpa, &.{ 2, 3 });
    defer base.deinit();
    var scattered = try base.scatter(1, idx, gathered);
    defer scattered.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 10, 11, 12, 20, 21, 22 }, scattered.data);
    var scattered_top = try base.scatter(1, idx, gathered);
    defer scattered_top.deinit();
    try std.testing.expectEqualSlices(f64, scattered.data, scattered_top.data);
    var scatter_add = try base.scatterAdd(1, idx, gathered);
    defer scatter_add.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 10, 11, 12, 20, 21, 22 }, scatter_add.data);

    var filled = try base.scatterScalar(1, idx, 7);
    defer filled.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 7, 7, 7, 7, 7, 7 }, filled.data);
    var scalar_added = try base.scatterAddScalar(1, idx, 2);
    defer scalar_added.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 2, 2, 2, 2, 2, 2 }, scalar_added.data);
}

test "array logsoftmax norm and matrix helpers" {
    const gpa = std.testing.allocator;
    var logits = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 1, 2, 3 }, &.{ 2, 3 });
    defer logits.deinit();
    var log_probs = try logits.logSoftmax(1);
    defer log_probs.deinit();
    var probs = try log_probs.exp();
    defer probs.deinit();
    var row_sums = try probs.sum(1, false);
    defer row_sums.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1), row_sums.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), row_sums.data[1], 1e-12);

    var v = try Array(f64).fromSlice(gpa, &.{ 3, 4 }, &.{2});
    defer v.deinit();
    var n = try v.norm(2, null, false);
    defer n.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 5), n.data[0], 1e-12);

    var w = try Array(f64).fromSlice(gpa, &.{ 2, 5, 7 }, &.{3});
    defer w.deinit();
    var out = try v.outer(w);
    defer out.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, out.shape);
    try std.testing.expectEqualSlices(f64, &.{ 6, 15, 21, 8, 20, 28 }, out.data);

    var m = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, &.{ 3, 3 });
    defer m.deinit();
    try std.testing.expectEqual(@as(f64, 15), try m.trace());
    try std.testing.expectEqual(@as(f64, 8), try m.traceOffset(1));
    try std.testing.expectEqual(@as(f64, 12), try m.traceOffset(-1));
    try std.testing.expectEqual(@as(f64, 0), try m.traceOffset(9));
    var d = try m.diagonal(0);
    defer d.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 5, 9 }, d.data);
    var upper = try m.triu(0);
    defer upper.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 3, 0, 5, 6, 0, 0, 9 }, upper.data);
    var lower = try m.tril(0);
    defer lower.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 0, 0, 4, 5, 0, 7, 8, 9 }, lower.data);

    var mt = try m.transposeView();
    defer mt.deinit();
    try std.testing.expectEqual(@as(f64, 15), try mt.trace());
    try std.testing.expectEqual(@as(f64, 12), try mt.traceOffset(1));
    var mt_diag = try mt.diagonal(1);
    defer mt_diag.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 4, 8 }, mt_diag.data);
    var mt_diag_main = try mt.diag(0);
    defer mt_diag_main.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 5, 9 }, mt_diag_main.data);
    var mt_upper = try mt.triu(1);
    defer mt_upper.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 4, 7, 0, 0, 8, 0, 0, 0 }, mt_upper.data);
    var mt_lower = try mt.tril(-1);
    defer mt_lower.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 0, 0, 2, 0, 0, 3, 6, 0 }, mt_lower.data);
    var mt_row = try mt.select(0, 0);
    defer mt_row.deinit();
    var mt_row_diag = try mt_row.diag(0);
    defer mt_row_diag.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 0, 0, 0, 4, 0, 0, 0, 7 }, mt_row_diag.data);
}

test "array min max arg reductions and topk" {
    const gpa = std.testing.allocator;
    var a = try Array(f64).fromSlice(gpa, &.{ 9, 1, 5, 4, 8, 2 }, &.{ 2, 3 });
    defer a.deinit();

    var min0 = try a.min(0, false);
    defer min0.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 4, 1, 2 }, min0.data);
    var max1 = try a.max(1, true);
    defer max1.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 1 }, max1.shape);
    try std.testing.expectEqualSlices(f64, &.{ 9, 8 }, max1.data);

    var arg0 = try a.argmaxAxis(0, false);
    defer arg0.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 0 }, arg0.data);
    var arg1 = try a.argminAxis(1, true);
    defer arg1.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 1 }, arg1.shape);
    try std.testing.expectEqualSlices(usize, &.{ 1, 2 }, arg1.data);

    var with_nan = try Array(f64).fromSlice(gpa, &.{
        std.math.nan(f64), 4,                 2,
        8,                 std.math.nan(f64), 9,
    }, &.{ 2, 3 });
    defer with_nan.deinit();
    try std.testing.expectEqual(@as(usize, 5), try with_nan.nanargmax());
    try std.testing.expectEqual(@as(usize, 2), try with_nan.nanargmin());
    var nan_argmax0 = try with_nan.nanargmaxAxis(0, false);
    defer nan_argmax0.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 0, 1 }, nan_argmax0.data);
    var nan_argmin1 = try with_nan.nanargminAxis(1, true);
    defer nan_argmin1.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 1 }, nan_argmin1.shape);
    try std.testing.expectEqualSlices(usize, &.{ 2, 0 }, nan_argmin1.data);
    var all_nan_args = try Array(f64).fromSlice(gpa, &.{ std.math.nan(f64), std.math.nan(f64) }, &.{2});
    defer all_nan_args.deinit();
    try std.testing.expectError(error.EmptyArray, all_nan_args.nanargmax());

    var flat_top = try a.topk(3, null, true, true);
    defer flat_top.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 9, 8, 5 }, flat_top.values.data);
    try std.testing.expectEqualSlices(usize, &.{ 0, 4, 2 }, flat_top.indices.data);

    var row_top = try a.topk(2, 1, false, true);
    defer row_top.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, row_top.values.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 5, 2, 4 }, row_top.values.data);
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 2, 0 }, row_top.indices.data);

    var flat_unsorted = try a.topk(3, null, true, false);
    defer flat_unsorted.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 9, 5, 8 }, flat_unsorted.values.data);
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 4 }, flat_unsorted.indices.data);

    var row_unsorted = try a.topk(2, 1, true, false);
    defer row_unsorted.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 9, 5, 4, 8 }, row_unsorted.values.data);
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 0, 1 }, row_unsorted.indices.data);
}

test "array median quantile covariance and corrcoef" {
    const gpa = std.testing.allocator;
    const nan = std.math.nan(f64);
    var a = try Array(f64).fromSlice(gpa, &.{ 1, 4, 2, 8, 3, 9 }, &.{ 2, 3 });
    defer a.deinit();

    var med_flat = try a.median(null, false);
    defer med_flat.deinit();
    try std.testing.expectEqual(@as(usize, 0), med_flat.shape.len);
    try std.testing.expectApproxEqAbs(@as(f64, 3.5), med_flat.data[0], 1e-12);

    var med_rows = try a.median(1, false);
    defer med_rows.deinit();
    try std.testing.expectEqualSlices(usize, &.{2}, med_rows.shape);
    try std.testing.expectEqualSlices(f64, &.{ 2, 8 }, med_rows.data);

    var q_cols = try a.quantile(0.25, 0, true);
    defer q_cols.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 3 }, q_cols.shape);
    try std.testing.expectEqualSlices(f64, &.{ 2.75, 3.25, 3.75 }, q_cols.data);

    var p_flat = try a.percentile(75, null, false);
    defer p_flat.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 7), p_flat.data[0], 1e-12);
    try std.testing.expectError(error.InvalidShape, a.quantile(1.5, null, false));

    var weights = try Array(f64).fromSlice(gpa, &.{ 1, 1, 1, 3, 3, 3 }, &.{ 2, 3 });
    defer weights.deinit();
    var weighted_mean_flat = try a.weightedMean(weights, null, false);
    defer weighted_mean_flat.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 67.0 / 12.0), weighted_mean_flat.data[0], 1e-12);
    var average_rows = try a.average(weights, 1, false);
    defer average_rows.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 7.0 / 3.0), average_rows.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0 / 3.0), average_rows.data[1], 1e-12);
    var unweighted_average = try a.average(null, null, false);
    defer unweighted_average.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 4.5), unweighted_average.data[0], 1e-12);
    var weighted_var_flat = try a.weightedVariance(weights, null, false, 0);
    defer weighted_var_flat.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 9.07638888888889), weighted_var_flat.data[0], 1e-12);
    var weighted_std_flat = try a.weightedStddev(weights, null, false, 0);
    defer weighted_std_flat.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 3.01270458042087), weighted_std_flat.data[0], 1e-12);
    var weighted_var_rows = try a.weightedVar(weights, 1, false, 0);
    defer weighted_var_rows.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 14.0 / 9.0), weighted_var_rows.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 62.0 / 9.0), weighted_var_rows.data[1], 1e-12);
    var weighted_std_rows = try a.weightedStd(weights, 1, false, 0);
    defer weighted_std_rows.deinit();
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 14.0 / 9.0)), weighted_std_rows.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 62.0 / 9.0)), weighted_std_rows.data[1], 1e-12);

    var value_vec = try Array(f64).fromSlice(gpa, &.{ 1, 10, 100 }, &.{3});
    defer value_vec.deinit();
    var weight_vec = try Array(f64).fromSlice(gpa, &.{ 1, 8, 1 }, &.{3});
    defer weight_vec.deinit();
    var weighted_med = try value_vec.weightedMedian(weight_vec, null, false);
    defer weighted_med.deinit();
    try std.testing.expectEqualSlices(f64, &.{10}, weighted_med.data);
    var weighted_q = try value_vec.weightedQuantile(weight_vec, 0.95, null, false);
    defer weighted_q.deinit();
    try std.testing.expectEqualSlices(f64, &.{100}, weighted_q.data);
    var view_source = try Array(f64).fromSlice(gpa, &.{ 1, 9, 2, 8, 3, 7 }, &.{ 2, 3 });
    defer view_source.deinit();
    var view = try view_source.sliceAxisView(1, .{ .start = 0, .stop = 3, .step = 2 });
    defer view.deinit();
    var view_weights = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4 }, &.{ 2, 2 });
    defer view_weights.deinit();
    var weights_view = try view_weights.asView();
    defer weights_view.deinit();
    var view_weighted_mean = try view.weightedMean(weights_view, null, false);
    defer view_weighted_mean.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 5.7), view_weighted_mean.data[0], 1e-12);
    var view_weighted_mean_axis = try view.weightedMeanArray(view_weights, 1, false);
    defer view_weighted_mean_axis.deinit();
    try std.testing.expectEqualSlices(usize, &.{2}, view_weighted_mean_axis.shape);
    var view_weighted_var = try view.weightedVariance(weights_view, null, false, 0);
    defer view_weighted_var.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 7.21), view_weighted_var.data[0], 1e-12);
    var view_weighted_std = try view.weightedStddevArray(view_weights, null, false, 0);
    defer view_weighted_std.deinit();
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 7.21)), view_weighted_std.data[0], 1e-12);
    var view_weighted_var_axis = try view.weightedVar(weights_view, 1, false, 0);
    defer view_weighted_var_axis.deinit();
    try std.testing.expectEqualSlices(usize, &.{2}, view_weighted_var_axis.shape);
    var view_weighted_std_axis = try view.weightedStdArray(view_weights, 1, false, 0);
    defer view_weighted_std_axis.deinit();
    try std.testing.expectEqualSlices(usize, &.{2}, view_weighted_std_axis.shape);
    var view_weighted_median = try view.weightedMedian(weights_view, null, false);
    defer view_weighted_median.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 7), view_weighted_median.data[0], 1e-12);
    var view_weighted_quantile = try view.weightedQuantileArray(view_weights, 0.25, null, false);
    defer view_weighted_quantile.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 2), view_weighted_quantile.data[0], 1e-12);
    var negative_weights = try Array(f64).fromSlice(gpa, &.{ -1, 1, 1 }, &.{3});
    defer negative_weights.deinit();
    try std.testing.expectError(error.InvalidShape, value_vec.weightedMean(negative_weights, null, false));

    var obs_by_var = try Array(f64).fromSlice(gpa, &.{
        1, 2,
        2, 4,
        3, 6,
    }, &.{ 3, 2 });
    defer obs_by_var.deinit();

    var covariance = try obs_by_var.cov(false, 1);
    defer covariance.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, covariance.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 2, 4 }, covariance.data);

    var corr = try obs_by_var.corrcoef(false);
    defer corr.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 1, 1, 1 }, corr.data);
    var obs_weights = try Array(f64).fromSlice(gpa, &.{ 1, 2, 1 }, &.{3});
    defer obs_weights.deinit();
    var weighted_covariance = try obs_by_var.weightedCov(obs_weights, false, 1);
    defer weighted_covariance.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), weighted_covariance.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0 / 3.0), weighted_covariance.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0 / 3.0), weighted_covariance.data[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 8.0 / 3.0), weighted_covariance.data[3], 1e-12);
    var weighted_corr = try obs_by_var.weightedCorrcoef(obs_weights, false);
    defer weighted_corr.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 1, 1, 1 }, weighted_corr.data);

    var nan_obs = try Array(f64).fromSlice(gpa, &.{
        1,   2,
        nan, nan,
        3,   6,
    }, &.{ 3, 2 });
    defer nan_obs.deinit();
    var nan_covariance = try nan_obs.nanCov(false, 1);
    defer nan_covariance.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 2, 4, 4, 8 }, nan_covariance.data);
    var nan_corr = try nan_obs.nanCorrcoef(false);
    defer nan_corr.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 1, 1, 1 }, nan_corr.data);

    var rowvar_data = try Array(f64).fromSlice(gpa, &.{
        1, 2, 3,
        2, 4, 6,
    }, &.{ 2, 3 });
    defer rowvar_data.deinit();
    var row_cov = try rowvar_data.cov(true, 1);
    defer row_cov.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 2, 4 }, row_cov.data);

    var v = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3 }, &.{3});
    defer v.deinit();
    var var_scalar = try v.cov(true, 1);
    defer var_scalar.deinit();
    try std.testing.expectEqual(@as(usize, 0), var_scalar.shape.len);
    try std.testing.expectApproxEqAbs(@as(f64, 1), var_scalar.data[0], 1e-12);
    var corr_scalar = try v.corrcoef(true);
    defer corr_scalar.deinit();
    try std.testing.expectEqual(@as(usize, 0), corr_scalar.shape.len);
    try std.testing.expectApproxEqAbs(@as(f64, 1), corr_scalar.data[0], 1e-12);
}

test "array nan cleanup and nan-aware statistics" {
    const gpa = std.testing.allocator;
    const nan = std.math.nan(f64);
    const inf = std.math.inf(f64);
    var a = try Array(f64).fromSlice(gpa, &.{
        1,   nan, 3,
        nan, nan, 6,
        7,   8,   inf,
    }, &.{ 3, 3 });
    defer a.deinit();

    var cleaned = try a.nanToNum(0, 99, -99);
    defer cleaned.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 0, 3, 0, 0, 6, 7, 8, 99 }, cleaned.data);

    var row_sum = try a.nansum(1, false);
    defer row_sum.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 4, 6, inf }, row_sum.data);
    var col_sum_keep = try a.nansum(0, true);
    defer col_sum_keep.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 3 }, col_sum_keep.shape);
    try std.testing.expectEqualSlices(f64, &.{ 8, 8, inf }, col_sum_keep.data);

    var row_mean = try a.nanmean(1, false);
    defer row_mean.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 2, 6, inf }, row_mean.data);
    var col_mean = try a.nanmean(0, false);
    defer col_mean.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 4, 8, inf }, col_mean.data);

    var clean_stats = try Array(f64).fromSlice(gpa, &.{
        1, nan, 3,
        2, nan, 6,
        3, 8,   9,
    }, &.{ 3, 3 });
    defer clean_stats.deinit();
    var variance_cols = try clean_stats.nanvar(0, false, 0);
    defer variance_cols.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), variance_cols.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0), variance_cols.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 6), variance_cols.data[2], 1e-12);
    var std_cols = try clean_stats.nanstd(0, false, 0);
    defer std_cols.deinit();
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 2.0 / 3.0)), std_cols.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0), std_cols.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 6)), std_cols.data[2], 1e-12);

    var mins = try clean_stats.nanmin(0, false);
    defer mins.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 8, 3 }, mins.data);
    var maxs = try clean_stats.nanmax(1, false);
    defer maxs.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 3, 6, 9 }, maxs.data);

    var med = try clean_stats.nanmedian(0, false);
    defer med.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 2, 8, 6 }, med.data);
    var q = try clean_stats.nanquantile(0.25, 0, true);
    defer q.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 3 }, q.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1.5, 8, 4.5 }, q.data);
    var pct = try clean_stats.nanpercentile(75, 1, false);
    defer pct.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 2.5, 5, 8.5 }, pct.data);

    var all_nan = try Array(f64).fromSlice(gpa, &.{ nan, nan }, &.{2});
    defer all_nan.deinit();
    var all_nan_mean = try all_nan.nanmean(null, false);
    defer all_nan_mean.deinit();
    try std.testing.expect(std.math.isNan(all_nan_mean.data[0]));
    var all_nan_min = try all_nan.nanmin(null, false);
    defer all_nan_min.deinit();
    try std.testing.expect(std.math.isNan(all_nan_min.data[0]));
    var all_nan_quantile = try all_nan.nanquantile(0.5, null, false);
    defer all_nan_quantile.deinit();
    try std.testing.expect(std.math.isNan(all_nan_quantile.data[0]));
}

test "array sort argsort and partition axes" {
    const gpa = std.testing.allocator;
    var a = try Array(f64).fromSlice(gpa, &.{ 8, 1, 5, 3, 7, 2 }, &.{ 2, 3 });
    defer a.deinit();

    var flat_desc = try a.sortDescending(null);
    defer flat_desc.deinit();
    try std.testing.expectEqualSlices(usize, &.{6}, flat_desc.shape);
    try std.testing.expectEqualSlices(f64, &.{ 8, 7, 5, 3, 2, 1 }, flat_desc.data);

    var row_sorted = try a.sort(1);
    defer row_sorted.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 5, 8, 2, 3, 7 }, row_sorted.data);

    var col_sorted_desc = try a.sortBy(0, true);
    defer col_sorted_desc.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 8, 7, 5, 3, 1, 2 }, col_sorted_desc.data);

    var row_order = try a.argsortAxis(1, false);
    defer row_order.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 0, 2, 0, 1 }, row_order.data);

    var flat_order_desc = try a.argsortDescending();
    defer flat_order_desc.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 4, 2, 3, 5, 1 }, flat_order_desc.data);

    var col_sorted = try a.sortWithIndices(0, false);
    defer col_sorted.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 3, 1, 2, 8, 7, 5 }, col_sorted.values.data);
    try std.testing.expectEqualSlices(usize, &.{ 1, 0, 1, 0, 1, 0 }, col_sorted.indices.data);

    var row_partition = try a.partition(1, 1, false);
    defer row_partition.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 5, 8, 2, 3, 7 }, row_partition.data);
    var row_argpartition = try a.argpartition(1, 1, false);
    defer row_argpartition.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 0, 2, 0, 1 }, row_argpartition.data);

    try std.testing.expectError(error.InvalidAxis, a.sort(2));
    try std.testing.expectError(error.InvalidShape, a.partition(3, 1, false));

    var flags = try Array(bool).fromSlice(gpa, &.{ true, false, false, true }, &.{ 2, 2 });
    defer flags.deinit();
    var sorted_flags = try flags.sort(1);
    defer sorted_flags.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, sorted_flags.data);
}

test "array bool all any axis reductions" {
    const gpa = std.testing.allocator;
    var mask = try Array(bool).fromSlice(gpa, &.{ true, true, false, true, false, false }, &.{ 2, 3 });
    defer mask.deinit();
    try std.testing.expect(!mask.all());
    try std.testing.expect(mask.any());

    var all0 = try mask.allAxis(0, false);
    defer all0.deinit();
    try std.testing.expectEqualSlices(usize, &.{3}, all0.shape);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false }, all0.data);

    var any1 = try mask.anyAxis(1, true);
    defer any1.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 1 }, any1.shape);
    try std.testing.expectEqualSlices(bool, &.{ true, true }, any1.data);

    var all_global = try mask.allAxis(null, true);
    defer all_global.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 1 }, all_global.shape);
    try std.testing.expectEqualSlices(bool, &.{false}, all_global.data);

    var view = try mask.sliceAxisView(1, .{ .start = 0, .stop = 3, .step = 2 });
    defer view.deinit();
    try std.testing.expect(!view.all());
    try std.testing.expect(view.any());
    var view_all0 = try view.allAxis(0, false);
    defer view_all0.deinit();
    try std.testing.expectEqualSlices(usize, &.{2}, view_all0.shape);
    try std.testing.expectEqualSlices(bool, &.{ true, false }, view_all0.data);
    var view_any1 = try view.anyAxis(1, true);
    defer view_any1.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 1 }, view_any1.shape);
    try std.testing.expectEqualSlices(bool, &.{ true, true }, view_any1.data);
    var view_all_global = try view.allAxis(null, true);
    defer view_all_global.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 1 }, view_all_global.shape);
    try std.testing.expectEqualSlices(bool, &.{false}, view_all_global.data);
    var not_view = try view.logicalNot();
    defer not_view.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, not_view.data);
    var rhs = try Array(bool).fromSlice(gpa, &.{ true, false }, &.{ 1, 2 });
    defer rhs.deinit();
    var rhs_view = try rhs.asView();
    defer rhs_view.deinit();
    var and_view = try view.logicalAnd(rhs_view);
    defer and_view.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, false }, and_view.data);
    var or_view = try view.logicalOrArray(rhs);
    defer or_view.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, false }, or_view.data);
    var xor_view = try view.logicalXor(rhs_view);
    defer xor_view.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false }, xor_view.data);
    var and_scalar = try view.logicalAndScalar(true);
    defer and_scalar.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, false }, and_scalar.data);
    var or_scalar = try view.logicalOrScalar(true);
    defer or_scalar.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, or_scalar.data);
    var xor_scalar = try view.logicalXorScalar(true);
    defer xor_scalar.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, xor_scalar.data);
    var bad = try Array(bool).fromSlice(gpa, &.{ true, false, true }, &.{3});
    defer bad.deinit();
    var bad_view = try bad.asView();
    defer bad_view.deinit();
    try std.testing.expectError(error.ShapeMismatch, view.logicalAnd(bad_view));
}

test "array aliases and alea-backed random distributions" {
    const gpa = std.testing.allocator;
    var a = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4 }, &.{ 2, 2 });
    defer a.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, a.shape);

    var logs = try Array(f64).logspace(gpa, 0, 2, 3, 10);
    defer logs.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 10, 100 }, logs.data);
    var geoms = try Array(f64).geomspace(gpa, 1, 100, 3);
    defer geoms.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1), geoms.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 10), geoms.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 100), geoms.data[2], 1e-12);

    var xs = try Array(f64).fromSlice(gpa, &.{ 1, 2 }, &.{2});
    defer xs.deinit();
    var ys = try Array(f64).fromSlice(gpa, &.{ 10, 20, 30 }, &.{3});
    defer ys.deinit();
    var grid_xy = try Array(f64).meshgrid(xs, ys, .xy);
    defer grid_xy.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, grid_xy.x.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 1, 2, 1, 2 }, grid_xy.x.data);
    try std.testing.expectEqualSlices(f64, &.{ 10, 10, 20, 20, 30, 30 }, grid_xy.y.data);
    var grid_ij = try Array(f64).meshgrid(xs, ys, .ij);
    defer grid_ij.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, grid_ij.x.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 1, 1, 2, 2, 2 }, grid_ij.x.data);
    try std.testing.expectEqualSlices(f64, &.{ 10, 20, 30, 10, 20, 30 }, grid_ij.y.data);

    var u = try Array(f64).uniform(gpa, &.{16}, -2.0, 3.0, 123);
    defer u.deinit();
    for (u.data) |v| try std.testing.expect(v >= -2.0 and v < 3.0);

    var n = try Array(f64).normal(gpa, &.{8}, 10.0, 0.0, 123);
    defer n.deinit();
    for (n.data) |v| try std.testing.expectEqual(@as(f64, 10.0), v);

    var r = try Array(i64).randint(gpa, &.{32}, 2, 7, 456);
    defer r.deinit();
    for (r.data) |v| try std.testing.expect(v >= 2 and v < 7);

    var b0 = try Array(bool).bernoulli(gpa, &.{4}, 0.0, 789);
    defer b0.deinit();
    try std.testing.expect(!b0.any());
    var b1 = try Array(bool).bernoulli(gpa, &.{4}, 1.0, 789);
    defer b1.deinit();
    try std.testing.expect(b1.all());
}

test "alea-backed object random permutation and sampling" {
    const gpa = std.testing.allocator;
    var source = try Array(i32).fromSlice(gpa, &.{ 10, 20, 30, 40, 50 }, &.{5});
    defer source.deinit();

    var perm = try Array(usize).permutation(gpa, 5, 1234);
    defer perm.deinit();
    try std.testing.expectEqualSlices(usize, &.{5}, perm.shape);
    var sorted_perm = try perm.sort(null);
    defer sorted_perm.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2, 3, 4 }, sorted_perm.data);

    var shuffled = try source.shuffle(5678);
    defer shuffled.deinit();
    var sorted_shuffled = try shuffled.sort(null);
    defer sorted_shuffled.deinit();
    try std.testing.expectEqualSlices(i32, &.{ 10, 20, 30, 40, 50 }, sorted_shuffled.data);
    try std.testing.expectEqualSlices(i32, &.{ 10, 20, 30, 40, 50 }, source.data);

    var in_place = try source.clone();
    defer in_place.deinit();
    in_place.shuffleInPlace(5678);
    try std.testing.expectEqualSlices(i32, shuffled.data, in_place.data);

    var picked = try source.choice(8, true, 9012);
    defer picked.deinit();
    try std.testing.expectEqualSlices(usize, &.{8}, picked.shape);
    for (picked.data) |value| try std.testing.expect(value == 10 or value == 20 or value == 30 or value == 40 or value == 50);

    var picked_unique = try source.choice(3, false, 3456);
    defer picked_unique.deinit();
    try std.testing.expectEqualSlices(usize, &.{3}, picked_unique.shape);
    var picked_sorted = try picked_unique.sort(null);
    defer picked_sorted.deinit();
    for (picked_sorted.data[1..], picked_sorted.data[0 .. picked_sorted.data.len - 1]) |value, prev| {
        try std.testing.expect(value != prev);
    }

    var weights = try Array(f64).fromSlice(gpa, &.{ 0, 0, 1, 0, 0 }, &.{5});
    defer weights.deinit();
    var weighted = try source.choiceWeighted(weights, 4, 7777);
    defer weighted.deinit();
    try std.testing.expectEqualSlices(i32, &.{ 30, 30, 30, 30 }, weighted.data);

    try std.testing.expectError(error.ShapeMismatch, source.choice(6, false, 1));
    var bad_weights = try Array(f64).fromSlice(gpa, &.{ 1, 2 }, &.{2});
    defer bad_weights.deinit();
    try std.testing.expectError(error.ShapeMismatch, source.choiceWeighted(bad_weights, 1, 1));
}

test "alea-backed object multinomial and dirichlet distributions" {
    const gpa = std.testing.allocator;
    var dir = try Array(f64).dirichlet(gpa, &.{ 1.0, 2.0, 3.0 }, 4, 12345);
    defer dir.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 4, 3 }, dir.shape);
    for (0..4) |row| {
        const total = dir.data[row * 3] + dir.data[row * 3 + 1] + dir.data[row * 3 + 2];
        try std.testing.expectApproxEqAbs(@as(f64, 1), total, 1e-12);
        try std.testing.expect(dir.data[row * 3] >= 0);
        try std.testing.expect(dir.data[row * 3 + 1] >= 0);
        try std.testing.expect(dir.data[row * 3 + 2] >= 0);
    }

    var degenerate_dir = try Array(f64).dirichlet(gpa, &.{ 2.0, std.math.inf(f64), 3.0 }, 2, 2222);
    defer degenerate_dir.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 1, 0, 0, 1, 0 }, degenerate_dir.data);

    var counts = try Array(u64).multinomial(gpa, 20, &.{ 1.0, 2.0, 3.0 }, 5, 54321);
    defer counts.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 5, 3 }, counts.shape);
    for (0..5) |row| {
        const total = counts.data[row * 3] + counts.data[row * 3 + 1] + counts.data[row * 3 + 2];
        try std.testing.expectEqual(@as(u64, 20), total);
    }

    var fixed_counts = try Array(u64).multinomial(gpa, 7, &.{ 0.0, 5.0, 0.0 }, 3, 4444);
    defer fixed_counts.deinit();
    try std.testing.expectEqualSlices(u64, &.{ 0, 7, 0, 0, 7, 0, 0, 7, 0 }, fixed_counts.data);

    try std.testing.expectError(error.InvalidShape, Array(f64).dirichlet(gpa, &.{ 1.0, 0.0 }, 1, 1));
    try std.testing.expectError(error.InvalidShape, Array(u64).multinomial(gpa, 5, &.{ 0.0, 0.0 }, 1, 1));
}

test "alea-backed advanced random distributions" {
    const gpa = std.testing.allocator;
    var e = try Array(f64).exponential(gpa, &.{16}, 2.0, 111);
    defer e.deinit();
    for (e.data) |v| try std.testing.expect(v >= 0);

    var g0 = try Array(f64).gamma(gpa, &.{4}, 2.0, 0.0, 222);
    defer g0.deinit();
    for (g0.data) |v| try std.testing.expectEqual(@as(f64, 0), v);

    var be = try Array(f64).beta(gpa, &.{16}, 2.0, 5.0, 333);
    defer be.deinit();
    for (be.data) |v| try std.testing.expect(v >= 0 and v <= 1);

    var p0 = try Array(u64).poisson(gpa, &.{8}, 0.0, 444);
    defer p0.deinit();
    try std.testing.expectEqualSlices(u64, &.{ 0, 0, 0, 0, 0, 0, 0, 0 }, p0.data);
}

test "alea-backed additional continuous distributions" {
    const gpa = std.testing.allocator;
    var ln = try Array(f64).lognormal(gpa, &.{8}, 0.0, 0.0, 555);
    defer ln.deinit();
    for (ln.data) |v| try std.testing.expectApproxEqAbs(@as(f64, 1), v, 1e-12);

    var st = try Array(f64).studentT(gpa, &.{8}, 8.0, 666);
    defer st.deinit();
    for (st.data) |v| try std.testing.expect(std.math.isFinite(v));

    var ca = try Array(f64).cauchy(gpa, &.{8}, 0.0, 1.0, 777);
    defer ca.deinit();
    for (ca.data) |v| try std.testing.expect(std.math.isFinite(v));

    var la = try Array(f64).laplace(gpa, &.{8}, 0.0, 2.0, 888);
    defer la.deinit();
    for (la.data) |v| try std.testing.expect(std.math.isFinite(v));

    var wb = try Array(f64).weibull(gpa, &.{8}, 2.0, 1.5, 999);
    defer wb.deinit();
    for (wb.data) |v| try std.testing.expect(v >= 0);
}

test "alea-backed expanded continuous distributions" {
    const gpa = std.testing.allocator;
    const dims = &.{3};

    var half_normal_zero = try Array(f64).halfNormal(gpa, dims, 0, 101);
    defer half_normal_zero.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 0, 0 }, half_normal_zero.data);

    var chi_squared_zero = try Array(f64).chiSquared(gpa, dims, 0, 102);
    defer chi_squared_zero.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 0, 0 }, chi_squared_zero.data);

    var chi_zero = try Array(f64).chi(gpa, dims, 0, 103);
    defer chi_zero.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 0, 0 }, chi_zero.data);

    var erlang_zero = try Array(f64).erlang(gpa, dims, 2, 0, 104);
    defer erlang_zero.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 0, 0 }, erlang_zero.data);

    var fisher_one = try Array(f64).fisherF(gpa, dims, std.math.inf(f64), std.math.inf(f64), 105);
    defer fisher_one.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 1, 1 }, fisher_one.data);

    var triangular_const = try Array(f64).triangular(gpa, dims, 2, 2, 2, 106);
    defer triangular_const.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 2, 2, 2 }, triangular_const.data);

    var arcsine_const = try Array(f64).arcsine(gpa, dims, 3, 3, 107);
    defer arcsine_const.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 3, 3, 3 }, arcsine_const.data);

    var logistic_const = try Array(f64).logistic(gpa, dims, 4, 0, 108);
    defer logistic_const.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 4, 4, 4 }, logistic_const.data);

    var log_logistic_zero = try Array(f64).logLogistic(gpa, dims, 0, 2, 109);
    defer log_logistic_zero.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 0, 0 }, log_logistic_zero.data);

    var kumaraswamy_one = try Array(f64).kumaraswamy(gpa, dims, std.math.inf(f64), 1, 110);
    defer kumaraswamy_one.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 1, 1 }, kumaraswamy_one.data);

    var power_const = try Array(f64).powerFunction(gpa, dims, 5, 5, 2, 111);
    defer power_const.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 5, 5, 5 }, power_const.data);

    var rayleigh_zero = try Array(f64).rayleigh(gpa, dims, 0, 112);
    defer rayleigh_zero.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 0, 0 }, rayleigh_zero.data);

    var maxwell_zero = try Array(f64).maxwell(gpa, dims, 0, 113);
    defer maxwell_zero.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 0, 0 }, maxwell_zero.data);

    var pareto_zero = try Array(f64).pareto(gpa, dims, 0, 2, 114);
    defer pareto_zero.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 0, 0 }, pareto_zero.data);

    var gumbel_const = try Array(f64).gumbel(gpa, dims, 6, 0, 115);
    defer gumbel_const.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 6, 6, 6 }, gumbel_const.data);

    var frechet_const = try Array(f64).frechet(gpa, dims, 7, 0, 2, 116);
    defer frechet_const.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 7, 7, 7 }, frechet_const.data);

    var skew_const = try Array(f64).skewNormal(gpa, dims, 8, 0, 0, 117);
    defer skew_const.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 8, 8, 8 }, skew_const.data);

    var pert_const = try Array(f64).pert(gpa, dims, 9, 9, 9, 4, 118);
    defer pert_const.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 9, 9, 9 }, pert_const.data);

    var inverse_gaussian_zero = try Array(f64).inverseGaussian(gpa, dims, 0, 1, 119);
    defer inverse_gaussian_zero.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 0, 0 }, inverse_gaussian_zero.data);

    var normal_inverse_gaussian_zero = try Array(f64).normalInverseGaussian(gpa, dims, std.math.inf(f64), 0, 120);
    defer normal_inverse_gaussian_zero.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 0, 0 }, normal_inverse_gaussian_zero.data);

    var bounded = try Array(f64).triangular(gpa, &.{16}, 0, 0.5, 1, 201);
    defer bounded.deinit();
    for (bounded.data) |v| try std.testing.expect(v >= 0 and v <= 1);

    var beta_like = try Array(f64).kumaraswamy(gpa, &.{16}, 2, 3, 202);
    defer beta_like.deinit();
    for (beta_like.data) |v| try std.testing.expect(v >= 0 and v <= 1);

    var pert = try Array(f64).pert(gpa, &.{16}, -1, 0, 2, 4, 203);
    defer pert.deinit();
    for (pert.data) |v| try std.testing.expect(v >= -1 and v <= 2);

    var positive_tail = try Array(f64).pareto(gpa, &.{16}, 1, 2, 204);
    defer positive_tail.deinit();
    for (positive_tail.data) |v| try std.testing.expect(v >= 1 and std.math.isFinite(v));

    var nig = try Array(f64).normalInverseGaussian(gpa, &.{16}, 2, 0.5, 205);
    defer nig.deinit();
    for (nig.data) |v| try std.testing.expect(std.math.isFinite(v));

    try std.testing.expectError(error.InvalidShape, Array(f64).halfNormal(gpa, &.{1}, -1, 1));
    try std.testing.expectError(error.InvalidShape, Array(f64).chiSquared(gpa, &.{1}, -1, 1));
    try std.testing.expectError(error.InvalidShape, Array(f64).erlang(gpa, &.{1}, 0, 1, 1));
    try std.testing.expectError(error.InvalidShape, Array(f64).triangular(gpa, &.{1}, 0, 2, 1, 1));
    try std.testing.expectError(error.InvalidShape, Array(f64).arcsine(gpa, &.{1}, 1, 0, 1));
    try std.testing.expectError(error.InvalidShape, Array(f64).logistic(gpa, &.{1}, 0, -1, 1));
    try std.testing.expectError(error.InvalidShape, Array(f64).logLogistic(gpa, &.{1}, 1, 0, 1));
    try std.testing.expectError(error.InvalidShape, Array(f64).kumaraswamy(gpa, &.{1}, 0, 1, 1));
    try std.testing.expectError(error.InvalidShape, Array(f64).powerFunction(gpa, &.{1}, 1, 0, 1, 1));
    try std.testing.expectError(error.InvalidShape, Array(f64).rayleigh(gpa, &.{1}, -1, 1));
    try std.testing.expectError(error.InvalidShape, Array(f64).maxwell(gpa, &.{1}, -1, 1));
    try std.testing.expectError(error.InvalidShape, Array(f64).pareto(gpa, &.{1}, 1, 0, 1));
    try std.testing.expectError(error.InvalidShape, Array(f64).gumbel(gpa, &.{1}, 0, -1, 1));
    try std.testing.expectError(error.InvalidShape, Array(f64).frechet(gpa, &.{1}, 0, 1, 0, 1));
    try std.testing.expectError(error.InvalidShape, Array(f64).skewNormal(gpa, &.{1}, 0, -1, 0, 1));
    try std.testing.expectError(error.InvalidShape, Array(f64).pert(gpa, &.{1}, 0, 2, 1, 4, 1));
    try std.testing.expectError(error.InvalidShape, Array(f64).inverseGaussian(gpa, &.{1}, 1, 0, 1));
    try std.testing.expectError(error.InvalidShape, Array(f64).normalInverseGaussian(gpa, &.{1}, 1, 1, 1));
}

test "array scatter add and reduce variants" {
    const gpa = std.testing.allocator;
    var base = try Array(f64).zeros(gpa, &.{ 2, 3 });
    defer base.deinit();
    var idx = try Array(usize).fromSlice(gpa, &.{ 0, 1, 1, 2, 0, 2 }, &.{ 2, 3 });
    defer idx.deinit();
    var src = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer src.deinit();

    var added = try base.scatterAdd(1, idx, src);
    defer added.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 5, 0, 5, 0, 10 }, added.data);

    var ones_base = try Array(f64).ones(gpa, &.{ 2, 3 });
    defer ones_base.deinit();
    var product_out = try ones_base.scatterReduce(1, idx, src, .prod);
    defer product_out.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 6, 1, 5, 1, 24 }, product_out.data);

    var max_base = try Array(f64).full(gpa, &.{ 2, 3 }, -100);
    defer max_base.deinit();
    var maxed = try max_base.scatterReduce(1, idx, src, .max);
    defer maxed.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, -100, 5, -100, 6 }, maxed.data);

    var scalar_added = try base.scatterAddScalar(1, idx, 2);
    defer scalar_added.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 2, 4, 0, 2, 0, 4 }, scalar_added.data);
}

test "array creation like scalar diag and diagflat" {
    const gpa = std.testing.allocator;
    var a = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a.deinit();

    var z = try a.zerosLike();
    defer z.deinit();
    try std.testing.expectEqualSlices(usize, a.shape, z.shape);
    try std.testing.expectEqualSlices(f64, &.{ 0, 0, 0, 0, 0, 0 }, z.data);

    var o = try a.onesLike();
    defer o.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 1, 1, 1, 1, 1 }, o.data);

    var f = try a.fullLike(7);
    defer f.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 7, 7, 7, 7, 7, 7 }, f.data);

    var s = try Array(f64).fromScalar(gpa, 42);
    defer s.deinit();
    try std.testing.expectEqual(@as(usize, 0), s.shape.len);
    try std.testing.expectEqual(@as(f64, 42), try s.item());

    var v = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3 }, &.{3});
    defer v.deinit();
    var d0 = try v.diag(0);
    defer d0.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 3 }, d0.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 0, 0, 0, 2, 0, 0, 0, 3 }, d0.data);

    var d1 = try v.diagflat(1);
    defer d1.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 4, 4 }, d1.shape);
    try std.testing.expectEqualSlices(f64, &.{ 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0 }, d1.data);

    var extracted = try a.diag(0);
    defer extracted.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 5 }, extracted.data);

    var ident = try Array(f64).identity(gpa, 3);
    defer ident.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 0, 0, 0, 1, 0, 0, 0, 1 }, ident.data);

    var rect = try Array(f64).eyeRect(gpa, 2, 4, 1);
    defer rect.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 4 }, rect.shape);
    try std.testing.expectEqualSlices(f64, &.{ 0, 1, 0, 0, 0, 0, 1, 0 }, rect.data);

    var neg_rect = try Array(f64).eyeRect(gpa, 3, 2, -1);
    defer neg_rect.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, neg_rect.shape);
    try std.testing.expectEqualSlices(f64, &.{ 0, 0, 1, 0, 0, 1 }, neg_rect.data);

    var out_of_band = try Array(f64).eyeRect(gpa, 2, 2, 3);
    defer out_of_band.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 0, 0, 0 }, out_of_band.data);
}

test "array advanced indexing and mask mutation helpers" {
    const gpa = std.testing.allocator;
    var a = try Array(f64).fromSlice(gpa, &.{ 10, 11, 12, 20, 21, 22 }, &.{ 2, 3 });
    defer a.deinit();
    var idx = try Array(usize).fromSlice(gpa, &.{ 2, 0, 1, 1, 2, 0 }, &.{ 2, 3 });
    defer idx.deinit();

    var taken = try a.takeAlongAxis(idx, 1);
    defer taken.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 12, 10, 11, 21, 22, 20 }, taken.data);

    var base = try Array(f64).zeros(gpa, &.{ 2, 3 });
    defer base.deinit();
    var put = try base.putAlongAxis(idx, taken, 1);
    defer put.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 10, 11, 12, 20, 21, 22 }, put.data);

    var mask = try Array(bool).fromSlice(gpa, &.{ true, false, true, false, true, false }, &.{ 2, 3 });
    defer mask.deinit();
    var filled = try a.maskedFill(mask, -1);
    defer filled.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -1, 11, -1, 20, -1, 22 }, filled.data);

    var src = try Array(f64).fromSlice(gpa, &.{ 100, 200, 300 }, &.{3});
    defer src.deinit();
    var scattered = try a.maskedScatter(mask, src);
    defer scattered.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 100, 11, 200, 20, 300, 22 }, scattered.data);

    var nz = try filled.nonzero();
    defer nz.deinit();
    try std.testing.expectEqual(@as(usize, 6), filled.countNonzero());
    try std.testing.expectEqualSlices(usize, &.{ 6, 2 }, nz.shape);
    try std.testing.expectEqualSlices(usize, &.{ 0, 0, 0, 1, 0, 2, 1, 0, 1, 1, 1, 2 }, nz.data);
}

test "array slice flip roll and constant padding" {
    const gpa = std.testing.allocator;
    var a = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a.deinit();

    var sliced = try a.sliceAxis(1, .{ .start = 0, .stop = 3, .step = 2 });
    defer sliced.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, sliced.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 4, 6 }, sliced.data);
    var sliced_top = try a.sliceAxis(1, .{ .start = 0, .stop = 3, .step = 2 });
    defer sliced_top.deinit();
    try std.testing.expectEqualSlices(f64, sliced.data, sliced_top.data);
    var multi_sliced = try a.slice(&.{
        .{ .start = 0, .stop = 2, .step = 1 },
        .{ .start = 1, .stop = 3, .step = 1 },
    });
    defer multi_sliced.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, multi_sliced.shape);
    try std.testing.expectEqualSlices(f64, &.{ 2, 3, 5, 6 }, multi_sliced.data);
    var v = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4, 5 }, &.{5});
    defer v.deinit();
    var sliced_1d = try v.slice1d(.{ .start = 1, .stop = 5, .step = 2 });
    defer sliced_1d.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 2, 4 }, sliced_1d.data);

    var flipped = try a.flip(1);
    defer flipped.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 3, 2, 1, 6, 5, 4 }, flipped.data);

    var flipped_axes = try a.flipAxes(&.{ 0, -1 });
    defer flipped_axes.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 6, 5, 4, 3, 2, 1 }, flipped_axes.data);

    var flip_clone = try a.flipAxes(&.{});
    defer flip_clone.deinit();
    try std.testing.expectEqualSlices(f64, a.data, flip_clone.data);

    var rolled = try a.roll(1, 1);
    defer rolled.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 3, 1, 2, 6, 4, 5 }, rolled.data);

    var rolled_neg = try a.roll(-1, 0);
    defer rolled_neg.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 4, 5, 6, 1, 2, 3 }, rolled_neg.data);

    var rolled_axes = try a.rollAxes(&.{ 1, -1 }, &.{ 0, 1 });
    defer rolled_axes.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 5, 6, 4, 2, 3, 1 }, rolled_axes.data);

    var rotated = try a.rot90(1, .{ 0, 1 });
    defer rotated.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, rotated.shape);
    try std.testing.expectEqualSlices(f64, &.{ 3, 6, 2, 5, 1, 4 }, rotated.data);

    var rotated_twice = try a.rot90(2, .{ 0, 1 });
    defer rotated_twice.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, rotated_twice.shape);
    try std.testing.expectEqualSlices(f64, &.{ 6, 5, 4, 3, 2, 1 }, rotated_twice.data);

    var rotated_clockwise = try a.rot90(-1, .{ 0, 1 });
    defer rotated_clockwise.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, rotated_clockwise.shape);
    try std.testing.expectEqualSlices(f64, &.{ 4, 1, 5, 2, 6, 3 }, rotated_clockwise.data);

    var rotated_reversed_axes = try a.rot90(1, .{ 1, 0 });
    defer rotated_reversed_axes.deinit();
    try std.testing.expectEqualSlices(f64, rotated_clockwise.data, rotated_reversed_axes.data);

    var rotated_clone = try a.rot90(4, .{ 0, 1 });
    defer rotated_clone.deinit();
    try std.testing.expectEqualSlices(f64, a.data, rotated_clone.data);

    var roll_clone = try a.rollAxes(&.{}, &.{});
    defer roll_clone.deinit();
    try std.testing.expectEqualSlices(f64, a.data, roll_clone.data);

    try std.testing.expectError(error.InvalidAxis, a.flipAxes(&.{ 0, 0 }));
    try std.testing.expectError(error.ShapeMismatch, a.rollAxes(&.{1}, &.{ 0, 1 }));
    try std.testing.expectError(error.InvalidAxis, a.rollAxes(&.{ 1, 1 }, &.{ 0, 0 }));
    try std.testing.expectError(error.InvalidAxis, a.rot90(1, .{ 0, 0 }));
    try std.testing.expectError(error.InvalidAxis, v.rot90(1, .{ 0, 1 }));

    var padded = try a.padConstant(&.{ 1, 1 }, &.{ 0, 2 }, 0);
    defer padded.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 6 }, padded.shape);
    try std.testing.expectEqualSlices(f64, &.{
        0, 0, 0, 0, 0, 0,
        0, 1, 2, 3, 0, 0,
        0, 4, 5, 6, 0, 0,
    }, padded.data);

    var edge_vec = try v.padEdge(&.{2}, &.{1});
    defer edge_vec.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 1, 1, 2, 3, 4, 5, 5 }, edge_vec.data);

    var reflect_vec = try v.padReflect(&.{2}, &.{1});
    defer reflect_vec.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 3, 2, 1, 2, 3, 4, 5, 4 }, reflect_vec.data);
    var wrap_vec = try v.padWrap(&.{2}, &.{1});
    defer wrap_vec.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 4, 5, 1, 2, 3, 4, 5, 1 }, wrap_vec.data);
    var symmetric_vec = try v.padSymmetric(&.{2}, &.{1});
    defer symmetric_vec.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 2, 1, 1, 2, 3, 4, 5, 5 }, symmetric_vec.data);

    var m = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4 }, &.{ 2, 2 });
    defer m.deinit();
    var edge_matrix = try m.padEdge(&.{ 1, 1 }, &.{ 1, 0 });
    defer edge_matrix.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 4, 3 }, edge_matrix.shape);
    try std.testing.expectEqualSlices(f64, &.{
        1, 1, 2,
        1, 1, 2,
        3, 3, 4,
        3, 3, 4,
    }, edge_matrix.data);

    var reflect_matrix = try m.padReflect(&.{ 1, 1 }, &.{ 0, 0 });
    defer reflect_matrix.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 3 }, reflect_matrix.shape);
    try std.testing.expectEqualSlices(f64, &.{
        4, 3, 4,
        2, 1, 2,
        4, 3, 4,
    }, reflect_matrix.data);
    var wrap_matrix = try m.padWrap(&.{ 1, 1 }, &.{ 0, 1 });
    defer wrap_matrix.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 4 }, wrap_matrix.shape);
    try std.testing.expectEqualSlices(f64, &.{
        4, 3, 4, 3,
        2, 1, 2, 1,
        4, 3, 4, 3,
    }, wrap_matrix.data);
    var symmetric_matrix = try m.padSymmetric(&.{ 1, 1 }, &.{ 0, 0 });
    defer symmetric_matrix.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 3 }, symmetric_matrix.shape);
    try std.testing.expectEqualSlices(f64, &.{
        1, 1, 2,
        1, 1, 2,
        3, 3, 4,
    }, symmetric_matrix.data);

    try std.testing.expectError(error.InvalidShape, v.padReflect(&.{5}, &.{0}));
    var empty = try Array(f64).empty(gpa, &.{0});
    defer empty.deinit();
    try std.testing.expectError(error.EmptyArray, empty.padEdge(&.{1}, &.{1}));
}

test "array dtype metadata and casts cover common numeric types" {
    try std.testing.expectEqual(DType.bf16, DType.of(BFloat16));
    try std.testing.expectEqual(DType.f16, DType.of(f16));
    try std.testing.expectEqual(DType.c64, DType.of(Complex64));
    try std.testing.expectEqual(DType.c128, DType.of(Complex128));
    try std.testing.expectEqual(DType.i8, DType.of(i8));
    try std.testing.expectEqual(DType.i16, DType.of(i16));
    try std.testing.expectEqual(DType.isize, DType.of(isize));
    try std.testing.expectEqual(DType.u16, DType.of(u16));
    try std.testing.expectEqual(DType.u32, DType.of(u32));
    try std.testing.expectEqual(DType.u64, DType.of(u64));
    try std.testing.expectEqualStrings("u64", DType.u64.name());
    try std.testing.expectEqual(@as(usize, 8), DType.u64.byteSize());
    try std.testing.expectEqualStrings("isize", DType.isize.name());
    try std.testing.expectEqual(@sizeOf(isize), DType.isize.byteSize());
    try std.testing.expectEqualStrings("bf16", DType.bf16.name());
    try std.testing.expectEqual(@as(usize, 2), DType.bf16.byteSize());
    try std.testing.expectEqualStrings("f16", DType.f16.name());
    try std.testing.expectEqual(@as(usize, 2), DType.f16.byteSize());
    try std.testing.expectEqualStrings("complex64", DType.c64.name());
    try std.testing.expectEqual(@as(usize, 8), DType.c64.byteSize());
    try std.testing.expect(DType.c128.isComplex());
    try std.testing.expect(DType.f32.isFloat());
    try std.testing.expect(DType.i16.isInteger());
    try std.testing.expect(DType.i16.isSigned());
    try std.testing.expect(DType.isize.isSigned());
    try std.testing.expect(DType.bool.isBool());
    try std.testing.expect(DType.bool.canCast(.f32));
    try std.testing.expect(DType.f32.canCast(.c64));
    try std.testing.expect(!DType.c64.canCast(.f32));
    try std.testing.expect(canCastDType(.bf16, .f32));
    try std.testing.expect(canCastDType(.f16, .f64));
    try std.testing.expectEqual(DType.f64, DType.promote(.f32, .f64));
    try std.testing.expectEqual(DType.f32, DType.promote(.bf16, .f32));
    try std.testing.expectEqual(DType.f32, promoteDType(.f16, .f32));
    try std.testing.expectEqual(DType.i32, resultDType(.i16, .u16));
    try std.testing.expectEqual(DType.u64, DType.promote(.u32, .u64));
    try std.testing.expectEqual(DType.i64, DType.promote(.isize, .i32));
    try std.testing.expectEqual(DType.c64, DType.promote(.f32, .c64));
    try std.testing.expectEqual(DType.c128, DType.promote(.f64, .c64));

    const gpa = std.testing.allocator;
    var ints = try Array(i16).fromSlice(gpa, &.{ -1, 0, 2 }, &.{3});
    defer ints.deinit();
    var floats = try ints.astype(f32);
    defer floats.deinit();
    try std.testing.expectEqualSlices(f32, &.{ -1, 0, 2 }, floats.data);
    var unsigned = try Array(u32).fromSlice(gpa, &.{ 1, 2, 3 }, &.{3});
    defer unsigned.deinit();
    var widened = try unsigned.astype(u64);
    defer widened.deinit();
    try std.testing.expectEqualSlices(u64, &.{ 1, 2, 3 }, widened.data);
    var halves = try Array(f16).fromSlice(gpa, &.{ @as(f16, 1.5), @as(f16, -2.0) }, &.{2});
    defer halves.deinit();
    var half_to_float = try halves.astype(f32);
    defer half_to_float.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1.5, -2.0 }, half_to_float.data);
    var brain_halves = try Array(BFloat16).fromSlice(gpa, &.{ BFloat16.fromF32(1.5), BFloat16.fromF32(-2.0) }, &.{2});
    defer brain_halves.deinit();
    var brain_to_float = try brain_halves.astype(f32);
    defer brain_to_float.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), brain_to_float.data[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), brain_to_float.data[1], 1e-3);
    try std.testing.expectEqual(f32, promoteType(f16, f32));
    try std.testing.expectEqual(f32, promoteType(BFloat16, f32));
    try std.testing.expectEqual(i32, promoteType(i16, u16));
    var small_signed = try Array(i16).fromSlice(gpa, &.{ -1, 2, 3 }, &.{3});
    defer small_signed.deinit();
    var small_unsigned = try Array(u16).fromSlice(gpa, &.{ 5, 6, 7 }, &.{3});
    defer small_unsigned.deinit();
    var promoted_sum = try small_signed.addPromote(u16, small_unsigned);
    defer promoted_sum.deinit();
    try std.testing.expectEqual(DType.i32, @TypeOf(promoted_sum).dtype);
    try std.testing.expectEqualSlices(i32, &.{ 4, 8, 10 }, promoted_sum.data);
    var promoted_max = try small_signed.maximumPromote(u16, small_unsigned);
    defer promoted_max.deinit();
    try std.testing.expectEqualSlices(i32, &.{ 5, 6, 7 }, promoted_max.data);
    var promoted_half = try halves.mulPromote(f32, half_to_float);
    defer promoted_half.deinit();
    try std.testing.expectEqual(DType.f32, @TypeOf(promoted_half).dtype);
    try std.testing.expectEqualSlices(f32, &.{ 2.25, 4.0 }, promoted_half.data);
    var promoted_brain = try brain_halves.addPromote(f32, brain_to_float);
    defer promoted_brain.deinit();
    try std.testing.expectEqual(DType.f32, @TypeOf(promoted_brain).dtype);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), promoted_brain.data[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, -4.0), promoted_brain.data[1], 1e-3);

    var view_source = try Array(i16).fromSlice(gpa, &.{ 1, 9, 2, 8, 3, 7 }, &.{ 2, 3 });
    defer view_source.deinit();
    var stepped_view = try view_source.sliceAxisView(1, .{ .start = 0, .stop = 3, .step = 2 });
    defer stepped_view.deinit();
    var view_as_f32 = try stepped_view.astype(f32);
    defer view_as_f32.deinit();
    try std.testing.expectEqual(DType.f32, @TypeOf(view_as_f32).dtype);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 8, 7 }, view_as_f32.data);
    var promote_rhs = try Array(u16).fromSlice(gpa, &.{ 10, 20 }, &.{ 1, 2 });
    defer promote_rhs.deinit();
    var view_promoted_sum = try stepped_view.addPromote(u16, promote_rhs);
    defer view_promoted_sum.deinit();
    try std.testing.expectEqual(DType.i32, @TypeOf(view_promoted_sum).dtype);
    try std.testing.expectEqualSlices(i32, &.{ 11, 22, 18, 27 }, view_promoted_sum.data);
    var view_promoted_mul = try stepped_view.mulPromote(f32, view_as_f32);
    defer view_promoted_mul.deinit();
    try std.testing.expectEqual(DType.f32, @TypeOf(view_promoted_mul).dtype);
    try std.testing.expectEqualSlices(f32, &.{ 1, 4, 64, 49 }, view_promoted_mul.data);
    var view_promoted_max = try stepped_view.maximumPromote(u16, promote_rhs);
    defer view_promoted_max.deinit();
    try std.testing.expectEqualSlices(i32, &.{ 10, 20, 10, 20 }, view_promoted_max.data);

    var r = try Array(u16).randint(gpa, &.{16}, 10, 20, 42);
    defer r.deinit();
    for (r.data) |v| try std.testing.expect(v >= 10 and v < 20);

    var cpu_source = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4 }, &.{ 2, 2 });
    defer cpu_source.deinit();
    var cpu_copy = try cpu_source.cpu();
    defer cpu_copy.deinit();
    try std.testing.expectEqual(Backend.cpu, cpu_copy.device.backend);
    try std.testing.expectEqualSlices(f64, cpu_source.data, cpu_copy.data);
    try std.testing.expect(Device.cpu.isAvailable());
    try std.testing.expect(!Device.cuda(0).isAvailable());
    try std.testing.expectError(error.InvalidDevice, cpu_source.cuda(0));

    var cpu_view = try cpu_source.sliceAxisView(1, .{ .start = 0, .stop = 2, .step = 1 });
    defer cpu_view.deinit();
    var view_cpu = try cpu_view.cpu();
    defer view_cpu.deinit();
    try std.testing.expectEqual(Backend.cpu, view_cpu.device.backend);
    try std.testing.expectEqualSlices(usize, cpu_view.shape, view_cpu.shape);
    try std.testing.expectEqualSlices(usize, cpu_view.strides, view_cpu.strides);
    try std.testing.expectError(error.InvalidDevice, cpu_view.cuda(0));
}

test "array object two dimensional convolution and correlation" {
    const gpa = std.testing.allocator;
    var image = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4 }, &.{ 2, 2 });
    defer image.deinit();
    var kernel = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4 }, &.{ 2, 2 });
    defer kernel.deinit();

    var conv_full = try image.convolve2d(kernel, .full);
    defer conv_full.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 3 }, conv_full.shape);
    try std.testing.expectEqualSlices(f64, &.{
        1, 4,  4,
        6, 20, 16,
        9, 24, 16,
    }, conv_full.data);

    var conv_same = try image.convolve2d(kernel, .same);
    defer conv_same.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, conv_same.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 4, 6, 20 }, conv_same.data);

    var conv_valid = try image.convolve2d(kernel, .valid);
    defer conv_valid.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 1 }, conv_valid.shape);
    try std.testing.expectEqualSlices(f64, &.{20}, conv_valid.data);

    var corr_full = try image.correlate2d(kernel, .full);
    defer corr_full.deinit();
    try std.testing.expectEqualSlices(f64, &.{
        4,  11, 6,
        14, 30, 14,
        6,  11, 4,
    }, corr_full.data);

    var corr_same = try image.correlate2d(kernel, .same);
    defer corr_same.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 4, 11, 14, 30 }, corr_same.data);

    var image_base = try Array(f64).fromSlice(gpa, &.{ 1, 9, 2, 8, 3, 7, 4, 6, 5, 5, 6, 4 }, &.{ 3, 4 });
    defer image_base.deinit();
    var image_view = try image_base.sliceAxisView(1, .{ .start = 0, .stop = 4, .step = 2 });
    defer image_view.deinit();
    var kernel_base = try Array(f64).fromSlice(gpa, &.{ 1, 0, 2, 0, 3, 0, 4, 0 }, &.{ 2, 4 });
    defer kernel_base.deinit();
    var kernel_view = try kernel_base.sliceAxisView(1, .{ .start = 0, .stop = 4, .step = 2 });
    defer kernel_view.deinit();
    var view_conv = try image_view.convolve2d(kernel_view, .valid);
    defer view_conv.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 1 }, view_conv.shape);
    try std.testing.expectEqualSlices(f64, &.{ 20, 40 }, view_conv.data);
    var view_corr = try image_view.correlate2d(kernel_view, .valid);
    defer view_corr.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 30, 50 }, view_corr.data);
    var view_conv_same = try image_view.convolve2dArray(kernel, .same);
    defer view_conv_same.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, view_conv_same.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 4, 6, 20, 14, 40 }, view_conv_same.data);

    var vector = try Array(f64).fromSlice(gpa, &.{ 1, 2 }, &.{2});
    defer vector.deinit();
    try std.testing.expectError(error.NonMatrixArray, vector.convolve2d(kernel, .full));
    var empty_image = try Array(f64).empty(gpa, &.{ 0, 2 });
    defer empty_image.deinit();
    try std.testing.expectError(error.EmptyArray, empty_image.convolve2d(kernel, .full));
}

test "array object one dimensional convolution and correlation" {
    const gpa = std.testing.allocator;
    var signal_values = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3 }, &.{3});
    defer signal_values.deinit();
    var kernel = try Array(f64).fromSlice(gpa, &.{ 1, 2 }, &.{2});
    defer kernel.deinit();

    var conv_full = try signal_values.convolve1d(kernel, .full);
    defer conv_full.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 4, 7, 6 }, conv_full.data);

    var conv_same = try signal_values.convolve1d(kernel, .same);
    defer conv_same.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 4, 7 }, conv_same.data);

    var conv_valid = try signal_values.convolve1d(kernel, .valid);
    defer conv_valid.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 4, 7 }, conv_valid.data);

    var corr_full = try signal_values.correlate1d(kernel, .full);
    defer corr_full.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 2, 5, 8, 3 }, corr_full.data);

    var corr_same = try signal_values.correlate1d(kernel, .same);
    defer corr_same.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 2, 5, 8 }, corr_same.data);

    var long_kernel = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4 }, &.{4});
    defer long_kernel.deinit();
    var valid_reversed = try kernel.convolve1d(long_kernel, .valid);
    defer valid_reversed.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 4, 7, 10 }, valid_reversed.data);

    var signal_base = try Array(f64).fromSlice(gpa, &.{ 1, 9, 2, 8, 3, 7, 4 }, &.{7});
    defer signal_base.deinit();
    var signal_view = try signal_base.sliceAxisView(0, .{ .start = 0, .stop = 7, .step = 2 });
    defer signal_view.deinit();
    var kernel_base = try Array(f64).fromSlice(gpa, &.{ 1, 9, 2 }, &.{3});
    defer kernel_base.deinit();
    var kernel_view = try kernel_base.sliceAxisView(0, .{ .start = 0, .stop = 3, .step = 2 });
    defer kernel_view.deinit();
    var view_conv = try signal_view.convolve1d(kernel_view, .full);
    defer view_conv.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 4, 7, 10, 8 }, view_conv.data);
    var view_corr = try signal_view.correlate1d(kernel_view, .same);
    defer view_corr.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 2, 5, 8, 11 }, view_corr.data);
    var view_conv_array = try signal_view.convolve1dArray(kernel, .valid);
    defer view_conv_array.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 4, 7, 10 }, view_conv_array.data);

    var matrix_values = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4 }, &.{ 2, 2 });
    defer matrix_values.deinit();
    try std.testing.expectError(error.NonVectorArray, matrix_values.convolve1d(kernel, .full));
    var empty_values = try Array(f64).empty(gpa, &.{0});
    defer empty_values.deinit();
    try std.testing.expectError(error.EmptyArray, empty_values.convolve1d(kernel, .full));
}

test "array complex unary math and predicates" {
    const gpa = std.testing.allocator;
    const C = Complex64;
    var values = try Array(C).fromSlice(gpa, &.{ C.init(0, 0), C.init(1, 0), C.init(0, std.math.pi / 2.0) }, &.{3});
    defer values.deinit();

    var exp_values = try values.exp();
    defer exp_values.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 1), exp_values.data[0].re, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0), exp_values.data[0].im, 1e-5);
    try std.testing.expectApproxEqAbs(std.math.e, exp_values.data[1].re, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0), exp_values.data[2].re, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1), exp_values.data[2].im, 1e-5);

    var sqrt_values = try values.sqrt();
    defer sqrt_values.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 1), sqrt_values.data[1].re, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0), sqrt_values.data[1].im, 1e-5);

    var log_values = try exp_values.log();
    defer log_values.deinit();
    try std.testing.expectApproxEqAbs(values.data[1].re, log_values.data[1].re, 1e-5);
    try std.testing.expectApproxEqAbs(values.data[1].im, log_values.data[1].im, 1e-5);

    var sin_values = try values.sin();
    defer sin_values.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 0), sin_values.data[0].re, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0), sin_values.data[0].im, 1e-5);

    var angles = try values.angle();
    defer angles.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 0), angles.data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0), angles.data[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, std.math.pi / 2.0), angles.data[2], 1e-5);
    var phases = try values.phase();
    defer phases.deinit();
    try std.testing.expectEqualSlices(f32, angles.data, phases.data);
    var real_mask = try values.isreal();
    defer real_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, false }, real_mask.data);
    var complex_mask = try values.iscomplex();
    defer complex_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, false, true }, complex_mask.data);

    var special = try Array(C).fromSlice(gpa, &.{ C.init(1, 0), C.init(std.math.nan(f32), 0), C.init(0, std.math.inf(f32)) }, &.{3});
    defer special.deinit();
    var nan_mask = try special.isNan();
    defer nan_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true, false }, nan_mask.data);
    var inf_mask = try special.isInf();
    defer inf_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, false, true }, inf_mask.data);
    var finite_mask = try special.isFinite();
    defer finite_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false, false }, finite_mask.data);
}

test "array complex fft and inverse fft" {
    const gpa = std.testing.allocator;
    const C = Complex64;
    var impulse = try Array(C).fromSlice(gpa, &.{ C.init(1, 0), C.init(0, 0), C.init(0, 0), C.init(0, 0) }, &.{4});
    defer impulse.deinit();

    var spectrum = try impulse.fft();
    defer spectrum.deinit();
    for (spectrum.data) |value| {
        try std.testing.expectApproxEqAbs(@as(f32, 1), value.re, 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, 0), value.im, 1e-5);
    }

    var recovered = try spectrum.ifft();
    defer recovered.deinit();
    for (recovered.data, impulse.data) |actual, expected| {
        try std.testing.expectApproxEqAbs(expected.re, actual.re, 1e-5);
        try std.testing.expectApproxEqAbs(expected.im, actual.im, 1e-5);
    }

    var signal = try Array(C).fromSlice(gpa, &.{ C.init(1, 0), C.init(2, -1), C.init(0, 3), C.init(-2, 0.5) }, &.{4});
    defer signal.deinit();
    var signal_spectrum = try signal.fft();
    defer signal_spectrum.deinit();
    var signal_roundtrip = try signal_spectrum.ifft();
    defer signal_roundtrip.deinit();
    for (signal_roundtrip.data, signal.data) |actual, expected| {
        try std.testing.expectApproxEqAbs(expected.re, actual.re, 1e-4);
        try std.testing.expectApproxEqAbs(expected.im, actual.im, 1e-4);
    }

    var matrix_complex = try Array(C).fromSlice(gpa, &.{ C.init(1, 0), C.init(0, 0), C.init(0, 0), C.init(1, 0) }, &.{ 2, 2 });
    defer matrix_complex.deinit();
    try std.testing.expectError(error.NonVectorArray, matrix_complex.fft());
}

test "array complex axis fft and inverse fft" {
    const gpa = std.testing.allocator;
    const C = Complex64;
    var rows = try Array(C).fromSlice(gpa, &.{
        C.init(1, 0), C.init(0, 0), C.init(0, 0), C.init(0, 0),
        C.init(0, 0), C.init(1, 0), C.init(0, 0), C.init(0, 0),
    }, &.{ 2, 4 });
    defer rows.deinit();

    var row_fft = try rows.fftAxis(1);
    defer row_fft.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 4 }, row_fft.shape);
    for (row_fft.data[0..4]) |value| {
        try std.testing.expectApproxEqAbs(@as(f32, 1), value.re, 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, 0), value.im, 1e-5);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1), row_fft.data[4].re, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0), row_fft.data[4].im, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0), row_fft.data[5].re, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -1), row_fft.data[5].im, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -1), row_fft.data[6].re, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0), row_fft.data[6].im, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0), row_fft.data[7].re, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1), row_fft.data[7].im, 1e-5);

    var row_roundtrip = try row_fft.ifftAxis(1);
    defer row_roundtrip.deinit();
    for (row_roundtrip.data, rows.data) |actual, expected| {
        try std.testing.expectApproxEqAbs(expected.re, actual.re, 1e-5);
        try std.testing.expectApproxEqAbs(expected.im, actual.im, 1e-5);
    }

    var cols = try Array(C).fromSlice(gpa, &.{ C.init(1, 0), C.init(0, 0), C.init(0, 0), C.init(1, 0) }, &.{ 2, 2 });
    defer cols.deinit();
    var col_fft = try cols.fftAxis(0);
    defer col_fft.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 1), col_fft.data[0].re, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1), col_fft.data[1].re, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1), col_fft.data[2].re, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -1), col_fft.data[3].re, 1e-5);

    var transposed = try rows.transposeView();
    defer transposed.deinit();
    var view_fft = try transposed.fftAxis(0);
    defer view_fft.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 4, 2 }, view_fft.shape);
    try std.testing.expectApproxEqAbs(@as(f32, 1), view_fft.data[0].re, 1e-5);
}

test "array real fft and inverse real fft" {
    const gpa = std.testing.allocator;
    var impulse = try Array(f64).fromSlice(gpa, &.{ 1, 0, 0, 0 }, &.{4});
    defer impulse.deinit();
    var spectrum = try impulse.rfft();
    defer spectrum.deinit();
    try std.testing.expectEqualSlices(usize, &.{3}, spectrum.shape);
    for (spectrum.data) |value| {
        try std.testing.expectApproxEqAbs(@as(f64, 1), value.re, 1e-12);
        try std.testing.expectApproxEqAbs(@as(f64, 0), value.im, 1e-12);
    }
    var recovered = try spectrum.irfft(null);
    defer recovered.deinit();
    try std.testing.expectEqualSlices(usize, &.{4}, recovered.shape);
    for (recovered.data, impulse.data) |actual, expected| try std.testing.expectApproxEqAbs(expected, actual, 1e-12);

    var signal = try Array(f64).fromSlice(gpa, &.{ 1, 2, -1, 0.5, 3 }, &.{5});
    defer signal.deinit();
    var signal_spectrum = try signal.rfft();
    defer signal_spectrum.deinit();
    try std.testing.expectEqualSlices(usize, &.{3}, signal_spectrum.shape);
    var signal_recovered = try signal_spectrum.irfft(5);
    defer signal_recovered.deinit();
    for (signal_recovered.data, signal.data) |actual, expected| try std.testing.expectApproxEqAbs(expected, actual, 1e-11);

    var as_f32 = try Array(f32).fromSlice(gpa, &.{ 1, 0, 0, 0 }, &.{4});
    defer as_f32.deinit();
    var spectrum32 = try as_f32.rfft();
    defer spectrum32.deinit();
    try std.testing.expectEqual(DType.c64, @TypeOf(spectrum32).dtype);
    var roundtrip32 = try spectrum32.irfft(null);
    defer roundtrip32.deinit();
    try std.testing.expectEqual(DType.f32, @TypeOf(roundtrip32).dtype);

    var matrix_real = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4 }, &.{ 2, 2 });
    defer matrix_real.deinit();
    try std.testing.expectError(error.NonVectorArray, matrix_real.rfft());
    try std.testing.expectError(error.ShapeMismatch, spectrum.irfft(6));
}

test "array complex two dimensional fft helpers" {
    const gpa = std.testing.allocator;
    const C = Complex64;
    var impulse2 = try Array(C).fromSlice(gpa, &.{ C.init(1, 0), C.init(0, 0), C.init(0, 0), C.init(0, 0) }, &.{ 2, 2 });
    defer impulse2.deinit();
    var spectrum2 = try impulse2.fft2();
    defer spectrum2.deinit();
    for (spectrum2.data) |value| {
        try std.testing.expectApproxEqAbs(@as(f32, 1), value.re, 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, 0), value.im, 1e-5);
    }
    var recovered2 = try spectrum2.ifft2();
    defer recovered2.deinit();
    for (recovered2.data, impulse2.data) |actual, expected| {
        try std.testing.expectApproxEqAbs(expected.re, actual.re, 1e-5);
        try std.testing.expectApproxEqAbs(expected.im, actual.im, 1e-5);
    }

    var signal2 = try Array(C).fromSlice(gpa, &.{ C.init(1, 0), C.init(2, 1), C.init(3, -1), C.init(4, 0) }, &.{ 2, 2 });
    defer signal2.deinit();
    var axes_fft = try signal2.fftAxes(&.{ 0, 1 });
    defer axes_fft.deinit();
    var fft2_out = try signal2.fft2();
    defer fft2_out.deinit();
    for (axes_fft.data, fft2_out.data) |a_value, b_value| {
        try std.testing.expectApproxEqAbs(a_value.re, b_value.re, 1e-5);
        try std.testing.expectApproxEqAbs(a_value.im, b_value.im, 1e-5);
    }
    var roundtrip = try fft2_out.ifftAxes(&.{ 0, 1 });
    defer roundtrip.deinit();
    for (roundtrip.data, signal2.data) |actual, expected| {
        try std.testing.expectApproxEqAbs(expected.re, actual.re, 1e-5);
        try std.testing.expectApproxEqAbs(expected.im, actual.im, 1e-5);
    }

    var view = try signal2.transposeView();
    defer view.deinit();
    var view_fft = try view.fft2();
    defer view_fft.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, view_fft.shape);
    var vector_complex = try Array(C).fromSlice(gpa, &.{C.init(1, 0)}, &.{1});
    defer vector_complex.deinit();
    try std.testing.expectError(error.InvalidAxis, vector_complex.fft2());
}

test "array bfloat16 arithmetic and reductions" {
    const gpa = std.testing.allocator;
    var a = try Array(BFloat16).fromSlice(gpa, &.{
        BFloat16.fromF32(1.25),
        BFloat16.fromF32(2.5),
        BFloat16.fromF32(-3.0),
    }, &.{3});
    defer a.deinit();

    var b = try a.addScalar(BFloat16.fromF32(1.0));
    defer b.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 2.25), b.data[0].toF32(), 1e-2);
    try std.testing.expectApproxEqAbs(@as(f32, 3.5), b.data[1].toF32(), 1e-2);
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), b.data[2].toF32(), 1e-2);

    var product_out = try a.mul(a);
    defer product_out.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 1.5625), product_out.data[0].toF32(), 2e-2);
    try std.testing.expectApproxEqAbs(@as(f32, 6.25), product_out.data[1].toF32(), 2e-2);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), product_out.data[2].toF32(), 2e-2);

    var summed = try a.sum(null, false);
    defer summed.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 0.75), summed.data[0].toF32(), 1e-2);

    var negated = try a.neg();
    defer negated.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, -1.25), negated.data[0].toF32(), 1e-2);
    var abs_out = try a.abs();
    defer abs_out.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), abs_out.data[2].toF32(), 1e-2);
    var cmp_rhs = try Array(BFloat16).fromSlice(gpa, &.{
        BFloat16.fromF32(1.0),
        BFloat16.fromF32(3.0),
        BFloat16.fromF32(-4.0),
    }, &.{3});
    defer cmp_rhs.deinit();
    var gt_mask = try a.gt(cmp_rhs);
    defer gt_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, gt_mask.data);
    var le_mask = try a.le(cmp_rhs);
    defer le_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true, false }, le_mask.data);
    var max_out = try a.maximum(cmp_rhs);
    defer max_out.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 1.25), max_out.data[0].toF32(), 1e-2);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), max_out.data[1].toF32(), 1e-2);
    try std.testing.expectApproxEqAbs(@as(f32, -3.0), max_out.data[2].toF32(), 1e-2);
    var max_reduce = try a.max(null, false);
    defer max_reduce.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), max_reduce.data[0].toF32(), 1e-2);
    var min_reduce = try a.min(null, false);
    defer min_reduce.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, -3.0), min_reduce.data[0].toF32(), 1e-2);
    var min_scalar = try a.minimumScalar(BFloat16.fromF32(2.0));
    defer min_scalar.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 1.25), min_scalar.data[0].toF32(), 1e-2);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), min_scalar.data[1].toF32(), 1e-2);
    try std.testing.expectApproxEqAbs(@as(f32, -3.0), min_scalar.data[2].toF32(), 1e-2);
    try std.testing.expectEqual(@as(usize, 1), try a.argmax());
    try std.testing.expectEqual(@as(usize, 2), try a.argmin());
    var a_row = try a.reshape(&.{ 1, 3 });
    defer a_row.deinit();
    var argmax_axis = try a_row.argmaxAxis(1, false);
    defer argmax_axis.deinit();
    try std.testing.expectEqualSlices(usize, &.{1}, argmax_axis.data);

    var unary_source = try Array(BFloat16).fromSlice(gpa, &.{
        BFloat16.fromF32(0.0),
        BFloat16.fromF32(1.0),
        BFloat16.fromF32(4.0),
    }, &.{3});
    defer unary_source.deinit();
    var exp_out = try unary_source.exp();
    defer exp_out.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), exp_out.data[0].toF32(), 2e-2);
    try std.testing.expectApproxEqAbs(@as(f32, @floatCast(std.math.e)), exp_out.data[1].toF32(), 2e-2);
    var log_out = try unary_source.addScalar(BFloat16.fromF32(1.0));
    defer log_out.deinit();
    var logged = try log_out.log();
    defer logged.deinit();
    try std.testing.expectApproxEqAbs(std.math.log(f32, std.math.e, @as(f32, 2.0)), logged.data[1].toF32(), 2e-2);
    var sqrt_out = try unary_source.sqrt();
    defer sqrt_out.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), sqrt_out.data[2].toF32(), 2e-2);
    var sin_out = try unary_source.sin();
    defer sin_out.deinit();
    try std.testing.expectApproxEqAbs(std.math.sin(@as(f32, 1.0)), sin_out.data[1].toF32(), 2e-2);
    var rounded_source = try Array(BFloat16).fromSlice(gpa, &.{ BFloat16.fromF32(-1.7), BFloat16.fromF32(0.2), BFloat16.fromF32(1.7) }, &.{3});
    defer rounded_source.deinit();
    var floored = try rounded_source.floor();
    defer floored.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, -2), floored.data[0].toF32(), 2e-2);
    var signs = try rounded_source.sign();
    defer signs.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, -1), signs.data[0].toF32(), 2e-2);
    try std.testing.expectApproxEqAbs(@as(f32, 1), signs.data[2].toF32(), 2e-2);
    var sign_bits = try rounded_source.signbit();
    defer sign_bits.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false, false }, sign_bits.data);
    var special = try Array(BFloat16).fromSlice(gpa, &.{ BFloat16.fromF32(1), BFloat16.fromF32(std.math.inf(f32)), BFloat16.fromF32(std.math.nan(f32)) }, &.{3});
    defer special.deinit();
    var finite = try special.isFinite();
    defer finite.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false, false }, finite.data);
    var inf_mask = try special.isInf();
    defer inf_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true, false }, inf_mask.data);
    var nan_mask = try special.isNan();
    defer nan_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, false, true }, nan_mask.data);
    var cleaned_default = try special.nan_to_num_default();
    defer cleaned_default.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 1), cleaned_default.data[0].toF32(), 1e-2);
    try std.testing.expect(cleaned_default.data[1].toF32() > 3e38);
    try std.testing.expectApproxEqAbs(@as(f32, 0), cleaned_default.data[2].toF32(), 1e-2);
}

test "array complex dtype and arithmetic" {
    const gpa = std.testing.allocator;
    const C = Complex64;
    var a = try Array(C).fromSlice(gpa, &.{
        C.init(1, 2),
        C.init(3, -1),
    }, &.{2});
    defer a.deinit();
    var b = try Array(C).fromSlice(gpa, &.{
        C.init(2, 1),
        C.init(-1, 4),
    }, &.{2});
    defer b.deinit();

    var sum_out = try a.add(b);
    defer sum_out.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 3), sum_out.data[0].re, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3), sum_out.data[0].im, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2), sum_out.data[1].re, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3), sum_out.data[1].im, 1e-6);

    var product_out = try a.mul(b);
    defer product_out.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 0), product_out.data[0].re, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5), product_out.data[0].im, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), product_out.data[1].re, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 13), product_out.data[1].im, 1e-6);

    var conjugated = try a.conj();
    defer conjugated.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, -2), conjugated.data[0].im, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), conjugated.data[1].im, 1e-6);

    var real_part = try a.real();
    defer real_part.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 3 }, real_part.data);
    var imag_part = try a.imag();
    defer imag_part.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 2, -1 }, imag_part.data);
    var mag = try a.magnitude();
    defer mag.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, @sqrt(5.0)), mag.data[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, @sqrt(10.0)), mag.data[1], 1e-6);

    var as128 = try a.astype(Complex128);
    defer as128.deinit();
    try std.testing.expectEqual(DType.c128, @TypeOf(as128).dtype);
    try std.testing.expectApproxEqAbs(@as(f64, 1), as128.data[0].re, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2), as128.data[0].im, 1e-12);

    var real_values = try Array(f32).fromSlice(gpa, &.{ 5, 6 }, &.{2});
    defer real_values.deinit();
    var promoted = try real_values.addPromote(Complex64, a);
    defer promoted.deinit();
    try std.testing.expectEqual(DType.c64, @TypeOf(promoted).dtype);
    try std.testing.expectApproxEqAbs(@as(f32, 6), promoted.data[0].re, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2), promoted.data[0].im, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 9), promoted.data[1].re, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -1), promoted.data[1].im, 1e-6);

    var scalar_added = try a.addScalar(C.init(1, -1));
    defer scalar_added.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 2), scalar_added.data[0].re, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), scalar_added.data[0].im, 1e-6);
}

test "array bytes and archive serialization roundtrip" {
    const gpa = std.testing.allocator;
    var a = try Array(i16).fromSlice(gpa, &.{ -1, 2, 300, -400 }, &.{ 2, 2 });
    defer a.deinit();

    const bytes = try a.toBytes(gpa);
    defer gpa.free(bytes);
    try std.testing.expectEqual(@as(usize, 8), bytes.len);
    var from_raw = try Array(i16).fromBytes(gpa, bytes, &.{ 2, 2 });
    defer from_raw.deinit();
    try std.testing.expectEqualSlices(i16, a.data, from_raw.data);
    try std.testing.expectEqualSlices(usize, a.shape, from_raw.shape);

    const archive = try a.toArchive(gpa);
    defer gpa.free(archive);
    var restored = try Array(i16).fromArchive(gpa, archive);
    defer restored.deinit();
    try std.testing.expectEqualSlices(i16, a.data, restored.data);
    try std.testing.expectEqualSlices(usize, a.shape, restored.shape);
    try std.testing.expectError(error.TypeUnsupported, Array(f32).fromArchive(gpa, archive));

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try a.saveArchiveToDir(tmp.dir, std.testing.io, "array.vxarr");
    var loaded = try Array(i16).loadArchiveFromDir(gpa, tmp.dir, std.testing.io, "array.vxarr", .limited(1024));
    defer loaded.deinit();
    try std.testing.expectEqualSlices(i16, a.data, loaded.data);
    try std.testing.expectEqualSlices(usize, a.shape, loaded.shape);
    try std.testing.expectError(error.TypeUnsupported, Array(f32).loadArchiveFromDir(gpa, tmp.dir, std.testing.io, "array.vxarr", .limited(1024)));
}

test "array axis cumulative operations and diff" {
    const gpa = std.testing.allocator;
    var a = try Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a.deinit();

    var cs0 = try a.cumsumAxis(0);
    defer cs0.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 3, 5, 7, 9 }, cs0.data);
    var cs1 = try a.cumsumAxis(1);
    defer cs1.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 6, 4, 9, 15 }, cs1.data);

    var cp1 = try a.cumprodAxis(1);
    defer cp1.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 6, 4, 20, 120 }, cp1.data);
    var cmx1 = try a.cummaxAxis(1);
    defer cmx1.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 3, 4, 5, 6 }, cmx1.data);
    var cmn1 = try a.cumminAxis(1);
    defer cmn1.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 1, 1, 4, 4, 4 }, cmn1.data);
    var lcse1 = try a.logcumsumexpAxis(1);
    defer lcse1.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1), lcse1.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2) + std.math.log1p(std.math.exp(@as(f64, -1))), lcse1.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3) + std.math.log1p(std.math.exp(@as(f64, -1)) + std.math.exp(@as(f64, -2))), lcse1.data[2], 1e-12);

    var d1 = try a.diff(1, 1);
    defer d1.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, d1.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 1, 1, 1 }, d1.data);

    var d2 = try a.diff(1, 2);
    defer d2.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 1 }, d2.shape);
    try std.testing.expectEqualSlices(f64, &.{ 0, 0 }, d2.data);

    var trap_rows = try a.trapezoid(null, 1, 1);
    defer trap_rows.deinit();
    try std.testing.expectEqualSlices(usize, &.{2}, trap_rows.shape);
    try std.testing.expectEqualSlices(f64, &.{ 4, 10 }, trap_rows.data);
    var x_values = try Array(f64).fromSlice(gpa, &.{ 0, 1, 3 }, &.{3});
    defer x_values.deinit();
    var trap_rows_x = try a.trapezoid(x_values, 1, 1);
    defer trap_rows_x.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 6.5, 15.5 }, trap_rows_x.data);
    var trap_cols = try a.trapz(null, 2, 0);
    defer trap_cols.deinit();
    try std.testing.expectEqualSlices(usize, &.{3}, trap_cols.shape);
    try std.testing.expectEqualSlices(f64, &.{ 5, 7, 9 }, trap_cols.data);
    var short = try Array(f64).fromSlice(gpa, &.{5}, &.{1});
    defer short.deinit();
    var short_trap = try short.trapezoid(null, 1, 0);
    defer short_trap.deinit();
    try std.testing.expectEqual(@as(usize, 0), short_trap.shape.len);
    try std.testing.expectEqual(@as(f64, 0), short_trap.data[0]);

    var view = try a.transposeView();
    defer view.deinit();
    var view_trap = try view.trapezoid(null, 1, 0);
    defer view_trap.deinit();
    try std.testing.expectEqualSlices(usize, &.{2}, view_trap.shape);
    try std.testing.expectEqualSlices(f64, &.{ 4, 10 }, view_trap.data);
    var view_trap_x = try view.trapz(x_values, 1, 0);
    defer view_trap_x.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 6.5, 15.5 }, view_trap_x.data);
    try std.testing.expectError(error.ShapeMismatch, a.trapezoid(short, 1, 1));

    var grad_rows = try a.gradient(null, 1, 1);
    defer grad_rows.deinit();
    try std.testing.expectEqualSlices(usize, a.shape, grad_rows.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 1, 1, 1, 1, 1 }, grad_rows.data);
    var grad_cols = try a.gradient(null, 2, 0);
    defer grad_cols.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1.5, 1.5, 1.5, 1.5, 1.5, 1.5 }, grad_cols.data);
    var y_quad = try Array(f64).fromSlice(gpa, &.{ 0, 1, 9 }, &.{3});
    defer y_quad.deinit();
    var x_quad = try Array(f64).fromSlice(gpa, &.{ 0, 1, 3 }, &.{3});
    defer x_quad.deinit();
    var grad_quad = try y_quad.gradient(x_quad, 1, 0);
    defer grad_quad.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 4 }, grad_quad.data);
    var view_grad = try view.gradient(null, 1, 0);
    defer view_grad.deinit();
    try std.testing.expectEqualSlices(usize, view.shape, view_grad.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 1, 1, 1, 1, 1 }, view_grad.data);
    var view_grad_x = try view.gradient(x_values, 1, 0);
    defer view_grad_x.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 1, 2.0 / 3.0, 2.0 / 3.0, 0.5, 0.5 }, view_grad_x.data);
    try std.testing.expectError(error.ShapeMismatch, a.gradient(short, 1, 1));
}

test "array unique bincount searchsorted and clipArray" {
    const gpa = std.testing.allocator;
    var a = try Array(i32).fromSlice(gpa, &.{ 3, 1, 2, 3, 2, 1, 4 }, &.{7});
    defer a.deinit();
    var u = try a.unique();
    defer u.deinit();
    try std.testing.expectEqualSlices(i32, &.{ 1, 2, 3, 4 }, u.data);
    var b = try Array(i32).fromSlice(gpa, &.{ 2, 4, 4, 5 }, &.{4});
    defer b.deinit();
    var uni = try a.union1d(b);
    defer uni.deinit();
    try std.testing.expectEqualSlices(i32, &.{ 1, 2, 3, 4, 5 }, uni.data);
    var inter = try a.intersect1d(b);
    defer inter.deinit();
    try std.testing.expectEqualSlices(i32, &.{ 2, 4 }, inter.data);
    var diff_set = try a.setdiff1d(b);
    defer diff_set.deinit();
    try std.testing.expectEqualSlices(i32, &.{ 1, 3 }, diff_set.data);
    var xor_set = try a.setxor1d(b);
    defer xor_set.deinit();
    try std.testing.expectEqualSlices(i32, &.{ 1, 3, 5 }, xor_set.data);
    var uc = try a.uniqueWithCounts();
    defer uc.deinit();
    try std.testing.expectEqualSlices(i32, &.{ 1, 2, 3, 4 }, uc.values.data);
    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 2, 1 }, uc.counts.data);

    var counts = try a.bincount(6);
    defer counts.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 2, 2, 1, 0 }, counts.data);
    var weights = try Array(f64).fromSlice(gpa, &.{ 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0 }, &.{7});
    defer weights.deinit();
    var weighted_counts = try a.bincountWeighted(f64, weights, 6);
    defer weighted_counts.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 4, 4, 2.5, 4, 0 }, weighted_counts.data);

    var sorted = try Array(f64).fromSlice(gpa, &.{ 1, 2, 2, 4 }, &.{4});
    defer sorted.deinit();
    var probes = try Array(f64).fromSlice(gpa, &.{ 0, 2, 3, 5 }, &.{4});
    defer probes.deinit();
    var left = try sorted.searchsorted(probes, .left);
    defer left.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 3, 4 }, left.data);
    var right = try sorted.searchsorted(probes, .right);
    defer right.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 3, 3, 4 }, right.data);
    var buckets = try probes.bucketize(sorted, .right);
    defer buckets.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 3, 3, 4 }, buckets.data);
    var digits_left_open = try probes.digitize(sorted, false);
    defer digits_left_open.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 3, 3, 4 }, digits_left_open.data);
    var digits_right_open = try probes.digitize(sorted, true);
    defer digits_right_open.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 3, 4 }, digits_right_open.data);

    var needles = try Array(i32).fromSlice(gpa, &.{ 2, 4 }, &.{2});
    defer needles.deinit();
    var members = try a.isin(needles, false);
    defer members.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, true, false, true }, members.data);
    var non_members = try a.isin(needles, true);
    defer non_members.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true, false, true, false }, non_members.data);

    var flags = try Array(bool).fromSlice(gpa, &.{ true, false, true }, &.{3});
    defer flags.deinit();
    var unique_flags = try flags.unique();
    defer unique_flags.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true }, unique_flags.data);

    var matrix_i = try Array(i32).fromSlice(gpa, &.{ 3, 1, 2, 3, 2, 1, 4, 2 }, &.{ 2, 4 });
    defer matrix_i.deinit();
    var view_i = try matrix_i.sliceAxisView(1, .{ .start = 0, .stop = 4, .step = 2 });
    defer view_i.deinit();
    var view_unique_counts = try view_i.uniqueWithCounts();
    defer view_unique_counts.deinit();
    try std.testing.expectEqualSlices(i32, &.{ 2, 3, 4 }, view_unique_counts.values.data);
    try std.testing.expectEqualSlices(usize, &.{ 2, 1, 1 }, view_unique_counts.counts.data);
    var other_i = try Array(i32).fromSlice(gpa, &.{ 2, 5 }, &.{2});
    defer other_i.deinit();
    var other_i_view = try other_i.asView();
    defer other_i_view.deinit();
    var union_view = try view_i.union1d(other_i_view);
    defer union_view.deinit();
    try std.testing.expectEqualSlices(i32, &.{ 2, 3, 4, 5 }, union_view.data);
    var inter_view = try view_i.intersect1dArray(other_i);
    defer inter_view.deinit();
    try std.testing.expectEqualSlices(i32, &.{2}, inter_view.data);
    var diff_view = try view_i.setdiff1d(other_i_view);
    defer diff_view.deinit();
    try std.testing.expectEqualSlices(i32, &.{ 3, 4 }, diff_view.data);
    var xor_view = try view_i.setxor1dArray(other_i);
    defer xor_view.deinit();
    try std.testing.expectEqualSlices(i32, &.{ 3, 4, 5 }, xor_view.data);
    var small_codes = try Array(u32).fromSlice(gpa, &.{ 0, 1, 2, 1, 2, 1 }, &.{ 2, 3 });
    defer small_codes.deinit();
    var codes_view = try small_codes.sliceAxisView(1, .{ .start = 0, .stop = 3, .step = 2 });
    defer codes_view.deinit();
    var code_counts = try codes_view.bincount(4);
    defer code_counts.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 1, 0 }, code_counts.data);
    var code_weights = try Array(f64).fromSlice(gpa, &.{ 0.5, 1.5, 2.5, 3.5 }, &.{4});
    defer code_weights.deinit();
    var weighted_code_counts = try codes_view.bincountWeighted(f64, code_weights, 4);
    defer weighted_code_counts.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0.5, 6.0, 1.5, 0 }, weighted_code_counts.data);

    var x = try Array(f64).fromSlice(gpa, &.{ -1, 0, 5, 10 }, &.{ 2, 2 });
    defer x.deinit();
    var lo = try Array(f64).fromSlice(gpa, &.{ 0, 2 }, &.{2});
    defer lo.deinit();
    var hi = try Array(f64).fromSlice(gpa, &.{4}, &.{1});
    defer hi.deinit();
    var clipped = try x.clipArray(lo, hi);
    defer clipped.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 2, 4, 4 }, clipped.data);

    var hist_source = try Array(f64).fromSlice(gpa, &.{ 0, 1, 2, 3, 4, 5 }, &.{ 2, 3 });
    defer hist_source.deinit();
    var hist_view = try hist_source.sliceAxisView(1, .{ .start = 0, .stop = 3, .step = 2 });
    defer hist_view.deinit();
    var hist = try hist_view.histogram(3, .{ .min = 0, .max = 6 });
    defer hist.counts.deinit();
    defer hist.edges.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 1 }, hist.counts.data);
    try std.testing.expectEqualSlices(f64, &.{ 0, 2, 4, 6 }, hist.edges.data);
}
