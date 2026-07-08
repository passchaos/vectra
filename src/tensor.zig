const std = @import("std");
const alea = @import("alea");

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
            // code today. A future backend can make this true without changing tensor
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

    pub fn of(comptime T: type) DType {
        return switch (T) {
            f32 => .f32,
            f64 => .f64,
            i8 => .i8,
            i16 => .i16,
            i32 => .i32,
            i64 => .i64,
            u8 => .u8,
            u16 => .u16,
            u32 => .u32,
            u64 => .u64,
            usize => .usize,
            bool => .bool,
            else => @compileError("unsupported Vectra dtype: " ++ @typeName(T)),
        };
    }

    pub fn name(self: DType) []const u8 {
        return switch (self) {
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
        };
    }

    pub fn byteSize(self: DType) usize {
        return switch (self) {
            .bool, .i8, .u8 => 1,
            .i16, .u16 => 2,
            .f32, .i32, .u32 => 4,
            .f64, .i64, .u64 => 8,
            .usize => @sizeOf(usize),
        };
    }

    pub fn isFloat(self: DType) bool {
        return switch (self) {
            .f32, .f64 => true,
            else => false,
        };
    }

    pub fn isInteger(self: DType) bool {
        return switch (self) {
            .i8, .i16, .i32, .i64, .u8, .u16, .u32, .u64, .usize => true,
            else => false,
        };
    }

    pub fn isSigned(self: DType) bool {
        return switch (self) {
            .i8, .i16, .i32, .i64 => true,
            else => false,
        };
    }

    pub fn isBool(self: DType) bool {
        return self == .bool;
    }

    pub fn tag(self: DType) u8 {
        return @intFromEnum(self);
    }

    pub fn fromTag(tag_value: u8) ?DType {
        return switch (tag_value) {
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
            else => null,
        };
    }
};

pub const Archive = struct {
    pub const magic = [_]u8{ 'V', 'X', 'A', 'R', 'R', '0', '1', 0 };
    pub const version: u8 = 1;
};

pub const TensorError = error{
    ShapeMismatch,
    InvalidShape,
    InvalidAxis,
    InvalidDevice,
    InvalidPermutation,
    IndexOutOfBounds,
    NonMatrixTensor,
    NonVectorTensor,
    EmptyTensor,
    TypeUnsupported,
} || std.mem.Allocator.Error;

pub const Shape = struct {
    allocator: std.mem.Allocator,
    dims: []usize,

    pub fn init(allocator: std.mem.Allocator, dims: []const usize) TensorError!Shape {
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

    pub fn numel(self: Shape) TensorError!usize {
        return numelFrom(self.dims);
    }
};

pub fn numelFrom(dims: []const usize) TensorError!usize {
    if (dims.len == 0) return 1;
    var n: usize = 1;
    for (dims) |d| {
        n = std.math.mul(usize, n, d) catch return error.InvalidShape;
    }
    return n;
}

pub fn stridesFor(allocator: std.mem.Allocator, dims: []const usize) TensorError![]usize {
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
    return @typeInfo(T) == .float;
}

fn isNumeric(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .int, .float, .comptime_int, .comptime_float => true,
        else => false,
    };
}

fn ensureNumeric(comptime T: type) void {
    if (comptime !isNumeric(T)) @compileError("operation requires a numeric tensor, got " ++ @typeName(T));
}

fn ensureFloat(comptime T: type) void {
    if (comptime !isFloat(T)) @compileError("operation requires a floating-point tensor, got " ++ @typeName(T));
}

fn ensureOrderable(comptime T: type) void {
    switch (@typeInfo(T)) {
        .bool, .int, .float, .comptime_int, .comptime_float => {},
        else => @compileError("ordering requires a bool or numeric tensor, got " ++ @typeName(T)),
    }
}

fn lessValue(comptime T: type, a: T, b: T) bool {
    return switch (@typeInfo(T)) {
        .bool => !a and b,
        .int, .float, .comptime_int, .comptime_float => a < b,
        else => @compileError("ordering requires a bool or numeric tensor, got " ++ @typeName(T)),
    };
}

fn castValue(comptime T: type, value: anytype) T {
    const V = @TypeOf(value);
    return switch (@typeInfo(T)) {
        .float => switch (@typeInfo(V)) {
            .float, .comptime_float => @floatCast(value),
            .int, .comptime_int => @floatFromInt(value),
            .bool => if (value) 1 else 0,
            else => @compileError("cannot cast " ++ @typeName(V) ++ " to " ++ @typeName(T)),
        },
        .int => switch (@typeInfo(V)) {
            .int, .comptime_int => @intCast(value),
            .float, .comptime_float => @intFromFloat(value),
            .bool => if (value) 1 else 0,
            else => @compileError("cannot cast " ++ @typeName(V) ++ " to " ++ @typeName(T)),
        },
        .bool => switch (@typeInfo(V)) {
            .bool => value,
            .int, .comptime_int => value != 0,
            .float, .comptime_float => value != 0,
            else => @compileError("cannot cast " ++ @typeName(V) ++ " to bool"),
        },
        else => @compileError("unsupported tensor scalar type: " ++ @typeName(T)),
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
    return switch (@typeInfo(T)) {
        .bool => a or b,
        else => a + b,
    };
}

fn mulValue(comptime T: type, a: T, b: T) T {
    return switch (@typeInfo(T)) {
        .bool => a and b,
        else => a * b,
    };
}

fn absValue(comptime T: type, value: T) T {
    return switch (@typeInfo(T)) {
        .int => if (@typeInfo(T).int.signedness == .signed and value < 0) -value else value,
        .float => @abs(value),
        else => @compileError("abs requires a numeric tensor"),
    };
}

fn normalizeDim(dim: isize, rank: usize) TensorError!usize {
    const signed_rank: isize = @intCast(rank);
    const normalized = if (dim < 0) signed_rank + dim else dim;
    if (normalized < 0 or normalized >= signed_rank) return error.InvalidAxis;
    return @intCast(normalized);
}

fn canonicalAxis(axis: usize, rank: usize) TensorError!usize {
    if (axis >= rank) return error.InvalidAxis;
    return axis;
}

fn product(dims: []const usize) usize {
    var out: usize = 1;
    for (dims) |d| out *= d;
    return out;
}

fn normalizeIndex(index: isize, len: usize) TensorError!usize {
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

fn broadcastShape(allocator: std.mem.Allocator, a: []const usize, b: []const usize) TensorError![]usize {
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

fn normalizeSlice(s: Slice, len: usize) TensorError!struct { start: usize, stop: usize, step: usize, count: usize } {
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

pub fn Tensor(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        data: []T,
        shape: []usize,
        strides: []usize,
        device: Device = .cpu,

        pub const Scalar = T;
        pub const dtype = DType.of(T);

        pub fn init(allocator: std.mem.Allocator, dims: []const usize) TensorError!Self {
            const n = try numelFrom(dims);
            const values = try allocator.alloc(T, n);
            @memset(values, zero(T));
            const shape = try allocator.dupe(usize, dims);
            errdefer allocator.free(shape);
            const strides = try stridesFor(allocator, shape);
            return .{ .allocator = allocator, .data = values, .shape = shape, .strides = strides };
        }

        pub fn full(allocator: std.mem.Allocator, dims: []const usize, value: T) TensorError!Self {
            const out = try Self.init(allocator, dims);
            @memset(out.data, value);
            return out;
        }

        pub fn zeros(allocator: std.mem.Allocator, dims: []const usize) TensorError!Self {
            return Self.full(allocator, dims, zero(T));
        }

        pub fn ones(allocator: std.mem.Allocator, dims: []const usize) TensorError!Self {
            return Self.full(allocator, dims, one(T));
        }

        pub fn empty(allocator: std.mem.Allocator, dims: []const usize) TensorError!Self {
            const n = try numelFrom(dims);
            const values = try allocator.alloc(T, n);
            const shape = try allocator.dupe(usize, dims);
            errdefer allocator.free(shape);
            const strides = try stridesFor(allocator, shape);
            return .{ .allocator = allocator, .data = values, .shape = shape, .strides = strides };
        }

        pub fn fromScalar(allocator: std.mem.Allocator, value: T) TensorError!Self {
            return Self.fromSlice(allocator, &.{value}, &.{});
        }

        pub fn emptyLike(self: Self) TensorError!Self {
            return Self.empty(self.allocator, self.shape);
        }

        pub fn zerosLike(self: Self) TensorError!Self {
            return Self.zeros(self.allocator, self.shape);
        }

        pub fn onesLike(self: Self) TensorError!Self {
            return Self.ones(self.allocator, self.shape);
        }

        pub fn fullLike(self: Self, value: T) TensorError!Self {
            return Self.full(self.allocator, self.shape, value);
        }

        pub fn arange(allocator: std.mem.Allocator, start: T, stop: T, step: T) TensorError!Self {
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

        pub fn linspace(allocator: std.mem.Allocator, start: T, stop: T, count: usize) TensorError!Self {
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

        pub fn rand(allocator: std.mem.Allocator, dims: []const usize, seed: u64) TensorError!Self {
            return Self.uniform(allocator, dims, zero(T), one(T), seed);
        }

        pub fn fromSlice(allocator: std.mem.Allocator, values: []const T, dims: []const usize) TensorError!Self {
            const n = try numelFrom(dims);
            if (values.len != n) return error.ShapeMismatch;
            const data = try allocator.dupe(T, values);
            errdefer allocator.free(data);
            const shape = try allocator.dupe(usize, dims);
            errdefer allocator.free(shape);
            const strides = try stridesFor(allocator, shape);
            return .{ .allocator = allocator, .data = data, .shape = shape, .strides = strides };
        }

        pub fn fromNested2D(allocator: std.mem.Allocator, comptime rows: usize, comptime cols: usize, values: [rows][cols]T) TensorError!Self {
            var out = try Self.empty(allocator, &.{ rows, cols });
            for (0..rows) |r| {
                @memcpy(out.data[r * cols ..][0..cols], values[r][0..]);
            }
            return out;
        }

        pub fn eye(allocator: std.mem.Allocator, n: usize) TensorError!Self {
            const out = try Self.zeros(allocator, &.{ n, n });
            for (0..n) |i| out.data[i * n + i] = one(T);
            return out;
        }

        pub fn randn(allocator: std.mem.Allocator, dims: []const usize, seed: u64) TensorError!Self {
            return Self.normal(allocator, dims, zero(T), one(T), seed);
        }

        pub fn uniform(allocator: std.mem.Allocator, dims: []const usize, low: T, high: T, seed: u64) TensorError!Self {
            if (comptime !isNumeric(T)) @compileError("uniform requires a numeric array type");
            if (low > high) return error.InvalidShape;
            var engine = alea.ScalarPrng.init(seed);
            const rng = alea.Rng.init(&engine);
            const out = try Self.empty(allocator, dims);
            for (out.data) |*slot| slot.* = alea.distributions.uniform(rng, T, low, high);
            return out;
        }

        pub fn normal(allocator: std.mem.Allocator, dims: []const usize, mean_value: T, stddev_value: T, seed: u64) TensorError!Self {
            ensureFloat(T);
            if (stddev_value < zero(T)) return error.InvalidShape;
            var engine = alea.ScalarPrng.init(seed);
            const rng = alea.Rng.init(&engine);
            const out = try Self.empty(allocator, dims);
            for (out.data) |*slot| slot.* = alea.distributions.normal(rng, T, mean_value, stddev_value);
            return out;
        }

        pub fn randint(allocator: std.mem.Allocator, dims: []const usize, low: T, high: T, seed: u64) TensorError!Self {
            if (comptime @typeInfo(T) != .int) @compileError("randint requires an integer array type");
            return Self.uniform(allocator, dims, low, high, seed);
        }

        pub fn bernoulli(allocator: std.mem.Allocator, dims: []const usize, p: f64, seed: u64) TensorError!Self {
            if (comptime T != bool) @compileError("bernoulli requires Array(bool)");
            if (p < 0 or p > 1) return error.InvalidShape;
            var engine = alea.ScalarPrng.init(seed);
            const rng = alea.Rng.init(&engine);
            const out = try Self.empty(allocator, dims);
            for (out.data) |*slot| slot.* = alea.distributions.bernoulli(rng, p);
            return out;
        }

        pub fn exponential(allocator: std.mem.Allocator, dims: []const usize, rate: T, seed: u64) TensorError!Self {
            ensureFloat(T);
            if (!(rate > zero(T))) return error.InvalidShape;
            var engine = alea.ScalarPrng.init(seed);
            const rng = alea.Rng.init(&engine);
            const out = try Self.empty(allocator, dims);
            for (out.data) |*slot| slot.* = alea.distributions.exponentialChecked(rng, T, rate) catch return error.InvalidShape;
            return out;
        }

        pub fn gamma(allocator: std.mem.Allocator, dims: []const usize, shape_param: T, scale: T, seed: u64) TensorError!Self {
            ensureFloat(T);
            if (!(shape_param > zero(T)) or !(scale >= zero(T))) return error.InvalidShape;
            var engine = alea.ScalarPrng.init(seed);
            const rng = alea.Rng.init(&engine);
            const out = try Self.empty(allocator, dims);
            for (out.data) |*slot| slot.* = alea.distributions.gammaChecked(rng, T, shape_param, scale) catch return error.InvalidShape;
            return out;
        }

        pub fn beta(allocator: std.mem.Allocator, dims: []const usize, alpha: T, beta_param: T, seed: u64) TensorError!Self {
            ensureFloat(T);
            if (!(alpha > zero(T)) or !(beta_param > zero(T))) return error.InvalidShape;
            var engine = alea.ScalarPrng.init(seed);
            const rng = alea.Rng.init(&engine);
            const out = try Self.empty(allocator, dims);
            for (out.data) |*slot| slot.* = alea.distributions.betaChecked(rng, T, alpha, beta_param) catch return error.InvalidShape;
            return out;
        }

        pub fn lognormal(allocator: std.mem.Allocator, dims: []const usize, mean_value: T, stddev_value: T, seed: u64) TensorError!Self {
            ensureFloat(T);
            if (stddev_value < zero(T)) return error.InvalidShape;
            var engine = alea.ScalarPrng.init(seed);
            const rng = alea.Rng.init(&engine);
            const out = try Self.empty(allocator, dims);
            for (out.data) |*slot| slot.* = alea.distributions.logNormalChecked(rng, T, mean_value, stddev_value) catch return error.InvalidShape;
            return out;
        }

        pub fn studentT(allocator: std.mem.Allocator, dims: []const usize, dof: T, seed: u64) TensorError!Self {
            ensureFloat(T);
            if (!(dof > zero(T))) return error.InvalidShape;
            var engine = alea.ScalarPrng.init(seed);
            const rng = alea.Rng.init(&engine);
            const out = try Self.empty(allocator, dims);
            for (out.data) |*slot| slot.* = alea.distributions.studentTChecked(rng, T, dof) catch return error.InvalidShape;
            return out;
        }

        pub fn cauchy(allocator: std.mem.Allocator, dims: []const usize, median_value: T, scale: T, seed: u64) TensorError!Self {
            ensureFloat(T);
            if (!(scale > zero(T))) return error.InvalidShape;
            var engine = alea.ScalarPrng.init(seed);
            const rng = alea.Rng.init(&engine);
            const out = try Self.empty(allocator, dims);
            for (out.data) |*slot| slot.* = alea.distributions.cauchyChecked(rng, T, median_value, scale) catch return error.InvalidShape;
            return out;
        }

        pub fn laplace(allocator: std.mem.Allocator, dims: []const usize, location: T, scale: T, seed: u64) TensorError!Self {
            ensureFloat(T);
            if (!(scale > zero(T))) return error.InvalidShape;
            var engine = alea.ScalarPrng.init(seed);
            const rng = alea.Rng.init(&engine);
            const out = try Self.empty(allocator, dims);
            for (out.data) |*slot| slot.* = alea.distributions.laplaceChecked(rng, T, location, scale) catch return error.InvalidShape;
            return out;
        }

        pub fn weibull(allocator: std.mem.Allocator, dims: []const usize, scale: T, shape_param: T, seed: u64) TensorError!Self {
            ensureFloat(T);
            if (!(scale > zero(T)) or !(shape_param > zero(T))) return error.InvalidShape;
            var engine = alea.ScalarPrng.init(seed);
            const rng = alea.Rng.init(&engine);
            const out = try Self.empty(allocator, dims);
            for (out.data) |*slot| slot.* = alea.distributions.weibullChecked(rng, T, scale, shape_param) catch return error.InvalidShape;
            return out;
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.data);
            self.allocator.free(self.shape);
            self.allocator.free(self.strides);
            self.* = undefined;
        }

        pub fn clone(self: Self) TensorError!Self {
            return Self.fromSlice(self.allocator, self.data, self.shape);
        }

        pub fn astype(self: Self, comptime U: type) TensorError!Tensor(U) {
            const out = try Tensor(U).empty(self.allocator, self.shape);
            for (self.data, out.data) |v, *slot| {
                slot.* = castValue(U, v);
            }
            return out;
        }

        pub fn to(self: Self, device: Device) TensorError!Self {
            if (!device.isAvailable()) return error.InvalidDevice;
            var out = try self.clone();
            out.device = device;
            return out;
        }

        pub fn cpu(self: Self) TensorError!Self {
            return self.to(.cpu);
        }

        pub fn cuda(self: Self, index: usize) TensorError!Self {
            return self.to(Device.cuda(index));
        }

        pub fn numel(self: Self) usize {
            return self.data.len;
        }

        pub fn ndim(self: Self) usize {
            return self.shape.len;
        }

        pub fn size(self: Self, axis_opt: ?isize) TensorError!usize {
            if (axis_opt) |d| return self.shape[try normalizeDim(d, self.shape.len)];
            return self.numel();
        }

        pub fn len(self: Self) TensorError!usize {
            if (self.shape.len == 0) return error.InvalidShape;
            return self.shape[0];
        }

        pub fn stride(self: Self, axis_index: isize) TensorError!usize {
            return self.strides[try normalizeDim(axis_index, self.shape.len)];
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

        pub fn contiguous(self: Self) TensorError!Self {
            return self.clone();
        }

        pub fn isScalar(self: Self) bool {
            return self.shape.len == 0 or (self.shape.len == 1 and self.shape[0] == 1);
        }

        fn offsetOf(self: Self, indices: []const usize) TensorError!usize {
            if (indices.len != self.shape.len) return error.InvalidShape;
            var offset: usize = 0;
            for (indices, self.shape, self.strides) |idx, extent, stride_value| {
                if (idx >= extent) return error.IndexOutOfBounds;
                offset += idx * stride_value;
            }
            return offset;
        }

        pub fn get(self: Self, indices: []const usize) TensorError!T {
            return self.data[try self.offsetOf(indices)];
        }

        pub fn set(self: *Self, indices: []const usize, value: T) TensorError!void {
            self.data[try self.offsetOf(indices)] = value;
        }

        pub fn at(self: Self, indices: []const usize) TensorError!T {
            return self.get(indices);
        }

        pub fn put(self: *Self, indices: []const usize, value: T) TensorError!void {
            return self.set(indices, value);
        }

        pub fn item(self: Self) TensorError!T {
            if (!self.isScalar()) return error.ShapeMismatch;
            if (self.data.len == 0) return error.EmptyTensor;
            return self.data[0];
        }

        pub fn reshape(self: Self, dims: []const usize) TensorError!Self {
            const n = try numelFrom(dims);
            if (n != self.data.len) return error.ShapeMismatch;
            var out = try self.clone();
            out.allocator.free(out.shape);
            out.allocator.free(out.strides);
            out.shape = try out.allocator.dupe(usize, dims);
            out.strides = try stridesFor(out.allocator, out.shape);
            return out;
        }

        pub fn flatten(self: Self) TensorError!Self {
            return self.reshape(&.{self.data.len});
        }

        pub fn ravel(self: Self) TensorError!Self {
            return self.flatten();
        }

        pub fn view(self: Self, dims: []const usize) TensorError!Self {
            return self.reshape(dims);
        }

        pub fn squeeze(self: Self, axis_opt: ?isize) TensorError!Self {
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

        pub fn unsqueeze(self: Self, axis_index: isize) TensorError!Self {
            const rank = self.shape.len + 1;
            const axis = if (axis_index < 0) blk: {
                const signed_rank: isize = @intCast(rank);
                const normalized = signed_rank + axis_index;
                if (normalized < 0 or normalized >= signed_rank) return error.InvalidAxis;
                break :blk @as(usize, @intCast(normalized));
            } else try canonicalAxis(@intCast(axis_index), rank);
            var dims = try self.allocator.alloc(usize, rank);
            defer self.allocator.free(dims);
            for (self.shape[0..axis], 0..) |d, i| dims[i] = d;
            dims[axis] = 1;
            for (self.shape[axis..], axis + 1..) |d, i| dims[i] = d;
            return self.reshape(dims);
        }

        pub fn broadcastTo(self: Self, dims: []const usize) TensorError!Self {
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

        pub fn repeat(self: Self, repeats: usize, axis_index: isize) TensorError!Self {
            if (self.shape.len == 0) return error.InvalidAxis;
            const axis = try normalizeDim(axis_index, self.shape.len);
            var out_shape = try self.allocator.dupe(usize, self.shape);
            defer self.allocator.free(out_shape);
            out_shape[axis] *= repeats;
            const out = try Self.empty(self.allocator, out_shape);
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

        pub fn sliceAxis(self: Self, axis_index: isize, slice: Slice) TensorError!Self {
            if (self.shape.len == 0) return error.InvalidAxis;
            const axis = try normalizeDim(axis_index, self.shape.len);
            const ns = try normalizeSlice(slice, self.shape[axis]);
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

        pub fn flip(self: Self, axis_index: isize) TensorError!Self {
            if (self.shape.len == 0) return error.InvalidAxis;
            const axis = try normalizeDim(axis_index, self.shape.len);
            const out = try Self.empty(self.allocator, self.shape);
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

        pub fn roll(self: Self, shift: isize, axis_index: isize) TensorError!Self {
            if (self.shape.len == 0) return error.InvalidAxis;
            const axis = try normalizeDim(axis_index, self.shape.len);
            const len_axis = self.shape[axis];
            if (len_axis == 0) return self.clone();
            const signed_len: isize = @intCast(len_axis);
            const normalized_shift: usize = @intCast(@mod(shift, signed_len));
            const out = try Self.empty(self.allocator, self.shape);
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

        pub fn padConstant(self: Self, before: []const usize, after: []const usize, value: T) TensorError!Self {
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

        pub fn tile(self: Self, repeats: []const usize) TensorError!Self {
            if (repeats.len != self.shape.len) return error.ShapeMismatch;
            var out_shape = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(out_shape);
            for (self.shape, repeats, 0..) |d, r, i| out_shape[i] = d * r;
            const out = try Self.empty(self.allocator, out_shape);
            if (out.data.len == 0) return out;
            const out_multi = try self.allocator.alloc(usize, out_shape.len);
            defer self.allocator.free(out_multi);
            var in_multi = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(in_multi);
            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, out_shape, out_multi);
                for (out_multi, self.shape, 0..) |coord, d, i| in_multi[i] = if (d == 0) 0 else coord % d;
                slot.* = self.data[ravelIndex(in_multi, self.strides)];
            }
            return out;
        }

        pub fn transpose(self: Self) TensorError!Self {
            if (self.shape.len != 2) return error.NonMatrixTensor;
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

        pub fn T_(self: Self) TensorError!Self {
            return self.transpose();
        }

        pub fn swapaxes(self: Self, dim0: isize, dim1: isize) TensorError!Self {
            const a0 = try normalizeDim(dim0, self.shape.len);
            const a1 = try normalizeDim(dim1, self.shape.len);
            var perm = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(perm);
            for (perm, 0..) |*slot, i| slot.* = i;
            std.mem.swap(usize, &perm[a0], &perm[a1]);
            return self.permute(perm);
        }

        pub fn permute(self: Self, axes: []const usize) TensorError!Self {
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

        pub fn movedim(self: Self, source: isize, destination: isize) TensorError!Self {
            const src = try normalizeDim(source, self.shape.len);
            const dst = try normalizeDim(destination, self.shape.len);
            var axes = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(axes);
            var write: usize = 0;
            for (0..self.shape.len) |i| {
                if (i == dst) {
                    axes[write] = src;
                    write += 1;
                }
                if (i != src) {
                    axes[write] = i;
                    write += 1;
                }
            }
            return self.permute(axes);
        }

        pub fn slice1d(self: Self, slice: Slice) TensorError!Self {
            if (self.shape.len != 1) return error.NonVectorTensor;
            const ns = try normalizeSlice(slice, self.shape[0]);
            const out = try Self.empty(self.allocator, &.{ns.count});
            var idx = ns.start;
            for (out.data) |*slot| {
                slot.* = self.data[idx];
                idx += ns.step;
            }
            return out;
        }

        pub fn select(self: Self, axis_index: isize, index: usize) TensorError!Self {
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

        pub fn narrow(self: Self, axis_index: isize, start: usize, length: usize) TensorError!Self {
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

        pub fn take(self: Self, indices: Tensor(usize), axis_opt: ?isize) TensorError!Self {
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

        pub fn indexSelect(self: Self, axis_index: isize, indices: Tensor(usize)) TensorError!Self {
            return self.take(indices, axis_index);
        }

        pub fn takeAlongAxis(self: Self, indices: Tensor(usize), axis_index: isize) TensorError!Self {
            return self.gather(axis_index, indices);
        }

        pub fn putAlongAxis(self: Self, indices: Tensor(usize), src: Self, axis_index: isize) TensorError!Self {
            return self.scatter(axis_index, indices, src);
        }

        pub fn maskedSelect(self: Self, mask: Tensor(bool)) TensorError!Self {
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

        pub fn maskedFill(self: Self, mask: Tensor(bool), value: T) TensorError!Self {
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

        pub fn maskedScatter(self: Self, mask: Tensor(bool), src: Self) TensorError!Self {
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

        pub fn maskedPut(self: Self, mask: Tensor(bool), values: Self) TensorError!Self {
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

        pub fn maskedPutScalar(self: Self, mask: Tensor(bool), value: T) TensorError!Self {
            return self.maskedFill(mask, value);
        }

        pub fn putFlat(self: Self, indices: Tensor(usize), values: Self) TensorError!Self {
            if (values.data.len != 1 and values.data.len != indices.data.len) return error.ShapeMismatch;
            var out = try self.clone();
            errdefer out.deinit();
            for (indices.data, 0..) |idx, i| {
                if (idx >= out.data.len) return error.IndexOutOfBounds;
                out.data[idx] = values.data[if (values.data.len == 1) 0 else i];
            }
            return out;
        }

        pub fn putFlatScalar(self: Self, indices: Tensor(usize), value: T) TensorError!Self {
            var out = try self.clone();
            errdefer out.deinit();
            for (indices.data) |idx| {
                if (idx >= out.data.len) return error.IndexOutOfBounds;
                out.data[idx] = value;
            }
            return out;
        }

        pub fn indexPut(self: Self, indices: Tensor(usize), values: Self) TensorError!Self {
            return self.putFlat(indices, values);
        }

        pub fn indexPutScalar(self: Self, indices: Tensor(usize), value: T) TensorError!Self {
            return self.putFlatScalar(indices, value);
        }

        pub fn countNonzero(self: Self) usize {
            var count: usize = 0;
            for (self.data) |v| {
                if (v != zero(T)) count += 1;
            }
            return count;
        }

        pub fn flatNonzero(self: Self) TensorError!Tensor(usize) {
            const count = self.countNonzero();
            const out = try Tensor(usize).empty(self.allocator, &.{count});
            var write: usize = 0;
            for (self.data, 0..) |value, flat| {
                if (value == zero(T)) continue;
                out.data[write] = flat;
                write += 1;
            }
            return out;
        }

        pub fn nonzero(self: Self) TensorError!Tensor(usize) {
            const count = self.countNonzero();
            const out = try Tensor(usize).empty(self.allocator, &.{ count, self.shape.len });
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

        pub fn argwhere(self: Self) TensorError!Tensor(usize) {
            return self.nonzero();
        }

        pub fn compress(self: Self, condition: Tensor(bool), axis_opt: ?isize) TensorError!Self {
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

        pub fn gather(self: Self, axis_index: isize, indices: Tensor(usize)) TensorError!Self {
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

        pub fn scatter(self: Self, axis_index: isize, indices: Tensor(usize), src: Self) TensorError!Self {
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

        pub fn scatterScalar(self: Self, axis_index: isize, indices: Tensor(usize), value: T) TensorError!Self {
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

        fn validateScatterShapes(self: Self, axis: usize, indices: Tensor(usize), src_shape: []const usize) TensorError!void {
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

        pub fn scatterReduce(self: Self, axis_index: isize, indices: Tensor(usize), src: Self, reduction: ScatterReduce) TensorError!Self {
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

        pub fn scatterAdd(self: Self, axis_index: isize, indices: Tensor(usize), src: Self) TensorError!Self {
            return self.scatterReduce(axis_index, indices, src, .sum);
        }

        pub fn scatterReduceScalar(self: Self, axis_index: isize, indices: Tensor(usize), value: T, reduction: ScatterReduce) TensorError!Self {
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

        pub fn scatterAddScalar(self: Self, axis_index: isize, indices: Tensor(usize), value: T) TensorError!Self {
            return self.scatterReduceScalar(axis_index, indices, value, .sum);
        }

        fn binaryTensor(self: Self, other: Self, comptime op: fn (T, T) T) TensorError!Self {
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

        fn binaryScalar(self: Self, scalar: T, comptime op: fn (T, T) T) TensorError!Self {
            const out = try Self.empty(self.allocator, self.shape);
            for (self.data, out.data) |v, *slot| slot.* = op(v, scalar);
            return out;
        }

        fn unary(self: Self, comptime op: fn (T) T) TensorError!Self {
            const out = try Self.empty(self.allocator, self.shape);
            for (self.data, out.data) |v, *slot| slot.* = op(v);
            return out;
        }

        fn unaryBool(self: Self, comptime op: fn (T) bool) TensorError!Tensor(bool) {
            const out = try Tensor(bool).empty(self.allocator, self.shape);
            for (self.data, out.data) |v, *slot| slot.* = op(v);
            return out;
        }

        fn opAdd(a: T, b: T) T {
            return addValue(T, a, b);
        }
        fn opSub(a: T, b: T) T {
            return a - b;
        }
        fn opMul(a: T, b: T) T {
            return mulValue(T, a, b);
        }
        fn opDiv(a: T, b: T) T {
            return a / b;
        }
        fn opPow(a: T, b: T) T {
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
        fn opCopysign(a: T, b: T) T {
            return std.math.copysign(a, b);
        }
        fn opHeaviside(a: T, b: T) T {
            return if (a < zero(T)) zero(T) else if (a > zero(T)) one(T) else b;
        }
        fn opNeg(a: T) T {
            return -a;
        }
        fn opAbs(a: T) T {
            return absValue(T, a);
        }
        fn opExp(a: T) T {
            return std.math.exp(a);
        }
        fn opLog(a: T) T {
            return std.math.log(T, std.math.e, a);
        }
        fn opLog2(a: T) T {
            return std.math.log2(a);
        }
        fn opLog10(a: T) T {
            return std.math.log10(a);
        }
        fn opSqrt(a: T) T {
            return std.math.sqrt(a);
        }
        fn opSin(a: T) T {
            return std.math.sin(a);
        }
        fn opCos(a: T) T {
            return std.math.cos(a);
        }
        fn opTan(a: T) T {
            return std.math.tan(a);
        }
        fn opAsin(a: T) T {
            return std.math.asin(a);
        }
        fn opAcos(a: T) T {
            return std.math.acos(a);
        }
        fn opAtan(a: T) T {
            return std.math.atan(a);
        }
        fn opSinh(a: T) T {
            return std.math.sinh(a);
        }
        fn opCosh(a: T) T {
            return std.math.cosh(a);
        }
        fn opLog1p(a: T) T {
            return std.math.log1p(a);
        }
        fn opExpm1(a: T) T {
            return std.math.expm1(a);
        }
        fn opDeg2rad(a: T) T {
            return a * castValue(T, std.math.pi / 180.0);
        }
        fn opRad2deg(a: T) T {
            return a * castValue(T, 180.0 / std.math.pi);
        }
        fn opFloor(a: T) T {
            return switch (@typeInfo(T)) {
                .float => @floor(a),
                .int, .comptime_int => a,
                else => @compileError("floor requires a numeric array"),
            };
        }
        fn opCeil(a: T) T {
            return switch (@typeInfo(T)) {
                .float => @ceil(a),
                .int, .comptime_int => a,
                else => @compileError("ceil requires a numeric array"),
            };
        }
        fn opRound(a: T) T {
            return switch (@typeInfo(T)) {
                .float => @round(a),
                .int, .comptime_int => a,
                else => @compileError("round requires a numeric array"),
            };
        }
        fn opTrunc(a: T) T {
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
            return one(T) / a;
        }
        fn opSign(a: T) T {
            return switch (@typeInfo(T)) {
                .float => if (std.math.isNan(a)) a else if (a > zero(T)) one(T) else if (a < zero(T)) -one(T) else zero(T),
                .int => |info| if (a == 0) zero(T) else if (info.signedness == .signed) (if (a < 0) -one(T) else one(T)) else one(T),
                .comptime_int, .comptime_float => if (a > 0) 1 else if (a < 0) -1 else 0,
                else => @compileError("sign requires a numeric array"),
            };
        }
        fn opIsNan(a: T) bool {
            return switch (@typeInfo(T)) {
                .float => std.math.isNan(a),
                .int, .comptime_int => false,
                else => @compileError("isNan requires a numeric array"),
            };
        }
        fn opIsInf(a: T) bool {
            return switch (@typeInfo(T)) {
                .float => std.math.isInf(a),
                .int, .comptime_int => false,
                else => @compileError("isInf requires a numeric array"),
            };
        }
        fn opIsFinite(a: T) bool {
            return switch (@typeInfo(T)) {
                .float => std.math.isFinite(a),
                .int, .comptime_int => true,
                else => @compileError("isFinite requires a numeric array"),
            };
        }
        fn opSignbit(a: T) bool {
            return switch (@typeInfo(T)) {
                .float => std.math.signbit(a),
                .int => |info| if (info.signedness == .signed) a < 0 else false,
                .comptime_int => a < 0,
                else => @compileError("signbit requires a numeric array"),
            };
        }

        pub fn add(self: Self, other: Self) TensorError!Self {
            ensureNumeric(T);
            return self.binaryTensor(other, opAdd);
        }

        pub fn sub(self: Self, other: Self) TensorError!Self {
            ensureNumeric(T);
            return self.binaryTensor(other, opSub);
        }

        pub fn mul(self: Self, other: Self) TensorError!Self {
            ensureNumeric(T);
            return self.binaryTensor(other, opMul);
        }

        pub fn div(self: Self, other: Self) TensorError!Self {
            ensureNumeric(T);
            return self.binaryTensor(other, opDiv);
        }

        pub fn pow(self: Self, other: Self) TensorError!Self {
            ensureNumeric(T);
            return self.binaryTensor(other, opPow);
        }

        pub fn floorDiv(self: Self, other: Self) TensorError!Self {
            ensureNumeric(T);
            return self.binaryTensor(other, opFloorDiv);
        }

        pub fn mod(self: Self, other: Self) TensorError!Self {
            ensureNumeric(T);
            return self.binaryTensor(other, opMod);
        }

        pub fn remainder(self: Self, other: Self) TensorError!Self {
            return self.mod(other);
        }

        pub fn hypot(self: Self, other: Self) TensorError!Self {
            ensureFloat(T);
            return self.binaryTensor(other, opHypot);
        }

        pub fn atan2(self: Self, other: Self) TensorError!Self {
            ensureFloat(T);
            return self.binaryTensor(other, opAtan2);
        }

        pub fn copysign(self: Self, sign_values: Self) TensorError!Self {
            ensureFloat(T);
            return self.binaryTensor(sign_values, opCopysign);
        }

        pub fn heaviside(self: Self, values_at_zero: Self) TensorError!Self {
            ensureNumeric(T);
            return self.binaryTensor(values_at_zero, opHeaviside);
        }

        pub fn maximum(self: Self, other: Self) TensorError!Self {
            ensureNumeric(T);
            return self.binaryTensor(other, struct {
                fn f(a: T, b: T) T {
                    return if (a >= b) a else b;
                }
            }.f);
        }

        pub fn minimum(self: Self, other: Self) TensorError!Self {
            ensureNumeric(T);
            return self.binaryTensor(other, struct {
                fn f(a: T, b: T) T {
                    return if (a <= b) a else b;
                }
            }.f);
        }

        pub fn addScalar(self: Self, scalar: T) TensorError!Self {
            ensureNumeric(T);
            return self.binaryScalar(scalar, opAdd);
        }

        pub fn subScalar(self: Self, scalar: T) TensorError!Self {
            ensureNumeric(T);
            return self.binaryScalar(scalar, opSub);
        }

        pub fn mulScalar(self: Self, scalar: T) TensorError!Self {
            ensureNumeric(T);
            return self.binaryScalar(scalar, opMul);
        }

        pub fn divScalar(self: Self, scalar: T) TensorError!Self {
            ensureNumeric(T);
            return self.binaryScalar(scalar, opDiv);
        }

        pub fn powScalar(self: Self, scalar: T) TensorError!Self {
            ensureNumeric(T);
            return self.binaryScalar(scalar, opPow);
        }

        pub fn floorDivScalar(self: Self, scalar: T) TensorError!Self {
            ensureNumeric(T);
            return self.binaryScalar(scalar, opFloorDiv);
        }

        pub fn modScalar(self: Self, scalar: T) TensorError!Self {
            ensureNumeric(T);
            return self.binaryScalar(scalar, opMod);
        }

        pub fn remainderScalar(self: Self, scalar: T) TensorError!Self {
            return self.modScalar(scalar);
        }

        pub fn maximumScalar(self: Self, scalar: T) TensorError!Self {
            ensureNumeric(T);
            return self.binaryScalar(scalar, struct {
                fn f(a: T, b: T) T {
                    return if (a >= b) a else b;
                }
            }.f);
        }

        pub fn minimumScalar(self: Self, scalar: T) TensorError!Self {
            ensureNumeric(T);
            return self.binaryScalar(scalar, struct {
                fn f(a: T, b: T) T {
                    return if (a <= b) a else b;
                }
            }.f);
        }

        pub fn hypotScalar(self: Self, scalar: T) TensorError!Self {
            ensureFloat(T);
            return self.binaryScalar(scalar, opHypot);
        }

        pub fn atan2Scalar(self: Self, scalar: T) TensorError!Self {
            ensureFloat(T);
            return self.binaryScalar(scalar, opAtan2);
        }

        pub fn copysignScalar(self: Self, scalar: T) TensorError!Self {
            ensureFloat(T);
            return self.binaryScalar(scalar, opCopysign);
        }

        pub fn heavisideScalar(self: Self, value_at_zero: T) TensorError!Self {
            ensureNumeric(T);
            return self.binaryScalar(value_at_zero, opHeaviside);
        }

        pub fn neg(self: Self) TensorError!Self {
            ensureNumeric(T);
            return self.unary(opNeg);
        }

        pub fn abs(self: Self) TensorError!Self {
            ensureNumeric(T);
            return self.unary(opAbs);
        }

        pub fn square(self: Self) TensorError!Self {
            ensureNumeric(T);
            return self.unary(opSquare);
        }

        pub fn reciprocal(self: Self) TensorError!Self {
            ensureFloat(T);
            return self.unary(opReciprocal);
        }

        pub fn sign(self: Self) TensorError!Self {
            ensureNumeric(T);
            return self.unary(opSign);
        }

        pub fn signbit(self: Self) TensorError!Tensor(bool) {
            ensureNumeric(T);
            return self.unaryBool(opSignbit);
        }

        pub fn exp(self: Self) TensorError!Self {
            ensureFloat(T);
            return self.unary(opExp);
        }

        pub fn expm1(self: Self) TensorError!Self {
            ensureFloat(T);
            return self.unary(opExpm1);
        }

        pub fn log(self: Self) TensorError!Self {
            ensureFloat(T);
            return self.unary(opLog);
        }

        pub fn log2(self: Self) TensorError!Self {
            ensureFloat(T);
            return self.unary(opLog2);
        }

        pub fn log10(self: Self) TensorError!Self {
            ensureFloat(T);
            return self.unary(opLog10);
        }

        pub fn log1p(self: Self) TensorError!Self {
            ensureFloat(T);
            return self.unary(opLog1p);
        }

        pub fn sqrt(self: Self) TensorError!Self {
            ensureFloat(T);
            return self.unary(opSqrt);
        }

        pub fn floor(self: Self) TensorError!Self {
            ensureNumeric(T);
            return self.unary(opFloor);
        }

        pub fn ceil(self: Self) TensorError!Self {
            ensureNumeric(T);
            return self.unary(opCeil);
        }

        pub fn round(self: Self) TensorError!Self {
            ensureNumeric(T);
            return self.unary(opRound);
        }

        pub fn trunc(self: Self) TensorError!Self {
            ensureNumeric(T);
            return self.unary(opTrunc);
        }

        pub fn deg2rad(self: Self) TensorError!Self {
            ensureFloat(T);
            return self.unary(opDeg2rad);
        }

        pub fn rad2deg(self: Self) TensorError!Self {
            ensureFloat(T);
            return self.unary(opRad2deg);
        }

        pub fn sin(self: Self) TensorError!Self {
            ensureFloat(T);
            return self.unary(opSin);
        }

        pub fn cos(self: Self) TensorError!Self {
            ensureFloat(T);
            return self.unary(opCos);
        }

        pub fn tan(self: Self) TensorError!Self {
            ensureFloat(T);
            return self.unary(opTan);
        }

        pub fn asin(self: Self) TensorError!Self {
            ensureFloat(T);
            return self.unary(opAsin);
        }

        pub fn acos(self: Self) TensorError!Self {
            ensureFloat(T);
            return self.unary(opAcos);
        }

        pub fn atan(self: Self) TensorError!Self {
            ensureFloat(T);
            return self.unary(opAtan);
        }

        pub fn sinh(self: Self) TensorError!Self {
            ensureFloat(T);
            return self.unary(opSinh);
        }

        pub fn cosh(self: Self) TensorError!Self {
            ensureFloat(T);
            return self.unary(opCosh);
        }

        pub fn tanh(self: Self) TensorError!Self {
            ensureFloat(T);
            return self.unary(struct {
                fn f(a: T) T {
                    return std.math.tanh(a);
                }
            }.f);
        }

        pub fn relu(self: Self) TensorError!Self {
            ensureNumeric(T);
            return self.unary(struct {
                fn f(a: T) T {
                    return if (a > zero(T)) a else zero(T);
                }
            }.f);
        }

        pub fn sigmoid(self: Self) TensorError!Self {
            ensureFloat(T);
            return self.unary(struct {
                fn f(a: T) T {
                    return one(T) / (one(T) + std.math.exp(-a));
                }
            }.f);
        }

        pub fn clip(self: Self, min_value: T, max_value: T) TensorError!Self {
            ensureNumeric(T);
            const out = try Self.empty(self.allocator, self.shape);
            for (self.data, out.data) |v, *slot| slot.* = @min(@max(v, min_value), max_value);
            return out;
        }

        pub fn clamp(self: Self, min_value: T, max_value: T) TensorError!Self {
            return self.clip(min_value, max_value);
        }

        pub fn isNan(self: Self) TensorError!Tensor(bool) {
            ensureNumeric(T);
            return self.unaryBool(opIsNan);
        }

        pub fn isnan(self: Self) TensorError!Tensor(bool) {
            return self.isNan();
        }

        pub fn isInf(self: Self) TensorError!Tensor(bool) {
            ensureNumeric(T);
            return self.unaryBool(opIsInf);
        }

        pub fn isinf(self: Self) TensorError!Tensor(bool) {
            return self.isInf();
        }

        pub fn isFinite(self: Self) TensorError!Tensor(bool) {
            ensureNumeric(T);
            return self.unaryBool(opIsFinite);
        }

        pub fn isfinite(self: Self) TensorError!Tensor(bool) {
            return self.isFinite();
        }

        pub fn logsumexp(self: Self, axis_index: isize, keepdims: bool) TensorError!Self {
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

        pub fn logSoftmax(self: Self, axis_index: isize) TensorError!Self {
            ensureFloat(T);
            var lse = try self.logsumexp(axis_index, true);
            defer lse.deinit();
            return self.sub(lse);
        }

        pub fn log_softmax(self: Self, axis_index: isize) TensorError!Self {
            return self.logSoftmax(axis_index);
        }

        pub fn eq(self: Self, other: Self) TensorError!Tensor(bool) {
            return self.compare(other, struct {
                fn f(a: T, b: T) bool {
                    return a == b;
                }
            }.f);
        }

        pub fn equal(self: Self, other: Self) TensorError!Tensor(bool) {
            return self.eq(other);
        }

        pub fn gt(self: Self, other: Self) TensorError!Tensor(bool) {
            ensureNumeric(T);
            return self.compare(other, struct {
                fn f(a: T, b: T) bool {
                    return a > b;
                }
            }.f);
        }

        pub fn greater(self: Self, other: Self) TensorError!Tensor(bool) {
            return self.gt(other);
        }

        pub fn lt(self: Self, other: Self) TensorError!Tensor(bool) {
            ensureNumeric(T);
            return self.compare(other, struct {
                fn f(a: T, b: T) bool {
                    return a < b;
                }
            }.f);
        }

        pub fn less(self: Self, other: Self) TensorError!Tensor(bool) {
            return self.lt(other);
        }

        pub fn ne(self: Self, other: Self) TensorError!Tensor(bool) {
            return self.compare(other, struct {
                fn f(a: T, b: T) bool {
                    return a != b;
                }
            }.f);
        }

        pub fn notEqual(self: Self, other: Self) TensorError!Tensor(bool) {
            return self.ne(other);
        }

        pub fn ge(self: Self, other: Self) TensorError!Tensor(bool) {
            ensureNumeric(T);
            return self.compare(other, struct {
                fn f(a: T, b: T) bool {
                    return a >= b;
                }
            }.f);
        }

        pub fn greaterEqual(self: Self, other: Self) TensorError!Tensor(bool) {
            return self.ge(other);
        }

        pub fn le(self: Self, other: Self) TensorError!Tensor(bool) {
            ensureNumeric(T);
            return self.compare(other, struct {
                fn f(a: T, b: T) bool {
                    return a <= b;
                }
            }.f);
        }

        pub fn lessEqual(self: Self, other: Self) TensorError!Tensor(bool) {
            return self.le(other);
        }

        pub fn eqScalar(self: Self, scalar: T) TensorError!Tensor(bool) {
            return self.compareScalar(scalar, struct {
                fn f(a: T, b: T) bool {
                    return a == b;
                }
            }.f);
        }

        pub fn neScalar(self: Self, scalar: T) TensorError!Tensor(bool) {
            return self.compareScalar(scalar, struct {
                fn f(a: T, b: T) bool {
                    return a != b;
                }
            }.f);
        }

        pub fn gtScalar(self: Self, scalar: T) TensorError!Tensor(bool) {
            ensureNumeric(T);
            return self.compareScalar(scalar, struct {
                fn f(a: T, b: T) bool {
                    return a > b;
                }
            }.f);
        }

        pub fn geScalar(self: Self, scalar: T) TensorError!Tensor(bool) {
            ensureNumeric(T);
            return self.compareScalar(scalar, struct {
                fn f(a: T, b: T) bool {
                    return a >= b;
                }
            }.f);
        }

        pub fn ltScalar(self: Self, scalar: T) TensorError!Tensor(bool) {
            ensureNumeric(T);
            return self.compareScalar(scalar, struct {
                fn f(a: T, b: T) bool {
                    return a < b;
                }
            }.f);
        }

        pub fn leScalar(self: Self, scalar: T) TensorError!Tensor(bool) {
            ensureNumeric(T);
            return self.compareScalar(scalar, struct {
                fn f(a: T, b: T) bool {
                    return a <= b;
                }
            }.f);
        }

        pub fn allclose(self: Self, other: Self, rtol: T, atol: T) TensorError!bool {
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

        pub fn isclose(self: Self, other: Self, rtol: T, atol: T) TensorError!Tensor(bool) {
            ensureFloat(T);
            const out_shape = try broadcastShape(self.allocator, self.shape, other.shape);
            defer self.allocator.free(out_shape);
            const out = try Tensor(bool).empty(self.allocator, out_shape);
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

        fn compare(self: Self, other: Self, comptime op: fn (T, T) bool) TensorError!Tensor(bool) {
            const out_shape = try broadcastShape(self.allocator, self.shape, other.shape);
            defer self.allocator.free(out_shape);
            const out = try Tensor(bool).empty(self.allocator, out_shape);

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

        fn compareScalar(self: Self, scalar: T, comptime op: fn (T, T) bool) TensorError!Tensor(bool) {
            const out = try Tensor(bool).empty(self.allocator, self.shape);
            for (self.data, out.data) |value, *slot| slot.* = op(value, scalar);
            return out;
        }

        pub fn whereMask(mask: Tensor(bool), a: Self, b: Self) TensorError!Self {
            const tmp_shape = try broadcastShape(a.allocator, a.shape, b.shape);
            defer a.allocator.free(tmp_shape);
            const out_shape = try broadcastShape(a.allocator, tmp_shape, mask.shape);
            defer a.allocator.free(out_shape);
            const out = try Self.empty(a.allocator, out_shape);

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
            if (comptime T != bool) @compileError("all requires Tensor(bool)");
            for (self.data) |v| if (!v) return false;
            return true;
        }

        pub fn any(self: Self) bool {
            if (comptime T != bool) @compileError("any requires Tensor(bool)");
            for (self.data) |v| if (v) return true;
            return false;
        }

        pub fn allAxis(self: Self, axis_opt: ?isize, keepdims: bool) TensorError!Self {
            if (comptime T != bool) @compileError("allAxis requires Tensor(bool)");
            return self.boolReduce(axis_opt, keepdims, true, struct {
                fn f(a: bool, b: bool) bool {
                    return a and b;
                }
            }.f);
        }

        pub fn anyAxis(self: Self, axis_opt: ?isize, keepdims: bool) TensorError!Self {
            if (comptime T != bool) @compileError("anyAxis requires Tensor(bool)");
            return self.boolReduce(axis_opt, keepdims, false, struct {
                fn f(a: bool, b: bool) bool {
                    return a or b;
                }
            }.f);
        }

        fn boolReduce(self: Self, axis_opt: ?isize, keepdims: bool, init_value: bool, comptime op: fn (bool, bool) bool) TensorError!Self {
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

        pub fn logicalNot(self: Self) TensorError!Self {
            if (comptime T != bool) @compileError("logicalNot requires Tensor(bool)");
            const out = try Self.empty(self.allocator, self.shape);
            for (self.data, out.data) |v, *slot| slot.* = !v;
            return out;
        }

        pub fn logicalAnd(self: Self, other: Self) TensorError!Self {
            if (comptime T != bool) @compileError("logicalAnd requires Tensor(bool)");
            return self.binaryTensor(other, struct {
                fn f(a: bool, b: bool) bool {
                    return a and b;
                }
            }.f);
        }

        pub fn logicalAndScalar(self: Self, scalar: bool) TensorError!Self {
            if (comptime T != bool) @compileError("logicalAndScalar requires Tensor(bool)");
            return self.binaryScalar(scalar, struct {
                fn f(a: bool, b: bool) bool {
                    return a and b;
                }
            }.f);
        }

        pub fn logicalOr(self: Self, other: Self) TensorError!Self {
            if (comptime T != bool) @compileError("logicalOr requires Tensor(bool)");
            return self.binaryTensor(other, struct {
                fn f(a: bool, b: bool) bool {
                    return a or b;
                }
            }.f);
        }

        pub fn logicalOrScalar(self: Self, scalar: bool) TensorError!Self {
            if (comptime T != bool) @compileError("logicalOrScalar requires Tensor(bool)");
            return self.binaryScalar(scalar, struct {
                fn f(a: bool, b: bool) bool {
                    return a or b;
                }
            }.f);
        }

        pub fn logicalXor(self: Self, other: Self) TensorError!Self {
            if (comptime T != bool) @compileError("logicalXor requires Tensor(bool)");
            return self.binaryTensor(other, struct {
                fn f(a: bool, b: bool) bool {
                    return a != b;
                }
            }.f);
        }

        pub fn logicalXorScalar(self: Self, scalar: bool) TensorError!Self {
            if (comptime T != bool) @compileError("logicalXorScalar requires Tensor(bool)");
            return self.binaryScalar(scalar, struct {
                fn f(a: bool, b: bool) bool {
                    return a != b;
                }
            }.f);
        }

        pub fn sum(self: Self, axis_opt: ?isize, keepdims: bool) TensorError!Self {
            ensureNumeric(T);
            return self.reduce(axis_opt, keepdims, zero(T), opAdd);
        }

        pub fn prod(self: Self, axis_opt: ?isize, keepdims: bool) TensorError!Self {
            ensureNumeric(T);
            return self.reduce(axis_opt, keepdims, one(T), opMul);
        }

        pub fn min(self: Self, axis_opt: ?isize, keepdims: bool) TensorError!Self {
            ensureNumeric(T);
            if (self.data.len == 0) return error.EmptyTensor;
            return self.reduceFirst(axis_opt, keepdims, struct {
                fn f(a: T, b: T) T {
                    return if (b < a) b else a;
                }
            }.f);
        }

        pub fn max(self: Self, axis_opt: ?isize, keepdims: bool) TensorError!Self {
            ensureNumeric(T);
            if (self.data.len == 0) return error.EmptyTensor;
            return self.reduceFirst(axis_opt, keepdims, struct {
                fn f(a: T, b: T) T {
                    return if (b > a) b else a;
                }
            }.f);
        }

        fn reducedShape(self: Self, axis: usize, keepdims: bool) TensorError![]usize {
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

        fn reduceFirst(self: Self, axis_opt: ?isize, keepdims: bool, comptime op: fn (T, T) T) TensorError!Self {
            if (self.data.len == 0) return error.EmptyTensor;
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
            if (self.shape[axis] == 0) return error.EmptyTensor;
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

        fn reduce(self: Self, axis_opt: ?isize, keepdims: bool, init_value: T, comptime op: fn (T, T) T) TensorError!Self {
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

        fn keepDimsAllOnes(allocator: std.mem.Allocator, rank: usize) TensorError![]usize {
            const dims = try allocator.alloc(usize, rank);
            @memset(dims, 1);
            return dims;
        }

        pub fn mean(self: Self, axis_opt: ?isize, keepdims: bool) TensorError!Self {
            ensureFloat(T);
            const out = try self.sum(axis_opt, keepdims);
            const divisor: T = if (axis_opt) |d| castValue(T, self.shape[try normalizeDim(d, self.shape.len)]) else castValue(T, self.data.len);
            for (out.data) |*v| v.* /= divisor;
            return out;
        }

        pub fn variance(self: Self, axis_opt: ?isize, keepdims: bool, correction: T) TensorError!Self {
            ensureFloat(T);
            if (axis_opt != null) {
                const axis = try normalizeDim(axis_opt.?, self.shape.len);
                const n = self.shape[axis];
                if (n == 0) return error.EmptyTensor;
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

            if (self.data.len == 0) return error.EmptyTensor;
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

        pub fn stddev(self: Self, axis_opt: ?isize, keepdims: bool, correction: T) TensorError!Self {
            const out = try self.variance(axis_opt, keepdims, correction);
            for (out.data) |*v| v.* = std.math.sqrt(v.*);
            return out;
        }

        pub fn nanToNum(self: Self, nan_value: T, posinf_value: T, neginf_value: T) TensorError!Self {
            ensureFloat(T);
            const out = try Self.empty(self.allocator, self.shape);
            for (self.data, out.data) |value, *slot| {
                slot.* = if (std.math.isNan(value))
                    nan_value
                else if (std.math.isPositiveInf(value))
                    posinf_value
                else if (std.math.isNegativeInf(value))
                    neginf_value
                else
                    value;
            }
            return out;
        }

        pub fn nan_to_num(self: Self, nan_value: T, posinf_value: T, neginf_value: T) TensorError!Self {
            return self.nanToNum(nan_value, posinf_value, neginf_value);
        }

        pub fn nansum(self: Self, axis_opt: ?isize, keepdims: bool) TensorError!Self {
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

        fn nanmeanWithCounts(self: Self, axis_opt: ?isize, keepdims: bool) TensorError!struct { values: Self, counts: Tensor(usize) } {
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
                var counts = try Tensor(usize).fromSlice(self.allocator, &.{count}, out_shape);
                errdefer counts.deinit();
                return .{ .values = values, .counts = counts };
            }

            const axis = try normalizeDim(axis_opt.?, self.shape.len);
            const out_shape = try self.reducedShape(axis, keepdims);
            defer self.allocator.free(out_shape);
            var values = try Self.zeros(self.allocator, out_shape);
            errdefer values.deinit();
            var counts = try Tensor(usize).zeros(self.allocator, out_shape);
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

        pub fn nanmean(self: Self, axis_opt: ?isize, keepdims: bool) TensorError!Self {
            ensureFloat(T);
            var result = try self.nanmeanWithCounts(axis_opt, keepdims);
            result.counts.deinit();
            return result.values;
        }

        pub fn nanvar(self: Self, axis_opt: ?isize, keepdims: bool, correction: T) TensorError!Self {
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

        pub fn nanstd(self: Self, axis_opt: ?isize, keepdims: bool, correction: T) TensorError!Self {
            const out = try self.nanvar(axis_opt, keepdims, correction);
            for (out.data) |*value| value.* = std.math.sqrt(value.*);
            return out;
        }

        fn nanExtreme(self: Self, axis_opt: ?isize, keepdims: bool, comptime better: fn (T, T) bool) TensorError!Self {
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

        pub fn nanmin(self: Self, axis_opt: ?isize, keepdims: bool) TensorError!Self {
            ensureFloat(T);
            return self.nanExtreme(axis_opt, keepdims, struct {
                fn f(a: T, b: T) bool {
                    return a < b;
                }
            }.f);
        }

        pub fn nanmax(self: Self, axis_opt: ?isize, keepdims: bool) TensorError!Self {
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

        pub fn quantile(self: Self, q: T, axis_opt: ?isize, keepdims: bool) TensorError!Self {
            ensureFloat(T);
            if (q < zero(T) or q > one(T)) return error.InvalidShape;
            if (self.data.len == 0) return error.EmptyTensor;
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
            if (self.shape[axis] == 0) return error.EmptyTensor;
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

        pub fn percentile(self: Self, p: T, axis_opt: ?isize, keepdims: bool) TensorError!Self {
            ensureFloat(T);
            return self.quantile(p / castValue(T, 100), axis_opt, keepdims);
        }

        pub fn median(self: Self, axis_opt: ?isize, keepdims: bool) TensorError!Self {
            ensureFloat(T);
            return self.quantile(castValue(T, 0.5), axis_opt, keepdims);
        }

        pub fn nanquantile(self: Self, q: T, axis_opt: ?isize, keepdims: bool) TensorError!Self {
            ensureFloat(T);
            if (q < zero(T) or q > one(T)) return error.InvalidShape;
            if (self.data.len == 0) return error.EmptyTensor;
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
            if (self.shape[axis] == 0) return error.EmptyTensor;
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

        pub fn nanpercentile(self: Self, p: T, axis_opt: ?isize, keepdims: bool) TensorError!Self {
            ensureFloat(T);
            return self.nanquantile(p / castValue(T, 100), axis_opt, keepdims);
        }

        pub fn nanmedian(self: Self, axis_opt: ?isize, keepdims: bool) TensorError!Self {
            ensureFloat(T);
            return self.nanquantile(castValue(T, 0.5), axis_opt, keepdims);
        }

        fn observationValue(self: Self, variable: usize, observation: usize, rowvar: bool) T {
            if (rowvar) return self.data[variable * self.shape[1] + observation];
            return self.data[observation * self.shape[1] + variable];
        }

        pub fn cov(self: Self, rowvar: bool, correction: T) TensorError!Self {
            ensureFloat(T);
            if (self.data.len == 0) return error.EmptyTensor;
            if (self.shape.len == 1) {
                const observations = self.data.len;
                const denom = castValue(T, observations) - correction;
                if (!(denom > zero(T))) return error.InvalidShape;
                return self.variance(null, false, correction);
            }
            if (self.shape.len != 2) return error.NonMatrixTensor;
            const variables = if (rowvar) self.shape[0] else self.shape[1];
            const observations = if (rowvar) self.shape[1] else self.shape[0];
            if (variables == 0 or observations == 0) return error.EmptyTensor;
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

        pub fn corrcoef(self: Self, rowvar: bool) TensorError!Self {
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

        pub fn norm(self: Self, p: T, axis_opt: ?isize, keepdims: bool) TensorError!Self {
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

        pub fn cumsum(self: Self) TensorError!Self {
            ensureNumeric(T);
            const out = try Self.empty(self.allocator, self.shape);
            var acc = zero(T);
            for (self.data, out.data) |v, *slot| {
                acc = addValue(T, acc, v);
                slot.* = acc;
            }
            return out;
        }

        pub fn cumprod(self: Self) TensorError!Self {
            ensureNumeric(T);
            const out = try Self.empty(self.allocator, self.shape);
            var acc = one(T);
            for (self.data, out.data) |v, *slot| {
                acc = mulValue(T, acc, v);
                slot.* = acc;
            }
            return out;
        }

        pub fn cumsumAxis(self: Self, axis_index: isize) TensorError!Self {
            ensureNumeric(T);
            return self.cumulativeAxis(axis_index, zero(T), opAdd);
        }

        pub fn cumprodAxis(self: Self, axis_index: isize) TensorError!Self {
            ensureNumeric(T);
            return self.cumulativeAxis(axis_index, one(T), opMul);
        }

        fn cumulativeAxis(self: Self, axis_index: isize, init_value: T, comptime op: fn (T, T) T) TensorError!Self {
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

        pub fn diff(self: Self, axis_index: isize, n: usize) TensorError!Self {
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

        fn diffOnce(self: Self, axis_index: isize) TensorError!Self {
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

        pub fn argmax(self: Self) TensorError!usize {
            ensureNumeric(T);
            if (self.data.len == 0) return error.EmptyTensor;
            var best: usize = 0;
            for (self.data[1..], 1..) |v, i| {
                if (v > self.data[best]) best = i;
            }
            return best;
        }

        pub fn argmin(self: Self) TensorError!usize {
            ensureNumeric(T);
            if (self.data.len == 0) return error.EmptyTensor;
            var best: usize = 0;
            for (self.data[1..], 1..) |v, i| {
                if (v < self.data[best]) best = i;
            }
            return best;
        }

        pub fn argmaxAxis(self: Self, axis_opt: ?isize, keepdims: bool) TensorError!Tensor(usize) {
            ensureNumeric(T);
            return self.argReduce(axis_opt, keepdims, struct {
                fn better(a: T, b: T) bool {
                    return a > b;
                }
            }.better);
        }

        pub fn argminAxis(self: Self, axis_opt: ?isize, keepdims: bool) TensorError!Tensor(usize) {
            ensureNumeric(T);
            return self.argReduce(axis_opt, keepdims, struct {
                fn better(a: T, b: T) bool {
                    return a < b;
                }
            }.better);
        }

        fn argReduce(self: Self, axis_opt: ?isize, keepdims: bool, comptime better: fn (T, T) bool) TensorError!Tensor(usize) {
            if (self.data.len == 0) return error.EmptyTensor;
            if (axis_opt == null) {
                var best: usize = 0;
                for (self.data[1..], 1..) |v, i| {
                    if (better(v, self.data[best])) best = i;
                }
                if (keepdims) {
                    const out_shape = try keepDimsAllOnes(self.allocator, self.shape.len);
                    defer self.allocator.free(out_shape);
                    return Tensor(usize).fromSlice(self.allocator, &.{best}, out_shape);
                }
                return Tensor(usize).fromSlice(self.allocator, &.{best}, &.{});
            }

            const axis = try normalizeDim(axis_opt.?, self.shape.len);
            if (self.shape[axis] == 0) return error.EmptyTensor;
            const out_shape = try self.reducedShape(axis, keepdims);
            defer self.allocator.free(out_shape);
            var out = try Tensor(usize).empty(self.allocator, out_shape);
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

        pub const TopK = struct {
            values: Self,
            indices: Tensor(usize),

            pub fn deinit(self: *@This()) void {
                self.values.deinit();
                self.indices.deinit();
                self.* = undefined;
            }
        };

        pub fn topk(self: Self, k: usize, axis_opt: ?isize, largest: bool, sorted: bool) TensorError!TopK {
            ensureNumeric(T);
            if (self.data.len == 0 and k > 0) return error.EmptyTensor;
            if (axis_opt == null) return self.topkFlat(k, largest, sorted);
            return self.topkAxis(k, try normalizeDim(axis_opt.?, self.shape.len), largest, sorted);
        }

        fn topkFlat(self: Self, k: usize, largest: bool, sorted: bool) TensorError!TopK {
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
            var indices = try Tensor(usize).empty(self.allocator, &.{k});
            errdefer indices.deinit();
            for (0..k) |i| {
                const idx = order[i];
                values.data[i] = self.data[idx];
                indices.data[i] = idx;
            }
            return .{ .values = values, .indices = indices };
        }

        fn topkAxis(self: Self, k: usize, axis: usize, largest: bool, sorted: bool) TensorError!TopK {
            const axis_len = self.shape[axis];
            if (k > axis_len) return error.InvalidShape;
            var out_shape = try self.allocator.dupe(usize, self.shape);
            defer self.allocator.free(out_shape);
            out_shape[axis] = k;
            var values = try Self.empty(self.allocator, out_shape);
            errdefer values.deinit();
            var indices = try Tensor(usize).empty(self.allocator, out_shape);
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
                tensor: Self,
                axis: usize,
                base_multi: []const usize,
                largest: bool,

                fn valueAt(ctx: @This(), axis_i: usize) T {
                    var offset: usize = 0;
                    for (ctx.tensor.shape, ctx.tensor.strides, 0..) |_, stride_value, dim_i| {
                        const coord = if (dim_i == ctx.axis) axis_i else ctx.base_multi[dim_i];
                        offset += coord * stride_value;
                    }
                    return ctx.tensor.data[offset];
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
                std.sort.insertion(usize, order, Ctx{ .tensor = self, .axis = axis, .base_multi = base_multi, .largest = largest }, Ctx.lessThan);
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

        pub fn matmul(self: Self, other: Self) TensorError!Self {
            ensureNumeric(T);
            if (self.shape.len != 2 or other.shape.len != 2) return error.NonMatrixTensor;
            const m = self.shape[0];
            const k = self.shape[1];
            if (other.shape[0] != k) return error.ShapeMismatch;
            const n = other.shape[1];
            var out = try Self.zeros(self.allocator, &.{ m, n });
            for (0..m) |i| {
                for (0..n) |j| {
                    var acc = zero(T);
                    for (0..k) |p| {
                        acc = addValue(T, acc, mulValue(T, self.data[i * k + p], other.data[p * n + j]));
                    }
                    out.data[i * n + j] = acc;
                }
            }
            return out;
        }

        pub fn mm(self: Self, other: Self) TensorError!Self {
            return self.matmul(other);
        }

        pub fn bmm(self: Self, other: Self) TensorError!Self {
            ensureNumeric(T);
            if (self.shape.len != 3 or other.shape.len != 3) return error.NonMatrixTensor;
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

        pub fn dot(self: Self, other: Self) TensorError!Self {
            ensureNumeric(T);
            if (self.shape.len != 1 or other.shape.len != 1) return error.NonVectorTensor;
            if (self.shape[0] != other.shape[0]) return error.ShapeMismatch;
            var acc = zero(T);
            for (self.data, other.data) |a, b| acc = addValue(T, acc, mulValue(T, a, b));
            return Self.fromSlice(self.allocator, &.{acc}, &.{});
        }

        pub fn outer(self: Self, other: Self) TensorError!Self {
            ensureNumeric(T);
            if (self.shape.len != 1 or other.shape.len != 1) return error.NonVectorTensor;
            const out = try Self.empty(self.allocator, &.{ self.shape[0], other.shape[0] });
            for (0..self.shape[0]) |i| {
                for (0..other.shape[0]) |j| {
                    out.data[i * other.shape[0] + j] = mulValue(T, self.data[i], other.data[j]);
                }
            }
            return out;
        }

        pub fn diagonal(self: Self, offset: isize) TensorError!Self {
            if (self.shape.len != 2) return error.NonMatrixTensor;
            const rows = self.shape[0];
            const cols = self.shape[1];
            const start_row: usize = if (offset < 0) blk: {
                const magnitude: usize = @intCast(-offset);
                if (magnitude >= rows) return Self.empty(self.allocator, &.{0});
                break :blk magnitude;
            } else 0;
            const start_col: usize = if (offset > 0) blk: {
                const magnitude: usize = @intCast(offset);
                if (magnitude >= cols) return Self.empty(self.allocator, &.{0});
                break :blk magnitude;
            } else 0;
            const count = @min(rows - start_row, cols - start_col);
            const out = try Self.empty(self.allocator, &.{count});
            for (out.data, 0..) |*slot, i| {
                slot.* = self.data[(start_row + i) * cols + start_col + i];
            }
            return out;
        }

        pub fn diag(self: Self, offset: isize) TensorError!Self {
            if (self.shape.len == 1) return self.diagflat(offset);
            if (self.shape.len == 2) return self.diagonal(offset);
            return error.InvalidShape;
        }

        pub fn diagflat(self: Self, offset: isize) TensorError!Self {
            var flat = try self.flatten();
            defer flat.deinit();
            const n = flat.data.len;
            const magnitude: usize = if (offset < 0) @intCast(-offset) else @intCast(offset);
            const matrix_size = n + magnitude;
            const out = try Self.zeros(self.allocator, &.{ matrix_size, matrix_size });
            const cols = matrix_size;
            for (flat.data, 0..) |value, i| {
                const row = if (offset < 0) i + magnitude else i;
                const col = if (offset > 0) i + magnitude else i;
                out.data[row * cols + col] = value;
            }
            return out;
        }

        pub fn trace(self: Self) TensorError!T {
            ensureNumeric(T);
            if (self.shape.len != 2) return error.NonMatrixTensor;
            const count = @min(self.shape[0], self.shape[1]);
            var total = zero(T);
            for (0..count) |i| total = addValue(T, total, self.data[i * self.shape[1] + i]);
            return total;
        }

        pub fn triu(self: Self, diagonal_offset: isize) TensorError!Self {
            ensureNumeric(T);
            if (self.shape.len != 2) return error.NonMatrixTensor;
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

        pub fn tril(self: Self, diagonal_offset: isize) TensorError!Self {
            ensureNumeric(T);
            if (self.shape.len != 2) return error.NonMatrixTensor;
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

        pub fn softmax(self: Self, axis_index: isize) TensorError!Self {
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
            indices: Tensor(usize),

            pub fn deinit(self: *@This()) void {
                self.values.deinit();
                self.indices.deinit();
                self.* = undefined;
            }
        };

        fn sortOrderLess(descending: bool, a: T, b: T) bool {
            return if (descending) lessValue(T, b, a) else lessValue(T, a, b);
        }

        pub fn sort(self: Self, axis_opt: ?isize) TensorError!Self {
            return self.sortBy(axis_opt, false);
        }

        pub fn sortDescending(self: Self, axis_opt: ?isize) TensorError!Self {
            return self.sortBy(axis_opt, true);
        }

        pub fn sortBy(self: Self, axis_opt: ?isize, descending: bool) TensorError!Self {
            var result = try self.sortWithIndices(axis_opt, descending);
            result.indices.deinit();
            return result.values;
        }

        pub fn argsort(self: Self) TensorError!Tensor(usize) {
            return self.argsortAxis(null, false);
        }

        pub fn argsortDescending(self: Self) TensorError!Tensor(usize) {
            return self.argsortAxis(null, true);
        }

        pub fn argsortAxis(self: Self, axis_opt: ?isize, descending: bool) TensorError!Tensor(usize) {
            var result = try self.sortWithIndices(axis_opt, descending);
            result.values.deinit();
            return result.indices;
        }

        pub fn sortWithIndices(self: Self, axis_opt: ?isize, descending: bool) TensorError!SortResult {
            ensureOrderable(T);
            if (axis_opt == null) {
                var values = try self.flatten();
                errdefer values.deinit();
                var indices = try Tensor(usize).empty(self.allocator, &.{self.data.len});
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
            var indices = try Tensor(usize).empty(self.allocator, self.shape);
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
                tensor: Self,
                axis: usize,
                base_multi: []const usize,
                descending: bool,

                fn valueAt(ctx: @This(), axis_i: usize) T {
                    var offset: usize = 0;
                    for (ctx.tensor.shape, ctx.tensor.strides, 0..) |_, stride_value, dim_i| {
                        const coord = if (dim_i == ctx.axis) axis_i else ctx.base_multi[dim_i];
                        offset += coord * stride_value;
                    }
                    return ctx.tensor.data[offset];
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
                std.sort.insertion(usize, order, Ctx{ .tensor = self, .axis = axis, .base_multi = base_multi, .descending = descending }, Ctx.lessThan);

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

        fn partitionLen(self: Self, axis_opt: ?isize) TensorError!usize {
            if (axis_opt) |axis_index| return self.shape[try normalizeDim(axis_index, self.shape.len)];
            return self.data.len;
        }

        pub fn partition(self: Self, kth: usize, axis_opt: ?isize, descending: bool) TensorError!Self {
            ensureOrderable(T);
            const len_axis = try self.partitionLen(axis_opt);
            if (kth >= len_axis) return error.InvalidShape;
            // Full sorting is a valid (stronger) partition: the kth item is in the
            // same position it would occupy in a sorted array, and all preceding
            // items compare before all following items. A future kernel can relax
            // this to O(n) selection while keeping the API stable.
            return self.sortBy(axis_opt, descending);
        }

        pub fn argpartition(self: Self, kth: usize, axis_opt: ?isize, descending: bool) TensorError!Tensor(usize) {
            ensureOrderable(T);
            const len_axis = try self.partitionLen(axis_opt);
            if (kth >= len_axis) return error.InvalidShape;
            return self.argsortAxis(axis_opt, descending);
        }

        pub const UniqueCounts = struct {
            values: Self,
            counts: Tensor(usize),

            pub fn deinit(self: *@This()) void {
                self.values.deinit();
                self.counts.deinit();
                self.* = undefined;
            }
        };

        pub fn unique(self: Self) TensorError!Self {
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

        pub fn uniqueWithCounts(self: Self) TensorError!UniqueCounts {
            if (comptime T != bool) ensureNumeric(T);
            if (self.data.len == 0) {
                var values = try Self.empty(self.allocator, &.{0});
                errdefer values.deinit();
                var counts = try Tensor(usize).empty(self.allocator, &.{0});
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
            var counts = try Tensor(usize).empty(self.allocator, &.{distinct});
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

        fn valueAsIndex(value: T) TensorError!usize {
            switch (@typeInfo(T)) {
                .int => |info| {
                    if (info.signedness == .signed and value < 0) return error.InvalidShape;
                    return @intCast(value);
                },
                .comptime_int => return @intCast(value),
                else => @compileError("bincount requires an integer array"),
            }
        }

        pub fn bincount(self: Self, minlength: usize) TensorError!Tensor(usize) {
            if (comptime @typeInfo(T) != .int) @compileError("bincount requires an integer array");
            var size_out = minlength;
            for (self.data) |value| {
                const idx = try valueAsIndex(value);
                if (idx + 1 > size_out) size_out = idx + 1;
            }
            var out = try Tensor(usize).zeros(self.allocator, &.{size_out});
            errdefer out.deinit();
            for (self.data) |value| out.data[try valueAsIndex(value)] += 1;
            return out;
        }

        pub fn bincountWeighted(self: Self, comptime W: type, weights: Tensor(W), minlength: usize) TensorError!Tensor(W) {
            if (comptime @typeInfo(T) != .int) @compileError("bincountWeighted requires an integer input array");
            if (comptime !isNumeric(W)) @compileError("bincountWeighted requires numeric weights");
            if (weights.data.len != self.data.len) return error.ShapeMismatch;
            var size_out = minlength;
            for (self.data) |value| {
                const idx = try valueAsIndex(value);
                if (idx + 1 > size_out) size_out = idx + 1;
            }
            var out = try Tensor(W).zeros(self.allocator, &.{size_out});
            errdefer out.deinit();
            for (self.data, weights.data) |value, weight| out.data[try valueAsIndex(value)] += weight;
            return out;
        }

        pub fn searchsorted(self: Self, values: Self, side: SearchSide) TensorError!Tensor(usize) {
            ensureNumeric(T);
            if (self.shape.len != 1) return error.NonVectorTensor;
            var out = try Tensor(usize).empty(self.allocator, values.shape);
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

        pub fn bucketize(self: Self, boundaries: Self, side: SearchSide) TensorError!Tensor(usize) {
            return boundaries.searchsorted(self, side);
        }

        pub fn digitize(self: Self, bins: Self, right: bool) TensorError!Tensor(usize) {
            return bins.searchsorted(self, if (right) .left else .right);
        }

        pub fn isin(self: Self, test_elements: Self, invert: bool) TensorError!Tensor(bool) {
            if (comptime T != bool) ensureNumeric(T);
            var out = try Tensor(bool).empty(self.allocator, self.shape);
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

        pub fn clipArray(self: Self, min_values: Self, max_values: Self) TensorError!Self {
            ensureNumeric(T);
            var lower = try self.maximum(min_values);
            defer lower.deinit();
            return lower.minimum(max_values);
        }

        pub fn concatenate(allocator: std.mem.Allocator, tensors: []const Self, axis_index: isize) TensorError!Self {
            if (tensors.len == 0) return error.EmptyTensor;
            const rank = tensors[0].shape.len;
            const axis = try normalizeDim(axis_index, rank);
            var out_shape = try allocator.dupe(usize, tensors[0].shape);
            defer allocator.free(out_shape);
            out_shape[axis] = 0;
            for (tensors) |t| {
                if (t.shape.len != rank) return error.ShapeMismatch;
                for (t.shape, 0..) |d, i| {
                    if (i == axis) continue;
                    if (d != tensors[0].shape[i]) return error.ShapeMismatch;
                }
                out_shape[axis] += t.shape[axis];
            }
            const out = try Self.empty(allocator, out_shape);
            if (out.data.len == 0) return out;
            const out_multi = try allocator.alloc(usize, out_shape.len);
            defer allocator.free(out_multi);
            var in_multi = try allocator.alloc(usize, rank);
            defer allocator.free(in_multi);
            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, out_shape, out_multi);
                var base: usize = 0;
                var selected: usize = 0;
                while (selected < tensors.len) : (selected += 1) {
                    const next = base + tensors[selected].shape[axis];
                    if (out_multi[axis] < next) break;
                    base = next;
                }
                @memcpy(in_multi, out_multi);
                in_multi[axis] = out_multi[axis] - base;
                slot.* = tensors[selected].data[ravelIndex(in_multi, tensors[selected].strides)];
            }
            return out;
        }

        pub fn cat(allocator: std.mem.Allocator, tensors: []const Self, axis_index: isize) TensorError!Self {
            return Self.concatenate(allocator, tensors, axis_index);
        }

        pub fn stack(allocator: std.mem.Allocator, tensors: []const Self, axis_index: isize) TensorError!Self {
            if (tensors.len == 0) return error.EmptyTensor;
            const rank = tensors[0].shape.len + 1;
            const axis = if (axis_index < 0) blk: {
                const signed_rank: isize = @intCast(rank);
                const normalized = signed_rank + axis_index;
                if (normalized < 0 or normalized >= signed_rank) return error.InvalidAxis;
                break :blk @as(usize, @intCast(normalized));
            } else try canonicalAxis(@intCast(axis_index), rank);
            const out_shape = try allocator.alloc(usize, rank);
            defer allocator.free(out_shape);
            for (tensors[1..]) |t| {
                if (!std.mem.eql(usize, t.shape, tensors[0].shape)) return error.ShapeMismatch;
            }
            for (out_shape, 0..) |*slot, i| {
                slot.* = if (i < axis) tensors[0].shape[i] else if (i == axis) tensors.len else tensors[0].shape[i - 1];
            }
            const out = try Self.empty(allocator, out_shape);
            if (out.data.len == 0) return out;
            const out_multi = try allocator.alloc(usize, out_shape.len);
            defer allocator.free(out_multi);
            var in_multi = try allocator.alloc(usize, tensors[0].shape.len);
            defer allocator.free(in_multi);
            for (out.data, 0..) |*slot, flat| {
                unravelIndexInto(flat, out_shape, out_multi);
                const tensor_index = out_multi[axis];
                for (out_multi[0..axis], 0..) |coord, i| in_multi[i] = coord;
                for (out_multi[axis + 1 ..], axis..) |coord, i| in_multi[i] = coord;
                slot.* = tensors[tensor_index].data[ravelIndex(in_multi, tensors[tensor_index].strides)];
            }
            return out;
        }

        pub fn histogram(self: Self, bins: usize, range: ?struct { min: T, max: T }) TensorError!struct { counts: Tensor(usize), edges: Self } {
            ensureFloat(T);
            if (bins == 0) return error.InvalidShape;
            if (self.data.len == 0) return error.EmptyTensor;
            var min_v = range orelse .{ .min = self.data[0], .max = self.data[0] };
            if (range == null) {
                for (self.data[1..]) |v| {
                    if (v < min_v.min) min_v.min = v;
                    if (v > min_v.max) min_v.max = v;
                }
            }
            var counts = try Tensor(usize).zeros(self.allocator, &.{bins});
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

        pub fn toBytes(self: Self, allocator: std.mem.Allocator) TensorError![]u8 {
            return allocator.dupe(u8, std.mem.sliceAsBytes(self.data));
        }

        pub fn fromBytes(allocator: std.mem.Allocator, bytes: []const u8, dims: []const usize) TensorError!Self {
            const n = try numelFrom(dims);
            if (bytes.len != n * @sizeOf(T)) return error.InvalidShape;
            const out = try Self.empty(allocator, dims);
            @memcpy(std.mem.sliceAsBytes(out.data), bytes);
            return out;
        }

        pub fn toArchive(self: Self, allocator: std.mem.Allocator) TensorError![]u8 {
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
            for (self.shape) |dim| {
                std.mem.writeInt(u64, out[offset..][0..8], @intCast(dim), .little);
                offset += 8;
            }
            @memcpy(out[offset..][0..data_bytes.len], data_bytes);
            return out;
        }

        pub fn fromArchive(allocator: std.mem.Allocator, archive: []const u8) TensorError!Self {
            const min_len = Archive.magic.len + 1 + 1 + 2 + 8;
            if (archive.len < min_len) return error.InvalidShape;
            if (!std.mem.eql(u8, archive[0..Archive.magic.len], Archive.magic[0..])) return error.InvalidShape;
            var offset: usize = Archive.magic.len;
            if (archive[offset] != Archive.version) return error.InvalidShape;
            offset += 1;
            const archived_dtype = DType.fromTag(archive[offset]) orelse return error.InvalidShape;
            if (archived_dtype != DType.of(T)) return error.TypeUnsupported;
            offset += 1;
            const rank = std.mem.readInt(u16, archive[offset..][0..2], .little);
            offset += 2;
            const element_count: usize = @intCast(std.mem.readInt(u64, archive[offset..][0..8], .little));
            offset += 8;
            if (archive.len < min_len + @as(usize, rank) * 8) return error.InvalidShape;
            const dims = try allocator.alloc(usize, rank);
            defer allocator.free(dims);
            for (dims) |*dim| {
                dim.* = @intCast(std.mem.readInt(u64, archive[offset..][0..8], .little));
                offset += 8;
            }
            const n = try numelFrom(dims);
            if (n != element_count) return error.InvalidShape;
            const data_len = n * @sizeOf(T);
            if (archive.len != offset + data_len) return error.InvalidShape;
            return Self.fromBytes(allocator, archive[offset..], dims);
        }

        pub fn print(self: Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
            try writer.print("Array({s}, shape=", .{@typeName(T)});
            try printShape(writer, self.shape);
            try writer.print(", data=", .{});
            try printFlatData(T, writer, self.data);
            try writer.print(")", .{});
        }

        pub fn toOwnedString(self: Self, allocator: std.mem.Allocator) TensorError![]u8 {
            var aw: std.Io.Writer.Allocating = .init(allocator);
            errdefer aw.deinit();
            self.print(&aw.writer) catch return error.OutOfMemory;
            return aw.toOwnedSlice();
        }
    };
}

pub const Array = Tensor;
pub const NDArray = Tensor;

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

pub fn tensor(comptime T: type, allocator: std.mem.Allocator, values: []const T, dims: []const usize) TensorError!Tensor(T) {
    return Tensor(T).fromSlice(allocator, values, dims);
}

pub fn array(comptime T: type, allocator: std.mem.Allocator, values: []const T, dims: []const usize) TensorError!Array(T) {
    return Array(T).fromSlice(allocator, values, dims);
}

pub fn ndarray(comptime T: type, allocator: std.mem.Allocator, values: []const T, dims: []const usize) TensorError!NDArray(T) {
    return NDArray(T).fromSlice(allocator, values, dims);
}

pub fn zeros(comptime T: type, allocator: std.mem.Allocator, dims: []const usize) TensorError!Tensor(T) {
    return Tensor(T).zeros(allocator, dims);
}

pub fn ones(comptime T: type, allocator: std.mem.Allocator, dims: []const usize) TensorError!Tensor(T) {
    return Tensor(T).ones(allocator, dims);
}

pub fn full(comptime T: type, allocator: std.mem.Allocator, dims: []const usize, value: T) TensorError!Tensor(T) {
    return Tensor(T).full(allocator, dims, value);
}

pub fn empty(comptime T: type, allocator: std.mem.Allocator, dims: []const usize) TensorError!Tensor(T) {
    return Tensor(T).empty(allocator, dims);
}

pub fn arrayScalar(comptime T: type, allocator: std.mem.Allocator, value: T) TensorError!Tensor(T) {
    return Tensor(T).fromScalar(allocator, value);
}

pub fn emptyLike(comptime T: type, input: Tensor(T)) TensorError!Tensor(T) {
    return input.emptyLike();
}

pub fn zerosLike(comptime T: type, input: Tensor(T)) TensorError!Tensor(T) {
    return input.zerosLike();
}

pub fn onesLike(comptime T: type, input: Tensor(T)) TensorError!Tensor(T) {
    return input.onesLike();
}

pub fn fullLike(comptime T: type, input: Tensor(T), value: T) TensorError!Tensor(T) {
    return input.fullLike(value);
}

pub fn arange(comptime T: type, allocator: std.mem.Allocator, start: T, stop: T, step: T) TensorError!Tensor(T) {
    return Tensor(T).arange(allocator, start, stop, step);
}

pub fn linspace(comptime T: type, allocator: std.mem.Allocator, start: T, stop: T, count: usize) TensorError!Tensor(T) {
    return Tensor(T).linspace(allocator, start, stop, count);
}

pub fn rand(comptime T: type, allocator: std.mem.Allocator, dims: []const usize, seed: u64) TensorError!Tensor(T) {
    return Tensor(T).rand(allocator, dims, seed);
}

pub fn randn(comptime T: type, allocator: std.mem.Allocator, dims: []const usize, seed: u64) TensorError!Tensor(T) {
    return Tensor(T).randn(allocator, dims, seed);
}

pub fn uniform(comptime T: type, allocator: std.mem.Allocator, dims: []const usize, low: T, high: T, seed: u64) TensorError!Tensor(T) {
    return Tensor(T).uniform(allocator, dims, low, high, seed);
}

pub fn normal(comptime T: type, allocator: std.mem.Allocator, dims: []const usize, mean_value: T, stddev_value: T, seed: u64) TensorError!Tensor(T) {
    return Tensor(T).normal(allocator, dims, mean_value, stddev_value, seed);
}

pub fn randint(comptime T: type, allocator: std.mem.Allocator, dims: []const usize, low: T, high: T, seed: u64) TensorError!Tensor(T) {
    return Tensor(T).randint(allocator, dims, low, high, seed);
}

pub fn bernoulli(allocator: std.mem.Allocator, dims: []const usize, p: f64, seed: u64) TensorError!Tensor(bool) {
    return Tensor(bool).bernoulli(allocator, dims, p, seed);
}

pub fn exponential(comptime T: type, allocator: std.mem.Allocator, dims: []const usize, rate: T, seed: u64) TensorError!Tensor(T) {
    return Tensor(T).exponential(allocator, dims, rate, seed);
}

pub fn gamma(comptime T: type, allocator: std.mem.Allocator, dims: []const usize, shape_param: T, scale: T, seed: u64) TensorError!Tensor(T) {
    return Tensor(T).gamma(allocator, dims, shape_param, scale, seed);
}

pub fn beta(comptime T: type, allocator: std.mem.Allocator, dims: []const usize, alpha: T, beta_param: T, seed: u64) TensorError!Tensor(T) {
    return Tensor(T).beta(allocator, dims, alpha, beta_param, seed);
}

pub fn poisson(allocator: std.mem.Allocator, dims: []const usize, lambda: f64, seed: u64) TensorError!Tensor(u64) {
    if (!(lambda >= 0)) return error.InvalidShape;
    var engine = alea.ScalarPrng.init(seed);
    const rng = alea.Rng.init(&engine);
    const out = try Tensor(u64).empty(allocator, dims);
    for (out.data) |*slot| slot.* = alea.distributions.poissonChecked(rng, lambda) catch return error.InvalidShape;
    return out;
}

pub fn lognormal(comptime T: type, allocator: std.mem.Allocator, dims: []const usize, mean_value: T, stddev_value: T, seed: u64) TensorError!Tensor(T) {
    return Tensor(T).lognormal(allocator, dims, mean_value, stddev_value, seed);
}

pub fn studentT(comptime T: type, allocator: std.mem.Allocator, dims: []const usize, dof: T, seed: u64) TensorError!Tensor(T) {
    return Tensor(T).studentT(allocator, dims, dof, seed);
}

pub fn cauchy(comptime T: type, allocator: std.mem.Allocator, dims: []const usize, median_value: T, scale: T, seed: u64) TensorError!Tensor(T) {
    return Tensor(T).cauchy(allocator, dims, median_value, scale, seed);
}

pub fn laplace(comptime T: type, allocator: std.mem.Allocator, dims: []const usize, location: T, scale: T, seed: u64) TensorError!Tensor(T) {
    return Tensor(T).laplace(allocator, dims, location, scale, seed);
}

pub fn weibull(comptime T: type, allocator: std.mem.Allocator, dims: []const usize, scale: T, shape_param: T, seed: u64) TensorError!Tensor(T) {
    return Tensor(T).weibull(allocator, dims, scale, shape_param, seed);
}

pub fn eye(comptime T: type, allocator: std.mem.Allocator, n: usize) TensorError!Tensor(T) {
    return Tensor(T).eye(allocator, n);
}

pub fn cat(comptime T: type, allocator: std.mem.Allocator, tensors: []const Tensor(T), dim: isize) TensorError!Tensor(T) {
    return Tensor(T).cat(allocator, tensors, dim);
}

pub fn stack(comptime T: type, allocator: std.mem.Allocator, tensors: []const Tensor(T), dim: isize) TensorError!Tensor(T) {
    return Tensor(T).stack(allocator, tensors, dim);
}

pub fn outer(comptime T: type, a: Tensor(T), b: Tensor(T)) TensorError!Tensor(T) {
    return a.outer(b);
}

pub fn where(comptime T: type, mask: Tensor(bool), a: Tensor(T), b: Tensor(T)) TensorError!Tensor(T) {
    return Tensor(T).whereMask(mask, a, b);
}

pub fn add(comptime T: type, a: Tensor(T), b: Tensor(T)) TensorError!Tensor(T) {
    return a.add(b);
}

pub fn sub(comptime T: type, a: Tensor(T), b: Tensor(T)) TensorError!Tensor(T) {
    return a.sub(b);
}

pub fn mul(comptime T: type, a: Tensor(T), b: Tensor(T)) TensorError!Tensor(T) {
    return a.mul(b);
}

pub fn div(comptime T: type, a: Tensor(T), b: Tensor(T)) TensorError!Tensor(T) {
    return a.div(b);
}

pub fn pow(comptime T: type, a: Tensor(T), b: Tensor(T)) TensorError!Tensor(T) {
    return a.pow(b);
}

pub fn floorDiv(comptime T: type, a: Tensor(T), b: Tensor(T)) TensorError!Tensor(T) {
    return a.floorDiv(b);
}

pub fn mod(comptime T: type, a: Tensor(T), b: Tensor(T)) TensorError!Tensor(T) {
    return a.mod(b);
}

pub fn remainder(comptime T: type, a: Tensor(T), b: Tensor(T)) TensorError!Tensor(T) {
    return a.remainder(b);
}

pub fn maximum(comptime T: type, a: Tensor(T), b: Tensor(T)) TensorError!Tensor(T) {
    return a.maximum(b);
}

pub fn minimum(comptime T: type, a: Tensor(T), b: Tensor(T)) TensorError!Tensor(T) {
    return a.minimum(b);
}

pub fn hypot(comptime T: type, a: Tensor(T), b: Tensor(T)) TensorError!Tensor(T) {
    return a.hypot(b);
}

pub fn atan2(comptime T: type, y: Tensor(T), x: Tensor(T)) TensorError!Tensor(T) {
    return y.atan2(x);
}

pub fn copysign(comptime T: type, magnitude: Tensor(T), sign_values: Tensor(T)) TensorError!Tensor(T) {
    return magnitude.copysign(sign_values);
}

pub fn heaviside(comptime T: type, input: Tensor(T), values_at_zero: Tensor(T)) TensorError!Tensor(T) {
    return input.heaviside(values_at_zero);
}

pub fn addScalar(comptime T: type, input: Tensor(T), scalar: T) TensorError!Tensor(T) {
    return input.addScalar(scalar);
}

pub fn subScalar(comptime T: type, input: Tensor(T), scalar: T) TensorError!Tensor(T) {
    return input.subScalar(scalar);
}

pub fn mulScalar(comptime T: type, input: Tensor(T), scalar: T) TensorError!Tensor(T) {
    return input.mulScalar(scalar);
}

pub fn divScalar(comptime T: type, input: Tensor(T), scalar: T) TensorError!Tensor(T) {
    return input.divScalar(scalar);
}

pub fn powScalar(comptime T: type, input: Tensor(T), scalar: T) TensorError!Tensor(T) {
    return input.powScalar(scalar);
}

pub fn floorDivScalar(comptime T: type, input: Tensor(T), scalar: T) TensorError!Tensor(T) {
    return input.floorDivScalar(scalar);
}

pub fn modScalar(comptime T: type, input: Tensor(T), scalar: T) TensorError!Tensor(T) {
    return input.modScalar(scalar);
}

pub fn remainderScalar(comptime T: type, input: Tensor(T), scalar: T) TensorError!Tensor(T) {
    return input.remainderScalar(scalar);
}

pub fn maximumScalar(comptime T: type, input: Tensor(T), scalar: T) TensorError!Tensor(T) {
    return input.maximumScalar(scalar);
}

pub fn minimumScalar(comptime T: type, input: Tensor(T), scalar: T) TensorError!Tensor(T) {
    return input.minimumScalar(scalar);
}

pub fn hypotScalar(comptime T: type, input: Tensor(T), scalar: T) TensorError!Tensor(T) {
    return input.hypotScalar(scalar);
}

pub fn atan2Scalar(comptime T: type, input: Tensor(T), scalar: T) TensorError!Tensor(T) {
    return input.atan2Scalar(scalar);
}

pub fn copysignScalar(comptime T: type, input: Tensor(T), scalar: T) TensorError!Tensor(T) {
    return input.copysignScalar(scalar);
}

pub fn heavisideScalar(comptime T: type, input: Tensor(T), value_at_zero: T) TensorError!Tensor(T) {
    return input.heavisideScalar(value_at_zero);
}

pub fn neg(comptime T: type, input: Tensor(T)) TensorError!Tensor(T) {
    return input.neg();
}

pub fn abs(comptime T: type, input: Tensor(T)) TensorError!Tensor(T) {
    return input.abs();
}

pub fn square(comptime T: type, input: Tensor(T)) TensorError!Tensor(T) {
    return input.square();
}

pub fn reciprocal(comptime T: type, input: Tensor(T)) TensorError!Tensor(T) {
    return input.reciprocal();
}

pub fn sign(comptime T: type, input: Tensor(T)) TensorError!Tensor(T) {
    return input.sign();
}

pub fn signbit(comptime T: type, input: Tensor(T)) TensorError!Tensor(bool) {
    return input.signbit();
}

pub fn exp(comptime T: type, input: Tensor(T)) TensorError!Tensor(T) {
    return input.exp();
}

pub fn expm1(comptime T: type, input: Tensor(T)) TensorError!Tensor(T) {
    return input.expm1();
}

pub fn log(comptime T: type, input: Tensor(T)) TensorError!Tensor(T) {
    return input.log();
}

pub fn log2(comptime T: type, input: Tensor(T)) TensorError!Tensor(T) {
    return input.log2();
}

pub fn log10(comptime T: type, input: Tensor(T)) TensorError!Tensor(T) {
    return input.log10();
}

pub fn log1p(comptime T: type, input: Tensor(T)) TensorError!Tensor(T) {
    return input.log1p();
}

pub fn sqrt(comptime T: type, input: Tensor(T)) TensorError!Tensor(T) {
    return input.sqrt();
}

pub fn floor(comptime T: type, input: Tensor(T)) TensorError!Tensor(T) {
    return input.floor();
}

pub fn ceil(comptime T: type, input: Tensor(T)) TensorError!Tensor(T) {
    return input.ceil();
}

pub fn round(comptime T: type, input: Tensor(T)) TensorError!Tensor(T) {
    return input.round();
}

pub fn trunc(comptime T: type, input: Tensor(T)) TensorError!Tensor(T) {
    return input.trunc();
}

pub fn deg2rad(comptime T: type, input: Tensor(T)) TensorError!Tensor(T) {
    return input.deg2rad();
}

pub fn rad2deg(comptime T: type, input: Tensor(T)) TensorError!Tensor(T) {
    return input.rad2deg();
}

pub fn sin(comptime T: type, input: Tensor(T)) TensorError!Tensor(T) {
    return input.sin();
}

pub fn cos(comptime T: type, input: Tensor(T)) TensorError!Tensor(T) {
    return input.cos();
}

pub fn tan(comptime T: type, input: Tensor(T)) TensorError!Tensor(T) {
    return input.tan();
}

pub fn asin(comptime T: type, input: Tensor(T)) TensorError!Tensor(T) {
    return input.asin();
}

pub fn acos(comptime T: type, input: Tensor(T)) TensorError!Tensor(T) {
    return input.acos();
}

pub fn atan(comptime T: type, input: Tensor(T)) TensorError!Tensor(T) {
    return input.atan();
}

pub fn sinh(comptime T: type, input: Tensor(T)) TensorError!Tensor(T) {
    return input.sinh();
}

pub fn cosh(comptime T: type, input: Tensor(T)) TensorError!Tensor(T) {
    return input.cosh();
}

pub fn tanh(comptime T: type, input: Tensor(T)) TensorError!Tensor(T) {
    return input.tanh();
}

pub fn relu(comptime T: type, input: Tensor(T)) TensorError!Tensor(T) {
    return input.relu();
}

pub fn sigmoid(comptime T: type, input: Tensor(T)) TensorError!Tensor(T) {
    return input.sigmoid();
}

pub fn clip(comptime T: type, input: Tensor(T), min_value: T, max_value: T) TensorError!Tensor(T) {
    return input.clip(min_value, max_value);
}

pub fn clamp(comptime T: type, input: Tensor(T), min_value: T, max_value: T) TensorError!Tensor(T) {
    return input.clamp(min_value, max_value);
}

pub fn eq(comptime T: type, a: Tensor(T), b: Tensor(T)) TensorError!Tensor(bool) {
    return a.eq(b);
}

pub fn equal(comptime T: type, a: Tensor(T), b: Tensor(T)) TensorError!Tensor(bool) {
    return a.equal(b);
}

pub fn ne(comptime T: type, a: Tensor(T), b: Tensor(T)) TensorError!Tensor(bool) {
    return a.ne(b);
}

pub fn notEqual(comptime T: type, a: Tensor(T), b: Tensor(T)) TensorError!Tensor(bool) {
    return a.notEqual(b);
}

pub fn gt(comptime T: type, a: Tensor(T), b: Tensor(T)) TensorError!Tensor(bool) {
    return a.gt(b);
}

pub fn greater(comptime T: type, a: Tensor(T), b: Tensor(T)) TensorError!Tensor(bool) {
    return a.greater(b);
}

pub fn ge(comptime T: type, a: Tensor(T), b: Tensor(T)) TensorError!Tensor(bool) {
    return a.ge(b);
}

pub fn greaterEqual(comptime T: type, a: Tensor(T), b: Tensor(T)) TensorError!Tensor(bool) {
    return a.greaterEqual(b);
}

pub fn lt(comptime T: type, a: Tensor(T), b: Tensor(T)) TensorError!Tensor(bool) {
    return a.lt(b);
}

pub fn less(comptime T: type, a: Tensor(T), b: Tensor(T)) TensorError!Tensor(bool) {
    return a.less(b);
}

pub fn le(comptime T: type, a: Tensor(T), b: Tensor(T)) TensorError!Tensor(bool) {
    return a.le(b);
}

pub fn lessEqual(comptime T: type, a: Tensor(T), b: Tensor(T)) TensorError!Tensor(bool) {
    return a.lessEqual(b);
}

pub fn eqScalar(comptime T: type, input: Tensor(T), scalar: T) TensorError!Tensor(bool) {
    return input.eqScalar(scalar);
}

pub fn neScalar(comptime T: type, input: Tensor(T), scalar: T) TensorError!Tensor(bool) {
    return input.neScalar(scalar);
}

pub fn gtScalar(comptime T: type, input: Tensor(T), scalar: T) TensorError!Tensor(bool) {
    return input.gtScalar(scalar);
}

pub fn geScalar(comptime T: type, input: Tensor(T), scalar: T) TensorError!Tensor(bool) {
    return input.geScalar(scalar);
}

pub fn ltScalar(comptime T: type, input: Tensor(T), scalar: T) TensorError!Tensor(bool) {
    return input.ltScalar(scalar);
}

pub fn leScalar(comptime T: type, input: Tensor(T), scalar: T) TensorError!Tensor(bool) {
    return input.leScalar(scalar);
}

pub fn allclose(comptime T: type, a: Tensor(T), b: Tensor(T), rtol: T, atol: T) TensorError!bool {
    return a.allclose(b, rtol, atol);
}

pub fn isclose(comptime T: type, a: Tensor(T), b: Tensor(T), rtol: T, atol: T) TensorError!Tensor(bool) {
    return a.isclose(b, rtol, atol);
}

pub fn logicalNot(input: Tensor(bool)) TensorError!Tensor(bool) {
    return input.logicalNot();
}

pub fn logicalAnd(a: Tensor(bool), b: Tensor(bool)) TensorError!Tensor(bool) {
    return a.logicalAnd(b);
}

pub fn logicalOr(a: Tensor(bool), b: Tensor(bool)) TensorError!Tensor(bool) {
    return a.logicalOr(b);
}

pub fn logicalXor(a: Tensor(bool), b: Tensor(bool)) TensorError!Tensor(bool) {
    return a.logicalXor(b);
}

pub fn logicalAndScalar(input: Tensor(bool), scalar: bool) TensorError!Tensor(bool) {
    return input.logicalAndScalar(scalar);
}

pub fn logicalOrScalar(input: Tensor(bool), scalar: bool) TensorError!Tensor(bool) {
    return input.logicalOrScalar(scalar);
}

pub fn logicalXorScalar(input: Tensor(bool), scalar: bool) TensorError!Tensor(bool) {
    return input.logicalXorScalar(scalar);
}

pub fn isNan(comptime T: type, input: Tensor(T)) TensorError!Tensor(bool) {
    return input.isNan();
}

pub fn isnan(comptime T: type, input: Tensor(T)) TensorError!Tensor(bool) {
    return input.isnan();
}

pub fn isInf(comptime T: type, input: Tensor(T)) TensorError!Tensor(bool) {
    return input.isInf();
}

pub fn isinf(comptime T: type, input: Tensor(T)) TensorError!Tensor(bool) {
    return input.isinf();
}

pub fn isFinite(comptime T: type, input: Tensor(T)) TensorError!Tensor(bool) {
    return input.isFinite();
}

pub fn isfinite(comptime T: type, input: Tensor(T)) TensorError!Tensor(bool) {
    return input.isfinite();
}

pub fn logsumexp(comptime T: type, input: Tensor(T), axis: isize, keepdims: bool) TensorError!Tensor(T) {
    return input.logsumexp(axis, keepdims);
}

pub fn logSoftmax(comptime T: type, input: Tensor(T), axis: isize) TensorError!Tensor(T) {
    return input.logSoftmax(axis);
}

pub fn log_softmax(comptime T: type, input: Tensor(T), axis: isize) TensorError!Tensor(T) {
    return input.log_softmax(axis);
}

pub fn sum(comptime T: type, input: Tensor(T), axis: ?isize, keepdims: bool) TensorError!Tensor(T) {
    return input.sum(axis, keepdims);
}

pub fn prod(comptime T: type, input: Tensor(T), axis: ?isize, keepdims: bool) TensorError!Tensor(T) {
    return input.prod(axis, keepdims);
}

pub fn min(comptime T: type, input: Tensor(T), axis: ?isize, keepdims: bool) TensorError!Tensor(T) {
    return input.min(axis, keepdims);
}

pub fn max(comptime T: type, input: Tensor(T), axis: ?isize, keepdims: bool) TensorError!Tensor(T) {
    return input.max(axis, keepdims);
}

pub fn mean(comptime T: type, input: Tensor(T), axis: ?isize, keepdims: bool) TensorError!Tensor(T) {
    return input.mean(axis, keepdims);
}

pub fn variance(comptime T: type, input: Tensor(T), axis: ?isize, keepdims: bool, correction: T) TensorError!Tensor(T) {
    return input.variance(axis, keepdims, correction);
}

pub fn stddev(comptime T: type, input: Tensor(T), axis: ?isize, keepdims: bool, correction: T) TensorError!Tensor(T) {
    return input.stddev(axis, keepdims, correction);
}

pub fn norm(comptime T: type, input: Tensor(T), p: T, axis: ?isize, keepdims: bool) TensorError!Tensor(T) {
    return input.norm(p, axis, keepdims);
}

pub fn cumsum(comptime T: type, input: Tensor(T)) TensorError!Tensor(T) {
    return input.cumsum();
}

pub fn cumprod(comptime T: type, input: Tensor(T)) TensorError!Tensor(T) {
    return input.cumprod();
}

pub fn argmax(comptime T: type, input: Tensor(T)) TensorError!usize {
    return input.argmax();
}

pub fn argmin(comptime T: type, input: Tensor(T)) TensorError!usize {
    return input.argmin();
}

pub fn argmaxAxis(comptime T: type, input: Tensor(T), axis: ?isize, keepdims: bool) TensorError!Tensor(usize) {
    return input.argmaxAxis(axis, keepdims);
}

pub fn argminAxis(comptime T: type, input: Tensor(T), axis: ?isize, keepdims: bool) TensorError!Tensor(usize) {
    return input.argminAxis(axis, keepdims);
}

pub fn allAxis(input: Tensor(bool), axis: ?isize, keepdims: bool) TensorError!Tensor(bool) {
    return input.allAxis(axis, keepdims);
}

pub fn anyAxis(input: Tensor(bool), axis: ?isize, keepdims: bool) TensorError!Tensor(bool) {
    return input.anyAxis(axis, keepdims);
}

pub fn median(comptime T: type, input: Tensor(T), axis: ?isize, keepdims: bool) TensorError!Tensor(T) {
    return input.median(axis, keepdims);
}

pub fn quantile(comptime T: type, input: Tensor(T), q: T, axis: ?isize, keepdims: bool) TensorError!Tensor(T) {
    return input.quantile(q, axis, keepdims);
}

pub fn percentile(comptime T: type, input: Tensor(T), p: T, axis: ?isize, keepdims: bool) TensorError!Tensor(T) {
    return input.percentile(p, axis, keepdims);
}

pub fn cov(comptime T: type, input: Tensor(T), rowvar: bool, correction: T) TensorError!Tensor(T) {
    return input.cov(rowvar, correction);
}

pub fn corrcoef(comptime T: type, input: Tensor(T), rowvar: bool) TensorError!Tensor(T) {
    return input.corrcoef(rowvar);
}

pub fn nanToNum(comptime T: type, input: Tensor(T), nan_value: T, posinf_value: T, neginf_value: T) TensorError!Tensor(T) {
    return input.nanToNum(nan_value, posinf_value, neginf_value);
}

pub fn nan_to_num(comptime T: type, input: Tensor(T), nan_value: T, posinf_value: T, neginf_value: T) TensorError!Tensor(T) {
    return input.nan_to_num(nan_value, posinf_value, neginf_value);
}

pub fn nansum(comptime T: type, input: Tensor(T), axis: ?isize, keepdims: bool) TensorError!Tensor(T) {
    return input.nansum(axis, keepdims);
}

pub fn nanmean(comptime T: type, input: Tensor(T), axis: ?isize, keepdims: bool) TensorError!Tensor(T) {
    return input.nanmean(axis, keepdims);
}

pub fn nanvar(comptime T: type, input: Tensor(T), axis: ?isize, keepdims: bool, correction: T) TensorError!Tensor(T) {
    return input.nanvar(axis, keepdims, correction);
}

pub fn nanstd(comptime T: type, input: Tensor(T), axis: ?isize, keepdims: bool, correction: T) TensorError!Tensor(T) {
    return input.nanstd(axis, keepdims, correction);
}

pub fn nanmin(comptime T: type, input: Tensor(T), axis: ?isize, keepdims: bool) TensorError!Tensor(T) {
    return input.nanmin(axis, keepdims);
}

pub fn nanmax(comptime T: type, input: Tensor(T), axis: ?isize, keepdims: bool) TensorError!Tensor(T) {
    return input.nanmax(axis, keepdims);
}

pub fn nanmedian(comptime T: type, input: Tensor(T), axis: ?isize, keepdims: bool) TensorError!Tensor(T) {
    return input.nanmedian(axis, keepdims);
}

pub fn nanquantile(comptime T: type, input: Tensor(T), q: T, axis: ?isize, keepdims: bool) TensorError!Tensor(T) {
    return input.nanquantile(q, axis, keepdims);
}

pub fn nanpercentile(comptime T: type, input: Tensor(T), p: T, axis: ?isize, keepdims: bool) TensorError!Tensor(T) {
    return input.nanpercentile(p, axis, keepdims);
}

pub fn sort(comptime T: type, input: Tensor(T), axis: ?isize) TensorError!Tensor(T) {
    return input.sort(axis);
}

pub fn sortBy(comptime T: type, input: Tensor(T), axis: ?isize, descending: bool) TensorError!Tensor(T) {
    return input.sortBy(axis, descending);
}

pub fn sortDescending(comptime T: type, input: Tensor(T), axis: ?isize) TensorError!Tensor(T) {
    return input.sortDescending(axis);
}

pub fn argsort(comptime T: type, input: Tensor(T)) TensorError!Tensor(usize) {
    return input.argsort();
}

pub fn argsortAxis(comptime T: type, input: Tensor(T), axis: ?isize, descending: bool) TensorError!Tensor(usize) {
    return input.argsortAxis(axis, descending);
}

pub fn argsortDescending(comptime T: type, input: Tensor(T)) TensorError!Tensor(usize) {
    return input.argsortDescending();
}

pub fn sortWithIndices(comptime T: type, input: Tensor(T), axis: ?isize, descending: bool) TensorError!Tensor(T).SortResult {
    return input.sortWithIndices(axis, descending);
}

pub fn partition(comptime T: type, input: Tensor(T), kth: usize, axis: ?isize, descending: bool) TensorError!Tensor(T) {
    return input.partition(kth, axis, descending);
}

pub fn argpartition(comptime T: type, input: Tensor(T), kth: usize, axis: ?isize, descending: bool) TensorError!Tensor(usize) {
    return input.argpartition(kth, axis, descending);
}

pub fn unique(comptime T: type, input: Tensor(T)) TensorError!Tensor(T) {
    return input.unique();
}

pub fn uniqueWithCounts(comptime T: type, input: Tensor(T)) TensorError!Tensor(T).UniqueCounts {
    return input.uniqueWithCounts();
}

pub fn bincount(comptime T: type, input: Tensor(T), minlength: usize) TensorError!Tensor(usize) {
    return input.bincount(minlength);
}

pub fn bincountWeighted(comptime T: type, comptime W: type, input: Tensor(T), weights: Tensor(W), minlength: usize) TensorError!Tensor(W) {
    return input.bincountWeighted(W, weights, minlength);
}

pub fn searchsorted(comptime T: type, sorted: Tensor(T), values: Tensor(T), side: SearchSide) TensorError!Tensor(usize) {
    return sorted.searchsorted(values, side);
}

pub fn bucketize(comptime T: type, input: Tensor(T), boundaries: Tensor(T), side: SearchSide) TensorError!Tensor(usize) {
    return input.bucketize(boundaries, side);
}

pub fn digitize(comptime T: type, input: Tensor(T), bins: Tensor(T), right: bool) TensorError!Tensor(usize) {
    return input.digitize(bins, right);
}

pub fn isin(comptime T: type, input: Tensor(T), test_elements: Tensor(T), invert: bool) TensorError!Tensor(bool) {
    return input.isin(test_elements, invert);
}

pub fn clipArray(comptime T: type, input: Tensor(T), min_values: Tensor(T), max_values: Tensor(T)) TensorError!Tensor(T) {
    return input.clipArray(min_values, max_values);
}

pub fn diag(comptime T: type, input: Tensor(T), offset: isize) TensorError!Tensor(T) {
    return input.diag(offset);
}

pub fn diagflat(comptime T: type, input: Tensor(T), offset: isize) TensorError!Tensor(T) {
    return input.diagflat(offset);
}

pub fn sliceAxis(comptime T: type, input: Tensor(T), axis: isize, slice: Slice) TensorError!Tensor(T) {
    return input.sliceAxis(axis, slice);
}

pub fn flip(comptime T: type, input: Tensor(T), axis: isize) TensorError!Tensor(T) {
    return input.flip(axis);
}

pub fn roll(comptime T: type, input: Tensor(T), shift: isize, axis: isize) TensorError!Tensor(T) {
    return input.roll(shift, axis);
}

pub fn padConstant(comptime T: type, input: Tensor(T), before: []const usize, after: []const usize, value: T) TensorError!Tensor(T) {
    return input.padConstant(before, after, value);
}

pub fn cumsumAxis(comptime T: type, input: Tensor(T), axis: isize) TensorError!Tensor(T) {
    return input.cumsumAxis(axis);
}

pub fn cumprodAxis(comptime T: type, input: Tensor(T), axis: isize) TensorError!Tensor(T) {
    return input.cumprodAxis(axis);
}

pub fn diff(comptime T: type, input: Tensor(T), axis: isize, n: usize) TensorError!Tensor(T) {
    return input.diff(axis, n);
}

pub fn toBytes(comptime T: type, input: Tensor(T), allocator: std.mem.Allocator) TensorError![]u8 {
    return input.toBytes(allocator);
}

pub fn fromBytes(comptime T: type, allocator: std.mem.Allocator, bytes: []const u8, dims: []const usize) TensorError!Tensor(T) {
    return Tensor(T).fromBytes(allocator, bytes, dims);
}

pub fn toArchive(comptime T: type, input: Tensor(T), allocator: std.mem.Allocator) TensorError![]u8 {
    return input.toArchive(allocator);
}

pub fn fromArchive(comptime T: type, allocator: std.mem.Allocator, archive: []const u8) TensorError!Tensor(T) {
    return Tensor(T).fromArchive(allocator, archive);
}

pub fn takeAlongAxis(comptime T: type, input: Tensor(T), indices: Tensor(usize), axis: isize) TensorError!Tensor(T) {
    return input.takeAlongAxis(indices, axis);
}

pub fn putAlongAxis(comptime T: type, input: Tensor(T), indices: Tensor(usize), src: Tensor(T), axis: isize) TensorError!Tensor(T) {
    return input.putAlongAxis(indices, src, axis);
}

pub fn maskedFill(comptime T: type, input: Tensor(T), mask: Tensor(bool), value: T) TensorError!Tensor(T) {
    return input.maskedFill(mask, value);
}

pub fn maskedScatter(comptime T: type, input: Tensor(T), mask: Tensor(bool), src: Tensor(T)) TensorError!Tensor(T) {
    return input.maskedScatter(mask, src);
}

pub fn maskedPut(comptime T: type, input: Tensor(T), mask: Tensor(bool), values: Tensor(T)) TensorError!Tensor(T) {
    return input.maskedPut(mask, values);
}

pub fn maskedPutScalar(comptime T: type, input: Tensor(T), mask: Tensor(bool), value: T) TensorError!Tensor(T) {
    return input.maskedPutScalar(mask, value);
}

pub fn putFlat(comptime T: type, input: Tensor(T), indices: Tensor(usize), values: Tensor(T)) TensorError!Tensor(T) {
    return input.putFlat(indices, values);
}

pub fn putFlatScalar(comptime T: type, input: Tensor(T), indices: Tensor(usize), value: T) TensorError!Tensor(T) {
    return input.putFlatScalar(indices, value);
}

pub fn indexPut(comptime T: type, input: Tensor(T), indices: Tensor(usize), values: Tensor(T)) TensorError!Tensor(T) {
    return input.indexPut(indices, values);
}

pub fn indexPutScalar(comptime T: type, input: Tensor(T), indices: Tensor(usize), value: T) TensorError!Tensor(T) {
    return input.indexPutScalar(indices, value);
}

pub fn countNonzero(comptime T: type, input: Tensor(T)) usize {
    return input.countNonzero();
}

pub fn flatNonzero(comptime T: type, input: Tensor(T)) TensorError!Tensor(usize) {
    return input.flatNonzero();
}

pub fn nonzero(comptime T: type, input: Tensor(T)) TensorError!Tensor(usize) {
    return input.nonzero();
}

pub fn argwhere(comptime T: type, input: Tensor(T)) TensorError!Tensor(usize) {
    return input.argwhere();
}

pub fn compress(comptime T: type, input: Tensor(T), condition: Tensor(bool), axis: ?isize) TensorError!Tensor(T) {
    return input.compress(condition, axis);
}

test "tensor creation, reshape and broadcasting" {
    const gpa = std.testing.allocator;
    var a = try tensor(f64, gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a.deinit();
    var b = try tensor(f64, gpa, &.{ 10, 20, 30 }, &.{3});
    defer b.deinit();
    var c = try a.add(b);
    defer c.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, c.shape);
    try std.testing.expectEqualSlices(f64, &.{ 11, 22, 33, 14, 25, 36 }, c.data);
    var flat = try c.flatten();
    defer flat.deinit();
    try std.testing.expectEqualSlices(usize, &.{6}, flat.shape);
}

test "array binary math wrappers and clamp aliases" {
    const gpa = std.testing.allocator;
    var a = try array(f64, gpa, &.{ 1, 2, 3, 4 }, &.{ 2, 2 });
    defer a.deinit();
    var b = try array(f64, gpa, &.{ 10, 20 }, &.{2});
    defer b.deinit();

    var added = try add(f64, a, b);
    defer added.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 11, 22, 13, 24 }, added.data);
    var subbed = try sub(f64, a, b);
    defer subbed.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -9, -18, -7, -16 }, subbed.data);
    var multiplied = try mul(f64, a, b);
    defer multiplied.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 10, 40, 30, 80 }, multiplied.data);
    var divided = try div(f64, multiplied, b);
    defer divided.deinit();
    try std.testing.expectEqualSlices(f64, a.data, divided.data);

    var exponent = try array(f64, gpa, &.{2}, &.{1});
    defer exponent.deinit();
    var powed = try pow(f64, a, exponent);
    defer powed.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 4, 9, 16 }, powed.data);

    var clamped = try clamp(f64, a, 1.5, 3.5);
    defer clamped.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1.5, 2, 3, 3.5 }, clamped.data);
    var clipped = try a.clamp(2, 3);
    defer clipped.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 2, 2, 3, 3 }, clipped.data);

    var maxed = try maximumScalar(f64, a, 2.5);
    defer maxed.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 2.5, 2.5, 3, 4 }, maxed.data);
    var mined = try minimumScalar(f64, a, 2.5);
    defer mined.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 2.5, 2.5 }, mined.data);

    var hyp_a = try array(f64, gpa, &.{ 3, 5 }, &.{2});
    defer hyp_a.deinit();
    var hyp_b = try array(f64, gpa, &.{ 4, 12 }, &.{2});
    defer hyp_b.deinit();
    var hyp = try hypot(f64, hyp_a, hyp_b);
    defer hyp.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 5, 13 }, hyp.data);
    var hyp_scalar = try hyp_a.hypotScalar(4);
    defer hyp_scalar.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 5), hyp_scalar.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 41)), hyp_scalar.data[1], 1e-12);

    var y = try array(f64, gpa, &.{ 0, 1 }, &.{2});
    defer y.deinit();
    var x = try array(f64, gpa, &.{ 1, 1 }, &.{2});
    defer x.deinit();
    var angles = try atan2(f64, y, x);
    defer angles.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0), angles.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.pi / 4.0, angles.data[1], 1e-12);

    var magnitudes = try array(f64, gpa, &.{ -1, 2, -3 }, &.{3});
    defer magnitudes.deinit();
    var signs_for_copy = try array(f64, gpa, &.{ 4, -5, -6 }, &.{3});
    defer signs_for_copy.deinit();
    var copied = try copysign(f64, magnitudes, signs_for_copy);
    defer copied.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, -2, -3 }, copied.data);
    var copied_scalar = try magnitudes.copysignScalar(-1);
    defer copied_scalar.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -1, -2, -3 }, copied_scalar.data);

    var heav = try array(f64, gpa, &.{ -2, 0, 3 }, &.{3});
    defer heav.deinit();
    var hzero = try array(f64, gpa, &.{0.5}, &.{1});
    defer hzero.deinit();
    var heav_out = try heaviside(f64, heav, hzero);
    defer heav_out.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 0.5, 1 }, heav_out.data);
    var heav_scalar = try heav.heavisideScalar(0.25);
    defer heav_scalar.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 0.25, 1 }, heav_scalar.data);

    var ints = try array(i32, gpa, &.{ -5, 5, 7 }, &.{3});
    defer ints.deinit();
    var divisors = try array(i32, gpa, &.{ 2, 2, 3 }, &.{3});
    defer divisors.deinit();
    var floor_div = try floorDiv(i32, ints, divisors);
    defer floor_div.deinit();
    try std.testing.expectEqualSlices(i32, &.{ -3, 2, 2 }, floor_div.data);
    var modulo = try mod(i32, ints, divisors);
    defer modulo.deinit();
    try std.testing.expectEqualSlices(i32, &.{ 1, 1, 1 }, modulo.data);
    var rem_scalar = try remainderScalar(i32, ints, 4);
    defer rem_scalar.deinit();
    try std.testing.expectEqualSlices(i32, &.{ 3, 1, 3 }, rem_scalar.data);
}

test "array comparison and logical wrappers" {
    const gpa = std.testing.allocator;
    var a = try array(f64, gpa, &.{ 1, 2, 3, 4 }, &.{ 2, 2 });
    defer a.deinit();
    var b = try array(f64, gpa, &.{ 1, 0 }, &.{2});
    defer b.deinit();

    var eq_out = try equal(f64, a, b);
    defer eq_out.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false }, eq_out.data);
    var ne_out = try notEqual(f64, a, b);
    defer ne_out.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, ne_out.data);
    var gt_out = try greater(f64, a, b);
    defer gt_out.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, gt_out.data);
    var ge_out = try ge(f64, a, b);
    defer ge_out.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, ge_out.data);
    var lt_out = try less(f64, a, b);
    defer lt_out.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false }, lt_out.data);
    var le_out = try le(f64, a, b);
    defer le_out.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false }, le_out.data);

    var eq_scalar_out = try eqScalar(f64, a, 2);
    defer eq_scalar_out.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false }, eq_scalar_out.data);
    var ge_scalar_out = try a.geScalar(3);
    defer ge_scalar_out.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, true }, ge_scalar_out.data);
    var lt_scalar_out = try ltScalar(f64, a, 3);
    defer lt_scalar_out.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false }, lt_scalar_out.data);

    try std.testing.expect(try allclose(f64, a, a, 1e-12, 1e-12));

    var m1 = try array(bool, gpa, &.{ true, false, true, false }, &.{ 2, 2 });
    defer m1.deinit();
    var m2 = try array(bool, gpa, &.{ true, true }, &.{2});
    defer m2.deinit();
    var not_out = try logicalNot(m1);
    defer not_out.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, not_out.data);
    var and_out = try logicalAnd(m1, m2);
    defer and_out.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, false }, and_out.data);
    var or_out = try logicalOr(m1, m2);
    defer or_out.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, or_out.data);
    var xor_out = try logicalXor(m1, m2);
    defer xor_out.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, xor_out.data);
    var xor_scalar_out = try logicalXorScalar(m1, true);
    defer xor_scalar_out.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, xor_scalar_out.data);

    var close_target = try array(f64, gpa, &.{ 1.0, 2.001, 2.9, 4.0 }, &.{ 2, 2 });
    defer close_target.deinit();
    var close_mask = try isclose(f64, a, close_target, 0.0, 0.01);
    defer close_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, close_mask.data);
    try std.testing.expect(!try allclose(f64, a, close_target, 0.0, 0.01));
}

test "tensor reductions and matmul" {
    const gpa = std.testing.allocator;
    var a = try tensor(f64, gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a.deinit();
    var s0 = try sum(f64, a, 0, false);
    defer s0.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 5, 7, 9 }, s0.data);
    var s1 = try sum(f64, a, 1, true);
    defer s1.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 1 }, s1.shape);
    try std.testing.expectEqualSlices(f64, &.{ 6, 15 }, s1.data);
    var p0 = try prod(f64, a, 0, false);
    defer p0.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 4, 10, 18 }, p0.data);
    var mn = try min(f64, a, null, false);
    defer mn.deinit();
    try std.testing.expectEqualSlices(f64, &.{1}, mn.data);
    var mx = try max(f64, a, 1, false);
    defer mx.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 3, 6 }, mx.data);
    var cs = try cumsum(f64, a);
    defer cs.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 6, 10, 15, 21 }, cs.data);
    var cp = try cumprod(f64, a);
    defer cp.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 6, 24, 120, 720 }, cp.data);
    try std.testing.expectEqual(@as(usize, 5), try argmax(f64, a));
    try std.testing.expectEqual(@as(usize, 0), try argmin(f64, a));
    var arg1 = try argmaxAxis(f64, a, 1, false);
    defer arg1.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, arg1.data);
    var t = try a.transpose();
    defer t.deinit();
    var mm = try a.matmul(t);
    defer mm.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, mm.shape);
    try std.testing.expectEqualSlices(f64, &.{ 14, 32, 32, 77 }, mm.data);
}

test "tensor scipy-like statistics and softmax" {
    const gpa = std.testing.allocator;
    var a = try tensor(f64, gpa, &.{ 1, 2, 3, 4 }, &.{4});
    defer a.deinit();
    var mean_value = try a.mean(null, false);
    defer mean_value.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), mean_value.data[0], 1e-12);
    var std_t = try a.stddev(null, false, 0);
    defer std_t.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1.118033988749895), std_t.data[0], 1e-12);
    var mean_top = try mean(f64, a, null, false);
    defer mean_top.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), mean_top.data[0], 1e-12);
    var var_top = try variance(f64, a, null, false, 0);
    defer var_top.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1.25), var_top.data[0], 1e-12);
    var std_top = try stddev(f64, a, null, false, 0);
    defer std_top.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1.118033988749895), std_top.data[0], 1e-12);
    var norm_top = try norm(f64, a, 2, null, false);
    defer norm_top.deinit();
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 30)), norm_top.data[0], 1e-12);

    var logits = try tensor(f64, gpa, &.{ 1, 2, 3, 1, 2, 3 }, &.{ 2, 3 });
    defer logits.deinit();
    var probs = try logits.softmax(1);
    defer probs.deinit();
    var row_sums = try probs.sum(1, false);
    defer row_sums.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1), row_sums.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), row_sums.data[1], 1e-12);

    var mask = try tensor(bool, gpa, &.{ true, true, false, true }, &.{ 2, 2 });
    defer mask.deinit();
    var all_rows = try allAxis(mask, 1, false);
    defer all_rows.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false }, all_rows.data);
    var any_cols = try anyAxis(mask, 0, false);
    defer any_cols.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true }, any_cols.data);
}

test "tensor pytorch numpy shape indexing and layout helpers" {
    const gpa = std.testing.allocator;
    var a = try tensor(f64, gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a.deinit();
    try std.testing.expectEqual(@as(usize, 2), a.ndim());
    try std.testing.expectEqual(@as(usize, 3), try a.size(1));
    try std.testing.expectEqual(@as(f64, 5), try a.at(&.{ 1, 1 }));

    var u = try a.unsqueeze(0);
    defer u.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 3 }, u.shape);
    var s2 = try u.squeeze(null);
    defer s2.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, s2.shape);

    var p = try a.permute(&.{ 1, 0 });
    defer p.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, p.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 4, 2, 5, 3, 6 }, p.data);

    var n = try a.narrow(1, 1, 2);
    defer n.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, n.shape);
    try std.testing.expectEqualSlices(f64, &.{ 2, 3, 5, 6 }, n.data);
}

test "tensor take mask stack cat and neural helpers" {
    const gpa = std.testing.allocator;
    var a = try tensor(f64, gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a.deinit();
    var idx = try tensor(usize, gpa, &.{ 2, 0 }, &.{2});
    defer idx.deinit();
    var picked = try a.indexSelect(1, idx);
    defer picked.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, picked.shape);
    try std.testing.expectEqualSlices(f64, &.{ 3, 1, 6, 4 }, picked.data);

    var mask = try tensor(bool, gpa, &.{ true, false, true, false, true, false }, &.{ 2, 3 });
    defer mask.deinit();
    var masked = try a.maskedSelect(mask);
    defer masked.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 5 }, masked.data);

    const pieces = [_]Tensor(f64){ a, a };
    var st = try Tensor(f64).stack(gpa, pieces[0..], 1);
    defer st.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 3 }, st.shape);
    var ca = try Tensor(f64).cat(gpa, pieces[0..], 0);
    defer ca.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 4, 3 }, ca.shape);

    var shifted = try a.subScalar(3);
    defer shifted.deinit();
    var relu_out = try shifted.relu();
    defer relu_out.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 0, 0, 1, 2, 3 }, relu_out.data);
    var cs = try a.cumsum();
    defer cs.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 6, 10, 15, 21 }, cs.data);
    try std.testing.expectEqual(@as(usize, 5), try a.argmax());
}

test "array advanced indexing mutation helpers" {
    const gpa = std.testing.allocator;
    var a = try array(f64, gpa, &.{ 1, 0, 3, 0, 5, 6 }, &.{ 2, 3 });
    defer a.deinit();

    var flat_idx = try a.flatNonzero();
    defer flat_idx.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 4, 5 }, flat_idx.data);

    var coords = try argwhere(f64, a);
    defer coords.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 4, 2 }, coords.shape);
    try std.testing.expectEqualSlices(usize, &.{ 0, 0, 0, 2, 1, 1, 1, 2 }, coords.data);

    var cond = try array(bool, gpa, &.{ true, false, true }, &.{3});
    defer cond.deinit();
    var compressed_cols = try compress(f64, a, cond, 1);
    defer compressed_cols.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, compressed_cols.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 0, 6 }, compressed_cols.data);

    var flat_cond = try array(bool, gpa, &.{ true, false, false, true, true, false }, &.{6});
    defer flat_cond.deinit();
    var compressed_flat = try a.compress(flat_cond, null);
    defer compressed_flat.deinit();
    try std.testing.expectEqualSlices(usize, &.{3}, compressed_flat.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 0, 5 }, compressed_flat.data);

    var mask = try array(bool, gpa, &.{ true, false, true, false, true, false }, &.{ 2, 3 });
    defer mask.deinit();
    var mask_values = try array(f64, gpa, &.{ 10, 20, 30 }, &.{3});
    defer mask_values.deinit();
    var mask_put = try maskedPut(f64, a, mask, mask_values);
    defer mask_put.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 10, 0, 20, 0, 30, 6 }, mask_put.data);

    var mask_scalar = try a.maskedPutScalar(mask, -1);
    defer mask_scalar.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -1, 0, -1, 0, -1, 6 }, mask_scalar.data);

    var put_idx = try array(usize, gpa, &.{ 1, 4 }, &.{2});
    defer put_idx.deinit();
    var put_values = try array(f64, gpa, &.{ 11, 44 }, &.{2});
    defer put_values.deinit();
    var put_flat = try putFlat(f64, a, put_idx, put_values);
    defer put_flat.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 11, 3, 0, 44, 6 }, put_flat.data);

    var put_scalar = try a.putFlatScalar(put_idx, 7);
    defer put_scalar.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 7, 3, 0, 7, 6 }, put_scalar.data);

    var index_put = try indexPut(f64, a, put_idx, put_values);
    defer index_put.deinit();
    try std.testing.expectEqualSlices(f64, put_flat.data, index_put.data);

    var index_put_scalar = try indexPutScalar(f64, a, put_idx, 9);
    defer index_put_scalar.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 9, 3, 0, 9, 6 }, index_put_scalar.data);

    var bad_values = try array(f64, gpa, &.{ 1, 2 }, &.{2});
    defer bad_values.deinit();
    try std.testing.expectError(error.ShapeMismatch, a.maskedPut(mask, bad_values));
    var bad_indices = try array(usize, gpa, &.{6}, &.{1});
    defer bad_indices.deinit();
    try std.testing.expectError(error.IndexOutOfBounds, a.putFlatScalar(bad_indices, 1));
}

test "array extended unary math and predicates" {
    const gpa = std.testing.allocator;
    var x = try array(f64, gpa, &.{ -1.7, -0.2, 0.0, 0.2, 1.7 }, &.{5});
    defer x.deinit();

    var floored = try floor(f64, x);
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

    var signs = try sign(f64, x);
    defer signs.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -1, -1, 0, 1, 1 }, signs.data);
    var bits = try x.signbit();
    defer bits.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false, false }, bits.data);

    var sq = try square(f64, x);
    defer sq.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 2.89), sq.data[0], 1e-12);
    var denom = try array(f64, gpa, &.{ 2, -4 }, &.{2});
    defer denom.deinit();
    var recip = try reciprocal(f64, denom);
    defer recip.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0.5, -0.25 }, recip.data);

    var stable = try array(f64, gpa, &.{ 0, 1 }, &.{2});
    defer stable.deinit();
    var e1 = try stable.expm1();
    defer e1.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0), e1.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.e - 1, e1.data[1], 1e-12);
    var l1 = try log1p(f64, stable);
    defer l1.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0), l1.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.ln2, l1.data[1], 1e-12);
    var powers = try array(f64, gpa, &.{ 1, 10, 100 }, &.{3});
    defer powers.deinit();
    var log2_out = try log2(f64, powers);
    defer log2_out.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0), log2_out.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log2(@as(f64, 10)), log2_out.data[1], 1e-12);
    var log10_out = try powers.log10();
    defer log10_out.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 1, 2 }, log10_out.data);

    var degrees = try array(f64, gpa, &.{ 0, 90, 180 }, &.{3});
    defer degrees.deinit();
    var radians = try deg2rad(f64, degrees);
    defer radians.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0), radians.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.pi / 2.0, radians.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.pi, radians.data[2], 1e-12);
    var roundtrip_degrees = try rad2deg(f64, radians);
    defer roundtrip_degrees.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0), roundtrip_degrees.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 90), roundtrip_degrees.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 180), roundtrip_degrees.data[2], 1e-12);

    var angles = try array(f64, gpa, &.{ 0, std.math.pi / 2.0 }, &.{2});
    defer angles.deinit();
    var sine = try sin(f64, angles);
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

    var unit = try array(f64, gpa, &.{ 0, 1 }, &.{2});
    defer unit.deinit();
    var arcs = try unit.asin();
    defer arcs.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0), arcs.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.pi / 2.0, arcs.data[1], 1e-12);
    var arcc = try unit.acos();
    defer arcc.deinit();
    try std.testing.expectApproxEqAbs(std.math.pi / 2.0, arcc.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0), arcc.data[1], 1e-12);
    var arct = try unit.atan();
    defer arct.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0), arct.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.pi / 4.0, arct.data[1], 1e-12);

    var hyp = try array(f64, gpa, &.{ 0, 1 }, &.{2});
    defer hyp.deinit();
    var sh = try hyp.sinh();
    defer sh.deinit();
    var ch = try cosh(f64, hyp);
    defer ch.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0), sh.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), ch.data[0], 1e-12);

    var special = try array(f64, gpa, &.{ 1, std.math.inf(f64), std.math.nan(f64) }, &.{3});
    defer special.deinit();
    var finite_mask = try isFinite(f64, special);
    defer finite_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false, false }, finite_mask.data);
    var inf_mask = try special.isinf();
    defer inf_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true, false }, inf_mask.data);
    var nan_mask = try isnan(f64, special);
    defer nan_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, false, true }, nan_mask.data);

    var ints = try array(i32, gpa, &.{ -2, 0, 7 }, &.{3});
    defer ints.deinit();
    var int_sign = try ints.sign();
    defer int_sign.deinit();
    try std.testing.expectEqualSlices(i32, &.{ -1, 0, 1 }, int_sign.data);
    var int_finite = try ints.isfinite();
    defer int_finite.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, true }, int_finite.data);
}

test "tensor gather scatter and scalar scatter" {
    const gpa = std.testing.allocator;
    var a = try tensor(f64, gpa, &.{ 10, 11, 12, 20, 21, 22 }, &.{ 2, 3 });
    defer a.deinit();
    var idx = try tensor(usize, gpa, &.{ 2, 1, 0, 0, 2, 1 }, &.{ 2, 3 });
    defer idx.deinit();

    var gathered = try a.gather(1, idx);
    defer gathered.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 12, 11, 10, 20, 22, 21 }, gathered.data);

    var base = try zeros(f64, gpa, &.{ 2, 3 });
    defer base.deinit();
    var scattered = try base.scatter(1, idx, gathered);
    defer scattered.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 10, 11, 12, 20, 21, 22 }, scattered.data);

    var filled = try base.scatterScalar(1, idx, 7);
    defer filled.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 7, 7, 7, 7, 7, 7 }, filled.data);
}

test "tensor logsoftmax norm and matrix helpers" {
    const gpa = std.testing.allocator;
    var logits = try tensor(f64, gpa, &.{ 1, 2, 3, 1, 2, 3 }, &.{ 2, 3 });
    defer logits.deinit();
    var log_probs = try logits.logSoftmax(1);
    defer log_probs.deinit();
    var probs = try log_probs.exp();
    defer probs.deinit();
    var row_sums = try probs.sum(1, false);
    defer row_sums.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1), row_sums.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), row_sums.data[1], 1e-12);

    var v = try tensor(f64, gpa, &.{ 3, 4 }, &.{2});
    defer v.deinit();
    var n = try v.norm(2, null, false);
    defer n.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 5), n.data[0], 1e-12);

    var w = try tensor(f64, gpa, &.{ 2, 5, 7 }, &.{3});
    defer w.deinit();
    var out = try v.outer(w);
    defer out.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, out.shape);
    try std.testing.expectEqualSlices(f64, &.{ 6, 15, 21, 8, 20, 28 }, out.data);

    var m = try tensor(f64, gpa, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, &.{ 3, 3 });
    defer m.deinit();
    try std.testing.expectEqual(@as(f64, 15), try m.trace());
    var d = try m.diagonal(0);
    defer d.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 5, 9 }, d.data);
    var upper = try m.triu(0);
    defer upper.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 3, 0, 5, 6, 0, 0, 9 }, upper.data);
    var lower = try m.tril(0);
    defer lower.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 0, 0, 4, 5, 0, 7, 8, 9 }, lower.data);
}

test "tensor min max arg reductions and topk" {
    const gpa = std.testing.allocator;
    var a = try tensor(f64, gpa, &.{ 9, 1, 5, 4, 8, 2 }, &.{ 2, 3 });
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
    var a = try array(f64, gpa, &.{ 1, 4, 2, 8, 3, 9 }, &.{ 2, 3 });
    defer a.deinit();

    var med_flat = try median(f64, a, null, false);
    defer med_flat.deinit();
    try std.testing.expectEqual(@as(usize, 0), med_flat.shape.len);
    try std.testing.expectApproxEqAbs(@as(f64, 3.5), med_flat.data[0], 1e-12);

    var med_rows = try a.median(1, false);
    defer med_rows.deinit();
    try std.testing.expectEqualSlices(usize, &.{2}, med_rows.shape);
    try std.testing.expectEqualSlices(f64, &.{ 2, 8 }, med_rows.data);

    var q_cols = try quantile(f64, a, 0.25, 0, true);
    defer q_cols.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 3 }, q_cols.shape);
    try std.testing.expectEqualSlices(f64, &.{ 2.75, 3.25, 3.75 }, q_cols.data);

    var p_flat = try percentile(f64, a, 75, null, false);
    defer p_flat.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 7), p_flat.data[0], 1e-12);
    try std.testing.expectError(error.InvalidShape, a.quantile(1.5, null, false));

    var obs_by_var = try array(f64, gpa, &.{
        1, 2,
        2, 4,
        3, 6,
    }, &.{ 3, 2 });
    defer obs_by_var.deinit();

    var covariance = try cov(f64, obs_by_var, false, 1);
    defer covariance.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, covariance.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 2, 4 }, covariance.data);

    var corr = try obs_by_var.corrcoef(false);
    defer corr.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 1, 1, 1 }, corr.data);

    var rowvar_data = try array(f64, gpa, &.{
        1, 2, 3,
        2, 4, 6,
    }, &.{ 2, 3 });
    defer rowvar_data.deinit();
    var row_cov = try rowvar_data.cov(true, 1);
    defer row_cov.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 2, 4 }, row_cov.data);

    var v = try array(f64, gpa, &.{ 1, 2, 3 }, &.{3});
    defer v.deinit();
    var var_scalar = try v.cov(true, 1);
    defer var_scalar.deinit();
    try std.testing.expectEqual(@as(usize, 0), var_scalar.shape.len);
    try std.testing.expectApproxEqAbs(@as(f64, 1), var_scalar.data[0], 1e-12);
    var corr_scalar = try corrcoef(f64, v, true);
    defer corr_scalar.deinit();
    try std.testing.expectEqual(@as(usize, 0), corr_scalar.shape.len);
    try std.testing.expectApproxEqAbs(@as(f64, 1), corr_scalar.data[0], 1e-12);
}

test "array nan cleanup and nan-aware statistics" {
    const gpa = std.testing.allocator;
    const nan = std.math.nan(f64);
    const inf = std.math.inf(f64);
    var a = try array(f64, gpa, &.{
        1,   nan, 3,
        nan, nan, 6,
        7,   8,   inf,
    }, &.{ 3, 3 });
    defer a.deinit();

    var cleaned = try nanToNum(f64, a, 0, 99, -99);
    defer cleaned.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 0, 3, 0, 0, 6, 7, 8, 99 }, cleaned.data);

    var row_sum = try a.nansum(1, false);
    defer row_sum.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 4, 6, inf }, row_sum.data);
    var col_sum_keep = try nansum(f64, a, 0, true);
    defer col_sum_keep.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 3 }, col_sum_keep.shape);
    try std.testing.expectEqualSlices(f64, &.{ 8, 8, inf }, col_sum_keep.data);

    var row_mean = try nanmean(f64, a, 1, false);
    defer row_mean.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 2, 6, inf }, row_mean.data);
    var col_mean = try a.nanmean(0, false);
    defer col_mean.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 4, 8, inf }, col_mean.data);

    var clean_stats = try array(f64, gpa, &.{
        1, nan, 3,
        2, nan, 6,
        3, 8,   9,
    }, &.{ 3, 3 });
    defer clean_stats.deinit();
    var variance_cols = try nanvar(f64, clean_stats, 0, false, 0);
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
    var maxs = try nanmax(f64, clean_stats, 1, false);
    defer maxs.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 3, 6, 9 }, maxs.data);

    var med = try nanmedian(f64, clean_stats, 0, false);
    defer med.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 2, 8, 6 }, med.data);
    var q = try clean_stats.nanquantile(0.25, 0, true);
    defer q.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 3 }, q.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1.5, 8, 4.5 }, q.data);
    var pct = try nanpercentile(f64, clean_stats, 75, 1, false);
    defer pct.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 2.5, 5, 8.5 }, pct.data);

    var all_nan = try array(f64, gpa, &.{ nan, nan }, &.{2});
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
    var a = try array(f64, gpa, &.{ 8, 1, 5, 3, 7, 2 }, &.{ 2, 3 });
    defer a.deinit();

    var flat_desc = try sortDescending(f64, a, null);
    defer flat_desc.deinit();
    try std.testing.expectEqualSlices(usize, &.{6}, flat_desc.shape);
    try std.testing.expectEqualSlices(f64, &.{ 8, 7, 5, 3, 2, 1 }, flat_desc.data);

    var row_sorted = try a.sort(1);
    defer row_sorted.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 5, 8, 2, 3, 7 }, row_sorted.data);

    var col_sorted_desc = try a.sortBy(0, true);
    defer col_sorted_desc.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 8, 7, 5, 3, 1, 2 }, col_sorted_desc.data);

    var row_order = try argsortAxis(f64, a, 1, false);
    defer row_order.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 0, 2, 0, 1 }, row_order.data);

    var flat_order_desc = try a.argsortDescending();
    defer flat_order_desc.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 4, 2, 3, 5, 1 }, flat_order_desc.data);

    var col_sorted = try sortWithIndices(f64, a, 0, false);
    defer col_sorted.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 3, 1, 2, 8, 7, 5 }, col_sorted.values.data);
    try std.testing.expectEqualSlices(usize, &.{ 1, 0, 1, 0, 1, 0 }, col_sorted.indices.data);

    var row_partition = try partition(f64, a, 1, 1, false);
    defer row_partition.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 5, 8, 2, 3, 7 }, row_partition.data);
    var row_argpartition = try a.argpartition(1, 1, false);
    defer row_argpartition.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 0, 2, 0, 1 }, row_argpartition.data);

    try std.testing.expectError(error.InvalidAxis, a.sort(2));
    try std.testing.expectError(error.InvalidShape, a.partition(3, 1, false));

    var flags = try array(bool, gpa, &.{ true, false, false, true }, &.{ 2, 2 });
    defer flags.deinit();
    var sorted_flags = try flags.sort(1);
    defer sorted_flags.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, sorted_flags.data);
}

test "tensor bool all any axis reductions" {
    const gpa = std.testing.allocator;
    var mask = try tensor(bool, gpa, &.{ true, true, false, true, false, false }, &.{ 2, 3 });
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
}

test "array aliases and alea-backed random distributions" {
    const gpa = std.testing.allocator;
    var a = try array(f64, gpa, &.{ 1, 2, 3, 4 }, &.{ 2, 2 });
    defer a.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, a.shape);

    var u = try uniform(f64, gpa, &.{16}, -2.0, 3.0, 123);
    defer u.deinit();
    for (u.data) |v| try std.testing.expect(v >= -2.0 and v < 3.0);

    var n = try normal(f64, gpa, &.{8}, 10.0, 0.0, 123);
    defer n.deinit();
    for (n.data) |v| try std.testing.expectEqual(@as(f64, 10.0), v);

    var r = try randint(i64, gpa, &.{32}, 2, 7, 456);
    defer r.deinit();
    for (r.data) |v| try std.testing.expect(v >= 2 and v < 7);

    var b0 = try bernoulli(gpa, &.{4}, 0.0, 789);
    defer b0.deinit();
    try std.testing.expect(!b0.any());
    var b1 = try bernoulli(gpa, &.{4}, 1.0, 789);
    defer b1.deinit();
    try std.testing.expect(b1.all());
}

test "alea-backed advanced random distributions" {
    const gpa = std.testing.allocator;
    var e = try exponential(f64, gpa, &.{16}, 2.0, 111);
    defer e.deinit();
    for (e.data) |v| try std.testing.expect(v >= 0);

    var g0 = try gamma(f64, gpa, &.{4}, 2.0, 0.0, 222);
    defer g0.deinit();
    for (g0.data) |v| try std.testing.expectEqual(@as(f64, 0), v);

    var be = try beta(f64, gpa, &.{16}, 2.0, 5.0, 333);
    defer be.deinit();
    for (be.data) |v| try std.testing.expect(v >= 0 and v <= 1);

    var p0 = try poisson(gpa, &.{8}, 0.0, 444);
    defer p0.deinit();
    try std.testing.expectEqualSlices(u64, &.{ 0, 0, 0, 0, 0, 0, 0, 0 }, p0.data);
}

test "alea-backed additional continuous distributions" {
    const gpa = std.testing.allocator;
    var ln = try lognormal(f64, gpa, &.{8}, 0.0, 0.0, 555);
    defer ln.deinit();
    for (ln.data) |v| try std.testing.expectApproxEqAbs(@as(f64, 1), v, 1e-12);

    var st = try studentT(f64, gpa, &.{8}, 8.0, 666);
    defer st.deinit();
    for (st.data) |v| try std.testing.expect(std.math.isFinite(v));

    var ca = try cauchy(f64, gpa, &.{8}, 0.0, 1.0, 777);
    defer ca.deinit();
    for (ca.data) |v| try std.testing.expect(std.math.isFinite(v));

    var la = try laplace(f64, gpa, &.{8}, 0.0, 2.0, 888);
    defer la.deinit();
    for (la.data) |v| try std.testing.expect(std.math.isFinite(v));

    var wb = try weibull(f64, gpa, &.{8}, 2.0, 1.5, 999);
    defer wb.deinit();
    for (wb.data) |v| try std.testing.expect(v >= 0);
}

test "array scatter add and reduce variants" {
    const gpa = std.testing.allocator;
    var base = try zeros(f64, gpa, &.{ 2, 3 });
    defer base.deinit();
    var idx = try array(usize, gpa, &.{ 0, 1, 1, 2, 0, 2 }, &.{ 2, 3 });
    defer idx.deinit();
    var src = try array(f64, gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer src.deinit();

    var added = try base.scatterAdd(1, idx, src);
    defer added.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 5, 0, 5, 0, 10 }, added.data);

    var ones_base = try ones(f64, gpa, &.{ 2, 3 });
    defer ones_base.deinit();
    var product_out = try ones_base.scatterReduce(1, idx, src, .prod);
    defer product_out.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 6, 1, 5, 1, 24 }, product_out.data);

    var max_base = try full(f64, gpa, &.{ 2, 3 }, -100);
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
    var a = try array(f64, gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a.deinit();

    var z = try a.zerosLike();
    defer z.deinit();
    try std.testing.expectEqualSlices(usize, a.shape, z.shape);
    try std.testing.expectEqualSlices(f64, &.{ 0, 0, 0, 0, 0, 0 }, z.data);

    var o = try onesLike(f64, a);
    defer o.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 1, 1, 1, 1, 1 }, o.data);

    var f = try fullLike(f64, a, 7);
    defer f.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 7, 7, 7, 7, 7, 7 }, f.data);

    var s = try arrayScalar(f64, gpa, 42);
    defer s.deinit();
    try std.testing.expectEqual(@as(usize, 0), s.shape.len);
    try std.testing.expectEqual(@as(f64, 42), try s.item());

    var v = try array(f64, gpa, &.{ 1, 2, 3 }, &.{3});
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
}

test "array advanced indexing and mask mutation helpers" {
    const gpa = std.testing.allocator;
    var a = try array(f64, gpa, &.{ 10, 11, 12, 20, 21, 22 }, &.{ 2, 3 });
    defer a.deinit();
    var idx = try array(usize, gpa, &.{ 2, 0, 1, 1, 2, 0 }, &.{ 2, 3 });
    defer idx.deinit();

    var taken = try a.takeAlongAxis(idx, 1);
    defer taken.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 12, 10, 11, 21, 22, 20 }, taken.data);

    var base = try zeros(f64, gpa, &.{ 2, 3 });
    defer base.deinit();
    var put = try base.putAlongAxis(idx, taken, 1);
    defer put.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 10, 11, 12, 20, 21, 22 }, put.data);

    var mask = try array(bool, gpa, &.{ true, false, true, false, true, false }, &.{ 2, 3 });
    defer mask.deinit();
    var filled = try a.maskedFill(mask, -1);
    defer filled.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -1, 11, -1, 20, -1, 22 }, filled.data);

    var src = try array(f64, gpa, &.{ 100, 200, 300 }, &.{3});
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
    var a = try array(f64, gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a.deinit();

    var sliced = try a.sliceAxis(1, .{ .start = 0, .stop = 3, .step = 2 });
    defer sliced.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, sliced.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 4, 6 }, sliced.data);

    var flipped = try a.flip(1);
    defer flipped.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 3, 2, 1, 6, 5, 4 }, flipped.data);

    var rolled = try a.roll(1, 1);
    defer rolled.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 3, 1, 2, 6, 4, 5 }, rolled.data);

    var rolled_neg = try a.roll(-1, 0);
    defer rolled_neg.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 4, 5, 6, 1, 2, 3 }, rolled_neg.data);

    var padded = try a.padConstant(&.{ 1, 1 }, &.{ 0, 2 }, 0);
    defer padded.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 6 }, padded.shape);
    try std.testing.expectEqualSlices(f64, &.{
        0, 0, 0, 0, 0, 0,
        0, 1, 2, 3, 0, 0,
        0, 4, 5, 6, 0, 0,
    }, padded.data);
}

test "array dtype metadata and casts cover common numeric types" {
    try std.testing.expectEqual(DType.i8, DType.of(i8));
    try std.testing.expectEqual(DType.i16, DType.of(i16));
    try std.testing.expectEqual(DType.u16, DType.of(u16));
    try std.testing.expectEqual(DType.u32, DType.of(u32));
    try std.testing.expectEqual(DType.u64, DType.of(u64));
    try std.testing.expectEqualStrings("u64", DType.u64.name());
    try std.testing.expectEqual(@as(usize, 8), DType.u64.byteSize());
    try std.testing.expect(DType.f32.isFloat());
    try std.testing.expect(DType.i16.isInteger());
    try std.testing.expect(DType.i16.isSigned());
    try std.testing.expect(DType.bool.isBool());

    const gpa = std.testing.allocator;
    var ints = try array(i16, gpa, &.{ -1, 0, 2 }, &.{3});
    defer ints.deinit();
    var floats = try ints.astype(f32);
    defer floats.deinit();
    try std.testing.expectEqualSlices(f32, &.{ -1, 0, 2 }, floats.data);
    var unsigned = try array(u32, gpa, &.{ 1, 2, 3 }, &.{3});
    defer unsigned.deinit();
    var widened = try unsigned.astype(u64);
    defer widened.deinit();
    try std.testing.expectEqualSlices(u64, &.{ 1, 2, 3 }, widened.data);

    var r = try randint(u16, gpa, &.{16}, 10, 20, 42);
    defer r.deinit();
    for (r.data) |v| try std.testing.expect(v >= 10 and v < 20);
}

test "array bytes and archive serialization roundtrip" {
    const gpa = std.testing.allocator;
    var a = try array(i16, gpa, &.{ -1, 2, 300, -400 }, &.{ 2, 2 });
    defer a.deinit();

    const bytes = try a.toBytes(gpa);
    defer gpa.free(bytes);
    try std.testing.expectEqual(@as(usize, 8), bytes.len);
    var from_raw = try fromBytes(i16, gpa, bytes, &.{ 2, 2 });
    defer from_raw.deinit();
    try std.testing.expectEqualSlices(i16, a.data, from_raw.data);
    try std.testing.expectEqualSlices(usize, a.shape, from_raw.shape);

    const archive = try a.toArchive(gpa);
    defer gpa.free(archive);
    var restored = try fromArchive(i16, gpa, archive);
    defer restored.deinit();
    try std.testing.expectEqualSlices(i16, a.data, restored.data);
    try std.testing.expectEqualSlices(usize, a.shape, restored.shape);
    try std.testing.expectError(error.TypeUnsupported, fromArchive(f32, gpa, archive));
}

test "array axis cumulative operations and diff" {
    const gpa = std.testing.allocator;
    var a = try array(f64, gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a.deinit();

    var cs0 = try a.cumsumAxis(0);
    defer cs0.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 3, 5, 7, 9 }, cs0.data);
    var cs1 = try cumsumAxis(f64, a, 1);
    defer cs1.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 6, 4, 9, 15 }, cs1.data);

    var cp1 = try a.cumprodAxis(1);
    defer cp1.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 6, 4, 20, 120 }, cp1.data);

    var d1 = try a.diff(1, 1);
    defer d1.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, d1.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 1, 1, 1 }, d1.data);

    var d2 = try diff(f64, a, 1, 2);
    defer d2.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 1 }, d2.shape);
    try std.testing.expectEqualSlices(f64, &.{ 0, 0 }, d2.data);
}

test "array unique bincount searchsorted and clipArray" {
    const gpa = std.testing.allocator;
    var a = try array(i32, gpa, &.{ 3, 1, 2, 3, 2, 1, 4 }, &.{7});
    defer a.deinit();
    var u = try a.unique();
    defer u.deinit();
    try std.testing.expectEqualSlices(i32, &.{ 1, 2, 3, 4 }, u.data);
    var uc = try uniqueWithCounts(i32, a);
    defer uc.deinit();
    try std.testing.expectEqualSlices(i32, &.{ 1, 2, 3, 4 }, uc.values.data);
    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 2, 1 }, uc.counts.data);

    var counts = try bincount(i32, a, 6);
    defer counts.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 2, 2, 1, 0 }, counts.data);
    var weights = try array(f64, gpa, &.{ 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0 }, &.{7});
    defer weights.deinit();
    var weighted_counts = try bincountWeighted(i32, f64, a, weights, 6);
    defer weighted_counts.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 4, 4, 2.5, 4, 0 }, weighted_counts.data);

    var sorted = try array(f64, gpa, &.{ 1, 2, 2, 4 }, &.{4});
    defer sorted.deinit();
    var probes = try array(f64, gpa, &.{ 0, 2, 3, 5 }, &.{4});
    defer probes.deinit();
    var left = try sorted.searchsorted(probes, .left);
    defer left.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 3, 4 }, left.data);
    var right = try searchsorted(f64, sorted, probes, .right);
    defer right.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 3, 3, 4 }, right.data);
    var buckets = try bucketize(f64, probes, sorted, .right);
    defer buckets.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 3, 3, 4 }, buckets.data);
    var digits_left_open = try digitize(f64, probes, sorted, false);
    defer digits_left_open.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 3, 3, 4 }, digits_left_open.data);
    var digits_right_open = try probes.digitize(sorted, true);
    defer digits_right_open.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 3, 4 }, digits_right_open.data);

    var needles = try array(i32, gpa, &.{ 2, 4 }, &.{2});
    defer needles.deinit();
    var members = try a.isin(needles, false);
    defer members.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, true, false, true }, members.data);
    var non_members = try isin(i32, a, needles, true);
    defer non_members.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true, false, true, false }, non_members.data);

    var flags = try array(bool, gpa, &.{ true, false, true }, &.{3});
    defer flags.deinit();
    var unique_flags = try flags.unique();
    defer unique_flags.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true }, unique_flags.data);

    var x = try array(f64, gpa, &.{ -1, 0, 5, 10 }, &.{ 2, 2 });
    defer x.deinit();
    var lo = try array(f64, gpa, &.{ 0, 2 }, &.{2});
    defer lo.deinit();
    var hi = try array(f64, gpa, &.{4}, &.{1});
    defer hi.deinit();
    var clipped = try x.clipArray(lo, hi);
    defer clipped.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 2, 4, 4 }, clipped.data);
}
