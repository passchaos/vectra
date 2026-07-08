const std = @import("std");

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
    i32,
    i64,
    u8,
    usize,
    bool,

    pub fn of(comptime T: type) DType {
        return switch (T) {
            f32 => .f32,
            f64 => .f64,
            i32 => .i32,
            i64 => .i64,
            u8 => .u8,
            usize => .usize,
            bool => .bool,
            else => @compileError("unsupported Vectra dtype: " ++ @typeName(T)),
        };
    }
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
            ensureFloat(T);
            var prng = std.Random.DefaultPrng.init(seed);
            const rng = prng.random();
            const out = try Self.empty(allocator, dims);
            for (out.data) |*slot| {
                slot.* = rng.float(T);
            }
            return out;
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
            ensureFloat(T);
            var prng = std.Random.DefaultPrng.init(seed);
            const rng = prng.random();
            const out = try Self.empty(allocator, dims);
            for (out.data) |*slot| slot.* = rng.floatNorm(T);
            return out;
        }

        pub fn randint(allocator: std.mem.Allocator, dims: []const usize, low: T, high: T, seed: u64) TensorError!Self {
            if (comptime @typeInfo(T) != .int) @compileError("randint requires an integer tensor type");
            if (low >= high) return error.InvalidShape;
            var prng = std.Random.DefaultPrng.init(seed);
            const rng = prng.random();
            const out = try Self.empty(allocator, dims);
            for (out.data) |*slot| slot.* = rng.intRangeLessThan(T, low, high);
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
        fn opSqrt(a: T) T {
            return std.math.sqrt(a);
        }
        fn opSin(a: T) T {
            return std.math.sin(a);
        }
        fn opCos(a: T) T {
            return std.math.cos(a);
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
            return self.binaryScalar(scalar, struct {
                fn f(a: T, b: T) T {
                    return @divFloor(a, b);
                }
            }.f);
        }

        pub fn modScalar(self: Self, scalar: T) TensorError!Self {
            ensureNumeric(T);
            return self.binaryScalar(scalar, struct {
                fn f(a: T, b: T) T {
                    return @mod(a, b);
                }
            }.f);
        }

        pub fn neg(self: Self) TensorError!Self {
            ensureNumeric(T);
            return self.unary(opNeg);
        }

        pub fn abs(self: Self) TensorError!Self {
            ensureNumeric(T);
            return self.unary(opAbs);
        }

        pub fn exp(self: Self) TensorError!Self {
            ensureFloat(T);
            return self.unary(opExp);
        }

        pub fn log(self: Self) TensorError!Self {
            ensureFloat(T);
            return self.unary(opLog);
        }

        pub fn sqrt(self: Self) TensorError!Self {
            ensureFloat(T);
            return self.unary(opSqrt);
        }

        pub fn sin(self: Self) TensorError!Self {
            ensureFloat(T);
            return self.unary(opSin);
        }

        pub fn cos(self: Self) TensorError!Self {
            ensureFloat(T);
            return self.unary(opCos);
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

        pub fn eq(self: Self, other: Self) TensorError!Tensor(bool) {
            return self.compare(other, struct {
                fn f(a: T, b: T) bool {
                    return a == b;
                }
            }.f);
        }

        pub fn gt(self: Self, other: Self) TensorError!Tensor(bool) {
            ensureNumeric(T);
            return self.compare(other, struct {
                fn f(a: T, b: T) bool {
                    return a > b;
                }
            }.f);
        }

        pub fn lt(self: Self, other: Self) TensorError!Tensor(bool) {
            ensureNumeric(T);
            return self.compare(other, struct {
                fn f(a: T, b: T) bool {
                    return a < b;
                }
            }.f);
        }

        pub fn ne(self: Self, other: Self) TensorError!Tensor(bool) {
            return self.compare(other, struct {
                fn f(a: T, b: T) bool {
                    return a != b;
                }
            }.f);
        }

        pub fn ge(self: Self, other: Self) TensorError!Tensor(bool) {
            ensureNumeric(T);
            return self.compare(other, struct {
                fn f(a: T, b: T) bool {
                    return a >= b;
                }
            }.f);
        }

        pub fn le(self: Self, other: Self) TensorError!Tensor(bool) {
            ensureNumeric(T);
            return self.compare(other, struct {
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

        pub fn logicalOr(self: Self, other: Self) TensorError!Self {
            if (comptime T != bool) @compileError("logicalOr requires Tensor(bool)");
            return self.binaryTensor(other, struct {
                fn f(a: bool, b: bool) bool {
                    return a or b;
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
            return self.reduce(axis_opt, keepdims, self.data[0], struct {
                fn f(a: T, b: T) T {
                    return if (b < a) b else a;
                }
            }.f);
        }

        pub fn max(self: Self, axis_opt: ?isize, keepdims: bool) TensorError!Self {
            ensureNumeric(T);
            if (self.data.len == 0) return error.EmptyTensor;
            return self.reduce(axis_opt, keepdims, self.data[0], struct {
                fn f(a: T, b: T) T {
                    return if (b > a) b else a;
                }
            }.f);
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
                    const diff = v - mean_t.data[ravelIndex(mean_multi, mean_t.strides)];
                    const oi = ravelIndex(out_multi, out.strides);
                    out.data[oi] += diff * diff;
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
                const diff = v - mean_value;
                total += diff * diff;
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

        pub fn sort(self: Self, axis_opt: ?isize) TensorError!Self {
            ensureNumeric(T);
            if (axis_opt == null) {
                const out = try self.flatten();
                std.sort.insertion(T, out.data, {}, struct {
                    fn lessThan(_: void, a: T, b: T) bool {
                        return a < b;
                    }
                }.lessThan);
                return out;
            }
            const axis = try normalizeDim(axis_opt.?, self.shape.len);
            if (axis != self.shape.len - 1) return error.InvalidAxis;
            var out = try self.clone();
            const width = self.shape[axis];
            if (width == 0) return out;
            const rows = self.data.len / width;
            for (0..rows) |r| {
                std.sort.insertion(T, out.data[r * width ..][0..width], {}, struct {
                    fn lessThan(_: void, a: T, b: T) bool {
                        return a < b;
                    }
                }.lessThan);
            }
            return out;
        }

        pub fn argsort(self: Self) TensorError!Tensor(usize) {
            ensureNumeric(T);
            const idx = try Tensor(usize).empty(self.allocator, &.{self.data.len});
            for (idx.data, 0..) |*slot, i| slot.* = i;
            const Ctx = struct {
                data: []const T,
                fn lessThan(ctx: @This(), a: usize, b: usize) bool {
                    return ctx.data[a] < ctx.data[b];
                }
            };
            std.sort.insertion(usize, idx.data, Ctx{ .data = self.data }, Ctx.lessThan);
            return idx;
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

        pub fn print(self: Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
            try writer.print("Tensor({s}, shape=", .{@typeName(T)});
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

pub fn randint(comptime T: type, allocator: std.mem.Allocator, dims: []const usize, low: T, high: T, seed: u64) TensorError!Tensor(T) {
    return Tensor(T).randint(allocator, dims, low, high, seed);
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

test "tensor reductions and matmul" {
    const gpa = std.testing.allocator;
    var a = try tensor(f64, gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a.deinit();
    var s0 = try a.sum(0, false);
    defer s0.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 5, 7, 9 }, s0.data);
    var s1 = try a.sum(1, true);
    defer s1.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 1 }, s1.shape);
    try std.testing.expectEqualSlices(f64, &.{ 6, 15 }, s1.data);
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
    var mean = try a.mean(null, false);
    defer mean.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), mean.data[0], 1e-12);
    var std_t = try a.stddev(null, false, 0);
    defer std_t.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1.118033988749895), std_t.data[0], 1e-12);

    var logits = try tensor(f64, gpa, &.{ 1, 2, 3, 1, 2, 3 }, &.{ 2, 3 });
    defer logits.deinit();
    var probs = try logits.softmax(1);
    defer probs.deinit();
    var row_sums = try probs.sum(1, false);
    defer row_sums.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1), row_sums.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), row_sums.data[1], 1e-12);
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
    var relu = try shifted.relu();
    defer relu.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 0, 0, 1, 2, 3 }, relu.data);
    var cs = try a.cumsum();
    defer cs.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 6, 10, 15, 21 }, cs.data);
    try std.testing.expectEqual(@as(usize, 5), try a.argmax());
}
