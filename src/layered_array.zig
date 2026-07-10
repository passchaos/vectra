//! Layered Array abstractions for Vectra.
//!
//! The layers are intentionally explicit:
//! - `StaticArray(T, Layout)`: dtype and layout are comptime type parameters.
//! - `Array(T)`: existing Vectra typed array with runtime shape/strides layout.
//! - `AnyArray`: dtype and layout are runtime metadata.
//!
//! `Device` remains runtime metadata in every layer.  CUDA devices can be named
//! in APIs today, but persistent CUDA-resident array storage is still a future
//! backend and therefore returns `error.InvalidDevice` through constructors.

const std = @import("std");
const array_mod = @import("array.zig");

pub const ArrayError = array_mod.ArrayError;
pub const Array = array_mod.Array;
pub const Device = array_mod.Device;
pub const DType = array_mod.DType;

pub const default_dtype = f32;

/// Runtime creation options whose type carries the dtype.
///
/// This mirrors the PyTorch idea that every creation can state `dtype` and
/// `device`, while preserving Zig's static dtype specialization:
///
/// ```
/// const opts = vx.options(f32);              // type is CreationOptions(f32), device defaults to CPU
/// const gpu = vx.onDevice(f32, vx.cuda(0));  // dtype in type, device in value
/// const seeded = vx.seeded(f32, 42);         // dtype in type, CPU device, seed in value
/// ```
pub fn CreationOptions(comptime T: type) type {
    return struct {
        pub const dtype = T;

        device: Device = .cpu,
        seed: u64 = 0,
    };
}

pub fn options(comptime T: type) CreationOptions(T) {
    return .{};
}

pub fn onDevice(comptime T: type, device: Device) CreationOptions(T) {
    return .{ .device = device };
}

pub fn seeded(comptime T: type, seed: u64) CreationOptions(T) {
    return .{ .seed = seed };
}

pub fn seededOn(comptime T: type, device: Device, seed: u64) CreationOptions(T) {
    return .{ .device = device, .seed = seed };
}

pub const DimBinding = struct {
    name: []const u8,
    value: usize,
};

/// Marker alias for the symbolic dimension type-level DSL.  A symbolic layout
/// stores an array of dimension expression *types* so the expression graph is
/// comptime-known while symbol values are bound at runtime.
pub const DimExpr = type;

pub fn dim(comptime value: usize) type {
    return struct {
        pub fn eval(bindings: []const DimBinding) ArrayError!usize {
            _ = bindings;
            return value;
        }
    };
}

pub fn symbol(comptime name: []const u8) type {
    return struct {
        pub const symbol_name = name;

        pub fn eval(bindings: []const DimBinding) ArrayError!usize {
            return lookupSymbol(name, bindings);
        }
    };
}

pub fn dimAdd(comptime lhs: type, comptime rhs: type) type {
    return struct {
        pub fn eval(bindings: []const DimBinding) ArrayError!usize {
            return checkedAdd(try lhs.eval(bindings), try rhs.eval(bindings));
        }
    };
}

pub fn dimSub(comptime lhs: type, comptime rhs: type) type {
    return struct {
        pub fn eval(bindings: []const DimBinding) ArrayError!usize {
            return checkedSub(try lhs.eval(bindings), try rhs.eval(bindings));
        }
    };
}

pub fn dimMul(comptime lhs: type, comptime rhs: type) type {
    return struct {
        pub fn eval(bindings: []const DimBinding) ArrayError!usize {
            return checkedMul(try lhs.eval(bindings), try rhs.eval(bindings));
        }
    };
}

pub const LayoutOrder = enum {
    /// Last index is contiguous: C/NumPy default for dense arrays.
    row_major,
    /// First index is contiguous: Fortran/BLAS column-major layout.
    column_major,
};

pub const RuntimeLayout = struct {
    shape: []const usize,
    strides: []const usize,

    pub fn rank(self: RuntimeLayout) usize {
        return self.shape.len;
    }

    pub fn numel(self: RuntimeLayout) ArrayError!usize {
        return array_mod.numelFrom(self.shape);
    }

    pub fn offset(self: RuntimeLayout, indices: []const usize) ArrayError!usize {
        if (indices.len != self.shape.len or self.strides.len != self.shape.len) return error.InvalidShape;
        var linear: usize = 0;
        for (indices, 0..) |index, axis| {
            if (index >= self.shape[axis]) return error.IndexOutOfBounds;
            linear += index * self.strides[axis];
        }
        return linear;
    }
};

pub fn runtimeLayoutOf(comptime T: type, input: Array(T)) RuntimeLayout {
    return .{ .shape = input.shape, .strides = input.strides };
}

pub fn StaticLayout(comptime rank_value: usize, comptime shape_value: [rank_value]usize, comptime order_value: LayoutOrder) type {
    const strides_value = computeStaticStrides(rank_value, shape_value, order_value);
    const numel_value = computeStaticNumel(rank_value, shape_value);
    return struct {
        pub const rank = rank_value;
        pub const shape = shape_value;
        pub const strides = strides_value;
        pub const order = order_value;
        pub const numel = numel_value;

        pub fn runtime() RuntimeLayout {
            return .{ .shape = shape[0..], .strides = strides[0..] };
        }

        pub fn offset(indices: [rank]usize) ArrayError!usize {
            var linear: usize = 0;
            inline for (0..rank) |axis| {
                if (indices[axis] >= shape[axis]) return error.IndexOutOfBounds;
                linear += indices[axis] * strides[axis];
            }
            return linear;
        }
    };
}

pub fn SymbolicLayout(comptime rank_value: usize, comptime shape_exprs_value: [rank_value]type, comptime order_value: LayoutOrder) type {
    return struct {
        pub const rank = rank_value;
        pub const shape_exprs = shape_exprs_value;
        pub const order = order_value;

        pub fn evaluateShape(bindings: []const DimBinding, out: *[rank]usize) ArrayError!void {
            inline for (0..rank) |axis| {
                out[axis] = try shape_exprs[axis].eval(bindings);
            }
        }

        pub fn evaluateStrides(shape: [rank]usize, out: *[rank]usize) void {
            computeRuntimeStrides(rank, shape, order, out);
        }

        pub fn runtime(bindings: []const DimBinding, shape_out: *[rank]usize, strides_out: *[rank]usize) ArrayError!RuntimeLayout {
            try evaluateShape(bindings, shape_out);
            evaluateStrides(shape_out.*, strides_out);
            return .{ .shape = shape_out[0..], .strides = strides_out[0..] };
        }

        pub fn numel(bindings: []const DimBinding) ArrayError!usize {
            var shape: [rank]usize = undefined;
            try evaluateShape(bindings, &shape);
            return computeRuntimeNumel(rank, shape);
        }

        pub fn offset(bindings: []const DimBinding, indices: [rank]usize) ArrayError!usize {
            var shape: [rank]usize = undefined;
            var strides: [rank]usize = undefined;
            const rt = try runtime(bindings, &shape, &strides);
            return rt.offset(indices[0..]);
        }
    };
}

pub fn StaticArray(comptime T: type, comptime Layout: type) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        data: []T,
        device: Device = .cpu,

        pub const Scalar = T;
        pub const dtype = DType.of(T);
        pub const layout = Layout;

        pub fn fromSlice(allocator: std.mem.Allocator, values: []const T, device: Device) ArrayError!Self {
            if (values.len != Layout.numel) return error.ShapeMismatch;
            if (!device.isAvailable()) return error.InvalidDevice;
            return .{
                .allocator = allocator,
                .data = try allocator.dupe(T, values),
                .device = device,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.data);
            self.* = undefined;
        }

        pub fn runtimeLayout(self: Self) RuntimeLayout {
            _ = self;
            return Layout.runtime();
        }

        pub fn dtypeName(self: Self) []const u8 {
            _ = self;
            return dtype.name();
        }

        pub fn get(self: Self, indices: [Layout.rank]usize) ArrayError!T {
            return self.data[try Layout.offset(indices)];
        }

        pub fn set(self: Self, indices: [Layout.rank]usize, value: T) ArrayError!void {
            self.data[try Layout.offset(indices)] = value;
        }

        pub fn toRuntimeArray(self: Self) ArrayError!Array(T) {
            if (!self.device.isAvailable()) return error.InvalidDevice;
            var out = try Array(T).fromSlice(self.allocator, self.data, Layout.shape[0..]);
            errdefer out.deinit();
            @memcpy(out.strides, Layout.strides[0..]);
            out.device = self.device;
            return out;
        }
    };
}

pub fn SymbolicArray(comptime T: type, comptime Layout: type) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        data: []T,
        shape: [Layout.rank]usize,
        strides: [Layout.rank]usize,
        device: Device = .cpu,

        pub const Scalar = T;
        pub const dtype = DType.of(T);
        pub const layout = Layout;

        pub fn fromSlice(
            allocator: std.mem.Allocator,
            values: []const T,
            bindings: []const DimBinding,
            device: Device,
        ) ArrayError!Self {
            if (!device.isAvailable()) return error.InvalidDevice;
            var shape: [Layout.rank]usize = undefined;
            var strides: [Layout.rank]usize = undefined;
            try Layout.evaluateShape(bindings, &shape);
            Layout.evaluateStrides(shape, &strides);
            if (values.len != try computeRuntimeNumel(Layout.rank, shape)) return error.ShapeMismatch;
            return .{
                .allocator = allocator,
                .data = try allocator.dupe(T, values),
                .shape = shape,
                .strides = strides,
                .device = device,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.data);
            self.* = undefined;
        }

        pub fn runtimeLayout(self: *const Self) RuntimeLayout {
            return .{ .shape = self.shape[0..], .strides = self.strides[0..] };
        }

        pub fn dtypeName(self: Self) []const u8 {
            _ = self;
            return dtype.name();
        }

        pub fn get(self: Self, indices: [Layout.rank]usize) ArrayError!T {
            return self.data[try self.runtimeLayout().offset(indices[0..])];
        }

        pub fn set(self: Self, indices: [Layout.rank]usize, value: T) ArrayError!void {
            self.data[try self.runtimeLayout().offset(indices[0..])] = value;
        }

        pub fn toRuntimeArray(self: Self) ArrayError!Array(T) {
            if (!self.device.isAvailable()) return error.InvalidDevice;
            var out = try Array(T).fromSlice(self.allocator, self.data, self.shape[0..]);
            errdefer out.deinit();
            @memcpy(out.strides, self.strides[0..]);
            out.device = self.device;
            return out;
        }
    };
}

pub const AnyArray = struct {
    allocator: std.mem.Allocator,
    dtype: DType,
    device: Device = .cpu,
    data: []u8,
    shape: []usize,
    strides: []usize,

    pub fn fromTyped(comptime T: type, allocator: std.mem.Allocator, input: Array(T)) ArrayError!AnyArray {
        const bytes = std.mem.sliceAsBytes(input.data);
        const data = try allocator.dupe(u8, bytes);
        errdefer allocator.free(data);
        const shape = try allocator.dupe(usize, input.shape);
        errdefer allocator.free(shape);
        const strides = try allocator.dupe(usize, input.strides);
        errdefer allocator.free(strides);
        return .{
            .allocator = allocator,
            .dtype = DType.of(T),
            .device = input.device,
            .data = data,
            .shape = shape,
            .strides = strides,
        };
    }

    pub fn fromStatic(comptime T: type, comptime Layout: type, allocator: std.mem.Allocator, input: StaticArray(T, Layout)) ArrayError!AnyArray {
        const bytes = std.mem.sliceAsBytes(input.data);
        const data = try allocator.dupe(u8, bytes);
        errdefer allocator.free(data);
        const shape = try allocator.dupe(usize, Layout.shape[0..]);
        errdefer allocator.free(shape);
        const strides = try allocator.dupe(usize, Layout.strides[0..]);
        errdefer allocator.free(strides);
        return .{
            .allocator = allocator,
            .dtype = DType.of(T),
            .device = input.device,
            .data = data,
            .shape = shape,
            .strides = strides,
        };
    }

    pub fn fromSymbolic(comptime T: type, comptime Layout: type, allocator: std.mem.Allocator, input: SymbolicArray(T, Layout)) ArrayError!AnyArray {
        const bytes = std.mem.sliceAsBytes(input.data);
        const data = try allocator.dupe(u8, bytes);
        errdefer allocator.free(data);
        const shape = try allocator.dupe(usize, input.shape[0..]);
        errdefer allocator.free(shape);
        const strides = try allocator.dupe(usize, input.strides[0..]);
        errdefer allocator.free(strides);
        return .{
            .allocator = allocator,
            .dtype = DType.of(T),
            .device = input.device,
            .data = data,
            .shape = shape,
            .strides = strides,
        };
    }

    pub fn deinit(self: *AnyArray) void {
        self.allocator.free(self.data);
        self.allocator.free(self.shape);
        self.allocator.free(self.strides);
        self.* = undefined;
    }

    pub fn layout(self: AnyArray) RuntimeLayout {
        return .{ .shape = self.shape, .strides = self.strides };
    }

    pub fn rank(self: AnyArray) usize {
        return self.shape.len;
    }

    pub fn elementCount(self: AnyArray) ArrayError!usize {
        return self.layout().numel();
    }

    pub fn dtypeName(self: AnyArray) []const u8 {
        return self.dtype.name();
    }

    pub fn byteSize(self: AnyArray) usize {
        return self.data.len;
    }
};

pub const Context = struct {
    allocator: std.mem.Allocator,

    pub fn array(self: Context, comptime T: type, values: []const T, dims: []const usize) ArrayError!Array(T) {
        return Array(T).fromSlice(self.allocator, values, dims);
    }

    pub fn arrayWith(
        self: Context,
        opts: anytype,
        values: []const optionDType(@TypeOf(opts)),
        dims: []const usize,
    ) ArrayError!Array(optionDType(@TypeOf(opts))) {
        const T = optionDType(@TypeOf(opts));
        const out = try Array(T).fromSlice(self.allocator, values, dims);
        return finishDevice(T, out, opts.device);
    }

    pub fn zeros(self: Context, comptime T: type, dims: []const usize) ArrayError!Array(T) {
        return Array(T).zeros(self.allocator, dims);
    }

    pub fn zerosWith(self: Context, opts: anytype, dims: []const usize) ArrayError!Array(optionDType(@TypeOf(opts))) {
        const T = optionDType(@TypeOf(opts));
        const out = try Array(T).zeros(self.allocator, dims);
        return finishDevice(T, out, opts.device);
    }

    pub fn ones(self: Context, comptime T: type, dims: []const usize) ArrayError!Array(T) {
        return Array(T).ones(self.allocator, dims);
    }

    pub fn onesWith(self: Context, opts: anytype, dims: []const usize) ArrayError!Array(optionDType(@TypeOf(opts))) {
        const T = optionDType(@TypeOf(opts));
        const out = try Array(T).ones(self.allocator, dims);
        return finishDevice(T, out, opts.device);
    }

    pub fn rand(self: Context, comptime T: type, dims: []const usize, seed: u64) ArrayError!Array(T) {
        return Array(T).rand(self.allocator, dims, seed);
    }

    pub fn randWith(self: Context, opts: anytype, dims: []const usize) ArrayError!Array(optionDType(@TypeOf(opts))) {
        const T = optionDType(@TypeOf(opts));
        const out = try Array(T).rand(self.allocator, dims, opts.seed);
        return finishDevice(T, out, opts.device);
    }

    pub fn anyFromTyped(self: Context, comptime T: type, input: Array(T)) ArrayError!AnyArray {
        return AnyArray.fromTyped(T, self.allocator, input);
    }
};

pub fn withAllocator(allocator: std.mem.Allocator) Context {
    return .{ .allocator = allocator };
}

fn finishDevice(comptime T: type, input: Array(T), device: Device) ArrayError!Array(T) {
    var out = input;
    if (!device.isAvailable()) {
        out.deinit();
        return error.InvalidDevice;
    }
    out.device = device;
    return out;
}

fn optionDType(comptime Options: type) type {
    if (!@hasDecl(Options, "dtype")) @compileError("array creation options must come from vx.options(T), vx.onDevice(T, device), vx.seeded(T, seed), or vx.seededOn(T, device, seed)");
    return Options.dtype;
}

fn lookupSymbol(name: []const u8, bindings: []const DimBinding) ArrayError!usize {
    for (bindings) |binding| {
        if (std.mem.eql(u8, name, binding.name)) return binding.value;
    }
    return error.InvalidShape;
}

fn checkedAdd(lhs: usize, rhs: usize) ArrayError!usize {
    if (lhs > std.math.maxInt(usize) - rhs) return error.InvalidShape;
    return lhs + rhs;
}

fn checkedSub(lhs: usize, rhs: usize) ArrayError!usize {
    if (lhs < rhs) return error.InvalidShape;
    return lhs - rhs;
}

fn checkedMul(lhs: usize, rhs: usize) ArrayError!usize {
    if (lhs != 0 and rhs > std.math.maxInt(usize) / lhs) return error.InvalidShape;
    return lhs * rhs;
}

fn computeRuntimeNumel(comptime rank: usize, shape: [rank]usize) ArrayError!usize {
    var n: usize = 1;
    for (shape) |extent| n = try checkedMul(n, extent);
    return n;
}

fn computeRuntimeStrides(comptime rank: usize, shape: [rank]usize, comptime order: LayoutOrder, out: *[rank]usize) void {
    switch (order) {
        .row_major => {
            var running: usize = 1;
            var i: usize = rank;
            while (i > 0) {
                i -= 1;
                out[i] = running;
                running *= shape[i];
            }
        },
        .column_major => {
            var running: usize = 1;
            for (0..rank) |i| {
                out[i] = running;
                running *= shape[i];
            }
        },
    }
}

fn computeStaticNumel(comptime rank: usize, comptime shape: [rank]usize) usize {
    var n: usize = 1;
    for (shape) |extent| n *= extent;
    return n;
}

fn computeStaticStrides(comptime rank: usize, comptime shape: [rank]usize, comptime order: LayoutOrder) [rank]usize {
    var strides: [rank]usize = undefined;
    switch (order) {
        .row_major => {
            var running: usize = 1;
            var i: usize = rank;
            while (i > 0) {
                i -= 1;
                strides[i] = running;
                running *= shape[i];
            }
        },
        .column_major => {
            var running: usize = 1;
            for (0..rank) |i| {
                strides[i] = running;
                running *= shape[i];
            }
        },
    }
    return strides;
}

test "static layout computes row and column major offsets" {
    const Row = StaticLayout(2, .{ 2, 3 }, .row_major);
    const Col = StaticLayout(2, .{ 2, 3 }, .column_major);
    try std.testing.expectEqualSlices(usize, &.{ 3, 1 }, Row.strides[0..]);
    try std.testing.expectEqualSlices(usize, &.{ 1, 2 }, Col.strides[0..]);
    try std.testing.expectEqual(@as(usize, 5), try Row.offset(.{ 1, 2 }));
    try std.testing.expectEqual(@as(usize, 5), try Col.offset(.{ 1, 2 }));
    try std.testing.expectEqual(@as(usize, 3), try Row.offset(.{ 1, 0 }));
    try std.testing.expectEqual(@as(usize, 1), try Col.offset(.{ 1, 0 }));
}

test "static typed and erased array layers preserve metadata" {
    const gpa = std.testing.allocator;
    const Layout = StaticLayout(2, .{ 2, 3 }, .row_major);
    var static_array = try StaticArray(f32, Layout).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 }, .cpu);
    defer static_array.deinit();
    try std.testing.expectEqual(@as(f32, 6), try static_array.get(.{ 1, 2 }));

    var runtime_array = try static_array.toRuntimeArray();
    defer runtime_array.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, runtime_array.shape);
    try std.testing.expectEqualSlices(usize, &.{ 3, 1 }, runtime_array.strides);

    var erased = try AnyArray.fromStatic(f32, Layout, gpa, static_array);
    defer erased.deinit();
    try std.testing.expectEqual(DType.f32, erased.dtype);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, erased.shape);
    try std.testing.expectEqualSlices(usize, &.{ 3, 1 }, erased.strides);
    try std.testing.expectEqual(@as(usize, 24), erased.byteSize());
}

test "symbolic layout evaluates dimension arithmetic and array metadata" {
    const gpa = std.testing.allocator;
    const M = symbol("M");
    const K = symbol("K");
    const Two = dim(2);
    const One = dim(1);
    const TwiceM = dimMul(M, Two);
    const KPlusOne = dimAdd(K, One);
    const KMinusOne = dimSub(K, One);
    const Layout = SymbolicLayout(2, .{ TwiceM, KPlusOne }, .row_major);
    const bindings = [_]DimBinding{
        .{ .name = "M", .value = 3 },
        .{ .name = "K", .value = 4 },
    };

    var shape: [2]usize = undefined;
    var strides: [2]usize = undefined;
    const rt = try Layout.runtime(bindings[0..], &shape, &strides);
    try std.testing.expectEqualSlices(usize, &.{ 6, 5 }, rt.shape);
    try std.testing.expectEqualSlices(usize, &.{ 5, 1 }, rt.strides);
    try std.testing.expectEqual(@as(usize, 29), try Layout.offset(bindings[0..], .{ 5, 4 }));
    try std.testing.expectEqual(@as(usize, 3), try KMinusOne.eval(bindings[0..]));

    var values: [30]f32 = undefined;
    for (&values, 0..) |*slot, i| slot.* = @floatFromInt(i);
    var array = try SymbolicArray(f32, Layout).fromSlice(gpa, values[0..], bindings[0..], .cpu);
    defer array.deinit();
    try std.testing.expectEqual(@as(f32, 29), try array.get(.{ 5, 4 }));

    var erased = try AnyArray.fromSymbolic(f32, Layout, gpa, array);
    defer erased.deinit();
    try std.testing.expectEqual(DType.f32, erased.dtype);
    try std.testing.expectEqualSlices(usize, &.{ 6, 5 }, erased.shape);
    try std.testing.expectEqualSlices(usize, &.{ 5, 1 }, erased.strides);

    const bad_bindings = [_]DimBinding{.{ .name = "K", .value = 0 }};
    try std.testing.expectError(error.InvalidShape, KMinusOne.eval(bad_bindings[0..]));
}

test "creation options carry dtype and runtime device" {
    const gpa = std.testing.allocator;
    const np = withAllocator(gpa);
    var x = try np.onesWith(options(f64), &.{ 2, 2 });
    defer x.deinit();
    try std.testing.expect(@TypeOf(x) == Array(f64));
    try std.testing.expectEqual(DType.f64, DType.of(@TypeOf(x).Scalar));
    try std.testing.expect(x.device.isCpu());
    try std.testing.expectError(error.InvalidDevice, np.zerosWith(onDevice(f32, Device.cuda(0)), &.{ 2, 2 }));
}
