//! Layered Array abstractions for Vectra.
//!
//! The layers are intentionally explicit:
//! - `StaticArray(T, Layout)`: dtype and layout are comptime type parameters.
//! - `Array(T)`: existing Vectra typed array with runtime shape/strides layout.
//! - `AnyArray`: dtype and layout are runtime metadata.
//!
//! `Device` remains runtime metadata in every layer.  When a CUDA device is
//! available through Axiom, deterministic creation options allocate runtime
//! `Array` payloads directly in device storage. Random device creation uses the
//! same options shape and stays explicit when a device RNG kernel is unavailable.

const std = @import("std");
const array_mod = @import("array.zig");

pub const ArrayError = array_mod.ArrayError;
pub const Array = array_mod.Array;
pub const Device = array_mod.Device;
pub const DType = array_mod.DType;

pub const default_dtype = f32;
pub const default_rng_seed: u64 = 0x0abc_7aaa_51eed001;
pub const rng_seed_stride: u64 = 0x9e37_79b9_7f4a_7c15;

/// Runtime creation options for Array creation helpers.
///
/// This mirrors the PyTorch idea that every creation can state `dtype`, `device`,
/// and reproducibility metadata, while preserving Zig's static dtype specialization:
///
/// ```
/// const opts = vx.options();          // device defaults to CPU
/// const gpu = vx.onDevice(vx.cuda(0));
/// const seeded = vx.seeded(42);       // optional per-call reproducibility
/// var x = try np.zerosWith(f32, &.{2, 3}, gpu);
/// ```
pub const CreationOptions = struct {
    device: Device = .cpu,
    seed: ?u64 = null,
};

pub fn options() CreationOptions {
    return .{};
}

pub fn onDevice(device: Device) CreationOptions {
    return .{ .device = device };
}

pub fn seeded(seed: u64) CreationOptions {
    return .{ .seed = seed };
}

pub fn seededOn(device: Device, seed: u64) CreationOptions {
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
    next_seed: u64 = default_rng_seed,

    pub fn array(self: Context, comptime T: type, values: []const T, dims: []const usize) ArrayError!Array(T) {
        return Array(T).fromSlice(self.allocator, values, dims);
    }

    pub fn arrayWith(
        self: Context,
        comptime T: type,
        values: []const T,
        dims: []const usize,
        opts: CreationOptions,
    ) ArrayError!Array(T) {
        return Array(T).fromSliceOn(self.allocator, values, dims, opts.device);
    }

    pub fn zeros(self: Context, comptime T: type, dims: []const usize) ArrayError!Array(T) {
        return Array(T).zeros(self.allocator, dims);
    }

    pub fn zerosWith(self: Context, comptime T: type, dims: []const usize, opts: CreationOptions) ArrayError!Array(T) {
        return Array(T).zerosOn(self.allocator, dims, opts.device);
    }

    pub fn ones(self: Context, comptime T: type, dims: []const usize) ArrayError!Array(T) {
        return Array(T).ones(self.allocator, dims);
    }

    pub fn onesWith(self: Context, comptime T: type, dims: []const usize, opts: CreationOptions) ArrayError!Array(T) {
        return Array(T).onesOn(self.allocator, dims, opts.device);
    }

    pub fn rand(self: *Context, comptime T: type, dims: []const usize) ArrayError!Array(T) {
        return Array(T).rand(self.allocator, dims, self.nextSeed());
    }

    pub fn randWith(self: *Context, comptime T: type, dims: []const usize, opts: CreationOptions) ArrayError!Array(T) {
        const seed = opts.seed orelse self.nextSeed();
        return Array(T).randOn(self.allocator, dims, seed, opts.device);
    }

    pub fn randn(self: *Context, comptime T: type, dims: []const usize) ArrayError!Array(T) {
        return Array(T).randn(self.allocator, dims, self.nextSeed());
    }

    pub fn randnWith(self: *Context, comptime T: type, dims: []const usize, opts: CreationOptions) ArrayError!Array(T) {
        const seed = opts.seed orelse self.nextSeed();
        return Array(T).randnOn(self.allocator, dims, seed, opts.device);
    }

    pub fn normal(self: *Context, comptime T: type, dims: []const usize, mean_value: T, stddev_value: T) ArrayError!Array(T) {
        return Array(T).normal(self.allocator, dims, mean_value, stddev_value, self.nextSeed());
    }

    pub fn normalWith(self: *Context, comptime T: type, dims: []const usize, mean_value: T, stddev_value: T, opts: CreationOptions) ArrayError!Array(T) {
        const seed = opts.seed orelse self.nextSeed();
        return Array(T).normalOn(self.allocator, dims, mean_value, stddev_value, seed, opts.device);
    }

    pub fn randSeeded(self: Context, comptime T: type, dims: []const usize, seed: u64) ArrayError!Array(T) {
        return Array(T).rand(self.allocator, dims, seed);
    }

    fn nextSeed(self: *Context) u64 {
        const seed = self.next_seed;
        self.next_seed +%= rng_seed_stride;
        return seed;
    }

    pub fn anyFromTyped(self: Context, comptime T: type, input: Array(T)) ArrayError!AnyArray {
        return AnyArray.fromTyped(T, self.allocator, input);
    }
};

pub fn withAllocator(allocator: std.mem.Allocator) Context {
    return .{ .allocator = allocator };
}

pub fn withSeed(allocator: std.mem.Allocator, seed: u64) Context {
    return .{ .allocator = allocator, .next_seed = seed };
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
    var x = try np.onesWith(f64, &.{ 2, 2 }, options());
    defer x.deinit();
    try std.testing.expect(@TypeOf(x) == Array(f64));
    try std.testing.expectEqual(DType.f64, DType.of(@TypeOf(x).Scalar));
    try std.testing.expect(x.device.isCpu());
    if (Device.cuda(0).isAvailable()) {
        var gpu = try np.zerosWith(f32, &.{ 2, 2 }, onDevice(Device.cuda(0)));
        defer gpu.deinit();
        try std.testing.expect(gpu.device.isCuda());
    } else {
        try std.testing.expectError(error.InvalidDevice, np.zerosWith(f32, &.{ 2, 2 }, onDevice(Device.cuda(0))));
    }
}

test "random creation options keep device explicit" {
    const gpa = std.testing.allocator;
    var np = withAllocator(gpa);
    const opts = seeded(1234);
    try std.testing.expect(@TypeOf(opts) == CreationOptions);

    var first = try np.randWith(f32, &.{4}, opts);
    defer first.deinit();
    var second = try np.randWith(f32, &.{4}, seeded(1234));
    defer second.deinit();

    try std.testing.expect(first.device.isCpu());
    try std.testing.expectEqualSlices(f32, first.data, second.data);

    const device_opts = seededOn(Device.cuda(0), 1234);
    try std.testing.expect(@TypeOf(device_opts) == CreationOptions);
    try std.testing.expect(device_opts.device.isCuda());
    if (device_opts.device.isAvailable()) {
        var cuda_random = try np.randWith(f32, &.{4}, device_opts);
        defer cuda_random.deinit();
        try std.testing.expect(cuda_random.device.isCuda());
        try std.testing.expect(cuda_random.device_storage != null);
        var cuda_random_f64 = try np.randWith(f64, &.{4}, device_opts);
        defer cuda_random_f64.deinit();
        var cuda_random_f64_back = try cuda_random_f64.cpu();
        defer cuda_random_f64_back.deinit();
        for (cuda_random_f64_back.data) |value| try std.testing.expect(value >= 0 and value < 1);
        var cuda_random_f16 = try np.randWith(f16, &.{4}, device_opts);
        defer cuda_random_f16.deinit();
        var cuda_random_f16_back = try cuda_random_f16.cpu();
        defer cuda_random_f16_back.deinit();
        for (cuda_random_f16_back.data) |value| try std.testing.expect(@as(f32, value) >= 0 and @as(f32, value) < 1);
        var cuda_random_bf16 = try np.randWith(array_mod.BFloat16, &.{4}, device_opts);
        defer cuda_random_bf16.deinit();
        var cuda_random_bf16_back = try cuda_random_bf16.cpu();
        defer cuda_random_bf16_back.deinit();
        for (cuda_random_bf16_back.data) |value| try std.testing.expect(value.toF32() >= 0 and value.toF32() < 1);
    } else {
        try std.testing.expectError(error.InvalidDevice, np.randWith(f32, &.{4}, device_opts));
    }
    const mps_opts = seededOn(Device.mps(0), 1234);
    if (mps_opts.device.isAvailable()) {
        var mps_random = try np.randWith(f32, &.{4}, mps_opts);
        defer mps_random.deinit();
        try std.testing.expect(mps_random.device.isMps());
        try std.testing.expect(mps_random.device_storage != null);
        var mps_random_f16 = try np.randWith(f16, &.{4}, mps_opts);
        defer mps_random_f16.deinit();
        try std.testing.expect(mps_random_f16.device.isMps());
        try std.testing.expect(mps_random_f16.device_storage != null);
        var mps_random_bf16 = try np.randWith(array_mod.BFloat16, &.{4}, mps_opts);
        defer mps_random_bf16.deinit();
        try std.testing.expect(mps_random_bf16.device.isMps());
        try std.testing.expect(mps_random_bf16.device_storage != null);
        var mps_normal = try np.normalWith(f32, &.{4}, 10, 0, mps_opts);
        defer mps_normal.deinit();
        try std.testing.expect(mps_normal.device.isMps());
        try std.testing.expect(mps_normal.device_storage != null);
        var mps_normal_back = try mps_normal.cpu();
        defer mps_normal_back.deinit();
        try std.testing.expectEqualSlices(f32, &.{ 10, 10, 10, 10 }, mps_normal_back.data);
        var mps_normal_f16 = try np.normalWith(f16, &.{4}, 10, 0, mps_opts);
        defer mps_normal_f16.deinit();
        try std.testing.expect(mps_normal_f16.device.isMps());
        try std.testing.expect(mps_normal_f16.device_storage != null);
        var mps_normal_f16_back = try mps_normal_f16.cpu();
        defer mps_normal_f16_back.deinit();
        try std.testing.expectEqualSlices(f16, &.{ 10, 10, 10, 10 }, mps_normal_f16_back.data);
        var mps_normal_bf16 = try np.normalWith(array_mod.BFloat16, &.{4}, array_mod.BFloat16.fromF32(10), array_mod.BFloat16.fromF32(0), mps_opts);
        defer mps_normal_bf16.deinit();
        try std.testing.expect(mps_normal_bf16.device.isMps());
        try std.testing.expect(mps_normal_bf16.device_storage != null);
        var mps_normal_bf16_back = try mps_normal_bf16.cpu();
        defer mps_normal_bf16_back.deinit();
        for (mps_normal_bf16_back.data) |value| try std.testing.expectApproxEqAbs(@as(f32, 10), value.toF32(), 0.125);
    }
    try std.testing.expectError(error.TypeUnsupported, np.randWith(f64, &.{4}, mps_opts));
    try std.testing.expectError(error.TypeUnsupported, np.normalWith(f64, &.{4}, 0, 1, mps_opts));
}
