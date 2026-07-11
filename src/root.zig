//! Vectra is an experimental Zig-native data processing and numerical computing
//! toolkit inspired by NumPy/CuPy/SciPy/Pandas/Polars, with PyTorch-style object
//! methods as the primary public array API.
//!
//! The current implementation provides a compact CPU core: typed arrays,
//! broadcasting arithmetic, reductions, linalg/stat helpers, typed Series,
//! heterogeneous DataFrame operations, sparse matrices, and CSV IO. CUDA/GPU is
//! available through the default Axiom backend when a CUDA device is
//! present; CUDA arrays own device-resident storage and dispatch supported f32/f64
//! operations without staging through host arrays.

const array_mod = @import("array.zig");
const series_mod = @import("series.zig");
const dataframe_mod = @import("dataframe.zig");
const layered_array_mod = @import("layered_array.zig");
pub const linalg = @import("linalg.zig");
pub const stats = @import("stats.zig");
pub const sparse = @import("sparse.zig");
pub const axiom_cuda = @import("backends/axiom_cuda.zig");

pub const Array = array_mod.Array;
pub const NDArray = array_mod.NDArray;
pub const ArrayView = array_mod.ArrayView;
pub const NDArrayView = array_mod.NDArrayView;
pub const BFloat16 = array_mod.BFloat16;
pub const Complex64 = array_mod.Complex64;
pub const Complex128 = array_mod.Complex128;
pub const Device = array_mod.Device;
pub const DType = array_mod.DType;
pub const canCastDType = array_mod.canCastDType;
pub const promoteDType = array_mod.promoteDType;
pub const resultDType = array_mod.resultDType;
pub const promoteType = array_mod.promoteType;
pub const Slice = array_mod.Slice;
pub const ScatterReduce = array_mod.ScatterReduce;
pub const SearchSide = array_mod.SearchSide;
pub const IndexMode = array_mod.IndexMode;
pub const MeshGridIndexing = array_mod.MeshGridIndexing;
pub const ConvMode = array_mod.ConvMode;
pub const ArrayError = array_mod.ArrayError;
pub const axiom_cpu = @import("backends/axiom_cpu.zig");
pub const axiom_backend = @import("backends/axiom_backend.zig");

pub const Series = series_mod.Series;
pub const DataFrame = dataframe_mod.DataFrame;
pub const Column = dataframe_mod.Column;
pub const ColumnDef = dataframe_mod.ColumnDef;
pub const DataError = dataframe_mod.DataError;

pub const CsrMatrix = sparse.CsrMatrix;
pub const CscMatrix = sparse.CscMatrix;
pub const csrFromDense = sparse.csrFromDense;
pub const csrFromCompressed = sparse.csrFromCompressed;
pub const cscFromDense = sparse.cscFromDense;
pub const cscFromCompressed = sparse.cscFromCompressed;

pub const MatrixNormOrder = array_mod.MatrixNormOrder;
pub const Triangle = array_mod.Triangle;
pub const Diagonal = array_mod.Diagonal;
pub const QrResult = array_mod.QrResult;
pub const SvdResult = array_mod.SvdResult;
pub const EighResult = array_mod.EighResult;
pub const LuResult = array_mod.LuResult;

pub const LayoutOrder = layered_array_mod.LayoutOrder;
pub const RuntimeLayout = layered_array_mod.RuntimeLayout;
pub const StaticLayout = layered_array_mod.StaticLayout;
pub const StaticArray = layered_array_mod.StaticArray;
pub const SymbolicLayout = layered_array_mod.SymbolicLayout;
pub const SymbolicArray = layered_array_mod.SymbolicArray;
pub const DimExpr = layered_array_mod.DimExpr;
pub const DimBinding = layered_array_mod.DimBinding;
pub const dim = layered_array_mod.dim;
pub const symbol = layered_array_mod.symbol;
pub const dimAdd = layered_array_mod.dimAdd;
pub const dimSub = layered_array_mod.dimSub;
pub const dimMul = layered_array_mod.dimMul;
pub const AnyArray = layered_array_mod.AnyArray;
pub const CreationOptions = layered_array_mod.CreationOptions;
pub const Context = layered_array_mod.Context;
pub const cpu = Device.cpu;
pub const options = layered_array_mod.options;
pub const onDevice = layered_array_mod.onDevice;
pub const seeded = layered_array_mod.seeded;
pub const seededOn = layered_array_mod.seededOn;

pub fn cuda(index: usize) Device {
    return Device.cuda(index);
}

pub fn withAllocator(allocator: @import("std").mem.Allocator) Context {
    return layered_array_mod.withAllocator(allocator);
}

pub fn withSeed(allocator: @import("std").mem.Allocator, seed: u64) Context {
    return layered_array_mod.withSeed(allocator, seed);
}

pub fn add(lhs: anytype, rhs: @TypeOf(lhs)) ArrayError!@TypeOf(lhs) {
    try requireSameDevice(lhs, rhs);
    return switch (lhs.device.backend) {
        .cpu => lhs.add(rhs),
        .cuda => lhs.add(rhs),
    };
}

pub fn matmul(lhs: anytype, rhs: @TypeOf(lhs)) ArrayError!@TypeOf(lhs) {
    try requireSameDevice(lhs, rhs);
    return switch (lhs.device.backend) {
        .cpu => lhs.matmul(rhs),
        .cuda => lhs.matmul(rhs),
    };
}

pub fn matmulAdd(lhs: anytype, rhs: @TypeOf(lhs), addend: @TypeOf(lhs)) ArrayError!@TypeOf(lhs) {
    try requireSameDevice(lhs, rhs);
    try requireSameDevice(lhs, addend);
    if (lhs.device.isCuda()) {
        var product = try lhs.matmul(rhs);
        defer product.deinit();
        return product.add(addend);
    }
    if (comptime @TypeOf(lhs) == Array(f32)) {
        if (try tryCpuMatmulAddF32(@as(Array(f32), lhs), @as(Array(f32), rhs), @as(Array(f32), addend))) |out| {
            return @as(@TypeOf(lhs), out);
        }
        if (try tryCudaMatmulAddF32(@as(Array(f32), lhs), @as(Array(f32), rhs), @as(Array(f32), addend))) |out| {
            return @as(@TypeOf(lhs), out);
        }
    } else if (comptime @TypeOf(lhs) == Array(f64)) {
        if (try tryCpuMatmulAddF64(@as(Array(f64), lhs), @as(Array(f64), rhs), @as(Array(f64), addend))) |out| {
            return @as(@TypeOf(lhs), out);
        }
        if (try tryCudaMatmulAddF64(@as(Array(f64), lhs), @as(Array(f64), rhs), @as(Array(f64), addend))) |out| {
            return @as(@TypeOf(lhs), out);
        }
    } else if (comptime @TypeOf(lhs) == Array(f16)) {
        if (try tryCudaMatmulAddF16(@as(Array(f16), lhs), @as(Array(f16), rhs), @as(Array(f16), addend))) |out| {
            return @as(@TypeOf(lhs), out);
        }
    } else if (comptime @TypeOf(lhs) == Array(BFloat16)) {
        if (try tryCudaMatmulAddBF16(@as(Array(BFloat16), lhs), @as(Array(BFloat16), rhs), @as(Array(BFloat16), addend))) |out| {
            return @as(@TypeOf(lhs), out);
        }
    }
    var product = try matmul(lhs, rhs);
    defer product.deinit();
    return add(product, addend);
}

pub fn tryCpuMatmulAddF32(lhs: Array(f32), rhs: Array(f32), addend: Array(f32)) ArrayError!?Array(f32) {
    try requireSameDevice(lhs, rhs);
    try requireSameDevice(lhs, addend);
    if (!lhs.device.isCpu()) return null;
    if (try axiom_cpu.tryMatmulAddF32(lhs, rhs, addend)) |out| return out;
    return null;
}

pub fn tryCpuMatmulAddF64(lhs: Array(f64), rhs: Array(f64), addend: Array(f64)) ArrayError!?Array(f64) {
    try requireSameDevice(lhs, rhs);
    try requireSameDevice(lhs, addend);
    if (!lhs.device.isCpu()) return null;
    if (try axiom_cpu.tryMatmulAddF64(lhs, rhs, addend)) |out| return out;
    return null;
}

pub fn tryCudaMatmulAddF32(lhs: Array(f32), rhs: Array(f32), addend: Array(f32)) ArrayError!?Array(f32) {
    try requireSameDevice(lhs, rhs);
    try requireSameDevice(lhs, addend);
    if (!lhs.device.isCuda()) return null;
    if (try axiom_cuda.tryDeviceMatmulAddF32(lhs, rhs, addend)) |out| return out;
    return null;
}

pub fn tryCudaMatmulAddF64(lhs: Array(f64), rhs: Array(f64), addend: Array(f64)) ArrayError!?Array(f64) {
    try requireSameDevice(lhs, rhs);
    try requireSameDevice(lhs, addend);
    if (!lhs.device.isCuda()) return null;
    if (try axiom_cuda.tryDeviceMatmulAddF64(lhs, rhs, addend)) |out| return out;
    return null;
}

pub fn tryCudaMatmulAddF16(lhs: Array(f16), rhs: Array(f16), addend: Array(f16)) ArrayError!?Array(f16) {
    try requireSameDevice(lhs, rhs);
    try requireSameDevice(lhs, addend);
    if (!lhs.device.isCuda()) return null;
    if (try axiom_cuda.tryDeviceMatmulAddF16(lhs, rhs, addend)) |out| return out;
    return null;
}

pub fn tryCudaMatmulAddBF16(lhs: Array(BFloat16), rhs: Array(BFloat16), addend: Array(BFloat16)) ArrayError!?Array(BFloat16) {
    try requireSameDevice(lhs, rhs);
    try requireSameDevice(lhs, addend);
    if (!lhs.device.isCuda()) return null;
    if (try axiom_cuda.tryDeviceMatmulAddBF16(lhs, rhs, addend)) |out| return out;
    return null;
}

fn requireSameDevice(lhs: anytype, rhs: @TypeOf(lhs)) ArrayError!void {
    if (!lhs.device.sameDevice(rhs.device)) return error.InvalidDevice;
    if (!lhs.device.isAvailable()) return error.InvalidDevice;
}

pub fn sum(input: anytype, axis_opt: ?isize, keepdims: bool) ArrayError!@TypeOf(input) {
    return input.sum(axis_opt, keepdims);
}

pub fn mulScalar(input: anytype, scalar: @TypeOf(input).Scalar) ArrayError!@TypeOf(input) {
    return input.mulScalar(scalar);
}

test "top-level ops respect device dispatch" {
    const gpa = @import("std").testing.allocator;
    var lhs = try Array(f32).fromSlice(gpa, &.{ 1, 2, 3, 4 }, &.{ 2, 2 });
    defer lhs.deinit();
    var rhs = try Array(f32).fromSlice(gpa, &.{ 10, 20, 30, 40 }, &.{ 2, 2 });
    defer rhs.deinit();

    var sum_cpu = try add(lhs, rhs);
    defer sum_cpu.deinit();
    try @import("std").testing.expect(sum_cpu.device.isCpu());
    try @import("std").testing.expectEqualSlices(f32, &.{ 11, 22, 33, 44 }, sum_cpu.data);

    var cuda_rhs = try rhs.clone();
    defer cuda_rhs.deinit();
    cuda_rhs.device = Device.cuda(0);
    try @import("std").testing.expectError(error.InvalidDevice, add(lhs, cuda_rhs));
}

test {
    _ = array_mod;
    _ = axiom_cpu;
    _ = axiom_backend;
    _ = series_mod;
    _ = dataframe_mod;
    _ = layered_array_mod;
    _ = linalg;
    _ = stats;
    _ = sparse;
    _ = axiom_cuda;
}
