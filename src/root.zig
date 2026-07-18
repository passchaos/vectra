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
const forge_interop_mod = @import("forge_interop.zig");
pub const linalg = @import("linalg.zig");
pub const stats = @import("stats.zig");
pub const sparse = @import("sparse.zig");
pub const axiom_cuda = @import("backends/axiom_cuda.zig");
pub const forge_interop = forge_interop_mod;
pub const ForgeInteropBoundary = forge_interop_mod.InteropBoundary;
pub const forgeArrayInteropBoundary = forge_interop_mod.forge_array_interop_boundary;
pub const forgeInteropBoundary = forge_interop_mod.forgeInteropBoundary;

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
pub const LossReduction = array_mod.LossReduction;
pub const ArrayError = array_mod.ArrayError;
pub const axiom_cpu = @import("backends/axiom_cpu.zig");
pub const axiom_backend = @import("backends/axiom_backend.zig");
pub const DialectBackend = axiom_backend.DialectBackend;
pub const setDefaultDialectBackend = axiom_backend.setDefaultDialectBackend;
pub const defaultDialectBackend = axiom_backend.defaultDialectBackend;
pub const defaultExecutionTarget = axiom_backend.defaultExecutionTarget;
pub const executionTargetForDevice = axiom_backend.executionTargetForDevice;
pub const resetDefaultDialectBackend = axiom_backend.resetDefaultDialectBackend;
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

pub fn mps(index: usize) Device {
    return Device.mps(index);
}

pub fn withAllocator(allocator: @import("std").mem.Allocator) Context {
    return layered_array_mod.withAllocator(allocator);
}

pub fn withSeed(allocator: @import("std").mem.Allocator, seed: u64) Context {
    return layered_array_mod.withSeed(allocator, seed);
}

pub fn add(lhs: anytype, rhs: @TypeOf(lhs)) ArrayError!@TypeOf(lhs) {
    try requireSameDevice(lhs, rhs);
    return lhs.add(rhs);
}

pub fn matmul(lhs: anytype, rhs: @TypeOf(lhs)) ArrayError!@TypeOf(lhs) {
    try requireSameDevice(lhs, rhs);
    return lhs.matmul(rhs);
}

pub fn matmulAdd(lhs: anytype, rhs: @TypeOf(lhs), addend: @TypeOf(lhs)) ArrayError!@TypeOf(lhs) {
    try requireSameDevice(lhs, rhs);
    try requireSameDevice(lhs, addend);
    return lhs.matmulAdd(rhs, addend);
}

pub fn einsum(subscripts: []const u8, lhs: anytype, rhs: @TypeOf(lhs)) ArrayError!@TypeOf(lhs) {
    try requireSameDevice(lhs, rhs);
    // Bounded NumPy/PyTorch-style front-end syntax over existing Array
    // primitives.  This deliberately keeps execution routed through Array/Axiom
    // matmul/dot/outer paths instead of adding a separate backend branch here.
    if (@import("std").mem.eql(u8, subscripts, "ij,jk->ik")) return lhs.matmul(rhs);
    if (@import("std").mem.eql(u8, subscripts, "ab,bc->ac")) return lhs.matmul(rhs);
    if (@import("std").mem.eql(u8, subscripts, "i,i->")) return lhs.dot(rhs);
    if (@import("std").mem.eql(u8, subscripts, "i,j->ij")) return lhs.outer(rhs);
    if (@import("std").mem.eql(u8, subscripts, "ij,j->i")) return lhs.matmul(rhs);
    return error.InvalidShape;
}

pub fn tryMatmulAddTarget(target: DialectBackend, lhs: anytype, rhs: @TypeOf(lhs), addend: @TypeOf(lhs)) ArrayError!?@TypeOf(lhs) {
    try requireSameDevice(lhs, rhs);
    try requireSameDevice(lhs, addend);
    return axiom_backend.executeMatmulAdd(@TypeOf(lhs).Scalar, target, lhs, rhs, addend);
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

    if (try tryMatmulAddTarget(.cpu, lhs, rhs, sum_cpu)) |target_add_value| {
        var target_add = target_add_value;
        defer target_add.deinit();
        try @import("std").testing.expect(target_add.device.isCpu());
        try @import("std").testing.expectEqualSlices(f32, &.{ 81, 122, 183, 264 }, target_add.data);
    }

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
    _ = forge_interop_mod;
    _ = linalg;
    _ = stats;
    _ = sparse;
    _ = axiom_cuda;
}
