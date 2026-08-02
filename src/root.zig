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
//! `DeviceDataFrame` extends the same device model to fixed-width tabular
//! columns on CPU/CUDA/MPS while the existing heterogeneous `DataFrame` remains
//! the compact host/CSV-oriented API.
//! Public backend diagnostics intentionally flow through `axiom_backend` so
//! callers choose targets and inspect capability reports instead of binding to a
//! target-specific bridge module.

const std = @import("std");
const build_options = @import("vectra_build_options");
const array_mod = @import("array.zig");
const series_mod = @import("series.zig");
const dataframe_mod = if (build_options.enable_boltha) @import("dataframe.zig") else @import("dataframe_no_boltha.zig");
const layered_array_mod = @import("layered_array.zig");
const forge_interop_mod = @import("forge_interop.zig");
pub const linalg = @import("linalg.zig");
pub const stats = @import("stats.zig");
pub const sparse = @import("sparse.zig");
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
pub const axiom_backend = @import("backends/axiom_backend.zig");
pub const DialectBackend = axiom_backend.DialectBackend;
pub const PreparedF32Matmul = axiom_backend.PreparedF32Matmul;
pub const PreparedF32TransposedMatmul = axiom_backend.PreparedF32TransposedMatmul;
pub const PreparedF64Matmul = axiom_backend.PreparedF64Matmul;
pub const PreparedF64TransposedMatmul = axiom_backend.PreparedF64TransposedMatmul;
pub const setDefaultDialectBackend = axiom_backend.setDefaultDialectBackend;
pub const defaultDialectBackend = axiom_backend.defaultDialectBackend;
pub const defaultExecutionTarget = axiom_backend.defaultExecutionTarget;
pub const executionTargetForDevice = axiom_backend.executionTargetForDevice;
pub const resetDefaultDialectBackend = axiom_backend.resetDefaultDialectBackend;
pub const cpuMatmulColumnMajorResult = axiom_backend.cpuMatmulColumnMajorResult;
pub const Series = series_mod.Series;
pub const DataFrame = dataframe_mod.DataFrame;
pub const Column = dataframe_mod.Column;
pub const ColumnDef = dataframe_mod.ColumnDef;
pub const DataError = dataframe_mod.DataError;
pub const DeviceDataFrame = dataframe_mod.DeviceDataFrame;
pub const DeviceLazyFrame = dataframe_mod.DeviceLazyFrame;
pub const DeviceLazyOp = dataframe_mod.DeviceLazyOp;
pub const DeviceLazySource = dataframe_mod.DeviceLazySource;
pub const DeviceLazyGroupByAggregation = dataframe_mod.DeviceLazyGroupByAggregation;
pub const DeviceLazyWeightedGroupByAggregation = dataframe_mod.DeviceLazyWeightedGroupByAggregation;
pub const DeviceLazyWeightedPairGroupByAggregation = dataframe_mod.DeviceLazyWeightedPairGroupByAggregation;
pub const DeviceLazyJoinKind = dataframe_mod.DeviceLazyJoinKind;
pub const DeviceParquetScan = dataframe_mod.DeviceParquetScan;
pub const DeviceParquetRangeFilter = dataframe_mod.DeviceParquetRangeFilter;
pub const ParquetRangePredicate = dataframe_mod.ParquetRangePredicate;
pub const DeviceDataFrameView = dataframe_mod.DeviceDataFrameView;
pub const DeviceColumn = dataframe_mod.DeviceColumn;
pub const DeviceColumnView = dataframe_mod.DeviceColumnView;
pub const DeviceColumnDef = dataframe_mod.DeviceColumnDef;
pub const DeviceTypedColumn = dataframe_mod.DeviceTypedColumn;
pub const DeviceDataError = dataframe_mod.DeviceDataError;
pub const ArrowInteropError = dataframe_mod.ArrowInteropError;
pub const ParquetInteropError = dataframe_mod.ParquetInteropError;
pub const DeviceDType = dataframe_mod.DeviceDType;
pub const DeviceDTypeClass = dataframe_mod.DeviceDTypeClass;
pub const DeviceScalar = dataframe_mod.DeviceScalar;
pub const Range = dataframe_mod.Range;
pub const DeviceColumnBinaryOp = dataframe_mod.DeviceColumnBinaryOp;
pub const DeviceColumnCompareOp = dataframe_mod.DeviceColumnCompareOp;
pub const DeviceColumnLogicalOp = dataframe_mod.DeviceColumnLogicalOp;
pub const DeviceGroupByAggregation = dataframe_mod.DeviceGroupByAggregation;
pub const DeviceSortOptions = dataframe_mod.DeviceSortOptions;
pub const DeviceClipOptions = dataframe_mod.DeviceClipOptions;
pub const DeviceThresholdOptions = dataframe_mod.DeviceThresholdOptions;
pub const DeviceRollingOptions = dataframe_mod.DeviceRollingOptions;
pub const DeviceLagOptions = dataframe_mod.DeviceLagOptions;
pub const DeviceExpandingOptions = dataframe_mod.DeviceExpandingOptions;
pub const DeviceExpandingRankOptions = dataframe_mod.DeviceExpandingRankOptions;
pub const DeviceStandardizeOptions = dataframe_mod.DeviceStandardizeOptions;
pub const DeviceRobustOptions = dataframe_mod.DeviceRobustOptions;
pub const DeviceDrawdownOptions = dataframe_mod.DeviceDrawdownOptions;
pub const DeviceExtremaOptions = dataframe_mod.DeviceExtremaOptions;
pub const DeviceTrendOptions = dataframe_mod.DeviceTrendOptions;
pub const DeviceCrossoverOptions = dataframe_mod.DeviceCrossoverOptions;
pub const DeviceBucketOptions = dataframe_mod.DeviceBucketOptions;
pub const DeviceEmaOptions = dataframe_mod.DeviceEmaOptions;
pub const DeviceLinearFitOptions = dataframe_mod.DeviceLinearFitOptions;
pub const DeviceRollingCorrelationOptions = dataframe_mod.DeviceRollingCorrelationOptions;
pub const DeviceRollingRankOptions = dataframe_mod.DeviceRollingRankOptions;
pub const DeviceRollingRobustOptions = dataframe_mod.DeviceRollingRobustOptions;
pub const NullPlacement = dataframe_mod.NullPlacement;
pub const DeviceJoinOptions = dataframe_mod.DeviceJoinOptions;
pub const AsofStrategy = dataframe_mod.AsofStrategy;
pub const DeviceAsofOptions = dataframe_mod.DeviceAsofOptions;
pub const DeviceValidityEncoding = dataframe_mod.DeviceValidityEncoding;

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

pub fn matmulOut(lhs: anytype, rhs: @TypeOf(lhs), out: @TypeOf(lhs)) ArrayError!void {
    try requireSameDevice(lhs, rhs);
    try requireSameDevice(lhs, out);
    return lhs.matmulOut(rhs, out);
}

pub fn matmulAddOut(lhs: anytype, rhs: @TypeOf(lhs), addend: @TypeOf(lhs), out: @TypeOf(lhs)) ArrayError!void {
    try requireSameDevice(lhs, rhs);
    try requireSameDevice(lhs, addend);
    try requireSameDevice(lhs, out);
    return lhs.matmulAddOut(rhs, addend, out);
}

pub fn matmulAdd(lhs: anytype, rhs: @TypeOf(lhs), addend: @TypeOf(lhs)) ArrayError!@TypeOf(lhs) {
    try requireSameDevice(lhs, rhs);
    try requireSameDevice(lhs, addend);
    return lhs.matmulAdd(rhs, addend);
}

pub fn matmulAddSqrt(lhs: anytype, rhs: @TypeOf(lhs), addend: @TypeOf(lhs)) ArrayError!@TypeOf(lhs) {
    try requireSameDevice(lhs, rhs);
    try requireSameDevice(lhs, addend);
    return lhs.matmulAddSqrt(rhs, addend);
}

pub fn einsum(subscripts: []const u8, lhs: anytype, rhs: @TypeOf(lhs)) ArrayError!@TypeOf(lhs) {
    try requireSameDevice(lhs, rhs);
    // Bounded NumPy/PyTorch-style front-end syntax over existing Array
    // primitives.  This parser intentionally supports a bounded binary subset
    // rather than full NumPy syntax.  The dedicated fast paths handle common
    // batched matmul forms, including the PyTorch/NumPy spelling
    // `...ij,...jk->...ik`, by forwarding to Array.matmul so backend selection
    // still flows through Axiom instead of a special einsum backend.
    if (ellipsisBatchedMatmulLikeSubscripts(subscripts, lhs.shape.len, rhs.shape.len)) return lhs.matmul(rhs);
    if (ellipsisBatchedMatvecLikeSubscripts(subscripts, lhs.shape.len, rhs.shape.len)) {
        var rhs_expanded = try rhs.unsqueeze(-2);
        defer rhs_expanded.deinit();
        var product = try lhs.mul(rhs_expanded);
        defer product.deinit();
        return product.sum(-1, false);
    }
    if (ellipsisBatchedVecmatLikeSubscripts(subscripts, lhs.shape.len, rhs.shape.len)) {
        var lhs_expanded = try lhs.unsqueeze(-1);
        defer lhs_expanded.deinit();
        var product = try lhs_expanded.mul(rhs);
        defer product.deinit();
        return product.sum(-2, false);
    }
    if (ellipsisBatchedDotLikeSubscripts(subscripts, lhs.shape.len, rhs.shape.len)) {
        var product = try lhs.mul(rhs);
        defer product.deinit();
        return product.sum(-1, false);
    }
    if (batchedMatmulLikeSubscripts(subscripts, lhs.shape.len, rhs.shape.len)) return lhs.matmul(rhs);
    const plan = try parseBinaryEinsum(subscripts, lhs.shape.len, rhs.shape.len);
    if (plan.matmulLike()) return lhs.matmul(rhs);
    if (plan.matvecLike()) return lhs.matmul(rhs);
    if (plan.vecmatLike()) return lhs.matmul(rhs);
    if (plan.dotLike()) return lhs.dot(rhs);
    if (plan.outerLike()) return lhs.outer(rhs);

    var contracted = try lhs.contractAxes(rhs, plan.lhsAxes(), plan.rhsAxes());
    errdefer contracted.deinit();
    if (plan.outputIsDefault()) return contracted;
    const permuted = try contracted.permute(plan.permuteAxes());
    contracted.deinit();
    return permuted;
}

pub fn tensordot(lhs: anytype, rhs: @TypeOf(lhs), axes_lhs: []const usize, axes_rhs: []const usize) ArrayError!@TypeOf(lhs) {
    try requireSameDevice(lhs, rhs);
    return lhs.contractAxes(rhs, axes_lhs, axes_rhs);
}

pub fn tensorDot(lhs: anytype, rhs: @TypeOf(lhs), axes_lhs: []const usize, axes_rhs: []const usize) ArrayError!@TypeOf(lhs) {
    return tensordot(lhs, rhs, axes_lhs, axes_rhs);
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

fn batchedMatmulLikeSubscripts(subscripts: []const u8, lhs_rank: usize, rhs_rank: usize) bool {
    if (lhs_rank != 3 or rhs_rank != 3) return false;
    const arrow = std.mem.indexOf(u8, subscripts, "->") orelse subscripts.len;
    const comma = std.mem.indexOfScalar(u8, subscripts[0..arrow], ',') orelse return false;
    const lhs = subscripts[0..comma];
    const rhs = subscripts[comma + 1 .. arrow];
    const out = if (arrow == subscripts.len) "" else subscripts[arrow + 2 ..];
    if (lhs.len != 3 or rhs.len != 3) return false;
    if (out.len != 0 and out.len != 3) return false;
    if (!allEinsumLabels(lhs) or !allEinsumLabels(rhs) or !allEinsumLabels(out)) return false;
    if (hasRepeatedLabels(lhs) or hasRepeatedLabels(rhs) or hasRepeatedLabels(out)) return false;
    if (out.len == 0) return lhs[0] == rhs[0] and lhs[2] == rhs[1];
    return lhs[0] == rhs[0] and
        lhs[0] == out[0] and
        lhs[1] == out[1] and
        rhs[2] == out[2] and
        lhs[2] == rhs[1];
}

fn ellipsisBatchedMatmulLikeSubscripts(subscripts: []const u8, lhs_rank: usize, rhs_rank: usize) bool {
    if (lhs_rank < 2 or rhs_rank < 2) return false;
    const arrow = std.mem.indexOf(u8, subscripts, "->") orelse subscripts.len;
    const comma = std.mem.indexOfScalar(u8, subscripts[0..arrow], ',') orelse return false;
    const lhs = subscripts[0..comma];
    const rhs = subscripts[comma + 1 .. arrow];
    const out = if (arrow == subscripts.len) "" else subscripts[arrow + 2 ..];
    if (!std.mem.startsWith(u8, lhs, "...") or !std.mem.startsWith(u8, rhs, "...")) return false;
    const lhs_tail = lhs[3..];
    const rhs_tail = rhs[3..];
    if (lhs_tail.len != 2 or rhs_tail.len != 2) return false;
    if (!allEinsumLabels(lhs_tail) or !allEinsumLabels(rhs_tail)) return false;
    if (hasRepeatedLabels(lhs_tail) or hasRepeatedLabels(rhs_tail)) return false;
    if (lhs_tail[1] != rhs_tail[0]) return false;
    if (out.len == 0) return true;
    if (!std.mem.startsWith(u8, out, "...")) return false;
    const out_tail = out[3..];
    return out_tail.len == 2 and
        allEinsumLabels(out_tail) and
        !hasRepeatedLabels(out_tail) and
        out_tail[0] == lhs_tail[0] and
        out_tail[1] == rhs_tail[1];
}

fn ellipsisBatchedMatvecLikeSubscripts(subscripts: []const u8, lhs_rank: usize, rhs_rank: usize) bool {
    if (lhs_rank < 1 or rhs_rank < 1) return false;
    const arrow = std.mem.indexOf(u8, subscripts, "->") orelse subscripts.len;
    const comma = std.mem.indexOfScalar(u8, subscripts[0..arrow], ',') orelse return false;
    const lhs = subscripts[0..comma];
    const rhs = subscripts[comma + 1 .. arrow];
    const out = if (arrow == subscripts.len) "" else subscripts[arrow + 2 ..];
    if (!std.mem.startsWith(u8, lhs, "...") or !std.mem.startsWith(u8, rhs, "...")) return false;
    const lhs_tail = lhs[3..];
    const rhs_tail = rhs[3..];
    if (!allEinsumLabels(lhs_tail) or !allEinsumLabels(rhs_tail)) return false;
    if (hasRepeatedLabels(lhs_tail) or hasRepeatedLabels(rhs_tail)) return false;
    if (!(lhs_tail.len == 2 and rhs_tail.len == 1 and lhs_tail[1] == rhs_tail[0])) return false;
    if (out.len == 0) return true;
    if (!std.mem.startsWith(u8, out, "...")) return false;
    const out_tail = out[3..];
    return out_tail.len == 1 and
        allEinsumLabels(out_tail) and
        out_tail[0] == lhs_tail[0];
}

fn ellipsisBatchedVecmatLikeSubscripts(subscripts: []const u8, lhs_rank: usize, rhs_rank: usize) bool {
    if (lhs_rank < 1 or rhs_rank < 1) return false;
    const arrow = std.mem.indexOf(u8, subscripts, "->") orelse subscripts.len;
    const comma = std.mem.indexOfScalar(u8, subscripts[0..arrow], ',') orelse return false;
    const lhs = subscripts[0..comma];
    const rhs = subscripts[comma + 1 .. arrow];
    const out = if (arrow == subscripts.len) "" else subscripts[arrow + 2 ..];
    if (!std.mem.startsWith(u8, lhs, "...") or !std.mem.startsWith(u8, rhs, "...")) return false;
    const lhs_tail = lhs[3..];
    const rhs_tail = rhs[3..];
    if (!allEinsumLabels(lhs_tail) or !allEinsumLabels(rhs_tail)) return false;
    if (hasRepeatedLabels(lhs_tail) or hasRepeatedLabels(rhs_tail)) return false;
    if (!(lhs_tail.len == 1 and rhs_tail.len == 2 and lhs_tail[0] == rhs_tail[0])) return false;
    if (out.len == 0) return true;
    if (!std.mem.startsWith(u8, out, "...")) return false;
    const out_tail = out[3..];
    return out_tail.len == 1 and
        allEinsumLabels(out_tail) and
        out_tail[0] == rhs_tail[1];
}

fn ellipsisBatchedDotLikeSubscripts(subscripts: []const u8, lhs_rank: usize, rhs_rank: usize) bool {
    if (lhs_rank < 1 or rhs_rank < 1) return false;
    const arrow = std.mem.indexOf(u8, subscripts, "->") orelse subscripts.len;
    const comma = std.mem.indexOfScalar(u8, subscripts[0..arrow], ',') orelse return false;
    const lhs = subscripts[0..comma];
    const rhs = subscripts[comma + 1 .. arrow];
    const out = if (arrow == subscripts.len) "" else subscripts[arrow + 2 ..];
    if (!std.mem.startsWith(u8, lhs, "...") or !std.mem.startsWith(u8, rhs, "...")) return false;
    const lhs_tail = lhs[3..];
    const rhs_tail = rhs[3..];
    if (lhs_tail.len != 1 or rhs_tail.len != 1) return false;
    if (!allEinsumLabels(lhs_tail) or !allEinsumLabels(rhs_tail)) return false;
    if (lhs_tail[0] != rhs_tail[0]) return false;
    if (out.len == 0) return true;
    return std.mem.eql(u8, out, "...");
}

const max_einsum_rank = 16;

const BinaryEinsumPlan = struct {
    lhs: [max_einsum_rank]u8 = [_]u8{0} ** max_einsum_rank,
    rhs: [max_einsum_rank]u8 = [_]u8{0} ** max_einsum_rank,
    out: [max_einsum_rank * 2]u8 = [_]u8{0} ** (max_einsum_rank * 2),
    default_out: [max_einsum_rank * 2]u8 = [_]u8{0} ** (max_einsum_rank * 2),
    lhs_contract_axes: [max_einsum_rank]usize = [_]usize{0} ** max_einsum_rank,
    rhs_contract_axes: [max_einsum_rank]usize = [_]usize{0} ** max_einsum_rank,
    permutation: [max_einsum_rank * 2]usize = [_]usize{0} ** (max_einsum_rank * 2),
    lhs_len: usize = 0,
    rhs_len: usize = 0,
    out_len: usize = 0,
    default_out_len: usize = 0,
    contract_len: usize = 0,

    fn lhsAxes(plan: *const BinaryEinsumPlan) []const usize {
        return plan.lhs_contract_axes[0..plan.contract_len];
    }

    fn rhsAxes(plan: *const BinaryEinsumPlan) []const usize {
        return plan.rhs_contract_axes[0..plan.contract_len];
    }

    fn permuteAxes(plan: *const BinaryEinsumPlan) []const usize {
        return plan.permutation[0..plan.out_len];
    }

    fn outputIsDefault(plan: BinaryEinsumPlan) bool {
        return plan.out_len == plan.default_out_len and std.mem.eql(u8, plan.out[0..plan.out_len], plan.default_out[0..plan.default_out_len]);
    }

    fn matmulLike(plan: BinaryEinsumPlan) bool {
        return plan.lhs_len == 2 and plan.rhs_len == 2 and plan.contract_len == 1 and
            plan.lhs_contract_axes[0] == 1 and plan.rhs_contract_axes[0] == 0 and
            plan.outputIsDefault();
    }

    fn matvecLike(plan: BinaryEinsumPlan) bool {
        return plan.lhs_len == 2 and plan.rhs_len == 1 and plan.contract_len == 1 and
            plan.lhs_contract_axes[0] == 1 and plan.rhs_contract_axes[0] == 0 and
            plan.outputIsDefault();
    }

    fn vecmatLike(plan: BinaryEinsumPlan) bool {
        return plan.lhs_len == 1 and plan.rhs_len == 2 and plan.contract_len == 1 and
            plan.lhs_contract_axes[0] == 0 and plan.rhs_contract_axes[0] == 0 and
            plan.outputIsDefault();
    }

    fn dotLike(plan: BinaryEinsumPlan) bool {
        return plan.lhs_len == 1 and plan.rhs_len == 1 and plan.contract_len == 1 and plan.out_len == 0;
    }

    fn outerLike(plan: BinaryEinsumPlan) bool {
        return plan.lhs_len == 1 and plan.rhs_len == 1 and plan.contract_len == 0 and plan.outputIsDefault();
    }
};

fn parseBinaryEinsum(subscripts: []const u8, lhs_rank: usize, rhs_rank: usize) ArrayError!BinaryEinsumPlan {
    if (lhs_rank > max_einsum_rank or rhs_rank > max_einsum_rank) return error.InvalidShape;
    if (std.mem.indexOf(u8, subscripts, "...") != null) return error.InvalidShape;
    const explicit_output = std.mem.indexOf(u8, subscripts, "->");
    const arrow = explicit_output orelse subscripts.len;
    if (explicit_output != null and std.mem.indexOf(u8, subscripts[arrow + 2 ..], "->") != null) return error.InvalidShape;
    const comma = std.mem.indexOfScalar(u8, subscripts[0..arrow], ',') orelse return error.InvalidShape;

    var plan: BinaryEinsumPlan = .{};
    plan.lhs_len = try parseEinsumLabels(subscripts[0..comma], lhs_rank, plan.lhs[0..]);
    plan.rhs_len = try parseEinsumLabels(subscripts[comma + 1 .. arrow], rhs_rank, plan.rhs[0..]);
    if (explicit_output) |_| {
        plan.out_len = try parseEinsumLabels(subscripts[arrow + 2 ..], null, plan.out[0..]);
    }

    var out_seen = [_]bool{false} ** 256;
    for (plan.out[0..plan.out_len]) |label| {
        if (out_seen[label]) return error.InvalidShape;
        out_seen[label] = true;
    }

    var default_seen = [_]bool{false} ** 256;
    for (plan.lhs[0..plan.lhs_len], 0..) |label, lhs_axis| {
        if (findLabel(plan.rhs[0..plan.rhs_len], label)) |rhs_axis| {
            if (out_seen[label]) return error.InvalidShape; // shared batch labels are a future extension.
            plan.lhs_contract_axes[plan.contract_len] = lhs_axis;
            plan.rhs_contract_axes[plan.contract_len] = rhs_axis;
            plan.contract_len += 1;
        } else {
            plan.default_out[plan.default_out_len] = label;
            default_seen[label] = true;
            plan.default_out_len += 1;
        }
    }
    for (plan.rhs[0..plan.rhs_len]) |label| {
        if (findLabel(plan.lhs[0..plan.lhs_len], label) == null) {
            plan.default_out[plan.default_out_len] = label;
            default_seen[label] = true;
            plan.default_out_len += 1;
        }
    }
    if (explicit_output == null) {
        @memcpy(plan.out[0..plan.default_out_len], plan.default_out[0..plan.default_out_len]);
        plan.out_len = plan.default_out_len;
    }
    if (plan.out_len != plan.default_out_len) return error.InvalidShape;
    for (plan.out[0..plan.out_len], 0..) |label, out_axis| {
        if (!default_seen[label]) return error.InvalidShape;
        plan.permutation[out_axis] = findLabel(plan.default_out[0..plan.default_out_len], label) orelse return error.InvalidShape;
    }
    return plan;
}

fn parseEinsumLabels(segment: []const u8, expected_rank: ?usize, out: []u8) ArrayError!usize {
    if (segment.len > out.len) return error.InvalidShape;
    if (expected_rank) |rank| {
        if (segment.len != rank) return error.InvalidShape;
    }
    var seen = [_]bool{false} ** 256;
    for (segment, 0..) |label, index| {
        if (!std.ascii.isAlphabetic(label)) return error.InvalidShape;
        if (seen[label]) return error.InvalidShape;
        seen[label] = true;
        out[index] = label;
    }
    return segment.len;
}

fn allEinsumLabels(segment: []const u8) bool {
    for (segment) |label| {
        if (!std.ascii.isAlphabetic(label)) return false;
    }
    return true;
}

fn hasRepeatedLabels(segment: []const u8) bool {
    var seen = [_]bool{false} ** 256;
    for (segment) |label| {
        if (seen[label]) return true;
        seen[label] = true;
    }
    return false;
}

fn findLabel(labels: []const u8, needle: u8) ?usize {
    for (labels, 0..) |label, index| {
        if (label == needle) return index;
    }
    return null;
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
    _ = axiom_backend;
    _ = series_mod;
    _ = dataframe_mod;
    _ = layered_array_mod;
    _ = forge_interop_mod;
    _ = linalg;
    _ = stats;
    _ = sparse;
}
