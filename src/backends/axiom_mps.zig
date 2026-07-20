//! Platform facade for Vectra's Axiom MPS backend bridge.
//!
//! MPS is a macOS-only backend. Keep this module importable on all targets so
//! the public `.mps` device spelling and availability diagnostics remain stable,
//! but compile the actual MPS storage/kernel bridge only for macOS. Non-macOS
//! builds use a tiny unavailable stub instead of semantically analyzing Metal or
//! Objective-C-backed code through Axiom.

const builtin = @import("builtin");
const axiom = @import("axiom");
const array_mod = @import("../array.zig");

const impl = if (builtin.os.tag == .macos)
    @import("axiom_mps_macos.zig")
else
    @import("axiom_mps_stub.zig");

pub const deviceAvailable = impl.deviceAvailable;
pub const allocateStorage = impl.allocateStorage;
pub const freeStorage = impl.freeStorage;
pub const uploadStorage = impl.uploadStorage;
pub const downloadStorage = impl.downloadStorage;
pub const copyStorage = impl.copyStorage;
pub const fillStorage = impl.fillStorage;
pub const fillPhiloxUniform = impl.fillPhiloxUniform;
pub const fillPhiloxNormal = impl.fillPhiloxNormal;
pub const tryBinaryF32 = impl.tryBinaryF32;
pub const tryRank3BroadcastBinaryF32 = impl.tryRank3BroadcastBinaryF32;
pub const tryRank4BroadcastBinaryF32 = impl.tryRank4BroadcastBinaryF32;
pub const tryRankedBroadcastBinaryF32 = impl.tryRankedBroadcastBinaryF32;
pub const tryScalarF32 = impl.tryScalarF32;
pub const tryUnaryF32 = impl.tryUnaryF32;
pub const tryBinaryF16 = impl.tryBinaryF16;
pub const tryRank3BroadcastBinaryF16 = impl.tryRank3BroadcastBinaryF16;
pub const tryRank4BroadcastBinaryF16 = impl.tryRank4BroadcastBinaryF16;
pub const tryRankedBroadcastBinaryF16 = impl.tryRankedBroadcastBinaryF16;
pub const tryScalarF16 = impl.tryScalarF16;
pub const tryUnaryF16 = impl.tryUnaryF16;
pub const tryBinaryBF16 = impl.tryBinaryBF16;
pub const tryRank3BroadcastBinaryBF16 = impl.tryRank3BroadcastBinaryBF16;
pub const tryRank4BroadcastBinaryBF16 = impl.tryRank4BroadcastBinaryBF16;
pub const tryRankedBroadcastBinaryBF16 = impl.tryRankedBroadcastBinaryBF16;
pub const tryScalarBF16 = impl.tryScalarBF16;
pub const tryUnaryBF16 = impl.tryUnaryBF16;
pub const tryMatmulF32 = impl.tryMatmulF32;
pub const tryMatmulF16 = impl.tryMatmulF16;
pub const tryMatmulBF16 = impl.tryMatmulBF16;
pub const tryBmmF32 = impl.tryBmmF32;
pub const tryBmmF16 = impl.tryBmmF16;
pub const tryBmmBF16 = impl.tryBmmBF16;
pub const tryBroadcastBmmF32 = impl.tryBroadcastBmmF32;
pub const tryBroadcastBmmF16 = impl.tryBroadcastBmmF16;
pub const tryBroadcastBmmBF16 = impl.tryBroadcastBmmBF16;
pub const tryRank4BroadcastBmmF32 = impl.tryRank4BroadcastBmmF32;
pub const tryRank4BroadcastBmmF16 = impl.tryRank4BroadcastBmmF16;
pub const tryRank4BroadcastBmmBF16 = impl.tryRank4BroadcastBmmBF16;
pub const tryRankedBroadcastBmmF32 = impl.tryRankedBroadcastBmmF32;
pub const tryRankedBroadcastBmmF16 = impl.tryRankedBroadcastBmmF16;
pub const tryRankedBroadcastBmmBF16 = impl.tryRankedBroadcastBmmBF16;
pub const tryBatchedMatvecF32 = impl.tryBatchedMatvecF32;
pub const tryBatchedVecmatF32 = impl.tryBatchedVecmatF32;
pub const tryBatchedMatvecF16 = impl.tryBatchedMatvecF16;
pub const tryBatchedVecmatF16 = impl.tryBatchedVecmatF16;
pub const tryBatchedMatvecBF16 = impl.tryBatchedMatvecBF16;
pub const tryBatchedVecmatBF16 = impl.tryBatchedVecmatBF16;
pub const tryMatmulAddF32 = impl.tryMatmulAddF32;
pub const tryMatmulAddF16 = impl.tryMatmulAddF16;
pub const tryMatmulAddBF16 = impl.tryMatmulAddBF16;
pub const tryTransposeF32 = impl.tryTransposeF32;
pub const tryBroadcastBinaryF32 = impl.tryBroadcastBinaryF32;
pub const tryBroadcastAddF32 = impl.tryBroadcastAddF32;
pub const tryReductionF32 = impl.tryReductionF32;
pub const tryTransposeF16 = impl.tryTransposeF16;
pub const tryBroadcastBinaryF16 = impl.tryBroadcastBinaryF16;
pub const tryBroadcastAddF16 = impl.tryBroadcastAddF16;
pub const tryReductionF16 = impl.tryReductionF16;
pub const tryTransposeBF16 = impl.tryTransposeBF16;
pub const tryBroadcastBinaryBF16 = impl.tryBroadcastBinaryBF16;
pub const tryBroadcastAddBF16 = impl.tryBroadcastAddBF16;
pub const tryReductionBF16 = impl.tryReductionBF16;
pub const trySoftmaxF32 = impl.trySoftmaxF32;
pub const trySoftmaxF16 = impl.trySoftmaxF16;
pub const trySoftmaxBF16 = impl.trySoftmaxBF16;

pub fn trySoftmax(comptime T: type, op: axiom.accelerator.MpsSoftmaxOp, input: array_mod.Array(T), axis: u1) array_mod.ArrayError!?array_mod.Array(T) {
    return if (T == f32)
        trySoftmaxF32(op, @as(array_mod.Array(f32), input), axis)
    else if (T == f16)
        trySoftmaxF16(op, @as(array_mod.Array(f16), input), axis)
    else if (T == array_mod.BFloat16)
        trySoftmaxBF16(op, @as(array_mod.Array(array_mod.BFloat16), input), axis)
    else
        null;
}

pub fn tryReduction(comptime T: type, op: axiom.accelerator.MpsReductionOp, input: array_mod.Array(T), axis: u1, keepdims: bool) array_mod.ArrayError!?array_mod.Array(T) {
    return if (T == f32)
        tryReductionF32(op, @as(array_mod.Array(f32), input), axis, keepdims)
    else if (T == f16)
        tryReductionF16(op, @as(array_mod.Array(f16), input), axis, keepdims)
    else if (T == array_mod.BFloat16)
        tryReductionBF16(op, @as(array_mod.Array(array_mod.BFloat16), input), axis, keepdims)
    else
        null;
}

pub fn tryTranspose(comptime T: type, input: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    return if (T == f32)
        tryTransposeF32(@as(array_mod.Array(f32), input))
    else if (T == f16)
        tryTransposeF16(@as(array_mod.Array(f16), input))
    else if (T == array_mod.BFloat16)
        tryTransposeBF16(@as(array_mod.Array(array_mod.BFloat16), input))
    else
        null;
}
