//! Non-macOS stub for Vectra's Axiom MPS backend bridge.
//!
//! This file intentionally avoids all MPS runtime calls. It preserves the
//! backend facade contract while making `.mps` unavailable on platforms where
//! Metal/MPS cannot participate in the build.

const std = @import("std");
const axiom = @import("axiom");
const array_mod = @import("../array.zig");

pub fn deviceAvailable(index: usize) bool {
    _ = index;
    return false;
}

pub fn allocateStorage(device: array_mod.Device, len: usize, element_size: usize) array_mod.ArrayError!?array_mod.DeviceStorage {
    _ = device;
    _ = len;
    _ = element_size;
    return error.InvalidDevice;
}

pub fn freeStorage(storage: array_mod.DeviceStorage) void {
    _ = storage;
    return;
}

pub fn uploadStorage(storage: array_mod.DeviceStorage, bytes: []const u8) array_mod.ArrayError!void {
    _ = storage;
    _ = bytes;
    return error.InvalidDevice;
}

pub fn downloadStorage(storage: array_mod.DeviceStorage, bytes: []u8) array_mod.ArrayError!void {
    _ = storage;
    _ = bytes;
    return error.InvalidDevice;
}

pub fn copyStorage(dst: array_mod.DeviceStorage, src: array_mod.DeviceStorage) array_mod.ArrayError!void {
    _ = dst;
    _ = src;
    return error.InvalidDevice;
}

pub fn fillStorage(comptime T: type, storage: array_mod.DeviceStorage, value: T) array_mod.ArrayError!void {
    _ = storage;
    _ = value;
    return error.InvalidDevice;
}

pub fn fillPhiloxUniform(comptime T: type, storage: array_mod.DeviceStorage, seed: u64) array_mod.ArrayError!void {
    _ = T;
    _ = storage;
    _ = seed;
    return error.InvalidDevice;
}

pub fn fillPhiloxNormal(comptime T: type, storage: array_mod.DeviceStorage, seed: u64, mean: T, stddev: T) array_mod.ArrayError!void {
    _ = storage;
    _ = seed;
    _ = mean;
    _ = stddev;
    return error.InvalidDevice;
}

pub const Histogram2DCountSession = struct {
    pub fn init(device: array_mod.Device, cols: u32, rows: u32) array_mod.ArrayError!Histogram2DCountSession {
        _ = device;
        _ = cols;
        _ = rows;
        return error.InvalidDevice;
    }

    pub fn deinit(self: *Histogram2DCountSession) void {
        self.* = undefined;
    }

    pub fn run(self: *Histogram2DCountSession, x: array_mod.Array(f32), y: array_mod.Array(f32), bounds: [4]f32, counts: []u32, representatives: []u32, diagnostics: *[2]u32) array_mod.ArrayError!void {
        _ = self;
        _ = x;
        _ = y;
        _ = bounds;
        _ = counts;
        _ = representatives;
        _ = diagnostics;
        return error.InvalidDevice;
    }

    pub fn runMasked(self: *Histogram2DCountSession, x: array_mod.Array(f32), y: array_mod.Array(f32), x_validity: ?array_mod.Array(bool), y_validity: ?array_mod.Array(bool), bounds: [4]f32, counts: []u32, representatives: []u32, diagnostics: *[3]u32) array_mod.ArrayError!void {
        _ = self;
        _ = x;
        _ = y;
        _ = x_validity;
        _ = y_validity;
        _ = bounds;
        _ = counts;
        _ = representatives;
        _ = diagnostics;
        return error.InvalidDevice;
    }
};

pub const Histogram2DExtremaOp = enum { min, max };

pub const Histogram2DExtremaSession = struct {
    pub fn init(device: array_mod.Device, cols: u32, rows: u32, op: Histogram2DExtremaOp) array_mod.ArrayError!Histogram2DExtremaSession {
        _ = device;
        _ = cols;
        _ = rows;
        _ = op;
        return error.InvalidDevice;
    }

    pub fn deinit(self: *Histogram2DExtremaSession) void {
        self.* = undefined;
    }

    pub fn run(self: *Histogram2DExtremaSession, x: array_mod.Array(f32), y: array_mod.Array(f32), values: array_mod.Array(f32), bounds: [4]f32, counts: []u32, extrema: []f32, representatives: []u32, diagnostics: *[3]u32) array_mod.ArrayError!void {
        _ = self;
        _ = x;
        _ = y;
        _ = values;
        _ = bounds;
        _ = counts;
        _ = extrema;
        _ = representatives;
        _ = diagnostics;
        return error.InvalidDevice;
    }

    pub fn runMasked(self: *Histogram2DExtremaSession, x: array_mod.Array(f32), y: array_mod.Array(f32), values: array_mod.Array(f32), x_validity: ?array_mod.Array(bool), y_validity: ?array_mod.Array(bool), value_validity: ?array_mod.Array(bool), bounds: [4]f32, counts: []u32, extrema: []f32, representatives: []u32, diagnostics: *[4]u32) array_mod.ArrayError!void {
        _ = self;
        _ = x;
        _ = y;
        _ = values;
        _ = x_validity;
        _ = y_validity;
        _ = value_validity;
        _ = bounds;
        _ = counts;
        _ = extrema;
        _ = representatives;
        _ = diagnostics;
        return error.InvalidDevice;
    }
};

pub const Histogram2DSumSession = struct {
    pub fn init(device: array_mod.Device, cols: u32, rows: u32) array_mod.ArrayError!Histogram2DSumSession {
        _ = device;
        _ = cols;
        _ = rows;
        return error.InvalidDevice;
    }

    pub fn deinit(self: *Histogram2DSumSession) void {
        self.* = undefined;
    }

    pub fn run(self: *Histogram2DSumSession, x: array_mod.Array(f32), y: array_mod.Array(f32), values: array_mod.Array(f32), bounds: [4]f32, counts: []u32, sums: []f32, representatives: []u32, diagnostics: *[4]u32) array_mod.ArrayError!void {
        _ = self;
        _ = x;
        _ = y;
        _ = values;
        _ = bounds;
        _ = counts;
        _ = sums;
        _ = representatives;
        _ = diagnostics;
        return error.InvalidDevice;
    }

    pub fn runMasked(self: *Histogram2DSumSession, x: array_mod.Array(f32), y: array_mod.Array(f32), values: array_mod.Array(f32), x_validity: ?array_mod.Array(bool), y_validity: ?array_mod.Array(bool), value_validity: ?array_mod.Array(bool), bounds: [4]f32, counts: []u32, sums: []f32, representatives: []u32, diagnostics: *[5]u32) array_mod.ArrayError!void {
        _ = self;
        _ = x;
        _ = y;
        _ = values;
        _ = x_validity;
        _ = y_validity;
        _ = value_validity;
        _ = bounds;
        _ = counts;
        _ = sums;
        _ = representatives;
        _ = diagnostics;
        return error.InvalidDevice;
    }
};

pub const CategoricalHistogram2DCountSession = struct {
    pub fn init(device: array_mod.Device, cols: u32, rows: u32, category_count: u32) array_mod.ArrayError!CategoricalHistogram2DCountSession {
        _ = device;
        _ = cols;
        _ = rows;
        _ = category_count;
        return error.InvalidDevice;
    }

    pub fn deinit(self: *CategoricalHistogram2DCountSession) void {
        self.* = undefined;
    }

    pub fn run(self: *CategoricalHistogram2DCountSession, x: array_mod.Array(f32), y: array_mod.Array(f32), categories: array_mod.Array(i32), bounds: [4]f32, category_counts: []u32, representatives: []u32, diagnostics: *[3]u32) array_mod.ArrayError!void {
        _ = self;
        _ = x;
        _ = y;
        _ = categories;
        _ = bounds;
        _ = category_counts;
        _ = representatives;
        _ = diagnostics;
        return error.InvalidDevice;
    }

    pub fn runMasked(self: *CategoricalHistogram2DCountSession, x: array_mod.Array(f32), y: array_mod.Array(f32), categories: array_mod.Array(i32), x_validity: ?array_mod.Array(bool), y_validity: ?array_mod.Array(bool), category_validity: ?array_mod.Array(bool), bounds: [4]f32, category_counts: []u32, representatives: []u32, diagnostics: *[4]u32) array_mod.ArrayError!void {
        _ = self;
        _ = x;
        _ = y;
        _ = categories;
        _ = x_validity;
        _ = y_validity;
        _ = category_validity;
        _ = bounds;
        _ = category_counts;
        _ = representatives;
        _ = diagnostics;
        return error.InvalidDevice;
    }
};

pub fn tryBinaryF32(op: axiom.accelerator.MpsBinaryOp, lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = op;
    _ = lhs;
    _ = rhs;
    return null;
}

pub fn tryRank3BroadcastBinaryF32(op: axiom.accelerator.MpsBinaryOp, lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = op;
    _ = lhs;
    _ = rhs;
    return null;
}

pub fn tryRank4BroadcastBinaryF32(op: axiom.accelerator.MpsBinaryOp, lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = op;
    _ = lhs;
    _ = rhs;
    return null;
}

pub fn tryRankedBroadcastBinaryF32(op: axiom.accelerator.MpsBinaryOp, lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = op;
    _ = lhs;
    _ = rhs;
    return null;
}

pub fn tryScalarF32(op: axiom.accelerator.MpsBinaryOp, input: array_mod.Array(f32), scalar: f32, scalar_left: bool) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = op;
    _ = input;
    _ = scalar;
    _ = scalar_left;
    return null;
}

pub fn tryUnaryF32(op: axiom.accelerator.MpsUnaryOp, input: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = op;
    _ = input;
    return null;
}

pub fn tryBinaryF16(op: axiom.accelerator.MpsBinaryOp, lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    _ = op;
    _ = lhs;
    _ = rhs;
    return null;
}

pub fn tryRank3BroadcastBinaryF16(op: axiom.accelerator.MpsBinaryOp, lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    _ = op;
    _ = lhs;
    _ = rhs;
    return null;
}

pub fn tryRank4BroadcastBinaryF16(op: axiom.accelerator.MpsBinaryOp, lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    _ = op;
    _ = lhs;
    _ = rhs;
    return null;
}

pub fn tryRankedBroadcastBinaryF16(op: axiom.accelerator.MpsBinaryOp, lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    _ = op;
    _ = lhs;
    _ = rhs;
    return null;
}

pub fn tryScalarF16(op: axiom.accelerator.MpsBinaryOp, input: array_mod.Array(f16), scalar: f16, scalar_left: bool) array_mod.ArrayError!?array_mod.Array(f16) {
    _ = op;
    _ = input;
    _ = scalar;
    _ = scalar_left;
    return null;
}

pub fn tryUnaryF16(op: axiom.accelerator.MpsUnaryOp, input: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    _ = op;
    _ = input;
    return null;
}

pub fn tryBinaryBF16(op: axiom.accelerator.MpsBinaryOp, lhs: array_mod.Array(array_mod.BFloat16), rhs: array_mod.Array(array_mod.BFloat16)) array_mod.ArrayError!?array_mod.Array(array_mod.BFloat16) {
    _ = op;
    _ = lhs;
    _ = rhs;
    return null;
}

pub fn tryRank3BroadcastBinaryBF16(op: axiom.accelerator.MpsBinaryOp, lhs: array_mod.Array(array_mod.BFloat16), rhs: array_mod.Array(array_mod.BFloat16)) array_mod.ArrayError!?array_mod.Array(array_mod.BFloat16) {
    _ = op;
    _ = lhs;
    _ = rhs;
    return null;
}

pub fn tryRank4BroadcastBinaryBF16(op: axiom.accelerator.MpsBinaryOp, lhs: array_mod.Array(array_mod.BFloat16), rhs: array_mod.Array(array_mod.BFloat16)) array_mod.ArrayError!?array_mod.Array(array_mod.BFloat16) {
    _ = op;
    _ = lhs;
    _ = rhs;
    return null;
}

pub fn tryRankedBroadcastBinaryBF16(op: axiom.accelerator.MpsBinaryOp, lhs: array_mod.Array(array_mod.BFloat16), rhs: array_mod.Array(array_mod.BFloat16)) array_mod.ArrayError!?array_mod.Array(array_mod.BFloat16) {
    _ = op;
    _ = lhs;
    _ = rhs;
    return null;
}

pub fn tryScalarBF16(op: axiom.accelerator.MpsBinaryOp, input: array_mod.Array(array_mod.BFloat16), scalar: array_mod.BFloat16, scalar_left: bool) array_mod.ArrayError!?array_mod.Array(array_mod.BFloat16) {
    _ = op;
    _ = input;
    _ = scalar;
    _ = scalar_left;
    return null;
}

pub fn tryUnaryBF16(op: axiom.accelerator.MpsUnaryOp, input: array_mod.Array(array_mod.BFloat16)) array_mod.ArrayError!?array_mod.Array(array_mod.BFloat16) {
    _ = op;
    _ = input;
    return null;
}

pub fn tryMatmulF32(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = lhs;
    _ = rhs;
    return null;
}

pub fn tryMatmulF16(lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    _ = lhs;
    _ = rhs;
    return null;
}

pub fn tryMatmulBF16(lhs: array_mod.Array(array_mod.BFloat16), rhs: array_mod.Array(array_mod.BFloat16)) array_mod.ArrayError!?array_mod.Array(array_mod.BFloat16) {
    _ = lhs;
    _ = rhs;
    return null;
}

pub fn tryBmmF32(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = lhs;
    _ = rhs;
    return null;
}

pub fn tryBmmF16(lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    _ = lhs;
    _ = rhs;
    return null;
}

pub fn tryBmmBF16(lhs: array_mod.Array(array_mod.BFloat16), rhs: array_mod.Array(array_mod.BFloat16)) array_mod.ArrayError!?array_mod.Array(array_mod.BFloat16) {
    _ = lhs;
    _ = rhs;
    return null;
}

pub fn tryBroadcastBmmF32(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = lhs;
    _ = rhs;
    return null;
}

pub fn tryBroadcastBmmF16(lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    _ = lhs;
    _ = rhs;
    return null;
}

pub fn tryBroadcastBmmBF16(lhs: array_mod.Array(array_mod.BFloat16), rhs: array_mod.Array(array_mod.BFloat16)) array_mod.ArrayError!?array_mod.Array(array_mod.BFloat16) {
    _ = lhs;
    _ = rhs;
    return null;
}

pub fn tryRank4BroadcastBmmF32(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = lhs;
    _ = rhs;
    return null;
}

pub fn tryRank4BroadcastBmmF16(lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    _ = lhs;
    _ = rhs;
    return null;
}

pub fn tryRank4BroadcastBmmBF16(lhs: array_mod.Array(array_mod.BFloat16), rhs: array_mod.Array(array_mod.BFloat16)) array_mod.ArrayError!?array_mod.Array(array_mod.BFloat16) {
    _ = lhs;
    _ = rhs;
    return null;
}

pub fn tryRankedBroadcastBmmF32(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = lhs;
    _ = rhs;
    return null;
}

pub fn tryRankedBroadcastBmmF16(lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    _ = lhs;
    _ = rhs;
    return null;
}

pub fn tryRankedBroadcastBmmBF16(lhs: array_mod.Array(array_mod.BFloat16), rhs: array_mod.Array(array_mod.BFloat16)) array_mod.ArrayError!?array_mod.Array(array_mod.BFloat16) {
    _ = lhs;
    _ = rhs;
    return null;
}

pub fn tryBatchedMatvecF32(matrix: array_mod.Array(f32), vector: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = matrix;
    _ = vector;
    return null;
}

pub fn tryBatchedVecmatF32(vector: array_mod.Array(f32), matrix: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = vector;
    _ = matrix;
    return null;
}

pub fn tryBatchedMatvecF16(matrix: array_mod.Array(f16), vector: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    _ = matrix;
    _ = vector;
    return null;
}

pub fn tryBatchedVecmatF16(vector: array_mod.Array(f16), matrix: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    _ = vector;
    _ = matrix;
    return null;
}

pub fn tryBatchedMatvecBF16(matrix: array_mod.Array(array_mod.BFloat16), vector: array_mod.Array(array_mod.BFloat16)) array_mod.ArrayError!?array_mod.Array(array_mod.BFloat16) {
    _ = matrix;
    _ = vector;
    return null;
}

pub fn tryBatchedVecmatBF16(vector: array_mod.Array(array_mod.BFloat16), matrix: array_mod.Array(array_mod.BFloat16)) array_mod.ArrayError!?array_mod.Array(array_mod.BFloat16) {
    _ = vector;
    _ = matrix;
    return null;
}

pub fn tryMatmulAddF32(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32), addend: array_mod.Array(f32), alpha: f32, beta: f32) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = lhs;
    _ = rhs;
    _ = addend;
    _ = alpha;
    _ = beta;
    return null;
}

pub fn tryMatmulAddF16(lhs: array_mod.Array(f16), rhs: array_mod.Array(f16), addend: array_mod.Array(f16), alpha: f32, beta: f32) array_mod.ArrayError!?array_mod.Array(f16) {
    _ = lhs;
    _ = rhs;
    _ = addend;
    _ = alpha;
    _ = beta;
    return null;
}

pub fn tryMatmulAddBF16(lhs: array_mod.Array(array_mod.BFloat16), rhs: array_mod.Array(array_mod.BFloat16), addend: array_mod.Array(array_mod.BFloat16), alpha: f32, beta: f32) array_mod.ArrayError!?array_mod.Array(array_mod.BFloat16) {
    _ = lhs;
    _ = rhs;
    _ = addend;
    _ = alpha;
    _ = beta;
    return null;
}

pub fn tryTransposeF32(input: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = input;
    return null;
}

pub fn tryBroadcastBinaryF32(op: axiom.accelerator.MpsBinaryOp, input: array_mod.Array(f32), bias: array_mod.Array(f32), axis: axiom.accelerator.DialectBroadcastAxis) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = op;
    _ = input;
    _ = bias;
    _ = axis;
    return null;
}

pub fn tryBroadcastAddF32(input: array_mod.Array(f32), bias: array_mod.Array(f32), axis: axiom.accelerator.DialectBroadcastAxis) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = input;
    _ = bias;
    _ = axis;
    return null;
}

pub fn tryReductionF32(op: axiom.accelerator.MpsReductionOp, input: array_mod.Array(f32), axis: u1, keepdims: bool) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = op;
    _ = input;
    _ = axis;
    _ = keepdims;
    return null;
}

pub fn tryTransposeF16(input: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    _ = input;
    return null;
}

pub fn tryBroadcastBinaryF16(op: axiom.accelerator.MpsBinaryOp, input: array_mod.Array(f16), bias: array_mod.Array(f16), axis: axiom.accelerator.DialectBroadcastAxis) array_mod.ArrayError!?array_mod.Array(f16) {
    _ = op;
    _ = input;
    _ = bias;
    _ = axis;
    return null;
}

pub fn tryBroadcastAddF16(input: array_mod.Array(f16), bias: array_mod.Array(f16), axis: axiom.accelerator.DialectBroadcastAxis) array_mod.ArrayError!?array_mod.Array(f16) {
    _ = input;
    _ = bias;
    _ = axis;
    return null;
}

pub fn tryReductionF16(op: axiom.accelerator.MpsReductionOp, input: array_mod.Array(f16), axis: u1, keepdims: bool) array_mod.ArrayError!?array_mod.Array(f16) {
    _ = op;
    _ = input;
    _ = axis;
    _ = keepdims;
    return null;
}

pub fn tryTransposeBF16(input: array_mod.Array(array_mod.BFloat16)) array_mod.ArrayError!?array_mod.Array(array_mod.BFloat16) {
    _ = input;
    return null;
}

pub fn tryBroadcastBinaryBF16(op: axiom.accelerator.MpsBinaryOp, input: array_mod.Array(array_mod.BFloat16), bias: array_mod.Array(array_mod.BFloat16), axis: axiom.accelerator.DialectBroadcastAxis) array_mod.ArrayError!?array_mod.Array(array_mod.BFloat16) {
    _ = op;
    _ = input;
    _ = bias;
    _ = axis;
    return null;
}

pub fn tryBroadcastAddBF16(input: array_mod.Array(array_mod.BFloat16), bias: array_mod.Array(array_mod.BFloat16), axis: axiom.accelerator.DialectBroadcastAxis) array_mod.ArrayError!?array_mod.Array(array_mod.BFloat16) {
    _ = input;
    _ = bias;
    _ = axis;
    return null;
}

pub fn tryReductionBF16(op: axiom.accelerator.MpsReductionOp, input: array_mod.Array(array_mod.BFloat16), axis: u1, keepdims: bool) array_mod.ArrayError!?array_mod.Array(array_mod.BFloat16) {
    _ = op;
    _ = input;
    _ = axis;
    _ = keepdims;
    return null;
}

pub fn trySoftmaxF32(op: axiom.accelerator.MpsSoftmaxOp, input: array_mod.Array(f32), axis: u1) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = op;
    _ = input;
    _ = axis;
    return null;
}

pub fn trySoftmaxF16(op: axiom.accelerator.MpsSoftmaxOp, input: array_mod.Array(f16), axis: u1) array_mod.ArrayError!?array_mod.Array(f16) {
    _ = op;
    _ = input;
    _ = axis;
    return null;
}

pub fn trySoftmaxBF16(op: axiom.accelerator.MpsSoftmaxOp, input: array_mod.Array(array_mod.BFloat16), axis: u1) array_mod.ArrayError!?array_mod.Array(array_mod.BFloat16) {
    _ = op;
    _ = input;
    _ = axis;
    return null;
}

test "non-macOS MPS bridge is unavailable without compiling Metal storage" {
    try std.testing.expect(!deviceAvailable(0));
    try std.testing.expectError(error.InvalidDevice, allocateStorage(.mps(0), 1, @sizeOf(f32)));
}
