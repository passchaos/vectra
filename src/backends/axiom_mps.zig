//! Axiom Metal/MPS storage bridge for Vectra.
//!
//! This module owns the MPS device-storage lifecycle behind Vectra's target
//! facade. Kernel execution will build on this ABI; the first milestone is real
//! Metal device discovery and MTLBuffer-backed Array storage.

const std = @import("std");
const array_mod = @import("../array.zig");
const axiom = @import("axiom");

pub fn deviceAvailable(index: usize) bool {
    return axiom.accelerator.mpsDeviceAvailable(index);
}

pub fn allocateStorage(device: array_mod.Device, len: usize, element_size: usize) array_mod.ArrayError!?array_mod.DeviceStorage {
    if (!device.isMps()) return null;
    const bytes = std.math.mul(usize, len, element_size) catch return error.InvalidShape;
    if (bytes == 0) return .{ .device = device, .ptr = 0, .len = len, .bytes = 0 };
    var runtime = axiom.accelerator.MpsRuntime.open(device.index) catch return error.InvalidDevice;
    defer runtime.close();
    const buffer = runtime.allocateBuffer(bytes) catch return error.BackendFailure;
    return .{ .device = device, .ptr = buffer.ptr, .len = len, .bytes = bytes };
}

pub fn freeStorage(storage: array_mod.DeviceStorage) void {
    if (!storage.device.isMps() or storage.ptr == 0 or !storage.owns) return;
    var runtime = axiom.accelerator.MpsRuntime.open(storage.device.index) catch return;
    defer runtime.close();
    runtime.freeBuffer(.{ .ptr = storage.ptr, .bytes = storage.bytes });
}

pub fn uploadStorage(storage: array_mod.DeviceStorage, bytes: []const u8) array_mod.ArrayError!void {
    if (!storage.device.isMps() or bytes.len > storage.bytes) return error.InvalidDevice;
    if (bytes.len == 0) return;
    var runtime = axiom.accelerator.MpsRuntime.open(storage.device.index) catch return error.InvalidDevice;
    defer runtime.close();
    runtime.uploadBuffer(.{ .ptr = storage.ptr, .bytes = storage.bytes }, bytes) catch return error.BackendFailure;
}

pub fn downloadStorage(storage: array_mod.DeviceStorage, bytes: []u8) array_mod.ArrayError!void {
    if (!storage.device.isMps() or bytes.len > storage.bytes) return error.InvalidDevice;
    if (bytes.len == 0) return;
    var runtime = axiom.accelerator.MpsRuntime.open(storage.device.index) catch return error.InvalidDevice;
    defer runtime.close();
    runtime.downloadBuffer(.{ .ptr = storage.ptr, .bytes = storage.bytes }, bytes) catch return error.BackendFailure;
}

pub fn copyStorage(dst: array_mod.DeviceStorage, src: array_mod.DeviceStorage) array_mod.ArrayError!void {
    if (!dst.device.sameDevice(src.device) or !dst.device.isMps()) return error.InvalidDevice;
    if (dst.bytes < src.bytes or dst.len != src.len) return error.ShapeMismatch;
    if (src.bytes == 0) return;
    var runtime = axiom.accelerator.MpsRuntime.open(dst.device.index) catch return error.InvalidDevice;
    defer runtime.close();
    runtime.copyBuffer(
        .{ .ptr = dst.ptr, .bytes = dst.bytes },
        .{ .ptr = src.ptr, .bytes = src.bytes },
        src.bytes,
    ) catch return error.BackendFailure;
}

pub fn fillStorage(comptime T: type, storage: array_mod.DeviceStorage, value: T) array_mod.ArrayError!void {
    if (!storage.device.isMps()) return error.InvalidDevice;
    if (storage.len == 0) return;
    if (storage.bytes != storage.len * @sizeOf(T)) return error.ShapeMismatch;
    const scratch = std.heap.smp_allocator;
    const tmp = try scratch.alloc(T, storage.len);
    defer scratch.free(tmp);
    @memset(tmp, value);
    return uploadStorage(storage, std.mem.sliceAsBytes(tmp));
}

pub fn fillPhiloxUniform(comptime T: type, storage: array_mod.DeviceStorage, seed: u64) array_mod.ArrayError!void {
    if (!storage.device.isMps()) return error.InvalidDevice;
    if (storage.len == 0) return;
    if (T != f32 and T != f16 and T != array_mod.BFloat16) return error.TypeUnsupported;
    if (storage.bytes != storage.len * @sizeOf(T)) return error.ShapeMismatch;
    var runtime = axiom.accelerator.MpsRuntime.open(storage.device.index) catch return error.InvalidDevice;
    defer runtime.close();
    const buffer = axiom.accelerator.MpsBuffer{ .ptr = storage.ptr, .bytes = storage.bytes };
    if (T == f32) {
        runtime.runPhiloxUniformF32(buffer, storage.len, seed) catch return error.BackendFailure;
    } else if (T == f16) {
        runtime.runPhiloxUniformF16(buffer, storage.len, seed) catch return error.BackendFailure;
    } else {
        runtime.runPhiloxUniformBF16(buffer, storage.len, seed) catch return error.BackendFailure;
    }
}

pub fn fillPhiloxNormal(comptime T: type, storage: array_mod.DeviceStorage, seed: u64, mean: T, stddev: T) array_mod.ArrayError!void {
    if (T != f32) return error.TypeUnsupported;
    if (!storage.device.isMps()) return error.InvalidDevice;
    if (storage.len == 0) return;
    if (storage.bytes != storage.len * @sizeOf(f32)) return error.ShapeMismatch;
    var runtime = axiom.accelerator.MpsRuntime.open(storage.device.index) catch return error.InvalidDevice;
    defer runtime.close();
    runtime.runPhiloxNormalF32(.{ .ptr = storage.ptr, .bytes = storage.bytes }, storage.len, seed, mean, stddev) catch return error.BackendFailure;
}

fn rank3BroadcastShape(lhs: []const usize, rhs: []const usize) ?[3]usize {
    if (lhs.len != 3 or rhs.len != 3) return null;
    return .{
        broadcastExtent(lhs[0], rhs[0]) orelse return null,
        broadcastExtent(lhs[1], rhs[1]) orelse return null,
        broadcastExtent(lhs[2], rhs[2]) orelse return null,
    };
}

fn rank4BroadcastShape(lhs: []const usize, rhs: []const usize) ?[4]usize {
    if (lhs.len != 4 or rhs.len != 4) return null;
    return .{
        broadcastExtent(lhs[0], rhs[0]) orelse return null,
        broadcastExtent(lhs[1], rhs[1]) orelse return null,
        broadcastExtent(lhs[2], rhs[2]) orelse return null,
        broadcastExtent(lhs[3], rhs[3]) orelse return null,
    };
}

const max_ranked_broadcast_rank = 6;

const RankedBroadcastShape = struct {
    rank: usize = 0,
    dims: [max_ranked_broadcast_rank]usize = [_]usize{1} ** max_ranked_broadcast_rank,
};

const max_ranked_bmm_batch_rank = 4;

const RankedBmmBatchShape = struct {
    rank: usize = 0,
    dims: [max_ranked_bmm_batch_rank]usize = [_]usize{1} ** max_ranked_bmm_batch_rank,
};

fn broadcastExtent(lhs: usize, rhs: usize) ?usize {
    if (lhs == rhs) return lhs;
    if (lhs == 1) return rhs;
    if (rhs == 1) return lhs;
    return null;
}

fn rankedBroadcastShape(lhs: []const usize, rhs: []const usize) ?RankedBroadcastShape {
    const rank = @max(lhs.len, rhs.len);
    if (rank == 0 or rank > max_ranked_broadcast_rank) return null;
    var out: RankedBroadcastShape = .{ .rank = rank };
    for (0..rank) |i| {
        const lhs_dim = if (i >= rank - lhs.len) lhs[i - (rank - lhs.len)] else 1;
        const rhs_dim = if (i >= rank - rhs.len) rhs[i - (rank - rhs.len)] else 1;
        out.dims[i] = broadcastExtent(lhs_dim, rhs_dim) orelse return null;
    }
    return out;
}

fn rankedBroadcastStrides(input_shape: []const usize, out_shape: RankedBroadcastShape) ?[max_ranked_broadcast_rank]usize {
    if (input_shape.len == 0 or input_shape.len > out_shape.rank) return null;
    var dense = [_]usize{0} ** max_ranked_broadcast_rank;
    var stride: usize = 1;
    var dim_index = input_shape.len;
    while (dim_index > 0) {
        dim_index -= 1;
        dense[dim_index] = stride;
        stride = std.math.mul(usize, stride, input_shape[dim_index]) catch return null;
    }
    var out = [_]usize{0} ** max_ranked_broadcast_rank;
    const rank_delta = out_shape.rank - input_shape.len;
    for (0..out_shape.rank) |out_index| {
        if (out_index < rank_delta) {
            out[out_index] = 0;
            continue;
        }
        const in_index = out_index - rank_delta;
        if (input_shape[in_index] == out_shape.dims[out_index]) {
            out[out_index] = dense[in_index];
        } else if (input_shape[in_index] == 1) {
            out[out_index] = 0;
        } else {
            return null;
        }
    }
    return out;
}

fn rankedBmmBatchShape(lhs_batch: []const usize, rhs_batch: []const usize) ?RankedBmmBatchShape {
    const rank = @max(lhs_batch.len, rhs_batch.len);
    if (rank == 0 or rank > max_ranked_bmm_batch_rank) return null;
    var out: RankedBmmBatchShape = .{ .rank = rank };
    for (0..rank) |i| {
        const lhs_dim = if (i >= rank - lhs_batch.len) lhs_batch[i - (rank - lhs_batch.len)] else 1;
        const rhs_dim = if (i >= rank - rhs_batch.len) rhs_batch[i - (rank - rhs_batch.len)] else 1;
        out.dims[i] = broadcastExtent(lhs_dim, rhs_dim) orelse return null;
    }
    return out;
}

fn rankedBmmBatchStrides(input_batch: []const usize, out_batch: RankedBmmBatchShape, matrix_stride: usize) ?[max_ranked_bmm_batch_rank]usize {
    if (input_batch.len == 0 or input_batch.len > out_batch.rank) return null;
    var dense = [_]usize{0} ** max_ranked_bmm_batch_rank;
    var stride = matrix_stride;
    var dim_index = input_batch.len;
    while (dim_index > 0) {
        dim_index -= 1;
        dense[dim_index] = stride;
        stride = std.math.mul(usize, stride, input_batch[dim_index]) catch return null;
    }
    var out = [_]usize{0} ** max_ranked_bmm_batch_rank;
    const rank_delta = out_batch.rank - input_batch.len;
    for (0..out_batch.rank) |out_index| {
        if (out_index < rank_delta) {
            out[out_index] = 0;
            continue;
        }
        const in_index = out_index - rank_delta;
        if (input_batch[in_index] == out_batch.dims[out_index]) {
            out[out_index] = dense[in_index];
        } else if (input_batch[in_index] == 1) {
            out[out_index] = 0;
        } else {
            return null;
        }
    }
    return out;
}

pub fn tryBinaryF32(op: axiom.accelerator.MpsBinaryOp, lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    if (!lhs.device.isMps() or !rhs.device.isMps() or !lhs.device.sameDevice(rhs.device)) return null;
    if (!std.mem.eql(usize, lhs.shape, rhs.shape) or !lhs.isContiguous() or !rhs.isContiguous()) return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;
    if (lhs_storage.len != rhs_storage.len) return null;

    var out = try array_mod.Array(f32).emptyOn(lhs.allocator, lhs.shape, lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(lhs.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runBinaryF32(
        op,
        .{ .ptr = lhs_storage.ptr, .bytes = lhs_storage.bytes },
        .{ .ptr = rhs_storage.ptr, .bytes = rhs_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        lhs_storage.len,
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryRank3BroadcastBinaryF32(op: axiom.accelerator.MpsBinaryOp, lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    if (!lhs.device.isMps() or !rhs.device.isMps() or !lhs.device.sameDevice(rhs.device)) return null;
    if (lhs.shape.len != 3 or rhs.shape.len != 3 or !lhs.isContiguous() or !rhs.isContiguous()) return null;
    const out_shape = rank3BroadcastShape(lhs.shape, rhs.shape) orelse return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;

    var out = try array_mod.Array(f32).emptyOn(lhs.allocator, &out_shape, lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(lhs.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runRank3BroadcastBinaryF32(
        op,
        .{ .ptr = lhs_storage.ptr, .bytes = lhs_storage.bytes },
        .{ .ptr = rhs_storage.ptr, .bytes = rhs_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        lhs.shape[0],
        lhs.shape[1],
        lhs.shape[2],
        rhs.shape[0],
        rhs.shape[1],
        rhs.shape[2],
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryRank4BroadcastBinaryF32(op: axiom.accelerator.MpsBinaryOp, lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    if (!lhs.device.isMps() or !rhs.device.isMps() or !lhs.device.sameDevice(rhs.device)) return null;
    if (lhs.shape.len != 4 or rhs.shape.len != 4 or !lhs.isContiguous() or !rhs.isContiguous()) return null;
    const out_shape = rank4BroadcastShape(lhs.shape, rhs.shape) orelse return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;

    var out = try array_mod.Array(f32).emptyOn(lhs.allocator, &out_shape, lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(lhs.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runRank4BroadcastBinaryF32(
        op,
        .{ .ptr = lhs_storage.ptr, .bytes = lhs_storage.bytes },
        .{ .ptr = rhs_storage.ptr, .bytes = rhs_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        lhs.shape[0],
        lhs.shape[1],
        lhs.shape[2],
        lhs.shape[3],
        rhs.shape[0],
        rhs.shape[1],
        rhs.shape[2],
        rhs.shape[3],
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryRankedBroadcastBinaryF32(op: axiom.accelerator.MpsBinaryOp, lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    if (!lhs.device.isMps() or !rhs.device.isMps() or !lhs.device.sameDevice(rhs.device)) return null;
    if (!lhs.isContiguous() or !rhs.isContiguous()) return null;
    const out_shape = rankedBroadcastShape(lhs.shape, rhs.shape) orelse return null;
    const lhs_strides = rankedBroadcastStrides(lhs.shape, out_shape) orelse return null;
    const rhs_strides = rankedBroadcastStrides(rhs.shape, out_shape) orelse return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;

    var out = try array_mod.Array(f32).emptyOn(lhs.allocator, out_shape.dims[0..out_shape.rank], lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(lhs.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runRankedBroadcastBinaryF32(
        op,
        .{ .ptr = lhs_storage.ptr, .bytes = lhs_storage.bytes },
        .{ .ptr = rhs_storage.ptr, .bytes = rhs_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        out_shape.rank,
        out_shape.dims[0..out_shape.rank],
        lhs_strides[0..out_shape.rank],
        rhs_strides[0..out_shape.rank],
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryScalarF32(op: axiom.accelerator.MpsBinaryOp, input: array_mod.Array(f32), scalar: f32, scalar_left: bool) array_mod.ArrayError!?array_mod.Array(f32) {
    if (!input.device.isMps() or !input.isContiguous()) return null;
    const input_storage = input.device_storage orelse return null;

    var out = try array_mod.Array(f32).emptyOn(input.allocator, input.shape, input.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(input.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runScalarF32(
        op,
        .{ .ptr = input_storage.ptr, .bytes = input_storage.bytes },
        scalar,
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        input_storage.len,
        scalar_left,
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryUnaryF32(op: axiom.accelerator.MpsUnaryOp, input: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    if (!input.device.isMps() or !input.isContiguous()) return null;
    const input_storage = input.device_storage orelse return null;

    var out = try array_mod.Array(f32).emptyOn(input.allocator, input.shape, input.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(input.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runUnaryF32(
        op,
        .{ .ptr = input_storage.ptr, .bytes = input_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        input_storage.len,
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryBinaryF16(op: axiom.accelerator.MpsBinaryOp, lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    if (!lhs.device.isMps() or !rhs.device.isMps() or !lhs.device.sameDevice(rhs.device)) return null;
    if (!std.mem.eql(usize, lhs.shape, rhs.shape) or !lhs.isContiguous() or !rhs.isContiguous()) return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;
    if (lhs_storage.len != rhs_storage.len) return null;

    var out = try array_mod.Array(f16).emptyOn(lhs.allocator, lhs.shape, lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(lhs.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runBinaryF16(
        op,
        .{ .ptr = lhs_storage.ptr, .bytes = lhs_storage.bytes },
        .{ .ptr = rhs_storage.ptr, .bytes = rhs_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        lhs_storage.len,
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryRank3BroadcastBinaryF16(op: axiom.accelerator.MpsBinaryOp, lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    if (!lhs.device.isMps() or !rhs.device.isMps() or !lhs.device.sameDevice(rhs.device)) return null;
    if (lhs.shape.len != 3 or rhs.shape.len != 3 or !lhs.isContiguous() or !rhs.isContiguous()) return null;
    const out_shape = rank3BroadcastShape(lhs.shape, rhs.shape) orelse return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;

    var out = try array_mod.Array(f16).emptyOn(lhs.allocator, &out_shape, lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(lhs.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runRank3BroadcastBinaryF16(
        op,
        .{ .ptr = lhs_storage.ptr, .bytes = lhs_storage.bytes },
        .{ .ptr = rhs_storage.ptr, .bytes = rhs_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        lhs.shape[0],
        lhs.shape[1],
        lhs.shape[2],
        rhs.shape[0],
        rhs.shape[1],
        rhs.shape[2],
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryRank4BroadcastBinaryF16(op: axiom.accelerator.MpsBinaryOp, lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    if (!lhs.device.isMps() or !rhs.device.isMps() or !lhs.device.sameDevice(rhs.device)) return null;
    if (lhs.shape.len != 4 or rhs.shape.len != 4 or !lhs.isContiguous() or !rhs.isContiguous()) return null;
    const out_shape = rank4BroadcastShape(lhs.shape, rhs.shape) orelse return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;

    var out = try array_mod.Array(f16).emptyOn(lhs.allocator, &out_shape, lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(lhs.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runRank4BroadcastBinaryF16(
        op,
        .{ .ptr = lhs_storage.ptr, .bytes = lhs_storage.bytes },
        .{ .ptr = rhs_storage.ptr, .bytes = rhs_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        lhs.shape[0],
        lhs.shape[1],
        lhs.shape[2],
        lhs.shape[3],
        rhs.shape[0],
        rhs.shape[1],
        rhs.shape[2],
        rhs.shape[3],
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryRankedBroadcastBinaryF16(op: axiom.accelerator.MpsBinaryOp, lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    if (!lhs.device.isMps() or !rhs.device.isMps() or !lhs.device.sameDevice(rhs.device)) return null;
    if (!lhs.isContiguous() or !rhs.isContiguous()) return null;
    const out_shape = rankedBroadcastShape(lhs.shape, rhs.shape) orelse return null;
    const lhs_strides = rankedBroadcastStrides(lhs.shape, out_shape) orelse return null;
    const rhs_strides = rankedBroadcastStrides(rhs.shape, out_shape) orelse return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;

    var out = try array_mod.Array(f16).emptyOn(lhs.allocator, out_shape.dims[0..out_shape.rank], lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(lhs.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runRankedBroadcastBinaryF16(
        op,
        .{ .ptr = lhs_storage.ptr, .bytes = lhs_storage.bytes },
        .{ .ptr = rhs_storage.ptr, .bytes = rhs_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        out_shape.rank,
        out_shape.dims[0..out_shape.rank],
        lhs_strides[0..out_shape.rank],
        rhs_strides[0..out_shape.rank],
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryScalarF16(op: axiom.accelerator.MpsBinaryOp, input: array_mod.Array(f16), scalar: f16, scalar_left: bool) array_mod.ArrayError!?array_mod.Array(f16) {
    if (!input.device.isMps() or !input.isContiguous()) return null;
    const input_storage = input.device_storage orelse return null;

    var out = try array_mod.Array(f16).emptyOn(input.allocator, input.shape, input.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(input.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runScalarF16(
        op,
        .{ .ptr = input_storage.ptr, .bytes = input_storage.bytes },
        @floatCast(scalar),
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        input_storage.len,
        scalar_left,
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryUnaryF16(op: axiom.accelerator.MpsUnaryOp, input: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    if (!input.device.isMps() or !input.isContiguous()) return null;
    const input_storage = input.device_storage orelse return null;

    var out = try array_mod.Array(f16).emptyOn(input.allocator, input.shape, input.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(input.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runUnaryF16(
        op,
        .{ .ptr = input_storage.ptr, .bytes = input_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        input_storage.len,
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryBinaryBF16(op: axiom.accelerator.MpsBinaryOp, lhs: array_mod.Array(array_mod.BFloat16), rhs: array_mod.Array(array_mod.BFloat16)) array_mod.ArrayError!?array_mod.Array(array_mod.BFloat16) {
    if (!lhs.device.isMps() or !rhs.device.isMps() or !lhs.device.sameDevice(rhs.device)) return null;
    if (!std.mem.eql(usize, lhs.shape, rhs.shape) or !lhs.isContiguous() or !rhs.isContiguous()) return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;
    if (lhs_storage.len != rhs_storage.len) return null;

    var out = try array_mod.Array(array_mod.BFloat16).emptyOn(lhs.allocator, lhs.shape, lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(lhs.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runBinaryBF16(
        op,
        .{ .ptr = lhs_storage.ptr, .bytes = lhs_storage.bytes },
        .{ .ptr = rhs_storage.ptr, .bytes = rhs_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        lhs_storage.len,
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryRank3BroadcastBinaryBF16(op: axiom.accelerator.MpsBinaryOp, lhs: array_mod.Array(array_mod.BFloat16), rhs: array_mod.Array(array_mod.BFloat16)) array_mod.ArrayError!?array_mod.Array(array_mod.BFloat16) {
    if (!lhs.device.isMps() or !rhs.device.isMps() or !lhs.device.sameDevice(rhs.device)) return null;
    if (lhs.shape.len != 3 or rhs.shape.len != 3 or !lhs.isContiguous() or !rhs.isContiguous()) return null;
    const out_shape = rank3BroadcastShape(lhs.shape, rhs.shape) orelse return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;

    var out = try array_mod.Array(array_mod.BFloat16).emptyOn(lhs.allocator, &out_shape, lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(lhs.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runRank3BroadcastBinaryBF16(
        op,
        .{ .ptr = lhs_storage.ptr, .bytes = lhs_storage.bytes },
        .{ .ptr = rhs_storage.ptr, .bytes = rhs_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        lhs.shape[0],
        lhs.shape[1],
        lhs.shape[2],
        rhs.shape[0],
        rhs.shape[1],
        rhs.shape[2],
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryRank4BroadcastBinaryBF16(op: axiom.accelerator.MpsBinaryOp, lhs: array_mod.Array(array_mod.BFloat16), rhs: array_mod.Array(array_mod.BFloat16)) array_mod.ArrayError!?array_mod.Array(array_mod.BFloat16) {
    if (!lhs.device.isMps() or !rhs.device.isMps() or !lhs.device.sameDevice(rhs.device)) return null;
    if (lhs.shape.len != 4 or rhs.shape.len != 4 or !lhs.isContiguous() or !rhs.isContiguous()) return null;
    const out_shape = rank4BroadcastShape(lhs.shape, rhs.shape) orelse return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;

    var out = try array_mod.Array(array_mod.BFloat16).emptyOn(lhs.allocator, &out_shape, lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(lhs.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runRank4BroadcastBinaryBF16(
        op,
        .{ .ptr = lhs_storage.ptr, .bytes = lhs_storage.bytes },
        .{ .ptr = rhs_storage.ptr, .bytes = rhs_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        lhs.shape[0],
        lhs.shape[1],
        lhs.shape[2],
        lhs.shape[3],
        rhs.shape[0],
        rhs.shape[1],
        rhs.shape[2],
        rhs.shape[3],
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryRankedBroadcastBinaryBF16(op: axiom.accelerator.MpsBinaryOp, lhs: array_mod.Array(array_mod.BFloat16), rhs: array_mod.Array(array_mod.BFloat16)) array_mod.ArrayError!?array_mod.Array(array_mod.BFloat16) {
    if (!lhs.device.isMps() or !rhs.device.isMps() or !lhs.device.sameDevice(rhs.device)) return null;
    if (!lhs.isContiguous() or !rhs.isContiguous()) return null;
    const out_shape = rankedBroadcastShape(lhs.shape, rhs.shape) orelse return null;
    const lhs_strides = rankedBroadcastStrides(lhs.shape, out_shape) orelse return null;
    const rhs_strides = rankedBroadcastStrides(rhs.shape, out_shape) orelse return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;

    var out = try array_mod.Array(array_mod.BFloat16).emptyOn(lhs.allocator, out_shape.dims[0..out_shape.rank], lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(lhs.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runRankedBroadcastBinaryBF16(
        op,
        .{ .ptr = lhs_storage.ptr, .bytes = lhs_storage.bytes },
        .{ .ptr = rhs_storage.ptr, .bytes = rhs_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        out_shape.rank,
        out_shape.dims[0..out_shape.rank],
        lhs_strides[0..out_shape.rank],
        rhs_strides[0..out_shape.rank],
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryScalarBF16(op: axiom.accelerator.MpsBinaryOp, input: array_mod.Array(array_mod.BFloat16), scalar: array_mod.BFloat16, scalar_left: bool) array_mod.ArrayError!?array_mod.Array(array_mod.BFloat16) {
    if (!input.device.isMps() or !input.isContiguous()) return null;
    const input_storage = input.device_storage orelse return null;

    var out = try array_mod.Array(array_mod.BFloat16).emptyOn(input.allocator, input.shape, input.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(input.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runScalarBF16(
        op,
        .{ .ptr = input_storage.ptr, .bytes = input_storage.bytes },
        scalar.toF32(),
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        input_storage.len,
        scalar_left,
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryUnaryBF16(op: axiom.accelerator.MpsUnaryOp, input: array_mod.Array(array_mod.BFloat16)) array_mod.ArrayError!?array_mod.Array(array_mod.BFloat16) {
    if (!input.device.isMps() or !input.isContiguous()) return null;
    const input_storage = input.device_storage orelse return null;

    var out = try array_mod.Array(array_mod.BFloat16).emptyOn(input.allocator, input.shape, input.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(input.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runUnaryBF16(
        op,
        .{ .ptr = input_storage.ptr, .bytes = input_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        input_storage.len,
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryMatmulF32(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    if (!lhs.device.isMps() or !rhs.device.isMps() or !lhs.device.sameDevice(rhs.device)) return null;
    if (lhs.shape.len != 2 or rhs.shape.len != 2 or !lhs.isContiguous() or !rhs.isContiguous()) return null;
    if (lhs.shape[1] != rhs.shape[0]) return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;
    const m = lhs.shape[0];
    const k = lhs.shape[1];
    const n = rhs.shape[1];

    var out = try array_mod.Array(f32).emptyOn(lhs.allocator, &.{ m, n }, lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(lhs.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runMatmulF32(
        .{ .ptr = lhs_storage.ptr, .bytes = lhs_storage.bytes },
        .{ .ptr = rhs_storage.ptr, .bytes = rhs_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        m,
        k,
        n,
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryMatmulF16(lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    if (!lhs.device.isMps() or !rhs.device.isMps() or !lhs.device.sameDevice(rhs.device)) return null;
    if (lhs.shape.len != 2 or rhs.shape.len != 2 or !lhs.isContiguous() or !rhs.isContiguous()) return null;
    if (lhs.shape[1] != rhs.shape[0]) return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;
    const m = lhs.shape[0];
    const k = lhs.shape[1];
    const n = rhs.shape[1];

    var out = try array_mod.Array(f16).emptyOn(lhs.allocator, &.{ m, n }, lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(lhs.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runMatmulF16(
        .{ .ptr = lhs_storage.ptr, .bytes = lhs_storage.bytes },
        .{ .ptr = rhs_storage.ptr, .bytes = rhs_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        m,
        k,
        n,
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryMatmulBF16(lhs: array_mod.Array(array_mod.BFloat16), rhs: array_mod.Array(array_mod.BFloat16)) array_mod.ArrayError!?array_mod.Array(array_mod.BFloat16) {
    if (!lhs.device.isMps() or !rhs.device.isMps() or !lhs.device.sameDevice(rhs.device)) return null;
    if (lhs.shape.len != 2 or rhs.shape.len != 2 or !lhs.isContiguous() or !rhs.isContiguous()) return null;
    if (lhs.shape[1] != rhs.shape[0]) return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;
    const m = lhs.shape[0];
    const k = lhs.shape[1];
    const n = rhs.shape[1];

    var out = try array_mod.Array(array_mod.BFloat16).emptyOn(lhs.allocator, &.{ m, n }, lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(lhs.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runMatmulBF16(
        .{ .ptr = lhs_storage.ptr, .bytes = lhs_storage.bytes },
        .{ .ptr = rhs_storage.ptr, .bytes = rhs_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        m,
        k,
        n,
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryBmmF32(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    if (!lhs.device.isMps() or !rhs.device.isMps() or !lhs.device.sameDevice(rhs.device)) return null;
    if (lhs.shape.len != 3 or rhs.shape.len != 3 or !lhs.isContiguous() or !rhs.isContiguous()) return null;
    if (lhs.shape[0] != rhs.shape[0] or lhs.shape[2] != rhs.shape[1]) return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;
    const batch = lhs.shape[0];
    const m = lhs.shape[1];
    const k = lhs.shape[2];
    const n = rhs.shape[2];

    var out = try array_mod.Array(f32).emptyOn(lhs.allocator, &.{ batch, m, n }, lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(lhs.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runBmmF32(
        .{ .ptr = lhs_storage.ptr, .bytes = lhs_storage.bytes },
        .{ .ptr = rhs_storage.ptr, .bytes = rhs_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        batch,
        m,
        k,
        n,
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryBmmF16(lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    if (!lhs.device.isMps() or !rhs.device.isMps() or !lhs.device.sameDevice(rhs.device)) return null;
    if (lhs.shape.len != 3 or rhs.shape.len != 3 or !lhs.isContiguous() or !rhs.isContiguous()) return null;
    if (lhs.shape[0] != rhs.shape[0] or lhs.shape[2] != rhs.shape[1]) return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;
    const batch = lhs.shape[0];
    const m = lhs.shape[1];
    const k = lhs.shape[2];
    const n = rhs.shape[2];

    var out = try array_mod.Array(f16).emptyOn(lhs.allocator, &.{ batch, m, n }, lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(lhs.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runBmmF16(
        .{ .ptr = lhs_storage.ptr, .bytes = lhs_storage.bytes },
        .{ .ptr = rhs_storage.ptr, .bytes = rhs_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        batch,
        m,
        k,
        n,
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryBmmBF16(lhs: array_mod.Array(array_mod.BFloat16), rhs: array_mod.Array(array_mod.BFloat16)) array_mod.ArrayError!?array_mod.Array(array_mod.BFloat16) {
    if (!lhs.device.isMps() or !rhs.device.isMps() or !lhs.device.sameDevice(rhs.device)) return null;
    if (lhs.shape.len != 3 or rhs.shape.len != 3 or !lhs.isContiguous() or !rhs.isContiguous()) return null;
    if (lhs.shape[0] != rhs.shape[0] or lhs.shape[2] != rhs.shape[1]) return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;
    const batch = lhs.shape[0];
    const m = lhs.shape[1];
    const k = lhs.shape[2];
    const n = rhs.shape[2];

    var out = try array_mod.Array(array_mod.BFloat16).emptyOn(lhs.allocator, &.{ batch, m, n }, lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(lhs.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runBmmBF16(
        .{ .ptr = lhs_storage.ptr, .bytes = lhs_storage.bytes },
        .{ .ptr = rhs_storage.ptr, .bytes = rhs_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        batch,
        m,
        k,
        n,
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryBroadcastBmmF32(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    if (!lhs.device.isMps() or !rhs.device.isMps() or !lhs.device.sameDevice(rhs.device)) return null;
    if (lhs.shape.len != 3 or rhs.shape.len != 3 or !lhs.isContiguous() or !rhs.isContiguous()) return null;
    if (lhs.shape[2] != rhs.shape[1]) return null;
    const lhs_broadcast = lhs.shape[0] == 1 and rhs.shape[0] > 1;
    const rhs_broadcast = rhs.shape[0] == 1 and lhs.shape[0] > 1;
    if (lhs_broadcast == rhs_broadcast) return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;
    const batch = if (lhs_broadcast) rhs.shape[0] else lhs.shape[0];
    const m = lhs.shape[1];
    const k = lhs.shape[2];
    const n = rhs.shape[2];

    var out = try array_mod.Array(f32).emptyOn(lhs.allocator, &.{ batch, m, n }, lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(lhs.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runBroadcastBmmF32(
        .{ .ptr = lhs_storage.ptr, .bytes = lhs_storage.bytes },
        .{ .ptr = rhs_storage.ptr, .bytes = rhs_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        batch,
        m,
        k,
        n,
        lhs_broadcast,
        rhs_broadcast,
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryBroadcastBmmF16(lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    if (!lhs.device.isMps() or !rhs.device.isMps() or !lhs.device.sameDevice(rhs.device)) return null;
    if (lhs.shape.len != 3 or rhs.shape.len != 3 or !lhs.isContiguous() or !rhs.isContiguous()) return null;
    if (lhs.shape[2] != rhs.shape[1]) return null;
    const lhs_broadcast = lhs.shape[0] == 1 and rhs.shape[0] > 1;
    const rhs_broadcast = rhs.shape[0] == 1 and lhs.shape[0] > 1;
    if (lhs_broadcast == rhs_broadcast) return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;
    const batch = if (lhs_broadcast) rhs.shape[0] else lhs.shape[0];
    const m = lhs.shape[1];
    const k = lhs.shape[2];
    const n = rhs.shape[2];

    var out = try array_mod.Array(f16).emptyOn(lhs.allocator, &.{ batch, m, n }, lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(lhs.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runBroadcastBmmF16(
        .{ .ptr = lhs_storage.ptr, .bytes = lhs_storage.bytes },
        .{ .ptr = rhs_storage.ptr, .bytes = rhs_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        batch,
        m,
        k,
        n,
        lhs_broadcast,
        rhs_broadcast,
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryBroadcastBmmBF16(lhs: array_mod.Array(array_mod.BFloat16), rhs: array_mod.Array(array_mod.BFloat16)) array_mod.ArrayError!?array_mod.Array(array_mod.BFloat16) {
    if (!lhs.device.isMps() or !rhs.device.isMps() or !lhs.device.sameDevice(rhs.device)) return null;
    if (lhs.shape.len != 3 or rhs.shape.len != 3 or !lhs.isContiguous() or !rhs.isContiguous()) return null;
    if (lhs.shape[2] != rhs.shape[1]) return null;
    const lhs_broadcast = lhs.shape[0] == 1 and rhs.shape[0] > 1;
    const rhs_broadcast = rhs.shape[0] == 1 and lhs.shape[0] > 1;
    if (lhs_broadcast == rhs_broadcast) return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;
    const batch = if (lhs_broadcast) rhs.shape[0] else lhs.shape[0];
    const m = lhs.shape[1];
    const k = lhs.shape[2];
    const n = rhs.shape[2];

    var out = try array_mod.Array(array_mod.BFloat16).emptyOn(lhs.allocator, &.{ batch, m, n }, lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(lhs.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runBroadcastBmmBF16(
        .{ .ptr = lhs_storage.ptr, .bytes = lhs_storage.bytes },
        .{ .ptr = rhs_storage.ptr, .bytes = rhs_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        batch,
        m,
        k,
        n,
        lhs_broadcast,
        rhs_broadcast,
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryRank4BroadcastBmmF32(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    if (!lhs.device.isMps() or !rhs.device.isMps() or !lhs.device.sameDevice(rhs.device)) return null;
    if (lhs.shape.len != 4 or rhs.shape.len != 4 or !lhs.isContiguous() or !rhs.isContiguous()) return null;
    if (lhs.shape[3] != rhs.shape[2]) return null;
    const batch0, const batch1 = broadcastBatch2(lhs.shape[0], lhs.shape[1], rhs.shape[0], rhs.shape[1]) orelse return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;
    const m = lhs.shape[2];
    const k = lhs.shape[3];
    const n = rhs.shape[3];

    var out = try array_mod.Array(f32).emptyOn(lhs.allocator, &.{ batch0, batch1, m, n }, lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(lhs.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runRank4BroadcastBmmF32(
        .{ .ptr = lhs_storage.ptr, .bytes = lhs_storage.bytes },
        .{ .ptr = rhs_storage.ptr, .bytes = rhs_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        batch0,
        batch1,
        m,
        k,
        n,
        lhs.shape[0],
        lhs.shape[1],
        rhs.shape[0],
        rhs.shape[1],
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryRank4BroadcastBmmF16(lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    if (!lhs.device.isMps() or !rhs.device.isMps() or !lhs.device.sameDevice(rhs.device)) return null;
    if (lhs.shape.len != 4 or rhs.shape.len != 4 or !lhs.isContiguous() or !rhs.isContiguous()) return null;
    if (lhs.shape[3] != rhs.shape[2]) return null;
    const batch0, const batch1 = broadcastBatch2(lhs.shape[0], lhs.shape[1], rhs.shape[0], rhs.shape[1]) orelse return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;
    const m = lhs.shape[2];
    const k = lhs.shape[3];
    const n = rhs.shape[3];

    var out = try array_mod.Array(f16).emptyOn(lhs.allocator, &.{ batch0, batch1, m, n }, lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(lhs.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runRank4BroadcastBmmF16(
        .{ .ptr = lhs_storage.ptr, .bytes = lhs_storage.bytes },
        .{ .ptr = rhs_storage.ptr, .bytes = rhs_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        batch0,
        batch1,
        m,
        k,
        n,
        lhs.shape[0],
        lhs.shape[1],
        rhs.shape[0],
        rhs.shape[1],
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryRank4BroadcastBmmBF16(lhs: array_mod.Array(array_mod.BFloat16), rhs: array_mod.Array(array_mod.BFloat16)) array_mod.ArrayError!?array_mod.Array(array_mod.BFloat16) {
    if (!lhs.device.isMps() or !rhs.device.isMps() or !lhs.device.sameDevice(rhs.device)) return null;
    if (lhs.shape.len != 4 or rhs.shape.len != 4 or !lhs.isContiguous() or !rhs.isContiguous()) return null;
    if (lhs.shape[3] != rhs.shape[2]) return null;
    const batch0, const batch1 = broadcastBatch2(lhs.shape[0], lhs.shape[1], rhs.shape[0], rhs.shape[1]) orelse return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;
    const m = lhs.shape[2];
    const k = lhs.shape[3];
    const n = rhs.shape[3];

    var out = try array_mod.Array(array_mod.BFloat16).emptyOn(lhs.allocator, &.{ batch0, batch1, m, n }, lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(lhs.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runRank4BroadcastBmmBF16(
        .{ .ptr = lhs_storage.ptr, .bytes = lhs_storage.bytes },
        .{ .ptr = rhs_storage.ptr, .bytes = rhs_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        batch0,
        batch1,
        m,
        k,
        n,
        lhs.shape[0],
        lhs.shape[1],
        rhs.shape[0],
        rhs.shape[1],
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryRankedBroadcastBmmF32(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    if (!lhs.device.isMps() or !rhs.device.isMps() or !lhs.device.sameDevice(rhs.device)) return null;
    if (lhs.shape.len < 5 or rhs.shape.len < 5 or !lhs.isContiguous() or !rhs.isContiguous()) return null;
    const lhs_batch = lhs.shape[0 .. lhs.shape.len - 2];
    const rhs_batch = rhs.shape[0 .. rhs.shape.len - 2];
    const out_batch = rankedBmmBatchShape(lhs_batch, rhs_batch) orelse return null;
    const m = lhs.shape[lhs.shape.len - 2];
    const k = lhs.shape[lhs.shape.len - 1];
    if (rhs.shape[rhs.shape.len - 2] != k) return null;
    const n = rhs.shape[rhs.shape.len - 1];
    const lhs_matrix_stride = std.math.mul(usize, m, k) catch return error.InvalidShape;
    const rhs_matrix_stride = std.math.mul(usize, k, n) catch return error.InvalidShape;
    const lhs_batch_strides = rankedBmmBatchStrides(lhs_batch, out_batch, lhs_matrix_stride) orelse return null;
    const rhs_batch_strides = rankedBmmBatchStrides(rhs_batch, out_batch, rhs_matrix_stride) orelse return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;

    var out_shape_buf = [_]usize{1} ** (max_ranked_bmm_batch_rank + 2);
    @memcpy(out_shape_buf[0..out_batch.rank], out_batch.dims[0..out_batch.rank]);
    out_shape_buf[out_batch.rank] = m;
    out_shape_buf[out_batch.rank + 1] = n;
    var out = try array_mod.Array(f32).emptyOn(lhs.allocator, out_shape_buf[0 .. out_batch.rank + 2], lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(lhs.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runRankedBroadcastBmmF32(
        .{ .ptr = lhs_storage.ptr, .bytes = lhs_storage.bytes },
        .{ .ptr = rhs_storage.ptr, .bytes = rhs_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        out_batch.rank,
        out_batch.dims[0..out_batch.rank],
        lhs_batch_strides[0..out_batch.rank],
        rhs_batch_strides[0..out_batch.rank],
        m,
        k,
        n,
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryRankedBroadcastBmmF16(lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    if (!lhs.device.isMps() or !rhs.device.isMps() or !lhs.device.sameDevice(rhs.device)) return null;
    if (lhs.shape.len < 5 or rhs.shape.len < 5 or !lhs.isContiguous() or !rhs.isContiguous()) return null;
    const lhs_batch = lhs.shape[0 .. lhs.shape.len - 2];
    const rhs_batch = rhs.shape[0 .. rhs.shape.len - 2];
    const out_batch = rankedBmmBatchShape(lhs_batch, rhs_batch) orelse return null;
    const m = lhs.shape[lhs.shape.len - 2];
    const k = lhs.shape[lhs.shape.len - 1];
    if (rhs.shape[rhs.shape.len - 2] != k) return null;
    const n = rhs.shape[rhs.shape.len - 1];
    const lhs_matrix_stride = std.math.mul(usize, m, k) catch return error.InvalidShape;
    const rhs_matrix_stride = std.math.mul(usize, k, n) catch return error.InvalidShape;
    const lhs_batch_strides = rankedBmmBatchStrides(lhs_batch, out_batch, lhs_matrix_stride) orelse return null;
    const rhs_batch_strides = rankedBmmBatchStrides(rhs_batch, out_batch, rhs_matrix_stride) orelse return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;

    var out_shape_buf = [_]usize{1} ** (max_ranked_bmm_batch_rank + 2);
    @memcpy(out_shape_buf[0..out_batch.rank], out_batch.dims[0..out_batch.rank]);
    out_shape_buf[out_batch.rank] = m;
    out_shape_buf[out_batch.rank + 1] = n;
    var out = try array_mod.Array(f16).emptyOn(lhs.allocator, out_shape_buf[0 .. out_batch.rank + 2], lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(lhs.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runRankedBroadcastBmmF16(
        .{ .ptr = lhs_storage.ptr, .bytes = lhs_storage.bytes },
        .{ .ptr = rhs_storage.ptr, .bytes = rhs_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        out_batch.rank,
        out_batch.dims[0..out_batch.rank],
        lhs_batch_strides[0..out_batch.rank],
        rhs_batch_strides[0..out_batch.rank],
        m,
        k,
        n,
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryRankedBroadcastBmmBF16(lhs: array_mod.Array(array_mod.BFloat16), rhs: array_mod.Array(array_mod.BFloat16)) array_mod.ArrayError!?array_mod.Array(array_mod.BFloat16) {
    if (!lhs.device.isMps() or !rhs.device.isMps() or !lhs.device.sameDevice(rhs.device)) return null;
    if (lhs.shape.len < 5 or rhs.shape.len < 5 or !lhs.isContiguous() or !rhs.isContiguous()) return null;
    const lhs_batch = lhs.shape[0 .. lhs.shape.len - 2];
    const rhs_batch = rhs.shape[0 .. rhs.shape.len - 2];
    const out_batch = rankedBmmBatchShape(lhs_batch, rhs_batch) orelse return null;
    const m = lhs.shape[lhs.shape.len - 2];
    const k = lhs.shape[lhs.shape.len - 1];
    if (rhs.shape[rhs.shape.len - 2] != k) return null;
    const n = rhs.shape[rhs.shape.len - 1];
    const lhs_matrix_stride = std.math.mul(usize, m, k) catch return error.InvalidShape;
    const rhs_matrix_stride = std.math.mul(usize, k, n) catch return error.InvalidShape;
    const lhs_batch_strides = rankedBmmBatchStrides(lhs_batch, out_batch, lhs_matrix_stride) orelse return null;
    const rhs_batch_strides = rankedBmmBatchStrides(rhs_batch, out_batch, rhs_matrix_stride) orelse return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;

    var out_shape_buf = [_]usize{1} ** (max_ranked_bmm_batch_rank + 2);
    @memcpy(out_shape_buf[0..out_batch.rank], out_batch.dims[0..out_batch.rank]);
    out_shape_buf[out_batch.rank] = m;
    out_shape_buf[out_batch.rank + 1] = n;
    var out = try array_mod.Array(array_mod.BFloat16).emptyOn(lhs.allocator, out_shape_buf[0 .. out_batch.rank + 2], lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(lhs.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runRankedBroadcastBmmBF16(
        .{ .ptr = lhs_storage.ptr, .bytes = lhs_storage.bytes },
        .{ .ptr = rhs_storage.ptr, .bytes = rhs_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        out_batch.rank,
        out_batch.dims[0..out_batch.rank],
        lhs_batch_strides[0..out_batch.rank],
        rhs_batch_strides[0..out_batch.rank],
        m,
        k,
        n,
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

fn broadcastBatch2(lhs0: usize, lhs1: usize, rhs0: usize, rhs1: usize) ?struct { usize, usize } {
    if (lhs0 != rhs0 and lhs0 != 1 and rhs0 != 1) return null;
    if (lhs1 != rhs1 and lhs1 != 1 and rhs1 != 1) return null;
    return .{ @max(lhs0, rhs0), @max(lhs1, rhs1) };
}

pub fn tryBatchedMatvecF32(matrix: array_mod.Array(f32), vector: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    if (!matrix.device.isMps() or !vector.device.isMps() or !matrix.device.sameDevice(vector.device)) return null;
    if (matrix.shape.len != 3 or vector.shape.len != 1 or !matrix.isContiguous() or !vector.isContiguous()) return null;
    if (matrix.shape[2] != vector.shape[0]) return null;
    const matrix_storage = matrix.device_storage orelse return null;
    const vector_storage = vector.device_storage orelse return null;
    const batch = matrix.shape[0];
    const m = matrix.shape[1];
    const k = matrix.shape[2];

    var out = try array_mod.Array(f32).emptyOn(matrix.allocator, &.{ batch, m }, matrix.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(matrix.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runBatchedMatvecF32(
        .{ .ptr = matrix_storage.ptr, .bytes = matrix_storage.bytes },
        .{ .ptr = vector_storage.ptr, .bytes = vector_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        batch,
        m,
        k,
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryBatchedVecmatF32(vector: array_mod.Array(f32), matrix: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    if (!vector.device.isMps() or !matrix.device.isMps() or !vector.device.sameDevice(matrix.device)) return null;
    if (vector.shape.len != 1 or matrix.shape.len != 3 or !vector.isContiguous() or !matrix.isContiguous()) return null;
    if (vector.shape[0] != matrix.shape[1]) return null;
    const vector_storage = vector.device_storage orelse return null;
    const matrix_storage = matrix.device_storage orelse return null;
    const batch = matrix.shape[0];
    const k = matrix.shape[1];
    const n = matrix.shape[2];

    var out = try array_mod.Array(f32).emptyOn(vector.allocator, &.{ batch, n }, vector.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(vector.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runBatchedVecmatF32(
        .{ .ptr = vector_storage.ptr, .bytes = vector_storage.bytes },
        .{ .ptr = matrix_storage.ptr, .bytes = matrix_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        batch,
        k,
        n,
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryBatchedMatvecF16(matrix: array_mod.Array(f16), vector: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    if (!matrix.device.isMps() or !vector.device.isMps() or !matrix.device.sameDevice(vector.device)) return null;
    if (matrix.shape.len != 3 or vector.shape.len != 1 or !matrix.isContiguous() or !vector.isContiguous()) return null;
    if (matrix.shape[2] != vector.shape[0]) return null;
    const matrix_storage = matrix.device_storage orelse return null;
    const vector_storage = vector.device_storage orelse return null;
    const batch = matrix.shape[0];
    const m = matrix.shape[1];
    const k = matrix.shape[2];

    var out = try array_mod.Array(f16).emptyOn(matrix.allocator, &.{ batch, m }, matrix.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(matrix.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runBatchedMatvecF16(
        .{ .ptr = matrix_storage.ptr, .bytes = matrix_storage.bytes },
        .{ .ptr = vector_storage.ptr, .bytes = vector_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        batch,
        m,
        k,
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryBatchedVecmatF16(vector: array_mod.Array(f16), matrix: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    if (!vector.device.isMps() or !matrix.device.isMps() or !vector.device.sameDevice(matrix.device)) return null;
    if (vector.shape.len != 1 or matrix.shape.len != 3 or !vector.isContiguous() or !matrix.isContiguous()) return null;
    if (vector.shape[0] != matrix.shape[1]) return null;
    const vector_storage = vector.device_storage orelse return null;
    const matrix_storage = matrix.device_storage orelse return null;
    const batch = matrix.shape[0];
    const k = matrix.shape[1];
    const n = matrix.shape[2];

    var out = try array_mod.Array(f16).emptyOn(vector.allocator, &.{ batch, n }, vector.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(vector.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runBatchedVecmatF16(
        .{ .ptr = vector_storage.ptr, .bytes = vector_storage.bytes },
        .{ .ptr = matrix_storage.ptr, .bytes = matrix_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        batch,
        k,
        n,
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryBatchedMatvecBF16(matrix: array_mod.Array(array_mod.BFloat16), vector: array_mod.Array(array_mod.BFloat16)) array_mod.ArrayError!?array_mod.Array(array_mod.BFloat16) {
    if (!matrix.device.isMps() or !vector.device.isMps() or !matrix.device.sameDevice(vector.device)) return null;
    if (matrix.shape.len != 3 or vector.shape.len != 1 or !matrix.isContiguous() or !vector.isContiguous()) return null;
    if (matrix.shape[2] != vector.shape[0]) return null;
    const matrix_storage = matrix.device_storage orelse return null;
    const vector_storage = vector.device_storage orelse return null;
    const batch = matrix.shape[0];
    const m = matrix.shape[1];
    const k = matrix.shape[2];

    var out = try array_mod.Array(array_mod.BFloat16).emptyOn(matrix.allocator, &.{ batch, m }, matrix.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(matrix.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runBatchedMatvecBF16(
        .{ .ptr = matrix_storage.ptr, .bytes = matrix_storage.bytes },
        .{ .ptr = vector_storage.ptr, .bytes = vector_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        batch,
        m,
        k,
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryBatchedVecmatBF16(vector: array_mod.Array(array_mod.BFloat16), matrix: array_mod.Array(array_mod.BFloat16)) array_mod.ArrayError!?array_mod.Array(array_mod.BFloat16) {
    if (!vector.device.isMps() or !matrix.device.isMps() or !vector.device.sameDevice(matrix.device)) return null;
    if (vector.shape.len != 1 or matrix.shape.len != 3 or !vector.isContiguous() or !matrix.isContiguous()) return null;
    if (vector.shape[0] != matrix.shape[1]) return null;
    const vector_storage = vector.device_storage orelse return null;
    const matrix_storage = matrix.device_storage orelse return null;
    const batch = matrix.shape[0];
    const k = matrix.shape[1];
    const n = matrix.shape[2];

    var out = try array_mod.Array(array_mod.BFloat16).emptyOn(vector.allocator, &.{ batch, n }, vector.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(vector.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runBatchedVecmatBF16(
        .{ .ptr = vector_storage.ptr, .bytes = vector_storage.bytes },
        .{ .ptr = matrix_storage.ptr, .bytes = matrix_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        batch,
        k,
        n,
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryMatmulAddF32(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32), addend: array_mod.Array(f32), alpha: f32, beta: f32) array_mod.ArrayError!?array_mod.Array(f32) {
    if (!lhs.device.isMps() or !rhs.device.isMps() or !addend.device.isMps()) return null;
    if (!lhs.device.sameDevice(rhs.device) or !lhs.device.sameDevice(addend.device)) return null;
    if (lhs.shape.len != 2 or rhs.shape.len != 2 or addend.shape.len != 2) return null;
    if (!lhs.isContiguous() or !rhs.isContiguous() or !addend.isContiguous()) return null;
    if (lhs.shape[1] != rhs.shape[0] or addend.shape[0] != lhs.shape[0] or addend.shape[1] != rhs.shape[1]) return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;
    const add_storage = addend.device_storage orelse return null;
    const m = lhs.shape[0];
    const k = lhs.shape[1];
    const n = rhs.shape[1];

    var out = try array_mod.Array(f32).emptyOn(lhs.allocator, addend.shape, lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(lhs.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runMatmulAddF32(
        .{ .ptr = lhs_storage.ptr, .bytes = lhs_storage.bytes },
        .{ .ptr = rhs_storage.ptr, .bytes = rhs_storage.bytes },
        .{ .ptr = add_storage.ptr, .bytes = add_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        m,
        k,
        n,
        alpha,
        beta,
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryMatmulAddF16(lhs: array_mod.Array(f16), rhs: array_mod.Array(f16), addend: array_mod.Array(f16), alpha: f32, beta: f32) array_mod.ArrayError!?array_mod.Array(f16) {
    if (!lhs.device.isMps() or !rhs.device.isMps() or !addend.device.isMps()) return null;
    if (!lhs.device.sameDevice(rhs.device) or !lhs.device.sameDevice(addend.device)) return null;
    if (lhs.shape.len != 2 or rhs.shape.len != 2 or addend.shape.len != 2) return null;
    if (!lhs.isContiguous() or !rhs.isContiguous() or !addend.isContiguous()) return null;
    if (lhs.shape[1] != rhs.shape[0] or addend.shape[0] != lhs.shape[0] or addend.shape[1] != rhs.shape[1]) return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;
    const add_storage = addend.device_storage orelse return null;
    const m = lhs.shape[0];
    const k = lhs.shape[1];
    const n = rhs.shape[1];

    var out = try array_mod.Array(f16).emptyOn(lhs.allocator, addend.shape, lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(lhs.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runMatmulAddF16(
        .{ .ptr = lhs_storage.ptr, .bytes = lhs_storage.bytes },
        .{ .ptr = rhs_storage.ptr, .bytes = rhs_storage.bytes },
        .{ .ptr = add_storage.ptr, .bytes = add_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        m,
        k,
        n,
        alpha,
        beta,
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryMatmulAddBF16(lhs: array_mod.Array(array_mod.BFloat16), rhs: array_mod.Array(array_mod.BFloat16), addend: array_mod.Array(array_mod.BFloat16), alpha: f32, beta: f32) array_mod.ArrayError!?array_mod.Array(array_mod.BFloat16) {
    if (!lhs.device.isMps() or !rhs.device.isMps() or !addend.device.isMps()) return null;
    if (!lhs.device.sameDevice(rhs.device) or !lhs.device.sameDevice(addend.device)) return null;
    if (lhs.shape.len != 2 or rhs.shape.len != 2 or addend.shape.len != 2) return null;
    if (!lhs.isContiguous() or !rhs.isContiguous() or !addend.isContiguous()) return null;
    if (lhs.shape[1] != rhs.shape[0] or addend.shape[0] != lhs.shape[0] or addend.shape[1] != rhs.shape[1]) return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;
    const add_storage = addend.device_storage orelse return null;
    const m = lhs.shape[0];
    const k = lhs.shape[1];
    const n = rhs.shape[1];

    var out = try array_mod.Array(array_mod.BFloat16).emptyOn(lhs.allocator, addend.shape, lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(lhs.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runMatmulAddBF16(
        .{ .ptr = lhs_storage.ptr, .bytes = lhs_storage.bytes },
        .{ .ptr = rhs_storage.ptr, .bytes = rhs_storage.bytes },
        .{ .ptr = add_storage.ptr, .bytes = add_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        m,
        k,
        n,
        alpha,
        beta,
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryTransposeF32(input: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    if (!input.device.isMps() or input.shape.len != 2 or !input.isContiguous()) return null;
    const input_storage = input.device_storage orelse return null;
    const rows = input.shape[0];
    const cols = input.shape[1];

    var out = try array_mod.Array(f32).emptyOn(input.allocator, &.{ cols, rows }, input.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(input.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runTransposeF32(
        .{ .ptr = input_storage.ptr, .bytes = input_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        rows,
        cols,
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryBroadcastBinaryF32(op: axiom.accelerator.MpsBinaryOp, input: array_mod.Array(f32), bias: array_mod.Array(f32), axis: axiom.accelerator.DialectBroadcastAxis) array_mod.ArrayError!?array_mod.Array(f32) {
    if (!input.device.isMps() or !bias.device.isMps() or !input.device.sameDevice(bias.device)) return null;
    if (input.shape.len != 2 or !input.isContiguous() or !bias.isContiguous()) return null;
    const input_storage = input.device_storage orelse return null;
    const bias_storage = bias.device_storage orelse return null;
    const rows = input.shape[0];
    const cols = input.shape[1];
    const expected_bias = switch (axis) {
        .row => cols,
        .column => rows,
    };
    if (bias.numel() != 1 and bias.numel() != expected_bias) return null;

    var out = try array_mod.Array(f32).emptyOn(input.allocator, input.shape, input.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(input.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runBroadcastBinaryF32(
        op,
        .{ .ptr = input_storage.ptr, .bytes = input_storage.bytes },
        .{ .ptr = bias_storage.ptr, .bytes = bias_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        rows,
        cols,
        bias_storage.len,
        switch (axis) {
            .row => .row,
            .column => .column,
        },
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryBroadcastAddF32(input: array_mod.Array(f32), bias: array_mod.Array(f32), axis: axiom.accelerator.DialectBroadcastAxis) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryBroadcastBinaryF32(.add, input, bias, axis);
}

pub fn tryReductionF32(op: axiom.accelerator.MpsReductionOp, input: array_mod.Array(f32), axis: u1, keepdims: bool) array_mod.ArrayError!?array_mod.Array(f32) {
    if (!input.device.isMps() or input.shape.len != 2 or !input.isContiguous()) return null;
    const input_storage = input.device_storage orelse return null;
    const rows = input.shape[0];
    const cols = input.shape[1];
    var out_shape_storage: [2]usize = undefined;
    const out_shape = if (keepdims) shape: {
        out_shape_storage = if (axis == 0)
            .{ 1, cols }
        else
            .{ rows, 1 };
        break :shape out_shape_storage[0..2];
    } else shape: {
        out_shape_storage[0] = if (axis == 0) cols else rows;
        break :shape out_shape_storage[0..1];
    };

    var out = try array_mod.Array(f32).emptyOn(input.allocator, out_shape, input.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(input.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runReductionF32(
        op,
        .{ .ptr = input_storage.ptr, .bytes = input_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        rows,
        cols,
        axis,
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryTransposeF16(input: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    if (!input.device.isMps() or input.shape.len != 2 or !input.isContiguous()) return null;
    const input_storage = input.device_storage orelse return null;
    const rows = input.shape[0];
    const cols = input.shape[1];

    var out = try array_mod.Array(f16).emptyOn(input.allocator, &.{ cols, rows }, input.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(input.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runTransposeF16(
        .{ .ptr = input_storage.ptr, .bytes = input_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        rows,
        cols,
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryBroadcastBinaryF16(op: axiom.accelerator.MpsBinaryOp, input: array_mod.Array(f16), bias: array_mod.Array(f16), axis: axiom.accelerator.DialectBroadcastAxis) array_mod.ArrayError!?array_mod.Array(f16) {
    if (!input.device.isMps() or !bias.device.isMps() or !input.device.sameDevice(bias.device)) return null;
    if (input.shape.len != 2 or !input.isContiguous() or !bias.isContiguous()) return null;
    const input_storage = input.device_storage orelse return null;
    const bias_storage = bias.device_storage orelse return null;
    const rows = input.shape[0];
    const cols = input.shape[1];
    const expected_bias = switch (axis) {
        .row => cols,
        .column => rows,
    };
    if (bias.numel() != 1 and bias.numel() != expected_bias) return null;

    var out = try array_mod.Array(f16).emptyOn(input.allocator, input.shape, input.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(input.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runBroadcastBinaryF16(
        op,
        .{ .ptr = input_storage.ptr, .bytes = input_storage.bytes },
        .{ .ptr = bias_storage.ptr, .bytes = bias_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        rows,
        cols,
        bias_storage.len,
        switch (axis) {
            .row => .row,
            .column => .column,
        },
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryBroadcastAddF16(input: array_mod.Array(f16), bias: array_mod.Array(f16), axis: axiom.accelerator.DialectBroadcastAxis) array_mod.ArrayError!?array_mod.Array(f16) {
    return tryBroadcastBinaryF16(.add, input, bias, axis);
}

pub fn tryReductionF16(op: axiom.accelerator.MpsReductionOp, input: array_mod.Array(f16), axis: u1, keepdims: bool) array_mod.ArrayError!?array_mod.Array(f16) {
    if (!input.device.isMps() or input.shape.len != 2 or !input.isContiguous()) return null;
    const input_storage = input.device_storage orelse return null;
    const rows = input.shape[0];
    const cols = input.shape[1];
    var out_shape_storage: [2]usize = undefined;
    const out_shape = if (keepdims) shape: {
        out_shape_storage = if (axis == 0)
            .{ 1, cols }
        else
            .{ rows, 1 };
        break :shape out_shape_storage[0..2];
    } else shape: {
        out_shape_storage[0] = if (axis == 0) cols else rows;
        break :shape out_shape_storage[0..1];
    };

    var out = try array_mod.Array(f16).emptyOn(input.allocator, out_shape, input.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(input.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runReductionF16(
        op,
        .{ .ptr = input_storage.ptr, .bytes = input_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        rows,
        cols,
        axis,
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryTransposeBF16(input: array_mod.Array(array_mod.BFloat16)) array_mod.ArrayError!?array_mod.Array(array_mod.BFloat16) {
    if (!input.device.isMps() or input.shape.len != 2 or !input.isContiguous()) return null;
    const input_storage = input.device_storage orelse return null;
    const rows = input.shape[0];
    const cols = input.shape[1];

    var out = try array_mod.Array(array_mod.BFloat16).emptyOn(input.allocator, &.{ cols, rows }, input.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(input.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runTransposeBF16(
        .{ .ptr = input_storage.ptr, .bytes = input_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        rows,
        cols,
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryBroadcastBinaryBF16(op: axiom.accelerator.MpsBinaryOp, input: array_mod.Array(array_mod.BFloat16), bias: array_mod.Array(array_mod.BFloat16), axis: axiom.accelerator.DialectBroadcastAxis) array_mod.ArrayError!?array_mod.Array(array_mod.BFloat16) {
    if (!input.device.isMps() or !bias.device.isMps() or !input.device.sameDevice(bias.device)) return null;
    if (input.shape.len != 2 or !input.isContiguous() or !bias.isContiguous()) return null;
    const input_storage = input.device_storage orelse return null;
    const bias_storage = bias.device_storage orelse return null;
    const rows = input.shape[0];
    const cols = input.shape[1];
    const expected_bias = switch (axis) {
        .row => cols,
        .column => rows,
    };
    if (bias.numel() != 1 and bias.numel() != expected_bias) return null;

    var out = try array_mod.Array(array_mod.BFloat16).emptyOn(input.allocator, input.shape, input.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(input.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runBroadcastBinaryBF16(
        op,
        .{ .ptr = input_storage.ptr, .bytes = input_storage.bytes },
        .{ .ptr = bias_storage.ptr, .bytes = bias_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        rows,
        cols,
        bias_storage.len,
        switch (axis) {
            .row => .row,
            .column => .column,
        },
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn tryBroadcastAddBF16(input: array_mod.Array(array_mod.BFloat16), bias: array_mod.Array(array_mod.BFloat16), axis: axiom.accelerator.DialectBroadcastAxis) array_mod.ArrayError!?array_mod.Array(array_mod.BFloat16) {
    return tryBroadcastBinaryBF16(.add, input, bias, axis);
}

pub fn tryReductionBF16(op: axiom.accelerator.MpsReductionOp, input: array_mod.Array(array_mod.BFloat16), axis: u1, keepdims: bool) array_mod.ArrayError!?array_mod.Array(array_mod.BFloat16) {
    if (!input.device.isMps() or input.shape.len != 2 or !input.isContiguous()) return null;
    const input_storage = input.device_storage orelse return null;
    const rows = input.shape[0];
    const cols = input.shape[1];
    var out_shape_storage: [2]usize = undefined;
    const out_shape = if (keepdims) shape: {
        out_shape_storage = if (axis == 0)
            .{ 1, cols }
        else
            .{ rows, 1 };
        break :shape out_shape_storage[0..2];
    } else shape: {
        out_shape_storage[0] = if (axis == 0) cols else rows;
        break :shape out_shape_storage[0..1];
    };

    var out = try array_mod.Array(array_mod.BFloat16).emptyOn(input.allocator, out_shape, input.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(input.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runReductionBF16(
        op,
        .{ .ptr = input_storage.ptr, .bytes = input_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        rows,
        cols,
        axis,
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn trySoftmaxF32(op: axiom.accelerator.MpsSoftmaxOp, input: array_mod.Array(f32), axis: u1) array_mod.ArrayError!?array_mod.Array(f32) {
    if (!input.device.isMps() or input.shape.len != 2 or !input.isContiguous()) return null;
    const input_storage = input.device_storage orelse return null;
    const rows = input.shape[0];
    const cols = input.shape[1];

    var out = try array_mod.Array(f32).emptyOn(input.allocator, input.shape, input.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(input.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runSoftmaxF32(
        op,
        .{ .ptr = input_storage.ptr, .bytes = input_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        rows,
        cols,
        axis,
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn trySoftmaxF16(op: axiom.accelerator.MpsSoftmaxOp, input: array_mod.Array(f16), axis: u1) array_mod.ArrayError!?array_mod.Array(f16) {
    if (!input.device.isMps() or input.shape.len != 2 or !input.isContiguous()) return null;
    const input_storage = input.device_storage orelse return null;
    const rows = input.shape[0];
    const cols = input.shape[1];

    var out = try array_mod.Array(f16).emptyOn(input.allocator, input.shape, input.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(input.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runSoftmaxF16(
        op,
        .{ .ptr = input_storage.ptr, .bytes = input_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        rows,
        cols,
        axis,
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}

pub fn trySoftmaxBF16(op: axiom.accelerator.MpsSoftmaxOp, input: array_mod.Array(array_mod.BFloat16), axis: u1) array_mod.ArrayError!?array_mod.Array(array_mod.BFloat16) {
    if (!input.device.isMps() or input.shape.len != 2 or !input.isContiguous()) return null;
    const input_storage = input.device_storage orelse return null;
    const rows = input.shape[0];
    const cols = input.shape[1];

    var out = try array_mod.Array(array_mod.BFloat16).emptyOn(input.allocator, input.shape, input.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.MpsRuntime.open(input.device.index) catch {
        out.deinit();
        return null;
    };
    defer runtime.close();
    runtime.runSoftmaxBF16(
        op,
        .{ .ptr = input_storage.ptr, .bytes = input_storage.bytes },
        .{ .ptr = out_storage.ptr, .bytes = out_storage.bytes },
        rows,
        cols,
        axis,
    ) catch {
        out.deinit();
        return null;
    };
    return out;
}
