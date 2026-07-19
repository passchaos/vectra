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
