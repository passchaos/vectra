//! Axial CUDA acceleration bridge for Vectra.
//!
//! Vectra owns Array/NDArray storage and dtype/device metadata.  Axial owns the
//! CUDA-like host+kernel facade and reusable compute-route policy.  This bridge
//! is intentionally thin: it converts Vectra CUDA-owning Array storage into
//! Axial device views, asks Axial to lower/launch through Axiom, and falls back
//! to existing Vectra/Axiom paths when the CUDA runtime is unavailable.

const std = @import("std");
const build_options = @import("vectra_build_options");
pub const axial = @import("axial");
const array_mod = @import("../array.zig");

pub const Status = enum(u8) {
    disabled,
    planned,
    unavailable,
    launched,

    pub fn label(status: Status) []const u8 {
        return @tagName(status);
    }
};

pub const RouteKind = enum(u8) {
    cuda_host_kernel_facade,
    device_elementwise,
    device_saxpy,
    device_gemm,
    device_gemm_add,

    pub fn label(route: RouteKind) []const u8 {
        return @tagName(route);
    }
};

pub const AccelerationReport = struct {
    route: RouteKind = .cuda_host_kernel_facade,
    status: Status = if (build_options.enable_axial_acceleration) .planned else .disabled,
    dtype_name: []const u8 = "",
    device_ordinal: usize = 0,
    logical_elements: usize = 0,
    lhs_fingerprint: u64 = 0,
    rhs_fingerprint: u64 = 0,
    out_fingerprint: u64 = 0,
    axial_fingerprint: u64 = 0,
    issue_count: u8 = 0,

    pub fn ok(report: AccelerationReport) bool {
        return report.status == .launched and
            report.issue_count == 0 and
            report.route.label().len != 0 and
            report.dtype_name.len != 0 and
            report.logical_elements != 0 and
            report.axial_fingerprint != 0;
    }

    pub fn fingerprint(report: AccelerationReport) u64 {
        var hasher = std.hash.Wyhash.init(0x0abc_7aaa_a21a_0001);
        hashBool(&hasher, report.ok());
        hashBytes(&hasher, report.route.label());
        hashBytes(&hasher, report.status.label());
        hashBytes(&hasher, report.dtype_name);
        hashU64(&hasher, report.device_ordinal);
        hashU64(&hasher, report.logical_elements);
        hashU64(&hasher, report.lhs_fingerprint);
        hashU64(&hasher, report.rhs_fingerprint);
        hashU64(&hasher, report.out_fingerprint);
        hashU64(&hasher, report.axial_fingerprint);
        hashU64(&hasher, report.issue_count);
        return hasher.final();
    }
};

threadlocal var last_report: AccelerationReport = .{};

pub fn resetLastReport() void {
    last_report = .{};
}

pub fn lastReport() AccelerationReport {
    return last_report;
}

pub fn enabled() bool {
    return build_options.enable_axial_acceleration;
}

pub fn saxpyKernelFingerprint() u64 {
    const spec = axial.cuda.saxpyKernel(128, "sm_89") catch return 0;
    return spec.fingerprint();
}

pub fn saxpyLaunchFingerprint(x_ptr: u64, y_ptr: u64, n: i32, alpha: f32) u64 {
    const launch = axial.cuda.saxpyLaunch(128, "sm_89", x_ptr, y_ptr, n, alpha) catch return 0;
    return launch.fingerprint();
}

pub fn tryDeviceBinaryF32(op: axial.cuda.BinaryOp, lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    if (!enabled() or !sameCudaShape(lhs, rhs)) return null;
    var out = try array_mod.Array(f32).emptyOn(lhs.allocator, lhs.shape, lhs.device);
    errdefer out.deinit();
    const lhs_view = tensorView(lhs) orelse return null;
    const rhs_view = tensorView(rhs) orelse return null;
    const out_view = tensorView(out) orelse return null;
    const axial_report = axial.cuda.runDeviceElementwiseF32(lhs.allocator, op, lhs_view, rhs_view, out_view) catch |err| switch (err) {
        error.RuntimeUnavailable, error.RuntimeFailure => return null,
        error.OutOfMemory => return error.OutOfMemory,
        else => return null,
    };
    record(.device_elementwise, axial_report, lhs.dtypeName(), lhs_view, rhs_view, out_view);
    if (!axial_report.ok()) return null;
    return out;
}

pub fn tryDeviceSaxpyF32(alpha: f32, x: array_mod.Array(f32), y: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    if (!enabled() or !sameCudaShape(x, y)) return null;
    var out = try y.clone();
    errdefer out.deinit();
    const x_view = tensorView(x) orelse return null;
    const out_view = tensorView(out) orelse return null;
    const axial_report = axial.cuda.runDeviceSaxpyF32(x.allocator, x_view, out_view, alpha) catch |err| switch (err) {
        error.RuntimeUnavailable, error.RuntimeFailure => return null,
        error.OutOfMemory => return error.OutOfMemory,
        else => return null,
    };
    record(.device_saxpy, axial_report, x.dtypeName(), x_view, out_view, out_view);
    if (!axial_report.ok()) return null;
    return out;
}

pub fn tryDeviceMatmulF32(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    if (!enabled() or !matrixCudaCompatible(lhs, rhs)) return null;
    var out = try array_mod.Array(f32).emptyOn(lhs.allocator, &.{ lhs.shape[0], rhs.shape[1] }, lhs.device);
    errdefer out.deinit();
    const lhs_view = matrixView(lhs) orelse return null;
    const rhs_view = matrixView(rhs) orelse return null;
    const out_view = matrixView(out) orelse return null;
    const axial_report = axial.cuda.runDeviceGemmF32(lhs.allocator, lhs_view, rhs_view, out_view, 1.0, 0.0) catch |err| switch (err) {
        error.RuntimeUnavailable, error.RuntimeFailure => return null,
        error.OutOfMemory => return error.OutOfMemory,
        else => return null,
    };
    recordMatrix(.device_gemm, axial_report, lhs.dtypeName(), lhs_view, rhs_view, out_view);
    if (!axial_report.ok()) return null;
    return out;
}

pub fn tryDeviceMatmulAddF32(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32), addend: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    if (!enabled() or !matrixCudaCompatible(lhs, rhs) or !lhs.device.sameDevice(addend.device) or addend.shape.len != 2 or addend.shape[0] != lhs.shape[0] or addend.shape[1] != rhs.shape[1]) return null;
    var out = try array_mod.Array(f32).emptyOn(lhs.allocator, addend.shape, lhs.device);
    errdefer out.deinit();
    const lhs_view = matrixView(lhs) orelse return null;
    const rhs_view = matrixView(rhs) orelse return null;
    const add_view = matrixView(addend) orelse return null;
    const out_view = matrixView(out) orelse return null;
    const axial_report = axial.cuda.runDeviceGemmAddF32(lhs.allocator, lhs_view, rhs_view, add_view, out_view, 1.0, 1.0) catch |err| switch (err) {
        error.RuntimeUnavailable, error.RuntimeFailure => return null,
        error.OutOfMemory => return error.OutOfMemory,
        else => return null,
    };
    recordMatrix(.device_gemm_add, axial_report, lhs.dtypeName(), lhs_view, rhs_view, out_view);
    if (!axial_report.ok()) return null;
    return out;
}

pub fn runPendingMatmulF32(
    allocator: std.mem.Allocator,
    device: array_mod.Device,
    m: usize,
    n: usize,
    k: usize,
    lhs_ptr: u64,
    rhs_ptr: u64,
    out_ptr: u64,
    alpha: f32,
    beta: f32,
) array_mod.ArrayError!bool {
    if (!enabled() or !device.isCuda()) return false;
    const lhs: axial.cuda.MatrixView = .{ .device_ordinal = device.index, .device_ptr = lhs_ptr, .rows = m, .cols = k };
    const rhs: axial.cuda.MatrixView = .{ .device_ordinal = device.index, .device_ptr = rhs_ptr, .rows = k, .cols = n };
    const out: axial.cuda.MatrixView = .{ .device_ordinal = device.index, .device_ptr = out_ptr, .rows = m, .cols = n };
    const axial_report = axial.cuda.runDeviceGemmF32(allocator, lhs, rhs, out, alpha, beta) catch |err| switch (err) {
        error.RuntimeUnavailable, error.RuntimeFailure => return false,
        error.OutOfMemory => return error.OutOfMemory,
        else => return false,
    };
    recordMatrix(.device_gemm, axial_report, "f32", lhs, rhs, out);
    return axial_report.ok();
}

pub fn runPendingMatmulAddF32(
    allocator: std.mem.Allocator,
    device: array_mod.Device,
    m: usize,
    n: usize,
    k: usize,
    lhs_ptr: u64,
    rhs_ptr: u64,
    add_ptr: u64,
    out_ptr: u64,
    alpha: f32,
    beta: f32,
) array_mod.ArrayError!bool {
    if (!enabled() or !device.isCuda()) return false;
    const lhs: axial.cuda.MatrixView = .{ .device_ordinal = device.index, .device_ptr = lhs_ptr, .rows = m, .cols = k };
    const rhs: axial.cuda.MatrixView = .{ .device_ordinal = device.index, .device_ptr = rhs_ptr, .rows = k, .cols = n };
    const addend: axial.cuda.MatrixView = .{ .device_ordinal = device.index, .device_ptr = add_ptr, .rows = m, .cols = n };
    const out: axial.cuda.MatrixView = .{ .device_ordinal = device.index, .device_ptr = out_ptr, .rows = m, .cols = n };
    const axial_report = axial.cuda.runDeviceGemmAddF32(allocator, lhs, rhs, addend, out, alpha, beta) catch |err| switch (err) {
        error.RuntimeUnavailable, error.RuntimeFailure => return false,
        error.OutOfMemory => return error.OutOfMemory,
        else => return false,
    };
    recordMatrix(.device_gemm_add, axial_report, "f32", lhs, rhs, out);
    return axial_report.ok();
}

fn sameCudaShape(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) bool {
    return lhs.device.isCuda() and lhs.device.sameDevice(rhs.device) and std.mem.eql(usize, lhs.shape, rhs.shape) and lhs.data.len == 0 and rhs.data.len == 0;
}

fn matrixCudaCompatible(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) bool {
    return lhs.device.isCuda() and lhs.device.sameDevice(rhs.device) and lhs.shape.len == 2 and rhs.shape.len == 2 and lhs.shape[1] == rhs.shape[0];
}

fn tensorView(array: array_mod.Array(f32)) ?axial.cuda.TensorView {
    const storage = array.device_storage orelse return null;
    if (!storage.isAllocated()) return null;
    return .{ .device_ordinal = array.device.index, .device_ptr = storage.ptr, .len = storage.len };
}

fn matrixView(array: array_mod.Array(f32)) ?axial.cuda.MatrixView {
    const storage = array.device_storage orelse return null;
    if (!storage.isAllocated() or array.shape.len != 2) return null;
    return .{ .device_ordinal = array.device.index, .device_ptr = storage.ptr, .rows = array.shape[0], .cols = array.shape[1] };
}

fn record(route: RouteKind, axial_report: axial.cuda.DeviceOpReport, dtype_name: []const u8, lhs: axial.cuda.TensorView, rhs: axial.cuda.TensorView, out: axial.cuda.TensorView) void {
    last_report = .{
        .route = route,
        .status = if (axial_report.ok()) .launched else .unavailable,
        .dtype_name = dtype_name,
        .device_ordinal = lhs.device_ordinal,
        .logical_elements = lhs.len,
        .lhs_fingerprint = lhs.fingerprint(),
        .rhs_fingerprint = rhs.fingerprint(),
        .out_fingerprint = out.fingerprint(),
        .axial_fingerprint = axial_report.fingerprint(),
        .issue_count = if (axial_report.ok()) 0 else 1,
    };
}

fn recordMatrix(route: RouteKind, axial_report: axial.cuda.DeviceOpReport, dtype_name: []const u8, lhs: axial.cuda.MatrixView, rhs: axial.cuda.MatrixView, out: axial.cuda.MatrixView) void {
    last_report = .{
        .route = route,
        .status = if (axial_report.ok()) .launched else .unavailable,
        .dtype_name = dtype_name,
        .device_ordinal = lhs.device_ordinal,
        .logical_elements = out.rows * out.cols,
        .lhs_fingerprint = lhs.fingerprint(),
        .rhs_fingerprint = rhs.fingerprint(),
        .out_fingerprint = out.fingerprint(),
        .axial_fingerprint = axial_report.fingerprint(),
        .issue_count = if (axial_report.ok()) 0 else 1,
    };
}

fn hashBytes(hasher: *std.hash.Wyhash, bytes: []const u8) void {
    var len_bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_bytes, bytes.len, .little);
    hasher.update(&len_bytes);
    hasher.update(bytes);
}

fn hashBool(hasher: *std.hash.Wyhash, value: bool) void {
    hasher.update(&[_]u8{if (value) 1 else 0});
}

fn hashU64(hasher: *std.hash.Wyhash, value: anytype) void {
    var bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &bytes, @intCast(value), .little);
    hasher.update(&bytes);
}

test "Axial CUDA bridge exposes comptime SAXPY metadata" {
    try std.testing.expect(saxpyKernelFingerprint() != 0);
    try std.testing.expect(saxpyLaunchFingerprint(0x1000, 0x2000, 64, 2.0) != 0);
}
