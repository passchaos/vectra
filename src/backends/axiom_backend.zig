//! Unified Axiom backend policy for Vectra.
//!
//! This small policy layer makes CPU-via-Axiom/Veyra and CUDA-via-Axiom visible
//! as data so callers can audit which route was selected before Vectra grows a
//! persistent `.cuda()` storage backend.

const std = @import("std");
const build_options = @import("vectra_build_options");
const array_mod = @import("../array.zig");
const axiom_cpu = @import("axiom_cpu.zig");
const axiom_cuda = @import("axiom_cuda.zig");

pub const BackendRoute = enum(u8) {
    direct_cpu,
    axiom_cpu_veyra,
    axiom_cuda,

    pub fn label(route: BackendRoute) []const u8 {
        return @tagName(route);
    }
};

pub const BackendPolicy = enum(u8) {
    prefer_cuda,
    prefer_axiom_cpu,
    force_direct_cpu,

    pub fn label(policy: BackendPolicy) []const u8 {
        return @tagName(policy);
    }
};

pub const BackendReport = struct {
    policy: BackendPolicy = .prefer_cuda,
    selected: BackendRoute = .direct_cpu,
    axiom_cpu_enabled: bool = build_options.enable_axiom_cpu_dispatch,
    axiom_cuda_enabled: bool = build_options.enable_axiom_cuda,
    dtype_name: []const u8 = "",
    supported_shape: bool = false,
    fingerprint_value: u64 = 0,

    pub fn ok(report: BackendReport) bool {
        return report.dtype_name.len != 0 and report.supported_shape and report.fingerprint_value != 0;
    }

    pub fn fingerprint(report: BackendReport) u64 {
        var hasher = std.hash.Wyhash.init(0x0abc_beef_0001);
        hashBytes(&hasher, report.policy.label());
        hashBytes(&hasher, report.selected.label());
        hashBool(&hasher, report.axiom_cpu_enabled);
        hashBool(&hasher, report.axiom_cuda_enabled);
        hashBytes(&hasher, report.dtype_name);
        hashBool(&hasher, report.supported_shape);
        hashU64(&hasher, report.fingerprint_value);
        return hasher.final();
    }
};

pub fn selectMatmul(comptime T: type, policy: BackendPolicy, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) BackendReport {
    const supported = supportedMatmul2d(T, lhs, rhs);
    const selected: BackendRoute = if (!supported)
        .direct_cpu
    else switch (policy) {
        .force_direct_cpu => .direct_cpu,
        .prefer_axiom_cpu => if (axiom_cpu.enabled()) .axiom_cpu_veyra else .direct_cpu,
        .prefer_cuda => if (T == f32 and axiom_cuda.enabled()) .axiom_cuda else if (axiom_cpu.enabled()) .axiom_cpu_veyra else .direct_cpu,
    };
    var report: BackendReport = .{
        .policy = policy,
        .selected = selected,
        .dtype_name = @typeName(T),
        .supported_shape = supported,
    };
    report.fingerprint_value = computeShapeFingerprint(T, lhs, rhs, selected);
    return report;
}

pub fn matmul(comptime T: type, policy: BackendPolicy, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!array_mod.Array(T) {
    const report = selectMatmul(T, policy, lhs, rhs);
    switch (report.selected) {
        .axiom_cuda => if (T == f32) {
            const out = try axiom_cuda.tryMatmulF32(@as(array_mod.Array(f32), lhs), @as(array_mod.Array(f32), rhs));
            if (out) |value| return @as(array_mod.Array(T), value);
        },
        .axiom_cpu_veyra => {
            const out = if (T == f32)
                try axiom_cpu.tryMatmulF32(@as(array_mod.Array(f32), lhs), @as(array_mod.Array(f32), rhs))
            else if (T == f64)
                try axiom_cpu.tryMatmulF64(@as(array_mod.Array(f64), lhs), @as(array_mod.Array(f64), rhs))
            else
                null;
            if (out) |value| return @as(array_mod.Array(T), value);
        },
        .direct_cpu => {},
    }
    return lhs.matmul(rhs);
}

fn supportedMatmul2d(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) bool {
    return lhs.device.isCpu() and rhs.device.isCpu() and lhs.shape.len == 2 and rhs.shape.len == 2 and lhs.shape[1] == rhs.shape[0] and lhs.isContiguous() and rhs.isContiguous();
}

fn computeShapeFingerprint(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T), selected: BackendRoute) u64 {
    var hasher = std.hash.Wyhash.init(0x0abc_beef_0002);
    hashBytes(&hasher, @typeName(T));
    hashBytes(&hasher, selected.label());
    for (lhs.shape) |dim| hashU64(&hasher, dim);
    for (rhs.shape) |dim| hashU64(&hasher, dim);
    return hasher.final();
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

test "Axiom backend policy reports matmul route" {
    const gpa = std.testing.allocator;
    var a = try array_mod.Array(f32).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a.deinit();
    var b = try array_mod.Array(f32).fromSlice(gpa, &.{ 7, 8, 9, 10, 11, 12 }, &.{ 3, 2 });
    defer b.deinit();
    const report = selectMatmul(f32, .prefer_cuda, a, b);
    try std.testing.expect(report.ok());
    try std.testing.expect(report.fingerprint() != 0);
    var out = try matmul(f32, .prefer_axiom_cpu, a, b);
    defer out.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 58, 64, 139, 154 }, out.data);
}
