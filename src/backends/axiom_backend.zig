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

pub const ElementwiseOp = enum(u8) {
    add,
    sub,
    mul,
    div,

    pub fn label(op: ElementwiseOp) []const u8 {
        return @tagName(op);
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

pub fn selectElementwise(comptime T: type, op: ElementwiseOp, policy: BackendPolicy, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) BackendReport {
    const supported = supportedElementwiseSameShapeContiguous(T, lhs, rhs);
    const selected: BackendRoute = if (!supported)
        .direct_cpu
    else switch (policy) {
        .force_direct_cpu => .direct_cpu,
        .prefer_axiom_cpu => if (supportsAxiomCpuElementwise(T) and axiom_cpu.enabled()) .axiom_cpu_veyra else .direct_cpu,
        .prefer_cuda => if (T == f32 and axiom_cuda.enabled()) .axiom_cuda else if (supportsAxiomCpuElementwise(T) and axiom_cpu.enabled()) .axiom_cpu_veyra else .direct_cpu,
    };
    var report: BackendReport = .{
        .policy = policy,
        .selected = selected,
        .dtype_name = @typeName(T),
        .supported_shape = supported,
    };
    report.fingerprint_value = computeElementwiseFingerprint(T, op, lhs, rhs, selected);
    return report;
}

pub fn elementwise(comptime T: type, op: ElementwiseOp, policy: BackendPolicy, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!array_mod.Array(T) {
    const report = selectElementwise(T, op, policy, lhs, rhs);
    switch (report.selected) {
        .axiom_cuda => if (T == f32) {
            const out = switch (op) {
                .add => try axiom_cuda.tryAddF32(@as(array_mod.Array(f32), lhs), @as(array_mod.Array(f32), rhs)),
                .sub => try axiom_cuda.trySubF32(@as(array_mod.Array(f32), lhs), @as(array_mod.Array(f32), rhs)),
                .mul => try axiom_cuda.tryMulF32(@as(array_mod.Array(f32), lhs), @as(array_mod.Array(f32), rhs)),
                .div => try axiom_cuda.tryDivF32(@as(array_mod.Array(f32), lhs), @as(array_mod.Array(f32), rhs)),
            };
            if (out) |value| return @as(array_mod.Array(T), value);
        },
        .axiom_cpu_veyra => {
            const out = if (T == f32) switch (op) {
                .add => try axiom_cpu.tryAddF32(@as(array_mod.Array(f32), lhs), @as(array_mod.Array(f32), rhs)),
                .sub => try axiom_cpu.trySubF32(@as(array_mod.Array(f32), lhs), @as(array_mod.Array(f32), rhs)),
                .mul => try axiom_cpu.tryMulF32(@as(array_mod.Array(f32), lhs), @as(array_mod.Array(f32), rhs)),
                .div => try axiom_cpu.tryDivF32(@as(array_mod.Array(f32), lhs), @as(array_mod.Array(f32), rhs)),
            } else if (T == f64) switch (op) {
                .add => try axiom_cpu.tryAddF64(@as(array_mod.Array(f64), lhs), @as(array_mod.Array(f64), rhs)),
                .sub => try axiom_cpu.trySubF64(@as(array_mod.Array(f64), lhs), @as(array_mod.Array(f64), rhs)),
                .mul => try axiom_cpu.tryMulF64(@as(array_mod.Array(f64), lhs), @as(array_mod.Array(f64), rhs)),
                .div => try axiom_cpu.tryDivF64(@as(array_mod.Array(f64), lhs), @as(array_mod.Array(f64), rhs)),
            } else null;
            if (out) |value| return @as(array_mod.Array(T), value);
        },
        .direct_cpu => {},
    }
    return directElementwise(T, op, lhs, rhs);
}

fn directElementwise(comptime T: type, op: ElementwiseOp, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!array_mod.Array(T) {
    if (!std.mem.eql(usize, lhs.shape, rhs.shape)) return error.ShapeMismatch;
    var out = try array_mod.Array(T).empty(lhs.allocator, lhs.shape);
    errdefer out.deinit();
    for (lhs.data, rhs.data, out.data) |a, b, *slot| slot.* = switch (op) {
        .add => a + b,
        .sub => a - b,
        .mul => a * b,
        .div => a / b,
    };
    return out;
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
    return directMatmul(T, lhs, rhs);
}

fn directMatmul(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!array_mod.Array(T) {
    if (!supportedMatmul2d(T, lhs, rhs)) return error.NonMatrixArray;
    var out = try array_mod.Array(T).zeros(lhs.allocator, &.{ lhs.shape[0], rhs.shape[1] });
    errdefer out.deinit();
    for (0..lhs.shape[0]) |row| {
        for (0..rhs.shape[1]) |col| {
            var acc: T = 0;
            for (0..lhs.shape[1]) |kk| {
                acc += lhs.data[row * lhs.shape[1] + kk] * rhs.data[kk * rhs.shape[1] + col];
            }
            out.data[row * rhs.shape[1] + col] = acc;
        }
    }
    return out;
}

fn supportedMatmul2d(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) bool {
    return lhs.device.isCpu() and rhs.device.isCpu() and lhs.shape.len == 2 and rhs.shape.len == 2 and lhs.shape[1] == rhs.shape[0] and lhs.isContiguous() and rhs.isContiguous();
}

fn supportedElementwiseSameShapeContiguous(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) bool {
    return supportsAxiomCpuElementwise(T) and
        lhs.device.isCpu() and
        rhs.device.isCpu() and
        lhs.data.len != 0 and
        lhs.sameShape(rhs) and
        lhs.isContiguous() and
        rhs.isContiguous();
}

fn supportsAxiomCpuElementwise(comptime T: type) bool {
    return T == f32 or T == f64;
}

fn computeShapeFingerprint(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T), selected: BackendRoute) u64 {
    var hasher = std.hash.Wyhash.init(0x0abc_beef_0002);
    hashBytes(&hasher, @typeName(T));
    hashBytes(&hasher, selected.label());
    for (lhs.shape) |dim| hashU64(&hasher, dim);
    for (rhs.shape) |dim| hashU64(&hasher, dim);
    return hasher.final();
}

fn computeElementwiseFingerprint(comptime T: type, op: ElementwiseOp, lhs: array_mod.Array(T), rhs: array_mod.Array(T), selected: BackendRoute) u64 {
    var hasher = std.hash.Wyhash.init(0x0abc_beef_0003);
    hashBytes(&hasher, @typeName(T));
    hashBytes(&hasher, op.label());
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

test "Axiom backend policy reports elementwise route" {
    const gpa = std.testing.allocator;
    var lhs32 = try array_mod.Array(f32).fromSlice(gpa, &.{ 1, 2, 3, 4 }, &.{4});
    defer lhs32.deinit();
    var rhs32 = try array_mod.Array(f32).fromSlice(gpa, &.{ 10, 20, 30, 40 }, &.{4});
    defer rhs32.deinit();
    const add_report = selectElementwise(f32, .add, .prefer_cuda, lhs32, rhs32);
    try std.testing.expect(add_report.ok());
    try std.testing.expect(add_report.fingerprint() != 0);
    var add_out = try elementwise(f32, .add, .prefer_cuda, lhs32, rhs32);
    defer add_out.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 33, 44 }, add_out.data);

    var lhs64 = try array_mod.Array(f64).fromSlice(gpa, &.{ 8, 6, 4, 2 }, &.{4});
    defer lhs64.deinit();
    var rhs64 = try array_mod.Array(f64).fromSlice(gpa, &.{ 2, 3, 4, 2 }, &.{4});
    defer rhs64.deinit();
    const div_report = selectElementwise(f64, .div, .prefer_axiom_cpu, lhs64, rhs64);
    try std.testing.expect(div_report.ok());
    if (build_options.enable_axiom_cpu_dispatch) {
        try std.testing.expectEqual(BackendRoute.axiom_cpu_veyra, div_report.selected);
    } else {
        try std.testing.expectEqual(BackendRoute.direct_cpu, div_report.selected);
    }
    var div_out = try elementwise(f64, .div, .prefer_axiom_cpu, lhs64, rhs64);
    defer div_out.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 4, 2, 1, 1 }, div_out.data);
}
