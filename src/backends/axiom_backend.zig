//! Unified Axiom target facade for Vectra.
//!
//! Vectra should describe array work and call Axiom with a target instead of
//! open-coding CPU/CUDA/MPS branches in Array methods.  This module is the
//! intentional seam: high-level code chooses `.cpu`, `.cuda`, or `.mps`, while
//! the per-target implementation details stay concentrated here until Axiom
//! grows a fully public execution ABI for every operation.

const std = @import("std");
const build_options = @import("vectra_build_options");
const array_mod = @import("../array.zig");
const axiom = @import("axiom");
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

pub const ScalarSide = enum(u8) {
    lhs,
    rhs,

    pub fn label(side: ScalarSide) []const u8 {
        return @tagName(side);
    }
};

pub const ExecutionUnaryOp = enum(u8) {
    abs,
    square,
    sqrt,
    exp,

    pub fn label(op: ExecutionUnaryOp) []const u8 {
        return @tagName(op);
    }
};

pub fn QrResult(comptime T: type) type {
    return struct {
        q: array_mod.Array(T),
        r: array_mod.Array(T),

        pub fn deinit(self: *@This()) void {
            self.q.deinit();
            self.r.deinit();
            self.* = undefined;
        }
    };
}

pub fn SvdResult(comptime T: type) type {
    return struct {
        u: array_mod.Array(T),
        s: array_mod.Array(T),
        vt: array_mod.Array(T),

        pub fn deinit(self: *@This()) void {
            self.u.deinit();
            self.s.deinit();
            self.vt.deinit();
            self.* = undefined;
        }
    };
}

pub fn EighResult(comptime T: type) type {
    return struct {
        values: array_mod.Array(T),
        vectors: array_mod.Array(T),

        pub fn deinit(self: *@This()) void {
            self.values.deinit();
            self.vectors.deinit();
            self.* = undefined;
        }
    };
}

pub fn LuResult(comptime T: type) type {
    return struct {
        p: array_mod.Array(T),
        l: array_mod.Array(T),
        u: array_mod.Array(T),

        pub fn deinit(self: *@This()) void {
            self.p.deinit();
            self.l.deinit();
            self.u.deinit();
            self.* = undefined;
        }
    };
}

pub const PendingMatmulSpec = struct {
    lhs_storage: array_mod.DeviceStorage,
    rhs_storage: array_mod.DeviceStorage,
    add_storage: ?array_mod.DeviceStorage = null,
    lhs_shape: [2]usize,
    rhs_shape: [2]usize,
    alpha: f32 = 1.0,
    beta: f32 = 0.0,
    unary: ?ExecutionUnaryOp = null,
};

pub const DialectBackend = axiom.accelerator.DialectBackend;
pub const DialectMatmulLoweringReport = axiom.accelerator.DialectMatmulLoweringReport;
pub const DialectMatmulLoweringStatus = axiom.accelerator.DialectMatmulLoweringStatus;
pub const DialectElementwiseOp = axiom.accelerator.DialectElementwiseOp;
pub const DialectElementwiseLoweringReport = axiom.accelerator.DialectElementwiseLoweringReport;
pub const DialectElementwiseLoweringStatus = axiom.accelerator.DialectElementwiseLoweringStatus;
pub const DialectReductionOp = axiom.accelerator.DialectReductionOp;
pub const DialectReductionLoweringReport = axiom.accelerator.DialectReductionLoweringReport;
pub const DialectReductionLoweringStatus = axiom.accelerator.DialectReductionLoweringStatus;
pub const DialectBroadcastAxis = axiom.accelerator.DialectBroadcastAxis;
pub const DialectBroadcastLoweringReport = axiom.accelerator.DialectBroadcastLoweringReport;
pub const DialectBroadcastLoweringStatus = axiom.accelerator.DialectBroadcastLoweringStatus;
pub const DialectUnaryOp = axiom.accelerator.DialectUnaryOp;
pub const DialectUnaryLoweringReport = axiom.accelerator.DialectUnaryLoweringReport;
pub const DialectUnaryLoweringStatus = axiom.accelerator.DialectUnaryLoweringStatus;
pub const DialectTransposeLoweringReport = axiom.accelerator.DialectTransposeLoweringReport;
pub const DialectTransposeLoweringStatus = axiom.accelerator.DialectTransposeLoweringStatus;
pub const MpsRuntimeAbiStatus = axiom.accelerator.MpsRuntimeAbiStatus;
pub const MpsRuntimeAbiReport = axiom.accelerator.MpsRuntimeAbiReport;

pub const RuntimeCapabilityStatus = enum(u8) {
    unavailable,
    lowering_only,
    executable,

    pub fn label(status: RuntimeCapabilityStatus) []const u8 {
        return @tagName(status);
    }
};

pub const RuntimeCapabilityReport = struct {
    target: DialectBackend,
    operation: []const u8,
    status: RuntimeCapabilityStatus,
    reason: []const u8,

    pub fn executable(report: RuntimeCapabilityReport) bool {
        return report.status == .executable;
    }

    pub fn fingerprint(report: RuntimeCapabilityReport) u64 {
        var hasher = std.hash.Wyhash.init(0x0c0a_b17e_0001);
        hashBytes(&hasher, @tagName(report.target));
        hashBytes(&hasher, report.operation);
        hashBytes(&hasher, report.status.label());
        hashBytes(&hasher, report.reason);
        return hasher.final();
    }
};

pub fn mpsDeviceReport(index: usize) MpsRuntimeAbiReport {
    return axiom.accelerator.mpsDeviceReport(index);
}

pub fn mpsDeviceAvailable(index: usize) bool {
    return axiom.accelerator.mpsDeviceAvailable(index);
}

pub fn deviceAvailable(device: array_mod.Device) bool {
    return switch (device.backend) {
        .cpu => true,
        .cuda => build_options.enable_axiom_cuda and axiom_cuda.deviceAvailable(device.index),
        .mps => mpsDeviceAvailable(device.index),
    };
}

pub fn allocateStorage(device: array_mod.Device, len: usize, element_size: usize) array_mod.ArrayError!?array_mod.DeviceStorage {
    return switch (executionTargetForDevice(device)) {
        .cpu => null,
        .cuda => axiom_cuda.allocateStorage(device, len, element_size),
        .mps => null,
    };
}

pub fn hostElementCapacity(device: array_mod.Device, len: usize) usize {
    return if (allocateStorageOptionalForDevice(device)) 0 else len;
}

fn allocateStorageOptionalForDevice(device: array_mod.Device) bool {
    return switch (executionTargetForDevice(device)) {
        .cpu => false,
        .cuda, .mps => true,
    };
}

pub fn freeStorage(storage: array_mod.DeviceStorage) void {
    switch (executionTargetForDevice(storage.device)) {
        .cpu => {},
        .cuda => axiom_cuda.freeStorage(storage),
        .mps => {},
    }
}

pub fn fillStorage(comptime T: type, storage: array_mod.DeviceStorage, value: T) array_mod.ArrayError!void {
    return switch (executionTargetForDevice(storage.device)) {
        .cpu => error.InvalidDevice,
        .cuda => axiom_cuda.fillStorage(T, storage, value),
        .mps => error.InvalidDevice,
    };
}

pub fn fillAllocated(comptime T: type, device: array_mod.Device, host_data: []T, storage: ?array_mod.DeviceStorage, value: T) array_mod.ArrayError!void {
    if (storage) |device_storage| {
        try fillStorage(T, device_storage, value);
        return;
    }
    if (!hostFallbackAllowed(device)) return error.InvalidDevice;
    @memset(host_data, value);
}

pub fn uploadStorage(storage: array_mod.DeviceStorage, bytes: []const u8) array_mod.ArrayError!void {
    return switch (executionTargetForDevice(storage.device)) {
        .cpu => error.InvalidDevice,
        .cuda => axiom_cuda.uploadStorage(storage, bytes),
        .mps => error.InvalidDevice,
    };
}

pub fn downloadStorage(storage: array_mod.DeviceStorage, bytes: []u8) array_mod.ArrayError!void {
    return switch (executionTargetForDevice(storage.device)) {
        .cpu => error.InvalidDevice,
        .cuda => axiom_cuda.downloadStorage(storage, bytes),
        .mps => error.InvalidDevice,
    };
}

pub fn copyStorage(dst: array_mod.DeviceStorage, src: array_mod.DeviceStorage) array_mod.ArrayError!void {
    if (!dst.device.sameDevice(src.device)) return error.InvalidDevice;
    return switch (executionTargetForDevice(dst.device)) {
        .cpu => error.InvalidDevice,
        .cuda => axiom_cuda.copyStorage(dst, src),
        .mps => error.InvalidDevice,
    };
}

pub const StorageSource = struct {
    device: array_mod.Device,
    host_bytes: []const u8,
    storage: ?array_mod.DeviceStorage = null,
};

pub const StorageDestination = struct {
    device: array_mod.Device,
    host_bytes: []u8,
    storage: ?array_mod.DeviceStorage = null,
};

/// Move array bytes between Vectra host slices and target-owned storage.
///
/// `Array` creation/clone/transfer code should hand the desired source and
/// destination devices to this facade instead of spelling out CPU↔CUDA cases.
/// That keeps the public array layer target-oriented while this backend module
/// remains the only place that knows which Axiom runtime ABI currently backs a
/// target.  MPS deliberately stays unsupported here until Axiom exposes real
/// Metal/MPS storage semantics.
pub fn transferStorage(dst: StorageDestination, src: StorageSource) array_mod.ArrayError!void {
    return switch (executionTargetForDevice(src.device)) {
        .cpu => switch (executionTargetForDevice(dst.device)) {
            .cpu => {
                if (dst.host_bytes.len != src.host_bytes.len) return error.ShapeMismatch;
                @memcpy(dst.host_bytes, src.host_bytes);
            },
            .cuda => {
                const dst_storage = dst.storage orelse return error.InvalidDevice;
                try uploadStorage(dst_storage, src.host_bytes);
            },
            .mps => error.InvalidDevice,
        },
        .cuda => switch (executionTargetForDevice(dst.device)) {
            .cpu => {
                const src_storage = src.storage orelse return error.InvalidDevice;
                try downloadStorage(src_storage, dst.host_bytes);
            },
            .cuda => {
                const src_storage = src.storage orelse return error.InvalidDevice;
                const dst_storage = dst.storage orelse return error.InvalidDevice;
                try copyStorage(dst_storage, src_storage);
            },
            .mps => error.InvalidDevice,
        },
        .mps => error.InvalidDevice,
    };
}

threadlocal var default_dialect_backend: DialectBackend = .cpu;

pub fn setDefaultDialectBackend(backend: DialectBackend) void {
    default_dialect_backend = backend;
}

pub fn defaultDialectBackend() DialectBackend {
    return default_dialect_backend;
}

pub fn resetDefaultDialectBackend() void {
    default_dialect_backend = .cpu;
}

pub fn defaultBackendPolicy() BackendPolicy {
    return switch (defaultDialectBackend()) {
        .cpu => .prefer_axiom_cpu,
        .cuda => .prefer_cuda,
        // Axiom's MPS runtime ABI is currently planned/unavailable.  A default
        // MPS selection should still be legal for planning evidence, but eager
        // CPU arrays must keep a real execution fallback rather than pretending
        // MPS ran.
        .mps => .prefer_axiom_cpu,
    };
}

pub fn defaultExecutionTarget() DialectBackend {
    return switch (defaultDialectBackend()) {
        .cpu => .cpu,
        .cuda => .cuda,
        // MPS remains a valid dialect-lowering target, but eager execution must
        // keep a real runtime path until Axiom owns a Metal/MPS storage ABI.
        // Centralizing the fallback here prevents Array methods from scattering
        // ad-hoc `.mps -> .cpu` branches.
        .mps => .cpu,
    };
}

pub fn executionTargetForDevice(device: array_mod.Device) DialectBackend {
    return switch (device.backend) {
        .cpu => .cpu,
        .cuda => .cuda,
        // Real MPS arrays cannot be created until Axiom reports a usable
        // Metal/MPS storage ABI.  If a caller still passes one explicitly, keep
        // the target honest so execution returns unsupported instead of falling
        // through a CPU path with the wrong storage semantics.
        .mps => .mps,
    };
}

fn policyExecutionTarget(policy: BackendPolicy) DialectBackend {
    return switch (policy) {
        .prefer_cuda => .cuda,
        .prefer_axiom_cpu, .force_direct_cpu => .cpu,
    };
}

fn targetCanAccessDevice(target: DialectBackend, device: array_mod.Device) bool {
    return switch (device.backend) {
        .cpu => target == .cpu or target == .cuda,
        .cuda => target == .cuda,
        .mps => target == .mps,
    };
}

fn defaultTargetForDevice(device: array_mod.Device) DialectBackend {
    return if (device.isCpu()) defaultExecutionTarget() else executionTargetForDevice(device);
}

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

pub fn lowerMatmulDialect(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T), backend: DialectBackend) array_mod.ArrayError!DialectMatmulLoweringReport {
    if (!supportedMatmulLowering2d(T, lhs, rhs)) return error.ShapeMismatch;
    const element = dialectElement(T) orelse return error.TypeUnsupported;
    return axiom.accelerator.lowerDialectMatmul(.{
        .name = "vectra.matmul",
        .element = element,
        .m = lhs.shape[0],
        .n = rhs.shape[1],
        .k = lhs.shape[1],
        .backend = backend,
    }) catch error.BackendFailure;
}

pub fn lowerMatmulDialectDefault(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!DialectMatmulLoweringReport {
    return lowerMatmulDialect(T, lhs, rhs, defaultDialectBackend());
}

pub fn lowerMatmulDialectForRoute(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T), route: BackendRoute) array_mod.ArrayError!DialectMatmulLoweringReport {
    return lowerMatmulDialect(T, lhs, rhs, switch (route) {
        .direct_cpu, .axiom_cpu_veyra => .cpu,
        .axiom_cuda => .cuda,
    });
}

pub fn lowerElementwiseDialect(comptime T: type, op: ElementwiseOp, lhs: array_mod.Array(T), rhs: array_mod.Array(T), backend: DialectBackend) array_mod.ArrayError!DialectElementwiseLoweringReport {
    if (!supportedElementwiseLowering(T, lhs, rhs)) return error.ShapeMismatch;
    const element = dialectElement(T) orelse return error.TypeUnsupported;
    return axiom.accelerator.lowerDialectElementwise(.{
        .name = "vectra.elementwise",
        .element = element,
        .rows = if (lhs.shape.len == 1) 1 else lhs.shape[0],
        .cols = if (lhs.shape.len == 1) lhs.shape[0] else lhs.shape[1],
        .op = dialectElementwiseOp(op),
        .backend = backend,
    }) catch error.BackendFailure;
}

pub fn lowerElementwiseDialectDefault(comptime T: type, op: ElementwiseOp, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!DialectElementwiseLoweringReport {
    return lowerElementwiseDialect(T, op, lhs, rhs, defaultDialectBackend());
}

fn dialectElementwiseOp(op: ElementwiseOp) DialectElementwiseOp {
    return switch (op) {
        .add => .add,
        .sub => .sub,
        .mul => .mul,
        .div => .div,
    };
}

pub fn lowerReductionDialect(comptime T: type, input: array_mod.Array(T), op: DialectReductionOp, axis: u1, backend: DialectBackend) array_mod.ArrayError!DialectReductionLoweringReport {
    if (!supportedReductionLowering2d(T, input)) return error.ShapeMismatch;
    const element = dialectElement(T) orelse return error.TypeUnsupported;
    return axiom.accelerator.lowerDialectReduction(.{
        .name = "vectra.reduction",
        .element = element,
        .rows = input.shape[0],
        .cols = input.shape[1],
        .axis = axis,
        .op = op,
        .backend = backend,
    }) catch error.BackendFailure;
}

pub fn lowerReductionDialectDefault(comptime T: type, input: array_mod.Array(T), op: DialectReductionOp, axis: u1) array_mod.ArrayError!DialectReductionLoweringReport {
    return lowerReductionDialect(T, input, op, axis, defaultDialectBackend());
}

pub fn reductionRuntimeCapability(target: DialectBackend) RuntimeCapabilityReport {
    return switch (target) {
        .cpu => .{
            .target = target,
            .operation = "reduction",
            .status = .executable,
            .reason = "Axiom CPU reduction runtime is routed through Veyra for contiguous f32/f64 2D axis reductions.",
        },
        .cuda => .{
            .target = target,
            .operation = "reduction",
            .status = .executable,
            .reason = "Axiom CUDA exposes eager f32 2D sum/prod/min/max reduction runtimes; other reduction dtypes remain capability-gated.",
        },
        .mps => .{
            .target = target,
            .operation = "reduction",
            .status = .unavailable,
            .reason = "Axiom MPS runtime ABI is planned/unavailable.",
        },
    };
}

pub fn broadcastAddRuntimeCapability(target: DialectBackend) RuntimeCapabilityReport {
    return switch (target) {
        .cpu => .{
            .target = target,
            .operation = "broadcast_add",
            .status = .executable,
            .reason = "Axiom CPU broadcast-add runtime is routed through Veyra for contiguous f32/f64 row/column 2D bias adds.",
        },
        .cuda => .{
            .target = target,
            .operation = "broadcast_add",
            .status = .executable,
            .reason = "Axiom CUDA exposes eager f32/f64 2D row/column broadcast-add runtimes; other broadcast dtypes/shapes remain capability-gated.",
        },
        .mps => .{
            .target = target,
            .operation = "broadcast_add",
            .status = .unavailable,
            .reason = "Axiom MPS runtime ABI is planned/unavailable.",
        },
    };
}

pub fn transposeRuntimeCapability(target: DialectBackend) RuntimeCapabilityReport {
    return switch (target) {
        .cpu => .{
            .target = target,
            .operation = "transpose2d",
            .status = .executable,
            .reason = "Axiom CPU transpose runtime is routed through Veyra for contiguous f32/f64 2D transpose.",
        },
        .cuda => .{
            .target = target,
            .operation = "transpose2d",
            .status = .executable,
            .reason = "Axiom CUDA exposes eager f32/f64 2D transpose runtimes; other transpose dtypes/shapes remain capability-gated.",
        },
        .mps => .{
            .target = target,
            .operation = "transpose2d",
            .status = .unavailable,
            .reason = "Axiom MPS runtime ABI is planned/unavailable.",
        },
    };
}

pub fn unaryRuntimeCapability(target: DialectBackend, op: DialectUnaryOp) RuntimeCapabilityReport {
    return switch (target) {
        .cpu => .{
            .target = target,
            .operation = dialectUnaryRuntimeOperation(op),
            .status = if (op == .square) .executable else .lowering_only,
            .reason = if (op == .square)
                "Axiom CPU square runtime is routed through Veyra unary elementwise execution."
            else
                "Axiom CPU unary dialect lowering exists for this op, but Vectra has no dedicated eager Axiom runtime ABI for it yet.",
        },
        .cuda => .{
            .target = target,
            .operation = dialectUnaryRuntimeOperation(op),
            .status = if (op == .square) .executable else .lowering_only,
            .reason = if (op == .square)
                "Axiom CUDA square eager execution is routed through the device elementwise multiply runtime."
            else
                "Axiom CUDA unary dialect lowering exists for this op, but Vectra has no dedicated eager CUDA runtime ABI for it yet.",
        },
        .mps => .{
            .target = target,
            .operation = dialectUnaryRuntimeOperation(op),
            .status = .unavailable,
            .reason = "Axiom MPS runtime ABI is planned/unavailable.",
        },
    };
}

fn dialectUnaryRuntimeOperation(op: DialectUnaryOp) []const u8 {
    return switch (op) {
        .copy => "unary.copy",
        .square => "unary.square",
        .cube => "unary.cube",
    };
}

pub fn lowerBroadcastAddDialect(comptime T: type, input: array_mod.Array(T), bias: array_mod.Array(T), axis: DialectBroadcastAxis, backend: DialectBackend) array_mod.ArrayError!DialectBroadcastLoweringReport {
    if (!supportedBroadcastAddLowering(T, input, bias, axis)) return error.ShapeMismatch;
    const element = dialectElement(T) orelse return error.TypeUnsupported;
    return axiom.accelerator.lowerDialectBroadcastAdd(.{
        .name = "vectra.broadcast_add",
        .element = element,
        .rows = input.shape[0],
        .cols = input.shape[1],
        .axis = axis,
        .backend = backend,
    }) catch error.BackendFailure;
}

pub fn lowerBroadcastAddDialectDefault(comptime T: type, input: array_mod.Array(T), bias: array_mod.Array(T), axis: DialectBroadcastAxis) array_mod.ArrayError!DialectBroadcastLoweringReport {
    return lowerBroadcastAddDialect(T, input, bias, axis, defaultDialectBackend());
}

pub fn executeBroadcastAdd(
    comptime T: type,
    target: DialectBackend,
    input: array_mod.Array(T),
    bias: array_mod.Array(T),
    axis: DialectBroadcastAxis,
) array_mod.ArrayError!?array_mod.Array(T) {
    if (!supportedBroadcastAddExecution(T, target, input, bias, axis)) return null;
    return switch (target) {
        .cpu => executeCpuBroadcastAdd(T, input, bias, axis),
        .cuda => executeCudaBroadcastAdd(T, input, bias, axis),
        .mps => null,
    };
}

pub fn executeBroadcastAddDefault(comptime T: type, input: array_mod.Array(T), bias: array_mod.Array(T), axis: DialectBroadcastAxis) array_mod.ArrayError!?array_mod.Array(T) {
    return executeBroadcastAdd(T, defaultTargetForDevice(input.device), input, bias, axis);
}

pub fn tryBroadcastAdd(
    comptime T: type,
    target: DialectBackend,
    lhs: array_mod.Array(T),
    rhs: array_mod.Array(T),
) array_mod.ArrayError!?array_mod.Array(T) {
    if (!lhs.device.sameDevice(rhs.device)) return error.InvalidDevice;
    if (lhs.shape.len == 2) {
        if (broadcastBiasMatchesArrayAdd(T, lhs, rhs, .row)) return executeBroadcastAdd(T, target, lhs, rhs, .row);
        if (broadcastBiasMatchesArrayAdd(T, lhs, rhs, .column)) return executeBroadcastAdd(T, target, lhs, rhs, .column);
    }
    if (rhs.shape.len == 2) {
        if (broadcastBiasMatchesArrayAdd(T, rhs, lhs, .row)) return executeBroadcastAdd(T, target, rhs, lhs, .row);
        if (broadcastBiasMatchesArrayAdd(T, rhs, lhs, .column)) return executeBroadcastAdd(T, target, rhs, lhs, .column);
    }
    return null;
}

pub fn tryBroadcastAddDefault(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    return tryBroadcastAdd(T, defaultTargetForDevice(lhs.device), lhs, rhs);
}

pub fn lowerUnaryDialect(comptime T: type, input: array_mod.Array(T), op: DialectUnaryOp, backend: DialectBackend) array_mod.ArrayError!DialectUnaryLoweringReport {
    if (!supportedUnaryLowering2d(T, input)) return error.ShapeMismatch;
    const element = dialectElement(T) orelse return error.TypeUnsupported;
    return axiom.accelerator.lowerDialectUnary(.{
        .name = "vectra.unary",
        .element = element,
        .rows = input.shape[0],
        .cols = input.shape[1],
        .op = op,
        .backend = backend,
    }) catch error.BackendFailure;
}

pub fn lowerUnaryDialectDefault(comptime T: type, input: array_mod.Array(T), op: DialectUnaryOp) array_mod.ArrayError!DialectUnaryLoweringReport {
    return lowerUnaryDialect(T, input, op, defaultDialectBackend());
}

pub fn lowerTransposeDialect(comptime T: type, input: array_mod.Array(T), backend: DialectBackend) array_mod.ArrayError!DialectTransposeLoweringReport {
    if (!supportedUnaryLowering2d(T, input)) return error.ShapeMismatch;
    const element = dialectElement(T) orelse return error.TypeUnsupported;
    return axiom.accelerator.lowerDialectTranspose(.{
        .name = "vectra.transpose2d",
        .element = element,
        .rows = input.shape[0],
        .cols = input.shape[1],
        .backend = backend,
    }) catch error.BackendFailure;
}

pub fn lowerTransposeDialectDefault(comptime T: type, input: array_mod.Array(T)) array_mod.ArrayError!DialectTransposeLoweringReport {
    return lowerTransposeDialect(T, input, defaultDialectBackend());
}

fn dialectElement(comptime T: type) ?axiom.linalg_dialect.Element {
    return if (T == f32)
        .f32
    else if (T == f16)
        .f16
    else if (T == f64)
        .f64
    else if (T == array_mod.BFloat16)
        .bf16
    else
        null;
}

pub fn executeMatmul(
    comptime T: type,
    target: DialectBackend,
    lhs: array_mod.Array(T),
    rhs: array_mod.Array(T),
) array_mod.ArrayError!?array_mod.Array(T) {
    if (!targetCanAccessDevice(target, lhs.device)) return null;
    if (!supportedMatmulExecution(T, lhs, rhs)) return null;
    return switch (target) {
        .cpu => executeCpuMatmul(T, lhs, rhs),
        .cuda => executeCudaMatmul(T, lhs, rhs),
        .mps => null,
    };
}

pub fn executeMatmulDefault(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    return executeMatmul(T, defaultTargetForDevice(lhs.device), lhs, rhs);
}

fn bufferView(comptime T: type, input: array_mod.Array(T), name: []const u8) ?axiom.accelerator.TensorBufferView {
    if (input.shape.len != 1 or input.strides.len != 1) return null;
    const stride = std.math.cast(isize, input.strides[0]) orelse return null;
    var view = axiom.accelerator.TensorBufferView.strided(name, @intCast(@intFromPtr(input.data.ptr)), input.shape[0], stride);
    view.element_type = if (T == f32) .f32 else if (T == f64) .f64 else return null;
    return view;
}

fn executeCpuDotTarget(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    const lhs_view = bufferView(T, lhs, "lhs") orelse return null;
    const rhs_view = bufferView(T, rhs, "rhs") orelse return null;
    if (T == f32) {
        var value: f32 = 0;
        const report = axiom.accelerator.cpu_veyra.runTargetDotF32(.cpu, lhs_view, rhs_view, @as(array_mod.Array(f32), lhs).data, @as(array_mod.Array(f32), rhs).data, &value) catch return null;
        if (!report.ok()) return null;
        return @as(array_mod.Array(T), try array_mod.Array(f32).fromSlice(lhs.allocator, &.{value}, &.{}));
    } else if (T == f64) {
        var value: f64 = 0;
        const report = axiom.accelerator.cpu_veyra.runTargetDotF64(.cpu, lhs_view, rhs_view, @as(array_mod.Array(f64), lhs).data, @as(array_mod.Array(f64), rhs).data, &value) catch return null;
        if (!report.ok()) return null;
        return @as(array_mod.Array(T), try array_mod.Array(f64).fromSlice(lhs.allocator, &.{value}, &.{}));
    }
    return null;
}

fn executeCpuMatvecTarget(comptime T: type, matrix: array_mod.Array(T), vector: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    const matrix_view = matrixView(T, matrix, "matrix") orelse return null;
    const vector_view = bufferView(T, vector, "vector") orelse return null;
    if (T == f32) {
        var out = try array_mod.Array(f32).empty(matrix.allocator, &.{matrix.shape[0]});
        errdefer out.deinit();
        var out_view = axiom.accelerator.TensorBufferView.contiguous("out", @intCast(@intFromPtr(out.data.ptr)), out.data.len);
        out_view.element_type = .f32;
        const report = axiom.accelerator.cpu_veyra.runTargetMatvecF32(.cpu, matrix_view, vector_view, out_view, @as(array_mod.Array(f32), matrix).data, @as(array_mod.Array(f32), vector).data, out.data) catch {
            out.deinit();
            return null;
        };
        if (!report.ok()) {
            out.deinit();
            return null;
        }
        return @as(array_mod.Array(T), out);
    } else if (T == f64) {
        var out = try array_mod.Array(f64).empty(matrix.allocator, &.{matrix.shape[0]});
        errdefer out.deinit();
        var out_view = axiom.accelerator.TensorBufferView.contiguous("out", @intCast(@intFromPtr(out.data.ptr)), out.data.len);
        out_view.element_type = .f64;
        const report = axiom.accelerator.cpu_veyra.runTargetMatvecF64(.cpu, matrix_view, vector_view, out_view, @as(array_mod.Array(f64), matrix).data, @as(array_mod.Array(f64), vector).data, out.data) catch {
            out.deinit();
            return null;
        };
        if (!report.ok()) {
            out.deinit();
            return null;
        }
        return @as(array_mod.Array(T), out);
    }
    return null;
}

fn executeCpuVecmatTarget(comptime T: type, vector: array_mod.Array(T), matrix: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    const vector_view = bufferView(T, vector, "vector") orelse return null;
    const matrix_view = matrixView(T, matrix, "matrix") orelse return null;
    if (T == f32) {
        var out = try array_mod.Array(f32).empty(vector.allocator, &.{matrix.shape[1]});
        errdefer out.deinit();
        var out_view = axiom.accelerator.TensorBufferView.contiguous("out", @intCast(@intFromPtr(out.data.ptr)), out.data.len);
        out_view.element_type = .f32;
        const report = axiom.accelerator.cpu_veyra.runTargetVecmatF32(.cpu, vector_view, matrix_view, out_view, @as(array_mod.Array(f32), vector).data, @as(array_mod.Array(f32), matrix).data, out.data) catch {
            out.deinit();
            return null;
        };
        if (!report.ok()) {
            out.deinit();
            return null;
        }
        return @as(array_mod.Array(T), out);
    } else if (T == f64) {
        var out = try array_mod.Array(f64).empty(vector.allocator, &.{matrix.shape[1]});
        errdefer out.deinit();
        var out_view = axiom.accelerator.TensorBufferView.contiguous("out", @intCast(@intFromPtr(out.data.ptr)), out.data.len);
        out_view.element_type = .f64;
        const report = axiom.accelerator.cpu_veyra.runTargetVecmatF64(.cpu, vector_view, matrix_view, out_view, @as(array_mod.Array(f64), vector).data, @as(array_mod.Array(f64), matrix).data, out.data) catch {
            out.deinit();
            return null;
        };
        if (!report.ok()) {
            out.deinit();
            return null;
        }
        return @as(array_mod.Array(T), out);
    }
    return null;
}

fn executeCpuMatmul(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (T == f32) {
        const lhs32 = @as(array_mod.Array(f32), lhs);
        const rhs32 = @as(array_mod.Array(f32), rhs);
        if (lhs.shape.len == 1 and rhs.shape.len == 1) {
            if (try executeCpuDotTarget(f32, lhs32, rhs32)) |out| return @as(array_mod.Array(T), out);
        } else if (lhs.shape.len == 2 and rhs.shape.len == 1) {
            if (try executeCpuMatvecTarget(f32, lhs32, rhs32)) |out| return @as(array_mod.Array(T), out);
        } else if (lhs.shape.len == 1 and rhs.shape.len == 2) {
            if (try executeCpuVecmatTarget(f32, lhs32, rhs32)) |out| return @as(array_mod.Array(T), out);
        } else if (try executeCpuGemmTarget(f32, lhs32, rhs32)) |out| return @as(array_mod.Array(T), out);
    } else if (T == f64) {
        const lhs64 = @as(array_mod.Array(f64), lhs);
        const rhs64 = @as(array_mod.Array(f64), rhs);
        if (lhs.shape.len == 1 and rhs.shape.len == 1) {
            if (try executeCpuDotTarget(f64, lhs64, rhs64)) |out| return @as(array_mod.Array(T), out);
        } else if (lhs.shape.len == 2 and rhs.shape.len == 1) {
            if (try executeCpuMatvecTarget(f64, lhs64, rhs64)) |out| return @as(array_mod.Array(T), out);
        } else if (lhs.shape.len == 1 and rhs.shape.len == 2) {
            if (try executeCpuVecmatTarget(f64, lhs64, rhs64)) |out| return @as(array_mod.Array(T), out);
        } else if (try executeCpuGemmTarget(f64, lhs64, rhs64)) |out| return @as(array_mod.Array(T), out);
    }
    return null;
}

fn executeCpuGemmTarget(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (lhs.shape.len != 2 or rhs.shape.len != 2) return null;
    const m = lhs.shape[0];
    const k = lhs.shape[1];
    const n = rhs.shape[1];
    var c = try array_mod.Array(T).zeros(lhs.allocator, &.{ m, n });
    defer c.deinit();
    var out = try array_mod.Array(T).empty(lhs.allocator, &.{ m, n });
    errdefer out.deinit();
    const spec = axiom.accelerator.TensorGemmSpec.rowMajor(
        .rowMajor("lhs", @intCast(@intFromPtr(lhs.data.ptr)), m, k),
        .rowMajor("rhs", @intCast(@intFromPtr(rhs.data.ptr)), k, n),
        .rowMajor("out", @intCast(@intFromPtr(out.data.ptr)), m, n),
    );
    const report = if (T == f32)
        axiom.accelerator.cpu_veyra.runTargetGemmF32(.cpu, spec, @as(array_mod.Array(f32), lhs).data, @as(array_mod.Array(f32), rhs).data, @as(array_mod.Array(f32), c).data, @as(array_mod.Array(f32), out).data) catch {
            out.deinit();
            return null;
        }
    else
        axiom.accelerator.cpu_veyra.runTargetGemmF64(.cpu, spec, @as(array_mod.Array(f64), lhs).data, @as(array_mod.Array(f64), rhs).data, @as(array_mod.Array(f64), c).data, @as(array_mod.Array(f64), out).data) catch {
            out.deinit();
            return null;
        };
    if (!report.ok()) {
        out.deinit();
        return null;
    }
    return out;
}

fn executeCpuGemmAddTarget(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T), addend: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    return executeCpuGemmScaledTarget(T, lhs, rhs, addend, 1.0, 1.0);
}

fn executeCpuGemmScaledTarget(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T), addend: array_mod.Array(T), alpha: f32, beta: f32) array_mod.ArrayError!?array_mod.Array(T) {
    if (lhs.shape.len != 2 or rhs.shape.len != 2 or addend.shape.len != 2) return null;
    const m = lhs.shape[0];
    const k = lhs.shape[1];
    const n = rhs.shape[1];
    var out = try array_mod.Array(T).empty(lhs.allocator, &.{ m, n });
    errdefer out.deinit();
    var spec = axiom.accelerator.TensorGemmSpec.rowMajor(
        .rowMajor("lhs", @intCast(@intFromPtr(lhs.data.ptr)), m, k),
        .rowMajor("rhs", @intCast(@intFromPtr(rhs.data.ptr)), k, n),
        .rowMajor("out", @intCast(@intFromPtr(out.data.ptr)), m, n),
    );
    spec.alpha = alpha;
    spec.beta = beta;
    const report = if (T == f32)
        axiom.accelerator.cpu_veyra.runTargetGemmF32(.cpu, spec, @as(array_mod.Array(f32), lhs).data, @as(array_mod.Array(f32), rhs).data, @as(array_mod.Array(f32), addend).data, @as(array_mod.Array(f32), out).data) catch {
            out.deinit();
            return null;
        }
    else
        axiom.accelerator.cpu_veyra.runTargetGemmF64(.cpu, spec, @as(array_mod.Array(f64), lhs).data, @as(array_mod.Array(f64), rhs).data, @as(array_mod.Array(f64), addend).data, @as(array_mod.Array(f64), out).data) catch {
            out.deinit();
            return null;
        };
    if (!report.ok()) {
        out.deinit();
        return null;
    }
    return out;
}

fn executeCudaMatmul(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (lhs.shape.len != 2 or rhs.shape.len != 2) return null;
    if (T == f32) {
        if (try axiom_cuda.tryMatmulF32(@as(array_mod.Array(f32), lhs), @as(array_mod.Array(f32), rhs))) |out| return @as(array_mod.Array(T), out);
    } else if (T == f64) {
        if (try axiom_cuda.tryDeviceMatmulF64(@as(array_mod.Array(f64), lhs), @as(array_mod.Array(f64), rhs))) |out| return @as(array_mod.Array(T), out);
    } else if (T == f16) {
        if (try axiom_cuda.tryMatmulF16(@as(array_mod.Array(f16), lhs), @as(array_mod.Array(f16), rhs))) |out| return @as(array_mod.Array(T), out);
    } else if (T == array_mod.BFloat16) {
        if (try axiom_cuda.tryMatmulBF16(@as(array_mod.Array(array_mod.BFloat16), lhs), @as(array_mod.Array(array_mod.BFloat16), rhs))) |out| return @as(array_mod.Array(T), out);
    }
    return null;
}

pub fn executeMatmulAdd(
    comptime T: type,
    target: DialectBackend,
    lhs: array_mod.Array(T),
    rhs: array_mod.Array(T),
    addend: array_mod.Array(T),
) array_mod.ArrayError!?array_mod.Array(T) {
    if (!targetCanAccessDevice(target, lhs.device)) return null;
    if (!supportedMatmulAddExecution(T, lhs, rhs, addend)) return null;
    return switch (target) {
        .cpu => executeCpuMatmulAdd(T, lhs, rhs, addend),
        .cuda => executeCudaMatmulAdd(T, lhs, rhs, addend),
        .mps => null,
    };
}

pub fn executeMatmulAddDefault(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T), addend: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    return executeMatmulAdd(T, defaultTargetForDevice(lhs.device), lhs, rhs, addend);
}

pub fn executeMatmulAddScaled(
    comptime T: type,
    target: DialectBackend,
    lhs: array_mod.Array(T),
    rhs: array_mod.Array(T),
    addend: array_mod.Array(T),
    alpha: f32,
    beta: f32,
) array_mod.ArrayError!?array_mod.Array(T) {
    if (!targetCanAccessDevice(target, lhs.device)) return null;
    if (!supportedMatmulAddExecution(T, lhs, rhs, addend)) return null;
    return switch (target) {
        .cpu => executeCpuGemmScaledTarget(T, lhs, rhs, addend, alpha, beta),
        .cuda => executeCudaMatmulAddScaled(T, lhs, rhs, addend, alpha, beta),
        .mps => null,
    };
}

pub fn executeMatmulAddScaledDefault(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T), addend: array_mod.Array(T), alpha: f32, beta: f32) array_mod.ArrayError!?array_mod.Array(T) {
    return executeMatmulAddScaled(T, defaultTargetForDevice(lhs.device), lhs, rhs, addend, alpha, beta);
}

pub fn planPendingMatmul(
    comptime T: type,
    target: DialectBackend,
    lhs: array_mod.Array(T),
    rhs: array_mod.Array(T),
) array_mod.ArrayError!?PendingMatmulSpec {
    if (target != .cuda or executionTargetForDevice(lhs.device) != .cuda) return null;
    if (!supportedMatmulExecution(T, lhs, rhs)) return null;
    if (lhs.shape.len != 2 or rhs.shape.len != 2) return null;
    const lhs_storage = lhs.device_storage orelse return error.InvalidDevice;
    const rhs_storage = rhs.device_storage orelse return error.InvalidDevice;
    return .{
        .lhs_storage = lhs_storage,
        .rhs_storage = rhs_storage,
        .lhs_shape = .{ lhs.shape[0], lhs.shape[1] },
        .rhs_shape = .{ rhs.shape[0], rhs.shape[1] },
    };
}

pub fn planPendingMatmulDefault(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?PendingMatmulSpec {
    return planPendingMatmul(T, defaultTargetForDevice(lhs.device), lhs, rhs);
}

pub fn hostFallbackAllowed(device: array_mod.Device) bool {
    return executionTargetForDevice(device) == .cpu;
}

pub fn shouldRestoreDeviceAfterHostCast(device: array_mod.Device) bool {
    return switch (executionTargetForDevice(device)) {
        .cpu => false,
        .cuda => true,
        .mps => false,
    };
}

pub fn pendingMatmulDeviceSupported(device: array_mod.Device) bool {
    return executionTargetForDevice(device) == .cuda;
}

pub fn pendingMatmulSameDeviceSupported(lhs: array_mod.Device, rhs: array_mod.Device) bool {
    return lhs.sameDevice(rhs) and pendingMatmulDeviceSupported(lhs);
}

pub fn executePendingMatmul(
    comptime T: type,
    target: DialectBackend,
    allocator: std.mem.Allocator,
    shape: []const usize,
    device: array_mod.Device,
    pending: PendingMatmulSpec,
) array_mod.ArrayError!?array_mod.Array(T) {
    if (target != .cuda or !device.isCuda()) return null;
    if (comptime T != f32 and T != f64 and T != f16 and T != array_mod.BFloat16) return null;
    if (shape.len != 2) return null;
    const m = pending.lhs_shape[0];
    const k = pending.lhs_shape[1];
    const n = pending.rhs_shape[1];
    if (shape[0] != m or shape[1] != n or pending.rhs_shape[0] != k) return null;
    var out = try array_mod.Array(T).emptyOn(allocator, shape, device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse return error.InvalidDevice;
    const ok = try runCudaPendingMatmul(T, allocator, device, m, n, k, pending, out_storage.ptr);
    if (!ok) return error.BackendFailure;
    return out;
}

pub fn executePendingMatmulDefault(
    comptime T: type,
    allocator: std.mem.Allocator,
    shape: []const usize,
    device: array_mod.Device,
    pending: PendingMatmulSpec,
) array_mod.ArrayError!?array_mod.Array(T) {
    return executePendingMatmul(T, executionTargetForDevice(device), allocator, shape, device, pending);
}

fn runCudaPendingMatmul(
    comptime T: type,
    allocator: std.mem.Allocator,
    device: array_mod.Device,
    m: usize,
    n: usize,
    k: usize,
    pending: PendingMatmulSpec,
    out_ptr: u64,
) array_mod.ArrayError!bool {
    if (pending.unary) |unary| {
        return runCudaPendingMatmulUnary(T, allocator, device, m, n, k, pending, out_ptr, unary);
    }
    if (pending.add_storage) |add_storage| {
        return if (T == f32)
            axiom_cuda.runPendingMatmulAddF32(allocator, device, m, n, k, pending.lhs_storage.ptr, pending.rhs_storage.ptr, add_storage.ptr, out_ptr, pending.alpha, pending.beta)
        else if (T == f64)
            axiom_cuda.runPendingMatmulAddF64(allocator, device, m, n, k, pending.lhs_storage.ptr, pending.rhs_storage.ptr, add_storage.ptr, out_ptr, pending.alpha, pending.beta)
        else if (T == f16)
            axiom_cuda.runPendingMatmulAddF16(allocator, device, m, n, k, pending.lhs_storage.ptr, pending.rhs_storage.ptr, add_storage.ptr, out_ptr, pending.alpha, pending.beta)
        else if (T == array_mod.BFloat16)
            axiom_cuda.runPendingMatmulAddBF16(allocator, device, m, n, k, pending.lhs_storage.ptr, pending.rhs_storage.ptr, add_storage.ptr, out_ptr, pending.alpha, pending.beta)
        else
            error.TypeUnsupported;
    }
    return if (T == f32)
        axiom_cuda.runPendingMatmulF32(allocator, device, m, n, k, pending.lhs_storage.ptr, pending.rhs_storage.ptr, out_ptr)
    else if (T == f64)
        axiom_cuda.runPendingMatmulF64(allocator, device, m, n, k, pending.lhs_storage.ptr, pending.rhs_storage.ptr, out_ptr)
    else if (T == f16)
        axiom_cuda.runPendingMatmulF16(allocator, device, m, n, k, pending.lhs_storage.ptr, pending.rhs_storage.ptr, out_ptr)
    else if (T == array_mod.BFloat16)
        axiom_cuda.runPendingMatmulBF16(allocator, device, m, n, k, pending.lhs_storage.ptr, pending.rhs_storage.ptr, out_ptr)
    else
        error.TypeUnsupported;
}

fn runCudaPendingMatmulUnary(
    comptime T: type,
    allocator: std.mem.Allocator,
    device: array_mod.Device,
    m: usize,
    n: usize,
    k: usize,
    pending: PendingMatmulSpec,
    out_ptr: u64,
    unary: ExecutionUnaryOp,
) array_mod.ArrayError!bool {
    if (T == f32 and pending.add_storage != null) {
        const ops = std.math.mul(usize, std.math.mul(usize, m, n) catch return error.InvalidShape, k) catch return error.InvalidShape;
        if (ops <= 4 * 1024 * 1024) {
            const cuda_unary: axiom_cuda.UnaryOp = switch (unary) {
                .sqrt => .sqrt,
                .exp => .exp,
                .abs, .square => return false,
            };
            return axiom_cuda.runPendingMatmulAddUnaryF32(
                allocator,
                device,
                cuda_unary,
                m,
                n,
                k,
                pending.lhs_storage.ptr,
                pending.rhs_storage.ptr,
                pending.add_storage.?.ptr,
                out_ptr,
                pending.alpha,
                pending.beta,
            );
        }
    }

    var without_unary = pending;
    without_unary.unary = null;
    var materialized = (try executePendingMatmul(T, .cuda, allocator, &.{ m, n }, device, without_unary)) orelse return false;
    defer materialized.deinit();
    if (try executeUnary(T, unary, .cuda, materialized)) |out_value| {
        var out = out_value;
        defer out.deinit();
        const src_storage = out.device_storage orelse return error.InvalidDevice;
        const byte_count = std.math.mul(usize, m, n) catch return error.InvalidShape;
        const bytes = std.math.mul(usize, byte_count, @sizeOf(T)) catch return error.InvalidShape;
        try axiom_cuda.copyStorage(
            .{ .device = device, .ptr = out_ptr, .len = byte_count, .bytes = bytes, .owns = false },
            src_storage,
        );
        return true;
    }
    return false;
}

fn executeCpuMatmulAdd(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T), addend: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (T == f32) {
        if (try executeCpuGemmAddTarget(f32, @as(array_mod.Array(f32), lhs), @as(array_mod.Array(f32), rhs), @as(array_mod.Array(f32), addend))) |out| return @as(array_mod.Array(T), out);
    } else if (T == f64) {
        if (try executeCpuGemmAddTarget(f64, @as(array_mod.Array(f64), lhs), @as(array_mod.Array(f64), rhs), @as(array_mod.Array(f64), addend))) |out| return @as(array_mod.Array(T), out);
    }
    return null;
}

fn executeCudaMatmulAdd(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T), addend: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (T == f32) {
        if (try axiom_cuda.tryDeviceMatmulAddF32(@as(array_mod.Array(f32), lhs), @as(array_mod.Array(f32), rhs), @as(array_mod.Array(f32), addend))) |out| return @as(array_mod.Array(T), out);
    } else if (T == f64) {
        if (try axiom_cuda.tryDeviceMatmulAddF64(@as(array_mod.Array(f64), lhs), @as(array_mod.Array(f64), rhs), @as(array_mod.Array(f64), addend))) |out| return @as(array_mod.Array(T), out);
    } else if (T == f16) {
        if (try axiom_cuda.tryDeviceMatmulAddF16(@as(array_mod.Array(f16), lhs), @as(array_mod.Array(f16), rhs), @as(array_mod.Array(f16), addend))) |out| return @as(array_mod.Array(T), out);
    } else if (T == array_mod.BFloat16) {
        if (try axiom_cuda.tryDeviceMatmulAddBF16(@as(array_mod.Array(array_mod.BFloat16), lhs), @as(array_mod.Array(array_mod.BFloat16), rhs), @as(array_mod.Array(array_mod.BFloat16), addend))) |out| return @as(array_mod.Array(T), out);
    }
    return null;
}

fn executeCudaMatmulAddScaled(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T), addend: array_mod.Array(T), alpha: f32, beta: f32) array_mod.ArrayError!?array_mod.Array(T) {
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;
    const add_storage = addend.device_storage orelse return null;
    return executePendingMatmul(T, .cuda, lhs.allocator, addend.shape, lhs.device, .{
        .lhs_storage = lhs_storage,
        .rhs_storage = rhs_storage,
        .add_storage = add_storage,
        .lhs_shape = .{ lhs.shape[0], lhs.shape[1] },
        .rhs_shape = .{ rhs.shape[0], rhs.shape[1] },
        .alpha = alpha,
        .beta = beta,
    });
}

pub fn executeUnary(
    comptime T: type,
    op: ExecutionUnaryOp,
    target: DialectBackend,
    input: array_mod.Array(T),
) array_mod.ArrayError!?array_mod.Array(T) {
    if (!targetCanAccessDevice(target, input.device)) return null;
    if (!supportedUnaryExecution(T, input)) return null;
    return switch (target) {
        .cpu => executeCpuUnary(T, op, input),
        .cuda => executeCudaUnary(T, op, input),
        .mps => null,
    };
}

pub fn executeUnaryDefault(comptime T: type, op: ExecutionUnaryOp, input: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    return executeUnary(T, op, defaultTargetForDevice(input.device), input);
}

pub fn executeDialectUnaryDefault(comptime T: type, op: DialectUnaryOp, input: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    return executeUnaryDefault(T, switch (op) {
        .square => .square,
        else => return null,
    }, input);
}

pub fn executeReduction(
    comptime T: type,
    op: DialectReductionOp,
    target: DialectBackend,
    input: array_mod.Array(T),
    axis: u1,
    keepdims: bool,
) array_mod.ArrayError!?array_mod.Array(T) {
    if (!supportedReductionExecution(T, target, input)) return null;
    if (!reductionRuntimeCapability(target).executable()) return null;
    return switch (target) {
        .cpu => executeCpuReduction(T, op, input, axis, keepdims),
        .cuda => executeCudaReduction(T, op, input, axis, keepdims),
        .mps => null,
    };
}

pub fn executeReductionDefault(
    comptime T: type,
    op: DialectReductionOp,
    input: array_mod.Array(T),
    axis: u1,
    keepdims: bool,
) array_mod.ArrayError!?array_mod.Array(T) {
    return executeReduction(T, op, defaultTargetForDevice(input.device), input, axis, keepdims);
}

pub fn executeTranspose(
    comptime T: type,
    target: DialectBackend,
    input: array_mod.Array(T),
) array_mod.ArrayError!?array_mod.Array(T) {
    if (!supportedTransposeExecution(T, target, input)) return null;
    if (!transposeRuntimeCapability(target).executable()) return null;
    return switch (target) {
        .cpu => executeCpuTranspose(T, input),
        .cuda => executeCudaTranspose(T, input),
        .mps => null,
    };
}

pub fn executeTransposeDefault(comptime T: type, input: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    return executeTranspose(T, defaultTargetForDevice(input.device), input);
}

pub fn executeTrace(comptime T: type, target: DialectBackend, input: array_mod.Array(T), offset: isize) array_mod.ArrayError!?T {
    if (!supportedUnary2d(T, input)) return null;
    return switch (target) {
        .cpu => executeCpuTrace(T, input, offset),
        .cuda => null,
        .mps => null,
    };
}

pub fn executeTraceDefault(comptime T: type, input: array_mod.Array(T), offset: isize) array_mod.ArrayError!?T {
    return executeTrace(T, defaultTargetForDevice(input.device), input, offset);
}

pub fn executeDet(comptime T: type, target: DialectBackend, input: array_mod.Array(T)) array_mod.ArrayError!?T {
    if (!supportedSquareMatrixExecution(T, input)) return null;
    return switch (target) {
        .cpu => executeCpuDet(T, input),
        .cuda => null,
        .mps => null,
    };
}

pub fn executeDetDefault(comptime T: type, input: array_mod.Array(T)) array_mod.ArrayError!?T {
    return executeDet(T, defaultTargetForDevice(input.device), input);
}

pub fn executeInverse(comptime T: type, target: DialectBackend, input: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (!supportedSquareMatrixExecution(T, input)) return null;
    return switch (target) {
        .cpu => executeCpuInverse(T, input),
        .cuda => null,
        .mps => null,
    };
}

pub fn executeInverseDefault(comptime T: type, input: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    return executeInverse(T, defaultTargetForDevice(input.device), input);
}

pub fn executeSolve(comptime T: type, target: DialectBackend, matrix: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (!supportedSolveExecution(T, matrix, rhs)) return null;
    return switch (target) {
        .cpu => executeCpuSolve(T, matrix, rhs),
        .cuda => null,
        .mps => null,
    };
}

pub fn executeSolveDefault(comptime T: type, matrix: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    return executeSolve(T, defaultTargetForDevice(matrix.device), matrix, rhs);
}

pub fn executeCholesky(comptime T: type, target: DialectBackend, input: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (!supportedSquareMatrixExecution(T, input)) return null;
    return switch (target) {
        .cpu => executeCpuCholesky(T, input),
        .cuda => null,
        .mps => null,
    };
}

pub fn executeCholeskyDefault(comptime T: type, input: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    return executeCholesky(T, defaultTargetForDevice(input.device), input);
}

pub fn executeQr(comptime T: type, target: DialectBackend, input: array_mod.Array(T)) array_mod.ArrayError!?QrResult(T) {
    if (!supportedMatrixExecution(T, input)) return null;
    return switch (target) {
        .cpu => executeCpuQr(T, input),
        .cuda => null,
        .mps => null,
    };
}

pub fn executeQrDefault(comptime T: type, input: array_mod.Array(T)) array_mod.ArrayError!?QrResult(T) {
    return executeQr(T, defaultTargetForDevice(input.device), input);
}

pub fn executeLu(comptime T: type, target: DialectBackend, input: array_mod.Array(T)) array_mod.ArrayError!?LuResult(T) {
    if (!supportedSquareMatrixExecution(T, input)) return null;
    return switch (target) {
        .cpu => executeCpuLu(T, input),
        .cuda => null,
        .mps => null,
    };
}

pub fn executeLuDefault(comptime T: type, input: array_mod.Array(T)) array_mod.ArrayError!?LuResult(T) {
    return executeLu(T, defaultTargetForDevice(input.device), input);
}

pub fn executeSolveTriangular(
    comptime T: type,
    target: DialectBackend,
    matrix: array_mod.Array(T),
    rhs: array_mod.Array(T),
    triangle: array_mod.Triangle,
    diagonal: array_mod.Diagonal,
) array_mod.ArrayError!?array_mod.Array(T) {
    if (!supportedSolveExecution(T, matrix, rhs)) return null;
    return switch (target) {
        .cpu => executeCpuSolveTriangular(T, matrix, rhs, triangle, diagonal),
        .cuda => null,
        .mps => null,
    };
}

pub fn executeSolveTriangularDefault(comptime T: type, matrix: array_mod.Array(T), rhs: array_mod.Array(T), triangle: array_mod.Triangle, diagonal: array_mod.Diagonal) array_mod.ArrayError!?array_mod.Array(T) {
    return executeSolveTriangular(T, defaultTargetForDevice(matrix.device), matrix, rhs, triangle, diagonal);
}

pub fn executeMatrixNorm(comptime T: type, target: DialectBackend, input: array_mod.Array(T), order: array_mod.MatrixNormOrder) array_mod.ArrayError!?T {
    if (!supportedMatrixExecution(T, input)) return null;
    const cpu_order = cpuMatrixNormOrder(order) orelse return null;
    return switch (target) {
        .cpu => executeCpuMatrixNorm(T, input, cpu_order),
        .cuda => null,
        .mps => null,
    };
}

pub fn executeMatrixNormDefault(comptime T: type, input: array_mod.Array(T), order: array_mod.MatrixNormOrder) array_mod.ArrayError!?T {
    return executeMatrixNorm(T, defaultTargetForDevice(input.device), input, order);
}

pub fn executeSingularValues(comptime T: type, target: DialectBackend, input: array_mod.Array(T), tolerance: T) array_mod.ArrayError!?array_mod.Array(T) {
    if (!supportedMatrixExecution(T, input)) return null;
    return switch (target) {
        .cpu => executeCpuSingularValues(T, input, tolerance),
        .cuda => null,
        .mps => null,
    };
}

pub fn executeSingularValuesDefault(comptime T: type, input: array_mod.Array(T), tolerance: T) array_mod.ArrayError!?array_mod.Array(T) {
    return executeSingularValues(T, defaultTargetForDevice(input.device), input, tolerance);
}

pub fn executeSvd(comptime T: type, target: DialectBackend, input: array_mod.Array(T), tolerance: T) array_mod.ArrayError!?SvdResult(T) {
    if (!supportedMatrixExecution(T, input)) return null;
    return switch (target) {
        .cpu => executeCpuSvd(T, input, tolerance),
        .cuda => null,
        .mps => null,
    };
}

pub fn executeSvdDefault(comptime T: type, input: array_mod.Array(T), tolerance: T) array_mod.ArrayError!?SvdResult(T) {
    return executeSvd(T, defaultTargetForDevice(input.device), input, tolerance);
}

pub fn executeEigh(comptime T: type, target: DialectBackend, input: array_mod.Array(T), max_sweeps: usize, tolerance: T) array_mod.ArrayError!?EighResult(T) {
    if (!supportedSquareMatrixExecution(T, input)) return null;
    return switch (target) {
        .cpu => executeCpuEigh(T, input, max_sweeps, tolerance),
        .cuda => null,
        .mps => null,
    };
}

pub fn executeEighDefault(comptime T: type, input: array_mod.Array(T), max_sweeps: usize, tolerance: T) array_mod.ArrayError!?EighResult(T) {
    return executeEigh(T, defaultTargetForDevice(input.device), input, max_sweeps, tolerance);
}

pub fn executeEigvalsh(comptime T: type, target: DialectBackend, input: array_mod.Array(T), max_sweeps: usize, tolerance: T) array_mod.ArrayError!?array_mod.Array(T) {
    if (try executeEigh(T, target, input, max_sweeps, tolerance)) |result_value| {
        var result = result_value;
        defer result.vectors.deinit();
        return result.values;
    }
    return null;
}

pub fn executeEigvalshDefault(comptime T: type, input: array_mod.Array(T), max_sweeps: usize, tolerance: T) array_mod.ArrayError!?array_mod.Array(T) {
    return executeEigvalsh(T, defaultTargetForDevice(input.device), input, max_sweeps, tolerance);
}

pub fn executeMatrixRank(comptime T: type, target: DialectBackend, input: array_mod.Array(T), tolerance: T) array_mod.ArrayError!?usize {
    if (!supportedMatrixExecution(T, input)) return null;
    return switch (target) {
        .cpu => executeCpuMatrixRank(T, input, tolerance),
        .cuda => null,
        .mps => null,
    };
}

pub fn executeMatrixRankDefault(comptime T: type, input: array_mod.Array(T), tolerance: T) array_mod.ArrayError!?usize {
    return executeMatrixRank(T, defaultTargetForDevice(input.device), input, tolerance);
}

pub fn executeCond(comptime T: type, target: DialectBackend, input: array_mod.Array(T), tolerance: T) array_mod.ArrayError!?T {
    if (!supportedMatrixExecution(T, input)) return null;
    return switch (target) {
        .cpu => executeCpuCond(T, input, tolerance),
        .cuda => null,
        .mps => null,
    };
}

pub fn executeCondDefault(comptime T: type, input: array_mod.Array(T), tolerance: T) array_mod.ArrayError!?T {
    return executeCond(T, defaultTargetForDevice(input.device), input, tolerance);
}

pub fn executePinv(comptime T: type, target: DialectBackend, input: array_mod.Array(T), tolerance: T) array_mod.ArrayError!?array_mod.Array(T) {
    if (!supportedMatrixExecution(T, input)) return null;
    return switch (target) {
        .cpu => executeCpuPinv(T, input, tolerance),
        .cuda => null,
        .mps => null,
    };
}

pub fn executePinvDefault(comptime T: type, input: array_mod.Array(T), tolerance: T) array_mod.ArrayError!?array_mod.Array(T) {
    return executePinv(T, defaultTargetForDevice(input.device), input, tolerance);
}

pub fn executeLstsq(comptime T: type, target: DialectBackend, matrix: array_mod.Array(T), rhs: array_mod.Array(T), tolerance: T) array_mod.ArrayError!?array_mod.Array(T) {
    if (!supportedLstsqExecution(T, matrix, rhs)) return null;
    return switch (target) {
        .cpu => executeCpuLstsq(T, matrix, rhs, tolerance),
        .cuda => null,
        .mps => null,
    };
}

pub fn executeLstsqDefault(comptime T: type, matrix: array_mod.Array(T), rhs: array_mod.Array(T), tolerance: T) array_mod.ArrayError!?array_mod.Array(T) {
    return executeLstsq(T, defaultTargetForDevice(matrix.device), matrix, rhs, tolerance);
}

fn executeCpuLstsq(comptime T: type, matrix: array_mod.Array(T), rhs: array_mod.Array(T), tolerance: T) array_mod.ArrayError!?array_mod.Array(T) {
    const matrix_view = matrixView(T, matrix, "matrix") orelse return null;
    const rhs_view = matrixOrVectorColumnView(T, rhs, "rhs") orelse return null;
    const out_shape = if (rhs.shape.len == 1) &.{matrix.shape[1]} else &.{ matrix.shape[1], rhs.shape[1] };
    var out = try array_mod.Array(T).empty(matrix.allocator, out_shape);
    errdefer out.deinit();
    const out_view = matrixOrVectorColumnView(T, out, "out") orelse {
        out.deinit();
        return null;
    };
    const report = if (T == f32)
        axiom.accelerator.cpu_veyra.runTargetLstsqF32(.cpu, matrix_view, rhs_view, out_view, @as(array_mod.Array(f32), matrix).data, @as(array_mod.Array(f32), rhs).data, @as(array_mod.Array(f32), out).data, @as(f32, tolerance)) catch {
            out.deinit();
            return null;
        }
    else
        axiom.accelerator.cpu_veyra.runTargetLstsqF64(.cpu, matrix_view, rhs_view, out_view, @as(array_mod.Array(f64), matrix).data, @as(array_mod.Array(f64), rhs).data, @as(array_mod.Array(f64), out).data, @as(f64, tolerance)) catch {
            out.deinit();
            return null;
        };
    if (!report.ok()) {
        out.deinit();
        return null;
    }
    return out;
}

fn executeCpuPinv(comptime T: type, input: array_mod.Array(T), tolerance: T) array_mod.ArrayError!?array_mod.Array(T) {
    const matrix_view = matrixView(T, input, "input") orelse return null;
    var out = try array_mod.Array(T).empty(input.allocator, &.{ input.shape[1], input.shape[0] });
    errdefer out.deinit();
    const out_view = matrixView(T, out, "out") orelse {
        out.deinit();
        return null;
    };
    const report = if (T == f32)
        axiom.accelerator.cpu_veyra.runTargetPinvF32(.cpu, matrix_view, out_view, @as(array_mod.Array(f32), input).data, @as(array_mod.Array(f32), out).data, @as(f32, tolerance)) catch {
            out.deinit();
            return null;
        }
    else
        axiom.accelerator.cpu_veyra.runTargetPinvF64(.cpu, matrix_view, out_view, @as(array_mod.Array(f64), input).data, @as(array_mod.Array(f64), out).data, @as(f64, tolerance)) catch {
            out.deinit();
            return null;
        };
    if (!report.ok()) {
        out.deinit();
        return null;
    }
    return out;
}

fn executeCpuCond(comptime T: type, input: array_mod.Array(T), tolerance: T) array_mod.ArrayError!?T {
    const matrix_view = matrixView(T, input, "input") orelse return null;
    if (T == f32) {
        var value: f32 = 0;
        const report = axiom.accelerator.cpu_veyra.runTargetConditionNumberF32(.cpu, matrix_view, @as(array_mod.Array(f32), input).data, @as(f32, tolerance), &value) catch return null;
        if (!report.ok()) return null;
        return @as(T, value);
    } else if (T == f64) {
        var value: f64 = 0;
        const report = axiom.accelerator.cpu_veyra.runTargetConditionNumberF64(.cpu, matrix_view, @as(array_mod.Array(f64), input).data, @as(f64, tolerance), &value) catch return null;
        if (!report.ok()) return null;
        return @as(T, value);
    }
    return null;
}

fn executeCpuMatrixRank(comptime T: type, input: array_mod.Array(T), tolerance: T) array_mod.ArrayError!?usize {
    const matrix_view = matrixView(T, input, "input") orelse return null;
    var value: usize = 0;
    const report = if (T == f32)
        axiom.accelerator.cpu_veyra.runTargetMatrixRankF32(.cpu, matrix_view, @as(array_mod.Array(f32), input).data, @as(f32, tolerance), &value) catch return null
    else
        axiom.accelerator.cpu_veyra.runTargetMatrixRankF64(.cpu, matrix_view, @as(array_mod.Array(f64), input).data, @as(f64, tolerance), &value) catch return null;
    if (!report.ok()) return null;
    return value;
}

fn executeCpuSvd(comptime T: type, input: array_mod.Array(T), tolerance: T) array_mod.ArrayError!?SvdResult(T) {
    const matrix_view = matrixView(T, input, "input") orelse return null;
    const factor_dim = @min(input.shape[0], input.shape[1]);
    var u = try array_mod.Array(T).empty(input.allocator, &.{ input.shape[0], factor_dim });
    errdefer u.deinit();
    var s = try array_mod.Array(T).empty(input.allocator, &.{factor_dim});
    errdefer s.deinit();
    var vt = try array_mod.Array(T).empty(input.allocator, &.{ factor_dim, input.shape[1] });
    errdefer vt.deinit();
    const u_view = matrixView(T, u, "u") orelse return null;
    var s_view = axiom.accelerator.TensorBufferView.contiguous("s", @intCast(@intFromPtr(s.data.ptr)), s.data.len);
    s_view.element_type = if (T == f32) .f32 else if (T == f64) .f64 else return null;
    const vt_view = matrixView(T, vt, "vt") orelse return null;
    const report = if (T == f32)
        axiom.accelerator.cpu_veyra.runTargetSvdF32(.cpu, matrix_view, u_view, s_view, vt_view, @as(array_mod.Array(f32), input).data, @as(array_mod.Array(f32), u).data, @as(array_mod.Array(f32), s).data, @as(array_mod.Array(f32), vt).data, @as(f32, tolerance)) catch return null
    else
        axiom.accelerator.cpu_veyra.runTargetSvdF64(.cpu, matrix_view, u_view, s_view, vt_view, @as(array_mod.Array(f64), input).data, @as(array_mod.Array(f64), u).data, @as(array_mod.Array(f64), s).data, @as(array_mod.Array(f64), vt).data, @as(f64, tolerance)) catch return null;
    if (!report.ok()) return null;
    return .{ .u = u, .s = s, .vt = vt };
}

fn executeCpuSingularValues(comptime T: type, input: array_mod.Array(T), tolerance: T) array_mod.ArrayError!?array_mod.Array(T) {
    const matrix_view = matrixView(T, input, "input") orelse return null;
    const len = @min(input.shape[0], input.shape[1]);
    var out = try array_mod.Array(T).empty(input.allocator, &.{len});
    errdefer out.deinit();
    var out_view = axiom.accelerator.TensorBufferView.contiguous("out", @intCast(@intFromPtr(out.data.ptr)), out.data.len);
    out_view.element_type = if (T == f32) .f32 else if (T == f64) .f64 else return null;
    const report = if (T == f32)
        axiom.accelerator.cpu_veyra.runTargetSingularValuesF32(.cpu, matrix_view, out_view, @as(array_mod.Array(f32), input).data, @as(array_mod.Array(f32), out).data, @as(f32, tolerance)) catch {
            out.deinit();
            return null;
        }
    else
        axiom.accelerator.cpu_veyra.runTargetSingularValuesF64(.cpu, matrix_view, out_view, @as(array_mod.Array(f64), input).data, @as(array_mod.Array(f64), out).data, @as(f64, tolerance)) catch {
            out.deinit();
            return null;
        };
    if (!report.ok()) {
        out.deinit();
        return null;
    }
    return out;
}

fn executeCpuEigh(comptime T: type, input: array_mod.Array(T), max_sweeps: usize, tolerance: T) array_mod.ArrayError!?EighResult(T) {
    const matrix_view = matrixView(T, input, "input") orelse return null;
    var values = try array_mod.Array(T).empty(input.allocator, &.{input.shape[0]});
    errdefer values.deinit();
    var vectors = try array_mod.Array(T).empty(input.allocator, input.shape);
    errdefer vectors.deinit();
    var values_view = axiom.accelerator.TensorBufferView.contiguous("values", @intCast(@intFromPtr(values.data.ptr)), values.data.len);
    values_view.element_type = if (T == f32) .f32 else if (T == f64) .f64 else return null;
    const vectors_view = matrixView(T, vectors, "vectors") orelse {
        vectors.deinit();
        return null;
    };
    const report = if (T == f32)
        axiom.accelerator.cpu_veyra.runTargetSymmetricEigenF32(.cpu, matrix_view, values_view, vectors_view, @as(array_mod.Array(f32), input).data, @as(array_mod.Array(f32), values).data, @as(array_mod.Array(f32), vectors).data, max_sweeps, @as(f32, tolerance)) catch {
            values.deinit();
            vectors.deinit();
            return null;
        }
    else
        axiom.accelerator.cpu_veyra.runTargetSymmetricEigenF64(.cpu, matrix_view, values_view, vectors_view, @as(array_mod.Array(f64), input).data, @as(array_mod.Array(f64), values).data, @as(array_mod.Array(f64), vectors).data, max_sweeps, @as(f64, tolerance)) catch {
            values.deinit();
            vectors.deinit();
            return null;
        };
    if (!report.ok()) {
        values.deinit();
        vectors.deinit();
        return null;
    }
    return .{ .values = values, .vectors = vectors };
}

fn executeCpuMatrixNorm(comptime T: type, input: array_mod.Array(T), order: axiom.accelerator.cpu_veyra.CpuVeyraMatrixNormOrder) array_mod.ArrayError!?T {
    const matrix_view = matrixView(T, input, "input") orelse return null;
    if (T == f32) {
        var value: f32 = 0;
        const report = axiom.accelerator.cpu_veyra.runTargetMatrixNormF32(.cpu, matrix_view, @as(array_mod.Array(f32), input).data, order, &value, normTolerance(T)) catch return null;
        if (!report.ok()) return null;
        return @as(T, value);
    } else if (T == f64) {
        var value: f64 = 0;
        const report = axiom.accelerator.cpu_veyra.runTargetMatrixNormF64(.cpu, matrix_view, @as(array_mod.Array(f64), input).data, order, &value, normTolerance(T)) catch return null;
        if (!report.ok()) return null;
        return @as(T, value);
    }
    return null;
}

fn cpuMatrixNormOrder(order: array_mod.MatrixNormOrder) ?axiom.accelerator.cpu_veyra.CpuVeyraMatrixNormOrder {
    return switch (order) {
        .fro => .fro,
        .one => .one,
        .inf => .inf,
        .two => .two,
        .nuclear => .nuclear,
    };
}

fn normTolerance(comptime T: type) T {
    return if (T == f32) 1e-5 else 1e-12;
}

fn executeCpuSolveTriangular(
    comptime T: type,
    matrix: array_mod.Array(T),
    rhs: array_mod.Array(T),
    triangle: array_mod.Triangle,
    diagonal: array_mod.Diagonal,
) array_mod.ArrayError!?array_mod.Array(T) {
    const matrix_view = matrixView(T, matrix, "matrix") orelse return null;
    const rhs_view = matrixOrVectorColumnView(T, rhs, "rhs") orelse return null;
    const out_shape = if (rhs.shape.len == 1) rhs.shape else &.{ matrix.shape[1], rhs.shape[1] };
    var out = try array_mod.Array(T).empty(matrix.allocator, out_shape);
    errdefer out.deinit();
    const out_view = matrixOrVectorColumnView(T, out, "out") orelse {
        out.deinit();
        return null;
    };
    const cpu_triangle = cpuTriangle(triangle);
    const cpu_diagonal = cpuDiagonal(diagonal);
    const report = if (T == f32)
        axiom.accelerator.cpu_veyra.runTargetSolveTriangularF32(.cpu, matrix_view, rhs_view, out_view, @as(array_mod.Array(f32), matrix).data, @as(array_mod.Array(f32), rhs).data, @as(array_mod.Array(f32), out).data, cpu_triangle, cpu_diagonal) catch {
            out.deinit();
            return null;
        }
    else
        axiom.accelerator.cpu_veyra.runTargetSolveTriangularF64(.cpu, matrix_view, rhs_view, out_view, @as(array_mod.Array(f64), matrix).data, @as(array_mod.Array(f64), rhs).data, @as(array_mod.Array(f64), out).data, cpu_triangle, cpu_diagonal) catch {
            out.deinit();
            return null;
        };
    if (!report.ok()) {
        out.deinit();
        return null;
    }
    return out;
}

fn cpuTriangle(triangle: array_mod.Triangle) axiom.accelerator.cpu_veyra.CpuVeyraTriangle {
    return switch (triangle) {
        .lower => .lower,
        .upper => .upper,
    };
}

fn cpuDiagonal(diagonal: array_mod.Diagonal) axiom.accelerator.cpu_veyra.CpuVeyraDiagonal {
    return switch (diagonal) {
        .non_unit => .non_unit,
        .unit => .unit,
    };
}

fn executeCpuLu(comptime T: type, input: array_mod.Array(T)) array_mod.ArrayError!?LuResult(T) {
    const matrix_view = matrixView(T, input, "input") orelse return null;
    var p = try array_mod.Array(T).empty(input.allocator, input.shape);
    errdefer p.deinit();
    var l = try array_mod.Array(T).empty(input.allocator, input.shape);
    errdefer l.deinit();
    var u = try array_mod.Array(T).empty(input.allocator, input.shape);
    errdefer u.deinit();
    const p_view = matrixView(T, p, "p") orelse return null;
    const l_view = matrixView(T, l, "l") orelse return null;
    const u_view = matrixView(T, u, "u") orelse return null;
    const report = if (T == f32)
        axiom.accelerator.cpu_veyra.runTargetLuF32(.cpu, matrix_view, p_view, l_view, u_view, @as(array_mod.Array(f32), input).data, @as(array_mod.Array(f32), p).data, @as(array_mod.Array(f32), l).data, @as(array_mod.Array(f32), u).data) catch {
            return null;
        }
    else
        axiom.accelerator.cpu_veyra.runTargetLuF64(.cpu, matrix_view, p_view, l_view, u_view, @as(array_mod.Array(f64), input).data, @as(array_mod.Array(f64), p).data, @as(array_mod.Array(f64), l).data, @as(array_mod.Array(f64), u).data) catch {
            return null;
        };
    if (!report.ok()) return null;
    return .{ .p = p, .l = l, .u = u };
}

fn executeCpuQr(comptime T: type, input: array_mod.Array(T)) array_mod.ArrayError!?QrResult(T) {
    const matrix_view = matrixView(T, input, "input") orelse return null;
    var q = try array_mod.Array(T).empty(input.allocator, &.{ input.shape[0], input.shape[0] });
    errdefer q.deinit();
    var r = try array_mod.Array(T).empty(input.allocator, input.shape);
    errdefer r.deinit();
    const q_view = matrixView(T, q, "q") orelse return null;
    const r_view = matrixView(T, r, "r") orelse return null;
    const report = if (T == f32)
        axiom.accelerator.cpu_veyra.runTargetQrF32(.cpu, matrix_view, q_view, r_view, @as(array_mod.Array(f32), input).data, @as(array_mod.Array(f32), q).data, @as(array_mod.Array(f32), r).data) catch {
            return null;
        }
    else
        axiom.accelerator.cpu_veyra.runTargetQrF64(.cpu, matrix_view, q_view, r_view, @as(array_mod.Array(f64), input).data, @as(array_mod.Array(f64), q).data, @as(array_mod.Array(f64), r).data) catch {
            return null;
        };
    if (!report.ok()) return null;
    return .{ .q = q, .r = r };
}

fn executeCpuCholesky(comptime T: type, input: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    const matrix_view = matrixView(T, input, "input") orelse return null;
    var out = try array_mod.Array(T).empty(input.allocator, input.shape);
    errdefer out.deinit();
    const out_view = matrixView(T, out, "out") orelse {
        out.deinit();
        return null;
    };
    const report = if (T == f32)
        axiom.accelerator.cpu_veyra.runTargetCholeskyF32(.cpu, matrix_view, out_view, @as(array_mod.Array(f32), input).data, @as(array_mod.Array(f32), out).data) catch {
            out.deinit();
            return null;
        }
    else
        axiom.accelerator.cpu_veyra.runTargetCholeskyF64(.cpu, matrix_view, out_view, @as(array_mod.Array(f64), input).data, @as(array_mod.Array(f64), out).data) catch {
            out.deinit();
            return null;
        };
    if (!report.ok()) {
        out.deinit();
        return null;
    }
    return out;
}

fn executeCpuSolve(comptime T: type, matrix: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    const matrix_view = matrixView(T, matrix, "matrix") orelse return null;
    const rhs_view = matrixOrVectorColumnView(T, rhs, "rhs") orelse return null;
    const out_shape = if (rhs.shape.len == 1) rhs.shape else &.{ matrix.shape[1], rhs.shape[1] };
    var out = try array_mod.Array(T).empty(matrix.allocator, out_shape);
    errdefer out.deinit();
    const out_view = matrixOrVectorColumnView(T, out, "out") orelse {
        out.deinit();
        return null;
    };
    const report = if (T == f32)
        axiom.accelerator.cpu_veyra.runTargetSolveF32(.cpu, matrix_view, rhs_view, out_view, @as(array_mod.Array(f32), matrix).data, @as(array_mod.Array(f32), rhs).data, @as(array_mod.Array(f32), out).data) catch {
            out.deinit();
            return null;
        }
    else
        axiom.accelerator.cpu_veyra.runTargetSolveF64(.cpu, matrix_view, rhs_view, out_view, @as(array_mod.Array(f64), matrix).data, @as(array_mod.Array(f64), rhs).data, @as(array_mod.Array(f64), out).data) catch {
            out.deinit();
            return null;
        };
    if (!report.ok()) {
        out.deinit();
        return null;
    }
    return out;
}

fn executeCpuInverse(comptime T: type, input: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    const matrix_view = matrixView(T, input, "input") orelse return null;
    var out = try array_mod.Array(T).empty(input.allocator, input.shape);
    errdefer out.deinit();
    const out_view = matrixView(T, out, "out") orelse {
        out.deinit();
        return null;
    };
    const report = if (T == f32)
        axiom.accelerator.cpu_veyra.runTargetInverseF32(.cpu, matrix_view, out_view, @as(array_mod.Array(f32), input).data, @as(array_mod.Array(f32), out).data) catch {
            out.deinit();
            return null;
        }
    else
        axiom.accelerator.cpu_veyra.runTargetInverseF64(.cpu, matrix_view, out_view, @as(array_mod.Array(f64), input).data, @as(array_mod.Array(f64), out).data) catch {
            out.deinit();
            return null;
        };
    if (!report.ok()) {
        out.deinit();
        return null;
    }
    return out;
}

fn executeCpuDet(comptime T: type, input: array_mod.Array(T)) array_mod.ArrayError!?T {
    const matrix_view = matrixView(T, input, "input") orelse return null;
    if (T == f32) {
        var value: f32 = 0;
        const report = axiom.accelerator.cpu_veyra.runTargetDetF32(.cpu, matrix_view, @as(array_mod.Array(f32), input).data, &value) catch return null;
        if (!report.ok()) return null;
        return @as(T, value);
    } else if (T == f64) {
        var value: f64 = 0;
        const report = axiom.accelerator.cpu_veyra.runTargetDetF64(.cpu, matrix_view, @as(array_mod.Array(f64), input).data, &value) catch return null;
        if (!report.ok()) return null;
        return @as(T, value);
    }
    return null;
}

fn executeCpuTrace(comptime T: type, input: array_mod.Array(T), offset: isize) array_mod.ArrayError!?T {
    const matrix_view = matrixView(T, input, "input") orelse return null;
    if (T == f32) {
        var value: f32 = 0;
        const report = axiom.accelerator.cpu_veyra.runTargetTraceF32(.cpu, matrix_view, offset, @as(array_mod.Array(f32), input).data, &value) catch return null;
        if (!report.ok()) return null;
        return @as(T, value);
    } else if (T == f64) {
        var value: f64 = 0;
        const report = axiom.accelerator.cpu_veyra.runTargetTraceF64(.cpu, matrix_view, offset, @as(array_mod.Array(f64), input).data, &value) catch return null;
        if (!report.ok()) return null;
        return @as(T, value);
    }
    return null;
}

fn executeCpuUnary(comptime T: type, op: ExecutionUnaryOp, input: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    const cpu_op: axiom.accelerator.cpu_veyra.TensorUnaryElementwiseOp = switch (op) {
        .abs => .abs,
        .square => .square,
        .sqrt => .sqrt,
        .exp => .exp,
    };
    if (T == f32) {
        var out = try array_mod.Array(f32).empty(input.allocator, input.shape);
        errdefer out.deinit();
        const report = axiom.accelerator.cpu_veyra.runTargetUnaryElementwiseF32(.cpu, cpu_op, @as(array_mod.Array(f32), input).data, out.data) catch {
            out.deinit();
            return null;
        };
        if (!report.ok()) {
            out.deinit();
            return null;
        }
        return @as(array_mod.Array(T), out);
    } else if (T == f64) {
        var out = try array_mod.Array(f64).empty(input.allocator, input.shape);
        errdefer out.deinit();
        const report = axiom.accelerator.cpu_veyra.runTargetUnaryElementwiseF64(.cpu, cpu_op, @as(array_mod.Array(f64), input).data, out.data) catch {
            out.deinit();
            return null;
        };
        if (!report.ok()) {
            out.deinit();
            return null;
        }
        return @as(array_mod.Array(T), out);
    }
    return null;
}

fn executeCudaUnary(comptime T: type, op: ExecutionUnaryOp, input: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (op == .square) return executeCudaElementwise(T, .mul, input, input);
    const cuda_op: axiom_cuda.UnaryOp = switch (op) {
        .abs => .abs,
        .sqrt => .sqrt,
        .exp => .exp,
        .square => unreachable,
    };
    if (T == f32) {
        if (try axiom_cuda.tryDeviceUnaryF32(cuda_op, @as(array_mod.Array(f32), input))) |out| return @as(array_mod.Array(T), out);
    } else if (T == f16) {
        if (try axiom_cuda.tryDeviceUnaryF16(cuda_op, @as(array_mod.Array(f16), input))) |out| return @as(array_mod.Array(T), out);
    } else if (T == f64) {
        if (try axiom_cuda.tryDeviceUnaryF64(cuda_op, @as(array_mod.Array(f64), input))) |out| return @as(array_mod.Array(T), out);
    } else if (T == array_mod.BFloat16) {
        if (try axiom_cuda.tryDeviceUnaryBF16(cuda_op, @as(array_mod.Array(array_mod.BFloat16), input))) |out| return @as(array_mod.Array(T), out);
    }
    return null;
}

fn matrixView(comptime T: type, input: array_mod.Array(T), name: []const u8) ?axiom.accelerator.TensorMatrixView {
    if (input.shape.len != 2 or input.strides.len != 2) return null;
    const row_stride = std.math.cast(isize, input.strides[0]) orelse return null;
    const col_stride = std.math.cast(isize, input.strides[1]) orelse return null;
    var view = axiom.accelerator.TensorMatrixView.strided(
        name,
        @intCast(@intFromPtr(input.data.ptr)),
        input.shape[0],
        input.shape[1],
        row_stride,
        col_stride,
    );
    view.element_type = if (T == f32) .f32 else if (T == f64) .f64 else return null;
    return view;
}

fn matrixOrVectorColumnView(comptime T: type, input: array_mod.Array(T), name: []const u8) ?axiom.accelerator.TensorMatrixView {
    if (input.shape.len == 2) return matrixView(T, input, name);
    if (input.shape.len != 1 or input.strides.len != 1) return null;
    const row_stride = std.math.cast(isize, input.strides[0]) orelse return null;
    var view = axiom.accelerator.TensorMatrixView.strided(name, @intCast(@intFromPtr(input.data.ptr)), input.shape[0], 1, row_stride, 1);
    view.element_type = if (T == f32) .f32 else if (T == f64) .f64 else return null;
    return view;
}

fn broadcastBiasView(comptime T: type, bias: array_mod.Array(T), axis: DialectBroadcastAxis, name: []const u8) ?axiom.accelerator.TensorBufferView {
    if (axis == .row or bias.shape.len == 1) return bufferView(T, bias, name);
    if (bias.shape.len != 2 or bias.shape[1] != 1 or bias.strides.len != 2) return null;
    const stride = std.math.cast(isize, bias.strides[0]) orelse return null;
    var view = axiom.accelerator.TensorBufferView.strided(name, @intCast(@intFromPtr(bias.data.ptr)), bias.shape[0], stride);
    view.element_type = if (T == f32) .f32 else if (T == f64) .f64 else return null;
    return view;
}

fn executeCpuReduction(
    comptime T: type,
    op: DialectReductionOp,
    input: array_mod.Array(T),
    axis: u1,
    keepdims: bool,
) array_mod.ArrayError!?array_mod.Array(T) {
    const cpu_op: axiom.accelerator.cpu_veyra.CpuVeyraReductionOp = switch (op) {
        .sum => .sum,
        .prod => .prod,
        .min => .min,
        .max => .max,
    };
    var out_shape_storage: [2]usize = undefined;
    const out_shape = if (keepdims) shape: {
        out_shape_storage = if (axis == 0)
            .{ 1, input.shape[1] }
        else
            .{ input.shape[0], 1 };
        break :shape out_shape_storage[0..2];
    } else shape: {
        out_shape_storage[0] = if (axis == 0) input.shape[1] else input.shape[0];
        break :shape out_shape_storage[0..1];
    };
    if (T == f32) {
        const input32 = @as(array_mod.Array(f32), input);
        var out = try array_mod.Array(f32).empty(input.allocator, out_shape);
        errdefer out.deinit();
        const matrix_view = matrixView(f32, input32, "input") orelse {
            out.deinit();
            return null;
        };
        var out_view = axiom.accelerator.TensorBufferView.contiguous("out", @intCast(@intFromPtr(out.data.ptr)), out.data.len);
        out_view.element_type = .f32;
        const report = axiom.accelerator.cpu_veyra.runTargetReductionF32(.cpu, cpu_op, axis, matrix_view, out_view, input32.data, out.data) catch {
            out.deinit();
            return null;
        };
        if (!report.ok()) {
            out.deinit();
            return null;
        }
        return @as(array_mod.Array(T), out);
    } else if (T == f64) {
        const input64 = @as(array_mod.Array(f64), input);
        var out = try array_mod.Array(f64).empty(input.allocator, out_shape);
        errdefer out.deinit();
        const matrix_view = matrixView(f64, input64, "input") orelse {
            out.deinit();
            return null;
        };
        var out_view = axiom.accelerator.TensorBufferView.contiguous("out", @intCast(@intFromPtr(out.data.ptr)), out.data.len);
        out_view.element_type = .f64;
        const report = axiom.accelerator.cpu_veyra.runTargetReductionF64(.cpu, cpu_op, axis, matrix_view, out_view, input64.data, out.data) catch {
            out.deinit();
            return null;
        };
        if (!report.ok()) {
            out.deinit();
            return null;
        }
        return @as(array_mod.Array(T), out);
    }
    return null;
}

fn executeCudaReduction(
    comptime T: type,
    op: DialectReductionOp,
    input: array_mod.Array(T),
    axis: u1,
    keepdims: bool,
) array_mod.ArrayError!?array_mod.Array(T) {
    if (T == f32) {
        if (try axiom_cuda.tryDeviceReductionF32(op, @as(array_mod.Array(f32), input), axis, keepdims)) |out| return @as(array_mod.Array(T), out);
    }
    return null;
}

fn executeCpuBroadcastAdd(comptime T: type, input: array_mod.Array(T), bias: array_mod.Array(T), axis: DialectBroadcastAxis) array_mod.ArrayError!?array_mod.Array(T) {
    if (T == f32) {
        const input32 = @as(array_mod.Array(f32), input);
        const bias32 = @as(array_mod.Array(f32), bias);
        var out = try array_mod.Array(f32).empty(input.allocator, input.shape);
        errdefer out.deinit();
        const matrix_view = matrixView(f32, input32, "input") orelse {
            out.deinit();
            return null;
        };
        const bias_view = broadcastBiasView(f32, bias32, axis, "bias") orelse {
            out.deinit();
            return null;
        };
        const out_view = matrixView(f32, out, "out") orelse {
            out.deinit();
            return null;
        };
        const report = axiom.accelerator.cpu_veyra.runTargetBroadcastAddF32(.cpu, axis, matrix_view, bias_view, out_view, input32.data, bias32.data, out.data) catch {
            out.deinit();
            return null;
        };
        if (!report.ok()) {
            out.deinit();
            return null;
        }
        return @as(array_mod.Array(T), out);
    } else if (T == f64) {
        const input64 = @as(array_mod.Array(f64), input);
        const bias64 = @as(array_mod.Array(f64), bias);
        var out = try array_mod.Array(f64).empty(input.allocator, input.shape);
        errdefer out.deinit();
        const matrix_view = matrixView(f64, input64, "input") orelse {
            out.deinit();
            return null;
        };
        const bias_view = broadcastBiasView(f64, bias64, axis, "bias") orelse {
            out.deinit();
            return null;
        };
        const out_view = matrixView(f64, out, "out") orelse {
            out.deinit();
            return null;
        };
        const report = axiom.accelerator.cpu_veyra.runTargetBroadcastAddF64(.cpu, axis, matrix_view, bias_view, out_view, input64.data, bias64.data, out.data) catch {
            out.deinit();
            return null;
        };
        if (!report.ok()) {
            out.deinit();
            return null;
        }
        return @as(array_mod.Array(T), out);
    }
    return null;
}

fn executeCudaBroadcastAdd(comptime T: type, input: array_mod.Array(T), bias: array_mod.Array(T), axis: DialectBroadcastAxis) array_mod.ArrayError!?array_mod.Array(T) {
    if (T == f32) {
        if (try axiom_cuda.tryDeviceBroadcastAddF32(@as(array_mod.Array(f32), input), @as(array_mod.Array(f32), bias), axis)) |out| return @as(array_mod.Array(T), out);
    } else if (T == f64) {
        if (try axiom_cuda.tryDeviceBroadcastAddF64(@as(array_mod.Array(f64), input), @as(array_mod.Array(f64), bias), axis)) |out| return @as(array_mod.Array(T), out);
    }
    return null;
}

fn executeCpuTranspose(comptime T: type, input: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (T == f32) {
        const input32 = @as(array_mod.Array(f32), input);
        var out = try array_mod.Array(f32).empty(input.allocator, &.{ input.shape[1], input.shape[0] });
        errdefer out.deinit();
        const matrix_view = matrixView(f32, input32, "input") orelse {
            out.deinit();
            return null;
        };
        const out_view = matrixView(f32, out, "out") orelse {
            out.deinit();
            return null;
        };
        const report = axiom.accelerator.cpu_veyra.runTargetTransposeF32(.cpu, matrix_view, out_view, input32.data, out.data) catch {
            out.deinit();
            return null;
        };
        if (!report.ok()) {
            out.deinit();
            return null;
        }
        return @as(array_mod.Array(T), out);
    } else if (T == f64) {
        const input64 = @as(array_mod.Array(f64), input);
        var out = try array_mod.Array(f64).empty(input.allocator, &.{ input.shape[1], input.shape[0] });
        errdefer out.deinit();
        const matrix_view = matrixView(f64, input64, "input") orelse {
            out.deinit();
            return null;
        };
        const out_view = matrixView(f64, out, "out") orelse {
            out.deinit();
            return null;
        };
        const report = axiom.accelerator.cpu_veyra.runTargetTransposeF64(.cpu, matrix_view, out_view, input64.data, out.data) catch {
            out.deinit();
            return null;
        };
        if (!report.ok()) {
            out.deinit();
            return null;
        }
        return @as(array_mod.Array(T), out);
    }
    return null;
}

fn executeCudaTranspose(comptime T: type, input: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (T == f32) {
        if (try axiom_cuda.tryDeviceTransposeF32(@as(array_mod.Array(f32), input))) |out| return @as(array_mod.Array(T), out);
    } else if (T == f64) {
        if (try axiom_cuda.tryDeviceTransposeF64(@as(array_mod.Array(f64), input))) |out| return @as(array_mod.Array(T), out);
    }
    return null;
}

fn tensorBinaryOp(op: ElementwiseOp) axiom.accelerator.TensorBinaryElementwiseOp {
    return switch (op) {
        .add => .add,
        .sub => .sub,
        .mul => .mul,
        .div => .div,
    };
}

fn executeCpuElementwiseTarget(comptime T: type, op: ElementwiseOp, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (T == f32) {
        var out = try array_mod.Array(f32).empty(lhs.allocator, lhs.shape);
        errdefer out.deinit();
        const report = axiom.accelerator.cpu_veyra.runTargetElementwiseF32(.cpu, tensorBinaryOp(op), @as(array_mod.Array(f32), lhs).data, @as(array_mod.Array(f32), rhs).data, out.data) catch {
            out.deinit();
            return null;
        };
        if (!report.ok()) {
            out.deinit();
            return null;
        }
        return @as(array_mod.Array(T), out);
    } else if (T == f64) {
        var out = try array_mod.Array(f64).empty(lhs.allocator, lhs.shape);
        errdefer out.deinit();
        const report = axiom.accelerator.cpu_veyra.runTargetElementwiseF64(.cpu, tensorBinaryOp(op), @as(array_mod.Array(f64), lhs).data, @as(array_mod.Array(f64), rhs).data, out.data) catch {
            out.deinit();
            return null;
        };
        if (!report.ok()) {
            out.deinit();
            return null;
        }
        return @as(array_mod.Array(T), out);
    }
    return null;
}

pub fn executeElementwise(
    comptime T: type,
    op: ElementwiseOp,
    target: DialectBackend,
    lhs: array_mod.Array(T),
    rhs: array_mod.Array(T),
) array_mod.ArrayError!?array_mod.Array(T) {
    if (!supportedElementwiseExecution(T, target, lhs, rhs)) return null;
    return switch (target) {
        .cpu => executeCpuElementwiseTarget(T, op, lhs, rhs),
        .cuda => executeCudaElementwise(T, op, lhs, rhs),
        .mps => null,
    };
}

pub fn executeElementwiseDefault(comptime T: type, op: ElementwiseOp, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    return executeElementwise(T, op, defaultTargetForDevice(lhs.device), lhs, rhs);
}

fn executeCudaElementwise(comptime T: type, op: ElementwiseOp, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    const cuda_op = cudaBinaryOp(op);
    if (T == f32) {
        if (try axiom_cuda.tryDeviceBinaryF32(cuda_op, @as(array_mod.Array(f32), lhs), @as(array_mod.Array(f32), rhs))) |out| return @as(array_mod.Array(T), out);
        if (lhs.device.isCpu()) {
            const out = switch (op) {
                .add => try axiom_cuda.tryAddF32(@as(array_mod.Array(f32), lhs), @as(array_mod.Array(f32), rhs)),
                .sub => try axiom_cuda.trySubF32(@as(array_mod.Array(f32), lhs), @as(array_mod.Array(f32), rhs)),
                .mul => try axiom_cuda.tryMulF32(@as(array_mod.Array(f32), lhs), @as(array_mod.Array(f32), rhs)),
                .div => try axiom_cuda.tryDivF32(@as(array_mod.Array(f32), lhs), @as(array_mod.Array(f32), rhs)),
            };
            if (out) |value| return @as(array_mod.Array(T), value);
        }
    } else if (T == f64) {
        if (try axiom_cuda.tryDeviceBinaryF64(cuda_op, @as(array_mod.Array(f64), lhs), @as(array_mod.Array(f64), rhs))) |out| return @as(array_mod.Array(T), out);
        if (lhs.device.isCpu()) {
            if (try axiom_cuda.tryBinaryF64(cuda_op, @as(array_mod.Array(f64), lhs), @as(array_mod.Array(f64), rhs))) |out| return @as(array_mod.Array(T), out);
        }
    } else if (T == array_mod.BFloat16) {
        if (try axiom_cuda.tryDeviceBinaryBF16(cuda_op, @as(array_mod.Array(array_mod.BFloat16), lhs), @as(array_mod.Array(array_mod.BFloat16), rhs))) |out| return @as(array_mod.Array(T), out);
        if (lhs.device.isCpu()) {
            const out = switch (op) {
                .add => try axiom_cuda.tryAddBF16(@as(array_mod.Array(array_mod.BFloat16), lhs), @as(array_mod.Array(array_mod.BFloat16), rhs)),
                .sub => try axiom_cuda.trySubBF16(@as(array_mod.Array(array_mod.BFloat16), lhs), @as(array_mod.Array(array_mod.BFloat16), rhs)),
                .mul => try axiom_cuda.tryMulBF16(@as(array_mod.Array(array_mod.BFloat16), lhs), @as(array_mod.Array(array_mod.BFloat16), rhs)),
                .div => try axiom_cuda.tryDivBF16(@as(array_mod.Array(array_mod.BFloat16), lhs), @as(array_mod.Array(array_mod.BFloat16), rhs)),
            };
            if (out) |value| return @as(array_mod.Array(T), value);
        }
    } else if (T == f16) {
        if (try axiom_cuda.tryDeviceBinaryF16(cuda_op, @as(array_mod.Array(f16), lhs), @as(array_mod.Array(f16), rhs))) |out| return @as(array_mod.Array(T), out);
        if (lhs.device.isCpu()) {
            const out = switch (op) {
                .add => try axiom_cuda.tryAddF16(@as(array_mod.Array(f16), lhs), @as(array_mod.Array(f16), rhs)),
                .sub => try axiom_cuda.trySubF16(@as(array_mod.Array(f16), lhs), @as(array_mod.Array(f16), rhs)),
                .mul => try axiom_cuda.tryMulF16(@as(array_mod.Array(f16), lhs), @as(array_mod.Array(f16), rhs)),
                .div => try axiom_cuda.tryDivF16(@as(array_mod.Array(f16), lhs), @as(array_mod.Array(f16), rhs)),
            };
            if (out) |value| return @as(array_mod.Array(T), value);
        }
    }
    return null;
}

fn cudaBinaryOp(op: ElementwiseOp) axiom_cuda.BinaryOp {
    return switch (op) {
        .add => .add,
        .sub => .sub,
        .mul => .mul,
        .div => .div,
    };
}

pub fn selectMatmul(comptime T: type, policy: BackendPolicy, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) BackendReport {
    const supported = supportedMatmul2d(T, lhs, rhs);
    const selected: BackendRoute = if (!supported)
        .direct_cpu
    else switch (policy) {
        .force_direct_cpu => .direct_cpu,
        .prefer_axiom_cpu => if (supportsAxiomCpuMatmul(T) and axiom_cpu.enabled()) .axiom_cpu_veyra else if (supportsAxiomCudaMatmul(T) and axiom_cuda.enabled()) .axiom_cuda else .direct_cpu,
        .prefer_cuda => if (supportsAxiomCudaMatmul(T) and axiom_cuda.enabled()) .axiom_cuda else if (supportsAxiomCpuMatmul(T) and axiom_cpu.enabled()) .axiom_cpu_veyra else .direct_cpu,
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
        .prefer_axiom_cpu => if (supportsAxiomCpuElementwise(T) and axiom_cpu.enabled()) .axiom_cpu_veyra else if (supportsAxiomCudaElementwise(T) and axiom_cuda.enabled()) .axiom_cuda else .direct_cpu,
        .prefer_cuda => if (supportsAxiomCudaElementwise(T) and axiom_cuda.enabled()) .axiom_cuda else if (supportsAxiomCpuElementwise(T) and axiom_cpu.enabled()) .axiom_cpu_veyra else .direct_cpu,
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

pub fn selectScalarElementwise(
    comptime T: type,
    op: ElementwiseOp,
    policy: BackendPolicy,
    input: array_mod.Array(T),
    scalar: T,
    scalar_side: ScalarSide,
) BackendReport {
    const supported = supportedScalarElementwise(T, input);
    const selected: BackendRoute = if (!supported)
        .direct_cpu
    else switch (policy) {
        .force_direct_cpu => .direct_cpu,
        .prefer_axiom_cpu => if (supportsAxiomCpuElementwise(T) and axiom_cpu.enabled()) .axiom_cpu_veyra else if (supportsAxiomCudaElementwise(T) and axiom_cuda.enabled()) .axiom_cuda else .direct_cpu,
        .prefer_cuda => if (supportsAxiomCudaElementwise(T) and axiom_cuda.enabled()) .axiom_cuda else if (supportsAxiomCpuElementwise(T) and axiom_cpu.enabled()) .axiom_cpu_veyra else .direct_cpu,
    };
    var report: BackendReport = .{
        .policy = policy,
        .selected = selected,
        .dtype_name = @typeName(T),
        .supported_shape = supported,
    };
    report.fingerprint_value = computeScalarElementwiseFingerprint(T, op, input, scalar, scalar_side, selected);
    return report;
}

pub fn elementwise(comptime T: type, op: ElementwiseOp, policy: BackendPolicy, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!array_mod.Array(T) {
    const target = policyExecutionTarget(policy);
    if (try executeElementwise(T, op, target, lhs, rhs)) |out| return out;
    if (!lhs.device.isCpu()) return error.BackendFailure;
    return directElementwise(T, op, lhs, rhs);
}

pub fn executeElementwiseScalar(
    comptime T: type,
    op: ElementwiseOp,
    target: DialectBackend,
    input: array_mod.Array(T),
    scalar: T,
    scalar_side: ScalarSide,
) array_mod.ArrayError!?array_mod.Array(T) {
    if (!supportedScalarElementwise(T, input)) return null;
    var scalar_array = try array_mod.Array(T).fullOn(input.allocator, input.shape, scalar, input.device);
    defer scalar_array.deinit();
    return switch (scalar_side) {
        .lhs => executeElementwise(T, op, target, scalar_array, input),
        .rhs => executeElementwise(T, op, target, input, scalar_array),
    };
}

pub fn executeElementwiseScalarDefault(
    comptime T: type,
    op: ElementwiseOp,
    input: array_mod.Array(T),
    scalar: T,
    scalar_side: ScalarSide,
) array_mod.ArrayError!?array_mod.Array(T) {
    return executeElementwiseScalar(T, op, defaultTargetForDevice(input.device), input, scalar, scalar_side);
}

pub fn elementwiseScalar(
    comptime T: type,
    op: ElementwiseOp,
    policy: BackendPolicy,
    input: array_mod.Array(T),
    scalar: T,
    scalar_side: ScalarSide,
) array_mod.ArrayError!array_mod.Array(T) {
    const target = policyExecutionTarget(policy);
    if (try executeElementwiseScalar(T, op, target, input, scalar, scalar_side)) |out| return out;
    if (!input.device.isCpu()) return error.BackendFailure;
    return directScalarElementwise(T, op, input, scalar, scalar_side);
}

pub fn tryElementwiseScalarBroadcastDefault(comptime T: type, op: ElementwiseOp, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (lhs.data.len == rhs.data.len) return null;
    if (lhs.data.len == 1 and rhs.data.len != 0 and scalarBroadcastPreservesVectorShape(lhs.shape, rhs.shape)) return try executeElementwiseScalarDefault(T, op, rhs, lhs.data[0], .lhs);
    if (rhs.data.len == 1 and lhs.data.len != 0 and scalarBroadcastPreservesVectorShape(rhs.shape, lhs.shape)) return try executeElementwiseScalarDefault(T, op, lhs, rhs.data[0], .rhs);
    return null;
}

pub fn tryElementwiseScalarBroadcast(comptime T: type, op: ElementwiseOp, policy: BackendPolicy, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    const target = policyExecutionTarget(policy);
    if (lhs.data.len == rhs.data.len) return null;
    if (lhs.data.len == 1 and rhs.data.len != 0 and scalarBroadcastPreservesVectorShape(lhs.shape, rhs.shape)) return try executeElementwiseScalar(T, op, target, rhs, lhs.data[0], .lhs);
    if (rhs.data.len == 1 and lhs.data.len != 0 and scalarBroadcastPreservesVectorShape(rhs.shape, lhs.shape)) return try executeElementwiseScalar(T, op, target, lhs, rhs.data[0], .rhs);
    return null;
}

fn directElementwise(comptime T: type, op: ElementwiseOp, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!array_mod.Array(T) {
    if (!std.mem.eql(usize, lhs.shape, rhs.shape)) return error.ShapeMismatch;
    var out = try array_mod.Array(T).empty(lhs.allocator, lhs.shape);
    errdefer out.deinit();
    for (lhs.data, rhs.data, out.data) |a, b, *slot| slot.* = elementwiseValue(T, op, a, b);
    return out;
}

fn directScalarElementwise(comptime T: type, op: ElementwiseOp, input: array_mod.Array(T), scalar: T, scalar_side: ScalarSide) array_mod.ArrayError!array_mod.Array(T) {
    var out = try array_mod.Array(T).empty(input.allocator, input.shape);
    errdefer out.deinit();
    for (input.data, out.data) |value, *slot| {
        const lhs = if (scalar_side == .lhs) scalar else value;
        const rhs = if (scalar_side == .lhs) value else scalar;
        slot.* = elementwiseValue(T, op, lhs, rhs);
    }
    return out;
}

pub fn matmul(comptime T: type, policy: BackendPolicy, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!array_mod.Array(T) {
    if (try executeMatmul(T, policyExecutionTarget(policy), lhs, rhs)) |out| return out;
    if (!lhs.device.isCpu()) return error.BackendFailure;
    return directMatmul(T, lhs, rhs);
}

fn directMatmul(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!array_mod.Array(T) {
    if (!supportedMatmul2d(T, lhs, rhs)) return error.NonMatrixArray;
    var out = try array_mod.Array(T).zeros(lhs.allocator, &.{ lhs.shape[0], rhs.shape[1] });
    errdefer out.deinit();
    for (0..lhs.shape[0]) |row| {
        for (0..rhs.shape[1]) |col| {
            var acc = zeroValue(T);
            for (0..lhs.shape[1]) |kk| {
                acc = elementwiseValue(T, .add, acc, elementwiseValue(T, .mul, lhs.data[row * lhs.shape[1] + kk], rhs.data[kk * rhs.shape[1] + col]));
            }
            out.data[row * rhs.shape[1] + col] = acc;
        }
    }
    return out;
}

// Dialect lowering is a structural contract: it models the operation's element
// type, shape/layout constraints, and requested Axiom target.  It must not
// require host slices, CPU-only arrays, or executable device storage; those are
// eager runtime ABI constraints and stay in the `*Execution` helpers below.
// Keeping the predicates split lets Vectra behave like an MLIR-style frontend:
// arrays describe linalg/memref/gpu work for `.cpu/.cuda/.mps`, while
// `RuntimeCapabilityReport` says whether that lowered program can run today.
fn supportedMatmulLowering2d(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) bool {
    return lhs.device.sameDevice(rhs.device) and lhs.shape.len == 2 and rhs.shape.len == 2 and lhs.shape[1] == rhs.shape[0] and lhs.isContiguous() and rhs.isContiguous();
}

fn supportedMatmul2d(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) bool {
    return lhs.device.isCpu() and supportedMatmulLowering2d(T, lhs, rhs);
}

fn supportedMatmulExecution(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) bool {
    if (!lhs.device.sameDevice(rhs.device) or !lhs.isContiguous() or !rhs.isContiguous()) return false;
    if (lhs.shape.len == 0 or rhs.shape.len == 0) return false;
    const lhs_k = lhs.shape[lhs.shape.len - 1];
    const rhs_k = if (rhs.shape.len == 1) rhs.shape[0] else rhs.shape[rhs.shape.len - 2];
    if (lhs_k != rhs_k) return false;
    if (lhs.device.isCpu()) {
        return (T == f32 or T == f64) and
            (lhs.shape.len == 1 or lhs.shape.len == 2) and
            (rhs.shape.len == 1 or rhs.shape.len == 2);
    }
    return lhs.device.isCuda() and lhs.shape.len == 2 and rhs.shape.len == 2 and (T == f32 or T == f64 or T == f16 or T == array_mod.BFloat16);
}

fn supportedMatmulAddExecution(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T), addend: array_mod.Array(T)) bool {
    if (!lhs.device.sameDevice(rhs.device) or !lhs.device.sameDevice(addend.device)) return false;
    if (lhs.shape.len != 2 or rhs.shape.len != 2 or addend.shape.len != 2) return false;
    if (lhs.shape[1] != rhs.shape[0] or addend.shape[0] != lhs.shape[0] or addend.shape[1] != rhs.shape[1]) return false;
    if (!lhs.isContiguous() or !rhs.isContiguous() or !addend.isContiguous()) return false;
    if (lhs.device.isCpu()) return T == f32 or T == f64;
    return lhs.device.isCuda() and (T == f32 or T == f64 or T == f16 or T == array_mod.BFloat16);
}

fn supportedReduction2d(comptime T: type, input: array_mod.Array(T)) bool {
    return dialectElement(T) != null and input.device.isCpu() and input.shape.len == 2 and input.isContiguous();
}

fn supportedReductionExecution(comptime T: type, target: DialectBackend, input: array_mod.Array(T)) bool {
    if (input.shape.len != 2 or !input.isContiguous()) return false;
    return switch (target) {
        .cpu => supportedReduction2d(T, input),
        .cuda => input.device.isCuda() and T == f32 and input.device_storage != null,
        .mps => false,
    };
}

fn supportedReductionLowering2d(comptime T: type, input: array_mod.Array(T)) bool {
    return dialectElement(T) != null and input.shape.len == 2 and input.isContiguous();
}

fn supportedUnary2d(comptime T: type, input: array_mod.Array(T)) bool {
    return dialectElement(T) != null and input.device.isCpu() and input.shape.len == 2 and input.isContiguous();
}

fn supportedTransposeExecution(comptime T: type, target: DialectBackend, input: array_mod.Array(T)) bool {
    if (input.shape.len != 2 or !input.isContiguous()) return false;
    return switch (target) {
        .cpu => supportedUnary2d(T, input),
        .cuda => input.device.isCuda() and (T == f32 or T == f64) and input.device_storage != null,
        .mps => false,
    };
}

fn supportedUnaryLowering2d(comptime T: type, input: array_mod.Array(T)) bool {
    return dialectElement(T) != null and input.shape.len == 2 and input.isContiguous();
}

fn supportedMatrixExecution(comptime T: type, input: array_mod.Array(T)) bool {
    return (T == f32 or T == f64) and
        input.device.isCpu() and
        input.shape.len == 2 and
        input.data.len != 0 and
        input.isContiguous();
}

fn supportedSquareMatrixExecution(comptime T: type, input: array_mod.Array(T)) bool {
    return supportedMatrixExecution(T, input) and input.shape[0] == input.shape[1];
}

fn supportedSolveExecution(comptime T: type, matrix: array_mod.Array(T), rhs: array_mod.Array(T)) bool {
    const rhs_rank_ok = rhs.shape.len == 1 or rhs.shape.len == 2;
    return supportedSquareMatrixExecution(T, matrix) and
        rhs.device.isCpu() and
        rhs_rank_ok and
        rhs.shape[0] == matrix.shape[0] and
        rhs.data.len != 0 and
        rhs.isContiguous();
}

fn supportedLstsqExecution(comptime T: type, matrix: array_mod.Array(T), rhs: array_mod.Array(T)) bool {
    const rhs_rank_ok = rhs.shape.len == 1 or rhs.shape.len == 2;
    return supportedMatrixExecution(T, matrix) and
        rhs.device.isCpu() and
        rhs_rank_ok and
        rhs.shape[0] == matrix.shape[0] and
        rhs.data.len != 0 and
        rhs.isContiguous();
}

fn supportedUnaryExecution(comptime T: type, input: array_mod.Array(T)) bool {
    if (!(T == f32 or T == f64 or T == f16 or T == array_mod.BFloat16)) return false;
    return (input.device.isCpu() or input.device.isCuda()) and
        nonEmptyAccessibleData(T, input) and
        input.isContiguous();
}

fn supportedBroadcastAdd(comptime T: type, input: array_mod.Array(T), bias: array_mod.Array(T), axis: DialectBroadcastAxis) bool {
    if (dialectElement(T) == null) return false;
    if (!input.device.isCpu() or !bias.device.isCpu() or input.shape.len != 2) return false;
    if (!input.isContiguous() or !bias.isContiguous()) return false;
    return broadcastBiasMatchesArrayAdd(T, input, bias, axis);
}

fn supportedBroadcastAddExecution(comptime T: type, target: DialectBackend, input: array_mod.Array(T), bias: array_mod.Array(T), axis: DialectBroadcastAxis) bool {
    return broadcastAddRuntimeCapability(target).executable() and
        targetCanAccessDevice(target, input.device) and
        input.device.sameDevice(bias.device) and
        switch (target) {
            .cpu => supportedBroadcastAdd(T, input, bias, axis),
            .cuda => (T == f32 or T == f64) and input.device.isCuda() and supportedBroadcastAddLowering(T, input, bias, axis),
            .mps => false,
        };
}

fn broadcastBiasMatches(comptime T: type, input: array_mod.Array(T), bias: array_mod.Array(T), axis: DialectBroadcastAxis) bool {
    if (!input.device.sameDevice(bias.device) or input.shape.len != 2) return false;
    return switch (axis) {
        .row => bias.shape.len == 1 and bias.shape[0] == input.shape[1],
        .column => bias.shape.len == 1 and bias.shape[0] == input.shape[0],
    };
}

fn broadcastBiasMatchesArrayAdd(comptime T: type, input: array_mod.Array(T), bias: array_mod.Array(T), axis: DialectBroadcastAxis) bool {
    if (broadcastBiasMatches(T, input, bias, axis)) return true;
    return switch (axis) {
        .row => false,
        .column => input.device.sameDevice(bias.device) and
            input.shape.len == 2 and
            bias.shape.len == 2 and
            bias.shape[0] == input.shape[0] and
            bias.shape[1] == 1,
    };
}

fn supportedBroadcastAddLowering(comptime T: type, input: array_mod.Array(T), bias: array_mod.Array(T), axis: DialectBroadcastAxis) bool {
    if (dialectElement(T) == null) return false;
    if (!input.device.sameDevice(bias.device) or input.shape.len != 2) return false;
    if (!input.isContiguous() or !bias.isContiguous()) return false;
    return broadcastBiasMatchesArrayAdd(T, input, bias, axis);
}

fn supportedElementwiseLowering(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) bool {
    return dialectElement(T) != null and
        (lhs.shape.len == 1 or lhs.shape.len == 2) and
        lhs.device.sameDevice(rhs.device) and
        lhs.sameShape(rhs) and
        lhs.isContiguous() and
        rhs.isContiguous();
}

fn supportedElementwiseSameShapeContiguous(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) bool {
    return supportedElementwiseExecution(T, defaultTargetForDevice(lhs.device), lhs, rhs);
}

fn supportedElementwiseExecution(comptime T: type, target: DialectBackend, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) bool {
    return supportsAxiomElementwise(T) and
        lhs.device.sameDevice(rhs.device) and
        targetCanAccessDevice(target, lhs.device) and
        nonEmptyAccessibleData(T, lhs) and
        nonEmptyAccessibleData(T, rhs) and
        lhs.sameShape(rhs) and
        lhs.isContiguous() and
        rhs.isContiguous();
}

fn supportsAxiomElementwise(comptime T: type) bool {
    return T == f32 or T == f64 or T == f16 or T == array_mod.BFloat16;
}

fn supportsAxiomCudaElementwise(comptime T: type) bool {
    return T == f32 or T == f64 or T == f16 or T == array_mod.BFloat16;
}

fn supportsAxiomCpuElementwise(comptime T: type) bool {
    return T == f32 or T == f64;
}

fn supportsAxiomCpuMatmul(comptime T: type) bool {
    return T == f32 or T == f64;
}

fn supportsAxiomCudaMatmul(comptime T: type) bool {
    return T == f32 or T == f64 or T == f16 or T == array_mod.BFloat16;
}

fn supportedScalarElementwise(comptime T: type, input: array_mod.Array(T)) bool {
    return supportsAxiomElementwise(T) and
        (input.device.isCpu() or input.device.isCuda()) and
        nonEmptyAccessibleData(T, input) and
        input.isContiguous();
}

fn nonEmptyAccessibleData(comptime T: type, input: array_mod.Array(T)) bool {
    if (input.device.isCuda()) {
        const storage = input.device_storage orelse return false;
        return storage.len != 0;
    }
    return input.data.len != 0;
}

fn scalarBroadcastPreservesVectorShape(scalar_shape: []const usize, vector_shape: []const usize) bool {
    if (scalar_shape.len > vector_shape.len) return false;
    var scalar_index = scalar_shape.len;
    var vector_index = vector_shape.len;
    while (scalar_index > 0) {
        scalar_index -= 1;
        vector_index -= 1;
        const scalar_dim = scalar_shape[scalar_index];
        const vector_dim = vector_shape[vector_index];
        if (scalar_dim != 1 and scalar_dim != vector_dim) return false;
    }
    return true;
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

fn computeScalarElementwiseFingerprint(comptime T: type, op: ElementwiseOp, input: array_mod.Array(T), scalar: T, scalar_side: ScalarSide, selected: BackendRoute) u64 {
    var hasher = std.hash.Wyhash.init(0x0abc_beef_0004);
    hashBytes(&hasher, @typeName(T));
    hashBytes(&hasher, op.label());
    hashBytes(&hasher, scalar_side.label());
    hashBytes(&hasher, selected.label());
    for (input.shape) |dim| hashU64(&hasher, dim);
    hashElementValue(T, &hasher, scalar);
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

fn hashElementValue(comptime T: type, hasher: *std.hash.Wyhash, value: T) void {
    if (T == f32) {
        var bytes: [4]u8 = undefined;
        std.mem.writeInt(u32, &bytes, @bitCast(value), .little);
        hasher.update(&bytes);
    } else if (T == f64) {
        var bytes: [8]u8 = undefined;
        std.mem.writeInt(u64, &bytes, @bitCast(value), .little);
        hasher.update(&bytes);
    } else if (T == f16) {
        var bytes: [2]u8 = undefined;
        std.mem.writeInt(u16, &bytes, @bitCast(value), .little);
        hasher.update(&bytes);
    } else if (T == array_mod.BFloat16) {
        var bytes: [2]u8 = undefined;
        std.mem.writeInt(u16, &bytes, value.bits, .little);
        hasher.update(&bytes);
    } else {
        hashU64(hasher, value);
    }
}

fn zeroValue(comptime T: type) T {
    if (T == array_mod.BFloat16) return array_mod.BFloat16.fromF32(0);
    return 0;
}

fn elementwiseValue(comptime T: type, op: ElementwiseOp, lhs: T, rhs: T) T {
    if (T == array_mod.BFloat16) {
        return switch (op) {
            .add => lhs.add(rhs),
            .sub => lhs.sub(rhs),
            .mul => lhs.mul(rhs),
            .div => lhs.div(rhs),
        };
    }
    return switch (op) {
        .add => lhs + rhs,
        .sub => lhs - rhs,
        .mul => lhs * rhs,
        .div => lhs / rhs,
    };
}

test "Axiom dialect lowering reports linalg memref gpu route" {
    const gpa = std.testing.allocator;
    var a = try array_mod.Array(f32).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a.deinit();
    var b = try array_mod.Array(f32).fromSlice(gpa, &.{ 7, 8, 9, 10, 11, 12 }, &.{ 3, 2 });
    defer b.deinit();

    const cpu_report = try lowerMatmulDialect(f32, a, b, .cpu);
    try std.testing.expect(cpu_report.ok());
    try std.testing.expectEqual(DialectMatmulLoweringStatus.lowered_cpu, cpu_report.status);
    try std.testing.expect(cpu_report.registration.ok());

    const cuda_report = try lowerMatmulDialect(f32, a, b, .cuda);
    try std.testing.expect(cuda_report.ok());
    try std.testing.expectEqual(DialectMatmulLoweringStatus.lowered_cuda, cuda_report.status);
    try std.testing.expect(cuda_report.cuda_tile_projection_fingerprint != 0);

    const mps_report = try lowerMatmulDialect(f32, a, b, .mps);
    try std.testing.expect(mps_report.ok());
    try std.testing.expectEqual(DialectMatmulLoweringStatus.planned_mps, mps_report.status);

    resetDefaultDialectBackend();
    try std.testing.expectEqual(DialectBackend.cpu, defaultDialectBackend());
    setDefaultDialectBackend(.cuda);
    const default_cuda_report = try lowerMatmulDialectDefault(f32, a, b);
    try std.testing.expect(default_cuda_report.ok());
    try std.testing.expectEqual(DialectMatmulLoweringStatus.lowered_cuda, default_cuda_report.status);
    setDefaultDialectBackend(.mps);
    const default_mps_report = try lowerMatmulDialectDefault(f32, a, b);
    try std.testing.expect(default_mps_report.ok());
    try std.testing.expectEqual(DialectMatmulLoweringStatus.planned_mps, default_mps_report.status);
    resetDefaultDialectBackend();

    setDefaultDialectBackend(.cuda);
    try std.testing.expectEqual(BackendPolicy.prefer_cuda, defaultBackendPolicy());
    setDefaultDialectBackend(.mps);
    try std.testing.expectEqual(BackendPolicy.prefer_axiom_cpu, defaultBackendPolicy());
    resetDefaultDialectBackend();
}

test "Axiom dialect lowering reports elementwise generic route" {
    const gpa = std.testing.allocator;
    var lhs = try array_mod.Array(f32).fromSlice(gpa, &.{ 1, 2, 3, 4 }, &.{ 2, 2 });
    defer lhs.deinit();
    var rhs = try array_mod.Array(f32).fromSlice(gpa, &.{ 10, 20, 30, 40 }, &.{ 2, 2 });
    defer rhs.deinit();

    const cuda_report = try lowerElementwiseDialect(f32, .add, lhs, rhs, .cuda);
    try std.testing.expect(cuda_report.ok());
    try std.testing.expectEqual(DialectElementwiseLoweringStatus.lowered_cuda, cuda_report.status);
    try std.testing.expect(cuda_report.vector_fragment_fingerprint != 0);
    try std.testing.expect(cuda_report.gpu_mapping_fingerprint != 0);

    setDefaultDialectBackend(.mps);
    const default_mps_report = try lowerElementwiseDialectDefault(f32, .mul, lhs, rhs);
    try std.testing.expect(default_mps_report.ok());
    try std.testing.expectEqual(DialectElementwiseLoweringStatus.planned_mps, default_mps_report.status);
    resetDefaultDialectBackend();
}

test "Axiom dialect lowering reports reduction generic route" {
    const gpa = std.testing.allocator;
    var input = try array_mod.Array(f32).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer input.deinit();

    const cuda_report = try lowerReductionDialect(f32, input, .sum, 1, .cuda);
    try std.testing.expect(cuda_report.ok());
    try std.testing.expectEqual(DialectReductionLoweringStatus.lowered_cuda, cuda_report.status);
    try std.testing.expect(cuda_report.vector_fragment_fingerprint != 0);
    try std.testing.expect(cuda_report.gpu_mapping_fingerprint != 0);

    setDefaultDialectBackend(.mps);
    const default_mps_report = try lowerReductionDialectDefault(f32, input, .max, 0);
    try std.testing.expect(default_mps_report.ok());
    try std.testing.expectEqual(DialectReductionLoweringStatus.planned_mps, default_mps_report.status);
    resetDefaultDialectBackend();
}

test "Axiom dialect lowering reports broadcast generic route" {
    const gpa = std.testing.allocator;
    var input = try array_mod.Array(f32).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer input.deinit();
    var row = try array_mod.Array(f32).fromSlice(gpa, &.{ 10, 20, 30 }, &.{3});
    defer row.deinit();
    var column = try array_mod.Array(f32).fromSlice(gpa, &.{ 100, 200 }, &.{2});
    defer column.deinit();
    var column2d = try array_mod.Array(f32).fromSlice(gpa, &.{ 100, 200 }, &.{ 2, 1 });
    defer column2d.deinit();

    const cuda_report = try lowerBroadcastAddDialect(f32, input, row, .row, .cuda);
    try std.testing.expect(cuda_report.ok());
    try std.testing.expectEqual(DialectBroadcastLoweringStatus.lowered_cuda, cuda_report.status);
    try std.testing.expect(cuda_report.vector_fragment_fingerprint != 0);
    try std.testing.expect(cuda_report.gpu_mapping_fingerprint != 0);
    const cpu_runtime = broadcastAddRuntimeCapability(.cpu);
    try std.testing.expect(cpu_runtime.executable());
    var row_out = (try executeBroadcastAdd(f32, .cpu, input, row, .row)) orelse return error.BackendFailure;
    defer row_out.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 33, 14, 25, 36 }, row_out.data);
    var reversed_row_out = (try tryBroadcastAdd(f32, .cpu, row, input)) orelse return error.BackendFailure;
    defer reversed_row_out.deinit();
    try std.testing.expectEqualSlices(f32, row_out.data, reversed_row_out.data);
    var column2d_out = (try tryBroadcastAdd(f32, .cpu, input, column2d)) orelse return error.BackendFailure;
    defer column2d_out.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 101, 102, 103, 204, 205, 206 }, column2d_out.data);

    setDefaultDialectBackend(.mps);
    const default_mps_report = try lowerBroadcastAddDialectDefault(f32, input, column, .column);
    try std.testing.expect(default_mps_report.ok());
    try std.testing.expectEqual(DialectBroadcastLoweringStatus.planned_mps, default_mps_report.status);
    resetDefaultDialectBackend();
}

test "Axiom dialect lowering reports unary generic route" {
    const gpa = std.testing.allocator;
    var input = try array_mod.Array(f32).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer input.deinit();

    const cuda_report = try lowerUnaryDialect(f32, input, .square, .cuda);
    try std.testing.expect(cuda_report.ok());
    try std.testing.expectEqual(DialectUnaryLoweringStatus.lowered_cuda, cuda_report.status);
    try std.testing.expect(cuda_report.vector_fragment_fingerprint != 0);
    try std.testing.expect(cuda_report.gpu_mapping_fingerprint != 0);

    setDefaultDialectBackend(.mps);
    const default_mps_report = try lowerUnaryDialectDefault(f32, input, .cube);
    try std.testing.expect(default_mps_report.ok());
    try std.testing.expectEqual(DialectUnaryLoweringStatus.planned_mps, default_mps_report.status);
    resetDefaultDialectBackend();
}

test "Axiom dialect lowering reports transpose generic route" {
    const gpa = std.testing.allocator;
    var input = try array_mod.Array(f32).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer input.deinit();

    const cuda_report = try lowerTransposeDialect(f32, input, .cuda);
    try std.testing.expect(cuda_report.ok());
    try std.testing.expectEqual(DialectTransposeLoweringStatus.lowered_cuda, cuda_report.status);
    try std.testing.expect(cuda_report.vector_fragment_fingerprint != 0);
    try std.testing.expect(cuda_report.gpu_mapping_fingerprint != 0);

    setDefaultDialectBackend(.mps);
    const default_mps_report = try lowerTransposeDialectDefault(f32, input);
    try std.testing.expect(default_mps_report.ok());
    try std.testing.expectEqual(DialectTransposeLoweringStatus.planned_mps, default_mps_report.status);
    resetDefaultDialectBackend();
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

    const scalar_report = selectScalarElementwise(f64, .sub, .prefer_axiom_cpu, lhs64, 2.0, .rhs);
    try std.testing.expect(scalar_report.ok());
    var scalar_out = try elementwiseScalar(f64, .sub, .prefer_axiom_cpu, lhs64, 2.0, .rhs);
    defer scalar_out.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 6, 4, 2, 0 }, scalar_out.data);

    var scalar_lhs = try array_mod.Array(f32).fromSlice(gpa, &.{2}, &.{1});
    defer scalar_lhs.deinit();
    const scalar_broadcast = try tryElementwiseScalarBroadcast(f32, .sub, .prefer_cuda, scalar_lhs, rhs32);
    try std.testing.expect(scalar_broadcast != null);
    var scalar_broadcast_out = scalar_broadcast.?;
    defer scalar_broadcast_out.deinit();
    try std.testing.expectEqualSlices(f32, &.{ -8, -18, -28, -38 }, scalar_broadcast_out.data);

    var leading_singleton = try array_mod.Array(f32).fromSlice(gpa, &.{2}, &.{ 1, 1, 1 });
    defer leading_singleton.deinit();
    const unsupported_scalar_broadcast = try tryElementwiseScalarBroadcast(f32, .add, .prefer_cuda, leading_singleton, rhs32);
    try std.testing.expect(unsupported_scalar_broadcast == null);

    var lhs_bf16 = try array_mod.Array(array_mod.BFloat16).fromSlice(gpa, &.{
        array_mod.BFloat16.fromF32(1),
        array_mod.BFloat16.fromF32(2),
        array_mod.BFloat16.fromF32(3),
        array_mod.BFloat16.fromF32(4),
    }, &.{ 2, 2 });
    defer lhs_bf16.deinit();
    var rhs_bf16 = try array_mod.Array(array_mod.BFloat16).fromSlice(gpa, &.{
        array_mod.BFloat16.fromF32(10),
        array_mod.BFloat16.fromF32(20),
        array_mod.BFloat16.fromF32(30),
        array_mod.BFloat16.fromF32(40),
    }, &.{ 2, 2 });
    defer rhs_bf16.deinit();
    const bf16_report = selectElementwise(array_mod.BFloat16, .add, .prefer_cuda, lhs_bf16, rhs_bf16);
    try std.testing.expect(bf16_report.ok());
    var bf16_add = try elementwise(array_mod.BFloat16, .add, .prefer_cuda, lhs_bf16, rhs_bf16);
    defer bf16_add.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 11), bf16_add.data[0].toF32(), 0.125);
    try std.testing.expectApproxEqAbs(@as(f32, 44), bf16_add.data[3].toF32(), 0.125);
    var bf16_matmul = try matmul(array_mod.BFloat16, .prefer_cuda, lhs_bf16, rhs_bf16);
    defer bf16_matmul.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 70), bf16_matmul.data[0].toF32(), 0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 220), bf16_matmul.data[3].toF32(), 0.5);
}
