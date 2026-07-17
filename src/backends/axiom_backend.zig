//! Unified Axiom backend policy for Vectra.
//!
//! This small policy layer makes CPU-via-Axiom/Veyra and CUDA-via-Axiom visible
//! as data so callers can audit which route was selected before Vectra grows a
//! persistent `.cuda()` storage backend.

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

pub fn mpsDeviceReport(index: usize) MpsRuntimeAbiReport {
    return axiom.accelerator.mpsDeviceReport(index);
}

pub fn mpsDeviceAvailable(index: usize) bool {
    return axiom.accelerator.mpsDeviceAvailable(index);
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
    if (!supportedMatmul2d(T, lhs, rhs)) return error.ShapeMismatch;
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
    if (!supportedElementwiseSameShapeContiguous(T, lhs, rhs)) return error.ShapeMismatch;
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
    if (!supportedReduction2d(T, input)) return error.ShapeMismatch;
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

pub fn lowerBroadcastAddDialect(comptime T: type, input: array_mod.Array(T), bias: array_mod.Array(T), axis: DialectBroadcastAxis, backend: DialectBackend) array_mod.ArrayError!DialectBroadcastLoweringReport {
    if (!supportedBroadcastAdd(T, input, bias, axis)) return error.ShapeMismatch;
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

pub fn lowerUnaryDialect(comptime T: type, input: array_mod.Array(T), op: DialectUnaryOp, backend: DialectBackend) array_mod.ArrayError!DialectUnaryLoweringReport {
    if (!supportedUnary2d(T, input)) return error.ShapeMismatch;
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
    if (!supportedUnary2d(T, input)) return error.ShapeMismatch;
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
    if (!supportedMatmulExecution(T, lhs, rhs)) return null;
    return switch (target) {
        .cpu => executeCpuMatmul(T, lhs, rhs),
        .cuda => executeCudaMatmul(T, lhs, rhs),
        .mps => null,
    };
}

pub fn executeMatmulDefault(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    return executeMatmul(T, defaultExecutionTarget(), lhs, rhs);
}

fn executeCpuMatmul(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (T == f32) {
        const lhs32 = @as(array_mod.Array(f32), lhs);
        const rhs32 = @as(array_mod.Array(f32), rhs);
        if (lhs.shape.len == 1 and rhs.shape.len == 1) {
            if (try axiom_cpu.tryDotF32(lhs32, rhs32)) |value| return @as(array_mod.Array(T), try array_mod.Array(f32).fromSlice(lhs.allocator, &.{value}, &.{}));
        } else if (lhs.shape.len == 2 and rhs.shape.len == 1) {
            if (try axiom_cpu.tryMatvecF32(lhs32, rhs32)) |out| return @as(array_mod.Array(T), out);
        } else if (lhs.shape.len == 1 and rhs.shape.len == 2) {
            if (try axiom_cpu.tryVecmatF32(lhs32, rhs32)) |out| return @as(array_mod.Array(T), out);
        } else if (try axiom_cpu.tryMatmulF32(lhs32, rhs32)) |out| return @as(array_mod.Array(T), out);
    } else if (T == f64) {
        const lhs64 = @as(array_mod.Array(f64), lhs);
        const rhs64 = @as(array_mod.Array(f64), rhs);
        if (lhs.shape.len == 1 and rhs.shape.len == 1) {
            if (try axiom_cpu.tryDotF64(lhs64, rhs64)) |value| return @as(array_mod.Array(T), try array_mod.Array(f64).fromSlice(lhs.allocator, &.{value}, &.{}));
        } else if (lhs.shape.len == 2 and rhs.shape.len == 1) {
            if (try axiom_cpu.tryMatvecF64(lhs64, rhs64)) |out| return @as(array_mod.Array(T), out);
        } else if (lhs.shape.len == 1 and rhs.shape.len == 2) {
            if (try axiom_cpu.tryVecmatF64(lhs64, rhs64)) |out| return @as(array_mod.Array(T), out);
        } else if (try axiom_cpu.tryMatmulF64(lhs64, rhs64)) |out| return @as(array_mod.Array(T), out);
    }
    return null;
}

fn executeCudaMatmul(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (lhs.shape.len != 2 or rhs.shape.len != 2) return null;
    if (T == f32) {
        if (try axiom_cuda.tryMatmulF32(@as(array_mod.Array(f32), lhs), @as(array_mod.Array(f32), rhs))) |out| return @as(array_mod.Array(T), out);
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
    if (!supportedMatmulAddExecution(T, lhs, rhs, addend)) return null;
    return switch (target) {
        .cpu => executeCpuMatmulAdd(T, lhs, rhs, addend),
        .cuda => executeCudaMatmulAdd(T, lhs, rhs, addend),
        .mps => null,
    };
}

pub fn executeMatmulAddDefault(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T), addend: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    return executeMatmulAdd(T, defaultExecutionTarget(), lhs, rhs, addend);
}

fn executeCpuMatmulAdd(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T), addend: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (T == f32) {
        if (try axiom_cpu.tryMatmulAddF32(@as(array_mod.Array(f32), lhs), @as(array_mod.Array(f32), rhs), @as(array_mod.Array(f32), addend))) |out| return @as(array_mod.Array(T), out);
    } else if (T == f64) {
        if (try axiom_cpu.tryMatmulAddF64(@as(array_mod.Array(f64), lhs), @as(array_mod.Array(f64), rhs), @as(array_mod.Array(f64), addend))) |out| return @as(array_mod.Array(T), out);
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

pub fn executeUnary(
    comptime T: type,
    op: DialectUnaryOp,
    target: DialectBackend,
    input: array_mod.Array(T),
) array_mod.ArrayError!?array_mod.Array(T) {
    if (!supportedUnaryExecution(T, input)) return null;
    return switch (target) {
        .cpu => executeCpuUnary(T, op, input),
        .cuda => null,
        .mps => null,
    };
}

pub fn executeUnaryDefault(comptime T: type, op: DialectUnaryOp, input: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    return executeUnary(T, op, defaultExecutionTarget(), input);
}

pub fn executeReduction(
    comptime T: type,
    op: DialectReductionOp,
    target: DialectBackend,
    input: array_mod.Array(T),
    axis: u1,
    keepdims: bool,
) array_mod.ArrayError!?array_mod.Array(T) {
    if (!supportedReduction2d(T, input)) return null;
    return switch (target) {
        .cpu => executeCpuReduction(T, op, input, axis, keepdims),
        .cuda => null,
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
    return executeReduction(T, op, defaultExecutionTarget(), input, axis, keepdims);
}

pub fn executeTranspose(
    comptime T: type,
    target: DialectBackend,
    input: array_mod.Array(T),
) array_mod.ArrayError!?array_mod.Array(T) {
    if (!supportedUnary2d(T, input)) return null;
    return switch (target) {
        .cpu => executeCpuTranspose(T, input),
        .cuda => null,
        .mps => null,
    };
}

pub fn executeTransposeDefault(comptime T: type, input: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    return executeTranspose(T, defaultExecutionTarget(), input);
}

fn executeCpuUnary(comptime T: type, op: DialectUnaryOp, input: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (op != .square) return null;
    if (T == f32) {
        if (try axiom_cpu.trySquareF32(@as(array_mod.Array(f32), input))) |out| return @as(array_mod.Array(T), out);
    } else if (T == f64) {
        if (try axiom_cpu.trySquareF64(@as(array_mod.Array(f64), input))) |out| return @as(array_mod.Array(T), out);
    }
    return null;
}

fn executeCpuReduction(
    comptime T: type,
    op: DialectReductionOp,
    input: array_mod.Array(T),
    axis: u1,
    keepdims: bool,
) array_mod.ArrayError!?array_mod.Array(T) {
    if (T == f32) {
        const maybe = switch (op) {
            .sum => try axiom_cpu.trySumF32(@as(array_mod.Array(f32), input), axis, keepdims),
            .prod => try axiom_cpu.tryProdF32(@as(array_mod.Array(f32), input), axis, keepdims),
            .min => try axiom_cpu.tryMinF32(@as(array_mod.Array(f32), input), axis, keepdims),
            .max => try axiom_cpu.tryMaxF32(@as(array_mod.Array(f32), input), axis, keepdims),
        };
        if (maybe) |out| return @as(array_mod.Array(T), out);
    } else if (T == f64) {
        const maybe = switch (op) {
            .sum => try axiom_cpu.trySumF64(@as(array_mod.Array(f64), input), axis, keepdims),
            .prod => try axiom_cpu.tryProdF64(@as(array_mod.Array(f64), input), axis, keepdims),
            .min => try axiom_cpu.tryMinF64(@as(array_mod.Array(f64), input), axis, keepdims),
            .max => try axiom_cpu.tryMaxF64(@as(array_mod.Array(f64), input), axis, keepdims),
        };
        if (maybe) |out| return @as(array_mod.Array(T), out);
    }
    return null;
}

fn executeCpuTranspose(comptime T: type, input: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (T == f32) {
        if (try axiom_cpu.tryTransposeF32(@as(array_mod.Array(f32), input))) |out| return @as(array_mod.Array(T), out);
    } else if (T == f64) {
        if (try axiom_cpu.tryTransposeF64(@as(array_mod.Array(f64), input))) |out| return @as(array_mod.Array(T), out);
    }
    return null;
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
        .prefer_axiom_cpu => if (supportsAxiomCpuElementwise(T) and axiom_cpu.enabled()) .axiom_cpu_veyra else if ((T == f32 or T == f16 or T == array_mod.BFloat16) and axiom_cuda.enabled()) .axiom_cuda else .direct_cpu,
        .prefer_cuda => if ((T == f32 or T == f16 or T == array_mod.BFloat16) and axiom_cuda.enabled()) .axiom_cuda else if (supportsAxiomCpuElementwise(T) and axiom_cpu.enabled()) .axiom_cpu_veyra else .direct_cpu,
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
        .prefer_axiom_cpu => if (supportsAxiomCpuElementwise(T) and axiom_cpu.enabled()) .axiom_cpu_veyra else if ((T == f32 or T == f16 or T == array_mod.BFloat16) and axiom_cuda.enabled()) .axiom_cuda else .direct_cpu,
        .prefer_cuda => if ((T == f32 or T == f16 or T == array_mod.BFloat16) and axiom_cuda.enabled()) .axiom_cuda else if (supportsAxiomCpuElementwise(T) and axiom_cpu.enabled()) .axiom_cpu_veyra else .direct_cpu,
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
        } else if (T == array_mod.BFloat16) {
            const out = switch (op) {
                .add => try axiom_cuda.tryAddBF16(@as(array_mod.Array(array_mod.BFloat16), lhs), @as(array_mod.Array(array_mod.BFloat16), rhs)),
                .sub => try axiom_cuda.trySubBF16(@as(array_mod.Array(array_mod.BFloat16), lhs), @as(array_mod.Array(array_mod.BFloat16), rhs)),
                .mul => try axiom_cuda.tryMulBF16(@as(array_mod.Array(array_mod.BFloat16), lhs), @as(array_mod.Array(array_mod.BFloat16), rhs)),
                .div => try axiom_cuda.tryDivBF16(@as(array_mod.Array(array_mod.BFloat16), lhs), @as(array_mod.Array(array_mod.BFloat16), rhs)),
            };
            if (out) |value| return @as(array_mod.Array(T), value);
        } else if (T == f16) {
            const out = switch (op) {
                .add => try axiom_cuda.tryAddF16(@as(array_mod.Array(f16), lhs), @as(array_mod.Array(f16), rhs)),
                .sub => try axiom_cuda.trySubF16(@as(array_mod.Array(f16), lhs), @as(array_mod.Array(f16), rhs)),
                .mul => try axiom_cuda.tryMulF16(@as(array_mod.Array(f16), lhs), @as(array_mod.Array(f16), rhs)),
                .div => try axiom_cuda.tryDivF16(@as(array_mod.Array(f16), lhs), @as(array_mod.Array(f16), rhs)),
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

pub fn elementwiseScalar(
    comptime T: type,
    op: ElementwiseOp,
    policy: BackendPolicy,
    input: array_mod.Array(T),
    scalar: T,
    scalar_side: ScalarSide,
) array_mod.ArrayError!array_mod.Array(T) {
    const report = selectScalarElementwise(T, op, policy, input, scalar, scalar_side);
    switch (report.selected) {
        .axiom_cuda, .axiom_cpu_veyra => {
            var scalar_array = try array_mod.Array(T).full(input.allocator, input.shape, scalar);
            defer scalar_array.deinit();
            return switch (scalar_side) {
                .lhs => elementwise(T, op, policy, scalar_array, input),
                .rhs => elementwise(T, op, policy, input, scalar_array),
            };
        },
        .direct_cpu => {},
    }
    return directScalarElementwise(T, op, input, scalar, scalar_side);
}

pub fn tryElementwiseScalarBroadcast(comptime T: type, op: ElementwiseOp, policy: BackendPolicy, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (lhs.data.len == rhs.data.len) return null;
    if (lhs.data.len == 1 and rhs.data.len != 0 and scalarBroadcastPreservesVectorShape(lhs.shape, rhs.shape)) return try elementwiseScalar(T, op, policy, rhs, lhs.data[0], .lhs);
    if (rhs.data.len == 1 and lhs.data.len != 0 and scalarBroadcastPreservesVectorShape(rhs.shape, lhs.shape)) return try elementwiseScalar(T, op, policy, lhs, rhs.data[0], .rhs);
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
    const report = selectMatmul(T, policy, lhs, rhs);
    switch (report.selected) {
        .axiom_cuda => if (T == f32) {
            const out = try axiom_cuda.tryMatmulF32(@as(array_mod.Array(f32), lhs), @as(array_mod.Array(f32), rhs));
            if (out) |value| return @as(array_mod.Array(T), value);
        } else if (T == array_mod.BFloat16) {
            const out = try axiom_cuda.tryMatmulBF16(@as(array_mod.Array(array_mod.BFloat16), lhs), @as(array_mod.Array(array_mod.BFloat16), rhs));
            if (out) |value| return @as(array_mod.Array(T), value);
        } else if (T == f16) {
            const out = try axiom_cuda.tryMatmulF16(@as(array_mod.Array(f16), lhs), @as(array_mod.Array(f16), rhs));
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
            var acc = zeroValue(T);
            for (0..lhs.shape[1]) |kk| {
                acc = elementwiseValue(T, .add, acc, elementwiseValue(T, .mul, lhs.data[row * lhs.shape[1] + kk], rhs.data[kk * rhs.shape[1] + col]));
            }
            out.data[row * rhs.shape[1] + col] = acc;
        }
    }
    return out;
}

fn supportedMatmul2d(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) bool {
    return lhs.device.isCpu() and rhs.device.isCpu() and lhs.shape.len == 2 and rhs.shape.len == 2 and lhs.shape[1] == rhs.shape[0] and lhs.isContiguous() and rhs.isContiguous();
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
    return lhs.device.isCuda() and lhs.shape.len == 2 and rhs.shape.len == 2 and (T == f32 or T == f16 or T == array_mod.BFloat16);
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

fn supportedUnary2d(comptime T: type, input: array_mod.Array(T)) bool {
    return dialectElement(T) != null and input.device.isCpu() and input.shape.len == 2 and input.isContiguous();
}

fn supportedUnaryExecution(comptime T: type, input: array_mod.Array(T)) bool {
    return (T == f32 or T == f64) and
        input.device.isCpu() and
        input.data.len != 0 and
        input.isContiguous();
}

fn supportedBroadcastAdd(comptime T: type, input: array_mod.Array(T), bias: array_mod.Array(T), axis: DialectBroadcastAxis) bool {
    if (dialectElement(T) == null) return false;
    if (!input.device.isCpu() or !bias.device.isCpu() or input.shape.len != 2 or bias.shape.len != 1) return false;
    if (!input.isContiguous() or !bias.isContiguous()) return false;
    return bias.shape[0] == switch (axis) {
        .row => input.shape[1],
        .column => input.shape[0],
    };
}

fn supportedElementwiseSameShapeContiguous(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) bool {
    return supportsAxiomElementwise(T) and
        lhs.device.isCpu() and
        rhs.device.isCpu() and
        lhs.data.len != 0 and
        lhs.sameShape(rhs) and
        lhs.isContiguous() and
        rhs.isContiguous();
}

fn supportsAxiomElementwise(comptime T: type) bool {
    return T == f32 or T == f64 or T == f16 or T == array_mod.BFloat16;
}

fn supportsAxiomCpuElementwise(comptime T: type) bool {
    return T == f32 or T == f64;
}

fn supportsAxiomCpuMatmul(comptime T: type) bool {
    return T == f32 or T == f64;
}

fn supportsAxiomCudaMatmul(comptime T: type) bool {
    return T == f32 or T == f16 or T == array_mod.BFloat16;
}

fn supportedScalarElementwise(comptime T: type, input: array_mod.Array(T)) bool {
    return supportsAxiomElementwise(T) and
        input.device.isCpu() and
        input.data.len != 0 and
        input.isContiguous();
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

    const cuda_report = try lowerBroadcastAddDialect(f32, input, row, .row, .cuda);
    try std.testing.expect(cuda_report.ok());
    try std.testing.expectEqual(DialectBroadcastLoweringStatus.lowered_cuda, cuda_report.status);
    try std.testing.expect(cuda_report.vector_fragment_fingerprint != 0);
    try std.testing.expect(cuda_report.gpu_mapping_fingerprint != 0);

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
