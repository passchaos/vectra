//! Unified Axiom target facade for Vectra.
//!
//! Vectra should describe array work and call Axiom with a target instead of
//! open-coding CPU/CUDA/MPS branches in Array methods.  This module is the
//! intentional seam: high-level code chooses `.cpu`, `.cuda`, or `.mps`, while
//! the per-target implementation details stay concentrated here until Axiom
//! grows a fully public execution ABI for every operation.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("vectra_build_options");
const array_mod = @import("../array.zig");
const axiom = @import("axiom");
const axiom_cpu = @import("axiom_cpu.zig");
const axiom_cuda = @import("axiom_cuda.zig");
const axiom_mps = @import("axiom_mps.zig");

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
    log,
    exp2,
    expm1,
    log1p,
    log2,
    log10,
    sin,
    cos,
    tan,
    asin,
    acos,
    atan,

    pub fn label(op: ExecutionUnaryOp) []const u8 {
        return @tagName(op);
    }
};

pub const TensorMemRefDescriptor = axiom.accelerator.TensorMemRefDescriptor;
pub const TensorMemRefAddressSpace = axiom.accelerator.TensorMemRefAddressSpace;
pub const TensorGemmMemRefLoweringPlan = axiom.accelerator.TensorGemmMemRefLoweringPlan;
pub const TensorGemmMemRefLoweringStatus = axiom.accelerator.TensorGemmMemRefLoweringStatus;
pub const TensorGemmMemRefBufferizationReport = axiom.accelerator.TensorGemmMemRefBufferizationReport;
pub const TensorGemmMemRefDeviceBufferizationPlan = axiom.accelerator.TensorGemmMemRefDeviceBufferizationPlan;
pub const TensorGemmMemRefDeviceBufferizationStatus = axiom.accelerator.TensorGemmMemRefDeviceBufferizationStatus;
pub const TensorBatchedGemmMemRefLoweringPlan = axiom.accelerator.TensorBatchedGemmMemRefLoweringPlan;
pub const TensorBatchedGemmMemRefLoweringStatus = axiom.accelerator.TensorBatchedGemmMemRefLoweringStatus;

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

pub const CpuScalarElementwiseReportSnapshot = struct {
    ok: bool = false,
    operation: []const u8 = "",
    len: usize = 0,
    scalar_on_lhs: bool = false,
    report_fingerprint: u64 = 0,

    pub fn valid(report: CpuScalarElementwiseReportSnapshot) bool {
        return report.ok and report.operation.len != 0 and report.len != 0 and report.report_fingerprint != 0;
    }
};

threadlocal var last_cpu_scalar_elementwise_report: CpuScalarElementwiseReportSnapshot = .{};

pub const CpuViewElementwiseReportSnapshot = struct {
    ok: bool = false,
    operation: []const u8 = "",
    len: usize = 0,
    spec_fingerprint: u64 = 0,
    report_fingerprint: u64 = 0,

    pub fn valid(report: CpuViewElementwiseReportSnapshot) bool {
        return report.ok and report.operation.len != 0 and report.len != 0 and report.spec_fingerprint != 0 and report.report_fingerprint != 0;
    }
};

threadlocal var last_cpu_view_elementwise_report: CpuViewElementwiseReportSnapshot = .{};

pub const cpu = struct {
    pub const ScalarElementwiseReportSnapshot = CpuScalarElementwiseReportSnapshot;
    pub const ViewElementwiseReportSnapshot = CpuViewElementwiseReportSnapshot;

    pub fn enabled() bool {
        return axiom_cpu.enabled();
    }

    pub fn resetLastScalarElementwiseReport() void {
        last_cpu_scalar_elementwise_report = .{};
    }

    pub fn lastScalarElementwiseReport() ScalarElementwiseReportSnapshot {
        return last_cpu_scalar_elementwise_report;
    }

    pub fn resetLastViewElementwiseReport() void {
        last_cpu_view_elementwise_report = .{};
    }

    pub fn lastViewElementwiseReport() ViewElementwiseReportSnapshot {
        return last_cpu_view_elementwise_report;
    }
};

pub const cuda = struct {
    pub const Status = axiom_cuda.Status;
    pub const SmokeReport = axiom_cuda.SmokeReport;
    pub const DeviceArrayF32 = axiom_cuda.DeviceArrayF32;
    pub const CudaDeviceMemRefReportSnapshot = axiom_cuda.CudaDeviceMemRefReportSnapshot;
    pub const CudaDeviceGemmReportSnapshot = axiom_cuda.CudaDeviceGemmReportSnapshot;
    pub const CudaDeviceBatchedGemmReportSnapshot = axiom_cuda.CudaDeviceBatchedGemmReportSnapshot;
    pub const CudaDTypeBridgeStatus = axiom_cuda.CudaDTypeBridgeStatus;
    pub const CudaDTypeSupportRecord = axiom_cuda.CudaDTypeSupportRecord;

    pub fn enabled() bool {
        return axiom_cuda.enabled();
    }

    pub fn runSmoke(allocator: std.mem.Allocator) SmokeReport {
        return axiom_cuda.runSmoke(allocator);
    }

    pub fn cudaDTypeSupportRecords() []const CudaDTypeSupportRecord {
        return axiom_cuda.cudaDTypeSupportRecords();
    }

    pub fn findCudaDTypeSupport(cuda_name: []const u8) ?CudaDTypeSupportRecord {
        return axiom_cuda.findCudaDTypeSupport(cuda_name);
    }

    pub fn findVectraDTypeSupport(dtype: array_mod.DType) ?CudaDTypeSupportRecord {
        return axiom_cuda.findVectraDTypeSupport(dtype);
    }

    pub fn cudaDTypeNativeSeedCount() usize {
        return axiom_cuda.cudaDTypeNativeSeedCount();
    }

    pub fn cudaDTypeWidenedSeedCount() usize {
        return axiom_cuda.cudaDTypeWidenedSeedCount();
    }

    pub fn cudaDTypeBridgeCount() usize {
        return axiom_cuda.cudaDTypeBridgeCount();
    }

    pub fn cudaDTypeSupportFingerprint() u64 {
        return axiom_cuda.cudaDTypeSupportFingerprint();
    }

    pub fn toDeviceF32(allocator: std.mem.Allocator, host: array_mod.Array(f32)) array_mod.ArrayError!?DeviceArrayF32 {
        return axiom_cuda.toDeviceF32(allocator, host);
    }

    pub fn synchronizeDevice(allocator: std.mem.Allocator, device: array_mod.Device) array_mod.ArrayError!void {
        return axiom_cuda.synchronizeDevice(allocator, device);
    }

    pub fn resetLastCudaDeviceMemRefReport() void {
        axiom_cuda.resetLastCudaDeviceMemRefReport();
    }

    pub fn lastCudaDeviceMemRefReport() CudaDeviceMemRefReportSnapshot {
        return axiom_cuda.lastCudaDeviceMemRefReport();
    }

    pub fn resetLastCudaDeviceGemmReport() void {
        axiom_cuda.resetLastCudaDeviceGemmReport();
    }

    pub fn lastCudaDeviceGemmReport() CudaDeviceGemmReportSnapshot {
        return axiom_cuda.lastCudaDeviceGemmReport();
    }

    pub fn resetLastCudaDeviceBatchedGemmReport() void {
        axiom_cuda.resetLastCudaDeviceBatchedGemmReport();
    }

    pub fn lastCudaDeviceBatchedGemmReport() CudaDeviceBatchedGemmReportSnapshot {
        return axiom_cuda.lastCudaDeviceBatchedGemmReport();
    }
};

pub const RuntimeCapabilityStatus = enum(u8) {
    planned,
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
        .mps => axiom_mps.deviceAvailable(device.index),
    };
}

pub fn allocateStorage(device: array_mod.Device, len: usize, element_size: usize) array_mod.ArrayError!?array_mod.DeviceStorage {
    return switch (executionTargetForDevice(device)) {
        .cpu => null,
        .cuda => axiom_cuda.allocateStorage(device, len, element_size),
        .mps => axiom_mps.allocateStorage(device, len, element_size),
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
        .mps => axiom_mps.freeStorage(storage),
    }
}

pub fn fillStorage(comptime T: type, storage: array_mod.DeviceStorage, value: T) array_mod.ArrayError!void {
    return switch (executionTargetForDevice(storage.device)) {
        .cpu => error.InvalidDevice,
        .cuda => axiom_cuda.fillStorage(T, storage, value),
        .mps => axiom_mps.fillStorage(T, storage, value),
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
        .mps => axiom_mps.uploadStorage(storage, bytes),
    };
}

pub fn downloadStorage(storage: array_mod.DeviceStorage, bytes: []u8) array_mod.ArrayError!void {
    return switch (executionTargetForDevice(storage.device)) {
        .cpu => error.InvalidDevice,
        .cuda => axiom_cuda.downloadStorage(storage, bytes),
        .mps => axiom_mps.downloadStorage(storage, bytes),
    };
}

pub fn copyStorage(dst: array_mod.DeviceStorage, src: array_mod.DeviceStorage) array_mod.ArrayError!void {
    if (!dst.device.sameDevice(src.device)) return error.InvalidDevice;
    return switch (executionTargetForDevice(dst.device)) {
        .cpu => error.InvalidDevice,
        .cuda => axiom_cuda.copyStorage(dst, src),
        .mps => axiom_mps.copyStorage(dst, src),
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
/// target.  MPS storage is backed by Axiom-owned Metal shared buffers; eager
/// kernel execution remains capability-gated until MPSGraph/Metal kernels land.
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
            .mps => {
                const dst_storage = dst.storage orelse return error.InvalidDevice;
                try uploadStorage(dst_storage, src.host_bytes);
            },
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
        .mps => switch (executionTargetForDevice(dst.device)) {
            .cpu => {
                const src_storage = src.storage orelse return error.InvalidDevice;
                try downloadStorage(src_storage, dst.host_bytes);
            },
            .cuda => error.InvalidDevice,
            .mps => {
                const src_storage = src.storage orelse return error.InvalidDevice;
                const dst_storage = dst.storage orelse return error.InvalidDevice;
                try copyStorage(dst_storage, src_storage);
            },
        },
    };
}

fn platformDefaultDialectBackend() DialectBackend {
    return if (builtin.os.tag == .macos) .mps else .cpu;
}

threadlocal var default_dialect_backend: DialectBackend = platformDefaultDialectBackend();

pub fn setDefaultDialectBackend(backend: DialectBackend) void {
    default_dialect_backend = backend;
}

pub fn defaultDialectBackend() DialectBackend {
    return default_dialect_backend;
}

pub fn resetDefaultDialectBackend() void {
    default_dialect_backend = platformDefaultDialectBackend();
}

pub fn tensorElementType(comptime T: type) ?axiom.accelerator.TensorElementType {
    return if (T == f32)
        .f32
    else if (T == f64)
        .f64
    else if (T == f16)
        .f16
    else if (T == array_mod.BFloat16)
        .bf16
    else if (T == i8)
        .i8
    else if (T == u8)
        .u8
    else if (T == i16)
        .i16
    else if (T == u16)
        .u16
    else if (T == i32)
        .i32
    else if (T == u32)
        .u32
    else if (T == i64)
        .i64
    else if (T == u64)
        .u64
    else
        null;
}

pub fn memRefAddressSpaceForDevice(device: array_mod.Device) TensorMemRefAddressSpace {
    return switch (device.backend) {
        .cpu => .host,
        .cuda => .cuda,
        .mps => .mps,
    };
}

pub fn describeArrayMemRef(comptime T: type, input: array_mod.Array(T), name: []const u8) array_mod.ArrayError!TensorMemRefDescriptor {
    const element = tensorElementType(T) orelse return error.TypeUnsupported;
    const strides = try usizeStridesToIsize(input.strides);
    const base_ptr: u64 = if (input.device_storage) |storage| storage.ptr else @intCast(@intFromPtr(input.data.ptr));
    return axiom.accelerator.TensorMemRefDescriptor.init(
        name,
        base_ptr,
        element,
        memRefAddressSpaceForDevice(input.device),
        0,
        input.shape,
        strides[0..input.strides.len],
    ) catch mapTensorViewError();
}

pub fn describeViewMemRef(comptime T: type, input: array_mod.ArrayView(T), name: []const u8) array_mod.ArrayError!TensorMemRefDescriptor {
    const element = tensorElementType(T) orelse return error.TypeUnsupported;
    const strides = try usizeStridesToIsize(input.strides);
    return axiom.accelerator.TensorMemRefDescriptor.init(
        name,
        @intCast(@intFromPtr(input.data.ptr)),
        element,
        memRefAddressSpaceForDevice(input.device),
        input.offset,
        input.shape,
        strides[0..input.strides.len],
    ) catch mapTensorViewError();
}

pub fn planGemmMemRefLowering(
    comptime T: type,
    lhs: array_mod.ArrayView(T),
    rhs: array_mod.ArrayView(T),
    out: array_mod.ArrayView(T),
) array_mod.ArrayError!TensorGemmMemRefLoweringPlan {
    const lhs_desc = try describeViewMemRef(T, lhs, "lhs");
    const rhs_desc = try describeViewMemRef(T, rhs, "rhs");
    const out_desc = try describeViewMemRef(T, out, "out");
    const plan = TensorGemmMemRefLoweringPlan.fromMemRefs(lhs_desc, rhs_desc, out_desc);
    if (plan.status == .invalid) return error.InvalidShape;
    return plan;
}

pub fn computeGemmMemRefBufferizedReference(
    comptime T: type,
    allocator: std.mem.Allocator,
    lhs: array_mod.ArrayView(T),
    rhs: array_mod.ArrayView(T),
    out: array_mod.ArrayView(T),
) array_mod.ArrayError!TensorGemmMemRefBufferizationReport {
    if (T != f32) return error.TypeUnsupported;
    const lhs_desc = try describeViewMemRef(T, lhs, "lhs");
    const rhs_desc = try describeViewMemRef(T, rhs, "rhs");
    const out_desc = try describeViewMemRef(T, out, "out");
    var spec = axiom.accelerator.TensorGemmSpec.fromMemRefs(lhs_desc, rhs_desc, out_desc) catch return error.InvalidShape;
    spec.alpha = 1.0;
    spec.beta = 0.0;
    return axiom.accelerator.computeGemmMemRefBufferizedCpuReference(
        allocator,
        spec,
        @as([]const f32, lhs.data),
        @as([]const f32, rhs.data),
        @as([]const f32, out.data),
        @as([]f32, out.data),
    ) catch error.BackendFailure;
}

pub fn planGemmMemRefDeviceBufferization(
    comptime T: type,
    lhs: array_mod.ArrayView(T),
    rhs: array_mod.ArrayView(T),
    out: array_mod.ArrayView(T),
) array_mod.ArrayError!TensorGemmMemRefDeviceBufferizationPlan {
    const lhs_desc = try describeViewMemRef(T, lhs, "lhs");
    const rhs_desc = try describeViewMemRef(T, rhs, "rhs");
    const out_desc = try describeViewMemRef(T, out, "out");
    const spec = axiom.accelerator.TensorGemmSpec.fromMemRefs(lhs_desc, rhs_desc, out_desc) catch return error.InvalidShape;
    return axiom.accelerator.TensorGemmMemRefDeviceBufferizationPlan.fromSpec(spec) catch error.BackendFailure;
}

pub fn planBatchedGemmMemRefLowering(
    comptime T: type,
    lhs: array_mod.ArrayView(T),
    rhs: array_mod.ArrayView(T),
    out: array_mod.ArrayView(T),
) array_mod.ArrayError!TensorBatchedGemmMemRefLoweringPlan {
    const lhs_desc = try describeViewMemRef(T, lhs, "lhs_batch");
    const rhs_desc = try describeViewMemRef(T, rhs, "rhs_batch");
    const out_desc = try describeViewMemRef(T, out, "out_batch");
    return axiom.accelerator.TensorBatchedGemmMemRefLoweringPlan.fromMemRefs(lhs_desc, rhs_desc, out_desc) catch error.InvalidShape;
}

fn usizeStridesToIsize(strides: []const usize) array_mod.ArrayError![4]isize {
    if (strides.len > 4) return error.InvalidShape;
    var out: [4]isize = .{ 1, 1, 1, 1 };
    for (strides, 0..) |stride, index| {
        out[index] = std.math.cast(isize, stride) orelse return error.InvalidShape;
    }
    return out;
}

fn mapTensorViewError() array_mod.ArrayError {
    return error.InvalidShape;
}

pub fn defaultBackendPolicy() BackendPolicy {
    return switch (defaultDialectBackend()) {
        .cpu => .prefer_axiom_cpu,
        .cuda => .prefer_cuda,
        // CPU arrays cannot execute on MPS without an explicit device transfer.
        // Keep default eager CPU execution on CPU; MPS-resident arrays use the
        // MPS target through defaultTargetForDevice().
        .mps => .prefer_axiom_cpu,
    };
}

pub fn defaultExecutionTarget() DialectBackend {
    return switch (defaultDialectBackend()) {
        .cpu => .cpu,
        .cuda => .cuda,
        // MPS remains a valid dialect-lowering target, while eager CPU arrays
        // keep a real CPU runtime path unless they are explicitly moved to MPS.
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
            .reason = "Axiom CUDA exposes eager f32/f64/f16/BFloat16 2D sum/prod/min/max reduction runtimes; other reduction dtypes remain capability-gated.",
        },
        .mps => .{
            .target = target,
            .operation = "reduction",
            .status = .executable,
            .reason = "Axiom MPS exposes eager f32/f16/BFloat16 2D sum/prod/min/max reductions over Metal shared-buffer storage; other dtypes/shapes remain capability-gated.",
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
            .reason = "Axiom CUDA exposes eager f32/f64/f16/BFloat16 2D row/column broadcast add/sub/mul/div runtimes; other broadcast dtypes/shapes remain capability-gated.",
        },
        .mps => .{
            .target = target,
            .operation = "broadcast_add",
            .status = .executable,
            .reason = "Axiom MPS exposes eager f32/f16/BFloat16 2D row/column broadcast add/sub/mul/div over Metal shared-buffer storage; other dtypes/shapes remain capability-gated.",
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
            .reason = "Axiom CUDA exposes eager f32/f64/f16/BFloat16 2D transpose runtimes; other transpose dtypes/shapes remain capability-gated.",
        },
        .mps => .{
            .target = target,
            .operation = "transpose2d",
            .status = .executable,
            .reason = "Axiom MPS exposes eager f32/f16/BFloat16 2D transpose over Metal shared-buffer storage; other dtypes/shapes remain capability-gated.",
        },
    };
}

pub fn logSoftmaxRuntimeCapability(target: DialectBackend) RuntimeCapabilityReport {
    return switch (target) {
        .cpu => .{
            .target = target,
            .operation = "log_softmax2d",
            .status = .lowering_only,
            .reason = "Vectra composes CPU logSoftmax from logsumexp/sub today; no dedicated Axiom CPU logSoftmax runtime ABI is exposed yet.",
        },
        .cuda => .{
            .target = target,
            .operation = "log_softmax2d",
            .status = .executable,
            .reason = "Axiom CUDA exposes eager f32/f64/f16/BFloat16 2D axis logSoftmax runtimes; other logSoftmax dtypes/shapes remain capability-gated.",
        },
        .mps => .{
            .target = target,
            .operation = "log_softmax2d",
            .status = .executable,
            .reason = "Axiom MPS exposes eager f32/f16/BFloat16 2D axis logSoftmax over Metal shared-buffer storage; other dtypes/shapes remain capability-gated.",
        },
    };
}

pub fn softmaxRuntimeCapability(target: DialectBackend) RuntimeCapabilityReport {
    return switch (target) {
        .cpu => .{
            .target = target,
            .operation = "softmax2d",
            .status = .lowering_only,
            .reason = "Vectra composes CPU softmax from max/sub/exp/sum/div today; no dedicated Axiom CPU softmax runtime ABI is exposed yet.",
        },
        .cuda => .{
            .target = target,
            .operation = "softmax2d",
            .status = .executable,
            .reason = "Axiom CUDA exposes eager f32/f64/f16/BFloat16 2D axis softmax runtimes; other softmax dtypes/shapes remain capability-gated.",
        },
        .mps => .{
            .target = target,
            .operation = "softmax2d",
            .status = .executable,
            .reason = "Axiom MPS exposes eager f32/f16/BFloat16 2D axis softmax over Metal shared-buffer storage; other dtypes/shapes remain capability-gated.",
        },
    };
}

pub fn unaryRuntimeCapability(target: DialectBackend, op: DialectUnaryOp) RuntimeCapabilityReport {
    return switch (target) {
        .cpu => .{
            .target = target,
            .operation = dialectUnaryRuntimeOperation(op),
            .status = if (dialectUnaryRuntimeExecutable(.cpu, op)) .executable else .lowering_only,
            .reason = if (dialectUnaryRuntimeExecutable(.cpu, op))
                "Axiom CPU unary runtime is routed through Veyra unary elementwise execution for f32/f64 arrays."
            else
                "Axiom CPU unary dialect lowering exists for this op, but Vectra has no dedicated eager Axiom runtime ABI for it yet.",
        },
        .cuda => .{
            .target = target,
            .operation = dialectUnaryRuntimeOperation(op),
            .status = if (dialectUnaryRuntimeExecutable(.cuda, op)) .executable else .lowering_only,
            .reason = if (dialectUnaryRuntimeExecutable(.cuda, op))
                "Axiom CUDA unary runtime is available for supported dtypes through typed device unary or elementwise routes; log is currently executable for f32."
            else
                "Axiom CUDA unary dialect lowering exists for this op, but Vectra has no dedicated eager CUDA runtime ABI for it yet.",
        },
        .mps => .{
            .target = target,
            .operation = dialectUnaryRuntimeOperation(op),
            .status = if (dialectUnaryRuntimeExecutable(.mps, op)) .executable else .planned,
            .reason = if (dialectUnaryRuntimeExecutable(.mps, op))
                "Axiom MPS unary runtime is available for supported f32 contiguous arrays through Metal kernels."
            else
                "This MPS unary op is not in the current executable Metal kernel slice.",
        },
    };
}

fn plannedMpsRuntimeCapability(operation: []const u8) RuntimeCapabilityReport {
    return .{
        .target = .mps,
        .operation = operation,
        .status = .planned,
        .reason = "This MPS operation is not in the current executable Metal kernel slice; remaining MPS dtype/shape coverage stays capability-gated.",
    };
}

fn dialectUnaryRuntimeExecutable(target: DialectBackend, op: DialectUnaryOp) bool {
    return switch (target) {
        .cpu => switch (op) {
            .abs, .square, .sqrt, .exp, .log => true,
            .copy, .cube => false,
        },
        .cuda => switch (op) {
            .abs, .square, .sqrt, .exp, .log => true,
            .copy, .cube => false,
        },
        .mps => switch (op) {
            .abs, .square, .sqrt, .exp, .log => true,
            .copy, .cube => false,
        },
    };
}

fn dialectUnaryRuntimeOperation(op: DialectUnaryOp) []const u8 {
    return switch (op) {
        .copy => "unary.copy",
        .square => "unary.square",
        .cube => "unary.cube",
        .abs => "unary.abs",
        .sqrt => "unary.sqrt",
        .exp => "unary.exp",
        .log => "unary.log",
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
    return executeBroadcastBinary(T, .add, target, input, bias, axis);
}

pub fn executeBroadcastAddDefault(comptime T: type, input: array_mod.Array(T), bias: array_mod.Array(T), axis: DialectBroadcastAxis) array_mod.ArrayError!?array_mod.Array(T) {
    return executeBroadcastAdd(T, defaultTargetForDevice(input.device), input, bias, axis);
}

pub fn executeBroadcastBinary(
    comptime T: type,
    op: ElementwiseOp,
    target: DialectBackend,
    input: array_mod.Array(T),
    bias: array_mod.Array(T),
    axis: DialectBroadcastAxis,
) array_mod.ArrayError!?array_mod.Array(T) {
    if (!supportedBroadcastBinaryExecution(T, op, target, input, bias, axis)) return null;
    return switch (target) {
        // Veyra currently exposes only row/column broadcast add.  Other CPU
        // broadcast ops intentionally fall back to Array's generic CPU path
        // rather than pretending they went through the Axiom runtime.
        .cpu => if (op == .add) executeCpuBroadcastAdd(T, input, bias, axis) else null,
        .cuda => executeCudaBroadcastBinary(T, op, input, bias, axis),
        .mps => executeMpsBroadcastBinary(T, op, input, bias, axis),
    };
}

pub fn executeBroadcastBinaryDefault(comptime T: type, op: ElementwiseOp, input: array_mod.Array(T), bias: array_mod.Array(T), axis: DialectBroadcastAxis) array_mod.ArrayError!?array_mod.Array(T) {
    return executeBroadcastBinary(T, op, defaultTargetForDevice(input.device), input, bias, axis);
}

pub fn tryBroadcastAdd(
    comptime T: type,
    target: DialectBackend,
    lhs: array_mod.Array(T),
    rhs: array_mod.Array(T),
) array_mod.ArrayError!?array_mod.Array(T) {
    return tryBroadcastBinary(T, .add, target, lhs, rhs);
}

pub fn tryBroadcastBinary(
    comptime T: type,
    op: ElementwiseOp,
    target: DialectBackend,
    lhs: array_mod.Array(T),
    rhs: array_mod.Array(T),
) array_mod.ArrayError!?array_mod.Array(T) {
    if (!lhs.device.sameDevice(rhs.device)) return error.InvalidDevice;
    if (try tryMpsRank3Broadcast(T, op, target, lhs, rhs)) |out| return out;
    if (try tryMpsLastDimBroadcast(T, op, target, lhs, rhs)) |out| return out;
    if (try tryCudaLastDimBroadcast(T, op, target, lhs, rhs)) |out| return out;
    if (lhs.shape.len == 2) {
        if (broadcastBiasMatchesArrayAdd(T, lhs, rhs, .row)) return executeBroadcastBinary(T, op, target, lhs, rhs, .row);
        if (broadcastBiasMatchesArrayAdd(T, lhs, rhs, .column)) return executeBroadcastBinary(T, op, target, lhs, rhs, .column);
    }
    if ((op == .add or op == .mul) and rhs.shape.len == 2) {
        // Only commute operations whose semantics survive swapping matrix and
        // bias operands.  Sub/div need a distinct reversed-broadcast lowering,
        // so they stay capability-gated instead of silently changing meaning.
        if (broadcastBiasMatchesArrayAdd(T, rhs, lhs, .row)) return executeBroadcastBinary(T, op, target, rhs, lhs, .row);
        if (broadcastBiasMatchesArrayAdd(T, rhs, lhs, .column)) return executeBroadcastBinary(T, op, target, rhs, lhs, .column);
    }
    if (try tryCudaGenericBroadcast(T, op, target, lhs, rhs)) |out| return out;
    return null;
}

pub fn tryBroadcastAddDefault(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    return tryBroadcastAdd(T, defaultTargetForDevice(lhs.device), lhs, rhs);
}

pub fn tryBroadcastBinaryDefault(comptime T: type, op: ElementwiseOp, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    return tryBroadcastBinary(T, op, defaultTargetForDevice(lhs.device), lhs, rhs);
}

fn tryCudaGenericBroadcast(comptime T: type, op: ElementwiseOp, target: DialectBackend, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (target != .cuda or !lhs.device.sameDevice(rhs.device) or !lhs.device.isCuda()) return null;
    if (lhs.numel() == 1 or rhs.numel() == 1) return null;
    if (T == f32) {
        if (try axiom_cuda.tryDeviceBroadcastF32(cudaBinaryOp(op), @as(array_mod.Array(f32), lhs), @as(array_mod.Array(f32), rhs))) |out| return @as(array_mod.Array(T), out);
    } else if (T == f64) {
        if (try axiom_cuda.tryDeviceBroadcastF64(cudaBinaryOp(op), @as(array_mod.Array(f64), lhs), @as(array_mod.Array(f64), rhs))) |out| return @as(array_mod.Array(T), out);
    } else if (T == f16) {
        if (try axiom_cuda.tryDeviceBroadcastF16(cudaBinaryOp(op), @as(array_mod.Array(f16), lhs), @as(array_mod.Array(f16), rhs))) |out| return @as(array_mod.Array(T), out);
    } else if (T == array_mod.BFloat16) {
        if (try axiom_cuda.tryDeviceBroadcastBF16(cudaBinaryOp(op), @as(array_mod.Array(array_mod.BFloat16), lhs), @as(array_mod.Array(array_mod.BFloat16), rhs))) |out| return @as(array_mod.Array(T), out);
    }
    return null;
}

fn tryMpsRank3Broadcast(comptime T: type, op: ElementwiseOp, target: DialectBackend, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (target != .mps or !lhs.device.sameDevice(rhs.device) or !lhs.device.isMps()) return null;
    if (T == f32) {
        if (try axiom_mps.tryRank3BroadcastBinaryF32(mpsBinaryOp(op), @as(array_mod.Array(f32), lhs), @as(array_mod.Array(f32), rhs))) |out| return @as(array_mod.Array(T), out);
    } else if (T == f16) {
        if (try axiom_mps.tryRank3BroadcastBinaryF16(mpsBinaryOp(op), @as(array_mod.Array(f16), lhs), @as(array_mod.Array(f16), rhs))) |out| return @as(array_mod.Array(T), out);
    } else if (T == array_mod.BFloat16) {
        if (try axiom_mps.tryRank3BroadcastBinaryBF16(mpsBinaryOp(op), @as(array_mod.Array(array_mod.BFloat16), lhs), @as(array_mod.Array(array_mod.BFloat16), rhs))) |out| return @as(array_mod.Array(T), out);
    }
    return null;
}

fn tryMpsLastDimBroadcast(comptime T: type, op: ElementwiseOp, target: DialectBackend, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (target != .mps or !lhs.device.sameDevice(rhs.device) or !lhs.device.isMps()) return null;
    if (T != f32 and T != f16 and T != array_mod.BFloat16) return null;
    if (try tryMpsLastDimBroadcastOrdered(T, op, lhs, rhs, false)) |out| return out;
    return tryMpsLastDimBroadcastOrdered(T, op, rhs, lhs, true);
}

fn tryMpsLastDimBroadcastOrdered(
    comptime T: type,
    op: ElementwiseOp,
    input: array_mod.Array(T),
    bias: array_mod.Array(T),
    reversed: bool,
) array_mod.ArrayError!?array_mod.Array(T) {
    if (input.shape.len <= 2 or !input.isContiguous() or !bias.isContiguous()) return null;
    const last_dim = input.shape[input.shape.len - 1];
    if (last_dim == 0 or input.numel() == 0) return null;
    if (!lastDimBiasMatches(input.shape, bias.shape, last_dim)) return null;

    var matrix = try input.reshape(&.{ input.numel() / last_dim, last_dim });
    defer matrix.deinit();
    var row_bias = try bias.reshape(&.{last_dim});
    defer row_bias.deinit();
    var out_2d = if (reversed) reversed_blk: {
        switch (op) {
            .add, .mul => break :reversed_blk (try executeBroadcastBinary(T, op, .mps, matrix, row_bias, .row)) orelse return null,
            .sub => {
                var diff = (try executeBroadcastBinary(T, .sub, .mps, matrix, row_bias, .row)) orelse return null;
                defer diff.deinit();
                break :reversed_blk (try executeElementwiseScalar(T, .mul, .mps, diff, scalarValue(T, -1), .rhs)) orelse return null;
            },
            .div => {
                var recip = (try executeElementwiseScalar(T, .div, .mps, matrix, scalarValue(T, 1), .lhs)) orelse return null;
                defer recip.deinit();
                break :reversed_blk (try executeBroadcastBinary(T, .mul, .mps, recip, row_bias, .row)) orelse return null;
            },
        }
    } else (try executeBroadcastBinary(T, op, .mps, matrix, row_bias, .row)) orelse return null;
    errdefer out_2d.deinit();
    const reshaped = try out_2d.reshape(input.shape);
    out_2d.deinit();
    return reshaped;
}

fn lastDimBiasMatches(input_shape: []const usize, bias_shape: []const usize, last_dim: usize) bool {
    if (bias_shape.len == 1) return bias_shape[0] == last_dim;
    if (bias_shape.len == 0 or bias_shape.len > input_shape.len) return false;
    const offset = input_shape.len - bias_shape.len;
    for (bias_shape, 0..) |extent, i| {
        const input_axis = offset + i;
        const expected = if (input_axis + 1 == input_shape.len) last_dim else 1;
        if (extent != expected) return false;
    }
    return true;
}

fn scalarValue(comptime T: type, comptime value: comptime_int) T {
    if (T == array_mod.BFloat16) return array_mod.BFloat16.fromF32(@floatFromInt(value));
    return value;
}

fn tryCudaLastDimBroadcast(comptime T: type, op: ElementwiseOp, target: DialectBackend, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (target != .cuda or !lhs.device.sameDevice(rhs.device) or !lhs.device.isCuda()) return null;
    const cuda_op = cudaBinaryOp(op);
    if (T == f32) {
        if (try axiom_cuda.tryDeviceLastDimBroadcastF32(cuda_op, @as(array_mod.Array(f32), lhs), @as(array_mod.Array(f32), rhs), false)) |out| return @as(array_mod.Array(T), out);
        if (try axiom_cuda.tryDeviceLastDimBroadcastF32(cuda_op, @as(array_mod.Array(f32), rhs), @as(array_mod.Array(f32), lhs), true)) |out| return @as(array_mod.Array(T), out);
    } else if (T == f64) {
        if (try axiom_cuda.tryDeviceLastDimBroadcastF64(cuda_op, @as(array_mod.Array(f64), lhs), @as(array_mod.Array(f64), rhs), false)) |out| return @as(array_mod.Array(T), out);
        if (try axiom_cuda.tryDeviceLastDimBroadcastF64(cuda_op, @as(array_mod.Array(f64), rhs), @as(array_mod.Array(f64), lhs), true)) |out| return @as(array_mod.Array(T), out);
    } else if (T == f16) {
        if (try axiom_cuda.tryDeviceLastDimBroadcastF16(cuda_op, @as(array_mod.Array(f16), lhs), @as(array_mod.Array(f16), rhs), false)) |out| return @as(array_mod.Array(T), out);
        if (try axiom_cuda.tryDeviceLastDimBroadcastF16(cuda_op, @as(array_mod.Array(f16), rhs), @as(array_mod.Array(f16), lhs), true)) |out| return @as(array_mod.Array(T), out);
    } else if (T == array_mod.BFloat16) {
        if (try axiom_cuda.tryDeviceLastDimBroadcastBF16(cuda_op, @as(array_mod.Array(array_mod.BFloat16), lhs), @as(array_mod.Array(array_mod.BFloat16), rhs), false)) |out| return @as(array_mod.Array(T), out);
        if (try axiom_cuda.tryDeviceLastDimBroadcastBF16(cuda_op, @as(array_mod.Array(array_mod.BFloat16), rhs), @as(array_mod.Array(array_mod.BFloat16), lhs), true)) |out| return @as(array_mod.Array(T), out);
    }
    return null;
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
        .mps => executeMpsMatmul(T, lhs, rhs),
    };
}

pub fn executeMatmulDefault(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    return executeMatmul(T, defaultTargetForDevice(lhs.device), lhs, rhs);
}

pub fn executeBmm(
    comptime T: type,
    target: DialectBackend,
    lhs: array_mod.Array(T),
    rhs: array_mod.Array(T),
) array_mod.ArrayError!?array_mod.Array(T) {
    if (!targetCanAccessDevice(target, lhs.device)) return null;
    if (!supportedBmmExecution(T, lhs, rhs)) return null;
    return switch (target) {
        .cpu => null,
        .cuda => executeCudaBmm(T, lhs, rhs),
        .mps => executeMpsBmm(T, lhs, rhs),
    };
}

pub fn executeBmmDefault(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    return executeBmm(T, defaultTargetForDevice(lhs.device), lhs, rhs);
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
    if (T == f32) {
        const lhs32 = @as(array_mod.Array(f32), lhs);
        const rhs32 = @as(array_mod.Array(f32), rhs);
        if (lhs.shape.len == 1 and rhs.shape.len == 1) {
            if (try axiom_cuda.tryDeviceDotF32(lhs32, rhs32)) |out| return @as(array_mod.Array(T), out);
        } else if (lhs.shape.len >= 2 and rhs.shape.len == 1) {
            if (try axiom_cuda.tryDeviceMatvecF32(lhs32, rhs32)) |out| return @as(array_mod.Array(T), out);
        } else if (lhs.shape.len == 1 and rhs.shape.len >= 2) {
            if (try axiom_cuda.tryDeviceVecmatF32(lhs32, rhs32)) |out| return @as(array_mod.Array(T), out);
        } else if (lhs.shape.len == 2 and rhs.shape.len == 2) {
            if (try axiom_cuda.tryMatmulF32(lhs32, rhs32)) |out| return @as(array_mod.Array(T), out);
        }
    } else if (T == f64) {
        const lhs64 = @as(array_mod.Array(f64), lhs);
        const rhs64 = @as(array_mod.Array(f64), rhs);
        if (lhs.shape.len == 1 and rhs.shape.len == 1) {
            if (try axiom_cuda.tryDeviceDotF64(lhs64, rhs64)) |out| return @as(array_mod.Array(T), out);
        } else if (lhs.shape.len >= 2 and rhs.shape.len == 1) {
            if (try axiom_cuda.tryDeviceMatvecF64(lhs64, rhs64)) |out| return @as(array_mod.Array(T), out);
        } else if (lhs.shape.len == 1 and rhs.shape.len >= 2) {
            if (try axiom_cuda.tryDeviceVecmatF64(lhs64, rhs64)) |out| return @as(array_mod.Array(T), out);
        } else if (lhs.shape.len == 2 and rhs.shape.len == 2) {
            if (try axiom_cuda.tryDeviceMatmulF64(lhs64, rhs64)) |out| return @as(array_mod.Array(T), out);
        }
    } else if (T == f16) {
        const lhs16 = @as(array_mod.Array(f16), lhs);
        const rhs16 = @as(array_mod.Array(f16), rhs);
        if (lhs.shape.len == 1 and rhs.shape.len == 1) {
            if (try axiom_cuda.tryDeviceDotF16(lhs16, rhs16)) |out| return @as(array_mod.Array(T), out);
        } else if (lhs.shape.len >= 2 and rhs.shape.len == 1) {
            if (try axiom_cuda.tryDeviceMatvecF16(lhs16, rhs16)) |out| return @as(array_mod.Array(T), out);
        } else if (lhs.shape.len == 1 and rhs.shape.len >= 2) {
            if (try axiom_cuda.tryDeviceVecmatF16(lhs16, rhs16)) |out| return @as(array_mod.Array(T), out);
        } else if (lhs.shape.len == 2 and rhs.shape.len == 2) {
            if (try axiom_cuda.tryMatmulF16(lhs16, rhs16)) |out| return @as(array_mod.Array(T), out);
        }
    } else if (T == array_mod.BFloat16) {
        const lhs_bf16 = @as(array_mod.Array(array_mod.BFloat16), lhs);
        const rhs_bf16 = @as(array_mod.Array(array_mod.BFloat16), rhs);
        if (lhs.shape.len == 1 and rhs.shape.len == 1) {
            if (try axiom_cuda.tryDeviceDotBF16(lhs_bf16, rhs_bf16)) |out| return @as(array_mod.Array(T), out);
        } else if (lhs.shape.len >= 2 and rhs.shape.len == 1) {
            if (try axiom_cuda.tryDeviceMatvecBF16(lhs_bf16, rhs_bf16)) |out| return @as(array_mod.Array(T), out);
        } else if (lhs.shape.len == 1 and rhs.shape.len >= 2) {
            if (try axiom_cuda.tryDeviceVecmatBF16(lhs_bf16, rhs_bf16)) |out| return @as(array_mod.Array(T), out);
        } else if (lhs.shape.len == 2 and rhs.shape.len == 2) {
            if (try axiom_cuda.tryMatmulBF16(lhs_bf16, rhs_bf16)) |out| return @as(array_mod.Array(T), out);
        }
    }
    return null;
}

fn executeMpsMatmul(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (T == f32 and lhs.shape.len == 2 and rhs.shape.len == 2) {
        if (try axiom_mps.tryMatmulF32(@as(array_mod.Array(f32), lhs), @as(array_mod.Array(f32), rhs))) |out| return @as(array_mod.Array(T), out);
    } else if (T == f16 and lhs.shape.len == 2 and rhs.shape.len == 2) {
        if (try axiom_mps.tryMatmulF16(@as(array_mod.Array(f16), lhs), @as(array_mod.Array(f16), rhs))) |out| return @as(array_mod.Array(T), out);
    } else if (T == array_mod.BFloat16 and lhs.shape.len == 2 and rhs.shape.len == 2) {
        if (try axiom_mps.tryMatmulBF16(@as(array_mod.Array(array_mod.BFloat16), lhs), @as(array_mod.Array(array_mod.BFloat16), rhs))) |out| return @as(array_mod.Array(T), out);
    }
    if (T == f32 or T == f16 or T == array_mod.BFloat16) {
        if (lhs.shape.len == 3 and rhs.shape.len == 1) {
            return executeMpsBatchedMatvec(T, lhs, rhs);
        }
        if (lhs.shape.len == 1 and rhs.shape.len == 3) {
            return executeMpsBatchedVecmat(T, lhs, rhs);
        }
        if (lhs.shape.len == 1 and rhs.shape.len == 1) {
            var lhs_matrix = try lhs.reshape(&.{ 1, lhs.shape[0] });
            defer lhs_matrix.deinit();
            var rhs_matrix = try rhs.reshape(&.{ rhs.shape[0], 1 });
            defer rhs_matrix.deinit();
            var matrix_out = (try executeMpsMatmul(T, lhs_matrix, rhs_matrix)) orelse return null;
            errdefer matrix_out.deinit();
            const scalar = try matrix_out.reshape(&.{});
            matrix_out.deinit();
            return scalar;
        }
        if (lhs.shape.len == 2 and rhs.shape.len == 1) {
            var rhs_matrix = try rhs.reshape(&.{ rhs.shape[0], 1 });
            defer rhs_matrix.deinit();
            var matrix_out = (try executeMpsMatmul(T, lhs, rhs_matrix)) orelse return null;
            errdefer matrix_out.deinit();
            const vector = try matrix_out.reshape(&.{lhs.shape[0]});
            matrix_out.deinit();
            return vector;
        }
        if (lhs.shape.len == 1 and rhs.shape.len == 2) {
            var lhs_matrix = try lhs.reshape(&.{ 1, lhs.shape[0] });
            defer lhs_matrix.deinit();
            var matrix_out = (try executeMpsMatmul(T, lhs_matrix, rhs)) orelse return null;
            errdefer matrix_out.deinit();
            const vector = try matrix_out.reshape(&.{rhs.shape[1]});
            matrix_out.deinit();
            return vector;
        }
    }
    return null;
}

fn executeMpsBatchedMatvec(comptime T: type, matrix: array_mod.Array(T), vector: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (T == f32) {
        if (try axiom_mps.tryBatchedMatvecF32(@as(array_mod.Array(f32), matrix), @as(array_mod.Array(f32), vector))) |out| return @as(array_mod.Array(T), out);
    } else if (T == f16) {
        if (try axiom_mps.tryBatchedMatvecF16(@as(array_mod.Array(f16), matrix), @as(array_mod.Array(f16), vector))) |out| return @as(array_mod.Array(T), out);
    } else if (T == array_mod.BFloat16) {
        if (try axiom_mps.tryBatchedMatvecBF16(@as(array_mod.Array(array_mod.BFloat16), matrix), @as(array_mod.Array(array_mod.BFloat16), vector))) |out| return @as(array_mod.Array(T), out);
    }
    return null;
}

fn executeMpsBatchedVecmat(comptime T: type, vector: array_mod.Array(T), matrix: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (T == f32) {
        if (try axiom_mps.tryBatchedVecmatF32(@as(array_mod.Array(f32), vector), @as(array_mod.Array(f32), matrix))) |out| return @as(array_mod.Array(T), out);
    } else if (T == f16) {
        if (try axiom_mps.tryBatchedVecmatF16(@as(array_mod.Array(f16), vector), @as(array_mod.Array(f16), matrix))) |out| return @as(array_mod.Array(T), out);
    } else if (T == array_mod.BFloat16) {
        if (try axiom_mps.tryBatchedVecmatBF16(@as(array_mod.Array(array_mod.BFloat16), vector), @as(array_mod.Array(array_mod.BFloat16), matrix))) |out| return @as(array_mod.Array(T), out);
    }
    return null;
}

fn executeCudaBmm(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (T == f32) {
        const lhs32 = @as(array_mod.Array(f32), lhs);
        const rhs32 = @as(array_mod.Array(f32), rhs);
        if (try axiom_cuda.tryDeviceBmmF32(lhs32, rhs32)) |out| return @as(array_mod.Array(T), out);
        if (try axiom_cuda.tryDeviceBatchedMatmulF32(lhs32, rhs32)) |out| return @as(array_mod.Array(T), out);
    } else if (T == f64) {
        const lhs64 = @as(array_mod.Array(f64), lhs);
        const rhs64 = @as(array_mod.Array(f64), rhs);
        if (try axiom_cuda.tryDeviceBmmF64(lhs64, rhs64)) |out| return @as(array_mod.Array(T), out);
        if (try axiom_cuda.tryDeviceBatchedMatmulF64(lhs64, rhs64)) |out| return @as(array_mod.Array(T), out);
    } else if (T == f16) {
        const lhs16 = @as(array_mod.Array(f16), lhs);
        const rhs16 = @as(array_mod.Array(f16), rhs);
        if (try axiom_cuda.tryDeviceBmmF16(lhs16, rhs16)) |out| return @as(array_mod.Array(T), out);
        if (try axiom_cuda.tryDeviceBatchedMatmulF16(lhs16, rhs16)) |out| return @as(array_mod.Array(T), out);
    } else if (T == array_mod.BFloat16) {
        const lhs_bf16 = @as(array_mod.Array(array_mod.BFloat16), lhs);
        const rhs_bf16 = @as(array_mod.Array(array_mod.BFloat16), rhs);
        if (try axiom_cuda.tryDeviceBmmBF16(lhs_bf16, rhs_bf16)) |out| return @as(array_mod.Array(T), out);
        if (try axiom_cuda.tryDeviceBatchedMatmulBF16(lhs_bf16, rhs_bf16)) |out| return @as(array_mod.Array(T), out);
    }
    return null;
}

fn executeMpsBmm(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (try executeMpsRank4BroadcastBmm(T, lhs, rhs)) |out| return out;
    if (try executeMpsBroadcastBatchBmm(T, lhs, rhs)) |out| return out;
    if (try executeMpsFlattenedEqualBatchBmm(T, lhs, rhs)) |out| return out;
    if (T == f32) {
        if (try axiom_mps.tryBmmF32(@as(array_mod.Array(f32), lhs), @as(array_mod.Array(f32), rhs))) |out| return @as(array_mod.Array(T), out);
    } else if (T == f16) {
        if (try axiom_mps.tryBmmF16(@as(array_mod.Array(f16), lhs), @as(array_mod.Array(f16), rhs))) |out| return @as(array_mod.Array(T), out);
    } else if (T == array_mod.BFloat16) {
        if (try axiom_mps.tryBmmBF16(@as(array_mod.Array(array_mod.BFloat16), lhs), @as(array_mod.Array(array_mod.BFloat16), rhs))) |out| return @as(array_mod.Array(T), out);
    }
    return null;
}

fn executeMpsRank4BroadcastBmm(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (T != f32 and T != f16 and T != array_mod.BFloat16) return null;
    if (!lhs.device.isMps() or !rhs.device.isMps() or !lhs.device.sameDevice(rhs.device)) return null;
    if (lhs.shape.len != 4 or rhs.shape.len != 4 or !lhs.isContiguous() or !rhs.isContiguous()) return null;
    if (T == f32) {
        if (try axiom_mps.tryRank4BroadcastBmmF32(@as(array_mod.Array(f32), lhs), @as(array_mod.Array(f32), rhs))) |out| return @as(array_mod.Array(T), out);
    } else if (T == f16) {
        if (try axiom_mps.tryRank4BroadcastBmmF16(@as(array_mod.Array(f16), lhs), @as(array_mod.Array(f16), rhs))) |out| return @as(array_mod.Array(T), out);
    } else if (T == array_mod.BFloat16) {
        if (try axiom_mps.tryRank4BroadcastBmmBF16(@as(array_mod.Array(array_mod.BFloat16), lhs), @as(array_mod.Array(array_mod.BFloat16), rhs))) |out| return @as(array_mod.Array(T), out);
    }
    return null;
}

fn executeMpsBroadcastBatchBmm(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (T != f32 and T != f16 and T != array_mod.BFloat16) return null;
    if (!lhs.device.isMps() or !rhs.device.isMps() or !lhs.device.sameDevice(rhs.device)) return null;
    if (lhs.shape.len != 3 or rhs.shape.len != 3 or !lhs.isContiguous() or !rhs.isContiguous()) return null;
    const lhs_broadcast = lhs.shape[0] == 1 and rhs.shape[0] > 1;
    const rhs_broadcast = rhs.shape[0] == 1 and lhs.shape[0] > 1;
    if (lhs_broadcast == rhs_broadcast) return null;
    if (T == f32) {
        if (try axiom_mps.tryBroadcastBmmF32(@as(array_mod.Array(f32), lhs), @as(array_mod.Array(f32), rhs))) |out| return @as(array_mod.Array(T), out);
    } else if (T == f16) {
        if (try axiom_mps.tryBroadcastBmmF16(@as(array_mod.Array(f16), lhs), @as(array_mod.Array(f16), rhs))) |out| return @as(array_mod.Array(T), out);
    } else if (T == array_mod.BFloat16) {
        if (try axiom_mps.tryBroadcastBmmBF16(@as(array_mod.Array(array_mod.BFloat16), lhs), @as(array_mod.Array(array_mod.BFloat16), rhs))) |out| return @as(array_mod.Array(T), out);
    }
    return null;
}

fn executeMpsFlattenedEqualBatchBmm(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (T != f32 and T != f16 and T != array_mod.BFloat16) return null;
    if (!lhs.device.isMps() or !rhs.device.isMps() or !lhs.device.sameDevice(rhs.device)) return null;
    if (lhs.shape.len <= 3 or rhs.shape.len <= 3 or !lhs.isContiguous() or !rhs.isContiguous()) return null;
    const lhs_batch = lhs.shape[0 .. lhs.shape.len - 2];
    const rhs_batch = rhs.shape[0 .. rhs.shape.len - 2];
    if (!std.mem.eql(usize, lhs_batch, rhs_batch)) return null;

    const m = lhs.shape[lhs.shape.len - 2];
    const k = lhs.shape[lhs.shape.len - 1];
    const n = rhs.shape[rhs.shape.len - 1];
    var batch_count: usize = 1;
    for (lhs_batch) |extent| batch_count = std.math.mul(usize, batch_count, extent) catch return error.InvalidShape;

    const lhs_3d_shape = [_]usize{ batch_count, m, k };
    const rhs_3d_shape = [_]usize{ batch_count, k, n };
    var lhs_3d = try lhs.reshape(&lhs_3d_shape);
    defer lhs_3d.deinit();
    var rhs_3d = try rhs.reshape(&rhs_3d_shape);
    defer rhs_3d.deinit();
    var out_3d = (try executeMpsBmm(T, lhs_3d, rhs_3d)) orelse return null;
    defer out_3d.deinit();

    var out_shape = try lhs.allocator.alloc(usize, lhs_batch.len + 2);
    defer lhs.allocator.free(out_shape);
    @memcpy(out_shape[0..lhs_batch.len], lhs_batch);
    out_shape[lhs_batch.len] = m;
    out_shape[lhs_batch.len + 1] = n;
    return try out_3d.reshape(out_shape);
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
        .mps => executeMpsMatmulAdd(T, lhs, rhs, addend),
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
        .cpu => if (T == f32 or T == f64) executeCpuGemmScaledTarget(T, lhs, rhs, addend, alpha, beta) else null,
        .cuda => executeCudaMatmulAddScaled(T, lhs, rhs, addend, alpha, beta),
        .mps => executeMpsMatmulAddScaled(T, lhs, rhs, addend, alpha, beta),
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
    return switch (executionTargetForDevice(device)) {
        .cpu => true,
        .cuda, .mps => build_options.enable_device_host_fallback,
    };
}

pub fn deviceHostFallbackEnabled() bool {
    return build_options.enable_device_host_fallback;
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

pub fn composableElementwiseDeviceSupported(comptime T: type, device: array_mod.Device) bool {
    const target = executionTargetForDevice(device);
    return target == .cuda or (target == .mps and (T == f32 or T == f16 or T == array_mod.BFloat16));
}

pub fn composableElementwiseSameDeviceSupported(comptime T: type, lhs: array_mod.Device, rhs: array_mod.Device) bool {
    return lhs.sameDevice(rhs) and composableElementwiseDeviceSupported(T, lhs);
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
                .abs, .square, .log, .exp2, .expm1, .log1p, .log2, .log10, .sin, .cos, .tan, .asin, .acos, .atan => return false,
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

fn executeMpsMatmulAdd(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T), addend: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    return executeMpsMatmulAddScaled(T, lhs, rhs, addend, 1.0, 1.0);
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

fn executeMpsMatmulAddScaled(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T), addend: array_mod.Array(T), alpha: f32, beta: f32) array_mod.ArrayError!?array_mod.Array(T) {
    if (T == f32) {
        if (try axiom_mps.tryMatmulAddF32(@as(array_mod.Array(f32), lhs), @as(array_mod.Array(f32), rhs), @as(array_mod.Array(f32), addend), alpha, beta)) |out| return @as(array_mod.Array(T), out);
    } else if (T == f16) {
        if (try axiom_mps.tryMatmulAddF16(@as(array_mod.Array(f16), lhs), @as(array_mod.Array(f16), rhs), @as(array_mod.Array(f16), addend), alpha, beta)) |out| return @as(array_mod.Array(T), out);
    } else if (T == array_mod.BFloat16) {
        if (try axiom_mps.tryMatmulAddBF16(@as(array_mod.Array(array_mod.BFloat16), lhs), @as(array_mod.Array(array_mod.BFloat16), rhs), @as(array_mod.Array(array_mod.BFloat16), addend), alpha, beta)) |out| return @as(array_mod.Array(T), out);
    }
    return null;
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
        .mps => executeMpsUnary(T, op, input),
    };
}

pub fn executeUnaryDefault(comptime T: type, op: ExecutionUnaryOp, input: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    return executeUnary(T, op, defaultTargetForDevice(input.device), input);
}

pub fn executeDialectUnaryDefault(comptime T: type, op: DialectUnaryOp, input: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    return executeUnaryDefault(T, switch (op) {
        .abs => .abs,
        .square => .square,
        .sqrt => .sqrt,
        .exp => .exp,
        .log => .log,
        else => return null,
    }, input);
}

pub fn executeLogSoftmax(
    comptime T: type,
    target: DialectBackend,
    input: array_mod.Array(T),
    axis: u1,
) array_mod.ArrayError!?array_mod.Array(T) {
    if (!logSoftmaxRuntimeCapability(target).executable()) return null;
    if (!targetCanAccessDevice(target, input.device)) return null;
    if (!supportedSoftmaxExecution(T, target, input)) return null;
    return switch (target) {
        .cpu => null,
        .cuda => executeCudaLogSoftmax(T, input, axis),
        .mps => executeMpsLogSoftmax(T, input, axis),
    };
}

pub fn executeLogSoftmaxDefault(comptime T: type, input: array_mod.Array(T), axis: u1) array_mod.ArrayError!?array_mod.Array(T) {
    return executeLogSoftmax(T, defaultTargetForDevice(input.device), input, axis);
}

pub fn executeSoftmax(
    comptime T: type,
    target: DialectBackend,
    input: array_mod.Array(T),
    axis: u1,
) array_mod.ArrayError!?array_mod.Array(T) {
    if (!softmaxRuntimeCapability(target).executable()) return null;
    if (!targetCanAccessDevice(target, input.device)) return null;
    if (!supportedSoftmaxExecution(T, target, input)) return null;
    return switch (target) {
        .cpu => null,
        .cuda => executeCudaSoftmax(T, input, axis),
        .mps => executeMpsSoftmax(T, input, axis),
    };
}

pub fn executeSoftmaxDefault(comptime T: type, input: array_mod.Array(T), axis: u1) array_mod.ArrayError!?array_mod.Array(T) {
    return executeSoftmax(T, defaultTargetForDevice(input.device), input, axis);
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
        .mps => executeMpsReduction(T, op, input, axis, keepdims),
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
        .mps => executeMpsTranspose(T, input),
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
        .log => .log,
        .exp2 => .exp2,
        .expm1 => .expm1,
        .log1p => .log1p,
        .log2 => .log2,
        .log10 => .log10,
        .sin => .sin,
        .cos => .cos,
        .tan => .tan,
        .asin => .asin,
        .acos => .acos,
        .atan => .atan,
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
        .log => .log,
        .sin => .sin,
        .cos => .cos,
        .tan => .tan,
        .exp2 => .exp2,
        .expm1 => .expm1,
        .log1p => .log1p,
        .log2 => .log2,
        .log10 => .log10,
        .square, .asin, .acos, .atan => unreachable,
    };
    if (T == f32) {
        if (try axiom_cuda.tryDeviceUnaryF32(cuda_op, @as(array_mod.Array(f32), input))) |out| return @as(array_mod.Array(T), out);
    } else if (op == .log or op == .sin or op == .cos or op == .tan or op == .exp2 or op == .expm1 or op == .log1p or op == .log2 or op == .log10 or op == .asin or op == .acos or op == .atan) {
        return null;
    } else if (T == f16) {
        if (try axiom_cuda.tryDeviceUnaryF16(cuda_op, @as(array_mod.Array(f16), input))) |out| return @as(array_mod.Array(T), out);
    } else if (T == f64) {
        if (try axiom_cuda.tryDeviceUnaryF64(cuda_op, @as(array_mod.Array(f64), input))) |out| return @as(array_mod.Array(T), out);
    } else if (T == array_mod.BFloat16) {
        if (try axiom_cuda.tryDeviceUnaryBF16(cuda_op, @as(array_mod.Array(array_mod.BFloat16), input))) |out| return @as(array_mod.Array(T), out);
    }
    return null;
}

fn executeMpsUnary(comptime T: type, op: ExecutionUnaryOp, input: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (T == f32) {
        if (mpsUnaryOp(op)) |mps_op| {
            if (try axiom_mps.tryUnaryF32(mps_op, @as(array_mod.Array(f32), input))) |out| return @as(array_mod.Array(T), out);
        }
    } else if (T == f16) {
        if (mpsUnaryOpF16(op)) |mps_op| {
            if (try axiom_mps.tryUnaryF16(mps_op, @as(array_mod.Array(f16), input))) |out| return @as(array_mod.Array(T), out);
        }
    } else if (T == array_mod.BFloat16) {
        if (mpsUnaryOpBF16(op)) |mps_op| {
            if (try axiom_mps.tryUnaryBF16(mps_op, @as(array_mod.Array(array_mod.BFloat16), input))) |out| return @as(array_mod.Array(T), out);
        }
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
    if (bias.shape.len == 1) return bufferView(T, bias, name);
    if (bias.shape.len != 2 or bias.strides.len != 2) return null;
    const len, const stride_value = switch (axis) {
        // NumPy/PyTorch commonly keep reduced dimensions (`keepdims=True`),
        // producing row bias tensors shaped `[1, N]`.  Axiom's eager
        // row/column broadcast runtime consumes a vector ABI, so preserve the
        // source layout by projecting the non-singleton axis into a
        // TensorBufferView instead of falling back before the Axiom boundary.
        .row => .{ if (bias.shape[0] == 1) bias.shape[1] else return null, bias.strides[1] },
        .column => .{ if (bias.shape[1] == 1) bias.shape[0] else return null, bias.strides[0] },
    };
    const stride = std.math.cast(isize, stride_value) orelse return null;
    var view = axiom.accelerator.TensorBufferView.strided(name, @intCast(@intFromPtr(bias.data.ptr)), len, stride);
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

fn executeCudaLogSoftmax(comptime T: type, input: array_mod.Array(T), axis: u1) array_mod.ArrayError!?array_mod.Array(T) {
    if (T == f32) {
        if (try axiom_cuda.tryDeviceLogSoftmaxF32(@as(array_mod.Array(f32), input), axis)) |out| return @as(array_mod.Array(T), out);
    } else if (T == f64) {
        if (try axiom_cuda.tryDeviceLogSoftmaxF64(@as(array_mod.Array(f64), input), axis)) |out| return @as(array_mod.Array(T), out);
    } else if (T == f16) {
        if (try axiom_cuda.tryDeviceLogSoftmaxF16(@as(array_mod.Array(f16), input), axis)) |out| return @as(array_mod.Array(T), out);
    } else if (T == array_mod.BFloat16) {
        if (try axiom_cuda.tryDeviceLogSoftmaxBF16(@as(array_mod.Array(array_mod.BFloat16), input), axis)) |out| return @as(array_mod.Array(T), out);
    }
    return null;
}

fn executeCudaSoftmax(comptime T: type, input: array_mod.Array(T), axis: u1) array_mod.ArrayError!?array_mod.Array(T) {
    if (T == f32) {
        if (try axiom_cuda.tryDeviceSoftmaxF32(@as(array_mod.Array(f32), input), axis)) |out| return @as(array_mod.Array(T), out);
    } else if (T == f64) {
        if (try axiom_cuda.tryDeviceSoftmaxF64(@as(array_mod.Array(f64), input), axis)) |out| return @as(array_mod.Array(T), out);
    } else if (T == f16) {
        if (try axiom_cuda.tryDeviceSoftmaxF16(@as(array_mod.Array(f16), input), axis)) |out| return @as(array_mod.Array(T), out);
    } else if (T == array_mod.BFloat16) {
        if (try axiom_cuda.tryDeviceSoftmaxBF16(@as(array_mod.Array(array_mod.BFloat16), input), axis)) |out| return @as(array_mod.Array(T), out);
    }
    return null;
}

fn executeMpsLogSoftmax(comptime T: type, input: array_mod.Array(T), axis: u1) array_mod.ArrayError!?array_mod.Array(T) {
    if (T == f32) {
        if (try axiom_mps.trySoftmaxF32(.log_softmax, @as(array_mod.Array(f32), input), axis)) |out| return @as(array_mod.Array(T), out);
    } else if (T == f16) {
        if (try axiom_mps.trySoftmaxF16(.log_softmax, @as(array_mod.Array(f16), input), axis)) |out| return @as(array_mod.Array(T), out);
    } else if (T == array_mod.BFloat16) {
        if (try axiom_mps.trySoftmaxBF16(.log_softmax, @as(array_mod.Array(array_mod.BFloat16), input), axis)) |out| return @as(array_mod.Array(T), out);
    }
    return null;
}

fn executeMpsSoftmax(comptime T: type, input: array_mod.Array(T), axis: u1) array_mod.ArrayError!?array_mod.Array(T) {
    if (T == f32) {
        if (try axiom_mps.trySoftmaxF32(.softmax, @as(array_mod.Array(f32), input), axis)) |out| return @as(array_mod.Array(T), out);
    } else if (T == f16) {
        if (try axiom_mps.trySoftmaxF16(.softmax, @as(array_mod.Array(f16), input), axis)) |out| return @as(array_mod.Array(T), out);
    } else if (T == array_mod.BFloat16) {
        if (try axiom_mps.trySoftmaxBF16(.softmax, @as(array_mod.Array(array_mod.BFloat16), input), axis)) |out| return @as(array_mod.Array(T), out);
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
    if (T == f64) {
        if (try axiom_cuda.tryDeviceReductionF64(op, @as(array_mod.Array(f64), input), axis, keepdims)) |out| return @as(array_mod.Array(T), out);
    }
    if (T == f16) {
        if (try axiom_cuda.tryDeviceReductionF16(op, @as(array_mod.Array(f16), input), axis, keepdims)) |out| return @as(array_mod.Array(T), out);
    }
    if (T == array_mod.BFloat16) {
        if (try axiom_cuda.tryDeviceReductionBF16(op, @as(array_mod.Array(array_mod.BFloat16), input), axis, keepdims)) |out| return @as(array_mod.Array(T), out);
    }
    return null;
}

fn executeMpsReduction(
    comptime T: type,
    op: DialectReductionOp,
    input: array_mod.Array(T),
    axis: u1,
    keepdims: bool,
) array_mod.ArrayError!?array_mod.Array(T) {
    if (T == f32) {
        if (try axiom_mps.tryReductionF32(mpsReductionOp(op), @as(array_mod.Array(f32), input), axis, keepdims)) |out| return @as(array_mod.Array(T), out);
    } else if (T == f16) {
        if (try axiom_mps.tryReductionF16(mpsReductionOp(op), @as(array_mod.Array(f16), input), axis, keepdims)) |out| return @as(array_mod.Array(T), out);
    } else if (T == array_mod.BFloat16) {
        if (try axiom_mps.tryReductionBF16(mpsReductionOp(op), @as(array_mod.Array(array_mod.BFloat16), input), axis, keepdims)) |out| return @as(array_mod.Array(T), out);
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
    return executeCudaBroadcastBinary(T, .add, input, bias, axis);
}

fn executeCudaBroadcastBinary(comptime T: type, op: ElementwiseOp, input: array_mod.Array(T), bias: array_mod.Array(T), axis: DialectBroadcastAxis) array_mod.ArrayError!?array_mod.Array(T) {
    const cuda_op = cudaBinaryOp(op);
    if (T == f32) {
        if (try axiom_cuda.tryDeviceBroadcastBinaryF32(cuda_op, @as(array_mod.Array(f32), input), @as(array_mod.Array(f32), bias), axis)) |out| return @as(array_mod.Array(T), out);
    } else if (T == f64) {
        if (try axiom_cuda.tryDeviceBroadcastBinaryF64(cuda_op, @as(array_mod.Array(f64), input), @as(array_mod.Array(f64), bias), axis)) |out| return @as(array_mod.Array(T), out);
    } else if (T == f16) {
        if (try axiom_cuda.tryDeviceBroadcastBinaryF16(cuda_op, @as(array_mod.Array(f16), input), @as(array_mod.Array(f16), bias), axis)) |out| return @as(array_mod.Array(T), out);
    } else if (T == array_mod.BFloat16) {
        if (try axiom_cuda.tryDeviceBroadcastBinaryBF16(cuda_op, @as(array_mod.Array(array_mod.BFloat16), input), @as(array_mod.Array(array_mod.BFloat16), bias), axis)) |out| return @as(array_mod.Array(T), out);
    }
    return null;
}

fn executeMpsBroadcastAdd(comptime T: type, input: array_mod.Array(T), bias: array_mod.Array(T), axis: DialectBroadcastAxis) array_mod.ArrayError!?array_mod.Array(T) {
    return executeMpsBroadcastBinary(T, .add, input, bias, axis);
}

fn executeMpsBroadcastBinary(comptime T: type, op: ElementwiseOp, input: array_mod.Array(T), bias: array_mod.Array(T), axis: DialectBroadcastAxis) array_mod.ArrayError!?array_mod.Array(T) {
    const mps_op = mpsBinaryOp(op);
    if (T == f32) {
        if (try axiom_mps.tryBroadcastBinaryF32(mps_op, @as(array_mod.Array(f32), input), @as(array_mod.Array(f32), bias), axis)) |out| return @as(array_mod.Array(T), out);
    } else if (T == f16) {
        if (try axiom_mps.tryBroadcastBinaryF16(mps_op, @as(array_mod.Array(f16), input), @as(array_mod.Array(f16), bias), axis)) |out| return @as(array_mod.Array(T), out);
    } else if (T == array_mod.BFloat16) {
        if (try axiom_mps.tryBroadcastBinaryBF16(mps_op, @as(array_mod.Array(array_mod.BFloat16), input), @as(array_mod.Array(array_mod.BFloat16), bias), axis)) |out| return @as(array_mod.Array(T), out);
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
    } else if (T == f16) {
        if (try axiom_cuda.tryDeviceTransposeF16(@as(array_mod.Array(f16), input))) |out| return @as(array_mod.Array(T), out);
    } else if (T == array_mod.BFloat16) {
        if (try axiom_cuda.tryDeviceTransposeBF16(@as(array_mod.Array(array_mod.BFloat16), input))) |out| return @as(array_mod.Array(T), out);
    }
    return null;
}

fn executeMpsTranspose(comptime T: type, input: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (T == f32) {
        if (try axiom_mps.tryTransposeF32(@as(array_mod.Array(f32), input))) |out| return @as(array_mod.Array(T), out);
    } else if (T == f16) {
        if (try axiom_mps.tryTransposeF16(@as(array_mod.Array(f16), input))) |out| return @as(array_mod.Array(T), out);
    } else if (T == array_mod.BFloat16) {
        if (try axiom_mps.tryTransposeBF16(@as(array_mod.Array(array_mod.BFloat16), input))) |out| return @as(array_mod.Array(T), out);
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

fn recordCpuScalarElementwiseReport(report: anytype) void {
    last_cpu_scalar_elementwise_report = .{
        .ok = report.ok(),
        .operation = report.op.label(),
        .len = report.len,
        .scalar_on_lhs = report.scalar_on_lhs,
        .report_fingerprint = report.fingerprint(),
    };
}

fn executeCpuElementwiseScalar(
    comptime T: type,
    target: DialectBackend,
    op: ElementwiseOp,
    input: array_mod.Array(T),
    scalar: T,
    scalar_side: ScalarSide,
) array_mod.ArrayError!?array_mod.Array(T) {
    if (target != .cpu or !input.device.isCpu()) return null;
    if (T == f32) {
        var out = try array_mod.Array(f32).empty(input.allocator, input.shape);
        errdefer out.deinit();
        const report = axiom.accelerator.cpu_veyra.runTargetScalarElementwiseF32(
            .cpu,
            tensorBinaryOp(op),
            @as(array_mod.Array(f32), input).data,
            @as(f32, scalar),
            scalar_side == .lhs,
            out.data,
        ) catch {
            out.deinit();
            return null;
        };
        if (!report.ok()) {
            out.deinit();
            return null;
        }
        recordCpuScalarElementwiseReport(report);
        return @as(array_mod.Array(T), out);
    } else if (T == f64) {
        var out = try array_mod.Array(f64).empty(input.allocator, input.shape);
        errdefer out.deinit();
        const report = axiom.accelerator.cpu_veyra.runTargetScalarElementwiseF64(
            .cpu,
            tensorBinaryOp(op),
            @as(array_mod.Array(f64), input).data,
            @as(f64, scalar),
            scalar_side == .lhs,
            out.data,
        ) catch {
            out.deinit();
            return null;
        };
        if (!report.ok()) {
            out.deinit();
            return null;
        }
        recordCpuScalarElementwiseReport(report);
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
        .mps => executeMpsElementwise(T, op, lhs, rhs),
    };
}

pub fn executeElementwiseDefault(comptime T: type, op: ElementwiseOp, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    return executeElementwise(T, op, defaultTargetForDevice(lhs.device), lhs, rhs);
}

pub fn executeViewElementwise(
    comptime T: type,
    op: ElementwiseOp,
    target: DialectBackend,
    lhs: array_mod.ArrayView(T),
    rhs: array_mod.ArrayView(T),
) array_mod.ArrayError!?array_mod.Array(T) {
    if (!lhs.device.sameDevice(rhs.device) or !targetCanAccessDevice(target, lhs.device)) return null;
    return switch (target) {
        .cpu => executeCpuViewElementwise(T, op, lhs, rhs),
        .mps => null,
        .cuda => executeCudaViewElementwise(T, op, lhs, rhs),
    };
}

pub fn executeViewElementwiseDefault(comptime T: type, op: ElementwiseOp, lhs: array_mod.ArrayView(T), rhs: array_mod.ArrayView(T)) array_mod.ArrayError!?array_mod.Array(T) {
    return executeViewElementwise(T, op, defaultTargetForDevice(lhs.device), lhs, rhs);
}

pub fn executeViewElementwiseScalar(
    comptime T: type,
    op: ElementwiseOp,
    target: DialectBackend,
    input: array_mod.ArrayView(T),
    scalar: T,
    scalar_side: ScalarSide,
) array_mod.ArrayError!?array_mod.Array(T) {
    if (!targetCanAccessDevice(target, input.device)) return null;
    return switch (target) {
        .cpu => executeCpuViewElementwiseScalar(T, op, input, scalar, scalar_side),
        .mps => null,
        .cuda => executeCudaViewElementwiseScalar(T, op, input, scalar, scalar_side),
    };
}

pub fn executeViewElementwiseScalarDefault(comptime T: type, op: ElementwiseOp, input: array_mod.ArrayView(T), scalar: T, scalar_side: ScalarSide) array_mod.ArrayError!?array_mod.Array(T) {
    return executeViewElementwiseScalar(T, op, defaultTargetForDevice(input.device), input, scalar, scalar_side);
}

pub fn executeViewUnary(
    comptime T: type,
    op: ExecutionUnaryOp,
    target: DialectBackend,
    input: array_mod.ArrayView(T),
) array_mod.ArrayError!?array_mod.Array(T) {
    if (!targetCanAccessDevice(target, input.device)) return null;
    return switch (target) {
        .cpu => executeCpuViewUnary(T, op, input),
        .mps => null,
        .cuda => executeCudaViewUnary(T, op, input),
    };
}

pub fn executeViewUnaryDefault(comptime T: type, op: ExecutionUnaryOp, input: array_mod.ArrayView(T)) array_mod.ArrayError!?array_mod.Array(T) {
    return executeViewUnary(T, op, defaultTargetForDevice(input.device), input);
}

fn recordCpuViewElementwiseReport(report: anytype) void {
    last_cpu_view_elementwise_report = .{
        .ok = report.ok(),
        .operation = report.op.label(),
        .len = report.len,
        .spec_fingerprint = report.spec_fingerprint,
        .report_fingerprint = report.fingerprint(),
    };
}

fn hostViewBackingSlice(comptime T: type, view: array_mod.ArrayView(T)) ?[]const T {
    if (!view.device.isCpu() or view.shape.len != 1 or view.shape[0] == 0 or view.strides.len != 1 or view.strides[0] == 0) return null;
    const last_delta = std.math.mul(usize, view.shape[0] - 1, view.strides[0]) catch return null;
    const end_index = std.math.add(usize, view.offset, last_delta) catch return null;
    if (end_index >= view.data.len) return null;
    return view.data[view.offset .. end_index + 1];
}

fn cpuUnaryOp(op: ExecutionUnaryOp) axiom.accelerator.cpu_veyra.TensorUnaryElementwiseOp {
    return switch (op) {
        .abs => .abs,
        .square => .square,
        .sqrt => .sqrt,
        .exp => .exp,
        .log => .log,
        .exp2 => .exp2,
        .expm1 => .expm1,
        .log1p => .log1p,
        .log2 => .log2,
        .log10 => .log10,
        .sin => .sin,
        .cos => .cos,
        .tan => .tan,
        .asin => .asin,
        .acos => .acos,
        .atan => .atan,
    };
}

fn executeCpuViewElementwise(comptime T: type, op: ElementwiseOp, lhs: array_mod.ArrayView(T), rhs: array_mod.ArrayView(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (!lhs.device.isCpu() or !std.mem.eql(usize, lhs.shape, rhs.shape) or lhs.shape.len != 1) return null;
    if (T != f32 and T != f64) return null;
    const lhs_slice = hostViewBackingSlice(T, lhs) orelse return null;
    const rhs_slice = hostViewBackingSlice(T, rhs) orelse return null;
    var out = try array_mod.Array(T).empty(lhs.allocator, lhs.shape);
    errdefer out.deinit();
    const lhs_descriptor = describeViewMemRef(T, lhs, "lhs") catch {
        out.deinit();
        return null;
    };
    const rhs_descriptor = describeViewMemRef(T, rhs, "rhs") catch {
        out.deinit();
        return null;
    };
    const out_descriptor = describeArrayMemRef(T, out, "out") catch {
        out.deinit();
        return null;
    };
    const report = if (T == f32)
        axiom.accelerator.cpu_veyra.runTargetElementwiseMemRefsF32(
            .cpu,
            tensorBinaryOp(op),
            lhs_descriptor,
            rhs_descriptor,
            out_descriptor,
            @as([]const f32, lhs_slice),
            @as([]const f32, rhs_slice),
            out.data,
        ) catch {
            out.deinit();
            return null;
        }
    else
        axiom.accelerator.cpu_veyra.runTargetElementwiseMemRefsF64(
            .cpu,
            tensorBinaryOp(op),
            lhs_descriptor,
            rhs_descriptor,
            out_descriptor,
            @as([]const f64, lhs_slice),
            @as([]const f64, rhs_slice),
            out.data,
        ) catch {
            out.deinit();
            return null;
        };
    if (!report.ok()) {
        out.deinit();
        return null;
    }
    recordCpuViewElementwiseReport(report);
    return out;
}

fn executeCpuViewUnary(comptime T: type, op: ExecutionUnaryOp, input: array_mod.ArrayView(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (!input.device.isCpu() or input.shape.len != 1) return null;
    if (T != f32 and T != f64) return null;
    const input_slice = hostViewBackingSlice(T, input) orelse return null;
    var out = try array_mod.Array(T).empty(input.allocator, input.shape);
    errdefer out.deinit();
    const input_descriptor = describeViewMemRef(T, input, "input") catch {
        out.deinit();
        return null;
    };
    const out_descriptor = describeArrayMemRef(T, out, "out") catch {
        out.deinit();
        return null;
    };
    const report = if (T == f32)
        axiom.accelerator.cpu_veyra.runTargetUnaryElementwiseMemRefsF32(
            .cpu,
            cpuUnaryOp(op),
            input_descriptor,
            out_descriptor,
            @as([]const f32, input_slice),
            out.data,
        ) catch {
            out.deinit();
            return null;
        }
    else
        axiom.accelerator.cpu_veyra.runTargetUnaryElementwiseMemRefsF64(
            .cpu,
            cpuUnaryOp(op),
            input_descriptor,
            out_descriptor,
            @as([]const f64, input_slice),
            out.data,
        ) catch {
            out.deinit();
            return null;
        };
    if (!report.ok()) {
        out.deinit();
        return null;
    }
    recordCpuViewElementwiseReport(report);
    return out;
}

fn executeCpuViewElementwiseScalar(comptime T: type, op: ElementwiseOp, input: array_mod.ArrayView(T), scalar: T, scalar_side: ScalarSide) array_mod.ArrayError!?array_mod.Array(T) {
    if (!input.device.isCpu() or input.shape.len != 1) return null;
    if (T != f32 and T != f64) return null;
    const input_slice = hostViewBackingSlice(T, input) orelse return null;
    const scalar_values = [_]T{scalar};
    const scalar_shape = [_]usize{input.shape[0]};
    const scalar_strides = [_]isize{0};
    var out = try array_mod.Array(T).empty(input.allocator, input.shape);
    errdefer out.deinit();
    const input_descriptor = describeViewMemRef(T, input, "input") catch {
        out.deinit();
        return null;
    };
    const scalar_descriptor = axiom.accelerator.TensorMemRefDescriptor.init(
        "scalar",
        @intCast(@intFromPtr(&scalar_values[0])),
        tensorElementType(T) orelse {
            out.deinit();
            return null;
        },
        .host,
        0,
        scalar_shape[0..],
        scalar_strides[0..],
    ) catch {
        out.deinit();
        return null;
    };
    const out_descriptor = describeArrayMemRef(T, out, "out") catch {
        out.deinit();
        return null;
    };
    const lhs_descriptor = if (scalar_side == .lhs) scalar_descriptor else input_descriptor;
    const rhs_descriptor = if (scalar_side == .lhs) input_descriptor else scalar_descriptor;
    const report = if (T == f32)
        axiom.accelerator.cpu_veyra.runTargetElementwiseMemRefsF32(
            .cpu,
            tensorBinaryOp(op),
            lhs_descriptor,
            rhs_descriptor,
            out_descriptor,
            if (scalar_side == .lhs) scalar_values[0..] else @as([]const f32, input_slice),
            if (scalar_side == .lhs) @as([]const f32, input_slice) else scalar_values[0..],
            out.data,
        ) catch {
            out.deinit();
            return null;
        }
    else
        axiom.accelerator.cpu_veyra.runTargetElementwiseMemRefsF64(
            .cpu,
            tensorBinaryOp(op),
            lhs_descriptor,
            rhs_descriptor,
            out_descriptor,
            if (scalar_side == .lhs) scalar_values[0..] else @as([]const f64, input_slice),
            if (scalar_side == .lhs) @as([]const f64, input_slice) else scalar_values[0..],
            out.data,
        ) catch {
            out.deinit();
            return null;
        };
    if (!report.ok()) {
        out.deinit();
        return null;
    }
    recordCpuViewElementwiseReport(report);
    return out;
}

fn executeCudaViewElementwise(comptime T: type, op: ElementwiseOp, lhs: array_mod.ArrayView(T), rhs: array_mod.ArrayView(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (T == f32) {
        const out = switch (op) {
            .add => try axiom_cuda.tryAddViewF32(@as(array_mod.ArrayView(f32), lhs), @as(array_mod.ArrayView(f32), rhs)),
            .sub => try axiom_cuda.trySubViewF32(@as(array_mod.ArrayView(f32), lhs), @as(array_mod.ArrayView(f32), rhs)),
            .mul => try axiom_cuda.tryMulViewF32(@as(array_mod.ArrayView(f32), lhs), @as(array_mod.ArrayView(f32), rhs)),
            .div => try axiom_cuda.tryDivViewF32(@as(array_mod.ArrayView(f32), lhs), @as(array_mod.ArrayView(f32), rhs)),
        };
        if (out) |value| return @as(array_mod.Array(T), value);
    } else if (T == f64) {
        const out = switch (op) {
            .add => try axiom_cuda.tryAddViewF64(@as(array_mod.ArrayView(f64), lhs), @as(array_mod.ArrayView(f64), rhs)),
            .sub => try axiom_cuda.trySubViewF64(@as(array_mod.ArrayView(f64), lhs), @as(array_mod.ArrayView(f64), rhs)),
            .mul => try axiom_cuda.tryMulViewF64(@as(array_mod.ArrayView(f64), lhs), @as(array_mod.ArrayView(f64), rhs)),
            .div => try axiom_cuda.tryDivViewF64(@as(array_mod.ArrayView(f64), lhs), @as(array_mod.ArrayView(f64), rhs)),
        };
        if (out) |value| return @as(array_mod.Array(T), value);
    } else if (T == f16) {
        const out = switch (op) {
            .add => try axiom_cuda.tryAddViewF16(@as(array_mod.ArrayView(f16), lhs), @as(array_mod.ArrayView(f16), rhs)),
            .sub => try axiom_cuda.trySubViewF16(@as(array_mod.ArrayView(f16), lhs), @as(array_mod.ArrayView(f16), rhs)),
            .mul => try axiom_cuda.tryMulViewF16(@as(array_mod.ArrayView(f16), lhs), @as(array_mod.ArrayView(f16), rhs)),
            .div => try axiom_cuda.tryDivViewF16(@as(array_mod.ArrayView(f16), lhs), @as(array_mod.ArrayView(f16), rhs)),
        };
        if (out) |value| return @as(array_mod.Array(T), value);
    } else if (T == array_mod.BFloat16) {
        const out = switch (op) {
            .add => try axiom_cuda.tryAddViewBF16(@as(array_mod.ArrayView(array_mod.BFloat16), lhs), @as(array_mod.ArrayView(array_mod.BFloat16), rhs)),
            .sub => try axiom_cuda.trySubViewBF16(@as(array_mod.ArrayView(array_mod.BFloat16), lhs), @as(array_mod.ArrayView(array_mod.BFloat16), rhs)),
            .mul => try axiom_cuda.tryMulViewBF16(@as(array_mod.ArrayView(array_mod.BFloat16), lhs), @as(array_mod.ArrayView(array_mod.BFloat16), rhs)),
            .div => try axiom_cuda.tryDivViewBF16(@as(array_mod.ArrayView(array_mod.BFloat16), lhs), @as(array_mod.ArrayView(array_mod.BFloat16), rhs)),
        };
        if (out) |value| return @as(array_mod.Array(T), value);
    }
    return null;
}

fn executeCudaViewUnary(comptime T: type, op: ExecutionUnaryOp, input: array_mod.ArrayView(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (T == f32) {
        const out = switch (op) {
            .abs => try axiom_cuda.tryAbsViewF32(@as(array_mod.ArrayView(f32), input)),
            .sqrt => try axiom_cuda.trySqrtViewF32(@as(array_mod.ArrayView(f32), input)),
            .exp => try axiom_cuda.tryExpViewF32(@as(array_mod.ArrayView(f32), input)),
            .log => try axiom_cuda.tryLogViewF32(@as(array_mod.ArrayView(f32), input)),
            .exp2 => try axiom_cuda.tryExp2ViewF32(@as(array_mod.ArrayView(f32), input)),
            .expm1 => try axiom_cuda.tryExpm1ViewF32(@as(array_mod.ArrayView(f32), input)),
            .log1p => try axiom_cuda.tryLog1pViewF32(@as(array_mod.ArrayView(f32), input)),
            .log2 => try axiom_cuda.tryLog2ViewF32(@as(array_mod.ArrayView(f32), input)),
            .log10 => try axiom_cuda.tryLog10ViewF32(@as(array_mod.ArrayView(f32), input)),
            .sin => try axiom_cuda.trySinViewF32(@as(array_mod.ArrayView(f32), input)),
            .cos => try axiom_cuda.tryCosViewF32(@as(array_mod.ArrayView(f32), input)),
            .tan => try axiom_cuda.tryTanViewF32(@as(array_mod.ArrayView(f32), input)),
            .square, .asin, .acos, .atan => null,
        };
        if (out) |value| return @as(array_mod.Array(T), value);
    } else if (T == f64) {
        const out = switch (op) {
            .abs => try axiom_cuda.tryAbsViewF64(@as(array_mod.ArrayView(f64), input)),
            .sqrt => try axiom_cuda.trySqrtViewF64(@as(array_mod.ArrayView(f64), input)),
            .exp => try axiom_cuda.tryExpViewF64(@as(array_mod.ArrayView(f64), input)),
            .square, .log, .exp2, .expm1, .log1p, .log2, .log10, .sin, .cos, .tan, .asin, .acos, .atan => null,
        };
        if (out) |value| return @as(array_mod.Array(T), value);
    } else if (T == f16) {
        const out = switch (op) {
            .abs => try axiom_cuda.tryAbsViewF16(@as(array_mod.ArrayView(f16), input)),
            .sqrt => try axiom_cuda.trySqrtViewF16(@as(array_mod.ArrayView(f16), input)),
            .exp => try axiom_cuda.tryExpViewF16(@as(array_mod.ArrayView(f16), input)),
            .square, .log, .exp2, .expm1, .log1p, .log2, .log10, .sin, .cos, .tan, .asin, .acos, .atan => null,
        };
        if (out) |value| return @as(array_mod.Array(T), value);
    } else if (T == array_mod.BFloat16) {
        const out = switch (op) {
            .abs => try axiom_cuda.tryAbsViewBF16(@as(array_mod.ArrayView(array_mod.BFloat16), input)),
            .sqrt => try axiom_cuda.trySqrtViewBF16(@as(array_mod.ArrayView(array_mod.BFloat16), input)),
            .exp => try axiom_cuda.tryExpViewBF16(@as(array_mod.ArrayView(array_mod.BFloat16), input)),
            .square, .log, .exp2, .expm1, .log1p, .log2, .log10, .sin, .cos, .tan, .asin, .acos, .atan => null,
        };
        if (out) |value| return @as(array_mod.Array(T), value);
    }
    return null;
}

fn executeCudaViewElementwiseScalar(comptime T: type, op: ElementwiseOp, input: array_mod.ArrayView(T), scalar: T, scalar_side: ScalarSide) array_mod.ArrayError!?array_mod.Array(T) {
    if (T == f32) {
        if (try axiom_cuda.tryViewScalarF32(cudaBinaryOp(op), @as(array_mod.ArrayView(f32), input), @as(f32, scalar), scalar_side == .lhs)) |out| return @as(array_mod.Array(T), out);
    } else if (T == f64) {
        if (try axiom_cuda.tryViewScalarF64(cudaBinaryOp(op), @as(array_mod.ArrayView(f64), input), @as(f64, scalar), scalar_side == .lhs)) |out| return @as(array_mod.Array(T), out);
    } else if (T == f16) {
        if (try axiom_cuda.tryViewScalarF16(cudaBinaryOp(op), @as(array_mod.ArrayView(f16), input), @as(f16, scalar), scalar_side == .lhs)) |out| return @as(array_mod.Array(T), out);
    } else if (T == array_mod.BFloat16) {
        if (try axiom_cuda.tryViewScalarBF16(cudaBinaryOp(op), @as(array_mod.ArrayView(array_mod.BFloat16), input), @as(array_mod.BFloat16, scalar), scalar_side == .lhs)) |out| return @as(array_mod.Array(T), out);
    }
    return null;
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

fn executeMpsElementwise(comptime T: type, op: ElementwiseOp, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (T == f32) {
        if (try axiom_mps.tryBinaryF32(mpsBinaryOp(op), @as(array_mod.Array(f32), lhs), @as(array_mod.Array(f32), rhs))) |out| return @as(array_mod.Array(T), out);
    } else if (T == f16) {
        if (try axiom_mps.tryBinaryF16(mpsBinaryOp(op), @as(array_mod.Array(f16), lhs), @as(array_mod.Array(f16), rhs))) |out| return @as(array_mod.Array(T), out);
    } else if (T == array_mod.BFloat16) {
        if (try axiom_mps.tryBinaryBF16(mpsBinaryOp(op), @as(array_mod.Array(array_mod.BFloat16), lhs), @as(array_mod.Array(array_mod.BFloat16), rhs))) |out| return @as(array_mod.Array(T), out);
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

fn mpsBinaryOp(op: ElementwiseOp) axiom.accelerator.MpsBinaryOp {
    return switch (op) {
        .add => .add,
        .sub => .sub,
        .mul => .mul,
        .div => .div,
    };
}

fn mpsUnaryOp(op: ExecutionUnaryOp) ?axiom.accelerator.MpsUnaryOp {
    return switch (op) {
        .abs => .abs,
        .square => .square,
        .sqrt => .sqrt,
        .exp => .exp,
        .log => .log,
        .exp2 => .exp2,
        .expm1 => .expm1,
        .log1p => .log1p,
        .log2 => .log2,
        .log10 => .log10,
        .sin => .sin,
        .cos => .cos,
        .tan => .tan,
        else => null,
    };
}

fn mpsUnaryOpF16(op: ExecutionUnaryOp) ?axiom.accelerator.MpsUnaryOp {
    return switch (op) {
        .abs => .abs,
        .square => .square,
        .sqrt => .sqrt,
        .exp => .exp,
        else => null,
    };
}

fn mpsUnaryOpBF16(op: ExecutionUnaryOp) ?axiom.accelerator.MpsUnaryOp {
    return switch (op) {
        .abs => .abs,
        .square => .square,
        .sqrt => .sqrt,
        .exp => .exp,
        else => null,
    };
}

fn mpsReductionOp(op: DialectReductionOp) axiom.accelerator.MpsReductionOp {
    return switch (op) {
        .sum => .sum,
        .prod => .prod,
        .min => .min,
        .max => .max,
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
    if (try executeCudaElementwiseScalar(T, op, target, input, scalar, scalar_side)) |out| return out;
    if (try executeMpsElementwiseScalar(T, op, target, input, scalar, scalar_side)) |out| return out;
    if (try executeCpuElementwiseScalar(T, target, op, input, scalar, scalar_side)) |out| return out;
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

fn executeCudaElementwiseScalar(
    comptime T: type,
    op: ElementwiseOp,
    target: DialectBackend,
    input: array_mod.Array(T),
    scalar: T,
    scalar_side: ScalarSide,
) array_mod.ArrayError!?array_mod.Array(T) {
    if (target != .cuda or !input.device.isCuda() or input.shape.len == 0) return null;
    if (T != f32 and T != f64 and T != f16 and T != array_mod.BFloat16) return null;

    // Preserve the Axiom memref/runtime boundary for scalar Array APIs without
    // materializing a same-shape device buffer full of repeated scalar values.
    // A one-element device array is lowered as a zero-stride row-broadcast
    // memref by `tryCudaDeviceScalarArrayBroadcast`, matching NumPy/PyTorch
    // scalar-array semantics while keeping allocation and launch provenance in
    // Axiom's descriptor-backed CUDA path.
    var scalar_array = try array_mod.Array(T).fullOn(input.allocator, &.{1}, scalar, input.device);
    defer scalar_array.deinit();
    return switch (scalar_side) {
        .lhs => tryCudaDeviceScalarArrayBroadcast(T, op, target, scalar_array, input),
        .rhs => tryCudaDeviceScalarArrayBroadcast(T, op, target, input, scalar_array),
    };
}

fn executeMpsElementwiseScalar(
    comptime T: type,
    op: ElementwiseOp,
    target: DialectBackend,
    input: array_mod.Array(T),
    scalar: T,
    scalar_side: ScalarSide,
) array_mod.ArrayError!?array_mod.Array(T) {
    if (target != .mps or !input.device.isMps() or input.shape.len == 0) return null;
    if (T == f32) {
        if (try axiom_mps.tryScalarF32(mpsBinaryOp(op), @as(array_mod.Array(f32), input), @as(f32, scalar), scalar_side == .lhs)) |out| return @as(array_mod.Array(T), out);
    } else if (T == f16) {
        if (try axiom_mps.tryScalarF16(mpsBinaryOp(op), @as(array_mod.Array(f16), input), @as(f16, scalar), scalar_side == .lhs)) |out| return @as(array_mod.Array(T), out);
    } else if (T == array_mod.BFloat16) {
        if (try axiom_mps.tryScalarBF16(mpsBinaryOp(op), @as(array_mod.Array(array_mod.BFloat16), input), @as(array_mod.BFloat16, scalar), scalar_side == .lhs)) |out| return @as(array_mod.Array(T), out);
    }
    return null;
}

pub fn tryElementwiseScalarBroadcastDefault(comptime T: type, op: ElementwiseOp, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (try tryCudaDeviceScalarArrayBroadcast(T, op, defaultTargetForDevice(lhs.device), lhs, rhs)) |out| return out;
    if (lhs.data.len == rhs.data.len) return null;
    if (lhs.data.len == 1 and rhs.data.len != 0 and scalarBroadcastPreservesVectorShape(lhs.shape, rhs.shape)) return try executeElementwiseScalarDefault(T, op, rhs, lhs.data[0], .lhs);
    if (rhs.data.len == 1 and lhs.data.len != 0 and scalarBroadcastPreservesVectorShape(rhs.shape, lhs.shape)) return try executeElementwiseScalarDefault(T, op, lhs, rhs.data[0], .rhs);
    return null;
}

pub fn tryElementwiseScalarBroadcast(comptime T: type, op: ElementwiseOp, policy: BackendPolicy, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    const target = policyExecutionTarget(policy);
    if (try tryCudaDeviceScalarArrayBroadcast(T, op, target, lhs, rhs)) |out| return out;
    if (lhs.data.len == rhs.data.len) return null;
    if (lhs.data.len == 1 and rhs.data.len != 0 and scalarBroadcastPreservesVectorShape(lhs.shape, rhs.shape)) return try executeElementwiseScalar(T, op, target, rhs, lhs.data[0], .lhs);
    if (rhs.data.len == 1 and lhs.data.len != 0 and scalarBroadcastPreservesVectorShape(rhs.shape, lhs.shape)) return try executeElementwiseScalar(T, op, target, lhs, rhs.data[0], .rhs);
    return null;
}

fn tryCudaDeviceScalarArrayBroadcast(comptime T: type, op: ElementwiseOp, target: DialectBackend, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (target != .cuda or !lhs.device.sameDevice(rhs.device) or !lhs.device.isCuda()) return null;
    const lhs_scalar = lhs.numel() == 1;
    const rhs_scalar = rhs.numel() == 1;
    if (lhs_scalar == rhs_scalar) return null;
    const scalar_left = lhs_scalar;
    const vector = if (scalar_left) rhs else lhs;
    const scalar = if (scalar_left) lhs else rhs;
    if (vector.shape.len == 0) return null;
    const cuda_op = cudaBinaryOp(op);
    if (T == f32) {
        if (try axiom_cuda.tryDeviceContiguousScalarBroadcastF32(cuda_op, @as(array_mod.Array(f32), vector), @as(array_mod.Array(f32), scalar), scalar_left)) |out| return @as(array_mod.Array(T), out);
    } else if (T == f64) {
        if (try axiom_cuda.tryDeviceContiguousScalarBroadcastF64(cuda_op, @as(array_mod.Array(f64), vector), @as(array_mod.Array(f64), scalar), scalar_left)) |out| return @as(array_mod.Array(T), out);
    } else if (T == f16) {
        if (try axiom_cuda.tryDeviceContiguousScalarBroadcastF16(cuda_op, @as(array_mod.Array(f16), vector), @as(array_mod.Array(f16), scalar), scalar_left)) |out| return @as(array_mod.Array(T), out);
    } else if (T == array_mod.BFloat16) {
        if (try axiom_cuda.tryDeviceContiguousScalarBroadcastBF16(cuda_op, @as(array_mod.Array(array_mod.BFloat16), vector), @as(array_mod.Array(array_mod.BFloat16), scalar), scalar_left)) |out| return @as(array_mod.Array(T), out);
    }
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
    if (lhs.device.isMps()) {
        if (T != f32 and T != f16 and T != array_mod.BFloat16) return false;
        return (lhs.shape.len == 1 and rhs.shape.len == 1) or
            (lhs.shape.len == 2 and rhs.shape.len == 2) or
            (lhs.shape.len == 2 and rhs.shape.len == 1) or
            (lhs.shape.len == 1 and rhs.shape.len == 2) or
            (lhs.shape.len == 3 and rhs.shape.len == 1) or
            (lhs.shape.len == 1 and rhs.shape.len == 3);
    }
    if (!lhs.device.isCuda() or (T != f32 and T != f64 and T != f16 and T != array_mod.BFloat16)) return false;
    return (lhs.shape.len == 1 and rhs.shape.len == 1) or
        (lhs.shape.len == 2 and rhs.shape.len == 2) or
        (lhs.shape.len >= 2 and rhs.shape.len == 1) or
        (lhs.shape.len == 1 and rhs.shape.len >= 2);
}

fn supportedBmmExecution(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) bool {
    if (!lhs.device.sameDevice(rhs.device) or !lhs.isContiguous() or !rhs.isContiguous()) return false;
    if (lhs.shape.len < 3 or rhs.shape.len < 3) return false;
    if (!batchShapesBroadcastable(lhs.shape[0 .. lhs.shape.len - 2], rhs.shape[0 .. rhs.shape.len - 2])) return false;
    if (lhs.shape[lhs.shape.len - 2] == 0 or lhs.shape[lhs.shape.len - 1] == 0 or rhs.shape[rhs.shape.len - 1] == 0) return false;
    if (lhs.shape[lhs.shape.len - 1] != rhs.shape[rhs.shape.len - 2]) return false;
    if (lhs.device.isMps()) {
        return (T == f32 or T == f16 or T == array_mod.BFloat16) and
            lhs.device_storage != null and
            rhs.device_storage != null;
    }
    return lhs.device.isCuda() and (T == f32 or T == f64 or T == f16 or T == array_mod.BFloat16);
}

fn batchShapesBroadcastable(lhs: []const usize, rhs: []const usize) bool {
    const rank = @max(lhs.len, rhs.len);
    var index: usize = 0;
    while (index < rank) : (index += 1) {
        const lhs_dim: usize = if (index >= rank - lhs.len) lhs[index - (rank - lhs.len)] else 1;
        const rhs_dim: usize = if (index >= rank - rhs.len) rhs[index - (rank - rhs.len)] else 1;
        if (lhs_dim != rhs_dim and lhs_dim != 1 and rhs_dim != 1) return false;
    }
    return true;
}

fn supportedMatmulAddExecution(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T), addend: array_mod.Array(T)) bool {
    if (!lhs.device.sameDevice(rhs.device) or !lhs.device.sameDevice(addend.device)) return false;
    if (lhs.shape.len != 2 or rhs.shape.len != 2 or addend.shape.len != 2) return false;
    if (lhs.shape[1] != rhs.shape[0] or addend.shape[0] != lhs.shape[0] or addend.shape[1] != rhs.shape[1]) return false;
    if (!lhs.isContiguous() or !rhs.isContiguous() or !addend.isContiguous()) return false;
    if (lhs.device.isCpu()) return T == f32 or T == f64;
    if (lhs.device.isMps()) return (T == f32 or T == f16 or T == array_mod.BFloat16) and lhs.device_storage != null and rhs.device_storage != null and addend.device_storage != null;
    return lhs.device.isCuda() and (T == f32 or T == f64 or T == f16 or T == array_mod.BFloat16);
}

fn supportedReduction2d(comptime T: type, input: array_mod.Array(T)) bool {
    return dialectElement(T) != null and input.device.isCpu() and input.shape.len == 2 and input.isContiguous();
}

fn supportedReductionExecution(comptime T: type, target: DialectBackend, input: array_mod.Array(T)) bool {
    if (input.shape.len != 2 or !input.isContiguous()) return false;
    return switch (target) {
        .cpu => supportedReduction2d(T, input),
        .cuda => input.device.isCuda() and (T == f32 or T == f64 or T == f16 or T == array_mod.BFloat16) and input.device_storage != null,
        .mps => input.device.isMps() and (T == f32 or T == f16 or T == array_mod.BFloat16) and input.device_storage != null,
    };
}

fn supportedSoftmaxExecution(comptime T: type, target: DialectBackend, input: array_mod.Array(T)) bool {
    if (input.shape.len != 2 or !input.isContiguous()) return false;
    return switch (target) {
        .cpu => false,
        .cuda => input.device.isCuda() and (T == f32 or T == f64 or T == f16 or T == array_mod.BFloat16) and input.device_storage != null,
        .mps => input.device.isMps() and (T == f32 or T == f16 or T == array_mod.BFloat16) and input.device_storage != null,
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
        .cuda => input.device.isCuda() and (T == f32 or T == f64 or T == f16 or T == array_mod.BFloat16) and input.device_storage != null,
        .mps => input.device.isMps() and (T == f32 or T == f16 or T == array_mod.BFloat16) and input.device_storage != null,
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
    return (input.device.isCpu() or input.device.isCuda() or input.device.isMps()) and
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
    return supportedBroadcastBinaryExecution(T, .add, target, input, bias, axis);
}

fn supportedBroadcastBinaryExecution(comptime T: type, op: ElementwiseOp, target: DialectBackend, input: array_mod.Array(T), bias: array_mod.Array(T), axis: DialectBroadcastAxis) bool {
    return broadcastAddRuntimeCapability(target).executable() and
        targetCanAccessDevice(target, input.device) and
        input.device.sameDevice(bias.device) and
        switch (target) {
            .cpu => op == .add and supportedBroadcastAdd(T, input, bias, axis),
            .cuda => (T == f32 or T == f64 or T == f16 or T == array_mod.BFloat16) and input.device.isCuda() and supportedBroadcastAddLowering(T, input, bias, axis),
            .mps => (T == f32 or T == f16 or T == array_mod.BFloat16) and input.device.isMps() and supportedBroadcastAddLowering(T, input, bias, axis),
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
    if (input.device.sameDevice(bias.device) and input.shape.len == 2 and bias.numel() == 1) return true;
    return switch (axis) {
        .row => input.device.sameDevice(bias.device) and
            input.shape.len == 2 and
            bias.shape.len == 2 and
            bias.shape[0] == 1 and
            bias.shape[1] == input.shape[1],
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
        (input.device.isCpu() or input.device.isCuda() or input.device.isMps()) and
        nonEmptyAccessibleData(T, input) and
        input.isContiguous();
}

fn nonEmptyAccessibleData(comptime T: type, input: array_mod.Array(T)) bool {
    if (input.device.isCuda() or input.device.isMps()) {
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
    const platform_default = if (builtin.os.tag == .macos) DialectBackend.mps else DialectBackend.cpu;
    try std.testing.expectEqual(platform_default, defaultDialectBackend());
    const platform_default_report = try lowerMatmulDialectDefault(f32, a, b);
    try std.testing.expect(platform_default_report.ok());
    try std.testing.expectEqual(if (builtin.os.tag == .macos) DialectMatmulLoweringStatus.planned_mps else DialectMatmulLoweringStatus.lowered_cpu, platform_default_report.status);
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

    const log_report = try lowerUnaryDialect(f32, input, .log, .cuda);
    try std.testing.expect(log_report.ok());
    try std.testing.expectEqual(DialectUnaryLoweringStatus.lowered_cuda, log_report.status);
    try std.testing.expect(log_report.linalg_generic_fingerprint != cuda_report.linalg_generic_fingerprint);
    try std.testing.expect(unaryRuntimeCapability(.cuda, .log).executable());
    try std.testing.expect(unaryRuntimeCapability(.cpu, .log).executable());

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

test "Axiom runtime capability reports MPS executable and planned kernel slices" {
    const mps_reduction = reductionRuntimeCapability(.mps);
    try std.testing.expectEqual(RuntimeCapabilityStatus.executable, mps_reduction.status);
    try std.testing.expect(mps_reduction.executable());
    try std.testing.expect(mps_reduction.fingerprint() != 0);

    const mps_broadcast = broadcastAddRuntimeCapability(.mps);
    try std.testing.expectEqual(RuntimeCapabilityStatus.executable, mps_broadcast.status);
    try std.testing.expect(mps_broadcast.executable());

    const mps_unary = unaryRuntimeCapability(.mps, .log);
    try std.testing.expectEqual(RuntimeCapabilityStatus.executable, mps_unary.status);
    try std.testing.expect(mps_unary.executable());

    const mps_unary_unimplemented = unaryRuntimeCapability(.mps, .cube);
    try std.testing.expectEqual(RuntimeCapabilityStatus.planned, mps_unary_unimplemented.status);
    try std.testing.expect(!mps_unary_unimplemented.executable());

    const mps_transpose = transposeRuntimeCapability(.mps);
    try std.testing.expectEqual(RuntimeCapabilityStatus.executable, mps_transpose.status);
    try std.testing.expect(mps_transpose.executable());

    const mps_softmax = softmaxRuntimeCapability(.mps);
    try std.testing.expectEqual(RuntimeCapabilityStatus.executable, mps_softmax.status);
    try std.testing.expect(mps_softmax.executable());

    const mps_log_softmax = logSoftmaxRuntimeCapability(.mps);
    try std.testing.expectEqual(RuntimeCapabilityStatus.executable, mps_log_softmax.status);
    try std.testing.expect(mps_log_softmax.executable());

    const mps_runtime = mpsDeviceReport(0);
    if (builtin.os.tag == .macos) {
        try std.testing.expectEqual(MpsRuntimeAbiStatus.available, mps_runtime.status);
        try std.testing.expect(mps_runtime.ok());
        try std.testing.expect(mpsDeviceAvailable(0));
    } else {
        try std.testing.expectEqual(MpsRuntimeAbiStatus.unavailable, mps_runtime.status);
        try std.testing.expect(!mps_runtime.ok());
        try std.testing.expect(!mpsDeviceAvailable(0));
    }
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
    if (array_mod.Device.cuda(0).isAvailable()) {
        try std.testing.expect(scalar_broadcast != null);
        var scalar_broadcast_out = scalar_broadcast.?;
        defer scalar_broadcast_out.deinit();
        try std.testing.expectEqualSlices(f32, &.{ -8, -18, -28, -38 }, scalar_broadcast_out.data);
    } else {
        try std.testing.expect(scalar_broadcast == null);
        var scalar_broadcast_cpu = try elementwiseScalar(f32, .sub, .prefer_axiom_cpu, rhs32, 2.0, .lhs);
        defer scalar_broadcast_cpu.deinit();
        try std.testing.expectEqualSlices(f32, &.{ -8, -18, -28, -38 }, scalar_broadcast_cpu.data);
    }

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
