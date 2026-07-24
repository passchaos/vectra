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
const veyra = @import("veyra");
const axiom = @import("axiom");
const axiom_cpu = @import("axiom_cpu.zig");
const axiom_cuda = if (build_options.enable_axiom_cuda) @import("axiom_cuda.zig") else @import("axiom_cuda_stub.zig");
const axiom_mps = @import("axiom_mps.zig");

// Production CPU fast paths intentionally bypass Axiom's diagnostic
// report/hash/verify layer once the operation is large enough that the extra
// full-array scans dominate user-visible runtime.  Keep the thresholds
// centralized so new CPU fast paths do not drift independently.
// Unit tests use a much smaller streaming threshold so they exercise the same
// dispatch branches without allocating production-sized buffers.
const cpu_streaming_fast_path_min_elements: usize = if (builtin.is_test) 64 else 1 << 20;
const cpu_unary_fast_path_min_elements: usize = if (builtin.is_test) 64 else 8 * 1024;
const cpu_matmul_like_fast_path_min_ops: usize = 4 * 1024 * 1024;

pub fn cpuStreamingFastPathMinElements() usize {
    return cpu_streaming_fast_path_min_elements;
}

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
threadlocal var cached_f32_gemm_workspace: ?*veyra.GemmF32Workspace = null;
threadlocal var cached_f64_amx_gemm_workspace: ?*veyra.GemmF64AppleAmxWorkspace = null;
threadlocal var cached_f64_mt_gemm_workspace: ?*veyra.GemmF64MtWorkspace = null;
threadlocal var cached_f32_mt_gemm_workspace: ?*veyra.GemmF32MtWorkspace = null;

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

pub fn synchronizeDevice(allocator: std.mem.Allocator, device: array_mod.Device) array_mod.ArrayError!void {
    return switch (device.backend) {
        .cpu => {},
        .cuda => axiom_cuda.synchronizeDevice(allocator, device),
        // Current Axiom MPS commands commit and wait before returning from each
        // operation.  Keep the target-level completion API explicit so Array
        // callers have one synchronization boundary independent of backend,
        // while still rejecting fabricated unavailable MPS devices.
        .mps => if (axiom_mps.deviceAvailable(device.index)) {} else error.InvalidDevice,
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

pub fn fillPhiloxUniform(comptime T: type, storage: array_mod.DeviceStorage, seed: u64) array_mod.ArrayError!void {
    return switch (executionTargetForDevice(storage.device)) {
        .cpu => error.InvalidDevice,
        .cuda => axiom_cuda.fillPhiloxUniform(T, storage, seed),
        .mps => axiom_mps.fillPhiloxUniform(T, storage, seed),
    };
}

pub fn fillPhiloxNormal(comptime T: type, storage: array_mod.DeviceStorage, seed: u64, mean: T, stddev: T) array_mod.ArrayError!void {
    return switch (executionTargetForDevice(storage.device)) {
        .cpu => error.InvalidDevice,
        .cuda => axiom_cuda.fillPhiloxNormal(T, storage, seed, mean, stddev),
        .mps => axiom_mps.fillPhiloxNormal(T, storage, seed, mean, stddev),
    };
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
        .cpu => executeCpuBroadcastBinary(T, op, input, bias, axis),
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
    if (try tryMpsRank4Broadcast(T, op, target, lhs, rhs)) |out| return out;
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
    if (try tryMpsRankedBroadcast(T, op, target, lhs, rhs)) |out| return out;
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
    if (comptime !supportsAxiomCudaElementwise(T)) return null;
    return try axiom_cuda.tryDeviceBroadcast(T, cudaBinaryOp(op), lhs, rhs);
}

fn tryMpsRank4Broadcast(comptime T: type, op: ElementwiseOp, target: DialectBackend, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (target != .mps or !lhs.device.sameDevice(rhs.device) or !lhs.device.isMps()) return null;
    if (T == f32) {
        if (try axiom_mps.tryRank4BroadcastBinaryF32(mpsBinaryOp(op), @as(array_mod.Array(f32), lhs), @as(array_mod.Array(f32), rhs))) |out| return @as(array_mod.Array(T), out);
    } else if (T == f16) {
        if (try axiom_mps.tryRank4BroadcastBinaryF16(mpsBinaryOp(op), @as(array_mod.Array(f16), lhs), @as(array_mod.Array(f16), rhs))) |out| return @as(array_mod.Array(T), out);
    } else if (T == array_mod.BFloat16) {
        if (try axiom_mps.tryRank4BroadcastBinaryBF16(mpsBinaryOp(op), @as(array_mod.Array(array_mod.BFloat16), lhs), @as(array_mod.Array(array_mod.BFloat16), rhs))) |out| return @as(array_mod.Array(T), out);
    }
    return null;
}

fn tryMpsRankedBroadcast(comptime T: type, op: ElementwiseOp, target: DialectBackend, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (target != .mps or !lhs.device.sameDevice(rhs.device) or !lhs.device.isMps()) return null;
    if (lhs.shape.len <= 4 and rhs.shape.len <= 4) return null;
    if (T == f32) {
        if (try axiom_mps.tryRankedBroadcastBinaryF32(mpsBinaryOp(op), @as(array_mod.Array(f32), lhs), @as(array_mod.Array(f32), rhs))) |out| return @as(array_mod.Array(T), out);
    } else if (T == f16) {
        if (try axiom_mps.tryRankedBroadcastBinaryF16(mpsBinaryOp(op), @as(array_mod.Array(f16), lhs), @as(array_mod.Array(f16), rhs))) |out| return @as(array_mod.Array(T), out);
    } else if (T == array_mod.BFloat16) {
        if (try axiom_mps.tryRankedBroadcastBinaryBF16(mpsBinaryOp(op), @as(array_mod.Array(array_mod.BFloat16), lhs), @as(array_mod.Array(array_mod.BFloat16), rhs))) |out| return @as(array_mod.Array(T), out);
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
    if (try executeCpuDotFastPath(T, lhs, rhs)) |out| return out;

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
    if (try executeCpuMatvecFastPath(T, matrix, vector)) |out| return out;

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
    if (try executeCpuVecmatFastPath(T, vector, matrix)) |out| return out;

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

fn executeCpuDotFastPath(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (T != f32 and T != f64) return null;
    if (!lhs.device.isCpu() or !rhs.device.isCpu()) return null;
    if (lhs.shape.len != 1 or rhs.shape.len != 1 or lhs.shape[0] != rhs.shape[0]) return null;
    if (lhs.data.len < cpu_matmul_like_fast_path_min_ops) return null;
    if (!lhs.isContiguous() or !rhs.isContiguous()) return null;
    const value = if (T == f32)
        cpuDotSimd(f32, 8, lhs.data, rhs.data)
    else
        cpuDotSimd(f64, 4, lhs.data, rhs.data);
    return try array_mod.Array(T).fromSlice(lhs.allocator, &.{value}, &.{});
}

fn executeCpuMatvecFastPath(comptime T: type, matrix: array_mod.Array(T), vector: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (T != f32 and T != f64) return null;
    if (!matrix.device.isCpu() or !vector.device.isCpu()) return null;
    if (matrix.shape.len != 2 or vector.shape.len != 1 or matrix.shape[1] != vector.shape[0]) return null;
    if (matrix.data.len < cpu_matmul_like_fast_path_min_ops) return null;
    if (!matrix.isContiguous() or !vector.isContiguous()) return null;
    const rows = matrix.shape[0];
    const cols = matrix.shape[1];
    var out = try array_mod.Array(T).empty(matrix.allocator, &.{rows});
    errdefer out.deinit();
    if (T == f32)
        cpuMatvecSimd(f32, 8, out.data, matrix.data, vector.data, rows, cols)
    else
        cpuMatvecSimd(f64, 4, out.data, matrix.data, vector.data, rows, cols);
    return out;
}

fn executeCpuVecmatFastPath(comptime T: type, vector: array_mod.Array(T), matrix: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (T != f32 and T != f64) return null;
    if (!vector.device.isCpu() or !matrix.device.isCpu()) return null;
    if (vector.shape.len != 1 or matrix.shape.len != 2 or vector.shape[0] != matrix.shape[0]) return null;
    if (matrix.data.len < cpu_matmul_like_fast_path_min_ops) return null;
    if (!vector.isContiguous() or !matrix.isContiguous()) return null;
    const rows = matrix.shape[0];
    const cols = matrix.shape[1];
    var out = try array_mod.Array(T).empty(vector.allocator, &.{cols});
    errdefer out.deinit();
    @memset(out.data, 0);
    if (T == f32)
        cpuVecmatStreaming(f32, 8, out.data, vector.data, matrix.data, rows, cols)
    else
        cpuVecmatStreaming(f64, 4, out.data, vector.data, matrix.data, rows, cols);
    return out;
}

fn cpuDotSimd(comptime T: type, comptime lanes: usize, lhs: []const T, rhs: []const T) T {
    const Vec = @Vector(lanes, T);
    var i: usize = 0;
    var acc0: Vec = @splat(0);
    var acc1: Vec = @splat(0);
    while (i + lanes * 2 <= lhs.len) : (i += lanes * 2) {
        const lhs0: Vec = lhs[i..][0..lanes].*;
        const rhs0: Vec = rhs[i..][0..lanes].*;
        const lhs1: Vec = lhs[i + lanes ..][0..lanes].*;
        const rhs1: Vec = rhs[i + lanes ..][0..lanes].*;
        acc0 += lhs0 * rhs0;
        acc1 += lhs1 * rhs1;
    }
    while (i + lanes <= lhs.len) : (i += lanes) {
        const lhs_value: Vec = lhs[i..][0..lanes].*;
        const rhs_value: Vec = rhs[i..][0..lanes].*;
        acc0 += lhs_value * rhs_value;
    }
    var total: T = 0;
    inline for (0..lanes) |lane| {
        total += acc0[lane] + acc1[lane];
    }
    while (i < lhs.len) : (i += 1) {
        total += lhs[i] * rhs[i];
    }
    return total;
}

fn cpuMatvecSimd(comptime T: type, comptime lanes: usize, out: []T, matrix: []const T, vector: []const T, rows: usize, cols: usize) void {
    var row: usize = 0;
    while (row < rows) : (row += 1) {
        out[row] = cpuDotSimd(T, lanes, matrix[row * cols ..][0..cols], vector[0..cols]);
    }
}

fn cpuVecmatStreaming(comptime T: type, comptime lanes: usize, out: []T, vector: []const T, matrix: []const T, rows: usize, cols: usize) void {
    const Vec = @Vector(lanes, T);
    var row: usize = 0;
    while (row < rows) : (row += 1) {
        const scale_vec: Vec = @splat(vector[row]);
        const row_values = matrix[row * cols ..][0..cols];
        var col: usize = 0;
        while (col + lanes * 2 <= cols) : (col += lanes * 2) {
            const current0: Vec = out[col..][0..lanes].*;
            const values0: Vec = row_values[col..][0..lanes].*;
            const current1: Vec = out[col + lanes ..][0..lanes].*;
            const values1: Vec = row_values[col + lanes ..][0..lanes].*;
            out[col..][0..lanes].* = current0 + scale_vec * values0;
            out[col + lanes ..][0..lanes].* = current1 + scale_vec * values1;
        }
        while (col + lanes <= cols) : (col += lanes) {
            const current: Vec = out[col..][0..lanes].*;
            const values: Vec = row_values[col..][0..lanes].*;
            out[col..][0..lanes].* = current + scale_vec * values;
        }
        while (col < cols) : (col += 1) {
            out[col] += vector[row] * row_values[col];
        }
    }
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
    if (T == f32 and shouldDirectCpuF32NativeGemm(m, n, k)) {
        return executeCpuGemmDirect(T, lhs, rhs, null, 1.0, 0.0);
    }
    if (T == f32 and shouldMaterializeCpuF32ColumnMajorGemm(m, n, k)) {
        var out = try array_mod.Array(T).empty(lhs.allocator, &.{ m, n });
        errdefer out.deinit();
        if (try cpuMatmulColumnMajorResult(T, lhs, rhs)) |column_out| {
            var materialized = column_out;
            defer materialized.deinit();
            copyColumnMajorMatrixToRowMajor(T, out.data, materialized.data, m, n);
            return out;
        }
    }
    if (T == f64 and shouldMaterializeCpuF64ColumnMajorGemm(m, n, k)) {
        var out = try array_mod.Array(T).empty(lhs.allocator, &.{ m, n });
        errdefer out.deinit();
        if (try cpuMatmulColumnMajorResult(T, lhs, rhs)) |column_out| {
            var materialized = column_out;
            defer materialized.deinit();
            copyColumnMajorMatrixToRowMajor(T, out.data, materialized.data, m, n);
            return out;
        }
    }
    if (largeCpuGemm(m, n, k)) return executeCpuGemmDirect(T, lhs, rhs, null, 1.0, 0.0);

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

fn largeCpuGemm(m: usize, n: usize, k: usize) bool {
    const mn = std.math.mul(usize, m, n) catch return true;
    const work = std.math.mul(usize, mn, k) catch return true;
    return work >= cpu_matmul_like_fast_path_min_ops;
}

fn copyColumnMajorMatrixToRowMajor(comptime T: type, out: []T, column_major: []const T, rows: usize, cols: usize) void {
    std.debug.assert(out.len >= rows * cols);
    std.debug.assert(column_major.len >= rows * cols);
    if (comptime T == f32 or T == f64) {
        copyColumnMajorMatrixToRowMajor8x4(T, out[0 .. rows * cols], column_major[0 .. rows * cols], rows, cols);
        return;
    }

    const block: usize = 32;
    var row0: usize = 0;
    while (row0 < rows) : (row0 += block) {
        const row_end = @min(row0 + block, rows);
        var col0: usize = 0;
        while (col0 < cols) : (col0 += block) {
            const col_end = @min(col0 + block, cols);
            var row = row0;
            while (row < row_end) : (row += 1) {
                const dst_row = out[row * cols .. row * cols + cols];
                var col = col0;
                while (col < col_end) : (col += 1) {
                    dst_row[col] = column_major[col * rows + row];
                }
            }
        }
    }
}

fn copyColumnMajorMatrixToRowMajor8x4(comptime T: type, out: []T, column_major: []const T, rows: usize, cols: usize) void {
    const block: usize = 32;
    var row0: usize = 0;
    while (row0 < rows) : (row0 += block) {
        const row_end = @min(row0 + block, rows);
        var col0: usize = 0;
        while (col0 < cols) : (col0 += block) {
            const col_end = @min(col0 + block, cols);
            var row = row0;
            while (row + 8 <= row_end) : (row += 8) {
                var col = col0;
                while (col + 4 <= col_end) : (col += 4) {
                    const c0a: @Vector(4, T) = column_major[(col + 0) * rows + row ..][0..4].*;
                    const c0b: @Vector(4, T) = column_major[(col + 0) * rows + row + 4 ..][0..4].*;
                    const c1a: @Vector(4, T) = column_major[(col + 1) * rows + row ..][0..4].*;
                    const c1b: @Vector(4, T) = column_major[(col + 1) * rows + row + 4 ..][0..4].*;
                    const c2a: @Vector(4, T) = column_major[(col + 2) * rows + row ..][0..4].*;
                    const c2b: @Vector(4, T) = column_major[(col + 2) * rows + row + 4 ..][0..4].*;
                    const c3a: @Vector(4, T) = column_major[(col + 3) * rows + row ..][0..4].*;
                    const c3b: @Vector(4, T) = column_major[(col + 3) * rows + row + 4 ..][0..4].*;
                    out[(row + 0) * cols + col ..][0..4].* = .{ c0a[0], c1a[0], c2a[0], c3a[0] };
                    out[(row + 1) * cols + col ..][0..4].* = .{ c0a[1], c1a[1], c2a[1], c3a[1] };
                    out[(row + 2) * cols + col ..][0..4].* = .{ c0a[2], c1a[2], c2a[2], c3a[2] };
                    out[(row + 3) * cols + col ..][0..4].* = .{ c0a[3], c1a[3], c2a[3], c3a[3] };
                    out[(row + 4) * cols + col ..][0..4].* = .{ c0b[0], c1b[0], c2b[0], c3b[0] };
                    out[(row + 5) * cols + col ..][0..4].* = .{ c0b[1], c1b[1], c2b[1], c3b[1] };
                    out[(row + 6) * cols + col ..][0..4].* = .{ c0b[2], c1b[2], c2b[2], c3b[2] };
                    out[(row + 7) * cols + col ..][0..4].* = .{ c0b[3], c1b[3], c2b[3], c3b[3] };
                }
                while (col < col_end) : (col += 1) {
                    out[(row + 0) * cols + col] = column_major[col * rows + row + 0];
                    out[(row + 1) * cols + col] = column_major[col * rows + row + 1];
                    out[(row + 2) * cols + col] = column_major[col * rows + row + 2];
                    out[(row + 3) * cols + col] = column_major[col * rows + row + 3];
                    out[(row + 4) * cols + col] = column_major[col * rows + row + 4];
                    out[(row + 5) * cols + col] = column_major[col * rows + row + 5];
                    out[(row + 6) * cols + col] = column_major[col * rows + row + 6];
                    out[(row + 7) * cols + col] = column_major[col * rows + row + 7];
                }
            }
            while (row < row_end) : (row += 1) {
                const dst_row = out[row * cols .. row * cols + cols];
                var col = col0;
                while (col < col_end) : (col += 1) {
                    dst_row[col] = column_major[col * rows + row];
                }
            }
        }
    }
}

fn materializeColumnMajorGemmAdd(
    comptime T: type,
    out: []T,
    column_major: []const T,
    addend: array_mod.Array(T),
    rows: usize,
    cols: usize,
    alpha: f32,
    beta: f32,
) bool {
    if (addend.shape.len != 2 or addend.shape[0] != rows or addend.shape[1] != cols or !addend.isContiguous()) return false;
    std.debug.assert(out.len >= rows * cols);
    std.debug.assert(column_major.len >= rows * cols);

    const alpha_t: T = @floatCast(alpha);
    const beta_t: T = @floatCast(beta);
    const alpha_vec: @Vector(4, T) = @splat(alpha_t);
    const beta_vec: @Vector(4, T) = @splat(beta_t);

    const block: usize = 32;
    var row0: usize = 0;
    while (row0 < rows) : (row0 += block) {
        const row_end = @min(row0 + block, rows);
        var col0: usize = 0;
        while (col0 < cols) : (col0 += block) {
            const col_end = @min(col0 + block, cols);
            var row = row0;
            while (row + 4 <= row_end) : (row += 4) {
                var col = col0;
                while (col + 4 <= col_end) : (col += 4) {
                    const c0: @Vector(4, T) = column_major[(col + 0) * rows + row ..][0..4].*;
                    const c1: @Vector(4, T) = column_major[(col + 1) * rows + row ..][0..4].*;
                    const c2: @Vector(4, T) = column_major[(col + 2) * rows + row ..][0..4].*;
                    const c3: @Vector(4, T) = column_major[(col + 3) * rows + row ..][0..4].*;

                    const add0: @Vector(4, T) = addend.data[(row + 0) * cols + col ..][0..4].*;
                    const add1: @Vector(4, T) = addend.data[(row + 1) * cols + col ..][0..4].*;
                    const add2: @Vector(4, T) = addend.data[(row + 2) * cols + col ..][0..4].*;
                    const add3: @Vector(4, T) = addend.data[(row + 3) * cols + col ..][0..4].*;

                    out[(row + 0) * cols + col ..][0..4].* = alpha_vec * @as(@Vector(4, T), .{ c0[0], c1[0], c2[0], c3[0] }) + beta_vec * add0;
                    out[(row + 1) * cols + col ..][0..4].* = alpha_vec * @as(@Vector(4, T), .{ c0[1], c1[1], c2[1], c3[1] }) + beta_vec * add1;
                    out[(row + 2) * cols + col ..][0..4].* = alpha_vec * @as(@Vector(4, T), .{ c0[2], c1[2], c2[2], c3[2] }) + beta_vec * add2;
                    out[(row + 3) * cols + col ..][0..4].* = alpha_vec * @as(@Vector(4, T), .{ c0[3], c1[3], c2[3], c3[3] }) + beta_vec * add3;
                }
                while (col < col_end) : (col += 1) {
                    out[(row + 0) * cols + col] = alpha_t * column_major[col * rows + row + 0] + beta_t * addend.data[(row + 0) * cols + col];
                    out[(row + 1) * cols + col] = alpha_t * column_major[col * rows + row + 1] + beta_t * addend.data[(row + 1) * cols + col];
                    out[(row + 2) * cols + col] = alpha_t * column_major[col * rows + row + 2] + beta_t * addend.data[(row + 2) * cols + col];
                    out[(row + 3) * cols + col] = alpha_t * column_major[col * rows + row + 3] + beta_t * addend.data[(row + 3) * cols + col];
                }
            }
            while (row < row_end) : (row += 1) {
                var col = col0;
                while (col < col_end) : (col += 1) {
                    out[row * cols + col] = alpha_t * column_major[col * rows + row] + beta_t * addend.data[row * cols + col];
                }
            }
        }
    }
    return true;
}

fn getCachedF32GemmWorkspace() !*veyra.GemmF32Workspace {
    if (cached_f32_gemm_workspace) |workspace| return workspace;

    const workspace = try std.heap.smp_allocator.create(veyra.GemmF32Workspace);
    errdefer std.heap.smp_allocator.destroy(workspace);
    workspace.* = try veyra.GemmF32Workspace.init(std.heap.smp_allocator, 1);
    cached_f32_gemm_workspace = workspace;
    return workspace;
}

fn getCachedF64MtGemmWorkspace() !*veyra.GemmF64MtWorkspace {
    if (cached_f64_mt_gemm_workspace) |workspace| return workspace;

    const workspace = try std.heap.smp_allocator.create(veyra.GemmF64MtWorkspace);
    errdefer std.heap.smp_allocator.destroy(workspace);
    workspace.* = try veyra.GemmF64MtWorkspace.init(std.heap.smp_allocator, veyra.recommendedGemmF64ThreadCount());
    cached_f64_mt_gemm_workspace = workspace;
    return workspace;
}

fn getCachedF64AmxGemmWorkspace(rows: usize, cols: usize, depth: usize) !*veyra.GemmF64AppleAmxWorkspace {
    const thread_count = veyra.recommendedGemmF64ThreadCount();
    if (cached_f64_amx_gemm_workspace) |workspace| {
        if (workspace.a_panels.len >= rows * depth and
            workspace.b_panels.len >= cols * depth)
        {
            return workspace;
        }
        workspace.deinit();
        std.heap.smp_allocator.destroy(workspace);
        cached_f64_amx_gemm_workspace = null;
    }

    const workspace = try std.heap.smp_allocator.create(veyra.GemmF64AppleAmxWorkspace);
    errdefer std.heap.smp_allocator.destroy(workspace);
    workspace.* = try veyra.GemmF64AppleAmxWorkspace.init(std.heap.smp_allocator, rows, cols, depth, thread_count);
    cached_f64_amx_gemm_workspace = workspace;
    return workspace;
}

fn getCachedF32MtGemmWorkspace() !*veyra.GemmF32MtWorkspace {
    if (cached_f32_mt_gemm_workspace) |workspace| return workspace;

    const workspace = try std.heap.smp_allocator.create(veyra.GemmF32MtWorkspace);
    errdefer std.heap.smp_allocator.destroy(workspace);
    workspace.* = try veyra.GemmF32MtWorkspace.init(std.heap.smp_allocator, veyra.recommendedGemmF32ThreadCount());
    cached_f32_mt_gemm_workspace = workspace;
    return workspace;
}

fn executeCpuGemmDirect(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T), addend: ?array_mod.Array(T), alpha: f32, beta: f32) array_mod.ArrayError!?array_mod.Array(T) {
    if (T != f32 and T != f64) return null;
    const m = lhs.shape[0];
    const k = lhs.shape[1];
    const n = rhs.shape[1];
    var out = try array_mod.Array(T).empty(lhs.allocator, &.{ m, n });
    errdefer out.deinit();
    if (addend) |c| {
        if (c.shape.len != 2 or c.shape[0] != m or c.shape[1] != n or !c.isContiguous()) {
            out.deinit();
            return null;
        }
        if (beta != 0) {
            @memcpy(out.data, c.data);
        }
    }
    // For beta == 0 the GEMM contract overwrites `out` without reading the
    // previous destination values.  Avoid a full pre-zero of huge production
    // outputs; on the large CPU example that would be another multi-GB memory
    // pass before the actual compute starts.

    const lhs_view = veyra.MatrixView(T).fromSlice(lhs.data, m, k, .row_major) catch {
        out.deinit();
        return null;
    };
    const rhs_view = veyra.MatrixView(T).fromSlice(rhs.data, k, n, .row_major) catch {
        out.deinit();
        return null;
    };
    const out_view = veyra.MatrixMut(T).fromSlice(out.data, m, n, .row_major) catch {
        out.deinit();
        return null;
    };
    if (T == f32) {
        const options: veyra.GemmOptions(f32) = .{ .alpha = @floatCast(alpha), .beta = @floatCast(beta) };
        const workspace = getCachedF32GemmWorkspace() catch {
            out.deinit();
            return null;
        };
        restoreCpuGemmDestination(T, out.data, addend, beta);
        veyra.gemmF32WithWorkspace(lhs_view, rhs_view, out_view, options, workspace) catch {
            out.deinit();
            return null;
        };
    } else {
        const options: veyra.GemmOptions(f64) = .{ .alpha = @floatCast(alpha), .beta = @floatCast(beta) };
        var threaded_ran = false;
        const mt_workspace = getCachedF64MtGemmWorkspace() catch null;
        if (mt_workspace) |workspace| {
            veyra.ensureGemmF64MtAppleAmxWorkspace(workspace, m, n, k) catch {};
            threaded_ran = blk: {
                veyra.gemmThreadedWithWorkspace(f64, lhs_view, rhs_view, out_view, options, workspace) catch break :blk false;
                break :blk true;
            };
        }
        if (!threaded_ran) {
            restoreCpuGemmDestination(T, out.data, addend, beta);
            var workspace = veyra.GemmF64Workspace.init(std.heap.smp_allocator, @max(n, 1)) catch {
                out.deinit();
                return null;
            };
            defer workspace.deinit();
            veyra.gemmF64WithWorkspace(lhs_view, rhs_view, out_view, options, &workspace) catch {
                out.deinit();
                return null;
            };
        }
    }
    return out;
}

fn shouldDirectCpuF32NativeGemm(m: usize, n: usize, k: usize) bool {
    return m <= 32 and k <= 32 and n >= 64;
}

fn isSquareGemm(m: usize, n: usize, k: usize) bool {
    return m == n and n == k;
}

fn isCpuF32ColumnMajorSquareGemm(m: usize, n: usize, k: usize) bool {
    if (!isSquareGemm(m, n, k)) return false;
    return switch (m) {
        64, 96, 128, 130, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 224 => true,
        else => false,
    };
}

fn isCpuF32ColumnMajorMeasuredRectGemm(m: usize, n: usize, k: usize) bool {
    return (n == 100 and k == 100 and (m == 10 or m == 50)) or
        (m == 128 and n == 128 and (k == 16 or k == 32)) or
        (m == 192 and n == 96 and (k == 16 or k == 32)) or
        (m == 64 and n == 192 and (k == 16 or k == 32)) or
        (m == 64 and n == 128 and k == 128) or
        (m == 128 and n == 64 and k == 128) or
        (m == 128 and n == 128 and k == 64) or
        (m == 64 and n == 192 and k == 192) or
        (m == 64 and n == 192 and k == 128) or
        (m == 96 and n == 192 and k == 128) or
        (m == 192 and n == 96 and k == 128) or
        (m == 192 and n == 64 and k == 192) or
        (m == 192 and n == 192 and k == 64) or
        (m == 64 and n == 64 and k == 192) or
        (m == 64 and n == 256 and k == 256) or
        (m == 128 and n == 192 and k == 192) or
        (m == 192 and n == 256 and k == 192) or
        (m == 256 and n == 192 and k == 256) or
        (m == 128 and n == 512 and k == 128) or
        (m == 512 and n == 128 and k == 512) or
        (m == 128 and n == 384 and k == 128) or
        (m == 384 and n == 128 and k == 128) or
        (m == 128 and n == 256 and k == 256) or
        (m == 256 and n == 128 and k == 256) or
        (n == 64 and m == 128 and k == 256) or
        (n == 64 and m == 256 and (k == 128 or k == 192 or k == 256));
}

fn isCpuF32ColumnMajorPanelRuleGemm(m: usize, n: usize, k: usize) bool {
    const aligned_medium = m % 16 == 0 and n % 16 == 0 and k % 16 == 0 and
        m >= 128 and n >= 128 and k >= 128 and
        m <= 512 and n <= 512 and k <= 256;
    const low_k_large_square = m == n and m == 768 and k == 128;
    return aligned_medium or
        low_k_large_square or
        (m <= 256 and n <= 256 and k == 64 and n >= 128) or
        (k == 128 and ((m == 128 and n == 256) or
            (n == 128 and (m == 192 or m == 256)) or
            (m >= 192 and n >= 192 and m <= 256 and n <= 256))) or
        (k == 192 and ((m == 128 and n == 256) or (n == 128 and (m == 192 or m == 256))));
}

fn shouldMaterializeCpuF32ColumnMajorGemm(m: usize, n: usize, k: usize) bool {
    return isCpuF32ColumnMajorSquareGemm(m, n, k) or
        isCpuF32ColumnMajorMeasuredRectGemm(m, n, k) or
        isCpuF32ColumnMajorPanelRuleGemm(m, n, k);
}

fn isCpuF64ColumnMajorSquareGemm(m: usize, n: usize, k: usize) bool {
    if (!isSquareGemm(m, n, k)) return false;
    return switch (m) {
        100, 130, 132, 136, 140, 148, 150, 152, 156, 164, 168, 172, 180, 184, 188 => true,
        else => false,
    };
}

fn isCpuF64ColumnMajorMeasuredRectGemm(m: usize, n: usize, k: usize) bool {
    if (m % 16 != 0 or n % 16 != 0 or k % 16 != 0) return false;
    return (m == 96 and n == 96 and k == 96) or
        (m == 16 and n == 512 and k == 16) or
        (m == 16 and n == 1024 and k == 16) or
        (m == 32 and n == 512 and k == 32) or
        (m == 32 and n == 1024 and k == 32) or
        (m <= 32 and k <= 32 and n >= 64 and n <= 256) or
        (m <= 256 and n <= 256 and k <= 64 and n >= 64) or
        (m == 128 and n == 256 and k == 128) or
        (m == 256 and n == 128 and k == 128) or
        (m == 128 and n == 256 and k == 256) or
        (m == 256 and n == 128 and k == 256) or
        (m == 256 and n == 256 and k == 128) or
        (m == 192 and n == 192 and k == 128) or
        (m == 192 and n == 224 and k == 128) or
        (m == 224 and n == 192 and k == 128) or
        (m == 128 and n == 384 and k == 128) or
        (m == 384 and n == 128 and k == 128) or
        (m == 384 and n == 384 and k == 128) or
        (m == 144 and n == 224 and k == 192) or
        (m == 128 and n == 256 and k == 192) or
        (m == 256 and n == 128 and k == 192) or
        (m == 144 and n == 144 and k == 144) or
        (m == 160 and n == 160 and k == 160) or
        (m == 176 and n == 176 and k == 176) or
        (m == 128 and n == 128 and k == 128) or
        (m == 192 and n == 128 and k == 128) or
        (m == 128 and n == 128 and k == 192) or
        (m == 64 and ((n == 128 and k == 128) or
            (n == 192 and k == 128) or
            (n == 192 and k == 192) or
            (n == 256 and k == 256))) or
        (m == 96 and n == 192 and k == 128) or
        (m == 192 and n == 96 and k == 128) or
        (m == 128 and n == 192 and k == 192) or
        (m == 192 and ((n == 128 and k == 192) or
            (n == 256 and k == 192))) or
        (m == 256 and n == 192 and k == 256) or
        (n == 64 and ((m == 128 and (k == 128 or k == 256)) or
            (m == 192 and k == 192) or
            (m == 256 and (k == 128 or k == 192 or k == 256))));
}

fn shouldMaterializeCpuF64ColumnMajorGemm(m: usize, n: usize, k: usize) bool {
    return isCpuF64ColumnMajorSquareGemm(m, n, k) or
        isCpuF64ColumnMajorMeasuredRectGemm(m, n, k);
}

fn shouldMaterializeCpuF64ColumnMajorGemmAdd(m: usize, n: usize, k: usize) bool {
    return shouldMaterializeCpuF64ColumnMajorGemm(m, n, k) or
        (m == 512 and n == 512 and k == 128);
}

pub fn cpuMatmulColumnMajorResult(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (T != f32 and T != f64) return null;
    if (!lhs.device.isCpu() or !rhs.device.isCpu()) return null;
    if (lhs.shape.len != 2 or rhs.shape.len != 2) return null;
    if (lhs.shape[1] != rhs.shape[0]) return error.ShapeMismatch;
    if (!lhs.isContiguous() or !rhs.isContiguous()) return null;

    const m = lhs.shape[0];
    const k = lhs.shape[1];
    const n = rhs.shape[1];
    const element_count = std.math.mul(usize, m, n) catch return error.InvalidShape;
    const values = try lhs.allocator.alloc(T, element_count);
    errdefer lhs.allocator.free(values);
    const shape = try lhs.allocator.dupe(usize, &.{ m, n });
    errdefer lhs.allocator.free(shape);
    const strides = try lhs.allocator.dupe(usize, &.{ @as(usize, 1), m });
    errdefer lhs.allocator.free(strides);

    var out = array_mod.Array(T){
        .allocator = lhs.allocator,
        .data = values,
        .shape = shape,
        .strides = strides,
        .device = .cpu,
    };
    errdefer out.deinit();

    const lhs_view = veyra.MatrixView(T).fromSlice(lhs.data, m, k, .row_major) catch return null;
    const rhs_view = veyra.MatrixView(T).fromSlice(rhs.data, k, n, .row_major) catch return null;
    const out_view = veyra.MatrixMut(T).fromSlice(out.data, m, n, .column_major) catch return null;

    if (T == f32) {
        const workspace = getCachedF32GemmWorkspace() catch return null;
        if (veyra.gemmF32AppleAmxFullPrepackedColumnMajorCandidateWithWorkspace(lhs_view, rhs_view, out_view, .{}, workspace)) {
            return out;
        } else |_| {}
        const mt_workspace = getCachedF32MtGemmWorkspace() catch {
            veyra.gemmF32WithWorkspace(lhs_view, rhs_view, out_view, .{}, workspace) catch return null;
            return out;
        };
        veyra.gemmThreadedWithWorkspace(f32, lhs_view, rhs_view, out_view, .{}, mt_workspace) catch {
            veyra.gemmF32WithWorkspace(lhs_view, rhs_view, out_view, .{}, workspace) catch return null;
        };
    } else {
        const amx_workspace = getCachedF64AmxGemmWorkspace(m, n, k) catch null;
        if (amx_workspace) |workspace| {
            if (veyra.gemmF64AppleAmxFullPrepackedColumnMajorCandidateWithWorkspace(lhs_view, rhs_view, out_view, .{}, workspace)) {
                return out;
            } else |_| {}
        }
        const mt_workspace = getCachedF64MtGemmWorkspace() catch return null;
        veyra.ensureGemmF64MtAppleAmxWorkspace(mt_workspace, m, n, k) catch {};
        veyra.gemmThreadedWithWorkspace(f64, lhs_view, rhs_view, out_view, .{}, mt_workspace) catch return null;
    }
    return out;
}

fn restoreCpuGemmDestination(comptime T: type, out: []T, addend: ?array_mod.Array(T), beta: f32) void {
    if (beta == 0) return;
    if (addend) |c| {
        @memcpy(out, c.data);
    } else {
        @memset(out, 0);
    }
}

fn executeCpuGemmScaledTarget(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T), addend: array_mod.Array(T), alpha: f32, beta: f32) array_mod.ArrayError!?array_mod.Array(T) {
    if (lhs.shape.len != 2 or rhs.shape.len != 2 or addend.shape.len != 2) return null;
    const m = lhs.shape[0];
    const k = lhs.shape[1];
    const n = rhs.shape[1];
    if (T == f32 and shouldMaterializeCpuF32ColumnMajorGemm(m, n, k)) {
        var out = try array_mod.Array(T).empty(lhs.allocator, &.{ m, n });
        errdefer out.deinit();
        if (try cpuMatmulColumnMajorResult(T, lhs, rhs)) |column_out| {
            var materialized = column_out;
            defer materialized.deinit();
            if (materializeColumnMajorGemmAdd(T, out.data, materialized.data, addend, m, n, alpha, beta)) return out;
        }
    }
    if (T == f64 and shouldMaterializeCpuF64ColumnMajorGemmAdd(m, n, k)) {
        var out = try array_mod.Array(T).empty(lhs.allocator, &.{ m, n });
        errdefer out.deinit();
        if (try cpuMatmulColumnMajorResult(T, lhs, rhs)) |column_out| {
            var materialized = column_out;
            defer materialized.deinit();
            if (materializeColumnMajorGemmAdd(T, out.data, materialized.data, addend, m, n, alpha, beta)) return out;
        }
    }
    if (largeCpuGemm(m, n, k)) return executeCpuGemmDirect(T, lhs, rhs, addend, alpha, beta);

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
    if (comptime !supportsAxiomCudaMatmul(T)) return null;
    if (lhs.shape.len == 1 and rhs.shape.len == 1) return try axiom_cuda.tryDeviceDot(T, lhs, rhs);
    if (lhs.shape.len >= 2 and rhs.shape.len == 1) return try axiom_cuda.tryDeviceMatvec(T, lhs, rhs);
    if (lhs.shape.len == 1 and rhs.shape.len >= 2) return try axiom_cuda.tryDeviceVecmat(T, lhs, rhs);
    if (lhs.shape.len == 2 and rhs.shape.len == 2) {
        if (lhs.device.isCuda()) return try axiom_cuda.tryDeviceMatmul(T, lhs, rhs);
        return executeCudaHostMatmul2d(T, lhs, rhs);
    }
    return null;
}

fn executeCudaHostMatmul2d(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (!lhs.device.isCpu()) return null;
    if (T == f32) {
        if (try axiom_cuda.tryMatmulF32(@as(array_mod.Array(f32), lhs), @as(array_mod.Array(f32), rhs))) |out| return @as(array_mod.Array(T), out);
    } else if (T == f16) {
        if (try axiom_cuda.tryMatmulF16(@as(array_mod.Array(f16), lhs), @as(array_mod.Array(f16), rhs))) |out| return @as(array_mod.Array(T), out);
    } else if (T == array_mod.BFloat16) {
        if (try axiom_cuda.tryMatmulBF16(@as(array_mod.Array(array_mod.BFloat16), lhs), @as(array_mod.Array(array_mod.BFloat16), rhs))) |out| return @as(array_mod.Array(T), out);
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
        if (lhs.shape.len > 3 and rhs.shape.len == 1) {
            return executeMpsFlattenedBatchedMatvec(T, lhs, rhs);
        }
        if (lhs.shape.len == 1 and rhs.shape.len > 3) {
            return executeMpsFlattenedBatchedVecmat(T, lhs, rhs);
        }
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

fn executeMpsFlattenedBatchedMatvec(comptime T: type, matrix: array_mod.Array(T), vector: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (T != f32 and T != f16 and T != array_mod.BFloat16) return null;
    if (!matrix.device.isMps() or !vector.device.isMps() or !matrix.device.sameDevice(vector.device)) return null;
    if (matrix.shape.len <= 3 or vector.shape.len != 1 or !matrix.isContiguous() or !vector.isContiguous()) return null;
    const batch_shape = matrix.shape[0 .. matrix.shape.len - 2];
    const m = matrix.shape[matrix.shape.len - 2];
    const k = matrix.shape[matrix.shape.len - 1];
    if (vector.shape[0] != k) return null;
    var batch_count: usize = 1;
    for (batch_shape) |extent| batch_count = std.math.mul(usize, batch_count, extent) catch return error.InvalidShape;

    const matrix_3d_shape = [_]usize{ batch_count, m, k };
    var matrix_3d = try matrix.reshape(&matrix_3d_shape);
    defer matrix_3d.deinit();
    var out_2d = (try executeMpsBatchedMatvec(T, matrix_3d, vector)) orelse return null;
    defer out_2d.deinit();

    var out_shape = try matrix.allocator.alloc(usize, batch_shape.len + 1);
    defer matrix.allocator.free(out_shape);
    @memcpy(out_shape[0..batch_shape.len], batch_shape);
    out_shape[batch_shape.len] = m;
    return try out_2d.reshape(out_shape);
}

fn executeMpsFlattenedBatchedVecmat(comptime T: type, vector: array_mod.Array(T), matrix: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (T != f32 and T != f16 and T != array_mod.BFloat16) return null;
    if (!vector.device.isMps() or !matrix.device.isMps() or !vector.device.sameDevice(matrix.device)) return null;
    if (vector.shape.len != 1 or matrix.shape.len <= 3 or !vector.isContiguous() or !matrix.isContiguous()) return null;
    const batch_shape = matrix.shape[0 .. matrix.shape.len - 2];
    const k = matrix.shape[matrix.shape.len - 2];
    const n = matrix.shape[matrix.shape.len - 1];
    if (vector.shape[0] != k) return null;
    var batch_count: usize = 1;
    for (batch_shape) |extent| batch_count = std.math.mul(usize, batch_count, extent) catch return error.InvalidShape;

    const matrix_3d_shape = [_]usize{ batch_count, k, n };
    var matrix_3d = try matrix.reshape(&matrix_3d_shape);
    defer matrix_3d.deinit();
    var out_2d = (try executeMpsBatchedVecmat(T, vector, matrix_3d)) orelse return null;
    defer out_2d.deinit();

    var out_shape = try matrix.allocator.alloc(usize, batch_shape.len + 1);
    defer matrix.allocator.free(out_shape);
    @memcpy(out_shape[0..batch_shape.len], batch_shape);
    out_shape[batch_shape.len] = n;
    return try out_2d.reshape(out_shape);
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
    if (comptime !supportsAxiomCudaMatmul(T)) return null;
    if (try axiom_cuda.tryDeviceBmm(T, lhs, rhs)) |out| return out;
    return try axiom_cuda.tryDeviceBatchedMatmul(T, lhs, rhs);
}

fn executeMpsBmm(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (try executeMpsRank4BroadcastBmm(T, lhs, rhs)) |out| return out;
    if (try executeMpsBroadcastBatchBmm(T, lhs, rhs)) |out| return out;
    if (try executeMpsFlattenedEqualBatchBmm(T, lhs, rhs)) |out| return out;
    if (try executeMpsRankedBroadcastBmm(T, lhs, rhs)) |out| return out;
    if (T == f32) {
        if (try axiom_mps.tryBmmF32(@as(array_mod.Array(f32), lhs), @as(array_mod.Array(f32), rhs))) |out| return @as(array_mod.Array(T), out);
    } else if (T == f16) {
        if (try axiom_mps.tryBmmF16(@as(array_mod.Array(f16), lhs), @as(array_mod.Array(f16), rhs))) |out| return @as(array_mod.Array(T), out);
    } else if (T == array_mod.BFloat16) {
        if (try axiom_mps.tryBmmBF16(@as(array_mod.Array(array_mod.BFloat16), lhs), @as(array_mod.Array(array_mod.BFloat16), rhs))) |out| return @as(array_mod.Array(T), out);
    }
    return null;
}

fn executeMpsRankedBroadcastBmm(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (T != f32 and T != f16 and T != array_mod.BFloat16) return null;
    if (!lhs.device.isMps() or !rhs.device.isMps() or !lhs.device.sameDevice(rhs.device)) return null;
    if (lhs.shape.len < 5 or rhs.shape.len < 5 or !lhs.isContiguous() or !rhs.isContiguous()) return null;
    if (T == f32) {
        if (try axiom_mps.tryRankedBroadcastBmmF32(@as(array_mod.Array(f32), lhs), @as(array_mod.Array(f32), rhs))) |out| return @as(array_mod.Array(T), out);
    } else if (T == f16) {
        if (try axiom_mps.tryRankedBroadcastBmmF16(@as(array_mod.Array(f16), lhs), @as(array_mod.Array(f16), rhs))) |out| return @as(array_mod.Array(T), out);
    } else if (T == array_mod.BFloat16) {
        if (try axiom_mps.tryRankedBroadcastBmmBF16(@as(array_mod.Array(array_mod.BFloat16), lhs), @as(array_mod.Array(array_mod.BFloat16), rhs))) |out| return @as(array_mod.Array(T), out);
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
        return axiom_cuda.runPendingMatmulAdd(T, allocator, device, m, n, k, pending.lhs_storage.ptr, pending.rhs_storage.ptr, add_storage.ptr, out_ptr, pending.alpha, pending.beta);
    }
    return axiom_cuda.runPendingMatmul(T, allocator, device, m, n, k, pending.lhs_storage.ptr, pending.rhs_storage.ptr, out_ptr);
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
    if (comptime !supportsAxiomCudaMatmul(T)) return null;
    return try axiom_cuda.tryDeviceMatmulAdd(T, lhs, rhs, addend);
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
    if (try executeCpuUnaryFastPath(T, op, input)) |out| return out;

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

fn executeCpuUnaryFastPath(comptime T: type, op: ExecutionUnaryOp, input: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (T != f32 and T != f64) return null;
    if (!input.device.isCpu() or input.data.len < cpu_unary_fast_path_min_elements or !input.isContiguous()) return null;

    // Axiom CPU unary reports intentionally hash and verify outputs for smoke
    // evidence.  That is useful for small diagnostic runs, but on production
    // CPU arrays (for example large GEMM+add+exp examples) it means another
    // full output scan and, for transcendental ops, often a second expensive
    // math evaluation.  Keep small arrays on the evidence-rich Axiom path while
    // letting large materialized CPU arrays use the same typed operation
    // directly without report generation.
    var out = try array_mod.Array(T).empty(input.allocator, input.shape);
    errdefer out.deinit();
    if (T == f32)
        cpuUnarySimd(f32, 8, op, out.data, input.data)
    else
        cpuUnarySimd(f64, 4, op, out.data, input.data);
    return out;
}

fn cpuUnarySimd(comptime T: type, comptime lanes: usize, op: ExecutionUnaryOp, out: []T, input: []const T) void {
    switch (op) {
        .abs => cpuUnarySimdOp(T, lanes, .abs, out, input),
        .square => cpuUnarySimdOp(T, lanes, .square, out, input),
        .sqrt => cpuUnarySimdOp(T, lanes, .sqrt, out, input),
        .exp => cpuUnarySimdOp(T, lanes, .exp, out, input),
        .log => cpuUnarySimdOp(T, lanes, .log, out, input),
        .exp2 => cpuUnarySimdOp(T, lanes, .exp2, out, input),
        // Zig exposes vector builtins for the common transcendental ops, but
        // not for the accuracy-sensitive `expm1/log1p` pair or inverse trig.
        // Keep those scalar instead of replacing them with less accurate
        // identities such as `exp(x) - 1` or `log(1 + x)`.
        .expm1, .log1p, .asin, .acos, .atan => cpuUnaryScalar(T, out, input, op),
        .log2 => cpuUnarySimdOp(T, lanes, .log2, out, input),
        .log10 => cpuUnarySimdOp(T, lanes, .log10, out, input),
        .sin => cpuUnarySimdOp(T, lanes, .sin, out, input),
        .cos => cpuUnarySimdOp(T, lanes, .cos, out, input),
        .tan => cpuUnarySimdOp(T, lanes, .tan, out, input),
    }
}

fn cpuUnaryScalar(comptime T: type, out: []T, input: []const T, op: ExecutionUnaryOp) void {
    for (input, out) |value, *slot| {
        slot.* = cpuUnaryValue(T, op, value);
    }
}

fn cpuUnarySimdOp(
    comptime T: type,
    comptime lanes: usize,
    comptime op: ExecutionUnaryOp,
    out: []T,
    input: []const T,
) void {
    const Vec = @Vector(lanes, T);
    var i: usize = 0;
    while (i + lanes * 2 <= out.len) : (i += lanes * 2) {
        const value0: Vec = input[i..][0..lanes].*;
        const value1: Vec = input[i + lanes ..][0..lanes].*;
        out[i..][0..lanes].* = vectorUnaryValue(T, lanes, op, value0);
        out[i + lanes ..][0..lanes].* = vectorUnaryValue(T, lanes, op, value1);
    }
    while (i + lanes <= out.len) : (i += lanes) {
        const value: Vec = input[i..][0..lanes].*;
        out[i..][0..lanes].* = vectorUnaryValue(T, lanes, op, value);
    }
    while (i < out.len) : (i += 1) {
        out[i] = cpuUnaryValue(T, op, input[i]);
    }
}

fn cpuUnaryValue(comptime T: type, op: ExecutionUnaryOp, value: T) T {
    return switch (op) {
        .abs => @abs(value),
        .square => value * value,
        .sqrt => std.math.sqrt(value),
        .exp => std.math.exp(value),
        .log => std.math.log(T, std.math.e, value),
        .exp2 => std.math.exp2(value),
        .expm1 => std.math.expm1(value),
        .log1p => std.math.log1p(value),
        .log2 => std.math.log2(value),
        .log10 => std.math.log10(value),
        .sin => std.math.sin(value),
        .cos => std.math.cos(value),
        .tan => std.math.tan(value),
        .asin => std.math.asin(value),
        .acos => std.math.acos(value),
        .atan => std.math.atan(value),
    };
}

fn vectorUnaryValue(
    comptime T: type,
    comptime lanes: usize,
    comptime op: ExecutionUnaryOp,
    value: @Vector(lanes, T),
) @Vector(lanes, T) {
    return switch (op) {
        .abs => @abs(value),
        .square => value * value,
        .sqrt => @sqrt(value),
        .exp => @exp(value),
        .log => @log(value),
        .exp2 => @exp2(value),
        .log2 => @log2(value),
        .log10 => @log10(value),
        .sin => @sin(value),
        .cos => @cos(value),
        .tan => @tan(value),
        .expm1, .log1p, .asin, .acos, .atan => @compileError("vectorUnaryValue does not support this op"),
    };
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
        .square, .asin, .acos, .atan => return null,
    };
    if (!axiom_cuda.unaryElementSupported(T, cuda_op)) return null;
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
    if (try executeCpuReductionFastPath(T, op, input, axis, keepdims)) |out| return out;

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

fn executeCpuReductionFastPath(
    comptime T: type,
    op: DialectReductionOp,
    input: array_mod.Array(T),
    axis: u1,
    keepdims: bool,
) array_mod.ArrayError!?array_mod.Array(T) {
    if (T != f32 and T != f64) return null;
    if (!input.device.isCpu() or input.data.len < cpu_streaming_fast_path_min_elements) return null;
    if (input.shape.len != 2 or !input.isContiguous()) return null;
    const rows = input.shape[0];
    const cols = input.shape[1];
    if (rows == 0 or cols == 0) return null;

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
    var out = try array_mod.Array(T).empty(input.allocator, out_shape);
    errdefer out.deinit();

    if (axis == 0) {
        if (T == f32)
            cpuReductionColumns(f32, 8, op, out.data, input.data, rows, cols)
        else
            cpuReductionColumns(f64, 4, op, out.data, input.data, rows, cols);
    } else {
        if (T == f32)
            cpuReductionRows(f32, 8, op, out.data, input.data, rows, cols)
        else
            cpuReductionRows(f64, 4, op, out.data, input.data, rows, cols);
    }
    return out;
}

fn cpuReductionRows(
    comptime T: type,
    comptime lanes: usize,
    op: DialectReductionOp,
    out: []T,
    input: []const T,
    rows: usize,
    cols: usize,
) void {
    switch (op) {
        .sum => cpuReductionRowsOp(T, lanes, .sum, out, input, rows, cols),
        .prod => cpuReductionRowsOp(T, lanes, .prod, out, input, rows, cols),
        .min => cpuReductionRowsOp(T, lanes, .min, out, input, rows, cols),
        .max => cpuReductionRowsOp(T, lanes, .max, out, input, rows, cols),
    }
}

fn cpuReductionRowsOp(
    comptime T: type,
    comptime lanes: usize,
    comptime op: DialectReductionOp,
    out: []T,
    input: []const T,
    rows: usize,
    cols: usize,
) void {
    const Vec = @Vector(lanes, T);
    var row: usize = 0;
    while (row < rows) : (row += 1) {
        const row_values = input[row * cols ..][0..cols];
        var i: usize = 0;
        var acc0: Vec = @splat(reductionIdentity(T, op));
        var acc1: Vec = @splat(reductionIdentity(T, op));
        while (i + lanes * 2 <= cols) : (i += lanes * 2) {
            const v0: Vec = row_values[i..][0..lanes].*;
            const v1: Vec = row_values[i + lanes ..][0..lanes].*;
            acc0 = vectorReductionCombine(T, lanes, op, acc0, v0);
            acc1 = vectorReductionCombine(T, lanes, op, acc1, v1);
        }
        while (i + lanes <= cols) : (i += lanes) {
            const v: Vec = row_values[i..][0..lanes].*;
            acc0 = vectorReductionCombine(T, lanes, op, acc0, v);
        }

        var acc = reductionIdentity(T, op);
        inline for (0..lanes) |lane| {
            acc = reductionCombine(T, op, acc, acc0[lane]);
            acc = reductionCombine(T, op, acc, acc1[lane]);
        }
        while (i < cols) : (i += 1) {
            acc = reductionCombine(T, op, acc, row_values[i]);
        }
        out[row] = acc;
    }
}

fn cpuReductionColumns(
    comptime T: type,
    comptime lanes: usize,
    op: DialectReductionOp,
    out: []T,
    input: []const T,
    rows: usize,
    cols: usize,
) void {
    switch (op) {
        .sum => cpuReductionColumnsOp(T, lanes, .sum, out, input, rows, cols),
        .prod => cpuReductionColumnsOp(T, lanes, .prod, out, input, rows, cols),
        .min => cpuReductionColumnsOp(T, lanes, .min, out, input, rows, cols),
        .max => cpuReductionColumnsOp(T, lanes, .max, out, input, rows, cols),
    }
}

fn cpuReductionColumnsOp(
    comptime T: type,
    comptime lanes: usize,
    comptime op: DialectReductionOp,
    out: []T,
    input: []const T,
    rows: usize,
    cols: usize,
) void {
    const Vec = @Vector(lanes, T);
    var col: usize = 0;
    while (col + lanes * 2 <= cols) : (col += lanes * 2) {
        var acc0: Vec = @splat(reductionIdentity(T, op));
        var acc1: Vec = @splat(reductionIdentity(T, op));
        var row: usize = 0;
        while (row < rows) : (row += 1) {
            const row_values = input[row * cols ..][0..cols];
            const v0: Vec = row_values[col..][0..lanes].*;
            const v1: Vec = row_values[col + lanes ..][0..lanes].*;
            acc0 = vectorReductionCombine(T, lanes, op, acc0, v0);
            acc1 = vectorReductionCombine(T, lanes, op, acc1, v1);
        }
        out[col..][0..lanes].* = acc0;
        out[col + lanes ..][0..lanes].* = acc1;
    }
    while (col + lanes <= cols) : (col += lanes) {
        var acc: Vec = @splat(reductionIdentity(T, op));
        var row: usize = 0;
        while (row < rows) : (row += 1) {
            const row_values = input[row * cols ..][0..cols];
            const value: Vec = row_values[col..][0..lanes].*;
            acc = vectorReductionCombine(T, lanes, op, acc, value);
        }
        out[col..][0..lanes].* = acc;
    }
    while (col < cols) : (col += 1) {
        var acc = reductionIdentity(T, op);
        var row: usize = 0;
        while (row < rows) : (row += 1) {
            acc = reductionCombine(T, op, acc, input[row * cols + col]);
        }
        out[col] = acc;
    }
}

fn reductionIdentity(comptime T: type, comptime op: DialectReductionOp) T {
    return switch (op) {
        .sum => 0,
        .prod => 1,
        .min => std.math.inf(T),
        .max => -std.math.inf(T),
    };
}

fn reductionCombine(comptime T: type, comptime op: DialectReductionOp, lhs: T, rhs: T) T {
    return switch (op) {
        .sum => lhs + rhs,
        .prod => lhs * rhs,
        .min => @min(lhs, rhs),
        .max => @max(lhs, rhs),
    };
}

fn vectorReductionCombine(
    comptime T: type,
    comptime lanes: usize,
    comptime op: DialectReductionOp,
    lhs: @Vector(lanes, T),
    rhs: @Vector(lanes, T),
) @Vector(lanes, T) {
    return switch (op) {
        .sum => lhs + rhs,
        .prod => lhs * rhs,
        .min => @min(lhs, rhs),
        .max => @max(lhs, rhs),
    };
}

fn executeCudaLogSoftmax(comptime T: type, input: array_mod.Array(T), axis: u1) array_mod.ArrayError!?array_mod.Array(T) {
    if (comptime !supportsAxiomCudaElementwise(T)) return null;
    return try axiom_cuda.tryDeviceLogSoftmax(T, input, axis);
}

fn executeCudaSoftmax(comptime T: type, input: array_mod.Array(T), axis: u1) array_mod.ArrayError!?array_mod.Array(T) {
    if (comptime !supportsAxiomCudaElementwise(T)) return null;
    return try axiom_cuda.tryDeviceSoftmax(T, input, axis);
}

fn executeMpsLogSoftmax(comptime T: type, input: array_mod.Array(T), axis: u1) array_mod.ArrayError!?array_mod.Array(T) {
    return try axiom_mps.trySoftmax(T, .log_softmax, input, axis);
}

fn executeMpsSoftmax(comptime T: type, input: array_mod.Array(T), axis: u1) array_mod.ArrayError!?array_mod.Array(T) {
    return try axiom_mps.trySoftmax(T, .softmax, input, axis);
}

fn executeCudaReduction(
    comptime T: type,
    op: DialectReductionOp,
    input: array_mod.Array(T),
    axis: u1,
    keepdims: bool,
) array_mod.ArrayError!?array_mod.Array(T) {
    if (comptime !supportsAxiomCudaElementwise(T)) return null;
    return try axiom_cuda.tryDeviceReduction(T, op, input, axis, keepdims);
}

fn executeMpsReduction(
    comptime T: type,
    op: DialectReductionOp,
    input: array_mod.Array(T),
    axis: u1,
    keepdims: bool,
) array_mod.ArrayError!?array_mod.Array(T) {
    return try axiom_mps.tryReduction(T, mpsReductionOp(op), input, axis, keepdims);
}

fn executeCpuBroadcastBinary(comptime T: type, op: ElementwiseOp, input: array_mod.Array(T), bias: array_mod.Array(T), axis: DialectBroadcastAxis) array_mod.ArrayError!?array_mod.Array(T) {
    if (try executeCpuBroadcastFastPath(T, op, input, bias, axis)) |out| return out;
    if (op != .add) return null;

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

fn executeCpuBroadcastFastPath(comptime T: type, op: ElementwiseOp, input: array_mod.Array(T), bias: array_mod.Array(T), axis: DialectBroadcastAxis) array_mod.ArrayError!?array_mod.Array(T) {
    if (T != f32 and T != f64) return null;
    if (!input.device.isCpu() or !bias.device.isCpu()) return null;
    if (input.data.len < cpu_streaming_fast_path_min_elements) return null;
    if (input.shape.len != 2 or !input.isContiguous() or !bias.isContiguous()) return null;
    if (!broadcastBiasMatchesArrayAdd(T, input, bias, axis)) return null;

    var out = try array_mod.Array(T).empty(input.allocator, input.shape);
    errdefer out.deinit();
    if (bias.numel() == 1) {
        const scalar = bias.data[0];
        if (T == f32)
            cpuScalarElementwiseSimd(f32, 8, op, out.data, input.data, scalar, .rhs)
        else
            cpuScalarElementwiseSimd(f64, 4, op, out.data, input.data, scalar, .rhs);
        return out;
    }

    const rows = input.shape[0];
    const cols = input.shape[1];
    switch (axis) {
        .row => {
            const row_bias = if (bias.shape.len == 1) bias.data else bias.data[0..cols];
            if (T == f32)
                cpuBroadcastRowBinary(f32, 8, op, out.data, input.data, row_bias, rows, cols)
            else
                cpuBroadcastRowBinary(f64, 4, op, out.data, input.data, row_bias, rows, cols);
        },
        .column => {
            const col_bias = if (bias.shape.len == 1) bias.data else bias.data[0..rows];
            if (T == f32)
                cpuBroadcastColumnBinary(f32, 8, op, out.data, input.data, col_bias, rows, cols)
            else
                cpuBroadcastColumnBinary(f64, 4, op, out.data, input.data, col_bias, rows, cols);
        },
    }
    return out;
}

fn cpuBroadcastRowBinary(
    comptime T: type,
    comptime lanes: usize,
    op: ElementwiseOp,
    out: []T,
    input: []const T,
    bias: []const T,
    rows: usize,
    cols: usize,
) void {
    var row: usize = 0;
    while (row < rows) : (row += 1) {
        const start = row * cols;
        cpuElementwiseSimd(T, lanes, op, out[start..][0..cols], input[start..][0..cols], bias[0..cols]);
    }
}

fn cpuBroadcastColumnBinary(
    comptime T: type,
    comptime lanes: usize,
    op: ElementwiseOp,
    out: []T,
    input: []const T,
    bias: []const T,
    rows: usize,
    cols: usize,
) void {
    var row: usize = 0;
    while (row < rows) : (row += 1) {
        const start = row * cols;
        cpuScalarElementwiseSimd(T, lanes, op, out[start..][0..cols], input[start..][0..cols], bias[row], .rhs);
    }
}

fn executeCudaBroadcastAdd(comptime T: type, input: array_mod.Array(T), bias: array_mod.Array(T), axis: DialectBroadcastAxis) array_mod.ArrayError!?array_mod.Array(T) {
    return executeCudaBroadcastBinary(T, .add, input, bias, axis);
}

fn executeCudaBroadcastBinary(comptime T: type, op: ElementwiseOp, input: array_mod.Array(T), bias: array_mod.Array(T), axis: DialectBroadcastAxis) array_mod.ArrayError!?array_mod.Array(T) {
    if (comptime !supportsAxiomCudaElementwise(T)) return null;
    return try axiom_cuda.tryDeviceBroadcastBinary(T, cudaBinaryOp(op), input, bias, axis);
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
    if (try executeCpuTransposeFastPath(T, input)) |out| return out;

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

fn executeCpuTransposeFastPath(comptime T: type, input: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (T != f32 and T != f64) return null;
    if (!input.device.isCpu() or input.data.len < cpu_streaming_fast_path_min_elements) return null;
    if (input.shape.len != 2 or !input.isContiguous()) return null;
    const rows = input.shape[0];
    const cols = input.shape[1];
    var out = try array_mod.Array(T).empty(input.allocator, &.{ cols, rows });
    errdefer out.deinit();
    cpuTransposeBlocked(T, out.data, input.data, rows, cols);
    return out;
}

fn cpuTransposeBlocked(comptime T: type, out: []T, input: []const T, rows: usize, cols: usize) void {
    if (comptime T == f32 or T == f64) {
        cpuTransposeBlocked8x4(T, out, input, rows, cols);
        return;
    }

    const block: usize = 32;
    var row0: usize = 0;
    while (row0 < rows) : (row0 += block) {
        const row_end = @min(row0 + block, rows);
        var col0: usize = 0;
        while (col0 < cols) : (col0 += block) {
            const col_end = @min(col0 + block, cols);
            var row = row0;
            while (row < row_end) : (row += 1) {
                var col = col0;
                while (col < col_end) : (col += 1) {
                    out[col * rows + row] = input[row * cols + col];
                }
            }
        }
    }
}

fn cpuTransposeBlocked8x4(comptime T: type, out: []T, input: []const T, rows: usize, cols: usize) void {
    const block: usize = 32;
    var row0: usize = 0;
    while (row0 < rows) : (row0 += block) {
        const row_end = @min(row0 + block, rows);
        var col0: usize = 0;
        while (col0 < cols) : (col0 += block) {
            const col_end = @min(col0 + block, cols);
            var row = row0;
            while (row + 8 <= row_end) : (row += 8) {
                var col = col0;
                while (col + 4 <= col_end) : (col += 4) {
                    const r0a: @Vector(4, T) = input[(row + 0) * cols + col ..][0..4].*;
                    const r1a: @Vector(4, T) = input[(row + 1) * cols + col ..][0..4].*;
                    const r2a: @Vector(4, T) = input[(row + 2) * cols + col ..][0..4].*;
                    const r3a: @Vector(4, T) = input[(row + 3) * cols + col ..][0..4].*;
                    const r4a: @Vector(4, T) = input[(row + 4) * cols + col ..][0..4].*;
                    const r5a: @Vector(4, T) = input[(row + 5) * cols + col ..][0..4].*;
                    const r6a: @Vector(4, T) = input[(row + 6) * cols + col ..][0..4].*;
                    const r7a: @Vector(4, T) = input[(row + 7) * cols + col ..][0..4].*;
                    out[(col + 0) * rows + row ..][0..4].* = .{ r0a[0], r1a[0], r2a[0], r3a[0] };
                    out[(col + 0) * rows + row + 4 ..][0..4].* = .{ r4a[0], r5a[0], r6a[0], r7a[0] };
                    out[(col + 1) * rows + row ..][0..4].* = .{ r0a[1], r1a[1], r2a[1], r3a[1] };
                    out[(col + 1) * rows + row + 4 ..][0..4].* = .{ r4a[1], r5a[1], r6a[1], r7a[1] };
                    out[(col + 2) * rows + row ..][0..4].* = .{ r0a[2], r1a[2], r2a[2], r3a[2] };
                    out[(col + 2) * rows + row + 4 ..][0..4].* = .{ r4a[2], r5a[2], r6a[2], r7a[2] };
                    out[(col + 3) * rows + row ..][0..4].* = .{ r0a[3], r1a[3], r2a[3], r3a[3] };
                    out[(col + 3) * rows + row + 4 ..][0..4].* = .{ r4a[3], r5a[3], r6a[3], r7a[3] };
                }
                while (col < col_end) : (col += 1) {
                    out[col * rows + row + 0] = input[(row + 0) * cols + col];
                    out[col * rows + row + 1] = input[(row + 1) * cols + col];
                    out[col * rows + row + 2] = input[(row + 2) * cols + col];
                    out[col * rows + row + 3] = input[(row + 3) * cols + col];
                    out[col * rows + row + 4] = input[(row + 4) * cols + col];
                    out[col * rows + row + 5] = input[(row + 5) * cols + col];
                    out[col * rows + row + 6] = input[(row + 6) * cols + col];
                    out[col * rows + row + 7] = input[(row + 7) * cols + col];
                }
            }
            while (row < row_end) : (row += 1) {
                var col = col0;
                while (col < col_end) : (col += 1) {
                    out[col * rows + row] = input[row * cols + col];
                }
            }
        }
    }
}

fn executeCudaTranspose(comptime T: type, input: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (comptime !supportsAxiomCudaElementwise(T)) return null;
    return try axiom_cuda.tryDeviceTranspose(T, input);
}

fn executeMpsTranspose(comptime T: type, input: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    return try axiom_mps.tryTranspose(T, input);
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
    if (try executeCpuElementwiseFastPath(T, op, lhs, rhs)) |out| return out;

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

fn cpuElementwiseSimd(
    comptime T: type,
    comptime lanes: usize,
    op: ElementwiseOp,
    out: []T,
    lhs: []const T,
    rhs: []const T,
) void {
    switch (op) {
        .add => cpuElementwiseSimdOp(T, lanes, .add, out, lhs, rhs),
        .sub => cpuElementwiseSimdOp(T, lanes, .sub, out, lhs, rhs),
        .mul => cpuElementwiseSimdOp(T, lanes, .mul, out, lhs, rhs),
        .div => cpuElementwiseSimdOp(T, lanes, .div, out, lhs, rhs),
    }
}

fn cpuElementwiseSimdOp(
    comptime T: type,
    comptime lanes: usize,
    comptime op: ElementwiseOp,
    out: []T,
    lhs: []const T,
    rhs: []const T,
) void {
    const Vec = @Vector(lanes, T);
    var i: usize = 0;
    while (i + lanes * 2 <= out.len) : (i += lanes * 2) {
        const lhs0: Vec = lhs[i..][0..lanes].*;
        const rhs0: Vec = rhs[i..][0..lanes].*;
        const lhs1: Vec = lhs[i + lanes ..][0..lanes].*;
        const rhs1: Vec = rhs[i + lanes ..][0..lanes].*;
        out[i..][0..lanes].* = vectorElementwiseValue(T, lanes, op, lhs0, rhs0);
        out[i + lanes ..][0..lanes].* = vectorElementwiseValue(T, lanes, op, lhs1, rhs1);
    }
    while (i + lanes <= out.len) : (i += lanes) {
        const lhs_value: Vec = lhs[i..][0..lanes].*;
        const rhs_value: Vec = rhs[i..][0..lanes].*;
        out[i..][0..lanes].* = vectorElementwiseValue(T, lanes, op, lhs_value, rhs_value);
    }
    while (i < out.len) : (i += 1) {
        out[i] = elementwiseValue(T, op, lhs[i], rhs[i]);
    }
}

fn executeCpuElementwiseFastPath(comptime T: type, op: ElementwiseOp, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (T != f32 and T != f64) return null;
    if (!lhs.device.isCpu() or !rhs.device.isCpu()) return null;
    if (lhs.data.len < cpu_streaming_fast_path_min_elements) return null;
    if (!lhs.sameShape(rhs) or !lhs.isContiguous() or !rhs.isContiguous()) return null;

    // Like large unary, same-shape CPU elementwise on production arrays should
    // not pay Axiom's report hash/verify cost.  Keep small arrays on the
    // evidence-rich path used by smokes, but let large materialized GEMM results
    // perform the single expected streaming pass for `add/sub/mul/div`.
    var out = try array_mod.Array(T).empty(lhs.allocator, lhs.shape);
    errdefer out.deinit();
    if (T == f32)
        cpuElementwiseSimd(f32, 8, op, out.data, lhs.data, rhs.data)
    else
        cpuElementwiseSimd(f64, 4, op, out.data, lhs.data, rhs.data);
    return out;
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
    if (try executeCpuScalarFastPath(T, op, input, scalar, scalar_side)) |out| return out;

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

fn executeCpuScalarFastPath(comptime T: type, op: ElementwiseOp, input: array_mod.Array(T), scalar: T, scalar_side: ScalarSide) array_mod.ArrayError!?array_mod.Array(T) {
    if (T != f32 and T != f64) return null;
    if (!input.device.isCpu() or input.data.len < cpu_streaming_fast_path_min_elements or !input.isContiguous()) return null;

    // Large scalar elementwise is another production streaming op where Axiom's
    // diagnostic report would add a full verification/hash pass.  Preserve the
    // small-array report path for tests while keeping large CPU arrays to one
    // data pass.
    var out = try array_mod.Array(T).empty(input.allocator, input.shape);
    errdefer out.deinit();
    if (T == f32)
        cpuScalarElementwiseSimd(f32, 8, op, out.data, input.data, scalar, scalar_side)
    else
        cpuScalarElementwiseSimd(f64, 4, op, out.data, input.data, scalar, scalar_side);
    return out;
}

fn cpuScalarElementwiseSimd(
    comptime T: type,
    comptime lanes: usize,
    op: ElementwiseOp,
    out: []T,
    input: []const T,
    scalar: T,
    scalar_side: ScalarSide,
) void {
    switch (scalar_side) {
        .lhs => switch (op) {
            .add => cpuScalarElementwiseSimdOp(T, lanes, .add, .lhs, out, input, scalar),
            .sub => cpuScalarElementwiseSimdOp(T, lanes, .sub, .lhs, out, input, scalar),
            .mul => cpuScalarElementwiseSimdOp(T, lanes, .mul, .lhs, out, input, scalar),
            .div => cpuScalarElementwiseSimdOp(T, lanes, .div, .lhs, out, input, scalar),
        },
        .rhs => switch (op) {
            .add => cpuScalarElementwiseSimdOp(T, lanes, .add, .rhs, out, input, scalar),
            .sub => cpuScalarElementwiseSimdOp(T, lanes, .sub, .rhs, out, input, scalar),
            .mul => cpuScalarElementwiseSimdOp(T, lanes, .mul, .rhs, out, input, scalar),
            .div => cpuScalarElementwiseSimdOp(T, lanes, .div, .rhs, out, input, scalar),
        },
    }
}

fn cpuScalarElementwiseSimdOp(
    comptime T: type,
    comptime lanes: usize,
    comptime op: ElementwiseOp,
    comptime scalar_side: ScalarSide,
    out: []T,
    input: []const T,
    scalar: T,
) void {
    const Vec = @Vector(lanes, T);
    const scalar_vec: Vec = @splat(scalar);
    var i: usize = 0;
    while (i + lanes * 2 <= out.len) : (i += lanes * 2) {
        const value0: Vec = input[i..][0..lanes].*;
        const value1: Vec = input[i + lanes ..][0..lanes].*;
        const lhs0 = if (comptime scalar_side == .lhs) scalar_vec else value0;
        const rhs0 = if (comptime scalar_side == .lhs) value0 else scalar_vec;
        const lhs1 = if (comptime scalar_side == .lhs) scalar_vec else value1;
        const rhs1 = if (comptime scalar_side == .lhs) value1 else scalar_vec;
        out[i..][0..lanes].* = vectorElementwiseValue(T, lanes, op, lhs0, rhs0);
        out[i + lanes ..][0..lanes].* = vectorElementwiseValue(T, lanes, op, lhs1, rhs1);
    }
    while (i + lanes <= out.len) : (i += lanes) {
        const value: Vec = input[i..][0..lanes].*;
        const lhs = if (comptime scalar_side == .lhs) scalar_vec else value;
        const rhs = if (comptime scalar_side == .lhs) value else scalar_vec;
        out[i..][0..lanes].* = vectorElementwiseValue(T, lanes, op, lhs, rhs);
    }
    while (i < out.len) : (i += 1) {
        const lhs = if (comptime scalar_side == .lhs) scalar else input[i];
        const rhs = if (comptime scalar_side == .lhs) input[i] else scalar;
        out[i] = elementwiseValue(T, op, lhs, rhs);
    }
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

fn hostContiguousViewSlice(comptime T: type, view: array_mod.ArrayView(T)) ?[]const T {
    if (!view.device.isCpu() or !view.isContiguous()) return null;
    const len = view.numel();
    const end_index = std.math.add(usize, view.offset, len) catch return null;
    if (end_index > view.data.len) return null;
    return view.data[view.offset..end_index];
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
    if (try executeCpuViewElementwiseFastPath(T, op, lhs, rhs)) |out| return out;

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

fn executeCpuViewElementwiseFastPath(comptime T: type, op: ElementwiseOp, lhs: array_mod.ArrayView(T), rhs: array_mod.ArrayView(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (T != f32 and T != f64) return null;
    if (!lhs.device.isCpu() or !rhs.device.isCpu() or !std.mem.eql(usize, lhs.shape, rhs.shape)) return null;
    if (lhs.numel() < cpu_streaming_fast_path_min_elements) return null;
    const lhs_slice = hostContiguousViewSlice(T, lhs) orelse return null;
    const rhs_slice = hostContiguousViewSlice(T, rhs) orelse return null;
    var out = try array_mod.Array(T).empty(lhs.allocator, lhs.shape);
    errdefer out.deinit();
    if (T == f32)
        cpuElementwiseSimd(f32, 8, op, out.data, lhs_slice, rhs_slice)
    else
        cpuElementwiseSimd(f64, 4, op, out.data, lhs_slice, rhs_slice);
    return out;
}

fn executeCpuViewUnary(comptime T: type, op: ExecutionUnaryOp, input: array_mod.ArrayView(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (try executeCpuViewUnaryFastPath(T, op, input)) |out| return out;

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

fn executeCpuViewUnaryFastPath(comptime T: type, op: ExecutionUnaryOp, input: array_mod.ArrayView(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (T != f32 and T != f64) return null;
    if (!input.device.isCpu() or input.numel() < cpu_streaming_fast_path_min_elements) return null;
    const input_slice = hostContiguousViewSlice(T, input) orelse return null;
    var out = try array_mod.Array(T).empty(input.allocator, input.shape);
    errdefer out.deinit();
    if (T == f32)
        cpuUnarySimd(f32, 8, op, out.data, input_slice)
    else
        cpuUnarySimd(f64, 4, op, out.data, input_slice);
    return out;
}

fn executeCpuViewElementwiseScalar(comptime T: type, op: ElementwiseOp, input: array_mod.ArrayView(T), scalar: T, scalar_side: ScalarSide) array_mod.ArrayError!?array_mod.Array(T) {
    if (try executeCpuViewElementwiseScalarFastPath(T, op, input, scalar, scalar_side)) |out| return out;

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

fn executeCpuViewElementwiseScalarFastPath(comptime T: type, op: ElementwiseOp, input: array_mod.ArrayView(T), scalar: T, scalar_side: ScalarSide) array_mod.ArrayError!?array_mod.Array(T) {
    if (T != f32 and T != f64) return null;
    if (!input.device.isCpu() or input.numel() < cpu_streaming_fast_path_min_elements) return null;
    const input_slice = hostContiguousViewSlice(T, input) orelse return null;
    var out = try array_mod.Array(T).empty(input.allocator, input.shape);
    errdefer out.deinit();
    if (T == f32)
        cpuScalarElementwiseSimd(f32, 8, op, out.data, input_slice, scalar, scalar_side)
    else
        cpuScalarElementwiseSimd(f64, 4, op, out.data, input_slice, scalar, scalar_side);
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

fn mpsUnaryOpBF16(op: ExecutionUnaryOp) ?axiom.accelerator.MpsUnaryOp {
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
    if (try tryMpsDeviceScalarArrayBroadcast(T, op, defaultTargetForDevice(lhs.device), lhs, rhs)) |out| return out;
    if (lhs.data.len == rhs.data.len) return null;
    if (lhs.data.len == 1 and rhs.data.len != 0 and scalarBroadcastPreservesVectorShape(lhs.shape, rhs.shape)) return try executeElementwiseScalarDefault(T, op, rhs, lhs.data[0], .lhs);
    if (rhs.data.len == 1 and lhs.data.len != 0 and scalarBroadcastPreservesVectorShape(rhs.shape, lhs.shape)) return try executeElementwiseScalarDefault(T, op, lhs, rhs.data[0], .rhs);
    return null;
}

pub fn tryElementwiseScalarBroadcast(comptime T: type, op: ElementwiseOp, policy: BackendPolicy, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    const target = policyExecutionTarget(policy);
    if (try tryCudaDeviceScalarArrayBroadcast(T, op, target, lhs, rhs)) |out| return out;
    if (try tryMpsDeviceScalarArrayBroadcast(T, op, target, lhs, rhs)) |out| return out;
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

fn tryMpsDeviceScalarArrayBroadcast(comptime T: type, op: ElementwiseOp, target: DialectBackend, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (target != .mps or !lhs.device.sameDevice(rhs.device) or !lhs.device.isMps()) return null;
    if (T != f32 and T != f16 and T != array_mod.BFloat16) return null;
    const lhs_scalar = lhs.numel() == 1;
    const rhs_scalar = rhs.numel() == 1;
    if (lhs_scalar == rhs_scalar) return null;
    const scalar_left = lhs_scalar;
    const vector = if (scalar_left) rhs else lhs;
    const scalar_array = if (scalar_left) lhs else rhs;
    if (vector.shape.len == 0 or !vector.isContiguous() or !scalar_array.isContiguous()) return null;
    const scalar_storage = scalar_array.device_storage orelse return null;
    if (scalar_storage.len != 1) return null;
    var scalar_value_array = try scalar_array.cpu();
    defer scalar_value_array.deinit();
    if (scalar_value_array.data.len != 1) return null;
    return executeElementwiseScalar(T, op, target, vector, scalar_value_array.data[0], if (scalar_left) .lhs else .rhs);
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
            (lhs.shape.len >= 2 and rhs.shape.len == 1) or
            (lhs.shape.len == 1 and rhs.shape.len >= 2);
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
    _ = op;
    return broadcastAddRuntimeCapability(target).executable() and
        targetCanAccessDevice(target, input.device) and
        input.device.sameDevice(bias.device) and
        switch (target) {
            .cpu => supportedBroadcastAdd(T, input, bias, axis),
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

fn vectorElementwiseValue(
    comptime T: type,
    comptime lanes: usize,
    comptime op: ElementwiseOp,
    lhs: @Vector(lanes, T),
    rhs: @Vector(lanes, T),
) @Vector(lanes, T) {
    return switch (op) {
        .add => lhs + rhs,
        .sub => lhs - rhs,
        .mul => lhs * rhs,
        .div => lhs / rhs,
    };
}

test "CPU elementwise SIMD helpers cover vector blocks and tails" {
    const lhs32 = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 };
    const rhs32 = [_]f32{ 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };
    var out32: [lhs32.len]f32 = undefined;

    cpuElementwiseSimd(f32, 8, .sub, &out32, &lhs32, &rhs32);
    try std.testing.expectEqualSlices(f32, &.{ -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16 }, &out32);

    cpuScalarElementwiseSimd(f32, 8, .div, &out32, &lhs32, 2.0, .rhs);
    try std.testing.expectEqualSlices(f32, &.{ 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5 }, &out32);

    const lhs64 = [_]f64{ 2, 4, 6, 8, 10, 12, 14, 16, 18 };
    const rhs64 = [_]f64{ 1, 3, 5, 7, 9, 11, 13, 15, 17 };
    var out64: [lhs64.len]f64 = undefined;

    cpuElementwiseSimd(f64, 4, .add, &out64, &lhs64, &rhs64);
    try std.testing.expectEqualSlices(f64, &.{ 3, 7, 11, 15, 19, 23, 27, 31, 35 }, &out64);

    cpuScalarElementwiseSimd(f64, 4, .sub, &out64, &rhs64, 20.0, .lhs);
    try std.testing.expectEqualSlices(f64, &.{ 19, 17, 15, 13, 11, 9, 7, 5, 3 }, &out64);
}

test "CPU unary SIMD helper preserves vector and scalar fallback semantics" {
    const input32 = [_]f32{ 1, 4, 9, 16, 25, 36, 49, 64, 81 };
    var sqrt32: [input32.len]f32 = undefined;
    cpuUnarySimd(f32, 8, .sqrt, &sqrt32, &input32);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, &sqrt32);

    const trig_input64 = [_]f64{ 0, 0.25, 0.5, 0.75, 1.0 };
    var atan64: [trig_input64.len]f64 = undefined;
    cpuUnarySimd(f64, 4, .atan, &atan64, &trig_input64);
    for (trig_input64, atan64) |value, actual| {
        try std.testing.expectApproxEqAbs(std.math.atan(value), actual, 1e-15);
    }

    const exp_input64 = [_]f64{ 0, 1, 2, 3, 4 };
    var exp64: [exp_input64.len]f64 = undefined;
    cpuUnarySimd(f64, 4, .exp, &exp64, &exp_input64);
    for (exp_input64, exp64) |value, actual| {
        try std.testing.expectApproxEqAbs(std.math.exp(value), actual, 1e-12);
    }
}

test "CPU row reduction SIMD helper covers reduction ops and tails" {
    const rows32: usize = 2;
    const cols32: usize = 9;
    const input32 = [_]f32{
        1, 2, 3, 4, 5, 6, 7, 8, 9,
        9, 8, 7, 6, 5, 4, 3, 2, 1,
    };
    var out32: [rows32]f32 = undefined;

    cpuReductionRows(f32, 8, .sum, &out32, &input32, rows32, cols32);
    try std.testing.expectEqualSlices(f32, &.{ 45, 45 }, &out32);
    cpuReductionRows(f32, 8, .max, &out32, &input32, rows32, cols32);
    try std.testing.expectEqualSlices(f32, &.{ 9, 9 }, &out32);

    const rows64: usize = 2;
    const cols64: usize = 5;
    const input64 = [_]f64{
        1, 2, 3, 4, 5,
        5, 4, 3, 2, 1,
    };
    var out64: [rows64]f64 = undefined;

    cpuReductionRows(f64, 4, .prod, &out64, &input64, rows64, cols64);
    try std.testing.expectEqualSlices(f64, &.{ 120, 120 }, &out64);
    cpuReductionRows(f64, 4, .min, &out64, &input64, rows64, cols64);
    try std.testing.expectEqualSlices(f64, &.{ 1, 1 }, &out64);
}

test "CPU column reduction SIMD helper covers vector blocks and tails" {
    const rows32: usize = 3;
    const cols32: usize = 9;
    const input32 = [_]f32{
        1,   2,   3,   4,   5,   6,   7,   8,   9,
        10,  20,  30,  40,  50,  60,  70,  80,  90,
        100, 200, 300, 400, 500, 600, 700, 800, 900,
    };
    var out32: [cols32]f32 = undefined;

    cpuReductionColumns(f32, 8, .sum, &out32, &input32, rows32, cols32);
    try std.testing.expectEqualSlices(f32, &.{ 111, 222, 333, 444, 555, 666, 777, 888, 999 }, &out32);
    cpuReductionColumns(f32, 8, .max, &out32, &input32, rows32, cols32);
    try std.testing.expectEqualSlices(f32, &.{ 100, 200, 300, 400, 500, 600, 700, 800, 900 }, &out32);

    const rows64: usize = 3;
    const cols64: usize = 5;
    const input64 = [_]f64{
        1, 2, 3, 4, 5,
        2, 3, 4, 5, 6,
        3, 4, 5, 6, 7,
    };
    var out64: [cols64]f64 = undefined;

    cpuReductionColumns(f64, 4, .prod, &out64, &input64, rows64, cols64);
    try std.testing.expectEqualSlices(f64, &.{ 6, 24, 60, 120, 210 }, &out64);
    cpuReductionColumns(f64, 4, .min, &out64, &input64, rows64, cols64);
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 3, 4, 5 }, &out64);
}

test "CPU broadcast add SIMD helpers cover row column and scalar bias" {
    const rows32: usize = 2;
    const cols32: usize = 9;
    const input32 = [_]f32{
        1,  2,  3,  4,  5,  6,  7,  8,  9,
        10, 20, 30, 40, 50, 60, 70, 80, 90,
    };
    const row_bias32 = [_]f32{ 100, 200, 300, 400, 500, 600, 700, 800, 900 };
    const col_bias32 = [_]f32{ 1000, 2000 };
    var out32: [input32.len]f32 = undefined;

    cpuBroadcastRowBinary(f32, 8, .add, &out32, &input32, &row_bias32, rows32, cols32);
    try std.testing.expectEqualSlices(f32, &.{
        101, 202, 303, 404, 505, 606, 707, 808, 909,
        110, 220, 330, 440, 550, 660, 770, 880, 990,
    }, &out32);
    cpuBroadcastRowBinary(f32, 8, .sub, &out32, &input32, &row_bias32, rows32, cols32);
    try std.testing.expectEqualSlices(f32, &.{
        -99, -198, -297, -396, -495, -594, -693, -792, -891,
        -90, -180, -270, -360, -450, -540, -630, -720, -810,
    }, &out32);

    cpuBroadcastColumnBinary(f32, 8, .add, &out32, &input32, &col_bias32, rows32, cols32);
    try std.testing.expectEqualSlices(f32, &.{
        1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009,
        2010, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090,
    }, &out32);
    cpuBroadcastColumnBinary(f32, 8, .mul, &out32, &input32, &col_bias32, rows32, cols32);
    try std.testing.expectEqualSlices(f32, &.{
        1000,  2000,  3000,  4000,  5000,   6000,   7000,   8000,   9000,
        20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000,
    }, &out32);

    const input64 = [_]f64{
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
    };
    var out64: [input64.len]f64 = undefined;
    cpuScalarElementwiseSimd(f64, 4, .add, &out64, &input64, 0.5, .rhs);
    try std.testing.expectEqualSlices(f64, &.{ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5 }, &out64);
}

test "CPU blocked transpose helper handles non-square tails" {
    const rows32: usize = 3;
    const cols32: usize = 5;
    const input32 = [_]f32{
        1,  2,  3,  4,  5,
        6,  7,  8,  9,  10,
        11, 12, 13, 14, 15,
    };
    var out32: [input32.len]f32 = undefined;
    cpuTransposeBlocked(f32, &out32, &input32, rows32, cols32);
    try std.testing.expectEqualSlices(f32, &.{
        1, 6,  11,
        2, 7,  12,
        3, 8,  13,
        4, 9,  14,
        5, 10, 15,
    }, &out32);

    const rows64: usize = 2;
    const cols64: usize = 4;
    const input64 = [_]f64{
        1, 2, 3, 4,
        5, 6, 7, 8,
    };
    var out64: [input64.len]f64 = undefined;
    cpuTransposeBlocked(f64, &out64, &input64, rows64, cols64);
    try std.testing.expectEqualSlices(f64, &.{
        1, 5,
        2, 6,
        3, 7,
        4, 8,
    }, &out64);
}

test "CPU blocked transpose helper covers 8x4 unroll and tails" {
    const rows: usize = 9;
    const cols: usize = 7;
    var input: [rows * cols]f64 = undefined;
    var row: usize = 0;
    while (row < rows) : (row += 1) {
        var col: usize = 0;
        while (col < cols) : (col += 1) {
            input[row * cols + col] = @floatFromInt(row * 100 + col);
        }
    }

    var out: [rows * cols]f64 = undefined;
    cpuTransposeBlocked(f64, &out, &input, rows, cols);

    row = 0;
    while (row < rows) : (row += 1) {
        var col: usize = 0;
        while (col < cols) : (col += 1) {
            try std.testing.expectEqual(input[row * cols + col], out[col * rows + row]);
        }
    }
}

test "CPU vector matmul SIMD helpers cover dot matvec and vecmat tails" {
    const lhs32 = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const rhs32 = [_]f32{ 9, 8, 7, 6, 5, 4, 3, 2, 1 };
    try std.testing.expectEqual(@as(f32, 165), cpuDotSimd(f32, 8, &lhs32, &rhs32));

    const rows32: usize = 2;
    const cols32: usize = 5;
    const matrix32 = [_]f32{
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
    };
    const vector32 = [_]f32{ 1, 10, 100, 1000, 10000 };
    var matvec32: [rows32]f32 = undefined;
    cpuMatvecSimd(f32, 8, &matvec32, &matrix32, &vector32, rows32, cols32);
    try std.testing.expectEqualSlices(f32, &.{ 54321, 109876 }, &matvec32);

    const rows64: usize = 3;
    const cols64: usize = 5;
    const vector64 = [_]f64{ 1, 2, 3 };
    const matrix64 = [_]f64{
        1,   2,   3,   4,   5,
        10,  20,  30,  40,  50,
        100, 200, 300, 400, 500,
    };
    var vecmat64: [cols64]f64 = .{ 0, 0, 0, 0, 0 };
    cpuVecmatStreaming(f64, 4, &vecmat64, &vector64, &matrix64, rows64, cols64);
    try std.testing.expectEqualSlices(f64, &.{ 321, 642, 963, 1284, 1605 }, &vecmat64);
}

test "CPU contiguous view fast paths bypass memref report for large vectors" {
    const gpa = std.testing.allocator;
    const n = cpu_streaming_fast_path_min_elements;
    var lhs = try array_mod.Array(f32).empty(gpa, &.{n});
    defer lhs.deinit();
    var rhs = try array_mod.Array(f32).empty(gpa, &.{n});
    defer rhs.deinit();
    for (lhs.data, rhs.data, 0..) |*lhs_slot, *rhs_slot, i| {
        lhs_slot.* = @floatFromInt(i % 17);
        rhs_slot.* = @floatFromInt(i % 5);
    }

    var lhs_view = try lhs.asView();
    defer lhs_view.deinit();
    var rhs_view = try rhs.asView();
    defer rhs_view.deinit();

    var added = (try executeCpuViewElementwiseFastPath(f32, .add, lhs_view, rhs_view)) orelse return error.BackendFailure;
    defer added.deinit();
    try std.testing.expectEqual(@as(usize, n), added.data.len);
    try std.testing.expectEqual(lhs.data[13] + rhs.data[13], added.data[13]);

    var scaled = (try executeCpuViewElementwiseScalarFastPath(f32, .mul, lhs_view, 2.0, .rhs)) orelse return error.BackendFailure;
    defer scaled.deinit();
    try std.testing.expectEqual(lhs.data[26] * 2.0, scaled.data[26]);

    var rooted = (try executeCpuViewUnaryFastPath(f32, .sqrt, lhs_view)) orelse return error.BackendFailure;
    defer rooted.deinit();
    try std.testing.expectApproxEqAbs(std.math.sqrt(lhs.data[39]), rooted.data[39], 1e-6);

    var strided_source = try array_mod.Array(f32).empty(gpa, &.{n * 2});
    defer strided_source.deinit();
    var strided_view = try strided_source.sliceAxisView(0, .{ .start = 0, .stop = @intCast(n * 2), .step = 2 });
    defer strided_view.deinit();
    try std.testing.expect((try executeCpuViewUnaryFastPath(f32, .sqrt, strided_view)) == null);

    var matrix_lhs = try lhs.reshape(&.{ 8, 8 });
    defer matrix_lhs.deinit();
    var matrix_rhs = try rhs.reshape(&.{ 8, 8 });
    defer matrix_rhs.deinit();
    var matrix_lhs_view = try matrix_lhs.asView();
    defer matrix_lhs_view.deinit();
    var matrix_rhs_view = try matrix_rhs.asView();
    defer matrix_rhs_view.deinit();
    var matrix_sum = (try executeCpuViewElementwiseFastPath(f32, .add, matrix_lhs_view, matrix_rhs_view)) orelse return error.BackendFailure;
    defer matrix_sum.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 8, 8 }, matrix_sum.shape);
    try std.testing.expectEqual(matrix_lhs.data[16] + matrix_rhs.data[16], matrix_sum.data[16]);
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

test "CPU f32 128 GEMM fast path returns contiguous row-major output" {
    const gpa = std.testing.allocator;
    try checkCpuF32SquareGemmFastPath(gpa, 64);
    try checkCpuF32SquareGemmFastPath(gpa, 96);
    try checkCpuF32SquareGemmFastPath(gpa, 128);
    try checkCpuF32SquareGemmFastPath(gpa, 130);
    try checkCpuF32SquareGemmFastPath(gpa, 132);
    try checkCpuF32SquareGemmFastPath(gpa, 136);
    try checkCpuF32SquareGemmFastPath(gpa, 140);
    try checkCpuF32SquareGemmFastPath(gpa, 144);
    try checkCpuF32SquareGemmFastPath(gpa, 148);
    try checkCpuF32SquareGemmFastPath(gpa, 152);
    try checkCpuF32SquareGemmFastPath(gpa, 156);
    try checkCpuF32SquareGemmFastPath(gpa, 160);
    try checkCpuF32SquareGemmFastPath(gpa, 164);
    try checkCpuF32SquareGemmFastPath(gpa, 168);
    try checkCpuF32SquareGemmFastPath(gpa, 172);
    try checkCpuF32SquareGemmFastPath(gpa, 176);
    try checkCpuF32SquareGemmFastPath(gpa, 180);
    try checkCpuF32SquareGemmFastPath(gpa, 184);
    try checkCpuF32SquareGemmFastPath(gpa, 188);
    try checkCpuF32SquareGemmFastPath(gpa, 192);
    try checkCpuF32SquareGemmFastPath(gpa, 224);
    try checkCpuF32GemmFastPath(gpa, 10, 100, 100);
    try checkCpuF32GemmFastPath(gpa, 50, 100, 100);
    try checkCpuF32GemmFastPath(gpa, 64, 128, 128);
    try checkCpuF32GemmFastPath(gpa, 128, 64, 128);
    try checkCpuF32GemmFastPath(gpa, 128, 128, 64);
    try checkCpuF32GemmFastPath(gpa, 64, 192, 192);
    try checkCpuF32GemmFastPath(gpa, 192, 64, 192);
    try checkCpuF32GemmFastPath(gpa, 192, 192, 64);
    try checkCpuF32GemmFastPath(gpa, 64, 64, 192);
    try checkCpuF32GemmFastPath(gpa, 64, 256, 256);
    try checkCpuF32GemmFastPath(gpa, 128, 192, 192);
    try checkCpuF32GemmFastPath(gpa, 192, 256, 192);
    try checkCpuF32GemmFastPath(gpa, 256, 192, 256);
    try checkCpuF32GemmFastPath(gpa, 128, 512, 128);
    try checkCpuF32GemmFastPath(gpa, 512, 128, 512);
    try checkCpuF32GemmFastPath(gpa, 128, 64, 256);
    try checkCpuF32GemmFastPath(gpa, 256, 64, 128);
    try checkCpuF32GemmFastPath(gpa, 256, 64, 192);
    try checkCpuF32GemmFastPath(gpa, 256, 64, 256);
    try checkCpuF32GemmFastPath(gpa, 16, 128, 16);
    try checkCpuF32GemmFastPath(gpa, 16, 256, 16);
    try checkCpuF32GemmFastPath(gpa, 16, 64, 16);
    try checkCpuF32GemmFastPath(gpa, 16, 128, 32);
    try checkCpuF32GemmFastPath(gpa, 16, 256, 32);
    try checkCpuF32GemmFastPath(gpa, 32, 128, 16);
    try checkCpuF32GemmFastPath(gpa, 32, 128, 32);
    try checkCpuF32GemmFastPath(gpa, 32, 256, 16);
    try checkCpuF32GemmFastPath(gpa, 32, 256, 32);
    try checkCpuF32GemmFastPath(gpa, 128, 256, 64);
    try checkCpuF32GemmFastPath(gpa, 256, 128, 64);
    try checkCpuF32GemmFastPath(gpa, 256, 256, 64);
    try checkCpuF32GemmFastPath(gpa, 128, 256, 128);
    try checkCpuF32GemmFastPath(gpa, 192, 128, 128);
    try checkCpuF32GemmFastPath(gpa, 256, 128, 128);
    try checkCpuF32GemmFastPath(gpa, 192, 256, 128);
    try checkCpuF32GemmFastPath(gpa, 256, 192, 128);
    try checkCpuF32GemmFastPath(gpa, 256, 256, 128);
    try checkCpuF32GemmFastPath(gpa, 128, 256, 192);
    try checkCpuF32GemmFastPath(gpa, 192, 128, 192);
    try checkCpuF32GemmFastPath(gpa, 256, 128, 192);
}

fn checkCpuF32SquareGemmFastPath(gpa: std.mem.Allocator, comptime n: usize) !void {
    try checkCpuF32GemmFastPath(gpa, n, n, n);
}

fn checkCpuF32GemmFastPath(gpa: std.mem.Allocator, comptime m: usize, comptime n: usize, comptime k: usize) !void {
    var lhs = try array_mod.Array(f32).empty(gpa, &.{ m, k });
    defer lhs.deinit();
    var rhs = try array_mod.Array(f32).empty(gpa, &.{ k, n });
    defer rhs.deinit();

    var row: usize = 0;
    while (row < m) : (row += 1) {
        var col: usize = 0;
        while (col < k) : (col += 1) {
            lhs.data[row * k + col] = @as(f32, @floatFromInt(((row + 3) * (col + 5)) % 17 + 1)) * 0.03125;
        }
    }
    row = 0;
    while (row < k) : (row += 1) {
        var col: usize = 0;
        while (col < n) : (col += 1) {
            rhs.data[row * n + col] = @as(f32, @floatFromInt(((row + 7) * (col + 11)) % 19 + 1)) * -0.015625;
        }
    }

    var out = try matmul(f32, .prefer_axiom_cpu, lhs, rhs);
    defer out.deinit();
    try std.testing.expect(out.isContiguous());
    try std.testing.expectEqualSlices(usize, &.{ m, n }, out.shape);

    const checks = [_][2]usize{
        .{ 0, 0 },
        .{ @min(@as(usize, 3), m - 1), @min(@as(usize, 17), n - 1) },
        .{ m / 2, @min(@as(usize, 5), n - 1) },
        .{ m - 1, n - 1 },
    };
    for (checks) |idx| {
        var expected: f32 = 0;
        var kk: usize = 0;
        while (kk < k) : (kk += 1) {
            expected += lhs.data[idx[0] * k + kk] * rhs.data[kk * n + idx[1]];
        }
        try std.testing.expectApproxEqAbs(expected, out.data[idx[0] * n + idx[1]], 1e-4);
    }
}

test "CPU f64 100 GEMM fast path returns contiguous row-major output" {
    const gpa = std.testing.allocator;
    const n: usize = 100;
    var lhs = try array_mod.Array(f64).empty(gpa, &.{ n, n });
    defer lhs.deinit();
    var rhs = try array_mod.Array(f64).empty(gpa, &.{ n, n });
    defer rhs.deinit();

    var row: usize = 0;
    while (row < n) : (row += 1) {
        var col: usize = 0;
        while (col < n) : (col += 1) {
            lhs.data[row * n + col] = @as(f64, @floatFromInt(((row + 5) * (col + 7)) % 23 + 1)) * 0.015625;
            rhs.data[row * n + col] = @as(f64, @floatFromInt(((row + 11) * (col + 13)) % 29 + 1)) * -0.01171875;
        }
    }

    var out = try matmul(f64, .prefer_axiom_cpu, lhs, rhs);
    defer out.deinit();
    try std.testing.expect(out.isContiguous());
    try std.testing.expectEqualSlices(usize, &.{ n, n }, out.shape);

    const checks = [_][2]usize{
        .{ 0, 0 },
        .{ 3, 17 },
        .{ 64, 5 },
        .{ 99, 99 },
    };
    for (checks) |idx| {
        var expected: f64 = 0;
        var kk: usize = 0;
        while (kk < n) : (kk += 1) {
            expected += lhs.data[idx[0] * n + kk] * rhs.data[kk * n + idx[1]];
        }
        try std.testing.expectApproxEqAbs(expected, out.data[idx[0] * n + idx[1]], 1e-10);
    }
}

test "CPU f64 AMX GEMM fast path returns contiguous row-major output" {
    const gpa = std.testing.allocator;
    try checkCpuF64GemmFastPath(gpa, 16, 64, 16);
    try checkCpuF64GemmFastPath(gpa, 16, 128, 16);
    try checkCpuF64GemmFastPath(gpa, 16, 512, 16);
    try checkCpuF64GemmFastPath(gpa, 16, 1024, 16);
    try checkCpuF64GemmFastPath(gpa, 32, 64, 32);
    try checkCpuF64GemmFastPath(gpa, 32, 128, 32);
    try checkCpuF64GemmFastPath(gpa, 32, 256, 32);
    try checkCpuF64GemmFastPath(gpa, 32, 512, 32);
    try checkCpuF64GemmFastPath(gpa, 32, 1024, 32);
    try checkCpuF64GemmFastPath(gpa, 64, 64, 64);
    try checkCpuF64GemmFastPath(gpa, 64, 128, 32);
    try checkCpuF64GemmFastPath(gpa, 64, 128, 64);
    try checkCpuF64GemmFastPath(gpa, 128, 64, 64);
    try checkCpuF64GemmFastPath(gpa, 64, 128, 128);
    try checkCpuF64GemmFastPath(gpa, 64, 192, 192);
    try checkCpuF64GemmFastPath(gpa, 64, 256, 256);
    try checkCpuF64GemmFastPath(gpa, 128, 64, 128);
    try checkCpuF64GemmFastPath(gpa, 128, 64, 256);
    try checkCpuF64GemmFastPath(gpa, 128, 192, 192);
    try checkCpuF64GemmFastPath(gpa, 192, 128, 192);
    try checkCpuF64GemmFastPath(gpa, 192, 64, 192);
    try checkCpuF64GemmFastPath(gpa, 192, 256, 192);
    try checkCpuF64GemmFastPath(gpa, 256, 64, 128);
    try checkCpuF64GemmFastPath(gpa, 256, 64, 192);
    try checkCpuF64GemmFastPath(gpa, 256, 64, 256);
    try checkCpuF64GemmFastPath(gpa, 256, 192, 256);
    try checkCpuF64GemmFastPath(gpa, 100, 100, 100);
    try checkCpuF64GemmFastPath(gpa, 130, 130, 130);
    try checkCpuF64GemmFastPath(gpa, 132, 132, 132);
    try checkCpuF64GemmFastPath(gpa, 136, 136, 136);
    try checkCpuF64GemmFastPath(gpa, 140, 140, 140);
    try checkCpuF64GemmFastPath(gpa, 144, 144, 144);
    try checkCpuF64GemmFastPath(gpa, 148, 148, 148);
    try checkCpuF64GemmFastPath(gpa, 150, 150, 150);
    try checkCpuF64GemmFastPath(gpa, 152, 152, 152);
    try checkCpuF64GemmFastPath(gpa, 156, 156, 156);
    try checkCpuF64GemmFastPath(gpa, 160, 160, 160);
    try checkCpuF64GemmFastPath(gpa, 164, 164, 164);
    try checkCpuF64GemmFastPath(gpa, 168, 168, 168);
    try checkCpuF64GemmFastPath(gpa, 172, 172, 172);
    try checkCpuF64GemmFastPath(gpa, 176, 176, 176);
    try checkCpuF64GemmFastPath(gpa, 180, 180, 180);
    try checkCpuF64GemmFastPath(gpa, 184, 184, 184);
    try checkCpuF64GemmFastPath(gpa, 188, 188, 188);
    try checkCpuF64GemmFastPath(gpa, 128, 128, 128);
    try checkCpuF64GemmFastPath(gpa, 128, 128, 192);
    try checkCpuF64GemmFastPath(gpa, 192, 128, 128);
    try checkCpuF64GemmFastPath(gpa, 128, 256, 64);
    try checkCpuF64GemmFastPath(gpa, 192, 192, 64);
    try checkCpuF64GemmFastPath(gpa, 192, 256, 64);
    try checkCpuF64GemmFastPath(gpa, 256, 128, 64);
    try checkCpuF64GemmFastPath(gpa, 256, 192, 64);
    try checkCpuF64GemmFastPath(gpa, 256, 256, 64);
    try checkCpuF64GemmFastPath(gpa, 128, 256, 128);
    try checkCpuF64GemmFastPath(gpa, 256, 128, 128);
    try checkCpuF64GemmFastPath(gpa, 128, 256, 192);
    try checkCpuF64GemmFastPath(gpa, 256, 128, 192);
}

fn checkCpuF64GemmFastPath(gpa: std.mem.Allocator, comptime m: usize, comptime n: usize, comptime k: usize) !void {
    var lhs = try array_mod.Array(f64).empty(gpa, &.{ m, k });
    defer lhs.deinit();
    var rhs = try array_mod.Array(f64).empty(gpa, &.{ k, n });
    defer rhs.deinit();

    var row: usize = 0;
    while (row < m) : (row += 1) {
        var col: usize = 0;
        while (col < k) : (col += 1) {
            lhs.data[row * k + col] = @as(f64, @floatFromInt(((row + 5) * (col + 7)) % 23 + 1)) * 0.015625;
        }
    }
    row = 0;
    while (row < k) : (row += 1) {
        var col: usize = 0;
        while (col < n) : (col += 1) {
            rhs.data[row * n + col] = @as(f64, @floatFromInt(((row + 11) * (col + 13)) % 29 + 1)) * -0.01171875;
        }
    }

    var out = try matmul(f64, .prefer_axiom_cpu, lhs, rhs);
    defer out.deinit();
    try std.testing.expect(out.isContiguous());
    try std.testing.expectEqualSlices(usize, &.{ m, n }, out.shape);

    const checks = [_][2]usize{
        .{ 0, 0 },
        .{ @min(@as(usize, 3), m - 1), @min(@as(usize, 17), n - 1) },
        .{ m / 2, @min(@as(usize, 5), n - 1) },
        .{ m - 1, n - 1 },
    };
    for (checks) |idx| {
        var expected: f64 = 0;
        var kk: usize = 0;
        while (kk < k) : (kk += 1) {
            expected += lhs.data[idx[0] * k + kk] * rhs.data[kk * n + idx[1]];
        }
        try std.testing.expectApproxEqAbs(expected, out.data[idx[0] * n + idx[1]], 1e-9);
    }
}

test "CPU column-major matmul result preserves logical array order" {
    const gpa = std.testing.allocator;
    var lhs = try array_mod.Array(f32).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer lhs.deinit();
    var rhs = try array_mod.Array(f32).fromSlice(gpa, &.{ 7, 8, 9, 10, 11, 12 }, &.{ 3, 2 });
    defer rhs.deinit();

    var out = (try cpuMatmulColumnMajorResult(f32, lhs, rhs)) orelse return error.BackendFailure;
    defer out.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, out.shape);
    try std.testing.expectEqualSlices(usize, &.{ 1, 2 }, out.strides);
    try std.testing.expect(!out.isContiguous());
    try std.testing.expectEqual(@as(f32, 58), try out.get(&.{ 0, 0 }));
    try std.testing.expectEqual(@as(f32, 64), try out.get(&.{ 0, 1 }));
    try std.testing.expectEqual(@as(f32, 139), try out.get(&.{ 1, 0 }));
    try std.testing.expectEqual(@as(f32, 154), try out.get(&.{ 1, 1 }));

    var logical: [4]f32 = undefined;
    try out.copyToSlice(&logical);
    try std.testing.expectEqualSlices(f32, &.{ 58, 64, 139, 154 }, &logical);
    var cloned = try out.clone();
    defer cloned.deinit();
    try std.testing.expect(cloned.isContiguous());
    try std.testing.expectEqualSlices(f32, &.{ 58, 64, 139, 154 }, cloned.data);
}

test "CPU column-major matmul result materializes efficiently" {
    const gpa = std.testing.allocator;
    var lhs = try array_mod.Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer lhs.deinit();
    var rhs = try array_mod.Array(f64).fromSlice(gpa, &.{ 7, 8, 9, 10, 11, 12 }, &.{ 3, 2 });
    defer rhs.deinit();

    var out = (try cpuMatmulColumnMajorResult(f64, lhs, rhs)) orelse return error.BackendFailure;
    defer out.deinit();
    var materialized: [4]f64 = undefined;
    try out.copyToSlice(&materialized);
    try std.testing.expectEqualSlices(f64, &.{ 58, 64, 139, 154 }, &materialized);

    var view = try out.asView();
    defer view.deinit();
    var contiguous = try view.toArray();
    defer contiguous.deinit();
    try std.testing.expect(contiguous.isContiguous());
    try std.testing.expectEqualSlices(f64, &.{ 58, 64, 139, 154 }, contiguous.data);
}

test "CPU f32 matmulAdd uses column-major materialization shapes" {
    const gpa = std.testing.allocator;
    const m: usize = 128;
    const n: usize = 384;
    const k: usize = 128;
    var lhs = try array_mod.Array(f32).empty(gpa, &.{ m, k });
    defer lhs.deinit();
    var rhs = try array_mod.Array(f32).empty(gpa, &.{ k, n });
    defer rhs.deinit();
    var addend = try array_mod.Array(f32).empty(gpa, &.{ m, n });
    defer addend.deinit();

    var row: usize = 0;
    while (row < m) : (row += 1) {
        var col: usize = 0;
        while (col < k) : (col += 1) {
            lhs.data[row * k + col] = @as(f32, @floatFromInt(((row + 3) * (col + 5)) % 17 + 1)) * 0.03125;
        }
    }
    row = 0;
    while (row < k) : (row += 1) {
        var col: usize = 0;
        while (col < n) : (col += 1) {
            rhs.data[row * n + col] = @as(f32, @floatFromInt(((row + 7) * (col + 11)) % 19 + 1)) * -0.015625;
        }
    }
    row = 0;
    while (row < m) : (row += 1) {
        var col: usize = 0;
        while (col < n) : (col += 1) {
            addend.data[row * n + col] = @as(f32, @floatFromInt(((row + 13) * (col + 17)) % 23 + 1)) * 0.0078125;
        }
    }

    var product = try matmul(f32, .prefer_axiom_cpu, lhs, rhs);
    defer product.deinit();
    var fused = (try executeCpuGemmScaledTarget(f32, lhs, rhs, addend, 1.0, 1.0)) orelse return error.BackendFailure;
    defer fused.deinit();
    try std.testing.expect(fused.isContiguous());
    try std.testing.expectEqualSlices(usize, &.{ m, n }, fused.shape);

    const checks = [_][2]usize{
        .{ 0, 0 },
        .{ 3, 17 },
        .{ m / 2, 5 },
        .{ m - 1, n - 1 },
    };
    for (checks) |idx| {
        const index = idx[0] * n + idx[1];
        try std.testing.expectApproxEqAbs(product.data[index] + addend.data[index], fused.data[index], 1e-4);
    }
}

test "CPU f64 matmulAdd uses low-K column-major materialization shapes" {
    const gpa = std.testing.allocator;
    const m: usize = 384;
    const n: usize = 128;
    const k: usize = 128;
    var lhs = try array_mod.Array(f64).empty(gpa, &.{ m, k });
    defer lhs.deinit();
    var rhs = try array_mod.Array(f64).empty(gpa, &.{ k, n });
    defer rhs.deinit();
    var addend = try array_mod.Array(f64).empty(gpa, &.{ m, n });
    defer addend.deinit();

    var row: usize = 0;
    while (row < m) : (row += 1) {
        var col: usize = 0;
        while (col < k) : (col += 1) {
            lhs.data[row * k + col] = @as(f64, @floatFromInt(((row + 3) * (col + 5)) % 17 + 1)) * 0.015625;
        }
    }
    row = 0;
    while (row < k) : (row += 1) {
        var col: usize = 0;
        while (col < n) : (col += 1) {
            rhs.data[row * n + col] = @as(f64, @floatFromInt(((row + 7) * (col + 11)) % 19 + 1)) * -0.0078125;
        }
    }
    row = 0;
    while (row < m) : (row += 1) {
        var col: usize = 0;
        while (col < n) : (col += 1) {
            addend.data[row * n + col] = @as(f64, @floatFromInt(((row + 13) * (col + 17)) % 23 + 1)) * 0.00390625;
        }
    }

    var product = try matmul(f64, .prefer_axiom_cpu, lhs, rhs);
    defer product.deinit();
    var fused = (try executeCpuGemmScaledTarget(f64, lhs, rhs, addend, 1.0, 1.0)) orelse return error.BackendFailure;
    defer fused.deinit();
    try std.testing.expect(fused.isContiguous());
    try std.testing.expectEqualSlices(usize, &.{ m, n }, fused.shape);

    const checks = [_][2]usize{
        .{ 0, 0 },
        .{ 3, 17 },
        .{ m / 2, 5 },
        .{ m - 1, n - 1 },
    };
    for (checks) |idx| {
        const index = idx[0] * n + idx[1];
        try std.testing.expectApproxEqAbs(product.data[index] + addend.data[index], fused.data[index], 1e-9);
    }
}

test "CPU f64 matmulAdd has separate full-prepack materialization predicate" {
    try std.testing.expect(!shouldMaterializeCpuF64ColumnMajorGemm(512, 512, 128));
    try std.testing.expect(shouldMaterializeCpuF64ColumnMajorGemmAdd(512, 512, 128));
    try std.testing.expect(shouldMaterializeCpuF64ColumnMajorGemm(100, 100, 100));
    try std.testing.expect(shouldMaterializeCpuF64ColumnMajorGemm(16, 512, 16));
    try std.testing.expect(shouldMaterializeCpuF64ColumnMajorGemm(16, 1024, 16));
    try std.testing.expect(shouldMaterializeCpuF64ColumnMajorGemm(32, 512, 32));
    try std.testing.expect(shouldMaterializeCpuF64ColumnMajorGemm(32, 1024, 32));
    try std.testing.expect(shouldMaterializeCpuF64ColumnMajorGemm(96, 96, 96));
    try std.testing.expect(shouldMaterializeCpuF64ColumnMajorGemm(64, 192, 128));
    try std.testing.expect(shouldMaterializeCpuF64ColumnMajorGemm(96, 192, 128));
    try std.testing.expect(shouldMaterializeCpuF64ColumnMajorGemm(192, 96, 128));
    try std.testing.expect(shouldMaterializeCpuF64ColumnMajorGemm(192, 192, 128));
    try std.testing.expect(shouldMaterializeCpuF64ColumnMajorGemm(384, 384, 128));
    try std.testing.expect(shouldMaterializeCpuF64ColumnMajorGemmAdd(384, 128, 128));
}

test "CPU f32 materialization predicate covers low-K large square AMX shape" {
    try std.testing.expect(shouldMaterializeCpuF32ColumnMajorGemm(64, 64, 64));
    try std.testing.expect(shouldMaterializeCpuF32ColumnMajorGemm(768, 768, 128));
    try std.testing.expect(shouldMaterializeCpuF32ColumnMajorGemm(64, 192, 128));
    try std.testing.expect(shouldMaterializeCpuF32ColumnMajorGemm(96, 192, 128));
    try std.testing.expect(shouldMaterializeCpuF32ColumnMajorGemm(192, 96, 128));
    try std.testing.expect(shouldMaterializeCpuF32ColumnMajorGemm(128, 128, 16));
    try std.testing.expect(shouldMaterializeCpuF32ColumnMajorGemm(128, 128, 32));
    try std.testing.expect(shouldMaterializeCpuF32ColumnMajorGemm(192, 96, 16));
    try std.testing.expect(shouldMaterializeCpuF32ColumnMajorGemm(192, 96, 32));
    try std.testing.expect(shouldMaterializeCpuF32ColumnMajorGemm(64, 192, 16));
    try std.testing.expect(shouldMaterializeCpuF32ColumnMajorGemm(64, 192, 32));
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

    var cuda_like = try fakeCudaArrayForUnaryTest(f32, gpa, 4);
    defer cuda_like.deinit();
    // Inverse trig is intentionally not a CUDA unary runtime op today.  The
    // dispatcher must decline it cleanly even for CUDA-resident arrays; this
    // catches accidental `unreachable` paths without requiring CUDA hardware.
    const unsupported_cuda_unary = try executeUnary(f32, .asin, .cuda, cuda_like);
    try std.testing.expect(unsupported_cuda_unary == null);

    var cuda_like64 = try fakeCudaArrayForUnaryTest(f64, gpa, 4);
    defer cuda_like64.deinit();
    // Axiom's cached CUDA unary kernels only expose extended transcendental ops
    // for f32.  f64/f16/bf16 should be rejected before output storage/runtime
    // work is attempted.
    const unsupported_f64_log = try executeUnary(f64, .log, .cuda, cuda_like64);
    try std.testing.expect(unsupported_f64_log == null);
}

fn fakeCudaArrayForUnaryTest(comptime T: type, allocator: std.mem.Allocator, len: usize) !array_mod.Array(T) {
    const shape = try allocator.dupe(usize, &.{len});
    errdefer allocator.free(shape);
    const strides = try allocator.dupe(usize, &.{@as(usize, 1)});
    errdefer allocator.free(strides);
    const values = try allocator.alloc(T, 0);
    errdefer allocator.free(values);
    const bytes = try std.math.mul(usize, len, @sizeOf(T));
    return .{
        .allocator = allocator,
        .data = values,
        .shape = shape,
        .strides = strides,
        .device = array_mod.Device.cuda(0),
        .device_storage = .{
            .device = array_mod.Device.cuda(0),
            // Unsupported-op tests must not require CUDA hardware.  A nonzero
            // borrowed pointer is enough to satisfy the Array/device shape
            // preconditions while remaining safe to deinit because `owns=false`.
            .ptr = 1,
            .len = len,
            .bytes = bytes,
            .owns = false,
        },
    };
}
