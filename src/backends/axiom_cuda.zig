//! Axiom CUDA bridge for Vectra.
//!
//! Vectra now imports Axiom by default.  CUDA availability is runtime-gated by
//! the CUDA driver/device, while supported CUDA-resident f32 elementwise/matmul
//! paths launch through Axiom with existing device pointers.  Device matmul uses
//! Axiom's cuBLAS-backed SGEMM wrapper first for PyTorch-class throughput and
//! falls back to the existing Axiom PTX seed if cuBLAS is unavailable.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("vectra_build_options");
const array_mod = @import("../array.zig");

const axiom = if (build_options.enable_axiom_cuda) @import("axiom") else struct {};
const BFloat16 = array_mod.BFloat16;

pub const Status = enum {
    disabled,
    skipped,
    ran,
    failed,

    pub fn label(status: Status) []const u8 {
        return @tagName(status);
    }
};

pub const BinaryOp = enum {
    add,
    sub,
    mul,
    div,
};

pub const UnaryOp = enum {
    sqrt,
    exp,
    abs,
    log,
    sin,
    cos,
    tan,
    exp2,
    expm1,
    log1p,
    log2,
    log10,
};

pub const CudaDeviceGemmReportSnapshot = struct {
    ok: bool = false,
    backend: []const u8 = "",
    device_ordinal: usize = 0,
    m: usize = 0,
    n: usize = 0,
    k: usize = 0,
    lhs_device_ptr: u64 = 0,
    rhs_device_ptr: u64 = 0,
    out_device_ptr: u64 = 0,
    alpha: f32 = 1.0,
    beta: f32 = 0.0,
    cache_hit: bool = false,
    lt_plan_cache_hit: bool = false,
    lt_algo_cache_hit: bool = false,
    memref_spec_fingerprint: u64 = 0,
    fingerprint: u64 = 0,

    pub fn valid(report: CudaDeviceGemmReportSnapshot) bool {
        return report.ok and report.m != 0 and report.n != 0 and report.k != 0 and report.lhs_device_ptr != 0 and report.rhs_device_ptr != 0 and report.out_device_ptr != 0;
    }
};

threadlocal var last_cuda_device_gemm_report: CudaDeviceGemmReportSnapshot = .{};

pub fn resetLastCudaDeviceGemmReport() void {
    last_cuda_device_gemm_report = .{};
}

pub fn lastCudaDeviceGemmReport() CudaDeviceGemmReportSnapshot {
    return last_cuda_device_gemm_report;
}

pub const CudaDeviceBatchedGemmReportSnapshot = struct {
    ok: bool = false,
    backend: []const u8 = "",
    device_ordinal: usize = 0,
    batch_count: usize = 0,
    m: usize = 0,
    n: usize = 0,
    k: usize = 0,
    plan_fingerprint: u64 = 0,
    first_batch_fingerprint: u64 = 0,
    last_batch_fingerprint: u64 = 0,
    combined_batch_fingerprint: u64 = 0,
    fingerprint: u64 = 0,

    pub fn valid(report: CudaDeviceBatchedGemmReportSnapshot) bool {
        return report.ok and
            report.batch_count != 0 and
            report.m != 0 and
            report.n != 0 and
            report.k != 0 and
            report.plan_fingerprint != 0 and
            report.first_batch_fingerprint != 0 and
            report.last_batch_fingerprint != 0 and
            report.combined_batch_fingerprint != 0 and
            report.fingerprint != 0;
    }
};

threadlocal var last_cuda_device_batched_gemm_report: CudaDeviceBatchedGemmReportSnapshot = .{};

pub fn resetLastCudaDeviceBatchedGemmReport() void {
    last_cuda_device_batched_gemm_report = .{};
}

pub fn lastCudaDeviceBatchedGemmReport() CudaDeviceBatchedGemmReportSnapshot {
    return last_cuda_device_batched_gemm_report;
}

pub const CudaDeviceMemRefReportSnapshot = struct {
    ok: bool = false,
    operation: []const u8 = "",
    device_ordinal: usize = 0,
    rows: usize = 0,
    cols: usize = 0,
    axis: usize = 0,
    input_device_ptr: u64 = 0,
    aux_device_ptr: u64 = 0,
    out_device_ptr: u64 = 0,
    memref_spec_fingerprint: u64 = 0,
    report_fingerprint: u64 = 0,

    pub fn valid(report: CudaDeviceMemRefReportSnapshot) bool {
        return report.ok and
            report.operation.len != 0 and
            report.rows != 0 and
            report.cols != 0 and
            report.input_device_ptr != 0 and
            report.out_device_ptr != 0 and
            report.memref_spec_fingerprint != 0 and
            report.report_fingerprint != 0;
    }
};

threadlocal var last_cuda_device_memref_report: CudaDeviceMemRefReportSnapshot = .{};

pub fn resetLastCudaDeviceMemRefReport() void {
    last_cuda_device_memref_report = .{};
}

pub fn lastCudaDeviceMemRefReport() CudaDeviceMemRefReportSnapshot {
    return last_cuda_device_memref_report;
}

pub fn synchronizeDevice(allocator: std.mem.Allocator, device: array_mod.Device) array_mod.ArrayError!void {
    if (!build_options.enable_axiom_cuda or !device.isCuda()) return error.InvalidDevice;
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(allocator);
    runtime.synchronizeCudaDevice(device.index) catch return error.BackendFailure;
}

fn recordCudaDeviceGemmReport(report: anytype) void {
    last_cuda_device_gemm_report = .{
        .ok = report.ok,
        .backend = report.backend,
        .device_ordinal = report.device_ordinal,
        .m = report.m,
        .n = report.n,
        .k = report.k,
        .lhs_device_ptr = report.lhs_device_ptr,
        .rhs_device_ptr = report.rhs_device_ptr,
        .out_device_ptr = report.out_device_ptr,
        .alpha = report.alpha,
        .beta = report.beta,
        .cache_hit = report.cache_hit,
        .lt_plan_cache_hit = report.lt_plan_cache_hit,
        .lt_algo_cache_hit = report.lt_algo_cache_hit,
        .memref_spec_fingerprint = cudaDeviceReportMemRefSpecFingerprint(report),
        .fingerprint = report.fingerprint(),
    };
}

fn recordCudaDeviceBatchedGemmReport(report: anytype) void {
    last_cuda_device_batched_gemm_report = .{
        .ok = report.ok,
        .backend = report.backend,
        .device_ordinal = report.device_ordinal,
        .batch_count = report.batch_count,
        .m = report.m,
        .n = report.n,
        .k = report.k,
        .plan_fingerprint = report.plan_fingerprint,
        .first_batch_fingerprint = report.first_batch_fingerprint,
        .last_batch_fingerprint = report.last_batch_fingerprint,
        .combined_batch_fingerprint = report.combined_batch_fingerprint,
        .fingerprint = report.fingerprint(),
    };
}

fn cudaDeviceReportMemRefSpecFingerprint(report: anytype) u64 {
    const Report = @TypeOf(report);
    if (comptime !@hasField(Report, "memref_spec_fingerprint")) return 0;
    return report.memref_spec_fingerprint;
}

fn cudaDeviceMemRefReportAxis(report: anytype) usize {
    const Report = @TypeOf(report);
    if (comptime !@hasField(Report, "axis")) return 0;
    const axis = report.axis;
    const Axis = @TypeOf(axis);
    return switch (@typeInfo(Axis)) {
        .int, .comptime_int => @intCast(axis),
        .@"enum" => @intCast(@intFromEnum(axis)),
        else => 0,
    };
}

fn cudaDeviceMemRefReportRows(report: anytype) usize {
    const Report = @TypeOf(report);
    if (comptime @hasField(Report, "rows")) return report.rows;
    if (comptime @hasField(Report, "len")) return report.len;
    return 0;
}

fn cudaDeviceMemRefReportCols(report: anytype) usize {
    const Report = @TypeOf(report);
    if (comptime @hasField(Report, "cols")) return report.cols;
    if (comptime @hasField(Report, "len")) return 1;
    return 0;
}

fn cudaDeviceMemRefReportInputPtr(report: anytype) u64 {
    const Report = @TypeOf(report);
    if (comptime @hasField(Report, "input_device_ptr")) return report.input_device_ptr;
    if (comptime @hasField(Report, "lhs_device_ptr")) return report.lhs_device_ptr;
    return 0;
}

fn cudaDeviceMemRefReportAuxiliaryPtr(report: anytype) u64 {
    const Report = @TypeOf(report);
    if (comptime @hasField(Report, "bias_device_ptr")) return report.bias_device_ptr;
    if (comptime @hasField(Report, "rhs_device_ptr")) return report.rhs_device_ptr;
    return 0;
}

fn recordCudaDeviceMemRefReport(operation: []const u8, report: anytype) void {
    last_cuda_device_memref_report = .{
        .ok = report.ok,
        .operation = operation,
        .device_ordinal = report.device_ordinal,
        .rows = cudaDeviceMemRefReportRows(report),
        .cols = cudaDeviceMemRefReportCols(report),
        .axis = cudaDeviceMemRefReportAxis(report),
        .input_device_ptr = cudaDeviceMemRefReportInputPtr(report),
        .aux_device_ptr = cudaDeviceMemRefReportAuxiliaryPtr(report),
        .out_device_ptr = report.out_device_ptr,
        .memref_spec_fingerprint = report.memref_spec_fingerprint,
        .report_fingerprint = report.fingerprint(),
    };
}

pub const CudaDTypeBridgeStatus = enum(u8) {
    native_cuda_seed,
    widened_f32_seed,
    cpu_veyra_seed,
    planned,
    not_exposed,

    pub fn label(status: CudaDTypeBridgeStatus) []const u8 {
        return @tagName(status);
    }

    pub fn hasCudaBridge(status: CudaDTypeBridgeStatus) bool {
        return status == .native_cuda_seed or status == .widened_f32_seed;
    }
};

pub const CudaDTypeSupportRecord = struct {
    cuda_name: []const u8,
    cuda_value: i32,
    meaning: []const u8,
    vectra_dtype: ?array_mod.DType = null,
    status: CudaDTypeBridgeStatus = .planned,
    same_shape_elementwise: bool = false,
    scalar_broadcast: bool = false,
    matmul: bool = false,
    smoke_covered: bool = false,

    pub fn valid(record: CudaDTypeSupportRecord) bool {
        return record.cuda_name.len != 0 and record.meaning.len != 0;
    }

    pub fn vectraName(record: CudaDTypeSupportRecord) []const u8 {
        return if (record.vectra_dtype) |dtype| dtype.name() else "not_exposed";
    }

    pub fn fingerprint(record: CudaDTypeSupportRecord) u64 {
        var hasher = std.hash.Wyhash.init(0x0abc_7aaa_d79e_0001);
        hashBytes(&hasher, record.cuda_name);
        hashI32(&hasher, record.cuda_value);
        hashBytes(&hasher, record.meaning);
        hashBytes(&hasher, record.vectraName());
        hashBytes(&hasher, record.status.label());
        hashBool(&hasher, record.same_shape_elementwise);
        hashBool(&hasher, record.scalar_broadcast);
        hashBool(&hasher, record.matmul);
        hashBool(&hasher, record.smoke_covered);
        return hasher.final();
    }
};

/// CUDA dtype evidence mirrored from `/usr/local/cuda/include/library_types.h`.
///
/// This registry is intentionally explicit data so downstream libraries can
/// audit which CUDA dtypes Vectra exposes today, which route through Axiom CUDA,
/// and which are only planned.
pub const cuda_dtype_support = [_]CudaDTypeSupportRecord{
    .{ .cuda_name = "CUDA_R_16F", .cuda_value = 2, .meaning = "real half", .vectra_dtype = .f16, .status = .widened_f32_seed, .same_shape_elementwise = true, .matmul = true, .smoke_covered = true },
    .{ .cuda_name = "CUDA_C_16F", .cuda_value = 6, .meaning = "complex half pair", .status = .not_exposed },
    .{ .cuda_name = "CUDA_R_16BF", .cuda_value = 14, .meaning = "real bfloat16", .vectra_dtype = .bf16, .status = .widened_f32_seed, .same_shape_elementwise = true, .matmul = true, .smoke_covered = true },
    .{ .cuda_name = "CUDA_C_16BF", .cuda_value = 15, .meaning = "complex bfloat16 pair", .status = .not_exposed },
    .{ .cuda_name = "CUDA_R_32F", .cuda_value = 0, .meaning = "real float", .vectra_dtype = .f32, .status = .native_cuda_seed, .same_shape_elementwise = true, .scalar_broadcast = true, .matmul = true, .smoke_covered = true },
    .{ .cuda_name = "CUDA_C_32F", .cuda_value = 4, .meaning = "complex float pair", .vectra_dtype = .c64, .status = .planned },
    .{ .cuda_name = "CUDA_R_64F", .cuda_value = 1, .meaning = "real double", .vectra_dtype = .f64, .status = .native_cuda_seed, .same_shape_elementwise = true, .scalar_broadcast = true, .matmul = true, .smoke_covered = true },
    .{ .cuda_name = "CUDA_C_64F", .cuda_value = 5, .meaning = "complex double pair", .vectra_dtype = .c128, .status = .planned },
    .{ .cuda_name = "CUDA_R_4I", .cuda_value = 16, .meaning = "signed 4-bit integer", .status = .not_exposed },
    .{ .cuda_name = "CUDA_C_4I", .cuda_value = 17, .meaning = "signed 4-bit integer pair", .status = .not_exposed },
    .{ .cuda_name = "CUDA_R_4U", .cuda_value = 18, .meaning = "unsigned 4-bit integer", .status = .not_exposed },
    .{ .cuda_name = "CUDA_C_4U", .cuda_value = 19, .meaning = "unsigned 4-bit integer pair", .status = .not_exposed },
    .{ .cuda_name = "CUDA_R_8I", .cuda_value = 3, .meaning = "signed 8-bit integer", .vectra_dtype = .i8, .status = .planned },
    .{ .cuda_name = "CUDA_C_8I", .cuda_value = 7, .meaning = "signed 8-bit integer pair", .status = .not_exposed },
    .{ .cuda_name = "CUDA_R_8U", .cuda_value = 8, .meaning = "unsigned 8-bit integer", .vectra_dtype = .u8, .status = .planned },
    .{ .cuda_name = "CUDA_C_8U", .cuda_value = 9, .meaning = "unsigned 8-bit integer pair", .status = .not_exposed },
    .{ .cuda_name = "CUDA_R_16I", .cuda_value = 20, .meaning = "signed 16-bit integer", .vectra_dtype = .i16, .status = .planned },
    .{ .cuda_name = "CUDA_C_16I", .cuda_value = 21, .meaning = "signed 16-bit integer pair", .status = .not_exposed },
    .{ .cuda_name = "CUDA_R_16U", .cuda_value = 22, .meaning = "unsigned 16-bit integer", .vectra_dtype = .u16, .status = .planned },
    .{ .cuda_name = "CUDA_C_16U", .cuda_value = 23, .meaning = "unsigned 16-bit integer pair", .status = .not_exposed },
    .{ .cuda_name = "CUDA_R_32I", .cuda_value = 10, .meaning = "signed 32-bit integer", .vectra_dtype = .i32, .status = .planned },
    .{ .cuda_name = "CUDA_C_32I", .cuda_value = 11, .meaning = "signed 32-bit integer pair", .status = .not_exposed },
    .{ .cuda_name = "CUDA_R_32U", .cuda_value = 12, .meaning = "unsigned 32-bit integer", .vectra_dtype = .u32, .status = .planned },
    .{ .cuda_name = "CUDA_C_32U", .cuda_value = 13, .meaning = "unsigned 32-bit integer pair", .status = .not_exposed },
    .{ .cuda_name = "CUDA_R_64I", .cuda_value = 24, .meaning = "signed 64-bit integer", .vectra_dtype = .i64, .status = .planned },
    .{ .cuda_name = "CUDA_C_64I", .cuda_value = 25, .meaning = "signed 64-bit integer pair", .status = .not_exposed },
    .{ .cuda_name = "CUDA_R_64U", .cuda_value = 26, .meaning = "unsigned 64-bit integer", .vectra_dtype = .u64, .status = .planned },
    .{ .cuda_name = "CUDA_C_64U", .cuda_value = 27, .meaning = "unsigned 64-bit integer pair", .status = .not_exposed },
    .{ .cuda_name = "CUDA_R_8F_E4M3", .cuda_value = 28, .meaning = "fp8 e4m3", .status = .not_exposed },
    .{ .cuda_name = "CUDA_R_8F_E5M2", .cuda_value = 29, .meaning = "fp8 e5m2", .status = .not_exposed },
    .{ .cuda_name = "CUDA_R_8F_UE8M0", .cuda_value = 30, .meaning = "fp8 e8m0 exponent-only", .status = .not_exposed },
    .{ .cuda_name = "CUDA_R_6F_E2M3", .cuda_value = 31, .meaning = "fp6 e2m3", .status = .not_exposed },
    .{ .cuda_name = "CUDA_R_6F_E3M2", .cuda_value = 32, .meaning = "fp6 e3m2", .status = .not_exposed },
    .{ .cuda_name = "CUDA_R_4F_E2M1", .cuda_value = 33, .meaning = "fp4 e2m1", .status = .not_exposed },
};

pub fn cudaDTypeSupportRecords() []const CudaDTypeSupportRecord {
    return &cuda_dtype_support;
}

pub fn findCudaDTypeSupport(cuda_name: []const u8) ?CudaDTypeSupportRecord {
    for (cuda_dtype_support) |record| {
        if (std.mem.eql(u8, record.cuda_name, cuda_name)) return record;
    }
    return null;
}

pub fn findVectraDTypeSupport(dtype: array_mod.DType) ?CudaDTypeSupportRecord {
    for (cuda_dtype_support) |record| {
        if (record.vectra_dtype != null and record.vectra_dtype.? == dtype) return record;
    }
    return null;
}

pub fn cudaDTypeNativeSeedCount() usize {
    var count: usize = 0;
    for (cuda_dtype_support) |record| {
        if (record.status == .native_cuda_seed) count += 1;
    }
    return count;
}

pub fn cudaDTypeWidenedSeedCount() usize {
    var count: usize = 0;
    for (cuda_dtype_support) |record| {
        if (record.status == .widened_f32_seed) count += 1;
    }
    return count;
}

pub fn cudaDTypeBridgeCount() usize {
    var count: usize = 0;
    for (cuda_dtype_support) |record| {
        if (record.status.hasCudaBridge()) count += 1;
    }
    return count;
}

pub fn cudaDTypeSupportFingerprint() u64 {
    var hasher = std.hash.Wyhash.init(0x0abc_7aaa_d79e_511c);
    hashU64(&hasher, cuda_dtype_support.len);
    for (cuda_dtype_support) |record| {
        std.debug.assert(record.valid());
        hashU64(&hasher, record.fingerprint());
    }
    hashU64(&hasher, cudaDTypeNativeSeedCount());
    hashU64(&hasher, cudaDTypeWidenedSeedCount());
    hashU64(&hasher, cudaDTypeBridgeCount());
    return hasher.final();
}

pub const BufferPlanEvidence = struct {
    ok: bool = false,
    logical_elements: usize = 0,
    required_span: usize = 0,
    logical_bytes: usize = 0,
    required_bytes: usize = 0,
    linear_copy: bool = false,
    fingerprint: u64 = 0,
    copy_ok: bool = false,
    copy_requires_strided: bool = false,
    copy_fingerprint: u64 = 0,
};

pub const TypedGemmPlanEvidence = struct {
    ok: bool = false,
    element_name: []const u8 = "",
    readiness_status: []const u8 = "",
    m: usize = 0,
    n: usize = 0,
    k: usize = 0,
    tile_m: usize = 0,
    tile_n: usize = 0,
    tile_k: usize = 0,
    grid_m: usize = 0,
    grid_n: usize = 0,
    total_ctas: usize = 0,
    threads_per_cta: usize = 0,
    argument_bytes: usize = 0,
    plan_fingerprint: u64 = 0,
    seed_fingerprint: u64 = 0,
    readiness_fingerprint: u64 = 0,

    pub fn fingerprint(evidence: TypedGemmPlanEvidence) u64 {
        var hasher = std.hash.Wyhash.init(0x0abc_7aaa_9e00_719e);
        hashBool(&hasher, evidence.ok);
        hashBytes(&hasher, evidence.element_name);
        hashBytes(&hasher, evidence.readiness_status);
        hashU64(&hasher, evidence.m);
        hashU64(&hasher, evidence.n);
        hashU64(&hasher, evidence.k);
        hashU64(&hasher, evidence.tile_m);
        hashU64(&hasher, evidence.tile_n);
        hashU64(&hasher, evidence.tile_k);
        hashU64(&hasher, evidence.grid_m);
        hashU64(&hasher, evidence.grid_n);
        hashU64(&hasher, evidence.total_ctas);
        hashU64(&hasher, evidence.threads_per_cta);
        hashU64(&hasher, evidence.argument_bytes);
        hashU64(&hasher, evidence.plan_fingerprint);
        hashU64(&hasher, evidence.seed_fingerprint);
        hashU64(&hasher, evidence.readiness_fingerprint);
        return hasher.final();
    }
};

const TypedGemmElement = enum { f16, bf16 };

pub const DeviceArrayF32 = struct {
    allocator: std.mem.Allocator,
    shape: []usize,
    device_ptr: u64,
    required_bytes: usize,
    logical_elements: usize,
    allocation_fingerprint: u64,
    pool_fingerprint: u64,
    released: bool = false,

    pub fn fromHost(allocator: std.mem.Allocator, host: array_mod.Array(f32)) array_mod.ArrayError!?DeviceArrayF32 {
        if (!build_options.enable_axiom_cuda) return null;
        if (!supportedNonEmptyContiguous(host)) return null;
        const plan = axiom.accelerator.TensorDeviceBufferPlan.fromBufferView(
            axiom.accelerator.TensorBufferView.contiguous("vectra_device_array", @intCast(@intFromPtr(host.data.ptr)), host.data.len),
        ) catch return null;
        var pool = axiom.accelerator.TensorDeviceBufferPool.init(axiom.accelerator.AcceleratorRuntime.cuda(allocator));
        const acquired = pool.acquire(plan) catch return null;
        if (!acquired.ok()) return null;
        const shape = try allocator.dupe(usize, host.shape);
        return .{
            .allocator = allocator,
            .shape = shape,
            .device_ptr = acquired.allocation.device_ptr,
            .required_bytes = acquired.allocation.requested_bytes,
            .logical_elements = host.data.len,
            .allocation_fingerprint = acquired.allocation.fingerprint(),
            .pool_fingerprint = pool.fingerprint(),
        };
    }

    pub fn deinit(self: *DeviceArrayF32) void {
        if (!self.released and build_options.enable_axiom_cuda and self.device_ptr != 0) {
            _ = axiom.accelerator.AcceleratorRuntime.cuda(self.allocator).freeTensorDeviceBuffer(self.device_ptr) catch null;
        }
        self.allocator.free(self.shape);
        self.* = undefined;
    }

    pub fn release(self: *DeviceArrayF32) void {
        if (!self.released and build_options.enable_axiom_cuda and self.device_ptr != 0) {
            _ = axiom.accelerator.AcceleratorRuntime.cuda(self.allocator).freeTensorDeviceBuffer(self.device_ptr) catch null;
            self.released = true;
        }
    }

    pub fn ok(self: DeviceArrayF32) bool {
        return self.device_ptr != 0 and self.required_bytes != 0 and self.logical_elements != 0 and self.allocation_fingerprint != 0 and self.pool_fingerprint != 0 and !self.released;
    }

    pub fn fingerprint(self: DeviceArrayF32) u64 {
        var hasher = std.hash.Wyhash.init(0x0abc_7aaa_deb0_0001);
        hashU64(&hasher, self.device_ptr);
        hashU64(&hasher, self.required_bytes);
        hashU64(&hasher, self.logical_elements);
        hashU64(&hasher, self.allocation_fingerprint);
        hashU64(&hasher, self.pool_fingerprint);
        hashBool(&hasher, self.released);
        for (self.shape) |dim| hashU64(&hasher, dim);
        return hasher.final();
    }
};

pub fn toDeviceF32(allocator: std.mem.Allocator, host: array_mod.Array(f32)) array_mod.ArrayError!?DeviceArrayF32 {
    return DeviceArrayF32.fromHost(allocator, host);
}

const CudaResult = c_int;
const CudaDevice = c_int;
const CudaDevicePtr = u64;
const CudaContext = ?*anyopaque;
const CudaModule = ?*anyopaque;
const CudaFunction = ?*anyopaque;
const CudaStream = ?*anyopaque;
const CudaUChar = u8;
const CudaUShort = u16;

const cuda_success: CudaResult = 0;
const cuda_device_attribute_compute_capability_major: c_int = 75;
const cuda_device_attribute_compute_capability_minor: c_int = 76;

const CudaPrimaryContext = struct {
    device: CudaDevice,
    handle: CudaContext,

    fn release(context: *CudaPrimaryContext, driver: *CudaDriver) void {
        _ = driver.cuDevicePrimaryCtxRelease(context.device);
    }
};

const CudaDriver = struct {
    lib: std.DynLib,
    cuInit: *const fn (c_uint) callconv(.c) CudaResult,
    cuDeviceGet: *const fn (*CudaDevice, c_int) callconv(.c) CudaResult,
    cuDeviceGetAttribute: *const fn (*c_int, c_int, CudaDevice) callconv(.c) CudaResult,
    cuDevicePrimaryCtxRetain: *const fn (*CudaContext, CudaDevice) callconv(.c) CudaResult,
    cuDevicePrimaryCtxRelease: *const fn (CudaDevice) callconv(.c) CudaResult,
    cuCtxSetCurrent: *const fn (CudaContext) callconv(.c) CudaResult,
    cuMemAlloc_v2: *const fn (*CudaDevicePtr, usize) callconv(.c) CudaResult,
    cuMemFree_v2: *const fn (CudaDevicePtr) callconv(.c) CudaResult,
    cuMemcpyHtoD_v2: *const fn (CudaDevicePtr, *const anyopaque, usize) callconv(.c) CudaResult,
    cuMemcpyDtoH_v2: *const fn (*anyopaque, CudaDevicePtr, usize) callconv(.c) CudaResult,
    cuMemcpyDtoD_v2: *const fn (CudaDevicePtr, CudaDevicePtr, usize) callconv(.c) CudaResult,
    cuMemsetD8_v2: *const fn (CudaDevicePtr, CudaUChar, usize) callconv(.c) CudaResult,
    cuMemsetD16_v2: *const fn (CudaDevicePtr, CudaUShort, usize) callconv(.c) CudaResult,
    cuMemsetD32_v2: *const fn (CudaDevicePtr, c_uint, usize) callconv(.c) CudaResult,
    cuModuleLoadData: *const fn (*CudaModule, *const anyopaque) callconv(.c) CudaResult,
    cuModuleUnload: *const fn (CudaModule) callconv(.c) CudaResult,
    cuModuleGetFunction: *const fn (*CudaFunction, CudaModule, [*:0]const u8) callconv(.c) CudaResult,
    cuLaunchKernel: *const fn (CudaFunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, CudaStream, [*]?*anyopaque, ?*?*anyopaque) callconv(.c) CudaResult,
    cuCtxSynchronize: *const fn () callconv(.c) CudaResult,

    fn load() !CudaDriver {
        if (!builtin.link_libc) return error.CudaUnavailable;
        var lib = openCudaDynLib() catch return error.CudaUnavailable;
        errdefer lib.close();
        return .{
            .lib = lib,
            .cuInit = try lookupCuda(&lib, *const fn (c_uint) callconv(.c) CudaResult, "cuInit"),
            .cuDeviceGet = try lookupCuda(&lib, *const fn (*CudaDevice, c_int) callconv(.c) CudaResult, "cuDeviceGet"),
            .cuDeviceGetAttribute = try lookupCuda(&lib, *const fn (*c_int, c_int, CudaDevice) callconv(.c) CudaResult, "cuDeviceGetAttribute"),
            .cuDevicePrimaryCtxRetain = try lookupCuda(&lib, *const fn (*CudaContext, CudaDevice) callconv(.c) CudaResult, "cuDevicePrimaryCtxRetain"),
            .cuDevicePrimaryCtxRelease = try lookupCuda(&lib, *const fn (CudaDevice) callconv(.c) CudaResult, "cuDevicePrimaryCtxRelease"),
            .cuCtxSetCurrent = try lookupCuda(&lib, *const fn (CudaContext) callconv(.c) CudaResult, "cuCtxSetCurrent"),
            .cuMemAlloc_v2 = try lookupCuda(&lib, *const fn (*CudaDevicePtr, usize) callconv(.c) CudaResult, "cuMemAlloc_v2"),
            .cuMemFree_v2 = try lookupCuda(&lib, *const fn (CudaDevicePtr) callconv(.c) CudaResult, "cuMemFree_v2"),
            .cuMemcpyHtoD_v2 = try lookupCuda(&lib, *const fn (CudaDevicePtr, *const anyopaque, usize) callconv(.c) CudaResult, "cuMemcpyHtoD_v2"),
            .cuMemcpyDtoH_v2 = try lookupCuda(&lib, *const fn (*anyopaque, CudaDevicePtr, usize) callconv(.c) CudaResult, "cuMemcpyDtoH_v2"),
            .cuMemcpyDtoD_v2 = try lookupCuda(&lib, *const fn (CudaDevicePtr, CudaDevicePtr, usize) callconv(.c) CudaResult, "cuMemcpyDtoD_v2"),
            .cuMemsetD8_v2 = try lookupCuda(&lib, *const fn (CudaDevicePtr, CudaUChar, usize) callconv(.c) CudaResult, "cuMemsetD8_v2"),
            .cuMemsetD16_v2 = try lookupCuda(&lib, *const fn (CudaDevicePtr, CudaUShort, usize) callconv(.c) CudaResult, "cuMemsetD16_v2"),
            .cuMemsetD32_v2 = try lookupCuda(&lib, *const fn (CudaDevicePtr, c_uint, usize) callconv(.c) CudaResult, "cuMemsetD32_v2"),
            .cuModuleLoadData = try lookupCuda(&lib, *const fn (*CudaModule, *const anyopaque) callconv(.c) CudaResult, "cuModuleLoadData"),
            .cuModuleUnload = try lookupCuda(&lib, *const fn (CudaModule) callconv(.c) CudaResult, "cuModuleUnload"),
            .cuModuleGetFunction = try lookupCuda(&lib, *const fn (*CudaFunction, CudaModule, [*:0]const u8) callconv(.c) CudaResult, "cuModuleGetFunction"),
            .cuLaunchKernel = try lookupCuda(&lib, *const fn (CudaFunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, CudaStream, [*]?*anyopaque, ?*?*anyopaque) callconv(.c) CudaResult, "cuLaunchKernel"),
            .cuCtxSynchronize = try lookupCuda(&lib, *const fn () callconv(.c) CudaResult, "cuCtxSynchronize"),
        };
    }

    fn close(driver: *CudaDriver) void {
        driver.lib.close();
    }

    fn init(driver: *CudaDriver) !void {
        try checkCuda(driver.cuInit(0));
    }

    fn primaryContextRetain(driver: *CudaDriver, index: usize) !CudaPrimaryContext {
        if (index > @as(usize, @intCast(std.math.maxInt(c_int)))) return error.InvalidDevice;
        var device: CudaDevice = 0;
        try checkCuda(driver.cuDeviceGet(&device, @intCast(index)));
        var context: CudaContext = null;
        try checkCuda(driver.cuDevicePrimaryCtxRetain(&context, device));
        return .{ .device = device, .handle = context };
    }

    fn setCurrent(driver: *CudaDriver, context: CudaContext) !void {
        try checkCuda(driver.cuCtxSetCurrent(context));
    }

    fn deviceAttribute(driver: *CudaDriver, device: CudaDevice, attribute: c_int) !c_int {
        var value: c_int = 0;
        try checkCuda(driver.cuDeviceGetAttribute(&value, attribute, device));
        return value;
    }

    fn resolveCudaArch(driver: *CudaDriver, device: CudaDevice, requested: []const u8, buffer: *[16]u8) ![]const u8 {
        if (std.mem.eql(u8, requested, "auto")) {
            const major = try driver.deviceAttribute(device, cuda_device_attribute_compute_capability_major);
            const minor = try driver.deviceAttribute(device, cuda_device_attribute_compute_capability_minor);
            if (major < 0 or minor < 0 or major > 99 or minor > 99) return error.CudaUnavailable;
            return std.fmt.bufPrint(buffer, "sm_{d}{d}", .{ major, minor });
        }
        if (std.mem.startsWith(u8, requested, "compute_")) {
            const suffix = requested["compute_".len..];
            if (suffix.len == 0 or suffix.len + "sm_".len > buffer.len) return error.CudaUnavailable;
            @memcpy(buffer[0.."sm_".len], "sm_");
            @memcpy(buffer["sm_".len .. "sm_".len + suffix.len], suffix);
            return buffer[0 .. "sm_".len + suffix.len];
        }
        return requested;
    }

    fn memAlloc(driver: *CudaDriver, bytes: usize) !CudaDevicePtr {
        var ptr: CudaDevicePtr = 0;
        try checkCuda(driver.cuMemAlloc_v2(&ptr, bytes));
        return ptr;
    }

    fn memFree(driver: *CudaDriver, ptr: CudaDevicePtr) void {
        _ = driver.cuMemFree_v2(ptr);
    }

    fn memcpyHtoD(driver: *CudaDriver, dst: CudaDevicePtr, src: *const anyopaque, bytes: usize) !void {
        try checkCuda(driver.cuMemcpyHtoD_v2(dst, src, bytes));
    }

    fn memcpyDtoH(driver: *CudaDriver, dst: *anyopaque, src: CudaDevicePtr, bytes: usize) !void {
        try checkCuda(driver.cuMemcpyDtoH_v2(dst, src, bytes));
    }

    fn memcpyDtoD(driver: *CudaDriver, dst: CudaDevicePtr, src: CudaDevicePtr, bytes: usize) !void {
        try checkCuda(driver.cuMemcpyDtoD_v2(dst, src, bytes));
    }

    fn memsetD8(driver: *CudaDriver, dst: CudaDevicePtr, value: u8, bytes: usize) !void {
        try checkCuda(driver.cuMemsetD8_v2(dst, @intCast(value), bytes));
    }

    fn memsetD16(driver: *CudaDriver, dst: CudaDevicePtr, value: u16, count: usize) !void {
        try checkCuda(driver.cuMemsetD16_v2(dst, @intCast(value), count));
    }

    fn memsetD32(driver: *CudaDriver, dst: CudaDevicePtr, value: u32, count: usize) !void {
        try checkCuda(driver.cuMemsetD32_v2(dst, @intCast(value), count));
    }

    fn moduleLoadData(driver: *CudaDriver, image: []const u8) !CudaModule {
        var module: CudaModule = null;
        try checkCuda(driver.cuModuleLoadData(&module, image.ptr));
        return module;
    }

    fn moduleUnload(driver: *CudaDriver, module: CudaModule) void {
        _ = driver.cuModuleUnload(module);
    }

    fn moduleGetFunction(driver: *CudaDriver, module: CudaModule, symbol: [*:0]const u8) !CudaFunction {
        var function: CudaFunction = null;
        try checkCuda(driver.cuModuleGetFunction(&function, module, symbol));
        return function;
    }

    fn launchKernel(
        driver: *CudaDriver,
        function: CudaFunction,
        grid: anytype,
        block: anytype,
        shared_memory_bytes: u32,
        args: [*]?*anyopaque,
    ) !void {
        try checkCuda(driver.cuLaunchKernel(
            function,
            grid.x,
            grid.y,
            grid.z,
            block.x,
            block.y,
            block.z,
            shared_memory_bytes,
            null,
            args,
            null,
        ));
    }

    fn synchronize(driver: *CudaDriver) !void {
        try checkCuda(driver.cuCtxSynchronize());
    }

    fn hasDevice(index: usize) bool {
        var driver = CudaDriver.load() catch return false;
        defer driver.close();
        driver.init() catch return false;
        var context = driver.primaryContextRetain(index) catch return false;
        defer context.release(&driver);
        driver.setCurrent(context.handle) catch return false;
        return true;
    }
};

fn openCudaDynLib() !std.DynLib {
    var last_error: anyerror = error.FileNotFound;
    for (&[_][]const u8{
        "libcuda.so.1",
        "libcuda.so",
        "/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/lib/x86_64-linux-gnu/libcuda.so",
    }) |path| {
        return std.DynLib.open(path) catch |err| {
            last_error = err;
            continue;
        };
    }
    return last_error;
}

fn lookupCuda(lib: *std.DynLib, comptime T: type, name: [:0]const u8) !T {
    if (lib.lookup(T, name)) |symbol| return symbol;
    if (@hasDecl(@TypeOf(lib.inner), "lookupAddress")) {
        if (lib.inner.lookupAddress("libcuda.so.1", name)) |address| return @as(T, @ptrFromInt(address));
    }
    return error.MissingCudaSymbol;
}

fn checkCuda(result: CudaResult) !void {
    if (result != cuda_success) return error.CudaError;
}

fn withCudaContext(index: usize) !struct { driver: CudaDriver, context: CudaPrimaryContext } {
    var driver = try CudaDriver.load();
    errdefer driver.close();
    try driver.init();
    var context = try driver.primaryContextRetain(index);
    errdefer context.release(&driver);
    try driver.setCurrent(context.handle);
    return .{ .driver = driver, .context = context };
}

const cuda_storage_cache_capacity = 8;
const cuda_storage_cache_max_bytes: usize = @as(usize, 2) * 1024 * 1024 * 1024;

const CachedCudaStorage = struct {
    device: array_mod.Device = .cpu,
    ptr: u64 = 0,
    len: usize = 0,
    bytes: usize = 0,
    age: u64 = 0,

    fn live(entry: CachedCudaStorage) bool {
        return entry.ptr != 0 and entry.bytes != 0;
    }

    fn toStorage(entry: CachedCudaStorage) array_mod.DeviceStorage {
        return .{ .device = entry.device, .ptr = entry.ptr, .len = entry.len, .bytes = entry.bytes };
    }
};

threadlocal var cuda_storage_cache: [cuda_storage_cache_capacity]CachedCudaStorage = [_]CachedCudaStorage{.{}} ** cuda_storage_cache_capacity;
threadlocal var cuda_storage_cache_clock: u64 = 0;

fn takeCachedStorage(device: array_mod.Device, len: usize, bytes: usize) ?array_mod.DeviceStorage {
    for (&cuda_storage_cache) |*entry| {
        if (entry.live() and entry.device.sameDevice(device) and entry.len == len and entry.bytes == bytes) {
            const storage = entry.toStorage();
            entry.* = .{};
            return storage;
        }
    }
    return null;
}

fn cachedStorageBytes() usize {
    var total: usize = 0;
    for (cuda_storage_cache) |entry| {
        if (entry.live()) total += entry.bytes;
    }
    return total;
}

fn oldestCachedStorageSlot() ?usize {
    var found: ?usize = null;
    for (cuda_storage_cache, 0..) |entry, index| {
        if (!entry.live()) continue;
        if (found == null or entry.age < cuda_storage_cache[found.?].age) found = index;
    }
    return found;
}

fn emptyCachedStorageSlot() ?usize {
    for (cuda_storage_cache, 0..) |entry, index| {
        if (!entry.live()) return index;
    }
    return null;
}

fn freeStorageDriver(storage: array_mod.DeviceStorage) void {
    if (!build_options.enable_axiom_cuda or !storage.device.isCuda() or storage.ptr == 0 or !storage.owns) return;
    var session = withCudaContext(storage.device.index) catch return;
    defer session.driver.close();
    defer session.context.release(&session.driver);
    session.driver.memFree(storage.ptr);
    _ = session.driver.cuDevicePrimaryCtxRelease(session.context.device);
}

fn evictCachedStorageSlot(index: usize) void {
    const entry = cuda_storage_cache[index];
    cuda_storage_cache[index] = .{};
    if (entry.live()) freeStorageDriver(entry.toStorage());
}

fn cacheStorage(storage: array_mod.DeviceStorage) bool {
    if (!build_options.enable_axiom_cuda or !storage.device.isCuda() or storage.ptr == 0 or !storage.owns) return false;
    if (storage.bytes == 0 or storage.bytes > cuda_storage_cache_max_bytes) return false;

    while (cachedStorageBytes() + storage.bytes > cuda_storage_cache_max_bytes) {
        const slot = oldestCachedStorageSlot() orelse return false;
        evictCachedStorageSlot(slot);
    }

    const slot = emptyCachedStorageSlot() orelse blk: {
        const oldest = oldestCachedStorageSlot() orelse return false;
        evictCachedStorageSlot(oldest);
        break :blk oldest;
    };
    cuda_storage_cache_clock +%= 1;
    cuda_storage_cache[slot] = .{
        .device = storage.device,
        .ptr = storage.ptr,
        .len = storage.len,
        .bytes = storage.bytes,
        .age = cuda_storage_cache_clock,
    };
    return true;
}

pub fn flushStorageCache() void {
    for (0..cuda_storage_cache.len) |index| evictCachedStorageSlot(index);
}

pub fn allocateStorage(device: array_mod.Device, len: usize, element_size: usize) array_mod.ArrayError!?array_mod.DeviceStorage {
    if (!build_options.enable_axiom_cuda or !device.isCuda()) return null;
    const bytes = std.math.mul(usize, len, element_size) catch return error.InvalidShape;
    if (bytes == 0) return .{ .device = device, .ptr = 0, .len = len, .bytes = 0 };
    if (takeCachedStorage(device, len, bytes)) |storage| return storage;
    var session = withCudaContext(device.index) catch return error.InvalidDevice;
    // Keep one primary-context retain alive for the lifetime of the device
    // allocation; otherwise the driver may destroy the primary context and make
    // the returned device pointer invalid before the next operation.
    defer session.driver.close();
    const ptr = session.driver.memAlloc(bytes) catch {
        session.context.release(&session.driver);
        return error.BackendFailure;
    };
    return .{ .device = device, .ptr = ptr, .len = len, .bytes = bytes };
}

pub fn freeStorage(storage: array_mod.DeviceStorage) void {
    if (!build_options.enable_axiom_cuda or !storage.device.isCuda() or storage.ptr == 0 or !storage.owns) return;
    if (cacheStorage(storage)) return;
    freeStorageDriver(storage);
}

pub fn uploadStorage(storage: array_mod.DeviceStorage, bytes: []const u8) array_mod.ArrayError!void {
    if (!build_options.enable_axiom_cuda or !storage.device.isCuda() or bytes.len > storage.bytes) return error.InvalidDevice;
    if (bytes.len == 0) return;
    var session = withCudaContext(storage.device.index) catch return error.InvalidDevice;
    defer session.driver.close();
    defer session.context.release(&session.driver);
    session.driver.memcpyHtoD(storage.ptr, bytes.ptr, bytes.len) catch return error.BackendFailure;
}

pub fn downloadStorage(storage: array_mod.DeviceStorage, bytes: []u8) array_mod.ArrayError!void {
    if (!build_options.enable_axiom_cuda or !storage.device.isCuda() or bytes.len > storage.bytes) return error.InvalidDevice;
    if (bytes.len == 0) return;
    var session = withCudaContext(storage.device.index) catch return error.InvalidDevice;
    defer session.driver.close();
    defer session.context.release(&session.driver);
    session.driver.memcpyDtoH(bytes.ptr, storage.ptr, bytes.len) catch return error.BackendFailure;
}

pub fn copyStorage(dst: array_mod.DeviceStorage, src: array_mod.DeviceStorage) array_mod.ArrayError!void {
    if (!build_options.enable_axiom_cuda or !dst.device.sameDevice(src.device) or !dst.device.isCuda()) return error.InvalidDevice;
    if (dst.bytes < src.bytes or dst.len != src.len) return error.ShapeMismatch;
    if (src.bytes == 0) return;
    var session = withCudaContext(dst.device.index) catch return error.InvalidDevice;
    defer session.driver.close();
    defer session.context.release(&session.driver);
    session.driver.memcpyDtoD(dst.ptr, src.ptr, src.bytes) catch return error.BackendFailure;
}

pub fn zeroStorage(storage: array_mod.DeviceStorage) array_mod.ArrayError!void {
    if (!build_options.enable_axiom_cuda or !storage.device.isCuda()) return error.InvalidDevice;
    if (storage.bytes == 0) return;
    var session = withCudaContext(storage.device.index) catch return error.InvalidDevice;
    defer session.driver.close();
    defer session.context.release(&session.driver);
    session.driver.memsetD8(storage.ptr, 0, storage.bytes) catch return error.BackendFailure;
}

pub fn fillStorage(comptime T: type, storage: array_mod.DeviceStorage, value: T) array_mod.ArrayError!void {
    if (!build_options.enable_axiom_cuda or !storage.device.isCuda()) return error.InvalidDevice;
    if (storage.len == 0) return;
    if (storage.bytes != storage.len * @sizeOf(T)) return error.ShapeMismatch;
    if (std.meta.eql(value, std.mem.zeroes(T))) return zeroStorage(storage);
    var session = withCudaContext(storage.device.index) catch return error.InvalidDevice;
    defer session.driver.close();
    defer session.context.release(&session.driver);
    if (comptime T == bool) {
        const pattern: u8 = if (value) 1 else 0;
        session.driver.memsetD8(storage.ptr, pattern, storage.bytes) catch return error.BackendFailure;
        return;
    }
    if (comptime T == BFloat16) {
        session.driver.memsetD16(storage.ptr, value.bits, storage.len) catch return error.BackendFailure;
        return;
    }
    if (comptime @typeInfo(T) == .int and @sizeOf(T) == 1) {
        const pattern: u8 = @bitCast(value);
        session.driver.memsetD8(storage.ptr, pattern, storage.bytes) catch return error.BackendFailure;
        return;
    }
    if (comptime (@typeInfo(T) == .int or @typeInfo(T) == .float) and @sizeOf(T) == 2) {
        const pattern: u16 = @bitCast(value);
        session.driver.memsetD16(storage.ptr, pattern, storage.len) catch return error.BackendFailure;
        return;
    }
    if (comptime (@typeInfo(T) == .int or @typeInfo(T) == .float) and @sizeOf(T) == 4) {
        const pattern: u32 = @bitCast(value);
        session.driver.memsetD32(storage.ptr, pattern, storage.len) catch return error.BackendFailure;
        return;
    }
    const scratch = std.heap.smp_allocator;
    const tmp = try scratch.alloc(T, storage.len);
    defer scratch.free(tmp);
    @memset(tmp, value);
    return uploadStorage(storage, std.mem.sliceAsBytes(tmp));
}

const cached_philox_uniform_f32_ptx =
    \\.version 7.8
    \\.target sm_70
    \\.address_size 64
    \\
    \\.visible .entry vectra_philox_uniform_f32(
    \\    .param .u64 out_ptr,
    \\    .param .u32 n,
    \\    .param .u32 seed_lo,
    \\    .param .u32 seed_hi
    \\) {
    \\    .reg .pred %p<8>;
    \\    .reg .b32 %r<80>;
    \\    .reg .b64 %rd<6>;
    \\    .reg .f32 %f<3>;
    \\    ld.param.u64 %rd1, [out_ptr];
    \\    ld.param.u32 %r1, [n];
    \\    ld.param.u32 %r2, [seed_lo];
    \\    ld.param.u32 %r3, [seed_hi];
    \\    mov.u32 %r4, %ctaid.x;
    \\    mov.u32 %r5, %ntid.x;
    \\    mov.u32 %r6, %tid.x;
    \\    mad.lo.u32 %r7, %r4, %r5, %r6;
    \\    setp.ge.u32 %p1, %r7, %r1;
    \\    @%p1 bra DONE;
    \\    shr.u32 %r8, %r7, 2;
    \\    and.b32 %r9, %r7, 3;
    \\    mov.u32 %r10, 0;
    \\    mov.u32 %r11, 0;
    \\    mov.u32 %r12, 0;
    \\    mov.u32 %r13, %r2;
    \\    mov.u32 %r14, %r3;
    \\    mov.u32 %r15, 0;
    \\LOOP:
    \\    mul.lo.u32 %r16, %r8, 0xD2511F53;
    \\    mul.hi.u32 %r17, %r8, 0xD2511F53;
    \\    mul.lo.u32 %r18, %r11, 0xCD9E8D57;
    \\    mul.hi.u32 %r19, %r11, 0xCD9E8D57;
    \\    xor.b32 %r20, %r19, %r10;
    \\    xor.b32 %r20, %r20, %r13;
    \\    xor.b32 %r21, %r17, %r12;
    \\    xor.b32 %r21, %r21, %r14;
    \\    mov.u32 %r8, %r20;
    \\    mov.u32 %r10, %r18;
    \\    mov.u32 %r11, %r21;
    \\    mov.u32 %r12, %r16;
    \\    add.u32 %r13, %r13, 0x9E3779B9;
    \\    add.u32 %r14, %r14, 0xBB67AE85;
    \\    add.u32 %r15, %r15, 1;
    \\    setp.lt.u32 %p2, %r15, 10;
    \\    @%p2 bra LOOP;
    \\    setp.eq.u32 %p3, %r9, 0;
    \\    @%p3 mov.u32 %r22, %r8;
    \\    setp.eq.u32 %p4, %r9, 1;
    \\    @%p4 mov.u32 %r22, %r10;
    \\    setp.eq.u32 %p5, %r9, 2;
    \\    @%p5 mov.u32 %r22, %r11;
    \\    setp.eq.u32 %p6, %r9, 3;
    \\    @%p6 mov.u32 %r22, %r12;
    \\    shr.u32 %r23, %r22, 8;
    \\    cvt.rn.f32.u32 %f1, %r23;
    \\    mul.rn.f32 %f2, %f1, 0f33800000;
    \\    mul.wide.u32 %rd2, %r7, 4;
    \\    add.u64 %rd3, %rd1, %rd2;
    \\    st.global.f32 [%rd3], %f2;
    \\DONE:
    \\    ret;
    \\}
;

const cached_philox_uniform_f64_ptx =
    \\.version 7.8
    \\.target sm_70
    \\.address_size 64
    \\
    \\.visible .entry vectra_philox_uniform_f64(
    \\    .param .u64 out_ptr,
    \\    .param .u32 n,
    \\    .param .u32 seed_lo,
    \\    .param .u32 seed_hi
    \\) {
    \\    .reg .pred %p<8>;
    \\    .reg .b32 %r<80>;
    \\    .reg .b64 %rd<6>;
    \\    .reg .f32 %f<3>;
    \\    .reg .f64 %fd<3>;
    \\    ld.param.u64 %rd1, [out_ptr];
    \\    ld.param.u32 %r1, [n];
    \\    ld.param.u32 %r2, [seed_lo];
    \\    ld.param.u32 %r3, [seed_hi];
    \\    mov.u32 %r4, %ctaid.x;
    \\    mov.u32 %r5, %ntid.x;
    \\    mov.u32 %r6, %tid.x;
    \\    mad.lo.u32 %r7, %r4, %r5, %r6;
    \\    setp.ge.u32 %p1, %r7, %r1;
    \\    @%p1 bra DONE;
    \\    shr.u32 %r8, %r7, 2;
    \\    and.b32 %r9, %r7, 3;
    \\    mov.u32 %r10, 0;
    \\    mov.u32 %r11, 0;
    \\    mov.u32 %r12, 0;
    \\    mov.u32 %r13, %r2;
    \\    mov.u32 %r14, %r3;
    \\    mov.u32 %r15, 0;
    \\LOOP:
    \\    mul.lo.u32 %r16, %r8, 0xD2511F53;
    \\    mul.hi.u32 %r17, %r8, 0xD2511F53;
    \\    mul.lo.u32 %r18, %r11, 0xCD9E8D57;
    \\    mul.hi.u32 %r19, %r11, 0xCD9E8D57;
    \\    xor.b32 %r20, %r19, %r10;
    \\    xor.b32 %r20, %r20, %r13;
    \\    xor.b32 %r21, %r17, %r12;
    \\    xor.b32 %r21, %r21, %r14;
    \\    mov.u32 %r8, %r20;
    \\    mov.u32 %r10, %r18;
    \\    mov.u32 %r11, %r21;
    \\    mov.u32 %r12, %r16;
    \\    add.u32 %r13, %r13, 0x9E3779B9;
    \\    add.u32 %r14, %r14, 0xBB67AE85;
    \\    add.u32 %r15, %r15, 1;
    \\    setp.lt.u32 %p2, %r15, 10;
    \\    @%p2 bra LOOP;
    \\    setp.eq.u32 %p3, %r9, 0;
    \\    @%p3 mov.u32 %r22, %r8;
    \\    setp.eq.u32 %p4, %r9, 1;
    \\    @%p4 mov.u32 %r22, %r10;
    \\    setp.eq.u32 %p5, %r9, 2;
    \\    @%p5 mov.u32 %r22, %r11;
    \\    setp.eq.u32 %p6, %r9, 3;
    \\    @%p6 mov.u32 %r22, %r12;
    \\    shr.u32 %r23, %r22, 8;
    \\    cvt.rn.f32.u32 %f1, %r23;
    \\    mul.rn.f32 %f2, %f1, 0f33800000;
    \\    cvt.f64.f32 %fd1, %f2;
    \\    mul.wide.u32 %rd2, %r7, 8;
    \\    add.u64 %rd3, %rd1, %rd2;
    \\    st.global.f64 [%rd3], %fd1;
    \\DONE:
    \\    ret;
    \\}
;

const cached_philox_uniform_f16_ptx =
    \\.version 7.8
    \\.target sm_70
    \\.address_size 64
    \\
    \\.visible .entry vectra_philox_uniform_f16(
    \\    .param .u64 out_ptr,
    \\    .param .u32 n,
    \\    .param .u32 seed_lo,
    \\    .param .u32 seed_hi
    \\) {
    \\    .reg .pred %p<8>;
    \\    .reg .b32 %r<80>;
    \\    .reg .b64 %rd<6>;
    \\    .reg .f32 %f<3>;
    \\    .reg .b16 %h<2>;
    \\    ld.param.u64 %rd1, [out_ptr];
    \\    ld.param.u32 %r1, [n];
    \\    ld.param.u32 %r2, [seed_lo];
    \\    ld.param.u32 %r3, [seed_hi];
    \\    mov.u32 %r4, %ctaid.x;
    \\    mov.u32 %r5, %ntid.x;
    \\    mov.u32 %r6, %tid.x;
    \\    mad.lo.u32 %r7, %r4, %r5, %r6;
    \\    setp.ge.u32 %p1, %r7, %r1;
    \\    @%p1 bra DONE;
    \\    shr.u32 %r8, %r7, 2;
    \\    and.b32 %r9, %r7, 3;
    \\    mov.u32 %r10, 0;
    \\    mov.u32 %r11, 0;
    \\    mov.u32 %r12, 0;
    \\    mov.u32 %r13, %r2;
    \\    mov.u32 %r14, %r3;
    \\    mov.u32 %r15, 0;
    \\LOOP:
    \\    mul.lo.u32 %r16, %r8, 0xD2511F53;
    \\    mul.hi.u32 %r17, %r8, 0xD2511F53;
    \\    mul.lo.u32 %r18, %r11, 0xCD9E8D57;
    \\    mul.hi.u32 %r19, %r11, 0xCD9E8D57;
    \\    xor.b32 %r20, %r19, %r10;
    \\    xor.b32 %r20, %r20, %r13;
    \\    xor.b32 %r21, %r17, %r12;
    \\    xor.b32 %r21, %r21, %r14;
    \\    mov.u32 %r8, %r20;
    \\    mov.u32 %r10, %r18;
    \\    mov.u32 %r11, %r21;
    \\    mov.u32 %r12, %r16;
    \\    add.u32 %r13, %r13, 0x9E3779B9;
    \\    add.u32 %r14, %r14, 0xBB67AE85;
    \\    add.u32 %r15, %r15, 1;
    \\    setp.lt.u32 %p2, %r15, 10;
    \\    @%p2 bra LOOP;
    \\    setp.eq.u32 %p3, %r9, 0;
    \\    @%p3 mov.u32 %r22, %r8;
    \\    setp.eq.u32 %p4, %r9, 1;
    \\    @%p4 mov.u32 %r22, %r10;
    \\    setp.eq.u32 %p5, %r9, 2;
    \\    @%p5 mov.u32 %r22, %r11;
    \\    setp.eq.u32 %p6, %r9, 3;
    \\    @%p6 mov.u32 %r22, %r12;
    \\    shr.u32 %r23, %r22, 8;
    \\    cvt.rn.f32.u32 %f1, %r23;
    \\    mul.rn.f32 %f2, %f1, 0f33800000;
    \\    cvt.rn.f16.f32 %h1, %f2;
    \\    mul.wide.u32 %rd2, %r7, 2;
    \\    add.u64 %rd3, %rd1, %rd2;
    \\    st.global.b16 [%rd3], %h1;
    \\DONE:
    \\    ret;
    \\}
;

const cached_philox_uniform_bf16_ptx =
    \\.version 7.8
    \\.target sm_70
    \\.address_size 64
    \\
    \\.visible .entry vectra_philox_uniform_bf16(
    \\    .param .u64 out_ptr,
    \\    .param .u32 n,
    \\    .param .u32 seed_lo,
    \\    .param .u32 seed_hi
    \\) {
    \\    .reg .pred %p<8>;
    \\    .reg .b32 %r<80>;
    \\    .reg .b64 %rd<6>;
    \\    .reg .f32 %f<3>;
    \\    .reg .b16 %h<2>;
    \\    ld.param.u64 %rd1, [out_ptr];
    \\    ld.param.u32 %r1, [n];
    \\    ld.param.u32 %r2, [seed_lo];
    \\    ld.param.u32 %r3, [seed_hi];
    \\    mov.u32 %r4, %ctaid.x;
    \\    mov.u32 %r5, %ntid.x;
    \\    mov.u32 %r6, %tid.x;
    \\    mad.lo.u32 %r7, %r4, %r5, %r6;
    \\    setp.ge.u32 %p1, %r7, %r1;
    \\    @%p1 bra DONE;
    \\    shr.u32 %r8, %r7, 2;
    \\    and.b32 %r9, %r7, 3;
    \\    mov.u32 %r10, 0;
    \\    mov.u32 %r11, 0;
    \\    mov.u32 %r12, 0;
    \\    mov.u32 %r13, %r2;
    \\    mov.u32 %r14, %r3;
    \\    mov.u32 %r15, 0;
    \\LOOP:
    \\    mul.lo.u32 %r16, %r8, 0xD2511F53;
    \\    mul.hi.u32 %r17, %r8, 0xD2511F53;
    \\    mul.lo.u32 %r18, %r11, 0xCD9E8D57;
    \\    mul.hi.u32 %r19, %r11, 0xCD9E8D57;
    \\    xor.b32 %r20, %r19, %r10;
    \\    xor.b32 %r20, %r20, %r13;
    \\    xor.b32 %r21, %r17, %r12;
    \\    xor.b32 %r21, %r21, %r14;
    \\    mov.u32 %r8, %r20;
    \\    mov.u32 %r10, %r18;
    \\    mov.u32 %r11, %r21;
    \\    mov.u32 %r12, %r16;
    \\    add.u32 %r13, %r13, 0x9E3779B9;
    \\    add.u32 %r14, %r14, 0xBB67AE85;
    \\    add.u32 %r15, %r15, 1;
    \\    setp.lt.u32 %p2, %r15, 10;
    \\    @%p2 bra LOOP;
    \\    setp.eq.u32 %p3, %r9, 0;
    \\    @%p3 mov.u32 %r22, %r8;
    \\    setp.eq.u32 %p4, %r9, 1;
    \\    @%p4 mov.u32 %r22, %r10;
    \\    setp.eq.u32 %p5, %r9, 2;
    \\    @%p5 mov.u32 %r22, %r11;
    \\    setp.eq.u32 %p6, %r9, 3;
    \\    @%p6 mov.u32 %r22, %r12;
    \\    shr.u32 %r23, %r22, 8;
    \\    cvt.rn.f32.u32 %f1, %r23;
    \\    mul.rn.f32 %f2, %f1, 0f33800000;
    \\    mov.b32 %r24, %f2;
    \\    shr.u32 %r25, %r24, 16;
    \\    and.b32 %r26, %r25, 1;
    \\    add.u32 %r26, %r26, 0x7fff;
    \\    add.u32 %r24, %r24, %r26;
    \\    shr.u32 %r27, %r24, 16;
    \\    cvt.u16.u32 %h1, %r27;
    \\    mul.wide.u32 %rd2, %r7, 2;
    \\    add.u64 %rd3, %rd1, %rd2;
    \\    st.global.b16 [%rd3], %h1;
    \\DONE:
    \\    ret;
    \\}
;

pub fn fillPhiloxUniform(comptime T: type, storage: array_mod.DeviceStorage, seed: u64) array_mod.ArrayError!void {
    const ptx = comptime if (T == f32)
        cached_philox_uniform_f32_ptx
    else if (T == f64)
        cached_philox_uniform_f64_ptx
    else if (T == f16)
        cached_philox_uniform_f16_ptx
    else if (T == array_mod.BFloat16)
        cached_philox_uniform_bf16_ptx
    else
        @compileError("CUDA Philox uniform supports f32/f64/f16/BFloat16 arrays");
    const function_name = comptime if (T == f32)
        "vectra_philox_uniform_f32"
    else if (T == f64)
        "vectra_philox_uniform_f64"
    else if (T == f16)
        "vectra_philox_uniform_f16"
    else
        "vectra_philox_uniform_bf16";
    if (!build_options.enable_axiom_cuda or !storage.device.isCuda()) return error.InvalidDevice;
    if (storage.len == 0) return;
    if (storage.bytes != storage.len * @sizeOf(T)) return error.ShapeMismatch;
    if (storage.ptr == 0) return error.InvalidDevice;
    if (storage.len > std.math.maxInt(u32)) return error.InvalidShape;
    var session = withCudaContext(storage.device.index) catch return error.InvalidDevice;
    defer session.driver.close();
    defer session.context.release(&session.driver);
    const module = session.driver.moduleLoadData(ptx) catch return error.BackendFailure;
    defer session.driver.moduleUnload(module);
    const function = session.driver.moduleGetFunction(module, function_name) catch return error.BackendFailure;
    const threads: u32 = 256;
    const grid_x: u32 = @intCast((storage.len + threads - 1) / threads);
    var out_arg = storage.ptr;
    var n_arg: u32 = @intCast(storage.len);
    var seed_lo: u32 = @truncate(seed);
    var seed_hi: u32 = @truncate(seed >> 32);
    var args = [_]?*anyopaque{
        @ptrCast(&out_arg),
        @ptrCast(&n_arg),
        @ptrCast(&seed_lo),
        @ptrCast(&seed_hi),
    };
    session.driver.launchKernel(
        function,
        .{ .x = grid_x, .y = 1, .z = 1 },
        .{ .x = threads, .y = 1, .z = 1 },
        0,
        &args,
    ) catch return error.BackendFailure;
    session.driver.synchronize() catch return error.BackendFailure;
}

pub fn fillPhiloxNormal(comptime T: type, storage: array_mod.DeviceStorage, seed: u64, mean: T, stddev: T) array_mod.ArrayError!void {
    _ = storage;
    _ = seed;
    _ = mean;
    _ = stddev;
    return error.TypeUnsupported;
}

fn ptxFallbackNameForImage(file_name: []const u8, buffer: *[256]u8) ![]const u8 {
    if (std.mem.endsWith(u8, file_name, ".ptx")) return file_name;
    if (!std.mem.endsWith(u8, file_name, ".cubin")) return error.BackendFailure;
    const stem = file_name[0 .. file_name.len - ".cubin".len];
    return std.fmt.bufPrint(buffer, "{s}.ptx", .{stem});
}

fn readRuntimeImage(allocator: std.mem.Allocator, root_dir: []const u8, file_name: []const u8) ![:0]u8 {
    var runtime_threaded_io = std.Io.Threaded.init(allocator, .{});
    defer runtime_threaded_io.deinit();
    const runtime_io = runtime_threaded_io.io();
    var dir = if (std.fs.path.isAbsolute(root_dir))
        try std.Io.Dir.openDirAbsolute(runtime_io, root_dir, .{})
    else
        try std.Io.Dir.cwd().openDir(runtime_io, root_dir, .{});
    defer dir.close(runtime_io);
    return dir.readFileAllocOptions(runtime_io, file_name, allocator, .limited(64 * 1024 * 1024), .of(u8), 0);
}

pub const SmokeReport = struct {
    enabled: bool = build_options.enable_axiom_cuda,
    status: Status = if (build_options.enable_axiom_cuda) .skipped else .disabled,
    add_ok: bool = false,
    sub_ok: bool = false,
    mul_ok: bool = false,
    div_ok: bool = false,
    saxpy_ok: bool = false,
    matmul_ok: bool = false,
    matmul_tile_ir_ok: bool = false,
    f16_add_ok: bool = false,
    f16_matmul_ok: bool = false,
    bf16_add_ok: bool = false,
    bf16_matmul_ok: bool = false,
    typed_f16_gemm_plan: TypedGemmPlanEvidence = .{},
    typed_bf16_gemm_plan: TypedGemmPlanEvidence = .{},
    scalar_add_ok: bool = false,
    scalar_mul_ok: bool = false,
    scalar_saxpy_ok: bool = false,
    strided_add_ok: bool = false,
    strided_sub_ok: bool = false,
    strided_mul_ok: bool = false,
    strided_div_ok: bool = false,
    strided_abs_ok: bool = false,
    strided_sqrt_ok: bool = false,
    strided_exp_ok: bool = false,
    strided_log_ok: bool = false,
    strided_memref_legality_fingerprint: u64 = 0,
    strided_unary_memref_legality_fingerprint: u64 = 0,
    strided_scalar_memref_legality_fingerprint: u64 = 0,
    strided_scalar_add_ok: bool = false,
    strided_scalar_sub_ok: bool = false,
    strided_scalar_mul_ok: bool = false,
    strided_scalar_div_ok: bool = false,
    f64_strided_add_ok: bool = false,
    f64_strided_sub_ok: bool = false,
    f64_strided_mul_ok: bool = false,
    f64_strided_div_ok: bool = false,
    f64_strided_abs_ok: bool = false,
    f64_strided_sqrt_ok: bool = false,
    f64_strided_exp_ok: bool = false,
    f64_strided_memref_legality_fingerprint: u64 = 0,
    f64_strided_unary_memref_legality_fingerprint: u64 = 0,
    f64_strided_scalar_memref_legality_fingerprint: u64 = 0,
    f64_strided_scalar_add_ok: bool = false,
    f64_strided_scalar_sub_ok: bool = false,
    f64_strided_scalar_mul_ok: bool = false,
    f64_strided_scalar_div_ok: bool = false,
    f16_strided_add_ok: bool = false,
    f16_strided_sub_ok: bool = false,
    f16_strided_mul_ok: bool = false,
    f16_strided_div_ok: bool = false,
    f16_strided_abs_ok: bool = false,
    f16_strided_sqrt_ok: bool = false,
    f16_strided_exp_ok: bool = false,
    f16_strided_memref_legality_fingerprint: u64 = 0,
    f16_strided_unary_memref_legality_fingerprint: u64 = 0,
    f16_strided_scalar_memref_legality_fingerprint: u64 = 0,
    f16_strided_scalar_add_ok: bool = false,
    f16_strided_scalar_sub_ok: bool = false,
    f16_strided_scalar_mul_ok: bool = false,
    f16_strided_scalar_div_ok: bool = false,
    bf16_strided_add_ok: bool = false,
    bf16_strided_sub_ok: bool = false,
    bf16_strided_mul_ok: bool = false,
    bf16_strided_div_ok: bool = false,
    bf16_strided_abs_ok: bool = false,
    bf16_strided_sqrt_ok: bool = false,
    bf16_strided_exp_ok: bool = false,
    bf16_strided_memref_legality_fingerprint: u64 = 0,
    bf16_strided_unary_memref_legality_fingerprint: u64 = 0,
    bf16_strided_scalar_memref_legality_fingerprint: u64 = 0,
    bf16_strided_scalar_add_ok: bool = false,
    bf16_strided_scalar_sub_ok: bool = false,
    bf16_strided_scalar_mul_ok: bool = false,
    bf16_strided_scalar_div_ok: bool = false,
    device_array_ok: bool = false,
    max_abs_error: f32 = 0.0,
    lhs_plan: BufferPlanEvidence = .{},
    dtype_support_count: usize = 0,
    dtype_bridge_count: usize = 0,
    dtype_native_seed_count: usize = 0,
    dtype_widened_seed_count: usize = 0,
    dtype_support_fingerprint: u64 = 0,
    f16_native_execution_fingerprint: u64 = 0,
    bf16_native_execution_fingerprint: u64 = 0,
    f16_widened_execution_fingerprint: u64 = 0,
    bf16_widened_execution_fingerprint: u64 = 0,
    typed_f16_gemm_route_fingerprint: u64 = 0,
    typed_bf16_gemm_route_fingerprint: u64 = 0,
    typed_f16_gemm_route: []const u8 = "",
    typed_bf16_gemm_route: []const u8 = "",
    output_fingerprint: u64 = 0,
    issue_count: u8 = 0,

    pub fn ok(report: SmokeReport) bool {
        return report.issue_count == 0 and switch (report.status) {
            .disabled => !report.enabled,
            .skipped => report.enabled,
            .ran => report.enabled and report.executionEvidenceOk(),
            .failed => false,
        };
    }

    fn executionEvidenceOk(report: SmokeReport) bool {
        // Descriptor fingerprints are part of the executable contract, not just
        // diagnostics: without them a smoke run could regress to a legacy
        // stride-only bridge while still producing numerically correct outputs.
        return report.lhs_plan.ok and
            report.lhs_plan.copy_ok and
            report.add_ok and
            report.sub_ok and
            report.mul_ok and
            report.div_ok and
            report.saxpy_ok and
            report.matmul_ok and
            report.matmul_tile_ir_ok and
            report.f16_add_ok and
            report.f16_matmul_ok and
            report.bf16_add_ok and
            report.bf16_matmul_ok and
            report.typed_f16_gemm_plan.ok and
            report.typed_bf16_gemm_plan.ok and
            report.f16_widened_execution_fingerprint != 0 and
            report.bf16_widened_execution_fingerprint != 0 and
            report.typed_f16_gemm_route_fingerprint != 0 and
            report.typed_bf16_gemm_route_fingerprint != 0 and
            std.mem.eql(u8, report.typed_f16_gemm_route, "widened_f32_cuda_compute") and
            std.mem.eql(u8, report.typed_bf16_gemm_route, "widened_f32_cuda_compute") and
            report.scalar_add_ok and
            report.scalar_mul_ok and
            report.scalar_saxpy_ok and
            report.strided_add_ok and
            report.strided_sub_ok and
            report.strided_mul_ok and
            report.strided_div_ok and
            report.strided_abs_ok and
            report.strided_sqrt_ok and
            report.strided_exp_ok and
            report.strided_log_ok and
            report.strided_memref_legality_fingerprint != 0 and
            report.strided_unary_memref_legality_fingerprint != 0 and
            report.strided_scalar_memref_legality_fingerprint != 0 and
            report.strided_scalar_add_ok and
            report.strided_scalar_sub_ok and
            report.strided_scalar_mul_ok and
            report.strided_scalar_div_ok and
            report.f64_strided_add_ok and
            report.f64_strided_sub_ok and
            report.f64_strided_mul_ok and
            report.f64_strided_div_ok and
            report.f64_strided_abs_ok and
            report.f64_strided_sqrt_ok and
            report.f64_strided_exp_ok and
            report.f64_strided_memref_legality_fingerprint != 0 and
            report.f64_strided_unary_memref_legality_fingerprint != 0 and
            report.f64_strided_scalar_memref_legality_fingerprint != 0 and
            report.f64_strided_scalar_add_ok and
            report.f64_strided_scalar_sub_ok and
            report.f64_strided_scalar_mul_ok and
            report.f64_strided_scalar_div_ok and
            report.f16_strided_add_ok and
            report.f16_strided_sub_ok and
            report.f16_strided_mul_ok and
            report.f16_strided_div_ok and
            report.f16_strided_abs_ok and
            report.f16_strided_sqrt_ok and
            report.f16_strided_exp_ok and
            report.f16_strided_memref_legality_fingerprint != 0 and
            report.f16_strided_unary_memref_legality_fingerprint != 0 and
            report.f16_strided_scalar_memref_legality_fingerprint != 0 and
            report.f16_strided_scalar_add_ok and
            report.f16_strided_scalar_sub_ok and
            report.f16_strided_scalar_mul_ok and
            report.f16_strided_scalar_div_ok and
            report.bf16_strided_add_ok and
            report.bf16_strided_sub_ok and
            report.bf16_strided_mul_ok and
            report.bf16_strided_div_ok and
            report.bf16_strided_abs_ok and
            report.bf16_strided_sqrt_ok and
            report.bf16_strided_exp_ok and
            report.bf16_strided_memref_legality_fingerprint != 0 and
            report.bf16_strided_unary_memref_legality_fingerprint != 0 and
            report.bf16_strided_scalar_memref_legality_fingerprint != 0 and
            report.bf16_strided_scalar_add_ok and
            report.bf16_strided_scalar_sub_ok and
            report.bf16_strided_scalar_mul_ok and
            report.bf16_strided_scalar_div_ok;
    }

    fn executionIssueCount(report: SmokeReport) u8 {
        return @as(u8, @intFromBool(!report.lhs_plan.ok)) +
            @as(u8, @intFromBool(!report.lhs_plan.copy_ok)) +
            @as(u8, @intFromBool(!report.add_ok)) +
            @as(u8, @intFromBool(!report.sub_ok)) +
            @as(u8, @intFromBool(!report.mul_ok)) +
            @as(u8, @intFromBool(!report.div_ok)) +
            @as(u8, @intFromBool(!report.saxpy_ok)) +
            @as(u8, @intFromBool(!report.matmul_ok)) +
            @as(u8, @intFromBool(!report.matmul_tile_ir_ok)) +
            @as(u8, @intFromBool(!report.f16_add_ok)) +
            @as(u8, @intFromBool(!report.f16_matmul_ok)) +
            @as(u8, @intFromBool(!report.bf16_add_ok)) +
            @as(u8, @intFromBool(!report.bf16_matmul_ok)) +
            @as(u8, @intFromBool(!report.typed_f16_gemm_plan.ok)) +
            @as(u8, @intFromBool(!report.typed_bf16_gemm_plan.ok)) +
            @as(u8, @intFromBool(report.f16_widened_execution_fingerprint == 0)) +
            @as(u8, @intFromBool(report.bf16_widened_execution_fingerprint == 0)) +
            @as(u8, @intFromBool(report.typed_f16_gemm_route_fingerprint == 0)) +
            @as(u8, @intFromBool(report.typed_bf16_gemm_route_fingerprint == 0)) +
            @as(u8, @intFromBool(!std.mem.eql(u8, report.typed_f16_gemm_route, "widened_f32_cuda_compute"))) +
            @as(u8, @intFromBool(!std.mem.eql(u8, report.typed_bf16_gemm_route, "widened_f32_cuda_compute"))) +
            @as(u8, @intFromBool(!report.scalar_add_ok)) +
            @as(u8, @intFromBool(!report.scalar_mul_ok)) +
            @as(u8, @intFromBool(!report.scalar_saxpy_ok)) +
            @as(u8, @intFromBool(!report.strided_add_ok)) +
            @as(u8, @intFromBool(!report.strided_sub_ok)) +
            @as(u8, @intFromBool(!report.strided_mul_ok)) +
            @as(u8, @intFromBool(!report.strided_div_ok)) +
            @as(u8, @intFromBool(!report.strided_abs_ok)) +
            @as(u8, @intFromBool(!report.strided_sqrt_ok)) +
            @as(u8, @intFromBool(!report.strided_exp_ok)) +
            @as(u8, @intFromBool(!report.strided_log_ok)) +
            @as(u8, @intFromBool(report.strided_memref_legality_fingerprint == 0)) +
            @as(u8, @intFromBool(report.strided_unary_memref_legality_fingerprint == 0)) +
            @as(u8, @intFromBool(report.strided_scalar_memref_legality_fingerprint == 0)) +
            @as(u8, @intFromBool(!report.strided_scalar_add_ok)) +
            @as(u8, @intFromBool(!report.strided_scalar_sub_ok)) +
            @as(u8, @intFromBool(!report.strided_scalar_mul_ok)) +
            @as(u8, @intFromBool(!report.strided_scalar_div_ok)) +
            @as(u8, @intFromBool(!report.f64_strided_add_ok)) +
            @as(u8, @intFromBool(!report.f64_strided_sub_ok)) +
            @as(u8, @intFromBool(!report.f64_strided_mul_ok)) +
            @as(u8, @intFromBool(!report.f64_strided_div_ok)) +
            @as(u8, @intFromBool(!report.f64_strided_abs_ok)) +
            @as(u8, @intFromBool(!report.f64_strided_sqrt_ok)) +
            @as(u8, @intFromBool(!report.f64_strided_exp_ok)) +
            @as(u8, @intFromBool(report.f64_strided_memref_legality_fingerprint == 0)) +
            @as(u8, @intFromBool(report.f64_strided_unary_memref_legality_fingerprint == 0)) +
            @as(u8, @intFromBool(report.f64_strided_scalar_memref_legality_fingerprint == 0)) +
            @as(u8, @intFromBool(!report.f64_strided_scalar_add_ok)) +
            @as(u8, @intFromBool(!report.f64_strided_scalar_sub_ok)) +
            @as(u8, @intFromBool(!report.f64_strided_scalar_mul_ok)) +
            @as(u8, @intFromBool(!report.f64_strided_scalar_div_ok)) +
            @as(u8, @intFromBool(!report.f16_strided_add_ok)) +
            @as(u8, @intFromBool(!report.f16_strided_sub_ok)) +
            @as(u8, @intFromBool(!report.f16_strided_mul_ok)) +
            @as(u8, @intFromBool(!report.f16_strided_div_ok)) +
            @as(u8, @intFromBool(!report.f16_strided_abs_ok)) +
            @as(u8, @intFromBool(!report.f16_strided_sqrt_ok)) +
            @as(u8, @intFromBool(!report.f16_strided_exp_ok)) +
            @as(u8, @intFromBool(report.f16_strided_memref_legality_fingerprint == 0)) +
            @as(u8, @intFromBool(report.f16_strided_unary_memref_legality_fingerprint == 0)) +
            @as(u8, @intFromBool(report.f16_strided_scalar_memref_legality_fingerprint == 0)) +
            @as(u8, @intFromBool(!report.f16_strided_scalar_add_ok)) +
            @as(u8, @intFromBool(!report.f16_strided_scalar_sub_ok)) +
            @as(u8, @intFromBool(!report.f16_strided_scalar_mul_ok)) +
            @as(u8, @intFromBool(!report.f16_strided_scalar_div_ok)) +
            @as(u8, @intFromBool(!report.bf16_strided_add_ok)) +
            @as(u8, @intFromBool(!report.bf16_strided_sub_ok)) +
            @as(u8, @intFromBool(!report.bf16_strided_mul_ok)) +
            @as(u8, @intFromBool(!report.bf16_strided_div_ok)) +
            @as(u8, @intFromBool(!report.bf16_strided_abs_ok)) +
            @as(u8, @intFromBool(!report.bf16_strided_sqrt_ok)) +
            @as(u8, @intFromBool(!report.bf16_strided_exp_ok)) +
            @as(u8, @intFromBool(report.bf16_strided_memref_legality_fingerprint == 0)) +
            @as(u8, @intFromBool(report.bf16_strided_unary_memref_legality_fingerprint == 0)) +
            @as(u8, @intFromBool(report.bf16_strided_scalar_memref_legality_fingerprint == 0)) +
            @as(u8, @intFromBool(!report.bf16_strided_scalar_add_ok)) +
            @as(u8, @intFromBool(!report.bf16_strided_scalar_sub_ok)) +
            @as(u8, @intFromBool(!report.bf16_strided_scalar_mul_ok)) +
            @as(u8, @intFromBool(!report.bf16_strided_scalar_div_ok));
    }

    pub fn fingerprint(report: SmokeReport) u64 {
        var hasher = std.hash.Wyhash.init(0x0abc_7aaa_11cc_0001);
        hashBool(&hasher, report.enabled);
        hashBytes(&hasher, report.status.label());
        hashBool(&hasher, report.add_ok);
        hashBool(&hasher, report.sub_ok);
        hashBool(&hasher, report.mul_ok);
        hashBool(&hasher, report.div_ok);
        hashBool(&hasher, report.saxpy_ok);
        hashBool(&hasher, report.matmul_ok);
        hashBool(&hasher, report.matmul_tile_ir_ok);
        hashBool(&hasher, report.f16_add_ok);
        hashBool(&hasher, report.f16_matmul_ok);
        hashBool(&hasher, report.bf16_add_ok);
        hashBool(&hasher, report.bf16_matmul_ok);
        hashU64(&hasher, report.typed_f16_gemm_plan.fingerprint());
        hashU64(&hasher, report.typed_bf16_gemm_plan.fingerprint());
        hashBool(&hasher, report.scalar_add_ok);
        hashBool(&hasher, report.scalar_mul_ok);
        hashBool(&hasher, report.scalar_saxpy_ok);
        hashBool(&hasher, report.strided_add_ok);
        hashBool(&hasher, report.strided_sub_ok);
        hashBool(&hasher, report.strided_mul_ok);
        hashBool(&hasher, report.strided_div_ok);
        hashBool(&hasher, report.strided_abs_ok);
        hashBool(&hasher, report.strided_sqrt_ok);
        hashBool(&hasher, report.strided_exp_ok);
        hashBool(&hasher, report.strided_log_ok);
        hashU64(&hasher, report.strided_memref_legality_fingerprint);
        hashU64(&hasher, report.strided_unary_memref_legality_fingerprint);
        hashU64(&hasher, report.strided_scalar_memref_legality_fingerprint);
        hashBool(&hasher, report.strided_scalar_add_ok);
        hashBool(&hasher, report.strided_scalar_sub_ok);
        hashBool(&hasher, report.strided_scalar_mul_ok);
        hashBool(&hasher, report.strided_scalar_div_ok);
        hashBool(&hasher, report.f64_strided_add_ok);
        hashBool(&hasher, report.f64_strided_sub_ok);
        hashBool(&hasher, report.f64_strided_mul_ok);
        hashBool(&hasher, report.f64_strided_div_ok);
        hashBool(&hasher, report.f64_strided_abs_ok);
        hashBool(&hasher, report.f64_strided_sqrt_ok);
        hashBool(&hasher, report.f64_strided_exp_ok);
        hashU64(&hasher, report.f64_strided_memref_legality_fingerprint);
        hashU64(&hasher, report.f64_strided_unary_memref_legality_fingerprint);
        hashU64(&hasher, report.f64_strided_scalar_memref_legality_fingerprint);
        hashBool(&hasher, report.f64_strided_scalar_add_ok);
        hashBool(&hasher, report.f64_strided_scalar_sub_ok);
        hashBool(&hasher, report.f64_strided_scalar_mul_ok);
        hashBool(&hasher, report.f64_strided_scalar_div_ok);
        hashBool(&hasher, report.f16_strided_add_ok);
        hashBool(&hasher, report.f16_strided_sub_ok);
        hashBool(&hasher, report.f16_strided_mul_ok);
        hashBool(&hasher, report.f16_strided_div_ok);
        hashBool(&hasher, report.f16_strided_abs_ok);
        hashBool(&hasher, report.f16_strided_sqrt_ok);
        hashBool(&hasher, report.f16_strided_exp_ok);
        hashU64(&hasher, report.f16_strided_memref_legality_fingerprint);
        hashU64(&hasher, report.f16_strided_unary_memref_legality_fingerprint);
        hashU64(&hasher, report.f16_strided_scalar_memref_legality_fingerprint);
        hashBool(&hasher, report.f16_strided_scalar_add_ok);
        hashBool(&hasher, report.f16_strided_scalar_sub_ok);
        hashBool(&hasher, report.f16_strided_scalar_mul_ok);
        hashBool(&hasher, report.f16_strided_scalar_div_ok);
        hashBool(&hasher, report.bf16_strided_add_ok);
        hashBool(&hasher, report.bf16_strided_sub_ok);
        hashBool(&hasher, report.bf16_strided_mul_ok);
        hashBool(&hasher, report.bf16_strided_div_ok);
        hashBool(&hasher, report.bf16_strided_abs_ok);
        hashBool(&hasher, report.bf16_strided_sqrt_ok);
        hashBool(&hasher, report.bf16_strided_exp_ok);
        hashU64(&hasher, report.bf16_strided_memref_legality_fingerprint);
        hashU64(&hasher, report.bf16_strided_unary_memref_legality_fingerprint);
        hashU64(&hasher, report.bf16_strided_scalar_memref_legality_fingerprint);
        hashBool(&hasher, report.bf16_strided_scalar_add_ok);
        hashBool(&hasher, report.bf16_strided_scalar_sub_ok);
        hashBool(&hasher, report.bf16_strided_scalar_mul_ok);
        hashBool(&hasher, report.bf16_strided_scalar_div_ok);
        hashBool(&hasher, report.device_array_ok);
        hashF32(&hasher, report.max_abs_error);
        hashBool(&hasher, report.lhs_plan.ok);
        hashU64(&hasher, report.lhs_plan.logical_elements);
        hashU64(&hasher, report.lhs_plan.required_span);
        hashU64(&hasher, report.lhs_plan.logical_bytes);
        hashU64(&hasher, report.lhs_plan.required_bytes);
        hashBool(&hasher, report.lhs_plan.linear_copy);
        hashU64(&hasher, report.lhs_plan.fingerprint);
        hashBool(&hasher, report.lhs_plan.copy_ok);
        hashBool(&hasher, report.lhs_plan.copy_requires_strided);
        hashU64(&hasher, report.lhs_plan.copy_fingerprint);
        hashU64(&hasher, report.dtype_support_count);
        hashU64(&hasher, report.dtype_bridge_count);
        hashU64(&hasher, report.dtype_native_seed_count);
        hashU64(&hasher, report.dtype_widened_seed_count);
        hashU64(&hasher, report.dtype_support_fingerprint);
        hashU64(&hasher, report.f16_native_execution_fingerprint);
        hashU64(&hasher, report.bf16_native_execution_fingerprint);
        hashU64(&hasher, report.f16_widened_execution_fingerprint);
        hashU64(&hasher, report.bf16_widened_execution_fingerprint);
        hashU64(&hasher, report.typed_f16_gemm_route_fingerprint);
        hashU64(&hasher, report.typed_bf16_gemm_route_fingerprint);
        hashBytes(&hasher, report.typed_f16_gemm_route);
        hashBytes(&hasher, report.typed_bf16_gemm_route);
        hashU64(&hasher, report.output_fingerprint);
        hashU64(&hasher, report.issue_count);
        return hasher.final();
    }

    pub fn writeText(report: SmokeReport, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        try writer.print(
            "vectra_axiom_cuda_smoke enabled={} status={s} ok={} issues={d} add={} sub={} mul={} div={} saxpy={} matmul={} matmul_tile_ir={} f16_add={} f16_matmul={} bf16_add={} bf16_matmul={} typed_f16_gemm={} typed_bf16_gemm={} scalar_add={} scalar_mul={} scalar_saxpy={} device_array={} max_abs_error={d}\n",
            .{
                report.enabled,
                report.status.label(),
                report.ok(),
                report.issue_count,
                report.add_ok,
                report.sub_ok,
                report.mul_ok,
                report.div_ok,
                report.saxpy_ok,
                report.matmul_ok,
                report.matmul_tile_ir_ok,
                report.f16_add_ok,
                report.f16_matmul_ok,
                report.bf16_add_ok,
                report.bf16_matmul_ok,
                report.typed_f16_gemm_plan.ok,
                report.typed_bf16_gemm_plan.ok,
                report.scalar_add_ok,
                report.scalar_mul_ok,
                report.scalar_saxpy_ok,
                report.device_array_ok,
                report.max_abs_error,
            },
        );
        try writer.print(
            "vectra_axiom_cuda_strided_f32_f64 strided_add={} strided_sub={} strided_mul={} strided_div={} strided_abs={} strided_sqrt={} strided_exp={} strided_log={} strided_memref={x} strided_unary_memref={x} strided_scalar_memref={x} strided_scalar_add={} strided_scalar_sub={} strided_scalar_mul={} strided_scalar_div={} f64_strided_add={} f64_strided_sub={} f64_strided_mul={} f64_strided_div={} f64_strided_abs={} f64_strided_sqrt={} f64_strided_exp={} f64_strided_memref={x} f64_strided_unary_memref={x} f64_strided_scalar_memref={x} f64_strided_scalar_add={} f64_strided_scalar_sub={} f64_strided_scalar_mul={} f64_strided_scalar_div={}\n",
            .{
                report.strided_add_ok,
                report.strided_sub_ok,
                report.strided_mul_ok,
                report.strided_div_ok,
                report.strided_abs_ok,
                report.strided_sqrt_ok,
                report.strided_exp_ok,
                report.strided_log_ok,
                report.strided_memref_legality_fingerprint,
                report.strided_unary_memref_legality_fingerprint,
                report.strided_scalar_memref_legality_fingerprint,
                report.strided_scalar_add_ok,
                report.strided_scalar_sub_ok,
                report.strided_scalar_mul_ok,
                report.strided_scalar_div_ok,
                report.f64_strided_add_ok,
                report.f64_strided_sub_ok,
                report.f64_strided_mul_ok,
                report.f64_strided_div_ok,
                report.f64_strided_abs_ok,
                report.f64_strided_sqrt_ok,
                report.f64_strided_exp_ok,
                report.f64_strided_memref_legality_fingerprint,
                report.f64_strided_unary_memref_legality_fingerprint,
                report.f64_strided_scalar_memref_legality_fingerprint,
                report.f64_strided_scalar_add_ok,
                report.f64_strided_scalar_sub_ok,
                report.f64_strided_scalar_mul_ok,
                report.f64_strided_scalar_div_ok,
            },
        );
        try writer.print(
            "vectra_axiom_cuda_strided_half f16_strided_add={} f16_strided_sub={} f16_strided_mul={} f16_strided_div={} f16_strided_abs={} f16_strided_sqrt={} f16_strided_exp={} f16_strided_memref={x} f16_strided_unary_memref={x} f16_strided_scalar_memref={x} f16_strided_scalar_add={} f16_strided_scalar_sub={} f16_strided_scalar_mul={} f16_strided_scalar_div={} bf16_strided_add={} bf16_strided_sub={} bf16_strided_mul={} bf16_strided_div={} bf16_strided_abs={} bf16_strided_sqrt={} bf16_strided_exp={} bf16_strided_memref={x} bf16_strided_unary_memref={x} bf16_strided_scalar_memref={x} bf16_strided_scalar_add={} bf16_strided_scalar_sub={} bf16_strided_scalar_mul={} bf16_strided_scalar_div={}\n",
            .{
                report.f16_strided_add_ok,
                report.f16_strided_sub_ok,
                report.f16_strided_mul_ok,
                report.f16_strided_div_ok,
                report.f16_strided_abs_ok,
                report.f16_strided_sqrt_ok,
                report.f16_strided_exp_ok,
                report.f16_strided_memref_legality_fingerprint,
                report.f16_strided_unary_memref_legality_fingerprint,
                report.f16_strided_scalar_memref_legality_fingerprint,
                report.f16_strided_scalar_add_ok,
                report.f16_strided_scalar_sub_ok,
                report.f16_strided_scalar_mul_ok,
                report.f16_strided_scalar_div_ok,
                report.bf16_strided_add_ok,
                report.bf16_strided_sub_ok,
                report.bf16_strided_mul_ok,
                report.bf16_strided_div_ok,
                report.bf16_strided_abs_ok,
                report.bf16_strided_sqrt_ok,
                report.bf16_strided_exp_ok,
                report.bf16_strided_memref_legality_fingerprint,
                report.bf16_strided_unary_memref_legality_fingerprint,
                report.bf16_strided_scalar_memref_legality_fingerprint,
                report.bf16_strided_scalar_add_ok,
                report.bf16_strided_scalar_sub_ok,
                report.bf16_strided_scalar_mul_ok,
                report.bf16_strided_scalar_div_ok,
            },
        );
        try writer.print(
            "vectra_axiom_cuda_buffers logical_elements={d} required_bytes={d} linear_copy={} copy_plan_ok={} copy_requires_strided={} output={x} fingerprint={x}\n",
            .{
                report.lhs_plan.logical_elements,
                report.lhs_plan.required_bytes,
                report.lhs_plan.linear_copy,
                report.lhs_plan.copy_ok,
                report.lhs_plan.copy_requires_strided,
                report.output_fingerprint,
                report.fingerprint(),
            },
        );
        try writer.print(
            "vectra_axiom_cuda_dtype_support count={d} bridge={d} native_seed={d} widened_seed={d} fingerprint={x} f16_native_execution={x} bf16_native_execution={x} f16_widened_execution={x} bf16_widened_execution={x} typed_f16_gemm={x} typed_bf16_gemm={x} typed_f16_route={s} typed_bf16_route={s}\n",
            .{
                report.dtype_support_count,
                report.dtype_bridge_count,
                report.dtype_native_seed_count,
                report.dtype_widened_seed_count,
                report.dtype_support_fingerprint,
                report.f16_native_execution_fingerprint,
                report.bf16_native_execution_fingerprint,
                report.f16_widened_execution_fingerprint,
                report.bf16_widened_execution_fingerprint,
                report.typed_f16_gemm_plan.fingerprint(),
                report.typed_bf16_gemm_plan.fingerprint(),
                report.typed_f16_gemm_route,
                report.typed_bf16_gemm_route,
            },
        );
    }

    pub fn writeJson(report: SmokeReport, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        try writer.print(
            "{{\n" ++
                "  \"kind\": \"vectra_axiom_cuda_smoke\",\n" ++
                "  \"enabled\": {},\n" ++
                "  \"status\": \"{s}\",\n" ++
                "  \"ok\": {},\n" ++
                "  \"issue_count\": {d},\n" ++
                "  \"add_ok\": {},\n" ++
                "  \"sub_ok\": {},\n" ++
                "  \"mul_ok\": {},\n" ++
                "  \"div_ok\": {},\n" ++
                "  \"saxpy_ok\": {},\n" ++
                "  \"matmul_ok\": {},\n" ++
                "  \"matmul_tile_ir_ok\": {},\n" ++
                "  \"f16_add_ok\": {},\n" ++
                "  \"f16_matmul_ok\": {},\n" ++
                "  \"bf16_add_ok\": {},\n" ++
                "  \"bf16_matmul_ok\": {},\n" ++
                "  \"typed_f16_gemm_plan_ok\": {},\n" ++
                "  \"typed_bf16_gemm_plan_ok\": {},\n",
            .{
                report.enabled,
                report.status.label(),
                report.ok(),
                report.issue_count,
                report.add_ok,
                report.sub_ok,
                report.mul_ok,
                report.div_ok,
                report.saxpy_ok,
                report.matmul_ok,
                report.matmul_tile_ir_ok,
                report.f16_add_ok,
                report.f16_matmul_ok,
                report.bf16_add_ok,
                report.bf16_matmul_ok,
                report.typed_f16_gemm_plan.ok,
                report.typed_bf16_gemm_plan.ok,
            },
        );
        try writer.print(
            "  \"typed_f16_gemm_element\": \"{s}\",\n" ++
                "  \"typed_f16_gemm_readiness\": \"{s}\",\n" ++
                "  \"typed_f16_gemm_m\": {d},\n" ++
                "  \"typed_f16_gemm_n\": {d},\n" ++
                "  \"typed_f16_gemm_k\": {d},\n" ++
                "  \"typed_f16_gemm_tile_m\": {d},\n" ++
                "  \"typed_f16_gemm_tile_n\": {d},\n" ++
                "  \"typed_f16_gemm_tile_k\": {d},\n" ++
                "  \"typed_f16_gemm_grid_m\": {d},\n" ++
                "  \"typed_f16_gemm_grid_n\": {d},\n" ++
                "  \"typed_f16_gemm_total_ctas\": {d},\n" ++
                "  \"typed_f16_gemm_threads_per_cta\": {d},\n" ++
                "  \"typed_f16_gemm_argument_bytes\": {d},\n" ++
                "  \"typed_f16_gemm_runtime_route\": \"{s}\",\n" ++
                "  \"typed_f16_gemm_runtime_route_fingerprint\": {d},\n",
            .{
                report.typed_f16_gemm_plan.element_name,
                report.typed_f16_gemm_plan.readiness_status,
                report.typed_f16_gemm_plan.m,
                report.typed_f16_gemm_plan.n,
                report.typed_f16_gemm_plan.k,
                report.typed_f16_gemm_plan.tile_m,
                report.typed_f16_gemm_plan.tile_n,
                report.typed_f16_gemm_plan.tile_k,
                report.typed_f16_gemm_plan.grid_m,
                report.typed_f16_gemm_plan.grid_n,
                report.typed_f16_gemm_plan.total_ctas,
                report.typed_f16_gemm_plan.threads_per_cta,
                report.typed_f16_gemm_plan.argument_bytes,
                report.typed_f16_gemm_route,
                report.typed_f16_gemm_route_fingerprint,
            },
        );
        try writer.print(
            "  \"typed_bf16_gemm_element\": \"{s}\",\n" ++
                "  \"typed_bf16_gemm_readiness\": \"{s}\",\n" ++
                "  \"typed_bf16_gemm_m\": {d},\n" ++
                "  \"typed_bf16_gemm_n\": {d},\n" ++
                "  \"typed_bf16_gemm_k\": {d},\n" ++
                "  \"typed_bf16_gemm_tile_m\": {d},\n" ++
                "  \"typed_bf16_gemm_tile_n\": {d},\n" ++
                "  \"typed_bf16_gemm_tile_k\": {d},\n" ++
                "  \"typed_bf16_gemm_grid_m\": {d},\n" ++
                "  \"typed_bf16_gemm_grid_n\": {d},\n" ++
                "  \"typed_bf16_gemm_total_ctas\": {d},\n" ++
                "  \"typed_bf16_gemm_threads_per_cta\": {d},\n" ++
                "  \"typed_bf16_gemm_argument_bytes\": {d},\n" ++
                "  \"typed_bf16_gemm_runtime_route\": \"{s}\",\n" ++
                "  \"typed_bf16_gemm_runtime_route_fingerprint\": {d},\n",
            .{
                report.typed_bf16_gemm_plan.element_name,
                report.typed_bf16_gemm_plan.readiness_status,
                report.typed_bf16_gemm_plan.m,
                report.typed_bf16_gemm_plan.n,
                report.typed_bf16_gemm_plan.k,
                report.typed_bf16_gemm_plan.tile_m,
                report.typed_bf16_gemm_plan.tile_n,
                report.typed_bf16_gemm_plan.tile_k,
                report.typed_bf16_gemm_plan.grid_m,
                report.typed_bf16_gemm_plan.grid_n,
                report.typed_bf16_gemm_plan.total_ctas,
                report.typed_bf16_gemm_plan.threads_per_cta,
                report.typed_bf16_gemm_plan.argument_bytes,
                report.typed_bf16_gemm_route,
                report.typed_bf16_gemm_route_fingerprint,
            },
        );
        try writer.print(
            "  \"scalar_add_ok\": {},\n" ++
                "  \"scalar_mul_ok\": {},\n" ++
                "  \"scalar_saxpy_ok\": {},\n" ++
                "  \"strided_add_ok\": {},\n" ++
                "  \"strided_sub_ok\": {},\n" ++
                "  \"strided_mul_ok\": {},\n" ++
                "  \"strided_div_ok\": {},\n" ++
                "  \"strided_abs_ok\": {},\n" ++
                "  \"strided_sqrt_ok\": {},\n" ++
                "  \"strided_exp_ok\": {},\n" ++
                "  \"strided_log_ok\": {},\n" ++
                "  \"strided_memref_legality_fingerprint\": {d},\n" ++
                "  \"strided_unary_memref_legality_fingerprint\": {d},\n" ++
                "  \"strided_scalar_memref_legality_fingerprint\": {d},\n" ++
                "  \"strided_scalar_add_ok\": {},\n" ++
                "  \"strided_scalar_sub_ok\": {},\n" ++
                "  \"strided_scalar_mul_ok\": {},\n" ++
                "  \"strided_scalar_div_ok\": {},\n" ++
                "  \"f64_strided_add_ok\": {},\n" ++
                "  \"f64_strided_sub_ok\": {},\n" ++
                "  \"f64_strided_mul_ok\": {},\n" ++
                "  \"f64_strided_div_ok\": {},\n" ++
                "  \"f64_strided_abs_ok\": {},\n" ++
                "  \"f64_strided_sqrt_ok\": {},\n" ++
                "  \"f64_strided_exp_ok\": {},\n" ++
                "  \"f64_strided_memref_legality_fingerprint\": {d},\n" ++
                "  \"f64_strided_unary_memref_legality_fingerprint\": {d},\n" ++
                "  \"f64_strided_scalar_memref_legality_fingerprint\": {d},\n" ++
                "  \"f64_strided_scalar_add_ok\": {},\n" ++
                "  \"f64_strided_scalar_sub_ok\": {},\n" ++
                "  \"f64_strided_scalar_mul_ok\": {},\n" ++
                "  \"f64_strided_scalar_div_ok\": {},\n",
            .{
                report.scalar_add_ok,
                report.scalar_mul_ok,
                report.scalar_saxpy_ok,
                report.strided_add_ok,
                report.strided_sub_ok,
                report.strided_mul_ok,
                report.strided_div_ok,
                report.strided_abs_ok,
                report.strided_sqrt_ok,
                report.strided_exp_ok,
                report.strided_log_ok,
                report.strided_memref_legality_fingerprint,
                report.strided_unary_memref_legality_fingerprint,
                report.strided_scalar_memref_legality_fingerprint,
                report.strided_scalar_add_ok,
                report.strided_scalar_sub_ok,
                report.strided_scalar_mul_ok,
                report.strided_scalar_div_ok,
                report.f64_strided_add_ok,
                report.f64_strided_sub_ok,
                report.f64_strided_mul_ok,
                report.f64_strided_div_ok,
                report.f64_strided_abs_ok,
                report.f64_strided_sqrt_ok,
                report.f64_strided_exp_ok,
                report.f64_strided_memref_legality_fingerprint,
                report.f64_strided_unary_memref_legality_fingerprint,
                report.f64_strided_scalar_memref_legality_fingerprint,
                report.f64_strided_scalar_add_ok,
                report.f64_strided_scalar_sub_ok,
                report.f64_strided_scalar_mul_ok,
                report.f64_strided_scalar_div_ok,
            },
        );
        try writer.print(
            "  \"f16_strided_add_ok\": {},\n" ++
                "  \"f16_strided_sub_ok\": {},\n" ++
                "  \"f16_strided_mul_ok\": {},\n" ++
                "  \"f16_strided_div_ok\": {},\n" ++
                "  \"f16_strided_abs_ok\": {},\n" ++
                "  \"f16_strided_sqrt_ok\": {},\n" ++
                "  \"f16_strided_exp_ok\": {},\n" ++
                "  \"f16_strided_memref_legality_fingerprint\": {d},\n" ++
                "  \"f16_strided_unary_memref_legality_fingerprint\": {d},\n" ++
                "  \"f16_strided_scalar_memref_legality_fingerprint\": {d},\n" ++
                "  \"f16_strided_scalar_add_ok\": {},\n" ++
                "  \"f16_strided_scalar_sub_ok\": {},\n" ++
                "  \"f16_strided_scalar_mul_ok\": {},\n" ++
                "  \"f16_strided_scalar_div_ok\": {},\n" ++
                "  \"bf16_strided_add_ok\": {},\n" ++
                "  \"bf16_strided_sub_ok\": {},\n" ++
                "  \"bf16_strided_mul_ok\": {},\n" ++
                "  \"bf16_strided_div_ok\": {},\n" ++
                "  \"bf16_strided_abs_ok\": {},\n" ++
                "  \"bf16_strided_sqrt_ok\": {},\n" ++
                "  \"bf16_strided_exp_ok\": {},\n",
            .{
                report.f16_strided_add_ok,
                report.f16_strided_sub_ok,
                report.f16_strided_mul_ok,
                report.f16_strided_div_ok,
                report.f16_strided_abs_ok,
                report.f16_strided_sqrt_ok,
                report.f16_strided_exp_ok,
                report.f16_strided_memref_legality_fingerprint,
                report.f16_strided_unary_memref_legality_fingerprint,
                report.f16_strided_scalar_memref_legality_fingerprint,
                report.f16_strided_scalar_add_ok,
                report.f16_strided_scalar_sub_ok,
                report.f16_strided_scalar_mul_ok,
                report.f16_strided_scalar_div_ok,
                report.bf16_strided_add_ok,
                report.bf16_strided_sub_ok,
                report.bf16_strided_mul_ok,
                report.bf16_strided_div_ok,
                report.bf16_strided_abs_ok,
                report.bf16_strided_sqrt_ok,
                report.bf16_strided_exp_ok,
            },
        );
        try writer.print(
            "  \"bf16_strided_memref_legality_fingerprint\": {d},\n" ++
                "  \"bf16_strided_unary_memref_legality_fingerprint\": {d},\n" ++
                "  \"bf16_strided_scalar_memref_legality_fingerprint\": {d},\n" ++
                "  \"bf16_strided_scalar_add_ok\": {},\n" ++
                "  \"bf16_strided_scalar_sub_ok\": {},\n" ++
                "  \"bf16_strided_scalar_mul_ok\": {},\n" ++
                "  \"bf16_strided_scalar_div_ok\": {},\n",
            .{
                report.bf16_strided_memref_legality_fingerprint,
                report.bf16_strided_unary_memref_legality_fingerprint,
                report.bf16_strided_scalar_memref_legality_fingerprint,
                report.bf16_strided_scalar_add_ok,
                report.bf16_strided_scalar_sub_ok,
                report.bf16_strided_scalar_mul_ok,
                report.bf16_strided_scalar_div_ok,
            },
        );
        try writer.print(
            "  \"device_array_ok\": {},\n" ++
                "  \"max_abs_error\": {d},\n" ++
                "  \"lhs_plan_ok\": {},\n" ++
                "  \"lhs_plan_logical_elements\": {d},\n" ++
                "  \"lhs_plan_required_span\": {d},\n" ++
                "  \"lhs_plan_logical_bytes\": {d},\n" ++
                "  \"lhs_plan_required_bytes\": {d},\n" ++
                "  \"lhs_plan_linear_copy\": {},\n" ++
                "  \"lhs_plan_fingerprint\": {d},\n" ++
                "  \"lhs_copy_plan_ok\": {},\n" ++
                "  \"lhs_copy_plan_requires_strided\": {},\n" ++
                "  \"lhs_copy_plan_fingerprint\": {d},\n" ++
                "  \"dtype_support_count\": {d},\n" ++
                "  \"dtype_bridge_count\": {d},\n" ++
                "  \"dtype_native_seed_count\": {d},\n" ++
                "  \"dtype_widened_seed_count\": {d},\n" ++
                "  \"dtype_support_fingerprint\": {d},\n" ++
                "  \"f16_native_execution_fingerprint\": {d},\n" ++
                "  \"bf16_native_execution_fingerprint\": {d},\n" ++
                "  \"f16_widened_execution_fingerprint\": {d},\n" ++
                "  \"bf16_widened_execution_fingerprint\": {d},\n",
            .{
                report.device_array_ok,
                report.max_abs_error,
                report.lhs_plan.ok,
                report.lhs_plan.logical_elements,
                report.lhs_plan.required_span,
                report.lhs_plan.logical_bytes,
                report.lhs_plan.required_bytes,
                report.lhs_plan.linear_copy,
                report.lhs_plan.fingerprint,
                report.lhs_plan.copy_ok,
                report.lhs_plan.copy_requires_strided,
                report.lhs_plan.copy_fingerprint,
                report.dtype_support_count,
                report.dtype_bridge_count,
                report.dtype_native_seed_count,
                report.dtype_widened_seed_count,
                report.dtype_support_fingerprint,
                report.f16_native_execution_fingerprint,
                report.bf16_native_execution_fingerprint,
                report.f16_widened_execution_fingerprint,
                report.bf16_widened_execution_fingerprint,
            },
        );
        try writer.print(
            "  \"typed_f16_gemm_plan_fingerprint\": {d},\n" ++
                "  \"typed_bf16_gemm_plan_fingerprint\": {d},\n" ++
                "  \"typed_f16_gemm_seed_fingerprint\": {d},\n" ++
                "  \"typed_bf16_gemm_seed_fingerprint\": {d},\n" ++
                "  \"typed_f16_gemm_readiness_fingerprint\": {d},\n" ++
                "  \"typed_bf16_gemm_readiness_fingerprint\": {d},\n" ++
                "  \"output_fingerprint\": {d},\n" ++
                "  \"fingerprint\": {d}\n" ++
                "}}\n",
            .{
                report.typed_f16_gemm_plan.plan_fingerprint,
                report.typed_bf16_gemm_plan.plan_fingerprint,
                report.typed_f16_gemm_plan.seed_fingerprint,
                report.typed_bf16_gemm_plan.seed_fingerprint,
                report.typed_f16_gemm_plan.readiness_fingerprint,
                report.typed_bf16_gemm_plan.readiness_fingerprint,
                report.output_fingerprint,
                report.fingerprint(),
            },
        );
    }
};

pub fn enabled() bool {
    return build_options.enable_axiom_cuda;
}

pub fn deviceAvailable(index: usize) bool {
    if (!build_options.enable_axiom_cuda) return false;
    return CudaDriver.hasDevice(index);
}

pub fn planArrayF32(input: array_mod.Array(f32), name: []const u8) BufferPlanEvidence {
    if (!build_options.enable_axiom_cuda) return .{};
    if (!input.device.isCpu() or !input.isContiguous() or input.data.len == 0) return .{};
    const device_ptr: u64 = @intCast(@intFromPtr(input.data.ptr));
    const plan = axiom.accelerator.TensorDeviceBufferPlan.fromBufferView(
        axiom.accelerator.TensorBufferView.contiguous(name, device_ptr, input.data.len),
    ) catch return .{};
    const copy_plan = axiom.accelerator.TensorDeviceCopyPlan.fromBufferPlan(plan, .host_to_device) catch return .{};
    return .{
        .ok = plan.ok(),
        .logical_elements = plan.logical_element_count,
        .required_span = plan.required_element_span,
        .logical_bytes = plan.logical_byte_count,
        .required_bytes = plan.required_byte_span,
        .linear_copy = plan.linearCopyCompatible(),
        .fingerprint = plan.fingerprint(),
        .copy_ok = copy_plan.ok(),
        .copy_requires_strided = copy_plan.requires_strided_copy,
        .copy_fingerprint = copy_plan.fingerprint(),
    };
}

pub fn planTypedGemmF16(lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!TypedGemmPlanEvidence {
    if (!build_options.enable_axiom_cuda) return .{};
    if (!supportedMatmul2dContiguousF16(lhs, rhs)) return error.ShapeMismatch;
    const program = buildTypedMatmulTileIr(
        lhs.shape[0],
        rhs.shape[1],
        lhs.shape[1],
        .f16,
        "vectra_axiom_f16_typed_gemm_plan",
    );
    const plan = axiom.accelerator.TensorTypedGemmLaunchPlan.fromCudaTileProgram(program, 1.0, 0.0) catch |err| return mapTensorAdapterError(err);
    return typedGemmPlanEvidenceFromAxiom(plan);
}

pub fn planTypedGemmBF16(lhs: array_mod.Array(BFloat16), rhs: array_mod.Array(BFloat16)) array_mod.ArrayError!TypedGemmPlanEvidence {
    if (!build_options.enable_axiom_cuda) return .{};
    if (!supportedMatmul2dContiguousBF16(lhs, rhs)) return error.ShapeMismatch;
    const program = buildTypedMatmulTileIr(
        lhs.shape[0],
        rhs.shape[1],
        lhs.shape[1],
        .bf16,
        "vectra_axiom_bf16_typed_gemm_plan",
    );
    const plan = axiom.accelerator.TensorTypedGemmLaunchPlan.fromCudaTileProgram(program, 1.0, 0.0) catch |err| return mapTensorAdapterError(err);
    return typedGemmPlanEvidenceFromAxiom(plan);
}

fn baseSmokeReport() SmokeReport {
    return .{
        .enabled = build_options.enable_axiom_cuda,
        .status = if (build_options.enable_axiom_cuda) .skipped else .disabled,
        .dtype_support_count = cuda_dtype_support.len,
        .dtype_bridge_count = cudaDTypeBridgeCount(),
        .dtype_native_seed_count = cudaDTypeNativeSeedCount(),
        .dtype_widened_seed_count = cudaDTypeWidenedSeedCount(),
        .dtype_support_fingerprint = cudaDTypeSupportFingerprint(),
    };
}

pub fn tryAddF32(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    if (try tryDeviceBinaryF32(.add, lhs, rhs)) |out| return out;
    return tryBinaryF32(.add, lhs, rhs);
}

pub fn trySubF32(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    if (try tryDeviceBinaryF32(.sub, lhs, rhs)) |out| return out;
    return tryBinaryF32(.sub, lhs, rhs);
}

pub fn tryMulF32(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    if (try tryDeviceBinaryF32(.mul, lhs, rhs)) |out| return out;
    return tryBinaryF32(.mul, lhs, rhs);
}

pub fn tryDivF32(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    if (try tryDeviceBinaryF32(.div, lhs, rhs)) |out| return out;
    return tryBinaryF32(.div, lhs, rhs);
}

fn tryDeviceBinaryMemRefs(comptime T: type, op: BinaryOp, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (!build_options.enable_axiom_cuda) return null;
    if (T != f32 and T != f64 and T != f16 and T != BFloat16) return null;
    if (!lhs.device.isCuda() or !rhs.device.isCuda() or !lhs.device.sameDevice(rhs.device)) return null;
    if (!lhs.sameShape(rhs) or lhs.data.len != 0 or rhs.data.len != 0 or !lhs.isContiguous() or !rhs.isContiguous()) return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;
    if (lhs_storage.len == 0 or lhs_storage.len != rhs_storage.len) return null;

    var out = try array_mod.Array(T).emptyOn(lhs.allocator, lhs.shape, lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };
    const lhs_descriptor = describeDeviceArrayMemRef(T, lhs, lhs_storage, "lhs") catch {
        out.deinit();
        return null;
    };
    const rhs_descriptor = describeDeviceArrayMemRef(T, rhs, rhs_storage, "rhs") catch {
        out.deinit();
        return null;
    };
    const out_descriptor = describeDeviceArrayMemRef(T, out, out_storage, "out") catch {
        out.deinit();
        return null;
    };
    const spec = axiom.accelerator.TensorElementwiseBinarySpec.fromMemRefs(axiomBinaryOp(op), lhs_descriptor, rhs_descriptor, out_descriptor) catch {
        out.deinit();
        return null;
    };
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(lhs.allocator);
    const report = runtime.runCudaDeviceElementwiseBinaryMemRefs(lhs.device.index, spec) catch {
        out.deinit();
        return null;
    };
    if (!report.valid()) {
        out.deinit();
        return null;
    }
    recordCudaDeviceMemRefReport("elementwise_binary", report);
    return out;
}

pub fn tryDeviceBinaryF32(op: BinaryOp, lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryDeviceBinaryMemRefs(f32, op, lhs, rhs);
}

pub fn tryDeviceBinaryF16(op: BinaryOp, lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    return tryDeviceBinaryMemRefs(f16, op, lhs, rhs);
}

pub fn tryDeviceBinaryF64(op: BinaryOp, lhs: array_mod.Array(f64), rhs: array_mod.Array(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    return tryDeviceBinaryMemRefs(f64, op, lhs, rhs);
}

pub fn tryDeviceBinaryBF16(op: BinaryOp, lhs: array_mod.Array(BFloat16), rhs: array_mod.Array(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    return tryDeviceBinaryMemRefs(BFloat16, op, lhs, rhs);
}

pub fn trySqrtF32(input: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryDeviceUnaryF32(.sqrt, input);
}

pub fn tryExpF32(input: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryDeviceUnaryF32(.exp, input);
}

pub fn tryLogF32(input: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryDeviceUnaryF32(.log, input);
}

pub fn trySinF32(input: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryDeviceUnaryF32(.sin, input);
}

pub fn tryCosF32(input: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryDeviceUnaryF32(.cos, input);
}

pub fn tryTanF32(input: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryDeviceUnaryF32(.tan, input);
}

pub fn tryExp2F32(input: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryDeviceUnaryF32(.exp2, input);
}

pub fn tryExpm1F32(input: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryDeviceUnaryF32(.expm1, input);
}

pub fn tryLog1pF32(input: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryDeviceUnaryF32(.log1p, input);
}

pub fn tryLog2F32(input: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryDeviceUnaryF32(.log2, input);
}

pub fn tryLog10F32(input: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryDeviceUnaryF32(.log10, input);
}

pub fn trySqrtF16(input: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    return tryDeviceUnaryF16(.sqrt, input);
}

pub fn tryExpF16(input: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    return tryDeviceUnaryF16(.exp, input);
}

pub fn trySqrtBF16(input: array_mod.Array(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    return tryDeviceUnaryBF16(.sqrt, input);
}

pub fn tryExpBF16(input: array_mod.Array(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    return tryDeviceUnaryBF16(.exp, input);
}

pub fn trySqrtF64(input: array_mod.Array(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    return tryDeviceUnaryF64(.sqrt, input);
}

pub fn tryExpF64(input: array_mod.Array(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    return tryDeviceUnaryF64(.exp, input);
}

pub fn tryAbsF32(input: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryDeviceUnaryF32(.abs, input);
}

pub fn tryAbsF16(input: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    return tryDeviceUnaryF16(.abs, input);
}

pub fn tryAbsBF16(input: array_mod.Array(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    return tryDeviceUnaryBF16(.abs, input);
}

pub fn tryAbsF64(input: array_mod.Array(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    return tryDeviceUnaryF64(.abs, input);
}

fn tryDeviceUnaryMemRefs(comptime T: type, op: UnaryOp, input: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (!build_options.enable_axiom_cuda) return null;
    if (T != f32 and T != f64 and T != f16 and T != BFloat16) return null;
    if (!input.device.isCuda() or input.data.len != 0 or !input.isContiguous()) return null;
    const in_storage = input.device_storage orelse return null;
    if (in_storage.len == 0) return null;
    var out = try array_mod.Array(T).emptyOn(input.allocator, input.shape, input.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };
    const input_descriptor = describeDeviceArrayMemRef(T, input, in_storage, "input") catch {
        out.deinit();
        return null;
    };
    const out_descriptor = describeDeviceArrayMemRef(T, out, out_storage, "out") catch {
        out.deinit();
        return null;
    };
    const spec = axiom.accelerator.TensorElementwiseUnarySpec.fromMemRefs(axiomUnaryOp(op), input_descriptor, out_descriptor) catch {
        out.deinit();
        return null;
    };
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(input.allocator);
    const report = runtime.runCudaDeviceUnaryElementwiseMemRefs(input.device.index, spec) catch {
        out.deinit();
        return null;
    };
    if (!report.valid()) {
        out.deinit();
        return null;
    }
    recordCudaDeviceMemRefReport("elementwise_unary", report);
    return out;
}

pub fn tryDeviceUnaryF32(op: UnaryOp, input: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryDeviceUnaryMemRefs(f32, op, input);
}

pub fn tryDeviceUnaryF16(op: UnaryOp, input: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    return tryDeviceUnaryMemRefs(f16, op, input);
}

pub fn tryDeviceUnaryBF16(op: UnaryOp, input: array_mod.Array(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    return tryDeviceUnaryMemRefs(BFloat16, op, input);
}

pub fn tryDeviceUnaryF64(op: UnaryOp, input: array_mod.Array(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    return tryDeviceUnaryMemRefs(f64, op, input);
}

pub fn tryDeviceReductionF32(op: axiom.accelerator.DialectReductionOp, input: array_mod.Array(f32), axis: u1, keepdims: bool) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryDeviceReduction(f32, op, input, axis, keepdims);
}

pub fn tryDeviceReductionF64(op: axiom.accelerator.DialectReductionOp, input: array_mod.Array(f64), axis: u1, keepdims: bool) array_mod.ArrayError!?array_mod.Array(f64) {
    return tryDeviceReduction(f64, op, input, axis, keepdims);
}

pub fn tryDeviceReductionF16(op: axiom.accelerator.DialectReductionOp, input: array_mod.Array(f16), axis: u1, keepdims: bool) array_mod.ArrayError!?array_mod.Array(f16) {
    return tryDeviceReduction(f16, op, input, axis, keepdims);
}

pub fn tryDeviceReductionBF16(op: axiom.accelerator.DialectReductionOp, input: array_mod.Array(BFloat16), axis: u1, keepdims: bool) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    return tryDeviceReduction(BFloat16, op, input, axis, keepdims);
}

fn tryDeviceReduction(comptime T: type, op: axiom.accelerator.DialectReductionOp, input: array_mod.Array(T), axis: u1, keepdims: bool) array_mod.ArrayError!?array_mod.Array(T) {
    if (!build_options.enable_axiom_cuda) return null;
    if (T != f32 and T != f64 and T != f16 and T != BFloat16) return null;
    if (!input.device.isCuda() or input.data.len != 0 or !input.isContiguous()) return null;
    if (input.shape.len != 2) return null;
    const in_storage = input.device_storage orelse return null;
    if (in_storage.len == 0) return null;
    var keep_shape: [2]usize = undefined;
    var drop_shape: [1]usize = undefined;
    const out_shape: []const usize = if (keepdims) shape: {
        keep_shape = if (axis == 0) .{ 1, input.shape[1] } else .{ input.shape[0], 1 };
        break :shape keep_shape[0..2];
    } else shape: {
        drop_shape[0] = if (axis == 0) input.shape[1] else input.shape[0];
        break :shape drop_shape[0..1];
    };
    var out = try array_mod.Array(T).emptyOn(input.allocator, out_shape, input.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };
    const input_descriptor = describeDeviceArrayMemRef(T, input, in_storage, "input") catch {
        out.deinit();
        return null;
    };
    // Axiom's reduction runtime writes a dense rank-1 vector even when the
    // Vectra-facing Array keeps the reduced axis as a size-1 dimension.  Keep
    // the public `out.shape` untouched, but hand Axiom the actual runtime ABI
    // shape so descriptor legality matches the kernel contract.
    const runtime_out_shape = [_]usize{if (axis == 0) input.shape[1] else input.shape[0]};
    const out_descriptor = describeDeviceBufferMemRef(T, out_storage, runtime_out_shape[0..], &.{1}, "out") catch {
        out.deinit();
        return null;
    };
    const spec = axiom.accelerator.TensorReduction2DSpec.fromMemRefs(
        reductionOpFromDialect(op),
        reductionAxisFromU1(axis),
        input_descriptor,
        out_descriptor,
    ) catch {
        out.deinit();
        return null;
    };
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(input.allocator);
    const report = runtime.runCudaDeviceReductionMemRefs(input.device.index, spec) catch {
        out.deinit();
        return null;
    };
    if (!report.valid()) {
        out.deinit();
        return null;
    }
    recordCudaDeviceMemRefReport("reduction2d", report);
    return out;
}

pub fn tryDeviceBroadcastAddF32(input: array_mod.Array(f32), bias: array_mod.Array(f32), axis: axiom.accelerator.DialectBroadcastAxis) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryDeviceBroadcastBinary(f32, .add, input, bias, axis);
}

pub fn tryDeviceBroadcastBinaryF32(op: BinaryOp, input: array_mod.Array(f32), bias: array_mod.Array(f32), axis: axiom.accelerator.DialectBroadcastAxis) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryDeviceBroadcastBinary(f32, op, input, bias, axis);
}

pub fn tryDeviceBroadcastAddF64(input: array_mod.Array(f64), bias: array_mod.Array(f64), axis: axiom.accelerator.DialectBroadcastAxis) array_mod.ArrayError!?array_mod.Array(f64) {
    return tryDeviceBroadcastBinary(f64, .add, input, bias, axis);
}

pub fn tryDeviceBroadcastBinaryF64(op: BinaryOp, input: array_mod.Array(f64), bias: array_mod.Array(f64), axis: axiom.accelerator.DialectBroadcastAxis) array_mod.ArrayError!?array_mod.Array(f64) {
    return tryDeviceBroadcastBinary(f64, op, input, bias, axis);
}

pub fn tryDeviceBroadcastAddF16(input: array_mod.Array(f16), bias: array_mod.Array(f16), axis: axiom.accelerator.DialectBroadcastAxis) array_mod.ArrayError!?array_mod.Array(f16) {
    return tryDeviceBroadcastBinary(f16, .add, input, bias, axis);
}

pub fn tryDeviceBroadcastBinaryF16(op: BinaryOp, input: array_mod.Array(f16), bias: array_mod.Array(f16), axis: axiom.accelerator.DialectBroadcastAxis) array_mod.ArrayError!?array_mod.Array(f16) {
    return tryDeviceBroadcastBinary(f16, op, input, bias, axis);
}

pub fn tryDeviceBroadcastAddBF16(input: array_mod.Array(BFloat16), bias: array_mod.Array(BFloat16), axis: axiom.accelerator.DialectBroadcastAxis) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    return tryDeviceBroadcastBinary(BFloat16, .add, input, bias, axis);
}

pub fn tryDeviceBroadcastBinaryBF16(op: BinaryOp, input: array_mod.Array(BFloat16), bias: array_mod.Array(BFloat16), axis: axiom.accelerator.DialectBroadcastAxis) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    return tryDeviceBroadcastBinary(BFloat16, op, input, bias, axis);
}

pub fn tryDeviceVectorScalarBroadcastF32(op: BinaryOp, vector: array_mod.Array(f32), scalar: array_mod.Array(f32), scalar_left: bool) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryDeviceContiguousScalarBroadcast(f32, op, vector, scalar, scalar_left);
}

pub fn tryDeviceMatrixScalarBroadcastF32(op: BinaryOp, matrix: array_mod.Array(f32), scalar: array_mod.Array(f32), scalar_left: bool) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryDeviceContiguousScalarBroadcast(f32, op, matrix, scalar, scalar_left);
}

pub fn tryDeviceContiguousScalarBroadcastF32(op: BinaryOp, input: array_mod.Array(f32), scalar: array_mod.Array(f32), scalar_left: bool) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryDeviceContiguousScalarBroadcast(f32, op, input, scalar, scalar_left);
}

pub fn tryDeviceLastDimBroadcastF32(op: BinaryOp, input: array_mod.Array(f32), bias: array_mod.Array(f32), bias_left: bool) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryDeviceLastDimBroadcast(f32, op, input, bias, bias_left);
}

pub fn tryDeviceBroadcastF32(op: BinaryOp, lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    if (try tryDeviceGenericBroadcastF32(op, lhs, rhs)) |out| return out;
    return null;
}

pub fn tryDeviceVectorScalarBroadcastF64(op: BinaryOp, vector: array_mod.Array(f64), scalar: array_mod.Array(f64), scalar_left: bool) array_mod.ArrayError!?array_mod.Array(f64) {
    return tryDeviceContiguousScalarBroadcast(f64, op, vector, scalar, scalar_left);
}

pub fn tryDeviceMatrixScalarBroadcastF64(op: BinaryOp, matrix: array_mod.Array(f64), scalar: array_mod.Array(f64), scalar_left: bool) array_mod.ArrayError!?array_mod.Array(f64) {
    return tryDeviceContiguousScalarBroadcast(f64, op, matrix, scalar, scalar_left);
}

pub fn tryDeviceContiguousScalarBroadcastF64(op: BinaryOp, input: array_mod.Array(f64), scalar: array_mod.Array(f64), scalar_left: bool) array_mod.ArrayError!?array_mod.Array(f64) {
    return tryDeviceContiguousScalarBroadcast(f64, op, input, scalar, scalar_left);
}

pub fn tryDeviceLastDimBroadcastF64(op: BinaryOp, input: array_mod.Array(f64), bias: array_mod.Array(f64), bias_left: bool) array_mod.ArrayError!?array_mod.Array(f64) {
    return tryDeviceLastDimBroadcast(f64, op, input, bias, bias_left);
}

pub fn tryDeviceBroadcastF64(op: BinaryOp, lhs: array_mod.Array(f64), rhs: array_mod.Array(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    if (try tryDeviceGenericBroadcastF64(op, lhs, rhs)) |out| return out;
    return null;
}

pub fn tryDeviceVectorScalarBroadcastF16(op: BinaryOp, vector: array_mod.Array(f16), scalar: array_mod.Array(f16), scalar_left: bool) array_mod.ArrayError!?array_mod.Array(f16) {
    return tryDeviceContiguousScalarBroadcast(f16, op, vector, scalar, scalar_left);
}

pub fn tryDeviceMatrixScalarBroadcastF16(op: BinaryOp, matrix: array_mod.Array(f16), scalar: array_mod.Array(f16), scalar_left: bool) array_mod.ArrayError!?array_mod.Array(f16) {
    return tryDeviceContiguousScalarBroadcast(f16, op, matrix, scalar, scalar_left);
}

pub fn tryDeviceContiguousScalarBroadcastF16(op: BinaryOp, input: array_mod.Array(f16), scalar: array_mod.Array(f16), scalar_left: bool) array_mod.ArrayError!?array_mod.Array(f16) {
    return tryDeviceContiguousScalarBroadcast(f16, op, input, scalar, scalar_left);
}

pub fn tryDeviceLastDimBroadcastF16(op: BinaryOp, input: array_mod.Array(f16), bias: array_mod.Array(f16), bias_left: bool) array_mod.ArrayError!?array_mod.Array(f16) {
    return tryDeviceLastDimBroadcast(f16, op, input, bias, bias_left);
}

pub fn tryDeviceBroadcastF16(op: BinaryOp, lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    if (try tryDeviceGenericBroadcastF16(op, lhs, rhs)) |out| return out;
    return null;
}

pub fn tryDeviceVectorScalarBroadcastBF16(op: BinaryOp, vector: array_mod.Array(BFloat16), scalar: array_mod.Array(BFloat16), scalar_left: bool) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    return tryDeviceContiguousScalarBroadcast(BFloat16, op, vector, scalar, scalar_left);
}

pub fn tryDeviceMatrixScalarBroadcastBF16(op: BinaryOp, matrix: array_mod.Array(BFloat16), scalar: array_mod.Array(BFloat16), scalar_left: bool) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    return tryDeviceContiguousScalarBroadcast(BFloat16, op, matrix, scalar, scalar_left);
}

pub fn tryDeviceContiguousScalarBroadcastBF16(op: BinaryOp, input: array_mod.Array(BFloat16), scalar: array_mod.Array(BFloat16), scalar_left: bool) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    return tryDeviceContiguousScalarBroadcast(BFloat16, op, input, scalar, scalar_left);
}

pub fn tryDeviceLastDimBroadcastBF16(op: BinaryOp, input: array_mod.Array(BFloat16), bias: array_mod.Array(BFloat16), bias_left: bool) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    return tryDeviceLastDimBroadcast(BFloat16, op, input, bias, bias_left);
}

pub fn tryDeviceBroadcastBF16(op: BinaryOp, lhs: array_mod.Array(BFloat16), rhs: array_mod.Array(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    if (try tryDeviceGenericBroadcastBF16(op, lhs, rhs)) |out| return out;
    return null;
}

fn tryDeviceContiguousScalarBroadcast(comptime T: type, op: BinaryOp, input: array_mod.Array(T), scalar: array_mod.Array(T), scalar_left: bool) array_mod.ArrayError!?array_mod.Array(T) {
    if (!build_options.enable_axiom_cuda) return null;
    if (T != f32 and T != f64 and T != f16 and T != BFloat16) return null;
    if (!input.device.isCuda() or !scalar.device.isCuda() or !input.device.sameDevice(scalar.device)) return null;
    if (input.shape.len == 0 or scalar.numel() != 1) return null;
    if (input.data.len != 0 or scalar.data.len != 0 or !input.isContiguous() or !scalar.isContiguous()) return null;
    const input_storage = input.device_storage orelse return null;
    const scalar_storage = scalar.device_storage orelse return null;
    if (input_storage.len == 0 or scalar_storage.len != 1) return null;

    var out = try array_mod.Array(T).emptyOn(input.allocator, input.shape, input.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };
    // Until Axiom exposes a native N-D broadcast kernel, contiguous owning
    // arrays can safely lower to the 2-D row-broadcast ABI as a single row.
    // This preserves the Axiom memref/runtime boundary for common NumPy/PyTorch
    // scalar-array broadcasts such as `[B,M,N] / [1]` without materializing a
    // same-shape temporary in Vectra.
    const n = input_storage.len;
    const matrix_shape = [_]usize{ 1, n };
    const matrix_strides = [_]usize{ n, 1 };
    const bias_shape = [_]usize{n};
    const bias_strides = [_]usize{0};
    const input_descriptor = describeDeviceBufferMemRef(T, input_storage, matrix_shape[0..], matrix_strides[0..], "input") catch {
        out.deinit();
        return null;
    };
    const bias_descriptor = describeDeviceBufferMemRef(T, scalar_storage, bias_shape[0..], bias_strides[0..], "bias") catch {
        out.deinit();
        return null;
    };
    const out_descriptor = describeDeviceBufferMemRef(T, out_storage, matrix_shape[0..], matrix_strides[0..], "out") catch {
        out.deinit();
        return null;
    };
    const spec = axiom.accelerator.TensorBroadcastBinary2DSpec.fromMemRefsWithOpOrder(
        axiomBinaryOp(op),
        scalar_left,
        .row,
        input_descriptor,
        bias_descriptor,
        out_descriptor,
    ) catch {
        out.deinit();
        return null;
    };
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(input.allocator);
    const report = runtime.runCudaDeviceBroadcastBinaryMemRefs(input.device.index, spec) catch {
        out.deinit();
        return null;
    };
    if (!report.valid()) {
        out.deinit();
        return null;
    }
    recordCudaDeviceMemRefReport("broadcast_binary2d", report);
    return out;
}

fn tryDeviceLastDimBroadcast(comptime T: type, op: BinaryOp, input: array_mod.Array(T), bias: array_mod.Array(T), bias_left: bool) array_mod.ArrayError!?array_mod.Array(T) {
    if (!build_options.enable_axiom_cuda) return null;
    if (T != f32 and T != f64 and T != f16 and T != BFloat16) return null;
    if (!input.device.isCuda() or !bias.device.isCuda() or !input.device.sameDevice(bias.device)) return null;
    if (input.shape.len < 2 or bias.shape.len == 0 or bias.shape.len > input.shape.len) return null;
    if (input.data.len != 0 or bias.data.len != 0 or !input.isContiguous() or !bias.isContiguous()) return null;
    const cols = input.shape[input.shape.len - 1];
    if (cols == 0 or !lastDimBiasMatches(input.shape, bias.shape)) return null;
    const input_storage = input.device_storage orelse return null;
    const bias_storage = bias.device_storage orelse return null;
    if (input_storage.len == 0 or bias_storage.len != cols) return null;
    const rows = input_storage.len / cols;
    if (rows == 0 or rows * cols != input_storage.len) return null;

    var out = try array_mod.Array(T).emptyOn(input.allocator, input.shape, input.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };
    // Contiguous arrays whose bias broadcasts across leading dimensions and
    // matches the last dimension map exactly to Axiom's 2-D row-broadcast ABI
    // after flattening leading dimensions into rows.  This covers common
    // NumPy/PyTorch cases like `[B,M,N] - [N]` and keepdims-style
    // `[B,M,N] - [1,1,N]` without losing device residency or adding a
    // Vectra-owned CUDA kernel.
    const matrix_shape = [_]usize{ rows, cols };
    const matrix_strides = [_]usize{ cols, 1 };
    const bias_shape = [_]usize{cols};
    const bias_strides = [_]usize{1};
    const input_descriptor = describeDeviceBufferMemRef(T, input_storage, matrix_shape[0..], matrix_strides[0..], "input") catch {
        out.deinit();
        return null;
    };
    const bias_descriptor = describeDeviceBufferMemRef(T, bias_storage, bias_shape[0..], bias_strides[0..], "bias") catch {
        out.deinit();
        return null;
    };
    const out_descriptor = describeDeviceBufferMemRef(T, out_storage, matrix_shape[0..], matrix_strides[0..], "out") catch {
        out.deinit();
        return null;
    };
    const spec = axiom.accelerator.TensorBroadcastBinary2DSpec.fromMemRefsWithOpOrder(
        axiomBinaryOp(op),
        bias_left,
        .row,
        input_descriptor,
        bias_descriptor,
        out_descriptor,
    ) catch {
        out.deinit();
        return null;
    };
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(input.allocator);
    const report = runtime.runCudaDeviceBroadcastBinaryMemRefs(input.device.index, spec) catch {
        out.deinit();
        return null;
    };
    if (!report.valid()) {
        out.deinit();
        return null;
    }
    recordCudaDeviceMemRefReport(if (op == .add and !bias_left) "broadcast_add2d" else "broadcast_binary2d", report);
    return out;
}

fn tryDeviceGenericBroadcastF32(op: BinaryOp, lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    if (!build_options.enable_axiom_cuda) return null;
    if (!lhs.device.isCuda() or !rhs.device.isCuda() or !lhs.device.sameDevice(rhs.device)) return null;
    if (lhs.data.len != 0 or rhs.data.len != 0 or !lhs.isContiguous() or !rhs.isContiguous()) return null;
    const out_shape = broadcastShapeStack(lhs.shape, rhs.shape) orelse return null;
    if (out_shape.rank == 0 or out_shape.rank > 4) return null;
    var out_dims = [_]usize{ 1, 1, 1, 1 };
    alignTrailingDims(out_shape.dims[0..out_shape.rank], &out_dims);
    const lhs_strides = broadcastDeviceStrides(lhs.shape, out_shape.dims[0..out_shape.rank]) orelse return null;
    const rhs_strides = broadcastDeviceStrides(rhs.shape, out_shape.dims[0..out_shape.rank]) orelse return null;
    var out = try array_mod.Array(f32).emptyOn(lhs.allocator, out_shape.dims[0..out_shape.rank], lhs.device);
    errdefer out.deinit();
    const lhs_storage = lhs.device_storage orelse {
        out.deinit();
        return null;
    };
    const rhs_storage = rhs.device_storage orelse {
        out.deinit();
        return null;
    };
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };
    if (lhs_storage.len == 0 or rhs_storage.len == 0 or out_storage.len == 0) {
        out.deinit();
        return null;
    }
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(lhs.allocator);
    const report = runtime.runCudaDeviceBroadcast4F32(
        lhs.device.index,
        axiomBinaryOp(op),
        false,
        out_shape.rank,
        out_dims,
        lhs_strides,
        rhs_strides,
        lhs_storage.ptr,
        rhs_storage.ptr,
        out_storage.ptr,
        broadcast4SpecFingerprint(op, out_dims, lhs_strides, rhs_strides),
    ) catch {
        out.deinit();
        return null;
    };
    if (!report.valid()) {
        out.deinit();
        return null;
    }
    recordCudaDeviceMemRefReport("broadcast4_f32", report);
    return out;
}

fn tryDeviceGenericBroadcastF64(op: BinaryOp, lhs: array_mod.Array(f64), rhs: array_mod.Array(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    if (!build_options.enable_axiom_cuda) return null;
    if (!lhs.device.isCuda() or !rhs.device.isCuda() or !lhs.device.sameDevice(rhs.device)) return null;
    if (lhs.data.len != 0 or rhs.data.len != 0 or !lhs.isContiguous() or !rhs.isContiguous()) return null;
    const out_shape = broadcastShapeStack(lhs.shape, rhs.shape) orelse return null;
    if (out_shape.rank == 0 or out_shape.rank > 4) return null;
    var out_dims = [_]usize{ 1, 1, 1, 1 };
    alignTrailingDims(out_shape.dims[0..out_shape.rank], &out_dims);
    const lhs_strides = broadcastDeviceStrides(lhs.shape, out_shape.dims[0..out_shape.rank]) orelse return null;
    const rhs_strides = broadcastDeviceStrides(rhs.shape, out_shape.dims[0..out_shape.rank]) orelse return null;
    var out = try array_mod.Array(f64).emptyOn(lhs.allocator, out_shape.dims[0..out_shape.rank], lhs.device);
    errdefer out.deinit();
    const lhs_storage = lhs.device_storage orelse {
        out.deinit();
        return null;
    };
    const rhs_storage = rhs.device_storage orelse {
        out.deinit();
        return null;
    };
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };
    if (lhs_storage.len == 0 or rhs_storage.len == 0 or out_storage.len == 0) {
        out.deinit();
        return null;
    }
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(lhs.allocator);
    const report = runtime.runCudaDeviceBroadcast4F64(
        lhs.device.index,
        axiomBinaryOp(op),
        false,
        out_shape.rank,
        out_dims,
        lhs_strides,
        rhs_strides,
        lhs_storage.ptr,
        rhs_storage.ptr,
        out_storage.ptr,
        broadcast4SpecFingerprint(op, out_dims, lhs_strides, rhs_strides),
    ) catch {
        out.deinit();
        return null;
    };
    if (!report.valid()) {
        out.deinit();
        return null;
    }
    recordCudaDeviceMemRefReport("broadcast4_f64", report);
    return out;
}

fn tryDeviceGenericBroadcastF16(op: BinaryOp, lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    if (!build_options.enable_axiom_cuda) return null;
    if (!lhs.device.isCuda() or !rhs.device.isCuda() or !lhs.device.sameDevice(rhs.device)) return null;
    if (lhs.data.len != 0 or rhs.data.len != 0 or !lhs.isContiguous() or !rhs.isContiguous()) return null;
    const out_shape = broadcastShapeStack(lhs.shape, rhs.shape) orelse return null;
    if (out_shape.rank == 0 or out_shape.rank > 4) return null;
    var out_dims = [_]usize{ 1, 1, 1, 1 };
    alignTrailingDims(out_shape.dims[0..out_shape.rank], &out_dims);
    const lhs_strides = broadcastDeviceStrides(lhs.shape, out_shape.dims[0..out_shape.rank]) orelse return null;
    const rhs_strides = broadcastDeviceStrides(rhs.shape, out_shape.dims[0..out_shape.rank]) orelse return null;
    var out = try array_mod.Array(f16).emptyOn(lhs.allocator, out_shape.dims[0..out_shape.rank], lhs.device);
    errdefer out.deinit();
    const lhs_storage = lhs.device_storage orelse {
        out.deinit();
        return null;
    };
    const rhs_storage = rhs.device_storage orelse {
        out.deinit();
        return null;
    };
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };
    if (lhs_storage.len == 0 or rhs_storage.len == 0 or out_storage.len == 0) {
        out.deinit();
        return null;
    }
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(lhs.allocator);
    const report = runtime.runCudaDeviceBroadcast4F16(
        lhs.device.index,
        axiomBinaryOp(op),
        false,
        out_shape.rank,
        out_dims,
        lhs_strides,
        rhs_strides,
        lhs_storage.ptr,
        rhs_storage.ptr,
        out_storage.ptr,
        broadcast4SpecFingerprint(op, out_dims, lhs_strides, rhs_strides),
    ) catch {
        out.deinit();
        return null;
    };
    if (!report.valid()) {
        out.deinit();
        return null;
    }
    recordCudaDeviceMemRefReport("broadcast4_f16", report);
    return out;
}

fn tryDeviceGenericBroadcastBF16(op: BinaryOp, lhs: array_mod.Array(BFloat16), rhs: array_mod.Array(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    if (!build_options.enable_axiom_cuda) return null;
    if (!lhs.device.isCuda() or !rhs.device.isCuda() or !lhs.device.sameDevice(rhs.device)) return null;
    if (lhs.data.len != 0 or rhs.data.len != 0 or !lhs.isContiguous() or !rhs.isContiguous()) return null;
    const out_shape = broadcastShapeStack(lhs.shape, rhs.shape) orelse return null;
    if (out_shape.rank == 0 or out_shape.rank > 4) return null;
    var out_dims = [_]usize{ 1, 1, 1, 1 };
    alignTrailingDims(out_shape.dims[0..out_shape.rank], &out_dims);
    const lhs_strides = broadcastDeviceStrides(lhs.shape, out_shape.dims[0..out_shape.rank]) orelse return null;
    const rhs_strides = broadcastDeviceStrides(rhs.shape, out_shape.dims[0..out_shape.rank]) orelse return null;
    var out = try array_mod.Array(BFloat16).emptyOn(lhs.allocator, out_shape.dims[0..out_shape.rank], lhs.device);
    errdefer out.deinit();
    const lhs_storage = lhs.device_storage orelse {
        out.deinit();
        return null;
    };
    const rhs_storage = rhs.device_storage orelse {
        out.deinit();
        return null;
    };
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };
    if (lhs_storage.len == 0 or rhs_storage.len == 0 or out_storage.len == 0) {
        out.deinit();
        return null;
    }
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(lhs.allocator);
    const report = runtime.runCudaDeviceBroadcast4BF16(
        lhs.device.index,
        axiomBinaryOp(op),
        false,
        out_shape.rank,
        out_dims,
        lhs_strides,
        rhs_strides,
        lhs_storage.ptr,
        rhs_storage.ptr,
        out_storage.ptr,
        broadcast4SpecFingerprint(op, out_dims, lhs_strides, rhs_strides),
    ) catch {
        out.deinit();
        return null;
    };
    if (!report.valid()) {
        out.deinit();
        return null;
    }
    recordCudaDeviceMemRefReport("broadcast4_bf16", report);
    return out;
}

const StackBroadcastShape = struct {
    rank: u8 = 0,
    dims: [4]usize = .{ 1, 1, 1, 1 },
};

fn broadcastShapeStack(lhs_shape: []const usize, rhs_shape: []const usize) ?StackBroadcastShape {
    const rank = @max(lhs_shape.len, rhs_shape.len);
    if (rank == 0 or rank > 4) return null;
    var shape: StackBroadcastShape = .{ .rank = @intCast(rank) };
    var index: usize = 0;
    while (index < rank) : (index += 1) {
        const lhs_dim: usize = if (index >= rank - lhs_shape.len) lhs_shape[index - (rank - lhs_shape.len)] else 1;
        const rhs_dim: usize = if (index >= rank - rhs_shape.len) rhs_shape[index - (rank - rhs_shape.len)] else 1;
        if (lhs_dim != rhs_dim and lhs_dim != 1 and rhs_dim != 1) return null;
        shape.dims[index] = @max(lhs_dim, rhs_dim);
    }
    return shape;
}

fn broadcastDeviceStrides(input_shape: []const usize, out_shape: []const usize) ?[4]usize {
    if (input_shape.len == 0 or input_shape.len > out_shape.len or out_shape.len > 4) return null;
    var dense_strides = [_]usize{ 0, 0, 0, 0 };
    var stride: usize = 1;
    var dim_index: usize = input_shape.len;
    while (dim_index > 0) {
        dim_index -= 1;
        dense_strides[dim_index] = stride;
        stride = std.math.mul(usize, stride, input_shape[dim_index]) catch return null;
    }
    var out_strides = [_]usize{ 0, 0, 0, 0 };
    const rank_delta = out_shape.len - input_shape.len;
    var out_index: usize = 0;
    while (out_index < out_shape.len) : (out_index += 1) {
        if (out_index < rank_delta) {
            out_strides[out_index] = 0;
            continue;
        }
        const input_index = out_index - rank_delta;
        const input_dim = input_shape[input_index];
        const out_dim = out_shape[out_index];
        if (input_dim == out_dim) {
            out_strides[out_index] = dense_strides[input_index];
        } else if (input_dim == 1) {
            out_strides[out_index] = 0;
        } else {
            return null;
        }
    }
    var aligned = [_]usize{ 0, 0, 0, 0 };
    alignTrailingDims(out_strides[0..out_shape.len], &aligned);
    return aligned;
}

fn alignTrailingDims(values: []const usize, out: *[4]usize) void {
    out.* = .{ 1, 1, 1, 1 };
    const offset = 4 - values.len;
    for (values, 0..) |value, index| out[offset + index] = value;
}

fn broadcast4SpecFingerprint(op: BinaryOp, dims: [4]usize, lhs_strides: [4]usize, rhs_strides: [4]usize) u64 {
    var hasher = std.hash.Wyhash.init(0x0b20_a4f3_0001);
    hashBytes(&hasher, @tagName(op));
    for (dims) |dim| hashU64(&hasher, dim);
    for (lhs_strides) |stride| hashU64(&hasher, stride);
    for (rhs_strides) |stride| hashU64(&hasher, stride);
    return hasher.final();
}

fn lastDimBiasMatches(input_shape: []const usize, bias_shape: []const usize) bool {
    if (input_shape.len < 2 or bias_shape.len == 0 or bias_shape.len > input_shape.len) return false;
    const input_last = input_shape[input_shape.len - 1];
    if (bias_shape[bias_shape.len - 1] != input_last) return false;
    var index: usize = 0;
    while (index + 1 < bias_shape.len) : (index += 1) {
        if (bias_shape[index] != 1) return false;
    }
    return true;
}

fn tryDeviceBroadcastBinary(comptime T: type, op: BinaryOp, input: array_mod.Array(T), bias: array_mod.Array(T), axis: axiom.accelerator.DialectBroadcastAxis) array_mod.ArrayError!?array_mod.Array(T) {
    return tryDeviceBroadcastBinaryWithOrder(T, op, false, input, bias, axis);
}

fn tryDeviceBroadcastBinaryWithOrder(comptime T: type, op: BinaryOp, reverse_operands: bool, input: array_mod.Array(T), bias: array_mod.Array(T), axis: axiom.accelerator.DialectBroadcastAxis) array_mod.ArrayError!?array_mod.Array(T) {
    if (!build_options.enable_axiom_cuda) return null;
    if (T != f32 and T != f64 and T != f16 and T != BFloat16) return null;
    if (!input.device.isCuda() or !bias.device.isCuda() or !input.device.sameDevice(bias.device)) return null;
    if (input.data.len != 0 or bias.data.len != 0 or !input.isContiguous() or !bias.isContiguous()) return null;
    if (input.shape.len != 2) return null;
    const expected_axis_len = switch (axis) {
        .row => input.shape[1],
        .column => input.shape[0],
    };
    const in_storage = input.device_storage orelse return null;
    const bias_storage = bias.device_storage orelse return null;
    if (in_storage.len == 0) return null;
    const bias_is_scalar = bias_storage.len == 1 and bias.numel() == 1;
    if (!bias_is_scalar and bias_storage.len != expected_axis_len) return null;
    var out = try array_mod.Array(T).emptyOn(input.allocator, input.shape, input.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };
    const input_descriptor = describeDeviceArrayMemRef(T, input, in_storage, "input") catch {
        out.deinit();
        return null;
    };
    // Axiom's 2D row/column broadcast ABI consumes a vector-shaped bias.  A
    // one-element CUDA array is the scalar-broadcast specialization of that
    // vector: keep the logical vector length expected by the selected axis, but
    // set stride=0 so every lane reads the single device element.  This keeps
    // NumPy/PyTorch scalar-array broadcasting on the Axiom memref path without
    // materializing a same-shape temporary in Vectra.
    const runtime_bias_shape = [_]usize{expected_axis_len};
    const runtime_bias_strides = [_]usize{if (bias_is_scalar) 0 else 1};
    const bias_descriptor = describeDeviceBufferMemRef(T, bias_storage, runtime_bias_shape[0..], runtime_bias_strides[0..], "bias") catch {
        out.deinit();
        return null;
    };
    const out_descriptor = describeDeviceArrayMemRef(T, out, out_storage, "out") catch {
        out.deinit();
        return null;
    };
    const spec = axiom.accelerator.TensorBroadcastBinary2DSpec.fromMemRefsWithOpOrder(
        axiomBinaryOp(op),
        reverse_operands,
        broadcastAxisFromDialect(axis),
        input_descriptor,
        bias_descriptor,
        out_descriptor,
    ) catch {
        out.deinit();
        return null;
    };
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(input.allocator);
    const report = runtime.runCudaDeviceBroadcastAddMemRefs(input.device.index, spec) catch {
        out.deinit();
        return null;
    };
    if (!report.valid()) {
        out.deinit();
        return null;
    }
    recordCudaDeviceMemRefReport(if (op == .add) "broadcast_add2d" else "broadcast_binary2d", report);
    return out;
}

pub fn tryDeviceLogSoftmaxF32(input: array_mod.Array(f32), axis: u1) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryDeviceLogSoftmax(f32, input, axis);
}

pub fn tryDeviceLogSoftmaxF64(input: array_mod.Array(f64), axis: u1) array_mod.ArrayError!?array_mod.Array(f64) {
    return tryDeviceLogSoftmax(f64, input, axis);
}

pub fn tryDeviceLogSoftmaxF16(input: array_mod.Array(f16), axis: u1) array_mod.ArrayError!?array_mod.Array(f16) {
    return tryDeviceLogSoftmax(f16, input, axis);
}

pub fn tryDeviceLogSoftmaxBF16(input: array_mod.Array(BFloat16), axis: u1) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    return tryDeviceLogSoftmax(BFloat16, input, axis);
}

fn tryDeviceLogSoftmax(comptime T: type, input: array_mod.Array(T), axis: u1) array_mod.ArrayError!?array_mod.Array(T) {
    if (!build_options.enable_axiom_cuda) return null;
    if (T != f32 and T != f64 and T != f16 and T != BFloat16) return null;
    if (!input.device.isCuda() or input.data.len != 0 or !input.isContiguous()) return null;
    if (input.shape.len != 2) return null;
    const in_storage = input.device_storage orelse return null;
    if (in_storage.len == 0) return null;
    var out = try array_mod.Array(T).emptyOn(input.allocator, input.shape, input.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };
    const input_descriptor = describeDeviceArrayMemRef(T, input, in_storage, "input") catch {
        out.deinit();
        return null;
    };
    const out_descriptor = describeDeviceArrayMemRef(T, out, out_storage, "out") catch {
        out.deinit();
        return null;
    };
    const spec = axiom.accelerator.TensorSoftmax2DSpec.fromMemRefs(.log_softmax, reductionAxisFromU1(axis), input_descriptor, out_descriptor) catch {
        out.deinit();
        return null;
    };
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(input.allocator);
    const report = runtime.runCudaDeviceLogSoftmaxMemRefs(input.device.index, spec) catch {
        out.deinit();
        return null;
    };
    if (!report.valid()) {
        out.deinit();
        return null;
    }
    recordCudaDeviceMemRefReport("log_softmax2d", report);
    return out;
}

pub fn tryDeviceSoftmaxF32(input: array_mod.Array(f32), axis: u1) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryDeviceSoftmax(f32, input, axis);
}

pub fn tryDeviceSoftmaxF64(input: array_mod.Array(f64), axis: u1) array_mod.ArrayError!?array_mod.Array(f64) {
    return tryDeviceSoftmax(f64, input, axis);
}

pub fn tryDeviceSoftmaxF16(input: array_mod.Array(f16), axis: u1) array_mod.ArrayError!?array_mod.Array(f16) {
    return tryDeviceSoftmax(f16, input, axis);
}

pub fn tryDeviceSoftmaxBF16(input: array_mod.Array(BFloat16), axis: u1) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    return tryDeviceSoftmax(BFloat16, input, axis);
}

fn tryDeviceSoftmax(comptime T: type, input: array_mod.Array(T), axis: u1) array_mod.ArrayError!?array_mod.Array(T) {
    if (!build_options.enable_axiom_cuda) return null;
    if (T != f32 and T != f64 and T != f16 and T != BFloat16) return null;
    if (!input.device.isCuda() or input.data.len != 0 or !input.isContiguous()) return null;
    if (input.shape.len != 2) return null;
    const in_storage = input.device_storage orelse return null;
    if (in_storage.len == 0) return null;
    var out = try array_mod.Array(T).emptyOn(input.allocator, input.shape, input.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };
    const input_descriptor = describeDeviceArrayMemRef(T, input, in_storage, "input") catch {
        out.deinit();
        return null;
    };
    const out_descriptor = describeDeviceArrayMemRef(T, out, out_storage, "out") catch {
        out.deinit();
        return null;
    };
    const spec = axiom.accelerator.TensorSoftmax2DSpec.fromMemRefs(.softmax, reductionAxisFromU1(axis), input_descriptor, out_descriptor) catch {
        out.deinit();
        return null;
    };
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(input.allocator);
    const report = runtime.runCudaDeviceSoftmaxMemRefs(input.device.index, spec) catch {
        out.deinit();
        return null;
    };
    if (!report.valid()) {
        out.deinit();
        return null;
    }
    recordCudaDeviceMemRefReport("softmax2d", report);
    return out;
}

pub fn tryDeviceTransposeF32(input: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryDeviceTranspose(f32, input);
}

pub fn tryDeviceTransposeF64(input: array_mod.Array(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    return tryDeviceTranspose(f64, input);
}

pub fn tryDeviceTransposeF16(input: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    return tryDeviceTranspose(f16, input);
}

pub fn tryDeviceTransposeBF16(input: array_mod.Array(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    return tryDeviceTranspose(BFloat16, input);
}

fn tryDeviceTranspose(comptime T: type, input: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (!build_options.enable_axiom_cuda) return null;
    if (T != f32 and T != f64 and T != f16 and T != BFloat16) return null;
    if (!input.device.isCuda() or input.data.len != 0 or !input.isContiguous()) return null;
    if (input.shape.len != 2) return null;
    const in_storage = input.device_storage orelse return null;
    if (in_storage.len == 0) return null;
    var out = try array_mod.Array(T).emptyOn(input.allocator, &.{ input.shape[1], input.shape[0] }, input.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };
    const input_descriptor = describeDeviceArrayMemRef(T, input, in_storage, "input") catch {
        out.deinit();
        return null;
    };
    const out_descriptor = describeDeviceArrayMemRef(T, out, out_storage, "out") catch {
        out.deinit();
        return null;
    };
    const spec = axiom.accelerator.TensorTranspose2DSpec.fromMemRefs(input_descriptor, out_descriptor) catch {
        out.deinit();
        return null;
    };
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(input.allocator);
    const report = runtime.runCudaDeviceTransposeMemRefs(input.device.index, spec) catch {
        out.deinit();
        return null;
    };
    if (!report.valid()) {
        out.deinit();
        return null;
    }
    recordCudaDeviceMemRefReport("transpose2d", report);
    return out;
}

pub fn tryAddF16(lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    return tryBinaryF16(.add, lhs, rhs);
}

pub fn trySubF16(lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    return tryBinaryF16(.sub, lhs, rhs);
}

pub fn tryMulF16(lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    return tryBinaryF16(.mul, lhs, rhs);
}

pub fn tryDivF16(lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    return tryBinaryF16(.div, lhs, rhs);
}

pub fn tryAddBF16(lhs: array_mod.Array(BFloat16), rhs: array_mod.Array(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    return tryBinaryBF16(.add, lhs, rhs);
}

pub fn trySubBF16(lhs: array_mod.Array(BFloat16), rhs: array_mod.Array(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    return tryBinaryBF16(.sub, lhs, rhs);
}

pub fn tryMulBF16(lhs: array_mod.Array(BFloat16), rhs: array_mod.Array(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    return tryBinaryBF16(.mul, lhs, rhs);
}

pub fn tryDivBF16(lhs: array_mod.Array(BFloat16), rhs: array_mod.Array(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    return tryBinaryBF16(.div, lhs, rhs);
}

pub fn trySaxpyF32(alpha: f32, x: array_mod.Array(f32), y: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    if (!build_options.enable_axiom_cuda) return null;
    if (!supportedSameShapeContiguous(x, y)) return null;

    var out = try array_mod.Array(f32).fromSlice(x.allocator, y.data, y.shape);
    errdefer out.deinit();

    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(x.allocator);
    const result = runtime.runTensorSaxpy(x.data, out.data, .{
        .alpha = alpha,
        .len = x.data.len,
        .kernel_symbol = "vectra_axiom_saxpy",
        .prefer_cached_device = true,
    }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => {
            out.deinit();
            return null;
        },
    };
    if (!result.verified) {
        out.deinit();
        return null;
    }
    return out;
}

pub fn tryAddScalarF32(input: array_mod.Array(f32), scalar: f32) array_mod.ArrayError!?array_mod.Array(f32) {
    if (!build_options.enable_axiom_cuda) return null;
    if (!supportedNonEmptyContiguous(input)) return null;

    // Stable first bridge: materialize the scalar broadcast into a Vectra-owned
    // same-shape host buffer, then reuse Axiom's proven contiguous add runtime.
    // Axiom already models zero-stride scalar input views, so Vectra can switch
    // to that no-fill path once persistent device-buffer reuse is available.
    var scalar_array = try array_mod.Array(f32).full(input.allocator, input.shape, scalar);
    defer scalar_array.deinit();
    return tryBinaryF32(.add, scalar_array, input);
}

pub fn tryMulScalarF32(input: array_mod.Array(f32), scalar: f32) array_mod.ArrayError!?array_mod.Array(f32) {
    if (!build_options.enable_axiom_cuda) return null;
    if (!supportedNonEmptyContiguous(input)) return null;

    var scalar_array = try array_mod.Array(f32).full(input.allocator, input.shape, scalar);
    defer scalar_array.deinit();
    return tryBinaryF32(.mul, scalar_array, input);
}

pub fn tryDivScalarF32(input: array_mod.Array(f32), scalar: f32) array_mod.ArrayError!?array_mod.Array(f32) {
    if (!build_options.enable_axiom_cuda) return null;
    if (!supportedNonEmptyContiguous(input)) return null;

    var scalar_array = try array_mod.Array(f32).full(input.allocator, input.shape, scalar);
    defer scalar_array.deinit();
    return tryBinaryF32(.div, input, scalar_array);
}

pub fn tryBinaryScalarBroadcastF32(op: BinaryOp, lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    if (!build_options.enable_axiom_cuda) return null;
    if (!supportedNonEmptyContiguous(lhs) or !supportedNonEmptyContiguous(rhs)) return null;
    if (lhs.data.len == rhs.data.len) return null;
    if (lhs.data.len != 1 and rhs.data.len != 1) return null;

    const scalar_left = lhs.data.len == 1;
    const vector = if (scalar_left) rhs else lhs;
    var scalar_array = try array_mod.Array(f32).full(vector.allocator, vector.shape, if (scalar_left) lhs.data[0] else rhs.data[0]);
    defer scalar_array.deinit();
    return if (scalar_left)
        tryBinaryF32(op, scalar_array, vector)
    else
        tryBinaryF32(op, vector, scalar_array);
}

pub fn trySaxpyScalarF32(alpha: f32, scalar_x: f32, y: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    if (!build_options.enable_axiom_cuda) return null;
    if (!supportedNonEmptyContiguous(y)) return null;

    var scalar_array = try array_mod.Array(f32).full(y.allocator, y.shape, scalar_x);
    defer scalar_array.deinit();
    return trySaxpyF32(alpha, scalar_array, y);
}

pub fn tryAddViewF32(lhs: array_mod.ArrayView(f32), rhs: array_mod.ArrayView(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryBinaryViewF32(.add, lhs, rhs);
}

pub fn trySubViewF32(lhs: array_mod.ArrayView(f32), rhs: array_mod.ArrayView(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryBinaryViewF32(.sub, lhs, rhs);
}

pub fn tryMulViewF32(lhs: array_mod.ArrayView(f32), rhs: array_mod.ArrayView(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryBinaryViewF32(.mul, lhs, rhs);
}

pub fn tryDivViewF32(lhs: array_mod.ArrayView(f32), rhs: array_mod.ArrayView(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryBinaryViewF32(.div, lhs, rhs);
}

pub fn tryViewScalarF32(op: BinaryOp, input: array_mod.ArrayView(f32), scalar: f32, scalar_left: bool) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryBinaryViewScalarF32(op, input, scalar, scalar_left);
}

pub fn tryAbsViewF32(input: array_mod.ArrayView(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryUnaryViewF32(.abs, input);
}

pub fn trySqrtViewF32(input: array_mod.ArrayView(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryUnaryViewF32(.sqrt, input);
}

pub fn tryExpViewF32(input: array_mod.ArrayView(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryUnaryViewF32(.exp, input);
}

pub fn tryLogViewF32(input: array_mod.ArrayView(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryUnaryViewF32(.log, input);
}

pub fn tryExp2ViewF32(input: array_mod.ArrayView(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryUnaryViewF32(.exp2, input);
}

pub fn tryExpm1ViewF32(input: array_mod.ArrayView(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryUnaryViewF32(.expm1, input);
}

pub fn tryLog1pViewF32(input: array_mod.ArrayView(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryUnaryViewF32(.log1p, input);
}

pub fn tryLog2ViewF32(input: array_mod.ArrayView(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryUnaryViewF32(.log2, input);
}

pub fn tryLog10ViewF32(input: array_mod.ArrayView(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryUnaryViewF32(.log10, input);
}

pub fn trySinViewF32(input: array_mod.ArrayView(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryUnaryViewF32(.sin, input);
}

pub fn tryCosViewF32(input: array_mod.ArrayView(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryUnaryViewF32(.cos, input);
}

pub fn tryTanViewF32(input: array_mod.ArrayView(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryUnaryViewF32(.tan, input);
}

pub fn tryAddViewF64(lhs: array_mod.ArrayView(f64), rhs: array_mod.ArrayView(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    return tryBinaryViewF64(.add, lhs, rhs);
}

pub fn trySubViewF64(lhs: array_mod.ArrayView(f64), rhs: array_mod.ArrayView(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    return tryBinaryViewF64(.sub, lhs, rhs);
}

pub fn tryMulViewF64(lhs: array_mod.ArrayView(f64), rhs: array_mod.ArrayView(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    return tryBinaryViewF64(.mul, lhs, rhs);
}

pub fn tryDivViewF64(lhs: array_mod.ArrayView(f64), rhs: array_mod.ArrayView(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    return tryBinaryViewF64(.div, lhs, rhs);
}

pub fn tryViewScalarF64(op: BinaryOp, input: array_mod.ArrayView(f64), scalar: f64, scalar_left: bool) array_mod.ArrayError!?array_mod.Array(f64) {
    return tryBinaryViewScalarF64(op, input, scalar, scalar_left);
}

pub fn tryAbsViewF64(input: array_mod.ArrayView(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    return tryUnaryViewF64(.abs, input);
}

pub fn trySqrtViewF64(input: array_mod.ArrayView(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    return tryUnaryViewF64(.sqrt, input);
}

pub fn tryExpViewF64(input: array_mod.ArrayView(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    return tryUnaryViewF64(.exp, input);
}

pub fn tryAddViewF16(lhs: array_mod.ArrayView(f16), rhs: array_mod.ArrayView(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    return tryBinaryViewF16(.add, lhs, rhs);
}

pub fn trySubViewF16(lhs: array_mod.ArrayView(f16), rhs: array_mod.ArrayView(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    return tryBinaryViewF16(.sub, lhs, rhs);
}

pub fn tryMulViewF16(lhs: array_mod.ArrayView(f16), rhs: array_mod.ArrayView(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    return tryBinaryViewF16(.mul, lhs, rhs);
}

pub fn tryDivViewF16(lhs: array_mod.ArrayView(f16), rhs: array_mod.ArrayView(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    return tryBinaryViewF16(.div, lhs, rhs);
}

pub fn tryViewScalarF16(op: BinaryOp, input: array_mod.ArrayView(f16), scalar: f16, scalar_left: bool) array_mod.ArrayError!?array_mod.Array(f16) {
    return tryBinaryViewScalarF16(op, input, scalar, scalar_left);
}

pub fn tryAbsViewF16(input: array_mod.ArrayView(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    return tryUnaryViewF16(.abs, input);
}

pub fn trySqrtViewF16(input: array_mod.ArrayView(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    return tryUnaryViewF16(.sqrt, input);
}

pub fn tryExpViewF16(input: array_mod.ArrayView(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    return tryUnaryViewF16(.exp, input);
}

pub fn tryAddViewBF16(lhs: array_mod.ArrayView(BFloat16), rhs: array_mod.ArrayView(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    return tryBinaryViewBF16(.add, lhs, rhs);
}

pub fn trySubViewBF16(lhs: array_mod.ArrayView(BFloat16), rhs: array_mod.ArrayView(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    return tryBinaryViewBF16(.sub, lhs, rhs);
}

pub fn tryMulViewBF16(lhs: array_mod.ArrayView(BFloat16), rhs: array_mod.ArrayView(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    return tryBinaryViewBF16(.mul, lhs, rhs);
}

pub fn tryDivViewBF16(lhs: array_mod.ArrayView(BFloat16), rhs: array_mod.ArrayView(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    return tryBinaryViewBF16(.div, lhs, rhs);
}

pub fn tryViewScalarBF16(op: BinaryOp, input: array_mod.ArrayView(BFloat16), scalar: BFloat16, scalar_left: bool) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    return tryBinaryViewScalarBF16(op, input, scalar, scalar_left);
}

pub fn tryAbsViewBF16(input: array_mod.ArrayView(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    return tryUnaryViewBF16(.abs, input);
}

pub fn trySqrtViewBF16(input: array_mod.ArrayView(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    return tryUnaryViewBF16(.sqrt, input);
}

pub fn tryExpViewBF16(input: array_mod.ArrayView(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    return tryUnaryViewBF16(.exp, input);
}

pub fn tryMatmulF32(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    if (try tryDeviceMatmulF32(lhs, rhs)) |out| return out;
    if (!build_options.enable_axiom_cuda) return null;
    if (!supportedMatmul2dContiguous(lhs, rhs)) return null;

    const m = lhs.shape[0];
    const k = lhs.shape[1];
    const n = rhs.shape[1];
    var c = try array_mod.Array(f32).zeros(lhs.allocator, &.{ m, n });
    defer c.deinit();
    var out = try array_mod.Array(f32).empty(lhs.allocator, &.{ m, n });
    errdefer out.deinit();

    const tile_program = buildMatmulTileIr(m, n, k, "vectra_axiom_tile_gemm");
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(lhs.allocator);
    const result = runtime.runCudaTileGemmHostSlices(tile_program, 1.0, 0.0, "auto", lhs.data, rhs.data, c.data, out.data) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => {
            out.deinit();
            return null;
        },
    };
    if (!result.verified) {
        out.deinit();
        return null;
    }
    return out;
}

pub fn tryDeviceMatmulF32(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    resetLastCudaDeviceGemmReport();
    if (!build_options.enable_axiom_cuda) return null;
    if (!lhs.device.isCuda() or !rhs.device.isCuda() or !lhs.device.sameDevice(rhs.device)) return null;
    if (lhs.data.len != 0 or rhs.data.len != 0 or lhs.shape.len != 2 or rhs.shape.len != 2 or lhs.shape[1] != rhs.shape[0] or !lhs.isContiguous() or !rhs.isContiguous()) return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;
    if (lhs_storage.len == 0 or rhs_storage.len == 0) return null;
    const m = lhs.shape[0];
    const k = lhs.shape[1];
    const n = rhs.shape[1];
    var out = try array_mod.Array(f32).emptyOn(lhs.allocator, &.{ m, n }, lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };
    const spec = describeDeviceGemmMemRefSpec(f32, m, n, k, lhs_storage.ptr, rhs_storage.ptr, out_storage.ptr, "lhs", "rhs", "out") catch {
        out.deinit();
        return null;
    };
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(lhs.allocator);
    const cublas_report = runtime.runCudaDeviceGemmMemRefs(lhs.device.index, spec) catch null;
    if (cublas_report) |report| {
        recordCudaDeviceGemmReport(report);
        if (report.valid()) return out;
    }
    out.deinit();
    return null;
}

pub fn tryDeviceMatmulF64(lhs: array_mod.Array(f64), rhs: array_mod.Array(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    resetLastCudaDeviceGemmReport();
    if (!build_options.enable_axiom_cuda) return null;
    if (!lhs.device.isCuda() or !rhs.device.isCuda() or !lhs.device.sameDevice(rhs.device)) return null;
    if (lhs.data.len != 0 or rhs.data.len != 0 or lhs.shape.len != 2 or rhs.shape.len != 2 or lhs.shape[1] != rhs.shape[0] or !lhs.isContiguous() or !rhs.isContiguous()) return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;
    if (lhs_storage.len == 0 or rhs_storage.len == 0) return null;
    const m = lhs.shape[0];
    const k = lhs.shape[1];
    const n = rhs.shape[1];
    var out = try array_mod.Array(f64).emptyOn(lhs.allocator, &.{ m, n }, lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    const spec = describeDeviceGemmMemRefSpec(f64, m, n, k, lhs_storage.ptr, rhs_storage.ptr, out_storage.ptr, "lhs64", "rhs64", "out64") catch {
        out.deinit();
        return null;
    };
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(lhs.allocator);
    const report = runtime.runCudaDeviceGemmMemRefs(lhs.device.index, spec) catch null;
    if (report) |value| {
        recordCudaDeviceGemmReport(value);
        if (value.valid()) return out;
    }
    return null;
}

pub fn tryDeviceMatmulF16(lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    resetLastCudaDeviceGemmReport();
    if (!build_options.enable_axiom_cuda) return null;
    if (!lhs.device.isCuda() or !rhs.device.isCuda() or !lhs.device.sameDevice(rhs.device)) return null;
    if (lhs.data.len != 0 or rhs.data.len != 0 or lhs.shape.len != 2 or rhs.shape.len != 2 or lhs.shape[1] != rhs.shape[0] or !lhs.isContiguous() or !rhs.isContiguous()) return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;
    if (lhs_storage.len == 0 or rhs_storage.len == 0) return null;
    const m = lhs.shape[0];
    const k = lhs.shape[1];
    const n = rhs.shape[1];
    var out = try array_mod.Array(f16).emptyOn(lhs.allocator, &.{ m, n }, lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    const spec = describeDeviceGemmMemRefSpec(f16, m, n, k, lhs_storage.ptr, rhs_storage.ptr, out_storage.ptr, "lhs16", "rhs16", "out16") catch {
        out.deinit();
        return null;
    };
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(lhs.allocator);
    const report = runtime.runCudaDeviceGemmMemRefs(lhs.device.index, spec) catch null;
    if (report) |value| {
        recordCudaDeviceGemmReport(value);
        if (value.valid()) return out;
    }
    return null;
}

pub fn tryDeviceMatmulAddF32(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32), addend: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    resetLastCudaDeviceGemmReport();
    if (!build_options.enable_axiom_cuda) return null;
    if (!lhs.device.isCuda() or !rhs.device.isCuda() or !addend.device.isCuda()) return null;
    if (!lhs.device.sameDevice(rhs.device) or !lhs.device.sameDevice(addend.device)) return null;
    if (lhs.data.len != 0 or rhs.data.len != 0 or addend.data.len != 0) return null;
    if (lhs.shape.len != 2 or rhs.shape.len != 2 or addend.shape.len != 2) return null;
    if (lhs.shape[1] != rhs.shape[0] or addend.shape[0] != lhs.shape[0] or addend.shape[1] != rhs.shape[1]) return null;
    if (!lhs.isContiguous() or !rhs.isContiguous() or !addend.isContiguous()) return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;
    const add_storage = addend.device_storage orelse return null;
    if (lhs_storage.len == 0 or rhs_storage.len == 0 or add_storage.len == 0) return null;
    const m = lhs.shape[0];
    const k = lhs.shape[1];
    const n = rhs.shape[1];

    {
        var out = try array_mod.Array(f32).emptyOn(lhs.allocator, &.{ m, n }, lhs.device);
        errdefer out.deinit();
        const out_storage = out.device_storage orelse {
            out.deinit();
            return null;
        };

        const spec = describeDeviceGemmAddMemRefSpec(f32, m, n, k, lhs_storage.ptr, rhs_storage.ptr, add_storage.ptr, out_storage.ptr, "lhs", "rhs", "addend", "out") catch {
            out.deinit();
            return null;
        };
        var runtime = axiom.accelerator.AcceleratorRuntime.cuda(lhs.allocator);
        const report = runtime.runCudaDeviceMatmulAddMemRefs(lhs.device.index, spec) catch null;
        if (report) |value| {
            recordCudaDeviceGemmReport(value);
            if (value.valid()) return out;
        }
        out.deinit();
    }

    var product = tryDeviceMatmulF32(lhs, rhs) catch return null;
    if (product) |*matmul_out| {
        defer matmul_out.deinit();
        return tryDeviceBinaryF32(.add, matmul_out.*, addend);
    }
    return null;
}

pub fn tryDeviceMatmulAddF64(lhs: array_mod.Array(f64), rhs: array_mod.Array(f64), addend: array_mod.Array(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    resetLastCudaDeviceGemmReport();
    if (!build_options.enable_axiom_cuda) return null;
    if (!lhs.device.isCuda() or !rhs.device.isCuda() or !addend.device.isCuda()) return null;
    if (!lhs.device.sameDevice(rhs.device) or !lhs.device.sameDevice(addend.device)) return null;
    if (lhs.data.len != 0 or rhs.data.len != 0 or addend.data.len != 0) return null;
    if (lhs.shape.len != 2 or rhs.shape.len != 2 or addend.shape.len != 2) return null;
    if (lhs.shape[1] != rhs.shape[0] or addend.shape[0] != lhs.shape[0] or addend.shape[1] != rhs.shape[1]) return null;
    if (!lhs.isContiguous() or !rhs.isContiguous() or !addend.isContiguous()) return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;
    const add_storage = addend.device_storage orelse return null;
    if (lhs_storage.len == 0 or rhs_storage.len == 0 or add_storage.len == 0) return null;
    const m = lhs.shape[0];
    const k = lhs.shape[1];
    const n = rhs.shape[1];

    var out = try array_mod.Array(f64).emptyOn(lhs.allocator, &.{ m, n }, lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    const spec = describeDeviceGemmAddMemRefSpec(f64, m, n, k, lhs_storage.ptr, rhs_storage.ptr, add_storage.ptr, out_storage.ptr, "lhs64", "rhs64", "add64", "out64") catch {
        out.deinit();
        return null;
    };
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(lhs.allocator);
    const report = runtime.runCudaDeviceMatmulAddMemRefs(lhs.device.index, spec) catch null;
    if (report) |value| {
        recordCudaDeviceGemmReport(value);
        if (value.valid()) return out;
    }
    out.deinit();
    return null;
}

pub fn tryDeviceMatmulAddF16(lhs: array_mod.Array(f16), rhs: array_mod.Array(f16), addend: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    resetLastCudaDeviceGemmReport();
    if (!build_options.enable_axiom_cuda) return null;
    if (!lhs.device.isCuda() or !rhs.device.isCuda() or !addend.device.isCuda()) return null;
    if (!lhs.device.sameDevice(rhs.device) or !lhs.device.sameDevice(addend.device)) return null;
    if (lhs.data.len != 0 or rhs.data.len != 0 or addend.data.len != 0) return null;
    if (lhs.shape.len != 2 or rhs.shape.len != 2 or addend.shape.len != 2) return null;
    if (lhs.shape[1] != rhs.shape[0] or addend.shape[0] != lhs.shape[0] or addend.shape[1] != rhs.shape[1]) return null;
    if (!lhs.isContiguous() or !rhs.isContiguous() or !addend.isContiguous()) return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;
    const add_storage = addend.device_storage orelse return null;
    if (lhs_storage.len == 0 or rhs_storage.len == 0 or add_storage.len == 0) return null;
    const m = lhs.shape[0];
    const k = lhs.shape[1];
    const n = rhs.shape[1];

    var out = try array_mod.Array(f16).emptyOn(lhs.allocator, &.{ m, n }, lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    const spec = describeDeviceGemmAddMemRefSpec(f16, m, n, k, lhs_storage.ptr, rhs_storage.ptr, add_storage.ptr, out_storage.ptr, "lhs16", "rhs16", "add16", "out16") catch {
        out.deinit();
        return null;
    };
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(lhs.allocator);
    const report = runtime.runCudaDeviceMatmulAddMemRefs(lhs.device.index, spec) catch null;
    if (report) |value| {
        recordCudaDeviceGemmReport(value);
        if (value.valid()) return out;
    }
    out.deinit();
    return null;
}

pub fn tryDeviceMatmulBF16(lhs: array_mod.Array(BFloat16), rhs: array_mod.Array(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    resetLastCudaDeviceGemmReport();
    if (!build_options.enable_axiom_cuda) return null;
    if (!lhs.device.isCuda() or !rhs.device.isCuda() or !lhs.device.sameDevice(rhs.device)) return null;
    if (lhs.data.len != 0 or rhs.data.len != 0 or lhs.shape.len != 2 or rhs.shape.len != 2 or lhs.shape[1] != rhs.shape[0] or !lhs.isContiguous() or !rhs.isContiguous()) return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;
    if (lhs_storage.len == 0 or rhs_storage.len == 0) return null;
    const m = lhs.shape[0];
    const k = lhs.shape[1];
    const n = rhs.shape[1];
    var out = try array_mod.Array(BFloat16).emptyOn(lhs.allocator, &.{ m, n }, lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    const spec = describeDeviceGemmMemRefSpec(BFloat16, m, n, k, lhs_storage.ptr, rhs_storage.ptr, out_storage.ptr, "lhs_bf16", "rhs_bf16", "out_bf16") catch {
        out.deinit();
        return null;
    };
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(lhs.allocator);
    const report = runtime.runCudaDeviceGemmMemRefs(lhs.device.index, spec) catch null;
    if (report) |value| {
        recordCudaDeviceGemmReport(value);
        if (value.valid()) return out;
    }
    return null;
}

pub fn tryDeviceMatvecF32(matrix: array_mod.Array(f32), vector: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryDeviceMatvec(f32, matrix, vector, "matvec_lhs_f32", "matvec_rhs_f32", "matvec_out_f32");
}

pub fn tryDeviceMatvecF64(matrix: array_mod.Array(f64), vector: array_mod.Array(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    return tryDeviceMatvec(f64, matrix, vector, "matvec_lhs_f64", "matvec_rhs_f64", "matvec_out_f64");
}

pub fn tryDeviceMatvecF16(matrix: array_mod.Array(f16), vector: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    return tryDeviceMatvec(f16, matrix, vector, "matvec_lhs_f16", "matvec_rhs_f16", "matvec_out_f16");
}

pub fn tryDeviceMatvecBF16(matrix: array_mod.Array(BFloat16), vector: array_mod.Array(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    return tryDeviceMatvec(BFloat16, matrix, vector, "matvec_lhs_bf16", "matvec_rhs_bf16", "matvec_out_bf16");
}

pub fn tryDeviceVecmatF32(vector: array_mod.Array(f32), matrix: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryDeviceVecmat(f32, vector, matrix, "vecmat_lhs_f32", "vecmat_rhs_f32", "vecmat_out_f32");
}

pub fn tryDeviceVecmatF64(vector: array_mod.Array(f64), matrix: array_mod.Array(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    return tryDeviceVecmat(f64, vector, matrix, "vecmat_lhs_f64", "vecmat_rhs_f64", "vecmat_out_f64");
}

pub fn tryDeviceVecmatF16(vector: array_mod.Array(f16), matrix: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    return tryDeviceVecmat(f16, vector, matrix, "vecmat_lhs_f16", "vecmat_rhs_f16", "vecmat_out_f16");
}

pub fn tryDeviceVecmatBF16(vector: array_mod.Array(BFloat16), matrix: array_mod.Array(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    return tryDeviceVecmat(BFloat16, vector, matrix, "vecmat_lhs_bf16", "vecmat_rhs_bf16", "vecmat_out_bf16");
}

pub fn tryDeviceDotF32(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryDeviceDot(f32, lhs, rhs, "dot_lhs_f32", "dot_rhs_f32", "dot_out_f32");
}

pub fn tryDeviceDotF64(lhs: array_mod.Array(f64), rhs: array_mod.Array(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    return tryDeviceDot(f64, lhs, rhs, "dot_lhs_f64", "dot_rhs_f64", "dot_out_f64");
}

pub fn tryDeviceDotF16(lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    return tryDeviceDot(f16, lhs, rhs, "dot_lhs_f16", "dot_rhs_f16", "dot_out_f16");
}

pub fn tryDeviceDotBF16(lhs: array_mod.Array(BFloat16), rhs: array_mod.Array(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    return tryDeviceDot(BFloat16, lhs, rhs, "dot_lhs_bf16", "dot_rhs_bf16", "dot_out_bf16");
}

fn tryDeviceDot(
    comptime T: type,
    lhs: array_mod.Array(T),
    rhs: array_mod.Array(T),
    lhs_name: []const u8,
    rhs_name: []const u8,
    out_name: []const u8,
) array_mod.ArrayError!?array_mod.Array(T) {
    resetLastCudaDeviceGemmReport();
    if (!build_options.enable_axiom_cuda) return null;
    if (!lhs.device.isCuda() or !rhs.device.isCuda() or !lhs.device.sameDevice(rhs.device)) return null;
    if (lhs.data.len != 0 or rhs.data.len != 0) return null;
    if (lhs.shape.len != 1 or rhs.shape.len != 1 or lhs.shape[0] == 0 or lhs.shape[0] != rhs.shape[0]) return null;
    if (!lhs.isContiguous() or !rhs.isContiguous()) return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;
    if (lhs_storage.len == 0 or rhs_storage.len == 0) return null;

    var out = try array_mod.Array(T).emptyOn(lhs.allocator, &.{}, lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };
    const k = lhs.shape[0];
    const lhs_desc = describeDeviceBufferMemRef(T, lhs_storage, &.{ 1, k }, &.{ k, 1 }, lhs_name) catch {
        out.deinit();
        return null;
    };
    const rhs_desc = describeDeviceBufferMemRef(T, rhs_storage, &.{ k, 1 }, &.{ 1, 1 }, rhs_name) catch {
        out.deinit();
        return null;
    };
    const out_desc = describeDeviceBufferMemRef(T, out_storage, &.{ 1, 1 }, &.{ 1, 1 }, out_name) catch {
        out.deinit();
        return null;
    };
    const spec = axiom.accelerator.TensorGemmSpec.fromMemRefs(lhs_desc, rhs_desc, out_desc) catch {
        out.deinit();
        return null;
    };
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(lhs.allocator);
    const report = runtime.runCudaDeviceGemmMemRefs(lhs.device.index, spec) catch null;
    if (report) |value| {
        recordCudaDeviceGemmReport(value);
        if (value.valid()) return out;
    }
    out.deinit();
    return null;
}

pub fn tryDeviceBmmF32(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryDeviceBmm(f32, lhs, rhs, "bmm_lhs_f32", "bmm_rhs_f32", "bmm_out_f32");
}

pub fn tryDeviceBmmF64(lhs: array_mod.Array(f64), rhs: array_mod.Array(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    return tryDeviceBmm(f64, lhs, rhs, "bmm_lhs_f64", "bmm_rhs_f64", "bmm_out_f64");
}

pub fn tryDeviceBmmF16(lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    return tryDeviceBmm(f16, lhs, rhs, "bmm_lhs_f16", "bmm_rhs_f16", "bmm_out_f16");
}

pub fn tryDeviceBmmBF16(lhs: array_mod.Array(BFloat16), rhs: array_mod.Array(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    return tryDeviceBmm(BFloat16, lhs, rhs, "bmm_lhs_bf16", "bmm_rhs_bf16", "bmm_out_bf16");
}

fn tryDeviceMatvec(
    comptime T: type,
    matrix: array_mod.Array(T),
    vector: array_mod.Array(T),
    matrix_name: []const u8,
    vector_name: []const u8,
    out_name: []const u8,
) array_mod.ArrayError!?array_mod.Array(T) {
    resetLastCudaDeviceBatchedGemmReport();
    if (!build_options.enable_axiom_cuda) return null;
    if (!matrix.device.isCuda() or !vector.device.isCuda() or !matrix.device.sameDevice(vector.device)) return null;
    if (matrix.data.len != 0 or vector.data.len != 0) return null;
    if (matrix.shape.len < 2 or vector.shape.len != 1) return null;
    if (!matrix.isContiguous() or !vector.isContiguous()) return null;
    const batch_shape = matrix.shape[0 .. matrix.shape.len - 2];
    const batch_count = try array_mod.numelFrom(batch_shape);
    const m = matrix.shape[matrix.shape.len - 2];
    const k = matrix.shape[matrix.shape.len - 1];
    if (batch_count == 0 or m == 0 or k == 0 or vector.shape[0] != k) return null;
    const matrix_storage = matrix.device_storage orelse return null;
    const vector_storage = vector.device_storage orelse return null;
    if (matrix_storage.len == 0 or vector_storage.len == 0) return null;

    const out_rank = batch_shape.len + 1;
    const out_shape = try matrix.allocator.alloc(usize, out_rank);
    defer matrix.allocator.free(out_shape);
    @memcpy(out_shape[0..batch_shape.len], batch_shape);
    out_shape[out_rank - 1] = m;
    var out = try array_mod.Array(T).emptyOn(matrix.allocator, out_shape, matrix.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    const matrix_batch_stride = std.math.mul(usize, m, k) catch return error.InvalidShape;
    const out_batch_stride = m;
    const matrix_shape: [3]usize = .{ batch_count, m, k };
    const vector_shape: [3]usize = .{ batch_count, k, 1 };
    const out_memref_shape: [3]usize = .{ batch_count, m, 1 };
    const matrix_strides: [3]usize = .{ matrix_batch_stride, k, 1 };
    const vector_strides: [3]usize = .{ 0, 1, 1 };
    const out_strides: [3]usize = .{ out_batch_stride, 1, 1 };

    const matrix_desc = describeDeviceBufferMemRef(T, matrix_storage, &matrix_shape, &matrix_strides, matrix_name) catch {
        out.deinit();
        return null;
    };
    const vector_desc = describeDeviceBufferMemRef(T, vector_storage, &vector_shape, &vector_strides, vector_name) catch {
        out.deinit();
        return null;
    };
    const out_desc = describeDeviceBufferMemRef(T, out_storage, &out_memref_shape, &out_strides, out_name) catch {
        out.deinit();
        return null;
    };
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(matrix.allocator);
    const report = runtime.runCudaDeviceBatchedGemmMemRefs(matrix.device.index, matrix_desc, vector_desc, out_desc) catch null;
    if (report) |value| {
        recordCudaDeviceBatchedGemmReport(value);
        if (value.valid()) return out;
    }
    out.deinit();
    return null;
}

fn tryDeviceVecmat(
    comptime T: type,
    vector: array_mod.Array(T),
    matrix: array_mod.Array(T),
    vector_name: []const u8,
    matrix_name: []const u8,
    out_name: []const u8,
) array_mod.ArrayError!?array_mod.Array(T) {
    resetLastCudaDeviceBatchedGemmReport();
    if (!build_options.enable_axiom_cuda) return null;
    if (!vector.device.isCuda() or !matrix.device.isCuda() or !vector.device.sameDevice(matrix.device)) return null;
    if (vector.data.len != 0 or matrix.data.len != 0) return null;
    if (vector.shape.len != 1 or matrix.shape.len < 2) return null;
    if (!vector.isContiguous() or !matrix.isContiguous()) return null;
    const batch_shape = matrix.shape[0 .. matrix.shape.len - 2];
    const batch_count = try array_mod.numelFrom(batch_shape);
    const k = vector.shape[0];
    const n = matrix.shape[matrix.shape.len - 1];
    if (batch_count == 0 or k == 0 or n == 0 or matrix.shape[matrix.shape.len - 2] != k) return null;
    const vector_storage = vector.device_storage orelse return null;
    const matrix_storage = matrix.device_storage orelse return null;
    if (vector_storage.len == 0 or matrix_storage.len == 0) return null;

    const out_rank = batch_shape.len + 1;
    const out_shape = try vector.allocator.alloc(usize, out_rank);
    defer vector.allocator.free(out_shape);
    @memcpy(out_shape[0..batch_shape.len], batch_shape);
    out_shape[out_rank - 1] = n;
    var out = try array_mod.Array(T).emptyOn(vector.allocator, out_shape, vector.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    const matrix_batch_stride = std.math.mul(usize, k, n) catch return error.InvalidShape;
    const out_batch_stride = n;
    const vector_shape: [3]usize = .{ batch_count, 1, k };
    const matrix_shape: [3]usize = .{ batch_count, k, n };
    const out_memref_shape: [3]usize = .{ batch_count, 1, n };
    const vector_strides: [3]usize = .{ 0, k, 1 };
    const matrix_strides: [3]usize = .{ matrix_batch_stride, n, 1 };
    const out_strides: [3]usize = .{ out_batch_stride, n, 1 };

    const vector_desc = describeDeviceBufferMemRef(T, vector_storage, &vector_shape, &vector_strides, vector_name) catch {
        out.deinit();
        return null;
    };
    const matrix_desc = describeDeviceBufferMemRef(T, matrix_storage, &matrix_shape, &matrix_strides, matrix_name) catch {
        out.deinit();
        return null;
    };
    const out_desc = describeDeviceBufferMemRef(T, out_storage, &out_memref_shape, &out_strides, out_name) catch {
        out.deinit();
        return null;
    };
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(vector.allocator);
    const report = runtime.runCudaDeviceBatchedGemmMemRefs(vector.device.index, vector_desc, matrix_desc, out_desc) catch null;
    if (report) |value| {
        recordCudaDeviceBatchedGemmReport(value);
        if (value.valid()) return out;
    }
    out.deinit();
    return null;
}

pub fn tryDeviceBatchedMatmulF32(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryDeviceBatchedMatmul(f32, lhs, rhs, "matmul_batch_lhs_f32", "matmul_batch_rhs_f32", "matmul_batch_out_f32");
}

pub fn tryDeviceBatchedMatmulF64(lhs: array_mod.Array(f64), rhs: array_mod.Array(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    return tryDeviceBatchedMatmul(f64, lhs, rhs, "matmul_batch_lhs_f64", "matmul_batch_rhs_f64", "matmul_batch_out_f64");
}

pub fn tryDeviceBatchedMatmulF16(lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    return tryDeviceBatchedMatmul(f16, lhs, rhs, "matmul_batch_lhs_f16", "matmul_batch_rhs_f16", "matmul_batch_out_f16");
}

pub fn tryDeviceBatchedMatmulBF16(lhs: array_mod.Array(BFloat16), rhs: array_mod.Array(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    return tryDeviceBatchedMatmul(BFloat16, lhs, rhs, "matmul_batch_lhs_bf16", "matmul_batch_rhs_bf16", "matmul_batch_out_bf16");
}

fn tryDeviceBmm(
    comptime T: type,
    lhs: array_mod.Array(T),
    rhs: array_mod.Array(T),
    lhs_name: []const u8,
    rhs_name: []const u8,
    out_name: []const u8,
) array_mod.ArrayError!?array_mod.Array(T) {
    resetLastCudaDeviceBatchedGemmReport();
    if (!build_options.enable_axiom_cuda) return null;
    if (!lhs.device.isCuda() or !rhs.device.isCuda() or !lhs.device.sameDevice(rhs.device)) return null;
    if (lhs.data.len != 0 or rhs.data.len != 0) return null;
    if (lhs.shape.len != 3 or rhs.shape.len != 3) return null;
    if (lhs.shape[0] == 0 or lhs.shape[1] == 0 or lhs.shape[2] == 0 or rhs.shape[2] == 0) return null;
    if (lhs.shape[0] != rhs.shape[0] or lhs.shape[2] != rhs.shape[1]) return null;
    if (!lhs.isContiguous() or !rhs.isContiguous()) return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;
    if (lhs_storage.len == 0 or rhs_storage.len == 0) return null;

    const batch = lhs.shape[0];
    const m = lhs.shape[1];
    const n = rhs.shape[2];
    var out = try array_mod.Array(T).emptyOn(lhs.allocator, &.{ batch, m, n }, lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    // Preserve bmm as a rank-3 memref contract all the way into Axiom.  Axiom
    // currently lowers this to a loop over per-batch GEMM descriptors, and can
    // later swap in a native strided-batched kernel without changing Vectra's
    // Array API or reintroducing target-specific array code.
    const lhs_desc = describeDeviceArrayMemRef(T, lhs, lhs_storage, lhs_name) catch {
        out.deinit();
        return null;
    };
    const rhs_desc = describeDeviceArrayMemRef(T, rhs, rhs_storage, rhs_name) catch {
        out.deinit();
        return null;
    };
    const out_desc = describeDeviceArrayMemRef(T, out, out_storage, out_name) catch {
        out.deinit();
        return null;
    };
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(lhs.allocator);
    const report = runtime.runCudaDeviceBatchedGemmMemRefs(lhs.device.index, lhs_desc, rhs_desc, out_desc) catch null;
    if (report) |value| {
        recordCudaDeviceBatchedGemmReport(value);
        if (value.valid()) return out;
    }
    out.deinit();
    return null;
}

fn tryDeviceBatchedMatmul(
    comptime T: type,
    lhs: array_mod.Array(T),
    rhs: array_mod.Array(T),
    lhs_name: []const u8,
    rhs_name: []const u8,
    out_name: []const u8,
) array_mod.ArrayError!?array_mod.Array(T) {
    resetLastCudaDeviceBatchedGemmReport();
    if (!build_options.enable_axiom_cuda) return null;
    if (!lhs.device.isCuda() or !rhs.device.isCuda() or !lhs.device.sameDevice(rhs.device)) return null;
    if (lhs.data.len != 0 or rhs.data.len != 0) return null;
    if (lhs.shape.len < 3 or rhs.shape.len < 3) return null;
    if (!lhs.isContiguous() or !rhs.isContiguous()) return null;
    const lhs_batch = lhs.shape[0 .. lhs.shape.len - 2];
    const rhs_batch = rhs.shape[0 .. rhs.shape.len - 2];
    const out_batch = computeBatchBroadcastShape(lhs.allocator, lhs_batch, rhs_batch) catch return null;
    defer lhs.allocator.free(out_batch);
    const batch_count = try array_mod.numelFrom(out_batch);
    const m = lhs.shape[lhs.shape.len - 2];
    const k = lhs.shape[lhs.shape.len - 1];
    const n = rhs.shape[rhs.shape.len - 1];
    if (batch_count == 0 or m == 0 or k == 0 or n == 0 or rhs.shape[rhs.shape.len - 2] != k) return null;

    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;
    if (lhs_storage.len == 0 or rhs_storage.len == 0) return null;
    const out_rank = out_batch.len + 2;
    const out_shape = try lhs.allocator.alloc(usize, out_rank);
    defer lhs.allocator.free(out_shape);
    @memcpy(out_shape[0..out_batch.len], out_batch);
    out_shape[out_rank - 2] = m;
    out_shape[out_rank - 1] = n;
    var out = try array_mod.Array(T).emptyOn(lhs.allocator, out_shape, lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    const lhs_batch_stride = std.math.mul(usize, m, k) catch return error.InvalidShape;
    const rhs_batch_stride = std.math.mul(usize, k, n) catch return error.InvalidShape;
    const out_batch_stride = std.math.mul(usize, m, n) catch return error.InvalidShape;
    const lhs_flat_batch_stride = try flattenedBatchStride(lhs_batch, out_batch, lhs_batch_stride);
    const rhs_flat_batch_stride = try flattenedBatchStride(rhs_batch, out_batch, rhs_batch_stride);
    const lhs_desc, const rhs_desc, const out_desc = if (lhs_flat_batch_stride != null and rhs_flat_batch_stride != null) blk: {
        const lhs_shape: [3]usize = .{ batch_count, m, k };
        const rhs_shape: [3]usize = .{ batch_count, k, n };
        const out_memref_shape: [3]usize = .{ batch_count, m, n };
        const lhs_strides: [3]usize = .{ lhs_flat_batch_stride.?, k, 1 };
        const rhs_strides: [3]usize = .{ rhs_flat_batch_stride.?, n, 1 };
        const out_strides: [3]usize = .{ out_batch_stride, n, 1 };
        const lhs_desc = describeDeviceBufferMemRef(T, lhs_storage, &lhs_shape, &lhs_strides, lhs_name) catch {
            out.deinit();
            return null;
        };
        const rhs_desc = describeDeviceBufferMemRef(T, rhs_storage, &rhs_shape, &rhs_strides, rhs_name) catch {
            out.deinit();
            return null;
        };
        const out_desc = describeDeviceBufferMemRef(T, out_storage, &out_memref_shape, &out_strides, out_name) catch {
            out.deinit();
            return null;
        };
        break :blk .{ lhs_desc, rhs_desc, out_desc };
    } else blk: {
        // Mixed per-axis batch broadcasts cannot be expressed as one affine
        // batch stride.  Use Axiom's higher-rank batched GEMM memref contract
        // instead: broadcasted axes get zero strides, and Axiom's runtime loop
        // maps each flattened batch index back through those per-axis strides.
        const lhs_desc = describeBroadcastedBatchedMatmulMemRef(T, lhs, lhs_storage, out_batch, m, k, lhs_name) catch {
            out.deinit();
            return null;
        };
        const rhs_desc = describeBroadcastedBatchedMatmulMemRef(T, rhs, rhs_storage, out_batch, k, n, rhs_name) catch {
            out.deinit();
            return null;
        };
        const out_desc = describeDeviceArrayMemRef(T, out, out_storage, out_name) catch {
            out.deinit();
            return null;
        };
        break :blk .{ lhs_desc, rhs_desc, out_desc };
    };
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(lhs.allocator);
    const report = runtime.runCudaDeviceBatchedGemmMemRefs(lhs.device.index, lhs_desc, rhs_desc, out_desc) catch null;
    if (report) |value| {
        recordCudaDeviceBatchedGemmReport(value);
        if (value.valid()) return out;
    }
    out.deinit();
    return null;
}

fn computeBatchBroadcastShape(allocator: std.mem.Allocator, lhs: []const usize, rhs: []const usize) array_mod.ArrayError![]usize {
    const rank = @max(lhs.len, rhs.len);
    const out = try allocator.alloc(usize, rank);
    errdefer allocator.free(out);
    for (out, 0..) |*slot, index| {
        const lhs_dim: usize = if (index >= rank - lhs.len) lhs[index - (rank - lhs.len)] else 1;
        const rhs_dim: usize = if (index >= rank - rhs.len) rhs[index - (rank - rhs.len)] else 1;
        if (lhs_dim == rhs_dim or lhs_dim == 1 or rhs_dim == 1) {
            slot.* = @max(lhs_dim, rhs_dim);
        } else {
            return error.ShapeMismatch;
        }
    }
    return out;
}

fn flattenedBatchStride(input_batch: []const usize, out_batch: []const usize, contiguous_matrix_stride: usize) array_mod.ArrayError!?usize {
    if (std.mem.eql(usize, input_batch, out_batch)) return contiguous_matrix_stride;
    if ((try array_mod.numelFrom(input_batch)) == 1) return 0;
    return null;
}

fn describeBroadcastedBatchedMatmulMemRef(
    comptime T: type,
    input: array_mod.Array(T),
    storage: array_mod.DeviceStorage,
    out_batch: []const usize,
    rows: usize,
    cols: usize,
    name: []const u8,
) array_mod.ArrayError!axiom.accelerator.TensorMemRefDescriptor {
    const rank = out_batch.len + 2;
    if (rank > 4) return error.InvalidShape;
    const input_batch = input.shape[0 .. input.shape.len - 2];
    if (input_batch.len > out_batch.len) return error.InvalidShape;
    var shape_buf: [4]usize = .{ 1, 1, 1, 1 };
    var stride_buf: [4]usize = .{ 1, 1, 1, 1 };
    const leading = out_batch.len - input_batch.len;
    for (out_batch, 0..) |out_dim, axis| {
        shape_buf[axis] = out_dim;
        if (axis < leading) {
            stride_buf[axis] = 0;
        } else {
            const input_axis = axis - leading;
            const input_dim = input_batch[input_axis];
            if (input_dim == out_dim) {
                stride_buf[axis] = input.strides[input_axis];
            } else if (input_dim == 1) {
                stride_buf[axis] = 0;
            } else {
                return error.ShapeMismatch;
            }
        }
    }
    shape_buf[rank - 2] = rows;
    shape_buf[rank - 1] = cols;
    stride_buf[rank - 2] = input.strides[input.shape.len - 2];
    stride_buf[rank - 1] = input.strides[input.shape.len - 1];
    return describeDeviceBufferMemRef(T, storage, shape_buf[0..rank], stride_buf[0..rank], name);
}

pub fn runPendingMatmulF32(allocator: std.mem.Allocator, device: array_mod.Device, m: usize, n: usize, k: usize, lhs_ptr: u64, rhs_ptr: u64, out_ptr: u64) array_mod.ArrayError!bool {
    resetLastCudaDeviceGemmReport();
    const spec = try describeDeviceGemmMemRefSpec(f32, m, n, k, lhs_ptr, rhs_ptr, out_ptr, "pending_lhs", "pending_rhs", "pending_out");
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(allocator);
    const report = runtime.runCudaDeviceGemmMemRefs(device.index, spec) catch return error.BackendFailure;
    recordCudaDeviceGemmReport(report);
    return report.valid();
}

pub fn runPendingMatmulBF16(allocator: std.mem.Allocator, device: array_mod.Device, m: usize, n: usize, k: usize, lhs_ptr: u64, rhs_ptr: u64, out_ptr: u64) array_mod.ArrayError!bool {
    resetLastCudaDeviceGemmReport();
    const spec = try describeDeviceGemmMemRefSpec(BFloat16, m, n, k, lhs_ptr, rhs_ptr, out_ptr, "pending_lhs_bf16", "pending_rhs_bf16", "pending_out_bf16");
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(allocator);
    const report = runtime.runCudaDeviceGemmMemRefs(device.index, spec) catch return error.BackendFailure;
    recordCudaDeviceGemmReport(report);
    return report.valid();
}

pub fn runPendingMatmulF16(allocator: std.mem.Allocator, device: array_mod.Device, m: usize, n: usize, k: usize, lhs_ptr: u64, rhs_ptr: u64, out_ptr: u64) array_mod.ArrayError!bool {
    resetLastCudaDeviceGemmReport();
    const spec = try describeDeviceGemmMemRefSpec(f16, m, n, k, lhs_ptr, rhs_ptr, out_ptr, "pending_lhs_f16", "pending_rhs_f16", "pending_out_f16");
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(allocator);
    const report = runtime.runCudaDeviceGemmMemRefs(device.index, spec) catch return error.BackendFailure;
    recordCudaDeviceGemmReport(report);
    return report.valid();
}

pub fn runPendingMatmulF64(allocator: std.mem.Allocator, device: array_mod.Device, m: usize, n: usize, k: usize, lhs_ptr: u64, rhs_ptr: u64, out_ptr: u64) array_mod.ArrayError!bool {
    resetLastCudaDeviceGemmReport();
    const spec = try describeDeviceGemmMemRefSpec(f64, m, n, k, lhs_ptr, rhs_ptr, out_ptr, "pending_lhs_f64", "pending_rhs_f64", "pending_out_f64");
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(allocator);
    const report = runtime.runCudaDeviceGemmMemRefs(device.index, spec) catch return error.BackendFailure;
    recordCudaDeviceGemmReport(report);
    return report.valid();
}

pub fn runPendingMatmulAddF32(allocator: std.mem.Allocator, device: array_mod.Device, m: usize, n: usize, k: usize, lhs_ptr: u64, rhs_ptr: u64, add_ptr: u64, out_ptr: u64, alpha: f32, beta: f32) array_mod.ArrayError!bool {
    resetLastCudaDeviceGemmReport();
    var spec = try describeDeviceGemmAddMemRefSpec(f32, m, n, k, lhs_ptr, rhs_ptr, add_ptr, out_ptr, "pending_lhs", "pending_rhs", "pending_add", "pending_out");
    spec.alpha = alpha;
    spec.beta = beta;
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(allocator);
    const report = runtime.runCudaDeviceMatmulAddMemRefs(device.index, spec) catch return error.BackendFailure;
    recordCudaDeviceGemmReport(report);
    return report.valid();
}

pub fn runPendingMatmulAddBF16(allocator: std.mem.Allocator, device: array_mod.Device, m: usize, n: usize, k: usize, lhs_ptr: u64, rhs_ptr: u64, add_ptr: u64, out_ptr: u64, alpha: f32, beta: f32) array_mod.ArrayError!bool {
    resetLastCudaDeviceGemmReport();
    var spec = try describeDeviceGemmAddMemRefSpec(BFloat16, m, n, k, lhs_ptr, rhs_ptr, add_ptr, out_ptr, "pending_lhs_bf16", "pending_rhs_bf16", "pending_add_bf16", "pending_out_bf16");
    spec.alpha = alpha;
    spec.beta = beta;
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(allocator);
    const report = runtime.runCudaDeviceMatmulAddMemRefs(device.index, spec) catch return error.BackendFailure;
    recordCudaDeviceGemmReport(report);
    return report.valid();
}

pub fn runPendingMatmulAddF16(allocator: std.mem.Allocator, device: array_mod.Device, m: usize, n: usize, k: usize, lhs_ptr: u64, rhs_ptr: u64, add_ptr: u64, out_ptr: u64, alpha: f32, beta: f32) array_mod.ArrayError!bool {
    resetLastCudaDeviceGemmReport();
    var spec = try describeDeviceGemmAddMemRefSpec(f16, m, n, k, lhs_ptr, rhs_ptr, add_ptr, out_ptr, "pending_lhs_f16", "pending_rhs_f16", "pending_add_f16", "pending_out_f16");
    spec.alpha = alpha;
    spec.beta = beta;
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(allocator);
    const report = runtime.runCudaDeviceMatmulAddMemRefs(device.index, spec) catch return error.BackendFailure;
    recordCudaDeviceGemmReport(report);
    return report.valid();
}

pub fn runPendingMatmulAddF64(allocator: std.mem.Allocator, device: array_mod.Device, m: usize, n: usize, k: usize, lhs_ptr: u64, rhs_ptr: u64, add_ptr: u64, out_ptr: u64, alpha: f32, beta: f32) array_mod.ArrayError!bool {
    resetLastCudaDeviceGemmReport();
    var spec = try describeDeviceGemmAddMemRefSpec(f64, m, n, k, lhs_ptr, rhs_ptr, add_ptr, out_ptr, "pending_lhs_f64", "pending_rhs_f64", "pending_add_f64", "pending_out_f64");
    spec.alpha = alpha;
    spec.beta = beta;
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(allocator);
    const report = runtime.runCudaDeviceMatmulAddMemRefs(device.index, spec) catch return error.BackendFailure;
    recordCudaDeviceGemmReport(report);
    return report.valid();
}

pub fn runPendingMatmulAddUnaryF32(
    allocator: std.mem.Allocator,
    device: array_mod.Device,
    op: UnaryOp,
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
    resetLastCudaDeviceGemmReport();
    var spec = try describeDeviceGemmAddMemRefSpec(f32, m, n, k, lhs_ptr, rhs_ptr, add_ptr, out_ptr, "pending_unary_lhs", "pending_unary_rhs", "pending_unary_add", "pending_unary_out");
    spec.alpha = alpha;
    spec.beta = beta;
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(allocator);
    const report = runtime.runCudaDeviceF32MatmulAddUnaryMemRefs(
        device.index,
        switch (op) {
            .sqrt => axiom.accelerator.TensorUnaryElementwiseOp.sqrt,
            .exp => axiom.accelerator.TensorUnaryElementwiseOp.exp,
            .abs, .log, .sin, .cos, .tan, .exp2, .expm1, .log1p, .log2, .log10 => return error.TypeUnsupported,
        },
        spec,
    ) catch return error.BackendFailure;
    recordCudaDeviceGemmReport(report);
    return report.valid();
}

pub fn tryDeviceMatmulAddBF16(lhs: array_mod.Array(BFloat16), rhs: array_mod.Array(BFloat16), addend: array_mod.Array(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    resetLastCudaDeviceGemmReport();
    if (!build_options.enable_axiom_cuda) return null;
    if (!lhs.device.isCuda() or !rhs.device.isCuda() or !addend.device.isCuda()) return null;
    if (!lhs.device.sameDevice(rhs.device) or !lhs.device.sameDevice(addend.device)) return null;
    if (lhs.data.len != 0 or rhs.data.len != 0 or addend.data.len != 0) return null;
    if (lhs.shape.len != 2 or rhs.shape.len != 2 or addend.shape.len != 2) return null;
    if (lhs.shape[1] != rhs.shape[0] or addend.shape[0] != lhs.shape[0] or addend.shape[1] != rhs.shape[1]) return null;
    if (!lhs.isContiguous() or !rhs.isContiguous() or !addend.isContiguous()) return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;
    const add_storage = addend.device_storage orelse return null;
    if (lhs_storage.len == 0 or rhs_storage.len == 0 or add_storage.len == 0) return null;
    const m = lhs.shape[0];
    const k = lhs.shape[1];
    const n = rhs.shape[1];

    var out = try array_mod.Array(BFloat16).emptyOn(lhs.allocator, &.{ m, n }, lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    const spec = describeDeviceGemmAddMemRefSpec(BFloat16, m, n, k, lhs_storage.ptr, rhs_storage.ptr, add_storage.ptr, out_storage.ptr, "lhs_bf16", "rhs_bf16", "add_bf16", "out_bf16") catch {
        out.deinit();
        return null;
    };
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(lhs.allocator);
    const report = runtime.runCudaDeviceMatmulAddMemRefs(lhs.device.index, spec) catch null;
    if (report) |value| {
        recordCudaDeviceGemmReport(value);
        if (value.valid()) return out;
    }
    out.deinit();
    return null;
}

pub fn tryMatmulBF16(lhs: array_mod.Array(BFloat16), rhs: array_mod.Array(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    if (try tryDeviceMatmulBF16(lhs, rhs)) |out| return out;
    if (!build_options.enable_axiom_cuda) return null;
    if (!supportedMatmul2dContiguousBF16(lhs, rhs)) return null;
    if (try tryMatmulBF16AxiomTypedSimtSeed(lhs, rhs)) |out| return out;

    var lhs32 = try bf16ArrayToF32(lhs);
    defer lhs32.deinit();
    var rhs32 = try bf16ArrayToF32(rhs);
    defer rhs32.deinit();
    var out32 = try tryMatmulF32(lhs32, rhs32) orelse return null;
    defer out32.deinit();
    return try f32ArrayToBF16(out32);
}

pub fn tryMatmulF16(lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    if (try tryDeviceMatmulF16(lhs, rhs)) |out| return out;
    if (!build_options.enable_axiom_cuda) return null;
    if (!supportedMatmul2dContiguousF16(lhs, rhs)) return null;
    if (try tryMatmulF16AxiomTypedSimtSeed(lhs, rhs)) |out| return out;

    var lhs32 = try f16ArrayToF32(lhs);
    defer lhs32.deinit();
    var rhs32 = try f16ArrayToF32(rhs);
    defer rhs32.deinit();
    var out32 = try tryMatmulF32(lhs32, rhs32) orelse return null;
    defer out32.deinit();
    return try f32ArrayToF16(out32);
}

fn tryMatmulF16AxiomTypedSimtSeed(lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    if (!supportedMatmul2dContiguousF16(lhs, rhs)) return null;
    const m = lhs.shape[0];
    const k = lhs.shape[1];
    const n = rhs.shape[1];
    var out = try array_mod.Array(f16).empty(lhs.allocator, &.{ m, n });
    errdefer out.deinit();
    const c = try lhs.allocator.alloc(f16, m * n);
    defer lhs.allocator.free(c);
    @memset(c, @as(f16, 0.0));
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(lhs.allocator);
    const result = runtime.runTensorGemmF16TypedSimtSeed(lhs.data, rhs.data, c, out.data, .{
        .m = m,
        .n = n,
        .k = k,
        .tile_x = @intCast(@min(n, @as(usize, 16))),
        .tile_y = @intCast(@min(m, @as(usize, 16))),
        .kernel_symbol = "vectra_axiom_typed_f16_gemm_seed",
    }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => {
            out.deinit();
            return null;
        },
    };
    if (!result.ok()) {
        out.deinit();
        return null;
    }
    return out;
}

fn tryMatmulBF16AxiomTypedSimtSeed(lhs: array_mod.Array(BFloat16), rhs: array_mod.Array(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    if (!supportedMatmul2dContiguousBF16(lhs, rhs)) return null;
    const m = lhs.shape[0];
    const k = lhs.shape[1];
    const n = rhs.shape[1];
    var out = try array_mod.Array(BFloat16).empty(lhs.allocator, &.{ m, n });
    errdefer out.deinit();
    const lhs_bits = try lhs.allocator.alloc(u16, lhs.data.len);
    defer lhs.allocator.free(lhs_bits);
    const rhs_bits = try lhs.allocator.alloc(u16, rhs.data.len);
    defer lhs.allocator.free(rhs_bits);
    const c_bits = try lhs.allocator.alloc(u16, m * n);
    defer lhs.allocator.free(c_bits);
    const out_bits = try lhs.allocator.alloc(u16, m * n);
    defer lhs.allocator.free(out_bits);
    for (lhs.data, lhs_bits) |value, *slot| slot.* = value.bits;
    for (rhs.data, rhs_bits) |value, *slot| slot.* = value.bits;
    @memset(c_bits, @as(u16, 0));
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(lhs.allocator);
    const result = runtime.runTensorGemmBF16TypedSimtSeed(lhs_bits, rhs_bits, c_bits, out_bits, .{
        .m = m,
        .n = n,
        .k = k,
        .tile_x = @intCast(@min(n, @as(usize, 16))),
        .tile_y = @intCast(@min(m, @as(usize, 16))),
        .kernel_symbol = "vectra_axiom_typed_bf16_gemm_seed",
    }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => {
            out.deinit();
            return null;
        },
    };
    if (!result.ok()) {
        out.deinit();
        return null;
    }
    for (out_bits, out.data) |bits, *slot| slot.* = .{ .bits = bits };
    return out;
}

pub fn runSmoke(allocator: std.mem.Allocator) SmokeReport {
    if (!build_options.enable_axiom_cuda) return baseSmokeReport();

    var lhs = array_mod.Array(f32).fromSlice(allocator, &.{ 1, 2, 3, 4 }, &.{4}) catch return failedReport();
    defer lhs.deinit();
    var rhs = array_mod.Array(f32).fromSlice(allocator, &.{ 10, 20, 30, 40 }, &.{4}) catch return failedReport();
    defer rhs.deinit();

    var report = baseSmokeReport();
    report.enabled = true;
    report.status = .skipped;
    report.lhs_plan = planArrayF32(lhs, "lhs");

    var device_array = toDeviceF32(allocator, lhs) catch return failedReport();
    if (device_array) |*dev| {
        report.device_array_ok = dev.ok();
        report.output_fingerprint ^= dev.fingerprint();
        dev.deinit();
    }

    var scalar_add_out = tryAddScalarF32(rhs, 2.0) catch return failedReport();
    if (scalar_add_out) |*out| {
        defer out.deinit();
        report.scalar_add_ok = sliceClose(out.data, &.{ 12, 22, 32, 42 }, 0.0);
        report.max_abs_error = @max(report.max_abs_error, maxAbsError(out.data, &.{ 12, 22, 32, 42 }));
        report.output_fingerprint ^= hashF32Slice(out.data);
    }

    var scalar_mul_out = tryMulScalarF32(rhs, 2.0) catch return failedReport();
    if (scalar_mul_out) |*out| {
        defer out.deinit();
        report.scalar_mul_ok = sliceClose(out.data, &.{ 20, 40, 60, 80 }, 0.0);
        report.max_abs_error = @max(report.max_abs_error, maxAbsError(out.data, &.{ 20, 40, 60, 80 }));
        report.output_fingerprint ^= hashF32Slice(out.data);
    }

    var scalar_saxpy_out = trySaxpyScalarF32(3.0, 2.0, rhs) catch return failedReport();
    if (scalar_saxpy_out) |*out| {
        defer out.deinit();
        report.scalar_saxpy_ok = sliceClose(out.data, &.{ 16, 26, 36, 46 }, 0.0);
        report.max_abs_error = @max(report.max_abs_error, maxAbsError(out.data, &.{ 16, 26, 36, 46 }));
        report.output_fingerprint ^= hashF32Slice(out.data);
    }

    var add_out = tryAddF32(lhs, rhs) catch return failedReport();
    if (add_out) |*out| {
        defer out.deinit();
        report.add_ok = sliceClose(out.data, &.{ 11, 22, 33, 44 }, 0.0);
        report.max_abs_error = @max(report.max_abs_error, maxAbsError(out.data, &.{ 11, 22, 33, 44 }));
        report.output_fingerprint ^= hashF32Slice(out.data);
    }

    var sub_out = trySubF32(rhs, lhs) catch return failedReport();
    if (sub_out) |*out| {
        defer out.deinit();
        report.sub_ok = sliceClose(out.data, &.{ 9, 18, 27, 36 }, 0.0);
        report.max_abs_error = @max(report.max_abs_error, maxAbsError(out.data, &.{ 9, 18, 27, 36 }));
        report.output_fingerprint ^= hashF32Slice(out.data);
    }

    var mul_out = tryMulF32(lhs, rhs) catch return failedReport();
    if (mul_out) |*out| {
        defer out.deinit();
        report.mul_ok = sliceClose(out.data, &.{ 10, 40, 90, 160 }, 0.0);
        report.max_abs_error = @max(report.max_abs_error, maxAbsError(out.data, &.{ 10, 40, 90, 160 }));
        report.output_fingerprint ^= hashF32Slice(out.data);
    }

    var div_out = tryDivF32(rhs, lhs) catch return failedReport();
    if (div_out) |*out| {
        defer out.deinit();
        report.div_ok = sliceClose(out.data, &.{ 10, 10, 10, 10 }, 0.0);
        report.max_abs_error = @max(report.max_abs_error, maxAbsError(out.data, &.{ 10, 10, 10, 10 }));
        report.output_fingerprint ^= hashF32Slice(out.data);
    }

    var strided_lhs = array_mod.Array(f32).fromSlice(allocator, &.{ 1, 99, 2, 99, 3, 99, 4, 99 }, &.{8}) catch return failedReport();
    defer strided_lhs.deinit();
    var strided_rhs = array_mod.Array(f32).fromSlice(allocator, &.{ 10, 99, 20, 99, 30, 99, 40, 99 }, &.{8}) catch return failedReport();
    defer strided_rhs.deinit();
    var lhs_view = strided_lhs.asStrided(&.{4}, &.{2}, 0) catch return failedReport();
    defer lhs_view.deinit();
    var rhs_view = strided_rhs.asStrided(&.{4}, &.{2}, 0) catch return failedReport();
    defer rhs_view.deinit();
    const lhs_descriptor = describeHostViewMemRef(f32, lhs_view, "lhs") catch return failedReport();
    const rhs_descriptor = describeHostViewMemRef(f32, rhs_view, "rhs") catch return failedReport();
    const out_descriptor = axiom.accelerator.TensorMemRefDescriptor.init("out", 0x3000, .f32, .host, 0, &.{4}, &.{1}) catch return failedReport();
    const legality = axiom.accelerator.TensorMemRefLegalityReport.binaryElementwise(lhs_descriptor, rhs_descriptor, out_descriptor);
    if (!legality.ok()) return failedReport();
    report.strided_memref_legality_fingerprint = legality.fingerprint();
    const unary_out_descriptor = axiom.accelerator.TensorMemRefDescriptor.init("unary_out", 0x9000, .f32, .host, 0, &.{4}, &.{1}) catch return failedReport();
    const unary_legality = axiom.accelerator.TensorUnaryMemRefLegalityReport.unaryElementwise(lhs_descriptor, unary_out_descriptor);
    if (!unary_legality.ok()) return failedReport();
    report.strided_unary_memref_legality_fingerprint = unary_legality.fingerprint();
    var strided_add = tryAddViewF32(lhs_view, rhs_view) catch return failedReport();
    if (strided_add) |*out| {
        defer out.deinit();
        report.strided_add_ok = sliceClose(out.data, &.{ 11, 22, 33, 44 }, 0.0);
        report.output_fingerprint ^= hashF32Slice(out.data);
    }
    var strided_sub = trySubViewF32(rhs_view, lhs_view) catch return failedReport();
    if (strided_sub) |*out| {
        defer out.deinit();
        report.strided_sub_ok = sliceClose(out.data, &.{ 9, 18, 27, 36 }, 0.0);
        report.output_fingerprint ^= hashF32Slice(out.data);
    }
    var strided_mul = tryMulViewF32(lhs_view, rhs_view) catch return failedReport();
    if (strided_mul) |*out| {
        defer out.deinit();
        report.strided_mul_ok = sliceClose(out.data, &.{ 10, 40, 90, 160 }, 0.0);
        report.output_fingerprint ^= hashF32Slice(out.data);
    }
    var strided_div = tryDivViewF32(rhs_view, lhs_view) catch return failedReport();
    if (strided_div) |*out| {
        defer out.deinit();
        report.strided_div_ok = sliceClose(out.data, &.{ 10, 10, 10, 10 }, 0.0);
        report.output_fingerprint ^= hashF32Slice(out.data);
    }
    var strided_abs = tryAbsViewF32(lhs_view) catch return failedReport();
    if (strided_abs) |*out| {
        defer out.deinit();
        report.strided_abs_ok = sliceClose(out.data, &.{ 1, 2, 3, 4 }, 0.0);
        report.output_fingerprint ^= hashF32Slice(out.data);
    }
    var strided_sqrt = trySqrtViewF32(lhs_view) catch return failedReport();
    if (strided_sqrt) |*out| {
        defer out.deinit();
        report.strided_sqrt_ok = sliceClose(out.data, &.{ 1, 1.4142135, 1.7320508, 2 }, 1e-5);
        report.output_fingerprint ^= hashF32Slice(out.data);
    }
    var strided_exp = tryExpViewF32(lhs_view) catch return failedReport();
    if (strided_exp) |*out| {
        defer out.deinit();
        report.strided_exp_ok = sliceClose(out.data, &.{ 2.7182817, 7.389056, 20.085537, 54.59815 }, 0.05);
        report.output_fingerprint ^= hashF32Slice(out.data);
    }
    var strided_log = tryLogViewF32(lhs_view) catch return failedReport();
    if (strided_log) |*out| {
        defer out.deinit();
        report.strided_log_ok = sliceClose(out.data, &.{
            0,
            std.math.log(f32, std.math.e, 2),
            std.math.log(f32, std.math.e, 3),
            std.math.log(f32, std.math.e, 4),
        }, 0.01);
        report.output_fingerprint ^= hashF32Slice(out.data);
    }
    const scalar_descriptor = axiom.accelerator.TensorMemRefDescriptor.init("scalar", 0x4000, .f32, .host, 0, &.{4}, &.{0}) catch return failedReport();
    const scalar_out_descriptor = axiom.accelerator.TensorMemRefDescriptor.init("scalar_out", 0x5000, .f32, .host, 0, &.{4}, &.{1}) catch return failedReport();
    const scalar_legality = axiom.accelerator.TensorMemRefLegalityReport.binaryElementwise(lhs_descriptor, scalar_descriptor, scalar_out_descriptor);
    if (!scalar_legality.ok()) return failedReport();
    report.strided_scalar_memref_legality_fingerprint = scalar_legality.fingerprint();
    var strided_scalar_add = tryViewScalarF32(.add, lhs_view, 2.0, false) catch return failedReport();
    if (strided_scalar_add) |*out| {
        defer out.deinit();
        report.strided_scalar_add_ok = sliceClose(out.data, &.{ 3, 4, 5, 6 }, 0.0);
        report.output_fingerprint ^= hashF32Slice(out.data);
    }
    var strided_scalar_sub = tryViewScalarF32(.sub, lhs_view, 2.0, false) catch return failedReport();
    if (strided_scalar_sub) |*out| {
        defer out.deinit();
        report.strided_scalar_sub_ok = sliceClose(out.data, &.{ -1, 0, 1, 2 }, 0.0);
        report.output_fingerprint ^= hashF32Slice(out.data);
    }
    var strided_scalar_mul = tryViewScalarF32(.mul, lhs_view, 2.0, false) catch return failedReport();
    if (strided_scalar_mul) |*out| {
        defer out.deinit();
        report.strided_scalar_mul_ok = sliceClose(out.data, &.{ 2, 4, 6, 8 }, 0.0);
        report.output_fingerprint ^= hashF32Slice(out.data);
    }
    var strided_scalar_div = tryViewScalarF32(.div, rhs_view, 10.0, false) catch return failedReport();
    if (strided_scalar_div) |*out| {
        defer out.deinit();
        report.strided_scalar_div_ok = sliceClose(out.data, &.{ 1, 2, 3, 4 }, 0.0);
        report.output_fingerprint ^= hashF32Slice(out.data);
    }

    var f64_strided_lhs = array_mod.Array(f64).fromSlice(allocator, &.{ 1, 99, 2, 99, 3, 99, 4, 99 }, &.{8}) catch return failedReport();
    defer f64_strided_lhs.deinit();
    var f64_strided_rhs = array_mod.Array(f64).fromSlice(allocator, &.{ 10, 99, 20, 99, 30, 99, 40, 99 }, &.{8}) catch return failedReport();
    defer f64_strided_rhs.deinit();
    var f64_lhs_view = f64_strided_lhs.asStrided(&.{4}, &.{2}, 0) catch return failedReport();
    defer f64_lhs_view.deinit();
    var f64_rhs_view = f64_strided_rhs.asStrided(&.{4}, &.{2}, 0) catch return failedReport();
    defer f64_rhs_view.deinit();
    const f64_lhs_descriptor = describeHostViewMemRef(f64, f64_lhs_view, "lhs64") catch return failedReport();
    const f64_rhs_descriptor = describeHostViewMemRef(f64, f64_rhs_view, "rhs64") catch return failedReport();
    const f64_out_descriptor = axiom.accelerator.TensorMemRefDescriptor.init("out64", 0x6000, .f64, .host, 0, &.{4}, &.{1}) catch return failedReport();
    const f64_legality = axiom.accelerator.TensorMemRefLegalityReport.binaryElementwise(f64_lhs_descriptor, f64_rhs_descriptor, f64_out_descriptor);
    if (!f64_legality.ok()) return failedReport();
    report.f64_strided_memref_legality_fingerprint = f64_legality.fingerprint();
    const f64_unary_out_descriptor = axiom.accelerator.TensorMemRefDescriptor.init("unary_out64", 0x9000, .f64, .host, 0, &.{4}, &.{1}) catch return failedReport();
    const f64_unary_legality = axiom.accelerator.TensorUnaryMemRefLegalityReport.unaryElementwise(f64_lhs_descriptor, f64_unary_out_descriptor);
    if (!f64_unary_legality.ok()) return failedReport();
    report.f64_strided_unary_memref_legality_fingerprint = f64_unary_legality.fingerprint();
    var f64_strided_add = tryAddViewF64(f64_lhs_view, f64_rhs_view) catch return failedReport();
    if (f64_strided_add) |*out| {
        defer out.deinit();
        report.f64_strided_add_ok = sliceCloseF64(out.data, &.{ 11, 22, 33, 44 }, 0.0);
        report.output_fingerprint ^= hashF64Slice(out.data);
    }
    var f64_strided_sub = trySubViewF64(f64_rhs_view, f64_lhs_view) catch return failedReport();
    if (f64_strided_sub) |*out| {
        defer out.deinit();
        report.f64_strided_sub_ok = sliceCloseF64(out.data, &.{ 9, 18, 27, 36 }, 0.0);
        report.output_fingerprint ^= hashF64Slice(out.data);
    }
    var f64_strided_mul = tryMulViewF64(f64_lhs_view, f64_rhs_view) catch return failedReport();
    if (f64_strided_mul) |*out| {
        defer out.deinit();
        report.f64_strided_mul_ok = sliceCloseF64(out.data, &.{ 10, 40, 90, 160 }, 0.0);
        report.output_fingerprint ^= hashF64Slice(out.data);
    }
    var f64_strided_div = tryDivViewF64(f64_rhs_view, f64_lhs_view) catch return failedReport();
    if (f64_strided_div) |*out| {
        defer out.deinit();
        report.f64_strided_div_ok = sliceCloseF64(out.data, &.{ 10, 10, 10, 10 }, 0.0);
        report.output_fingerprint ^= hashF64Slice(out.data);
    }
    var f64_strided_neg_source = array_mod.Array(f64).fromSlice(allocator, &.{ -1, 99, -2, 99, -3, 99, -4, 99 }, &.{8}) catch return failedReport();
    defer f64_strided_neg_source.deinit();
    var f64_neg_view = f64_strided_neg_source.asStrided(&.{4}, &.{2}, 0) catch return failedReport();
    defer f64_neg_view.deinit();
    var f64_strided_abs = tryAbsViewF64(f64_neg_view) catch return failedReport();
    if (f64_strided_abs) |*out| {
        defer out.deinit();
        report.f64_strided_abs_ok = sliceCloseF64(out.data, &.{ 1, 2, 3, 4 }, 0.0);
        report.output_fingerprint ^= hashF64Slice(out.data);
    }
    var f64_strided_sqrt = trySqrtViewF64(f64_lhs_view) catch return failedReport();
    if (f64_strided_sqrt) |*out| {
        defer out.deinit();
        report.f64_strided_sqrt_ok = sliceCloseF64(out.data, &.{ 1, 1.4142135623730951, 1.7320508075688772, 2 }, 1e-12);
        report.output_fingerprint ^= hashF64Slice(out.data);
    }
    var f64_strided_exp = tryExpViewF64(f64_lhs_view) catch return failedReport();
    if (f64_strided_exp) |*out| {
        defer out.deinit();
        report.f64_strided_exp_ok = sliceCloseF64(out.data, &.{ 2.718281828459045, 7.38905609893065, 20.085536923187668, 54.598150033144236 }, 0.01);
        report.output_fingerprint ^= hashF64Slice(out.data);
    }
    const f64_scalar_descriptor = axiom.accelerator.TensorMemRefDescriptor.init("scalar64", 0x7000, .f64, .host, 0, &.{4}, &.{0}) catch return failedReport();
    const f64_scalar_out_descriptor = axiom.accelerator.TensorMemRefDescriptor.init("scalar_out64", 0x8000, .f64, .host, 0, &.{4}, &.{1}) catch return failedReport();
    const f64_scalar_legality = axiom.accelerator.TensorMemRefLegalityReport.binaryElementwise(f64_lhs_descriptor, f64_scalar_descriptor, f64_scalar_out_descriptor);
    if (!f64_scalar_legality.ok()) return failedReport();
    report.f64_strided_scalar_memref_legality_fingerprint = f64_scalar_legality.fingerprint();
    var f64_strided_scalar_add = tryViewScalarF64(.add, f64_lhs_view, 2.0, false) catch return failedReport();
    if (f64_strided_scalar_add) |*out| {
        defer out.deinit();
        report.f64_strided_scalar_add_ok = sliceCloseF64(out.data, &.{ 3, 4, 5, 6 }, 0.0);
        report.output_fingerprint ^= hashF64Slice(out.data);
    }
    var f64_strided_scalar_sub = tryViewScalarF64(.sub, f64_lhs_view, 2.0, false) catch return failedReport();
    if (f64_strided_scalar_sub) |*out| {
        defer out.deinit();
        report.f64_strided_scalar_sub_ok = sliceCloseF64(out.data, &.{ -1, 0, 1, 2 }, 0.0);
        report.output_fingerprint ^= hashF64Slice(out.data);
    }
    var f64_strided_scalar_mul = tryViewScalarF64(.mul, f64_lhs_view, 2.0, false) catch return failedReport();
    if (f64_strided_scalar_mul) |*out| {
        defer out.deinit();
        report.f64_strided_scalar_mul_ok = sliceCloseF64(out.data, &.{ 2, 4, 6, 8 }, 0.0);
        report.output_fingerprint ^= hashF64Slice(out.data);
    }
    var f64_strided_scalar_div = tryViewScalarF64(.div, f64_rhs_view, 10.0, false) catch return failedReport();
    if (f64_strided_scalar_div) |*out| {
        defer out.deinit();
        report.f64_strided_scalar_div_ok = sliceCloseF64(out.data, &.{ 1, 2, 3, 4 }, 0.0);
        report.output_fingerprint ^= hashF64Slice(out.data);
    }

    var f16_strided_lhs = array_mod.Array(f16).fromSlice(allocator, &.{
        @as(f16, 1), @as(f16, 99),
        @as(f16, 2), @as(f16, 99),
        @as(f16, 3), @as(f16, 99),
        @as(f16, 4), @as(f16, 99),
    }, &.{8}) catch return failedReport();
    defer f16_strided_lhs.deinit();
    var f16_strided_rhs = array_mod.Array(f16).fromSlice(allocator, &.{
        @as(f16, 10), @as(f16, 99),
        @as(f16, 20), @as(f16, 99),
        @as(f16, 30), @as(f16, 99),
        @as(f16, 40), @as(f16, 99),
    }, &.{8}) catch return failedReport();
    defer f16_strided_rhs.deinit();
    var f16_lhs_view = f16_strided_lhs.asStrided(&.{4}, &.{2}, 0) catch return failedReport();
    defer f16_lhs_view.deinit();
    var f16_rhs_view = f16_strided_rhs.asStrided(&.{4}, &.{2}, 0) catch return failedReport();
    defer f16_rhs_view.deinit();
    const f16_lhs_descriptor = describeHostViewMemRef(f16, f16_lhs_view, "lhs16") catch return failedReport();
    const f16_rhs_descriptor = describeHostViewMemRef(f16, f16_rhs_view, "rhs16") catch return failedReport();
    const f16_out_descriptor = axiom.accelerator.TensorMemRefDescriptor.init("out16", 0x9100, .f16, .host, 0, &.{4}, &.{1}) catch return failedReport();
    const f16_legality = axiom.accelerator.TensorMemRefLegalityReport.binaryElementwise(f16_lhs_descriptor, f16_rhs_descriptor, f16_out_descriptor);
    if (!f16_legality.ok()) return failedReport();
    report.f16_strided_memref_legality_fingerprint = f16_legality.fingerprint();
    const f16_unary_out_descriptor = axiom.accelerator.TensorMemRefDescriptor.init("unary_out16", 0x9400, .f16, .host, 0, &.{4}, &.{1}) catch return failedReport();
    const f16_unary_legality = axiom.accelerator.TensorUnaryMemRefLegalityReport.unaryElementwise(f16_lhs_descriptor, f16_unary_out_descriptor);
    if (!f16_unary_legality.ok()) return failedReport();
    report.f16_strided_unary_memref_legality_fingerprint = f16_unary_legality.fingerprint();
    var f16_strided_add = tryAddViewF16(f16_lhs_view, f16_rhs_view) catch return failedReport();
    if (f16_strided_add) |*out| {
        defer out.deinit();
        report.f16_strided_add_ok = f16Close(out.data, &.{ 11, 22, 33, 44 }, 0.02);
        report.output_fingerprint ^= hashF16Slice(out.data);
    }
    var f16_strided_sub = trySubViewF16(f16_rhs_view, f16_lhs_view) catch return failedReport();
    if (f16_strided_sub) |*out| {
        defer out.deinit();
        report.f16_strided_sub_ok = f16Close(out.data, &.{ 9, 18, 27, 36 }, 0.02);
        report.output_fingerprint ^= hashF16Slice(out.data);
    }
    var f16_strided_mul = tryMulViewF16(f16_lhs_view, f16_rhs_view) catch return failedReport();
    if (f16_strided_mul) |*out| {
        defer out.deinit();
        report.f16_strided_mul_ok = f16Close(out.data, &.{ 10, 40, 90, 160 }, 0.02);
        report.output_fingerprint ^= hashF16Slice(out.data);
    }
    var f16_strided_div = tryDivViewF16(f16_rhs_view, f16_lhs_view) catch return failedReport();
    if (f16_strided_div) |*out| {
        defer out.deinit();
        report.f16_strided_div_ok = f16Close(out.data, &.{ 10, 10, 10, 10 }, 0.02);
        report.output_fingerprint ^= hashF16Slice(out.data);
    }
    var f16_strided_neg_source = array_mod.Array(f16).fromSlice(allocator, &.{
        @as(f16, -1), @as(f16, 99),
        @as(f16, -2), @as(f16, 99),
        @as(f16, -3), @as(f16, 99),
        @as(f16, -4), @as(f16, 99),
    }, &.{8}) catch return failedReport();
    defer f16_strided_neg_source.deinit();
    var f16_neg_view = f16_strided_neg_source.asStrided(&.{4}, &.{2}, 0) catch return failedReport();
    defer f16_neg_view.deinit();
    var f16_strided_abs = tryAbsViewF16(f16_neg_view) catch return failedReport();
    if (f16_strided_abs) |*out| {
        defer out.deinit();
        report.f16_strided_abs_ok = f16Close(out.data, &.{ 1, 2, 3, 4 }, 0.02);
        report.output_fingerprint ^= hashF16Slice(out.data);
    }
    var f16_strided_sqrt = trySqrtViewF16(f16_lhs_view) catch return failedReport();
    if (f16_strided_sqrt) |*out| {
        defer out.deinit();
        report.f16_strided_sqrt_ok = f16Close(out.data, &.{ 1, 1.4142135, 1.7320508, 2 }, 0.03);
        report.output_fingerprint ^= hashF16Slice(out.data);
    }
    var f16_strided_exp = tryExpViewF16(f16_lhs_view) catch return failedReport();
    if (f16_strided_exp) |*out| {
        defer out.deinit();
        report.f16_strided_exp_ok = f16Close(out.data, &.{ 2.7182817, 7.389056, 20.085537, 54.59815 }, 0.25);
        report.output_fingerprint ^= hashF16Slice(out.data);
    }
    const f16_scalar_descriptor = axiom.accelerator.TensorMemRefDescriptor.init("scalar16", 0x9200, .f16, .host, 0, &.{4}, &.{0}) catch return failedReport();
    const f16_scalar_out_descriptor = axiom.accelerator.TensorMemRefDescriptor.init("scalar_out16", 0x9300, .f16, .host, 0, &.{4}, &.{1}) catch return failedReport();
    const f16_scalar_legality = axiom.accelerator.TensorMemRefLegalityReport.binaryElementwise(f16_lhs_descriptor, f16_scalar_descriptor, f16_scalar_out_descriptor);
    if (!f16_scalar_legality.ok()) return failedReport();
    report.f16_strided_scalar_memref_legality_fingerprint = f16_scalar_legality.fingerprint();
    var f16_strided_scalar_add = tryViewScalarF16(.add, f16_lhs_view, @as(f16, 2.0), false) catch return failedReport();
    if (f16_strided_scalar_add) |*out| {
        defer out.deinit();
        report.f16_strided_scalar_add_ok = f16Close(out.data, &.{ 3, 4, 5, 6 }, 0.02);
        report.output_fingerprint ^= hashF16Slice(out.data);
    }
    var f16_strided_scalar_sub = tryViewScalarF16(.sub, f16_lhs_view, @as(f16, 2.0), false) catch return failedReport();
    if (f16_strided_scalar_sub) |*out| {
        defer out.deinit();
        report.f16_strided_scalar_sub_ok = f16Close(out.data, &.{ -1, 0, 1, 2 }, 0.02);
        report.output_fingerprint ^= hashF16Slice(out.data);
    }
    var f16_strided_scalar_mul = tryViewScalarF16(.mul, f16_lhs_view, @as(f16, 2.0), false) catch return failedReport();
    if (f16_strided_scalar_mul) |*out| {
        defer out.deinit();
        report.f16_strided_scalar_mul_ok = f16Close(out.data, &.{ 2, 4, 6, 8 }, 0.02);
        report.output_fingerprint ^= hashF16Slice(out.data);
    }
    var f16_strided_scalar_div = tryViewScalarF16(.div, f16_rhs_view, @as(f16, 10.0), false) catch return failedReport();
    if (f16_strided_scalar_div) |*out| {
        defer out.deinit();
        report.f16_strided_scalar_div_ok = f16Close(out.data, &.{ 1, 2, 3, 4 }, 0.02);
        report.output_fingerprint ^= hashF16Slice(out.data);
    }

    var bf16_strided_lhs = array_mod.Array(BFloat16).fromSlice(allocator, &.{
        BFloat16.fromF32(1), BFloat16.fromF32(99),
        BFloat16.fromF32(2), BFloat16.fromF32(99),
        BFloat16.fromF32(3), BFloat16.fromF32(99),
        BFloat16.fromF32(4), BFloat16.fromF32(99),
    }, &.{8}) catch return failedReport();
    defer bf16_strided_lhs.deinit();
    var bf16_strided_rhs = array_mod.Array(BFloat16).fromSlice(allocator, &.{
        BFloat16.fromF32(10), BFloat16.fromF32(99),
        BFloat16.fromF32(20), BFloat16.fromF32(99),
        BFloat16.fromF32(30), BFloat16.fromF32(99),
        BFloat16.fromF32(40), BFloat16.fromF32(99),
    }, &.{8}) catch return failedReport();
    defer bf16_strided_rhs.deinit();
    var bf16_lhs_view = bf16_strided_lhs.asStrided(&.{4}, &.{2}, 0) catch return failedReport();
    defer bf16_lhs_view.deinit();
    var bf16_rhs_view = bf16_strided_rhs.asStrided(&.{4}, &.{2}, 0) catch return failedReport();
    defer bf16_rhs_view.deinit();
    const bf16_lhs_descriptor = describeHostViewMemRef(BFloat16, bf16_lhs_view, "lhs_bf16") catch return failedReport();
    const bf16_rhs_descriptor = describeHostViewMemRef(BFloat16, bf16_rhs_view, "rhs_bf16") catch return failedReport();
    const bf16_out_descriptor = axiom.accelerator.TensorMemRefDescriptor.init("out_bf16", 0xa100, .bf16, .host, 0, &.{4}, &.{1}) catch return failedReport();
    const bf16_legality = axiom.accelerator.TensorMemRefLegalityReport.binaryElementwise(bf16_lhs_descriptor, bf16_rhs_descriptor, bf16_out_descriptor);
    if (!bf16_legality.ok()) return failedReport();
    report.bf16_strided_memref_legality_fingerprint = bf16_legality.fingerprint();
    const bf16_unary_out_descriptor = axiom.accelerator.TensorMemRefDescriptor.init("unary_out_bf16", 0xa400, .bf16, .host, 0, &.{4}, &.{1}) catch return failedReport();
    const bf16_unary_legality = axiom.accelerator.TensorUnaryMemRefLegalityReport.unaryElementwise(bf16_lhs_descriptor, bf16_unary_out_descriptor);
    if (!bf16_unary_legality.ok()) return failedReport();
    report.bf16_strided_unary_memref_legality_fingerprint = bf16_unary_legality.fingerprint();
    var bf16_strided_add = tryAddViewBF16(bf16_lhs_view, bf16_rhs_view) catch return failedReport();
    if (bf16_strided_add) |*out| {
        defer out.deinit();
        report.bf16_strided_add_ok = bf16Close(out.data, &.{ 11, 22, 33, 44 }, 0.125);
        report.output_fingerprint ^= hashBF16Slice(out.data);
    }
    var bf16_strided_sub = trySubViewBF16(bf16_rhs_view, bf16_lhs_view) catch return failedReport();
    if (bf16_strided_sub) |*out| {
        defer out.deinit();
        report.bf16_strided_sub_ok = bf16Close(out.data, &.{ 9, 18, 27, 36 }, 0.125);
        report.output_fingerprint ^= hashBF16Slice(out.data);
    }
    var bf16_strided_mul = tryMulViewBF16(bf16_lhs_view, bf16_rhs_view) catch return failedReport();
    if (bf16_strided_mul) |*out| {
        defer out.deinit();
        report.bf16_strided_mul_ok = bf16Close(out.data, &.{ 10, 40, 90, 160 }, 0.125);
        report.output_fingerprint ^= hashBF16Slice(out.data);
    }
    var bf16_strided_div = tryDivViewBF16(bf16_rhs_view, bf16_lhs_view) catch return failedReport();
    if (bf16_strided_div) |*out| {
        defer out.deinit();
        report.bf16_strided_div_ok = bf16Close(out.data, &.{ 10, 10, 10, 10 }, 0.125);
        report.output_fingerprint ^= hashBF16Slice(out.data);
    }
    var bf16_strided_neg_source = array_mod.Array(BFloat16).fromSlice(allocator, &.{
        BFloat16.fromF32(-1), BFloat16.fromF32(99),
        BFloat16.fromF32(-2), BFloat16.fromF32(99),
        BFloat16.fromF32(-3), BFloat16.fromF32(99),
        BFloat16.fromF32(-4), BFloat16.fromF32(99),
    }, &.{8}) catch return failedReport();
    defer bf16_strided_neg_source.deinit();
    var bf16_neg_view = bf16_strided_neg_source.asStrided(&.{4}, &.{2}, 0) catch return failedReport();
    defer bf16_neg_view.deinit();
    var bf16_strided_abs = tryAbsViewBF16(bf16_neg_view) catch return failedReport();
    if (bf16_strided_abs) |*out| {
        defer out.deinit();
        report.bf16_strided_abs_ok = bf16Close(out.data, &.{ 1, 2, 3, 4 }, 0.125);
        report.output_fingerprint ^= hashBF16Slice(out.data);
    }
    var bf16_strided_sqrt = trySqrtViewBF16(bf16_lhs_view) catch return failedReport();
    if (bf16_strided_sqrt) |*out| {
        defer out.deinit();
        report.bf16_strided_sqrt_ok = bf16Close(out.data, &.{ 1, 1.4142135, 1.7320508, 2 }, 0.125);
        report.output_fingerprint ^= hashBF16Slice(out.data);
    }
    var bf16_strided_exp = tryExpViewBF16(bf16_lhs_view) catch return failedReport();
    if (bf16_strided_exp) |*out| {
        defer out.deinit();
        report.bf16_strided_exp_ok = bf16Close(out.data, &.{ 2.7182817, 7.389056, 20.085537, 54.59815 }, 0.75);
        report.output_fingerprint ^= hashBF16Slice(out.data);
    }
    const bf16_scalar_descriptor = axiom.accelerator.TensorMemRefDescriptor.init("scalar_bf16", 0xa200, .bf16, .host, 0, &.{4}, &.{0}) catch return failedReport();
    const bf16_scalar_out_descriptor = axiom.accelerator.TensorMemRefDescriptor.init("scalar_out_bf16", 0xa300, .bf16, .host, 0, &.{4}, &.{1}) catch return failedReport();
    const bf16_scalar_legality = axiom.accelerator.TensorMemRefLegalityReport.binaryElementwise(bf16_lhs_descriptor, bf16_scalar_descriptor, bf16_scalar_out_descriptor);
    if (!bf16_scalar_legality.ok()) return failedReport();
    report.bf16_strided_scalar_memref_legality_fingerprint = bf16_scalar_legality.fingerprint();
    var bf16_strided_scalar_add = tryViewScalarBF16(.add, bf16_lhs_view, BFloat16.fromF32(2.0), false) catch return failedReport();
    if (bf16_strided_scalar_add) |*out| {
        defer out.deinit();
        report.bf16_strided_scalar_add_ok = bf16Close(out.data, &.{ 3, 4, 5, 6 }, 0.125);
        report.output_fingerprint ^= hashBF16Slice(out.data);
    }
    var bf16_strided_scalar_sub = tryViewScalarBF16(.sub, bf16_lhs_view, BFloat16.fromF32(2.0), false) catch return failedReport();
    if (bf16_strided_scalar_sub) |*out| {
        defer out.deinit();
        report.bf16_strided_scalar_sub_ok = bf16Close(out.data, &.{ -1, 0, 1, 2 }, 0.125);
        report.output_fingerprint ^= hashBF16Slice(out.data);
    }
    var bf16_strided_scalar_mul = tryViewScalarBF16(.mul, bf16_lhs_view, BFloat16.fromF32(2.0), false) catch return failedReport();
    if (bf16_strided_scalar_mul) |*out| {
        defer out.deinit();
        report.bf16_strided_scalar_mul_ok = bf16Close(out.data, &.{ 2, 4, 6, 8 }, 0.125);
        report.output_fingerprint ^= hashBF16Slice(out.data);
    }
    var bf16_strided_scalar_div = tryViewScalarBF16(.div, bf16_rhs_view, BFloat16.fromF32(10.0), false) catch return failedReport();
    if (bf16_strided_scalar_div) |*out| {
        defer out.deinit();
        report.bf16_strided_scalar_div_ok = bf16Close(out.data, &.{ 1, 2, 3, 4 }, 0.125);
        report.output_fingerprint ^= hashBF16Slice(out.data);
    }

    var saxpy_out = trySaxpyF32(2.0, lhs, rhs) catch return failedReport();
    if (saxpy_out) |*out| {
        defer out.deinit();
        report.saxpy_ok = sliceClose(out.data, &.{ 12, 24, 36, 48 }, 0.0);
        report.max_abs_error = @max(report.max_abs_error, maxAbsError(out.data, &.{ 12, 24, 36, 48 }));
        report.output_fingerprint ^= hashF32Slice(out.data);
    }

    var mat_lhs = array_mod.Array(f32).fromSlice(allocator, &.{
        1, 2, 3,
        4, 5, 6,
    }, &.{ 2, 3 }) catch return failedReport();
    defer mat_lhs.deinit();
    var mat_rhs = array_mod.Array(f32).fromSlice(allocator, &.{
        7,  8,
        9,  10,
        11, 12,
    }, &.{ 3, 2 }) catch return failedReport();
    defer mat_rhs.deinit();
    var matmul_out = tryMatmulF32(mat_lhs, mat_rhs) catch return failedReport();
    if (matmul_out) |*out| {
        defer out.deinit();
        report.matmul_ok = sliceClose(out.data, &.{ 58, 64, 139, 154 }, 0.0);
        report.matmul_tile_ir_ok = report.matmul_ok;
        report.max_abs_error = @max(report.max_abs_error, maxAbsError(out.data, &.{ 58, 64, 139, 154 }));
        report.output_fingerprint ^= hashF32Slice(out.data);
    }

    var f16_lhs = array_mod.Array(f16).fromSlice(allocator, &.{
        @as(f16, 1.0),
        @as(f16, 2.0),
        @as(f16, 3.0),
        @as(f16, 4.0),
    }, &.{ 2, 2 }) catch return failedReport();
    defer f16_lhs.deinit();
    var f16_rhs = array_mod.Array(f16).fromSlice(allocator, &.{
        @as(f16, 10.0),
        @as(f16, 20.0),
        @as(f16, 30.0),
        @as(f16, 40.0),
    }, &.{ 2, 2 }) catch return failedReport();
    defer f16_rhs.deinit();
    var f16_add_out = tryAddF16(f16_lhs, f16_rhs) catch return failedReport();
    if (f16_add_out) |*out| {
        defer out.deinit();
        report.f16_add_ok = f16Close(out.data, &.{ 11, 22, 33, 44 }, 0.02);
        report.output_fingerprint ^= hashF16Slice(out.data);
    }
    var f16_matmul_out = tryMatmulF16(f16_lhs, f16_rhs) catch return failedReport();
    if (f16_matmul_out) |*out| {
        defer out.deinit();
        report.f16_matmul_ok = f16Close(out.data, &.{ 70, 100, 150, 220 }, 0.25);
        report.output_fingerprint ^= hashF16Slice(out.data);
    }
    report.typed_f16_gemm_plan = planTypedGemmF16(f16_lhs, f16_rhs) catch return failedReport();
    if (f16_add_out != null and f16_matmul_out != null) {
        report.f16_native_execution_fingerprint =
            nativeF16BinaryExecutionFingerprint(allocator, .add, f16_lhs, f16_rhs) catch 0;
        const typed_f16_runtime = typedF16MatmulRuntimeEvidence(allocator, f16_lhs, f16_rhs) catch return failedReport();
        report.f16_widened_execution_fingerprint =
            (widenedF16BinaryProvenanceFingerprint(allocator, "add", .add, f16_lhs, f16_rhs) catch return failedReport()) ^
            typed_f16_runtime.fingerprint;
        report.typed_f16_gemm_route_fingerprint = typed_f16_runtime.route_fingerprint;
        report.typed_f16_gemm_route = typed_f16_runtime.route;
    }

    var bf16_lhs = array_mod.Array(BFloat16).fromSlice(allocator, &.{
        BFloat16.fromF32(1.0),
        BFloat16.fromF32(2.0),
        BFloat16.fromF32(3.0),
        BFloat16.fromF32(4.0),
    }, &.{ 2, 2 }) catch return failedReport();
    defer bf16_lhs.deinit();
    var bf16_rhs = array_mod.Array(BFloat16).fromSlice(allocator, &.{
        BFloat16.fromF32(10.0),
        BFloat16.fromF32(20.0),
        BFloat16.fromF32(30.0),
        BFloat16.fromF32(40.0),
    }, &.{ 2, 2 }) catch return failedReport();
    defer bf16_rhs.deinit();
    var bf16_add_out = tryAddBF16(bf16_lhs, bf16_rhs) catch return failedReport();
    if (bf16_add_out) |*out| {
        defer out.deinit();
        report.bf16_add_ok = bf16Close(out.data, &.{ 11, 22, 33, 44 }, 0.125);
        report.output_fingerprint ^= hashBF16Slice(out.data);
    }
    var bf16_matmul_out = tryMatmulBF16(bf16_lhs, bf16_rhs) catch return failedReport();
    if (bf16_matmul_out) |*out| {
        defer out.deinit();
        report.bf16_matmul_ok = bf16Close(out.data, &.{ 70, 100, 150, 220 }, 0.5);
        report.output_fingerprint ^= hashBF16Slice(out.data);
    }
    report.typed_bf16_gemm_plan = planTypedGemmBF16(bf16_lhs, bf16_rhs) catch return failedReport();
    if (bf16_add_out != null and bf16_matmul_out != null) {
        report.bf16_native_execution_fingerprint =
            nativeBF16BinaryExecutionFingerprint(allocator, .add, bf16_lhs, bf16_rhs) catch 0;
        const typed_bf16_runtime = typedBF16MatmulRuntimeEvidence(allocator, bf16_lhs, bf16_rhs) catch return failedReport();
        report.bf16_widened_execution_fingerprint =
            (widenedBF16BinaryProvenanceFingerprint(allocator, "add", .add, bf16_lhs, bf16_rhs) catch return failedReport()) ^
            typed_bf16_runtime.fingerprint;
        report.typed_bf16_gemm_route_fingerprint = typed_bf16_runtime.route_fingerprint;
        report.typed_bf16_gemm_route = typed_bf16_runtime.route;
    }

    if (report.executionEvidenceOk()) {
        report.status = .ran;
        report.issue_count = 0;
    } else if (add_out == null and sub_out == null and mul_out == null and div_out == null and saxpy_out == null and matmul_out == null and f16_add_out == null and f16_matmul_out == null and bf16_add_out == null and bf16_matmul_out == null and scalar_add_out == null and scalar_mul_out == null and scalar_saxpy_out == null and strided_add == null and strided_sub == null and strided_mul == null and strided_div == null and strided_abs == null and strided_sqrt == null and strided_exp == null and strided_scalar_add == null and strided_scalar_sub == null and strided_scalar_mul == null and strided_scalar_div == null and f64_strided_add == null and f64_strided_sub == null and f64_strided_mul == null and f64_strided_div == null and f64_strided_abs == null and f64_strided_sqrt == null and f64_strided_exp == null and f64_strided_scalar_add == null and f64_strided_scalar_sub == null and f64_strided_scalar_mul == null and f64_strided_scalar_div == null and f16_strided_add == null and f16_strided_sub == null and f16_strided_mul == null and f16_strided_div == null and f16_strided_scalar_add == null and f16_strided_scalar_sub == null and f16_strided_scalar_mul == null and f16_strided_scalar_div == null and bf16_strided_add == null and bf16_strided_sub == null and bf16_strided_mul == null and bf16_strided_div == null and bf16_strided_scalar_add == null and bf16_strided_scalar_sub == null and bf16_strided_scalar_mul == null and bf16_strided_scalar_div == null) {
        report.status = .skipped;
        report.issue_count = 0;
    } else {
        report.status = .failed;
        report.issue_count = report.executionIssueCount();
    }
    return report;
}

fn tryBinaryViewF32(op: BinaryOp, lhs: array_mod.ArrayView(f32), rhs: array_mod.ArrayView(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    if (!build_options.enable_axiom_cuda) return null;
    if (!supportedOneDimensionalView(lhs) or !supportedOneDimensionalView(rhs) or !std.mem.eql(usize, lhs.shape, rhs.shape)) return null;

    const lhs_slice = viewBackingSlice(lhs) orelse return null;
    const rhs_slice = viewBackingSlice(rhs) orelse return null;
    var out = try array_mod.Array(f32).empty(lhs.allocator, lhs.shape);
    errdefer out.deinit();
    const lhs_descriptor = describeHostViewMemRef(f32, lhs, "lhs") catch {
        out.deinit();
        return null;
    };
    const rhs_descriptor = describeHostViewMemRef(f32, rhs, "rhs") catch {
        out.deinit();
        return null;
    };
    const out_descriptor = describeHostArrayMemRef(f32, out, "out") catch {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(lhs.allocator);
    const result = runtime.runTensorElementwiseBinaryMemRefsF32(lhs_slice, rhs_slice, out.data, lhs_descriptor, rhs_descriptor, out_descriptor, .{
        .op = axiomBinaryOp(op),
        .kernel_symbol = switch (op) {
            .add => "vectra_axiom_strided_add",
            .sub => "vectra_axiom_strided_sub",
            .mul => "vectra_axiom_strided_mul",
            .div => "vectra_axiom_strided_div",
        },
    }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => {
            out.deinit();
            return null;
        },
    };
    if (!result.verified) {
        out.deinit();
        return null;
    }
    return out;
}

fn tryBinaryViewScalarF32(op: BinaryOp, input: array_mod.ArrayView(f32), scalar: f32, scalar_left: bool) array_mod.ArrayError!?array_mod.Array(f32) {
    if (!build_options.enable_axiom_cuda) return null;
    if (!supportedOneDimensionalView(input)) return null;

    const input_slice = viewBackingSlice(input) orelse return null;
    const scalar_values = [_]f32{scalar};
    var out = try array_mod.Array(f32).empty(input.allocator, input.shape);
    errdefer out.deinit();

    const lhs_slice = if (scalar_left) scalar_values[0..] else input_slice;
    const rhs_slice = if (scalar_left) input_slice else scalar_values[0..];
    const input_descriptor = try describeHostViewMemRef(f32, input, "input");
    const scalar_descriptor = axiom.accelerator.TensorMemRefDescriptor.init("scalar", @intCast(@intFromPtr(&scalar_values[0])), .f32, .host, 0, &.{input.shape[0]}, &.{0}) catch {
        out.deinit();
        return null;
    };
    const out_descriptor = describeHostArrayMemRef(f32, out, "out") catch {
        out.deinit();
        return null;
    };
    const lhs_descriptor = if (scalar_left) scalar_descriptor else input_descriptor;
    const rhs_descriptor = if (scalar_left) input_descriptor else scalar_descriptor;

    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(input.allocator);
    const result = runtime.runTensorElementwiseBinaryMemRefsF32(lhs_slice, rhs_slice, out.data, lhs_descriptor, rhs_descriptor, out_descriptor, .{
        .op = axiomBinaryOp(op),
        .kernel_symbol = switch (op) {
            .add => "vectra_axiom_strided_scalar_add",
            .sub => "vectra_axiom_strided_scalar_sub",
            .mul => "vectra_axiom_strided_scalar_mul",
            .div => "vectra_axiom_strided_scalar_div",
        },
    }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => {
            out.deinit();
            return null;
        },
    };
    if (!result.verified) {
        out.deinit();
        return null;
    }
    return out;
}

fn tryUnaryViewF32(op: UnaryOp, input: array_mod.ArrayView(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    if (!build_options.enable_axiom_cuda) return null;
    if (!supportedOneDimensionalView(input)) return null;

    const input_slice = viewBackingSlice(input) orelse return null;
    var out = try array_mod.Array(f32).empty(input.allocator, input.shape);
    errdefer out.deinit();
    const input_descriptor = try describeHostViewMemRef(f32, input, "input");
    const out_descriptor = describeHostArrayMemRef(f32, out, "out") catch {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(input.allocator);
    const result = runtime.runTensorUnaryElementwiseMemRefsF32(input_slice, out.data, input_descriptor, out_descriptor, .{
        .op = axiomUnaryOp(op),
    }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => {
            out.deinit();
            return null;
        },
    };
    if (!result.verified) {
        out.deinit();
        return null;
    }
    return out;
}

fn tryBinaryViewF64(op: BinaryOp, lhs: array_mod.ArrayView(f64), rhs: array_mod.ArrayView(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    if (!build_options.enable_axiom_cuda) return null;
    if (!supportedOneDimensionalViewTyped(f64, lhs) or !supportedOneDimensionalViewTyped(f64, rhs) or !std.mem.eql(usize, lhs.shape, rhs.shape)) return null;

    const lhs_slice = viewBackingSliceTyped(f64, lhs) orelse return null;
    const rhs_slice = viewBackingSliceTyped(f64, rhs) orelse return null;
    var out = try array_mod.Array(f64).empty(lhs.allocator, lhs.shape);
    errdefer out.deinit();
    const lhs_descriptor = try describeHostViewMemRef(f64, lhs, "lhs");
    const rhs_descriptor = try describeHostViewMemRef(f64, rhs, "rhs");
    const out_descriptor = describeHostArrayMemRef(f64, out, "out") catch {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(lhs.allocator);
    const result = runtime.runTensorElementwiseBinaryMemRefsNative(f64, .f64, lhs_slice, rhs_slice, out.data, lhs_descriptor, rhs_descriptor, out_descriptor, .{
        .op = axiomBinaryOp(op),
        .kernel_symbol = switch (op) {
            .add => "vectra_axiom_f64_strided_add",
            .sub => "vectra_axiom_f64_strided_sub",
            .mul => "vectra_axiom_f64_strided_mul",
            .div => "vectra_axiom_f64_strided_div",
        },
    }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => {
            out.deinit();
            return null;
        },
    };
    if (!result.ok()) {
        out.deinit();
        return null;
    }
    return out;
}

fn tryBinaryViewScalarF64(op: BinaryOp, input: array_mod.ArrayView(f64), scalar: f64, scalar_left: bool) array_mod.ArrayError!?array_mod.Array(f64) {
    if (!build_options.enable_axiom_cuda) return null;
    if (!supportedOneDimensionalViewTyped(f64, input)) return null;

    const input_slice = viewBackingSliceTyped(f64, input) orelse return null;
    const scalar_values = [_]f64{scalar};
    var out = try array_mod.Array(f64).empty(input.allocator, input.shape);
    errdefer out.deinit();

    const lhs_slice = if (scalar_left) scalar_values[0..] else input_slice;
    const rhs_slice = if (scalar_left) input_slice else scalar_values[0..];
    const input_descriptor = try describeHostViewMemRef(f64, input, "input");
    const scalar_descriptor = axiom.accelerator.TensorMemRefDescriptor.init("scalar", @intCast(@intFromPtr(&scalar_values[0])), .f64, .host, 0, &.{input.shape[0]}, &.{0}) catch {
        out.deinit();
        return null;
    };
    const out_descriptor = describeHostArrayMemRef(f64, out, "out") catch {
        out.deinit();
        return null;
    };
    const lhs_descriptor = if (scalar_left) scalar_descriptor else input_descriptor;
    const rhs_descriptor = if (scalar_left) input_descriptor else scalar_descriptor;

    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(input.allocator);
    const result = runtime.runTensorElementwiseBinaryMemRefsNative(f64, .f64, lhs_slice, rhs_slice, out.data, lhs_descriptor, rhs_descriptor, out_descriptor, .{
        .op = axiomBinaryOp(op),
        .kernel_symbol = switch (op) {
            .add => "vectra_axiom_f64_strided_scalar_add",
            .sub => "vectra_axiom_f64_strided_scalar_sub",
            .mul => "vectra_axiom_f64_strided_scalar_mul",
            .div => "vectra_axiom_f64_strided_scalar_div",
        },
    }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => {
            out.deinit();
            return null;
        },
    };
    if (!result.ok()) {
        out.deinit();
        return null;
    }
    return out;
}

fn tryUnaryViewF64(op: UnaryOp, input: array_mod.ArrayView(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    if (!build_options.enable_axiom_cuda) return null;
    if (!supportedOneDimensionalViewTyped(f64, input)) return null;

    const input_slice = viewBackingSliceTyped(f64, input) orelse return null;
    var out = try array_mod.Array(f64).empty(input.allocator, input.shape);
    errdefer out.deinit();
    const input_descriptor = try describeHostViewMemRef(f64, input, "input");
    const out_descriptor = describeHostArrayMemRef(f64, out, "out") catch {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(input.allocator);
    const result = runtime.runTensorUnaryElementwiseMemRefsF64(input_slice, out.data, input_descriptor, out_descriptor, .{
        .op = axiomUnaryOp(op),
    }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => {
            out.deinit();
            return null;
        },
    };
    if (!result.verified) {
        out.deinit();
        return null;
    }
    return out;
}

fn tryBinaryViewF16(op: BinaryOp, lhs: array_mod.ArrayView(f16), rhs: array_mod.ArrayView(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    if (!build_options.enable_axiom_cuda) return null;
    if (!supportedOneDimensionalViewTyped(f16, lhs) or !supportedOneDimensionalViewTyped(f16, rhs) or !std.mem.eql(usize, lhs.shape, rhs.shape)) return null;

    const lhs_slice = viewBackingSliceTyped(f16, lhs) orelse return null;
    const rhs_slice = viewBackingSliceTyped(f16, rhs) orelse return null;
    var out = try array_mod.Array(f16).empty(lhs.allocator, lhs.shape);
    errdefer out.deinit();
    const lhs_descriptor = try describeHostViewMemRef(f16, lhs, "lhs");
    const rhs_descriptor = try describeHostViewMemRef(f16, rhs, "rhs");
    const out_descriptor = describeHostArrayMemRef(f16, out, "out") catch {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(lhs.allocator);
    const result = runtime.runTensorElementwiseBinaryMemRefsNative(f16, .f16, lhs_slice, rhs_slice, out.data, lhs_descriptor, rhs_descriptor, out_descriptor, .{
        .op = axiomBinaryOp(op),
        .kernel_symbol = switch (op) {
            .add => "vectra_axiom_f16_strided_add",
            .sub => "vectra_axiom_f16_strided_sub",
            .mul => "vectra_axiom_f16_strided_mul",
            .div => "vectra_axiom_f16_strided_div",
        },
    }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => {
            out.deinit();
            return null;
        },
    };
    if (!result.ok()) {
        out.deinit();
        return null;
    }
    return out;
}

fn tryBinaryViewScalarF16(op: BinaryOp, input: array_mod.ArrayView(f16), scalar: f16, scalar_left: bool) array_mod.ArrayError!?array_mod.Array(f16) {
    if (!build_options.enable_axiom_cuda) return null;
    if (!supportedOneDimensionalViewTyped(f16, input)) return null;

    const input_slice = viewBackingSliceTyped(f16, input) orelse return null;
    const scalar_values = [_]f16{scalar};
    var out = try array_mod.Array(f16).empty(input.allocator, input.shape);
    errdefer out.deinit();

    const lhs_slice = if (scalar_left) scalar_values[0..] else input_slice;
    const rhs_slice = if (scalar_left) input_slice else scalar_values[0..];
    const input_descriptor = try describeHostViewMemRef(f16, input, "input");
    const scalar_descriptor = axiom.accelerator.TensorMemRefDescriptor.init("scalar", @intCast(@intFromPtr(&scalar_values[0])), .f16, .host, 0, &.{input.shape[0]}, &.{0}) catch {
        out.deinit();
        return null;
    };
    const out_descriptor = describeHostArrayMemRef(f16, out, "out") catch {
        out.deinit();
        return null;
    };
    const lhs_descriptor = if (scalar_left) scalar_descriptor else input_descriptor;
    const rhs_descriptor = if (scalar_left) input_descriptor else scalar_descriptor;

    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(input.allocator);
    const result = runtime.runTensorElementwiseBinaryMemRefsNative(f16, .f16, lhs_slice, rhs_slice, out.data, lhs_descriptor, rhs_descriptor, out_descriptor, .{
        .op = axiomBinaryOp(op),
        .kernel_symbol = switch (op) {
            .add => "vectra_axiom_f16_strided_scalar_add",
            .sub => "vectra_axiom_f16_strided_scalar_sub",
            .mul => "vectra_axiom_f16_strided_scalar_mul",
            .div => "vectra_axiom_f16_strided_scalar_div",
        },
    }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => {
            out.deinit();
            return null;
        },
    };
    if (!result.ok()) {
        out.deinit();
        return null;
    }
    return out;
}

fn tryUnaryViewF16(op: UnaryOp, input: array_mod.ArrayView(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    if (!build_options.enable_axiom_cuda) return null;
    if (!supportedOneDimensionalViewTyped(f16, input)) return null;

    const input_slice = viewBackingSliceTyped(f16, input) orelse return null;
    var out = try array_mod.Array(f16).empty(input.allocator, input.shape);
    errdefer out.deinit();
    const input_descriptor = try describeHostViewMemRef(f16, input, "input");
    const out_descriptor = describeHostArrayMemRef(f16, out, "out") catch {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(input.allocator);
    const result = runtime.runTensorUnaryElementwiseMemRefsF16(input_slice, out.data, input_descriptor, out_descriptor, .{
        .op = axiomUnaryOp(op),
    }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => {
            out.deinit();
            return null;
        },
    };
    if (!result.verified) {
        out.deinit();
        return null;
    }
    return out;
}

fn tryBinaryViewBF16(op: BinaryOp, lhs: array_mod.ArrayView(BFloat16), rhs: array_mod.ArrayView(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    if (!build_options.enable_axiom_cuda) return null;
    if (!supportedOneDimensionalViewTyped(BFloat16, lhs) or !supportedOneDimensionalViewTyped(BFloat16, rhs) or !std.mem.eql(usize, lhs.shape, rhs.shape)) return null;

    const lhs_slice = viewBackingSliceTyped(BFloat16, lhs) orelse return null;
    const rhs_slice = viewBackingSliceTyped(BFloat16, rhs) orelse return null;
    var out = try array_mod.Array(BFloat16).empty(lhs.allocator, lhs.shape);
    errdefer out.deinit();

    const lhs_bits = try lhs.allocator.alloc(u16, lhs_slice.len);
    defer lhs.allocator.free(lhs_bits);
    const rhs_bits = try lhs.allocator.alloc(u16, rhs_slice.len);
    defer lhs.allocator.free(rhs_bits);
    const out_bits = try lhs.allocator.alloc(u16, out.data.len);
    defer lhs.allocator.free(out_bits);
    for (lhs_slice, lhs_bits) |value, *slot| slot.* = value.bits;
    for (rhs_slice, rhs_bits) |value, *slot| slot.* = value.bits;
    const lhs_descriptor = describeHostBitsViewMemRef(.bf16, lhs_bits.ptr, lhs.shape, lhs.strides, "lhs") catch {
        out.deinit();
        return null;
    };
    const rhs_descriptor = describeHostBitsViewMemRef(.bf16, rhs_bits.ptr, rhs.shape, rhs.strides, "rhs") catch {
        out.deinit();
        return null;
    };
    const out_descriptor = describeHostBitsViewMemRef(.bf16, out_bits.ptr, out.shape, out.strides, "out") catch {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(lhs.allocator);
    const result = runtime.runTensorElementwiseBinaryMemRefsNative(u16, .bf16, lhs_bits, rhs_bits, out_bits, lhs_descriptor, rhs_descriptor, out_descriptor, .{
        .op = axiomBinaryOp(op),
        .kernel_symbol = switch (op) {
            .add => "vectra_axiom_bf16_strided_add",
            .sub => "vectra_axiom_bf16_strided_sub",
            .mul => "vectra_axiom_bf16_strided_mul",
            .div => "vectra_axiom_bf16_strided_div",
        },
    }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => {
            out.deinit();
            return null;
        },
    };
    if (!result.ok()) {
        out.deinit();
        return null;
    }
    for (out_bits, out.data) |bits, *slot| slot.* = .{ .bits = bits };
    return out;
}

fn tryBinaryViewScalarBF16(op: BinaryOp, input: array_mod.ArrayView(BFloat16), scalar: BFloat16, scalar_left: bool) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    if (!build_options.enable_axiom_cuda) return null;
    if (!supportedOneDimensionalViewTyped(BFloat16, input)) return null;

    const input_slice = viewBackingSliceTyped(BFloat16, input) orelse return null;
    var out = try array_mod.Array(BFloat16).empty(input.allocator, input.shape);
    errdefer out.deinit();

    const input_bits = try input.allocator.alloc(u16, input_slice.len);
    defer input.allocator.free(input_bits);
    const scalar_bits = [_]u16{scalar.bits};
    const out_bits = try input.allocator.alloc(u16, out.data.len);
    defer input.allocator.free(out_bits);
    for (input_slice, input_bits) |value, *slot| slot.* = value.bits;

    const lhs_bits = if (scalar_left) scalar_bits[0..] else input_bits;
    const rhs_bits = if (scalar_left) input_bits else scalar_bits[0..];
    const input_descriptor = describeHostBitsViewMemRef(.bf16, input_bits.ptr, input.shape, input.strides, "input") catch {
        out.deinit();
        return null;
    };
    const scalar_descriptor = axiom.accelerator.TensorMemRefDescriptor.init("scalar", @intCast(@intFromPtr(&scalar_bits[0])), .bf16, .host, 0, &.{input.shape[0]}, &.{0}) catch {
        out.deinit();
        return null;
    };
    const out_descriptor = describeHostBitsViewMemRef(.bf16, out_bits.ptr, out.shape, out.strides, "out") catch {
        out.deinit();
        return null;
    };
    const lhs_descriptor = if (scalar_left) scalar_descriptor else input_descriptor;
    const rhs_descriptor = if (scalar_left) input_descriptor else scalar_descriptor;

    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(input.allocator);
    const result = runtime.runTensorElementwiseBinaryMemRefsNative(u16, .bf16, lhs_bits, rhs_bits, out_bits, lhs_descriptor, rhs_descriptor, out_descriptor, .{
        .op = axiomBinaryOp(op),
        .kernel_symbol = switch (op) {
            .add => "vectra_axiom_bf16_strided_scalar_add",
            .sub => "vectra_axiom_bf16_strided_scalar_sub",
            .mul => "vectra_axiom_bf16_strided_scalar_mul",
            .div => "vectra_axiom_bf16_strided_scalar_div",
        },
    }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => {
            out.deinit();
            return null;
        },
    };
    if (!result.ok()) {
        out.deinit();
        return null;
    }
    for (out_bits, out.data) |bits, *slot| slot.* = .{ .bits = bits };
    return out;
}

fn tryUnaryViewBF16(op: UnaryOp, input: array_mod.ArrayView(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    if (!build_options.enable_axiom_cuda) return null;
    if (!supportedOneDimensionalViewTyped(BFloat16, input)) return null;

    const input_slice = viewBackingSliceTyped(BFloat16, input) orelse return null;
    var out = try array_mod.Array(BFloat16).empty(input.allocator, input.shape);
    errdefer out.deinit();

    const input_bits = try input.allocator.alloc(u16, input_slice.len);
    defer input.allocator.free(input_bits);
    const out_bits = try input.allocator.alloc(u16, out.data.len);
    defer input.allocator.free(out_bits);
    for (input_slice, input_bits) |value, *slot| slot.* = value.bits;

    const input_descriptor = describeHostBitsViewMemRef(.bf16, input_bits.ptr, input.shape, input.strides, "input") catch {
        out.deinit();
        return null;
    };
    const out_descriptor = describeHostBitsViewMemRef(.bf16, out_bits.ptr, out.shape, out.strides, "out") catch {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(input.allocator);
    const result = runtime.runTensorUnaryElementwiseMemRefsBF16(input_bits, out_bits, input_descriptor, out_descriptor, .{
        .op = axiomUnaryOp(op),
    }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => {
            out.deinit();
            return null;
        },
    };
    if (!result.verified) {
        out.deinit();
        return null;
    }
    for (out_bits, out.data) |bits, *slot| slot.* = .{ .bits = bits };
    return out;
}

fn tryBinaryF32(op: BinaryOp, lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    if (!build_options.enable_axiom_cuda) return null;
    if (!supportedSameShapeContiguous(lhs, rhs)) return null;

    var out = try array_mod.Array(f32).empty(lhs.allocator, lhs.shape);
    errdefer out.deinit();

    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(lhs.allocator);
    const result = runtime.runTensorElementwiseBinary(lhs.data, rhs.data, out.data, .{
        .op = axiomBinaryOp(op),
        .len = lhs.data.len,
        .kernel_symbol = switch (op) {
            .add => "vectra_axiom_add",
            .sub => "vectra_axiom_sub",
            .mul => "vectra_axiom_mul",
            .div => "vectra_axiom_div",
        },
        .prefer_cached_device = true,
    }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => {
            out.deinit();
            return null;
        },
    };
    if (!result.verified) {
        out.deinit();
        return null;
    }
    return out;
}

pub fn tryBinaryF64(op: BinaryOp, lhs: array_mod.Array(f64), rhs: array_mod.Array(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    if (!build_options.enable_axiom_cuda) return null;
    if (!supportedSameShapeContiguousF64(lhs, rhs)) return null;
    var out = try array_mod.Array(f64).empty(lhs.allocator, lhs.shape);
    errdefer out.deinit();
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(lhs.allocator);
    const result = runtime.runTensorElementwiseBinaryF64Native(lhs.data, rhs.data, out.data, .{
        .op = axiomBinaryOp(op),
        .len = lhs.data.len,
        .kernel_symbol = switch (op) {
            .add => "vectra_axiom_f64_add",
            .sub => "vectra_axiom_f64_sub",
            .mul => "vectra_axiom_f64_mul",
            .div => "vectra_axiom_f64_div",
        },
    }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => {
            out.deinit();
            return null;
        },
    };
    if (!result.ok()) {
        out.deinit();
        return null;
    }
    return out;
}

fn axiomBinaryOp(op: BinaryOp) axiom.accelerator.TensorBinaryElementwiseOp {
    return switch (op) {
        .add => .add,
        .sub => .sub,
        .mul => .mul,
        .div => .div,
    };
}

fn axiomUnaryOp(op: UnaryOp) axiom.accelerator.TensorUnaryElementwiseOp {
    return switch (op) {
        .sqrt => .sqrt,
        .exp => .exp,
        .abs => .abs,
        .log => .log,
        .sin => .sin,
        .cos => .cos,
        .tan => .tan,
        .exp2 => .exp2,
        .expm1 => .expm1,
        .log1p => .log1p,
        .log2 => .log2,
        .log10 => .log10,
    };
}

fn tryBinaryBF16(op: BinaryOp, lhs: array_mod.Array(BFloat16), rhs: array_mod.Array(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    if (!build_options.enable_axiom_cuda) return null;
    if (try tryDeviceBinaryBF16(op, lhs, rhs)) |device| return device;
    if (!supportedSameShapeContiguousBF16(lhs, rhs)) return null;
    if (try tryBinaryBF16Native(op, lhs, rhs)) |native| return native;
    var lhs32 = try bf16ArrayToF32(lhs);
    defer lhs32.deinit();
    var rhs32 = try bf16ArrayToF32(rhs);
    defer rhs32.deinit();
    var out32 = try tryBinaryF32(op, lhs32, rhs32) orelse return null;
    defer out32.deinit();
    return try f32ArrayToBF16(out32);
}

fn tryBinaryBF16Native(op: BinaryOp, lhs: array_mod.Array(BFloat16), rhs: array_mod.Array(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    if (!build_options.enable_axiom_cuda) return null;
    if (!supportedSameShapeContiguousBF16(lhs, rhs)) return null;
    var out = try array_mod.Array(BFloat16).empty(lhs.allocator, lhs.shape);
    errdefer out.deinit();
    const lhs_bits = try lhs.allocator.alloc(u16, lhs.data.len);
    defer lhs.allocator.free(lhs_bits);
    const rhs_bits = try lhs.allocator.alloc(u16, rhs.data.len);
    defer lhs.allocator.free(rhs_bits);
    const out_bits = try lhs.allocator.alloc(u16, out.data.len);
    defer lhs.allocator.free(out_bits);
    for (lhs.data, lhs_bits) |value, *slot| slot.* = value.bits;
    for (rhs.data, rhs_bits) |value, *slot| slot.* = value.bits;
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(lhs.allocator);
    const result = runtime.runTensorElementwiseBinaryBF16Native(lhs_bits, rhs_bits, out_bits, .{
        .op = axiomBinaryOp(op),
        .len = lhs.data.len,
        .kernel_symbol = switch (op) {
            .add => "vectra_axiom_bf16_native_add",
            .sub => "vectra_axiom_bf16_native_sub",
            .mul => "vectra_axiom_bf16_native_mul",
            .div => "vectra_axiom_bf16_native_div",
        },
    }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => {
            out.deinit();
            return null;
        },
    };
    if (!result.ok()) {
        out.deinit();
        return null;
    }
    for (out_bits, out.data) |bits, *slot| slot.* = .{ .bits = bits };
    return out;
}

fn tryBinaryF16(op: BinaryOp, lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    if (!build_options.enable_axiom_cuda) return null;
    if (try tryDeviceBinaryF16(op, lhs, rhs)) |device| return device;
    if (!supportedSameShapeContiguousF16(lhs, rhs)) return null;
    if (try tryBinaryF16Native(op, lhs, rhs)) |native| return native;
    var lhs32 = try f16ArrayToF32(lhs);
    defer lhs32.deinit();
    var rhs32 = try f16ArrayToF32(rhs);
    defer rhs32.deinit();
    var out32 = try tryBinaryF32(op, lhs32, rhs32) orelse return null;
    defer out32.deinit();
    return try f32ArrayToF16(out32);
}

fn tryBinaryF16Native(op: BinaryOp, lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    if (!build_options.enable_axiom_cuda) return null;
    if (!supportedSameShapeContiguousF16(lhs, rhs)) return null;
    var out = try array_mod.Array(f16).empty(lhs.allocator, lhs.shape);
    errdefer out.deinit();
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(lhs.allocator);
    const result = runtime.runTensorElementwiseBinaryF16Native(lhs.data, rhs.data, out.data, .{
        .op = axiomBinaryOp(op),
        .len = lhs.data.len,
        .kernel_symbol = switch (op) {
            .add => "vectra_axiom_f16_native_add",
            .sub => "vectra_axiom_f16_native_sub",
            .mul => "vectra_axiom_f16_native_mul",
            .div => "vectra_axiom_f16_native_div",
        },
    }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => {
            out.deinit();
            return null;
        },
    };
    if (!result.ok()) {
        out.deinit();
        return null;
    }
    return out;
}

fn supportedSameShapeContiguous(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) bool {
    return supportedNonEmptyContiguous(lhs) and
        supportedNonEmptyContiguous(rhs) and
        lhs.sameShape(rhs);
}

fn supportedSameShapeContiguousF64(lhs: array_mod.Array(f64), rhs: array_mod.Array(f64)) bool {
    return supportedNonEmptyContiguousF64(lhs) and
        supportedNonEmptyContiguousF64(rhs) and
        lhs.sameShape(rhs);
}

fn supportedSameShapeContiguousF16(lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) bool {
    return supportedNonEmptyContiguousF16(lhs) and
        supportedNonEmptyContiguousF16(rhs) and
        lhs.sameShape(rhs);
}

fn supportedSameShapeContiguousBF16(lhs: array_mod.Array(BFloat16), rhs: array_mod.Array(BFloat16)) bool {
    return supportedNonEmptyContiguousBF16(lhs) and
        supportedNonEmptyContiguousBF16(rhs) and
        lhs.sameShape(rhs);
}

fn supportedNonEmptyContiguous(input: array_mod.Array(f32)) bool {
    return input.device.isCpu() and input.data.len != 0 and input.isContiguous();
}

fn supportedNonEmptyContiguousF64(input: array_mod.Array(f64)) bool {
    return input.device.isCpu() and input.data.len != 0 and input.isContiguous();
}

fn supportedNonEmptyContiguousF16(input: array_mod.Array(f16)) bool {
    return input.device.isCpu() and input.data.len != 0 and input.isContiguous();
}

fn supportedNonEmptyContiguousBF16(input: array_mod.Array(BFloat16)) bool {
    return input.device.isCpu() and input.data.len != 0 and input.isContiguous();
}

fn supportedMatmul2dContiguous(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) bool {
    return lhs.device.isCpu() and
        rhs.device.isCpu() and
        lhs.shape.len == 2 and
        rhs.shape.len == 2 and
        lhs.shape[1] == rhs.shape[0] and
        lhs.data.len != 0 and
        rhs.data.len != 0 and
        lhs.isContiguous() and
        rhs.isContiguous();
}

fn supportedMatmul2dContiguousF16(lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) bool {
    return lhs.device.isCpu() and
        rhs.device.isCpu() and
        lhs.shape.len == 2 and
        rhs.shape.len == 2 and
        lhs.shape[1] == rhs.shape[0] and
        lhs.data.len != 0 and
        rhs.data.len != 0 and
        lhs.isContiguous() and
        rhs.isContiguous();
}

fn supportedMatmul2dContiguousBF16(lhs: array_mod.Array(BFloat16), rhs: array_mod.Array(BFloat16)) bool {
    return lhs.device.isCpu() and
        rhs.device.isCpu() and
        lhs.shape.len == 2 and
        rhs.shape.len == 2 and
        lhs.shape[1] == rhs.shape[0] and
        lhs.data.len != 0 and
        rhs.data.len != 0 and
        lhs.isContiguous() and
        rhs.isContiguous();
}

fn buildMatmulTileIr(m: usize, n: usize, k: usize, kernel_symbol: []const u8) axiom.accelerator.CudaTileProgram {
    const tile_m: usize = @min(m, @as(usize, 16));
    const tile_n: usize = @min(n, @as(usize, 16));
    const tile_k: usize = @min(k, @as(usize, 16));
    return axiom.accelerator.cuda_tile_ir.builder(kernel_symbol)
        .tensor(.init("lhs", .f32, .rowMajor2d(m, k), .global))
        .tensor(.init("rhs", .f32, .rowMajor2d(k, n), .global))
        .tensor(.init("out", .f32, .rowMajor2d(m, n), .global))
        .fragment(.init("lhs_tile", .f32, .rowMajor2d(tile_m, tile_k), .shared))
        .fragment(.init("rhs_tile", .f32, .rowMajor2d(tile_k, tile_n), .shared))
        .fragment(.init("acc", .f32, .rowMajor2d(tile_m, tile_n), .accumulator))
        .cta(tile_m, tile_n, tile_k)
        .warp(tile_m, @min(tile_n, @as(usize, 8)), tile_k)
        .mmaTile(@min(tile_m, @as(usize, 16)), @min(tile_n, @as(usize, 8)), @min(tile_k, @as(usize, 8)))
        .load("lhs_tile", "lhs")
        .load("rhs_tile", "rhs")
        .mma("acc", "lhs_tile", "rhs_tile")
        .store("out", "acc")
        .build();
}

fn buildTypedMatmulTileIr(
    m: usize,
    n: usize,
    k: usize,
    element: TypedGemmElement,
    kernel_symbol: []const u8,
) axiom.accelerator.CudaTileProgram {
    const tile_m: usize = @min(m, @as(usize, 16));
    const tile_n: usize = @min(n, @as(usize, 16));
    const tile_k: usize = @min(k, @as(usize, 16));
    return switch (element) {
        .f16 => axiom.accelerator.cuda_tile_ir.builder(kernel_symbol)
            .tensor(.init("lhs", .f16, .rowMajor2d(m, k), .global))
            .tensor(.init("rhs", .f16, .rowMajor2d(k, n), .global))
            .tensor(.init("out", .f16, .rowMajor2d(m, n), .global))
            .fragment(.init("lhs_tile", .f16, .rowMajor2d(tile_m, tile_k), .shared))
            .fragment(.init("rhs_tile", .f16, .rowMajor2d(tile_k, tile_n), .shared))
            .fragment(.init("acc", .f16, .rowMajor2d(tile_m, tile_n), .accumulator))
            .cta(tile_m, tile_n, tile_k)
            .warp(tile_m, @min(tile_n, @as(usize, 8)), tile_k)
            .mmaTile(@min(tile_m, @as(usize, 16)), @min(tile_n, @as(usize, 8)), @min(tile_k, @as(usize, 8)))
            .load("lhs_tile", "lhs")
            .load("rhs_tile", "rhs")
            .mma("acc", "lhs_tile", "rhs_tile")
            .store("out", "acc")
            .build(),
        .bf16 => axiom.accelerator.cuda_tile_ir.builder(kernel_symbol)
            .tensor(.init("lhs", .bf16, .rowMajor2d(m, k), .global))
            .tensor(.init("rhs", .bf16, .rowMajor2d(k, n), .global))
            .tensor(.init("out", .bf16, .rowMajor2d(m, n), .global))
            .fragment(.init("lhs_tile", .bf16, .rowMajor2d(tile_m, tile_k), .shared))
            .fragment(.init("rhs_tile", .bf16, .rowMajor2d(tile_k, tile_n), .shared))
            .fragment(.init("acc", .bf16, .rowMajor2d(tile_m, tile_n), .accumulator))
            .cta(tile_m, tile_n, tile_k)
            .warp(tile_m, @min(tile_n, @as(usize, 8)), tile_k)
            .mmaTile(@min(tile_m, @as(usize, 16)), @min(tile_n, @as(usize, 8)), @min(tile_k, @as(usize, 8)))
            .load("lhs_tile", "lhs")
            .load("rhs_tile", "rhs")
            .mma("acc", "lhs_tile", "rhs_tile")
            .store("out", "acc")
            .build(),
    };
}

fn typedGemmPlanEvidenceFromAxiom(plan: axiom.accelerator.TensorTypedGemmLaunchPlan) TypedGemmPlanEvidence {
    return .{
        .ok = plan.ok(),
        .element_name = @tagName(plan.element_type),
        .readiness_status = plan.readiness_status.label(),
        .m = plan.m,
        .n = plan.n,
        .k = plan.k,
        .tile_m = plan.tile_m,
        .tile_n = plan.tile_n,
        .tile_k = plan.tile_k,
        .grid_m = plan.grid_m,
        .grid_n = plan.grid_n,
        .total_ctas = plan.total_ctas,
        .threads_per_cta = plan.threads_per_cta,
        .argument_bytes = plan.argument_bytes,
        .plan_fingerprint = plan.fingerprint(),
        .seed_fingerprint = plan.seed_fingerprint,
        .readiness_fingerprint = plan.readiness_fingerprint,
    };
}

fn supportedOneDimensionalView(view: array_mod.ArrayView(f32)) bool {
    return supportedOneDimensionalViewTyped(f32, view);
}

fn viewBackingSlice(view: array_mod.ArrayView(f32)) ?[]const f32 {
    return viewBackingSliceTyped(f32, view);
}

fn supportedOneDimensionalViewTyped(comptime T: type, view: array_mod.ArrayView(T)) bool {
    return view.device.isCpu() and view.shape.len == 1 and view.shape[0] != 0 and view.strides.len == 1 and view.strides[0] != 0;
}

fn viewBackingSliceTyped(comptime T: type, view: array_mod.ArrayView(T)) ?[]const T {
    if (!supportedOneDimensionalViewTyped(T, view)) return null;
    const last_delta = std.math.mul(usize, view.shape[0] - 1, view.strides[0]) catch return null;
    const end_index = std.math.add(usize, view.offset, last_delta) catch return null;
    if (end_index >= view.data.len) return null;
    return view.data[view.offset .. end_index + 1];
}

fn describeHostArrayMemRef(comptime T: type, input: array_mod.Array(T), name: []const u8) array_mod.ArrayError!axiom.accelerator.TensorMemRefDescriptor {
    const element = axiomTensorElementType(T) orelse return error.TypeUnsupported;
    const strides = try usizeStridesToIsize(input.strides);
    return axiom.accelerator.TensorMemRefDescriptor.init(
        name,
        @intCast(@intFromPtr(input.data.ptr)),
        element,
        .host,
        0,
        input.shape,
        strides[0..input.strides.len],
    ) catch error.InvalidShape;
}

fn describeHostViewMemRef(comptime T: type, input: array_mod.ArrayView(T), name: []const u8) array_mod.ArrayError!axiom.accelerator.TensorMemRefDescriptor {
    const element = axiomTensorElementType(T) orelse return error.TypeUnsupported;
    const strides = try usizeStridesToIsize(input.strides);
    return axiom.accelerator.TensorMemRefDescriptor.init(
        name,
        @intCast(@intFromPtr(input.data.ptr)),
        element,
        .host,
        input.offset,
        input.shape,
        strides[0..input.strides.len],
    ) catch error.InvalidShape;
}

fn describeHostBitsViewMemRef(
    element: axiom.accelerator.TensorElementType,
    base_ptr: [*]const u16,
    shape: []const usize,
    stride_values: []const usize,
    name: []const u8,
) array_mod.ArrayError!axiom.accelerator.TensorMemRefDescriptor {
    const strides = try usizeStridesToIsize(stride_values);
    return axiom.accelerator.TensorMemRefDescriptor.init(
        name,
        @intCast(@intFromPtr(base_ptr)),
        element,
        .host,
        0,
        shape,
        strides[0..stride_values.len],
    ) catch error.InvalidShape;
}

fn describeDeviceArrayMemRef(comptime T: type, input: array_mod.Array(T), storage: array_mod.DeviceStorage, name: []const u8) array_mod.ArrayError!axiom.accelerator.TensorMemRefDescriptor {
    return describeDeviceBufferMemRef(T, storage, input.shape, input.strides, name);
}

fn describeDeviceBufferMemRef(
    comptime T: type,
    storage: array_mod.DeviceStorage,
    shape: []const usize,
    stride_values: []const usize,
    name: []const u8,
) array_mod.ArrayError!axiom.accelerator.TensorMemRefDescriptor {
    const element = axiomTensorElementType(T) orelse return error.TypeUnsupported;
    const strides = try usizeStridesToIsize(stride_values);
    return axiom.accelerator.TensorMemRefDescriptor.init(
        name,
        storage.ptr,
        element,
        .cuda,
        0,
        shape,
        strides[0..stride_values.len],
    ) catch error.InvalidShape;
}

fn describeDeviceGemmMemRefSpec(
    comptime T: type,
    m: usize,
    n: usize,
    k: usize,
    lhs_ptr: u64,
    rhs_ptr: u64,
    out_ptr: u64,
    lhs_name: []const u8,
    rhs_name: []const u8,
    out_name: []const u8,
) array_mod.ArrayError!axiom.accelerator.TensorGemmSpec {
    const element = axiomTensorElementType(T) orelse return error.TypeUnsupported;
    const lhs_descriptor = axiom.accelerator.TensorMemRefDescriptor.init(
        lhs_name,
        lhs_ptr,
        element,
        .cuda,
        0,
        &.{ m, k },
        &.{ @as(isize, @intCast(k)), 1 },
    ) catch return error.InvalidShape;
    const rhs_descriptor = axiom.accelerator.TensorMemRefDescriptor.init(
        rhs_name,
        rhs_ptr,
        element,
        .cuda,
        0,
        &.{ k, n },
        &.{ @as(isize, @intCast(n)), 1 },
    ) catch return error.InvalidShape;
    const out_descriptor = axiom.accelerator.TensorMemRefDescriptor.init(
        out_name,
        out_ptr,
        element,
        .cuda,
        0,
        &.{ m, n },
        &.{ @as(isize, @intCast(n)), 1 },
    ) catch return error.InvalidShape;
    return axiom.accelerator.TensorGemmSpec.fromMemRefs(lhs_descriptor, rhs_descriptor, out_descriptor) catch error.InvalidShape;
}

fn describeDeviceGemmAddMemRefSpec(
    comptime T: type,
    m: usize,
    n: usize,
    k: usize,
    lhs_ptr: u64,
    rhs_ptr: u64,
    add_ptr: u64,
    out_ptr: u64,
    lhs_name: []const u8,
    rhs_name: []const u8,
    add_name: []const u8,
    out_name: []const u8,
) array_mod.ArrayError!axiom.accelerator.TensorGemmAddSpec {
    const element = axiomTensorElementType(T) orelse return error.TypeUnsupported;
    const lhs_descriptor = axiom.accelerator.TensorMemRefDescriptor.init(
        lhs_name,
        lhs_ptr,
        element,
        .cuda,
        0,
        &.{ m, k },
        &.{ @as(isize, @intCast(k)), 1 },
    ) catch return error.InvalidShape;
    const rhs_descriptor = axiom.accelerator.TensorMemRefDescriptor.init(
        rhs_name,
        rhs_ptr,
        element,
        .cuda,
        0,
        &.{ k, n },
        &.{ @as(isize, @intCast(n)), 1 },
    ) catch return error.InvalidShape;
    const add_descriptor = axiom.accelerator.TensorMemRefDescriptor.init(
        add_name,
        add_ptr,
        element,
        .cuda,
        0,
        &.{ m, n },
        &.{ @as(isize, @intCast(n)), 1 },
    ) catch return error.InvalidShape;
    const out_descriptor = axiom.accelerator.TensorMemRefDescriptor.init(
        out_name,
        out_ptr,
        element,
        .cuda,
        0,
        &.{ m, n },
        &.{ @as(isize, @intCast(n)), 1 },
    ) catch return error.InvalidShape;
    return axiom.accelerator.TensorGemmAddSpec.fromMemRefs(lhs_descriptor, rhs_descriptor, add_descriptor, out_descriptor) catch error.InvalidShape;
}

fn reductionOpFromDialect(op: axiom.accelerator.DialectReductionOp) axiom.accelerator.TensorReduction2DOp {
    return switch (op) {
        .sum => .sum,
        .max => .max,
        .min => .min,
        .prod => .prod,
    };
}

fn reductionAxisFromU1(axis: u1) axiom.accelerator.TensorReduction2DAxis {
    return switch (axis) {
        0 => .axis0,
        1 => .axis1,
    };
}

fn broadcastAxisFromDialect(axis: axiom.accelerator.DialectBroadcastAxis) axiom.accelerator.TensorBroadcastAdd2DAxis {
    return switch (axis) {
        .row => .row,
        .column => .column,
    };
}

fn axiomTensorElementType(comptime T: type) ?axiom.accelerator.TensorElementType {
    return if (T == f32)
        .f32
    else if (T == f64)
        .f64
    else if (T == f16)
        .f16
    else if (T == BFloat16)
        .bf16
    else
        null;
}

fn usizeStridesToIsize(strides: []const usize) array_mod.ArrayError![4]isize {
    if (strides.len > 4) return error.InvalidShape;
    var out: [4]isize = .{ 1, 1, 1, 1 };
    for (strides, 0..) |stride, index| {
        out[index] = std.math.cast(isize, stride) orelse return error.InvalidShape;
    }
    return out;
}

fn failedReport() SmokeReport {
    var report = baseSmokeReport();
    report.status = if (build_options.enable_axiom_cuda) .failed else .disabled;
    report.issue_count = 1;
    return report;
}

fn f16ArrayToF32(input: array_mod.Array(f16)) array_mod.ArrayError!array_mod.Array(f32) {
    var out = try array_mod.Array(f32).empty(input.allocator, input.shape);
    errdefer out.deinit();
    _ = axiom.accelerator.tensor_adapter.widenF16ToF32(input.data, out.data) catch |err| return mapTensorAdapterError(err);
    return out;
}

fn bf16ArrayToF32(input: array_mod.Array(BFloat16)) array_mod.ArrayError!array_mod.Array(f32) {
    var out = try array_mod.Array(f32).empty(input.allocator, input.shape);
    errdefer out.deinit();
    const bits = try input.allocator.alloc(u16, input.data.len);
    defer input.allocator.free(bits);
    for (input.data, bits) |value, *slot| slot.* = value.bits;
    _ = axiom.accelerator.tensor_adapter.widenBF16ToF32(bits, out.data) catch |err| return mapTensorAdapterError(err);
    return out;
}

fn f32ArrayToF16(input: array_mod.Array(f32)) array_mod.ArrayError!array_mod.Array(f16) {
    var out = try array_mod.Array(f16).empty(input.allocator, input.shape);
    errdefer out.deinit();
    _ = axiom.accelerator.tensor_adapter.narrowF32ToF16(input.data, out.data) catch |err| return mapTensorAdapterError(err);
    return out;
}

fn f32ArrayToBF16(input: array_mod.Array(f32)) array_mod.ArrayError!array_mod.Array(BFloat16) {
    var out = try array_mod.Array(BFloat16).empty(input.allocator, input.shape);
    errdefer out.deinit();
    const bits = try input.allocator.alloc(u16, input.data.len);
    defer input.allocator.free(bits);
    _ = axiom.accelerator.tensor_adapter.narrowF32ToBF16(input.data, bits) catch |err| return mapTensorAdapterError(err);
    for (bits, out.data) |value, *slot| slot.* = .{ .bits = value };
    return out;
}

fn widenedF16BinaryProvenanceFingerprint(allocator: std.mem.Allocator, operation: []const u8, op: BinaryOp, lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!u64 {
    _ = operation;
    if (!supportedSameShapeContiguousF16(lhs, rhs)) return error.ShapeMismatch;
    const out = try allocator.alloc(f16, lhs.data.len);
    defer allocator.free(out);
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(allocator);
    const result = runtime.runTensorElementwiseBinaryF16Widened(lhs.data, rhs.data, out, .{
        .op = axiomBinaryOp(op),
        .len = lhs.data.len,
    }) catch |err| return mapTensorAdapterError(err);
    if (!result.ok()) return error.BackendFailure;
    return result.fingerprint();
}

fn nativeF16BinaryExecutionFingerprint(allocator: std.mem.Allocator, op: BinaryOp, lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!u64 {
    if (!supportedSameShapeContiguousF16(lhs, rhs)) return error.ShapeMismatch;
    const out = try allocator.alloc(f16, lhs.data.len);
    defer allocator.free(out);
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(allocator);
    const result = runtime.runTensorElementwiseBinaryF16Native(lhs.data, rhs.data, out, .{
        .op = axiomBinaryOp(op),
        .len = lhs.data.len,
        .kernel_symbol = "vectra_axiom_f16_native_probe",
    }) catch |err| return mapTensorAdapterError(err);
    if (!result.ok()) return error.BackendFailure;
    return result.fingerprint();
}

const TypedGemmRuntimeEvidence = struct {
    fingerprint: u64 = 0,
    route_fingerprint: u64 = 0,
    route: []const u8 = "",

    fn ok(evidence: TypedGemmRuntimeEvidence) bool {
        return evidence.fingerprint != 0 and evidence.route_fingerprint != 0 and std.mem.eql(u8, evidence.route, "widened_f32_cuda_compute");
    }
};

fn typedF16MatmulRuntimeEvidence(allocator: std.mem.Allocator, lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!TypedGemmRuntimeEvidence {
    if (!supportedMatmul2dContiguousF16(lhs, rhs)) return error.ShapeMismatch;
    const m = lhs.shape[0];
    const k = lhs.shape[1];
    const n = rhs.shape[1];
    const c = try allocator.alloc(f16, m * n);
    defer allocator.free(c);
    const out = try allocator.alloc(f16, m * n);
    defer allocator.free(out);
    @memset(c, @as(f16, 0.0));
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(allocator);
    const result = runtime.runTensorGemmF16TypedSimtSeed(lhs.data, rhs.data, c, out, .{
        .m = m,
        .n = n,
        .k = k,
        .tile_x = @intCast(@min(n, @as(usize, 16))),
        .tile_y = @intCast(@min(m, @as(usize, 16))),
        .kernel_symbol = "vectra_axiom_typed_f16_gemm_probe",
    }) catch |err| return mapTensorAdapterError(err);
    if (!result.ok()) return error.BackendFailure;
    return .{
        .fingerprint = result.fingerprint(),
        .route_fingerprint = result.runtime_route_fingerprint,
        .route = result.compute_route.label(),
    };
}

fn widenedBF16BinaryProvenanceFingerprint(allocator: std.mem.Allocator, operation: []const u8, op: BinaryOp, lhs: array_mod.Array(BFloat16), rhs: array_mod.Array(BFloat16)) array_mod.ArrayError!u64 {
    _ = operation;
    if (!supportedSameShapeContiguousBF16(lhs, rhs)) return error.ShapeMismatch;
    const lhs_bits = try allocator.alloc(u16, lhs.data.len);
    defer allocator.free(lhs_bits);
    for (lhs.data, lhs_bits) |value, *slot| slot.* = value.bits;
    const rhs_bits = try allocator.alloc(u16, rhs.data.len);
    defer allocator.free(rhs_bits);
    for (rhs.data, rhs_bits) |value, *slot| slot.* = value.bits;
    const out_bits = try allocator.alloc(u16, lhs.data.len);
    defer allocator.free(out_bits);
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(allocator);
    const result = runtime.runTensorElementwiseBinaryBF16Widened(lhs_bits, rhs_bits, out_bits, .{
        .op = axiomBinaryOp(op),
        .len = lhs.data.len,
    }) catch |err| return mapTensorAdapterError(err);
    if (!result.ok()) return error.BackendFailure;
    return result.fingerprint();
}

fn nativeBF16BinaryExecutionFingerprint(allocator: std.mem.Allocator, op: BinaryOp, lhs: array_mod.Array(BFloat16), rhs: array_mod.Array(BFloat16)) array_mod.ArrayError!u64 {
    if (!supportedSameShapeContiguousBF16(lhs, rhs)) return error.ShapeMismatch;
    const lhs_bits = try allocator.alloc(u16, lhs.data.len);
    defer allocator.free(lhs_bits);
    const rhs_bits = try allocator.alloc(u16, rhs.data.len);
    defer allocator.free(rhs_bits);
    const out_bits = try allocator.alloc(u16, lhs.data.len);
    defer allocator.free(out_bits);
    for (lhs.data, lhs_bits) |value, *slot| slot.* = value.bits;
    for (rhs.data, rhs_bits) |value, *slot| slot.* = value.bits;
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(allocator);
    const result = runtime.runTensorElementwiseBinaryBF16Native(lhs_bits, rhs_bits, out_bits, .{
        .op = axiomBinaryOp(op),
        .len = lhs.data.len,
        .kernel_symbol = "vectra_axiom_bf16_native_probe",
    }) catch |err| return mapTensorAdapterError(err);
    if (!result.ok()) return error.BackendFailure;
    return result.fingerprint();
}

fn typedBF16MatmulRuntimeEvidence(allocator: std.mem.Allocator, lhs: array_mod.Array(BFloat16), rhs: array_mod.Array(BFloat16)) array_mod.ArrayError!TypedGemmRuntimeEvidence {
    if (!supportedMatmul2dContiguousBF16(lhs, rhs)) return error.ShapeMismatch;
    const m = lhs.shape[0];
    const k = lhs.shape[1];
    const n = rhs.shape[1];
    const lhs_bits = try allocator.alloc(u16, lhs.data.len);
    defer allocator.free(lhs_bits);
    const rhs_bits = try allocator.alloc(u16, rhs.data.len);
    defer allocator.free(rhs_bits);
    const c_bits = try allocator.alloc(u16, m * n);
    defer allocator.free(c_bits);
    const out_bits = try allocator.alloc(u16, m * n);
    defer allocator.free(out_bits);
    for (lhs.data, lhs_bits) |value, *slot| slot.* = value.bits;
    for (rhs.data, rhs_bits) |value, *slot| slot.* = value.bits;
    @memset(c_bits, @as(u16, 0));
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(allocator);
    const result = runtime.runTensorGemmBF16TypedSimtSeed(lhs_bits, rhs_bits, c_bits, out_bits, .{
        .m = m,
        .n = n,
        .k = k,
        .tile_x = @intCast(@min(n, @as(usize, 16))),
        .tile_y = @intCast(@min(m, @as(usize, 16))),
        .kernel_symbol = "vectra_axiom_typed_bf16_gemm_probe",
    }) catch |err| return mapTensorAdapterError(err);
    if (!result.ok()) return error.BackendFailure;
    return .{
        .fingerprint = result.fingerprint(),
        .route_fingerprint = result.runtime_route_fingerprint,
        .route = result.compute_route.label(),
    };
}

fn mapTensorAdapterError(err: anyerror) array_mod.ArrayError {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.TensorShapeMismatch, error.InvalidTensorView => error.ShapeMismatch,
        else => error.BackendFailure,
    };
}

fn sliceClose(actual: []const f32, expected: []const f32, tolerance: f32) bool {
    if (actual.len != expected.len) return false;
    return maxAbsError(actual, expected) <= tolerance;
}

fn sliceCloseF64(actual: []const f64, expected: []const f64, tolerance: f64) bool {
    if (actual.len != expected.len) return false;
    return maxAbsErrorF64(actual, expected) <= tolerance;
}

fn f16Close(actual: []const f16, expected: []const f32, tolerance: f32) bool {
    if (actual.len != expected.len) return false;
    for (actual, expected) |a, e| {
        if (@abs(@as(f32, @floatCast(a)) - e) > tolerance) return false;
    }
    return true;
}

fn bf16Close(actual: []const BFloat16, expected: []const f32, tolerance: f32) bool {
    if (actual.len != expected.len) return false;
    for (actual, expected) |a, e| {
        if (@abs(a.toF32() - e) > tolerance) return false;
    }
    return true;
}

fn maxAbsError(actual: []const f32, expected: []const f32) f32 {
    if (actual.len != expected.len) return std.math.inf(f32);
    var max_error: f32 = 0.0;
    for (actual, expected) |a, e| {
        const err = @abs(a - e);
        if (err > max_error) max_error = err;
    }
    return max_error;
}

fn maxAbsErrorF64(actual: []const f64, expected: []const f64) f64 {
    if (actual.len != expected.len) return std.math.inf(f64);
    var max_error: f64 = 0.0;
    for (actual, expected) |a, e| {
        const err = @abs(a - e);
        if (err > max_error) max_error = err;
    }
    return max_error;
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

fn hashI32(hasher: *std.hash.Wyhash, value: i32) void {
    var bytes: [4]u8 = undefined;
    std.mem.writeInt(u32, &bytes, @bitCast(value), .little);
    hasher.update(&bytes);
}

fn hashF32(hasher: *std.hash.Wyhash, value: f32) void {
    var bytes: [4]u8 = undefined;
    std.mem.writeInt(u32, &bytes, @bitCast(value), .little);
    hasher.update(&bytes);
}

fn hashF64(hasher: *std.hash.Wyhash, value: f64) void {
    var bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &bytes, @bitCast(value), .little);
    hasher.update(&bytes);
}

fn hashF16(hasher: *std.hash.Wyhash, value: f16) void {
    var bytes: [2]u8 = undefined;
    std.mem.writeInt(u16, &bytes, @bitCast(value), .little);
    hasher.update(&bytes);
}

fn hashBF16(hasher: *std.hash.Wyhash, value: BFloat16) void {
    var bytes: [2]u8 = undefined;
    std.mem.writeInt(u16, &bytes, value.bits, .little);
    hasher.update(&bytes);
}

fn hashF32Slice(values: []const f32) u64 {
    var hasher = std.hash.Wyhash.init(0x0abc_7aaa_f325_511c);
    hashU64(&hasher, values.len);
    for (values) |value| hashF32(&hasher, value);
    return hasher.final();
}

fn hashF64Slice(values: []const f64) u64 {
    var hasher = std.hash.Wyhash.init(0x0abc_7aaa_f625_511c);
    hashU64(&hasher, values.len);
    for (values) |value| hashF64(&hasher, value);
    return hasher.final();
}

fn hashF16Slice(values: []const f16) u64 {
    var hasher = std.hash.Wyhash.init(0x0abc_7aaa_f016_511c);
    hashU64(&hasher, values.len);
    for (values) |value| hashF16(&hasher, value);
    return hasher.final();
}

fn hashBF16Slice(values: []const BFloat16) u64 {
    var hasher = std.hash.Wyhash.init(0x0abc_7aaa_bf16_511c);
    hashU64(&hasher, values.len);
    for (values) |value| hashBF16(&hasher, value);
    return hasher.final();
}

test "Axiom CUDA bridge reports dtype metadata deterministically" {
    const report = runSmoke(std.testing.allocator);
    try std.testing.expectEqual(cuda_dtype_support.len, report.dtype_support_count);
    try std.testing.expectEqual(@as(usize, 4), report.dtype_bridge_count);
    try std.testing.expectEqual(@as(usize, 2), report.dtype_native_seed_count);
    try std.testing.expectEqual(@as(usize, 2), report.dtype_widened_seed_count);
    try std.testing.expect(report.dtype_support_fingerprint != 0);
    const f16_record = findCudaDTypeSupport("CUDA_R_16F").?;
    try std.testing.expectEqual(CudaDTypeBridgeStatus.widened_f32_seed, f16_record.status);
    try std.testing.expectEqual(array_mod.DType.f16, f16_record.vectra_dtype.?);
    const bf16_record = findVectraDTypeSupport(.bf16).?;
    try std.testing.expectEqualStrings("CUDA_R_16BF", bf16_record.cuda_name);
    const f32_record = findVectraDTypeSupport(.f32).?;
    try std.testing.expectEqual(CudaDTypeBridgeStatus.native_cuda_seed, f32_record.status);
    const f64_record = findVectraDTypeSupport(.f64).?;
    try std.testing.expectEqual(CudaDTypeBridgeStatus.native_cuda_seed, f64_record.status);
    try std.testing.expect(f64_record.matmul);
    try std.testing.expect(findCudaDTypeSupport("CUDA_R_8F_E4M3") != null);
    if (build_options.enable_axiom_cuda) {
        try std.testing.expect(report.status == .ran or report.status == .skipped or report.status == .failed);
    } else {
        try std.testing.expect(!enabled());
        try std.testing.expect(report.ok());
        try std.testing.expectEqual(Status.disabled, report.status);
        try std.testing.expectEqual(@as(u8, 0), report.issue_count);
    }
}

test "Axiom CUDA bridge snapshots last GEMM plan-cache evidence" {
    resetLastCudaDeviceGemmReport();
    try std.testing.expect(!lastCudaDeviceGemmReport().valid());

    const StubReport = struct {
        ok: bool = true,
        backend: []const u8 = "cublaslt_f32",
        device_ordinal: usize = 7,
        m: usize = 8,
        n: usize = 4,
        k: usize = 6,
        lhs_device_ptr: u64 = 0x1000,
        rhs_device_ptr: u64 = 0x2000,
        out_device_ptr: u64 = 0x3000,
        alpha: f32 = 1.0,
        beta: f32 = 1.0,
        cache_hit: bool = true,
        lt_plan_cache_hit: bool = true,
        lt_algo_cache_hit: bool = true,

        fn fingerprint(report: @This()) u64 {
            return if (report.lt_plan_cache_hit and report.lt_algo_cache_hit) 0xa11c_acc1_c061_a501 else 0;
        }
    };
    recordCudaDeviceGemmReport(StubReport{});

    const snapshot = lastCudaDeviceGemmReport();
    try std.testing.expect(snapshot.valid());
    try std.testing.expectEqualStrings("cublaslt_f32", snapshot.backend);
    try std.testing.expectEqual(@as(usize, 7), snapshot.device_ordinal);
    try std.testing.expectEqual(@as(usize, 8), snapshot.m);
    try std.testing.expectEqual(@as(usize, 4), snapshot.n);
    try std.testing.expectEqual(@as(usize, 6), snapshot.k);
    try std.testing.expect(snapshot.cache_hit);
    try std.testing.expect(snapshot.lt_plan_cache_hit);
    try std.testing.expect(snapshot.lt_algo_cache_hit);
    try std.testing.expectEqual(@as(u64, 0xa11c_acc1_c061_a501), snapshot.fingerprint);

    resetLastCudaDeviceGemmReport();
    try std.testing.expect(!lastCudaDeviceGemmReport().valid());
}

test "Axiom CUDA bridge snapshots batched GEMM runtime evidence" {
    resetLastCudaDeviceBatchedGemmReport();
    try std.testing.expect(!lastCudaDeviceBatchedGemmReport().valid());

    const StubReport = struct {
        ok: bool = true,
        backend: []const u8 = "loop_over_gemm_memrefs",
        device_ordinal: usize = 2,
        batch_count: usize = 3,
        m: usize = 4,
        n: usize = 5,
        k: usize = 6,
        plan_fingerprint: u64 = 0x1001,
        first_batch_fingerprint: u64 = 0x2002,
        last_batch_fingerprint: u64 = 0x3003,
        combined_batch_fingerprint: u64 = 0x4004,

        fn fingerprint(report: @This()) u64 {
            return report.plan_fingerprint ^ report.combined_batch_fingerprint;
        }
    };
    recordCudaDeviceBatchedGemmReport(StubReport{});

    const snapshot = lastCudaDeviceBatchedGemmReport();
    try std.testing.expect(snapshot.valid());
    try std.testing.expectEqualStrings("loop_over_gemm_memrefs", snapshot.backend);
    try std.testing.expectEqual(@as(usize, 2), snapshot.device_ordinal);
    try std.testing.expectEqual(@as(usize, 3), snapshot.batch_count);
    try std.testing.expectEqual(@as(usize, 4), snapshot.m);
    try std.testing.expectEqual(@as(usize, 5), snapshot.n);
    try std.testing.expectEqual(@as(usize, 6), snapshot.k);
    try std.testing.expectEqual(@as(u64, 0x1001), snapshot.plan_fingerprint);
    try std.testing.expectEqual(@as(u64, 0x2002), snapshot.first_batch_fingerprint);
    try std.testing.expectEqual(@as(u64, 0x3003), snapshot.last_batch_fingerprint);
    try std.testing.expectEqual(@as(u64, 0x4004), snapshot.combined_batch_fingerprint);
    try std.testing.expectEqual(@as(u64, 0x5005), snapshot.fingerprint);

    resetLastCudaDeviceBatchedGemmReport();
    try std.testing.expect(!lastCudaDeviceBatchedGemmReport().valid());
}
