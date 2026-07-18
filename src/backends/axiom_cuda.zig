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
        .fingerprint = report.fingerprint(),
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
    strided_mul_ok: bool = false,
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
            .ran => report.enabled and report.add_ok and report.sub_ok and report.mul_ok and report.div_ok and report.saxpy_ok and report.matmul_ok and report.matmul_tile_ir_ok and report.f16_add_ok and report.f16_matmul_ok and report.bf16_add_ok and report.bf16_matmul_ok and report.typed_f16_gemm_plan.ok and report.typed_bf16_gemm_plan.ok and report.f16_widened_execution_fingerprint != 0 and report.bf16_widened_execution_fingerprint != 0 and report.typed_f16_gemm_route_fingerprint != 0 and report.typed_bf16_gemm_route_fingerprint != 0 and std.mem.eql(u8, report.typed_f16_gemm_route, "widened_f32_cuda_compute") and std.mem.eql(u8, report.typed_bf16_gemm_route, "widened_f32_cuda_compute") and report.scalar_add_ok and report.scalar_mul_ok and report.scalar_saxpy_ok,
            .failed => false,
        };
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
        hashBool(&hasher, report.strided_mul_ok);
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
            "vectra_axiom_cuda_smoke enabled={} status={s} ok={} issues={d} add={} sub={} mul={} div={} saxpy={} matmul={} matmul_tile_ir={} f16_add={} f16_matmul={} bf16_add={} bf16_matmul={} typed_f16_gemm={} typed_bf16_gemm={} scalar_add={} scalar_mul={} scalar_saxpy={} strided_add={} strided_mul={} device_array={} max_abs_error={d} logical_elements={d} required_bytes={d} linear_copy={} copy_plan_ok={} copy_requires_strided={} output={x} fingerprint={x}\n",
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
                report.strided_add_ok,
                report.strided_mul_ok,
                report.device_array_ok,
                report.max_abs_error,
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
                "  \"strided_mul_ok\": {},\n" ++
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
                report.scalar_add_ok,
                report.scalar_mul_ok,
                report.scalar_saxpy_ok,
                report.strided_add_ok,
                report.strided_mul_ok,
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

pub fn tryDeviceBinaryF32(op: BinaryOp, lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    if (!build_options.enable_axiom_cuda) return null;
    if (!lhs.device.isCuda() or !rhs.device.isCuda() or !lhs.device.sameDevice(rhs.device)) return null;
    if (!lhs.sameShape(rhs) or lhs.data.len != 0 or rhs.data.len != 0 or !lhs.isContiguous() or !rhs.isContiguous()) return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;
    if (lhs_storage.len == 0 or lhs_storage.len != rhs_storage.len) return null;

    var out = try array_mod.Array(f32).emptyOn(lhs.allocator, lhs.shape, lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(lhs.allocator);
    const cached_report = runtime.runCudaDeviceElementwiseF32(
        lhs.device.index,
        axiomBinaryOp(op),
        lhs_storage.len,
        lhs_storage.ptr,
        rhs_storage.ptr,
        out_storage.ptr,
    ) catch null;
    if (cached_report) |report| {
        if (report.valid()) return out;
    }

    var session = withCudaContext(lhs.device.index) catch return error.InvalidDevice;
    defer session.driver.close();
    defer session.context.release(&session.driver);
    var cuda_arch_buffer: [16]u8 = undefined;
    const resolved_arch = session.driver.resolveCudaArch(session.context.device, "auto", &cuda_arch_buffer) catch return error.BackendFailure;

    var spec = switch (op) {
        .add => axiom.accelerator.TensorElementwiseBinarySpec.add(
            .contiguous("lhs", lhs_storage.ptr, lhs_storage.len),
            .contiguous("rhs", rhs_storage.ptr, rhs_storage.len),
            .contiguous("out", out_storage.ptr, out_storage.len),
        ),
        .sub => axiom.accelerator.TensorElementwiseBinarySpec.sub(
            .contiguous("lhs", lhs_storage.ptr, lhs_storage.len),
            .contiguous("rhs", rhs_storage.ptr, rhs_storage.len),
            .contiguous("out", out_storage.ptr, out_storage.len),
        ),
        .mul => axiom.accelerator.TensorElementwiseBinarySpec.mul(
            .contiguous("lhs", lhs_storage.ptr, lhs_storage.len),
            .contiguous("rhs", rhs_storage.ptr, rhs_storage.len),
            .contiguous("out", out_storage.ptr, out_storage.len),
        ),
        .div => axiom.accelerator.TensorElementwiseBinarySpec.div(
            .contiguous("lhs", lhs_storage.ptr, lhs_storage.len),
            .contiguous("rhs", rhs_storage.ptr, rhs_storage.len),
            .contiguous("out", out_storage.ptr, out_storage.len),
        ),
    };
    spec.blocks = std.math.cast(u32, (lhs_storage.len + 127) / 128) orelse return error.InvalidShape;
    if (spec.blocks == 0) spec.blocks = 1;
    spec.threads = 128;
    spec.target_arch = resolved_arch;
    spec.kernel_symbol = switch (op) {
        .add => "vectra_device_add",
        .sub => "vectra_device_sub",
        .mul => "vectra_device_mul",
        .div => "vectra_device_div",
    };

    var runtime_threaded_io = std.Io.Threaded.init(lhs.allocator, .{});
    defer runtime_threaded_io.deinit();
    const runtime_io = runtime_threaded_io.io();
    const launch_plan = axiom.accelerator.tensor_adapter.buildElementwiseBinaryLaunchPlan(runtime_io, lhs.allocator, spec, true) catch return null;
    if (launch_plan.runtime_image.image_kind == .none) return null;
    const image = readRuntimeImage(lhs.allocator, launch_plan.runtime_image.rootSlice(), launch_plan.runtime_image.fileNameSlice()) catch return null;
    defer lhs.allocator.free(image);
    var fallback_image: ?[:0]u8 = null;
    defer if (fallback_image) |bytes| lhs.allocator.free(bytes);
    const module = session.driver.moduleLoadData(image) catch module_fallback: {
        if (launch_plan.runtime_image.image_kind != .cubin) return null;
        var ptx_name_buffer: [256]u8 = undefined;
        const ptx_name = ptxFallbackNameForImage(launch_plan.runtime_image.fileNameSlice(), &ptx_name_buffer) catch return null;
        fallback_image = readRuntimeImage(lhs.allocator, launch_plan.runtime_image.rootSlice(), ptx_name) catch return null;
        break :module_fallback session.driver.moduleLoadData(fallback_image.?) catch return null;
    };
    defer session.driver.moduleUnload(module);
    var symbol_buffer: [128]u8 = undefined;
    const symbol = std.fmt.bufPrintSentinel(&symbol_buffer, "{s}", .{spec.kernel_symbol}, 0) catch return null;
    const function = session.driver.moduleGetFunction(module, symbol.ptr) catch return null;
    var lhs_ptr = lhs_storage.ptr;
    var rhs_ptr = rhs_storage.ptr;
    var out_ptr = out_storage.ptr;
    var n_arg: i32 = std.math.cast(i32, lhs_storage.len) orelse return error.InvalidShape;
    var args = [_]?*anyopaque{
        @ptrCast(&lhs_ptr),
        @ptrCast(&rhs_ptr),
        @ptrCast(&out_ptr),
        @ptrCast(&n_arg),
    };
    session.driver.launchKernel(
        function,
        launch_plan.driver_launch.grid,
        launch_plan.driver_launch.block,
        launch_plan.driver_launch.shared_memory_bytes,
        &args,
    ) catch return null;
    session.driver.synchronize() catch return null;
    return out;
}

pub fn tryDeviceBinaryF16(op: BinaryOp, lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    if (!build_options.enable_axiom_cuda) return null;
    if (!lhs.device.isCuda() or !rhs.device.isCuda() or !lhs.device.sameDevice(rhs.device)) return null;
    if (!lhs.sameShape(rhs) or lhs.data.len != 0 or rhs.data.len != 0 or !lhs.isContiguous() or !rhs.isContiguous()) return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;
    if (lhs_storage.len == 0 or lhs_storage.len != rhs_storage.len) return null;

    var out = try array_mod.Array(f16).emptyOn(lhs.allocator, lhs.shape, lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(lhs.allocator);
    const report = runtime.runCudaDeviceElementwiseF16(
        lhs.device.index,
        axiomBinaryOp(op),
        lhs_storage.len,
        lhs_storage.ptr,
        rhs_storage.ptr,
        out_storage.ptr,
    ) catch {
        out.deinit();
        return null;
    };
    if (!report.valid()) {
        out.deinit();
        return null;
    }
    return out;
}

pub fn tryDeviceBinaryF64(op: BinaryOp, lhs: array_mod.Array(f64), rhs: array_mod.Array(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    if (!build_options.enable_axiom_cuda) return null;
    if (!lhs.device.isCuda() or !rhs.device.isCuda() or !lhs.device.sameDevice(rhs.device)) return null;
    if (!lhs.sameShape(rhs) or lhs.data.len != 0 or rhs.data.len != 0 or !lhs.isContiguous() or !rhs.isContiguous()) return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;
    if (lhs_storage.len == 0 or lhs_storage.len != rhs_storage.len) return null;

    var out = try array_mod.Array(f64).emptyOn(lhs.allocator, lhs.shape, lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(lhs.allocator);
    const report = runtime.runCudaDeviceElementwiseF64(
        lhs.device.index,
        axiomBinaryOp(op),
        lhs_storage.len,
        lhs_storage.ptr,
        rhs_storage.ptr,
        out_storage.ptr,
    ) catch {
        out.deinit();
        return null;
    };
    if (!report.valid()) {
        out.deinit();
        return null;
    }
    return out;
}

pub fn tryDeviceBinaryBF16(op: BinaryOp, lhs: array_mod.Array(BFloat16), rhs: array_mod.Array(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    if (!build_options.enable_axiom_cuda) return null;
    if (!lhs.device.isCuda() or !rhs.device.isCuda() or !lhs.device.sameDevice(rhs.device)) return null;
    if (!lhs.sameShape(rhs) or lhs.data.len != 0 or rhs.data.len != 0 or !lhs.isContiguous() or !rhs.isContiguous()) return null;
    const lhs_storage = lhs.device_storage orelse return null;
    const rhs_storage = rhs.device_storage orelse return null;
    if (lhs_storage.len == 0 or lhs_storage.len != rhs_storage.len) return null;

    var out = try array_mod.Array(BFloat16).emptyOn(lhs.allocator, lhs.shape, lhs.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(lhs.allocator);
    const report = runtime.runCudaDeviceElementwiseBF16(
        lhs.device.index,
        axiomBinaryOp(op),
        lhs_storage.len,
        lhs_storage.ptr,
        rhs_storage.ptr,
        out_storage.ptr,
    ) catch {
        out.deinit();
        return null;
    };
    if (!report.valid()) {
        out.deinit();
        return null;
    }
    return out;
}

pub fn trySqrtF32(input: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryDeviceUnaryF32(.sqrt, input);
}

pub fn tryExpF32(input: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryDeviceUnaryF32(.exp, input);
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

pub fn tryDeviceUnaryF32(op: UnaryOp, input: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    if (!build_options.enable_axiom_cuda) return null;
    if (!input.device.isCuda() or input.data.len != 0 or !input.isContiguous()) return null;
    const in_storage = input.device_storage orelse return null;
    if (in_storage.len == 0) return null;
    var out = try array_mod.Array(f32).emptyOn(input.allocator, input.shape, input.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(input.allocator);
    const report = runtime.runCudaDeviceUnaryElementwiseF32(
        input.device.index,
        switch (op) {
            .sqrt => axiom.accelerator.TensorUnaryElementwiseOp.sqrt,
            .exp => axiom.accelerator.TensorUnaryElementwiseOp.exp,
            .abs => axiom.accelerator.TensorUnaryElementwiseOp.abs,
        },
        in_storage.len,
        in_storage.ptr,
        out_storage.ptr,
    ) catch {
        out.deinit();
        return null;
    };
    if (!report.valid()) {
        out.deinit();
        return null;
    }
    return out;
}

pub fn tryDeviceUnaryF16(op: UnaryOp, input: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    if (!build_options.enable_axiom_cuda) return null;
    if (!input.device.isCuda() or input.data.len != 0 or !input.isContiguous()) return null;
    const in_storage = input.device_storage orelse return null;
    if (in_storage.len == 0) return null;
    var out = try array_mod.Array(f16).emptyOn(input.allocator, input.shape, input.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(input.allocator);
    const report = runtime.runCudaDeviceUnaryElementwiseF16(
        input.device.index,
        switch (op) {
            .sqrt => axiom.accelerator.TensorUnaryElementwiseOp.sqrt,
            .exp => axiom.accelerator.TensorUnaryElementwiseOp.exp,
            .abs => axiom.accelerator.TensorUnaryElementwiseOp.abs,
        },
        in_storage.len,
        in_storage.ptr,
        out_storage.ptr,
    ) catch {
        out.deinit();
        return null;
    };
    if (!report.valid()) {
        out.deinit();
        return null;
    }
    return out;
}

pub fn tryDeviceUnaryBF16(op: UnaryOp, input: array_mod.Array(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    if (!build_options.enable_axiom_cuda) return null;
    if (!input.device.isCuda() or input.data.len != 0 or !input.isContiguous()) return null;
    const in_storage = input.device_storage orelse return null;
    if (in_storage.len == 0) return null;
    var out = try array_mod.Array(BFloat16).emptyOn(input.allocator, input.shape, input.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(input.allocator);
    const report = runtime.runCudaDeviceUnaryElementwiseBF16(
        input.device.index,
        switch (op) {
            .sqrt => axiom.accelerator.TensorUnaryElementwiseOp.sqrt,
            .exp => axiom.accelerator.TensorUnaryElementwiseOp.exp,
            .abs => axiom.accelerator.TensorUnaryElementwiseOp.abs,
        },
        in_storage.len,
        in_storage.ptr,
        out_storage.ptr,
    ) catch {
        out.deinit();
        return null;
    };
    if (!report.valid()) {
        out.deinit();
        return null;
    }
    return out;
}

pub fn tryDeviceUnaryF64(op: UnaryOp, input: array_mod.Array(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    if (!build_options.enable_axiom_cuda) return null;
    if (!input.device.isCuda() or input.data.len != 0 or !input.isContiguous()) return null;
    const in_storage = input.device_storage orelse return null;
    if (in_storage.len == 0) return null;
    var out = try array_mod.Array(f64).emptyOn(input.allocator, input.shape, input.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };

    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(input.allocator);
    const report = runtime.runCudaDeviceUnaryElementwiseF64(
        input.device.index,
        switch (op) {
            .sqrt => axiom.accelerator.TensorUnaryElementwiseOp.sqrt,
            .exp => axiom.accelerator.TensorUnaryElementwiseOp.exp,
            .abs => axiom.accelerator.TensorUnaryElementwiseOp.abs,
        },
        in_storage.len,
        in_storage.ptr,
        out_storage.ptr,
    ) catch {
        out.deinit();
        return null;
    };
    if (!report.valid()) {
        out.deinit();
        return null;
    }
    return out;
}

pub fn tryDeviceReductionF32(op: axiom.accelerator.DialectReductionOp, input: array_mod.Array(f32), axis: u1, keepdims: bool) array_mod.ArrayError!?array_mod.Array(f32) {
    if (!build_options.enable_axiom_cuda) return null;
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
    var out = try array_mod.Array(f32).emptyOn(input.allocator, out_shape, input.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(input.allocator);
    const report = runtime.runCudaDeviceReductionF32(
        input.device.index,
        op,
        input.shape[0],
        input.shape[1],
        axis,
        in_storage.ptr,
        out_storage.ptr,
    ) catch {
        out.deinit();
        return null;
    };
    if (!report.valid()) {
        out.deinit();
        return null;
    }
    return out;
}

pub fn tryDeviceBroadcastAddF32(input: array_mod.Array(f32), bias: array_mod.Array(f32), axis: axiom.accelerator.DialectBroadcastAxis) array_mod.ArrayError!?array_mod.Array(f32) {
    if (!build_options.enable_axiom_cuda) return null;
    if (!input.device.isCuda() or !bias.device.isCuda() or !input.device.sameDevice(bias.device)) return null;
    if (input.data.len != 0 or bias.data.len != 0 or !input.isContiguous() or !bias.isContiguous()) return null;
    if (input.shape.len != 2) return null;
    const expected_bias_len = switch (axis) {
        .row => input.shape[1],
        .column => input.shape[0],
    };
    const in_storage = input.device_storage orelse return null;
    const bias_storage = bias.device_storage orelse return null;
    if (in_storage.len == 0 or bias_storage.len != expected_bias_len) return null;
    var out = try array_mod.Array(f32).emptyOn(input.allocator, input.shape, input.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(input.allocator);
    const report = runtime.runCudaDeviceBroadcastAddF32(
        input.device.index,
        input.shape[0],
        input.shape[1],
        axis,
        in_storage.ptr,
        bias_storage.ptr,
        out_storage.ptr,
    ) catch {
        out.deinit();
        return null;
    };
    if (!report.valid()) {
        out.deinit();
        return null;
    }
    return out;
}

pub fn tryDeviceTransposeF32(input: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    if (!build_options.enable_axiom_cuda) return null;
    if (!input.device.isCuda() or input.data.len != 0 or !input.isContiguous()) return null;
    if (input.shape.len != 2) return null;
    const in_storage = input.device_storage orelse return null;
    if (in_storage.len == 0) return null;
    var out = try array_mod.Array(f32).emptyOn(input.allocator, &.{ input.shape[1], input.shape[0] }, input.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(input.allocator);
    const report = runtime.runCudaDeviceTransposeF32(
        input.device.index,
        input.shape[0],
        input.shape[1],
        in_storage.ptr,
        out_storage.ptr,
    ) catch {
        out.deinit();
        return null;
    };
    if (!report.valid()) {
        out.deinit();
        return null;
    }
    return out;
}

pub fn tryDeviceTransposeF64(input: array_mod.Array(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    if (!build_options.enable_axiom_cuda) return null;
    if (!input.device.isCuda() or input.data.len != 0 or !input.isContiguous()) return null;
    if (input.shape.len != 2) return null;
    const in_storage = input.device_storage orelse return null;
    if (in_storage.len == 0) return null;
    var out = try array_mod.Array(f64).emptyOn(input.allocator, &.{ input.shape[1], input.shape[0] }, input.device);
    errdefer out.deinit();
    const out_storage = out.device_storage orelse {
        out.deinit();
        return null;
    };
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(input.allocator);
    const report = runtime.runCudaDeviceTransposeF64(
        input.device.index,
        input.shape[0],
        input.shape[1],
        in_storage.ptr,
        out_storage.ptr,
    ) catch {
        out.deinit();
        return null;
    };
    if (!report.valid()) {
        out.deinit();
        return null;
    }
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
        else => return null,
    };
    if (!result.verified) return null;
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

pub fn tryMulViewF32(lhs: array_mod.ArrayView(f32), rhs: array_mod.ArrayView(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryBinaryViewF32(.mul, lhs, rhs);
}

pub fn tryDivViewF32(lhs: array_mod.ArrayView(f32), rhs: array_mod.ArrayView(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryBinaryViewF32(.div, lhs, rhs);
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
        else => return null,
    };
    if (!result.verified) return null;
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

    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(lhs.allocator);
    const cublas_report = runtime.runCudaDeviceSgemm(lhs.device.index, m, n, k, lhs_storage.ptr, rhs_storage.ptr, out_storage.ptr) catch null;
    if (cublas_report) |report| {
        recordCudaDeviceGemmReport(report);
        if (report.valid()) return out;
    }

    try zeroStorage(out_storage);
    var session = withCudaContext(lhs.device.index) catch return error.InvalidDevice;
    defer session.driver.close();
    defer session.context.release(&session.driver);
    var cuda_arch_buffer: [16]u8 = undefined;
    const resolved_arch = session.driver.resolveCudaArch(session.context.device, "auto", &cuda_arch_buffer) catch return error.BackendFailure;
    const target = axiom.accelerator.cuda_backend.CudaBackendTarget.fromArch(resolved_arch);
    const plan = axiom.accelerator.GemmBuilder.init()
        .dimensions(m, n, k)
        .alpha(1.0)
        .beta(0.0)
        .tile(@intCast(@min(n, @as(usize, 16))), @intCast(@min(m, @as(usize, 16))))
        .cudaArch(resolved_arch)
        .kernelSymbol("vectra_device_tile_gemm")
        .build();
    var artifact = axiom.accelerator.cuda_backend.lowerGemmKernelIrToCudaArtifact(lhs.allocator, plan.lowerToKernelIr(), target) catch return null;
    defer artifact.deinit(lhs.allocator);
    if (!artifact.valid()) return null;
    const module = session.driver.moduleLoadData(artifact.ptx) catch return null;
    defer session.driver.moduleUnload(module);
    var symbol_buffer: [128]u8 = undefined;
    const symbol = std.fmt.bufPrintSentinel(&symbol_buffer, "{s}", .{plan.kernel_symbol}, 0) catch return null;
    const function = session.driver.moduleGetFunction(module, symbol.ptr) catch return null;
    const invocation = plan.invocation() catch return null;

    var a_ptr = lhs_storage.ptr;
    var b_ptr = rhs_storage.ptr;
    var c_ptr = out_storage.ptr;
    var m_arg: i32 = std.math.cast(i32, m) orelse return error.InvalidShape;
    var n_arg: i32 = std.math.cast(i32, n) orelse return error.InvalidShape;
    var k_arg: i32 = std.math.cast(i32, k) orelse return error.InvalidShape;
    var args = [_]?*anyopaque{
        @ptrCast(&a_ptr),
        @ptrCast(&b_ptr),
        @ptrCast(&c_ptr),
        @ptrCast(&m_arg),
        @ptrCast(&n_arg),
        @ptrCast(&k_arg),
    };
    session.driver.launchKernel(function, invocation.grid, invocation.block, invocation.shared_memory_bytes, &args) catch return null;
    session.driver.synchronize() catch return null;
    return out;
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

    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(lhs.allocator);
    const report = runtime.runCudaDeviceDgemm(lhs.device.index, m, n, k, lhs_storage.ptr, rhs_storage.ptr, out_storage.ptr) catch null;
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

    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(lhs.allocator);
    const report = runtime.runCudaDeviceF16Gemm(lhs.device.index, m, n, k, lhs_storage.ptr, rhs_storage.ptr, out_storage.ptr) catch null;
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

        var runtime = axiom.accelerator.AcceleratorRuntime.cuda(lhs.allocator);
        const report = runtime.runCudaDeviceSgemmLtMatmulAdd(lhs.device.index, m, n, k, lhs_storage.ptr, rhs_storage.ptr, add_storage.ptr, out_storage.ptr) catch null;
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

    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(lhs.allocator);
    const report = runtime.runCudaDeviceDgemmLtMatmulAdd(lhs.device.index, m, n, k, lhs_storage.ptr, rhs_storage.ptr, add_storage.ptr, out_storage.ptr) catch null;
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

    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(lhs.allocator);
    const report = runtime.runCudaDeviceF16GemmLtMatmulAdd(lhs.device.index, m, n, k, lhs_storage.ptr, rhs_storage.ptr, add_storage.ptr, out_storage.ptr) catch null;
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

    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(lhs.allocator);
    const report = runtime.runCudaDeviceBf16Gemm(lhs.device.index, m, n, k, lhs_storage.ptr, rhs_storage.ptr, out_storage.ptr) catch null;
    if (report) |value| {
        recordCudaDeviceGemmReport(value);
        if (value.valid()) return out;
    }
    return null;
}

pub fn runPendingMatmulF32(allocator: std.mem.Allocator, device: array_mod.Device, m: usize, n: usize, k: usize, lhs_ptr: u64, rhs_ptr: u64, out_ptr: u64) array_mod.ArrayError!bool {
    resetLastCudaDeviceGemmReport();
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(allocator);
    const report = runtime.runCudaDeviceSgemm(device.index, m, n, k, lhs_ptr, rhs_ptr, out_ptr) catch return error.BackendFailure;
    recordCudaDeviceGemmReport(report);
    return report.valid();
}

pub fn runPendingMatmulBF16(allocator: std.mem.Allocator, device: array_mod.Device, m: usize, n: usize, k: usize, lhs_ptr: u64, rhs_ptr: u64, out_ptr: u64) array_mod.ArrayError!bool {
    resetLastCudaDeviceGemmReport();
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(allocator);
    const report = runtime.runCudaDeviceBf16Gemm(device.index, m, n, k, lhs_ptr, rhs_ptr, out_ptr) catch return error.BackendFailure;
    recordCudaDeviceGemmReport(report);
    return report.valid();
}

pub fn runPendingMatmulF16(allocator: std.mem.Allocator, device: array_mod.Device, m: usize, n: usize, k: usize, lhs_ptr: u64, rhs_ptr: u64, out_ptr: u64) array_mod.ArrayError!bool {
    resetLastCudaDeviceGemmReport();
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(allocator);
    const report = runtime.runCudaDeviceF16Gemm(device.index, m, n, k, lhs_ptr, rhs_ptr, out_ptr) catch return error.BackendFailure;
    recordCudaDeviceGemmReport(report);
    return report.valid();
}

pub fn runPendingMatmulF64(allocator: std.mem.Allocator, device: array_mod.Device, m: usize, n: usize, k: usize, lhs_ptr: u64, rhs_ptr: u64, out_ptr: u64) array_mod.ArrayError!bool {
    resetLastCudaDeviceGemmReport();
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(allocator);
    const report = runtime.runCudaDeviceDgemm(device.index, m, n, k, lhs_ptr, rhs_ptr, out_ptr) catch return error.BackendFailure;
    recordCudaDeviceGemmReport(report);
    return report.valid();
}

pub fn runPendingMatmulAddF32(allocator: std.mem.Allocator, device: array_mod.Device, m: usize, n: usize, k: usize, lhs_ptr: u64, rhs_ptr: u64, add_ptr: u64, out_ptr: u64, alpha: f32, beta: f32) array_mod.ArrayError!bool {
    resetLastCudaDeviceGemmReport();
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(allocator);
    const report = runtime.runCudaDeviceSgemmLtMatmulAddEx(device.index, m, n, k, lhs_ptr, rhs_ptr, add_ptr, out_ptr, alpha, beta) catch return error.BackendFailure;
    recordCudaDeviceGemmReport(report);
    return report.valid();
}

pub fn runPendingMatmulAddBF16(allocator: std.mem.Allocator, device: array_mod.Device, m: usize, n: usize, k: usize, lhs_ptr: u64, rhs_ptr: u64, add_ptr: u64, out_ptr: u64, alpha: f32, beta: f32) array_mod.ArrayError!bool {
    resetLastCudaDeviceGemmReport();
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(allocator);
    const report = runtime.runCudaDeviceBf16GemmLtMatmulAddEx(device.index, m, n, k, lhs_ptr, rhs_ptr, add_ptr, out_ptr, alpha, beta) catch return error.BackendFailure;
    recordCudaDeviceGemmReport(report);
    return report.valid();
}

pub fn runPendingMatmulAddF16(allocator: std.mem.Allocator, device: array_mod.Device, m: usize, n: usize, k: usize, lhs_ptr: u64, rhs_ptr: u64, add_ptr: u64, out_ptr: u64, alpha: f32, beta: f32) array_mod.ArrayError!bool {
    resetLastCudaDeviceGemmReport();
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(allocator);
    const report = runtime.runCudaDeviceF16GemmLtMatmulAddEx(device.index, m, n, k, lhs_ptr, rhs_ptr, add_ptr, out_ptr, alpha, beta) catch return error.BackendFailure;
    recordCudaDeviceGemmReport(report);
    return report.valid();
}

pub fn runPendingMatmulAddF64(allocator: std.mem.Allocator, device: array_mod.Device, m: usize, n: usize, k: usize, lhs_ptr: u64, rhs_ptr: u64, add_ptr: u64, out_ptr: u64, alpha: f32, beta: f32) array_mod.ArrayError!bool {
    resetLastCudaDeviceGemmReport();
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(allocator);
    const byte_count = std.math.mul(usize, m, n) catch return error.InvalidShape;
    const bytes = std.math.mul(usize, byte_count, @sizeOf(f64)) catch return error.InvalidShape;
    try copyStorage(
        .{ .device = device, .ptr = out_ptr, .len = byte_count, .bytes = bytes, .owns = false },
        .{ .device = device, .ptr = add_ptr, .len = byte_count, .bytes = bytes, .owns = false },
    );
    const report = runtime.runCudaDeviceDgemmEx(device.index, m, n, k, lhs_ptr, rhs_ptr, out_ptr, alpha, beta) catch return error.BackendFailure;
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
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(allocator);
    const report = runtime.runCudaDeviceF32MatmulAddUnary(
        device.index,
        switch (op) {
            .sqrt => axiom.accelerator.TensorUnaryElementwiseOp.sqrt,
            .exp => axiom.accelerator.TensorUnaryElementwiseOp.exp,
            .abs => return error.TypeUnsupported,
        },
        m,
        n,
        k,
        lhs_ptr,
        rhs_ptr,
        add_ptr,
        out_ptr,
        alpha,
        beta,
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

    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(lhs.allocator);
    const report = runtime.runCudaDeviceBf16GemmLtMatmulAdd(lhs.device.index, m, n, k, lhs_storage.ptr, rhs_storage.ptr, add_storage.ptr, out_storage.ptr) catch null;
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
        else => return null,
    };
    if (!result.ok()) return null;
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
        else => return null,
    };
    if (!result.ok()) return null;
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

    if (report.add_ok and report.sub_ok and report.mul_ok and report.div_ok and report.saxpy_ok and report.matmul_ok and report.matmul_tile_ir_ok and report.f16_add_ok and report.f16_matmul_ok and report.bf16_add_ok and report.bf16_matmul_ok and report.typed_f16_gemm_plan.ok and report.typed_bf16_gemm_plan.ok and report.f16_widened_execution_fingerprint != 0 and report.bf16_widened_execution_fingerprint != 0 and report.typed_f16_gemm_route_fingerprint != 0 and report.typed_bf16_gemm_route_fingerprint != 0 and std.mem.eql(u8, report.typed_f16_gemm_route, "widened_f32_cuda_compute") and std.mem.eql(u8, report.typed_bf16_gemm_route, "widened_f32_cuda_compute") and report.scalar_add_ok and report.scalar_mul_ok and report.scalar_saxpy_ok) {
        report.status = .ran;
        report.issue_count = @as(u8, @intFromBool(!report.lhs_plan.ok)) +
            @as(u8, @intFromBool(!report.lhs_plan.copy_ok));
    } else if (add_out == null and sub_out == null and mul_out == null and div_out == null and saxpy_out == null and matmul_out == null and f16_add_out == null and f16_matmul_out == null and bf16_add_out == null and bf16_matmul_out == null and scalar_add_out == null and scalar_mul_out == null and scalar_saxpy_out == null) {
        report.status = .skipped;
        report.issue_count = 0;
    } else {
        report.status = .failed;
        report.issue_count = @as(u8, @intFromBool(!report.lhs_plan.ok)) +
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
            @as(u8, @intFromBool(!report.scalar_saxpy_ok));
    }
    return report;
}

fn tryBinaryViewF32(op: BinaryOp, lhs: array_mod.ArrayView(f32), rhs: array_mod.ArrayView(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    if (!build_options.enable_axiom_cuda) return null;
    if (!supportedOneDimensionalView(lhs) or !supportedOneDimensionalView(rhs) or !std.mem.eql(usize, lhs.shape, rhs.shape)) return null;

    var out = try array_mod.Array(f32).empty(lhs.allocator, lhs.shape);
    errdefer out.deinit();

    const lhs_slice = viewBackingSlice(lhs) orelse return null;
    const rhs_slice = viewBackingSlice(rhs) orelse return null;
    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(lhs.allocator);
    const result = runtime.runTensorElementwiseBinary(lhs_slice, rhs_slice, out.data, .{
        .op = axiomBinaryOp(op),
        .len = lhs.shape[0],
        .lhs_stride = @intCast(lhs.strides[0]),
        .rhs_stride = @intCast(rhs.strides[0]),
        .out_stride = 1,
        .kernel_symbol = switch (op) {
            .add => "vectra_axiom_strided_add",
            .sub => "vectra_axiom_strided_sub",
            .mul => "vectra_axiom_strided_mul",
            .div => "vectra_axiom_strided_div",
        },
    }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return null,
    };
    if (!result.verified) return null;
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
        else => return null,
    };
    if (!result.verified) return null;
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
        else => return null,
    };
    if (!result.ok()) return null;
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
        else => return null,
    };
    if (!result.ok()) return null;
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
        else => return null,
    };
    if (!result.ok()) return null;
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
    return view.device.isCpu() and view.shape.len == 1 and view.shape[0] != 0 and view.strides.len == 1 and view.strides[0] != 0;
}

fn viewBackingSlice(view: array_mod.ArrayView(f32)) ?[]const f32 {
    if (!supportedOneDimensionalView(view)) return null;
    const last_delta = std.math.mul(usize, view.shape[0] - 1, view.strides[0]) catch return null;
    const end_index = std.math.add(usize, view.offset, last_delta) catch return null;
    if (end_index >= view.data.len) return null;
    return view.data[view.offset .. end_index + 1];
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
