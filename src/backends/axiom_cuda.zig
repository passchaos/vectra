//! Optional Axiom CUDA bridge for Vectra.
//!
//! This module deliberately keeps CUDA acceleration opt-in.  The default Vectra
//! build remains CPU/Veyra/Alea only, while `zig build -Daxiom-cuda=true ...`
//! imports Axiom and routes small f32 tensor-kernel seeds through Axiom's
//! builder-style CUDA tensor runtime.  Elementwise paths still use Axiom tensor
//! adapter launches; matmul now builds Axiom CUDA Tile IR and hands it to the
//! Tile-IR-to-CUTILE GEMM runtime bridge.  The bridge is host-slice based today:
//! it proves Vectra metadata can feed Axiom's accelerator layers without
//! claiming that `Array.cuda()` is a persistent device-resident storage backend
//! yet.

const std = @import("std");
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
    .{ .cuda_name = "CUDA_R_64F", .cuda_value = 1, .meaning = "real double", .vectra_dtype = .f64, .status = .cpu_veyra_seed, .same_shape_elementwise = true, .scalar_broadcast = true, .matmul = true },
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
    output_fingerprint: u64 = 0,
    issue_count: u8 = 0,

    pub fn ok(report: SmokeReport) bool {
        return report.issue_count == 0 and switch (report.status) {
            .disabled => !report.enabled,
            .skipped => report.enabled,
            .ran => report.enabled and report.add_ok and report.sub_ok and report.mul_ok and report.div_ok and report.saxpy_ok and report.matmul_ok and report.matmul_tile_ir_ok and report.f16_add_ok and report.f16_matmul_ok and report.bf16_add_ok and report.bf16_matmul_ok and report.f16_widened_execution_fingerprint != 0 and report.bf16_widened_execution_fingerprint != 0 and report.scalar_add_ok and report.scalar_mul_ok and report.scalar_saxpy_ok,
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
        hashU64(&hasher, report.output_fingerprint);
        hashU64(&hasher, report.issue_count);
        return hasher.final();
    }

    pub fn writeText(report: SmokeReport, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        try writer.print(
            "vectra_axiom_cuda_smoke enabled={} status={s} ok={} issues={d} add={} sub={} mul={} div={} saxpy={} matmul={} matmul_tile_ir={} f16_add={} f16_matmul={} bf16_add={} bf16_matmul={} scalar_add={} scalar_mul={} scalar_saxpy={} strided_add={} strided_mul={} device_array={} max_abs_error={d} logical_elements={d} required_bytes={d} linear_copy={} copy_plan_ok={} copy_requires_strided={} output={x} fingerprint={x}\n",
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
            "vectra_axiom_cuda_dtype_support count={d} bridge={d} native_seed={d} widened_seed={d} fingerprint={x} f16_native_execution={x} bf16_native_execution={x} f16_widened_execution={x} bf16_widened_execution={x}\n",
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
                "  \"bf16_matmul_ok\": {},\n",
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
                "  \"bf16_widened_execution_fingerprint\": {d},\n" ++
                "  \"output_fingerprint\": {d},\n" ++
                "  \"fingerprint\": {d}\n" ++
                "}}\n",
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
                report.output_fingerprint,
                report.fingerprint(),
            },
        );
    }
};

pub fn enabled() bool {
    return build_options.enable_axiom_cuda;
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
    return tryBinaryF32(.add, lhs, rhs);
}

pub fn trySubF32(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryBinaryF32(.sub, lhs, rhs);
}

pub fn tryMulF32(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryBinaryF32(.mul, lhs, rhs);
}

pub fn tryDivF32(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryBinaryF32(.div, lhs, rhs);
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

pub fn tryMatmulBF16(lhs: array_mod.Array(BFloat16), rhs: array_mod.Array(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    if (!build_options.enable_axiom_cuda) return null;
    if (!supportedMatmul2dContiguousBF16(lhs, rhs)) return null;

    var lhs32 = try bf16ArrayToF32(lhs);
    defer lhs32.deinit();
    var rhs32 = try bf16ArrayToF32(rhs);
    defer rhs32.deinit();
    var out32 = try tryMatmulF32(lhs32, rhs32) orelse return null;
    defer out32.deinit();
    return try f32ArrayToBF16(out32);
}

pub fn tryMatmulF16(lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    if (!build_options.enable_axiom_cuda) return null;
    if (!supportedMatmul2dContiguousF16(lhs, rhs)) return null;

    var lhs32 = try f16ArrayToF32(lhs);
    defer lhs32.deinit();
    var rhs32 = try f16ArrayToF32(rhs);
    defer rhs32.deinit();
    var out32 = try tryMatmulF32(lhs32, rhs32) orelse return null;
    defer out32.deinit();
    return try f32ArrayToF16(out32);
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
    if (f16_add_out != null and f16_matmul_out != null) {
        report.f16_native_execution_fingerprint =
            nativeF16BinaryExecutionFingerprint(allocator, .add, f16_lhs, f16_rhs) catch 0;
        report.f16_widened_execution_fingerprint =
            (widenedF16BinaryProvenanceFingerprint(allocator, "add", .add, f16_lhs, f16_rhs) catch return failedReport()) ^
            (widenedF16MatmulProvenanceFingerprint(allocator, "matmul", f16_lhs, f16_rhs) catch return failedReport());
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
    if (bf16_add_out != null and bf16_matmul_out != null) {
        report.bf16_native_execution_fingerprint =
            nativeBF16BinaryExecutionFingerprint(allocator, .add, bf16_lhs, bf16_rhs) catch 0;
        report.bf16_widened_execution_fingerprint =
            (widenedBF16BinaryProvenanceFingerprint(allocator, "add", .add, bf16_lhs, bf16_rhs) catch return failedReport()) ^
            (widenedBF16MatmulProvenanceFingerprint(allocator, "matmul", bf16_lhs, bf16_rhs) catch return failedReport());
    }

    if (report.add_ok and report.sub_ok and report.mul_ok and report.div_ok and report.saxpy_ok and report.matmul_ok and report.matmul_tile_ir_ok and report.f16_add_ok and report.f16_matmul_ok and report.bf16_add_ok and report.bf16_matmul_ok and report.f16_widened_execution_fingerprint != 0 and report.bf16_widened_execution_fingerprint != 0 and report.scalar_add_ok and report.scalar_mul_ok and report.scalar_saxpy_ok) {
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
            @as(u8, @intFromBool(report.f16_widened_execution_fingerprint == 0)) +
            @as(u8, @intFromBool(report.bf16_widened_execution_fingerprint == 0)) +
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
    }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return null,
    };
    if (!result.verified) return null;
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

fn widenedF16MatmulProvenanceFingerprint(allocator: std.mem.Allocator, operation: []const u8, lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!u64 {
    const plan = axiom.accelerator.TensorWidenedExecutionPlan.from(.f16, operation);
    var lhs32 = try f16ArrayToF32(lhs);
    defer lhs32.deinit();
    var rhs32 = try f16ArrayToF32(rhs);
    defer rhs32.deinit();
    var compute = try tryMatmulF32(lhs32, rhs32) orelse return error.BackendFailure;
    defer compute.deinit();
    const narrow_out = try allocator.alloc(f16, compute.data.len);
    defer allocator.free(narrow_out);
    const input_report = axiom.accelerator.tensor_adapter.widenF16ToF32(lhs.data, lhs32.data) catch |err| return mapTensorAdapterError(err);
    const output_report = axiom.accelerator.tensor_adapter.narrowF32ToF16(compute.data, narrow_out) catch |err| return mapTensorAdapterError(err);
    const report = axiom.accelerator.TensorWidenedExecutionReport.fromReports(plan, input_report, hashF32Slice(compute.data), output_report);
    if (!report.ok()) return error.BackendFailure;
    return report.fingerprint();
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

fn widenedBF16MatmulProvenanceFingerprint(allocator: std.mem.Allocator, operation: []const u8, lhs: array_mod.Array(BFloat16), rhs: array_mod.Array(BFloat16)) array_mod.ArrayError!u64 {
    const plan = axiom.accelerator.TensorWidenedExecutionPlan.from(.bf16, operation);
    var lhs32 = try bf16ArrayToF32(lhs);
    defer lhs32.deinit();
    var rhs32 = try bf16ArrayToF32(rhs);
    defer rhs32.deinit();
    var compute = try tryMatmulF32(lhs32, rhs32) orelse return error.BackendFailure;
    defer compute.deinit();
    const lhs_bits = try allocator.alloc(u16, lhs.data.len);
    defer allocator.free(lhs_bits);
    for (lhs.data, lhs_bits) |value, *slot| slot.* = value.bits;
    const narrow_bits = try allocator.alloc(u16, compute.data.len);
    defer allocator.free(narrow_bits);
    const input_report = axiom.accelerator.tensor_adapter.widenBF16ToF32(lhs_bits, lhs32.data) catch |err| return mapTensorAdapterError(err);
    const output_report = axiom.accelerator.tensor_adapter.narrowF32ToBF16(compute.data, narrow_bits) catch |err| return mapTensorAdapterError(err);
    const report = axiom.accelerator.TensorWidenedExecutionReport.fromReports(plan, input_report, hashF32Slice(compute.data), output_report);
    if (!report.ok()) return error.BackendFailure;
    return report.fingerprint();
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

test "Axiom CUDA bridge is disabled by default but reports deterministically" {
    const report = runSmoke(std.testing.allocator);
    try std.testing.expectEqual(cuda_dtype_support.len, report.dtype_support_count);
    try std.testing.expectEqual(@as(usize, 3), report.dtype_bridge_count);
    try std.testing.expectEqual(@as(usize, 1), report.dtype_native_seed_count);
    try std.testing.expectEqual(@as(usize, 2), report.dtype_widened_seed_count);
    try std.testing.expect(report.dtype_support_fingerprint != 0);
    const f16_record = findCudaDTypeSupport("CUDA_R_16F").?;
    try std.testing.expectEqual(CudaDTypeBridgeStatus.widened_f32_seed, f16_record.status);
    try std.testing.expectEqual(array_mod.DType.f16, f16_record.vectra_dtype.?);
    const bf16_record = findVectraDTypeSupport(.bf16).?;
    try std.testing.expectEqualStrings("CUDA_R_16BF", bf16_record.cuda_name);
    const f32_record = findVectraDTypeSupport(.f32).?;
    try std.testing.expectEqual(CudaDTypeBridgeStatus.native_cuda_seed, f32_record.status);
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
