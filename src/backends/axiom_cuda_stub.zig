//! Unavailable CUDA backend bridge.
//!
//! macOS builds intentionally do not compile Vectra's real CUDA bridge. This
//! stub preserves the backend facade shape while reporting CUDA as unavailable.

const std = @import("std");
const axiom = @import("axiom");
const array_mod = @import("../array.zig");

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
        _ = report;
        return false;
    }
};

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
        _ = report;
        return false;
    }
};

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
        _ = report;
        return false;
    }
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
        _ = record;
        return 0;
    }
};

pub const cuda_dtype_support = [_]CudaDTypeSupportRecord{};

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
        _ = evidence;
        return 0;
    }
};

pub const DeviceArrayF32 = struct {
    allocator: std.mem.Allocator,
    shape: []usize,
    device_ptr: u64 = 0,
    required_bytes: usize = 0,
    logical_elements: usize = 0,
    allocation_fingerprint: u64 = 0,
    pool_fingerprint: u64 = 0,
    released: bool = true,

    pub fn fromHost(allocator: std.mem.Allocator, host: array_mod.Array(f32)) array_mod.ArrayError!?DeviceArrayF32 {
        _ = allocator;
        _ = host;
        return null;
    }

    pub fn deinit(self: *DeviceArrayF32) void {
        if (self.shape.len != 0) self.allocator.free(self.shape);
        self.* = undefined;
    }

    pub fn release(self: *DeviceArrayF32) void {
        self.released = true;
    }

    pub fn ok(self: DeviceArrayF32) bool {
        _ = self;
        return false;
    }

    pub fn fingerprint(self: DeviceArrayF32) u64 {
        _ = self;
        return 0;
    }
};

pub const SmokeReport = struct {
    enabled: bool = false,
    status: Status = .disabled,
    issue_count: u8 = 0,

    pub fn ok(report: SmokeReport) bool {
        return !report.enabled and report.status == .disabled and report.issue_count == 0;
    }

    pub fn fingerprint(report: SmokeReport) u64 {
        _ = report;
        return 0;
    }

    pub fn writeText(report: SmokeReport, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        try writer.print("vectra_axiom_cuda_smoke enabled={} status={s} ok={} issues={d}\n", .{
            report.enabled,
            report.status.label(),
            report.ok(),
            report.issue_count,
        });
    }

    pub fn writeJson(report: SmokeReport, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        try writer.print(
            "{{\n  \"kind\": \"vectra_axiom_cuda_smoke\",\n  \"enabled\": {},\n  \"status\": \"{s}\",\n  \"ok\": {},\n  \"issue_count\": {d},\n  \"fingerprint\": {d}\n}}\n",
            .{
                report.enabled,
                report.status.label(),
                report.ok(),
                report.issue_count,
                report.fingerprint(),
            },
        );
    }
};

pub fn resetLastCudaDeviceGemmReport() void {}

pub fn lastCudaDeviceGemmReport() CudaDeviceGemmReportSnapshot {
    return .{};
}

pub fn resetLastCudaDeviceBatchedGemmReport() void {}

pub fn lastCudaDeviceBatchedGemmReport() CudaDeviceBatchedGemmReportSnapshot {
    return .{};
}

pub fn resetLastCudaDeviceMemRefReport() void {}

pub fn lastCudaDeviceMemRefReport() CudaDeviceMemRefReportSnapshot {
    return .{};
}

pub fn synchronizeDevice(allocator: std.mem.Allocator, device: array_mod.Device) array_mod.ArrayError!void {
    _ = allocator;
    _ = device;
    return error.InvalidDevice;
}

pub fn cudaDTypeSupportRecords() []const CudaDTypeSupportRecord {
    return &cuda_dtype_support;
}

pub fn findCudaDTypeSupport(cuda_name: []const u8) ?CudaDTypeSupportRecord {
    _ = cuda_name;
    return null;
}

pub fn findVectraDTypeSupport(dtype: array_mod.DType) ?CudaDTypeSupportRecord {
    _ = dtype;
    return null;
}

pub fn cudaDTypeNativeSeedCount() usize {
    return 0;
}

pub fn cudaDTypeWidenedSeedCount() usize {
    return 0;
}

pub fn cudaDTypeBridgeCount() usize {
    return 0;
}

pub fn cudaDTypeSupportFingerprint() u64 {
    return 0;
}

pub fn toDeviceF32(allocator: std.mem.Allocator, host: array_mod.Array(f32)) array_mod.ArrayError!?DeviceArrayF32 {
    _ = allocator;
    _ = host;
    return null;
}

pub fn flushStorageCache() void {}

pub fn allocateStorage(device: array_mod.Device, len: usize, element_size: usize) array_mod.ArrayError!?array_mod.DeviceStorage {
    _ = device;
    _ = len;
    _ = element_size;
    return error.InvalidDevice;
}

pub fn freeStorage(storage: array_mod.DeviceStorage) void {
    _ = storage;
}

pub fn uploadStorage(storage: array_mod.DeviceStorage, bytes: []const u8) array_mod.ArrayError!void {
    _ = storage;
    _ = bytes;
    return error.InvalidDevice;
}

pub fn downloadStorage(storage: array_mod.DeviceStorage, bytes: []u8) array_mod.ArrayError!void {
    _ = storage;
    _ = bytes;
    return error.InvalidDevice;
}

pub fn copyStorage(dst: array_mod.DeviceStorage, src: array_mod.DeviceStorage) array_mod.ArrayError!void {
    _ = dst;
    _ = src;
    return error.InvalidDevice;
}

pub fn zeroStorage(storage: array_mod.DeviceStorage) array_mod.ArrayError!void {
    _ = storage;
    return error.InvalidDevice;
}

pub fn fillStorage(comptime T: type, storage: array_mod.DeviceStorage, value: T) array_mod.ArrayError!void {
    _ = storage;
    _ = value;
    return error.InvalidDevice;
}

pub fn fillPhiloxUniform(comptime T: type, storage: array_mod.DeviceStorage, seed: u64) array_mod.ArrayError!void {
    _ = T;
    _ = storage;
    _ = seed;
    return error.InvalidDevice;
}

pub fn fillPhiloxNormal(comptime T: type, storage: array_mod.DeviceStorage, seed: u64, mean: T, stddev: T) array_mod.ArrayError!void {
    _ = storage;
    _ = seed;
    _ = mean;
    _ = stddev;
    return error.InvalidDevice;
}

pub fn enabled() bool {
    return false;
}

pub fn deviceAvailable(index: usize) bool {
    _ = index;
    return false;
}

pub fn planArrayF32(input: array_mod.Array(f32), name: []const u8) BufferPlanEvidence {
    _ = input;
    _ = name;
    return .{};
}

pub fn planTypedGemmF16(lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!TypedGemmPlanEvidence {
    _ = lhs;
    _ = rhs;
    return .{};
}

pub fn planTypedGemmBF16(lhs: array_mod.Array(BFloat16), rhs: array_mod.Array(BFloat16)) array_mod.ArrayError!TypedGemmPlanEvidence {
    _ = lhs;
    _ = rhs;
    return .{};
}

pub fn unaryElementSupported(comptime T: type, op: UnaryOp) bool {
    _ = T;
    _ = op;
    return false;
}

pub fn runSmoke(allocator: std.mem.Allocator) SmokeReport {
    _ = allocator;
    return .{};
}

fn noArray(comptime T: type) array_mod.ArrayError!?array_mod.Array(T) {
    return null;
}

fn noDeviceArray(comptime T: type) array_mod.ArrayError!?array_mod.Array(T) {
    return null;
}

pub fn tryAddF32(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = lhs;
    _ = rhs;
    return noArray(f32);
}

pub fn trySubF32(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = lhs;
    _ = rhs;
    return noArray(f32);
}

pub fn tryMulF32(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = lhs;
    _ = rhs;
    return noArray(f32);
}

pub fn tryDivF32(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = lhs;
    _ = rhs;
    return noArray(f32);
}

pub fn tryBinaryF64(op: BinaryOp, lhs: array_mod.Array(f64), rhs: array_mod.Array(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    _ = op;
    _ = lhs;
    _ = rhs;
    return noArray(f64);
}

pub fn tryAddF16(lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    _ = lhs;
    _ = rhs;
    return noArray(f16);
}

pub fn trySubF16(lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    _ = lhs;
    _ = rhs;
    return noArray(f16);
}

pub fn tryMulF16(lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    _ = lhs;
    _ = rhs;
    return noArray(f16);
}

pub fn tryDivF16(lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    _ = lhs;
    _ = rhs;
    return noArray(f16);
}

pub fn tryAddBF16(lhs: array_mod.Array(BFloat16), rhs: array_mod.Array(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    _ = lhs;
    _ = rhs;
    return noArray(BFloat16);
}

pub fn trySubBF16(lhs: array_mod.Array(BFloat16), rhs: array_mod.Array(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    _ = lhs;
    _ = rhs;
    return noArray(BFloat16);
}

pub fn tryMulBF16(lhs: array_mod.Array(BFloat16), rhs: array_mod.Array(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    _ = lhs;
    _ = rhs;
    return noArray(BFloat16);
}

pub fn tryDivBF16(lhs: array_mod.Array(BFloat16), rhs: array_mod.Array(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    _ = lhs;
    _ = rhs;
    return noArray(BFloat16);
}

pub fn tryDeviceBinaryF32(op: BinaryOp, lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = op;
    _ = lhs;
    _ = rhs;
    return noDeviceArray(f32);
}

pub fn tryDeviceBinaryF16(op: BinaryOp, lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    _ = op;
    _ = lhs;
    _ = rhs;
    return noDeviceArray(f16);
}

pub fn tryDeviceBinaryF64(op: BinaryOp, lhs: array_mod.Array(f64), rhs: array_mod.Array(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    _ = op;
    _ = lhs;
    _ = rhs;
    return noDeviceArray(f64);
}

pub fn tryDeviceBinaryBF16(op: BinaryOp, lhs: array_mod.Array(BFloat16), rhs: array_mod.Array(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    _ = op;
    _ = lhs;
    _ = rhs;
    return noDeviceArray(BFloat16);
}

pub fn tryDeviceUnaryF32(op: UnaryOp, input: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = op;
    _ = input;
    return noDeviceArray(f32);
}

pub fn tryDeviceUnaryF16(op: UnaryOp, input: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    _ = op;
    _ = input;
    return noDeviceArray(f16);
}

pub fn tryDeviceUnaryBF16(op: UnaryOp, input: array_mod.Array(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    _ = op;
    _ = input;
    return noDeviceArray(BFloat16);
}

pub fn tryDeviceUnaryF64(op: UnaryOp, input: array_mod.Array(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    _ = op;
    _ = input;
    return noDeviceArray(f64);
}

pub fn tryDeviceReduction(comptime T: type, op: axiom.accelerator.DialectReductionOp, input: array_mod.Array(T), axis: u1, keepdims: bool) array_mod.ArrayError!?array_mod.Array(T) {
    _ = op;
    _ = input;
    _ = axis;
    _ = keepdims;
    return noDeviceArray(T);
}

pub fn tryDeviceBroadcast(comptime T: type, op: BinaryOp, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    _ = op;
    _ = lhs;
    _ = rhs;
    return noDeviceArray(T);
}

pub fn tryDeviceBroadcastBinary(comptime T: type, op: BinaryOp, input: array_mod.Array(T), bias: array_mod.Array(T), axis: axiom.accelerator.DialectBroadcastAxis) array_mod.ArrayError!?array_mod.Array(T) {
    _ = op;
    _ = input;
    _ = bias;
    _ = axis;
    return noDeviceArray(T);
}

pub fn tryDeviceLastDimBroadcastF32(op: BinaryOp, input: array_mod.Array(f32), bias: array_mod.Array(f32), bias_left: bool) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = op;
    _ = input;
    _ = bias;
    _ = bias_left;
    return noDeviceArray(f32);
}

pub fn tryDeviceLastDimBroadcastF64(op: BinaryOp, input: array_mod.Array(f64), bias: array_mod.Array(f64), bias_left: bool) array_mod.ArrayError!?array_mod.Array(f64) {
    _ = op;
    _ = input;
    _ = bias;
    _ = bias_left;
    return noDeviceArray(f64);
}

pub fn tryDeviceLastDimBroadcastF16(op: BinaryOp, input: array_mod.Array(f16), bias: array_mod.Array(f16), bias_left: bool) array_mod.ArrayError!?array_mod.Array(f16) {
    _ = op;
    _ = input;
    _ = bias;
    _ = bias_left;
    return noDeviceArray(f16);
}

pub fn tryDeviceLastDimBroadcastBF16(op: BinaryOp, input: array_mod.Array(BFloat16), bias: array_mod.Array(BFloat16), bias_left: bool) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    _ = op;
    _ = input;
    _ = bias;
    _ = bias_left;
    return noDeviceArray(BFloat16);
}

pub fn tryDeviceContiguousScalarBroadcastF32(op: BinaryOp, input: array_mod.Array(f32), scalar: array_mod.Array(f32), scalar_left: bool) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = op;
    _ = input;
    _ = scalar;
    _ = scalar_left;
    return noDeviceArray(f32);
}

pub fn tryDeviceContiguousScalarBroadcastF64(op: BinaryOp, input: array_mod.Array(f64), scalar: array_mod.Array(f64), scalar_left: bool) array_mod.ArrayError!?array_mod.Array(f64) {
    _ = op;
    _ = input;
    _ = scalar;
    _ = scalar_left;
    return noDeviceArray(f64);
}

pub fn tryDeviceContiguousScalarBroadcastF16(op: BinaryOp, input: array_mod.Array(f16), scalar: array_mod.Array(f16), scalar_left: bool) array_mod.ArrayError!?array_mod.Array(f16) {
    _ = op;
    _ = input;
    _ = scalar;
    _ = scalar_left;
    return noDeviceArray(f16);
}

pub fn tryDeviceContiguousScalarBroadcastBF16(op: BinaryOp, input: array_mod.Array(BFloat16), scalar: array_mod.Array(BFloat16), scalar_left: bool) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    _ = op;
    _ = input;
    _ = scalar;
    _ = scalar_left;
    return noDeviceArray(BFloat16);
}

pub fn tryDeviceLogSoftmax(comptime T: type, input: array_mod.Array(T), axis: u1) array_mod.ArrayError!?array_mod.Array(T) {
    _ = input;
    _ = axis;
    return noDeviceArray(T);
}

pub fn tryDeviceSoftmax(comptime T: type, input: array_mod.Array(T), axis: u1) array_mod.ArrayError!?array_mod.Array(T) {
    _ = input;
    _ = axis;
    return noDeviceArray(T);
}

pub fn tryDeviceTranspose(comptime T: type, input: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    _ = input;
    return noDeviceArray(T);
}

fn noView(comptime T: type) array_mod.ArrayError!?array_mod.Array(T) {
    return null;
}

pub fn tryAddViewF32(lhs: array_mod.ArrayView(f32), rhs: array_mod.ArrayView(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = lhs;
    _ = rhs;
    return noView(f32);
}

pub fn trySubViewF32(lhs: array_mod.ArrayView(f32), rhs: array_mod.ArrayView(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = lhs;
    _ = rhs;
    return noView(f32);
}

pub fn tryMulViewF32(lhs: array_mod.ArrayView(f32), rhs: array_mod.ArrayView(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = lhs;
    _ = rhs;
    return noView(f32);
}

pub fn tryDivViewF32(lhs: array_mod.ArrayView(f32), rhs: array_mod.ArrayView(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = lhs;
    _ = rhs;
    return noView(f32);
}

pub fn tryAddViewF64(lhs: array_mod.ArrayView(f64), rhs: array_mod.ArrayView(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    _ = lhs;
    _ = rhs;
    return noView(f64);
}

pub fn trySubViewF64(lhs: array_mod.ArrayView(f64), rhs: array_mod.ArrayView(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    _ = lhs;
    _ = rhs;
    return noView(f64);
}

pub fn tryMulViewF64(lhs: array_mod.ArrayView(f64), rhs: array_mod.ArrayView(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    _ = lhs;
    _ = rhs;
    return noView(f64);
}

pub fn tryDivViewF64(lhs: array_mod.ArrayView(f64), rhs: array_mod.ArrayView(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    _ = lhs;
    _ = rhs;
    return noView(f64);
}

pub fn tryAddViewF16(lhs: array_mod.ArrayView(f16), rhs: array_mod.ArrayView(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    _ = lhs;
    _ = rhs;
    return noView(f16);
}

pub fn trySubViewF16(lhs: array_mod.ArrayView(f16), rhs: array_mod.ArrayView(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    _ = lhs;
    _ = rhs;
    return noView(f16);
}

pub fn tryMulViewF16(lhs: array_mod.ArrayView(f16), rhs: array_mod.ArrayView(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    _ = lhs;
    _ = rhs;
    return noView(f16);
}

pub fn tryDivViewF16(lhs: array_mod.ArrayView(f16), rhs: array_mod.ArrayView(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    _ = lhs;
    _ = rhs;
    return noView(f16);
}

pub fn tryAddViewBF16(lhs: array_mod.ArrayView(BFloat16), rhs: array_mod.ArrayView(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    _ = lhs;
    _ = rhs;
    return noView(BFloat16);
}

pub fn trySubViewBF16(lhs: array_mod.ArrayView(BFloat16), rhs: array_mod.ArrayView(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    _ = lhs;
    _ = rhs;
    return noView(BFloat16);
}

pub fn tryMulViewBF16(lhs: array_mod.ArrayView(BFloat16), rhs: array_mod.ArrayView(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    _ = lhs;
    _ = rhs;
    return noView(BFloat16);
}

pub fn tryDivViewBF16(lhs: array_mod.ArrayView(BFloat16), rhs: array_mod.ArrayView(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    _ = lhs;
    _ = rhs;
    return noView(BFloat16);
}

pub fn tryViewScalarF32(op: BinaryOp, input: array_mod.ArrayView(f32), scalar: f32, scalar_left: bool) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = op;
    _ = input;
    _ = scalar;
    _ = scalar_left;
    return noView(f32);
}

pub fn tryViewScalarF64(op: BinaryOp, input: array_mod.ArrayView(f64), scalar: f64, scalar_left: bool) array_mod.ArrayError!?array_mod.Array(f64) {
    _ = op;
    _ = input;
    _ = scalar;
    _ = scalar_left;
    return noView(f64);
}

pub fn tryViewScalarF16(op: BinaryOp, input: array_mod.ArrayView(f16), scalar: f16, scalar_left: bool) array_mod.ArrayError!?array_mod.Array(f16) {
    _ = op;
    _ = input;
    _ = scalar;
    _ = scalar_left;
    return noView(f16);
}

pub fn tryViewScalarBF16(op: BinaryOp, input: array_mod.ArrayView(BFloat16), scalar: BFloat16, scalar_left: bool) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    _ = op;
    _ = input;
    _ = scalar;
    _ = scalar_left;
    return noView(BFloat16);
}

pub fn tryAbsViewF32(input: array_mod.ArrayView(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = input;
    return noView(f32);
}

pub fn trySqrtViewF32(input: array_mod.ArrayView(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = input;
    return noView(f32);
}

pub fn tryExpViewF32(input: array_mod.ArrayView(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = input;
    return noView(f32);
}

pub fn tryLogViewF32(input: array_mod.ArrayView(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = input;
    return noView(f32);
}

pub fn tryExp2ViewF32(input: array_mod.ArrayView(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = input;
    return noView(f32);
}

pub fn tryExpm1ViewF32(input: array_mod.ArrayView(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = input;
    return noView(f32);
}

pub fn tryLog1pViewF32(input: array_mod.ArrayView(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = input;
    return noView(f32);
}

pub fn tryLog2ViewF32(input: array_mod.ArrayView(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = input;
    return noView(f32);
}

pub fn tryLog10ViewF32(input: array_mod.ArrayView(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = input;
    return noView(f32);
}

pub fn trySinViewF32(input: array_mod.ArrayView(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = input;
    return noView(f32);
}

pub fn tryCosViewF32(input: array_mod.ArrayView(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = input;
    return noView(f32);
}

pub fn tryTanViewF32(input: array_mod.ArrayView(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = input;
    return noView(f32);
}

pub fn tryAbsViewF64(input: array_mod.ArrayView(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    _ = input;
    return noView(f64);
}

pub fn trySqrtViewF64(input: array_mod.ArrayView(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    _ = input;
    return noView(f64);
}

pub fn tryExpViewF64(input: array_mod.ArrayView(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    _ = input;
    return noView(f64);
}

pub fn tryAbsViewF16(input: array_mod.ArrayView(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    _ = input;
    return noView(f16);
}

pub fn trySqrtViewF16(input: array_mod.ArrayView(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    _ = input;
    return noView(f16);
}

pub fn tryExpViewF16(input: array_mod.ArrayView(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    _ = input;
    return noView(f16);
}

pub fn tryAbsViewBF16(input: array_mod.ArrayView(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    _ = input;
    return noView(BFloat16);
}

pub fn trySqrtViewBF16(input: array_mod.ArrayView(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    _ = input;
    return noView(BFloat16);
}

pub fn tryExpViewBF16(input: array_mod.ArrayView(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    _ = input;
    return noView(BFloat16);
}

pub fn tryMatmulF32(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    _ = lhs;
    _ = rhs;
    return noArray(f32);
}

pub fn tryMatmulF16(lhs: array_mod.Array(f16), rhs: array_mod.Array(f16)) array_mod.ArrayError!?array_mod.Array(f16) {
    _ = lhs;
    _ = rhs;
    return noArray(f16);
}

pub fn tryMatmulBF16(lhs: array_mod.Array(BFloat16), rhs: array_mod.Array(BFloat16)) array_mod.ArrayError!?array_mod.Array(BFloat16) {
    _ = lhs;
    _ = rhs;
    return noArray(BFloat16);
}

pub fn tryDeviceMatmul(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    _ = lhs;
    _ = rhs;
    return noDeviceArray(T);
}

pub fn tryDeviceMatmulAdd(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T), addend: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    _ = lhs;
    _ = rhs;
    _ = addend;
    return noDeviceArray(T);
}

pub fn tryDeviceMatvec(comptime T: type, matrix: array_mod.Array(T), vector: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    _ = matrix;
    _ = vector;
    return noDeviceArray(T);
}

pub fn tryDeviceVecmat(comptime T: type, vector: array_mod.Array(T), matrix: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    _ = vector;
    _ = matrix;
    return noDeviceArray(T);
}

pub fn tryDeviceDot(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    _ = lhs;
    _ = rhs;
    return noDeviceArray(T);
}

pub fn tryDeviceBmm(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    _ = lhs;
    _ = rhs;
    return noDeviceArray(T);
}

pub fn tryDeviceBatchedMatmul(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    _ = lhs;
    _ = rhs;
    return noDeviceArray(T);
}

pub fn runPendingMatmul(
    comptime T: type,
    allocator: std.mem.Allocator,
    device: array_mod.Device,
    m: usize,
    n: usize,
    k: usize,
    lhs_ptr: u64,
    rhs_ptr: u64,
    out_ptr: u64,
) array_mod.ArrayError!bool {
    _ = T;
    _ = allocator;
    _ = device;
    _ = m;
    _ = n;
    _ = k;
    _ = lhs_ptr;
    _ = rhs_ptr;
    _ = out_ptr;
    return false;
}

pub fn runPendingMatmulAdd(
    comptime T: type,
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
    _ = T;
    _ = allocator;
    _ = device;
    _ = m;
    _ = n;
    _ = k;
    _ = lhs_ptr;
    _ = rhs_ptr;
    _ = add_ptr;
    _ = out_ptr;
    _ = alpha;
    _ = beta;
    return false;
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
    _ = allocator;
    _ = device;
    _ = op;
    _ = m;
    _ = n;
    _ = k;
    _ = lhs_ptr;
    _ = rhs_ptr;
    _ = add_ptr;
    _ = out_ptr;
    _ = alpha;
    _ = beta;
    return false;
}
