//! Optional Axiom CUDA bridge for Vectra.
//!
//! This module deliberately keeps CUDA acceleration opt-in.  The default Vectra
//! build remains CPU/Veyra/Alea only, while `zig build -Daxiom-cuda=true ...`
//! imports Axiom and routes a small f32 tensor-kernel seed through Axiom's
//! builder-style CUDA tensor runtime.  The bridge is host-slice based today: it
//! proves Vectra metadata can feed Axiom's tensor adapter without claiming that
//! `Array.cuda()` is a persistent device-resident storage backend yet.

const std = @import("std");
const build_options = @import("vectra_build_options");
const array_mod = @import("../array.zig");

const axiom = if (build_options.enable_axiom_cuda) @import("axiom") else struct {};

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
    mul,
};

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

pub const SmokeReport = struct {
    enabled: bool = build_options.enable_axiom_cuda,
    status: Status = if (build_options.enable_axiom_cuda) .skipped else .disabled,
    add_ok: bool = false,
    mul_ok: bool = false,
    saxpy_ok: bool = false,
    matmul_ok: bool = false,
    max_abs_error: f32 = 0.0,
    lhs_plan: BufferPlanEvidence = .{},
    output_fingerprint: u64 = 0,
    issue_count: u8 = 0,

    pub fn ok(report: SmokeReport) bool {
        return report.issue_count == 0 and switch (report.status) {
            .disabled => !report.enabled,
            .skipped => report.enabled,
            .ran => report.enabled and report.add_ok and report.mul_ok and report.saxpy_ok and report.matmul_ok,
            .failed => false,
        };
    }

    pub fn fingerprint(report: SmokeReport) u64 {
        var hasher = std.hash.Wyhash.init(0x0abc_7aaa_11cc_0001);
        hashBool(&hasher, report.enabled);
        hashBytes(&hasher, report.status.label());
        hashBool(&hasher, report.add_ok);
        hashBool(&hasher, report.mul_ok);
        hashBool(&hasher, report.saxpy_ok);
        hashBool(&hasher, report.matmul_ok);
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
        hashU64(&hasher, report.output_fingerprint);
        hashU64(&hasher, report.issue_count);
        return hasher.final();
    }

    pub fn writeText(report: SmokeReport, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        try writer.print(
            "vectra_axiom_cuda_smoke enabled={} status={s} ok={} issues={d} add={} mul={} saxpy={} matmul={} max_abs_error={d} logical_elements={d} required_bytes={d} linear_copy={} copy_plan_ok={} copy_requires_strided={} output={x} fingerprint={x}\n",
            .{
                report.enabled,
                report.status.label(),
                report.ok(),
                report.issue_count,
                report.add_ok,
                report.mul_ok,
                report.saxpy_ok,
                report.matmul_ok,
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
                "  \"mul_ok\": {},\n" ++
                "  \"saxpy_ok\": {},\n" ++
                "  \"matmul_ok\": {},\n" ++
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
                "  \"output_fingerprint\": {d},\n" ++
                "  \"fingerprint\": {d}\n" ++
                "}}\n",
            .{
                report.enabled,
                report.status.label(),
                report.ok(),
                report.issue_count,
                report.add_ok,
                report.mul_ok,
                report.saxpy_ok,
                report.matmul_ok,
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

pub fn tryAddF32(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryBinaryF32(.add, lhs, rhs);
}

pub fn tryMulF32(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryBinaryF32(.mul, lhs, rhs);
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

    var spec = axiom.accelerator.TensorGemmSpec.rowMajor(
        .rowMajor("lhs", @intCast(@intFromPtr(lhs.data.ptr)), m, k),
        .rowMajor("rhs", @intCast(@intFromPtr(rhs.data.ptr)), k, n),
        .rowMajor("out", @intCast(@intFromPtr(out.data.ptr)), m, n),
    );
    spec.alpha = 1.0;
    spec.beta = 0.0;
    spec.tile_x = 16;
    spec.tile_y = 16;
    spec.kernel_symbol = "vectra_axiom_gemm";

    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(lhs.allocator);
    const result = runtime.runTensorGemmHostSlices(spec, lhs.data, rhs.data, c.data, out.data) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return null,
    };
    if (!result.verified) return null;
    return out;
}

pub fn runSmoke(allocator: std.mem.Allocator) SmokeReport {
    if (!build_options.enable_axiom_cuda) return .{};

    var lhs = array_mod.Array(f32).fromSlice(allocator, &.{ 1, 2, 3, 4 }, &.{4}) catch return failedReport();
    defer lhs.deinit();
    var rhs = array_mod.Array(f32).fromSlice(allocator, &.{ 10, 20, 30, 40 }, &.{4}) catch return failedReport();
    defer rhs.deinit();

    var report: SmokeReport = .{
        .enabled = true,
        .status = .skipped,
        .lhs_plan = planArrayF32(lhs, "lhs"),
    };

    var add_out = tryAddF32(lhs, rhs) catch return failedReport();
    if (add_out) |*out| {
        defer out.deinit();
        report.add_ok = sliceClose(out.data, &.{ 11, 22, 33, 44 }, 0.0);
        report.max_abs_error = @max(report.max_abs_error, maxAbsError(out.data, &.{ 11, 22, 33, 44 }));
        report.output_fingerprint ^= hashF32Slice(out.data);
    }

    var mul_out = tryMulF32(lhs, rhs) catch return failedReport();
    if (mul_out) |*out| {
        defer out.deinit();
        report.mul_ok = sliceClose(out.data, &.{ 10, 40, 90, 160 }, 0.0);
        report.max_abs_error = @max(report.max_abs_error, maxAbsError(out.data, &.{ 10, 40, 90, 160 }));
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
        report.max_abs_error = @max(report.max_abs_error, maxAbsError(out.data, &.{ 58, 64, 139, 154 }));
        report.output_fingerprint ^= hashF32Slice(out.data);
    }

    if (report.add_ok and report.mul_ok and report.saxpy_ok and report.matmul_ok) {
        report.status = .ran;
        report.issue_count = @as(u8, @intFromBool(!report.lhs_plan.ok)) +
            @as(u8, @intFromBool(!report.lhs_plan.copy_ok));
    } else if (add_out == null and mul_out == null and saxpy_out == null and matmul_out == null) {
        report.status = .skipped;
        report.issue_count = 0;
    } else {
        report.status = .failed;
        report.issue_count = @as(u8, @intFromBool(!report.lhs_plan.ok)) +
            @as(u8, @intFromBool(!report.lhs_plan.copy_ok)) +
            @as(u8, @intFromBool(!report.add_ok)) +
            @as(u8, @intFromBool(!report.mul_ok)) +
            @as(u8, @intFromBool(!report.saxpy_ok)) +
            @as(u8, @intFromBool(!report.matmul_ok));
    }
    return report;
}

fn tryBinaryF32(op: BinaryOp, lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    if (!build_options.enable_axiom_cuda) return null;
    if (!supportedSameShapeContiguous(lhs, rhs)) return null;

    var out = try array_mod.Array(f32).empty(lhs.allocator, lhs.shape);
    errdefer out.deinit();

    var runtime = axiom.accelerator.AcceleratorRuntime.cuda(lhs.allocator);
    const axiom_op: axiom.accelerator.TensorBinaryElementwiseOp = switch (op) {
        .add => .add,
        .mul => .mul,
    };
    const result = runtime.runTensorElementwiseBinary(lhs.data, rhs.data, out.data, .{
        .op = axiom_op,
        .len = lhs.data.len,
        .kernel_symbol = switch (op) {
            .add => "vectra_axiom_add",
            .mul => "vectra_axiom_mul",
        },
    }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return null,
    };
    if (!result.verified) return null;
    return out;
}

fn supportedSameShapeContiguous(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) bool {
    return lhs.device.isCpu() and
        rhs.device.isCpu() and
        lhs.data.len != 0 and
        lhs.sameShape(rhs) and
        lhs.isContiguous() and
        rhs.isContiguous();
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

fn failedReport() SmokeReport {
    return .{
        .enabled = build_options.enable_axiom_cuda,
        .status = if (build_options.enable_axiom_cuda) .failed else .disabled,
        .issue_count = 1,
    };
}

fn sliceClose(actual: []const f32, expected: []const f32, tolerance: f32) bool {
    if (actual.len != expected.len) return false;
    return maxAbsError(actual, expected) <= tolerance;
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

fn hashF32(hasher: *std.hash.Wyhash, value: f32) void {
    var bytes: [4]u8 = undefined;
    std.mem.writeInt(u32, &bytes, @bitCast(value), .little);
    hasher.update(&bytes);
}

fn hashF32Slice(values: []const f32) u64 {
    var hasher = std.hash.Wyhash.init(0x0abc_7aaa_f325_511c);
    hashU64(&hasher, values.len);
    for (values) |value| hashF32(&hasher, value);
    return hasher.final();
}

test "Axiom CUDA bridge is disabled by default but reports deterministically" {
    const report = runSmoke(std.testing.allocator);
    if (build_options.enable_axiom_cuda) {
        try std.testing.expect(report.status == .ran or report.status == .skipped or report.status == .failed);
    } else {
        try std.testing.expect(!enabled());
        try std.testing.expect(report.ok());
        try std.testing.expectEqual(Status.disabled, report.status);
        try std.testing.expectEqual(@as(u8, 0), report.issue_count);
    }
}
