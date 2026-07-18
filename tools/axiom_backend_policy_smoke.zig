const std = @import("std");
const vx = @import("vectra");

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;
    var a = try vx.Array(f32).fromSlice(allocator, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a.deinit();
    var b = try vx.Array(f32).fromSlice(allocator, &.{ 7, 8, 9, 10, 11, 12 }, &.{ 3, 2 });
    defer b.deinit();
    vx.resetDefaultDialectBackend();
    const default_cpu_policy = vx.axiom_backend.defaultBackendPolicy();
    vx.setDefaultDialectBackend(.cuda);
    const default_cuda_policy = vx.axiom_backend.defaultBackendPolicy();
    vx.setDefaultDialectBackend(.mps);
    const default_mps_policy = vx.axiom_backend.defaultBackendPolicy();
    const default_mps_execution_target = vx.defaultExecutionTarget();
    vx.resetDefaultDialectBackend();
    const report = vx.axiom_backend.selectMatmul(f32, .prefer_cuda, a, b);
    var out = try vx.axiom_backend.matmul(f32, vx.axiom_backend.defaultBackendPolicy(), a, b);
    defer out.deinit();
    var matmul64_lhs = try vx.Array(f64).fromSlice(allocator, &.{ 1, 2, 3, 4 }, &.{ 2, 2 });
    defer matmul64_lhs.deinit();
    var matmul64_rhs = try vx.Array(f64).ones(allocator, &.{ 2, 2 });
    defer matmul64_rhs.deinit();
    const matmul64_report = vx.axiom_backend.selectMatmul(f64, .prefer_cuda, matmul64_lhs, matmul64_rhs);
    vx.resetDefaultDialectBackend();
    var eager_cpu = try a.matmul(b);
    defer eager_cpu.deinit();
    vx.setDefaultDialectBackend(.cuda);
    const default_cuda_execution_target = vx.defaultExecutionTarget();
    var eager_cuda_default = try a.matmul(b);
    defer eager_cuda_default.deinit();
    vx.setDefaultDialectBackend(.mps);
    const default_mps_fallback_execution_target = vx.defaultExecutionTarget();
    var eager_mps_default = try a.matmul(b);
    defer eager_mps_default.deinit();
    const cpu_device_target = vx.executionTargetForDevice(vx.cpu);
    const cuda_device_target = vx.executionTargetForDevice(vx.cuda(0));
    const mps_device_target = vx.executionTargetForDevice(vx.mps(0));
    vx.resetDefaultDialectBackend();
    var lhs32 = try vx.Array(f32).fromSlice(allocator, &.{ 1, 2, 3, 4 }, &.{4});
    defer lhs32.deinit();
    var rhs32 = try vx.Array(f32).fromSlice(allocator, &.{ 10, 20, 30, 40 }, &.{4});
    defer rhs32.deinit();
    const ew32_report = vx.axiom_backend.selectElementwise(f32, .add, .prefer_cuda, lhs32, rhs32);
    var ew32 = try vx.axiom_backend.elementwise(f32, .add, .prefer_cuda, lhs32, rhs32);
    defer ew32.deinit();

    var lhs64 = try vx.Array(f64).fromSlice(allocator, &.{ 8, 6, 4, 2 }, &.{4});
    defer lhs64.deinit();
    var rhs64 = try vx.Array(f64).fromSlice(allocator, &.{ 2, 3, 4, 2 }, &.{4});
    defer rhs64.deinit();
    const ew64_report = vx.axiom_backend.selectElementwise(f64, .div, .prefer_axiom_cpu, lhs64, rhs64);
    const ew64_cuda_report = vx.axiom_backend.selectElementwise(f64, .div, .prefer_cuda, lhs64, rhs64);
    var ew64 = try vx.axiom_backend.elementwise(f64, .div, .prefer_axiom_cpu, lhs64, rhs64);
    defer ew64.deinit();
    const scalar64_report = vx.axiom_backend.selectScalarElementwise(f64, .sub, .prefer_axiom_cpu, lhs64, 2.0, .rhs);
    const scalar64_cuda_report = vx.axiom_backend.selectScalarElementwise(f64, .sub, .prefer_cuda, lhs64, 2.0, .rhs);
    var scalar64 = try vx.axiom_backend.elementwiseScalar(f64, .sub, .prefer_axiom_cpu, lhs64, 2.0, .rhs);
    defer scalar64.deinit();

    var strided_lhs = try vx.Array(f32).fromSlice(allocator, &.{ 1, 99, 2, 99, 3, 99, 4, 99 }, &.{8});
    defer strided_lhs.deinit();
    var strided_rhs = try vx.Array(f32).fromSlice(allocator, &.{ 10, 99, 20, 99, 30, 99, 40, 99 }, &.{8});
    defer strided_rhs.deinit();
    var lhs_view = try strided_lhs.asStrided(&.{4}, &.{2}, 0);
    defer lhs_view.deinit();
    var rhs_view = try strided_rhs.asStrided(&.{4}, &.{2}, 0);
    defer rhs_view.deinit();
    vx.setDefaultDialectBackend(.cuda);
    var view_add = try lhs_view.add(rhs_view);
    defer view_add.deinit();
    var view_sub = try rhs_view.sub(lhs_view);
    defer view_sub.deinit();
    var view_mul = try lhs_view.mul(rhs_view);
    defer view_mul.deinit();
    var view_div = try rhs_view.div(lhs_view);
    defer view_div.deinit();
    var view_scalar_add = try lhs_view.addScalar(2.0);
    defer view_scalar_add.deinit();
    var view_scalar_sub = try lhs_view.subScalar(2.0);
    defer view_scalar_sub.deinit();
    var view_scalar_mul = try lhs_view.mulScalar(2.0);
    defer view_scalar_mul.deinit();
    var view_scalar_div = try rhs_view.divScalar(10.0);
    defer view_scalar_div.deinit();
    var view_neg = try lhs_view.neg();
    defer view_neg.deinit();
    var view_abs = try view_neg.abs();
    defer view_abs.deinit();
    var view_square = try lhs_view.square();
    defer view_square.deinit();
    var view_reciprocal = try lhs_view.reciprocal();
    defer view_reciprocal.deinit();

    var f64_strided_lhs = try vx.Array(f64).fromSlice(allocator, &.{ 1, 99, 2, 99, 3, 99, 4, 99 }, &.{8});
    defer f64_strided_lhs.deinit();
    var f64_strided_rhs = try vx.Array(f64).fromSlice(allocator, &.{ 10, 99, 20, 99, 30, 99, 40, 99 }, &.{8});
    defer f64_strided_rhs.deinit();
    var f64_lhs_view = try f64_strided_lhs.asStrided(&.{4}, &.{2}, 0);
    defer f64_lhs_view.deinit();
    var f64_rhs_view = try f64_strided_rhs.asStrided(&.{4}, &.{2}, 0);
    defer f64_rhs_view.deinit();
    var f64_view_add = try f64_lhs_view.add(f64_rhs_view);
    defer f64_view_add.deinit();
    var f64_view_sub = try f64_rhs_view.sub(f64_lhs_view);
    defer f64_view_sub.deinit();
    var f64_view_mul = try f64_lhs_view.mul(f64_rhs_view);
    defer f64_view_mul.deinit();
    var f64_view_div = try f64_rhs_view.div(f64_lhs_view);
    defer f64_view_div.deinit();
    var f64_view_scalar_add = try f64_lhs_view.addScalar(2.0);
    defer f64_view_scalar_add.deinit();
    var f64_view_scalar_sub = try f64_lhs_view.subScalar(2.0);
    defer f64_view_scalar_sub.deinit();
    var f64_view_scalar_mul = try f64_lhs_view.mulScalar(2.0);
    defer f64_view_scalar_mul.deinit();
    var f64_view_scalar_div = try f64_rhs_view.divScalar(10.0);
    defer f64_view_scalar_div.deinit();
    var f64_view_neg = try f64_lhs_view.neg();
    defer f64_view_neg.deinit();
    var f64_view_square = try f64_lhs_view.square();
    defer f64_view_square.deinit();
    var f64_view_reciprocal = try f64_lhs_view.reciprocal();
    defer f64_view_reciprocal.deinit();

    var f16_strided_lhs = try vx.Array(f16).fromSlice(allocator, &.{
        @as(f16, 1), @as(f16, 99), @as(f16, 2), @as(f16, 99),
        @as(f16, 3), @as(f16, 99), @as(f16, 4), @as(f16, 99),
    }, &.{8});
    defer f16_strided_lhs.deinit();
    var f16_strided_rhs = try vx.Array(f16).fromSlice(allocator, &.{
        @as(f16, 10), @as(f16, 99), @as(f16, 20), @as(f16, 99),
        @as(f16, 30), @as(f16, 99), @as(f16, 40), @as(f16, 99),
    }, &.{8});
    defer f16_strided_rhs.deinit();
    var f16_lhs_view = try f16_strided_lhs.asStrided(&.{4}, &.{2}, 0);
    defer f16_lhs_view.deinit();
    var f16_rhs_view = try f16_strided_rhs.asStrided(&.{4}, &.{2}, 0);
    defer f16_rhs_view.deinit();
    var f16_view_add = try f16_lhs_view.add(f16_rhs_view);
    defer f16_view_add.deinit();
    var f16_view_sub = try f16_rhs_view.sub(f16_lhs_view);
    defer f16_view_sub.deinit();
    var f16_view_mul = try f16_lhs_view.mul(f16_rhs_view);
    defer f16_view_mul.deinit();
    var f16_view_div = try f16_rhs_view.div(f16_lhs_view);
    defer f16_view_div.deinit();
    var f16_view_scalar_add = try f16_lhs_view.addScalar(@as(f16, 2.0));
    defer f16_view_scalar_add.deinit();
    var f16_view_scalar_sub = try f16_lhs_view.subScalar(@as(f16, 2.0));
    defer f16_view_scalar_sub.deinit();
    var f16_view_scalar_mul = try f16_lhs_view.mulScalar(@as(f16, 2.0));
    defer f16_view_scalar_mul.deinit();
    var f16_view_scalar_div = try f16_rhs_view.divScalar(@as(f16, 10.0));
    defer f16_view_scalar_div.deinit();
    var f16_view_neg = try f16_lhs_view.neg();
    defer f16_view_neg.deinit();
    var f16_view_square = try f16_lhs_view.square();
    defer f16_view_square.deinit();
    var f16_view_reciprocal = try f16_lhs_view.reciprocal();
    defer f16_view_reciprocal.deinit();

    var bf16_strided_lhs = try vx.Array(vx.BFloat16).fromSlice(allocator, &.{
        vx.BFloat16.fromF32(1), vx.BFloat16.fromF32(99), vx.BFloat16.fromF32(2), vx.BFloat16.fromF32(99),
        vx.BFloat16.fromF32(3), vx.BFloat16.fromF32(99), vx.BFloat16.fromF32(4), vx.BFloat16.fromF32(99),
    }, &.{8});
    defer bf16_strided_lhs.deinit();
    var bf16_strided_rhs = try vx.Array(vx.BFloat16).fromSlice(allocator, &.{
        vx.BFloat16.fromF32(10), vx.BFloat16.fromF32(99), vx.BFloat16.fromF32(20), vx.BFloat16.fromF32(99),
        vx.BFloat16.fromF32(30), vx.BFloat16.fromF32(99), vx.BFloat16.fromF32(40), vx.BFloat16.fromF32(99),
    }, &.{8});
    defer bf16_strided_rhs.deinit();
    var bf16_lhs_view = try bf16_strided_lhs.asStrided(&.{4}, &.{2}, 0);
    defer bf16_lhs_view.deinit();
    var bf16_rhs_view = try bf16_strided_rhs.asStrided(&.{4}, &.{2}, 0);
    defer bf16_rhs_view.deinit();
    var bf16_view_add = try bf16_lhs_view.add(bf16_rhs_view);
    defer bf16_view_add.deinit();
    var bf16_view_sub = try bf16_rhs_view.sub(bf16_lhs_view);
    defer bf16_view_sub.deinit();
    var bf16_view_mul = try bf16_lhs_view.mul(bf16_rhs_view);
    defer bf16_view_mul.deinit();
    var bf16_view_div = try bf16_rhs_view.div(bf16_lhs_view);
    defer bf16_view_div.deinit();
    var bf16_view_scalar_add = try bf16_lhs_view.addScalar(vx.BFloat16.fromF32(2.0));
    defer bf16_view_scalar_add.deinit();
    var bf16_view_scalar_sub = try bf16_lhs_view.subScalar(vx.BFloat16.fromF32(2.0));
    defer bf16_view_scalar_sub.deinit();
    var bf16_view_scalar_mul = try bf16_lhs_view.mulScalar(vx.BFloat16.fromF32(2.0));
    defer bf16_view_scalar_mul.deinit();
    var bf16_view_scalar_div = try bf16_rhs_view.divScalar(vx.BFloat16.fromF32(10.0));
    defer bf16_view_scalar_div.deinit();
    var bf16_view_neg = try bf16_lhs_view.neg();
    defer bf16_view_neg.deinit();
    var bf16_view_square = try bf16_lhs_view.square();
    defer bf16_view_square.deinit();
    var bf16_view_reciprocal = try bf16_lhs_view.reciprocal();
    defer bf16_view_reciprocal.deinit();
    vx.resetDefaultDialectBackend();

    const matmul_ok = report.ok() and matmul64_report.ok() and matmul64_report.selected == .axiom_cuda and out.data[0] == 58 and out.data[3] == 154;
    const elementwise_ok = ew32_report.ok() and ew64_report.ok() and ew64_cuda_report.ok() and
        ew64_cuda_report.selected == .axiom_cuda and
        equalF32(ew32.data, &.{ 11, 22, 33, 44 }) and
        equalF64(ew64.data, &.{ 4, 2, 1, 1 });
    const scalar_ok = scalar64_report.ok() and scalar64_cuda_report.ok() and
        scalar64_cuda_report.selected == .axiom_cuda and
        equalF64(scalar64.data, &.{ 6, 4, 2, 0 });
    const view_ok = equalF32(view_add.data, &.{ 11, 22, 33, 44 }) and
        equalF32(view_sub.data, &.{ 9, 18, 27, 36 }) and
        equalF32(view_mul.data, &.{ 10, 40, 90, 160 }) and
        equalF32(view_div.data, &.{ 10, 10, 10, 10 });
    const view_scalar_ok = equalF32(view_scalar_add.data, &.{ 3, 4, 5, 6 }) and
        equalF32(view_scalar_sub.data, &.{ -1, 0, 1, 2 }) and
        equalF32(view_scalar_mul.data, &.{ 2, 4, 6, 8 }) and
        equalF32(view_scalar_div.data, &.{ 1, 2, 3, 4 });
    const view64_ok = equalF64(f64_view_add.data, &.{ 11, 22, 33, 44 }) and
        equalF64(f64_view_sub.data, &.{ 9, 18, 27, 36 }) and
        equalF64(f64_view_mul.data, &.{ 10, 40, 90, 160 }) and
        equalF64(f64_view_div.data, &.{ 10, 10, 10, 10 });
    const view64_scalar_ok = equalF64(f64_view_scalar_add.data, &.{ 3, 4, 5, 6 }) and
        equalF64(f64_view_scalar_sub.data, &.{ -1, 0, 1, 2 }) and
        equalF64(f64_view_scalar_mul.data, &.{ 2, 4, 6, 8 }) and
        equalF64(f64_view_scalar_div.data, &.{ 1, 2, 3, 4 });
    const view16_ok = equalF16(f16_view_add.data, &.{ 11, 22, 33, 44 }, 0.02) and
        equalF16(f16_view_sub.data, &.{ 9, 18, 27, 36 }, 0.02) and
        equalF16(f16_view_mul.data, &.{ 10, 40, 90, 160 }, 0.02) and
        equalF16(f16_view_div.data, &.{ 10, 10, 10, 10 }, 0.02);
    const view16_scalar_ok = equalF16(f16_view_scalar_add.data, &.{ 3, 4, 5, 6 }, 0.02) and
        equalF16(f16_view_scalar_sub.data, &.{ -1, 0, 1, 2 }, 0.02) and
        equalF16(f16_view_scalar_mul.data, &.{ 2, 4, 6, 8 }, 0.02) and
        equalF16(f16_view_scalar_div.data, &.{ 1, 2, 3, 4 }, 0.02);
    const view_bf16_ok = equalBF16(bf16_view_add.data, &.{ 11, 22, 33, 44 }, 0.125) and
        equalBF16(bf16_view_sub.data, &.{ 9, 18, 27, 36 }, 0.125) and
        equalBF16(bf16_view_mul.data, &.{ 10, 40, 90, 160 }, 0.125) and
        equalBF16(bf16_view_div.data, &.{ 10, 10, 10, 10 }, 0.125);
    const view_bf16_scalar_ok = equalBF16(bf16_view_scalar_add.data, &.{ 3, 4, 5, 6 }, 0.125) and
        equalBF16(bf16_view_scalar_sub.data, &.{ -1, 0, 1, 2 }, 0.125) and
        equalBF16(bf16_view_scalar_mul.data, &.{ 2, 4, 6, 8 }, 0.125) and
        equalBF16(bf16_view_scalar_div.data, &.{ 1, 2, 3, 4 }, 0.125);
    const view_unary_ok = equalF32(view_neg.data, &.{ -1, -2, -3, -4 }) and
        equalF32(view_abs.data, &.{ 1, 2, 3, 4 }) and
        equalF32(view_square.data, &.{ 1, 4, 9, 16 }) and
        closeF32(view_reciprocal.data, &.{ 1, 0.5, 0.33333334, 0.25 }, 1e-6) and
        equalF64(f64_view_neg.data, &.{ -1, -2, -3, -4 }) and
        equalF64(f64_view_square.data, &.{ 1, 4, 9, 16 }) and
        closeF64(f64_view_reciprocal.data, &.{ 1, 0.5, 1.0 / 3.0, 0.25 }, 1e-12) and
        equalF16(f16_view_neg.data, &.{ -1, -2, -3, -4 }, 0.02) and
        equalF16(f16_view_square.data, &.{ 1, 4, 9, 16 }, 0.02) and
        equalF16(f16_view_reciprocal.data, &.{ 1, 0.5, 0.33333334, 0.25 }, 0.001) and
        equalBF16(bf16_view_neg.data, &.{ -1, -2, -3, -4 }, 0.125) and
        equalBF16(bf16_view_square.data, &.{ 1, 4, 9, 16 }, 0.125) and
        equalBF16(bf16_view_reciprocal.data, &.{ 1, 0.5, 0.33333334, 0.25 }, 0.01);
    const default_policy_ok = default_cpu_policy == .prefer_axiom_cpu and
        default_cuda_policy == .prefer_cuda and
        default_mps_policy == .prefer_axiom_cpu;
    const dynamic_execution_ok = default_cuda_execution_target == .cuda and
        default_mps_execution_target == .cpu and
        default_mps_fallback_execution_target == .cpu and
        cpu_device_target == .cpu and
        cuda_device_target == .cuda and
        mps_device_target == .mps and
        equalF32(eager_cpu.data, &.{ 58, 64, 139, 154 }) and
        equalF32(eager_cuda_default.data, eager_cpu.data) and
        equalF32(eager_mps_default.data, eager_cpu.data);
    const ok = matmul_ok and elementwise_ok and scalar_ok and view_ok and view_scalar_ok and view64_ok and view64_scalar_ok and view16_ok and view16_scalar_ok and view_bf16_ok and view_bf16_scalar_ok and view_unary_ok and default_policy_ok and dynamic_execution_ok;
    var stdout_buffer: [1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout.interface.print(
        "{{\"kind\":\"vectra_axiom_backend_policy_smoke\",\"ok\":{},\"matmul_ok\":{},\"elementwise_ok\":{},\"scalar_ok\":{},\"view_ok\":{},\"view_scalar_ok\":{},\"view64_ok\":{},\"view64_scalar_ok\":{},\"view16_ok\":{},\"view16_scalar_ok\":{},\"view_bf16_ok\":{},\"view_bf16_scalar_ok\":{},\"view_unary_ok\":{},\"default_policy_ok\":{},\"dynamic_execution_ok\":{},\"default_cpu_policy\":\"{s}\",\"default_cuda_policy\":\"{s}\",\"default_mps_policy\":\"{s}\",\"default_cuda_execution_target\":\"{s}\",\"default_mps_execution_target\":\"{s}\",\"cpu_device_target\":\"{s}\",\"cuda_device_target\":\"{s}\",\"mps_device_target\":\"{s}\",\"selected\":\"{s}\",\"matmul64_selected\":\"{s}\",\"elementwise32_selected\":\"{s}\",\"elementwise64_selected\":\"{s}\",\"elementwise64_cuda_selected\":\"{s}\",\"scalar64_selected\":\"{s}\",\"scalar64_cuda_selected\":\"{s}\",\"cpu_enabled\":{},\"cuda_enabled\":{}",
        .{ ok, matmul_ok, elementwise_ok, scalar_ok, view_ok, view_scalar_ok, view64_ok, view64_scalar_ok, view16_ok, view16_scalar_ok, view_bf16_ok, view_bf16_scalar_ok, view_unary_ok, default_policy_ok, dynamic_execution_ok, default_cpu_policy.label(), default_cuda_policy.label(), default_mps_policy.label(), default_cuda_execution_target.label(), default_mps_fallback_execution_target.label(), cpu_device_target.label(), cuda_device_target.label(), mps_device_target.label(), report.selected.label(), matmul64_report.selected.label(), ew32_report.selected.label(), ew64_report.selected.label(), ew64_cuda_report.selected.label(), scalar64_report.selected.label(), scalar64_cuda_report.selected.label(), report.axiom_cpu_enabled, report.axiom_cuda_enabled },
    );
    try stdout.interface.print(
        ",\"fingerprint\":{d},\"elementwise_fingerprint\":{d},\"scalar_fingerprint\":{d},\"view_fingerprint\":{d},\"view_scalar_fingerprint\":{d},\"view64_fingerprint\":{d},\"view64_scalar_fingerprint\":{d},\"view16_fingerprint\":{d},\"view16_scalar_fingerprint\":{d},\"view_bf16_fingerprint\":{d},\"view_bf16_scalar_fingerprint\":{d},\"view_unary_fingerprint\":{d}}}\n",
        .{ report.fingerprint(), ew32_report.fingerprint() ^ ew64_report.fingerprint(), scalar64_report.fingerprint(), hashF32(view_add.data) ^ hashF32(view_sub.data) ^ hashF32(view_mul.data) ^ hashF32(view_div.data), hashF32(view_scalar_add.data) ^ hashF32(view_scalar_sub.data) ^ hashF32(view_scalar_mul.data) ^ hashF32(view_scalar_div.data), hashF64(f64_view_add.data) ^ hashF64(f64_view_sub.data) ^ hashF64(f64_view_mul.data) ^ hashF64(f64_view_div.data), hashF64(f64_view_scalar_add.data) ^ hashF64(f64_view_scalar_sub.data) ^ hashF64(f64_view_scalar_mul.data) ^ hashF64(f64_view_scalar_div.data), hashF16(f16_view_add.data) ^ hashF16(f16_view_sub.data) ^ hashF16(f16_view_mul.data) ^ hashF16(f16_view_div.data), hashF16(f16_view_scalar_add.data) ^ hashF16(f16_view_scalar_sub.data) ^ hashF16(f16_view_scalar_mul.data) ^ hashF16(f16_view_scalar_div.data), hashBF16(bf16_view_add.data) ^ hashBF16(bf16_view_sub.data) ^ hashBF16(bf16_view_mul.data) ^ hashBF16(bf16_view_div.data), hashBF16(bf16_view_scalar_add.data) ^ hashBF16(bf16_view_scalar_sub.data) ^ hashBF16(bf16_view_scalar_mul.data) ^ hashBF16(bf16_view_scalar_div.data), hashF32(view_neg.data) ^ hashF32(view_square.data) ^ hashF32(view_reciprocal.data) ^ hashF64(f64_view_neg.data) ^ hashF64(f64_view_square.data) ^ hashF64(f64_view_reciprocal.data) ^ hashF16(f16_view_neg.data) ^ hashF16(f16_view_square.data) ^ hashF16(f16_view_reciprocal.data) ^ hashBF16(bf16_view_neg.data) ^ hashBF16(bf16_view_square.data) ^ hashBF16(bf16_view_reciprocal.data) },
    );
    try stdout.interface.flush();
    if (!ok) std.process.exit(1);
}

fn equalF32(actual: []const f32, expected: []const f32) bool {
    if (actual.len != expected.len) return false;
    for (actual, expected) |a, e| {
        if (a != e) return false;
    }
    return true;
}

fn closeF32(actual: []const f32, expected: []const f32, tolerance: f32) bool {
    if (actual.len != expected.len) return false;
    for (actual, expected) |a, e| {
        if (@abs(a - e) > tolerance) return false;
    }
    return true;
}

fn hashF32(values: []const f32) u64 {
    var hasher = std.hash.Wyhash.init(0x0abc_beef_5731_ded0);
    var len_bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_bytes, values.len, .little);
    hasher.update(&len_bytes);
    for (values) |value| {
        var bytes: [4]u8 = undefined;
        std.mem.writeInt(u32, &bytes, @bitCast(value), .little);
        hasher.update(&bytes);
    }
    return hasher.final();
}

fn hashF64(values: []const f64) u64 {
    var hasher = std.hash.Wyhash.init(0x0abc_beef_5731_ded1);
    var len_bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_bytes, values.len, .little);
    hasher.update(&len_bytes);
    for (values) |value| {
        var bytes: [8]u8 = undefined;
        std.mem.writeInt(u64, &bytes, @bitCast(value), .little);
        hasher.update(&bytes);
    }
    return hasher.final();
}

fn hashF16(values: []const f16) u64 {
    var hasher = std.hash.Wyhash.init(0x0abc_beef_5731_ded2);
    var len_bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_bytes, values.len, .little);
    hasher.update(&len_bytes);
    for (values) |value| {
        var bytes: [2]u8 = undefined;
        std.mem.writeInt(u16, &bytes, @bitCast(value), .little);
        hasher.update(&bytes);
    }
    return hasher.final();
}

fn hashBF16(values: []const vx.BFloat16) u64 {
    var hasher = std.hash.Wyhash.init(0x0abc_beef_5731_ded3);
    var len_bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_bytes, values.len, .little);
    hasher.update(&len_bytes);
    for (values) |value| {
        var bytes: [2]u8 = undefined;
        std.mem.writeInt(u16, &bytes, value.bits, .little);
        hasher.update(&bytes);
    }
    return hasher.final();
}

fn equalF64(actual: []const f64, expected: []const f64) bool {
    if (actual.len != expected.len) return false;
    for (actual, expected) |a, e| {
        if (a != e) return false;
    }
    return true;
}

fn closeF64(actual: []const f64, expected: []const f64, tolerance: f64) bool {
    if (actual.len != expected.len) return false;
    for (actual, expected) |a, e| {
        if (@abs(a - e) > tolerance) return false;
    }
    return true;
}

fn equalF16(actual: []const f16, expected: []const f32, tolerance: f32) bool {
    if (actual.len != expected.len) return false;
    for (actual, expected) |a, e| {
        if (@abs(@as(f32, @floatCast(a)) - e) > tolerance) return false;
    }
    return true;
}

fn equalBF16(actual: []const vx.BFloat16, expected: []const f32, tolerance: f32) bool {
    if (actual.len != expected.len) return false;
    for (actual, expected) |a, e| {
        if (@abs(a.toF32() - e) > tolerance) return false;
    }
    return true;
}
