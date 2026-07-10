const std = @import("std");
const vx = @import("vectra");

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;
    var a32 = try vx.Array(f32).fromSlice(allocator, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a32.deinit();
    var b32 = try vx.Array(f32).fromSlice(allocator, &.{ 7, 8, 9, 10, 11, 12 }, &.{ 3, 2 });
    defer b32.deinit();
    var out32 = try a32.matmul(b32);
    defer out32.deinit();
    var c32 = try vx.Array(f32).ones(allocator, &.{ 2, 2 });
    defer c32.deinit();
    var matmul_add32 = try vx.matmulAdd(a32, b32, c32);
    defer matmul_add32.deinit();
    var auto_add32 = try out32.add(c32);
    defer auto_add32.deinit();
    const cpu_fusion_status32_ok = out32.fusionStatus() == .cpu_matmul and auto_add32.fusionStatus() == .none;
    var add32 = try a32.add(a32);
    defer add32.deinit();
    var sub32 = try a32.sub(a32);
    defer sub32.deinit();
    var mul32 = try a32.mul(a32);
    defer mul32.deinit();
    var div32 = try a32.div(a32);
    defer div32.deinit();
    var sqrt32 = try a32.sqrt();
    defer sqrt32.deinit();
    var exp32 = try a32.exp();
    defer exp32.deinit();
    var add_scalar32 = try a32.addScalar(2);
    defer add_scalar32.deinit();
    var sub_scalar32 = try a32.subScalar(2);
    defer sub_scalar32.deinit();
    var mul_scalar32 = try a32.mulScalar(2);
    defer mul_scalar32.deinit();
    var div_scalar32 = try a32.divScalar(2);
    defer div_scalar32.deinit();
    var matvec32_rhs = try vx.Array(f32).fromSlice(allocator, &.{ 1, 2, 3 }, &.{3});
    defer matvec32_rhs.deinit();
    var matvec32 = try a32.matvec(matvec32_rhs);
    defer matvec32.deinit();
    var dot32 = try vx.Array(f32).fromSlice(allocator, &.{ 1, 2, 3 }, &.{3});
    defer dot32.deinit();
    var dot32_out = try dot32.dot(dot32);
    defer dot32_out.deinit();

    var a64 = try vx.Array(f64).fromSlice(allocator, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a64.deinit();
    var b64 = try vx.Array(f64).fromSlice(allocator, &.{ 7, 8, 9, 10, 11, 12 }, &.{ 3, 2 });
    defer b64.deinit();
    var out64 = try a64.matmul(b64);
    defer out64.deinit();
    var c64 = try vx.Array(f64).ones(allocator, &.{ 2, 2 });
    defer c64.deinit();
    var matmul_add64 = try vx.matmulAdd(a64, b64, c64);
    defer matmul_add64.deinit();
    var auto_add64 = try out64.add(c64);
    defer auto_add64.deinit();
    const cpu_fusion_status64_ok = out64.fusionStatus() == .cpu_matmul and auto_add64.fusionStatus() == .none;
    var add64 = try a64.add(a64);
    defer add64.deinit();
    var sub64 = try a64.sub(a64);
    defer sub64.deinit();
    var mul64 = try a64.mul(a64);
    defer mul64.deinit();
    var div64 = try a64.div(a64);
    defer div64.deinit();
    var sqrt64 = try a64.sqrt();
    defer sqrt64.deinit();
    var exp64 = try a64.exp();
    defer exp64.deinit();
    var add_scalar64 = try a64.addScalar(2);
    defer add_scalar64.deinit();
    var sub_scalar64 = try a64.subScalar(2);
    defer sub_scalar64.deinit();
    var mul_scalar64 = try a64.mulScalar(2);
    defer mul_scalar64.deinit();
    var div_scalar64 = try a64.divScalar(2);
    defer div_scalar64.deinit();
    var broadcast_scalar64 = try vx.Array(f64).fromSlice(allocator, &.{2}, &.{1});
    defer broadcast_scalar64.deinit();
    var broadcast_sub64 = try broadcast_scalar64.sub(a64);
    defer broadcast_sub64.deinit();
    var broadcast_div64 = try a64.div(broadcast_scalar64);
    defer broadcast_div64.deinit();
    var matvec64_rhs = try vx.Array(f64).fromSlice(allocator, &.{ 1, 2, 3 }, &.{3});
    defer matvec64_rhs.deinit();
    var matvec64 = try vx.linalg.matvec(f64, a64, matvec64_rhs);
    defer matvec64.deinit();
    var vec64 = try vx.Array(f64).fromSlice(allocator, &.{ 1, 2 }, &.{2});
    defer vec64.deinit();
    var vecmat64 = try vec64.matmul(a64);
    defer vecmat64.deinit();
    const trace64 = try vx.linalg.trace(f64, out64);
    var solve_matrix64 = try vx.Array(f64).fromSlice(allocator, &.{ 4, 7, 2, 6 }, &.{ 2, 2 });
    defer solve_matrix64.deinit();
    var solve_rhs64 = try vx.Array(f64).fromSlice(allocator, &.{ 18, 16 }, &.{2});
    defer solve_rhs64.deinit();
    const det64 = try solve_matrix64.det();
    var inverse64 = try solve_matrix64.inverse();
    defer inverse64.deinit();
    var solve64 = try solve_matrix64.solve(solve_rhs64);
    defer solve64.deinit();
    var spd64 = try vx.Array(f64).fromSlice(allocator, &.{ 25, 15, -5, 15, 18, 0, -5, 0, 11 }, &.{ 3, 3 });
    defer spd64.deinit();
    var cholesky64 = try spd64.cholesky();
    defer cholesky64.deinit();
    var qr_input64 = try vx.Array(f64).fromSlice(allocator, &.{ 1, 1, 1, 2, 1, 3 }, &.{ 3, 2 });
    defer qr_input64.deinit();
    var qr64 = try qr_input64.qr();
    defer qr64.deinit();
    var qr_reconstructed64 = try qr64.q.matmul(qr64.r);
    defer qr_reconstructed64.deinit();
    var lu64 = try solve_matrix64.lu();
    defer lu64.deinit();
    var lu_product64 = try lu64.l.matmul(lu64.u);
    defer lu_product64.deinit();
    var lu_reconstructed64 = try lu64.p.matmul(lu_product64);
    defer lu_reconstructed64.deinit();
    var triangular64 = try vx.Array(f64).fromSlice(allocator, &.{ 2, 0, 0, -1, 3, 0, 4, 2, 5 }, &.{ 3, 3 });
    defer triangular64.deinit();
    var triangular_rhs64 = try vx.Array(f64).fromSlice(allocator, &.{ 2, 2, 25 }, &.{3});
    defer triangular_rhs64.deinit();
    var triangular_solve64 = try triangular64.solveTriangular(triangular_rhs64, .lower, .non_unit);
    defer triangular_solve64.deinit();
    var norm_source64 = try vx.Array(f64).fromSlice(allocator, &.{ 1, -2, 3, -4, 5, -6 }, &.{ 2, 3 });
    defer norm_source64.deinit();
    const fro64 = try norm_source64.matrixNorm(.fro, 1e-12);
    const one_norm64 = try norm_source64.matrixNorm(.one, 1e-12);
    const inf_norm64 = try norm_source64.matrixNorm(.inf, 1e-12);
    var svd64 = try qr_input64.svd(1e-12);
    defer svd64.deinit();
    var svd_sigma64 = try vx.Array(f64).zeros(allocator, &.{ 2, 2 });
    defer svd_sigma64.deinit();
    svd_sigma64.data[0] = svd64.s.data[0];
    svd_sigma64.data[3] = svd64.s.data[1];
    var svd_us64 = try svd64.u.matmul(svd_sigma64);
    defer svd_us64.deinit();
    var svd_reconstructed64 = try svd_us64.matmul(svd64.vt);
    defer svd_reconstructed64.deinit();
    var singular_values64 = try qr_input64.singularValues(1e-12);
    defer singular_values64.deinit();
    const rank64 = try qr_input64.matrixRank(1e-12);
    const cond64 = try qr_input64.cond(1e-12);
    const two_norm64 = try qr_input64.matrixNorm(.two, 1e-12);
    const nuclear_norm64 = try qr_input64.matrixNorm(.nuclear, 1e-12);
    var pinv64 = try qr_input64.pinv(1e-12);
    defer pinv64.deinit();
    var qr_pinv64 = try qr_input64.matmul(pinv64);
    defer qr_pinv64.deinit();
    var qr_pinv_qr64 = try qr_pinv64.matmul(qr_input64);
    defer qr_pinv_qr64.deinit();
    var lstsq_rhs64 = try vx.Array(f64).fromSlice(allocator, &.{ 1, 2, 2 }, &.{3});
    defer lstsq_rhs64.deinit();
    var lstsq64 = try qr_input64.lstsq(lstsq_rhs64, 1e-12);
    defer lstsq64.deinit();

    const matmul_ok = out32.data[0] == 58 and out32.data[3] == 154 and
        matmul_add32.data[0] == 59 and matmul_add32.data[3] == 155 and
        auto_add32.data[0] == 59 and auto_add32.data[3] == 155 and
        out64.data[0] == 58 and out64.data[3] == 154 and
        matmul_add64.data[0] == 59 and matmul_add64.data[3] == 155 and
        auto_add64.data[0] == 59 and auto_add64.data[3] == 155 and
        cpu_fusion_status32_ok and cpu_fusion_status64_ok;
    const elementwise_ok = equalF32(add32.data, &.{ 2, 4, 6, 8, 10, 12 }) and
        equalF32(sub32.data, &.{ 0, 0, 0, 0, 0, 0 }) and
        equalF32(mul32.data, &.{ 1, 4, 9, 16, 25, 36 }) and
        equalF32(div32.data, &.{ 1, 1, 1, 1, 1, 1 }) and
        approxF32(sqrt32.data[3], 2.0, 0.001) and
        approxF32(exp32.data[0], std.math.exp(@as(f32, 1.0)), 0.001) and
        equalF64(add64.data, &.{ 2, 4, 6, 8, 10, 12 }) and
        equalF64(sub64.data, &.{ 0, 0, 0, 0, 0, 0 }) and
        equalF64(mul64.data, &.{ 1, 4, 9, 16, 25, 36 }) and
        equalF64(div64.data, &.{ 1, 1, 1, 1, 1, 1 }) and
        approxF64(sqrt64.data[3], 2.0, 1e-12) and
        approxF64(exp64.data[0], std.math.exp(@as(f64, 1.0)), 1e-12);
    const scalar_ok = equalF32(add_scalar32.data, &.{ 3, 4, 5, 6, 7, 8 }) and
        equalF32(sub_scalar32.data, &.{ -1, 0, 1, 2, 3, 4 }) and
        equalF32(mul_scalar32.data, &.{ 2, 4, 6, 8, 10, 12 }) and
        equalF32(div_scalar32.data, &.{ 0.5, 1, 1.5, 2, 2.5, 3 }) and
        equalF64(add_scalar64.data, &.{ 3, 4, 5, 6, 7, 8 }) and
        equalF64(sub_scalar64.data, &.{ -1, 0, 1, 2, 3, 4 }) and
        equalF64(mul_scalar64.data, &.{ 2, 4, 6, 8, 10, 12 }) and
        equalF64(div_scalar64.data, &.{ 0.5, 1, 1.5, 2, 2.5, 3 }) and
        equalF64(broadcast_sub64.data, &.{ 1, 0, -1, -2, -3, -4 }) and
        equalF64(broadcast_div64.data, &.{ 0.5, 1, 1.5, 2, 2.5, 3 });
    const vector_ok = equalF32(matvec32.data, &.{ 14, 32 }) and
        dot32_out.data[0] == 14 and
        equalF64(matvec64.data, &.{ 14, 32 }) and
        equalF64(vecmat64.data, &.{ 9, 12, 15 }) and
        trace64 == 212;
    const dense_linalg_ok = det64 == 10 and
        approxF64(inverse64.data[0], 0.6, 1e-12) and
        approxF64(inverse64.data[3], 0.4, 1e-12) and
        equalF64Approx(solve64.data, &.{ -0.4, 2.8 }, 1e-12) and
        approxF64(cholesky64.data[0], 5, 1e-12) and
        approxF64(cholesky64.data[3], 3, 1e-12) and
        approxF64(cholesky64.data[6], -1, 1e-12) and
        try qr_reconstructed64.allclose(qr_input64, 1e-10, 1e-10) and
        try lu_reconstructed64.allclose(solve_matrix64, 1e-12, 1e-12) and
        approxF64(triangular_solve64.data[2], 3.8, 1e-12) and
        approxF64(fro64, @sqrt(91.0), 1e-12) and
        approxF64(one_norm64, 9, 1e-12) and
        approxF64(inf_norm64, 15, 1e-12) and
        try svd_reconstructed64.allclose(qr_input64, 1e-10, 1e-10) and
        equalF64Approx(singular_values64.data, svd64.s.data, 1e-12) and
        rank64 == 2 and
        approxF64(cond64, singular_values64.data[0] / singular_values64.data[1], 1e-12) and
        approxF64(two_norm64, singular_values64.data[0], 1e-12) and
        approxF64(nuclear_norm64, singular_values64.data[0] + singular_values64.data[1], 1e-12) and
        try qr_pinv_qr64.allclose(qr_input64, 1e-10, 1e-10) and
        equalF64Approx(lstsq64.data, &.{ 2.0 / 3.0, 0.5 }, 1e-10);
    const ok = matmul_ok and elementwise_ok and scalar_ok and vector_ok and dense_linalg_ok;
    var stdout_buffer: [2048]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout.interface.print("{{\"kind\":\"vectra_axiom_cpu_dispatch_smoke\",\"enabled\":{},\"ok\":{},\"matmul_ok\":{},\"elementwise_ok\":{},\"scalar_ok\":{},\"vector_ok\":{},\"dense_linalg_ok\":{},\"cpu_fusion_status32_ok\":{},\"cpu_fusion_status64_ok\":{},\"f32_0\":{d},\"f64_3\":{d},\"add32_5\":{d},\"sqrt32_3\":{d},\"exp32_0\":{d},\"div64_0\":{d},\"sqrt64_3\":{d},\"exp64_0\":{d},\"sub_scalar64_0\":{d},", .{ vx.axiom_cpu.enabled(), ok, matmul_ok, elementwise_ok, scalar_ok, vector_ok, dense_linalg_ok, cpu_fusion_status32_ok, cpu_fusion_status64_ok, out32.data[0], out64.data[3], add32.data[5], sqrt32.data[3], exp32.data[0], div64.data[0], sqrt64.data[3], exp64.data[0], sub_scalar64.data[0] });
    try stdout.interface.print("\"matvec32_1\":{d},\"vecmat64_2\":{d},\"trace64\":{d},\"det64\":{d},\"solve64_1\":{d},\"chol64_0\":{d},\"qr64_r00\":{d},\"lu64_u00\":{d},\"tri64_2\":{d},\"fro64\":{d},\"svd64_s0\":{d},\"singular64_s0\":{d},\"rank64\":{},\"cond64\":{d},\"two_norm64\":{d},\"nuclear64\":{d},\"pinv64_0\":{d},\"lstsq64_0\":{d}}}\n", .{ matvec32.data[1], vecmat64.data[2], trace64, det64, solve64.data[1], cholesky64.data[0], qr64.r.data[0], lu64.u.data[0], triangular_solve64.data[2], fro64, svd64.s.data[0], singular_values64.data[0], rank64, cond64, two_norm64, nuclear_norm64, pinv64.data[0], lstsq64.data[0] });
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

fn equalF64(actual: []const f64, expected: []const f64) bool {
    if (actual.len != expected.len) return false;
    for (actual, expected) |a, e| {
        if (a != e) return false;
    }
    return true;
}

fn approxF64(actual: f64, expected: f64, tolerance: f64) bool {
    return @abs(actual - expected) <= tolerance;
}

fn approxF32(actual: f32, expected: f32, tolerance: f32) bool {
    return @abs(actual - expected) <= tolerance;
}

fn equalF64Approx(actual: []const f64, expected: []const f64, tolerance: f64) bool {
    if (actual.len != expected.len) return false;
    for (actual, expected) |a, e| {
        if (!approxF64(a, e, tolerance)) return false;
    }
    return true;
}
