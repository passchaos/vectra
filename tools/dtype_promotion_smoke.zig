//! Smoke gate for representative NumPy/PyTorch-style dtype promotion behavior.
//!
//! This intentionally verifies value behavior, not only public symbol presence.
//! It is a bounded compatibility gate for dense Array computation; autograd is
//! explicitly out of scope for Vectra.

const std = @import("std");
const vx = @import("vectra");

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;

    const matrix_ok = promotionMatrixOk();

    const metadata_ok =
        vx.promoteDType(.bool, .i32) == .i32 and
        vx.promoteDType(.i16, .u16) == .i32 and
        vx.promoteDType(.u32, .u64) == .u64 and
        vx.promoteDType(.isize, .i32) == .i64 and
        vx.promoteDType(.f16, .f32) == .f32 and
        vx.promoteDType(.bf16, .f32) == .f32 and
        vx.promoteDType(.f32, .f64) == .f64 and
        vx.promoteDType(.f32, .c64) == .c64 and
        vx.promoteDType(.f64, .c64) == .c128 and
        vx.promoteType(i16, u16) == i32 and
        vx.promoteType(f16, f32) == f32 and
        vx.promoteType(vx.BFloat16, f32) == f32 and
        vx.promoteType(f32, vx.Complex64) == vx.Complex64;

    var small_signed = try vx.Array(i16).fromSlice(allocator, &.{ -1, 2, 3 }, &.{3});
    defer small_signed.deinit();
    var small_unsigned = try vx.Array(u16).fromSlice(allocator, &.{ 5, 6, 7 }, &.{3});
    defer small_unsigned.deinit();
    var promoted_sum = try small_signed.addPromote(u16, small_unsigned);
    defer promoted_sum.deinit();
    var promoted_sub = try small_signed.subPromote(u16, small_unsigned);
    defer promoted_sub.deinit();
    var promoted_min = try small_signed.minimumPromote(u16, small_unsigned);
    defer promoted_min.deinit();
    const int_promote_ok =
        @TypeOf(promoted_sum).dtype == .i32 and
        eql(i32, promoted_sum.data, &.{ 4, 8, 10 }) and
        eql(i32, promoted_sub.data, &.{ -6, -4, -4 }) and
        eql(i32, promoted_min.data, &.{ -1, 2, 3 });
    var promoted_scalar_sum = try small_signed.addScalarPromote(u16, 5);
    defer promoted_scalar_sum.deinit();
    var promoted_scalar_sub = try small_signed.subScalarPromote(u16, 5);
    defer promoted_scalar_sub.deinit();
    const int_scalar_promote_ok =
        @TypeOf(promoted_scalar_sum).dtype == .i32 and
        eql(i32, promoted_scalar_sum.data, &.{ 4, 7, 8 }) and
        eql(i32, promoted_scalar_sub.data, &.{ -6, -3, -2 });

    var bool_values = try vx.Array(bool).fromSlice(allocator, &.{ true, false, true }, &.{3});
    defer bool_values.deinit();
    var bool_scalar_promote = try bool_values.addScalarPromote(i32, 2);
    defer bool_scalar_promote.deinit();
    const bool_scalar_promote_ok =
        @TypeOf(bool_scalar_promote).dtype == .i32 and
        eql(i32, bool_scalar_promote.data, &.{ 3, 2, 3 });

    var halves = try vx.Array(f16).fromSlice(allocator, &.{ @as(f16, 1.5), @as(f16, 2.0) }, &.{2});
    defer halves.deinit();
    var floats = try vx.Array(f32).fromSlice(allocator, &.{ 1.5, 2.0 }, &.{2});
    defer floats.deinit();
    var promoted_half = try halves.mulPromote(f32, floats);
    defer promoted_half.deinit();
    const half_promote_ok =
        @TypeOf(promoted_half).dtype == .f32 and
        eql(f32, promoted_half.data, &.{ 2.25, 4.0 });
    var promoted_half_scalar = try halves.mulScalarPromote(f32, 2.0);
    defer promoted_half_scalar.deinit();
    const half_scalar_promote_ok =
        @TypeOf(promoted_half_scalar).dtype == .f32 and
        eql(f32, promoted_half_scalar.data, &.{ 3.0, 4.0 });

    var bf16_values = try vx.Array(vx.BFloat16).fromSlice(allocator, &.{ vx.BFloat16.fromF32(1.0), vx.BFloat16.fromF32(-2.0) }, &.{2});
    defer bf16_values.deinit();
    var f32_values = try vx.Array(f32).fromSlice(allocator, &.{ 2.0, -2.0 }, &.{2});
    defer f32_values.deinit();
    var promoted_bf16 = try bf16_values.addPromote(f32, f32_values);
    defer promoted_bf16.deinit();
    const bf16_promote_ok =
        @TypeOf(promoted_bf16).dtype == .f32 and
        approx(promoted_bf16.data[0], 3.0, 1e-3) and
        approx(promoted_bf16.data[1], -4.0, 1e-3);

    var real_values = try vx.Array(f32).fromSlice(allocator, &.{ 5.0, 10.0 }, &.{2});
    defer real_values.deinit();
    var complex_values = try vx.Array(vx.Complex64).fromSlice(allocator, &.{
        .{ .re = 1.0, .im = 2.0 },
        .{ .re = -1.0, .im = -3.0 },
    }, &.{2});
    defer complex_values.deinit();
    var promoted_complex = try real_values.addPromote(vx.Complex64, complex_values);
    defer promoted_complex.deinit();
    const complex_promote_ok =
        @TypeOf(promoted_complex).dtype == .c64 and
        approx(promoted_complex.data[0].re, 6.0, 1e-6) and
        approx(promoted_complex.data[0].im, 2.0, 1e-6) and
        approx(promoted_complex.data[1].re, 9.0, 1e-6) and
        approx(promoted_complex.data[1].im, -3.0, 1e-6);
    var promoted_real_scalar = try real_values.addScalarPromote(f64, 0.5);
    defer promoted_real_scalar.deinit();
    var promoted_real_rsub = try vx.rsubScalarPromote(real_values, f64, 20.0);
    defer promoted_real_rsub.deinit();
    var promoted_real_rdiv = try vx.scalarDivPromote(real_values, f64, 20.0);
    defer promoted_real_rdiv.deinit();
    var promoted_complex_scalar = try real_values.addScalarPromote(vx.Complex64, .{ .re = 1.0, .im = -1.0 });
    defer promoted_complex_scalar.deinit();
    var promoted_complex_rsub = try vx.rsubScalarPromote(real_values, vx.Complex64, .{ .re = 1.0, .im = -1.0 });
    defer promoted_complex_rsub.deinit();
    var top_level_rsub = try vx.rsubScalar(real_values, 20.0);
    defer top_level_rsub.deinit();
    var top_level_rdiv = try vx.scalarDiv(real_values, 20.0);
    defer top_level_rdiv.deinit();
    const scalar_promote_ok =
        int_scalar_promote_ok and
        bool_scalar_promote_ok and
        half_scalar_promote_ok and
        @TypeOf(promoted_real_scalar).dtype == .f64 and
        eql(f64, promoted_real_scalar.data, &.{ 5.5, 10.5 }) and
        @TypeOf(promoted_real_rsub).dtype == .f64 and
        eql(f64, promoted_real_rsub.data, &.{ 15.0, 10.0 }) and
        @TypeOf(promoted_real_rdiv).dtype == .f64 and
        eql(f64, promoted_real_rdiv.data, &.{ 4.0, 2.0 }) and
        eql(f32, top_level_rsub.data, &.{ 15.0, 10.0 }) and
        eql(f32, top_level_rdiv.data, &.{ 4.0, 2.0 }) and
        @TypeOf(promoted_complex_scalar).dtype == .c64 and
        approx(promoted_complex_scalar.data[0].re, 6.0, 1e-6) and
        approx(promoted_complex_scalar.data[0].im, -1.0, 1e-6) and
        approx(promoted_complex_scalar.data[1].re, 11.0, 1e-6) and
        approx(promoted_complex_scalar.data[1].im, -1.0, 1e-6) and
        @TypeOf(promoted_complex_rsub).dtype == .c64 and
        approx(promoted_complex_rsub.data[0].re, -4.0, 1e-6) and
        approx(promoted_complex_rsub.data[0].im, -1.0, 1e-6) and
        approx(promoted_complex_rsub.data[1].re, -9.0, 1e-6) and
        approx(promoted_complex_rsub.data[1].im, -1.0, 1e-6);

    const ok = matrix_ok and metadata_ok and int_promote_ok and scalar_promote_ok and half_promote_ok and bf16_promote_ok and complex_promote_ok;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout.interface.print(
        "{{\"kind\":\"vectra_dtype_promotion_smoke\",\"ok\":{},\"matrix_ok\":{},\"metadata_ok\":{},\"int_promote_ok\":{},\"scalar_promote_ok\":{},\"half_promote_ok\":{},\"bf16_promote_ok\":{},\"complex_promote_ok\":{},\"cases\":{d},\"matrix_cases\":{d}}}\n",
        .{ ok, matrix_ok, metadata_ok, int_promote_ok, scalar_promote_ok, half_promote_ok, bf16_promote_ok, complex_promote_ok, 24, dtype_order.len * dtype_order.len },
    );
    try stdout.interface.flush();
    if (!ok) std.process.exit(1);
}

const dtype_order = [_]vx.DType{
    .bool,
    .i8,
    .u8,
    .i16,
    .u16,
    .i32,
    .u32,
    .i64,
    .u64,
    .isize,
    .usize,
    .f16,
    .bf16,
    .f32,
    .f64,
    .c64,
    .c128,
};

fn promotionMatrixOk() bool {
    for (dtype_order) |lhs| {
        for (dtype_order) |rhs| {
            const expected = expectedPromote(lhs, rhs);
            if (vx.promoteDType(lhs, rhs) != expected) return false;
            if (vx.promoteDType(rhs, lhs) != expected) return false;
            if (vx.resultDType(lhs, rhs) != expected) return false;
        }
    }
    return true;
}

fn expectedPromote(lhs: vx.DType, rhs: vx.DType) vx.DType {
    if (lhs == rhs) return lhs;
    if (isComplex(lhs) or isComplex(rhs)) {
        if (lhs == .c128 or rhs == .c128 or lhs == .f64 or rhs == .f64) return .c128;
        return .c64;
    }
    if (lhs == .bool) return rhs;
    if (rhs == .bool) return lhs;
    if (isFloat(lhs) or isFloat(rhs)) {
        const rank = @max(floatRank(lhs), floatRank(rhs));
        if (rank >= 3) return .f64;
        if (rank == 2) return .f32;
        return .f16;
    }
    var rank = @max(intRank(lhs), intRank(rhs));
    const lhs_unsigned = isInteger(lhs) and !isSigned(lhs);
    const rhs_unsigned = isInteger(rhs) and !isSigned(rhs);
    const mixed_signedness = (isSigned(lhs) and rhs_unsigned) or (isSigned(rhs) and lhs_unsigned);
    if (mixed_signedness) {
        const signed_rank = if (isSigned(lhs)) intRank(lhs) else intRank(rhs);
        const unsigned_rank = if (lhs_unsigned) intRank(lhs) else intRank(rhs);
        if (unsigned_rank >= signed_rank) rank += 1;
        if (rank > 4) return .f64;
    }
    const signed = isSigned(lhs) or isSigned(rhs);
    if (rank <= 1) return if (signed) .i8 else .u8;
    if (rank == 2) return if (signed) .i16 else .u16;
    if (rank == 3) return if (signed) .i32 else .u32;
    return if (signed) .i64 else .u64;
}

fn isComplex(dtype: vx.DType) bool {
    return dtype == .c64 or dtype == .c128;
}

fn isFloat(dtype: vx.DType) bool {
    return switch (dtype) {
        .bf16, .f16, .f32, .f64 => true,
        else => false,
    };
}

fn isInteger(dtype: vx.DType) bool {
    return switch (dtype) {
        .i8, .i16, .i32, .i64, .isize, .u8, .u16, .u32, .u64, .usize => true,
        else => false,
    };
}

fn isSigned(dtype: vx.DType) bool {
    return switch (dtype) {
        .i8, .i16, .i32, .i64, .isize => true,
        else => false,
    };
}

fn floatRank(dtype: vx.DType) usize {
    return switch (dtype) {
        .bf16, .f16 => 1,
        .f32 => 2,
        .f64 => 3,
        else => 0,
    };
}

fn intRank(dtype: vx.DType) usize {
    return switch (dtype) {
        .bool => 0,
        .i8, .u8 => 1,
        .i16, .u16 => 2,
        .i32, .u32 => 3,
        .i64, .u64, .usize, .isize => 4,
        else => 0,
    };
}

fn eql(comptime T: type, actual: []const T, expected: []const T) bool {
    return std.mem.eql(T, actual, expected);
}

fn approx(actual: f32, expected: f32, tolerance: f32) bool {
    return @abs(actual - expected) <= tolerance;
}
