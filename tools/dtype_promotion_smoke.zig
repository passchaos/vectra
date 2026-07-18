//! Smoke gate for representative NumPy/PyTorch-style dtype promotion behavior.
//!
//! This intentionally verifies value behavior, not only public symbol presence.
//! It is a bounded compatibility gate for dense Array computation; autograd is
//! explicitly out of scope for Vectra.

const std = @import("std");
const vx = @import("vectra");

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;

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

    var halves = try vx.Array(f16).fromSlice(allocator, &.{ @as(f16, 1.5), @as(f16, 2.0) }, &.{2});
    defer halves.deinit();
    var floats = try vx.Array(f32).fromSlice(allocator, &.{ 1.5, 2.0 }, &.{2});
    defer floats.deinit();
    var promoted_half = try halves.mulPromote(f32, floats);
    defer promoted_half.deinit();
    const half_promote_ok =
        @TypeOf(promoted_half).dtype == .f32 and
        eql(f32, promoted_half.data, &.{ 2.25, 4.0 });

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

    const ok = metadata_ok and int_promote_ok and half_promote_ok and bf16_promote_ok and complex_promote_ok;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout.interface.print(
        "{{\"kind\":\"vectra_dtype_promotion_smoke\",\"ok\":{},\"metadata_ok\":{},\"int_promote_ok\":{},\"half_promote_ok\":{},\"bf16_promote_ok\":{},\"complex_promote_ok\":{},\"cases\":{d}}}\n",
        .{ ok, metadata_ok, int_promote_ok, half_promote_ok, bf16_promote_ok, complex_promote_ok, 13 },
    );
    try stdout.interface.flush();
    if (!ok) std.process.exit(1);
}

fn eql(comptime T: type, actual: []const T, expected: []const T) bool {
    return std.mem.eql(T, actual, expected);
}

fn approx(actual: f32, expected: f32, tolerance: f32) bool {
    return @abs(actual - expected) <= tolerance;
}
