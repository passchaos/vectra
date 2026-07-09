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
    var a64 = try vx.Array(f64).fromSlice(allocator, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a64.deinit();
    var b64 = try vx.Array(f64).fromSlice(allocator, &.{ 7, 8, 9, 10, 11, 12 }, &.{ 3, 2 });
    defer b64.deinit();
    var out64 = try a64.matmul(b64);
    defer out64.deinit();
    const ok = out32.data[0] == 58 and out32.data[3] == 154 and out64.data[0] == 58 and out64.data[3] == 154;
    var stdout_buffer: [1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout.interface.print("{{\"kind\":\"vectra_axiom_cpu_dispatch_smoke\",\"enabled\":{},\"ok\":{},\"f32_0\":{d},\"f64_3\":{d}}}\n", .{ vx.axiom_cpu.enabled(), ok, out32.data[0], out64.data[3] });
    try stdout.interface.flush();
    if (!ok) std.process.exit(1);
}
