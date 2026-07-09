const std = @import("std");
const vx = @import("vectra");

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;
    var a = try vx.Array(f32).fromSlice(allocator, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a.deinit();
    var b = try vx.Array(f32).fromSlice(allocator, &.{ 7, 8, 9, 10, 11, 12 }, &.{ 3, 2 });
    defer b.deinit();
    const report = vx.axiom_backend.selectMatmul(f32, .prefer_cuda, a, b);
    var out = try vx.axiom_backend.matmul(f32, .prefer_axiom_cpu, a, b);
    defer out.deinit();
    const ok = report.ok() and out.data[0] == 58 and out.data[3] == 154;
    var stdout_buffer: [1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout.interface.print("{{\"kind\":\"vectra_axiom_backend_policy_smoke\",\"ok\":{},\"selected\":\"{s}\",\"cpu_enabled\":{},\"cuda_enabled\":{},\"fingerprint\":{d}}}\n", .{ ok, report.selected.label(), report.axiom_cpu_enabled, report.axiom_cuda_enabled, report.fingerprint() });
    try stdout.interface.flush();
    if (!ok) std.process.exit(1);
}
