const std = @import("std");
const Io = std.Io;
const vx = @import("vectra");

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const io = init.io;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const out = &stdout_file_writer.interface;

    var a = try vx.tensor(f64, allocator, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a.deinit();
    var b = try vx.ones(f64, allocator, &.{3});
    defer b.deinit();
    var c = try a.add(b);
    defer c.deinit();
    try c.print(out);
    try out.print("\n", .{});

    var df = try vx.DataFrame.init(allocator, &.{
        .{ .name = "city", .data = .{ .string = &.{ "hz", "bj", "hz" } } },
        .{ .name = "sales", .data = .{ .f64 = &.{ 2.0, 3.0, 5.0 } } },
    });
    defer df.deinit();
    var grouped = try df.groupBySum("city", "sales");
    defer grouped.deinit();
    try grouped.print(out);

    try out.flush();
}
