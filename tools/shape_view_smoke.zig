//! Smoke gate for NumPy/PyTorch-style shape/view/broadcast behavior.

const std = @import("std");
const vx = @import("vectra");

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;

    var base = try vx.Array(i32).fromSlice(allocator, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer base.deinit();

    var reshaped = try base.reshape(&.{ 3, 2 });
    defer reshaped.deinit();
    var inferred = try base.reshapeInfer(&.{ -1, 2 });
    defer inferred.deinit();
    var permuted = try base.permute(&.{ 1, 0 });
    defer permuted.deinit();
    var transposed = try base.transpose();
    defer transposed.deinit();

    var row = try vx.Array(i32).fromSlice(allocator, &.{ 10, 20, 30 }, &.{ 1, 3 });
    defer row.deinit();
    var broadcasted = try row.broadcastTo(&.{ 2, 3 });
    defer broadcasted.deinit();
    var expanded = try row.expand(&.{ 2, 3 });
    defer expanded.deinit();
    var expanded_owned = try expanded.toArray();
    defer expanded_owned.deinit();

    var singleton = try base.reshape(&.{ 1, 2, 3 });
    defer singleton.deinit();
    var squeezed = try singleton.squeeze(null);
    defer squeezed.deinit();
    var unsqueezed = try base.unsqueeze(1);
    defer unsqueezed.deinit();

    var repeated = try base.repeat(2, 0);
    defer repeated.deinit();
    var tiled = try base.tile(&.{ 1, 2 });
    defer tiled.deinit();
    var unfolded = try base.unfold(1, 2, 1);
    defer unfolded.deinit();
    var unfolded_owned = try unfolded.toArray();
    defer unfolded_owned.deinit();
    var strided = try base.asStrided(&.{ 2, 2 }, &.{ 3, 1 }, 1);
    defer strided.deinit();
    var strided_owned = try strided.toArray();
    defer strided_owned.deinit();

    const invalid_broadcast_rejected = blk: {
        var bad = row.broadcastTo(&.{ 2, 2 }) catch |err| {
            break :blk err == error.ShapeMismatch;
        };
        bad.deinit();
        break :blk false;
    };

    const ok =
        std.mem.eql(usize, reshaped.shape, &.{ 3, 2 }) and
        eql(i32, reshaped.data, &.{ 1, 2, 3, 4, 5, 6 }) and
        std.mem.eql(usize, inferred.shape, &.{ 3, 2 }) and
        std.mem.eql(usize, permuted.shape, &.{ 3, 2 }) and
        eql(i32, permuted.data, &.{ 1, 4, 2, 5, 3, 6 }) and
        eql(i32, transposed.data, permuted.data) and
        std.mem.eql(usize, broadcasted.shape, &.{ 2, 3 }) and
        eql(i32, broadcasted.data, &.{ 10, 20, 30, 10, 20, 30 }) and
        eql(i32, expanded_owned.data, broadcasted.data) and
        std.mem.eql(usize, squeezed.shape, &.{ 2, 3 }) and
        std.mem.eql(usize, unsqueezed.shape, &.{ 2, 1, 3 }) and
        std.mem.eql(usize, repeated.shape, &.{ 4, 3 }) and
        eql(i32, repeated.data, &.{ 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6 }) and
        std.mem.eql(usize, tiled.shape, &.{ 2, 6 }) and
        eql(i32, tiled.data, &.{ 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6 }) and
        std.mem.eql(usize, unfolded.shape, &.{ 2, 2, 2 }) and
        eql(i32, unfolded_owned.data, &.{ 1, 2, 2, 3, 4, 5, 5, 6 }) and
        std.mem.eql(usize, strided.shape, &.{ 2, 2 }) and
        eql(i32, strided_owned.data, &.{ 2, 3, 5, 6 }) and
        invalid_broadcast_rejected;

    var stdout_buffer: [512]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout.interface.print(
        "{{\"kind\":\"vectra_shape_view_smoke\",\"ok\":{},\"reshape_ok\":{},\"permute_ok\":{},\"broadcast_ok\":{},\"squeeze_ok\":{},\"repeat_tile_ok\":{},\"unfold_ok\":{},\"as_strided_ok\":{},\"invalid_rejected\":{}}}\n",
        .{
            ok,
            std.mem.eql(usize, reshaped.shape, &.{ 3, 2 }) and std.mem.eql(usize, inferred.shape, &.{ 3, 2 }),
            eql(i32, permuted.data, &.{ 1, 4, 2, 5, 3, 6 }),
            eql(i32, broadcasted.data, &.{ 10, 20, 30, 10, 20, 30 }),
            std.mem.eql(usize, squeezed.shape, &.{ 2, 3 }) and std.mem.eql(usize, unsqueezed.shape, &.{ 2, 1, 3 }),
            eql(i32, repeated.data, &.{ 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6 }) and eql(i32, tiled.data, &.{ 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6 }),
            eql(i32, unfolded_owned.data, &.{ 1, 2, 2, 3, 4, 5, 5, 6 }),
            eql(i32, strided_owned.data, &.{ 2, 3, 5, 6 }),
            invalid_broadcast_rejected,
        },
    );
    try stdout.interface.flush();
    if (!ok) std.process.exit(1);
}

fn eql(comptime T: type, actual: []const T, expected: []const T) bool {
    return std.mem.eql(T, actual, expected);
}
