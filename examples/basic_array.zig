//! Basic Vectra Array usage with broadcasting, reductions, and dtype metadata.
//!
//! Run with:
//!   zig build example-basic-array

const std = @import("std");
const vx = @import("vectra");

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;
    const np = vx.withAllocator(allocator);

    var x = try np.array(f32, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer x.deinit();

    var bias = try np.array(f32, &.{ 10, 20, 30 }, &.{3});
    defer bias.deinit();

    var weights = try np.array(f32, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 3, 2 });
    defer weights.deinit();

    var output_bias = try np.array(f32, &.{ 1, 10, 100, 1000 }, &.{ 2, 2 });
    defer output_bias.deinit();

    const broadcast_shape = try x.broadcastShape(bias);
    defer allocator.free(broadcast_shape);

    var y = try x.add(bias);
    defer y.deinit();

    var logits = try x.matmul(weights);
    defer logits.deinit();

    var fused_logits = try x.matmulAdd(weights, output_bias);
    defer fused_logits.deinit();

    var row_sums = try vx.sum(y, 1, false);
    defer row_sums.deinit();

    var scaled = try vx.mulScalar(y, 0.5);
    defer scaled.deinit();

    try expectSlice(f32, y.data, &.{ 11, 22, 33, 14, 25, 36 });
    try expectSlice(f32, logits.data, &.{ 22, 28, 49, 64 });
    try expectSlice(f32, fused_logits.data, &.{ 23, 38, 149, 1064 });
    try expectSlice(f32, row_sums.data, &.{ 66, 75 });
    try expectSlice(f32, scaled.data, &.{ 5.5, 11, 16.5, 7, 12.5, 18 });

    var stdout_buffer: [2048]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout.interface.print(
        \\
        \\{{
        \\  "example": "basic_array",
        \\  "dtype": "{s}",
        \\  "shape": [{d},{d}],
        \\  "bias_shape": [{d}],
        \\  "weights_shape": [{d},{d}],
        \\  "broadcast_shape": [{d},{d}],
        \\  "y": [{d:.1},{d:.1},{d:.1},{d:.1},{d:.1},{d:.1}],
        \\  "logits": [{d:.1},{d:.1},{d:.1},{d:.1}],
        \\  "fused_logits": [{d:.1},{d:.1},{d:.1},{d:.1}],
        \\  "row_sums": [{d:.1},{d:.1}],
        \\  "scaled_first": {d:.1},
        \\  "tensor": "{f}",
        \\  "ok": true
        \\}}
        \\
    , .{
        x.dtypeName(),
        x.shape[0],
        x.shape[1],
        bias.shape[0],
        weights.shape[0],
        weights.shape[1],
        broadcast_shape[0],
        broadcast_shape[1],
        y.data[0],
        y.data[1],
        y.data[2],
        y.data[3],
        y.data[4],
        y.data[5],
        logits.data[0],
        logits.data[1],
        logits.data[2],
        logits.data[3],
        fused_logits.data[0],
        fused_logits.data[1],
        fused_logits.data[2],
        fused_logits.data[3],
        row_sums.data[0],
        row_sums.data[1],
        scaled.data[0],
        row_sums,
    });
    try stdout.interface.flush();
}

fn expectSlice(comptime T: type, actual: []const T, expected: []const T) !void {
    if (actual.len != expected.len) return error.UnexpectedResult;
    for (actual, expected) |a, e| {
        if (a != e) return error.UnexpectedResult;
    }
}
