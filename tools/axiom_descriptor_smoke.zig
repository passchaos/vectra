const std = @import("std");
const vx = @import("vectra");

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;

    var contiguous = try vx.Array(f32).fromSlice(allocator, &.{ 1, 2, 3, 4 }, &.{4});
    defer contiguous.deinit();
    const contiguous_desc = try vx.axiom_backend.describeArrayMemRef(f32, contiguous, "contiguous");

    var strided_source = try vx.Array(f32).fromSlice(allocator, &.{ 1, 99, 2, 99, 3, 99, 4, 99 }, &.{8});
    defer strided_source.deinit();
    var strided = try strided_source.asStrided(&.{4}, &.{2}, 0);
    defer strided.deinit();
    const strided_desc = try vx.axiom_backend.describeViewMemRef(f32, strided, "strided");

    var scalar_source = try vx.Array(f32).fromSlice(allocator, &.{42}, &.{1});
    defer scalar_source.deinit();
    var broadcast = try scalar_source.asStrided(&.{4}, &.{0}, 0);
    defer broadcast.deinit();
    const broadcast_desc = try vx.axiom_backend.describeViewMemRef(f32, broadcast, "broadcast");

    var matrix = try vx.Array(f64).fromSlice(allocator, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer matrix.deinit();
    const matrix_desc = try vx.axiom_backend.describeArrayMemRef(f64, matrix, "matrix");

    var cuda_desc_ok = false;
    var cuda_space = vx.axiom_backend.TensorMemRefAddressSpace.unknown;
    if (vx.cuda(0).isAvailable()) {
        var cuda_array = try contiguous.cuda(0);
        defer cuda_array.deinit();
        const cuda_desc = try vx.axiom_backend.describeArrayMemRef(f32, cuda_array, "cuda_array");
        cuda_desc_ok = cuda_desc.ok() and cuda_desc.address_space == .cuda and cuda_desc.base_ptr != 0;
        cuda_space = cuda_desc.address_space;
    }

    var text_buffer: [1024]u8 = undefined;
    var text_writer = std.Io.Writer.fixed(&text_buffer);
    try strided_desc.writeText(&text_writer);
    const text_ok = std.mem.indexOf(u8, text_writer.buffered(), "tensor_memref_descriptor ok=true") != null and
        std.mem.indexOf(u8, text_writer.buffered(), "dense_strided=true") != null;

    const ok = contiguous_desc.ok() and
        contiguous_desc.layout.contiguous and
        contiguous_desc.address_space == .host and
        strided_desc.ok() and
        strided_desc.layout.dense_strided and
        strided_desc.required_element_span == 7 and
        broadcast_desc.ok() and
        broadcast_desc.layout.broadcast and
        broadcast_desc.layout.overlapping and
        matrix_desc.ok() and
        matrix_desc.rank == 2 and
        matrix_desc.element_type == .f64 and
        text_ok and
        (!vx.cuda(0).isAvailable() or cuda_desc_ok);

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    try stdout.print(
        "{{\"kind\":\"vectra_axiom_descriptor_smoke\",\"ok\":{},\"contiguous_ok\":{},\"strided_ok\":{},\"broadcast_ok\":{},\"matrix_ok\":{},\"text_ok\":{},\"cuda_available\":{},\"cuda_desc_ok\":{},\"cuda_space\":\"{s}\",\"contiguous_fp\":{d},\"strided_fp\":{d},\"broadcast_fp\":{d},\"matrix_fp\":{d}}}\n",
        .{
            ok,
            contiguous_desc.ok() and contiguous_desc.layout.contiguous,
            strided_desc.ok() and strided_desc.layout.dense_strided,
            broadcast_desc.ok() and broadcast_desc.layout.broadcast and broadcast_desc.layout.overlapping,
            matrix_desc.ok() and matrix_desc.rank == 2,
            text_ok,
            vx.cuda(0).isAvailable(),
            cuda_desc_ok,
            cuda_space.label(),
            contiguous_desc.fingerprint(),
            strided_desc.fingerprint(),
            broadcast_desc.fingerprint(),
            matrix_desc.fingerprint(),
        },
    );
    try stdout.flush();
    if (!ok) std.process.exit(1);
}
