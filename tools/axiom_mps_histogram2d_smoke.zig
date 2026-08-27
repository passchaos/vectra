const std = @import("std");
const vx = @import("vectra");

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;
    const device = vx.mps(0);
    const available = device.isAvailable();
    var exact = !available;
    var transferred_bytes: usize = 0;
    if (available) {
        const x_values = [_]f32{ 0.0, 0.24, 0.5, 1.0, -0.1, std.math.nan(f32), 0.75 };
        const y_values = [_]f32{ 0.0, 0.24, 0.5, 1.0, 0.5, 0.5, std.math.inf(f32) };
        var x = try vx.Array(f32).fromSliceOn(allocator, &x_values, &.{x_values.len}, device);
        defer x.deinit();
        var y = try vx.Array(f32).fromSliceOn(allocator, &y_values, &.{y_values.len}, device);
        defer y.deinit();
        var session = try vx.DeviceHistogram2DCountSession.init(allocator, device, 2, 2);
        defer session.deinit();
        const result = try session.run(x, y, .{ .x_min = 0, .x_max = 1, .y_min = 0, .y_max = 1 });
        transferred_bytes = result.transferredBytes();
        exact = std.mem.eql(u32, result.counts, &.{ 2, 0, 0, 2 }) and
            std.mem.eql(u32, result.representative_source_indices, &.{ 0, std.math.maxInt(u32), std.math.maxInt(u32), 2 }) and
            result.input_row_count == 7 and result.finite_coordinate_count == 5 and result.included_row_count == 4 and
            result.omitted_non_finite_coordinate_count == 2 and result.out_of_range_count == 1;
    }
    if (!exact) return error.MpsHistogram2DCountMismatch;
    var buffer: [384]u8 = undefined;
    var writer = std.Io.File.stdout().writerStreaming(init.io, &buffer);
    try writer.interface.print("{{\"kind\":\"vectra_axiom_mps_histogram2d_smoke\",\"ok\":true,\"available\":{},\"device_native\":{},\"transferred_bytes\":{}}}\n", .{ available, available, transferred_bytes });
    try writer.interface.flush();
}
