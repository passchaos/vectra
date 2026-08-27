//! Device-resident two-dimensional histogram primitives.
//!
//! The first backend is Metal/MPS and intentionally returns only a bounded
//! aggregate grid to the host. Domain-specific rendering policy stays in
//! consumers such as zui-plot.

const std = @import("std");
const array_mod = @import("array.zig");
const axiom_mps = @import("backends/axiom_mps.zig");

pub const BoundsF32 = struct {
    x_min: f32,
    x_max: f32,
    y_min: f32,
    y_max: f32,

    pub fn valid(self: BoundsF32) bool {
        return std.math.isFinite(self.x_min) and std.math.isFinite(self.x_max) and std.math.isFinite(self.y_min) and std.math.isFinite(self.y_max) and self.x_min < self.x_max and self.y_min < self.y_max;
    }
};

pub const CountView = struct {
    counts: []const u32,
    representative_source_indices: []const u32,
    cols: u32,
    rows: u32,
    input_row_count: usize,
    finite_coordinate_count: usize,
    included_row_count: usize,
    omitted_non_finite_coordinate_count: usize,
    out_of_range_count: usize,

    pub fn transferredBytes(self: CountView) usize {
        return self.counts.len * @sizeOf(u32) + self.representative_source_indices.len * @sizeOf(u32) + 2 * @sizeOf(u32);
    }
};

pub const CategoricalCountView = struct {
    category_counts: []const u32,
    representative_source_indices: []const u32,
    cols: u32,
    rows: u32,
    category_count: u32,
    input_row_count: usize,
    finite_coordinate_count: usize,
    included_row_count: usize,
    omitted_non_finite_coordinate_count: usize,
    omitted_unknown_category_count: usize,
    out_of_range_count: usize,

    pub fn transferredBytes(self: CategoricalCountView) usize {
        return self.category_counts.len * @sizeOf(u32) + self.representative_source_indices.len * @sizeOf(u32) + 3 * @sizeOf(u32);
    }
};

/// Reuses its compiled MPS pipeline and fixed-size output buffers across chunks.
/// The returned view borrows this session and is overwritten by the next run.
pub const DeviceHistogram2DCountSession = struct {
    allocator: std.mem.Allocator,
    device: array_mod.Device,
    cols: u32,
    rows: u32,
    counts: []u32,
    representatives: []u32,
    mps: axiom_mps.Histogram2DCountSession,

    pub fn init(allocator: std.mem.Allocator, device: array_mod.Device, cols: u32, rows: u32) array_mod.ArrayError!DeviceHistogram2DCountSession {
        if (!device.isMps() or !device.isAvailable()) return error.InvalidDevice;
        if (cols == 0 or rows == 0) return error.InvalidShape;
        const bin_count = std.math.mul(usize, @as(usize, cols), @as(usize, rows)) catch return error.InvalidShape;
        if (bin_count > std.math.maxInt(u32)) return error.InvalidShape;
        const counts = try allocator.alloc(u32, bin_count);
        errdefer allocator.free(counts);
        const representatives = try allocator.alloc(u32, bin_count);
        errdefer allocator.free(representatives);
        const mps = try axiom_mps.Histogram2DCountSession.init(device, cols, rows);
        return .{ .allocator = allocator, .device = device, .cols = cols, .rows = rows, .counts = counts, .representatives = representatives, .mps = mps };
    }

    pub fn deinit(self: *DeviceHistogram2DCountSession) void {
        self.mps.deinit();
        self.allocator.free(self.counts);
        self.allocator.free(self.representatives);
        self.* = undefined;
    }

    pub fn run(self: *DeviceHistogram2DCountSession, x: array_mod.Array(f32), y: array_mod.Array(f32), bounds: BoundsF32) array_mod.ArrayError!CountView {
        if (!bounds.valid()) return error.InvalidShape;
        if (!x.device.sameDevice(self.device) or !y.device.sameDevice(self.device)) return error.InvalidDevice;
        if (x.shape.len != 1 or y.shape.len != 1 or x.shape[0] != y.shape[0] or !x.isContiguous() or !y.isContiguous()) return error.InvalidShape;
        var diagnostics: [2]u32 = undefined;
        self.mps.run(
            x,
            y,
            .{ bounds.x_min, bounds.x_max, bounds.y_min, bounds.y_max },
            self.counts,
            self.representatives,
            &diagnostics,
        ) catch return error.BackendFailure;
        var included: usize = 0;
        for (self.counts) |count| included +|= count;
        const omitted_non_finite: usize = diagnostics[0];
        const out_of_range: usize = diagnostics[1];
        const finite = x.shape[0] -| omitted_non_finite;
        if (included +| out_of_range != finite) return error.BackendFailure;
        return .{
            .counts = self.counts,
            .representative_source_indices = self.representatives,
            .cols = self.cols,
            .rows = self.rows,
            .input_row_count = x.shape[0],
            .finite_coordinate_count = finite,
            .included_row_count = included,
            .omitted_non_finite_coordinate_count = omitted_non_finite,
            .out_of_range_count = out_of_range,
        };
    }
};

pub const DeviceCategoricalHistogram2DCountSession = struct {
    allocator: std.mem.Allocator,
    device: array_mod.Device,
    cols: u32,
    rows: u32,
    category_count: u32,
    category_counts: []u32,
    representatives: []u32,
    mps: axiom_mps.CategoricalHistogram2DCountSession,

    pub fn init(allocator: std.mem.Allocator, device: array_mod.Device, cols: u32, rows: u32, category_count: u32) array_mod.ArrayError!DeviceCategoricalHistogram2DCountSession {
        if (!device.isMps() or !device.isAvailable()) return error.InvalidDevice;
        if (cols == 0 or rows == 0 or category_count == 0) return error.InvalidShape;
        const bin_count = std.math.mul(usize, @as(usize, cols), @as(usize, rows)) catch return error.InvalidShape;
        const slot_count = std.math.mul(usize, bin_count, @as(usize, category_count)) catch return error.InvalidShape;
        if (slot_count > std.math.maxInt(u32)) return error.InvalidShape;
        const category_counts = try allocator.alloc(u32, slot_count);
        errdefer allocator.free(category_counts);
        const representatives = try allocator.alloc(u32, slot_count);
        errdefer allocator.free(representatives);
        const mps = try axiom_mps.CategoricalHistogram2DCountSession.init(device, cols, rows, category_count);
        return .{ .allocator = allocator, .device = device, .cols = cols, .rows = rows, .category_count = category_count, .category_counts = category_counts, .representatives = representatives, .mps = mps };
    }

    pub fn deinit(self: *DeviceCategoricalHistogram2DCountSession) void {
        self.mps.deinit();
        self.allocator.free(self.category_counts);
        self.allocator.free(self.representatives);
        self.* = undefined;
    }

    pub fn run(self: *DeviceCategoricalHistogram2DCountSession, x: array_mod.Array(f32), y: array_mod.Array(f32), categories: array_mod.Array(i32), bounds: BoundsF32) array_mod.ArrayError!CategoricalCountView {
        if (!bounds.valid()) return error.InvalidShape;
        if (!x.device.sameDevice(self.device) or !y.device.sameDevice(self.device) or !categories.device.sameDevice(self.device)) return error.InvalidDevice;
        if (x.shape.len != 1 or y.shape.len != 1 or categories.shape.len != 1 or x.shape[0] != y.shape[0] or x.shape[0] != categories.shape[0] or !x.isContiguous() or !y.isContiguous() or !categories.isContiguous()) return error.InvalidShape;
        var diagnostics: [3]u32 = undefined;
        self.mps.run(x, y, categories, .{ bounds.x_min, bounds.x_max, bounds.y_min, bounds.y_max }, self.category_counts, self.representatives, &diagnostics) catch return error.BackendFailure;
        var included: usize = 0;
        for (self.category_counts) |count| included +|= count;
        const omitted_non_finite: usize = diagnostics[0];
        const out_of_range: usize = diagnostics[1];
        const omitted_unknown: usize = diagnostics[2];
        const finite = x.shape[0] -| omitted_non_finite;
        if (included +| out_of_range +| omitted_unknown != finite) return error.BackendFailure;
        return .{
            .category_counts = self.category_counts,
            .representative_source_indices = self.representatives,
            .cols = self.cols,
            .rows = self.rows,
            .category_count = self.category_count,
            .input_row_count = x.shape[0],
            .finite_coordinate_count = finite,
            .included_row_count = included,
            .omitted_non_finite_coordinate_count = omitted_non_finite,
            .omitted_unknown_category_count = omitted_unknown,
            .out_of_range_count = out_of_range,
        };
    }
};

test "MPS histogram2d count matches CPU bounds and provenance semantics" {
    const device = array_mod.Device.mps(0);
    if (!device.isAvailable()) return error.SkipZigTest;
    const x_values = [_]f32{ 0.0, 0.24, 0.5, 1.0, -0.1, std.math.nan(f32), 0.75 };
    const y_values = [_]f32{ 0.0, 0.24, 0.5, 1.0, 0.5, 0.5, std.math.inf(f32) };
    var x = try array_mod.Array(f32).fromSliceOn(std.testing.allocator, &x_values, &.{x_values.len}, device);
    defer x.deinit();
    var y = try array_mod.Array(f32).fromSliceOn(std.testing.allocator, &y_values, &.{y_values.len}, device);
    defer y.deinit();
    var session = try DeviceHistogram2DCountSession.init(std.testing.allocator, device, 2, 2);
    defer session.deinit();
    const first = try session.run(x, y, .{ .x_min = 0, .x_max = 1, .y_min = 0, .y_max = 1 });
    const expected_counts = [_]u32{ 2, 0, 0, 2 };
    const expected_representatives = [_]u32{ 0, std.math.maxInt(u32), std.math.maxInt(u32), 2 };
    try std.testing.expectEqualSlices(u32, &expected_counts, first.counts);
    try std.testing.expectEqualSlices(u32, &expected_representatives, first.representative_source_indices);
    try std.testing.expectEqual(@as(usize, 7), first.input_row_count);
    try std.testing.expectEqual(@as(usize, 5), first.finite_coordinate_count);
    try std.testing.expectEqual(@as(usize, 4), first.included_row_count);
    try std.testing.expectEqual(@as(usize, 2), first.omitted_non_finite_coordinate_count);
    try std.testing.expectEqual(@as(usize, 1), first.out_of_range_count);
    const repeat = try session.run(x, y, .{ .x_min = 0, .x_max = 1, .y_min = 0, .y_max = 1 });
    try std.testing.expectEqualSlices(u32, &expected_counts, repeat.counts);
    try std.testing.expectEqualSlices(u32, &expected_representatives, repeat.representative_source_indices);
}

test "device histogram2d session rejects CPU placement" {
    try std.testing.expectError(error.InvalidDevice, DeviceHistogram2DCountSession.init(std.testing.allocator, .cpu, 2, 2));
}

test "MPS categorical histogram2d matches count and unknown-category semantics" {
    const device = array_mod.Device.mps(0);
    if (!device.isAvailable()) return error.SkipZigTest;
    const x_values = [_]f32{ 0.0, 0.24, 0.5, 1.0, -0.1, std.math.nan(f32), 0.75, 0.75 };
    const y_values = [_]f32{ 0.0, 0.24, 0.5, 1.0, 0.5, 0.5, std.math.inf(f32), 0.75 };
    const categories_values = [_]i32{ 0, 1, 1, 0, 0, 1, -1, -1 };
    var x = try array_mod.Array(f32).fromSliceOn(std.testing.allocator, &x_values, &.{x_values.len}, device);
    defer x.deinit();
    var y = try array_mod.Array(f32).fromSliceOn(std.testing.allocator, &y_values, &.{y_values.len}, device);
    defer y.deinit();
    var categories = try array_mod.Array(i32).fromSliceOn(std.testing.allocator, &categories_values, &.{categories_values.len}, device);
    defer categories.deinit();
    var session = try DeviceCategoricalHistogram2DCountSession.init(std.testing.allocator, device, 2, 2, 2);
    defer session.deinit();
    const result = try session.run(x, y, categories, .{ .x_min = 0, .x_max = 1, .y_min = 0, .y_max = 1 });
    try std.testing.expectEqualSlices(u32, &.{ 1, 1, 0, 0, 0, 0, 1, 1 }, result.category_counts);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, std.math.maxInt(u32), std.math.maxInt(u32), std.math.maxInt(u32), std.math.maxInt(u32), 3, 2 }, result.representative_source_indices);
    try std.testing.expectEqual(@as(usize, 8), result.input_row_count);
    try std.testing.expectEqual(@as(usize, 6), result.finite_coordinate_count);
    try std.testing.expectEqual(@as(usize, 4), result.included_row_count);
    try std.testing.expectEqual(@as(usize, 2), result.omitted_non_finite_coordinate_count);
    try std.testing.expectEqual(@as(usize, 1), result.omitted_unknown_category_count);
    try std.testing.expectEqual(@as(usize, 1), result.out_of_range_count);
}
