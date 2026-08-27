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
    omitted_null_row_count: usize,
    finite_coordinate_count: usize,
    included_row_count: usize,
    omitted_non_finite_coordinate_count: usize,
    out_of_range_count: usize,
    nullable: bool = false,

    pub fn transferredBytes(self: CountView) usize {
        const diagnostics: usize = if (self.nullable) 3 else 2;
        return self.counts.len * @sizeOf(u32) + self.representative_source_indices.len * @sizeOf(u32) + diagnostics * @sizeOf(u32);
    }
};

pub const ExtremaOp = enum { min, max };

pub const ExtremaView = struct {
    counts: []const u32,
    values: []const f32,
    representative_source_indices: []const u32,
    cols: u32,
    rows: u32,
    op: ExtremaOp,
    input_row_count: usize,
    omitted_null_row_count: usize,
    finite_coordinate_count: usize,
    included_row_count: usize,
    omitted_non_finite_coordinate_count: usize,
    omitted_non_finite_value_count: usize,
    out_of_range_count: usize,
    nullable: bool = false,

    pub fn transferredBytes(self: ExtremaView) usize {
        const diagnostics: usize = if (self.nullable) 4 else 3;
        return self.counts.len * @sizeOf(u32) + self.values.len * @sizeOf(f32) + self.representative_source_indices.len * @sizeOf(u32) + diagnostics * @sizeOf(u32);
    }
};

pub const SumView = struct {
    counts: []const u32,
    sums: []const f32,
    representative_source_indices: []const u32,
    cols: u32,
    rows: u32,
    input_row_count: usize,
    omitted_null_row_count: usize,
    finite_coordinate_count: usize,
    included_row_count: usize,
    omitted_non_finite_coordinate_count: usize,
    omitted_non_finite_value_count: usize,
    negative_value_count: usize,
    out_of_range_count: usize,
    nullable: bool = false,

    pub fn transferredBytes(self: SumView) usize {
        const diagnostics: usize = if (self.nullable) 5 else 4;
        return self.counts.len * @sizeOf(u32) + self.sums.len * @sizeOf(f32) + self.representative_source_indices.len * @sizeOf(u32) + diagnostics * @sizeOf(u32);
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
            .omitted_null_row_count = 0,
            .finite_coordinate_count = finite,
            .included_row_count = included,
            .omitted_non_finite_coordinate_count = omitted_non_finite,
            .out_of_range_count = out_of_range,
        };
    }

    pub fn runNullable(self: *DeviceHistogram2DCountSession, x: array_mod.Array(f32), y: array_mod.Array(f32), x_validity: ?array_mod.Array(bool), y_validity: ?array_mod.Array(bool), bounds: BoundsF32) array_mod.ArrayError!CountView {
        try validateNullableInputs(self.device, x, y, null, x_validity, y_validity, null);
        if (!bounds.valid()) return error.InvalidShape;
        var diagnostics: [3]u32 = undefined;
        self.mps.runMasked(x, y, x_validity, y_validity, .{ bounds.x_min, bounds.x_max, bounds.y_min, bounds.y_max }, self.counts, self.representatives, &diagnostics) catch return error.BackendFailure;
        var included: usize = 0;
        for (self.counts) |count| included +|= count;
        const omitted_null: usize = diagnostics[0];
        const omitted_non_finite: usize = diagnostics[1];
        const out_of_range: usize = diagnostics[2];
        const materialized = x.shape[0] -| omitted_null;
        const finite = materialized -| omitted_non_finite;
        if (included +| out_of_range != finite) return error.BackendFailure;
        return .{
            .counts = self.counts,
            .representative_source_indices = self.representatives,
            .cols = self.cols,
            .rows = self.rows,
            .input_row_count = x.shape[0],
            .omitted_null_row_count = omitted_null,
            .finite_coordinate_count = finite,
            .included_row_count = included,
            .omitted_non_finite_coordinate_count = omitted_non_finite,
            .out_of_range_count = out_of_range,
            .nullable = true,
        };
    }
};

/// Reuses a precise Metal pipeline to compute deterministic f32 min/max grids.
/// The returned view borrows this session and is overwritten by the next run.
pub const DeviceHistogram2DExtremaSession = struct {
    allocator: std.mem.Allocator,
    device: array_mod.Device,
    cols: u32,
    rows: u32,
    op: ExtremaOp,
    counts: []u32,
    values: []f32,
    representatives: []u32,
    mps: axiom_mps.Histogram2DExtremaSession,

    pub fn init(allocator: std.mem.Allocator, device: array_mod.Device, cols: u32, rows: u32, op: ExtremaOp) array_mod.ArrayError!DeviceHistogram2DExtremaSession {
        if (!device.isMps() or !device.isAvailable()) return error.InvalidDevice;
        if (cols == 0 or rows == 0) return error.InvalidShape;
        const bin_count = std.math.mul(usize, @as(usize, cols), @as(usize, rows)) catch return error.InvalidShape;
        if (bin_count > std.math.maxInt(u32)) return error.InvalidShape;
        const counts = try allocator.alloc(u32, bin_count);
        errdefer allocator.free(counts);
        const values = try allocator.alloc(f32, bin_count);
        errdefer allocator.free(values);
        const representatives = try allocator.alloc(u32, bin_count);
        errdefer allocator.free(representatives);
        const mps_op: axiom_mps.Histogram2DExtremaOp = switch (op) {
            .min => .min,
            .max => .max,
        };
        const mps = try axiom_mps.Histogram2DExtremaSession.init(device, cols, rows, mps_op);
        return .{ .allocator = allocator, .device = device, .cols = cols, .rows = rows, .op = op, .counts = counts, .values = values, .representatives = representatives, .mps = mps };
    }

    pub fn deinit(self: *DeviceHistogram2DExtremaSession) void {
        self.mps.deinit();
        self.allocator.free(self.counts);
        self.allocator.free(self.values);
        self.allocator.free(self.representatives);
        self.* = undefined;
    }

    pub fn run(self: *DeviceHistogram2DExtremaSession, x: array_mod.Array(f32), y: array_mod.Array(f32), values: array_mod.Array(f32), bounds: BoundsF32) array_mod.ArrayError!ExtremaView {
        if (!bounds.valid()) return error.InvalidShape;
        if (!x.device.sameDevice(self.device) or !y.device.sameDevice(self.device) or !values.device.sameDevice(self.device)) return error.InvalidDevice;
        if (x.shape.len != 1 or y.shape.len != 1 or values.shape.len != 1 or x.shape[0] != y.shape[0] or x.shape[0] != values.shape[0] or !x.isContiguous() or !y.isContiguous() or !values.isContiguous()) return error.InvalidShape;
        var diagnostics: [3]u32 = undefined;
        self.mps.run(x, y, values, .{ bounds.x_min, bounds.x_max, bounds.y_min, bounds.y_max }, self.counts, self.values, self.representatives, &diagnostics) catch return error.BackendFailure;
        var included: usize = 0;
        for (self.counts) |count| included +|= count;
        const omitted_non_finite_coordinate: usize = diagnostics[0];
        const out_of_range: usize = diagnostics[1];
        const omitted_non_finite_value: usize = diagnostics[2];
        const finite_coordinate = x.shape[0] -| omitted_non_finite_coordinate;
        if (included +| out_of_range +| omitted_non_finite_value != finite_coordinate) return error.BackendFailure;
        return .{
            .counts = self.counts,
            .values = self.values,
            .representative_source_indices = self.representatives,
            .cols = self.cols,
            .rows = self.rows,
            .op = self.op,
            .input_row_count = x.shape[0],
            .omitted_null_row_count = 0,
            .finite_coordinate_count = finite_coordinate,
            .included_row_count = included,
            .omitted_non_finite_coordinate_count = omitted_non_finite_coordinate,
            .omitted_non_finite_value_count = omitted_non_finite_value,
            .out_of_range_count = out_of_range,
        };
    }

    pub fn runNullable(self: *DeviceHistogram2DExtremaSession, x: array_mod.Array(f32), y: array_mod.Array(f32), values: array_mod.Array(f32), x_validity: ?array_mod.Array(bool), y_validity: ?array_mod.Array(bool), value_validity: ?array_mod.Array(bool), bounds: BoundsF32) array_mod.ArrayError!ExtremaView {
        try validateNullableInputs(self.device, x, y, values, x_validity, y_validity, value_validity);
        if (!bounds.valid()) return error.InvalidShape;
        var diagnostics: [4]u32 = undefined;
        self.mps.runMasked(x, y, values, x_validity, y_validity, value_validity, .{ bounds.x_min, bounds.x_max, bounds.y_min, bounds.y_max }, self.counts, self.values, self.representatives, &diagnostics) catch return error.BackendFailure;
        var included: usize = 0;
        for (self.counts) |count| included +|= count;
        const omitted_null: usize = diagnostics[0];
        const omitted_non_finite_coordinate: usize = diagnostics[1];
        const out_of_range: usize = diagnostics[2];
        const omitted_non_finite_value: usize = diagnostics[3];
        const materialized = x.shape[0] -| omitted_null;
        const finite_coordinate = materialized -| omitted_non_finite_coordinate;
        if (included +| out_of_range +| omitted_non_finite_value != finite_coordinate) return error.BackendFailure;
        return .{
            .counts = self.counts,
            .values = self.values,
            .representative_source_indices = self.representatives,
            .cols = self.cols,
            .rows = self.rows,
            .op = self.op,
            .input_row_count = x.shape[0],
            .omitted_null_row_count = omitted_null,
            .finite_coordinate_count = finite_coordinate,
            .included_row_count = included,
            .omitted_non_finite_coordinate_count = omitted_non_finite_coordinate,
            .omitted_non_finite_value_count = omitted_non_finite_value,
            .out_of_range_count = out_of_range,
            .nullable = true,
        };
    }
};

/// Reuses one MPS pipeline for weighted per-bin sums. Counts and provenance
/// are exact; sums use the backend's parallel f32 accumulation order.
pub const DeviceHistogram2DSumSession = struct {
    allocator: std.mem.Allocator,
    device: array_mod.Device,
    cols: u32,
    rows: u32,
    counts: []u32,
    sums: []f32,
    representatives: []u32,
    mps: axiom_mps.Histogram2DSumSession,

    pub fn init(allocator: std.mem.Allocator, device: array_mod.Device, cols: u32, rows: u32) array_mod.ArrayError!DeviceHistogram2DSumSession {
        if (!device.isMps() or !device.isAvailable()) return error.InvalidDevice;
        if (cols == 0 or rows == 0) return error.InvalidShape;
        const bin_count = std.math.mul(usize, @as(usize, cols), @as(usize, rows)) catch return error.InvalidShape;
        if (bin_count > std.math.maxInt(u32)) return error.InvalidShape;
        const counts = try allocator.alloc(u32, bin_count);
        errdefer allocator.free(counts);
        const sums = try allocator.alloc(f32, bin_count);
        errdefer allocator.free(sums);
        const representatives = try allocator.alloc(u32, bin_count);
        errdefer allocator.free(representatives);
        const mps = try axiom_mps.Histogram2DSumSession.init(device, cols, rows);
        return .{ .allocator = allocator, .device = device, .cols = cols, .rows = rows, .counts = counts, .sums = sums, .representatives = representatives, .mps = mps };
    }

    pub fn deinit(self: *DeviceHistogram2DSumSession) void {
        self.mps.deinit();
        self.allocator.free(self.counts);
        self.allocator.free(self.sums);
        self.allocator.free(self.representatives);
        self.* = undefined;
    }

    pub fn run(self: *DeviceHistogram2DSumSession, x: array_mod.Array(f32), y: array_mod.Array(f32), values: array_mod.Array(f32), bounds: BoundsF32) array_mod.ArrayError!SumView {
        try validateNullableInputs(self.device, x, y, values, null, null, null);
        if (!bounds.valid()) return error.InvalidShape;
        var diagnostics: [4]u32 = undefined;
        try self.mps.run(x, y, values, .{ bounds.x_min, bounds.x_max, bounds.y_min, bounds.y_max }, self.counts, self.sums, self.representatives, &diagnostics);
        return makeSumView(self, x.shape[0], 0, diagnostics[0], diagnostics[1], diagnostics[2], diagnostics[3], false);
    }

    pub fn runNullable(self: *DeviceHistogram2DSumSession, x: array_mod.Array(f32), y: array_mod.Array(f32), values: array_mod.Array(f32), x_validity: ?array_mod.Array(bool), y_validity: ?array_mod.Array(bool), value_validity: ?array_mod.Array(bool), bounds: BoundsF32) array_mod.ArrayError!SumView {
        try validateNullableInputs(self.device, x, y, values, x_validity, y_validity, value_validity);
        if (!bounds.valid()) return error.InvalidShape;
        var diagnostics: [5]u32 = undefined;
        try self.mps.runMasked(x, y, values, x_validity, y_validity, value_validity, .{ bounds.x_min, bounds.x_max, bounds.y_min, bounds.y_max }, self.counts, self.sums, self.representatives, &diagnostics);
        return makeSumView(self, x.shape[0], diagnostics[0], diagnostics[1], diagnostics[2], diagnostics[3], diagnostics[4], true);
    }

    fn makeSumView(self: *DeviceHistogram2DSumSession, input: usize, omitted_null: usize, omitted_non_finite_coordinate: usize, out_of_range: usize, omitted_non_finite_value: usize, negative_value_count: usize, nullable: bool) array_mod.ArrayError!SumView {
        var included: usize = 0;
        for (self.counts) |count| included +|= count;
        const materialized = input -| omitted_null;
        const finite_coordinate = materialized -| omitted_non_finite_coordinate;
        if (included +| out_of_range +| omitted_non_finite_value != finite_coordinate) return error.BackendFailure;
        if (negative_value_count > included) return error.BackendFailure;
        return .{ .counts = self.counts, .sums = self.sums, .representative_source_indices = self.representatives, .cols = self.cols, .rows = self.rows, .input_row_count = input, .omitted_null_row_count = omitted_null, .finite_coordinate_count = finite_coordinate, .included_row_count = included, .omitted_non_finite_coordinate_count = omitted_non_finite_coordinate, .omitted_non_finite_value_count = omitted_non_finite_value, .negative_value_count = negative_value_count, .out_of_range_count = out_of_range, .nullable = nullable };
    }
};

fn validateNullableInputs(device: array_mod.Device, x: array_mod.Array(f32), y: array_mod.Array(f32), values: ?array_mod.Array(f32), x_validity: ?array_mod.Array(bool), y_validity: ?array_mod.Array(bool), value_validity: ?array_mod.Array(bool)) array_mod.ArrayError!void {
    if (!x.device.sameDevice(device) or !y.device.sameDevice(device)) return error.InvalidDevice;
    if (x.shape.len != 1 or y.shape.len != 1 or x.shape[0] != y.shape[0] or !x.isContiguous() or !y.isContiguous()) return error.InvalidShape;
    if (values) |column| {
        if (!column.device.sameDevice(device)) return error.InvalidDevice;
        if (column.shape.len != 1 or column.shape[0] != x.shape[0] or !column.isContiguous()) return error.InvalidShape;
    } else if (value_validity != null) return error.InvalidShape;
    inline for (.{ x_validity, y_validity, value_validity }) |maybe_validity| {
        if (maybe_validity) |validity| {
            if (!validity.device.sameDevice(device)) return error.InvalidDevice;
            if (validity.shape.len != 1 or validity.shape[0] != x.shape[0] or !validity.isContiguous()) return error.InvalidShape;
        }
    }
}

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

test "MPS histogram2d extrema matches CPU value and provenance semantics" {
    const device = array_mod.Device.mps(0);
    if (!device.isAvailable()) return error.SkipZigTest;
    const x_values = [_]f32{ 0.0, 0.24, 0.5, 1.0, -0.1, std.math.nan(f32), 0.75, 0.75, 0.75 };
    const y_values = [_]f32{ 0.0, 0.24, 0.5, 1.0, 0.5, 0.5, std.math.inf(f32), 0.75, 0.75 };
    const sample_values = [_]f32{ 5, -2, 9, 4, 1, 2, 3, std.math.nan(f32), -7 };
    var x = try array_mod.Array(f32).fromSliceOn(std.testing.allocator, &x_values, &.{x_values.len}, device);
    defer x.deinit();
    var y = try array_mod.Array(f32).fromSliceOn(std.testing.allocator, &y_values, &.{y_values.len}, device);
    defer y.deinit();
    var values = try array_mod.Array(f32).fromSliceOn(std.testing.allocator, &sample_values, &.{sample_values.len}, device);
    defer values.deinit();
    inline for (.{ ExtremaOp.min, ExtremaOp.max }) |op| {
        var session = try DeviceHistogram2DExtremaSession.init(std.testing.allocator, device, 2, 2, op);
        defer session.deinit();
        const result = try session.run(x, y, values, .{ .x_min = 0, .x_max = 1, .y_min = 0, .y_max = 1 });
        try std.testing.expectEqualSlices(u32, &.{ 2, 0, 0, 3 }, result.counts);
        try std.testing.expectEqualSlices(f32, if (op == .min) &.{ -2, 0, 0, -7 } else &.{ 5, 0, 0, 9 }, result.values);
        try std.testing.expectEqualSlices(u32, &.{ 0, std.math.maxInt(u32), std.math.maxInt(u32), 2 }, result.representative_source_indices);
        try std.testing.expectEqual(@as(usize, 9), result.input_row_count);
        try std.testing.expectEqual(@as(usize, 7), result.finite_coordinate_count);
        try std.testing.expectEqual(@as(usize, 5), result.included_row_count);
        try std.testing.expectEqual(@as(usize, 2), result.omitted_non_finite_coordinate_count);
        try std.testing.expectEqual(@as(usize, 1), result.omitted_non_finite_value_count);
        try std.testing.expectEqual(@as(usize, 1), result.out_of_range_count);
    }
}

test "MPS histogram2d weighted sums preserve counts and provenance" {
    const device = array_mod.Device.mps(0);
    if (!device.isAvailable()) return error.SkipZigTest;
    const x_values = [_]f32{ 0.0, 0.24, 0.5, 1.0, -0.1, std.math.nan(f32), 0.75, 0.75, 0.75 };
    const y_values = [_]f32{ 0.0, 0.24, 0.5, 1.0, 0.5, 0.5, std.math.inf(f32), 0.75, 0.75 };
    const sample_values = [_]f32{ 5, -2, 9, 4, 1, 2, 3, std.math.nan(f32), -7 };
    var x = try array_mod.Array(f32).fromSliceOn(std.testing.allocator, &x_values, &.{x_values.len}, device);
    defer x.deinit();
    var y = try array_mod.Array(f32).fromSliceOn(std.testing.allocator, &y_values, &.{y_values.len}, device);
    defer y.deinit();
    var values = try array_mod.Array(f32).fromSliceOn(std.testing.allocator, &sample_values, &.{sample_values.len}, device);
    defer values.deinit();
    var session = try DeviceHistogram2DSumSession.init(std.testing.allocator, device, 2, 2);
    defer session.deinit();
    const result = try session.run(x, y, values, .{ .x_min = 0, .x_max = 1, .y_min = 0, .y_max = 1 });
    try std.testing.expectEqualSlices(u32, &.{ 2, 0, 0, 3 }, result.counts);
    try std.testing.expectEqualSlices(f32, &.{ 3, 0, 0, 6 }, result.sums);
    try std.testing.expectEqualSlices(u32, &.{ 0, std.math.maxInt(u32), std.math.maxInt(u32), 2 }, result.representative_source_indices);
    try std.testing.expectEqual(@as(usize, 5), result.included_row_count);
    try std.testing.expectEqual(@as(usize, 1), result.omitted_non_finite_value_count);
    try std.testing.expectEqual(@as(usize, 2), result.negative_value_count);
}

test "MPS nullable histogram2d preserves original row provenance" {
    const device = array_mod.Device.mps(0);
    if (!device.isAvailable()) return error.SkipZigTest;
    const x_values = [_]f32{ 0.0, 0.24, 0.5, 1.0, -0.1, std.math.nan(f32), 0.75, 0.75, 0.75 };
    const y_values = [_]f32{ 0.0, 0.24, 0.5, 1.0, 0.5, 0.5, std.math.inf(f32), 0.75, 0.75 };
    const sample_values = [_]f32{ 5, -2, 9, 4, 1, 2, 3, std.math.nan(f32), -7 };
    const x_validity_values = [_]bool{ true, true, true, true, true, false, true, true, true };
    const y_validity_values = [_]bool{ true, true, true, true, true, true, true, true, true };
    const value_validity_values = [_]bool{ true, true, true, true, true, true, true, true, false };
    var x = try array_mod.Array(f32).fromSliceOn(std.testing.allocator, &x_values, &.{x_values.len}, device);
    defer x.deinit();
    var y = try array_mod.Array(f32).fromSliceOn(std.testing.allocator, &y_values, &.{y_values.len}, device);
    defer y.deinit();
    var values = try array_mod.Array(f32).fromSliceOn(std.testing.allocator, &sample_values, &.{sample_values.len}, device);
    defer values.deinit();
    var x_validity = try array_mod.Array(bool).fromSliceOn(std.testing.allocator, &x_validity_values, &.{x_validity_values.len}, device);
    defer x_validity.deinit();
    var y_validity = try array_mod.Array(bool).fromSliceOn(std.testing.allocator, &y_validity_values, &.{y_validity_values.len}, device);
    defer y_validity.deinit();
    var value_validity = try array_mod.Array(bool).fromSliceOn(std.testing.allocator, &value_validity_values, &.{value_validity_values.len}, device);
    defer value_validity.deinit();

    var count_session = try DeviceHistogram2DCountSession.init(std.testing.allocator, device, 2, 2);
    defer count_session.deinit();
    const count_result = try count_session.runNullable(x, y, x_validity, y_validity, .{ .x_min = 0, .x_max = 1, .y_min = 0, .y_max = 1 });
    try std.testing.expectEqualSlices(u32, &.{ 2, 0, 0, 4 }, count_result.counts);
    try std.testing.expectEqualSlices(u32, &.{ 0, std.math.maxInt(u32), std.math.maxInt(u32), 2 }, count_result.representative_source_indices);
    try std.testing.expectEqual(@as(usize, 1), count_result.omitted_null_row_count);
    try std.testing.expectEqual(@as(usize, 1), count_result.omitted_non_finite_coordinate_count);
    try std.testing.expectEqual(@as(usize, 1), count_result.out_of_range_count);

    inline for (.{ ExtremaOp.min, ExtremaOp.max }) |op| {
        var session = try DeviceHistogram2DExtremaSession.init(std.testing.allocator, device, 2, 2, op);
        defer session.deinit();
        const result = try session.runNullable(x, y, values, x_validity, y_validity, value_validity, .{ .x_min = 0, .x_max = 1, .y_min = 0, .y_max = 1 });
        try std.testing.expectEqualSlices(u32, &.{ 2, 0, 0, 2 }, result.counts);
        try std.testing.expectEqualSlices(f32, if (op == .min) &.{ -2, 0, 0, 4 } else &.{ 5, 0, 0, 9 }, result.values);
        try std.testing.expectEqualSlices(u32, &.{ 0, std.math.maxInt(u32), std.math.maxInt(u32), 2 }, result.representative_source_indices);
        try std.testing.expectEqual(@as(usize, 2), result.omitted_null_row_count);
        try std.testing.expectEqual(@as(usize, 1), result.omitted_non_finite_coordinate_count);
        try std.testing.expectEqual(@as(usize, 1), result.omitted_non_finite_value_count);
        try std.testing.expectEqual(@as(usize, 1), result.out_of_range_count);
    }
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
