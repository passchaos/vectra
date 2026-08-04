const std = @import("std");
const veyra = @import("veyra");
const array_mod = @import("array.zig");

pub const SparseError = array_mod.ArrayError || error{BackendFailure} || std.mem.Allocator.Error;

pub const Triangle = enum { lower, upper };
pub const Diagonal = enum { non_unit, unit };

pub const SparseProfile = struct {
    lower: usize,
    upper: usize,

    pub fn total(self: SparseProfile) SparseError!usize {
        return std.math.add(usize, self.lower, self.upper) catch return error.InvalidShape;
    }

    pub fn meetsBounds(self: SparseProfile, max_lower: usize, max_upper: usize) bool {
        return self.lower <= max_lower and self.upper <= max_upper;
    }

    pub fn totalMeetsBound(self: SparseProfile, max_total: usize) SparseError!bool {
        return (try self.total()) <= max_total;
    }
};

pub const SparseDiffSummary = struct {
    dot: f64,
    max_abs_diff: f64,
    max_rel_diff: f64,
    squared_distance: f64,
    lhs_frobenius_norm: f64,
    rhs_frobenius_norm: f64,

    pub fn frobeniusDistance(self: SparseDiffSummary) f64 {
        return @sqrt(self.squared_distance);
    }

    pub fn relativeFrobeniusDistance(self: SparseDiffSummary) f64 {
        const scale = @max(@as(f64, 1), self.lhs_frobenius_norm + self.rhs_frobenius_norm);
        return self.frobeniusDistance() / scale;
    }

    pub fn maxAbsDiffMeetsBound(self: SparseDiffSummary, max_absolute_diff: f64) SparseError!bool {
        try validateNonNegativeRange(max_absolute_diff, max_absolute_diff);
        return self.max_abs_diff <= max_absolute_diff;
    }

    pub fn maxRelDiffMeetsBound(self: SparseDiffSummary, max_relative_diff: f64) SparseError!bool {
        try validateNonNegativeRange(max_relative_diff, max_relative_diff);
        return self.max_rel_diff <= max_relative_diff;
    }

    pub fn squaredDistanceMeetsBound(self: SparseDiffSummary, max_squared_distance: f64) SparseError!bool {
        try validateNonNegativeRange(max_squared_distance, max_squared_distance);
        return self.squared_distance <= max_squared_distance;
    }

    pub fn frobeniusDistanceMeetsBound(self: SparseDiffSummary, max_distance: f64) SparseError!bool {
        try validateNonNegativeRange(max_distance, max_distance);
        return self.frobeniusDistance() <= max_distance;
    }

    pub fn relativeFrobeniusDistanceMeetsBound(self: SparseDiffSummary, max_relative_distance: f64) SparseError!bool {
        try validateNonNegativeRange(max_relative_distance, max_relative_distance);
        return self.relativeFrobeniusDistance() <= max_relative_distance;
    }

    pub fn meetsBounds(
        self: SparseDiffSummary,
        max_absolute_diff: f64,
        max_relative_diff: f64,
        max_squared_distance: f64,
        max_frobenius_distance: f64,
        max_relative_frobenius_distance: f64,
    ) SparseError!bool {
        return try self.maxAbsDiffMeetsBound(max_absolute_diff) and
            try self.maxRelDiffMeetsBound(max_relative_diff) and
            try self.squaredDistanceMeetsBound(max_squared_distance) and
            try self.frobeniusDistanceMeetsBound(max_frobenius_distance) and
            try self.relativeFrobeniusDistanceMeetsBound(max_relative_frobenius_distance);
    }
};

const SparseScalarComparison = enum { eq, ne, gt, ge, lt, le };
const SparseFinitePredicate = enum { nan, inf, pos_inf, neg_inf, finite, normal };

fn ensureSparseComparisonSupported(comptime T: type, comptime comparison: SparseScalarComparison) void {
    switch (@typeInfo(T)) {
        .bool, .int, .float => {},
        else => @compileError("sparse comparison requires bool, integer, or floating-point values"),
    }
    switch (comparison) {
        .eq, .ne => {},
        .gt, .ge, .lt, .le => ensureNumeric(T),
    }
}

fn sparseCompareValue(comptime T: type, lhs: T, rhs: T, comptime comparison: SparseScalarComparison) bool {
    ensureSparseComparisonSupported(T, comparison);
    return switch (comparison) {
        .eq => lhs == rhs,
        .ne => lhs != rhs,
        .gt => lhs > rhs,
        .ge => lhs >= rhs,
        .lt => lhs < rhs,
        .le => lhs <= rhs,
    };
}

fn sparseCompareScalarValues(
    comptime T: type,
    allocator: std.mem.Allocator,
    values: []const T,
    scalar: T,
    comptime comparison: SparseScalarComparison,
) SparseError![]bool {
    var out = try allocator.alloc(bool, values.len);
    errdefer allocator.free(out);
    for (values, 0..) |value, index| {
        out[index] = sparseCompareValue(T, value, scalar, comparison);
    }
    return out;
}

fn sparseCompareSameStructureValues(
    comptime T: type,
    allocator: std.mem.Allocator,
    lhs_values: []const T,
    rhs_values: []const T,
    comptime comparison: SparseScalarComparison,
) SparseError![]bool {
    if (lhs_values.len != rhs_values.len) return error.ShapeMismatch;
    var out = try allocator.alloc(bool, lhs_values.len);
    errdefer allocator.free(out);
    for (lhs_values, rhs_values, 0..) |lhs, rhs, index| {
        out[index] = sparseCompareValue(T, lhs, rhs, comparison);
    }
    return out;
}

fn sparseFinitePredicateValue(comptime T: type, value: T, comptime predicate: SparseFinitePredicate) bool {
    switch (predicate) {
        .nan, .inf, .pos_inf, .neg_inf, .finite => ensureNumeric(T),
        .normal => ensureFloat(T),
    }
    return switch (predicate) {
        .nan => switch (@typeInfo(T)) {
            .float => std.math.isNan(value),
            .int => false,
            else => @compileError("sparse isNan requires numeric values"),
        },
        .inf => switch (@typeInfo(T)) {
            .float => std.math.isInf(value),
            .int => false,
            else => @compileError("sparse isInf requires numeric values"),
        },
        .pos_inf => switch (@typeInfo(T)) {
            .float => std.math.isPositiveInf(value),
            .int => false,
            else => @compileError("sparse isPosInf requires numeric values"),
        },
        .neg_inf => switch (@typeInfo(T)) {
            .float => std.math.isNegativeInf(value),
            .int => false,
            else => @compileError("sparse isNegInf requires numeric values"),
        },
        .finite => switch (@typeInfo(T)) {
            .float => std.math.isFinite(value),
            .int => true,
            else => @compileError("sparse isFinite requires numeric values"),
        },
        .normal => switch (@typeInfo(T)) {
            .float => std.math.isNormal(value),
            else => @compileError("sparse isNormal requires floating-point values"),
        },
    };
}

const SparseLogicalBinary = enum { and_, or_, xor_ };

fn sparseDenseLogicalNot(matrix: anytype) SparseError!array_mod.Array(bool) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.logicalNot();
}

fn sparseDenseLogicalBinary(lhs: anytype, rhs: @TypeOf(lhs), comptime op: SparseLogicalBinary) SparseError!array_mod.Array(bool) {
    if (lhs.rows != rhs.rows or lhs.cols != rhs.cols) return error.ShapeMismatch;
    var lhs_dense = try lhs.toDense();
    defer lhs_dense.deinit();
    var rhs_dense = try rhs.toDense();
    defer rhs_dense.deinit();
    return switch (op) {
        .and_ => lhs_dense.logicalAnd(rhs_dense),
        .or_ => lhs_dense.logicalOr(rhs_dense),
        .xor_ => lhs_dense.logicalXor(rhs_dense),
    };
}

fn sparseDenseLogicalScalar(matrix: anytype, scalar: bool, comptime op: SparseLogicalBinary) SparseError!array_mod.Array(bool) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return switch (op) {
        .and_ => dense.logicalAndScalar(scalar),
        .or_ => dense.logicalOrScalar(scalar),
        .xor_ => dense.logicalXorScalar(scalar),
    };
}

fn sparseDenseWhere(comptime T: type, lhs: anytype, mask: array_mod.Array(bool), rhs: @TypeOf(lhs)) SparseError!array_mod.Array(T) {
    try validateDenseMatrixShape(lhs.rows, lhs.cols, mask.shape);
    if (lhs.rows != rhs.rows or lhs.cols != rhs.cols) return error.ShapeMismatch;
    var lhs_dense = try lhs.toDense();
    defer lhs_dense.deinit();
    var rhs_dense = try rhs.toDense();
    defer rhs_dense.deinit();
    return lhs_dense.where(mask, rhs_dense);
}

fn sparseDenseWhereArray(comptime T: type, lhs: anytype, mask: array_mod.Array(bool), rhs: array_mod.Array(T)) SparseError!array_mod.Array(T) {
    try validateDenseMatrixShape(lhs.rows, lhs.cols, mask.shape);
    try validateDenseMatrixShape(lhs.rows, lhs.cols, rhs.shape);
    var lhs_dense = try lhs.toDense();
    defer lhs_dense.deinit();
    return lhs_dense.where(mask, rhs);
}

fn sparseDenseWhereScalar(comptime T: type, lhs: anytype, mask: array_mod.Array(bool), other_value: T) SparseError!array_mod.Array(T) {
    try validateDenseMatrixShape(lhs.rows, lhs.cols, mask.shape);
    var lhs_dense = try lhs.toDense();
    defer lhs_dense.deinit();
    return lhs_dense.whereScalar(mask, other_value);
}

fn sparseDenseMaskedSelect(comptime T: type, matrix: anytype, mask: array_mod.Array(bool)) SparseError!array_mod.Array(T) {
    try validateDenseMatrixShape(matrix.rows, matrix.cols, mask.shape);
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.maskedSelect(mask);
}

fn sparseDenseCompress(comptime T: type, matrix: anytype, condition: array_mod.Array(bool), axis_opt: ?isize) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.compress(condition, axis_opt);
}

fn sparseDenseMaskedFill(comptime T: type, matrix: anytype, mask: array_mod.Array(bool), value: T) SparseError!array_mod.Array(T) {
    try validateDenseMatrixShape(matrix.rows, matrix.cols, mask.shape);
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.maskedFill(mask, value);
}

fn sparseDenseMaskedScatter(comptime T: type, matrix: anytype, mask: array_mod.Array(bool), src: array_mod.Array(T)) SparseError!array_mod.Array(T) {
    try validateDenseMatrixShape(matrix.rows, matrix.cols, mask.shape);
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.maskedScatter(mask, src);
}

fn sparseDenseMaskedPut(comptime T: type, matrix: anytype, mask: array_mod.Array(bool), values: array_mod.Array(T)) SparseError!array_mod.Array(T) {
    try validateDenseMatrixShape(matrix.rows, matrix.cols, mask.shape);
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.maskedPut(mask, values);
}

fn sparseDenseTake(comptime T: type, matrix: anytype, indices: array_mod.Array(usize), axis_opt: ?isize) SparseError!array_mod.Array(T) {
    const extent = try sparseTakeExtent(matrix.rows, matrix.cols, axis_opt);
    for (indices.data) |index| {
        if (index >= extent) return error.IndexOutOfBounds;
    }
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.take(indices, axis_opt);
}

fn sparseDenseTakeSigned(comptime T: type, matrix: anytype, indices: array_mod.Array(isize), axis_opt: ?isize) SparseError!array_mod.Array(T) {
    const extent = try sparseTakeExtent(matrix.rows, matrix.cols, axis_opt);
    for (indices.data) |index| {
        _ = try sparseNormalizeSignedIndex(index, extent);
    }
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.takeSigned(indices, axis_opt);
}

fn sparseDenseTakeMode(comptime T: type, matrix: anytype, indices: array_mod.Array(usize), axis_opt: ?isize, mode: array_mod.IndexMode) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.takeMode(indices, axis_opt, mode);
}

fn sparseDenseTakeSignedMode(comptime T: type, matrix: anytype, indices: array_mod.Array(isize), axis_opt: ?isize, mode: array_mod.IndexMode) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.takeSignedMode(indices, axis_opt, mode);
}

fn sparseTakeExtent(rows: usize, cols: usize, axis_opt: ?isize) SparseError!usize {
    if (axis_opt) |axis| {
        return switch (try sparseNormalizeMatrixAxis(axis)) {
            0 => rows,
            1 => cols,
            else => unreachable,
        };
    }
    return sparseElementCount(rows, cols);
}

fn sparseNormalizeMatrixAxis(axis: isize) SparseError!usize {
    const normalized = if (axis < 0) 2 + axis else axis;
    if (normalized < 0 or normalized >= 2) return error.InvalidAxis;
    return @intCast(normalized);
}

fn sparseNormalizeSignedIndex(index: isize, len: usize) SparseError!usize {
    const signed_len: isize = @intCast(len);
    const normalized = if (index < 0) signed_len + index else index;
    if (normalized < 0 or normalized >= signed_len) return error.IndexOutOfBounds;
    return @intCast(normalized);
}

fn sparseValidateGatherShape(rows: usize, cols: usize, indices_shape: []const usize, axis_index: isize) SparseError!usize {
    if (indices_shape.len != 2) return error.ShapeMismatch;
    const axis = try sparseNormalizeMatrixAxis(axis_index);
    const major_extent = if (axis == 0) cols else rows;
    if (indices_shape[1 - axis] > major_extent) return error.ShapeMismatch;
    return axis;
}

fn sparseValidateGatherIndices(rows: usize, cols: usize, indices: array_mod.Array(usize), axis_index: isize) SparseError!void {
    const axis = try sparseValidateGatherShape(rows, cols, indices.shape, axis_index);
    const extent = if (axis == 0) rows else cols;
    for (indices.data) |index| {
        if (index >= extent) return error.IndexOutOfBounds;
    }
}

fn sparseValidateGatherSignedIndices(rows: usize, cols: usize, indices: array_mod.Array(isize), axis_index: isize) SparseError!void {
    const axis = try sparseValidateGatherShape(rows, cols, indices.shape, axis_index);
    const extent = if (axis == 0) rows else cols;
    for (indices.data) |index| {
        _ = try sparseNormalizeSignedIndex(index, extent);
    }
}

fn sparseDenseGather(comptime T: type, matrix: anytype, axis_index: isize, indices: array_mod.Array(usize)) SparseError!array_mod.Array(T) {
    try sparseValidateGatherIndices(matrix.rows, matrix.cols, indices, axis_index);
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.gather(axis_index, indices);
}

fn sparseDenseGatherSigned(comptime T: type, matrix: anytype, axis_index: isize, indices: array_mod.Array(isize)) SparseError!array_mod.Array(T) {
    try sparseValidateGatherSignedIndices(matrix.rows, matrix.cols, indices, axis_index);
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.gatherSigned(axis_index, indices);
}

fn sparseValidateScatterShapes(rows: usize, cols: usize, axis_index: isize, indices_shape: []const usize, src_shape: []const usize) SparseError!void {
    const axis = try sparseValidateGatherShape(rows, cols, indices_shape, axis_index);
    if (!std.mem.eql(usize, indices_shape, src_shape)) return error.ShapeMismatch;
    _ = axis;
}

fn sparseValidateScatterIndices(rows: usize, cols: usize, axis_index: isize, indices: array_mod.Array(usize), src_shape: []const usize) SparseError!void {
    try sparseValidateScatterShapes(rows, cols, axis_index, indices.shape, src_shape);
    try sparseValidateGatherIndices(rows, cols, indices, axis_index);
}

fn sparseDenseScatter(comptime T: type, matrix: anytype, axis_index: isize, indices: array_mod.Array(usize), src: array_mod.Array(T)) SparseError!array_mod.Array(T) {
    try sparseValidateScatterIndices(matrix.rows, matrix.cols, axis_index, indices, src.shape);
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.scatter(axis_index, indices, src);
}

fn sparseDenseScatterScalar(comptime T: type, matrix: anytype, axis_index: isize, indices: array_mod.Array(usize), value: T) SparseError!array_mod.Array(T) {
    try sparseValidateGatherIndices(matrix.rows, matrix.cols, indices, axis_index);
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.scatterScalar(axis_index, indices, value);
}

fn sparseDenseScatterReduce(comptime T: type, matrix: anytype, axis_index: isize, indices: array_mod.Array(usize), src: array_mod.Array(T), reduction: array_mod.ScatterReduce) SparseError!array_mod.Array(T) {
    try sparseValidateScatterIndices(matrix.rows, matrix.cols, axis_index, indices, src.shape);
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.scatterReduce(axis_index, indices, src, reduction);
}

fn sparseDenseScatterReduceScalar(comptime T: type, matrix: anytype, axis_index: isize, indices: array_mod.Array(usize), value: T, reduction: array_mod.ScatterReduce) SparseError!array_mod.Array(T) {
    try sparseValidateGatherIndices(matrix.rows, matrix.cols, indices, axis_index);
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.scatterReduceScalar(axis_index, indices, value, reduction);
}

fn sparseDenseSelect(comptime T: type, matrix: anytype, axis_index: isize, index: usize) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.select(axis_index, index);
}

fn sparseDenseSelectSigned(comptime T: type, matrix: anytype, axis_index: isize, index: isize) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.selectSigned(axis_index, index);
}

fn sparseDenseNarrow(comptime T: type, matrix: anytype, axis_index: isize, start: usize, length: usize) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.narrow(axis_index, start, length);
}

fn sparseDenseNarrowSigned(comptime T: type, matrix: anytype, axis_index: isize, start: isize, length: usize) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.narrowSigned(axis_index, start, length);
}

fn sparseDenseDiagonalOffset(comptime T: type, matrix: anytype, offset: isize) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.diagonal(offset);
}

fn sparseDenseRavelCoords(matrix: anytype, coords: array_mod.Array(usize)) SparseError!array_mod.Array(usize) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.ravelCoords(coords);
}

fn sparseDenseUnravelFlat(matrix: anytype, indices: array_mod.Array(usize)) SparseError!array_mod.Array(usize) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.unravelFlat(indices);
}

fn sparseDenseTakeCoords(comptime T: type, matrix: anytype, coords: array_mod.Array(usize)) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.takeCoords(coords);
}

fn sparseDensePutCoords(comptime T: type, matrix: anytype, coords: array_mod.Array(usize), values: array_mod.Array(T)) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.putCoords(coords, values);
}

fn sparseDensePutCoordsScalar(comptime T: type, matrix: anytype, coords: array_mod.Array(usize), value: T) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.putCoordsScalar(coords, value);
}

fn sparseDenseReshape(comptime T: type, matrix: anytype, dims: []const usize) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.reshape(dims);
}

fn sparseDenseReshapeInfer(comptime T: type, matrix: anytype, dims: []const isize) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.reshapeInfer(dims);
}

fn sparseDenseFlattenAxes(comptime T: type, matrix: anytype, start_axis: isize, end_axis: isize) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.flattenAxes(start_axis, end_axis);
}

fn sparseDenseAtLeast1d(comptime T: type, matrix: anytype) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.atLeast1d();
}

fn sparseDenseAtLeast2d(comptime T: type, matrix: anytype) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.atLeast2d();
}

fn sparseDenseAtLeast3d(comptime T: type, matrix: anytype) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.atLeast3d();
}

fn sparseDenseUnflatten(comptime T: type, matrix: anytype, axis_index: isize, dims: []const usize) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.unflatten(axis_index, dims);
}

fn sparseDenseFlatten(comptime T: type, matrix: anytype) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.flatten();
}

fn sparseDenseSqueeze(comptime T: type, matrix: anytype, axis_opt: ?isize) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.squeeze(axis_opt);
}

fn sparseDenseUnsqueeze(comptime T: type, matrix: anytype, axis_index: isize) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.unsqueeze(axis_index);
}

fn sparseDenseSqueezeAxes(comptime T: type, matrix: anytype, axes: []const isize) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.squeezeAxes(axes);
}

fn sparseDenseUnsqueezeAxes(comptime T: type, matrix: anytype, axes: []const isize) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.unsqueezeAxes(axes);
}

fn sparseDenseBroadcastTo(comptime T: type, matrix: anytype, dims: []const usize) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.broadcastTo(dims);
}

fn sparseDensePermute(comptime T: type, matrix: anytype, axes: []const usize) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.permute(axes);
}

fn sparseDenseSwapAxes(comptime T: type, matrix: anytype, dim0: isize, dim1: isize) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.swapaxes(dim0, dim1);
}

fn sparseDenseMoveDim(comptime T: type, matrix: anytype, source: isize, destination: isize) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.movedim(source, destination);
}

fn sparseDenseFlip(comptime T: type, matrix: anytype, axis_index: isize) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.flip(axis_index);
}

fn sparseDenseFlipAxes(comptime T: type, matrix: anytype, axes: []const isize) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.flipAxes(axes);
}

fn sparseDenseRoll(comptime T: type, matrix: anytype, shift: isize, axis_index: isize) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.roll(shift, axis_index);
}

fn sparseDenseRollFlat(comptime T: type, matrix: anytype, shift: isize) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.rollFlat(shift);
}

fn sparseDenseRollAxes(comptime T: type, matrix: anytype, shifts: []const isize, axes: []const isize) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.rollAxes(shifts, axes);
}

fn sparseDenseRot90(comptime T: type, matrix: anytype, k: isize, axes: [2]isize) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.rot90(k, axes);
}

fn sparseDensePadConstant(comptime T: type, matrix: anytype, before: []const usize, after: []const usize, value: T) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.padConstant(before, after, value);
}

fn sparseDensePadEdge(comptime T: type, matrix: anytype, before: []const usize, after: []const usize) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.padEdge(before, after);
}

fn sparseDensePadReflect(comptime T: type, matrix: anytype, before: []const usize, after: []const usize) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.padReflect(before, after);
}

fn sparseDensePadWrap(comptime T: type, matrix: anytype, before: []const usize, after: []const usize) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.padWrap(before, after);
}

fn sparseDensePadSymmetric(comptime T: type, matrix: anytype, before: []const usize, after: []const usize) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.padSymmetric(before, after);
}

fn sparseDenseSplit(comptime T: type, matrix: anytype, split_size: usize, axis_index: isize) SparseError!array_mod.Array(T).SplitResult {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.split(split_size, axis_index);
}

fn sparseDenseSplitWithSizes(comptime T: type, matrix: anytype, sizes: []const usize, axis_index: isize) SparseError!array_mod.Array(T).SplitResult {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.splitWithSizes(sizes, axis_index);
}

fn sparseDenseSplitAtIndices(comptime T: type, matrix: anytype, indices: []const usize, axis_index: isize) SparseError!array_mod.Array(T).SplitResult {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.splitAtIndices(indices, axis_index);
}

fn sparseDenseChunk(comptime T: type, matrix: anytype, chunks: usize, axis_index: isize) SparseError!array_mod.Array(T).SplitResult {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.chunk(chunks, axis_index);
}

fn sparseDenseUnbind(comptime T: type, matrix: anytype, axis_index: isize) SparseError!array_mod.Array(T).SplitResult {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.unbind(axis_index);
}

fn sparseDenseRepeat(comptime T: type, matrix: anytype, repeats: usize, axis_index: isize) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.repeat(repeats, axis_index);
}

fn sparseDenseTile(comptime T: type, matrix: anytype, repeats: []const usize) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.tile(repeats);
}

fn sparseDenseRepeatInterleave(comptime T: type, matrix: anytype, repeats: array_mod.Array(usize), axis_opt: ?isize) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.repeatInterleave(repeats, axis_opt);
}

fn sparseDenseRepeatInterleaveScalar(comptime T: type, matrix: anytype, repeat_count: usize, axis_opt: ?isize) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.repeatInterleaveScalar(repeat_count, axis_opt);
}

fn sparseValidatePutFlatValues(comptime T: type, indices: array_mod.Array(usize), values: array_mod.Array(T)) SparseError!void {
    if (values.data.len != 1 and values.data.len != indices.data.len) return error.ShapeMismatch;
}

fn sparseValidatePutFlatIndices(rows: usize, cols: usize, indices: array_mod.Array(usize)) SparseError!void {
    const extent = try sparseElementCount(rows, cols);
    for (indices.data) |index| {
        if (index >= extent) return error.IndexOutOfBounds;
    }
}

fn sparseValidatePutFlatSignedIndices(rows: usize, cols: usize, indices: array_mod.Array(isize)) SparseError!void {
    const extent = try sparseElementCount(rows, cols);
    for (indices.data) |index| {
        _ = try sparseNormalizeSignedIndex(index, extent);
    }
}

fn sparseDensePutFlat(comptime T: type, matrix: anytype, indices: array_mod.Array(usize), values: array_mod.Array(T)) SparseError!array_mod.Array(T) {
    try sparseValidatePutFlatValues(T, indices, values);
    try sparseValidatePutFlatIndices(matrix.rows, matrix.cols, indices);
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.putFlat(indices, values);
}

fn sparseDensePutFlatMode(comptime T: type, matrix: anytype, indices: array_mod.Array(usize), values: array_mod.Array(T), mode: array_mod.IndexMode) SparseError!array_mod.Array(T) {
    try sparseValidatePutFlatValues(T, indices, values);
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.putFlatMode(indices, values, mode);
}

fn sparseDensePutFlatScalar(comptime T: type, matrix: anytype, indices: array_mod.Array(usize), value: T) SparseError!array_mod.Array(T) {
    try sparseValidatePutFlatIndices(matrix.rows, matrix.cols, indices);
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.putFlatScalar(indices, value);
}

fn sparseDensePutFlatScalarMode(comptime T: type, matrix: anytype, indices: array_mod.Array(usize), value: T, mode: array_mod.IndexMode) SparseError!array_mod.Array(T) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.putFlatScalarMode(indices, value, mode);
}

fn sparseDensePutFlatSigned(comptime T: type, matrix: anytype, indices: array_mod.Array(isize), values: array_mod.Array(T)) SparseError!array_mod.Array(T) {
    if (values.data.len != 1 and values.data.len != indices.data.len) return error.ShapeMismatch;
    try sparseValidatePutFlatSignedIndices(matrix.rows, matrix.cols, indices);
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.putFlatSigned(indices, values);
}

fn sparseDensePutFlatScalarSigned(comptime T: type, matrix: anytype, indices: array_mod.Array(isize), value: T) SparseError!array_mod.Array(T) {
    try sparseValidatePutFlatSignedIndices(matrix.rows, matrix.cols, indices);
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.putFlatScalarSigned(indices, value);
}

fn validateDenseMatrixShape(rows: usize, cols: usize, shape: []const usize) SparseError!void {
    if (shape.len != 2) return error.NonMatrixArray;
    if (rows != shape[0] or cols != shape[1]) return error.ShapeMismatch;
}

fn sparseCompareDenseArrays(
    comptime T: type,
    lhs: array_mod.Array(T),
    rhs: array_mod.Array(T),
    comptime comparison: SparseScalarComparison,
) SparseError!array_mod.Array(bool) {
    return switch (comparison) {
        .eq => lhs.eq(rhs),
        .ne => lhs.ne(rhs),
        .gt => lhs.gt(rhs),
        .ge => lhs.ge(rhs),
        .lt => lhs.lt(rhs),
        .le => lhs.le(rhs),
    };
}

fn sparseDenseCountNonzero(matrix: anytype) SparseError!usize {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.countNonzero();
}

fn sparseDenseCountNonzeroAxis(matrix: anytype, axis_opt: ?isize, keepdims: bool) SparseError!array_mod.Array(usize) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.countNonzeroAxis(axis_opt, keepdims);
}

fn sparseDenseCountNonzeroAxes(matrix: anytype, axes: []const isize, keepdims: bool) SparseError!array_mod.Array(usize) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.countNonzeroAxes(axes, keepdims);
}

fn sparseDenseFlatNonzero(matrix: anytype) SparseError!array_mod.Array(usize) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.flatNonzero();
}

fn sparseDenseNonzero(matrix: anytype) SparseError!array_mod.Array(usize) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.nonzero();
}

fn sparseDenseAll(matrix: anytype) SparseError!bool {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.all();
}

fn sparseDenseAny(matrix: anytype) SparseError!bool {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.any();
}

fn sparseDenseAllAxis(matrix: anytype, axis_opt: ?isize, keepdims: bool) SparseError!array_mod.Array(bool) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.allAxis(axis_opt, keepdims);
}

fn sparseDenseAllAxes(matrix: anytype, axes: []const isize, keepdims: bool) SparseError!array_mod.Array(bool) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.allAxes(axes, keepdims);
}

fn sparseDenseAnyAxis(matrix: anytype, axis_opt: ?isize, keepdims: bool) SparseError!array_mod.Array(bool) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.anyAxis(axis_opt, keepdims);
}

fn sparseDenseAnyAxes(matrix: anytype, axes: []const isize, keepdims: bool) SparseError!array_mod.Array(bool) {
    var dense = try matrix.toDense();
    defer dense.deinit();
    return dense.anyAxes(axes, keepdims);
}

pub const SparseResidualSummary = struct {
    residual_norm: f64,
    relative_residual_norm: f64,
    operator_frobenius_norm: f64,
    input_norm: f64,
    rhs_norm: f64,

    pub fn residualNormMeetsBound(self: SparseResidualSummary, max_residual: f64) SparseError!bool {
        try validateNonNegativeRange(max_residual, max_residual);
        return self.residual_norm <= max_residual;
    }

    pub fn relativeResidualNormMeetsBound(self: SparseResidualSummary, max_relative_residual: f64) SparseError!bool {
        try validateNonNegativeRange(max_relative_residual, max_relative_residual);
        return self.relative_residual_norm <= max_relative_residual;
    }

    pub fn meetsBounds(self: SparseResidualSummary, max_residual: f64, max_relative_residual: f64) SparseError!bool {
        return try self.residualNormMeetsBound(max_residual) and
            try self.relativeResidualNormMeetsBound(max_relative_residual);
    }
};

fn zero(comptime T: type) T {
    return switch (@typeInfo(T)) {
        .bool => false,
        else => @as(T, 0),
    };
}

fn oneValue(comptime T: type) T {
    return switch (@typeInfo(T)) {
        .bool => true,
        else => @as(T, 1),
    };
}

fn addSparseValue(comptime T: type, lhs: T, rhs: T) T {
    return switch (@typeInfo(T)) {
        .bool => lhs or rhs,
        else => lhs + rhs,
    };
}

fn mulSparseValue(comptime T: type, lhs: T, rhs: T) T {
    return switch (@typeInfo(T)) {
        .bool => lhs and rhs,
        else => lhs * rhs,
    };
}

fn negSparseValue(comptime T: type, value: T) T {
    return switch (@typeInfo(T)) {
        .float => -value,
        .int => if (@typeInfo(T).int.signedness == .signed)
            -value
        else
            @compileError("sparse negation requires signed integer or floating-point values"),
        else => @compileError("sparse negation requires signed integer or floating-point values"),
    };
}

fn signSparseValue(comptime T: type, value: T) T {
    ensureNumeric(T);
    return switch (@typeInfo(T)) {
        .float => if (std.math.isNan(value)) value else if (value > zero(T)) oneValue(T) else if (value < zero(T)) -oneValue(T) else zero(T),
        .int => |info| if (value == zero(T)) zero(T) else if (info.signedness == .signed) (if (value < zero(T)) -oneValue(T) else oneValue(T)) else oneValue(T),
        else => @compileError("sparse sign requires numeric values"),
    };
}

fn signbitSparseValue(comptime T: type, value: T) bool {
    ensureNumeric(T);
    return switch (@typeInfo(T)) {
        .float => std.math.signbit(value),
        .int => |info| if (info.signedness == .signed) value < zero(T) else false,
        else => @compileError("sparse signbit requires numeric values"),
    };
}

fn floorSparseValue(comptime T: type, value: T) T {
    ensureNumeric(T);
    return switch (@typeInfo(T)) {
        .float => @floor(value),
        .int => value,
        else => @compileError("sparse floor requires numeric values"),
    };
}

fn ceilSparseValue(comptime T: type, value: T) T {
    ensureNumeric(T);
    return switch (@typeInfo(T)) {
        .float => @ceil(value),
        .int => value,
        else => @compileError("sparse ceil requires numeric values"),
    };
}

fn roundSparseValue(comptime T: type, value: T) T {
    ensureNumeric(T);
    return switch (@typeInfo(T)) {
        .float => @round(value),
        .int => value,
        else => @compileError("sparse round requires numeric values"),
    };
}

fn truncSparseValue(comptime T: type, value: T) T {
    ensureNumeric(T);
    return switch (@typeInfo(T)) {
        .float => @trunc(value),
        .int => value,
        else => @compileError("sparse trunc requires numeric values"),
    };
}

fn cbrtSparseValue(comptime T: type, value: T) T {
    ensureFloat(T);
    return switch (T) {
        f16 => @floatCast(std.math.cbrt(@as(f32, @floatCast(value)))),
        f32, f64 => std.math.cbrt(value),
        else => @compileError("sparse cbrt requires f16, f32, or f64 values"),
    };
}

fn isNonZero(comptime T: type, value: T) bool {
    return switch (@typeInfo(T)) {
        .bool => value,
        else => value != zero(T),
    };
}

fn absValue(comptime T: type, value: T) T {
    return switch (@typeInfo(T)) {
        .float => @abs(value),
        .int => if (@typeInfo(T).int.signedness == .signed and value < 0) -value else value,
        else => @compileError("sparse absValue requires numeric values"),
    };
}

fn absDifference(comptime T: type, lhs: T, rhs: T) T {
    return switch (@typeInfo(T)) {
        .float => @abs(lhs - rhs),
        .int => if (@typeInfo(T).int.signedness == .signed)
            absValue(T, lhs - rhs)
        else if (lhs >= rhs)
            lhs - rhs
        else
            rhs - lhs,
        else => @compileError("sparse absolute difference requires numeric values"),
    };
}

fn sparseAbsValueExceedsTolerance(comptime T: type, value: T, tolerance: T) bool {
    ensureNumeric(T);
    return absValue(T, value) > tolerance;
}

fn sparseDropZerosNnz(comptime T: type, values: []const T) usize {
    var nonzero_count: usize = 0;
    for (values) |value| {
        if (isNonZero(T, value)) nonzero_count += 1;
    }
    return nonzero_count;
}

fn sparseDenseNnz(comptime T: type, input: array_mod.Array(T)) SparseError!usize {
    if (input.shape.len != 2) return error.NonMatrixArray;
    return sparseDropZerosNnz(T, input.data);
}

fn sparseDiagonalNnz(comptime T: type, diagonal_values: []const T, offset: isize) SparseError!usize {
    _ = try diagonalExtent(diagonal_values.len, offset);
    return sparseDropZerosNnz(T, diagonal_values);
}

fn sparsePruneZerosNnz(comptime T: type, values: []const T, tolerance: T) SparseError!usize {
    try validateSparseValueRange(T, zero(T), tolerance);
    var keep_count: usize = 0;
    for (values) |value| {
        if (sparseAbsValueExceedsTolerance(T, value, tolerance)) keep_count += 1;
    }
    return keep_count;
}

fn sparseDensePrunedNnz(comptime T: type, input: array_mod.Array(T), tolerance: T) SparseError!usize {
    if (input.shape.len != 2) return error.NonMatrixArray;
    return sparsePruneZerosNnz(T, input.data, tolerance);
}

fn sparseDiagonalPrunedNnz(comptime T: type, diagonal_values: []const T, offset: isize, tolerance: T) SparseError!usize {
    _ = try diagonalExtent(diagonal_values.len, offset);
    return sparsePruneZerosNnz(T, diagonal_values, tolerance);
}

fn valueLess(comptime T: type, lhs: T, rhs: T) bool {
    return lhs < rhs;
}

fn valueGreater(comptime T: type, lhs: T, rhs: T) bool {
    return lhs > rhs;
}

fn ensureNumeric(comptime T: type) void {
    switch (@typeInfo(T)) {
        .int, .float => {},
        else => @compileError("sparse statistic requires numeric values"),
    }
}

fn minStoredValue(comptime T: type, values: []const T) SparseError!T {
    ensureNumeric(T);
    if (values.len == 0) return error.EmptyArray;
    var result = values[0];
    for (values[1..]) |value| {
        if (value < result) result = value;
    }
    return result;
}

fn minStoredValueIndex(comptime T: type, values: []const T) SparseError!usize {
    ensureNumeric(T);
    if (values.len == 0) return error.EmptyArray;
    var result_index: usize = 0;
    var result = values[0];
    for (values[1..], 1..) |value, index| {
        if (value < result) {
            result = value;
            result_index = index;
        }
    }
    return result_index;
}

fn maxStoredValue(comptime T: type, values: []const T) SparseError!T {
    ensureNumeric(T);
    if (values.len == 0) return error.EmptyArray;
    var result = values[0];
    for (values[1..]) |value| {
        if (value > result) result = value;
    }
    return result;
}

fn maxStoredValueIndex(comptime T: type, values: []const T) SparseError!usize {
    ensureNumeric(T);
    if (values.len == 0) return error.EmptyArray;
    var result_index: usize = 0;
    var result = values[0];
    for (values[1..], 1..) |value, index| {
        if (value > result) {
            result = value;
            result_index = index;
        }
    }
    return result_index;
}

fn minStoredAbsValue(comptime T: type, values: []const T) SparseError!T {
    ensureNumeric(T);
    if (values.len == 0) return error.EmptyArray;
    var result = absValue(T, values[0]);
    for (values[1..]) |value| {
        const magnitude = absValue(T, value);
        if (magnitude < result) result = magnitude;
    }
    return result;
}

fn minStoredAbsValueIndex(comptime T: type, values: []const T) SparseError!usize {
    ensureNumeric(T);
    if (values.len == 0) return error.EmptyArray;
    var result_index: usize = 0;
    var result = absValue(T, values[0]);
    for (values[1..], 1..) |value, index| {
        const magnitude = absValue(T, value);
        if (magnitude < result) {
            result = magnitude;
            result_index = index;
        }
    }
    return result_index;
}

fn maxStoredAbsValue(comptime T: type, values: []const T) SparseError!T {
    ensureNumeric(T);
    if (values.len == 0) return error.EmptyArray;
    var result = absValue(T, values[0]);
    for (values[1..]) |value| {
        const magnitude = absValue(T, value);
        if (magnitude > result) result = magnitude;
    }
    return result;
}

fn maxStoredAbsValueIndex(comptime T: type, values: []const T) SparseError!usize {
    ensureNumeric(T);
    if (values.len == 0) return error.EmptyArray;
    var result_index: usize = 0;
    var result = absValue(T, values[0]);
    for (values[1..], 1..) |value, index| {
        const magnitude = absValue(T, value);
        if (magnitude > result) {
            result = magnitude;
            result_index = index;
        }
    }
    return result_index;
}

fn ensureFloat(comptime T: type) void {
    if (@typeInfo(T) != .float) @compileError("sparse norm requires floating-point values");
}

fn sparseValueToF64(comptime T: type, value: T) f64 {
    return switch (@typeInfo(T)) {
        .float => @floatCast(value),
        .int => @floatFromInt(value),
        else => @compileError("sparse mean requires numeric values"),
    };
}

fn sparseCastValue(comptime U: type, value: anytype) U {
    const V = @TypeOf(value);
    return switch (@typeInfo(U)) {
        .bool => switch (@typeInfo(V)) {
            .bool => value,
            .int, .comptime_int => value != 0,
            .float, .comptime_float => value != 0,
            else => @compileError("unsupported sparse cast from " ++ @typeName(V) ++ " to bool"),
        },
        .int => switch (@typeInfo(V)) {
            .bool => if (value) @as(U, 1) else @as(U, 0),
            .int, .comptime_int => @intCast(value),
            .float, .comptime_float => @intFromFloat(value),
            else => @compileError("unsupported sparse cast from " ++ @typeName(V) ++ " to " ++ @typeName(U)),
        },
        .float => switch (@typeInfo(V)) {
            .bool => if (value) @as(U, 1) else @as(U, 0),
            .int, .comptime_int => @floatFromInt(value),
            .float, .comptime_float => @floatCast(value),
            else => @compileError("unsupported sparse cast from " ++ @typeName(V) ++ " to " ++ @typeName(U)),
        },
        else => @compileError("unsupported sparse cast target type: " ++ @typeName(U)),
    };
}

fn sparseSizeToF64(value: usize) f64 {
    return @floatFromInt(value);
}

fn validateNonNegativeRange(min_value: f64, max_value: f64) SparseError!void {
    if (!std.math.isFinite(min_value) or !std.math.isFinite(max_value) or min_value < 0 or max_value < 0 or min_value > max_value) return error.InvalidShape;
}

fn validateFiniteRange(min_value: f64, max_value: f64) SparseError!void {
    if (!std.math.isFinite(min_value) or !std.math.isFinite(max_value) or min_value > max_value) return error.InvalidShape;
}

fn valueInF64Range(value: f64, min_value: f64, max_value: f64) bool {
    return value >= min_value and value <= max_value;
}

fn sparseStddevInValidatedRangeFromVariance(variance_value: f64, min_stddev: f64, max_stddev: f64) bool {
    const stddev_value = @sqrt(variance_value);
    return valueInF64Range(stddev_value, min_stddev, max_stddev);
}

fn sparseNormalizedTraceFromTrace(comptime T: type, trace_value: T, size: usize) SparseError!f64 {
    ensureNumeric(T);
    if (size == 0) return error.EmptyArray;
    return sparseValueToF64(T, trace_value) / sparseSizeToF64(size);
}

fn sparseNormalizedTraceInRangeFromTrace(
    comptime T: type,
    trace_value: T,
    size: usize,
    min_value: f64,
    max_value: f64,
) SparseError!bool {
    if (size == 0) return error.EmptyArray;
    const trace_f64 = sparseValueToF64(T, trace_value);
    const size_f64 = sparseSizeToF64(size);
    const scaled_min = min_value * size_f64;
    const scaled_max = max_value * size_f64;
    // Compare on the unnormalized trace when scaling remains finite; this
    // avoids one rounding step and preserves the Veyra-style overflow fallback
    // for extremely large user bounds.
    if (std.math.isFinite(scaled_min) and std.math.isFinite(scaled_max)) {
        return trace_f64 >= scaled_min and trace_f64 <= scaled_max;
    }
    const normalized = trace_f64 / size_f64;
    return normalized >= min_value and normalized <= max_value;
}

fn sparseCountAverage(count: usize, divisor: usize) SparseError!f64 {
    if (divisor == 0) return error.EmptyArray;
    return @as(f64, @floatFromInt(count)) / @as(f64, @floatFromInt(divisor));
}

fn sparseCountAverageInRange(count: usize, divisor: usize, min_average: f64, max_average: f64) SparseError!bool {
    try validateNonNegativeRange(min_average, max_average);
    const average = try sparseCountAverage(count, divisor);
    return average >= min_average and average <= max_average;
}

fn sparseCountFraction(count: usize, divisor: usize) SparseError!f64 {
    if (divisor == 0) return error.EmptyArray;
    return @as(f64, @floatFromInt(count)) / @as(f64, @floatFromInt(divisor));
}

fn sparseCountFractionInRange(count: usize, divisor: usize, min_fraction: f64, max_fraction: f64) SparseError!bool {
    try validateNonNegativeRange(min_fraction, max_fraction);
    const fraction = try sparseCountFraction(count, divisor);
    return fraction >= min_fraction and fraction <= max_fraction;
}

fn sparseMinCount(counts: []const usize) usize {
    if (counts.len == 0) return 0;
    var result = counts[0];
    for (counts[1..]) |count| {
        if (count < result) result = count;
    }
    return result;
}

fn sparseMaxCount(counts: []const usize) usize {
    var result: usize = 0;
    for (counts) |count| {
        if (count > result) result = count;
    }
    return result;
}

fn validateCountRange(min_count: usize, max_count: usize) SparseError!void {
    if (min_count > max_count) return error.InvalidShape;
}

fn sparseCountRangeInRange(counts: []const usize, min_count: usize, max_count: usize) SparseError!bool {
    try validateCountRange(min_count, max_count);
    for (counts) |count| {
        if (count < min_count or count > max_count) return false;
    }
    return true;
}

fn sparseCountInValidatedRange(count: usize, min_count: usize, max_count: usize) bool {
    return count >= min_count and count <= max_count;
}

fn sparseCountSpread(counts: []const usize) usize {
    return sparseMaxCount(counts) - sparseMinCount(counts);
}

fn sparseCountSpreadMeetsBound(counts: []const usize, max_spread: usize) bool {
    return sparseCountSpread(counts) <= max_spread;
}

fn sparseColumnIntersectionBandwidthFromCoo(
    allocator: std.mem.Allocator,
    rows: usize,
    cols: usize,
    row_indices: []const usize,
    col_indices: []const usize,
) SparseError!usize {
    if (cols == 0) return 0;
    var row_seen = try allocator.alloc(bool, rows);
    defer allocator.free(row_seen);

    var bandwidth: usize = 0;
    for (0..cols) |left_col| {
        @memset(row_seen, false);
        for (row_indices, col_indices) |row, col| {
            if (col == left_col) row_seen[row] = true;
        }

        for ((left_col + 1)..cols) |right_col| {
            for (row_indices, col_indices) |row, col| {
                if (col == right_col and row_seen[row]) {
                    const distance = right_col - left_col;
                    if (distance > bandwidth) bandwidth = distance;
                    break;
                }
            }
        }
    }
    return bandwidth;
}

fn sparseColumnIntersectionBandwidthMeetsBoundFromCoo(
    allocator: std.mem.Allocator,
    rows: usize,
    cols: usize,
    row_indices: []const usize,
    col_indices: []const usize,
    max_bandwidth: usize,
) SparseError!bool {
    if (cols == 0) return true;
    var row_seen = try allocator.alloc(bool, rows);
    defer allocator.free(row_seen);

    for (0..cols) |left_col| {
        @memset(row_seen, false);
        for (row_indices, col_indices) |row, col| {
            if (col == left_col) row_seen[row] = true;
        }

        for ((left_col + 1)..cols) |right_col| {
            if (right_col - left_col <= max_bandwidth) continue;
            for (row_indices, col_indices) |row, col| {
                if (col == right_col and row_seen[row]) return false;
            }
        }
    }
    return true;
}

fn sparseElementCount(rows: usize, cols: usize) SparseError!usize {
    return std.math.mul(usize, rows, cols) catch return error.InvalidShape;
}

fn sparseValueSquareToF64(comptime T: type, value: T) f64 {
    const numeric = sparseValueToF64(T, value);
    return numeric * numeric;
}

fn sparseVectorL2Norm(comptime T: type, values: []const T) T {
    ensureFloat(T);
    var total = zero(T);
    for (values) |value| total += value * value;
    return @sqrt(total);
}

fn sparseVectorResidualNorm(comptime T: type, lhs: []const T, rhs: []const T) SparseError!T {
    ensureFloat(T);
    if (lhs.len != rhs.len) return error.ShapeMismatch;
    var total = zero(T);
    for (lhs, rhs) |lhs_value, rhs_value| {
        const diff = lhs_value - rhs_value;
        total += diff * diff;
    }
    return @sqrt(total);
}

fn sparseRelativeResidualNorm(
    comptime T: type,
    residual_norm: T,
    operator_norm: T,
    x_values: []const T,
    rhs_values: []const T,
) T {
    ensureFloat(T);
    const scale = @max(oneValue(T), operator_norm * sparseVectorL2Norm(T, x_values) + sparseVectorL2Norm(T, rhs_values));
    return residual_norm / scale;
}

fn sparseMatrixResidualNorm(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) SparseError!T {
    ensureFloat(T);
    if (lhs.shape.len != 2 or rhs.shape.len != 2) return error.NonMatrixArray;
    if (lhs.shape[0] != rhs.shape[0] or lhs.shape[1] != rhs.shape[1]) return error.ShapeMismatch;
    return sparseVectorResidualNorm(T, lhs.data, rhs.data);
}

fn sparseRelativeMatrixResidualNorm(
    comptime T: type,
    residual_norm: T,
    operator_norm: T,
    x_values: []const T,
    rhs_values: []const T,
) T {
    return sparseRelativeResidualNorm(T, residual_norm, operator_norm, x_values, rhs_values);
}

fn sparseResidualSummary(
    comptime T: type,
    residual_norm: T,
    operator_norm: T,
    x_values: []const T,
    rhs_values: []const T,
) SparseResidualSummary {
    const input_norm = sparseVectorL2Norm(T, x_values);
    const rhs_norm = sparseVectorL2Norm(T, rhs_values);
    const scale = @max(oneValue(T), operator_norm * input_norm + rhs_norm);
    return .{
        .residual_norm = sparseValueToF64(T, residual_norm),
        .relative_residual_norm = sparseValueToF64(T, residual_norm / scale),
        .operator_frobenius_norm = sparseValueToF64(T, operator_norm),
        .input_norm = sparseValueToF64(T, input_norm),
        .rhs_norm = sparseValueToF64(T, rhs_norm),
    };
}

fn sparseSymmetryResidualFrobeniusNormFromDense(comptime T: type, values: []const T, rows: usize, cols: usize) SparseError!T {
    ensureFloat(T);
    if (rows != cols) return error.NonMatrixArray;
    var total = zero(T);
    for (0..rows) |row| {
        for ((row + 1)..cols) |col| {
            const diff = values[row * cols + col] - values[col * cols + row];
            total += diff * diff * @as(T, @floatFromInt(2));
        }
    }
    return @sqrt(total);
}

fn sparseValueIsFinite(comptime T: type, value: T) bool {
    if (comptime T == array_mod.BFloat16) return std.math.isFinite(value.toF32());
    if (comptime T == array_mod.Complex64 or T == array_mod.Complex128) return std.math.isFinite(value.re) and std.math.isFinite(value.im);
    return switch (@typeInfo(T)) {
        .bool, .int => true,
        .float => std.math.isFinite(value),
        else => @compileError("sparse finite diagnostics require bool, integer, or floating-point values"),
    };
}

fn sparseNonFiniteCount(comptime T: type, values: []const T) usize {
    var count: usize = 0;
    for (values) |value| {
        if (!sparseValueIsFinite(T, value)) count += 1;
    }
    return count;
}

fn sparseAllFinite(comptime T: type, values: []const T) bool {
    for (values) |value| {
        if (!sparseValueIsFinite(T, value)) return false;
    }
    return true;
}

fn validateSparseValueRange(comptime T: type, min_value: T, max_value: T) SparseError!void {
    ensureNumeric(T);
    if (!sparseValueIsFinite(T, min_value) or !sparseValueIsFinite(T, max_value) or min_value > max_value) return error.InvalidShape;
}

fn sparseValueInValidatedRange(comptime T: type, value: T, min_value: T, max_value: T) bool {
    return value >= min_value and value <= max_value;
}

fn sparseValueRangeInRange(comptime T: type, values: []const T, min_value: T, max_value: T) SparseError!bool {
    try validateSparseValueRange(T, min_value, max_value);
    if (values.len == 0) return error.EmptyArray;
    for (values) |value| {
        if (!(value >= min_value and value <= max_value)) return false;
    }
    return true;
}

fn sparseAbsValueRangeInRange(comptime T: type, values: []const T, min_abs_value: T, max_abs_value: T) SparseError!bool {
    try validateSparseValueRange(T, min_abs_value, max_abs_value);
    if (values.len == 0) return error.EmptyArray;
    for (values) |value| {
        const magnitude = absValue(T, value);
        if (!(magnitude >= min_abs_value and magnitude <= max_abs_value)) return false;
    }
    return true;
}

fn sparseStoredValueDynamicRange(comptime T: type, values: []const T) SparseError!f64 {
    ensureNumeric(T);
    if (values.len == 0) return error.EmptyArray;
    var min_abs = std.math.inf(f64);
    var max_abs: f64 = 0;
    for (values) |value| {
        if (!sparseValueIsFinite(T, value)) return error.InvalidShape;
        const magnitude = @abs(sparseValueToF64(T, value));
        if (magnitude == 0) return error.SingularMatrix;
        if (magnitude < min_abs) min_abs = magnitude;
        if (magnitude > max_abs) max_abs = magnitude;
    }
    return max_abs / min_abs;
}

fn sparseStoredValueDynamicRangeMeetsBound(comptime T: type, values: []const T, max_dynamic_range: f64) SparseError!bool {
    if (!std.math.isFinite(max_dynamic_range) or max_dynamic_range < 0) return error.InvalidShape;
    return (try sparseStoredValueDynamicRange(T, values)) <= max_dynamic_range;
}

fn sparseDiagonalAbsRange(comptime T: type, values: []const T) SparseError!struct { min_abs: T, max_abs: T } {
    ensureNumeric(T);
    if (values.len == 0) return error.EmptyArray;
    var min_abs = absValue(T, values[0]);
    var max_abs = min_abs;
    for (values[1..]) |value| {
        const magnitude = absValue(T, value);
        if (magnitude < min_abs) min_abs = magnitude;
        if (magnitude > max_abs) max_abs = magnitude;
    }
    return .{ .min_abs = min_abs, .max_abs = max_abs };
}

fn sparseDiagonalDynamicRangeFromValues(comptime T: type, values: []const T) SparseError!f64 {
    const range = try sparseDiagonalAbsRange(T, values);
    if (range.min_abs == zero(T)) return error.SingularMatrix;
    return sparseValueToF64(T, range.max_abs) / sparseValueToF64(T, range.min_abs);
}

fn sparseDiagonalDynamicRangeMeetsBoundFromValues(comptime T: type, values: []const T, max_dynamic_range: f64) SparseError!bool {
    if (!std.math.isFinite(max_dynamic_range) or max_dynamic_range < 0) return error.InvalidShape;
    return (try sparseDiagonalDynamicRangeFromValues(T, values)) <= max_dynamic_range;
}

fn sparseDiagonalDominanceFromCanonicalEntries(
    comptime T: type,
    allocator: std.mem.Allocator,
    rows: usize,
    cols: usize,
    row_indices: []const usize,
    col_indices: []const usize,
    values: []const T,
    comptime strict: bool,
) SparseError!bool {
    ensureNumeric(T);
    if (rows != cols) return error.NonMatrixArray;
    if (rows == 0) return error.EmptyArray;

    var diagonal_seen = try allocator.alloc(bool, rows);
    defer allocator.free(diagonal_seen);
    @memset(diagonal_seen, false);
    var diagonal_abs = try allocator.alloc(f64, rows);
    defer allocator.free(diagonal_abs);
    @memset(diagonal_abs, 0);
    var off_diagonal_abs_sums = try allocator.alloc(f64, rows);
    defer allocator.free(off_diagonal_abs_sums);
    @memset(off_diagonal_abs_sums, 0);

    // Callers pass canonicalized structural entries so duplicate coordinates
    // have already been summed. This keeps dominance checks aligned with
    // Vectra's dense materialization semantics instead of counting duplicate
    // off-diagonal magnitudes before cancellation.
    for (values, row_indices, col_indices) |value, row, col| {
        const magnitude = sparseValueToF64(T, absValue(T, value));
        if (row == col) {
            diagonal_seen[row] = true;
            diagonal_abs[row] = magnitude;
        } else {
            off_diagonal_abs_sums[row] += magnitude;
        }
    }

    for (diagonal_seen, diagonal_abs, off_diagonal_abs_sums) |seen, diag, offdiag| {
        if (!seen) return false;
        if (strict) {
            if (!(diag > offdiag)) return false;
        } else {
            if (!(diag >= offdiag)) return false;
        }
    }
    return true;
}

fn sparseDiagonalDominanceMarginFromCanonicalEntries(
    comptime T: type,
    allocator: std.mem.Allocator,
    rows: usize,
    cols: usize,
    row_indices: []const usize,
    col_indices: []const usize,
    values: []const T,
) SparseError!f64 {
    ensureNumeric(T);
    if (rows != cols) return error.NonMatrixArray;
    if (rows == 0) return error.EmptyArray;

    var diagonal_seen = try allocator.alloc(bool, rows);
    defer allocator.free(diagonal_seen);
    @memset(diagonal_seen, false);
    var diagonal_abs = try allocator.alloc(f64, rows);
    defer allocator.free(diagonal_abs);
    @memset(diagonal_abs, 0);
    var off_diagonal_abs_sums = try allocator.alloc(f64, rows);
    defer allocator.free(off_diagonal_abs_sums);
    @memset(off_diagonal_abs_sums, 0);

    for (values, row_indices, col_indices) |value, row, col| {
        const magnitude = sparseValueToF64(T, absValue(T, value));
        if (row == col) {
            diagonal_seen[row] = true;
            diagonal_abs[row] = magnitude;
        } else {
            off_diagonal_abs_sums[row] += magnitude;
        }
    }

    var margin = std.math.inf(f64);
    for (diagonal_seen, diagonal_abs, off_diagonal_abs_sums) |seen, diag, offdiag| {
        const row_margin = if (seen) diag - offdiag else -offdiag;
        if (row_margin < margin) margin = row_margin;
    }
    return margin;
}

fn sparseDiagonalDominanceMarginMeetsBoundFromCanonicalEntries(
    comptime T: type,
    allocator: std.mem.Allocator,
    rows: usize,
    cols: usize,
    row_indices: []const usize,
    col_indices: []const usize,
    values: []const T,
    min_margin: f64,
) SparseError!bool {
    if (!std.math.isFinite(min_margin)) return error.InvalidShape;
    return (try sparseDiagonalDominanceMarginFromCanonicalEntries(T, allocator, rows, cols, row_indices, col_indices, values)) >= min_margin;
}

fn sparseSameStructure(
    rows: usize,
    cols: usize,
    lhs_major_offsets: []const usize,
    lhs_minor_indices: []const usize,
    rhs_rows: usize,
    rhs_cols: usize,
    rhs_major_offsets: []const usize,
    rhs_minor_indices: []const usize,
) bool {
    return rows == rhs_rows and
        cols == rhs_cols and
        std.mem.eql(usize, lhs_major_offsets, rhs_major_offsets) and
        std.mem.eql(usize, lhs_minor_indices, rhs_minor_indices);
}

fn sparseDotSameStructure(comptime T: type, lhs_values: []const T, rhs_values: []const T) SparseError!T {
    ensureNumeric(T);
    if (lhs_values.len != rhs_values.len) return error.ShapeMismatch;
    var total = zero(T);
    for (lhs_values, rhs_values) |lhs, rhs| total += lhs * rhs;
    return total;
}

fn sparseMaxAbsDiffSameStructure(comptime T: type, lhs_values: []const T, rhs_values: []const T) SparseError!T {
    ensureNumeric(T);
    if (lhs_values.len != rhs_values.len) return error.ShapeMismatch;
    var max_diff = zero(T);
    for (lhs_values, rhs_values) |lhs, rhs| {
        const diff = absDifference(T, lhs, rhs);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

fn sparseMaxAbsDiffSameStructureMeetsBound(comptime T: type, lhs_values: []const T, rhs_values: []const T, max_absolute_diff: T) SparseError!bool {
    ensureNumeric(T);
    if (!sparseValueIsFinite(T, max_absolute_diff) or max_absolute_diff < zero(T)) return error.InvalidShape;
    return (try sparseMaxAbsDiffSameStructure(T, lhs_values, rhs_values)) <= max_absolute_diff;
}

fn sparseRelativeDiff(comptime T: type, lhs: T, rhs: T) T {
    const abs_diff = absDifference(T, lhs, rhs);
    const denom = @max(absValue(T, lhs), absValue(T, rhs));
    return if (denom == zero(T)) zero(T) else abs_diff / denom;
}

fn sparseMaxRelDiffSameStructure(comptime T: type, lhs_values: []const T, rhs_values: []const T) SparseError!T {
    ensureFloat(T);
    if (lhs_values.len != rhs_values.len) return error.ShapeMismatch;
    var max_diff = zero(T);
    for (lhs_values, rhs_values) |lhs, rhs| {
        const diff = sparseRelativeDiff(T, lhs, rhs);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

fn sparseMaxRelDiffSameStructureMeetsBound(comptime T: type, lhs_values: []const T, rhs_values: []const T, max_relative_diff: T) SparseError!bool {
    ensureFloat(T);
    if (!sparseValueIsFinite(T, max_relative_diff) or max_relative_diff < zero(T)) return error.InvalidShape;
    return (try sparseMaxRelDiffSameStructure(T, lhs_values, rhs_values)) <= max_relative_diff;
}

fn sparseSquaredDistanceSameStructure(comptime T: type, lhs_values: []const T, rhs_values: []const T) SparseError!T {
    ensureNumeric(T);
    if (lhs_values.len != rhs_values.len) return error.ShapeMismatch;
    var total = zero(T);
    for (lhs_values, rhs_values) |lhs, rhs| {
        const diff = lhs - rhs;
        total += diff * diff;
    }
    return total;
}

fn sparseSquaredDistanceSameStructureMeetsBound(comptime T: type, lhs_values: []const T, rhs_values: []const T, max_squared_distance: T) SparseError!bool {
    ensureNumeric(T);
    if (!sparseValueIsFinite(T, max_squared_distance) or max_squared_distance < zero(T)) return error.InvalidShape;
    return (try sparseSquaredDistanceSameStructure(T, lhs_values, rhs_values)) <= max_squared_distance;
}

fn sparseFrobeniusDistanceSameStructure(comptime T: type, lhs_values: []const T, rhs_values: []const T) SparseError!T {
    ensureFloat(T);
    return @sqrt(try sparseSquaredDistanceSameStructure(T, lhs_values, rhs_values));
}

fn sparseFrobeniusDistanceSameStructureMeetsBound(comptime T: type, lhs_values: []const T, rhs_values: []const T, max_distance: T) SparseError!bool {
    ensureFloat(T);
    if (!sparseValueIsFinite(T, max_distance) or max_distance < zero(T)) return error.InvalidShape;
    return (try sparseFrobeniusDistanceSameStructure(T, lhs_values, rhs_values)) <= max_distance;
}

fn sparseRelativeFrobeniusDistanceSameStructure(comptime T: type, lhs_values: []const T, rhs_values: []const T) SparseError!T {
    ensureFloat(T);
    const squared_distance = try sparseSquaredDistanceSameStructure(T, lhs_values, rhs_values);
    var lhs_norm_sq = zero(T);
    var rhs_norm_sq = zero(T);
    for (lhs_values, rhs_values) |lhs, rhs| {
        lhs_norm_sq += lhs * lhs;
        rhs_norm_sq += rhs * rhs;
    }
    const scale = @max(oneValue(T), @sqrt(lhs_norm_sq) + @sqrt(rhs_norm_sq));
    return @sqrt(squared_distance) / scale;
}

fn sparseRelativeFrobeniusDistanceSameStructureMeetsBound(comptime T: type, lhs_values: []const T, rhs_values: []const T, max_relative_distance: T) SparseError!bool {
    ensureFloat(T);
    if (!sparseValueIsFinite(T, max_relative_distance) or max_relative_distance < zero(T)) return error.InvalidShape;
    return (try sparseRelativeFrobeniusDistanceSameStructure(T, lhs_values, rhs_values)) <= max_relative_distance;
}

fn sparseSameStructureDiffSummary(comptime T: type, lhs_values: []const T, rhs_values: []const T) SparseError!SparseDiffSummary {
    ensureNumeric(T);
    if (lhs_values.len != rhs_values.len) return error.ShapeMismatch;

    var dot: f64 = 0;
    var max_abs_diff: f64 = 0;
    var max_rel_diff: f64 = 0;
    var squared_distance: f64 = 0;
    var lhs_norm_sq: f64 = 0;
    var rhs_norm_sq: f64 = 0;

    for (lhs_values, rhs_values) |lhs_raw, rhs_raw| {
        const lhs = sparseValueToF64(T, lhs_raw);
        const rhs = sparseValueToF64(T, rhs_raw);
        const diff = @abs(lhs - rhs);
        const denominator = @max(@abs(lhs), @abs(rhs));
        const rel_diff = if (denominator == 0) 0 else diff / denominator;

        dot += lhs * rhs;
        if (diff > max_abs_diff) max_abs_diff = diff;
        if (rel_diff > max_rel_diff) max_rel_diff = rel_diff;
        squared_distance += diff * diff;
        lhs_norm_sq += lhs * lhs;
        rhs_norm_sq += rhs * rhs;
    }

    return .{
        .dot = dot,
        .max_abs_diff = max_abs_diff,
        .max_rel_diff = max_rel_diff,
        .squared_distance = squared_distance,
        .lhs_frobenius_norm = @sqrt(lhs_norm_sq),
        .rhs_frobenius_norm = @sqrt(rhs_norm_sq),
    };
}

fn denseDiffSummary(comptime T: type, lhs_values: []const T, rhs_values: []const T) SparseError!SparseDiffSummary {
    return sparseSameStructureDiffSummary(T, lhs_values, rhs_values);
}

fn denseDiffSummaryMeetsBounds(
    comptime T: type,
    lhs_values: []const T,
    rhs_values: []const T,
    max_absolute_diff: f64,
    max_relative_diff: f64,
    max_squared_distance: f64,
    max_frobenius_distance: f64,
    max_relative_frobenius_distance: f64,
) SparseError!bool {
    const summary = try denseDiffSummary(T, lhs_values, rhs_values);
    return summary.meetsBounds(
        max_absolute_diff,
        max_relative_diff,
        max_squared_distance,
        max_frobenius_distance,
        max_relative_frobenius_distance,
    );
}

fn triangularIndexMatches(row: usize, col: usize, comptime strict: bool, comptime lower: bool) bool {
    return if (lower)
        if (strict) col < row else col <= row
    else if (strict) col > row else col >= row;
}

const SparseProfileBuilder = struct {
    // Profile metrics only need each logical row's left-most lower-triangle
    // coordinate and right-most upper-triangle coordinate.  Keeping this
    // scratch state shared across COO/CSR/CSC preserves identical semantics
    // even when compressed inputs are not sorted or contain duplicate entries.
    lower_found: []bool,
    upper_found: []bool,
    min_cols: []usize,
    max_cols: []usize,

    fn init(allocator: std.mem.Allocator, rows: usize) SparseError!SparseProfileBuilder {
        const lower_found = try allocator.alloc(bool, rows);
        errdefer allocator.free(lower_found);
        const upper_found = try allocator.alloc(bool, rows);
        errdefer allocator.free(upper_found);
        const min_cols = try allocator.alloc(usize, rows);
        errdefer allocator.free(min_cols);
        const max_cols = try allocator.alloc(usize, rows);
        errdefer allocator.free(max_cols);

        @memset(lower_found, false);
        @memset(upper_found, false);
        @memset(min_cols, 0);
        @memset(max_cols, 0);

        return .{
            .lower_found = lower_found,
            .upper_found = upper_found,
            .min_cols = min_cols,
            .max_cols = max_cols,
        };
    }

    fn deinit(self: *SparseProfileBuilder, allocator: std.mem.Allocator) void {
        allocator.free(self.lower_found);
        allocator.free(self.upper_found);
        allocator.free(self.min_cols);
        allocator.free(self.max_cols);
        self.* = undefined;
    }

    fn observe(self: *SparseProfileBuilder, row: usize, col: usize) void {
        if (col <= row and (!self.lower_found[row] or col < self.min_cols[row])) {
            self.lower_found[row] = true;
            self.min_cols[row] = col;
        }
        if (col >= row and (!self.upper_found[row] or col > self.max_cols[row])) {
            self.upper_found[row] = true;
            self.max_cols[row] = col;
        }
    }

    fn lowerProfile(self: SparseProfileBuilder) SparseError!usize {
        var total_profile: usize = 0;
        for (self.lower_found, self.min_cols, 0..) |found, min_col, row| {
            if (found) total_profile = std.math.add(usize, total_profile, row - min_col) catch return error.InvalidShape;
        }
        return total_profile;
    }

    fn upperProfile(self: SparseProfileBuilder) SparseError!usize {
        var total_profile: usize = 0;
        for (self.upper_found, self.max_cols, 0..) |found, max_col, row| {
            if (found) total_profile = std.math.add(usize, total_profile, max_col - row) catch return error.InvalidShape;
        }
        return total_profile;
    }

    fn profile(self: SparseProfileBuilder) SparseError!SparseProfile {
        return .{
            .lower = try self.lowerProfile(),
            .upper = try self.upperProfile(),
        };
    }
};

fn sparseVarianceFromSums(sum: f64, sum_sq: f64, count: usize, correction: f64) SparseError!f64 {
    if (count == 0) return error.EmptyArray;
    const count_float = sparseSizeToF64(count);
    if (correction < 0 or correction >= count_float) return error.InvalidShape;
    // Sparse matrices have implicit zeros.  `sum_sq` only needs stored values,
    // but the divisor and mean are over the full dense logical shape.
    return (sum_sq - (sum * sum) / count_float) / (count_float - correction);
}

fn finalizeVarianceArray(values: []f64, sums: []const f64, count: usize, correction: f64) SparseError!void {
    if (count == 0) return error.EmptyArray;
    const count_float = sparseSizeToF64(count);
    if (correction < 0 or correction >= count_float) return error.InvalidShape;
    for (values, sums) |*variance_value, sum_value| {
        variance_value.* = (variance_value.* - (sum_value * sum_value) / count_float) / (count_float - correction);
    }
}

fn sqrtArray(values: []f64) void {
    for (values) |*value| value.* = @sqrt(value.*);
}

fn diagonalExtent(diagonal_len: usize, offset: isize) SparseError!struct { size: usize, magnitude: usize, upper: bool } {
    const upper = offset >= 0;
    const magnitude: usize = if (upper)
        @intCast(offset)
    else
        @as(usize, @intCast(-(offset + 1))) + 1;
    const size = std.math.add(usize, diagonal_len, magnitude) catch return error.InvalidShape;
    return .{ .size = size, .magnitude = magnitude, .upper = upper };
}

fn toVeyraTriangle(triangle: Triangle) veyra.Triangle {
    return switch (triangle) {
        .lower => .lower,
        .upper => .upper,
    };
}

fn toVeyraDiagonal(diagonal: Diagonal) veyra.DiagonalKind {
    return switch (diagonal) {
        .non_unit => .non_unit,
        .unit => .unit,
    };
}

pub fn CooMatrix(comptime T: type) type {
    return struct {
        const Self = @This();
        const Entry = struct {
            row: usize,
            col: usize,
            value: T,
        };

        allocator: std.mem.Allocator,
        rows: usize,
        cols: usize,
        row_indices: []usize,
        col_indices: []usize,
        values: []T,

        fn entryLessThan(_: void, lhs: Entry, rhs: Entry) bool {
            return lhs.row < rhs.row or (lhs.row == rhs.row and lhs.col < rhs.col);
        }

        fn coordLess(lhs_row: usize, lhs_col: usize, rhs_row: usize, rhs_col: usize) bool {
            return lhs_row < rhs_row or (lhs_row == rhs_row and lhs_col < rhs_col);
        }

        pub fn eye(allocator: std.mem.Allocator, rows: usize, cols: usize) SparseError!Self {
            const diag_len = @min(rows, cols);
            var row_indices = try allocator.alloc(usize, diag_len);
            errdefer allocator.free(row_indices);
            var col_indices = try allocator.alloc(usize, diag_len);
            errdefer allocator.free(col_indices);
            var values = try allocator.alloc(T, diag_len);
            errdefer allocator.free(values);

            for (0..diag_len) |i| {
                row_indices[i] = i;
                col_indices[i] = i;
                values[i] = oneValue(T);
            }
            return .{
                .allocator = allocator,
                .rows = rows,
                .cols = cols,
                .row_indices = row_indices,
                .col_indices = col_indices,
                .values = values,
            };
        }

        pub fn identity(allocator: std.mem.Allocator, size: usize) SparseError!Self {
            return Self.eye(allocator, size, size);
        }

        pub fn fromDiagonal(allocator: std.mem.Allocator, diagonal_values: []const T, offset: isize) SparseError!Self {
            const extent = try diagonalExtent(diagonal_values.len, offset);
            const nonzero_count = try Self.fromDiagonalNnz(diagonal_values, offset);

            var row_indices = try allocator.alloc(usize, nonzero_count);
            errdefer allocator.free(row_indices);
            var col_indices = try allocator.alloc(usize, nonzero_count);
            errdefer allocator.free(col_indices);
            var values = try allocator.alloc(T, nonzero_count);
            errdefer allocator.free(values);

            var write: usize = 0;
            for (diagonal_values, 0..) |value, i| {
                if (isNonZero(T, value)) {
                    if (extent.upper) {
                        row_indices[write] = i;
                        col_indices[write] = i + extent.magnitude;
                    } else {
                        row_indices[write] = i + extent.magnitude;
                        col_indices[write] = i;
                    }
                    values[write] = value;
                    write += 1;
                }
            }
            std.debug.assert(write == nonzero_count);

            return .{
                .allocator = allocator,
                .rows = extent.size,
                .cols = extent.size,
                .row_indices = row_indices,
                .col_indices = col_indices,
                .values = values,
            };
        }

        pub fn fromDiagonalNnz(diagonal_values: []const T, offset: isize) SparseError!usize {
            return sparseDiagonalNnz(T, diagonal_values, offset);
        }

        pub fn fromDiagonalPruned(allocator: std.mem.Allocator, diagonal_values: []const T, offset: isize, tolerance: T) SparseError!Self {
            const extent = try diagonalExtent(diagonal_values.len, offset);
            const keep_count = try Self.fromDiagonalPrunedNnz(diagonal_values, offset, tolerance);

            var row_indices = try allocator.alloc(usize, keep_count);
            errdefer allocator.free(row_indices);
            var col_indices = try allocator.alloc(usize, keep_count);
            errdefer allocator.free(col_indices);
            var values = try allocator.alloc(T, keep_count);
            errdefer allocator.free(values);

            var write: usize = 0;
            for (diagonal_values, 0..) |value, i| {
                if (sparseAbsValueExceedsTolerance(T, value, tolerance)) {
                    if (extent.upper) {
                        row_indices[write] = i;
                        col_indices[write] = i + extent.magnitude;
                    } else {
                        row_indices[write] = i + extent.magnitude;
                        col_indices[write] = i;
                    }
                    values[write] = value;
                    write += 1;
                }
            }
            std.debug.assert(write == keep_count);

            return .{
                .allocator = allocator,
                .rows = extent.size,
                .cols = extent.size,
                .row_indices = row_indices,
                .col_indices = col_indices,
                .values = values,
            };
        }

        pub fn fromDiagonalPrunedNnz(diagonal_values: []const T, offset: isize, tolerance: T) SparseError!usize {
            return sparseDiagonalPrunedNnz(T, diagonal_values, offset, tolerance);
        }

        pub fn fromSlices(
            allocator: std.mem.Allocator,
            rows: usize,
            cols: usize,
            row_indices: []const usize,
            col_indices: []const usize,
            values: []const T,
        ) SparseError!Self {
            if (row_indices.len != col_indices.len or row_indices.len != values.len) return error.ShapeMismatch;
            for (row_indices) |row| if (row >= rows) return error.IndexOutOfBounds;
            for (col_indices) |col| if (col >= cols) return error.IndexOutOfBounds;
            return .{
                .allocator = allocator,
                .rows = rows,
                .cols = cols,
                .row_indices = try allocator.dupe(usize, row_indices),
                .col_indices = try allocator.dupe(usize, col_indices),
                .values = try allocator.dupe(T, values),
            };
        }

        pub fn fromDense(input: array_mod.Array(T)) SparseError!Self {
            if (input.shape.len != 2) return error.NonMatrixArray;
            const rows = input.shape[0];
            const cols = input.shape[1];
            const nonzero_count = try Self.fromDenseNnz(input);

            var row_indices = try input.allocator.alloc(usize, nonzero_count);
            errdefer input.allocator.free(row_indices);
            var col_indices = try input.allocator.alloc(usize, nonzero_count);
            errdefer input.allocator.free(col_indices);
            var values = try input.allocator.alloc(T, nonzero_count);
            errdefer input.allocator.free(values);

            var write: usize = 0;
            for (0..rows) |row| {
                for (0..cols) |col| {
                    const value = input.data[row * cols + col];
                    if (isNonZero(T, value)) {
                        row_indices[write] = row;
                        col_indices[write] = col;
                        values[write] = value;
                        write += 1;
                    }
                }
            }
            return .{
                .allocator = input.allocator,
                .rows = rows,
                .cols = cols,
                .row_indices = row_indices,
                .col_indices = col_indices,
                .values = values,
            };
        }

        pub fn fromDenseNnz(input: array_mod.Array(T)) SparseError!usize {
            return sparseDenseNnz(T, input);
        }

        pub fn fromDensePruned(input: array_mod.Array(T), tolerance: T) SparseError!Self {
            if (input.shape.len != 2) return error.NonMatrixArray;
            const rows = input.shape[0];
            const cols = input.shape[1];
            const keep_count = try Self.fromDensePrunedNnz(input, tolerance);

            var row_indices = try input.allocator.alloc(usize, keep_count);
            errdefer input.allocator.free(row_indices);
            var col_indices = try input.allocator.alloc(usize, keep_count);
            errdefer input.allocator.free(col_indices);
            var values = try input.allocator.alloc(T, keep_count);
            errdefer input.allocator.free(values);

            var write: usize = 0;
            for (0..rows) |row| {
                for (0..cols) |col| {
                    const value = input.data[row * cols + col];
                    if (sparseAbsValueExceedsTolerance(T, value, tolerance)) {
                        row_indices[write] = row;
                        col_indices[write] = col;
                        values[write] = value;
                        write += 1;
                    }
                }
            }
            std.debug.assert(write == keep_count);
            return .{
                .allocator = input.allocator,
                .rows = rows,
                .cols = cols,
                .row_indices = row_indices,
                .col_indices = col_indices,
                .values = values,
            };
        }

        pub fn fromDensePrunedNnz(input: array_mod.Array(T), tolerance: T) SparseError!usize {
            return sparseDensePrunedNnz(T, input, tolerance);
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.row_indices);
            self.allocator.free(self.col_indices);
            self.allocator.free(self.values);
            self.* = undefined;
        }

        pub fn fillStoredValues(self: *Self, value: T) void {
            @memset(self.values, value);
        }

        pub fn fillValues(self: *Self, value: T) void {
            self.fillStoredValues(value);
        }

        pub fn scaleStoredValues(self: *Self, alpha: T) void {
            ensureNumeric(T);
            for (self.values) |*value| value.* = value.* * alpha;
        }

        pub fn scaleValues(self: *Self, alpha: T) void {
            self.scaleStoredValues(alpha);
        }

        pub fn setStoredValuesFromDense(self: *Self, src: array_mod.Array(T)) SparseError!void {
            if (src.shape.len != 2) return error.NonMatrixArray;
            if (src.shape[0] != self.rows or src.shape[1] != self.cols) return error.ShapeMismatch;
            for (self.values, 0..) |*value, index| {
                value.* = src.data[self.row_indices[index] * self.cols + self.col_indices[index]];
            }
        }

        pub fn clone(self: Self) SparseError!Self {
            const row_indices = try self.allocator.dupe(usize, self.row_indices);
            errdefer self.allocator.free(row_indices);
            const col_indices = try self.allocator.dupe(usize, self.col_indices);
            errdefer self.allocator.free(col_indices);
            const values = try self.allocator.dupe(T, self.values);
            errdefer self.allocator.free(values);
            return .{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = self.cols,
                .row_indices = row_indices,
                .col_indices = col_indices,
                .values = values,
            };
        }

        pub fn copy(self: Self) SparseError!Self {
            return self.clone();
        }

        pub fn detach(self: Self) SparseError!Self {
            return self.clone();
        }

        pub fn astype(self: Self, comptime U: type) SparseError!CooMatrix(U) {
            const row_indices = try self.allocator.dupe(usize, self.row_indices);
            errdefer self.allocator.free(row_indices);
            const col_indices = try self.allocator.dupe(usize, self.col_indices);
            errdefer self.allocator.free(col_indices);
            var values = try self.allocator.alloc(U, self.values.len);
            errdefer self.allocator.free(values);
            for (self.values, 0..) |value, index| {
                values[index] = sparseCastValue(U, value);
            }
            return .{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = self.cols,
                .row_indices = row_indices,
                .col_indices = col_indices,
                .values = values,
            };
        }

        pub fn mapValues(self: Self, comptime U: type, comptime map: fn (T) U) SparseError!CooMatrix(U) {
            const row_indices = try self.allocator.dupe(usize, self.row_indices);
            errdefer self.allocator.free(row_indices);
            const col_indices = try self.allocator.dupe(usize, self.col_indices);
            errdefer self.allocator.free(col_indices);
            var values = try self.allocator.alloc(U, self.values.len);
            errdefer self.allocator.free(values);
            for (self.values, 0..) |value, index| {
                values[index] = map(value);
            }
            return .{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = self.cols,
                .row_indices = row_indices,
                .col_indices = col_indices,
                .values = values,
            };
        }

        pub fn mapStoredValues(self: Self, comptime U: type, comptime map: fn (T) U) SparseError!CooMatrix(U) {
            return self.mapValues(U, map);
        }

        pub fn mapValuesInPlace(self: *Self, comptime map: fn (T) T) void {
            for (self.values) |*value| value.* = map(value.*);
        }

        pub fn mapStoredValuesInPlace(self: *Self, comptime map: fn (T) T) void {
            self.mapValuesInPlace(map);
        }

        pub fn positive(self: Self) SparseError!Self {
            ensureNumeric(T);
            return self.clone();
        }

        pub fn abs(self: Self) SparseError!Self {
            ensureNumeric(T);
            const out = try self.clone();
            for (out.values) |*value| value.* = absValue(T, value.*);
            return out;
        }

        pub fn absolute(self: Self) SparseError!Self {
            return self.abs();
        }

        pub fn fabs(self: Self) SparseError!Self {
            return self.abs();
        }

        pub fn square(self: Self) SparseError!Self {
            ensureNumeric(T);
            const out = try self.clone();
            for (out.values) |*value| value.* = value.* * value.*;
            return out;
        }

        pub fn sqrt(self: Self) SparseError!Self {
            ensureFloat(T);
            const out = try self.clone();
            for (out.values) |*value| value.* = @sqrt(value.*);
            return out;
        }

        pub fn rsqrt(self: Self) SparseError!Self {
            ensureFloat(T);
            const out = try self.clone();
            for (out.values) |*value| value.* = oneValue(T) / @sqrt(value.*);
            return out;
        }

        pub fn cbrt(self: Self) SparseError!Self {
            ensureFloat(T);
            const out = try self.clone();
            for (out.values) |*value| value.* = cbrtSparseValue(T, value.*);
            return out;
        }

        pub fn reciprocal(self: Self) SparseError!Self {
            ensureFloat(T);
            const out = try self.clone();
            for (out.values) |*value| value.* = oneValue(T) / value.*;
            return out;
        }

        pub fn floor(self: Self) SparseError!Self {
            ensureNumeric(T);
            const out = try self.clone();
            for (out.values) |*value| value.* = floorSparseValue(T, value.*);
            return out;
        }

        pub fn ceil(self: Self) SparseError!Self {
            ensureNumeric(T);
            const out = try self.clone();
            for (out.values) |*value| value.* = ceilSparseValue(T, value.*);
            return out;
        }

        pub fn round(self: Self) SparseError!Self {
            ensureNumeric(T);
            const out = try self.clone();
            for (out.values) |*value| value.* = roundSparseValue(T, value.*);
            return out;
        }

        pub fn trunc(self: Self) SparseError!Self {
            ensureNumeric(T);
            const out = try self.clone();
            for (out.values) |*value| value.* = truncSparseValue(T, value.*);
            return out;
        }

        pub fn clipMin(self: Self, min_value: T) SparseError!Self {
            ensureNumeric(T);
            const out = try self.clone();
            for (out.values) |*value| {
                if (value.* < min_value) value.* = min_value;
            }
            return out;
        }

        pub fn maximumScalar(self: Self, scalar: T) SparseError!Self {
            return self.clipMin(scalar);
        }

        pub fn maxScalar(self: Self, scalar: T) SparseError!Self {
            return self.maximumScalar(scalar);
        }

        pub fn clampMin(self: Self, min_value: T) SparseError!Self {
            return self.clipMin(min_value);
        }

        pub fn clipMax(self: Self, max_value: T) SparseError!Self {
            ensureNumeric(T);
            const out = try self.clone();
            for (out.values) |*value| {
                if (value.* > max_value) value.* = max_value;
            }
            return out;
        }

        pub fn minimumScalar(self: Self, scalar: T) SparseError!Self {
            return self.clipMax(scalar);
        }

        pub fn minScalar(self: Self, scalar: T) SparseError!Self {
            return self.minimumScalar(scalar);
        }

        pub fn clampMax(self: Self, max_value: T) SparseError!Self {
            return self.clipMax(max_value);
        }

        pub fn clip(self: Self, min_value: T, max_value: T) SparseError!Self {
            try validateSparseValueRange(T, min_value, max_value);
            const out = try self.clone();
            for (out.values) |*value| {
                if (value.* < min_value) {
                    value.* = min_value;
                } else if (value.* > max_value) {
                    value.* = max_value;
                }
            }
            return out;
        }

        pub fn clamp(self: Self, min_value: T, max_value: T) SparseError!Self {
            return self.clip(min_value, max_value);
        }

        pub fn sign(self: Self) SparseError!Self {
            ensureNumeric(T);
            const out = try self.clone();
            for (out.values) |*value| value.* = signSparseValue(T, value.*);
            return out;
        }

        pub fn signbit(self: Self) SparseError!CooMatrix(bool) {
            ensureNumeric(T);
            const row_indices = try self.allocator.dupe(usize, self.row_indices);
            errdefer self.allocator.free(row_indices);
            const col_indices = try self.allocator.dupe(usize, self.col_indices);
            errdefer self.allocator.free(col_indices);
            var values = try self.allocator.alloc(bool, self.values.len);
            errdefer self.allocator.free(values);
            for (self.values, 0..) |value, index| {
                values[index] = signbitSparseValue(T, value);
            }
            return .{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = self.cols,
                .row_indices = row_indices,
                .col_indices = col_indices,
                .values = values,
            };
        }

        fn finitePredicateMask(self: Self, comptime predicate: SparseFinitePredicate) SparseError!CooMatrix(bool) {
            return self.mapValues(bool, struct {
                fn f(value: T) bool {
                    return sparseFinitePredicateValue(T, value, predicate);
                }
            }.f);
        }

        pub fn isNan(self: Self) SparseError!CooMatrix(bool) {
            return self.finitePredicateMask(.nan);
        }

        pub fn isnan(self: Self) SparseError!CooMatrix(bool) {
            return self.isNan();
        }

        pub fn isInf(self: Self) SparseError!CooMatrix(bool) {
            return self.finitePredicateMask(.inf);
        }

        pub fn isinf(self: Self) SparseError!CooMatrix(bool) {
            return self.isInf();
        }

        pub fn isPosInf(self: Self) SparseError!CooMatrix(bool) {
            return self.finitePredicateMask(.pos_inf);
        }

        pub fn isposinf(self: Self) SparseError!CooMatrix(bool) {
            return self.isPosInf();
        }

        pub fn isNegInf(self: Self) SparseError!CooMatrix(bool) {
            return self.finitePredicateMask(.neg_inf);
        }

        pub fn isneginf(self: Self) SparseError!CooMatrix(bool) {
            return self.isNegInf();
        }

        pub fn isFinite(self: Self) SparseError!CooMatrix(bool) {
            return self.finitePredicateMask(.finite);
        }

        pub fn isfinite(self: Self) SparseError!CooMatrix(bool) {
            return self.isFinite();
        }

        pub fn isNormal(self: Self) SparseError!CooMatrix(bool) {
            return self.finitePredicateMask(.normal);
        }

        pub fn isnormal(self: Self) SparseError!CooMatrix(bool) {
            return self.isNormal();
        }

        fn compareScalar(self: Self, scalar: T, comptime comparison: SparseScalarComparison) SparseError!CooMatrix(bool) {
            const row_indices = try self.allocator.dupe(usize, self.row_indices);
            errdefer self.allocator.free(row_indices);
            const col_indices = try self.allocator.dupe(usize, self.col_indices);
            errdefer self.allocator.free(col_indices);
            const values = try sparseCompareScalarValues(T, self.allocator, self.values, scalar, comparison);
            errdefer self.allocator.free(values);
            return .{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = self.cols,
                .row_indices = row_indices,
                .col_indices = col_indices,
                .values = values,
            };
        }

        pub fn equalScalar(self: Self, scalar: T) SparseError!CooMatrix(bool) {
            return self.compareScalar(scalar, .eq);
        }

        pub fn eqScalar(self: Self, scalar: T) SparseError!CooMatrix(bool) {
            return self.equalScalar(scalar);
        }

        pub fn notEqualScalar(self: Self, scalar: T) SparseError!CooMatrix(bool) {
            return self.compareScalar(scalar, .ne);
        }

        pub fn neScalar(self: Self, scalar: T) SparseError!CooMatrix(bool) {
            return self.notEqualScalar(scalar);
        }

        pub fn greaterScalar(self: Self, scalar: T) SparseError!CooMatrix(bool) {
            return self.compareScalar(scalar, .gt);
        }

        pub fn gtScalar(self: Self, scalar: T) SparseError!CooMatrix(bool) {
            return self.greaterScalar(scalar);
        }

        pub fn greaterEqualScalar(self: Self, scalar: T) SparseError!CooMatrix(bool) {
            return self.compareScalar(scalar, .ge);
        }

        pub fn geScalar(self: Self, scalar: T) SparseError!CooMatrix(bool) {
            return self.greaterEqualScalar(scalar);
        }

        pub fn lessScalar(self: Self, scalar: T) SparseError!CooMatrix(bool) {
            return self.compareScalar(scalar, .lt);
        }

        pub fn ltScalar(self: Self, scalar: T) SparseError!CooMatrix(bool) {
            return self.lessScalar(scalar);
        }

        pub fn lessEqualScalar(self: Self, scalar: T) SparseError!CooMatrix(bool) {
            return self.compareScalar(scalar, .le);
        }

        pub fn leScalar(self: Self, scalar: T) SparseError!CooMatrix(bool) {
            return self.lessEqualScalar(scalar);
        }

        fn compareSameStructure(self: Self, rhs: Self, comptime comparison: SparseScalarComparison) SparseError!CooMatrix(bool) {
            if (self.rows != rhs.rows or self.cols != rhs.cols or self.values.len != rhs.values.len) return error.ShapeMismatch;
            if (!self.sameStructure(rhs)) return error.InvalidShape;
            const row_indices = try self.allocator.dupe(usize, self.row_indices);
            errdefer self.allocator.free(row_indices);
            const col_indices = try self.allocator.dupe(usize, self.col_indices);
            errdefer self.allocator.free(col_indices);
            const values = try sparseCompareSameStructureValues(T, self.allocator, self.values, rhs.values, comparison);
            errdefer self.allocator.free(values);
            return .{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = self.cols,
                .row_indices = row_indices,
                .col_indices = col_indices,
                .values = values,
            };
        }

        pub fn equalSameStructure(self: Self, rhs: Self) SparseError!CooMatrix(bool) {
            return self.compareSameStructure(rhs, .eq);
        }

        pub fn eqSameStructure(self: Self, rhs: Self) SparseError!CooMatrix(bool) {
            return self.equalSameStructure(rhs);
        }

        pub fn notEqualSameStructure(self: Self, rhs: Self) SparseError!CooMatrix(bool) {
            return self.compareSameStructure(rhs, .ne);
        }

        pub fn neSameStructure(self: Self, rhs: Self) SparseError!CooMatrix(bool) {
            return self.notEqualSameStructure(rhs);
        }

        pub fn greaterSameStructure(self: Self, rhs: Self) SparseError!CooMatrix(bool) {
            return self.compareSameStructure(rhs, .gt);
        }

        pub fn gtSameStructure(self: Self, rhs: Self) SparseError!CooMatrix(bool) {
            return self.greaterSameStructure(rhs);
        }

        pub fn greaterEqualSameStructure(self: Self, rhs: Self) SparseError!CooMatrix(bool) {
            return self.compareSameStructure(rhs, .ge);
        }

        pub fn geSameStructure(self: Self, rhs: Self) SparseError!CooMatrix(bool) {
            return self.greaterEqualSameStructure(rhs);
        }

        pub fn lessSameStructure(self: Self, rhs: Self) SparseError!CooMatrix(bool) {
            return self.compareSameStructure(rhs, .lt);
        }

        pub fn ltSameStructure(self: Self, rhs: Self) SparseError!CooMatrix(bool) {
            return self.lessSameStructure(rhs);
        }

        pub fn lessEqualSameStructure(self: Self, rhs: Self) SparseError!CooMatrix(bool) {
            return self.compareSameStructure(rhs, .le);
        }

        pub fn leSameStructure(self: Self, rhs: Self) SparseError!CooMatrix(bool) {
            return self.lessEqualSameStructure(rhs);
        }

        pub fn nnz(self: Self) usize {
            return self.values.len;
        }

        pub fn countNonzero(self: Self) SparseError!usize {
            return sparseDenseCountNonzero(self);
        }

        pub fn countNonzeroAxis(self: Self, axis_opt: ?isize, keepdims: bool) SparseError!array_mod.Array(usize) {
            return sparseDenseCountNonzeroAxis(self, axis_opt, keepdims);
        }

        pub fn countNonzeroAxes(self: Self, axes: []const isize, keepdims: bool) SparseError!array_mod.Array(usize) {
            return sparseDenseCountNonzeroAxes(self, axes, keepdims);
        }

        pub fn countNonzeroDim(self: Self, dim_opt: ?isize, keepdim: bool) SparseError!array_mod.Array(usize) {
            return self.countNonzeroAxis(dim_opt, keepdim);
        }

        pub fn countNonzeroDims(self: Self, dims: []const isize, keepdim: bool) SparseError!array_mod.Array(usize) {
            return self.countNonzeroAxes(dims, keepdim);
        }

        pub fn flatNonzero(self: Self) SparseError!array_mod.Array(usize) {
            return sparseDenseFlatNonzero(self);
        }

        pub fn nonzero(self: Self) SparseError!array_mod.Array(usize) {
            return sparseDenseNonzero(self);
        }

        pub fn argwhere(self: Self) SparseError!array_mod.Array(usize) {
            return self.nonzero();
        }

        pub fn whereIndices(self: Self) SparseError!array_mod.Array(usize) {
            return self.nonzero();
        }

        pub fn all(self: Self) SparseError!bool {
            return sparseDenseAll(self);
        }

        pub fn any(self: Self) SparseError!bool {
            return sparseDenseAny(self);
        }

        pub fn allAxis(self: Self, axis_opt: ?isize, keepdims: bool) SparseError!array_mod.Array(bool) {
            return sparseDenseAllAxis(self, axis_opt, keepdims);
        }

        pub fn allAxes(self: Self, axes: []const isize, keepdims: bool) SparseError!array_mod.Array(bool) {
            return sparseDenseAllAxes(self, axes, keepdims);
        }

        pub fn allDim(self: Self, dim_opt: ?isize, keepdim: bool) SparseError!array_mod.Array(bool) {
            return self.allAxis(dim_opt, keepdim);
        }

        pub fn allDims(self: Self, dims: []const isize, keepdim: bool) SparseError!array_mod.Array(bool) {
            return self.allAxes(dims, keepdim);
        }

        pub fn anyAxis(self: Self, axis_opt: ?isize, keepdims: bool) SparseError!array_mod.Array(bool) {
            return sparseDenseAnyAxis(self, axis_opt, keepdims);
        }

        pub fn anyAxes(self: Self, axes: []const isize, keepdims: bool) SparseError!array_mod.Array(bool) {
            return sparseDenseAnyAxes(self, axes, keepdims);
        }

        pub fn anyDim(self: Self, dim_opt: ?isize, keepdim: bool) SparseError!array_mod.Array(bool) {
            return self.anyAxis(dim_opt, keepdim);
        }

        pub fn anyDims(self: Self, dims: []const isize, keepdim: bool) SparseError!array_mod.Array(bool) {
            return self.anyAxes(dims, keepdim);
        }

        pub fn logicalNot(self: Self) SparseError!array_mod.Array(bool) {
            return sparseDenseLogicalNot(self);
        }

        pub fn logicalAnd(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return sparseDenseLogicalBinary(self, rhs, .and_);
        }

        pub fn logicalAndScalar(self: Self, scalar: bool) SparseError!array_mod.Array(bool) {
            return sparseDenseLogicalScalar(self, scalar, .and_);
        }

        pub fn logicalOr(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return sparseDenseLogicalBinary(self, rhs, .or_);
        }

        pub fn logicalOrScalar(self: Self, scalar: bool) SparseError!array_mod.Array(bool) {
            return sparseDenseLogicalScalar(self, scalar, .or_);
        }

        pub fn logicalXor(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return sparseDenseLogicalBinary(self, rhs, .xor_);
        }

        pub fn logicalXorScalar(self: Self, scalar: bool) SparseError!array_mod.Array(bool) {
            return sparseDenseLogicalScalar(self, scalar, .xor_);
        }

        pub fn maskedSelect(self: Self, mask: array_mod.Array(bool)) SparseError!array_mod.Array(T) {
            return sparseDenseMaskedSelect(T, self, mask);
        }

        pub fn maskedFill(self: Self, mask: array_mod.Array(bool), value: T) SparseError!array_mod.Array(T) {
            return sparseDenseMaskedFill(T, self, mask, value);
        }

        pub fn maskedPutScalar(self: Self, mask: array_mod.Array(bool), value: T) SparseError!array_mod.Array(T) {
            return self.maskedFill(mask, value);
        }

        pub fn putMaskScalar(self: Self, mask: array_mod.Array(bool), value: T) SparseError!array_mod.Array(T) {
            return self.maskedPutScalar(mask, value);
        }

        pub fn maskedScatter(self: Self, mask: array_mod.Array(bool), src: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            return sparseDenseMaskedScatter(T, self, mask, src);
        }

        pub fn maskedPut(self: Self, mask: array_mod.Array(bool), values: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            return sparseDenseMaskedPut(T, self, mask, values);
        }

        pub fn putMask(self: Self, mask: array_mod.Array(bool), values: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            return self.maskedPut(mask, values);
        }

        pub fn take(self: Self, indices: array_mod.Array(usize), axis_opt: ?isize) SparseError!array_mod.Array(T) {
            return sparseDenseTake(T, self, indices, axis_opt);
        }

        pub fn takeSigned(self: Self, indices: array_mod.Array(isize), axis_opt: ?isize) SparseError!array_mod.Array(T) {
            return sparseDenseTakeSigned(T, self, indices, axis_opt);
        }

        pub fn takeMode(self: Self, indices: array_mod.Array(usize), axis_opt: ?isize, mode: array_mod.IndexMode) SparseError!array_mod.Array(T) {
            return sparseDenseTakeMode(T, self, indices, axis_opt, mode);
        }

        pub fn takeSignedMode(self: Self, indices: array_mod.Array(isize), axis_opt: ?isize, mode: array_mod.IndexMode) SparseError!array_mod.Array(T) {
            return sparseDenseTakeSignedMode(T, self, indices, axis_opt, mode);
        }

        pub fn indexSelect(self: Self, axis_index: isize, indices: array_mod.Array(usize)) SparseError!array_mod.Array(T) {
            return self.take(indices, axis_index);
        }

        pub fn indexSelectSigned(self: Self, axis_index: isize, indices: array_mod.Array(isize)) SparseError!array_mod.Array(T) {
            return self.takeSigned(indices, axis_index);
        }

        pub fn gather(self: Self, axis_index: isize, indices: array_mod.Array(usize)) SparseError!array_mod.Array(T) {
            return sparseDenseGather(T, self, axis_index, indices);
        }

        pub fn gatherSigned(self: Self, axis_index: isize, indices: array_mod.Array(isize)) SparseError!array_mod.Array(T) {
            return sparseDenseGatherSigned(T, self, axis_index, indices);
        }

        pub fn takeAlongAxis(self: Self, indices: array_mod.Array(usize), axis_index: isize) SparseError!array_mod.Array(T) {
            return self.gather(axis_index, indices);
        }

        pub fn takeAlongAxisSigned(self: Self, indices: array_mod.Array(isize), axis_index: isize) SparseError!array_mod.Array(T) {
            return self.gatherSigned(axis_index, indices);
        }

        pub fn scatter(self: Self, axis_index: isize, indices: array_mod.Array(usize), src: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            return sparseDenseScatter(T, self, axis_index, indices, src);
        }

        pub fn scatterScalar(self: Self, axis_index: isize, indices: array_mod.Array(usize), value: T) SparseError!array_mod.Array(T) {
            return sparseDenseScatterScalar(T, self, axis_index, indices, value);
        }

        pub fn putAlongAxis(self: Self, indices: array_mod.Array(usize), src: array_mod.Array(T), axis_index: isize) SparseError!array_mod.Array(T) {
            return self.scatter(axis_index, indices, src);
        }

        pub fn select(self: Self, axis_index: isize, index: usize) SparseError!array_mod.Array(T) {
            return sparseDenseSelect(T, self, axis_index, index);
        }

        pub fn selectSigned(self: Self, axis_index: isize, index: isize) SparseError!array_mod.Array(T) {
            return sparseDenseSelectSigned(T, self, axis_index, index);
        }

        pub fn narrow(self: Self, axis_index: isize, start: usize, length: usize) SparseError!array_mod.Array(T) {
            return sparseDenseNarrow(T, self, axis_index, start, length);
        }

        pub fn narrowSigned(self: Self, axis_index: isize, start: isize, length: usize) SparseError!array_mod.Array(T) {
            return sparseDenseNarrowSigned(T, self, axis_index, start, length);
        }

        pub fn scatterReduce(self: Self, axis_index: isize, indices: array_mod.Array(usize), src: array_mod.Array(T), reduction: array_mod.ScatterReduce) SparseError!array_mod.Array(T) {
            return sparseDenseScatterReduce(T, self, axis_index, indices, src, reduction);
        }

        pub fn scatterAdd(self: Self, axis_index: isize, indices: array_mod.Array(usize), src: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            return self.scatterReduce(axis_index, indices, src, .sum);
        }

        pub fn scatterReduceScalar(self: Self, axis_index: isize, indices: array_mod.Array(usize), value: T, reduction: array_mod.ScatterReduce) SparseError!array_mod.Array(T) {
            return sparseDenseScatterReduceScalar(T, self, axis_index, indices, value, reduction);
        }

        pub fn putFlat(self: Self, indices: array_mod.Array(usize), values: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            return sparseDensePutFlat(T, self, indices, values);
        }

        pub fn putFlatMode(self: Self, indices: array_mod.Array(usize), values: array_mod.Array(T), mode: array_mod.IndexMode) SparseError!array_mod.Array(T) {
            return sparseDensePutFlatMode(T, self, indices, values, mode);
        }

        pub fn putFlatScalar(self: Self, indices: array_mod.Array(usize), value: T) SparseError!array_mod.Array(T) {
            return sparseDensePutFlatScalar(T, self, indices, value);
        }

        pub fn putFlatScalarMode(self: Self, indices: array_mod.Array(usize), value: T, mode: array_mod.IndexMode) SparseError!array_mod.Array(T) {
            return sparseDensePutFlatScalarMode(T, self, indices, value, mode);
        }

        pub fn putFlatSigned(self: Self, indices: array_mod.Array(isize), values: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            return sparseDensePutFlatSigned(T, self, indices, values);
        }

        pub fn putFlatScalarSigned(self: Self, indices: array_mod.Array(isize), value: T) SparseError!array_mod.Array(T) {
            return sparseDensePutFlatScalarSigned(T, self, indices, value);
        }

        pub fn ravelCoords(self: Self, coords: array_mod.Array(usize)) SparseError!array_mod.Array(usize) {
            return sparseDenseRavelCoords(self, coords);
        }

        pub fn unravelFlat(self: Self, indices: array_mod.Array(usize)) SparseError!array_mod.Array(usize) {
            return sparseDenseUnravelFlat(self, indices);
        }

        pub fn takeCoords(self: Self, coords: array_mod.Array(usize)) SparseError!array_mod.Array(T) {
            return sparseDenseTakeCoords(T, self, coords);
        }

        pub fn putCoords(self: Self, coords: array_mod.Array(usize), values: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            return sparseDensePutCoords(T, self, coords, values);
        }

        pub fn putCoordsScalar(self: Self, coords: array_mod.Array(usize), value: T) SparseError!array_mod.Array(T) {
            return sparseDensePutCoordsScalar(T, self, coords, value);
        }

        pub fn reshape(self: Self, dims: []const usize) SparseError!array_mod.Array(T) {
            return sparseDenseReshape(T, self, dims);
        }

        pub fn reshapeInfer(self: Self, dims: []const isize) SparseError!array_mod.Array(T) {
            return sparseDenseReshapeInfer(T, self, dims);
        }

        pub fn reshapeAs(self: Self, other: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            return self.reshape(other.shape);
        }

        pub fn view(self: Self, dims: []const usize) SparseError!array_mod.Array(T) {
            return self.reshape(dims);
        }

        pub fn viewInfer(self: Self, dims: []const isize) SparseError!array_mod.Array(T) {
            return self.reshapeInfer(dims);
        }

        pub fn viewAs(self: Self, other: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            return self.view(other.shape);
        }

        pub fn flatten(self: Self) SparseError!array_mod.Array(T) {
            return sparseDenseFlatten(T, self);
        }

        pub fn flattenAxes(self: Self, start_axis: isize, end_axis: isize) SparseError!array_mod.Array(T) {
            return sparseDenseFlattenAxes(T, self, start_axis, end_axis);
        }

        pub fn flattenRange(self: Self, start_axis: isize, end_axis: isize) SparseError!array_mod.Array(T) {
            return self.flattenAxes(start_axis, end_axis);
        }

        pub fn flattenFrom(self: Self, start_axis: isize) SparseError!array_mod.Array(T) {
            return self.flattenAxes(start_axis, -1);
        }

        pub fn ravel(self: Self) SparseError!array_mod.Array(T) {
            return self.flatten();
        }

        pub fn atLeast1d(self: Self) SparseError!array_mod.Array(T) {
            return sparseDenseAtLeast1d(T, self);
        }

        pub fn atLeast2d(self: Self) SparseError!array_mod.Array(T) {
            return sparseDenseAtLeast2d(T, self);
        }

        pub fn atLeast3d(self: Self) SparseError!array_mod.Array(T) {
            return sparseDenseAtLeast3d(T, self);
        }

        pub fn unflatten(self: Self, axis_index: isize, dims: []const usize) SparseError!array_mod.Array(T) {
            return sparseDenseUnflatten(T, self, axis_index, dims);
        }

        pub fn squeeze(self: Self, axis_opt: ?isize) SparseError!array_mod.Array(T) {
            return sparseDenseSqueeze(T, self, axis_opt);
        }

        pub fn squeezeDim(self: Self, axis_index: isize) SparseError!array_mod.Array(T) {
            return self.squeeze(axis_index);
        }

        pub fn unsqueeze(self: Self, axis_index: isize) SparseError!array_mod.Array(T) {
            return sparseDenseUnsqueeze(T, self, axis_index);
        }

        pub fn unsqueezeDim(self: Self, axis_index: isize) SparseError!array_mod.Array(T) {
            return self.unsqueeze(axis_index);
        }

        pub fn squeezeAxes(self: Self, axes: []const isize) SparseError!array_mod.Array(T) {
            return sparseDenseSqueezeAxes(T, self, axes);
        }

        pub fn unsqueezeAxes(self: Self, axes: []const isize) SparseError!array_mod.Array(T) {
            return sparseDenseUnsqueezeAxes(T, self, axes);
        }

        pub fn expandDims(self: Self, axes: []const isize) SparseError!array_mod.Array(T) {
            return self.unsqueezeAxes(axes);
        }

        pub fn broadcastTo(self: Self, dims: []const usize) SparseError!array_mod.Array(T) {
            return sparseDenseBroadcastTo(T, self, dims);
        }

        pub fn expand(self: Self, dims: []const usize) SparseError!array_mod.Array(T) {
            return self.broadcastTo(dims);
        }

        pub fn expandAs(self: Self, other: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            return self.expand(other.shape);
        }

        pub fn broadcastAs(self: Self, other: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            return self.expandAs(other);
        }

        pub fn permute(self: Self, axes: []const usize) SparseError!array_mod.Array(T) {
            return sparseDensePermute(T, self, axes);
        }

        pub fn swapaxes(self: Self, dim0: isize, dim1: isize) SparseError!array_mod.Array(T) {
            return sparseDenseSwapAxes(T, self, dim0, dim1);
        }

        pub fn swapDims(self: Self, dim0: isize, dim1: isize) SparseError!array_mod.Array(T) {
            return self.swapaxes(dim0, dim1);
        }

        pub fn movedim(self: Self, source: isize, destination: isize) SparseError!array_mod.Array(T) {
            return sparseDenseMoveDim(T, self, source, destination);
        }

        pub fn moveaxis(self: Self, source: isize, destination: isize) SparseError!array_mod.Array(T) {
            return self.movedim(source, destination);
        }

        pub fn flip(self: Self, axis_index: isize) SparseError!array_mod.Array(T) {
            return sparseDenseFlip(T, self, axis_index);
        }

        pub fn flipud(self: Self) SparseError!array_mod.Array(T) {
            return self.flip(0);
        }

        pub fn fliplr(self: Self) SparseError!array_mod.Array(T) {
            return self.flip(1);
        }

        pub fn flipAxes(self: Self, axes: []const isize) SparseError!array_mod.Array(T) {
            return sparseDenseFlipAxes(T, self, axes);
        }

        pub fn roll(self: Self, shift: isize, axis_index: isize) SparseError!array_mod.Array(T) {
            return sparseDenseRoll(T, self, shift, axis_index);
        }

        pub fn rollFlat(self: Self, shift: isize) SparseError!array_mod.Array(T) {
            return sparseDenseRollFlat(T, self, shift);
        }

        pub fn rollAxes(self: Self, shifts: []const isize, axes: []const isize) SparseError!array_mod.Array(T) {
            return sparseDenseRollAxes(T, self, shifts, axes);
        }

        pub fn rot90(self: Self, k: isize, axes: [2]isize) SparseError!array_mod.Array(T) {
            return sparseDenseRot90(T, self, k, axes);
        }

        pub fn padConstant(self: Self, before: []const usize, after: []const usize, value: T) SparseError!array_mod.Array(T) {
            return sparseDensePadConstant(T, self, before, after, value);
        }

        pub fn padEdge(self: Self, before: []const usize, after: []const usize) SparseError!array_mod.Array(T) {
            return sparseDensePadEdge(T, self, before, after);
        }

        pub fn padReflect(self: Self, before: []const usize, after: []const usize) SparseError!array_mod.Array(T) {
            return sparseDensePadReflect(T, self, before, after);
        }

        pub fn padWrap(self: Self, before: []const usize, after: []const usize) SparseError!array_mod.Array(T) {
            return sparseDensePadWrap(T, self, before, after);
        }

        pub fn padSymmetric(self: Self, before: []const usize, after: []const usize) SparseError!array_mod.Array(T) {
            return sparseDensePadSymmetric(T, self, before, after);
        }

        pub fn split(self: Self, split_size: usize, axis_index: isize) SparseError!array_mod.Array(T).SplitResult {
            return sparseDenseSplit(T, self, split_size, axis_index);
        }

        pub fn splitWithSizes(self: Self, sizes: []const usize, axis_index: isize) SparseError!array_mod.Array(T).SplitResult {
            return sparseDenseSplitWithSizes(T, self, sizes, axis_index);
        }

        pub fn splitAtIndices(self: Self, indices: []const usize, axis_index: isize) SparseError!array_mod.Array(T).SplitResult {
            return sparseDenseSplitAtIndices(T, self, indices, axis_index);
        }

        pub fn chunk(self: Self, chunks: usize, axis_index: isize) SparseError!array_mod.Array(T).SplitResult {
            return sparseDenseChunk(T, self, chunks, axis_index);
        }

        pub fn unbind(self: Self, axis_index: isize) SparseError!array_mod.Array(T).SplitResult {
            return sparseDenseUnbind(T, self, axis_index);
        }

        pub fn repeat(self: Self, repeats: usize, axis_index: isize) SparseError!array_mod.Array(T) {
            return sparseDenseRepeat(T, self, repeats, axis_index);
        }

        pub fn repeatInterleave(self: Self, repeats: array_mod.Array(usize), axis_opt: ?isize) SparseError!array_mod.Array(T) {
            return sparseDenseRepeatInterleave(T, self, repeats, axis_opt);
        }

        pub fn repeatInterleaveScalar(self: Self, repeat_count: usize, axis_opt: ?isize) SparseError!array_mod.Array(T) {
            return sparseDenseRepeatInterleaveScalar(T, self, repeat_count, axis_opt);
        }

        pub fn tile(self: Self, repeats: []const usize) SparseError!array_mod.Array(T) {
            return sparseDenseTile(T, self, repeats);
        }

        pub fn indexPut(self: Self, indices: array_mod.Array(usize), values: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            return self.putFlat(indices, values);
        }

        pub fn indexPutScalar(self: Self, indices: array_mod.Array(usize), value: T) SparseError!array_mod.Array(T) {
            return self.putFlatScalar(indices, value);
        }

        pub fn compress(self: Self, condition: array_mod.Array(bool), axis_opt: ?isize) SparseError!array_mod.Array(T) {
            return sparseDenseCompress(T, self, condition, axis_opt);
        }

        pub fn where(self: Self, mask: array_mod.Array(bool), other: Self) SparseError!array_mod.Array(T) {
            return sparseDenseWhere(T, self, mask, other);
        }

        pub fn whereArray(self: Self, mask: array_mod.Array(bool), other: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            return sparseDenseWhereArray(T, self, mask, other);
        }

        pub fn whereScalar(self: Self, mask: array_mod.Array(bool), other_value: T) SparseError!array_mod.Array(T) {
            return sparseDenseWhereScalar(T, self, mask, other_value);
        }

        pub fn copyWhere(self: Self, mask: array_mod.Array(bool), src: Self) SparseError!array_mod.Array(T) {
            return sparseDenseWhere(T, src, mask, self);
        }

        pub fn sameStructure(self: Self, rhs: Self) bool {
            return self.rows == rhs.rows and
                self.cols == rhs.cols and
                std.mem.eql(usize, self.row_indices, rhs.row_indices) and
                std.mem.eql(usize, self.col_indices, rhs.col_indices);
        }

        pub fn dotSameStructure(self: Self, rhs: Self) SparseError!T {
            if (self.rows != rhs.rows or self.cols != rhs.cols or self.values.len != rhs.values.len) return error.ShapeMismatch;
            if (!self.sameStructure(rhs)) return error.InvalidShape;
            return sparseDotSameStructure(T, self.values, rhs.values);
        }

        pub fn sameStructureDiffSummary(self: Self, rhs: Self) SparseError!SparseDiffSummary {
            if (self.rows != rhs.rows or self.cols != rhs.cols or self.values.len != rhs.values.len) return error.ShapeMismatch;
            if (!self.sameStructure(rhs)) return error.InvalidShape;
            return sparseSameStructureDiffSummary(T, self.values, rhs.values);
        }

        pub fn diffSummary(self: Self, rhs: Self) SparseError!SparseDiffSummary {
            if (self.rows != rhs.rows or self.cols != rhs.cols) return error.ShapeMismatch;
            var lhs_dense = try self.toDense();
            defer lhs_dense.deinit();
            var rhs_dense = try rhs.toDense();
            defer rhs_dense.deinit();
            return denseDiffSummary(T, lhs_dense.data, rhs_dense.data);
        }

        pub fn diffSummaryMeetsBounds(
            self: Self,
            rhs: Self,
            max_absolute_diff: f64,
            max_relative_diff: f64,
            max_squared_distance: f64,
            max_frobenius_distance: f64,
            max_relative_frobenius_distance: f64,
        ) SparseError!bool {
            if (self.rows != rhs.rows or self.cols != rhs.cols) return error.ShapeMismatch;
            var lhs_dense = try self.toDense();
            defer lhs_dense.deinit();
            var rhs_dense = try rhs.toDense();
            defer rhs_dense.deinit();
            return denseDiffSummaryMeetsBounds(
                T,
                lhs_dense.data,
                rhs_dense.data,
                max_absolute_diff,
                max_relative_diff,
                max_squared_distance,
                max_frobenius_distance,
                max_relative_frobenius_distance,
            );
        }

        pub fn maxAbsDiff(self: Self, rhs: Self) SparseError!f64 {
            return (try self.diffSummary(rhs)).max_abs_diff;
        }

        pub fn maxRelDiff(self: Self, rhs: Self) SparseError!f64 {
            return (try self.diffSummary(rhs)).max_rel_diff;
        }

        pub fn squaredDistance(self: Self, rhs: Self) SparseError!f64 {
            return (try self.diffSummary(rhs)).squared_distance;
        }

        pub fn frobeniusDistance(self: Self, rhs: Self) SparseError!f64 {
            return (try self.diffSummary(rhs)).frobeniusDistance();
        }

        pub fn relativeFrobeniusDistance(self: Self, rhs: Self) SparseError!f64 {
            return (try self.diffSummary(rhs)).relativeFrobeniusDistance();
        }

        pub fn maxAbsDiffMeetsBound(self: Self, rhs: Self, max_absolute_diff: f64) SparseError!bool {
            return (try self.diffSummary(rhs)).maxAbsDiffMeetsBound(max_absolute_diff);
        }

        pub fn maxRelDiffMeetsBound(self: Self, rhs: Self, max_relative_diff: f64) SparseError!bool {
            return (try self.diffSummary(rhs)).maxRelDiffMeetsBound(max_relative_diff);
        }

        pub fn squaredDistanceMeetsBound(self: Self, rhs: Self, max_squared_distance: f64) SparseError!bool {
            return (try self.diffSummary(rhs)).squaredDistanceMeetsBound(max_squared_distance);
        }

        pub fn frobeniusDistanceMeetsBound(self: Self, rhs: Self, max_distance: f64) SparseError!bool {
            return (try self.diffSummary(rhs)).frobeniusDistanceMeetsBound(max_distance);
        }

        pub fn relativeFrobeniusDistanceMeetsBound(self: Self, rhs: Self, max_relative_distance: f64) SparseError!bool {
            return (try self.diffSummary(rhs)).relativeFrobeniusDistanceMeetsBound(max_relative_distance);
        }

        fn compareDenseMaterialized(self: Self, rhs: Self, comptime comparison: SparseScalarComparison) SparseError!array_mod.Array(bool) {
            if (self.rows != rhs.rows or self.cols != rhs.cols) return error.ShapeMismatch;
            var lhs_dense = try self.toDense();
            defer lhs_dense.deinit();
            var rhs_dense = try rhs.toDense();
            defer rhs_dense.deinit();
            return sparseCompareDenseArrays(T, lhs_dense, rhs_dense, comparison);
        }

        fn compareDenseArray(self: Self, rhs: array_mod.Array(T), comptime comparison: SparseScalarComparison) SparseError!array_mod.Array(bool) {
            try validateDenseMatrixShape(self.rows, self.cols, rhs.shape);
            var lhs_dense = try self.toDense();
            defer lhs_dense.deinit();
            return sparseCompareDenseArrays(T, lhs_dense, rhs, comparison);
        }

        pub fn eq(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return self.compareDenseMaterialized(rhs, .eq);
        }

        pub fn equal(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return self.eq(rhs);
        }

        pub fn ne(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return self.compareDenseMaterialized(rhs, .ne);
        }

        pub fn notEqual(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return self.ne(rhs);
        }

        pub fn gt(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return self.compareDenseMaterialized(rhs, .gt);
        }

        pub fn greater(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return self.gt(rhs);
        }

        pub fn ge(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return self.compareDenseMaterialized(rhs, .ge);
        }

        pub fn greaterEqual(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return self.ge(rhs);
        }

        pub fn lt(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return self.compareDenseMaterialized(rhs, .lt);
        }

        pub fn less(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return self.lt(rhs);
        }

        pub fn le(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return self.compareDenseMaterialized(rhs, .le);
        }

        pub fn lessEqual(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return self.le(rhs);
        }

        pub fn eqDense(self: Self, rhs: array_mod.Array(T)) SparseError!array_mod.Array(bool) {
            return self.compareDenseArray(rhs, .eq);
        }

        pub fn equalDense(self: Self, rhs: array_mod.Array(T)) SparseError!array_mod.Array(bool) {
            return self.eqDense(rhs);
        }

        pub fn neDense(self: Self, rhs: array_mod.Array(T)) SparseError!array_mod.Array(bool) {
            return self.compareDenseArray(rhs, .ne);
        }

        pub fn notEqualDense(self: Self, rhs: array_mod.Array(T)) SparseError!array_mod.Array(bool) {
            return self.neDense(rhs);
        }

        pub fn gtDense(self: Self, rhs: array_mod.Array(T)) SparseError!array_mod.Array(bool) {
            return self.compareDenseArray(rhs, .gt);
        }

        pub fn greaterDense(self: Self, rhs: array_mod.Array(T)) SparseError!array_mod.Array(bool) {
            return self.gtDense(rhs);
        }

        pub fn geDense(self: Self, rhs: array_mod.Array(T)) SparseError!array_mod.Array(bool) {
            return self.compareDenseArray(rhs, .ge);
        }

        pub fn greaterEqualDense(self: Self, rhs: array_mod.Array(T)) SparseError!array_mod.Array(bool) {
            return self.geDense(rhs);
        }

        pub fn ltDense(self: Self, rhs: array_mod.Array(T)) SparseError!array_mod.Array(bool) {
            return self.compareDenseArray(rhs, .lt);
        }

        pub fn lessDense(self: Self, rhs: array_mod.Array(T)) SparseError!array_mod.Array(bool) {
            return self.ltDense(rhs);
        }

        pub fn leDense(self: Self, rhs: array_mod.Array(T)) SparseError!array_mod.Array(bool) {
            return self.compareDenseArray(rhs, .le);
        }

        pub fn lessEqualDense(self: Self, rhs: array_mod.Array(T)) SparseError!array_mod.Array(bool) {
            return self.leDense(rhs);
        }

        pub fn diffSummaryDense(self: Self, rhs: array_mod.Array(T)) SparseError!SparseDiffSummary {
            if (rhs.shape.len != 2) return error.NonMatrixArray;
            if (self.rows != rhs.shape[0] or self.cols != rhs.shape[1]) return error.ShapeMismatch;
            var lhs_dense = try self.toDense();
            defer lhs_dense.deinit();
            return denseDiffSummary(T, lhs_dense.data, rhs.data);
        }

        pub fn diffSummaryDenseMeetsBounds(
            self: Self,
            rhs: array_mod.Array(T),
            max_absolute_diff: f64,
            max_relative_diff: f64,
            max_squared_distance: f64,
            max_frobenius_distance: f64,
            max_relative_frobenius_distance: f64,
        ) SparseError!bool {
            if (rhs.shape.len != 2) return error.NonMatrixArray;
            if (self.rows != rhs.shape[0] or self.cols != rhs.shape[1]) return error.ShapeMismatch;
            var lhs_dense = try self.toDense();
            defer lhs_dense.deinit();
            return denseDiffSummaryMeetsBounds(
                T,
                lhs_dense.data,
                rhs.data,
                max_absolute_diff,
                max_relative_diff,
                max_squared_distance,
                max_frobenius_distance,
                max_relative_frobenius_distance,
            );
        }

        pub fn maxAbsDiffDense(self: Self, rhs: array_mod.Array(T)) SparseError!f64 {
            return (try self.diffSummaryDense(rhs)).max_abs_diff;
        }

        pub fn maxRelDiffDense(self: Self, rhs: array_mod.Array(T)) SparseError!f64 {
            return (try self.diffSummaryDense(rhs)).max_rel_diff;
        }

        pub fn squaredDistanceDense(self: Self, rhs: array_mod.Array(T)) SparseError!f64 {
            return (try self.diffSummaryDense(rhs)).squared_distance;
        }

        pub fn frobeniusDistanceDense(self: Self, rhs: array_mod.Array(T)) SparseError!f64 {
            return (try self.diffSummaryDense(rhs)).frobeniusDistance();
        }

        pub fn relativeFrobeniusDistanceDense(self: Self, rhs: array_mod.Array(T)) SparseError!f64 {
            return (try self.diffSummaryDense(rhs)).relativeFrobeniusDistance();
        }

        pub fn maxAbsDiffDenseMeetsBound(self: Self, rhs: array_mod.Array(T), max_absolute_diff: f64) SparseError!bool {
            return (try self.diffSummaryDense(rhs)).maxAbsDiffMeetsBound(max_absolute_diff);
        }

        pub fn maxRelDiffDenseMeetsBound(self: Self, rhs: array_mod.Array(T), max_relative_diff: f64) SparseError!bool {
            return (try self.diffSummaryDense(rhs)).maxRelDiffMeetsBound(max_relative_diff);
        }

        pub fn squaredDistanceDenseMeetsBound(self: Self, rhs: array_mod.Array(T), max_squared_distance: f64) SparseError!bool {
            return (try self.diffSummaryDense(rhs)).squaredDistanceMeetsBound(max_squared_distance);
        }

        pub fn frobeniusDistanceDenseMeetsBound(self: Self, rhs: array_mod.Array(T), max_distance: f64) SparseError!bool {
            return (try self.diffSummaryDense(rhs)).frobeniusDistanceMeetsBound(max_distance);
        }

        pub fn relativeFrobeniusDistanceDenseMeetsBound(self: Self, rhs: array_mod.Array(T), max_relative_distance: f64) SparseError!bool {
            return (try self.diffSummaryDense(rhs)).relativeFrobeniusDistanceMeetsBound(max_relative_distance);
        }

        pub fn iscloseDense(self: Self, rhs: array_mod.Array(T), rtol: T, atol: T) SparseError!array_mod.Array(bool) {
            return self.iscloseDenseEqualNan(rhs, rtol, atol, false);
        }

        pub fn isCloseDense(self: Self, rhs: array_mod.Array(T), rtol: T, atol: T) SparseError!array_mod.Array(bool) {
            return self.iscloseDense(rhs, rtol, atol);
        }

        pub fn iscloseDenseEqualNan(self: Self, rhs: array_mod.Array(T), rtol: T, atol: T, equal_nan: bool) SparseError!array_mod.Array(bool) {
            if (rhs.shape.len != 2) return error.NonMatrixArray;
            if (self.rows != rhs.shape[0] or self.cols != rhs.shape[1]) return error.ShapeMismatch;
            var lhs_dense = try self.toDense();
            defer lhs_dense.deinit();
            return lhs_dense.iscloseEqualNan(rhs, rtol, atol, equal_nan);
        }

        pub fn isCloseDenseEqualNan(self: Self, rhs: array_mod.Array(T), rtol: T, atol: T, equal_nan: bool) SparseError!array_mod.Array(bool) {
            return self.iscloseDenseEqualNan(rhs, rtol, atol, equal_nan);
        }

        pub fn allcloseDense(self: Self, rhs: array_mod.Array(T), rtol: T, atol: T) SparseError!bool {
            return self.allcloseDenseEqualNan(rhs, rtol, atol, false);
        }

        pub fn allCloseDense(self: Self, rhs: array_mod.Array(T), rtol: T, atol: T) SparseError!bool {
            return self.allcloseDense(rhs, rtol, atol);
        }

        pub fn allcloseDenseEqualNan(self: Self, rhs: array_mod.Array(T), rtol: T, atol: T, equal_nan: bool) SparseError!bool {
            if (rhs.shape.len != 2) return error.NonMatrixArray;
            if (self.rows != rhs.shape[0] or self.cols != rhs.shape[1]) return error.ShapeMismatch;
            var lhs_dense = try self.toDense();
            defer lhs_dense.deinit();
            return lhs_dense.allcloseEqualNan(rhs, rtol, atol, equal_nan);
        }

        pub fn allCloseDenseEqualNan(self: Self, rhs: array_mod.Array(T), rtol: T, atol: T, equal_nan: bool) SparseError!bool {
            return self.allcloseDenseEqualNan(rhs, rtol, atol, equal_nan);
        }

        pub fn iscloseScalar(self: Self, scalar: T, rtol: T, atol: T) SparseError!array_mod.Array(bool) {
            return self.iscloseScalarEqualNan(scalar, rtol, atol, false);
        }

        pub fn isCloseScalar(self: Self, scalar: T, rtol: T, atol: T) SparseError!array_mod.Array(bool) {
            return self.iscloseScalar(scalar, rtol, atol);
        }

        pub fn iscloseScalarEqualNan(self: Self, scalar: T, rtol: T, atol: T, equal_nan: bool) SparseError!array_mod.Array(bool) {
            var dense = try self.toDense();
            defer dense.deinit();
            return dense.iscloseScalarEqualNan(scalar, rtol, atol, equal_nan);
        }

        pub fn isCloseScalarEqualNan(self: Self, scalar: T, rtol: T, atol: T, equal_nan: bool) SparseError!array_mod.Array(bool) {
            return self.iscloseScalarEqualNan(scalar, rtol, atol, equal_nan);
        }

        pub fn allcloseScalar(self: Self, scalar: T, rtol: T, atol: T) SparseError!bool {
            return self.allcloseScalarEqualNan(scalar, rtol, atol, false);
        }

        pub fn allCloseScalar(self: Self, scalar: T, rtol: T, atol: T) SparseError!bool {
            return self.allcloseScalar(scalar, rtol, atol);
        }

        pub fn allcloseScalarEqualNan(self: Self, scalar: T, rtol: T, atol: T, equal_nan: bool) SparseError!bool {
            var dense = try self.toDense();
            defer dense.deinit();
            return dense.allcloseScalarEqualNan(scalar, rtol, atol, equal_nan);
        }

        pub fn allCloseScalarEqualNan(self: Self, scalar: T, rtol: T, atol: T, equal_nan: bool) SparseError!bool {
            return self.allcloseScalarEqualNan(scalar, rtol, atol, equal_nan);
        }

        pub fn isclose(self: Self, rhs: Self, rtol: T, atol: T) SparseError!array_mod.Array(bool) {
            return self.iscloseEqualNan(rhs, rtol, atol, false);
        }

        pub fn isClose(self: Self, rhs: Self, rtol: T, atol: T) SparseError!array_mod.Array(bool) {
            return self.isclose(rhs, rtol, atol);
        }

        pub fn iscloseEqualNan(self: Self, rhs: Self, rtol: T, atol: T, equal_nan: bool) SparseError!array_mod.Array(bool) {
            if (self.rows != rhs.rows or self.cols != rhs.cols) return error.ShapeMismatch;
            var lhs_dense = try self.toDense();
            defer lhs_dense.deinit();
            var rhs_dense = try rhs.toDense();
            defer rhs_dense.deinit();
            return lhs_dense.iscloseEqualNan(rhs_dense, rtol, atol, equal_nan);
        }

        pub fn isCloseEqualNan(self: Self, rhs: Self, rtol: T, atol: T, equal_nan: bool) SparseError!array_mod.Array(bool) {
            return self.iscloseEqualNan(rhs, rtol, atol, equal_nan);
        }

        pub fn allclose(self: Self, rhs: Self, rtol: T, atol: T) SparseError!bool {
            return self.allcloseEqualNan(rhs, rtol, atol, false);
        }

        pub fn allClose(self: Self, rhs: Self, rtol: T, atol: T) SparseError!bool {
            return self.allclose(rhs, rtol, atol);
        }

        pub fn allcloseEqualNan(self: Self, rhs: Self, rtol: T, atol: T, equal_nan: bool) SparseError!bool {
            if (self.rows != rhs.rows or self.cols != rhs.cols) return error.ShapeMismatch;
            var lhs_dense = try self.toDense();
            defer lhs_dense.deinit();
            var rhs_dense = try rhs.toDense();
            defer rhs_dense.deinit();
            return lhs_dense.allcloseEqualNan(rhs_dense, rtol, atol, equal_nan);
        }

        pub fn allCloseEqualNan(self: Self, rhs: Self, rtol: T, atol: T, equal_nan: bool) SparseError!bool {
            return self.allcloseEqualNan(rhs, rtol, atol, equal_nan);
        }

        pub fn maxAbsDiffSameStructure(self: Self, rhs: Self) SparseError!T {
            if (self.rows != rhs.rows or self.cols != rhs.cols or self.values.len != rhs.values.len) return error.ShapeMismatch;
            if (!self.sameStructure(rhs)) return error.InvalidShape;
            return sparseMaxAbsDiffSameStructure(T, self.values, rhs.values);
        }

        pub fn maxAbsDiffSameStructureMeetsBound(self: Self, rhs: Self, max_absolute_diff: T) SparseError!bool {
            if (self.rows != rhs.rows or self.cols != rhs.cols or self.values.len != rhs.values.len) return error.ShapeMismatch;
            if (!self.sameStructure(rhs)) return error.InvalidShape;
            return sparseMaxAbsDiffSameStructureMeetsBound(T, self.values, rhs.values, max_absolute_diff);
        }

        pub fn maxRelDiffSameStructure(self: Self, rhs: Self) SparseError!T {
            if (self.rows != rhs.rows or self.cols != rhs.cols or self.values.len != rhs.values.len) return error.ShapeMismatch;
            if (!self.sameStructure(rhs)) return error.InvalidShape;
            return sparseMaxRelDiffSameStructure(T, self.values, rhs.values);
        }

        pub fn maxRelDiffSameStructureMeetsBound(self: Self, rhs: Self, max_relative_diff: T) SparseError!bool {
            if (self.rows != rhs.rows or self.cols != rhs.cols or self.values.len != rhs.values.len) return error.ShapeMismatch;
            if (!self.sameStructure(rhs)) return error.InvalidShape;
            return sparseMaxRelDiffSameStructureMeetsBound(T, self.values, rhs.values, max_relative_diff);
        }

        pub fn squaredDistanceSameStructure(self: Self, rhs: Self) SparseError!T {
            if (self.rows != rhs.rows or self.cols != rhs.cols or self.values.len != rhs.values.len) return error.ShapeMismatch;
            if (!self.sameStructure(rhs)) return error.InvalidShape;
            return sparseSquaredDistanceSameStructure(T, self.values, rhs.values);
        }

        pub fn squaredDistanceSameStructureMeetsBound(self: Self, rhs: Self, max_squared_distance: T) SparseError!bool {
            if (self.rows != rhs.rows or self.cols != rhs.cols or self.values.len != rhs.values.len) return error.ShapeMismatch;
            if (!self.sameStructure(rhs)) return error.InvalidShape;
            return sparseSquaredDistanceSameStructureMeetsBound(T, self.values, rhs.values, max_squared_distance);
        }

        pub fn frobeniusDistanceSameStructure(self: Self, rhs: Self) SparseError!T {
            if (self.rows != rhs.rows or self.cols != rhs.cols or self.values.len != rhs.values.len) return error.ShapeMismatch;
            if (!self.sameStructure(rhs)) return error.InvalidShape;
            return sparseFrobeniusDistanceSameStructure(T, self.values, rhs.values);
        }

        pub fn frobeniusDistanceSameStructureMeetsBound(self: Self, rhs: Self, max_distance: T) SparseError!bool {
            if (self.rows != rhs.rows or self.cols != rhs.cols or self.values.len != rhs.values.len) return error.ShapeMismatch;
            if (!self.sameStructure(rhs)) return error.InvalidShape;
            return sparseFrobeniusDistanceSameStructureMeetsBound(T, self.values, rhs.values, max_distance);
        }

        pub fn relativeFrobeniusDistanceSameStructure(self: Self, rhs: Self) SparseError!T {
            if (self.rows != rhs.rows or self.cols != rhs.cols or self.values.len != rhs.values.len) return error.ShapeMismatch;
            if (!self.sameStructure(rhs)) return error.InvalidShape;
            return sparseRelativeFrobeniusDistanceSameStructure(T, self.values, rhs.values);
        }

        pub fn relativeFrobeniusDistanceSameStructureMeetsBound(self: Self, rhs: Self, max_relative_distance: T) SparseError!bool {
            if (self.rows != rhs.rows or self.cols != rhs.cols or self.values.len != rhs.values.len) return error.ShapeMismatch;
            if (!self.sameStructure(rhs)) return error.InvalidShape;
            return sparseRelativeFrobeniusDistanceSameStructureMeetsBound(T, self.values, rhs.values, max_relative_distance);
        }

        pub fn coalesced(self: Self) SparseError!Self {
            const entries = try self.allocator.alloc(Entry, self.values.len);
            defer self.allocator.free(entries);
            for (entries, 0..) |*entry, i| {
                entry.* = .{
                    .row = self.row_indices[i],
                    .col = self.col_indices[i],
                    .value = self.values[i],
                };
            }
            std.sort.insertion(Entry, entries, {}, entryLessThan);

            var unique_count: usize = 0;
            for (entries, 0..) |entry, i| {
                if (i == 0 or entry.row != entries[i - 1].row or entry.col != entries[i - 1].col) unique_count += 1;
            }

            var row_indices = try self.allocator.alloc(usize, unique_count);
            errdefer self.allocator.free(row_indices);
            var col_indices = try self.allocator.alloc(usize, unique_count);
            errdefer self.allocator.free(col_indices);
            var values = try self.allocator.alloc(T, unique_count);
            errdefer self.allocator.free(values);

            var write: usize = 0;
            for (entries) |entry| {
                if (write > 0 and row_indices[write - 1] == entry.row and col_indices[write - 1] == entry.col) {
                    values[write - 1] = addSparseValue(T, values[write - 1], entry.value);
                } else {
                    row_indices[write] = entry.row;
                    col_indices[write] = entry.col;
                    values[write] = entry.value;
                    write += 1;
                }
            }
            std.debug.assert(write == unique_count);

            return .{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = self.cols,
                .row_indices = row_indices,
                .col_indices = col_indices,
                .values = values,
            };
        }

        pub fn add(self: Self, rhs: Self) SparseError!Self {
            if (self.rows != rhs.rows or self.cols != rhs.cols) return error.ShapeMismatch;
            const total_nnz = self.values.len + rhs.values.len;
            var row_indices = try self.allocator.alloc(usize, total_nnz);
            errdefer self.allocator.free(row_indices);
            var col_indices = try self.allocator.alloc(usize, total_nnz);
            errdefer self.allocator.free(col_indices);
            var values = try self.allocator.alloc(T, total_nnz);
            errdefer self.allocator.free(values);

            @memcpy(row_indices[0..self.row_indices.len], self.row_indices);
            @memcpy(col_indices[0..self.col_indices.len], self.col_indices);
            @memcpy(values[0..self.values.len], self.values);
            @memcpy(row_indices[self.row_indices.len..], rhs.row_indices);
            @memcpy(col_indices[self.col_indices.len..], rhs.col_indices);
            @memcpy(values[self.values.len..], rhs.values);

            // Canonicalize after concatenation so sparse addition has the same
            // duplicate-coordinate semantics as dense materialization and the
            // explicit COO coalescing API.
            var combined = Self{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = self.cols,
                .row_indices = row_indices,
                .col_indices = col_indices,
                .values = values,
            };
            defer combined.deinit();
            return combined.coalesced();
        }

        pub fn dropZerosNnz(self: Self) usize {
            return sparseDropZerosNnz(T, self.values);
        }

        pub fn dropZeros(self: Self) SparseError!Self {
            const nonzero_count = self.dropZerosNnz();

            var row_indices = try self.allocator.alloc(usize, nonzero_count);
            errdefer self.allocator.free(row_indices);
            var col_indices = try self.allocator.alloc(usize, nonzero_count);
            errdefer self.allocator.free(col_indices);
            var values = try self.allocator.alloc(T, nonzero_count);
            errdefer self.allocator.free(values);

            var write: usize = 0;
            for (self.values, 0..) |value, i| {
                if (isNonZero(T, value)) {
                    row_indices[write] = self.row_indices[i];
                    col_indices[write] = self.col_indices[i];
                    values[write] = value;
                    write += 1;
                }
            }
            std.debug.assert(write == nonzero_count);

            return .{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = self.cols,
                .row_indices = row_indices,
                .col_indices = col_indices,
                .values = values,
            };
        }

        pub fn pruneZerosNnz(self: Self, tolerance: T) SparseError!usize {
            return sparsePruneZerosNnz(T, self.values, tolerance);
        }

        pub fn pruneZeros(self: Self, tolerance: T) SparseError!Self {
            const keep_count = try self.pruneZerosNnz(tolerance);

            var row_indices = try self.allocator.alloc(usize, keep_count);
            errdefer self.allocator.free(row_indices);
            var col_indices = try self.allocator.alloc(usize, keep_count);
            errdefer self.allocator.free(col_indices);
            var values = try self.allocator.alloc(T, keep_count);
            errdefer self.allocator.free(values);

            var write: usize = 0;
            for (self.values, 0..) |value, i| {
                if (sparseAbsValueExceedsTolerance(T, value, tolerance)) {
                    row_indices[write] = self.row_indices[i];
                    col_indices[write] = self.col_indices[i];
                    values[write] = value;
                    write += 1;
                }
            }
            std.debug.assert(write == keep_count);
            return .{ .allocator = self.allocator, .rows = self.rows, .cols = self.cols, .row_indices = row_indices, .col_indices = col_indices, .values = values };
        }

        pub fn neg(self: Self) SparseError!Self {
            ensureNumeric(T);
            return self.scale(negSparseValue(T, oneValue(T)));
        }

        pub fn negative(self: Self) SparseError!Self {
            return self.neg();
        }

        pub fn sub(self: Self, rhs: Self) SparseError!Self {
            if (self.rows != rhs.rows or self.cols != rhs.cols) return error.ShapeMismatch;
            var neg_rhs = try rhs.neg();
            defer neg_rhs.deinit();
            return self.add(neg_rhs);
        }

        pub fn hadamard(self: Self, rhs: Self) SparseError!Self {
            if (self.rows != rhs.rows or self.cols != rhs.cols) return error.ShapeMismatch;
            var lhs_canonical = try self.coalesced();
            defer lhs_canonical.deinit();
            var rhs_canonical = try rhs.coalesced();
            defer rhs_canonical.deinit();

            var count: usize = 0;
            var lhs_pos: usize = 0;
            var rhs_pos: usize = 0;
            while (lhs_pos < lhs_canonical.values.len and rhs_pos < rhs_canonical.values.len) {
                const lhs_row = lhs_canonical.row_indices[lhs_pos];
                const lhs_col = lhs_canonical.col_indices[lhs_pos];
                const rhs_row = rhs_canonical.row_indices[rhs_pos];
                const rhs_col = rhs_canonical.col_indices[rhs_pos];
                if (lhs_row == rhs_row and lhs_col == rhs_col) {
                    count += 1;
                    lhs_pos += 1;
                    rhs_pos += 1;
                } else if (coordLess(lhs_row, lhs_col, rhs_row, rhs_col)) {
                    lhs_pos += 1;
                } else {
                    rhs_pos += 1;
                }
            }

            var row_indices = try self.allocator.alloc(usize, count);
            errdefer self.allocator.free(row_indices);
            var col_indices = try self.allocator.alloc(usize, count);
            errdefer self.allocator.free(col_indices);
            var values = try self.allocator.alloc(T, count);
            errdefer self.allocator.free(values);

            lhs_pos = 0;
            rhs_pos = 0;
            var write: usize = 0;
            while (lhs_pos < lhs_canonical.values.len and rhs_pos < rhs_canonical.values.len) {
                const lhs_row = lhs_canonical.row_indices[lhs_pos];
                const lhs_col = lhs_canonical.col_indices[lhs_pos];
                const rhs_row = rhs_canonical.row_indices[rhs_pos];
                const rhs_col = rhs_canonical.col_indices[rhs_pos];
                if (lhs_row == rhs_row and lhs_col == rhs_col) {
                    row_indices[write] = lhs_row;
                    col_indices[write] = lhs_col;
                    values[write] = mulSparseValue(T, lhs_canonical.values[lhs_pos], rhs_canonical.values[rhs_pos]);
                    write += 1;
                    lhs_pos += 1;
                    rhs_pos += 1;
                } else if (coordLess(lhs_row, lhs_col, rhs_row, rhs_col)) {
                    lhs_pos += 1;
                } else {
                    rhs_pos += 1;
                }
            }
            std.debug.assert(write == count);

            return .{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = self.cols,
                .row_indices = row_indices,
                .col_indices = col_indices,
                .values = values,
            };
        }

        pub fn mul(self: Self, rhs: Self) SparseError!Self {
            return self.hadamard(rhs);
        }

        pub fn multiply(self: Self, rhs: Self) SparseError!Self {
            return self.hadamard(rhs);
        }

        pub fn scale(self: Self, alpha: T) SparseError!Self {
            ensureNumeric(T);
            const row_indices = try self.allocator.dupe(usize, self.row_indices);
            errdefer self.allocator.free(row_indices);
            const col_indices = try self.allocator.dupe(usize, self.col_indices);
            errdefer self.allocator.free(col_indices);
            var values = try self.allocator.alloc(T, self.values.len);
            errdefer self.allocator.free(values);

            for (self.values, 0..) |value, i| values[i] = value * alpha;

            return .{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = self.cols,
                .row_indices = row_indices,
                .col_indices = col_indices,
                .values = values,
            };
        }

        pub fn scaleRows(self: Self, row_scale: []const T) SparseError!Self {
            ensureNumeric(T);
            if (row_scale.len != self.rows) return error.ShapeMismatch;
            const row_indices = try self.allocator.dupe(usize, self.row_indices);
            errdefer self.allocator.free(row_indices);
            const col_indices = try self.allocator.dupe(usize, self.col_indices);
            errdefer self.allocator.free(col_indices);
            var values = try self.allocator.alloc(T, self.values.len);
            errdefer self.allocator.free(values);

            for (self.values, 0..) |value, i| values[i] = row_scale[self.row_indices[i]] * value;

            return .{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = self.cols,
                .row_indices = row_indices,
                .col_indices = col_indices,
                .values = values,
            };
        }

        pub fn scaleRowsInPlace(self: *Self, row_scale: []const T) SparseError!void {
            ensureNumeric(T);
            if (row_scale.len != self.rows) return error.ShapeMismatch;
            for (self.values, self.row_indices) |*value, row| value.* *= row_scale[row];
        }

        pub fn scaleColumns(self: Self, col_scale: []const T) SparseError!Self {
            ensureNumeric(T);
            if (col_scale.len != self.cols) return error.ShapeMismatch;
            const row_indices = try self.allocator.dupe(usize, self.row_indices);
            errdefer self.allocator.free(row_indices);
            const col_indices = try self.allocator.dupe(usize, self.col_indices);
            errdefer self.allocator.free(col_indices);
            var values = try self.allocator.alloc(T, self.values.len);
            errdefer self.allocator.free(values);

            for (self.values, 0..) |value, i| values[i] = value * col_scale[self.col_indices[i]];

            return .{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = self.cols,
                .row_indices = row_indices,
                .col_indices = col_indices,
                .values = values,
            };
        }

        pub fn scaleColumnsInPlace(self: *Self, col_scale: []const T) SparseError!void {
            ensureNumeric(T);
            if (col_scale.len != self.cols) return error.ShapeMismatch;
            for (self.values, self.col_indices) |*value, col| value.* *= col_scale[col];
        }

        pub fn scaleRowsAndColumns(self: Self, row_scale: []const T, col_scale: []const T) SparseError!Self {
            ensureNumeric(T);
            if (row_scale.len != self.rows or col_scale.len != self.cols) return error.ShapeMismatch;
            const row_indices = try self.allocator.dupe(usize, self.row_indices);
            errdefer self.allocator.free(row_indices);
            const col_indices = try self.allocator.dupe(usize, self.col_indices);
            errdefer self.allocator.free(col_indices);
            var values = try self.allocator.alloc(T, self.values.len);
            errdefer self.allocator.free(values);

            for (self.values, 0..) |value, i| values[i] = row_scale[self.row_indices[i]] * value * col_scale[self.col_indices[i]];

            return .{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = self.cols,
                .row_indices = row_indices,
                .col_indices = col_indices,
                .values = values,
            };
        }

        pub fn scaleRowsAndColumnsInPlace(self: *Self, row_scale: []const T, col_scale: []const T) SparseError!void {
            ensureNumeric(T);
            if (row_scale.len != self.rows or col_scale.len != self.cols) return error.ShapeMismatch;
            for (self.values, self.row_indices, self.col_indices) |*value, row, col| value.* *= row_scale[row] * col_scale[col];
        }

        pub fn sum(self: Self) T {
            ensureNumeric(T);
            var total = zero(T);
            for (self.values) |value| total = addSparseValue(T, total, value);
            return total;
        }

        pub fn absSum(self: Self) T {
            ensureNumeric(T);
            var total = zero(T);
            for (self.values) |value| total += absValue(T, value);
            return total;
        }

        pub fn sumInRange(self: Self, min_sum: T, max_sum: T) SparseError!bool {
            try validateSparseValueRange(T, min_sum, max_sum);
            return sparseValueInValidatedRange(T, self.sum(), min_sum, max_sum);
        }

        pub fn absSumInRange(self: Self, min_abs_sum: T, max_abs_sum: T) SparseError!bool {
            try validateSparseValueRange(T, min_abs_sum, max_abs_sum);
            return sparseValueInValidatedRange(T, self.absSum(), min_abs_sum, max_abs_sum);
        }

        pub fn minValue(self: Self) SparseError!T {
            return minStoredValue(T, self.values);
        }

        pub fn minValueIndex(self: Self) SparseError!usize {
            return minStoredValueIndex(T, self.values);
        }

        pub fn maxValue(self: Self) SparseError!T {
            return maxStoredValue(T, self.values);
        }

        pub fn maxValueIndex(self: Self) SparseError!usize {
            return maxStoredValueIndex(T, self.values);
        }

        pub fn minAbsValue(self: Self) SparseError!T {
            return minStoredAbsValue(T, self.values);
        }

        pub fn minAbsValueIndex(self: Self) SparseError!usize {
            return minStoredAbsValueIndex(T, self.values);
        }

        pub fn maxAbsValue(self: Self) SparseError!T {
            return maxStoredAbsValue(T, self.values);
        }

        pub fn maxAbsValueIndex(self: Self) SparseError!usize {
            return maxStoredAbsValueIndex(T, self.values);
        }

        pub fn nonFiniteCount(self: Self) usize {
            return sparseNonFiniteCount(T, self.values);
        }

        pub fn nonFiniteCountMeetsBound(self: Self, max_count: usize) bool {
            return self.nonFiniteCount() <= max_count;
        }

        pub fn nonFiniteCountInRange(self: Self, min_count: usize, max_count: usize) SparseError!bool {
            try validateCountRange(min_count, max_count);
            return sparseCountInValidatedRange(self.nonFiniteCount(), min_count, max_count);
        }

        pub fn allFinite(self: Self) bool {
            return sparseAllFinite(T, self.values);
        }

        pub fn rowNonFiniteCounts(self: Self) SparseError!array_mod.Array(usize) {
            var out = try array_mod.Array(usize).zeros(self.allocator, &.{self.rows});
            errdefer out.deinit();
            for (self.values, self.row_indices) |value, row| {
                if (!sparseValueIsFinite(T, value)) out.data[row] += 1;
            }
            return out;
        }

        pub fn columnNonFiniteCounts(self: Self) SparseError!array_mod.Array(usize) {
            var out = try array_mod.Array(usize).zeros(self.allocator, &.{self.cols});
            errdefer out.deinit();
            for (self.values, self.col_indices) |value, col| {
                if (!sparseValueIsFinite(T, value)) out.data[col] += 1;
            }
            return out;
        }

        pub fn rowNonFiniteCountsInRange(self: Self, min_count: usize, max_count: usize) SparseError!bool {
            var counts = try self.rowNonFiniteCounts();
            defer counts.deinit();
            return sparseCountRangeInRange(counts.data, min_count, max_count);
        }

        pub fn rowNonFiniteCountsMeetBound(self: Self, max_count: usize) SparseError!bool {
            return self.rowNonFiniteCountsInRange(0, max_count);
        }

        pub fn columnNonFiniteCountsInRange(self: Self, min_count: usize, max_count: usize) SparseError!bool {
            var counts = try self.columnNonFiniteCounts();
            defer counts.deinit();
            return sparseCountRangeInRange(counts.data, min_count, max_count);
        }

        pub fn columnNonFiniteCountsMeetBound(self: Self, max_count: usize) SparseError!bool {
            return self.columnNonFiniteCountsInRange(0, max_count);
        }

        pub fn valueRangeInRange(self: Self, min_value: T, max_value: T) SparseError!bool {
            return sparseValueRangeInRange(T, self.values, min_value, max_value);
        }

        pub fn absValueRangeInRange(self: Self, min_abs_value: T, max_abs_value: T) SparseError!bool {
            return sparseAbsValueRangeInRange(T, self.values, min_abs_value, max_abs_value);
        }

        pub fn valueDynamicRange(self: Self) SparseError!f64 {
            return sparseStoredValueDynamicRange(T, self.values);
        }

        pub fn valueDynamicRangeMeetsBound(self: Self, max_dynamic_range: f64) SparseError!bool {
            return sparseStoredValueDynamicRangeMeetsBound(T, self.values, max_dynamic_range);
        }

        pub fn mean(self: Self) SparseError!f64 {
            ensureNumeric(T);
            const count = try sparseElementCount(self.rows, self.cols);
            if (count == 0) return error.EmptyArray;
            var total: f64 = 0;
            for (self.values) |value| total += sparseValueToF64(T, value);
            return total / sparseSizeToF64(count);
        }

        pub fn meanInRange(self: Self, min_mean: f64, max_mean: f64) SparseError!bool {
            try validateFiniteRange(min_mean, max_mean);
            return valueInF64Range(try self.mean(), min_mean, max_mean);
        }

        pub fn rowMeans(self: Self) SparseError!array_mod.Array(f64) {
            ensureNumeric(T);
            if (self.cols == 0) return error.EmptyArray;
            var out = try array_mod.Array(f64).zeros(self.allocator, &.{self.rows});
            errdefer out.deinit();
            for (self.values, self.row_indices) |value, row| out.data[row] += sparseValueToF64(T, value);
            const divisor = sparseSizeToF64(self.cols);
            for (out.data) |*value| value.* /= divisor;
            return out;
        }

        pub fn columnMeans(self: Self) SparseError!array_mod.Array(f64) {
            ensureNumeric(T);
            if (self.rows == 0) return error.EmptyArray;
            var out = try array_mod.Array(f64).zeros(self.allocator, &.{self.cols});
            errdefer out.deinit();
            for (self.values, self.col_indices) |value, col| out.data[col] += sparseValueToF64(T, value);
            const divisor = sparseSizeToF64(self.rows);
            for (out.data) |*value| value.* /= divisor;
            return out;
        }

        pub fn rowMeansInRange(self: Self, min_mean: f64, max_mean: f64) SparseError!bool {
            try validateFiniteRange(min_mean, max_mean);
            var means = try self.rowMeans();
            defer means.deinit();
            for (means.data) |value| {
                if (!valueInF64Range(value, min_mean, max_mean)) return false;
            }
            return true;
        }

        pub fn columnMeansInRange(self: Self, min_mean: f64, max_mean: f64) SparseError!bool {
            try validateFiniteRange(min_mean, max_mean);
            var means = try self.columnMeans();
            defer means.deinit();
            for (means.data) |value| {
                if (!valueInF64Range(value, min_mean, max_mean)) return false;
            }
            return true;
        }

        pub fn variance(self: Self, correction: f64) SparseError!f64 {
            ensureNumeric(T);
            const count = try sparseElementCount(self.rows, self.cols);
            var sum_value: f64 = 0;
            var sum_sq: f64 = 0;
            for (self.values) |value| {
                sum_value += sparseValueToF64(T, value);
                sum_sq += sparseValueSquareToF64(T, value);
            }
            return sparseVarianceFromSums(sum_value, sum_sq, count, correction);
        }

        pub fn varianceInRange(self: Self, correction: f64, min_variance: f64, max_variance: f64) SparseError!bool {
            try validateNonNegativeRange(min_variance, max_variance);
            return valueInF64Range(try self.variance(correction), min_variance, max_variance);
        }

        pub fn stddev(self: Self, correction: f64) SparseError!f64 {
            return @sqrt(try self.variance(correction));
        }

        pub fn stddevInRange(self: Self, correction: f64, min_stddev: f64, max_stddev: f64) SparseError!bool {
            try validateNonNegativeRange(min_stddev, max_stddev);
            return sparseStddevInValidatedRangeFromVariance(try self.variance(correction), min_stddev, max_stddev);
        }

        pub fn sampleVariance(self: Self) SparseError!f64 {
            return self.variance(1);
        }

        pub fn sampleVarianceInRange(self: Self, min_variance: f64, max_variance: f64) SparseError!bool {
            return self.varianceInRange(1, min_variance, max_variance);
        }

        pub fn sampleStddev(self: Self) SparseError!f64 {
            return self.stddev(1);
        }

        pub fn sampleStddevInRange(self: Self, min_stddev: f64, max_stddev: f64) SparseError!bool {
            return self.stddevInRange(1, min_stddev, max_stddev);
        }

        pub fn rowVariances(self: Self, correction: f64) SparseError!array_mod.Array(f64) {
            ensureNumeric(T);
            if (self.cols == 0) return error.EmptyArray;
            var sums = try self.allocator.alloc(f64, self.rows);
            defer self.allocator.free(sums);
            @memset(sums, 0);
            var out = try array_mod.Array(f64).zeros(self.allocator, &.{self.rows});
            errdefer out.deinit();
            for (self.values, self.row_indices) |value, row| {
                const numeric = sparseValueToF64(T, value);
                sums[row] += numeric;
                out.data[row] += numeric * numeric;
            }
            try finalizeVarianceArray(out.data, sums, self.cols, correction);
            return out;
        }

        pub fn rowVariancesInRange(self: Self, correction: f64, min_variance: f64, max_variance: f64) SparseError!bool {
            try validateNonNegativeRange(min_variance, max_variance);
            var variances = try self.rowVariances(correction);
            defer variances.deinit();
            for (variances.data) |value| {
                if (!valueInF64Range(value, min_variance, max_variance)) return false;
            }
            return true;
        }

        pub fn columnVariancesInRange(self: Self, correction: f64, min_variance: f64, max_variance: f64) SparseError!bool {
            try validateNonNegativeRange(min_variance, max_variance);
            var variances = try self.columnVariances(correction);
            defer variances.deinit();
            for (variances.data) |value| {
                if (!valueInF64Range(value, min_variance, max_variance)) return false;
            }
            return true;
        }

        pub fn rowStddevs(self: Self, correction: f64) SparseError!array_mod.Array(f64) {
            const out = try self.rowVariances(correction);
            sqrtArray(out.data);
            return out;
        }

        pub fn rowStddevsInRange(self: Self, correction: f64, min_stddev: f64, max_stddev: f64) SparseError!bool {
            try validateNonNegativeRange(min_stddev, max_stddev);
            var stddevs = try self.rowStddevs(correction);
            defer stddevs.deinit();
            for (stddevs.data) |value| {
                if (!valueInF64Range(value, min_stddev, max_stddev)) return false;
            }
            return true;
        }

        pub fn columnStddevsInRange(self: Self, correction: f64, min_stddev: f64, max_stddev: f64) SparseError!bool {
            try validateNonNegativeRange(min_stddev, max_stddev);
            var stddevs = try self.columnStddevs(correction);
            defer stddevs.deinit();
            for (stddevs.data) |value| {
                if (!valueInF64Range(value, min_stddev, max_stddev)) return false;
            }
            return true;
        }

        pub fn rowSampleVariances(self: Self) SparseError!array_mod.Array(f64) {
            return self.rowVariances(1);
        }

        pub fn rowSampleStddevs(self: Self) SparseError!array_mod.Array(f64) {
            return self.rowStddevs(1);
        }

        pub fn columnVariances(self: Self, correction: f64) SparseError!array_mod.Array(f64) {
            ensureNumeric(T);
            if (self.rows == 0) return error.EmptyArray;
            var sums = try self.allocator.alloc(f64, self.cols);
            defer self.allocator.free(sums);
            @memset(sums, 0);
            var out = try array_mod.Array(f64).zeros(self.allocator, &.{self.cols});
            errdefer out.deinit();
            for (self.values, self.col_indices) |value, col| {
                const numeric = sparseValueToF64(T, value);
                sums[col] += numeric;
                out.data[col] += numeric * numeric;
            }
            try finalizeVarianceArray(out.data, sums, self.rows, correction);
            return out;
        }

        pub fn columnStddevs(self: Self, correction: f64) SparseError!array_mod.Array(f64) {
            const out = try self.columnVariances(correction);
            sqrtArray(out.data);
            return out;
        }

        pub fn columnSampleVariances(self: Self) SparseError!array_mod.Array(f64) {
            return self.columnVariances(1);
        }

        pub fn columnSampleStddevs(self: Self) SparseError!array_mod.Array(f64) {
            return self.columnStddevs(1);
        }

        pub fn frobeniusNorm(self: Self) T {
            ensureFloat(T);
            var total = zero(T);
            for (self.values) |value| total += value * value;
            return @sqrt(total);
        }

        pub fn frobeniusNormMeetsBound(self: Self, max_norm: T) SparseError!bool {
            ensureFloat(T);
            try validateSparseValueRange(T, zero(T), max_norm);
            return self.frobeniusNorm() <= max_norm;
        }

        pub fn density(self: Self) SparseError!f64 {
            const total = self.rows * self.cols;
            if (total == 0) return 0;
            return @as(f64, @floatFromInt(self.values.len)) / @as(f64, @floatFromInt(total));
        }

        pub fn densityInRange(self: Self, min_density: f64, max_density: f64) SparseError!bool {
            try validateNonNegativeRange(min_density, max_density);
            const current = try self.density();
            return current >= min_density and current <= max_density;
        }

        pub fn oneNorm(self: Self) SparseError!T {
            ensureNumeric(T);
            var col_sums = try self.allocator.alloc(T, self.cols);
            defer self.allocator.free(col_sums);
            @memset(col_sums, zero(T));
            for (self.values, self.col_indices) |value, col| col_sums[col] += absValue(T, value);
            var max_sum = zero(T);
            for (col_sums) |sum_value| {
                if (sum_value > max_sum) max_sum = sum_value;
            }
            return max_sum;
        }

        pub fn oneNormMeetsBound(self: Self, max_norm: T) SparseError!bool {
            ensureNumeric(T);
            try validateSparseValueRange(T, zero(T), max_norm);
            return (try self.oneNorm()) <= max_norm;
        }

        pub fn infNorm(self: Self) SparseError!T {
            ensureNumeric(T);
            var row_sums = try self.allocator.alloc(T, self.rows);
            defer self.allocator.free(row_sums);
            @memset(row_sums, zero(T));
            for (self.values, self.row_indices) |value, row| row_sums[row] += absValue(T, value);
            var max_sum = zero(T);
            for (row_sums) |sum_value| {
                if (sum_value > max_sum) max_sum = sum_value;
            }
            return max_sum;
        }

        pub fn infNormMeetsBound(self: Self, max_norm: T) SparseError!bool {
            ensureNumeric(T);
            try validateSparseValueRange(T, zero(T), max_norm);
            return (try self.infNorm()) <= max_norm;
        }

        pub fn rowNnz(self: Self) SparseError!array_mod.Array(usize) {
            var out = try array_mod.Array(usize).zeros(self.allocator, &.{self.rows});
            errdefer out.deinit();
            for (self.row_indices) |row| out.data[row] += 1;
            return out;
        }

        pub fn columnNnz(self: Self) SparseError!array_mod.Array(usize) {
            var out = try array_mod.Array(usize).zeros(self.allocator, &.{self.cols});
            errdefer out.deinit();
            for (self.col_indices) |col| out.data[col] += 1;
            return out;
        }

        pub fn averageRowNnz(self: Self) SparseError!f64 {
            return sparseCountAverage(self.values.len, self.rows);
        }

        pub fn averageColumnNnz(self: Self) SparseError!f64 {
            return sparseCountAverage(self.values.len, self.cols);
        }

        pub fn averageRowNnzInRange(self: Self, min_average: f64, max_average: f64) SparseError!bool {
            return sparseCountAverageInRange(self.values.len, self.rows, min_average, max_average);
        }

        pub fn averageColumnNnzInRange(self: Self, min_average: f64, max_average: f64) SparseError!bool {
            return sparseCountAverageInRange(self.values.len, self.cols, min_average, max_average);
        }

        pub fn emptyRowCount(self: Self) SparseError!usize {
            var counts = try self.rowNnz();
            defer counts.deinit();
            var empty_count: usize = 0;
            for (counts.data) |count| {
                if (count == 0) empty_count += 1;
            }
            return empty_count;
        }

        pub fn emptyColumnCount(self: Self) SparseError!usize {
            var counts = try self.columnNnz();
            defer counts.deinit();
            var empty_count: usize = 0;
            for (counts.data) |count| {
                if (count == 0) empty_count += 1;
            }
            return empty_count;
        }

        pub fn emptyRowCountMeetsBound(self: Self, max_count: usize) SparseError!bool {
            return (try self.emptyRowCount()) <= max_count;
        }

        pub fn emptyRowCountInRange(self: Self, min_count: usize, max_count: usize) SparseError!bool {
            try validateCountRange(min_count, max_count);
            return sparseCountInValidatedRange(try self.emptyRowCount(), min_count, max_count);
        }

        pub fn emptyColumnCountMeetsBound(self: Self, max_count: usize) SparseError!bool {
            return (try self.emptyColumnCount()) <= max_count;
        }

        pub fn emptyColumnCountInRange(self: Self, min_count: usize, max_count: usize) SparseError!bool {
            try validateCountRange(min_count, max_count);
            return sparseCountInValidatedRange(try self.emptyColumnCount(), min_count, max_count);
        }

        pub fn emptyRowFraction(self: Self) SparseError!f64 {
            return sparseCountFraction(try self.emptyRowCount(), self.rows);
        }

        pub fn emptyColumnFraction(self: Self) SparseError!f64 {
            return sparseCountFraction(try self.emptyColumnCount(), self.cols);
        }

        pub fn emptyRowFractionInRange(self: Self, min_fraction: f64, max_fraction: f64) SparseError!bool {
            return sparseCountFractionInRange(try self.emptyRowCount(), self.rows, min_fraction, max_fraction);
        }

        pub fn emptyColumnFractionInRange(self: Self, min_fraction: f64, max_fraction: f64) SparseError!bool {
            return sparseCountFractionInRange(try self.emptyColumnCount(), self.cols, min_fraction, max_fraction);
        }

        pub fn minRowNnz(self: Self) SparseError!usize {
            var counts = try self.rowNnz();
            defer counts.deinit();
            return sparseMinCount(counts.data);
        }

        pub fn maxRowNnz(self: Self) SparseError!usize {
            var counts = try self.rowNnz();
            defer counts.deinit();
            return sparseMaxCount(counts.data);
        }

        pub fn minColumnNnz(self: Self) SparseError!usize {
            var counts = try self.columnNnz();
            defer counts.deinit();
            return sparseMinCount(counts.data);
        }

        pub fn maxColumnNnz(self: Self) SparseError!usize {
            var counts = try self.columnNnz();
            defer counts.deinit();
            return sparseMaxCount(counts.data);
        }

        pub fn rowNnzRangeInRange(self: Self, min_count: usize, max_count: usize) SparseError!bool {
            var counts = try self.rowNnz();
            defer counts.deinit();
            return sparseCountRangeInRange(counts.data, min_count, max_count);
        }

        pub fn columnNnzRangeInRange(self: Self, min_count: usize, max_count: usize) SparseError!bool {
            var counts = try self.columnNnz();
            defer counts.deinit();
            return sparseCountRangeInRange(counts.data, min_count, max_count);
        }

        pub fn rowNnzSpread(self: Self) SparseError!usize {
            var counts = try self.rowNnz();
            defer counts.deinit();
            return sparseCountSpread(counts.data);
        }

        pub fn columnNnzSpread(self: Self) SparseError!usize {
            var counts = try self.columnNnz();
            defer counts.deinit();
            return sparseCountSpread(counts.data);
        }

        pub fn rowNnzSpreadMeetsBound(self: Self, max_spread: usize) SparseError!bool {
            var counts = try self.rowNnz();
            defer counts.deinit();
            return sparseCountSpreadMeetsBound(counts.data, max_spread);
        }

        pub fn columnNnzSpreadMeetsBound(self: Self, max_spread: usize) SparseError!bool {
            var counts = try self.columnNnz();
            defer counts.deinit();
            return sparseCountSpreadMeetsBound(counts.data, max_spread);
        }

        pub fn rowSums(self: Self) SparseError!array_mod.Array(T) {
            ensureNumeric(T);
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.rows});
            errdefer out.deinit();
            for (self.values, 0..) |value, i| out.data[self.row_indices[i]] += value;
            return out;
        }

        pub fn columnSums(self: Self) SparseError!array_mod.Array(T) {
            ensureNumeric(T);
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.cols});
            errdefer out.deinit();
            for (self.values, 0..) |value, i| out.data[self.col_indices[i]] += value;
            return out;
        }

        pub fn rowSumsInRange(self: Self, min_sum: T, max_sum: T) SparseError!bool {
            var sums = try self.rowSums();
            defer sums.deinit();
            return sparseValueRangeInRange(T, sums.data, min_sum, max_sum);
        }

        pub fn columnSumsInRange(self: Self, min_sum: T, max_sum: T) SparseError!bool {
            var sums = try self.columnSums();
            defer sums.deinit();
            return sparseValueRangeInRange(T, sums.data, min_sum, max_sum);
        }

        pub fn rowMins(self: Self) SparseError!array_mod.Array(T) {
            ensureNumeric(T);
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.rows});
            errdefer out.deinit();
            for (self.values, self.row_indices) |value, row| {
                if (valueLess(T, value, out.data[row])) out.data[row] = value;
            }
            return out;
        }

        pub fn rowMaxes(self: Self) SparseError!array_mod.Array(T) {
            ensureNumeric(T);
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.rows});
            errdefer out.deinit();
            for (self.values, self.row_indices) |value, row| {
                if (valueGreater(T, value, out.data[row])) out.data[row] = value;
            }
            return out;
        }

        pub fn columnMins(self: Self) SparseError!array_mod.Array(T) {
            ensureNumeric(T);
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.cols});
            errdefer out.deinit();
            for (self.values, self.col_indices) |value, col| {
                if (valueLess(T, value, out.data[col])) out.data[col] = value;
            }
            return out;
        }

        pub fn columnMaxes(self: Self) SparseError!array_mod.Array(T) {
            ensureNumeric(T);
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.cols});
            errdefer out.deinit();
            for (self.values, self.col_indices) |value, col| {
                if (valueGreater(T, value, out.data[col])) out.data[col] = value;
            }
            return out;
        }

        pub fn rowMinsInRange(self: Self, min_value: T, max_value: T) SparseError!bool {
            var values = try self.rowMins();
            defer values.deinit();
            return sparseValueRangeInRange(T, values.data, min_value, max_value);
        }

        pub fn columnMinsInRange(self: Self, min_value: T, max_value: T) SparseError!bool {
            var values = try self.columnMins();
            defer values.deinit();
            return sparseValueRangeInRange(T, values.data, min_value, max_value);
        }

        pub fn rowMaxesInRange(self: Self, min_value: T, max_value: T) SparseError!bool {
            var values = try self.rowMaxes();
            defer values.deinit();
            return sparseValueRangeInRange(T, values.data, min_value, max_value);
        }

        pub fn columnMaxesInRange(self: Self, min_value: T, max_value: T) SparseError!bool {
            var values = try self.columnMaxes();
            defer values.deinit();
            return sparseValueRangeInRange(T, values.data, min_value, max_value);
        }

        pub fn rowMinAbs(self: Self) SparseError!array_mod.Array(T) {
            ensureNumeric(T);
            var out = try self.rowMaxAbs();
            for (self.values, self.row_indices) |value, row| {
                const magnitude = absValue(T, value);
                if (magnitude < out.data[row]) out.data[row] = magnitude;
            }
            return out;
        }

        pub fn rowMaxAbs(self: Self) SparseError!array_mod.Array(T) {
            ensureNumeric(T);
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.rows});
            errdefer out.deinit();
            for (self.values, self.row_indices) |value, row| {
                const magnitude = absValue(T, value);
                if (magnitude > out.data[row]) out.data[row] = magnitude;
            }
            return out;
        }

        pub fn columnMinAbs(self: Self) SparseError!array_mod.Array(T) {
            ensureNumeric(T);
            var out = try self.columnMaxAbs();
            for (self.values, self.col_indices) |value, col| {
                const magnitude = absValue(T, value);
                if (magnitude < out.data[col]) out.data[col] = magnitude;
            }
            return out;
        }

        pub fn columnMaxAbs(self: Self) SparseError!array_mod.Array(T) {
            ensureNumeric(T);
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.cols});
            errdefer out.deinit();
            for (self.values, self.col_indices) |value, col| {
                const magnitude = absValue(T, value);
                if (magnitude > out.data[col]) out.data[col] = magnitude;
            }
            return out;
        }

        pub fn rowSampleVariancesInRange(self: Self, min_variance: f64, max_variance: f64) SparseError!bool {
            return self.rowVariancesInRange(1, min_variance, max_variance);
        }

        pub fn rowSampleStddevsInRange(self: Self, min_stddev: f64, max_stddev: f64) SparseError!bool {
            return self.rowStddevsInRange(1, min_stddev, max_stddev);
        }

        pub fn columnSampleVariancesInRange(self: Self, min_variance: f64, max_variance: f64) SparseError!bool {
            return self.columnVariancesInRange(1, min_variance, max_variance);
        }

        pub fn columnSampleStddevsInRange(self: Self, min_stddev: f64, max_stddev: f64) SparseError!bool {
            return self.columnStddevsInRange(1, min_stddev, max_stddev);
        }

        pub fn rowAbsSums(self: Self) SparseError!array_mod.Array(T) {
            ensureNumeric(T);
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.rows});
            errdefer out.deinit();
            for (self.values, 0..) |value, i| out.data[self.row_indices[i]] += absValue(T, value);
            return out;
        }

        pub fn columnAbsSums(self: Self) SparseError!array_mod.Array(T) {
            ensureNumeric(T);
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.cols});
            errdefer out.deinit();
            for (self.values, 0..) |value, i| out.data[self.col_indices[i]] += absValue(T, value);
            return out;
        }

        pub fn rowMinAbsInRange(self: Self, min_value: T, max_value: T) SparseError!bool {
            var values = try self.rowMinAbs();
            defer values.deinit();
            return sparseValueRangeInRange(T, values.data, min_value, max_value);
        }

        pub fn columnMinAbsInRange(self: Self, min_value: T, max_value: T) SparseError!bool {
            var values = try self.columnMinAbs();
            defer values.deinit();
            return sparseValueRangeInRange(T, values.data, min_value, max_value);
        }

        pub fn rowMaxAbsInRange(self: Self, min_value: T, max_value: T) SparseError!bool {
            var values = try self.rowMaxAbs();
            defer values.deinit();
            return sparseValueRangeInRange(T, values.data, min_value, max_value);
        }

        pub fn columnMaxAbsInRange(self: Self, min_value: T, max_value: T) SparseError!bool {
            var values = try self.columnMaxAbs();
            defer values.deinit();
            return sparseValueRangeInRange(T, values.data, min_value, max_value);
        }

        pub fn rowAbsSumsInRange(self: Self, min_sum: T, max_sum: T) SparseError!bool {
            var sums = try self.rowAbsSums();
            defer sums.deinit();
            return sparseValueRangeInRange(T, sums.data, min_sum, max_sum);
        }

        pub fn columnAbsSumsInRange(self: Self, min_sum: T, max_sum: T) SparseError!bool {
            var sums = try self.columnAbsSums();
            defer sums.deinit();
            return sparseValueRangeInRange(T, sums.data, min_sum, max_sum);
        }

        pub fn rowNorms(self: Self) SparseError!array_mod.Array(T) {
            ensureFloat(T);
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.rows});
            errdefer out.deinit();
            for (self.values, 0..) |value, i| out.data[self.row_indices[i]] += value * value;
            for (out.data) |*value| value.* = @sqrt(value.*);
            return out;
        }

        pub fn columnNorms(self: Self) SparseError!array_mod.Array(T) {
            ensureFloat(T);
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.cols});
            errdefer out.deinit();
            for (self.values, 0..) |value, i| out.data[self.col_indices[i]] += value * value;
            for (out.data) |*value| value.* = @sqrt(value.*);
            return out;
        }

        pub fn rowNormsInRange(self: Self, min_norm: T, max_norm: T) SparseError!bool {
            try validateSparseValueRange(T, zero(T), max_norm);
            if (min_norm < zero(T)) return error.InvalidShape;
            var norms = try self.rowNorms();
            defer norms.deinit();
            return sparseValueRangeInRange(T, norms.data, min_norm, max_norm);
        }

        pub fn columnNormsInRange(self: Self, min_norm: T, max_norm: T) SparseError!bool {
            try validateSparseValueRange(T, zero(T), max_norm);
            if (min_norm < zero(T)) return error.InvalidShape;
            var norms = try self.columnNorms();
            defer norms.deinit();
            return sparseValueRangeInRange(T, norms.data, min_norm, max_norm);
        }

        pub fn get(self: Self, row: usize, col: usize) ?T {
            if (row >= self.rows or col >= self.cols) return null;
            var found = false;
            var total = zero(T);
            for (self.values, 0..) |value, i| {
                if (self.row_indices[i] == row and self.col_indices[i] == col) {
                    total = addSparseValue(T, total, value);
                    found = true;
                }
            }
            return if (found) total else null;
        }

        fn hasEntry(self: Self, row: usize, col: usize) bool {
            if (row >= self.rows or col >= self.cols) return false;
            for (self.row_indices, self.col_indices) |entry_row, entry_col| {
                if (entry_row == row and entry_col == col) return true;
            }
            return false;
        }

        pub fn setExisting(self: *Self, row: usize, col: usize, value: T) SparseError!void {
            if (row >= self.rows or col >= self.cols) return error.IndexOutOfBounds;
            var found = false;
            for (self.row_indices, self.col_indices, self.values) |entry_row, entry_col, *entry_value| {
                if (entry_row == row and entry_col == col) {
                    entry_value.* = value;
                    found = true;
                }
            }
            if (!found) return error.InvalidShape;
        }

        pub fn setStoredValue(self: *Self, row: usize, col: usize, value: T) SparseError!void {
            return self.setExisting(row, col, value);
        }

        pub fn setDiagonal(self: *Self, value: T) SparseError!void {
            if (self.rows != self.cols) return error.NonMatrixArray;
            for (0..self.rows) |index| {
                var found = false;
                for (self.row_indices, self.col_indices, self.values) |entry_row, entry_col, *entry_value| {
                    if (entry_row == index and entry_col == index) {
                        entry_value.* = value;
                        found = true;
                    }
                }
                if (!found) return error.InvalidShape;
            }
        }

        pub fn addToDiagonal(self: *Self, alpha: T) SparseError!void {
            ensureNumeric(T);
            if (alpha == zero(T)) return;
            if (self.rows != self.cols) return error.NonMatrixArray;
            for (0..self.rows) |index| {
                var found = false;
                for (self.row_indices, self.col_indices, self.values) |entry_row, entry_col, *entry_value| {
                    if (entry_row == index and entry_col == index) {
                        entry_value.* += alpha;
                        found = true;
                    }
                }
                if (!found) return error.InvalidShape;
            }
        }

        pub fn setDiagonalValues(self: *Self, diagonal_values: []const T) SparseError!void {
            if (self.rows != self.cols) return error.NonMatrixArray;
            if (diagonal_values.len != self.rows) return error.ShapeMismatch;
            for (0..self.rows) |index| {
                var found = false;
                for (self.row_indices, self.col_indices, self.values) |entry_row, entry_col, *entry_value| {
                    if (entry_row == index and entry_col == index) {
                        entry_value.* = diagonal_values[index];
                        found = true;
                    }
                }
                if (!found) return error.InvalidShape;
            }
        }

        pub fn setDiagonalVector(self: *Self, diagonal_values: []const T) SparseError!void {
            return self.setDiagonalValues(diagonal_values);
        }

        pub fn addDiagonalValues(self: *Self, diagonal_values: []const T) SparseError!void {
            ensureNumeric(T);
            if (self.rows != self.cols) return error.NonMatrixArray;
            if (diagonal_values.len != self.rows) return error.ShapeMismatch;
            var any_nonzero = false;
            for (diagonal_values) |value| {
                if (isNonZero(T, value)) {
                    any_nonzero = true;
                    break;
                }
            }
            if (!any_nonzero) return;
            for (0..self.rows) |index| {
                var found = false;
                for (self.row_indices, self.col_indices, self.values) |entry_row, entry_col, *entry_value| {
                    if (entry_row == index and entry_col == index) {
                        entry_value.* += diagonal_values[index];
                        found = true;
                    }
                }
                if (!found) return error.InvalidShape;
            }
        }

        pub fn addDiagonalVector(self: *Self, diagonal_values: []const T) SparseError!void {
            return self.addDiagonalValues(diagonal_values);
        }

        pub fn multiplyDiagonalValues(self: *Self, diagonal_values: []const T) SparseError!void {
            ensureNumeric(T);
            if (self.rows != self.cols) return error.NonMatrixArray;
            if (diagonal_values.len != self.rows) return error.ShapeMismatch;
            var all_one = true;
            for (diagonal_values) |value| {
                if (value != oneValue(T)) {
                    all_one = false;
                    break;
                }
            }
            if (all_one) return;
            for (0..self.rows) |index| {
                var found = false;
                for (self.row_indices, self.col_indices, self.values) |entry_row, entry_col, *entry_value| {
                    if (entry_row == index and entry_col == index) {
                        entry_value.* *= diagonal_values[index];
                        found = true;
                    }
                }
                if (!found) return error.InvalidShape;
            }
        }

        pub fn multiplyDiagonalVector(self: *Self, diagonal_values: []const T) SparseError!void {
            return self.multiplyDiagonalValues(diagonal_values);
        }

        pub fn diagonal(self: Self) SparseError!array_mod.Array(T) {
            if (self.rows != self.cols) return error.NonMatrixArray;
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.rows});
            errdefer out.deinit();
            for (self.values, 0..) |value, i| {
                const row = self.row_indices[i];
                if (row == self.col_indices[i]) out.data[row] = addSparseValue(T, out.data[row], value);
            }
            return out;
        }

        pub fn diagonalOffset(self: Self, offset: isize) SparseError!array_mod.Array(T) {
            return sparseDenseDiagonalOffset(T, self, offset);
        }

        pub fn diagonalWithOffset(self: Self, offset: isize) SparseError!array_mod.Array(T) {
            return self.diagonalOffset(offset);
        }

        pub fn minAbsDiagonal(self: Self) SparseError!T {
            var diagonal_values = try self.diagonal();
            defer diagonal_values.deinit();
            return (try sparseDiagonalAbsRange(T, diagonal_values.data)).min_abs;
        }

        pub fn maxAbsDiagonal(self: Self) SparseError!T {
            var diagonal_values = try self.diagonal();
            defer diagonal_values.deinit();
            return (try sparseDiagonalAbsRange(T, diagonal_values.data)).max_abs;
        }

        pub fn diagonalDynamicRange(self: Self) SparseError!f64 {
            var diagonal_values = try self.diagonal();
            defer diagonal_values.deinit();
            return sparseDiagonalDynamicRangeFromValues(T, diagonal_values.data);
        }

        pub fn diagonalDynamicRangeMeetsBound(self: Self, max_dynamic_range: f64) SparseError!bool {
            var diagonal_values = try self.diagonal();
            defer diagonal_values.deinit();
            return sparseDiagonalDynamicRangeMeetsBoundFromValues(T, diagonal_values.data, max_dynamic_range);
        }

        pub fn trace(self: Self) SparseError!T {
            ensureNumeric(T);
            if (self.rows != self.cols) return error.NonMatrixArray;
            var total = zero(T);
            for (self.values, 0..) |value, i| {
                if (self.row_indices[i] == self.col_indices[i]) total += value;
            }
            return total;
        }

        pub fn traceInRange(self: Self, min_value: T, max_value: T) SparseError!bool {
            try validateSparseValueRange(T, min_value, max_value);
            const trace_value = try self.trace();
            return trace_value >= min_value and trace_value <= max_value;
        }

        pub fn normalizedTrace(self: Self) SparseError!f64 {
            return sparseNormalizedTraceFromTrace(T, try self.trace(), self.rows);
        }

        pub fn normalizedTraceInRange(self: Self, min_value: f64, max_value: f64) SparseError!bool {
            try validateFiniteRange(min_value, max_value);
            return sparseNormalizedTraceInRangeFromTrace(T, try self.trace(), self.rows, min_value, max_value);
        }

        pub fn missingDiagonalCount(self: Self) SparseError!usize {
            if (self.rows != self.cols) return error.NonMatrixArray;
            var seen = try self.allocator.alloc(bool, self.rows);
            defer self.allocator.free(seen);
            @memset(seen, false);
            for (self.row_indices, self.col_indices) |row, col| {
                if (row == col) seen[row] = true;
            }
            var count: usize = 0;
            for (seen) |present| {
                if (!present) count += 1;
            }
            return count;
        }

        pub fn missingDiagonalCountMeetsBound(self: Self, max_count: usize) SparseError!bool {
            return (try self.missingDiagonalCount()) <= max_count;
        }

        pub fn missingDiagonalCountInRange(self: Self, min_count: usize, max_count: usize) SparseError!bool {
            try validateCountRange(min_count, max_count);
            return sparseCountInValidatedRange(try self.missingDiagonalCount(), min_count, max_count);
        }

        pub fn zeroDiagonalCount(self: Self) SparseError!usize {
            if (self.rows != self.cols) return error.NonMatrixArray;
            var seen = try self.allocator.alloc(bool, self.rows);
            defer self.allocator.free(seen);
            @memset(seen, false);
            var diagonal_values = try self.allocator.alloc(T, self.rows);
            defer self.allocator.free(diagonal_values);
            @memset(diagonal_values, zero(T));

            for (self.values, 0..) |value, i| {
                const row = self.row_indices[i];
                if (row == self.col_indices[i]) {
                    diagonal_values[row] = addSparseValue(T, diagonal_values[row], value);
                    seen[row] = true;
                }
            }
            var count: usize = 0;
            for (seen, diagonal_values) |present, value| {
                if (present and value == zero(T)) count += 1;
            }
            return count;
        }

        pub fn zeroDiagonalCountMeetsBound(self: Self, max_count: usize) SparseError!bool {
            return (try self.zeroDiagonalCount()) <= max_count;
        }

        pub fn zeroDiagonalCountInRange(self: Self, min_count: usize, max_count: usize) SparseError!bool {
            try validateCountRange(min_count, max_count);
            return sparseCountInValidatedRange(try self.zeroDiagonalCount(), min_count, max_count);
        }

        pub fn nonPositiveDiagonalCount(self: Self) SparseError!usize {
            ensureNumeric(T);
            if (self.rows != self.cols) return error.NonMatrixArray;
            var seen = try self.allocator.alloc(bool, self.rows);
            defer self.allocator.free(seen);
            @memset(seen, false);
            var diagonal_values = try self.allocator.alloc(T, self.rows);
            defer self.allocator.free(diagonal_values);
            @memset(diagonal_values, zero(T));

            for (self.values, 0..) |value, i| {
                const row = self.row_indices[i];
                if (row == self.col_indices[i]) {
                    diagonal_values[row] = addSparseValue(T, diagonal_values[row], value);
                    seen[row] = true;
                }
            }

            var count: usize = 0;
            for (seen, diagonal_values) |present, value| {
                if (present and value <= zero(T)) count += 1;
            }
            return count;
        }

        pub fn nonPositiveDiagonalCountMeetsBound(self: Self, max_count: usize) SparseError!bool {
            return (try self.nonPositiveDiagonalCount()) <= max_count;
        }

        pub fn nonPositiveDiagonalCountInRange(self: Self, min_count: usize, max_count: usize) SparseError!bool {
            try validateCountRange(min_count, max_count);
            return sparseCountInValidatedRange(try self.nonPositiveDiagonalCount(), min_count, max_count);
        }

        pub fn bandwidth(self: Self) SparseError!usize {
            if (self.rows != self.cols) return error.NonMatrixArray;
            var bw: usize = 0;
            for (self.row_indices, self.col_indices) |row, col| {
                const distance = if (row > col) row - col else col - row;
                if (distance > bw) bw = distance;
            }
            return bw;
        }

        pub fn bandwidthMeetsBound(self: Self, max_bandwidth: usize) SparseError!bool {
            if (self.rows != self.cols) return error.NonMatrixArray;
            for (self.row_indices, self.col_indices) |row, col| {
                const distance = if (row > col) row - col else col - row;
                if (distance > max_bandwidth) return false;
            }
            return true;
        }

        pub fn columnIntersectionBandwidth(self: Self) SparseError!usize {
            return sparseColumnIntersectionBandwidthFromCoo(self.allocator, self.rows, self.cols, self.row_indices, self.col_indices);
        }

        pub fn columnIntersectionBandwidthMeetsBound(self: Self, max_bandwidth: usize) SparseError!bool {
            return sparseColumnIntersectionBandwidthMeetsBoundFromCoo(self.allocator, self.rows, self.cols, self.row_indices, self.col_indices, max_bandwidth);
        }

        pub fn lowerNnz(self: Self, comptime strict: bool) SparseError!usize {
            if (self.rows != self.cols) return error.NonMatrixArray;
            var count: usize = 0;
            for (self.row_indices, self.col_indices) |row, col| {
                if (triangularIndexMatches(row, col, strict, true)) count += 1;
            }
            return count;
        }

        pub fn upperNnz(self: Self, comptime strict: bool) SparseError!usize {
            if (self.rows != self.cols) return error.NonMatrixArray;
            var count: usize = 0;
            for (self.row_indices, self.col_indices) |row, col| {
                if (triangularIndexMatches(row, col, strict, false)) count += 1;
            }
            return count;
        }

        pub fn lowerNnzMeetsBound(self: Self, comptime strict: bool, max_count: usize) SparseError!bool {
            return (try self.lowerNnz(strict)) <= max_count;
        }

        pub fn upperNnzMeetsBound(self: Self, comptime strict: bool, max_count: usize) SparseError!bool {
            return (try self.upperNnz(strict)) <= max_count;
        }

        pub fn lowerNnzInRange(self: Self, comptime strict: bool, min_count: usize, max_count: usize) SparseError!bool {
            if (min_count > max_count) return error.InvalidShape;
            const count = try self.lowerNnz(strict);
            return count >= min_count and count <= max_count;
        }

        pub fn upperNnzInRange(self: Self, comptime strict: bool, min_count: usize, max_count: usize) SparseError!bool {
            if (min_count > max_count) return error.InvalidShape;
            const count = try self.upperNnz(strict);
            return count >= min_count and count <= max_count;
        }

        pub fn lowerProfile(self: Self) SparseError!usize {
            if (self.rows != self.cols) return error.NonMatrixArray;
            var builder = try SparseProfileBuilder.init(self.allocator, self.rows);
            defer builder.deinit(self.allocator);
            for (self.row_indices, self.col_indices) |row, col| builder.observe(row, col);
            return builder.lowerProfile();
        }

        pub fn upperProfile(self: Self) SparseError!usize {
            if (self.rows != self.cols) return error.NonMatrixArray;
            var builder = try SparseProfileBuilder.init(self.allocator, self.rows);
            defer builder.deinit(self.allocator);
            for (self.row_indices, self.col_indices) |row, col| builder.observe(row, col);
            return builder.upperProfile();
        }

        pub fn profile(self: Self) SparseError!SparseProfile {
            if (self.rows != self.cols) return error.NonMatrixArray;
            var builder = try SparseProfileBuilder.init(self.allocator, self.rows);
            defer builder.deinit(self.allocator);
            for (self.row_indices, self.col_indices) |row, col| builder.observe(row, col);
            return builder.profile();
        }

        pub fn lowerProfileMeetsBound(self: Self, max_profile: usize) SparseError!bool {
            return (try self.lowerProfile()) <= max_profile;
        }

        pub fn upperProfileMeetsBound(self: Self, max_profile: usize) SparseError!bool {
            return (try self.upperProfile()) <= max_profile;
        }

        pub fn profileMeetsBounds(self: Self, max_lower_profile: usize, max_upper_profile: usize) SparseError!bool {
            const current = try self.profile();
            return current.meetsBounds(max_lower_profile, max_upper_profile);
        }

        pub fn profileTotalMeetsBound(self: Self, max_total_profile: usize) SparseError!bool {
            const current = try self.profile();
            return current.totalMeetsBound(max_total_profile);
        }

        pub fn diagonallyDominant(self: Self) SparseError!bool {
            var canonical = try self.coalesced();
            defer canonical.deinit();
            return sparseDiagonalDominanceFromCanonicalEntries(
                T,
                self.allocator,
                canonical.rows,
                canonical.cols,
                canonical.row_indices,
                canonical.col_indices,
                canonical.values,
                false,
            );
        }

        pub fn diagonalDominanceMargin(self: Self) SparseError!f64 {
            var canonical = try self.coalesced();
            defer canonical.deinit();
            return sparseDiagonalDominanceMarginFromCanonicalEntries(
                T,
                self.allocator,
                canonical.rows,
                canonical.cols,
                canonical.row_indices,
                canonical.col_indices,
                canonical.values,
            );
        }

        pub fn diagonalDominanceMarginMeetsBound(self: Self, min_margin: f64) SparseError!bool {
            var canonical = try self.coalesced();
            defer canonical.deinit();
            return sparseDiagonalDominanceMarginMeetsBoundFromCanonicalEntries(
                T,
                self.allocator,
                canonical.rows,
                canonical.cols,
                canonical.row_indices,
                canonical.col_indices,
                canonical.values,
                min_margin,
            );
        }

        pub fn strictlyDiagonallyDominant(self: Self) SparseError!bool {
            var canonical = try self.coalesced();
            defer canonical.deinit();
            return sparseDiagonalDominanceFromCanonicalEntries(
                T,
                self.allocator,
                canonical.rows,
                canonical.cols,
                canonical.row_indices,
                canonical.col_indices,
                canonical.values,
                true,
            );
        }

        pub fn structurallySymmetric(self: Self) SparseError!bool {
            if (self.rows != self.cols) return error.NonMatrixArray;
            for (self.row_indices, self.col_indices) |row, col| {
                if (!self.hasEntry(col, row)) return false;
            }
            return true;
        }

        pub fn numericallySymmetric(self: Self, tolerance: T) SparseError!bool {
            ensureNumeric(T);
            if (self.rows != self.cols) return error.NonMatrixArray;
            for (self.row_indices, self.col_indices) |row, col| {
                const value = self.get(row, col) orelse return false;
                const mirror = self.get(col, row) orelse return false;
                if (absDifference(T, value, mirror) > tolerance) return false;
            }
            return true;
        }

        pub fn symmetryResidualFrobeniusNorm(self: Self) SparseError!T {
            ensureFloat(T);
            var dense = try self.toDense();
            defer dense.deinit();
            return sparseSymmetryResidualFrobeniusNormFromDense(T, dense.data, self.rows, self.cols);
        }

        pub fn symmetryRelativeResidualFrobeniusNorm(self: Self) SparseError!T {
            const residual = try self.symmetryResidualFrobeniusNorm();
            return residual / @max(oneValue(T), self.frobeniusNorm());
        }

        pub fn symmetryResidualFrobeniusNormMeetsBound(self: Self, max_residual: T) SparseError!bool {
            try validateSparseValueRange(T, zero(T), max_residual);
            return (try self.symmetryResidualFrobeniusNorm()) <= max_residual;
        }

        pub fn symmetryRelativeResidualFrobeniusNormMeetsBound(self: Self, max_relative_residual: T) SparseError!bool {
            try validateSparseValueRange(T, zero(T), max_relative_residual);
            return (try self.symmetryRelativeResidualFrobeniusNorm()) <= max_relative_residual;
        }

        pub fn toDense(self: Self) SparseError!array_mod.Array(T) {
            var out = try array_mod.Array(T).zeros(self.allocator, &.{ self.rows, self.cols });
            errdefer out.deinit();
            for (self.values, 0..) |value, i| {
                const index = self.row_indices[i] * self.cols + self.col_indices[i];
                out.data[index] = addSparseValue(T, out.data[index], value);
            }
            return out;
        }

        pub fn matvec(self: Self, x: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            if (x.shape.len != 1) return error.NonVectorArray;
            if (x.shape[0] != self.cols) return error.ShapeMismatch;
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.rows});
            errdefer out.deinit();
            for (self.values, 0..) |value, i| {
                out.data[self.row_indices[i]] += value * x.data[self.col_indices[i]];
            }
            return out;
        }

        pub fn matvecResidualNorm(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T)) SparseError!T {
            ensureFloat(T);
            if (rhs.shape.len != 1) return error.NonVectorArray;
            var predicted = try self.matvec(x);
            defer predicted.deinit();
            return sparseVectorResidualNorm(T, predicted.data, rhs.data);
        }

        pub fn matvecResidualNormMeetsBound(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T), max_residual: T) SparseError!bool {
            try validateSparseValueRange(T, zero(T), max_residual);
            return (try self.matvecResidualNorm(x, rhs)) <= max_residual;
        }

        pub fn matvecRelativeResidualNorm(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T)) SparseError!T {
            const residual = try self.matvecResidualNorm(x, rhs);
            return sparseRelativeResidualNorm(T, residual, self.frobeniusNorm(), x.data, rhs.data);
        }

        pub fn matvecResidualSummary(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T)) SparseError!SparseResidualSummary {
            const residual = try self.matvecResidualNorm(x, rhs);
            return sparseResidualSummary(T, residual, self.frobeniusNorm(), x.data, rhs.data);
        }

        pub fn matvecResidualSummaryMeetsBounds(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T), max_residual: f64, max_relative_residual: f64) SparseError!bool {
            const summary = try self.matvecResidualSummary(x, rhs);
            return summary.meetsBounds(max_residual, max_relative_residual);
        }

        pub fn matvecRelativeResidualNormMeetsBound(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T), max_relative_residual: T) SparseError!bool {
            try validateSparseValueRange(T, zero(T), max_relative_residual);
            return (try self.matvecRelativeResidualNorm(x, rhs)) <= max_relative_residual;
        }

        pub fn matmat(self: Self, rhs: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            if (rhs.shape.len != 2) return error.NonMatrixArray;
            if (rhs.shape[0] != self.cols) return error.ShapeMismatch;
            var out = try array_mod.Array(T).zeros(self.allocator, &.{ self.rows, rhs.shape[1] });
            errdefer out.deinit();
            for (self.values, 0..) |value, i| {
                const row = self.row_indices[i];
                const col = self.col_indices[i];
                for (0..rhs.shape[1]) |out_col| {
                    out.data[row * rhs.shape[1] + out_col] += value * rhs.data[col * rhs.shape[1] + out_col];
                }
            }
            return out;
        }

        pub fn matmatResidualFrobeniusNorm(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T)) SparseError!T {
            ensureFloat(T);
            var predicted = try self.matmat(x);
            defer predicted.deinit();
            return sparseMatrixResidualNorm(T, predicted, rhs);
        }

        pub fn matmatResidualFrobeniusNormMeetsBound(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T), max_residual: T) SparseError!bool {
            try validateSparseValueRange(T, zero(T), max_residual);
            return (try self.matmatResidualFrobeniusNorm(x, rhs)) <= max_residual;
        }

        pub fn matmatRelativeResidualFrobeniusNorm(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T)) SparseError!T {
            const residual = try self.matmatResidualFrobeniusNorm(x, rhs);
            return sparseRelativeMatrixResidualNorm(T, residual, self.frobeniusNorm(), x.data, rhs.data);
        }

        pub fn matmatResidualSummary(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T)) SparseError!SparseResidualSummary {
            const residual = try self.matmatResidualFrobeniusNorm(x, rhs);
            return sparseResidualSummary(T, residual, self.frobeniusNorm(), x.data, rhs.data);
        }

        pub fn matmatResidualSummaryMeetsBounds(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T), max_residual: f64, max_relative_residual: f64) SparseError!bool {
            const summary = try self.matmatResidualSummary(x, rhs);
            return summary.meetsBounds(max_residual, max_relative_residual);
        }

        pub fn matmatRelativeResidualFrobeniusNormMeetsBound(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T), max_relative_residual: T) SparseError!bool {
            try validateSparseValueRange(T, zero(T), max_relative_residual);
            return (try self.matmatRelativeResidualFrobeniusNorm(x, rhs)) <= max_relative_residual;
        }

        pub fn matmulSparse(self: Self, rhs: Self) SparseError!Self {
            if (self.cols != rhs.rows) return error.ShapeMismatch;
            var lhs_csr = try self.toCsr();
            defer lhs_csr.deinit();
            var rhs_csr = try rhs.toCsr();
            defer rhs_csr.deinit();
            var product = try lhs_csr.matmulSparse(rhs_csr);
            defer product.deinit();
            return product.toCoo();
        }

        pub fn transposeMatvec(self: Self, x: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            if (x.shape.len != 1) return error.NonVectorArray;
            if (x.shape[0] != self.rows) return error.ShapeMismatch;
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.cols});
            errdefer out.deinit();
            for (self.values, 0..) |value, i| {
                out.data[self.col_indices[i]] += value * x.data[self.row_indices[i]];
            }
            return out;
        }

        pub fn transposeMatvecResidualNorm(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T)) SparseError!T {
            ensureFloat(T);
            if (rhs.shape.len != 1) return error.NonVectorArray;
            var predicted = try self.transposeMatvec(x);
            defer predicted.deinit();
            return sparseVectorResidualNorm(T, predicted.data, rhs.data);
        }

        pub fn transposeMatvecResidualNormMeetsBound(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T), max_residual: T) SparseError!bool {
            try validateSparseValueRange(T, zero(T), max_residual);
            return (try self.transposeMatvecResidualNorm(x, rhs)) <= max_residual;
        }

        pub fn transposeMatvecRelativeResidualNorm(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T)) SparseError!T {
            const residual = try self.transposeMatvecResidualNorm(x, rhs);
            return sparseRelativeResidualNorm(T, residual, self.frobeniusNorm(), x.data, rhs.data);
        }

        pub fn transposeMatvecResidualSummary(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T)) SparseError!SparseResidualSummary {
            const residual = try self.transposeMatvecResidualNorm(x, rhs);
            return sparseResidualSummary(T, residual, self.frobeniusNorm(), x.data, rhs.data);
        }

        pub fn transposeMatvecResidualSummaryMeetsBounds(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T), max_residual: f64, max_relative_residual: f64) SparseError!bool {
            const summary = try self.transposeMatvecResidualSummary(x, rhs);
            return summary.meetsBounds(max_residual, max_relative_residual);
        }

        pub fn transposeMatvecRelativeResidualNormMeetsBound(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T), max_relative_residual: T) SparseError!bool {
            try validateSparseValueRange(T, zero(T), max_relative_residual);
            return (try self.transposeMatvecRelativeResidualNorm(x, rhs)) <= max_relative_residual;
        }

        pub fn transposeMatmat(self: Self, rhs: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            if (rhs.shape.len != 2) return error.NonMatrixArray;
            if (rhs.shape[0] != self.rows) return error.ShapeMismatch;
            var out = try array_mod.Array(T).zeros(self.allocator, &.{ self.cols, rhs.shape[1] });
            errdefer out.deinit();
            for (self.values, 0..) |value, i| {
                const row = self.row_indices[i];
                const col = self.col_indices[i];
                for (0..rhs.shape[1]) |rhs_col| {
                    out.data[col * rhs.shape[1] + rhs_col] += value * rhs.data[row * rhs.shape[1] + rhs_col];
                }
            }
            return out;
        }

        pub fn transposeMatmatResidualFrobeniusNorm(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T)) SparseError!T {
            ensureFloat(T);
            var predicted = try self.transposeMatmat(x);
            defer predicted.deinit();
            return sparseMatrixResidualNorm(T, predicted, rhs);
        }

        pub fn transposeMatmatResidualFrobeniusNormMeetsBound(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T), max_residual: T) SparseError!bool {
            try validateSparseValueRange(T, zero(T), max_residual);
            return (try self.transposeMatmatResidualFrobeniusNorm(x, rhs)) <= max_residual;
        }

        pub fn transposeMatmatRelativeResidualFrobeniusNorm(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T)) SparseError!T {
            const residual = try self.transposeMatmatResidualFrobeniusNorm(x, rhs);
            return sparseRelativeMatrixResidualNorm(T, residual, self.frobeniusNorm(), x.data, rhs.data);
        }

        pub fn transposeMatmatResidualSummary(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T)) SparseError!SparseResidualSummary {
            const residual = try self.transposeMatmatResidualFrobeniusNorm(x, rhs);
            return sparseResidualSummary(T, residual, self.frobeniusNorm(), x.data, rhs.data);
        }

        pub fn transposeMatmatResidualSummaryMeetsBounds(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T), max_residual: f64, max_relative_residual: f64) SparseError!bool {
            const summary = try self.transposeMatmatResidualSummary(x, rhs);
            return summary.meetsBounds(max_residual, max_relative_residual);
        }

        pub fn transposeMatmatRelativeResidualFrobeniusNormMeetsBound(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T), max_relative_residual: T) SparseError!bool {
            try validateSparseValueRange(T, zero(T), max_relative_residual);
            return (try self.transposeMatmatRelativeResidualFrobeniusNorm(x, rhs)) <= max_relative_residual;
        }

        pub fn transpose(self: Self) SparseError!Self {
            const row_indices = try self.allocator.dupe(usize, self.col_indices);
            errdefer self.allocator.free(row_indices);
            const col_indices = try self.allocator.dupe(usize, self.row_indices);
            errdefer self.allocator.free(col_indices);
            const values = try self.allocator.dupe(T, self.values);
            errdefer self.allocator.free(values);
            return .{
                .allocator = self.allocator,
                .rows = self.cols,
                .cols = self.rows,
                .row_indices = row_indices,
                .col_indices = col_indices,
                .values = values,
            };
        }

        pub fn matrixTranspose(self: Self) SparseError!Self {
            return self.transpose();
        }

        pub fn mT(self: Self) SparseError!Self {
            return self.matrixTranspose();
        }

        pub fn toCsr(self: Self) SparseError!CsrMatrix(T) {
            var row_offsets = try self.allocator.alloc(usize, self.rows + 1);
            errdefer self.allocator.free(row_offsets);
            @memset(row_offsets, 0);
            for (self.row_indices) |row| row_offsets[row + 1] += 1;
            for (1..row_offsets.len) |i| row_offsets[i] += row_offsets[i - 1];

            var col_indices = try self.allocator.alloc(usize, self.values.len);
            errdefer self.allocator.free(col_indices);
            var values = try self.allocator.alloc(T, self.values.len);
            errdefer self.allocator.free(values);
            const next = try self.allocator.dupe(usize, row_offsets[0..self.rows]);
            defer self.allocator.free(next);

            for (self.values, 0..) |value, i| {
                const row = self.row_indices[i];
                const dst = next[row];
                col_indices[dst] = self.col_indices[i];
                values[dst] = value;
                next[row] += 1;
            }
            return .{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = self.cols,
                .row_offsets = row_offsets,
                .col_indices = col_indices,
                .values = values,
            };
        }

        pub fn toCsc(self: Self) SparseError!CscMatrix(T) {
            var col_offsets = try self.allocator.alloc(usize, self.cols + 1);
            errdefer self.allocator.free(col_offsets);
            @memset(col_offsets, 0);
            for (self.col_indices) |col| col_offsets[col + 1] += 1;
            for (1..col_offsets.len) |i| col_offsets[i] += col_offsets[i - 1];

            var row_indices = try self.allocator.alloc(usize, self.values.len);
            errdefer self.allocator.free(row_indices);
            var values = try self.allocator.alloc(T, self.values.len);
            errdefer self.allocator.free(values);
            const next = try self.allocator.dupe(usize, col_offsets[0..self.cols]);
            defer self.allocator.free(next);

            for (self.values, 0..) |value, i| {
                const col = self.col_indices[i];
                const dst = next[col];
                row_indices[dst] = self.row_indices[i];
                values[dst] = value;
                next[col] += 1;
            }
            return .{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = self.cols,
                .col_offsets = col_offsets,
                .row_indices = row_indices,
                .values = values,
            };
        }
    };
}

pub fn CsrMatrix(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        rows: usize,
        cols: usize,
        row_offsets: []usize,
        col_indices: []usize,
        values: []T,

        pub fn eye(allocator: std.mem.Allocator, rows: usize, cols: usize) SparseError!Self {
            const diag_len = @min(rows, cols);
            var row_offsets = try allocator.alloc(usize, rows + 1);
            errdefer allocator.free(row_offsets);
            var col_indices = try allocator.alloc(usize, diag_len);
            errdefer allocator.free(col_indices);
            var values = try allocator.alloc(T, diag_len);
            errdefer allocator.free(values);

            var write: usize = 0;
            row_offsets[0] = 0;
            for (0..rows) |row| {
                if (row < cols) {
                    col_indices[write] = row;
                    values[write] = oneValue(T);
                    write += 1;
                }
                row_offsets[row + 1] = write;
            }
            std.debug.assert(write == diag_len);
            return .{
                .allocator = allocator,
                .rows = rows,
                .cols = cols,
                .row_offsets = row_offsets,
                .col_indices = col_indices,
                .values = values,
            };
        }

        pub fn identity(allocator: std.mem.Allocator, size: usize) SparseError!Self {
            return Self.eye(allocator, size, size);
        }

        pub fn fromDiagonal(allocator: std.mem.Allocator, diagonal_values: []const T, offset: isize) SparseError!Self {
            var coo = try CooMatrix(T).fromDiagonal(allocator, diagonal_values, offset);
            defer coo.deinit();
            return coo.toCsr();
        }

        pub fn fromDiagonalNnz(diagonal_values: []const T, offset: isize) SparseError!usize {
            return CooMatrix(T).fromDiagonalNnz(diagonal_values, offset);
        }

        pub fn fromDiagonalPruned(allocator: std.mem.Allocator, diagonal_values: []const T, offset: isize, tolerance: T) SparseError!Self {
            var coo = try CooMatrix(T).fromDiagonalPruned(allocator, diagonal_values, offset, tolerance);
            defer coo.deinit();
            return coo.toCsr();
        }

        pub fn fromDiagonalPrunedNnz(diagonal_values: []const T, offset: isize, tolerance: T) SparseError!usize {
            return CooMatrix(T).fromDiagonalPrunedNnz(diagonal_values, offset, tolerance);
        }

        pub fn fromCompressedSlices(
            allocator: std.mem.Allocator,
            rows: usize,
            cols: usize,
            row_offsets: []const usize,
            col_indices: []const usize,
            values: []const T,
        ) SparseError!Self {
            if (row_offsets.len != rows + 1) return error.ShapeMismatch;
            if (col_indices.len != values.len) return error.ShapeMismatch;
            if (row_offsets[0] != 0 or row_offsets[row_offsets.len - 1] != values.len) return error.ShapeMismatch;
            for (row_offsets[1..], row_offsets[0 .. row_offsets.len - 1]) |current, previous| {
                if (current < previous) return error.ShapeMismatch;
            }
            for (col_indices) |col| if (col >= cols) return error.IndexOutOfBounds;
            return .{
                .allocator = allocator,
                .rows = rows,
                .cols = cols,
                .row_offsets = try allocator.dupe(usize, row_offsets),
                .col_indices = try allocator.dupe(usize, col_indices),
                .values = try allocator.dupe(T, values),
            };
        }

        pub fn fromDense(input: array_mod.Array(T)) SparseError!Self {
            if (input.shape.len != 2) return error.NonMatrixArray;
            const rows = input.shape[0];
            const cols = input.shape[1];
            const nonzero_count = try Self.fromDenseNnz(input);

            var row_offsets = try input.allocator.alloc(usize, rows + 1);
            errdefer input.allocator.free(row_offsets);
            var col_indices = try input.allocator.alloc(usize, nonzero_count);
            errdefer input.allocator.free(col_indices);
            var values = try input.allocator.alloc(T, nonzero_count);
            errdefer input.allocator.free(values);

            var write: usize = 0;
            row_offsets[0] = 0;
            for (0..rows) |r| {
                for (0..cols) |c| {
                    const value = input.data[r * cols + c];
                    if (isNonZero(T, value)) {
                        col_indices[write] = c;
                        values[write] = value;
                        write += 1;
                    }
                }
                row_offsets[r + 1] = write;
            }
            return .{
                .allocator = input.allocator,
                .rows = rows,
                .cols = cols,
                .row_offsets = row_offsets,
                .col_indices = col_indices,
                .values = values,
            };
        }

        pub fn fromDenseNnz(input: array_mod.Array(T)) SparseError!usize {
            return sparseDenseNnz(T, input);
        }

        pub fn fromDensePruned(input: array_mod.Array(T), tolerance: T) SparseError!Self {
            if (input.shape.len != 2) return error.NonMatrixArray;
            const rows = input.shape[0];
            const cols = input.shape[1];
            const keep_count = try Self.fromDensePrunedNnz(input, tolerance);

            var row_offsets = try input.allocator.alloc(usize, rows + 1);
            errdefer input.allocator.free(row_offsets);
            var col_indices = try input.allocator.alloc(usize, keep_count);
            errdefer input.allocator.free(col_indices);
            var values = try input.allocator.alloc(T, keep_count);
            errdefer input.allocator.free(values);

            var write: usize = 0;
            row_offsets[0] = 0;
            for (0..rows) |row| {
                for (0..cols) |col| {
                    const value = input.data[row * cols + col];
                    if (sparseAbsValueExceedsTolerance(T, value, tolerance)) {
                        col_indices[write] = col;
                        values[write] = value;
                        write += 1;
                    }
                }
                row_offsets[row + 1] = write;
            }
            std.debug.assert(write == keep_count);
            return .{
                .allocator = input.allocator,
                .rows = rows,
                .cols = cols,
                .row_offsets = row_offsets,
                .col_indices = col_indices,
                .values = values,
            };
        }

        pub fn fromDensePrunedNnz(input: array_mod.Array(T), tolerance: T) SparseError!usize {
            return sparseDensePrunedNnz(T, input, tolerance);
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.row_offsets);
            self.allocator.free(self.col_indices);
            self.allocator.free(self.values);
            self.* = undefined;
        }

        pub fn fillStoredValues(self: *Self, value: T) void {
            @memset(self.values, value);
        }

        pub fn fillValues(self: *Self, value: T) void {
            self.fillStoredValues(value);
        }

        pub fn scaleStoredValues(self: *Self, alpha: T) void {
            ensureNumeric(T);
            for (self.values) |*value| value.* = value.* * alpha;
        }

        pub fn scaleValues(self: *Self, alpha: T) void {
            self.scaleStoredValues(alpha);
        }

        pub fn setStoredValuesFromDense(self: *Self, src: array_mod.Array(T)) SparseError!void {
            if (src.shape.len != 2) return error.NonMatrixArray;
            if (src.shape[0] != self.rows or src.shape[1] != self.cols) return error.ShapeMismatch;
            for (0..self.rows) |row| {
                for (self.row_offsets[row]..self.row_offsets[row + 1]) |pos| {
                    self.values[pos] = src.data[row * self.cols + self.col_indices[pos]];
                }
            }
        }

        pub fn clone(self: Self) SparseError!Self {
            const row_offsets = try self.allocator.dupe(usize, self.row_offsets);
            errdefer self.allocator.free(row_offsets);
            const col_indices = try self.allocator.dupe(usize, self.col_indices);
            errdefer self.allocator.free(col_indices);
            const values = try self.allocator.dupe(T, self.values);
            errdefer self.allocator.free(values);
            return .{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = self.cols,
                .row_offsets = row_offsets,
                .col_indices = col_indices,
                .values = values,
            };
        }

        pub fn copy(self: Self) SparseError!Self {
            return self.clone();
        }

        pub fn detach(self: Self) SparseError!Self {
            return self.clone();
        }

        pub fn astype(self: Self, comptime U: type) SparseError!CsrMatrix(U) {
            const row_offsets = try self.allocator.dupe(usize, self.row_offsets);
            errdefer self.allocator.free(row_offsets);
            const col_indices = try self.allocator.dupe(usize, self.col_indices);
            errdefer self.allocator.free(col_indices);
            var values = try self.allocator.alloc(U, self.values.len);
            errdefer self.allocator.free(values);
            for (self.values, 0..) |value, index| {
                values[index] = sparseCastValue(U, value);
            }
            return .{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = self.cols,
                .row_offsets = row_offsets,
                .col_indices = col_indices,
                .values = values,
            };
        }

        pub fn mapValues(self: Self, comptime U: type, comptime map: fn (T) U) SparseError!CsrMatrix(U) {
            const row_offsets = try self.allocator.dupe(usize, self.row_offsets);
            errdefer self.allocator.free(row_offsets);
            const col_indices = try self.allocator.dupe(usize, self.col_indices);
            errdefer self.allocator.free(col_indices);
            var values = try self.allocator.alloc(U, self.values.len);
            errdefer self.allocator.free(values);
            for (self.values, 0..) |value, index| {
                values[index] = map(value);
            }
            return .{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = self.cols,
                .row_offsets = row_offsets,
                .col_indices = col_indices,
                .values = values,
            };
        }

        pub fn mapStoredValues(self: Self, comptime U: type, comptime map: fn (T) U) SparseError!CsrMatrix(U) {
            return self.mapValues(U, map);
        }

        pub fn mapValuesInPlace(self: *Self, comptime map: fn (T) T) void {
            for (self.values) |*value| value.* = map(value.*);
        }

        pub fn mapStoredValuesInPlace(self: *Self, comptime map: fn (T) T) void {
            self.mapValuesInPlace(map);
        }

        pub fn positive(self: Self) SparseError!Self {
            ensureNumeric(T);
            return self.clone();
        }

        pub fn abs(self: Self) SparseError!Self {
            ensureNumeric(T);
            const out = try self.clone();
            for (out.values) |*value| value.* = absValue(T, value.*);
            return out;
        }

        pub fn absolute(self: Self) SparseError!Self {
            return self.abs();
        }

        pub fn fabs(self: Self) SparseError!Self {
            return self.abs();
        }

        pub fn square(self: Self) SparseError!Self {
            ensureNumeric(T);
            const out = try self.clone();
            for (out.values) |*value| value.* = value.* * value.*;
            return out;
        }

        pub fn sqrt(self: Self) SparseError!Self {
            ensureFloat(T);
            const out = try self.clone();
            for (out.values) |*value| value.* = @sqrt(value.*);
            return out;
        }

        pub fn rsqrt(self: Self) SparseError!Self {
            ensureFloat(T);
            const out = try self.clone();
            for (out.values) |*value| value.* = oneValue(T) / @sqrt(value.*);
            return out;
        }

        pub fn cbrt(self: Self) SparseError!Self {
            ensureFloat(T);
            const out = try self.clone();
            for (out.values) |*value| value.* = cbrtSparseValue(T, value.*);
            return out;
        }

        pub fn reciprocal(self: Self) SparseError!Self {
            ensureFloat(T);
            const out = try self.clone();
            for (out.values) |*value| value.* = oneValue(T) / value.*;
            return out;
        }

        pub fn floor(self: Self) SparseError!Self {
            ensureNumeric(T);
            const out = try self.clone();
            for (out.values) |*value| value.* = floorSparseValue(T, value.*);
            return out;
        }

        pub fn ceil(self: Self) SparseError!Self {
            ensureNumeric(T);
            const out = try self.clone();
            for (out.values) |*value| value.* = ceilSparseValue(T, value.*);
            return out;
        }

        pub fn round(self: Self) SparseError!Self {
            ensureNumeric(T);
            const out = try self.clone();
            for (out.values) |*value| value.* = roundSparseValue(T, value.*);
            return out;
        }

        pub fn trunc(self: Self) SparseError!Self {
            ensureNumeric(T);
            const out = try self.clone();
            for (out.values) |*value| value.* = truncSparseValue(T, value.*);
            return out;
        }

        pub fn clipMin(self: Self, min_value: T) SparseError!Self {
            ensureNumeric(T);
            const out = try self.clone();
            for (out.values) |*value| {
                if (value.* < min_value) value.* = min_value;
            }
            return out;
        }

        pub fn maximumScalar(self: Self, scalar: T) SparseError!Self {
            return self.clipMin(scalar);
        }

        pub fn maxScalar(self: Self, scalar: T) SparseError!Self {
            return self.maximumScalar(scalar);
        }

        pub fn clampMin(self: Self, min_value: T) SparseError!Self {
            return self.clipMin(min_value);
        }

        pub fn clipMax(self: Self, max_value: T) SparseError!Self {
            ensureNumeric(T);
            const out = try self.clone();
            for (out.values) |*value| {
                if (value.* > max_value) value.* = max_value;
            }
            return out;
        }

        pub fn minimumScalar(self: Self, scalar: T) SparseError!Self {
            return self.clipMax(scalar);
        }

        pub fn minScalar(self: Self, scalar: T) SparseError!Self {
            return self.minimumScalar(scalar);
        }

        pub fn clampMax(self: Self, max_value: T) SparseError!Self {
            return self.clipMax(max_value);
        }

        pub fn clip(self: Self, min_value: T, max_value: T) SparseError!Self {
            try validateSparseValueRange(T, min_value, max_value);
            const out = try self.clone();
            for (out.values) |*value| {
                if (value.* < min_value) {
                    value.* = min_value;
                } else if (value.* > max_value) {
                    value.* = max_value;
                }
            }
            return out;
        }

        pub fn clamp(self: Self, min_value: T, max_value: T) SparseError!Self {
            return self.clip(min_value, max_value);
        }

        pub fn sign(self: Self) SparseError!Self {
            ensureNumeric(T);
            const out = try self.clone();
            for (out.values) |*value| value.* = signSparseValue(T, value.*);
            return out;
        }

        pub fn signbit(self: Self) SparseError!CsrMatrix(bool) {
            ensureNumeric(T);
            const row_offsets = try self.allocator.dupe(usize, self.row_offsets);
            errdefer self.allocator.free(row_offsets);
            const col_indices = try self.allocator.dupe(usize, self.col_indices);
            errdefer self.allocator.free(col_indices);
            var values = try self.allocator.alloc(bool, self.values.len);
            errdefer self.allocator.free(values);
            for (self.values, 0..) |value, index| {
                values[index] = signbitSparseValue(T, value);
            }
            return .{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = self.cols,
                .row_offsets = row_offsets,
                .col_indices = col_indices,
                .values = values,
            };
        }

        fn finitePredicateMask(self: Self, comptime predicate: SparseFinitePredicate) SparseError!CsrMatrix(bool) {
            return self.mapValues(bool, struct {
                fn f(value: T) bool {
                    return sparseFinitePredicateValue(T, value, predicate);
                }
            }.f);
        }

        pub fn isNan(self: Self) SparseError!CsrMatrix(bool) {
            return self.finitePredicateMask(.nan);
        }

        pub fn isnan(self: Self) SparseError!CsrMatrix(bool) {
            return self.isNan();
        }

        pub fn isInf(self: Self) SparseError!CsrMatrix(bool) {
            return self.finitePredicateMask(.inf);
        }

        pub fn isinf(self: Self) SparseError!CsrMatrix(bool) {
            return self.isInf();
        }

        pub fn isPosInf(self: Self) SparseError!CsrMatrix(bool) {
            return self.finitePredicateMask(.pos_inf);
        }

        pub fn isposinf(self: Self) SparseError!CsrMatrix(bool) {
            return self.isPosInf();
        }

        pub fn isNegInf(self: Self) SparseError!CsrMatrix(bool) {
            return self.finitePredicateMask(.neg_inf);
        }

        pub fn isneginf(self: Self) SparseError!CsrMatrix(bool) {
            return self.isNegInf();
        }

        pub fn isFinite(self: Self) SparseError!CsrMatrix(bool) {
            return self.finitePredicateMask(.finite);
        }

        pub fn isfinite(self: Self) SparseError!CsrMatrix(bool) {
            return self.isFinite();
        }

        pub fn isNormal(self: Self) SparseError!CsrMatrix(bool) {
            return self.finitePredicateMask(.normal);
        }

        pub fn isnormal(self: Self) SparseError!CsrMatrix(bool) {
            return self.isNormal();
        }

        fn compareScalar(self: Self, scalar: T, comptime comparison: SparseScalarComparison) SparseError!CsrMatrix(bool) {
            const row_offsets = try self.allocator.dupe(usize, self.row_offsets);
            errdefer self.allocator.free(row_offsets);
            const col_indices = try self.allocator.dupe(usize, self.col_indices);
            errdefer self.allocator.free(col_indices);
            const values = try sparseCompareScalarValues(T, self.allocator, self.values, scalar, comparison);
            errdefer self.allocator.free(values);
            return .{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = self.cols,
                .row_offsets = row_offsets,
                .col_indices = col_indices,
                .values = values,
            };
        }

        pub fn equalScalar(self: Self, scalar: T) SparseError!CsrMatrix(bool) {
            return self.compareScalar(scalar, .eq);
        }

        pub fn eqScalar(self: Self, scalar: T) SparseError!CsrMatrix(bool) {
            return self.equalScalar(scalar);
        }

        pub fn notEqualScalar(self: Self, scalar: T) SparseError!CsrMatrix(bool) {
            return self.compareScalar(scalar, .ne);
        }

        pub fn neScalar(self: Self, scalar: T) SparseError!CsrMatrix(bool) {
            return self.notEqualScalar(scalar);
        }

        pub fn greaterScalar(self: Self, scalar: T) SparseError!CsrMatrix(bool) {
            return self.compareScalar(scalar, .gt);
        }

        pub fn gtScalar(self: Self, scalar: T) SparseError!CsrMatrix(bool) {
            return self.greaterScalar(scalar);
        }

        pub fn greaterEqualScalar(self: Self, scalar: T) SparseError!CsrMatrix(bool) {
            return self.compareScalar(scalar, .ge);
        }

        pub fn geScalar(self: Self, scalar: T) SparseError!CsrMatrix(bool) {
            return self.greaterEqualScalar(scalar);
        }

        pub fn lessScalar(self: Self, scalar: T) SparseError!CsrMatrix(bool) {
            return self.compareScalar(scalar, .lt);
        }

        pub fn ltScalar(self: Self, scalar: T) SparseError!CsrMatrix(bool) {
            return self.lessScalar(scalar);
        }

        pub fn lessEqualScalar(self: Self, scalar: T) SparseError!CsrMatrix(bool) {
            return self.compareScalar(scalar, .le);
        }

        pub fn leScalar(self: Self, scalar: T) SparseError!CsrMatrix(bool) {
            return self.lessEqualScalar(scalar);
        }

        fn compareSameStructure(self: Self, rhs: Self, comptime comparison: SparseScalarComparison) SparseError!CsrMatrix(bool) {
            if (self.rows != rhs.rows or self.cols != rhs.cols or self.values.len != rhs.values.len) return error.ShapeMismatch;
            if (!self.sameStructure(rhs)) return error.InvalidShape;
            const row_offsets = try self.allocator.dupe(usize, self.row_offsets);
            errdefer self.allocator.free(row_offsets);
            const col_indices = try self.allocator.dupe(usize, self.col_indices);
            errdefer self.allocator.free(col_indices);
            const values = try sparseCompareSameStructureValues(T, self.allocator, self.values, rhs.values, comparison);
            errdefer self.allocator.free(values);
            return .{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = self.cols,
                .row_offsets = row_offsets,
                .col_indices = col_indices,
                .values = values,
            };
        }

        pub fn equalSameStructure(self: Self, rhs: Self) SparseError!CsrMatrix(bool) {
            return self.compareSameStructure(rhs, .eq);
        }

        pub fn eqSameStructure(self: Self, rhs: Self) SparseError!CsrMatrix(bool) {
            return self.equalSameStructure(rhs);
        }

        pub fn notEqualSameStructure(self: Self, rhs: Self) SparseError!CsrMatrix(bool) {
            return self.compareSameStructure(rhs, .ne);
        }

        pub fn neSameStructure(self: Self, rhs: Self) SparseError!CsrMatrix(bool) {
            return self.notEqualSameStructure(rhs);
        }

        pub fn greaterSameStructure(self: Self, rhs: Self) SparseError!CsrMatrix(bool) {
            return self.compareSameStructure(rhs, .gt);
        }

        pub fn gtSameStructure(self: Self, rhs: Self) SparseError!CsrMatrix(bool) {
            return self.greaterSameStructure(rhs);
        }

        pub fn greaterEqualSameStructure(self: Self, rhs: Self) SparseError!CsrMatrix(bool) {
            return self.compareSameStructure(rhs, .ge);
        }

        pub fn geSameStructure(self: Self, rhs: Self) SparseError!CsrMatrix(bool) {
            return self.greaterEqualSameStructure(rhs);
        }

        pub fn lessSameStructure(self: Self, rhs: Self) SparseError!CsrMatrix(bool) {
            return self.compareSameStructure(rhs, .lt);
        }

        pub fn ltSameStructure(self: Self, rhs: Self) SparseError!CsrMatrix(bool) {
            return self.lessSameStructure(rhs);
        }

        pub fn lessEqualSameStructure(self: Self, rhs: Self) SparseError!CsrMatrix(bool) {
            return self.compareSameStructure(rhs, .le);
        }

        pub fn leSameStructure(self: Self, rhs: Self) SparseError!CsrMatrix(bool) {
            return self.lessEqualSameStructure(rhs);
        }

        pub fn nnz(self: Self) usize {
            return self.values.len;
        }

        pub fn countNonzero(self: Self) SparseError!usize {
            return sparseDenseCountNonzero(self);
        }

        pub fn countNonzeroAxis(self: Self, axis_opt: ?isize, keepdims: bool) SparseError!array_mod.Array(usize) {
            return sparseDenseCountNonzeroAxis(self, axis_opt, keepdims);
        }

        pub fn countNonzeroAxes(self: Self, axes: []const isize, keepdims: bool) SparseError!array_mod.Array(usize) {
            return sparseDenseCountNonzeroAxes(self, axes, keepdims);
        }

        pub fn countNonzeroDim(self: Self, dim_opt: ?isize, keepdim: bool) SparseError!array_mod.Array(usize) {
            return self.countNonzeroAxis(dim_opt, keepdim);
        }

        pub fn countNonzeroDims(self: Self, dims: []const isize, keepdim: bool) SparseError!array_mod.Array(usize) {
            return self.countNonzeroAxes(dims, keepdim);
        }

        pub fn flatNonzero(self: Self) SparseError!array_mod.Array(usize) {
            return sparseDenseFlatNonzero(self);
        }

        pub fn nonzero(self: Self) SparseError!array_mod.Array(usize) {
            return sparseDenseNonzero(self);
        }

        pub fn argwhere(self: Self) SparseError!array_mod.Array(usize) {
            return self.nonzero();
        }

        pub fn whereIndices(self: Self) SparseError!array_mod.Array(usize) {
            return self.nonzero();
        }

        pub fn all(self: Self) SparseError!bool {
            return sparseDenseAll(self);
        }

        pub fn any(self: Self) SparseError!bool {
            return sparseDenseAny(self);
        }

        pub fn allAxis(self: Self, axis_opt: ?isize, keepdims: bool) SparseError!array_mod.Array(bool) {
            return sparseDenseAllAxis(self, axis_opt, keepdims);
        }

        pub fn allAxes(self: Self, axes: []const isize, keepdims: bool) SparseError!array_mod.Array(bool) {
            return sparseDenseAllAxes(self, axes, keepdims);
        }

        pub fn allDim(self: Self, dim_opt: ?isize, keepdim: bool) SparseError!array_mod.Array(bool) {
            return self.allAxis(dim_opt, keepdim);
        }

        pub fn allDims(self: Self, dims: []const isize, keepdim: bool) SparseError!array_mod.Array(bool) {
            return self.allAxes(dims, keepdim);
        }

        pub fn anyAxis(self: Self, axis_opt: ?isize, keepdims: bool) SparseError!array_mod.Array(bool) {
            return sparseDenseAnyAxis(self, axis_opt, keepdims);
        }

        pub fn anyAxes(self: Self, axes: []const isize, keepdims: bool) SparseError!array_mod.Array(bool) {
            return sparseDenseAnyAxes(self, axes, keepdims);
        }

        pub fn anyDim(self: Self, dim_opt: ?isize, keepdim: bool) SparseError!array_mod.Array(bool) {
            return self.anyAxis(dim_opt, keepdim);
        }

        pub fn anyDims(self: Self, dims: []const isize, keepdim: bool) SparseError!array_mod.Array(bool) {
            return self.anyAxes(dims, keepdim);
        }

        pub fn logicalNot(self: Self) SparseError!array_mod.Array(bool) {
            return sparseDenseLogicalNot(self);
        }

        pub fn logicalAnd(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return sparseDenseLogicalBinary(self, rhs, .and_);
        }

        pub fn logicalAndScalar(self: Self, scalar: bool) SparseError!array_mod.Array(bool) {
            return sparseDenseLogicalScalar(self, scalar, .and_);
        }

        pub fn logicalOr(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return sparseDenseLogicalBinary(self, rhs, .or_);
        }

        pub fn logicalOrScalar(self: Self, scalar: bool) SparseError!array_mod.Array(bool) {
            return sparseDenseLogicalScalar(self, scalar, .or_);
        }

        pub fn logicalXor(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return sparseDenseLogicalBinary(self, rhs, .xor_);
        }

        pub fn logicalXorScalar(self: Self, scalar: bool) SparseError!array_mod.Array(bool) {
            return sparseDenseLogicalScalar(self, scalar, .xor_);
        }

        pub fn maskedSelect(self: Self, mask: array_mod.Array(bool)) SparseError!array_mod.Array(T) {
            return sparseDenseMaskedSelect(T, self, mask);
        }

        pub fn maskedFill(self: Self, mask: array_mod.Array(bool), value: T) SparseError!array_mod.Array(T) {
            return sparseDenseMaskedFill(T, self, mask, value);
        }

        pub fn maskedPutScalar(self: Self, mask: array_mod.Array(bool), value: T) SparseError!array_mod.Array(T) {
            return self.maskedFill(mask, value);
        }

        pub fn putMaskScalar(self: Self, mask: array_mod.Array(bool), value: T) SparseError!array_mod.Array(T) {
            return self.maskedPutScalar(mask, value);
        }

        pub fn maskedScatter(self: Self, mask: array_mod.Array(bool), src: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            return sparseDenseMaskedScatter(T, self, mask, src);
        }

        pub fn maskedPut(self: Self, mask: array_mod.Array(bool), values: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            return sparseDenseMaskedPut(T, self, mask, values);
        }

        pub fn putMask(self: Self, mask: array_mod.Array(bool), values: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            return self.maskedPut(mask, values);
        }

        pub fn take(self: Self, indices: array_mod.Array(usize), axis_opt: ?isize) SparseError!array_mod.Array(T) {
            return sparseDenseTake(T, self, indices, axis_opt);
        }

        pub fn takeSigned(self: Self, indices: array_mod.Array(isize), axis_opt: ?isize) SparseError!array_mod.Array(T) {
            return sparseDenseTakeSigned(T, self, indices, axis_opt);
        }

        pub fn takeMode(self: Self, indices: array_mod.Array(usize), axis_opt: ?isize, mode: array_mod.IndexMode) SparseError!array_mod.Array(T) {
            return sparseDenseTakeMode(T, self, indices, axis_opt, mode);
        }

        pub fn takeSignedMode(self: Self, indices: array_mod.Array(isize), axis_opt: ?isize, mode: array_mod.IndexMode) SparseError!array_mod.Array(T) {
            return sparseDenseTakeSignedMode(T, self, indices, axis_opt, mode);
        }

        pub fn indexSelect(self: Self, axis_index: isize, indices: array_mod.Array(usize)) SparseError!array_mod.Array(T) {
            return self.take(indices, axis_index);
        }

        pub fn indexSelectSigned(self: Self, axis_index: isize, indices: array_mod.Array(isize)) SparseError!array_mod.Array(T) {
            return self.takeSigned(indices, axis_index);
        }

        pub fn gather(self: Self, axis_index: isize, indices: array_mod.Array(usize)) SparseError!array_mod.Array(T) {
            return sparseDenseGather(T, self, axis_index, indices);
        }

        pub fn gatherSigned(self: Self, axis_index: isize, indices: array_mod.Array(isize)) SparseError!array_mod.Array(T) {
            return sparseDenseGatherSigned(T, self, axis_index, indices);
        }

        pub fn takeAlongAxis(self: Self, indices: array_mod.Array(usize), axis_index: isize) SparseError!array_mod.Array(T) {
            return self.gather(axis_index, indices);
        }

        pub fn takeAlongAxisSigned(self: Self, indices: array_mod.Array(isize), axis_index: isize) SparseError!array_mod.Array(T) {
            return self.gatherSigned(axis_index, indices);
        }

        pub fn scatter(self: Self, axis_index: isize, indices: array_mod.Array(usize), src: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            return sparseDenseScatter(T, self, axis_index, indices, src);
        }

        pub fn scatterScalar(self: Self, axis_index: isize, indices: array_mod.Array(usize), value: T) SparseError!array_mod.Array(T) {
            return sparseDenseScatterScalar(T, self, axis_index, indices, value);
        }

        pub fn putAlongAxis(self: Self, indices: array_mod.Array(usize), src: array_mod.Array(T), axis_index: isize) SparseError!array_mod.Array(T) {
            return self.scatter(axis_index, indices, src);
        }

        pub fn select(self: Self, axis_index: isize, index: usize) SparseError!array_mod.Array(T) {
            return sparseDenseSelect(T, self, axis_index, index);
        }

        pub fn selectSigned(self: Self, axis_index: isize, index: isize) SparseError!array_mod.Array(T) {
            return sparseDenseSelectSigned(T, self, axis_index, index);
        }

        pub fn narrow(self: Self, axis_index: isize, start: usize, length: usize) SparseError!array_mod.Array(T) {
            return sparseDenseNarrow(T, self, axis_index, start, length);
        }

        pub fn narrowSigned(self: Self, axis_index: isize, start: isize, length: usize) SparseError!array_mod.Array(T) {
            return sparseDenseNarrowSigned(T, self, axis_index, start, length);
        }

        pub fn scatterReduce(self: Self, axis_index: isize, indices: array_mod.Array(usize), src: array_mod.Array(T), reduction: array_mod.ScatterReduce) SparseError!array_mod.Array(T) {
            return sparseDenseScatterReduce(T, self, axis_index, indices, src, reduction);
        }

        pub fn scatterAdd(self: Self, axis_index: isize, indices: array_mod.Array(usize), src: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            return self.scatterReduce(axis_index, indices, src, .sum);
        }

        pub fn scatterReduceScalar(self: Self, axis_index: isize, indices: array_mod.Array(usize), value: T, reduction: array_mod.ScatterReduce) SparseError!array_mod.Array(T) {
            return sparseDenseScatterReduceScalar(T, self, axis_index, indices, value, reduction);
        }

        pub fn putFlat(self: Self, indices: array_mod.Array(usize), values: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            return sparseDensePutFlat(T, self, indices, values);
        }

        pub fn putFlatMode(self: Self, indices: array_mod.Array(usize), values: array_mod.Array(T), mode: array_mod.IndexMode) SparseError!array_mod.Array(T) {
            return sparseDensePutFlatMode(T, self, indices, values, mode);
        }

        pub fn putFlatScalar(self: Self, indices: array_mod.Array(usize), value: T) SparseError!array_mod.Array(T) {
            return sparseDensePutFlatScalar(T, self, indices, value);
        }

        pub fn putFlatScalarMode(self: Self, indices: array_mod.Array(usize), value: T, mode: array_mod.IndexMode) SparseError!array_mod.Array(T) {
            return sparseDensePutFlatScalarMode(T, self, indices, value, mode);
        }

        pub fn putFlatSigned(self: Self, indices: array_mod.Array(isize), values: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            return sparseDensePutFlatSigned(T, self, indices, values);
        }

        pub fn putFlatScalarSigned(self: Self, indices: array_mod.Array(isize), value: T) SparseError!array_mod.Array(T) {
            return sparseDensePutFlatScalarSigned(T, self, indices, value);
        }

        pub fn ravelCoords(self: Self, coords: array_mod.Array(usize)) SparseError!array_mod.Array(usize) {
            return sparseDenseRavelCoords(self, coords);
        }

        pub fn unravelFlat(self: Self, indices: array_mod.Array(usize)) SparseError!array_mod.Array(usize) {
            return sparseDenseUnravelFlat(self, indices);
        }

        pub fn takeCoords(self: Self, coords: array_mod.Array(usize)) SparseError!array_mod.Array(T) {
            return sparseDenseTakeCoords(T, self, coords);
        }

        pub fn putCoords(self: Self, coords: array_mod.Array(usize), values: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            return sparseDensePutCoords(T, self, coords, values);
        }

        pub fn putCoordsScalar(self: Self, coords: array_mod.Array(usize), value: T) SparseError!array_mod.Array(T) {
            return sparseDensePutCoordsScalar(T, self, coords, value);
        }

        pub fn reshape(self: Self, dims: []const usize) SparseError!array_mod.Array(T) {
            return sparseDenseReshape(T, self, dims);
        }

        pub fn reshapeInfer(self: Self, dims: []const isize) SparseError!array_mod.Array(T) {
            return sparseDenseReshapeInfer(T, self, dims);
        }

        pub fn reshapeAs(self: Self, other: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            return self.reshape(other.shape);
        }

        pub fn view(self: Self, dims: []const usize) SparseError!array_mod.Array(T) {
            return self.reshape(dims);
        }

        pub fn viewInfer(self: Self, dims: []const isize) SparseError!array_mod.Array(T) {
            return self.reshapeInfer(dims);
        }

        pub fn viewAs(self: Self, other: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            return self.view(other.shape);
        }

        pub fn flatten(self: Self) SparseError!array_mod.Array(T) {
            return sparseDenseFlatten(T, self);
        }

        pub fn flattenAxes(self: Self, start_axis: isize, end_axis: isize) SparseError!array_mod.Array(T) {
            return sparseDenseFlattenAxes(T, self, start_axis, end_axis);
        }

        pub fn flattenRange(self: Self, start_axis: isize, end_axis: isize) SparseError!array_mod.Array(T) {
            return self.flattenAxes(start_axis, end_axis);
        }

        pub fn flattenFrom(self: Self, start_axis: isize) SparseError!array_mod.Array(T) {
            return self.flattenAxes(start_axis, -1);
        }

        pub fn ravel(self: Self) SparseError!array_mod.Array(T) {
            return self.flatten();
        }

        pub fn atLeast1d(self: Self) SparseError!array_mod.Array(T) {
            return sparseDenseAtLeast1d(T, self);
        }

        pub fn atLeast2d(self: Self) SparseError!array_mod.Array(T) {
            return sparseDenseAtLeast2d(T, self);
        }

        pub fn atLeast3d(self: Self) SparseError!array_mod.Array(T) {
            return sparseDenseAtLeast3d(T, self);
        }

        pub fn unflatten(self: Self, axis_index: isize, dims: []const usize) SparseError!array_mod.Array(T) {
            return sparseDenseUnflatten(T, self, axis_index, dims);
        }

        pub fn squeeze(self: Self, axis_opt: ?isize) SparseError!array_mod.Array(T) {
            return sparseDenseSqueeze(T, self, axis_opt);
        }

        pub fn squeezeDim(self: Self, axis_index: isize) SparseError!array_mod.Array(T) {
            return self.squeeze(axis_index);
        }

        pub fn unsqueeze(self: Self, axis_index: isize) SparseError!array_mod.Array(T) {
            return sparseDenseUnsqueeze(T, self, axis_index);
        }

        pub fn unsqueezeDim(self: Self, axis_index: isize) SparseError!array_mod.Array(T) {
            return self.unsqueeze(axis_index);
        }

        pub fn squeezeAxes(self: Self, axes: []const isize) SparseError!array_mod.Array(T) {
            return sparseDenseSqueezeAxes(T, self, axes);
        }

        pub fn unsqueezeAxes(self: Self, axes: []const isize) SparseError!array_mod.Array(T) {
            return sparseDenseUnsqueezeAxes(T, self, axes);
        }

        pub fn expandDims(self: Self, axes: []const isize) SparseError!array_mod.Array(T) {
            return self.unsqueezeAxes(axes);
        }

        pub fn broadcastTo(self: Self, dims: []const usize) SparseError!array_mod.Array(T) {
            return sparseDenseBroadcastTo(T, self, dims);
        }

        pub fn expand(self: Self, dims: []const usize) SparseError!array_mod.Array(T) {
            return self.broadcastTo(dims);
        }

        pub fn expandAs(self: Self, other: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            return self.expand(other.shape);
        }

        pub fn broadcastAs(self: Self, other: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            return self.expandAs(other);
        }

        pub fn permute(self: Self, axes: []const usize) SparseError!array_mod.Array(T) {
            return sparseDensePermute(T, self, axes);
        }

        pub fn swapaxes(self: Self, dim0: isize, dim1: isize) SparseError!array_mod.Array(T) {
            return sparseDenseSwapAxes(T, self, dim0, dim1);
        }

        pub fn swapDims(self: Self, dim0: isize, dim1: isize) SparseError!array_mod.Array(T) {
            return self.swapaxes(dim0, dim1);
        }

        pub fn movedim(self: Self, source: isize, destination: isize) SparseError!array_mod.Array(T) {
            return sparseDenseMoveDim(T, self, source, destination);
        }

        pub fn moveaxis(self: Self, source: isize, destination: isize) SparseError!array_mod.Array(T) {
            return self.movedim(source, destination);
        }

        pub fn flip(self: Self, axis_index: isize) SparseError!array_mod.Array(T) {
            return sparseDenseFlip(T, self, axis_index);
        }

        pub fn flipud(self: Self) SparseError!array_mod.Array(T) {
            return self.flip(0);
        }

        pub fn fliplr(self: Self) SparseError!array_mod.Array(T) {
            return self.flip(1);
        }

        pub fn flipAxes(self: Self, axes: []const isize) SparseError!array_mod.Array(T) {
            return sparseDenseFlipAxes(T, self, axes);
        }

        pub fn roll(self: Self, shift: isize, axis_index: isize) SparseError!array_mod.Array(T) {
            return sparseDenseRoll(T, self, shift, axis_index);
        }

        pub fn rollFlat(self: Self, shift: isize) SparseError!array_mod.Array(T) {
            return sparseDenseRollFlat(T, self, shift);
        }

        pub fn rollAxes(self: Self, shifts: []const isize, axes: []const isize) SparseError!array_mod.Array(T) {
            return sparseDenseRollAxes(T, self, shifts, axes);
        }

        pub fn rot90(self: Self, k: isize, axes: [2]isize) SparseError!array_mod.Array(T) {
            return sparseDenseRot90(T, self, k, axes);
        }

        pub fn padConstant(self: Self, before: []const usize, after: []const usize, value: T) SparseError!array_mod.Array(T) {
            return sparseDensePadConstant(T, self, before, after, value);
        }

        pub fn padEdge(self: Self, before: []const usize, after: []const usize) SparseError!array_mod.Array(T) {
            return sparseDensePadEdge(T, self, before, after);
        }

        pub fn padReflect(self: Self, before: []const usize, after: []const usize) SparseError!array_mod.Array(T) {
            return sparseDensePadReflect(T, self, before, after);
        }

        pub fn padWrap(self: Self, before: []const usize, after: []const usize) SparseError!array_mod.Array(T) {
            return sparseDensePadWrap(T, self, before, after);
        }

        pub fn padSymmetric(self: Self, before: []const usize, after: []const usize) SparseError!array_mod.Array(T) {
            return sparseDensePadSymmetric(T, self, before, after);
        }

        pub fn split(self: Self, split_size: usize, axis_index: isize) SparseError!array_mod.Array(T).SplitResult {
            return sparseDenseSplit(T, self, split_size, axis_index);
        }

        pub fn splitWithSizes(self: Self, sizes: []const usize, axis_index: isize) SparseError!array_mod.Array(T).SplitResult {
            return sparseDenseSplitWithSizes(T, self, sizes, axis_index);
        }

        pub fn splitAtIndices(self: Self, indices: []const usize, axis_index: isize) SparseError!array_mod.Array(T).SplitResult {
            return sparseDenseSplitAtIndices(T, self, indices, axis_index);
        }

        pub fn chunk(self: Self, chunks: usize, axis_index: isize) SparseError!array_mod.Array(T).SplitResult {
            return sparseDenseChunk(T, self, chunks, axis_index);
        }

        pub fn unbind(self: Self, axis_index: isize) SparseError!array_mod.Array(T).SplitResult {
            return sparseDenseUnbind(T, self, axis_index);
        }

        pub fn repeat(self: Self, repeats: usize, axis_index: isize) SparseError!array_mod.Array(T) {
            return sparseDenseRepeat(T, self, repeats, axis_index);
        }

        pub fn repeatInterleave(self: Self, repeats: array_mod.Array(usize), axis_opt: ?isize) SparseError!array_mod.Array(T) {
            return sparseDenseRepeatInterleave(T, self, repeats, axis_opt);
        }

        pub fn repeatInterleaveScalar(self: Self, repeat_count: usize, axis_opt: ?isize) SparseError!array_mod.Array(T) {
            return sparseDenseRepeatInterleaveScalar(T, self, repeat_count, axis_opt);
        }

        pub fn tile(self: Self, repeats: []const usize) SparseError!array_mod.Array(T) {
            return sparseDenseTile(T, self, repeats);
        }

        pub fn indexPut(self: Self, indices: array_mod.Array(usize), values: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            return self.putFlat(indices, values);
        }

        pub fn indexPutScalar(self: Self, indices: array_mod.Array(usize), value: T) SparseError!array_mod.Array(T) {
            return self.putFlatScalar(indices, value);
        }

        pub fn compress(self: Self, condition: array_mod.Array(bool), axis_opt: ?isize) SparseError!array_mod.Array(T) {
            return sparseDenseCompress(T, self, condition, axis_opt);
        }

        pub fn where(self: Self, mask: array_mod.Array(bool), other: Self) SparseError!array_mod.Array(T) {
            return sparseDenseWhere(T, self, mask, other);
        }

        pub fn whereArray(self: Self, mask: array_mod.Array(bool), other: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            return sparseDenseWhereArray(T, self, mask, other);
        }

        pub fn whereScalar(self: Self, mask: array_mod.Array(bool), other_value: T) SparseError!array_mod.Array(T) {
            return sparseDenseWhereScalar(T, self, mask, other_value);
        }

        pub fn copyWhere(self: Self, mask: array_mod.Array(bool), src: Self) SparseError!array_mod.Array(T) {
            return sparseDenseWhere(T, src, mask, self);
        }

        pub fn asVeyraView(self: Self) SparseError!veyra.CsrView(T) {
            return veyra.CsrView(T).fromSlices(self.rows, self.cols, self.row_offsets, self.col_indices, self.values) catch return error.BackendFailure;
        }

        pub fn sameStructure(self: Self, rhs: Self) bool {
            return sparseSameStructure(self.rows, self.cols, self.row_offsets, self.col_indices, rhs.rows, rhs.cols, rhs.row_offsets, rhs.col_indices);
        }

        pub fn dotSameStructure(self: Self, rhs: Self) SparseError!T {
            if (self.rows != rhs.rows or self.cols != rhs.cols or self.values.len != rhs.values.len) return error.ShapeMismatch;
            if (!self.sameStructure(rhs)) return error.InvalidShape;
            return sparseDotSameStructure(T, self.values, rhs.values);
        }

        pub fn sameStructureDiffSummary(self: Self, rhs: Self) SparseError!SparseDiffSummary {
            if (self.rows != rhs.rows or self.cols != rhs.cols or self.values.len != rhs.values.len) return error.ShapeMismatch;
            if (!self.sameStructure(rhs)) return error.InvalidShape;
            return sparseSameStructureDiffSummary(T, self.values, rhs.values);
        }

        pub fn diffSummary(self: Self, rhs: Self) SparseError!SparseDiffSummary {
            if (self.rows != rhs.rows or self.cols != rhs.cols) return error.ShapeMismatch;
            var lhs_dense = try self.toDense();
            defer lhs_dense.deinit();
            var rhs_dense = try rhs.toDense();
            defer rhs_dense.deinit();
            return denseDiffSummary(T, lhs_dense.data, rhs_dense.data);
        }

        pub fn diffSummaryMeetsBounds(
            self: Self,
            rhs: Self,
            max_absolute_diff: f64,
            max_relative_diff: f64,
            max_squared_distance: f64,
            max_frobenius_distance: f64,
            max_relative_frobenius_distance: f64,
        ) SparseError!bool {
            if (self.rows != rhs.rows or self.cols != rhs.cols) return error.ShapeMismatch;
            var lhs_dense = try self.toDense();
            defer lhs_dense.deinit();
            var rhs_dense = try rhs.toDense();
            defer rhs_dense.deinit();
            return denseDiffSummaryMeetsBounds(
                T,
                lhs_dense.data,
                rhs_dense.data,
                max_absolute_diff,
                max_relative_diff,
                max_squared_distance,
                max_frobenius_distance,
                max_relative_frobenius_distance,
            );
        }

        pub fn maxAbsDiff(self: Self, rhs: Self) SparseError!f64 {
            return (try self.diffSummary(rhs)).max_abs_diff;
        }

        pub fn maxRelDiff(self: Self, rhs: Self) SparseError!f64 {
            return (try self.diffSummary(rhs)).max_rel_diff;
        }

        pub fn squaredDistance(self: Self, rhs: Self) SparseError!f64 {
            return (try self.diffSummary(rhs)).squared_distance;
        }

        pub fn frobeniusDistance(self: Self, rhs: Self) SparseError!f64 {
            return (try self.diffSummary(rhs)).frobeniusDistance();
        }

        pub fn relativeFrobeniusDistance(self: Self, rhs: Self) SparseError!f64 {
            return (try self.diffSummary(rhs)).relativeFrobeniusDistance();
        }

        pub fn maxAbsDiffMeetsBound(self: Self, rhs: Self, max_absolute_diff: f64) SparseError!bool {
            return (try self.diffSummary(rhs)).maxAbsDiffMeetsBound(max_absolute_diff);
        }

        pub fn maxRelDiffMeetsBound(self: Self, rhs: Self, max_relative_diff: f64) SparseError!bool {
            return (try self.diffSummary(rhs)).maxRelDiffMeetsBound(max_relative_diff);
        }

        pub fn squaredDistanceMeetsBound(self: Self, rhs: Self, max_squared_distance: f64) SparseError!bool {
            return (try self.diffSummary(rhs)).squaredDistanceMeetsBound(max_squared_distance);
        }

        pub fn frobeniusDistanceMeetsBound(self: Self, rhs: Self, max_distance: f64) SparseError!bool {
            return (try self.diffSummary(rhs)).frobeniusDistanceMeetsBound(max_distance);
        }

        pub fn relativeFrobeniusDistanceMeetsBound(self: Self, rhs: Self, max_relative_distance: f64) SparseError!bool {
            return (try self.diffSummary(rhs)).relativeFrobeniusDistanceMeetsBound(max_relative_distance);
        }

        fn compareDenseMaterialized(self: Self, rhs: Self, comptime comparison: SparseScalarComparison) SparseError!array_mod.Array(bool) {
            if (self.rows != rhs.rows or self.cols != rhs.cols) return error.ShapeMismatch;
            var lhs_dense = try self.toDense();
            defer lhs_dense.deinit();
            var rhs_dense = try rhs.toDense();
            defer rhs_dense.deinit();
            return sparseCompareDenseArrays(T, lhs_dense, rhs_dense, comparison);
        }

        fn compareDenseArray(self: Self, rhs: array_mod.Array(T), comptime comparison: SparseScalarComparison) SparseError!array_mod.Array(bool) {
            try validateDenseMatrixShape(self.rows, self.cols, rhs.shape);
            var lhs_dense = try self.toDense();
            defer lhs_dense.deinit();
            return sparseCompareDenseArrays(T, lhs_dense, rhs, comparison);
        }

        pub fn eq(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return self.compareDenseMaterialized(rhs, .eq);
        }

        pub fn equal(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return self.eq(rhs);
        }

        pub fn ne(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return self.compareDenseMaterialized(rhs, .ne);
        }

        pub fn notEqual(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return self.ne(rhs);
        }

        pub fn gt(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return self.compareDenseMaterialized(rhs, .gt);
        }

        pub fn greater(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return self.gt(rhs);
        }

        pub fn ge(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return self.compareDenseMaterialized(rhs, .ge);
        }

        pub fn greaterEqual(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return self.ge(rhs);
        }

        pub fn lt(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return self.compareDenseMaterialized(rhs, .lt);
        }

        pub fn less(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return self.lt(rhs);
        }

        pub fn le(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return self.compareDenseMaterialized(rhs, .le);
        }

        pub fn lessEqual(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return self.le(rhs);
        }

        pub fn eqDense(self: Self, rhs: array_mod.Array(T)) SparseError!array_mod.Array(bool) {
            return self.compareDenseArray(rhs, .eq);
        }

        pub fn equalDense(self: Self, rhs: array_mod.Array(T)) SparseError!array_mod.Array(bool) {
            return self.eqDense(rhs);
        }

        pub fn neDense(self: Self, rhs: array_mod.Array(T)) SparseError!array_mod.Array(bool) {
            return self.compareDenseArray(rhs, .ne);
        }

        pub fn notEqualDense(self: Self, rhs: array_mod.Array(T)) SparseError!array_mod.Array(bool) {
            return self.neDense(rhs);
        }

        pub fn gtDense(self: Self, rhs: array_mod.Array(T)) SparseError!array_mod.Array(bool) {
            return self.compareDenseArray(rhs, .gt);
        }

        pub fn greaterDense(self: Self, rhs: array_mod.Array(T)) SparseError!array_mod.Array(bool) {
            return self.gtDense(rhs);
        }

        pub fn geDense(self: Self, rhs: array_mod.Array(T)) SparseError!array_mod.Array(bool) {
            return self.compareDenseArray(rhs, .ge);
        }

        pub fn greaterEqualDense(self: Self, rhs: array_mod.Array(T)) SparseError!array_mod.Array(bool) {
            return self.geDense(rhs);
        }

        pub fn ltDense(self: Self, rhs: array_mod.Array(T)) SparseError!array_mod.Array(bool) {
            return self.compareDenseArray(rhs, .lt);
        }

        pub fn lessDense(self: Self, rhs: array_mod.Array(T)) SparseError!array_mod.Array(bool) {
            return self.ltDense(rhs);
        }

        pub fn leDense(self: Self, rhs: array_mod.Array(T)) SparseError!array_mod.Array(bool) {
            return self.compareDenseArray(rhs, .le);
        }

        pub fn lessEqualDense(self: Self, rhs: array_mod.Array(T)) SparseError!array_mod.Array(bool) {
            return self.leDense(rhs);
        }

        pub fn diffSummaryDense(self: Self, rhs: array_mod.Array(T)) SparseError!SparseDiffSummary {
            if (rhs.shape.len != 2) return error.NonMatrixArray;
            if (self.rows != rhs.shape[0] or self.cols != rhs.shape[1]) return error.ShapeMismatch;
            var lhs_dense = try self.toDense();
            defer lhs_dense.deinit();
            return denseDiffSummary(T, lhs_dense.data, rhs.data);
        }

        pub fn diffSummaryDenseMeetsBounds(
            self: Self,
            rhs: array_mod.Array(T),
            max_absolute_diff: f64,
            max_relative_diff: f64,
            max_squared_distance: f64,
            max_frobenius_distance: f64,
            max_relative_frobenius_distance: f64,
        ) SparseError!bool {
            if (rhs.shape.len != 2) return error.NonMatrixArray;
            if (self.rows != rhs.shape[0] or self.cols != rhs.shape[1]) return error.ShapeMismatch;
            var lhs_dense = try self.toDense();
            defer lhs_dense.deinit();
            return denseDiffSummaryMeetsBounds(
                T,
                lhs_dense.data,
                rhs.data,
                max_absolute_diff,
                max_relative_diff,
                max_squared_distance,
                max_frobenius_distance,
                max_relative_frobenius_distance,
            );
        }

        pub fn maxAbsDiffDense(self: Self, rhs: array_mod.Array(T)) SparseError!f64 {
            return (try self.diffSummaryDense(rhs)).max_abs_diff;
        }

        pub fn maxRelDiffDense(self: Self, rhs: array_mod.Array(T)) SparseError!f64 {
            return (try self.diffSummaryDense(rhs)).max_rel_diff;
        }

        pub fn squaredDistanceDense(self: Self, rhs: array_mod.Array(T)) SparseError!f64 {
            return (try self.diffSummaryDense(rhs)).squared_distance;
        }

        pub fn frobeniusDistanceDense(self: Self, rhs: array_mod.Array(T)) SparseError!f64 {
            return (try self.diffSummaryDense(rhs)).frobeniusDistance();
        }

        pub fn relativeFrobeniusDistanceDense(self: Self, rhs: array_mod.Array(T)) SparseError!f64 {
            return (try self.diffSummaryDense(rhs)).relativeFrobeniusDistance();
        }

        pub fn maxAbsDiffDenseMeetsBound(self: Self, rhs: array_mod.Array(T), max_absolute_diff: f64) SparseError!bool {
            return (try self.diffSummaryDense(rhs)).maxAbsDiffMeetsBound(max_absolute_diff);
        }

        pub fn maxRelDiffDenseMeetsBound(self: Self, rhs: array_mod.Array(T), max_relative_diff: f64) SparseError!bool {
            return (try self.diffSummaryDense(rhs)).maxRelDiffMeetsBound(max_relative_diff);
        }

        pub fn squaredDistanceDenseMeetsBound(self: Self, rhs: array_mod.Array(T), max_squared_distance: f64) SparseError!bool {
            return (try self.diffSummaryDense(rhs)).squaredDistanceMeetsBound(max_squared_distance);
        }

        pub fn frobeniusDistanceDenseMeetsBound(self: Self, rhs: array_mod.Array(T), max_distance: f64) SparseError!bool {
            return (try self.diffSummaryDense(rhs)).frobeniusDistanceMeetsBound(max_distance);
        }

        pub fn relativeFrobeniusDistanceDenseMeetsBound(self: Self, rhs: array_mod.Array(T), max_relative_distance: f64) SparseError!bool {
            return (try self.diffSummaryDense(rhs)).relativeFrobeniusDistanceMeetsBound(max_relative_distance);
        }

        pub fn iscloseDense(self: Self, rhs: array_mod.Array(T), rtol: T, atol: T) SparseError!array_mod.Array(bool) {
            return self.iscloseDenseEqualNan(rhs, rtol, atol, false);
        }

        pub fn isCloseDense(self: Self, rhs: array_mod.Array(T), rtol: T, atol: T) SparseError!array_mod.Array(bool) {
            return self.iscloseDense(rhs, rtol, atol);
        }

        pub fn iscloseDenseEqualNan(self: Self, rhs: array_mod.Array(T), rtol: T, atol: T, equal_nan: bool) SparseError!array_mod.Array(bool) {
            if (rhs.shape.len != 2) return error.NonMatrixArray;
            if (self.rows != rhs.shape[0] or self.cols != rhs.shape[1]) return error.ShapeMismatch;
            var lhs_dense = try self.toDense();
            defer lhs_dense.deinit();
            return lhs_dense.iscloseEqualNan(rhs, rtol, atol, equal_nan);
        }

        pub fn isCloseDenseEqualNan(self: Self, rhs: array_mod.Array(T), rtol: T, atol: T, equal_nan: bool) SparseError!array_mod.Array(bool) {
            return self.iscloseDenseEqualNan(rhs, rtol, atol, equal_nan);
        }

        pub fn allcloseDense(self: Self, rhs: array_mod.Array(T), rtol: T, atol: T) SparseError!bool {
            return self.allcloseDenseEqualNan(rhs, rtol, atol, false);
        }

        pub fn allCloseDense(self: Self, rhs: array_mod.Array(T), rtol: T, atol: T) SparseError!bool {
            return self.allcloseDense(rhs, rtol, atol);
        }

        pub fn allcloseDenseEqualNan(self: Self, rhs: array_mod.Array(T), rtol: T, atol: T, equal_nan: bool) SparseError!bool {
            if (rhs.shape.len != 2) return error.NonMatrixArray;
            if (self.rows != rhs.shape[0] or self.cols != rhs.shape[1]) return error.ShapeMismatch;
            var lhs_dense = try self.toDense();
            defer lhs_dense.deinit();
            return lhs_dense.allcloseEqualNan(rhs, rtol, atol, equal_nan);
        }

        pub fn allCloseDenseEqualNan(self: Self, rhs: array_mod.Array(T), rtol: T, atol: T, equal_nan: bool) SparseError!bool {
            return self.allcloseDenseEqualNan(rhs, rtol, atol, equal_nan);
        }

        pub fn iscloseScalar(self: Self, scalar: T, rtol: T, atol: T) SparseError!array_mod.Array(bool) {
            return self.iscloseScalarEqualNan(scalar, rtol, atol, false);
        }

        pub fn isCloseScalar(self: Self, scalar: T, rtol: T, atol: T) SparseError!array_mod.Array(bool) {
            return self.iscloseScalar(scalar, rtol, atol);
        }

        pub fn iscloseScalarEqualNan(self: Self, scalar: T, rtol: T, atol: T, equal_nan: bool) SparseError!array_mod.Array(bool) {
            var dense = try self.toDense();
            defer dense.deinit();
            return dense.iscloseScalarEqualNan(scalar, rtol, atol, equal_nan);
        }

        pub fn isCloseScalarEqualNan(self: Self, scalar: T, rtol: T, atol: T, equal_nan: bool) SparseError!array_mod.Array(bool) {
            return self.iscloseScalarEqualNan(scalar, rtol, atol, equal_nan);
        }

        pub fn allcloseScalar(self: Self, scalar: T, rtol: T, atol: T) SparseError!bool {
            return self.allcloseScalarEqualNan(scalar, rtol, atol, false);
        }

        pub fn allCloseScalar(self: Self, scalar: T, rtol: T, atol: T) SparseError!bool {
            return self.allcloseScalar(scalar, rtol, atol);
        }

        pub fn allcloseScalarEqualNan(self: Self, scalar: T, rtol: T, atol: T, equal_nan: bool) SparseError!bool {
            var dense = try self.toDense();
            defer dense.deinit();
            return dense.allcloseScalarEqualNan(scalar, rtol, atol, equal_nan);
        }

        pub fn allCloseScalarEqualNan(self: Self, scalar: T, rtol: T, atol: T, equal_nan: bool) SparseError!bool {
            return self.allcloseScalarEqualNan(scalar, rtol, atol, equal_nan);
        }

        pub fn isclose(self: Self, rhs: Self, rtol: T, atol: T) SparseError!array_mod.Array(bool) {
            return self.iscloseEqualNan(rhs, rtol, atol, false);
        }

        pub fn isClose(self: Self, rhs: Self, rtol: T, atol: T) SparseError!array_mod.Array(bool) {
            return self.isclose(rhs, rtol, atol);
        }

        pub fn iscloseEqualNan(self: Self, rhs: Self, rtol: T, atol: T, equal_nan: bool) SparseError!array_mod.Array(bool) {
            if (self.rows != rhs.rows or self.cols != rhs.cols) return error.ShapeMismatch;
            var lhs_dense = try self.toDense();
            defer lhs_dense.deinit();
            var rhs_dense = try rhs.toDense();
            defer rhs_dense.deinit();
            return lhs_dense.iscloseEqualNan(rhs_dense, rtol, atol, equal_nan);
        }

        pub fn isCloseEqualNan(self: Self, rhs: Self, rtol: T, atol: T, equal_nan: bool) SparseError!array_mod.Array(bool) {
            return self.iscloseEqualNan(rhs, rtol, atol, equal_nan);
        }

        pub fn allclose(self: Self, rhs: Self, rtol: T, atol: T) SparseError!bool {
            return self.allcloseEqualNan(rhs, rtol, atol, false);
        }

        pub fn allClose(self: Self, rhs: Self, rtol: T, atol: T) SparseError!bool {
            return self.allclose(rhs, rtol, atol);
        }

        pub fn allcloseEqualNan(self: Self, rhs: Self, rtol: T, atol: T, equal_nan: bool) SparseError!bool {
            if (self.rows != rhs.rows or self.cols != rhs.cols) return error.ShapeMismatch;
            var lhs_dense = try self.toDense();
            defer lhs_dense.deinit();
            var rhs_dense = try rhs.toDense();
            defer rhs_dense.deinit();
            return lhs_dense.allcloseEqualNan(rhs_dense, rtol, atol, equal_nan);
        }

        pub fn allCloseEqualNan(self: Self, rhs: Self, rtol: T, atol: T, equal_nan: bool) SparseError!bool {
            return self.allcloseEqualNan(rhs, rtol, atol, equal_nan);
        }

        pub fn maxAbsDiffSameStructure(self: Self, rhs: Self) SparseError!T {
            if (self.rows != rhs.rows or self.cols != rhs.cols or self.values.len != rhs.values.len) return error.ShapeMismatch;
            if (!self.sameStructure(rhs)) return error.InvalidShape;
            return sparseMaxAbsDiffSameStructure(T, self.values, rhs.values);
        }

        pub fn maxAbsDiffSameStructureMeetsBound(self: Self, rhs: Self, max_absolute_diff: T) SparseError!bool {
            if (self.rows != rhs.rows or self.cols != rhs.cols or self.values.len != rhs.values.len) return error.ShapeMismatch;
            if (!self.sameStructure(rhs)) return error.InvalidShape;
            return sparseMaxAbsDiffSameStructureMeetsBound(T, self.values, rhs.values, max_absolute_diff);
        }

        pub fn maxRelDiffSameStructure(self: Self, rhs: Self) SparseError!T {
            if (self.rows != rhs.rows or self.cols != rhs.cols or self.values.len != rhs.values.len) return error.ShapeMismatch;
            if (!self.sameStructure(rhs)) return error.InvalidShape;
            return sparseMaxRelDiffSameStructure(T, self.values, rhs.values);
        }

        pub fn maxRelDiffSameStructureMeetsBound(self: Self, rhs: Self, max_relative_diff: T) SparseError!bool {
            if (self.rows != rhs.rows or self.cols != rhs.cols or self.values.len != rhs.values.len) return error.ShapeMismatch;
            if (!self.sameStructure(rhs)) return error.InvalidShape;
            return sparseMaxRelDiffSameStructureMeetsBound(T, self.values, rhs.values, max_relative_diff);
        }

        pub fn squaredDistanceSameStructure(self: Self, rhs: Self) SparseError!T {
            if (self.rows != rhs.rows or self.cols != rhs.cols or self.values.len != rhs.values.len) return error.ShapeMismatch;
            if (!self.sameStructure(rhs)) return error.InvalidShape;
            return sparseSquaredDistanceSameStructure(T, self.values, rhs.values);
        }

        pub fn squaredDistanceSameStructureMeetsBound(self: Self, rhs: Self, max_squared_distance: T) SparseError!bool {
            if (self.rows != rhs.rows or self.cols != rhs.cols or self.values.len != rhs.values.len) return error.ShapeMismatch;
            if (!self.sameStructure(rhs)) return error.InvalidShape;
            return sparseSquaredDistanceSameStructureMeetsBound(T, self.values, rhs.values, max_squared_distance);
        }

        pub fn frobeniusDistanceSameStructure(self: Self, rhs: Self) SparseError!T {
            if (self.rows != rhs.rows or self.cols != rhs.cols or self.values.len != rhs.values.len) return error.ShapeMismatch;
            if (!self.sameStructure(rhs)) return error.InvalidShape;
            return sparseFrobeniusDistanceSameStructure(T, self.values, rhs.values);
        }

        pub fn frobeniusDistanceSameStructureMeetsBound(self: Self, rhs: Self, max_distance: T) SparseError!bool {
            if (self.rows != rhs.rows or self.cols != rhs.cols or self.values.len != rhs.values.len) return error.ShapeMismatch;
            if (!self.sameStructure(rhs)) return error.InvalidShape;
            return sparseFrobeniusDistanceSameStructureMeetsBound(T, self.values, rhs.values, max_distance);
        }

        pub fn relativeFrobeniusDistanceSameStructure(self: Self, rhs: Self) SparseError!T {
            if (self.rows != rhs.rows or self.cols != rhs.cols or self.values.len != rhs.values.len) return error.ShapeMismatch;
            if (!self.sameStructure(rhs)) return error.InvalidShape;
            return sparseRelativeFrobeniusDistanceSameStructure(T, self.values, rhs.values);
        }

        pub fn relativeFrobeniusDistanceSameStructureMeetsBound(self: Self, rhs: Self, max_relative_distance: T) SparseError!bool {
            if (self.rows != rhs.rows or self.cols != rhs.cols or self.values.len != rhs.values.len) return error.ShapeMismatch;
            if (!self.sameStructure(rhs)) return error.InvalidShape;
            return sparseRelativeFrobeniusDistanceSameStructureMeetsBound(T, self.values, rhs.values, max_relative_distance);
        }

        pub fn toDense(self: Self) SparseError!array_mod.Array(T) {
            var out = try array_mod.Array(T).zeros(self.allocator, &.{ self.rows, self.cols });
            errdefer out.deinit();
            for (0..self.rows) |r| {
                const start = self.row_offsets[r];
                const end = self.row_offsets[r + 1];
                for (start..end) |pos| {
                    const index = r * self.cols + self.col_indices[pos];
                    out.data[index] = addSparseValue(T, out.data[index], self.values[pos]);
                }
            }
            return out;
        }

        pub fn toCoo(self: Self) SparseError!CooMatrix(T) {
            var row_indices = try self.allocator.alloc(usize, self.values.len);
            errdefer self.allocator.free(row_indices);
            const col_indices = try self.allocator.dupe(usize, self.col_indices);
            errdefer self.allocator.free(col_indices);
            const values = try self.allocator.dupe(T, self.values);
            errdefer self.allocator.free(values);

            for (0..self.rows) |row| {
                for (self.row_offsets[row]..self.row_offsets[row + 1]) |pos| row_indices[pos] = row;
            }

            return .{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = self.cols,
                .row_indices = row_indices,
                .col_indices = col_indices,
                .values = values,
            };
        }

        pub fn coalesced(self: Self) SparseError!Self {
            // Reuse COO canonicalization so CSR/CSC/COO all share one
            // duplicate-coordinate policy: sort by full matrix coordinates and
            // aggregate repeated entries without implicitly dropping explicit
            // structural zeros.
            var coo = try self.toCoo();
            defer coo.deinit();
            var canonical = try coo.coalesced();
            defer canonical.deinit();
            return canonical.toCsr();
        }

        pub fn add(self: Self, rhs: Self) SparseError!Self {
            if (self.rows != rhs.rows or self.cols != rhs.cols) return error.ShapeMismatch;
            var lhs_coo = try self.toCoo();
            defer lhs_coo.deinit();
            var rhs_coo = try rhs.toCoo();
            defer rhs_coo.deinit();
            var sum_coo = try lhs_coo.add(rhs_coo);
            defer sum_coo.deinit();
            return sum_coo.toCsr();
        }

        pub fn dropZerosNnz(self: Self) usize {
            return sparseDropZerosNnz(T, self.values);
        }

        pub fn dropZeros(self: Self) SparseError!Self {
            const nonzero_count = self.dropZerosNnz();

            var row_offsets = try self.allocator.alloc(usize, self.rows + 1);
            errdefer self.allocator.free(row_offsets);
            var col_indices = try self.allocator.alloc(usize, nonzero_count);
            errdefer self.allocator.free(col_indices);
            var values = try self.allocator.alloc(T, nonzero_count);
            errdefer self.allocator.free(values);

            var write: usize = 0;
            row_offsets[0] = 0;
            for (0..self.rows) |row| {
                for (self.row_offsets[row]..self.row_offsets[row + 1]) |pos| {
                    const value = self.values[pos];
                    if (isNonZero(T, value)) {
                        col_indices[write] = self.col_indices[pos];
                        values[write] = value;
                        write += 1;
                    }
                }
                row_offsets[row + 1] = write;
            }
            std.debug.assert(write == nonzero_count);

            return .{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = self.cols,
                .row_offsets = row_offsets,
                .col_indices = col_indices,
                .values = values,
            };
        }

        pub fn pruneZerosNnz(self: Self, tolerance: T) SparseError!usize {
            return sparsePruneZerosNnz(T, self.values, tolerance);
        }

        pub fn pruneZeros(self: Self, tolerance: T) SparseError!Self {
            const keep_count = try self.pruneZerosNnz(tolerance);

            var row_offsets = try self.allocator.alloc(usize, self.rows + 1);
            errdefer self.allocator.free(row_offsets);
            var col_indices = try self.allocator.alloc(usize, keep_count);
            errdefer self.allocator.free(col_indices);
            var values = try self.allocator.alloc(T, keep_count);
            errdefer self.allocator.free(values);

            var write: usize = 0;
            row_offsets[0] = 0;
            for (0..self.rows) |row| {
                for (self.row_offsets[row]..self.row_offsets[row + 1]) |pos| {
                    const value = self.values[pos];
                    if (sparseAbsValueExceedsTolerance(T, value, tolerance)) {
                        col_indices[write] = self.col_indices[pos];
                        values[write] = value;
                        write += 1;
                    }
                }
                row_offsets[row + 1] = write;
            }
            std.debug.assert(write == keep_count);
            return .{ .allocator = self.allocator, .rows = self.rows, .cols = self.cols, .row_offsets = row_offsets, .col_indices = col_indices, .values = values };
        }

        pub fn scale(self: Self, alpha: T) SparseError!Self {
            ensureNumeric(T);
            const row_offsets = try self.allocator.dupe(usize, self.row_offsets);
            errdefer self.allocator.free(row_offsets);
            const col_indices = try self.allocator.dupe(usize, self.col_indices);
            errdefer self.allocator.free(col_indices);
            var values = try self.allocator.alloc(T, self.values.len);
            errdefer self.allocator.free(values);

            // Scaling preserves the sparse structure intentionally; callers can
            // use `dropZeros()` afterwards when multiplying by zero or when an
            // integer factor creates explicit zeros.
            for (self.values, 0..) |value, i| values[i] = value * alpha;

            return .{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = self.cols,
                .row_offsets = row_offsets,
                .col_indices = col_indices,
                .values = values,
            };
        }

        pub fn scaleRows(self: Self, row_scale: []const T) SparseError!Self {
            ensureNumeric(T);
            if (row_scale.len != self.rows) return error.ShapeMismatch;
            const row_offsets = try self.allocator.dupe(usize, self.row_offsets);
            errdefer self.allocator.free(row_offsets);
            const col_indices = try self.allocator.dupe(usize, self.col_indices);
            errdefer self.allocator.free(col_indices);
            var values = try self.allocator.alloc(T, self.values.len);
            errdefer self.allocator.free(values);

            for (0..self.rows) |row| {
                for (self.row_offsets[row]..self.row_offsets[row + 1]) |pos| values[pos] = row_scale[row] * self.values[pos];
            }

            return .{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = self.cols,
                .row_offsets = row_offsets,
                .col_indices = col_indices,
                .values = values,
            };
        }

        pub fn scaleRowsInPlace(self: *Self, row_scale: []const T) SparseError!void {
            ensureNumeric(T);
            if (row_scale.len != self.rows) return error.ShapeMismatch;
            for (0..self.rows) |row| {
                for (self.row_offsets[row]..self.row_offsets[row + 1]) |pos| self.values[pos] *= row_scale[row];
            }
        }

        pub fn scaleColumns(self: Self, col_scale: []const T) SparseError!Self {
            ensureNumeric(T);
            if (col_scale.len != self.cols) return error.ShapeMismatch;
            const row_offsets = try self.allocator.dupe(usize, self.row_offsets);
            errdefer self.allocator.free(row_offsets);
            const col_indices = try self.allocator.dupe(usize, self.col_indices);
            errdefer self.allocator.free(col_indices);
            var values = try self.allocator.alloc(T, self.values.len);
            errdefer self.allocator.free(values);

            for (self.values, 0..) |value, pos| values[pos] = value * col_scale[self.col_indices[pos]];

            return .{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = self.cols,
                .row_offsets = row_offsets,
                .col_indices = col_indices,
                .values = values,
            };
        }

        pub fn scaleColumnsInPlace(self: *Self, col_scale: []const T) SparseError!void {
            ensureNumeric(T);
            if (col_scale.len != self.cols) return error.ShapeMismatch;
            for (self.values, self.col_indices) |*value, col| value.* *= col_scale[col];
        }

        pub fn scaleRowsAndColumns(self: Self, row_scale: []const T, col_scale: []const T) SparseError!Self {
            ensureNumeric(T);
            if (row_scale.len != self.rows or col_scale.len != self.cols) return error.ShapeMismatch;
            const row_offsets = try self.allocator.dupe(usize, self.row_offsets);
            errdefer self.allocator.free(row_offsets);
            const col_indices = try self.allocator.dupe(usize, self.col_indices);
            errdefer self.allocator.free(col_indices);
            var values = try self.allocator.alloc(T, self.values.len);
            errdefer self.allocator.free(values);

            for (0..self.rows) |row| {
                for (self.row_offsets[row]..self.row_offsets[row + 1]) |pos| {
                    values[pos] = row_scale[row] * self.values[pos] * col_scale[self.col_indices[pos]];
                }
            }

            return .{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = self.cols,
                .row_offsets = row_offsets,
                .col_indices = col_indices,
                .values = values,
            };
        }

        pub fn scaleRowsAndColumnsInPlace(self: *Self, row_scale: []const T, col_scale: []const T) SparseError!void {
            ensureNumeric(T);
            if (row_scale.len != self.rows or col_scale.len != self.cols) return error.ShapeMismatch;
            for (0..self.rows) |row| {
                for (self.row_offsets[row]..self.row_offsets[row + 1]) |pos| self.values[pos] *= row_scale[row] * col_scale[self.col_indices[pos]];
            }
        }

        pub fn neg(self: Self) SparseError!Self {
            ensureNumeric(T);
            return self.scale(negSparseValue(T, oneValue(T)));
        }

        pub fn negative(self: Self) SparseError!Self {
            return self.neg();
        }

        pub fn sub(self: Self, rhs: Self) SparseError!Self {
            if (self.rows != rhs.rows or self.cols != rhs.cols) return error.ShapeMismatch;
            var neg_rhs = try rhs.neg();
            defer neg_rhs.deinit();
            return self.add(neg_rhs);
        }

        pub fn hadamard(self: Self, rhs: Self) SparseError!Self {
            if (self.rows != rhs.rows or self.cols != rhs.cols) return error.ShapeMismatch;
            var lhs_coo = try self.toCoo();
            defer lhs_coo.deinit();
            var rhs_coo = try rhs.toCoo();
            defer rhs_coo.deinit();
            var product_coo = try lhs_coo.hadamard(rhs_coo);
            defer product_coo.deinit();
            return product_coo.toCsr();
        }

        pub fn mul(self: Self, rhs: Self) SparseError!Self {
            return self.hadamard(rhs);
        }

        pub fn multiply(self: Self, rhs: Self) SparseError!Self {
            return self.hadamard(rhs);
        }

        pub fn toCsc(self: Self) SparseError!CscMatrix(T) {
            var col_offsets = try self.allocator.alloc(usize, self.cols + 1);
            errdefer self.allocator.free(col_offsets);
            @memset(col_offsets, 0);
            for (self.col_indices) |col| col_offsets[col + 1] += 1;
            for (1..col_offsets.len) |i| col_offsets[i] += col_offsets[i - 1];

            var row_indices = try self.allocator.alloc(usize, self.values.len);
            errdefer self.allocator.free(row_indices);
            var values = try self.allocator.alloc(T, self.values.len);
            errdefer self.allocator.free(values);
            const next = try self.allocator.dupe(usize, col_offsets[0..self.cols]);
            defer self.allocator.free(next);

            // Fill by column using a mutable copy of the offsets.  Preserving
            // the CSR row traversal order keeps each CSC column's row indices
            // sorted without requiring a post-pass sort.
            for (0..self.rows) |row| {
                for (self.row_offsets[row]..self.row_offsets[row + 1]) |pos| {
                    const col = self.col_indices[pos];
                    const dst = next[col];
                    row_indices[dst] = row;
                    values[dst] = self.values[pos];
                    next[col] += 1;
                }
            }

            return .{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = self.cols,
                .col_offsets = col_offsets,
                .row_indices = row_indices,
                .values = values,
            };
        }

        pub fn matvec(self: Self, x: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            if (x.shape.len != 1) return error.NonVectorArray;
            if (x.shape[0] != self.cols) return error.ShapeMismatch;
            if (T == f64) return self.matvecF64(@as(array_mod.Array(f64), x));
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.rows});
            errdefer out.deinit();
            for (0..self.rows) |r| {
                var acc = zero(T);
                const start = self.row_offsets[r];
                const end = self.row_offsets[r + 1];
                for (start..end) |pos| acc += self.values[pos] * x.data[self.col_indices[pos]];
                out.data[r] = acc;
            }
            return out;
        }

        fn matvecF64(self: Self, x: array_mod.Array(f64)) SparseError!array_mod.Array(f64) {
            const veyra_view = try @as(CsrMatrix(f64), self).asVeyraView();
            var rhs = veyra.Vector(f64).fromSlice(self.allocator, x.data) catch return error.BackendFailure;
            defer rhs.deinit();
            var dst = veyra.Vector(f64).zeros(self.allocator, self.rows) catch return error.BackendFailure;
            defer dst.deinit();
            veyra.csrMatvec(f64, veyra_view, rhs.asView(), dst.asMut()) catch return error.BackendFailure;
            return array_mod.Array(f64).fromSlice(self.allocator, dst.data, &.{self.rows});
        }

        pub fn matvecResidualNorm(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T)) SparseError!T {
            ensureFloat(T);
            if (rhs.shape.len != 1) return error.NonVectorArray;
            var predicted = try self.matvec(x);
            defer predicted.deinit();
            return sparseVectorResidualNorm(T, predicted.data, rhs.data);
        }

        pub fn matvecResidualNormMeetsBound(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T), max_residual: T) SparseError!bool {
            try validateSparseValueRange(T, zero(T), max_residual);
            return (try self.matvecResidualNorm(x, rhs)) <= max_residual;
        }

        pub fn matvecRelativeResidualNorm(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T)) SparseError!T {
            const residual = try self.matvecResidualNorm(x, rhs);
            return sparseRelativeResidualNorm(T, residual, self.frobeniusNorm(), x.data, rhs.data);
        }

        pub fn matvecResidualSummary(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T)) SparseError!SparseResidualSummary {
            const residual = try self.matvecResidualNorm(x, rhs);
            return sparseResidualSummary(T, residual, self.frobeniusNorm(), x.data, rhs.data);
        }

        pub fn matvecResidualSummaryMeetsBounds(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T), max_residual: f64, max_relative_residual: f64) SparseError!bool {
            const summary = try self.matvecResidualSummary(x, rhs);
            return summary.meetsBounds(max_residual, max_relative_residual);
        }

        pub fn matvecRelativeResidualNormMeetsBound(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T), max_relative_residual: T) SparseError!bool {
            try validateSparseValueRange(T, zero(T), max_relative_residual);
            return (try self.matvecRelativeResidualNorm(x, rhs)) <= max_relative_residual;
        }

        pub fn matmat(self: Self, rhs: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            if (rhs.shape.len != 2) return error.NonMatrixArray;
            if (rhs.shape[0] != self.cols) return error.ShapeMismatch;
            if (T == f64) return self.matmatF64(@as(array_mod.Array(f64), rhs));
            var out = try array_mod.Array(T).zeros(self.allocator, &.{ self.rows, rhs.shape[1] });
            errdefer out.deinit();
            for (0..self.rows) |r| {
                const start = self.row_offsets[r];
                const end = self.row_offsets[r + 1];
                for (start..end) |pos| {
                    const col = self.col_indices[pos];
                    const value = self.values[pos];
                    for (0..rhs.shape[1]) |c| out.data[r * rhs.shape[1] + c] += value * rhs.data[col * rhs.shape[1] + c];
                }
            }
            return out;
        }

        fn matmatF64(self: Self, rhs: array_mod.Array(f64)) SparseError!array_mod.Array(f64) {
            const veyra_view = try @as(CsrMatrix(f64), self).asVeyraView();
            var rhs_matrix = veyra.Matrix(f64).fromSlice(self.allocator, rhs.shape[0], rhs.shape[1], .row_major, rhs.data) catch return error.BackendFailure;
            defer rhs_matrix.deinit();
            var dst = veyra.Matrix(f64).zeros(self.allocator, self.rows, rhs.shape[1], .row_major) catch return error.BackendFailure;
            defer dst.deinit();
            veyra.csrMatmat(f64, veyra_view, rhs_matrix.asView(), dst.asMut()) catch return error.BackendFailure;
            return array_mod.Array(f64).fromSlice(self.allocator, dst.data, &.{ self.rows, rhs.shape[1] });
        }

        pub fn matmatResidualFrobeniusNorm(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T)) SparseError!T {
            ensureFloat(T);
            var predicted = try self.matmat(x);
            defer predicted.deinit();
            return sparseMatrixResidualNorm(T, predicted, rhs);
        }

        pub fn matmatResidualFrobeniusNormMeetsBound(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T), max_residual: T) SparseError!bool {
            try validateSparseValueRange(T, zero(T), max_residual);
            return (try self.matmatResidualFrobeniusNorm(x, rhs)) <= max_residual;
        }

        pub fn matmatRelativeResidualFrobeniusNorm(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T)) SparseError!T {
            const residual = try self.matmatResidualFrobeniusNorm(x, rhs);
            return sparseRelativeMatrixResidualNorm(T, residual, self.frobeniusNorm(), x.data, rhs.data);
        }

        pub fn matmatResidualSummary(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T)) SparseError!SparseResidualSummary {
            const residual = try self.matmatResidualFrobeniusNorm(x, rhs);
            return sparseResidualSummary(T, residual, self.frobeniusNorm(), x.data, rhs.data);
        }

        pub fn matmatResidualSummaryMeetsBounds(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T), max_residual: f64, max_relative_residual: f64) SparseError!bool {
            const summary = try self.matmatResidualSummary(x, rhs);
            return summary.meetsBounds(max_residual, max_relative_residual);
        }

        pub fn matmatRelativeResidualFrobeniusNormMeetsBound(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T), max_relative_residual: T) SparseError!bool {
            try validateSparseValueRange(T, zero(T), max_relative_residual);
            return (try self.matmatRelativeResidualFrobeniusNorm(x, rhs)) <= max_relative_residual;
        }

        pub fn matmulSparse(self: Self, rhs: Self) SparseError!Self {
            ensureNumeric(T);
            if (self.cols != rhs.rows) return error.ShapeMismatch;
            var lhs = try self.coalesced();
            defer lhs.deinit();
            var rhs_canonical = try rhs.coalesced();
            defer rhs_canonical.deinit();

            var row_offsets = try self.allocator.alloc(usize, self.rows + 1);
            errdefer self.allocator.free(row_offsets);
            row_offsets[0] = 0;
            var markers = try self.allocator.alloc(bool, rhs.cols);
            defer self.allocator.free(markers);
            @memset(markers, false);
            const accum = try self.allocator.alloc(T, rhs.cols);
            defer self.allocator.free(accum);
            var touched = try self.allocator.alloc(usize, rhs.cols);
            defer self.allocator.free(touched);

            var total_nnz: usize = 0;
            for (0..lhs.rows) |row| {
                const touched_count = lhs.accumulateProductRow(rhs_canonical, row, markers, accum, touched);
                std.sort.insertion(usize, touched[0..touched_count], {}, struct {
                    fn less(_: void, lhs_col: usize, rhs_col: usize) bool {
                        return lhs_col < rhs_col;
                    }
                }.less);
                var row_nnz: usize = 0;
                for (touched[0..touched_count]) |col| {
                    if (isNonZero(T, accum[col])) row_nnz += 1;
                    markers[col] = false;
                }
                total_nnz = std.math.add(usize, total_nnz, row_nnz) catch return error.InvalidShape;
                row_offsets[row + 1] = total_nnz;
            }

            var col_indices = try self.allocator.alloc(usize, total_nnz);
            errdefer self.allocator.free(col_indices);
            var values = try self.allocator.alloc(T, total_nnz);
            errdefer self.allocator.free(values);

            var write: usize = 0;
            @memset(markers, false);
            for (0..lhs.rows) |row| {
                const touched_count = lhs.accumulateProductRow(rhs_canonical, row, markers, accum, touched);
                std.sort.insertion(usize, touched[0..touched_count], {}, struct {
                    fn less(_: void, lhs_col: usize, rhs_col: usize) bool {
                        return lhs_col < rhs_col;
                    }
                }.less);
                for (touched[0..touched_count]) |col| {
                    if (isNonZero(T, accum[col])) {
                        col_indices[write] = col;
                        values[write] = accum[col];
                        write += 1;
                    }
                    markers[col] = false;
                }
            }
            std.debug.assert(write == total_nnz);

            return .{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = rhs.cols,
                .row_offsets = row_offsets,
                .col_indices = col_indices,
                .values = values,
            };
        }

        fn accumulateProductRow(
            self: Self,
            rhs: Self,
            row: usize,
            markers: []bool,
            accum: []T,
            touched: []usize,
        ) usize {
            var touched_count: usize = 0;
            for (self.row_offsets[row]..self.row_offsets[row + 1]) |lhs_pos| {
                const inner = self.col_indices[lhs_pos];
                const lhs_value = self.values[lhs_pos];
                for (rhs.row_offsets[inner]..rhs.row_offsets[inner + 1]) |rhs_pos| {
                    const col = rhs.col_indices[rhs_pos];
                    if (!markers[col]) {
                        markers[col] = true;
                        touched[touched_count] = col;
                        touched_count += 1;
                        accum[col] = zero(T);
                    }
                    accum[col] = addSparseValue(T, accum[col], mulSparseValue(T, lhs_value, rhs.values[rhs_pos]));
                }
            }
            return touched_count;
        }

        pub fn transposeMatvec(self: Self, x: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            if (x.shape.len != 1) return error.NonVectorArray;
            if (x.shape[0] != self.rows) return error.ShapeMismatch;
            if (comptime T == f64) return self.transposeMatvecF64(@as(array_mod.Array(f64), x));
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.cols});
            errdefer out.deinit();
            for (0..self.rows) |r| {
                for (self.row_offsets[r]..self.row_offsets[r + 1]) |pos| out.data[self.col_indices[pos]] += self.values[pos] * x.data[r];
            }
            return out;
        }

        fn transposeMatvecF64(self: Self, x: array_mod.Array(f64)) SparseError!array_mod.Array(f64) {
            const veyra_view = try @as(CsrMatrix(f64), self).asVeyraView();
            var rhs = veyra.Vector(f64).fromSlice(self.allocator, x.data) catch return error.BackendFailure;
            defer rhs.deinit();
            var dst = veyra.Vector(f64).zeros(self.allocator, self.cols) catch return error.BackendFailure;
            defer dst.deinit();
            veyra.csrTransposeMatvec(f64, veyra_view, rhs.asView(), dst.asMut()) catch return error.BackendFailure;
            return array_mod.Array(f64).fromSlice(self.allocator, dst.data, &.{self.cols});
        }

        pub fn transposeMatvecResidualNorm(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T)) SparseError!T {
            ensureFloat(T);
            if (rhs.shape.len != 1) return error.NonVectorArray;
            var predicted = try self.transposeMatvec(x);
            defer predicted.deinit();
            return sparseVectorResidualNorm(T, predicted.data, rhs.data);
        }

        pub fn transposeMatvecResidualNormMeetsBound(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T), max_residual: T) SparseError!bool {
            try validateSparseValueRange(T, zero(T), max_residual);
            return (try self.transposeMatvecResidualNorm(x, rhs)) <= max_residual;
        }

        pub fn transposeMatvecRelativeResidualNorm(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T)) SparseError!T {
            const residual = try self.transposeMatvecResidualNorm(x, rhs);
            return sparseRelativeResidualNorm(T, residual, self.frobeniusNorm(), x.data, rhs.data);
        }

        pub fn transposeMatvecResidualSummary(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T)) SparseError!SparseResidualSummary {
            const residual = try self.transposeMatvecResidualNorm(x, rhs);
            return sparseResidualSummary(T, residual, self.frobeniusNorm(), x.data, rhs.data);
        }

        pub fn transposeMatvecResidualSummaryMeetsBounds(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T), max_residual: f64, max_relative_residual: f64) SparseError!bool {
            const summary = try self.transposeMatvecResidualSummary(x, rhs);
            return summary.meetsBounds(max_residual, max_relative_residual);
        }

        pub fn transposeMatvecRelativeResidualNormMeetsBound(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T), max_relative_residual: T) SparseError!bool {
            try validateSparseValueRange(T, zero(T), max_relative_residual);
            return (try self.transposeMatvecRelativeResidualNorm(x, rhs)) <= max_relative_residual;
        }

        pub fn transposeMatmat(self: Self, rhs: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            if (rhs.shape.len != 2) return error.NonMatrixArray;
            if (rhs.shape[0] != self.rows) return error.ShapeMismatch;
            if (comptime T == f64) return self.transposeMatmatF64(@as(array_mod.Array(f64), rhs));
            var out = try array_mod.Array(T).zeros(self.allocator, &.{ self.cols, rhs.shape[1] });
            errdefer out.deinit();
            for (0..self.rows) |r| {
                for (self.row_offsets[r]..self.row_offsets[r + 1]) |pos| {
                    const col = self.col_indices[pos];
                    for (0..rhs.shape[1]) |c| out.data[col * rhs.shape[1] + c] += self.values[pos] * rhs.data[r * rhs.shape[1] + c];
                }
            }
            return out;
        }

        fn transposeMatmatF64(self: Self, rhs: array_mod.Array(f64)) SparseError!array_mod.Array(f64) {
            const veyra_view = try @as(CsrMatrix(f64), self).asVeyraView();
            var rhs_matrix = veyra.Matrix(f64).fromSlice(self.allocator, rhs.shape[0], rhs.shape[1], .row_major, rhs.data) catch return error.BackendFailure;
            defer rhs_matrix.deinit();
            var dst = veyra.Matrix(f64).zeros(self.allocator, self.cols, rhs.shape[1], .row_major) catch return error.BackendFailure;
            defer dst.deinit();
            veyra.csrTransposeMatmat(f64, veyra_view, rhs_matrix.asView(), dst.asMut()) catch return error.BackendFailure;
            return array_mod.Array(f64).fromSlice(self.allocator, dst.data, &.{ self.cols, rhs.shape[1] });
        }

        pub fn transposeMatmatResidualFrobeniusNorm(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T)) SparseError!T {
            ensureFloat(T);
            var predicted = try self.transposeMatmat(x);
            defer predicted.deinit();
            return sparseMatrixResidualNorm(T, predicted, rhs);
        }

        pub fn transposeMatmatResidualFrobeniusNormMeetsBound(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T), max_residual: T) SparseError!bool {
            try validateSparseValueRange(T, zero(T), max_residual);
            return (try self.transposeMatmatResidualFrobeniusNorm(x, rhs)) <= max_residual;
        }

        pub fn transposeMatmatRelativeResidualFrobeniusNorm(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T)) SparseError!T {
            const residual = try self.transposeMatmatResidualFrobeniusNorm(x, rhs);
            return sparseRelativeMatrixResidualNorm(T, residual, self.frobeniusNorm(), x.data, rhs.data);
        }

        pub fn transposeMatmatResidualSummary(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T)) SparseError!SparseResidualSummary {
            const residual = try self.transposeMatmatResidualFrobeniusNorm(x, rhs);
            return sparseResidualSummary(T, residual, self.frobeniusNorm(), x.data, rhs.data);
        }

        pub fn transposeMatmatResidualSummaryMeetsBounds(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T), max_residual: f64, max_relative_residual: f64) SparseError!bool {
            const summary = try self.transposeMatmatResidualSummary(x, rhs);
            return summary.meetsBounds(max_residual, max_relative_residual);
        }

        pub fn transposeMatmatRelativeResidualFrobeniusNormMeetsBound(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T), max_relative_residual: T) SparseError!bool {
            try validateSparseValueRange(T, zero(T), max_relative_residual);
            return (try self.transposeMatmatRelativeResidualFrobeniusNorm(x, rhs)) <= max_relative_residual;
        }

        pub fn transpose(self: Self) SparseError!Self {
            var counts = try self.allocator.alloc(usize, self.cols);
            defer self.allocator.free(counts);
            @memset(counts, 0);
            for (self.col_indices) |col| counts[col] += 1;

            var row_offsets = try self.allocator.alloc(usize, self.cols + 1);
            errdefer self.allocator.free(row_offsets);
            row_offsets[0] = 0;
            for (counts, 0..) |count, i| row_offsets[i + 1] = row_offsets[i] + count;

            var next = try self.allocator.dupe(usize, row_offsets[0..self.cols]);
            defer self.allocator.free(next);
            var col_indices = try self.allocator.alloc(usize, self.values.len);
            errdefer self.allocator.free(col_indices);
            var values = try self.allocator.alloc(T, self.values.len);
            errdefer self.allocator.free(values);

            for (0..self.rows) |r| {
                const start = self.row_offsets[r];
                const end = self.row_offsets[r + 1];
                for (start..end) |pos| {
                    const dst_pos = next[self.col_indices[pos]];
                    next[self.col_indices[pos]] += 1;
                    col_indices[dst_pos] = r;
                    values[dst_pos] = self.values[pos];
                }
            }
            return .{ .allocator = self.allocator, .rows = self.cols, .cols = self.rows, .row_offsets = row_offsets, .col_indices = col_indices, .values = values };
        }

        pub fn matrixTranspose(self: Self) SparseError!Self {
            return self.transpose();
        }

        pub fn mT(self: Self) SparseError!Self {
            return self.matrixTranspose();
        }

        pub fn sum(self: Self) T {
            ensureNumeric(T);
            var total = zero(T);
            for (self.values) |value| total += value;
            return total;
        }

        pub fn absSum(self: Self) T {
            ensureNumeric(T);
            var total = zero(T);
            for (self.values) |value| total += absValue(T, value);
            return total;
        }

        pub fn sumInRange(self: Self, min_sum: T, max_sum: T) SparseError!bool {
            try validateSparseValueRange(T, min_sum, max_sum);
            return sparseValueInValidatedRange(T, self.sum(), min_sum, max_sum);
        }

        pub fn absSumInRange(self: Self, min_abs_sum: T, max_abs_sum: T) SparseError!bool {
            try validateSparseValueRange(T, min_abs_sum, max_abs_sum);
            return sparseValueInValidatedRange(T, self.absSum(), min_abs_sum, max_abs_sum);
        }

        pub fn minValue(self: Self) SparseError!T {
            return minStoredValue(T, self.values);
        }

        pub fn minValueIndex(self: Self) SparseError!usize {
            return minStoredValueIndex(T, self.values);
        }

        pub fn maxValue(self: Self) SparseError!T {
            return maxStoredValue(T, self.values);
        }

        pub fn maxValueIndex(self: Self) SparseError!usize {
            return maxStoredValueIndex(T, self.values);
        }

        pub fn minAbsValue(self: Self) SparseError!T {
            return minStoredAbsValue(T, self.values);
        }

        pub fn minAbsValueIndex(self: Self) SparseError!usize {
            return minStoredAbsValueIndex(T, self.values);
        }

        pub fn maxAbsValue(self: Self) SparseError!T {
            return maxStoredAbsValue(T, self.values);
        }

        pub fn maxAbsValueIndex(self: Self) SparseError!usize {
            return maxStoredAbsValueIndex(T, self.values);
        }

        pub fn nonFiniteCount(self: Self) usize {
            return sparseNonFiniteCount(T, self.values);
        }

        pub fn nonFiniteCountMeetsBound(self: Self, max_count: usize) bool {
            return self.nonFiniteCount() <= max_count;
        }

        pub fn nonFiniteCountInRange(self: Self, min_count: usize, max_count: usize) SparseError!bool {
            try validateCountRange(min_count, max_count);
            return sparseCountInValidatedRange(self.nonFiniteCount(), min_count, max_count);
        }

        pub fn allFinite(self: Self) bool {
            return sparseAllFinite(T, self.values);
        }

        pub fn rowNonFiniteCounts(self: Self) SparseError!array_mod.Array(usize) {
            var out = try array_mod.Array(usize).zeros(self.allocator, &.{self.rows});
            errdefer out.deinit();
            for (0..self.rows) |row| {
                for (self.row_offsets[row]..self.row_offsets[row + 1]) |pos| {
                    if (!sparseValueIsFinite(T, self.values[pos])) out.data[row] += 1;
                }
            }
            return out;
        }

        pub fn columnNonFiniteCounts(self: Self) SparseError!array_mod.Array(usize) {
            var out = try array_mod.Array(usize).zeros(self.allocator, &.{self.cols});
            errdefer out.deinit();
            for (self.values, self.col_indices) |value, col| {
                if (!sparseValueIsFinite(T, value)) out.data[col] += 1;
            }
            return out;
        }

        pub fn rowNonFiniteCountsInRange(self: Self, min_count: usize, max_count: usize) SparseError!bool {
            var counts = try self.rowNonFiniteCounts();
            defer counts.deinit();
            return sparseCountRangeInRange(counts.data, min_count, max_count);
        }

        pub fn rowNonFiniteCountsMeetBound(self: Self, max_count: usize) SparseError!bool {
            return self.rowNonFiniteCountsInRange(0, max_count);
        }

        pub fn columnNonFiniteCountsInRange(self: Self, min_count: usize, max_count: usize) SparseError!bool {
            var counts = try self.columnNonFiniteCounts();
            defer counts.deinit();
            return sparseCountRangeInRange(counts.data, min_count, max_count);
        }

        pub fn columnNonFiniteCountsMeetBound(self: Self, max_count: usize) SparseError!bool {
            return self.columnNonFiniteCountsInRange(0, max_count);
        }

        pub fn valueRangeInRange(self: Self, min_value: T, max_value: T) SparseError!bool {
            return sparseValueRangeInRange(T, self.values, min_value, max_value);
        }

        pub fn absValueRangeInRange(self: Self, min_abs_value: T, max_abs_value: T) SparseError!bool {
            return sparseAbsValueRangeInRange(T, self.values, min_abs_value, max_abs_value);
        }

        pub fn valueDynamicRange(self: Self) SparseError!f64 {
            return sparseStoredValueDynamicRange(T, self.values);
        }

        pub fn valueDynamicRangeMeetsBound(self: Self, max_dynamic_range: f64) SparseError!bool {
            return sparseStoredValueDynamicRangeMeetsBound(T, self.values, max_dynamic_range);
        }

        pub fn mean(self: Self) SparseError!f64 {
            ensureNumeric(T);
            const count = try sparseElementCount(self.rows, self.cols);
            if (count == 0) return error.EmptyArray;
            var total: f64 = 0;
            for (self.values) |value| total += sparseValueToF64(T, value);
            return total / sparseSizeToF64(count);
        }

        pub fn meanInRange(self: Self, min_mean: f64, max_mean: f64) SparseError!bool {
            try validateFiniteRange(min_mean, max_mean);
            return valueInF64Range(try self.mean(), min_mean, max_mean);
        }

        pub fn rowMeans(self: Self) SparseError!array_mod.Array(f64) {
            ensureNumeric(T);
            if (self.cols == 0) return error.EmptyArray;
            var out = try array_mod.Array(f64).zeros(self.allocator, &.{self.rows});
            errdefer out.deinit();
            for (0..self.rows) |row| {
                for (self.row_offsets[row]..self.row_offsets[row + 1]) |pos| out.data[row] += sparseValueToF64(T, self.values[pos]);
            }
            const divisor = sparseSizeToF64(self.cols);
            for (out.data) |*value| value.* /= divisor;
            return out;
        }

        pub fn columnMeans(self: Self) SparseError!array_mod.Array(f64) {
            ensureNumeric(T);
            if (self.rows == 0) return error.EmptyArray;
            var out = try array_mod.Array(f64).zeros(self.allocator, &.{self.cols});
            errdefer out.deinit();
            for (self.values, self.col_indices) |value, col| out.data[col] += sparseValueToF64(T, value);
            const divisor = sparseSizeToF64(self.rows);
            for (out.data) |*value| value.* /= divisor;
            return out;
        }

        pub fn rowMeansInRange(self: Self, min_mean: f64, max_mean: f64) SparseError!bool {
            try validateFiniteRange(min_mean, max_mean);
            var means = try self.rowMeans();
            defer means.deinit();
            for (means.data) |value| {
                if (!valueInF64Range(value, min_mean, max_mean)) return false;
            }
            return true;
        }

        pub fn columnMeansInRange(self: Self, min_mean: f64, max_mean: f64) SparseError!bool {
            try validateFiniteRange(min_mean, max_mean);
            var means = try self.columnMeans();
            defer means.deinit();
            for (means.data) |value| {
                if (!valueInF64Range(value, min_mean, max_mean)) return false;
            }
            return true;
        }

        pub fn variance(self: Self, correction: f64) SparseError!f64 {
            ensureNumeric(T);
            const count = try sparseElementCount(self.rows, self.cols);
            var sum_value: f64 = 0;
            var sum_sq: f64 = 0;
            for (self.values) |value| {
                sum_value += sparseValueToF64(T, value);
                sum_sq += sparseValueSquareToF64(T, value);
            }
            return sparseVarianceFromSums(sum_value, sum_sq, count, correction);
        }

        pub fn varianceInRange(self: Self, correction: f64, min_variance: f64, max_variance: f64) SparseError!bool {
            try validateNonNegativeRange(min_variance, max_variance);
            return valueInF64Range(try self.variance(correction), min_variance, max_variance);
        }

        pub fn stddev(self: Self, correction: f64) SparseError!f64 {
            return @sqrt(try self.variance(correction));
        }

        pub fn stddevInRange(self: Self, correction: f64, min_stddev: f64, max_stddev: f64) SparseError!bool {
            try validateNonNegativeRange(min_stddev, max_stddev);
            return sparseStddevInValidatedRangeFromVariance(try self.variance(correction), min_stddev, max_stddev);
        }

        pub fn sampleVariance(self: Self) SparseError!f64 {
            return self.variance(1);
        }

        pub fn sampleVarianceInRange(self: Self, min_variance: f64, max_variance: f64) SparseError!bool {
            return self.varianceInRange(1, min_variance, max_variance);
        }

        pub fn sampleStddev(self: Self) SparseError!f64 {
            return self.stddev(1);
        }

        pub fn sampleStddevInRange(self: Self, min_stddev: f64, max_stddev: f64) SparseError!bool {
            return self.stddevInRange(1, min_stddev, max_stddev);
        }

        pub fn rowVariances(self: Self, correction: f64) SparseError!array_mod.Array(f64) {
            ensureNumeric(T);
            if (self.cols == 0) return error.EmptyArray;
            var sums = try self.allocator.alloc(f64, self.rows);
            defer self.allocator.free(sums);
            @memset(sums, 0);
            var out = try array_mod.Array(f64).zeros(self.allocator, &.{self.rows});
            errdefer out.deinit();
            for (0..self.rows) |row| {
                for (self.row_offsets[row]..self.row_offsets[row + 1]) |pos| {
                    const numeric = sparseValueToF64(T, self.values[pos]);
                    sums[row] += numeric;
                    out.data[row] += numeric * numeric;
                }
            }
            try finalizeVarianceArray(out.data, sums, self.cols, correction);
            return out;
        }

        pub fn rowVariancesInRange(self: Self, correction: f64, min_variance: f64, max_variance: f64) SparseError!bool {
            try validateNonNegativeRange(min_variance, max_variance);
            var variances = try self.rowVariances(correction);
            defer variances.deinit();
            for (variances.data) |value| {
                if (!valueInF64Range(value, min_variance, max_variance)) return false;
            }
            return true;
        }

        pub fn columnVariancesInRange(self: Self, correction: f64, min_variance: f64, max_variance: f64) SparseError!bool {
            try validateNonNegativeRange(min_variance, max_variance);
            var variances = try self.columnVariances(correction);
            defer variances.deinit();
            for (variances.data) |value| {
                if (!valueInF64Range(value, min_variance, max_variance)) return false;
            }
            return true;
        }

        pub fn rowStddevs(self: Self, correction: f64) SparseError!array_mod.Array(f64) {
            const out = try self.rowVariances(correction);
            sqrtArray(out.data);
            return out;
        }

        pub fn rowStddevsInRange(self: Self, correction: f64, min_stddev: f64, max_stddev: f64) SparseError!bool {
            try validateNonNegativeRange(min_stddev, max_stddev);
            var stddevs = try self.rowStddevs(correction);
            defer stddevs.deinit();
            for (stddevs.data) |value| {
                if (!valueInF64Range(value, min_stddev, max_stddev)) return false;
            }
            return true;
        }

        pub fn columnStddevsInRange(self: Self, correction: f64, min_stddev: f64, max_stddev: f64) SparseError!bool {
            try validateNonNegativeRange(min_stddev, max_stddev);
            var stddevs = try self.columnStddevs(correction);
            defer stddevs.deinit();
            for (stddevs.data) |value| {
                if (!valueInF64Range(value, min_stddev, max_stddev)) return false;
            }
            return true;
        }

        pub fn rowSampleVariances(self: Self) SparseError!array_mod.Array(f64) {
            return self.rowVariances(1);
        }

        pub fn rowSampleStddevs(self: Self) SparseError!array_mod.Array(f64) {
            return self.rowStddevs(1);
        }

        pub fn columnVariances(self: Self, correction: f64) SparseError!array_mod.Array(f64) {
            ensureNumeric(T);
            if (self.rows == 0) return error.EmptyArray;
            var sums = try self.allocator.alloc(f64, self.cols);
            defer self.allocator.free(sums);
            @memset(sums, 0);
            var out = try array_mod.Array(f64).zeros(self.allocator, &.{self.cols});
            errdefer out.deinit();
            for (self.values, self.col_indices) |value, col| {
                const numeric = sparseValueToF64(T, value);
                sums[col] += numeric;
                out.data[col] += numeric * numeric;
            }
            try finalizeVarianceArray(out.data, sums, self.rows, correction);
            return out;
        }

        pub fn columnStddevs(self: Self, correction: f64) SparseError!array_mod.Array(f64) {
            const out = try self.columnVariances(correction);
            sqrtArray(out.data);
            return out;
        }

        pub fn columnSampleVariances(self: Self) SparseError!array_mod.Array(f64) {
            return self.columnVariances(1);
        }

        pub fn columnSampleStddevs(self: Self) SparseError!array_mod.Array(f64) {
            return self.columnStddevs(1);
        }

        pub fn frobeniusNorm(self: Self) T {
            ensureFloat(T);
            if (T == f64) {
                const veyra_view = @as(CsrMatrix(f64), self).asVeyraView() catch return 0;
                return @as(T, @floatCast(veyra.csrFrobeniusNorm(f64, veyra_view)));
            }
            var total = zero(T);
            for (self.values) |value| total += value * value;
            return @sqrt(total);
        }

        pub fn frobeniusNormMeetsBound(self: Self, max_norm: T) SparseError!bool {
            ensureFloat(T);
            try validateSparseValueRange(T, zero(T), max_norm);
            return self.frobeniusNorm() <= max_norm;
        }

        pub fn density(self: Self) SparseError!f64 {
            const total = self.rows * self.cols;
            if (total == 0) return 0;
            if (T == f64) {
                const veyra_view = try @as(CsrMatrix(f64), self).asVeyraView();
                return veyra.csrDensity(f64, veyra_view) catch return error.BackendFailure;
            }
            return @as(f64, @floatFromInt(self.values.len)) / @as(f64, @floatFromInt(total));
        }

        pub fn densityInRange(self: Self, min_density: f64, max_density: f64) SparseError!bool {
            try validateNonNegativeRange(min_density, max_density);
            const current = try self.density();
            return current >= min_density and current <= max_density;
        }

        pub fn oneNorm(self: Self) SparseError!T {
            ensureNumeric(T);
            var col_sums = try self.allocator.alloc(T, self.cols);
            defer self.allocator.free(col_sums);
            @memset(col_sums, zero(T));
            for (self.values, self.col_indices) |value, col| col_sums[col] += absValue(T, value);
            var max_sum = zero(T);
            for (col_sums) |sum_value| {
                if (sum_value > max_sum) max_sum = sum_value;
            }
            return max_sum;
        }

        pub fn oneNormMeetsBound(self: Self, max_norm: T) SparseError!bool {
            ensureNumeric(T);
            try validateSparseValueRange(T, zero(T), max_norm);
            return (try self.oneNorm()) <= max_norm;
        }

        pub fn infNorm(self: Self) SparseError!T {
            ensureNumeric(T);
            var max_sum = zero(T);
            for (0..self.rows) |row| {
                var row_sum = zero(T);
                for (self.row_offsets[row]..self.row_offsets[row + 1]) |pos| row_sum += absValue(T, self.values[pos]);
                if (row_sum > max_sum) max_sum = row_sum;
            }
            return max_sum;
        }

        pub fn infNormMeetsBound(self: Self, max_norm: T) SparseError!bool {
            ensureNumeric(T);
            try validateSparseValueRange(T, zero(T), max_norm);
            return (try self.infNorm()) <= max_norm;
        }

        pub fn rowNnz(self: Self) SparseError!array_mod.Array(usize) {
            var out = try array_mod.Array(usize).zeros(self.allocator, &.{self.rows});
            errdefer out.deinit();
            for (0..self.rows) |r| out.data[r] = self.row_offsets[r + 1] - self.row_offsets[r];
            return out;
        }

        pub fn columnNnz(self: Self) SparseError!array_mod.Array(usize) {
            var out = try array_mod.Array(usize).zeros(self.allocator, &.{self.cols});
            errdefer out.deinit();
            for (self.col_indices) |col| out.data[col] += 1;
            return out;
        }

        pub fn averageRowNnz(self: Self) SparseError!f64 {
            return sparseCountAverage(self.values.len, self.rows);
        }

        pub fn averageColumnNnz(self: Self) SparseError!f64 {
            return sparseCountAverage(self.values.len, self.cols);
        }

        pub fn averageRowNnzInRange(self: Self, min_average: f64, max_average: f64) SparseError!bool {
            return sparseCountAverageInRange(self.values.len, self.rows, min_average, max_average);
        }

        pub fn averageColumnNnzInRange(self: Self, min_average: f64, max_average: f64) SparseError!bool {
            return sparseCountAverageInRange(self.values.len, self.cols, min_average, max_average);
        }

        pub fn emptyRowCount(self: Self) usize {
            var empty_count: usize = 0;
            for (0..self.rows) |row| {
                if (self.row_offsets[row] == self.row_offsets[row + 1]) empty_count += 1;
            }
            return empty_count;
        }

        pub fn emptyColumnCount(self: Self) SparseError!usize {
            var counts = try self.columnNnz();
            defer counts.deinit();
            var empty_count: usize = 0;
            for (counts.data) |count| {
                if (count == 0) empty_count += 1;
            }
            return empty_count;
        }

        pub fn emptyRowCountMeetsBound(self: Self, max_count: usize) bool {
            return self.emptyRowCount() <= max_count;
        }

        pub fn emptyRowCountInRange(self: Self, min_count: usize, max_count: usize) SparseError!bool {
            try validateCountRange(min_count, max_count);
            return sparseCountInValidatedRange(self.emptyRowCount(), min_count, max_count);
        }

        pub fn emptyColumnCountMeetsBound(self: Self, max_count: usize) SparseError!bool {
            return (try self.emptyColumnCount()) <= max_count;
        }

        pub fn emptyColumnCountInRange(self: Self, min_count: usize, max_count: usize) SparseError!bool {
            try validateCountRange(min_count, max_count);
            return sparseCountInValidatedRange(try self.emptyColumnCount(), min_count, max_count);
        }

        pub fn emptyRowFraction(self: Self) SparseError!f64 {
            return sparseCountFraction(self.emptyRowCount(), self.rows);
        }

        pub fn emptyColumnFraction(self: Self) SparseError!f64 {
            return sparseCountFraction(try self.emptyColumnCount(), self.cols);
        }

        pub fn emptyRowFractionInRange(self: Self, min_fraction: f64, max_fraction: f64) SparseError!bool {
            return sparseCountFractionInRange(self.emptyRowCount(), self.rows, min_fraction, max_fraction);
        }

        pub fn emptyColumnFractionInRange(self: Self, min_fraction: f64, max_fraction: f64) SparseError!bool {
            return sparseCountFractionInRange(try self.emptyColumnCount(), self.cols, min_fraction, max_fraction);
        }

        pub fn minRowNnz(self: Self) SparseError!usize {
            var counts = try self.rowNnz();
            defer counts.deinit();
            return sparseMinCount(counts.data);
        }

        pub fn maxRowNnz(self: Self) SparseError!usize {
            var counts = try self.rowNnz();
            defer counts.deinit();
            return sparseMaxCount(counts.data);
        }

        pub fn minColumnNnz(self: Self) SparseError!usize {
            var counts = try self.columnNnz();
            defer counts.deinit();
            return sparseMinCount(counts.data);
        }

        pub fn maxColumnNnz(self: Self) SparseError!usize {
            var counts = try self.columnNnz();
            defer counts.deinit();
            return sparseMaxCount(counts.data);
        }

        pub fn rowNnzRangeInRange(self: Self, min_count: usize, max_count: usize) SparseError!bool {
            var counts = try self.rowNnz();
            defer counts.deinit();
            return sparseCountRangeInRange(counts.data, min_count, max_count);
        }

        pub fn columnNnzRangeInRange(self: Self, min_count: usize, max_count: usize) SparseError!bool {
            var counts = try self.columnNnz();
            defer counts.deinit();
            return sparseCountRangeInRange(counts.data, min_count, max_count);
        }

        pub fn rowNnzSpread(self: Self) SparseError!usize {
            var counts = try self.rowNnz();
            defer counts.deinit();
            return sparseCountSpread(counts.data);
        }

        pub fn columnNnzSpread(self: Self) SparseError!usize {
            var counts = try self.columnNnz();
            defer counts.deinit();
            return sparseCountSpread(counts.data);
        }

        pub fn rowNnzSpreadMeetsBound(self: Self, max_spread: usize) SparseError!bool {
            var counts = try self.rowNnz();
            defer counts.deinit();
            return sparseCountSpreadMeetsBound(counts.data, max_spread);
        }

        pub fn columnNnzSpreadMeetsBound(self: Self, max_spread: usize) SparseError!bool {
            var counts = try self.columnNnz();
            defer counts.deinit();
            return sparseCountSpreadMeetsBound(counts.data, max_spread);
        }

        pub fn rowSums(self: Self) SparseError!array_mod.Array(T) {
            ensureNumeric(T);
            if (comptime T == f64) return self.rowSumsF64();
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.rows});
            errdefer out.deinit();
            for (0..self.rows) |r| {
                for (self.row_offsets[r]..self.row_offsets[r + 1]) |pos| out.data[r] += self.values[pos];
            }
            return out;
        }

        fn rowSumsF64(self: Self) SparseError!array_mod.Array(f64) {
            const veyra_view = try @as(CsrMatrix(f64), self).asVeyraView();
            var out = veyra.Vector(f64).zeros(self.allocator, self.rows) catch return error.BackendFailure;
            defer out.deinit();
            veyra.csrRowSums(f64, veyra_view, out.asMut()) catch return error.BackendFailure;
            return array_mod.Array(f64).fromSlice(self.allocator, out.data, &.{self.rows});
        }

        pub fn columnSums(self: Self) SparseError!array_mod.Array(T) {
            ensureNumeric(T);
            if (comptime T == f64) return self.columnSumsF64();
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.cols});
            errdefer out.deinit();
            for (0..self.rows) |r| {
                for (self.row_offsets[r]..self.row_offsets[r + 1]) |pos| out.data[self.col_indices[pos]] += self.values[pos];
            }
            return out;
        }

        fn columnSumsF64(self: Self) SparseError!array_mod.Array(f64) {
            const veyra_view = try @as(CsrMatrix(f64), self).asVeyraView();
            var out = veyra.Vector(f64).zeros(self.allocator, self.cols) catch return error.BackendFailure;
            defer out.deinit();
            veyra.csrColumnSumsWithWorkspace(f64, veyra_view, out.asMut()) catch return error.BackendFailure;
            return array_mod.Array(f64).fromSlice(self.allocator, out.data, &.{self.cols});
        }

        pub fn rowSumsInRange(self: Self, min_sum: T, max_sum: T) SparseError!bool {
            var sums = try self.rowSums();
            defer sums.deinit();
            return sparseValueRangeInRange(T, sums.data, min_sum, max_sum);
        }

        pub fn columnSumsInRange(self: Self, min_sum: T, max_sum: T) SparseError!bool {
            var sums = try self.columnSums();
            defer sums.deinit();
            return sparseValueRangeInRange(T, sums.data, min_sum, max_sum);
        }

        pub fn rowMins(self: Self) SparseError!array_mod.Array(T) {
            ensureNumeric(T);
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.rows});
            errdefer out.deinit();
            for (0..self.rows) |row| {
                for (self.row_offsets[row]..self.row_offsets[row + 1]) |pos| {
                    const value = self.values[pos];
                    if (valueLess(T, value, out.data[row])) out.data[row] = value;
                }
            }
            return out;
        }

        pub fn rowMaxes(self: Self) SparseError!array_mod.Array(T) {
            ensureNumeric(T);
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.rows});
            errdefer out.deinit();
            for (0..self.rows) |row| {
                for (self.row_offsets[row]..self.row_offsets[row + 1]) |pos| {
                    const value = self.values[pos];
                    if (valueGreater(T, value, out.data[row])) out.data[row] = value;
                }
            }
            return out;
        }

        pub fn columnMins(self: Self) SparseError!array_mod.Array(T) {
            ensureNumeric(T);
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.cols});
            errdefer out.deinit();
            for (self.values, self.col_indices) |value, col| {
                if (valueLess(T, value, out.data[col])) out.data[col] = value;
            }
            return out;
        }

        pub fn columnMaxes(self: Self) SparseError!array_mod.Array(T) {
            ensureNumeric(T);
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.cols});
            errdefer out.deinit();
            for (self.values, self.col_indices) |value, col| {
                if (valueGreater(T, value, out.data[col])) out.data[col] = value;
            }
            return out;
        }

        pub fn rowMinsInRange(self: Self, min_value: T, max_value: T) SparseError!bool {
            var values = try self.rowMins();
            defer values.deinit();
            return sparseValueRangeInRange(T, values.data, min_value, max_value);
        }

        pub fn columnMinsInRange(self: Self, min_value: T, max_value: T) SparseError!bool {
            var values = try self.columnMins();
            defer values.deinit();
            return sparseValueRangeInRange(T, values.data, min_value, max_value);
        }

        pub fn rowMaxesInRange(self: Self, min_value: T, max_value: T) SparseError!bool {
            var values = try self.rowMaxes();
            defer values.deinit();
            return sparseValueRangeInRange(T, values.data, min_value, max_value);
        }

        pub fn columnMaxesInRange(self: Self, min_value: T, max_value: T) SparseError!bool {
            var values = try self.columnMaxes();
            defer values.deinit();
            return sparseValueRangeInRange(T, values.data, min_value, max_value);
        }

        pub fn rowMinAbs(self: Self) SparseError!array_mod.Array(T) {
            ensureNumeric(T);
            var out = try self.rowMaxAbs();
            for (0..self.rows) |row| {
                for (self.row_offsets[row]..self.row_offsets[row + 1]) |pos| {
                    const magnitude = absValue(T, self.values[pos]);
                    if (magnitude < out.data[row]) out.data[row] = magnitude;
                }
            }
            return out;
        }

        pub fn rowMaxAbs(self: Self) SparseError!array_mod.Array(T) {
            ensureNumeric(T);
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.rows});
            errdefer out.deinit();
            for (0..self.rows) |row| {
                for (self.row_offsets[row]..self.row_offsets[row + 1]) |pos| {
                    const magnitude = absValue(T, self.values[pos]);
                    if (magnitude > out.data[row]) out.data[row] = magnitude;
                }
            }
            return out;
        }

        pub fn columnMinAbs(self: Self) SparseError!array_mod.Array(T) {
            ensureNumeric(T);
            var out = try self.columnMaxAbs();
            for (self.values, self.col_indices) |value, col| {
                const magnitude = absValue(T, value);
                if (magnitude < out.data[col]) out.data[col] = magnitude;
            }
            return out;
        }

        pub fn columnMaxAbs(self: Self) SparseError!array_mod.Array(T) {
            ensureNumeric(T);
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.cols});
            errdefer out.deinit();
            for (self.values, self.col_indices) |value, col| {
                const magnitude = absValue(T, value);
                if (magnitude > out.data[col]) out.data[col] = magnitude;
            }
            return out;
        }

        pub fn rowSampleVariancesInRange(self: Self, min_variance: f64, max_variance: f64) SparseError!bool {
            return self.rowVariancesInRange(1, min_variance, max_variance);
        }

        pub fn rowSampleStddevsInRange(self: Self, min_stddev: f64, max_stddev: f64) SparseError!bool {
            return self.rowStddevsInRange(1, min_stddev, max_stddev);
        }

        pub fn columnSampleVariancesInRange(self: Self, min_variance: f64, max_variance: f64) SparseError!bool {
            return self.columnVariancesInRange(1, min_variance, max_variance);
        }

        pub fn columnSampleStddevsInRange(self: Self, min_stddev: f64, max_stddev: f64) SparseError!bool {
            return self.columnStddevsInRange(1, min_stddev, max_stddev);
        }

        pub fn rowAbsSums(self: Self) SparseError!array_mod.Array(T) {
            ensureNumeric(T);
            if (comptime T == f64) return self.rowAbsSumsF64();
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.rows});
            errdefer out.deinit();
            for (0..self.rows) |r| {
                for (self.row_offsets[r]..self.row_offsets[r + 1]) |pos| out.data[r] += absValue(T, self.values[pos]);
            }
            return out;
        }

        fn rowAbsSumsF64(self: Self) SparseError!array_mod.Array(f64) {
            const veyra_view = try @as(CsrMatrix(f64), self).asVeyraView();
            var out = veyra.Vector(f64).zeros(self.allocator, self.rows) catch return error.BackendFailure;
            defer out.deinit();
            veyra.csrRowAbsSums(f64, veyra_view, out.asMut()) catch return error.BackendFailure;
            return array_mod.Array(f64).fromSlice(self.allocator, out.data, &.{self.rows});
        }

        pub fn columnAbsSums(self: Self) SparseError!array_mod.Array(T) {
            ensureNumeric(T);
            if (comptime T == f64) return self.columnAbsSumsF64();
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.cols});
            errdefer out.deinit();
            for (0..self.rows) |r| {
                for (self.row_offsets[r]..self.row_offsets[r + 1]) |pos| out.data[self.col_indices[pos]] += absValue(T, self.values[pos]);
            }
            return out;
        }

        fn columnAbsSumsF64(self: Self) SparseError!array_mod.Array(f64) {
            const veyra_view = try @as(CsrMatrix(f64), self).asVeyraView();
            var out = veyra.Vector(f64).zeros(self.allocator, self.cols) catch return error.BackendFailure;
            defer out.deinit();
            veyra.csrColumnAbsSumsWithWorkspace(f64, veyra_view, out.asMut()) catch return error.BackendFailure;
            return array_mod.Array(f64).fromSlice(self.allocator, out.data, &.{self.cols});
        }

        pub fn rowMinAbsInRange(self: Self, min_value: T, max_value: T) SparseError!bool {
            var values = try self.rowMinAbs();
            defer values.deinit();
            return sparseValueRangeInRange(T, values.data, min_value, max_value);
        }

        pub fn columnMinAbsInRange(self: Self, min_value: T, max_value: T) SparseError!bool {
            var values = try self.columnMinAbs();
            defer values.deinit();
            return sparseValueRangeInRange(T, values.data, min_value, max_value);
        }

        pub fn rowMaxAbsInRange(self: Self, min_value: T, max_value: T) SparseError!bool {
            var values = try self.rowMaxAbs();
            defer values.deinit();
            return sparseValueRangeInRange(T, values.data, min_value, max_value);
        }

        pub fn columnMaxAbsInRange(self: Self, min_value: T, max_value: T) SparseError!bool {
            var values = try self.columnMaxAbs();
            defer values.deinit();
            return sparseValueRangeInRange(T, values.data, min_value, max_value);
        }

        pub fn rowAbsSumsInRange(self: Self, min_sum: T, max_sum: T) SparseError!bool {
            var sums = try self.rowAbsSums();
            defer sums.deinit();
            return sparseValueRangeInRange(T, sums.data, min_sum, max_sum);
        }

        pub fn columnAbsSumsInRange(self: Self, min_sum: T, max_sum: T) SparseError!bool {
            var sums = try self.columnAbsSums();
            defer sums.deinit();
            return sparseValueRangeInRange(T, sums.data, min_sum, max_sum);
        }

        pub fn rowNorms(self: Self) SparseError!array_mod.Array(T) {
            ensureFloat(T);
            if (comptime T == f64) return self.rowNormsF64();
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.rows});
            errdefer out.deinit();
            for (0..self.rows) |r| {
                var total = zero(T);
                for (self.row_offsets[r]..self.row_offsets[r + 1]) |pos| total += self.values[pos] * self.values[pos];
                out.data[r] = @sqrt(total);
            }
            return out;
        }

        fn rowNormsF64(self: Self) SparseError!array_mod.Array(f64) {
            const veyra_view = try @as(CsrMatrix(f64), self).asVeyraView();
            var out = veyra.Vector(f64).zeros(self.allocator, self.rows) catch return error.BackendFailure;
            defer out.deinit();
            veyra.csrRowNorms(f64, veyra_view, out.asMut()) catch return error.BackendFailure;
            return array_mod.Array(f64).fromSlice(self.allocator, out.data, &.{self.rows});
        }

        pub fn columnNorms(self: Self) SparseError!array_mod.Array(T) {
            ensureFloat(T);
            if (comptime T == f64) return self.columnNormsF64();
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.cols});
            errdefer out.deinit();
            for (0..self.rows) |r| {
                for (self.row_offsets[r]..self.row_offsets[r + 1]) |pos| {
                    const col = self.col_indices[pos];
                    out.data[col] += self.values[pos] * self.values[pos];
                }
            }
            for (out.data) |*value| value.* = @sqrt(value.*);
            return out;
        }

        fn columnNormsF64(self: Self) SparseError!array_mod.Array(f64) {
            const veyra_view = try @as(CsrMatrix(f64), self).asVeyraView();
            var out = veyra.Vector(f64).zeros(self.allocator, self.cols) catch return error.BackendFailure;
            defer out.deinit();
            veyra.csrColumnNormsWithWorkspace(f64, veyra_view, out.asMut()) catch return error.BackendFailure;
            return array_mod.Array(f64).fromSlice(self.allocator, out.data, &.{self.cols});
        }

        pub fn rowNormsInRange(self: Self, min_norm: T, max_norm: T) SparseError!bool {
            try validateSparseValueRange(T, zero(T), max_norm);
            if (min_norm < zero(T)) return error.InvalidShape;
            var norms = try self.rowNorms();
            defer norms.deinit();
            return sparseValueRangeInRange(T, norms.data, min_norm, max_norm);
        }

        pub fn columnNormsInRange(self: Self, min_norm: T, max_norm: T) SparseError!bool {
            try validateSparseValueRange(T, zero(T), max_norm);
            if (min_norm < zero(T)) return error.InvalidShape;
            var norms = try self.columnNorms();
            defer norms.deinit();
            return sparseValueRangeInRange(T, norms.data, min_norm, max_norm);
        }

        pub fn diagonal(self: Self) SparseError!array_mod.Array(T) {
            if (self.rows != self.cols) return error.NonMatrixArray;
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.rows});
            errdefer out.deinit();
            for (0..self.rows) |r| out.data[r] = self.get(r, r) orelse zero(T);
            return out;
        }

        pub fn diagonalOffset(self: Self, offset: isize) SparseError!array_mod.Array(T) {
            return sparseDenseDiagonalOffset(T, self, offset);
        }

        pub fn diagonalWithOffset(self: Self, offset: isize) SparseError!array_mod.Array(T) {
            return self.diagonalOffset(offset);
        }

        pub fn setDiagonal(self: *Self, value: T) SparseError!void {
            if (self.rows != self.cols) return error.NonMatrixArray;
            for (0..self.rows) |row| {
                var found = false;
                for (self.row_offsets[row]..self.row_offsets[row + 1]) |pos| {
                    if (self.col_indices[pos] == row) {
                        self.values[pos] = value;
                        found = true;
                    }
                }
                if (!found) return error.InvalidShape;
            }
        }

        pub fn addToDiagonal(self: *Self, alpha: T) SparseError!void {
            ensureNumeric(T);
            if (alpha == zero(T)) return;
            if (self.rows != self.cols) return error.NonMatrixArray;
            for (0..self.rows) |row| {
                var found = false;
                for (self.row_offsets[row]..self.row_offsets[row + 1]) |pos| {
                    if (self.col_indices[pos] == row) {
                        self.values[pos] += alpha;
                        found = true;
                    }
                }
                if (!found) return error.InvalidShape;
            }
        }

        pub fn setDiagonalValues(self: *Self, diagonal_values: []const T) SparseError!void {
            if (self.rows != self.cols) return error.NonMatrixArray;
            if (diagonal_values.len != self.rows) return error.ShapeMismatch;
            for (0..self.rows) |row| {
                var found = false;
                for (self.row_offsets[row]..self.row_offsets[row + 1]) |pos| {
                    if (self.col_indices[pos] == row) {
                        self.values[pos] = diagonal_values[row];
                        found = true;
                    }
                }
                if (!found) return error.InvalidShape;
            }
        }

        pub fn setDiagonalVector(self: *Self, diagonal_values: []const T) SparseError!void {
            return self.setDiagonalValues(diagonal_values);
        }

        pub fn addDiagonalValues(self: *Self, diagonal_values: []const T) SparseError!void {
            ensureNumeric(T);
            if (self.rows != self.cols) return error.NonMatrixArray;
            if (diagonal_values.len != self.rows) return error.ShapeMismatch;
            var any_nonzero = false;
            for (diagonal_values) |value| {
                if (isNonZero(T, value)) {
                    any_nonzero = true;
                    break;
                }
            }
            if (!any_nonzero) return;
            for (0..self.rows) |row| {
                var found = false;
                for (self.row_offsets[row]..self.row_offsets[row + 1]) |pos| {
                    if (self.col_indices[pos] == row) {
                        self.values[pos] += diagonal_values[row];
                        found = true;
                    }
                }
                if (!found) return error.InvalidShape;
            }
        }

        pub fn addDiagonalVector(self: *Self, diagonal_values: []const T) SparseError!void {
            return self.addDiagonalValues(diagonal_values);
        }

        pub fn multiplyDiagonalValues(self: *Self, diagonal_values: []const T) SparseError!void {
            ensureNumeric(T);
            if (self.rows != self.cols) return error.NonMatrixArray;
            if (diagonal_values.len != self.rows) return error.ShapeMismatch;
            var all_one = true;
            for (diagonal_values) |value| {
                if (value != oneValue(T)) {
                    all_one = false;
                    break;
                }
            }
            if (all_one) return;
            for (0..self.rows) |row| {
                var found = false;
                for (self.row_offsets[row]..self.row_offsets[row + 1]) |pos| {
                    if (self.col_indices[pos] == row) {
                        self.values[pos] *= diagonal_values[row];
                        found = true;
                    }
                }
                if (!found) return error.InvalidShape;
            }
        }

        pub fn multiplyDiagonalVector(self: *Self, diagonal_values: []const T) SparseError!void {
            return self.multiplyDiagonalValues(diagonal_values);
        }

        pub fn minAbsDiagonal(self: Self) SparseError!T {
            var diagonal_values = try self.diagonal();
            defer diagonal_values.deinit();
            return (try sparseDiagonalAbsRange(T, diagonal_values.data)).min_abs;
        }

        pub fn maxAbsDiagonal(self: Self) SparseError!T {
            var diagonal_values = try self.diagonal();
            defer diagonal_values.deinit();
            return (try sparseDiagonalAbsRange(T, diagonal_values.data)).max_abs;
        }

        pub fn diagonalDynamicRange(self: Self) SparseError!f64 {
            var diagonal_values = try self.diagonal();
            defer diagonal_values.deinit();
            return sparseDiagonalDynamicRangeFromValues(T, diagonal_values.data);
        }

        pub fn diagonalDynamicRangeMeetsBound(self: Self, max_dynamic_range: f64) SparseError!bool {
            var diagonal_values = try self.diagonal();
            defer diagonal_values.deinit();
            return sparseDiagonalDynamicRangeMeetsBoundFromValues(T, diagonal_values.data, max_dynamic_range);
        }

        pub fn trace(self: Self) SparseError!T {
            ensureNumeric(T);
            if (self.rows != self.cols) return error.NonMatrixArray;
            var total = zero(T);
            for (0..self.rows) |r| total = addSparseValue(T, total, self.get(r, r) orelse zero(T));
            return total;
        }

        pub fn traceInRange(self: Self, min_value: T, max_value: T) SparseError!bool {
            try validateSparseValueRange(T, min_value, max_value);
            const trace_value = try self.trace();
            return trace_value >= min_value and trace_value <= max_value;
        }

        pub fn normalizedTrace(self: Self) SparseError!f64 {
            return sparseNormalizedTraceFromTrace(T, try self.trace(), self.rows);
        }

        pub fn normalizedTraceInRange(self: Self, min_value: f64, max_value: f64) SparseError!bool {
            try validateFiniteRange(min_value, max_value);
            return sparseNormalizedTraceInRangeFromTrace(T, try self.trace(), self.rows, min_value, max_value);
        }

        pub fn missingDiagonalCount(self: Self) SparseError!usize {
            if (self.rows != self.cols) return error.NonMatrixArray;
            var count: usize = 0;
            for (0..self.rows) |r| {
                if (!self.hasEntry(r, r)) count += 1;
            }
            return count;
        }

        pub fn missingDiagonalCountMeetsBound(self: Self, max_count: usize) SparseError!bool {
            return (try self.missingDiagonalCount()) <= max_count;
        }

        pub fn missingDiagonalCountInRange(self: Self, min_count: usize, max_count: usize) SparseError!bool {
            try validateCountRange(min_count, max_count);
            return sparseCountInValidatedRange(try self.missingDiagonalCount(), min_count, max_count);
        }

        pub fn zeroDiagonalCount(self: Self) SparseError!usize {
            if (self.rows != self.cols) return error.NonMatrixArray;
            var count: usize = 0;
            for (0..self.rows) |r| {
                if (self.get(r, r)) |value| {
                    if (value == zero(T)) count += 1;
                }
            }
            return count;
        }

        pub fn zeroDiagonalCountMeetsBound(self: Self, max_count: usize) SparseError!bool {
            return (try self.zeroDiagonalCount()) <= max_count;
        }

        pub fn zeroDiagonalCountInRange(self: Self, min_count: usize, max_count: usize) SparseError!bool {
            try validateCountRange(min_count, max_count);
            return sparseCountInValidatedRange(try self.zeroDiagonalCount(), min_count, max_count);
        }

        pub fn nonPositiveDiagonalCount(self: Self) SparseError!usize {
            ensureNumeric(T);
            if (self.rows != self.cols) return error.NonMatrixArray;
            var count: usize = 0;
            for (0..self.rows) |row| {
                if (self.get(row, row)) |value| {
                    if (value <= zero(T)) count += 1;
                }
            }
            return count;
        }

        pub fn nonPositiveDiagonalCountMeetsBound(self: Self, max_count: usize) SparseError!bool {
            return (try self.nonPositiveDiagonalCount()) <= max_count;
        }

        pub fn nonPositiveDiagonalCountInRange(self: Self, min_count: usize, max_count: usize) SparseError!bool {
            try validateCountRange(min_count, max_count);
            return sparseCountInValidatedRange(try self.nonPositiveDiagonalCount(), min_count, max_count);
        }

        pub fn bandwidth(self: Self) SparseError!usize {
            if (self.rows != self.cols) return error.NonMatrixArray;
            var bw: usize = 0;
            for (0..self.rows) |r| {
                for (self.row_offsets[r]..self.row_offsets[r + 1]) |pos| {
                    const c = self.col_indices[pos];
                    const distance = if (r > c) r - c else c - r;
                    if (distance > bw) bw = distance;
                }
            }
            return bw;
        }

        pub fn bandwidthMeetsBound(self: Self, max_bandwidth: usize) SparseError!bool {
            if (self.rows != self.cols) return error.NonMatrixArray;
            for (0..self.rows) |r| {
                for (self.row_offsets[r]..self.row_offsets[r + 1]) |pos| {
                    const c = self.col_indices[pos];
                    const distance = if (r > c) r - c else c - r;
                    if (distance > max_bandwidth) return false;
                }
            }
            return true;
        }

        pub fn columnIntersectionBandwidth(self: Self) SparseError!usize {
            var coo = try self.toCoo();
            defer coo.deinit();
            return coo.columnIntersectionBandwidth();
        }

        pub fn columnIntersectionBandwidthMeetsBound(self: Self, max_bandwidth: usize) SparseError!bool {
            var coo = try self.toCoo();
            defer coo.deinit();
            return coo.columnIntersectionBandwidthMeetsBound(max_bandwidth);
        }

        pub fn lowerNnz(self: Self, comptime strict: bool) SparseError!usize {
            if (self.rows != self.cols) return error.NonMatrixArray;
            var count: usize = 0;
            for (0..self.rows) |row| {
                for (self.row_offsets[row]..self.row_offsets[row + 1]) |pos| {
                    if (triangularIndexMatches(row, self.col_indices[pos], strict, true)) count += 1;
                }
            }
            return count;
        }

        pub fn upperNnz(self: Self, comptime strict: bool) SparseError!usize {
            if (self.rows != self.cols) return error.NonMatrixArray;
            var count: usize = 0;
            for (0..self.rows) |row| {
                for (self.row_offsets[row]..self.row_offsets[row + 1]) |pos| {
                    if (triangularIndexMatches(row, self.col_indices[pos], strict, false)) count += 1;
                }
            }
            return count;
        }

        pub fn lowerNnzMeetsBound(self: Self, comptime strict: bool, max_count: usize) SparseError!bool {
            return (try self.lowerNnz(strict)) <= max_count;
        }

        pub fn upperNnzMeetsBound(self: Self, comptime strict: bool, max_count: usize) SparseError!bool {
            return (try self.upperNnz(strict)) <= max_count;
        }

        pub fn lowerNnzInRange(self: Self, comptime strict: bool, min_count: usize, max_count: usize) SparseError!bool {
            if (min_count > max_count) return error.InvalidShape;
            const count = try self.lowerNnz(strict);
            return count >= min_count and count <= max_count;
        }

        pub fn upperNnzInRange(self: Self, comptime strict: bool, min_count: usize, max_count: usize) SparseError!bool {
            if (min_count > max_count) return error.InvalidShape;
            const count = try self.upperNnz(strict);
            return count >= min_count and count <= max_count;
        }

        pub fn lowerProfile(self: Self) SparseError!usize {
            if (self.rows != self.cols) return error.NonMatrixArray;
            var builder = try SparseProfileBuilder.init(self.allocator, self.rows);
            defer builder.deinit(self.allocator);
            for (0..self.rows) |row| {
                for (self.row_offsets[row]..self.row_offsets[row + 1]) |pos| builder.observe(row, self.col_indices[pos]);
            }
            return builder.lowerProfile();
        }

        pub fn upperProfile(self: Self) SparseError!usize {
            if (self.rows != self.cols) return error.NonMatrixArray;
            var builder = try SparseProfileBuilder.init(self.allocator, self.rows);
            defer builder.deinit(self.allocator);
            for (0..self.rows) |row| {
                for (self.row_offsets[row]..self.row_offsets[row + 1]) |pos| builder.observe(row, self.col_indices[pos]);
            }
            return builder.upperProfile();
        }

        pub fn profile(self: Self) SparseError!SparseProfile {
            if (self.rows != self.cols) return error.NonMatrixArray;
            var builder = try SparseProfileBuilder.init(self.allocator, self.rows);
            defer builder.deinit(self.allocator);
            for (0..self.rows) |row| {
                for (self.row_offsets[row]..self.row_offsets[row + 1]) |pos| builder.observe(row, self.col_indices[pos]);
            }
            return builder.profile();
        }

        pub fn lowerProfileMeetsBound(self: Self, max_profile: usize) SparseError!bool {
            return (try self.lowerProfile()) <= max_profile;
        }

        pub fn upperProfileMeetsBound(self: Self, max_profile: usize) SparseError!bool {
            return (try self.upperProfile()) <= max_profile;
        }

        pub fn profileMeetsBounds(self: Self, max_lower_profile: usize, max_upper_profile: usize) SparseError!bool {
            const current = try self.profile();
            return current.meetsBounds(max_lower_profile, max_upper_profile);
        }

        pub fn profileTotalMeetsBound(self: Self, max_total_profile: usize) SparseError!bool {
            const current = try self.profile();
            return current.totalMeetsBound(max_total_profile);
        }

        pub fn diagonallyDominant(self: Self) SparseError!bool {
            var canonical = try self.coalesced();
            defer canonical.deinit();
            var coo = try canonical.toCoo();
            defer coo.deinit();
            return sparseDiagonalDominanceFromCanonicalEntries(
                T,
                self.allocator,
                canonical.rows,
                canonical.cols,
                coo.row_indices,
                coo.col_indices,
                coo.values,
                false,
            );
        }

        pub fn diagonalDominanceMargin(self: Self) SparseError!f64 {
            var canonical = try self.coalesced();
            defer canonical.deinit();
            var coo = try canonical.toCoo();
            defer coo.deinit();
            return sparseDiagonalDominanceMarginFromCanonicalEntries(
                T,
                self.allocator,
                canonical.rows,
                canonical.cols,
                coo.row_indices,
                coo.col_indices,
                coo.values,
            );
        }

        pub fn diagonalDominanceMarginMeetsBound(self: Self, min_margin: f64) SparseError!bool {
            var canonical = try self.coalesced();
            defer canonical.deinit();
            var coo = try canonical.toCoo();
            defer coo.deinit();
            return sparseDiagonalDominanceMarginMeetsBoundFromCanonicalEntries(
                T,
                self.allocator,
                canonical.rows,
                canonical.cols,
                coo.row_indices,
                coo.col_indices,
                coo.values,
                min_margin,
            );
        }

        pub fn strictlyDiagonallyDominant(self: Self) SparseError!bool {
            var canonical = try self.coalesced();
            defer canonical.deinit();
            var coo = try canonical.toCoo();
            defer coo.deinit();
            return sparseDiagonalDominanceFromCanonicalEntries(
                T,
                self.allocator,
                canonical.rows,
                canonical.cols,
                coo.row_indices,
                coo.col_indices,
                coo.values,
                true,
            );
        }

        pub fn structurallySymmetric(self: Self) SparseError!bool {
            if (self.rows != self.cols) return error.NonMatrixArray;
            for (0..self.rows) |r| {
                for (self.row_offsets[r]..self.row_offsets[r + 1]) |pos| {
                    if (!self.hasEntry(self.col_indices[pos], r)) return false;
                }
            }
            return true;
        }

        pub fn numericallySymmetric(self: Self, tolerance: T) SparseError!bool {
            ensureNumeric(T);
            if (self.rows != self.cols) return error.NonMatrixArray;
            for (0..self.rows) |r| {
                for (self.row_offsets[r]..self.row_offsets[r + 1]) |pos| {
                    const value = self.get(r, self.col_indices[pos]) orelse return false;
                    const mirror = self.get(self.col_indices[pos], r) orelse return false;
                    if (absDifference(T, value, mirror) > tolerance) return false;
                }
            }
            return true;
        }

        pub fn symmetryResidualFrobeniusNorm(self: Self) SparseError!T {
            ensureFloat(T);
            var dense = try self.toDense();
            defer dense.deinit();
            return sparseSymmetryResidualFrobeniusNormFromDense(T, dense.data, self.rows, self.cols);
        }

        pub fn symmetryRelativeResidualFrobeniusNorm(self: Self) SparseError!T {
            const residual = try self.symmetryResidualFrobeniusNorm();
            return residual / @max(oneValue(T), self.frobeniusNorm());
        }

        pub fn symmetryResidualFrobeniusNormMeetsBound(self: Self, max_residual: T) SparseError!bool {
            try validateSparseValueRange(T, zero(T), max_residual);
            return (try self.symmetryResidualFrobeniusNorm()) <= max_residual;
        }

        pub fn symmetryRelativeResidualFrobeniusNormMeetsBound(self: Self, max_relative_residual: T) SparseError!bool {
            try validateSparseValueRange(T, zero(T), max_relative_residual);
            return (try self.symmetryRelativeResidualFrobeniusNorm()) <= max_relative_residual;
        }

        pub fn get(self: Self, row: usize, col: usize) ?T {
            if (row >= self.rows or col >= self.cols) return null;
            var found = false;
            var total = zero(T);
            // CSR input can legally contain duplicate structural entries.
            // Accumulating here keeps point access and diagnostics aligned
            // with `toDense()` materialization instead of exposing whichever
            // duplicate happened to appear first.
            for (self.row_offsets[row]..self.row_offsets[row + 1]) |pos| {
                if (self.col_indices[pos] == col) {
                    total = addSparseValue(T, total, self.values[pos]);
                    found = true;
                }
            }
            return if (found) total else null;
        }

        fn hasEntry(self: Self, row: usize, col: usize) bool {
            if (row >= self.rows or col >= self.cols) return false;
            for (self.row_offsets[row]..self.row_offsets[row + 1]) |pos| {
                if (self.col_indices[pos] == col) return true;
            }
            return false;
        }

        pub fn setExisting(self: *Self, row: usize, col: usize, value: T) SparseError!void {
            if (row >= self.rows or col >= self.cols) return error.IndexOutOfBounds;
            var found = false;
            for (self.row_offsets[row]..self.row_offsets[row + 1]) |pos| {
                if (self.col_indices[pos] == col) {
                    self.values[pos] = value;
                    found = true;
                }
            }
            if (!found) return error.InvalidShape;
        }

        pub fn setStoredValue(self: *Self, row: usize, col: usize, value: T) SparseError!void {
            return self.setExisting(row, col, value);
        }

        pub fn solveTriangular(self: Self, rhs: array_mod.Array(T), triangle: Triangle, diag_kind: Diagonal) SparseError!array_mod.Array(T) {
            if (self.rows != self.cols) return error.NonMatrixArray;
            if (rhs.shape.len != 1 and rhs.shape.len != 2) return error.InvalidShape;
            if (rhs.shape[0] != self.rows) return error.ShapeMismatch;
            if (comptime T == f64) return self.solveTriangularF64(@as(array_mod.Array(f64), rhs), triangle, diag_kind);
            return self.solveTriangularReference(rhs, triangle, diag_kind);
        }

        fn solveTriangularF64(self: Self, rhs: array_mod.Array(f64), triangle: Triangle, diag_kind: Diagonal) SparseError!array_mod.Array(f64) {
            const veyra_view = try @as(CsrMatrix(f64), self).asVeyraView();
            if (rhs.shape.len == 1) {
                var rhs_vec = veyra.Vector(f64).fromSlice(self.allocator, rhs.data) catch return error.BackendFailure;
                defer rhs_vec.deinit();
                var dst = veyra.Vector(f64).zeros(self.allocator, self.rows) catch return error.BackendFailure;
                defer dst.deinit();
                veyra.csrSolveTriangular(f64, veyra_view, rhs_vec.asView(), dst.asMut(), toVeyraTriangle(triangle), toVeyraDiagonal(diag_kind)) catch return error.BackendFailure;
                return array_mod.Array(f64).fromSlice(self.allocator, dst.data, &.{self.rows});
            }
            var rhs_mat = veyra.Matrix(f64).fromSlice(self.allocator, rhs.shape[0], rhs.shape[1], .row_major, rhs.data) catch return error.BackendFailure;
            defer rhs_mat.deinit();
            var dst = veyra.Matrix(f64).zeros(self.allocator, self.rows, rhs.shape[1], .row_major) catch return error.BackendFailure;
            defer dst.deinit();
            veyra.csrSolveTriangularMatrix(f64, veyra_view, rhs_mat.asView(), dst.asMut(), toVeyraTriangle(triangle), toVeyraDiagonal(diag_kind)) catch return error.BackendFailure;
            return array_mod.Array(f64).fromSlice(self.allocator, dst.data, &.{ self.rows, rhs.shape[1] });
        }

        fn solveTriangularReference(self: Self, rhs: array_mod.Array(T), triangle: Triangle, diag_kind: Diagonal) SparseError!array_mod.Array(T) {
            if (rhs.shape.len == 1) {
                var out = try array_mod.Array(T).zeros(self.allocator, &.{self.rows});
                errdefer out.deinit();
                try self.solveTriangularVector(rhs.data, out.data, triangle, diag_kind);
                return out;
            }
            var out = try array_mod.Array(T).zeros(self.allocator, &.{ self.rows, rhs.shape[1] });
            errdefer out.deinit();
            for (0..rhs.shape[1]) |c| {
                var rhs_col = try self.allocator.alloc(T, self.rows);
                defer self.allocator.free(rhs_col);
                const out_col = try self.allocator.alloc(T, self.rows);
                defer self.allocator.free(out_col);
                for (0..self.rows) |r| rhs_col[r] = rhs.data[r * rhs.shape[1] + c];
                try self.solveTriangularVector(rhs_col, out_col, triangle, diag_kind);
                for (0..self.rows) |r| out.data[r * rhs.shape[1] + c] = out_col[r];
            }
            return out;
        }

        fn solveTriangularVector(self: Self, rhs: []const T, out: []T, triangle: Triangle, diag_kind: Diagonal) SparseError!void {
            switch (triangle) {
                .lower => {
                    for (0..self.rows) |r| {
                        var acc = rhs[r];
                        var diag: ?T = if (diag_kind == .unit) oneValue(T) else null;
                        for (self.row_offsets[r]..self.row_offsets[r + 1]) |pos| {
                            const c = self.col_indices[pos];
                            if (c < r) acc -= self.values[pos] * out[c] else if (c == r) diag = self.values[pos];
                        }
                        const d = diag orelse return error.BackendFailure;
                        if (d == zero(T)) return error.BackendFailure;
                        out[r] = acc / d;
                    }
                },
                .upper => {
                    var r = self.rows;
                    while (r > 0) {
                        r -= 1;
                        var acc = rhs[r];
                        var diag: ?T = if (diag_kind == .unit) oneValue(T) else null;
                        for (self.row_offsets[r]..self.row_offsets[r + 1]) |pos| {
                            const c = self.col_indices[pos];
                            if (c > r) acc -= self.values[pos] * out[c] else if (c == r) diag = self.values[pos];
                        }
                        const d = diag orelse return error.BackendFailure;
                        if (d == zero(T)) return error.BackendFailure;
                        out[r] = acc / d;
                    }
                },
            }
        }
    };
}

pub fn CscMatrix(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        rows: usize,
        cols: usize,
        col_offsets: []usize,
        row_indices: []usize,
        values: []T,

        pub fn eye(allocator: std.mem.Allocator, rows: usize, cols: usize) SparseError!Self {
            const diag_len = @min(rows, cols);
            var col_offsets = try allocator.alloc(usize, cols + 1);
            errdefer allocator.free(col_offsets);
            var row_indices = try allocator.alloc(usize, diag_len);
            errdefer allocator.free(row_indices);
            var values = try allocator.alloc(T, diag_len);
            errdefer allocator.free(values);

            var write: usize = 0;
            col_offsets[0] = 0;
            for (0..cols) |col| {
                if (col < rows) {
                    row_indices[write] = col;
                    values[write] = oneValue(T);
                    write += 1;
                }
                col_offsets[col + 1] = write;
            }
            std.debug.assert(write == diag_len);
            return .{
                .allocator = allocator,
                .rows = rows,
                .cols = cols,
                .col_offsets = col_offsets,
                .row_indices = row_indices,
                .values = values,
            };
        }

        pub fn identity(allocator: std.mem.Allocator, size: usize) SparseError!Self {
            return Self.eye(allocator, size, size);
        }

        pub fn fromDiagonal(allocator: std.mem.Allocator, diagonal_values: []const T, offset: isize) SparseError!Self {
            var coo = try CooMatrix(T).fromDiagonal(allocator, diagonal_values, offset);
            defer coo.deinit();
            return coo.toCsc();
        }

        pub fn fromDiagonalNnz(diagonal_values: []const T, offset: isize) SparseError!usize {
            return CooMatrix(T).fromDiagonalNnz(diagonal_values, offset);
        }

        pub fn fromDiagonalPruned(allocator: std.mem.Allocator, diagonal_values: []const T, offset: isize, tolerance: T) SparseError!Self {
            var coo = try CooMatrix(T).fromDiagonalPruned(allocator, diagonal_values, offset, tolerance);
            defer coo.deinit();
            return coo.toCsc();
        }

        pub fn fromDiagonalPrunedNnz(diagonal_values: []const T, offset: isize, tolerance: T) SparseError!usize {
            return CooMatrix(T).fromDiagonalPrunedNnz(diagonal_values, offset, tolerance);
        }

        pub fn fromCompressedSlices(
            allocator: std.mem.Allocator,
            rows: usize,
            cols: usize,
            col_offsets: []const usize,
            row_indices: []const usize,
            values: []const T,
        ) SparseError!Self {
            if (col_offsets.len != cols + 1) return error.ShapeMismatch;
            if (row_indices.len != values.len) return error.ShapeMismatch;
            if (col_offsets[0] != 0 or col_offsets[col_offsets.len - 1] != values.len) return error.ShapeMismatch;
            for (col_offsets[1..], col_offsets[0 .. col_offsets.len - 1]) |current, previous| {
                if (current < previous) return error.ShapeMismatch;
            }
            for (row_indices) |row| if (row >= rows) return error.IndexOutOfBounds;
            return .{
                .allocator = allocator,
                .rows = rows,
                .cols = cols,
                .col_offsets = try allocator.dupe(usize, col_offsets),
                .row_indices = try allocator.dupe(usize, row_indices),
                .values = try allocator.dupe(T, values),
            };
        }

        pub fn fromDense(input: array_mod.Array(T)) SparseError!Self {
            if (input.shape.len != 2) return error.NonMatrixArray;
            const rows = input.shape[0];
            const cols = input.shape[1];
            const nonzero_count = try Self.fromDenseNnz(input);
            var col_offsets = try input.allocator.alloc(usize, cols + 1);
            errdefer input.allocator.free(col_offsets);
            var row_indices = try input.allocator.alloc(usize, nonzero_count);
            errdefer input.allocator.free(row_indices);
            var values = try input.allocator.alloc(T, nonzero_count);
            errdefer input.allocator.free(values);
            var write: usize = 0;
            col_offsets[0] = 0;
            for (0..cols) |c| {
                for (0..rows) |r| {
                    const value = input.data[r * cols + c];
                    if (isNonZero(T, value)) {
                        row_indices[write] = r;
                        values[write] = value;
                        write += 1;
                    }
                }
                col_offsets[c + 1] = write;
            }
            return .{ .allocator = input.allocator, .rows = rows, .cols = cols, .col_offsets = col_offsets, .row_indices = row_indices, .values = values };
        }

        pub fn fromDenseNnz(input: array_mod.Array(T)) SparseError!usize {
            return sparseDenseNnz(T, input);
        }

        pub fn fromDensePruned(input: array_mod.Array(T), tolerance: T) SparseError!Self {
            if (input.shape.len != 2) return error.NonMatrixArray;
            const rows = input.shape[0];
            const cols = input.shape[1];
            const keep_count = try Self.fromDensePrunedNnz(input, tolerance);

            var col_offsets = try input.allocator.alloc(usize, cols + 1);
            errdefer input.allocator.free(col_offsets);
            var row_indices = try input.allocator.alloc(usize, keep_count);
            errdefer input.allocator.free(row_indices);
            var values = try input.allocator.alloc(T, keep_count);
            errdefer input.allocator.free(values);

            var write: usize = 0;
            col_offsets[0] = 0;
            for (0..cols) |col| {
                for (0..rows) |row| {
                    const value = input.data[row * cols + col];
                    if (sparseAbsValueExceedsTolerance(T, value, tolerance)) {
                        row_indices[write] = row;
                        values[write] = value;
                        write += 1;
                    }
                }
                col_offsets[col + 1] = write;
            }
            std.debug.assert(write == keep_count);
            return .{ .allocator = input.allocator, .rows = rows, .cols = cols, .col_offsets = col_offsets, .row_indices = row_indices, .values = values };
        }

        pub fn fromDensePrunedNnz(input: array_mod.Array(T), tolerance: T) SparseError!usize {
            return sparseDensePrunedNnz(T, input, tolerance);
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.col_offsets);
            self.allocator.free(self.row_indices);
            self.allocator.free(self.values);
            self.* = undefined;
        }

        pub fn fillStoredValues(self: *Self, value: T) void {
            @memset(self.values, value);
        }

        pub fn fillValues(self: *Self, value: T) void {
            self.fillStoredValues(value);
        }

        pub fn scaleStoredValues(self: *Self, alpha: T) void {
            ensureNumeric(T);
            for (self.values) |*value| value.* = value.* * alpha;
        }

        pub fn scaleValues(self: *Self, alpha: T) void {
            self.scaleStoredValues(alpha);
        }

        pub fn setStoredValuesFromDense(self: *Self, src: array_mod.Array(T)) SparseError!void {
            if (src.shape.len != 2) return error.NonMatrixArray;
            if (src.shape[0] != self.rows or src.shape[1] != self.cols) return error.ShapeMismatch;
            for (0..self.cols) |col| {
                for (self.col_offsets[col]..self.col_offsets[col + 1]) |pos| {
                    self.values[pos] = src.data[self.row_indices[pos] * self.cols + col];
                }
            }
        }

        pub fn clone(self: Self) SparseError!Self {
            const col_offsets = try self.allocator.dupe(usize, self.col_offsets);
            errdefer self.allocator.free(col_offsets);
            const row_indices = try self.allocator.dupe(usize, self.row_indices);
            errdefer self.allocator.free(row_indices);
            const values = try self.allocator.dupe(T, self.values);
            errdefer self.allocator.free(values);
            return .{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = self.cols,
                .col_offsets = col_offsets,
                .row_indices = row_indices,
                .values = values,
            };
        }

        pub fn copy(self: Self) SparseError!Self {
            return self.clone();
        }

        pub fn detach(self: Self) SparseError!Self {
            return self.clone();
        }

        pub fn astype(self: Self, comptime U: type) SparseError!CscMatrix(U) {
            const col_offsets = try self.allocator.dupe(usize, self.col_offsets);
            errdefer self.allocator.free(col_offsets);
            const row_indices = try self.allocator.dupe(usize, self.row_indices);
            errdefer self.allocator.free(row_indices);
            var values = try self.allocator.alloc(U, self.values.len);
            errdefer self.allocator.free(values);
            for (self.values, 0..) |value, index| {
                values[index] = sparseCastValue(U, value);
            }
            return .{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = self.cols,
                .col_offsets = col_offsets,
                .row_indices = row_indices,
                .values = values,
            };
        }

        pub fn mapValues(self: Self, comptime U: type, comptime map: fn (T) U) SparseError!CscMatrix(U) {
            const col_offsets = try self.allocator.dupe(usize, self.col_offsets);
            errdefer self.allocator.free(col_offsets);
            const row_indices = try self.allocator.dupe(usize, self.row_indices);
            errdefer self.allocator.free(row_indices);
            var values = try self.allocator.alloc(U, self.values.len);
            errdefer self.allocator.free(values);
            for (self.values, 0..) |value, index| {
                values[index] = map(value);
            }
            return .{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = self.cols,
                .col_offsets = col_offsets,
                .row_indices = row_indices,
                .values = values,
            };
        }

        pub fn mapStoredValues(self: Self, comptime U: type, comptime map: fn (T) U) SparseError!CscMatrix(U) {
            return self.mapValues(U, map);
        }

        pub fn mapValuesInPlace(self: *Self, comptime map: fn (T) T) void {
            for (self.values) |*value| value.* = map(value.*);
        }

        pub fn mapStoredValuesInPlace(self: *Self, comptime map: fn (T) T) void {
            self.mapValuesInPlace(map);
        }

        pub fn positive(self: Self) SparseError!Self {
            ensureNumeric(T);
            return self.clone();
        }

        pub fn abs(self: Self) SparseError!Self {
            ensureNumeric(T);
            const out = try self.clone();
            for (out.values) |*value| value.* = absValue(T, value.*);
            return out;
        }

        pub fn absolute(self: Self) SparseError!Self {
            return self.abs();
        }

        pub fn fabs(self: Self) SparseError!Self {
            return self.abs();
        }

        pub fn square(self: Self) SparseError!Self {
            ensureNumeric(T);
            const out = try self.clone();
            for (out.values) |*value| value.* = value.* * value.*;
            return out;
        }

        pub fn sqrt(self: Self) SparseError!Self {
            ensureFloat(T);
            const out = try self.clone();
            for (out.values) |*value| value.* = @sqrt(value.*);
            return out;
        }

        pub fn rsqrt(self: Self) SparseError!Self {
            ensureFloat(T);
            const out = try self.clone();
            for (out.values) |*value| value.* = oneValue(T) / @sqrt(value.*);
            return out;
        }

        pub fn cbrt(self: Self) SparseError!Self {
            ensureFloat(T);
            const out = try self.clone();
            for (out.values) |*value| value.* = cbrtSparseValue(T, value.*);
            return out;
        }

        pub fn reciprocal(self: Self) SparseError!Self {
            ensureFloat(T);
            const out = try self.clone();
            for (out.values) |*value| value.* = oneValue(T) / value.*;
            return out;
        }

        pub fn floor(self: Self) SparseError!Self {
            ensureNumeric(T);
            const out = try self.clone();
            for (out.values) |*value| value.* = floorSparseValue(T, value.*);
            return out;
        }

        pub fn ceil(self: Self) SparseError!Self {
            ensureNumeric(T);
            const out = try self.clone();
            for (out.values) |*value| value.* = ceilSparseValue(T, value.*);
            return out;
        }

        pub fn round(self: Self) SparseError!Self {
            ensureNumeric(T);
            const out = try self.clone();
            for (out.values) |*value| value.* = roundSparseValue(T, value.*);
            return out;
        }

        pub fn trunc(self: Self) SparseError!Self {
            ensureNumeric(T);
            const out = try self.clone();
            for (out.values) |*value| value.* = truncSparseValue(T, value.*);
            return out;
        }

        pub fn clipMin(self: Self, min_value: T) SparseError!Self {
            ensureNumeric(T);
            const out = try self.clone();
            for (out.values) |*value| {
                if (value.* < min_value) value.* = min_value;
            }
            return out;
        }

        pub fn maximumScalar(self: Self, scalar: T) SparseError!Self {
            return self.clipMin(scalar);
        }

        pub fn maxScalar(self: Self, scalar: T) SparseError!Self {
            return self.maximumScalar(scalar);
        }

        pub fn clampMin(self: Self, min_value: T) SparseError!Self {
            return self.clipMin(min_value);
        }

        pub fn clipMax(self: Self, max_value: T) SparseError!Self {
            ensureNumeric(T);
            const out = try self.clone();
            for (out.values) |*value| {
                if (value.* > max_value) value.* = max_value;
            }
            return out;
        }

        pub fn minimumScalar(self: Self, scalar: T) SparseError!Self {
            return self.clipMax(scalar);
        }

        pub fn minScalar(self: Self, scalar: T) SparseError!Self {
            return self.minimumScalar(scalar);
        }

        pub fn clampMax(self: Self, max_value: T) SparseError!Self {
            return self.clipMax(max_value);
        }

        pub fn clip(self: Self, min_value: T, max_value: T) SparseError!Self {
            try validateSparseValueRange(T, min_value, max_value);
            const out = try self.clone();
            for (out.values) |*value| {
                if (value.* < min_value) {
                    value.* = min_value;
                } else if (value.* > max_value) {
                    value.* = max_value;
                }
            }
            return out;
        }

        pub fn clamp(self: Self, min_value: T, max_value: T) SparseError!Self {
            return self.clip(min_value, max_value);
        }

        pub fn sign(self: Self) SparseError!Self {
            ensureNumeric(T);
            const out = try self.clone();
            for (out.values) |*value| value.* = signSparseValue(T, value.*);
            return out;
        }

        pub fn signbit(self: Self) SparseError!CscMatrix(bool) {
            ensureNumeric(T);
            const col_offsets = try self.allocator.dupe(usize, self.col_offsets);
            errdefer self.allocator.free(col_offsets);
            const row_indices = try self.allocator.dupe(usize, self.row_indices);
            errdefer self.allocator.free(row_indices);
            var values = try self.allocator.alloc(bool, self.values.len);
            errdefer self.allocator.free(values);
            for (self.values, 0..) |value, index| {
                values[index] = signbitSparseValue(T, value);
            }
            return .{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = self.cols,
                .col_offsets = col_offsets,
                .row_indices = row_indices,
                .values = values,
            };
        }

        fn finitePredicateMask(self: Self, comptime predicate: SparseFinitePredicate) SparseError!CscMatrix(bool) {
            return self.mapValues(bool, struct {
                fn f(value: T) bool {
                    return sparseFinitePredicateValue(T, value, predicate);
                }
            }.f);
        }

        pub fn isNan(self: Self) SparseError!CscMatrix(bool) {
            return self.finitePredicateMask(.nan);
        }

        pub fn isnan(self: Self) SparseError!CscMatrix(bool) {
            return self.isNan();
        }

        pub fn isInf(self: Self) SparseError!CscMatrix(bool) {
            return self.finitePredicateMask(.inf);
        }

        pub fn isinf(self: Self) SparseError!CscMatrix(bool) {
            return self.isInf();
        }

        pub fn isPosInf(self: Self) SparseError!CscMatrix(bool) {
            return self.finitePredicateMask(.pos_inf);
        }

        pub fn isposinf(self: Self) SparseError!CscMatrix(bool) {
            return self.isPosInf();
        }

        pub fn isNegInf(self: Self) SparseError!CscMatrix(bool) {
            return self.finitePredicateMask(.neg_inf);
        }

        pub fn isneginf(self: Self) SparseError!CscMatrix(bool) {
            return self.isNegInf();
        }

        pub fn isFinite(self: Self) SparseError!CscMatrix(bool) {
            return self.finitePredicateMask(.finite);
        }

        pub fn isfinite(self: Self) SparseError!CscMatrix(bool) {
            return self.isFinite();
        }

        pub fn isNormal(self: Self) SparseError!CscMatrix(bool) {
            return self.finitePredicateMask(.normal);
        }

        pub fn isnormal(self: Self) SparseError!CscMatrix(bool) {
            return self.isNormal();
        }

        fn compareScalar(self: Self, scalar: T, comptime comparison: SparseScalarComparison) SparseError!CscMatrix(bool) {
            const col_offsets = try self.allocator.dupe(usize, self.col_offsets);
            errdefer self.allocator.free(col_offsets);
            const row_indices = try self.allocator.dupe(usize, self.row_indices);
            errdefer self.allocator.free(row_indices);
            const values = try sparseCompareScalarValues(T, self.allocator, self.values, scalar, comparison);
            errdefer self.allocator.free(values);
            return .{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = self.cols,
                .col_offsets = col_offsets,
                .row_indices = row_indices,
                .values = values,
            };
        }

        pub fn equalScalar(self: Self, scalar: T) SparseError!CscMatrix(bool) {
            return self.compareScalar(scalar, .eq);
        }

        pub fn eqScalar(self: Self, scalar: T) SparseError!CscMatrix(bool) {
            return self.equalScalar(scalar);
        }

        pub fn notEqualScalar(self: Self, scalar: T) SparseError!CscMatrix(bool) {
            return self.compareScalar(scalar, .ne);
        }

        pub fn neScalar(self: Self, scalar: T) SparseError!CscMatrix(bool) {
            return self.notEqualScalar(scalar);
        }

        pub fn greaterScalar(self: Self, scalar: T) SparseError!CscMatrix(bool) {
            return self.compareScalar(scalar, .gt);
        }

        pub fn gtScalar(self: Self, scalar: T) SparseError!CscMatrix(bool) {
            return self.greaterScalar(scalar);
        }

        pub fn greaterEqualScalar(self: Self, scalar: T) SparseError!CscMatrix(bool) {
            return self.compareScalar(scalar, .ge);
        }

        pub fn geScalar(self: Self, scalar: T) SparseError!CscMatrix(bool) {
            return self.greaterEqualScalar(scalar);
        }

        pub fn lessScalar(self: Self, scalar: T) SparseError!CscMatrix(bool) {
            return self.compareScalar(scalar, .lt);
        }

        pub fn ltScalar(self: Self, scalar: T) SparseError!CscMatrix(bool) {
            return self.lessScalar(scalar);
        }

        pub fn lessEqualScalar(self: Self, scalar: T) SparseError!CscMatrix(bool) {
            return self.compareScalar(scalar, .le);
        }

        pub fn leScalar(self: Self, scalar: T) SparseError!CscMatrix(bool) {
            return self.lessEqualScalar(scalar);
        }

        fn compareSameStructure(self: Self, rhs: Self, comptime comparison: SparseScalarComparison) SparseError!CscMatrix(bool) {
            if (self.rows != rhs.rows or self.cols != rhs.cols or self.values.len != rhs.values.len) return error.ShapeMismatch;
            if (!self.sameStructure(rhs)) return error.InvalidShape;
            const col_offsets = try self.allocator.dupe(usize, self.col_offsets);
            errdefer self.allocator.free(col_offsets);
            const row_indices = try self.allocator.dupe(usize, self.row_indices);
            errdefer self.allocator.free(row_indices);
            const values = try sparseCompareSameStructureValues(T, self.allocator, self.values, rhs.values, comparison);
            errdefer self.allocator.free(values);
            return .{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = self.cols,
                .col_offsets = col_offsets,
                .row_indices = row_indices,
                .values = values,
            };
        }

        pub fn equalSameStructure(self: Self, rhs: Self) SparseError!CscMatrix(bool) {
            return self.compareSameStructure(rhs, .eq);
        }

        pub fn eqSameStructure(self: Self, rhs: Self) SparseError!CscMatrix(bool) {
            return self.equalSameStructure(rhs);
        }

        pub fn notEqualSameStructure(self: Self, rhs: Self) SparseError!CscMatrix(bool) {
            return self.compareSameStructure(rhs, .ne);
        }

        pub fn neSameStructure(self: Self, rhs: Self) SparseError!CscMatrix(bool) {
            return self.notEqualSameStructure(rhs);
        }

        pub fn greaterSameStructure(self: Self, rhs: Self) SparseError!CscMatrix(bool) {
            return self.compareSameStructure(rhs, .gt);
        }

        pub fn gtSameStructure(self: Self, rhs: Self) SparseError!CscMatrix(bool) {
            return self.greaterSameStructure(rhs);
        }

        pub fn greaterEqualSameStructure(self: Self, rhs: Self) SparseError!CscMatrix(bool) {
            return self.compareSameStructure(rhs, .ge);
        }

        pub fn geSameStructure(self: Self, rhs: Self) SparseError!CscMatrix(bool) {
            return self.greaterEqualSameStructure(rhs);
        }

        pub fn lessSameStructure(self: Self, rhs: Self) SparseError!CscMatrix(bool) {
            return self.compareSameStructure(rhs, .lt);
        }

        pub fn ltSameStructure(self: Self, rhs: Self) SparseError!CscMatrix(bool) {
            return self.lessSameStructure(rhs);
        }

        pub fn lessEqualSameStructure(self: Self, rhs: Self) SparseError!CscMatrix(bool) {
            return self.compareSameStructure(rhs, .le);
        }

        pub fn leSameStructure(self: Self, rhs: Self) SparseError!CscMatrix(bool) {
            return self.lessEqualSameStructure(rhs);
        }

        pub fn nnz(self: Self) usize {
            return self.values.len;
        }

        pub fn countNonzero(self: Self) SparseError!usize {
            return sparseDenseCountNonzero(self);
        }

        pub fn countNonzeroAxis(self: Self, axis_opt: ?isize, keepdims: bool) SparseError!array_mod.Array(usize) {
            return sparseDenseCountNonzeroAxis(self, axis_opt, keepdims);
        }

        pub fn countNonzeroAxes(self: Self, axes: []const isize, keepdims: bool) SparseError!array_mod.Array(usize) {
            return sparseDenseCountNonzeroAxes(self, axes, keepdims);
        }

        pub fn countNonzeroDim(self: Self, dim_opt: ?isize, keepdim: bool) SparseError!array_mod.Array(usize) {
            return self.countNonzeroAxis(dim_opt, keepdim);
        }

        pub fn countNonzeroDims(self: Self, dims: []const isize, keepdim: bool) SparseError!array_mod.Array(usize) {
            return self.countNonzeroAxes(dims, keepdim);
        }

        pub fn flatNonzero(self: Self) SparseError!array_mod.Array(usize) {
            return sparseDenseFlatNonzero(self);
        }

        pub fn nonzero(self: Self) SparseError!array_mod.Array(usize) {
            return sparseDenseNonzero(self);
        }

        pub fn argwhere(self: Self) SparseError!array_mod.Array(usize) {
            return self.nonzero();
        }

        pub fn whereIndices(self: Self) SparseError!array_mod.Array(usize) {
            return self.nonzero();
        }

        pub fn all(self: Self) SparseError!bool {
            return sparseDenseAll(self);
        }

        pub fn any(self: Self) SparseError!bool {
            return sparseDenseAny(self);
        }

        pub fn allAxis(self: Self, axis_opt: ?isize, keepdims: bool) SparseError!array_mod.Array(bool) {
            return sparseDenseAllAxis(self, axis_opt, keepdims);
        }

        pub fn allAxes(self: Self, axes: []const isize, keepdims: bool) SparseError!array_mod.Array(bool) {
            return sparseDenseAllAxes(self, axes, keepdims);
        }

        pub fn allDim(self: Self, dim_opt: ?isize, keepdim: bool) SparseError!array_mod.Array(bool) {
            return self.allAxis(dim_opt, keepdim);
        }

        pub fn allDims(self: Self, dims: []const isize, keepdim: bool) SparseError!array_mod.Array(bool) {
            return self.allAxes(dims, keepdim);
        }

        pub fn anyAxis(self: Self, axis_opt: ?isize, keepdims: bool) SparseError!array_mod.Array(bool) {
            return sparseDenseAnyAxis(self, axis_opt, keepdims);
        }

        pub fn anyAxes(self: Self, axes: []const isize, keepdims: bool) SparseError!array_mod.Array(bool) {
            return sparseDenseAnyAxes(self, axes, keepdims);
        }

        pub fn anyDim(self: Self, dim_opt: ?isize, keepdim: bool) SparseError!array_mod.Array(bool) {
            return self.anyAxis(dim_opt, keepdim);
        }

        pub fn anyDims(self: Self, dims: []const isize, keepdim: bool) SparseError!array_mod.Array(bool) {
            return self.anyAxes(dims, keepdim);
        }

        pub fn logicalNot(self: Self) SparseError!array_mod.Array(bool) {
            return sparseDenseLogicalNot(self);
        }

        pub fn logicalAnd(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return sparseDenseLogicalBinary(self, rhs, .and_);
        }

        pub fn logicalAndScalar(self: Self, scalar: bool) SparseError!array_mod.Array(bool) {
            return sparseDenseLogicalScalar(self, scalar, .and_);
        }

        pub fn logicalOr(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return sparseDenseLogicalBinary(self, rhs, .or_);
        }

        pub fn logicalOrScalar(self: Self, scalar: bool) SparseError!array_mod.Array(bool) {
            return sparseDenseLogicalScalar(self, scalar, .or_);
        }

        pub fn logicalXor(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return sparseDenseLogicalBinary(self, rhs, .xor_);
        }

        pub fn logicalXorScalar(self: Self, scalar: bool) SparseError!array_mod.Array(bool) {
            return sparseDenseLogicalScalar(self, scalar, .xor_);
        }

        pub fn maskedSelect(self: Self, mask: array_mod.Array(bool)) SparseError!array_mod.Array(T) {
            return sparseDenseMaskedSelect(T, self, mask);
        }

        pub fn maskedFill(self: Self, mask: array_mod.Array(bool), value: T) SparseError!array_mod.Array(T) {
            return sparseDenseMaskedFill(T, self, mask, value);
        }

        pub fn maskedPutScalar(self: Self, mask: array_mod.Array(bool), value: T) SparseError!array_mod.Array(T) {
            return self.maskedFill(mask, value);
        }

        pub fn putMaskScalar(self: Self, mask: array_mod.Array(bool), value: T) SparseError!array_mod.Array(T) {
            return self.maskedPutScalar(mask, value);
        }

        pub fn maskedScatter(self: Self, mask: array_mod.Array(bool), src: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            return sparseDenseMaskedScatter(T, self, mask, src);
        }

        pub fn maskedPut(self: Self, mask: array_mod.Array(bool), values: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            return sparseDenseMaskedPut(T, self, mask, values);
        }

        pub fn putMask(self: Self, mask: array_mod.Array(bool), values: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            return self.maskedPut(mask, values);
        }

        pub fn take(self: Self, indices: array_mod.Array(usize), axis_opt: ?isize) SparseError!array_mod.Array(T) {
            return sparseDenseTake(T, self, indices, axis_opt);
        }

        pub fn takeSigned(self: Self, indices: array_mod.Array(isize), axis_opt: ?isize) SparseError!array_mod.Array(T) {
            return sparseDenseTakeSigned(T, self, indices, axis_opt);
        }

        pub fn takeMode(self: Self, indices: array_mod.Array(usize), axis_opt: ?isize, mode: array_mod.IndexMode) SparseError!array_mod.Array(T) {
            return sparseDenseTakeMode(T, self, indices, axis_opt, mode);
        }

        pub fn takeSignedMode(self: Self, indices: array_mod.Array(isize), axis_opt: ?isize, mode: array_mod.IndexMode) SparseError!array_mod.Array(T) {
            return sparseDenseTakeSignedMode(T, self, indices, axis_opt, mode);
        }

        pub fn indexSelect(self: Self, axis_index: isize, indices: array_mod.Array(usize)) SparseError!array_mod.Array(T) {
            return self.take(indices, axis_index);
        }

        pub fn indexSelectSigned(self: Self, axis_index: isize, indices: array_mod.Array(isize)) SparseError!array_mod.Array(T) {
            return self.takeSigned(indices, axis_index);
        }

        pub fn gather(self: Self, axis_index: isize, indices: array_mod.Array(usize)) SparseError!array_mod.Array(T) {
            return sparseDenseGather(T, self, axis_index, indices);
        }

        pub fn gatherSigned(self: Self, axis_index: isize, indices: array_mod.Array(isize)) SparseError!array_mod.Array(T) {
            return sparseDenseGatherSigned(T, self, axis_index, indices);
        }

        pub fn takeAlongAxis(self: Self, indices: array_mod.Array(usize), axis_index: isize) SparseError!array_mod.Array(T) {
            return self.gather(axis_index, indices);
        }

        pub fn takeAlongAxisSigned(self: Self, indices: array_mod.Array(isize), axis_index: isize) SparseError!array_mod.Array(T) {
            return self.gatherSigned(axis_index, indices);
        }

        pub fn scatter(self: Self, axis_index: isize, indices: array_mod.Array(usize), src: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            return sparseDenseScatter(T, self, axis_index, indices, src);
        }

        pub fn scatterScalar(self: Self, axis_index: isize, indices: array_mod.Array(usize), value: T) SparseError!array_mod.Array(T) {
            return sparseDenseScatterScalar(T, self, axis_index, indices, value);
        }

        pub fn putAlongAxis(self: Self, indices: array_mod.Array(usize), src: array_mod.Array(T), axis_index: isize) SparseError!array_mod.Array(T) {
            return self.scatter(axis_index, indices, src);
        }

        pub fn select(self: Self, axis_index: isize, index: usize) SparseError!array_mod.Array(T) {
            return sparseDenseSelect(T, self, axis_index, index);
        }

        pub fn selectSigned(self: Self, axis_index: isize, index: isize) SparseError!array_mod.Array(T) {
            return sparseDenseSelectSigned(T, self, axis_index, index);
        }

        pub fn narrow(self: Self, axis_index: isize, start: usize, length: usize) SparseError!array_mod.Array(T) {
            return sparseDenseNarrow(T, self, axis_index, start, length);
        }

        pub fn narrowSigned(self: Self, axis_index: isize, start: isize, length: usize) SparseError!array_mod.Array(T) {
            return sparseDenseNarrowSigned(T, self, axis_index, start, length);
        }

        pub fn scatterReduce(self: Self, axis_index: isize, indices: array_mod.Array(usize), src: array_mod.Array(T), reduction: array_mod.ScatterReduce) SparseError!array_mod.Array(T) {
            return sparseDenseScatterReduce(T, self, axis_index, indices, src, reduction);
        }

        pub fn scatterAdd(self: Self, axis_index: isize, indices: array_mod.Array(usize), src: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            return self.scatterReduce(axis_index, indices, src, .sum);
        }

        pub fn scatterReduceScalar(self: Self, axis_index: isize, indices: array_mod.Array(usize), value: T, reduction: array_mod.ScatterReduce) SparseError!array_mod.Array(T) {
            return sparseDenseScatterReduceScalar(T, self, axis_index, indices, value, reduction);
        }

        pub fn putFlat(self: Self, indices: array_mod.Array(usize), values: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            return sparseDensePutFlat(T, self, indices, values);
        }

        pub fn putFlatMode(self: Self, indices: array_mod.Array(usize), values: array_mod.Array(T), mode: array_mod.IndexMode) SparseError!array_mod.Array(T) {
            return sparseDensePutFlatMode(T, self, indices, values, mode);
        }

        pub fn putFlatScalar(self: Self, indices: array_mod.Array(usize), value: T) SparseError!array_mod.Array(T) {
            return sparseDensePutFlatScalar(T, self, indices, value);
        }

        pub fn putFlatScalarMode(self: Self, indices: array_mod.Array(usize), value: T, mode: array_mod.IndexMode) SparseError!array_mod.Array(T) {
            return sparseDensePutFlatScalarMode(T, self, indices, value, mode);
        }

        pub fn putFlatSigned(self: Self, indices: array_mod.Array(isize), values: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            return sparseDensePutFlatSigned(T, self, indices, values);
        }

        pub fn putFlatScalarSigned(self: Self, indices: array_mod.Array(isize), value: T) SparseError!array_mod.Array(T) {
            return sparseDensePutFlatScalarSigned(T, self, indices, value);
        }

        pub fn ravelCoords(self: Self, coords: array_mod.Array(usize)) SparseError!array_mod.Array(usize) {
            return sparseDenseRavelCoords(self, coords);
        }

        pub fn unravelFlat(self: Self, indices: array_mod.Array(usize)) SparseError!array_mod.Array(usize) {
            return sparseDenseUnravelFlat(self, indices);
        }

        pub fn takeCoords(self: Self, coords: array_mod.Array(usize)) SparseError!array_mod.Array(T) {
            return sparseDenseTakeCoords(T, self, coords);
        }

        pub fn putCoords(self: Self, coords: array_mod.Array(usize), values: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            return sparseDensePutCoords(T, self, coords, values);
        }

        pub fn putCoordsScalar(self: Self, coords: array_mod.Array(usize), value: T) SparseError!array_mod.Array(T) {
            return sparseDensePutCoordsScalar(T, self, coords, value);
        }

        pub fn reshape(self: Self, dims: []const usize) SparseError!array_mod.Array(T) {
            return sparseDenseReshape(T, self, dims);
        }

        pub fn reshapeInfer(self: Self, dims: []const isize) SparseError!array_mod.Array(T) {
            return sparseDenseReshapeInfer(T, self, dims);
        }

        pub fn reshapeAs(self: Self, other: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            return self.reshape(other.shape);
        }

        pub fn view(self: Self, dims: []const usize) SparseError!array_mod.Array(T) {
            return self.reshape(dims);
        }

        pub fn viewInfer(self: Self, dims: []const isize) SparseError!array_mod.Array(T) {
            return self.reshapeInfer(dims);
        }

        pub fn viewAs(self: Self, other: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            return self.view(other.shape);
        }

        pub fn flatten(self: Self) SparseError!array_mod.Array(T) {
            return sparseDenseFlatten(T, self);
        }

        pub fn flattenAxes(self: Self, start_axis: isize, end_axis: isize) SparseError!array_mod.Array(T) {
            return sparseDenseFlattenAxes(T, self, start_axis, end_axis);
        }

        pub fn flattenRange(self: Self, start_axis: isize, end_axis: isize) SparseError!array_mod.Array(T) {
            return self.flattenAxes(start_axis, end_axis);
        }

        pub fn flattenFrom(self: Self, start_axis: isize) SparseError!array_mod.Array(T) {
            return self.flattenAxes(start_axis, -1);
        }

        pub fn ravel(self: Self) SparseError!array_mod.Array(T) {
            return self.flatten();
        }

        pub fn atLeast1d(self: Self) SparseError!array_mod.Array(T) {
            return sparseDenseAtLeast1d(T, self);
        }

        pub fn atLeast2d(self: Self) SparseError!array_mod.Array(T) {
            return sparseDenseAtLeast2d(T, self);
        }

        pub fn atLeast3d(self: Self) SparseError!array_mod.Array(T) {
            return sparseDenseAtLeast3d(T, self);
        }

        pub fn unflatten(self: Self, axis_index: isize, dims: []const usize) SparseError!array_mod.Array(T) {
            return sparseDenseUnflatten(T, self, axis_index, dims);
        }

        pub fn squeeze(self: Self, axis_opt: ?isize) SparseError!array_mod.Array(T) {
            return sparseDenseSqueeze(T, self, axis_opt);
        }

        pub fn squeezeDim(self: Self, axis_index: isize) SparseError!array_mod.Array(T) {
            return self.squeeze(axis_index);
        }

        pub fn unsqueeze(self: Self, axis_index: isize) SparseError!array_mod.Array(T) {
            return sparseDenseUnsqueeze(T, self, axis_index);
        }

        pub fn unsqueezeDim(self: Self, axis_index: isize) SparseError!array_mod.Array(T) {
            return self.unsqueeze(axis_index);
        }

        pub fn squeezeAxes(self: Self, axes: []const isize) SparseError!array_mod.Array(T) {
            return sparseDenseSqueezeAxes(T, self, axes);
        }

        pub fn unsqueezeAxes(self: Self, axes: []const isize) SparseError!array_mod.Array(T) {
            return sparseDenseUnsqueezeAxes(T, self, axes);
        }

        pub fn expandDims(self: Self, axes: []const isize) SparseError!array_mod.Array(T) {
            return self.unsqueezeAxes(axes);
        }

        pub fn broadcastTo(self: Self, dims: []const usize) SparseError!array_mod.Array(T) {
            return sparseDenseBroadcastTo(T, self, dims);
        }

        pub fn expand(self: Self, dims: []const usize) SparseError!array_mod.Array(T) {
            return self.broadcastTo(dims);
        }

        pub fn expandAs(self: Self, other: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            return self.expand(other.shape);
        }

        pub fn broadcastAs(self: Self, other: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            return self.expandAs(other);
        }

        pub fn permute(self: Self, axes: []const usize) SparseError!array_mod.Array(T) {
            return sparseDensePermute(T, self, axes);
        }

        pub fn swapaxes(self: Self, dim0: isize, dim1: isize) SparseError!array_mod.Array(T) {
            return sparseDenseSwapAxes(T, self, dim0, dim1);
        }

        pub fn swapDims(self: Self, dim0: isize, dim1: isize) SparseError!array_mod.Array(T) {
            return self.swapaxes(dim0, dim1);
        }

        pub fn movedim(self: Self, source: isize, destination: isize) SparseError!array_mod.Array(T) {
            return sparseDenseMoveDim(T, self, source, destination);
        }

        pub fn moveaxis(self: Self, source: isize, destination: isize) SparseError!array_mod.Array(T) {
            return self.movedim(source, destination);
        }

        pub fn flip(self: Self, axis_index: isize) SparseError!array_mod.Array(T) {
            return sparseDenseFlip(T, self, axis_index);
        }

        pub fn flipud(self: Self) SparseError!array_mod.Array(T) {
            return self.flip(0);
        }

        pub fn fliplr(self: Self) SparseError!array_mod.Array(T) {
            return self.flip(1);
        }

        pub fn flipAxes(self: Self, axes: []const isize) SparseError!array_mod.Array(T) {
            return sparseDenseFlipAxes(T, self, axes);
        }

        pub fn roll(self: Self, shift: isize, axis_index: isize) SparseError!array_mod.Array(T) {
            return sparseDenseRoll(T, self, shift, axis_index);
        }

        pub fn rollFlat(self: Self, shift: isize) SparseError!array_mod.Array(T) {
            return sparseDenseRollFlat(T, self, shift);
        }

        pub fn rollAxes(self: Self, shifts: []const isize, axes: []const isize) SparseError!array_mod.Array(T) {
            return sparseDenseRollAxes(T, self, shifts, axes);
        }

        pub fn rot90(self: Self, k: isize, axes: [2]isize) SparseError!array_mod.Array(T) {
            return sparseDenseRot90(T, self, k, axes);
        }

        pub fn padConstant(self: Self, before: []const usize, after: []const usize, value: T) SparseError!array_mod.Array(T) {
            return sparseDensePadConstant(T, self, before, after, value);
        }

        pub fn padEdge(self: Self, before: []const usize, after: []const usize) SparseError!array_mod.Array(T) {
            return sparseDensePadEdge(T, self, before, after);
        }

        pub fn padReflect(self: Self, before: []const usize, after: []const usize) SparseError!array_mod.Array(T) {
            return sparseDensePadReflect(T, self, before, after);
        }

        pub fn padWrap(self: Self, before: []const usize, after: []const usize) SparseError!array_mod.Array(T) {
            return sparseDensePadWrap(T, self, before, after);
        }

        pub fn padSymmetric(self: Self, before: []const usize, after: []const usize) SparseError!array_mod.Array(T) {
            return sparseDensePadSymmetric(T, self, before, after);
        }

        pub fn split(self: Self, split_size: usize, axis_index: isize) SparseError!array_mod.Array(T).SplitResult {
            return sparseDenseSplit(T, self, split_size, axis_index);
        }

        pub fn splitWithSizes(self: Self, sizes: []const usize, axis_index: isize) SparseError!array_mod.Array(T).SplitResult {
            return sparseDenseSplitWithSizes(T, self, sizes, axis_index);
        }

        pub fn splitAtIndices(self: Self, indices: []const usize, axis_index: isize) SparseError!array_mod.Array(T).SplitResult {
            return sparseDenseSplitAtIndices(T, self, indices, axis_index);
        }

        pub fn chunk(self: Self, chunks: usize, axis_index: isize) SparseError!array_mod.Array(T).SplitResult {
            return sparseDenseChunk(T, self, chunks, axis_index);
        }

        pub fn unbind(self: Self, axis_index: isize) SparseError!array_mod.Array(T).SplitResult {
            return sparseDenseUnbind(T, self, axis_index);
        }

        pub fn repeat(self: Self, repeats: usize, axis_index: isize) SparseError!array_mod.Array(T) {
            return sparseDenseRepeat(T, self, repeats, axis_index);
        }

        pub fn repeatInterleave(self: Self, repeats: array_mod.Array(usize), axis_opt: ?isize) SparseError!array_mod.Array(T) {
            return sparseDenseRepeatInterleave(T, self, repeats, axis_opt);
        }

        pub fn repeatInterleaveScalar(self: Self, repeat_count: usize, axis_opt: ?isize) SparseError!array_mod.Array(T) {
            return sparseDenseRepeatInterleaveScalar(T, self, repeat_count, axis_opt);
        }

        pub fn tile(self: Self, repeats: []const usize) SparseError!array_mod.Array(T) {
            return sparseDenseTile(T, self, repeats);
        }

        pub fn indexPut(self: Self, indices: array_mod.Array(usize), values: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            return self.putFlat(indices, values);
        }

        pub fn indexPutScalar(self: Self, indices: array_mod.Array(usize), value: T) SparseError!array_mod.Array(T) {
            return self.putFlatScalar(indices, value);
        }

        pub fn compress(self: Self, condition: array_mod.Array(bool), axis_opt: ?isize) SparseError!array_mod.Array(T) {
            return sparseDenseCompress(T, self, condition, axis_opt);
        }

        pub fn where(self: Self, mask: array_mod.Array(bool), other: Self) SparseError!array_mod.Array(T) {
            return sparseDenseWhere(T, self, mask, other);
        }

        pub fn whereArray(self: Self, mask: array_mod.Array(bool), other: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            return sparseDenseWhereArray(T, self, mask, other);
        }

        pub fn whereScalar(self: Self, mask: array_mod.Array(bool), other_value: T) SparseError!array_mod.Array(T) {
            return sparseDenseWhereScalar(T, self, mask, other_value);
        }

        pub fn copyWhere(self: Self, mask: array_mod.Array(bool), src: Self) SparseError!array_mod.Array(T) {
            return sparseDenseWhere(T, src, mask, self);
        }

        pub fn asVeyraView(self: Self) SparseError!veyra.CscView(T) {
            return veyra.CscView(T).fromSlices(self.rows, self.cols, self.col_offsets, self.row_indices, self.values) catch return error.BackendFailure;
        }

        pub fn sameStructure(self: Self, rhs: Self) bool {
            return sparseSameStructure(self.rows, self.cols, self.col_offsets, self.row_indices, rhs.rows, rhs.cols, rhs.col_offsets, rhs.row_indices);
        }

        pub fn dotSameStructure(self: Self, rhs: Self) SparseError!T {
            if (self.rows != rhs.rows or self.cols != rhs.cols or self.values.len != rhs.values.len) return error.ShapeMismatch;
            if (!self.sameStructure(rhs)) return error.InvalidShape;
            return sparseDotSameStructure(T, self.values, rhs.values);
        }

        pub fn sameStructureDiffSummary(self: Self, rhs: Self) SparseError!SparseDiffSummary {
            if (self.rows != rhs.rows or self.cols != rhs.cols or self.values.len != rhs.values.len) return error.ShapeMismatch;
            if (!self.sameStructure(rhs)) return error.InvalidShape;
            return sparseSameStructureDiffSummary(T, self.values, rhs.values);
        }

        pub fn diffSummary(self: Self, rhs: Self) SparseError!SparseDiffSummary {
            if (self.rows != rhs.rows or self.cols != rhs.cols) return error.ShapeMismatch;
            var lhs_dense = try self.toDense();
            defer lhs_dense.deinit();
            var rhs_dense = try rhs.toDense();
            defer rhs_dense.deinit();
            return denseDiffSummary(T, lhs_dense.data, rhs_dense.data);
        }

        pub fn diffSummaryMeetsBounds(
            self: Self,
            rhs: Self,
            max_absolute_diff: f64,
            max_relative_diff: f64,
            max_squared_distance: f64,
            max_frobenius_distance: f64,
            max_relative_frobenius_distance: f64,
        ) SparseError!bool {
            if (self.rows != rhs.rows or self.cols != rhs.cols) return error.ShapeMismatch;
            var lhs_dense = try self.toDense();
            defer lhs_dense.deinit();
            var rhs_dense = try rhs.toDense();
            defer rhs_dense.deinit();
            return denseDiffSummaryMeetsBounds(
                T,
                lhs_dense.data,
                rhs_dense.data,
                max_absolute_diff,
                max_relative_diff,
                max_squared_distance,
                max_frobenius_distance,
                max_relative_frobenius_distance,
            );
        }

        pub fn maxAbsDiff(self: Self, rhs: Self) SparseError!f64 {
            return (try self.diffSummary(rhs)).max_abs_diff;
        }

        pub fn maxRelDiff(self: Self, rhs: Self) SparseError!f64 {
            return (try self.diffSummary(rhs)).max_rel_diff;
        }

        pub fn squaredDistance(self: Self, rhs: Self) SparseError!f64 {
            return (try self.diffSummary(rhs)).squared_distance;
        }

        pub fn frobeniusDistance(self: Self, rhs: Self) SparseError!f64 {
            return (try self.diffSummary(rhs)).frobeniusDistance();
        }

        pub fn relativeFrobeniusDistance(self: Self, rhs: Self) SparseError!f64 {
            return (try self.diffSummary(rhs)).relativeFrobeniusDistance();
        }

        pub fn maxAbsDiffMeetsBound(self: Self, rhs: Self, max_absolute_diff: f64) SparseError!bool {
            return (try self.diffSummary(rhs)).maxAbsDiffMeetsBound(max_absolute_diff);
        }

        pub fn maxRelDiffMeetsBound(self: Self, rhs: Self, max_relative_diff: f64) SparseError!bool {
            return (try self.diffSummary(rhs)).maxRelDiffMeetsBound(max_relative_diff);
        }

        pub fn squaredDistanceMeetsBound(self: Self, rhs: Self, max_squared_distance: f64) SparseError!bool {
            return (try self.diffSummary(rhs)).squaredDistanceMeetsBound(max_squared_distance);
        }

        pub fn frobeniusDistanceMeetsBound(self: Self, rhs: Self, max_distance: f64) SparseError!bool {
            return (try self.diffSummary(rhs)).frobeniusDistanceMeetsBound(max_distance);
        }

        pub fn relativeFrobeniusDistanceMeetsBound(self: Self, rhs: Self, max_relative_distance: f64) SparseError!bool {
            return (try self.diffSummary(rhs)).relativeFrobeniusDistanceMeetsBound(max_relative_distance);
        }

        fn compareDenseMaterialized(self: Self, rhs: Self, comptime comparison: SparseScalarComparison) SparseError!array_mod.Array(bool) {
            if (self.rows != rhs.rows or self.cols != rhs.cols) return error.ShapeMismatch;
            var lhs_dense = try self.toDense();
            defer lhs_dense.deinit();
            var rhs_dense = try rhs.toDense();
            defer rhs_dense.deinit();
            return sparseCompareDenseArrays(T, lhs_dense, rhs_dense, comparison);
        }

        fn compareDenseArray(self: Self, rhs: array_mod.Array(T), comptime comparison: SparseScalarComparison) SparseError!array_mod.Array(bool) {
            try validateDenseMatrixShape(self.rows, self.cols, rhs.shape);
            var lhs_dense = try self.toDense();
            defer lhs_dense.deinit();
            return sparseCompareDenseArrays(T, lhs_dense, rhs, comparison);
        }

        pub fn eq(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return self.compareDenseMaterialized(rhs, .eq);
        }

        pub fn equal(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return self.eq(rhs);
        }

        pub fn ne(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return self.compareDenseMaterialized(rhs, .ne);
        }

        pub fn notEqual(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return self.ne(rhs);
        }

        pub fn gt(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return self.compareDenseMaterialized(rhs, .gt);
        }

        pub fn greater(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return self.gt(rhs);
        }

        pub fn ge(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return self.compareDenseMaterialized(rhs, .ge);
        }

        pub fn greaterEqual(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return self.ge(rhs);
        }

        pub fn lt(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return self.compareDenseMaterialized(rhs, .lt);
        }

        pub fn less(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return self.lt(rhs);
        }

        pub fn le(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return self.compareDenseMaterialized(rhs, .le);
        }

        pub fn lessEqual(self: Self, rhs: Self) SparseError!array_mod.Array(bool) {
            return self.le(rhs);
        }

        pub fn eqDense(self: Self, rhs: array_mod.Array(T)) SparseError!array_mod.Array(bool) {
            return self.compareDenseArray(rhs, .eq);
        }

        pub fn equalDense(self: Self, rhs: array_mod.Array(T)) SparseError!array_mod.Array(bool) {
            return self.eqDense(rhs);
        }

        pub fn neDense(self: Self, rhs: array_mod.Array(T)) SparseError!array_mod.Array(bool) {
            return self.compareDenseArray(rhs, .ne);
        }

        pub fn notEqualDense(self: Self, rhs: array_mod.Array(T)) SparseError!array_mod.Array(bool) {
            return self.neDense(rhs);
        }

        pub fn gtDense(self: Self, rhs: array_mod.Array(T)) SparseError!array_mod.Array(bool) {
            return self.compareDenseArray(rhs, .gt);
        }

        pub fn greaterDense(self: Self, rhs: array_mod.Array(T)) SparseError!array_mod.Array(bool) {
            return self.gtDense(rhs);
        }

        pub fn geDense(self: Self, rhs: array_mod.Array(T)) SparseError!array_mod.Array(bool) {
            return self.compareDenseArray(rhs, .ge);
        }

        pub fn greaterEqualDense(self: Self, rhs: array_mod.Array(T)) SparseError!array_mod.Array(bool) {
            return self.geDense(rhs);
        }

        pub fn ltDense(self: Self, rhs: array_mod.Array(T)) SparseError!array_mod.Array(bool) {
            return self.compareDenseArray(rhs, .lt);
        }

        pub fn lessDense(self: Self, rhs: array_mod.Array(T)) SparseError!array_mod.Array(bool) {
            return self.ltDense(rhs);
        }

        pub fn leDense(self: Self, rhs: array_mod.Array(T)) SparseError!array_mod.Array(bool) {
            return self.compareDenseArray(rhs, .le);
        }

        pub fn lessEqualDense(self: Self, rhs: array_mod.Array(T)) SparseError!array_mod.Array(bool) {
            return self.leDense(rhs);
        }

        pub fn diffSummaryDense(self: Self, rhs: array_mod.Array(T)) SparseError!SparseDiffSummary {
            if (rhs.shape.len != 2) return error.NonMatrixArray;
            if (self.rows != rhs.shape[0] or self.cols != rhs.shape[1]) return error.ShapeMismatch;
            var lhs_dense = try self.toDense();
            defer lhs_dense.deinit();
            return denseDiffSummary(T, lhs_dense.data, rhs.data);
        }

        pub fn diffSummaryDenseMeetsBounds(
            self: Self,
            rhs: array_mod.Array(T),
            max_absolute_diff: f64,
            max_relative_diff: f64,
            max_squared_distance: f64,
            max_frobenius_distance: f64,
            max_relative_frobenius_distance: f64,
        ) SparseError!bool {
            if (rhs.shape.len != 2) return error.NonMatrixArray;
            if (self.rows != rhs.shape[0] or self.cols != rhs.shape[1]) return error.ShapeMismatch;
            var lhs_dense = try self.toDense();
            defer lhs_dense.deinit();
            return denseDiffSummaryMeetsBounds(
                T,
                lhs_dense.data,
                rhs.data,
                max_absolute_diff,
                max_relative_diff,
                max_squared_distance,
                max_frobenius_distance,
                max_relative_frobenius_distance,
            );
        }

        pub fn maxAbsDiffDense(self: Self, rhs: array_mod.Array(T)) SparseError!f64 {
            return (try self.diffSummaryDense(rhs)).max_abs_diff;
        }

        pub fn maxRelDiffDense(self: Self, rhs: array_mod.Array(T)) SparseError!f64 {
            return (try self.diffSummaryDense(rhs)).max_rel_diff;
        }

        pub fn squaredDistanceDense(self: Self, rhs: array_mod.Array(T)) SparseError!f64 {
            return (try self.diffSummaryDense(rhs)).squared_distance;
        }

        pub fn frobeniusDistanceDense(self: Self, rhs: array_mod.Array(T)) SparseError!f64 {
            return (try self.diffSummaryDense(rhs)).frobeniusDistance();
        }

        pub fn relativeFrobeniusDistanceDense(self: Self, rhs: array_mod.Array(T)) SparseError!f64 {
            return (try self.diffSummaryDense(rhs)).relativeFrobeniusDistance();
        }

        pub fn maxAbsDiffDenseMeetsBound(self: Self, rhs: array_mod.Array(T), max_absolute_diff: f64) SparseError!bool {
            return (try self.diffSummaryDense(rhs)).maxAbsDiffMeetsBound(max_absolute_diff);
        }

        pub fn maxRelDiffDenseMeetsBound(self: Self, rhs: array_mod.Array(T), max_relative_diff: f64) SparseError!bool {
            return (try self.diffSummaryDense(rhs)).maxRelDiffMeetsBound(max_relative_diff);
        }

        pub fn squaredDistanceDenseMeetsBound(self: Self, rhs: array_mod.Array(T), max_squared_distance: f64) SparseError!bool {
            return (try self.diffSummaryDense(rhs)).squaredDistanceMeetsBound(max_squared_distance);
        }

        pub fn frobeniusDistanceDenseMeetsBound(self: Self, rhs: array_mod.Array(T), max_distance: f64) SparseError!bool {
            return (try self.diffSummaryDense(rhs)).frobeniusDistanceMeetsBound(max_distance);
        }

        pub fn relativeFrobeniusDistanceDenseMeetsBound(self: Self, rhs: array_mod.Array(T), max_relative_distance: f64) SparseError!bool {
            return (try self.diffSummaryDense(rhs)).relativeFrobeniusDistanceMeetsBound(max_relative_distance);
        }

        pub fn iscloseDense(self: Self, rhs: array_mod.Array(T), rtol: T, atol: T) SparseError!array_mod.Array(bool) {
            return self.iscloseDenseEqualNan(rhs, rtol, atol, false);
        }

        pub fn isCloseDense(self: Self, rhs: array_mod.Array(T), rtol: T, atol: T) SparseError!array_mod.Array(bool) {
            return self.iscloseDense(rhs, rtol, atol);
        }

        pub fn iscloseDenseEqualNan(self: Self, rhs: array_mod.Array(T), rtol: T, atol: T, equal_nan: bool) SparseError!array_mod.Array(bool) {
            if (rhs.shape.len != 2) return error.NonMatrixArray;
            if (self.rows != rhs.shape[0] or self.cols != rhs.shape[1]) return error.ShapeMismatch;
            var lhs_dense = try self.toDense();
            defer lhs_dense.deinit();
            return lhs_dense.iscloseEqualNan(rhs, rtol, atol, equal_nan);
        }

        pub fn isCloseDenseEqualNan(self: Self, rhs: array_mod.Array(T), rtol: T, atol: T, equal_nan: bool) SparseError!array_mod.Array(bool) {
            return self.iscloseDenseEqualNan(rhs, rtol, atol, equal_nan);
        }

        pub fn allcloseDense(self: Self, rhs: array_mod.Array(T), rtol: T, atol: T) SparseError!bool {
            return self.allcloseDenseEqualNan(rhs, rtol, atol, false);
        }

        pub fn allCloseDense(self: Self, rhs: array_mod.Array(T), rtol: T, atol: T) SparseError!bool {
            return self.allcloseDense(rhs, rtol, atol);
        }

        pub fn allcloseDenseEqualNan(self: Self, rhs: array_mod.Array(T), rtol: T, atol: T, equal_nan: bool) SparseError!bool {
            if (rhs.shape.len != 2) return error.NonMatrixArray;
            if (self.rows != rhs.shape[0] or self.cols != rhs.shape[1]) return error.ShapeMismatch;
            var lhs_dense = try self.toDense();
            defer lhs_dense.deinit();
            return lhs_dense.allcloseEqualNan(rhs, rtol, atol, equal_nan);
        }

        pub fn allCloseDenseEqualNan(self: Self, rhs: array_mod.Array(T), rtol: T, atol: T, equal_nan: bool) SparseError!bool {
            return self.allcloseDenseEqualNan(rhs, rtol, atol, equal_nan);
        }

        pub fn iscloseScalar(self: Self, scalar: T, rtol: T, atol: T) SparseError!array_mod.Array(bool) {
            return self.iscloseScalarEqualNan(scalar, rtol, atol, false);
        }

        pub fn isCloseScalar(self: Self, scalar: T, rtol: T, atol: T) SparseError!array_mod.Array(bool) {
            return self.iscloseScalar(scalar, rtol, atol);
        }

        pub fn iscloseScalarEqualNan(self: Self, scalar: T, rtol: T, atol: T, equal_nan: bool) SparseError!array_mod.Array(bool) {
            var dense = try self.toDense();
            defer dense.deinit();
            return dense.iscloseScalarEqualNan(scalar, rtol, atol, equal_nan);
        }

        pub fn isCloseScalarEqualNan(self: Self, scalar: T, rtol: T, atol: T, equal_nan: bool) SparseError!array_mod.Array(bool) {
            return self.iscloseScalarEqualNan(scalar, rtol, atol, equal_nan);
        }

        pub fn allcloseScalar(self: Self, scalar: T, rtol: T, atol: T) SparseError!bool {
            return self.allcloseScalarEqualNan(scalar, rtol, atol, false);
        }

        pub fn allCloseScalar(self: Self, scalar: T, rtol: T, atol: T) SparseError!bool {
            return self.allcloseScalar(scalar, rtol, atol);
        }

        pub fn allcloseScalarEqualNan(self: Self, scalar: T, rtol: T, atol: T, equal_nan: bool) SparseError!bool {
            var dense = try self.toDense();
            defer dense.deinit();
            return dense.allcloseScalarEqualNan(scalar, rtol, atol, equal_nan);
        }

        pub fn allCloseScalarEqualNan(self: Self, scalar: T, rtol: T, atol: T, equal_nan: bool) SparseError!bool {
            return self.allcloseScalarEqualNan(scalar, rtol, atol, equal_nan);
        }

        pub fn isclose(self: Self, rhs: Self, rtol: T, atol: T) SparseError!array_mod.Array(bool) {
            return self.iscloseEqualNan(rhs, rtol, atol, false);
        }

        pub fn isClose(self: Self, rhs: Self, rtol: T, atol: T) SparseError!array_mod.Array(bool) {
            return self.isclose(rhs, rtol, atol);
        }

        pub fn iscloseEqualNan(self: Self, rhs: Self, rtol: T, atol: T, equal_nan: bool) SparseError!array_mod.Array(bool) {
            if (self.rows != rhs.rows or self.cols != rhs.cols) return error.ShapeMismatch;
            var lhs_dense = try self.toDense();
            defer lhs_dense.deinit();
            var rhs_dense = try rhs.toDense();
            defer rhs_dense.deinit();
            return lhs_dense.iscloseEqualNan(rhs_dense, rtol, atol, equal_nan);
        }

        pub fn isCloseEqualNan(self: Self, rhs: Self, rtol: T, atol: T, equal_nan: bool) SparseError!array_mod.Array(bool) {
            return self.iscloseEqualNan(rhs, rtol, atol, equal_nan);
        }

        pub fn allclose(self: Self, rhs: Self, rtol: T, atol: T) SparseError!bool {
            return self.allcloseEqualNan(rhs, rtol, atol, false);
        }

        pub fn allClose(self: Self, rhs: Self, rtol: T, atol: T) SparseError!bool {
            return self.allclose(rhs, rtol, atol);
        }

        pub fn allcloseEqualNan(self: Self, rhs: Self, rtol: T, atol: T, equal_nan: bool) SparseError!bool {
            if (self.rows != rhs.rows or self.cols != rhs.cols) return error.ShapeMismatch;
            var lhs_dense = try self.toDense();
            defer lhs_dense.deinit();
            var rhs_dense = try rhs.toDense();
            defer rhs_dense.deinit();
            return lhs_dense.allcloseEqualNan(rhs_dense, rtol, atol, equal_nan);
        }

        pub fn allCloseEqualNan(self: Self, rhs: Self, rtol: T, atol: T, equal_nan: bool) SparseError!bool {
            return self.allcloseEqualNan(rhs, rtol, atol, equal_nan);
        }

        pub fn maxAbsDiffSameStructure(self: Self, rhs: Self) SparseError!T {
            if (self.rows != rhs.rows or self.cols != rhs.cols or self.values.len != rhs.values.len) return error.ShapeMismatch;
            if (!self.sameStructure(rhs)) return error.InvalidShape;
            return sparseMaxAbsDiffSameStructure(T, self.values, rhs.values);
        }

        pub fn maxAbsDiffSameStructureMeetsBound(self: Self, rhs: Self, max_absolute_diff: T) SparseError!bool {
            if (self.rows != rhs.rows or self.cols != rhs.cols or self.values.len != rhs.values.len) return error.ShapeMismatch;
            if (!self.sameStructure(rhs)) return error.InvalidShape;
            return sparseMaxAbsDiffSameStructureMeetsBound(T, self.values, rhs.values, max_absolute_diff);
        }

        pub fn maxRelDiffSameStructure(self: Self, rhs: Self) SparseError!T {
            if (self.rows != rhs.rows or self.cols != rhs.cols or self.values.len != rhs.values.len) return error.ShapeMismatch;
            if (!self.sameStructure(rhs)) return error.InvalidShape;
            return sparseMaxRelDiffSameStructure(T, self.values, rhs.values);
        }

        pub fn maxRelDiffSameStructureMeetsBound(self: Self, rhs: Self, max_relative_diff: T) SparseError!bool {
            if (self.rows != rhs.rows or self.cols != rhs.cols or self.values.len != rhs.values.len) return error.ShapeMismatch;
            if (!self.sameStructure(rhs)) return error.InvalidShape;
            return sparseMaxRelDiffSameStructureMeetsBound(T, self.values, rhs.values, max_relative_diff);
        }

        pub fn squaredDistanceSameStructure(self: Self, rhs: Self) SparseError!T {
            if (self.rows != rhs.rows or self.cols != rhs.cols or self.values.len != rhs.values.len) return error.ShapeMismatch;
            if (!self.sameStructure(rhs)) return error.InvalidShape;
            return sparseSquaredDistanceSameStructure(T, self.values, rhs.values);
        }

        pub fn squaredDistanceSameStructureMeetsBound(self: Self, rhs: Self, max_squared_distance: T) SparseError!bool {
            if (self.rows != rhs.rows or self.cols != rhs.cols or self.values.len != rhs.values.len) return error.ShapeMismatch;
            if (!self.sameStructure(rhs)) return error.InvalidShape;
            return sparseSquaredDistanceSameStructureMeetsBound(T, self.values, rhs.values, max_squared_distance);
        }

        pub fn frobeniusDistanceSameStructure(self: Self, rhs: Self) SparseError!T {
            if (self.rows != rhs.rows or self.cols != rhs.cols or self.values.len != rhs.values.len) return error.ShapeMismatch;
            if (!self.sameStructure(rhs)) return error.InvalidShape;
            return sparseFrobeniusDistanceSameStructure(T, self.values, rhs.values);
        }

        pub fn frobeniusDistanceSameStructureMeetsBound(self: Self, rhs: Self, max_distance: T) SparseError!bool {
            if (self.rows != rhs.rows or self.cols != rhs.cols or self.values.len != rhs.values.len) return error.ShapeMismatch;
            if (!self.sameStructure(rhs)) return error.InvalidShape;
            return sparseFrobeniusDistanceSameStructureMeetsBound(T, self.values, rhs.values, max_distance);
        }

        pub fn relativeFrobeniusDistanceSameStructure(self: Self, rhs: Self) SparseError!T {
            if (self.rows != rhs.rows or self.cols != rhs.cols or self.values.len != rhs.values.len) return error.ShapeMismatch;
            if (!self.sameStructure(rhs)) return error.InvalidShape;
            return sparseRelativeFrobeniusDistanceSameStructure(T, self.values, rhs.values);
        }

        pub fn relativeFrobeniusDistanceSameStructureMeetsBound(self: Self, rhs: Self, max_relative_distance: T) SparseError!bool {
            if (self.rows != rhs.rows or self.cols != rhs.cols or self.values.len != rhs.values.len) return error.ShapeMismatch;
            if (!self.sameStructure(rhs)) return error.InvalidShape;
            return sparseRelativeFrobeniusDistanceSameStructureMeetsBound(T, self.values, rhs.values, max_relative_distance);
        }

        pub fn toDense(self: Self) SparseError!array_mod.Array(T) {
            var out = try array_mod.Array(T).zeros(self.allocator, &.{ self.rows, self.cols });
            errdefer out.deinit();
            for (0..self.cols) |c| {
                for (self.col_offsets[c]..self.col_offsets[c + 1]) |pos| {
                    const index = self.row_indices[pos] * self.cols + c;
                    out.data[index] = addSparseValue(T, out.data[index], self.values[pos]);
                }
            }
            return out;
        }

        pub fn toCoo(self: Self) SparseError!CooMatrix(T) {
            var row_indices = try self.allocator.alloc(usize, self.values.len);
            errdefer self.allocator.free(row_indices);
            var col_indices = try self.allocator.alloc(usize, self.values.len);
            errdefer self.allocator.free(col_indices);
            const values = try self.allocator.dupe(T, self.values);
            errdefer self.allocator.free(values);

            for (0..self.cols) |col| {
                for (self.col_offsets[col]..self.col_offsets[col + 1]) |pos| {
                    row_indices[pos] = self.row_indices[pos];
                    col_indices[pos] = col;
                }
            }

            return .{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = self.cols,
                .row_indices = row_indices,
                .col_indices = col_indices,
                .values = values,
            };
        }

        pub fn coalesced(self: Self) SparseError!Self {
            // Keep compressed-format canonicalization semantically identical to
            // COO coalescing while still returning CSC ownership to callers.
            var coo = try self.toCoo();
            defer coo.deinit();
            var canonical = try coo.coalesced();
            defer canonical.deinit();
            return canonical.toCsc();
        }

        pub fn add(self: Self, rhs: Self) SparseError!Self {
            if (self.rows != rhs.rows or self.cols != rhs.cols) return error.ShapeMismatch;
            var lhs_coo = try self.toCoo();
            defer lhs_coo.deinit();
            var rhs_coo = try rhs.toCoo();
            defer rhs_coo.deinit();
            var sum_coo = try lhs_coo.add(rhs_coo);
            defer sum_coo.deinit();
            return sum_coo.toCsc();
        }

        pub fn dropZerosNnz(self: Self) usize {
            return sparseDropZerosNnz(T, self.values);
        }

        pub fn dropZeros(self: Self) SparseError!Self {
            const nonzero_count = self.dropZerosNnz();

            var col_offsets = try self.allocator.alloc(usize, self.cols + 1);
            errdefer self.allocator.free(col_offsets);
            var row_indices = try self.allocator.alloc(usize, nonzero_count);
            errdefer self.allocator.free(row_indices);
            var values = try self.allocator.alloc(T, nonzero_count);
            errdefer self.allocator.free(values);

            var write: usize = 0;
            col_offsets[0] = 0;
            for (0..self.cols) |col| {
                for (self.col_offsets[col]..self.col_offsets[col + 1]) |pos| {
                    const value = self.values[pos];
                    if (isNonZero(T, value)) {
                        row_indices[write] = self.row_indices[pos];
                        values[write] = value;
                        write += 1;
                    }
                }
                col_offsets[col + 1] = write;
            }
            std.debug.assert(write == nonzero_count);

            return .{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = self.cols,
                .col_offsets = col_offsets,
                .row_indices = row_indices,
                .values = values,
            };
        }

        pub fn pruneZerosNnz(self: Self, tolerance: T) SparseError!usize {
            return sparsePruneZerosNnz(T, self.values, tolerance);
        }

        pub fn pruneZeros(self: Self, tolerance: T) SparseError!Self {
            const keep_count = try self.pruneZerosNnz(tolerance);

            var col_offsets = try self.allocator.alloc(usize, self.cols + 1);
            errdefer self.allocator.free(col_offsets);
            var row_indices = try self.allocator.alloc(usize, keep_count);
            errdefer self.allocator.free(row_indices);
            var values = try self.allocator.alloc(T, keep_count);
            errdefer self.allocator.free(values);

            var write: usize = 0;
            col_offsets[0] = 0;
            for (0..self.cols) |col| {
                for (self.col_offsets[col]..self.col_offsets[col + 1]) |pos| {
                    const value = self.values[pos];
                    if (sparseAbsValueExceedsTolerance(T, value, tolerance)) {
                        row_indices[write] = self.row_indices[pos];
                        values[write] = value;
                        write += 1;
                    }
                }
                col_offsets[col + 1] = write;
            }
            std.debug.assert(write == keep_count);
            return .{ .allocator = self.allocator, .rows = self.rows, .cols = self.cols, .col_offsets = col_offsets, .row_indices = row_indices, .values = values };
        }

        pub fn scale(self: Self, alpha: T) SparseError!Self {
            ensureNumeric(T);
            const col_offsets = try self.allocator.dupe(usize, self.col_offsets);
            errdefer self.allocator.free(col_offsets);
            const row_indices = try self.allocator.dupe(usize, self.row_indices);
            errdefer self.allocator.free(row_indices);
            var values = try self.allocator.alloc(T, self.values.len);
            errdefer self.allocator.free(values);

            // Preserve CSC column structure; zero pruning remains an explicit
            // opt-in so callers can keep structural zeros when they carry
            // semantic meaning.
            for (self.values, 0..) |value, i| values[i] = value * alpha;

            return .{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = self.cols,
                .col_offsets = col_offsets,
                .row_indices = row_indices,
                .values = values,
            };
        }

        pub fn scaleRows(self: Self, row_scale: []const T) SparseError!Self {
            ensureNumeric(T);
            if (row_scale.len != self.rows) return error.ShapeMismatch;
            const col_offsets = try self.allocator.dupe(usize, self.col_offsets);
            errdefer self.allocator.free(col_offsets);
            const row_indices = try self.allocator.dupe(usize, self.row_indices);
            errdefer self.allocator.free(row_indices);
            var values = try self.allocator.alloc(T, self.values.len);
            errdefer self.allocator.free(values);

            for (self.values, 0..) |value, pos| values[pos] = row_scale[self.row_indices[pos]] * value;

            return .{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = self.cols,
                .col_offsets = col_offsets,
                .row_indices = row_indices,
                .values = values,
            };
        }

        pub fn scaleRowsInPlace(self: *Self, row_scale: []const T) SparseError!void {
            ensureNumeric(T);
            if (row_scale.len != self.rows) return error.ShapeMismatch;
            for (self.values, self.row_indices) |*value, row| value.* *= row_scale[row];
        }

        pub fn scaleColumns(self: Self, col_scale: []const T) SparseError!Self {
            ensureNumeric(T);
            if (col_scale.len != self.cols) return error.ShapeMismatch;
            const col_offsets = try self.allocator.dupe(usize, self.col_offsets);
            errdefer self.allocator.free(col_offsets);
            const row_indices = try self.allocator.dupe(usize, self.row_indices);
            errdefer self.allocator.free(row_indices);
            var values = try self.allocator.alloc(T, self.values.len);
            errdefer self.allocator.free(values);

            for (0..self.cols) |col| {
                for (self.col_offsets[col]..self.col_offsets[col + 1]) |pos| values[pos] = self.values[pos] * col_scale[col];
            }

            return .{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = self.cols,
                .col_offsets = col_offsets,
                .row_indices = row_indices,
                .values = values,
            };
        }

        pub fn scaleColumnsInPlace(self: *Self, col_scale: []const T) SparseError!void {
            ensureNumeric(T);
            if (col_scale.len != self.cols) return error.ShapeMismatch;
            for (0..self.cols) |col| {
                for (self.col_offsets[col]..self.col_offsets[col + 1]) |pos| self.values[pos] *= col_scale[col];
            }
        }

        pub fn scaleRowsAndColumns(self: Self, row_scale: []const T, col_scale: []const T) SparseError!Self {
            ensureNumeric(T);
            if (row_scale.len != self.rows or col_scale.len != self.cols) return error.ShapeMismatch;
            const col_offsets = try self.allocator.dupe(usize, self.col_offsets);
            errdefer self.allocator.free(col_offsets);
            const row_indices = try self.allocator.dupe(usize, self.row_indices);
            errdefer self.allocator.free(row_indices);
            var values = try self.allocator.alloc(T, self.values.len);
            errdefer self.allocator.free(values);

            for (0..self.cols) |col| {
                for (self.col_offsets[col]..self.col_offsets[col + 1]) |pos| values[pos] = row_scale[self.row_indices[pos]] * self.values[pos] * col_scale[col];
            }

            return .{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = self.cols,
                .col_offsets = col_offsets,
                .row_indices = row_indices,
                .values = values,
            };
        }

        pub fn scaleRowsAndColumnsInPlace(self: *Self, row_scale: []const T, col_scale: []const T) SparseError!void {
            ensureNumeric(T);
            if (row_scale.len != self.rows or col_scale.len != self.cols) return error.ShapeMismatch;
            for (0..self.cols) |col| {
                for (self.col_offsets[col]..self.col_offsets[col + 1]) |pos| self.values[pos] *= row_scale[self.row_indices[pos]] * col_scale[col];
            }
        }

        pub fn neg(self: Self) SparseError!Self {
            ensureNumeric(T);
            return self.scale(negSparseValue(T, oneValue(T)));
        }

        pub fn negative(self: Self) SparseError!Self {
            return self.neg();
        }

        pub fn sub(self: Self, rhs: Self) SparseError!Self {
            if (self.rows != rhs.rows or self.cols != rhs.cols) return error.ShapeMismatch;
            var neg_rhs = try rhs.neg();
            defer neg_rhs.deinit();
            return self.add(neg_rhs);
        }

        pub fn hadamard(self: Self, rhs: Self) SparseError!Self {
            if (self.rows != rhs.rows or self.cols != rhs.cols) return error.ShapeMismatch;
            var lhs_coo = try self.toCoo();
            defer lhs_coo.deinit();
            var rhs_coo = try rhs.toCoo();
            defer rhs_coo.deinit();
            var product_coo = try lhs_coo.hadamard(rhs_coo);
            defer product_coo.deinit();
            return product_coo.toCsc();
        }

        pub fn mul(self: Self, rhs: Self) SparseError!Self {
            return self.hadamard(rhs);
        }

        pub fn multiply(self: Self, rhs: Self) SparseError!Self {
            return self.hadamard(rhs);
        }

        pub fn transpose(self: Self) SparseError!Self {
            var counts = try self.allocator.alloc(usize, self.rows);
            defer self.allocator.free(counts);
            @memset(counts, 0);
            for (self.row_indices) |row| counts[row] += 1;

            var col_offsets = try self.allocator.alloc(usize, self.rows + 1);
            errdefer self.allocator.free(col_offsets);
            col_offsets[0] = 0;
            for (counts, 0..) |count, i| col_offsets[i + 1] = col_offsets[i] + count;

            var next = try self.allocator.dupe(usize, col_offsets[0..self.rows]);
            defer self.allocator.free(next);
            var row_indices = try self.allocator.alloc(usize, self.values.len);
            errdefer self.allocator.free(row_indices);
            var values = try self.allocator.alloc(T, self.values.len);
            errdefer self.allocator.free(values);

            for (0..self.cols) |col| {
                for (self.col_offsets[col]..self.col_offsets[col + 1]) |pos| {
                    const row = self.row_indices[pos];
                    const dst_pos = next[row];
                    next[row] += 1;
                    row_indices[dst_pos] = col;
                    values[dst_pos] = self.values[pos];
                }
            }

            return .{
                .allocator = self.allocator,
                .rows = self.cols,
                .cols = self.rows,
                .col_offsets = col_offsets,
                .row_indices = row_indices,
                .values = values,
            };
        }

        pub fn matrixTranspose(self: Self) SparseError!Self {
            return self.transpose();
        }

        pub fn mT(self: Self) SparseError!Self {
            return self.matrixTranspose();
        }

        pub fn toCsr(self: Self) SparseError!CsrMatrix(T) {
            var row_offsets = try self.allocator.alloc(usize, self.rows + 1);
            errdefer self.allocator.free(row_offsets);
            @memset(row_offsets, 0);
            for (self.row_indices) |row| row_offsets[row + 1] += 1;
            for (1..row_offsets.len) |i| row_offsets[i] += row_offsets[i - 1];

            var col_indices = try self.allocator.alloc(usize, self.values.len);
            errdefer self.allocator.free(col_indices);
            var values = try self.allocator.alloc(T, self.values.len);
            errdefer self.allocator.free(values);
            const next = try self.allocator.dupe(usize, row_offsets[0..self.rows]);
            defer self.allocator.free(next);

            // Fill by row using a mutable copy of the offsets.  Walking CSC
            // columns in ascending order keeps each CSR row's column indices
            // sorted without a separate sort pass.
            for (0..self.cols) |col| {
                for (self.col_offsets[col]..self.col_offsets[col + 1]) |pos| {
                    const row = self.row_indices[pos];
                    const dst = next[row];
                    col_indices[dst] = col;
                    values[dst] = self.values[pos];
                    next[row] += 1;
                }
            }

            return .{
                .allocator = self.allocator,
                .rows = self.rows,
                .cols = self.cols,
                .row_offsets = row_offsets,
                .col_indices = col_indices,
                .values = values,
            };
        }

        pub fn matvec(self: Self, x: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            if (x.shape.len != 1) return error.NonVectorArray;
            if (x.shape[0] != self.cols) return error.ShapeMismatch;
            if (comptime T == f64) return self.matvecF64(@as(array_mod.Array(f64), x));
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.rows});
            errdefer out.deinit();
            for (0..self.cols) |c| {
                for (self.col_offsets[c]..self.col_offsets[c + 1]) |pos| out.data[self.row_indices[pos]] += self.values[pos] * x.data[c];
            }
            return out;
        }

        fn matvecF64(self: Self, x: array_mod.Array(f64)) SparseError!array_mod.Array(f64) {
            const veyra_view = try @as(CscMatrix(f64), self).asVeyraView();
            var rhs = veyra.Vector(f64).fromSlice(self.allocator, x.data) catch return error.BackendFailure;
            defer rhs.deinit();
            var dst = veyra.Vector(f64).zeros(self.allocator, self.rows) catch return error.BackendFailure;
            defer dst.deinit();
            veyra.cscMatvec(f64, veyra_view, rhs.asView(), dst.asMut()) catch return error.BackendFailure;
            return array_mod.Array(f64).fromSlice(self.allocator, dst.data, &.{self.rows});
        }

        pub fn matvecResidualNorm(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T)) SparseError!T {
            ensureFloat(T);
            if (rhs.shape.len != 1) return error.NonVectorArray;
            var predicted = try self.matvec(x);
            defer predicted.deinit();
            return sparseVectorResidualNorm(T, predicted.data, rhs.data);
        }

        pub fn matvecResidualNormMeetsBound(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T), max_residual: T) SparseError!bool {
            try validateSparseValueRange(T, zero(T), max_residual);
            return (try self.matvecResidualNorm(x, rhs)) <= max_residual;
        }

        pub fn matvecRelativeResidualNorm(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T)) SparseError!T {
            const residual = try self.matvecResidualNorm(x, rhs);
            return sparseRelativeResidualNorm(T, residual, self.frobeniusNorm(), x.data, rhs.data);
        }

        pub fn matvecResidualSummary(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T)) SparseError!SparseResidualSummary {
            const residual = try self.matvecResidualNorm(x, rhs);
            return sparseResidualSummary(T, residual, self.frobeniusNorm(), x.data, rhs.data);
        }

        pub fn matvecResidualSummaryMeetsBounds(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T), max_residual: f64, max_relative_residual: f64) SparseError!bool {
            const summary = try self.matvecResidualSummary(x, rhs);
            return summary.meetsBounds(max_residual, max_relative_residual);
        }

        pub fn matvecRelativeResidualNormMeetsBound(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T), max_relative_residual: T) SparseError!bool {
            try validateSparseValueRange(T, zero(T), max_relative_residual);
            return (try self.matvecRelativeResidualNorm(x, rhs)) <= max_relative_residual;
        }

        pub fn matmat(self: Self, rhs: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            if (rhs.shape.len != 2) return error.NonMatrixArray;
            if (rhs.shape[0] != self.cols) return error.ShapeMismatch;
            if (comptime T == f64) return self.matmatF64(@as(array_mod.Array(f64), rhs));
            var out = try array_mod.Array(T).zeros(self.allocator, &.{ self.rows, rhs.shape[1] });
            errdefer out.deinit();
            for (0..self.cols) |c| {
                for (self.col_offsets[c]..self.col_offsets[c + 1]) |pos| {
                    const row = self.row_indices[pos];
                    const value = self.values[pos];
                    for (0..rhs.shape[1]) |rhs_col| out.data[row * rhs.shape[1] + rhs_col] += value * rhs.data[c * rhs.shape[1] + rhs_col];
                }
            }
            return out;
        }

        fn matmatF64(self: Self, rhs: array_mod.Array(f64)) SparseError!array_mod.Array(f64) {
            const veyra_view = try @as(CscMatrix(f64), self).asVeyraView();
            var rhs_matrix = veyra.Matrix(f64).fromSlice(self.allocator, rhs.shape[0], rhs.shape[1], .row_major, rhs.data) catch return error.BackendFailure;
            defer rhs_matrix.deinit();
            var dst = veyra.Matrix(f64).zeros(self.allocator, self.rows, rhs.shape[1], .row_major) catch return error.BackendFailure;
            defer dst.deinit();
            veyra.cscMatmat(f64, veyra_view, rhs_matrix.asView(), dst.asMut()) catch return error.BackendFailure;
            return array_mod.Array(f64).fromSlice(self.allocator, dst.data, &.{ self.rows, rhs.shape[1] });
        }

        pub fn matmatResidualFrobeniusNorm(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T)) SparseError!T {
            ensureFloat(T);
            var predicted = try self.matmat(x);
            defer predicted.deinit();
            return sparseMatrixResidualNorm(T, predicted, rhs);
        }

        pub fn matmatResidualFrobeniusNormMeetsBound(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T), max_residual: T) SparseError!bool {
            try validateSparseValueRange(T, zero(T), max_residual);
            return (try self.matmatResidualFrobeniusNorm(x, rhs)) <= max_residual;
        }

        pub fn matmatRelativeResidualFrobeniusNorm(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T)) SparseError!T {
            const residual = try self.matmatResidualFrobeniusNorm(x, rhs);
            return sparseRelativeMatrixResidualNorm(T, residual, self.frobeniusNorm(), x.data, rhs.data);
        }

        pub fn matmatResidualSummary(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T)) SparseError!SparseResidualSummary {
            const residual = try self.matmatResidualFrobeniusNorm(x, rhs);
            return sparseResidualSummary(T, residual, self.frobeniusNorm(), x.data, rhs.data);
        }

        pub fn matmatResidualSummaryMeetsBounds(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T), max_residual: f64, max_relative_residual: f64) SparseError!bool {
            const summary = try self.matmatResidualSummary(x, rhs);
            return summary.meetsBounds(max_residual, max_relative_residual);
        }

        pub fn matmatRelativeResidualFrobeniusNormMeetsBound(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T), max_relative_residual: T) SparseError!bool {
            try validateSparseValueRange(T, zero(T), max_relative_residual);
            return (try self.matmatRelativeResidualFrobeniusNorm(x, rhs)) <= max_relative_residual;
        }

        pub fn matmulSparse(self: Self, rhs: Self) SparseError!Self {
            if (self.cols != rhs.rows) return error.ShapeMismatch;
            var lhs_csr = try self.toCsr();
            defer lhs_csr.deinit();
            var rhs_csr = try rhs.toCsr();
            defer rhs_csr.deinit();
            var product = try lhs_csr.matmulSparse(rhs_csr);
            defer product.deinit();
            return product.toCsc();
        }

        pub fn transposeMatvec(self: Self, x: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            if (x.shape.len != 1) return error.NonVectorArray;
            if (x.shape[0] != self.rows) return error.ShapeMismatch;
            if (comptime T == f64) return self.transposeMatvecF64(@as(array_mod.Array(f64), x));
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.cols});
            errdefer out.deinit();
            for (0..self.cols) |c| {
                var acc = zero(T);
                for (self.col_offsets[c]..self.col_offsets[c + 1]) |pos| acc += self.values[pos] * x.data[self.row_indices[pos]];
                out.data[c] = acc;
            }
            return out;
        }

        fn transposeMatvecF64(self: Self, x: array_mod.Array(f64)) SparseError!array_mod.Array(f64) {
            const veyra_view = try @as(CscMatrix(f64), self).asVeyraView();
            var rhs = veyra.Vector(f64).fromSlice(self.allocator, x.data) catch return error.BackendFailure;
            defer rhs.deinit();
            var dst = veyra.Vector(f64).zeros(self.allocator, self.cols) catch return error.BackendFailure;
            defer dst.deinit();
            veyra.cscTransposeMatvec(f64, veyra_view, rhs.asView(), dst.asMut()) catch return error.BackendFailure;
            return array_mod.Array(f64).fromSlice(self.allocator, dst.data, &.{self.cols});
        }

        pub fn transposeMatvecResidualNorm(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T)) SparseError!T {
            ensureFloat(T);
            if (rhs.shape.len != 1) return error.NonVectorArray;
            var predicted = try self.transposeMatvec(x);
            defer predicted.deinit();
            return sparseVectorResidualNorm(T, predicted.data, rhs.data);
        }

        pub fn transposeMatvecResidualNormMeetsBound(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T), max_residual: T) SparseError!bool {
            try validateSparseValueRange(T, zero(T), max_residual);
            return (try self.transposeMatvecResidualNorm(x, rhs)) <= max_residual;
        }

        pub fn transposeMatvecRelativeResidualNorm(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T)) SparseError!T {
            const residual = try self.transposeMatvecResidualNorm(x, rhs);
            return sparseRelativeResidualNorm(T, residual, self.frobeniusNorm(), x.data, rhs.data);
        }

        pub fn transposeMatvecResidualSummary(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T)) SparseError!SparseResidualSummary {
            const residual = try self.transposeMatvecResidualNorm(x, rhs);
            return sparseResidualSummary(T, residual, self.frobeniusNorm(), x.data, rhs.data);
        }

        pub fn transposeMatvecResidualSummaryMeetsBounds(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T), max_residual: f64, max_relative_residual: f64) SparseError!bool {
            const summary = try self.transposeMatvecResidualSummary(x, rhs);
            return summary.meetsBounds(max_residual, max_relative_residual);
        }

        pub fn transposeMatvecRelativeResidualNormMeetsBound(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T), max_relative_residual: T) SparseError!bool {
            try validateSparseValueRange(T, zero(T), max_relative_residual);
            return (try self.transposeMatvecRelativeResidualNorm(x, rhs)) <= max_relative_residual;
        }

        pub fn transposeMatmat(self: Self, rhs: array_mod.Array(T)) SparseError!array_mod.Array(T) {
            if (rhs.shape.len != 2) return error.NonMatrixArray;
            if (rhs.shape[0] != self.rows) return error.ShapeMismatch;
            if (comptime T == f64) return self.transposeMatmatF64(@as(array_mod.Array(f64), rhs));
            var out = try array_mod.Array(T).zeros(self.allocator, &.{ self.cols, rhs.shape[1] });
            errdefer out.deinit();
            for (0..self.cols) |c| {
                for (0..rhs.shape[1]) |rhs_col| {
                    var acc = zero(T);
                    for (self.col_offsets[c]..self.col_offsets[c + 1]) |pos| acc += self.values[pos] * rhs.data[self.row_indices[pos] * rhs.shape[1] + rhs_col];
                    out.data[c * rhs.shape[1] + rhs_col] = acc;
                }
            }
            return out;
        }

        fn transposeMatmatF64(self: Self, rhs: array_mod.Array(f64)) SparseError!array_mod.Array(f64) {
            const veyra_view = try @as(CscMatrix(f64), self).asVeyraView();
            var rhs_matrix = veyra.Matrix(f64).fromSlice(self.allocator, rhs.shape[0], rhs.shape[1], .row_major, rhs.data) catch return error.BackendFailure;
            defer rhs_matrix.deinit();
            var dst = veyra.Matrix(f64).zeros(self.allocator, self.cols, rhs.shape[1], .row_major) catch return error.BackendFailure;
            defer dst.deinit();
            veyra.cscTransposeMatmat(f64, veyra_view, rhs_matrix.asView(), dst.asMut()) catch return error.BackendFailure;
            return array_mod.Array(f64).fromSlice(self.allocator, dst.data, &.{ self.cols, rhs.shape[1] });
        }

        pub fn transposeMatmatResidualFrobeniusNorm(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T)) SparseError!T {
            ensureFloat(T);
            var predicted = try self.transposeMatmat(x);
            defer predicted.deinit();
            return sparseMatrixResidualNorm(T, predicted, rhs);
        }

        pub fn transposeMatmatResidualFrobeniusNormMeetsBound(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T), max_residual: T) SparseError!bool {
            try validateSparseValueRange(T, zero(T), max_residual);
            return (try self.transposeMatmatResidualFrobeniusNorm(x, rhs)) <= max_residual;
        }

        pub fn transposeMatmatRelativeResidualFrobeniusNorm(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T)) SparseError!T {
            const residual = try self.transposeMatmatResidualFrobeniusNorm(x, rhs);
            return sparseRelativeMatrixResidualNorm(T, residual, self.frobeniusNorm(), x.data, rhs.data);
        }

        pub fn transposeMatmatResidualSummary(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T)) SparseError!SparseResidualSummary {
            const residual = try self.transposeMatmatResidualFrobeniusNorm(x, rhs);
            return sparseResidualSummary(T, residual, self.frobeniusNorm(), x.data, rhs.data);
        }

        pub fn transposeMatmatResidualSummaryMeetsBounds(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T), max_residual: f64, max_relative_residual: f64) SparseError!bool {
            const summary = try self.transposeMatmatResidualSummary(x, rhs);
            return summary.meetsBounds(max_residual, max_relative_residual);
        }

        pub fn transposeMatmatRelativeResidualFrobeniusNormMeetsBound(self: Self, x: array_mod.Array(T), rhs: array_mod.Array(T), max_relative_residual: T) SparseError!bool {
            try validateSparseValueRange(T, zero(T), max_relative_residual);
            return (try self.transposeMatmatRelativeResidualFrobeniusNorm(x, rhs)) <= max_relative_residual;
        }

        pub fn sum(self: Self) T {
            ensureNumeric(T);
            var total = zero(T);
            for (self.values) |value| total += value;
            return total;
        }

        pub fn absSum(self: Self) T {
            ensureNumeric(T);
            var total = zero(T);
            for (self.values) |value| total += absValue(T, value);
            return total;
        }

        pub fn sumInRange(self: Self, min_sum: T, max_sum: T) SparseError!bool {
            try validateSparseValueRange(T, min_sum, max_sum);
            return sparseValueInValidatedRange(T, self.sum(), min_sum, max_sum);
        }

        pub fn absSumInRange(self: Self, min_abs_sum: T, max_abs_sum: T) SparseError!bool {
            try validateSparseValueRange(T, min_abs_sum, max_abs_sum);
            return sparseValueInValidatedRange(T, self.absSum(), min_abs_sum, max_abs_sum);
        }

        pub fn minValue(self: Self) SparseError!T {
            return minStoredValue(T, self.values);
        }

        pub fn minValueIndex(self: Self) SparseError!usize {
            return minStoredValueIndex(T, self.values);
        }

        pub fn maxValue(self: Self) SparseError!T {
            return maxStoredValue(T, self.values);
        }

        pub fn maxValueIndex(self: Self) SparseError!usize {
            return maxStoredValueIndex(T, self.values);
        }

        pub fn minAbsValue(self: Self) SparseError!T {
            return minStoredAbsValue(T, self.values);
        }

        pub fn minAbsValueIndex(self: Self) SparseError!usize {
            return minStoredAbsValueIndex(T, self.values);
        }

        pub fn maxAbsValue(self: Self) SparseError!T {
            return maxStoredAbsValue(T, self.values);
        }

        pub fn maxAbsValueIndex(self: Self) SparseError!usize {
            return maxStoredAbsValueIndex(T, self.values);
        }

        pub fn nonFiniteCount(self: Self) usize {
            return sparseNonFiniteCount(T, self.values);
        }

        pub fn nonFiniteCountMeetsBound(self: Self, max_count: usize) bool {
            return self.nonFiniteCount() <= max_count;
        }

        pub fn nonFiniteCountInRange(self: Self, min_count: usize, max_count: usize) SparseError!bool {
            try validateCountRange(min_count, max_count);
            return sparseCountInValidatedRange(self.nonFiniteCount(), min_count, max_count);
        }

        pub fn allFinite(self: Self) bool {
            return sparseAllFinite(T, self.values);
        }

        pub fn columnNonFiniteCounts(self: Self) SparseError!array_mod.Array(usize) {
            var out = try array_mod.Array(usize).zeros(self.allocator, &.{self.cols});
            errdefer out.deinit();
            for (0..self.cols) |col| {
                for (self.col_offsets[col]..self.col_offsets[col + 1]) |pos| {
                    if (!sparseValueIsFinite(T, self.values[pos])) out.data[col] += 1;
                }
            }
            return out;
        }

        pub fn rowNonFiniteCounts(self: Self) SparseError!array_mod.Array(usize) {
            var out = try array_mod.Array(usize).zeros(self.allocator, &.{self.rows});
            errdefer out.deinit();
            for (self.values, self.row_indices) |value, row| {
                if (!sparseValueIsFinite(T, value)) out.data[row] += 1;
            }
            return out;
        }

        pub fn rowNonFiniteCountsInRange(self: Self, min_count: usize, max_count: usize) SparseError!bool {
            var counts = try self.rowNonFiniteCounts();
            defer counts.deinit();
            return sparseCountRangeInRange(counts.data, min_count, max_count);
        }

        pub fn rowNonFiniteCountsMeetBound(self: Self, max_count: usize) SparseError!bool {
            return self.rowNonFiniteCountsInRange(0, max_count);
        }

        pub fn columnNonFiniteCountsInRange(self: Self, min_count: usize, max_count: usize) SparseError!bool {
            var counts = try self.columnNonFiniteCounts();
            defer counts.deinit();
            return sparseCountRangeInRange(counts.data, min_count, max_count);
        }

        pub fn columnNonFiniteCountsMeetBound(self: Self, max_count: usize) SparseError!bool {
            return self.columnNonFiniteCountsInRange(0, max_count);
        }

        pub fn valueRangeInRange(self: Self, min_value: T, max_value: T) SparseError!bool {
            return sparseValueRangeInRange(T, self.values, min_value, max_value);
        }

        pub fn absValueRangeInRange(self: Self, min_abs_value: T, max_abs_value: T) SparseError!bool {
            return sparseAbsValueRangeInRange(T, self.values, min_abs_value, max_abs_value);
        }

        pub fn valueDynamicRange(self: Self) SparseError!f64 {
            return sparseStoredValueDynamicRange(T, self.values);
        }

        pub fn valueDynamicRangeMeetsBound(self: Self, max_dynamic_range: f64) SparseError!bool {
            return sparseStoredValueDynamicRangeMeetsBound(T, self.values, max_dynamic_range);
        }

        pub fn mean(self: Self) SparseError!f64 {
            ensureNumeric(T);
            const count = try sparseElementCount(self.rows, self.cols);
            if (count == 0) return error.EmptyArray;
            var total: f64 = 0;
            for (self.values) |value| total += sparseValueToF64(T, value);
            return total / sparseSizeToF64(count);
        }

        pub fn meanInRange(self: Self, min_mean: f64, max_mean: f64) SparseError!bool {
            try validateFiniteRange(min_mean, max_mean);
            return valueInF64Range(try self.mean(), min_mean, max_mean);
        }

        pub fn columnMeans(self: Self) SparseError!array_mod.Array(f64) {
            ensureNumeric(T);
            if (self.rows == 0) return error.EmptyArray;
            var out = try array_mod.Array(f64).zeros(self.allocator, &.{self.cols});
            errdefer out.deinit();
            for (0..self.cols) |col| {
                for (self.col_offsets[col]..self.col_offsets[col + 1]) |pos| out.data[col] += sparseValueToF64(T, self.values[pos]);
            }
            const divisor = sparseSizeToF64(self.rows);
            for (out.data) |*value| value.* /= divisor;
            return out;
        }

        pub fn rowMeans(self: Self) SparseError!array_mod.Array(f64) {
            ensureNumeric(T);
            if (self.cols == 0) return error.EmptyArray;
            var out = try array_mod.Array(f64).zeros(self.allocator, &.{self.rows});
            errdefer out.deinit();
            for (self.values, self.row_indices) |value, row| out.data[row] += sparseValueToF64(T, value);
            const divisor = sparseSizeToF64(self.cols);
            for (out.data) |*value| value.* /= divisor;
            return out;
        }

        pub fn rowMeansInRange(self: Self, min_mean: f64, max_mean: f64) SparseError!bool {
            try validateFiniteRange(min_mean, max_mean);
            var means = try self.rowMeans();
            defer means.deinit();
            for (means.data) |value| {
                if (!valueInF64Range(value, min_mean, max_mean)) return false;
            }
            return true;
        }

        pub fn columnMeansInRange(self: Self, min_mean: f64, max_mean: f64) SparseError!bool {
            try validateFiniteRange(min_mean, max_mean);
            var means = try self.columnMeans();
            defer means.deinit();
            for (means.data) |value| {
                if (!valueInF64Range(value, min_mean, max_mean)) return false;
            }
            return true;
        }

        pub fn variance(self: Self, correction: f64) SparseError!f64 {
            ensureNumeric(T);
            const count = try sparseElementCount(self.rows, self.cols);
            var sum_value: f64 = 0;
            var sum_sq: f64 = 0;
            for (self.values) |value| {
                sum_value += sparseValueToF64(T, value);
                sum_sq += sparseValueSquareToF64(T, value);
            }
            return sparseVarianceFromSums(sum_value, sum_sq, count, correction);
        }

        pub fn varianceInRange(self: Self, correction: f64, min_variance: f64, max_variance: f64) SparseError!bool {
            try validateNonNegativeRange(min_variance, max_variance);
            return valueInF64Range(try self.variance(correction), min_variance, max_variance);
        }

        pub fn stddev(self: Self, correction: f64) SparseError!f64 {
            return @sqrt(try self.variance(correction));
        }

        pub fn stddevInRange(self: Self, correction: f64, min_stddev: f64, max_stddev: f64) SparseError!bool {
            try validateNonNegativeRange(min_stddev, max_stddev);
            return sparseStddevInValidatedRangeFromVariance(try self.variance(correction), min_stddev, max_stddev);
        }

        pub fn sampleVariance(self: Self) SparseError!f64 {
            return self.variance(1);
        }

        pub fn sampleVarianceInRange(self: Self, min_variance: f64, max_variance: f64) SparseError!bool {
            return self.varianceInRange(1, min_variance, max_variance);
        }

        pub fn sampleStddev(self: Self) SparseError!f64 {
            return self.stddev(1);
        }

        pub fn sampleStddevInRange(self: Self, min_stddev: f64, max_stddev: f64) SparseError!bool {
            return self.stddevInRange(1, min_stddev, max_stddev);
        }

        pub fn columnVariances(self: Self, correction: f64) SparseError!array_mod.Array(f64) {
            ensureNumeric(T);
            if (self.rows == 0) return error.EmptyArray;
            var sums = try self.allocator.alloc(f64, self.cols);
            defer self.allocator.free(sums);
            @memset(sums, 0);
            var out = try array_mod.Array(f64).zeros(self.allocator, &.{self.cols});
            errdefer out.deinit();
            for (0..self.cols) |col| {
                for (self.col_offsets[col]..self.col_offsets[col + 1]) |pos| {
                    const numeric = sparseValueToF64(T, self.values[pos]);
                    sums[col] += numeric;
                    out.data[col] += numeric * numeric;
                }
            }
            try finalizeVarianceArray(out.data, sums, self.rows, correction);
            return out;
        }

        pub fn rowVariancesInRange(self: Self, correction: f64, min_variance: f64, max_variance: f64) SparseError!bool {
            try validateNonNegativeRange(min_variance, max_variance);
            var variances = try self.rowVariances(correction);
            defer variances.deinit();
            for (variances.data) |value| {
                if (!valueInF64Range(value, min_variance, max_variance)) return false;
            }
            return true;
        }

        pub fn columnVariancesInRange(self: Self, correction: f64, min_variance: f64, max_variance: f64) SparseError!bool {
            try validateNonNegativeRange(min_variance, max_variance);
            var variances = try self.columnVariances(correction);
            defer variances.deinit();
            for (variances.data) |value| {
                if (!valueInF64Range(value, min_variance, max_variance)) return false;
            }
            return true;
        }

        pub fn columnStddevs(self: Self, correction: f64) SparseError!array_mod.Array(f64) {
            const out = try self.columnVariances(correction);
            sqrtArray(out.data);
            return out;
        }

        pub fn rowStddevsInRange(self: Self, correction: f64, min_stddev: f64, max_stddev: f64) SparseError!bool {
            try validateNonNegativeRange(min_stddev, max_stddev);
            var stddevs = try self.rowStddevs(correction);
            defer stddevs.deinit();
            for (stddevs.data) |value| {
                if (!valueInF64Range(value, min_stddev, max_stddev)) return false;
            }
            return true;
        }

        pub fn columnStddevsInRange(self: Self, correction: f64, min_stddev: f64, max_stddev: f64) SparseError!bool {
            try validateNonNegativeRange(min_stddev, max_stddev);
            var stddevs = try self.columnStddevs(correction);
            defer stddevs.deinit();
            for (stddevs.data) |value| {
                if (!valueInF64Range(value, min_stddev, max_stddev)) return false;
            }
            return true;
        }

        pub fn columnSampleVariances(self: Self) SparseError!array_mod.Array(f64) {
            return self.columnVariances(1);
        }

        pub fn columnSampleStddevs(self: Self) SparseError!array_mod.Array(f64) {
            return self.columnStddevs(1);
        }

        pub fn rowVariances(self: Self, correction: f64) SparseError!array_mod.Array(f64) {
            ensureNumeric(T);
            if (self.cols == 0) return error.EmptyArray;
            var sums = try self.allocator.alloc(f64, self.rows);
            defer self.allocator.free(sums);
            @memset(sums, 0);
            var out = try array_mod.Array(f64).zeros(self.allocator, &.{self.rows});
            errdefer out.deinit();
            for (self.values, self.row_indices) |value, row| {
                const numeric = sparseValueToF64(T, value);
                sums[row] += numeric;
                out.data[row] += numeric * numeric;
            }
            try finalizeVarianceArray(out.data, sums, self.cols, correction);
            return out;
        }

        pub fn rowStddevs(self: Self, correction: f64) SparseError!array_mod.Array(f64) {
            const out = try self.rowVariances(correction);
            sqrtArray(out.data);
            return out;
        }

        pub fn rowSampleVariances(self: Self) SparseError!array_mod.Array(f64) {
            return self.rowVariances(1);
        }

        pub fn rowSampleStddevs(self: Self) SparseError!array_mod.Array(f64) {
            return self.rowStddevs(1);
        }

        pub fn frobeniusNorm(self: Self) T {
            ensureFloat(T);
            if (comptime T == f64) {
                const veyra_view = @as(CscMatrix(f64), self).asVeyraView() catch return 0;
                return @as(T, @floatCast(veyra.cscFrobeniusNorm(f64, veyra_view)));
            }
            var total = zero(T);
            for (self.values) |value| total += value * value;
            return @sqrt(total);
        }

        pub fn frobeniusNormMeetsBound(self: Self, max_norm: T) SparseError!bool {
            ensureFloat(T);
            try validateSparseValueRange(T, zero(T), max_norm);
            return self.frobeniusNorm() <= max_norm;
        }

        pub fn density(self: Self) SparseError!f64 {
            const total = self.rows * self.cols;
            if (total == 0) return 0;
            if (comptime T == f64) {
                const veyra_view = try @as(CscMatrix(f64), self).asVeyraView();
                return veyra.cscDensity(f64, veyra_view) catch return error.BackendFailure;
            }
            return @as(f64, @floatFromInt(self.values.len)) / @as(f64, @floatFromInt(total));
        }

        pub fn densityInRange(self: Self, min_density: f64, max_density: f64) SparseError!bool {
            try validateNonNegativeRange(min_density, max_density);
            const current = try self.density();
            return current >= min_density and current <= max_density;
        }

        pub fn oneNorm(self: Self) SparseError!T {
            ensureNumeric(T);
            var max_sum = zero(T);
            for (0..self.cols) |col| {
                var col_sum = zero(T);
                for (self.col_offsets[col]..self.col_offsets[col + 1]) |pos| col_sum += absValue(T, self.values[pos]);
                if (col_sum > max_sum) max_sum = col_sum;
            }
            return max_sum;
        }

        pub fn oneNormMeetsBound(self: Self, max_norm: T) SparseError!bool {
            ensureNumeric(T);
            try validateSparseValueRange(T, zero(T), max_norm);
            return (try self.oneNorm()) <= max_norm;
        }

        pub fn infNorm(self: Self) SparseError!T {
            ensureNumeric(T);
            var row_sums = try self.allocator.alloc(T, self.rows);
            defer self.allocator.free(row_sums);
            @memset(row_sums, zero(T));
            for (self.values, self.row_indices) |value, row| row_sums[row] += absValue(T, value);
            var max_sum = zero(T);
            for (row_sums) |sum_value| {
                if (sum_value > max_sum) max_sum = sum_value;
            }
            return max_sum;
        }

        pub fn infNormMeetsBound(self: Self, max_norm: T) SparseError!bool {
            ensureNumeric(T);
            try validateSparseValueRange(T, zero(T), max_norm);
            return (try self.infNorm()) <= max_norm;
        }

        pub fn columnNnz(self: Self) SparseError!array_mod.Array(usize) {
            var out = try array_mod.Array(usize).zeros(self.allocator, &.{self.cols});
            errdefer out.deinit();
            for (0..self.cols) |c| out.data[c] = self.col_offsets[c + 1] - self.col_offsets[c];
            return out;
        }

        pub fn rowNnz(self: Self) SparseError!array_mod.Array(usize) {
            var out = try array_mod.Array(usize).zeros(self.allocator, &.{self.rows});
            errdefer out.deinit();
            for (self.row_indices) |row| out.data[row] += 1;
            return out;
        }

        pub fn averageRowNnz(self: Self) SparseError!f64 {
            return sparseCountAverage(self.values.len, self.rows);
        }

        pub fn averageColumnNnz(self: Self) SparseError!f64 {
            return sparseCountAverage(self.values.len, self.cols);
        }

        pub fn averageRowNnzInRange(self: Self, min_average: f64, max_average: f64) SparseError!bool {
            return sparseCountAverageInRange(self.values.len, self.rows, min_average, max_average);
        }

        pub fn averageColumnNnzInRange(self: Self, min_average: f64, max_average: f64) SparseError!bool {
            return sparseCountAverageInRange(self.values.len, self.cols, min_average, max_average);
        }

        pub fn emptyColumnCount(self: Self) usize {
            var empty_count: usize = 0;
            for (0..self.cols) |col| {
                if (self.col_offsets[col] == self.col_offsets[col + 1]) empty_count += 1;
            }
            return empty_count;
        }

        pub fn emptyRowCount(self: Self) SparseError!usize {
            var counts = try self.rowNnz();
            defer counts.deinit();
            var empty_count: usize = 0;
            for (counts.data) |count| {
                if (count == 0) empty_count += 1;
            }
            return empty_count;
        }

        pub fn emptyRowCountMeetsBound(self: Self, max_count: usize) SparseError!bool {
            return (try self.emptyRowCount()) <= max_count;
        }

        pub fn emptyRowCountInRange(self: Self, min_count: usize, max_count: usize) SparseError!bool {
            try validateCountRange(min_count, max_count);
            return sparseCountInValidatedRange(try self.emptyRowCount(), min_count, max_count);
        }

        pub fn emptyColumnCountMeetsBound(self: Self, max_count: usize) bool {
            return self.emptyColumnCount() <= max_count;
        }

        pub fn emptyColumnCountInRange(self: Self, min_count: usize, max_count: usize) SparseError!bool {
            try validateCountRange(min_count, max_count);
            return sparseCountInValidatedRange(self.emptyColumnCount(), min_count, max_count);
        }

        pub fn emptyRowFraction(self: Self) SparseError!f64 {
            return sparseCountFraction(try self.emptyRowCount(), self.rows);
        }

        pub fn emptyColumnFraction(self: Self) SparseError!f64 {
            return sparseCountFraction(self.emptyColumnCount(), self.cols);
        }

        pub fn emptyRowFractionInRange(self: Self, min_fraction: f64, max_fraction: f64) SparseError!bool {
            return sparseCountFractionInRange(try self.emptyRowCount(), self.rows, min_fraction, max_fraction);
        }

        pub fn emptyColumnFractionInRange(self: Self, min_fraction: f64, max_fraction: f64) SparseError!bool {
            return sparseCountFractionInRange(self.emptyColumnCount(), self.cols, min_fraction, max_fraction);
        }

        pub fn minRowNnz(self: Self) SparseError!usize {
            var counts = try self.rowNnz();
            defer counts.deinit();
            return sparseMinCount(counts.data);
        }

        pub fn maxRowNnz(self: Self) SparseError!usize {
            var counts = try self.rowNnz();
            defer counts.deinit();
            return sparseMaxCount(counts.data);
        }

        pub fn minColumnNnz(self: Self) SparseError!usize {
            var counts = try self.columnNnz();
            defer counts.deinit();
            return sparseMinCount(counts.data);
        }

        pub fn maxColumnNnz(self: Self) SparseError!usize {
            var counts = try self.columnNnz();
            defer counts.deinit();
            return sparseMaxCount(counts.data);
        }

        pub fn rowNnzRangeInRange(self: Self, min_count: usize, max_count: usize) SparseError!bool {
            var counts = try self.rowNnz();
            defer counts.deinit();
            return sparseCountRangeInRange(counts.data, min_count, max_count);
        }

        pub fn columnNnzRangeInRange(self: Self, min_count: usize, max_count: usize) SparseError!bool {
            var counts = try self.columnNnz();
            defer counts.deinit();
            return sparseCountRangeInRange(counts.data, min_count, max_count);
        }

        pub fn rowNnzSpread(self: Self) SparseError!usize {
            var counts = try self.rowNnz();
            defer counts.deinit();
            return sparseCountSpread(counts.data);
        }

        pub fn columnNnzSpread(self: Self) SparseError!usize {
            var counts = try self.columnNnz();
            defer counts.deinit();
            return sparseCountSpread(counts.data);
        }

        pub fn rowNnzSpreadMeetsBound(self: Self, max_spread: usize) SparseError!bool {
            var counts = try self.rowNnz();
            defer counts.deinit();
            return sparseCountSpreadMeetsBound(counts.data, max_spread);
        }

        pub fn columnNnzSpreadMeetsBound(self: Self, max_spread: usize) SparseError!bool {
            var counts = try self.columnNnz();
            defer counts.deinit();
            return sparseCountSpreadMeetsBound(counts.data, max_spread);
        }

        pub fn columnSums(self: Self) SparseError!array_mod.Array(T) {
            ensureNumeric(T);
            if (comptime T == f64) return self.columnSumsF64();
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.cols});
            errdefer out.deinit();
            for (0..self.cols) |c| {
                for (self.col_offsets[c]..self.col_offsets[c + 1]) |pos| out.data[c] += self.values[pos];
            }
            return out;
        }

        fn columnSumsF64(self: Self) SparseError!array_mod.Array(f64) {
            const veyra_view = try @as(CscMatrix(f64), self).asVeyraView();
            var out = veyra.Vector(f64).zeros(self.allocator, self.cols) catch return error.BackendFailure;
            defer out.deinit();
            veyra.cscColumnSums(f64, veyra_view, out.asMut()) catch return error.BackendFailure;
            return array_mod.Array(f64).fromSlice(self.allocator, out.data, &.{self.cols});
        }

        pub fn rowSums(self: Self) SparseError!array_mod.Array(T) {
            ensureNumeric(T);
            if (comptime T == f64) return self.rowSumsF64();
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.rows});
            errdefer out.deinit();
            for (0..self.cols) |c| {
                for (self.col_offsets[c]..self.col_offsets[c + 1]) |pos| out.data[self.row_indices[pos]] += self.values[pos];
            }
            return out;
        }

        fn rowSumsF64(self: Self) SparseError!array_mod.Array(f64) {
            const veyra_view = try @as(CscMatrix(f64), self).asVeyraView();
            var out = veyra.Vector(f64).zeros(self.allocator, self.rows) catch return error.BackendFailure;
            defer out.deinit();
            veyra.cscRowSumsWithWorkspace(f64, veyra_view, out.asMut()) catch return error.BackendFailure;
            return array_mod.Array(f64).fromSlice(self.allocator, out.data, &.{self.rows});
        }

        pub fn rowSumsInRange(self: Self, min_sum: T, max_sum: T) SparseError!bool {
            var sums = try self.rowSums();
            defer sums.deinit();
            return sparseValueRangeInRange(T, sums.data, min_sum, max_sum);
        }

        pub fn columnSumsInRange(self: Self, min_sum: T, max_sum: T) SparseError!bool {
            var sums = try self.columnSums();
            defer sums.deinit();
            return sparseValueRangeInRange(T, sums.data, min_sum, max_sum);
        }

        pub fn columnMins(self: Self) SparseError!array_mod.Array(T) {
            ensureNumeric(T);
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.cols});
            errdefer out.deinit();
            for (0..self.cols) |col| {
                for (self.col_offsets[col]..self.col_offsets[col + 1]) |pos| {
                    const value = self.values[pos];
                    if (valueLess(T, value, out.data[col])) out.data[col] = value;
                }
            }
            return out;
        }

        pub fn columnMaxes(self: Self) SparseError!array_mod.Array(T) {
            ensureNumeric(T);
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.cols});
            errdefer out.deinit();
            for (0..self.cols) |col| {
                for (self.col_offsets[col]..self.col_offsets[col + 1]) |pos| {
                    const value = self.values[pos];
                    if (valueGreater(T, value, out.data[col])) out.data[col] = value;
                }
            }
            return out;
        }

        pub fn rowMins(self: Self) SparseError!array_mod.Array(T) {
            ensureNumeric(T);
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.rows});
            errdefer out.deinit();
            for (self.values, self.row_indices) |value, row| {
                if (valueLess(T, value, out.data[row])) out.data[row] = value;
            }
            return out;
        }

        pub fn rowMaxes(self: Self) SparseError!array_mod.Array(T) {
            ensureNumeric(T);
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.rows});
            errdefer out.deinit();
            for (self.values, self.row_indices) |value, row| {
                if (valueGreater(T, value, out.data[row])) out.data[row] = value;
            }
            return out;
        }

        pub fn columnMinAbs(self: Self) SparseError!array_mod.Array(T) {
            ensureNumeric(T);
            var out = try self.columnMaxAbs();
            for (0..self.cols) |col| {
                for (self.col_offsets[col]..self.col_offsets[col + 1]) |pos| {
                    const magnitude = absValue(T, self.values[pos]);
                    if (magnitude < out.data[col]) out.data[col] = magnitude;
                }
            }
            return out;
        }

        pub fn columnMaxAbs(self: Self) SparseError!array_mod.Array(T) {
            ensureNumeric(T);
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.cols});
            errdefer out.deinit();
            for (0..self.cols) |col| {
                for (self.col_offsets[col]..self.col_offsets[col + 1]) |pos| {
                    const magnitude = absValue(T, self.values[pos]);
                    if (magnitude > out.data[col]) out.data[col] = magnitude;
                }
            }
            return out;
        }

        pub fn rowMinsInRange(self: Self, min_value: T, max_value: T) SparseError!bool {
            var values = try self.rowMins();
            defer values.deinit();
            return sparseValueRangeInRange(T, values.data, min_value, max_value);
        }

        pub fn columnMinsInRange(self: Self, min_value: T, max_value: T) SparseError!bool {
            var values = try self.columnMins();
            defer values.deinit();
            return sparseValueRangeInRange(T, values.data, min_value, max_value);
        }

        pub fn rowMaxesInRange(self: Self, min_value: T, max_value: T) SparseError!bool {
            var values = try self.rowMaxes();
            defer values.deinit();
            return sparseValueRangeInRange(T, values.data, min_value, max_value);
        }

        pub fn columnMaxesInRange(self: Self, min_value: T, max_value: T) SparseError!bool {
            var values = try self.columnMaxes();
            defer values.deinit();
            return sparseValueRangeInRange(T, values.data, min_value, max_value);
        }

        pub fn rowMinAbs(self: Self) SparseError!array_mod.Array(T) {
            ensureNumeric(T);
            var out = try self.rowMaxAbs();
            for (self.values, self.row_indices) |value, row| {
                const magnitude = absValue(T, value);
                if (magnitude < out.data[row]) out.data[row] = magnitude;
            }
            return out;
        }

        pub fn rowMaxAbs(self: Self) SparseError!array_mod.Array(T) {
            ensureNumeric(T);
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.rows});
            errdefer out.deinit();
            for (self.values, self.row_indices) |value, row| {
                const magnitude = absValue(T, value);
                if (magnitude > out.data[row]) out.data[row] = magnitude;
            }
            return out;
        }

        pub fn rowSampleVariancesInRange(self: Self, min_variance: f64, max_variance: f64) SparseError!bool {
            return self.rowVariancesInRange(1, min_variance, max_variance);
        }

        pub fn rowSampleStddevsInRange(self: Self, min_stddev: f64, max_stddev: f64) SparseError!bool {
            return self.rowStddevsInRange(1, min_stddev, max_stddev);
        }

        pub fn columnSampleVariancesInRange(self: Self, min_variance: f64, max_variance: f64) SparseError!bool {
            return self.columnVariancesInRange(1, min_variance, max_variance);
        }

        pub fn columnSampleStddevsInRange(self: Self, min_stddev: f64, max_stddev: f64) SparseError!bool {
            return self.columnStddevsInRange(1, min_stddev, max_stddev);
        }

        pub fn columnAbsSums(self: Self) SparseError!array_mod.Array(T) {
            ensureNumeric(T);
            if (comptime T == f64) return self.columnAbsSumsF64();
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.cols});
            errdefer out.deinit();
            for (0..self.cols) |c| {
                for (self.col_offsets[c]..self.col_offsets[c + 1]) |pos| out.data[c] += absValue(T, self.values[pos]);
            }
            return out;
        }

        fn columnAbsSumsF64(self: Self) SparseError!array_mod.Array(f64) {
            const veyra_view = try @as(CscMatrix(f64), self).asVeyraView();
            var out = veyra.Vector(f64).zeros(self.allocator, self.cols) catch return error.BackendFailure;
            defer out.deinit();
            veyra.cscColumnAbsSums(f64, veyra_view, out.asMut()) catch return error.BackendFailure;
            return array_mod.Array(f64).fromSlice(self.allocator, out.data, &.{self.cols});
        }

        pub fn rowAbsSums(self: Self) SparseError!array_mod.Array(T) {
            ensureNumeric(T);
            if (comptime T == f64) return self.rowAbsSumsF64();
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.rows});
            errdefer out.deinit();
            for (0..self.cols) |c| {
                for (self.col_offsets[c]..self.col_offsets[c + 1]) |pos| out.data[self.row_indices[pos]] += absValue(T, self.values[pos]);
            }
            return out;
        }

        fn rowAbsSumsF64(self: Self) SparseError!array_mod.Array(f64) {
            const veyra_view = try @as(CscMatrix(f64), self).asVeyraView();
            var out = veyra.Vector(f64).zeros(self.allocator, self.rows) catch return error.BackendFailure;
            defer out.deinit();
            veyra.cscRowAbsSumsWithWorkspace(f64, veyra_view, out.asMut()) catch return error.BackendFailure;
            return array_mod.Array(f64).fromSlice(self.allocator, out.data, &.{self.rows});
        }

        pub fn rowMinAbsInRange(self: Self, min_value: T, max_value: T) SparseError!bool {
            var values = try self.rowMinAbs();
            defer values.deinit();
            return sparseValueRangeInRange(T, values.data, min_value, max_value);
        }

        pub fn columnMinAbsInRange(self: Self, min_value: T, max_value: T) SparseError!bool {
            var values = try self.columnMinAbs();
            defer values.deinit();
            return sparseValueRangeInRange(T, values.data, min_value, max_value);
        }

        pub fn rowMaxAbsInRange(self: Self, min_value: T, max_value: T) SparseError!bool {
            var values = try self.rowMaxAbs();
            defer values.deinit();
            return sparseValueRangeInRange(T, values.data, min_value, max_value);
        }

        pub fn columnMaxAbsInRange(self: Self, min_value: T, max_value: T) SparseError!bool {
            var values = try self.columnMaxAbs();
            defer values.deinit();
            return sparseValueRangeInRange(T, values.data, min_value, max_value);
        }

        pub fn rowAbsSumsInRange(self: Self, min_sum: T, max_sum: T) SparseError!bool {
            var sums = try self.rowAbsSums();
            defer sums.deinit();
            return sparseValueRangeInRange(T, sums.data, min_sum, max_sum);
        }

        pub fn columnAbsSumsInRange(self: Self, min_sum: T, max_sum: T) SparseError!bool {
            var sums = try self.columnAbsSums();
            defer sums.deinit();
            return sparseValueRangeInRange(T, sums.data, min_sum, max_sum);
        }

        pub fn columnNorms(self: Self) SparseError!array_mod.Array(T) {
            ensureFloat(T);
            if (comptime T == f64) return self.columnNormsF64();
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.cols});
            errdefer out.deinit();
            for (0..self.cols) |c| {
                var total = zero(T);
                for (self.col_offsets[c]..self.col_offsets[c + 1]) |pos| total += self.values[pos] * self.values[pos];
                out.data[c] = @sqrt(total);
            }
            return out;
        }

        fn columnNormsF64(self: Self) SparseError!array_mod.Array(f64) {
            const veyra_view = try @as(CscMatrix(f64), self).asVeyraView();
            var out = veyra.Vector(f64).zeros(self.allocator, self.cols) catch return error.BackendFailure;
            defer out.deinit();
            veyra.cscColumnNorms(f64, veyra_view, out.asMut()) catch return error.BackendFailure;
            return array_mod.Array(f64).fromSlice(self.allocator, out.data, &.{self.cols});
        }

        pub fn rowNorms(self: Self) SparseError!array_mod.Array(T) {
            ensureFloat(T);
            if (comptime T == f64) return self.rowNormsF64();
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.rows});
            errdefer out.deinit();
            for (0..self.cols) |c| {
                for (self.col_offsets[c]..self.col_offsets[c + 1]) |pos| {
                    const row = self.row_indices[pos];
                    out.data[row] += self.values[pos] * self.values[pos];
                }
            }
            for (out.data) |*value| value.* = @sqrt(value.*);
            return out;
        }

        fn rowNormsF64(self: Self) SparseError!array_mod.Array(f64) {
            const veyra_view = try @as(CscMatrix(f64), self).asVeyraView();
            var out = veyra.Vector(f64).zeros(self.allocator, self.rows) catch return error.BackendFailure;
            defer out.deinit();
            veyra.cscRowNormsWithWorkspace(f64, veyra_view, out.asMut()) catch return error.BackendFailure;
            return array_mod.Array(f64).fromSlice(self.allocator, out.data, &.{self.rows});
        }

        pub fn rowNormsInRange(self: Self, min_norm: T, max_norm: T) SparseError!bool {
            try validateSparseValueRange(T, zero(T), max_norm);
            if (min_norm < zero(T)) return error.InvalidShape;
            var norms = try self.rowNorms();
            defer norms.deinit();
            return sparseValueRangeInRange(T, norms.data, min_norm, max_norm);
        }

        pub fn columnNormsInRange(self: Self, min_norm: T, max_norm: T) SparseError!bool {
            try validateSparseValueRange(T, zero(T), max_norm);
            if (min_norm < zero(T)) return error.InvalidShape;
            var norms = try self.columnNorms();
            defer norms.deinit();
            return sparseValueRangeInRange(T, norms.data, min_norm, max_norm);
        }

        pub fn get(self: Self, row: usize, col: usize) ?T {
            if (row >= self.rows or col >= self.cols) return null;
            var found = false;
            var total = zero(T);
            // Match dense materialization for duplicate coordinates.  This is
            // intentionally not delegated to Veyra point lookup because Veyra
            // views assume canonical compressed rows/columns in several fast
            // diagnostics paths.
            for (self.col_offsets[col]..self.col_offsets[col + 1]) |pos| {
                if (self.row_indices[pos] == row) {
                    total = addSparseValue(T, total, self.values[pos]);
                    found = true;
                }
            }
            return if (found) total else null;
        }

        fn hasEntry(self: Self, row: usize, col: usize) bool {
            if (row >= self.rows or col >= self.cols) return false;
            for (self.col_offsets[col]..self.col_offsets[col + 1]) |pos| {
                if (self.row_indices[pos] == row) return true;
            }
            return false;
        }

        pub fn setExisting(self: *Self, row: usize, col: usize, value: T) SparseError!void {
            if (row >= self.rows or col >= self.cols) return error.IndexOutOfBounds;
            var found = false;
            for (self.col_offsets[col]..self.col_offsets[col + 1]) |pos| {
                if (self.row_indices[pos] == row) {
                    self.values[pos] = value;
                    found = true;
                }
            }
            if (!found) return error.InvalidShape;
        }

        pub fn setStoredValue(self: *Self, row: usize, col: usize, value: T) SparseError!void {
            return self.setExisting(row, col, value);
        }

        pub fn diagonal(self: Self) SparseError!array_mod.Array(T) {
            if (self.rows != self.cols) return error.NonMatrixArray;
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.rows});
            errdefer out.deinit();
            for (0..self.rows) |i| out.data[i] = self.get(i, i) orelse zero(T);
            return out;
        }

        pub fn diagonalOffset(self: Self, offset: isize) SparseError!array_mod.Array(T) {
            return sparseDenseDiagonalOffset(T, self, offset);
        }

        pub fn diagonalWithOffset(self: Self, offset: isize) SparseError!array_mod.Array(T) {
            return self.diagonalOffset(offset);
        }

        pub fn setDiagonal(self: *Self, value: T) SparseError!void {
            if (self.rows != self.cols) return error.NonMatrixArray;
            for (0..self.cols) |col| {
                var found = false;
                for (self.col_offsets[col]..self.col_offsets[col + 1]) |pos| {
                    if (self.row_indices[pos] == col) {
                        self.values[pos] = value;
                        found = true;
                    }
                }
                if (!found) return error.InvalidShape;
            }
        }

        pub fn addToDiagonal(self: *Self, alpha: T) SparseError!void {
            ensureNumeric(T);
            if (alpha == zero(T)) return;
            if (self.rows != self.cols) return error.NonMatrixArray;
            for (0..self.cols) |col| {
                var found = false;
                for (self.col_offsets[col]..self.col_offsets[col + 1]) |pos| {
                    if (self.row_indices[pos] == col) {
                        self.values[pos] += alpha;
                        found = true;
                    }
                }
                if (!found) return error.InvalidShape;
            }
        }

        pub fn setDiagonalValues(self: *Self, diagonal_values: []const T) SparseError!void {
            if (self.rows != self.cols) return error.NonMatrixArray;
            if (diagonal_values.len != self.rows) return error.ShapeMismatch;
            for (0..self.cols) |col| {
                var found = false;
                for (self.col_offsets[col]..self.col_offsets[col + 1]) |pos| {
                    if (self.row_indices[pos] == col) {
                        self.values[pos] = diagonal_values[col];
                        found = true;
                    }
                }
                if (!found) return error.InvalidShape;
            }
        }

        pub fn setDiagonalVector(self: *Self, diagonal_values: []const T) SparseError!void {
            return self.setDiagonalValues(diagonal_values);
        }

        pub fn addDiagonalValues(self: *Self, diagonal_values: []const T) SparseError!void {
            ensureNumeric(T);
            if (self.rows != self.cols) return error.NonMatrixArray;
            if (diagonal_values.len != self.rows) return error.ShapeMismatch;
            var any_nonzero = false;
            for (diagonal_values) |value| {
                if (isNonZero(T, value)) {
                    any_nonzero = true;
                    break;
                }
            }
            if (!any_nonzero) return;
            for (0..self.cols) |col| {
                var found = false;
                for (self.col_offsets[col]..self.col_offsets[col + 1]) |pos| {
                    if (self.row_indices[pos] == col) {
                        self.values[pos] += diagonal_values[col];
                        found = true;
                    }
                }
                if (!found) return error.InvalidShape;
            }
        }

        pub fn addDiagonalVector(self: *Self, diagonal_values: []const T) SparseError!void {
            return self.addDiagonalValues(diagonal_values);
        }

        pub fn multiplyDiagonalValues(self: *Self, diagonal_values: []const T) SparseError!void {
            ensureNumeric(T);
            if (self.rows != self.cols) return error.NonMatrixArray;
            if (diagonal_values.len != self.rows) return error.ShapeMismatch;
            var all_one = true;
            for (diagonal_values) |value| {
                if (value != oneValue(T)) {
                    all_one = false;
                    break;
                }
            }
            if (all_one) return;
            for (0..self.cols) |col| {
                var found = false;
                for (self.col_offsets[col]..self.col_offsets[col + 1]) |pos| {
                    if (self.row_indices[pos] == col) {
                        self.values[pos] *= diagonal_values[col];
                        found = true;
                    }
                }
                if (!found) return error.InvalidShape;
            }
        }

        pub fn multiplyDiagonalVector(self: *Self, diagonal_values: []const T) SparseError!void {
            return self.multiplyDiagonalValues(diagonal_values);
        }

        pub fn minAbsDiagonal(self: Self) SparseError!T {
            var diagonal_values = try self.diagonal();
            defer diagonal_values.deinit();
            return (try sparseDiagonalAbsRange(T, diagonal_values.data)).min_abs;
        }

        pub fn maxAbsDiagonal(self: Self) SparseError!T {
            var diagonal_values = try self.diagonal();
            defer diagonal_values.deinit();
            return (try sparseDiagonalAbsRange(T, diagonal_values.data)).max_abs;
        }

        pub fn diagonalDynamicRange(self: Self) SparseError!f64 {
            var diagonal_values = try self.diagonal();
            defer diagonal_values.deinit();
            return sparseDiagonalDynamicRangeFromValues(T, diagonal_values.data);
        }

        pub fn diagonalDynamicRangeMeetsBound(self: Self, max_dynamic_range: f64) SparseError!bool {
            var diagonal_values = try self.diagonal();
            defer diagonal_values.deinit();
            return sparseDiagonalDynamicRangeMeetsBoundFromValues(T, diagonal_values.data, max_dynamic_range);
        }

        pub fn trace(self: Self) SparseError!T {
            ensureNumeric(T);
            if (self.rows != self.cols) return error.NonMatrixArray;
            var total = zero(T);
            for (0..self.rows) |i| total = addSparseValue(T, total, self.get(i, i) orelse zero(T));
            return total;
        }

        pub fn traceInRange(self: Self, min_value: T, max_value: T) SparseError!bool {
            try validateSparseValueRange(T, min_value, max_value);
            const trace_value = try self.trace();
            return trace_value >= min_value and trace_value <= max_value;
        }

        pub fn normalizedTrace(self: Self) SparseError!f64 {
            return sparseNormalizedTraceFromTrace(T, try self.trace(), self.rows);
        }

        pub fn normalizedTraceInRange(self: Self, min_value: f64, max_value: f64) SparseError!bool {
            try validateFiniteRange(min_value, max_value);
            return sparseNormalizedTraceInRangeFromTrace(T, try self.trace(), self.rows, min_value, max_value);
        }

        pub fn missingDiagonalCount(self: Self) SparseError!usize {
            if (self.rows != self.cols) return error.NonMatrixArray;
            var count: usize = 0;
            for (0..self.rows) |i| {
                if (!self.hasEntry(i, i)) count += 1;
            }
            return count;
        }

        pub fn missingDiagonalCountMeetsBound(self: Self, max_count: usize) SparseError!bool {
            return (try self.missingDiagonalCount()) <= max_count;
        }

        pub fn missingDiagonalCountInRange(self: Self, min_count: usize, max_count: usize) SparseError!bool {
            try validateCountRange(min_count, max_count);
            return sparseCountInValidatedRange(try self.missingDiagonalCount(), min_count, max_count);
        }

        pub fn zeroDiagonalCount(self: Self) SparseError!usize {
            if (self.rows != self.cols) return error.NonMatrixArray;
            var count: usize = 0;
            for (0..self.rows) |i| {
                if (self.get(i, i)) |value| {
                    if (value == zero(T)) count += 1;
                }
            }
            return count;
        }

        pub fn zeroDiagonalCountMeetsBound(self: Self, max_count: usize) SparseError!bool {
            return (try self.zeroDiagonalCount()) <= max_count;
        }

        pub fn zeroDiagonalCountInRange(self: Self, min_count: usize, max_count: usize) SparseError!bool {
            try validateCountRange(min_count, max_count);
            return sparseCountInValidatedRange(try self.zeroDiagonalCount(), min_count, max_count);
        }

        pub fn nonPositiveDiagonalCount(self: Self) SparseError!usize {
            ensureNumeric(T);
            if (self.rows != self.cols) return error.NonMatrixArray;
            var count: usize = 0;
            for (0..self.rows) |i| {
                if (self.get(i, i)) |value| {
                    if (value <= zero(T)) count += 1;
                }
            }
            return count;
        }

        pub fn nonPositiveDiagonalCountMeetsBound(self: Self, max_count: usize) SparseError!bool {
            return (try self.nonPositiveDiagonalCount()) <= max_count;
        }

        pub fn nonPositiveDiagonalCountInRange(self: Self, min_count: usize, max_count: usize) SparseError!bool {
            try validateCountRange(min_count, max_count);
            return sparseCountInValidatedRange(try self.nonPositiveDiagonalCount(), min_count, max_count);
        }

        pub fn bandwidth(self: Self) SparseError!usize {
            if (self.rows != self.cols) return error.NonMatrixArray;
            var bw: usize = 0;
            for (0..self.cols) |c| {
                for (self.col_offsets[c]..self.col_offsets[c + 1]) |pos| {
                    const r = self.row_indices[pos];
                    const distance = if (r > c) r - c else c - r;
                    if (distance > bw) bw = distance;
                }
            }
            return bw;
        }

        pub fn bandwidthMeetsBound(self: Self, max_bandwidth: usize) SparseError!bool {
            if (self.rows != self.cols) return error.NonMatrixArray;
            for (0..self.cols) |c| {
                for (self.col_offsets[c]..self.col_offsets[c + 1]) |pos| {
                    const r = self.row_indices[pos];
                    const distance = if (r > c) r - c else c - r;
                    if (distance > max_bandwidth) return false;
                }
            }
            return true;
        }

        pub fn columnIntersectionBandwidth(self: Self) SparseError!usize {
            var coo = try self.toCoo();
            defer coo.deinit();
            return coo.columnIntersectionBandwidth();
        }

        pub fn columnIntersectionBandwidthMeetsBound(self: Self, max_bandwidth: usize) SparseError!bool {
            var coo = try self.toCoo();
            defer coo.deinit();
            return coo.columnIntersectionBandwidthMeetsBound(max_bandwidth);
        }

        pub fn lowerNnz(self: Self, comptime strict: bool) SparseError!usize {
            if (self.rows != self.cols) return error.NonMatrixArray;
            var count: usize = 0;
            for (0..self.cols) |col| {
                for (self.col_offsets[col]..self.col_offsets[col + 1]) |pos| {
                    if (triangularIndexMatches(self.row_indices[pos], col, strict, true)) count += 1;
                }
            }
            return count;
        }

        pub fn upperNnz(self: Self, comptime strict: bool) SparseError!usize {
            if (self.rows != self.cols) return error.NonMatrixArray;
            var count: usize = 0;
            for (0..self.cols) |col| {
                for (self.col_offsets[col]..self.col_offsets[col + 1]) |pos| {
                    if (triangularIndexMatches(self.row_indices[pos], col, strict, false)) count += 1;
                }
            }
            return count;
        }

        pub fn lowerNnzMeetsBound(self: Self, comptime strict: bool, max_count: usize) SparseError!bool {
            return (try self.lowerNnz(strict)) <= max_count;
        }

        pub fn upperNnzMeetsBound(self: Self, comptime strict: bool, max_count: usize) SparseError!bool {
            return (try self.upperNnz(strict)) <= max_count;
        }

        pub fn lowerNnzInRange(self: Self, comptime strict: bool, min_count: usize, max_count: usize) SparseError!bool {
            if (min_count > max_count) return error.InvalidShape;
            const count = try self.lowerNnz(strict);
            return count >= min_count and count <= max_count;
        }

        pub fn upperNnzInRange(self: Self, comptime strict: bool, min_count: usize, max_count: usize) SparseError!bool {
            if (min_count > max_count) return error.InvalidShape;
            const count = try self.upperNnz(strict);
            return count >= min_count and count <= max_count;
        }

        pub fn lowerProfile(self: Self) SparseError!usize {
            if (self.rows != self.cols) return error.NonMatrixArray;
            var builder = try SparseProfileBuilder.init(self.allocator, self.rows);
            defer builder.deinit(self.allocator);
            for (0..self.cols) |col| {
                for (self.col_offsets[col]..self.col_offsets[col + 1]) |pos| builder.observe(self.row_indices[pos], col);
            }
            return builder.lowerProfile();
        }

        pub fn upperProfile(self: Self) SparseError!usize {
            if (self.rows != self.cols) return error.NonMatrixArray;
            var builder = try SparseProfileBuilder.init(self.allocator, self.rows);
            defer builder.deinit(self.allocator);
            for (0..self.cols) |col| {
                for (self.col_offsets[col]..self.col_offsets[col + 1]) |pos| builder.observe(self.row_indices[pos], col);
            }
            return builder.upperProfile();
        }

        pub fn profile(self: Self) SparseError!SparseProfile {
            if (self.rows != self.cols) return error.NonMatrixArray;
            var builder = try SparseProfileBuilder.init(self.allocator, self.rows);
            defer builder.deinit(self.allocator);
            for (0..self.cols) |col| {
                for (self.col_offsets[col]..self.col_offsets[col + 1]) |pos| builder.observe(self.row_indices[pos], col);
            }
            return builder.profile();
        }

        pub fn lowerProfileMeetsBound(self: Self, max_profile: usize) SparseError!bool {
            return (try self.lowerProfile()) <= max_profile;
        }

        pub fn upperProfileMeetsBound(self: Self, max_profile: usize) SparseError!bool {
            return (try self.upperProfile()) <= max_profile;
        }

        pub fn profileMeetsBounds(self: Self, max_lower_profile: usize, max_upper_profile: usize) SparseError!bool {
            const current = try self.profile();
            return current.meetsBounds(max_lower_profile, max_upper_profile);
        }

        pub fn profileTotalMeetsBound(self: Self, max_total_profile: usize) SparseError!bool {
            const current = try self.profile();
            return current.totalMeetsBound(max_total_profile);
        }

        pub fn diagonallyDominant(self: Self) SparseError!bool {
            var canonical = try self.coalesced();
            defer canonical.deinit();
            var coo = try canonical.toCoo();
            defer coo.deinit();
            return sparseDiagonalDominanceFromCanonicalEntries(
                T,
                self.allocator,
                canonical.rows,
                canonical.cols,
                coo.row_indices,
                coo.col_indices,
                coo.values,
                false,
            );
        }

        pub fn diagonalDominanceMargin(self: Self) SparseError!f64 {
            var canonical = try self.coalesced();
            defer canonical.deinit();
            var coo = try canonical.toCoo();
            defer coo.deinit();
            return sparseDiagonalDominanceMarginFromCanonicalEntries(
                T,
                self.allocator,
                canonical.rows,
                canonical.cols,
                coo.row_indices,
                coo.col_indices,
                coo.values,
            );
        }

        pub fn diagonalDominanceMarginMeetsBound(self: Self, min_margin: f64) SparseError!bool {
            var canonical = try self.coalesced();
            defer canonical.deinit();
            var coo = try canonical.toCoo();
            defer coo.deinit();
            return sparseDiagonalDominanceMarginMeetsBoundFromCanonicalEntries(
                T,
                self.allocator,
                canonical.rows,
                canonical.cols,
                coo.row_indices,
                coo.col_indices,
                coo.values,
                min_margin,
            );
        }

        pub fn strictlyDiagonallyDominant(self: Self) SparseError!bool {
            var canonical = try self.coalesced();
            defer canonical.deinit();
            var coo = try canonical.toCoo();
            defer coo.deinit();
            return sparseDiagonalDominanceFromCanonicalEntries(
                T,
                self.allocator,
                canonical.rows,
                canonical.cols,
                coo.row_indices,
                coo.col_indices,
                coo.values,
                true,
            );
        }

        pub fn structurallySymmetric(self: Self) SparseError!bool {
            if (self.rows != self.cols) return error.NonMatrixArray;
            for (0..self.cols) |c| {
                for (self.col_offsets[c]..self.col_offsets[c + 1]) |pos| {
                    if (!self.hasEntry(c, self.row_indices[pos])) return false;
                }
            }
            return true;
        }

        pub fn numericallySymmetric(self: Self, tolerance: T) SparseError!bool {
            ensureNumeric(T);
            if (self.rows != self.cols) return error.NonMatrixArray;
            for (0..self.cols) |c| {
                for (self.col_offsets[c]..self.col_offsets[c + 1]) |pos| {
                    const r = self.row_indices[pos];
                    const value = self.get(r, c) orelse return false;
                    const mirror = self.get(c, r) orelse return false;
                    if (absDifference(T, value, mirror) > tolerance) return false;
                }
            }
            return true;
        }

        pub fn symmetryResidualFrobeniusNorm(self: Self) SparseError!T {
            ensureFloat(T);
            var dense = try self.toDense();
            defer dense.deinit();
            return sparseSymmetryResidualFrobeniusNormFromDense(T, dense.data, self.rows, self.cols);
        }

        pub fn symmetryRelativeResidualFrobeniusNorm(self: Self) SparseError!T {
            const residual = try self.symmetryResidualFrobeniusNorm();
            return residual / @max(oneValue(T), self.frobeniusNorm());
        }

        pub fn symmetryResidualFrobeniusNormMeetsBound(self: Self, max_residual: T) SparseError!bool {
            try validateSparseValueRange(T, zero(T), max_residual);
            return (try self.symmetryResidualFrobeniusNorm()) <= max_residual;
        }

        pub fn symmetryRelativeResidualFrobeniusNormMeetsBound(self: Self, max_relative_residual: T) SparseError!bool {
            try validateSparseValueRange(T, zero(T), max_relative_residual);
            return (try self.symmetryRelativeResidualFrobeniusNorm()) <= max_relative_residual;
        }

        pub fn solveTriangular(self: Self, rhs: array_mod.Array(T), triangle: Triangle, diag_kind: Diagonal) SparseError!array_mod.Array(T) {
            if (self.rows != self.cols) return error.NonMatrixArray;
            if (rhs.shape.len != 1 and rhs.shape.len != 2) return error.InvalidShape;
            if (rhs.shape[0] != self.rows) return error.ShapeMismatch;
            if (comptime T == f64) return self.solveTriangularF64(@as(array_mod.Array(f64), rhs), triangle, diag_kind);
            var csr = try self.toCsr();
            defer csr.deinit();
            return csr.solveTriangular(rhs, triangle, diag_kind);
        }

        fn solveTriangularF64(self: Self, rhs: array_mod.Array(f64), triangle: Triangle, diag_kind: Diagonal) SparseError!array_mod.Array(f64) {
            const veyra_view = try @as(CscMatrix(f64), self).asVeyraView();
            if (rhs.shape.len == 1) {
                var rhs_vec = veyra.Vector(f64).fromSlice(self.allocator, rhs.data) catch return error.BackendFailure;
                defer rhs_vec.deinit();
                var dst = veyra.Vector(f64).zeros(self.allocator, self.rows) catch return error.BackendFailure;
                defer dst.deinit();
                veyra.cscSolveTriangular(f64, veyra_view, rhs_vec.asView(), dst.asMut(), toVeyraTriangle(triangle), toVeyraDiagonal(diag_kind)) catch return error.BackendFailure;
                return array_mod.Array(f64).fromSlice(self.allocator, dst.data, &.{self.rows});
            }
            var rhs_mat = veyra.Matrix(f64).fromSlice(self.allocator, rhs.shape[0], rhs.shape[1], .row_major, rhs.data) catch return error.BackendFailure;
            defer rhs_mat.deinit();
            var dst = veyra.Matrix(f64).zeros(self.allocator, self.rows, rhs.shape[1], .row_major) catch return error.BackendFailure;
            defer dst.deinit();
            veyra.cscSolveTriangularMatrix(f64, veyra_view, rhs_mat.asView(), dst.asMut(), toVeyraTriangle(triangle), toVeyraDiagonal(diag_kind)) catch return error.BackendFailure;
            return array_mod.Array(f64).fromSlice(self.allocator, dst.data, &.{ self.rows, rhs.shape[1] });
        }
    };
}

pub fn cscFromDense(comptime T: type, input: array_mod.Array(T)) SparseError!CscMatrix(T) {
    return CscMatrix(T).fromDense(input);
}

pub fn cscFromDenseNnz(comptime T: type, input: array_mod.Array(T)) SparseError!usize {
    return CscMatrix(T).fromDenseNnz(input);
}

pub fn cscFromDensePruned(comptime T: type, input: array_mod.Array(T), tolerance: T) SparseError!CscMatrix(T) {
    return CscMatrix(T).fromDensePruned(input, tolerance);
}

pub fn cscFromDensePrunedNnz(comptime T: type, input: array_mod.Array(T), tolerance: T) SparseError!usize {
    return CscMatrix(T).fromDensePrunedNnz(input, tolerance);
}

pub fn cooEye(comptime T: type, allocator: std.mem.Allocator, rows: usize, cols: usize) SparseError!CooMatrix(T) {
    return CooMatrix(T).eye(allocator, rows, cols);
}

pub fn cooIdentity(comptime T: type, allocator: std.mem.Allocator, size: usize) SparseError!CooMatrix(T) {
    return CooMatrix(T).identity(allocator, size);
}

pub fn cooFromDiagonal(comptime T: type, allocator: std.mem.Allocator, diagonal_values: []const T, offset: isize) SparseError!CooMatrix(T) {
    return CooMatrix(T).fromDiagonal(allocator, diagonal_values, offset);
}

pub fn cooFromDiagonalNnz(comptime T: type, diagonal_values: []const T, offset: isize) SparseError!usize {
    return CooMatrix(T).fromDiagonalNnz(diagonal_values, offset);
}

pub fn cooFromDiagonalPruned(comptime T: type, allocator: std.mem.Allocator, diagonal_values: []const T, offset: isize, tolerance: T) SparseError!CooMatrix(T) {
    return CooMatrix(T).fromDiagonalPruned(allocator, diagonal_values, offset, tolerance);
}

pub fn cooFromDiagonalPrunedNnz(comptime T: type, diagonal_values: []const T, offset: isize, tolerance: T) SparseError!usize {
    return CooMatrix(T).fromDiagonalPrunedNnz(diagonal_values, offset, tolerance);
}

pub fn cooFromDense(comptime T: type, input: array_mod.Array(T)) SparseError!CooMatrix(T) {
    return CooMatrix(T).fromDense(input);
}

pub fn cooFromDenseNnz(comptime T: type, input: array_mod.Array(T)) SparseError!usize {
    return CooMatrix(T).fromDenseNnz(input);
}

pub fn cooFromDensePruned(comptime T: type, input: array_mod.Array(T), tolerance: T) SparseError!CooMatrix(T) {
    return CooMatrix(T).fromDensePruned(input, tolerance);
}

pub fn cooFromDensePrunedNnz(comptime T: type, input: array_mod.Array(T), tolerance: T) SparseError!usize {
    return CooMatrix(T).fromDensePrunedNnz(input, tolerance);
}

pub fn cooFromSlices(
    comptime T: type,
    allocator: std.mem.Allocator,
    rows: usize,
    cols: usize,
    row_indices: []const usize,
    col_indices: []const usize,
    values: []const T,
) SparseError!CooMatrix(T) {
    return CooMatrix(T).fromSlices(allocator, rows, cols, row_indices, col_indices, values);
}

pub fn cscFromCompressed(
    comptime T: type,
    allocator: std.mem.Allocator,
    rows: usize,
    cols: usize,
    col_offsets: []const usize,
    row_indices: []const usize,
    values: []const T,
) SparseError!CscMatrix(T) {
    return CscMatrix(T).fromCompressedSlices(allocator, rows, cols, col_offsets, row_indices, values);
}

pub fn cscEye(comptime T: type, allocator: std.mem.Allocator, rows: usize, cols: usize) SparseError!CscMatrix(T) {
    return CscMatrix(T).eye(allocator, rows, cols);
}

pub fn cscIdentity(comptime T: type, allocator: std.mem.Allocator, size: usize) SparseError!CscMatrix(T) {
    return CscMatrix(T).identity(allocator, size);
}

pub fn cscFromDiagonal(comptime T: type, allocator: std.mem.Allocator, diagonal_values: []const T, offset: isize) SparseError!CscMatrix(T) {
    return CscMatrix(T).fromDiagonal(allocator, diagonal_values, offset);
}

pub fn cscFromDiagonalNnz(comptime T: type, diagonal_values: []const T, offset: isize) SparseError!usize {
    return CscMatrix(T).fromDiagonalNnz(diagonal_values, offset);
}

pub fn cscFromDiagonalPruned(comptime T: type, allocator: std.mem.Allocator, diagonal_values: []const T, offset: isize, tolerance: T) SparseError!CscMatrix(T) {
    return CscMatrix(T).fromDiagonalPruned(allocator, diagonal_values, offset, tolerance);
}

pub fn cscFromDiagonalPrunedNnz(comptime T: type, diagonal_values: []const T, offset: isize, tolerance: T) SparseError!usize {
    return CscMatrix(T).fromDiagonalPrunedNnz(diagonal_values, offset, tolerance);
}

pub fn csrFromDense(comptime T: type, input: array_mod.Array(T)) SparseError!CsrMatrix(T) {
    return CsrMatrix(T).fromDense(input);
}

pub fn csrFromDenseNnz(comptime T: type, input: array_mod.Array(T)) SparseError!usize {
    return CsrMatrix(T).fromDenseNnz(input);
}

pub fn csrFromDensePruned(comptime T: type, input: array_mod.Array(T), tolerance: T) SparseError!CsrMatrix(T) {
    return CsrMatrix(T).fromDensePruned(input, tolerance);
}

pub fn csrFromDensePrunedNnz(comptime T: type, input: array_mod.Array(T), tolerance: T) SparseError!usize {
    return CsrMatrix(T).fromDensePrunedNnz(input, tolerance);
}

pub fn csrFromCompressed(
    comptime T: type,
    allocator: std.mem.Allocator,
    rows: usize,
    cols: usize,
    row_offsets: []const usize,
    col_indices: []const usize,
    values: []const T,
) SparseError!CsrMatrix(T) {
    return CsrMatrix(T).fromCompressedSlices(allocator, rows, cols, row_offsets, col_indices, values);
}

pub fn csrEye(comptime T: type, allocator: std.mem.Allocator, rows: usize, cols: usize) SparseError!CsrMatrix(T) {
    return CsrMatrix(T).eye(allocator, rows, cols);
}

pub fn csrIdentity(comptime T: type, allocator: std.mem.Allocator, size: usize) SparseError!CsrMatrix(T) {
    return CsrMatrix(T).identity(allocator, size);
}

pub fn csrFromDiagonal(comptime T: type, allocator: std.mem.Allocator, diagonal_values: []const T, offset: isize) SparseError!CsrMatrix(T) {
    return CsrMatrix(T).fromDiagonal(allocator, diagonal_values, offset);
}

pub fn csrFromDiagonalNnz(comptime T: type, diagonal_values: []const T, offset: isize) SparseError!usize {
    return CsrMatrix(T).fromDiagonalNnz(diagonal_values, offset);
}

pub fn csrFromDiagonalPruned(comptime T: type, allocator: std.mem.Allocator, diagonal_values: []const T, offset: isize, tolerance: T) SparseError!CsrMatrix(T) {
    return CsrMatrix(T).fromDiagonalPruned(allocator, diagonal_values, offset, tolerance);
}

pub fn csrFromDiagonalPrunedNnz(comptime T: type, diagonal_values: []const T, offset: isize, tolerance: T) SparseError!usize {
    return CsrMatrix(T).fromDiagonalPrunedNnz(diagonal_values, offset, tolerance);
}

test "sparse eye and identity constructors" {
    const gpa = std.testing.allocator;

    var coo_eye = try cooEye(f64, gpa, 2, 4);
    defer coo_eye.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, coo_eye.row_indices);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, coo_eye.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 1, 1 }, coo_eye.values);
    var coo_eye_dense = try coo_eye.toDense();
    defer coo_eye_dense.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 0, 0, 0, 0, 1, 0, 0 }, coo_eye_dense.data);

    var coo_identity = try cooIdentity(i32, gpa, 3);
    defer coo_identity.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2 }, coo_identity.row_indices);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2 }, coo_identity.col_indices);
    try std.testing.expectEqualSlices(i32, &.{ 1, 1, 1 }, coo_identity.values);

    var csr_eye = try csrEye(f64, gpa, 4, 2);
    defer csr_eye.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2, 2, 2 }, csr_eye.row_offsets);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, csr_eye.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 1, 1 }, csr_eye.values);
    var csr_eye_dense = try csr_eye.toDense();
    defer csr_eye_dense.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 0, 0, 1, 0, 0, 0, 0 }, csr_eye_dense.data);

    var csc_eye = try cscEye(f64, gpa, 2, 4);
    defer csc_eye.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2, 2, 2 }, csc_eye.col_offsets);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, csc_eye.row_indices);
    try std.testing.expectEqualSlices(f64, &.{ 1, 1 }, csc_eye.values);
    var csc_eye_dense = try csc_eye.toDense();
    defer csc_eye_dense.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 0, 0, 0, 0, 1, 0, 0 }, csc_eye_dense.data);

    var csc_identity = try cscIdentity(f64, gpa, 0);
    defer csc_identity.deinit();
    try std.testing.expectEqualSlices(usize, &.{0}, csc_identity.col_offsets);
    try std.testing.expectEqual(@as(usize, 0), csc_identity.nnz());
    try std.testing.expectError(error.EmptyArray, csc_identity.minValue());
    try std.testing.expectError(error.EmptyArray, csc_identity.mean());
    try std.testing.expectError(error.EmptyArray, csc_identity.variance(0));
    try std.testing.expectError(error.EmptyArray, csc_identity.columnVariances(0));

    var upper_diag = try cooFromDiagonal(f64, gpa, &.{ 2, 0, 3 }, 2);
    defer upper_diag.deinit();
    try std.testing.expectEqual(@as(usize, 2), try cooFromDiagonalNnz(f64, &.{ 2, 0, 3 }, 2));
    try std.testing.expectEqual(@as(usize, 5), upper_diag.rows);
    try std.testing.expectEqual(@as(usize, 5), upper_diag.cols);
    try std.testing.expectEqualSlices(usize, &.{ 0, 2 }, upper_diag.row_indices);
    try std.testing.expectEqualSlices(usize, &.{ 2, 4 }, upper_diag.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 2, 3 }, upper_diag.values);
    var upper_dense = try upper_diag.toDense();
    defer upper_dense.deinit();
    try std.testing.expectEqualSlices(f64, &.{
        0, 0, 2, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 3,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    }, upper_dense.data);
    var upper_pruned = try cooFromDiagonalPruned(f64, gpa, &.{ 2, 0, 3 }, 2, 2);
    defer upper_pruned.deinit();
    try std.testing.expectEqual(@as(usize, 1), try cooFromDiagonalPrunedNnz(f64, &.{ 2, 0, 3 }, 2, 2));
    try std.testing.expectEqualSlices(usize, &.{2}, upper_pruned.row_indices);
    try std.testing.expectEqualSlices(usize, &.{4}, upper_pruned.col_indices);
    try std.testing.expectEqualSlices(f64, &.{3}, upper_pruned.values);
    try std.testing.expectError(error.InvalidShape, cooFromDiagonalPrunedNnz(f64, &.{ 2, 0, 3 }, 2, std.math.nan(f64)));
    try std.testing.expectError(error.InvalidShape, upper_diag.setDiagonal(1));
    var upper_offset_diag = try upper_diag.diagonalOffset(2);
    defer upper_offset_diag.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 2, 0, 3 }, upper_offset_diag.data);
    var upper_alias_diag = try upper_diag.diagonalWithOffset(2);
    defer upper_alias_diag.deinit();
    try std.testing.expectEqualSlices(f64, upper_offset_diag.data, upper_alias_diag.data);
    var upper_empty_diag = try upper_diag.diagonalOffset(5);
    defer upper_empty_diag.deinit();
    try std.testing.expectEqual(@as(usize, 0), upper_empty_diag.data.len);

    var lower_csr = try csrFromDiagonal(f64, gpa, &.{ 4, 5 }, -1);
    defer lower_csr.deinit();
    try std.testing.expectEqual(@as(usize, 2), try csrFromDiagonalNnz(f64, &.{ 4, 5 }, -1));
    try std.testing.expectEqualSlices(usize, &.{ 0, 0, 1, 2 }, lower_csr.row_offsets);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, lower_csr.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 4, 5 }, lower_csr.values);
    var lower_csr_pruned = try csrFromDiagonalPruned(f64, gpa, &.{ 4, 5 }, -1, 4);
    defer lower_csr_pruned.deinit();
    try std.testing.expectEqual(@as(usize, 1), try csrFromDiagonalPrunedNnz(f64, &.{ 4, 5 }, -1, 4));
    try std.testing.expectEqualSlices(usize, &.{ 0, 0, 0, 1 }, lower_csr_pruned.row_offsets);
    try std.testing.expectEqualSlices(usize, &.{1}, lower_csr_pruned.col_indices);
    try std.testing.expectEqualSlices(f64, &.{5}, lower_csr_pruned.values);
    try std.testing.expectError(error.InvalidShape, csrFromDiagonalPrunedNnz(f64, &.{ 4, 5 }, -1, std.math.inf(f64)));
    try std.testing.expectError(error.InvalidShape, lower_csr.addToDiagonal(1));
    var lower_csr_offset_diag = try lower_csr.diagonalOffset(-1);
    defer lower_csr_offset_diag.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 4, 5 }, lower_csr_offset_diag.data);
    var lower_csr_upper_empty = try lower_csr.diagonalOffset(3);
    defer lower_csr_upper_empty.deinit();
    try std.testing.expectEqual(@as(usize, 0), lower_csr_upper_empty.data.len);
    var lower_csc = try cscFromDiagonal(f64, gpa, &.{ 4, 5 }, -1);
    defer lower_csc.deinit();
    try std.testing.expectEqual(@as(usize, 2), try cscFromDiagonalNnz(f64, &.{ 4, 5 }, -1));
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2, 2 }, lower_csc.col_offsets);
    try std.testing.expectEqualSlices(usize, &.{ 1, 2 }, lower_csc.row_indices);
    try std.testing.expectEqualSlices(f64, &.{ 4, 5 }, lower_csc.values);
    var lower_csc_pruned = try cscFromDiagonalPruned(f64, gpa, &.{ 4, 5 }, -1, 4);
    defer lower_csc_pruned.deinit();
    try std.testing.expectEqual(@as(usize, 1), try cscFromDiagonalPrunedNnz(f64, &.{ 4, 5 }, -1, 4));
    try std.testing.expectEqualSlices(usize, &.{ 0, 0, 1, 1 }, lower_csc_pruned.col_offsets);
    try std.testing.expectEqualSlices(usize, &.{2}, lower_csc_pruned.row_indices);
    try std.testing.expectEqualSlices(f64, &.{5}, lower_csc_pruned.values);
    try std.testing.expectError(error.InvalidShape, cscFromDiagonalPrunedNnz(f64, &.{ 4, 5 }, -1, -1));
    try std.testing.expectError(error.InvalidShape, lower_csc.setDiagonal(1));
    var lower_csc_offset_diag = try lower_csc.diagonalWithOffset(-1);
    defer lower_csc_offset_diag.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 4, 5 }, lower_csc_offset_diag.data);
    var lower_csc_upper_empty = try lower_csc.diagonalOffset(3);
    defer lower_csc_upper_empty.deinit();
    try std.testing.expectEqual(@as(usize, 0), lower_csc_upper_empty.data.len);
}

test "coo sparse dense roundtrip and compressed conversions" {
    const gpa = std.testing.allocator;
    var dense = try array_mod.Array(f64).fromSlice(gpa, &.{
        10, 0, 2, 0,
        0,  3, 0, 4,
        5,  0, 0, 6,
    }, &.{ 3, 4 });
    defer dense.deinit();

    var coo = try cooFromDense(f64, dense);
    defer coo.deinit();
    try std.testing.expectEqual(@as(usize, 6), try cooFromDenseNnz(f64, dense));
    try std.testing.expectEqual(@as(usize, 6), coo.nnz());
    try std.testing.expectEqualSlices(usize, &.{ 0, 0, 1, 1, 2, 2 }, coo.row_indices);
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 1, 3, 0, 3 }, coo.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 10, 2, 3, 4, 5, 6 }, coo.values);
    var coo_clone = try coo.clone();
    defer coo_clone.deinit();
    try std.testing.expectEqualSlices(usize, coo.row_indices, coo_clone.row_indices);
    try std.testing.expectEqualSlices(usize, coo.col_indices, coo_clone.col_indices);
    try std.testing.expectEqualSlices(f64, coo.values, coo_clone.values);
    coo_clone.values[0] = 99;
    try std.testing.expectEqual(@as(f64, 10), coo.values[0]);
    var coo_i32 = try coo.astype(i32);
    defer coo_i32.deinit();
    try std.testing.expectEqualSlices(usize, coo.row_indices, coo_i32.row_indices);
    try std.testing.expectEqualSlices(usize, coo.col_indices, coo_i32.col_indices);
    try std.testing.expectEqualSlices(i32, &.{ 10, 2, 3, 4, 5, 6 }, coo_i32.values);
    var coo_mapped = try coo.mapValues(f64, struct {
        fn f(value: f64) f64 {
            return value + 1;
        }
    }.f);
    defer coo_mapped.deinit();
    try std.testing.expectEqualSlices(usize, coo.row_indices, coo_mapped.row_indices);
    try std.testing.expectEqualSlices(usize, coo.col_indices, coo_mapped.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 11, 3, 4, 5, 6, 7 }, coo_mapped.values);
    var coo_equal_scalar = try coo.equalScalar(3);
    defer coo_equal_scalar.deinit();
    try std.testing.expectEqualSlices(usize, coo.row_indices, coo_equal_scalar.row_indices);
    try std.testing.expectEqualSlices(usize, coo.col_indices, coo_equal_scalar.col_indices);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, false, false }, coo_equal_scalar.values);
    var coo_gt_scalar = try coo.gtScalar(4);
    defer coo_gt_scalar.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false, true, true }, coo_gt_scalar.values);
    var coo_filled = try coo.clone();
    defer coo_filled.deinit();
    coo_filled.fillStoredValues(-2);
    try std.testing.expectEqualSlices(usize, coo.row_indices, coo_filled.row_indices);
    try std.testing.expectEqualSlices(usize, coo.col_indices, coo_filled.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ -2, -2, -2, -2, -2, -2 }, coo_filled.values);
    coo_filled.mapValuesInPlace(struct {
        fn f(value: f64) f64 {
            return value * 3;
        }
    }.f);
    try std.testing.expectEqualSlices(f64, &.{ -6, -6, -6, -6, -6, -6 }, coo_filled.values);
    coo_filled.scaleStoredValues(0.5);
    try std.testing.expectEqualSlices(f64, &.{ -3, -3, -3, -3, -3, -3 }, coo_filled.values);
    var source_dense = try array_mod.Array(f64).fromSlice(gpa, &.{
        1, 2,  3,  4,
        5, 6,  7,  8,
        9, 10, 11, 12,
    }, &.{ 3, 4 });
    defer source_dense.deinit();
    try coo_filled.setStoredValuesFromDense(source_dense);
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 6, 8, 9, 12 }, coo_filled.values);
    var wrong_dense_shape = try array_mod.Array(f64).zeros(gpa, &.{ 2, 2 });
    defer wrong_dense_shape.deinit();
    try std.testing.expectError(error.ShapeMismatch, coo_filled.setStoredValuesFromDense(wrong_dense_shape));
    try std.testing.expectApproxEqAbs(@as(f64, 30), coo.sum(), 1e-12);
    try std.testing.expect(try coo.sumInRange(30, 30));
    try std.testing.expect(try coo.sumInRange(29.5, 30.5));
    try std.testing.expect(!(try coo.sumInRange(30.5, 31)));
    try std.testing.expectError(error.InvalidShape, coo.sumInRange(31, 30));
    try std.testing.expectError(error.InvalidShape, coo.sumInRange(std.math.inf(f64), 31));
    try std.testing.expectApproxEqAbs(@as(f64, 30), coo.absSum(), 1e-12);
    try std.testing.expect(try coo.absSumInRange(30, 30));
    try std.testing.expect(try coo.absSumInRange(29, 31));
    try std.testing.expect(!(try coo.absSumInRange(31, 32)));
    try std.testing.expectError(error.InvalidShape, coo.absSumInRange(31, 30));
    try std.testing.expectApproxEqAbs(@as(f64, @sqrt(190.0)), coo.frobeniusNorm(), 1e-12);
    try std.testing.expect(try coo.frobeniusNormMeetsBound(@sqrt(@as(f64, 190.0))));
    try std.testing.expect(!(try coo.frobeniusNormMeetsBound(@sqrt(@as(f64, 190.0)) - 1e-12)));
    try std.testing.expectError(error.InvalidShape, coo.frobeniusNormMeetsBound(std.math.nan(f64)));
    try std.testing.expectApproxEqAbs(@as(f64, 15), try coo.oneNorm(), 1e-12);
    try std.testing.expect(try coo.oneNormMeetsBound(15));
    try std.testing.expect(!(try coo.oneNormMeetsBound(14.999)));
    try std.testing.expectError(error.InvalidShape, coo.oneNormMeetsBound(-1));
    try std.testing.expectApproxEqAbs(@as(f64, 12), try coo.infNorm(), 1e-12);
    try std.testing.expect(try coo.infNormMeetsBound(12));
    try std.testing.expect(!(try coo.infNormMeetsBound(11.999)));
    try std.testing.expectError(error.InvalidShape, coo.infNormMeetsBound(std.math.inf(f64)));
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), try coo.density(), 1e-12);

    var coo_pruned_dense = try cooFromDensePruned(f64, dense, 4);
    defer coo_pruned_dense.deinit();
    try std.testing.expectEqual(@as(usize, 3), try cooFromDensePrunedNnz(f64, dense, 4));
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 2 }, coo_pruned_dense.row_indices);
    try std.testing.expectEqualSlices(usize, &.{ 0, 0, 3 }, coo_pruned_dense.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 10, 5, 6 }, coo_pruned_dense.values);
    try std.testing.expectEqual(@as(usize, 3), try coo_pruned_dense.pruneZerosNnz(4));
    try std.testing.expectError(error.InvalidShape, cooFromDensePruned(f64, dense, std.math.nan(f64)));
    try std.testing.expectError(error.InvalidShape, cooFromDensePrunedNnz(f64, dense, std.math.nan(f64)));

    var dense_roundtrip = try coo.toDense();
    defer dense_roundtrip.deinit();
    try std.testing.expectEqualSlices(f64, dense.data, dense_roundtrip.data);

    var x = try array_mod.Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4 }, &.{4});
    defer x.deinit();
    var y = try coo.matvec(x);
    defer y.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 16, 22, 29 }, y.data);

    var tx_rhs = try array_mod.Array(f64).fromSlice(gpa, &.{ 1, 2, 3 }, &.{3});
    defer tx_rhs.deinit();
    var tx = try coo.transposeMatvec(tx_rhs);
    defer tx.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 25, 6, 2, 26 }, tx.data);

    var rhs = try array_mod.Array(f64).fromSlice(gpa, &.{
        1, 2,
        2, 4,
        3, 6,
        4, 8,
    }, &.{ 4, 2 });
    defer rhs.deinit();
    var product = try coo.matmat(rhs);
    defer product.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 16, 32, 22, 44, 29, 58 }, product.data);

    var transpose_rhs = try array_mod.Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 3, 2 });
    defer transpose_rhs.deinit();
    var transpose_product = try coo.transposeMatmat(transpose_rhs);
    defer transpose_product.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 35, 50, 9, 12, 2, 4, 42, 52 }, transpose_product.data);

    var transposed = try coo.transpose();
    defer transposed.deinit();
    try std.testing.expectEqual(@as(usize, 4), transposed.rows);
    try std.testing.expectEqual(@as(usize, 3), transposed.cols);
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 1, 3, 0, 3 }, transposed.row_indices);
    try std.testing.expectEqualSlices(usize, &.{ 0, 0, 1, 1, 2, 2 }, transposed.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 10, 2, 3, 4, 5, 6 }, transposed.values);
    var transposed_dense = try transposed.toDense();
    defer transposed_dense.deinit();
    try std.testing.expectEqualSlices(f64, &.{
        10, 0, 5,
        0,  3, 0,
        2,  0, 0,
        0,  4, 6,
    }, transposed_dense.data);
    var coo_mt = try coo.matrixTranspose();
    defer coo_mt.deinit();
    var coo_mt_dense = try coo_mt.toDense();
    defer coo_mt_dense.deinit();
    try std.testing.expectEqualSlices(f64, transposed_dense.data, coo_mt_dense.data);
    var coo_m_t = try coo.mT();
    defer coo_m_t.deinit();
    var coo_m_t_dense = try coo_m_t.toDense();
    defer coo_m_t_dense.deinit();
    try std.testing.expectEqualSlices(f64, transposed_dense.data, coo_m_t_dense.data);

    var csr = try coo.toCsr();
    defer csr.deinit();
    try std.testing.expectEqual(@as(usize, 6), try csrFromDenseNnz(f64, dense));
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 4, 6 }, csr.row_offsets);
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 1, 3, 0, 3 }, csr.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 10, 2, 3, 4, 5, 6 }, csr.values);
    var csr_copy = try csr.copy();
    defer csr_copy.deinit();
    try std.testing.expectEqualSlices(usize, csr.row_offsets, csr_copy.row_offsets);
    try std.testing.expectEqualSlices(usize, csr.col_indices, csr_copy.col_indices);
    try std.testing.expectEqualSlices(f64, csr.values, csr_copy.values);
    csr_copy.values[0] = 99;
    try std.testing.expectEqual(@as(f64, 10), csr.values[0]);
    var csr_f32 = try csr.astype(f32);
    defer csr_f32.deinit();
    try std.testing.expectEqualSlices(usize, csr.row_offsets, csr_f32.row_offsets);
    try std.testing.expectEqualSlices(usize, csr.col_indices, csr_f32.col_indices);
    try std.testing.expectEqualSlices(f32, &.{ 10, 2, 3, 4, 5, 6 }, csr_f32.values);
    var csr_mapped = try csr.mapStoredValues(f64, struct {
        fn f(value: f64) f64 {
            return value * 2;
        }
    }.f);
    defer csr_mapped.deinit();
    try std.testing.expectEqualSlices(usize, csr.row_offsets, csr_mapped.row_offsets);
    try std.testing.expectEqualSlices(usize, csr.col_indices, csr_mapped.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 20, 4, 6, 8, 10, 12 }, csr_mapped.values);
    var csr_le_scalar = try csr.lessEqualScalar(4);
    defer csr_le_scalar.deinit();
    try std.testing.expectEqualSlices(usize, csr.row_offsets, csr_le_scalar.row_offsets);
    try std.testing.expectEqualSlices(usize, csr.col_indices, csr_le_scalar.col_indices);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, false, false }, csr_le_scalar.values);
    var csr_ne_scalar = try csr.neScalar(3);
    defer csr_ne_scalar.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true, true, true }, csr_ne_scalar.values);
    var csr_filled = try csr.clone();
    defer csr_filled.deinit();
    csr_filled.fillValues(7);
    try std.testing.expectEqualSlices(usize, csr.row_offsets, csr_filled.row_offsets);
    try std.testing.expectEqualSlices(usize, csr.col_indices, csr_filled.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 7, 7, 7, 7, 7, 7 }, csr_filled.values);
    csr_filled.mapStoredValuesInPlace(struct {
        fn f(value: f64) f64 {
            return value - 2;
        }
    }.f);
    try std.testing.expectEqualSlices(f64, &.{ 5, 5, 5, 5, 5, 5 }, csr_filled.values);
    csr_filled.scaleValues(2);
    try std.testing.expectEqualSlices(f64, &.{ 10, 10, 10, 10, 10, 10 }, csr_filled.values);
    try csr_filled.setStoredValuesFromDense(source_dense);
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 6, 8, 9, 12 }, csr_filled.values);
    try std.testing.expectError(error.ShapeMismatch, csr_filled.setStoredValuesFromDense(wrong_dense_shape));
    var csr_pruned_dense = try csrFromDensePruned(f64, dense, 4);
    defer csr_pruned_dense.deinit();
    try std.testing.expectEqual(@as(usize, 3), try csrFromDensePrunedNnz(f64, dense, 4));
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 1, 3 }, csr_pruned_dense.row_offsets);
    try std.testing.expectEqualSlices(usize, &.{ 0, 0, 3 }, csr_pruned_dense.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 10, 5, 6 }, csr_pruned_dense.values);
    try std.testing.expectError(error.InvalidShape, csrFromDensePruned(f64, dense, std.math.inf(f64)));
    try std.testing.expectError(error.InvalidShape, csrFromDensePrunedNnz(f64, dense, std.math.inf(f64)));

    var csc = try coo.toCsc();
    defer csc.deinit();
    try std.testing.expectEqual(@as(usize, 6), try cscFromDenseNnz(f64, dense));
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 3, 4, 6 }, csc.col_offsets);
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 1, 0, 1, 2 }, csc.row_indices);
    try std.testing.expectEqualSlices(f64, &.{ 10, 5, 3, 2, 4, 6 }, csc.values);
    var csc_detached = try csc.detach();
    defer csc_detached.deinit();
    try std.testing.expectEqualSlices(usize, csc.col_offsets, csc_detached.col_offsets);
    try std.testing.expectEqualSlices(usize, csc.row_indices, csc_detached.row_indices);
    try std.testing.expectEqualSlices(f64, csc.values, csc_detached.values);
    csc_detached.values[0] = 99;
    try std.testing.expectEqual(@as(f64, 10), csc.values[0]);
    var csc_bool = try csc.astype(bool);
    defer csc_bool.deinit();
    try std.testing.expectEqualSlices(usize, csc.col_offsets, csc_bool.col_offsets);
    try std.testing.expectEqualSlices(usize, csc.row_indices, csc_bool.row_indices);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, true, true }, csc_bool.values);
    var csc_mapped = try csc.mapValues(f64, struct {
        fn f(value: f64) f64 {
            return value - 1;
        }
    }.f);
    defer csc_mapped.deinit();
    try std.testing.expectEqualSlices(usize, csc.col_offsets, csc_mapped.col_offsets);
    try std.testing.expectEqualSlices(usize, csc.row_indices, csc_mapped.row_indices);
    try std.testing.expectEqualSlices(f64, &.{ 9, 4, 2, 1, 3, 5 }, csc_mapped.values);
    var csc_ne_scalar = try csc.notEqualScalar(3);
    defer csc_ne_scalar.deinit();
    try std.testing.expectEqualSlices(usize, csc.col_offsets, csc_ne_scalar.col_offsets);
    try std.testing.expectEqualSlices(usize, csc.row_indices, csc_ne_scalar.row_indices);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true, true, true }, csc_ne_scalar.values);
    var csc_ge_scalar = try csc.geScalar(5);
    defer csc_ge_scalar.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false, false, true }, csc_ge_scalar.values);
    var csc_filled = try csc.clone();
    defer csc_filled.deinit();
    csc_filled.fillStoredValues(8);
    try std.testing.expectEqualSlices(usize, csc.col_offsets, csc_filled.col_offsets);
    try std.testing.expectEqualSlices(usize, csc.row_indices, csc_filled.row_indices);
    try std.testing.expectEqualSlices(f64, &.{ 8, 8, 8, 8, 8, 8 }, csc_filled.values);
    csc_filled.mapValuesInPlace(struct {
        fn f(value: f64) f64 {
            return value / 2;
        }
    }.f);
    try std.testing.expectEqualSlices(f64, &.{ 4, 4, 4, 4, 4, 4 }, csc_filled.values);
    csc_filled.scaleStoredValues(-0.25);
    try std.testing.expectEqualSlices(f64, &.{ -1, -1, -1, -1, -1, -1 }, csc_filled.values);
    try csc_filled.setStoredValuesFromDense(source_dense);
    try std.testing.expectEqualSlices(f64, &.{ 1, 9, 6, 3, 8, 12 }, csc_filled.values);
    try std.testing.expectError(error.ShapeMismatch, csc_filled.setStoredValuesFromDense(wrong_dense_shape));
    var csc_pruned_dense = try cscFromDensePruned(f64, dense, 4);
    defer csc_pruned_dense.deinit();
    try std.testing.expectEqual(@as(usize, 3), try cscFromDensePrunedNnz(f64, dense, 4));
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 2, 2, 3 }, csc_pruned_dense.col_offsets);
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 2 }, csc_pruned_dense.row_indices);
    try std.testing.expectEqualSlices(f64, &.{ 10, 5, 6 }, csc_pruned_dense.values);
    try std.testing.expectError(error.InvalidShape, cscFromDensePruned(f64, dense, -1));
    try std.testing.expectError(error.InvalidShape, cscFromDensePrunedNnz(f64, dense, -1));

    var coo_from_csr = try csr.toCoo();
    defer coo_from_csr.deinit();
    try std.testing.expectEqualSlices(usize, coo.row_indices, coo_from_csr.row_indices);
    try std.testing.expectEqualSlices(usize, coo.col_indices, coo_from_csr.col_indices);
    try std.testing.expectEqualSlices(f64, coo.values, coo_from_csr.values);

    var coo_from_csc = try csc.toCoo();
    defer coo_from_csc.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 1, 0, 1, 2 }, coo_from_csc.row_indices);
    try std.testing.expectEqualSlices(usize, &.{ 0, 0, 1, 2, 3, 3 }, coo_from_csc.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 10, 5, 3, 2, 4, 6 }, coo_from_csc.values);
    var csc_coo_dense = try coo_from_csc.toDense();
    defer csc_coo_dense.deinit();
    try std.testing.expectEqualSlices(f64, dense.data, csc_coo_dense.data);

    var manual = try cooFromSlices(f64, gpa, 2, 3, &.{ 0, 1, 1 }, &.{ 2, 0, 2 }, &.{ 4.0, 5.0, 6.0 });
    defer manual.deinit();
    var manual_dense = try manual.toDense();
    defer manual_dense.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 0, 4, 5, 0, 6 }, manual_dense.data);

    var duplicate = try cooFromSlices(f64, gpa, 2, 2, &.{ 0, 0, 1 }, &.{ 1, 1, 0 }, &.{ 2.0, 3.0, 4.0 });
    defer duplicate.deinit();
    var duplicate_dense = try duplicate.toDense();
    defer duplicate_dense.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 5, 4, 0 }, duplicate_dense.data);

    var unsorted_duplicate = try cooFromSlices(f64, gpa, 2, 2, &.{ 1, 0, 1, 0 }, &.{ 0, 1, 0, 1 }, &.{ 1.0, 2.0, 3.0, 4.0 });
    defer unsorted_duplicate.deinit();
    var coalesced = try unsorted_duplicate.coalesced();
    defer coalesced.deinit();
    try std.testing.expectEqual(@as(usize, 2), coalesced.nnz());
    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, coalesced.row_indices);
    try std.testing.expectEqualSlices(usize, &.{ 1, 0 }, coalesced.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 6, 4 }, coalesced.values);
    var coalesced_dense = try coalesced.toDense();
    defer coalesced_dense.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 6, 4, 0 }, coalesced_dense.data);

    var duplicate_csr = try csrFromCompressed(f64, gpa, 1, 2, &.{ 0, 2 }, &.{ 1, 1 }, &.{ 2.0, 3.0 });
    defer duplicate_csr.deinit();
    var duplicate_csr_dense = try duplicate_csr.toDense();
    defer duplicate_csr_dense.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 5 }, duplicate_csr_dense.data);

    var duplicate_csc = try cscFromCompressed(f64, gpa, 1, 2, &.{ 0, 0, 2 }, &.{ 0, 0 }, &.{ 2.0, 3.0 });
    defer duplicate_csc.deinit();
    var duplicate_csc_dense = try duplicate_csc.toDense();
    defer duplicate_csc_dense.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 5 }, duplicate_csc_dense.data);
}

test "coo sparse row and column statistics" {
    const gpa = std.testing.allocator;
    var dense = try array_mod.Array(f64).fromSlice(gpa, &.{
        1, 0, -2,
        0, 3, 0,
        4, 0, 5,
    }, &.{ 3, 3 });
    defer dense.deinit();
    var coo = try cooFromDense(f64, dense);
    defer coo.deinit();

    try std.testing.expectApproxEqAbs(@as(f64, -2), try coo.minValue(), 1e-12);
    try std.testing.expectEqual(@as(usize, 1), try coo.minValueIndex());
    try std.testing.expectApproxEqAbs(@as(f64, 5), try coo.maxValue(), 1e-12);
    try std.testing.expectEqual(@as(usize, 4), try coo.maxValueIndex());
    try std.testing.expectApproxEqAbs(@as(f64, 1), try coo.minAbsValue(), 1e-12);
    try std.testing.expectEqual(@as(usize, 0), try coo.minAbsValueIndex());
    try std.testing.expectApproxEqAbs(@as(f64, 5), try coo.maxAbsValue(), 1e-12);
    try std.testing.expectEqual(@as(usize, 4), try coo.maxAbsValueIndex());
    const expected_mean = @as(f64, 11.0 / 9.0);
    const expected_variance = @as(f64, 55.0 / 9.0) - expected_mean * expected_mean;
    const expected_stddev = @sqrt(expected_variance);
    const expected_sample_variance = @as(f64, (55.0 - (11.0 * 11.0) / 9.0) / 8.0);
    const expected_sample_stddev = @sqrt(expected_sample_variance);
    try std.testing.expectApproxEqAbs(expected_mean, try coo.mean(), 1e-12);
    try std.testing.expect(try coo.meanInRange(expected_mean, expected_mean));
    try std.testing.expect(try coo.meanInRange(1.2, 1.3));
    try std.testing.expect(!(try coo.meanInRange(1.3, 1.4)));
    try std.testing.expectError(error.InvalidShape, coo.meanInRange(2, 1));
    try std.testing.expectApproxEqAbs(expected_variance, try coo.variance(0), 1e-12);
    try std.testing.expect(try coo.varianceInRange(0, expected_variance - 1e-12, expected_variance + 1e-12));
    try std.testing.expect(try coo.varianceInRange(0, 4.6, 4.7));
    try std.testing.expect(!(try coo.varianceInRange(0, 4.7, 4.8)));
    try std.testing.expectError(error.InvalidShape, coo.varianceInRange(0, -0.1, 1));
    try std.testing.expectApproxEqAbs(expected_stddev, try coo.stddev(0), 1e-12);
    try std.testing.expect(try coo.stddevInRange(0, expected_stddev - 1e-12, expected_stddev + 1e-12));
    try std.testing.expect(!(try coo.stddevInRange(0, expected_stddev + 0.1, expected_stddev + 0.2)));
    try std.testing.expectError(error.InvalidShape, coo.stddevInRange(0, -0.1, expected_stddev));
    try std.testing.expectApproxEqAbs(expected_sample_variance, try coo.sampleVariance(), 1e-12);
    try std.testing.expect(try coo.sampleVarianceInRange(expected_sample_variance - 1e-12, expected_sample_variance + 1e-12));
    try std.testing.expect(!(try coo.sampleVarianceInRange(expected_sample_variance + 0.1, expected_sample_variance + 0.2)));
    try std.testing.expectApproxEqAbs(expected_sample_stddev, try coo.sampleStddev(), 1e-12);
    try std.testing.expect(try coo.sampleStddevInRange(expected_sample_stddev - 1e-12, expected_sample_stddev + 1e-12));
    try std.testing.expectError(error.InvalidShape, coo.sampleStddevInRange(2, 1));

    var row_vars = try coo.rowVariances(0);
    defer row_vars.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 14.0 / 9.0), row_vars.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2), row_vars.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 14.0 / 3.0), row_vars.data[2], 1e-12);
    try std.testing.expect(try coo.rowVariancesInRange(0, 14.0 / 9.0, 14.0 / 3.0));
    try std.testing.expect(!(try coo.rowVariancesInRange(0, 2.1, 14.0 / 3.0)));
    try std.testing.expectError(error.InvalidShape, coo.rowVariancesInRange(0, -0.1, 1));
    var row_sample_vars = try coo.rowSampleVariances();
    defer row_sample_vars.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 7.0 / 3.0), row_sample_vars.data[0], 1e-12);
    try std.testing.expect(try coo.rowSampleVariancesInRange(7.0 / 3.0, 7));
    try std.testing.expect(!(try coo.rowSampleVariancesInRange(2.5, 7)));
    try std.testing.expect(try coo.rowSampleStddevsInRange(@sqrt(7.0 / 3.0), @sqrt(7.0)));
    try std.testing.expect(!(try coo.rowSampleStddevsInRange(1.6, @sqrt(7.0))));
    var col_vars = try coo.columnVariances(0);
    defer col_vars.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 26.0 / 9.0), col_vars.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2), col_vars.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 26.0 / 3.0), col_vars.data[2], 1e-12);
    try std.testing.expect(try coo.columnVariancesInRange(0, 2, 26.0 / 3.0));
    try std.testing.expect(!(try coo.columnVariancesInRange(0, 2.1, 26.0 / 3.0)));
    try std.testing.expect(try coo.columnSampleVariancesInRange(3, 13));
    try std.testing.expect(!(try coo.columnSampleVariancesInRange(4, 13)));
    try std.testing.expect(try coo.columnSampleStddevsInRange(@sqrt(3.0), @sqrt(13.0)));
    try std.testing.expect(!(try coo.columnSampleStddevsInRange(1.8, @sqrt(13.0))));
    var row_stds = try coo.rowStddevs(0);
    defer row_stds.deinit();
    try std.testing.expectApproxEqAbs(@sqrt(14.0 / 9.0), row_stds.data[0], 1e-12);
    try std.testing.expect(try coo.rowStddevsInRange(0, @sqrt(14.0 / 9.0), @sqrt(14.0 / 3.0)));
    try std.testing.expect(!(try coo.rowStddevsInRange(0, 1.5, 2)));
    try std.testing.expectError(error.InvalidShape, coo.rowStddevsInRange(0, -0.1, 1));
    var col_stds = try coo.columnStddevs(0);
    defer col_stds.deinit();
    try std.testing.expectApproxEqAbs(@sqrt(26.0 / 9.0), col_stds.data[0], 1e-12);
    try std.testing.expect(try coo.columnStddevsInRange(0, @sqrt(2.0), @sqrt(26.0 / 3.0)));
    try std.testing.expect(!(try coo.columnStddevsInRange(0, 1.5, @sqrt(26.0 / 3.0))));

    var row_means = try coo.rowMeans();
    defer row_means.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, -1.0 / 3.0), row_means.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), row_means.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3), row_means.data[2], 1e-12);
    try std.testing.expect(try coo.rowMeansInRange(-1.0 / 3.0, 3));
    try std.testing.expect(!(try coo.rowMeansInRange(0, 3)));
    try std.testing.expectError(error.InvalidShape, coo.rowMeansInRange(3, 2));
    var col_means = try coo.columnMeans();
    defer col_means.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 3.0), col_means.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), col_means.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), col_means.data[2], 1e-12);
    try std.testing.expect(try coo.columnMeansInRange(1, 5.0 / 3.0));
    try std.testing.expect(!(try coo.columnMeansInRange(1.1, 5.0 / 3.0)));

    var row_mins = try coo.rowMins();
    defer row_mins.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -2, 0, 0 }, row_mins.data);
    try std.testing.expect(try coo.rowMinsInRange(-2, 0));
    try std.testing.expect(!(try coo.rowMinsInRange(-1, 0)));
    try std.testing.expectError(error.InvalidShape, coo.rowMinsInRange(1, 0));
    var row_maxes = try coo.rowMaxes();
    defer row_maxes.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 5 }, row_maxes.data);
    try std.testing.expect(try coo.rowMaxesInRange(1, 5));
    try std.testing.expect(!(try coo.rowMaxesInRange(2, 5)));
    var col_mins = try coo.columnMins();
    defer col_mins.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 0, -2 }, col_mins.data);
    try std.testing.expect(try coo.columnMinsInRange(-2, 0));
    try std.testing.expect(!(try coo.columnMinsInRange(-1, 0)));
    var col_maxes = try coo.columnMaxes();
    defer col_maxes.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 4, 3, 5 }, col_maxes.data);
    try std.testing.expect(try coo.columnMaxesInRange(3, 5));
    try std.testing.expect(!(try coo.columnMaxesInRange(4, 5)));
    var row_min_abs = try coo.rowMinAbs();
    defer row_min_abs.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 4 }, row_min_abs.data);
    try std.testing.expect(try coo.rowMinAbsInRange(1, 4));
    try std.testing.expect(!(try coo.rowMinAbsInRange(2, 4)));
    try std.testing.expectError(error.InvalidShape, coo.rowMinAbsInRange(5, 4));
    var row_max_abs = try coo.rowMaxAbs();
    defer row_max_abs.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 2, 3, 5 }, row_max_abs.data);
    try std.testing.expect(try coo.rowMaxAbsInRange(2, 5));
    try std.testing.expect(!(try coo.rowMaxAbsInRange(3, 5)));
    var col_min_abs = try coo.columnMinAbs();
    defer col_min_abs.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 2 }, col_min_abs.data);
    try std.testing.expect(try coo.columnMinAbsInRange(1, 3));
    try std.testing.expect(!(try coo.columnMinAbsInRange(2, 3)));
    var col_max_abs = try coo.columnMaxAbs();
    defer col_max_abs.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 4, 3, 5 }, col_max_abs.data);
    try std.testing.expect(try coo.columnMaxAbsInRange(3, 5));
    try std.testing.expect(!(try coo.columnMaxAbsInRange(4, 5)));

    var row_nnz = try coo.rowNnz();
    defer row_nnz.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 1, 2 }, row_nnz.data);
    var col_nnz = try coo.columnNnz();
    defer col_nnz.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 1, 2 }, col_nnz.data);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 3.0), try coo.averageRowNnz(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 3.0), try coo.averageColumnNnz(), 1e-12);
    try std.testing.expect(try coo.averageRowNnzInRange(1.6, 1.7));
    try std.testing.expect(!(try coo.averageColumnNnzInRange(0, 1.6)));
    try std.testing.expectEqual(@as(usize, 0), try coo.emptyRowCount());
    try std.testing.expectEqual(@as(usize, 0), try coo.emptyColumnCount());
    try std.testing.expectApproxEqAbs(@as(f64, 0), try coo.emptyRowFraction(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0), try coo.emptyColumnFraction(), 1e-12);
    try std.testing.expect(try coo.emptyRowFractionInRange(0, 0));
    try std.testing.expect(try coo.emptyColumnFractionInRange(0, 0));

    var row_sums = try coo.rowSums();
    defer row_sums.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -1, 3, 9 }, row_sums.data);
    try std.testing.expect(try coo.rowSumsInRange(-1, 9));
    try std.testing.expect(!(try coo.rowSumsInRange(0, 9)));
    try std.testing.expectError(error.InvalidShape, coo.rowSumsInRange(10, 9));
    var col_sums = try coo.columnSums();
    defer col_sums.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 5, 3, 3 }, col_sums.data);
    try std.testing.expect(try coo.columnSumsInRange(3, 5));
    try std.testing.expect(!(try coo.columnSumsInRange(4, 5)));

    var row_abs = try coo.rowAbsSums();
    defer row_abs.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 3, 3, 9 }, row_abs.data);
    try std.testing.expect(try coo.rowAbsSumsInRange(3, 9));
    try std.testing.expect(!(try coo.rowAbsSumsInRange(4, 9)));
    try std.testing.expectError(error.InvalidShape, coo.rowAbsSumsInRange(10, 9));
    var col_abs = try coo.columnAbsSums();
    defer col_abs.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 5, 3, 7 }, col_abs.data);
    try std.testing.expect(try coo.columnAbsSumsInRange(3, 7));
    try std.testing.expect(!(try coo.columnAbsSumsInRange(4, 7)));

    var row_norms = try coo.rowNorms();
    defer row_norms.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, @sqrt(5.0)), row_norms.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3), row_norms.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, @sqrt(41.0)), row_norms.data[2], 1e-12);
    try std.testing.expect(try coo.rowNormsInRange(@sqrt(@as(f64, 5.0)), @sqrt(@as(f64, 41.0))));
    try std.testing.expect(!(try coo.rowNormsInRange(3.1, @sqrt(@as(f64, 41.0)))));
    try std.testing.expectError(error.InvalidShape, coo.rowNormsInRange(-0.1, 1));
    var col_norms = try coo.columnNorms();
    defer col_norms.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, @sqrt(17.0)), col_norms.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3), col_norms.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, @sqrt(29.0)), col_norms.data[2], 1e-12);
    try std.testing.expect(try coo.columnNormsInRange(3, @sqrt(@as(f64, 29.0))));
    try std.testing.expect(!(try coo.columnNormsInRange(3.1, @sqrt(@as(f64, 29.0)))));
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 9.0), try coo.density(), 1e-12);
    try std.testing.expect(try coo.densityInRange(5.0 / 9.0, 5.0 / 9.0));
    try std.testing.expect(!(try coo.densityInRange(0, 0.5)));
    try std.testing.expectError(error.InvalidShape, coo.densityInRange(std.math.nan(f64), 1));
}

test "sparse bool reductions use dense materialization" {
    const gpa = std.testing.allocator;
    const expectBoolReductions = struct {
        fn expectMask(mask: array_mod.Array(bool), shape: []const usize, values: []const bool) !void {
            try std.testing.expectEqualSlices(usize, shape, mask.shape);
            try std.testing.expectEqualSlices(bool, values, mask.data);
        }

        fn checkReductions(comptime Matrix: type, matrix: Matrix) !void {
            try std.testing.expect(!(try matrix.all()));
            try std.testing.expect(try matrix.any());

            var all_axis0 = try matrix.allAxis(0, false);
            defer all_axis0.deinit();
            try expectMask(all_axis0, &.{3}, &.{ false, false, false });
            var any_axis0 = try matrix.anyAxis(0, false);
            defer any_axis0.deinit();
            try expectMask(any_axis0, &.{3}, &.{ true, false, true });

            var all_axis1_keep = try matrix.allDim(1, true);
            defer all_axis1_keep.deinit();
            try expectMask(all_axis1_keep, &.{ 2, 1 }, &.{ false, false });
            var any_axis1_keep = try matrix.anyDim(1, true);
            defer any_axis1_keep.deinit();
            try expectMask(any_axis1_keep, &.{ 2, 1 }, &.{ true, true });

            var all_axes = try matrix.allAxes(&.{ 0, 1 }, false);
            defer all_axes.deinit();
            try expectMask(all_axes, &.{}, &.{false});
            var any_axes = try matrix.anyDims(&.{ 0, 1 }, false);
            defer any_axes.deinit();
            try expectMask(any_axes, &.{}, &.{true});
        }

        fn checkLogical(comptime Matrix: type, matrix: Matrix, rhs: Matrix) !void {
            var not_mask = try matrix.logicalNot();
            defer not_mask.deinit();
            try expectMask(not_mask, &.{ 2, 3 }, &.{ false, true, true, true, true, false });

            var and_mask = try matrix.logicalAnd(rhs);
            defer and_mask.deinit();
            try expectMask(and_mask, &.{ 2, 3 }, &.{ false, false, false, false, false, true });

            var or_mask = try matrix.logicalOr(rhs);
            defer or_mask.deinit();
            try expectMask(or_mask, &.{ 2, 3 }, &.{ true, true, false, false, true, true });

            var xor_mask = try matrix.logicalXor(rhs);
            defer xor_mask.deinit();
            try expectMask(xor_mask, &.{ 2, 3 }, &.{ true, true, false, false, true, false });

            var and_true = try matrix.logicalAndScalar(true);
            defer and_true.deinit();
            try expectMask(and_true, &.{ 2, 3 }, &.{ true, false, false, false, false, true });

            var and_false = try matrix.logicalAndScalar(false);
            defer and_false.deinit();
            try expectMask(and_false, &.{ 2, 3 }, &.{ false, false, false, false, false, false });

            var or_true = try matrix.logicalOrScalar(true);
            defer or_true.deinit();
            try expectMask(or_true, &.{ 2, 3 }, &.{ true, true, true, true, true, true });

            var or_false = try matrix.logicalOrScalar(false);
            defer or_false.deinit();
            try expectMask(or_false, &.{ 2, 3 }, &.{ true, false, false, false, false, true });

            var xor_true = try matrix.logicalXorScalar(true);
            defer xor_true.deinit();
            try expectMask(xor_true, &.{ 2, 3 }, &.{ false, true, true, true, true, false });
        }
    };

    var coo = try cooFromSlices(bool, gpa, 2, 3, &.{ 0, 1, 1 }, &.{ 0, 1, 2 }, &.{ true, false, true });
    defer coo.deinit();
    var rhs_coo = try cooFromSlices(bool, gpa, 2, 3, &.{ 0, 0, 1, 1 }, &.{ 0, 1, 1, 2 }, &.{ false, true, true, true });
    defer rhs_coo.deinit();
    try expectBoolReductions.checkReductions(@TypeOf(coo), coo);
    try expectBoolReductions.checkLogical(@TypeOf(coo), coo, rhs_coo);

    var csr = try coo.toCsr();
    defer csr.deinit();
    var rhs_csr = try rhs_coo.toCsr();
    defer rhs_csr.deinit();
    try expectBoolReductions.checkReductions(@TypeOf(csr), csr);
    try expectBoolReductions.checkLogical(@TypeOf(csr), csr, rhs_csr);

    var csc = try coo.toCsc();
    defer csc.deinit();
    var rhs_csc = try rhs_coo.toCsc();
    defer rhs_csc.deinit();
    try expectBoolReductions.checkReductions(@TypeOf(csc), csc);
    try expectBoolReductions.checkLogical(@TypeOf(csc), csc, rhs_csc);

    var all_true = try cooFromSlices(bool, gpa, 1, 2, &.{ 0, 0 }, &.{ 0, 1 }, &.{ true, true });
    defer all_true.deinit();
    try std.testing.expect(try all_true.all());
    try std.testing.expect(try all_true.any());

    var mismatched = try cooFromSlices(bool, gpa, 3, 3, &.{0}, &.{0}, &.{true});
    defer mismatched.deinit();
    try std.testing.expectError(error.ShapeMismatch, coo.logicalAnd(mismatched));
}

test "sparse stored non-finite diagnostics" {
    const gpa = std.testing.allocator;
    var coo = try cooFromSlices(f64, gpa, 3, 3, &.{ 0, 0, 1, 2, 2 }, &.{ 0, 2, 1, 0, 2 }, &.{ 1.0, std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64), 5.0 });
    defer coo.deinit();
    try std.testing.expectEqual(@as(usize, 3), coo.nonFiniteCount());
    try std.testing.expect(coo.nonFiniteCountMeetsBound(3));
    try std.testing.expect(!coo.nonFiniteCountMeetsBound(2));
    try std.testing.expect(try coo.nonFiniteCountInRange(3, 3));
    try std.testing.expect(try coo.nonFiniteCountInRange(2, 4));
    try std.testing.expect(!(try coo.nonFiniteCountInRange(0, 2)));
    try std.testing.expectError(error.InvalidShape, coo.nonFiniteCountInRange(4, 3));
    try std.testing.expect(!coo.allFinite());
    var coo_nan = try coo.isNan();
    defer coo_nan.deinit();
    try std.testing.expectEqualSlices(usize, coo.row_indices, coo_nan.row_indices);
    try std.testing.expectEqualSlices(usize, coo.col_indices, coo_nan.col_indices);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false, false }, coo_nan.values);
    var coo_inf = try coo.isinf();
    defer coo_inf.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, true, false }, coo_inf.values);
    var coo_pos_inf = try coo.isPosInf();
    defer coo_pos_inf.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, false }, coo_pos_inf.values);
    var coo_neg_inf = try coo.isneginf();
    defer coo_neg_inf.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, true, false }, coo_neg_inf.values);
    var coo_finite = try coo.isFinite();
    defer coo_finite.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false, true }, coo_finite.values);
    var coo_normal = try coo.isnormal();
    defer coo_normal.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false, true }, coo_normal.values);

    var coo_rows = try coo.rowNonFiniteCounts();
    defer coo_rows.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 1, 1 }, coo_rows.data);
    try std.testing.expect(try coo.rowNonFiniteCountsMeetBound(1));
    try std.testing.expect(!(try coo.rowNonFiniteCountsMeetBound(0)));
    try std.testing.expect(try coo.rowNonFiniteCountsInRange(1, 1));
    try std.testing.expectError(error.InvalidShape, coo.rowNonFiniteCountsInRange(2, 1));
    var coo_cols = try coo.columnNonFiniteCounts();
    defer coo_cols.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 1, 1 }, coo_cols.data);
    try std.testing.expect(try coo.columnNonFiniteCountsMeetBound(1));
    try std.testing.expect(!(try coo.columnNonFiniteCountsMeetBound(0)));
    try std.testing.expect(try coo.columnNonFiniteCountsInRange(1, 1));

    var csr = try coo.toCsr();
    defer csr.deinit();
    try std.testing.expectEqual(@as(usize, 3), csr.nonFiniteCount());
    try std.testing.expect(csr.nonFiniteCountMeetsBound(3));
    try std.testing.expect(!csr.nonFiniteCountMeetsBound(2));
    try std.testing.expect(try csr.nonFiniteCountInRange(3, 3));
    try std.testing.expect(!(try csr.nonFiniteCountInRange(0, 2)));
    try std.testing.expectError(error.InvalidShape, csr.nonFiniteCountInRange(4, 3));
    try std.testing.expect(!csr.allFinite());
    var csr_nan = try csr.isnan();
    defer csr_nan.deinit();
    try std.testing.expectEqualSlices(usize, csr.row_offsets, csr_nan.row_offsets);
    try std.testing.expectEqualSlices(usize, csr.col_indices, csr_nan.col_indices);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false, false }, csr_nan.values);
    var csr_inf = try csr.isInf();
    defer csr_inf.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, true, false }, csr_inf.values);
    var csr_pos_inf = try csr.isposinf();
    defer csr_pos_inf.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, false }, csr_pos_inf.values);
    var csr_neg_inf = try csr.isNegInf();
    defer csr_neg_inf.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, true, false }, csr_neg_inf.values);
    var csr_finite = try csr.isfinite();
    defer csr_finite.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false, true }, csr_finite.values);
    var csr_normal = try csr.isNormal();
    defer csr_normal.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false, true }, csr_normal.values);
    var csr_rows = try csr.rowNonFiniteCounts();
    defer csr_rows.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 1, 1 }, csr_rows.data);
    try std.testing.expect(try csr.rowNonFiniteCountsMeetBound(1));
    try std.testing.expect(!(try csr.rowNonFiniteCountsMeetBound(0)));
    try std.testing.expect(try csr.rowNonFiniteCountsInRange(1, 1));
    try std.testing.expectError(error.InvalidShape, csr.rowNonFiniteCountsInRange(2, 1));
    var csr_cols = try csr.columnNonFiniteCounts();
    defer csr_cols.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 1, 1 }, csr_cols.data);
    try std.testing.expect(try csr.columnNonFiniteCountsMeetBound(1));
    try std.testing.expect(!(try csr.columnNonFiniteCountsMeetBound(0)));
    try std.testing.expect(try csr.columnNonFiniteCountsInRange(1, 1));

    var csc = try coo.toCsc();
    defer csc.deinit();
    try std.testing.expectEqual(@as(usize, 3), csc.nonFiniteCount());
    try std.testing.expect(csc.nonFiniteCountMeetsBound(3));
    try std.testing.expect(!csc.nonFiniteCountMeetsBound(2));
    try std.testing.expect(try csc.nonFiniteCountInRange(3, 3));
    try std.testing.expect(!(try csc.nonFiniteCountInRange(0, 2)));
    try std.testing.expectError(error.InvalidShape, csc.nonFiniteCountInRange(4, 3));
    try std.testing.expect(!csc.allFinite());
    var csc_nan = try csc.isNan();
    defer csc_nan.deinit();
    try std.testing.expectEqualSlices(usize, csc.col_offsets, csc_nan.col_offsets);
    try std.testing.expectEqualSlices(usize, csc.row_indices, csc_nan.row_indices);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, true, false }, csc_nan.values);
    var csc_inf = try csc.isinf();
    defer csc_inf.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, false, false }, csc_inf.values);
    var csc_pos_inf = try csc.isPosInf();
    defer csc_pos_inf.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, false }, csc_pos_inf.values);
    var csc_neg_inf = try csc.isneginf();
    defer csc_neg_inf.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false, false }, csc_neg_inf.values);
    var csc_finite = try csc.isFinite();
    defer csc_finite.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false, true }, csc_finite.values);
    var csc_normal = try csc.isnormal();
    defer csc_normal.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false, true }, csc_normal.values);
    var csc_rows = try csc.rowNonFiniteCounts();
    defer csc_rows.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 1, 1 }, csc_rows.data);
    try std.testing.expect(try csc.rowNonFiniteCountsMeetBound(1));
    try std.testing.expect(!(try csc.rowNonFiniteCountsMeetBound(0)));
    try std.testing.expect(try csc.rowNonFiniteCountsInRange(1, 1));
    try std.testing.expectError(error.InvalidShape, csc.rowNonFiniteCountsInRange(2, 1));
    var csc_cols = try csc.columnNonFiniteCounts();
    defer csc_cols.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 1, 1 }, csc_cols.data);
    try std.testing.expect(try csc.columnNonFiniteCountsMeetBound(1));
    try std.testing.expect(!(try csc.columnNonFiniteCountsMeetBound(0)));
    try std.testing.expect(try csc.columnNonFiniteCountsInRange(1, 1));

    var finite = try cooFromSlices(f64, gpa, 2, 2, &.{ 0, 1 }, &.{ 1, 0 }, &.{ 2.0, 3.0 });
    defer finite.deinit();
    try std.testing.expectEqual(@as(usize, 0), finite.nonFiniteCount());
    try std.testing.expect(finite.nonFiniteCountMeetsBound(0));
    try std.testing.expect(try finite.nonFiniteCountInRange(0, 0));
    try std.testing.expect(!(try finite.nonFiniteCountInRange(1, 1)));
    try std.testing.expect(try finite.rowNonFiniteCountsMeetBound(0));
    try std.testing.expect(try finite.columnNonFiniteCountsInRange(0, 0));
    try std.testing.expect(finite.allFinite());
}

test "sparse occupancy diagnostics" {
    const gpa = std.testing.allocator;
    var coo = try cooFromSlices(f64, gpa, 3, 4, &.{ 0, 0, 2 }, &.{ 0, 2, 0 }, &.{ 1.0, 2.0, 3.0 });
    defer coo.deinit();

    try std.testing.expectApproxEqAbs(@as(f64, 1), try coo.averageRowNnz(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), try coo.averageColumnNnz(), 1e-12);
    try std.testing.expect(try coo.averageRowNnzInRange(1, 1));
    try std.testing.expect(!(try coo.averageColumnNnzInRange(0, 0.7)));
    try std.testing.expectError(error.InvalidShape, coo.averageRowNnzInRange(2, 1));
    try std.testing.expectEqual(@as(usize, 1), try coo.emptyRowCount());
    try std.testing.expectEqual(@as(usize, 2), try coo.emptyColumnCount());
    try std.testing.expect(try coo.emptyRowCountMeetsBound(1));
    try std.testing.expect(!(try coo.emptyRowCountMeetsBound(0)));
    try std.testing.expect(try coo.emptyRowCountInRange(1, 1));
    try std.testing.expect(!(try coo.emptyRowCountInRange(0, 0)));
    try std.testing.expectError(error.InvalidShape, coo.emptyRowCountInRange(2, 1));
    try std.testing.expect(try coo.emptyColumnCountMeetsBound(2));
    try std.testing.expect(!(try coo.emptyColumnCountMeetsBound(1)));
    try std.testing.expect(try coo.emptyColumnCountInRange(2, 2));
    try std.testing.expect(!(try coo.emptyColumnCountInRange(0, 1)));
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), try coo.emptyRowFraction(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), try coo.emptyColumnFraction(), 1e-12);
    try std.testing.expect(try coo.emptyRowFractionInRange(0.3, 0.4));
    try std.testing.expect(!(try coo.emptyColumnFractionInRange(0, 0.49)));
    try std.testing.expectError(error.InvalidShape, coo.emptyRowFractionInRange(std.math.nan(f64), 1));
    try std.testing.expectEqual(@as(usize, 0), try coo.minRowNnz());
    try std.testing.expectEqual(@as(usize, 2), try coo.maxRowNnz());
    try std.testing.expectEqual(@as(usize, 0), try coo.minColumnNnz());
    try std.testing.expectEqual(@as(usize, 2), try coo.maxColumnNnz());
    try std.testing.expect(try coo.rowNnzRangeInRange(0, 2));
    try std.testing.expect(!(try coo.columnNnzRangeInRange(1, 2)));
    try std.testing.expectError(error.InvalidShape, coo.rowNnzRangeInRange(2, 1));
    try std.testing.expectEqual(@as(usize, 2), try coo.rowNnzSpread());
    try std.testing.expectEqual(@as(usize, 2), try coo.columnNnzSpread());
    try std.testing.expect(try coo.rowNnzSpreadMeetsBound(2));
    try std.testing.expect(!(try coo.columnNnzSpreadMeetsBound(1)));

    var csr = try coo.toCsr();
    defer csr.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1), try csr.averageRowNnz(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), try csr.averageColumnNnz(), 1e-12);
    try std.testing.expectEqual(@as(usize, 1), csr.emptyRowCount());
    try std.testing.expectEqual(@as(usize, 2), try csr.emptyColumnCount());
    try std.testing.expect(csr.emptyRowCountMeetsBound(1));
    try std.testing.expect(!csr.emptyRowCountMeetsBound(0));
    try std.testing.expect(try csr.emptyRowCountInRange(1, 1));
    try std.testing.expect(!(try csr.emptyRowCountInRange(0, 0)));
    try std.testing.expectError(error.InvalidShape, csr.emptyRowCountInRange(2, 1));
    try std.testing.expect(try csr.emptyColumnCountMeetsBound(2));
    try std.testing.expect(!(try csr.emptyColumnCountMeetsBound(1)));
    try std.testing.expect(try csr.emptyColumnCountInRange(2, 2));
    try std.testing.expect(!(try csr.emptyColumnCountInRange(0, 1)));
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), try csr.emptyRowFraction(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), try csr.emptyColumnFraction(), 1e-12);
    try std.testing.expectEqual(@as(usize, 0), try csr.minRowNnz());
    try std.testing.expectEqual(@as(usize, 2), try csr.maxRowNnz());
    try std.testing.expectEqual(@as(usize, 0), try csr.minColumnNnz());
    try std.testing.expectEqual(@as(usize, 2), try csr.maxColumnNnz());
    try std.testing.expect(try csr.rowNnzRangeInRange(0, 2));
    try std.testing.expectEqual(@as(usize, 2), try csr.rowNnzSpread());
    try std.testing.expect(!(try csr.columnNnzSpreadMeetsBound(1)));

    var csc = try coo.toCsc();
    defer csc.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1), try csc.averageRowNnz(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), try csc.averageColumnNnz(), 1e-12);
    try std.testing.expectEqual(@as(usize, 1), try csc.emptyRowCount());
    try std.testing.expectEqual(@as(usize, 2), csc.emptyColumnCount());
    try std.testing.expect(try csc.emptyRowCountMeetsBound(1));
    try std.testing.expect(!(try csc.emptyRowCountMeetsBound(0)));
    try std.testing.expect(try csc.emptyRowCountInRange(1, 1));
    try std.testing.expect(!(try csc.emptyRowCountInRange(0, 0)));
    try std.testing.expectError(error.InvalidShape, csc.emptyRowCountInRange(2, 1));
    try std.testing.expect(csc.emptyColumnCountMeetsBound(2));
    try std.testing.expect(!csc.emptyColumnCountMeetsBound(1));
    try std.testing.expect(try csc.emptyColumnCountInRange(2, 2));
    try std.testing.expect(!(try csc.emptyColumnCountInRange(0, 1)));
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), try csc.emptyRowFraction(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), try csc.emptyColumnFraction(), 1e-12);
    try std.testing.expectEqual(@as(usize, 0), try csc.minRowNnz());
    try std.testing.expectEqual(@as(usize, 2), try csc.maxRowNnz());
    try std.testing.expectEqual(@as(usize, 0), try csc.minColumnNnz());
    try std.testing.expectEqual(@as(usize, 2), try csc.maxColumnNnz());
    try std.testing.expect(try csc.columnNnzRangeInRange(0, 2));
    try std.testing.expectEqual(@as(usize, 2), try csc.columnNnzSpread());
    try std.testing.expect(try csc.rowNnzSpreadMeetsBound(2));

    var zero_rows = try cooFromSlices(f64, gpa, 0, 2, &.{}, &.{}, &.{});
    defer zero_rows.deinit();
    try std.testing.expectError(error.EmptyArray, zero_rows.averageRowNnz());
    try std.testing.expectError(error.EmptyArray, zero_rows.emptyRowFraction());
    try std.testing.expectEqual(@as(usize, 0), try zero_rows.minRowNnz());
    try std.testing.expectEqual(@as(usize, 0), try zero_rows.rowNnzSpread());

    var zero_cols = try cooFromSlices(f64, gpa, 2, 0, &.{}, &.{}, &.{});
    defer zero_cols.deinit();
    try std.testing.expectError(error.EmptyArray, zero_cols.averageColumnNnz());
    try std.testing.expectError(error.EmptyArray, zero_cols.emptyColumnFraction());
    try std.testing.expectEqual(@as(usize, 0), try zero_cols.minColumnNnz());
    try std.testing.expectEqual(@as(usize, 0), try zero_cols.columnNnzSpread());
}

test "sparse stored value range diagnostics" {
    const gpa = std.testing.allocator;
    var coo = try cooFromSlices(f64, gpa, 2, 3, &.{ 0, 0, 1, 1 }, &.{ 0, 2, 1, 2 }, &.{ -2.0, 4.0, 1.0, 8.0 });
    defer coo.deinit();
    try std.testing.expect(try coo.valueRangeInRange(-2, 8));
    try std.testing.expect(!(try coo.valueRangeInRange(-1, 8)));
    try std.testing.expect(try coo.absValueRangeInRange(1, 8));
    try std.testing.expect(!(try coo.absValueRangeInRange(2, 8)));
    try std.testing.expectApproxEqAbs(@as(f64, 8), try coo.valueDynamicRange(), 1e-12);
    try std.testing.expect(try coo.valueDynamicRangeMeetsBound(8));
    try std.testing.expect(!(try coo.valueDynamicRangeMeetsBound(7.999)));

    var csr = try coo.toCsr();
    defer csr.deinit();
    try std.testing.expect(try csr.valueRangeInRange(-2, 8));
    try std.testing.expect(!(try csr.valueRangeInRange(-2, 7.9)));
    try std.testing.expect(try csr.absValueRangeInRange(1, 8));
    try std.testing.expectApproxEqAbs(@as(f64, 8), try csr.valueDynamicRange(), 1e-12);

    var csc = try coo.toCsc();
    defer csc.deinit();
    try std.testing.expect(try csc.valueRangeInRange(-2, 8));
    try std.testing.expect(!(try csc.absValueRangeInRange(1.1, 8)));
    try std.testing.expectApproxEqAbs(@as(f64, 8), try csc.valueDynamicRange(), 1e-12);

    var zero_value = try cooFromSlices(f64, gpa, 1, 1, &.{0}, &.{0}, &.{0.0});
    defer zero_value.deinit();
    try std.testing.expectError(error.SingularMatrix, zero_value.valueDynamicRange());
    try std.testing.expectError(error.SingularMatrix, zero_value.valueDynamicRangeMeetsBound(1));

    var empty = try cooFromSlices(f64, gpa, 1, 1, &.{}, &.{}, &.{});
    defer empty.deinit();
    try std.testing.expectError(error.EmptyArray, empty.minValueIndex());
    try std.testing.expectError(error.EmptyArray, empty.maxValueIndex());
    try std.testing.expectError(error.EmptyArray, empty.minAbsValueIndex());
    try std.testing.expectError(error.EmptyArray, empty.maxAbsValueIndex());
    try std.testing.expectError(error.EmptyArray, empty.valueRangeInRange(0, 1));
    try std.testing.expectError(error.EmptyArray, empty.absValueRangeInRange(0, 1));
    try std.testing.expectError(error.EmptyArray, empty.valueDynamicRange());
    try std.testing.expectError(error.InvalidShape, coo.valueRangeInRange(std.math.nan(f64), 1));
    try std.testing.expectError(error.InvalidShape, coo.absValueRangeInRange(2, 1));
    try std.testing.expectError(error.InvalidShape, coo.valueDynamicRangeMeetsBound(std.math.inf(f64)));
}

test "sparse diagonal absolute diagnostics" {
    const gpa = std.testing.allocator;
    var dense = try array_mod.Array(f64).fromSlice(gpa, &.{
        4, 1,  0,
        1, -5, 2,
        0, 2,  6,
    }, &.{ 3, 3 });
    defer dense.deinit();
    var coo = try cooFromDense(f64, dense);
    defer coo.deinit();

    try std.testing.expectApproxEqAbs(@as(f64, 4), try coo.minAbsDiagonal(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 6), try coo.maxAbsDiagonal(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), try coo.diagonalDynamicRange(), 1e-12);
    try std.testing.expect(try coo.diagonalDynamicRangeMeetsBound(1.5));
    try std.testing.expect(!(try coo.diagonalDynamicRangeMeetsBound(1.499)));

    var csr = try coo.toCsr();
    defer csr.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 4), try csr.minAbsDiagonal(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 6), try csr.maxAbsDiagonal(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), try csr.diagonalDynamicRange(), 1e-12);

    var csc = try coo.toCsc();
    defer csc.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 4), try csc.minAbsDiagonal(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 6), try csc.maxAbsDiagonal(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), try csc.diagonalDynamicRange(), 1e-12);

    var missing = try cooFromSlices(f64, gpa, 3, 3, &.{ 0, 1, 2 }, &.{ 0, 2, 2 }, &.{ 1.0, 3.0, 4.0 });
    defer missing.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0), try missing.minAbsDiagonal(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4), try missing.maxAbsDiagonal(), 1e-12);
    try std.testing.expectError(error.SingularMatrix, missing.diagonalDynamicRange());
    try std.testing.expectError(error.SingularMatrix, missing.diagonalDynamicRangeMeetsBound(1));

    var rectangular = try cooFromSlices(f64, gpa, 2, 3, &.{ 0, 1 }, &.{ 0, 2 }, &.{ 1.0, 2.0 });
    defer rectangular.deinit();
    try std.testing.expectError(error.NonMatrixArray, rectangular.minAbsDiagonal());
    try std.testing.expectError(error.NonMatrixArray, rectangular.maxAbsDiagonal());
    try std.testing.expectError(error.NonMatrixArray, rectangular.diagonalDynamicRange());
    try std.testing.expectError(error.NonMatrixArray, rectangular.diagonalDynamicRangeMeetsBound(1));

    var rectangular_csr = try rectangular.toCsr();
    defer rectangular_csr.deinit();
    try std.testing.expectError(error.NonMatrixArray, rectangular_csr.minAbsDiagonal());
    try std.testing.expectError(error.NonMatrixArray, rectangular_csr.diagonalDynamicRange());
    var rectangular_csc = try rectangular.toCsc();
    defer rectangular_csc.deinit();
    try std.testing.expectError(error.NonMatrixArray, rectangular_csc.maxAbsDiagonal());
    try std.testing.expectError(error.NonMatrixArray, rectangular_csc.diagonalDynamicRangeMeetsBound(1));

    var empty = try cooFromSlices(f64, gpa, 0, 0, &.{}, &.{}, &.{});
    defer empty.deinit();
    try std.testing.expectError(error.EmptyArray, empty.minAbsDiagonal());
    try std.testing.expectError(error.EmptyArray, empty.diagonalDynamicRange());
    try std.testing.expectError(error.InvalidShape, coo.diagonalDynamicRangeMeetsBound(std.math.nan(f64)));
}

test "sparse diagonal dominance diagnostics" {
    const gpa = std.testing.allocator;
    var strict_dense = try array_mod.Array(f64).fromSlice(gpa, &.{
        4, 1, 0,
        1, 5, 2,
        0, 2, 6,
    }, &.{ 3, 3 });
    defer strict_dense.deinit();
    var strict_coo = try cooFromDense(f64, strict_dense);
    defer strict_coo.deinit();
    try std.testing.expect(try strict_coo.diagonallyDominant());
    try std.testing.expect(try strict_coo.strictlyDiagonallyDominant());
    try std.testing.expectApproxEqAbs(@as(f64, 2), try strict_coo.diagonalDominanceMargin(), 1e-12);
    try std.testing.expect(try strict_coo.diagonalDominanceMarginMeetsBound(2));
    try std.testing.expect(!(try strict_coo.diagonalDominanceMarginMeetsBound(2.001)));
    try std.testing.expectError(error.InvalidShape, strict_coo.diagonalDominanceMarginMeetsBound(std.math.nan(f64)));

    var strict_csr = try strict_coo.toCsr();
    defer strict_csr.deinit();
    try std.testing.expect(try strict_csr.diagonallyDominant());
    try std.testing.expect(try strict_csr.strictlyDiagonallyDominant());
    try std.testing.expectApproxEqAbs(@as(f64, 2), try strict_csr.diagonalDominanceMargin(), 1e-12);
    try std.testing.expect(try strict_csr.diagonalDominanceMarginMeetsBound(2));

    var strict_csc = try strict_coo.toCsc();
    defer strict_csc.deinit();
    try std.testing.expect(try strict_csc.diagonallyDominant());
    try std.testing.expect(try strict_csc.strictlyDiagonallyDominant());
    try std.testing.expectApproxEqAbs(@as(f64, 2), try strict_csc.diagonalDominanceMargin(), 1e-12);
    try std.testing.expect(try strict_csc.diagonalDominanceMarginMeetsBound(2));

    var weak_dense = try array_mod.Array(f64).fromSlice(gpa, &.{
        1, 1,
        0, 2,
    }, &.{ 2, 2 });
    defer weak_dense.deinit();
    var weak = try csrFromDense(f64, weak_dense);
    defer weak.deinit();
    try std.testing.expect(try weak.diagonallyDominant());
    try std.testing.expect(!(try weak.strictlyDiagonallyDominant()));
    try std.testing.expectApproxEqAbs(@as(f64, 0), try weak.diagonalDominanceMargin(), 1e-12);
    try std.testing.expect(try weak.diagonalDominanceMarginMeetsBound(0));
    try std.testing.expect(!(try weak.diagonalDominanceMarginMeetsBound(1e-12)));

    var non_dominant_dense = try array_mod.Array(f64).fromSlice(gpa, &.{
        1, 2,
        0, 1,
    }, &.{ 2, 2 });
    defer non_dominant_dense.deinit();
    var non_dominant = try cscFromDense(f64, non_dominant_dense);
    defer non_dominant.deinit();
    try std.testing.expect(!(try non_dominant.diagonallyDominant()));
    try std.testing.expect(!(try non_dominant.strictlyDiagonallyDominant()));
    try std.testing.expectApproxEqAbs(@as(f64, -1), try non_dominant.diagonalDominanceMargin(), 1e-12);
    try std.testing.expect(try non_dominant.diagonalDominanceMarginMeetsBound(-1));
    try std.testing.expect(!(try non_dominant.diagonalDominanceMarginMeetsBound(-0.999)));

    var missing = try cooFromSlices(f64, gpa, 2, 2, &.{ 0, 0, 1 }, &.{ 0, 1, 0 }, &.{ 3.0, 1.0, 1.0 });
    defer missing.deinit();
    try std.testing.expect(!(try missing.diagonallyDominant()));
    try std.testing.expect(!(try missing.strictlyDiagonallyDominant()));
    try std.testing.expectApproxEqAbs(@as(f64, -1), try missing.diagonalDominanceMargin(), 1e-12);

    var duplicate_cancel = try cooFromSlices(f64, gpa, 2, 2, &.{ 0, 0, 0, 1 }, &.{ 0, 1, 1, 1 }, &.{ 1.0, 5.0, -5.0, 1.0 });
    defer duplicate_cancel.deinit();
    try std.testing.expect(try duplicate_cancel.diagonallyDominant());
    try std.testing.expect(try duplicate_cancel.strictlyDiagonallyDominant());

    var rectangular = try cooFromSlices(f64, gpa, 2, 3, &.{ 0, 1 }, &.{ 0, 2 }, &.{ 1.0, 2.0 });
    defer rectangular.deinit();
    try std.testing.expectError(error.NonMatrixArray, rectangular.diagonallyDominant());
    try std.testing.expectError(error.NonMatrixArray, rectangular.strictlyDiagonallyDominant());
    try std.testing.expectError(error.NonMatrixArray, rectangular.diagonalDominanceMargin());
    try std.testing.expectError(error.NonMatrixArray, rectangular.diagonalDominanceMarginMeetsBound(0));

    var rectangular_csr = try rectangular.toCsr();
    defer rectangular_csr.deinit();
    try std.testing.expectError(error.NonMatrixArray, rectangular_csr.diagonallyDominant());
    try std.testing.expectError(error.NonMatrixArray, rectangular_csr.strictlyDiagonallyDominant());
    try std.testing.expectError(error.NonMatrixArray, rectangular_csr.diagonalDominanceMargin());

    var rectangular_csc = try rectangular.toCsc();
    defer rectangular_csc.deinit();
    try std.testing.expectError(error.NonMatrixArray, rectangular_csc.diagonallyDominant());
    try std.testing.expectError(error.NonMatrixArray, rectangular_csc.strictlyDiagonallyDominant());
    try std.testing.expectError(error.NonMatrixArray, rectangular_csc.diagonalDominanceMarginMeetsBound(0));

    var empty = try cooFromSlices(f64, gpa, 0, 0, &.{}, &.{}, &.{});
    defer empty.deinit();
    try std.testing.expectError(error.EmptyArray, empty.diagonallyDominant());
    try std.testing.expectError(error.EmptyArray, empty.strictlyDiagonallyDominant());
    try std.testing.expectError(error.EmptyArray, empty.diagonalDominanceMargin());
}

test "sparse non-positive diagonal diagnostics" {
    const gpa = std.testing.allocator;
    var coo = try cooFromSlices(f64, gpa, 3, 3, &.{ 0, 1, 2, 2 }, &.{ 0, 1, 2, 2 }, &.{ 3.0, 0.0, 4.0, -5.0 });
    defer coo.deinit();
    try std.testing.expectEqual(@as(usize, 2), try coo.nonPositiveDiagonalCount());
    try std.testing.expect(try coo.nonPositiveDiagonalCountMeetsBound(2));
    try std.testing.expect(!(try coo.nonPositiveDiagonalCountMeetsBound(1)));
    try std.testing.expect(try coo.nonPositiveDiagonalCountInRange(2, 2));
    try std.testing.expect(!(try coo.nonPositiveDiagonalCountInRange(0, 1)));
    try std.testing.expectError(error.InvalidShape, coo.nonPositiveDiagonalCountInRange(3, 2));

    var csr = try coo.toCsr();
    defer csr.deinit();
    try std.testing.expectEqual(@as(usize, 2), try csr.nonPositiveDiagonalCount());
    try std.testing.expect(try csr.nonPositiveDiagonalCountMeetsBound(2));
    try std.testing.expect(try csr.nonPositiveDiagonalCountInRange(1, 2));

    var csc = try coo.toCsc();
    defer csc.deinit();
    try std.testing.expectEqual(@as(usize, 2), try csc.nonPositiveDiagonalCount());
    try std.testing.expect(!(try csc.nonPositiveDiagonalCountMeetsBound(1)));
    try std.testing.expect(try csc.nonPositiveDiagonalCountInRange(2, 3));

    var missing = try cooFromSlices(f64, gpa, 3, 3, &.{ 0, 2 }, &.{ 0, 2 }, &.{ 1.0, 2.0 });
    defer missing.deinit();
    try std.testing.expectEqual(@as(usize, 0), try missing.nonPositiveDiagonalCount());

    var rectangular = try cooFromSlices(f64, gpa, 2, 3, &.{ 0, 1 }, &.{ 0, 2 }, &.{ 1.0, 0.0 });
    defer rectangular.deinit();
    try std.testing.expectError(error.NonMatrixArray, rectangular.nonPositiveDiagonalCount());
    try std.testing.expectError(error.NonMatrixArray, rectangular.nonPositiveDiagonalCountMeetsBound(1));
    try std.testing.expectError(error.InvalidShape, rectangular.nonPositiveDiagonalCountInRange(2, 1));

    var rectangular_csr = try rectangular.toCsr();
    defer rectangular_csr.deinit();
    try std.testing.expectError(error.NonMatrixArray, rectangular_csr.nonPositiveDiagonalCount());
    var rectangular_csc = try rectangular.toCsc();
    defer rectangular_csc.deinit();
    try std.testing.expectError(error.NonMatrixArray, rectangular_csc.nonPositiveDiagonalCountMeetsBound(1));
}

test "sparse addition canonicalizes duplicate coordinates" {
    const gpa = std.testing.allocator;
    const expectCooBool = struct {
        fn check(mask: CooMatrix(bool), row_indices: []const usize, col_indices: []const usize, values: []const bool) !void {
            try std.testing.expectEqualSlices(usize, row_indices, mask.row_indices);
            try std.testing.expectEqualSlices(usize, col_indices, mask.col_indices);
            try std.testing.expectEqualSlices(bool, values, mask.values);
        }
    }.check;
    const expectCsrBool = struct {
        fn check(mask: CsrMatrix(bool), row_offsets: []const usize, col_indices: []const usize, values: []const bool) !void {
            try std.testing.expectEqualSlices(usize, row_offsets, mask.row_offsets);
            try std.testing.expectEqualSlices(usize, col_indices, mask.col_indices);
            try std.testing.expectEqualSlices(bool, values, mask.values);
        }
    }.check;
    const expectCscBool = struct {
        fn check(mask: CscMatrix(bool), col_offsets: []const usize, row_indices: []const usize, values: []const bool) !void {
            try std.testing.expectEqualSlices(usize, col_offsets, mask.col_offsets);
            try std.testing.expectEqualSlices(usize, row_indices, mask.row_indices);
            try std.testing.expectEqualSlices(bool, values, mask.values);
        }
    }.check;
    const expectDenseComparisons = struct {
        fn expectMask(mask: array_mod.Array(bool), values: []const bool) !void {
            try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, mask.shape);
            try std.testing.expectEqualSlices(bool, values, mask.data);
        }

        fn check(comptime Matrix: type, lhs_matrix: Matrix, rhs_matrix: Matrix, rhs_dense: array_mod.Array(f64)) !void {
            const eq_values = &.{ false, true, true, true, false, false };
            const ne_values = &.{ true, false, false, false, true, true };
            const gt_values = &.{ false, false, false, false, true, false };
            const ge_values = &.{ false, true, true, true, true, false };
            const lt_values = &.{ true, false, false, false, false, true };
            const le_values = &.{ true, true, true, true, false, true };

            var eq_mask = try lhs_matrix.eq(rhs_matrix);
            defer eq_mask.deinit();
            try expectMask(eq_mask, eq_values);
            var equal_mask = try lhs_matrix.equal(rhs_matrix);
            defer equal_mask.deinit();
            try expectMask(equal_mask, eq_values);
            var ne_mask = try lhs_matrix.ne(rhs_matrix);
            defer ne_mask.deinit();
            try expectMask(ne_mask, ne_values);
            var not_equal_mask = try lhs_matrix.notEqual(rhs_matrix);
            defer not_equal_mask.deinit();
            try expectMask(not_equal_mask, ne_values);
            var gt_mask = try lhs_matrix.gt(rhs_matrix);
            defer gt_mask.deinit();
            try expectMask(gt_mask, gt_values);
            var greater_mask = try lhs_matrix.greater(rhs_matrix);
            defer greater_mask.deinit();
            try expectMask(greater_mask, gt_values);
            var ge_mask = try lhs_matrix.ge(rhs_matrix);
            defer ge_mask.deinit();
            try expectMask(ge_mask, ge_values);
            var greater_equal_mask = try lhs_matrix.greaterEqual(rhs_matrix);
            defer greater_equal_mask.deinit();
            try expectMask(greater_equal_mask, ge_values);
            var lt_mask = try lhs_matrix.lt(rhs_matrix);
            defer lt_mask.deinit();
            try expectMask(lt_mask, lt_values);
            var less_mask = try lhs_matrix.less(rhs_matrix);
            defer less_mask.deinit();
            try expectMask(less_mask, lt_values);
            var le_mask = try lhs_matrix.le(rhs_matrix);
            defer le_mask.deinit();
            try expectMask(le_mask, le_values);
            var less_equal_mask = try lhs_matrix.lessEqual(rhs_matrix);
            defer less_equal_mask.deinit();
            try expectMask(less_equal_mask, le_values);

            var eq_dense_mask = try lhs_matrix.eqDense(rhs_dense);
            defer eq_dense_mask.deinit();
            try expectMask(eq_dense_mask, eq_values);
            var equal_dense_mask = try lhs_matrix.equalDense(rhs_dense);
            defer equal_dense_mask.deinit();
            try expectMask(equal_dense_mask, eq_values);
            var ne_dense_mask = try lhs_matrix.neDense(rhs_dense);
            defer ne_dense_mask.deinit();
            try expectMask(ne_dense_mask, ne_values);
            var not_equal_dense_mask = try lhs_matrix.notEqualDense(rhs_dense);
            defer not_equal_dense_mask.deinit();
            try expectMask(not_equal_dense_mask, ne_values);
            var gt_dense_mask = try lhs_matrix.gtDense(rhs_dense);
            defer gt_dense_mask.deinit();
            try expectMask(gt_dense_mask, gt_values);
            var greater_dense_mask = try lhs_matrix.greaterDense(rhs_dense);
            defer greater_dense_mask.deinit();
            try expectMask(greater_dense_mask, gt_values);
            var ge_dense_mask = try lhs_matrix.geDense(rhs_dense);
            defer ge_dense_mask.deinit();
            try expectMask(ge_dense_mask, ge_values);
            var greater_equal_dense_mask = try lhs_matrix.greaterEqualDense(rhs_dense);
            defer greater_equal_dense_mask.deinit();
            try expectMask(greater_equal_dense_mask, ge_values);
            var lt_dense_mask = try lhs_matrix.ltDense(rhs_dense);
            defer lt_dense_mask.deinit();
            try expectMask(lt_dense_mask, lt_values);
            var less_dense_mask = try lhs_matrix.lessDense(rhs_dense);
            defer less_dense_mask.deinit();
            try expectMask(less_dense_mask, lt_values);
            var le_dense_mask = try lhs_matrix.leDense(rhs_dense);
            defer le_dense_mask.deinit();
            try expectMask(le_dense_mask, le_values);
            var less_equal_dense_mask = try lhs_matrix.lessEqualDense(rhs_dense);
            defer less_equal_dense_mask.deinit();
            try expectMask(less_equal_dense_mask, le_values);
        }
    }.check;
    const expectScalarCloseness = struct {
        fn expectMask(mask: array_mod.Array(bool), values: []const bool) !void {
            try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, mask.shape);
            try std.testing.expectEqualSlices(bool, values, mask.data);
        }

        fn check(comptime Matrix: type, matrix: Matrix) !void {
            var exact_zero_mask = try matrix.iscloseScalar(0, 0, 0);
            defer exact_zero_mask.deinit();
            try expectMask(exact_zero_mask, &.{ false, true, true, true, false, false });
            var exact_zero_alias = try matrix.isCloseScalar(0, 0, 0);
            defer exact_zero_alias.deinit();
            try expectMask(exact_zero_alias, exact_zero_mask.data);
            try std.testing.expect(!(try matrix.allcloseScalar(0, 0, 0)));
            try std.testing.expect(!(try matrix.allCloseScalar(0, 0, 0)));

            var loose_zero_mask = try matrix.iscloseScalarEqualNan(0, 0, 3, false);
            defer loose_zero_mask.deinit();
            try expectMask(loose_zero_mask, &.{ true, true, true, true, true, true });
            var loose_zero_alias = try matrix.isCloseScalarEqualNan(0, 0, 3, false);
            defer loose_zero_alias.deinit();
            try expectMask(loose_zero_alias, loose_zero_mask.data);
            try std.testing.expect(try matrix.allcloseScalarEqualNan(0, 0, 3, false));
            try std.testing.expect(try matrix.allCloseScalarEqualNan(0, 0, 3, false));
        }
    }.check;
    const expectNonzero = struct {
        fn check(comptime Matrix: type, matrix: Matrix) !void {
            try std.testing.expectEqual(@as(usize, 3), try matrix.countNonzero());

            var axis0 = try matrix.countNonzeroAxis(0, false);
            defer axis0.deinit();
            try std.testing.expectEqualSlices(usize, &.{3}, axis0.shape);
            try std.testing.expectEqualSlices(usize, &.{ 1, 1, 1 }, axis0.data);

            var axis1_keepdim = try matrix.countNonzeroDim(1, true);
            defer axis1_keepdim.deinit();
            try std.testing.expectEqualSlices(usize, &.{ 2, 1 }, axis1_keepdim.shape);
            try std.testing.expectEqualSlices(usize, &.{ 1, 2 }, axis1_keepdim.data);

            var all_axes = try matrix.countNonzeroAxes(&.{ 0, 1 }, false);
            defer all_axes.deinit();
            try std.testing.expectEqualSlices(usize, &.{}, all_axes.shape);
            try std.testing.expectEqualSlices(usize, &.{3}, all_axes.data);

            var no_axes = try matrix.countNonzeroDims(&.{}, false);
            defer no_axes.deinit();
            try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, no_axes.shape);
            try std.testing.expectEqualSlices(usize, &.{ 1, 0, 0, 0, 1, 1 }, no_axes.data);

            var flat = try matrix.flatNonzero();
            defer flat.deinit();
            try std.testing.expectEqualSlices(usize, &.{3}, flat.shape);
            try std.testing.expectEqualSlices(usize, &.{ 0, 4, 5 }, flat.data);

            var nonzero = try matrix.nonzero();
            defer nonzero.deinit();
            try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, nonzero.shape);
            try std.testing.expectEqualSlices(usize, &.{ 0, 0, 1, 1, 1, 2 }, nonzero.data);

            var argwhere = try matrix.argwhere();
            defer argwhere.deinit();
            try std.testing.expectEqualSlices(usize, nonzero.shape, argwhere.shape);
            try std.testing.expectEqualSlices(usize, nonzero.data, argwhere.data);

            var where_indices = try matrix.whereIndices();
            defer where_indices.deinit();
            try std.testing.expectEqualSlices(usize, nonzero.shape, where_indices.shape);
            try std.testing.expectEqualSlices(usize, nonzero.data, where_indices.data);
        }
    }.check;
    const expectWhere = struct {
        fn expectArray(values: array_mod.Array(f64), shape: []const usize, expected: []const f64) !void {
            try std.testing.expectEqualSlices(usize, shape, values.shape);
            try std.testing.expectEqualSlices(f64, expected, values.data);
        }

        fn expectMatrix(values: array_mod.Array(f64), expected: []const f64) !void {
            try expectArray(values, &.{ 2, 3 }, expected);
        }

        fn check(comptime Matrix: type, matrix: Matrix, rhs_matrix: Matrix, rhs_dense: array_mod.Array(f64), mask: array_mod.Array(bool)) !void {
            var selected = try matrix.where(mask, rhs_matrix);
            defer selected.deinit();
            try expectMatrix(selected, &.{ 1, 0, 0, 0, 2, 6 });

            var selected_array = try matrix.whereArray(mask, rhs_dense);
            defer selected_array.deinit();
            try expectMatrix(selected_array, &.{ 1, 0, 0, 0, 2, 6 });

            var selected_scalar = try matrix.whereScalar(mask, -9);
            defer selected_scalar.deinit();
            try expectMatrix(selected_scalar, &.{ 1, -9, 0, -9, 2, -9 });

            var copied = try matrix.copyWhere(mask, rhs_matrix);
            defer copied.deinit();
            try expectMatrix(copied, &.{ 4, 0, 0, 0, -2, 3 });

            var masked = try matrix.maskedSelect(mask);
            defer masked.deinit();
            try std.testing.expectEqualSlices(usize, &.{3}, masked.shape);
            try std.testing.expectEqualSlices(f64, &.{ 1, 0, 2 }, masked.data);

            var filled = try matrix.maskedFill(mask, -7);
            defer filled.deinit();
            try expectMatrix(filled, &.{ -7, 0, -7, 0, -7, 3 });

            var put_scalar = try matrix.maskedPutScalar(mask, -8);
            defer put_scalar.deinit();
            try expectMatrix(put_scalar, &.{ -8, 0, -8, 0, -8, 3 });

            var put_mask_scalar = try matrix.putMaskScalar(mask, -6);
            defer put_mask_scalar.deinit();
            try expectMatrix(put_mask_scalar, &.{ -6, 0, -6, 0, -6, 3 });

            var scatter_values = try array_mod.Array(f64).fromSlice(matrix.allocator, &.{ 7, 8, 9 }, &.{3});
            defer scatter_values.deinit();
            var scattered = try matrix.maskedScatter(mask, scatter_values);
            defer scattered.deinit();
            try expectMatrix(scattered, &.{ 7, 0, 8, 0, 9, 3 });

            var put_values = try array_mod.Array(f64).fromSlice(matrix.allocator, &.{ 10, 11, 12 }, &.{3});
            defer put_values.deinit();
            var put = try matrix.maskedPut(mask, put_values);
            defer put.deinit();
            try expectMatrix(put, &.{ 10, 0, 11, 0, 12, 3 });

            var put_alias = try matrix.putMask(mask, put_values);
            defer put_alias.deinit();
            try expectMatrix(put_alias, &.{ 10, 0, 11, 0, 12, 3 });

            var put_scalar_array = try array_mod.Array(f64).fromSlice(matrix.allocator, &.{13}, &.{1});
            defer put_scalar_array.deinit();
            var put_scalar_broadcast = try matrix.maskedPut(mask, put_scalar_array);
            defer put_scalar_broadcast.deinit();
            try expectMatrix(put_scalar_broadcast, &.{ 13, 0, 13, 0, 13, 3 });

            var bad_values = try array_mod.Array(f64).fromSlice(matrix.allocator, &.{ 1, 2 }, &.{2});
            defer bad_values.deinit();
            try std.testing.expectError(error.ShapeMismatch, matrix.maskedScatter(mask, bad_values));
            try std.testing.expectError(error.ShapeMismatch, matrix.maskedPut(mask, bad_values));

            var row_condition = try array_mod.Array(bool).fromSlice(matrix.allocator, &.{ true, false }, &.{2});
            defer row_condition.deinit();
            var compressed_rows = try matrix.compress(row_condition, 0);
            defer compressed_rows.deinit();
            try std.testing.expectEqualSlices(usize, &.{ 1, 3 }, compressed_rows.shape);
            try std.testing.expectEqualSlices(f64, &.{ 1, 0, 0 }, compressed_rows.data);

            var column_condition = try array_mod.Array(bool).fromSlice(matrix.allocator, &.{ true, false, true }, &.{3});
            defer column_condition.deinit();
            var compressed_cols = try matrix.compress(column_condition, 1);
            defer compressed_cols.deinit();
            try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, compressed_cols.shape);
            try std.testing.expectEqualSlices(f64, &.{ 1, 0, 0, 3 }, compressed_cols.data);

            var flat_condition = try array_mod.Array(bool).fromSlice(matrix.allocator, &.{ true, false, true, false, true, false }, &.{6});
            defer flat_condition.deinit();
            var compressed_flat = try matrix.compress(flat_condition, null);
            defer compressed_flat.deinit();
            try std.testing.expectEqualSlices(usize, &.{3}, compressed_flat.shape);
            try std.testing.expectEqualSlices(f64, &.{ 1, 0, 2 }, compressed_flat.data);

            var flat_indices = try array_mod.Array(usize).fromSlice(matrix.allocator, &.{ 5, 0, 4 }, &.{3});
            defer flat_indices.deinit();
            var flat_take = try matrix.take(flat_indices, null);
            defer flat_take.deinit();
            try expectArray(flat_take, &.{3}, &.{ 3, 1, 2 });

            var row_indices = try array_mod.Array(usize).fromSlice(matrix.allocator, &.{ 1, 0 }, &.{2});
            defer row_indices.deinit();
            var row_take = try matrix.take(row_indices, 0);
            defer row_take.deinit();
            try expectArray(row_take, &.{ 2, 3 }, &.{ 0, 2, 3, 1, 0, 0 });

            var column_indices = try array_mod.Array(usize).fromSlice(matrix.allocator, &.{ 2, 0 }, &.{2});
            defer column_indices.deinit();
            var column_take = try matrix.indexSelect(1, column_indices);
            defer column_take.deinit();
            try expectArray(column_take, &.{ 2, 2 }, &.{ 0, 1, 3, 0 });

            var signed_flat_indices = try array_mod.Array(isize).fromSlice(matrix.allocator, &.{ -1, 0 }, &.{2});
            defer signed_flat_indices.deinit();
            var signed_flat_take = try matrix.takeSigned(signed_flat_indices, null);
            defer signed_flat_take.deinit();
            try expectArray(signed_flat_take, &.{2}, &.{ 3, 1 });

            var signed_column_indices = try array_mod.Array(isize).fromSlice(matrix.allocator, &.{ -1, 0 }, &.{2});
            defer signed_column_indices.deinit();
            var signed_column_take = try matrix.indexSelectSigned(1, signed_column_indices);
            defer signed_column_take.deinit();
            try expectArray(signed_column_take, &.{ 2, 2 }, &.{ 0, 1, 3, 0 });

            var wrapped_flat_indices = try array_mod.Array(usize).fromSlice(matrix.allocator, &.{ 6, 7 }, &.{2});
            defer wrapped_flat_indices.deinit();
            var wrapped_flat_take = try matrix.takeMode(wrapped_flat_indices, null, .wrap);
            defer wrapped_flat_take.deinit();
            try expectArray(wrapped_flat_take, &.{2}, &.{ 1, 0 });

            var clipped_signed_columns = try array_mod.Array(isize).fromSlice(matrix.allocator, &.{ -4, 99 }, &.{2});
            defer clipped_signed_columns.deinit();
            var clipped_signed_take = try matrix.takeSignedMode(clipped_signed_columns, 1, .clip);
            defer clipped_signed_take.deinit();
            try expectArray(clipped_signed_take, &.{ 2, 2 }, &.{ 1, 0, 0, 3 });

            var bad_take_indices = try array_mod.Array(usize).fromSlice(matrix.allocator, &.{6}, &.{1});
            defer bad_take_indices.deinit();
            try std.testing.expectError(error.IndexOutOfBounds, matrix.take(bad_take_indices, null));

            var gather_indices = try array_mod.Array(usize).fromSlice(matrix.allocator, &.{
                0, 0, 0,
                2, 1, 0,
            }, &.{ 2, 3 });
            defer gather_indices.deinit();
            var gathered = try matrix.gather(1, gather_indices);
            defer gathered.deinit();
            try expectArray(gathered, &.{ 2, 3 }, &.{ 1, 1, 1, 3, 2, 0 });

            var take_along = try matrix.takeAlongAxis(gather_indices, 1);
            defer take_along.deinit();
            try expectArray(take_along, &.{ 2, 3 }, &.{ 1, 1, 1, 3, 2, 0 });

            var signed_gather_indices = try array_mod.Array(isize).fromSlice(matrix.allocator, &.{
                -1, 0,  -2,
                0,  -2, -1,
            }, &.{ 2, 3 });
            defer signed_gather_indices.deinit();
            var signed_gathered = try matrix.gatherSigned(1, signed_gather_indices);
            defer signed_gathered.deinit();
            try expectArray(signed_gathered, &.{ 2, 3 }, &.{ 0, 1, 0, 0, 2, 3 });

            var signed_take_along = try matrix.takeAlongAxisSigned(signed_gather_indices, 1);
            defer signed_take_along.deinit();
            try expectArray(signed_take_along, &.{ 2, 3 }, &.{ 0, 1, 0, 0, 2, 3 });

            var bad_gather_indices = try array_mod.Array(usize).fromSlice(matrix.allocator, &.{
                0, 3, 0,
                0, 0, 0,
            }, &.{ 2, 3 });
            defer bad_gather_indices.deinit();
            try std.testing.expectError(error.IndexOutOfBounds, matrix.gather(1, bad_gather_indices));

            var bad_gather_shape = try array_mod.Array(usize).fromSlice(matrix.allocator, &.{ 0, 0, 0 }, &.{3});
            defer bad_gather_shape.deinit();
            try std.testing.expectError(error.ShapeMismatch, matrix.takeAlongAxis(bad_gather_shape, 1));

            var put_indices = try array_mod.Array(usize).fromSlice(matrix.allocator, &.{ 1, 3, 5 }, &.{3});
            defer put_indices.deinit();
            var scatter_indices = try array_mod.Array(usize).fromSlice(matrix.allocator, &.{
                0, 1, 2,
                2, 1, 0,
            }, &.{ 2, 3 });
            defer scatter_indices.deinit();
            var scatter_src = try array_mod.Array(f64).fromSlice(matrix.allocator, &.{
                9, 8, 7,
                6, 5, 4,
            }, &.{ 2, 3 });
            defer scatter_src.deinit();
            var scatter_out = try matrix.scatter(1, scatter_indices, scatter_src);
            defer scatter_out.deinit();
            try expectArray(scatter_out, &.{ 2, 3 }, &.{ 9, 8, 7, 4, 5, 6 });

            var put_along = try matrix.putAlongAxis(scatter_indices, scatter_src, 1);
            defer put_along.deinit();
            try expectArray(put_along, &.{ 2, 3 }, &.{ 9, 8, 7, 4, 5, 6 });

            var scatter_scalar = try matrix.scatterScalar(1, scatter_indices, -5);
            defer scatter_scalar.deinit();
            try expectArray(scatter_scalar, &.{ 2, 3 }, &.{ -5, -5, -5, -5, -5, -5 });

            var scatter_sum = try matrix.scatterReduce(1, scatter_indices, scatter_src, .sum);
            defer scatter_sum.deinit();
            try expectArray(scatter_sum, &.{ 2, 3 }, &.{ 10, 8, 7, 4, 7, 9 });

            var scatter_add = try matrix.scatterAdd(1, scatter_indices, scatter_src);
            defer scatter_add.deinit();
            try expectArray(scatter_add, &.{ 2, 3 }, &.{ 10, 8, 7, 4, 7, 9 });

            var scatter_max = try matrix.scatterReduce(1, scatter_indices, scatter_src, .max);
            defer scatter_max.deinit();
            try expectArray(scatter_max, &.{ 2, 3 }, &.{ 9, 8, 7, 4, 5, 6 });

            var scatter_scalar_sum = try matrix.scatterReduceScalar(1, scatter_indices, 2, .sum);
            defer scatter_scalar_sum.deinit();
            try expectArray(scatter_scalar_sum, &.{ 2, 3 }, &.{ 3, 2, 2, 2, 4, 5 });

            var bad_scatter_indices = try array_mod.Array(usize).fromSlice(matrix.allocator, &.{
                0, 3, 0,
                0, 0, 0,
            }, &.{ 2, 3 });
            defer bad_scatter_indices.deinit();
            try std.testing.expectError(error.IndexOutOfBounds, matrix.scatter(1, bad_scatter_indices, scatter_src));

            var bad_scatter_src = try array_mod.Array(f64).fromSlice(matrix.allocator, &.{ 1, 2, 3 }, &.{3});
            defer bad_scatter_src.deinit();
            try std.testing.expectError(error.ShapeMismatch, matrix.scatter(1, scatter_indices, bad_scatter_src));

            var selected_row = try matrix.select(0, 1);
            defer selected_row.deinit();
            try expectArray(selected_row, &.{3}, &.{ 0, 2, 3 });

            var selected_column = try matrix.selectSigned(1, -1);
            defer selected_column.deinit();
            try expectArray(selected_column, &.{2}, &.{ 0, 3 });

            var narrowed_rows = try matrix.narrow(0, 0, 1);
            defer narrowed_rows.deinit();
            try expectArray(narrowed_rows, &.{ 1, 3 }, &.{ 1, 0, 0 });

            var narrowed_columns = try matrix.narrowSigned(1, -2, 2);
            defer narrowed_columns.deinit();
            try expectArray(narrowed_columns, &.{ 2, 2 }, &.{ 0, 0, 2, 3 });

            try std.testing.expectError(error.IndexOutOfBounds, matrix.select(0, 2));
            try std.testing.expectError(error.IndexOutOfBounds, matrix.narrow(1, 2, 2));

            var put_flat_values = try array_mod.Array(f64).fromSlice(matrix.allocator, &.{ 20, 21, 22 }, &.{3});
            defer put_flat_values.deinit();
            var put_flat = try matrix.putFlat(put_indices, put_flat_values);
            defer put_flat.deinit();
            try expectArray(put_flat, &.{ 2, 3 }, &.{ 1, 20, 0, 21, 2, 22 });

            var put_flat_alias = try matrix.indexPut(put_indices, put_flat_values);
            defer put_flat_alias.deinit();
            try expectArray(put_flat_alias, &.{ 2, 3 }, &.{ 1, 20, 0, 21, 2, 22 });

            var put_flat_scalar = try matrix.putFlatScalar(put_indices, 30);
            defer put_flat_scalar.deinit();
            try expectArray(put_flat_scalar, &.{ 2, 3 }, &.{ 1, 30, 0, 30, 2, 30 });

            var put_flat_scalar_alias = try matrix.indexPutScalar(put_indices, 31);
            defer put_flat_scalar_alias.deinit();
            try expectArray(put_flat_scalar_alias, &.{ 2, 3 }, &.{ 1, 31, 0, 31, 2, 31 });

            var scalar_values = try array_mod.Array(f64).fromSlice(matrix.allocator, &.{40}, &.{1});
            defer scalar_values.deinit();
            var put_flat_broadcast = try matrix.putFlat(put_indices, scalar_values);
            defer put_flat_broadcast.deinit();
            try expectArray(put_flat_broadcast, &.{ 2, 3 }, &.{ 1, 40, 0, 40, 2, 40 });

            var wrapped_put_indices = try array_mod.Array(usize).fromSlice(matrix.allocator, &.{ 6, 7 }, &.{2});
            defer wrapped_put_indices.deinit();
            var mode_values = try array_mod.Array(f64).fromSlice(matrix.allocator, &.{ 50, 51 }, &.{2});
            defer mode_values.deinit();
            var put_mode = try matrix.putFlatMode(wrapped_put_indices, mode_values, .wrap);
            defer put_mode.deinit();
            try expectArray(put_mode, &.{ 2, 3 }, &.{ 50, 51, 0, 0, 2, 3 });

            var put_flat_scalar_mode = try matrix.putFlatScalarMode(wrapped_put_indices, 60, .wrap);
            defer put_flat_scalar_mode.deinit();
            try expectArray(put_flat_scalar_mode, &.{ 2, 3 }, &.{ 60, 60, 0, 0, 2, 3 });

            var signed_put_indices = try array_mod.Array(isize).fromSlice(matrix.allocator, &.{ -1, 0 }, &.{2});
            defer signed_put_indices.deinit();
            var signed_values = try array_mod.Array(f64).fromSlice(matrix.allocator, &.{ 70, 71 }, &.{2});
            defer signed_values.deinit();
            var put_signed = try matrix.putFlatSigned(signed_put_indices, signed_values);
            defer put_signed.deinit();
            try expectArray(put_signed, &.{ 2, 3 }, &.{ 71, 0, 0, 0, 2, 70 });

            var put_signed_scalar = try matrix.putFlatScalarSigned(signed_put_indices, 80);
            defer put_signed_scalar.deinit();
            try expectArray(put_signed_scalar, &.{ 2, 3 }, &.{ 80, 0, 0, 0, 2, 80 });

            var coords = try array_mod.Array(usize).fromSlice(matrix.allocator, &.{
                0, 0,
                1, 1,
                1, 2,
            }, &.{ 3, 2 });
            defer coords.deinit();
            var flat_from_coords = try matrix.ravelCoords(coords);
            defer flat_from_coords.deinit();
            try std.testing.expectEqualSlices(usize, &.{3}, flat_from_coords.shape);
            try std.testing.expectEqualSlices(usize, &.{ 0, 4, 5 }, flat_from_coords.data);

            var coords_roundtrip = try matrix.unravelFlat(flat_from_coords);
            defer coords_roundtrip.deinit();
            try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, coords_roundtrip.shape);
            try std.testing.expectEqualSlices(usize, coords.data, coords_roundtrip.data);

            var coord_values = try matrix.takeCoords(coords);
            defer coord_values.deinit();
            try expectArray(coord_values, &.{3}, &.{ 1, 2, 3 });

            var coord_replacements = try array_mod.Array(f64).fromSlice(matrix.allocator, &.{ 90, 91, 92 }, &.{3});
            defer coord_replacements.deinit();
            var coord_put = try matrix.putCoords(coords, coord_replacements);
            defer coord_put.deinit();
            try expectArray(coord_put, &.{ 2, 3 }, &.{ 90, 0, 0, 0, 91, 92 });

            var coord_scalar_put = try matrix.putCoordsScalar(coords, 99);
            defer coord_scalar_put.deinit();
            try expectArray(coord_scalar_put, &.{ 2, 3 }, &.{ 99, 0, 0, 0, 99, 99 });

            var bad_coords = try array_mod.Array(usize).fromSlice(matrix.allocator, &.{ 0, 3 }, &.{ 1, 2 });
            defer bad_coords.deinit();
            try std.testing.expectError(error.IndexOutOfBounds, matrix.takeCoords(bad_coords));

            var bad_coord_shape = try array_mod.Array(usize).fromSlice(matrix.allocator, &.{ 0, 0, 0 }, &.{3});
            defer bad_coord_shape.deinit();
            try std.testing.expectError(error.ShapeMismatch, matrix.ravelCoords(bad_coord_shape));

            var bad_unravel = try array_mod.Array(usize).fromSlice(matrix.allocator, &.{6}, &.{1});
            defer bad_unravel.deinit();
            try std.testing.expectError(error.IndexOutOfBounds, matrix.unravelFlat(bad_unravel));

            var reshaped = try matrix.reshape(&.{ 3, 2 });
            defer reshaped.deinit();
            try expectArray(reshaped, &.{ 3, 2 }, &.{ 1, 0, 0, 0, 2, 3 });

            var reshaped_infer = try matrix.reshapeInfer(&.{ -1, 2 });
            defer reshaped_infer.deinit();
            try expectArray(reshaped_infer, &.{ 3, 2 }, reshaped.data);

            var reshape_target = try array_mod.Array(f64).zeros(matrix.allocator, &.{ 3, 2 });
            defer reshape_target.deinit();
            var reshaped_as = try matrix.reshapeAs(reshape_target);
            defer reshaped_as.deinit();
            try expectArray(reshaped_as, &.{ 3, 2 }, reshaped.data);

            var viewed = try matrix.view(&.{ 3, 2 });
            defer viewed.deinit();
            try expectArray(viewed, &.{ 3, 2 }, reshaped.data);

            var viewed_infer = try matrix.viewInfer(&.{ -1, 2 });
            defer viewed_infer.deinit();
            try expectArray(viewed_infer, &.{ 3, 2 }, reshaped.data);

            var viewed_as = try matrix.viewAs(reshape_target);
            defer viewed_as.deinit();
            try expectArray(viewed_as, &.{ 3, 2 }, reshaped.data);

            var flattened = try matrix.flatten();
            defer flattened.deinit();
            try expectArray(flattened, &.{6}, &.{ 1, 0, 0, 0, 2, 3 });

            var flatten_axes = try matrix.flattenAxes(0, 1);
            defer flatten_axes.deinit();
            try expectArray(flatten_axes, &.{6}, flattened.data);

            var flatten_range = try matrix.flattenRange(0, -1);
            defer flatten_range.deinit();
            try expectArray(flatten_range, &.{6}, flattened.data);

            var flatten_from = try matrix.flattenFrom(0);
            defer flatten_from.deinit();
            try expectArray(flatten_from, &.{6}, flattened.data);

            var raveled = try matrix.ravel();
            defer raveled.deinit();
            try expectArray(raveled, &.{6}, flattened.data);

            var at_least_1d = try matrix.atLeast1d();
            defer at_least_1d.deinit();
            try expectArray(at_least_1d, &.{ 2, 3 }, &.{ 1, 0, 0, 0, 2, 3 });

            var at_least_2d = try matrix.atLeast2d();
            defer at_least_2d.deinit();
            try expectArray(at_least_2d, &.{ 2, 3 }, &.{ 1, 0, 0, 0, 2, 3 });

            var at_least_3d = try matrix.atLeast3d();
            defer at_least_3d.deinit();
            try expectArray(at_least_3d, &.{ 2, 3, 1 }, &.{ 1, 0, 0, 0, 2, 3 });

            var unflattened = try matrix.unflatten(1, &.{ 3, 1 });
            defer unflattened.deinit();
            try expectArray(unflattened, &.{ 2, 3, 1 }, &.{ 1, 0, 0, 0, 2, 3 });

            var unsqueezed = try matrix.unsqueeze(0);
            defer unsqueezed.deinit();
            try expectArray(unsqueezed, &.{ 1, 2, 3 }, &.{ 1, 0, 0, 0, 2, 3 });

            var unsqueezed_dim = try matrix.unsqueezeDim(-1);
            defer unsqueezed_dim.deinit();
            try expectArray(unsqueezed_dim, &.{ 2, 3, 1 }, &.{ 1, 0, 0, 0, 2, 3 });

            var squeezed_back = try unsqueezed.squeeze(0);
            defer squeezed_back.deinit();
            try expectArray(squeezed_back, &.{ 2, 3 }, &.{ 1, 0, 0, 0, 2, 3 });

            var squeezed_dim = try unsqueezed_dim.squeezeDim(-1);
            defer squeezed_dim.deinit();
            try expectArray(squeezed_dim, &.{ 2, 3 }, &.{ 1, 0, 0, 0, 2, 3 });

            var expanded = try matrix.expandDims(&.{ 0, -1 });
            defer expanded.deinit();
            try expectArray(expanded, &.{ 1, 2, 3, 1 }, &.{ 1, 0, 0, 0, 2, 3 });

            var unsqueezed_axes = try matrix.unsqueezeAxes(&.{ 0, 3 });
            defer unsqueezed_axes.deinit();
            try expectArray(unsqueezed_axes, &.{ 1, 2, 3, 1 }, &.{ 1, 0, 0, 0, 2, 3 });

            var squeezed_axes = try expanded.squeezeAxes(&.{ 0, 3 });
            defer squeezed_axes.deinit();
            try expectArray(squeezed_axes, &.{ 2, 3 }, &.{ 1, 0, 0, 0, 2, 3 });

            var broadcasted = try matrix.broadcastTo(&.{ 1, 2, 3 });
            defer broadcasted.deinit();
            try expectArray(broadcasted, &.{ 1, 2, 3 }, &.{ 1, 0, 0, 0, 2, 3 });

            var expanded_to = try matrix.expand(&.{ 1, 2, 3 });
            defer expanded_to.deinit();
            try expectArray(expanded_to, &.{ 1, 2, 3 }, broadcasted.data);

            var broadcast_target = try array_mod.Array(f64).zeros(matrix.allocator, &.{ 1, 2, 3 });
            defer broadcast_target.deinit();
            var expanded_as = try matrix.expandAs(broadcast_target);
            defer expanded_as.deinit();
            try expectArray(expanded_as, &.{ 1, 2, 3 }, broadcasted.data);

            var broadcast_as = try matrix.broadcastAs(broadcast_target);
            defer broadcast_as.deinit();
            try expectArray(broadcast_as, &.{ 1, 2, 3 }, broadcasted.data);

            var permuted = try matrix.permute(&.{ 1, 0 });
            defer permuted.deinit();
            try expectArray(permuted, &.{ 3, 2 }, &.{
                1, 0,
                0, 2,
                0, 3,
            });

            var swapped = try matrix.swapaxes(0, 1);
            defer swapped.deinit();
            try expectArray(swapped, &.{ 3, 2 }, permuted.data);

            var swap_dims = try matrix.swapDims(0, 1);
            defer swap_dims.deinit();
            try expectArray(swap_dims, &.{ 3, 2 }, permuted.data);

            var moved = try matrix.movedim(0, 1);
            defer moved.deinit();
            try expectArray(moved, &.{ 3, 2 }, permuted.data);

            var move_axis = try matrix.moveaxis(0, 1);
            defer move_axis.deinit();
            try expectArray(move_axis, &.{ 3, 2 }, permuted.data);

            try std.testing.expectError(error.ShapeMismatch, matrix.broadcastTo(&.{ 2, 2 }));
            try std.testing.expectError(error.InvalidPermutation, matrix.permute(&.{ 0, 0 }));

            var flipped_rows = try matrix.flip(0);
            defer flipped_rows.deinit();
            try expectArray(flipped_rows, &.{ 2, 3 }, &.{ 0, 2, 3, 1, 0, 0 });

            var flipped_columns = try matrix.fliplr();
            defer flipped_columns.deinit();
            try expectArray(flipped_columns, &.{ 2, 3 }, &.{ 0, 0, 1, 3, 2, 0 });

            var flipped_axes = try matrix.flipAxes(&.{ 0, 1 });
            defer flipped_axes.deinit();
            try expectArray(flipped_axes, &.{ 2, 3 }, &.{ 3, 2, 0, 0, 0, 1 });

            var flipped_up = try matrix.flipud();
            defer flipped_up.deinit();
            try expectArray(flipped_up, &.{ 2, 3 }, &.{ 0, 2, 3, 1, 0, 0 });

            var rolled_columns = try matrix.roll(1, 1);
            defer rolled_columns.deinit();
            try expectArray(rolled_columns, &.{ 2, 3 }, &.{ 0, 1, 0, 3, 0, 2 });

            var rolled_flat = try matrix.rollFlat(1);
            defer rolled_flat.deinit();
            try expectArray(rolled_flat, &.{ 2, 3 }, &.{ 3, 1, 0, 0, 0, 2 });

            var rolled_axes = try matrix.rollAxes(&.{ 1, -1 }, &.{ 0, 1 });
            defer rolled_axes.deinit();
            try expectArray(rolled_axes, &.{ 2, 3 }, &.{ 2, 3, 0, 0, 0, 1 });

            var rotated = try matrix.rot90(1, .{ 0, 1 });
            defer rotated.deinit();
            try expectArray(rotated, &.{ 3, 2 }, &.{
                0, 3,
                0, 2,
                1, 0,
            });
            try std.testing.expectError(error.InvalidAxis, matrix.rot90(1, .{ 0, 0 }));

            var padded_constant = try matrix.padConstant(&.{ 1, 1 }, &.{ 0, 1 }, -1);
            defer padded_constant.deinit();
            try expectArray(padded_constant, &.{ 3, 5 }, &.{
                -1, -1, -1, -1, -1,
                -1, 1,  0,  0,  -1,
                -1, 0,  2,  3,  -1,
            });

            var padded_edge = try matrix.padEdge(&.{ 1, 1 }, &.{ 0, 1 });
            defer padded_edge.deinit();
            try expectArray(padded_edge, &.{ 3, 5 }, &.{
                1, 1, 0, 0, 0,
                1, 1, 0, 0, 0,
                0, 0, 2, 3, 3,
            });

            var padded_reflect = try matrix.padReflect(&.{ 1, 1 }, &.{ 0, 1 });
            defer padded_reflect.deinit();
            try expectArray(padded_reflect, &.{ 3, 5 }, &.{
                2, 0, 2, 3, 2,
                0, 1, 0, 0, 0,
                2, 0, 2, 3, 2,
            });

            var padded_wrap = try matrix.padWrap(&.{ 1, 1 }, &.{ 0, 1 });
            defer padded_wrap.deinit();
            try expectArray(padded_wrap, &.{ 3, 5 }, &.{
                3, 0, 2, 3, 0,
                0, 1, 0, 0, 1,
                3, 0, 2, 3, 0,
            });

            var padded_symmetric = try matrix.padSymmetric(&.{ 1, 1 }, &.{ 0, 1 });
            defer padded_symmetric.deinit();
            try expectArray(padded_symmetric, &.{ 3, 5 }, &.{
                1, 1, 0, 0, 0,
                1, 1, 0, 0, 0,
                0, 0, 2, 3, 3,
            });

            try std.testing.expectError(error.ShapeMismatch, matrix.padConstant(&.{1}, &.{ 0, 0 }, -1));
            try std.testing.expectError(error.InvalidShape, matrix.padReflect(&.{ 2, 0 }, &.{ 0, 0 }));

            var split_rows = try matrix.split(1, 0);
            defer split_rows.deinit();
            try std.testing.expectEqual(@as(usize, 2), split_rows.items.len);
            try expectArray(split_rows.items[0], &.{ 1, 3 }, &.{ 1, 0, 0 });
            try expectArray(split_rows.items[1], &.{ 1, 3 }, &.{ 0, 2, 3 });

            var split_columns = try matrix.splitWithSizes(&.{ 1, 2 }, 1);
            defer split_columns.deinit();
            try std.testing.expectEqual(@as(usize, 2), split_columns.items.len);
            try expectArray(split_columns.items[0], &.{ 2, 1 }, &.{ 1, 0 });
            try expectArray(split_columns.items[1], &.{ 2, 2 }, &.{ 0, 0, 2, 3 });

            var split_at_columns = try matrix.splitAtIndices(&.{ 1, 2 }, 1);
            defer split_at_columns.deinit();
            try std.testing.expectEqual(@as(usize, 3), split_at_columns.items.len);
            try expectArray(split_at_columns.items[0], &.{ 2, 1 }, &.{ 1, 0 });
            try expectArray(split_at_columns.items[1], &.{ 2, 1 }, &.{ 0, 2 });
            try expectArray(split_at_columns.items[2], &.{ 2, 1 }, &.{ 0, 3 });

            var chunks = try matrix.chunk(2, 1);
            defer chunks.deinit();
            try std.testing.expectEqual(@as(usize, 2), chunks.items.len);
            try expectArray(chunks.items[0], &.{ 2, 2 }, &.{ 1, 0, 0, 2 });
            try expectArray(chunks.items[1], &.{ 2, 1 }, &.{ 0, 3 });

            var unbound_rows = try matrix.unbind(0);
            defer unbound_rows.deinit();
            try std.testing.expectEqual(@as(usize, 2), unbound_rows.items.len);
            try expectArray(unbound_rows.items[0], &.{3}, &.{ 1, 0, 0 });
            try expectArray(unbound_rows.items[1], &.{3}, &.{ 0, 2, 3 });

            try std.testing.expectError(error.InvalidShape, matrix.split(0, 0));
            try std.testing.expectError(error.ShapeMismatch, matrix.splitWithSizes(&.{ 1, 1 }, 1));
            try std.testing.expectError(error.InvalidShape, matrix.splitAtIndices(&.{ 2, 1 }, 1));
            try std.testing.expectError(error.InvalidShape, matrix.chunk(0, 1));

            var repeated_rows = try matrix.repeat(2, 0);
            defer repeated_rows.deinit();
            try expectArray(repeated_rows, &.{ 4, 3 }, &.{
                1, 0, 0,
                1, 0, 0,
                0, 2, 3,
                0, 2, 3,
            });

            var repeated_columns = try matrix.repeat(2, 1);
            defer repeated_columns.deinit();
            try expectArray(repeated_columns, &.{ 2, 6 }, &.{
                1, 1, 0, 0, 0, 0,
                0, 0, 2, 2, 3, 3,
            });

            var column_repeats = try array_mod.Array(usize).fromSlice(matrix.allocator, &.{ 1, 0, 2 }, &.{3});
            defer column_repeats.deinit();
            var interleaved_columns = try matrix.repeatInterleave(column_repeats, 1);
            defer interleaved_columns.deinit();
            try expectArray(interleaved_columns, &.{ 2, 3 }, &.{ 1, 0, 0, 0, 3, 3 });

            var interleaved_flat = try matrix.repeatInterleaveScalar(2, null);
            defer interleaved_flat.deinit();
            try expectArray(interleaved_flat, &.{12}, &.{ 1, 1, 0, 0, 0, 0, 0, 0, 2, 2, 3, 3 });

            var interleaved_axis_scalar = try matrix.repeatInterleaveScalar(2, -1);
            defer interleaved_axis_scalar.deinit();
            try expectArray(interleaved_axis_scalar, &.{ 2, 6 }, &.{
                1, 1, 0, 0, 0, 0,
                0, 0, 2, 2, 3, 3,
            });

            var bad_repeats = try array_mod.Array(usize).fromSlice(matrix.allocator, &.{ 1, 2 }, &.{2});
            defer bad_repeats.deinit();
            try std.testing.expectError(error.ShapeMismatch, matrix.repeatInterleave(bad_repeats, 1));

            var tiled = try matrix.tile(&.{ 2, 1 });
            defer tiled.deinit();
            try expectArray(tiled, &.{ 4, 3 }, &.{
                1, 0, 0,
                0, 2, 3,
                1, 0, 0,
                0, 2, 3,
            });

            try std.testing.expectError(error.ShapeMismatch, matrix.reshape(&.{ 4, 2 }));

            var bad_put_values = try array_mod.Array(f64).fromSlice(matrix.allocator, &.{ 1, 2 }, &.{2});
            defer bad_put_values.deinit();
            try std.testing.expectError(error.ShapeMismatch, matrix.putFlat(put_indices, bad_put_values));
            try std.testing.expectError(error.IndexOutOfBounds, matrix.putFlat(bad_take_indices, scalar_values));
        }
    }.check;

    var lhs = try cooFromSlices(f64, gpa, 2, 3, &.{ 1, 0, 1 }, &.{ 2, 0, 1 }, &.{ 3, 1, 2 });
    defer lhs.deinit();
    var rhs = try cooFromSlices(f64, gpa, 2, 3, &.{ 1, 0, 1, 1 }, &.{ 2, 0, 1, 2 }, &.{ 5, 4, -2, 1 });
    defer rhs.deinit();

    var coo_sum = try lhs.add(rhs);
    defer coo_sum.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 1 }, coo_sum.row_indices);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2 }, coo_sum.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 5, 0, 9 }, coo_sum.values);
    var coo_dense = try coo_sum.toDense();
    defer coo_dense.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 5, 0, 0, 0, 0, 9 }, coo_dense.data);
    var coo_pruned = try coo_sum.dropZeros();
    defer coo_pruned.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, coo_pruned.row_indices);
    try std.testing.expectEqualSlices(usize, &.{ 0, 2 }, coo_pruned.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 5, 9 }, coo_pruned.values);
    try std.testing.expectEqual(@as(usize, 2), coo_sum.dropZerosNnz());
    var coo_tolerance_pruned = try coo_sum.pruneZeros(5);
    defer coo_tolerance_pruned.deinit();
    try std.testing.expectEqualSlices(usize, &.{1}, coo_tolerance_pruned.row_indices);
    try std.testing.expectEqualSlices(usize, &.{2}, coo_tolerance_pruned.col_indices);
    try std.testing.expectEqualSlices(f64, &.{9}, coo_tolerance_pruned.values);
    try std.testing.expectEqual(@as(usize, 1), try coo_sum.pruneZerosNnz(5));
    try std.testing.expectError(error.InvalidShape, coo_sum.pruneZeros(std.math.nan(f64)));
    try std.testing.expectError(error.InvalidShape, coo_sum.pruneZerosNnz(std.math.nan(f64)));
    var coo_scaled = try coo_pruned.scale(2);
    defer coo_scaled.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 10, 18 }, coo_scaled.values);
    var coo_row_scaled = try coo_pruned.scaleRows(&.{ 2, 3 });
    defer coo_row_scaled.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 10, 27 }, coo_row_scaled.values);
    var coo_row_scaled_inplace = try coo_pruned.clone();
    defer coo_row_scaled_inplace.deinit();
    try coo_row_scaled_inplace.scaleRowsInPlace(&.{ 2, 3 });
    try std.testing.expectEqualSlices(usize, coo_pruned.row_indices, coo_row_scaled_inplace.row_indices);
    try std.testing.expectEqualSlices(usize, coo_pruned.col_indices, coo_row_scaled_inplace.col_indices);
    try std.testing.expectEqualSlices(f64, coo_row_scaled.values, coo_row_scaled_inplace.values);
    var coo_col_scaled = try coo_pruned.scaleColumns(&.{ 4, 5, 6 });
    defer coo_col_scaled.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 20, 54 }, coo_col_scaled.values);
    var coo_col_scaled_inplace = try coo_pruned.clone();
    defer coo_col_scaled_inplace.deinit();
    try coo_col_scaled_inplace.scaleColumnsInPlace(&.{ 4, 5, 6 });
    try std.testing.expectEqualSlices(f64, coo_col_scaled.values, coo_col_scaled_inplace.values);
    var coo_rc_scaled_inplace = try coo_pruned.clone();
    defer coo_rc_scaled_inplace.deinit();
    try coo_rc_scaled_inplace.scaleRowsAndColumnsInPlace(&.{ 2, 3 }, &.{ 4, 5, 6 });
    try std.testing.expectEqualSlices(f64, &.{ 40, 162 }, coo_rc_scaled_inplace.values);
    try std.testing.expectError(error.ShapeMismatch, coo_rc_scaled_inplace.scaleRowsInPlace(&.{1}));
    var coo_neg = try coo_pruned.neg();
    defer coo_neg.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -5, -9 }, coo_neg.values);
    var coo_diff = try lhs.sub(rhs);
    defer coo_diff.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 1 }, coo_diff.row_indices);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2 }, coo_diff.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ -3, 4, -3 }, coo_diff.values);
    var coo_abs = try coo_diff.abs();
    defer coo_abs.deinit();
    try std.testing.expectEqualSlices(usize, coo_diff.row_indices, coo_abs.row_indices);
    try std.testing.expectEqualSlices(usize, coo_diff.col_indices, coo_abs.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 3, 4, 3 }, coo_abs.values);
    var coo_squared = try coo_diff.square();
    defer coo_squared.deinit();
    try std.testing.expectEqualSlices(usize, coo_diff.row_indices, coo_squared.row_indices);
    try std.testing.expectEqualSlices(usize, coo_diff.col_indices, coo_squared.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 9, 16, 9 }, coo_squared.values);
    var coo_sqrt = try coo_squared.sqrt();
    defer coo_sqrt.deinit();
    try std.testing.expectEqualSlices(usize, coo_squared.row_indices, coo_sqrt.row_indices);
    try std.testing.expectEqualSlices(usize, coo_squared.col_indices, coo_sqrt.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 3, 4, 3 }, coo_sqrt.values);
    var coo_rsqrt = try coo_squared.rsqrt();
    defer coo_rsqrt.deinit();
    try std.testing.expectEqualSlices(usize, coo_squared.row_indices, coo_rsqrt.row_indices);
    try std.testing.expectEqualSlices(usize, coo_squared.col_indices, coo_rsqrt.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 1.0 / 3.0, 0.25, 1.0 / 3.0 }, coo_rsqrt.values);
    var coo_reciprocal = try coo_squared.reciprocal();
    defer coo_reciprocal.deinit();
    try std.testing.expectEqualSlices(usize, coo_squared.row_indices, coo_reciprocal.row_indices);
    try std.testing.expectEqualSlices(usize, coo_squared.col_indices, coo_reciprocal.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 1.0 / 9.0, 1.0 / 16.0, 1.0 / 9.0 }, coo_reciprocal.values);
    var coo_cubes = try cooFromSlices(f64, gpa, 2, 2, &.{ 0, 0, 1 }, &.{ 0, 1, 1 }, &.{ -8, 27, 64 });
    defer coo_cubes.deinit();
    var coo_cbrt = try coo_cubes.cbrt();
    defer coo_cbrt.deinit();
    try std.testing.expectEqualSlices(usize, coo_cubes.row_indices, coo_cbrt.row_indices);
    try std.testing.expectEqualSlices(usize, coo_cubes.col_indices, coo_cbrt.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ -2, 3, 4 }, coo_cbrt.values);
    var coo_positive = try coo_diff.positive();
    defer coo_positive.deinit();
    try std.testing.expectEqualSlices(usize, coo_diff.row_indices, coo_positive.row_indices);
    try std.testing.expectEqualSlices(usize, coo_diff.col_indices, coo_positive.col_indices);
    try std.testing.expectEqualSlices(f64, coo_diff.values, coo_positive.values);
    var coo_sign = try coo_diff.sign();
    defer coo_sign.deinit();
    try std.testing.expectEqualSlices(usize, coo_diff.row_indices, coo_sign.row_indices);
    try std.testing.expectEqualSlices(usize, coo_diff.col_indices, coo_sign.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ -1, 1, -1 }, coo_sign.values);
    var coo_signbit = try coo_diff.signbit();
    defer coo_signbit.deinit();
    try std.testing.expectEqualSlices(usize, coo_diff.row_indices, coo_signbit.row_indices);
    try std.testing.expectEqualSlices(usize, coo_diff.col_indices, coo_signbit.col_indices);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, coo_signbit.values);
    const full_summary = try lhs.diffSummary(rhs);
    try std.testing.expectApproxEqAbs(@as(f64, 18), full_summary.dot, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4), full_summary.max_abs_diff, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2), full_summary.max_rel_diff, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 34), full_summary.squared_distance, 1e-12);
    try std.testing.expectApproxEqAbs(@sqrt(@as(f64, 34)), full_summary.frobeniusDistance(), 1e-12);
    const full_relative = full_summary.relativeFrobeniusDistance();
    try std.testing.expect(try lhs.diffSummaryMeetsBounds(rhs, 4, 2, 34, @sqrt(@as(f64, 34)), full_relative));
    try std.testing.expect(!(try lhs.diffSummaryMeetsBounds(rhs, 3.999, 2, 34, @sqrt(@as(f64, 34)), full_relative)));
    try std.testing.expectError(error.InvalidShape, lhs.diffSummaryMeetsBounds(rhs, 4, std.math.nan(f64), 34, @sqrt(@as(f64, 34)), full_relative));
    try std.testing.expectApproxEqAbs(@as(f64, 4), try lhs.maxAbsDiff(rhs), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2), try lhs.maxRelDiff(rhs), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 34), try lhs.squaredDistance(rhs), 1e-12);
    try std.testing.expectApproxEqAbs(@sqrt(@as(f64, 34)), try lhs.frobeniusDistance(rhs), 1e-12);
    try std.testing.expectApproxEqAbs(full_relative, try lhs.relativeFrobeniusDistance(rhs), 1e-12);
    try std.testing.expect(try lhs.maxAbsDiffMeetsBound(rhs, 4));
    try std.testing.expect(!(try lhs.maxAbsDiffMeetsBound(rhs, 3.999)));
    try std.testing.expect(try lhs.maxRelDiffMeetsBound(rhs, 2));
    try std.testing.expect(try lhs.squaredDistanceMeetsBound(rhs, 34));
    try std.testing.expect(try lhs.frobeniusDistanceMeetsBound(rhs, @sqrt(@as(f64, 34))));
    try std.testing.expect(try lhs.relativeFrobeniusDistanceMeetsBound(rhs, full_relative));
    try std.testing.expectError(error.InvalidShape, lhs.maxAbsDiffMeetsBound(rhs, std.math.nan(f64)));
    var rhs_dense_for_summary = try rhs.toDense();
    defer rhs_dense_for_summary.deinit();
    try expectDenseComparisons(@TypeOf(lhs), lhs, rhs, rhs_dense_for_summary);
    try expectScalarCloseness(@TypeOf(lhs), lhs);
    try expectNonzero(@TypeOf(lhs), lhs);
    var where_mask = try array_mod.Array(bool).fromSlice(gpa, &.{
        true,  false, true,
        false, true,  false,
    }, &.{ 2, 3 });
    defer where_mask.deinit();
    try expectWhere(@TypeOf(lhs), lhs, rhs, rhs_dense_for_summary, where_mask);
    const dense_summary = try lhs.diffSummaryDense(rhs_dense_for_summary);
    try std.testing.expectApproxEqAbs(full_summary.dot, dense_summary.dot, 1e-12);
    try std.testing.expectApproxEqAbs(full_summary.squared_distance, dense_summary.squared_distance, 1e-12);
    try std.testing.expect(try lhs.diffSummaryDenseMeetsBounds(rhs_dense_for_summary, 4, 2, 34, @sqrt(@as(f64, 34)), full_relative));
    try std.testing.expect(!(try lhs.diffSummaryDenseMeetsBounds(rhs_dense_for_summary, 3.999, 2, 34, @sqrt(@as(f64, 34)), full_relative)));
    try std.testing.expectError(error.InvalidShape, lhs.diffSummaryDenseMeetsBounds(rhs_dense_for_summary, 4, std.math.nan(f64), 34, @sqrt(@as(f64, 34)), full_relative));
    try std.testing.expectApproxEqAbs(@as(f64, 4), try lhs.maxAbsDiffDense(rhs_dense_for_summary), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2), try lhs.maxRelDiffDense(rhs_dense_for_summary), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 34), try lhs.squaredDistanceDense(rhs_dense_for_summary), 1e-12);
    try std.testing.expectApproxEqAbs(@sqrt(@as(f64, 34)), try lhs.frobeniusDistanceDense(rhs_dense_for_summary), 1e-12);
    try std.testing.expectApproxEqAbs(full_relative, try lhs.relativeFrobeniusDistanceDense(rhs_dense_for_summary), 1e-12);
    try std.testing.expect(try lhs.maxAbsDiffDenseMeetsBound(rhs_dense_for_summary, 4));
    try std.testing.expect(!(try lhs.maxAbsDiffDenseMeetsBound(rhs_dense_for_summary, 3.999)));
    try std.testing.expect(try lhs.maxRelDiffDenseMeetsBound(rhs_dense_for_summary, 2));
    try std.testing.expect(try lhs.squaredDistanceDenseMeetsBound(rhs_dense_for_summary, 34));
    try std.testing.expect(try lhs.frobeniusDistanceDenseMeetsBound(rhs_dense_for_summary, @sqrt(@as(f64, 34))));
    try std.testing.expect(try lhs.relativeFrobeniusDistanceDenseMeetsBound(rhs_dense_for_summary, full_relative));
    try std.testing.expect(try lhs.allcloseDense(rhs_dense_for_summary, 1, 4));
    try std.testing.expect(try lhs.allCloseDense(rhs_dense_for_summary, 1, 4));
    try std.testing.expect(!(try lhs.allcloseDense(rhs_dense_for_summary, 1e-12, 1e-12)));
    var dense_close_mask = try lhs.iscloseDense(rhs_dense_for_summary, 1, 4);
    defer dense_close_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, true, true }, dense_close_mask.data);
    var dense_strict_close_mask = try lhs.isCloseDense(rhs_dense_for_summary, 1e-12, 1e-12);
    defer dense_strict_close_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, false, false }, dense_strict_close_mask.data);
    var mismatched_dense_for_summary = try array_mod.Array(f64).zeros(gpa, &.{ 3, 3 });
    defer mismatched_dense_for_summary.deinit();
    try std.testing.expectError(error.ShapeMismatch, lhs.diffSummaryDense(mismatched_dense_for_summary));
    try std.testing.expectError(error.ShapeMismatch, lhs.eqDense(mismatched_dense_for_summary));
    try std.testing.expectError(error.ShapeMismatch, lhs.whereArray(where_mask, mismatched_dense_for_summary));
    var non_matrix_dense_for_summary = try array_mod.Array(f64).zeros(gpa, &.{6});
    defer non_matrix_dense_for_summary.deinit();
    try std.testing.expectError(error.NonMatrixArray, lhs.lessEqualDense(non_matrix_dense_for_summary));
    try std.testing.expectError(error.NonMatrixArray, lhs.whereArray(where_mask, non_matrix_dense_for_summary));
    var wrong_where_mask = try array_mod.Array(bool).zeros(gpa, &.{ 3, 3 });
    defer wrong_where_mask.deinit();
    try std.testing.expectError(error.ShapeMismatch, lhs.whereScalar(wrong_where_mask, 0));
    try std.testing.expectError(error.ShapeMismatch, lhs.maskedSelect(wrong_where_mask));
    try std.testing.expectError(error.ShapeMismatch, lhs.maskedFill(wrong_where_mask, 0));
    var coo_product = try lhs.hadamard(rhs);
    defer coo_product.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 1 }, coo_product.row_indices);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2 }, coo_product.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 4, -4, 18 }, coo_product.values);
    var dot_rhs = try cooFromSlices(f64, gpa, 2, 3, &.{ 1, 0, 1 }, &.{ 2, 0, 1 }, &.{ 5, 4, -2 });
    defer dot_rhs.deinit();
    try std.testing.expect(lhs.sameStructure(dot_rhs));
    try std.testing.expectApproxEqAbs(@as(f64, 15), try lhs.dotSameStructure(dot_rhs), 1e-12);
    var coo_equal_same = try lhs.equalSameStructure(dot_rhs);
    defer coo_equal_same.deinit();
    try expectCooBool(coo_equal_same, lhs.row_indices, lhs.col_indices, &.{ false, false, false });
    var coo_ne_same = try lhs.neSameStructure(dot_rhs);
    defer coo_ne_same.deinit();
    try expectCooBool(coo_ne_same, lhs.row_indices, lhs.col_indices, &.{ true, true, true });
    var coo_greater_same = try lhs.greaterSameStructure(dot_rhs);
    defer coo_greater_same.deinit();
    try expectCooBool(coo_greater_same, lhs.row_indices, lhs.col_indices, &.{ false, false, true });
    var coo_ge_same = try lhs.geSameStructure(dot_rhs);
    defer coo_ge_same.deinit();
    try expectCooBool(coo_ge_same, lhs.row_indices, lhs.col_indices, &.{ false, false, true });
    var coo_less_same = try lhs.lessSameStructure(dot_rhs);
    defer coo_less_same.deinit();
    try expectCooBool(coo_less_same, lhs.row_indices, lhs.col_indices, &.{ true, true, false });
    var coo_le_same = try lhs.leSameStructure(dot_rhs);
    defer coo_le_same.deinit();
    try expectCooBool(coo_le_same, lhs.row_indices, lhs.col_indices, &.{ true, true, false });
    var bool_lhs = try cooFromSlices(bool, gpa, 1, 2, &.{ 0, 0 }, &.{ 0, 1 }, &.{ true, false });
    defer bool_lhs.deinit();
    var bool_rhs = try cooFromSlices(bool, gpa, 1, 2, &.{ 0, 0 }, &.{ 0, 1 }, &.{ true, true });
    defer bool_rhs.deinit();
    var bool_eq_same = try bool_lhs.eqSameStructure(bool_rhs);
    defer bool_eq_same.deinit();
    try expectCooBool(bool_eq_same, bool_lhs.row_indices, bool_lhs.col_indices, &.{ true, false });
    var bool_not_equal_same = try bool_lhs.notEqualSameStructure(bool_rhs);
    defer bool_not_equal_same.deinit();
    try expectCooBool(bool_not_equal_same, bool_lhs.row_indices, bool_lhs.col_indices, &.{ false, true });
    const coo_summary = try lhs.sameStructureDiffSummary(dot_rhs);
    try std.testing.expectApproxEqAbs(@as(f64, 15), coo_summary.dot, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4), coo_summary.max_abs_diff, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2), coo_summary.max_rel_diff, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 29), coo_summary.squared_distance, 1e-12);
    try std.testing.expectApproxEqAbs(@sqrt(@as(f64, 14)), coo_summary.lhs_frobenius_norm, 1e-12);
    try std.testing.expectApproxEqAbs(@sqrt(@as(f64, 45)), coo_summary.rhs_frobenius_norm, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4), try lhs.maxAbsDiffSameStructure(dot_rhs), 1e-12);
    try std.testing.expect(try lhs.maxAbsDiffSameStructureMeetsBound(dot_rhs, 4));
    try std.testing.expect(!(try lhs.maxAbsDiffSameStructureMeetsBound(dot_rhs, 3.999)));
    try std.testing.expectError(error.InvalidShape, lhs.maxAbsDiffSameStructureMeetsBound(dot_rhs, std.math.nan(f64)));
    try std.testing.expectApproxEqAbs(@as(f64, 2), try lhs.maxRelDiffSameStructure(dot_rhs), 1e-12);
    try std.testing.expect(try lhs.maxRelDiffSameStructureMeetsBound(dot_rhs, 2));
    try std.testing.expect(!(try lhs.maxRelDiffSameStructureMeetsBound(dot_rhs, 1.999)));
    try std.testing.expectError(error.InvalidShape, lhs.maxRelDiffSameStructureMeetsBound(dot_rhs, std.math.inf(f64)));
    try std.testing.expectApproxEqAbs(@as(f64, 29), try lhs.squaredDistanceSameStructure(dot_rhs), 1e-12);
    try std.testing.expect(try lhs.squaredDistanceSameStructureMeetsBound(dot_rhs, 29));
    try std.testing.expect(!(try lhs.squaredDistanceSameStructureMeetsBound(dot_rhs, 28.999)));
    try std.testing.expectApproxEqAbs(@sqrt(@as(f64, 29)), try lhs.frobeniusDistanceSameStructure(dot_rhs), 1e-12);
    try std.testing.expect(try lhs.frobeniusDistanceSameStructureMeetsBound(dot_rhs, @sqrt(@as(f64, 29))));
    try std.testing.expect(!(try lhs.frobeniusDistanceSameStructureMeetsBound(dot_rhs, @sqrt(@as(f64, 29)) - 1e-12)));
    const relative_distance = @sqrt(@as(f64, 29)) / (@sqrt(@as(f64, 14)) + @sqrt(@as(f64, 45)));
    try std.testing.expectApproxEqAbs(@sqrt(@as(f64, 29)), coo_summary.frobeniusDistance(), 1e-12);
    try std.testing.expectApproxEqAbs(relative_distance, coo_summary.relativeFrobeniusDistance(), 1e-12);
    try std.testing.expect(try coo_summary.meetsBounds(4, 2, 29, @sqrt(@as(f64, 29)), relative_distance));
    try std.testing.expect(!(try coo_summary.meetsBounds(3.999, 2, 29, @sqrt(@as(f64, 29)), relative_distance)));
    try std.testing.expectError(error.InvalidShape, coo_summary.meetsBounds(std.math.nan(f64), 2, 29, @sqrt(@as(f64, 29)), relative_distance));
    try std.testing.expectApproxEqAbs(relative_distance, try lhs.relativeFrobeniusDistanceSameStructure(dot_rhs), 1e-12);
    try std.testing.expect(try lhs.relativeFrobeniusDistanceSameStructureMeetsBound(dot_rhs, relative_distance));
    try std.testing.expect(!(try lhs.relativeFrobeniusDistanceSameStructureMeetsBound(dot_rhs, relative_distance - 1e-12)));
    try std.testing.expect(try lhs.allclose(lhs, 1e-12, 1e-12));
    try std.testing.expect(try lhs.allClose(lhs, 1e-12, 1e-12));
    try std.testing.expect(!(try lhs.allclose(rhs, 1e-12, 1e-12)));
    try std.testing.expect(try lhs.allclose(rhs, 1, 4));
    var close_mask = try lhs.isclose(rhs, 1, 4);
    defer close_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, true, true }, close_mask.data);
    var strict_close_mask = try lhs.isClose(rhs, 1e-12, 1e-12);
    defer strict_close_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, false, false }, strict_close_mask.data);

    var nan_lhs = try cooFromSlices(f64, gpa, 2, 2, &.{ 0, 1 }, &.{ 0, 1 }, &.{ std.math.nan(f64), 2 });
    defer nan_lhs.deinit();
    var nan_rhs = try cooFromSlices(f64, gpa, 2, 2, &.{ 0, 1 }, &.{ 0, 1 }, &.{ std.math.nan(f64), 2 });
    defer nan_rhs.deinit();
    try std.testing.expect(!(try nan_lhs.allclose(nan_rhs, 0, 0)));
    try std.testing.expect(try nan_lhs.allcloseEqualNan(nan_rhs, 0, 0, true));
    try std.testing.expect(try nan_lhs.allCloseEqualNan(nan_rhs, 0, 0, true));
    var nan_close_mask = try nan_lhs.iscloseEqualNan(nan_rhs, 0, 0, true);
    defer nan_close_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, nan_close_mask.data);
    var nan_scalar_close_mask = try nan_lhs.iscloseScalarEqualNan(std.math.nan(f64), 0, 0, true);
    defer nan_scalar_close_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false }, nan_scalar_close_mask.data);
    try std.testing.expect(!(try nan_lhs.allcloseScalarEqualNan(std.math.nan(f64), 0, 0, true)));
    var nan_rhs_dense = try nan_rhs.toDense();
    defer nan_rhs_dense.deinit();
    try std.testing.expect(!(try nan_lhs.allcloseDense(nan_rhs_dense, 0, 0)));
    try std.testing.expect(try nan_lhs.allcloseDenseEqualNan(nan_rhs_dense, 0, 0, true));
    try std.testing.expect(try nan_lhs.allCloseDenseEqualNan(nan_rhs_dense, 0, 0, true));
    var nan_dense_close_mask = try nan_lhs.iscloseDenseEqualNan(nan_rhs_dense, 0, 0, true);
    defer nan_dense_close_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, nan_dense_close_mask.data);
    var nan_dense_alias_mask = try nan_lhs.isCloseDenseEqualNan(nan_rhs_dense, 0, 0, true);
    defer nan_dense_alias_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, nan_dense_alias_mask.data);
    var nan_lhs_csr = try nan_lhs.toCsr();
    defer nan_lhs_csr.deinit();
    var nan_rhs_csr = try nan_rhs.toCsr();
    defer nan_rhs_csr.deinit();
    try std.testing.expect(!(try nan_lhs_csr.allclose(nan_rhs_csr, 0, 0)));
    try std.testing.expect(try nan_lhs_csr.allcloseEqualNan(nan_rhs_csr, 0, 0, true));
    var nan_lhs_csc = try nan_lhs.toCsc();
    defer nan_lhs_csc.deinit();
    var nan_rhs_csc = try nan_rhs.toCsc();
    defer nan_rhs_csc.deinit();
    try std.testing.expect(!(try nan_lhs_csc.allclose(nan_rhs_csc, 0, 0)));
    try std.testing.expect(try nan_lhs_csc.allCloseEqualNan(nan_rhs_csc, 0, 0, true));

    var different_structure = try cooFromSlices(f64, gpa, 2, 3, &.{ 0, 1, 1 }, &.{ 0, 1, 2 }, &.{ 4, 5, 6 });
    defer different_structure.deinit();
    try std.testing.expect(!lhs.sameStructure(different_structure));
    try std.testing.expectError(error.InvalidShape, lhs.dotSameStructure(different_structure));
    try std.testing.expectError(error.InvalidShape, lhs.sameStructureDiffSummary(different_structure));
    try std.testing.expectError(error.InvalidShape, lhs.maxAbsDiffSameStructure(different_structure));
    try std.testing.expectError(error.InvalidShape, lhs.maxRelDiffSameStructure(different_structure));
    try std.testing.expectError(error.InvalidShape, lhs.squaredDistanceSameStructure(different_structure));
    try std.testing.expectError(error.InvalidShape, lhs.eqSameStructure(different_structure));
    var different_shape = try cooFromSlices(f64, gpa, 3, 3, &.{ 0, 1, 1 }, &.{ 0, 1, 2 }, &.{ 4, 5, 6 });
    defer different_shape.deinit();
    try std.testing.expectError(error.ShapeMismatch, lhs.dotSameStructure(different_shape));
    try std.testing.expectError(error.ShapeMismatch, lhs.sameStructureDiffSummary(different_shape));
    try std.testing.expectError(error.ShapeMismatch, lhs.maxAbsDiffSameStructure(different_shape));
    try std.testing.expectError(error.ShapeMismatch, lhs.maxRelDiffSameStructure(different_shape));
    try std.testing.expectError(error.ShapeMismatch, lhs.frobeniusDistanceSameStructure(different_shape));
    try std.testing.expectError(error.ShapeMismatch, lhs.greaterSameStructure(different_shape));

    var lhs_csr = try lhs.toCsr();
    defer lhs_csr.deinit();
    var rhs_csr = try rhs.toCsr();
    defer rhs_csr.deinit();
    var dot_rhs_csr = try dot_rhs.toCsr();
    defer dot_rhs_csr.deinit();
    try expectDenseComparisons(@TypeOf(lhs_csr), lhs_csr, rhs_csr, rhs_dense_for_summary);
    try expectScalarCloseness(@TypeOf(lhs_csr), lhs_csr);
    try expectNonzero(@TypeOf(lhs_csr), lhs_csr);
    try expectWhere(@TypeOf(lhs_csr), lhs_csr, rhs_csr, rhs_dense_for_summary, where_mask);
    try std.testing.expect(lhs_csr.sameStructure(dot_rhs_csr));
    try std.testing.expectApproxEqAbs(@as(f64, 15), try lhs_csr.dotSameStructure(dot_rhs_csr), 1e-12);
    var csr_eq_same = try lhs_csr.eqSameStructure(dot_rhs_csr);
    defer csr_eq_same.deinit();
    try expectCsrBool(csr_eq_same, lhs_csr.row_offsets, lhs_csr.col_indices, &.{ false, false, false });
    var csr_not_equal_same = try lhs_csr.notEqualSameStructure(dot_rhs_csr);
    defer csr_not_equal_same.deinit();
    try expectCsrBool(csr_not_equal_same, lhs_csr.row_offsets, lhs_csr.col_indices, &.{ true, true, true });
    var csr_gt_same = try lhs_csr.gtSameStructure(dot_rhs_csr);
    defer csr_gt_same.deinit();
    try expectCsrBool(csr_gt_same, lhs_csr.row_offsets, lhs_csr.col_indices, &.{ false, false, true });
    var csr_greater_equal_same = try lhs_csr.greaterEqualSameStructure(dot_rhs_csr);
    defer csr_greater_equal_same.deinit();
    try expectCsrBool(csr_greater_equal_same, lhs_csr.row_offsets, lhs_csr.col_indices, &.{ false, false, true });
    var csr_lt_same = try lhs_csr.ltSameStructure(dot_rhs_csr);
    defer csr_lt_same.deinit();
    try expectCsrBool(csr_lt_same, lhs_csr.row_offsets, lhs_csr.col_indices, &.{ true, true, false });
    var csr_less_equal_same = try lhs_csr.lessEqualSameStructure(dot_rhs_csr);
    defer csr_less_equal_same.deinit();
    try expectCsrBool(csr_less_equal_same, lhs_csr.row_offsets, lhs_csr.col_indices, &.{ true, true, false });
    const csr_summary = try lhs_csr.sameStructureDiffSummary(dot_rhs_csr);
    try std.testing.expectApproxEqAbs(coo_summary.dot, csr_summary.dot, 1e-12);
    try std.testing.expectApproxEqAbs(coo_summary.squared_distance, csr_summary.squared_distance, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4), try lhs_csr.maxAbsDiffSameStructure(dot_rhs_csr), 1e-12);
    try std.testing.expect(try lhs_csr.maxAbsDiffSameStructureMeetsBound(dot_rhs_csr, 4));
    try std.testing.expect(!(try lhs_csr.maxAbsDiffSameStructureMeetsBound(dot_rhs_csr, 3.999)));
    try std.testing.expectApproxEqAbs(@as(f64, 2), try lhs_csr.maxRelDiffSameStructure(dot_rhs_csr), 1e-12);
    try std.testing.expect(try lhs_csr.maxRelDiffSameStructureMeetsBound(dot_rhs_csr, 2));
    try std.testing.expect(!(try lhs_csr.maxRelDiffSameStructureMeetsBound(dot_rhs_csr, 1.999)));
    try std.testing.expectApproxEqAbs(@as(f64, 29), try lhs_csr.squaredDistanceSameStructure(dot_rhs_csr), 1e-12);
    try std.testing.expect(try lhs_csr.squaredDistanceSameStructureMeetsBound(dot_rhs_csr, 29));
    try std.testing.expectApproxEqAbs(@sqrt(@as(f64, 29)), try lhs_csr.frobeniusDistanceSameStructure(dot_rhs_csr), 1e-12);
    try std.testing.expect(try lhs_csr.frobeniusDistanceSameStructureMeetsBound(dot_rhs_csr, @sqrt(@as(f64, 29))));
    try std.testing.expectApproxEqAbs(relative_distance, try lhs_csr.relativeFrobeniusDistanceSameStructure(dot_rhs_csr), 1e-12);
    try std.testing.expect(try lhs_csr.relativeFrobeniusDistanceSameStructureMeetsBound(dot_rhs_csr, relative_distance));
    try std.testing.expect(try lhs_csr.allclose(lhs_csr, 1e-12, 1e-12));
    try std.testing.expect(!(try lhs_csr.allclose(rhs_csr, 1e-12, 1e-12)));
    try std.testing.expect(try lhs_csr.allClose(rhs_csr, 1, 4));
    var csr_close_mask = try lhs_csr.isclose(rhs_csr, 1, 4);
    defer csr_close_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, true, true }, csr_close_mask.data);
    var different_shape_csr = try different_shape.toCsr();
    defer different_shape_csr.deinit();
    try std.testing.expectError(error.ShapeMismatch, lhs_csr.allclose(different_shape_csr, 1e-12, 1e-12));
    try std.testing.expectError(error.ShapeMismatch, lhs_csr.eqSameStructure(different_shape_csr));
    var different_structure_csr = try different_structure.toCsr();
    defer different_structure_csr.deinit();
    try std.testing.expectError(error.InvalidShape, lhs_csr.leSameStructure(different_structure_csr));
    var csr_sum = try lhs_csr.add(rhs_csr);
    defer csr_sum.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 3 }, csr_sum.row_offsets);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2 }, csr_sum.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 5, 0, 9 }, csr_sum.values);
    var csr_pruned = try csr_sum.dropZeros();
    defer csr_pruned.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2 }, csr_pruned.row_offsets);
    try std.testing.expectEqualSlices(usize, &.{ 0, 2 }, csr_pruned.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 5, 9 }, csr_pruned.values);
    try std.testing.expectEqual(@as(usize, 2), csr_sum.dropZerosNnz());
    var csr_tolerance_pruned = try csr_sum.pruneZeros(5);
    defer csr_tolerance_pruned.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 0, 1 }, csr_tolerance_pruned.row_offsets);
    try std.testing.expectEqualSlices(usize, &.{2}, csr_tolerance_pruned.col_indices);
    try std.testing.expectEqualSlices(f64, &.{9}, csr_tolerance_pruned.values);
    try std.testing.expectEqual(@as(usize, 1), try csr_sum.pruneZerosNnz(5));
    try std.testing.expectError(error.InvalidShape, csr_sum.pruneZeros(std.math.inf(f64)));
    try std.testing.expectError(error.InvalidShape, csr_sum.pruneZerosNnz(std.math.inf(f64)));
    var csr_scaled = try csr_pruned.scale(3);
    defer csr_scaled.deinit();
    try std.testing.expectEqualSlices(usize, csr_pruned.row_offsets, csr_scaled.row_offsets);
    try std.testing.expectEqualSlices(usize, csr_pruned.col_indices, csr_scaled.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 15, 27 }, csr_scaled.values);
    var csr_rc_scaled = try csr_pruned.scaleRowsAndColumns(&.{ 2, 3 }, &.{ 4, 5, 6 });
    defer csr_rc_scaled.deinit();
    try std.testing.expectEqualSlices(usize, csr_pruned.row_offsets, csr_rc_scaled.row_offsets);
    try std.testing.expectEqualSlices(usize, csr_pruned.col_indices, csr_rc_scaled.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 40, 162 }, csr_rc_scaled.values);
    var csr_rc_scaled_inplace = try csr_pruned.clone();
    defer csr_rc_scaled_inplace.deinit();
    try csr_rc_scaled_inplace.scaleRowsAndColumnsInPlace(&.{ 2, 3 }, &.{ 4, 5, 6 });
    try std.testing.expectEqualSlices(usize, csr_pruned.row_offsets, csr_rc_scaled_inplace.row_offsets);
    try std.testing.expectEqualSlices(usize, csr_pruned.col_indices, csr_rc_scaled_inplace.col_indices);
    try std.testing.expectEqualSlices(f64, csr_rc_scaled.values, csr_rc_scaled_inplace.values);
    var csr_row_scaled_inplace = try csr_pruned.clone();
    defer csr_row_scaled_inplace.deinit();
    try csr_row_scaled_inplace.scaleRowsInPlace(&.{ 2, 3 });
    try std.testing.expectEqualSlices(f64, &.{ 10, 27 }, csr_row_scaled_inplace.values);
    var csr_col_scaled_inplace = try csr_pruned.clone();
    defer csr_col_scaled_inplace.deinit();
    try csr_col_scaled_inplace.scaleColumnsInPlace(&.{ 4, 5, 6 });
    try std.testing.expectEqualSlices(f64, &.{ 20, 54 }, csr_col_scaled_inplace.values);
    try std.testing.expectError(error.ShapeMismatch, csr_col_scaled_inplace.scaleColumnsInPlace(&.{1}));
    var csr_diff = try lhs_csr.sub(rhs_csr);
    defer csr_diff.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 3 }, csr_diff.row_offsets);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2 }, csr_diff.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ -3, 4, -3 }, csr_diff.values);
    var csr_abs = try csr_diff.absolute();
    defer csr_abs.deinit();
    try std.testing.expectEqualSlices(usize, csr_diff.row_offsets, csr_abs.row_offsets);
    try std.testing.expectEqualSlices(usize, csr_diff.col_indices, csr_abs.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 3, 4, 3 }, csr_abs.values);
    var csr_squared = try csr_diff.square();
    defer csr_squared.deinit();
    try std.testing.expectEqualSlices(usize, csr_diff.row_offsets, csr_squared.row_offsets);
    try std.testing.expectEqualSlices(usize, csr_diff.col_indices, csr_squared.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 9, 16, 9 }, csr_squared.values);
    var csr_sqrt = try csr_squared.sqrt();
    defer csr_sqrt.deinit();
    try std.testing.expectEqualSlices(usize, csr_squared.row_offsets, csr_sqrt.row_offsets);
    try std.testing.expectEqualSlices(usize, csr_squared.col_indices, csr_sqrt.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 3, 4, 3 }, csr_sqrt.values);
    var csr_rsqrt = try csr_squared.rsqrt();
    defer csr_rsqrt.deinit();
    try std.testing.expectEqualSlices(usize, csr_squared.row_offsets, csr_rsqrt.row_offsets);
    try std.testing.expectEqualSlices(usize, csr_squared.col_indices, csr_rsqrt.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 1.0 / 3.0, 0.25, 1.0 / 3.0 }, csr_rsqrt.values);
    var csr_reciprocal = try csr_squared.reciprocal();
    defer csr_reciprocal.deinit();
    try std.testing.expectEqualSlices(usize, csr_squared.row_offsets, csr_reciprocal.row_offsets);
    try std.testing.expectEqualSlices(usize, csr_squared.col_indices, csr_reciprocal.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 1.0 / 9.0, 1.0 / 16.0, 1.0 / 9.0 }, csr_reciprocal.values);
    var csr_cubes = try coo_cubes.toCsr();
    defer csr_cubes.deinit();
    var csr_cbrt = try csr_cubes.cbrt();
    defer csr_cbrt.deinit();
    try std.testing.expectEqualSlices(usize, csr_cubes.row_offsets, csr_cbrt.row_offsets);
    try std.testing.expectEqualSlices(usize, csr_cubes.col_indices, csr_cbrt.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ -2, 3, 4 }, csr_cbrt.values);
    var csr_sign = try csr_diff.sign();
    defer csr_sign.deinit();
    try std.testing.expectEqualSlices(usize, csr_diff.row_offsets, csr_sign.row_offsets);
    try std.testing.expectEqualSlices(usize, csr_diff.col_indices, csr_sign.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ -1, 1, -1 }, csr_sign.values);
    var csr_signbit = try csr_diff.signbit();
    defer csr_signbit.deinit();
    try std.testing.expectEqualSlices(usize, csr_diff.row_offsets, csr_signbit.row_offsets);
    try std.testing.expectEqualSlices(usize, csr_diff.col_indices, csr_signbit.col_indices);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, csr_signbit.values);
    const csr_full_summary = try lhs_csr.diffSummary(rhs_csr);
    try std.testing.expectApproxEqAbs(full_summary.dot, csr_full_summary.dot, 1e-12);
    try std.testing.expectApproxEqAbs(full_summary.squared_distance, csr_full_summary.squared_distance, 1e-12);
    try std.testing.expect(try lhs_csr.diffSummaryMeetsBounds(rhs_csr, 4, 2, 34, @sqrt(@as(f64, 34)), full_relative));
    try std.testing.expectApproxEqAbs(@as(f64, 4), try lhs_csr.maxAbsDiff(rhs_csr), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2), try lhs_csr.maxRelDiff(rhs_csr), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 34), try lhs_csr.squaredDistance(rhs_csr), 1e-12);
    try std.testing.expectApproxEqAbs(@sqrt(@as(f64, 34)), try lhs_csr.frobeniusDistance(rhs_csr), 1e-12);
    try std.testing.expectApproxEqAbs(full_relative, try lhs_csr.relativeFrobeniusDistance(rhs_csr), 1e-12);
    try std.testing.expect(try lhs_csr.maxAbsDiffMeetsBound(rhs_csr, 4));
    try std.testing.expect(!(try lhs_csr.maxAbsDiffMeetsBound(rhs_csr, 3.999)));
    const csr_dense_summary = try lhs_csr.diffSummaryDense(rhs_dense_for_summary);
    try std.testing.expectApproxEqAbs(csr_full_summary.dot, csr_dense_summary.dot, 1e-12);
    try std.testing.expectApproxEqAbs(csr_full_summary.squared_distance, csr_dense_summary.squared_distance, 1e-12);
    try std.testing.expect(try lhs_csr.diffSummaryDenseMeetsBounds(rhs_dense_for_summary, 4, 2, 34, @sqrt(@as(f64, 34)), full_relative));
    try std.testing.expectApproxEqAbs(@as(f64, 4), try lhs_csr.maxAbsDiffDense(rhs_dense_for_summary), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2), try lhs_csr.maxRelDiffDense(rhs_dense_for_summary), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 34), try lhs_csr.squaredDistanceDense(rhs_dense_for_summary), 1e-12);
    try std.testing.expectApproxEqAbs(@sqrt(@as(f64, 34)), try lhs_csr.frobeniusDistanceDense(rhs_dense_for_summary), 1e-12);
    try std.testing.expectApproxEqAbs(full_relative, try lhs_csr.relativeFrobeniusDistanceDense(rhs_dense_for_summary), 1e-12);
    try std.testing.expect(try lhs_csr.maxAbsDiffDenseMeetsBound(rhs_dense_for_summary, 4));
    try std.testing.expect(try lhs_csr.allcloseDense(rhs_dense_for_summary, 1, 4));
    try std.testing.expect(!(try lhs_csr.allcloseDense(rhs_dense_for_summary, 1e-12, 1e-12)));
    var csr_dense_close_mask = try lhs_csr.iscloseDense(rhs_dense_for_summary, 1, 4);
    defer csr_dense_close_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, true, true }, csr_dense_close_mask.data);
    var csr_product = try lhs_csr.multiply(rhs_csr);
    defer csr_product.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 3 }, csr_product.row_offsets);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2 }, csr_product.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 4, -4, 18 }, csr_product.values);

    var lhs_csc = try lhs.toCsc();
    defer lhs_csc.deinit();
    var rhs_csc = try rhs.toCsc();
    defer rhs_csc.deinit();
    var dot_rhs_csc = try dot_rhs.toCsc();
    defer dot_rhs_csc.deinit();
    try expectDenseComparisons(@TypeOf(lhs_csc), lhs_csc, rhs_csc, rhs_dense_for_summary);
    try expectScalarCloseness(@TypeOf(lhs_csc), lhs_csc);
    try expectNonzero(@TypeOf(lhs_csc), lhs_csc);
    try expectWhere(@TypeOf(lhs_csc), lhs_csc, rhs_csc, rhs_dense_for_summary, where_mask);
    var csc_sum = try lhs_csc.add(rhs_csc);
    defer csc_sum.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2, 3 }, csc_sum.col_offsets);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 1 }, csc_sum.row_indices);
    try std.testing.expectEqualSlices(f64, &.{ 5, 0, 9 }, csc_sum.values);
    var csc_pruned = try csc_sum.dropZeros();
    defer csc_pruned.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 1, 2 }, csc_pruned.col_offsets);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, csc_pruned.row_indices);
    try std.testing.expectEqualSlices(f64, &.{ 5, 9 }, csc_pruned.values);
    try std.testing.expectEqual(@as(usize, 2), csc_sum.dropZerosNnz());
    var csc_tolerance_pruned = try csc_sum.pruneZeros(5);
    defer csc_tolerance_pruned.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 0, 0, 1 }, csc_tolerance_pruned.col_offsets);
    try std.testing.expectEqualSlices(usize, &.{1}, csc_tolerance_pruned.row_indices);
    try std.testing.expectEqualSlices(f64, &.{9}, csc_tolerance_pruned.values);
    try std.testing.expectEqual(@as(usize, 1), try csc_sum.pruneZerosNnz(5));
    try std.testing.expectError(error.InvalidShape, csc_sum.pruneZeros(std.math.nan(f64)));
    try std.testing.expectError(error.InvalidShape, csc_sum.pruneZerosNnz(std.math.nan(f64)));
    var csc_scaled_zero = try csc_pruned.scale(0);
    defer csc_scaled_zero.deinit();
    try std.testing.expectEqualSlices(usize, csc_pruned.col_offsets, csc_scaled_zero.col_offsets);
    try std.testing.expectEqualSlices(f64, &.{ 0, 0 }, csc_scaled_zero.values);
    var csc_scaled_zero_pruned = try csc_scaled_zero.dropZeros();
    defer csc_scaled_zero_pruned.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 0, 0, 0 }, csc_scaled_zero_pruned.col_offsets);
    try std.testing.expectEqual(@as(usize, 0), csc_scaled_zero_pruned.nnz());
    var csc_row_scaled = try csc_pruned.scaleRows(&.{ 2, 3 });
    defer csc_row_scaled.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 10, 27 }, csc_row_scaled.values);
    var csc_row_scaled_inplace = try csc_pruned.clone();
    defer csc_row_scaled_inplace.deinit();
    try csc_row_scaled_inplace.scaleRowsInPlace(&.{ 2, 3 });
    try std.testing.expectEqualSlices(usize, csc_pruned.col_offsets, csc_row_scaled_inplace.col_offsets);
    try std.testing.expectEqualSlices(usize, csc_pruned.row_indices, csc_row_scaled_inplace.row_indices);
    try std.testing.expectEqualSlices(f64, csc_row_scaled.values, csc_row_scaled_inplace.values);
    var csc_col_scaled_inplace = try csc_pruned.clone();
    defer csc_col_scaled_inplace.deinit();
    try csc_col_scaled_inplace.scaleColumnsInPlace(&.{ 4, 5, 6 });
    try std.testing.expectEqualSlices(f64, &.{ 20, 54 }, csc_col_scaled_inplace.values);
    var csc_rc_scaled_inplace = try csc_pruned.clone();
    defer csc_rc_scaled_inplace.deinit();
    try csc_rc_scaled_inplace.scaleRowsAndColumnsInPlace(&.{ 2, 3 }, &.{ 4, 5, 6 });
    try std.testing.expectEqualSlices(f64, &.{ 40, 162 }, csc_rc_scaled_inplace.values);
    try std.testing.expectError(error.ShapeMismatch, csc_pruned.scaleColumns(&.{1}));
    try std.testing.expectError(error.ShapeMismatch, csc_rc_scaled_inplace.scaleRowsAndColumnsInPlace(&.{2}, &.{ 4, 5, 6 }));
    var csc_diff = try lhs_csc.sub(rhs_csc);
    defer csc_diff.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2, 3 }, csc_diff.col_offsets);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 1 }, csc_diff.row_indices);
    try std.testing.expectEqualSlices(f64, &.{ -3, 4, -3 }, csc_diff.values);
    var csc_abs = try csc_diff.fabs();
    defer csc_abs.deinit();
    try std.testing.expectEqualSlices(usize, csc_diff.col_offsets, csc_abs.col_offsets);
    try std.testing.expectEqualSlices(usize, csc_diff.row_indices, csc_abs.row_indices);
    try std.testing.expectEqualSlices(f64, &.{ 3, 4, 3 }, csc_abs.values);
    var csc_squared = try csc_diff.square();
    defer csc_squared.deinit();
    try std.testing.expectEqualSlices(usize, csc_diff.col_offsets, csc_squared.col_offsets);
    try std.testing.expectEqualSlices(usize, csc_diff.row_indices, csc_squared.row_indices);
    try std.testing.expectEqualSlices(f64, &.{ 9, 16, 9 }, csc_squared.values);
    var csc_sqrt = try csc_squared.sqrt();
    defer csc_sqrt.deinit();
    try std.testing.expectEqualSlices(usize, csc_squared.col_offsets, csc_sqrt.col_offsets);
    try std.testing.expectEqualSlices(usize, csc_squared.row_indices, csc_sqrt.row_indices);
    try std.testing.expectEqualSlices(f64, &.{ 3, 4, 3 }, csc_sqrt.values);
    var csc_rsqrt = try csc_squared.rsqrt();
    defer csc_rsqrt.deinit();
    try std.testing.expectEqualSlices(usize, csc_squared.col_offsets, csc_rsqrt.col_offsets);
    try std.testing.expectEqualSlices(usize, csc_squared.row_indices, csc_rsqrt.row_indices);
    try std.testing.expectEqualSlices(f64, &.{ 1.0 / 3.0, 0.25, 1.0 / 3.0 }, csc_rsqrt.values);
    var csc_reciprocal = try csc_squared.reciprocal();
    defer csc_reciprocal.deinit();
    try std.testing.expectEqualSlices(usize, csc_squared.col_offsets, csc_reciprocal.col_offsets);
    try std.testing.expectEqualSlices(usize, csc_squared.row_indices, csc_reciprocal.row_indices);
    try std.testing.expectEqualSlices(f64, &.{ 1.0 / 9.0, 1.0 / 16.0, 1.0 / 9.0 }, csc_reciprocal.values);
    var csc_cubes = try coo_cubes.toCsc();
    defer csc_cubes.deinit();
    var csc_cbrt = try csc_cubes.cbrt();
    defer csc_cbrt.deinit();
    try std.testing.expectEqualSlices(usize, csc_cubes.col_offsets, csc_cbrt.col_offsets);
    try std.testing.expectEqualSlices(usize, csc_cubes.row_indices, csc_cbrt.row_indices);
    try std.testing.expectEqualSlices(f64, &.{ -2, 3, 4 }, csc_cbrt.values);
    var csc_sign = try csc_diff.sign();
    defer csc_sign.deinit();
    try std.testing.expectEqualSlices(usize, csc_diff.col_offsets, csc_sign.col_offsets);
    try std.testing.expectEqualSlices(usize, csc_diff.row_indices, csc_sign.row_indices);
    try std.testing.expectEqualSlices(f64, &.{ -1, 1, -1 }, csc_sign.values);
    var csc_signbit = try csc_diff.signbit();
    defer csc_signbit.deinit();
    try std.testing.expectEqualSlices(usize, csc_diff.col_offsets, csc_signbit.col_offsets);
    try std.testing.expectEqualSlices(usize, csc_diff.row_indices, csc_signbit.row_indices);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, csc_signbit.values);
    const csc_full_summary = try lhs_csc.diffSummary(rhs_csc);
    try std.testing.expectApproxEqAbs(full_summary.dot, csc_full_summary.dot, 1e-12);
    try std.testing.expectApproxEqAbs(full_summary.squared_distance, csc_full_summary.squared_distance, 1e-12);
    try std.testing.expect(try lhs_csc.diffSummaryMeetsBounds(rhs_csc, 4, 2, 34, @sqrt(@as(f64, 34)), full_relative));
    try std.testing.expectApproxEqAbs(@as(f64, 4), try lhs_csc.maxAbsDiff(rhs_csc), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2), try lhs_csc.maxRelDiff(rhs_csc), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 34), try lhs_csc.squaredDistance(rhs_csc), 1e-12);
    try std.testing.expectApproxEqAbs(@sqrt(@as(f64, 34)), try lhs_csc.frobeniusDistance(rhs_csc), 1e-12);
    try std.testing.expectApproxEqAbs(full_relative, try lhs_csc.relativeFrobeniusDistance(rhs_csc), 1e-12);
    try std.testing.expect(try lhs_csc.maxAbsDiffMeetsBound(rhs_csc, 4));
    try std.testing.expect(!(try lhs_csc.maxAbsDiffMeetsBound(rhs_csc, 3.999)));
    const csc_dense_summary = try lhs_csc.diffSummaryDense(rhs_dense_for_summary);
    try std.testing.expectApproxEqAbs(csc_full_summary.dot, csc_dense_summary.dot, 1e-12);
    try std.testing.expectApproxEqAbs(csc_full_summary.squared_distance, csc_dense_summary.squared_distance, 1e-12);
    try std.testing.expect(try lhs_csc.diffSummaryDenseMeetsBounds(rhs_dense_for_summary, 4, 2, 34, @sqrt(@as(f64, 34)), full_relative));
    try std.testing.expectApproxEqAbs(@as(f64, 4), try lhs_csc.maxAbsDiffDense(rhs_dense_for_summary), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2), try lhs_csc.maxRelDiffDense(rhs_dense_for_summary), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 34), try lhs_csc.squaredDistanceDense(rhs_dense_for_summary), 1e-12);
    try std.testing.expectApproxEqAbs(@sqrt(@as(f64, 34)), try lhs_csc.frobeniusDistanceDense(rhs_dense_for_summary), 1e-12);
    try std.testing.expectApproxEqAbs(full_relative, try lhs_csc.relativeFrobeniusDistanceDense(rhs_dense_for_summary), 1e-12);
    try std.testing.expect(try lhs_csc.maxAbsDiffDenseMeetsBound(rhs_dense_for_summary, 4));
    try std.testing.expect(try lhs_csc.allcloseDense(rhs_dense_for_summary, 1, 4));
    try std.testing.expect(!(try lhs_csc.allcloseDense(rhs_dense_for_summary, 1e-12, 1e-12)));
    var csc_dense_close_mask = try lhs_csc.iscloseDense(rhs_dense_for_summary, 1, 4);
    defer csc_dense_close_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, true, true }, csc_dense_close_mask.data);
    var csc_product = try lhs_csc.mul(rhs_csc);
    defer csc_product.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2, 3 }, csc_product.col_offsets);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 1 }, csc_product.row_indices);
    try std.testing.expectEqualSlices(f64, &.{ 4, -4, 18 }, csc_product.values);
    try std.testing.expect(lhs_csc.sameStructure(dot_rhs_csc));
    try std.testing.expectApproxEqAbs(@as(f64, 15), try lhs_csc.dotSameStructure(dot_rhs_csc), 1e-12);
    var csc_equal_same = try lhs_csc.equalSameStructure(dot_rhs_csc);
    defer csc_equal_same.deinit();
    try expectCscBool(csc_equal_same, lhs_csc.col_offsets, lhs_csc.row_indices, &.{ false, false, false });
    var csc_ne_same = try lhs_csc.neSameStructure(dot_rhs_csc);
    defer csc_ne_same.deinit();
    try expectCscBool(csc_ne_same, lhs_csc.col_offsets, lhs_csc.row_indices, &.{ true, true, true });
    var csc_greater_same = try lhs_csc.greaterSameStructure(dot_rhs_csc);
    defer csc_greater_same.deinit();
    try expectCscBool(csc_greater_same, lhs_csc.col_offsets, lhs_csc.row_indices, &.{ false, true, false });
    var csc_ge_same = try lhs_csc.geSameStructure(dot_rhs_csc);
    defer csc_ge_same.deinit();
    try expectCscBool(csc_ge_same, lhs_csc.col_offsets, lhs_csc.row_indices, &.{ false, true, false });
    var csc_less_same = try lhs_csc.lessSameStructure(dot_rhs_csc);
    defer csc_less_same.deinit();
    try expectCscBool(csc_less_same, lhs_csc.col_offsets, lhs_csc.row_indices, &.{ true, false, true });
    var csc_le_same = try lhs_csc.leSameStructure(dot_rhs_csc);
    defer csc_le_same.deinit();
    try expectCscBool(csc_le_same, lhs_csc.col_offsets, lhs_csc.row_indices, &.{ true, false, true });
    const csc_summary = try lhs_csc.sameStructureDiffSummary(dot_rhs_csc);
    try std.testing.expectApproxEqAbs(coo_summary.dot, csc_summary.dot, 1e-12);
    try std.testing.expectApproxEqAbs(coo_summary.squared_distance, csc_summary.squared_distance, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4), try lhs_csc.maxAbsDiffSameStructure(dot_rhs_csc), 1e-12);
    try std.testing.expect(try lhs_csc.maxAbsDiffSameStructureMeetsBound(dot_rhs_csc, 4));
    try std.testing.expect(!(try lhs_csc.maxAbsDiffSameStructureMeetsBound(dot_rhs_csc, 3.999)));
    try std.testing.expectApproxEqAbs(@as(f64, 2), try lhs_csc.maxRelDiffSameStructure(dot_rhs_csc), 1e-12);
    try std.testing.expect(try lhs_csc.maxRelDiffSameStructureMeetsBound(dot_rhs_csc, 2));
    try std.testing.expect(!(try lhs_csc.maxRelDiffSameStructureMeetsBound(dot_rhs_csc, 1.999)));
    try std.testing.expectApproxEqAbs(@as(f64, 29), try lhs_csc.squaredDistanceSameStructure(dot_rhs_csc), 1e-12);
    try std.testing.expect(try lhs_csc.squaredDistanceSameStructureMeetsBound(dot_rhs_csc, 29));
    try std.testing.expectApproxEqAbs(@sqrt(@as(f64, 29)), try lhs_csc.frobeniusDistanceSameStructure(dot_rhs_csc), 1e-12);
    try std.testing.expect(try lhs_csc.frobeniusDistanceSameStructureMeetsBound(dot_rhs_csc, @sqrt(@as(f64, 29))));
    try std.testing.expectApproxEqAbs(relative_distance, try lhs_csc.relativeFrobeniusDistanceSameStructure(dot_rhs_csc), 1e-12);
    try std.testing.expect(try lhs_csc.relativeFrobeniusDistanceSameStructureMeetsBound(dot_rhs_csc, relative_distance));
    try std.testing.expect(try lhs_csc.allclose(lhs_csc, 1e-12, 1e-12));
    try std.testing.expect(!(try lhs_csc.allclose(rhs_csc, 1e-12, 1e-12)));
    try std.testing.expect(try lhs_csc.allClose(rhs_csc, 1, 4));
    var csc_close_mask = try lhs_csc.isClose(rhs_csc, 1, 4);
    defer csc_close_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, true, true }, csc_close_mask.data);
    var different_shape_csc = try different_shape.toCsc();
    defer different_shape_csc.deinit();
    try std.testing.expectError(error.ShapeMismatch, lhs_csc.eqSameStructure(different_shape_csc));
    var different_structure_for_csc = try cooFromSlices(f64, gpa, 2, 3, &.{ 0, 0, 1 }, &.{ 0, 1, 2 }, &.{ 4, 5, 6 });
    defer different_structure_for_csc.deinit();
    var different_structure_csc = try different_structure_for_csc.toCsc();
    defer different_structure_csc.deinit();
    try std.testing.expectError(error.InvalidShape, lhs_csc.gtSameStructure(different_structure_csc));

    var mismatched = try cooFromSlices(f64, gpa, 3, 3, &.{0}, &.{0}, &.{1});
    defer mismatched.deinit();
    try std.testing.expectError(error.ShapeMismatch, lhs.add(mismatched));
    try std.testing.expectError(error.ShapeMismatch, lhs.sub(mismatched));
    try std.testing.expectError(error.ShapeMismatch, lhs.hadamard(mismatched));
    try std.testing.expectError(error.ShapeMismatch, lhs.diffSummary(mismatched));
}

test "sparse stored rounding unary helpers preserve structure" {
    const gpa = std.testing.allocator;
    var coo = try cooFromSlices(f64, gpa, 2, 2, &.{ 0, 0, 1 }, &.{ 0, 1, 1 }, &.{ -1.7, 2.2, 3.8 });
    defer coo.deinit();

    var coo_floor = try coo.floor();
    defer coo_floor.deinit();
    try std.testing.expectEqualSlices(usize, coo.row_indices, coo_floor.row_indices);
    try std.testing.expectEqualSlices(usize, coo.col_indices, coo_floor.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ -2, 2, 3 }, coo_floor.values);
    var coo_ceil = try coo.ceil();
    defer coo_ceil.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -1, 3, 4 }, coo_ceil.values);
    var coo_round = try coo.round();
    defer coo_round.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -2, 2, 4 }, coo_round.values);
    var coo_trunc = try coo.trunc();
    defer coo_trunc.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -1, 2, 3 }, coo_trunc.values);
    var coo_clip_min = try coo.clipMin(0);
    defer coo_clip_min.deinit();
    try std.testing.expectEqualSlices(usize, coo.row_indices, coo_clip_min.row_indices);
    try std.testing.expectEqualSlices(usize, coo.col_indices, coo_clip_min.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 0, 2.2, 3.8 }, coo_clip_min.values);
    var coo_max_scalar = try coo.maximumScalar(0);
    defer coo_max_scalar.deinit();
    try std.testing.expectEqualSlices(f64, coo_clip_min.values, coo_max_scalar.values);
    var coo_clip_max = try coo.clipMax(3);
    defer coo_clip_max.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -1.7, 2.2, 3 }, coo_clip_max.values);
    var coo_min_scalar = try coo.minimumScalar(3);
    defer coo_min_scalar.deinit();
    try std.testing.expectEqualSlices(f64, coo_clip_max.values, coo_min_scalar.values);
    var coo_clip = try coo.clip(-1, 3);
    defer coo_clip.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -1, 2.2, 3 }, coo_clip.values);
    try std.testing.expectError(error.InvalidShape, coo.clip(3, -1));

    var csr = try coo.toCsr();
    defer csr.deinit();
    var csr_floor = try csr.floor();
    defer csr_floor.deinit();
    try std.testing.expectEqualSlices(usize, csr.row_offsets, csr_floor.row_offsets);
    try std.testing.expectEqualSlices(usize, csr.col_indices, csr_floor.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ -2, 2, 3 }, csr_floor.values);
    var csr_ceil = try csr.ceil();
    defer csr_ceil.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -1, 3, 4 }, csr_ceil.values);
    var csr_round = try csr.round();
    defer csr_round.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -2, 2, 4 }, csr_round.values);
    var csr_trunc = try csr.trunc();
    defer csr_trunc.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -1, 2, 3 }, csr_trunc.values);
    var csr_clip_min = try csr.clampMin(0);
    defer csr_clip_min.deinit();
    try std.testing.expectEqualSlices(usize, csr.row_offsets, csr_clip_min.row_offsets);
    try std.testing.expectEqualSlices(usize, csr.col_indices, csr_clip_min.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 0, 2.2, 3.8 }, csr_clip_min.values);
    var csr_max_scalar = try csr.maxScalar(0);
    defer csr_max_scalar.deinit();
    try std.testing.expectEqualSlices(f64, csr_clip_min.values, csr_max_scalar.values);
    var csr_clip_max = try csr.clampMax(3);
    defer csr_clip_max.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -1.7, 2.2, 3 }, csr_clip_max.values);
    var csr_min_scalar = try csr.minScalar(3);
    defer csr_min_scalar.deinit();
    try std.testing.expectEqualSlices(f64, csr_clip_max.values, csr_min_scalar.values);
    var csr_clip = try csr.clamp(-1, 3);
    defer csr_clip.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -1, 2.2, 3 }, csr_clip.values);
    try std.testing.expectError(error.InvalidShape, csr.clamp(3, -1));

    var csc = try coo.toCsc();
    defer csc.deinit();
    var csc_floor = try csc.floor();
    defer csc_floor.deinit();
    try std.testing.expectEqualSlices(usize, csc.col_offsets, csc_floor.col_offsets);
    try std.testing.expectEqualSlices(usize, csc.row_indices, csc_floor.row_indices);
    try std.testing.expectEqualSlices(f64, &.{ -2, 2, 3 }, csc_floor.values);
    var csc_ceil = try csc.ceil();
    defer csc_ceil.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -1, 3, 4 }, csc_ceil.values);
    var csc_round = try csc.round();
    defer csc_round.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -2, 2, 4 }, csc_round.values);
    var csc_trunc = try csc.trunc();
    defer csc_trunc.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -1, 2, 3 }, csc_trunc.values);
    var csc_clip_min = try csc.clipMin(0);
    defer csc_clip_min.deinit();
    try std.testing.expectEqualSlices(usize, csc.col_offsets, csc_clip_min.col_offsets);
    try std.testing.expectEqualSlices(usize, csc.row_indices, csc_clip_min.row_indices);
    try std.testing.expectEqualSlices(f64, &.{ 0, 2.2, 3.8 }, csc_clip_min.values);
    var csc_max_scalar = try csc.maximumScalar(0);
    defer csc_max_scalar.deinit();
    try std.testing.expectEqualSlices(f64, csc_clip_min.values, csc_max_scalar.values);
    var csc_clip_max = try csc.clipMax(3);
    defer csc_clip_max.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -1.7, 2.2, 3 }, csc_clip_max.values);
    var csc_min_scalar = try csc.minimumScalar(3);
    defer csc_min_scalar.deinit();
    try std.testing.expectEqualSlices(f64, csc_clip_max.values, csc_min_scalar.values);
    var csc_clip = try csc.clip(-1, 3);
    defer csc_clip.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -1, 2.2, 3 }, csc_clip.values);
    try std.testing.expectError(error.InvalidShape, csc.clip(3, -1));
}

test "coo sparse diagnostics and duplicate coordinate access" {
    const gpa = std.testing.allocator;
    var symmetric_dense = try array_mod.Array(f64).fromSlice(gpa, &.{
        4, 1, 0,
        1, 5, 2,
        0, 2, 6,
    }, &.{ 3, 3 });
    defer symmetric_dense.deinit();
    var symmetric = try cooFromDense(f64, symmetric_dense);
    defer symmetric.deinit();

    var diagonal = try symmetric.diagonal();
    defer diagonal.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 4, 5, 6 }, diagonal.data);
    var diagonal_mut = try symmetric.clone();
    defer diagonal_mut.deinit();
    try diagonal_mut.addToDiagonal(1);
    var diagonal_mut_values = try diagonal_mut.diagonal();
    defer diagonal_mut_values.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 5, 6, 7 }, diagonal_mut_values.data);
    try diagonal_mut.setDiagonal(9);
    var diagonal_set_values = try diagonal_mut.diagonal();
    defer diagonal_set_values.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 9, 9, 9 }, diagonal_set_values.data);
    try diagonal_mut.setDiagonalValues(&.{ 1, 2, 3 });
    var diagonal_set_vector_values = try diagonal_mut.diagonal();
    defer diagonal_set_vector_values.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 3 }, diagonal_set_vector_values.data);
    try diagonal_mut.addDiagonalValues(&.{ 1, 1, 1 });
    var diagonal_add_vector_values = try diagonal_mut.diagonal();
    defer diagonal_add_vector_values.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 2, 3, 4 }, diagonal_add_vector_values.data);
    try diagonal_mut.multiplyDiagonalValues(&.{ 2, 3, 4 });
    var diagonal_mul_vector_values = try diagonal_mut.diagonal();
    defer diagonal_mul_vector_values.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 4, 9, 16 }, diagonal_mul_vector_values.data);
    try std.testing.expectError(error.ShapeMismatch, diagonal_mut.setDiagonalVector(&.{1}));
    try std.testing.expectApproxEqAbs(@as(f64, 15), try symmetric.trace(), 1e-12);
    try std.testing.expect(try symmetric.traceInRange(15, 15));
    try std.testing.expect(try symmetric.traceInRange(14.5, 15.5));
    try std.testing.expect(!(try symmetric.traceInRange(15.5, 16)));
    try std.testing.expectError(error.InvalidShape, symmetric.traceInRange(std.math.nan(f64), 15));
    try std.testing.expectApproxEqAbs(@as(f64, 5), try symmetric.normalizedTrace(), 1e-12);
    try std.testing.expect(try symmetric.normalizedTraceInRange(5, 5));
    try std.testing.expect(try symmetric.normalizedTraceInRange(4.9, 5.1));
    try std.testing.expect(!(try symmetric.normalizedTraceInRange(5.1, 6)));
    try std.testing.expectError(error.InvalidShape, symmetric.normalizedTraceInRange(6, 5));
    try std.testing.expectEqual(@as(usize, 0), try symmetric.missingDiagonalCount());
    try std.testing.expect(try symmetric.missingDiagonalCountMeetsBound(0));
    try std.testing.expect(try symmetric.missingDiagonalCountInRange(0, 0));
    try std.testing.expectEqual(@as(usize, 0), try symmetric.zeroDiagonalCount());
    try std.testing.expect(try symmetric.zeroDiagonalCountMeetsBound(0));
    try std.testing.expect(try symmetric.zeroDiagonalCountInRange(0, 0));
    try std.testing.expectEqual(@as(usize, 1), try symmetric.bandwidth());
    try std.testing.expect(try symmetric.bandwidthMeetsBound(1));
    try std.testing.expect(!(try symmetric.bandwidthMeetsBound(0)));
    try std.testing.expectEqual(@as(usize, 2), try symmetric.columnIntersectionBandwidth());
    try std.testing.expect(try symmetric.columnIntersectionBandwidthMeetsBound(2));
    try std.testing.expect(!(try symmetric.columnIntersectionBandwidthMeetsBound(1)));
    try std.testing.expectEqual(@as(usize, 5), try symmetric.lowerNnz(false));
    try std.testing.expectEqual(@as(usize, 2), try symmetric.lowerNnz(true));
    try std.testing.expectEqual(@as(usize, 5), try symmetric.upperNnz(false));
    try std.testing.expectEqual(@as(usize, 2), try symmetric.upperNnz(true));
    try std.testing.expect(try symmetric.lowerNnzMeetsBound(false, 5));
    try std.testing.expect(!(try symmetric.lowerNnzMeetsBound(false, 4)));
    try std.testing.expect(try symmetric.upperNnzMeetsBound(true, 2));
    try std.testing.expect(!(try symmetric.upperNnzMeetsBound(true, 1)));
    try std.testing.expect(try symmetric.lowerNnzInRange(false, 5, 5));
    try std.testing.expect(try symmetric.upperNnzInRange(true, 0, 2));
    try std.testing.expect(!(try symmetric.lowerNnzInRange(true, 3, 5)));
    try std.testing.expectError(error.InvalidShape, symmetric.upperNnzInRange(false, 6, 5));
    try std.testing.expectEqual(@as(usize, 2), try symmetric.lowerProfile());
    try std.testing.expectEqual(@as(usize, 2), try symmetric.upperProfile());
    const symmetric_profile = try symmetric.profile();
    try std.testing.expectEqual(@as(usize, 2), symmetric_profile.lower);
    try std.testing.expectEqual(@as(usize, 2), symmetric_profile.upper);
    try std.testing.expectEqual(@as(usize, 4), try symmetric_profile.total());
    try std.testing.expect(symmetric_profile.meetsBounds(2, 2));
    try std.testing.expect(try symmetric_profile.totalMeetsBound(4));
    try std.testing.expect(try symmetric.lowerProfileMeetsBound(2));
    try std.testing.expect(!(try symmetric.lowerProfileMeetsBound(1)));
    try std.testing.expect(try symmetric.upperProfileMeetsBound(2));
    try std.testing.expect(!(try symmetric.upperProfileMeetsBound(1)));
    try std.testing.expect(try symmetric.profileMeetsBounds(2, 2));
    try std.testing.expect(!(try symmetric.profileMeetsBounds(1, 2)));
    try std.testing.expect(try symmetric.profileTotalMeetsBound(4));
    try std.testing.expect(!(try symmetric.profileTotalMeetsBound(3)));
    try std.testing.expect(try symmetric.structurallySymmetric());
    try std.testing.expect(try symmetric.numericallySymmetric(1e-12));
    try std.testing.expectApproxEqAbs(@as(f64, 0), try symmetric.symmetryResidualFrobeniusNorm(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0), try symmetric.symmetryRelativeResidualFrobeniusNorm(), 1e-12);
    try std.testing.expect(try symmetric.symmetryResidualFrobeniusNormMeetsBound(0));
    try std.testing.expect(try symmetric.symmetryRelativeResidualFrobeniusNormMeetsBound(0));
    try std.testing.expectApproxEqAbs(@as(f64, 2), symmetric.get(1, 2).?, 1e-12);
    try std.testing.expect(symmetric.get(0, 2) == null);
    var point_mut = try symmetric.clone();
    defer point_mut.deinit();
    try point_mut.setExisting(1, 2, 7);
    try std.testing.expectApproxEqAbs(@as(f64, 7), point_mut.get(1, 2).?, 1e-12);
    try std.testing.expectError(error.InvalidShape, point_mut.setStoredValue(0, 2, 9));
    try std.testing.expectError(error.IndexOutOfBounds, point_mut.setExisting(3, 0, 9));

    var nonsym_dense = try array_mod.Array(f64).fromSlice(gpa, &.{
        1, 2, 0,
        0, 0, 3,
        0, 0, 4,
    }, &.{ 3, 3 });
    defer nonsym_dense.deinit();
    var nonsym = try cooFromDense(f64, nonsym_dense);
    defer nonsym.deinit();
    try std.testing.expectEqual(@as(usize, 1), try nonsym.missingDiagonalCount());
    try std.testing.expect(try nonsym.missingDiagonalCountMeetsBound(1));
    try std.testing.expect(!(try nonsym.missingDiagonalCountMeetsBound(0)));
    try std.testing.expect(try nonsym.missingDiagonalCountInRange(1, 2));
    try std.testing.expect(!(try nonsym.missingDiagonalCountInRange(0, 0)));
    try std.testing.expectError(error.InvalidShape, nonsym.missingDiagonalCountInRange(2, 1));
    try std.testing.expectEqual(@as(usize, 0), try nonsym.zeroDiagonalCount());
    try std.testing.expect(try nonsym.zeroDiagonalCountMeetsBound(0));
    try std.testing.expect(!(try nonsym.zeroDiagonalCountInRange(1, 1)));
    try std.testing.expectEqual(@as(usize, 1), try nonsym.bandwidth());
    try std.testing.expect(!(try nonsym.structurallySymmetric()));
    try std.testing.expect(!(try nonsym.numericallySymmetric(1e-12)));
    try std.testing.expectApproxEqAbs(@sqrt(@as(f64, 26)), try nonsym.symmetryResidualFrobeniusNorm(), 1e-12);
    try std.testing.expectApproxEqAbs(@sqrt(@as(f64, 26)) / @sqrt(@as(f64, 30)), try nonsym.symmetryRelativeResidualFrobeniusNorm(), 1e-12);
    try std.testing.expect(try nonsym.symmetryResidualFrobeniusNormMeetsBound(@sqrt(@as(f64, 26))));
    try std.testing.expect(!(try nonsym.symmetryResidualFrobeniusNormMeetsBound(@sqrt(@as(f64, 26)) - 1e-12)));
    try std.testing.expect(try nonsym.symmetryRelativeResidualFrobeniusNormMeetsBound(@sqrt(@as(f64, 26)) / @sqrt(@as(f64, 30)) + 1e-12));
    try std.testing.expectError(error.InvalidShape, nonsym.symmetryResidualFrobeniusNormMeetsBound(-1));

    var rectangular = try cooFromSlices(f64, gpa, 2, 3, &.{ 0, 1 }, &.{ 0, 2 }, &.{ 1, 2 });
    defer rectangular.deinit();
    try std.testing.expectError(error.NonMatrixArray, rectangular.lowerNnz(false));
    try std.testing.expectError(error.NonMatrixArray, rectangular.upperNnz(false));
    try std.testing.expectError(error.NonMatrixArray, rectangular.missingDiagonalCountMeetsBound(0));
    try std.testing.expectError(error.NonMatrixArray, rectangular.zeroDiagonalCountInRange(0, 1));
    try std.testing.expectError(error.InvalidShape, rectangular.missingDiagonalCountInRange(2, 1));
    try std.testing.expectError(error.NonMatrixArray, rectangular.traceInRange(0, 3));
    try std.testing.expectError(error.NonMatrixArray, rectangular.normalizedTrace());
    try std.testing.expectError(error.NonMatrixArray, rectangular.normalizedTraceInRange(0, 1));
    try std.testing.expectError(error.NonMatrixArray, rectangular.bandwidthMeetsBound(1));
    try std.testing.expectError(error.NonMatrixArray, rectangular.lowerNnzMeetsBound(false, 1));
    try std.testing.expectError(error.NonMatrixArray, rectangular.upperNnzInRange(false, 0, 1));
    try std.testing.expectError(error.NonMatrixArray, rectangular.profile());
    try std.testing.expectError(error.NonMatrixArray, rectangular.symmetryResidualFrobeniusNorm());
    try std.testing.expectError(error.NonMatrixArray, rectangular.symmetryRelativeResidualFrobeniusNorm());
    try std.testing.expectError(error.NonMatrixArray, rectangular.symmetryResidualFrobeniusNormMeetsBound(1));
    try std.testing.expectError(error.NonMatrixArray, rectangular.profileMeetsBounds(1, 1));
    try std.testing.expectError(error.NonMatrixArray, rectangular.profileTotalMeetsBound(1));

    var duplicate_diagonal = try cooFromSlices(f64, gpa, 2, 2, &.{ 0, 0, 1, 1, 1 }, &.{ 0, 0, 0, 1, 1 }, &.{ 1, 2, 3, 4, -4 });
    defer duplicate_diagonal.deinit();
    var duplicate_diag = try duplicate_diagonal.diagonal();
    defer duplicate_diag.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 3, 0 }, duplicate_diag.data);
    var duplicate_diagonal_mut = try duplicate_diagonal.clone();
    defer duplicate_diagonal_mut.deinit();
    try duplicate_diagonal_mut.addToDiagonal(1);
    try std.testing.expectEqualSlices(f64, &.{ 2, 3, 3, 5, -3 }, duplicate_diagonal_mut.values);
    try duplicate_diagonal_mut.setDiagonal(7);
    try std.testing.expectEqualSlices(f64, &.{ 7, 7, 3, 7, 7 }, duplicate_diagonal_mut.values);
    try std.testing.expectApproxEqAbs(@as(f64, 3), try duplicate_diagonal.trace(), 1e-12);
    try std.testing.expect(try duplicate_diagonal.traceInRange(3, 3));
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), try duplicate_diagonal.normalizedTrace(), 1e-12);
    try std.testing.expect(try duplicate_diagonal.normalizedTraceInRange(1.4, 1.6));
    try std.testing.expectEqual(@as(usize, 0), try duplicate_diagonal.missingDiagonalCount());
    try std.testing.expectEqual(@as(usize, 1), try duplicate_diagonal.zeroDiagonalCount());
    try std.testing.expect(try duplicate_diagonal.zeroDiagonalCountMeetsBound(1));
    try std.testing.expect(!(try duplicate_diagonal.zeroDiagonalCountMeetsBound(0)));
    try std.testing.expect(try duplicate_diagonal.zeroDiagonalCountInRange(1, 1));
    try std.testing.expect(!(try duplicate_diagonal.zeroDiagonalCountInRange(0, 0)));
    try std.testing.expectApproxEqAbs(@as(f64, 3), duplicate_diagonal.get(0, 0).?, 1e-12);
    try std.testing.expect(!(try duplicate_diagonal.structurallySymmetric()));

    var empty_square = try cooFromSlices(f64, gpa, 0, 0, &.{}, &.{}, &.{});
    defer empty_square.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0), try empty_square.trace(), 1e-12);
    try std.testing.expectError(error.EmptyArray, empty_square.normalizedTrace());
    try std.testing.expectError(error.EmptyArray, empty_square.normalizedTraceInRange(0, 1));

    var duplicate_symmetric = try cooFromSlices(f64, gpa, 2, 2, &.{ 0, 0, 1, 1 }, &.{ 1, 1, 0, 0 }, &.{ 1, 2, 1.5, 1.5 });
    defer duplicate_symmetric.deinit();
    try std.testing.expectEqual(@as(usize, 2), try duplicate_symmetric.missingDiagonalCount());
    try std.testing.expectError(error.InvalidShape, duplicate_symmetric.addToDiagonal(1));
    try std.testing.expect(try duplicate_symmetric.structurallySymmetric());
    try std.testing.expect(try duplicate_symmetric.numericallySymmetric(1e-12));
}

test "csr sparse bridge dense roundtrip and matvec" {
    const gpa = std.testing.allocator;
    var dense = try array_mod.Array(f64).fromSlice(gpa, &.{
        10, 0, 2, 0,
        0,  3, 0, 4,
        5,  0, 0, 6,
    }, &.{ 3, 4 });
    defer dense.deinit();

    var csr = try csrFromDense(f64, dense);
    defer csr.deinit();
    try std.testing.expectEqual(@as(usize, 6), csr.nnz());
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 4, 6 }, csr.row_offsets);
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 1, 3, 0, 3 }, csr.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 10, 2, 3, 4, 5, 6 }, csr.values);
    var csr_point_mut = try csr.clone();
    defer csr_point_mut.deinit();
    try csr_point_mut.setExisting(0, 2, 11);
    try std.testing.expectApproxEqAbs(@as(f64, 11), csr_point_mut.get(0, 2).?, 1e-12);
    try std.testing.expectError(error.InvalidShape, csr_point_mut.setStoredValue(0, 1, 9));
    try std.testing.expectError(error.IndexOutOfBounds, csr_point_mut.setExisting(3, 0, 9));
    var square_dense = try array_mod.Array(f64).fromSlice(gpa, &.{
        4, 1, 0,
        1, 5, 2,
        0, 2, 6,
    }, &.{ 3, 3 });
    defer square_dense.deinit();
    var square_csr = try csrFromDense(f64, square_dense);
    defer square_csr.deinit();
    try square_csr.addToDiagonal(2);
    var square_csr_diag = try square_csr.diagonal();
    defer square_csr_diag.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 6, 7, 8 }, square_csr_diag.data);
    try square_csr.setDiagonal(3);
    var square_csr_set_diag = try square_csr.diagonal();
    defer square_csr_set_diag.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 3, 3, 3 }, square_csr_set_diag.data);
    try square_csr.setDiagonalVector(&.{ 1, 2, 3 });
    try square_csr.addDiagonalVector(&.{ 1, 1, 1 });
    try square_csr.multiplyDiagonalVector(&.{ 2, 3, 4 });
    var square_csr_vector_diag = try square_csr.diagonal();
    defer square_csr_vector_diag.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 4, 9, 16 }, square_csr_vector_diag.data);
    try std.testing.expectError(error.ShapeMismatch, square_csr.addDiagonalValues(&.{1}));
    var missing_csr = try csrFromCompressed(f64, gpa, 2, 2, &.{ 0, 1, 2 }, &.{ 1, 0 }, &.{ 1, 1 });
    defer missing_csr.deinit();
    try std.testing.expectError(error.InvalidShape, missing_csr.setDiagonal(1));

    var dense2 = try csr.toDense();
    defer dense2.deinit();
    try std.testing.expectEqualSlices(f64, dense.data, dense2.data);

    var csc = try csr.toCsc();
    defer csc.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 3, 4, 6 }, csc.col_offsets);
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 1, 0, 1, 2 }, csc.row_indices);
    try std.testing.expectEqualSlices(f64, &.{ 10, 5, 3, 2, 4, 6 }, csc.values);
    var csc_point_mut = try csc.clone();
    defer csc_point_mut.deinit();
    try csc_point_mut.setExisting(2, 0, 12);
    try std.testing.expectApproxEqAbs(@as(f64, 12), csc_point_mut.get(2, 0).?, 1e-12);
    try std.testing.expectError(error.InvalidShape, csc_point_mut.setStoredValue(0, 1, 9));
    try std.testing.expectError(error.IndexOutOfBounds, csc_point_mut.setExisting(0, 4, 9));
    var square_csc = try cscFromDense(f64, square_dense);
    defer square_csc.deinit();
    try square_csc.addToDiagonal(2);
    var square_csc_diag = try square_csc.diagonal();
    defer square_csc_diag.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 6, 7, 8 }, square_csc_diag.data);
    try square_csc.setDiagonal(3);
    var square_csc_set_diag = try square_csc.diagonal();
    defer square_csc_set_diag.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 3, 3, 3 }, square_csc_set_diag.data);
    try square_csc.setDiagonalVector(&.{ 1, 2, 3 });
    try square_csc.addDiagonalVector(&.{ 1, 1, 1 });
    try square_csc.multiplyDiagonalVector(&.{ 2, 3, 4 });
    var square_csc_vector_diag = try square_csc.diagonal();
    defer square_csc_vector_diag.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 4, 9, 16 }, square_csc_vector_diag.data);
    try std.testing.expectError(error.ShapeMismatch, square_csc.multiplyDiagonalValues(&.{1}));
    var missing_csc = try cscFromCompressed(f64, gpa, 2, 2, &.{ 0, 1, 2 }, &.{ 1, 0 }, &.{ 1, 1 });
    defer missing_csc.deinit();
    try std.testing.expectError(error.InvalidShape, missing_csc.addToDiagonal(1));
    var csc_dense = try csc.toDense();
    defer csc_dense.deinit();
    try std.testing.expectEqualSlices(f64, dense.data, csc_dense.data);

    var x = try array_mod.Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4 }, &.{4});
    defer x.deinit();
    var y = try csr.matvec(x);
    defer y.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 16, 22, 29 }, y.data);
}

test "csr sparse matmat transpose and statistics" {
    const gpa = std.testing.allocator;
    var dense = try array_mod.Array(f64).fromSlice(gpa, &.{
        1, 0, 2,
        0, 3, 0,
    }, &.{ 2, 3 });
    defer dense.deinit();
    var csr = try csrFromDense(f64, dense);
    defer csr.deinit();

    var rhs = try array_mod.Array(f64).fromSlice(gpa, &.{
        1, 2,
        3, 4,
        5, 6,
    }, &.{ 3, 2 });
    defer rhs.deinit();
    var product = try csr.matmat(rhs);
    defer product.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, product.shape);
    try std.testing.expectEqualSlices(f64, &.{ 11, 14, 9, 12 }, product.data);

    var sparse_rhs_dense = try array_mod.Array(f64).fromSlice(gpa, &.{
        8, 4,
        5, 0,
        6, 7,
    }, &.{ 3, 2 });
    defer sparse_rhs_dense.deinit();
    var sparse_rhs = try csrFromDense(f64, sparse_rhs_dense);
    defer sparse_rhs.deinit();
    var sparse_product = try csr.matmulSparse(sparse_rhs);
    defer sparse_product.deinit();
    try std.testing.expectEqual(@as(usize, 3), sparse_product.nnz());
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 3 }, sparse_product.row_offsets);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 0 }, sparse_product.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 20, 18, 15 }, sparse_product.values);
    var sparse_product_dense = try sparse_product.toDense();
    defer sparse_product_dense.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 20, 18, 15, 0 }, sparse_product_dense.data);

    var coo_lhs = try csr.toCoo();
    defer coo_lhs.deinit();
    var coo_rhs = try sparse_rhs.toCoo();
    defer coo_rhs.deinit();
    var coo_product = try coo_lhs.matmulSparse(coo_rhs);
    defer coo_product.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 0, 1 }, coo_product.row_indices);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 0 }, coo_product.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 20, 18, 15 }, coo_product.values);

    var csc_lhs = try csr.toCsc();
    defer csc_lhs.deinit();
    var csc_rhs = try sparse_rhs.toCsc();
    defer csc_rhs.deinit();
    var csc_product = try csc_lhs.matmulSparse(csc_rhs);
    defer csc_product.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 3 }, csc_product.col_offsets);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 0 }, csc_product.row_indices);
    try std.testing.expectEqualSlices(f64, &.{ 20, 15, 18 }, csc_product.values);
    try std.testing.expectError(error.ShapeMismatch, sparse_rhs.matmulSparse(sparse_rhs));

    var transposed = try csr.transpose();
    defer transposed.deinit();
    try std.testing.expectEqual(@as(usize, 3), transposed.rows);
    try std.testing.expectEqual(@as(usize, 2), transposed.cols);
    var transposed_dense = try transposed.toDense();
    defer transposed_dense.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 0, 0, 3, 2, 0 }, transposed_dense.data);
    var csr_mt = try csr.matrixTranspose();
    defer csr_mt.deinit();
    var csr_mt_dense = try csr_mt.toDense();
    defer csr_mt_dense.deinit();
    try std.testing.expectEqualSlices(f64, transposed_dense.data, csr_mt_dense.data);
    var csr_m_t = try csr.mT();
    defer csr_m_t.deinit();
    var csr_m_t_dense = try csr_m_t.toDense();
    defer csr_m_t_dense.deinit();
    try std.testing.expectEqualSlices(f64, transposed_dense.data, csr_m_t_dense.data);

    var csc_transposed = try csc_lhs.transpose();
    defer csc_transposed.deinit();
    var csc_transposed_dense = try csc_transposed.toDense();
    defer csc_transposed_dense.deinit();
    try std.testing.expectEqualSlices(f64, transposed_dense.data, csc_transposed_dense.data);
    var csc_mt = try csc_lhs.matrixTranspose();
    defer csc_mt.deinit();
    var csc_mt_dense = try csc_mt.toDense();
    defer csc_mt_dense.deinit();
    try std.testing.expectEqualSlices(f64, transposed_dense.data, csc_mt_dense.data);
    var csc_m_t = try csc_lhs.mT();
    defer csc_m_t.deinit();
    var csc_m_t_dense = try csc_m_t.toDense();
    defer csc_m_t_dense.deinit();
    try std.testing.expectEqualSlices(f64, transposed_dense.data, csc_m_t_dense.data);

    try std.testing.expectApproxEqAbs(@as(f64, 6), csr.sum(), 1e-12);
    try std.testing.expect(try csr.sumInRange(6, 6));
    try std.testing.expect(try csr.sumInRange(5.5, 6.5));
    try std.testing.expect(!(try csr.sumInRange(6.5, 7)));
    try std.testing.expectError(error.InvalidShape, csr.sumInRange(7, 6));
    try std.testing.expectApproxEqAbs(@as(f64, 6), csr.absSum(), 1e-12);
    try std.testing.expect(try csr.absSumInRange(6, 6));
    try std.testing.expect(try csr.absSumInRange(5, 7));
    try std.testing.expect(!(try csr.absSumInRange(7, 8)));
    try std.testing.expectError(error.InvalidShape, csr.absSumInRange(std.math.nan(f64), 6));
    try std.testing.expectApproxEqAbs(@as(f64, @sqrt(14.0)), csr.frobeniusNorm(), 1e-12);
    try std.testing.expect(try csr.frobeniusNormMeetsBound(@sqrt(@as(f64, 14.0))));
    try std.testing.expect(!(try csr.frobeniusNormMeetsBound(@sqrt(@as(f64, 14.0)) - 1e-12)));
    try std.testing.expectError(error.InvalidShape, csr.frobeniusNormMeetsBound(std.math.nan(f64)));
    try std.testing.expectApproxEqAbs(@as(f64, 3), try csr.oneNorm(), 1e-12);
    try std.testing.expect(try csr.oneNormMeetsBound(3));
    try std.testing.expect(!(try csr.oneNormMeetsBound(2.999)));
    try std.testing.expectError(error.InvalidShape, csr.oneNormMeetsBound(-1));
    try std.testing.expectApproxEqAbs(@as(f64, 3), try csr.infNorm(), 1e-12);
    try std.testing.expect(try csr.infNormMeetsBound(3));
    try std.testing.expect(!(try csr.infNormMeetsBound(2.999)));
    try std.testing.expectError(error.InvalidShape, csr.infNormMeetsBound(std.math.inf(f64)));
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), try csr.density(), 1e-12);
}

test "csr sparse row and column statistics" {
    const gpa = std.testing.allocator;
    var dense = try array_mod.Array(f64).fromSlice(gpa, &.{
        1, 0, -2,
        0, 3, 0,
        4, 0, 5,
    }, &.{ 3, 3 });
    defer dense.deinit();
    var csr = try csrFromDense(f64, dense);
    defer csr.deinit();

    try std.testing.expectApproxEqAbs(@as(f64, -2), try csr.minValue(), 1e-12);
    try std.testing.expectEqual(@as(usize, 1), try csr.minValueIndex());
    try std.testing.expectApproxEqAbs(@as(f64, 5), try csr.maxValue(), 1e-12);
    try std.testing.expectEqual(@as(usize, 4), try csr.maxValueIndex());
    try std.testing.expectApproxEqAbs(@as(f64, 1), try csr.minAbsValue(), 1e-12);
    try std.testing.expectEqual(@as(usize, 0), try csr.minAbsValueIndex());
    try std.testing.expectApproxEqAbs(@as(f64, 5), try csr.maxAbsValue(), 1e-12);
    try std.testing.expectEqual(@as(usize, 4), try csr.maxAbsValueIndex());
    const expected_mean = @as(f64, 11.0 / 9.0);
    const expected_variance = @as(f64, 55.0 / 9.0) - expected_mean * expected_mean;
    const expected_stddev = @sqrt(expected_variance);
    const expected_sample_variance = @as(f64, (55.0 - (11.0 * 11.0) / 9.0) / 8.0);
    const expected_sample_stddev = @sqrt(expected_sample_variance);
    try std.testing.expectApproxEqAbs(expected_mean, try csr.mean(), 1e-12);
    try std.testing.expect(try csr.meanInRange(expected_mean, expected_mean));
    try std.testing.expect(try csr.meanInRange(1.2, 1.3));
    try std.testing.expect(!(try csr.meanInRange(1.3, 1.4)));
    try std.testing.expectError(error.InvalidShape, csr.meanInRange(std.math.nan(f64), expected_mean));
    try std.testing.expectApproxEqAbs(expected_variance, try csr.variance(0), 1e-12);
    try std.testing.expect(try csr.varianceInRange(0, expected_variance - 1e-12, expected_variance + 1e-12));
    try std.testing.expect(try csr.varianceInRange(0, 4.6, 4.7));
    try std.testing.expect(!(try csr.varianceInRange(0, 4.7, 4.8)));
    try std.testing.expectError(error.InvalidShape, csr.varianceInRange(0, -0.1, 1));
    try std.testing.expectApproxEqAbs(expected_stddev, try csr.stddev(0), 1e-12);
    try std.testing.expect(try csr.stddevInRange(0, expected_stddev - 1e-12, expected_stddev + 1e-12));
    try std.testing.expect(!(try csr.stddevInRange(0, expected_stddev + 0.1, expected_stddev + 0.2)));
    try std.testing.expectError(error.InvalidShape, csr.stddevInRange(0, 2, 1));
    try std.testing.expectApproxEqAbs(expected_sample_variance, try csr.sampleVariance(), 1e-12);
    try std.testing.expect(try csr.sampleVarianceInRange(expected_sample_variance - 1e-12, expected_sample_variance + 1e-12));
    try std.testing.expect(!(try csr.sampleVarianceInRange(expected_sample_variance + 0.1, expected_sample_variance + 0.2)));
    try std.testing.expectApproxEqAbs(expected_sample_stddev, try csr.sampleStddev(), 1e-12);
    try std.testing.expect(try csr.sampleStddevInRange(expected_sample_stddev - 1e-12, expected_sample_stddev + 1e-12));
    try std.testing.expectError(error.InvalidShape, csr.sampleStddevInRange(-0.1, expected_sample_stddev));

    var row_vars = try csr.rowVariances(0);
    defer row_vars.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 14.0 / 9.0), row_vars.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2), row_vars.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 14.0 / 3.0), row_vars.data[2], 1e-12);
    try std.testing.expect(try csr.rowVariancesInRange(0, 14.0 / 9.0, 14.0 / 3.0));
    try std.testing.expect(!(try csr.rowVariancesInRange(0, 2.1, 14.0 / 3.0)));
    try std.testing.expectError(error.InvalidShape, csr.rowVariancesInRange(0, -0.1, 1));
    var col_vars = try csr.columnVariances(0);
    defer col_vars.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 26.0 / 9.0), col_vars.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2), col_vars.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 26.0 / 3.0), col_vars.data[2], 1e-12);
    try std.testing.expect(try csr.columnVariancesInRange(0, 2, 26.0 / 3.0));
    try std.testing.expect(!(try csr.columnVariancesInRange(0, 2.1, 26.0 / 3.0)));
    try std.testing.expect(try csr.columnSampleVariancesInRange(3, 13));
    try std.testing.expect(!(try csr.columnSampleVariancesInRange(4, 13)));
    try std.testing.expect(try csr.columnSampleStddevsInRange(@sqrt(3.0), @sqrt(13.0)));
    try std.testing.expect(!(try csr.columnSampleStddevsInRange(1.8, @sqrt(13.0))));
    var row_stds = try csr.rowStddevs(0);
    defer row_stds.deinit();
    try std.testing.expectApproxEqAbs(@sqrt(14.0 / 9.0), row_stds.data[0], 1e-12);
    try std.testing.expect(try csr.rowStddevsInRange(0, @sqrt(14.0 / 9.0), @sqrt(14.0 / 3.0)));
    try std.testing.expect(!(try csr.rowStddevsInRange(0, 1.5, 2)));
    try std.testing.expectError(error.InvalidShape, csr.rowStddevsInRange(0, -0.1, 1));
    var col_stds = try csr.columnStddevs(0);
    defer col_stds.deinit();
    try std.testing.expectApproxEqAbs(@sqrt(26.0 / 9.0), col_stds.data[0], 1e-12);
    try std.testing.expect(try csr.columnStddevsInRange(0, @sqrt(2.0), @sqrt(26.0 / 3.0)));
    try std.testing.expect(!(try csr.columnStddevsInRange(0, 1.5, @sqrt(26.0 / 3.0))));
    var row_sample_vars = try csr.rowSampleVariances();
    defer row_sample_vars.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 7.0 / 3.0), row_sample_vars.data[0], 1e-12);
    try std.testing.expect(try csr.rowSampleVariancesInRange(7.0 / 3.0, 7));
    try std.testing.expect(!(try csr.rowSampleVariancesInRange(2.5, 7)));
    try std.testing.expect(try csr.rowSampleStddevsInRange(@sqrt(7.0 / 3.0), @sqrt(7.0)));
    try std.testing.expect(!(try csr.rowSampleStddevsInRange(1.6, @sqrt(7.0))));

    var row_means = try csr.rowMeans();
    defer row_means.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, -1.0 / 3.0), row_means.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), row_means.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3), row_means.data[2], 1e-12);
    try std.testing.expect(try csr.rowMeansInRange(-1.0 / 3.0, 3));
    try std.testing.expect(!(try csr.rowMeansInRange(0, 3)));
    try std.testing.expectError(error.InvalidShape, csr.rowMeansInRange(3, 2));
    var col_means = try csr.columnMeans();
    defer col_means.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 3.0), col_means.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), col_means.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), col_means.data[2], 1e-12);
    try std.testing.expect(try csr.columnMeansInRange(1, 5.0 / 3.0));
    try std.testing.expect(!(try csr.columnMeansInRange(1.1, 5.0 / 3.0)));

    var row_mins = try csr.rowMins();
    defer row_mins.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -2, 0, 0 }, row_mins.data);
    try std.testing.expect(try csr.rowMinsInRange(-2, 0));
    try std.testing.expect(!(try csr.rowMinsInRange(-1, 0)));
    try std.testing.expectError(error.InvalidShape, csr.rowMinsInRange(1, 0));
    var row_maxes = try csr.rowMaxes();
    defer row_maxes.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 5 }, row_maxes.data);
    try std.testing.expect(try csr.rowMaxesInRange(1, 5));
    try std.testing.expect(!(try csr.rowMaxesInRange(2, 5)));
    var col_mins = try csr.columnMins();
    defer col_mins.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 0, -2 }, col_mins.data);
    try std.testing.expect(try csr.columnMinsInRange(-2, 0));
    try std.testing.expect(!(try csr.columnMinsInRange(-1, 0)));
    var col_maxes = try csr.columnMaxes();
    defer col_maxes.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 4, 3, 5 }, col_maxes.data);
    try std.testing.expect(try csr.columnMaxesInRange(3, 5));
    try std.testing.expect(!(try csr.columnMaxesInRange(4, 5)));
    var row_min_abs = try csr.rowMinAbs();
    defer row_min_abs.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 4 }, row_min_abs.data);
    try std.testing.expect(try csr.rowMinAbsInRange(1, 4));
    try std.testing.expect(!(try csr.rowMinAbsInRange(2, 4)));
    try std.testing.expectError(error.InvalidShape, csr.rowMinAbsInRange(5, 4));
    var row_max_abs = try csr.rowMaxAbs();
    defer row_max_abs.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 2, 3, 5 }, row_max_abs.data);
    try std.testing.expect(try csr.rowMaxAbsInRange(2, 5));
    try std.testing.expect(!(try csr.rowMaxAbsInRange(3, 5)));
    var col_min_abs = try csr.columnMinAbs();
    defer col_min_abs.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 2 }, col_min_abs.data);
    try std.testing.expect(try csr.columnMinAbsInRange(1, 3));
    try std.testing.expect(!(try csr.columnMinAbsInRange(2, 3)));
    var col_max_abs = try csr.columnMaxAbs();
    defer col_max_abs.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 4, 3, 5 }, col_max_abs.data);
    try std.testing.expect(try csr.columnMaxAbsInRange(3, 5));
    try std.testing.expect(!(try csr.columnMaxAbsInRange(4, 5)));

    var row_nnz = try csr.rowNnz();
    defer row_nnz.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 1, 2 }, row_nnz.data);
    var col_nnz = try csr.columnNnz();
    defer col_nnz.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 1, 2 }, col_nnz.data);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 3.0), try csr.averageRowNnz(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 3.0), try csr.averageColumnNnz(), 1e-12);
    try std.testing.expect(try csr.averageRowNnzInRange(1.6, 1.7));
    try std.testing.expect(!(try csr.averageColumnNnzInRange(0, 1.6)));
    try std.testing.expectEqual(@as(usize, 0), csr.emptyRowCount());
    try std.testing.expectEqual(@as(usize, 0), try csr.emptyColumnCount());
    try std.testing.expectApproxEqAbs(@as(f64, 0), try csr.emptyRowFraction(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0), try csr.emptyColumnFraction(), 1e-12);
    try std.testing.expect(try csr.emptyRowFractionInRange(0, 0));
    try std.testing.expect(try csr.emptyColumnFractionInRange(0, 0));

    var row_sums = try csr.rowSums();
    defer row_sums.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -1, 3, 9 }, row_sums.data);
    try std.testing.expect(try csr.rowSumsInRange(-1, 9));
    try std.testing.expect(!(try csr.rowSumsInRange(0, 9)));
    try std.testing.expectError(error.InvalidShape, csr.rowSumsInRange(10, 9));
    var col_sums = try csr.columnSums();
    defer col_sums.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 5, 3, 3 }, col_sums.data);
    try std.testing.expect(try csr.columnSumsInRange(3, 5));
    try std.testing.expect(!(try csr.columnSumsInRange(4, 5)));

    var row_abs = try csr.rowAbsSums();
    defer row_abs.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 3, 3, 9 }, row_abs.data);
    try std.testing.expect(try csr.rowAbsSumsInRange(3, 9));
    try std.testing.expect(!(try csr.rowAbsSumsInRange(4, 9)));
    try std.testing.expectError(error.InvalidShape, csr.rowAbsSumsInRange(10, 9));
    var col_abs = try csr.columnAbsSums();
    defer col_abs.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 5, 3, 7 }, col_abs.data);
    try std.testing.expect(try csr.columnAbsSumsInRange(3, 7));
    try std.testing.expect(!(try csr.columnAbsSumsInRange(4, 7)));

    var row_norms = try csr.rowNorms();
    defer row_norms.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, @sqrt(5.0)), row_norms.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3), row_norms.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, @sqrt(41.0)), row_norms.data[2], 1e-12);
    try std.testing.expect(try csr.rowNormsInRange(@sqrt(@as(f64, 5.0)), @sqrt(@as(f64, 41.0))));
    try std.testing.expect(!(try csr.rowNormsInRange(3.1, @sqrt(@as(f64, 41.0)))));
    try std.testing.expectError(error.InvalidShape, csr.rowNormsInRange(-0.1, 1));
    var col_norms = try csr.columnNorms();
    defer col_norms.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, @sqrt(17.0)), col_norms.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3), col_norms.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, @sqrt(29.0)), col_norms.data[2], 1e-12);
    try std.testing.expect(try csr.columnNormsInRange(3, @sqrt(@as(f64, 29.0))));
    try std.testing.expect(!(try csr.columnNormsInRange(3.1, @sqrt(@as(f64, 29.0)))));
    try std.testing.expect(try csr.densityInRange(5.0 / 9.0, 5.0 / 9.0));
    try std.testing.expect(!(try csr.densityInRange(0, 0.5)));
    try std.testing.expectError(error.InvalidShape, csr.densityInRange(0.6, 0.5));
}

test "csr sparse diagonal trace bandwidth and symmetry" {
    const gpa = std.testing.allocator;
    var symmetric_dense = try array_mod.Array(f64).fromSlice(gpa, &.{
        4, 1, 0,
        1, 5, 2,
        0, 2, 6,
    }, &.{ 3, 3 });
    defer symmetric_dense.deinit();
    var symmetric = try csrFromDense(f64, symmetric_dense);
    defer symmetric.deinit();

    var diagonal = try symmetric.diagonal();
    defer diagonal.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 4, 5, 6 }, diagonal.data);
    try std.testing.expectApproxEqAbs(@as(f64, 15), try symmetric.trace(), 1e-12);
    try std.testing.expect(try symmetric.traceInRange(15, 15));
    try std.testing.expect(try symmetric.traceInRange(14.5, 15.5));
    try std.testing.expect(!(try symmetric.traceInRange(15.5, 16)));
    try std.testing.expectError(error.InvalidShape, symmetric.traceInRange(std.math.inf(f64), 15));
    try std.testing.expectApproxEqAbs(@as(f64, 5), try symmetric.normalizedTrace(), 1e-12);
    try std.testing.expect(try symmetric.normalizedTraceInRange(5, 5));
    try std.testing.expect(try symmetric.normalizedTraceInRange(4.9, 5.1));
    try std.testing.expect(!(try symmetric.normalizedTraceInRange(5.1, 6)));
    try std.testing.expectError(error.InvalidShape, symmetric.normalizedTraceInRange(std.math.nan(f64), 5));
    try std.testing.expectEqual(@as(usize, 0), try symmetric.missingDiagonalCount());
    try std.testing.expect(try symmetric.missingDiagonalCountMeetsBound(0));
    try std.testing.expect(try symmetric.missingDiagonalCountInRange(0, 0));
    try std.testing.expectEqual(@as(usize, 0), try symmetric.zeroDiagonalCount());
    try std.testing.expect(try symmetric.zeroDiagonalCountMeetsBound(0));
    try std.testing.expect(try symmetric.zeroDiagonalCountInRange(0, 0));
    try std.testing.expectEqual(@as(usize, 1), try symmetric.bandwidth());
    try std.testing.expect(try symmetric.bandwidthMeetsBound(1));
    try std.testing.expect(!(try symmetric.bandwidthMeetsBound(0)));
    try std.testing.expectEqual(@as(usize, 2), try symmetric.columnIntersectionBandwidth());
    try std.testing.expect(try symmetric.columnIntersectionBandwidthMeetsBound(2));
    try std.testing.expect(!(try symmetric.columnIntersectionBandwidthMeetsBound(1)));
    try std.testing.expectEqual(@as(usize, 5), try symmetric.lowerNnz(false));
    try std.testing.expectEqual(@as(usize, 2), try symmetric.lowerNnz(true));
    try std.testing.expectEqual(@as(usize, 5), try symmetric.upperNnz(false));
    try std.testing.expectEqual(@as(usize, 2), try symmetric.upperNnz(true));
    try std.testing.expect(try symmetric.lowerNnzMeetsBound(false, 5));
    try std.testing.expect(!(try symmetric.lowerNnzMeetsBound(false, 4)));
    try std.testing.expect(try symmetric.upperNnzMeetsBound(true, 2));
    try std.testing.expect(!(try symmetric.upperNnzMeetsBound(true, 1)));
    try std.testing.expect(try symmetric.lowerNnzInRange(false, 5, 5));
    try std.testing.expect(try symmetric.upperNnzInRange(true, 0, 2));
    try std.testing.expect(!(try symmetric.lowerNnzInRange(true, 3, 5)));
    try std.testing.expectError(error.InvalidShape, symmetric.upperNnzInRange(false, 6, 5));
    try std.testing.expectEqual(@as(usize, 2), try symmetric.lowerProfile());
    try std.testing.expectEqual(@as(usize, 2), try symmetric.upperProfile());
    const symmetric_profile = try symmetric.profile();
    try std.testing.expectEqual(@as(usize, 2), symmetric_profile.lower);
    try std.testing.expectEqual(@as(usize, 2), symmetric_profile.upper);
    try std.testing.expectEqual(@as(usize, 4), try symmetric_profile.total());
    try std.testing.expect(symmetric_profile.meetsBounds(2, 2));
    try std.testing.expect(try symmetric_profile.totalMeetsBound(4));
    try std.testing.expect(try symmetric.lowerProfileMeetsBound(2));
    try std.testing.expect(!(try symmetric.lowerProfileMeetsBound(1)));
    try std.testing.expect(try symmetric.upperProfileMeetsBound(2));
    try std.testing.expect(!(try symmetric.upperProfileMeetsBound(1)));
    try std.testing.expect(try symmetric.profileMeetsBounds(2, 2));
    try std.testing.expect(!(try symmetric.profileMeetsBounds(1, 2)));
    try std.testing.expect(try symmetric.profileTotalMeetsBound(4));
    try std.testing.expect(!(try symmetric.profileTotalMeetsBound(3)));
    try std.testing.expect(try symmetric.structurallySymmetric());
    try std.testing.expect(try symmetric.numericallySymmetric(1e-12));
    try std.testing.expectApproxEqAbs(@as(f64, 0), try symmetric.symmetryResidualFrobeniusNorm(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0), try symmetric.symmetryRelativeResidualFrobeniusNorm(), 1e-12);
    try std.testing.expect(try symmetric.symmetryResidualFrobeniusNormMeetsBound(0));
    try std.testing.expect(try symmetric.symmetryRelativeResidualFrobeniusNormMeetsBound(0));

    var nonsym_dense = try array_mod.Array(f64).fromSlice(gpa, &.{
        1, 2, 0,
        0, 0, 3,
        0, 0, 4,
    }, &.{ 3, 3 });
    defer nonsym_dense.deinit();
    var nonsym = try csrFromDense(f64, nonsym_dense);
    defer nonsym.deinit();
    try std.testing.expectEqual(@as(usize, 1), try nonsym.missingDiagonalCount());
    try std.testing.expect(try nonsym.missingDiagonalCountMeetsBound(1));
    try std.testing.expect(!(try nonsym.missingDiagonalCountMeetsBound(0)));
    try std.testing.expect(try nonsym.missingDiagonalCountInRange(1, 2));
    try std.testing.expect(!(try nonsym.missingDiagonalCountInRange(0, 0)));
    try std.testing.expectError(error.InvalidShape, nonsym.missingDiagonalCountInRange(2, 1));
    try std.testing.expectEqual(@as(usize, 0), try nonsym.zeroDiagonalCount());
    try std.testing.expect(try nonsym.zeroDiagonalCountMeetsBound(0));
    try std.testing.expect(!(try nonsym.zeroDiagonalCountInRange(1, 1)));
    try std.testing.expectEqual(@as(usize, 1), try nonsym.bandwidth());
    try std.testing.expect(!(try nonsym.structurallySymmetric()));
    try std.testing.expect(!(try nonsym.numericallySymmetric(1e-12)));
    try std.testing.expectApproxEqAbs(@sqrt(@as(f64, 26)), try nonsym.symmetryResidualFrobeniusNorm(), 1e-12);
    try std.testing.expectApproxEqAbs(@sqrt(@as(f64, 26)) / @sqrt(@as(f64, 30)), try nonsym.symmetryRelativeResidualFrobeniusNorm(), 1e-12);
    try std.testing.expect(try nonsym.symmetryResidualFrobeniusNormMeetsBound(@sqrt(@as(f64, 26))));
    try std.testing.expect(!(try nonsym.symmetryResidualFrobeniusNormMeetsBound(@sqrt(@as(f64, 26)) - 1e-12)));
    try std.testing.expect(try nonsym.symmetryRelativeResidualFrobeniusNormMeetsBound(@sqrt(@as(f64, 26)) / @sqrt(@as(f64, 30)) + 1e-12));
    try std.testing.expectError(error.InvalidShape, nonsym.symmetryResidualFrobeniusNormMeetsBound(-1));

    var rectangular = try csrFromCompressed(f64, gpa, 2, 3, &.{ 0, 1, 2 }, &.{ 0, 2 }, &.{ 1, 2 });
    defer rectangular.deinit();
    try std.testing.expectError(error.NonMatrixArray, rectangular.lowerNnz(false));
    try std.testing.expectError(error.NonMatrixArray, rectangular.upperNnz(false));
    try std.testing.expectError(error.NonMatrixArray, rectangular.missingDiagonalCountMeetsBound(0));
    try std.testing.expectError(error.NonMatrixArray, rectangular.zeroDiagonalCountInRange(0, 1));
    try std.testing.expectError(error.InvalidShape, rectangular.missingDiagonalCountInRange(2, 1));
    try std.testing.expectError(error.NonMatrixArray, rectangular.traceInRange(0, 3));
    try std.testing.expectError(error.NonMatrixArray, rectangular.normalizedTrace());
    try std.testing.expectError(error.NonMatrixArray, rectangular.normalizedTraceInRange(0, 1));
    try std.testing.expectError(error.NonMatrixArray, rectangular.bandwidthMeetsBound(1));
    try std.testing.expectError(error.NonMatrixArray, rectangular.lowerNnzMeetsBound(false, 1));
    try std.testing.expectError(error.NonMatrixArray, rectangular.upperNnzInRange(false, 0, 1));
    try std.testing.expectError(error.NonMatrixArray, rectangular.profile());
    try std.testing.expectError(error.NonMatrixArray, rectangular.symmetryResidualFrobeniusNorm());
    try std.testing.expectError(error.NonMatrixArray, rectangular.symmetryRelativeResidualFrobeniusNorm());
    try std.testing.expectError(error.NonMatrixArray, rectangular.symmetryResidualFrobeniusNormMeetsBound(1));
    try std.testing.expectError(error.NonMatrixArray, rectangular.profileMeetsBounds(1, 1));
    try std.testing.expectError(error.NonMatrixArray, rectangular.profileTotalMeetsBound(1));

    var duplicate = try csrFromCompressed(f64, gpa, 2, 2, &.{ 0, 3, 5 }, &.{ 0, 0, 1, 0, 1 }, &.{ 1.0, -1.0, 2.0, 2.0, 0.0 });
    defer duplicate.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0), duplicate.get(0, 0).?, 1e-12);
    var duplicate_diag = try duplicate.diagonal();
    defer duplicate_diag.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 0 }, duplicate_diag.data);
    try std.testing.expectApproxEqAbs(@as(f64, 0), try duplicate.trace(), 1e-12);
    try std.testing.expect(try duplicate.traceInRange(0, 0));
    try std.testing.expectApproxEqAbs(@as(f64, 0), try duplicate.normalizedTrace(), 1e-12);
    try std.testing.expect(try duplicate.normalizedTraceInRange(0, 0));
    try std.testing.expectEqual(@as(usize, 0), try duplicate.missingDiagonalCount());
    try std.testing.expectEqual(@as(usize, 2), try duplicate.zeroDiagonalCount());
    try std.testing.expect(try duplicate.zeroDiagonalCountMeetsBound(2));
    try std.testing.expect(!(try duplicate.zeroDiagonalCountMeetsBound(1)));
    try std.testing.expect(try duplicate.zeroDiagonalCountInRange(2, 2));
    try std.testing.expect(!(try duplicate.zeroDiagonalCountInRange(0, 1)));
    try std.testing.expectEqual(@as(usize, 1), try duplicate.bandwidth());
    try std.testing.expect(try duplicate.structurallySymmetric());
    try std.testing.expect(try duplicate.numericallySymmetric(1e-12));

    var duplicate_coalesced = try duplicate.coalesced();
    defer duplicate_coalesced.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 4 }, duplicate_coalesced.row_offsets);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 0, 1 }, duplicate_coalesced.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 0, 2, 2, 0 }, duplicate_coalesced.values);
    var duplicate_coalesced_dense = try duplicate_coalesced.toDense();
    defer duplicate_coalesced_dense.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 2, 2, 0 }, duplicate_coalesced_dense.data);

    var empty_square = try csrFromCompressed(f64, gpa, 0, 0, &.{0}, &.{}, &.{});
    defer empty_square.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0), try empty_square.trace(), 1e-12);
    try std.testing.expectError(error.EmptyArray, empty_square.normalizedTrace());
    try std.testing.expectError(error.EmptyArray, empty_square.normalizedTraceInRange(0, 1));
}

test "csr sparse transpose products and triangular solves" {
    const gpa = std.testing.allocator;
    var dense = try array_mod.Array(f64).fromSlice(gpa, &.{
        1, 0, 2,
        0, 3, 0,
    }, &.{ 2, 3 });
    defer dense.deinit();
    var csr = try csrFromDense(f64, dense);
    defer csr.deinit();

    var x = try array_mod.Array(f64).fromSlice(gpa, &.{ 4, 5 }, &.{2});
    defer x.deinit();
    var tx = try csr.transposeMatvec(x);
    defer tx.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 4, 15, 8 }, tx.data);

    var rhs = try array_mod.Array(f64).fromSlice(gpa, &.{
        1, 2,
        3, 4,
    }, &.{ 2, 2 });
    defer rhs.deinit();
    var tm = try csr.transposeMatmat(rhs);
    defer tm.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, tm.shape);
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 9, 12, 2, 4 }, tm.data);

    var lower_dense = try array_mod.Array(f64).fromSlice(gpa, &.{
        2,  0, 0,
        -1, 3, 0,
        4,  2, 5,
    }, &.{ 3, 3 });
    defer lower_dense.deinit();
    var lower = try csrFromDense(f64, lower_dense);
    defer lower.deinit();
    var lower_rhs = try array_mod.Array(f64).fromSlice(gpa, &.{ 2, 2, 25 }, &.{3});
    defer lower_rhs.deinit();
    var solved = try lower.solveTriangular(lower_rhs, .lower, .non_unit);
    defer solved.deinit();
    var check = try lower.matvec(solved);
    defer check.deinit();
    try std.testing.expect(try check.allclose(lower_rhs, 1e-12, 1e-12));

    var lower_rhs_m = try array_mod.Array(f64).fromSlice(gpa, &.{ 2, 4, 2, 4, 25, 50 }, &.{ 3, 2 });
    defer lower_rhs_m.deinit();
    var solved_m = try lower.solveTriangular(lower_rhs_m, .lower, .non_unit);
    defer solved_m.deinit();
    var check_m = try lower.matmat(solved_m);
    defer check_m.deinit();
    try std.testing.expect(try check_m.allclose(lower_rhs_m, 1e-12, 1e-12));
}

test "csc sparse bridge dense roundtrip matvec matmat and csr transpose" {
    const gpa = std.testing.allocator;
    var dense = try array_mod.Array(f64).fromSlice(gpa, &.{
        10, 0, 2, 0,
        0,  3, 0, 4,
        5,  0, 0, 6,
    }, &.{ 3, 4 });
    defer dense.deinit();
    var csc = try cscFromDense(f64, dense);
    defer csc.deinit();
    try std.testing.expectEqual(@as(usize, 6), csc.nnz());
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 3, 4, 6 }, csc.col_offsets);
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 1, 0, 1, 2 }, csc.row_indices);
    try std.testing.expectEqualSlices(f64, &.{ 10, 5, 3, 2, 4, 6 }, csc.values);

    var dense2 = try csc.toDense();
    defer dense2.deinit();
    try std.testing.expectEqualSlices(f64, dense.data, dense2.data);

    var x = try array_mod.Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4 }, &.{4});
    defer x.deinit();
    var y = try csc.matvec(x);
    defer y.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 16, 22, 29 }, y.data);

    var rhs = try array_mod.Array(f64).fromSlice(gpa, &.{
        1, 2,
        2, 4,
        3, 6,
        4, 8,
    }, &.{ 4, 2 });
    defer rhs.deinit();
    var product = try csc.matmat(rhs);
    defer product.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 16, 32, 22, 44, 29, 58 }, product.data);

    var csr = try csc.toCsr();
    defer csr.deinit();
    var csr_dense = try csr.toDense();
    defer csr_dense.deinit();
    try std.testing.expectEqualSlices(f64, dense.data, csr_dense.data);
    try std.testing.expectApproxEqAbs(@as(f64, 30), csc.sum(), 1e-12);
    try std.testing.expect(try csc.sumInRange(30, 30));
    try std.testing.expect(try csc.sumInRange(29.5, 30.5));
    try std.testing.expect(!(try csc.sumInRange(30.5, 31)));
    try std.testing.expectError(error.InvalidShape, csc.sumInRange(31, 30));
    try std.testing.expectApproxEqAbs(@as(f64, 30), csc.absSum(), 1e-12);
    try std.testing.expect(try csc.absSumInRange(30, 30));
    try std.testing.expect(try csc.absSumInRange(29, 31));
    try std.testing.expect(!(try csc.absSumInRange(31, 32)));
    try std.testing.expectError(error.InvalidShape, csc.absSumInRange(std.math.inf(f64), 31));
    try std.testing.expectApproxEqAbs(@as(f64, @sqrt(190.0)), csc.frobeniusNorm(), 1e-12);
    try std.testing.expect(try csc.frobeniusNormMeetsBound(@sqrt(@as(f64, 190.0))));
    try std.testing.expect(!(try csc.frobeniusNormMeetsBound(@sqrt(@as(f64, 190.0)) - 1e-12)));
    try std.testing.expectError(error.InvalidShape, csc.frobeniusNormMeetsBound(std.math.nan(f64)));
    try std.testing.expectApproxEqAbs(@as(f64, 15), try csc.oneNorm(), 1e-12);
    try std.testing.expect(try csc.oneNormMeetsBound(15));
    try std.testing.expect(!(try csc.oneNormMeetsBound(14.999)));
    try std.testing.expectError(error.InvalidShape, csc.oneNormMeetsBound(-1));
    try std.testing.expectApproxEqAbs(@as(f64, 12), try csc.infNorm(), 1e-12);
    try std.testing.expect(try csc.infNormMeetsBound(12));
    try std.testing.expect(!(try csc.infNormMeetsBound(11.999)));
    try std.testing.expectError(error.InvalidShape, csc.infNormMeetsBound(std.math.inf(f64)));
}

test "sparse matvec residual diagnostics" {
    const gpa = std.testing.allocator;
    var dense = try array_mod.Array(f64).fromSlice(gpa, &.{
        1, 0, 2,
        0, 3, 0,
    }, &.{ 2, 3 });
    defer dense.deinit();
    var coo = try cooFromDense(f64, dense);
    defer coo.deinit();
    var csr = try coo.toCsr();
    defer csr.deinit();
    var csc = try coo.toCsc();
    defer csc.deinit();

    var x = try array_mod.Array(f64).fromSlice(gpa, &.{ 1, 2, 3 }, &.{3});
    defer x.deinit();
    var exact = try array_mod.Array(f64).fromSlice(gpa, &.{ 7, 6 }, &.{2});
    defer exact.deinit();
    var perturbed = try array_mod.Array(f64).fromSlice(gpa, &.{ 7, 8 }, &.{2});
    defer perturbed.deinit();
    var short_rhs = try array_mod.Array(f64).fromSlice(gpa, &.{7}, &.{1});
    defer short_rhs.deinit();
    const relative = @as(f64, 2) / (@as(f64, 14) + @sqrt(@as(f64, 113)));

    try std.testing.expectApproxEqAbs(@as(f64, 0), try coo.matvecResidualNorm(x, exact), 1e-12);
    try std.testing.expect(try coo.matvecResidualNormMeetsBound(x, exact, 0));
    try std.testing.expectApproxEqAbs(@as(f64, 2), try coo.matvecResidualNorm(x, perturbed), 1e-12);
    try std.testing.expect(try coo.matvecResidualNormMeetsBound(x, perturbed, 2));
    try std.testing.expect(!(try coo.matvecResidualNormMeetsBound(x, perturbed, 1.999)));
    try std.testing.expectApproxEqAbs(relative, try coo.matvecRelativeResidualNorm(x, perturbed), 1e-12);
    try std.testing.expect(try coo.matvecRelativeResidualNormMeetsBound(x, perturbed, relative + 1e-12));
    try std.testing.expect(!(try coo.matvecRelativeResidualNormMeetsBound(x, perturbed, relative * 0.5)));
    const coo_summary = try coo.matvecResidualSummary(x, perturbed);
    try std.testing.expectApproxEqAbs(@as(f64, 2), coo_summary.residual_norm, 1e-12);
    try std.testing.expectApproxEqAbs(relative, coo_summary.relative_residual_norm, 1e-12);
    try std.testing.expect(try coo_summary.residualNormMeetsBound(2));
    try std.testing.expect(try coo_summary.relativeResidualNormMeetsBound(relative + 1e-12));
    try std.testing.expect(try coo_summary.meetsBounds(2, relative + 1e-12));
    try std.testing.expect(!(try coo_summary.meetsBounds(1.999, relative + 1e-12)));
    try std.testing.expect(try coo.matvecResidualSummaryMeetsBounds(x, perturbed, 2, relative + 1e-12));
    try std.testing.expect(!(try coo.matvecResidualSummaryMeetsBounds(x, perturbed, 1.999, relative + 1e-12)));
    try std.testing.expectError(error.InvalidShape, coo.matvecResidualNormMeetsBound(x, perturbed, -1));
    try std.testing.expectError(error.ShapeMismatch, coo.matvecResidualNorm(x, short_rhs));

    try std.testing.expectApproxEqAbs(@as(f64, 2), try csr.matvecResidualNorm(x, perturbed), 1e-12);
    try std.testing.expect(try csr.matvecResidualNormMeetsBound(x, perturbed, 2));
    try std.testing.expect(!(try csr.matvecResidualNormMeetsBound(x, perturbed, 1.999)));
    try std.testing.expectApproxEqAbs(relative, try csr.matvecRelativeResidualNorm(x, perturbed), 1e-12);
    try std.testing.expect(try csr.matvecRelativeResidualNormMeetsBound(x, perturbed, relative + 1e-12));
    try std.testing.expect(!(try csr.matvecRelativeResidualNormMeetsBound(x, perturbed, relative * 0.5)));

    try std.testing.expectApproxEqAbs(@as(f64, 2), try csc.matvecResidualNorm(x, perturbed), 1e-12);
    try std.testing.expect(try csc.matvecResidualNormMeetsBound(x, perturbed, 2));
    try std.testing.expect(!(try csc.matvecResidualNormMeetsBound(x, perturbed, 1.999)));
    try std.testing.expectApproxEqAbs(relative, try csc.matvecRelativeResidualNorm(x, perturbed), 1e-12);
    try std.testing.expect(try csc.matvecRelativeResidualNormMeetsBound(x, perturbed, relative + 1e-12));
    try std.testing.expect(!(try csc.matvecRelativeResidualNormMeetsBound(x, perturbed, relative * 0.5)));

    var tx = try array_mod.Array(f64).fromSlice(gpa, &.{ 4, 5 }, &.{2});
    defer tx.deinit();
    var exact_t = try array_mod.Array(f64).fromSlice(gpa, &.{ 4, 15, 8 }, &.{3});
    defer exact_t.deinit();
    var perturbed_t = try array_mod.Array(f64).fromSlice(gpa, &.{ 4, 16, 8 }, &.{3});
    defer perturbed_t.deinit();
    const transpose_relative = @as(f64, 1) / (@sqrt(@as(f64, 14)) * @sqrt(@as(f64, 41)) + @sqrt(@as(f64, 336)));

    try std.testing.expectApproxEqAbs(@as(f64, 0), try coo.transposeMatvecResidualNorm(tx, exact_t), 1e-12);
    try std.testing.expect(try coo.transposeMatvecResidualNormMeetsBound(tx, exact_t, 0));
    try std.testing.expectApproxEqAbs(@as(f64, 1), try coo.transposeMatvecResidualNorm(tx, perturbed_t), 1e-12);
    try std.testing.expect(try coo.transposeMatvecResidualNormMeetsBound(tx, perturbed_t, 1));
    try std.testing.expect(!(try coo.transposeMatvecResidualNormMeetsBound(tx, perturbed_t, 0.999)));
    try std.testing.expectApproxEqAbs(transpose_relative, try coo.transposeMatvecRelativeResidualNorm(tx, perturbed_t), 1e-12);
    try std.testing.expect(try coo.transposeMatvecRelativeResidualNormMeetsBound(tx, perturbed_t, transpose_relative + 1e-12));
    try std.testing.expect(!(try coo.transposeMatvecRelativeResidualNormMeetsBound(tx, perturbed_t, transpose_relative * 0.5)));

    try std.testing.expectApproxEqAbs(@as(f64, 1), try csr.transposeMatvecResidualNorm(tx, perturbed_t), 1e-12);
    try std.testing.expect(try csr.transposeMatvecResidualNormMeetsBound(tx, perturbed_t, 1));
    try std.testing.expectApproxEqAbs(transpose_relative, try csr.transposeMatvecRelativeResidualNorm(tx, perturbed_t), 1e-12);
    try std.testing.expect(try csr.transposeMatvecRelativeResidualNormMeetsBound(tx, perturbed_t, transpose_relative + 1e-12));

    try std.testing.expectApproxEqAbs(@as(f64, 1), try csc.transposeMatvecResidualNorm(tx, perturbed_t), 1e-12);
    try std.testing.expect(try csc.transposeMatvecResidualNormMeetsBound(tx, perturbed_t, 1));
    try std.testing.expectApproxEqAbs(transpose_relative, try csc.transposeMatvecRelativeResidualNorm(tx, perturbed_t), 1e-12);
    try std.testing.expect(try csc.transposeMatvecRelativeResidualNormMeetsBound(tx, perturbed_t, transpose_relative + 1e-12));

    var matrix_x = try array_mod.Array(f64).fromSlice(gpa, &.{
        1, 2,
        3, 4,
        5, 6,
    }, &.{ 3, 2 });
    defer matrix_x.deinit();
    var matrix_exact = try array_mod.Array(f64).fromSlice(gpa, &.{
        11, 14,
        9,  12,
    }, &.{ 2, 2 });
    defer matrix_exact.deinit();
    var matrix_perturbed = try array_mod.Array(f64).fromSlice(gpa, &.{
        12, 14,
        9,  14,
    }, &.{ 2, 2 });
    defer matrix_perturbed.deinit();
    const matrix_residual = @sqrt(@as(f64, 5));
    const matrix_relative = matrix_residual / (@sqrt(@as(f64, 14)) * @sqrt(@as(f64, 91)) + @sqrt(@as(f64, 617)));

    try std.testing.expectApproxEqAbs(@as(f64, 0), try coo.matmatResidualFrobeniusNorm(matrix_x, matrix_exact), 1e-12);
    try std.testing.expect(try coo.matmatResidualFrobeniusNormMeetsBound(matrix_x, matrix_exact, 0));
    try std.testing.expectApproxEqAbs(matrix_residual, try coo.matmatResidualFrobeniusNorm(matrix_x, matrix_perturbed), 1e-12);
    try std.testing.expect(try coo.matmatResidualFrobeniusNormMeetsBound(matrix_x, matrix_perturbed, matrix_residual + 1e-12));
    try std.testing.expect(!(try coo.matmatResidualFrobeniusNormMeetsBound(matrix_x, matrix_perturbed, matrix_residual * 0.999)));
    try std.testing.expectApproxEqAbs(matrix_relative, try coo.matmatRelativeResidualFrobeniusNorm(matrix_x, matrix_perturbed), 1e-12);
    try std.testing.expect(try coo.matmatRelativeResidualFrobeniusNormMeetsBound(matrix_x, matrix_perturbed, matrix_relative + 1e-12));
    try std.testing.expect(!(try coo.matmatRelativeResidualFrobeniusNormMeetsBound(matrix_x, matrix_perturbed, matrix_relative * 0.5)));
    const matrix_summary = try coo.matmatResidualSummary(matrix_x, matrix_perturbed);
    try std.testing.expectApproxEqAbs(matrix_residual, matrix_summary.residual_norm, 1e-12);
    try std.testing.expectApproxEqAbs(matrix_relative, matrix_summary.relative_residual_norm, 1e-12);
    try std.testing.expect(try matrix_summary.meetsBounds(matrix_residual + 1e-12, matrix_relative + 1e-12));
    try std.testing.expect(try coo.matmatResidualSummaryMeetsBounds(matrix_x, matrix_perturbed, matrix_residual + 1e-12, matrix_relative + 1e-12));
    try std.testing.expect(!(try coo.matmatResidualSummaryMeetsBounds(matrix_x, matrix_perturbed, matrix_residual * 0.999, matrix_relative + 1e-12)));
    try std.testing.expectError(error.InvalidShape, coo.matmatResidualFrobeniusNormMeetsBound(matrix_x, matrix_perturbed, -1));

    try std.testing.expectApproxEqAbs(matrix_residual, try csr.matmatResidualFrobeniusNorm(matrix_x, matrix_perturbed), 1e-12);
    try std.testing.expect(try csr.matmatResidualFrobeniusNormMeetsBound(matrix_x, matrix_perturbed, matrix_residual + 1e-12));
    try std.testing.expect(!(try csr.matmatResidualFrobeniusNormMeetsBound(matrix_x, matrix_perturbed, matrix_residual * 0.999)));
    try std.testing.expectApproxEqAbs(matrix_relative, try csr.matmatRelativeResidualFrobeniusNorm(matrix_x, matrix_perturbed), 1e-12);
    try std.testing.expect(try csr.matmatRelativeResidualFrobeniusNormMeetsBound(matrix_x, matrix_perturbed, matrix_relative + 1e-12));

    try std.testing.expectApproxEqAbs(matrix_residual, try csc.matmatResidualFrobeniusNorm(matrix_x, matrix_perturbed), 1e-12);
    try std.testing.expect(try csc.matmatResidualFrobeniusNormMeetsBound(matrix_x, matrix_perturbed, matrix_residual + 1e-12));
    try std.testing.expect(!(try csc.matmatResidualFrobeniusNormMeetsBound(matrix_x, matrix_perturbed, matrix_residual * 0.999)));
    try std.testing.expectApproxEqAbs(matrix_relative, try csc.matmatRelativeResidualFrobeniusNorm(matrix_x, matrix_perturbed), 1e-12);
    try std.testing.expect(try csc.matmatRelativeResidualFrobeniusNormMeetsBound(matrix_x, matrix_perturbed, matrix_relative + 1e-12));

    var transpose_matrix_x = try array_mod.Array(f64).fromSlice(gpa, &.{
        1, 2,
        3, 4,
    }, &.{ 2, 2 });
    defer transpose_matrix_x.deinit();
    var transpose_matrix_exact = try array_mod.Array(f64).fromSlice(gpa, &.{
        1, 2,
        9, 12,
        2, 4,
    }, &.{ 3, 2 });
    defer transpose_matrix_exact.deinit();
    var transpose_matrix_perturbed = try array_mod.Array(f64).fromSlice(gpa, &.{
        2, 2,
        9, 10,
        2, 4,
    }, &.{ 3, 2 });
    defer transpose_matrix_perturbed.deinit();
    const transpose_matrix_residual = @sqrt(@as(f64, 5));
    const transpose_matrix_relative = transpose_matrix_residual / (@sqrt(@as(f64, 14)) * @sqrt(@as(f64, 30)) + @sqrt(@as(f64, 209)));

    try std.testing.expectApproxEqAbs(@as(f64, 0), try coo.transposeMatmatResidualFrobeniusNorm(transpose_matrix_x, transpose_matrix_exact), 1e-12);
    try std.testing.expect(try coo.transposeMatmatResidualFrobeniusNormMeetsBound(transpose_matrix_x, transpose_matrix_exact, 0));
    try std.testing.expectApproxEqAbs(transpose_matrix_residual, try coo.transposeMatmatResidualFrobeniusNorm(transpose_matrix_x, transpose_matrix_perturbed), 1e-12);
    try std.testing.expect(try coo.transposeMatmatResidualFrobeniusNormMeetsBound(transpose_matrix_x, transpose_matrix_perturbed, transpose_matrix_residual + 1e-12));
    try std.testing.expect(!(try coo.transposeMatmatResidualFrobeniusNormMeetsBound(transpose_matrix_x, transpose_matrix_perturbed, transpose_matrix_residual * 0.999)));
    try std.testing.expectApproxEqAbs(transpose_matrix_relative, try coo.transposeMatmatRelativeResidualFrobeniusNorm(transpose_matrix_x, transpose_matrix_perturbed), 1e-12);
    try std.testing.expect(try coo.transposeMatmatRelativeResidualFrobeniusNormMeetsBound(transpose_matrix_x, transpose_matrix_perturbed, transpose_matrix_relative + 1e-12));
    try std.testing.expect(!(try coo.transposeMatmatRelativeResidualFrobeniusNormMeetsBound(transpose_matrix_x, transpose_matrix_perturbed, transpose_matrix_relative * 0.5)));
    const transpose_matrix_summary = try coo.transposeMatmatResidualSummary(transpose_matrix_x, transpose_matrix_perturbed);
    try std.testing.expectApproxEqAbs(transpose_matrix_residual, transpose_matrix_summary.residual_norm, 1e-12);
    try std.testing.expectApproxEqAbs(transpose_matrix_relative, transpose_matrix_summary.relative_residual_norm, 1e-12);
    try std.testing.expect(try coo.transposeMatmatResidualSummaryMeetsBounds(transpose_matrix_x, transpose_matrix_perturbed, transpose_matrix_residual + 1e-12, transpose_matrix_relative + 1e-12));
    try std.testing.expect(!(try coo.transposeMatmatResidualSummaryMeetsBounds(transpose_matrix_x, transpose_matrix_perturbed, transpose_matrix_residual * 0.999, transpose_matrix_relative + 1e-12)));

    try std.testing.expectApproxEqAbs(transpose_matrix_residual, try csr.transposeMatmatResidualFrobeniusNorm(transpose_matrix_x, transpose_matrix_perturbed), 1e-12);
    try std.testing.expect(try csr.transposeMatmatResidualFrobeniusNormMeetsBound(transpose_matrix_x, transpose_matrix_perturbed, transpose_matrix_residual + 1e-12));
    try std.testing.expectApproxEqAbs(transpose_matrix_relative, try csr.transposeMatmatRelativeResidualFrobeniusNorm(transpose_matrix_x, transpose_matrix_perturbed), 1e-12);
    try std.testing.expect(try csr.transposeMatmatRelativeResidualFrobeniusNormMeetsBound(transpose_matrix_x, transpose_matrix_perturbed, transpose_matrix_relative + 1e-12));

    try std.testing.expectApproxEqAbs(transpose_matrix_residual, try csc.transposeMatmatResidualFrobeniusNorm(transpose_matrix_x, transpose_matrix_perturbed), 1e-12);
    try std.testing.expect(try csc.transposeMatmatResidualFrobeniusNormMeetsBound(transpose_matrix_x, transpose_matrix_perturbed, transpose_matrix_residual + 1e-12));
    try std.testing.expectApproxEqAbs(transpose_matrix_relative, try csc.transposeMatmatRelativeResidualFrobeniusNorm(transpose_matrix_x, transpose_matrix_perturbed), 1e-12);
    try std.testing.expect(try csc.transposeMatmatRelativeResidualFrobeniusNormMeetsBound(transpose_matrix_x, transpose_matrix_perturbed, transpose_matrix_relative + 1e-12));
}

test "csc sparse transpose products and row column stats" {
    const gpa = std.testing.allocator;
    var dense = try array_mod.Array(f64).fromSlice(gpa, &.{
        1, 0, -2,
        0, 3, 0,
        4, 0, 5,
    }, &.{ 3, 3 });
    defer dense.deinit();
    var csc = try cscFromDense(f64, dense);
    defer csc.deinit();

    var x = try array_mod.Array(f64).fromSlice(gpa, &.{ 1, 2, 3 }, &.{3});
    defer x.deinit();
    var tx = try csc.transposeMatvec(x);
    defer tx.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 13, 6, 13 }, tx.data);

    var rhs = try array_mod.Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 3, 2 });
    defer rhs.deinit();
    var tm = try csc.transposeMatmat(rhs);
    defer tm.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 21, 26, 9, 12, 23, 26 }, tm.data);

    try std.testing.expectApproxEqAbs(@as(f64, -2), try csc.minValue(), 1e-12);
    try std.testing.expectEqual(@as(usize, 3), try csc.minValueIndex());
    try std.testing.expectApproxEqAbs(@as(f64, 5), try csc.maxValue(), 1e-12);
    try std.testing.expectEqual(@as(usize, 4), try csc.maxValueIndex());
    try std.testing.expectApproxEqAbs(@as(f64, 1), try csc.minAbsValue(), 1e-12);
    try std.testing.expectEqual(@as(usize, 0), try csc.minAbsValueIndex());
    try std.testing.expectApproxEqAbs(@as(f64, 5), try csc.maxAbsValue(), 1e-12);
    try std.testing.expectEqual(@as(usize, 4), try csc.maxAbsValueIndex());
    const expected_mean = @as(f64, 11.0 / 9.0);
    const expected_variance = @as(f64, 55.0 / 9.0) - expected_mean * expected_mean;
    const expected_stddev = @sqrt(expected_variance);
    const expected_sample_variance = @as(f64, (55.0 - (11.0 * 11.0) / 9.0) / 8.0);
    const expected_sample_stddev = @sqrt(expected_sample_variance);
    try std.testing.expectApproxEqAbs(expected_mean, try csc.mean(), 1e-12);
    try std.testing.expect(try csc.meanInRange(expected_mean, expected_mean));
    try std.testing.expect(try csc.meanInRange(1.2, 1.3));
    try std.testing.expect(!(try csc.meanInRange(1.3, 1.4)));
    try std.testing.expectError(error.InvalidShape, csc.meanInRange(2, 1));
    try std.testing.expectApproxEqAbs(expected_variance, try csc.variance(0), 1e-12);
    try std.testing.expect(try csc.varianceInRange(0, expected_variance - 1e-12, expected_variance + 1e-12));
    try std.testing.expect(try csc.varianceInRange(0, 4.6, 4.7));
    try std.testing.expect(!(try csc.varianceInRange(0, 4.7, 4.8)));
    try std.testing.expectError(error.InvalidShape, csc.varianceInRange(0, -0.1, 1));
    try std.testing.expectApproxEqAbs(expected_stddev, try csc.stddev(0), 1e-12);
    try std.testing.expect(try csc.stddevInRange(0, expected_stddev - 1e-12, expected_stddev + 1e-12));
    try std.testing.expect(!(try csc.stddevInRange(0, expected_stddev + 0.1, expected_stddev + 0.2)));
    try std.testing.expectError(error.InvalidShape, csc.stddevInRange(0, -0.1, expected_stddev));
    try std.testing.expectApproxEqAbs(expected_sample_variance, try csc.sampleVariance(), 1e-12);
    try std.testing.expect(try csc.sampleVarianceInRange(expected_sample_variance - 1e-12, expected_sample_variance + 1e-12));
    try std.testing.expect(!(try csc.sampleVarianceInRange(expected_sample_variance + 0.1, expected_sample_variance + 0.2)));
    try std.testing.expectApproxEqAbs(expected_sample_stddev, try csc.sampleStddev(), 1e-12);
    try std.testing.expect(try csc.sampleStddevInRange(expected_sample_stddev - 1e-12, expected_sample_stddev + 1e-12));
    try std.testing.expectError(error.InvalidShape, csc.sampleStddevInRange(2, 1));

    var row_vars = try csc.rowVariances(0);
    defer row_vars.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 14.0 / 9.0), row_vars.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2), row_vars.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 14.0 / 3.0), row_vars.data[2], 1e-12);
    try std.testing.expect(try csc.rowVariancesInRange(0, 14.0 / 9.0, 14.0 / 3.0));
    try std.testing.expect(!(try csc.rowVariancesInRange(0, 2.1, 14.0 / 3.0)));
    try std.testing.expectError(error.InvalidShape, csc.rowVariancesInRange(0, -0.1, 1));
    var col_vars = try csc.columnVariances(0);
    defer col_vars.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 26.0 / 9.0), col_vars.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2), col_vars.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 26.0 / 3.0), col_vars.data[2], 1e-12);
    try std.testing.expect(try csc.columnVariancesInRange(0, 2, 26.0 / 3.0));
    try std.testing.expect(!(try csc.columnVariancesInRange(0, 2.1, 26.0 / 3.0)));
    try std.testing.expect(try csc.columnSampleVariancesInRange(3, 13));
    try std.testing.expect(!(try csc.columnSampleVariancesInRange(4, 13)));
    try std.testing.expect(try csc.columnSampleStddevsInRange(@sqrt(3.0), @sqrt(13.0)));
    try std.testing.expect(!(try csc.columnSampleStddevsInRange(1.8, @sqrt(13.0))));
    var col_stds = try csc.columnStddevs(0);
    defer col_stds.deinit();
    try std.testing.expectApproxEqAbs(@sqrt(26.0 / 9.0), col_stds.data[0], 1e-12);
    try std.testing.expect(try csc.columnStddevsInRange(0, @sqrt(2.0), @sqrt(26.0 / 3.0)));
    try std.testing.expect(!(try csc.columnStddevsInRange(0, 1.5, @sqrt(26.0 / 3.0))));
    var row_stds = try csc.rowStddevs(0);
    defer row_stds.deinit();
    try std.testing.expectApproxEqAbs(@sqrt(14.0 / 9.0), row_stds.data[0], 1e-12);
    try std.testing.expect(try csc.rowStddevsInRange(0, @sqrt(14.0 / 9.0), @sqrt(14.0 / 3.0)));
    try std.testing.expect(!(try csc.rowStddevsInRange(0, 1.5, 2)));
    try std.testing.expectError(error.InvalidShape, csc.rowStddevsInRange(0, -0.1, 1));
    var row_sample_vars = try csc.rowSampleVariances();
    defer row_sample_vars.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 7.0 / 3.0), row_sample_vars.data[0], 1e-12);
    try std.testing.expect(try csc.rowSampleVariancesInRange(7.0 / 3.0, 7));
    try std.testing.expect(!(try csc.rowSampleVariancesInRange(2.5, 7)));
    try std.testing.expect(try csc.rowSampleStddevsInRange(@sqrt(7.0 / 3.0), @sqrt(7.0)));
    try std.testing.expect(!(try csc.rowSampleStddevsInRange(1.6, @sqrt(7.0))));

    var row_means = try csc.rowMeans();
    defer row_means.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, -1.0 / 3.0), row_means.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), row_means.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3), row_means.data[2], 1e-12);
    try std.testing.expect(try csc.rowMeansInRange(-1.0 / 3.0, 3));
    try std.testing.expect(!(try csc.rowMeansInRange(0, 3)));
    try std.testing.expectError(error.InvalidShape, csc.rowMeansInRange(3, 2));
    var col_means = try csc.columnMeans();
    defer col_means.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 3.0), col_means.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), col_means.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), col_means.data[2], 1e-12);
    try std.testing.expect(try csc.columnMeansInRange(1, 5.0 / 3.0));
    try std.testing.expect(!(try csc.columnMeansInRange(1.1, 5.0 / 3.0)));

    var row_mins = try csc.rowMins();
    defer row_mins.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -2, 0, 0 }, row_mins.data);
    try std.testing.expect(try csc.rowMinsInRange(-2, 0));
    try std.testing.expect(!(try csc.rowMinsInRange(-1, 0)));
    try std.testing.expectError(error.InvalidShape, csc.rowMinsInRange(1, 0));
    var row_maxes = try csc.rowMaxes();
    defer row_maxes.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 5 }, row_maxes.data);
    try std.testing.expect(try csc.rowMaxesInRange(1, 5));
    try std.testing.expect(!(try csc.rowMaxesInRange(2, 5)));
    var col_mins = try csc.columnMins();
    defer col_mins.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 0, -2 }, col_mins.data);
    try std.testing.expect(try csc.columnMinsInRange(-2, 0));
    try std.testing.expect(!(try csc.columnMinsInRange(-1, 0)));
    var col_maxes = try csc.columnMaxes();
    defer col_maxes.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 4, 3, 5 }, col_maxes.data);
    try std.testing.expect(try csc.columnMaxesInRange(3, 5));
    try std.testing.expect(!(try csc.columnMaxesInRange(4, 5)));
    var row_min_abs = try csc.rowMinAbs();
    defer row_min_abs.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 4 }, row_min_abs.data);
    try std.testing.expect(try csc.rowMinAbsInRange(1, 4));
    try std.testing.expect(!(try csc.rowMinAbsInRange(2, 4)));
    try std.testing.expectError(error.InvalidShape, csc.rowMinAbsInRange(5, 4));
    var row_max_abs = try csc.rowMaxAbs();
    defer row_max_abs.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 2, 3, 5 }, row_max_abs.data);
    try std.testing.expect(try csc.rowMaxAbsInRange(2, 5));
    try std.testing.expect(!(try csc.rowMaxAbsInRange(3, 5)));
    var col_min_abs = try csc.columnMinAbs();
    defer col_min_abs.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 2 }, col_min_abs.data);
    try std.testing.expect(try csc.columnMinAbsInRange(1, 3));
    try std.testing.expect(!(try csc.columnMinAbsInRange(2, 3)));
    var col_max_abs = try csc.columnMaxAbs();
    defer col_max_abs.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 4, 3, 5 }, col_max_abs.data);
    try std.testing.expect(try csc.columnMaxAbsInRange(3, 5));
    try std.testing.expect(!(try csc.columnMaxAbsInRange(4, 5)));

    var row_nnz = try csc.rowNnz();
    defer row_nnz.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 1, 2 }, row_nnz.data);
    var col_nnz = try csc.columnNnz();
    defer col_nnz.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 1, 2 }, col_nnz.data);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 3.0), try csc.averageRowNnz(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 3.0), try csc.averageColumnNnz(), 1e-12);
    try std.testing.expect(try csc.averageRowNnzInRange(1.6, 1.7));
    try std.testing.expect(!(try csc.averageColumnNnzInRange(0, 1.6)));
    try std.testing.expectEqual(@as(usize, 0), try csc.emptyRowCount());
    try std.testing.expectEqual(@as(usize, 0), csc.emptyColumnCount());
    try std.testing.expectApproxEqAbs(@as(f64, 0), try csc.emptyRowFraction(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0), try csc.emptyColumnFraction(), 1e-12);
    try std.testing.expect(try csc.emptyRowFractionInRange(0, 0));
    try std.testing.expect(try csc.emptyColumnFractionInRange(0, 0));
    var row_sums = try csc.rowSums();
    defer row_sums.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -1, 3, 9 }, row_sums.data);
    try std.testing.expect(try csc.rowSumsInRange(-1, 9));
    try std.testing.expect(!(try csc.rowSumsInRange(0, 9)));
    try std.testing.expectError(error.InvalidShape, csc.rowSumsInRange(10, 9));
    var col_sums = try csc.columnSums();
    defer col_sums.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 5, 3, 3 }, col_sums.data);
    try std.testing.expect(try csc.columnSumsInRange(3, 5));
    try std.testing.expect(!(try csc.columnSumsInRange(4, 5)));
    var row_abs = try csc.rowAbsSums();
    defer row_abs.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 3, 3, 9 }, row_abs.data);
    try std.testing.expect(try csc.rowAbsSumsInRange(3, 9));
    try std.testing.expect(!(try csc.rowAbsSumsInRange(4, 9)));
    try std.testing.expectError(error.InvalidShape, csc.rowAbsSumsInRange(10, 9));
    var col_abs = try csc.columnAbsSums();
    defer col_abs.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 5, 3, 7 }, col_abs.data);
    try std.testing.expect(try csc.columnAbsSumsInRange(3, 7));
    try std.testing.expect(!(try csc.columnAbsSumsInRange(4, 7)));
    var row_norms = try csc.rowNorms();
    defer row_norms.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, @sqrt(5.0)), row_norms.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3), row_norms.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, @sqrt(41.0)), row_norms.data[2], 1e-12);
    try std.testing.expect(try csc.rowNormsInRange(@sqrt(@as(f64, 5.0)), @sqrt(@as(f64, 41.0))));
    try std.testing.expect(!(try csc.rowNormsInRange(3.1, @sqrt(@as(f64, 41.0)))));
    try std.testing.expectError(error.InvalidShape, csc.rowNormsInRange(-0.1, 1));
    var col_norms = try csc.columnNorms();
    defer col_norms.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, @sqrt(17.0)), col_norms.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3), col_norms.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, @sqrt(29.0)), col_norms.data[2], 1e-12);
    try std.testing.expect(try csc.columnNormsInRange(3, @sqrt(@as(f64, 29.0))));
    try std.testing.expect(!(try csc.columnNormsInRange(3.1, @sqrt(@as(f64, 29.0)))));
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 9.0), try csc.density(), 1e-12);
    try std.testing.expect(try csc.densityInRange(5.0 / 9.0, 5.0 / 9.0));
    try std.testing.expect(!(try csc.densityInRange(0, 0.5)));
    try std.testing.expectError(error.InvalidShape, csc.densityInRange(-0.1, 0.5));
    try std.testing.expectApproxEqAbs(@as(f64, 15), csc.absSum(), 1e-12);
}

test "csc sparse diagnostics and triangular solve" {
    const gpa = std.testing.allocator;
    var symmetric_dense = try array_mod.Array(f64).fromSlice(gpa, &.{
        4, 1, 0,
        1, 5, 2,
        0, 2, 6,
    }, &.{ 3, 3 });
    defer symmetric_dense.deinit();
    var symmetric = try cscFromDense(f64, symmetric_dense);
    defer symmetric.deinit();
    var diag = try symmetric.diagonal();
    defer diag.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 4, 5, 6 }, diag.data);
    try std.testing.expectApproxEqAbs(@as(f64, 15), try symmetric.trace(), 1e-12);
    try std.testing.expect(try symmetric.traceInRange(15, 15));
    try std.testing.expect(try symmetric.traceInRange(14.5, 15.5));
    try std.testing.expect(!(try symmetric.traceInRange(15.5, 16)));
    try std.testing.expectError(error.InvalidShape, symmetric.traceInRange(16, 15));
    try std.testing.expectApproxEqAbs(@as(f64, 5), try symmetric.normalizedTrace(), 1e-12);
    try std.testing.expect(try symmetric.normalizedTraceInRange(5, 5));
    try std.testing.expect(try symmetric.normalizedTraceInRange(4.9, 5.1));
    try std.testing.expect(!(try symmetric.normalizedTraceInRange(5.1, 6)));
    try std.testing.expectError(error.InvalidShape, symmetric.normalizedTraceInRange(std.math.inf(f64), 5));
    try std.testing.expectEqual(@as(usize, 0), try symmetric.missingDiagonalCount());
    try std.testing.expect(try symmetric.missingDiagonalCountMeetsBound(0));
    try std.testing.expect(try symmetric.missingDiagonalCountInRange(0, 0));
    try std.testing.expectEqual(@as(usize, 0), try symmetric.zeroDiagonalCount());
    try std.testing.expect(try symmetric.zeroDiagonalCountMeetsBound(0));
    try std.testing.expect(try symmetric.zeroDiagonalCountInRange(0, 0));
    try std.testing.expectEqual(@as(usize, 1), try symmetric.bandwidth());
    try std.testing.expect(try symmetric.bandwidthMeetsBound(1));
    try std.testing.expect(!(try symmetric.bandwidthMeetsBound(0)));
    try std.testing.expectEqual(@as(usize, 2), try symmetric.columnIntersectionBandwidth());
    try std.testing.expect(try symmetric.columnIntersectionBandwidthMeetsBound(2));
    try std.testing.expect(!(try symmetric.columnIntersectionBandwidthMeetsBound(1)));
    try std.testing.expectEqual(@as(usize, 5), try symmetric.lowerNnz(false));
    try std.testing.expectEqual(@as(usize, 2), try symmetric.lowerNnz(true));
    try std.testing.expectEqual(@as(usize, 5), try symmetric.upperNnz(false));
    try std.testing.expectEqual(@as(usize, 2), try symmetric.upperNnz(true));
    try std.testing.expect(try symmetric.lowerNnzMeetsBound(false, 5));
    try std.testing.expect(!(try symmetric.lowerNnzMeetsBound(false, 4)));
    try std.testing.expect(try symmetric.upperNnzMeetsBound(true, 2));
    try std.testing.expect(!(try symmetric.upperNnzMeetsBound(true, 1)));
    try std.testing.expect(try symmetric.lowerNnzInRange(false, 5, 5));
    try std.testing.expect(try symmetric.upperNnzInRange(true, 0, 2));
    try std.testing.expect(!(try symmetric.lowerNnzInRange(true, 3, 5)));
    try std.testing.expectError(error.InvalidShape, symmetric.upperNnzInRange(false, 6, 5));
    try std.testing.expectEqual(@as(usize, 2), try symmetric.lowerProfile());
    try std.testing.expectEqual(@as(usize, 2), try symmetric.upperProfile());
    const symmetric_profile = try symmetric.profile();
    try std.testing.expectEqual(@as(usize, 2), symmetric_profile.lower);
    try std.testing.expectEqual(@as(usize, 2), symmetric_profile.upper);
    try std.testing.expectEqual(@as(usize, 4), try symmetric_profile.total());
    try std.testing.expect(symmetric_profile.meetsBounds(2, 2));
    try std.testing.expect(try symmetric_profile.totalMeetsBound(4));
    try std.testing.expect(try symmetric.lowerProfileMeetsBound(2));
    try std.testing.expect(!(try symmetric.lowerProfileMeetsBound(1)));
    try std.testing.expect(try symmetric.upperProfileMeetsBound(2));
    try std.testing.expect(!(try symmetric.upperProfileMeetsBound(1)));
    try std.testing.expect(try symmetric.profileMeetsBounds(2, 2));
    try std.testing.expect(!(try symmetric.profileMeetsBounds(1, 2)));
    try std.testing.expect(try symmetric.profileTotalMeetsBound(4));
    try std.testing.expect(!(try symmetric.profileTotalMeetsBound(3)));
    try std.testing.expect(try symmetric.structurallySymmetric());
    try std.testing.expect(try symmetric.numericallySymmetric(1e-12));
    try std.testing.expectApproxEqAbs(@as(f64, 0), try symmetric.symmetryResidualFrobeniusNorm(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0), try symmetric.symmetryRelativeResidualFrobeniusNorm(), 1e-12);
    try std.testing.expect(try symmetric.symmetryResidualFrobeniusNormMeetsBound(0));
    try std.testing.expect(try symmetric.symmetryRelativeResidualFrobeniusNormMeetsBound(0));

    var missing_dense = try array_mod.Array(f64).fromSlice(gpa, &.{
        1, 2, 0,
        0, 0, 3,
        0, 0, 4,
    }, &.{ 3, 3 });
    defer missing_dense.deinit();
    var missing = try cscFromDense(f64, missing_dense);
    defer missing.deinit();
    try std.testing.expectEqual(@as(usize, 1), try missing.missingDiagonalCount());
    try std.testing.expect(try missing.missingDiagonalCountMeetsBound(1));
    try std.testing.expect(!(try missing.missingDiagonalCountMeetsBound(0)));
    try std.testing.expect(try missing.missingDiagonalCountInRange(1, 2));
    try std.testing.expect(!(try missing.missingDiagonalCountInRange(0, 0)));
    try std.testing.expectEqual(@as(usize, 0), try missing.zeroDiagonalCount());
    try std.testing.expect(try missing.zeroDiagonalCountMeetsBound(0));
    try std.testing.expect(!(try missing.zeroDiagonalCountInRange(1, 1)));
    try std.testing.expectApproxEqAbs(@sqrt(@as(f64, 26)), try missing.symmetryResidualFrobeniusNorm(), 1e-12);
    try std.testing.expectApproxEqAbs(@sqrt(@as(f64, 26)) / @sqrt(@as(f64, 30)), try missing.symmetryRelativeResidualFrobeniusNorm(), 1e-12);
    try std.testing.expect(try missing.symmetryResidualFrobeniusNormMeetsBound(@sqrt(@as(f64, 26))));
    try std.testing.expect(!(try missing.symmetryResidualFrobeniusNormMeetsBound(@sqrt(@as(f64, 26)) - 1e-12)));
    try std.testing.expect(try missing.symmetryRelativeResidualFrobeniusNormMeetsBound(@sqrt(@as(f64, 26)) / @sqrt(@as(f64, 30)) + 1e-12));
    try std.testing.expectError(error.InvalidShape, missing.symmetryResidualFrobeniusNormMeetsBound(-1));

    var rectangular = try cscFromCompressed(f64, gpa, 2, 3, &.{ 0, 1, 1, 2 }, &.{ 0, 1 }, &.{ 1, 2 });
    defer rectangular.deinit();
    try std.testing.expectError(error.NonMatrixArray, rectangular.lowerNnz(false));
    try std.testing.expectError(error.NonMatrixArray, rectangular.upperNnz(false));
    try std.testing.expectError(error.NonMatrixArray, rectangular.missingDiagonalCountMeetsBound(0));
    try std.testing.expectError(error.NonMatrixArray, rectangular.zeroDiagonalCountInRange(0, 1));
    try std.testing.expectError(error.InvalidShape, rectangular.missingDiagonalCountInRange(2, 1));
    try std.testing.expectError(error.NonMatrixArray, rectangular.traceInRange(0, 3));
    try std.testing.expectError(error.NonMatrixArray, rectangular.normalizedTrace());
    try std.testing.expectError(error.NonMatrixArray, rectangular.normalizedTraceInRange(0, 1));
    try std.testing.expectError(error.NonMatrixArray, rectangular.bandwidthMeetsBound(1));
    try std.testing.expectError(error.NonMatrixArray, rectangular.lowerNnzMeetsBound(false, 1));
    try std.testing.expectError(error.NonMatrixArray, rectangular.upperNnzInRange(false, 0, 1));
    try std.testing.expectError(error.NonMatrixArray, rectangular.profile());
    try std.testing.expectError(error.NonMatrixArray, rectangular.symmetryResidualFrobeniusNorm());
    try std.testing.expectError(error.NonMatrixArray, rectangular.symmetryRelativeResidualFrobeniusNorm());
    try std.testing.expectError(error.NonMatrixArray, rectangular.symmetryResidualFrobeniusNormMeetsBound(1));
    try std.testing.expectError(error.NonMatrixArray, rectangular.profileMeetsBounds(1, 1));
    try std.testing.expectError(error.NonMatrixArray, rectangular.profileTotalMeetsBound(1));

    var duplicate = try cscFromCompressed(f64, gpa, 2, 2, &.{ 0, 3, 5 }, &.{ 0, 0, 1, 0, 1 }, &.{ 1.0, -1.0, 2.0, 2.0, 0.0 });
    defer duplicate.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0), duplicate.get(0, 0).?, 1e-12);
    var duplicate_diag = try duplicate.diagonal();
    defer duplicate_diag.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 0 }, duplicate_diag.data);
    try std.testing.expectApproxEqAbs(@as(f64, 0), try duplicate.trace(), 1e-12);
    try std.testing.expect(try duplicate.traceInRange(0, 0));
    try std.testing.expectApproxEqAbs(@as(f64, 0), try duplicate.normalizedTrace(), 1e-12);
    try std.testing.expect(try duplicate.normalizedTraceInRange(0, 0));
    try std.testing.expectEqual(@as(usize, 0), try duplicate.missingDiagonalCount());
    try std.testing.expectEqual(@as(usize, 2), try duplicate.zeroDiagonalCount());
    try std.testing.expect(try duplicate.zeroDiagonalCountMeetsBound(2));
    try std.testing.expect(!(try duplicate.zeroDiagonalCountMeetsBound(1)));
    try std.testing.expect(try duplicate.zeroDiagonalCountInRange(2, 2));
    try std.testing.expect(!(try duplicate.zeroDiagonalCountInRange(0, 1)));
    try std.testing.expectEqual(@as(usize, 1), try duplicate.bandwidth());
    try std.testing.expect(try duplicate.structurallySymmetric());
    try std.testing.expect(try duplicate.numericallySymmetric(1e-12));

    var duplicate_coalesced = try duplicate.coalesced();
    defer duplicate_coalesced.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 4 }, duplicate_coalesced.col_offsets);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 0, 1 }, duplicate_coalesced.row_indices);
    try std.testing.expectEqualSlices(f64, &.{ 0, 2, 2, 0 }, duplicate_coalesced.values);
    var duplicate_coalesced_dense = try duplicate_coalesced.toDense();
    defer duplicate_coalesced_dense.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 2, 2, 0 }, duplicate_coalesced_dense.data);

    var empty_square = try cscFromCompressed(f64, gpa, 0, 0, &.{0}, &.{}, &.{});
    defer empty_square.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0), try empty_square.trace(), 1e-12);
    try std.testing.expectError(error.EmptyArray, empty_square.normalizedTrace());
    try std.testing.expectError(error.EmptyArray, empty_square.normalizedTraceInRange(0, 1));

    var lower_dense = try array_mod.Array(f64).fromSlice(gpa, &.{
        2,  0, 0,
        -1, 3, 0,
        4,  2, 5,
    }, &.{ 3, 3 });
    defer lower_dense.deinit();
    var lower = try cscFromDense(f64, lower_dense);
    defer lower.deinit();
    var rhs = try array_mod.Array(f64).fromSlice(gpa, &.{ 2, 2, 25 }, &.{3});
    defer rhs.deinit();
    var x = try lower.solveTriangular(rhs, .lower, .non_unit);
    defer x.deinit();
    var check = try lower.matvec(x);
    defer check.deinit();
    try std.testing.expect(try check.allclose(rhs, 1e-12, 1e-12));

    var rhs_m = try array_mod.Array(f64).fromSlice(gpa, &.{ 2, 4, 2, 4, 25, 50 }, &.{ 3, 2 });
    defer rhs_m.deinit();
    var xm = try lower.solveTriangular(rhs_m, .lower, .non_unit);
    defer xm.deinit();
    var check_m = try lower.matmat(xm);
    defer check_m.deinit();
    try std.testing.expect(try check_m.allclose(rhs_m, 1e-12, 1e-12));
}
