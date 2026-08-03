const std = @import("std");
const veyra = @import("veyra");
const array_mod = @import("array.zig");

pub const SparseError = array_mod.ArrayError || error{BackendFailure} || std.mem.Allocator.Error;

pub const Triangle = enum { lower, upper };
pub const Diagonal = enum { non_unit, unit };

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

fn maxStoredValue(comptime T: type, values: []const T) SparseError!T {
    ensureNumeric(T);
    if (values.len == 0) return error.EmptyArray;
    var result = values[0];
    for (values[1..]) |value| {
        if (value > result) result = value;
    }
    return result;
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

fn sparseSizeToF64(value: usize) f64 {
    return @floatFromInt(value);
}

fn sparseElementCount(rows: usize, cols: usize) SparseError!usize {
    return std.math.mul(usize, rows, cols) catch return error.InvalidShape;
}

fn sparseValueSquareToF64(comptime T: type, value: T) f64 {
    const numeric = sparseValueToF64(T, value);
    return numeric * numeric;
}

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
            var nonzero_count: usize = 0;
            for (diagonal_values) |value| {
                if (isNonZero(T, value)) nonzero_count += 1;
            }

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
            var nonzero_count: usize = 0;
            for (input.data) |value| {
                if (isNonZero(T, value)) nonzero_count += 1;
            }

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

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.row_indices);
            self.allocator.free(self.col_indices);
            self.allocator.free(self.values);
            self.* = undefined;
        }

        pub fn nnz(self: Self) usize {
            return self.values.len;
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

        pub fn dropZeros(self: Self) SparseError!Self {
            var nonzero_count: usize = 0;
            for (self.values) |value| {
                if (isNonZero(T, value)) nonzero_count += 1;
            }

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

        pub fn minValue(self: Self) SparseError!T {
            return minStoredValue(T, self.values);
        }

        pub fn maxValue(self: Self) SparseError!T {
            return maxStoredValue(T, self.values);
        }

        pub fn minAbsValue(self: Self) SparseError!T {
            return minStoredAbsValue(T, self.values);
        }

        pub fn maxAbsValue(self: Self) SparseError!T {
            return maxStoredAbsValue(T, self.values);
        }

        pub fn mean(self: Self) SparseError!f64 {
            ensureNumeric(T);
            const count = try sparseElementCount(self.rows, self.cols);
            if (count == 0) return error.EmptyArray;
            var total: f64 = 0;
            for (self.values) |value| total += sparseValueToF64(T, value);
            return total / sparseSizeToF64(count);
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

        pub fn stddev(self: Self, correction: f64) SparseError!f64 {
            return @sqrt(try self.variance(correction));
        }

        pub fn sampleVariance(self: Self) SparseError!f64 {
            return self.variance(1);
        }

        pub fn sampleStddev(self: Self) SparseError!f64 {
            return self.stddev(1);
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

        pub fn density(self: Self) SparseError!f64 {
            const total = self.rows * self.cols;
            if (total == 0) return 0;
            return @as(f64, @floatFromInt(self.values.len)) / @as(f64, @floatFromInt(total));
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

        pub fn trace(self: Self) SparseError!T {
            ensureNumeric(T);
            if (self.rows != self.cols) return error.NonMatrixArray;
            var total = zero(T);
            for (self.values, 0..) |value, i| {
                if (self.row_indices[i] == self.col_indices[i]) total += value;
            }
            return total;
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

        pub fn bandwidth(self: Self) SparseError!usize {
            if (self.rows != self.cols) return error.NonMatrixArray;
            var bw: usize = 0;
            for (self.row_indices, self.col_indices) |row, col| {
                const distance = if (row > col) row - col else col - row;
                if (distance > bw) bw = distance;
            }
            return bw;
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
            var nonzero_count: usize = 0;
            for (input.data) |value| {
                if (isNonZero(T, value)) nonzero_count += 1;
            }

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

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.row_offsets);
            self.allocator.free(self.col_indices);
            self.allocator.free(self.values);
            self.* = undefined;
        }

        pub fn nnz(self: Self) usize {
            return self.values.len;
        }

        pub fn asVeyraView(self: Self) SparseError!veyra.CsrView(T) {
            return veyra.CsrView(T).fromSlices(self.rows, self.cols, self.row_offsets, self.col_indices, self.values) catch return error.BackendFailure;
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

        pub fn dropZeros(self: Self) SparseError!Self {
            var nonzero_count: usize = 0;
            for (self.values) |value| {
                if (isNonZero(T, value)) nonzero_count += 1;
            }

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
            const view = try @as(CsrMatrix(f64), self).asVeyraView();
            var rhs = veyra.Vector(f64).fromSlice(self.allocator, x.data) catch return error.BackendFailure;
            defer rhs.deinit();
            var dst = veyra.Vector(f64).zeros(self.allocator, self.rows) catch return error.BackendFailure;
            defer dst.deinit();
            veyra.csrMatvec(f64, view, rhs.asView(), dst.asMut()) catch return error.BackendFailure;
            return array_mod.Array(f64).fromSlice(self.allocator, dst.data, &.{self.rows});
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
            const view = try @as(CsrMatrix(f64), self).asVeyraView();
            var rhs_matrix = veyra.Matrix(f64).fromSlice(self.allocator, rhs.shape[0], rhs.shape[1], .row_major, rhs.data) catch return error.BackendFailure;
            defer rhs_matrix.deinit();
            var dst = veyra.Matrix(f64).zeros(self.allocator, self.rows, rhs.shape[1], .row_major) catch return error.BackendFailure;
            defer dst.deinit();
            veyra.csrMatmat(f64, view, rhs_matrix.asView(), dst.asMut()) catch return error.BackendFailure;
            return array_mod.Array(f64).fromSlice(self.allocator, dst.data, &.{ self.rows, rhs.shape[1] });
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
            const view = try @as(CsrMatrix(f64), self).asVeyraView();
            var rhs = veyra.Vector(f64).fromSlice(self.allocator, x.data) catch return error.BackendFailure;
            defer rhs.deinit();
            var dst = veyra.Vector(f64).zeros(self.allocator, self.cols) catch return error.BackendFailure;
            defer dst.deinit();
            veyra.csrTransposeMatvec(f64, view, rhs.asView(), dst.asMut()) catch return error.BackendFailure;
            return array_mod.Array(f64).fromSlice(self.allocator, dst.data, &.{self.cols});
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
            const view = try @as(CsrMatrix(f64), self).asVeyraView();
            var rhs_matrix = veyra.Matrix(f64).fromSlice(self.allocator, rhs.shape[0], rhs.shape[1], .row_major, rhs.data) catch return error.BackendFailure;
            defer rhs_matrix.deinit();
            var dst = veyra.Matrix(f64).zeros(self.allocator, self.cols, rhs.shape[1], .row_major) catch return error.BackendFailure;
            defer dst.deinit();
            veyra.csrTransposeMatmat(f64, view, rhs_matrix.asView(), dst.asMut()) catch return error.BackendFailure;
            return array_mod.Array(f64).fromSlice(self.allocator, dst.data, &.{ self.cols, rhs.shape[1] });
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

        pub fn minValue(self: Self) SparseError!T {
            return minStoredValue(T, self.values);
        }

        pub fn maxValue(self: Self) SparseError!T {
            return maxStoredValue(T, self.values);
        }

        pub fn minAbsValue(self: Self) SparseError!T {
            return minStoredAbsValue(T, self.values);
        }

        pub fn maxAbsValue(self: Self) SparseError!T {
            return maxStoredAbsValue(T, self.values);
        }

        pub fn mean(self: Self) SparseError!f64 {
            ensureNumeric(T);
            const count = try sparseElementCount(self.rows, self.cols);
            if (count == 0) return error.EmptyArray;
            var total: f64 = 0;
            for (self.values) |value| total += sparseValueToF64(T, value);
            return total / sparseSizeToF64(count);
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

        pub fn stddev(self: Self, correction: f64) SparseError!f64 {
            return @sqrt(try self.variance(correction));
        }

        pub fn sampleVariance(self: Self) SparseError!f64 {
            return self.variance(1);
        }

        pub fn sampleStddev(self: Self) SparseError!f64 {
            return self.stddev(1);
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
                const view = @as(CsrMatrix(f64), self).asVeyraView() catch return 0;
                return @as(T, @floatCast(veyra.csrFrobeniusNorm(f64, view)));
            }
            var total = zero(T);
            for (self.values) |value| total += value * value;
            return @sqrt(total);
        }

        pub fn density(self: Self) SparseError!f64 {
            const total = self.rows * self.cols;
            if (total == 0) return 0;
            if (T == f64) {
                const view = try @as(CsrMatrix(f64), self).asVeyraView();
                return veyra.csrDensity(f64, view) catch return error.BackendFailure;
            }
            return @as(f64, @floatFromInt(self.values.len)) / @as(f64, @floatFromInt(total));
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
            const view = try @as(CsrMatrix(f64), self).asVeyraView();
            var out = veyra.Vector(f64).zeros(self.allocator, self.rows) catch return error.BackendFailure;
            defer out.deinit();
            veyra.csrRowSums(f64, view, out.asMut()) catch return error.BackendFailure;
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
            const view = try @as(CsrMatrix(f64), self).asVeyraView();
            var out = veyra.Vector(f64).zeros(self.allocator, self.cols) catch return error.BackendFailure;
            defer out.deinit();
            veyra.csrColumnSumsWithWorkspace(f64, view, out.asMut()) catch return error.BackendFailure;
            return array_mod.Array(f64).fromSlice(self.allocator, out.data, &.{self.cols});
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
            const view = try @as(CsrMatrix(f64), self).asVeyraView();
            var out = veyra.Vector(f64).zeros(self.allocator, self.rows) catch return error.BackendFailure;
            defer out.deinit();
            veyra.csrRowAbsSums(f64, view, out.asMut()) catch return error.BackendFailure;
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
            const view = try @as(CsrMatrix(f64), self).asVeyraView();
            var out = veyra.Vector(f64).zeros(self.allocator, self.cols) catch return error.BackendFailure;
            defer out.deinit();
            veyra.csrColumnAbsSumsWithWorkspace(f64, view, out.asMut()) catch return error.BackendFailure;
            return array_mod.Array(f64).fromSlice(self.allocator, out.data, &.{self.cols});
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
            const view = try @as(CsrMatrix(f64), self).asVeyraView();
            var out = veyra.Vector(f64).zeros(self.allocator, self.rows) catch return error.BackendFailure;
            defer out.deinit();
            veyra.csrRowNorms(f64, view, out.asMut()) catch return error.BackendFailure;
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
            const view = try @as(CsrMatrix(f64), self).asVeyraView();
            var out = veyra.Vector(f64).zeros(self.allocator, self.cols) catch return error.BackendFailure;
            defer out.deinit();
            veyra.csrColumnNormsWithWorkspace(f64, view, out.asMut()) catch return error.BackendFailure;
            return array_mod.Array(f64).fromSlice(self.allocator, out.data, &.{self.cols});
        }

        pub fn diagonal(self: Self) SparseError!array_mod.Array(T) {
            if (self.rows != self.cols) return error.NonMatrixArray;
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.rows});
            errdefer out.deinit();
            for (0..self.rows) |r| out.data[r] = self.get(r, r) orelse zero(T);
            return out;
        }

        pub fn trace(self: Self) SparseError!T {
            ensureNumeric(T);
            if (self.rows != self.cols) return error.NonMatrixArray;
            var total = zero(T);
            for (0..self.rows) |r| total = addSparseValue(T, total, self.get(r, r) orelse zero(T));
            return total;
        }

        pub fn missingDiagonalCount(self: Self) SparseError!usize {
            if (self.rows != self.cols) return error.NonMatrixArray;
            var count: usize = 0;
            for (0..self.rows) |r| {
                if (!self.hasEntry(r, r)) count += 1;
            }
            return count;
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

        pub fn solveTriangular(self: Self, rhs: array_mod.Array(T), triangle: Triangle, diag_kind: Diagonal) SparseError!array_mod.Array(T) {
            if (self.rows != self.cols) return error.NonMatrixArray;
            if (rhs.shape.len != 1 and rhs.shape.len != 2) return error.InvalidShape;
            if (rhs.shape[0] != self.rows) return error.ShapeMismatch;
            if (comptime T == f64) return self.solveTriangularF64(@as(array_mod.Array(f64), rhs), triangle, diag_kind);
            return self.solveTriangularReference(rhs, triangle, diag_kind);
        }

        fn solveTriangularF64(self: Self, rhs: array_mod.Array(f64), triangle: Triangle, diag_kind: Diagonal) SparseError!array_mod.Array(f64) {
            const view = try @as(CsrMatrix(f64), self).asVeyraView();
            if (rhs.shape.len == 1) {
                var rhs_vec = veyra.Vector(f64).fromSlice(self.allocator, rhs.data) catch return error.BackendFailure;
                defer rhs_vec.deinit();
                var dst = veyra.Vector(f64).zeros(self.allocator, self.rows) catch return error.BackendFailure;
                defer dst.deinit();
                veyra.csrSolveTriangular(f64, view, rhs_vec.asView(), dst.asMut(), toVeyraTriangle(triangle), toVeyraDiagonal(diag_kind)) catch return error.BackendFailure;
                return array_mod.Array(f64).fromSlice(self.allocator, dst.data, &.{self.rows});
            }
            var rhs_mat = veyra.Matrix(f64).fromSlice(self.allocator, rhs.shape[0], rhs.shape[1], .row_major, rhs.data) catch return error.BackendFailure;
            defer rhs_mat.deinit();
            var dst = veyra.Matrix(f64).zeros(self.allocator, self.rows, rhs.shape[1], .row_major) catch return error.BackendFailure;
            defer dst.deinit();
            veyra.csrSolveTriangularMatrix(f64, view, rhs_mat.asView(), dst.asMut(), toVeyraTriangle(triangle), toVeyraDiagonal(diag_kind)) catch return error.BackendFailure;
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
            var nonzero_count: usize = 0;
            for (input.data) |value| {
                if (isNonZero(T, value)) nonzero_count += 1;
            }
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

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.col_offsets);
            self.allocator.free(self.row_indices);
            self.allocator.free(self.values);
            self.* = undefined;
        }

        pub fn nnz(self: Self) usize {
            return self.values.len;
        }

        pub fn asVeyraView(self: Self) SparseError!veyra.CscView(T) {
            return veyra.CscView(T).fromSlices(self.rows, self.cols, self.col_offsets, self.row_indices, self.values) catch return error.BackendFailure;
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

        pub fn dropZeros(self: Self) SparseError!Self {
            var nonzero_count: usize = 0;
            for (self.values) |value| {
                if (isNonZero(T, value)) nonzero_count += 1;
            }

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
            const view = try @as(CscMatrix(f64), self).asVeyraView();
            var rhs = veyra.Vector(f64).fromSlice(self.allocator, x.data) catch return error.BackendFailure;
            defer rhs.deinit();
            var dst = veyra.Vector(f64).zeros(self.allocator, self.rows) catch return error.BackendFailure;
            defer dst.deinit();
            veyra.cscMatvec(f64, view, rhs.asView(), dst.asMut()) catch return error.BackendFailure;
            return array_mod.Array(f64).fromSlice(self.allocator, dst.data, &.{self.rows});
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
            const view = try @as(CscMatrix(f64), self).asVeyraView();
            var rhs_matrix = veyra.Matrix(f64).fromSlice(self.allocator, rhs.shape[0], rhs.shape[1], .row_major, rhs.data) catch return error.BackendFailure;
            defer rhs_matrix.deinit();
            var dst = veyra.Matrix(f64).zeros(self.allocator, self.rows, rhs.shape[1], .row_major) catch return error.BackendFailure;
            defer dst.deinit();
            veyra.cscMatmat(f64, view, rhs_matrix.asView(), dst.asMut()) catch return error.BackendFailure;
            return array_mod.Array(f64).fromSlice(self.allocator, dst.data, &.{ self.rows, rhs.shape[1] });
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
            const view = try @as(CscMatrix(f64), self).asVeyraView();
            var rhs = veyra.Vector(f64).fromSlice(self.allocator, x.data) catch return error.BackendFailure;
            defer rhs.deinit();
            var dst = veyra.Vector(f64).zeros(self.allocator, self.cols) catch return error.BackendFailure;
            defer dst.deinit();
            veyra.cscTransposeMatvec(f64, view, rhs.asView(), dst.asMut()) catch return error.BackendFailure;
            return array_mod.Array(f64).fromSlice(self.allocator, dst.data, &.{self.cols});
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
            const view = try @as(CscMatrix(f64), self).asVeyraView();
            var rhs_matrix = veyra.Matrix(f64).fromSlice(self.allocator, rhs.shape[0], rhs.shape[1], .row_major, rhs.data) catch return error.BackendFailure;
            defer rhs_matrix.deinit();
            var dst = veyra.Matrix(f64).zeros(self.allocator, self.cols, rhs.shape[1], .row_major) catch return error.BackendFailure;
            defer dst.deinit();
            veyra.cscTransposeMatmat(f64, view, rhs_matrix.asView(), dst.asMut()) catch return error.BackendFailure;
            return array_mod.Array(f64).fromSlice(self.allocator, dst.data, &.{ self.cols, rhs.shape[1] });
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

        pub fn minValue(self: Self) SparseError!T {
            return minStoredValue(T, self.values);
        }

        pub fn maxValue(self: Self) SparseError!T {
            return maxStoredValue(T, self.values);
        }

        pub fn minAbsValue(self: Self) SparseError!T {
            return minStoredAbsValue(T, self.values);
        }

        pub fn maxAbsValue(self: Self) SparseError!T {
            return maxStoredAbsValue(T, self.values);
        }

        pub fn mean(self: Self) SparseError!f64 {
            ensureNumeric(T);
            const count = try sparseElementCount(self.rows, self.cols);
            if (count == 0) return error.EmptyArray;
            var total: f64 = 0;
            for (self.values) |value| total += sparseValueToF64(T, value);
            return total / sparseSizeToF64(count);
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

        pub fn stddev(self: Self, correction: f64) SparseError!f64 {
            return @sqrt(try self.variance(correction));
        }

        pub fn sampleVariance(self: Self) SparseError!f64 {
            return self.variance(1);
        }

        pub fn sampleStddev(self: Self) SparseError!f64 {
            return self.stddev(1);
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
                const view = @as(CscMatrix(f64), self).asVeyraView() catch return 0;
                return @as(T, @floatCast(veyra.cscFrobeniusNorm(f64, view)));
            }
            var total = zero(T);
            for (self.values) |value| total += value * value;
            return @sqrt(total);
        }

        pub fn density(self: Self) SparseError!f64 {
            const total = self.rows * self.cols;
            if (total == 0) return 0;
            if (comptime T == f64) {
                const view = try @as(CscMatrix(f64), self).asVeyraView();
                return veyra.cscDensity(f64, view) catch return error.BackendFailure;
            }
            return @as(f64, @floatFromInt(self.values.len)) / @as(f64, @floatFromInt(total));
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
            const view = try @as(CscMatrix(f64), self).asVeyraView();
            var out = veyra.Vector(f64).zeros(self.allocator, self.cols) catch return error.BackendFailure;
            defer out.deinit();
            veyra.cscColumnSums(f64, view, out.asMut()) catch return error.BackendFailure;
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
            const view = try @as(CscMatrix(f64), self).asVeyraView();
            var out = veyra.Vector(f64).zeros(self.allocator, self.rows) catch return error.BackendFailure;
            defer out.deinit();
            veyra.cscRowSumsWithWorkspace(f64, view, out.asMut()) catch return error.BackendFailure;
            return array_mod.Array(f64).fromSlice(self.allocator, out.data, &.{self.rows});
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
            const view = try @as(CscMatrix(f64), self).asVeyraView();
            var out = veyra.Vector(f64).zeros(self.allocator, self.cols) catch return error.BackendFailure;
            defer out.deinit();
            veyra.cscColumnAbsSums(f64, view, out.asMut()) catch return error.BackendFailure;
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
            const view = try @as(CscMatrix(f64), self).asVeyraView();
            var out = veyra.Vector(f64).zeros(self.allocator, self.rows) catch return error.BackendFailure;
            defer out.deinit();
            veyra.cscRowAbsSumsWithWorkspace(f64, view, out.asMut()) catch return error.BackendFailure;
            return array_mod.Array(f64).fromSlice(self.allocator, out.data, &.{self.rows});
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
            const view = try @as(CscMatrix(f64), self).asVeyraView();
            var out = veyra.Vector(f64).zeros(self.allocator, self.cols) catch return error.BackendFailure;
            defer out.deinit();
            veyra.cscColumnNorms(f64, view, out.asMut()) catch return error.BackendFailure;
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
            const view = try @as(CscMatrix(f64), self).asVeyraView();
            var out = veyra.Vector(f64).zeros(self.allocator, self.rows) catch return error.BackendFailure;
            defer out.deinit();
            veyra.cscRowNormsWithWorkspace(f64, view, out.asMut()) catch return error.BackendFailure;
            return array_mod.Array(f64).fromSlice(self.allocator, out.data, &.{self.rows});
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

        pub fn diagonal(self: Self) SparseError!array_mod.Array(T) {
            if (self.rows != self.cols) return error.NonMatrixArray;
            var out = try array_mod.Array(T).zeros(self.allocator, &.{self.rows});
            errdefer out.deinit();
            for (0..self.rows) |i| out.data[i] = self.get(i, i) orelse zero(T);
            return out;
        }

        pub fn trace(self: Self) SparseError!T {
            ensureNumeric(T);
            if (self.rows != self.cols) return error.NonMatrixArray;
            var total = zero(T);
            for (0..self.rows) |i| total = addSparseValue(T, total, self.get(i, i) orelse zero(T));
            return total;
        }

        pub fn missingDiagonalCount(self: Self) SparseError!usize {
            if (self.rows != self.cols) return error.NonMatrixArray;
            var count: usize = 0;
            for (0..self.rows) |i| {
                if (!self.hasEntry(i, i)) count += 1;
            }
            return count;
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
            const view = try @as(CscMatrix(f64), self).asVeyraView();
            if (rhs.shape.len == 1) {
                var rhs_vec = veyra.Vector(f64).fromSlice(self.allocator, rhs.data) catch return error.BackendFailure;
                defer rhs_vec.deinit();
                var dst = veyra.Vector(f64).zeros(self.allocator, self.rows) catch return error.BackendFailure;
                defer dst.deinit();
                veyra.cscSolveTriangular(f64, view, rhs_vec.asView(), dst.asMut(), toVeyraTriangle(triangle), toVeyraDiagonal(diag_kind)) catch return error.BackendFailure;
                return array_mod.Array(f64).fromSlice(self.allocator, dst.data, &.{self.rows});
            }
            var rhs_mat = veyra.Matrix(f64).fromSlice(self.allocator, rhs.shape[0], rhs.shape[1], .row_major, rhs.data) catch return error.BackendFailure;
            defer rhs_mat.deinit();
            var dst = veyra.Matrix(f64).zeros(self.allocator, self.rows, rhs.shape[1], .row_major) catch return error.BackendFailure;
            defer dst.deinit();
            veyra.cscSolveTriangularMatrix(f64, view, rhs_mat.asView(), dst.asMut(), toVeyraTriangle(triangle), toVeyraDiagonal(diag_kind)) catch return error.BackendFailure;
            return array_mod.Array(f64).fromSlice(self.allocator, dst.data, &.{ self.rows, rhs.shape[1] });
        }
    };
}

pub fn cscFromDense(comptime T: type, input: array_mod.Array(T)) SparseError!CscMatrix(T) {
    return CscMatrix(T).fromDense(input);
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

pub fn cooFromDense(comptime T: type, input: array_mod.Array(T)) SparseError!CooMatrix(T) {
    return CooMatrix(T).fromDense(input);
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

pub fn csrFromDense(comptime T: type, input: array_mod.Array(T)) SparseError!CsrMatrix(T) {
    return CsrMatrix(T).fromDense(input);
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

    var lower_csr = try csrFromDiagonal(f64, gpa, &.{ 4, 5 }, -1);
    defer lower_csr.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 0, 1, 2 }, lower_csr.row_offsets);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, lower_csr.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 4, 5 }, lower_csr.values);
    var lower_csc = try cscFromDiagonal(f64, gpa, &.{ 4, 5 }, -1);
    defer lower_csc.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2, 2 }, lower_csc.col_offsets);
    try std.testing.expectEqualSlices(usize, &.{ 1, 2 }, lower_csc.row_indices);
    try std.testing.expectEqualSlices(f64, &.{ 4, 5 }, lower_csc.values);
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
    try std.testing.expectEqual(@as(usize, 6), coo.nnz());
    try std.testing.expectEqualSlices(usize, &.{ 0, 0, 1, 1, 2, 2 }, coo.row_indices);
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 1, 3, 0, 3 }, coo.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 10, 2, 3, 4, 5, 6 }, coo.values);
    try std.testing.expectApproxEqAbs(@as(f64, 30), coo.sum(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 30), coo.absSum(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, @sqrt(190.0)), coo.frobeniusNorm(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 15), try coo.oneNorm(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 12), try coo.infNorm(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), try coo.density(), 1e-12);

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

    var csr = try coo.toCsr();
    defer csr.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 4, 6 }, csr.row_offsets);
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 1, 3, 0, 3 }, csr.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 10, 2, 3, 4, 5, 6 }, csr.values);

    var csc = try coo.toCsc();
    defer csc.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 3, 4, 6 }, csc.col_offsets);
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 1, 0, 1, 2 }, csc.row_indices);
    try std.testing.expectEqualSlices(f64, &.{ 10, 5, 3, 2, 4, 6 }, csc.values);

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
    try std.testing.expectApproxEqAbs(@as(f64, 5), try coo.maxValue(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), try coo.minAbsValue(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5), try coo.maxAbsValue(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 11.0 / 9.0), try coo.mean(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 55.0 / 9.0 - (11.0 / 9.0) * (11.0 / 9.0)), try coo.variance(0), 1e-12);
    try std.testing.expectApproxEqAbs(@sqrt(55.0 / 9.0 - (11.0 / 9.0) * (11.0 / 9.0)), try coo.stddev(0), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, (55.0 - (11.0 * 11.0) / 9.0) / 8.0), try coo.sampleVariance(), 1e-12);

    var row_vars = try coo.rowVariances(0);
    defer row_vars.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 14.0 / 9.0), row_vars.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2), row_vars.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 14.0 / 3.0), row_vars.data[2], 1e-12);
    var row_sample_vars = try coo.rowSampleVariances();
    defer row_sample_vars.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 7.0 / 3.0), row_sample_vars.data[0], 1e-12);
    var col_vars = try coo.columnVariances(0);
    defer col_vars.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 26.0 / 9.0), col_vars.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2), col_vars.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 26.0 / 3.0), col_vars.data[2], 1e-12);

    var row_means = try coo.rowMeans();
    defer row_means.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, -1.0 / 3.0), row_means.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), row_means.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3), row_means.data[2], 1e-12);
    var col_means = try coo.columnMeans();
    defer col_means.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 3.0), col_means.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), col_means.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), col_means.data[2], 1e-12);

    var row_nnz = try coo.rowNnz();
    defer row_nnz.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 1, 2 }, row_nnz.data);
    var col_nnz = try coo.columnNnz();
    defer col_nnz.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 1, 2 }, col_nnz.data);

    var row_sums = try coo.rowSums();
    defer row_sums.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -1, 3, 9 }, row_sums.data);
    var col_sums = try coo.columnSums();
    defer col_sums.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 5, 3, 3 }, col_sums.data);

    var row_abs = try coo.rowAbsSums();
    defer row_abs.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 3, 3, 9 }, row_abs.data);
    var col_abs = try coo.columnAbsSums();
    defer col_abs.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 5, 3, 7 }, col_abs.data);

    var row_norms = try coo.rowNorms();
    defer row_norms.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, @sqrt(5.0)), row_norms.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3), row_norms.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, @sqrt(41.0)), row_norms.data[2], 1e-12);
    var col_norms = try coo.columnNorms();
    defer col_norms.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, @sqrt(17.0)), col_norms.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3), col_norms.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, @sqrt(29.0)), col_norms.data[2], 1e-12);
}

test "sparse addition canonicalizes duplicate coordinates" {
    const gpa = std.testing.allocator;
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
    var coo_scaled = try coo_pruned.scale(2);
    defer coo_scaled.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 10, 18 }, coo_scaled.values);
    var coo_row_scaled = try coo_pruned.scaleRows(&.{ 2, 3 });
    defer coo_row_scaled.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 10, 27 }, coo_row_scaled.values);
    var coo_col_scaled = try coo_pruned.scaleColumns(&.{ 4, 5, 6 });
    defer coo_col_scaled.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 20, 54 }, coo_col_scaled.values);
    var coo_neg = try coo_pruned.neg();
    defer coo_neg.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -5, -9 }, coo_neg.values);
    var coo_diff = try lhs.sub(rhs);
    defer coo_diff.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 1 }, coo_diff.row_indices);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2 }, coo_diff.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ -3, 4, -3 }, coo_diff.values);
    var coo_product = try lhs.hadamard(rhs);
    defer coo_product.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 1 }, coo_product.row_indices);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2 }, coo_product.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 4, -4, 18 }, coo_product.values);

    var lhs_csr = try lhs.toCsr();
    defer lhs_csr.deinit();
    var rhs_csr = try rhs.toCsr();
    defer rhs_csr.deinit();
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
    var csr_diff = try lhs_csr.sub(rhs_csr);
    defer csr_diff.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 3 }, csr_diff.row_offsets);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2 }, csr_diff.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ -3, 4, -3 }, csr_diff.values);
    var csr_product = try lhs_csr.multiply(rhs_csr);
    defer csr_product.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 3 }, csr_product.row_offsets);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2 }, csr_product.col_indices);
    try std.testing.expectEqualSlices(f64, &.{ 4, -4, 18 }, csr_product.values);

    var lhs_csc = try lhs.toCsc();
    defer lhs_csc.deinit();
    var rhs_csc = try rhs.toCsc();
    defer rhs_csc.deinit();
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
    try std.testing.expectError(error.ShapeMismatch, csc_pruned.scaleColumns(&.{1}));
    var csc_diff = try lhs_csc.sub(rhs_csc);
    defer csc_diff.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2, 3 }, csc_diff.col_offsets);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 1 }, csc_diff.row_indices);
    try std.testing.expectEqualSlices(f64, &.{ -3, 4, -3 }, csc_diff.values);
    var csc_product = try lhs_csc.mul(rhs_csc);
    defer csc_product.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2, 3 }, csc_product.col_offsets);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 1 }, csc_product.row_indices);
    try std.testing.expectEqualSlices(f64, &.{ 4, -4, 18 }, csc_product.values);

    var mismatched = try cooFromSlices(f64, gpa, 3, 3, &.{0}, &.{0}, &.{1});
    defer mismatched.deinit();
    try std.testing.expectError(error.ShapeMismatch, lhs.add(mismatched));
    try std.testing.expectError(error.ShapeMismatch, lhs.sub(mismatched));
    try std.testing.expectError(error.ShapeMismatch, lhs.hadamard(mismatched));
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
    try std.testing.expectApproxEqAbs(@as(f64, 15), try symmetric.trace(), 1e-12);
    try std.testing.expectEqual(@as(usize, 0), try symmetric.missingDiagonalCount());
    try std.testing.expectEqual(@as(usize, 0), try symmetric.zeroDiagonalCount());
    try std.testing.expectEqual(@as(usize, 1), try symmetric.bandwidth());
    try std.testing.expect(try symmetric.structurallySymmetric());
    try std.testing.expect(try symmetric.numericallySymmetric(1e-12));
    try std.testing.expectApproxEqAbs(@as(f64, 2), symmetric.get(1, 2).?, 1e-12);
    try std.testing.expect(symmetric.get(0, 2) == null);

    var nonsym_dense = try array_mod.Array(f64).fromSlice(gpa, &.{
        1, 2, 0,
        0, 0, 3,
        0, 0, 4,
    }, &.{ 3, 3 });
    defer nonsym_dense.deinit();
    var nonsym = try cooFromDense(f64, nonsym_dense);
    defer nonsym.deinit();
    try std.testing.expectEqual(@as(usize, 1), try nonsym.missingDiagonalCount());
    try std.testing.expectEqual(@as(usize, 0), try nonsym.zeroDiagonalCount());
    try std.testing.expectEqual(@as(usize, 1), try nonsym.bandwidth());
    try std.testing.expect(!(try nonsym.structurallySymmetric()));
    try std.testing.expect(!(try nonsym.numericallySymmetric(1e-12)));

    var duplicate_diagonal = try cooFromSlices(f64, gpa, 2, 2, &.{ 0, 0, 1, 1, 1 }, &.{ 0, 0, 0, 1, 1 }, &.{ 1, 2, 3, 4, -4 });
    defer duplicate_diagonal.deinit();
    var duplicate_diag = try duplicate_diagonal.diagonal();
    defer duplicate_diag.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 3, 0 }, duplicate_diag.data);
    try std.testing.expectApproxEqAbs(@as(f64, 3), try duplicate_diagonal.trace(), 1e-12);
    try std.testing.expectEqual(@as(usize, 0), try duplicate_diagonal.missingDiagonalCount());
    try std.testing.expectEqual(@as(usize, 1), try duplicate_diagonal.zeroDiagonalCount());
    try std.testing.expectApproxEqAbs(@as(f64, 3), duplicate_diagonal.get(0, 0).?, 1e-12);
    try std.testing.expect(!(try duplicate_diagonal.structurallySymmetric()));

    var duplicate_symmetric = try cooFromSlices(f64, gpa, 2, 2, &.{ 0, 0, 1, 1 }, &.{ 1, 1, 0, 0 }, &.{ 1, 2, 1.5, 1.5 });
    defer duplicate_symmetric.deinit();
    try std.testing.expectEqual(@as(usize, 2), try duplicate_symmetric.missingDiagonalCount());
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

    var dense2 = try csr.toDense();
    defer dense2.deinit();
    try std.testing.expectEqualSlices(f64, dense.data, dense2.data);

    var csc = try csr.toCsc();
    defer csc.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 3, 4, 6 }, csc.col_offsets);
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 1, 0, 1, 2 }, csc.row_indices);
    try std.testing.expectEqualSlices(f64, &.{ 10, 5, 3, 2, 4, 6 }, csc.values);
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

    try std.testing.expectApproxEqAbs(@as(f64, 6), csr.sum(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 6), csr.absSum(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, @sqrt(14.0)), csr.frobeniusNorm(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3), try csr.oneNorm(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3), try csr.infNorm(), 1e-12);
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
    try std.testing.expectApproxEqAbs(@as(f64, 5), try csr.maxValue(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), try csr.minAbsValue(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5), try csr.maxAbsValue(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 11.0 / 9.0), try csr.mean(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 55.0 / 9.0 - (11.0 / 9.0) * (11.0 / 9.0)), try csr.variance(0), 1e-12);
    try std.testing.expectApproxEqAbs(@sqrt(55.0 / 9.0 - (11.0 / 9.0) * (11.0 / 9.0)), try csr.stddev(0), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, (55.0 - (11.0 * 11.0) / 9.0) / 8.0), try csr.sampleVariance(), 1e-12);

    var row_vars = try csr.rowVariances(0);
    defer row_vars.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 14.0 / 9.0), row_vars.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2), row_vars.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 14.0 / 3.0), row_vars.data[2], 1e-12);
    var col_vars = try csr.columnVariances(0);
    defer col_vars.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 26.0 / 9.0), col_vars.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2), col_vars.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 26.0 / 3.0), col_vars.data[2], 1e-12);
    var row_stds = try csr.rowStddevs(0);
    defer row_stds.deinit();
    try std.testing.expectApproxEqAbs(@sqrt(14.0 / 9.0), row_stds.data[0], 1e-12);
    var row_sample_vars = try csr.rowSampleVariances();
    defer row_sample_vars.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 7.0 / 3.0), row_sample_vars.data[0], 1e-12);

    var row_means = try csr.rowMeans();
    defer row_means.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, -1.0 / 3.0), row_means.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), row_means.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3), row_means.data[2], 1e-12);
    var col_means = try csr.columnMeans();
    defer col_means.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 3.0), col_means.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), col_means.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), col_means.data[2], 1e-12);

    var row_nnz = try csr.rowNnz();
    defer row_nnz.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 1, 2 }, row_nnz.data);
    var col_nnz = try csr.columnNnz();
    defer col_nnz.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 1, 2 }, col_nnz.data);

    var row_sums = try csr.rowSums();
    defer row_sums.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -1, 3, 9 }, row_sums.data);
    var col_sums = try csr.columnSums();
    defer col_sums.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 5, 3, 3 }, col_sums.data);

    var row_abs = try csr.rowAbsSums();
    defer row_abs.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 3, 3, 9 }, row_abs.data);
    var col_abs = try csr.columnAbsSums();
    defer col_abs.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 5, 3, 7 }, col_abs.data);

    var row_norms = try csr.rowNorms();
    defer row_norms.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, @sqrt(5.0)), row_norms.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3), row_norms.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, @sqrt(41.0)), row_norms.data[2], 1e-12);
    var col_norms = try csr.columnNorms();
    defer col_norms.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, @sqrt(17.0)), col_norms.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3), col_norms.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, @sqrt(29.0)), col_norms.data[2], 1e-12);
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
    try std.testing.expectEqual(@as(usize, 0), try symmetric.missingDiagonalCount());
    try std.testing.expectEqual(@as(usize, 0), try symmetric.zeroDiagonalCount());
    try std.testing.expectEqual(@as(usize, 1), try symmetric.bandwidth());
    try std.testing.expect(try symmetric.structurallySymmetric());
    try std.testing.expect(try symmetric.numericallySymmetric(1e-12));

    var nonsym_dense = try array_mod.Array(f64).fromSlice(gpa, &.{
        1, 2, 0,
        0, 0, 3,
        0, 0, 4,
    }, &.{ 3, 3 });
    defer nonsym_dense.deinit();
    var nonsym = try csrFromDense(f64, nonsym_dense);
    defer nonsym.deinit();
    try std.testing.expectEqual(@as(usize, 1), try nonsym.missingDiagonalCount());
    try std.testing.expectEqual(@as(usize, 0), try nonsym.zeroDiagonalCount());
    try std.testing.expectEqual(@as(usize, 1), try nonsym.bandwidth());
    try std.testing.expect(!(try nonsym.structurallySymmetric()));
    try std.testing.expect(!(try nonsym.numericallySymmetric(1e-12)));

    var duplicate = try csrFromCompressed(f64, gpa, 2, 2, &.{ 0, 3, 5 }, &.{ 0, 0, 1, 0, 1 }, &.{ 1.0, -1.0, 2.0, 2.0, 0.0 });
    defer duplicate.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0), duplicate.get(0, 0).?, 1e-12);
    var duplicate_diag = try duplicate.diagonal();
    defer duplicate_diag.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 0 }, duplicate_diag.data);
    try std.testing.expectApproxEqAbs(@as(f64, 0), try duplicate.trace(), 1e-12);
    try std.testing.expectEqual(@as(usize, 0), try duplicate.missingDiagonalCount());
    try std.testing.expectEqual(@as(usize, 2), try duplicate.zeroDiagonalCount());
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
    try std.testing.expectApproxEqAbs(@as(f64, @sqrt(190.0)), csc.frobeniusNorm(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 15), try csc.oneNorm(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 12), try csc.infNorm(), 1e-12);
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
    try std.testing.expectApproxEqAbs(@as(f64, 5), try csc.maxValue(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), try csc.minAbsValue(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5), try csc.maxAbsValue(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 11.0 / 9.0), try csc.mean(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 55.0 / 9.0 - (11.0 / 9.0) * (11.0 / 9.0)), try csc.variance(0), 1e-12);
    try std.testing.expectApproxEqAbs(@sqrt(55.0 / 9.0 - (11.0 / 9.0) * (11.0 / 9.0)), try csc.stddev(0), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, (55.0 - (11.0 * 11.0) / 9.0) / 8.0), try csc.sampleVariance(), 1e-12);

    var row_vars = try csc.rowVariances(0);
    defer row_vars.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 14.0 / 9.0), row_vars.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2), row_vars.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 14.0 / 3.0), row_vars.data[2], 1e-12);
    var col_vars = try csc.columnVariances(0);
    defer col_vars.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 26.0 / 9.0), col_vars.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2), col_vars.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 26.0 / 3.0), col_vars.data[2], 1e-12);
    var col_stds = try csc.columnStddevs(0);
    defer col_stds.deinit();
    try std.testing.expectApproxEqAbs(@sqrt(26.0 / 9.0), col_stds.data[0], 1e-12);
    var row_sample_vars = try csc.rowSampleVariances();
    defer row_sample_vars.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 7.0 / 3.0), row_sample_vars.data[0], 1e-12);

    var row_means = try csc.rowMeans();
    defer row_means.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, -1.0 / 3.0), row_means.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), row_means.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3), row_means.data[2], 1e-12);
    var col_means = try csc.columnMeans();
    defer col_means.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 3.0), col_means.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), col_means.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), col_means.data[2], 1e-12);

    var row_nnz = try csc.rowNnz();
    defer row_nnz.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 1, 2 }, row_nnz.data);
    var col_nnz = try csc.columnNnz();
    defer col_nnz.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 1, 2 }, col_nnz.data);
    var row_sums = try csc.rowSums();
    defer row_sums.deinit();
    try std.testing.expectEqualSlices(f64, &.{ -1, 3, 9 }, row_sums.data);
    var col_sums = try csc.columnSums();
    defer col_sums.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 5, 3, 3 }, col_sums.data);
    var row_abs = try csc.rowAbsSums();
    defer row_abs.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 3, 3, 9 }, row_abs.data);
    var col_abs = try csc.columnAbsSums();
    defer col_abs.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 5, 3, 7 }, col_abs.data);
    var row_norms = try csc.rowNorms();
    defer row_norms.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, @sqrt(5.0)), row_norms.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3), row_norms.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, @sqrt(41.0)), row_norms.data[2], 1e-12);
    var col_norms = try csc.columnNorms();
    defer col_norms.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, @sqrt(17.0)), col_norms.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3), col_norms.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, @sqrt(29.0)), col_norms.data[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 9.0), try csc.density(), 1e-12);
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
    try std.testing.expectEqual(@as(usize, 0), try symmetric.missingDiagonalCount());
    try std.testing.expectEqual(@as(usize, 0), try symmetric.zeroDiagonalCount());
    try std.testing.expectEqual(@as(usize, 1), try symmetric.bandwidth());
    try std.testing.expect(try symmetric.structurallySymmetric());
    try std.testing.expect(try symmetric.numericallySymmetric(1e-12));

    var duplicate = try cscFromCompressed(f64, gpa, 2, 2, &.{ 0, 3, 5 }, &.{ 0, 0, 1, 0, 1 }, &.{ 1.0, -1.0, 2.0, 2.0, 0.0 });
    defer duplicate.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0), duplicate.get(0, 0).?, 1e-12);
    var duplicate_diag = try duplicate.diagonal();
    defer duplicate_diag.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 0, 0 }, duplicate_diag.data);
    try std.testing.expectApproxEqAbs(@as(f64, 0), try duplicate.trace(), 1e-12);
    try std.testing.expectEqual(@as(usize, 0), try duplicate.missingDiagonalCount());
    try std.testing.expectEqual(@as(usize, 2), try duplicate.zeroDiagonalCount());
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
