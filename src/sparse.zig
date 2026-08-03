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

fn ensureFloat(comptime T: type) void {
    if (@typeInfo(T) != .float) @compileError("sparse norm requires floating-point values");
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
