//! Host-side dataframe fallback used when Boltha-backed device dataframe
//! interop is disabled.
//!
//! Keeping the host-only table implementation here lets `dataframe_no_boltha.zig`
//! focus on feature-unavailable device/lazy stubs while preserving the same
//! public `DataFrame`/`Column` API in non-Boltha builds.

const std = @import("std");
const series_mod = @import("series.zig");

pub const DataError = series_mod.DataError || error{FeatureUnavailable};
pub const DType = enum { f64, i64, bool, string };

pub const Column = union(DType) {
    f64: []const f64,
    i64: []const i64,
    bool: []const bool,
    string: []const []const u8,

    pub fn len(self: Column) usize {
        return switch (self) {
            .f64 => |v| v.len,
            .i64 => |v| v.len,
            .bool => |v| v.len,
            .string => |v| v.len,
        };
    }
};

pub const ColumnDef = struct {
    name: []const u8,
    data: Column,
};

pub const DataFrame = struct {
    allocator: std.mem.Allocator,
    names: [][]const u8,
    columns: []Column,
    rows: usize,

    pub fn init(allocator: std.mem.Allocator, defs: []const ColumnDef) DataError!DataFrame {
        if (defs.len == 0) {
            return .{ .allocator = allocator, .names = &.{}, .columns = &.{}, .rows = 0 };
        }
        const rows = defs[0].data.len();
        for (defs) |def| {
            if (def.data.len() != rows) return error.LengthMismatch;
        }

        var names = try allocator.alloc([]const u8, defs.len);
        errdefer allocator.free(names);
        var columns = try allocator.alloc(Column, defs.len);
        errdefer allocator.free(columns);

        var initialized: usize = 0;
        errdefer {
            for (0..initialized) |i| {
                allocator.free(names[i]);
                freeColumn(allocator, columns[i]);
            }
        }

        for (defs, 0..) |def, i| {
            names[i] = try allocator.dupe(u8, def.name);
            columns[i] = try cloneColumn(allocator, def.data);
            initialized += 1;
        }
        return .{ .allocator = allocator, .names = names, .columns = columns, .rows = rows };
    }

    pub fn deinit(self: *DataFrame) void {
        for (self.names) |name| self.allocator.free(name);
        for (self.columns) |col| freeColumn(self.allocator, col);
        if (self.names.len != 0) self.allocator.free(self.names);
        if (self.columns.len != 0) self.allocator.free(self.columns);
        self.* = undefined;
    }

    pub fn height(self: DataFrame) usize {
        return self.rows;
    }

    pub fn width(self: DataFrame) usize {
        return self.columns.len;
    }

    pub fn shape(self: DataFrame) struct { rows: usize, cols: usize } {
        return .{ .rows = self.rows, .cols = self.columns.len };
    }

    pub fn columnIndex(self: DataFrame, name: []const u8) ?usize {
        for (self.names, 0..) |n, i| {
            if (std.mem.eql(u8, n, name)) return i;
        }
        return null;
    }

    pub fn column(self: DataFrame, name: []const u8) DataError!Column {
        const idx = self.columnIndex(name) orelse return error.ColumnNotFound;
        return self.columns[idx];
    }

    pub fn groupBySum(self: DataFrame, key_name: []const u8, value_name: []const u8) DataError!DataFrame {
        const key_col = try self.column(key_name);
        const value_col = try self.column(value_name);
        if (std.meta.activeTag(key_col) != .string or std.meta.activeTag(value_col) != .f64) return error.TypeMismatch;
        var keys: std.ArrayList([]const u8) = .empty;
        defer keys.deinit(self.allocator);
        var sums: std.ArrayList(f64) = .empty;
        defer sums.deinit(self.allocator);

        for (key_col.string, value_col.f64) |key, value| {
            var found: ?usize = null;
            for (keys.items, 0..) |existing, i| {
                if (std.mem.eql(u8, existing, key)) {
                    found = i;
                    break;
                }
            }
            if (found) |i| {
                sums.items[i] += value;
            } else {
                try keys.append(self.allocator, key);
                try sums.append(self.allocator, value);
            }
        }

        return DataFrame.init(self.allocator, &.{
            .{ .name = key_name, .data = .{ .string = keys.items } },
            .{ .name = value_name, .data = .{ .f64 = sums.items } },
        });
    }

    pub fn print(self: DataFrame, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        try writer.print("DataFrame(shape=({}, {}))\n", .{ self.rows, self.columns.len });
        for (self.names, 0..) |name, i| {
            if (i != 0) try writer.print("\t", .{});
            try writer.print("{s}", .{name});
        }
        try writer.print("\n", .{});
        const limit = @min(self.rows, 12);
        for (0..limit) |r| {
            for (self.columns, 0..) |col, c| {
                if (c != 0) try writer.print("\t", .{});
                try printCell(writer, col, r);
            }
            try writer.print("\n", .{});
        }
    }
};

pub fn dataframe(allocator: std.mem.Allocator, defs: []const ColumnDef) DataError!DataFrame {
    return DataFrame.init(allocator, defs);
}

fn cloneColumn(allocator: std.mem.Allocator, col: Column) DataError!Column {
    return switch (col) {
        .f64 => |v| .{ .f64 = try allocator.dupe(f64, v) },
        .i64 => |v| .{ .i64 = try allocator.dupe(i64, v) },
        .bool => |v| .{ .bool = try allocator.dupe(bool, v) },
        .string => |v| blk: {
            var strings = try allocator.alloc([]const u8, v.len);
            errdefer {
                for (strings) |s| allocator.free(s);
                allocator.free(strings);
            }
            for (v, 0..) |s, i| strings[i] = try allocator.dupe(u8, s);
            break :blk .{ .string = strings };
        },
    };
}

fn freeColumn(allocator: std.mem.Allocator, col: Column) void {
    switch (col) {
        .f64 => |v| allocator.free(v),
        .i64 => |v| allocator.free(v),
        .bool => |v| allocator.free(v),
        .string => |v| {
            for (v) |s| allocator.free(s);
            allocator.free(v);
        },
    }
}

fn printCell(writer: *std.Io.Writer, col: Column, row: usize) std.Io.Writer.Error!void {
    switch (col) {
        .f64 => |v| try writer.print("{d}", .{v[row]}),
        .i64 => |v| try writer.print("{d}", .{v[row]}),
        .bool => |v| try writer.print("{}", .{v[row]}),
        .string => |v| try writer.print("{s}", .{v[row]}),
    }
}
