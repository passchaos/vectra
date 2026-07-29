const std = @import("std");
const series_mod = @import("series.zig");
const array_mod = @import("array.zig");
const dataframe_column_mod = @import("dataframe_column.zig");
const host_device_mod = @import("dataframe_host_device.zig");
const host_io_mod = @import("dataframe_host_io.zig");
const numeric_mod = @import("dataframe_numeric.zig");

pub const DataError = series_mod.DataError;
const cloneColumn = dataframe_column_mod.cloneColumn;
const freeColumn = dataframe_column_mod.freeColumn;
const filterColumn = dataframe_column_mod.filterColumn;
const sliceColumn = dataframe_column_mod.sliceColumn;
const takeColumn = dataframe_column_mod.takeColumn;
const describeF64 = numeric_mod.describeF64;
const describeI64 = numeric_mod.describeI64;

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

    pub fn dtype(self: Column) DType {
        return std.meta.activeTag(self);
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

    pub fn series(self: DataFrame, comptime T: type, name: []const u8) DataError!series_mod.Series(T) {
        const col = try self.column(name);
        return switch (col) {
            .f64 => |v| if (T == f64) series_mod.Series(T).init(self.allocator, name, v) else error.TypeMismatch,
            .i64 => |v| if (T == i64) series_mod.Series(T).init(self.allocator, name, v) else error.TypeMismatch,
            .bool => |v| if (T == bool) series_mod.Series(T).init(self.allocator, name, v) else error.TypeMismatch,
            .string => error.TypeMismatch,
        };
    }

    pub fn select(self: DataFrame, names: []const []const u8) DataError!DataFrame {
        var defs = try self.allocator.alloc(ColumnDef, names.len);
        defer self.allocator.free(defs);
        for (names, 0..) |name, i| {
            defs[i] = .{ .name = name, .data = try self.column(name) };
        }
        return DataFrame.init(self.allocator, defs);
    }

    pub fn withColumn(self: DataFrame, name: []const u8, data: Column) DataError!DataFrame {
        if (self.rows != data.len()) return error.LengthMismatch;
        var defs = try self.allocator.alloc(ColumnDef, self.columns.len + 1);
        defer self.allocator.free(defs);
        for (self.names, self.columns, 0..) |n, c, i| defs[i] = .{ .name = n, .data = c };
        defs[self.columns.len] = .{ .name = name, .data = data };
        return DataFrame.init(self.allocator, defs);
    }

    pub fn filter(self: DataFrame, mask: []const bool) DataError!DataFrame {
        if (mask.len != self.rows) return error.LengthMismatch;
        var defs = try self.allocator.alloc(ColumnDef, self.columns.len);
        defer self.allocator.free(defs);
        for (self.names, self.columns, 0..) |name, col, i| {
            defs[i] = .{ .name = name, .data = try filterColumn(self.allocator, col, mask) };
        }
        defer {
            for (defs) |def| freeColumn(self.allocator, def.data);
        }
        return DataFrame.init(self.allocator, defs);
    }

    pub fn head(self: DataFrame, n: usize) DataError!DataFrame {
        return self.sliceRows(0, @min(n, self.rows));
    }

    pub fn tail(self: DataFrame, n: usize) DataError!DataFrame {
        const count = @min(n, self.rows);
        return self.sliceRows(self.rows - count, self.rows);
    }

    pub fn sliceRows(self: DataFrame, start: usize, stop: usize) DataError!DataFrame {
        const end = @min(stop, self.rows);
        const begin = @min(start, end);
        var defs = try self.allocator.alloc(ColumnDef, self.columns.len);
        defer self.allocator.free(defs);
        for (self.names, self.columns, 0..) |name, col, i| defs[i] = .{ .name = name, .data = sliceColumn(col, begin, end) };
        return DataFrame.init(self.allocator, defs);
    }

    pub fn sortBy(self: DataFrame, name: []const u8, descending: bool) DataError!DataFrame {
        const idx = self.columnIndex(name) orelse return error.ColumnNotFound;
        const order = try self.allocator.alloc(usize, self.rows);
        defer self.allocator.free(order);
        for (order, 0..) |*slot, i| slot.* = i;
        const Ctx = struct {
            col: Column,
            desc: bool,
            fn lessThan(ctx: @This(), a: usize, b: usize) bool {
                const less = switch (ctx.col) {
                    .f64 => |v| v[a] < v[b],
                    .i64 => |v| v[a] < v[b],
                    .bool => |v| !v[a] and v[b],
                    .string => |v| std.mem.lessThan(u8, v[a], v[b]),
                };
                return if (ctx.desc) !less else less;
            }
        };
        std.sort.insertion(usize, order, Ctx{ .col = self.columns[idx], .desc = descending }, Ctx.lessThan);
        return self.take(order);
    }

    pub fn take(self: DataFrame, row_indices: []const usize) DataError!DataFrame {
        var defs = try self.allocator.alloc(ColumnDef, self.columns.len);
        defer self.allocator.free(defs);
        for (self.names, self.columns, 0..) |name, col, i| {
            defs[i] = .{ .name = name, .data = try takeColumn(self.allocator, col, row_indices) };
        }
        defer {
            for (defs) |def| freeColumn(self.allocator, def.data);
        }
        return DataFrame.init(self.allocator, defs);
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

    pub fn describe(self: DataFrame) DataError!DataFrame {
        const stat_names = [_][]const u8{ "count", "mean", "min", "max" };
        var defs_list: std.ArrayList(ColumnDef) = .empty;
        defer defs_list.deinit(self.allocator);
        try defs_list.append(self.allocator, .{ .name = "stat", .data = .{ .string = stat_names[0..] } });
        for (self.names, self.columns) |name, col| {
            switch (col) {
                .f64 => |v| {
                    const vals = try describeF64(self.allocator, v);
                    try defs_list.append(self.allocator, .{ .name = name, .data = .{ .f64 = vals } });
                },
                .i64 => |v| {
                    const vals = try describeI64(self.allocator, v);
                    try defs_list.append(self.allocator, .{ .name = name, .data = .{ .f64 = vals } });
                },
                else => {},
            }
        }
        defer {
            for (defs_list.items[1..]) |def| freeColumn(self.allocator, def.data);
        }
        return DataFrame.init(self.allocator, defs_list.items);
    }

    pub fn toArray(self: DataFrame, comptime T: type, names: []const []const u8) (DataError || array_mod.ArrayError)!array_mod.Array(T) {
        var values = try self.allocator.alloc(T, self.rows * names.len);
        defer self.allocator.free(values);
        for (0..self.rows) |r| {
            for (names, 0..) |name, c| {
                const col = try self.column(name);
                values[r * names.len + c] = switch (col) {
                    .f64 => |v| if (T == f64) v[r] else @as(T, @intFromFloat(v[r])),
                    .i64 => |v| if (T == i64) v[r] else @as(T, @floatFromInt(v[r])),
                    .bool => |v| if (T == bool) v[r] else if (v[r]) 1 else 0,
                    .string => return error.TypeMismatch,
                };
            }
        }
        return array_mod.Array(T).fromSlice(self.allocator, values, &.{ self.rows, names.len });
    }

    pub fn toNDArray(self: DataFrame, comptime T: type, names: []const []const u8) (DataError || array_mod.ArrayError)!array_mod.NDArray(T) {
        return self.toArray(T, names);
    }

    pub fn readCsv(allocator: std.mem.Allocator, text: []const u8, has_header: bool) DataError!DataFrame {
        return host_io_mod.readCsv(DataFrame, Column, ColumnDef, allocator, text, has_header);
    }

    pub fn writeCsv(self: DataFrame, allocator: std.mem.Allocator) DataError![]u8 {
        return host_io_mod.writeCsv(self, allocator);
    }

    pub fn print(self: DataFrame, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return host_io_mod.print(self, writer);
    }
};

pub fn dataframe(allocator: std.mem.Allocator, defs: []const ColumnDef) DataError!DataFrame {
    return DataFrame.init(allocator, defs);
}

pub fn deviceDataFrameFromDataFrame(
    comptime DeviceDataFrame: type,
    comptime DeviceColumnDef: type,
    comptime DeviceColumn: type,
    allocator: std.mem.Allocator,
    frame: DataFrame,
    device_value: array_mod.Device,
) (DataError || array_mod.ArrayError)!DeviceDataFrame {
    return host_device_mod.deviceDataFrameFromDataFrame(DeviceDataFrame, DeviceColumnDef, DeviceColumn, allocator, frame, device_value);
}

pub fn deviceDataFrameToDataFrame(frame: anytype) (DataError || array_mod.ArrayError)!DataFrame {
    return host_device_mod.deviceDataFrameToDataFrame(DataFrame, ColumnDef, frame);
}
